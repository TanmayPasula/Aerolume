"""Exact signed distance from grid points to polygons, in torch (GPU or CPU).

The solver builds its immersed-boundary masks from this field, and the optimiser
differentiates through it, so it has to be exact, fast and differentiable:

* **Exact**: distance to the nearest *segment* (not vertex), sign from the
  non-zero winding number -- robust for any simple polygon, either orientation.
  Several polygons combine as a union, ``sdf = min_i sdf_i``. Negative inside.
* **Fast**: brute force is points x segments, ~200 M distance evaluations for a
  1024 x 512 grid and a 400-vertex body. Instead points are grouped into spatial
  tiles at two levels (16 x 16 and 4 x 4 for grid-shaped input). For a tile with
  centre ``c`` and radius ``R``, the triangle inequality says only segments with
  ``d(c, s) <= d(c, poly) + 2 R`` can be nearest to any of its points. Coarse
  tiles pick candidates from all segments, fine tiles from their coarse tile's
  candidates, and each point finally looks at ~``k`` segments. Whenever a
  candidate list cannot be proven complete, that tile falls back to the larger
  list (ultimately brute force), so the result is exact for *any* input; only the
  speed depends on the points being spatially coherent. The winding number is
  likewise evaluated at tile centres and re-evaluated per point only in tiles the
  boundary actually passes through.
* **Bounded memory**: every pass is chunked to a byte budget (default 256 MB).
* **Differentiable**: the search runs without autograd; the distance is then
  recomputed from the selected segment's endpoints with autograd on. The gradient
  of a minimum is the gradient of the active element, so this equals the autograd
  gradient of the brute-force expression -- with O(points) instead of
  O(points x segments) saved activations. Gradients flow to polygon vertices that
  require grad and to ``x``/``y``. The sign is piecewise constant (no gradient).

A float64 request runs the *search* in float32 by default (half the memory
traffic on CPU; consumer GPUs have little fp64 throughput at all) and then
re-picks the nearest of the selected segment's neighbours and recomputes the
distance in float64, which restores full float64 accuracy (verified to 4e-16
against a float64 brute force). The sign of points within ~1e-7 of the boundary
may come from the float32 test. Pass ``search_dtype=torch.float64`` for a fully
float64 search.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

__all__ = ["signed_distance", "volume_fraction"]

_COARSE = 4      # coarse tile = _COARSE x _COARSE fine tiles


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def _seg_d2_(px, py, ax, ay, ex, ey, inv):
    """Squared point-segment distance, broadcasting, in place where possible (no grad)."""
    wx = px - ax
    wy = py - ay
    t = torch.addcmul(wx * ex, wy, ey).mul_(inv).clamp_(0.0, 1.0)
    dx = torch.addcmul(wx, t, ex, value=-1.0)
    dy = wy.addcmul_(t, ey, value=-1.0)
    return dx.mul_(dx).addcmul_(dy, dy)


def _seg_d2(px, py, ax, ay, ex, ey):
    """Squared point-segment distance with autograd (no in-place ops)."""
    wx = px - ax
    wy = py - ay
    ee = (ex * ex + ey * ey).clamp_min(torch.finfo(ex.dtype).tiny)
    t = ((wx * ex + wy * ey) / ee).clamp(0.0, 1.0)
    dx = wx - t * ex
    dy = wy - t * ey
    return dx * dx + dy * dy


def _winding_(px, py, ax, ay, by, ex, ey):
    """Non-zero winding number of points about the segments (summed over the last dim).

    Half-open rule on y (a vertex exactly at the point's height counts once), so
    points are classified consistently even on horizontal edges and vertices.
    """
    span = (ay > py) != (by > py)
    cross = torch.addcmul(ex * (py - ay), ey, px - ax, value=-1.0)
    hit = span & ((cross * ey) > 0)
    sgn = torch.sign(ey).to(torch.int32)
    return (hit.to(torch.int32) * sgn).sum(-1)


class _Segs:
    """Detached segment arrays of one polygon in the search dtype."""

    def __init__(self, v: torch.Tensor) -> None:
        b = torch.roll(v, -1, dims=0)
        self.ax, self.ay = v[:, 0].contiguous(), v[:, 1].contiguous()
        self.bx, self.by = b[:, 0].contiguous(), b[:, 1].contiguous()
        self.ex, self.ey = self.bx - self.ax, self.by - self.ay
        self.inv = 1.0 / (self.ex * self.ex + self.ey * self.ey).clamp_min(torch.finfo(v.dtype).tiny)
        self.n = v.shape[0]

    def take(self, idx: torch.Tensor):
        """Segment arrays gathered at ``idx`` (any shape)."""
        return self.ax[idx], self.ay[idx], self.ex[idx], self.ey[idx], self.inv[idx]


# ---------------------------------------------------------------------------
# search passes (all without autograd, in the search dtype)
# ---------------------------------------------------------------------------

def _centre_pass(cx, cy, sg: _Segs, budget: int, winding: bool = True):
    """Distance and winding number of centres against all segments."""
    n = cx.numel()
    d = torch.empty(n, device=cx.device, dtype=cx.dtype)
    w = torch.zeros(n, device=cx.device, dtype=torch.int32)
    step = max(1, budget // sg.n)
    for s0 in range(0, n, step):
        sl = slice(s0, s0 + step)
        px, py = cx[sl, None], cy[sl, None]
        d[sl] = _seg_d2_(px, py, sg.ax, sg.ay, sg.ex, sg.ey, sg.inv).amin(1).sqrt_()
        if winding:
            w[sl] = _winding_(px, py, sg.ax, sg.ay, sg.by, sg.ex, sg.ey)
    return d, w


def _candidates(cx, cy, R, sg: _Segs, C: torch.Tensor | None, K: int, budget: int):
    """Candidate segments for tiles (centre ``c``, radius ``R``).

    ``C``: per-tile parent candidate lists ``(n, KC)`` (global segment indices), or
    None for "all segments". Returns ``(dmin, cand, valid, need)``: the centre's
    nearest distance, the K nearest candidates (sorted by distance), whether they
    provably contain the nearest segment of every point within R of ``c``, and how
    many of them are needed for that (a prefix of ``cand``). A point p with |p - c| <= R
    has d(p, poly) <= d(c, poly) + R and d(p, s) >= d(c, s) - R, so only segments
    with d(c, s) <= d(c, poly) + 2R can be nearest; if the (K+1)-th smallest d(c, s)
    exceeds that bound, the K smallest contain them all.
    """
    n = cx.numel()
    KC = sg.n if C is None else C.shape[1]
    K = min(K, KC)
    dmin = torch.empty(n, device=cx.device, dtype=cx.dtype)
    cand = torch.empty((n, K), device=cx.device, dtype=torch.long)
    valid = torch.ones(n, device=cx.device, dtype=torch.bool)
    need = torch.full((n,), K, device=cx.device, dtype=torch.long)
    step = max(1, budget // KC)
    for s0 in range(0, n, step):
        sl = slice(s0, s0 + step)
        if C is None:
            d2 = _seg_d2_(cx[sl, None], cy[sl, None], sg.ax, sg.ay, sg.ex, sg.ey, sg.inv)
        else:
            d2 = _seg_d2_(cx[sl, None], cy[sl, None], *sg.take(C[sl]))
        if K < KC:
            vals, idx = torch.topk(d2, K + 1, dim=1, largest=False, sorted=True)
            dm = vals[:, 0].sqrt()
            # relative slack: sqrt(v)^2 can round below v, and a zero-radius tile (padding
            # replicates points) must still count its nearest segment as needed
            bound = ((dm + 2.0 * R[sl]) ** 2 * (1.0 + 1e-5) + torch.finfo(d2.dtype).tiny)[:, None]
            valid[sl] = vals[:, K] > bound[:, 0]
            need[sl] = (vals[:, :K] <= bound).sum(1).clamp_min(1)
            idx = idx[:, :K]
        else:
            vals, idx = d2.min(1, keepdim=True)
            dm = vals[:, 0].sqrt()
            idx = torch.arange(KC, device=cx.device).expand(d2.shape[0], KC)
        dmin[sl] = dm
        cand[sl] = idx if C is None else torch.gather(C[sl], 1, idx)
    return dmin, cand, valid, need


def _point_pass(X, Y, sg: _Segs, C: torch.Tensor | None, budget: int):
    """Nearest squared distance and segment for points ``X, Y`` of shape (n, M)."""
    n, M = X.shape
    d2min = torch.empty((n, M), device=X.device, dtype=X.dtype)
    arg = torch.empty((n, M), device=X.device, dtype=torch.long)
    KC = sg.n if C is None else C.shape[1]
    step = max(1, budget // (M * KC))
    for s0 in range(0, n, step):
        sl = slice(s0, s0 + step)
        px, py = X[sl, :, None], Y[sl, :, None]
        if C is None:
            d2 = _seg_d2_(px, py, sg.ax, sg.ay, sg.ex, sg.ey, sg.inv)
            m, a = d2.min(2)
        else:
            ax, ay, ex, ey, inv = (t[:, None, :] for t in sg.take(C[sl]))
            d2 = _seg_d2_(px, py, ax, ay, ex, ey, inv)
            m, a = d2.min(2)
            a = torch.gather(C[sl], 1, a)
        d2min[sl] = m
        arg[sl] = a
    return d2min, arg


def _point_winding(X, Y, sg: _Segs, budget: int):
    """Winding numbers of points (n, M) against all segments."""
    n, M = X.shape
    px, py = X.reshape(-1), Y.reshape(-1)
    w = torch.empty(px.shape, device=X.device, dtype=torch.int32)
    step = max(1, budget // sg.n)
    for s0 in range(0, px.numel(), step):
        sl = slice(s0, s0 + step)
        w[sl] = _winding_(px[sl, None], py[sl, None], sg.ax, sg.ay, sg.by, sg.ex, sg.ey)
    return w.reshape(n, M)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def _layout(x: torch.Tensor, tile: int) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Group points into ``(Tc, F, M)``: coarse tiles of F fine tiles of M points.

    Grid-shaped input (>= 2 dims) is cut into ``tile x tile`` fine blocks of its last
    two dims, grouped ``_COARSE x _COARSE`` into coarse blocks (padding by edge
    replication); anything else into runs of consecutive points. Correctness does
    not depend on the grouping, only the speed. Returns the grouped tensor and the
    inverse mapping back to ``x``'s shape.
    """
    shape = x.shape
    c = _COARSE
    if x.dim() >= 2 and shape[-1] >= 2 and shape[-2] >= 2:
        H, W = shape[-2], shape[-1]
        X = x.reshape(-1, H, W)
        Bn = X.shape[0]
        blk = tile * c
        Hp, Wp = -(-H // blk) * blk, -(-W // blk) * blk
        if Hp != H:
            X = torch.cat([X, X[:, -1:, :].expand(Bn, Hp - H, W)], 1)
        if Wp != W:
            X = torch.cat([X, X[:, :, -1:].expand(Bn, Hp, Wp - W)], 2)
        hb, wb = Hp // blk, Wp // blk
        # dims: B, hb, c(fine row), tile(row), wb, c(fine col), tile(col)
        G = X.reshape(Bn, hb, c, tile, wb, c, tile).permute(0, 1, 4, 2, 5, 3, 6)
        G = G.reshape(Bn * hb * wb, c * c, tile * tile)

        def untile(v: torch.Tensor) -> torch.Tensor:
            v = v.reshape(Bn, hb, wb, c, c, tile, tile).permute(0, 1, 3, 5, 2, 4, 6).reshape(Bn, Hp, Wp)
            return v[:, :H, :W].reshape(shape)
        return G, untile
    flat = x.reshape(-1)
    n = flat.numel()
    M, F = tile * tile, c * c
    npad = -(-max(n, 1) // (M * F)) * (M * F)
    if npad != n:
        flat = torch.cat([flat, flat[-1:].expand(npad - n)]) if n else flat.new_zeros(npad)

    def untile1(v: torch.Tensor) -> torch.Tensor:
        return v.reshape(-1)[:n].reshape(shape)
    return flat.reshape(-1, F, M), untile1


def _centres(X: torch.Tensor, Y: torch.Tensor, dims: tuple[int, ...]):
    """Bounding-box centres and enclosing radii of point groups over ``dims``."""
    cx = 0.5 * (X.amin(dims) + X.amax(dims))
    cy = 0.5 * (Y.amin(dims) + Y.amax(dims))
    shp = cx.shape + (1,) * len(dims)
    r2 = (X - cx.reshape(shp)) ** 2 + (Y - cy.reshape(shp)) ** 2
    return cx, cy, r2.amax(dims).sqrt()


def _as_tensor(p, device, dtype) -> torch.Tensor:
    if isinstance(p, torch.Tensor):
        return p.to(device=device, dtype=dtype).reshape(-1, 2)
    return torch.as_tensor(np.asarray(p, dtype=np.float64).reshape(-1, 2), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def _polygon_search(G_x, G_y, sg: _Segs, k: int, budget: int):
    """Nearest squared distance, segment index (local) and inside flag per point."""
    Tc, F, M = G_x.shape
    dev = G_x.device
    tiny = torch.finfo(G_x.dtype).eps
    ccx, ccy, Rc = _centres(G_x, G_y, (1, 2))                    # coarse tiles
    Xf, Yf = G_x.reshape(Tc * F, M), G_y.reshape(Tc * F, M)
    fcx, fcy, Rf = _centres(Xf, Yf, (1,))                        # fine tiles
    parent = torch.arange(Tc * F, device=dev) // F

    # coarse: candidates from all segments (wide list), distance + winding at centres
    dC, wC = _centre_pass(ccx, ccy, sg, budget)
    _, candA, validA, _ = _candidates(ccx, ccy, Rc, sg, None, 4 * k, budget)

    # every entry is overwritten below; inf / 0 only make a logic slip visible, never UB
    d2 = torch.full((Tc * F, M), math.inf, device=dev, dtype=G_x.dtype)
    arg = torch.zeros((Tc * F, M), device=dev, dtype=torch.long)
    dF = torch.empty(Tc * F, device=dev, dtype=G_x.dtype)
    # fine tiles under a valid coarse tile choose from its list, others from all
    fine_ok = validA[parent]
    for group, use_parent in ((torch.nonzero(fine_ok).squeeze(1), True),
                              (torch.nonzero(~fine_ok).squeeze(1), False)):
        if group.numel() == 0:
            continue
        C = candA[parent[group]] if use_parent else None
        dmin, candB, validB, need = _candidates(fcx[group], fcy[group], Rf[group], sg, C, k, budget)
        dF[group] = dmin
        # bucket the proven tiles by how many candidates they need: most need far
        # fewer than k, and the per-point pass dominates the run time
        lo = 0
        for kb in sorted({min(b, candB.shape[1]) for b in (8, 16, k)}):
            ok = torch.nonzero(validB & (need > lo) & (need <= kb)).squeeze(1)
            lo = kb
            if ok.numel():
                d2[group[ok]], arg[group[ok]] = _point_pass(Xf[group[ok]], Yf[group[ok]], sg,
                                                            candB[ok, :kb], budget)
        bad = torch.nonzero(~validB).squeeze(1)
        if bad.numel():
            d2[group[bad]], arg[group[bad]] = _point_pass(Xf[group[bad]], Yf[group[bad]], sg,
                                                          C[bad] if C is not None else None, budget)

    # inside: coarse centre's winding unless the boundary may cross the tile
    wF = wC[parent].clone()
    near_c = (dC <= Rc * (1 + 1e-6) + tiny * (1 + dC))[parent]
    nf = torch.nonzero(near_c).squeeze(1)
    if nf.numel():
        _, w = _centre_pass(fcx[nf], fcy[nf], sg, budget)
        wF[nf] = w
    inside = (wF != 0)[:, None].expand(Tc * F, M).clone()
    bt = torch.nonzero(dF <= Rf * (1 + 1e-6) + tiny * (1 + dF)).squeeze(1)
    if bt.numel():
        inside[bt] = _point_winding(Xf[bt], Yf[bt], sg, budget) != 0
    return d2.reshape(Tc, F, M), arg.reshape(Tc, F, M), inside.reshape(Tc, F, M)


def signed_distance(polys: Sequence[torch.Tensor | np.ndarray] | torch.Tensor | np.ndarray,
                    x: torch.Tensor, y: torch.Tensor, *, tile: int = 4, k: int = 32,
                    budget_mb: float = 256.0, search_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Signed distance from points ``(x, y)`` to the union of polygons; < 0 inside.

    ``polys``: one ``(N, 2)`` polygon or a list of them (numpy or torch; tensors may
    require grad). ``x``, ``y``: tensors of equal shape (typically an ``(ny, nx)``
    grid); the result has that shape, dtype and device. ``tile`` / ``k``: fine tile
    edge and per-point candidate count of the pruned search (tuning only).
    ``budget_mb`` bounds the size of the temporaries. ``search_dtype``: precision of
    the nearest-segment search (default float32; the returned distance is always
    computed in the input dtype from the selected segment).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not torch.is_floating_point(x):
        x = x.to(torch.get_default_dtype())
    y = y.to(device=x.device, dtype=x.dtype)
    dev, dt = x.device, x.dtype
    if isinstance(polys, (torch.Tensor, np.ndarray)) and polys.ndim == 2:
        polys = [polys]
    verts = [_as_tensor(p, dev, dt) for p in polys]
    verts = [v for v in verts if v.shape[0] >= 2]
    if not verts:
        return torch.full_like(x, math.inf)
    if x.numel() == 0:
        return torch.empty_like(x)

    sdt = search_dtype or (torch.float32 if dt == torch.float64 else dt)
    esize = torch.finfo(sdt).bits // 8
    budget = max(int(budget_mb * 2**20 / (esize * 8)), 4096)   # elements per temporary (~8 alive)

    with torch.no_grad():
        G_x, untile = _layout(x.detach().to(sdt), tile)
        G_y, _ = _layout(y.detach().to(sdt), tile)
        best = torch.full(G_x.shape, math.inf, device=dev, dtype=sdt)
        best_idx = torch.zeros(G_x.shape, device=dev, dtype=torch.long)
        best_neg = torch.zeros(G_x.shape, device=dev, dtype=torch.bool)
        offset = 0
        for v in verts:
            sg = _Segs(v.detach().to(sdt))
            d2, arg, inside = _polygon_search(G_x, G_y, sg, k, budget)
            d = d2.sqrt()
            sdf_i = torch.where(inside, -d, d)
            better = sdf_i < best
            best = torch.where(better, sdf_i, best)
            best_idx = torch.where(better, arg + offset, best_idx)
            best_neg = torch.where(better, inside, best_neg)
            offset += sg.n
        idx = untile(best_idx)
        neg = untile(best_neg)

    # --- differentiable recompute from the selected segments
    A = torch.cat(verts, 0)
    B = torch.cat([torch.roll(v, -1, dims=0) for v in verts], 0)
    flat = idx.reshape(-1)
    if sdt != dt:
        flat = _refine(flat, verts, A, B, x.detach().reshape(-1), y.detach().reshape(-1))
    # index_select (backward: atomic index_add) rather than A[flat] (backward: a
    # sort-based scatter that is ~100x slower on CUDA when, as here, half a million
    # points share a few hundred vertices)
    a = torch.index_select(A, 0, flat)
    b = torch.index_select(B, 0, flat)
    d2 = _seg_d2(x.reshape(-1), y.reshape(-1), a[:, 0], a[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    d = torch.sqrt(d2.clamp_min(torch.finfo(dt).tiny))
    sign = 1.0 - 2.0 * neg.reshape(-1).to(dt)
    return (d * sign).reshape(x.shape)


def _refine(flat: torch.Tensor, verts: list[torch.Tensor], A: torch.Tensor, B: torch.Tensor,
            px: torch.Tensor, py: torch.Tensor, reach: int = 2) -> torch.Tensor:
    """Re-pick the nearest segment among the +-``reach`` neighbours in full precision.

    A reduced-precision search can only confuse segments whose distances agree to
    its rounding error; away from the medial axis those are the neighbours of the
    true nearest one along the outline (a smooth stretch of boundary seen from
    afar), so checking them in the input dtype restores full accuracy cheaply.
    """
    with torch.no_grad():
        sizes = torch.tensor([v.shape[0] for v in verts], device=flat.device)
        ends = torch.cumsum(sizes, 0)
        pid = torch.searchsorted(ends, flat, right=True)
        st, n = (ends - sizes)[pid], sizes[pid]
        loc = flat - st
        Ad, Bd = A.detach(), B.detach()
        best_i, best_d = flat, None
        for j in range(-reach, reach + 1):
            cand = st + torch.remainder(loc + j, n)
            a, b = Ad[cand], Bd[cand]
            d2 = _seg_d2(px, py, a[:, 0], a[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
            if best_d is None:
                best_d, best_i = d2, cand
            else:
                better = d2 < best_d
                best_d = torch.where(better, d2, best_d)
                best_i = torch.where(better, cand, best_i)
        return best_i


def volume_fraction(sdf: torch.Tensor, h: float, width: float = 1.0) -> torch.Tensor:
    """Solid fraction of each cell from the signed distance at its centre.

    A C1 ramp from 1 (``sdf <= -width h``) to 0 (``sdf >= width h``), 0.5 on the
    boundary: the integral of a hat function of half-width ``width`` cells, i.e. a
    quadratic B-spline step. With ``width = 1`` its derivative is a partition of
    unity on the grid, so for a grid-aligned straight boundary the summed fractions
    equal the exact solid area *whatever the offset* -- the property that makes the
    exact cut-cell fraction attractive -- while staying smooth enough to give a
    clean gradient for shape optimisation (a hard threshold has none, the plain
    linear cut-cell ramp has kinks). For oblique and curved boundaries it is a
    close approximation (second order in h).
    """
    u = (-sdf / (width * h)).clamp(-1.0, 1.0)
    return torch.where(u <= 0, 0.5 * (1.0 + u) ** 2, 1.0 - 0.5 * (1.0 - u) ** 2)
