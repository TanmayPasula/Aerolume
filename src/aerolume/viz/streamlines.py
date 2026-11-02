"""Evenly spaced streamlines (Jobard & Lefer 1997) for live and results views.

Seeding streamlines on a regular grid gives clumps where the flow converges
and holes where it diverges -- exactly around the body, where people look.
Jobard-Lefer placement fixes that: every new line is seeded a distance
``d_sep`` beside an accepted line and is integrated until it comes closer than
``d_test = d_sep / 2`` to any line already drawn. The result is a uniform
"pencil drawing" of the flow at any density, with long, unbroken lines.

The algorithm is inherently sequential (each line depends on the ones before
it), so the inner loop is plain Python on Python floats -- numpy scalar
indexing would be several times slower. Three things keep it fast enough for
an interactive view (tens of milliseconds at the default density):

* the velocity field is block-averaged onto a working grid no finer than the
  integration step (the line cannot resolve anything smaller anyway);
* "is anything within d_test / d_sep of here?" is a single byte lookup in
  occupancy rasters that are stamped with a disc whenever a line is accepted
  (stamping is vectorised numpy writing straight into the ``bytearray``);
* a closed orbit is detected against the line's own visited pixels, keyed by
  arc length, so recirculation bubbles produce one loop, not a knot.

Inputs follow the solver orientation (row 0 = ymin) and ``extent`` is the
outer boundary of the cells, ``(xmin, xmax, ymin, ymax)``. Output polylines are
in the same length units as ``extent`` (metres in Aerolume), ordered along the
flow direction. Only numpy is used; torch tensors are accepted by duck typing.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):                     # torch tensor, without importing torch
        a = a.detach().float().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f x f blocks, padding the far edges by replication."""
    if f == 1:
        return a
    ny, nx = a.shape
    py, px = (-ny) % f, (-nx) % f
    if py or px:
        a = np.pad(a, ((0, py), (0, px)), mode="edge")
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).mean(axis=(1, 3))


def _disc(radius: float) -> np.ndarray:
    """Integer (dy, dx) offsets of a disc of the given pixel radius."""
    r = math.ceil(radius)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = dx * dx + dy * dy <= radius * radius + 1e-9
    return np.stack([dy[keep], dx[keep]], axis=1)


def streamlines(u: Any, v: Any, extent: tuple[float, float, float, float], *,
                density: float = 1.0, mask: Any = None, max_len: float | None = None,
                min_len: float | None = None, spacing: float | None = None,
                seeds: Any = None, max_lines: int = 2000) -> list[np.ndarray]:
    """Evenly spaced streamlines of the velocity field ``(u, v)``.

    ``density`` scales the number of lines: at 1.0 the separation ``d_sep`` is
    1/30 of the shorter side of ``extent`` (``spacing`` sets ``d_sep``
    directly, in length units). ``mask`` is the solid fraction (> 1/2 = solid):
    lines never enter solids, and seeds there are skipped. ``max_len`` caps a
    single line's length (default: twice the domain perimeter) and lines
    shorter than ``min_len`` (default ``d_sep``) are dropped as clutter.
    ``seeds`` is an optional (k, 2) array of points tried first; otherwise the
    first line starts at the inlet mid-height and an even grid of fallback
    seeds fills regions no line reached (closed recirculation, behind bodies).

    Returns a list of (n, 2) float64 polylines, each running downstream.
    """
    u = _to_numpy(u)
    v = _to_numpy(v)
    if u.ndim != 2 or u.shape != v.shape:
        raise ValueError(f"u and v must be equal 2D arrays, got {u.shape} and {v.shape}")
    ny, nx = u.shape
    xmin, xmax, ymin, ymax = (float(e) for e in extent)
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0 or density <= 0:
        raise ValueError("extent must have positive size and density must be > 0")
    hx, hy = width / nx, height / ny

    d_sep = float(spacing) if spacing else min(width, height) / (30.0 * density)
    d_test = 0.5 * d_sep
    ds = d_sep / 4.0                                   # RK2 step: 4 per separation
    max_len = float(max_len) if max_len else 4.0 * (width + height)
    min_len = d_sep if min_len is None else float(min_len)
    max_steps = max(2, int(max_len / ds))

    # -- working grid: fluid-weighted block means, cells no larger than ds --
    fluid = np.ones_like(u)
    if mask is not None:
        m = np.clip(_to_numpy(mask), 0.0, 1.0)
        if m.shape != u.shape:
            raise ValueError(f"mask shape {m.shape} != velocity shape {u.shape}")
        fluid = 1.0 - m
    good = np.isfinite(u) & np.isfinite(v)
    fluid = np.where(good, fluid, 0.0)
    u = np.where(good, u, 0.0)
    v = np.where(good, v, 0.0)
    fct = max(1, int(ds / max(hx, hy)))
    wsum = _block_mean(fluid, fct)
    uw = _block_mean(u * fluid, fct) / np.maximum(wsum, 1e-12)
    vw = _block_mean(v * fluid, fct) / np.maximum(wsum, 1e-12)
    solid_w = wsum < 0.5
    nyw, nxw = uw.shape
    if nxw < 2 or nyw < 2:
        return []
    cx, cy = hx * fct, hy * fct
    speed = np.hypot(uw, vw)[~solid_w]
    if speed.size == 0 or not np.isfinite(speed).all() or speed.max() <= 0.0:
        return []
    vel_eps = 1e-4 * float(np.percentile(speed, 99))

    U = uw.ravel().tolist()
    V = vw.ravel().tolist()
    SOLID = bytes(solid_w.astype(np.uint8).ravel())
    gx_max, gy_max = nxw - 1.000001, nyw - 1.000001
    i_max, j_max = nxw - 2, nyw - 2
    inv_cx, inv_cy = 1.0 / cx, 1.0 / cy

    def direction(x: float, y: float) -> tuple[float, float] | None:
        """Unit flow direction at (x, y) by bilinear interpolation, or None if stagnant."""
        gx = (x - xmin) * inv_cx - 0.5
        gy = (y - ymin) * inv_cy - 0.5
        # Conditional expressions, not min()/max(): this runs twice per step and
        # a builtin call costs more than the comparison itself.
        gx = 0.0 if gx < 0.0 else (gx_max if gx > gx_max else gx)  # noqa: FURB136
        gy = 0.0 if gy < 0.0 else (gy_max if gy > gy_max else gy)  # noqa: FURB136
        i = int(gx)
        j = int(gy)
        if i > i_max:  # noqa: PLR1730
            i = i_max
        if j > j_max:  # noqa: PLR1730
            j = j_max
        fx = gx - i
        fy = gy - j
        k = j * nxw + i
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        uu = U[k] * w00 + U[k + 1] * w10 + U[k + nxw] * w01 + U[k + nxw + 1] * w11
        vv = V[k] * w00 + V[k + 1] * w10 + V[k + nxw] * w01 + V[k + nxw + 1] * w11
        mag = math.sqrt(uu * uu + vv * vv)
        if mag < vel_eps:
            return None
        return uu / mag, vv / mag

    def is_solid(x: float, y: float) -> bool:
        i = int((x - xmin) * inv_cx)
        j = int((y - ymin) * inv_cy)
        if i >= nxw:
            i = nxw - 1
        if j >= nyw:
            j = nyw - 1
        return SOLID[j * nxw + i] == 1

    # -- occupancy rasters (1-pixel pad so 3x3 neighbourhoods never wrap) --
    rpx = d_sep / 8.0
    inv_r = 1.0 / rpx
    nrx = math.ceil(width * inv_r) + 3
    nry = math.ceil(height * inv_r) + 3
    occ_test = bytearray(nrx * nry)                    # within d_test of a line
    occ_seed = bytearray(nrx * nry)                    # within 0.75 d_sep of a line
    occ_test_np = np.frombuffer(occ_test, dtype=np.uint8).reshape(nry, nrx)
    occ_seed_np = np.frombuffer(occ_seed, dtype=np.uint8).reshape(nry, nrx)
    disc_test = _disc(d_test * inv_r)
    # 0.75 rather than 1.0: candidate seeds sit exactly d_sep from their parent
    # line and must survive the +-1 pixel quantisation of the raster.
    disc_seed = _disc(0.75 * d_sep * inv_r)
    neigh = (-nrx - 1, -nrx, -nrx + 1, -1, 0, 1, nrx - 1, nrx, nrx + 1)
    self_gap = 4.0 * ds                                # arc length that counts as "a loop"

    def pixel(x: float, y: float) -> int:
        return (int((y - ymin) * inv_r) + 1) * nrx + int((x - xmin) * inv_r) + 1

    def stamp(pts: np.ndarray) -> None:
        iy = ((pts[:, 1] - ymin) * inv_r).astype(np.int64) + 1
        ix = ((pts[:, 0] - xmin) * inv_r).astype(np.int64) + 1
        for raster, disc in ((occ_test_np, disc_test), (occ_seed_np, disc_seed)):
            yy = np.clip(iy[:, None] + disc[None, :, 0], 0, nry - 1)
            xx = np.clip(ix[:, None] + disc[None, :, 1], 0, nrx - 1)
            raster[yy, xx] = 1

    def trace(x: float, y: float, sign: float, own: dict, steps: int) -> list[tuple[float, float]]:
        """Integrate from (x, y) along sign * direction until a stop condition."""
        out: list[tuple[float, float]] = []
        pdx = pdy = 0.0
        s = 0.0
        h = 0.5 * ds
        for _ in range(steps):
            d = direction(x, y)
            if d is None:
                break
            dx, dy = d[0] * sign, d[1] * sign
            if dx * pdx + dy * pdy < 0.0:              # flow reverses: stagnation / shear line
                break
            d = direction(x + h * dx, y + h * dy)      # RK2 midpoint
            if d is None:
                break
            dx, dy = d[0] * sign, d[1] * sign
            xn, yn = x + ds * dx, y + ds * dy
            if not (xmin <= xn <= xmax and ymin <= yn <= ymax):
                # Leave through the boundary: end exactly on it so lines reach the frame.
                t = 1.0
                if xn > xmax:
                    t = min(t, (xmax - x) / (xn - x))
                elif xn < xmin:
                    t = min(t, (xmin - x) / (xn - x))
                if yn > ymax:
                    t = min(t, (ymax - y) / (yn - y))
                elif yn < ymin:
                    t = min(t, (ymin - y) / (yn - y))
                if t > 1e-6:
                    out.append((x + t * (xn - x), y + t * (yn - y)))
                break
            if is_solid(xn, yn):
                break
            k = pixel(xn, yn)
            if occ_test[k]:
                break
            s += ds * sign
            loop = False
            for o in neigh:
                sk = own.get(k + o)
                if sk is not None and abs(s - sk) > self_gap:
                    loop = True
                    break
            if loop:
                break
            if k not in own:
                own[k] = s
            out.append((xn, yn))
            x, y, pdx, pdy = xn, yn, dx, dy
        return out

    def seed_ok(x: float, y: float) -> bool:
        if not (xmin < x < xmax and ymin < y < ymax):
            return False
        if occ_seed[pixel(x, y)] or is_solid(x, y):
            return False
        return direction(x, y) is not None

    def build(x: float, y: float) -> np.ndarray | None:
        own = {pixel(x, y): 0.0}
        fwd = trace(x, y, 1.0, own, max_steps)
        bwd = trace(x, y, -1.0, own, max_steps - len(fwd))
        if len(fwd) + len(bwd) < 1:
            return None
        pts = np.array(bwd[::-1] + [(x, y)] + fwd, dtype=np.float64)
        length = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        return pts if length >= min_len else None

    lines: list[np.ndarray] = []
    queue: deque[np.ndarray] = deque()

    def try_seed(x: float, y: float) -> None:
        if len(lines) >= max_lines or not seed_ok(x, y):
            return
        line = build(x, y)
        if line is None:
            return
        lines.append(line)
        stamp(line)
        queue.append(line)

    def drain() -> None:
        """Jobard-Lefer: seed new lines d_sep either side of every accepted line."""
        while queue and len(lines) < max_lines:
            pts = queue.popleft().tolist()             # Python floats: fast scalar access
            n = len(pts)
            for k in range(0, n, 2):
                ax, ay = pts[max(k - 1, 0)]
                bx, by = pts[min(k + 1, n - 1)]
                tx, ty = bx - ax, by - ay
                tn = math.hypot(tx, ty)
                if tn == 0.0:
                    continue
                nxv, nyv = -ty / tn * d_sep, tx / tn * d_sep
                px, py = pts[k]
                try_seed(px + nxv, py + nyv)
                try_seed(px - nxv, py - nyv)

    if seeds is not None:
        for sx, sy in np.asarray(seeds, dtype=np.float64).reshape(-1, 2):
            try_seed(float(sx), float(sy))
            drain()
    try_seed(xmin + 0.5 * d_sep, 0.5 * (ymin + ymax))
    drain()
    # Fallback grid, column by column from the inlet: picks up regions the
    # propagation could not reach (closed orbits, flow cut off by bodies).
    for gx in np.arange(xmin + 0.5 * d_sep, xmax, d_sep):
        for gy in np.arange(ymin + 0.5 * d_sep, ymax, d_sep):
            try_seed(float(gx), float(gy))
            drain()
    return lines
