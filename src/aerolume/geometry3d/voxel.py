"""Robust voxelisation and narrow-band signed distance of triangle meshes (torch, GPU or CPU).

The 3D solver sees a body only through two fields on its uniform grid: which
cell centres are inside, and how far each cell near the wall is from it. Real
imported meshes break the textbook inside test in every way there is: holes and
gaps (missing patches, unwelded seams), flipped or inconsistent winding, parts
overlapping each other, duplicated faces, internal walls, open sheets. This
module is built so that none of those changes the answer by more than a few
cells near the defect.

Inside test: ray-parity voting
------------------------------
For each of the three axes, one ray runs along every grid line through the cell
centres. Crossings with triangles are found by binning the triangles over the
ray lattice (a triangle is tested only against the rays inside its projected
bounding box, so the cost is ~ triangles + crossings, never rays x triangles), and
a cumulative sum of crossing counts along the ray gives every cell the parity
of crossings before it -- inside when odd. Orientation is never used, so flipped
normals cannot matter. The robustness comes from three rules:

* **A ray with an odd total abstains.** A closed surface is crossed an even
  number of times; an odd count means the ray slipped through a hole, so its
  verdict is unreliable for its whole length and it does not vote. A hole is
  crossed by the rays of at most a few axes, and the others still decide.
* **Majority of the votes.** Each cell is inside when more axes vote inside than
  outside. Cells with a tie (two axes disagree, one abstains, or all abstain --
  only possible right next to a hole) are filled from the majority of their
  decided neighbours.
* **Overlapping parts are tested separately.** Parity is wrong where two closed
  shells overlap (a wheel pushed into the body, CAD unions exported as
  intersecting shells: two crossings in, "outside"). The surface is therefore
  split into units -- each closed component (faces connected through shared
  edges; touching at a vertex does not join them) on its own, and per mesh the
  union of its open components (so open patches that close each other stay
  together) -- and the units are packed greedily into *layers* whose bounding
  boxes do not overlap. Parity is evaluated per layer and the layers are united.

With this a mesh with a missing patch, a mesh with randomly flipped faces and
two interpenetrating spheres all voxelise like their clean, closed versions.

Narrow-band distance
--------------------
:func:`sdf_band` adds the exact distance to the nearest triangle for every cell
within ``band`` of the surface (sign from the inside test, negative inside):
triangles are binned to the cells of their bounding box grown by the band
(large triangles are first split, so a slanted 100-cell face does not enumerate
a million-cell box), each (cell, triangle) pair's exact point-triangle distance
is packed with the triangle id into one int64 key whose bit pattern orders like
the distance, and a ``scatter_reduce(amin)`` keeps the nearest triangle per cell.
That search runs without autograd in float32 grid coordinates; the distance is
then recomputed from the selected triangle's own vertices with autograd on, so
it is differentiable with respect to vertex tensors that require grad (the
gradient of a minimum is that of its active element: exactly the brute-force
gradient, at O(cells in band) memory). This is the 2D ``geometry.sdf`` design
carried to 3D.

*Through the surface.* ``sign * sqrt(d²)`` has no derivative where ``d = 0`` --
a cell centre exactly on a face, which every grid-aligned flat face of a box
produces in rows -- and its sign comes from the inside test, whose rays are
shifted by ~0.002 cells (below) and so may put a centre within that distance of
a face on the wrong side. Within ``_PLANE_ZONE`` (0.02) cells of a face whose
plane holds the nearest point, the distance is therefore the face's signed plane
distance ``s n.(p - a)``, linear through zero, where ``s`` says which side of
the face is outside: the vote of all band cells clearly off that face (parity
sign times the side of the plane they are on). Elsewhere nothing changes. A
sensitivity check on a box with faces on grid planes went from 1.5 % (CD) and
14 % (one CL mode) to the agreement of an off-grid box.

Buried surfaces -- the part of a wheel inside the body, the floor patch inside a
part pushed through it, internal walls -- must not count as boundary, or cells
deep in the solid would see a surface next to them and get a fraction below 1.
Triangle pieces are therefore classified before the search (``_classify``):

* *Across parts.* When the surface has several layers, each layer gets its own
  narrow-band SDF (its own triangles, its own parity as the sign); a piece is
  buried when all four of its sample points (centroid and three points near its
  corners) lie more than 0.1 cells inside another layer's solid, by trilinear
  interpolation of that SDF -- a sub-cell accurate test.
* *Within a part.* A piece is buried when the inside mask is solid at all 8 cell
  centres around probes 0.75 cells off both sides at those four points. Next to
  a locally flat boundary one of the 8 centres always lies beyond the probe, so a
  real boundary piece can never be mistaken for a buried one (nor for a sheet).
* Pieces that are neither clearly buried nor clearly on the boundary (no inside
  centre around a probe 2 cells out on one side) are split to one cell, and those
  still partly inside another part to a quarter cell, and tested again.

Measured on a 4-cell-wide box pushed through a floor and a sphere half sunk in a
box: outside the union the distance is exact (1e-15); inside, the depth is within
about half a cell of the exact union depth (only near the crease), everything
more than ~1.3 cells deep is fully solid, and the fraction differs from that of
the exact union by at most 0.11.

An open sheet (a spoiler or a fin modelled as a single surface) encloses no
volume, so no cell centre is inside it: through the ramp it would only show up
as a half-strength smear a quarter to half a cell thick. Pieces of faces of
*open* components with no inside centre around either side are therefore
treated as sheets and given a minimum thickness (``sheet_thickness``, one cell
by default): the field becomes the union of the solid and a slab of that
thickness around them. Closed parts are never thickened -- a thin but closed
trailing edge keeps its true (under-resolved) shape, as in 2D, and
:func:`~aerolume.geometry3d.check_design3d` reports it.

Grids that cut the surface
--------------------------
A solver grid often holds only part of the body: the half domain at a symmetry
plane, a road through the wheels, a window of a larger domain. The inside test
counts every crossing along the infinite ray, so the sign is right there; but
the classification probes of pieces near the cut read centres beyond the grid,
where solid would look like empty space -- a boundary piece would pass for a
sheet and be thickened by half a cell (distances up to 0.45 h too small next to
a symmetry plane, measured by the solver), and a buried piece would be kept.
So the inside test the classification reads is evaluated on the grid grown by
as many cells as the probes of the pieces within the band reach
(``_cut_padding``: the band, the piece size and three cells, and only where the
surface continues beyond the grid). The pieces themselves do not depend on the
grid, so the distances inside the grid equal those of a grid that holds the
whole surface (tests: half domains, windows, a road through open tyres, sheets
crossing a symmetry plane). As a last guard, a piece is only called a sheet when
every centre its probes read is known (in the field, or beyond the surface's
bounding box).

:func:`solid_fraction` turns the distance into the smooth solid fraction the
solver uses, with the same C1 ramp as the 2D code
(:func:`aerolume.geometry.sdf.volume_fraction`), and :func:`voxelize` does all of
it in one call. Everything is chunked to a byte budget (default 384 MB of
temporaries), so memory stays bounded for any mesh and grid.

Grid convention: ``Grid(nx, ny, nz, h, x0, y0, z0)`` with ``(x0, y0, z0)`` the min
corner of the cell edges; cell (k, j, i) has its centre at
``(x0 + (i + 0.5) h, y0 + (j + 0.5) h, z0 + (k + 0.5) h)`` and arrays are
``(nz, ny, nx)``, matching ``VolumeData`` and the 3D solver.
"""
from __future__ import annotations

import math
import time
from typing import Any, NamedTuple

import numpy as np
import torch

from ..model import Mesh3D
from ._rows import unique_rows

__all__ = ["Grid", "grid_around", "inside_mask", "sdf_band", "sdf_bands", "solid_fraction", "voxelize"]

#: ray offsets (in cells) off the cell-centre lattice: an irrational-looking shift makes
#: a ray through a mesh vertex or along an edge (double / missed crossings) practically
#: impossible for meshes built on round coordinates; 0.002 cells is far below anything
#: that matters for an inside test at cell centres
_JITTER = {0: (0.00213, 0.00137), 1: (0.00171, 0.00229), 2: (0.00119, 0.00191)}
_MAX_LAYERS = 8
_GREEDY_UNITS = 4000
_BYTES_PER_PAIR = 320


class Grid(NamedTuple):
    """A uniform cell grid: ``nx * ny * nz`` cells of size ``h`` from the min corner (x0, y0, z0)."""
    nx: int
    ny: int
    nz: int
    h: float
    x0: float
    y0: float
    z0: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """Array shape (nz, ny, nx)."""
        return (int(self.nz), int(self.ny), int(self.nx))

    @property
    def cells(self) -> int:
        return int(self.nx) * int(self.ny) * int(self.nz)

    @property
    def extent(self) -> tuple[float, float, float, float, float, float]:
        """Outer cell edges (x0, x1, y0, y1, z0, z1)."""
        return (self.x0, self.x0 + self.nx * self.h, self.y0, self.y0 + self.ny * self.h,
                self.z0, self.z0 + self.nz * self.h)

    def centres(self, axis: str) -> np.ndarray:
        """Cell-centre coordinates along ``axis`` ("x", "y" or "z")."""
        n, o = {"x": (self.nx, self.x0), "y": (self.ny, self.y0), "z": (self.nz, self.z0)}[axis]
        return o + (np.arange(n) + 0.5) * self.h

    def faces(self, axis: str) -> Grid:
        """The grid whose cell centres are this grid's face centres normal to ``axis``.

        One more point along ``axis``, shifted back half a cell: for "x" its
        centres are ``(x0 + i h, y0 + (j + 1/2) h, z0 + (k + 1/2) h)``, i = 0..nx --
        the u points of a MAC grid (likewise v for "y", w for "z"), where a
        staggered solver needs the solid fraction.
        """
        h = self.h
        if axis == "x":
            return Grid(self.nx + 1, self.ny, self.nz, h, self.x0 - 0.5 * h, self.y0, self.z0)
        if axis == "y":
            return Grid(self.nx, self.ny + 1, self.nz, h, self.x0, self.y0 - 0.5 * h, self.z0)
        if axis == "z":
            return Grid(self.nx, self.ny, self.nz + 1, h, self.x0, self.y0, self.z0 - 0.5 * h)
        raise ValueError("axis must be 'x', 'y' or 'z'")


def grid_around(meshes, h: float, pad: float = 3.0) -> Grid:
    """The smallest grid of cell size ``h`` holding the meshes plus ``pad`` cells on every side."""
    V, _, _, _ = _gather(meshes)
    if len(V) == 0:
        raise ValueError("no vertices")
    lo = V.min(0) - pad * h
    hi = V.max(0) + pad * h
    n = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    return Grid(int(n[0]), int(n[1]), int(n[2]), float(h), float(lo[0]), float(lo[1]), float(lo[2]))


def _as_grid(grid) -> Grid:
    g = Grid(*grid)
    if min(g.nx, g.ny, g.nz) < 1 or not g.h > 0:
        raise ValueError(f"invalid grid {tuple(grid)!r}")
    return Grid(int(g.nx), int(g.ny), int(g.nz), float(g.h), float(g.x0), float(g.y0), float(g.z0))


# ---------------------------------------------------------------------------
# input + topology (numpy, CPU, no autograd)
# ---------------------------------------------------------------------------

def _items(meshes) -> list[tuple[Any, Any]]:
    """Normalise to a list of (vertices, faces); accepts Mesh3D, (V, F) or lists of them."""
    if isinstance(meshes, Mesh3D):
        return [(meshes.vertices, meshes.faces)]
    if isinstance(meshes, tuple) and len(meshes) == 2 and not isinstance(meshes[0], (Mesh3D, tuple)):
        v0 = meshes[0]
        if getattr(v0, "ndim", 0) == 2 and v0.shape[-1] == 3:
            return [meshes]
    out = []
    for m in meshes:
        if isinstance(m, Mesh3D):
            out.append((m.vertices, m.faces))
        elif isinstance(m, dict):
            out.append((m["vertices"], m["faces"]))
        else:
            v, f = m
            out.append((v, f))
    return out


def _np64(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().to("cpu", torch.float64).numpy().reshape(-1, 3)
    return np.asarray(v, dtype=np.float64).reshape(-1, 3)


def _np_faces(f) -> np.ndarray:
    if isinstance(f, torch.Tensor):
        f = f.detach().cpu().numpy()
    return np.asarray(f, dtype=np.int64).reshape(-1, 3)


def _gather(meshes) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Concatenated float64 vertices, global faces, mesh id per face, and the raw items."""
    items = _items(meshes)
    vs, fs, ms, off = [], [], [], 0
    for k, (v, f) in enumerate(items):
        vv, ff = _np64(v), _np_faces(f)
        if len(ff) and (ff.min() < 0 or ff.max() >= len(vv)):
            raise ValueError(f"mesh {k}: face index out of range")
        vs.append(vv)
        fs.append(ff + off)
        ms.append(np.full(len(ff), k, dtype=np.int64))
        off += len(vv)
    V = np.concatenate(vs) if vs else np.zeros((0, 3))
    F = np.concatenate(fs) if fs else np.zeros((0, 3), dtype=np.int64)
    M = np.concatenate(ms) if ms else np.zeros(0, dtype=np.int64)
    return V, F, M, items


class _Topo:
    """Cleaned faces plus the units (closed components / per-mesh open parts) for layering.

    ``keep``: indices of the faces kept (non-degenerate, not duplicated) in the
    concatenated face list; ``unit``: unit id per kept face; ``ubox``: (U, 6)
    bounding boxes (lo xyz, hi xyz) in metres.
    """

    def __init__(self, V: np.ndarray, F: np.ndarray, M: np.ndarray) -> None:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        self.n_input = len(F)
        self.open_face = np.zeros(0, dtype=bool)
        if len(F) == 0:
            self.keep = np.zeros(0, dtype=np.int64)
            self.unit = np.zeros(0, dtype=np.int64)
            self.ubox = np.zeros((0, 6))
            self.n_units = 0
            self.n_closed = 0
            self.removed = 0
            return
        # weld bit-identical positions (topology only; geometry is untouched)
        wid = unique_rows(V)[2]
        W = wid[F]
        t = V[F]
        cr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        ok = (W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 0] != W[:, 2]) & np.any(cr != 0, axis=1)
        # coincident duplicates (same welded corners, same mesh, any order): keep one
        key = np.concatenate([M[:, None], np.sort(W, axis=1)], axis=1)
        first = unique_rows(key)[1]
        dup = np.ones(len(F), dtype=bool)
        dup[first] = False
        ok &= ~dup
        self.keep = np.flatnonzero(ok)
        self.removed = int(len(F) - len(self.keep))
        Wk, Mk = W[self.keep], M[self.keep]
        nk = len(Wk)
        if nk == 0:
            self.unit = np.zeros(0, dtype=np.int64)
            self.ubox = np.zeros((0, 6))
            self.n_units = 0
            self.n_closed = 0
            return
        nw = int(wid.max()) + 1
        vkey = Wk + Mk[:, None] * nw                      # vertices are per mesh
        _, vinv = np.unique(vkey.reshape(-1), return_inverse=True)
        vinv = vinv.reshape(-1, 3)
        e = np.sort(np.stack([vinv[:, [0, 1]], vinv[:, [1, 2]], vinv[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        _, _, einv, ecnt = unique_rows(e)
        # components through shared *edges*: two shells that merely touch at a vertex
        # (overlapping primitives often do) must stay separate units, or parity fails
        # in their overlap
        rows = np.repeat(np.arange(nk), 3)
        ne = len(ecnt)
        g = coo_matrix((np.ones(3 * nk, dtype=np.int8), (rows, nk + einv)), shape=(nk + ne, nk + ne))
        _, lab = connected_components(g, directed=False)
        _, comp = np.unique(lab[:nk], return_inverse=True)
        comp = comp.reshape(-1)
        # boundary edges -> open components
        open_face = (ecnt[einv.reshape(-1)] == 1).reshape(-1, 3).any(1)
        ncomp = int(comp.max()) + 1
        comp_open = np.zeros(ncomp, dtype=bool)
        np.logical_or.at(comp_open, comp, open_face)
        comp_mesh = np.zeros(ncomp, dtype=np.int64)
        comp_mesh[comp] = Mk
        self.open_face = comp_open[comp]                  # kept face belongs to an open component
        # units: every closed component alone; all open components of one mesh together
        n_mesh = int(M.max()) + 1
        closed_ids = np.flatnonzero(~comp_open)
        cmap = np.full(ncomp, -1, dtype=np.int64)
        cmap[closed_ids] = np.arange(len(closed_ids))
        open_mesh_unit = len(closed_ids) + np.arange(n_mesh)
        unit_of_comp = np.where(comp_open, open_mesh_unit[comp_mesh], cmap)
        u = unit_of_comp[comp]
        _, self.unit = np.unique(u, return_inverse=True)
        self.unit = self.unit.reshape(-1)
        self.n_units = int(self.unit.max()) + 1
        self.n_closed = len(closed_ids)
        tk = t[self.keep]
        lo, hi = tk.min(1), tk.max(1)
        ub = np.empty((self.n_units, 6))
        for a in range(3):
            ub[:, a] = np.full(self.n_units, np.inf)
            np.minimum.at(ub[:, a], self.unit, lo[:, a])
            ub[:, 3 + a] = np.full(self.n_units, -np.inf)
            np.maximum.at(ub[:, 3 + a], self.unit, hi[:, a])
        self.ubox = ub

    def layers(self, h: float) -> list[np.ndarray]:
        """Unit ids grouped into layers of pairwise disjoint (one-cell padded) boxes."""
        if self.n_units == 0:
            return []
        ub = self.ubox
        vol = np.prod(np.maximum(ub[:, 3:] - ub[:, :3], h), axis=1)
        order = np.argsort(-vol, kind="stable")
        layers: list[list[int]] = []
        buf: list[np.ndarray] = []          # per layer: (capacity, 6) boxes, first cnt[L] rows used
        cnt: list[int] = []

        def add(L: int, b: np.ndarray) -> None:
            if cnt[L] == len(buf[L]):
                buf[L] = np.concatenate([buf[L], np.empty_like(buf[L])])
            buf[L][cnt[L]] = b
            cnt[L] += 1

        for rank, u in enumerate(order):
            b = ub[u]
            placed = False
            if rank < _GREEDY_UNITS:
                for L in range(len(layers)):
                    arr = buf[L][:cnt[L]]
                    if not np.any(np.all((b[:3] <= arr[:, 3:] + h) & (arr[:, :3] <= b[3:] + h), axis=1)):
                        layers[L].append(int(u))
                        add(L, b)
                        placed = True
                        break
                if not placed and len(layers) < _MAX_LAYERS:
                    layers.append([int(u)])
                    buf.append(np.empty((16, 6)))
                    cnt.append(0)
                    add(len(layers) - 1, b)
                    placed = True
            if not placed:
                # beyond the caps: the smallest units share the last layer (only their own
                # overlaps, tiny by then, can be misjudged)
                layers[-1].append(int(u))
        return [np.asarray(sorted(L), dtype=np.int64) for L in layers]


# ---------------------------------------------------------------------------
# device helpers
# ---------------------------------------------------------------------------

def _device(device, items) -> torch.device:
    for v, _ in items:
        if isinstance(v, torch.Tensor):
            return v.device if device in (None, "auto") else torch.device(device)
    if device in (None, "auto"):
        from ..core.device import resolve
        return torch.device(resolve("auto"))
    return torch.device(device)


def _index_coords(V: np.ndarray, g: Grid) -> np.ndarray:
    """Vertex coordinates in cell units with cell centres on the integers."""
    o = np.array([g.x0, g.y0, g.z0])
    return (V - o) / g.h - 0.5


# ---------------------------------------------------------------------------
# inside test
# ---------------------------------------------------------------------------

def _chunks(cnt: torch.Tensor, max_pairs: int) -> list[tuple[int, int]]:
    """Split a per-triangle pair count into contiguous chunks of at most ~max_pairs pairs."""
    cs = torch.cumsum(cnt, 0).cpu().numpy()
    out, start, base = [], 0, 0
    n = len(cs)
    while start < n:
        stop = int(np.searchsorted(cs, base + max_pairs, side="right"))
        stop = min(max(stop, start + 1), n)
        out.append((start, stop))
        base = int(cs[stop - 1])
        start = stop
    return out


def _axis_votes(T: torch.Tensor, axis: int, g: Grid, max_pairs: int, stats: dict
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Parity of crossings before each cell and ray validity for rays along ``axis``.

    ``T``: (M, 3, 3) float32 triangle corners in index coordinates. Returns
    (inside, valid) as bool tensors of shape (nz, ny, nx); ``valid`` is False on
    rays with an odd number of crossings (they abstain).
    """
    dev = T.device
    n = (g.nx, g.ny, g.nz)
    pa, qa = [k for k in range(3) if k != axis]
    npp, nq, nc = n[pa], n[qa], n[axis]
    dp, dq = _JITTER[axis]
    P, Q, C = T[..., pa], T[..., qa], T[..., axis]
    p0 = torch.ceil(P.amin(1) - dp).clamp_(min=0).to(torch.int64)
    p1 = torch.floor(P.amax(1) - dp).clamp_(max=npp - 1).to(torch.int64)
    q0 = torch.ceil(Q.amin(1) - dq).clamp_(min=0).to(torch.int64)
    q1 = torch.floor(Q.amax(1) - dq).clamp_(max=nq - 1).to(torch.int64)
    wp = (p1 - p0 + 1).clamp_(min=0)
    wq = (q1 - q0 + 1).clamp_(min=0)
    cnt = wp * wq
    sel = torch.nonzero(cnt > 0).squeeze(1)
    counts = torch.zeros(nq * npp * (nc + 1), dtype=torch.int32, device=dev)
    if sel.numel():
        P, Q, C = P[sel], Q[sel], C[sel]
        p0, q0, wp, cnt = p0[sel], q0[sel], wp[sel], cnt[sel]
        for s0, s1 in _chunks(cnt, max_pairs):
            c = cnt[s0:s1]
            tot = int(c.sum())
            tri = torch.repeat_interleave(torch.arange(s0, s1, device=dev), c)
            start = torch.cumsum(c, 0) - c
            local = torch.arange(tot, device=dev) - torch.repeat_interleave(start, c)
            w = wp[tri]
            pi = p0[tri] + local % w
            qi = q0[tri] + local // w
            del local, w
            px = pi.to(torch.float32) + dp
            qx = qi.to(torch.float32) + dq
            Pt, Qt = P[tri], Q[tri]
            ax_, bx_, cx_ = Pt[:, 0] - px, Pt[:, 1] - px, Pt[:, 2] - px
            ay_, by_, cy_ = Qt[:, 0] - qx, Qt[:, 1] - qx, Qt[:, 2] - qx
            wA = bx_ * cy_ - by_ * cx_
            wB = cx_ * ay_ - cy_ * ax_
            wC = ax_ * by_ - ay_ * bx_
            s = wA + wB + wC
            hit = (((wA >= 0) & (wB >= 0) & (wC >= 0)) | ((wA <= 0) & (wB <= 0) & (wC <= 0))) & (s != 0)
            h_idx = torch.nonzero(hit).squeeze(1)
            if h_idx.numel():
                Ct = C[tri[h_idx]]
                ss = s[h_idx]
                cc = (wA[h_idx] * Ct[:, 0] + wB[h_idx] * Ct[:, 1] + wC[h_idx] * Ct[:, 2]) / ss
                b = (torch.floor(cc) + 1).clamp_(0, nc).to(torch.int64)
                flat = (qi[h_idx] * npp + pi[h_idx]) * (nc + 1) + b
                counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
            stats["pairs"] = stats.get("pairs", 0) + tot
            stats["crossings"] = stats.get("crossings", 0) + int(h_idx.numel())
    Cn = counts.view(nq, npp, nc + 1)
    odd = (Cn.sum(-1) % 2).bool()
    par = (torch.cumsum(Cn[..., :nc], -1, dtype=torch.int32) % 2).bool()
    stats.setdefault("odd_rays", [0, 0, 0])[axis] += int(odd.sum())
    valid = (~odd)[..., None].expand_as(par)
    perm = {0: (0, 1, 2), 1: (0, 2, 1), 2: (2, 0, 1)}[axis]
    return par.permute(*perm), valid.permute(*perm)


def _inside(V: np.ndarray, F: np.ndarray, topo: _Topo, g: Grid, dev: torch.device, budget_mb: float,
            stats: dict, layer_out: list | None = None) -> torch.Tensor:
    shape = g.shape
    inside = torch.zeros(shape, dtype=torch.bool, device=dev)
    undecided = torch.zeros(shape, dtype=torch.bool, device=dev)
    if len(topo.keep) == 0:
        return inside
    Fk = F[topo.keep]
    TI = _index_coords(V, g)[Fk]                   # (M, 3, 3) float64
    max_pairs = max(int(budget_mb * 2**20 / 160), 65536)
    layers = topo.layers(g.h)
    stats["layers"] = len(layers)
    stats["units"] = topo.n_units
    stats["closed_units"] = topo.n_closed
    for L in layers:
        fsel = np.flatnonzero(np.isin(topo.unit, L))
        T = torch.as_tensor(TI[fsel], dtype=torch.float32, device=dev)
        vin = torch.zeros(shape, dtype=torch.int8, device=dev)
        vout = torch.zeros(shape, dtype=torch.int8, device=dev)
        for axis in range(3):
            par, valid = _axis_votes(T, axis, g, max_pairs, stats)
            vin += par & valid
            vout += (~par) & valid
            del par, valid
        lin = vin > vout
        inside |= lin
        undecided |= vin == vout
        if layer_out is not None:
            layer_out.append((fsel, lin))           # kept-face indices of the layer, its own solid
        del vin, vout, T
    undecided &= ~inside
    nund = int(undecided.sum())
    stats["undecided_cells"] = nund
    # fill ties from the decided neighbours (they sit next to holes)
    decided = ~undecided
    for _ in range(6):
        if nund == 0:
            break
        num = torch.nn.functional.avg_pool3d((inside & decided).to(torch.float32)[None, None], 3, 1, 1)[0, 0]
        den = torch.nn.functional.avg_pool3d(decided.to(torch.float32)[None, None], 3, 1, 1)[0, 0]
        take = undecided & (den > 0)
        inside |= take & (num > 0.5 * den)
        decided |= take
        undecided &= ~take
        nund = int(undecided.sum())
    stats["unresolved_cells"] = nund
    return inside


def inside_mask(meshes, grid, *, device: str | torch.device | None = None, budget_mb: float = 384.0,
                info: dict | None = None) -> torch.Tensor:
    """Bool (nz, ny, nx): which cell centres are inside the union of the meshes.

    Robust to holes, flipped faces, overlapping parts and duplicates (module
    docstring). ``meshes``: Mesh3D, (vertices, faces) pairs (numpy or torch) or a
    list of them. ``device`` defaults to the vertex tensors' device, else the
    best available. ``info`` (a dict) receives counts and timings.
    """
    t0 = time.perf_counter()
    g = _as_grid(grid)
    V, F, M, items = _gather(meshes)
    dev = _device(device, items)
    topo = _Topo(V, F, M)
    stats: dict = {}
    with torch.no_grad():
        ins = _inside(V, F, topo, g, dev, budget_mb, stats)
    if info is not None:
        info.update(stats)
        info.update(triangles=len(topo.keep), removed_faces=topo.removed, device=str(dev),
                    seconds_inside=time.perf_counter() - t0)
    return ins


# ---------------------------------------------------------------------------
# narrow-band distance
# ---------------------------------------------------------------------------

def _seg_d2(ap: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(ap.dtype).tiny
    t = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(tiny)).clamp(0.0, 1.0)
    d = ap - t[..., None] * ab
    return (d * d).sum(-1)


def _tri_core(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Point-triangle geometry, row-wise, all (K, 3): (squared distance, the nearest
    point is in the triangle's interior or on its boundary, ``n . (p - a)``, ``|n|²``)
    with ``n = (b - a) x (c - a)`` the winding normal (not normalised)."""
    tiny = torch.finfo(p.dtype).tiny
    ab, bc, ca = b - a, c - b, a - c
    ap, bp, cp = p - a, p - b, p - c
    n = torch.linalg.cross(ab, -ca)
    nn = (n * n).sum(-1)
    s0 = (torch.linalg.cross(ab, ap) * n).sum(-1)
    s1 = (torch.linalg.cross(bc, bp) * n).sum(-1)
    s2 = (torch.linalg.cross(ca, cp) * n).sum(-1)
    inside = (s0 >= 0) & (s1 >= 0) & (s2 >= 0) & (nn > tiny)
    dpl = (ap * n).sum(-1)
    plane = dpl * dpl / nn.clamp_min(tiny)
    edge = torch.minimum(torch.minimum(_seg_d2(ap, ab), _seg_d2(bp, bc)), _seg_d2(cp, ca))
    return torch.where(inside, plane, edge), inside, dpl, nn


def _tri_d2(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Squared distance from points p to triangles (a, b, c), row-wise, all (K, 3).

    Inside the prism over the triangle it is the plane distance, elsewhere the
    nearest of the three edges -- exact, branch-free, differentiable almost
    everywhere (written without in-place ops so autograd can use it), and exact
    for degenerate (needle or point) triangles too.
    """
    return _tri_core(p, a, b, c)[0]


def _subdivide(T: torch.Tensor, max_edge: float, W: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split triangles longer than ``max_edge`` (cells) 1-to-4 until none is.

    Returns (pieces, parent row of each piece, barycentric weights): piece corner
    ``j`` is ``sum_k W[j, k] * corner_k`` of the original triangle, so a piece can
    be rebuilt -- differentiably -- from its parent's vertices. ``W`` gives the
    weights of ``T`` itself when it already holds pieces (identity otherwise).
    Every piece keeps its parent's orientation (the 1-to-4 split preserves the
    winding), so their winding normals agree.
    """
    n = T.shape[0]
    parent = torch.arange(n, device=T.device)
    if W is None:
        W = torch.eye(3, dtype=T.dtype, device=T.device).expand(n, 3, 3)
    X = torch.cat([T, W.to(T.dtype)], 2)                  # geometry and weights split alike
    for _ in range(16):
        P = X[..., :3]
        e = torch.stack([(P[:, 1] - P[:, 0]).norm(dim=-1), (P[:, 2] - P[:, 1]).norm(dim=-1),
                         (P[:, 0] - P[:, 2]).norm(dim=-1)], 1).amax(1)
        big = e > max_edge
        if not bool(big.any()):
            break
        Xb = X[big]
        a, b, c = Xb[:, 0], Xb[:, 1], Xb[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        new = torch.stack([torch.stack([a, ab, ca], 1), torch.stack([ab, b, bc], 1),
                           torch.stack([ca, bc, c], 1), torch.stack([ab, bc, ca], 1)])
        X = torch.cat([X[~big], new.reshape(-1, 3, 6)])
        parent = torch.cat([parent[~big], parent[big].repeat(4)])
    return X[..., :3].contiguous(), parent, X[..., 3:].contiguous()


_KEY_EMPTY = torch.iinfo(torch.int64).max


_PROBE = 0.75     # cells off a face centre at which its two sides are classified
_PROBE_FAR = 2.0  # cells: the probe that tells a buried piece from one next to the boundary
_GATHER_CHUNK = 1 << 20   # points per 8-corner gather (bounds the (n, 8) index temporaries)
#: Cells a classification probe can reach from a piece: the far probe plus the 2x2x2 block
#: of centres around it. A grid that cuts the surface grows this far for the classification.
_PROBE_REACH = _PROBE_FAR + 1.0
#: Half-width (cells) of the zone around a face in which the distance is its signed plane
#: distance (see _sdf): well above the ray jitter (0.0033 cells) and float32 index rounding.
_PLANE_ZONE = 0.02


class _Pad:
    """A cell field padded by two cells of ``fill`` and flattened, for 8-corner gathers.

    The padding makes every corner of a clamped base cell a valid index, so the 8
    centres around any point come from one flat gather with no bounds tests:
    points off the field simply read ``fill``. The field may be larger than the
    grid the queries are written in (a grid that cuts the surface, see
    :func:`_cut_padding`): ``off`` is the field index of that grid's cell
    (0, 0, 0). ``lo`` / ``hi`` bound the surface in field index coordinates:
    centres beyond them are outside the solid whatever the field holds, which is
    what makes an off-field read of ``False`` trustworthy there (:meth:`known`).
    """

    def __init__(self, F: torch.Tensor, fill, off=(0, 0, 0), lo=None, hi=None) -> None:
        nz, ny, nx = F.shape
        dev = F.device
        P = torch.full((nz + 4, ny + 4, nx + 4), fill, dtype=F.dtype, device=dev)
        P[2:-2, 2:-2, 2:-2] = F
        self.flat = P.reshape(-1)
        self.hi = torch.tensor([nx, ny, nz], device=dev)
        sy, sz = nx + 4, (nx + 4) * (ny + 4)
        self.stride = torch.tensor([1, sy, sz], device=dev)
        self.offs = torch.tensor([(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)],
                                 device=dev)                              # (8, 3)
        self.flat_offs = (self.offs * self.stride).sum(1)                 # (8,)
        self.off = torch.tensor([float(o) for o in off], dtype=torch.float32, device=dev)
        inf = float("inf")
        self.slo = torch.tensor(lo if lo is not None else (-inf, -inf, -inf), dtype=torch.float32, device=dev)
        self.shi = torch.tensor(hi if hi is not None else (inf, inf, inf), dtype=torch.float32, device=dev)

    def corners(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Values (n, 8) at the 8 cell centres around query points ``q`` (n, 3), and the
        fractional position (n, 3) of ``q`` within them."""
        q = q + self.off
        b = torch.floor(q)
        t = q - b
        b = torch.minimum(torch.maximum(b.to(torch.int64), torch.full_like(self.hi, -2)), self.hi)
        base = ((b + 2) * self.stride).sum(1)
        return self.flat[base[:, None] + self.flat_offs[None, :]], t

    def known(self, q: torch.Tensor) -> torch.Tensor:
        """(n,) bool: all 8 centres around ``q`` lie in the field, or all lie beyond the
        surface's bounding box along one axis (outside the solid for certain)."""
        b = torch.floor(q + self.off)
        in_field = ((b >= 0) & (b + 1 <= (self.hi - 1).to(b.dtype))).all(1)
        beyond = ((b + 1 < self.slo) | (b > self.shi)).any(1)
        return in_field | beyond


def _sample_points(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(4 sample points per piece (4, M, 3), unit normals (M, 3), non-degenerate (M,))."""
    n = torch.linalg.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = n.norm(dim=1, keepdim=True)
    ok = ln[:, 0] > 0
    n = n / ln.clamp_min(torch.finfo(T.dtype).tiny)
    cen = T.mean(1)
    pts = torch.stack([cen] + [0.8 * T[:, i] + 0.2 * cen for i in range(3)])
    return pts, n, ok


def _probe_sides(T: torch.Tensor, inside: _Pad, probe: float = _PROBE) -> tuple[torch.Tensor, ...]:
    """How the two sides of each triangle look in the inside mask.

    ``T``: (M, 3, 3) corners in index coordinates; ``inside``: the padded mask.
    Each side is probed at four points of the triangle -- its centroid and three
    points near its corners (``0.8 v_i + 0.2 c``) -- moved ``probe`` cells along
    the unit normal, each through the 8 cell centres around it; off-field centres
    read as outside. Returns ``(all_in(+), any_in(+), all_in(-), any_in(-),
    non-degenerate, known)``, "all" and "any" over every centre of every point of
    that side, ``known`` when every centre of both sides is known
    (:meth:`_Pad.known`: in the field, or beyond the surface's bounding box). The
    8-centre rule makes the classification safe: next to a locally flat boundary
    at least one of the 8 centres lies beyond the probe point, so a boundary
    face is never "inside on both sides" (buried) nor "outside on both sides"
    (sheet) however it sits between cell centres (the nearest single centre can
    be up to 0.87 cells away, on the wrong side). The four points stop a piece
    that straddles the boundary from being judged by its centroid alone.
    """
    M = T.shape[0]
    out = [torch.empty(M, dtype=torch.bool, device=T.device) for _ in range(6)]
    for s0 in range(0, M, _GATHER_CHUNK):
        sl = slice(s0, s0 + _GATHER_CHUNK)
        pts, n, ok = _sample_points(T[sl])
        res = []
        known = torch.ones_like(ok)
        for sgn in (1.0, -1.0):
            q = (pts + sgn * probe * n[None]).reshape(-1, 3)
            v, _ = inside.corners(q)                                   # (4 m, 8)
            v = v.reshape(4, -1, 8)
            res += [v.all(2).all(0), v.any(2).any(0)]
            known &= inside.known(q).reshape(4, -1).all(0)
        for k, r in enumerate(res + [ok, known]):
            out[k][sl] = r
    return tuple(out)


_COVER_MARGIN = 0.1    # cells: a piece counts as inside another part only this far in


def _trilinear(S: _Pad, q: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a padded cell-centred field at index points ``q`` (n, 3)."""
    v, t = S.corners(q)
    w = torch.where(S.offs[None].bool(), t[:, None, :], 1 - t[:, None, :]).prod(2)    # (n, 8)
    return (w * v.to(torch.float32)).sum(1)


def _covered(T: torch.Tensor, piece_layer: torch.Tensor, sdfs: list[_Pad]) -> tuple[torch.Tensor, torch.Tensor]:
    """(all four sample points, any sample point) of each piece strictly inside another layer.

    ``sdfs``: the per-layer narrow-band SDFs in cells (padded with +band). A point
    is inside layer B when B's own SDF, trilinearly interpolated, is below
    -_COVER_MARGIN: sub-cell accurate, unlike any test on the cell-centre mask.
    """
    M = T.shape[0]
    all_c = torch.empty(M, dtype=torch.bool, device=T.device)
    any_c = torch.empty_like(all_c)
    for s0 in range(0, M, _GATHER_CHUNK):
        sl = slice(s0, s0 + _GATHER_CHUNK)
        pts, _, _ = _sample_points(T[sl])
        q = pts.reshape(-1, 3)
        pl = piece_layer[sl].repeat(4)
        cov = torch.zeros(q.shape[0], dtype=torch.bool, device=T.device)
        for b, S in enumerate(sdfs):
            cov |= (_trilinear(S, q) < -_COVER_MARGIN) & (pl != b)
        cov = cov.reshape(4, -1)
        all_c[sl] = cov.all(0)
        any_c[sl] = cov.any(0)
    return all_c, any_c


def _classify(T: torch.Tensor, parent: torch.Tensor, W: torch.Tensor, pin: _Pad, cross: list[_Pad] | None,
              face_layer: torch.Tensor | None, open_face: torch.Tensor, stats: dict
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop the triangle pieces buried in the solid and flag the open sheets.

    Returns the kept pieces as (pieces, parents, weights, sheet flags). Two
    burial tests. Across layers (parts that overlap: a wheel pushed into the
    body, a CAD union of intersecting shells) a piece is buried when all four
    of its sample points lie inside another layer's solid, by that layer's own
    SDF (:func:`_covered`). Within one layer (internal walls) it is buried when
    the inside mask is solid all around both sides (:func:`_probe_sides`).
    Pieces that straddle the boundary -- partly covered, or neither clearly
    buried nor clearly on the boundary (no inside cell around a probe 2 cells
    out; the 8 centres reach back at most sqrt(3) cells) -- are split to one
    cell and tested again, so only their buried part goes.

    A kept piece of an open component (``open_face`` of its parent) with no
    inside centre around either side is a *sheet* (module docstring) -- but only
    when every centre its probes read is known: next to a grid that cuts the
    surface, an unknown centre could be solid, and calling a boundary piece a
    sheet would thicken it by half a cell.
    """
    def test(Tp: torch.Tensor, par: torch.Tensor):
        all_p, any_p, all_m, any_m, ok, known = _probe_sides(Tp, pin)
        bur = all_p & all_m & ok
        sheet = ok & known & ~any_p & ~any_m & open_face[par]
        _, far_p, _, far_m, _, _ = _probe_sides(Tp, pin, probe=_PROBE_FAR)
        unsure = ok & ~bur & far_p & far_m
        partly = torch.zeros_like(bur)
        # the across-layer test only where the mask cannot tell (pieces clearly on the
        # boundary need none: that is most of any surface)
        idx = torch.nonzero(unsure).squeeze(1)
        if cross is not None and idx.numel():
            cov_all, cov_any = _covered(Tp[idx], face_layer[par[idx]], cross)
            bur[idx] = cov_all
            partly[idx] = cov_any & ~cov_all
            unsure = unsure & ~bur
        return bur, unsure, partly, sheet & ~bur

    buried, mixed, partly, sheet = test(T, parent)
    n_buried = int(buried.sum())
    if n_buried:
        keep = ~buried
        T, parent, W, mixed, partly, sheet = T[keep], parent[keep], W[keep], mixed[keep], partly[keep], sheet[keep]
    # refine what is undecided: to one cell (the resolution of the mask tests), then the
    # pieces still partly inside another part to a quarter cell (that test is sub-cell
    # accurate), so at most a quarter-cell sliver of buried surface survives at a crease.
    # Only the new pieces are tested again (partly-covered pieces are always "mixed").
    for size, use_partly in ((1.0, False), (0.25, True)):
        sel = partly if use_partly else mixed
        if not bool(sel.any()):
            continue
        Tm, pm, Wm = _subdivide(T[sel], size, W[sel])
        pm = parent[sel][pm]
        b2, m2, p2, s2 = test(Tm, pm)
        n_buried += int(b2.sum())
        old = ~sel
        n_old = int(old.sum())
        T = torch.cat([T[old], Tm[~b2]])
        parent = torch.cat([parent[old], pm[~b2]])
        W = torch.cat([W[old], Wm[~b2]])
        sheet = torch.cat([sheet[old], s2[~b2]])
        mixed = torch.cat([torch.zeros(n_old, dtype=torch.bool, device=T.device), m2[~b2]])
        partly = torch.cat([torch.zeros(n_old, dtype=torch.bool, device=T.device), p2[~b2]])
    stats["buried_triangles"] = stats.get("buried_triangles", 0) + n_buried
    return T, parent, W, sheet


def _near(T: torch.Tensor, g: Grid, band_cells: float) -> torch.Tensor:
    """(M,) bool: pieces whose bounding box comes within ``band_cells`` of a cell centre of ``g``."""
    top = torch.tensor([g.nx - 1, g.ny - 1, g.nz - 1], dtype=T.dtype, device=T.device)
    return ((T.amin(1) <= top + band_cells) & (T.amax(1) >= -band_cells)).all(1)


def _near_pieces(TI: np.ndarray, g: Grid, band_cells: float, max_edge: float, dev: torch.device
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The triangle pieces that can reach a cell centre of ``g`` within ``band_cells``.

    Triangles (index coordinates, (M, 3, 3)) are split to at most ``max_edge``
    cells (:func:`_subdivide`), so a slanted 100-cell face does not enumerate a
    million-cell box. The split depends on the triangle alone, never on where
    the grid ends, so a grid that cuts the surface sees exactly the pieces -- and
    makes exactly the burial and sheet decisions -- of a grid that holds all of
    it (:func:`_cut_padding`). Returns (pieces float32, parent rows of TI,
    barycentric weights).
    """
    if len(TI) == 0:
        e = torch.zeros(0, dtype=torch.int64, device=dev)
        return torch.zeros((0, 3, 3), device=dev), e, torch.zeros((0, 3, 3), device=dev)
    T0 = torch.as_tensor(TI, dtype=torch.float32, device=dev)
    pre = torch.nonzero(_near(T0, g, band_cells)).squeeze(1)
    T, parent, W = _subdivide(T0[pre], max_edge)
    parent = pre[parent]
    keep = _near(T, g, band_cells)
    return T[keep], parent[keep], W[keep]


def _pairs(T: torch.Tensor, g: Grid, band_cells: float, dev: torch.device, budget_mb: float, stats: dict):
    """Every (piece, cell) pair closer than ``band_cells``, in chunks bounded by ``budget_mb``.

    Yields (piece rows of ``T``, flat cell indices, squared distances in cells²
    (float32)). Each piece enumerates the cells of its bounding box grown by the
    band (clipped to the grid).
    """
    bc = float(band_cells)
    lo = torch.ceil(T.amin(1) - bc).to(torch.int64)
    hi = torch.floor(T.amax(1) + bc).to(torch.int64)
    lim = torch.tensor([g.nx - 1, g.ny - 1, g.nz - 1], device=dev)
    lo = torch.maximum(lo, torch.zeros_like(lo))
    hi = torch.minimum(hi, lim)
    w = (hi - lo + 1).clamp_(min=0)
    cnt = w[:, 0] * w[:, 1] * w[:, 2]
    sel = torch.nonzero(cnt > 0).squeeze(1)
    if sel.numel() == 0:
        return
    T, lo, w, cnt = T[sel], lo[sel], w[sel], cnt[sel]
    max_pairs = max(int(budget_mb * 2**20 / _BYTES_PER_PAIR), 65536)
    bc2 = bc * bc
    nxy = g.nx * g.ny
    for s0, s1 in _chunks(cnt, max_pairs):
        c = cnt[s0:s1]
        tot = int(c.sum())
        stats["band_pairs"] = stats.get("band_pairs", 0) + tot
        tri = torch.repeat_interleave(torch.arange(s0, s1, device=dev), c)
        start = torch.cumsum(c, 0) - c
        local = torch.arange(tot, device=dev) - torch.repeat_interleave(start, c)
        wx, wy = w[tri, 0], w[tri, 1]
        i = lo[tri, 0] + local % wx
        r = local // wx
        j = lo[tri, 1] + r % wy
        k = lo[tri, 2] + r // wy
        del local, r, wx, wy
        p = torch.stack([i, j, k], 1).to(torch.float32)
        Tt = T[tri]
        d2 = _tri_d2(p, Tt[:, 0], Tt[:, 1], Tt[:, 2])
        del p, Tt
        near = torch.nonzero(d2 < bc2).squeeze(1)
        if near.numel():
            yield sel[tri[near]], k[near] * nxy + j[near] * g.nx + i[near], d2[near]
        del d2, tri, i, j, k


def _cells_near(T: torch.Tensor, parent: torch.Tensor, W: torch.Tensor, g: Grid, band_cells: float,
                dev: torch.device, budget_mb: float, stats: dict, *, detail: bool = False):
    """Cells of ``g`` within ``band_cells`` of any piece and their nearest piece.

    Each (cell, piece) pair's exact point-triangle distance is packed with the
    piece id into one int64 key whose bit pattern orders like the distance, and a
    ``scatter_reduce(amin)`` keeps the nearest piece per cell. Returns (flat cell
    indices, parent triangle rows, barycentric weights (n, 3, 3) of the nearest
    piece within its parent, see :func:`_subdivide`); with ``detail`` also the
    nearest piece's row of ``T`` and its float32 distance [cells] per cell.
    """
    keys = torch.full((g.cells,), _KEY_EMPTY, dtype=torch.int64, device=dev)
    stats["band_triangles"] = stats.get("band_triangles", 0) + int(T.shape[0])
    stats.setdefault("band_pairs", 0)
    for tri, flat, d2 in _pairs(T, g, band_cells, dev, budget_mb, stats):
        keys.scatter_reduce_(0, flat, (d2.view(torch.int32).to(torch.int64) << 32) | tri, reduce="amin")
    cells = torch.nonzero(keys != _KEY_EMPTY).squeeze(1)
    k = keys[cells]
    piece = k & 0xFFFFFFFF
    if not detail:
        return cells, parent[piece], W[piece]
    dmin = torch.sqrt((k >> 32).to(torch.int32).view(torch.float32))
    return cells, parent[piece], W[piece], piece, dmin


#: float32 slack (cells) within which a second piece is a candidate exact tie of a cell's nearest
_TIE_SLACK = 1e-3


def _tie_candidates(T: torch.Tensor, g: Grid, band_cells: float, dev: torch.device, budget_mb: float,
                    cells: torch.Tensor, piece: torch.Tensor, dmin: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """(row into ``cells``, piece row of ``T``) of every piece other than a band cell's nearest
    whose float32 distance is within ``_TIE_SLACK`` cells of it: candidates for an exact tie,
    which the float64 recompute confirms or rejects (see :func:`_sdf`)."""
    row_of = torch.full((g.cells,), -1, dtype=torch.int64, device=dev)
    row_of[cells] = torch.arange(cells.numel(), device=dev)
    rows, pcs = [], []
    for tri, flat, d2 in _pairs(T, g, band_cells, dev, budget_mb, {}):
        row = row_of[flat]
        ok = row >= 0
        rr = row.clamp(min=0)
        cand = ok & (tri != piece[rr]) & (torch.sqrt(d2) <= dmin[rr] + _TIE_SLACK)
        rows.append(row[cand])
        pcs.append(tri[cand])
    if not rows:
        e = torch.zeros(0, dtype=torch.int64, device=dev)
        return e, e
    return torch.cat(rows), torch.cat(pcs)


def _band_search(TI: np.ndarray, g: Grid, band_cells: float, dev: torch.device, budget_mb: float,
                 stats: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cells within ``band_cells`` of any triangle of ``TI`` and their nearest piece, with no
    burial or sheet classification (the per-layer SDFs): see :func:`_cells_near`."""
    T, parent, W = _near_pieces(TI, g, band_cells, max(4.0, 2.0 * band_cells), dev)
    return _cells_near(T, parent, W, g, band_cells, dev, budget_mb, stats)


class _Prepared:
    """Everything derived from the input once (arrays, device, topology), for any grid."""

    def __init__(self, meshes, device) -> None:
        self.V, self.F, self.M, self.items = _gather(meshes)
        self.dev = _device(device, self.items)
        self.topo = _Topo(self.V, self.F, self.M)
        self.Fk = self.F[self.topo.keep]
        self.tensors = [v for v, _ in self.items if isinstance(v, torch.Tensor)]
        self.open_face = torch.as_tensor(self.topo.open_face, dtype=torch.bool, device=self.dev)

    def ti(self, g: Grid) -> np.ndarray:
        """Kept triangles' corners in ``g``'s index coordinates, (M, 3, 3) float64."""
        return _index_coords(self.V, g)[self.Fk]

    def vertices(self, dtype: torch.dtype) -> torch.Tensor:
        """All vertices on the device in ``dtype``, keeping the autograd link of tensor inputs."""
        vt = []
        for v, _ in self.items:
            if isinstance(v, torch.Tensor):
                vt.append(v.to(device=self.dev, dtype=dtype).reshape(-1, 3))
            else:
                vt.append(torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(-1, 3), device=self.dev,
                                          dtype=dtype))
        return torch.cat(vt) if vt else torch.zeros((0, 3), device=self.dev, dtype=dtype)


_LAYER_BAND = 1.5      # cells: half-width of the per-layer SDFs (only their sign near 0 is used)


def _layer_sdfs(TI: np.ndarray, layers: list, g: Grid, dev: torch.device, budget_mb: float, stats: dict,
                off=(0, 0, 0)) -> tuple[list[_Pad], torch.Tensor]:
    """Each layer's own narrow-band SDF in cells (float16, +-_LAYER_BAND outside the band),
    with that layer's parity as the sign, and the layer index of every kept face.

    ``TI`` and ``layers`` belong to ``g`` (the classification field); ``off`` is passed
    to the returned :class:`_Pad` objects, which are queried in the output grid's index
    coordinates."""
    face_layer = np.full(len(TI), -1, dtype=np.int64)
    sdfs = []
    nxy = g.nx * g.ny
    for li, (fsel, mask) in enumerate(layers):
        face_layer[fsel] = li
        sub: dict = {}
        cells, tri, _ = _band_search(TI[fsel], g, _LAYER_BAND, dev, budget_mb, sub)
        flat = mask.reshape(-1)
        S = torch.where(flat, -_LAYER_BAND, _LAYER_BAND).to(torch.float16)
        if cells.numel():
            k, r = cells // nxy, cells % nxy
            p = torch.stack([r % g.nx, r // g.nx, k], 1).to(torch.float32)
            Tt = torch.as_tensor(TI[fsel], dtype=torch.float32, device=dev)[tri]
            d = torch.sqrt(_tri_d2(p, Tt[:, 0], Tt[:, 1], Tt[:, 2]))
            S[cells] = torch.where(flat[cells], -d, d).to(torch.float16)
        sdfs.append(_Pad(S.reshape(g.shape), _LAYER_BAND, off))
        stats["layer_band_pairs"] = stats.get("layer_band_pairs", 0) + sub.get("band_pairs", 0)
    return sdfs, torch.as_tensor(face_layer, device=dev)


def _cut_padding(T: torch.Tensor, TI: np.ndarray, g: Grid) -> tuple[int, int, int, int, int, int]:
    """Cells to grow ``g`` by -- (x-, x+, y-, y+, z-, z+) -- so every centre the classification
    probes of the pieces ``T`` read is in the field, where the surface could be solid.

    A grid that cuts the surface (a half domain at a symmetry plane, a road, a solver
    window) holds only part of the body; read outside it, solid looks like empty
    space, and boundary pieces near the cut would be taken for sheets (thickened by
    half a cell) or kept although buried. Beyond the surface's bounding box nothing
    is solid, so the field never needs to reach past it: a grid that holds the whole
    surface is not grown at all.
    """
    if T.shape[0] == 0 or len(TI) == 0:
        return (0, 0, 0, 0, 0, 0)
    qlo = (T.amin((0, 1)) - _PROBE_REACH).double().cpu().numpy()
    qhi = (T.amax((0, 1)) + _PROBE_REACH).double().cpu().numpy()
    slo, shi = TI.min((0, 1)), TI.max((0, 1))
    n = (g.nx, g.ny, g.nz)
    out = []
    for a in range(3):
        need_lo = math.ceil(max(qlo[a], slo[a]))
        need_hi = math.floor(min(qhi[a], shi[a]))
        out += [max(0, -need_lo), max(0, need_hi - (n[a] - 1))]
    return tuple(int(v) for v in out)  # type: ignore[return-value]


def _sdf(prep: _Prepared, g: Grid, band: float, inside: torch.Tensor | None, sheet_thickness: float | None,
         dtype: torch.dtype, budget_mb: float, stats: dict, Vt: torch.Tensor | None = None,
         layers: list | None = None) -> torch.Tensor:
    """Narrow-band signed distance on ``g`` (module docstring). ``inside``: the inside mask
    on ``g`` to take the sign from (None: computed here); ``layers``: the per-layer masks
    of that inside test, when the caller has them."""
    dev = prep.dev
    TI = prep.ti(g)
    bc = band / g.h
    t_sheet = g.h if sheet_thickness is None else float(sheet_thickness)
    sheets = t_sheet > 0 and bool(prep.topo.open_face.any())
    reach = bc + (0.5 * t_sheet / g.h if sheets else 0.0)
    grad_on = torch.is_grad_enabled() and any(v.requires_grad for v in prep.tensors)
    with torch.no_grad():
        T, parent, W = _near_pieces(TI, g, reach, max(4.0, 2.0 * bc), dev)
        # the field the classification reads: the inside test of g itself, or of g grown
        # where it cuts the surface (_cut_padding); its g part is g's own inside test
        pad = _cut_padding(T, TI, g)
        off = (pad[0], pad[2], pad[4])
        if any(pad):
            gc = Grid(g.nx + pad[0] + pad[1], g.ny + pad[2] + pad[3], g.nz + pad[4] + pad[5], g.h,
                      g.x0 - pad[0] * g.h, g.y0 - pad[2] * g.h, g.z0 - pad[4] * g.h)
            layers_c: list = []
            inside_c = _inside(prep.V, prep.F, prep.topo, gc, dev, budget_mb,
                               stats if inside is None else {}, layer_out=layers_c)
            if inside is None:
                inside = inside_c[pad[4]:pad[4] + g.nz, pad[2]:pad[2] + g.ny, pad[0]:pad[0] + g.nx]
            stats["cut_padding"] = pad
        else:
            gc = g
            if inside is None:
                layers = []
                inside = _inside(prep.V, prep.F, prep.topo, g, dev, budget_mb, stats, layer_out=layers)
            elif layers is None and prep.topo.n_units > 1:
                layers = []
                _inside(prep.V, prep.F, prep.topo, g, dev, budget_mb, {}, layer_out=layers)
            inside_c, layers_c = inside, (layers or [])
        TIc = TI + np.asarray(off, dtype=np.float64)
        if len(TIc):
            pin = _Pad(inside_c, False, off, lo=tuple(TIc.min((0, 1))), hi=tuple(TIc.max((0, 1))))
        else:
            pin = _Pad(inside_c, False, off)
        cross = face_layer = None
        if len(layers_c) > 1:
            cross, face_layer = _layer_sdfs(TIc, layers_c, gc, dev, budget_mb, stats, off)
        stats["buried_triangles"] = stats.get("buried_triangles", 0)
        if T.shape[0]:
            T, parent, W, sheet = _classify(T, parent, W, pin, cross, face_layer, prep.open_face, stats)
        else:
            sheet = torch.zeros(0, dtype=torch.bool, device=dev)
        del cross, pin
        tie_rows = tie_tri = tie_w = None
        if grad_on:
            cells, tri, wts, piece, dmin = _cells_near(T, parent, W, g, bc, dev, budget_mb, stats, detail=True)
            rows, pcs = _tie_candidates(T, g, bc, dev, budget_mb, cells, piece, dmin)
            if rows.numel():
                tie_rows, tie_tri, tie_w = rows, parent[pcs], W[pcs]
            del piece, dmin
        else:
            cells, tri, wts = _cells_near(T, parent, W, g, bc, dev, budget_mb, stats)
        sheet_cells = sheet_tri = sheet_w = None
        stats["sheet_faces"] = 0
        if sheets and bool(sheet.any()):
            stats["sheet_faces"] = int(torch.unique(parent[sheet]).numel())
            sheet_cells, sheet_tri, sheet_w = _cells_near(T[sheet], parent[sheet], W[sheet], g,
                                                          bc + 0.5 * t_sheet / g.h, dev, budget_mb, stats)

    # differentiable recompute from the selected pieces' own (parent) vertices
    if Vt is None:
        Vt = prep.vertices(dtype)
    Fk_t = torch.as_tensor(prep.Fk, device=dev)
    origin = torch.tensor([g.x0, g.y0, g.z0], device=dev, dtype=dtype)
    nxy = g.nx * g.ny
    tiny = torch.finfo(dtype).tiny
    lat_tol = 1e-9 if dtype == torch.float64 else 1e-5      # rounding of the working precision, cells

    def geom(cells_: torch.Tensor, tri_: torch.Tensor, w_: torch.Tensor):
        k = cells_ // nxy
        r = cells_ % nxy
        j, i = r // g.nx, r % g.nx
        p = origin + (torch.stack([i, j, k], 1).to(dtype) + 0.5) * g.h
        f = Fk_t[tri_]
        # index_select: its backward is an index_add, far faster than advanced indexing's
        # when many cells share a few vertices
        P = torch.stack([torch.index_select(Vt, 0, f[:, c]) for c in range(3)], 1)
        # the nearest *piece* (buried parts of its triangle may have been dropped)
        Q = torch.bmm(w_.to(dtype), P)
        d2, interior, dpl, nn = _tri_core(p, Q[:, 0], Q[:, 1], Q[:, 2])
        dp = dpl / torch.sqrt(nn.clamp_min(tiny))
        # "in the plane": the nearest point is the foot of the perpendicular -- inside the
        # piece, or on its boundary up to rounding (a centre on a piece edge can fail the
        # inclusive inside test by 1e-18, and the edge formula's direction is then noise)
        with torch.no_grad():
            lat2 = (d2 - dp * dp).detach()
            in_plane = interior | (lat2 <= (lat_tol * g.h) ** 2 + 1e-12 * d2.detach())
        return torch.sqrt(d2.clamp_min(tiny)), in_plane, dp, Q

    flat_in = inside.reshape(-1)
    out = torch.where(flat_in, torch.tensor(-band, dtype=dtype, device=dev),
                      torch.tensor(band, dtype=dtype, device=dev))
    zone = _PLANE_ZONE * g.h
    with torch.set_grad_enabled(grad_on):
        if cells.numel():
            d, interior, dp, Qn = geom(cells, tri, wts)
            sig = torch.where(flat_in[cells], -1.0, 1.0).to(dtype)
            # Through the surface: within _PLANE_ZONE cells of a face the distance is the face's
            # signed plane distance, the sign of the side being the face's outward side. Each
            # face's outward side is the vote of the band cells clearly off it (the parity sign
            # times the side of its plane they are on), so a cell exactly on a face, or within
            # the inside test's ray jitter of it, gets the derivative and the sign of the true
            # surface instead of a zero derivative (sqrt at 0) or the other side's sign.
            with torch.no_grad():
                dpd = dp.detach()
                vote = interior & (dpd.abs() > zone)
                votes = torch.zeros(len(prep.Fk), dtype=dtype, device=dev)
                votes.index_add_(0, tri[vote], (sig * torch.sign(dpd))[vote])
                face_side = torch.sign(votes)

            def signed(d_, interior_, dp_, sig_, tri_):
                side = face_side[tri_]
                on = interior_ & (dp_.detach().abs() <= zone) & (side != 0)
                v = sig_ * d_.clamp(max=band)
                return torch.where(on, side * dp_, v) if bool(on.any()) else v

            val = signed(d, interior, dp, sig, tri)
            if tie_rows is not None:
                val = _tie_average(val, d, Qn, tri, tie_rows, tie_tri, geom(cells[tie_rows], tie_tri, tie_w),
                                   lambda dd, ii, pp, rr, tt: signed(dd, ii, pp, sig[rr], tt), face_side, g.h,
                                   dtype, stats)
            out = out.index_put((cells,), val)
        if sheet_cells is not None and sheet_cells.numel():
            ds = (geom(sheet_cells, sheet_tri, sheet_w)[0] - 0.5 * t_sheet).clamp(min=-band, max=band)
            out = out.index_put((sheet_cells,), torch.minimum(out[sheet_cells], ds))
    stats["band_cells"] = int(cells.numel())
    return out.reshape(g.shape)


def _plane_keys(Q: torch.Tensor, side: torch.Tensor, h: float) -> torch.Tensor:
    """(n, 4) int64 key of each piece's plane: outward unit normal and offset, quantised to
    1e-6 (normal) and 1e-6 h (offset), so coplanar pieces -- of one face, or the two
    triangles of a box side -- share a key."""
    n = torch.linalg.cross(Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0])
    n = n / n.norm(dim=1, keepdim=True).clamp_min(torch.finfo(n.dtype).tiny)
    n = n * torch.where(side == 0, 1.0, side).to(n.dtype)[:, None]
    off = (n * Q[:, 0]).sum(1) / h
    return torch.round(torch.cat([n, off[:, None]], 1) * 1e6).to(torch.int64)


def _tie_average(val, d, Qn, tri, rows, tie_tri, tie_geom, signed, face_side, h, dtype, stats):
    """Average the signed distance of a cell over the surfaces it is exactly as far from.

    A cell as far from two different faces -- on the medial plane between opposite
    faces of a box whose offsets in the grid agree, on a box edge -- sits on a kink of
    the distance, where the derivative depends on which face is credited. Tied
    candidates (:func:`_tie_candidates`) confirmed in the working precision are grouped
    by plane (pieces of one face, or coplanar faces, have one derivative there and count
    once), and the cell takes the mean over the planes: for a two-way tie that is
    exactly the derivative central differences measure; where three or more faces meet
    no single derivative can match them in every direction, and the mean is the
    symmetric choice. The values being equal, only the derivative changes. "Exactly"
    means to the rounding of the working precision (1e-11 h in float64): a near tie a
    finite difference can resolve keeps the nearest face's derivative.
    """
    dt, it, dpt, Qt = tie_geom
    with torch.no_grad():
        tol = (1e-11 if dtype == torch.float64 else 1e-5) * h
        tie = (dt.detach() - d.detach()[rows]).abs() <= tol
    if not bool(tie.any()):
        return val
    r = rows[tie]
    vt = signed(dt[tie], it[tie], dpt[tie], r, tie_tri[tie])
    with torch.no_grad():
        uniq = torch.unique(r)
        local = torch.searchsorted(uniq, r)
        own_rows = uniq
        keys = torch.cat([_plane_keys(Qn[own_rows].detach(), face_side[tri[own_rows]], h),
                          _plane_keys(Qt[tie].detach(), face_side[tie_tri[tie]], h)])
        ent_row = torch.cat([torch.arange(uniq.numel(), device=val.device), local])
        grp, ginv = torch.unique(torch.cat([ent_row[:, None], keys], 1), dim=0, return_inverse=True)
        n_grp = grp.shape[0]
        gcnt = torch.zeros(n_grp, dtype=dtype, device=val.device).index_add_(
            0, ginv, torch.ones(ginv.numel(), dtype=dtype, device=val.device))
        grow = grp[:, 0]
        rcnt = torch.zeros(uniq.numel(), dtype=dtype, device=val.device).index_add_(
            0, grow, torch.ones(n_grp, dtype=dtype, device=val.device))
        weight = 1.0 / (gcnt[ginv] * rcnt[ent_row])           # 1 / (members of the group x groups of the row)
    ent_val = torch.cat([val[own_rows], vt])
    new = torch.zeros(uniq.numel(), dtype=ent_val.dtype, device=val.device).index_add(0, ent_row,
                                                                                      ent_val * weight)
    stats["tied_cells"] = stats.get("tied_cells", 0) + int(uniq.numel())
    return val.index_put((own_rows,), new)


def sdf_band(meshes_t, grid, band: float, *, inside: torch.Tensor | None = None,
             sheet_thickness: float | None = None, device: str | torch.device | None = None,
             dtype: torch.dtype | None = None, budget_mb: float = 384.0, info: dict | None = None
             ) -> torch.Tensor:
    """Narrow-band signed distance (nz, ny, nx), negative inside, in the grid's length unit.

    Cells within ``band`` of the surface get the exact distance to the nearest
    triangle with the sign of the robust inside test; all other cells get
    ``-band`` (inside) or ``+band`` (outside). Differentiable with respect to
    vertex tensors that require grad (``meshes_t`` items as (vertices, faces) with
    a torch vertex tensor), also through the surface itself (module docstring).
    ``inside``: a precomputed :func:`inside_mask` to reuse (it must be for the
    same geometry and grid). ``sheet_thickness``: minimum thickness given to open
    sheets (pieces of open components with no inside on either side; ``None`` =
    one cell ``h``, ``0`` = off). The grid may cut the surface (a half domain at
    a symmetry plane, a road, a window of a larger domain): the result inside it
    is the same as on a grid holding the whole surface.
    ``dtype``: output precision (default: the vertex tensors' dtype, else float32).
    """
    t0 = time.perf_counter()
    g = _as_grid(grid)
    if not band > 0:
        raise ValueError("band must be positive")
    prep = _Prepared(meshes_t, device)
    if dtype is None:
        tt = prep.tensors
        dtype = tt[0].dtype if tt and tt[0].is_floating_point() else torch.float32
    stats: dict = {}
    if inside is not None:
        inside = inside.to(device=prep.dev, dtype=torch.bool).reshape(g.shape)
    out = _sdf(prep, g, band, inside, sheet_thickness, dtype, budget_mb, stats)
    if info is not None:
        info.update(stats)
        info.update(triangles=len(prep.topo.keep), removed_faces=prep.topo.removed, device=str(prep.dev),
                    seconds=time.perf_counter() - t0)
    return out


def sdf_bands(meshes_t, grids, band: float, *, sheet_thickness: float | None = None,
              device: str | torch.device | None = None, dtype: torch.dtype | None = None,
              budget_mb: float = 384.0, info: dict | None = None) -> list[torch.Tensor]:
    """:func:`sdf_band` on several grids, sharing the mesh preparation (topology, units).

    Meant for a staggered (MAC) solver, which needs the solid fraction at the
    cell centres and at the three face families::

        grids = [g, g.faces("x"), g.faces("y"), g.faces("z")]
        sd_c, sd_u, sd_v, sd_w = sdf_bands(meshes, grids, band)

    Each grid gets its own inside test (parity is evaluated at that grid's
    points); gradients flow to vertex tensors as for :func:`sdf_band`. A ramp of
    half-width ``w`` cells centred ``s`` cells inside the surface (the 2D
    solver's mask width and wall shift) needs ``band >= (w + |s|) h``.
    """
    t0 = time.perf_counter()
    if not band > 0:
        raise ValueError("band must be positive")
    gs = [_as_grid(gr) for gr in grids]
    prep = _Prepared(meshes_t, device)
    if dtype is None:
        tt = prep.tensors
        dtype = tt[0].dtype if tt and tt[0].is_floating_point() else torch.float32
    Vt = prep.vertices(dtype)
    stats: dict = {}
    out = [_sdf(prep, g, band, None, sheet_thickness, dtype, budget_mb, stats, Vt=Vt) for g in gs]
    if info is not None:
        info.update(stats)
        info.update(triangles=len(prep.topo.keep), removed_faces=prep.topo.removed, device=str(prep.dev),
                    seconds=time.perf_counter() - t0)
    return out


def solid_fraction(sdf: torch.Tensor, h: float, width: float = 1.0) -> torch.Tensor:
    """Smooth solid fraction from a signed distance: the 2D C1 ramp, unchanged.

    1 for ``sdf <= -width h``, 0 for ``sdf >= width h``, 0.5 on the surface; the
    integral of a hat function (a quadratic B-spline step), so a grid-aligned
    plane gets the exact volume at any offset and the field is smooth enough to
    differentiate. Identical to :func:`aerolume.geometry.sdf.volume_fraction`.
    """
    from ..geometry.sdf import volume_fraction
    return volume_fraction(sdf, h, width)


def voxelize(meshes, grid, *, width: float = 1.0, smooth: bool = True, sheet_thickness: float | None = None,
             device: str | torch.device | None = None, dtype: torch.dtype = torch.float32,
             budget_mb: float = 384.0, info: dict | None = None) -> torch.Tensor:
    """Solid fraction (nz, ny, nx) in [0, 1] of the union of the meshes on ``grid``.

    The robust inside test plus the narrow-band distance through the C1 ramp of
    half-width ``width`` cells (:func:`solid_fraction`); ``smooth=False`` returns
    the bare 0/1 inside test. No autograd (use :func:`sdf_band` for gradients).
    ``meshes``: Mesh3D, (vertices, faces) or a list; any orientation, holes allowed.
    """
    t0 = time.perf_counter()
    g = _as_grid(grid)
    stats: dict = {}
    with torch.no_grad():
        prep = _Prepared(meshes, device)
        if smooth:
            sdf = _sdf(prep, g, width * g.h, None, sheet_thickness, dtype, budget_mb, stats)
            out = solid_fraction(sdf, g.h, width)
        else:
            out = _inside(prep.V, prep.F, prep.topo, g, prep.dev, budget_mb, stats).to(dtype)
    if info is not None:
        info.update(stats)
        info.update(triangles=len(prep.topo.keep), removed_faces=prep.topo.removed, device=str(prep.dev),
                    seconds=time.perf_counter() - t0)
    return out
