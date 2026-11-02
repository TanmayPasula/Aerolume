"""Operations on triangle meshes (numpy; scipy for neighbour searches and graphs).

The frame is DESIGN.md section 7.2: x downstream, y span, z up, road at z = 0.
Everything here works on :class:`~aerolume.model.Mesh3D` and returns new meshes
(inputs are never modified), so the Geometry page can preview a transform and
throw it away.

Rotations follow the right-hand rule about each axis. ``transform(...,
rotate_deg=(rx, ry, rz))`` applies rx about x first, then ry about y, then rz
about z (``R = Rz @ Ry @ Rx``), all about the point ``about``. With that sign a
positive ry lifts a nose pointing to -x, which is exactly the solver's pitch
(+alpha about +y, section 7.2), and a positive rz turns a nose at -x towards -y.

What "inside" means is decided elsewhere (the voxeliser, :mod:`.voxel`, which
does not need consistent winding). Winding matters for normals, the signed
volume and the outward direction of sensitivity maps; :func:`orient_outward`
makes it consistent where the topology allows.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..model import Mesh3D
from ..solvers.fpmode import ieee_subnormals
from ._rows import unique_rows

__all__ = [
    "ORIENT_PRESETS",
    "area",
    "auto_orient",
    "bbox",
    "boundary_edges",
    "centre_y",
    "closest_points_on_triangles",
    "compact",
    "components",
    "decimate",
    "edge_counts",
    "face_areas",
    "face_normals",
    "frontal_area",
    "inconsistent_edges",
    "is_watertight",
    "merge",
    "mesh_info",
    "mirror",
    "normals",
    "orient_outward",
    "orient_preset",
    "place_nose",
    "place_on_road",
    "projection_mask",
    "remove_degenerate",
    "rotation_matrix",
    "sample_surface",
    "symmetry_error",
    "transform",
    "translate",
    "vertex_normals",
    "volume",
    "weld",
    "wheel_params",
]

_AXES = {"x": 0, "y": 1, "z": 2}


def _list(meshes: Mesh3D | Sequence[Mesh3D]) -> list[Mesh3D]:
    return [meshes] if isinstance(meshes, Mesh3D) else list(meshes)


def _same_kind(src: Mesh3D | Sequence[Mesh3D], out: list[Mesh3D]):
    return out[0] if isinstance(src, Mesh3D) else out


def _copy(m: Mesh3D, vertices: np.ndarray | None = None, faces: np.ndarray | None = None, **kw) -> Mesh3D:
    """A new Mesh3D with the same metadata (name, role, omega, centre, axis)."""
    d = {"name": m.name, "role": m.role, "omega": m.omega, "center": m.center, "axis": m.axis}
    d.update(kw)
    return Mesh3D(vertices=m.vertices.copy() if vertices is None else vertices,
                  faces=m.faces.copy() if faces is None else faces, **d)


# ---------------------------------------------------------------------------
# basic measures
# ---------------------------------------------------------------------------

def bbox(meshes: Mesh3D | Sequence[Mesh3D]) -> tuple[float, float, float, float, float, float]:
    """(xmin, xmax, ymin, ymax, zmin, zmax) over all vertices used by faces, metres."""
    ms = [m for m in _list(meshes) if len(m.faces)]
    if not ms:
        raise ValueError("no triangles")
    v = np.concatenate([m.vertices[np.unique(m.faces)] for m in ms])
    lo, hi = v.min(0), v.max(0)
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


def _lohi(meshes) -> tuple[np.ndarray, np.ndarray]:
    b = bbox(meshes)
    return np.array(b[0::2]), np.array(b[1::2])


def face_normals(mesh: Mesh3D, unit: bool = True) -> np.ndarray:
    """Per-face normals (M, 3) from the stored winding (right-hand rule).

    ``unit=False`` returns the raw cross products, whose length is twice the
    face area (handy for area-weighted sums). Degenerate faces get zero.
    """
    t = mesh.vertices[mesh.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    if not unit:
        return n
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)


def vertex_normals(mesh: Mesh3D) -> np.ndarray:
    """Area-weighted average of the incident face normals at each vertex, unit length."""
    n = face_normals(mesh, unit=False)
    acc = np.zeros_like(mesh.vertices)
    for k in range(3):
        np.add.at(acc, mesh.faces[:, k], n)
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    return np.divide(acc, ln, out=np.zeros_like(acc), where=ln > 0)


def normals(mesh: Mesh3D, per: str = "face") -> np.ndarray:
    """Unit normals per ``"face"`` (M, 3) or per ``"vertex"`` (N, 3)."""
    if per == "face":
        return face_normals(mesh)
    if per == "vertex":
        return vertex_normals(mesh)
    raise ValueError("per must be 'face' or 'vertex'")


def face_areas(mesh: Mesh3D) -> np.ndarray:
    """Area of every triangle, m²."""
    return 0.5 * np.linalg.norm(face_normals(mesh, unit=False), axis=1)


def area(meshes: Mesh3D | Sequence[Mesh3D]) -> float:
    """Total surface area, m²."""
    return float(sum(face_areas(m).sum() for m in _list(meshes)))


def volume(meshes: Mesh3D | Sequence[Mesh3D]) -> float:
    """Signed enclosed volume by the divergence theorem, m³.

    Positive for a closed mesh with outward (counter-clockwise seen from
    outside) winding, negative when the winding is inward. Meaningless for an
    open mesh (it then depends on where the origin is); see :func:`is_watertight`.
    """
    tot = 0.0
    for m in _list(meshes):
        t = m.vertices[m.faces]
        c = t.mean(axis=(0, 1)) if len(t) else np.zeros(3)   # shift: better rounding
        a, b, d = t[:, 0] - c, t[:, 1] - c, t[:, 2] - c
        tot += float(np.einsum("ij,ij->i", a, np.cross(b, d)).sum()) / 6.0
    return tot


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------

def _edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Directed edges (3M, 2) in face order and their undirected keys (sorted pairs)."""
    e = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1).reshape(-1, 2)
    return e, np.sort(e, axis=1)


def edge_counts(mesh: Mesh3D) -> tuple[np.ndarray, np.ndarray]:
    """Unique undirected edges (K, 2) and how many faces use each (K,)."""
    _, und = _edges(mesh.faces)
    if len(und) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    uniq, _, _, cnt = unique_rows(und)
    return uniq, cnt


def boundary_edges(mesh: Mesh3D) -> np.ndarray:
    """Edges used by exactly one face (K, 2): the rims of holes and open sheets."""
    e, c = edge_counts(mesh)
    return e[c == 1]


def is_watertight(mesh: Mesh3D) -> bool:
    """True when every edge is shared by exactly two faces (closed, manifold)."""
    _, c = edge_counts(mesh)
    return bool(len(c) and np.all(c == 2))


def inconsistent_edges(mesh: Mesh3D) -> int:
    """Manifold edges whose two faces run along them in the same direction.

    In a consistently wound surface neighbours traverse a shared edge in
    opposite directions; each same-direction edge marks a flipped face (or a
    non-orientable surface).
    """
    e, und = _edges(mesh.faces)
    if len(e) == 0:
        return 0
    uniq, _, inv, cnt = unique_rows(und)
    fwd = (e[:, 0] < e[:, 1]).astype(np.int64)
    s = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(s, inv, fwd)
    two = cnt == 2
    return int(np.sum(two & (s != 1)))


def components(mesh: Mesh3D) -> tuple[int, np.ndarray]:
    """Connected pieces of the surface: (count, label per face).

    Faces are connected through shared *edges*, so two parts that only touch at
    a vertex (overlapping primitives often share one) are separate pieces --
    the same rule the voxeliser uses to keep overlapping shells apart.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nf = len(mesh.faces)
    if nf == 0:
        return 0, np.zeros(0, dtype=np.int64)
    _, und = _edges(mesh.faces)
    _, _, einv, ecnt = unique_rows(und)
    ne = len(ecnt)
    rows = np.repeat(np.arange(nf), 3)
    cols = nf + einv
    g = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nf + ne, nf + ne))
    _, lab = connected_components(g, directed=False)
    flab = lab[:nf]
    uniq, inv = np.unique(flab, return_inverse=True)
    return len(uniq), inv.reshape(-1).astype(np.int64)


def mesh_info(mesh: Mesh3D) -> dict:
    """What the Geometry page shows: counts, watertightness, holes, winding, size.

    ``holes`` counts the closed loops of boundary edges (an open sheet has one);
    ``inconsistent_edges`` counts flipped neighbours (see :func:`inconsistent_edges`).
    """
    e, c = edge_counts(mesh)
    bnd = e[c == 1]
    holes = 0
    if len(bnd):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        nv = len(mesh.vertices)
        g = coo_matrix((np.ones(len(bnd)), (bnd[:, 0], bnd[:, 1])), shape=(nv, nv))
        _, lab = connected_components(g, directed=False)
        holes = len(np.unique(lab[np.unique(bnd)]))
    t = mesh.vertices[mesh.faces]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    a = 0.5 * np.linalg.norm(n, axis=1)
    ncomp, _ = components(mesh)
    size = float(np.ptp(mesh.vertices, axis=0).max()) if len(mesh.vertices) else 0.0
    watertight = bool(len(c) and np.all(c == 2))
    return {
        "triangles": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "watertight": watertight,
        "boundary_edges": len(bnd),
        "holes": int(holes),
        "nonmanifold_edges": int(np.sum(c > 2)),
        "inconsistent_edges": inconsistent_edges(mesh),
        "degenerate_faces": int(np.sum(a <= 1e-14 * size * size)),
        "components": int(ncomp),
        "area": float(a.sum()),
        "volume": volume(mesh) if watertight else float("nan"),
        "bounds": mesh.bounds() if len(mesh.vertices) else (0.0,) * 6,
    }


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def compact(mesh: Mesh3D) -> Mesh3D:
    """Drop vertices no face uses (indices renumbered)."""
    if len(mesh.faces) == 0:
        return _copy(mesh, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    used, inv = np.unique(mesh.faces.reshape(-1), return_inverse=True)
    return _copy(mesh, mesh.vertices[used], inv.reshape(-1, 3))


def weld(mesh: Mesh3D, tol: float | None = None) -> Mesh3D:
    """Merge vertices closer than ``tol`` (default 1e-6 of the bounding-box diagonal).

    Clusters are the connected groups of the "closer than tol" relation (so a
    chain of near points merges into one), each replaced by its mean. Faces that
    collapse are removed. ``tol=0`` merges only bit-identical coordinates.
    """
    from .meshio import _weld_exact

    v, f = mesh.vertices, mesh.faces
    if len(v) == 0:
        return _copy(mesh)
    if tol is None:
        tol = 1e-6 * float(np.linalg.norm(np.ptp(v, axis=0)))
    if tol <= 0:
        v2, f2 = _weld_exact(v, f)
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        with ieee_subnormals():                 # k-d trees crash under flush-to-zero (fpmode)
            pairs = cKDTree(v).query_pairs(tol, output_type="ndarray")
        if len(pairs) == 0:
            return remove_degenerate(_copy(mesh), duplicates=False)
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(v), len(v)))
        _, lab = connected_components(g, directed=False)
        cnt = np.bincount(lab).astype(np.float64)
        v2 = np.stack([np.bincount(lab, weights=v[:, k]) for k in range(3)], axis=1) / cnt[:, None]
        f2 = lab[f]
    return remove_degenerate(_copy(mesh, v2, f2), duplicates=False)


def remove_degenerate(mesh: Mesh3D, *, area_tol: float | None = None, duplicates: bool = True) -> Mesh3D:
    """Remove faces with repeated corners or (near) zero area, and duplicate faces.

    ``area_tol`` [m²] defaults to 1e-14 times the squared bounding-box size.
    ``duplicates``: also drop faces that use the same three vertices as an
    earlier one (in any order). Coincident duplicates make crossing counts
    even where they should be odd, and a double-sided sheet stored as two
    opposite faces is still the same sheet. Unused vertices are dropped.
    """
    f = mesh.faces
    if len(f) == 0:
        return compact(mesh)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    a = face_areas(mesh)
    if area_tol is None:
        size = float(np.ptp(mesh.vertices[np.unique(f)], axis=0).max()) if len(f) else 0.0
        area_tol = 1e-14 * size * size
    keep &= a > area_tol
    if duplicates:
        key = np.sort(f, axis=1)
        _, first, _, _ = unique_rows(key)
        dup = np.ones(len(f), dtype=bool)
        dup[first] = False
        keep &= ~dup
    return compact(_copy(mesh, mesh.vertices, f[keep]))


def orient_outward(mesh: Mesh3D) -> Mesh3D:
    """Consistent winding with outward normals, per connected piece where possible.

    Faces are flipped to agree with their neighbours across manifold edges
    (propagated over a spanning tree of the face-adjacency graph with pointer
    jumping, so it is vectorised), then every piece whose signed volume is
    negative is turned inside out. A non-orientable piece (a Moebius strip, or
    faces joined only through non-manifold edges) keeps whatever the tree gives.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    f = mesh.faces.copy()
    nf = len(f)
    if nf == 0:
        return _copy(mesh)
    e, und = _edges(f)
    face_of = np.repeat(np.arange(nf), 3)
    _, _, inv, cnt = unique_rows(und)
    order = np.argsort(inv, kind="stable")
    sinv = inv[order]
    # the two half-edges of a manifold edge are consecutive after sorting by edge
    first = np.flatnonzero((cnt[sinv] == 2) & np.r_[True, sinv[1:] != sinv[:-1]])
    ha, hb = order[first], order[first + 1]
    fa, fb = face_of[ha], face_of[hb]
    same = e[ha, 0] == e[hb, 0]          # same direction along the edge: one of the two is flipped
    pair = unique_rows(np.sort(np.stack([fa, fb], 1), 1))[1]
    fa, fb, same = fa[pair], fb[pair], same[pair]
    g = coo_matrix((np.ones(len(fa)), (fa, fb)), shape=(nf, nf)).tocsr()
    g = g + g.T
    ncomp, lab = connected_components(g, directed=False)
    roots = np.unique(lab, return_index=True)[1]
    # one BFS over all pieces through a virtual root (node nf) joined to one face of each
    rr = np.concatenate([fa, fb, np.full(len(roots), nf)])
    cc = np.concatenate([fb, fa, roots])
    G = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nf + 1, nf + 1)).tocsr()
    _, pred = breadth_first_order(G, nf, directed=False, return_predecessors=True)
    pred = pred[:nf].astype(np.int64)
    # relative flip of each face against its BFS parent (stored +1 so zeros are "no edge")
    Rm = coo_matrix((same.astype(np.int8) + 1, (fa, fb)), shape=(nf, nf)).tocsr()
    Rm = Rm + Rm.T
    acc = np.zeros(nf + 1, dtype=bool)
    tree = pred < nf
    idx = np.flatnonzero(tree)
    if len(idx):
        acc[idx] = np.asarray(Rm[idx, pred[idx]]).ravel() == 2
    ptr = np.concatenate([np.where(tree, pred, nf), [nf]])
    # pointer jumping: acc[i] becomes the XOR of the relative flips on the path to the root
    while np.any(ptr[:nf] != nf):
        acc = acc ^ acc[ptr]
        ptr = ptr[ptr]
    flip = acc[:nf]
    f[flip] = f[flip][:, [0, 2, 1]]
    # outward: the signed volume of each piece must be positive
    t = mesh.vertices[f]
    c0 = mesh.vertices.mean(0)
    vol6 = np.einsum("ij,ij->i", t[:, 0] - c0, np.cross(t[:, 1] - c0, t[:, 2] - c0))
    pv = np.bincount(lab, weights=vol6, minlength=ncomp)
    inside_out = pv[lab] < 0
    f[inside_out] = f[inside_out][:, [0, 2, 1]]
    return _copy(mesh, mesh.vertices, f)


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def rotation_matrix(rotate_deg: Sequence[float]) -> np.ndarray:
    """``Rz @ Ry @ Rx`` for right-hand rotations (rx, ry, rz) in degrees about x, y, z."""
    rx, ry, rz = (math.radians(float(a)) for a in rotate_deg)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    R[np.abs(R) < 1e-15] = 0.0          # exact zeros for multiples of 90 degrees
    return R


def transform(meshes: Mesh3D | Sequence[Mesh3D], scale: float | Sequence[float] = 1.0,
              rotate_deg: Sequence[float] = (0.0, 0.0, 0.0), about: Sequence[float] = (0.0, 0.0, 0.0),
              translate: Sequence[float] = (0.0, 0.0, 0.0), mirror_axis: str | None = None):
    """Mirror, scale and rotate about ``about``, then translate (mesh or list -> same).

    ``scale``: a number or (sx, sy, sz). ``rotate_deg``: (rx, ry, rz), applied
    about x, then y, then z, right-hand rule (module docstring). ``mirror_axis``:
    "x", "y" or "z" reflects across the plane through ``about`` normal to that
    axis. A reflection (mirror, or an odd number of negative scale factors)
    reverses the winding so outward normals stay outward. Wheel centres and axes
    move with their mesh.
    """
    o = np.asarray(about, dtype=float)
    S = np.ones(3) * np.asarray(scale, dtype=float) if np.ndim(scale) else np.full(3, float(scale))
    if np.any(S == 0):
        raise ValueError("scale factors must be non-zero")
    M = np.diag(S)
    if mirror_axis is not None:
        if mirror_axis not in _AXES:
            raise ValueError("mirror_axis must be 'x', 'y' or 'z'")
        Mi = np.eye(3)
        Mi[_AXES[mirror_axis], _AXES[mirror_axis]] = -1.0
        M = M @ Mi
    A = rotation_matrix(rotate_deg) @ M
    flip = np.linalg.det(A) < 0
    t = np.asarray(translate, dtype=float)
    out = []
    for m in _list(meshes):
        v = (m.vertices - o) @ A.T + o + t
        f = m.faces[:, [0, 2, 1]] if flip else m.faces.copy()
        ctr = None if m.center is None else tuple((np.asarray(m.center) - o) @ A.T + o + t)
        ax = None
        if m.axis is not None:
            a = np.asarray(m.axis, dtype=float) @ A.T
            n = np.linalg.norm(a)
            # an axis is a pseudo-vector: under a reflection its sense flips with the winding
            ax = tuple((-a if flip else a) / n) if n > 0 else m.axis
        out.append(_copy(m, v, f, center=ctr, axis=ax))
    return _same_kind(meshes, out)


def translate(meshes: Mesh3D | Sequence[Mesh3D], offset: Sequence[float]):
    """Shift by ``offset`` (metres)."""
    return transform(meshes, translate=offset)


def mirror(meshes: Mesh3D | Sequence[Mesh3D], axis: str = "y", about: float = 0.0):
    """Reflect across the plane ``axis = about`` (winding reversed, normals stay outward)."""
    o = np.zeros(3)
    o[_AXES[axis]] = about
    return transform(meshes, about=o, mirror_axis=axis)


def merge(meshes: Sequence[Mesh3D], name: str | None = None) -> Mesh3D:
    """One mesh holding all triangles (role and wheel data of the first)."""
    ms = list(meshes)
    if not ms:
        raise ValueError("nothing to merge")
    vs, fs, off = [], [], 0
    for m in ms:
        vs.append(m.vertices)
        fs.append(m.faces + off)
        off += len(m.vertices)
    return _copy(ms[0], np.concatenate(vs), np.concatenate(fs), name=name or ms[0].name)


#: orientation presets as ``rotate_deg`` for :func:`transform` (applied about the origin)
ORIENT_PRESETS: dict[str, tuple[float, float, float]] = {
    "none": (0.0, 0.0, 0.0),
    "y-up->z-up": (90.0, 0.0, 0.0),        # most OBJ / glTF / game exports: y is up
    "z-up->y-up": (-90.0, 0.0, 0.0),
    "x-up->z-up": (0.0, -90.0, 0.0),
    "upside-down": (180.0, 0.0, 0.0),
    "nose+x->nose-x": (0.0, 0.0, 180.0),   # model faces downstream: turn it to face the flow
    "nose+y->nose-x": (0.0, 0.0, 90.0),
    "nose-y->nose-x": (0.0, 0.0, -90.0),
}


def orient_preset(meshes: Mesh3D | Sequence[Mesh3D], name: str):
    """Apply one of :data:`ORIENT_PRESETS` (rotation about the origin)."""
    if name not in ORIENT_PRESETS:
        raise ValueError(f"unknown orientation preset {name!r} (have: {', '.join(ORIENT_PRESETS)})")
    return transform(meshes, rotate_deg=ORIENT_PRESETS[name])


def place_on_road(meshes: Mesh3D | Sequence[Mesh3D], clearance: float = 0.0):
    """Shift vertically so the lowest point is ``clearance`` above the road (z = 0)."""
    lo, _ = _lohi(meshes)
    return transform(meshes, translate=(0.0, 0.0, clearance - lo[2]))


def centre_y(meshes: Mesh3D | Sequence[Mesh3D]):
    """Shift sideways so the bounding box is centred on y = 0 (the symmetry plane)."""
    lo, hi = _lohi(meshes)
    return transform(meshes, translate=(0.0, -0.5 * (lo[1] + hi[1]), 0.0))


def place_nose(meshes: Mesh3D | Sequence[Mesh3D], x: float = 0.0):
    """Shift along the flow so the most upstream point is at ``x`` (vehicles: x = 0)."""
    lo, _ = _lohi(meshes)
    return transform(meshes, translate=(x - lo[0], 0.0, 0.0))


# ---------------------------------------------------------------------------
# surface sampling + distances
# ---------------------------------------------------------------------------

def sample_surface(meshes: Mesh3D | Sequence[Mesh3D], n: int, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` area-weighted points on the surface: (points (n,3), mesh index, face index).

    Deterministic for a given seed (systematic sampling on the cumulative area,
    random barycentric coordinates), so measurements built on it are repeatable.
    """
    ms = _list(meshes)
    tris = [m.vertices[m.faces] for m in ms]
    a = np.concatenate([0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
                        for t in tris]) if tris else np.zeros(0)
    if a.sum() <= 0 or n <= 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    mid = np.concatenate([np.full(len(t), i) for i, t in enumerate(tris)])
    fid = np.concatenate([np.arange(len(t)) for t in tris])
    allt = np.concatenate(tris)
    rng = np.random.default_rng(seed)
    cum = np.cumsum(a)
    u = (np.arange(n) + rng.random()) / n * cum[-1]
    k = np.minimum(np.searchsorted(cum, u, side="right"), len(a) - 1)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    t = allt[k]
    pts = w0[:, None] * t[:, 0] + w1[:, None] * t[:, 1] + w2[:, None] * t[:, 2]
    return pts, mid[k], fid[k]


def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on triangle (a, b, c) to p, row-wise (Ericson, *Real-Time Collision
    Detection*, 5.1.5); all arrays (K, 3). Exact for degenerate triangles too."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    out = np.empty_like(p)
    done = np.zeros(len(p), dtype=bool)

    def put(mask, val):
        m = mask & ~done
        out[m] = val[m] if np.ndim(val) == 2 else val
        done[:] |= m

    with np.errstate(divide="ignore", invalid="ignore"):
        put((d1 <= 0) & (d2 <= 0), a)
        put((d3 >= 0) & (d4 <= d3), b)
        put((d6 >= 0) & (d5 <= d6), c)
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w[:, None] * (c - b))
        den = va + vb + vc
        v = vb / den
        w = vc / den
        inner = a + v[:, None] * ab + w[:, None] * ac
        put(np.isfinite(inner).all(1), inner)
    if not done.all():          # fully degenerate leftovers (all three corners equal)
        out[~done] = a[~done]
    return out


def _surface_distance(points: np.ndarray, meshes: list[Mesh3D], k: int = 8, n_samples: int = 40000
                      ) -> np.ndarray:
    """Exact distance from points to the surface of ``meshes``.

    Two passes. First an upper bound ``u``: the exact distance to the faces of
    the ``k`` nearest reference points (every face centroid plus area-weighted
    samples, so large faces are found from anywhere near them). Then every face
    that could still be nearer: a face with centroid ``c`` and radius ``rho``
    (farthest corner from ``c``) can only be within ``u`` of ``p`` if
    ``|p - c| <= u + rho``. Faces are grouped into radius classes (powers of
    two) with one k-d tree each, so a few large faces do not widen the search
    for the many small ones. The result is the exact point-triangle distance
    (zero to rounding for a point on the surface), whatever the tessellation or
    how close other parts are.
    """
    from scipy.spatial import cKDTree

    with ieee_subnormals():                 # k-d trees crash under flush-to-zero (fpmode)
        tris = np.concatenate([m.vertices[m.faces] for m in meshes])
        cents = tris.mean(1)
        sp, smi, sfi = sample_surface(meshes, n_samples, seed=1)
        offs = np.cumsum([0] + [len(m.faces) for m in meshes])[:-1]
        ref = np.concatenate([cents, sp])
        ref_face = np.concatenate([np.arange(len(tris)), offs[smi] + sfi])
        tree = cKDTree(ref)
        kk = min(k, len(ref))
        _, nn = tree.query(points, k=kk)
        nn = nn.reshape(len(points), kk)
        best = np.full(len(points), np.inf)
        for j in range(kk):
            f = ref_face[nn[:, j]]
            q = closest_points_on_triangles(points, tris[f, 0], tris[f, 1], tris[f, 2])
            np.minimum(best, np.linalg.norm(points - q, axis=1), out=best)
        # exact pass over every face that could beat the bound
        rho = np.linalg.norm(tris - cents[:, None, :], axis=2).max(1)
        rmin = max(float(rho.max()) * 1e-9, float(rho[rho > 0].min()) if np.any(rho > 0) else 1e-300)
        cls = np.floor(np.log2(np.maximum(rho, rmin) / rmin)).astype(np.int64)
        for c in np.unique(cls):
            fid = np.flatnonzero(cls == c)
            R = float(rho[fid].max())
            lists = cKDTree(cents[fid]).query_ball_point(points, r=best * (1 + 1e-12) + R)
            lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(points))
            if lens.sum() == 0:
                continue
            pi = np.repeat(np.arange(len(points)), lens)
            fi = fid[np.concatenate([np.asarray(x, dtype=np.int64) for x in lists if len(x)])]
            q = closest_points_on_triangles(points[pi], tris[fi, 0], tris[fi, 1], tris[fi, 2])
            np.minimum.at(best, pi, np.linalg.norm(points[pi] - q, axis=1))
    return best


def symmetry_error(meshes: Mesh3D | Sequence[Mesh3D], plane_y: float | None = 0.0, n_samples: int = 20000
                   ) -> float:
    """RMS distance between the surface mirrored across ``y = plane_y`` and the surface, m.

    Area-weighted surface samples are reflected and their exact distance to the
    original surface is measured (zero to rounding for a mirror-symmetric mesh,
    whatever its tessellation). An appendage on one side only (a mirror, an
    exhaust) shows up as the RMS of how far its reflection is from anything.
    ``plane_y=None`` uses the bounding-box centre.
    """
    ms = [m for m in _list(meshes) if len(m.faces)]
    if not ms:
        return 0.0
    if plane_y is None:
        lo, hi = _lohi(ms)
        plane_y = 0.5 * (lo[1] + hi[1])
    pts, _, _ = sample_surface(ms, n_samples, seed=0)
    pts[:, 1] = 2.0 * plane_y - pts[:, 1]
    d = _surface_distance(pts, ms)
    return float(np.sqrt(np.mean(d * d)))


# ---------------------------------------------------------------------------
# frontal area
# ---------------------------------------------------------------------------

def frontal_area(meshes: Mesh3D | Sequence[Mesh3D], axis: str = "x", resolution: int = 1024,
                 *, budget: int = 4_000_000) -> float:
    """Projected area along ``axis`` (the union of all projected triangles), m².

    The triangles are rasterised onto a ``resolution``-pixel grid across the
    larger extent of the projection with a pixel-centre coverage test, so
    overlapping parts (wheels behind a bumper, the far side of the body) count
    once and holes in the mesh hardly matter. Pixel-centre sampling is unbiased
    on curved outlines (about 0.1 % at 1024 pixels for a sphere); an edge that
    happens to run along the pixel grid can be off by up to half a pixel (0.4 %
    for a 10:1 rectangle at 1024 pixels), so pass a larger ``resolution`` for
    slender, boxy shapes. ``axis="x"`` gives the frontal area used as the
    vehicle reference.
    """
    mask, px = projection_mask(meshes, axis, resolution, budget=budget)
    return float(mask.sum()) * px * px


def projection_mask(meshes: Mesh3D | Sequence[Mesh3D], axis: str = "x", resolution: int = 1024, *,
                    budget: int = 4_000_000) -> tuple[np.ndarray, float]:
    """``(mask, pixel size)``: the rasterised union of the triangles projected along ``axis``.

    ``mask`` is (rows, cols) bool over the two remaining axes in order (for x:
    rows along z, columns along y), with a one-pixel margin around the
    projection's bounding box. Each triangle is filled span by span: for every
    pixel row its exact x-interval at the row centre, accumulated in a
    difference array, so the cost is the number of (triangle, row) pairs rather
    than pixels; a pixel is covered when its centre lies in a triangle (closed
    test: shared edges leave no cracks).
    """
    ms = [m for m in _list(meshes) if len(m.faces)]
    if not ms:
        return np.zeros((0, 0), dtype=bool), 0.0
    a = _AXES[axis]
    pa, pb = [k for k in range(3) if k != a]
    tri = np.concatenate([m.vertices[m.faces][:, :, [pa, pb]] for m in ms])
    lo = tri.reshape(-1, 2).min(0)
    ext = tri.reshape(-1, 2).max(0) - lo
    if not ext.max() > 0:
        return np.zeros((0, 0), dtype=bool), 0.0
    px = float(ext.max()) / int(resolution)
    nu = math.ceil(ext[0] / px) + 2
    nv = math.ceil(ext[1] / px) + 2
    q = (tri - lo) / px - 0.5 + 1.0          # pixel centres at integers, one pixel margin
    u, v = q[..., 0], q[..., 1]
    area2 = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])
    ok = np.abs(area2) > 1e-12
    u, v = u[ok], v[ok]
    j0 = np.clip(np.ceil(v.min(1)), 0, nv - 1).astype(np.int64)
    j1 = np.clip(np.floor(v.max(1)), 0, nv - 1).astype(np.int64)
    rows = np.maximum(j1 - j0 + 1, 0)
    W = nu + 1
    diff = np.zeros(nv * W, dtype=np.int64)
    csum = np.cumsum(rows)
    start = 0
    while start < len(rows):
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + budget, side="right")), start + 1)
        c = rows[start:stop]
        tot = int(c.sum())
        if tot:
            t = np.repeat(np.arange(start, stop), c)
            y = (j0[t] + np.arange(tot) - np.repeat(np.cumsum(c) - c, c)).astype(np.float64)
            xl = np.full(tot, np.inf)
            xr = np.full(tot, -np.inf)
            for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                ya, yb, xa, xb = v[t, e0], v[t, e1], u[t, e0], u[t, e1]
                dy = yb - ya
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (y - ya) / dy
                hit = (dy != 0) & (s >= 0) & (s <= 1)     # horizontal edges: their ends come from the others
                x = xa + s * (xb - xa)
                np.minimum(xl, np.where(hit, x, np.inf), out=xl)
                np.maximum(xr, np.where(hit, x, -np.inf), out=xr)
            ok_ = np.isfinite(xl) & np.isfinite(xr)
            i0 = np.clip(np.ceil(xl[ok_]), 0, nu - 1).astype(np.int64)
            i1 = np.clip(np.floor(xr[ok_]), -1, nu - 1).astype(np.int64)
            yy = y[ok_].astype(np.int64)
            span = i0 <= i1
            row0 = yy[span] * W
            diff += np.bincount(row0 + i0[span], minlength=nv * W)
            diff -= np.bincount(row0 + i1[span] + 1, minlength=nv * W)
        start = stop
    cover = np.cumsum(diff.reshape(nv, W), axis=1)[:, :nu] > 0
    return cover, px


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def _flat_area(meshes: list[Mesh3D], axis: int) -> tuple[float, float]:
    """Area of near-horizontal faces (|n_axis| > 0.9) in the lowest / highest third."""
    lo, hi = _lohi(meshes)
    span = hi[axis] - lo[axis]
    low = high = 0.0
    for m in meshes:
        n = face_normals(m, unit=False)
        a = 0.5 * np.linalg.norm(n, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            flat = np.abs(n[:, axis]) > 0.9 * 2.0 * a
        c = m.vertices[m.faces][:, :, axis].mean(1)
        low += float(a[flat & (c < lo[axis] + span / 3)].sum())
        high += float(a[flat & (c > hi[axis] - span / 3)].sum())
    return low, high


def _mirror_rms(meshes: list[Mesh3D], axis: int, plane: float, n: int = 6000) -> float:
    pts, _, _ = sample_surface(meshes, n, seed=0)
    pts[:, axis] = 2.0 * plane - pts[:, axis]
    d = _surface_distance(pts, meshes, n_samples=12000)
    return float(np.sqrt(np.mean(d * d)))


def auto_orient(meshes: Mesh3D | Sequence[Mesh3D], notes: list[str] | None = None):
    """Rotate by multiples of 90 degrees so the longest dimension runs along x and the
    flat side faces down (z up); returns the same kind (mesh or list).

    Of the two remaining axes the one the shape is mirror-symmetric across
    becomes the span (y): vehicles, riders, drones and aircraft are symmetric
    left-right but never top-bottom, which decides the up axis even for a truck
    that is taller than it is wide. Then the end of the up axis with more flat,
    horizontal area (a car's floor, a box's base) goes down. Undecidable cases
    keep the file's own choice. The sense along x is kept: use the
    "nose+x->nose-x" preset if the model faces downstream. A short description
    of what was done is appended to ``notes``.
    """
    ms = [m for m in _list(meshes) if len(m.faces)]
    if not ms:
        return meshes
    lo, hi = _lohi(ms)
    ext = hi - lo
    names = "xyz"
    i_long = int(np.argmax(ext))
    if np.sum(ext >= ext[i_long] * (1 - 1e-9)) > 1 and ext[0] >= ext[i_long] * (1 - 1e-9):
        i_long = 0                                       # ties: keep x
    rest = [k for k in range(3) if k != i_long]
    centre = 0.5 * (lo + hi)
    errs = [_mirror_rms(ms, k, centre[k]) for k in rest]
    tol = 0.005 * float(ext.max())
    if (errs[0] <= tol and errs[1] <= tol) or (min(errs) > 0 and max(errs) / min(errs) < 1.25):
        up = 2 if 2 in rest else 1                      # undecided: keep the file's up axis
        why = "symmetric both ways: kept the file's up axis"
    else:
        lat = rest[int(np.argmin(errs))]
        up = rest[0] if lat == rest[1] else rest[1]
        why = f"mirror-symmetric across file {names[lat]}"
    low, high = _flat_area(ms, up)
    s_up = 1.0
    if high > 1.3 * max(low, 1e-300):
        s_up = -1.0
    ex = np.zeros(3)
    ex[i_long] = 1.0
    ez = np.zeros(3)
    ez[up] = s_up
    ey = np.cross(ez, ex)
    R = np.stack([ex, ey, ez])           # rows: new axes in file coordinates (proper rotation)
    if np.allclose(R, np.eye(3)):
        if notes is not None:
            notes.append("Orientation kept: the longest dimension is already along x with z up.")
        return meshes
    out = []
    for m in _list(meshes):
        v = m.vertices @ R.T
        ctr = None if m.center is None else tuple(np.asarray(m.center) @ R.T)
        ax = None if m.axis is None else tuple(np.asarray(m.axis) @ R.T)
        out.append(_copy(m, v, m.faces.copy(), center=ctr, axis=ax))
    if notes is not None:
        sgn = "" if s_up > 0 else "-"
        notes.append(f"Re-oriented: file {names[i_long]} (longest) is now x, file {sgn}{names[up]} is up "
                     f"({why}; flat side down).")
    return _same_kind(meshes, out)


# ---------------------------------------------------------------------------
# wheels + decimation
# ---------------------------------------------------------------------------

def wheel_params(mesh: Mesh3D) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """(centre, axis, radius) of a wheel-like mesh.

    The axle is the principal direction with the smallest spread of the surface
    (a tyre is a short, wide cylinder), from the exact area second moment of the
    triangles (``int x x^T dA = A/12 (sum v_i v_i^T + (sum v_i)(sum v_i)^T)`` per
    triangle), oriented to the positive side of its dominant component and
    snapped to a coordinate axis when within 1 degree of it (wheels in a model
    are axis-aligned; tessellation noise should not tilt them). The centre is
    the bounding-box centre in the axle frame; the radius half of the larger
    extent across the axle.
    """
    t = mesh.vertices[mesh.faces]
    a = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    if len(t) == 0 or a.sum() <= 0:
        raise ValueError("empty wheel mesh")
    c = (a[:, None] * t.mean(1)).sum(0) / a.sum()
    tt = t - c
    s = tt.sum(1)
    cov = (np.einsum("m,mia,mib->ab", a, tt, tt) + np.einsum("m,ma,mb->ab", a, s, s)) / 12.0
    _, vecs = np.linalg.eigh(cov)
    ax = vecs[:, 0]
    k = int(np.argmax(np.abs(ax)))
    if ax[k] < 0:
        ax = -ax
    if ax[k] > math.cos(math.radians(1.0)):
        ax = np.eye(3)[k]
    ax = ax / np.linalg.norm(ax)
    # frame: axle + two orthonormal directions across it
    e1 = np.cross(ax, [1.0, 0.0, 0.0] if abs(ax[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    v = mesh.vertices[np.unique(mesh.faces)]
    P = np.stack([v @ ax, v @ e1, v @ e2], axis=1)
    mid = 0.5 * (P.min(0) + P.max(0))
    centre = mid[0] * ax + mid[1] * e1 + mid[2] * e2
    r = 0.5 * float(max(np.ptp(P[:, 1]), np.ptp(P[:, 2])))
    return tuple(float(x) for x in centre), tuple(float(x) for x in ax), r


def decimate(mesh: Mesh3D, target_faces: int, *, iterations: int = 12) -> Mesh3D:
    """Reduce to about ``target_faces`` triangles by vertex clustering.

    Vertices are snapped to a uniform grid and every cell's vertices replaced
    by their mean (Rossignac-Borrel clustering); the cell size is searched so the
    face count lands near the target. Fast and robust on any mesh, meant for
    previews and quick looks, not for the solver (which never needs it: the
    voxeliser costs little per triangle). Small features below the cell size
    are lost.
    """
    nf = len(mesh.faces)
    if target_faces <= 0:
        raise ValueError("target_faces must be positive")
    if nf <= target_faces:
        return _copy(mesh)
    v = mesh.vertices
    lo = v.min(0)
    diag = float(np.linalg.norm(np.ptp(v, axis=0)))
    a_lo, a_hi = 1e-6 * diag, diag
    best = None

    def run(cell: float) -> Mesh3D:
        key = np.floor((v - lo) / cell).astype(np.int64)
        lab = unique_rows(key)[2]
        cnt = np.bincount(lab).astype(np.float64)
        v2 = np.stack([np.bincount(lab, weights=v[:, k]) for k in range(3)], 1) / cnt[:, None]
        return remove_degenerate(_copy(mesh, v2, lab[mesh.faces]))

    for _ in range(iterations):
        cell = math.sqrt(a_lo * a_hi)
        m = run(cell)
        if len(m.faces) > target_faces:
            a_lo = cell
        else:
            a_hi = cell
            best = m
        if best is not None and len(best.faces) >= 0.85 * target_faces:
            break
    return best if best is not None else run(a_hi)
