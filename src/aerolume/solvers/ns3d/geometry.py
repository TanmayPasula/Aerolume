"""Immersed-mesh geometry: signed distance -> smooth solid masks -> force weights.

Bodies enter the 3D solver only through Brinkman volume penalisation, exactly as
in 2D (``aerolume.solvers.ns2d.geometry``), so the geometry is a set of fields on
the staggered grid:

* ``chi_u``, ``chi_v``, ``chi_w`` - solid fraction at the u-, v- and w-faces where
  the momentum lives, evaluated from the signed distance *at the face positions*
  (not averaged from cell centres), which keeps the effective wall where the
  distance puts it for every velocity component;
* ``a_u``, ``a_v``, ``a_w`` - ``sum_b chi_b u_solid_b``, the penalisation target
  pre-multiplied by the mask (rigid rotation of wheels costs nothing per step);
* ``k_u``, ``k_v``, ``k_w`` - per-body force / moment weights that turn the
  momentum removed by the penalisation into forces and moments (core.step3d);
* ``chi_c`` - solid fraction at cell centres (LES switch-off, display).

The mask is the compact C1 half-sine step of the signed distance shifted
``MASK_SHIFT h`` inwards (the 2D calibration: for walls aligned with the grid the
stiff-penalisation wall offset is a 1D property of the discretisation normal to
the wall, so it carries over unchanged; ``docs/theory_ns3d.md`` 6).

Signed distance
---------------
The shared narrow-band distance of :mod:`aerolume.geometry3d.voxel` is used when
that module is importable: fast (triangles binned to cells), and its ray-parity
voting is robust to holes, randomly flipped triangles and overlapping parts. It
is evaluated directly on the solver's window, which the domain often cuts (a
symmetry plane, the road): the voxeliser grows its own classification field past
such a cut, so the result equals that of a grid holding the whole body (until
it did, this bridge evaluated a covering grid, up to twice the work).
Otherwise a private exact implementation runs: distance to the nearest
*triangle* (chunked brute force searched without autograd, then recomputed
from the selected triangle with autograd, so the gradient w.r.t. the vertices
is exact with O(points) saved activations) and inside/outside from the
generalised winding number (|w| > 1/2; van Oosterom & Strackee solid angles),
which tolerates small holes and a globally flipped orientation but not
inconsistent winding. It is exact for clean meshes; its speed (points x
triangles) limits it to coarse meshes or small windows. Both agree to 1e-8 on
a sphere (tests). Both are differentiable through the surface: a point exactly
on a face (or, for the shared one, within its ray jitter) takes the face's
signed plane distance, whose derivative is the face's, instead of
``sqrt(d^2)`` at d = 0, which has none.

All fields are stored on a *window* of cells covering the bodies plus a margin,
so penalisation, force sums and distance evaluations cost in proportion to the
bodies, not the domain.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from .config import ImmersedMesh, _is_tensor

#: point-triangle pairs evaluated per chunk by the private distance (bounds memory)
PAIR_CHUNK = 1 << 20


# ---------------------------------------------------------------------------
# private exact signed distance to a triangle soup
# ---------------------------------------------------------------------------

def _tri_d2(px, py, pz, a, b, c):
    """Squared distance from points to triangles, broadcasting (P, 1) against (1, T) or (P,).

    ``a, b, c`` are tuples of coordinate tensors. Closest point: the projection on
    the plane when it falls inside the triangle, else the nearest edge point.
    Smooth (differentiable) almost everywhere; degenerate triangles reduce to
    their edges.
    """
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    tiny = torch.finfo(px.dtype).tiny
    # edges and normal
    e0 = (bx - ax, by - ay, bz - az)
    e1 = (cx - bx, cy - by, cz - bz)
    e2 = (ax - cx, ay - cy, az - cz)
    nx_ = e0[1] * (-e2[2]) - e0[2] * (-e2[1])
    ny_ = e0[2] * (-e2[0]) - e0[0] * (-e2[2])
    nz_ = e0[0] * (-e2[1]) - e0[1] * (-e2[0])
    nn = nx_ * nx_ + ny_ * ny_ + nz_ * nz_
    w0 = (px - ax, py - ay, pz - az)
    w1 = (px - bx, py - by, pz - bz)
    w2 = (px - cx, py - cy, pz - cz)

    def side(e, w):     # n . (e x w): >= 0 when w is on the inner side of edge e
        cxp = e[1] * w[2] - e[2] * w[1]
        cyp = e[2] * w[0] - e[0] * w[2]
        czp = e[0] * w[1] - e[1] * w[0]
        return nx_ * cxp + ny_ * cyp + nz_ * czp

    inside = (side(e0, w0) >= 0) & (side(e1, w1) >= 0) & (side(e2, w2) >= 0) & (nn > tiny)
    pd = nx_ * w0[0] + ny_ * w0[1] + nz_ * w0[2]
    plane = pd * pd / nn.clamp_min(tiny)

    def seg(e, w):
        ee = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).clamp_min(tiny)
        t = ((w[0] * e[0] + w[1] * e[1] + w[2] * e[2]) / ee).clamp(0.0, 1.0)
        dx, dy, dz = w[0] - t * e[0], w[1] - t * e[1], w[2] - t * e[2]
        return dx * dx + dy * dy + dz * dz

    edge = torch.minimum(torch.minimum(seg(e0, w0), seg(e1, w1)), seg(e2, w2))
    return torch.where(inside, plane, edge)


def _solid_angle(px, py, pz, a, b, c):
    """Signed solid angle of triangles seen from points (van Oosterom & Strackee 1983)."""
    ax, ay, az = a[0] - px, a[1] - py, a[2] - pz
    bx, by, bz = b[0] - px, b[1] - py, b[2] - pz
    cx, cy, cz = c[0] - px, c[1] - py, c[2] - pz
    la = torch.sqrt(ax * ax + ay * ay + az * az)
    lb = torch.sqrt(bx * bx + by * by + bz * bz)
    lc = torch.sqrt(cx * cx + cy * cy + cz * cz)
    det = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    den = la * lb * lc + (ax * bx + ay * by + az * bz) * lc + (bx * cx + by * cy + bz * cz) * la \
        + (cx * ax + cy * ay + cz * az) * lb
    return 2.0 * torch.atan2(det, den)


def mesh_signed_distance(vertices: torch.Tensor, faces: torch.Tensor, points: torch.Tensor, *,
                         chunk: int = PAIR_CHUNK, plane_zone: float = 0.0) -> torch.Tensor:
    """Exact signed distance from ``points`` (P, 3) to a triangle soup, negative inside.

    The nearest triangle is searched without autograd (chunked brute force), then
    the distance to it is recomputed with autograd, so gradients flow to the
    vertices (and points) that require grad. Inside means a generalised winding
    number of magnitude above 1/2. A point whose nearest point is the foot of its
    perpendicular on a face and lies within ``plane_zone`` of the face's plane
    (0: exactly on it) gets the face's signed plane distance, which is
    differentiable through zero; the face's outward side is the vote of the other
    points nearest to it (winding-number sign times the side they are on), or,
    when no other point votes, the mesh's orientation (the sign of its signed
    volume: this fallback assumes consistent winding anyway).
    """
    dev, dt = points.device, points.dtype
    v = vertices.to(device=dev, dtype=dt)
    f = torch.as_tensor(faces, device=dev, dtype=torch.long)
    nt, npt = f.shape[0], points.shape[0]
    if npt == 0:
        return points.new_zeros(0)
    vd = v.detach()
    tri = [vd[f[:, k]] for k in range(3)]                   # (T, 3) each
    best = torch.empty(npt, device=dev, dtype=torch.long)
    gwn = torch.zeros(npt, device=dev, dtype=dt)
    pdet = points.detach()
    tstep = max(1, min(nt, chunk // 64))
    pstep = max(1, chunk // tstep)
    with torch.no_grad():
        for s in range(0, npt, pstep):
            p = pdet[s:s + pstep]
            px, py, pz = p[:, 0:1], p[:, 1:2], p[:, 2:3]
            dmin = torch.full((p.shape[0],), math.inf, device=dev, dtype=dt)
            imin = torch.zeros(p.shape[0], device=dev, dtype=torch.long)
            wsum = torch.zeros(p.shape[0], device=dev, dtype=dt)
            for t0 in range(0, nt, tstep):
                a, b, c = (tuple(x[t0:t0 + tstep, k][None, :] for k in range(3)) for x in tri)
                d2 = _tri_d2(px, py, pz, a, b, c)
                m, i = d2.min(dim=1)
                better = m < dmin
                dmin = torch.where(better, m, dmin)
                imin = torch.where(better, i + t0, imin)
                wsum += _solid_angle(px, py, pz, a, b, c).sum(dim=1)
            best[s:s + pstep] = imin
            gwn[s:s + pstep] = wsum
    inside = (gwn.abs() > 2.0 * math.pi)                    # |w| = |sum Omega| / 4 pi > 1/2
    fb = f[best]                                            # (P, 3) nearest triangle, with autograd below
    a = tuple(v[fb[:, 0], k] for k in range(3))
    b = tuple(v[fb[:, 1], k] for k in range(3))
    c = tuple(v[fb[:, 2], k] for k in range(3))
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    d2 = _tri_d2(px, py, pz, a, b, c)
    d = torch.sqrt(d2.clamp_min(1e-24))
    sig = torch.where(inside, -1.0, 1.0).to(dt)
    # through the surface (see the docstring): the signed plane distance of the nearest face
    e0 = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    e1 = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0])
    nn = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).clamp_min(torch.finfo(dt).tiny)
    dp = (n[0] * (px - a[0]) + n[1] * (py - a[1]) + n[2] * (pz - a[2])) / torch.sqrt(nn)
    with torch.no_grad():
        dpd, d2d = dp.detach(), d2.detach()
        in_plane = (d2d - dpd * dpd) <= 1e-18 + 1e-12 * d2d
        vote = in_plane & (dpd.abs() > plane_zone) & (dpd != 0)
        votes = torch.zeros(nt, device=dev, dtype=dt).index_add_(0, best[vote], (sig * torch.sign(dpd))[vote])
        orient = torch.sign((tri[0] * torch.linalg.cross(tri[1], tri[2])).sum())     # +1: outward winding
        side = torch.where(votes != 0, torch.sign(votes), orient)[best]
        on = in_plane & (dpd.abs() <= plane_zone) & (side != 0)
    val = sig * d
    return torch.where(on, side * dp, val) if bool(on.any()) else val


# ---------------------------------------------------------------------------
# shared voxeliser bridge
# ---------------------------------------------------------------------------

_SHARED = None          # resolved lazily: the shared grid-SDF callable, or False
_SHARED_WARNED = False


def _warn_shared_failure(exc: BaseException) -> None:
    """Say once per process that the shared voxeliser failed and the (slower,
    winding-sensitive) private fallback is used instead."""
    global _SHARED_WARNED
    if not _SHARED_WARNED:
        _SHARED_WARNED = True
        warnings.warn(f"geometry3d.voxel.sdf_band failed ({type(exc).__name__}: {exc}); the 3D solver uses its "
                      "private exact signed distance instead (slower, needs consistently wound meshes)",
                      RuntimeWarning, stacklevel=3)


def _shared_grid_sdf():
    """``aerolume.geometry3d.voxel.sdf_band`` when importable and usable, else None."""
    global _SHARED
    if _SHARED is None:
        try:
            from aerolume.geometry3d.voxel import sdf_band as fn
            _SHARED = fn
        except Exception:  # noqa: BLE001 - module missing or broken while being written: use ours
            _SHARED = False
    return _SHARED or None


def grid_signed_distance(meshes: Sequence[tuple[torch.Tensor, torch.Tensor]],
                         grid: tuple[int, int, int, float, float, float, float], band: float, *,
                         use_shared: bool = True) -> torch.Tensor:
    """Signed distance (negative inside) at the cell centres of ``grid``, shape (nz, ny, nx).

    ``grid = (nx, ny, nz, h, x0, y0, z0)`` with centres at ``x0 + (i + 1/2) h``
    (the geometry3d convention). Values are exact within ``band`` of the
    surface; further away only their sign matters to the solver. ``meshes`` are
    (vertices (N, 3) tensor, faces (M, 3) long tensor) pairs of ONE body (their
    union). The shared narrow-band implementation is used when available. The
    grid may cut the body (a symmetry plane, the road): the values inside it are
    those of a grid holding the whole body.
    """
    nx, ny, nz, h, x0, y0, z0 = grid
    fn = _shared_grid_sdf() if use_shared else None
    if fn is not None:
        try:
            d = fn(list(meshes), grid, band)
            if tuple(d.shape) == (nz, ny, nx):
                return d
        except Exception as exc:  # noqa: BLE001 - an incompatible shared module must not stop the solver
            _warn_shared_failure(exc)
    v0 = meshes[0][0]
    dev, dt = v0.device, v0.dtype
    zs = z0 + (torch.arange(nz, device=dev, dtype=dt) + 0.5) * h
    ys = y0 + (torch.arange(ny, device=dev, dtype=dt) + 0.5) * h
    xs = x0 + (torch.arange(nx, device=dev, dtype=dt) + 0.5) * h
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
    d = None
    for verts, faces in meshes:
        dm = mesh_signed_distance(verts, faces, pts, plane_zone=0.02 * h)
        d = dm if d is None else torch.minimum(d, dm)
    return d.reshape(nz, ny, nx)


def smooth_step(d: torch.Tensor, w: float) -> torch.Tensor:
    """Compact C1 solid fraction of a signed distance (1 inside, 0 outside)."""
    s = (d / w).clamp(-1.0, 1.0)
    return 0.5 - 0.5 * torch.sin(0.5 * math.pi * s)


# ---------------------------------------------------------------------------
# geometry container
# ---------------------------------------------------------------------------

class Geometry3D(NamedTuple):
    """Masks and weights of all immersed meshes on the solver window.

    ``window = (k0, k1, j0, j1, i0, i1)`` is a block of *cells*. The u-face arrays
    cover layers k0..k1-1, rows j0..j1-1 and faces i0..i1 (shape (wz, wy, wx+1));
    v-face arrays faces j0..j1 (wz, wy+1, wx); w-face arrays faces k0..k1
    (wz+1, wy, wx). An empty window means "no bodies".

    Force weights ``k_u`` (nb, 3, ...) map u-increments to [Fx, My, Mz] parts,
    ``k_v`` v-increments to [Fy, Mx, Mz] parts and ``k_w`` w-increments to
    [Fz, Mx, My] parts (lever arms about the moment reference).
    """
    window: tuple[int, int, int, int, int, int]
    chi_u: torch.Tensor
    chi_v: torch.Tensor
    chi_w: torch.Tensor
    chi_c: torch.Tensor
    a_u: torch.Tensor
    a_v: torch.Tensor
    a_w: torch.Tensor
    k_u: torch.Tensor
    k_v: torch.Tensor
    k_w: torch.Tensor
    nbody: int

    @property
    def empty(self) -> bool:
        k0, k1, j0, j1, i0, i1 = self.window
        return k1 <= k0 or j1 <= j0 or i1 <= i0


def _vertices_numpy(b) -> np.ndarray:
    if isinstance(b, ImmersedMesh):
        return b.vertices_numpy()
    v = b[0] if isinstance(b, (tuple, list)) else b
    if _is_tensor(v):
        return v.detach().cpu().double().numpy()
    return np.asarray(v, dtype=np.float64).reshape(-1, 3)


def rotation_matrix(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """DESIGN 7.2 attitude: yaw about +z by -yaw, then pitch about +y by +pitch."""
    a, b = math.radians(pitch_deg), math.radians(-yaw_deg)
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    return ry @ rz


def body_window3d(vertex_sets: Sequence[np.ndarray], ops, margin_cells: float | None = None, *,
                  pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> tuple[int, int, int, int, int, int]:
    """Cell window covering all vertices plus a margin, clipped to the domain.

    With ``pitch_deg`` / ``yaw_deg`` the window also covers the bodies rotated
    about the moment reference by up to those angles (live attitude edits keep
    the window, and with it any captured CUDA graph).
    """
    sets = [np.asarray(v, dtype=np.float64).reshape(-1, 3) for v in vertex_sets if len(v)]
    if not sets:
        return (0, 0, 0, 0, 0, 0)
    pts = np.vstack(sets)
    if pitch_deg > 0 or yaw_deg > 0:
        ref = np.asarray(ops.moment_ref, dtype=np.float64)
        rot = []
        for pa in np.linspace(-pitch_deg, pitch_deg, 7 if pitch_deg > 0 else 1):
            for ya in np.linspace(-yaw_deg, yaw_deg, 7 if yaw_deg > 0 else 1):
                rot.append((pts - ref) @ rotation_matrix(pa, ya).T + ref)
        pts = np.vstack(rot)
    margin = (ops.mask_width + 3.0 + abs(ops.mask_shift)) if margin_cells is None else margin_cells
    h = ops.h
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    i0 = math.floor((lo[0] - ops.x0) / h - margin)
    i1 = math.ceil((hi[0] - ops.x0) / h + margin)
    j0 = math.floor((lo[1] - ops.y0) / h - margin)
    j1 = math.ceil((hi[1] - ops.y0) / h + margin)
    k0 = math.floor((lo[2] - ops.z0) / h - margin)
    k1 = math.ceil((hi[2] - ops.z0) / h + margin)
    i0, i1 = max(i0, 0), min(i1, ops.nx)
    j0, j1 = max(j0, 0), min(j1, ops.ny)
    k0, k1 = max(k0, 0), min(k1, ops.nz)
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return (0, 0, 0, 0, 0, 0)
    return (k0, k1, j0, j1, i0, i1)


def window_contains(outer, inner) -> bool:
    """True when cell window ``inner`` lies inside ``outer``."""
    return all(outer[2 * a] <= inner[2 * a] and inner[2 * a + 1] <= outer[2 * a + 1] for a in range(3))


def empty_geometry(nb: int, device, dtype) -> Geometry3D:
    z = torch.zeros(0, 0, 0, device=device, dtype=dtype)
    k = torch.zeros(0, 3, 0, 0, 0, device=device, dtype=dtype)
    return Geometry3D((0, 0, 0, 0, 0, 0), z, z, z, z, z, z, z, k, k, k, nb)


def _mesh_tensors(b, device, dtype):
    """(vertices tensor, faces long tensor, omega, center, axis) of one body item."""
    if isinstance(b, ImmersedMesh):
        v, f = b.vertices, b.faces
        rot = (b.omega, b.center, b.axis)
    else:
        v, f = b[0], b[1]
        rot = (0.0, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    vt = v.to(device=device, dtype=dtype) if _is_tensor(v) else torch.as_tensor(
        np.asarray(v, dtype=np.float64), device=device, dtype=dtype)
    ft = torch.as_tensor(np.asarray(f.detach().cpu().numpy() if _is_tensor(f) else f, dtype=np.int64),
                         device=device)
    return vt.reshape(-1, 3), ft.reshape(-1, 3), rot


def build_geometry3d(bodies, ops, *, window: tuple[int, int, int, int, int, int] | None = None,
                     use_shared: bool = True) -> Geometry3D:
    """Masks, solid velocities and force weights for a list of immersed meshes.

    ``bodies`` items may be :class:`ImmersedMesh` (vertices as arrays or tensors
    that require grad) or ``(vertices, faces)`` pairs; rotation data comes from
    ImmersedMesh items. ``window`` forces a cell window (it must contain the
    bodies): the live solver and finite-difference checks use it to keep tensor
    shapes - and captured CUDA graphs - unchanged when a body moves slightly.
    """
    dev, dt = ops.device, ops.dtype
    items = [_mesh_tensors(b, dev, dt) for b in bodies]
    nb = len(items)
    if window is None:
        window = body_window3d([_vertices_numpy(b) for b in bodies], ops)
    k0, k1, j0, j1, i0, i1 = window
    wz, wy, wx = k1 - k0, j1 - j0, i1 - i0
    if nb == 0 or wz <= 0 or wy <= 0 or wx <= 0:
        return empty_geometry(nb, dev, dt)
    h = ops.h
    w = ops.mask_width * h
    shift = ops.mask_shift * h
    band = (ops.mask_width + abs(ops.mask_shift) + 1.0) * h
    X0, Y0, Z0 = ops.x0 + i0 * h, ops.y0 + j0 * h, ops.z0 + k0 * h
    # grids whose cell centres are the staggered positions (geometry3d convention)
    grids = {
        "u": (wx + 1, wy, wz, h, X0 - 0.5 * h, Y0, Z0),
        "v": (wx, wy + 1, wz, h, X0, Y0 - 0.5 * h, Z0),
        "w": (wx, wy, wz + 1, h, X0, Y0, Z0 - 0.5 * h),
        "c": (wx, wy, wz, h, X0, Y0, Z0),
    }
    chis: dict[str, list[torch.Tensor]] = {k: [] for k in grids}
    for vt, ft, _ in items:
        for key, g in grids.items():
            d = grid_signed_distance([(vt, ft)], g, band, use_shared=use_shared)
            chis[key].append(smooth_step(d + shift, w))
    cu, cv, cw, cc = (torch.stack(chis[k]) for k in ("u", "v", "w", "c"))   # (nb, ...)

    # boundary faces carry boundary conditions, never penalisation
    px, py, pz = ops.periodic_axes
    if not px:
        fu = torch.ones(wx + 1, device=dev, dtype=dt)
        if i0 == 0:
            fu[0] = 0.0
        if i1 == ops.nx:
            fu[-1] = 0.0
        cu = cu * fu
    if not py:
        fv = torch.ones(wy + 1, device=dev, dtype=dt)
        if j0 == 0:
            fv[0] = 0.0
        if j1 == ops.ny:
            fv[-1] = 0.0
        cv = cv * fv[:, None]
    if not pz:
        fw = torch.ones(wz + 1, device=dev, dtype=dt)
        if k0 == 0:
            fw[0] = 0.0
        if k1 == ops.nz:
            fw[-1] = 0.0
        cw = cw * fw[:, None, None]

    chi_u = cu.sum(0).clamp(max=1.0)
    chi_v = cv.sum(0).clamp(max=1.0)
    chi_w = cw.sum(0).clamp(max=1.0)
    chi_c = cc.sum(0).clamp(max=1.0)

    def coords(g):
        nx_, ny_, nz_, _, gx, gy, gz = g
        zs = gz + (torch.arange(nz_, device=dev, dtype=dt) + 0.5) * h
        ys = gy + (torch.arange(ny_, device=dev, dtype=dt) + 0.5) * h
        xs = gx + (torch.arange(nx_, device=dev, dtype=dt) + 0.5) * h
        return zs[:, None, None], ys[None, :, None], xs[None, None, :]

    zu, yu, _ = coords(grids["u"])       # u_s,x does not depend on x, and so on
    zv, _, xv = coords(grids["v"])
    _, yw, xw = coords(grids["w"])

    # rigid rotation u_s = omega * axis x (x - c), pre-multiplied by each body's mask
    a_u = torch.zeros_like(chi_u)
    a_v = torch.zeros_like(chi_v)
    a_w = torch.zeros_like(chi_w)
    for b, (_, _, (om, cen, ax)) in enumerate(items):
        if om == 0.0:
            continue
        cx, cy, cz = cen
        axx, axy, axz = ax
        a_u = a_u + cu[b] * (om * (axy * (zu - cz) - axz * (yu - cy)))
        a_v = a_v + cv[b] * (om * (axz * (xv - cx) - axx * (zv - cz)))
        a_w = a_w + cw[b] * (om * (axx * (yw - cy) - axy * (xw - cx)))

    # per-body share of the penalisation at each face (1 on a body's own support
    # when bodies do not overlap), then lever arms about the moment reference
    if nb == 1:
        wu, wv, ww = torch.ones_like(cu), torch.ones_like(cv), torch.ones_like(cw)
    else:
        wu = cu / (cu.sum(0, keepdim=True) + 1e-30)
        wv = cv / (cv.sum(0, keepdim=True) + 1e-30)
        ww = cw / (cw.sum(0, keepdim=True) + 1e-30)
    xr, yr, zr = ops.moment_ref
    one_u = torch.ones_like(chi_u)
    one_v = torch.ones_like(chi_v)
    one_w = torch.ones_like(chi_w)
    # M = r x F: u-faces carry Fx -> My += (z-zr) Fx, Mz -= (y-yr) Fx
    k_u = torch.stack([wu * one_u, wu * (zu - zr), -wu * (yu - yr)], dim=1)
    # v-faces carry Fy -> Mx -= (z-zr) Fy, Mz += (x-xr) Fy
    k_v = torch.stack([wv * one_v, -wv * (zv - zr), wv * (xv - xr)], dim=1)
    # w-faces carry Fz -> Mx += (y-yr) Fz, My -= (x-xr) Fz
    k_w = torch.stack([ww * one_w, ww * (yw - yr), -ww * (xw - xr)], dim=1)
    return Geometry3D(tuple(int(c) for c in window), chi_u, chi_v, chi_w, chi_c, a_u, a_v, a_w,
                      k_u, k_v, k_w, nb)


__all__ = [
    "Geometry3D",
    "body_window3d",
    "build_geometry3d",
    "empty_geometry",
    "grid_signed_distance",
    "mesh_signed_distance",
    "rotation_matrix",
    "smooth_step",
    "window_contains",
]
