"""Verification and validation cases for the 3D Navier-Stokes solver.

Shared by ``tests/test_ns3d*.py`` (slow validation tests) and
``validation/bench_ns3d.py`` (tables for docs/theory_ns3d.md). Each case returns
plain numbers so the callers can assert on them or print them.

Reference values, unconfined sphere (steady and axisymmetric below Re ~ 210):

* Drag: Clift, Grace & Weber (1978) correlation for 20 < Re < 260,
  ``Cd = 24/Re (1 + 0.1935 Re^0.6305)``: Re 20 -> 2.735, Re 50 -> 1.574,
  Re 100 -> 1.087; Johnson & Patel (1999, JFM 378) computed 1.087 at Re 100.
  Re 300 (periodic, planar-symmetric hairpin shedding, Johnson & Patel): mean
  Cd 0.656, mean lift 0.069, St 0.137.
* Separation angle: the Clift et al. fit to Taneda's (1956) measurements,
  ``180 - 42.5 (ln(Re/20))^0.483`` degrees from the front stagnation point
  (separation starts near Re 20). Recirculation length behind the sphere: about
  0.4 D at Re 50 and 0.88 D at Re 100; the DESIGN 7.12 band for Re 100 is 0.8-0.95.

A confined domain raises the drag, and at low Re much more than the classical
high-Re blockage estimates suggest: measured for the 2.85 %-blockage box of
:func:`sphere_config` (slip walls 2.6 D from the axis), +3.4 % at Re 100 and
+4.6 % at Re 20 against wider boxes extrapolated to zero blockage
(``bench_ns3d.py domain``, docs/theory_ns3d.md 10). That is why the Re 100 band
is 1.05-1.15 rather than 1.087 +- 2 %.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from aerolume.solvers.ns3d import ImmersedMesh, SolverConfig3D

#: Johnson & Patel (1999), sphere at Re 300: mean drag, mean lift and Strouhal number
SPHERE_RE300 = {"cd": 0.656, "cl": 0.069, "st": 0.137}
SPHERE_RE300_ST = SPHERE_RE300["st"]


def clift_cd(re: float) -> float:
    """Clift, Grace & Weber (1978) standard drag curve, 20 < Re < 260."""
    return 24.0 / re * (1.0 + 0.1935 * re ** 0.6305)


def separation_angle_ref(re: float) -> float:
    """Separation angle from the front stagnation point [deg], 20 < Re < 400: the fit
    180 - 42.5 (ln(Re/20))^0.483 to Taneda's (1956) measurements given by Clift et al. (1978)."""
    return 180.0 - 42.5 * math.log(re / 20.0) ** 0.483


# ---------------------------------------------------------------------------
# meshes (non-dimensional; outward-oriented, closed unless stated)
# ---------------------------------------------------------------------------

def icosphere(d: float = 1.0, level: int = 3, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic sphere of diameter ``d``: 20 * 4^level triangles, outward winding.

    Level 3 has 1280 faces (edge ~0.055 d), level 4 5120 (edge ~0.027 d): the
    chord deviation of level 4 is 0.04 % of d, far below a cell at <= 48 cells/D.
    """
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.asarray(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    faces = list(f)
    for _ in range(level):
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int, cache: dict = cache) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    vv = np.asarray(verts) * (0.5 * d) + np.asarray(center, dtype=np.float64)
    return vv, np.asarray(faces, dtype=np.int64)


def box_mesh(lx: float, ly: float, lz: float, center=(0.0, 0.0, 0.0), n: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box, each face split into ``n x n`` quads (2 triangles each), outward winding."""
    c = np.asarray(center, dtype=np.float64)
    half = np.array([lx, ly, lz]) / 2.0
    verts, faces = [], []
    s = np.linspace(-1.0, 1.0, n + 1)
    for axis in range(3):
        for sign in (-1.0, 1.0):
            a1, a2 = [k for k in range(3) if k != axis]
            base = len(verts)
            for p in s:
                for q in s:
                    x = np.zeros(3)
                    x[axis], x[a1], x[a2] = sign, p, q
                    verts.append(c + x * half)
            for i in range(n):
                for j in range(n):
                    v00 = base + i * (n + 1) + j
                    v01, v10, v11 = v00 + 1, v00 + n + 1, v00 + n + 2
                    # (a1, a2, axis) right-handed for axis = 0, 2 and left-handed for 1: flip so normals point out
                    flip = (sign < 0) != (axis == 1)
                    if flip:
                        faces += [(v00, v01, v11), (v00, v11, v10)]
                    else:
                        faces += [(v00, v10, v11), (v00, v11, v01)]
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


def cylinder_mesh(d: float, length: float, center=(0.0, 0.0, 0.0), axis: int = 1, n: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Closed circular cylinder (a wheel) of diameter ``d`` along coordinate ``axis``."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    a1, a2 = [k for k in range(3) if k != axis]
    ring = np.zeros((n, 3))
    ring[:, a1] = 0.5 * d * np.cos(th)
    ring[:, a2] = 0.5 * d * np.sin(th)
    lo, hi = ring.copy(), ring.copy()
    lo[:, axis], hi[:, axis] = -0.5 * length, 0.5 * length
    cl, ch = np.zeros(3), np.zeros(3)
    cl[axis], ch[axis] = -0.5 * length, 0.5 * length
    verts = np.vstack([lo, hi, cl[None], ch[None]]) + np.asarray(center, dtype=np.float64)
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, n + j, j), (i, n + i, n + j), (2 * n, i, j), (2 * n + 1, n + j, n + i)]
    return verts, np.asarray(faces, dtype=np.int64)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed mesh (positive for outward winding)."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    cells_per_d: float
    grid: tuple[int, int, int]
    steps: int
    wall_s: float
    values: dict = field(default_factory=dict)

    def row(self) -> str:
        vals = "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in self.values.items())
        return (f"{self.name:<16} {self.cells_per_d:>5.0f}/D grid {self.grid[0]}x{self.grid[1]}x{self.grid[2]} "
                f"steps {self.steps:>6d} wall {self.wall_s:7.1f}s  {vals}")


# ---------------------------------------------------------------------------
# Taylor-Green (periodic): order of accuracy
# ---------------------------------------------------------------------------

def taylor_green_error(n: int, *, advection: str = "central", re: float = 10.0, t_end: float = 0.5,
                       dtype: str = "float64", device: str = "cpu", plane: str = "xy") -> tuple[float, float]:
    """Max velocity error at ``t_end`` vs the exact decaying 2D Taylor-Green vortex,
    set up in a periodic cube (it is an exact Navier-Stokes solution in 3D too),
    and the energy-decay rate divided by the exact ``4 / Re``.

    ``plane`` chooses which coordinate plane the vortex lies in ("xy", "xz",
    "yz"), so every pair of velocity components and every pair of axes is
    exercised; the third direction is resolved by only 4 cells (the solution is
    uniform along it).
    """
    from aerolume.solvers.ns3d import build_geometry3d, init_state3d, make_operators3d, step3d
    L = 2 * math.pi
    h = L / n
    sizes = {"xy": (n, n, 4), "xz": (n, 4, n), "yz": (4, n, n)}[plane]
    cfg = SolverConfig3D(nx=sizes[0], ny=sizes[1], nz=sizes[2], h=h, x0=0.0, y0=0.0, z0=0.0, re=re,
                         periodic=True, advection=advection, dtype=dtype, device=device, perturb=False)
    ops = make_operators3d(cfg)
    d, td = ops.device, ops.dtype
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    face = {k: torch.arange(m + 1, device=d, dtype=td) * h for k, m in (("x", nx), ("y", ny), ("z", nz))}
    cen = {k: (torch.arange(m, device=d, dtype=td) + 0.5) * h for k, m in (("x", nx), ("y", ny), ("z", nz))}

    def grid(comp):
        zs = face["z"] if comp == "w" else cen["z"]
        ys = face["y"] if comp == "v" else cen["y"]
        xs = face["x"] if comp == "u" else cen["x"]
        return torch.meshgrid(zs, ys, xs, indexing="ij")

    a, b = plane                                     # velocity (A, B) = (sin a cos b, -cos a sin b)
    comps = {"x": "u", "y": "v", "z": "w"}
    exact = {}
    for comp in ("u", "v", "w"):
        z, y, x = grid(comp)
        coord = {"x": x, "y": y, "z": z}
        if comp == comps[a]:
            exact[comp] = torch.sin(coord[a]) * torch.cos(coord[b])
        elif comp == comps[b]:
            exact[comp] = -torch.cos(coord[a]) * torch.sin(coord[b])
        else:
            exact[comp] = torch.zeros_like(x)
    st = init_state3d(cfg, ops)._replace(u=exact["u"], v=exact["v"], w=exact["w"])
    geom = build_geometry3d([], ops)
    dt_lim = min(0.25 * h, 0.15 * h * h * re)
    nsteps = math.ceil(t_end / dt_lim)
    dt = t_end / nsteps
    e0 = float(sum((exact[c] ** 2).sum() for c in "uvw"))
    with torch.no_grad():
        for _ in range(nsteps):
            st, _ = step3d(st, geom, ops, dt)
    decay = math.exp(-2.0 * t_end / re)
    err = max(float((getattr(st, c) - exact[c] * decay).abs().max()) for c in "uvw")
    e1 = float(sum((getattr(st, c) ** 2).sum() for c in "uvw"))
    rate = -math.log(e1 / e0) / t_end
    return err, rate / (4.0 / re)


# ---------------------------------------------------------------------------
# sphere
# ---------------------------------------------------------------------------

def sphere_config(re: float, cells_per_d: int, *, half: bool = True, upstream: float = 5.0,
                  downstream: float = 10.0, half_width: float = 2.625, level: int = 4, device: str = "auto",
                  dtype: str = "float32", perturb: bool = False, **kw) -> SolverConfig3D:
    """Sphere of diameter 1 at the origin in a box [-upstream, downstream] x [-w, w]^2.

    ``half=True`` simulates y >= 0 with a symmetry plane at y = 0 (the reported
    coefficients are full-body values). Blockage = (pi/4) / (2 w)^2: 2.85 % for
    w = 2.625, which is a whole number of cells at 8, 16, 24 and 32 cells/D, so a
    grid study changes the resolution only, never the domain (at low Re the
    confinement alone moves the drag by several per cent). The z extent is
    centred on the sphere with an even number of cells, so the axis y = z = 0
    lies on cell faces. Slip side walls.
    """
    h = 1.0 / cells_per_d
    nx = round((upstream + downstream) / h)
    nw = round(half_width / h)
    v, f = icosphere(1.0, level)
    common = {"nx": nx, "nz": 2 * nw, "h": h, "x0": -upstream, "z0": -nw * h, "re": re,
              "bodies": [ImmersedMesh(v, f, name="sphere")], "ref_area": math.pi / 4.0, "length_ref": 1.0,
              "span_ref": 1.0, "moment_ref": (0.0, 0.0, 0.0), "device": device, "dtype": dtype, "perturb": perturb}
    common.update(kw)
    if half:
        return SolverConfig3D(ny=nw, y0=0.0, bc_y="symmetry_ymin", **common)
    return SolverConfig3D(ny=2 * nw, y0=-nw * h, bc_y="slip", **common)


def axis_profile(solver, xs: np.ndarray, *, mean: bool = True) -> np.ndarray:
    """Streamwise velocity on the line y = z = 0 at stations ``xs`` (trilinear from the u-faces)."""
    from aerolume.solvers.ns3d.fields import trilinear
    U = solver._staggered_means()[0] if mean else solver.state.u
    pts = torch.as_tensor(np.c_[xs, np.zeros_like(xs), np.zeros_like(xs)], device=solver.device,
                          dtype=solver.dtype)
    with torch.no_grad():
        return trilinear(U, pts, solver.ops, (0.0, 0.5, 0.5)).double().cpu().numpy()


def wake_length(solver, x_rear: float = 0.5, *, mean: bool = True, x_max: float = 4.0) -> float:
    """Recirculation length L/D behind a sphere at the origin: the last downstream zero
    of the axis velocity, minus x_rear (0 when there is no reversed flow)."""
    xs = np.linspace(x_rear, x_rear + x_max, int(x_max / solver.ops.h * 4) + 1)
    u = axis_profile(solver, xs, mean=mean)
    neg = np.where(u < 0)[0]
    if not len(neg):
        return 0.0
    k = neg[-1]
    if k + 1 >= len(u):
        return float("nan")
    xz = xs[k] - u[k] * (xs[k + 1] - xs[k]) / (u[k + 1] - u[k])
    return float(xz - x_rear)


def surface_forces(surf: dict, ref_area: float) -> dict:
    """Pressure and friction force coefficients integrated over one mesh's surface data
    (face averages of the vertex values, face normals oriented like the vertex normals)."""
    v, f = surf["vertices"], surf["faces"]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = 0.5 * np.cross(b - a, c - a)                                  # area-weighted face normal
    vn = surf["normals"][f].mean(axis=1)
    fn = np.where((np.sum(fn * vn, axis=1) < 0)[:, None], -fn, fn)
    cp = surf["cp"][f].mean(axis=1)
    tau = surf["tau"][f].mean(axis=1)
    area = np.linalg.norm(fn, axis=1)
    fp = -(cp[:, None] * fn).sum(axis=0) / ref_area             # C_F = (1/A) oint (-cp n + cf_vec) dA
    ff = (tau * area[:, None]).sum(axis=0) / ref_area
    return {"cd_surface": float(fp[0] + ff[0]), "cd_p": float(fp[0]), "cd_f": float(ff[0]),
            "cl_surface": float(fp[2] + ff[2])}


def sphere_landmarks(surf: dict, center=(0.0, 0.0, 0.0), bin_deg: float = 2.0) -> dict:
    """Front / rear cp and the separation angle (degrees from the front stagnation point)
    of an axisymmetric sphere flow: the first sign change of the meridional wall shear
    (binned over all azimuths) beyond 20 degrees."""
    p = surf["vertices"] - np.asarray(center)
    r = np.linalg.norm(p, axis=1)
    theta = np.degrees(np.arccos(np.clip(-p[:, 0] / r, -1.0, 1.0)))       # 0 at the front
    rho = np.hypot(p[:, 1], p[:, 2])
    safe = np.where(rho > 0, rho, 1.0)
    cosphi = np.where(rho > 0, p[:, 1] / safe, 1.0)
    sinphi = np.where(rho > 0, p[:, 2] / safe, 0.0)
    th = np.radians(theta)
    e_theta = np.stack([np.sin(th), np.cos(th) * cosphi, np.cos(th) * sinphi], axis=1)
    tt = np.sum(surf["tau"] * e_theta, axis=1)
    out = {"cp_front": float(surf["cp"][np.argmin(theta)]), "cp_rear": float(surf["cp"][np.argmax(theta)])}
    edges = np.arange(0.0, 180.0 + bin_deg, bin_deg)
    centres, vals = [], []
    for lo, hi in itertools.pairwise(edges):
        m = (theta >= lo) & (theta < hi)
        if m.any():
            centres.append(0.5 * (lo + hi))
            vals.append(tt[m].mean())
    centres, vals = np.asarray(centres), np.asarray(vals)
    sel = centres > 20.0
    c, val = centres[sel], vals[sel]
    flips = np.where((val[:-1] > 0) & (val[1:] <= 0))[0]
    if len(flips):
        i = flips[0]
        out["sep_deg"] = float(c[i] + (c[i + 1] - c[i]) * val[i] / (val[i] - val[i + 1]))
    else:
        out["sep_deg"] = float("nan")
    return out


def control_volume_drag(solver, box: tuple[float, float, float, float, float, float], *,
                        mean: bool = True) -> dict:
    """Drag from the steady momentum balance on a grid-aligned box around the body.

    ``F_x = -oint [ p n_x + u (u.n) - nu (du/dx_j + du_j/dx) n_j ] dA`` with every
    quantity taken at second order from its own staggered position; the box
    faces lie on the cell faces nearest to ``box`` = (xa, xb, ya, yb, za, zb).
    In a half-domain run a box reaching the symmetry plane has no face there
    (v = 0 and du/dy = 0 on it) and the result is doubled to the full body.
    Valid for steady flows (time averages ignore the Reynolds stresses).
    """
    ops = solver.ops
    if mean:
        U, V, W, P, _ = solver._staggered_means()
    else:
        s = solver.state
        U, V, W, P = s.u, s.v, s.w, s.p
    U, V, W, P = (t.double() for t in (U, V, W, P))
    h, nu = ops.h, 1.0 / solver.cfg.re

    def idx(val, origin, n):
        return int(min(max(round((val - origin) / h), 1), n - 1))
    ia, ib = idx(box[0], ops.x0, ops.nx), idx(box[1], ops.x0, ops.nx)
    ja, jb = idx(box[2], ops.y0, ops.ny), idx(box[3], ops.y0, ops.ny)
    if ops.symmetry and box[2] <= ops.y0:
        ja = 0
    ka, kb = idx(box[4], ops.z0, ops.nz), idx(box[5], ops.z0, ops.nz)
    K, J = slice(ka, kb), slice(ja, jb)
    parts = {"pressure": 0.0, "momentum": 0.0, "viscous": 0.0}
    for i, sgn in ((ib, 1.0), (ia, -1.0)):                       # x faces (u-faces)
        pf = 0.5 * (P[K, J, i - 1] + P[K, J, i])
        dudx = (U[K, J, i + 1] - U[K, J, i - 1]) / (2 * h)
        parts["pressure"] += sgn * float(pf.sum()) * h * h
        parts["momentum"] += sgn * float((U[K, J, i] ** 2).sum()) * h * h
        parts["viscous"] -= sgn * float((2 * nu * dudx).sum()) * h * h
    ux0, ux1 = slice(ia, ib), slice(ia + 1, ib + 1)                 # u at the two x-faces of each cell
    for j, sgn in ((jb, 1.0), (ja, -1.0)):                       # y faces (v-faces)
        if ops.symmetry and j == 0:
            continue
        lo = U[K, j - 1, ux0] + U[K, j - 1, ux1]
        hi = U[K, j, ux0] + U[K, j, ux1]
        dvdx = (V[K, j, ia + 1:ib + 1] - V[K, j, ia - 1:ib - 1]) / (2 * h)
        parts["momentum"] += sgn * float((0.25 * (lo + hi) * V[K, j, ia:ib]).sum()) * h * h
        parts["viscous"] -= sgn * float((nu * (0.5 * (hi - lo) / h + dvdx)).sum()) * h * h
    for k, sgn in ((kb, 1.0), (ka, -1.0)):                       # z faces (w-faces)
        lo = U[k - 1, J, ux0] + U[k - 1, J, ux1]
        hi = U[k, J, ux0] + U[k, J, ux1]
        dwdx = (W[k, J, ia + 1:ib + 1] - W[k, J, ia - 1:ib - 1]) / (2 * h)
        parts["momentum"] += sgn * float((0.25 * (lo + hi) * W[k, J, ia:ib]).sum()) * h * h
        parts["viscous"] -= sgn * float((nu * (0.5 * (hi - lo) / h + dwdx)).sum()) * h * h
    flux = parts["pressure"] + parts["momentum"] + parts["viscous"]
    scale = (2.0 if ops.symmetry else 1.0) / (0.5 * ops.ref_area)
    return {"cd_cv": -flux * scale, **{f"cd_cv_{k}": -v * scale for k, v in parts.items()}}


def strouhal(t: np.ndarray, x: np.ndarray, length: float = 1.0) -> float | None:
    """Shedding frequency from upward zero crossings of x - mean (needs >= 4 crossings)."""
    y = x - x.mean()
    idx = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(idx) < 4:
        return None
    tc = t[idx] - y[idx] * (t[idx + 1] - t[idx]) / (y[idx + 1] - y[idx])
    return length / float(np.mean(np.diff(tc[1:])))


def run_sphere_unsteady(re: float, cells_per_d: int, t_end: float = 150.0, t_avg: float = 80.0, *,
                        chunk: int = 500, **kw) -> CaseResult:
    """Periodic sphere wake (Re 300: planar-symmetric hairpin shedding, Johnson & Patel 1999).

    Half domain (the symmetry plane is the plane of the shedding), perturbation
    on so the shedding locks in promptly; mean Cd, mean and amplitude of the lift
    (in z) and St from the lift over [t_avg, t_end].
    """
    from aerolume.solvers.ns3d import FlowSolver3D
    kw.setdefault("perturb", True)
    cfg = sphere_config(re, cells_per_d, **kw)
    s = FlowSolver3D(cfg)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    hst = s.history()
    m = hst["t"] >= t_avg
    t, cd, cl = hst["t"][m], hst["cd"][m], hst["cl"][m]
    st = strouhal(t, cl)
    vals = {"cd": float(cd.mean()), "cd_amp": float(0.5 * (cd.max() - cd.min())), "cl_mean": float(cl.mean()),
            "cl_amp": float(0.5 * (cl.max() - cl.min())), "st": float(st) if st is not None else float("nan"),
            "t_end": float(s.t), "steps_per_s": s.steps / wall}
    return CaseResult(f"sphere Re{re:g} unsteady", cells_per_d, (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


def moving_road_config(cells_per_l: int = 64, *, re: float = 1e6, turbulence: str = "wale", device: str = "auto",
                       dtype: str = "float32", wheels: bool = True) -> SolverConfig3D:
    """A box "car" (length 1, width 0.4, height 0.3) 0.08 above a road moving at U, with
    wheels (radius 0.07, width 0.06) that touch the road and roll with it (omega =
    -U/r about +y), half domain with a symmetry plane (so only the two wheels on
    the y > 0 side are bodies; the solver's full-body forces include their
    mirror images), upwind3 + LES.

    A robustness test of the moving-road boundary, rotating wheels, WALE at
    Re 1e6 and the symmetry plane together - not a validation case.
    """
    h = 1.0 / cells_per_l
    x0, x1, width, top = -1.0, 3.0, 0.8, 0.9
    nx, ny, nz = round((x1 - x0) / h), round(width / h), round(top / h)
    v, f = box_mesh(1.0, 0.4, 0.3, center=(0.5, 0.0, 0.08 + 0.15), n=4)
    bodies = [ImmersedMesh(v, f, name="body")]
    if wheels:
        r = 0.07
        for xw in (0.2, 0.8):
            for yw in (0.15,):
                wv, wf = cylinder_mesh(2 * r, 0.06, center=(xw, yw, r), axis=1, n=24)
                bodies.append(ImmersedMesh(wv, wf, name=f"wheel@{xw},{yw}", role="wheel", omega=-1.0 / r,
                                           center=(xw, yw, r), axis=(0.0, 1.0, 0.0)))
    return SolverConfig3D(nx=nx, ny=ny, nz=nz, h=h, x0=x0, y0=0.0, z0=0.0, re=re, bodies=bodies,
                          bc_y="symmetry_ymin", bc_zmin="moving", ground_speed=1.0, bc_zmax="slip",
                          advection="upwind3", turbulence=turbulence, device=device, dtype=dtype,
                          ref_area=0.4 * 0.3, length_ref=1.0, span_ref=0.4, moment_ref=(0.5, 0.0, 0.0))


def run_moving_road(cells_per_l: int = 64, t_end: float = 6.0, *, chunk: int = 500, **kw) -> CaseResult:
    """Run :func:`moving_road_config` and report stability diagnostics and mean forces (t > t_end/2)."""
    from aerolume.solvers.ns3d import FlowSolver3D
    cfg = moving_road_config(cells_per_l, **kw)
    s = FlowSolver3D(cfg)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    peak = 0.0
    while s.t < t_end:
        s.advance(chunk)
        peak = max(peak, float(s.field("speed").max()))
    if s.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    hst = s.history()
    m = hst["t"] > 0.5 * t_end
    finite = all(bool(torch.isfinite(getattr(s.state, n)).all()) for n in ("u", "v", "w", "p"))
    vals = {"finite": finite, "max_speed": peak, "divergence_h": float(s.field("divergence").abs().max()) * cfg.h,
            "nut_max": float(s.field("nut").max()), "cd_mean": float(hst["cd"][m].mean()),
            "cl_mean": float(hst["cl"][m].mean()), "cm_mean": float(hst["cm"][m].mean()),
            "dt_last": float(hst["t"][-1] - hst["t"][-2]), "t_end": float(s.t), "steps_per_s": s.steps / wall}
    return CaseResult("moving road box", cells_per_l, (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


def run_sphere(re: float, cells_per_d: int, t_end: float = 40.0, t_avg: float | None = None, *,
               chunk: int = 200, **kw) -> CaseResult:
    """Steady sphere: run to ``t_end``, average over [t_avg, t_end] (default the last
    quarter), report Cd, its drift over the averaging window, the wake length, the
    surface-integrated and control-volume drag, surface landmarks and blockage."""
    from aerolume.solvers.ns3d import FlowSolver3D
    cfg = sphere_config(re, cells_per_d, **kw)
    s = FlowSolver3D(cfg)
    t_avg = 0.75 * t_end if t_avg is None else t_avg
    if s.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    averaging = False
    while s.t < t_end:
        if not averaging and s.t >= t_avg:
            s.start_averaging()
            averaging = True
        s.advance(chunk)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    hst = s.history()
    m = hst["t"] >= t_avg
    cd = hst["cd"][m]
    half = len(cd) // 2
    vals = {"cd": float(cd.mean()), "cd_drift": float(abs(cd[half:].mean() - cd[:half].mean()) / abs(cd.mean())),
            "cl": float(hst["cl"][m].mean()), "L/D": wake_length(s, mean=averaging)}
    surf = s.surface(mean=averaging)[0]
    vals.update(surface_forces(surf, cfg.ref_area))
    vals.update(sphere_landmarks(surf))
    ext = cfg.extent
    vals.update(control_volume_drag(s, (-1.0, 1.5, ext[2], 1.0, -1.0, 1.0), mean=averaging))
    width = (ext[3] - ext[2]) * (2.0 if cfg.symmetry else 1.0)
    vals["blockage"] = float(cfg.ref_area / (width * (ext[5] - ext[4])))
    vals["t_end"] = float(s.t)
    vals["steps_per_s"] = s.steps / wall
    vals["fused"] = s.uses_fused_kernels
    if s.device.type == "cuda":
        vals["peak_mem_gb"] = torch.cuda.max_memory_allocated(s.device) / 1e9
    name = f"sphere Re{re:g}" + (" half" if cfg.symmetry else " full")
    return CaseResult(name, cells_per_d, (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)
