"""Verification and validation cases for the 2D Navier-Stokes solver.

Shared by ``tests/test_ns2d*.py`` (slow validation tests) and
``validation/bench_ns2d.py`` (benchmark tables for docs/theory_ns2d.md). Each
case returns plain numbers so the callers can assert on them or print them.

Reference values (unconfined cylinder, 2D laminar):

* Re = 40 (steady): Cd 1.50-1.55 (Dennis & Chang 1970: 1.522; Fornberg 1980:
  1.498; Linnick & Fasel 2005: 1.54; Taira & Colonius 2007: 1.54), recirculation
  length L/D 2.13-2.35 measured from the rear of the cylinder (Coutanceau &
  Bouard 1977: 2.13; Fornberg: 2.24; Linnick & Fasel: 2.28).
* Re = 100 (periodic shedding): mean Cd 1.33-1.38, Cl amplitude 0.32-0.34,
  St 0.164-0.167 (Williamson 1989 experiments: 0.164; Liu et al. 1998: Cd 1.35,
  Cl +-0.339, St 0.164; Linnick & Fasel: 1.34, +-0.333, 0.166).

Confined and sharp-edged benchmarks (end of the file): Schaefer & Turek (1996)
2D-1 and 2D-2 (SCHAFER_TUREK_REF) and the square cylinder at Re 100
(SQUARE_RE100_REF); results in docs/validation.md section 3.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from aerolume.solvers.ns2d import FlowSolver, ImmersedBody, SolverConfig

# ---------------------------------------------------------------------------
# shapes (non-dimensional, CCW, implicitly closed)
# ---------------------------------------------------------------------------


def circle(n: int = 256, d: float = 1.0, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[cx + 0.5 * d * np.cos(th), cy + 0.5 * d * np.sin(th)]


def naca4(code: str = "0012", n: int = 200, chord: float = 1.0) -> np.ndarray:
    """NACA 4-digit section with a closed trailing edge, CCW (TE, upper, LE, lower)."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0.0, np.pi, n // 2 + 1)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    if m > 0:
        yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
        dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    upper = np.c_[xu, yu][::-1]          # TE -> LE
    lower = np.c_[xl, yl][1:-1]          # LE -> TE (both ends dropped: shared points)
    pts = np.vstack([upper, lower]) * chord
    return pts


def rounded_box(length: float = 1.0, height: float = 0.3, radius: float = 0.08, x0: float = 0.0,
                y0: float = 0.05, n_arc: int = 16) -> np.ndarray:
    """A CCW rounded rectangle: a crude 2D car body for moving-ground smoke tests."""
    r = min(radius, 0.5 * height, 0.5 * length)
    corners = [(x0 + length - r, y0 + r, -0.5 * np.pi), (x0 + length - r, y0 + height - r, 0.0),
               (x0 + r, y0 + height - r, 0.5 * np.pi), (x0 + r, y0 + r, np.pi)]
    pts = []
    for cx, cy, a0 in corners:
        a = a0 + np.linspace(0.0, 0.5 * np.pi, n_arc, endpoint=False)
        pts.append(np.c_[cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.vstack(pts)


def rotate(pts: np.ndarray, deg: float, about=(0.0, 0.0)) -> np.ndarray:
    """Rotate counter-clockwise by ``deg`` about a point."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    p = np.asarray(pts, float) - about
    return np.c_[c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]] + about


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------

def strouhal(t: np.ndarray, cl: np.ndarray, length: float = 1.0) -> float | None:
    """Shedding frequency from upward zero crossings of Cl - mean (needs >= 3 cycles)."""
    x = cl - cl.mean()
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(idx) < 4:
        return None
    tc = t[idx] - x[idx] * (t[idx + 1] - t[idx]) / (x[idx + 1] - x[idx])
    period = np.mean(np.diff(tc[1:]))      # skip the first crossing
    return length / period


def surface_drag(surf: dict) -> dict:
    """Pressure, friction and total drag coefficient integrated from one body's
    :meth:`FlowSolver.surface` data (uniform arc-length spacing, force_ref = 1)."""
    ds = surf["s"][1] - surf["s"][0]
    cdp = float(np.sum(-surf["cp"] * surf["nx"]) * ds)
    cdf = float(np.sum(surf["cf"] * -surf["ny"]) * ds)       # tangent along increasing s is (-ny, nx)
    return {"cd_surface": cdp + cdf, "cd_p": cdp, "cd_f": cdf}


def surface_landmarks(surf: dict) -> dict:
    """cp at the front / rear of a body centred on the x axis and the separation
    angle on its upper half (first cf sign change, degrees from the front).

    The first 20 degrees are skipped: at the front stagnation point cf is ~0 and
    its sign is round-off, especially in time averages."""
    x, y, cf = surf["x"], surf["y"], surf["cf"]
    i_front, i_rear = int(np.argmin(x)), int(np.argmax(x))
    out = {"cp_front": float(surf["cp"][i_front]), "cp_rear": float(surf["cp"][i_rear])}
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    ang = np.degrees(np.arctan2(y - cy, -(x - cx)))          # 0 at the front, 180 at the rear
    up = np.where(ang > 20.0)[0]
    up = up[np.argsort(ang[up])]
    flips = np.where(np.diff(np.sign(cf[up])) != 0)[0]
    out["sep_deg"] = float(ang[up][flips[0]]) if len(flips) else float("nan")
    return out


def wake_length(solver: FlowSolver, x_rear: float = 0.5, y_axis: float = 0.0, mean: bool = False) -> float | None:
    """Recirculation length behind a body: last zero of the centreline u, minus x_rear."""
    u = solver.mean_fields()["u"] if mean else solver.field("u")
    ops = solver.ops
    yc = ops.y0 + (np.arange(ops.ny) + 0.5) * ops.h
    j = int(np.searchsorted(yc, y_axis))
    w = (y_axis - yc[j - 1]) / ops.h
    line = ((1 - w) * u[j - 1] + w * u[j]).detach().cpu().double().numpy()
    xc = ops.x0 + (np.arange(ops.nx) + 0.5) * ops.h
    m = xc > x_rear
    xs, us = xc[m], line[m]
    neg = np.where(us < 0)[0]
    if not len(neg):
        return None
    k = neg[-1]
    if k + 1 >= len(us):
        return None
    xz = xs[k] - us[k] * (xs[k + 1] - xs[k]) / (us[k + 1] - us[k])
    return float(xz - x_rear)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    cells_per_d: float
    grid: tuple[int, int]
    steps: int
    wall_s: float
    values: dict = field(default_factory=dict)

    def row(self) -> str:
        vals = "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in self.values.items())
        return (f"{self.name:<14} {self.cells_per_d:>5.0f}/D grid {self.grid[0]}x{self.grid[1]} "
                f"steps {self.steps:>6d} wall {self.wall_s:7.1f}s  {vals}")


def cylinder_config(re: float, cells_per_d: int, *, upstream: float = 10.0, downstream: float = 20.0,
                    half_height: float = 10.0, device: str = "auto", dtype: str = "float32",
                    advection: str = "upwind3", perturb: bool = True, **kw) -> SolverConfig:
    """Cylinder of diameter 1 at the origin; slip top/bottom (blockage 1 / (2 half_height))."""
    h = 1.0 / cells_per_d
    nx = round((upstream + downstream) / h)
    ny = round(2 * half_height / h)
    n_poly = max(64, int(2 * math.pi / h))           # ~2 vertices per cell along the surface
    return SolverConfig(nx=nx, ny=ny, h=h, x0=-upstream, y0=-half_height, re=re,
                        bodies=[ImmersedBody(circle(n_poly), name="cylinder")],
                        bc_top="slip", bc_bottom="slip", advection=advection, device=device, dtype=dtype,
                        force_ref=1.0, moment_ref=(0.0, 0.0), dye=False, perturb=perturb, **kw)


def run_cylinder(re: float, cells_per_d: int, t_end: float, t_avg: float, *, chunk: int = 200,
                 **kw) -> CaseResult:
    """Run a cylinder case; average forces over [t_avg, t_end]."""
    cfg = cylinder_config(re, cells_per_d, **kw)
    s = FlowSolver(cfg)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
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
    t, cd, cl = hst["t"][m], hst["cd"][m], hst["cl"][m]
    vals = {"cd": float(cd.mean()), "cl_amp": float(0.5 * (cl.max() - cl.min())), "cl_mean": float(cl.mean())}
    st = strouhal(t, cl)
    vals["st"] = float(st) if st is not None else float("nan")
    lw = wake_length(s, mean=averaging)
    vals["L/D"] = float(lw) if lw is not None else float("nan")
    # surface-integrated drag (pressure + friction) from the mean surface data, as a cross-check
    surf = s.surface(720, mean=averaging)[0]
    vals.update(surface_drag(surf))
    vals.update(surface_landmarks(surf))
    vals["t_end"] = float(s.t)
    return CaseResult(f"cylinder Re{re:g}", cells_per_d, (cfg.nx, cfg.ny), s.steps, wall, vals)


def taylor_green_error(n: int, *, advection: str = "central", re: float = 10.0, t_end: float = 1.0,
                       dtype: str = "float64", device: str = "cpu") -> tuple[float, float]:
    """Max velocity error at t_end vs the exact decaying Taylor-Green vortex, and
    the measured energy-decay rate divided by the exact one (2 nu per unit... = 4/Re for |u|^2)."""
    from aerolume.solvers.ns2d import build_geometry, init_state, make_operators, step
    L = 2 * math.pi
    h = L / n
    cfg = SolverConfig(nx=n, ny=n, h=h, x0=0.0, y0=0.0, re=re, periodic=True, advection=advection,
                       dtype=dtype, device=device, dye=False, perturb=False)
    ops = make_operators(cfg)
    d, td = ops.device, ops.dtype
    xu = torch.arange(n + 1, device=d, dtype=td) * h
    xc = (torch.arange(n, device=d, dtype=td) + 0.5) * h
    yu, xx = torch.meshgrid(xc, xu, indexing="ij")
    u = torch.sin(xx) * torch.cos(yu)
    yv, xv = torch.meshgrid(xu, xc, indexing="ij")
    v = -torch.cos(xv) * torch.sin(yv)
    st = init_state(cfg, ops)._replace(u=u, v=v)
    geom = build_geometry([], ops)
    dt_lim = min(0.25 * h, 0.2 * h * h * re)
    nsteps = math.ceil(t_end / dt_lim)
    dt = t_end / nsteps
    e0 = float((u * u).sum() + (v * v).sum())
    with torch.no_grad():
        for _ in range(nsteps):
            st, _ = step(st, geom, ops, dt)
    decay = math.exp(-2.0 * t_end / re)
    err = float((st.u - u * decay).abs().max())
    e1 = float((st.u * st.u).sum() + (st.v * st.v).sum())
    rate = -math.log(e1 / e0) / t_end
    return err, rate / (4.0 / re)


def moving_ground_config(cells_per_l: int = 128, *, re: float = 1e6, turbulence: str = "wale",
                         device: str = "auto", dtype: str = "float32", wheels: bool = True) -> SolverConfig:
    """Rounded box (length 1, height 0.3) 0.08 above a moving road, optional rolling wheels.

    The wheels (radius 0.07) touch the road and overlap the body, as a planner
    would place them, and roll with the road: omega = +ground_speed / r
    (counter-clockwise, so the contact point moves with the road in +x). In 2D
    such wheels seal the underbody between them into a closed pocket, whose
    pressure is only set by seepage through the diffuse walls; the body's lift
    is therefore geometry-dominated here (theory doc, section 11). The case is
    a robustness test, not a validation.
    """
    h = 1.0 / cells_per_l
    x0, x1, top = -1.0, 3.0, 1.2
    nx, ny = round((x1 - x0) / h), round(top / h)
    bodies = [ImmersedBody(rounded_box(1.0, 0.3, 0.08, 0.0, 0.08), name="body")]
    if wheels:
        r = 0.07
        for xw in (0.2, 0.8):
            bodies.append(ImmersedBody(circle(64, 2 * r, xw, r), name=f"wheel@{xw}", role="wheel",
                                       omega=1.0 / r, center=(xw, r)))   # rolls with the road
    return SolverConfig(nx=nx, ny=ny, h=h, x0=x0, y0=0.0, re=re, bodies=bodies, bc_bottom="moving",
                        ground_speed=1.0, bc_top="slip", advection="upwind3", turbulence=turbulence,
                        device=device, dtype=dtype, force_ref=0.3, moment_ref=(0.5, 0.0))


# ---------------------------------------------------------------------------
# confined and sharp-edged benchmarks
# ---------------------------------------------------------------------------

#: Schaefer & Turek (1996) benchmark bands, in the benchmark's own units
#: (rho = 1, D = 0.1 m, mean inflow 0.2 m/s for 2D-1 and 1 m/s for 2D-2).
SCHAFER_TUREK_REF = {
    "2D-1": {"cd": (5.57, 5.59), "cl": (0.0104, 0.0110), "La": (0.0842, 0.0852), "dp": (0.1172, 0.1176)},
    "2D-2": {"cd_max": (3.22, 3.24), "cl_max": (0.99, 1.01), "st": (0.295, 0.305)},
}

#: Square cylinder, Re = 100, about 5 % blockage: Sohankar et al. (1998) Cd 1.477, St 0.146;
#: Darekar & Sherwin (2001) Cd 1.488, St 0.145; Sharma & Eswaran (2004) Cd 1.494, St 0.149.
SQUARE_RE100_REF = {"cd": (1.477, 1.494), "st": (0.145, 0.149)}


def square(side: float = 1.0, n_per_side: int = 64, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """Axis-aligned square, counter-clockwise, ``n_per_side`` points per side (sharp corners)."""
    a = 0.5 * side
    s = np.linspace(-a, a, n_per_side, endpoint=False)
    edges = [np.c_[s, np.full_like(s, -a)],          # bottom, left -> right
             np.c_[np.full_like(s, a), s],           # right, bottom -> top
             np.c_[-s, np.full_like(s, a)],          # top, right -> left
             np.c_[np.full_like(s, -a), -s]]         # left, top -> bottom
    return np.vstack(edges) + (cx, cy)


def schafer_turek_config(re: float, cells_per_d: int, *, device: str = "auto", dtype: str = "float32",
                         perturb: bool = True, **kw) -> SolverConfig:
    """Schaefer-Turek channel in units of the cylinder diameter D.

    The benchmark channel is 2.2 m x 0.41 m with a 0.1 m cylinder centred at
    (0.2, 0.2): 22 D x 4.1 D, the cylinder 2 D from the inlet and 0.05 D below
    the channel's mid-plane (the slight asymmetry gives the steady case its small
    lift). No-slip walls, parabolic inflow whose mean velocity is the reference
    speed, Re = U_mean D / nu. ``cells_per_d`` must make 4.1 D a whole number of
    cells (a multiple of 10).
    """
    if (41 * cells_per_d) % 10:
        raise ValueError("cells_per_d must be a multiple of 10 so the 4.1 D channel fits the grid")
    h = 1.0 / cells_per_d
    nx, ny = 22 * cells_per_d, 41 * cells_per_d // 10
    n_poly = max(64, int(2 * math.pi / h))
    return SolverConfig(nx=nx, ny=ny, h=h, x0=-2.0, y0=-2.0, re=re, inlet="parabolic",
                        bodies=[ImmersedBody(circle(n_poly), name="cylinder")],
                        bc_top="noslip", bc_bottom="noslip", advection="upwind3", device=device, dtype=dtype,
                        force_ref=1.0, moment_ref=(0.0, 0.0), dye=False, perturb=perturb, **kw)


def run_schafer_turek(case: str, cells_per_d: int, *, t_end: float | None = None, t_avg: float | None = None,
                      chunk: int = 200, **kw) -> CaseResult:
    """Run benchmark 2D-1 (Re 20, steady) or 2D-2 (Re 100, periodic).

    Values are reported in the benchmark's own units so they compare directly
    with its bands: the coefficients use the mean inflow speed (as in the
    benchmark), the recirculation length ``La`` is in metres (D = 0.1 m) and the
    pressure difference between the front and rear points of the cylinder ``dp``
    is in Pa for rho = 1 and a mean speed of 0.2 m/s, i.e. 0.02 times the
    difference of the surface cp.
    """
    if case not in SCHAFER_TUREK_REF:
        raise ValueError(f"unknown case {case!r}")
    steady = case == "2D-1"
    re = 20.0 if steady else 100.0
    t_end = t_end if t_end is not None else (40.0 if steady else 60.0)
    t_avg = t_avg if t_avg is not None else (30.0 if steady else 40.0)
    cfg = schafer_turek_config(re, cells_per_d, **kw)
    s = FlowSolver(cfg)
    if s.device.type == "cuda":
        torch.cuda.synchronize()
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
    t, cd, cl = hst["t"][m], hst["cd"][m], hst["cl"][m]
    vals: dict = {}
    if steady:
        surf = s.surface(720)[0]
        land = surface_landmarks(surf)
        lw = wake_length(s)
        vals.update(cd=float(cd[-1]), cl=float(cl[-1]),
                    La=float(lw) * 0.1 if lw is not None else float("nan"),
                    dp=0.02 * (land["cp_front"] - land["cp_rear"]),
                    cd_drift=float(abs(cd[-1] - cd[0]) / abs(cd[-1])))
    else:
        st = strouhal(t, cl)
        vals.update(cd_max=float(cd.max()), cl_max=float(cl.max()), cd_mean=float(cd.mean()),
                    st=float(st) if st is not None else float("nan"))
    vals["t_end"] = float(s.t)
    return CaseResult(f"Schafer-Turek {case}", cells_per_d, (cfg.nx, cfg.ny), s.steps, wall, vals)


def run_square(re: float, cells_per_d: int, t_end: float = 200.0, t_avg: float = 100.0, *, chunk: int = 200,
               upstream: float = 10.0, downstream: float = 20.0, half_height: float = 10.0,
               device: str = "auto", dtype: str = "float32", **kw) -> CaseResult:
    """Square cylinder (side 1, faces normal to the flow) in the cylinder's wide domain.

    Same domain, outlet and slip walls as :func:`cylinder_config` (blockage 5 %),
    so the effect of sharp, grid-aligned corners can be read against the
    circular-cylinder rows. The faces sit on cell faces, so the diffuse
    interface is symmetric about the true wall.
    """
    h = 1.0 / cells_per_d
    nx, ny = round((upstream + downstream) / h), round(2 * half_height / h)
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=-upstream, y0=-half_height, re=re,
                       bodies=[ImmersedBody(square(1.0, max(32, 2 * cells_per_d)), name="square")],
                       bc_top="slip", bc_bottom="slip", advection="upwind3", device=device, dtype=dtype,
                       force_ref=1.0, moment_ref=(0.0, 0.0), dye=False, **kw)
    s = FlowSolver(cfg)
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
    vals = {"cd": float(cd.mean()), "cl_amp": float(0.5 * (cl.max() - cl.min())),
            "st": float(st) if st is not None else float("nan"), "t_end": float(s.t)}
    return CaseResult(f"square Re{re:g}", cells_per_d, (cfg.nx, cfg.ny), s.steps, wall, vals)
