"""Validation cases for the placed physics of the 3D solver (DESIGN.md section 10):
natural convection, forced convection (held temperature and uniform heat flux), buoyancy
on bodies and the time step of buoyancy-dominated runs, actuator discs, emitters.

Shared by ``tests/test_ns3d_*.py`` and ``validation/bench_ns3d.py thermal`` (the
tables of docs/theory_ns3d.md sections 15-18). Every case returns plain numbers.

References (every value below was read from the cited table; secondary sources are
named where the primary was not accessible)
------------------------------------------------------------------------------------
* **Differentially heated cubic cavity** (air, Pr = 0.71; hot wall theta = +1/2
  at x = 0, cold wall -1/2 at x = 1, the other four walls adiabatic, all walls
  no-slip, gravity along -z). Mean Nusselt number of the hot wall:

  ======================================  =========  =========
  source                                  Ra = 1e4   Ra = 1e5
  ======================================  =========  =========
  Tric, Labrosse & Betrouni (2000),        2.0542     4.3371
  Chebyshev pseudo-spectral
  Wakashima & Saitoh (2004)                2.062      4.366
  Fusegi, Hyun, Kuwahara & Farouk (1991)   2.100      4.361
  ======================================  =========  =========

  Tric et al. and Fusegi et al. as tabulated by Wang, Sadat & Prax (2018,
  arXiv:1809.02502, Table 1: "PSC [29]" and "FDM [30]"); Wakashima & Saitoh by
  Oztop et al. (2019, Entropy 21, 116, Table 1) and Al-Rashed et al. (2017, J.
  Therm. Sci. Tech. 12, Table 3). Li, Yang & Zhang (arXiv:1511.04633, Tables 1-2)
  give the same to three figures, and the hot-wall Nusselt number of the symmetry
  plane y = 1/2 of Tric et al.: 2.25 / 4.61. Centre-line velocity maxima in the
  symmetry plane (free-fall units U = sqrt(g beta dT L); Al-Rashed et al. Table 3,
  positions in our axes: x heated direction, z vertical): u_max on the vertical
  centre line 0.198 at z = 0.825 / 0.147 at 0.85 (Wakashima & Saitoh), 0.201 at
  0.817 / 0.147 at 0.855 (Fusegi et al.); w_max on the horizontal centre line
  0.222 at x = 0.117 / 0.246 at 0.068 (Wakashima), 0.225 at 0.117 / 0.247 at 0.065
  (Fusegi). The four-figure Fusegi mid-plane values (Hajabdollahi & Premnath
  2017, arXiv:1710.04751, Table 1): Nu 2.302 / 4.646, u_max 0.2013 / 0.1468,
  w_max 0.2252 / 0.2471.
* **Differentially heated square cavity** (2D limit, de Vahl Davis 1983, Int. J.
  Numer. Meth. Fluids 3, 249, Table V; kappa/L units): Ra 1e4: Nu 2.243, u_max
  16.178 at z = 0.823, w_max 19.617 at x = 0.119; Ra 1e5: Nu 4.519, u_max 34.73 at
  0.855, w_max 68.59 at 0.066.
* **Forced convection from an isothermal sphere** (Pr = 0.71), mean Nusselt
  number: Ranz & Marshall (1952) ``2 + 0.6 Re^1/2 Pr^1/3`` (5.785 / 7.353 at Re 50
  / 100); Whitaker (1972) ``2 + (0.4 Re^1/2 + 0.06 Re^2/3) Pr^0.4`` (5.177 /
  6.616; stated maximum deviation 30 %); Yuge (1960), air experiments 10 < Re <
  1.8e3, ``2 + 0.493 Re^1/2`` (5.486 / 6.930) -- these three as compiled by Will,
  Kruyt & Venner (2017, Int. J. Heat Mass Transf. 109, 1059, open access);
  Clift, Grace & Weber (1978) ``1 + (1 + 1/(Re Pr))^1/3 Re^0.41 Pr^1/3`` (5.479 /
  6.922), which Chouippe et al. (2019, arXiv:1902.08255, eq. 19) describe as
  validated against direct numerical simulation for 1 < Re < 400 and 0.25 < Pr <
  100. The Clift and Yuge values agree within 0.2 %: their mean is the reference
  band centre here, the spread of all four the band.
* **Natural convection from an isothermal sphere** in a quiescent fluid: Churchill (1983),
  ``Nu = 2 + 0.589 Ra^1/4 / [1 + (0.469 / Pr)^(9/16)]^(4/9)`` (laminar, Pr >= 0.7; the form
  as reproduced in M. M. Yovanovich's University of Waterloo ME 353 notes, "Natural
  Convection from Isothermal Spheres"): 6.545 at Ra 1e4, Pr 0.71. An empirical fit to experiments in an unbounded fluid; the source states no
  accuracy, so the band run_all applies to it is this suite's choice and says so.
* **Laminar flat plate at uniform heat flux** (Incropera & DeWitt, *Fundamentals of Heat
  and Mass Transfer*, eq. 7.45, Pr >= 0.6): ``Nu_x = 0.453 Re_x^1/2 Pr^1/3``; with the
  length-mean wall excess temperature ``Nu_L = 0.680 Re_L^1/2 Pr^1/3`` (12.12 / 19.17 at
  Re_L 400 / 1000, Pr 0.71). The uniform-flux sphere has no accessible tabulated reference
  here (Dhole, Chhabra & Eswaran 2006 give correlations): it is compared with the
  isothermal band and the isothermal run on the same grid.
* **Archimedes**: a body in a fluid with temperature theta(z) at rest feels
  ``F_z = -Ri integral(theta) dV`` (the Boussinesq perturbation pressure), exactly
  ``-Ri theta_c V`` for a linear profile.
* **Actuator disc** (Froude / Rankine momentum theory, inviscid): a uniformly
  loaded disc with thrust coefficient ``C_T = T / (1/2 rho U^2 A)`` that pushes
  the flow downstream (a propeller or fan) has ``C_T = 4 a (1 + a)``, the axial
  velocity ``(1 + a) U`` at the disc and ``(1 + 2 a) U`` in the far wake; one
  that holds it back (a turbine) ``C_T = 4 a (1 - a)``, ``(1 - a) U`` and ``(1 -
  2 a) U``. In a tunnel of finite cross-section the streamtube model of Garrett &
  Cummins (2007, J. Fluid Mech. 588, 243: continuity, Bernoulli in the wake core
  and the bypass, the channel momentum balance) gives the confined values, which
  tend to Froude's for blockage B = A / A_tunnel -> 0 (:func:`confined_disc_theory`).
* **Laminar round jet** (Schlichting 1933, ZAMM 13, 260; the boundary-layer form of
  the exact Landau-Squire jet; e.g. Schlichting & Gersten, *Boundary-Layer Theory*,
  9th ed. 2017, "axisymmetric free jet"; White, *Viscous Fluid Flow*): a point source
  of kinematic momentum flux K = int u^2 dA in fluid at rest gives
  ``u = u_c / (1 + xi^2 / 4)^2``, ``u_c = 3 K / (8 pi nu x)``, ``xi = sqrt(3 K / (16
  pi)) r / (nu x)``; K is conserved, the volume flow ``Q = 8 pi nu x`` grows
  independently of K, and the half-velocity radius is ``r_1/2 = 2 sqrt(sqrt 2 - 1) nu
  x / sqrt(3 K / (16 pi))``. (Both integrals follow from the profile: int u^2 dA = K,
  int u dA = 8 pi nu x.)
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from aerolume.solvers.ns3d import FlowSolver3D, ImmersedMesh, SolverConfig3D
from aerolume.solvers.ns3d.config import Emitter3D, Fan3D, ThermalConfig3D

#: mean hot-wall Nusselt numbers of the cubic cavity (see the module docstring)
CAVITY_NU = {1e4: {"tric": 2.0542, "wakashima": 2.062, "fusegi": 2.100},
             1e5: {"tric": 4.3371, "wakashima": 4.366, "fusegi": 4.361}}
#: symmetry plane y = 1/2: hot-wall Nu (Tric; Fusegi) and centre-line velocity maxima with positions
#: (free-fall units; Wakashima & Saitoh, Fusegi et al.), see the module docstring
CAVITY_MIDPLANE = {
    1e4: {"nu_tric": 2.25, "nu_fusegi": 2.302,
          "u_max": {"wakashima": (0.198, 0.825), "fusegi": (0.2013, 0.817)},
          "w_max": {"wakashima": (0.222, 0.117), "fusegi": (0.2252, 0.117)}},
    1e5: {"nu_tric": 4.61, "nu_fusegi": 4.646,
          "u_max": {"wakashima": (0.147, 0.85), "fusegi": (0.1468, 0.855)},
          "w_max": {"wakashima": (0.246, 0.068), "fusegi": (0.2471, 0.065)}}}
#: de Vahl Davis (1983) Table V, the square cavity (kappa / L units)
DVD_2D = {1e4: {"nu": 2.243, "u": 16.178, "z_u": 0.823, "w": 19.617, "x_w": 0.119},
          1e5: {"nu": 4.519, "u": 34.73, "z_u": 0.855, "w": 68.59, "x_w": 0.066}}


def nu_ranz_marshall(re: float, pr: float) -> float:
    return 2.0 + 0.6 * re ** 0.5 * pr ** (1.0 / 3.0)


def nu_whitaker(re: float, pr: float) -> float:
    """Whitaker (1972) without the viscosity-ratio factor (constant properties)."""
    return 2.0 + (0.4 * re ** 0.5 + 0.06 * re ** (2.0 / 3.0)) * pr ** 0.4


def nu_yuge(re: float) -> float:
    """Yuge (1960), air (Pr = 0.715), 10 < Re < 1.8e3."""
    return 2.0 + 0.493 * re ** 0.5


def nu_clift(re: float, pr: float) -> float:
    """Clift, Grace & Weber (1978), 1 < Re <= 400."""
    return 1.0 + (1.0 + 1.0 / (re * pr)) ** (1.0 / 3.0) * re ** 0.41 * pr ** (1.0 / 3.0)


def momentum_theory(ct: float, *, propeller: bool = True) -> dict[str, float]:
    """Unconfined Froude momentum theory (U = 1): axial induction a, disc and far-wake
    velocities and the wake / disc area ratio of an ideal, uniformly loaded disc with thrust
    coefficient ``ct`` > 0. ``propeller``: the disc pushes the flow downstream, C_T = 4 a (1 +
    a); otherwise it holds it back (a turbine), C_T = 4 a (1 - a), the light-loading root
    (C_T <= 1)."""
    if not ct > 0:
        raise ValueError("the thrust coefficient must be positive (pick the branch with propeller=)")
    if propeller:
        a = 0.5 * (math.sqrt(1.0 + ct) - 1.0)
        return {"a": a, "u_disc": 1.0 + a, "u_wake": 1.0 + 2.0 * a, "area_ratio_wake": (1.0 + a) / (1.0 + 2.0 * a)}
    if ct > 1.0:
        raise ValueError("momentum theory has no turbine state above C_T = 1")
    a = 0.5 * (1.0 - math.sqrt(1.0 - ct))
    return {"a": a, "u_disc": 1.0 - a, "u_wake": 1.0 - 2.0 * a, "area_ratio_wake": (1.0 - a) / (1.0 - 2.0 * a)}


def confined_disc_theory(ct: float, blockage: float, *, propeller: bool = True) -> dict[str, float]:
    """Quasi-inviscid actuator disc in a tunnel with frictionless walls (Garrett & Cummins
    2007), U = 1 upstream.

    Unknowns: the bypass velocity u_b, the wake-core velocity u_w and the core's share s_w of
    the tunnel section far downstream, where the pressure is uniform. With the signed pressure
    jump dp = +-C_T / 2 (rho U^2 units; + for a propeller, whose force on the fluid points
    downstream) and B = disc area / tunnel section:

    * Bernoulli in the bypass and the core: p_0 - p_w = (u_b^2 - 1) / 2, u_w^2 = u_b^2 + 2 dp;
    * continuity: s_w u_w + (1 - s_w) u_b = 1, disc velocity u_t = s_w u_w / B;
    * momentum of the tunnel: (p_0 - p_w) + dp B = s_w u_w^2 + (1 - s_w) u_b^2 - 1.

    Solved for u_b by bracketing (0 < s_w < 1). B -> 0 recovers :func:`momentum_theory`.
    """
    from scipy.optimize import brentq
    B = float(blockage)
    if not (0.0 < B < 1.0 and ct > 0):
        raise ValueError("need 0 < blockage < 1 and a positive thrust coefficient")
    dp = (0.5 if propeller else -0.5) * float(ct)

    def state(ub: float) -> tuple[float, float]:
        uw = math.sqrt(ub * ub + 2.0 * dp)
        return uw, (1.0 - ub) / (uw - ub)

    def resid(ub: float) -> float:
        uw, sw = state(ub)
        return 0.5 * (ub * ub - 1.0) + dp * B - (sw * uw * uw + (1.0 - sw) * ub * ub - 1.0)

    eps = 1e-12
    if propeller:                          # fast core, slow bypass: sqrt(1 - 2 dp) < u_b < 1
        lo, hi = math.sqrt(max(1.0 - 2.0 * dp, 0.0)) + eps, 1.0 - eps
    else:                                  # slow core, fast bypass: 1 < u_b < sqrt(1 - 2 dp)
        lo, hi = max(1.0, math.sqrt(-2.0 * dp)) + eps, math.sqrt(1.0 - 2.0 * dp) - eps
    ub = brentq(resid, lo, hi, xtol=1e-15)
    uw, sw = state(ub)
    ut = sw * uw / B
    return {"u_disc": ut, "a": abs(ut - 1.0), "u_wake": uw, "u_bypass": ub, "s_wake": sw}


@dataclass
class ThermalResult:
    name: str
    grid: tuple[int, int, int]
    steps: int
    wall_s: float
    values: dict = field(default_factory=dict)

    def row(self) -> str:
        vals = "  ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}" for k, v in self.values.items())
        return f"{self.name:<22} grid {self.grid[0]}x{self.grid[1]}x{self.grid[2]} steps {self.steps:>6d} " \
               f"wall {self.wall_s:7.1f}s  {vals}"


def _sync(s) -> None:
    if s.device.type == "cuda":
        torch.cuda.synchronize(s.device)


# ---------------------------------------------------------------------------
# differentially heated cubic cavity
# ---------------------------------------------------------------------------

def cavity_config(ra: float, n: int, *, pr: float = 0.71, device: str = "auto", dtype: str = "float32",
                  advection: str = "upwind3", diagnostics: bool = True, **kw) -> SolverConfig3D:
    """The unit cube, closed, no-slip everywhere, hot wall x = 0 (theta = +1/2), cold wall
    x = 1 (-1/2), adiabatic elsewhere, gravity along -z. Free-fall scaling: U = sqrt(g beta
    dT L), so Re = sqrt(Ra / Pr), Pe = sqrt(Ra Pr) and Ri = 1."""
    re = math.sqrt(ra / pr)
    th = ThermalConfig3D(prandtl=pr, richardson=1.0,
                         walls={"left": ("temperature", 0.5), "right": ("temperature", -0.5)})
    base = {"nx": n, "ny": n, "nz": n, "h": 1.0 / n, "x0": 0.0, "y0": 0.0, "z0": 0.0, "re": re, "domain": "closed",
            "bc_x": "noslip", "bc_y": "noslip", "bc_zmin": "noslip", "bc_zmax": "noslip", "device": device,
            "dtype": dtype, "dye": False, "thermal": th, "diagnostics": diagnostics, "advection": advection}
    base.update(kw)
    return SolverConfig3D(**base)


def _line_profile(field: torch.Tensor, axis: str, n: int) -> np.ndarray:
    """A cell-centred field on the centre line through y = 1/2 along x or z (averaging the two
    cell rows that straddle the centre when n is even)."""
    f = field.double().cpu().numpy()
    j = (n // 2 - 1, n // 2) if n % 2 == 0 else (n // 2, n // 2)
    k = j
    if axis == "z":                                  # vertical line x = 1/2, y = 1/2
        return 0.25 * sum(f[:, jj, kk] for jj in j for kk in k) if n % 2 == 0 else f[:, j[0], k[0]]
    return 0.25 * sum(f[kk, jj, :] for jj in j for kk in k) if n % 2 == 0 else f[k[0], j[0], :]


def _peak(values: np.ndarray, coords: np.ndarray) -> tuple[float, float]:
    """Maximum of a sampled profile and its position, refined by a parabola through the
    three samples around the largest one."""
    i = int(np.argmax(values))
    if 0 < i < len(values) - 1:
        y0, y1, y2 = values[i - 1], values[i], values[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            d = 0.5 * (y0 - y2) / den
            h = coords[1] - coords[0]
            return float(y1 - 0.25 * (y0 - y2) * d), float(coords[i] + d * h)
    return float(values[i]), float(coords[i])


def cavity_measures(s: FlowSolver3D) -> dict[str, float]:
    """Hot / cold wall mean Nusselt numbers (the scheme's own wall fluxes), the total heat
    flux through the mid-plane x = 1/2 (equal at steady state), the hot-wall Nusselt number
    of the symmetry plane y = 1/2 and the velocity maxima on the centre lines."""
    ops = s.ops
    n = ops.nx
    h = ops.h
    w = s.wall_heat_rates()
    th = s.theta.double()
    st = s.state
    pe = s.physics.thermal.pe
    # mid-plane flux: the scheme's own total (advective + conductive) flux through the faces x = 1/2
    with torch.no_grad():
        fx, _, _ = s.physics.thermal.fluxes(s.theta, st.u, st.v, st.w, ops)
    q_mid = float(fx[..., n // 2].double().sum()) * h * h * pe
    # symmetry-plane local Nu on the hot wall: 2 (theta_w - theta_1) / h at the cells next to x = 0
    j = (n // 2 - 1, n // 2) if n % 2 == 0 else (n // 2,)
    local = np.mean([(2.0 * (0.5 - th[:, jj, 0]) / h).cpu().numpy() for jj in j], axis=0)
    zc = (np.arange(n) + 0.5) * h
    uline = _line_profile(s.field("u"), "z", n)
    wline = _line_profile(s.field("w"), "x", n)
    umax, zu = _peak(uline, zc)
    wmax, xw = _peak(wline, zc)
    return {"nu_hot": w["left"], "nu_cold": -w["right"], "nu_mid": q_mid, "nu_sym_plane": float(local.mean()),
            "u_max": umax, "z_umax": zu, "w_max": wmax, "x_wmax": xw}


def run_cavity(ra: float, n: int, *, t_max: float = 600.0, tol: float = 2e-5, window: float = 20.0,
               chunk: int = 1000, device: str = "auto", **kw) -> ThermalResult:
    """Run the cavity from the conduction profile to a steady state: stop when the hot-wall
    Nusselt number changes by less than ``tol`` (relative) over ``window`` time units, or at
    ``t_max``. Reports the measures of :func:`cavity_measures`, the remaining drift and the
    diagnostics of the last block (continuity residual, relative change per step)."""
    cfg = cavity_config(ra, n, device=device, **kw)
    s = FlowSolver3D(cfg)
    x = (torch.arange(n, device=s.device, dtype=s.dtype) + 0.5) / n
    with torch.no_grad():
        s.theta.copy_((0.5 - x).expand(n, n, n))
    _sync(s)
    t0 = time.perf_counter()
    marks = []
    drift = float("nan")
    while s.t < t_max:
        s.advance(chunk)
        nu = s.wall_heat_rates()["left"]
        marks.append((s.t, nu))
        old = [m for m in marks if m[0] <= s.t - window]
        if old:
            drift = abs(nu - old[-1][1]) / abs(nu)
            if drift < tol:
                break
    _sync(s)
    wall = time.perf_counter() - t0
    vals = cavity_measures(s)
    vals["drift"] = drift
    vals["t_end"] = s.t
    d = s.latest_diagnostics()
    for k in ("div_rms", "poisson_residual", "du_rel", "dtheta_rel", "steps_per_s"):
        vals[k] = d.get(k, float("nan"))
    vals["balance"] = vals["nu_hot"] - vals["nu_cold"]
    return ThermalResult(f"cavity Ra{ra:g}", (n, n, n), s.steps, wall, vals)


def richardson(values: list[float], hs: list[float]) -> dict[str, float]:
    """Observed order and Richardson-extrapolated limit from three grids (finest last)."""
    f1, f2, f3 = values[-3:]
    h1, h2, h3 = hs[-3:]
    r = h1 / h2
    if abs(f2 - f3) < 1e-14 or (f1 - f2) / (f2 - f3) <= 0:
        return {"order": float("nan"), "limit": float("nan"), "gci_fine": float("nan")}
    p = math.log(abs((f1 - f2) / (f2 - f3))) / math.log(r)
    r23 = h2 / h3
    limit = f3 + (f3 - f2) / (r23 ** p - 1.0)
    gci = 1.25 * abs((f3 - f2) / f3) / (r23 ** p - 1.0)
    return {"order": p, "limit": limit, "gci_fine": gci}


def run_cavity_2d(ra: float, n: int, *, t_max: float = 600.0, tol: float = 2e-5, window: float = 20.0,
                  chunk: int = 1000, device: str = "auto", **kw) -> ThermalResult:
    """The 3D solver in its 2D limit: the cavity four cells deep in y between slip, adiabatic
    walls (the flow is then exactly two-dimensional), against de Vahl Davis (1983) -- a check
    of the centre-line velocity maxima independent of the 3D references. Converged like
    :func:`run_cavity`; velocities in free-fall units (``DVD_2D`` values / sqrt(Ra Pr))."""
    cfg = cavity_config(ra, n, ny=4, bc_y="slip", device=device, **kw)
    s = FlowSolver3D(cfg)
    x = (torch.arange(n, device=s.device, dtype=s.dtype) + 0.5) / n
    s.set_temperature((0.5 - x).expand(n, 4, n).clone())
    _sync(s)
    t0 = time.perf_counter()
    marks, drift = [], float("nan")
    strip = 4.0 / n                                      # the wall area of the 4h-deep strip
    while s.t < t_max:
        s.advance(chunk)
        nu = s.wall_heat_rates()["left"] / strip
        marks.append((s.t, nu))
        old = [m for m in marks if m[0] <= s.t - window]
        if old:
            drift = abs(nu - old[-1][1]) / abs(nu)
            if drift < tol:
                break
    _sync(s)
    wall = time.perf_counter() - t0
    zc = (np.arange(n) + 0.5) / n
    u = s.field("u").double().cpu().numpy()[:, 0, :]
    w = s.field("w").double().cpu().numpy()[:, 0, :]
    umax, zu = _peak(0.5 * (u[:, n // 2 - 1] + u[:, n // 2]), zc)
    wmax, xw = _peak(0.5 * (w[n // 2 - 1, :] + w[n // 2, :]), zc)
    ff = math.sqrt(ra * 0.71)
    ref = DVD_2D[ra]
    w_ = s.wall_heat_rates()
    vals = {"nu_hot": w_["left"] / strip, "nu_cold": -w_["right"] / strip, "nu_ref": ref["nu"],
            "u_max": umax, "u_ref": ref["u"] / ff, "z_umax": zu, "z_ref": ref["z_u"],
            "w_max": wmax, "w_ref": ref["w"] / ff, "x_wmax": xw, "x_ref": ref["x_w"], "drift": drift, "t_end": s.t}
    vals["nu_err"] = vals["nu_hot"] / ref["nu"] - 1.0
    vals["u_err"] = umax / vals["u_ref"] - 1.0
    vals["w_err"] = wmax / vals["w_ref"] - 1.0
    return ThermalResult(f"square cavity Ra{ra:g}", (n, 4, n), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# buoyancy on a body: Archimedes in a stratified fluid at rest
# ---------------------------------------------------------------------------

def run_archimedes(n: int, *, d: float = 0.4, zc: float = 0.62, ri: float = 5.0, steps: int = 100,
                   device: str = "cpu", dtype: str = "float64") -> ThermalResult:
    """A fixed insulating sphere (diameter ``d``, centre height ``zc``) in a closed unit box with
    a linear stratification theta = z - 1/2 between isothermal top and bottom walls, in a
    non-conducting fluid (Pr = 1e6, so the stratification is an equilibrium): the fluid must
    stay at rest and the body feel exactly the Archimedes force ``-Ri theta_c V``."""
    import ns3d_cases as cases
    v, f = cases.icosphere(d, 3, (0.5, 0.5, zc))
    body = ImmersedMesh(v, f, name="sphere", thermal="adiabatic")
    th = ThermalConfig3D(prandtl=1e6, richardson=ri,
                         walls={"bottom": ("temperature", -0.5), "top": ("temperature", 0.5)})
    cfg = SolverConfig3D(nx=n, ny=n, nz=n, h=1.0 / n, x0=0.0, y0=0.0, z0=0.0, re=50.0, domain="closed",
                         bc_zmin="noslip", bc_zmax="noslip", device=device, dtype=dtype, dye=False, bodies=[body],
                         thermal=th)
    s = FlowSolver3D(cfg)
    z = (np.arange(n) + 0.5) / n - 0.5
    s.set_temperature(np.broadcast_to(z[:, None, None], (n, n, n)).copy())
    t0 = time.perf_counter()
    s.advance(steps)
    _sync(s)
    wall = time.perf_counter() - t0
    fz = float(s.coefficients()[2]) * 0.5 * cfg.ref_area
    ref = -ri * (zc - 0.5) * cases.mesh_volume(v, f)
    vals = {"cells_per_d": d * n, "fz": fz, "fz_ref": ref, "rel_err": fz / ref - 1.0,
            "max_speed": float(s.field("speed").max()), "t_end": s.t}
    return ThermalResult(f"Archimedes n{n}", (n, n, n), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# heated sphere, forced convection
# ---------------------------------------------------------------------------

def heated_sphere_config(re: float, cells_per_d: int, *, pr: float = 0.71, device: str = "auto",
                         dtype: str = "float32", mode: str = "temperature", value: float = 1.0,
                         richardson: float = 0.0, **kw) -> SolverConfig3D:
    """The drag-validation sphere (ns3d_cases.sphere_config: 2.85 % blockage, half domain)
    held at theta = ``value`` (``mode`` "temperature") or releasing the uniform flux G =
    ``value`` (``mode`` "heat_flux") in a free stream at theta = 0; ``richardson`` 0 is
    forced convection (gravity along -z otherwise)."""
    import ns3d_cases as cases  # the validation sibling (tests and bench put it on sys.path)
    cfg = cases.sphere_config(re, cells_per_d, device=device, dtype=dtype, **kw)
    b = cfg.bodies[0]
    body = ImmersedMesh(b.vertices, b.faces, name="sphere", thermal=mode, thermal_value=float(value))
    th = ThermalConfig3D(prandtl=pr, richardson=float(richardson))
    return dataclasses.replace(cfg, bodies=[body], thermal=th, dye=False, diagnostics=True)


def run_heated_sphere(re: float, cells_per_d: int, t_end: float = 40.0, *, pr: float = 0.71, chunk: int = 250,
                      **kw) -> ThermalResult:
    """Steady forced convection: Nu over the last quarter of the run (from the penalisation's
    heat), its drift, the heat carried out of the domain (energy balance), the drag and the
    surface-probe Nusselt number."""
    cfg = heated_sphere_config(re, cells_per_d, pr=pr, **kw)
    s = FlowSolver3D(cfg)
    _sync(s)
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    ph = s.physics_history()
    m = ph["t"] >= 0.75 * t_end
    nu = ph["nu"][m]
    half = len(nu) // 2
    hb = s.heat_balance()
    surf = s.surface()[0]
    v, f = surf["vertices"], surf["faces"]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    q_face = surf["heat_flux"][f].mean(axis=1)
    nu_surf = float((q_face * area).sum() / area.sum())
    hist = s.history()
    mh = hist["t"] >= 0.75 * t_end
    vals = {"nu": float(nu.mean()), "nu_drift": float(abs(nu[half:].mean() - nu[:half].mean()) / nu.mean()),
            "nu_surface": nu_surf, "heat_out": hb["heat_out_outlet"] - hb["heat_in_inlet"],
            "heat_body": hb["heat_in_bodies"], "cd": float(hist["cd"][mh].mean()),
            "nu_clift": nu_clift(re, pr), "nu_whitaker": nu_whitaker(re, pr), "nu_ranz": nu_ranz_marshall(re, pr),
            "nu_yuge": nu_yuge(re), "t_end": s.t}
    vals["balance_ratio"] = vals["heat_out"] / vals["heat_body"]
    vals["steps_per_s"] = s.steps / wall
    return ThermalResult(f"heated sphere Re{re:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# uniform heat flux: flat plate and sphere (the "heat_flux" body condition)
# ---------------------------------------------------------------------------

#: Laminar flat plate with a uniform surface heat flux (Incropera & DeWitt, *Fundamentals of
#: Heat and Mass Transfer*, eq. 7.45, Pr >= 0.6): Nu_x = 0.453 Re_x^1/2 Pr^1/3. With the
#: mean wall excess temperature over the length L (theta_w ~ x^1/2 averages to 2/3 of its
#: trailing-edge value) the mean is Nu_L = (3/2) 0.453 Re_L^1/2 Pr^1/3 = 0.680 Re_L^1/2 Pr^1/3;
#: the isothermal plate's 0.664 for comparison (eq. 7.30).
PLATE_FLUX_LOCAL = 0.453


def nu_plate_uniform_flux(re: float, pr: float) -> float:
    """Mean Nusselt number of a laminar flat plate at uniform heat flux, defined with the
    length-mean wall excess temperature (``1.5 * 0.453 Re^1/2 Pr^1/3``)."""
    return 1.5 * PLATE_FLUX_LOCAL * math.sqrt(re) * pr ** (1.0 / 3.0)


def plate_mesh(thickness: float, span: tuple[float, float], *, nose: float = 2.0, n_arc: int = 24):
    """A plate of chord 1 (x in [0, 1]) and thickness ``thickness`` centred on z = 0, extruded
    over y in ``span``: semi-elliptic nose and tail of length ``nose * thickness`` (a blunt
    nose would separate and add a heat-transfer peak that no plate correlation contains).
    Returns (vertices, faces) with outward-facing triangles."""
    a, b = nose * thickness, 0.5 * thickness
    phi_n = np.linspace(0.5 * math.pi, 1.5 * math.pi, n_arc)
    phi_t = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_arc)
    prof = np.r_[np.c_[a + a * np.cos(phi_n), b * np.sin(phi_n)],
                 np.c_[1.0 - a + a * np.cos(phi_t), b * np.sin(phi_t)]]
    n = len(prof)
    y0, y1 = float(span[0]), float(span[1])
    ring0 = np.c_[prof[:, 0], np.full(n, y0), prof[:, 1]]
    ring1 = np.c_[prof[:, 0], np.full(n, y1), prof[:, 1]]
    c = prof.mean(axis=0)
    verts = np.r_[ring0, ring1, [[c[0], y0, c[1]], [c[0], y1, c[1]]]]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    faces = np.asarray(faces, dtype=np.int64)
    # orient every triangle outwards (the plate is convex in x-z and a prism in y)
    centre = np.array([c[0], 0.5 * (y0 + y1), c[1]])
    tri = verts[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.sum(nrm * (tri.mean(axis=1) - centre), axis=1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return verts.astype(np.float64), faces


def flux_plate_config(re: float, cells_per_l: int, *, thickness: float = 0.1, pr: float = 0.71, flux: float = 1.0,
                      upstream: float = 1.0, downstream: float = 2.5, half_height: float = 1.5, ny: int = 4,
                      device: str = "auto", dtype: str = "float32") -> SolverConfig3D:
    """A plate of chord 1 releasing the uniform flux G = ``flux`` (conduction units) in a free
    stream, quasi-2D: the span is periodic, ``ny`` cells wide, and the plate runs through both
    periodic faces (its end caps lie outside the domain, so every in-domain skin cell belongs
    to the top, bottom, nose or tail). Slip walls ``half_height`` above and below (3.3 %
    blockage at t = 0.1), inlet ``upstream`` chords ahead of the nose. Ri = 0."""
    h = 1.0 / cells_per_l
    nx = round((upstream + 1.0 + downstream) / h)
    nz = 2 * round(half_height / h)
    ylo, yhi = 0.0, ny * h
    v, f = plate_mesh(thickness, (ylo - 6 * h, yhi + 6 * h))
    body = ImmersedMesh(v, f, name="plate", thermal="heat_flux", thermal_value=float(flux))
    return SolverConfig3D(nx=nx, ny=ny, nz=nz, h=h, x0=-upstream, y0=ylo, z0=-0.5 * nz * h, re=re, bodies=[body],
                          bc_y="periodic", ref_area=yhi - ylo, length_ref=1.0, span_ref=yhi - ylo,
                          moment_ref=(0.25, 0.5 * yhi, 0.0), device=device, dtype=dtype, perturb=False,
                          thermal=ThermalConfig3D(prandtl=pr, richardson=0.0), dye=False)


def _theta_at(s: FlowSolver3D, pts: np.ndarray) -> np.ndarray:
    from aerolume.solvers.ns3d.fields import trilinear
    q = torch.as_tensor(np.ascontiguousarray(pts), device=s.device, dtype=s.dtype)
    with torch.no_grad():
        return trilinear(s._theta, q, s.ops, (0.5, 0.5, 0.5)).double().cpu().numpy()


def _probe_wall_theta(s: FlowSolver3D, pts: np.ndarray, nrm: np.ndarray) -> np.ndarray:
    """The pre-fix wall temperature of a heat-flux body: theta 1.5 h and 2.5 h off the wall
    along the outward normal, extrapolated linearly to the wall (kept to show its bias)."""
    from aerolume.solvers.ns3d.fields import probe_distances
    d1, d2 = probe_distances(s.ops)
    t1, t2 = _theta_at(s, pts + d1 * nrm), _theta_at(s, pts + d2 * nrm)
    return t1 + (t1 - t2) * d1 / (d2 - d1)


def run_flux_plate(re: float, cells_per_l: int, t_end: float = 8.0, *, thickness: float = 0.1, pr: float = 0.71,
                   chunk: int = 200, **kw) -> ThermalResult:
    """Laminar plate at uniform heat flux: the mean Nusselt number G / theta_w (theta_w the
    solver's measured wall temperature, ``ThermalGeometry.wall_theta``) over the last
    quarter of the run, its drift, the surface-sampled and the pre-fix probe-extrapolated
    values, the local wall temperature along the top against the similarity solution
    (x from 0.2 to 0.8 of the chord) and the energy balance."""
    cfg = flux_plate_config(re, cells_per_l, thickness=thickness, pr=pr, **kw)
    s = FlowSolver3D(cfg)
    _sync(s)
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    ph = s.physics_history()
    m = ph["t"] >= 0.75 * t_end
    tw = ph["theta_wall_0"][m]
    half = len(tw) // 2
    hb = s.heat_balance()
    # the side faces (normal in x-z), sampled at mid-span: the mesh's vertices lie outside the
    # periodic domain, and the end caps are not part of the in-domain plate
    v, f = cfg.bodies[0].vertices_numpy(), np.asarray(cfg.bodies[0].faces, dtype=np.int64)
    tri = v[f]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    side = np.abs(cr[:, 1]) < 1e-9 * (2.0 * area + 1e-300)
    mid = cfg.y0 + 0.5 * cfg.ny * cfg.h
    fc = tri.mean(axis=1)[side]
    fc[:, 1] = mid
    nf = cr[side] / np.linalg.norm(cr[side], axis=1, keepdims=True)
    a_s = area[side]
    # the flux actually released per unit of in-domain area: the skin is normalised over the
    # whole mesh, end caps outside the periodic domain included, so the in-domain release
    # differs from the set G by the discrete delta's orientation dependence (-7 to +0.2 %
    # over 32-128 cells per chord); the Nusselt number uses what the fluid received
    span = cfg.ny * cfg.h
    a_in = a_s.sum() / (span + 12.0 * cfg.h) * span
    G = float(s._phys.thermal.geom.flux_rate[0]) / a_in
    tw_surface = float((_theta_at(s, fc) * a_s).sum() / a_s.sum())
    tw_probe = float((_probe_wall_theta(s, fc, nf) * a_s).sum() / a_s.sum())
    # local wall temperature along the top against theta_w(x) = G x / Nu_x
    xs = np.linspace(0.2, 0.8, 13)
    t_loc = _theta_at(s, np.c_[xs, np.full_like(xs, mid), np.full_like(xs, 0.5 * thickness)])
    t_sim = G * np.sqrt(xs) / (PLATE_FLUX_LOCAL * math.sqrt(re) * pr ** (1.0 / 3.0))
    tw_mean = float(tw.mean())
    ref = nu_plate_uniform_flux(re, pr)
    vals = {"nu": G / tw_mean, "nu_ref": ref, "nu_err": G / tw_mean / ref - 1.0,
            "flux_ratio": G / float(cfg.bodies[0].thermal_value),
            "nu_isothermal_ref": 0.664 * math.sqrt(re) * pr ** (1.0 / 3.0),
            "tw_drift": float(abs(tw[half:].mean() - tw[:half].mean()) / tw_mean),
            "nu_surface": G / tw_surface, "nu_probe": G / tw_probe if tw_probe > 0 else float("inf"),
            "local_ratio_mean": float(np.mean(t_loc / t_sim)), "local_ratio_spread": float(np.ptp(t_loc / t_sim)),
            "balance_ratio": (hb["heat_out_outlet"] - hb["heat_in_inlet"]) / hb["heat_in_bodies"],
            "t_end": s.t, "cells_across_bl": 5.0 / math.sqrt(re) * cells_per_l}
    vals["steps_per_s"] = s.steps / wall
    return ThermalResult(f"flux plate Re{re:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


def run_flux_sphere(re: float, cells_per_d: int, t_end: float = 40.0, *, pr: float = 0.71, chunk: int = 250,
                    **kw) -> ThermalResult:
    """The validation sphere releasing a uniform heat flux G = 1: Nu = G / theta_w with the
    measured wall temperature (last quarter of the run), next to the isothermal sphere's
    correlations and the pre-fix probe-extrapolated value. An isoflux sphere's
    mean-temperature Nusselt number is close to the isothermal one in this range, so the
    correlation band is a sanity range here, not a reference value (the plate is that)."""
    cfg = heated_sphere_config(re, cells_per_d, pr=pr, mode="heat_flux", value=1.0, **kw)
    s = FlowSolver3D(cfg)
    _sync(s)
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    ph = s.physics_history()
    m = ph["t"] >= 0.75 * t_end
    tw = ph["theta_wall_0"][m]
    half = len(tw) // 2
    surf = s.surface()[0]
    v, f = surf["vertices"], surf["faces"]
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    tw_surface = float((surf["theta_wall"][f].mean(axis=1) * area).sum() / area.sum())
    low = v[:, 1] < s.ops.y0                                  # the missing half: sample the mirror image
    vm, nm = v.copy(), surf["normals"].copy()
    vm[low, 1] = 2.0 * s.ops.y0 - vm[low, 1]
    nm[low, 1] = -nm[low, 1]
    tw_probe = float((_probe_wall_theta(s, vm, nm)[f].mean(axis=1) * area).sum() / area.sum())
    hb = s.heat_balance()
    vals = {"nu": 1.0 / float(tw.mean()), "nu_surface": 1.0 / tw_surface,
            "nu_probe": 1.0 / tw_probe if tw_probe > 0 else float("inf"),
            "tw_drift": float(abs(tw[half:].mean() - tw[:half].mean()) / tw.mean()),
            "nu_history": float(np.mean(ph["nu"][m])),
            "nu_clift": nu_clift(re, pr), "nu_whitaker": nu_whitaker(re, pr), "nu_ranz": nu_ranz_marshall(re, pr),
            "nu_yuge": nu_yuge(re),
            "balance_ratio": (hb["heat_out_outlet"] - hb["heat_in_inlet"]) / hb["heat_in_bodies"], "t_end": s.t}
    vals["steps_per_s"] = s.steps / wall
    return ThermalResult(f"flux sphere Re{re:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


def run_buoyant_sphere_steps(re: float, ri: float, cells_per_d: int, t_plume: float = 20.0, *, pr: float = 0.71,
                             forced_speed: float = 2.1, chunk: int = 50, **kw) -> ThermalResult:
    """A heated sphere (theta = 1) where buoyancy dominates (Ri >> 1): run ``t_plume`` buoyant
    times L / U_b (U_b = U sqrt(Ri), gravity along -z) and compare the steps the solver took
    with the planning rule ``t / thermal_dt_estimate(h, planning_speed_sum(Ri, forced), Pe)``
    (``ns3d.physics``). Also the late-time speed sum in U_b units (the second half of the run),
    which :data:`PLUME_SPEED_FACTOR` summarises."""
    from aerolume.solvers.ns3d.physics import planning_speed_sum, thermal_dt_estimate
    cfg = heated_sphere_config(re, cells_per_d, pr=pr, richardson=ri, **kw)
    s = FlowSolver3D(cfg)
    ub = math.sqrt(abs(ri))
    t_end = t_plume / max(ub, 1.0)
    late = []
    _sync(s)
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
        if s.t * max(ub, 1.0) > 0.5 * t_plume:
            st = s.state
            late.append(float(st.u.abs().max() + st.v.abs().max() + st.w.abs().max()) / max(ub, 1.0))
    _sync(s)
    wall = time.perf_counter() - t0
    dt_plan = thermal_dt_estimate(cfg.h, planning_speed_sum(ri, forced_speed), re * pr)
    steps_plan = s.t / dt_plan
    vals = {"ri": float(ri), "t_end": s.t, "mean_dt": s.t / s.steps, "dt_plan": dt_plan,
            "steps_ratio": s.steps / steps_plan, "speed_sum_late": float(np.mean(late)) if late else float("nan"),
            "free_fall_dt": cfg.cfl * math.sqrt(cfg.h / abs(ri)) if ri else float("inf")}
    vals["steps_per_s"] = s.steps / wall
    return ThermalResult(f"buoyant sphere Ri{ri:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# heated sphere, natural convection in a closed box (critic round 3.4: the hot sphere)
# ---------------------------------------------------------------------------

def nu_churchill_sphere(ra: float, pr: float) -> float:
    """Churchill (1983): mean Nusselt number of an isothermal sphere in a quiescent fluid,
    ``2 + 0.589 Ra^1/4 / [1 + (0.469 / Pr)^(9/16)]^(4/9)`` (laminar, Pr >= 0.7). 6.545 at Ra 1e4,
    Pr 0.71. Its conduction limit 2 is the sphere in an unbounded fluid."""
    return 2.0 + 0.589 * ra ** 0.25 / (1.0 + (0.469 / pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)


#: the closed box of :func:`natural_sphere_config` in sphere diameters: half-width across (x and y),
#: depth below the centre and height above it (the plume rises 6 D before it meets the lid)
NATURAL_BOX = {"half_width": 3.0, "below": 3.0, "above": 6.0}


def natural_sphere_config(ra: float, cells_per_d: int, *, pr: float = 0.71, level: int = 4, device: str = "auto",
                          dtype: str = "float32", **kw) -> SolverConfig3D:
    """An isothermal sphere of diameter 1 (theta = 1) at the origin in a closed box of fluid
    at rest, x in [-3, 3], y in [0, 3] (symmetry plane y = 0; the far y wall is slip), z in
    [-3, 6]; every wall is held at the ambient theta = 0, so the heat the sphere gives off
    leaves through them and the run reaches a steady state; x and z walls are no-slip;
    gravity along -z. Free-fall scaling U = sqrt(g beta dT D): Re = sqrt(Ra / Pr), Ri = 1.

    Every extent is a whole number of cells at 8, 12, 18, 24 and 27 cells/D, and the sphere's
    centre sits on cell faces, so a grid study changes the resolution only, never the box.
    The box is a confinement Churchill's unbounded fluid does not have: the run reports the
    fluid temperature beside the sphere (:func:`run_natural_sphere`) so its effect is visible."""
    import ns3d_cases as cases  # the validation sibling (tests and run_all put it on sys.path)
    h = 1.0 / cells_per_d
    nw = round(NATURAL_BOX["half_width"] / h)
    nb, na = round(NATURAL_BOX["below"] / h), round(NATURAL_BOX["above"] / h)
    v, f = cases.icosphere(1.0, level)
    body = ImmersedMesh(v, f, name="sphere", thermal="temperature", thermal_value=1.0)
    walls = {side: ("temperature", 0.0) for side in ("left", "right", "back", "bottom", "top")}
    th = ThermalConfig3D(prandtl=pr, richardson=1.0, walls=walls)
    base = {"nx": 2 * nw, "ny": nw, "nz": nb + na, "h": h, "x0": -nw * h, "y0": 0.0, "z0": -nb * h,
            "re": math.sqrt(ra / pr), "domain": "closed", "bc_x": "noslip", "bc_y": "symmetry_ymin",
            "bc_zmin": "noslip", "bc_zmax": "noslip", "bodies": [body], "thermal": th, "ref_area": math.pi / 4.0,
            "length_ref": 1.0, "span_ref": 1.0, "moment_ref": (0.0, 0.0, 0.0), "device": device, "dtype": dtype,
            "dye": False, "diagnostics": True, "perturb": False}
    base.update(kw)
    return SolverConfig3D(**base)


def run_natural_sphere(ra: float, cells_per_d: int, *, pr: float = 0.71, t_max: float = 400.0, t_min: float = 60.0,
                       window: float = 30.0, tol: float = 2e-3, chunk: int = 200, **kw) -> ThermalResult:
    """Run :func:`natural_sphere_config` from rest to a steady state: stop once the sphere's
    Nusselt number has changed by less than ``tol`` (relative) over the last ``window`` free-fall
    times and ``t_min`` has passed, or at ``t_max``. Reports:

    * ``nu``: Q / (A theta_s), the mean over the last ``window`` (from the penalisation's heat,
      the solver's own Nusselt number) and its ``nu_drift`` across that window's two halves;
    * ``balance_ratio``: the heat leaving through the walls over the heat from the sphere (1 at
      a steady state; the remainder is still being stored);
    * ``theta_side``: the mean fluid temperature beside the sphere (centre height, |z| < 1/2, at
      1.5-2.5 D from the axis), the box's warming of the "ambient", and ``nu_local`` = Nu
      rescaled to it, Q / (A (theta_s - theta_side));
    * ``nu_churchill`` and the plume's peak speed ``w_max`` (free-fall units)."""
    cfg = natural_sphere_config(ra, cells_per_d, pr=pr, **kw)
    s = FlowSolver3D(cfg)
    _sync(s)
    t0 = time.perf_counter()
    marks: list[tuple[float, float]] = []
    drift = float("nan")
    while s.t < t_max:
        s.advance(chunk)
        nu = s.heat_rates()[0]["nusselt"]
        marks.append((s.t, nu))
        old = [m for m in marks if m[0] <= s.t - window]
        if old and s.t >= t_min:
            drift = abs(nu - old[-1][1]) / abs(nu)
            if drift < tol:
                break
    _sync(s)
    wall = time.perf_counter() - t0
    ph = s.physics_history()
    m = ph["t"] >= s.t - window
    nu_w = ph["nu"][m]
    half = len(nu_w) // 2
    hb = s.heat_balance()
    q_body = hb["heat_in_bodies"]
    th = s.field("temperature").double().cpu().numpy()                 # (z, y, x)
    hh = cfg.h
    xc = cfg.x0 + (np.arange(cfg.nx) + 0.5) * hh
    yc = cfg.y0 + (np.arange(cfg.ny) + 0.5) * hh
    zc = cfg.z0 + (np.arange(cfg.nz) + 0.5) * hh
    zz, yy, xx = np.meshgrid(zc, yc, xc, indexing="ij")
    ring = (np.abs(zz) < 0.5) & (np.hypot(xx, yy) > 1.5) & (np.hypot(xx, yy) < 2.5)
    theta_side = float(th[ring].mean())
    nu_mean = float(nu_w.mean())
    vals = {"ra": float(ra), "cells_per_d": float(cells_per_d), "nu": nu_mean,
            "nu_drift": float(abs(nu_w[half:].mean() - nu_w[:half].mean()) / nu_mean) if half else float("nan"),
            "drift_window": drift, "balance_ratio": -hb["heat_in_walls"] / q_body if q_body else float("nan"),
            "theta_side": theta_side, "nu_local": nu_mean / (1.0 - theta_side),
            "nu_churchill": nu_churchill_sphere(ra, pr), "w_max": float(s.field("w").max()), "t_end": s.t,
            "steps_per_s": s.steps / wall}
    vals["nu_err"] = vals["nu"] / vals["nu_churchill"] - 1.0
    return ThermalResult(f"natural sphere Ra{ra:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# actuator disc
# ---------------------------------------------------------------------------

#: half-width of the actuator-disc tunnel in diameters: a square section of side 6 D, so the
#: blockage is (pi / 4) / 36 = 2.18 %
DISC_HALF_WIDTH = 3.0


def disc_blockage(half_width: float = DISC_HALF_WIDTH) -> float:
    """Disc area over the tunnel section for a disc of diameter 1 in a square tunnel."""
    return (math.pi / 4.0) / (2.0 * half_width) ** 2


def actuator_disc_config(ct: float, cells_per_d: int = 16, *, propeller: bool = True, re: float = 400.0,
                         mode: str = "thrust", half: bool = True, upstream: float = 4.0, downstream: float = 8.0,
                         half_width: float = DISC_HALF_WIDTH, device: str = "auto", dtype: str = "float32",
                         **kw) -> SolverConfig3D:
    """A disc of diameter 1 at the origin, normal to the free stream, in a square slip-walled
    tunnel of side 2 ``half_width``; laminar, no bodies, half domain with the symmetry plane
    through the disc axis (``half``).

    ``propeller``: the fan blows along +x (it pushes the flow; C_T = 4 a (1 + a) for an
    unconfined disc); otherwise along -x (a turbine-like momentum sink). Mode "thrust": T = C_T
    (1/2) A; mode "velocity": regulated to the Froude disc velocity of that C_T."""
    h = 1.0 / cells_per_d
    nx = round((upstream + downstream) / h)
    nw = round(half_width / h)
    area = math.pi / 4.0
    sign = 1.0 if propeller else -1.0
    value = ct * 0.5 * area if mode == "thrust" else sign * momentum_theory(ct, propeller=propeller)["u_disc"]
    fan = Fan3D(center=(0.0, 0.0, 0.0), direction=(sign, 0.0, 0.0), diameter=1.0, mode=mode, value=value,
                name="disc")
    common = {"nx": nx, "nz": 2 * nw, "h": h, "x0": -upstream, "z0": -nw * h, "re": re, "bodies": [], "fans": [fan],
              "device": device, "dtype": dtype, "dye": False, "perturb": False, "diagnostics": True}
    common.update(kw)
    if half:
        return SolverConfig3D(ny=nw, y0=0.0, bc_y="symmetry_ymin", **common)
    return SolverConfig3D(ny=2 * nw, y0=-nw * h, **common)


def run_actuator_disc(ct: float, cells_per_d: int = 16, t_end: float = 30.0, *, propeller: bool = True,
                      chunk: int = 250, half_width: float = DISC_HALF_WIDTH, **kw) -> ThermalResult:
    """Disc velocity, delivered thrust and the axial velocity on the axis 2 and 4 diameters
    behind the disc, averaged over the last third of the run, against confined
    (:func:`confined_disc_theory`) and unconfined (:func:`momentum_theory`) momentum theory.

    Velocities are along +x whatever the fan's direction (the fan report's disc velocity is
    along the fan axis and is turned here), ``a`` = |u_disc - 1|; ``err_confined`` is the disc
    velocity's relative error against the confined theory, the like-for-like reference."""
    from aerolume.solvers.ns3d.fields import trilinear
    cfg = actuator_disc_config(ct, cells_per_d, propeller=propeller, half_width=half_width, **kw)
    s = FlowSolver3D(cfg)
    _sync(s)
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    ph = s.physics_history()
    m = ph["t"] >= (2.0 / 3.0) * t_end
    nx_ = cfg.fans[0].direction[0]
    series = ph["fan0_velocity"][m] * nx_
    ud = float(series.mean())
    thrust = float(ph["fan0_thrust"][m].mean())
    area = math.pi / 4.0
    pts = torch.tensor([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], device=s.device, dtype=s.dtype)
    with torch.no_grad():
        uw = trilinear(s.state.u, pts, s.ops, (0.0, 0.5, 0.5)).double().cpu().numpy()
    blockage = disc_blockage(half_width)
    free = momentum_theory(ct, propeller=propeller)
    conf = confined_disc_theory(ct, blockage, propeller=propeller)
    ct_meas = thrust / (0.5 * area)
    a = abs(ud - 1.0)
    vals = {"propeller": float(propeller), "blockage": blockage, "ct_set": ct, "ct": ct_meas, "u_disc": ud,
            "u_disc_confined": conf["u_disc"], "u_disc_theory": free["u_disc"],
            "err_confined": ud / conf["u_disc"] - 1.0, "a": a, "a_confined": conf["a"], "a_theory": free["a"],
            "ct_from_a": 4.0 * a * (1.0 + a) if propeller else 4.0 * a * (1.0 - a),
            "u_axis_2d": float(uw[0]), "u_axis_4d": float(uw[1]), "u_wake_confined": conf["u_wake"],
            "u_wake_theory": free["u_wake"], "drift": float(abs(series[-1] - series[0])), "t_end": s.t,
            "steps_per_s": s.steps / wall}
    kind = "propeller" if propeller else "turbine"
    return ThermalResult(f"actuator disc {kind} CT{ct:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# laminar round jet (a jet port in a closed box)
# ---------------------------------------------------------------------------

#: xi at u = u_c / 2 in Schlichting's profile (1 + xi^2 / 4)^-2: 2 sqrt(sqrt(2) - 1)
XI_HALF = 2.0 * math.sqrt(math.sqrt(2.0) - 1.0)


#: share of Schlichting's volume flow inside r = 2 r_1/2: 1 - 1 / (1 + (2 xi_1/2)^2 / 4)
Q_SHARE_2R = 1.0 - 1.0 / (1.0 + XI_HALF ** 2)


def schlichting_jet(k: float, nu: float) -> dict[str, float]:
    """Slopes of the laminar round jet's similarity laws for momentum flux ``k`` and viscosity
    ``nu``: d(1/u_c)/dx, dQ/dx (the whole jet, and ``q2_slope`` inside r = 2 r_1/2) and
    d(r_1/2)/dx -- all independent of the virtual origin."""
    s = math.sqrt(3.0 * k / (16.0 * math.pi))
    return {"inv_uc_slope": 8.0 * math.pi * nu / (3.0 * k), "q_slope": 8.0 * math.pi * nu,
            "q2_slope": Q_SHARE_2R * 8.0 * math.pi * nu, "r_half_slope": XI_HALF * nu / s}


def round_jet_config(re_j: float = 100.0, cells_per_d: int = 4, *, length: float = 32.0, half_width: float = 12.0,
                     device: str = "auto", dtype: str = "float32", **kw) -> SolverConfig3D:
    """A jet port of diameter 1 and speed 1 (Re_j = 1 / nu) at x = 2 blowing along +x into a
    closed no-slip box ``length`` long and 2 ``half_width`` square, balanced by a wide, slow,
    two-diameter-thick vent (diameter 2 half_width - 4) whose footprint lies wholly inside the
    box near its far end (a footprint cut by the wall would leave part of the sink to the
    pressure solve's uniform compatibility correction). An odd number of cells across puts the
    jet axis on a row of cell centres, so the centre line is sampled without interpolation.
    The volume flow is exact by construction (the port is a mass source, theory doc 17)."""
    from aerolume.solvers.ns3d.config import Port3D
    h = 1.0 / cells_per_d
    nx = round(length / h)
    nc = 2 * round(half_width / h) + 1
    q = math.pi / 4.0
    vent_d = 2.0 * half_width - 4.0
    jet = Port3D(kind="jet", center=(2.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), diameter=1.0, speed=1.0, name="jet")
    vent = Port3D(kind="vent", center=(length - 2.5, 0.0, 0.0), direction=(1.0, 0.0, 0.0), diameter=vent_d,
                  thickness=2.0, speed=q / (math.pi / 4.0 * vent_d ** 2), name="vent")
    base = {"nx": nx, "ny": nc, "nz": nc, "h": h, "x0": 0.0, "y0": -0.5 * nc * h, "z0": -0.5 * nc * h, "re": re_j,
            "domain": "closed", "bc_x": "noslip", "bc_y": "noslip", "bc_zmin": "noslip", "bc_zmax": "noslip",
            "ports": [jet, vent], "device": device, "dtype": dtype, "dye": False, "perturb": False,
            "diagnostics": True}
    base.update(kw)
    return SolverConfig3D(**base)


def _jet_core(sec: np.ndarray, jc: int, kc: int) -> np.ndarray:
    """The forward-flow region (u > 0) connected to the axis cell (4-connectivity): the jet,
    without any forward flow of the box's recirculation further out."""
    from scipy.ndimage import label
    lab, _ = label(sec > 0)
    return lab == lab[kc, jc] if lab[kc, jc] > 0 else np.zeros_like(sec, dtype=bool)


def round_jet_measures(u: np.ndarray, p: np.ndarray | None, h: float, x0: float, nu: float, k0: float, *,
                       x_fit: tuple[float, float]) -> dict[str, float]:
    """Similarity measures and the momentum budget of a round jet along the axis j = k = (n -
    1) / 2, from the cell-centred mean axial velocity ``u`` and pressure ``p`` (nz, ny, nx).

    Per cross-section the jet core is the forward-flow region around the axis (u > 0; the closed
    box returns the entrained fluid as a counter-flow around it): K = int u^2 dA over it, Q2 =
    int u dA inside r = 2 r_1/2 and r_1/2 = sqrt(A_1/2 / pi) from the area where u > u_c / 2.
    Over ``x_fit`` the slopes of 1/u_c, Q2 and r_1/2 are fitted and compared with
    :func:`schlichting_jet` for the mean measured K (``*_err``) -- the virtual origin drops out
    of every slope. Q2 rather than the whole forward flux: Schlichting's profile has a slow
    tail (a tenth of Q lies beyond 5 r_1/2) that the box's counter-flow cancels, while inside
    2 r_1/2 the profile is the similarity one (``shape_rms``: rms of u / u_c - (1 + xi^2 / 4)^-2
    there). K is compared with the injected flux ``k0``.

    **Momentum budget of the core** (with ``p``). Integrating the steady x-momentum equation
    over the core, whose edge has u = 0 (so neither u^2 nor the convective flux u u_n crosses
    it), gives d/dx [K + Kp - Kv] = S - A dp_e/dx with Kp = int (p - p_e) dA, Kv = int nu
    du/dx dA, S = nu * contour integral of du/dn (the shear of the surrounding counter-flow on
    the core), A the core area and p_e the pressure just outside it. A free jet has S = 0 and
    dp_e/dx = 0, hence a constant K. The integrated terms over the span, in per cent of K at
    its start (``budget_*``), and the residual say whether a drift of K is the confinement (S,
    the box's pressure gradient) or the scheme. Discretely S is the exact lateral flux of the
    cell-centred 5-point Laplacian through the core's edge."""
    nz, ny, nx = u.shape
    jc, kc = (ny - 1) // 2, (nz - 1) // 2
    xc = x0 + (np.arange(nx) + 0.5) * h
    yc = (np.arange(ny) - jc) * h
    zc = (np.arange(nz) - kc) * h
    rr = np.sqrt(yc[None, :] ** 2 + zc[:, None] ** 2)
    sel = np.nonzero((xc >= x_fit[0]) & (xc <= x_fit[1]))[0]
    uc = u[kc, jc, sel]
    K, Q2, rh, dev = [], [], [], []
    bud = {k: [] for k in ("K", "Kp", "Kv", "S", "A", "pe")}
    for i, c in zip(sel, uc):
        sec = u[:, :, i]
        core = _jet_core(sec, jc, kc)
        K.append(float((sec[core] ** 2).sum()) * h * h)
        r_half = math.sqrt(float((sec > 0.5 * c).sum()) * h * h / math.pi)
        rh.append(r_half)
        Q2.append(float(sec[rr < 2.0 * r_half].sum()) * h * h)
        inner = rr < 2.0 * r_half
        prof = (1.0 + (XI_HALF * rr[inner] / r_half) ** 2 / 4.0) ** -2
        dev.append(float(np.sqrt(np.mean((sec[inner] / c - prof) ** 2))))
        if p is None:
            continue
        # edge pairs (core cell, outside neighbour) along y and z
        out = np.zeros_like(core)
        shear = 0.0
        for ax in (0, 1):
            for sh in (1, -1):
                nb_core = np.roll(core, sh, axis=ax)         # nb_core[k, j]: is the neighbour (k, j) - sh in the core
                edge = ~core & nb_core
                out |= edge
                shear += float((sec[edge] - np.roll(sec, sh, axis=ax)[edge]).sum())
        pe = float(p[:, :, i][out].mean())
        bud["K"].append(K[-1])
        bud["Kp"].append(float((p[:, :, i][core] - pe).sum()) * h * h)
        bud["Kv"].append(nu * float((u[:, :, i + 1][core] - u[:, :, i - 1][core]).sum()) * 0.5 * h)
        bud["S"].append(nu * shear)                         # nu sum (u_out - u_in) / h * h
        bud["A"].append(float(core.sum()) * h * h)
        bud["pe"].append(pe)
    K, Q2, rh = np.asarray(K), np.asarray(Q2), np.asarray(rh)
    x = xc[sel]
    inv_slope, inv_icpt = np.polyfit(x, 1.0 / uc, 1)
    q2_slope = np.polyfit(x, Q2, 1)[0]
    r_slope = np.polyfit(x, rh, 1)[0]
    k_mean = float(K.mean())
    th = schlichting_jet(k_mean, nu)
    vals = {"K0": k0, "K_mean": k_mean, "K_start": float(K[0]), "K_end": float(K[-1]),
            "K_drift_pct": 100.0 * float((K[-1] - K[0]) / K[0]), "K_loss_pct": 100.0 * (1.0 - k_mean / k0),
            "inv_uc_slope": float(inv_slope), "inv_uc_slope_theory": th["inv_uc_slope"],
            "inv_uc_err": float(inv_slope / th["inv_uc_slope"] - 1.0),
            "inv_uc_err_k0": float(inv_slope / schlichting_jet(k0, nu)["inv_uc_slope"] - 1.0),
            "x_virtual": float(-inv_icpt / inv_slope),
            "q2_slope": float(q2_slope), "q2_slope_theory": th["q2_slope"],
            "q2_err": float(q2_slope / th["q2_slope"] - 1.0),
            "r_half_slope": float(r_slope), "r_half_slope_theory": th["r_half_slope"],
            "r_half_err": float(r_slope / th["r_half_slope"] - 1.0), "shape_rms": float(np.mean(dev)),
            "uc_start": float(uc[0]), "uc_end": float(uc[-1]), "r_half_end": float(rh[-1]),
            "cells_per_r_half_start": float(rh[0] / h)}
    if p is not None:
        b = {k: np.asarray(v) for k, v in bud.items()}
        # trapezoidal integrals over the span; the derivative of p_e by central differences
        dpe = np.gradient(b["pe"], x)
        s_int = float(np.trapezoid(b["S"], x))
        p_int = float(np.trapezoid(-b["A"] * dpe, x))
        dk, dkp, dkv = (float(b[k][-1] - b[k][0]) for k in ("K", "Kp", "Kv"))
        ref = 100.0 / float(b["K"][0])
        vals.update({"budget_dK_pct": dk * ref, "budget_dKp_pct": dkp * ref, "budget_dKv_pct": -dkv * ref,
                     "budget_shear_pct": s_int * ref, "budget_pressure_pct": p_int * ref,
                     "budget_residual_pct": (dk + dkp - dkv - s_int - p_int) * ref})
    return vals


def run_round_jet(re_j: float = 100.0, cells_per_d: int = 4, *, t_end: float = 880.0, t_avg: float = 800.0,
                  x_fit: tuple[float, float] = (10.0, 22.0), chunk: int = 500, save=None, **kw) -> ThermalResult:
    """Run :func:`round_jet_config` to a steady state, average over [t_avg, t_end] and return
    :func:`round_jet_measures` plus the unsteadiness of the centre-line velocity at the end of
    the span (rms over the averaging window / mean) and the continuity residual. ``save`` (a
    path) keeps the mean u and p as .npz.

    Why so long: the jet itself settles within ~100 time units, but the box-scale return flow
    it drives spins up over several hundred (the box holds ~18 000 d^3 and the jet entrains
    ~5 d^3 per unit time). Until it has, the core's momentum budget does not close (measured
    at 4 cells/d: residual -3.0 % of K averaged over t in [160, 240], -0.36 % over [400, 480],
    -0.12 % over [800, 880])."""
    cfg = round_jet_config(re_j, cells_per_d, **kw)
    s = FlowSolver3D(cfg)
    h = cfg.h
    i_probe = round((x_fit[1] - cfg.x0) / h - 0.5)
    jc = (cfg.ny - 1) // 2
    probe = []
    _sync(s)
    t0 = time.perf_counter()
    averaging = False
    while s.t < t_end:
        if not averaging and s.t >= t_avg:
            s.start_averaging()
            averaging = True
        s.advance(chunk)
        if averaging:
            probe.append(float(s.state.u[jc, jc, i_probe]))
    _sync(s)
    wall = time.perf_counter() - t0
    mf = s.mean_fields(("u", "p"))
    u = mf["u"].double().cpu().numpy()
    p = mf["p"].double().cpu().numpy()
    if save is not None:
        np.savez_compressed(save, u=u, p=p, h=h, x0=cfg.x0)
    k0 = cfg.ports[0].flow * cfg.ports[0].speed
    vals = round_jet_measures(u, p, h, cfg.x0, 1.0 / re_j, k0, x_fit=x_fit)
    vals["notes"] = len(s.notes)
    pr = np.asarray(probe)
    vals["uc_unsteady"] = float(pr.std() / abs(pr.mean())) if len(pr) > 1 else float("nan")
    d = s.latest_diagnostics()
    vals["div_rms"] = d.get("div_rms", float("nan"))
    vals["t_end"] = s.t
    vals["steps_per_s"] = s.steps / wall
    return ThermalResult(f"round jet Re{re_j:g}", (cfg.nx, cfg.ny, cfg.nz), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# emitters: conservation and symmetry
# ---------------------------------------------------------------------------

def emitter_symmetry(device: str = "cpu", dtype: str = "float64", steps: int = 40) -> dict[str, float]:
    """A heated sphere on the symmetry plane with a smoke rake and a disc emitter straddling
    it, half domain vs full domain: the largest difference of smoke, temperature and the
    heat-transfer rate (mirror half of the full run)."""
    import ns3d_cases as cases
    v, f = cases.icosphere(1.0, 2)
    body = ImmersedMesh(v, f, name="sphere", thermal="temperature", thermal_value=1.0)
    em = [Emitter3D(kind="rake", position=(-1.0, -0.6, 0.0), end=(-1.0, 0.6, 0.0), count=5, channel=0),
          Emitter3D(kind="disc", position=(-1.0, 0.0, 0.3), size=(0.3,), channel=1, pulse_period=0.3)]
    th = ThermalConfig3D(prandtl=0.71, richardson=0.2)
    common = {"nx": 32, "nz": 16, "h": 1 / 6, "x0": -1.5, "z0": -4 / 3, "re": 50.0, "bodies": [body],
              "device": device, "dtype": dtype, "perturb": False, "ref_area": math.pi / 4, "thermal": th,
              "emitters": em}
    half = FlowSolver3D(SolverConfig3D(ny=8, y0=0.0, bc_y="symmetry_ymin", **common))
    full = FlowSolver3D(SolverConfig3D(ny=16, y0=-4 / 3, **common))
    half.advance(steps)
    full.advance(steps)
    out = {}
    for name in ("dye", "dye1", "temperature", "u"):
        a = half.field(name).double().cpu()
        b = full.field(name).double().cpu()[:, 8:]
        out[name] = float((a - b).abs().max() / max(float(b.abs().max()), 1e-30))
    qa = half.heat_rates()[0]["heat_rate"]
    qb = full.heat_rates()[0]["heat_rate"]
    out["heat_rate"] = abs(qa - qb) / abs(qb)
    return out


__all__ = [
    "CAVITY_MIDPLANE",
    "CAVITY_NU",
    "DISC_HALF_WIDTH",
    "DVD_2D",
    "NATURAL_BOX",
    "PLATE_FLUX_LOCAL",
    "XI_HALF",
    "ThermalResult",
    "actuator_disc_config",
    "cavity_config",
    "cavity_measures",
    "confined_disc_theory",
    "disc_blockage",
    "emitter_symmetry",
    "flux_plate_config",
    "heated_sphere_config",
    "momentum_theory",
    "natural_sphere_config",
    "nu_churchill_sphere",
    "nu_clift",
    "nu_plate_uniform_flux",
    "nu_ranz_marshall",
    "nu_whitaker",
    "nu_yuge",
    "plate_mesh",
    "richardson",
    "round_jet_config",
    "round_jet_measures",
    "run_actuator_disc",
    "run_archimedes",
    "run_buoyant_sphere_steps",
    "run_cavity",
    "run_cavity_2d",
    "run_flux_plate",
    "run_flux_sphere",
    "run_heated_sphere",
    "run_natural_sphere",
    "run_round_jet",
    "schlichting_jet",
]
