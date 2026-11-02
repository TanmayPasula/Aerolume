"""Verification and validation cases for the 2D placed physics (DESIGN.md section 10).

Shared by ``tests/test_ns2d_physics*.py`` and ``validation/bench_ns2d.py thermal``;
results are quoted in docs/theory_ns2d.md section 21 (the models in 15-20). Every reference number
below was read from the cited source (no value is quoted from memory):

* **de Vahl Davis (1983)**, "Natural convection of air in a square cavity: a bench
  mark numerical solution", Int. J. Numer. Meth. Fluids 3, 249-264, Table V
  (read from the scanned original). Pr = 0.71, velocities in units of kappa/L,
  u_max on the vertical mid-plane (with its height z), w_max (our v_max) on the
  horizontal mid-plane (with its x), Nu_bar the cavity average of the
  horizontal heat flux, Nu_0 the hot-wall average.
* **Heated circular cylinder, forced convection, Pr = 0.71, uniform wall
  temperature**, average Nusselt number as tabulated by Singhal, Dutta & Kalita
  (2023, Int. J. Thermal Sciences 194, 108588; arXiv:2303.17562, Table 2):
  Biswas & Sarkar (2009, IJHMT 52, 1897-1912), Jafroudi & Yang (1986, J. Comput.
  Phys. 65, 46-56), Apelt & Ledwich (1979, JFM 95, 761-777).
* **The same cylinder at Pr = 0.70** (:data:`CYLINDER_NU_REF_PR070`): Bharti,
  Chhabra & Eswaran (2007, Heat Mass Transfer 43, 639-648) and Paramane & Sharma
  (2009, IJHMT 52, 3205-3216, rotation rate 0) as tabulated by Sharma & Dhiman
  (2012, Thermal Science 16(1), 79-91, Tables 1 and 2); Dennis, Hudson & Smith
  (1968, Phys. Fluids 11, 933-940) and Lange, Durst & Breuer (1998, IJHMT 41,
  3409-3430) at Re 20 as tabulated in the Notus CFD validation pages ("heated
  cylinder flow", Re 20, Pr 0.70). The originals were not accessible, so these
  are secondary tabulations. Lange et al.'s correlation (:func:`lange_nusselt`,
  the form ``aerolume.api`` uses) reproduces their tabulated Re 20 value to its
  four decimals, which is the check that the correlation's form is theirs.
* **Rayleigh-Benard onset** between rigid, isothermal plates: Ra_c = 1707.76 at
  k_c = 3.117 (Chandrasekhar 1961; Reid & Harris 1958), independent of Pr. A box
  of width pi / k_c with stress-free, adiabatic side walls holds exactly one roll
  of that mode, so its onset is the infinite-layer one.
* **Bickley (1937) plane laminar jet**: u_c = 0.4543 (K^2 / (nu x))^(1/3) for
  kinematic momentum flux K = int u^2 dy (equivalently u_c^-3 is linear in x
  with slope nu / (0.4543^3 K^2)).
* **Actuator disc** (Froude / Rankine momentum theory; in a confined channel the
  quasi-inviscid model of Garrett & Cummins 2007, JFM 588, 243-251): solved by
  :func:`channel_disc_theory` from continuity, Bernoulli in the bypass and core
  streamtubes and the channel momentum balance; for blockage B -> 0 it gives
  C_T = 4 a (1 - a).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from aerolume.solvers.ns2d import (
    FanSpec,
    FlowSolver,
    ImmersedBody,
    PortSpec,
    SolverConfig,
    ThermalConfig,
    WallBC,
)

# ---------------------------------------------------------------------------
# reference data
# ---------------------------------------------------------------------------

#: de Vahl Davis (1983), Table V (the bench mark solution), Pr = 0.71, velocities in kappa / L
DE_VAHL_DAVIS = {
    1e3: {"psi_mid": 1.174, "u_max": 3.649, "u_max_y": 0.813, "v_max": 3.697, "v_max_x": 0.178,
          "nu_bar": 1.118, "nu_half": 1.118, "nu_0": 1.117},
    1e4: {"psi_mid": 5.071, "u_max": 16.178, "u_max_y": 0.823, "v_max": 19.617, "v_max_x": 0.119,
          "nu_bar": 2.243, "nu_half": 2.243, "nu_0": 2.238},
    1e5: {"psi_mid": 9.111, "u_max": 34.73, "u_max_y": 0.855, "v_max": 68.59, "v_max_x": 0.066,
          "nu_bar": 4.519, "nu_half": 4.519, "nu_0": 4.509},
    1e6: {"psi_mid": 16.32, "u_max": 64.63, "u_max_y": 0.850, "v_max": 219.36, "v_max_x": 0.0379,
          "nu_bar": 8.800, "nu_half": 8.799, "nu_0": 8.817},
}

#: Average Nusselt number of an isothermal cylinder, Pr = 0.71 (Singhal et al. 2023, Table 2)
CYLINDER_NU_REF = {
    20.0: {"Biswas & Sarkar 2009": 2.4483, "Jafroudi & Yang 1986": 2.433},
    40.0: {"Biswas & Sarkar 2009": 3.2351, "Jafroudi & Yang 1986": 3.2, "Apelt & Ledwich 1979": 3.255},
}

#: Average Nusselt number of an isothermal unconfined cylinder, Pr = 0.70 (secondary
#: tabulations, see the module docstring: Sharma & Dhiman 2012, Tables 1-2; Notus CFD, Re 20)
CYLINDER_NU_REF_PR070 = {
    20.0: {"Bharti et al. 2007": 2.4653, "Paramane & Sharma 2009": 2.4189, "Dennis et al. 1968": 2.5216,
           "Lange et al. 1998": 2.4087},
    40.0: {"Bharti et al. 2007": 3.2825, "Paramane & Sharma 2009": 3.2465},
}


def lange_nusselt(re: float) -> float:
    """Mean Nusselt number of a cylinder in laminar cross flow, Pr 0.7, from the
    correlation of Lange, Durst & Breuer (1998), valid for 1e-4 <= Re <= 200:
    ``Nu = 0.082 Re^0.5 + 0.734 Re^x`` with ``x = 0.05 + 0.226 Re^0.085``.

    The same form as ``aerolume.api.lange_nusselt``; it gives 2.4087 at Re 20, the
    value tabulated for Lange et al. by the Notus CFD validation (2.40868 rounds to
    it), 2.8774 at Re 30 and 3.2805 at Re 40."""
    x = 0.05 + 0.226 * re ** 0.085
    return 0.082 * math.sqrt(re) + 0.734 * re ** x

#: Rayleigh-Benard onset between rigid isothermal plates (Chandrasekhar 1961)
RB_RA_C = 1707.76
RB_K_C = 3.117

#: Bickley (1937) plane jet constants: u = 0.4543 (K^2/(nu x))^(1/3) sech^2(xi), xi = 0.2752 (K/nu^2)^(1/3) y / x^(2/3)
BICKLEY_UC = 0.4543
BICKLEY_XI = 0.2752


@dataclass
class CaseResult:
    name: str
    resolution: float
    grid: tuple[int, int]
    steps: int
    wall_s: float
    values: dict = field(default_factory=dict)

    def row(self) -> str:
        vals = "  ".join(f"{k} {v:.5g}" if isinstance(v, float) else f"{k} {v}" for k, v in self.values.items()
                         if not isinstance(v, np.ndarray))        # series stay in .values, not in the table
        return (f"{self.name:<22} {self.resolution:>6g} grid {self.grid[0]}x{self.grid[1]} "
                f"steps {self.steps:>7d} wall {self.wall_s:7.1f}s  {vals}")


def _sync(s: FlowSolver) -> None:
    if s.device.type == "cuda":
        torch.cuda.synchronize()


def richardson(values: list[float], ratio: float = 2.0) -> dict[str, float]:
    """Observed order and Richardson-extrapolated value from three systematically refined results."""
    f1, f2, f3 = values[-3:]                     # coarse, medium, fine
    d12, d23 = f1 - f2, f2 - f3
    if d23 == 0 or d12 / d23 <= 0:
        return {"order": float("nan"), "extrapolated": float("nan"), "gci_fine": float("nan")}
    p = math.log(d12 / d23) / math.log(ratio)
    ext = f3 + (f3 - f2) / (ratio ** p - 1.0)
    gci = 1.25 * abs((f3 - f2) / f3) / (ratio ** p - 1.0)       # Roache's grid convergence index
    return {"order": p, "extrapolated": ext, "gci_fine": gci}


# ---------------------------------------------------------------------------
# differentially heated square cavity (de Vahl Davis 1983)
# ---------------------------------------------------------------------------

def cavity_config(ra: float, n: int, *, pr: float = 0.71, device: str = "auto", dtype: str = "float32",
                  advection: str = "upwind3", **kw) -> SolverConfig:
    """Unit square, no-slip walls, hot left wall theta = +1/2, cold right -1/2, adiabatic top / bottom.

    Free-fall scaling U = sqrt(g beta dT L): Re = sqrt(Ra / Pr), Pe = sqrt(Ra Pr),
    Ri = 1 (DESIGN.md 10.2). Velocities convert to the benchmark's kappa / L by
    multiplying by Pe, heat rates per unit wall length to Nusselt numbers by Pe.
    """
    return SolverConfig(nx=n, ny=n, h=1.0 / n, x0=0.0, y0=0.0, re=math.sqrt(ra / pr), domain="closed",
                        bc_left="noslip", bc_right="noslip", bc_top="noslip", bc_bottom="noslip",
                        advection=advection, device=device, dtype=dtype, dye=False, perturb=False,
                        thermal=ThermalConfig(prandtl=pr, richardson=1.0,
                                              walls=[WallBC("left", "temperature", 0.5),
                                                     WallBC("right", "temperature", -0.5)]), **kw)


def _peak(profile: np.ndarray, coords: np.ndarray) -> tuple[float, float]:
    """Maximum of a sampled profile and its position by a parabola through the top three samples."""
    k = int(np.argmax(profile))
    if 0 < k < len(profile) - 1:
        y0, y1, y2 = profile[k - 1], profile[k], profile[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        off = max(-1.0, min(1.0, off))
        return float(y1 - 0.25 * (y0 - y2) * off), float(coords[k] + off * (coords[1] - coords[0]))
    return float(profile[k]), float(coords[k])


def cavity_quantities(s: FlowSolver, pe: float) -> dict[str, float]:
    """Benchmark quantities of a cavity solution in de Vahl Davis' units (kappa / L).

    * ``nu_hot`` / ``nu_cold``: the scheme's own wall heat flux (conservative
      boundary fluxes), the engineering result;
    * ``nu_half``: horizontal heat flux u T - dT/dx averaged over the vertical
      mid-plane, with the solver's face velocity and the face value of T
      (central), the benchmark's Nu_1/2 definition;
    * ``u_max`` / ``v_max`` with positions: maxima along the mid-planes, the
      position refined by a parabola through the three largest samples; u is
      taken on the x-faces at x = 1/2 (no interpolation for even n), v on the
      y-faces at y = 1/2;
    * ``psi_mid``: |stream function| at the centre, int_0^{1/2} u dy at x = 1/2.
    """
    ops = s.ops
    n, h = ops.nx, ops.h
    st = s.state
    w = s.heat_transfer()["walls"]
    out = {"nu_hot": w["left"] * pe, "nu_cold": -w["right"] * pe}
    u = st.u.detach().double().cpu().numpy()          # (ny, nx+1)
    v = st.v.detach().double().cpu().numpy()          # (ny+1, nx)
    th = st.theta.detach().double().cpu().numpy()
    yc = (np.arange(n) + 0.5) * h
    xc = yc.copy()
    if n % 2 == 0:
        i = n // 2                                     # the x-face at x = 1/2
        uline = u[:, i]
        flux = uline * 0.5 * (th[:, i - 1] + th[:, i]) * pe - (th[:, i] - th[:, i - 1]) / h
        out["nu_half"] = float(flux.mean())
        vline = v[n // 2, :]
        out["psi_mid"] = float(abs(np.sum(uline[: n // 2]) * h)) * pe
    else:
        i = n // 2
        uline = 0.5 * (u[:, i] + u[:, i + 1])
        vline = 0.5 * (v[i] + v[i + 1])
        out["nu_half"] = float("nan")
        out["psi_mid"] = float("nan")
    out["u_max"], out["u_max_y"] = _peak(uline * pe, yc)
    out["v_max"], out["v_max_x"] = _peak(vline * pe, xc)
    return out


def run_cavity(ra: float, n: int, *, t_end: float | None = None, tol: float = 1e-6, chunk: int = 1000,
               **kw) -> CaseResult:
    """Integrate to a steady state: stop when the hot-wall Nu changes by less than
    ``tol`` (relative) per free-fall time over two consecutive checks, or at ``t_end``."""
    cfg = cavity_config(ra, n, **kw)
    pe = math.sqrt(ra * cfg.thermal.prandtl)
    t_end = t_end if t_end is not None else max(60.0, 0.4 * pe)
    s = FlowSolver(cfg)
    _sync(s)
    t0 = time.perf_counter()
    last, calm = None, 0
    while s.t < t_end:
        s.advance(chunk)
        nu = s.heat_transfer()["walls"]["left"] * pe
        tnow = s.t
        if last is not None:
            rate = abs(nu - last[0]) / abs(nu) / max(tnow - last[1], 1e-12)
            calm = calm + 1 if rate < tol else 0
            if calm >= 2:
                break
        last = (nu, tnow)
    _sync(s)
    wall = time.perf_counter() - t0
    vals = cavity_quantities(s, pe)
    vals["t_end"] = float(s.t)
    vals["energy_balance"] = (vals["nu_hot"] - vals["nu_cold"]) / vals["nu_hot"]
    return CaseResult(f"cavity Ra{ra:g}", n, (cfg.nx, cfg.ny), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# thermal wall offset: steady conduction between penalised slabs (calibration check)
# ---------------------------------------------------------------------------

def slab(y_lo: float, y_hi: float, x0: float, x1: float) -> np.ndarray:
    """A CCW rectangle spanning [x0, x1] x [y_lo, y_hi] (a wall slab in a periodic box)."""
    return np.array([[x0, y_lo], [x1, y_lo], [x1, y_hi], [x0, y_hi]], dtype=np.float64)


def run_conduction_slabs(n_gap: int = 16, *, frac: float = 0.0, stiffness: float | None = None,
                         flux_g: float | None = None, device: str = "cpu", dtype: str = "float64",
                         t_end: float | None = None) -> dict[str, float]:
    """Steady conduction between two penalised slabs in a closed box (no flow).

    The gap between the slab surfaces is 1 (n_gap cells), shifted by ``frac`` of
    a cell relative to the grid. Hot slab theta = +1/2 below, cold slab -1/2
    above (or, with ``flux_g``, the lower slab releases the flux G and the upper
    one is held at 0). The exact solution is linear across the gap, so the
    measured heat flux gives the *effective* gap: offset = (gap_eff - 1) / 2 in
    cells per wall (positive = the effective wall lies inside the solid). With
    ``flux_g``: the error of the reported wall temperature against the exact
    G * gap (in units of G h).
    ``stiffness`` S = h^2 / (kappa eta) sets eta explicitly (default: the solver rule).
    """
    h = 1.0 / n_gap
    pe = 1.0
    ny = n_gap + 2 * 8
    y0 = -8 * h - frac * h
    nx = 4
    eta = None if stiffness is None else h * h * pe / stiffness
    big = 10.0
    lower = ImmersedBody(slab(-big, 0.0, -big, big), name="lower",
                         thermal="temperature" if flux_g is None else "heat_flux",
                         thermal_value=0.5 if flux_g is None else flux_g)
    upper = ImmersedBody(slab(1.0, 1.0 + big, -big, big), name="upper", thermal="temperature",
                         thermal_value=-0.5 if flux_g is None else 0.0)
    # closed (adiabatic) domain: the slabs extend past the side walls, so the solution is 1D; the
    # domain edges inside the slabs are adiabatic, so no heat leaks between the two slabs
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=0.0, y0=y0, re=1.0 / 0.71, domain="closed", bodies=[lower, upper],
                       device=device, dtype=dtype, dye=False, perturb=False,
                       thermal=ThermalConfig(prandtl=0.71, richardson=0.0, eta=eta))
    s = FlowSolver(cfg)
    t_end = t_end if t_end is not None else 3.0
    while s.t < t_end:
        s.advance(2000)
    width = nx * h
    ht = s.heat_transfer()
    out = {"frac": frac, "stiffness": h * h * pe / float(s.physics.eta_t)}
    if flux_g is None:
        q = ht["bodies"][0]["heat_rate"] / width                   # heat flux through the gap
        gap_eff = (1.0 / pe) * 1.0 / q
        out["offset_cells"] = (gap_eff - 1.0) / 2.0 / h
        out["q_upper"] = -ht["bodies"][1]["heat_rate"] / width
    else:
        tw = ht["bodies"][0]["theta_wall"]
        out["theta_wall"] = tw
        out["theta_wall_err_Gh"] = (tw - flux_g * 1.0) / (flux_g * h)
        out["q_upper"] = -ht["bodies"][1]["heat_rate"] / width * pe / flux_g
    return out


# ---------------------------------------------------------------------------
# forced convection from a heated cylinder
# ---------------------------------------------------------------------------

def circle(n: int = 256, d: float = 1.0, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[cx + 0.5 * d * np.cos(th), cy + 0.5 * d * np.sin(th)]


def heated_cylinder_config(re: float, cells_per_d: int, *, pr: float = 0.71, upstream: float = 10.0,
                           downstream: float = 20.0, half_height: float = 10.0, device: str = "auto",
                           dtype: str = "float32", **kw) -> SolverConfig:
    """The momentum validation's cylinder domain (5 % blockage, slip walls) with an
    isothermal cylinder (theta = 1) in a cold stream (theta = 0), no buoyancy."""
    h = 1.0 / cells_per_d
    nx, ny = round((upstream + downstream) / h), round(2 * half_height / h)
    n_poly = max(64, int(2 * math.pi / h))
    return SolverConfig(nx=nx, ny=ny, h=h, x0=-upstream, y0=-half_height, re=re,
                        bodies=[ImmersedBody(circle(n_poly), name="cylinder", thermal="temperature",
                                             thermal_value=1.0)],
                        bc_top="slip", bc_bottom="slip", advection="upwind3", device=device, dtype=dtype,
                        force_ref=1.0, moment_ref=(0.0, 0.0), dye=False, perturb=False,
                        thermal=ThermalConfig(prandtl=pr, richardson=0.0), **kw)


def run_heated_cylinder(re: float, cells_per_d: int, *, t_end: float = 60.0, t_avg: float = 50.0,
                        chunk: int = 500, **kw) -> CaseResult:
    cfg = heated_cylinder_config(re, cells_per_d, **kw)
    s = FlowSolver(cfg)
    _sync(s)
    t0 = time.perf_counter()
    averaging = False
    while s.t < t_end:
        if not averaging and s.t >= t_avg:
            s.start_averaging()
            averaging = True
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    ht = s.heat_transfer(mean=averaging)
    body = ht["bodies"][0]
    surf = s.surface(720, mean=averaging)[0]
    ds = surf["s"][1] - surf["s"][0]
    nu_surface = float(np.nansum(surf["nu_local"]) * ds / (np.pi * 1.0))
    hst = s.history()
    m = hst["t"] >= t_avg
    # steadiness: relative change of the body heat rate between the halves of the
    # averaging window (a steady Re <= 40 wake should leave it at the 1e-4 level)
    ph = s.physics_history()
    q, tq = ph.get("heat:cylinder"), ph.get("t")
    nu_drift = float("nan")
    if q is not None and tq is not None:
        w = tq >= t_avg
        if w.sum() >= 4:
            qw, half = q[w], int(w.sum()) // 2
            nu_drift = float((qw[half:].mean() - qw[:half].mean()) / abs(qw.mean()))
    vals = {"nu": body["nusselt"], "nu_surface": nu_surface, "cd": float(hst["cd"][m].mean()),
            "heat_rate": body["heat_rate"], "nu_front": float(surf["nu_local"][np.argmin(surf["x"])]),
            "nu_rear": float(surf["nu_local"][np.argmax(surf["x"])]), "nu_drift": nu_drift,
            "pr": cfg.thermal.prandtl, "blockage": 1.0 / (cfg.ny * cfg.h), "t_end": float(s.t)}
    return CaseResult(f"heated cylinder Re{re:g}", cells_per_d, (cfg.nx, cfg.ny), s.steps, wall, vals)


#: domains of the blockage study (blockage D / H -> extents in D). The 5 % domain is the
#: momentum validation's; the larger ones push the inflow and outflow out with the walls.
CYLINDER_DOMAINS = {0.05: {"upstream": 10.0, "downstream": 20.0, "half_height": 10.0},
                    0.025: {"upstream": 15.0, "downstream": 25.0, "half_height": 20.0},
                    0.0125: {"upstream": 20.0, "downstream": 30.0, "half_height": 40.0}}


def cylinder_reference_comparison(re: float, nu: float) -> dict[str, float]:
    """Relative difference (per cent) of a Pr 0.70 Nusselt number from the Lange et al.
    correlation and from every Pr 0.70 reference tabulated at that Re."""
    out = {"lange": lange_nusselt(re), "vs_lange_pct": 100.0 * (nu / lange_nusselt(re) - 1.0)}
    for src, val in CYLINDER_NU_REF_PR070.get(float(re), {}).items():
        out[f"vs {src} %"] = 100.0 * (nu / val - 1.0)
    return out


# ---------------------------------------------------------------------------
# Rayleigh-Benard onset
# ---------------------------------------------------------------------------

def rb_config(ra: float, n: int, *, pr: float = 0.71, device: str = "auto", dtype: str = "float64",
              **kw) -> SolverConfig:
    """One convection roll: width pi / k_c, height 1, slip adiabatic side walls,
    no-slip isothermal plates (bottom +1/2, top -1/2), free-fall scaling."""
    width = math.pi / RB_K_C
    nx = max(4, round(n * width))
    return SolverConfig(nx=nx, ny=n, h=1.0 / n, x0=0.0, y0=0.0, re=math.sqrt(ra / pr), domain="closed",
                        bc_left="slip", bc_right="slip", bc_top="noslip", bc_bottom="noslip", device=device,
                        dtype=dtype, dye=False, perturb=False,
                        thermal=ThermalConfig(prandtl=pr, richardson=1.0,
                                              walls=[WallBC("bottom", "temperature", 0.5),
                                                     WallBC("top", "temperature", -0.5)]), **kw)


def run_rb_growth(ra: float, n: int, *, t_end: float = 120.0, amp: float = 1e-4, chunk: int = 200,
                  **kw) -> CaseResult:
    """Growth rate of a small roll perturbation (in units of the diffusive rate kappa / L^2).

    Starts from the conduction profile plus ``amp`` cos(pi x / W) sin(pi y) in
    temperature; records the kinetic energy E(t) and fits log E over the last
    half of the run (after the initial adjustment): sigma = (d ln E / dt) / 2,
    converted from free-fall to diffusive units by multiplying by Pe.
    """
    cfg = rb_config(ra, n, **kw)
    s = FlowSolver(cfg)
    ops = s.ops
    yc = (torch.arange(ops.ny, dtype=torch.float64) + 0.5) * ops.h
    xc = (torch.arange(ops.nx, dtype=torch.float64) + 0.5) * ops.h
    width = ops.nx * ops.h
    Y, X = torch.meshgrid(yc, xc, indexing="ij")
    theta = 0.5 - Y + amp * torch.cos(math.pi * X / width) * torch.sin(math.pi * Y)
    s.set_temperature(theta)
    pe = math.sqrt(ra * cfg.thermal.prandtl)
    ts, es = [], []
    _sync(s)
    t0 = time.perf_counter()
    while s.t < t_end:
        s.advance(chunk)
        st = s.state
        e = float((st.u[:, 1:-1] ** 2).sum() + (st.v[1:-1] ** 2).sum()) * ops.h * ops.h
        ts.append(s.t)
        es.append(e)
    _sync(s)
    wall = time.perf_counter() - t0
    ts, es = np.asarray(ts), np.asarray(es)
    m = ts >= 0.5 * ts[-1]
    slope = float(np.polyfit(ts[m], np.log(np.maximum(es[m], 1e-300)), 1)[0])
    vals = {"sigma_diffusive": 0.5 * slope * pe, "energy_end": float(es[-1]), "t_end": float(ts[-1])}
    return CaseResult(f"RB Ra{ra:g}", n, (cfg.nx, cfg.ny), s.steps, wall, vals)


def rb_onset(ra_lo: float, ra_hi: float, n: int, **kw) -> dict[str, float]:
    """Bracket the onset: growth rates at two Rayleigh numbers and their linear zero crossing."""
    lo, hi = run_rb_growth(ra_lo, n, **kw), run_rb_growth(ra_hi, n, **kw)
    s1, s2 = lo.values["sigma_diffusive"], hi.values["sigma_diffusive"]
    ra_c = ra_lo - s1 * (ra_hi - ra_lo) / (s2 - s1)
    return {"ra_lo": ra_lo, "sigma_lo": s1, "ra_hi": ra_hi, "sigma_hi": s2, "ra_c": ra_c,
            "err_pct": 100.0 * (ra_c / RB_RA_C - 1.0), "wall_s": lo.wall_s + hi.wall_s}


# ---------------------------------------------------------------------------
# actuator disc in a channel (momentum theory)
# ---------------------------------------------------------------------------

def channel_disc_theory(ct: float, blockage: float, *, propeller: bool = False) -> dict[str, float]:
    """Quasi-inviscid actuator disc in a rigid-lid channel (Garrett & Cummins 2007 model).

    Unknowns (in units of the upstream speed U = 1): disc velocity u_t, core
    wake velocity u_w, bypass velocity u_b, core wake area fraction s_w.
    Equations: core continuity B u_t = s_w u_w; total continuity 1 = s_w u_w +
    (1 - s_w) u_b; Bernoulli in core and bypass with uniform pressure at the
    wake station gives dP = (u_b^2 - u_w^2) / 2 and p_0 - p_4 = (u_b^2 - 1) / 2;
    channel momentum (u_b^2 - 1) / 2 - dP B = s_w u_w^2 + (1 - s_w) u_b^2 - 1, where
    dP is the static-pressure drop across the disc and the disc's force on the fluid
    is -dP D. A turbine (momentum sink, the Garrett-Cummins case) has dP = C_T / 2
    with T = C_T (1/2) rho U^2 D, D = B H; a propeller / fan blowing downstream has
    dP = -C_T / 2 (the same balances, the wake faster than the bypass). Returns u_t
    and the induction a = |u_t - 1|; B -> 0 recovers C_T = 4 a (1 - a) (turbine) and
    C_T = 4 a (1 + a) (propeller).
    """
    from scipy.optimize import brentq

    B, dp = float(blockage), (-0.5 if propeller else 0.5) * float(ct)

    def resid_for(ub: float) -> float:
        # given u_b, dP fixes u_w; continuity fixes s_w; momentum is the residual
        uw2 = ub * ub - 2.0 * dp
        if uw2 <= 0:
            return 1e6
        uw = math.sqrt(uw2)
        sw = (ub - 1.0) / (ub - uw) if ub != uw else 1.0
        return (0.5 * (ub * ub - 1.0) - dp * B) - (sw * uw2 + (1.0 - sw) * ub * ub - 1.0)

    if propeller:
        ub = brentq(resid_for, 1e-6, 1.0 - 1e-12)          # the bypass slows down, the wake speeds up
    else:
        lo = math.sqrt(2.0 * dp) + 1e-9
        ub = brentq(resid_for, max(1.0, lo) + 1e-12, 10.0)
    uw = math.sqrt(ub * ub - 2.0 * dp)
    sw = (ub - 1.0) / (ub - uw)
    ut = sw * uw / B if B > 0 else float("nan")
    return {"u_t": ut, "a": abs(ut - 1.0), "u_w": uw, "u_b": ub, "s_w": sw}


def froude_induction(ct: float, *, propeller: bool = False) -> float:
    """Unconfined actuator disc: a from C_T = 4 a (1 - a) (turbine, the light-loading
    root) or C_T = 4 a (1 + a) (propeller), Froude / Rankine momentum theory."""
    if propeller:
        return 0.5 * (math.sqrt(1.0 + ct) - 1.0)
    return 0.5 * (1.0 - math.sqrt(1.0 - ct))


def run_actuator_disc(ct: float, cells_per_d: int = 32, *, propeller: bool = True, half_height: float = 5.0,
                      re: float = 500.0, upstream: float = 10.0, downstream: float = 30.0, t_end: float = 60.0,
                      t_avg: float = 30.0, thickness: float | None = None, chunk: int = 500, device: str = "auto",
                      dtype: str = "float32") -> CaseResult:
    """An actuator strip of width D = 1 normal to the flow in a slip-walled tunnel of
    height 2 half_height (blockage B = D / H) with thrust per span T = C_T / 2.

    ``propeller=True`` (the default, a fan pushing the flow downstream: direction
    +x, positive thrust; ``validation/run_all.py`` relies on this default) speeds
    the flow through the disc (a = u_disc - 1); ``propeller=False`` is a turbine:
    the fan blows against the stream (direction -x), a momentum sink that slows the
    flow through the disc (a = 1 - u_disc). The disc velocity is the
    footprint-averaged axial speed of the fan report.
    """
    h = 1.0 / cells_per_d
    nx, ny = round((upstream + downstream) / h), round(2 * half_height / h)
    direction = (1.0, 0.0) if propeller else (-1.0, 0.0)
    fan = FanSpec((0.0, 0.0), direction=direction, width=1.0, thickness=thickness, mode="thrust",
                  value=0.5 * ct, name="disc")
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=-upstream, y0=-half_height, re=re, fans=[fan],
                       bc_top="slip", bc_bottom="slip", device=device, dtype=dtype, dye=False, perturb=False)
    s = FlowSolver(cfg)
    _sync(s)
    t0 = time.perf_counter()
    averaging = False
    while s.t < t_end:
        if not averaging and s.t >= t_avg:
            s.start_averaging()
            averaging = True
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    rep = s.fans_report(mean=True)[0]
    u_disc = rep["disc_velocity"] * direction[0]                  # streamwise velocity through the disc
    B = 1.0 / (2 * half_height)
    th = channel_disc_theory(ct, B, propeller=propeller)
    vals = {"ct": ct, "blockage": B, "u_disc": u_disc, "a": abs(u_disc - 1.0), "a_theory": th["a"],
            "a_froude": froude_induction(ct, propeller=propeller), "thrust": rep["thrust"],
            "err_pct": 100.0 * (u_disc / th["u_t"] - 1.0),
            "a_err_pct": 100.0 * (abs(u_disc - 1.0) / th["a"] - 1.0), "t_end": float(s.t)}
    kind = "propeller" if propeller else "turbine"
    return CaseResult(f"disc {kind} CT{ct:g}", cells_per_d, (nx, ny), s.steps, wall, vals)


# ---------------------------------------------------------------------------
# plane jet (port) into a quiescent closed box
# ---------------------------------------------------------------------------

def run_plane_jet(re_j: float = 25.0, cells_per_b: int = 8, *, length: float = 80.0, half_height: float = 40.0,
                  t_end: float = 400.0, t_avg: float = 300.0, chunk: int = 1000, device: str = "auto",
                  dtype: str = "float32", x_fit: tuple[float, float] = (8.0, 24.0), save=None) -> CaseResult:
    """A jet port of width b = 1 at speed 1 (Re_j = 1 b / nu) blowing along +x into a closed
    box, balanced by a vent across the whole far wall (a uniform drift, not a second jet),
    averaged over t in [t_avg, t_end]; :func:`jet_measures` gives the momentum flux
    along the jet and the similarity checks. A closed box must return the entrained
    fluid upstream, so a weak counter-flow (a few per cent of the jet speed for a box
    80 b tall) flanks the jet and slowly drains its momentum flux; taller boxes reduce
    it (theory doc 21.5). ``save`` (a path) keeps the mean fields as .npz.
    """
    h = 1.0 / cells_per_b
    nx, ny = round(length / h), round(2 * half_height / h)
    jet = PortSpec((2.0, 0.0), direction=(1.0, 0.0), size=1.0, speed=1.0, kind="jet", name="jet")
    vent_size = 2.0 * half_height - 2.0
    vent = PortSpec((length - 2.0, 0.0), direction=(1.0, 0.0), size=vent_size, speed=1.0 / vent_size, kind="vent",
                    name="vent")
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=0.0, y0=-half_height, re=re_j, domain="closed", ports=[jet, vent],
                       bc_left="noslip", bc_right="noslip", bc_top="noslip", bc_bottom="noslip", device=device,
                       dtype=dtype, dye=False, perturb=False)
    s = FlowSolver(cfg)
    _sync(s)
    t0 = time.perf_counter()
    averaging = False
    while s.t < t_end:
        if not averaging and s.t >= t_avg:
            s.start_averaging()
            averaging = True
        s.advance(chunk)
    _sync(s)
    wall = time.perf_counter() - t0
    mf = s.mean_fields()
    u = mf["u"].double().cpu().numpy()
    p = mf["p"].double().cpu().numpy()
    if save is not None:
        np.savez_compressed(save, u=u, v=mf["v"].double().cpu().numpy(), p=p, h=h, y0=-half_height)
    return CaseResult(f"plane jet Re{re_j:g}", cells_per_b, (nx, ny), s.steps, wall,
                      {**jet_measures(u, p, h, -half_height, 1.0 / re_j, x_fit=x_fit), "t_end": float(s.t)})


def jet_measures(u: np.ndarray, p: np.ndarray, h: float, y0: float, nu: float, *,
                 x_fit: tuple[float, float] = (12.0, 40.0)) -> dict[str, float]:
    """Momentum flux and similarity measures of a plane jet along y = 0 (cell-centred mean fields).

    The closed box returns the entrained fluid as a slow counter-flow, so the flux is
    integrated over the jet itself: the forward-flow core between the nearest zero
    crossings of u on either side of the axis. ``K`` = int u^2 dy over the core,
    ``Kp`` = int (p - p_edge) dy (the pressure part of the momentum flux, p_edge the
    pressure at the core edges). The centreline decay is fitted as
    u_c^-3 = slope (x - x_v) over ``x_fit`` and compared with the Bickley (1937)
    similarity solution u_c = (3 K^2 / (32 nu x))^(1/3) (= 0.4543 (K^2 / (nu x))^(1/3)),
    i.e. slope = 32 nu / (3 K^2); the half-width b_1/2 (u = u_c / 2) with Bickley's
    b_1/2 = 0.8814 / 0.2752 (nu^2 / K)^(1/3) (x - x_v)^(2/3), and the profile shape
    by the rms deviation of u / u_c from sech^2(0.8814 y / b_1/2).

    **Momentum budget of the core** (the conservation statement itself). Integrating
    the steady x-momentum equation across the core, whose edges are where u = 0
    (so the convective flux u v through them vanishes), gives

        d/dx [K + Kp - Kv] = S - W dp_edge/dx,

    with Kv = int nu du/dx dy (streamwise diffusion), S = nu [du/dy] at the two edges
    (the shear of the return flow on the jet) and W dp_edge/dx the force of the box's
    streamwise pressure gradient on the core width W. A free jet in a quiescent
    ambient has S = 0 and dp_edge/dx = 0, hence a constant K; in a closed box both are
    small but finite. Over ``x_fit`` the function reports their shares of the change
    of K and the budget residual (all in per cent of K at the start of the span): a
    residual near zero means the discrete solution conserves momentum and the drift
    of K is the confinement, not the scheme.
    """
    ny, nx = u.shape
    xc = (np.arange(nx) + 0.5) * h
    yc = y0 + (np.arange(ny) + 0.5) * h
    j0 = ny // 2                                            # rows j0 - 1, j0 straddle y = 0
    K, Kp, Kv, shear, p_e, width = (np.full(nx, np.nan) for _ in range(6))
    uc = 0.5 * (u[j0 - 1] + u[j0])
    for i in range(1, nx - 1):
        col = u[:, i]
        lo, hi = j0 - 1, j0
        while lo > 0 and col[lo - 1] > 0:
            lo -= 1
        while hi < ny - 1 and col[hi + 1] > 0:
            hi += 1
        if lo == 0 or hi == ny - 1:                         # forward flow across the whole box (port / vent)
            continue
        core = slice(lo, hi + 1)
        K[i] = float((col[core] ** 2).sum()) * h
        p_e[i] = 0.5 * (p[lo - 1, i] + p[hi + 1, i])
        Kp[i] = float((p[core, i] - p_e[i]).sum()) * h
        Kv[i] = nu * float((u[core, i + 1] - u[core, i - 1]).sum()) * 0.5
        shear[i] = nu * ((col[hi + 1] - col[hi]) - (col[lo] - col[lo - 1])) / h
        width[i] = (hi - lo + 1) * h
    sel = (xc >= x_fit[0]) & (xc <= x_fit[1])
    fit = np.polyfit(xc[sel], uc[sel] ** -3.0, 1)
    k_mean = float(np.nanmean(K[sel]))
    slope_theory = nu / (BICKLEY_UC ** 3 * k_mean ** 2)
    x_v = float(-fit[1] / fit[0])
    ia, ib = int(np.searchsorted(xc, x_fit[0])), int(np.searchsorted(xc, x_fit[1]))
    total = K + Kp - Kv
    d_total = float(total[ib] - total[ia])
    s_int = float(np.nansum(shear[ia:ib])) * h
    g_int = -float(np.nansum(np.gradient(p_e, h)[ia:ib] * width[ia:ib])) * h
    k0 = float(K[ia])

    def at(xs: float) -> float:
        return float(K[min(int(np.searchsorted(xc, xs)), nx - 1)])

    out = {"K_x4": at(4.0), "K_x12": at(12.0), "K_x24": at(24.0), "K_x40": at(40.0),
           "K_mean": k_mean, "K_drift_pct": 100.0 * float(np.nanmax(K[sel]) - np.nanmin(K[sel])) / k_mean,
           "Kp_mean": float(np.nanmean(Kp[sel])),
           "K_change_pct": 100.0 * float(K[ib] - K[ia]) / k0,
           "K_budget_shear_pct": 100.0 * s_int / k0, "K_budget_pressure_pct": 100.0 * g_int / k0,
           "K_budget_residual_pct": 100.0 * (d_total - s_int - g_int) / k0,
           "slope": float(fit[0]), "slope_bickley": slope_theory,
           "slope_err_pct": 100.0 * (float(fit[0]) / slope_theory - 1.0), "virtual_origin": x_v}
    for xs in (20.0, 30.0):
        i = int(np.searchsorted(xc, xs))
        prof = 0.5 * (u[:, i] + u[:, i - 1]) if xc[i] - xs > 0.25 * h else u[:, i]
        ucen = float(0.5 * (prof[j0 - 1] + prof[j0]))
        half = np.flatnonzero(prof[j0:] < 0.5 * ucen)
        if not len(half):
            continue
        k = j0 + int(half[0])                                 # first row above the half value
        f0, f1 = prof[k - 1], prof[k]
        b_half = float(yc[k - 1] + (0.5 * ucen - f0) / (f1 - f0) * h)
        b_theory = 0.8814 / BICKLEY_XI * (nu * nu / k_mean) ** (1.0 / 3.0) * (xs - x_v) ** (2.0 / 3.0)
        m = np.abs(yc) <= 2.0 * b_half
        shape = prof[m] / ucen - 1.0 / np.cosh(0.8814 * yc[m] / b_half) ** 2
        out[f"b_half_x{xs:g}"] = b_half
        out[f"b_half_err_pct_x{xs:g}"] = 100.0 * (b_half / b_theory - 1.0)
        out[f"shape_rms_x{xs:g}"] = float(np.sqrt(np.mean(shape ** 2)))
    return out


# ---------------------------------------------------------------------------
# fan in a closed room: a grid study, not a validation (theory doc 18.4)
# ---------------------------------------------------------------------------

#: the app's fan_room_case (1 N/m, 0.5 m fan, 4 x 3 m air room) in solver units: L = 0.5 m,
#: U = 0.9035 m/s (the fan's static momentum-theory velocity), so the box is 8 x 6 L,
#: Re = U L / nu = 30 928 and the thrust T / (rho U^2 L) = 2.0 per unit span
FAN_ROOM = {"width": 8.0, "height": 6.0, "re": 30928.26, "thrust": 2.0, "u_ref": 0.9035, "l_ref": 0.5}


def fan_room_config(cells_per_l: float, *, wall_model: str = "none", device: str = "auto",
                    dtype: str = "float32") -> SolverConfig:
    """The planner's 2D fan room (Draft = 12 cells per L, 96 x 72) at any resolution.

    No-slip walls on every side, WALE LES and upwind3 advection exactly as the planner
    chooses them for this Reynolds number (checked against ``api.plan`` in the
    grid-study script). ``wall_model`` picks the domain-wall shear (theory doc 18.4).
    """
    h = 1.0 / cells_per_l
    nx, ny = round(FAN_ROOM["width"] / h), round(FAN_ROOM["height"] / h)
    fan = FanSpec((FAN_ROOM["width"] / 3.0, FAN_ROOM["height"] / 2.0), direction=(1.0, 0.0), width=1.0,
                  mode="thrust", value=FAN_ROOM["thrust"], name="Fan")
    return SolverConfig(nx=nx, ny=ny, h=h, x0=0.0, y0=0.0, re=FAN_ROOM["re"], domain="closed", bc_top="noslip",
                        bc_bottom="noslip", bc_left="noslip", bc_right="noslip", turbulence="wale",
                        advection="upwind3", fans=[fan], wall_model=wall_model, device=device, dtype=dtype,
                        dye=False, perturb=False)


def run_fan_room(cells_per_l: float, *, wall_model: str = "none", t_end: float = 1200.0, t_avg: float = 800.0,
                 sample: float = 2.0, device: str = "auto", dtype: str = "float32") -> CaseResult:
    """Spin a fan room up from rest and report its room-mean speed (cell centres) in m/s.

    ``mean_speed`` / ``max_speed`` / ``disc_velocity`` are averages of samples taken every
    ``sample`` L/U over [t_avg, t_end]; ``drift_pct`` compares the two halves of that window
    (a settled room drifts by a few per cent at most); ``series`` holds every sample
    (t, mean, max, disc velocity) in solver units.
    """
    s = FlowSolver(fan_room_config(cells_per_l, wall_model=wall_model, device=device, dtype=dtype))
    _sync(s)
    t0 = time.perf_counter()
    rows = []
    while s.t < t_end:
        target = s.t + sample
        while s.t < target:
            s.advance(max(1, int(0.9 * (target - s.t) / max(s.dt, 1e-6))))
        sp = s.field("speed")
        rows.append((float(s.t), float(sp.mean()), float(sp.max()), float(s.fans_report()[0]["disc_velocity"])))
    _sync(s)
    wall = time.perf_counter() - t0
    a = np.asarray(rows)
    win = a[a[:, 0] >= t_avg]
    half = len(win) // 2
    U = FAN_ROOM["u_ref"]
    vals = {"wall_model": s.cfg.wall_model,
            "h_mm": 1000.0 * FAN_ROOM["l_ref"] / cells_per_l,
            "mean_speed": U * float(win[:, 1].mean()), "max_speed": U * float(win[:, 2].mean()),
            "disc_velocity": U * float(win[:, 3].mean()),
            "drift_pct": 100.0 * float(win[half:, 1].mean() / win[:half, 1].mean() - 1.0),
            "series": a}
    return CaseResult(f"fan room {vals['wall_model']}", cells_per_l, (s.cfg.nx, s.cfg.ny), s.steps, wall, vals)


__all__ = [
    "BICKLEY_UC",
    "CYLINDER_NU_REF",
    "DE_VAHL_DAVIS",
    "FAN_ROOM",
    "RB_K_C",
    "RB_RA_C",
    "CaseResult",
    "cavity_config",
    "cavity_quantities",
    "channel_disc_theory",
    "fan_room_config",
    "froude_induction",
    "heated_cylinder_config",
    "jet_measures",
    "rb_config",
    "rb_onset",
    "richardson",
    "run_actuator_disc",
    "run_cavity",
    "run_conduction_slabs",
    "run_fan_room",
    "run_heated_cylinder",
    "run_plane_jet",
    "run_rb_growth",
]

