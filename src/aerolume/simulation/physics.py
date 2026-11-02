"""Placed physics in the simulation layer: from the SI model to solver units (DESIGN.md section 10).

The model describes emitters, heat transfer, heaters, fans, jets and closed rooms in
metres, seconds, degrees Celsius, watts and newtons (:mod:`aerolume.model`); the 2D
and 3D solvers take non-dimensional specs (``solvers.ns2d.config`` and
``solvers.ns3d.config``). This module is the one place that converts between them, so
the planners, the explore sessions, the runners and the result post-processing use one
set of scales and cannot disagree. It needs numpy only (the solver config classes are
imported where a spec is built), so the Setup page can call :func:`physics_scales` on the
GUI thread.

**Scales** (:class:`PhysicsScales`, DESIGN.md 10.2). The solver's unit length L is the
design's reference length, exactly as for the forces. The other three are chosen:

* **ΔT_ref**, the temperature scale: the spread of every temperature the case sets
  (ambient, fixed-temperature bodies and walls, hot jets) -- T_hot − T_cold, the
  definition of the cavity benchmarks' Rayleigh number (de Vahl Davis 1983) -- or,
  when larger, the rise a heat flux produces, q L / (k Nu) with Nu from a laminar
  correlation (:func:`flux_rise`), or a heater, P / (rho c_p u A) with u A the air its
  plume or the stream carries through it (:func:`heater_rise`). Then
  ``theta = (T − T_amb) / ΔT_ref``.
* **U**, the speed scale: the free stream in a tunnel. A closed box has none: the
  fastest jet, speed-controlled fan, thrust fan (its static momentum-theory disc
  velocity sqrt(T / 2 rho A)) or moving wall; failing those the free-fall velocity
  sqrt(g beta ΔT H) (so Re = sqrt(Gr), Ri = L / H: the cavity convention), or kappa / H
  for a conduction-dominated box (Ra < 10^3). H, the length of Gr and Ra, is the
  box height for a box heated only at its walls or by sources, else L.
* The solver numbers then follow: ``Re = U L / nu``, ``Pe = Re Pr``,
  ``Ri = g beta ΔT_ref L / U^2`` (the Boussinesq coefficient), heat-flux gradients
  ``G = q'' L / (k ΔT_ref)``, heater power ``Q / (rho c_p U ΔT_ref L)`` per unit span in
  2D (the 2D solver's heat unit) and ``Q / (k ΔT_ref L)`` in 3D (the 3D solver's
  conduction unit), fan thrust ``T / (rho U^2 L^(d-1))``, pressure jumps
  ``dp / (rho U^2)``, speeds ``/ U``, times ``* U / L``, particle rates ``* L / U``.

**Frames.** ``frame="body"`` items turn with the geometry (2D: by −alpha about the
moment point; 3D: yaw then pitch, :func:`rotation3`), ``"wind"`` items stay where they
are in the tunnel -- the rule the planners apply to the bodies, and the one the Setup
preview draws (``app.widgets.physics_editor.to_wind``).

**Fluid properties** (:func:`fluid_defaults`): air by Sutherland's law for the
conductivity (White, Viscous Fluid Flow, 3rd ed., Table 1-3), cp 1006 J/(kg K) and the
ideal-gas beta = 1/T; water from Incropera & DeWitt Table A.6. Pr is mu cp / k with the
fluid's own viscosity, so it matches the Reynolds number of the run. The tables are the
Setup page's, so the page and the run quote the same numbers.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core import units
from ..model import Design, Fluid, Simulation

__all__ = [
    "BOUSSINESQ_LIMIT",
    "DISC_WAKE_FACTOR",
    "DRAFT_BUOYANT_T_MAX",
    "DRAFT_BUOYANT_T_MIN",
    "ESTIMATED_RISE",
    "FAN_CELLS_MIN",
    "FLUID_AMBIENT_TOL_K",
    "PORT_CELLS_MIN",
    "RA_CONDUCTION",
    "RI_BUOYANT",
    "RI_BUOYANT_LAYER",
    "ROOM_2D_FAN_NOTE",
    "ROOM_2D_FAN_WARNING",
    "TEMPERATURE_MONITORS",
    "PhysicsScales",
    "Placed",
    "ThermalProps",
    "boussinesq_warning",
    "buoyancy_dominated_warning",
    "buoyancy_note",
    "buoyant_coefficients_note",
    "buoyant_layer",
    "buoyant_speed",
    "closed_box",
    "default_ambient",
    "delta_t_is_estimate",
    "draft_buoyant_duration",
    "fan_disc_speed",
    "fluid_ambient_warning",
    "fluid_defaults",
    "fluid_temperature",
    "flux_rise",
    "has_placed_physics",
    "heat_input",
    "heat_sink",
    "heater_rise",
    "item_bounds",
    "measured_delta_t",
    "merge_warnings",
    "peak_source",
    "peak_speed",
    "physics_rows",
    "physics_scales",
    "placed_physics_2d",
    "placed_physics_3d",
    "regime_not_validated_warning",
    "result_scales",
    "room_2d_fan_warning",
    "rotation3",
    "thermal_estimate",
    "thermal_props",
    "unbounded_heating_note",
    "warning_key",
    "wind_dir2",
    "wind_point2",
    "wind_points3",
    "with_delta_t",
]

G0 = 9.80665
#: Below this Rayleigh number a closed box heated at its walls moves heat mainly by
#: conduction (de Vahl Davis' Ra 10^3 cavity has Nu 1.12), so kappa / H is its speed scale.
RA_CONDUCTION = 1.0e3
#: Above this beta * ΔT the Boussinesq approximation (density constant except in the buoyancy
#: force) loses accuracy (the usual engineering bound; the Setup page uses the same).
BOUSSINESQ_LIMIT = 0.1
#: Fewer cells than this across a fan's diameter blur its disc (the actuator force is spread over
#: a smooth one-cell edge on both rims), and the delivered thrust and disc velocity are then only
#: indicative; the validation used 16-32 (theory_ns2d.md 21, theory_ns3d.md 17).
FAN_CELLS_MIN = 8.0
#: Fewer cells than this across a jet or vent opening and the jet's profile is two diffuse edges.
PORT_CELLS_MIN = 4.0
#: Relative mismatch between jets and vents of a closed box the planner reports (the solver
#: removes the imbalance uniformly to keep the pressure problem solvable).
BALANCE_RTOL = 0.02
#: Mean Nusselt number of a differentially heated cavity (aspect ratio ~1), the correlation of
#: Berkovsky & Polevikov as given by Incropera & DeWitt (eq. 9.50): Nu = 0.18 (Pr Ra / (0.2 + Pr))^0.29,
#: 10^3 < Pr Ra / (0.2 + Pr). At Pr 0.71 it gives 2.4 / 4.8 / 9.3 at Ra 10^4 / 10^5 / 10^6 against de
#: Vahl Davis' 2.24 / 4.52 / 8.80: good enough to size the thermal boundary layer delta_T = H / (2 Nu).
CAVITY_NU = (0.18, 0.2, 0.29)

_SIDES2 = ("left", "right", "bottom", "top")
_SIDES3 = ("left", "right", "front", "back", "bottom", "top")


# ---------------------------------------------------------------------------
# fluid properties
# ---------------------------------------------------------------------------

#: Fresh water, 0-100 °C: conductivity k [W/(m K)], specific heat cp [J/(kg K)], expansion coefficient
#: beta [1/K]; saturated liquid, Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table A.6
#: (beta is negative below 4 °C). The same table as the Setup page's.
_WATER = np.array([
    [0.0, 0.561, 4217.0, -0.68e-4],
    [10.0, 0.580, 4192.0, 0.88e-4],
    [20.0, 0.598, 4182.0, 2.07e-4],
    [30.0, 0.615, 4178.0, 3.03e-4],
    [40.0, 0.631, 4179.0, 3.85e-4],
    [50.0, 0.644, 4181.0, 4.58e-4],
    [60.0, 0.654, 4185.0, 5.22e-4],
    [70.0, 0.663, 4190.0, 5.81e-4],
    [80.0, 0.670, 4197.0, 6.32e-4],
    [90.0, 0.675, 4205.0, 6.95e-4],
    [100.0, 0.679, 4216.0, 7.52e-4],
])
#: Air: Sutherland's law for k (k0 0.0241 W/(m K) at 273.15 K, S = 194 K), cp of dry air at 1 atm.
AIR_K0, AIR_T0, AIR_SK, AIR_CP = 0.0241, 273.15, 194.0, 1006.0


def air_conductivity(t_k: float) -> float:
    """Thermal conductivity of air [W/(m K)] at ``t_k`` kelvin (Sutherland's law, within 2 % 160-2000 K)."""
    return AIR_K0 * (t_k / AIR_T0) ** 1.5 * (AIR_T0 + AIR_SK) / (t_k + AIR_SK)


def fluid_defaults(fluid: Fluid, t_ambient_c: float) -> tuple[dict[str, float] | None, str]:
    """({"prandtl", "conductivity", "cp", "beta"}, basis) for a preset fluid at the ambient
    temperature, or (None, why) for a custom fluid (whose name names neither air nor water).

    Every property is taken at the one temperature, and Pr = mu cp / k uses the fluid's
    own viscosity, so the Peclet number is consistent with the run's Reynolds number.
    """
    from ..core import fluid as core_fluid
    t_k = float(t_ambient_c) + 273.15
    name = str(getattr(fluid, "name", "")).lower()
    if name.startswith("air"):
        k = air_conductivity(t_k)
        mu = core_fluid.sutherland_viscosity(t_k)
        return ({"conductivity": k, "cp": AIR_CP, "beta": 1.0 / t_k, "prandtl": mu * AIR_CP / k},
                f"air at {t_ambient_c:g} °C (ideal gas, Sutherland's law)")
    if name.startswith("water"):
        tc = float(np.clip(t_ambient_c, _WATER[0, 0], _WATER[-1, 0]))
        k, cp, beta = (float(np.interp(tc, _WATER[:, 0], _WATER[:, j])) for j in (1, 2, 3))
        mu = core_fluid.water(tc).viscosity
        return ({"conductivity": k, "cp": cp, "beta": beta, "prandtl": mu * cp / k},
                f"water at {tc:g} °C (Incropera & DeWitt, Table A.6)")
    return None, "no defaults for a custom fluid"


@dataclass
class ThermalProps:
    """The thermal properties a run uses: typed values where the user typed them, defaults otherwise."""
    prandtl: float
    conductivity: float
    cp: float
    beta: float
    density: float
    viscosity: float
    basis: str = ""
    known: bool = True                 # False: a custom fluid with an untyped property (placeholders used)
    typed: tuple[str, ...] = ()

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m²/s]."""
        return self.viscosity / self.density

    @property
    def kappa(self) -> float:
        """Thermal diffusivity nu / Pr [m²/s]."""
        return self.nu / self.prandtl


def thermal_props(sim: Simulation) -> ThermalProps:
    """The resolved thermal properties of ``sim``'s fluid at its ambient temperature.

    A custom fluid without typed values gets air's values as placeholders and
    ``known=False``, which the planner turns into a warning: a run must never
    quietly take air's conductivity for some other fluid's.
    """
    th, fl = sim.thermal, sim.conditions.fluid
    defaults, basis = fluid_defaults(fl, th.t_ambient)
    vals: dict[str, float] = {}
    typed, known = [], True
    for key in ("prandtl", "conductivity", "cp", "beta"):
        v = getattr(th, key, None)
        if v is not None and math.isfinite(float(v)):
            vals[key] = float(v)
            typed.append(key)
        elif defaults is not None:
            vals[key] = float(defaults[key])
        else:
            t_k = float(th.t_ambient) + 273.15
            vals[key] = {"prandtl": 0.71, "conductivity": air_conductivity(t_k), "cp": AIR_CP, "beta": 1.0 / t_k}[key]
            known = False
    return ThermalProps(density=float(fl.density), viscosity=float(fl.viscosity), basis=basis, known=known,
                        typed=tuple(typed), **vals)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def rotation3(alpha_deg: float, yaw_deg: float) -> np.ndarray:
    """DESIGN.md 7.2 attitude matrix: yaw about +z by −yaw, then pitch about +y by +alpha
    (the same matrix as ``plan3d.rotation3``; tests/test_simulation_physics.py checks it)."""
    a, b = math.radians(alpha_deg), math.radians(-yaw_deg)
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    return ry @ rz


def _vec(v: Any, dim: int, default: float = 0.0) -> np.ndarray:
    out = np.full(dim, float(default))
    if v is not None:
        a = np.asarray(v, dtype=np.float64).ravel()[:dim]
        out[: len(a)] = a
    return out


def wind_point2(p: Any, design: Design, alpha_deg: float, frame: str) -> np.ndarray:
    """A 2D design-frame point in the wind frame: body items turn by −alpha about the moment point."""
    q = _vec(p, 2)
    if frame != "body" or not alpha_deg:
        return q
    a = math.radians(-alpha_deg)
    c, s = math.cos(a), math.sin(a)
    ref = _vec(design.moment_ref, 2)
    d = q - ref
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]) + ref


def wind_dir2(d: Any, alpha_deg: float, frame: str) -> np.ndarray:
    """A 2D direction in the wind frame (rotated like :func:`wind_point2`, not translated)."""
    v = _vec(d, 2)
    if frame != "body" or not alpha_deg:
        return v
    a = math.radians(-alpha_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def wind_points3(p: Any, design: Design, alpha_deg: float, yaw_deg: float, frame: str, *,
                 direction: bool = False) -> np.ndarray:
    """3D design-frame point(s) (or directions) in the wind frame (``frame="body"`` items turn with
    the geometry about ``Design.moment_ref3``)."""
    q = np.atleast_2d(np.asarray(p, dtype=np.float64))
    q = np.hstack([q[:, :3], np.zeros((len(q), max(0, 3 - q.shape[1])))])
    if frame != "body" or (not alpha_deg and not yaw_deg):
        return q
    r = rotation3(alpha_deg, yaw_deg)
    if direction:
        return q @ r.T
    ref = _vec(design.moment_ref3, 3)
    return (q - ref) @ r.T + ref


# ---------------------------------------------------------------------------
# what is placed, and where
# ---------------------------------------------------------------------------

def _enabled(items) -> list:
    return [x for x in (items or []) if getattr(x, "enabled", True)]


def has_placed_physics(sim: Simulation) -> bool:
    """Whether a simulation uses any DESIGN.md 10 feature (the plan then carries a physics block)."""
    return bool(_enabled(sim.emitters) or sim.thermal.enabled or _enabled(sim.heat_sources) or _enabled(sim.fans)
                or _enabled(sim.ports) or sim.cfd.domain_mode == "closed")


def _heated_bodies(design: Design, dim: int) -> list:
    objs = design.meshes if dim == 3 else design.bodies
    return [b for b in objs if getattr(b, "thermal", "adiabatic") != "adiabatic"]


def item_bounds(design: Design, sim: Simulation, dim: int, alpha_deg: float = 0.0,
                yaw_deg: float = 0.0) -> tuple[float, ...] | None:
    """Bounding box (wind frame, metres) of the enabled placed items: positions, line ends,
    fan and port discs, heater boxes. None when nothing is placed."""
    pts: list[np.ndarray] = []

    def add(p, frame, r=0.0):
        w = (wind_point2(p, design, alpha_deg, frame)[None] if dim == 2 else
             wind_points3(p, design, alpha_deg, yaw_deg, frame))
        pts.append(w - r)
        pts.append(w + r)

    for e in _enabled(sim.emitters):
        add(e.position, e.frame, max(e.radius, max(e.size) if e.size else 0.0))
        if e.end is not None:
            add(e.end, e.frame)
    for q in _enabled(sim.heat_sources):
        add(q.position, q.frame, max(q.size) if q.size else 0.0)
    for f in _enabled(sim.fans):
        add(f.position, f.frame, 0.5 * f.diameter)
    for p in _enabled(sim.ports):
        add(p.position, p.frame, 0.5 * p.size)
    if not pts:
        return None
    a = np.vstack(pts)
    lo, hi = a.min(axis=0), a.max(axis=0)
    return tuple(float(v) for i in range(dim) for v in (lo[i], hi[i]))


def closed_box(design: Design, sim: Simulation, dim: int, body_bounds: Sequence[float] | None,
               alpha_deg: float = 0.0, yaw_deg: float = 0.0) -> tuple[tuple[float, ...], bool]:
    """(box, explicit) of a closed domain, wind frame, metres: ``cfd.domain_box`` when set, else
    the bodies and the placed items with one size of room on every side (the Setup preview's
    rule, ``physics_editor.domain_extent``). ValueError when neither exists."""
    box = getattr(sim.cfd, "domain_box", None)
    if box is not None:
        b = tuple(float(v) for v in box)
        if len(b) != 2 * dim or not all(math.isfinite(v) for v in b) or any(b[2 * i + 1] <= b[2 * i]
                                                                          for i in range(dim)):
            raise ValueError(f"The closed domain's box must hold {2 * dim} finite numbers (min < max per axis), "
                             f"got {box!r}.")
        return b, True
    parts = [x for x in (body_bounds, item_bounds(design, sim, dim, alpha_deg, yaw_deg)) if x is not None]
    if not parts:
        raise ValueError("A closed domain without bodies needs its box: set Domain ▸ Extent (domain_box) to the "
                         "room's walls.")
    lo = [min(p[2 * i] for p in parts) for i in range(dim)]
    hi = [max(p[2 * i + 1] for p in parts) for i in range(dim)]
    size = max(h - l for l, h in zip(lo, hi)) or float(design.length_ref or 1.0)
    return tuple(v for i in range(dim) for v in (lo[i] - size, hi[i] + size)), False


# ---------------------------------------------------------------------------
# scales
# ---------------------------------------------------------------------------

def _surface(kind: str, size: Sequence[float], dim: int) -> tuple[float, float]:
    """(surface [m² or m per metre of span], size [m]) of a heater's shape (the Setup page's rule)."""
    s = [abs(float(v)) for v in size] or [0.1]
    s += [s[-1]] * (3 - len(s))
    if kind in ("disc", "sphere"):
        r = s[0]
        return (2 * math.pi * r, 2 * r) if dim == 2 else (4 * math.pi * r * r, 2 * r)
    if dim == 2:
        return 2 * (s[0] + s[1]), max(s[0], s[1])
    return 2 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]), max(s[:3])


def flux_rise(q: float, ls: float, props: ThermalProps, g_beta: float, u: float) -> float:
    """Temperature rise [K] a heat flux ``q`` [W/m²] off a surface of size ``ls`` [m] produces:
    q ls / (k Nu) with Nu the larger of natural convection off a constant-flux plate,
    0.60 Ra*^(1/5) (Sparrow & Gregg 1956; Ra* = g beta q ls^4 / (k nu kappa)), forced flow over
    a flat plate, 0.664 Re^(1/2) Pr^(1/3) (Pohlhausen), and 1 (conduction). A scale for ΔT_ref,
    not a prediction (the run computes the temperatures)."""
    q, ls = abs(float(q)), max(float(ls), 1e-9)
    if q == 0:
        return 0.0
    k, nu, kappa = props.conductivity, props.nu, props.kappa
    nu_n = 0.60 * (g_beta * q * ls ** 4 / (k * nu * kappa)) ** 0.2 if g_beta > 0 else 0.0
    nu_f = 0.664 * math.sqrt(u * ls / nu) * props.prandtl ** (1.0 / 3.0) if u > 0 else 0.0
    return q * ls / (k * max(nu_n, nu_f, 1.0))


def _heater_sections(kind: str, size: Sequence[float], dim: int) -> tuple[float, float, float]:
    """(plume cross-section, stream cross-section, horizontal width) of a heater [m² or m per metre
    of span]: the area a rising plume (gravity along −y in 2D, −z in 3D) and a stream along +x pass
    through, by the Setup page's sizes (a disc or sphere by its radius ``size[0]``)."""
    s = [abs(float(v)) for v in size] or [0.1]
    s += [s[-1]] * (3 - len(s))
    if kind in ("disc", "sphere"):
        d = 2.0 * s[0]
        a = d if dim == 2 else 0.25 * math.pi * d * d
        return a, a, d
    if dim == 2:
        return s[0], s[1], s[0]
    return s[0] * s[1], s[1] * s[2], math.sqrt(s[0] * s[1])


def heater_rise(power: float, kind: str, size: Sequence[float], props: ThermalProps, g_beta: float,
                u_stream: float, dim: int, *, box: Sequence[float] | None = None,
                u_box: float = 0.0) -> float:
    """Temperature rise [K] a heater of ``power`` [W, or W/m in 2D] produces in the fluid: the
    scale ΔT_ref and the groups on it (β·ΔT, Ra, Ri) are planned with. A heater is a volume source
    in the fluid, not a solid wall, so the air that crosses it carries its heat away:

    * **through-flow**: ΔT = P / (ρ c_p u A) with u A the volume flow through the heater: its own
      plume, u_p = (B / w^(d-2))^(1/3) with the buoyancy flux B = g β P / (ρ c_p) (the line-plume
      speed in 2D, the point plume's at one heater width in 3D; Morton, Taylor & Turner 1956,
      without their order-one constants), or a tunnel's free stream ``u_stream`` across the
      heater's frontal section, whichever moves more air. A closed box's fan or jet is not
      counted, because its stream need not pass the heater; with buoyancy off it is all there is;
    * **closed box**: plus the bulk warming of one turnover, P t / (ρ c_p V) with t = the box's
      longest side over the faster of the plume and ``u_box`` (the box's driving speed);
    * capped by the conduction-limited rise of a solid surface of the heater's size
      (:func:`flux_rise`), which the old estimate used alone.

    The q L / (k Nu) estimate alone planned the Draft 300 W/m fan room at 44.1 K where the run's
    warmest fluid reached 3.3 K (time-averaged; 3.97 K at any instant, critic round 3.4); this
    rule gives 4.1 K. It is a scale, not a prediction: the run measures the real temperatures,
    and a result restates the groups on them (:func:`measured_delta_t`).
    """
    p = abs(float(power))
    if p == 0:
        return 0.0
    rc = props.density * props.cp
    a_plume, a_stream, width = _heater_sections(kind, size, dim)
    area, ls = _surface(kind, size, dim)
    cap = flux_rise(p / max(area, 1e-12), ls, props, g_beta, u_stream)
    u_p = (g_beta * p / rc / max(width, 1e-9) ** (dim - 2)) ** (1.0 / 3.0) if g_beta > 0 else 0.0
    flow = max(u_p * a_plume, abs(float(u_stream)) * a_stream)
    if not flow > 0:
        return cap
    rise = p / (rc * flow)
    if box is not None and len(box) == 2 * dim:
        sides = [float(box[2 * i + 1] - box[2 * i]) for i in range(dim)]
        vol = math.prod(sides)
        u_turn = max(u_p, abs(float(u_box)))
        if vol > 0 and u_turn > 0:
            rise += p * (max(sides) / u_turn) / (rc * vol)
    return min(rise, cap)


@dataclass
class PhysicsScales:
    """The scales and dimensionless groups of a run with placed physics (see the module docstring).

    ``length`` / ``speed`` are the solver's units (L [m], U [m/s]); ``delta_t`` the temperature
    scale [K] (nan without heat transfer); ``length_ra`` the length of Gr and Ra. ``ri`` is the
    solver's Boussinesq coefficient g beta ΔT L / U^2 (0 without buoyancy) and ``ri_flow`` the
    Richardson number of a forced flow (Gr / Re^2 on ``length_ra``, None without one).
    """
    dim: int
    closed: bool
    length: float
    speed: float
    speed_basis: str
    nu: float
    thermal: bool = False
    buoyancy: bool = False
    t_ambient: float = 20.0
    delta_t: float = float("nan")
    delta_t_basis: str = ""
    length_ra: float = float("nan")
    length_ra_basis: str = ""
    props: ThermalProps | None = None
    g_beta: float = 0.0
    u_forced: float = 0.0
    re: float = float("nan")
    pr: float = float("nan")
    pe: float = float("nan")
    gr: float = 0.0
    ra: float = 0.0
    ri: float = 0.0
    ri_flow: float | None = None
    regime: str = ""
    natural: bool = False              # the speed scale is the free-fall (or conduction) velocity
    #: bodies and walls that exchange heat with the fluid (fixed temperature or heat flux): without
    #: one there is no thermal boundary layer to resolve and no Nusselt number to define
    surfaces: int = 0
    #: of those, the bodies (a natural or mixed regime around one is not validated)
    bodies: int = 0

    # -- conversions --------------------------------------------------------
    def theta(self, t_c: float) -> float:
        """Non-dimensional temperature of ``t_c`` °C."""
        return (float(t_c) - self.t_ambient) / self.delta_t

    def celsius(self, theta):
        """°C of a non-dimensional temperature (floats, arrays or tensors)."""
        return theta * self.delta_t + self.t_ambient

    def flux_g(self, q: float) -> float:
        """Dimensionless wall gradient G = q'' L / (k ΔT) of a heat flux [W/m², + into the fluid]."""
        return float(q) * self.length / (self.props.conductivity * self.delta_t)

    @property
    def heat_unit(self) -> float:
        """Watts (3D) or W/m (2D) per solver unit of heat rate: rho cp U ΔT L in 2D (the 2D solver's
        unit), k ΔT L in 3D (the 3D solver's conduction unit)."""
        p = self.props
        if self.dim == 2:
            return p.density * p.cp * self.speed * self.delta_t * self.length
        return p.conductivity * self.delta_t * self.length

    def conduction_unit(self, pe_solver: float | None = None) -> float:
        """Watts (3D) or W/m (2D) per solver unit of a *conducted* heat rate (bodies, walls).

        Why not :attr:`heat_unit`: the 2D solver states every heat rate in rho cp U ΔT L, and its
        Nusselt number is Pe_solver Q / (θ P). When the Reynolds number is capped the solver runs
        at Pe_solver = Re_sim Pr, i.e. a fluid Re_real/Re_sim times more conductive than the real
        one, so rho cp U ΔT L turned every body and wall rate into that fictitious fluid's rate --
        53× the rate its own Nusselt number gives for a NACA 0012 at Re 1.3×10⁵ capped at 2500
        (critic round 5). Converting with k ΔT Pe_solver instead gives Q = Nu k ΔT P / L: the
        capped flow's Nusselt number with the real conductivity, the convention the forces
        already follow (capped coefficients, real dynamic pressure). Uncapped, Pe_solver = U L/α
        and the two units are equal. Heater powers and advected heat (jets, inlet, outlet) keep
        :attr:`heat_unit`: their temperature rise depends on rho cp U, not on the conductivity.
        3D rates are already in the conduction unit k ΔT L of the solver's own Pe."""
        p = self.props
        if self.dim == 2:
            pe = float(pe_solver) if pe_solver is not None and math.isfinite(float(pe_solver)) else float(self.pe)
            if not math.isfinite(pe) or pe <= 0:
                return self.heat_unit
            return p.conductivity * self.delta_t * pe
        return p.conductivity * self.delta_t * self.length

    @property
    def time_unit(self) -> float:
        return self.length / self.speed

    @property
    def beta_dt(self) -> float:
        """β·ΔT_ref, the Boussinesq parameter (nan without heat transfer): every message that
        quotes it reads this one number."""
        if not self.thermal or self.props is None:
            return float("nan")
        return float(self.props.beta) * float(self.delta_t)

    def summary_rows(self) -> dict[str, str]:
        """Label -> value rows for ``SimPlan.summary`` (the Setup page's Estimate keys).

        The values read exactly like the Setup page's own heat-transfer rows
        (``app.widgets.physics_editor.thermal_rows``: a bare "45.0 K", "1.0×10⁵"), because the
        page shows the plan's row when there is one and its own otherwise; the basis of the
        temperature scale is in the plan's notes.
        """
        rows = {"Reference velocity": f"{units.sig(self.speed, 3)} m/s ({self.speed_basis})"}
        if not self.thermal:
            return rows
        rows["Temperature scale"] = f"{units.sig(self.delta_t, 3)} K"
        rows["Prandtl number"] = fmt_number(self.pr)
        if self.gr > 0 and math.isfinite(self.length_ra):
            # Gr, Ra and the Richardson row use H; the notes' solver coefficient Ri uses L
            rows["Length of Gr, Ra and Ri"] = f"{units.sig(self.length_ra, 3)} m ({self.length_ra_basis})"
        if self.gr > 0:
            rows["Grashof number"] = fmt_number(self.gr)
            rows["Rayleigh number"] = fmt_number(self.ra)
        if self.ri_flow is not None:
            rows["Richardson number"] = fmt_number(self.ri_flow)
        rows["Péclet number"] = fmt_number(self.pe)
        rows["Convection"] = self.regime
        return rows


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_number(v: float | None, digits: int = 2) -> str:
    """A dimensionless group as the Estimate shows it: "0.71", "4.5", "1.2×10⁵" (superscript
    exponents outside 0.01 .. 10^4). The same rule as the Setup page's ``fmt_number``, kept here
    because the simulation layer must not import the GUI."""
    if v is None or not math.isfinite(v):
        return "—"
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-2 or a >= 1e4:
        e = math.floor(math.log10(a))
        m = v / 10.0 ** e
        if abs(round(m, digits - 1)) >= 10:
            m, e = m / 10.0, e + 1
        return f"{m:.{digits - 1}f}×10{str(e).translate(_SUPERSCRIPT)}"
    return units.sig(v, digits + (1 if a >= 1 else 0))


def fan_disc_speed(fan, rho: float, dim: int) -> float:
    """A fan's disc velocity [m/s]: the set speed of a speed-controlled fan, else the static
    momentum-theory value, sqrt(T / (2 rho A)) for a thrust (A = the disc's area, or its width
    per metre of span in 2D) and sqrt(dp / (2 rho)) for a pressure jump."""
    if fan.mode == "velocity":
        return abs(float(fan.value))
    if fan.mode == "thrust":
        area = fan.diameter if dim == 2 else math.pi * 0.25 * fan.diameter ** 2
        return math.sqrt(abs(float(fan.value)) / (2.0 * rho * max(area, 1e-12)))
    return math.sqrt(abs(float(fan.value)) / (2.0 * rho))


def _forced_speed(design: Design, sim: Simulation, dim: int) -> tuple[float, str]:
    """(speed, what) of the flow that drives a closed box, or the free stream of a tunnel."""
    if sim.cfd.domain_mode != "closed":
        return float(sim.conditions.speed), "free-stream speed"
    best, what = 0.0, ""
    rho = float(sim.conditions.fluid.density)
    if sim.ground.enabled and sim.ground.moving and sim.conditions.speed > 0:
        best, what = float(sim.conditions.speed), "moving floor"
    for p in _enabled(sim.ports):
        if p.kind == "jet" and abs(p.speed) > best:
            best, what = abs(float(p.speed)), f"jet “{p.name}”"
    for f in _enabled(sim.fans):
        u = fan_disc_speed(f, rho, dim)
        if u > best:
            best, what = u, f"fan “{f.name}”" + ("" if f.mode == "velocity" else " (momentum theory)")
    return best, what


def physics_scales(design: Design, sim: Simulation, dim: int, *,
                   box: Sequence[float] | None = None) -> PhysicsScales:
    """The scales of a run (see the module docstring). ``box`` is the closed domain (wind frame, m).

    Raises ValueError for a closed box that nothing drives (no fan, jet, moving floor or heat)
    and for a heat-transfer setting the solver cannot use (non-positive properties).
    """
    closed = sim.cfd.domain_mode == "closed"
    L = float(design.length_ref)
    nu = float(sim.conditions.fluid.kinematic_viscosity)
    u_forced, u_what = _forced_speed(design, sim, dim)
    th = sim.thermal
    if not th.enabled:
        if closed and u_forced <= 0:
            raise ValueError("Nothing drives the flow in this closed box: add a fan or a jet, heat a wall or a body "
                             "with heat transfer on, move the floor, or switch the domain to a wind tunnel.")
        return PhysicsScales(dim=dim, closed=closed, length=L, speed=u_forced, speed_basis=u_what, nu=nu,
                             re=u_forced * L / nu, regime="Forced", u_forced=u_forced)
    props = thermal_props(sim)
    for k in ("prandtl", "conductivity", "cp", "beta"):
        v = getattr(props, k)
        if not math.isfinite(v) or (k != "beta" and v <= 0):
            raise ValueError(f"Heat transfer needs a positive {k} (got {v!r}).")
    g_beta = float(th.gravity) * props.beta if th.buoyancy else 0.0
    # -- length of Gr and Ra
    heated = _heated_bodies(design, dim)
    if closed and box is not None and not heated:
        up = dim - 1
        lra, lra_basis = float(box[2 * up + 1] - box[2 * up]), "height of the closed box"
    else:
        lra, lra_basis = L, "reference length of the design"
    # -- temperature scale
    objs = design.meshes if dim == 3 else design.bodies
    sides = set(_SIDES3 if dim == 3 else _SIDES2) - (set() if closed else {"left", "right"})
    temps, names = [float(th.t_ambient)], ["ambient"]
    n_bodies = sum(1 for b in objs if getattr(b, "thermal", "adiabatic") in ("temperature", "heat_flux"))
    n_surf = n_bodies + sum(1 for w in th.walls if w.mode in ("temperature", "heat_flux") and w.side in sides)
    for b in objs:
        if b.thermal == "temperature":
            temps.append(float(b.thermal_value))
            names.append(f"body “{b.name}”")
    for w in th.walls:
        if w.mode == "temperature" and w.side in sides:
            temps.append(float(w.value))
            names.append(f"{w.side} wall")
    for p in _enabled(sim.ports):
        if p.kind == "jet" and p.temperature is not None:
            temps.append(float(p.temperature))
            names.append(f"jet “{p.name}”")
    spread = max(temps) - min(temps)
    dt = spread
    dt_basis = (f"{names[int(np.argmax(temps))]} to {names[int(np.argmin(temps))]}" if spread > 0 else "")
    fluxes: list[tuple[float, float, str]] = []
    for b in objs:
        if b.thermal == "heat_flux" and b.thermal_value:
            fluxes.append((float(b.thermal_value), L, f"body “{b.name}”"))
    for w in th.walls:
        if w.mode == "heat_flux" and w.value and w.side in sides:
            fluxes.append((float(w.value), lra, f"{w.side} wall"))
    for q, ls, what in fluxes:
        rise = flux_rise(q, ls, props, g_beta, u_forced)
        if rise > dt:
            dt, dt_basis = rise, f"{ESTIMATED_RISE}{what}: q·L/(k·Nu)"
    for q in _enabled(sim.heat_sources):
        if q.power:
            # a closed box's fan or jet need not blow through the heater; with buoyancy off it must
            u_stream = 0.0 if (closed and g_beta > 0) else u_forced
            rise = heater_rise(float(q.power), q.kind, q.size, props, g_beta, u_stream, dim,
                               box=box if closed else None, u_box=u_forced)
            if rise > dt:
                dt, dt_basis = rise, f"{ESTIMATED_RISE}heater “{q.name}”: {HEATER_RISE_BASIS}"
    if dt <= 0:
        dt, dt_basis = 1.0, "1 K (nothing is heated or cooled)"
    gr = g_beta * dt * lra ** 3 / nu ** 2 if g_beta > 0 else 0.0
    ra = gr * props.prandtl
    natural = False
    if not closed or u_forced > 0:
        u, u_basis = u_forced, u_what
    elif g_beta > 0 and ra >= RA_CONDUCTION:
        u, u_basis, natural = math.sqrt(g_beta * dt * lra), "free-fall velocity √(gβΔT H)", True
    else:
        u, u_basis, natural = props.kappa / lra, "conduction velocity κ/H", True
    if not (u > 0 and math.isfinite(u)):
        raise ValueError("Nothing drives the flow: the free stream, the jets and the fans are all at rest.")
    re = u * L / nu
    ri_flow = g_beta * dt * lra / u_forced ** 2 if (g_beta > 0 and u_forced > 0) else None
    regime = _regime(bool(th.buoyancy), closed, u_forced, ra, ri_flow)
    return PhysicsScales(dim=dim, closed=closed, length=L, speed=u, speed_basis=u_basis, nu=nu, thermal=True,
                         buoyancy=bool(th.buoyancy), t_ambient=float(th.t_ambient), delta_t=dt, delta_t_basis=dt_basis,
                         length_ra=lra, length_ra_basis=lra_basis, props=props, g_beta=g_beta, u_forced=u_forced,
                         re=re, pr=props.prandtl, pe=re * props.prandtl, gr=gr, ra=ra,
                         ri=g_beta * dt * L / u ** 2 if g_beta > 0 else 0.0, ri_flow=ri_flow, regime=regime,
                         natural=natural, surfaces=n_surf, bodies=n_bodies)


def _regime(buoyancy: bool, closed: bool, u_forced: float, ra: float, ri_flow: float | None) -> str:
    """"Forced", "Mixed", "Natural" or "Conduction": from Ri = Gr/Re² with a forced flow (the usual
    0.1 and 10 bounds), from Ra in a box nothing else drives."""
    if not buoyancy:
        return "Forced (no buoyancy)" if u_forced > 0 else "Conduction (no buoyancy)"
    if closed and u_forced == 0:
        return "Conduction" if ra < RA_CONDUCTION else "Natural"
    if ri_flow is None:
        return "Natural"
    return "Forced" if ri_flow < 0.1 else ("Mixed" if ri_flow <= 10.0 else "Natural")


#: How :func:`physics_scales` words a temperature scale it estimated from a heat flux or a heater
#: (q L / (k Nu), :func:`flux_rise`) rather than read off the set temperatures.
ESTIMATED_RISE = "estimated rise at the "
#: How a heater's estimated rise (:func:`heater_rise`) is described after :data:`ESTIMATED_RISE`.
HEATER_RISE_BASIS = "P/(ρ·c_p·u·A), the air its plume or the stream carries through it"


def delta_t_is_estimate(scales: PhysicsScales | None) -> bool:
    """Whether the run's ΔT_ref is the planner's estimate of a heat flux's or a heater's rise (then
    the run's measured temperatures replace it in what a result states), not a set difference."""
    return bool(scales is not None and scales.thermal and str(scales.delta_t_basis).startswith(ESTIMATED_RISE))


def with_delta_t(scales: PhysicsScales, delta_t: float, basis: str) -> PhysicsScales:
    """A copy of ``scales`` that states the groups on another temperature difference: ΔT, Gr, Ra,
    Ri = Gr/Re² and the regime are re-derived, while the solver's own scales (U, Re, Pe and the
    Boussinesq coefficient ``ri``) stay those of the run.

    For reporting only: the copy's :meth:`PhysicsScales.celsius` no longer undoes the solver's
    θ (that needs the ΔT_ref the run was solved with), so never convert fields with it."""
    import dataclasses
    s = scales
    dt = float(delta_t)
    gr = s.g_beta * dt * s.length_ra ** 3 / s.nu ** 2 if s.g_beta > 0 else 0.0
    ra = gr * (s.pr if math.isfinite(s.pr) else 0.0)
    ri_flow = s.g_beta * dt * s.length_ra / s.u_forced ** 2 if (s.g_beta > 0 and s.u_forced > 0) else None
    return dataclasses.replace(s, delta_t=dt, delta_t_basis=basis, gr=gr, ra=ra, ri_flow=ri_flow,
                               regime=_regime(s.buoyancy, s.closed, s.u_forced, ra, ri_flow))


def measured_delta_t(scales: PhysicsScales | None, diagnostics: dict | None) -> tuple[float, str] | None:
    """(ΔT [K], what it is) the run measured: the larger of the warmest fluid's rise above the
    ambient and the coldest fluid's drop below it (``t_fluid_max`` / ``t_fluid_min`` of the result's
    diagnostics, time-averaged when the run averaged). None without a thermal run or those values.

    This is the scale a heater's room or a heat-flux body actually reached: the Draft fan room's
    300 W/m heater was planned at ΔT 43.4 K (q L / (k Nu)) and its warmest fluid was 2.96 K above
    the ambient on the time average (3.48 K at the hottest instant; critic round 3.3), so Ra and Gr
    on the plan's ΔT were about 15 times too large. The instantaneous excursion
    (``t_fluid_dev_max``) is what the Boussinesq check uses; Ra describes the mean flow."""
    if scales is None or not scales.thermal:
        return None
    d = diagnostics or {}
    t_hi, t_lo = d.get("t_fluid_max"), d.get("t_fluid_min")
    try:
        t_hi, t_lo, t0 = float(t_hi), float(t_lo), float(scales.t_ambient)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(t_hi) and math.isfinite(t_lo)):
        return None
    rise, drop = t_hi - t0, t0 - t_lo
    if not max(rise, drop) > 0:
        return None
    if rise >= drop:
        return rise, f"measured: the warmest fluid reached {units.sig(t_hi, 4)} °C at a {units.sig(t0, 4)} °C ambient"
    return drop, f"measured: the coldest fluid reached {units.sig(t_lo, 4)} °C at a {units.sig(t0, 4)} °C ambient"


def result_scales(scales: PhysicsScales | None, result: Any) -> tuple[PhysicsScales | None, str]:
    """(scales, "measured" | "planned estimate" | "") for the groups a result states next to its
    numbers (a Results page's Conditions, a report).

    When the plan's ΔT_ref was only an estimate (:func:`delta_t_is_estimate`) and the run measured
    its temperatures, the copy states ΔT, Gr, Ra and Ri on the measured rise
    (:func:`measured_delta_t`, :func:`with_delta_t`) and the label is "measured"; without the
    measurement it is "planned estimate". "" when ΔT is set by the walls, bodies and jets
    (the cavity's T_hot − T_cold), which the run then holds exactly."""
    if scales is None or not scales.thermal:
        return scales, ""
    if not delta_t_is_estimate(scales):
        return scales, ""
    m = measured_delta_t(scales, getattr(result, "diagnostics", None))
    if m is None:
        return scales, "planned estimate"
    return with_delta_t(scales, m[0], m[1]), "measured"


def thermal_estimate(design: Design, sim: Simulation, *, extent: Sequence[float] | None = None
                     ) -> PhysicsScales | None:
    """The heat-transfer scales of a setup exactly as the planner derives them, for any page that
    states ΔT, β·ΔT, Gr, Ra, Ri or Pe before (or without) a plan; None when heat transfer is off or
    the setup cannot be scaled (the planner then says why).

    ``extent`` is the planned domain (wind frame, m) when one is known; a closed box uses its
    ``cfd.domain_box`` first, as the planner does. Deriving these numbers once, here, is what
    keeps the Setup page's estimate, its warnings and the run's notes from quoting three
    different temperature scales for one room (a thrust fan's momentum-theory speed changes a
    heater's estimated rise, and with it β·ΔT and Ra). Cheap: numpy only, no plan is built.
    """
    if not getattr(getattr(sim, "thermal", None), "enabled", False):
        return None
    dim = 3 if getattr(design, "dimension", 2) == 3 else 2
    box = None
    if sim.cfd.domain_mode == "closed":
        raw = getattr(sim.cfd, "domain_box", None)
        cand = raw if raw is not None else extent
        if cand is not None and len(cand) == 2 * dim:
            b = tuple(float(v) for v in cand)
            if all(math.isfinite(v) for v in b) and all(b[2 * i + 1] > b[2 * i] for i in range(dim)):
                box = b
    try:
        return physics_scales(design, sim, dim, box=box)
    except (ValueError, ZeroDivisionError):
        return None


#: Far-wake speed of an actuator disc over its disc velocity: momentum theory's slipstream
#: doubles the induced velocity (Glauert), so a fan's jet runs at up to twice its disc velocity.
DISC_WAKE_FACTOR = 2.0
#: Richardson number Gr / Re^2 above which a tunnel run is buoyancy-dominated: the plume, not the
#: free stream, sets the speeds (the usual mixed-convection bounds are 0.1 and 10).
RI_BUOYANT = 10.0


def buoyant_speed(scales: PhysicsScales) -> float:
    """The speed buoyancy drives [m/s], 0 without it: the free-fall velocity sqrt(g beta ΔT H)
    when inertia limits it (Gr >= 1), the viscous speed sqrt(Gr) times that below (the Stokes
    balance g beta ΔT H^2 / nu), the same ΔT_ref and H as Gr and Ra."""
    if not (scales.thermal and scales.buoyancy and scales.g_beta > 0):
        return 0.0
    lra = scales.length_ra if math.isfinite(scales.length_ra) else scales.length
    u_ff = math.sqrt(max(scales.g_beta * scales.delta_t * lra, 0.0))
    return u_ff * min(1.0, math.sqrt(max(scales.gr, 0.0)))


def peak_speed(scales: PhysicsScales, sim: Simulation, dim: int) -> tuple[float, str]:
    """(u_peak [m/s], what sets it): the fastest air a run should be expected to reach, which
    sizes the planned time step and so the run-time estimate (:func:`peak_source` says which
    kind of source it is).

    The largest of the reference velocity U, the buoyant speed (:func:`buoyant_speed`), twice
    each fan's disc velocity (:data:`DISC_WAKE_FACTOR`), the jets and a moving floor. Sizing
    the step from U alone planned the electronics enclosure (a 2 m/s fan, 1.5 m/s jets) at
    0.0011 L/U where the run took 0.00028, and the hot sphere at Ri 627, whose plume moves 25
    times faster than its 0.015 m/s free stream, at 17 times its real step (critic round 3.2).
    The coefficients and the Reynolds number keep U; only the time step uses this.
    """
    U = float(scales.speed)
    best, what = U, scales.speed_basis or "reference velocity"
    ub = buoyant_speed(scales)
    if ub > best:
        best, what = ub, "buoyant velocity √(gβΔT H)"
    rho = float(sim.conditions.fluid.density)
    for f in _enabled(sim.fans):
        u = DISC_WAKE_FACTOR * fan_disc_speed(f, rho, dim)
        if u > best:
            best, what = u, f"slipstream of fan “{f.name}” (2 × its disc velocity)"
    for p in _enabled(sim.ports):
        if p.kind == "jet" and abs(float(p.speed)) > best:
            best, what = abs(float(p.speed)), f"jet “{p.name}”"
    if sim.ground.enabled and sim.ground.moving and abs(float(sim.conditions.speed)) > best:
        best, what = abs(float(sim.conditions.speed)), "moving floor"
    return best, what


def peak_source(basis: str) -> str:
    """"plume", "fan", "jet", "floor" or "reference": the kind of source a :func:`peak_speed`
    basis names (the planners keep a separate measured speed factor per kind)."""
    b = str(basis)
    if b.startswith("buoyant"):
        return "plume"
    if b.startswith("slipstream of fan"):
        return "fan"
    if b.startswith("jet"):
        return "jet"
    if b.startswith("moving floor"):
        return "floor"
    return "reference"


def default_ambient(fluid: Fluid, fallback: float = 20.0) -> float:
    """The ambient temperature [°C] heat transfer should default to for ``fluid``: the
    temperature a preset fluid was made at (15 °C for ISA sea level), ``fallback`` for a custom
    one. A default run whose ambient and fluid disagree carries a warning it did not ask for."""
    t = fluid_temperature(fluid)
    return float(t) if t is not None and math.isfinite(t) else float(fallback)


_FLUID_T = re.compile(r"^(?:Air|Water), (-?[\d.]+) C$")
_FLUID_ISA = re.compile(r"^Air, ISA (sea level|[\d,.]+ m)$")


def fluid_temperature(fluid: Fluid) -> float | None:
    """The temperature [°C] a preset fluid was made at, or None for a custom fluid.

    ``core.fluid`` writes it into the name ("Air, 20 C", "Water, 20 C", "Air, ISA sea level" =
    15 °C, "Air, ISA 1,000 m"). With heat transfer on, a fluid made at another temperature than
    the ambient gives the run the density and viscosity of one temperature and the conductivity,
    cp and β of another (:func:`thermal_checks` says so).
    """
    name = str(getattr(fluid, "name", ""))
    m = _FLUID_T.match(name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m = _FLUID_ISA.match(name)
    if m:
        from ..core import fluid as core_fluid
        try:
            alt = 0.0 if m.group(1) == "sea level" else float(m.group(1)[:-2].replace(",", ""))
            return float(core_fluid.isa(alt)[0]) - 273.15
        except (ValueError, TypeError, IndexError):
            return None
    return None


#: Kelvin between the fluid's own temperature and the ambient above which :func:`thermal_checks`
#: warns (2 K moves air's viscosity by about 1 % and its density by about 0.7 %).
FLUID_AMBIENT_TOL_K = 2.0


def cavity_nusselt(ra: float, pr: float) -> float:
    """Mean Nusselt number of a differentially heated cavity by :data:`CAVITY_NU` (>= 1)."""
    a, b, c = CAVITY_NU
    x = pr * ra / (b + pr)
    return max(1.0, a * x ** c) if x > 0 else 1.0


# ---------------------------------------------------------------------------
# the specs
# ---------------------------------------------------------------------------

@dataclass
class Placed:
    """The placed physics of one plan in solver units, plus what the outputs need to undo it.

    ``emitters`` .. ``ports`` are the solver specs (2D ``ns2d.config`` or 3D ``ns3d.config``
    types); ``body_thermal`` one (mode, value) per wind-frame body in solver units;
    ``dye_channels`` the smoke channels (``viz.colormaps.dye_channels`` entries, jets with smoke
    included); ``particle_palette`` one colour per particle emitter, in the solver's emitter
    order; ``fans_si`` / ``ports_si`` the model objects in solver order (for SI reports).
    """
    emitters: list = field(default_factory=list)
    thermal: Any = None
    heat_sources: list = field(default_factory=list)
    fans: list = field(default_factory=list)
    ports: list = field(default_factory=list)
    body_thermal: list[tuple[str, float]] = field(default_factory=list)
    dye_channels: list[dict] = field(default_factory=list)
    particle_palette: list[str] = field(default_factory=list)
    particle_names: list[str] = field(default_factory=list)
    fans_si: list = field(default_factory=list)
    ports_si: list = field(default_factory=list)
    heaters_si: list = field(default_factory=list)
    dye_inlet: bool = True
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return bool(self.emitters or self.thermal is not None or self.heat_sources or self.fans or self.ports)


def _channels(sim: Simulation) -> tuple[list[dict], dict[int, int], int | None]:
    """(channels, {emitter index: channel}, channel of jet smoke): smoke maps to channels by colour
    (``viz.colormaps.dye_channels``, the legend's rule); jets carrying smoke use the default
    first colour, so they share channel 0 with white emitters."""
    from ..viz.colormaps import DYE_COLORS, DYE_FIELDS, dye_channels, hex_rgb

    class _Jet:                      # a jet's smoke, as an emitter of the default colour
        species, enabled, color = "dye", True, DYE_COLORS[0]

        def __init__(self, name):
            self.name = name

    smoky = [p for p in _enabled(sim.ports) if p.kind == "jet" and p.dye]
    dyes = [e for e in sim.emitters if e.species == "dye" and e.enabled]
    chans = dye_channels(dyes + [_Jet(p.name) for p in smoky])
    by_rgb: dict[tuple, int] = {}
    for i, ch in enumerate(chans):
        by_rgb[hex_rgb(ch["color"])] = i
    mapping: dict[int, int] = {}
    for idx, e in enumerate(sim.emitters):
        if e.species != "dye" or not e.enabled:
            continue
        try:
            rgb = hex_rgb(e.color or DYE_COLORS[0])
        except ValueError:
            rgb = hex_rgb(DYE_COLORS[0])
        mapping[idx] = by_rgb.get(rgb, len(chans) - 1)
    jet = by_rgb.get(hex_rgb(DYE_COLORS[0]), len(chans) - 1) if smoky else None
    for i, ch in enumerate(chans):
        ch["field"] = DYE_FIELDS[i]
    return chans, mapping, jet


def _colour_count(sim: Simulation) -> int:
    """Distinct smoke colours the case asks for (enabled smoke emitters, plus the default colour of
    smoky jets). ``dye_channels`` folds every colour past the fourth into the last channel, so its
    length cannot tell that some were merged; this count can."""
    from ..viz.colormaps import DYE_COLORS, hex_rgb
    seen = set()
    for e in sim.emitters:
        if e.species == "dye" and e.enabled:
            try:
                seen.add(hex_rgb(e.color or DYE_COLORS[0]))
            except ValueError:
                seen.add(hex_rgb(DYE_COLORS[0]))
    if any(p.kind == "jet" and p.dye for p in _enabled(sim.ports)):
        seen.add(hex_rgb(DYE_COLORS[0]))
    return len(seen)


def _colour_warning(sim: Simulation, out: Placed) -> None:
    n = _colour_count(sim)
    if n > len(out.dye_channels) and out.dye_channels:
        out.warnings.append(f"{n} smoke colours, but the solver carries {len(out.dye_channels)} smoke channels: the "
                            f"colours past the fourth share the channel of “{out.dye_channels[-1]['color']}” (they "
                            "are drawn in its colour).")


def _inside(p: np.ndarray, extent: Sequence[float], tol: float) -> bool:
    return all(extent[2 * i] - tol <= p[i] <= extent[2 * i + 1] + tol for i in range(len(p)))


def placed_physics_2d(design: Design, sim: Simulation, scales: PhysicsScales, *, alpha_deg: float,
                      extent: Sequence[float], h_m: float, bodies_wind: Sequence | None = None,
                      closed: bool = False) -> Placed:
    """The 2D solver specs of ``sim``'s placed physics (``ns2d.config`` types; module docstring).

    ``extent`` is the planned domain (wind frame, metres) and ``h_m`` its cell size: items
    outside are dropped with a warning (they would act on nothing), and items the grid cannot
    resolve are reported. ``bodies_wind`` (the planner's wind-frame bodies) get their thermal
    condition in ``body_thermal``.
    """
    from ..solvers.ns2d.config import EmitterSpec, FanSpec, HeatSourceSpec, PortSpec, ThermalConfig, WallBC
    out = Placed()
    L, U = scales.length, scales.speed
    rho = float(sim.conditions.fluid.density)
    tol = 1e-9 * max(extent[1] - extent[0], extent[3] - extent[2])

    def nd(p) -> tuple[float, float]:
        return (float(p[0]) / L, float(p[1]) / L)

    def outside(what: str, name: str) -> None:
        out.warnings.append(f"{what} “{name}” is outside the domain, so it has no effect. Move it inside the "
                            "domain (the dashed frame in Setup).")

    # -- smoke and particles
    chans, mapping, jet_channel = _channels(sim)
    out.dye_channels = chans
    for idx, e in enumerate(sim.emitters):
        if not e.enabled:
            continue
        pos = wind_point2(e.position, design, alpha_deg, e.frame)
        end = wind_point2(e.end, design, alpha_deg, e.frame) if e.end is not None else None
        kind, size = e.kind, tuple(abs(float(s)) for s in e.size)
        if kind == "plane":
            # the 2D analogue of a smoke plane: a slot across the domain, normal to ``normal``
            n = wind_dir2(e.normal if e.normal is not None else (1.0, 0.0), alpha_deg, e.frame)
            n = n / max(float(np.hypot(*n)), 1e-12)
            t = np.array([-n[1], n[0]])
            half = 0.5 * size[0] if size else max(extent[1] - extent[0], extent[3] - extent[2])
            a, b = pos - half * t, pos + half * t
            # clip to the domain (a slot through a wall would release smoke into the wall)
            lo, hi = np.array([extent[0], extent[2]]), np.array([extent[1], extent[3]])
            a, b = np.clip(a, lo + 0.5 * h_m, hi - 0.5 * h_m), np.clip(b, lo + 0.5 * h_m, hi - 0.5 * h_m)
            kind, end, pos = "line", b, a
        if kind in ("line", "rake") and end is None:
            out.warnings.append(f"Emitter “{e.name}” is a {kind} without an end point; it was left out.")
            continue
        centre = 0.5 * (pos + end) if end is not None else pos
        if not _inside(centre, extent, tol):
            outside("Emitter", e.name)
            continue
        if kind == "rect" and len(size) < 2:
            size = (size[0], size[0]) if size else (2 * h_m, 2 * h_m)
        if kind == "disc" and not size:
            size = (2 * h_m,)
        if kind == "rect" and e.frame == "body" and alpha_deg:
            out.notes.append(f"Emitter “{e.name}” (a box) moves with the body but stays aligned with the axes.")
        common = {"position": nd(pos), "kind": kind, "end": nd(end) if end is not None else None,
                  "size": tuple(s / L for s in size), "count": max(1, int(e.count)), "radius": float(e.radius) / L,
                  "pulse_period": float(e.pulse_period) * U / L, "pulse_duty": float(e.pulse_duty),
                  "start": float(e.start) * U / L, "stop": None if e.stop is None else float(e.stop) * U / L,
                  "name": e.name}
        if e.species == "dye":
            rate = float(e.rate)
            if not 0.0 <= rate <= 1.0:
                out.notes.append(f"Smoke emitter “{e.name}”: concentration {rate:g} clipped to {min(max(rate, 0), 1):g} "
                                 "(smoke is a concentration from 0 to 1).")
                rate = min(max(rate, 0.0), 1.0)
            spec = EmitterSpec(species="dye", channel=mapping.get(idx, 0), rate=rate, **common)
        else:
            spec = EmitterSpec(species="particles", rate=max(float(e.rate), 0.0) * L / U, **common)
            out.particle_palette.append(str(e.color or "#f5f5f5"))
            out.particle_names.append(e.name)
        out.emitters.append(spec)
    if any(e.species == "dye" for e in out.emitters) or jet_channel is not None:
        out.dye_inlet = False                 # the user's smoke replaces the automatic inlet comb
    _colour_warning(sim, out)

    # -- heat transfer
    th = sim.thermal
    if th.enabled:
        walls = []
        for w in th.walls:
            if w.side not in _SIDES2:
                out.notes.append(f"The {w.side} wall condition applies to 3D runs only.")
                continue
            if not closed and w.side in ("left", "right"):
                out.notes.append(f"In a wind tunnel the {w.side} side is the "
                                 f"{'inlet (held at the ambient temperature)' if w.side == 'left' else 'outlet'}; "
                                 "its wall condition applies to closed domains only.")
                continue
            if w.mode == "temperature":
                walls.append(WallBC(w.side, "temperature", scales.theta(w.value)))
            elif w.mode == "heat_flux":
                walls.append(WallBC(w.side, "heat_flux", scales.flux_g(w.value)))
        out.thermal = ThermalConfig(prandtl=scales.pr, richardson=scales.ri if th.buoyancy else 0.0,
                                    gravity=(0.0, -1.0), walls=walls, theta_inlet=0.0, theta_initial=0.0)
        for b in bodies_wind or []:
            mode = getattr(b, "thermal", "adiabatic")
            val = float(getattr(b, "thermal_value", 0.0))
            if mode == "temperature":
                out.body_thermal.append(("temperature", scales.theta(val)))
            elif mode == "heat_flux":
                out.body_thermal.append(("heat_flux", scales.flux_g(val)))
            else:
                out.body_thermal.append(("adiabatic", 0.0))
        for q in _enabled(sim.heat_sources):
            pos = wind_point2(q.position, design, alpha_deg, q.frame)
            if not _inside(pos, extent, tol):
                outside("Heat source", q.name)
                continue
            kind = {"box": "rect", "sphere": "disc"}.get(q.kind, q.kind)
            s = [abs(float(v)) for v in q.size] or [0.1]
            size = (s[0], s[1] if len(s) > 1 else s[0]) if kind == "rect" else (s[0],)
            # a body-frame rect turns with the body (its first axis along the turned chord line)
            axis = wind_dir2((1.0, 0.0), alpha_deg, q.frame)
            out.heat_sources.append(HeatSourceSpec(position=(pos[0] / L, pos[1] / L), kind=kind,
                                                   size=tuple(v / L for v in size),
                                                   power=float(q.power) / scales.heat_unit, name=q.name,
                                                   direction=(float(axis[0]), float(axis[1]))))
            out.heaters_si.append(q)
            if min(size) < 2 * h_m:
                out.warnings.append(f"Heat source “{q.name}” is {units.sig(1000 * min(size), 3)} mm across, under "
                                    "two grid cells: its heat is released over the cells it touches (the total is "
                                    "exact, the shape is not).")
    else:
        out.body_thermal = [("adiabatic", 0.0)] * len(bodies_wind or [])

    # -- fans
    for f in _enabled(sim.fans):
        pos = wind_point2(f.position, design, alpha_deg, f.frame)
        d = wind_dir2(f.direction, alpha_deg, f.frame)
        if float(np.hypot(*d)) < 1e-12:
            out.warnings.append(f"Fan “{f.name}” has no direction in the plane of the 2D run; it was left out.")
            continue
        if not _inside(pos, extent, tol):
            outside("Fan", f.name)
            continue
        if f.mode == "thrust":
            value = float(f.value) / (rho * U * U * L)
        elif f.mode == "pressure_jump":
            value = float(f.value) / (rho * U * U)
        else:
            value = float(f.value) / U
        out.fans.append(FanSpec(position=(pos[0] / L, pos[1] / L), direction=tuple(d), width=float(f.diameter) / L,
                                thickness=None if f.thickness is None else float(f.thickness) / L, mode=f.mode,
                                value=value, name=f.name))
        out.fans_si.append(f)
        cells = f.diameter / h_m
        if cells < FAN_CELLS_MIN:
            out.warnings.append(f"Fan “{f.name}” spans {cells:.1f} grid cells (about {FAN_CELLS_MIN:.0f} resolve "
                                "its disc): its thrust is delivered, but the induced velocity and the flow rate are "
                                "indicative only. A finer grid or a larger fan resolves it.")
        if f.swirl:
            out.notes.append(f"Fan “{f.name}”: swirl acts in 3D only; a 2D fan pushes along its direction.")

    # -- ports
    for p in _enabled(sim.ports):
        pos = wind_point2(p.position, design, alpha_deg, p.frame)
        d = wind_dir2(p.direction, alpha_deg, p.frame)
        if float(np.hypot(*d)) < 1e-12:
            out.warnings.append(f"{p.kind.capitalize()} “{p.name}” has no direction in the plane; it was left out.")
            continue
        if not _inside(pos, extent, tol):
            outside(p.kind.capitalize(), p.name)
            continue
        theta = None
        if p.kind == "jet" and p.temperature is not None:
            if th.enabled:
                theta = scales.theta(p.temperature)
            else:
                out.notes.append(f"Jet “{p.name}” has a temperature, but heat transfer is off: it blows ambient air.")
        out.ports.append(PortSpec(position=(pos[0] / L, pos[1] / L), direction=tuple(d), size=float(p.size) / L,
                                  speed=abs(float(p.speed)) / U, kind=p.kind, theta=theta,
                                  dye_channel=jet_channel if (p.kind == "jet" and p.dye) else None, name=p.name))
        out.ports_si.append(p)
        if p.size / h_m < PORT_CELLS_MIN:
            out.warnings.append(f"{p.kind.capitalize()} “{p.name}” is {p.size / h_m:.1f} grid cells wide (about "
                                f"{PORT_CELLS_MIN:.0f} resolve its profile): its volume flow is exact, its velocity "
                                "profile is not.")
    _balance(out, closed, dim=2)
    return out


def _balance(out: Placed, closed: bool, dim: int) -> None:
    """Warn when a closed box's jets and vents do not balance (2D: m²/s per metre; 3D: m³/s)."""
    if not closed:
        return

    def flow(p) -> float:
        s = abs(float(p.size))
        return abs(float(p.speed)) * (s if dim == 2 else math.pi * 0.25 * s * s)
    q_in = sum(flow(p) for p in out.ports_si if p.kind == "jet")
    q_out = sum(flow(p) for p in out.ports_si if p.kind == "vent")
    if (q_in or q_out) and abs(q_in - q_out) > BALANCE_RTOL * max(q_in, q_out):
        unit = "m²/s per metre of span" if dim == 2 else "m³/s"
        out.warnings.append(f"The jets supply {units.sig(q_in, 3)} {unit} and the vents remove {units.sig(q_out, 3)}: "
                            "a closed box must balance, so the solver removes the difference evenly over the whole "
                            "box (the pressure problem has no solution otherwise). Match the openings or speeds.")


def placed_physics_3d(design: Design, sim: Simulation, scales: PhysicsScales, *, alpha_deg: float, yaw_deg: float,
                      extent3: Sequence[float], h_m: float, bodies_wind: Sequence | None = None,
                      closed: bool = False) -> Placed:
    """The 3D solver specs of ``sim``'s placed physics (``ns3d.config`` types; module docstring).

    ``extent3`` is the solved domain (wind frame, metres). With a symmetry plane the planner
    does not get here with placed items (it runs the full domain, see plan3d)."""
    from ..solvers.ns3d.config import Emitter3D, Fan3D, HeatSource3D, Port3D, ThermalConfig3D
    out = Placed()
    L, U = scales.length, scales.speed
    rho = float(sim.conditions.fluid.density)
    tol = 1e-9 * max(extent3[1] - extent3[0], extent3[3] - extent3[2], extent3[5] - extent3[4])

    def wp(p, frame):
        return wind_points3(_vec(p, 3), design, alpha_deg, yaw_deg, frame)[0]

    def wd(d, frame):
        v = wind_points3(_vec(d, 3), design, alpha_deg, yaw_deg, frame, direction=True)[0]
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def outside(what: str, name: str) -> None:
        out.warnings.append(f"{what} “{name}” is outside the domain, so it has no effect. Move it inside the "
                            "domain (the dashed box in Setup).")

    chans, mapping, jet_channel = _channels(sim)
    out.dye_channels = chans
    for idx, e in enumerate(sim.emitters):
        if not e.enabled:
            continue
        pos = wp(e.position, e.frame)
        end = wp(e.end, e.frame) if e.end is not None else None
        if e.kind in ("line", "rake") and end is None:
            out.warnings.append(f"Emitter “{e.name}” is a {e.kind} without an end point; it was left out.")
            continue
        centre = 0.5 * (pos + end) if end is not None else pos
        if not _inside(centre, extent3, tol):
            outside("Emitter", e.name)
            continue
        size = tuple(abs(float(s)) for s in e.size)
        kind = e.kind
        if kind == "rect" and len(size) != 3:
            size = (tuple(size) + (size[-1] if size else 2 * h_m,) * 3)[:3]
        if kind == "disc" and not size:
            size = (2 * h_m,)
        if kind == "plane" and len(size) not in (0, 2):
            size = (size[0], size[0]) if size else ()
        normal = wd(e.normal if e.normal is not None else (1.0, 0.0, 0.0), e.frame)
        if kind == "rect" and e.frame == "body" and (alpha_deg or yaw_deg):
            out.notes.append(f"Emitter “{e.name}” (a box) moves with the body but stays aligned with the axes.")
        rate = float(e.rate)
        if e.species == "dye" and not 0.0 <= rate <= 1.0:
            out.notes.append(f"Smoke emitter “{e.name}”: concentration {rate:g} clipped to {min(max(rate, 0), 1):g}.")
            rate = min(max(rate, 0.0), 1.0)
        spec = Emitter3D(species=e.species, kind=kind, position=tuple(pos / L),
                         end=None if end is None else tuple(end / L), size=tuple(s / L for s in size),
                         normal=tuple(normal), count=max(1, int(e.count)), radius=float(e.radius) / L,
                         rate=rate if e.species == "dye" else max(rate, 0.0) * L / U,
                         channel=mapping.get(idx, 0) if e.species == "dye" else 0,
                         pulse_period=float(e.pulse_period) * U / L, pulse_duty=float(e.pulse_duty),
                         start=float(e.start) * U / L, stop=None if e.stop is None else float(e.stop) * U / L,
                         name=e.name)
        if e.species == "particles":
            out.particle_palette.append(str(e.color or "#f5f5f5"))
            out.particle_names.append(e.name)
        out.emitters.append(spec)
    out.dye_inlet = not (any(s.species == "dye" for s in out.emitters) or jet_channel is not None)
    _colour_warning(sim, out)

    th = sim.thermal
    if th.enabled:
        walls: dict[str, tuple[str, float]] = {}
        for w in th.walls:
            if not closed and w.side in ("left", "right"):
                out.notes.append(f"In a wind tunnel the {w.side} side is the "
                                 f"{'inlet (held at the ambient temperature)' if w.side == 'left' else 'outlet'}; "
                                 "its wall condition applies to closed domains only.")
                continue
            if w.mode == "temperature":
                walls[w.side] = ("temperature", scales.theta(w.value))
            elif w.mode == "heat_flux":
                walls[w.side] = ("heat_flux", scales.flux_g(w.value))
        out.thermal = ThermalConfig3D(prandtl=scales.pr, richardson=scales.ri if th.buoyancy else 0.0, walls=walls,
                                      theta_inlet=0.0, theta_initial=0.0)
        for b in bodies_wind or []:
            mode = getattr(b, "thermal", "adiabatic")
            val = float(getattr(b, "thermal_value", 0.0))
            out.body_thermal.append(("temperature", scales.theta(val)) if mode == "temperature" else
                                    ("heat_flux", scales.flux_g(val)) if mode == "heat_flux" else ("adiabatic", 0.0))
        for q in _enabled(sim.heat_sources):
            pos = wp(q.position, q.frame)
            if not _inside(pos, extent3, tol):
                outside("Heat source", q.name)
                continue
            kind = {"rect": "box", "disc": "sphere"}.get(q.kind, q.kind)
            s = [abs(float(v)) for v in q.size] or [0.1]
            size = tuple((s + [s[-1]] * 3)[:3]) if kind == "box" else (s[0],)
            # a body-frame box turns with the body: its axes are the body axes in the wind frame
            axes = None
            if kind == "box" and q.frame == "body" and (alpha_deg or yaw_deg):
                axes = np.column_stack([wd(e, "body") for e in np.eye(3)])
            out.heat_sources.append(HeatSource3D(kind=kind, center=tuple(pos / L), size=tuple(v / L for v in size),
                                                 power=float(q.power) / scales.heat_unit, name=q.name, axes=axes))
            out.heaters_si.append(q)
            if min(size) < 2 * h_m:
                out.warnings.append(f"Heat source “{q.name}” is {units.sig(1000 * min(size), 3)} mm across, under two "
                                    "grid cells: its heat is released over the cells it touches (the total is exact).")
    else:
        out.body_thermal = [("adiabatic", 0.0)] * len(bodies_wind or [])

    for f in _enabled(sim.fans):
        pos, d = wp(f.position, f.frame), wd(f.direction, f.frame)
        if not _inside(pos, extent3, tol):
            outside("Fan", f.name)
            continue
        if f.mode == "thrust":
            value = float(f.value) / (rho * U * U * L * L)
        elif f.mode == "pressure_jump":
            value = float(f.value) / (rho * U * U)
        else:
            value = float(f.value) / U
        out.fans.append(Fan3D(center=tuple(pos / L), direction=tuple(d), diameter=float(f.diameter) / L,
                              thickness=None if f.thickness is None else float(f.thickness) / L, mode=f.mode,
                              value=value, swirl=float(f.swirl), name=f.name))
        out.fans_si.append(f)
        if f.diameter / h_m < FAN_CELLS_MIN:
            out.warnings.append(f"Fan “{f.name}” spans {f.diameter / h_m:.1f} grid cells (about {FAN_CELLS_MIN:.0f} "
                                "resolve its disc): its thrust is delivered, but the induced velocity and the flow "
                                "rate are indicative only.")
    for p in _enabled(sim.ports):
        pos, d = wp(p.position, p.frame), wd(p.direction, p.frame)
        if not _inside(pos, extent3, tol):
            outside(p.kind.capitalize(), p.name)
            continue
        theta = None
        if p.kind == "jet" and p.temperature is not None:
            if th.enabled:
                theta = scales.theta(p.temperature)
            else:
                out.notes.append(f"Jet “{p.name}” has a temperature, but heat transfer is off: it blows ambient air.")
        out.ports.append(Port3D(kind=p.kind, center=tuple(pos / L), direction=tuple(d), diameter=float(p.size) / L,
                                speed=abs(float(p.speed)) / U, theta=theta,
                                dye_channel=jet_channel if (p.kind == "jet" and p.dye) else None, name=p.name))
        out.ports_si.append(p)
        if p.size / h_m < PORT_CELLS_MIN:
            out.warnings.append(f"{p.kind.capitalize()} “{p.name}” is {p.size / h_m:.1f} grid cells across (about "
                                f"{PORT_CELLS_MIN:.0f} resolve its profile): its volume flow is exact, its velocity "
                                "profile is not.")
    _balance(out, closed, dim=3)
    return out


class PlacedPhysicsPlan:
    """What a plan (2D ``SimPlan``, 3D ``SimPlan3D``) says about its placed physics.

    The dataclass fields ``domain_mode``, ``physics`` (:class:`PhysicsScales`), ``placed``
    (:class:`Placed`), ``monitors``, ``calibration_kind`` and ``step_cost`` live on the plans;
    these properties are the names the Setup page checks to see what a run carries
    (``app.pages.setup.planner_gaps``: ``plan.thermal``, ``plan.fans`` ...)."""

    placed: Placed | None
    domain_mode: str

    @property
    def thermal(self):
        """The solver's heat-transfer config, or None (no heat transfer)."""
        return self.placed.thermal if self.placed is not None else None

    @property
    def emitters(self) -> list:
        return list(self.placed.emitters) if self.placed is not None else []

    @property
    def fans(self) -> list:
        return list(self.placed.fans) if self.placed is not None else []

    @property
    def ports(self) -> list:
        return list(self.placed.ports) if self.placed is not None else []

    @property
    def heat_sources(self) -> list:
        return list(self.placed.heat_sources) if self.placed is not None else []

    @property
    def dye_channels(self) -> list[dict]:
        """Smoke channels (``viz.colormaps.dye_channels`` entries: field, colour, emitter names)."""
        return list(self.placed.dye_channels) if self.placed is not None else []

    @property
    def closed(self) -> bool:
        return self.domain_mode == "closed"


# ---------------------------------------------------------------------------
# time step, cost and convergence monitors of a plan
# ---------------------------------------------------------------------------

#: Extra cost per step of each feature, as a share of a plain step: the solvers' own measurements
#: (2D: theory_ns2d.md 20.3, cylinder 1024 x 512, shared GPU, so indicative; 3D: theory_ns3d.md 18,
#: idle GPU). They add up: the 2D "everything at once" run measured 2.37x against 2.30x from the
#: sum, the 3D one 1.64x against 1.68x (a product would give 3.0x and 1.77x).
PHYSICS_COST = {
    2: {"diagnostics": 0.018, "pid": 0.011, "thermal": 0.55, "buoyancy": 0.09, "smoke": 0.21, "particles": 0.19,
        "fans": 0.09, "ports": 0.14},
    3: {"diagnostics": 0.03, "pid": 0.02, "thermal": 0.30, "buoyancy": 0.0, "smoke": 0.06, "particles": 0.11,
        "fans": 0.03, "ports": 0.03},
}


def physics_cost(placed: Placed | None, diagnostics: bool, pid: bool, dim: int) -> float:
    """Time per step relative to the plain solver (>= 1): :data:`PHYSICS_COST` summed over the
    features in use (3D smoke counts per extra channel, as measured)."""
    c = PHYSICS_COST[dim]
    extra = (c["diagnostics"] if diagnostics else 0.0) + (c["pid"] if pid else 0.0)
    if placed is not None:
        if placed.thermal is not None:
            extra += c["thermal"] + (c["buoyancy"] if getattr(placed.thermal, "richardson", 0.0) else 0.0)
        n_smoke = len(placed.dye_channels)
        if n_smoke:
            extra += c["smoke"] * (max(n_smoke - 1, 0) if dim == 3 else 1)
        if any(e.species == "particles" for e in placed.emitters):
            extra += c["particles"]
        if placed.fans:
            extra += c["fans"]
        if placed.ports:
            extra += c["ports"]
    return 1.0 + extra


def _diffusion_coef2(scales: PhysicsScales, sim: Simulation, closed: bool) -> float:
    """Largest diagonal coefficient of the 2D solver's temperature diffusion (``ns2d.physics.
    diffusion_coefficient``): 4, 5 next to a fixed-temperature side, 6 in a corner of two."""
    modes = {w.side: w.mode for w in sim.thermal.walls}
    dirichlet = {s: modes.get(s) == "temperature" for s in _SIDES2}
    if not closed:
        dirichlet["left"], dirichlet["right"] = True, False           # inlet held at ambient, outlet open
    best = 5.0 if any(dirichlet.values()) else 4.0
    for a, b in (("left", "bottom"), ("left", "top"), ("right", "bottom"), ("right", "top")):
        best = max(best, 2.0 + 2.0 * dirichlet[a] + 2.0 * dirichlet[b])
    return best


def physics_dt(h: float, kfac: float, scales: PhysicsScales | None, placed: Placed | None, sim: Simulation, *,
               closed: bool, nut: float = 0.0, cfl: float = 0.5, dim: int = 2) -> float:
    """The time-step limits the placed physics adds (solver units), for the planner's estimate.

    The solvers' rules: jets and speed-controlled fans enter the advective CFL bound with
    their own speed; the temperature step is the positivity bound of the explicit
    advection-diffusion update (2D: ``1 / (2 S / h + c_d kappa / h^2)``,
    ``ns2d.physics.physics_dt_limit``; 3D: ``1 / (S / (h / 2) + 6 kappa / h^2)``,
    ``ns3d.physics.ThermalPhysics.dt_limit``) with S = (max|u| + max|v| (+ max|w|)) / U the
    speed factor and kappa = 1 / Pe (+ nu_t / Pr_t with LES); and a buoyant parcel at rest
    may move at most ``cfl`` cells in one step, ``dt <= cfl sqrt(h / (Ri |theta|max))`` with
    |theta|max = 1 (the set temperatures span one ΔT).
    """
    dt = math.inf
    if placed is None:
        return dt
    u_src = max([float(p.speed) for p in placed.ports]
                + [abs(float(f.value)) for f in placed.fans if f.mode == "velocity"], default=0.0)
    if u_src > 0:
        dt = min(dt, cfl * h / u_src)
    if scales is not None and scales.thermal and placed.thermal is not None:
        kap = 1.0 / scales.pe + nut / 0.85
        if dim == 2:
            rate = kfac * 2.0 / h + _diffusion_coef2(scales, sim, closed) * kap / (h * h)
        else:
            rate = kfac / (0.5 * h) + 6.0 * kap / (h * h)
        dt = min(dt, 1.0 / rate)
        if scales.ri > 0:
            dt = min(dt, cfl * math.sqrt(h / scales.ri))
    return dt


def _wall_items(placed: Placed) -> list[tuple[str, float]]:
    """(mode, value) of the solver's thermal walls (a list of specs or a {side: (mode, value)} map)."""
    walls = getattr(placed.thermal, "walls", None) or []
    if isinstance(walls, list):
        return [(w.mode, float(getattr(w, "value", 0.0) or 0.0)) for w in walls]
    return [(m, float(v or 0.0)) for m, v in dict(walls).values()]


def heat_input(placed: Placed | None) -> bool:
    """Whether a run puts heat in at a set rate (heaters, heat-flux bodies or walls heating the
    fluid): its fluid temperatures are then the result and can be monitored (a box held only at set temperatures,
    the cavity, is judged by its Nusselt number instead)."""
    if placed is None or placed.thermal is None:
        return False
    if any(float(getattr(q, "power", 0.0) or 0.0) > 0 for q in placed.heaters_si):
        return True
    if any(m == "heat_flux" and v > 0 for m, v in placed.body_thermal):
        return True
    return any(m == "heat_flux" and v > 0 for m, v in _wall_items(placed))


def heat_sink(placed: Placed | None) -> bool:
    """Whether heat can leave a closed box: a vent carries warm air out, and a body or wall held at a
    set temperature, a negative heat flux or a negative heater takes heat up. Without one, a
    heated box warms without end (P / (ρ c_p V) per second), so its mean temperature never
    settles and must not be a convergence monitor."""
    if placed is None:
        return False
    if any(getattr(p, "kind", "") == "vent" for p in placed.ports_si):
        return True
    if any(m == "temperature" for m, _ in placed.body_thermal):
        return True
    if any(m == "temperature" or (m == "heat_flux" and v < 0) for m, v in _wall_items(placed)):
        return True
    if any(m == "heat_flux" and v < 0 for m, v in placed.body_thermal):
        return True
    return any(float(getattr(q, "power", 0.0) or 0.0) < 0 for q in placed.heaters_si)


#: The convergence monitors of the fluid temperatures of a heated closed box: the rise of the
#: warmest fluid and of the mean fluid above the ambient [K] (:func:`.physics_run.chunk_samples`).
TEMPERATURE_MONITORS = ("t_rise_max", "t_rise_mean")


def plan_monitors(scales: PhysicsScales | None, placed: Placed | None, *, closed: bool) -> tuple[str, ...]:
    """Series the runner's convergence verdict judges besides the forces: the Nusselt number of a
    heat-transfer run with a heated body or a heated wall (the bodies' heat rate, or their wall
    temperature for heat-flux bodies, in one number), each fan's flow (its disc velocity, or the
    thrust a speed-controlled fan needs), and for a closed box its kinetic energy and, when heat
    goes in at a set rate and can leave, the rise of the warmest and of the mean fluid above the
    ambient (:data:`TEMPERATURE_MONITORS`).

    A closed box is never judged on force coefficients (the runner skips them): the Electronics
    enclosure ended "Not converged" because its circuit board's Cd, which means nothing inside
    a box and which the Summary rightly hides, was known to only 12 % (critic round 3.4)."""
    out: list[str] = []
    if placed is None:
        return ()
    thermal = scales is not None and scales.thermal and placed.thermal is not None
    if thermal:
        heated_body = any(m != "adiabatic" for m, _ in placed.body_thermal)
        modes = [m for m, _ in _wall_items(placed)]
        if heated_body or (closed and "temperature" in modes):
            out.append("nu")
    for i, f in enumerate(placed.fans):
        out.append(f"fan{i}_thrust" if f.mode == "velocity" else f"fan{i}_velocity")
    if closed:
        if thermal and heat_input(placed) and heat_sink(placed):
            out.extend(TEMPERATURE_MONITORS)
        out.append("ke")
    return tuple(out)


def unbounded_heating_note(scales: PhysicsScales | None, placed: Placed | None, sim: Simulation,
                           box: Sequence[float] | None, dim: int) -> str:
    """The note of a closed box that is heated at a set rate but has no way to lose the heat
    (:func:`heat_sink`): it warms by P / (ρ c_p V) per second for as long as it runs, so there is
    no steady state and the temperatures are not monitored ("" for any other run)."""
    if not (scales is not None and scales.thermal and placed is not None and heat_input(placed)) or heat_sink(placed):
        return ""
    p = scales.props
    total = sum(float(q.power) for q in _enabled(sim.heat_sources) if q.power)
    if box is None or len(box) != 2 * dim or p is None:
        return ""
    vol = math.prod(float(box[2 * i + 1] - box[2 * i]) for i in range(dim))
    if not (vol > 0 and total > 0):
        return ("Nothing in this box takes heat out (no vent, no wall or body at a set temperature), so it warms "
                "for as long as it runs and never settles; its temperatures are not used to judge convergence.")
    rate = total / (p.density * p.cp * vol)
    return (f"Nothing in this box takes heat out (no vent, no wall or body at a set temperature): its "
            f"{units.sig(total, 3)} W{'/m' if dim == 2 else ''} warm the air by about {units.sig(rate * 60, 2)} K per "
            "minute for as long as it runs, so it never settles; its temperatures are not used to judge "
            "convergence. Add a vent or a cooled wall for a steady state.")


# ---------------------------------------------------------------------------
# checks and summary rows shared by both planners
# ---------------------------------------------------------------------------

def fluid_ambient_warning(sim: Simulation) -> str:
    """The warning for a preset fluid made at another temperature than the ambient ("" when they
    agree, heat transfer is off or the fluid is custom). Worded exactly as the Setup page words
    it, so the page shows the planner's line and its own once."""
    th = getattr(sim, "thermal", None)
    if th is None or not th.enabled:
        return ""
    t_fluid = fluid_temperature(sim.conditions.fluid)
    if t_fluid is None or abs(t_fluid - float(th.t_ambient)) <= FLUID_AMBIENT_TOL_K:
        return ""
    what = str(sim.conditions.fluid.name).split(",")[0].lower()
    return (f"The fluid is {what} at {t_fluid:.3g} °C but the ambient temperature is {float(th.t_ambient):.3g} °C: "
            "set Flow ▸ Fluid to that temperature, so the density and viscosity of the run match it.")


#: Warnings that state one quantity, by the words they start with: a list that holds two of
#: one key (the Setup page's own line and the planner's, or two derivations of ΔT_ref) states
#: two values for one number. :func:`warning_key` and :func:`merge_warnings` keep one per key.
_WARNING_KEYS = (
    ("boussinesq", re.compile(r"^β·ΔT = ")),
    ("fluid-ambient", re.compile(r"^The fluid is .* but the ambient temperature is ")),
    ("rayleigh", re.compile(r"^Ra = ")),
    ("custom-fluid", re.compile(r"^Custom fluid")),
    ("nothing-heated", re.compile(r"^Heat transfer is on but nothing is heated")),
    ("heat-off", re.compile(r"^Heat transfer is off, so ")),
    ("thermal-resolution", re.compile(r"^The grid gives [\d.]+ cells across the thermal")),
    ("buoyancy-dominated", re.compile(r"^Buoyancy dominates this run")),
    ("blockage", re.compile(r"^Blockage [\d.]+ % is above")),
    ("nothing-drives", re.compile(r"^Nothing drives the flow")),
    ("room-2d-qualitative", re.compile(r"^In a 2D closed room the fan")),
    ("regime-not-validated", re.compile(r"^(?:Natural|Mixed) convection around a heated body")),
)


def warning_key(text: str) -> str:
    """The quantity a warning states ("boussinesq", "fluid-ambient", "rayleigh", ...), or the
    text itself for a warning that has no key (it is then only equal to itself)."""
    s = str(text).strip()
    for key, pat in _WARNING_KEYS:
        if pat.match(s):
            return key
    return s


def merge_warnings(primary: Sequence[str], *others: Sequence[str]) -> list[str]:
    """``primary`` (de-duplicated by :func:`warning_key`, first one wins), then each warning of
    ``others`` whose key is not there yet. Give the plan's list first: its numbers come from the
    one ΔT_ref (:func:`physics_scales`), so a page's own estimate never contradicts them.

    A warning that changes how every number of the result reads (:data:`_LEADING_KEYS`: the
    coefficients of a buoyancy-dominated run) goes first, because a results banner shows the
    first warning and counts the rest."""
    out: list[str] = []
    seen: set[str] = set()
    for group in (primary, *others):
        for w in group or ():
            k = warning_key(w)
            if k not in seen:
                seen.add(k)
                out.append(w)
    return sorted(out, key=lambda w: warning_key(w) not in _LEADING_KEYS)       # stable: the rest keep order


#: Warning keys :func:`merge_warnings` puts first.
_LEADING_KEYS = frozenset({"buoyancy-dominated", "room-2d-qualitative", "regime-not-validated"})


#: The warning of every 2D closed domain with a fan (planner and result), worded as the 2D solver's
#: grid study states it (theory_ns2d.md §18.4: the 1 N/m example room spun up to 1200 L/U on 6, 12
#: and 24 cells per fan width). The thrust is imposed exactly, but the room's speeds have no
#: grid-converged answer, so no 2D room speed may be quoted as a result. The conclusion leads (p2d's
#: proposed sentences, reordered): the Setup planner list and the Results banner show the first
#: sentence or the first ~130 characters, and that is where "qualitative" has to be.
ROOM_2D_FAN_WARNING = (
    "In a 2D closed room the fan's speeds are qualitative and grid-dependent: speeds, the disc velocity, volume "
    "flow and convective heat transfer have no grid-converged answer (for a 1 N/m fan in a 4 × 3 m room the mean "
    "speed is 2.4 m/s at 6 and 12 cells per fan width and 3.4 m/s at 24, and the room needs 500–900 L/U to settle; "
    "theory_ns2d.md §18.4). The thrust is exact and the flow pattern is meaningful. The fan's power can only leave "
    "through wall friction and numerical dissipation (2D flow has no vortex stretching), so the room spins up into "
    "room-sized vortices. Use 3D for rooms.")
#: The one-sentence form for notes, docs, the API's fan-room note and the assistant's validity table.
ROOM_2D_FAN_NOTE = ("In 2D the room's speeds are qualitative and grid-dependent (no vortex stretching; "
                    "theory_ns2d.md §18.4); use dimension=3 for a room.")
#: L/U the example 2D room needed to leave its start-up transient (theory_ns2d.md §18.4: 500–900).
ROOM_2D_SETTLE_LU = 500.0


def room_2d_fan_warning(t_end: float | None = None) -> str:
    """:data:`ROOM_2D_FAN_WARNING`, plus a sentence when the run's planned duration ``t_end`` [L/U]
    ends before the rooms of the grid study had settled (the Draft room's 240 L/U do)."""
    w = ROOM_2D_FAN_WARNING
    if t_end is not None and math.isfinite(float(t_end)) and float(t_end) < ROOM_2D_SETTLE_LU:
        w += (f" This run's {float(t_end):.4g} L/U end inside that start-up transient, so its averages are one sample "
              "of it.")
    return w


def buoyancy_dominated_warning(scales: PhysicsScales, *, closed: bool) -> str:
    """The warning for a tunnel run whose buoyancy outruns its free stream (Ri > :data:`RI_BUOYANT`),
    "" otherwise. Its force coefficients are normalised by the free stream's q = ½ρU², which is
    then tiny: the Draft hot sphere at Re 100 with ΔT 40 K (Ri 627) read CL 120 and CD 5.8 on
    q = 0.00013 Pa. A closed box needs none: it already takes the buoyant velocity as U."""
    s = scales
    if closed or not (s.thermal and s.buoyancy) or s.ri_flow is None or not s.ri_flow > RI_BUOYANT:
        return ""
    ub, U = buoyant_speed(s), float(s.speed)
    rho = float(s.props.density) if s.props is not None else float("nan")
    q = 0.5 * rho * U * U
    return (f"Buoyancy dominates this run (Ri = Gr/Re² = {fmt_number(s.ri_flow)}): the plume moves at about "
            f"{units.sig(ub, 2)} m/s, {ub / U:.0f} times the {units.sig(U, 2)} m/s free stream, while the force "
            f"coefficients use the free stream's q = ½ρU² = {units.sig(q, 2)} Pa. They are then mostly the buoyant "
            "force over a small q (|C| ≫ 1), not aerodynamic coefficients: compare the forces in newtons, or the "
            "coefficients on the buoyant scale ½ρu_b² that the result reports. The time step is sized for the "
            "plume, so the run takes long.")


def buoyant_coefficients_note(result, plan, scales: PhysicsScales | None = None) -> str:
    """The result's force coefficients on the buoyant dynamic pressure ½ρu_b² (C·U²/u_b²), for a
    buoyancy-dominated tunnel run (:func:`buoyancy_dominated_warning`); "" otherwise. ``scales``
    overrides the plan's (the measured ones of :func:`result_scales`, whose ΔT sets u_b)."""
    s = scales if scales is not None else getattr(plan, "physics", None)
    if s is None or getattr(plan, "domain_mode", "tunnel") == "closed" or not buoyancy_dominated_warning(s, closed=False):
        return ""
    ub, U = buoyant_speed(s), float(s.speed)
    if not ub > 0:
        return ""
    f = (U / ub) ** 2
    parts = []
    for name, key in (("Cd", "cd"), ("Cl", "cl"), ("Cs", "cs")):
        st = getattr(result, key, None)
        mean = getattr(st, "mean", float("nan")) if st is not None else float("nan")
        if math.isfinite(mean) and not (key == "cs" and getattr(plan, "symmetry_plane", False)):
            parts.append(f"{name} {units.sig(mean * f, 3)}")
    if not parts:
        return ""
    rho = float(s.props.density) if s.props is not None else float("nan")
    return (f"On the buoyant scale q_b = ½ρu_b² = {units.sig(0.5 * rho * ub * ub, 3)} Pa (u_b = "
            f"{units.sig(ub, 3)} m/s, the buoyant velocity): " + ", ".join(parts) + f" (the coefficients above × "
            f"U²/u_b² = {units.sig(f, 3)}).")


def boussinesq_warning(scales: PhysicsScales) -> str:
    """The β·ΔT warning, from the one ΔT_ref of :func:`physics_scales` (the number the plan's notes
    and the Estimate's "Temperature scale" row quote)."""
    return (f"β·ΔT = {scales.beta_dt:.2f} (ΔT {units.sig(scales.delta_t, 3)} K): above about {BOUSSINESQ_LIMIT:g} "
            "the Boussinesq approximation (constant density except in the buoyancy force) loses accuracy.")


def thermal_checks(scales: PhysicsScales | None, sim: Simulation, *, h_m: float, n_bl: float, bl_target: float,
                   closed: bool, live: bool, dim: int) -> tuple[list[str], list[str]]:
    """(notes, warnings) on the heat transfer of a plan: properties, Boussinesq, regime, and whether
    the grid resolves the thermal boundary layer (``n_bl``: cells across the laminar momentum
    boundary layer at the simulated Re; delta_T ~ delta Pr^(-1/3) for Pr > 1, Pr^(-1/2) below)."""
    notes: list[str] = []
    warns: list[str] = []
    if scales is None or not scales.thermal:
        heat = [h.name for h in _enabled(sim.heat_sources) if h.power]
        heat += [p.name for p in _enabled(sim.ports) if p.kind == "jet" and p.temperature is not None]
        if heat:
            warns.append(f"Heat transfer is off, so {', '.join(f'“{n}”' for n in heat[:3])} add no heat: switch on "
                         "Heat transfer to solve for the temperature.")
        return notes, warns
    s, p = scales, scales.props
    if not p.known:
        warns.append("Custom fluid: its thermal properties are unknown, so air's are used where none were typed "
                     "(Prandtl number, conductivity, specific heat, expansion coefficient). Type them under Heat "
                     "transfer for this fluid's values.")
    else:
        notes.append(f"Heat transfer: {p.basis}: Pr {units.sig(p.prandtl, 3)}, k {units.sig(p.conductivity, 3)} "
                     f"W/(m·K), cp {units.sig(p.cp, 4)} J/(kg·K), β {units.sig(p.beta, 3)} 1/K"
                     + (f" ({', '.join(p.typed)} as typed)" if p.typed else "") + ".")
    notes.append(f"Temperature scale ΔT = {units.sig(s.delta_t, 3)} K ({s.delta_t_basis}); the solver carries "
                 f"θ = (T − {units.sig(s.t_ambient, 4)} °C) / ΔT.")
    if s.delta_t_basis.startswith("1 K"):
        warns.append("Heat transfer is on but nothing is heated or cooled: every body, wall and jet is at the ambient "
                     "temperature and there is no heat source, so the temperature stays uniform.")
    mismatch = fluid_ambient_warning(sim)
    if mismatch:
        warns.append(mismatch)
    if s.buoyancy and s.beta_dt > BOUSSINESQ_LIMIT:
        warns.append(boussinesq_warning(s))
    elif s.buoyancy and math.isfinite(s.beta_dt):
        # said as a note, so the Setup page's own estimate (an older q L / (k Nu) one) does not add a
        # contradicting warning next to it: the page drops a line on a topic the planner covers
        notes.append(f"β·ΔT = {s.beta_dt:.2g} (ΔT {units.sig(s.delta_t, 3)} K): inside the Boussinesq range "
                     f"(below about {BOUSSINESQ_LIMIT:g}).")
    if s.buoyancy:
        g = "−y" if dim == 2 else "−z"
        notes.append(buoyancy_note(s, g))
        if s.ra > 1e8:
            warns.append(f"Ra = {units.fmt_reynolds(s.ra)}: above about 10⁸ the thermal boundary layers become unsteady "
                         "and then turbulent; this laminar / LES grid resolves the mean flow at best, so treat the heat "
                         "transfer as indicative.")
        bd = buoyancy_dominated_warning(s, closed=closed)
        if bd:
            warns.append(bd)
    else:
        notes.append("Buoyancy off: the temperature is carried by the flow but does not move it (forced convection).")
    pr = s.pr
    n_t = n_bl * (pr ** (-1.0 / 3.0) if pr >= 1.0 else pr ** -0.5)
    buoyant_bl = buoyant_layer(s)
    if closed and s.natural and s.ra >= RA_CONDUCTION:
        nu_est = cavity_nusselt(s.ra, pr)
        n_t = s.length_ra / (2.0 * nu_est) / h_m
        what = (f"the thermal boundary layer of the heated walls (about H / (2 Nu) = "
                f"{units.sig(1000 * s.length_ra / (2 * nu_est), 3)} mm with Nu ≈ {nu_est:.2g} by the cavity "
                "correlation)")
    elif buoyant_bl is not None:
        # buoyancy outruns the stream (Ri > 1): the layer is the plume's, far thinner than the free
        # stream's Re suggests (the Draft hot sphere at Re 100, Ri 590: about 2 mm, not 12 cells of 5 mm)
        delta, how = buoyant_bl
        n_t = delta / h_m
        what = f"the thermal boundary layers (about {units.sig(1000 * delta, 2)} mm, {how})"
    else:
        what = "the thermal boundary layers"
    if not live and s.surfaces == 0:
        # a room warmed by heaters or hot jets: no wall or body exchanges heat, so there is no thermal
        # boundary layer whose resolution would bias a heat rate (the heaters' power is imposed)
        notes.append("Thermal resolution: no body or wall exchanges heat with the fluid (the heat comes from heaters "
                     "or jets), so there is no thermal boundary layer to resolve and no Nusselt number; the heat "
                     "input is imposed.")
    elif not live and n_t < bl_target * 0.9:
        warns.append(f"The grid gives {n_t:.1f} cells across {what}; about {bl_target:.0f} resolve it. Heat rates and "
                     "Nusselt numbers are then under-resolved and can be far off in either direction (the Draft hot "
                     "sphere read 2.4 times the natural-convection correlation): choose a finer quality.")
    elif not live:
        notes.append(f"Thermal resolution: about {n_t:.0f} cells across {what}.")
    nv = regime_not_validated_warning(s)
    if nv:
        warns.append(nv)
    return notes, warns


#: Richardson number Gr / Re² above which the thermal boundary layer is sized on the buoyant
#: velocity instead of the stream (:func:`buoyant_layer`): the plume is then the faster flow.
RI_BUOYANT_LAYER = 1.0


def buoyant_layer(s: PhysicsScales) -> tuple[float, str] | None:
    """(thickness [m], how) of the thermal boundary layer of a run whose buoyancy outruns its
    forced flow (Ri = Gr/Re² > :data:`RI_BUOYANT_LAYER`), else None.

    The planner's usual rule sizes the layer on the stream's Reynolds number, which in a
    buoyancy-dominated run is the slow flow: the Draft hot sphere (D 0.1 m, Re 100, Ri 590,
    Ra 4.2×10⁶) was told "about 12 cells across the thermal boundary layers" while its layer is
    about D·Ra^(-1/4) ≈ 2 mm, under half of a 5 mm cell (critic round 3.4). Here, on the length of
    Ra: natural convection (Ri > :data:`RI_BUOYANT`) δ ≈ L·Ra^(-1/4) (Bejan, Convection Heat
    Transfer, ch. 4 scaling), mixed convection δ ≈ L·(Re_b·Pr)^(-1/2) with Re_b on the buoyant
    velocity (:func:`buoyant_speed`)."""
    if not (s.thermal and s.buoyancy and s.ri_flow is not None and s.ri_flow > RI_BUOYANT_LAYER
            and s.gr > 0 and s.ra > 0):
        return None
    lra = s.length_ra if math.isfinite(s.length_ra) else s.length
    if s.ri_flow > RI_BUOYANT:
        return lra * s.ra ** -0.25, "L·Ra^(-1/4), natural convection"
    re_b = buoyant_speed(s) * lra / s.nu
    if not (re_b > 0 and math.isfinite(s.pr) and s.pr > 0):
        return None
    return lra / math.sqrt(re_b * s.pr), "L·(Re_b·Pr)^(-1/2) on the buoyant velocity, mixed convection"


def regime_not_validated_warning(s: PhysicsScales | None) -> str:
    """The warning of a run with natural or mixed convection around a heated or cooled body, ""
    otherwise: docs/validation.md §10 lists both as not validated (the validated heat-transfer
    cases are cavities heated at their walls and forced convection past a cylinder and a sphere).
    The Draft hot sphere at Ri 590 read Nu 53.5 where Churchill's (1983) natural-convection
    correlation gives 22.6 and nothing on its result said the regime was unvalidated."""
    if s is None or not (s.thermal and s.buoyancy) or s.regime not in ("Natural", "Mixed") or s.bodies <= 0:
        return ""
    what = "Natural convection" if s.regime == "Natural" else "Mixed convection"
    return (f"{what} around a heated body: regime not validated (docs/validation.md §10). The validated heat "
            "transfer is laminar natural convection in cavities heated at their walls and forced convection "
            "past a cylinder and a sphere, so treat this run's Nusselt number and heat rates as indicative.")


def buoyancy_note(s: PhysicsScales, g: str) -> str:
    """The plan's buoyancy note, naming the length each group is on: the solver's coefficient
    Ri = gβΔT L/U² uses the reference length L; Gr, Ra and the Estimate's Richardson number
    Gr/Re² use H (the box height in a closed box). One room once read "Richardson number 5.34" in
    its table and "Ri = 0.889" in its notes without saying the two lengths were 3 m and 0.5 m."""
    txt = (f"Buoyancy: Boussinesq, gravity along {g}. Ri = gβΔT L/U² = {units.sig(s.ri, 3)} on the reference length "
           f"L = {units.sig(s.length, 3)} m (the solver's buoyancy coefficient)")
    same_len = abs(float(s.length_ra) - float(s.length)) <= 1e-9 * max(abs(float(s.length)), 1e-30)
    if s.ri_flow is not None and not (same_len and abs(s.ri_flow - s.ri) <= 1e-6 * max(abs(s.ri), 1e-30)):
        txt += (f"; Ri = Gr/Re² = {fmt_number(s.ri_flow)} on the {s.length_ra_basis} {units.sig(s.length_ra, 3)} m "
                "(the Richardson number of the Estimate, which sets the regime)")
    return (txt + f". Gr {units.fmt_reynolds(s.gr)} and Ra {units.fmt_reynolds(s.ra)} on {units.sig(s.length_ra, 3)} "
            f"m, the {s.length_ra_basis}.")


#: Draft runs of a buoyancy-dominated tunnel case (Ri > :data:`RI_BUOYANT`) take at most this
#: share of the steps the same case would take without its plume (:func:`draft_buoyant_duration`),
DRAFT_BUOYANT_STEP_SHARE = 1.0
#: ... but never less than this many L/U (the plume must have time to rise clear of the body),
DRAFT_BUOYANT_T_MIN = 2.0
#: ... nor more than this many (the cap the critique proposed, critic round 3.3).
DRAFT_BUOYANT_T_MAX = 10.0


def draft_buoyant_duration(scales: PhysicsScales | None, t_end: float, u_peak: float) -> tuple[float, str]:
    """(t_end [L/U], note) for a Draft tunnel run whose plume outruns its free stream, else
    (``t_end``, "").

    The time step of such a run is sized for the plume (:func:`peak_speed`), so the free stream's
    duration costs u_peak/U times the steps: the Draft hot sphere (Ri 627, plume 24 times the
    0.015 m/s stream) was planned at 40 L/U, 189,682 steps and 40 minutes, for a quality meant to
    give pictures (critic round 3.3). Draft keeps the step count of the same case without the
    plume (t_end U/u_peak), within [:data:`DRAFT_BUOYANT_T_MIN`, :data:`DRAFT_BUOYANT_T_MAX`]
    L/U; in that time the plume rises t_end u_peak/U reference lengths, far enough to show it,
    not to average converged forces. The note says so, and how to get the full length."""
    if scales is None or not buoyancy_dominated_warning(scales, closed=False):
        return t_end, ""
    U = float(scales.speed)
    if not (U > 0 and u_peak > U and math.isfinite(t_end)):
        return t_end, ""
    cap = min(max(DRAFT_BUOYANT_STEP_SHARE * t_end * U / u_peak, DRAFT_BUOYANT_T_MIN), DRAFT_BUOYANT_T_MAX)
    cap = float(f"{cap:.2g}")
    if cap >= t_end:
        return t_end, ""
    ratio = u_peak / U
    return cap, (f"Draft runs this buoyancy-dominated case for {cap:g} L/U instead of {t_end:g}: its time step "
                 f"follows the plume, {ratio:.0f} times faster than the free stream, so {t_end:g} L/U would take "
                 f"{ratio:.0f} times the steps of the same case without heat. In {cap:g} L/U the plume travels about "
                 f"{cap * ratio:.0f} reference lengths, so it rises clear of the body and spreads: enough for "
                 "pictures, not for converged forces (the free stream has not yet flushed the warm air). Standard "
                 "quality, or a Duration set in Setup, runs the full length; a closed box (Domain ▸ Type: Closed box) "
                 "suits a heated body in nearly still air, since its time scale is the plume's own.")


def physics_rows(scales: PhysicsScales | None, placed: Placed | None, *, closed: bool) -> dict[str, str]:
    """Extra ``summary()`` rows of a plan with placed physics (empty without)."""
    rows: dict[str, str] = {}
    if closed:
        rows["Domain type"] = "Closed box (walls on every side)"
    if scales is not None and (closed or scales.thermal or (placed is not None and placed.any)):
        rows.update(scales.summary_rows())
        if not closed and not scales.thermal:
            rows.pop("Reference velocity", None)
    if placed is not None:
        n_smoke = sum(1 for e in placed.emitters if e.species == "dye")
        n_part = sum(1 for e in placed.emitters if e.species == "particles")
        if n_smoke or n_part:
            parts = []
            if n_smoke:
                parts.append(f"{n_smoke} smoke ({len(placed.dye_channels)} colour{'s' if len(placed.dye_channels) != 1 else ''})")
            if n_part:
                parts.append(f"{n_part} particle")
            rows["Emitters"] = " · ".join(parts)
        if placed.fans:
            rows["Fans"] = ", ".join(f"{f.name} ({f.mode.replace('_', ' ')} {units.sig(f.value, 3)} "
                                     f"{_fan_unit(f.mode, placed)})" for f in placed.fans_si[:3]) \
                + (f" +{len(placed.fans) - 3}" if len(placed.fans) > 3 else "")
        if placed.ports:
            jets = sum(1 for p in placed.ports_si if p.kind == "jet")
            vents = len(placed.ports_si) - jets
            rows["Jets and vents"] = f"{jets} jet{'s' if jets != 1 else ''}, {vents} vent{'s' if vents != 1 else ''}"
        if placed.heat_sources:
            tot = sum(float(q.power) for q in placed.heaters_si)
            rows["Heat sources"] = f"{len(placed.heat_sources)} ({units.sig(tot, 3)} W{'/m' if _is2(placed) else ''} in all)"
    return rows


def _is2(placed: Placed) -> bool:
    objs = placed.fans or placed.heat_sources or placed.ports or placed.emitters
    return bool(objs) and type(objs[0]).__module__.endswith("ns2d.config")


def _fan_unit(mode: str, placed: Placed) -> str:
    two = _is2(placed)
    return {"thrust": "N/m" if two else "N", "pressure_jump": "Pa", "velocity": "m/s"}[mode]
