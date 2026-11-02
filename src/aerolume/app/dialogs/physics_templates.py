"""Starter cases for the placed physics (DESIGN.md section 10): heat, fans, jets and vents.

Each template builds a :class:`~aerolume.model.Design` and its first
:class:`~aerolume.model.Simulation` (or two) with every number stated, so the
case runs as soon as it is created and its result can be checked against the
reference it names:

* **Electronics enclosure** (2D): a closed box with a circuit board, a chip that
  releases heat, an intake jet with a fan behind it and an exhaust vent sized to
  balance the intake.
* **Closed room** (3D): a 4 × 3 × 2.5 m room with a 200 W radiator under a cold window
  wall, a ceiling fan, a supply jet and an extract vent, and a cabinet.
* **Heated cylinder** (2D): forced convection at Re = 40, Pr = 0.71, buoyancy off -
  the classic case for the mean Nusselt number. Published simulations give
  3.200-3.255 (:data:`CYLINDER_NU_PUBLISHED`); the Lange, Durst & Breuer (1998)
  correlation gives 3.28, 0.8-2.5 % above them - the correlation's own scatter, so
  a single reference number would overstate how precisely Nu is known.
* **Natural-convection cavity** (2D, validation): the differentially heated square
  cavity of de Vahl Davis (1983), sized for the Rayleigh number asked for in air
  with Pr = 0.71. The solver measures the mean Nusselt number of the hot wall, so
  the reference is the paper's Nu₀ (1.117, 2.238, 4.509 and 8.817 at Ra = 10³,
  10⁴, 10⁵ and 10⁶; :data:`CAVITY_NU_HOT_WALL`), the same values the API, the
  validation suite and docs/validation.md compare with. The paper's cavity-average
  Nū (1.118, 2.243, 4.519, 8.800) is a different quantity, 0.1-0.4 % away.

:func:`build_template` returns the objects (no Qt, no project); :func:`add_template`
adds them to the open project through :class:`~aerolume.app.state.AppState`, as
every other way of creating a study does, so the Project tree, the undo-free
model and the saved project all see the same thing. The Setup page offers the
templates in its simulation menu; the Start page can list them as study types
(:data:`PHYSICS_TEMPLATES` carries the title, icon and description it needs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...core import fluid as fluids
from ...model import (
    Body,
    Design,
    Emitter,
    Fan,
    FlowConditions,
    HeatSource,
    Port,
    Simulation,
    ThermalSettings,
    WallThermal,
)

#: The reference of the differentially heated square cavity, Pr = 0.71, by Rayleigh number: Nu₀, the
#: mean Nusselt number of the hot wall (de Vahl Davis, G., 1983, "Natural convection of air in a square
#: cavity: a bench mark numerical solution", Int. J. Numer. Methods Fluids 3, 249-264, Table V) - the
#: quantity a cavity run measures (``Result.nusselt`` is the hot wall's). Equal to
#: ``aerolume.api.CAVITY_NU_2D`` and validation/thermal2d_cases.py ``nu_0`` (a test keeps them equal).
CAVITY_NU_HOT_WALL = {1e3: 1.117, 1e4: 2.238, 1e5: 4.509, 1e6: 8.817}
#: The same paper's Nū, the Nusselt number averaged over the whole cavity: a different quantity, kept
#: only to name it apart where both appear (a run is never compared with it).
CAVITY_NU_AVERAGE = {1e3: 1.118, 1e4: 2.243, 1e5: 4.519, 1e6: 8.800}
#: Published mean Nusselt numbers of an isothermal cylinder at Re 40, Pr 0.71 (as tabulated by Singhal,
#: Dutta & Kalita 2023, Int. J. Thermal Sciences 194, 108588, Table 2; the same values as
#: validation/thermal2d_cases.py ``CYLINDER_NU_REF[40]``): their spread is the range a run is judged by.
CYLINDER_NU_PUBLISHED = {"Jafroudi & Yang 1986": 3.2, "Biswas & Sarkar 2009": 3.2351,
                         "Apelt & Ledwich 1979": 3.255}
G0 = 9.80665


@dataclass(frozen=True)
class PhysicsTemplate:
    """One starter case: key, menu title, icon name, one-line description, dimension."""
    key: str
    title: str
    icon: str
    description: str
    dimension: int


PHYSICS_TEMPLATES: tuple[PhysicsTemplate, ...] = (
    PhysicsTemplate("enclosure", "Electronics enclosure (2D)", "chip",
                    "A closed box: a circuit board with a 40 W/m chip, an intake jet with a fan behind it and an "
                    "exhaust vent that balances it. Heat transfer with buoyancy.", 2),
    PhysicsTemplate("room", "Closed room (3D)", "cube",
                    "A 4 × 3 × 2.5 m room: a 200 W radiator under a cold window wall, a ceiling fan, a supply jet and "
                    "an extract vent. Heat transfer with buoyancy.", 3),
    PhysicsTemplate("heated_cylinder", "Heated cylinder (2D)", "circle",
                    "Forced convection from a cylinder 10 K above the air at Re = 40, buoyancy off: published "
                    "mean Nusselt numbers 3.200–3.255; the Lange et al. (1998) correlation gives 3.28.", 2),
    PhysicsTemplate("cavity", "Natural-convection cavity (2D, validation)", "grid",
                    "The differentially heated square cavity of de Vahl Davis (1983) at Ra = 10⁵, Pr = 0.71: "
                    "hot-wall mean Nusselt number Nu₀ 4.509.", 2),
)
_BY_KEY = {t.key: t for t in PHYSICS_TEMPLATES}


def template_info(key: str) -> PhysicsTemplate:
    """The :class:`PhysicsTemplate` of ``key``; KeyError names the valid keys."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown physics template {key!r}; available: {', '.join(_BY_KEY)}") from None


def _rect(x0: float, x1: float, y0: float, y1: float, n_per_side: int = 12) -> np.ndarray:
    """A counter-clockwise rectangle outline, each side resampled (the solver's SDF likes points)."""
    t = np.linspace(0.0, 1.0, n_per_side, endpoint=False)
    sides = [np.c_[x0 + (x1 - x0) * t, np.full_like(t, y0)], np.c_[np.full_like(t, x1), y0 + (y1 - y0) * t],
             np.c_[x1 - (x1 - x0) * t, np.full_like(t, y1)], np.c_[np.full_like(t, x0), y1 - (y1 - y0) * t]]
    return np.vstack(sides)


def _cfd(design: Design, cond: FlowConditions, device: str | None, precision: str | None):
    from ...simulation.defaults import default_cfd_settings
    kw = {}
    if device:
        kw["device"] = device
    if precision:
        kw["precision"] = precision
    try:
        return default_cfd_settings(design, cond, **kw)
    except TypeError:                     # a defaults module without keyword-only device / precision
        return default_cfd_settings(design, cond)


def lange_nusselt(re: float) -> float:
    """Mean Nusselt number of a cylinder in laminar cross flow, Pr 0.7, by the Lange, Durst & Breuer (1998)
    correlation: Nu = 0.082 Re^0.5 + 0.734 Re^x, x = 0.05 + 0.226 Re^0.085 (10⁻⁴ ≤ Re ≤ 200).

    The same formula as ``aerolume.api.lange_nusselt`` (a test keeps them equal), repeated so a menu of
    starter cases does not import the whole scripting API."""
    x = 0.05 + 0.226 * re ** 0.085
    return 0.082 * math.sqrt(re) + 0.734 * re ** x


def cavity_side(ra: float, fluid, *, t_mean_c: float, delta_t: float, prandtl: float = 0.71,
                g: float = G0) -> float:
    """Side L [m] of a square cavity of Rayleigh number ``ra``: Ra = g β ΔT L³ / (ν α), with the
    ideal-gas β = 1/T_mean, ν of ``fluid`` and α = ν / Pr."""
    nu = float(fluid.viscosity) / float(fluid.density)
    beta = 1.0 / (float(t_mean_c) + 273.15)
    alpha = nu / float(prandtl)
    return (float(ra) * nu * alpha / (g * beta * float(delta_t))) ** (1.0 / 3.0)


def _enclosure(device, precision) -> tuple[Design, list[Simulation]]:
    w, h = 0.30, 0.12
    board = Body("Circuit board", _rect(0.05, 0.25, 0.028, 0.032, 16))
    d = Design(name="Electronics enclosure", kind="generic", bodies=[board], length_ref=w, force_ref=h,
               moment_ref=(0.5 * w, 0.5 * h),
               source={"type": "physics_template", "template": "enclosure", "physics": True},
               notes="Closed 0.30 × 0.12 m box (a cross-section; heat per metre of depth).")
    air = fluids.air(25.0)
    cond = FlowConditions(fluid=air, speed=1.5, alpha_deg=0.0)
    sim = Simulation(name="Enclosure · heat and fan", design_id=d.id, analysis="cfd", conditions=cond,
                     cfd=_cfd(d, cond, device, precision))
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (0.0, w, 0.0, h)
    sim.thermal = ThermalSettings(enabled=True, t_ambient=25.0, buoyancy=True)
    sim.heat_sources = [HeatSource(name="Processor", kind="rect", position=(0.15, 0.036), size=(0.030, 0.008),
                                   power=40.0, frame="body")]
    sim.ports = [Port(name="Intake", kind="jet", position=(0.004, 0.08), direction=(1.0, 0.0), size=0.03,
                      speed=1.5, dye=True, frame="wind"),
                 Port(name="Exhaust", kind="vent", position=(0.296, 0.06), direction=(1.0, 0.0), size=0.03,
                      speed=1.5, frame="wind")]
    sim.fans = [Fan(name="Fan", position=(0.03, 0.08), direction=(1.0, 0.0), diameter=0.04, mode="velocity",
                    value=2.0, frame="wind")]
    sim.notes = ("Intake and exhaust both carry 1.5 m/s × 0.03 m = 0.045 m²/s per metre of depth, so the box "
                 "balances. 40 W/m ≈ 6 W in a 15 cm deep box.")
    return d, [sim]


#: where the room's cabinet stands (centre, m): on the floor (z = 0.5 for a 1 m tall cabinet), near the far
#: wall, clear of the radiator under the window
ROOM_CABINET_CENTRE = (3.2, 1.0, 0.5)


def _room(device, precision) -> tuple[Design, list[Simulation]]:
    from ...geometry3d import box as box_mesh
    from ...geometry3d import design_from_meshes, symmetry_error, transform
    lx, ly, lz = 4.0, 3.0, 2.5
    cabinet = box_mesh(0.8, 0.5, 1.0, center=ROOM_CABINET_CENTRE, name="Cabinet")
    d = design_from_meshes([cabinet], name="Closed room", kind="generic", on_road=False,
                           source={"type": "physics_template", "template": "room", "physics": True})
    # design_from_meshes centres a generic shape on the origin (a tunnel body's frame); in a room the
    # coordinates are the room's, so the cabinet goes back to where it stands. Left centred, it straddled
    # the window wall at x = 0 and sank 0.5 m under the floor, through the radiator.
    v = np.vstack([m.vertices for m in d.meshes])
    shift = np.asarray(ROOM_CABINET_CENTRE) - 0.5 * (v.min(axis=0) + v.max(axis=0))
    d.meshes = transform(d.meshes, translate=tuple(float(s) for s in shift))
    d.moment_ref3 = tuple(float(c) for c in ROOM_CABINET_CENTRE)
    d.source["symmetry_error"] = float(symmetry_error(d.meshes, plane_y=0.0))
    d.symmetric = False                                  # off the room's centre plane y = 0
    d.notes = "A 4 × 3 × 2.5 m room; the cabinet stands on the floor at the far wall."
    air = fluids.air(20.0)
    cond = FlowConditions(fluid=air, speed=1.0, alpha_deg=0.0)
    sim = Simulation(name="Room · radiator and fan", design_id=d.id, analysis="cfd", conditions=cond,
                     cfd=_cfd(d, cond, device, precision))
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (0.0, lx, -0.5 * ly, 0.5 * ly, 0.0, lz)
    sim.cfd.symmetry = "full"                            # the room is not mirror-symmetric
    sim.thermal = ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True,
                                  walls=[WallThermal(side="left", mode="temperature", value=12.0)])
    # a low-temperature panel radiator: 200 W into the air keeps its estimated surface rise near 27 K,
    # inside the Boussinesq range (beta * dT < 0.1) the solver's buoyancy model is accurate in
    sim.heat_sources = [HeatSource(name="Radiator", kind="box", position=(0.15, 0.0, 0.55), size=(0.1, 1.6, 0.7),
                                   power=200.0, frame="wind")]
    sim.fans = [Fan(name="Ceiling fan", position=(2.0, 0.0, 2.3), direction=(0.0, 0.0, -1.0), diameter=1.0,
                    mode="velocity", value=1.0, frame="wind")]
    sim.ports = [Port(name="Supply", kind="jet", position=(0.05, -1.0, 2.2), direction=(1.0, 0.0, 0.0), size=0.2,
                      speed=1.0, frame="wind"),
                 Port(name="Extract", kind="vent", position=(3.95, -1.0, 2.2), direction=(1.0, 0.0, 0.0), size=0.2,
                      speed=1.0, frame="wind")]
    sim.emitters = [Emitter(name="Smoke at the radiator", kind="point", position=(0.3, 0.0, 0.9), frame="wind")]
    q = math.pi * 0.1 ** 2 * 1.0                          # supply = extract, m³/s
    sim.notes = (f"The window wall (left, x = 0) is held at 12 °C; supply and extract move {q:.3f} m³/s each "
                 f"(about {q * 3600 / (lx * ly * lz):.1f} air changes per hour).")
    return d, [sim]


def _heated_cylinder(device, precision) -> tuple[Design, list[Simulation]]:
    diameter = 0.02
    try:
        from ...geometry import templates
        d = templates.make("cylinder", diameter=diameter)
    except Exception:  # noqa: BLE001 - a geometry package without templates: the circle by hand
        t = np.linspace(0.0, 2 * math.pi, 256, endpoint=False)
        d = Design(name="Cylinder", kind="generic", bodies=[Body("Cylinder", 0.5 * diameter * np.c_[np.cos(t),
                                                                                                    np.sin(t)])],
                   length_ref=diameter, force_ref=diameter)
    d.name = "Heated cylinder"
    d.source = {**(d.source or {}), "physics": True}
    for b in d.bodies:
        b.thermal, b.thermal_value = "temperature", 30.0
    air = fluids.air(20.0)
    nu = air.viscosity / air.density
    speed = 40.0 * nu / diameter                          # Re = 40 on the diameter
    cond = FlowConditions(fluid=air, speed=speed, alpha_deg=0.0)
    sim = Simulation(name="CFD · Re 40 · heated", design_id=d.id, analysis="cfd", conditions=cond,
                     cfd=_cfd(d, cond, device, precision))
    sim.cfd.reynolds_mode = "real"
    sim.thermal = ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=False)
    lo, hi = min(CYLINDER_NU_PUBLISHED.values()), max(CYLINDER_NU_PUBLISHED.values())
    corr = lange_nusselt(40.0)
    d.source.update({"benchmark": "isothermal cylinder, Re 40, Pr 0.71", "nu_reference_range": (lo, hi),
                     "nu_reference_sources": dict(CYLINDER_NU_PUBLISHED), "nu_correlation": round(corr, 4),
                     "nu_correlation_source": "Lange, Durst & Breuer (1998), Pr 0.7"})
    sim.notes = ("Forced convection: buoyancy is off (at this speed it would dominate: Ri ≈ 7), so the mean "
                 "Nusselt number can be compared with forced-convection references at Re = 40, Pr = 0.71: "
                 f"published simulations give {lo:.3f}–{hi:.3f} ({', '.join(CYLINDER_NU_PUBLISHED)}); the "
                 f"Lange, Durst & Breuer (1998) correlation gives {corr:.2f} (a fit, quoted to its precision), {100 * (corr / hi - 1):.1f}–"
                 f"{100 * (corr / lo - 1):.1f} % above that range, which is its own scatter. Judge a run by the "
                 "range, not by one number.")
    return d, [sim]


def _cavity(device, precision, ra: float = 1e5) -> tuple[Design, list[Simulation]]:
    t_hot, t_cold = 25.0, 15.0
    t_mean, dt = 0.5 * (t_hot + t_cold), t_hot - t_cold
    air = fluids.air(t_mean)
    side = cavity_side(ra, air, t_mean_c=t_mean, delta_t=dt)
    exp = round(math.log10(ra))
    sup = str(exp).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    published = math.isclose(ra, 10 ** exp, rel_tol=1e-9)
    nu_ref = CAVITY_NU_HOT_WALL.get(float(10 ** exp)) if published else None
    nu_avg = CAVITY_NU_AVERAGE.get(float(10 ** exp)) if published else None
    d = Design(name=f"Cavity · Ra 10{sup}", kind="generic", bodies=[], length_ref=side, force_ref=side,
               moment_ref=(0.5 * side, 0.5 * side),
               source={"type": "physics_template", "template": "cavity", "physics": True,
                       "benchmark": "de Vahl Davis 1983",
                       "rayleigh": float(ra), "prandtl": 0.71,
                       # the hot wall's mean Nu, Nu₀, is what the run measures; the cavity average Nū is
                       # named apart so the two are never compared with each other
                       "nu_hot_wall_reference": nu_ref, "nu_cavity_average": nu_avg},
               notes=f"Square cavity, side {side * 1e3:.2f} mm: no body, the walls drive the flow.")
    beta = 1.0 / (t_mean + 273.15)
    u_ff = math.sqrt(G0 * beta * dt * side)              # free-fall velocity: the time scale
    cond = FlowConditions(fluid=air, speed=float(f"{u_ff:.4g}"), alpha_deg=0.0)
    sim = Simulation(name=f"Cavity · Ra 10{sup} · Pr 0.71", design_id=d.id, analysis="cfd", conditions=cond,
                     cfd=_cfd(d, cond, device, precision))
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (0.0, side, 0.0, side)
    sim.cfd.reynolds_mode = "real"
    sim.thermal = ThermalSettings(enabled=True, t_ambient=t_mean, buoyancy=True, prandtl=0.71,
                                  walls=[WallThermal(side="left", mode="temperature", value=t_hot),
                                         WallThermal(side="right", mode="temperature", value=t_cold)])
    ref = (f" Reference: the hot wall's mean Nusselt number Nu₀ = {nu_ref:.3f} (the cavity-average Nū "
           f"{nu_avg:.3f} is a different quantity)." if nu_ref is not None and nu_avg is not None else
           " No published value at this Rayleigh number.")
    sim.notes = (f"de Vahl Davis (1983): hot left wall {t_hot:g} °C, cold right wall {t_cold:g} °C, adiabatic top "
                 f"and bottom, Ra = {ra:.0e}, Pr = 0.71.{ref} Nu₀ at other Rayleigh numbers: 1.117 (10³), "
                 "2.238 (10⁴), 4.509 (10⁵), 8.817 (10⁶).")
    return d, [sim]


def build_template(key: str, *, device: str | None = None, precision: str | None = None,
                   ra: float | None = None) -> tuple[Design, list[Simulation]]:
    """The design and starter simulations of physics template ``key`` (see the module docstring).

    ``ra`` sets the cavity's Rayleigh number (default 10⁵). Raises KeyError for an unknown key."""
    template_info(key)
    if key == "enclosure":
        return _enclosure(device, precision)
    if key == "room":
        return _room(device, precision)
    if key == "heated_cylinder":
        return _heated_cylinder(device, precision)
    return _cavity(device, precision, 1e5 if ra is None else float(ra))


def add_template(state, key: str, **kw) -> tuple[Design, list[Simulation]]:
    """Add template ``key`` to the open project through ``state`` (an AppState) and select it.

    Names are made unique ("Closed room 2"); the first simulation becomes the current one.
    Raises RuntimeError without an open project."""
    if getattr(state, "project", None) is None:
        raise RuntimeError("no project is open")
    if not kw.get("device") or not kw.get("precision"):
        try:
            from .. import settings as app_settings
            s = app_settings.get()
            kw.setdefault("device", s.device)
            kw.setdefault("precision", s.precision)
        except Exception:  # noqa: BLE001 - preferences are optional: the defaults module chooses
            kw.setdefault("device", None)
    design, sims = build_template(key, **kw)
    design.name = state.unique_design_name(design.name)
    state.add_design(design, select=True)
    for i, sim in enumerate(sims):
        sim.design_id = design.id
        sim.name = state.unique_simulation_name(design.id, sim.name)
        state.add_simulation(sim, select=i == 0)
    return design, sims


__all__ = ["CAVITY_NU_AVERAGE", "CAVITY_NU_HOT_WALL", "CYLINDER_NU_PUBLISHED", "PHYSICS_TEMPLATES", "PhysicsTemplate", "add_template", "build_template",
           "cavity_side", "lange_nusselt", "template_info"]
