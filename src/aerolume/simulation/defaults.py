"""What a *new* simulation starts from, without importing the solver.

Every place that creates a simulation (a new study, Setup's "New simulation",
the project tree, the command line) must hand the user the same starting point,
or the first Run behaves differently depending on which button made the
simulation. Those places include GUI code that runs on the event thread, so
this module imports only the data model: importing :mod:`.plan` would pull in
PyTorch and freeze the window for seconds.

:mod:`.plan` re-exports :func:`default_cfd_settings` and uses
:func:`geometry_class` for its layout rules, so there is one definition of each.
"""
from __future__ import annotations

import math

from ..core import units
from ..model import KINDS, CFDSettings, Design, FlowConditions, GroundConfig, Simulation, WingSettings

#: Quality a new simulation starts at. The first run is how a new user decides
#: whether the program works at all: Draft finishes the NACA 2412 case in about
#: 10 s on a mid-range GPU and the Draft sedan in 1.5-3 min, while Standard (1-2 M
#: cells for a car) takes 10 min or more, and minutes of a run that has not yet
#: produced numbers read as "it does not run". Setup's Quality steps up from here.
STARTER_QUALITY = "draft"

#: Above this chord Reynolds number a new airfoil simulation starts in the capped
#: Reynolds mode. No grid this tool can afford resolves an airfoil boundary layer at
#: Re 1e6 and there is no wall or transition model: NACA 2412 at Re 1e6, 4 deg, Draft
#: measured Cl 0.16 / Cd 0.096 with the real Re (WALE LES) against the viscous polar's
#: 0.72 / 0.007. A capped run is a resolved, deterministic low-Re flow and says so;
#: the polar gives the answer at the real Re.
AIRFOIL_CAP_RE = 1.0e5

#: The Reynolds number a capped airfoil simulation runs at, fixed when the
#: simulation is created and stored in ``cfd.reynolds_cap``. It is the highest
#: Re that *every* quality on *every* device resolves: the coarsest airfoil grid
#: (Draft on a CPU, 40 cells per chord, ``plan.CELLS_PER_SIZE``) holds the
#: planner's ``BL_CELLS_MIN`` = 4 cells across the Blasius thickness 5 c / sqrt(Re)
#: up to Re = (5 * 40 / 4)^2 = 2,500 (``tests/test_simulation_plan.py`` re-derives
#: it from the planner's tables). A cap left to the planner followed the grid
#: instead -- (5 / (4 h))^2 -- so the same NACA 2412 case simulated Re 2,500 on a
#: CPU, 6,400 at Draft and 25,600 at Standard on a GPU: stepping up the quality to
#: check grid convergence compared two different flows, and the drag "grid
#: effect" was mostly a Reynolds effect (Cd falls roughly as Re^-0.5 there).
AIRFOIL_DEFAULT_CAP_RE = 2500.0


def default_reynolds_cap(design: Design) -> float | None:
    """The fixed cap a capped simulation of ``design`` starts with, or None.

    Airfoils get :data:`AIRFOIL_DEFAULT_CAP_RE`. Vehicles and bluff bodies keep
    the real Reynolds number by default; if a user caps one, the planner falls
    back to the grid's resolvable Re and says that it then depends on the grid.
    3D designs have no default cap (see :func:`default_cfd_settings`).
    """
    if getattr(design, "dimension", 2) == 3:
        return None
    return AIRFOIL_DEFAULT_CAP_RE if geometry_class(design) == "airfoil" else None


def geometry_class(design: Design) -> str:
    """Which layout rules apply to a design: "airfoil", "vehicle" or "generic".

    A "vehicle" whose force reference is its length (the front-wing template:
    coefficients per chord) is a wing on a road: its domain and resolution
    follow the airfoil rules -- a ceiling a few chords away, cells per chord --
    while the road stays on.

    3D designs (DESIGN.md 7.7) use the 3D planner's classes: "vehicle",
    "generic" or "wing" by kind (a 3D design's ``force_ref`` is its frontal
    height, so the 2D wing-on-a-road rule does not apply to it).
    """
    if getattr(design, "dimension", 2) == 3:
        return design.kind if design.kind in ("vehicle", "wing") else "generic"
    kind = design.kind if design.kind in KINDS else "generic"
    if kind == "vehicle" and abs(float(design.force_ref) - float(design.length_ref)) <= 1e-9 * design.length_ref:
        return "airfoil"
    return kind


def default_cfd_settings(design: Design, conditions: FlowConditions | None = None, *,
                         device: str = "auto", precision: str = "auto") -> CFDSettings:
    """CFD settings a *new* simulation of ``design`` should start from.

    * **Draft quality** (:data:`STARTER_QUALITY`).
    * **Capped Reynolds number for airfoils above** :data:`AIRFOIL_CAP_RE`. A
      "real" run there returns Cl several times too low and Cd several times too
      high while looking authoritative; a capped run is a resolved low-Re flow,
      labelled as such, and its result lists the viscous polar at the real Re next
      to it (see ``runner``). Vehicles and bluff bodies keep the real Re: their
      separated flow is far less sensitive to it.
    * **The cap itself is fixed here** (:data:`AIRFOIL_DEFAULT_CAP_RE`), not left to
      the planner, so changing the quality or the device refines the grid of the
      *same* flow.

    ``conditions`` defaults to ``FlowConditions()`` (the speed the new simulation
    starts with).

    3D designs (DESIGN.md 7.7) also start at Draft -- a Draft car takes a few
    minutes on a mid-range GPU, Standard tens of minutes -- with the real
    Reynolds number (a 3D grid cannot resolve a wing's boundary layer at any
    affordable cap either; the Wing analysis is the tool for that) and the
    automatic symmetry plane.
    """
    cond = conditions if conditions is not None else FlowConditions()
    cfd = CFDSettings(quality=STARTER_QUALITY, device=device or "auto", precision=precision or "auto")
    if getattr(design, "dimension", 2) == 3:
        cfd.symmetry = "auto"
        return cfd
    try:
        re_real = cond.reynolds(float(design.length_ref))
    except (TypeError, ValueError, ZeroDivisionError):
        re_real = 0.0
    if geometry_class(design) == "airfoil" and math.isfinite(re_real) and re_real > AIRFOIL_CAP_RE:
        cfd.reynolds_mode = "capped"
        cfd.reynolds_cap = default_reynolds_cap(design)
    return cfd


def default_analysis(design: Design) -> str:
    """The analysis a *new* simulation of ``design`` starts with.

    Wings (kind "wing", 3D) start with the finite-wing analysis ("wing": vortex
    lattice on viscous section polars, seconds instead of minutes, and the right
    tool for lift and induced drag at flight Reynolds numbers); everything else
    starts with CFD.
    """
    return "wing" if design.kind == "wing" else "cfd"


def default_wing_settings(design: Design | None = None) -> WingSettings:
    """Wing-analysis settings a new simulation starts from: vortex lattice, viscous
    strip drag, the usual -4 ... 12 degree sweep in 1-degree steps (XFLR5's
    default kind of polar; the lattice handles any planform, sweep and dihedral,
    the lifting line only straight, moderately swept wings)."""
    return WingSettings(method="vlm", viscous=True)


def default_simulation(design: Design, name: str = "", conditions: FlowConditions | None = None, *,
                       device: str = "auto", precision: str = "auto") -> Simulation:
    """A new :class:`~aerolume.model.Simulation` of ``design`` with every default
    applied (analysis, road, CFD settings, wing settings), so the GUI, the CLI and
    scripts start from the same place. Vehicles (and designs imported standing on
    the road) start on a moving road, with spinning wheels when they have any."""
    cond = conditions if conditions is not None else FlowConditions()
    analysis = default_analysis(design)
    label = name or (f"{design.name} {'wing polar' if analysis == 'wing' else 'CFD'}").strip()
    src = design.source if isinstance(design.source, dict) else {}
    on_road = design.kind == "vehicle" or bool(src.get("ground")) or bool(src.get("on_road"))
    wheels = any(b.role == "wheel" for b in design.bodies) or any(m.role == "wheel" for m in design.meshes)
    return Simulation(name=label, design_id=design.id, analysis=analysis, conditions=cond,
                      ground=GroundConfig(enabled=on_road, moving=True, rotate_wheels=wheels),
                      cfd=default_cfd_settings(design, cond, device=device, precision=precision),
                      wing=default_wing_settings(design))


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------

#: Below this free-stream speed [m/s] a simulation is named by its Reynolds number.
#: Such runs are low-Reynolds-number cases set up by Re (the 0.2 m sphere at Re 100
#: of the scripting example runs at 0.0073 m/s in air): the speed is a by-product
#: of the reference length and the fluid, and "CFD 3D · 0.0073 m/s" in the project
#: tree, Compare and the report reads like a typing error. No vehicle, wing or
#: wind-tunnel model is studied this slowly, so the rule never renames those.
NAME_BY_RE_BELOW_SPEED = 0.1

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def reynolds_text(re_: float) -> str:
    """The Reynolds number as simulation names write it: "100", "2,500", "1.0×10⁶".

    Below 10⁴ the whole number reads best (and is exact for the usual 100 / 1,000
    validation cases); above, one decimal of the mantissa, as XFLR5 and most papers
    label polars. "—" for anything that is not a positive finite number.
    """
    try:
        r = float(re_)
    except (TypeError, ValueError):
        return "—"
    if not (math.isfinite(r) and r > 0):
        return "—"
    if r < 1e4:
        return f"{r:,.0f}"
    e = math.floor(math.log10(r))
    m = r / 10 ** e
    if round(m, 1) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{m:.1f}×10{str(e).translate(_SUP)}"


def _num(x: float, digits: int = 3) -> str:
    """``x`` to ``digits`` significant figures without trailing zeros (4 -> "4", 12.5 -> "12.5")."""
    v = float(x)
    if abs(v) < 1e-10:          # round-off (an angle of -1e-17 after a unit conversion)
        v = 0.0
    out = units.sig(v, digits, grouping=False)
    if "." in out and "e" not in out:
        out = out.rstrip("0").rstrip(".")
    return out


def names_by_reynolds(design: Design, sim: Simulation) -> bool:
    """Whether :func:`simulation_name` labels ``sim`` by Re rather than by speed when the
    caller does not say how the run was specified (:data:`NAME_BY_RE_BELOW_SPEED`)."""
    try:
        speed = float(sim.conditions.speed)
        re_ = sim.conditions.reynolds(float(design.length_ref))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return math.isfinite(speed) and speed < NAME_BY_RE_BELOW_SPEED and math.isfinite(re_) and re_ > 0


def simulation_name(design: Design, sim: Simulation, *, by_reynolds: bool | None = None) -> str:
    """The name a new simulation gets from its settings: "CFD 3D · 30 m/s · α 0°",
    "CFD 3D · Re 100 · α 0°", "Polar · Re 1.0×10⁶", "Wing polar · 25 m/s · LLT".

    One definition for the scripting API, the AI tools and the example projects, so
    a run is called the same whichever of them made it. The flow is named the way
    it was specified: ``by_reynolds=True`` when the caller set the Reynolds number
    (``run_cfd(body, re=100)``), False when it set the speed; None decides from the
    speed (:func:`names_by_reynolds`). Polars are always named by Re, which is what
    an airfoil polar is a function of. Non-default settings that change the answer
    (yaw, a quality above Draft, LLT, inviscid, a sensitivity map) are appended, so
    two runs of one design never share a name by accident.
    """
    c = sim.conditions
    re_ = c.reynolds(float(design.length_ref)) if design.length_ref else float("nan")
    if sim.analysis == "polar":
        return f"Polar · Re {reynolds_text(re_)}"
    by_re = names_by_reynolds(design, sim) if by_reynolds is None else bool(by_reynolds)
    flow = f"Re {reynolds_text(re_)}" if by_re and math.isfinite(re_) else f"{_num(c.speed)} m/s"
    if sim.analysis == "wing":
        bits = ["Wing polar", flow]
        if sim.wing.method == "llt":
            bits.append("LLT")
        if not sim.wing.viscous:
            bits.append("inviscid")
        if abs(c.yaw_deg) > 0:
            bits.append(f"β {_num(c.yaw_deg)}°")
        return " · ".join(bits)
    bits = ["CFD 3D" if getattr(design, "dimension", 2) == 3 else "CFD", flow, f"α {_num(c.alpha_deg)}°"]
    if abs(c.yaw_deg) > 0:
        bits.append(f"yaw {_num(c.yaw_deg)}°")
    if sim.cfd.quality != STARTER_QUALITY:
        bits.append(sim.cfd.quality.capitalize())
    if sim.cfd.compute_sensitivity:
        bits.append(f"sensitivity {sim.cfd.sensitivity_objective}")
    return " · ".join(bits)


__all__ = ["AIRFOIL_CAP_RE", "AIRFOIL_DEFAULT_CAP_RE", "NAME_BY_RE_BELOW_SPEED", "STARTER_QUALITY",
           "default_analysis", "default_cfd_settings", "default_reynolds_cap", "default_simulation",
           "default_wing_settings", "geometry_class", "names_by_reynolds", "reynolds_text", "simulation_name"]
