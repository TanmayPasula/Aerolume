"""Editors for the placed physics of a simulation (DESIGN.md section 10).

The Setup page edits a :class:`~aerolume.model.Simulation`'s emitters, heat
transfer, heat sources, fans, jets and vents, its domain (wind tunnel or closed
box) and its solver diagnostics. This module holds two layers:

* **Pure helpers** (numpy only, no Qt state): the wind-frame placement of an
  item (``to_wind`` / ``from_wind``: a ``frame="body"`` item turns with the
  pitched geometry, a ``frame="wind"`` one stays in the tunnel), the default
  thermal properties of a fluid, the dimensionless groups of a thermal case
  (:func:`thermal_numbers`: Re, Pr, Gr, Ra, Ri, Pe and the temperature scale
  they rest on, each with the rule that produced it), the warnings a planner
  would give about a setup (:func:`physics_warnings`: an unbalanced closed box,
  an item outside the domain, a Boussinesq approximation stretched too far),
  default placements for new items, and the drawing specs the 2D and 3D case
  previews paint (:func:`scene_specs`, plain dicts in wind-frame metres, so the
  canvases know nothing about the model).
* **Sections** (Qt): controllers that add their rows to a
  :class:`~aerolume.app.widgets.common.PropertyGroup` of the page's property
  grid, so every editor shares the inspector's label column, row height and
  unit column (docs/ui_guidelines.md section 3). A section edits the
  simulation in place and emits ``edited`` (``designEdited`` when it changed a
  body's thermal condition, which lives on the Design); the page saves with
  its usual debounce. Lists of placed items (:class:`PlacedItemsSection`) show
  the items, Add / Duplicate / Remove, and the selected item's fields; the
  selection is shared with the case preview, where the same item is drawn with
  handles.

Every quantity carries its unit, every choice its plain-language tooltip, and
numbers the program fills in (fluid properties on Auto, a default placement)
are shown, never hidden: the user sees what will run before pressing Run.
"""
from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

from ...core import units
from ...model import (
    Design,
    Emitter,
    Fan,
    Fluid,
    HeatSource,
    Port,
    Simulation,
    WallThermal,
)
from .param_form import (
    QuantitySpinBox,
    display_unit,
    hbox,
    keep_start_visible,
    label,
    set_line_text,
    set_prop,
)

# ---------------------------------------------------------------------------
# copy (docs/ui_guidelines.md section 6; reviewed with design:ux-copy)
# ---------------------------------------------------------------------------

#: The Simulation list attribute of each kind of placed item, in inspector order.
ITEM_LISTS = ("emitters", "heat_sources", "fans", "ports")
ITEM_NOUN = {"emitters": "emitter", "heat_sources": "heat source", "fans": "fan", "ports": "jet or vent"}
ITEM_TITLE = {"emitters": "Emitters", "heat_sources": "Heat sources", "fans": "Fans", "ports": "Jets and vents"}
#: What an empty list means, said where the list would be.
EMPTY_TEXT = {
    "emitters": {2: "Automatic: smoke enters with the inflow across the inlet.",
                 3: "Automatic: the smoke rake below, upstream of the body.",
                 "closed": "None. A closed box has no inflow to carry smoke in: add an emitter to see the flow."},
    "heat_sources": "None. Add one for a heater (a chip, a radiator) or, with negative power, a cooler.",
    "fans": "None. Add one to push the fluid: a fan is modelled as an actuator disc, a thin layer of force.",
    "ports": "None. Add a jet to blow fluid in, or a vent to draw it out.",
}
EMITTER_KINDS_2D = ("point", "line", "rake", "rect")
EMITTER_KINDS_3D = ("point", "line", "rake", "rect", "disc", "plane")
EMITTER_KIND_TEXT = {"point": "Point", "line": "Line (slot)", "rake": "Rake (nozzles)", "rect": "Rectangle",
                     "disc": "Disc", "plane": "Plane"}
EMITTER_KIND_TIP = {
    "point": "One nozzle: a single streak.",
    "line": "A continuous slot from the start to the end point: a smoke sheet.",
    "rake": "A row of separate nozzles from the start to the end point: one streak per nozzle.",
    "rect": "Releases over a whole rectangle (2D) or box (3D) centred on the position.",
    "disc": "Releases over a disc facing the normal direction (3D).",
    "plane": "Releases over a rectangle facing the normal direction (3D): a smoke sheet across the flow.",
}
SPECIES_ITEMS = (("dye", "Smoke", "A passive dye carried and spread by the flow. It does not change the flow."),
                 ("particles", "Particles", ("Tracer points carried by the flow (streaklines; pulsed: "
                                             "timelines). They do not change the flow.")))
SMOKE_COLOURS = (("#f5f5f5", "White"), ("#f2a33a", "Amber"), ("#4aa3df", "Blue"), ("#d65f97", "Magenta"),
                 ("#61b865", "Green"))
#: smoke channels the solvers carry (DESIGN.md 10.1): emitters map to them by colour
MAX_DYE_CHANNELS = 4
HEAT_KINDS = {2: ("rect", "disc"), 3: ("box", "sphere")}
HEAT_KIND_TEXT = {"rect": "Rectangle", "disc": "Disc", "box": "Box", "sphere": "Sphere"}
FAN_MODE_ITEMS = (
    ("thrust", "Thrust", "The force the fan applies to the fluid (2D: per metre of span)."),
    ("pressure_jump", "Pressure rise", "The static-pressure rise across the disc."),
    ("velocity", "Flow speed", "The mean axial speed through the disc; the solver adjusts the push to reach it."),
)
THERMAL_MODE_ITEMS = (
    ("adiabatic", "Adiabatic", "No heat crosses the surface."),
    ("temperature", "Temperature", "The surface is held at this temperature."),
    ("heat_flux", "Heat flux", "This heat flux leaves the surface into the fluid (negative: the surface cools it)."),
)
FRAME_ITEMS = (("wind", "Fixed in the tunnel", "Stays put when the angle of attack or yaw turns the body."),
               ("body", "Moves with the body", "Turns with the body when the angle of attack or yaw changes."))
#: the "wind" frame's name in a closed box, which has no tunnel (the same frame: the domain's axes)
FRAME_WIND_CLOSED = "Fixed in the box"
PORT_ITEMS = (("jet", "Jet", "Blows fluid into the domain at the set speed."),
              ("vent", "Vent", "Draws fluid out of the domain at the set speed."))
DOMAIN_ITEMS = (("tunnel", "Wind tunnel", "Flow enters on the left at the set speed and leaves on the right."),
                ("closed", "Closed box", ("Every side is a wall: a room, an enclosure or a cavity. Fans, jets "
                                          "and buoyancy drive the flow; the speed sets only the time scale.")))
TIMESTEP_ITEMS = (("cfl", "CFL limit", ("The largest stable step for the CFL number, recomputed as the flow "
                                        "changes.")),
                  ("pid", "PID controller", ("A controller holds the CFL number near its target and smooths "
                                             "changes of the step (AeroJAX).")))
#: domain sides in the order the walls are listed; front/back exist in 3D only
SIDE_TEXT = {2: {"left": "Left (x min)", "right": "Right (x max)", "bottom": "Bottom (y min)",
                 "top": "Top (y max)"},
             3: {"left": "Left (x min)", "right": "Right (x max)", "front": "Side (y min)", "back": "Side (y max)",
                 "bottom": "Bottom (z min)", "top": "Top (z max)"}}
AXIS_TIP = {2: ("x, downstream", "y, up"), 3: ("x, downstream", "y, to the left looking downstream", "z, up")}
#: a closed box has no stream: its axes say only where up is (gravity acts along -y in 2D, -z in 3D)
AXIS_TIP_CLOSED = {2: ("x", "y, up"), 3: ("x", "y", "z, up")}

HEAT_TIP = ("Adds the energy equation: temperature is carried by the flow and diffuses. With buoyancy off the "
            "flow and the forces are unchanged.")
BUOYANCY_TIP = ("Boussinesq buoyancy g·β·(T − T ambient), acting downwards along −y (2D) or −z (3D). "
                "Accurate while β·ΔT stays below about 0.1 (ΔT below about 30 K in air).")
AMBIENT_TIP = ("Free-stream temperature in a wind tunnel, initial temperature in a closed box. The inlet "
               "stays at it.")
PROPERTY_TIP = {
    "beta": "Thermal expansion coefficient β. Gases: 1/T (ideal gas). Auto shows the fluid's value.",
    "prandtl": "Prandtl number ν/α: how fast momentum diffuses relative to heat. Air ≈ 0.71, water ≈ 7.",
    "conductivity": "Thermal conductivity k. Sets the heat transfer rate and the Nusselt number.",
    "cp": "Specific heat at constant pressure. With the density, it converts heat to a temperature rise.",
}
AUTO_TIP = "Auto: the value for this fluid at the ambient temperature (shown). Typing a value fixes it."
DIAG_TIP = ("Records solver health per block of steps: RMS divergence, change of every field, CFL number, "
            "time step, steps per second and the pressure-solve residual. Costs a few % of run time; shown on "
            "the Simulate page and exported as CSV.")
#: positions are stored in the frame the item is attached to (to_wind / from_wind)
POSITION_TIP = ("Centre, metres: in the tunnel's (or box's) axes when the item is fixed there, in the body's "
                "axes, as drawn on the Geometry page, when it moves with the body.")
FRAME_TIP = ("What the position is attached to: the domain (it stays put), or the body (it turns with the "
             "angle of attack or yaw).")
ROW_BUTTON_W = 64     # the Auto switches, as wide as the inspector's other row buttons


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

def dimension_of(design: Design | None) -> int:
    """2 or 3: the dimension a design (and so its simulations' placements) uses."""
    return 3 if design is not None and getattr(design, "dimension", 2) == 3 else 2


def vec(v: Any, dim: int, default: float = 0.0) -> np.ndarray:
    """``v`` as a float vector of length ``dim`` (shorter input padded with ``default``)."""
    out = np.full(dim, float(default))
    if v is not None:
        a = np.asarray(v, dtype=float).ravel()[:dim]
        out[: len(a)] = a
    return out


def unit_vec(v: Any, dim: int, fallback: Sequence[float] | None = None) -> np.ndarray:
    """``v`` normalised (length ``dim``); +x when it has no direction."""
    a = vec(v, dim)
    n = float(np.linalg.norm(a))
    if n > 1e-12 and np.all(np.isfinite(a)):
        return a / n
    out = vec(fallback if fallback is not None else (1.0,), dim)
    return out / max(float(np.linalg.norm(out)), 1e-12)


def _tup(a: Any) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(a, dtype=float).ravel())


def fmt_number(v: float | None, digits: int = 2) -> str:
    """A dimensionless number for the Estimate: "0.71", "4.5", "1.2×10⁵" (superscript exponents)."""
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
        sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"{m:.{digits - 1}f}×10{sup}"
    return units.sig(v, digits + (1 if a >= 1 else 0))


def body_bounds(design: Design | None) -> tuple[float, ...] | None:
    """The design's own bounding box (design frame): (x0, x1, y0, y1) in 2D, + (z0, z1) in 3D, or None."""
    if design is None:
        return None
    if dimension_of(design) == 3:
        meshes = [m for m in design.meshes if len(m.faces) and len(m.vertices)]
        if meshes:
            b = np.array([m.bounds() for m in meshes])
            return (float(b[:, 0].min()), float(b[:, 1].max()), float(b[:, 2].min()), float(b[:, 3].max()),
                    float(b[:, 4].min()), float(b[:, 5].max()))
        if design.kind == "wing" and design.wing is not None:
            with contextlib.suppress(Exception):           # an unloftable planform has no box yet
                from .geometry_canvas import design_meshes3d, meshes_bounds
                return meshes_bounds(design_meshes3d(design))
        return None
    if not design.bodies:
        return None
    pts = np.vstack([np.asarray(b.points, dtype=float).reshape(-1, 2) for b in design.bodies])
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def explicit_box(sim: Simulation | None, dim: int) -> tuple[float, ...] | None:
    """``sim.cfd.domain_box`` when it is set and has the right length for ``dim``, else None."""
    box = getattr(getattr(sim, "cfd", None), "domain_box", None) if sim is not None else None
    if box is None or len(box) != 2 * dim:
        return None
    b = tuple(float(v) for v in box)
    if not all(math.isfinite(v) for v in b) or any(b[2 * i + 1] <= b[2 * i] for i in range(dim)):
        return None
    return b


def domain_extent(design: Design | None, sim: Simulation | None, plan_extent: Sequence[float] | None = None
                  ) -> tuple[float, ...] | None:
    """The flow domain to place items in (wind frame, metres): the explicit box, else the planned
    extent, else a generous box around the body (a tunnel: 3 lengths ahead, 6 behind)."""
    dim = dimension_of(design)
    box = explicit_box(sim, dim)
    if box is not None:
        return box
    if plan_extent is not None and len(plan_extent) == 2 * dim:
        return tuple(float(v) for v in plan_extent)
    b = body_bounds(design)
    if b is None:
        return None
    size = max(b[2 * i + 1] - b[2 * i] for i in range(dim)) or 1.0
    closed = getattr(getattr(sim, "cfd", None), "domain_mode", "tunnel") == "closed"
    up, down, side = (1.0, 1.0, 1.0) if closed else (3.0, 6.0, 2.0)
    out = [b[0] - up * size, b[1] + down * size]
    for i in range(1, dim):
        out += [b[2 * i] - side * size, b[2 * i + 1] + side * size]
    return tuple(out)


def attitude(design: Design | None, sim: Simulation | None) -> tuple[float, float, np.ndarray]:
    """(alpha deg, yaw deg, centre of rotation) the planner turns the geometry by (DESIGN.md 3 and 7.2)."""
    if design is None or sim is None:
        return 0.0, 0.0, np.zeros(2)
    c = sim.conditions
    if dimension_of(design) == 3:
        return float(c.alpha_deg), float(c.yaw_deg), np.asarray(design.moment_ref3, dtype=float)
    return float(c.alpha_deg), 0.0, np.asarray(design.moment_ref, dtype=float)


def _rotation(dim: int, alpha: float, yaw: float) -> np.ndarray:
    """Design frame -> wind frame rotation: 2D by -alpha (counter-clockwise positive); 3D yaw about +z by
    -yaw, then pitch about +y by +alpha (``simulation.plan3d.rotation3``)."""
    if dim == 2:
        a = math.radians(-alpha)
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    a, b = math.radians(alpha), math.radians(-yaw)
    ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    rz = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0], [0.0, 0.0, 1.0]])
    return ry @ rz


def to_wind(points: Any, design: Design | None, sim: Simulation | None, frame: str = "body", *,
            direction: bool = False) -> np.ndarray:
    """Design-frame point(s) (or, with ``direction``, vectors) of an item in the wind frame.

    A ``frame="wind"`` item is already where the tunnel has it; a ``"body"`` item turns with the
    geometry about the moment point by the angle of attack (and yaw in 3D), exactly as the planner
    turns the bodies, so a heater fixed to a pitched car stays on the car."""
    dim = dimension_of(design)
    p = np.atleast_2d(np.asarray(points, dtype=float))[:, :dim]
    alpha, yaw, ref = attitude(design, sim)
    if frame == "wind" or (abs(alpha) < 1e-12 and abs(yaw) < 1e-12):
        return p.copy()
    r = _rotation(dim, alpha, yaw)
    if direction:
        return p @ r.T
    return (p - ref[:dim]) @ r.T + ref[:dim]


def from_wind(points: Any, design: Design | None, sim: Simulation | None, frame: str = "body", *,
              direction: bool = False) -> np.ndarray:
    """Inverse of :func:`to_wind`: a wind-frame point dragged on the case preview, back to the design
    frame the item stores (unchanged for ``frame="wind"``)."""
    dim = dimension_of(design)
    p = np.atleast_2d(np.asarray(points, dtype=float))[:, :dim]
    alpha, yaw, ref = attitude(design, sim)
    if frame == "wind" or (abs(alpha) < 1e-12 and abs(yaw) < 1e-12):
        return p.copy()
    r = _rotation(dim, alpha, yaw)
    if direction:
        return p @ r
    return (p - ref[:dim]) @ r + ref[:dim]


def snap(value: float, step: float) -> float:
    """``value`` rounded to the nearest multiple of ``step`` (no rounding for a step <= 0)."""
    if not step or step <= 0 or not math.isfinite(value):
        return float(value)
    return float(round(value / step) * step) + 0.0       # + 0.0: no "-0.0" in a field


def snap_step(length: float) -> float:
    """A round 1-2-5 snapping step for a view in which ``length`` [m] is about 8 px (or 1/50 of a body)."""
    if not length > 0 or not math.isfinite(length):
        return 0.0
    e = math.floor(math.log10(length))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10.0 ** e >= length * 0.999:
            return m * 10.0 ** e
    return 10.0 ** (e + 1)


# ---------------------------------------------------------------------------
# items: anchors, moves and defaults
# ---------------------------------------------------------------------------

def items_of(sim: Simulation | None, kind: str) -> list:
    """The simulation's list of ``kind`` ("emitters", "heat_sources", "fans", "ports"); [] without one."""
    if sim is None or kind not in ITEM_LISTS:
        return []
    out = getattr(sim, kind, None)
    return out if isinstance(out, list) else []


def has_extent(e: Emitter) -> bool:
    """Whether an emitter runs from ``position`` to ``end`` (a line or a rake)."""
    return e.kind in ("line", "rake")


def anchor(obj: Any, dim: int) -> np.ndarray:
    """The point an item is held by (design frame): an emitter line's middle, else its position."""
    p = vec(getattr(obj, "position", None), dim)
    if isinstance(obj, Emitter) and has_extent(obj) and obj.end is not None:
        return 0.5 * (p + vec(obj.end, dim))
    return p


def move_item(obj: Any, handle: str, point: Any, dim: int) -> None:
    """Move an item's ``handle`` ("centre", "start" or "end") to ``point`` (design frame, metres).

    "centre" translates the whole item (both ends of a line or rake); "start" / "end" move one end
    of a line or a rake. Positions are stored with ``dim`` components (2D items keep (x, y))."""
    p = vec(point, dim)
    if handle == "centre":
        delta = p - anchor(obj, dim)
        obj.position = _tup(vec(obj.position, dim) + delta)
        if isinstance(obj, Emitter) and obj.end is not None:
            obj.end = _tup(vec(obj.end, dim) + delta)
    elif handle == "start":
        obj.position = _tup(p)
    elif handle == "end" and isinstance(obj, Emitter):
        obj.end = _tup(p)


def _span(bounds: Sequence[float], i: int) -> float:
    return float(bounds[2 * i + 1] - bounds[2 * i])


def _context_box(design: Design | None, sim: Simulation | None, extent) -> tuple[tuple[float, ...], int]:
    """(box to place relative to, dim): the body box (wind frame) when there is a body, else the domain."""
    dim = dimension_of(design)
    b = body_bounds(design)
    if b is None:
        b = domain_extent(design, sim, extent) or tuple([0.0, 1.0] * dim)
    return tuple(float(v) for v in b), dim


def _unique(names: set[str], base: str) -> str:
    if base not in names:
        return base
    k = 2
    while f"{base} {k}" in names:
        k += 1
    return f"{base} {k}"


def default_emitter(design: Design | None, sim: Simulation | None, extent=None) -> Emitter:
    """A new smoke emitter: in a tunnel a rake of nine nozzles upstream of the body across its
    height (2D) or its span (3D, at 40 % of the height, the solver's automatic rake rule); in a
    closed box a point at the domain's centre. Fixed in the tunnel, white, continuous."""
    b, dim = _context_box(design, sim, extent)
    names = {e.name for e in items_of(sim, "emitters")}
    name = _unique(names, "Smoke")
    closed = sim is not None and sim.cfd.domain_mode == "closed"
    dom = domain_extent(design, sim, extent)
    if closed and dom is not None:
        c = [0.5 * (dom[2 * i] + dom[2 * i + 1]) for i in range(dim)]
        return Emitter(name=name, kind="point", position=tuple(c), frame="wind", color=SMOKE_COLOURS[0][0])
    lx = _span(b, 0)
    if dim == 2:
        h = max(_span(b, 1), 0.3 * lx)
        yc = 0.5 * (b[2] + b[3])
        x = b[0] - 0.5 * max(lx, h)
        return Emitter(name=name, kind="rake", position=(x, yc - 0.6 * h), end=(x, yc + 0.6 * h), count=9,
                       frame="wind", color=SMOKE_COLOURS[0][0])
    wy, hz = _span(b, 1), _span(b, 2)
    x = b[0] - 0.4 * min(max(wy, hz), lx)
    z = b[4] + 0.4 * hz
    return Emitter(name=name, kind="rake", position=(x, b[2] - 0.2 * wy, z), end=(x, b[3] + 0.2 * wy, z),
                   count=9, frame="wind", color=SMOKE_COLOURS[0][0])


def default_heat_source(design: Design | None, sim: Simulation | None, extent=None) -> HeatSource:
    """A new heater: in a closed box near the middle of its floor, in a tunnel just below the body's
    front; a tenth of the reference size, 100 W (2D: W per metre of span)."""
    b, dim = _context_box(design, sim, extent)
    names = {h.name for h in items_of(sim, "heat_sources")}
    dom = domain_extent(design, sim, extent)
    closed = sim is not None and sim.cfd.domain_mode == "closed"
    ref = max(_span(b, i) for i in range(dim)) or 1.0
    s = 0.1 * ref
    if closed and dom is not None:
        c = [0.5 * (dom[0] + dom[1])] + [0.5 * (dom[2 * i] + dom[2 * i + 1]) for i in range(1, dim)]
        up = dim - 1                                      # the vertical axis
        c[up] = dom[2 * up] + 0.15 * (dom[2 * up + 1] - dom[2 * up])
    else:
        c = [b[0] - 0.5 * ref] + [0.5 * (b[2 * i] + b[2 * i + 1]) for i in range(1, dim)]
    kind = "rect" if dim == 2 else "box"
    return HeatSource(name=_unique(names, "Heater"), kind=kind, position=tuple(c), size=(s,) * dim, power=100.0,
                      frame="wind")


def default_fan(design: Design | None, sim: Simulation | None, extent=None) -> Fan:
    """A new fan blowing downstream (+x): in a closed box a tenth of the way in from its left wall,
    in a tunnel one body length ahead of the body; half the body's height across."""
    b, dim = _context_box(design, sim, extent)
    names = {f.name for f in items_of(sim, "fans")}
    dom = domain_extent(design, sim, extent)
    closed = sim is not None and sim.cfd.domain_mode == "closed"
    cross = max([_span(b, i) for i in range(1, dim)] or [1.0])
    if closed and dom is not None:
        c = [dom[0] + 0.1 * (dom[1] - dom[0])] + [0.5 * (dom[2 * i] + dom[2 * i + 1]) for i in range(1, dim)]
        cross = min(_span(dom, i) for i in range(1, dim))
        d = 0.3 * cross
    else:
        c = [b[0] - max(_span(b, 0), cross)] + [0.5 * (b[2 * i] + b[2 * i + 1]) for i in range(1, dim)]
        d = 0.5 * cross
    return Fan(name=_unique(names, "Fan"), position=tuple(c), direction=(1.0,) + (0.0,) * (dim - 1),
               diameter=max(d, 1e-4), mode="velocity", value=2.0, frame="wind")


def default_port(design: Design | None, sim: Simulation | None, extent=None, kind: str = "jet") -> Port:
    """A new jet (low on the left wall, blowing +x) or vent (high on the right wall, drawing +x): a
    closed box gets the classic supply-and-extract pair; in a tunnel the jet blows downstream ahead of
    the body. 2 m/s through an opening a fifth of the cross-stream size."""
    b, dim = _context_box(design, sim, extent)
    names = {p.name for p in items_of(sim, "ports")}
    dom = domain_extent(design, sim, extent) or b
    up = dim - 1
    w = _span(dom, 0)
    hi_lo = (dom[2 * up], dom[2 * up + 1])
    c = [0.5 * (dom[2 * i] + dom[2 * i + 1]) for i in range(dim)]
    if kind == "vent":
        c[0] = dom[1] - 0.03 * w
        c[up] = hi_lo[0] + 0.75 * (hi_lo[1] - hi_lo[0])
    else:
        closed = sim is not None and sim.cfd.domain_mode == "closed"
        c[0] = dom[0] + 0.03 * w if closed else b[0] - 0.5 * max(_span(b, 0), 1e-6)
        if closed:
            c[up] = hi_lo[0] + 0.25 * (hi_lo[1] - hi_lo[0])
    size = 0.2 * min(_span(dom, i) for i in range(1, dim))
    base = "Vent" if kind == "vent" else "Jet"
    return Port(name=_unique(names, base), kind=kind, position=tuple(c), direction=(1.0,) + (0.0,) * (dim - 1),
                size=max(size, 1e-4), speed=2.0, frame="wind")


def new_item(kind: str, design: Design | None, sim: Simulation | None, extent=None):
    """A default item of ``kind`` (one of :data:`ITEM_LISTS`) for this case."""
    if kind == "emitters":
        return default_emitter(design, sim, extent)
    if kind == "heat_sources":
        return default_heat_source(design, sim, extent)
    if kind == "fans":
        return default_fan(design, sim, extent)
    if kind == "ports":
        # a closed box with a jet and no vent gets the vent next, so the pair balances
        jets = sum(1 for p in items_of(sim, "ports") if p.kind == "jet")
        vents = len(items_of(sim, "ports")) - jets
        closed = sim is not None and sim.cfd.domain_mode == "closed"
        return default_port(design, sim, extent, "vent" if closed and jets > vents else "jet")
    raise ValueError(f"unknown item kind {kind!r}")


def copy_item(obj: Any, names: set[str]):
    """A duplicate of an item, shifted by a fifth of its size so both stay visible, with a fresh name."""
    dup = type(obj).from_dict(obj.to_dict())
    dup.name = _unique(names, f"{obj.name} copy")
    pos = np.asarray(dup.position, dtype=float)
    dim = len(pos)
    shift = item_size(obj) * 0.25
    delta = np.zeros(dim)
    if dim:
        delta[-1] = shift                                 # up: the vertical axis is the last
    dup.position = _tup(pos + delta)
    if isinstance(dup, Emitter) and dup.end is not None:
        dup.end = _tup(vec(dup.end, dim) + delta)
    return dup


def item_size(obj: Any) -> float:
    """A characteristic size [m] of an item (for offsets and hit tolerances)."""
    if isinstance(obj, Fan):
        return float(obj.diameter)
    if isinstance(obj, Port):
        return float(obj.size)
    if isinstance(obj, HeatSource):
        return float(max(obj.size) if obj.size else 0.1)
    if isinstance(obj, Emitter):
        if obj.end is not None and has_extent(obj):
            d = vec(obj.end, len(obj.position)) - vec(obj.position, len(obj.position))
            return float(np.linalg.norm(d)) or 0.1
        return float(max(obj.size) if obj.size else 0.05)
    return 0.1


def summary_text(kind: str, obj: Any, dim: int) -> str:
    """One line for the item list: "Smoke · rake, 9 nozzles", "Heater · 100 W", "Fan · 2.0 m/s"."""
    off = "" if getattr(obj, "enabled", True) else " (off)"
    if kind == "emitters":
        what = EMITTER_KIND_TEXT.get(obj.kind, obj.kind).split(" (")[0].lower()
        if obj.kind == "rake":
            what += f", {int(obj.count)} nozzles"
        if obj.species == "particles":
            what += ", particles"
        return f"{obj.name} · {what}{off}"
    if kind == "heat_sources":
        return f"{obj.name} · {units.sig(obj.power, 3)} {'W' if dim == 3 else 'W/m'}{off}"
    if kind == "fans":
        unit = {"thrust": "N" if dim == 3 else "N/m", "pressure_jump": "Pa", "velocity": "m/s"}[obj.mode]
        return f"{obj.name} · {units.sig(obj.value, 3)} {unit}{off}"
    if kind == "ports":
        return f"{obj.name} · {'jet' if obj.kind == 'jet' else 'vent'} {units.sig(obj.speed, 3)} m/s{off}"
    return str(getattr(obj, "name", ""))


# ---------------------------------------------------------------------------
# heat transfer: fluid properties and the dimensionless groups
# ---------------------------------------------------------------------------

#: Fresh water, 0-100 °C: conductivity k [W/(m K)], specific heat cp [J/(kg K)], expansion
#: coefficient beta [1/K]. Saturated-liquid values of Incropera & DeWitt, Fundamentals of Heat
#: and Mass Transfer, Table A.6, interpolated to round temperatures (beta is negative below 4 °C,
#: where water contracts as it warms). The viscosity comes from the Fluid (Vogel/IAPWS fit).
_WATER = np.array([
    # T [C], k, cp, beta
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
#: Air: conductivity by Sutherland's law (White, Viscous Fluid Flow, 3rd ed., Table 1-3:
#: k0 = 0.0241 W/(m K) at 273.15 K, S = 194 K; within 2 % from 160 to 2000 K), cp of dry air
#: at 1 atm (1006 J/(kg K) within 0.3 % from -20 to 60 °C) and beta = 1/T of an ideal gas.
AIR_K0, AIR_T0, AIR_SK, AIR_CP = 0.0241, 273.15, 194.0, 1006.0


def air_conductivity(t_k: float) -> float:
    """Thermal conductivity of air [W/(m K)] at ``t_k`` kelvin (Sutherland's law)."""
    return AIR_K0 * (t_k / AIR_T0) ** 1.5 * (AIR_T0 + AIR_SK) / (t_k + AIR_SK)


def fluid_display_name(name: str) -> str:
    """A fluid name as the app shows it: "Air, 20 C" reads "Air, 20 °C" (ui_guidelines §2).

    The one rule lives in :func:`aerolume.core.fluid.display_name`, shared with Results, the reports
    and run messages; the stored name (the key saved projects and the Setup page read back) is
    unchanged."""
    from ...core.fluid import display_name
    return display_name(name)


def fluid_defaults(fluid: Fluid, t_ambient_c: float) -> tuple[dict[str, float] | None, str]:
    """(defaults {"prandtl", "conductivity", "cp", "beta"}, what they are) for a fluid at the ambient
    temperature, or (None, why) for a fluid without known thermal properties (a custom fluid).

    A thermal-property function of :mod:`aerolume.core.fluid` (``thermal_properties``) is used when
    the core provides one, so the solver and this page agree; otherwise air (Sutherland's law for
    the conductivity, ideal-gas expansion) and water (tabulated) are known here. The Prandtl number
    is computed from the fluid's own viscosity, so it matches the Reynolds number the run uses.
    """
    t_k = float(t_ambient_c) + 273.15
    with contextlib.suppress(Exception):
        from ...core import fluid as core_fluid
        fn = getattr(core_fluid, "thermal_properties", None)
        if callable(fn):
            got = fn(fluid, t_ambient_c)
            vals = {k: (got.get(k) if isinstance(got, dict) else getattr(got, k, None))
                    for k in ("prandtl", "conductivity", "cp", "beta")}
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals.values()):
                return ({k: float(v) for k, v in vals.items()},
                        f"{fluid_display_name(fluid.name)} (core fluid properties)")
    from ...core import fluid as core_fluid
    name = str(fluid.name).lower()
    if name.startswith("air"):
        # every property at the one temperature, so Pr = mu cp / k is the fluid's (0.71), not a
        # ratio of a viscosity at one temperature to a conductivity at another
        k = air_conductivity(t_k)
        mu = core_fluid.sutherland_viscosity(t_k)
        return ({"conductivity": k, "cp": AIR_CP, "beta": 1.0 / t_k, "prandtl": mu * AIR_CP / k},
                f"air at {t_ambient_c:g} °C (ideal gas, Sutherland's law)")
    if name.startswith("water"):
        tc = float(np.clip(t_ambient_c, _WATER[0, 0], _WATER[-1, 0]))
        k, cp, beta = (float(np.interp(tc, _WATER[:, 0], _WATER[:, j])) for j in (1, 2, 3))
        mu = core_fluid.water(tc).viscosity
        note = "" if tc == t_ambient_c else f" (table ends at {tc:g} °C)"
        return ({"conductivity": k, "cp": cp, "beta": beta, "prandtl": mu * cp / k},
                f"water at {tc:g} °C (Incropera, Table A.6){note}")
    return None, "no defaults for a custom fluid: type all four values"


#: "C" or "°C": the core spells it "C" today; the symbol is accepted so a later rename keeps working
_FLUID_T = r"^(?:Air|Water), (-?[\d.]+) °?C$"
_FLUID_ISA = r"^Air, ISA (sea level|[\d,.]+ m)$"


def fluid_temperature(fluid: Fluid) -> float | None:
    """The temperature [°C] a preset fluid was made at ("Air, 20 C", "Water, 20 C", an ISA altitude),
    or None for a custom fluid (aerolume.core.fluid writes the temperature into the name)."""
    name = str(fluid.name)
    m = re.match(_FLUID_T, name)
    if m:
        return float(m.group(1))
    m = re.match(_FLUID_ISA, name)
    if m:
        from ...core import fluid as core_fluid
        alt = 0.0 if m.group(1) == "sea level" else float(m.group(1)[:-2].replace(",", ""))
        with contextlib.suppress(ValueError):
            return core_fluid.isa(alt)[0] - 273.15
    return None


@dataclass
class ThermalProps:
    """The thermal properties a run uses: typed values where the user typed them, defaults otherwise."""
    prandtl: float
    conductivity: float
    cp: float
    beta: float
    density: float
    viscosity: float
    auto: dict[str, bool] = field(default_factory=dict)
    basis: str = ""
    known: bool = True                 # False: a custom fluid with an untyped property

    @property
    def nu(self) -> float:
        return self.viscosity / self.density

    @property
    def kappa(self) -> float:
        """Thermal diffusivity alpha = nu / Pr [m²/s]."""
        return self.nu / self.prandtl if self.prandtl > 0 else float("nan")


def thermal_props(sim: Simulation) -> ThermalProps:
    """The resolved properties of ``sim``'s fluid at its ambient temperature (see :func:`fluid_defaults`)."""
    th = sim.thermal
    fl = sim.conditions.fluid
    defaults, basis = fluid_defaults(fl, th.t_ambient)
    vals, auto, known = {}, {}, True
    for key in ("prandtl", "conductivity", "cp", "beta"):
        typed = getattr(th, key, None)
        if typed is not None:
            vals[key], auto[key] = float(typed), False
        elif defaults is not None:
            vals[key], auto[key] = float(defaults[key]), True
        else:
            # placeholders that keep the arithmetic finite; ``known`` makes the page say so
            vals[key] = {"prandtl": 0.71, "conductivity": 0.026, "cp": 1006.0,
                         "beta": 1.0 / (th.t_ambient + 273.15)}[key]
            auto[key], known = True, False
    return ThermalProps(density=float(fl.density), viscosity=float(fl.viscosity), auto=auto, basis=basis,
                        known=known, **vals)


def _surface(kind: str, size: Sequence[float], dim: int) -> tuple[float, float]:
    """(surface area [m² or m per metre of span], size L [m]) of a heat source's shape."""
    s = [abs(float(v)) for v in size] or [0.1]
    s += [s[-1]] * (3 - len(s))
    if kind == "disc":
        r = s[0]
        return 2 * math.pi * r, 2 * r
    if kind == "sphere":
        r = s[0]
        return 4 * math.pi * r * r, 2 * r
    if dim == 2:
        return 2 * (s[0] + s[1]), max(s[0], s[1])
    return 2 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]), max(s[:3])


@dataclass
class ThermalNumbers:
    """The dimensionless groups of a heat-transfer case, with what each rests on.

    ``length`` / ``delta_t`` / ``u_ref`` are the scales (m, K, m/s) and ``*_basis`` the rule that
    chose each, shown in the Estimate tooltips. ``ri`` is None without buoyancy or a forced flow."""
    length: float
    length_basis: str
    delta_t: float
    delta_t_basis: str
    u_ref: float
    u_basis: str
    re: float
    pr: float
    gr: float
    ra: float
    ri: float | None
    pe: float
    beta_dt: float
    regime: str
    closed: bool
    props: ThermalProps


def _forced_speed(sim: Simulation) -> tuple[float, str]:
    """(speed, what it is) of the flow that sweeps heat away: the free stream in a tunnel, else the
    fastest enabled jet or flow-speed fan (0 when nothing blows)."""
    if sim.cfd.domain_mode != "closed":
        return float(sim.conditions.speed), "free-stream speed"
    best, what = 0.0, ""
    for p in sim.ports:
        if p.enabled and p.kind == "jet" and abs(p.speed) > best:
            best, what = abs(float(p.speed)), f"jet “{p.name}”"
    for f in sim.fans:
        if f.enabled and f.mode == "velocity" and abs(f.value) > best:
            best, what = abs(float(f.value)), f"fan “{f.name}”"
    return best, what


def _flux_delta_t(q: float, ls: float, props: ThermalProps, g_beta: float, u: float) -> float:
    """Temperature rise [K] that a heat flux ``q`` [W/m²] off a surface of size ``ls`` [m] produces.

    q·L/(k·Nu) with Nu from laminar correlations: natural convection off a constant-flux vertical
    plate, Nu = 0.60 Ra*^(1/5) (Sparrow & Gregg 1956; Ra* = g β q L⁴ / (k ν α)), and forced flow
    over a flat plate, Nu = 0.664 Re^(1/2) Pr^(1/3) (Pohlhausen); the larger wins, and Nu ≥ 1
    (conduction). An order-of-magnitude scale for the Estimate, not a prediction: the run computes
    the real temperatures."""
    q, ls = abs(float(q)), max(float(ls), 1e-9)
    if q == 0:
        return 0.0
    k, nu, kappa = props.conductivity, props.nu, props.kappa
    nu_n = 0.60 * (g_beta * q * ls ** 4 / (k * nu * kappa)) ** 0.2 if g_beta > 0 else 0.0
    nu_f = 0.664 * math.sqrt(u * ls / nu) * props.prandtl ** (1 / 3) if u > 0 else 0.0
    return q * ls / (k * max(nu_n, nu_f, 1.0))


def thermal_numbers(design: Design | None, sim: Simulation | None) -> ThermalNumbers | None:
    """Re, Pr, Gr, Ra, Ri and Pe of a heat-transfer case (None when heat transfer is off).

    Scales (DESIGN.md 10.2, and the definitions of the benchmarks the solver is validated on):

    * temperature: the spread of the set temperatures (ambient, fixed-temperature bodies and walls,
      jets) - T_hot − T_cold, as de Vahl Davis defines the cavity's Rayleigh number - or, when larger,
      the rise a heat flux or a heater produces (:func:`_flux_delta_t`);
    * length: the domain height of a closed box heated only at its walls or by sources (the cavity
      convention), else the design's reference length;
    * velocity: the free stream in a tunnel; in a closed box the fastest jet or fan, else the
      free-fall velocity √(g β ΔT L) (Re = √Gr), or κ/L for a conduction-dominated box (Ra < 10³).
    """
    if design is None or sim is None or not sim.thermal.enabled:
        return None
    th, dim = sim.thermal, dimension_of(design)
    props = thermal_props(sim)
    closed = sim.cfd.domain_mode == "closed"
    g_beta = float(th.gravity) * props.beta if th.buoyancy else 0.0
    # --- length
    box = explicit_box(sim, dim)
    heated_body = any(getattr(b, "thermal", "adiabatic") != "adiabatic"
                      for b in (design.meshes if dim == 3 else design.bodies))
    if closed and box is not None and not heated_body:
        up = dim - 1
        length, length_basis = box[2 * up + 1] - box[2 * up], "height of the closed box"
    else:
        length, length_basis = float(design.length_ref or 0.0), "reference length of the design"
        if not length > 0:
            b = body_bounds(design) or box
            length = max(b[2 * i + 1] - b[2 * i] for i in range(dim)) if b else 1.0
            length_basis = "size of the domain"
    u_forced, u_what = _forced_speed(sim)
    # --- temperature
    temps = [float(th.t_ambient)]
    names = ["ambient"]
    for b in (design.meshes if dim == 3 else design.bodies):
        if b.thermal == "temperature":
            temps.append(float(b.thermal_value))
            names.append(f"body “{b.name}”")
    sides_active = set(SIDE_TEXT[dim]) - (set() if closed else {"left", "right"})
    for w in th.walls:
        if w.mode == "temperature" and w.side in sides_active:
            temps.append(float(w.value))
            names.append(f"{w.side} wall")
    for p in sim.ports:
        if p.enabled and p.kind == "jet" and p.temperature is not None:
            temps.append(float(p.temperature))
            names.append(f"jet “{p.name}”")
    spread = max(temps) - min(temps)
    dt, dt_basis = spread, ("spread of the set temperatures, " + f"{names[int(np.argmax(temps))]} to "
                            f"{names[int(np.argmin(temps))]}" if spread > 0 else "no temperature difference set")
    fluxes: list[tuple[float, float, str]] = []
    for b in (design.meshes if dim == 3 else design.bodies):
        if b.thermal == "heat_flux" and b.thermal_value:
            fluxes.append((float(b.thermal_value), float(design.length_ref or length), f"body “{b.name}”"))
    for w in th.walls:
        if w.mode == "heat_flux" and w.value and w.side in sides_active:
            fluxes.append((float(w.value), length, f"{w.side} wall"))
    for h in sim.heat_sources:
        if h.enabled and h.power:
            area, ls = _surface(h.kind, h.size, dim)
            fluxes.append((float(h.power) / max(area, 1e-12), ls, f"heater “{h.name}”"))
    for q, ls, what in fluxes:
        rise = _flux_delta_t(q, ls, props, g_beta, u_forced)
        if rise > dt:
            dt, dt_basis = rise, f"estimated rise at {what}: q·L / (k·Nu), Nu from a laminar correlation"
    nu = props.nu
    gr = g_beta * dt * length ** 3 / nu ** 2 if g_beta > 0 else 0.0
    ra = gr * props.prandtl
    # --- velocity
    if not closed or u_forced > 0:
        u, u_basis = u_forced, u_what
    elif g_beta > 0 and dt > 0 and ra >= 1e3:
        u, u_basis = math.sqrt(g_beta * dt * length), "free-fall velocity √(g β ΔT L)"
    else:
        u, u_basis = props.kappa / length, "conduction velocity κ/L"
    re = u * length / nu
    ri = g_beta * dt * length / u ** 2 if (g_beta > 0 and u_forced > 0) else None
    if not th.buoyancy:
        regime = "Forced (no buoyancy)" if u_forced > 0 else "Conduction (no buoyancy)"
    elif closed and u_forced == 0:
        regime = "Conduction" if ra < 1e3 else "Natural"
    elif ri is None:
        regime = "Natural"
    else:
        regime = "Forced" if ri < 0.1 else ("Mixed" if ri <= 10.0 else "Natural")
    tn = ThermalNumbers(length=length, length_basis=length_basis, delta_t=dt, delta_t_basis=dt_basis,
                        u_ref=u, u_basis=u_basis, re=re, pr=props.prandtl, gr=gr, ra=ra, ri=ri,
                        pe=re * props.prandtl, beta_dt=props.beta * dt, regime=regime, closed=closed,
                        props=props)
    return _planner_numbers(tn, design, sim)


def _planner_numbers(tn: ThermalNumbers, design: Design, sim: Simulation) -> ThermalNumbers:
    """``tn`` with the temperature scale and the groups on it taken from the planner
    (:func:`aerolume.simulation.physics.thermal_estimate`) when it can scale the setup.

    The Estimate rows, the Planner list and the run must quote one ΔT: a heater's rise used to be
    q·L/(k·Nu) here (58.3 K for the example room) while the planner, which sizes the run, takes
    the air flowing through the heater (4.1 K; measured 3.3 K). The page's own derivation above
    stays as the fallback when the planner is unavailable or cannot scale the setup.

    A setup with no temperature difference and no heat input keeps the page's ΔT of 0: the
    planner scales such a run with a placeholder 1 K ("nothing is heated or cooled"), and taking
    that over would silence the warning that heat transfer is on with nothing to transfer."""
    if tn.delta_t == 0:
        return tn
    try:
        from ...simulation.physics import thermal_estimate
        sc = thermal_estimate(design, sim)
    except Exception:  # noqa: BLE001 - an advisory estimate: the page's own numbers stand
        return tn
    if sc is None or not getattr(sc, "thermal", False) or not (math.isfinite(sc.delta_t) and sc.delta_t >= 0):
        return tn
    length = sc.length_ra if math.isfinite(sc.length_ra) and sc.length_ra > 0 else tn.length
    import dataclasses
    return dataclasses.replace(
        tn, length=length, length_basis=sc.length_ra_basis or tn.length_basis, delta_t=float(sc.delta_t),
        delta_t_basis=sc.delta_t_basis or tn.delta_t_basis,
        u_ref=float(sc.speed) if math.isfinite(sc.speed) else tn.u_ref, u_basis=sc.speed_basis or tn.u_basis,
        re=float(sc.re) if math.isfinite(sc.re) else tn.re, pr=float(sc.pr) if math.isfinite(sc.pr) else tn.pr,
        gr=float(sc.gr), ra=float(sc.ra), ri=sc.ri_flow if sc.buoyancy else None,
        pe=float(sc.pe) if math.isfinite(sc.pe) else tn.pe, beta_dt=tn.props.beta * float(sc.delta_t),
        regime=sc.regime or tn.regime)


THERMAL_ROW_KEYS = ("Temperature scale", "Prandtl number", "Grashof number", "Rayleigh number",
                    "Richardson number", "Péclet number", "Convection")


def thermal_rows(tn: ThermalNumbers | None) -> list[tuple[str, str, str]]:
    """Estimate rows (label, value, tooltip) of a heat-transfer case; [] when there is none."""
    if tn is None:
        return []
    length = units.fmt(tn.length, "length", display_unit("length"), 3)
    rows = [("Temperature scale", f"{units.sig(tn.delta_t, 3)} K", f"ΔT: {tn.delta_t_basis}."),
            ("Prandtl number", fmt_number(tn.pr), "ν/α: the typed value." if not tn.props.auto.get("prandtl", True)
             else f"ν/α of {tn.props.basis or 'the fluid'}.")]
    if tn.gr > 0:
        rows.append(("Grashof number", fmt_number(tn.gr), f"g β ΔT L³ / ν² with L = {length} ({tn.length_basis})."))
        rows.append(("Rayleigh number", fmt_number(tn.ra),
                     ("Gr·Pr. Below about 10³ heat moves mainly by conduction; above about 10⁸ the thermal "
                      "boundary layers become unsteady.")))
    if tn.ri is not None:
        rows.append(("Richardson number", fmt_number(tn.ri),
                     "Gr / Re²: buoyancy against the forced flow. Below 0.1 forced, above 10 natural convection."))
    rows.append(("Péclet number", fmt_number(tn.pe), (f"Re·Pr with U = {units.sig(tn.u_ref, 3)} m/s "
                                                     f"({tn.u_basis}): heat carried by the flow against heat "
                                                     "conducted.")))
    rows.append(("Convection", tn.regime, ("From the Richardson number (tunnel) or the Rayleigh number "
                                           "(closed box).")))
    return rows


def port_flow(p: Port, dim: int) -> float:
    """Volume flow of a jet or vent: m³/s in 3D (a round opening), m²/s per metre of span in 2D."""
    s = abs(float(p.size))
    area = math.pi * (0.5 * s) ** 2 if dim == 3 else s
    return abs(float(p.speed)) * area


def _inside(p: np.ndarray, box: Sequence[float], dim: int, tol: float = 0.0) -> bool:
    return all(box[2 * i] - tol <= p[i] <= box[2 * i + 1] + tol for i in range(dim))


def physics_warnings(design: Design | None, sim: Simulation | None, extent: Sequence[float] | None = None
                     ) -> tuple[list[str], list[str]]:
    """(warnings, notes) about the placed physics, in the Planner list's words.

    Checks what a run would silently get wrong: a closed box that does not balance (jets supply
    more or less than vents remove) or that has nothing to drive any flow; items outside the
    domain; more smoke colours than dye channels; a temperature difference that stretches the
    Boussinesq approximation; a Rayleigh number in the transitional range; heat transfer that is
    on without anything heated; a custom fluid without thermal properties.
    """
    warns: list[str] = []
    notes: list[str] = []
    if design is None or sim is None or sim.analysis != "cfd":
        return warns, notes
    dim = dimension_of(design)
    closed = sim.cfd.domain_mode == "closed"
    if closed and explicit_box(sim, dim) is None:
        notes.append("Closed box: the planner sizes the box around the body; set Domain ▸ Extent to fix its walls.")
    box = domain_extent(design, sim, extent)
    jets = [p for p in sim.ports if p.enabled and p.kind == "jet"]
    vents = [p for p in sim.ports if p.enabled and p.kind == "vent"]
    if closed:
        q_in = sum(port_flow(p, dim) for p in jets)
        q_out = sum(port_flow(p, dim) for p in vents)
        unit = "m³/s" if dim == 3 else "m²/s per metre of span"
        if (q_in or q_out) and abs(q_in - q_out) > 0.02 * max(q_in, q_out):
            warns.append(f"The jets supply {units.sig(q_in, 3)} {unit} and the vents remove {units.sig(q_out, 3)}: "
                         "a closed box must balance. Match the speeds or the openings; the solver otherwise "
                         "corrects the imbalance uniformly.")
        heated = sim.thermal.enabled and sim.thermal.buoyancy and (
            any(h.enabled and h.power for h in sim.heat_sources)
            or any(w.mode != "adiabatic" and w.value for w in sim.thermal.walls)
            or any(b.thermal != "adiabatic" for b in (design.meshes if dim == 3 else design.bodies)))
        if not (heated or jets or any(f.enabled for f in sim.fans)):
            warns.append("Nothing drives the flow in this closed box: add a fan, a jet or a heat source, heat a "
                          "wall with buoyancy on, or switch Domain ▸ Type to Wind tunnel.")
    if box is not None:
        tol = 1e-9 * max(box[2 * i + 1] - box[2 * i] for i in range(dim))
        for kind in ITEM_LISTS:
            for obj in items_of(sim, kind):
                if not getattr(obj, "enabled", True):
                    continue
                p = to_wind(anchor(obj, dim), design, sim, getattr(obj, "frame", "body"))[0]
                if not _inside(p, box, dim, tol):
                    warns.append(f"{ITEM_NOUN[kind].capitalize()} “{obj.name}” is outside the domain: it has no "
                                 "effect. Move it inside the dashed box.")
    colours = {e.color.lower() for e in sim.emitters if e.enabled and e.species == "dye"}
    if len(colours) > MAX_DYE_CHANNELS:
        warns.append(f"{len(colours)} smoke colours, {MAX_DYE_CHANNELS} smoke channels: emitters of the fifth "
                     "colour and beyond share a channel with the first.")
    if dim == 2 and any(f.enabled and f.swirl for f in sim.fans):
        notes.append("Fan swirl acts in 3D only; a 2D fan pushes along its direction.")
    if not sim.thermal.enabled:
        # a heater, a hot jet or a heated body is inert without the energy equation (the API refuses
        # such a run; the page says it where the switch is)
        heat = [h.name for h in sim.heat_sources if h.enabled and h.power]
        heat += [p.name for p in sim.ports if p.enabled and p.kind == "jet" and p.temperature is not None]
        heat += [b.name for b in (design.meshes if dim == 3 else design.bodies) if b.thermal != "adiabatic"]
        if heat:
            names = ", ".join(f"“{n}”" for n in heat[:3]) + (" …" if len(heat) > 3 else "")
            warns.append(f"Heat transfer is off, so {names} add{'s' if len(heat) == 1 else ''} no heat: "
                         "switch on Heat transfer ▸ Solve for temperature.")
    tn = thermal_numbers(design, sim)
    if tn is not None:
        t_fluid = fluid_temperature(sim.conditions.fluid)
        if t_fluid is not None and abs(t_fluid - sim.thermal.t_ambient) > 2.0:
            what = sim.conditions.fluid.name.split(",")[0].lower()
            warns.append(f"The fluid is {what} at {t_fluid:.3g} °C but the ambient temperature is "
                         f"{sim.thermal.t_ambient:.3g} °C: set Flow ▸ Fluid to that temperature, so the density "
                         "and viscosity of the run match it.")
        if not tn.props.known:
            warns.append("Custom fluid: type the Prandtl number, conductivity, specific heat and expansion "
                         "coefficient under Heat transfer; placeholder values of air are used until then.")
        if tn.delta_t == 0:
            warns.append("Heat transfer is on but nothing is heated or cooled: every body, wall and jet is at the "
                         "ambient temperature and there is no heat source.")
        if sim.thermal.buoyancy and tn.beta_dt > 0.1:
            warns.append(f"β·ΔT = {tn.beta_dt:.2f} (ΔT {units.sig(tn.delta_t, 3)} K): above about 0.1 the Boussinesq "
                         "approximation (constant density except in the buoyancy force) loses accuracy.")
        if tn.gr > 0 and tn.ra > 1e8:
            notes.append(f"Ra = {fmt_number(tn.ra)}: above about 10⁸ the thermal boundary layers become unsteady "
                         "and then turbulent; resolve them with a fine grid or treat the result as indicative.")
        if tn.ri is not None and tn.ri < 1e-3 and sim.thermal.buoyancy:
            notes.append(f"Ri = {fmt_number(tn.ri)}: buoyancy is negligible against the free stream here.")
    return warns, notes


# ---------------------------------------------------------------------------
# what the case previews draw (plain dicts, wind frame, metres)
# ---------------------------------------------------------------------------

def emitter_nozzles(e: Emitter, dim: int) -> np.ndarray:
    """(N, dim) nozzle centres of a point / rake emitter in its own frame (design frame)."""
    p = vec(e.position, dim)
    if e.kind == "point":
        return p[None]
    if e.kind == "rake" and e.end is not None:
        n = max(int(e.count), 1)
        t = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        return p[None] + t[:, None] * (vec(e.end, dim) - p)[None]
    return np.zeros((0, dim))


def thermal_text(mode: str, value: float) -> str:
    """A thermal condition as the previews print it: "30 °C" or "500 W/m²"."""
    return f"{value:.4g} °C" if mode == "temperature" else f"{value:.4g} W/m²"


def _thermal_body_specs(design: Design, sim: Simulation, dim: int) -> list[dict]:
    """Heated or cooled bodies (heat transfer on): ``type="body_thermal"``, the wind-frame outline
    (2D) or bounding-box corners (3D) as ``points`` and the condition. They are drawn, never
    dragged (the body is edited on the Geometry page): the canvases keep them apart from the items."""
    if not sim.thermal.enabled:
        return []
    out = []
    parts = design.meshes if dim == 3 else design.bodies
    for i, b in enumerate(parts):
        if b.thermal == "adiabatic":
            continue
        if dim == 3:
            if not len(b.vertices):
                continue
            x0, x1, y0, y1, z0, z1 = b.bounds()
            pts = np.array([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)
        else:
            pts = np.asarray(b.points, dtype=float).reshape(-1, 2)
            if not len(pts):
                continue
        hot = float(b.thermal_value) - sim.thermal.t_ambient if b.thermal == "temperature" else float(b.thermal_value)
        out.append({"key": ("bodies", i), "type": "body_thermal", "name": b.name, "mode": b.thermal,
                    "value": float(b.thermal_value), "sign": 1 if hot > 0 else (-1 if hot < 0 else 0),
                    "points": to_wind(pts, design, sim, "body"), "handles": {}, "enabled": True,
                    "selected": False, "dim": dim})
    return out


def scene_specs(design: Design | None, sim: Simulation | None, *, selected: tuple | None = None,
                extent: Sequence[float] | None = None) -> list[dict]:
    """What the case preview draws for the placed physics, in the wind frame (metres).

    One dict per item: ``key`` (list name, index), ``type`` ("emitter", "heat_source", "fan",
    "port"), ``name``, ``enabled``, ``selected``, ``handles`` {name: point} and the shape: emitters
    ``shape`` + ``points`` (nozzles) / ``segment`` (line, rake) / ``size`` + ``normal`` (rect, disc,
    plane) and ``colour``; heat sources ``shape``, ``size``, ``power``; fans ``direction``,
    ``diameter``; ports ``kind``, ``direction``, ``size``. With heat transfer on, each heated or
    cooled body adds a ``type="body_thermal"`` entry (``points``, ``mode``, ``value``, ``sign``; no
    handles: it is drawn and labelled, not dragged). A final ``type="domain"`` entry carries the
    domain ``box`` and whether it is ``explicit`` (draggable walls). The canvases paint these and
    report drags of a handle back by key; they know nothing of the model.
    """
    out: list[dict] = []
    if design is None or sim is None:
        return out
    dim = dimension_of(design)
    for kind in ITEM_LISTS:
        for i, obj in enumerate(items_of(sim, kind)):
            frame = getattr(obj, "frame", "body")

            def w(pt, _frame=frame):
                return to_wind(pt, design, sim, _frame)[0]

            def wd(v, _frame=frame):
                return to_wind(v, design, sim, _frame, direction=True)[0]

            spec: dict = {"key": (kind, i), "type": kind[:-1] if kind != "heat_sources" else "heat_source",
                          "name": obj.name, "enabled": bool(getattr(obj, "enabled", True)),
                          "selected": selected == (kind, i), "dim": dim}
            if kind == "emitters":
                spec["type"] = "emitter"
                spec["shape"] = obj.kind
                spec["colour"] = obj.color
                spec["species"] = obj.species
                p = w(vec(obj.position, dim))
                handles = {"centre": w(anchor(obj, dim))}
                if has_extent(obj):
                    a, b = p, w(vec(obj.end if obj.end is not None else obj.position, dim))
                    spec["segment"] = (a, b)
                    handles.update(start=a, end=b)
                nozzles = emitter_nozzles(obj, dim)
                spec["points"] = to_wind(nozzles, design, sim, frame) if len(nozzles) else np.zeros((0, dim))
                spec["size"] = tuple(float(v) for v in obj.size) if obj.size else ()
                spec["normal"] = wd(unit_vec(obj.normal, dim)) if obj.normal is not None else wd(unit_vec(None, dim))
                spec["centre"] = handles["centre"]
                spec["handles"] = handles
            elif kind == "heat_sources":
                spec["shape"] = obj.kind
                spec["centre"] = w(vec(obj.position, dim))
                spec["size"] = tuple(float(v) for v in obj.size) if obj.size else (0.1,)
                spec["power"] = float(obj.power)
                spec["handles"] = {"centre": spec["centre"]}
            elif kind == "fans":
                spec["centre"] = w(vec(obj.position, dim))
                spec["direction"] = wd(unit_vec(obj.direction, dim))
                spec["diameter"] = float(obj.diameter)
                spec["handles"] = {"centre": spec["centre"]}
            else:
                spec["kind"] = obj.kind
                spec["centre"] = w(vec(obj.position, dim))
                spec["direction"] = wd(unit_vec(obj.direction, dim))
                spec["size"] = float(obj.size)
                spec["handles"] = {"centre": spec["centre"]}
            out.append(spec)
    out += _thermal_body_specs(design, sim, dim)
    box = domain_extent(design, sim, extent)
    if box is not None:
        closed = sim.cfd.domain_mode == "closed"
        walls = []
        if sim.thermal.enabled:
            # the heated and cooled sides (a tunnel's inlet and outlet carry no wall condition)
            active = set(SIDE_TEXT[dim]) - (set() if closed else {"left", "right"})
            for w in sim.thermal.walls:
                if w.side in active and w.mode != "adiabatic":
                    hot = w.value - sim.thermal.t_ambient if w.mode == "temperature" else w.value
                    walls.append({"side": w.side, "mode": w.mode, "value": float(w.value),
                                  "sign": 1 if hot > 0 else (-1 if hot < 0 else 0)})
        out.append({"key": ("domain",), "type": "domain", "box": tuple(box),
                    "explicit": explicit_box(sim, dim) is not None, "closed": closed, "walls": walls,
                    "selected": selected == ("domain",), "dim": dim, "enabled": True, "name": "Domain"})
    return out


def design_frame_specs(design: Design | None, sim: Simulation | None, *, selected: tuple | None = None
                       ) -> list[dict]:
    """:func:`scene_specs` in the *design* frame, for the Geometry page (which shows the shape as
    drawn, not pitched): every point and direction turned back by the attitude. The domain is kept
    only when it is a fixed box or a closed one and the case is not pitched or yawed (a tunnel's
    planned extent belongs to Setup; a turned box is no longer axis-aligned here)."""
    specs = scene_specs(design, sim, selected=selected)
    if not specs:
        return specs
    alpha, yaw, _ref = attitude(design, sim)
    turned = abs(alpha) > 1e-12 or abs(yaw) > 1e-12
    out = []
    for s in specs:
        if s["type"] == "domain":
            if not turned and (s.get("explicit") or s.get("closed")):
                out.append(s)
            continue
        if turned:
            s = dict(s)

            def back(p, _d=False):
                return from_wind(p, design, sim, "body", direction=_d)

            for k in ("centre",):
                if k in s:
                    s[k] = back(s[k])[0]
            for k in ("direction", "normal"):
                if k in s:
                    s[k] = back(s[k], True)[0]
            if "segment" in s:
                s["segment"] = (back(s["segment"][0])[0], back(s["segment"][1])[0])
            if len(s.get("points", ())):
                s["points"] = back(s["points"])
            s["handles"] = {h: back(p)[0] for h, p in s.get("handles", {}).items()}
        out.append(s)
    return out


def apply_drag(design: Design, sim: Simulation, key: tuple, handle: str, point: Sequence[float], *,
               extent: Sequence[float] | None = None) -> bool:
    """Apply a drag from the case preview: move item ``key``'s ``handle`` to wind-frame ``point``, or
    move a domain wall (``key == ("domain",)``, handle "x0", "x1", "y0", ..., or a corner "x0y1").

    A drag on the domain fixes its extent (Auto off) at the box drawn - ``extent``, the planned
    domain, while it was automatic - with the dragged walls moved. Returns whether anything changed."""
    dim = dimension_of(design)
    if key and key[0] == "domain":
        box = list(domain_extent(design, sim, extent) or ())
        if not box:
            return False
        p = vec(point, dim)
        axes = {"x": 0, "y": 1, "z": 2}
        changed = False
        for k in range(0, len(handle), 2):
            ax, side = axes.get(handle[k]), handle[k + 1:k + 2]
            if ax is None or ax >= dim or side not in ("0", "1"):
                continue
            j = 2 * ax + int(side)
            other = box[2 * ax + (1 - int(side))]
            gap = 1e-3 * max(abs(box[2 * ax + 1] - box[2 * ax]), 1e-6)
            v = float(p[ax])
            v = min(v, other - gap) if side == "0" else max(v, other + gap)
            if v != box[j]:
                box[j], changed = v, True
        if changed or explicit_box(sim, dim) is None:
            sim.cfd.domain_box = tuple(float(v) for v in box)
            return True
        return False
    kind, idx = key[0], int(key[1])
    objs = items_of(sim, kind)
    if not 0 <= idx < len(objs):
        return False
    obj = objs[idx]
    p_design = from_wind(point, design, sim, getattr(obj, "frame", "body"))[0]
    before = obj.to_dict()
    move_item(obj, handle, p_design, dim)
    return obj.to_dict() != before


# ---------------------------------------------------------------------------
# Qt: small editors
# ---------------------------------------------------------------------------

def auto_toggle(tooltip: str = AUTO_TIP) -> QToolButton:
    """A checkable "Auto" switch at the end of a row (the References pattern of the Geometry page)."""
    b = QToolButton()
    b.setText("Auto")
    b.setCheckable(True)
    b.setProperty("variant", "toggle")
    b.setToolTip(tooltip)
    b.setAccessibleName("Automatic value")
    b.setFixedSize(ROW_BUTTON_W, 24)
    return b


def _combo(items, name: str = "") -> QComboBox:
    c = QComboBox()
    for it in items:
        c.addItem(it[1], it[0])
        if len(it) > 2:
            c.setItemData(c.count() - 1, it[2], Qt.ItemDataRole.ToolTipRole)
    if name:
        c.setAccessibleName(name)
    return c


def _set_combo(c: QComboBox, key: Any) -> None:
    with QSignalBlocker(c):
        c.setCurrentIndex(max(c.findData(key), 0))


def _spin(minimum: float, maximum: float, step: float, decimals: int, suffix: str = "", *,
          special: str | None = None, name: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimum, maximum)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    s.setMinimumWidth(64)
    s.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if special:
        s.setSpecialValueText(special)
    if name:
        s.setAccessibleName(name)
    return s


def compact_box(box: QDoubleSpinBox) -> QDoubleSpinBox:
    """A number box without arrow buttons, for rows that hold two or three boxes (a vector, a domain
    range, a condition beside its mode). The arrows took 18 px of a box about 90 px wide and clipped
    values such as a 45.7 mm cavity edge ("0.0457" needs 39 px; 35 were left). The keyboard arrows,
    Page Up / Down and typing still change the value."""
    box.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
    # the theme pads every spin box for its arrows (18 px); without arrows that is lost room
    box.setStyleSheet("QAbstractSpinBox { padding-right: 4px; }")
    return box


def range_pair(lo: QWidget, hi: QWidget) -> QWidget:
    """"[lo] to [hi]" in one editor cell, the two boxes sharing the room equally.

    ``hbox`` gives its first widget the spare room, so the second keeps its size hint and the first
    shrinks below its own: a domain edge at -1.525 m then read "-1.5" (clipped, not rounded) beside a
    full "4.000". Equal stretch with an Ignored horizontal policy keeps both boxes the same width."""
    from PySide6.QtWidgets import QHBoxLayout
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    for box in (lo, hi):
        box.setSizePolicy(QSizePolicy.Policy.Ignored, box.sizePolicy().verticalPolicy())
        box.setMinimumWidth(56)
    h.addWidget(lo, 1)
    h.addWidget(label("to", "muted"), 0)
    h.addWidget(hi, 1)
    return w


def describe_parts(editor: QWidget, text: str) -> None:
    """Give every focusable part of ``editor`` without a tooltip of its own ``text`` as its accessible
    description (WCAG 3.3.2 / 4.1.2).

    ``PropertyGroup.add_row`` puts a row's explanation on its label and on the editor it is handed.
    For a composite editor (a mode selector beside its value, a number box beside an Auto switch)
    that editor is the container, which never takes focus: a mouse still gets the tooltip (Qt
    passes the hover up to the container), but a screen reader reads the focused part's own
    description, which was empty - measured over the four starter cases, 12 to 37 focusable parts
    per case had none ("Left (x min) wall: condition" was announced without what it sets). An
    existing description ("in m", from ``tie_label``) is kept in front."""
    if not text:
        return
    for w in [editor, *editor.findChildren(QWidget)]:
        if (w.focusPolicy() == Qt.FocusPolicy.NoFocus or w.objectName() == "qt_spinbox_lineedit"
                or w.toolTip()):
            continue
        have = w.accessibleDescription()
        if text not in have:
            w.setAccessibleDescription(f"{have.rstrip('.')}. {text}" if have else text)


def add_row(group, text: str, editor: QWidget, *, unit: str | None = None, tooltip: str | None = None):
    """``group.add_row`` (a :class:`~aerolume.app.widgets.common.PropertyGroup` row) plus
    :func:`describe_parts`, so the row's explanation reaches whichever part holds the focus."""
    row = group.add_row(text, editor, unit=unit, tooltip=tooltip)
    describe_parts(editor, tooltip or "")
    return row


def _article(noun: str) -> str:
    """The indefinite article before ``noun``: an emitter, a heat source, a fan, a jet or vent."""
    return "an" if noun[:1].lower() in "aeiou" else "a"


def _silent(w: QWidget, value: Any) -> None:
    with QSignalBlocker(w):
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QuantitySpinBox):
            w.set_si_value(float(value))
        elif isinstance(w, QLineEdit):
            set_line_text(w, str(value))     # the start of a long name in view, not its tail
        else:
            w.setValue(value)


class VectorField(QWidget):
    """Two or three quantity boxes for a point or a size (x, y[, z]); SI in and out.

    ``valueChanged(tuple)`` fires with the whole vector when one component is committed. The
    unit goes in the property row's unit column, so the boxes show bare numbers and the three
    still fit the inspector's editor column."""

    valueChanged = Signal(tuple)

    def __init__(self, dim: int, *, kind: str = "length", unit: str | None = None, minimum: float = -1e5,
                 maximum: float = 1e5, step: float = 0.01, name: str = "Position",
                 axis_names: Sequence[str] = ("x", "y", "z"), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.boxes: list[QuantitySpinBox] = []
        from PySide6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        for i in range(3):
            b = QuantitySpinBox(kind, unit or display_unit(kind), minimum=minimum, maximum=maximum, step=step,
                                show_suffix=False)
            b.setMinimumWidth(48)
            # no arrows: three boxes share the editor column, and the digits need the room (values
            # are typed, nudged with the keyboard arrows or dragged in the preview)
            compact_box(b)
            b.setAccessibleName(f"{name}, {axis_names[i]}")
            b.siValueChanged.connect(self._emit)
            lay.addWidget(b, 1)
            self.boxes.append(b)
        self._dim = 3
        self.set_dimension(dim)

    def set_dimension(self, dim: int, tips: Sequence[str] | None = None) -> None:
        self._dim = int(dim)
        for i, b in enumerate(self.boxes):
            b.setVisible(i < self._dim)
            if tips is not None and i < len(tips):
                b.setToolTip(tips[i])

    def set_range(self, minimum: float, maximum: float) -> None:
        for b in self.boxes:
            b.set_si_range(minimum, maximum)

    def unit(self) -> str:
        return self.boxes[0].unit

    def value(self) -> tuple[float, ...]:
        """The vector, SI: the components not typed keep their loaded value unrounded."""
        return tuple(float(b.exact_si_value()) for b in self.boxes[: self._dim])

    def set_value(self, v: Sequence[float]) -> None:
        a = vec(v, self._dim)
        for b, x in zip(self.boxes, a, strict=False):
            b.set_si_value(float(x))

    def _emit(self, _v: float) -> None:
        self.valueChanged.emit(self.value())


DIRECTION_PRESETS_3D = (("+x", (1.0, 0.0, 0.0), "+x (downstream)"), ("-x", (-1.0, 0.0, 0.0), "−x (upstream)"),
                        ("+y", (0.0, 1.0, 0.0), "+y"), ("-y", (0.0, -1.0, 0.0), "−y"),
                        ("+z", (0.0, 0.0, 1.0), "+z (up)"), ("-z", (0.0, 0.0, -1.0), "−z (down)"))


class DirectionField(QWidget):
    """A direction: in 2D an angle from +x (counter-clockwise, degrees); in 3D an axis preset or a
    custom vector (normalised when committed). ``valueChanged(tuple)`` gives the unit vector."""

    valueChanged = Signal(tuple)

    def __init__(self, dim: int, name: str = "Direction", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.angle = QuantitySpinBox("angle", "deg", minimum=-180.0, maximum=180.0, step=5.0, decimals=1)
        self.angle.setWrapping(True)
        self.angle.setAccessibleName(f"{name}, angle from +x")
        self.angle.setToolTip("Angle from +x (downstream), counter-clockwise: 90° points up, 180° upstream.")
        self.angle.siValueChanged.connect(lambda _v: self.valueChanged.emit(self.value()))
        lay.addWidget(self.angle, 1)
        self.preset = QComboBox()
        self.preset.setAccessibleName(f"{name}, axis")
        for key, _v, text in DIRECTION_PRESETS_3D:
            self.preset.addItem(text, key)
        self.preset.addItem("Custom", "custom")
        self.preset.currentIndexChanged.connect(self._preset_changed)
        lay.addWidget(self.preset, 1)
        self.comps: list[QDoubleSpinBox] = []
        for ax in "xyz":
            s = _spin(-1.0, 1.0, 0.1, 2, name=f"{name}, {ax} component")
            s.setMinimumWidth(44)
            s.editingFinished.connect(self._custom_edited)
            lay.addWidget(s, 1)
            self.comps.append(s)
        self._dim = dim
        self._vec = vec((1.0,), 3)
        self.set_dimension(dim)

    def set_dimension(self, dim: int) -> None:
        self._dim = int(dim)
        self.angle.setVisible(self._dim == 2)
        self.preset.setVisible(self._dim == 3)
        self._show_custom()

    def set_closed(self, closed: bool) -> None:
        """Word the axes for a closed box, which has no stream: "+x" rather than "+x (downstream)"
        (up and down stay: gravity still defines them)."""
        for i, (_key, _v, text) in enumerate(DIRECTION_PRESETS_3D):
            self.preset.setItemText(i, text.split(" (")[0] if closed and "stream)" in text else text)
        self.angle.setToolTip("Angle from +x, counter-clockwise: 90° points up, 180° along −x." if closed else
                              "Angle from +x (downstream), counter-clockwise: 90° points up, 180° upstream.")

    def _show_custom(self) -> None:
        custom = self._dim == 3 and self.preset.currentData() == "custom"
        for s in self.comps:
            s.setVisible(custom)

    def value(self) -> tuple[float, ...]:
        if self._dim == 2:
            a = math.radians(self.angle.si_value())
            return (math.cos(a), math.sin(a))
        return _tup(self._vec)

    def set_value(self, v: Sequence[float]) -> None:
        u = unit_vec(v, self._dim)
        if self._dim == 2:
            self.angle.set_si_value(math.degrees(math.atan2(u[1], u[0])))
            return
        self._vec = u
        key = next((k for k, pv, _t in DIRECTION_PRESETS_3D if np.allclose(pv, u, atol=1e-6)), "custom")
        with QSignalBlocker(self.preset):
            self.preset.setCurrentIndex(self.preset.findData(key))
        for s, x in zip(self.comps, u, strict=True):
            _silent(s, round(float(x), 2))
        self._show_custom()

    def _preset_changed(self, _i: int) -> None:
        key = self.preset.currentData()
        hit = next((pv for k, pv, _t in DIRECTION_PRESETS_3D if k == key), None)
        if hit is not None:
            self._vec = np.asarray(hit, dtype=float)
            for s, x in zip(self.comps, self._vec, strict=True):
                _silent(s, float(x))
            self.valueChanged.emit(self.value())
        self._show_custom()

    def _custom_edited(self) -> None:
        raw = np.array([s.value() for s in self.comps])
        if float(np.linalg.norm(raw)) < 1e-9:
            self.set_value(self._vec)               # a zero vector has no direction: keep the last one
            return
        self._vec = raw / np.linalg.norm(raw)
        self.valueChanged.emit(self.value())


class ColourCombo(QComboBox):
    """The smoke colours (one per dye channel), each item with its swatch; ``colour()`` is "#rrggbb"."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtGui import QColor, QIcon, QPixmap
        self.setAccessibleName("Smoke colour")
        for hexv, text in SMOKE_COLOURS:
            pm = QPixmap(12, 12)
            pm.fill(QColor(hexv))
            self.addItem(QIcon(pm), text, hexv)

    def colour(self) -> str:
        return str(self.currentData())

    def set_colour(self, hexv: str) -> None:
        i = self.findData(str(hexv).lower())
        if i < 0:                                   # a colour set by a script: keep it as a named entry
            from PySide6.QtGui import QColor, QIcon, QPixmap
            pm = QPixmap(12, 12)
            pm.fill(QColor(hexv))
            self.addItem(QIcon(pm), str(hexv), str(hexv).lower())
            i = self.count() - 1
        with QSignalBlocker(self):
            self.setCurrentIndex(i)


class ItemList(QListWidget):
    """The list of one kind of placed item: one line each ("Fan · 2.0 m/s"), single selection,
    a height that shows up to :attr:`MAX_ROWS` lines before it scrolls."""

    MAX_ROWS = 4

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("flat", "true")
        self.setAccessibleName(name)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setUniformItemSizes(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_lines(self, lines: list[str], tips: list[str] | None = None) -> None:
        with QSignalBlocker(self):
            cur = self.currentRow()
            self.clear()
            for i, t in enumerate(lines):
                it = QListWidgetItem(t)
                it.setToolTip(tips[i] if tips and i < len(tips) else t)
                self.addItem(it)
            if lines:
                self.setCurrentRow(min(max(cur, 0), len(lines) - 1))
        rows = max(1, min(len(lines), self.MAX_ROWS))
        h = self.sizeHintForRow(0) if lines else 22
        self.setFixedHeight(rows * max(h, 22) + 2 * self.frameWidth() + 2)


class _SectionBase(QObject):
    """Shared plumbing: the simulation and design under edit, a loading guard, change signals."""

    edited = Signal()                # the simulation changed (the page saves and replans)
    designEdited = Signal()          # a body's thermal condition changed (it lives on the Design)

    def __init__(self, group, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.group = group
        self.sim: Simulation | None = None
        self.design: Design | None = None
        self.extent: tuple | None = None
        self._loading = False

    @property
    def dim(self) -> int:
        return dimension_of(self.design)

    def set_context(self, sim: Simulation | None, design: Design | None, extent=None) -> None:
        self.sim, self.design, self.extent = sim, design, (tuple(extent) if extent is not None else None)
        self.reload()

    def reload(self) -> None:          # pragma: no cover - every section overrides it
        raise NotImplementedError

    def _live(self) -> bool:
        return not self._loading and self.sim is not None

    def _changed(self) -> None:
        if self._live():
            self.edited.emit()


# ---------------------------------------------------------------------------
# placed items: emitters, heat sources, fans, jets and vents
# ---------------------------------------------------------------------------

class PlacedItemsSection(_SectionBase):
    """One kind of placed item in a property group: the list, Add / Duplicate / Remove, and the
    selected item's fields (rows shown or hidden for its shape and the case's dimension).

    ``selectionChanged(kind, index)`` (-1: none) lets the page highlight the item in the preview;
    :meth:`select` does the reverse when the user clicks an item there. ``autoPlaceRequested`` asks
    the page to place an emitter upstream of the body by the solver's automatic rake rule (the page
    knows the planned body box and cell size)."""

    selectionChanged = Signal(str, int)
    autoPlaceRequested = Signal(int)

    def __init__(self, group, kind: str, parent: QObject | None = None) -> None:
        super().__init__(group, parent)
        if kind not in ITEM_LISTS:
            raise ValueError(f"unknown item kind {kind!r}")
        self.kind = kind
        self.rows: dict[str, Any] = {}
        noun = ITEM_NOUN[kind]
        self.list = ItemList(ITEM_TITLE[kind])
        self.list.setObjectName(f"{kind}_list")
        self.list.currentRowChanged.connect(self._row_changed)
        self.empty = label(EMPTY_TEXT[kind] if isinstance(EMPTY_TEXT[kind], str) else "", "muted", wrap=True)
        self.empty.setObjectName(f"{kind}_empty")
        self.rows["list"] = add_row(group, ITEM_TITLE[kind], hbox(self.list, self.empty, stretch=0),
                                    tooltip=f"The {ITEM_TITLE[kind].lower()} of this simulation; select one to "
                                            "edit it here or drag it in the case preview.")
        self.add_button = QPushButton("Add")
        self.add_button.setObjectName(f"{kind}_add")
        self.add_button.setToolTip(f"Add {_article(noun)} {noun} at a default place for this case")
        self.dup_button = QPushButton("Duplicate")
        self.dup_button.setObjectName(f"{kind}_duplicate")
        self.dup_button.setToolTip(f"Copy the selected {noun}, shifted a little")
        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName(f"{kind}_remove")
        self.remove_button.setToolTip(f"Remove the selected {noun} from this simulation")
        for b in (self.add_button, self.dup_button, self.remove_button):
            # never below the button's own text: a flat 56 px let a narrow inspector clip "Duplicate"
            # to "uplicate" (critic round 3.4, Setup at 1280 x 800 beside the Assistant)
            b.setMinimumWidth(max(56, b.sizeHint().width()))
        self.add_button.clicked.connect(self.add)
        self.dup_button.clicked.connect(self.duplicate)
        self.remove_button.clicked.connect(self.remove)
        self.rows["buttons"] = add_row(group, "", hbox(self.add_button, self.dup_button, self.remove_button,
                                                       stretch=None))
        self._build_fields()
        self.reload()

    # -- fields ---------------------------------------------------------------
    def _row(self, key: str, text: str, editor: QWidget, *, unit: str | None = None, tooltip: str = "") -> None:
        editor.setObjectName(f"{self.kind}_{key}")
        if not editor.accessibleName():
            editor.setAccessibleName(text)
        self.rows[key] = add_row(self.group, text, editor, unit=unit, tooltip=tooltip)

    def _build_fields(self) -> None:
        k = self.kind
        self.name_edit = keep_start_visible(QLineEdit())
        self.name_edit.editingFinished.connect(self._name_edited)
        self._row("name", "Name", self.name_edit,
                  tooltip="Shown in the case preview, in Messages and in the results.")
        self.enabled = QCheckBox("Active in the run")
        self.enabled.toggled.connect(lambda on: self._set("enabled", bool(on)))
        self._row("enabled", "Use", self.enabled, tooltip=f"Off keeps the {ITEM_NOUN[k]} in the setup without "
                                                        "it acting in the run.")
        lu = display_unit("length")
        if k == "emitters":
            self.species = _combo(SPECIES_ITEMS, "Tracer")
            self.species.currentIndexChanged.connect(lambda _i: self._set("species", self.species.currentData()))
            self._row("species", "Tracer", self.species, tooltip="Smoke (dye) or tracer particles.")
            self.shape = _combo([(s, EMITTER_KIND_TEXT[s], EMITTER_KIND_TIP[s]) for s in EMITTER_KINDS_3D], "Shape")
            self.shape.currentIndexChanged.connect(lambda _i: self._shape_edited(self.shape.currentData()))
            self._row("shape", "Shape", self.shape, tooltip="How the tracer is released: one nozzle, a slot, a "
                                                            "row of nozzles or an area.")
        if k == "heat_sources":
            self.shape = _combo([(s, HEAT_KIND_TEXT[s]) for s in ("rect", "disc", "box", "sphere")], "Shape")
            self.shape.currentIndexChanged.connect(lambda _i: self._shape_edited(self.shape.currentData()))
            self._row("shape", "Shape", self.shape, tooltip="The heated region; heat is released evenly over it "
                                                            "with a smooth edge a cell or two wide.")
        if k == "ports":
            self.port_kind = _combo(PORT_ITEMS, "Type")
            self.port_kind.currentIndexChanged.connect(lambda _i: self._port_kind_edited())
            self._row("port_kind", "Type", self.port_kind,
                      tooltip="Jet: blows fluid in. Vent: draws it out. In a closed box the two must balance.")
        self.position = VectorField(self.dim, unit=lu, name="Position")
        self.position.valueChanged.connect(lambda v: self._set_point("start" if self._extent_kind() else "centre",
                                                                     v))
        self._row("position", "Position", self.position, unit=lu, tooltip=POSITION_TIP)
        if k == "emitters":
            self.end = VectorField(self.dim, unit=lu, name="End")
            self.end.valueChanged.connect(lambda v: self._set_point("end", v))
            self._row("end", "End", self.end, unit=lu, tooltip="Where the line or rake ends, metres.")
            self.count = QSpinBox()
            self.count.setRange(1, 400)
            self.count.setKeyboardTracking(False)
            self.count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.count.valueChanged.connect(lambda v: self._set("count", int(v)))
            self._row("count", "Nozzles", self.count, tooltip="Nozzles along the rake, ends included. About six "
                                                              "nozzle radii apart keeps the streaks distinct.")
            self.size = VectorField(self.dim, unit=lu, minimum=1e-5, name="Size")
            self.size.valueChanged.connect(lambda v: self._set("size", tuple(abs(x) for x in v)))
            self._row("size", "Size", self.size, unit=lu, tooltip="Width × height (× depth in 3D), metres.")
            self.radius = QuantitySpinBox("length", lu, minimum=0.0, maximum=1e4, step=0.001, show_suffix=False)
            self.radius.setSpecialValueText("Auto")
            self.radius.siValueChanged.connect(lambda v: self._set_radius(float(v)))
            self._row("radius", "Nozzle radius", self.radius, unit=lu,
                      tooltip="Radius of each nozzle (disc: of the disc). Auto: about two grid cells.")
            self.normal = DirectionField(3, name="Normal")
            self.normal.valueChanged.connect(lambda v: self._set("normal", tuple(v)))
            self._row("normal", "Facing", self.normal, tooltip="The direction the disc or plane faces (its normal).")
            self.rate = _spin(0.0, 1e6, 0.1, 2, name="Release rate")
            self.rate.valueChanged.connect(lambda v: self._set("rate", float(v)))
            self._row("rate", "Release rate", self.rate,
                      tooltip="Smoke: the concentration released, 0 to 1. Particles: released per second.")
            self.colour = ColourCombo()
            self.colour.currentIndexChanged.connect(lambda _i: self._set("color", self.colour.colour()))
            self._row("colour", "Colour", self.colour, tooltip="Emitters of one colour share a smoke channel; "
                                                               f"the solver carries {MAX_DYE_CHANNELS}.")
            self.pulse = _spin(0.0, 1e5, 0.1, 3, special="Continuous", name="Pulse period")
            self.pulse.valueChanged.connect(lambda v: self._set("pulse_period", float(v)))
            self._row("pulse", "Pulse period", self.pulse, unit="s",
                      tooltip="Release in pulses this many seconds apart (timelines); Continuous: no pulses.")
            self.duty = _spin(1.0, 99.0, 5.0, 0, name="Pulse on")
            self.duty.valueChanged.connect(lambda v: self._set("pulse_duty", float(v) / 100.0))
            self._row("duty", "Pulse on", self.duty, unit="%", tooltip="Share of each pulse period spent releasing.")
            self.start = _spin(0.0, 1e6, 0.1, 3, name="Start")
            self.start.valueChanged.connect(lambda v: self._set("start", float(v)))
            self.stop = _spin(0.0, 1e6, 0.1, 3, special="Never", name="Stop")
            self.stop.valueChanged.connect(lambda v: self._set("stop", float(v) if v > 0 else None))
            self._row("window", "Release from", range_pair(self.start, self.stop), unit="s",
                      tooltip="Seconds of simulated flow time; Never: until the run ends.")
        if k == "heat_sources":
            self.size = VectorField(self.dim, unit=lu, minimum=1e-5, name="Size")
            self.size.valueChanged.connect(lambda v: self._set("size", tuple(abs(x) for x in v)))
            self._row("size", "Size", self.size, unit=lu, tooltip="Width × height (× depth in 3D), metres.")
            self.radius = QuantitySpinBox("length", lu, minimum=1e-5, maximum=1e4, step=0.005, show_suffix=False)
            self.radius.siValueChanged.connect(lambda v: self._set("size", (float(v),)))
            self._row("radius", "Radius", self.radius, unit=lu,
                      tooltip="Radius of the disc (2D) or sphere (3D); the heat is released evenly inside it.")
            self.power = _spin(-1e9, 1e9, 10.0, 2, name="Power")
            self.power.valueChanged.connect(lambda v: self._set("power", float(v)))
            self._row("power", "Power", self.power, unit="W/m",
                      tooltip="Heat released into the fluid; negative cools. In 2D, per metre of span.")
        if k == "fans":
            self.direction = DirectionField(self.dim, name="Blows toward")
            self.direction.valueChanged.connect(lambda v: self._set("direction", tuple(v)))
            self._row("direction", "Blows toward", self.direction, tooltip="The direction the fan pushes the fluid.")
            self.diameter = QuantitySpinBox("length", lu, minimum=1e-5, maximum=1e4, step=0.01, show_suffix=False)
            self.diameter.siValueChanged.connect(lambda v: self._set("diameter", float(v)))
            self._row("diameter", "Diameter", self.diameter, unit=lu,
                      tooltip="Diameter of the actuator disc; in 2D, the width of the strip.")
            self.thickness = QuantitySpinBox("length", lu, minimum=0.0, maximum=1e4, step=0.005, show_suffix=False)
            self.thickness.setSpecialValueText("Auto")
            self.thickness.siValueChanged.connect(lambda v: self._set("thickness", float(v) if v > 0 else None))
            self._row("thickness", "Thickness", self.thickness, unit=lu,
                      tooltip="Depth of the actuator disc along its direction. Auto: an eighth of the diameter.")
            self.mode = _combo(FAN_MODE_ITEMS, "Fan mode")
            self.mode.currentIndexChanged.connect(lambda _i: self._mode_edited())
            self._row("mode", "Set by", self.mode, tooltip="What the fan is told to deliver.")
            self.value = _spin(-1e7, 1e7, 0.5, 3, name="Fan value")
            self.value.valueChanged.connect(lambda v: self._set("value", float(v)))
            self._row("value", "Value", self.value, unit="m/s")
            self.swirl = _spin(-5.0, 5.0, 0.1, 2, name="Swirl")
            self.swirl.valueChanged.connect(lambda v: self._set("swirl", float(v)))
            self._row("swirl", "Swirl", self.swirl, tooltip="Tangential to axial speed ratio imparted, positive by "
                                                            "the right-hand rule about the direction (3D only).")
        if k == "ports":
            self.direction = DirectionField(self.dim, name="Flow direction")
            self.direction.valueChanged.connect(lambda v: self._set("direction", tuple(v)))
            self._row("direction", "Flow direction", self.direction,
                      tooltip="Jet: the direction it blows. Vent: the direction fluid moves as it leaves.")
            self.opening = QuantitySpinBox("length", lu, minimum=1e-5, maximum=1e4, step=0.005, show_suffix=False)
            self.opening.siValueChanged.connect(lambda v: self._set("size", float(v)))
            self._row("opening", "Opening", self.opening, unit=lu,
                      tooltip="Width of the opening (2D) or its diameter (3D).")
            self.speed = _spin(0.0, 1e4, 0.5, 3, name="Speed")
            self.speed.valueChanged.connect(lambda v: self._set("speed", float(v)))
            self._row("speed", "Speed", self.speed, unit="m/s", tooltip="Mean speed through the opening.")
            self.temperature = QuantitySpinBox("temperature", "C", minimum=-273.0, maximum=3000.0, step=1.0,
                                               decimals=1, show_suffix=False)
            self.temp_auto = auto_toggle("Auto: the jet blows fluid at the ambient temperature.")
            self.temp_auto.setAccessibleName("Jet temperature: automatic")
            self.temperature.siValueChanged.connect(lambda v: self._set("temperature", float(v)))
            self.temp_auto.clicked.connect(self._temp_auto)
            self._row("temperature", "Temperature", hbox(self.temperature, self.temp_auto), unit="°C",
                      tooltip="Temperature of the fluid the jet blows (heat transfer on).")
            self.dye = QCheckBox("Carries smoke")
            self.dye.toggled.connect(lambda on: self._set("dye", bool(on)))
            self._row("dye", "Smoke", self.dye, tooltip="The jet releases smoke, so its plume shows.")
        self.frame = _combo(FRAME_ITEMS, "Attached to")
        self.frame.currentIndexChanged.connect(lambda _i: self._frame_edited())
        self._row("frame", "Attached to", self.frame, tooltip=FRAME_TIP)
        if k == "emitters":
            self.place_button = QPushButton("Place upstream")
            self.place_button.setObjectName("emitters_place")
            self.place_button.setToolTip("Move this emitter to the solver's automatic rake: just upstream of the "
                                         "body, across it, at 40 % of its height in 3D.")
            self.place_button.clicked.connect(lambda: self.autoPlaceRequested.emit(self.current_index()))
            self.rows["place"] = add_row(self.group, "", hbox(self.place_button, stretch=None))
        self.note = label("", "muted", wrap=True)
        self.note.setObjectName(f"{k}_note")
        self.rows["note"] = add_row(self.group, "", self.note)

    # -- state ------------------------------------------------------------------
    def items(self) -> list:
        return items_of(self.sim, self.kind)

    def current_index(self) -> int:
        i = self.list.currentRow()
        return i if 0 <= i < len(self.items()) else -1

    def current(self):
        i = self.current_index()
        return self.items()[i] if i >= 0 else None

    def _extent_kind(self) -> bool:
        obj = self.current()
        return isinstance(obj, Emitter) and has_extent(obj)

    def reload(self) -> None:
        """Show the simulation's items and the selected one's fields (no edits are echoed back)."""
        self._loading = True
        try:
            objs = self.items()
            dim = self.dim
            self.list.set_lines([summary_text(self.kind, o, dim) for o in objs])
            has = bool(objs)
            self.list.setVisible(has)
            text = EMPTY_TEXT[self.kind]
            if isinstance(text, dict):
                closed = self.sim is not None and self.sim.cfd.domain_mode == "closed"
                text = text["closed" if closed else dim]
            self.empty.setText(text)
            self.empty.setVisible(not has)
            self.dup_button.setEnabled(has)
            self.remove_button.setEnabled(has)
            self.add_button.setEnabled(self.sim is not None)
            self._load_current()
        finally:
            self._loading = False

    def refresh_lines(self) -> None:
        """Update the list's text after an edit (keeps the selection and the fields)."""
        with QSignalBlocker(self.list):
            for i, o in enumerate(self.items()):
                it = self.list.item(i)
                if it is not None:
                    t = summary_text(self.kind, o, self.dim)
                    it.setText(t)
                    it.setToolTip(t)

    def select(self, index: int) -> None:
        """Select item ``index`` (from the preview); emits nothing."""
        if 0 <= index < len(self.items()):
            with QSignalBlocker(self.list):
                self.list.setCurrentRow(index)
            self._loading = True
            try:
                self._load_current()
            finally:
                self._loading = False

    def _row_changed(self, _row: int) -> None:
        self._loading = True
        try:
            self._load_current()
        finally:
            self._loading = False
        self.selectionChanged.emit(self.kind, self.current_index())

    def _show(self, key: str, on: bool) -> None:
        r = self.rows.get(key)
        if r is not None:
            r.set_visible(on)

    def _load_current(self) -> None:
        obj = self.current()
        dim = self.dim
        field_keys = [k for k in self.rows if k not in ("list", "buttons")]
        for k in field_keys:
            self._show(k, obj is not None)
        if obj is None:
            self.note.setText("")
            return
        closed = self.sim is not None and self.sim.cfd.domain_mode == "closed"
        tips = (AXIS_TIP_CLOSED if closed else AXIS_TIP)[dim]
        lu = display_unit("length")
        _silent(self.name_edit, obj.name)
        _silent(self.enabled, obj.enabled)
        for f in (getattr(self, "direction", None), getattr(self, "normal", None)):
            if f is not None:
                f.set_closed(closed)
        self.frame.setItemText(self.frame.findData("wind"), FRAME_WIND_CLOSED if closed else FRAME_ITEMS[0][1])
        _set_combo(self.frame, obj.frame)
        self.position.set_dimension(dim, tips)
        if self.kind == "emitters":
            allowed = EMITTER_KINDS_3D if dim == 3 else EMITTER_KINDS_2D
            with QSignalBlocker(self.shape):
                self.shape.clear()
                for s in allowed:
                    self.shape.addItem(EMITTER_KIND_TEXT[s] if dim == 3 or s != "rect" else "Rectangle", s)
                    self.shape.setItemData(self.shape.count() - 1, EMITTER_KIND_TIP[s], Qt.ItemDataRole.ToolTipRole)
                if dim == 3:
                    self.shape.setItemText(allowed.index("rect"), "Box")
            _set_combo(self.shape, obj.kind)
            _set_combo(self.species, obj.species)
            line = has_extent(obj)
            self.rows["position"].set_label("Start" if line else "Position")
            self.position.set_value(obj.position)
            self.end.set_dimension(dim, tips)
            self.end.set_value(obj.end if obj.end is not None else obj.position)
            self._show("end", line)
            self._show("count", obj.kind == "rake")
            _silent(self.count, int(obj.count))
            area = obj.kind in ("rect", "plane")
            self._show("size", area)
            if area:
                n = dim if obj.kind == "rect" else 2
                self.size.set_dimension(n, ("width", "height", "depth") if n == 3 else ("width", "height"))
                self.size.set_value(obj.size or (item_size(obj),) * n)
            self.rows["radius"].set_label("Disc radius" if obj.kind == "disc" else "Nozzle radius")
            r = float(obj.size[0]) if obj.kind == "disc" and obj.size else float(obj.radius)
            self.radius.setSpecialValueText("" if obj.kind == "disc" else "Auto")
            self.radius.set_si_value(r)
            self._show("radius", obj.kind in ("point", "rake", "disc"))
            self._show("normal", dim == 3 and obj.kind in ("disc", "plane"))
            self.normal.set_value(obj.normal if obj.normal is not None else (1.0, 0.0, 0.0))
            _silent(self.rate, float(obj.rate))
            self.rows["rate"].set_label("Release rate")
            self.rate.setSuffix(" /s" if obj.species == "particles" else "")
            self.rate.setRange(0.0, 1e6 if obj.species == "particles" else 1.0)
            _silent(self.rate, float(obj.rate))
            self.colour.set_colour(obj.color)
            self._show("colour", obj.species == "dye")
            _silent(self.pulse, float(obj.pulse_period))
            _silent(self.duty, 100.0 * float(obj.pulse_duty))
            self._show("duty", obj.pulse_period > 0)
            _silent(self.start, float(obj.start))
            _silent(self.stop, float(obj.stop or 0.0))
            # the automatic rake stands upstream of a body in a tunnel; a closed box (or a case without a
            # body) has no upstream to put it in
            bodies = self.design is not None and (self.design.kind == "wing" or bool(
                self.design.meshes if dim == 3 else self.design.bodies))
            self._show("place", bodies and not closed)
            self._set_note(self._emitter_note(obj))
        elif self.kind == "heat_sources":
            allowed = HEAT_KINDS[dim]
            with QSignalBlocker(self.shape):
                self.shape.clear()
                for s in allowed:
                    self.shape.addItem(HEAT_KIND_TEXT[s], s)
            kind = obj.kind if obj.kind in allowed else allowed[0]
            _set_combo(self.shape, kind)
            self.position.set_value(obj.position)
            round_ = kind in ("disc", "sphere")
            self._show("size", not round_)
            self._show("radius", round_)
            self.size.set_dimension(dim, ("width", "height", "depth") if dim == 3 else ("width", "height"))
            self.size.set_value(obj.size if len(obj.size) >= dim else (obj.size[0] if obj.size else 0.1,) * dim)
            self.radius.set_si_value(float(obj.size[0]) if obj.size else 0.05)
            self.rows["power"].unit.setText("W" if dim == 3 else "W/m")
            _silent(self.power, float(obj.power))
            self._set_note(self._heat_note(obj))
        elif self.kind == "fans":
            self.position.set_value(obj.position)
            self.direction.set_dimension(dim)
            self.direction.set_value(obj.direction)
            self.rows["diameter"].set_label("Diameter" if dim == 3 else "Width")
            self.diameter.set_si_value(float(obj.diameter))
            self.thickness.set_si_value(float(obj.thickness or 0.0))
            _set_combo(self.mode, obj.mode)
            self._fan_value_unit(obj.mode)
            _silent(self.value, float(obj.value))
            _silent(self.swirl, float(obj.swirl))
            self._show("swirl", dim == 3)
            self._set_note(self._fan_note(obj))
        elif self.kind == "ports":
            _set_combo(self.port_kind, obj.kind)
            self.position.set_value(obj.position)
            self.direction.set_dimension(dim)
            self.direction.set_value(obj.direction)
            self.opening.set_si_value(float(obj.size))
            _silent(self.speed, float(obj.speed))
            jet = obj.kind == "jet"
            heat = self.sim is not None and self.sim.thermal.enabled
            self._show("temperature", jet and heat)
            auto = obj.temperature is None
            with QSignalBlocker(self.temp_auto):
                self.temp_auto.setChecked(auto)
            amb = self.sim.thermal.t_ambient if self.sim is not None else 20.0
            self.temperature.set_si_value(float(amb if auto else obj.temperature))
            self._show("dye", jet)
            _silent(self.dye, bool(obj.dye))
            self._set_note(self._port_note(obj))
        del lu

    # -- notes -------------------------------------------------------------------
    def _set_note(self, text: str) -> None:
        """The muted line under the fields; its row is hidden when there is nothing to say, so an
        empty note does not leave a blank row before the next group."""
        self.note.setText(text)
        self._show("note", bool(text))

    def _emitter_note(self, e: Emitter) -> str:
        dim = self.dim
        if e.kind == "rake" and e.end is not None and e.count > 1:
            pitch = float(np.linalg.norm(vec(e.end, dim) - vec(e.position, dim))) / (e.count - 1)
            return f"{e.count} nozzles {units.fmt(pitch, 'length', display_unit('length'), 3)} apart."
        if e.kind in ("disc", "plane", "rect") and dim == 2 and e.kind != "rect":
            return "Discs and planes release in 3D only."
        return ""

    def _heat_note(self, h: HeatSource) -> str:
        area, _ls = _surface(h.kind, h.size, self.dim)
        return f"Mean flux through its surface ≈ {units.sig(h.power / max(area, 1e-12), 3)} W/m²."

    def _fan_note(self, f: Fan) -> str:
        dim = self.dim
        area = math.pi * (0.5 * f.diameter) ** 2 if dim == 3 else float(f.diameter)
        rho = self.sim.conditions.fluid.density if self.sim is not None else 1.2
        if f.mode == "thrust":
            dp = f.value / max(area, 1e-12)
            return f"Pressure rise ΔP = T / A = {units.sig(dp, 3)} Pa (actuator disc)."
        if f.mode == "pressure_jump":
            u = math.sqrt(max(f.value, 0.0) / (0.5 * rho)) if rho > 0 else float("nan")
            return f"In still fluid this rise gives about {units.sig(u, 3)} m/s through the disc (ΔP = ½ρu²)."
        q = f.value * area
        return f"Volume flow ≈ {units.sig(q, 3)} {'m³/s' if dim == 3 else 'm²/s per metre of span'}."

    def _port_note(self, p: Port) -> str:
        q = port_flow(p, self.dim)
        unit = "m³/s" if self.dim == 3 else "m²/s per metre of span"
        return f"{'Supplies' if p.kind == 'jet' else 'Removes'} {units.sig(q, 3)} {unit}."

    def _fan_value_unit(self, mode: str) -> None:
        unit = {"thrust": "N" if self.dim == 3 else "N/m", "pressure_jump": "Pa", "velocity": "m/s"}[mode]
        self.rows["value"].unit.setText(unit)
        self.rows["value"].set_label({k: t for k, t, _ in FAN_MODE_ITEMS}[mode])
        tip = {k: tip for k, _t, tip in FAN_MODE_ITEMS}[mode]
        self.value.setToolTip(tip)
        self.rows["value"].label.setToolTip(tip)

    # -- edits ------------------------------------------------------------------------
    def _after_edit(self, reload_fields: bool = False) -> None:
        if reload_fields:
            self._loading = True
            try:
                self._load_current()
            finally:
                self._loading = False
        else:
            obj = self.current()
            if obj is not None:
                with contextlib.suppress(Exception):
                    self._set_note({"emitters": self._emitter_note, "heat_sources": self._heat_note,
                                    "fans": self._fan_note, "ports": self._port_note}[self.kind](obj))
        self.refresh_lines()
        self.edited.emit()

    def _set(self, attr: str, value: Any) -> None:
        obj = self.current()
        if not self._live() or obj is None:
            return
        if getattr(obj, attr, None) == value:
            return
        setattr(obj, attr, value)
        self._after_edit(reload_fields=attr in ("species", "pulse_period"))

    def _set_point(self, handle: str, v: Sequence[float]) -> None:
        obj = self.current()
        if not self._live() or obj is None:
            return
        if handle == "centre" and isinstance(obj, Emitter):
            obj.position = _tup(vec(v, self.dim))           # a point or area emitter: its centre
        else:
            move_item(obj, handle if handle != "centre" else "start", v, self.dim)
        self._after_edit()

    def _set_radius(self, r: float) -> None:
        obj = self.current()
        if not self._live() or obj is None:
            return
        if obj.kind == "disc":
            obj.size = (max(r, 1e-6),)
        else:
            obj.radius = max(r, 0.0)
        self._after_edit()

    def _name_edited(self) -> None:
        obj = self.current()
        if not self._live() or obj is None:
            return
        name = self.name_edit.text().strip()
        if not name:
            set_line_text(self.name_edit, obj.name)
            return
        if name != obj.name:
            obj.name = name
            self._after_edit()

    def _shape_edited(self, shape: str) -> None:
        obj = self.current()
        if not self._live() or obj is None or shape is None or obj.kind == shape:
            return
        dim = self.dim
        size = item_size(obj)
        if isinstance(obj, Emitter):
            if shape in ("line", "rake") and obj.end is None:
                d = np.zeros(dim)
                d[-1] = size                                  # a line up the height
                obj.end = _tup(vec(obj.position, dim) + d)
            if shape not in ("line", "rake") and obj.end is not None and has_extent(obj):
                obj.position = _tup(anchor(obj, dim))          # an area or point takes the line's middle
            if shape == "rect" and len(obj.size) != dim:
                obj.size = (0.3 * size,) * dim
            if shape == "plane" and len(obj.size) != 2:
                obj.size = (size, size)
            if shape == "disc":
                obj.size = (0.5 * size,)
            if shape in ("disc", "plane") and obj.normal is None:
                obj.normal = (1.0, 0.0, 0.0)
            obj.kind = shape
        else:
            obj.size = (0.5 * size,) if shape in ("disc", "sphere") else (size,) * dim
            obj.kind = shape
        self._after_edit(reload_fields=True)

    def _mode_edited(self) -> None:
        obj = self.current()
        mode = self.mode.currentData()
        if not self._live() or obj is None or obj.mode == mode:
            return
        obj.mode = mode
        self._fan_value_unit(mode)
        self._after_edit()

    def _port_kind_edited(self) -> None:
        obj = self.current()
        kind = self.port_kind.currentData()
        if not self._live() or obj is None or obj.kind == kind:
            return
        obj.kind = kind
        if obj.name.startswith(("Jet", "Vent")):
            names = {p.name for p in self.items() if p is not obj}
            obj.name = _unique(names, "Jet" if kind == "jet" else "Vent")
        self._after_edit(reload_fields=True)

    def _frame_edited(self) -> None:
        """Switching what an item is attached to keeps it where it is drawn now (the wind frame)."""
        obj = self.current()
        frame = self.frame.currentData()
        if not self._live() or obj is None or obj.frame == frame:
            return
        dim = self.dim
        pw = to_wind(vec(obj.position, dim), self.design, self.sim, obj.frame)[0]
        ew = (to_wind(vec(obj.end, dim), self.design, self.sim, obj.frame)[0]
              if isinstance(obj, Emitter) and obj.end is not None else None)
        dirs = {}
        for attr in ("direction", "normal"):
            v = getattr(obj, attr, None)
            if v is not None:
                dirs[attr] = to_wind(vec(v, dim), self.design, self.sim, obj.frame, direction=True)[0]
        obj.frame = frame
        obj.position = _tup(from_wind(pw, self.design, self.sim, frame)[0])
        if ew is not None:
            obj.end = _tup(from_wind(ew, self.design, self.sim, frame)[0])
        for attr, v in dirs.items():
            setattr(obj, attr, _tup(from_wind(v, self.design, self.sim, frame, direction=True)[0]))
        self._after_edit(reload_fields=True)

    def _temp_auto(self, on: bool) -> None:
        obj = self.current()
        if not self._live() or obj is None:
            return
        obj.temperature = None if on else float(self.temperature.si_value())
        self._after_edit()

    # -- list actions -----------------------------------------------------------------
    def _announce(self, index: int) -> None:
        """Select row ``index`` (or none), show its fields and tell the page (the preview highlights it)."""
        with QSignalBlocker(self.list):
            self.list.setCurrentRow(index)
        self._loading = True
        try:
            self._load_current()
        finally:
            self._loading = False
        self.selectionChanged.emit(self.kind, self.current_index())

    def add(self) -> Any:
        """Add a default item for this case and select it."""
        if self.sim is None:
            return None
        obj = new_item(self.kind, self.design, self.sim, self.extent)
        self.items().append(obj)
        self.reload()
        self._announce(len(self.items()) - 1)
        self.edited.emit()
        return obj

    def append(self, obj: Any) -> None:
        """Add a prepared item (an emitter converted from the automatic rake) and select it."""
        if self.sim is None:
            return
        self.items().append(obj)
        self.reload()
        self._announce(len(self.items()) - 1)
        self.edited.emit()

    def duplicate(self) -> Any:
        obj = self.current()
        if self.sim is None or obj is None:
            return None
        dup = copy_item(obj, {o.name for o in self.items()})
        i = self.current_index() + 1
        self.items().insert(i, dup)
        self.reload()
        self._announce(i)
        self.edited.emit()
        return dup

    def remove(self) -> None:
        i = self.current_index()
        if self.sim is None or i < 0:
            return
        del self.items()[i]
        self.reload()
        self._announce(min(i, len(self.items()) - 1) if self.items() else -1)
        self.edited.emit()


# ---------------------------------------------------------------------------
# heat transfer
# ---------------------------------------------------------------------------

class ThermalSection(_SectionBase):
    """Heat transfer in a property group: the switch, the ambient temperature, buoyancy, the fluid's
    thermal properties (Auto shows the default of this fluid), the domain walls side by side and the
    surface condition of each body (a selector, so a mesh of fifty parts stays one row)."""

    def __init__(self, group, parent: QObject | None = None) -> None:
        super().__init__(group, parent)
        self.rows: dict[str, Any] = {}
        self.enabled = QCheckBox("Solve for temperature")
        self.enabled.setObjectName("heat_enabled")
        self.enabled.toggled.connect(self._enabled_edited)
        self.rows["enabled"] = add_row(group, "Heat transfer", self.enabled, tooltip=HEAT_TIP)
        self.ambient = QuantitySpinBox("temperature", "C", minimum=-273.0, maximum=3000.0, step=1.0, decimals=1)
        self.ambient.setObjectName("heat_ambient")
        self.ambient.setAccessibleName("Ambient temperature")
        self.ambient.siValueChanged.connect(lambda v: self._set("t_ambient", float(v), props=True))
        self.rows["ambient"] = add_row(group, "Ambient", self.ambient, tooltip=AMBIENT_TIP)
        self.buoyancy = QCheckBox("Warm fluid rises")
        self.buoyancy.setObjectName("heat_buoyancy")
        self.buoyancy.toggled.connect(lambda on: self._set("buoyancy", bool(on)))
        self.rows["buoyancy"] = add_row(group, "Buoyancy", self.buoyancy, tooltip=BUOYANCY_TIP)
        self.gravity = _spin(0.0, 1000.0, 0.1, 3, " m/s²", name="Gravity")
        self.gravity.setObjectName("heat_gravity")
        self.gravity.valueChanged.connect(lambda v: self._set("gravity", float(v)))
        self.rows["gravity"] = add_row(group, "Gravity", self.gravity,
                                       tooltip="Acceleration of gravity; 9.807 m/s² on Earth.")
        self.props: dict[str, tuple[QDoubleSpinBox, QToolButton]] = {}
        # units as suffixes, not in the unit column: the four rows keep one editor width
        for key, text, unit, rng, dec in (("beta", "Expansion β", " 1/K", (-1.0, 1.0), 6),
                                          ("prandtl", "Prandtl number", "", (1e-4, 1e5), 4),
                                          ("conductivity", "Conductivity", " W/(m·K)", (1e-6, 1e4), 4),
                                          ("cp", "Specific heat", " J/(kg·K)", (1.0, 1e6), 0)):
            box = _spin(rng[0], rng[1], 10.0 ** -max(dec - 2, 0), dec, unit, name=text)
            box.setObjectName(f"heat_{key}")
            auto = auto_toggle()
            auto.setObjectName(f"heat_{key}_auto")
            auto.setAccessibleName(f"{text}: automatic")
            box.valueChanged.connect(lambda v, k=key: self._prop_typed(k, float(v)))
            auto.clicked.connect(lambda on, k=key: self._prop_auto(k, bool(on)))
            self.props[key] = (box, auto)
            self.rows[key] = add_row(group, text, hbox(box, auto), tooltip=PROPERTY_TIP[key])
        self.basis = label("", "muted", wrap=True)
        self.basis.setObjectName("heat_basis")
        self.rows["basis"] = add_row(group, "", self.basis)
        self.walls: dict[str, tuple[QComboBox, QuantitySpinBox, QDoubleSpinBox]] = {}
        for side in WALL_ORDER:
            mode = _combo(THERMAL_MODE_ITEMS, f"{side} wall condition")
            mode.setObjectName(f"wall_{side}_mode")
            mode.setMinimumWidth(WALL_MODE_W)
            temp = QuantitySpinBox("temperature", "C", minimum=-273.0, maximum=3000.0, step=1.0, decimals=1)
            temp.setObjectName(f"wall_{side}_temperature")
            temp.setAccessibleName(f"{side} wall temperature")
            temp.setMaximumWidth(WALL_VALUE_W)
            compact_box(temp)
            flux = compact_box(_spin(-1e8, 1e8, 10.0, 1, " W/m²", name=f"{side} wall heat flux"))
            flux.setMaximumWidth(WALL_VALUE_W + 16)
            flux.setObjectName(f"wall_{side}_flux")
            mode.currentIndexChanged.connect(lambda _i, s=side: self._wall_edited(s))
            temp.siValueChanged.connect(lambda _v, s=side: self._wall_edited(s))
            flux.valueChanged.connect(lambda _v, s=side: self._wall_edited(s))
            self.walls[side] = (mode, temp, flux)
            self.rows[f"wall_{side}"] = add_row(group, side.capitalize(), hbox(mode, temp, flux, stretch=None),
                                                tooltip="Thermal condition on this side of the domain.")
        self.body = QComboBox()
        self.body.setObjectName("heat_body")
        self.body.setAccessibleName("Body")
        self.body.currentIndexChanged.connect(lambda _i: self._load_body())
        self.rows["body"] = add_row(group, "Body", self.body, tooltip="The body or part whose surface condition "
                                                                      "is shown below.")
        self.body_mode = _combo(THERMAL_MODE_ITEMS, "Surface condition")
        self.body_mode.setObjectName("heat_body_mode")
        self.body_mode.setMinimumWidth(WALL_MODE_W)
        self.body_temp = QuantitySpinBox("temperature", "C", minimum=-273.0, maximum=3000.0, step=1.0, decimals=1)
        self.body_temp.setObjectName("heat_body_temperature")
        self.body_temp.setAccessibleName("Surface temperature")
        compact_box(self.body_temp)
        self.body_flux = compact_box(_spin(-1e8, 1e8, 10.0, 1, " W/m²", name="Surface heat flux"))
        self.body_flux.setObjectName("heat_body_flux")
        self.body_mode.currentIndexChanged.connect(lambda _i: self._body_edited())
        self.body_temp.siValueChanged.connect(lambda _v: self._body_edited())
        self.body_flux.valueChanged.connect(lambda _v: self._body_edited())
        self.rows["surface"] = add_row(group, "Surface", hbox(self.body_mode, self.body_temp, self.body_flux,
                                                              stretch=None),
                                       tooltip="Adiabatic: no heat crosses the surface. Temperature: held "
                                               "at it. Heat flux: leaves the surface into the fluid.")
        self.body_note = label("", "muted", wrap=True)
        self.body_note.setObjectName("heat_body_note")
        self.rows["body_note"] = add_row(group, "", self.body_note)
        self.reload()

    def _bodies(self) -> list:
        d = self.design
        if d is None:
            return []
        return list(d.meshes) if self.dim == 3 else list(d.bodies)

    def reload(self) -> None:
        self._loading = True
        try:
            sim, th = self.sim, (self.sim.thermal if self.sim is not None else None)
            on = bool(th is not None and th.enabled)
            _silent(self.enabled, on)
            self.enabled.setEnabled(sim is not None)
            for key, row in self.rows.items():
                if key != "enabled":
                    row.set_visible(on)
            if th is None:
                return
            self.ambient.set_si_value(float(th.t_ambient))
            _silent(self.buoyancy, th.buoyancy)
            _silent(self.gravity, float(th.gravity))
            self.rows["gravity"].set_visible(on and th.buoyancy)
            self._load_props()
            dim = self.dim
            closed = sim.cfd.domain_mode == "closed"
            walls = {w.side: w for w in th.walls}
            for side, (mode, temp, flux) in self.walls.items():
                row = self.rows[f"wall_{side}"]
                show = on and side in SIDE_TEXT[dim]
                row.set_visible(show)
                if not show:
                    continue
                row.set_label(SIDE_TEXT[dim][side])
                # a screen reader names the side as the label does ("Side (y min) wall: temperature"),
                # not by the model's key ("front"), which a 3D room shows as a side wall
                for wdg, what in ((mode, "condition"), (temp, "temperature"), (flux, "heat flux")):
                    wdg.setAccessibleName(f"{SIDE_TEXT[dim][side]} wall: {what}")
                w = walls.get(side)
                m = w.mode if w is not None else "adiabatic"
                _set_combo(mode, m)
                temp.set_si_value(float(w.value) if w is not None and m == "temperature" else float(th.t_ambient))
                _silent(flux, float(w.value) if w is not None and m == "heat_flux" else 0.0)
                temp.setVisible(m == "temperature")
                flux.setVisible(m == "heat_flux")
                inflow = not closed and side in ("left", "right")
                row.set_enabled(not inflow)
                row.label.setToolTip("In a wind tunnel this side is the inlet, at the ambient temperature."
                                     if inflow and side == "left" else
                                     "In a wind tunnel this side is the outlet: heat leaves with the flow."
                                     if inflow else "Thermal condition on this side of the domain.")
            bodies = self._bodies()
            with QSignalBlocker(self.body):
                cur = self.body.currentIndex()
                self.body.clear()
                for i, b in enumerate(bodies):
                    self.body.addItem(b.name or f"Body {i + 1}", i)
                if bodies:
                    self.body.setCurrentIndex(min(max(cur, 0), len(bodies) - 1))
            has = bool(bodies)
            for key in ("body", "surface"):
                self.rows[key].set_visible(on and has)
            self._load_body()
            heated = sum(1 for b in bodies if b.thermal != "adiabatic")
            self.body_note.setText("No bodies in this design." if not has else
                                   f"{heated} of {len(bodies)} {'parts' if dim == 3 else 'bodies'} heated or cooled."
                                   if len(bodies) > 1 else "")
            self.rows["body_note"].set_visible(on and bool(self.body_note.text()))
        finally:
            self._loading = False

    def _load_props(self) -> None:
        sim = self.sim
        if sim is None:
            return
        props = thermal_props(sim)
        for key, (box, auto) in self.props.items():
            _silent(box, float(getattr(props, key)))
            with QSignalBlocker(auto):
                auto.setChecked(bool(props.auto.get(key, False)))
        self.basis.setText(f"Auto: {props.basis}." if any(props.auto.values()) else "")
        if not props.known:
            self.basis.setText(f"Auto: {props.basis}.")
        self.rows["basis"].set_visible(bool(sim.thermal.enabled) and bool(self.basis.text()))

    def _load_body(self) -> None:
        bodies = self._bodies()
        i = self.body.currentData()
        if i is None or not 0 <= int(i) < len(bodies):
            return
        b = bodies[int(i)]
        was, self._loading = self._loading, True
        try:
            _set_combo(self.body_mode, b.thermal)
            amb = self.sim.thermal.t_ambient if self.sim is not None else 20.0
            self.body_temp.set_si_value(float(b.thermal_value) if b.thermal == "temperature" else float(amb))
            _silent(self.body_flux, float(b.thermal_value) if b.thermal == "heat_flux" else 0.0)
            self.body_temp.setVisible(b.thermal == "temperature")
            self.body_flux.setVisible(b.thermal == "heat_flux")
        finally:
            self._loading = was

    # -- edits ----------------------------------------------------------------
    def _enabled_edited(self, on: bool) -> None:
        if not self._live():
            return
        self.sim.thermal.enabled = bool(on)
        self.reload()
        self.edited.emit()

    def _set(self, attr: str, value: Any, *, props: bool = False) -> None:
        if not self._live():
            return
        th = self.sim.thermal
        if getattr(th, attr) == value:
            return
        setattr(th, attr, value)
        if props:                       # the defaults depend on the ambient temperature
            self._loading = True
            try:
                self._load_props()
            finally:
                self._loading = False
        if attr == "buoyancy":
            self.rows["gravity"].set_visible(bool(value))
        self.edited.emit()

    def _prop_typed(self, key: str, value: float) -> None:
        if not self._live():
            return
        setattr(self.sim.thermal, key, float(value))
        _, auto = self.props[key]
        with QSignalBlocker(auto):
            auto.setChecked(False)
        self._loading = True
        try:
            self._load_props()
        finally:
            self._loading = False
        self.edited.emit()

    def _prop_auto(self, key: str, on: bool) -> None:
        if not self._live():
            return
        box, _ = self.props[key]
        setattr(self.sim.thermal, key, None if on else float(box.value()))
        self._loading = True
        try:
            self._load_props()
        finally:
            self._loading = False
        self.edited.emit()

    def _wall_edited(self, side: str) -> None:
        if not self._live():
            return
        mode_w, temp, flux = self.walls[side]
        mode = str(mode_w.currentData())
        th = self.sim.thermal
        temp.setVisible(mode == "temperature")
        flux.setVisible(mode == "heat_flux")
        rest = [w for w in th.walls if w.side != side]
        if mode == "temperature":
            rest.append(WallThermal(side=side, mode=mode, value=float(temp.si_value())))
        elif mode == "heat_flux":
            rest.append(WallThermal(side=side, mode=mode, value=float(flux.value())))
        order = {s: i for i, s in enumerate(WALL_ORDER)}
        th.walls = sorted(rest, key=lambda w: order.get(w.side, 99))     # adiabatic sides are not listed
        self.edited.emit()

    def _body_edited(self) -> None:
        if not self._live():
            return
        bodies = self._bodies()
        i = self.body.currentData()
        if i is None or not 0 <= int(i) < len(bodies):
            return
        b = bodies[int(i)]
        mode = str(self.body_mode.currentData())
        value = (float(self.body_temp.si_value()) if mode == "temperature" else
                 float(self.body_flux.value()) if mode == "heat_flux" else 0.0)
        self.body_temp.setVisible(mode == "temperature")
        self.body_flux.setVisible(mode == "heat_flux")
        if (b.thermal, float(b.thermal_value)) == (mode, value):
            return
        b.thermal, b.thermal_value = mode, value
        heated = sum(1 for x in bodies if x.thermal != "adiabatic")
        if len(bodies) > 1:
            self.body_note.setText(f"{heated} of {len(bodies)} {'parts' if self.dim == 3 else 'bodies'} heated or "
                                   "cooled.")
        self.designEdited.emit()
        self.edited.emit()


WALL_ORDER = ("left", "right", "bottom", "top", "front", "back")
#: the condition combo shows "Temperature" in full; the value beside it keeps the rest of the row
WALL_MODE_W, WALL_VALUE_W = 112, 92


# ---------------------------------------------------------------------------
# domain and diagnostics
# ---------------------------------------------------------------------------

class DomainSection(_SectionBase):
    """Domain type (wind tunnel or closed box) and its extent: Auto (the planner sizes it from the
    body and the quality) or a box typed or dragged in the preview."""

    def __init__(self, group, parent: QObject | None = None) -> None:
        super().__init__(group, parent)
        from .param_form import SegmentedButtons
        self.rows: dict[str, Any] = {}
        self.mode = SegmentedButtons([(k, t) for k, t, _ in DOMAIN_ITEMS])
        self.mode.setObjectName("domain_mode")
        for k, _t, tip in DOMAIN_ITEMS:
            self.mode.buttons[k].setToolTip(tip)
            self.mode.buttons[k].setAccessibleName(f"Domain type: {_t}")
        self.mode.currentChanged.connect(self._mode_edited)
        self.rows["mode"] = add_row(group, "Type", self.mode, tooltip="What surrounds the body.")
        self.auto = auto_toggle("Auto: the planner sizes the domain from the body and the quality. Typing an edge, "
                                "or dragging one in the preview, fixes the box.")
        self.auto.setObjectName("domain_auto")
        self.auto.setAccessibleName("Domain extent: automatic")
        self.auto.clicked.connect(self._auto_clicked)
        self.auto_note = label("", "muted")
        self.auto_note.setObjectName("domain_note")
        self.rows["extent"] = add_row(group, "Extent", hbox(self.auto_note, self.auto, stretch=0))
        self.ranges: dict[str, tuple[QuantitySpinBox, QuantitySpinBox]] = {}
        lu = display_unit("length")
        for ax in "xyz":
            # three significant figures: a 45.7 mm cavity must not read "0.046"
            lo = QuantitySpinBox("length", lu, minimum=-1e5, maximum=1e5, step=0.01, show_suffix=False,
                                 significant=3)
            hi = QuantitySpinBox("length", lu, minimum=-1e5, maximum=1e5, step=0.01, show_suffix=False,
                                 significant=3)
            lo.setObjectName(f"domain_{ax}0")
            hi.setObjectName(f"domain_{ax}1")
            lo.setAccessibleName(f"Domain {ax} from")
            hi.setAccessibleName(f"Domain {ax} to")
            lo.siValueChanged.connect(lambda _v: self._box_edited())
            hi.siValueChanged.connect(lambda _v: self._box_edited())
            compact_box(lo)
            compact_box(hi)
            self.ranges[ax] = (lo, hi)
            self.rows[ax] = add_row(group, f"{ax} range", range_pair(lo, hi), unit=lu,
                                    tooltip=f"Domain edges along {ax}, metres (wind frame).")
        self.reload()

    def reload(self) -> None:
        self._loading = True
        try:
            sim = self.sim
            if sim is None:
                return
            self.mode.set_current(sim.cfd.domain_mode)
            dim = self.dim
            explicit = explicit_box(sim, dim)
            with QSignalBlocker(self.auto):
                self.auto.setChecked(explicit is None)
            box = explicit or domain_extent(self.design, sim, self.extent)
            self.auto_note.setText("Fixed box" if explicit is not None else
                                   ("Planned around the body" if self.extent is not None else "Sized by the planner"))
            for i, ax in enumerate("xyz"):
                row = self.rows[ax]
                row.set_visible(i < dim)
                if i >= dim or box is None:
                    continue
                lo, hi = self.ranges[ax]
                lo.set_si_value(float(box[2 * i]))
                hi.set_si_value(float(box[2 * i + 1]))
        finally:
            self._loading = False

    def _mode_edited(self, key: str) -> None:
        if not self._live() or self.sim.cfd.domain_mode == key:
            return
        self.sim.cfd.domain_mode = key
        self.reload()
        self.edited.emit()

    def _auto_clicked(self, on: bool) -> None:
        if not self._live():
            return
        if on:
            self.sim.cfd.domain_box = None
        else:
            self.sim.cfd.domain_box = self._typed_box()
        self.reload()
        self.edited.emit()

    def _typed_box(self) -> tuple[float, ...]:
        out = []
        for ax in "xyz"[: self.dim]:
            lo, hi = (b.exact_si_value() for b in self.ranges[ax])      # untouched edges stay exact
            if hi <= lo:
                hi = lo + max(abs(lo) * 1e-3, 1e-3)
            out += [float(lo), float(hi)]
        return tuple(out)

    def _box_edited(self) -> None:
        if not self._live():
            return
        box = self._typed_box()
        for i, ax in enumerate("xyz"[: self.dim]):
            lo, hi = self.ranges[ax]
            invalid = not (box[2 * i + 1] > box[2 * i]) or hi.exact_si_value() <= lo.exact_si_value()
            set_prop(lo, "invalid", "true" if invalid else "")
            set_prop(hi, "invalid", "true" if invalid else "")
            # the red border is not the only sign: the fields say what is wrong and what was kept
            why = (f"The upper {ax} edge must be above the lower one; the box keeps a {ax} range of "
                   f"{units.sig(box[2 * i + 1] - box[2 * i], 3)} m." if invalid else
                   f"Domain edges along {ax}, metres (wind frame).")
            lo.setToolTip(why)
            hi.setToolTip(why)
        if self.sim.cfd.domain_box == box:
            return
        self.sim.cfd.domain_box = box
        with QSignalBlocker(self.auto):
            self.auto.setChecked(False)
        self.auto_note.setText("Fixed box")
        self.edited.emit()


class DiagnosticsSection(_SectionBase):
    """AeroJAX-style diagnostics (DESIGN.md 10.4): record the solver's health, and the time-step rule."""

    def __init__(self, group, parent: QObject | None = None) -> None:
        super().__init__(group, parent)
        self.record = QCheckBox("Solver health per step block")
        self.record.setObjectName("diagnostics_record")
        self.record.toggled.connect(self._record_edited)
        add_row(group, "Record", self.record, tooltip=DIAG_TIP)
        self.dt = _combo(TIMESTEP_ITEMS, "Time step rule")
        self.dt.setObjectName("adaptive_dt")
        self.dt.currentIndexChanged.connect(self._dt_edited)
        add_row(group, "Time step", self.dt, tooltip="How the solver chooses its time step.")
        self.reload()

    def reload(self) -> None:
        self._loading = True
        try:
            if self.sim is None:
                return
            _silent(self.record, bool(self.sim.cfd.record_diagnostics))
            _set_combo(self.dt, self.sim.cfd.adaptive_dt)
        finally:
            self._loading = False

    def _record_edited(self, on: bool) -> None:
        if self._live() and self.sim.cfd.record_diagnostics != bool(on):
            self.sim.cfd.record_diagnostics = bool(on)
            self.edited.emit()

    def _dt_edited(self, _i: int) -> None:
        key = str(self.dt.currentData())
        if self._live() and self.sim.cfd.adaptive_dt != key:
            self.sim.cfd.adaptive_dt = key
            self.edited.emit()


__all__ = [
    "AIR_CP",
    "ITEM_LISTS",
    "DiagnosticsSection",
    "DirectionField",
    "DomainSection",
    "PlacedItemsSection",
    "ThermalNumbers",
    "ThermalProps",
    "ThermalSection",
    "VectorField",
    "air_conductivity",
    "anchor",
    "apply_drag",
    "body_bounds",
    "copy_item",
    "default_emitter",
    "default_fan",
    "default_heat_source",
    "default_port",
    "dimension_of",
    "domain_extent",
    "emitter_nozzles",
    "explicit_box",
    "fluid_defaults",
    "fmt_number",
    "from_wind",
    "items_of",
    "move_item",
    "new_item",
    "physics_warnings",
    "port_flow",
    "scene_specs",
    "snap",
    "snap_step",
    "summary_text",
    "thermal_numbers",
    "thermal_props",
    "thermal_rows",
    "to_wind",
]
