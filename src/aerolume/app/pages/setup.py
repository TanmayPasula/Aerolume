"""Setup page: the flow conditions and solver settings of the current simulation.

Layout (docs/ui_guidelines.md sections 3-4)::

    +-- tool strip: Fit | Domain  Grid ----------------+-- Simulation name ---- status --+
    |                                                  |  Analysis                       |
    |   case preview: the geometry as the solver sees  |  Flow                           |
    |   it (rotated by the angle of attack), the flow  |  Flow visualisation (3D CFD)    |
    |   domain, the inflow and the road                |  Attitude                       |
    |                                                  |  Ground                         |
    |                                                  |  Mesh & time                    |
    +--------------------------------------------------+  Solver (collapsed)             |
    |  Estimate (key-value)   |  Planner (notes and    |  Polar (polar analyses only)    |
    |                         |  warnings, one a line) |              Explore live   Run |
    +--------------------------------------------------+---------------------------------+

**3D designs** (DESIGN.md 7.11) use the same page: the preview becomes a 3D
case view (:class:`~aerolume.app.widgets.geometry_canvas.CasePreview3D`: the
wind-frame body, the tunnel, the road, the symmetry plane), Flow gains the yaw
angle, Mesh & time shows each quality's 3D cell count and the symmetry-plane
decision with its reason, the Solver group states what the 3D sensitivity
map costs, and a **Flow visualisation** group sets the smoke of the run
(DESIGN.md 7.13): on or off, and its emitter - a rake (a row of nozzles across
the flow), a grid or a plane band - with the heights and the nozzle spacing,
automatic until typed. The group sits right under Flow, so the smoke the
preview draws is set without scrolling, and a click on the emitter in the
preview brings the group into view with its first field focused. The preview
draws the emitters where they will stand, so the user sees which streaks the
run will show before starting it. The
hand-over to the run is ``stored_flow_viz(flow_viz_of(sim))``: ``{"smoke": bool,
"rake": <viz3d.flow rake dict>}``, which the Simulate page and the job manager
read (``view3d_panel.setup_flow_viz``); where it is kept is explained at
:func:`_flow_viz_holder`. The
Estimate is ``SimPlan3D.summary()`` plus the smoke, ordered by
:func:`order_estimate` so the cost of the run (grid, cells, run time, device)
is read first. Wing designs add the
finite-wing analysis ("Wing": vortex lattice or lifting line on viscous
section polars, :class:`~aerolume.model.WingSettings`) with an estimate of the
section polars it must compute; "3D CFD" runs the lofted wing through the
Navier-Stokes solver instead.

The Project tree selects the simulation (there is no list on this page); the
inspector edits it. The case preview and the Estimate table show what the
choices *mean*: the real and the simulated Reynolds number, the grid, the cell
count, the time step and the estimated run time from
:func:`aerolume.simulation.plan.plan_simulation` (``SimPlan.summary()``, shown
as is except that the flow model reads "LES-WALE" with the long form in its
tooltip). DESIGN.md section 6 requires every number to carry its validity
context, and this is where the user first sees it. The planner's notes and
warnings are shown one line each - the first sentence, which carries the
decision and its number - with the full text in the tooltip; once per change
they are logged in full to the Messages dock (source "Planner") so the
decision trail survives.

Edits change the project's :class:`~aerolume.model.Simulation` in place and are
saved through ``AppState.update_simulation`` after a short debounce, so dragging
the angle slider does not write the project to disk sixty times a second. The
planner module imports PyTorch, so it is imported once on a background thread;
until it is ready the Estimate shows what can be computed without it.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import re
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core import fluid as fluids
from ...core import units
from ...model import (
    Design,
    Emitter,
    FlowConditions,
    Fluid,
    Simulation,
    ThermalSettings,
    WingSettings,
)
from ...simulation.defaults import default_cfd_settings, default_reynolds_cap
from ...simulation.defaults import default_simulation as core_default_simulation
from .. import theme
from ..dialogs.physics_templates import PHYSICS_TEMPLATES, add_template
from ..state import AppState
from ..widgets import physics_editor as phys
from ..widgets.common import Divider, EmptyState, IconButton, PropertyGrid, ToolStrip
from ..widgets.common import ElidedLabel as _CommonElidedLabel
from ..widgets.geometry_canvas import MAX_NOZZLES as MAX_SMOKE_NOZZLES
from ..widgets.geometry_canvas import (
    CasePreview,
    CasePreview3D,
    design_meshes3d,
    meshes_bounds,
    seed_rake_spec,
    smoke_rake_layout,
    stands_on_road,
)
from ..widgets.mesh_inspector import area_text
from ..widgets.param_form import (
    ElidedLabel,
    QuantitySpinBox,
    SegmentedButtons,
    clear_layout,
    display_unit,
    guard_wheel,
    hbox,
    keep_start_visible,
    label,
    set_line_text,
    set_prop,
    small_button,
    svg_icon,
)
from ..widgets.wing_editor import airfoil_resolver

INSPECTOR_WIDTH = 380
#: the inspector's floor; the form's own minimum width raises it (:class:`_FormMinWidth`)
INSPECTOR_MIN_WIDTH = 320
LABEL_W = 120


class _FormMinWidth(QObject):
    """Keeps the inspector panel at least as wide as the form it scrolls.

    The inspector scrolls only vertically, so a form wider than the panel was cut off on the
    right: at 1280 x 800 beside the Assistant, Setup's fields ran under the edge and the Emitters
    buttons read "uplicate" (critic round 3.4). A scroll area reports a small minimum whatever it
    holds, so the panel's minimum width is the page's only honest statement of what the form
    needs; with it, the page's minimum size hint is true, and the dock manager's page guard
    (``DockManager.page_guard_caps``) narrows the side panels instead of the form. Re-measured
    whenever the form lays itself out (groups and rows show and hide with the case)."""

    def __init__(self, panel: QWidget, scroll, form: QWidget, floor: int) -> None:
        super().__init__(panel)
        self._panel, self._scroll, self._form, self._floor = panel, scroll, form, int(floor)
        form.installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:  # Qt override
        from PySide6.QtCore import QEvent
        if obj is self._form and ev.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Show,
                                                QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self.update()
        return False

    def need(self) -> int:
        """The panel width at which the form is not cut: its minimum plus the vertical scroll bar."""
        bar = self._scroll.verticalScrollBar().sizeHint().width()
        frame = 2 * self._scroll.frameWidth()
        return self._form.minimumSizeHint().width() + bar + frame

    def update(self) -> None:
        want = min(max(self._floor, self.need()), self._panel.maximumWidth())
        if self._panel.minimumWidth() != want:
            self._panel.setMinimumWidth(want)


#: below this width [px] the Estimate / Planner strip stacks its two columns
INFO_STACK_WIDTH = 520


class _OnResize(QObject):
    """Calls ``callback()`` after each resize of ``widget`` (a layout decision that depends on
    the widget's own width, which the page's resize event does not see when a splitter moves)."""

    def __init__(self, widget: QWidget, callback) -> None:
        super().__init__(widget)
        self._widget, self._callback = widget, callback
        widget.installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:  # Qt override
        from PySide6.QtCore import QEvent
        if obj is self._widget and ev.type() == QEvent.Type.Resize:
            self._callback()
        return False
INFO_HEIGHT = 300            # initial height of the Estimate / Planner strip under the preview
PREVIEW_MIN_FRACTION = 0.38  # the strip grows to show the whole Estimate while the preview keeps this
#: The 3D case preview keeps half the height: a perspective view with the tunnel, the road and
#: the smoke emitters needs more room than a side view to read, and the Estimate scrolls instead
PREVIEW_MIN_FRACTION_3D = 0.5
PREVIEW_MIN_PX = 180
#: with placed physics the preview is a workspace (items are dragged there): it keeps half the height
PREVIEW_MIN_FRACTION_PHYSICS = 0.5
LOG_SOURCE = "Planner"
WHEELS_TEXT = "Rotate with the road"
NO_WHEELS_TEXT = "No wheels in this design"

# ---------------------------------------------------------------------------
# copy (docs/ui_guidelines.md section 6: labels are nouns, explanations are tooltips)
# ---------------------------------------------------------------------------

QUALITY_ITEMS = (
    ("draft", "Draft", ("Coarse grid and a short run: trends and setup checks in a minute or two. "
                        "Coefficients within roughly 10-20 %.")),
    ("standard", "Standard", "The default. Resolves design changes reliably; a few minutes on a GPU."),
    ("fine", "Fine", ("Finer grid and longer averaging for numbers you report. Several times slower "
                      "than Standard.")),
    ("custom", "Custom", "Grid, domain and duration from the fields below."),
)
QUALITY_TEXT = {k: (t, tip) for k, t, tip in QUALITY_ITEMS}
ANALYSIS_ITEMS = (
    ("cfd", "CFD", ("Navier-Stokes flow simulation: pressure and velocity fields, wake, unsteady "
                    "forces, sensitivity. Minutes.")),
    ("polar", "Polar", ("Viscous panel method (XFOIL-class): lift, drag and moment over a range of "
                        "angles. Seconds. Single-element airfoils only.")),
    ("wing", "Wing", ("Finite-wing analysis (XFLR5-class): vortex lattice or nonlinear lifting line on viscous "
                      "section polars. CL, CD (induced and profile), Cm, span loading and efficiency over a "
                      "range of angles. Seconds to a minute.")),
)
WING_CFD_TIP = ("The lofted wing in the 3D Navier-Stokes solver: the flow pattern (tip vortices, separation). "
                "For lift and drag at flight Reynolds numbers use the Wing analysis.")
WING_METHOD_ITEMS = (("vlm", "Vortex lattice (VLM)", "Lifting surface: any planform, sweep and dihedral"),
                     ("llt", "Lifting line (LLT)", ("Nonlinear lifting line: straight, moderately swept wings of "
                                                    "aspect ratio 4 or more; follows the section stall")))
SYMMETRY_ITEMS = (("auto", "Automatic", ("Half a domain with a symmetry plane at y = 0 when the design is "
                                         "mirror-symmetric and the yaw is zero (half the cells)")),
                  ("full", "Off (full domain)", "Always simulate both halves"))
#: Section-polar cost of a cold wing run (solvers.wing, measured: ~0.5 s per 2D polar point
#: at the default 140 panels; NACA 2412 two-section wing, 17 angles: 22 s) and the per-angle
#: cost of the 3D solve itself (lattice / lifting line, milliseconds to a tenth of a second).
WING_POINT_S = 0.5
WING_ANGLE_S = {"vlm": 0.06, "llt": 0.12}
GROUND_ITEMS = (
    ("none", "None (free flow)", "No road: a body far from the ground."),
    ("fixed", "Fixed floor", ("A wind tunnel without a belt: a boundary layer grows on the floor and "
                              "under-predicts underbody flow.")),
    ("moving", "Moving belt", "The road moves at the flow speed, as it does under a real car."),
)
TURBULENCE_ITEMS = (("auto", "Automatic"), ("laminar", "Laminar (no model)"), ("les", "LES (sub-grid model)"))
REYNOLDS_ITEMS = (("real", "Real"), ("capped", "Capped"))
PRECISION_ITEMS = (("auto", "Automatic"), ("float32", "Single (float32)"), ("float64", "Double (float64)"))
DEVICE_ITEMS = (("auto", "Automatic"), ("cuda", "GPU (CUDA)"), ("cpu", "CPU"), ("mps", "Apple GPU (MPS)"))
OBJECTIVE_ITEMS = (("cd", "Drag (Cd)"), ("cl", "Lift (Cl)"), ("ld", "Lift / drag"))
SPEED_UNITS = ("m/s", "km/h", "mph", "kn", "ft/s")
SPEED_TIP = "Free-stream speed"
#: a closed box has no free stream: the speed only sets the time unit L/U (and the Reynolds number shown)
CLOSED_SPEED_TIP = ("Reference speed of a closed box: there is no free stream, so it only sets the time unit "
                    "L/U and the time step. A speed near the fans' or jets' (or the free-fall velocity of a "
                    "buoyant flow) keeps the run length meaningful.")
NCRIT_TIP = ("Free-stream turbulence for the e^N transition model. 9 = average wind tunnel; 11-12 = very "
             "quiet air (sailplanes); 4-6 = turbulent air or a rough surface.")
STATUS_TEXT = {"new": ("Not run", "muted"), "queued": ("Queued", "secondary"), "running": ("Running", "secondary"),
               "done": ("Done", "status-good"), "failed": ("Failed — see Messages", "status-bad"),
               "cancelled": ("Stopped", "status-warn")}


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

#: CFD settings a new simulation copies from the project's latest comparable run (see
#: :func:`default_simulation`): everything that decides the flow and the grid except the
#: design-specific Reynolds handling, which default_cfd_settings re-derives.
_TEMPLATE_CFD_FIELDS = ("quality", "cells_per_ref", "domain_upstream", "domain_downstream", "domain_height",
                        "duration", "average_from", "turbulence", "precision", "device")


def default_simulation(design: Design, name: str, template: Simulation | None = None,
                       template_design: Design | None = None) -> Simulation:
    """A sensible first simulation for a design.

    Airfoils start at 4 deg so the first result shows lift; vehicles get the
    road, a moving belt and rotating wheels (if the design has wheels) because
    that is how a car really meets the air. The CFD settings come from
    :func:`aerolume.simulation.defaults.default_cfd_settings` (Draft; capped Re for
    airfoils above 1e5, with a fixed cap).

    ``template`` is the project's latest CFD simulation of a comparable design
    (:func:`comparable_template`). A second design is created to be compared
    with the first, and comparisons need the same conditions: without it, the
    NACA 4412 added next to the starter NACA 2412 (14.6 m/s, Re 1e6) got the
    model default of 30 m/s, and Compare showed 12.9 against 58.0 N/m of drag
    with no hint why. So the fluid, speed and angle, the quality and grid
    settings and the Reynolds mode and cap are copied. ``template_design`` is the
    template's design: when its reference length differs, this design has
    another Reynolds number, and the mode is derived for it afresh
    (``default_cfd_settings``) instead of copied.
    """
    if getattr(design, "dimension", 2) == 3:
        return _default_simulation3d(design, name, template)
    sim = Simulation(name=name, design_id=design.id)
    if design.kind == "airfoil":
        sim.conditions.alpha_deg = 4.0
    if template is not None:
        sim.conditions = FlowConditions.from_dict(template.conditions.to_dict())
    # Draft quality, and the capped Reynolds mode for airfoils above Re 1e5: the same
    # start as a new study (simulation.defaults), so the first Run comes back in
    # seconds to a minute and an airfoil's flow is resolved rather than a coarse LES.
    sim.cfd = default_cfd_settings(design, sim.conditions)
    if template is not None:
        for f in _TEMPLATE_CFD_FIELDS:
            setattr(sim.cfd, f, getattr(template.cfd, f))
        # the same Reynolds handling as the run it will be compared with when the two designs
        # share a reference length (and so a Reynolds number); else what suits this design
        same_re = template_design is None or math.isclose(float(template_design.length_ref),
                                                          float(design.length_ref), rel_tol=1e-9)
        if same_re:
            sim.cfd.reynolds_mode = template.cfd.reynolds_mode
            sim.cfd.reynolds_cap = template.cfd.reynolds_cap
        if sim.cfd.reynolds_mode == "capped" and not sim.cfd.reynolds_cap:
            sim.cfd.reynolds_cap = default_reynolds_cap(design)
        sim.cfd.compute_sensitivity = template.cfd.compute_sensitivity
        sim.cfd.sensitivity_objective = template.cfd.sensitivity_objective
        copy_physics(template, sim, same_design=template.design_id == design.id)
    if design.kind == "vehicle" or (design.source or {}).get("ground"):
        sim.ground.enabled = True
        sim.ground.moving = True
        sim.ground.rotate_wheels = any(b.role == "wheel" for b in design.bodies)
        if template is not None and template.ground.enabled:
            sim.ground.moving = template.ground.moving
            sim.ground.rotate_wheels = sim.ground.rotate_wheels and template.ground.rotate_wheels
    return sim


#: 3D CFD settings a new 3D simulation also copies from its template
_TEMPLATE_CFD_FIELDS_3D = (*_TEMPLATE_CFD_FIELDS, "symmetry", "domain_width")


def _default_simulation3d(design: Design, name: str, template: Simulation | None) -> Simulation:
    """A first simulation of a 3D design: :func:`aerolume.simulation.defaults.default_simulation`
    (Draft, real Re, automatic symmetry plane; the Wing analysis for wings; a moving road
    and spinning wheels for vehicles), then the conditions and settings of ``template``
    (the project's latest comparable run) so designs of one study are compared alike."""
    cond = FlowConditions.from_dict(template.conditions.to_dict()) if template is not None else None
    sim = core_default_simulation(design, name, cond)
    if template is None:
        return sim
    if template.analysis == "cfd" and sim.analysis == "cfd":
        for f in _TEMPLATE_CFD_FIELDS_3D:
            setattr(sim.cfd, f, getattr(template.cfd, f))
        sim.cfd.compute_sensitivity = template.cfd.compute_sensitivity
        sim.cfd.sensitivity_objective = template.cfd.sensitivity_objective
        # the same smoke as the run it will be compared with; heights and spacing fixed for
        # another body follow this one again (a typed height of a car means nothing on a sphere)
        if not template.emitters:
            fv = flow_viz_of(template)
            set_flow_viz(sim, {**fv, "z_min": None, "z_max": None, "spacing": None}
                         if template.design_id != design.id else fv)
        copy_physics(template, sim, same_design=template.design_id == design.id)
    if template.analysis == "wing" and sim.analysis == "wing":
        sim.wing = WingSettings.from_dict(template.wing.to_dict())
    if sim.ground.enabled and template.ground.enabled:
        sim.ground.moving = template.ground.moving
        sim.ground.rotate_wheels = sim.ground.rotate_wheels and template.ground.rotate_wheels
    return sim


#: CFD settings of the placed physics (DESIGN.md 10) a new simulation copies with them
_PHYSICS_CFD_FIELDS = ("domain_mode", "record_diagnostics", "adaptive_dt")


def copy_physics(template: Simulation, sim: Simulation, *, same_design: bool) -> None:
    """Give a new simulation the heat-transfer setup of the run it will be compared with.

    The thermal settings (on/off, ambient, buoyancy, fluid properties, domain walls), the domain
    type and the diagnostics are copied always: a variant of a heated case is compared at the same
    thermal conditions. The placed items (emitters, heat sources, fans, jets and vents) and an
    explicit domain box are copied only for another run of the *same* design - their positions are
    metres around that design and mean nothing around a different body."""
    sim.thermal = ThermalSettings.from_dict(template.thermal.to_dict())
    for f in _PHYSICS_CFD_FIELDS:
        setattr(sim.cfd, f, getattr(template.cfd, f))
    if same_design:
        sim.cfd.domain_box = template.cfd.domain_box
        for kind in phys.ITEM_LISTS:
            setattr(sim, kind, [type(o).from_dict(o.to_dict()) for o in getattr(template, kind)])


def comparable_template(project, design: Design) -> Simulation | None:
    """The project's most recently created simulation of a comparable design.

    Comparable: the same kind *and* dimension (a 3D car is not compared with a
    2D side profile), and the analysis a new simulation of ``design`` starts
    with - CFD, or the Wing analysis for a wing. Runs of ``design`` itself come
    first (a second run of the same design is usually a variation of the
    first), then any other design's. Polars are skipped: their speed is chosen
    for a Reynolds number and their angle is a sweep, so neither is what a CFD
    run should copy. None when there is nothing comparable (the first
    simulation of its kind keeps the model defaults).
    """
    if project is None:
        return None
    kinds = {did: (d.kind, getattr(d, "dimension", 2)) for did, d in project.designs.items()}
    want = "wing" if design.kind == "wing" else "cfd"
    me = (design.kind, getattr(design, "dimension", 2))
    cands = [s for s in project.simulations.values()
             if s.analysis == want and kinds.get(s.design_id) == me]
    if not cands:
        return None
    own = [s for s in cands if s.design_id == design.id]
    pool = own or cands
    # simulations keep insertion order; "created" breaks ties for copies made in one second
    return max(enumerate(pool), key=lambda t: (t[1].created or "", t[0]))[1]


# ---------------------------------------------------------------------------
# the design's own case: what a new run of it inherits, and the bodiless-tunnel gate
# ---------------------------------------------------------------------------

def has_body(design: Design | None) -> bool:
    """Whether a wind tunnel would have something in its flow: a 2D body, a 3D mesh or a wing (whose
    surface is built from its planform). A cavity or an empty room has none: its walls are the case."""
    if design is None:
        return False
    if design.kind == "wing":
        return True
    return bool(design.meshes if getattr(design, "dimension", 2) == 3 else design.bodies)


@dataclasses.dataclass
class CaseTemplate:
    """The setup a design is meant to run with: ``sim`` carries it (domain type and box, heat
    transfer, placed items, fluid and reference speed) and ``source`` says where it was read, for
    the Planner line and the log ("“Heated cavity · Ra 10⁴”", "the cavity case of this design")."""
    sim: Simulation
    source: str
    from_run: bool


def _reference_speed(design: Design, sim: Simulation) -> float | None:
    """The speed that drives a closed box (free-fall velocity, a fan's or a jet's), measured by the
    simulation layer's own scales (:func:`aerolume.simulation.physics.physics_scales`, the numbers
    the planner uses); None when it cannot say (nothing drives the box yet)."""
    try:
        from ...simulation.physics import physics_scales
        dim = 3 if getattr(design, "dimension", 2) == 3 else 2
        u = float(physics_scales(design, sim, dim, box=phys.explicit_box(sim, dim)).speed)
    except Exception:  # noqa: BLE001 - a box nothing drives keeps the speed it has
        return None
    return u if u > 0 and math.isfinite(u) else None


def _case_settings_sim(design: Design, case) -> Simulation:
    """A simulation of ``design`` with the settings of an :class:`aerolume.api.Case` (the keywords a
    script passes to ``run_cfd``): fluid, closed box, heat transfer and fans."""
    s = case.settings
    sim = Simulation(name=case.name, design_id=design.id)
    if s.get("fluid") is not None:
        sim.conditions.fluid = Fluid.from_dict(s["fluid"].to_dict())
    if s.get("domain") == "closed":
        sim.cfd.domain_mode = "closed"
        if s.get("domain_box") is not None:
            sim.cfd.domain_box = tuple(float(v) for v in s["domain_box"])
    if s.get("thermal") is not None:
        sim.thermal = ThermalSettings.from_dict(s["thermal"].to_dict())
    for kind in phys.ITEM_LISTS:
        if s.get(kind):
            setattr(sim, kind, [type(o).from_dict(o.to_dict()) for o in s[kind]])
    if sim.cfd.domain_mode == "closed":
        u = _reference_speed(design, sim)
        if u is not None:
            sim.conditions.speed = float(f"{u:.4g}")
    return sim


def case_source_template(design: Design) -> CaseTemplate | None:
    """The setup recorded in ``design.source`` by the case that made it, rebuilt, or None.

    Two kinds of case leave their parameters on the design: the starter cases of the New case menu
    (``{"type": "physics_template", "template": key}``, rebuilt with
    :func:`~aerolume.app.dialogs.physics_templates.build_template` - the same geometry, so its placed
    items stand where they stood) and the reference cases of the scripting API (``{"case": "cavity" |
    "fan_room" | "heated_cylinder", ...}``, rebuilt with the same :mod:`aerolume.api` constructor, so a
    cavity keeps the Rayleigh number it was built for). Rebuilding instead of copying a table of
    settings keeps one definition of each case."""
    src = dict(design.source or {})
    dim = 3 if getattr(design, "dimension", 2) == 3 else 2
    try:
        if src.get("type") == "physics_template" and src.get("template") in {t.key for t in PHYSICS_TEMPLATES}:
            from ..dialogs.physics_templates import build_template
            key = str(src["template"])
            ra = src.get("rayleigh") if key == "cavity" else None
            _d, sims = build_template(key, ra=float(ra) if ra else None)
            if not sims:
                return None
            sim = sims[0]
            sim.design_id = design.id
            title = next((t.title for t in PHYSICS_TEMPLATES if t.key == key), key)
            return CaseTemplate(sim, f"the {title} starter case", False)
        case = src.get("case")
        if case not in ("cavity", "fan_room", "heated_cylinder"):
            return None
        from ... import api
        if case == "cavity":
            built = api.cavity_case(float(src.get("ra", 1e4)), float(src.get("pr", 0.71)),
                                    size=float(src.get("size", design.length_ref)), dimension=dim)
            sim = _case_settings_sim(design, built)
            dt = src.get("delta_t")
            if dt and sim.thermal.walls:
                # ΔT is what the design was built with (its ambient may not be the constructor's default)
                t0 = float(sim.thermal.t_ambient)
                for w in sim.thermal.walls:
                    if w.mode == "temperature":
                        w.value = t0 + 0.5 * float(dt) * (1.0 if w.value > t0 else -1.0)
            return CaseTemplate(sim, "the heated-cavity case this design was built from", False)
        if case == "fan_room":
            pos = design.moment_ref3 if dim == 3 else design.moment_ref
            built = api.fan_room_case(float(src.get("thrust", 1.0)), room=src.get("room"),
                                      diameter=float(src.get("diameter", design.length_ref)), dimension=dim,
                                      position=tuple(float(v) for v in pos) if pos is not None else None)
            return CaseTemplate(_case_settings_sim(design, built), "the fan-in-a-room case this design was built from",
                                False)
        objs = design.meshes if dim == 3 else design.bodies
        held = [float(b.thermal_value) for b in objs if getattr(b, "thermal", "") == "temperature"]
        dt = float(src.get("delta_t", 10.0))
        dia = float((src.get("params") or {}).get("diameter", design.length_ref))
        kw = {"t_ambient": held[0] - dt} if held else {}
        built = api.heated_cylinder_case(d=dia, delta_t=dt, **kw)
        return CaseTemplate(_case_settings_sim(design, built), "the heated-cylinder case this design was built from",
                            False)
    except Exception:  # noqa: BLE001 - a case that cannot be rebuilt gives no template, never an error
        return None


def design_case_template(project, design: Design, *, exclude_id: str | None = None) -> CaseTemplate | None:
    """The setup a new (or a broken) run of ``design`` should start from.

    The design's own latest CFD run comes first: it is what the user last ran this design with. For a
    design without a body only its closed-box runs count - a tunnel run of a cavity cannot be run, and
    a copy of one is how "Heated cavity" got a new run that failed with "no bodies to simulate". Then
    the case recorded on the design (:func:`case_source_template`). ``exclude_id`` leaves out the run
    being repaired. None when neither exists."""
    if project is not None:
        own = [s for s in project.simulations.values()
               if s.design_id == design.id and s.analysis == "cfd" and s.id != exclude_id]
        if not has_body(design):
            own = [s for s in own if s.cfd.domain_mode == "closed"]
        if own:
            best = max(enumerate(own), key=lambda t: (t[1].created or "", t[0]))[1]
            return CaseTemplate(best, f"“{best.name}”", True)
    return case_source_template(design)


def inherit_design_physics(project, design: Design, sim: Simulation) -> str | None:
    """Give a new simulation of ``design`` the domain, heat transfer and placed items of its case.

    Without this a new run of a cavity started as a wind tunnel without heat transfer: the default
    of a body in a free stream, which a design without a body cannot run. Applies
    :func:`copy_physics` from :func:`design_case_template`; a closed box also takes the case's fluid
    and reference speed (its time unit) unless the template is a run whose conditions the new one
    already copied. Returns where the setup came from, or None when the design has no case."""
    if sim.analysis != "cfd":
        return None
    t = design_case_template(project, design, exclude_id=sim.id)
    if t is None:
        return None
    copy_physics(t.sim, sim, same_design=True)
    if sim.cfd.domain_mode == "closed":
        sim.conditions.fluid = Fluid.from_dict(t.sim.conditions.fluid.to_dict())
        sim.conditions.speed = float(t.sim.conditions.speed)
    return t.source


NBSP = chr(0xA0)  # no-break space
_ITEM_PLURAL = {"emitters": "emitters", "heat_sources": "heat sources", "fans": "fans", "ports": "jets and vents"}


@dataclasses.dataclass
class RunBlocker:
    """A setup that would only fail on the solver, said before Run with the change that fixes it.

    ``text`` is the Planner's error line, ``action`` the label of the button beside it and ``changes``
    what the button changes (its tooltip). :func:`apply_run_fix` applies it."""
    text: str
    action: str
    changes: list[str]
    template: Simulation
    copy_thermal: bool
    copy_items: tuple[str, ...]


def run_blocker(design: Design | None, sim: Simulation | None, project=None) -> RunBlocker | None:
    """What stops ``sim`` from running, with its fix, or None when nothing this page knows of does.

    A wind tunnel around a design without a body has nothing to simulate: the planner refuses it with
    "The design has no bodies to simulate" and a run fails. Such a design is a closed box (a cavity,
    an empty room), so the fix is the closed box of its case (:func:`design_case_template`) with that
    case's heat transfer and placed items where this run has none; a design without a recorded case
    gets a box of its reference length, which the planner then states as it does any closed box."""
    if design is None or sim is None or sim.analysis != "cfd" or sim.cfd.domain_mode == "closed":
        return None
    if has_body(design):
        return None
    t = design_case_template(project, design, exclude_id=sim.id)
    dim = 3 if getattr(design, "dimension", 2) == 3 else 2
    if t is None:
        L = float(design.length_ref) if design.length_ref and design.length_ref > 0 else 1.0
        tpl = Simulation(name="", design_id=design.id)
        tpl.cfd.domain_mode = "closed"
        tpl.cfd.domain_box = phys.explicit_box(sim, dim) or tuple(v for _ in range(dim) for v in (0.0, L))
        tpl.conditions = FlowConditions.from_dict(sim.conditions.to_dict())
        size = " × ".join(units.sig(tpl.cfd.domain_box[2 * i + 1] - tpl.cfd.domain_box[2 * i], 3)
                          for i in range(dim))
        return RunBlocker(
            text=("This design has no body, so a wind tunnel would have nothing in its flow: the run would fail. "
                  f"A design without a body is a closed box. Switch to Closed box ({size} m), then give it what "
                  "drives the flow: a fan, a jet or a heated wall."),
            action="Switch to Closed box", changes=[f"Domain ▸ Type: Closed box, {size} m"], template=tpl,
            copy_thermal=False, copy_items=())
    tpl = t.sim
    box = phys.explicit_box(tpl, dim) or phys.explicit_box(sim, dim)
    changes = ["Domain ▸ Type: Closed box" + (
        " (" + " × ".join(units.sig(box[2 * i + 1] - box[2 * i], 3) for i in range(dim)) + " m)" if box else "")]
    copy_thermal = tpl.thermal.enabled and not sim.thermal.enabled
    if copy_thermal:
        walls = [w for w in tpl.thermal.walls if w.mode == "temperature"]
        held = ", ".join(f"{w.side} wall {units.sig(w.value, 4)} °C" for w in walls)
        changes.append("Heat transfer on" + (f" ({held})" if held else ""))
    copy_items = tuple(k for k in phys.ITEM_LISTS if getattr(tpl, k) and not getattr(sim, k))
    for k in copy_items:
        n = len(getattr(tpl, k))
        noun = phys.ITEM_NOUN[k] if n == 1 else _ITEM_PLURAL[k]
        changes.append(f"{n} {noun} from the case")
    if tpl.cfd.domain_mode == "closed":
        changes.append(f"Reference speed {units.sig(tpl.conditions.speed, 3)} m/s (the box's own time unit)")
    # a short run name is kept on one line: "Ra 1.0×10⁴" broken after "Ra" read as two things
    source = t.source.replace(" ", NBSP) if t.from_run and len(t.source) <= 40 else t.source
    return RunBlocker(
        text=(f"This design is a closed box: it has no body, so a wind tunnel would have nothing in its flow and the "
              f"run would fail. Switch to Closed box to run it as {source}."),
        action="Switch to Closed box", changes=changes, template=tpl, copy_thermal=copy_thermal,
        copy_items=copy_items)


def apply_run_fix(sim: Simulation, blocker: RunBlocker, dim: int = 2) -> None:
    """Apply ``blocker``'s fix to ``sim``: the closed box (the template's, else the run's own fixed box),
    the template's heat transfer and fluid when the run had none, its placed items where the run's
    lists are empty, and a closed template's reference speed. Settings the user chose stay."""
    tpl = blocker.template
    sim.cfd.domain_mode = "closed"
    box = phys.explicit_box(tpl, dim) or phys.explicit_box(sim, dim)
    sim.cfd.domain_box = tuple(box) if box is not None else None
    if blocker.copy_thermal:
        sim.thermal = ThermalSettings.from_dict(tpl.thermal.to_dict())
        sim.conditions.fluid = Fluid.from_dict(tpl.conditions.fluid.to_dict())
    for k in blocker.copy_items:
        setattr(sim, k, [type(o).from_dict(o.to_dict()) for o in getattr(tpl, k)])
    if tpl.cfd.domain_mode == "closed" and tpl.conditions.speed > 0:
        sim.conditions.speed = float(tpl.conditions.speed)


def polar_allowed(design: Design | None) -> bool:
    """Polars need a single-element airfoil: the panel method has no wake model
    for gaps between elements and no notion of a road."""
    return (design is not None and getattr(design, "dimension", 2) == 2 and design.kind == "airfoil"
            and len(design.bodies) == 1)


def analyses_for(design: Design | None) -> list[str]:
    """The analyses a design offers, in the order of the Type control:
    2D airfoils CFD and Polar, wings Wing and (3D) CFD, everything else CFD."""
    if design is None:
        return ["cfd"]
    if design.kind == "wing":
        return ["wing", "cfd"]
    return ["cfd", "polar"] if polar_allowed(design) else ["cfd"]


def has_wheels(design: Design | None) -> bool:
    """Whether a design has wheel parts (2D wheel bodies or 3D wheel meshes)."""
    if design is None:
        return False
    return any(b.role == "wheel" for b in design.bodies) or any(m.role == "wheel" for m in design.meshes)


def ground_mode(sim: Simulation) -> str:
    """"none", "fixed" or "moving" from a simulation's GroundConfig."""
    if not sim.ground.enabled:
        return "none"
    return "moving" if sim.ground.moving else "fixed"


def shows_ground(design: Design | None) -> bool:
    """Whether the Ground group applies: vehicles and designs that declare a road
    (2D ``source["ground"]``, a 3D body imported standing on it: ``source["on_road"]``)."""
    if design is None:
        return False
    src = design.source or {}
    return design.kind == "vehicle" or bool(src.get("ground")) or bool(src.get("on_road"))


def symmetry_reason(design: Design, sim: Simulation, plan=None) -> tuple[bool, str]:
    """(half domain?, why) - the symmetry-plane decision of a 3D CFD run in words.

    The planner's own decision (``plan.symmetry_plane``) when a plan exists;
    otherwise the rule it applies: half a domain only when the setting is
    Automatic, the design is mirror-symmetric about y = 0 and the yaw is zero.
    """
    half = bool(getattr(plan, "symmetry_plane", False)) if plan is not None else (
        sim.cfd.symmetry == "auto" and bool(design.symmetric) and abs(sim.conditions.yaw_deg) < 1e-9)
    if sim.cfd.symmetry == "full":
        return False, "Off: both halves are simulated."
    if half:
        return True, "Half domain (y ≥ 0): the design is mirror-symmetric and the yaw is zero, so the cells halve."
    if abs(sim.conditions.yaw_deg) >= 1e-9:
        return False, "Full domain: the yaw angle makes the flow asymmetric."
    if not design.symmetric:
        return False, "Full domain: the design is not mirror-symmetric about y = 0 (Geometry ▸ References)."
    return False, "Full domain: the planner measured a mirror error above the limit (see the Planner list)."


# ---------------------------------------------------------------------------
# flow visualisation of a 3D run: the smoke and where it is released (DESIGN.md 7.13)
# ---------------------------------------------------------------------------

#: The Setup page's view of a simulation's smoke settings: a flat dict. ``None``
#: means automatic (``aerolume.viz3d.flow.resolve_rake`` decides for the body).
#: ``z_min`` is the height of a rake (``z_max`` then equals it); ``n`` holds the
#: nozzle counts derived from a typed ``spacing``.
FLOW_VIZ_DEFAULTS: dict[str, Any] = {"smoke": True, "kind": "rake", "z_min": None, "z_max": None,
                                     "spacing": None, "n": None}
FLOW_VIZ_KEYS = tuple(FLOW_VIZ_DEFAULTS)
RAKE_ITEMS = (
    ("rake", "Rake", ("A row of nozzles across the flow at one height: one streak per nozzle, the classic "
                      "wind-tunnel smoke rake")),
    ("vertical", "Vertical rake", ("A column of nozzles up the height on the centre plane: the streaks show "
                                   "the flow over the top and into the wake, as a smoke wand does")),
    ("grid", "Grid", ("Nozzles on a grid across the flow (width × height): streaks over the whole front of "
                      "the body")),
    ("plane", "Plane", ("A continuous band across the flow between two heights: a smoke sheet that shows "
                        "separation and the wake as a volume")),
)
RAKE_NOUN = {"rake": "Rake", "vertical": "Vertical rake", "grid": "Grid", "plane": "Plane"}
SMOKE_TIP = ("A passive tracer (dye) released upstream and carried by the flow: drawn as 3D smoke around the "
             "body during the run and in Results. One more field for the solver to transport; it does not "
             "change the forces.")
#: Settings of simulations whose model has nowhere to keep them yet (see :func:`_flow_viz_holder`),
#: by simulation id: they then hold for the session only.
_SESSION_FLOW_VIZ: dict[str, dict] = {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


_RAKE_NAMES = ("rake", "seed_rake", "smoke_rake", "dye_rake")
_FLAG_NAMES = ("smoke", "dye", "enabled")


def _flow_viz_holder(sim: Simulation) -> tuple[Any, str, str] | None:
    """(owner, attribute, mode) of where a simulation keeps its smoke settings, or None.

    The data model is owned by the core: DESIGN.md 7.13 defines the solver's
    ``DyeRake`` and the rake dict of :mod:`aerolume.viz3d.flow`
    (``ViewSettings.seed_rake``) but not, so far, the Simulation field the
    Setup page writes. So the page looks for the places a model may provide,
    on the Simulation or on its CFDSettings: a container ``flow_viz`` (a dict
    or a dataclass, with a nested rake dict or flat), a ``smoke`` / ``dye``
    that is either such a container or a bool with a sibling rake (mode
    "flag"), or a CFDSettings ``extras`` dict. It reads and writes whichever
    exists. Without any, the settings live for the session
    (:data:`_SESSION_FLOW_VIZ`); they are never written into an attribute the
    model does not declare, which ``to_dict`` would silently drop.

    What it writes is ``{"smoke": bool, "rake": <rake dict>}``, the rake dict
    in the format of ``viz3d.flow`` (kind, z, n) plus the typed ``spacing``,
    so a runner can hand it to ``live3d.dye_rake_for`` unchanged.
    """
    owners = (sim, sim.cfd)
    for owner in owners:
        if hasattr(owner, "flow_viz"):
            return owner, "flow_viz", "container"
    for name in ("smoke", "dye"):
        for owner in owners:
            if hasattr(owner, name):
                return owner, name, ("flag" if isinstance(getattr(owner, name), bool) else "container")
    extras = getattr(sim.cfd, "extras", None)
    if isinstance(extras, dict):
        return extras, "flow_viz", "container"
    return None


def _as_dict(obj: Any) -> Any:
    """A settings holder as a plain dict: model types by ``to_dict``, other dataclasses
    field by field, plain objects by their attributes; dicts and None as they are."""
    if obj is None or isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return dict(vars(obj)) if hasattr(obj, "__dict__") else obj


def _number(v: Any) -> float | None:
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    return float(v) if ok else None


def _normalise_flow_viz(raw: Any) -> dict:
    """Any of the shapes :func:`_flow_viz_holder` finds, as the flat Setup dict.

    Reads the rake dict of ``viz3d.flow`` (``z`` a height or a [low, high]
    pair, ``n`` an int or [n_span, n_height]) as well as flat ``z_min`` /
    ``z_max`` keys; anything damaged reads as automatic."""
    out = dict(FLOW_VIZ_DEFAULTS)
    raw = _as_dict(raw)
    if not isinstance(raw, dict):
        return out
    flat = dict(raw)
    for nest in _RAKE_NAMES:
        sub = _as_dict(raw.get(nest))
        if isinstance(sub, dict):
            flat = {**sub, **{k: v for k, v in raw.items() if k not in _RAKE_NAMES}}
            break
    for key in _FLAG_NAMES:
        if isinstance(flat.get(key), bool):
            out["smoke"] = flat[key]
            break
    if flat.get("kind") in RAKE_NOUN:
        out["kind"] = flat["kind"]
    z = flat.get("z")
    if isinstance(z, (list, tuple)) and len(z) == 2 and all(_number(v) is not None for v in z):
        out["z_min"], out["z_max"] = sorted((float(z[0]), float(z[1])))
    elif _number(z) is not None:
        out["z_min"] = out["z_max"] = float(z)
    else:
        out["z_min"], out["z_max"] = _number(flat.get("z_min")), _number(flat.get("z_max"))
    if out["kind"] == "rake" and out["z_min"] is not None:
        out["z_max"] = out["z_min"]
    if out["z_min"] is None or out["z_max"] is None:
        out["z_min"] = out["z_max"] = None
    out["spacing"] = _number(flat.get("spacing"))
    if out["spacing"] is not None and out["spacing"] <= 0:
        out["spacing"] = None
    n = flat.get("n")
    if isinstance(n, (list, tuple)) and len(n) == 2 and all(isinstance(v, int) and v > 0 for v in n):
        out["n"] = [int(n[0]), int(n[1])]
    elif isinstance(n, int) and not isinstance(n, bool) and n > 0:
        out["n"] = int(n)
    return out


#: nozzles a continuous emitter (a slot, an area) is sampled with for the seed-rake hand-over
_SAMPLES_LINE, _SAMPLES_AREA = 16, 5


def _emitter_points(e: Emitter) -> np.ndarray:
    """(N, 3) points standing in for one emitter in a seed rake: its nozzles (point, rake), a line
    sampled every 1/15 of its length, or an area sampled on a 5 × 5 grid across it."""
    dim = 3
    p = phys.vec(e.position, dim)
    if e.kind in ("point", "rake"):
        return phys.emitter_nozzles(e, dim)
    if e.kind == "line" and e.end is not None:
        t = np.linspace(0.0, 1.0, _SAMPLES_LINE)
        return p[None] + t[:, None] * (phys.vec(e.end, dim) - p)[None]
    size = list(e.size) or [0.1]
    if e.kind == "disc":
        u, v = _plane_basis(e.normal)
        r = float(size[0])
        g = np.linspace(-r, r, _SAMPLES_AREA)
        pts = [p + a * u + b * v for a in g for b in g if a * a + b * b <= r * r + 1e-12]
        return np.asarray(pts, dtype=float)
    if e.kind == "plane":
        u, v = _plane_basis(e.normal)
        w, h = float(size[0]), float(size[1] if len(size) > 1 else size[0])
        return np.asarray([p + a * u + b * v for a in np.linspace(-w / 2, w / 2, _SAMPLES_AREA)
                           for b in np.linspace(-h / 2, h / 2, _SAMPLES_AREA)], dtype=float)
    size += [size[-1]] * (3 - len(size))                      # a box: its mid-plane across the flow
    ys = np.linspace(-size[1] / 2, size[1] / 2, _SAMPLES_AREA)
    zs = np.linspace(-size[2] / 2, size[2] / 2, _SAMPLES_AREA)
    return np.asarray([p + (0.0, a, b) for a in ys for b in zs], dtype=float)


def _plane_basis(normal) -> tuple[np.ndarray, np.ndarray]:
    n = phys.unit_vec(normal, 3)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return u, np.cross(n, u)


def emitters_rake_spec(emitters: list) -> dict | None:
    """The seed-rake hand-over (a :mod:`aerolume.viz3d.flow` rake dict, metres) for a 3D run's
    emitters, or None when no 3D emitter is active.

    The 3D runner releases its smoke from ``ViewSettings.seed_rake`` (``jobs.seed_rake_for``) until
    the solver reads ``Simulation.emitters`` itself (DESIGN.md 10.1); this keeps the two in step. One
    rake across the span (or up the height) and one plane band facing the flow are passed as the
    rake / vertical / plane kinds the Simulate page's rake editor shows; anything else (several
    emitters, an oblique line, a disc) as their nozzle points. Positions are taken as stored:
    exact for emitters fixed in the tunnel (the default) and for body-fixed ones at zero pitch and yaw.
    The nozzle radius and the pulses (period and duty -> on / off seconds) of the first emitter ride
    along."""
    act = [e for e in emitters if e.enabled and len(e.position) == 3]
    if not act:
        return None
    first = act[0]
    extra: dict = {}
    if first.radius > 0 and first.kind != "disc":
        extra["radius"] = float(first.radius)
    if first.pulse_period > 0:
        on = float(first.pulse_period) * float(first.pulse_duty)
        extra["pulse"] = [on, float(first.pulse_period) - on]
    if len(act) == 1 and first.kind in ("rake", "line", "plane"):
        e = first
        p = phys.vec(e.position, 3)
        if e.kind == "plane" and np.allclose(np.abs(phys.unit_vec(e.normal, 3)), (1.0, 0.0, 0.0), atol=1e-9):
            w, h = (list(e.size) + [0.1, 0.1])[:2]
            return {"kind": "plane", "x": float(p[0]), "y": [float(p[1] - w / 2), float(p[1] + w / 2)],
                    "z": [float(p[2] - h / 2), float(p[2] + h / 2)], **extra}
        if e.kind in ("rake", "line") and e.end is not None:
            q = phys.vec(e.end, 3)
            n = int(e.count) if e.kind == "rake" else _SAMPLES_LINE
            if abs(q[0] - p[0]) < 1e-12 and abs(q[2] - p[2]) < 1e-12 and abs(q[1] - p[1]) > 0:
                return {"kind": "rake", "x": float(p[0]), "y": sorted([float(p[1]), float(q[1])]),
                        "z": float(p[2]), "n": n, **extra}
            if abs(q[0] - p[0]) < 1e-12 and abs(q[1] - p[1]) < 1e-12 and abs(q[2] - p[2]) > 0:
                return {"kind": "vertical", "x": float(p[0]), "y": float(p[1]),
                        "z": sorted([float(p[2]), float(q[2])]), "n": n, **extra}
    pts = np.vstack([_emitter_points(e) for e in act])
    return {"kind": "points", "points": pts.tolist(), **extra}


def flow_viz_of(sim: Simulation | None) -> dict:
    """The smoke settings of a simulation as a flat dict (defaults when it has none).

    A simulation with emitters (``Simulation.emitters``, DESIGN.md 10.1) is described by them:
    ``smoke`` says whether any is active and ``spec`` carries their seed-rake dict
    (:func:`emitters_rake_spec`), which :func:`stored_flow_viz` hands over as is. Without emitters,
    the automatic smoke rake of earlier versions applies, with whatever the user set for it."""
    if sim is None:
        return dict(FLOW_VIZ_DEFAULTS)
    emitters = list(getattr(sim, "emitters", None) or [])
    if emitters:
        spec = emitters_rake_spec(emitters)
        kind = spec["kind"] if spec is not None and spec["kind"] in RAKE_NOUN else "rake"
        return {**FLOW_VIZ_DEFAULTS, "smoke": spec is not None, "kind": kind, "spec": spec}
    holder = _flow_viz_holder(sim)
    if holder is None:
        return _normalise_flow_viz(_SESSION_FLOW_VIZ.get(sim.id))
    owner, name, mode = holder
    value = _get(owner, name)
    if mode == "flag":                         # a bool, its rake beside it
        rake = next((getattr(owner, r) for r in _RAKE_NAMES if getattr(owner, r, None) is not None), None)
        return _normalise_flow_viz({"smoke": bool(value), "rake": rake})
    return _normalise_flow_viz(value)


def _write_fields(target: Any, values: dict) -> Any:
    """Update a dict or a dataclass-like object with the keys it knows (dicts take all)."""
    if isinstance(target, dict):
        target.update(values)
        return target
    for k, v in values.items():
        if hasattr(target, k):
            setattr(target, k, v)
    return target


def stored_flow_viz(fv: dict) -> dict:
    """The flat settings as stored: ``{"smoke": bool, "rake": <viz3d.flow rake dict>}`` (the
    emitters' own rake dict when :func:`flow_viz_of` derived ``fv`` from emitters)."""
    if "spec" in fv:
        spec = fv["spec"]
        return {"smoke": bool(fv.get("smoke", spec is not None)), "rake": dict(spec) if spec else None}
    return {"smoke": bool(fv.get("smoke", True)), "rake": seed_rake_spec(fv)}


def set_flow_viz(sim: Simulation, fv: dict) -> None:
    """Store the flat smoke settings ``fv`` in ``sim`` (wherever :func:`_flow_viz_holder` says)."""
    fv = {k: fv.get(k, FLOW_VIZ_DEFAULTS[k]) for k in FLOW_VIZ_KEYS}
    stored = stored_flow_viz(fv)
    holder = _flow_viz_holder(sim)
    if holder is None:
        _SESSION_FLOW_VIZ[sim.id] = stored
        return
    owner, name, mode = holder
    rake = stored["rake"]
    # an object (not a dict) holding the rake may name the heights z_min / z_max instead of z
    rake_obj = {**rake, "z_min": fv["z_min"], "z_max": fv["z_max"], "spacing": fv["spacing"], "n": fv["n"]}
    if mode == "flag":
        setattr(owner, name, stored["smoke"])
        rname = next((r for r in _RAKE_NAMES if hasattr(owner, r)), None)
        if rname is not None:
            cur = getattr(owner, rname)
            if cur is None or isinstance(cur, dict):
                setattr(owner, rname, dict(rake))
            else:
                _write_fields(cur, rake_obj)
        return
    cur = _get(owner, name)
    if cur is None or isinstance(cur, dict):
        if isinstance(owner, dict):
            owner[name] = stored
        else:
            setattr(owner, name, stored)
        return

    def has(obj: Any, key: str) -> bool:
        return hasattr(obj, key)

    flag = next((k for k in _FLAG_NAMES if has(cur, k)), "smoke")    # the on/off under the holder's name
    rname = next((k for k in _RAKE_NAMES if has(cur, k)), None)
    if rname is not None:                         # an object with a nested rake
        _write_fields(cur, {flag: stored["smoke"]})
        nested = getattr(cur, rname)
        if nested is None or isinstance(nested, dict):
            setattr(cur, rname, dict(rake))
        else:
            _write_fields(nested, rake_obj)
    else:                                         # a flat object
        _write_fields(cur, {flag: stored["smoke"], **rake_obj})


def copy_flow_viz(src: Simulation, dst: Simulation) -> None:
    """Give ``dst`` the smoke settings of ``src`` (a copy made through ``to_dict`` carries
    them already when the model holds them; this covers the session store)."""
    if _flow_viz_holder(src) is None or _flow_viz_holder(dst) is None:
        set_flow_viz(dst, flow_viz_of(src))


def smoke_fact(layout: dict | None, fv: dict) -> str:
    """The Estimate's "Smoke" value: "Rake, 13 nozzles", "Plane band" or "Off"."""
    if not fv.get("smoke", True):
        return "Off"
    if layout is None or "error" in layout:
        return "—"
    if layout["kind"] == "plane":
        return "Plane band"
    n = int(layout["count"])
    return f"{RAKE_NOUN.get(layout['kind'], layout['kind'])}, {n:,} nozzle{'s' if n != 1 else ''}"


def smoke_note(layout: dict | None, fv: dict) -> str:
    """One line on where the smoke comes from: count, pitch, height and distance."""
    if not fv.get("smoke", True):
        return "Off: no tracer in this run."
    if layout is None:
        return ""
    if "error" in layout:
        return f"The smoke rake cannot be placed: {layout['error']}"
    lu = display_unit("length")

    def ln(v: float) -> str:
        return units.fmt(v, "length", lu, 3)

    (zl, zh), sp = layout["z"], layout["spacing"]
    where = f"{ln(abs(layout['x'] - layout.get('body_x0', layout['x'])))} ahead of the body"
    kind, n = layout["kind"], int(layout["count"])
    if kind == "plane":
        return f"A band from z = {ln(zl)} to {ln(zh)}, {where}."
    if kind == "rake":
        return f"{n} nozzles {ln(sp)} apart at z = {ln(zl)}, {where}."
    if kind == "vertical":
        return f"{n} nozzles {ln(sp)} apart from z = {ln(zl)} to {ln(zh)}, {where}."
    ny, nz = layout.get("n", (n, 1))
    return f"{ny} × {nz} = {n:,} nozzles {ln(sp)} apart, z = {ln(zl)} to {ln(zh)}, {where}."


#: a grid of more rows than this converts to one plane band rather than one rake per row
CONVERT_MAX_ROWS = 12


def emitters_from_layout(layout: dict | None, smoke_on: bool = True) -> list[Emitter]:
    """Emitters standing exactly where an automatic smoke rake (a :func:`smoke_rake_layout` dict)
    releases: a rake across the span or up the height becomes a rake emitter with its nozzle count
    and radius, a plane band a plane emitter facing the flow, a grid one rake per row (a plane band
    above :data:`CONVERT_MAX_ROWS` rows). Fixed in the tunnel, white smoke; inactive when the smoke
    was off. [] for a layout without a position."""
    if not layout or "error" in layout or "x" not in layout:
        return []
    x = float(layout["x"])
    (y0, y1), (z0, z1) = layout["y"], layout["z"]
    ny, nz = (int(v) for v in layout.get("n", (layout.get("count", 1), 1)))
    radius = float((layout.get("spec") or {}).get("radius") or 0.0)
    kind = layout["kind"]
    common = {"species": "dye", "frame": "wind", "enabled": bool(smoke_on), "radius": radius}
    if kind == "rake":
        return [Emitter(name="Smoke rake", kind="rake", position=(x, y0, z0), end=(x, y1, z0), count=max(ny, 1),
                        **common)]
    if kind == "vertical":
        yc = 0.5 * (y0 + y1)
        return [Emitter(name="Smoke rake", kind="rake", position=(x, yc, z0), end=(x, yc, z1), count=max(nz, 1),
                        **common)]
    if kind == "grid" and nz <= CONVERT_MAX_ROWS:
        zs = np.linspace(z0, z1, nz) if nz > 1 else [0.5 * (z0 + z1)]
        return [Emitter(name=f"Smoke row {i + 1}", kind="rake", position=(x, y0, float(z)), end=(x, y1, float(z)),
                        count=max(ny, 1), **common) for i, z in enumerate(zs)]
    return [Emitter(name="Smoke plane", kind="plane", position=(x, 0.5 * (y0 + y1), 0.5 * (z0 + z1)),
                    size=(y1 - y0, z1 - z0), normal=(1.0, 0.0, 0.0), **common)]


def frame_text(three_d: bool, closed: bool = False) -> str:
    """The preview's frame caption. A closed box has no free stream to come from the left: it says
    which way is up instead, the direction buoyancy acts against (DESIGN.md 10.2: −y in 2D, −z in 3D)."""
    if closed:
        return "Closed box · z up" if three_d else "Closed box · y up"
    return "Wind frame · flow along +x, z up" if three_d else "Wind frame · flow from the left"


class FrameCaption(_CommonElidedLabel):
    """The canvas tool strip's frame caption ("Closed box · y up"): elided at the right when the
    strip runs out of width, the full text in the tooltip.

    At 1280 x 800 with the Assistant and Script docks open the caption was cut mid-word to
    "Closed box · y" (the Atkinson face is wider than Segoe UI was); now it ends in an ellipsis
    and hovering shows the rest. ``text()`` answers the full caption, not the elided one shown,
    so code and tests that read it are independent of the window width.
    """

    def __init__(self, text: str = "") -> None:
        super().__init__(text, "muted", mode=Qt.TextElideMode.ElideRight)

    def text(self) -> str:
        return self.full_text()


def planner_count_text(lines: list[tuple[str, str]]) -> str:
    """The Planner header's count: "1 error", "4 warnings", "1 error, 2 warnings" or "" (an error was
    counted as a warning, so a refused setup read "1 warning" beside a red line)."""
    n_err = sum(1 for lv, _ in lines if lv == "error")
    n_warn = sum(1 for lv, _ in lines if lv == "warning")
    parts = [f"{n} {word}{'s' if n != 1 else ''}" for n, word in ((n_err, "error"), (n_warn, "warning")) if n]
    return ", ".join(parts)


def plan_failure_text(design: Design, sim: Simulation, exc: Exception) -> str:
    """The Planner line for a setup the installed planner refused.

    A closed cavity has no body on purpose (its walls drive the flow): a planner that only knows wind
    tunnels refuses it with "no bodies", which reads as a mistake in the case. Said as what it is - a
    limit of the installed planner, the case itself being complete and saved."""
    three = getattr(design, "dimension", 2) == 3
    bodies = bool(design.meshes if three else design.bodies) or design.kind == "wing"
    if not bodies and sim.cfd.domain_mode == "closed":
        return ("The installed planner needs a body and cannot plan a closed box without one yet. The case "
                f"is complete and saved; it can run once the planner supports closed domains (planner: "
                f"{str(exc).rstrip('.')}).")
    return f"This setup cannot be planned: {exc}"


def supported_physics(dimension: int) -> frozenset[str]:
    """The placed-physics features the installed planner and solvers run in ``dimension``.

    The simulation layer publishes them (``aerolume.simulation.SUPPORTED_PHYSICS``, read through
    :func:`aerolume.api.supported_physics`, the same answer a script gets); empty when it publishes
    nothing or cannot be imported, so every feature is then reported as not carried."""
    try:
        from ...api import supported_physics as api_supported
        return frozenset(api_supported(int(dimension)))
    except Exception:  # noqa: BLE001 - a missing or partial API means nothing new is supported
        return frozenset()


#: openings of the page's own physics lines (physics_editor.physics_warnings) that the planner states
#: in its own words once it plans heat transfer: the same fact twice, worded two ways, read as two
#: problems (the 3D room listed its Rayleigh number warning as a planner warning and a page note)
PLANNER_TOPICS = ("Ra = ", "β·ΔT = ", "The fluid is ", "Custom fluid", "Heat transfer is off, so",
                  "Heat transfer is on but nothing")


def said_by_planner(line: str, planner_lines: Sequence[str]) -> bool:
    """Whether the planner already says what the page's ``line`` says: same opening topic
    (:data:`PLANNER_TOPICS`), in any of its warnings or notes. A line on another topic, or any line
    when the planner says nothing on it (an older planner, a basic estimate), is kept."""
    topic = next((t for t in PLANNER_TOPICS if line.startswith(t)), None)
    return topic is not None and any(str(p).startswith(topic) for p in planner_lines)


def planner_gaps(sim: Simulation, plan, extent, rows, supported: frozenset[str] | None = None) -> list[str]:
    """Warnings for settings this planner did not carry into the run, measured on its plan.

    A Setup that shows a closed box, a fixed domain, heat transfer, fans or jets must not let a run
    quietly be a plain wind tunnel: the box is compared with the planned extent, heat transfer with
    the dimensionless groups the planner states in its summary (DESIGN.md 10.2: Ra, Pr, ... are part
    of ``summary()`` when it plans heat), and every placed item with the features the simulation
    layer says it runs (``supported``; :func:`supported_physics` when None) or a plan attribute of
    the same name. 3D emitters are carried as the seed rake (:func:`emitters_rake_spec`) until the
    solver reads them, which keeps their nozzles but not their colours, rates, times or particles:
    the line says which parts are dropped."""
    out: list[str] = []
    if plan is None or sim.analysis != "cfd":
        return out
    dim = 3 if getattr(plan, "dimension", 2) == 3 or getattr(plan, "extent3", None) is not None else 2
    have = supported_physics(dim) if supported is None else frozenset(supported)

    def carried(feature: str, attr: str | None = None) -> bool:
        return feature in have or (attr is not None and getattr(plan, attr, None) is not None)

    box = phys.explicit_box(sim, dim)
    if box is not None and extent is not None and len(extent) == len(box):
        size = max(box[2 * i + 1] - box[2 * i] for i in range(dim))
        if max(abs(a - b) for a, b in zip(box, extent, strict=True)) > 1e-3 * size:
            shown = " × ".join(f"{extent[2 * i + 1] - extent[2 * i]:.3g}" for i in range(dim))
            out.append(f"The installed planner ignores the fixed domain box: this run would use a {shown} m "
                       "domain (the dashed frame) instead.")
    if sim.cfd.domain_mode == "closed" and getattr(plan, "domain_mode", None) != "closed":
        out.append("The installed planner cannot plan a closed box yet: this run would be a wind tunnel with "
                   "inflow at the reference speed.")
    thermal_carried = True
    if sim.thermal.enabled:
        keys = " ".join(str(r[0]) for r in rows)
        thermal_carried = carried("thermal", "thermal") or any(
            w in keys for w in ("Rayleigh", "Prandtl", "Péclet", "Peclet"))
        if not thermal_carried:
            out.append("The installed planner cannot plan heat transfer yet: this run would solve the flow only. "
                       "The thermal numbers in the Estimate are computed by this page.")
    dropped = []
    if any(f.enabled for f in sim.fans) and not carried("fans", "fans"):
        dropped.append("fans")
    if not carried("ports", "ports"):
        dropped += [f"{k}s" for k in ("jet", "vent") if any(p.enabled and p.kind == k for p in sim.ports)]
    # heaters do nothing without heat transfer, which the line above already reports
    if thermal_carried and any(h.enabled for h in sim.heat_sources) and not carried("heat_sources", "heat_sources"):
        dropped.append("heat sources")
    emitters = [e for e in sim.emitters if e.enabled]
    if emitters and not carried("emitters", "emitters"):
        if dim == 2:
            dropped.append("emitters")
        else:
            lost = []
            if any(e.species == "particles" for e in emitters):
                lost.append("particles become smoke")
            if len({e.color.lower() for e in emitters}) > 1:
                lost.append("all colours become one")
            if any(e.rate != 1.0 or e.start > 0 or e.stop is not None for e in emitters):
                lost.append("rates and start / stop times are ignored")
            if lost:
                what = lost[0] if len(lost) == 1 else ", ".join(lost[:-1]) + " and " + lost[-1]
                out.append(f"The installed planner passes the emitters as one smoke rake at their nozzles: {what}.")
    if dropped:
        what = dropped[0] if len(dropped) == 1 else ", ".join(dropped[:-1]) + " and " + dropped[-1]
        out.append(f"The installed planner does not pass {what} to the solver yet: this run would leave them out.")
    return out


def emitters_fact(sim: Simulation) -> str:
    """The Estimate's "Smoke" value for a simulation with emitters: "Rake, 9 nozzles", "3 emitters", "Off"."""
    act = [e for e in sim.emitters if e.enabled]
    if not act:
        return "Off"
    if len(act) == 1:
        e = act[0]
        what = phys.EMITTER_KIND_TEXT.get(e.kind, e.kind).split(" (")[0]
        return f"{what}, {e.count} nozzles" if e.kind == "rake" else what
    particles = sum(1 for e in act if e.species == "particles")
    tail = f" ({particles} of particles)" if particles else ""
    return f"{len(act)} emitters{tail}"


def wing_estimate(design: Design, sim: Simulation, airfoil_coords=None) -> tuple[list, list[str], list[str]]:
    """(Estimate rows, notes, warnings) of a finite-wing analysis, without running it.

    The cost of a cold viscous run is the 2D section polars: one XFOIL-class
    polar per airfoil and Reynolds-number node, over the angles the sections
    see. The node set is what :class:`aerolume.solvers.wing.SectionPolars`
    would plan for the sections' chord Reynolds numbers (``plan`` / ``cost``,
    no computation), so the count matches the run; a polar point costs
    :data:`WING_POINT_S`. Section polars are cached between runs, so a second
    run of the study takes seconds.
    """
    from ...geometry3d import naca_code, planform
    rows: list = []
    notes: list[str] = []
    warnings: list[str] = []
    spec = design.wing
    if spec is None:
        return [], [], ["The wing design has no planform."]
    try:
        pf = planform(spec)
    except ValueError as exc:
        return [], [], [f"The planform is invalid: {exc}."]
    cond, ws = sim.conditions, sim.wing
    alphas = ws.alphas()
    def reyn(c: float) -> float:
        return cond.fluid.density * cond.speed * c / cond.fluid.viscosity

    chords = [s.chord for s in spec.sections]
    mach = mach_number(cond)
    rows.append(("Reynolds number (MAC)", units.fmt_reynolds(reyn(pf["mac"]))))
    rows.append(("Reynolds range", f"{units.fmt_reynolds(reyn(min(chords)))} – {units.fmt_reynolds(reyn(max(chords)))}"))
    rows.append(("Angles", f"{len(alphas)} ({_deg(alphas[0])} to {_deg(alphas[-1])})"))
    # short values: the Estimate column is ~170 px wide at 1280 px and elided the counts
    if ws.method == "vlm":
        rows.append(("Method", "Vortex lattice"))
        rows.append(("Panels per half", f"{spec.n_chord} × {spec.n_span}"))
    else:
        rows.append(("Method", "Lifting line"))
        rows.append(("Strips per half", f"{spec.n_span}"))
    rows.append(("Reference area", area_text(design.ref_area)))
    rows.append(("Reference chord", units.fmt(design.length_ref, "length", display_unit("length"), 4)))
    points = 0
    if ws.viscous:
        demands: dict[str, list] = {}
        tw = [s.twist_deg for s in spec.sections]
        lo, hi = float(alphas.min()) + min(tw) - 4.0, float(alphas.max()) + max(tw) + 4.0
        for s in spec.sections:
            demands.setdefault(str(s.airfoil), []).append((reyn(s.chord), lo, hi))
        try:
            from ...solvers.wing import SectionPolars
            sp = SectionPolars(ncrit=ws.ncrit, mach=ws.mach)
            todo = sp.plan(demands)
            points = sum(len(v) for v in todo.values())
            nodes = len(todo)
            n_af = len(demands)
            rows.append(("Section polars", f"{n_af} airfoil{'s' if n_af != 1 else ''} × {nodes} Re"))
            rows.append(("Polar points, first run", f"≈ {points:,}"))
        except Exception as exc:  # noqa: BLE001 - the estimate is advisory
            notes.append(f"(Section-polar count not available: {type(exc).__name__}.)")
        for s in spec.sections:
            a = str(s.airfoil)
            if a.lower().startswith("design:"):
                if airfoil_coords is None:
                    warnings.append(f"Section at y = {s.y:.3g} m uses a project airfoil that cannot be resolved here.")
                else:
                    try:
                        airfoil_coords(a)
                    except ValueError as exc:
                        warnings.append(str(exc).capitalize())
            elif naca_code(a) is None:
                warnings.append(f"Section at y = {s.y:.3g} m: airfoil {a!r} is not a NACA code.")
        notes.append(f"Profile drag: strip integration of 2D viscous section polars (Ncrit {ws.ncrit:g}) at each "
                     "section's Reynolds number; the first run computes them, later runs reuse them.")
    else:
        rows.append(("Profile drag", "None (inviscid: CD = induced drag)"))
        notes.append("Inviscid: the drag is the induced drag only; switch Viscous on for the profile drag.")
    est = WING_POINT_S * points + WING_ANGLE_S.get(ws.method, 0.1) * len(alphas) + 1.0
    rows.append(("Estimated run time", format_duration(est)))
    if mach is not None:
        rows.insert(2, ("Mach number", f"{mach:.3f}"))
    if ws.method == "llt" and (abs(pf["sweep_c4_deg"]) > 20 or pf["aspect_ratio"] < 4):
        warnings.append("The lifting line is not meant for quarter-chord sweep above 20° or an aspect ratio below "
                        "4; the vortex lattice handles this planform.")
    if ws.mach >= 0.6:
        warnings.append(f"Mach {ws.mach:.2f}: the Prandtl–Glauert correction is unreliable above about Mach 0.6.")
    elif mach is not None and mach > 0.3 and ws.mach == 0:
        notes.append(f"The flow speed is Mach {mach:.2f}; set the Mach number to include compressibility.")
    if reyn(min(chords)) < 5e4:
        warnings.append("A section Reynolds number is below 50,000: laminar separation bubbles dominate there and "
                        "the section polars are uncertain.")
    return rows, notes, warnings


_ISA_RE = re.compile(r"^Air, ISA ([\d,.]+) m$")
#: "C" or "°C" after the temperature: the core writes "C" today, and accepting the symbol keeps
#: projects readable whichever spelling a saved name (or a later core) uses
_AIR_T_RE = re.compile(r"^Air, (-?[\d.]+) °?C$")
_WATER_RE = re.compile(r"^Water, (-?[\d.]+) °?C$")

#: shared with the physics rows, so every on-screen fluid name spells degrees the same way
fluid_display_name = phys.fluid_display_name


def preset_key(name: str) -> str | None:
    """The :data:`aerolume.core.fluid.PRESETS` key a fluid name stands for, or None.

    Compared through :func:`fluid_display_name`, so "Air, 20 C" and "Air, 20 °C" find the same
    preset: a saved project and the fluid row then agree whichever spelling the name was saved in."""
    shown = fluid_display_name(name)
    return next((k for k in fluids.PRESETS if fluid_display_name(k) == shown), None)


def fluid_mode(f: Fluid) -> tuple[str, float | None]:
    """Recover how a Fluid was chosen from its name: ("preset", None),
    ("isa", altitude_m), ("air_t", temperature_c), ("water_t", temperature_c) or
    ("custom", None). Names are generated by :mod:`aerolume.core.fluid`."""
    key = preset_key(f.name)
    if key is not None:
        ref = fluids.PRESETS[key]()
        # loose tolerance: the model's default Fluid() stores rounded constants
        if math.isclose(ref.density, f.density, rel_tol=2e-3) and \
                math.isclose(ref.viscosity, f.viscosity, rel_tol=2e-3):
            return "preset", None
    for rx, mode in ((_ISA_RE, "isa"), (_AIR_T_RE, "air_t"), (_WATER_RE, "water_t")):
        m = rx.match(f.name)
        if m:
            return mode, float(m.group(1).replace(",", ""))
    return "custom", None


def make_fluid(mode: str, value: float | None, *, preset: str | None = None,
               density: float | None = None, viscosity: float | None = None) -> Fluid:
    """Build the Fluid for a fluid-row state (inverse of :func:`fluid_mode`)."""
    if mode == "preset" and preset in fluids.PRESETS:
        return fluids.PRESETS[preset]()
    if mode == "isa":
        return fluids.air_isa(float(value or 0.0))
    if mode == "air_t":
        return fluids.air(float(value if value is not None else 15.0))
    if mode == "water_t":
        return fluids.water(float(value if value is not None else 20.0))
    return Fluid(name="Custom fluid", density=float(density or 1.225), viscosity=float(viscosity or 1.8e-5))


def mach_number(cond: FlowConditions) -> float | None:
    """Free-stream Mach number for air (None for other fluids).

    The temperature is read back from the fluid name that :mod:`aerolume.core.fluid`
    generates (ISA altitude or "Air, T C"); 15 C is assumed otherwise. Only used
    to warn about compressibility, so that approximation is ample.
    """
    f = cond.fluid
    if not f.name.lower().startswith("air"):
        return None
    t_k = 288.15
    m = _ISA_RE.match(f.name)
    if m:
        t_k = fluids.isa(float(m.group(1).replace(",", "")))[0]
    else:
        m = _AIR_T_RE.match(f.name)
        if m:
            t_k = float(m.group(1)) + 273.15
    return cond.speed / fluids.speed_of_sound(t_k)


#: Polar sweep cost on one CPU core (the solver is serial): the first angle is cold-started
#: and takes longest, the rest are warm-started from a neighbour. Measured in the app, NACA
#: 2412 at Re 1e6, 19 angles (2026-10, i7-13700F): first angle 8.5 s, the sweep 30.4 s.
POLAR_FIRST_ANGLE_S = 8.5
POLAR_PER_ANGLE_S = 1.2


def polar_seconds(n_angles: int) -> float:
    """Rough wall time [s] of a polar sweep of ``n_angles`` angles (the Setup estimate).

    Without it the polar's Estimate table had no run time, and a sweep that sat at
    "0 of 19 angles" for its first nine seconds gave no hint that this was expected.
    """
    n = max(int(n_angles), 0)
    return 0.0 if n == 0 else POLAR_FIRST_ANGLE_S + POLAR_PER_ANGLE_S * (n - 1)


def format_duration(seconds: float) -> str:
    """Human run-time estimate: "~40 s", "~3 min", "~1 h 20 min"."""
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    if seconds < 90:
        return f"~{max(seconds, 1):.0f} s"
    if seconds < 3600:
        return f"~{seconds / 60:.0f} min"
    h, m = divmod(round(seconds / 60), 60)
    return f"~{h} h {m:02d} min"


def _deg(v: float) -> str:
    """An angle with a typographic minus: "−4°" (guidelines section 2: proper symbols)."""
    return f"{v:g}°".replace("-", "−")


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
_POW_RE = re.compile(r"10\^(-?\d+)")


def pretty(text: str) -> str:
    """Typographic clean-up for planner text: ``10^6`` -> ``10⁶``, `` -- `` -> `` — ``.

    The planner and :func:`aerolume.core.units.fmt_reynolds` write plain ASCII
    (they also feed logs and CSV files); the guidelines ask for proper symbols on
    screen (Re = 8.2×10⁵), so the conversion happens at display time only.
    """
    return _POW_RE.sub(lambda m: "10" + m.group(1).translate(_SUPERSCRIPT), text).replace(" -- ", " — ")


# a sentence ends at . ! ? followed by a capital or "(" - "i.e. 0.6" and "≈ 0.5" do not end one
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def planner_headline(text: str) -> str:
    """The first sentence of a planner note: the decision and its number.

    The planner writes each note as a decision followed by its rationale
    ("Domain: 2.50 chords upstream ... (2.6 % blockage) ... . The belt moves
    with ..."). The strip under the preview shows only the decision, one line
    per note; the full text is the line's tooltip and goes to the Messages
    dock, so nothing is lost.
    """
    t = pretty(str(text)).strip()
    return _SENTENCE_END.split(t, maxsplit=1)[0]


_FLOW_MODEL = re.compile(r"^(LES|Laminar)\b\s*(?:\(([^)]*)\))?\s*(?:,\s*([\w-]+)\s+advection)?", re.IGNORECASE)
ADVECTION_TEXT = {"upwind3": "third-order upwind advection", "central": "central-difference advection"}


def compact_flow_model(value: str) -> tuple[str, str]:
    """(short label, tooltip) for the planner's flow-model description.

    ``SimPlan.summary()`` writes "LES (WALE sub-grid model), upwind3 advection",
    which wraps the Estimate column and leaks the internal scheme name. The
    column shows the conventional short form ("LES-WALE", "Laminar", as on the
    Results header line); the tooltip spells it out in words.
    """
    m = _FLOW_MODEL.match(str(value).strip())
    if not m:
        return str(value), ""
    kind, inner, adv = m.groups()
    adv_text = ADVECTION_TEXT.get((adv or "").lower(), f"{adv} advection" if adv else "")
    if kind.upper() == "LES":
        model = (inner or "").split()[0].upper() if inner else ""
        named = bool(model) and model not in ("SUB-GRID", "SUBGRID")
        short = f"LES-{model}" if named else "LES"
        tip = ("Large-eddy simulation: " + (f"the {model} sub-grid model" if named else "a sub-grid model")
               + " represents the eddies smaller than the grid")
    else:
        short = "Laminar"
        tip = "Laminar: every scale of the flow is resolved by the grid, no turbulence model"
    return short, tip + (f"; {adv_text}." if adv_text else ".")


# "0.000882 L/U (138 µs)" and "30 L/U (4.70 s of flow)": a dimensionless value
# with its real-time equivalent in brackets
_WITH_REAL_TIME = re.compile(r"^(?P<main>.*?L/U)\s*\((?P<real>[^()]*?)(?:\s+of flow)?\)\s*$")
_REAL_TIME_LABEL = {"Time step": "Real time step", "Duration": "Real flow time"}
L_OVER_U_TIP = "In convective time units L/U: the time the flow takes to pass one reference length"
_STOP_RULE = re.compile(r"^Cd known to (?P<cd>±[^,]+),\s*Cl to (?P<cl>±.+)$")
_VENDOR = re.compile(r"^(?:NVIDIA\s+GeForce|NVIDIA|AMD\s+Radeon|AMD|Intel\(R\))\s+", re.IGNORECASE)
_THREADS = re.compile(r"(\d+)\s+solver threads?")


def compact_device(value: str) -> str:
    """"GPU · NVIDIA GeForce RTX 4060 Ti · float32" -> "RTX 4060 Ti · float32";
    "CPU · <processor id> · 8 solver threads · float64" -> "CPU · 8 threads · float64"."""
    parts = [p.strip() for p in str(value).split(" · ")]
    precision = parts.pop() if len(parts) > 1 and parts[-1].startswith("float") else ""
    kind = parts[0].upper() if parts else ""
    if kind == "GPU" and len(parts) > 1:
        head = _VENDOR.sub("", parts[1], count=1)
    elif kind == "CPU":
        m = _THREADS.search(" ".join(parts))
        head = f"CPU · {m.group(1)} thread{'s' if m.group(1) != '1' else ''}" if m else "CPU"
    else:
        return str(value)
    return f"{head} · {precision}" if precision else head


def compact_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Estimate rows as (label, shown value, tooltip), every value short enough
    to read in full in a narrow column.

    The planner's ``SimPlan.summary()`` writes sentences ("0.000882 L/U
    (138 µs)", "Cd known to ±0.5 %, Cl to ±1 %", "NVIDIA GeForce RTX 4060 Ti ·
    float32") that were elided at 1280 px wide, hiding the number the row
    exists for. Here a value with a real-time equivalent becomes two rows
    (Time step / Real time step, Duration / Real flow time), the stop rule
    keeps only its tolerances, the device drops the vendor brand, and the
    flow model takes its conventional short name. The long form, where one
    exists, is the row's tooltip.
    """
    out: list[tuple[str, str, str]] = []
    for row in rows:
        k, v = row[0], str(row[1])
        given = str(row[2]) if len(row) > 2 and row[2] else ""
        m = _WITH_REAL_TIME.match(v) if k in _REAL_TIME_LABEL else None
        if k == "Flow model":
            short, tip = compact_flow_model(v)
            out.append((k, short, tip))
        elif m:
            out.append((k, m["main"].strip(), L_OVER_U_TIP))
            out.append((_REAL_TIME_LABEL[k], m["real"].strip(), f"{k} in seconds at the real speed"))
        elif k == "Stops early when" and (s := _STOP_RULE.match(v)):
            out.append((k, f"Cd {s['cd'].strip()}, Cl {s['cl'].strip()}", v))
        elif k == "Device":
            short = compact_device(v)
            out.append((k, short, v if short != v else ""))
        elif k == "Cells (3D)" and (c := _HALF_CELLS.match(v)):
            out.append((k, f"{c['half']} (half domain)", f"{c['half']} cells solved; {c['full']} for both halves"))
        elif k in ("Blockage", "Symmetry plane") and " (" in v and v.endswith(")"):
            main, rest = v.split(" (", 1)
            out.append((k, main, rest[:-1][:1].upper() + rest[:-1][1:]))
        else:
            out.append((k, v, given))
    return out


_HALF_CELLS = re.compile(r"^(?P<half>[\d.]+ M) solved \(half domain; (?P<full>[\d.]+ M) for both halves\)$")

#: Estimate rows read first, in this order: what the run costs and where it runs. The
#: planner's summary leads with the Reynolds numbers, dynamic pressure and Mach number,
#: which left "Estimated run time" and "Device" (the numbers that decide whether to
#: press Run) below the fold of the strip at 1280 x 800 and at 1600 x 1000.
ESTIMATE_LEAD = ("Grid", "Cells (3D)", "Cells", "Estimated run time", "Device",
                 "Real Reynolds number", "Simulated Reynolds number")


def order_estimate(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """CFD Estimate rows with :data:`ESTIMATE_LEAD` first; the rest keep their order.

    Rows the planner did not write (a plan that failed, a planner still loading)
    are simply absent, so a partial table keeps the same reading order.
    """
    rank = {k: i for i, k in enumerate(ESTIMATE_LEAD)}
    lead = sorted((r for r in rows if r[0] in rank), key=lambda r: rank[r[0]])
    return lead + [r for r in rows if r[0] not in rank]


# ---------------------------------------------------------------------------
# planner import on a background thread (it imports torch)
# ---------------------------------------------------------------------------

class _PlannerLoader(QObject):
    """Imports ``aerolume.simulation.plan`` once, off the GUI thread."""

    loaded = Signal()
    _instance: _PlannerLoader | None = None

    def __init__(self) -> None:
        super().__init__()
        self.module = None
        self.error: str | None = None
        self.state = "idle"                  # idle | loading | ready | missing
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> _PlannerLoader:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        with self._lock:
            if self.state != "idle":
                return
            self.state = "loading"
        threading.Thread(target=self._run, name="aerolume-plan-import", daemon=True).start()

    def _run(self) -> None:
        try:
            from ...simulation import plan as module
            self.module = module
            self.state = "ready" if hasattr(module, "plan_simulation") else "missing"
        except Exception as exc:  # noqa: BLE001 - any import failure means "no planner"
            self.error = str(exc)
            self.state = "missing"
        self.loaded.emit()                   # queued to the GUI thread

    def wait(self, timeout: float = 60.0) -> bool:
        """Block until the import finished (tests and headless rendering)."""
        import time
        t0 = time.monotonic()
        while self.state in ("idle", "loading") and time.monotonic() - t0 < timeout:
            if self.state == "idle":
                self.start()
            time.sleep(0.02)
        return self.state == "ready"


# ---------------------------------------------------------------------------
# small widgets
# ---------------------------------------------------------------------------

def _spin(minimum: float, maximum: float, step: float, decimals: int, suffix: str = "",
          special: str | None = None) -> QDoubleSpinBox:
    """A right-aligned decimal spin box; ``special`` shows that text at the minimum ("Auto")."""
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
    return s


def _combo(items) -> QComboBox:
    """A combo of (key, text[, tooltip]) items; the key is the item data."""
    c = QComboBox()
    for item in items:
        c.addItem(item[1], item[0])
        if len(item) > 2:
            c.setItemData(c.count() - 1, item[2], Qt.ItemDataRole.ToolTipRole)
    return c


ROW_BUTTON_W = 64        # the "Auto" switches, as wide as the Geometry inspector's row buttons


def _auto_toggle(tooltip: str) -> QToolButton:
    """A checkable "Auto" switch at the end of a row (the Geometry References pattern)."""
    b = QToolButton()
    b.setText("Auto")
    b.setCheckable(True)
    b.setProperty("variant", "toggle")
    b.setToolTip(tooltip)
    b.setFixedSize(ROW_BUTTON_W, 24)
    return b


def _row_spacer() -> QWidget:
    """An empty cell as wide as an Auto switch, so editors in rows without one line up."""
    w = QLabel()
    w.setFixedWidth(ROW_BUTTON_W)
    return w


def _set_combo(c: QComboBox, key: Any) -> None:
    c.blockSignals(True)
    c.setCurrentIndex(max(c.findData(key), 0))
    c.blockSignals(False)


def _set_silent(w: QWidget, value: Any) -> None:
    """Set a spin box / check box without emitting (loading from the model)."""
    w.blockSignals(True)
    try:
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(value)
    finally:
        w.blockSignals(False)


class FactTable(QWidget):
    """The Estimate: key (text2) | value rows, values right-aligned on one line.

    Rows are ``(key, value)`` or ``(key, value, tooltip)``. A value too long
    for the column (the device name) is elided rather than wrapped, so every
    row stays one line and the column reads as a table; the full value, or
    the given tooltip, is shown on hover.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(3)
        self.grid.setColumnStretch(1, 1)
        self._items: list[tuple[str, str]] = []
        self._tips: dict[str, str] = {}

    def set_items(self, items: list[tuple]) -> None:
        rows = [(str(it[0]), str(it[1]), str(it[2]) if len(it) > 2 and it[2] else "") for it in items]
        self._items = [(k, v) for k, v, _t in rows]
        self._tips = {k: t for k, _v, t in rows if t}
        clear_layout(self.grid)
        for r, (k, v, tip) in enumerate(rows):
            kl = label(k, "secondary")
            kl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            vl = ElidedLabel(pretty(v), tip or None)
            vl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            vl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if tip:
                kl.setToolTip(tip)
            self.grid.addWidget(kl, r, 0)
            self.grid.addWidget(vl, r, 1)
        self.grid.setRowStretch(len(rows), 1)

    def items(self) -> list[tuple[str, str]]:
        return list(self._items)

    def row_pitch(self) -> int:
        """Height of one row plus the spacing to the next (from the row widgets,
        whose style sheet role may set a font other than the table's own)."""
        heights = [self.grid.itemAt(i).widget().sizeHint().height() for i in range(self.grid.count())
                   if self.grid.itemAt(i).widget() is not None]
        return max(heights or [self.fontMetrics().height()]) + self.grid.verticalSpacing()

    def height_for_rows(self, n: int) -> int:
        """Pixel height that shows exactly the first ``n`` rows (no partial row)."""
        n = max(0, min(int(n), len(self._items)))
        return n * self.row_pitch() - (self.grid.verticalSpacing() if n else 0)

    def value_text(self, key: str) -> str | None:
        return dict(self._items).get(key)

    def tooltip(self, key: str) -> str:
        return self._tips.get(key, "")


_LINE_GLYPH = {"error": ("error", "bad"), "warning": ("warning", "warn"), "info": ("info", "text2")}


class PlannerLine(QWidget):
    """One planner line: a small level glyph and the note's first sentence.

    Notes wrap to at most two lines and are elided after that, so the strip
    reads as a list of decisions whose numbers are visible without hovering
    (one elided line cut most of them off at 1280 px); warnings and errors
    wrap fully, because their first sentence is the problem itself. The full
    text is the tooltip (and is logged to Messages).
    """

    def __init__(self, text: str, level: str = "info", parent: QWidget | None = None, *,
                 action: tuple[str, str, Any] | None = None) -> None:
        super().__init__(parent)
        self.full_text = pretty(str(text))
        self.headline = planner_headline(text)
        self._level = level if level in _LINE_GLYPH else "info"
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.glyph = QLabel()
        self.glyph.setFixedSize(16, 16)
        lay.addWidget(self.glyph, 0, Qt.AlignmentFlag.AlignTop)
        tip = self.full_text if self.full_text != self.headline else None
        if action is not None:
            # a line that carries its fix reads in full: the sentence says why the button is there
            self.text = label(self.full_text, None, wrap=True)
        elif self._level == "info":
            self.text = ElidedLabel(self.headline, tip, max_lines=2)
        else:
            self.text = label(self.headline, None, wrap=True)
            self.text.setToolTip(tip or "")
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        #: the fix offered beside the problem ("Switch to Closed box"), or None
        self.action_button: QPushButton | None = None
        if action is None:
            lay.addWidget(self.text, 1)
        else:
            text_col = QVBoxLayout()
            text_col.setContentsMargins(0, 0, 0, 0)
            text_col.setSpacing(6)
            text_col.addWidget(self.text)
            act_label, act_tip, act_fn = action
            self.action_button = small_button(act_label, act_tip)
            self.action_button.setObjectName("planner_fix")
            self.action_button.setAccessibleName(act_label)
            self.action_button.setAccessibleDescription(act_tip)
            self.action_button.clicked.connect(act_fn)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self.action_button)
            row.addStretch(1)
            text_col.addLayout(row)
            lay.addLayout(text_col, 1)
        self._retint()
        theme.changed.connect(self._retint)        # bound method: dropped with the widget

    def _retint(self, *_a) -> None:
        name, colour = _LINE_GLYPH[self._level]
        with contextlib.suppress(Exception):          # the glyph is decoration
            self.glyph.setPixmap(theme.render_icon(name, 14, colour, self.devicePixelRatioF()))


def _section_head(title: str) -> tuple[QWidget, QLabel]:
    """A 24 px strip header ("Estimate", "Planner") with a right-aligned muted note."""
    w = QWidget()
    w.setFixedHeight(28)
    h = QHBoxLayout(w)
    h.setContentsMargins(12, 0, 12, 0)
    h.setSpacing(8)
    h.addWidget(label(title, "h3"))
    h.addStretch(1)
    note = label("", "muted")
    h.addWidget(note)
    return w, note


def _scroll(inner: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.Shape.NoFrame)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sc.setWidget(inner)
    return sc


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

class SetupPage(QWidget):
    """The Setup step (DESIGN.md 5.8): case preview, Estimate, Planner, property inspector."""

    SAVE_DELAY_MS = 350
    SUMMARY_DELAY_MS = 150
    LOG_DELAY_MS = 900           # planner lines are logged once the user pauses

    def __init__(self, state: AppState, jobs: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.jobs = jobs
        self._loading = False
        self._dirty = False
        self._sim: Simulation | None = None          # the instance the form edits
        self.last_plan = None                        # most recent SimPlan (or None)
        self.last_smoke: dict | None = None          # smoke emitters of the shown 3D case (smoke_rake_layout)
        self.last_summary: dict[str, str] = {}
        self.plan_status = ""                        # "ok" | "error" | "loading" | "basic" | "polar" | ""
        self.planner_lines: list[tuple[str, str]] = []   # (level, text) shown in the Planner strip
        #: what stops the current simulation from running, with its fix (:func:`run_blocker`), or None
        self.run_blocker: RunBlocker | None = None
        self._line_actions: dict[str, tuple[str, str, Any]] = {}  # Planner line text -> (button, tooltip, slot)
        self._logged: dict[str, set[str]] = {}       # sim id -> planner lines already logged
        self._progress: dict[str, float] = {}
        self._auto_created_id = ""                   # simulation on_shown created (see handle_action)
        self._physics_sel: tuple | None = None       # the placed item selected (scene_specs key)
        self._physics_sim_id = ""                    # whose items the selection belongs to
        self._domain_mode_shown = "tunnel"           # domain type the form was last laid out for
        self._domain_boxed_shown = False             # ... and whether its extent was fixed
        self._save_timer = self._timer(self.SAVE_DELAY_MS, self.flush)
        self._summary_timer = self._timer(self.SUMMARY_DELAY_MS, self.update_summary)
        self._summary_stale = False                  # a recompute was skipped while hidden
        self._log_timer = self._timer(self.LOG_DELAY_MS, self.flush_planner_log)
        self._fit_timer = self._timer(0, self.fit_info_height)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._pages = QStackedWidget()
        root.addWidget(self._pages)
        self.empty = EmptyState()
        self._empty_target = "start"
        self.empty.actionTriggered.connect(self._empty_action)
        self._pages.addWidget(self.empty)
        self._pages.addWidget(self._build_main())
        guard_wheel(self)

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.flush)
            app.aboutToQuit.connect(self.release_3d)
        self._planner = _PlannerLoader.get()
        self._planner.loaded.connect(self._schedule_summary)
        self._planner.start()

        state.projectChanged.connect(self._on_project_changed)
        state.designsChanged.connect(self._on_designs_changed)
        state.currentDesignChanged.connect(self._on_design_changed)
        state.currentSimulationChanged.connect(self._on_sim_changed)
        state.simulationsChanged.connect(self._on_sims_changed)
        if jobs is not None:
            for sig in ("jobQueued", "jobStarted", "jobFinished", "jobFailed", "jobCancelled"):
                s = getattr(jobs, sig, None)
                if s is not None:
                    s.connect(self._on_job_event)
            prog = getattr(jobs, "jobProgress", None)
            if prog is not None:
                prog.connect(self._on_job_progress)
        self._on_project_changed()

    def release_3d(self) -> None:
        """Free the 3D case view's OpenGL context now (window close), not at interpreter exit."""
        with contextlib.suppress(Exception):
            self.preview3d.scene.close()

    def _timer(self, ms: int, slot) -> QTimer:
        t = QTimer(self)
        t.setSingleShot(True)
        t.setInterval(ms)
        t.timeout.connect(slot)
        return t

    # ===================================================================
    # construction
    # ===================================================================
    def _build_main(self) -> QWidget:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setObjectName("setup_split")
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        split.addWidget(self._build_centre())
        split.addWidget(self._build_inspector())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1000, INSPECTOR_WIDTH])
        return split

    def _build_centre(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = ToolStrip()
        bar.setObjectName("setup_tools")
        self.fit_button = bar.add_button("fullscreen", "Fit", "Fit the view (F, or double-click)")
        bar.add_separator()
        # 3D camera presets (hidden for 2D designs)
        self.view_buttons: dict[str, Any] = {}
        for key, text, tip in (("iso", "Iso", "Isometric view from upstream, left and above"),
                               ("front", "Front", "Look downstream at the frontal area (along +x)"),
                               ("side", "Side", "Side view, flow from left to right"),
                               ("top", "Top", "Plan view from above"),
                               ("back", "Back", "Look upstream at the base and the wake side (along −x)")):
            self.view_buttons[key] = bar.add_button(None, text, tip,
                                                    on_click=lambda k=key: self.preview3d.set_view(k))
        self._view_sep = bar.add_separator()
        self.domain_toggle = bar.add_button(None, "Domain", "Frame the whole flow domain instead of the body",
                                            checkable=True)
        self.domain_toggle.setIcon(svg_icon("domain", active="accent"))
        self.domain_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.grid_toggle = bar.add_button("grid", "Grid", "Show the grid", checkable=True)
        self.grid_toggle.setChecked(True)
        self.physics_toggle = bar.add_button("layers", "Physics", "Show and edit the emitters, heat sources, fans, "
                                             "jets, vents and the domain box in the preview", checkable=True)
        self.physics_toggle.setObjectName("physics_layer")
        self.physics_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.physics_toggle.setChecked(True)
        bar.add_stretch()
        self.frame_label = FrameCaption("Wind frame · flow from the left")
        bar.add_widget(self.frame_label)
        bar.row.addSpacing(4)
        lay.addWidget(bar)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.setObjectName("setup_vsplit")
        vsplit.setChildrenCollapsible(False)
        vsplit.setHandleWidth(1)
        self.preview = CasePreview()
        self.preview.setObjectName("case_preview")
        self.preview.setAccessibleName("Case preview")
        self.preview.setMinimumHeight(180)
        self.preview3d = CasePreview3D()
        self.preview3d.setObjectName("case_preview3d")
        self.preview3d.setAccessibleName("3D case preview")
        self.preview3d.setMinimumHeight(180)
        self.preview3d.smokeClicked.connect(self.focus_flow_viz)
        self._preview_stack = QStackedWidget()
        self._preview_stack.addWidget(self.preview)
        self._preview_stack.addWidget(self.preview3d)
        vsplit.addWidget(self._preview_stack)
        vsplit.addWidget(self._build_info())
        vsplit.setStretchFactor(0, 1)
        vsplit.setStretchFactor(1, 0)
        vsplit.setSizes([600, INFO_HEIGHT])
        self._vsplit = vsplit
        # the strip is sized to whole Estimate rows until the user drags the
        # splitter (setSizes does not emit splitterMoved, so this only sees the user)
        self._info_user_sized = False
        vsplit.splitterMoved.connect(self._info_split_moved)
        lay.addWidget(vsplit, 1)

        self.fit_button.clicked.connect(self._fit_preview)
        self.domain_toggle.toggled.connect(self.preview.set_fit_domain)
        self.domain_toggle.toggled.connect(self.preview3d.set_fit_domain)
        self.grid_toggle.toggled.connect(self.preview.set_show_grid)
        self.physics_toggle.toggled.connect(self.set_physics_layer)
        for view in (self.preview, self.preview3d):
            view.physicsSelected.connect(self._preview_selected)
            view.physicsMoved.connect(self._preview_moved)
            view.physicsDropped.connect(self._preview_dropped)
        self._set_preview_mode(False)
        return w

    def _fit_preview(self) -> None:
        if self._preview_stack.currentIndex() == 1:
            self.preview3d.fit()
        else:
            self.preview.fit_view()

    def _set_preview_mode(self, three_d: bool, *, closed: bool = False) -> None:
        """2D canvas or 3D case view, with the tool strip that belongs to it."""
        self._preview_stack.setCurrentIndex(1 if three_d else 0)
        for b in self.view_buttons.values():
            b.setVisible(three_d)
        self._view_sep.setVisible(three_d)
        self.grid_toggle.setVisible(not three_d)
        self.frame_label.setText(frame_text(three_d, closed))
        self._fit_timer.start()                 # the preview's share of the height depends on the mode

    def _build_info(self) -> QWidget:
        """The strip under the preview: Estimate (left) and Planner messages (right)."""
        strip = QFrame()
        strip.setProperty("panel", "true")
        strip.setObjectName("setup_info")
        strip.setMinimumHeight(120)
        self._info = strip
        h = QHBoxLayout(strip)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        est = QWidget()
        el = QVBoxLayout(est)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(0)
        head, self.estimate_note = _section_head("Estimate")
        el.addWidget(head)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(12, 0, 12, 8)
        bl.setSpacing(4)
        self.estimate = FactTable()
        self.estimate.setObjectName("estimate")
        bl.addWidget(self.estimate)
        bl.addStretch(1)
        el.addWidget(_scroll(body), 1)
        # equal halves: at 4:5 the Estimate's value column was ~95 px at 1280 px
        # wide and elided "0.000882 L/U"; the planner lines wrap instead. Narrower than
        # INFO_STACK_WIDTH the two stack (:meth:`_sync_info_direction`).
        h.addWidget(est, 1)
        self._info_box = h
        self._info_vdiv = Divider(vertical=True)
        self._info_hdiv = Divider()
        self._info_hdiv.hide()
        h.addWidget(self._info_vdiv)
        h.addWidget(self._info_hdiv)

        pl = QWidget()
        pv = QVBoxLayout(pl)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        head, self.planner_note = _section_head("Planner")
        pv.addWidget(head)
        lines = QWidget()
        self._planner_lines_widget = lines
        self.planner_box = QVBoxLayout(lines)
        self.planner_box.setContentsMargins(12, 0, 12, 8)
        self.planner_box.setSpacing(6)
        self.planner_box.addStretch(1)
        pv.addWidget(_scroll(lines), 1)
        h.addWidget(pl, 1)
        self._info_head_h = head.minimumHeight()
        self._info_resize = _OnResize(strip, self._sync_info_direction)
        return strip

    def _sync_info_direction(self) -> None:
        """Side by side when the strip is wide, Estimate above Planner when it is narrow.

        At 1280 x 800 in the Analysis workspace (Project and Assistant open) the strip is about
        370 px wide: side by side, every Estimate value elided to "5…" and every Planner line
        after a few words (critic round 3.4). Stacked, each gets the whole width and scrolls.
        A 20 px band between the two thresholds keeps the layout from flipping back and forth."""
        w = self._info.width()
        if w <= 0:
            return
        vertical = self._info_box.direction() == QBoxLayout.Direction.TopToBottom
        narrow = w < (INFO_STACK_WIDTH + 20 if vertical else INFO_STACK_WIDTH)
        if narrow == vertical:
            return
        self._info_box.setDirection(QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight)
        self._info_vdiv.setVisible(not narrow)
        self._info_hdiv.setVisible(narrow)
        self._fit_timer.start()

    def info_stacked(self) -> bool:
        """Whether the Estimate sits above the Planner (a narrow strip)."""
        return self._info_box.direction() == QBoxLayout.Direction.TopToBottom

    def _build_inspector(self) -> QWidget:
        panel = QFrame()
        panel.setProperty("panel", "true")
        panel.setObjectName("setup_inspector")
        panel.setMinimumWidth(INSPECTOR_MIN_WIDTH)
        panel.setMaximumWidth(560)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(30)
        hl = QHBoxLayout(head)
        hl.setContentsMargins(12, 0, 6, 0)
        hl.setSpacing(8)
        self.title_label = label("Simulation", "h3")
        self.title_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status_label = label("", "muted")
        self.more_button = IconButton("more", "Simulation actions")
        # the menu is opened by hand rather than with setMenu(): a tool button with a
        # menu grows a drop-down arrow, which an overflow "..." button must not have
        self.more_menu = QMenu(self)
        self.more_menu.addAction(theme.icon("plus"), "New simulation", self.new_simulation)
        self.more_menu.addAction(theme.icon("copy"), "Duplicate", self.duplicate_simulation)
        # starter cases with heat, fans, jets and vents: a new design and its simulation
        self.template_menu = self.more_menu.addMenu(theme.icon("new"), "New case from template")
        self.template_menu.setObjectName("physics_templates")
        for t in PHYSICS_TEMPLATES:
            act = self.template_menu.addAction(theme.icon(t.icon), t.title,
                                               lambda k=t.key: self.new_from_template(k))
            act.setToolTip(t.description)
            act.setStatusTip(t.description)
        self.template_menu.setToolTipsVisible(True)
        self.more_menu.addSeparator()
        self.more_menu.addAction(theme.icon("trash"), "Delete…", self.delete_simulation)
        self.more_button.clicked.connect(
            lambda: self.more_menu.exec(self.more_button.mapToGlobal(self.more_button.rect().bottomLeft())))
        hl.addWidget(self.title_label, 1)
        hl.addWidget(self.status_label)
        hl.addWidget(self.more_button)
        lay.addWidget(head)
        lay.addWidget(Divider())

        self.grid = PropertyGrid(label_width=LABEL_W)
        self.grid.setObjectName("setup_properties")
        self._build_analysis_group()
        self._build_flow_group()
        # right under Flow: the smoke the preview draws is set without scrolling past the
        # attitude, road and mesh rows (it was the last group and needed scrolling at both sizes)
        self._build_flowviz_group()
        self._build_attitude_group()
        self._build_ground_group()
        # placed physics (DESIGN.md 10): what surrounds the body, heat, and what drives the flow
        self._build_physics_groups()
        self._build_mesh_group()
        self._build_solver_group()
        self._build_diagnostics_group()
        self._build_polar_group()
        self._build_wing_group()
        self.grid.add_stretch()
        # a composite row (a speed beside its unit, a height beside its Auto switch) has its
        # explanation on the label and the container; the part that takes the focus gets it as its
        # accessible description, as the physics rows do (physics_editor.describe_parts)
        for grp in self.grid.groups.values():
            for row in grp.rows.values():
                phys.describe_parts(row.editor, row.label.toolTip())
        self._form_widget = self.grid
        self.inspector_scroll = self.grid.scrollable()
        self.inspector_scroll.setObjectName("setup_inspector_scroll")
        lay.addWidget(self.inspector_scroll, 1)
        self._inspector_min = _FormMinWidth(panel, self.inspector_scroll, self.grid, INSPECTOR_MIN_WIDTH)

        lay.addWidget(Divider())
        foot = QWidget()
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(12, 8, 12, 8)
        fl.setSpacing(8)
        fl.addStretch(1)
        self.live_button = QPushButton("Explore live")
        self.live_button.setObjectName("explore_live")
        self.live_button.setToolTip("Real-time flow you can steer: change speed and angle while it runs")
        self.live_button.clicked.connect(self.explore_live)
        self.run_button = QPushButton("Run")
        self.run_button.setObjectName("run_simulation")
        self.run_button.setProperty("variant", "primary")
        self.run_button.setIcon(theme.icon("run", "on_accent"))
        self.run_button.setToolTip("Queue this simulation and show it on the Simulate page (F5)")
        self.run_button.setMinimumWidth(88)
        self.run_button.clicked.connect(self.run_simulation)
        fl.addWidget(self.live_button)
        fl.addWidget(self.run_button)
        lay.addWidget(foot)
        return panel

    # -- property groups ------------------------------------------------------
    def _build_analysis_group(self) -> None:
        g = self.grid.add_group("Analysis", key="analysis")
        self.analysis_group = g
        self.name_edit = keep_start_visible(QLineEdit())   # a long name shows its start, not its tail
        self.name_edit.setObjectName("sim_name")
        self.name_edit.editingFinished.connect(self._name_edited)
        g.add_row("Name", self.name_edit)
        self.analysis = SegmentedButtons([(k, t) for k, t, _ in ANALYSIS_ITEMS])
        self.analysis.setObjectName("analysis")
        for k, _t, tip in ANALYSIS_ITEMS:
            self.analysis.buttons[k].setToolTip(tip)
        self.analysis.currentChanged.connect(self._set_analysis)
        self.analysis_buttons = self.analysis.buttons
        # a meshed 3D design has one analysis: a plain fact instead of a one-segment control
        self.analysis_fixed = label("3D CFD (Navier–Stokes)", None)
        self.analysis_fixed.setObjectName("analysis_fixed")
        self.analysis_fixed.setToolTip(ANALYSIS_ITEMS[0][2])
        self.analysis_fixed.setVisible(False)
        g.add_row("Type", hbox(self.analysis, self.analysis_fixed, stretch=0))

    def _build_flow_group(self) -> None:
        g = self.grid.add_group("Flow", key="flow")
        self.flow_group = g
        self.fluid_combo = QComboBox()
        self.fluid_combo.setObjectName("fluid_combo")
        for name in fluids.PRESETS:
            self.fluid_combo.addItem(fluid_display_name(name), f"preset:{name}")
        self.fluid_combo.insertSeparator(self.fluid_combo.count())
        # item data are plain strings ("preset:<name>" or a mode): Qt cannot
        # findData() Python tuples reliably
        self.fluid_combo.addItem("Air at altitude (ISA)", "isa")
        self.fluid_combo.addItem("Air at temperature", "air_t")
        self.fluid_combo.addItem("Water at temperature", "water_t")
        self.fluid_combo.addItem("Custom", "custom")
        self.fluid_combo.currentIndexChanged.connect(self._fluid_mode_changed)
        g.add_row("Fluid", self.fluid_combo)
        self.altitude = QuantitySpinBox("length", "m", minimum=-1000, maximum=20000, step=100, decimals=0)
        self.altitude.setObjectName("altitude")
        self.altitude.siValueChanged.connect(lambda _v: self._fluid_edited())
        self._alt_row = g.add_row("Altitude", self.altitude, tooltip="Geometric altitude, ISA standard day")
        self.temperature = QuantitySpinBox("temperature", "C", minimum=-50, maximum=100, step=1, decimals=1)
        self.temperature.setObjectName("temperature")
        self.temperature.siValueChanged.connect(lambda _v: self._fluid_edited())
        self._temp_row = g.add_row("Temperature", self.temperature)
        self.density = _spin(1e-3, 2e4, 0.01, 4, " kg/m³")
        self.density.setObjectName("density")
        self.viscosity = _spin(1e-3, 1e7, 0.1, 3, " µPa·s")
        self.viscosity.setObjectName("viscosity")
        self.density.valueChanged.connect(lambda _v: self._fluid_edited())
        self.viscosity.valueChanged.connect(lambda _v: self._fluid_edited())
        g.add_row("Density", self.density)
        g.add_row("Viscosity", self.viscosity, tooltip="Dynamic viscosity (air ≈ 18 µPa·s, water ≈ 1000 µPa·s)")
        # no suffix: the unit selector beside the box already says the unit; three significant figures
        # for slow flows (a Re = 40 cylinder of 20 mm runs at 0.0301 m/s, which two decimals hid)
        self.speed = QuantitySpinBox("speed", display_unit("speed"), minimum=0.01, maximum=400.0, step=1.0,
                                     show_suffix=False, significant=3)
        self.speed.setObjectName("speed")
        self.speed.siValueChanged.connect(self._speed_edited)
        self.speed_unit = QComboBox()
        self.speed_unit.setObjectName("speed_unit")
        for u in SPEED_UNITS:
            self.speed_unit.addItem(u, u)
        _set_combo(self.speed_unit, self.speed.unit)
        self.speed_unit.setToolTip("Display unit (the value is kept)")
        self.speed_unit.currentIndexChanged.connect(
            lambda _i: self.speed.set_unit(str(self.speed_unit.currentData())))
        self.speed_unit.setFixedWidth(72)
        self._speed_row = g.add_row("Speed", hbox(self.speed, self.speed_unit), tooltip=SPEED_TIP)

    def _build_attitude_group(self) -> None:
        g = self.grid.add_group("Attitude", key="attitude")
        self.attitude_group = g
        self.alpha = QuantitySpinBox("angle", "deg", minimum=-30, maximum=30, step=0.5, decimals=1)
        self.alpha.setObjectName("alpha")
        self.alpha.setFixedWidth(84)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setObjectName("alpha_slider")
        self.alpha_slider.setRange(-60, 60)                 # half-degree steps
        self.alpha_slider.setPageStep(10)
        self.alpha_slider.valueChanged.connect(self._alpha_slider_moved)
        self.alpha.siValueChanged.connect(self._alpha_edited)
        self._alpha_row = g.add_row("Angle of attack", hbox(self.alpha, self.alpha_slider, stretch=1, spacing=8))
        # 3D: sideslip, applied by yawing the geometry about +z through the moment point (DESIGN.md 7.2)
        self.yaw = QuantitySpinBox("angle", "deg", minimum=-45, maximum=45, step=0.5, decimals=1)
        self.yaw.setObjectName("yaw")
        self.yaw.setFixedWidth(84)
        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setObjectName("yaw_slider")
        self.yaw_slider.setRange(-60, 60)                   # half-degree steps, ±30° on the slider
        self.yaw_slider.setPageStep(10)
        self.yaw_slider.valueChanged.connect(self._yaw_slider_moved)
        self.yaw.siValueChanged.connect(self._yaw_edited)
        self._yaw_row = g.add_row("Yaw angle", hbox(self.yaw, self.yaw_slider, stretch=1, spacing=8),
                                  tooltip="Sideslip: the wind comes from the side. The body is yawed about the "
                                          "vertical axis through its moment point; a yawed run needs the full "
                                          "domain (no symmetry plane).")

    def _build_ground_group(self) -> None:
        g = self.grid.add_group("Ground", key="ground")
        self.ground_group = g
        self.ground_combo = _combo(GROUND_ITEMS)
        self.ground_combo.setObjectName("ground_mode")
        self.ground_combo.currentIndexChanged.connect(lambda _i: self._ground_edited())
        g.add_row("Road", self.ground_combo,
                  tooltip="In the vehicle's frame the road moves backwards at the driving speed")
        self.ground_wheels = QCheckBox(WHEELS_TEXT)
        self.ground_wheels.setObjectName("ground_wheels")
        self._wheels_pref = True             # the user's choice while the box cannot apply
        self.ground_wheels.toggled.connect(lambda _on: self._ground_edited())
        # without wheel bodies the row is a plain muted note instead of a disabled
        # box (a disabled checkbox is drawn in the fainter disabled colour)
        self.no_wheels_label = label(NO_WHEELS_TEXT, "muted")
        self.no_wheels_label.setObjectName("no_wheels")
        self.no_wheels_label.setVisible(False)
        self._wheels_row = g.add_row("Wheels", hbox(self.ground_wheels, self.no_wheels_label, stretch=None))

    def _build_mesh_group(self) -> None:
        g = self.grid.add_group("Mesh && time", key="mesh")      # "&&": a lone "&" is a mnemonic
        self.mesh_group = g
        self.quality = _combo(QUALITY_ITEMS)
        self.quality.setObjectName("quality")
        self.quality.currentIndexChanged.connect(lambda _i: self._quality_edited(self.quality.currentData()))
        g.add_row("Quality", self.quality,
                  tooltip="Draft, Standard and Fine set the grid, domain and duration; Custom uses the "
                          "values below")
        self.cells_per_ref = QSpinBox()
        # 1024 per chord is ~34 M cells for an airfoil domain, already a long run on a GPU;
        # the planner still refuses what the device cannot hold (plan.MEMORY_FRACTION)
        self.cells_per_ref.setRange(0, 1024)
        self.cells_per_ref.setSingleStep(16)
        self.cells_per_ref.setSpecialValueText("Auto")
        self.cells_per_ref.setKeyboardTracking(False)
        self.cells_per_ref.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.cells_per_ref.setObjectName("cells_per_ref")
        self.cells_per_ref.valueChanged.connect(lambda v: self._cfd_edited("cells_per_ref", int(v) or None,
                                                                           custom=True))
        self._cells_row = g.add_row("Cells per chord", self.cells_per_ref,
                                    tooltip="Grid cells across one reference length. A value switches "
                                            "Quality to Custom; Auto lets the planner choose.")
        auto_tip = "Multiples of the reference length L. A value switches Quality to Custom."
        self.dom_up = _spin(0.0, 50.0, 0.5, 1, " × L", "Auto")
        self.dom_down = _spin(0.0, 100.0, 0.5, 1, " × L", "Auto")
        self.dom_height = _spin(0.0, 50.0, 0.5, 1, " × L", "Auto")
        self.dom_width = _spin(0.0, 50.0, 0.5, 1, " × L", "Auto")
        self._dom_rows = {}
        for w, f, text in ((self.dom_up, "domain_upstream", "Upstream"),
                           (self.dom_down, "domain_downstream", "Downstream"),
                           (self.dom_height, "domain_height", "Domain height"),
                           (self.dom_width, "domain_width", "Domain width")):
            w.setObjectName(f)
            w.valueChanged.connect(lambda v, f=f: self._cfd_edited(f, v or None, custom=True))
            self._dom_rows[f] = g.add_row(text, w, tooltip=auto_tip + (" Total width of the tunnel (both halves)."
                                                                       if f == "domain_width" else ""))
        time_tip = ("Convective time units: one L/U is the time the flow takes to travel one reference "
                    "length. A value switches Quality to Custom.")
        self.duration = _spin(0.0, 1000.0, 1.0, 1, " L/U", "Auto")
        self.average_from = _spin(0.0, 1000.0, 1.0, 1, " L/U", "Auto")
        self.duration.setObjectName("duration")
        self.average_from.setObjectName("average_from")
        self.duration.valueChanged.connect(lambda v: self._cfd_edited("duration", v or None, custom=True))
        self.average_from.valueChanged.connect(lambda v: self._cfd_edited("average_from", v or None,
                                                                          custom=True))
        g.add_row("Duration", self.duration, tooltip=time_tip)
        g.add_row("Average from", self.average_from,
                  tooltip="Start of the force averaging window, after the start-up transient. " + time_tip)
        self.symmetry = _combo(SYMMETRY_ITEMS)
        self.symmetry.setObjectName("symmetry")
        self.symmetry.currentIndexChanged.connect(lambda _i: self._cfd_edited("symmetry", self.symmetry.currentData()))
        self._symmetry_row = g.add_row("Symmetry plane", self.symmetry,
                                       tooltip="A mirror-symmetric design at zero yaw needs only the y ≥ 0 half "
                                               "of the tunnel; forces are reported for the whole body")
        self.symmetry_note = label("", "muted", wrap=True)
        self.symmetry_note.setObjectName("symmetry_note")
        self._symmetry_note_row = g.add_row("", self.symmetry_note)

    def _build_flowviz_group(self) -> None:
        """Smoke for 3D CFD runs (DESIGN.md 7.13): on/off and where it is released.

        The emitter fields follow the References pattern of the Geometry page:
        while a value is on Auto the field shows what the automatic rule gives
        for this body, and typing a value fixes it (the Auto switch turns off).
        One switch governs both heights, since a range is chosen as a whole.
        """
        g = self.grid.add_group("Flow visualisation", key="flowviz")
        self.flowviz_group = g
        # the emitters of DESIGN.md 10.1 come first; while there are none, a 3D run releases its
        # smoke from the automatic rake below (the v0.2 behaviour), which can become an emitter
        self.emitters_section = phys.PlacedItemsSection(g, "emitters", self)
        self._connect_section(self.emitters_section)
        self.emitters_section.autoPlaceRequested.connect(self.place_emitter_upstream)
        self.smoke = QCheckBox("Release smoke upstream")
        self.smoke.setObjectName("smoke")
        self.smoke.toggled.connect(lambda on: self._smoke_edited("smoke", bool(on)))
        self.smoke_row = g.add_row("Smoke", self.smoke, tooltip=SMOKE_TIP)
        self.smoke_kind = _combo(RAKE_ITEMS)
        self.smoke_kind.setObjectName("smoke_kind")
        self.smoke_kind.currentIndexChanged.connect(
            lambda _i: self._smoke_edited("kind", str(self.smoke_kind.currentData())))
        self._smoke_kind_row = g.add_row("Emitter", self.smoke_kind,
                                         tooltip="How the smoke is released, upstream of the body")
        lu = display_unit("length")
        self.smoke_z_min = QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.01)
        self.smoke_z_max = QuantitySpinBox("length", lu, minimum=-1e4, maximum=1e4, step=0.01)
        self.smoke_spacing = QuantitySpinBox("length", lu, minimum=1e-4, maximum=1e3, step=0.01)
        self.smoke_z_min.setObjectName("smoke_z_min")
        self.smoke_z_max.setObjectName("smoke_z_max")
        self.smoke_spacing.setObjectName("smoke_spacing")
        self.smoke_z_min.siValueChanged.connect(lambda v: self._smoke_height_edited("z_min", float(v)))
        self.smoke_z_max.siValueChanged.connect(lambda v: self._smoke_height_edited("z_max", float(v)))
        self.smoke_spacing.siValueChanged.connect(lambda v: self._smoke_edited("spacing", float(v)))
        self.smoke_auto_z = _auto_toggle("Follow the body: mid-height for a rake; from just above its lowest "
                                         "point to just over its top for a grid or a plane")
        self.smoke_auto_z.setObjectName("smoke_auto_z")
        self.smoke_auto_z.clicked.connect(lambda on: self._smoke_auto("z", bool(on)))
        self.smoke_auto_spacing = _auto_toggle("About a dozen nozzles across a rake, eight along the larger side "
                                               "of a grid, rounded to a 1-2-5 value")
        self.smoke_auto_spacing.setObjectName("smoke_auto_spacing")
        self.smoke_auto_spacing.clicked.connect(lambda on: self._smoke_auto("spacing", bool(on)))
        z_tip = "Height z in the wind frame (the road is z = 0). Typing a value fixes it."
        self._smoke_zmin_row = g.add_row("Height", hbox(self.smoke_z_min, self.smoke_auto_z), tooltip=z_tip)
        self._smoke_zmax_row = g.add_row("Height to", hbox(self.smoke_z_max, _row_spacer()), tooltip=z_tip)
        self._smoke_spacing_row = g.add_row("Spacing", hbox(self.smoke_spacing, self.smoke_auto_spacing),
                                            tooltip="Distance between neighbouring nozzles")
        self.smoke_note = label("", "muted", wrap=True)
        self.smoke_note.setObjectName("smoke_note")
        self._smoke_note_row = g.add_row("", self.smoke_note)
        self.convert_rake = small_button("Convert to emitter",
                                         "Keep this rake with the simulation as an emitter you can move and save; "
                                         "it then no longer follows the body.")
        self.convert_rake.setObjectName("convert_rake")
        self.convert_rake.clicked.connect(self.convert_rake_to_emitter)
        self._convert_row = g.add_row("", hbox(self.convert_rake, stretch=None))
        self._legacy_smoke_rows = (self.smoke_row, self._smoke_kind_row, self._smoke_zmin_row, self._smoke_zmax_row,
                                   self._smoke_spacing_row, self._smoke_note_row, self._convert_row)

    def _show_legacy_smoke(self, on: bool) -> None:
        """The automatic rake's rows: shown for a 3D CFD run without emitters (its smoke comes from it)."""
        for row in self._legacy_smoke_rows:
            row.set_visible(on)
        if on:                                 # rows that depend on the rake kind stay as the kind says
            self._load_flow_viz(flow_viz_of(self.simulation), self.last_smoke)

    def focus_flow_viz(self) -> None:
        """Bring the Flow visualisation group into view and focus its first live field.

        Called when the user clicks the smoke emitter in the 3D preview: the rake is
        drawn there as a first-class object, so a click on it should lead to its
        settings, as a click on a part does in CAD. The group is expanded, scrolled
        to the top of the inspector (or as far as the scroll range allows) and the
        Emitter combo takes the focus; with the smoke off, the Smoke switch does.
        """
        g = self.flowviz_group
        if not g.isVisibleTo(self):
            return
        g.set_expanded(True)
        self._scroll_to_flowviz()
        # the expansion (and a summary refresh that shows or hides rows above) is laid out
        # by posted events: scroll again once they have run, or the group lands lower
        QTimer.singleShot(0, self._scroll_to_flowviz)
        target = self.smoke_kind if self.smoke.isChecked() and self.smoke_kind.isEnabled() else self.smoke
        target.setFocus(Qt.FocusReason.MouseFocusReason)

    def _scroll_to_flowviz(self) -> None:
        """Scroll the inspector so the Flow visualisation header is at its top."""
        self._scroll_to_group(self.flowviz_group)

    def _scroll_to_group(self, group) -> None:
        """Scroll the inspector so ``group``'s header is at its top (or as far as the range allows)."""
        self.grid.layout_.activate()
        bar = self.inspector_scroll.verticalScrollBar()
        bar.setValue(min(max(group.y() - theme.SPACING, 0), bar.maximum()))

    # -- placed physics (DESIGN.md 10) ------------------------------------------
    def _build_physics_groups(self) -> None:
        """Domain, Heat transfer, Heat sources, Fans, Jets and vents: editors from
        :mod:`~aerolume.app.widgets.physics_editor` that add their rows to these groups, so they
        share the inspector's label column; each edits the simulation in place."""
        g = self.grid.add_group("Domain", key="domain")
        self.domain_group = g
        self.domain_section = phys.DomainSection(g, self)
        self._connect_section(self.domain_section)
        g = self.grid.add_group("Heat transfer", key="heat")
        self.heat_group = g
        self.heat_section = phys.ThermalSection(g, self)
        self._connect_section(self.heat_section)
        self.item_groups: dict[str, Any] = {"emitters": self.flowviz_group}
        self.item_sections: dict[str, phys.PlacedItemsSection] = {"emitters": self.emitters_section}
        for kind, title in (("heat_sources", "Heat sources"), ("fans", "Fans"), ("ports", phys.ITEM_TITLE["ports"])):
            g = self.grid.add_group(title, key=kind)
            sec = phys.PlacedItemsSection(g, kind, self)
            self._connect_section(sec)
            self.item_groups[kind] = g
            self.item_sections[kind] = sec

    def _build_diagnostics_group(self) -> None:
        g = self.grid.add_group("Diagnostics", expanded=False, key="diagnostics")
        self.diagnostics_group = g
        self.diagnostics_section = phys.DiagnosticsSection(g, self)
        self._connect_section(self.diagnostics_section)

    def _connect_section(self, sec) -> None:
        sec.edited.connect(self._physics_edited)
        sec.designEdited.connect(self._physics_design_edited)
        sel = getattr(sec, "selectionChanged", None)
        if sel is not None:
            sel.connect(self._physics_list_selected)

    def physics_sections(self) -> list:
        """Every placed-physics editor of the page (emitters first)."""
        out = [self.emitters_section, self.domain_section, self.heat_section]
        out += [s for k, s in getattr(self, "item_sections", {}).items() if k != "emitters"]
        out.append(self.diagnostics_section)
        return out

    def _physics_edited(self) -> None:
        """A placed-physics editor changed the simulation: redraw the items now, save and replan
        after the debounce, and keep the automatic rake's rows to cases without emitters."""
        sim = self.simulation
        if self._loading or sim is None:
            return
        boxed = sim.cfd.domain_box is not None
        if sim.cfd.domain_mode != self._domain_mode_shown or boxed != self._domain_boxed_shown:
            # tunnel <-> closed box: the inlet/outlet walls, the emitters' empty text, the speed's meaning
            self._domain_mode_shown = sim.cfd.domain_mode
            self._domain_boxed_shown = boxed
            self.heat_section.reload()
            # every placed-item list: the emitters' empty text, "Place upstream", the frame's name
            for sec in self.item_sections.values():           # emitters included
                sec.reload()
            d = self.state.design
            if d is not None:
                self._update_visibility(sim, d)
        self._sync_physics_rows()
        self._update_physics_preview()
        self._changed()

    def _physics_design_edited(self) -> None:
        """A body's thermal condition changed: it lives on the Design, which is saved at once."""
        d = self.state.design
        if d is not None and self.state.project is not None and d.id in self.state.project.designs:
            self.state.update_design(d)

    def _physics_list_selected(self, kind: str, index: int) -> None:
        self._physics_sel = (kind, index) if index >= 0 else None
        self._update_physics_preview()

    def _sync_physics_rows(self) -> None:
        sim, d = self.simulation, self.state.design
        three = d is not None and getattr(d, "dimension", 2) == 3
        legacy = bool(sim is not None and three and sim.analysis == "cfd" and not sim.emitters)
        self._show_legacy_smoke(legacy)

    def physics_extent(self) -> tuple | None:
        """The planned domain (wind frame, metres) the placed items are drawn in: the 2D extent or
        the full 3D extent of the last plan, None before one."""
        plan = self.last_plan
        if plan is None:
            return None
        if getattr(plan, "dimension", 2) == 3 or getattr(plan, "extent3", None) is not None:
            ext = getattr(plan, "extent3_full", None) or getattr(plan, "extent3", None)
        else:
            ext = getattr(plan, "extent", None)
        return tuple(float(v) for v in ext) if ext is not None else None

    def _update_physics_preview(self) -> None:
        """Hand the placed items to the case preview (2D canvas or 3D view) with the selection."""
        sim, d = self.simulation, self.state.design
        if sim is None or d is None or sim.analysis != "cfd":
            specs: list = []
        else:
            specs = phys.scene_specs(d, sim, selected=getattr(self, "_physics_sel", None),
                                     extent=self.physics_extent())
        three = d is not None and getattr(d, "dimension", 2) == 3
        (self.preview3d if three else self.preview).set_physics(specs)
        (self.preview if three else self.preview3d).set_physics([])

    def _preview_selected(self, key) -> None:
        """An item was clicked in the case preview: select it in its list, open and show its group."""
        self._physics_sel = tuple(key) if key is not None else None
        if key is not None and key[0] in getattr(self, "item_sections", {}):
            sec = self.item_sections[key[0]]
            sec.select(int(key[1]))
            group = self.item_groups[key[0]]
            group.set_expanded(True)
            self._scroll_to_group(group)
            QTimer.singleShot(0, lambda g=group: self._scroll_to_group(g))
        elif key is not None and key[0] == "domain":
            self.domain_group.set_expanded(True)
            self._scroll_to_group(self.domain_group)
        self._update_physics_preview()

    def _preview_moved(self, key, handle: str, point) -> None:
        """A handle is being dragged: move the item (the plan and the save wait for the drop)."""
        sim, d = self.simulation, self.state.design
        if sim is None or d is None or self._loading:
            return
        if not phys.apply_drag(d, sim, tuple(key), handle, point, extent=self.physics_extent()):
            return
        self._physics_sel = tuple(key)
        self._dirty = True
        self._update_physics_preview()
        if key[0] == "domain":
            self.domain_section.reload()
        elif key[0] in self.item_sections:
            sec = self.item_sections[key[0]]
            sec.select(int(key[1]))
            sec.refresh_lines()

    def _preview_dropped(self, key, handle: str, point) -> None:
        self._preview_moved(key, handle, point)
        self._changed()

    def set_physics_layer(self, on: bool) -> None:
        """The tool strip's Physics toggle: show (and edit) the placed items in the preview, or hide them."""
        self.preview.set_show_physics(on)
        self.preview3d.set_show_physics(on)
        if self.physics_toggle.isChecked() != bool(on):
            self.physics_toggle.setChecked(bool(on))

    def place_emitter_upstream(self, index: int) -> None:
        """Move emitter ``index`` to the solver's automatic rake for this body (3D: the rake rule of
        :func:`aerolume.viz3d.flow.resolve_rake` on the planned body and domain; 2D: a rake across
        the body's height ahead of it). Its tracer, colour and timing are kept."""
        sim, d = self.simulation, self.state.design
        if sim is None or d is None or not 0 <= index < len(sim.emitters):
            return
        e = sim.emitters[index]
        if getattr(d, "dimension", 2) == 3:
            layout = self._smoke_layout(d, sim, self.last_plan, {**FLOW_VIZ_DEFAULTS, "kind": "rake"}, force=True)
            made = emitters_from_layout(layout, True) if layout is not None and "error" not in layout else []
            if not made:
                self.state.message("The automatic rake cannot be placed for this body.", 5000)
                return
            src = made[0]
        else:
            src = phys.default_emitter(d, sim, self.physics_extent())
        e.kind, e.position, e.end, e.count, e.frame = src.kind, src.position, src.end, src.count, "wind"
        self.emitters_section.reload()
        self.emitters_section.select(index)
        self._physics_edited()

    def convert_rake_to_emitter(self) -> list:
        """Turn the automatic smoke rake of a 3D run into emitters at the nozzles it shows now.

        The automatic rake follows the body and is kept only for the session; an emitter is part of
        the simulation (saved with the project) and can be moved, retimed or recoloured. A rake, a
        vertical rake and a plane band become one emitter each; a grid becomes one rake per row."""
        sim, d = self.simulation, self.state.design
        if sim is None or d is None:
            return []
        fv = flow_viz_of(sim)
        layout = self.last_smoke or self._smoke_layout(d, sim, self.last_plan, {**fv, "smoke": True}, force=True)
        made = emitters_from_layout(layout, bool(fv.get("smoke", True))) if layout else []
        if not made:
            self.state.message("The smoke rake could not be converted: it has no position yet.", 5000)
            return []
        for e in made[:-1]:
            sim.emitters.append(e)
        self.emitters_section.append(made[-1])          # selects the last one and emits edited
        self.last_smoke = None
        self._physics_edited()
        self.update_summary()                           # the preview drops the automatic rake now
        return made

    def _build_solver_group(self) -> None:
        g = self.grid.add_group("Solver", expanded=False, key="solver")
        self.solver_group = g
        self.turbulence = _combo(TURBULENCE_ITEMS)
        self.turbulence.setObjectName("turbulence")
        self.turbulence.currentIndexChanged.connect(
            lambda _i: self._cfd_edited("turbulence", self.turbulence.currentData()))
        g.add_row("Turbulence", self.turbulence,
                  tooltip="Automatic: laminar when the grid resolves the boundary layer, LES otherwise")
        self.re_mode = _combo(REYNOLDS_ITEMS)
        self.re_mode.setObjectName("reynolds_mode")
        self.re_mode.currentIndexChanged.connect(self._re_mode_edited)
        g.add_row("Reynolds number", self.re_mode,
                  tooltip="Capped runs a lower Reynolds number the grid can resolve; the Estimate shows both")
        # The cap is stored with the simulation (a new capped airfoil starts at
        # simulation.defaults.AIRFOIL_DEFAULT_CAP_RE), so every quality and device simulates
        # the same flow. 0 shows "Auto" and stores None: the planner then caps at the highest
        # Reynolds number the grid resolves, which changes with the quality (it warns).
        self.re_cap = _spin(0.0, 1e8, 500.0, 0, special="Auto")
        self.re_cap.setObjectName("reynolds_cap")
        self.re_cap.setGroupSeparatorShown(True)
        self.re_cap.valueChanged.connect(lambda v: self._cfd_edited("reynolds_cap", float(v) if v > 0 else None))
        self._re_cap_row = g.add_row("Reynolds cap", self.re_cap,
                                     tooltip="Reynolds number simulated when the mode is Capped. Kept fixed "
                                             "across qualities and devices, so a finer grid refines the same "
                                             "flow. Auto: the highest the grid resolves (changes with the "
                                             "quality)")
        self.precision = _combo(PRECISION_ITEMS)
        self.precision.setObjectName("precision")
        self.precision.currentIndexChanged.connect(
            lambda _i: self._cfd_edited("precision", self.precision.currentData()))
        g.add_row("Precision", self.precision)
        self.device = _combo(DEVICE_ITEMS)
        self.device.setObjectName("device")
        self.device.currentIndexChanged.connect(
            lambda _i: self._cfd_edited("device", self.device.currentData()))
        g.add_row("Device", self.device)
        self.sensitivity = QCheckBox("Compute")
        self.sensitivity.setObjectName("sensitivity")
        self.sensitivity.toggled.connect(self._sensitivity_edited)
        g.add_row("Sensitivity map", self.sensitivity,
                  tooltip="Where pushing the surface in or out changes the objective most (autograd "
                          "adjoint). Adds roughly one extra run of compute.")
        self.sensitivity_note = label("", "muted", wrap=True)
        self.sensitivity_note.setObjectName("sensitivity_note")
        self._sensitivity_note_row = g.add_row("", self.sensitivity_note)
        self._sensitivity_note_row.set_visible(False)
        self.objective = _combo(OBJECTIVE_ITEMS)
        self.objective.setObjectName("objective")
        self.objective.currentIndexChanged.connect(
            lambda _i: self._cfd_edited("sensitivity_objective", self.objective.currentData()))
        g.add_row("Objective", self.objective, tooltip="Quantity the sensitivity map is computed for")

    def _build_polar_group(self) -> None:
        g = self.grid.add_group("Polar", key="polar")
        self.polar_group = g
        self.polar_card = g                     # historical name, kept for scripts
        self.alpha_start = QuantitySpinBox("angle", "deg", minimum=-30, maximum=30, step=0.5, decimals=1)
        self.alpha_end = QuantitySpinBox("angle", "deg", minimum=-30, maximum=30, step=0.5, decimals=1)
        self.alpha_step = QuantitySpinBox("angle", "deg", minimum=0.1, maximum=10, step=0.25, decimals=2)
        for w, f, text in ((self.alpha_start, "alpha_start", "First angle"),
                           (self.alpha_end, "alpha_end", "Last angle"),
                           (self.alpha_step, "alpha_step", "Angle step")):
            w.setObjectName(f)
            w.siValueChanged.connect(lambda v, f=f: self._airfoil_edited(f, v))
            g.add_row(text, w)
        self.ncrit = _spin(1.0, 14.0, 0.5, 1)
        self.ncrit.setObjectName("ncrit")
        self.ncrit.valueChanged.connect(lambda v: self._airfoil_edited("ncrit", v))
        g.add_row("Ncrit", self.ncrit, tooltip=NCRIT_TIP)
        xtr_tip = "Forced transition position; 1.00 = free transition. A trip strip near the leading edge ≈ 0.05."
        self.xtr_upper = _spin(0.0, 1.0, 0.05, 2, " x/c")
        self.xtr_lower = _spin(0.0, 1.0, 0.05, 2, " x/c")
        self.xtr_upper.setObjectName("xtr_upper")
        self.xtr_lower.setObjectName("xtr_lower")
        self.xtr_upper.valueChanged.connect(lambda v: self._airfoil_edited("xtr_upper", v))
        self.xtr_lower.valueChanged.connect(lambda v: self._airfoil_edited("xtr_lower", v))
        g.add_row("Transition, upper", self.xtr_upper, tooltip=xtr_tip)
        g.add_row("Transition, lower", self.xtr_lower, tooltip=xtr_tip)
        self.mach = _spin(0.0, 0.7, 0.01, 2)
        self.mach.setObjectName("mach")
        self.mach.valueChanged.connect(lambda v: self._airfoil_edited("mach", v))
        self.mach_from_speed = small_button("From speed", "Set the Mach number from the flow speed and fluid")
        self.mach_from_speed.clicked.connect(self._mach_from_speed)
        g.add_row("Mach number", hbox(self.mach, self.mach_from_speed),
                  tooltip="Karman-Tsien compressibility correction; 0 = incompressible")
        self.panels = QSpinBox()
        self.panels.setRange(60, 400)
        self.panels.setSingleStep(20)
        self.panels.setKeyboardTracking(False)
        self.panels.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.panels.setObjectName("panels")
        self.panels.valueChanged.connect(lambda v: self._airfoil_edited("panels", int(v)))
        g.add_row("Panels", self.panels, tooltip="Surface points for the panel method; 160-200 is plenty")

    def _build_wing_group(self) -> None:
        g = self.grid.add_group("Wing analysis", key="wing")
        self.wing_group = g
        self.w_alpha_start = QuantitySpinBox("angle", "deg", minimum=-20, maximum=25, step=0.5, decimals=1)
        self.w_alpha_end = QuantitySpinBox("angle", "deg", minimum=-20, maximum=25, step=0.5, decimals=1)
        self.w_alpha_step = QuantitySpinBox("angle", "deg", minimum=0.1, maximum=10, step=0.25, decimals=2)
        for w, f, text in ((self.w_alpha_start, "alpha_start", "First angle"),
                           (self.w_alpha_end, "alpha_end", "Last angle"),
                           (self.w_alpha_step, "alpha_step", "Angle step")):
            w.setObjectName(f"wing_{f}")
            w.siValueChanged.connect(lambda v, f=f: self._wing_edited(f, float(v)))
            g.add_row(text, w)
        self.w_method = _combo(WING_METHOD_ITEMS)
        self.w_method.setObjectName("wing_method")
        self.w_method.currentIndexChanged.connect(lambda _i: self._wing_edited("method", self.w_method.currentData()))
        g.add_row("Method", self.w_method, tooltip="Vortex lattice: any planform. Lifting line: straight wings, "
                                                   "nonlinear section lift up to stall.")
        self.w_viscous = QCheckBox("Profile drag from section polars")
        self.w_viscous.setObjectName("wing_viscous")
        self.w_viscous.toggled.connect(lambda on: self._wing_edited("viscous", bool(on)))
        g.add_row("Viscous", self.w_viscous,
                  tooltip="Adds each section's profile drag (and, for the lifting line, its nonlinear lift) from "
                          "2D viscous polars at the local Reynolds number")
        self.w_ncrit = _spin(1.0, 14.0, 0.5, 1)
        self.w_ncrit.setObjectName("wing_ncrit")
        self.w_ncrit.valueChanged.connect(lambda v: self._wing_edited("ncrit", float(v)))
        g.add_row("Ncrit", self.w_ncrit, tooltip=NCRIT_TIP)
        self.w_mach = _spin(0.0, 0.85, 0.01, 2)
        self.w_mach.setObjectName("wing_mach")
        self.w_mach.valueChanged.connect(lambda v: self._wing_edited("mach", float(v)))
        mach_btn = small_button("From speed", "Set the Mach number from the flow speed and fluid")
        mach_btn.clicked.connect(self._wing_mach_from_speed)
        g.add_row("Mach number", hbox(self.w_mach, mach_btn),
                  tooltip="Prandtl-Glauert correction on the lattice; 0 = incompressible")

    # ===================================================================
    # page protocol
    # ===================================================================
    def on_shown(self) -> None:
        """Make sure the current design has a simulation selected (create one if none).

        The id of a simulation created here is remembered until the next
        navigation, because ``navigate("setup:new")`` calls this and then
        ``handle_action("new")``: for a design without simulations both used to
        create one, leaving a never-run "… · run 1" next to the selected "… · run 2".
        """
        self._auto_created_id = ""
        d = self.state.design
        if self.state.project is not None and d is not None:
            sims = self._sims_of(d.id)
            if not sims:
                sim = self.new_simulation()
                self._auto_created_id = sim.id if sim is not None else ""
            elif self.state.simulation is None or self.state.simulation.design_id != d.id:
                self.state.select_simulation(sims[0].id)
        self._refresh()

    def hideEvent(self, event) -> None:
        # leaving the page must not lose an edit still waiting for its debounce
        self.flush()
        self._auto_created_id = ""
        super().hideEvent(event)

    def handle_action(self, action: str) -> None:
        if action.startswith("template:"):
            self.new_from_template(action.split(":", 1)[1])
            return
        if action == "new":
            auto, self._auto_created_id = getattr(self, "_auto_created_id", ""), ""
            sim = self.state.simulation
            if auto and sim is not None and sim.id == auto and sim.status == "new":
                return              # on_shown has just made the one new simulation this asks for
            self.new_simulation()
        elif action == "run":
            self.run_simulation()

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    # ===================================================================
    # state -> widgets
    # ===================================================================
    def _sims_of(self, design_id: str) -> list[Simulation]:
        p = self.state.project
        return [] if p is None else [s for s in p.simulations.values() if s.design_id == design_id]

    @property
    def simulation(self) -> Simulation | None:
        """The simulation shown in the form (the current one, if it belongs to the design)."""
        sim, d = self.state.simulation, self.state.design
        if sim is None or d is None or sim.design_id != d.id:
            return None
        return sim

    def _on_project_changed(self) -> None:
        self._save_timer.stop()
        self._dirty = False
        self._sim = None
        self._logged.clear()
        self._refresh()

    def _on_designs_changed(self) -> None:
        self._refresh(reload_form=False)
        self._schedule_summary()               # geometry/refs may have changed Re

    def _on_design_changed(self, _design_id: str) -> None:
        self.flush()
        d = self.state.design
        if d is not None and (self.state.simulation is None or self.state.simulation.design_id != d.id):
            sims = self._sims_of(d.id)
            self.state.select_simulation(sims[0].id if sims else "")
        self._refresh()

    def _on_sim_changed(self, _sim_id: str) -> None:
        self.flush()
        self._refresh()

    def _on_sims_changed(self) -> None:
        sim = self.simulation
        self._refresh(reload_form=sim is not self._sim)

    def _on_job_event(self, *args) -> None:
        if args and isinstance(args[0], str):
            self._progress.pop(args[0], None)
        self._update_status()

    def _on_job_progress(self, sim_id: str, fraction: float, _message: str = "") -> None:
        self._progress[sim_id] = float(fraction)
        sim = self.simulation
        if sim is not None and sim.id == sim_id:
            self._update_status()

    def _refresh(self, reload_form: bool = True) -> None:
        p, d = self.state.project, self.state.design
        if p is None:
            self._show_empty("No project open", "Designs and simulations belong to a project.",
                             "Show welcome page", "start")
            return
        if d is None:
            self._show_empty("No design selected", "Select a design in the Project panel, or create one.",
                             "Go to Geometry", "geometry")
            return
        self._pages.setCurrentIndex(1)
        sim = self.simulation
        has = sim is not None
        self._form_widget.setEnabled(has)
        self.run_button.setEnabled(has)
        self.live_button.setEnabled(has and sim.analysis == "cfd")
        self.more_button.setEnabled(True)
        if reload_form or sim is not self._sim:
            self._load_form(sim, d)
            self._summary_now()              # a new selection shows its numbers at once
        else:
            self._schedule_summary()
        self._update_status()

    def _show_empty(self, title: str, text: str, button: str, target: str) -> None:
        self.empty.set_text(title, text)
        self.empty.set_action(button)
        self._empty_target = target
        self._pages.setCurrentIndex(0)

    def _empty_action(self) -> None:
        self.state.navigate(self._empty_target)

    def _busy(self, sim: Simulation) -> str:
        """"running", "queued" or "" from the job manager (which knows before the model does)."""
        if self.jobs is None:
            return ""
        try:
            if sim.id == self.jobs.running_id():
                return "running"
            if sim.id in self.jobs.queued_ids():
                return "queued"
        except Exception:  # noqa: BLE001 - the job manager is optional here
            return ""
        return ""

    def status_text(self) -> str:
        return self.status_label.text()

    def _update_status(self) -> None:
        """Header status ("Not run", "Running 43 %", "Done") and the title."""
        sim = self.simulation
        if sim is None:
            self.title_label.setText("No simulation")
            self.status_label.setText("")
            return
        self.title_label.setText(sim.name)
        self.title_label.setToolTip(sim.name)
        status = self._busy(sim) or sim.status
        text, role = STATUS_TEXT.get(status, (status.capitalize(), "muted"))
        if status == "running" and sim.id in self._progress:
            text = f"Running {100 * self._progress[sim.id]:.0f} %"
        set_prop(self.status_label, "role", role)
        self.status_label.setText(text)
        if sim.status == "failed" and sim.error:
            self.status_label.setToolTip(str(sim.error))
        else:
            self.status_label.setToolTip("")

    def _load_form(self, sim: Simulation | None, d: Design) -> None:
        """Fill every control from ``sim`` without echoing edits back."""
        self._sim = sim
        if sim is None:
            self.name_edit.setText("")
            return
        self._loading = True
        try:
            c, cfd, af, ws = sim.conditions, sim.cfd, sim.airfoil, sim.wing
            three = getattr(d, "dimension", 2) == 3
            if not self.name_edit.hasFocus():
                set_line_text(self.name_edit, sim.name)
            offered = analyses_for(d)
            meshed3d = three and offered == ["cfd"]
            self.analysis.set_visible_keys(offered)
            self.analysis.setVisible(not meshed3d)
            self.analysis_fixed.setVisible(meshed3d)
            self.analysis.buttons["cfd"].setText("3D CFD" if d.kind == "wing" else "CFD")
            self.analysis.buttons["cfd"].setToolTip(WING_CFD_TIP if d.kind == "wing" else ANALYSIS_ITEMS[0][2])
            polar_ok = polar_allowed(d)
            self.analysis.buttons["polar"].setEnabled(polar_ok)
            self.analysis.buttons["polar"].setToolTip(
                ANALYSIS_ITEMS[1][2] if polar_ok else
                "Polars need a single-element airfoil design; this design is a "
                f"{'multi-element airfoil' if d.kind == 'airfoil' else d.kind}.")
            self.analysis.set_current(sim.analysis)
            # fluid
            mode, value = fluid_mode(c.fluid)
            idx = self.fluid_combo.findData(f"preset:{preset_key(c.fluid.name)}" if mode == "preset" else mode)
            self.fluid_combo.blockSignals(True)
            self.fluid_combo.setCurrentIndex(max(idx, 0))
            self.fluid_combo.blockSignals(False)
            if mode == "isa":
                self.altitude.set_si_value(value or 0.0)
            if mode in ("air_t", "water_t"):
                self.temperature.set_si_value(value if value is not None else 15.0)
            self._show_fluid(c.fluid, mode)
            self.speed.set_si_value(c.speed)
            self._set_alpha_widgets(c.alpha_deg)
            self.yaw.set_si_value(c.yaw_deg)
            _set_silent(self.yaw_slider, round(c.yaw_deg * 2))
            pitch = d.kind == "vehicle" or (three and d.kind == "generic")
            self._alpha_row.set_label("Pitch angle" if pitch else "Angle of attack")
            self._alpha_row.label.setToolTip(
                "Positive nose up (squat under acceleration); usually 0 for a car. The body is rotated "
                "about its moment reference point." if d.kind == "vehicle" else
                "Positive nose up. The geometry is rotated about its moment reference point; the flow "
                + ("always runs along +x." if three else "always comes from the left."))
            self._yaw_row.set_label("Sideslip angle" if d.kind == "wing" else "Yaw angle")
            # ground
            _set_combo(self.ground_combo, ground_mode(sim))
            # what the box should go back to if it becomes applicable: the stored
            # choice when it applies now, else the default (wheels rotate)
            applies = sim.ground.moving and has_wheels(d)
            self._wheels_pref = bool(sim.ground.rotate_wheels) if applies else True
            _set_silent(self.ground_wheels, self._wheels_pref)
            self._update_ground_enabled(d)
            # mesh & time, solver
            _set_combo(self.quality, cfd.quality)
            self._cells_row.set_label("Cells per chord" if d.kind in ("airfoil", "wing") else "Cells per length")
            for w, v in ((self.cells_per_ref, cfd.cells_per_ref or 0),
                         (self.dom_up, cfd.domain_upstream or 0.0),
                         (self.dom_down, cfd.domain_downstream or 0.0),
                         (self.dom_height, cfd.domain_height or 0.0),
                         (self.dom_width, cfd.domain_width or 0.0), (self.duration, cfd.duration or 0.0),
                         (self.average_from, cfd.average_from or 0.0)):
                _set_silent(w, v)
            _set_combo(self.symmetry, cfd.symmetry)
            _set_combo(self.turbulence, cfd.turbulence)
            _set_combo(self.re_mode, cfd.reynolds_mode)
            _set_silent(self.re_cap, float(cfd.reynolds_cap or 0.0))       # 0 = "Auto"
            self._re_cap_row.set_enabled(cfd.reynolds_mode == "capped")
            _set_combo(self.precision, cfd.precision)
            _set_combo(self.device, cfd.device)
            _set_silent(self.sensitivity, cfd.compute_sensitivity)
            _set_combo(self.objective, cfd.sensitivity_objective)
            self.objective.setEnabled(bool(cfd.compute_sensitivity))
            # polar
            self.alpha_start.set_si_value(af.alpha_start)
            self.alpha_end.set_si_value(af.alpha_end)
            self.alpha_step.set_si_value(af.alpha_step)
            for w, v in ((self.ncrit, af.ncrit), (self.xtr_upper, af.xtr_upper), (self.xtr_lower, af.xtr_lower),
                         (self.mach, af.mach)):
                _set_silent(w, float(v))
            _set_silent(self.panels, int(af.panels))
            # wing
            self.w_alpha_start.set_si_value(ws.alpha_start)
            self.w_alpha_end.set_si_value(ws.alpha_end)
            self.w_alpha_step.set_si_value(ws.alpha_step)
            _set_combo(self.w_method, ws.method)
            _set_silent(self.w_viscous, bool(ws.viscous))
            _set_silent(self.w_ncrit, float(ws.ncrit))
            _set_silent(self.w_mach, float(ws.mach))
            self.w_ncrit.setEnabled(bool(ws.viscous))
            # flow visualisation (3D CFD); the automatic values arrive with the summary
            self.last_smoke = None
            self._load_flow_viz(flow_viz_of(sim))
            # placed physics: the editors take the simulation; the planned domain arrives with the summary
            if self._physics_sel is not None and (self._physics_sim_id != sim.id):
                self._physics_sel = None
            self._physics_sim_id = sim.id
            self._domain_mode_shown = sim.cfd.domain_mode
            self._domain_boxed_shown = sim.cfd.domain_box is not None
            for sec in self.physics_sections():
                sec.set_context(sim, d, None)
            self._update_visibility(sim, d)
        finally:
            self._loading = False

    def _load_flow_viz(self, fv: dict, layout: dict | None = None) -> None:
        """Show smoke settings ``fv``; fields on Auto show ``layout``'s values (the automatic
        rule for this body) so the user sees what will be used before fixing anything."""
        was, self._loading = self._loading, True
        try:
            on = bool(fv["smoke"])
            _set_silent(self.smoke, on)
            _set_combo(self.smoke_kind, fv["kind"])
            rake, plane = fv["kind"] == "rake", fv["kind"] == "plane"
            self._smoke_zmin_row.set_label("Height" if rake else "Height from")
            self._smoke_zmax_row.set_visible(not rake)
            self._smoke_spacing_row.set_visible(not plane)
            self._smoke_spacing_row.label.setToolTip(
                "Distance between neighbouring nozzles, up the rake" if fv["kind"] == "vertical" else
                "Distance between neighbouring nozzles, across the flow" + (" and up" if fv["kind"] == "grid" else ""))
            self.smoke_auto_z.setChecked(fv["z_min"] is None)
            self.smoke_auto_spacing.setChecked(fv["spacing"] is None)
            shown = layout if layout is not None else self.last_smoke
            usable = shown if shown is not None and "error" not in shown else None
            zl = fv["z_min"] if fv["z_min"] is not None else (usable["z"][0] if usable else None)
            zh = fv["z_max"] if fv["z_max"] is not None else (usable["z"][1] if usable else None)
            sp = fv["spacing"] if fv["spacing"] is not None else (usable["spacing"] if usable else None)
            for box, v in ((self.smoke_z_min, zl), (self.smoke_z_max, zh), (self.smoke_spacing, sp)):
                if v is not None:
                    box.set_si_value(float(v))
            for row in (self._smoke_kind_row, self._smoke_zmin_row, self._smoke_zmax_row, self._smoke_spacing_row):
                row.set_enabled(on)
            text = smoke_note(shown, fv) if (shown is not None or not on) else ""
            sim = self.simulation
            if (sim is not None and not sim.emitters and _flow_viz_holder(sim) is None
                    and any(fv.get(k) != v for k, v in FLOW_VIZ_DEFAULTS.items())):
                # the model has no field for the automatic rake: say so rather than lose the edit on reopen
                text = (text + " " if text else "") + ("These rake settings last until the app closes; Convert to "
                                                       "emitter keeps them with the project.")
            self.smoke_note.setText(text)
        finally:
            self._loading = was

    def _update_visibility(self, sim: Simulation, d: Design) -> None:
        three = getattr(d, "dimension", 2) == 3
        cfd = sim.analysis == "cfd"
        wing = sim.analysis == "wing"
        # a design without bodies (a cavity) has nothing to pitch or yaw; in a closed box there is no
        # free stream to meet at an angle either, so the group only stays for an attitude already set
        bodies = bool(d.meshes if three else d.bodies) or d.kind == "wing"
        closed = cfd and sim.cfd.domain_mode == "closed"
        turned = abs(sim.conditions.alpha_deg) > 1e-9 or abs(getattr(sim.conditions, "yaw_deg", 0.0)) > 1e-9
        self.attitude_group.setVisible((cfd or wing) and bodies and (not closed or turned))
        self._alpha_row.set_visible(cfd)                 # the wing analysis sweeps the angle itself
        self._yaw_row.set_visible(three and (cfd or wing))
        self.ground_group.setVisible(cfd and shows_ground(d))
        self.mesh_group.setVisible(cfd)
        # a fixed domain box (Setup > Domain) sets the extent: the multiples of L do not apply
        boxed = phys.explicit_box(sim, 3 if three else 2) is not None
        for f in ("domain_upstream", "domain_downstream", "domain_height"):
            self._dom_rows[f].set_visible(not boxed)
        self._dom_rows["domain_width"].set_visible(three and not boxed)
        self._symmetry_row.set_visible(three)
        self._symmetry_note_row.set_visible(three)
        self.flowviz_group.setVisible(cfd)
        self._speed_row.set_label("Reference speed" if closed else "Speed")
        self._speed_row.label.setToolTip(CLOSED_SPEED_TIP if closed else SPEED_TIP)
        for g in (self.domain_group, self.heat_group, self.diagnostics_group,
                  *(self.item_groups[k] for k in ("heat_sources", "fans", "ports"))):
            g.setVisible(cfd)
        self._sync_physics_rows()
        self.solver_group.setVisible(cfd)
        self._sensitivity_note_row.set_visible(three and cfd)
        self.polar_group.setVisible(sim.analysis == "polar")
        self.wing_group.setVisible(wing)
        self.live_button.setEnabled(cfd)
        self.domain_toggle.setEnabled(cfd)
        self._set_preview_mode(three, closed=closed)

    def _show_fluid(self, f: Fluid, mode: str) -> None:
        self._alt_row.set_visible(mode == "isa")
        self._temp_row.set_visible(mode in ("air_t", "water_t"))
        custom = mode == "custom"
        for w in (self.density, self.viscosity):
            w.setReadOnly(not custom)
            w.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows if custom
                               else QDoubleSpinBox.ButtonSymbols.NoButtons)
            w.setToolTip("" if custom else "Set by the fluid; choose Custom to type a value")
        _set_silent(self.density, f.density)
        _set_silent(self.viscosity, f.viscosity * 1e6)

    def _set_alpha_widgets(self, deg: float) -> None:
        self.alpha.set_si_value(deg)
        _set_silent(self.alpha_slider, round(deg * 2))

    def _update_ground_enabled(self, d: Design) -> None:
        """Show the Wheels row as what will actually run.

        A disabled but checked "Rotate with the road" on a car without wheels
        read as a setting that is on. When the option cannot apply (no wheel
        bodies, or a fixed floor) the box is shown unchecked; without wheels
        its text says so. The user's choice is kept in ``_wheels_pref`` and
        comes back when the option applies again (belt switched back on).
        """
        wheels = has_wheels(d)
        moving = self.ground_combo.currentData() == "moving"
        applies = moving and wheels
        if self.ground_wheels.isEnabled():
            self._wheels_pref = self.ground_wheels.isChecked()
        self.ground_wheels.setEnabled(applies)
        _set_silent(self.ground_wheels, self._wheels_pref if applies else False)
        self.ground_wheels.setVisible(wheels)
        self.no_wheels_label.setVisible(not wheels)
        three = getattr(d, "dimension", 2) == 3
        self.no_wheels_label.setToolTip(
            "No part of this design is a wheel; mark the wheel parts in Geometry ▸ Parts" if three else
            "This design has no separate wheel bodies; turn Wheels on in the Geometry step to add them")
        self.ground_wheels.setToolTip(
            "Wheel parts spin at the road speed" if applies else
            "Needs a moving belt" if wheels else "")

    # ===================================================================
    # widgets -> model (in place, debounced save)
    # ===================================================================
    def _edit(self) -> Simulation | None:
        """The simulation to modify, or None while loading / without selection."""
        if self._loading:
            return None
        return self.simulation

    def _changed(self) -> None:
        self._dirty = True
        self._save_timer.start()
        self._schedule_summary()

    def flush(self) -> None:
        """Write pending edits now (before running, switching or closing)."""
        self._save_timer.stop()
        if not self._dirty:
            return
        self._dirty = False
        sim = self._sim
        p = self.state.project
        if sim is not None and p is not None and sim.id in p.simulations:
            self.state.update_simulation(sim)

    def _name_edited(self) -> None:
        sim = self._edit()
        name = self.name_edit.text().strip()
        if sim is None:
            return
        if not name:
            set_line_text(self.name_edit, sim.name)
            return
        if name != sim.name:
            sim.name = name
            self._dirty = True
            self.flush()                     # the Project tree shows the name at once

    def _set_analysis(self, key: str) -> None:
        sim = self._edit()
        if sim is None:
            return
        if key not in analyses_for(self.state.design):
            self.analysis.set_current(sim.analysis)
            return
        sim.analysis = key
        self.analysis.set_current(key)
        self._update_visibility(sim, self.state.design)
        self._changed()
        self.update_summary()                # the preview and Estimate change shape at once

    def _fluid_choice(self) -> tuple[str, str | None]:
        """(mode, preset name) of the fluid combo."""
        data = str(self.fluid_combo.currentData() or "custom")
        if data.startswith("preset:"):
            return "preset", data.split(":", 1)[1]
        return data, None

    def _fluid_mode_changed(self, _i: int) -> None:
        mode, _ = self._fluid_choice()
        sim = self.simulation
        if sim is not None and not self._loading:
            was = fluid_mode(sim.conditions.fluid)[0]
            if mode == "isa" and was != "isa":
                self.altitude.set_si_value(0.0)
            elif mode in ("air_t", "water_t") and was not in ("air_t", "water_t"):
                self.temperature.set_si_value(15.0 if mode == "air_t" else 20.0)
        self._fluid_edited()

    def _fluid_edited(self) -> None:
        sim = self._edit()
        if sim is None:
            return
        mode, preset = self._fluid_choice()
        value = None
        if mode == "isa":
            value = self.altitude.si_value()
        elif mode in ("air_t", "water_t"):
            value = self.temperature.si_value()
        try:
            f = make_fluid(mode, value, preset=preset, density=self.density.value(),
                           viscosity=self.viscosity.value() * 1e-6)
        except ValueError as exc:
            self.state.message(str(exc), 5000)
            return
        sim.conditions.fluid = f
        self._loading = True
        try:
            self._show_fluid(f, mode)
        finally:
            self._loading = False
        self._changed()

    def _speed_edited(self, v: float) -> None:
        sim = self._edit()
        if sim is None:
            return
        sim.conditions.speed = float(v)
        self._changed()

    def _alpha_slider_moved(self, v: int) -> None:
        if self._loading:
            return
        self.alpha.set_si_value(v / 2.0)
        self._alpha_edited(v / 2.0)

    def _alpha_edited(self, deg: float) -> None:
        sim = self._edit()
        if sim is None:
            return
        sim.conditions.alpha_deg = float(deg)
        _set_silent(self.alpha_slider, round(deg * 2))
        self._changed()

    def _yaw_slider_moved(self, v: int) -> None:
        if self._loading:
            return
        self.yaw.set_si_value(v / 2.0)
        self._yaw_edited(v / 2.0)

    def _yaw_edited(self, deg: float) -> None:
        sim = self._edit()
        if sim is None:
            return
        sim.conditions.yaw_deg = float(deg)
        _set_silent(self.yaw_slider, round(deg * 2))
        self._changed()

    def _wing_edited(self, field_name: str, value: Any) -> None:
        sim = self._edit()
        if sim is None:
            return
        setattr(sim.wing, field_name, value)
        if field_name == "viscous":
            self.w_ncrit.setEnabled(bool(value))
        self._changed()

    def _wing_mach_from_speed(self) -> None:
        sim = self.simulation
        if sim is None:
            return
        m = mach_number(sim.conditions)
        if m is None:
            self.state.message("The Mach number only applies to air.", 4000)
            return
        self.w_mach.setValue(min(m, 0.85))

    def _ground_edited(self) -> None:
        sim = self._edit()
        if sim is None:
            return
        mode = self.ground_combo.currentData()
        sim.ground.enabled = mode != "none"
        sim.ground.moving = mode == "moving"
        self._update_ground_enabled(self.state.design)
        sim.ground.rotate_wheels = self.ground_wheels.isEnabled() and self.ground_wheels.isChecked()
        self._changed()

    def _quality_edited(self, key: str) -> None:
        sim = self._edit()
        if sim is None:
            return
        sim.cfd.quality = key
        self._changed()

    def _cfd_edited(self, field_name: str, value: Any, *, custom: bool = False) -> None:
        sim = self._edit()
        if sim is None:
            return
        setattr(sim.cfd, field_name, value)
        if custom and value is not None and sim.cfd.quality != "custom":
            # explicit grid/domain/time values only take effect in Custom (model contract)
            sim.cfd.quality = "custom"
            _set_combo(self.quality, "custom")
        self._changed()

    def _re_mode_edited(self, _i: int) -> None:
        mode = self.re_mode.currentData()
        self._re_cap_row.set_enabled(mode == "capped")
        sim = self._edit()
        if sim is None:
            return
        sim.cfd.reynolds_mode = mode
        d = self.state.design
        if mode == "capped" and not sim.cfd.reynolds_cap and d is not None:
            # fix the cap now, as a new simulation does: left to the planner it would follow
            # the grid, and the next quality change would simulate a different flow
            sim.cfd.reynolds_cap = default_reynolds_cap(d)
            _set_silent(self.re_cap, float(sim.cfd.reynolds_cap or 0.0))
        self._changed()

    def _sensitivity_edited(self, on: bool) -> None:
        self.objective.setEnabled(on)
        self._cfd_edited("compute_sensitivity", bool(on))

    def _store_flow_viz(self, sim: Simulation, fv: dict) -> None:
        set_flow_viz(sim, fv)
        self._load_flow_viz(fv)
        self._changed()

    def _smoke_edited(self, key: str, value: Any) -> None:
        sim = self._edit()
        if sim is None:
            return
        fv = flow_viz_of(sim)
        if fv.get(key) == value:
            return
        if key == "kind":
            # the height fields change meaning (one height for a rake, a range otherwise):
            # a value fixed for one kind is no guide for the other, so they follow the body again
            fv["z_min"] = fv["z_max"] = None
        if key in ("kind", "spacing"):
            fv["n"] = None                    # counts follow the spacing on the next layout
        fv[key] = value
        self._store_flow_viz(sim, fv)

    def _smoke_height_edited(self, key: str, value: float) -> None:
        sim = self._edit()
        if sim is None:
            return
        fv = flow_viz_of(sim)
        if fv["z_min"] is None or fv["z_max"] is None:
            # leaving Auto: the range is fixed as a whole, at what the fields show
            fv["z_min"], fv["z_max"] = self.smoke_z_min.si_value(), self.smoke_z_max.si_value()
        fv[key] = float(value)
        if fv["kind"] == "rake":
            fv["z_max"] = fv["z_min"]
        fv["n"] = None                        # a new extent: the counts of a typed spacing change
        self._store_flow_viz(sim, fv)

    def _smoke_auto(self, what: str, on: bool) -> None:
        sim = self._edit()
        if sim is None:
            return
        fv = flow_viz_of(sim)
        if what == "z":
            if on:
                fv["z_min"] = fv["z_max"] = None
            else:
                lo = self.smoke_z_min.si_value()
                fv["z_min"], fv["z_max"] = lo, (lo if fv["kind"] == "rake" else self.smoke_z_max.si_value())
        else:
            fv["spacing"] = None if on else self.smoke_spacing.si_value()
        fv["n"] = None
        self._store_flow_viz(sim, fv)

    def _airfoil_edited(self, field_name: str, value: Any) -> None:
        sim = self._edit()
        if sim is None:
            return
        setattr(sim.airfoil, field_name, value)
        self._changed()

    def _mach_from_speed(self) -> None:
        sim = self.simulation
        if sim is None:
            return
        m = mach_number(sim.conditions)
        if m is None:
            self.state.message("The Mach number only applies to air.", 4000)
            return
        self.mach.setValue(min(m, 0.7))

    # ===================================================================
    # estimate, planner, preview
    # ===================================================================
    def _schedule_summary(self) -> None:
        """Recompute after the debounce - or, while the page is hidden, when it is shown.

        Planning takes ~50 ms on the GUI thread; the page hears every design edit
        (designsChanged) even while the user is on the Geometry page, so the work
        is deferred until somebody can see the result.
        """
        if self.isVisible():
            self._summary_timer.start()
        else:
            self._summary_stale = True

    def _summary_now(self) -> None:
        if self.isVisible():
            self.update_summary()
        else:
            self._summary_stale = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_timer.start()

    def _info_split_moved(self, *_args) -> None:
        self._info_user_sized = True

    def fit_info_height(self) -> None:
        """Size the Estimate / Planner strip to whole Estimate rows.

        The strip used to have a fixed initial height, which cut the last
        visible row in half ("Stagnation …" at the bottom edge) at every
        window size. It now grows until the whole Estimate shows, as long as
        the case preview keeps ``PREVIEW_MIN_FRACTION`` of the height (and at
        least ``PREVIEW_MIN_PX``); when it cannot, it shows as many complete
        rows as fit and the rest scrolls. Once the user drags the splitter
        their size is kept.
        """
        self._fit_timer.stop()
        if self._info_user_sized:
            return
        sizes = self._vsplit.sizes()
        total = sum(sizes)
        n = len(self.estimate.items())
        if len(sizes) != 2 or total <= 0 or n == 0:
            return
        chrome = self._info_head_h + 2 * self._info.frameWidth()
        bottom = 8                                     # the Estimate body's bottom margin
        full = chrome + self.estimate.height_for_rows(n) + bottom
        if self.run_blocker is not None:
            # a refused run shows its whole Planner line and the button that fixes it (the Estimate has
            # three rows then, and sizing to them cut the button off below the fold at 1280 x 800)
            lines = self._planner_lines_widget
            # measured at the width left beside a scroll bar, so the answer cannot itself need one
            bar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
            need = lines.heightForWidth(max(lines.width() - bar, 1))
            # a scroll area keeps its widget at least at its minimum size hint, which for wrapped labels
            # is measured at Qt's own trial width and can exceed the height for the actual width
            need = max(need, lines.minimumSizeHint().height())
            full = max(full, chrome + bottom + need)
        stacked = self.info_stacked()
        if stacked:
            # the Planner sits under the Estimate: the strip holds both blocks
            lines = self._planner_lines_widget
            bar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
            full += 1 + self._info_head_h + max(lines.heightForWidth(max(lines.width() - bar, 1)), 0) + bottom
        frac = PREVIEW_MIN_FRACTION_3D if self._preview_stack.currentIndex() == 1 else PREVIEW_MIN_FRACTION
        if self._has_placed_physics():
            frac = max(frac, PREVIEW_MIN_FRACTION_PHYSICS)
        cap = total - max(PREVIEW_MIN_PX, int(frac * total))
        if full <= cap:
            h = full
        elif stacked:
            h = cap                     # both blocks scroll in their halves
        else:
            pitch = max(self.estimate.row_pitch(), 1)
            k = max(1, (cap - chrome + self.estimate.grid.verticalSpacing()) // pitch)
            h = chrome + self.estimate.height_for_rows(k)
        h = max(h, self._info.minimumHeight())
        if sizes[1] != h:
            self._vsplit.setSizes([total - h, h])

    def _has_placed_physics(self) -> bool:
        sim = self.simulation
        return sim is not None and sim.analysis == "cfd" and (
            sim.thermal.enabled or sim.cfd.domain_mode == "closed"
            or any(phys.items_of(sim, k) for k in phys.ITEM_LISTS))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_timer.start()
        if self._summary_stale:
            self.update_summary()

    def update_summary(self) -> None:
        """Recompute the Estimate, the Planner lines and the case preview."""
        self._summary_timer.stop()
        self._summary_stale = False
        sim, d = self.simulation, self.state.design
        rows: list[tuple[str, str]] = []
        notes: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        self.last_plan = None
        self.run_blocker = None
        self._line_actions = {}
        if sim is None or d is None:
            self.plan_status = ""
            self._render([], [], self._context_lines(sim), "")
            self.preview.set_case(d)
            self._reset_quality_texts()
            self._update_physics_preview()
            return
        three = getattr(d, "dimension", 2) == 3
        if sim.analysis == "wing":
            self._wing_summary(d, sim)
            return
        cond = sim.conditions
        re_real = cond.reynolds(d.length_ref)
        mach = mach_number(cond)
        q = units.fmt(cond.dynamic_pressure(), "pressure", display_unit("pressure"), 3)
        if sim.analysis == "polar":
            alphas = sim.airfoil.alphas()
            rows = [("Reynolds number", units.fmt_reynolds(re_real)),
                    ("Mach number", f"{sim.airfoil.mach:.2f}"),
                    ("Angles", f"{len(alphas)} ({_deg(alphas[0])} to {_deg(alphas[-1])})"),
                    ("Ncrit", f"{sim.airfoil.ncrit:g}"),
                    ("Dynamic pressure", q),
                    ("Method", "Viscous-inviscid panel method (mfoil)"),
                    ("Estimated run time", format_duration(polar_seconds(len(alphas))))]
            if re_real < 1e4:
                warnings.append("Re below 10,000: panel methods are unreliable this low; use CFD.")
            if mach is not None and mach > 0.05 and sim.airfoil.mach == 0:
                notes.append(f"The flow speed is Mach {mach:.2f}; set the Mach number to include "
                             "compressibility.")
            self.plan_status = "polar"
            self.last_summary = {r[0]: r[1] for r in rows}
            self._render(rows, [("warning", w) for w in warnings] + [("info", n) for n in notes],
                         self._context_lines(sim), "")
            self.preview.set_case(d, alpha_deg=0.0, ground=None, speed_text="")
            self._update_physics_preview()
            return
        state = self._planner.state
        plan = None
        blocker = run_blocker(d, sim, self.state.project)
        self.run_blocker = blocker
        if blocker is not None:
            # a run that can only fail on the solver: said here, with the change that fixes it as a button
            errors = [blocker.text]
            self._line_actions[blocker.text] = (blocker.action, "Changes: " + "; ".join(blocker.changes) + ".",
                                                lambda *_a: QTimer.singleShot(0, self.apply_run_fix))
            rows = [("Real Reynolds number", units.fmt_reynolds(re_real))]
            self.plan_status = "error"
            note = "Not planned"
        elif state == "ready":
            try:
                if three:
                    plan = self._planner.module.plan_simulation(
                        d, sim, airfoil_coords=airfoil_resolver(self.state.project))
                else:
                    plan = self._planner.module.plan_simulation(d, sim)
                self.last_plan = plan
                rows = self._plan_rows(plan)
                notes = [str(n) for n in getattr(plan, "notes", [])]
                warnings = [str(w) for w in (self._measured_warnings(plan, d, sim)
                                             or getattr(plan, "warnings", []))]
                self.plan_status = "ok"
                note = ""
            except Exception as exc:  # noqa: BLE001 - a bad setup is reported, not raised
                errors = [plan_failure_text(d, sim, exc)]
                rows = [("Real Reynolds number", units.fmt_reynolds(re_real))]
                self.plan_status = "error"
                note = "Not planned"
        else:
            rows = [("Real Reynolds number", units.fmt_reynolds(re_real))]
            if state in ("idle", "loading"):
                self.plan_status = "loading"
                note = "Loading planner…"
            else:
                self.plan_status = "basic"
                note = "Planner unavailable"
                errors.append("The simulation planner could not be loaded"
                              + (f": {self._planner.error}" if self._planner.error else "") + ".")
        rows.insert(min(2, len(rows)), ("Dynamic pressure", q))
        if mach is not None:
            rows.insert(min(3, len(rows)), ("Mach number", f"{mach:.3f}"))
            if mach > 0.3:
                warnings.append(f"Mach {mach:.2f}: the solver is incompressible; above about Mach 0.3 "
                                "compressibility changes the forces noticeably.")
        smoke = None
        has_smoke_row = any(str(r[0]).lower().startswith(("smoke", "flow visualisation")) for r in rows)
        if three and not sim.emitters:
            # no emitters: the run releases its smoke from the automatic rake (v0.2), shown as such
            fv = flow_viz_of(sim)
            smoke = self._smoke_layout(d, sim, plan, fv)
            self.last_smoke = smoke
            if not has_smoke_row:
                rows.append(("Smoke", smoke_fact(smoke, fv)))
            if smoke is not None and smoke["count"] > MAX_SMOKE_NOZZLES:
                warnings.append(f"The smoke grid has {smoke['count']:,} nozzles: above about {MAX_SMOKE_NOZZLES} "
                                "the streaks merge into one sheet. Use a larger spacing, or the Plane emitter.")
        elif sim.emitters and not has_smoke_row:
            self.last_smoke = None
            rows.append(("Smoke", emitters_fact(sim)))
        # heat transfer: the planner's own groups when it states them, else this page's estimate
        if sim.thermal.enabled:
            have = {str(r[0]) for r in rows}
            rows += [r for r in phys.thermal_rows(phys.thermal_numbers(d, sim)) if r[0] not in have]
        ext = self.physics_extent() if plan is not None else None
        p_warn, p_notes = phys.physics_warnings(d, sim, ext)
        said = [*warnings, *notes]                       # the planner's own lines, before ours join them
        # what the installed planner would leave out reads first: it changes what every line below means
        gaps = planner_gaps(sim, plan, ext, self._plan_rows(plan) if plan is not None else [])
        warnings = gaps + [w for w in warnings if w not in gaps]
        warnings += [w for w in p_warn if w not in warnings and not said_by_planner(w, said)]
        notes += [n for n in p_notes if n not in notes and not said_by_planner(n, said)]
        for sec in self.physics_sections():
            sec.extent = ext
        self.domain_section.reload()
        rows = order_estimate(rows)
        self.last_summary = {r[0]: r[1] for r in rows}
        lines = ([("error", e) for e in errors] + [("warning", w) for w in warnings]
                 + [("info", n) for n in notes])
        self._render(rows, lines, self._context_lines(sim), note)
        if three:
            self._update_3d_extras(d, sim, plan)
            self._load_flow_viz(flow_viz_of(sim), smoke)
            self._update_preview3d(d, sim, plan, smoke)
        else:
            self._reset_quality_texts()
            self._update_preview(d, sim, plan)
        self._sync_physics_rows()
        self._update_physics_preview()

    # -- 3D and wing ----------------------------------------------------------------
    def _wing_summary(self, d: Design, sim: Simulation) -> None:
        """Estimate and planner lines of a finite-wing analysis (no CFD planner involved)."""
        resolver = airfoil_resolver(self.state.project)
        try:
            rows, notes, warnings = wing_estimate(d, sim, resolver)
        except Exception as exc:  # noqa: BLE001 - an estimate failure is reported, not raised
            rows, notes, warnings = [], [], [f"The wing analysis cannot be estimated: {exc}"]
        q = units.fmt(sim.conditions.dynamic_pressure(), "pressure", display_unit("pressure"), 3)
        if rows:
            rows.insert(min(3, len(rows)), ("Dynamic pressure", q))
        self.plan_status = "wing" if rows else "error"
        self.last_plan = None
        self.last_summary = dict(rows)
        self._render(rows, [("warning", w) for w in warnings] + [("info", n) for n in notes],
                     self._context_lines(sim), "" if rows else "Not estimated")
        self._reset_quality_texts()
        meshes = design_meshes3d(d, resolver)
        speed = units.fmt(sim.conditions.speed, "speed", self.speed.unit, 3)
        ws = sim.wing
        slip = f", sideslip {_deg(sim.conditions.yaw_deg)}" if sim.conditions.yaw_deg else ""
        self.preview3d.set_case(d, meshes=meshes, extent3=None, symmetry=False, ground=None,
                                speed_text=f"U = {speed}",
                                attitude_text=f"α {_deg(ws.alpha_start)} to {_deg(ws.alpha_end)}{slip}")
        self._update_physics_preview()

    def _reset_quality_texts(self) -> None:
        for i, (_k, t, _tip) in enumerate(QUALITY_ITEMS):
            if self.quality.itemText(i) != t:
                self.quality.setItemText(i, t)

    def quality_cells(self, d: Design, sim: Simulation) -> dict[str, int | None]:
        """Cells each preset quality gives this 3D case (None when it cannot be planned).

        Shown in the Quality combo ("Standard · 5.6 M cells"): in 3D the step from
        Draft to Fine is a factor of five to ten in cells and in run time, and the
        user should see it before choosing. Plans are cheap after the first (the
        planner caches the geometry facts), ~20 ms each. Counts are the cells
        solved (half the tunnel with a symmetry plane), which set the run time, as
        in the Estimate's "Cells (3D)" row.
        """
        out: dict[str, int | None] = {}
        mod = self._planner.module
        resolver = airfoil_resolver(self.state.project)
        for q in ("draft", "standard", "fine"):
            trial = Simulation.from_dict(sim.to_dict())
            trial.cfd.quality = q
            try:
                plan = mod.plan_simulation(d, trial, airfoil_coords=resolver)
                out[q] = int(plan.cells)
            except Exception:  # noqa: BLE001 - a preset that does not fit shows no count
                out[q] = None
        return out

    def _update_3d_extras(self, d: Design, sim: Simulation, plan) -> None:
        """Quality cell counts, the symmetry decision and the sensitivity-map cost."""
        if self._planner.state == "ready":
            counts = self.quality_cells(d, sim)
            for i, (k, t, _tip) in enumerate(QUALITY_ITEMS):
                if k in counts:
                    c = counts[k]
                    self.quality.setItemText(i, f"{t} · {c / 1e6:.1f} M cells" if c else f"{t} · does not fit")
        _half, why = symmetry_reason(d, sim, plan)
        self.symmetry_note.setText(why)
        self.sensitivity_note.setText(self._sensitivity_text(plan))

    def _sensitivity_text(self, plan) -> str:
        """What the 3D sensitivity map would add: time and device memory, or why it cannot run."""
        if plan is None:
            return ""
        try:
            import importlib
            plan3d = importlib.import_module("aerolume.simulation.plan3d")
            limit = plan3d.sensitivity_max_cells(plan.device)
            if plan.cells > limit:
                return (f"Not on this grid: {plan.cells / 1e6:.2f} M cells, the map is limited to "
                        f"{limit / 1e6:.1f} M (Draft or the symmetry plane makes it fit).")
            s3 = importlib.import_module("aerolume.optimize.sensitivity3d")
            sens = importlib.import_module("aerolume.optimize.sensitivity")
            secs = float(s3.estimate_seconds(plan))
            gpu = not str(plan.device).startswith("cpu")
            lead, avg = sens.STEADY_WINDOW["gpu" if gpu else "cpu"]
            unit = max(float(getattr(plan, "strouhal_length", plan.length_ref) or plan.length_ref)
                       / float(plan.length_ref), 1e-6)
            steps = int((lead + avg) * unit / max(s3.DT_SAFETY * float(plan.dt), 1e-12))
            mem = s3.estimate_memory_bytes(int(plan.cells), steps=steps)
            return f"Adds about {format_duration(secs).lstrip('~')} and {mem / 1e9:.1f} GB of device memory."
        except Exception:  # noqa: BLE001 - advisory only
            return ""

    def _case_meshes(self, d: Design, plan) -> tuple[list, tuple | None, bool]:
        """(wind-frame meshes, full domain, symmetry plane?) of the planned case, or the
        design's own surfaces when there is no plan yet."""
        if plan is not None and getattr(plan, "bodies_wind3", None):
            return list(plan.bodies_wind3), plan.extent3_full, bool(plan.symmetry_plane)
        return design_meshes3d(d, airfoil_resolver(self.state.project)), None, False

    def _smoke_layout(self, d: Design, sim: Simulation, plan, fv: dict | None = None, *,
                      force: bool = False) -> dict | None:
        """Where this case releases its smoke (``smoke_rake_layout``): the rake dict resolved
        against the plan's wind-frame body box, full domain and cell size, as the runner
        resolves it, so the preview shows the nozzles the run will use.

        A typed spacing is stored with the nozzle counts it gives (the rake dict the
        runner reads has counts, not a pitch); when the body or the domain changed
        those counts, they are brought back in step here and saved with the next edit.
        """
        fv = flow_viz_of(sim) if fv is None else fv
        if (not fv["smoke"] and not force) or sim.analysis != "cfd":
            return None
        if force:
            fv = {**fv, "smoke": True}
        meshes, extent, _sym = self._case_meshes(d, plan)
        road = (sim.ground.enabled and shows_ground(d)) or stands_on_road(d)
        bounds, cell = None, None
        if plan is not None:
            bounds = getattr(plan, "body_bounds", None)
            with contextlib.suppress(Exception):          # cell size: the default nozzle radius
                cell = float(plan.solver_config.h) * float(plan.length_ref)
        if bounds is None:
            bounds = meshes_bounds(meshes)
        layout = smoke_rake_layout(fv, bounds, on_road=road, extent3=extent, cell=cell)
        if (layout is not None and "error" not in layout and fv["spacing"] is not None and plan is not None
                and not force and not sim.emitters):
            n = layout["spec"].get("n")
            if n is not None and n != fv["n"]:
                fv["n"] = n
                set_flow_viz(sim, fv)
                self._dirty = True
                self._save_timer.start()
        return layout

    def _update_preview3d(self, d: Design, sim: Simulation, plan, smoke: dict | None = None) -> None:
        ground = None
        if sim.ground.enabled and shows_ground(d):
            ground = "moving" if sim.ground.moving else "fixed"
        speed = units.fmt(sim.conditions.speed, "speed", self.speed.unit, 3)
        meshes, extent, sym = self._case_meshes(d, plan)
        c = sim.conditions
        att = ("Pitch" if d.kind == "vehicle" else "α") + f" {_deg(c.alpha_deg)}, yaw {_deg(c.yaw_deg)}"
        closed = sim.cfd.domain_mode == "closed"
        self.preview3d.set_case(d, meshes=meshes, extent3=extent, symmetry=sym, ground=ground,
                                speed_text="" if closed else f"U = {speed}", attitude_text=att, smoke=smoke,
                                inflow=not closed)

    @staticmethod
    def _plan_rows(plan) -> list[tuple[str, str]]:
        """Label/value rows for a SimPlan: its own ``summary()`` when present."""
        summ = getattr(plan, "summary", None)
        d = None
        if callable(summ):
            with contextlib.suppress(Exception):      # fall back to the raw fields
                d = summ()
        if isinstance(d, dict) and d:
            rows = [(str(k), str(v)) for k, v in d.items()]
            if getattr(plan, "dimension", 2) == 3 and "Cells (3D)" in d:
                rows = [r for r in rows if r[0] != "Cells"]      # "Cells (3D)" says it with the half domain
            return rows
        rows = [("Real Reynolds number", units.fmt_reynolds(float(getattr(plan, "re_real", float("nan"))))),
                ("Simulated Reynolds number",
                 units.fmt_reynolds(float(getattr(plan, "re_sim", float("nan")))))]
        cfg = getattr(plan, "solver_config", None)
        if cfg is not None:
            rows.append(("Grid", f"{cfg.nx:,} × {cfg.ny:,}"))
        cells = getattr(plan, "cells", None)
        if cells:
            rows.append(("Cells", f"{int(cells):,}"))
        est = getattr(plan, "est_seconds", None)
        if est is not None and math.isfinite(est):
            rows.append(("Estimated run time", format_duration(float(est))))
        return rows

    def _context_lines(self, sim: Simulation | None) -> list[tuple[str, str]]:
        """Facts about the simulation itself that change what an edit means."""
        if sim is None:
            return []
        status = self._busy(sim) or sim.status
        if status in ("running", "queued"):
            return [("info", f"{status.capitalize()}: changes apply to the next run.")]
        if status == "done":
            return [("info", ("Has results: changes apply to the next run. Duplicate the simulation "
                              "to keep both."))]
        if status == "failed" and sim.error:
            return [("error", f"The last run failed: {sim.error}")]
        return []

    def _measured_warnings(self, plan, design, sim) -> list[str] | None:
        """The plan's warnings re-stated on the temperature rise the selected simulation's stored result
        measured (:func:`aerolume.simulation.physics_run.warnings_after_run`), or None.

        Only for a result run with exactly these settings (the same configuration hash): a heater
        room's "β·ΔT = 0.15 … loses accuracy" stayed on Setup after its run had measured β·ΔT 0.014
        (critic round 3.4), while a result of other settings says nothing about these. Cheap: the
        stored result's scalars (``result_summary``), never its arrays."""
        if getattr(plan, "placed", None) is None or not getattr(getattr(sim, "thermal", None), "enabled", False):
            return None
        project = getattr(self.state, "project", None)
        try:
            summ = project.result_summary(sim.id) if project is not None else None
            if not summ or not summ.get("diagnostics"):
                return None
            from ...project.provenance import config_hash
            rec = project.provenance(sim.id) or {}
            if not isinstance(rec, dict) or rec.get("config_hash") != config_hash(design, sim):
                return None
            import types

            from ...simulation.physics_run import warnings_after_run
            return warnings_after_run(plan, types.SimpleNamespace(diagnostics=dict(summ["diagnostics"])))
        except Exception:  # noqa: BLE001 - the plan's own warnings stand
            return None

    def _render(self, rows: list[tuple[str, str]], lines: list[tuple[str, str]],
                context: list[tuple[str, str]], note: str) -> None:
        self.estimate.set_items(compact_rows(rows))
        self.estimate_note.setText(note)
        set_prop(self.estimate_note, "role", "status-bad" if self.plan_status in ("error", "basic") else "muted")
        if self.simulation is not None:
            # a setup the planner refuses (a grid that cannot fit in memory, a car pitched into
            # the road) would only fail again on the worker thread: say so here instead
            self.run_button.setEnabled(self.plan_status != "error")
            self.run_button.setToolTip("Fix the setup first (see the Planner list)" if self.plan_status == "error"
                                       else "Queue this simulation and show it on the Simulate page (F5)")
        clear_layout(self.planner_box)
        shown = context + lines
        actions = getattr(self, "_line_actions", {})
        for level, text in shown:
            ml = PlannerLine(text, level, action=actions.get(text))
            ml.setProperty("summary_kind", {"warning": "warning", "error": "error"}.get(level, "note"))
            self.planner_box.addWidget(ml)
        if not shown:
            self.planner_box.addWidget(label("No notes.", "muted"))
        self.planner_box.addStretch(1)
        self.planner_note.setText(planner_count_text(lines))
        n_err = sum(1 for lv, _ in lines if lv == "error")
        n_warn = sum(1 for lv, _ in lines if lv == "warning")
        set_prop(self.planner_note, "role", "status-bad" if n_err else ("status-warn" if n_warn else "muted"))
        self.planner_lines = list(lines)
        if lines:
            self._log_timer.start()
        self._fit_timer.start()

    def flush_planner_log(self) -> None:
        """Log the planner lines that are new for this simulation to the Messages dock.

        Called after the user pauses (``LOG_DELAY_MS``), so dragging the angle
        slider logs the final decision once instead of every intermediate one;
        lines already logged for the same simulation are not repeated.
        """
        self._log_timer.stop()
        sim = self.simulation
        if sim is None or not self.planner_lines:
            return
        seen = self._logged.setdefault(sim.id, set())
        current = {text for _lv, text in self.planner_lines}
        for level, text in self.planner_lines:
            if text not in seen:
                self.state.log(pretty(text), "warning" if level == "warn" else level, LOG_SOURCE)
        self._logged[sim.id] = current

    def _update_preview(self, d: Design, sim: Simulation, plan) -> None:
        ground = None
        if sim.ground.enabled and shows_ground(d):
            ground = "moving" if sim.ground.moving else "fixed"
        speed = units.fmt(sim.conditions.speed, "speed", self.speed.unit, 3)
        extent = getattr(plan, "extent", None) if plan is not None else None
        bodies = getattr(plan, "bodies_wind", None) if plan is not None else None
        closed = sim.cfd.domain_mode == "closed"
        self.preview.set_case(d, alpha_deg=sim.conditions.alpha_deg, extent=extent,
                              speed_text="" if closed else f"U = {speed}", ground=ground, bodies_wind=bodies,
                              inflow=not closed)

    # ===================================================================
    # simulation actions (the Project tree has the same ones)
    # ===================================================================
    def _unique_name(self, base: str) -> str:
        p = self.state.project
        names = {s.name for s in p.simulations.values()} if p else set()
        if base not in names:
            return base
        i = 2
        while f"{base} {i}" in names:
            i += 1
        return f"{base} {i}"

    def new_simulation(self) -> Simulation | None:
        """Add a simulation of the current design and select it.

        It starts from the project's latest comparable CFD run (same conditions,
        quality and Reynolds handling; :func:`comparable_template`), so designs
        added to a study are compared at the conditions the study already uses.
        """
        d = self.state.design
        p = self.state.project
        if p is None or d is None:
            return None
        self.flush()
        n = len(self._sims_of(d.id)) + 1
        template = comparable_template(p, d)
        template_design = p.designs.get(template.design_id) if template is not None else None
        sim = default_simulation(d, self._unique_name(f"{d.name} · run {n}"), template, template_design)
        if template is None or template.design_id != d.id or not has_body(d):
            # no run of this design to copy (or, for a design without a body, possibly a tunnel copy):
            # the case the design was built from decides the domain and the heat transfer
            source = inherit_design_physics(p, d, sim)
            if source:
                self.state.log(f"“{sim.name}” starts from the domain and heat transfer of {source}.", "info",
                               "Setup")
        self.state.add_simulation(sim, select=True)
        return sim

    def new_from_template(self, key: str) -> tuple | None:
        """Add a starter case (:data:`~aerolume.app.dialogs.physics_templates.PHYSICS_TEMPLATES`) to the
        project: its design and first simulation, selected. ``navigate("setup:template:<key>")`` does
        the same from elsewhere."""
        if self.state.project is None:
            self.state.message("Open or create a project first.", 5000)
            return None
        self.flush()
        title = next((t.title for t in PHYSICS_TEMPLATES if t.key == key), key)
        try:
            design, sims = add_template(self.state, key)
        except Exception as exc:  # noqa: BLE001 - a template that cannot be built is reported, not raised
            self.state.log(f"Could not add the {title} case: {exc}", "error", "Setup")
            self.state.message(f"Could not add the {title} case: {exc}", 8000)
            return None
        self.state.log(f"Added the {title} case: design “{design.name}” with {len(sims)} simulation"
                       f"{'s' if len(sims) != 1 else ''}.", "info", "Setup")
        self._refresh()
        return design, sims

    def duplicate_simulation(self) -> Simulation | None:
        sim = self.simulation
        p = self.state.project
        if sim is None or p is None:
            return None
        self.flush()
        dup = Simulation.from_dict(sim.to_dict())
        from ...model import new_id, now_iso
        dup.id, dup.created = new_id(), now_iso()
        dup.status, dup.finished, dup.error = "new", None, None
        dup.name = self._unique_name(f"{sim.name} copy")
        copy_flow_viz(sim, dup)
        self.state.add_simulation(dup, select=True)
        return dup

    def delete_simulation(self) -> None:
        sim = self.simulation
        p = self.state.project
        if sim is None or p is None:
            return
        text = f"Delete “{sim.name}”" + (" and its results" if sim.status == "done" else "") + \
            "? This cannot be undone."
        if not self._confirm("Delete simulation", text):
            return
        if self.jobs is not None and self._busy(sim):
            with contextlib.suppress(Exception):      # job manager is optional here
                self.jobs.cancel(sim.id)
        self._save_timer.stop()
        self._dirty = False
        self._sim = None
        d = self.state.design
        self.state.remove_simulation(sim.id)
        rest = self._sims_of(d.id) if d is not None else []
        if rest:
            self.state.select_simulation(rest[0].id)
        else:
            self._refresh()

    # ===================================================================
    # run
    # ===================================================================
    def run_simulation(self) -> None:
        """Save, queue the current simulation with the JobManager and show the Simulate page."""
        sim = self.simulation
        if sim is None:
            return
        self.flush()
        blocker = run_blocker(self.state.design, sim, self.state.project)
        if blocker is not None:
            # never hand the solver a run it can only fail: the Planner line carries the fix
            self.state.message(f"“{sim.name}” cannot run as a wind tunnel: its design has no body. "
                               f"Use {blocker.action} in the Planner list.", 8000)
            self.update_summary()
            return
        if self.jobs is None:
            self.state.message("No job manager available to run simulations.", 6000)
            return
        try:
            self.jobs.submit(sim.id)
        except Exception as exc:  # noqa: BLE001 - surfaced in the status bar and the log
            self.state.log(f"Could not start “{sim.name}”: {exc}", "error", "Setup")
            self.state.message(f"Could not start the simulation: {exc}", 8000)
            return
        self.state.navigate("simulate:run")

    def explore_live(self) -> None:
        sim = self.simulation
        if sim is None:
            return
        self.flush()
        blocker = run_blocker(self.state.design, sim, self.state.project)
        if blocker is not None:
            self.state.message(f"“{sim.name}” cannot be explored as a wind tunnel: its design has no body. "
                               f"Use {blocker.action} in the Planner list.", 8000)
            self.update_summary()
            return
        self.state.navigate("simulate:live")

    def apply_run_fix(self) -> bool:
        """The Planner's fix button: apply :attr:`run_blocker`'s change to the current simulation, save it
        and show the result (the form, the preview and the replanned Estimate). False when there was
        nothing to fix, e.g. the simulation changed under the button."""
        sim, d = self._edit(), self.state.design
        if sim is None or d is None:
            return False
        blocker = run_blocker(d, sim, self.state.project)
        if blocker is None:
            return False
        apply_run_fix(sim, blocker, phys.dimension_of(d))
        self._dirty = True
        self.flush()
        self._refresh(reload_form=True)
        self.state.log(f"“{sim.name}”: {'; '.join(blocker.changes)}.", "info", "Setup")
        self.state.message(f"“{sim.name}” is now a closed box and can run.", 5000)
        return True
