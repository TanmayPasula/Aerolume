"""Simulate page: run a full simulation or explore the flow in real time.

The viewport is the page: the flow is what users came to see, so the
:class:`FieldView` takes all the space left of a narrow inspector, with a
compact viewport toolbar above it (field, colour map, overlays, fit). The
inspector follows the order a run unfolds, as property-grid groups:

* **Run** - status, a thin progress bar, phase, simulated time, rate, time
  remaining, the plan (grid, Reynolds number, estimate) and Run / Stop.
* **Monitors** - Cd, Cl (and L/D for airfoils) as numbers plus stacked
  Cd(t) / Cl(t) panels, which is how an engineer judges whether a run has
  settled. The time axis is fixed to the planned run, and the vertical range
  ignores the impulsive start, so the first frames do not look broken.
* **Explore** (collapsed) - a :class:`LiveSession` through a :class:`LiveWorker`:
  coarser and unaveraged, but the angle and speed sliders act while the flow
  runs; one button turns the explored settings into a full simulation.
* **Queue** - only shown while simulations wait.

A polar sweep has no flow field: the viewport then shows the four polar plots
filling in angle by angle (parsed from the solver's progress messages, then
replaced by the saved result), the field toolbar gives way to a one-line
caption, and the monitors plot Cd and Cl against the angle.

**3D CFD** (DESIGN.md 7.7, 7.13). The viewport is a live **3D view**
(:class:`~aerolume.app.widgets.view3d.View3D` through
:class:`~aerolume.app.widgets.view3d_panel.FlowScene`): the body, the **smoke**
the solver releases from the seed rake and carries through the volume (the
passive scalar ``dye`` of ``LiveFrame.volume``), optional **streamlines**
through the instantaneous velocity from the same rake, and the current **slice
plane**, painted with the live slice's own colours. The toggles sit in the
slice strip next to the slice controls (normal y = side view, z = plan view,
x = cross-flow view; position in metres), which are bound to the job manager's
shared ``ViewSettings`` (``slice_axis`` / ``slice_pos``) so the running solver
cuts where the user looks; ``view3d``, ``smoke``, ``streamlines3d`` and
``seed_rake`` tell it what volume data to send. The rake starts as the emitter
chosen in Setup ▸ Flow visualisation (rake, vertical rake, grid or plane band,
placed by the solver's own rule) and is moved in the inspector (position x,
height, span, lines) or by its handle in the view; the move belongs to that
simulation (``JobManager.set_seed_rake``), so its runs release their smoke
there and a run of another case keeps its own rake. The slice also
stays available as a 2D field view (View: Slice, or Split for both). 3D
updates are throttled by their measured cost (:data:`FLOW_DUTY`) so the window
keeps the display rate while a run streams frames. The Run panel names the 3D
grid and whether a symmetry plane halves the domain; the monitors show
C<sub>D</sub>, C<sub>L</sub> (and C<sub>S</sub> when the run is yawed). After a
run the view shows the stored result: mean-flow streamlines, the smoke
snapshot, the mean slice, surface C<sub>p</sub> on request.

**Finite-wing sweeps** have no flow field either. A cold viscous sweep spends
nearly all its time on 2D section polars, so the viewport lists the section
polars as they are computed and plots their cl(α) live (parsed from the
progress messages), then adds the wing's own CL-α when the result arrives.

The plan shown in the Run group is never stale. Plans are cached by a
fingerprint of everything that decides them (the simulation's settings and the
design, a sample of its mesh arrays included), not by the simulation's id, so
an edit in Setup is a new plan; and while a run is going the panel shows the
plan the runner itself made (``JobManager.jobPlanned``), not one the page made
from settings that may have changed since. Plans are made off the GUI thread
from deep copies (:func:`aerolume.app.jobs.clone_design`): a ``to_dict`` copy
of a 3D design has no triangles.

Full simulations go through the :class:`JobManager` queue and end with a saved,
time-averaged result. Only one of the two runs at a time, because both want the
whole GPU. The page writes the run log to the Messages dock
(``state.log(..., source="Solver")``): plan, progress milestones, the end of
the transient, the averaged coefficients and every warning of the result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
from collections import OrderedDict

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.units import sig
from ...model import CFD3DResult, CFDResult, PolarResult, Simulation, WingResult, now_iso
from .. import theme
from ..jobs import (
    JobManager,
    LiveWorker,
    airfoil_designs,
    clone_design,
    clone_simulation,
    require_api,
    run_in_background,
)
from ..widgets.common import ElidedLabel, MessageLine, PropertyGrid, SegmentedControl, ToolStrip
from ..widgets.diagnostics import DiagnosticsLog, DiagnosticsPanel
from ..widgets.draw_tools import (
    DrawToolStrip,
    MoveTool,
    ObstacleSketch,
    SketchOp,
    SketchTool,
    apply_move,
    placements_of,
    tool_icon,
)
from ..widgets.field_view import (
    SLICE_AXIS_NAMES,
    FieldView,
    LiveRangeTracker,
    ViewportToolbar,
    body_window,
    default_cmap3d,
    display_range3d,
    dye_legend,
    field_info3d,
    particle_colors,
    split_orientation,
    view_fields_for,
)
from ..widgets.kpi import (
    MINUS,
    STEADY_TEXT,
    EmptyState,
    StatusLabel,
    WarningLine,
    button,
    decimals_from_error,
    fmt_duration,
    fmt_fixed,
    fmt_num,
    fmt_re,
    fmt_speed,
    label,
    pretty,
    value_error_texts,
)
from ..widgets.plots import HistoryPlot, PolarGrid, lazy_import
from ..widgets.results3d import (
    crop_slice,
    frame_to_slice,
    mesh_sections,
    neutral_plane,
    view_mode_control,
)
from ..widgets.trace_viewer import TraceViewer
from ..widgets.view3d_panel import (
    DEFAULT_SHOWN_LIVE,
    FlowScene,
    FlowToggles,
    RakeEditor,
    RakeSpec,
    bounds_of,
    dye_colours_of,
    flow_focus_box,
    has_velocity,
    heat_display_range,
    heat_volume_of,
    keep_whole,
    near_field_box,
    plane_window,
    setup_flow_viz,
    smoke_channels,
    smoke_volume_of,
    temperature_field_of,
    volume_box,
)
from ..widgets.wing_plots import WingRunView
from .results import closed_rect, deg_text, ground_of, is_lifting

PHASE_TEXT = {"transient": "Transient", "averaging": "Averaging", "done": "Done", "live": "Live",
              "queued": "Queued", "starting": "Starting", "stopping": "Stopping", "polars": "Section polars",
              "solve": "Wing solve"}
STATUS_TEXT = {"new": ("Not run", "muted"), "queued": ("Queued", None), "running": ("Running", "accent"),
               "done": ("Done", "good"), "failed": ("Failed — see Messages", "bad"),
               "cancelled": ("Stopped", "warn")}
LOG_SOURCE = "Solver"
MILESTONES = (0.25, 0.5, 0.75)
#: "… (t = 64.8 of 92)" in the runner's progress messages: the end time it is actually
#: running to. The runner may extend a run past the plan's t_end (a long start-up
#: transient), and the status bar shows this message, so the panel reads it too.
_T_END_RE = re.compile(r"t = ([0-9.]+) of ([0-9.]+(?:e[+-]?\d+)?)")
#: One angle of a polar sweep as the airfoil solver reports it:
#: "Polar: alpha = 4 deg: Cl = 0.718, Cd = 0.00697" or "... alpha = 15 deg: not converged".
_POLAR_POINT_RE = re.compile(r"alpha = (-?[0-9.]+(?:e[+-]?\d+)?) deg: (?:Cl = (-?[0-9.]+), Cd = (-?[0-9.]+)|"
                             r"(not converged))")
#: A wing run's section-polar progress: "Wing: Section polar 2412, Re 3.6e+05: alpha = 4 deg: Cl = ..."
_WING_SECTION_RE = re.compile(r"Section polar (.+?), Re ([0-9.]+(?:e[+-]?\d+)?): (.*)$")
#: A wing run's own angle: "Wing: alpha = 4 deg" (the lattice / lifting-line solve of one angle)
_WING_ANGLE_RE = re.compile(r"(?:^|: )alpha = (-?[0-9.]+(?:e[+-]?\d+)?) deg$")
#: Until this share of the run, "Remaining" is the planner's estimate: the measured rate
#: of the first steps includes CUDA-graph capture and warm-up and is far too optimistic.
ETA_MEASURED_FROM = 0.05
#: Weight of the newest measurement in the smoothed "Remaining" value.
ETA_SMOOTHING = 0.15
#: The monitor's vertical range ignores samples before this time (L/U): the impulsive
#: start puts Cd near 100 in the first steps, which would flatten everything after it.
MONITOR_T_MIN = 0.5
#: Plans kept per page (a plan is small; this only bounds a long editing session).
PLAN_CACHE_SIZE = 32
#: Simulation fields that do not change the plan (identity, bookkeeping).
_PLAN_IRRELEVANT = ("id", "name", "status", "created", "finished", "notes", "error")
#: The live 3D view is updated in *ticks*: one tick applies the newest slice, smoke volume and
#: streamlines together, so it costs one render (~20 ms at 1000 x 800 with smoke, measured on
#: the RTX 4060 Ti) instead of one per layer. Ticks come at most every LIVE3D_INTERVAL_S and are
#: spaced further so that their measured cost (uploads + the render) stays under FLOW_DUTY of
#: the GUI thread's time; a frame arriving between ticks is drawn by the next one (always the
#: newest). Measured on a real Draft run of the Ahmed body (1.9 M cells, smoke volumes of 3.8 M
#: voxels at ~4.5 per second): a tick costs ~50 ms (27 ms smoke upload, 22 ms render), so the 3D
#: view updates ~2.2 times a second and the event loop keeps a 16.6 ms median interval. One
#: render per layer instead (the first version) rendered 4.6 times a second, and through GIL and
#: GPU contention the solver itself fell to ~22 steps/s; with ticks it ran at 81 (the GPU
#: otherwise idle in both runs).
LIVE3D_INTERVAL_S = 0.25
#: Streamlines traced on the GUI thread (only when a run sends velocity without traced lines):
#: at most this often, and spaced by their own cost.
STREAM_INTERVAL_S = 1.0
FLOW_DUTY = 0.15
#: Slice axes in the order of the slice strip: (key, button text, tooltip)
SLICE_CHOICES = (("y", "Side · y", "Slice normal to y: the side view (x–z) through the centre plane"),
                 ("z", "Plan · z", "Slice normal to z: the plan view (x–y) at the body's mid-height"),
                 ("x", "Cross · x", ("Slice normal to x: the cross-flow view (y–z) behind the body, seen from "
                                     "behind")))
#: Explore slider ranges (half-degree steps): 2D pitch, 3D pitch (free / on a road), 3D yaw
ALPHA_RANGE_2D, ALPHA_RANGE_3D, ALPHA_RANGE_3D_ROAD, YAW_RANGE_3D = (-40, 50), (-20, 20), (-6, 6), (-30, 30)

__all__ = ["SimulatePage", "fmt_duration", "mean_error_text", "parse_polar_point", "parse_wing_message", "physics_summary",
           "plan_key", "wind_bodies"]


def mean_error_text(stat) -> str:
    """"1.653 ± 0.0011", "≈0 ± 0.0029" or "4.562 (steady)": a time average for the Messages line, rounded
    to its standard error by the Results table's rule (``kpi.value_error_texts``).

    The line a finished run logs is the first number most people read. It printed four significant
    figures whatever the error ("Cl −0.001534 ± 0.0029") while the Summary table beside it said "≈0",
    so the two disagreed about what the run had measured.
    """
    value, err = value_error_texts(stat.mean, stat.stderr)
    if err in ("—", ""):
        return value
    if err == STEADY_TEXT:
        return f"{value} ({STEADY_TEXT})"
    return f"{value} ± {err}"


def physics_summary(result) -> str:
    """"Nu 4.562 (steady), Q 0.1021 W/m, fan “Fan” thrust 1.000 N/m": what a run with placed physics
    measured, for the Messages line ("" without heat transfer or fans).

    It leads the line because a heated cavity or a room is about its Nusselt number, heat rate and
    fan thrust; their forces are NaN (a closed box has no free stream) and printed as "Cd —, Cl —".
    """
    per = "W" if isinstance(result, CFD3DResult) else "W/m"
    force = "N" if isinstance(result, CFD3DResult) else "N/m"
    parts = []
    nu = getattr(result, "nusselt", None)
    if nu is not None and math.isfinite(nu.mean):
        parts.append(f"Nu {mean_error_text(nu)}")
    q = getattr(result, "heat_rate", float("nan"))
    if isinstance(q, (int, float)) and math.isfinite(q):
        parts.append(f"Q {fmt_num(q, 4)} {per}")
    for fan in getattr(result, "fans_report", None) or []:
        thrust = fan.get("thrust") if isinstance(fan, dict) else None
        if isinstance(thrust, (int, float)) and math.isfinite(thrust):
            parts.append(f"fan “{fan.get('name') or 'Fan'}” thrust {fmt_num(thrust, 4)} {force}")
    return ", ".join(parts)


def wind_bodies(design, alpha_deg: float, *, ground: bool = False) -> list[np.ndarray]:
    """Design outlines rotated into the wind frame (by -alpha about moment_ref, DESIGN.md 3).

    With a road the wheels stay where they are, as the planner does: a pitched
    car rotates its body, not its contact patches. A 3D design has no outlines
    (its slice sections come from the plan): an empty list.
    """
    if getattr(design, "dimension", 2) == 3:
        return []
    a = -math.radians(alpha_deg)
    c, s = math.cos(a), math.sin(a)
    cx, cy = design.moment_ref
    out = []
    for b in design.bodies:
        p = np.asarray(b.points, dtype=float)
        if ground and getattr(b, "role", "") == "wheel":
            out.append(p.copy())
            continue
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        out.append(np.stack([cx + c * dx - s * dy, cy + s * dx + c * dy], axis=1))
    return out


def _mesh_digest(design) -> str:
    """A cheap fingerprint of a design's mesh arrays (``to_dict`` leaves them out).

    A transform in Geometry changes every vertex, so a strided sample of
    ~20 000 coordinates plus the shapes is enough to see it, at well under a
    millisecond for a 200 000-triangle car.
    """
    meshes = getattr(design, "meshes", None) or []
    if not meshes:
        return ""
    h = hashlib.sha1()
    for m in meshes:
        v = np.asarray(m.vertices)
        f = np.asarray(m.faces)
        h.update(repr((v.shape, f.shape)).encode())
        h.update(np.ascontiguousarray(v[:: max(1, len(v) // 20000)]).tobytes())
        h.update(np.ascontiguousarray(f[:: max(1, len(f) // 20000)]).tobytes())
    return h.hexdigest()


def plan_key(sim: Simulation, design) -> str:
    """Fingerprint of everything that decides a plan: the simulation's settings and the design.

    The cache used to be keyed by the simulation's id, so after the speed was
    changed in Setup (30 -> 5 m/s) the Run panel kept showing Re 2.05×10⁶ and
    the 30 m/s estimate while the solver really ran at 3.42×10⁵. Any edit now
    changes the key, and the old plan is simply never looked up again. A 3D
    design's mesh arrays enter through :func:`_mesh_digest`.
    """
    d = sim.to_dict()
    for k in _PLAN_IRRELEVANT:
        d.pop(k, None)
    payload = json.dumps([d, design.to_dict() if design is not None else None, _mesh_digest(design)],
                         sort_keys=True, default=str)
    return f"{sim.id}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()}"


def parse_polar_point(message: str) -> tuple[float, float, float, bool] | None:
    """(alpha, cl, cd, converged) from one progress message of a polar sweep, None for other messages."""
    m = _POLAR_POINT_RE.search(message or "")
    if m is None:
        return None
    try:
        alpha = float(m.group(1))
        if m.group(4):
            return alpha, float("nan"), float("nan"), False
        return alpha, float(m.group(2)), float(m.group(3)), True
    except (TypeError, ValueError):
        return None


def parse_wing_message(message: str):
    """What one progress message of a wing run says.

    ``("section", airfoil, re, alpha, cl, cd, converged)`` for a section-polar
    angle, ``("angle", alpha)`` for an angle of the wing solve itself, None for
    anything else (planning, start and end messages).
    """
    text = (message or "").strip()
    m = _WING_SECTION_RE.search(text)
    if m is not None:
        pt = parse_polar_point(m.group(3))
        if pt is None:
            return None
        try:
            re_ = float(m.group(2))
        except ValueError:
            return None
        return ("section", m.group(1).strip(), re_, *pt)
    m = _WING_ANGLE_RE.search(text)
    if m is not None:
        try:
            return ("angle", float(m.group(1)))
        except ValueError:
            return None
    return None


#: Live monitors of a closed domain (a room, a cavity), in display order: (readout key or pattern,
#: label, unit, history-plot label). Forces on the walls of a room mean nothing (Cd = Cl = 0 on a
#: ±1e-4 axis); what says whether a room has settled is its fans, its temperatures and its kinetic
#: energy. Keys are ``LiveFrame.physics`` readouts (SI; ``fan<i>_velocity`` per fan). The plot labels are
#: short (a 70 px panel clipped "KE (m²/s²)"): the rows above carry the units.
CLOSED_MONITORS = (
    ("fan_velocity", "Fan {n} disc velocity", "m/s", "V<sub>fan</sub>"),
    ("t_mean", "Mean temperature", "°C", "T<sub>mean</sub>"),
    ("t_max", "Max temperature", "°C", "T<sub>max</sub>"),
    ("t_min", "Min temperature", "°C", "T<sub>min</sub>"),
    ("ke", "Kinetic energy", "m²/s²", "KE"),
    ("nu", "Nusselt number", "", "Nu"),
)
#: At most this many history panels for a closed domain (the monitor column is 240 px tall).
CLOSED_HISTORY_PANELS = 3


def is_closed_sim(sim) -> bool:
    """A closed domain (DESIGN.md 10.6: room, cavity), not a tunnel."""
    return getattr(getattr(sim, "cfd", None), "domain_mode", "tunnel") == "closed"


def closed_box3(sim, domain=None) -> tuple[float, ...] | None:
    """The box a closed 3D domain is framed by: the plan's domain (wind frame), else Setup's box; None for
    a tunnel or when no box is known."""
    if sim is None or not is_closed_sim(sim):
        return None
    for box in (domain, getattr(getattr(sim, "cfd", None), "domain_box", None)):
        try:
            b = tuple(float(v) for v in box) if box is not None else ()
        except (TypeError, ValueError):
            continue
        if len(b) == 6 and all(math.isfinite(v) for v in b) and b[1] > b[0] and b[3] > b[2] and b[5] > b[4]:
            return b
    return None


def closed_monitor_items(physics: dict | None) -> list[tuple[str, str, str]]:
    """(key, label, value text) of the closed-domain monitors a frame's ``physics`` readouts carry."""
    out: list[tuple[str, str, str]] = []
    phys = physics or {}
    for key, lab, unit, _h in CLOSED_MONITORS:
        if key == "fan_velocity":
            fans = sorted((int(k[3:-9]), k) for k in phys if k.startswith("fan") and k.endswith("_velocity")
                          and k[3:-9].isdigit())
            for i, k in fans:
                v = phys.get(k)
                if v is not None and math.isfinite(float(v)):
                    out.append((k, lab.format(n=i + 1), f"{fmt_num(float(v), 3)} {unit}"))
            continue
        v = phys.get(key)
        if v is None or not math.isfinite(float(v)):
            continue
        out.append((key, lab, f"{fmt_num(float(v), 4)} {unit}".rstrip()))
    return out


def closed_history_keys(keys) -> tuple[list[str], dict[str, str]]:
    """The history panels of a closed domain: the first fan's velocity, then the temperatures and the
    kinetic energy, at most :data:`CLOSED_HISTORY_PANELS`; returns (keys, HTML axis labels)."""
    keys = list(keys)
    chosen: list[str] = []
    labels: dict[str, str] = {}
    for key, _lab, _unit, hl in CLOSED_MONITORS:
        if key == "fan_velocity":
            fan = next((k for k in sorted(keys) if k.startswith("fan") and k.endswith("_velocity")), None)
            if fan is not None:
                chosen.append(fan)
                labels[fan] = hl
        elif key in keys and key != "t_min":
            chosen.append(key)
            labels[key] = hl
    return chosen[:CLOSED_HISTORY_PANELS], labels


def stored_diagnostics(jobs, sim_id: str):
    """The diagnostics a saved result keeps (:func:`aerolume.app.main_window.diagnostics_log_for`: its
    history's series, else its end-of-run values), or None."""
    try:
        from ..main_window import diagnostics_log_for
    except ImportError:
        return None
    try:
        log_ = diagnostics_log_for(jobs, sim_id)
    except Exception:  # noqa: BLE001 - a damaged result is reported by Results, not here
        return None
    return log_ if log_ is not None and len(log_) else None


def _ld(cl: float, cd: float) -> float:
    return cl / cd if cd and math.isfinite(cd) and math.isfinite(cl) else float("nan")


def _value(text: str = "—", mono: bool = False) -> QLabel:
    lab = label(text, "mono" if mono else None)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def _plan_with_sections(plan_fn, design, sim, airfoils):
    """Plan a 3D run (worker thread) and cut the wind-frame meshes at the default slices.

    The cross-sections let the slice view show the body before the first
    frame; they are cheap next to the planning itself.
    """
    kw = {}
    if airfoils:
        from ..jobs import _airfoil_coords
        coords = _airfoil_coords(airfoils)
        if coords is not None:
            kw["airfoil_coords"] = coords
    plan = plan_fn(design, sim, **kw)
    meshes = list(getattr(plan, "bodies_wind3", []) or [])
    sections = {}
    for axis, pos in (getattr(plan, "slice_positions", {}) or {}).items():
        try:
            sections[(axis, round(float(pos), 9))] = mesh_sections(meshes, axis, float(pos))
        except Exception:  # noqa: BLE001 - an outline must never cost the plan
            sections[(axis, round(float(pos), 9))] = []
    return plan, sections


class SimulatePage(QWidget):
    """Run / watch / explore page (DESIGN.md 5.8). ``handle_action("run" | "live")``."""

    #: The Project panel selects the simulation; a selector here would compete with it.
    #: Standalone hosts (tests, embedding) can turn it on.
    SHOW_SELECTOR = False

    # logarithmic speed slider: equal travel per doubling, 0.25x .. 2x of the simulation's
    # speed, 100 steps per doubling so the simulation's own speed sits exactly on a step
    SPEED_STEPS_PER_DOUBLING = 100

    #: What the page needs of its JobManager (jobPlanned, plan_of, job_inputs and
    #: cached_result are used when present).
    JOBS_API = ("running_id", "queued_ids", "is_busy", "progress_of", "result", "submit", "cancel", "view",
                "jobQueued", "jobStarted", "jobProgress", "jobFrame", "jobFinished", "jobFailed",
                "jobCancelled", "queueChanged")

    def __init__(self, state, jobs: JobManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        require_api(jobs, self.JOBS_API, "The Simulate page")
        self.state = state
        self.jobs = jobs
        self.live = LiveWorker(self)
        self._live_sim_id = ""
        self._shown_sim_id = ""          # whose frames the view currently shows
        self._rows_done_sim = ""         # the Run rows describe this finished run (see _done_rows)
        self._plan_call = None
        self._plan_call_key = ""
        self._plans: OrderedDict[str, object] = OrderedDict()   # plan_key -> plan
        self._sections: dict[str, dict] = {}                     # plan_key -> {(axis, pos): sections}
        self._job_plans: dict[str, object] = {}                 # sim id -> the plan its run executes
        self._finished_id = ""
        self._base_speed = 30.0
        self._mode = "field"             # "field" (2D CFD, explore), "field3d", "polar" or "wing"
        self._polar_points: dict[float, tuple[float, float, bool]] = {}   # alpha -> (cl, cd, converged)
        self._polar_total = 0
        self._polar_alpha_range: tuple[float, float] | None = None   # the sweep's angles, for the axes
        self._wing_angles: set[float] = set()
        self._milestone: dict[str, int] = {}
        self._phase_seen: dict[str, str] = {}
        self._run_t_end: dict[str, float] = {}     # end time the runner reports (may exceed the plan)
        self._eta: float | None = None             # smoothed "Remaining" [s] of the running job
        self._eta_at = 0.0                         # monotonic time of the last ETA update
        self._range = LiveRangeTracker()           # colour range of live speed frames
        self.view3d = None                         # View3D of 3D runs, built on first 3D use
        self.flow: FlowScene | None = None         # its flow layers (smoke, streamlines, slice, rake)
        self._flow_at = {"tick": 0.0, "streamlines": 0.0}   # last live 3D tick / GUI-side trace (monotonic)
        self._tick_s = LIVE3D_INTERVAL_S           # current spacing of live 3D ticks (cost-paced)
        self._last_caption3d = ""                 # time and step of the newest live 3D frame
        self._slice_where = ""                    # " · y = 0.03 m": added to it while a slice is drawn
        self._result_on_3d = ""                   # "last" / "stopped": a stored result is in the 3D view
        self._live_volume = None                   # newest LiveFrame.volume (smoke, velocity)
        self._live_serial = None                   # its LiveFrame.volume_serial (a new volume is a new serial)
        self._live_lines = None                    # newest LiveFrame.streamlines3d (traced on the solver thread)
        self._last_frame3d = None                  # newest 3D frame (redrawn when the 3D view shows again)
        self._rakes: dict[str, RakeSpec] = {}      # sim id -> the rake the user set
        self._rake_sim = ""                        # whose rake the editor shows
        self._framed_room: tuple | None = None     # (sim id, box) a closed 3D domain was last framed for
        self._rake_box = None                      # the domain it was placed in (None: not known yet)
        self._attitude_timer = QTimer(self)        # explore: the 3D body follows the sliders
        self._attitude_timer.setSingleShot(True)
        self._attitude_timer.setInterval(120)
        self._attitude_timer.timeout.connect(self._explore_bodies_to_3d)
        self._slice_lock = False                   # programmatic slider moves
        self._flow_timer = QTimer(self)            # the next live 3D tick (see _frame_to_3d)
        self._flow_timer.setSingleShot(True)
        self._flow_timer.timeout.connect(self._deferred_tick)
        # live editing (DESIGN.md 10.4): the drawing, the tools, and explore's own copy of the simulation
        # (whose emitters, fans and heaters the user moves; promoted to a full run on request)
        self.sketch = ObstacleSketch()
        self._sketch_sim = ""                      # whose drawing it is
        self._sketch_posted = -1                   # sketch revision the session last received
        self.move_tool = MoveTool(on_move=self._on_placement_moved, enabled=False)
        self.sketch_tool = SketchTool(self.sketch, on_change=self._on_sketch_changed)
        self._live_sim: Simulation | None = None
        self._moved_lists: set[str] = set()        # placement lists moved in this explore session
        self._post_timer = QTimer(self)            # throttles live posts while a stroke or a drag goes on
        self._post_timer.setSingleShot(True)
        self._post_timer.setInterval(120)
        self._post_timer.timeout.connect(self._flush_live_edits)
        self._dirty_edits: set[str] = set()
        self._nu = float("nan")                    # explored kinematic viscosity [m²/s]
        self._re_lock_busy = False
        self._diag_bound = None                    # the DiagnosticsLog the panel shows
        self._field_default_for = ""               # the simulation the 2D field menu was defaulted for
        self._trace_wanted = False
        self._build()
        self._connect()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.empty = EmptyState()
        self.stack.addWidget(self.empty)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        self.stack.addWidget(split)

        # viewport: a field view with its toolbar, the polar plots or the wing run view
        self.vp_stack = QStackedWidget()
        vp = QWidget()
        vl = QVBoxLayout(vp)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        self.toolbar = ViewportToolbar()
        # 3D runs: 3D view, the slice as a 2D field view, or both side by side. In a strip of its own that
        # sits left of the toolbar, or above it when the viewport is too narrow for one row (the Live
        # workspace at 1280 px squeezed it to "Slice Spli" and the colour map to "Viridis (def...")
        self.view_mode = keep_whole(view_mode_control("3d"))
        self.view_strip = ToolStrip()
        self.view_strip.add_widget(label("View", "secondary"))
        self.view_strip.add_widget(self.view_mode)
        self._view_mode_widgets = [self.view_strip]
        self.top_bar = QWidget()
        self._top_box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self.top_bar)
        self._top_box.setContentsMargins(0, 0, 0, 0)
        self._top_box.setSpacing(0)
        self._top_box.addWidget(self.view_strip)
        self._top_box.addWidget(self.toolbar, 1)
        self.top_bar.setMinimumWidth(1)
        self.top_bar.installEventFilter(self)
        # optional selector for standalone hosts; parented to the page because the main window never adds
        # it to a layout, and a parentless combo outlived every page as a hidden top-level window
        self.sim_combo = QComboBox(self)
        self.sim_combo.setMinimumWidth(220)
        self.sim_combo.setToolTip("Simulation")
        self.sim_combo.setVisible(self.SHOW_SELECTOR)
        self.view = FieldView()
        self.view.setMinimumSize(420, 300)
        self.toolbar.finish(self.view)
        self.field_combo = self.toolbar.field_combo
        self.cmap_combo = self.toolbar.cmap_combo
        # in a narrow viewport the combos give way (their text elides) before the View control does
        self.field_combo.setMinimumWidth(96)
        self.cmap_combo.setMinimumWidth(120)
        self._cmap_tip = self.cmap_combo.toolTip()
        self.particles_btn = self.toolbar.particles_btn
        self.streamlines_btn = self.toolbar.streamlines_btn
        self.fit_btn = self.toolbar.fit_btn
        # the solver-inspection panel (Diagnostics, Solver trace) under the viewport, on demand
        self.toolbar.add_separator()
        self.diag_btn = self.toolbar.add_button("gauge", "Diagnostics", "Diagnostics: the solver's divergence, "
                                                "change per step, CFL number and residual, and the solver trace",
                                                checkable=True)
        self.diag_btn.setProperty("variant", "toggle")
        # a narrow viewport (the Live workspace at 1280 px) elided these to "Par…les" and "Str…nes": they give
        # up their text for an icon (the tooltip names them), as Diagnostics does
        from .. import theme as _theme
        compact = []
        for b, ic, text in ((self.particles_btn, "dot", "Particles"), (self.streamlines_btn, "wind", "Streamlines"),
                            (self.fit_btn, None, "Fit")):
            if b is None:
                continue
            if ic:
                b.setIcon(_theme.icon(ic, "text2", active="accent"))
                b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            compact.append((b, text))
        self.toolbar.compactable = [*getattr(self.toolbar, "compactable", []), *compact, (self.diag_btn, "Diagnostics")]
        vl.addWidget(self.top_bar)
        self._build_slice_strip()
        vl.addWidget(self.slice_strip)
        # live editing (2D explore): draw obstacles, move emitters / fans / heaters (DESIGN.md 10.4)
        self.draw_strip = DrawToolStrip()
        self.draw_strip.hide()
        vl.addWidget(self.draw_strip)
        self.view_split = QSplitter(Qt.Orientation.Horizontal)
        self.view_split.setHandleWidth(1)
        self.view_split.setChildrenCollapsible(False)
        self.view3d_host = QFrame()                 # the 3D view goes here on first 3D use
        self.view3d_host.setMinimumWidth(280)
        ol = QVBoxLayout(self.view3d_host)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.setSpacing(0)
        self.view3d_host.hide()
        self.view_split.addWidget(self.view3d_host)
        self.view_split.addWidget(self.view)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 1)
        self.view_split.installEventFilter(self)       # Split re-orients with the viewport's shape
        self.vp_split = QSplitter(Qt.Orientation.Vertical)
        self.vp_split.setHandleWidth(1)
        self.vp_split.setChildrenCollapsible(False)
        self.vp_split.addWidget(self.view_split)
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setDocumentMode(True)
        self.diag_panel = DiagnosticsPanel()
        self.trace_view = TraceViewer()
        self.bottom_tabs.addTab(self.diag_panel, "Diagnostics")
        self.bottom_tabs.addTab(self.trace_view, "Solver trace")
        self.bottom_tabs.setMinimumHeight(180)
        self.bottom_tabs.hide()
        self.vp_split.addWidget(self.bottom_tabs)
        self.vp_split.setStretchFactor(0, 1)
        self.vp_split.setStretchFactor(1, 0)
        vl.addWidget(self.vp_split, 1)
        self.vp_stack.addWidget(vp)

        pp = QWidget()
        pl = QVBoxLayout(pp)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(0)
        strip = ToolStrip()
        self.polar_caption = label("", "secondary")
        strip.add_widget(self.polar_caption)
        strip.add_stretch()
        pl.addWidget(strip)
        self.polar_view = PolarGrid()
        pl.addWidget(self.polar_view, 1)
        self.vp_stack.addWidget(pp)

        wp = QWidget()
        wl = QVBoxLayout(wp)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(0)
        wstrip = ToolStrip()
        self.wing_caption = label("", "secondary")
        wstrip.add_widget(self.wing_caption)
        wstrip.add_stretch()
        wl.addWidget(wstrip)
        self.wing_view = WingRunView()
        wl.addWidget(self.wing_view, 1)
        self.vp_stack.addWidget(wp)
        split.addWidget(self.vp_stack)

        # inspector
        side = QFrame()
        side.setProperty("panel", "true")
        side.setMinimumWidth(320)        # Run, Stop and Show results in one row at 1280 px
        side.setMaximumWidth(480)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        grid = self.grid = PropertyGrid(label_width=96)

        run = grid.add_group("Run")
        if self.SHOW_SELECTOR:
            run.add_row("Simulation", self.sim_combo)
        self.sim_title = ElidedLabel("")
        self.sim_title.setMaximumWidth(400)
        self._row_sim = run.add_row("Simulation", self.sim_title)
        self._row_sim.set_visible(not self.SHOW_SELECTOR)
        self.status_text = StatusLabel("Not run", "muted")
        run.add_row("Status", self.status_text)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        run.add_widget(self.progress)
        # the viewport follows the running job; when the panel names another simulation
        # this line says whose flow is on screen (the rows below describe the selected one)
        self.other_note = label("", "muted", wrap=True)
        self.other_note.hide()
        run.add_widget(self.other_note)
        self.phase_value = _value()
        self._row_phase = run.add_row("Phase", self.phase_value)
        self.time_value = _value()
        self._row_time = run.add_row("Time", self.time_value,
                                     tooltip="Simulated time in convective units (reference length / speed)")
        self.rate_value = _value()
        self._row_rate = run.add_row("Rate", self.rate_value, tooltip="Solver time steps per second")
        self.eta_value = _value()
        self._row_eta = run.add_row("Remaining", self.eta_value)
        self.grid_value = _value()
        self._row_grid = run.add_row("Grid", self.grid_value)
        self.sym_value = _value()
        self._row_sym = run.add_row("Symmetry", self.sym_value,
                                    tooltip="A mirror-symmetric body at zero yaw is simulated as one half with a "
                                            "symmetry plane at y = 0; forces are reported for the whole body")
        self._row_sym.set_visible(False)
        self.re_value = _value()
        self._row_re = run.add_row("Reynolds number", self.re_value)
        self.estimate_value = _value()
        self._row_est = run.add_row("Estimate", self.estimate_value, tooltip="Estimated wall-clock time")
        # planner warnings: one line (full text in the tooltip, all of it in Messages); a
        # paragraph here pushed the Monitors below the fold at 1280 x 800
        self.plan_note = WarningLine()
        self.plan_note.hide()
        run.add_widget(self.plan_note)
        self.plan_error = MessageLine("", "error")
        self.plan_error.hide()
        run.add_widget(self.plan_error)
        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 4, 0, 0)
        bl.setSpacing(4)
        self.run_btn = button("Run", "primary", icon="run", tooltip="Run the selected simulation (F5)")
        self.stop_btn = button("Stop", None, icon="stop",
                               tooltip="Stop the running simulation; the partial result is kept (Shift+F5)")
        self.results_btn = button("Show results", "link", tooltip="Open the Results step for this simulation")
        self.results_btn.hide()
        bl.addWidget(self.run_btn)
        bl.addWidget(self.stop_btn)
        bl.addStretch(1)
        bl.addWidget(self.results_btn)
        run.add_widget(btns)

        mon = grid.add_group("Monitors")
        self.monitor_note = label("Latest time step", "muted")
        mon.add_widget(self.monitor_note)
        self.coef_values = {k: _value(mono=True) for k in ("cd", "cs", "cl", "ld")}
        self._row_cd = mon.add_row("C<sub>d</sub>", self.coef_values["cd"], tooltip="Drag coefficient", key="Cd")
        self._row_cs = mon.add_row("C<sub>S</sub>", self.coef_values["cs"], tooltip="Side-force coefficient",
                                   key="CS")
        self._row_cs.set_visible(False)
        self._row_cl = mon.add_row("C<sub>l</sub>", self.coef_values["cl"], tooltip="Lift coefficient", key="Cl")
        self._row_ld = mon.add_row("L/D", self.coef_values["ld"], tooltip="Lift-to-drag ratio")
        # closed domains (a room, a cavity) monitor their fans, temperatures and kinetic energy instead
        from ..widgets.diagnostics import latest_values_list
        self.closed_values = latest_values_list(mono_values=True)
        self.closed_values.hide()
        mon.add_widget(self.closed_values)
        self._closed_mon = False
        self._closed_hist: dict[str, list[float]] = {"t": []}
        self._closed_keys: tuple[str, ...] = ()
        self.history = HistoryPlot(keys=("cd", "cl"), compact=True)
        self.history.t_min = MONITOR_T_MIN
        self.history.setFixedHeight(240)
        mon.add_widget(self.history)
        self.polar_monitor = HistoryPlot(keys=("cd", "cl"), xlabel="α (°)", compact=True)
        self.polar_monitor.skip = 0.0
        self.polar_monitor.setFixedHeight(240)
        self.polar_monitor.hide()
        mon.add_widget(self.polar_monitor)

        self.rake_group = grid.add_group("Seed rake", key="rake")
        self.rake_editor = RakeEditor(self.rake_group, parent=self, show_kind=True)
        self.rake_group.setVisible(False)

        ex = grid.add_group("Explore", expanded=False)
        self.live_btn = button("Start explore", None, icon="play",
                               tooltip="Real-time preview on a coarser grid; angle and speed act while it runs")
        ex.add_widget(self.live_btn)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(*ALPHA_RANGE_2D)     # half-degree steps: -20 .. +25 deg
        self.alpha_value = _value(mono=True)
        self._row_alpha = ex.add_row("Angle of attack", self._slider_row(self.alpha_slider, self.alpha_value))
        self.yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self.yaw_slider.setRange(*YAW_RANGE_3D)
        self.yaw_value = _value(mono=True)
        self._row_yaw = ex.add_row("Yaw angle", self._slider_row(self.yaw_slider, self.yaw_value),
                                   tooltip="Sideslip: wind from the side, ±15°")
        self._row_yaw.set_visible(False)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 3 * self.SPEED_STEPS_PER_DOUBLING)
        self.speed_value = _value(mono=True)
        self._row_speed = ex.add_row("Speed", self._slider_row(self.speed_slider, self.speed_value),
                                     tooltip="0.25 to 2 times the simulation's speed")
        self._build_reynolds_lock(ex)
        self._build_playback(ex)
        self.live_error = MessageLine("", "error")
        self.live_error.hide()
        ex.add_widget(self.live_error)
        self.promote_btn = button("Run with these settings", None,
                                  tooltip="Queue a full simulation at the explored angle, speed, viscosity and "
                                          "positions of the emitters, fans and heaters")
        ex.add_widget(self.promote_btn)

        self.queue_group = grid.add_group("Queue")
        self.queue_list = QListWidget()
        self.queue_list.setProperty("flat", "true")
        self.queue_list.setFixedHeight(88)
        self.queue_group.add_widget(self.queue_list)
        self.dequeue_btn = button("Remove from queue", None)
        self.queue_group.add_widget(self.dequeue_btn)
        self.queue_group.hide()
        grid.add_stretch()
        self._side_scroll = grid.scrollable()
        sl.addWidget(self._side_scroll)
        # the monitor plot takes the height left below the Run rows (between MONITOR_H_MIN and
        # MONITOR_H_MAX), so at 1280 x 800 both C_D and C_L show without scrolling the panel
        self._side_scroll.viewport().installEventFilter(self)
        grid.installEventFilter(self)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1100, 340])

    def _build_slice_strip(self) -> None:
        """The 3D strip: what the 3D view shows (smoke, streamlines, slice plane, surface Cp), then the
        slice: normal (side / plan / cross-flow) and position in metres.

        Two tool strips side by side, stacked in two rows when the viewport is
        too narrow for both (1280 px with the Run panel and a dock open): the
        segmented normal and the layer buttons used to be squeezed until their
        labels read "S…e" and "iide · y".
        """
        box = self.slice_strip = QWidget()
        lay = self._slice_box = QBoxLayout(QBoxLayout.Direction.LeftToRight, box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        layers = self._layer_strip = ToolStrip()
        self.flow_toggles = FlowToggles(("smoke", "streamlines", "slice", "heat", "particles", "surface"),
                                        shown=dict(DEFAULT_SHOWN_LIVE), parent=self)
        self.flow_toggles.add_to(layers)
        # layers the shown simulation cannot have (no heat transfer, no particle emitters) have no button:
        # a greyed "Thermal plume" on every isothermal run would be noise (see _sync_physics3d)
        self._flow_absent: set[str] = {"heat", "particles"}
        self._devices_key = None
        self._heat_live_rng: tuple[float, float] | None = None
        self._flow_sep = layers.add_separator()
        # stacked in two rows the strip spans the viewport: the toggles keep their natural width,
        # left-aligned (without this they stretched evenly across 900 px and read like tabs)
        layers.add_stretch()
        s = self._slice_row = ToolStrip()
        s.add_widget(label("Slice", "secondary"))
        self.slice_axis = SegmentedControl([(k, t) for k, t, _tip in SLICE_CHOICES], "y")
        self.slice_axis.setToolTip("\n".join(tip for _k, _t, tip in SLICE_CHOICES))
        keep_whole(self.slice_axis)
        s.add_widget(self.slice_axis)
        s.row.addSpacing(8)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 1000)
        self.slice_slider.setMinimumWidth(80)
        self.slice_slider.setMaximumWidth(240)
        self.slice_slider.setToolTip("Where the slice cuts the flow, in metres along its normal")
        s.add_widget(self.slice_slider)
        # the distance only ("0.027 m"): the normal is the control beside it; the axis is in the tooltip
        self.slice_value = label("", "secondary")
        self.slice_value.setMinimumWidth(56)
        s.add_widget(self.slice_value)
        self.slice_reset = s.add_button("refresh", "", "Back to the default slice through the body "
                                                       "(centre plane, mid-height, or just behind it)")
        s.add_stretch()
        lay.addWidget(layers)
        lay.addWidget(s, 1)
        box.setMinimumWidth(1)
        box.installEventFilter(self)
        box.hide()

    def _reflow_slice_strip(self) -> None:
        """One row when the layer buttons and the slice controls fit side by side, else two rows.

        A group that is hidden takes no room: the slice controls only show when
        a slice is on screen (:meth:`_slice_shown`), so a live run with smoke
        alone has one 30 px strip under the toolbar instead of two.
        """
        show_a = self._layer_strip.isVisibleTo(self.slice_strip)
        show_b = self._slice_row.isVisibleTo(self.slice_strip)
        room = self.top_bar.width()          # the viewport's width (the strip itself is as wide as its minimum)
        if show_a and room > 1:
            # what does not fit the viewport goes into the strip's "⋯" menu (never clipped to "Sur")
            m = self._layer_strip.layout().contentsMargins() if self._layer_strip.layout() is not None else None
            pad = (m.left() + m.right()) if m is not None else 16
            self.flow_toggles.overflow_to(room - pad - 8)
        a = self._layer_strip.minimumSizeHint().width() if show_a else 0
        b = self._slice_row.minimumSizeHint().width() if show_b else 0
        # an explicit minimum (the wider group, i.e. the two-row width) takes precedence over the
        # one-row layout's, so the strip can become narrower than one row and reflow
        self.slice_strip.setMinimumWidth(max(a, b, 1))
        one_row = self.slice_strip.width() >= a + b
        want = QBoxLayout.Direction.LeftToRight if one_row else QBoxLayout.Direction.TopToBottom
        if self._slice_box.direction() != want:
            self._slice_box.setDirection(want)
        # the divider between the two groups only means something when they share the row
        self._flow_sep.setVisible(one_row and show_b and any(b.isVisibleTo(self._layer_strip)
                                                             for b in self.flow_toggles.buttons.values()))

    def _reflow_top_bar(self) -> None:
        """The View control beside the toolbar when both fit, else above it (two rows)."""
        if not self.view_strip.isVisibleTo(self.top_bar):
            want = QBoxLayout.Direction.LeftToRight
        else:
            need = self.view_strip.minimumSizeHint().width() + self.toolbar.minimumSizeHint().width()
            want = (QBoxLayout.Direction.LeftToRight if self.top_bar.width() >= need
                    else QBoxLayout.Direction.TopToBottom)
        if self._top_box.direction() != want:
            self._top_box.setDirection(want)
        self.top_bar.setMinimumWidth(max(self.view_strip.minimumSizeHint().width()
                                         if self.view_strip.isVisibleTo(self.top_bar) else 1, 1))
        if getattr(self, "slice_strip", None) is not None and self.slice_strip.isVisibleTo(self):
            self._reflow_slice_strip()       # the layer toggles' overflow follows the viewport's width

    def eventFilter(self, obj, event) -> bool:
        if obj is getattr(self, "top_bar", None) and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._reflow_top_bar()
        if obj is getattr(self, "slice_strip", None) and event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._reflow_slice_strip()
        elif obj is getattr(self, "view_split", None) and event.type() == QEvent.Type.Resize:
            self._orient_split()
        elif (event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest)
              and getattr(self, "_side_scroll", None) is not None
              and (obj is self._side_scroll.viewport() or obj is self.grid)):
            QTimer.singleShot(0, self, self._fit_monitor_plots)      # once the rows above have their places
        return super().eventFilter(obj, event)

    #: Height range [px] of the monitor plot (C_D over C_L): it fills what the Run panel leaves.
    MONITOR_H_MIN, MONITOR_H_MAX = 160, 240

    def _fit_monitor_plots(self) -> None:
        """Size the monitor plot to the room left under the rows above it in the panel's viewport.

        At 1280 x 800 the fixed 240 px plot ran off the bottom of the panel and
        cut C_L in half. Below MONITOR_H_MIN the panel scrolls instead (two
        stacked plots any shorter cannot be read).
        """
        sc = getattr(self, "_side_scroll", None)
        if sc is None:
            return
        vp_h = sc.viewport().height()
        for plot in (self.history, self.polar_monitor):
            if not plot.isVisibleTo(self.grid):
                continue
            top = plot.mapTo(self.grid, plot.rect().topLeft()).y()
            room = vp_h - top - 8
            h = int(min(self.MONITOR_H_MAX, max(self.MONITOR_H_MIN, room)))
            if plot.height() != h or plot.minimumHeight() != h:
                plot.setFixedHeight(h)

    def _set_slice_value(self, axis: str, pos: float | None) -> None:
        """The slice readout: the plane's distance along its normal, the full plane in the tooltip."""
        if pos is None or not math.isfinite(pos):
            self.slice_value.setText("—")
            self.slice_value.setToolTip("")
            return
        self.slice_value.setText(f"{sig(pos, 3)} m".replace("-", MINUS))
        self.slice_value.setToolTip(f"Slice plane at {axis} = {sig(pos, 4)} m".replace("-", MINUS))

    @staticmethod
    def _slider_row(slider: QSlider, value: QLabel) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(slider, 1)
        value.setMinimumWidth(64)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(value)
        return w

    #: Which of speed U, kinematic viscosity ν and Reynolds number Re follows the other two (AeroJAX's
    #: Reynolds lock: two are held, the third is derived): (key, combo text, tooltip).
    RE_FOLLOWS = (("re", "Reynolds number", "Speed and viscosity are set; the Reynolds number follows"),
                  ("nu", "Viscosity", "Speed and Reynolds number are held; the viscosity follows"),
                  ("speed", "Speed", "Viscosity and Reynolds number are held; the speed follows"))

    def _build_reynolds_lock(self, ex) -> None:
        """Rows of the Reynolds lock: what follows, the kinematic viscosity and the Reynolds number.

        Speed has its slider above; viscosity and Reynolds number are typed
        (scientific notation accepted: 1.5e-5). The derived one is read-only and
        shown in the text colour of a value, not of an editor.
        """
        self.re_follow = QComboBox()
        for key, text, tip in self.RE_FOLLOWS:
            self.re_follow.addItem(text, key)
            self.re_follow.setItemData(self.re_follow.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
        self.re_follow.setToolTip("Two of speed, viscosity and Reynolds number are held; this one follows them")
        self._row_follow = ex.add_row("Follows", self.re_follow)
        val = QDoubleValidator(1e-12, 1e12, 6)
        val.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.nu_edit = QLineEdit()
        self.nu_edit.setValidator(val)
        self._row_nu = ex.add_row("Viscosity ν", self.nu_edit, unit="m²/s",
                                  tooltip="Kinematic viscosity μ/ρ of the explored fluid")
        val_re = QDoubleValidator(1e-6, 1e12, 3)
        val_re.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.re_edit = QLineEdit()
        self.re_edit.setValidator(val_re)
        self._row_re_lock = ex.add_row("Reynolds number", self.re_edit,
                                       tooltip="Re = U L / ν with the design's reference length L")
        self.re_note = label("", "muted", wrap=True)
        self.re_note.hide()
        ex.add_widget(self.re_note)

    def _build_playback(self, ex) -> None:
        """Pause / step / reset, the target frame rate and the steps per frame of explore."""
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self.pause_btn = button("Pause", None, icon="pause", tooltip="Pause the explore solver (the view stays)")
        self.pause_btn.setCheckable(True)
        self.step_btn = button("Step", None, tooltip="Advance the paused flow by the steps on the right and redraw")
        self.step_btn.setIcon(tool_icon("step"))
        self.step_n = QSpinBox()
        self.step_n.setRange(1, 1000)
        self.step_n.setValue(1)
        self.step_n.setToolTip("Solver steps per Step click")
        self.reset_btn = button("", None, icon="refresh", tooltip="Restart the explored flow from the free stream")
        rl.addWidget(self.pause_btn)
        rl.addWidget(self.step_btn)
        rl.addWidget(self.step_n)
        rl.addStretch(1)
        rl.addWidget(self.reset_btn)
        ex.add_widget(row)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")
        self._row_fps = ex.add_row("Target rate", self.fps_spin,
                                   tooltip="Frames per second the explore loop aims for; the solver fills the time "
                                           "between frames with steps")
        self.spf_spin = QSpinBox()
        self.spf_spin.setRange(0, 2000)
        self.spf_spin.setSpecialValueText("Automatic")
        self._row_spf = ex.add_row("Steps per frame", self.spf_spin,
                                   tooltip="Fixed solver steps between frames (frame skip); Automatic fits as many as "
                                           "the target rate allows")
        self._sync_live_controls()

    def _connect(self) -> None:
        st, jb = self.state, self.jobs
        st.projectChanged.connect(self.refresh)
        st.simulationsChanged.connect(self._fill_sim_combo)
        st.simulationsChanged.connect(self._update_buttons)
        st.simulationsChanged.connect(self._on_setup_changed)
        st.designsChanged.connect(self._on_setup_changed)
        st.currentSimulationChanged.connect(lambda *_: self.refresh())
        self.sim_combo.activated.connect(self._on_sim_chosen)
        jb.jobStarted.connect(self._on_job_started)
        if hasattr(jb, "jobPlanned"):
            jb.jobPlanned.connect(self._on_job_planned)
        jb.jobProgress.connect(self._on_job_progress)
        jb.jobFrame.connect(self._on_job_frame)
        jb.jobFinished.connect(self._on_job_finished)
        jb.jobFailed.connect(self._on_job_failed)
        jb.jobCancelled.connect(self._on_job_cancelled)
        jb.jobQueued.connect(lambda *_: self._refresh_queue())
        jb.queueChanged.connect(self._refresh_queue)
        jb.queueChanged.connect(self._update_buttons)
        self.toolbar.changed.connect(self._on_view_changed)
        self.run_btn.clicked.connect(self.run_current)
        self.stop_btn.clicked.connect(self.stop)
        self.results_btn.clicked.connect(self._open_results)
        self.live_btn.clicked.connect(self.toggle_live)
        self.alpha_slider.valueChanged.connect(self._on_alpha)
        self.yaw_slider.valueChanged.connect(self._on_yaw)
        self.speed_slider.valueChanged.connect(self._on_speed)
        self.promote_btn.clicked.connect(self.promote_live_settings)
        self.dequeue_btn.clicked.connect(self._dequeue_selected)
        self.queue_list.itemSelectionChanged.connect(self._update_buttons)
        self.live.frameReady.connect(self._on_live_frame)
        self.live.started.connect(self._on_live_started)
        self.live.failed.connect(self._on_live_failed)
        self.live.stopped.connect(self._on_live_stopped)
        self.slice_axis.changed.connect(self._on_slice_axis)
        self.slice_slider.valueChanged.connect(self._on_slice_pos)
        self.slice_reset.clicked.connect(lambda *_: self._on_slice_axis(self.slice_axis.current()))
        self.view_mode.changed.connect(self._on_view_mode)
        self.flow_toggles.changed.connect(self._on_flow_toggle)
        self.rake_editor.changed.connect(self._on_rake_edited)
        # live editing, playback, the Reynolds lock and the solver-inspection panel
        self.draw_strip.toolChanged.connect(self._on_draw_tool)
        self.draw_strip.sizeChanged.connect(self._on_brush_size)
        self.draw_strip.undoRequested.connect(lambda: self._edit_sketch("undo"))
        self.draw_strip.redoRequested.connect(lambda: self._edit_sketch("redo"))
        self.draw_strip.clearRequested.connect(lambda: self._edit_sketch("clear"))
        self.pause_btn.toggled.connect(self._on_pause)
        self.step_btn.clicked.connect(self._on_step)
        self.reset_btn.clicked.connect(self._on_reset)
        self.fps_spin.valueChanged.connect(self._on_fps)
        self.spf_spin.valueChanged.connect(self._on_spf)
        self.re_follow.currentIndexChanged.connect(self._on_follow_changed)
        self.nu_edit.editingFinished.connect(self._on_nu_edited)
        self.re_edit.editingFinished.connect(self._on_re_edited)
        self.diag_btn.toggled.connect(self._on_diag_toggled)
        self.trace_view.captureRequested.connect(self._on_capture_trace)
        self.live.traceReady.connect(self._on_trace)
        if hasattr(jb, "jobTrace"):
            jb.jobTrace.connect(self._on_job_trace)
        self.view.set_tool(self.move_tool)

    # ------------------------------------------------------------- page API
    def on_shown(self) -> None:
        m = self._dock_manager()
        if m is not None:
            self._wire_dock_switch(m)
        self.refresh()

    def handle_action(self, action: str) -> None:
        """"run" queues the current simulation; "live" starts explore mode."""
        if action == "run":
            self.run_current()
        elif action == "live" and not self.live.is_running():
            self.grid.group("Explore").set_expanded(True)
            self.toggle_live()

    def shutdown(self) -> None:
        """Stop explore mode and wait for it (window close; frees the GPU). The JobManager
        is shut down by its owner."""
        stop = getattr(self.live, "shutdown", None) or self.live.stop
        stop()
        self._flow_timer.stop()
        if self.view3d is not None:
            try:
                self.view3d.scene.close()      # its OpenGL context
            except Exception:
                logging.getLogger(__name__).debug("closing the 3D scene failed", exc_info=True)

    # ------------------------------------------------------------- logging
    def _log(self, text: str, level: str = "info") -> None:
        log = getattr(self.state, "log", None)
        if callable(log):
            log(text, level, LOG_SOURCE)

    # ------------------------------------------------------------- refresh
    def _design_of(self, sim: Simulation | None):
        p = self.state.project
        return p.designs.get(sim.design_id) if p is not None and sim is not None else None

    def refresh(self) -> None:
        """Rebuild everything that depends on the project / current simulation."""
        self._fill_sim_combo()
        project, sim = self.state.project, self.state.simulation
        if project is None:
            self.empty.set_content("No project open", "", "Open or create a project",
                                   lambda: self.state.navigate("start"))
            self.stack.setCurrentIndex(0)
            return
        if sim is None:
            has_designs = bool(project.designs)
            self.empty.set_content(
                "No simulation selected",
                "Select a simulation in the Project panel." if project.simulations else "",
                "New simulation…" if has_designs else "Open Geometry",
                lambda: self.state.navigate("setup:new" if has_designs else "geometry"))
            self.stack.setCurrentIndex(0)
            return
        self.stack.setCurrentIndex(1)
        design = project.designs.get(sim.design_id)
        self._sync_sliders(sim)
        self._sync_particle_toggle(sim)
        self._update_buttons()
        self._refresh_queue()
        self._request_plan(sim, design)
        busy_id = self.jobs.running_id()
        if (busy_id == sim.id and not self.live.is_running()
                and (self._shown_sim_id != sim.id or self._mode != self._mode_for(sim))):
            # the selected simulation is running but this page has not shown it start (the page was
            # built mid-run, e.g. a script or the assistant started it before Simulate was opened)
            self._adopt_running(sim.id)
            return
        self._update_other_note()
        if not self.live.is_running() and not busy_id and self._diagnostics_shown():
            self._bind_diagnostics()         # another simulation: its own stored diagnostics, not explore's
        if self.live.is_running() or (busy_id and busy_id == self._shown_sim_id):
            # keep showing the live / running flow; the Run rows describe the selected one
            if not self.live.is_running():
                if busy_id == sim.id:
                    self._restore_running_rows(sim.id)
                else:
                    self._blank_run_rows(sim)
            return
        if busy_id == sim.id or (self._shown_sim_id == sim.id and self.view.has_image()
                                 and self._mode == self._mode_for(sim)):
            return  # running, or the last frame of this simulation is still on screen
        self._show_preview(sim, design)

    def _sync_particle_toggle(self, sim) -> None:
        """A newly selected simulation: Particles starts on when it has particle emitters (their tracers
        are what it was set up to show, as Setup's smoke choice sets Smoke in 3D), off otherwise; the
        tooltip says which tracers the toggle shows. Later clicks are the user's."""
        if self.particles_btn is None or sim is None or sim.id == getattr(self, "_particles_for", None):
            return
        self._particles_for = sim.id
        emits = any(getattr(e, "species", "") == "particles" and getattr(e, "enabled", True)
                    for e in (getattr(sim, "emitters", None) or []))
        self.particles_btn.setToolTip("Tracer particles from the particle emitters; their trails are streaklines"
                                      if emits else "Tracer particles carried by the flow from the inlet")
        if self.particles_btn.isChecked() != emits:
            self.particles_btn.setChecked(emits)          # through the toolbar's changed -> _on_view_changed

    def _mine(self, sim_id: str) -> bool:
        """Whether the Run panel describes ``sim_id`` (it names the selected simulation)."""
        return bool(sim_id) and sim_id == self.state.simulation_id

    def _update_other_note(self) -> None:
        """Say whose flow the viewport shows when it is not the selected simulation's."""
        sim = self.state.simulation
        busy_id = self.jobs.running_id()
        other = (sim is not None and not self.live.is_running() and bool(busy_id) and busy_id != sim.id
                 and busy_id == self._shown_sim_id)
        if other:
            self.other_note.setText(f"The view shows the running simulation, “{self._sim_name(busy_id)}”.")
        self.other_note.setVisible(other)

    def _blank_run_rows(self, sim: Simulation) -> None:
        """Run rows of a selected simulation that is not the running one (the view shows that one)."""
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 if sim.status == "done" else 0)
        for w in (self.phase_value, self.rate_value, self.eta_value):
            w.setText("—")
        queued = self.jobs.queued_ids()
        if sim.id in queued:
            pos = queued.index(sim.id) + 1
            self.time_value.setText(f"Waiting in the queue ({pos} of {len(queued)})")
        else:
            self.time_value.setText("—")

    def _restore_running_rows(self, sim_id: str) -> None:
        """Back on the running simulation: its progress, until the next update refreshes the rest."""
        frac, message, _elapsed = self.jobs.progress_of(sim_id)
        if message == "Stopping…":
            self.status_text.set_status("Stopping…", "warn")
        else:
            self.status_text.set_status(f"Running {100 * frac:.0f} %", "accent", message)
        if not (self._mode == "polar" and not self._polar_points and frac <= 0):
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(1000 * max(0.0, min(1.0, frac))))

    def _on_setup_changed(self, *_) -> None:
        """A simulation or design was edited (in Setup, the Project panel ...): re-plan if it is ours.

        Cheap when nothing that matters changed: the plan cache is keyed by a
        fingerprint of the settings, so a status change finds the same plan.
        A change of analysis (Navier-Stokes <-> polar <-> wing, 2D <-> 3D)
        rebuilds the page.
        """
        sim = self.state.simulation
        if sim is None or self.state.project is None or self.stack.currentIndex() != 1:
            return
        if self._mode_for(sim) != self._mode and not self.jobs.running_id() and not self.live.is_running():
            self.refresh()
            return
        self._request_plan(sim, self._design_of(sim))

    def _mode_for(self, sim: Simulation | None) -> str:
        """"polar", "wing", "field3d" (3D CFD) or "field" (2D CFD) for a simulation."""
        if sim is None:
            return "field"
        if sim.analysis in ("polar", "wing"):
            return sim.analysis
        design = self._design_of(sim) or self._job_inputs(sim.id)[0]
        if design is None:
            plan = self._run_plan(sim.id)
            return "field3d" if getattr(plan, "dimension", 2) == 3 else "field"
        return "field3d" if getattr(design, "dimension", 2) == 3 else "field"

    def _is3d(self) -> bool:
        return self._mode == "field3d"

    def _set_mode(self, mode: str, design=None, sim: Simulation | None = None) -> None:
        """Field viewport + time monitors, or polar plots / wing run + angle monitors."""
        self._mode = mode
        sweep = mode in ("polar", "wing")
        three_d = mode == "field3d"
        self.view.keep_body = three_d
        self.vp_stack.setCurrentIndex({"polar": 1, "wing": 2}.get(mode, 0))
        self.history.setVisible(not sweep)
        self.polar_monitor.setVisible(sweep)
        for row in (self._row_phase, self._row_rate):
            row.set_visible(not sweep)
        self._row_phase.set_visible(not sweep or mode == "wing")
        self._row_time.set_label("Progress" if sweep else "Time")
        self._row_time.editor.setToolTip("Angles of the sweep computed so far" if sweep else
                                         "Simulated time in convective units (reference length / speed)")
        self._row_ld.set_visible(is_lifting(design))
        self._row_sym.set_visible(three_d)
        self.slice_strip.setVisible(three_d)
        self.toolbar.set_fields(view_fields_for(sim, three_d=three_d, diagnostics=self._solver_has_divergence()),
                                three_d=three_d)
        if mode != "field":
            self.view.set_placements([])
            self.move_tool.set_placements([])
            self.view.set_sketch(None)
        for w in self._view_mode_widgets:
            w.setVisible(three_d)
        self._reflow_top_bar()
        if not three_d and sim is not None and sim.id != self._field_default_for:
            # the user placed smoke emitters to see smoke: explore and the run open on it (multi-colour
            # composite with several channels), once per simulation; later choices are the user's
            self._field_default_for = sim.id
            from ..widgets.view3d_panel import smoke_channel_labels
            self.view.channel_labels = smoke_channel_labels(sim)
            if any(getattr(e, "species", "dye") == "dye" and getattr(e, "enabled", True)
                   for e in getattr(sim, "emitters", None) or []):
                i = self.field_combo.findData("dye")
                if i >= 0 and self.field_combo.currentIndex() != i:
                    self.field_combo.setCurrentIndex(i)
        if self.streamlines_btn is not None:      # 3D: the strip's Streamlines toggle covers both views
            self.streamlines_btn.setVisible(not three_d)
        self.rake_group.setVisible(three_d)
        if three_d:
            self._ensure_view3d()
        self._apply_view_mode()
        # coefficient symbols: lower case for sections (2D), capitals for 3D bodies and wings
        upper = three_d or mode == "wing"
        self._row_cd.set_label("C<sub>D</sub>" if upper else "C<sub>d</sub>")
        self._row_cl.set_label("C<sub>L</sub>" if upper else "C<sub>l</sub>")
        yawed = three_d and sim is not None and abs(float(getattr(sim.conditions, "yaw_deg", 0.0) or 0.0)) > 1e-9
        self._row_cs.set_visible(yawed)
        labels = HistoryPlot.LABELS_3D if upper else HistoryPlot.LABELS
        self._closed_mon = not sweep and sim is not None and is_closed_sim(sim)
        if not self._closed_mon:
            self.history.set_keys(("cd", "cs", "cl") if yawed else ("cd", "cl"), labels)
        self._apply_monitor_kind(sim)
        self.polar_monitor.set_keys(("cd", "cl"), labels)
        self.grid.group("Explore").setVisible(mode != "wing")
        self._row_yaw.set_visible(three_d)
        road = sim is not None and ground_of(sim) is not None
        lo, hi = (ALPHA_RANGE_3D_ROAD if road else ALPHA_RANGE_3D) if three_d else ALPHA_RANGE_2D
        if (self.alpha_slider.minimum(), self.alpha_slider.maximum()) != (lo, hi):
            self.alpha_slider.blockSignals(True)
            self.alpha_slider.setRange(lo, hi)
            self.alpha_slider.blockSignals(False)
            self._label_sliders()

    def _apply_monitor_kind(self, sim) -> None:
        """Force rows for a body in a stream; fan, temperature and energy rows for a closed domain.

        The rake group says what its rake does: with smoke emitters set up it
        only seeds streamlines (the emitters release the smoke), so it is named
        "Streamline seeds" as on the Results page.
        """
        closed = self._closed_mon
        for row in (self._row_cd, self._row_cl):
            row.set_visible(not closed)
        if closed:
            self._row_cs.set_visible(False)
            self._row_ld.set_visible(False)
        self.closed_values.setVisible(closed)
        emitters = [e for e in (getattr(sim, "emitters", None) or []) if getattr(e, "enabled", True)
                    and getattr(e, "species", "dye") == "dye"] if sim is not None else []
        header = getattr(self.rake_group, "header", None)
        if header is not None and hasattr(header, "setText"):
            header.setText("Streamline seeds" if emitters else "Seed rake")

    def _closed_monitor_frame(self, frame) -> None:
        """A closed domain's monitors from one frame: the readouts as rows, their series in the plot."""
        phys = getattr(frame, "physics", None) or {}
        items = closed_monitor_items(phys)
        if not items:
            self.closed_values.update_items([("Monitors", "waiting for the first readout")])
            self.history.set_history(None)
            return
        self.closed_values.update_items([(lab, text) for _k, lab, text in items])
        t = getattr(frame, "t", float("nan"))
        h = self._closed_hist
        if math.isfinite(t) and (not h["t"] or t > h["t"][-1]):
            n = len(h["t"])
            h["t"].append(float(t))
            for k, v in phys.items():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                col = h.setdefault(k, [float("nan")] * n)
                col.append(f)
            for k, col in h.items():
                if len(col) < n + 1:
                    col.append(float("nan"))
            if n + 1 > 4000:                 # a long explore: keep the newest 4000 samples
                for k in h:
                    del h[k][:n + 1 - 4000]
        keys, labels = closed_history_keys([k for k, _lab, _t in items])
        if tuple(keys) != self._closed_keys:
            self._closed_keys = tuple(keys)
            if keys:
                self.history.set_keys(tuple(keys), labels)
        if keys:
            self.history.set_history({k: np.asarray(v, dtype=float) for k, v in h.items()})

    def _closed_monitor_result(self, res) -> None:
        """A closed domain's stored run: the fan / Nusselt / energy series its history kept."""
        hist = getattr(res, "history", None) or {}
        keys, labels = closed_history_keys([k for k in hist if k != "t"])
        last = {k: (np.asarray(v, dtype=float)[-1] if len(v) else float("nan")) for k, v in hist.items() if k != "t"}
        items = closed_monitor_items(last)
        self.closed_values.update_items([(lab, text) for _k, lab, text in items] or
                                        [("Monitors", "this run stored no fan or heat series")])
        self._closed_keys = tuple(keys)
        if keys:
            self.history.set_keys(tuple(keys), labels)
            self.history.set_history(hist, (res.t_avg_start, res.t_end) if math.isfinite(res.t_avg_start) else None)
        else:
            self.history.set_history(None)

    @staticmethod
    def _solver_has_divergence() -> bool:
        """Whether the live renderer can show the divergence field (it lists it in its view fields)."""
        live = lazy_import("aerolume.simulation.live")
        return "divergence" in tuple(getattr(live, "VIEW_FIELDS", ()) or ())

    def _set_placeholder(self, title: str, subtitle: str = "") -> None:
        """A message in the slice view and, for 3D runs, the same words as the 3D view's caption."""
        self.view.set_placeholder(title, subtitle)
        if self.flow is not None and self._is3d():
            short = subtitle if subtitle and len(subtitle) <= 72 else ""
            self.flow.set_caption(f"{title} · {short}" if short else title)

    def _reset_run_rows(self) -> None:
        self._rows_done_sim = ""
        for w in (self.phase_value, self.time_value, self.rate_value, self.eta_value):
            w.setText("—")

    def _clear_monitors(self, note: str = "Latest time step") -> None:
        self.monitor_note.setText(note)
        for k in ("cd", "cs", "cl", "ld"):
            self.coef_values[k].setText("—")
        self._closed_hist = {"t": []}
        self.closed_values.update_items([])
        self.history.set_history(None)
        self.polar_monitor.set_history(None)

    def _frame_closed(self, sim: Simulation | None) -> None:
        """Frame a closed box whole (letterboxed), as the Results page does; a tunnel keeps the bodies'
        near field.

        The default framing fills the view with the bodies' near field and crops its longer side. In a
        wide enclosure (0.30 m x 0.12 m, a circuit board inside) that showed 0.14 m of it in Explore: the
        fan, the intake and the exhaust were outside the picture while you were meant to move them.
        """
        rect = closed_rect(sim)
        if rect is not None:
            self.view.set_focus_rect(rect, contain=True)

    def _show_preview(self, sim: Simulation, design) -> None:
        """Before any frame: the shape in the wind frame and the status of the simulation."""
        self._shown_sim_id = ""
        mode = self._mode_for(sim)
        self._set_mode(mode, design, sim)
        self.view.clear()
        self.view.set_axis_names(("x", "y"))
        self.history.set_x_limits(None)
        self._clear_monitors("Latest angle" if mode in ("polar", "wing") else "Latest time step")
        ground = ground_of(sim)
        self.view.set_ground(ground)
        if design is not None and mode == "field":
            self.view.set_bodies(wind_bodies(design, sim.conditions.alpha_deg, ground=ground is not None))
            self._frame_closed(sim)
        self._show_placements(sim, design)
        if sim.status == "failed":
            self._set_placeholder("The last run failed", sim.error or "Details are in Messages.")
        else:
            self._set_placeholder("Not run", "Run (F5) starts the simulation.")
        self._set_status_from_sim(sim)
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 if sim.status == "done" else 0)
        self._reset_run_rows()
        self.other_note.hide()
        self.results_btn.setText("Show results")
        self.results_btn.setVisible(sim.status in ("done", "cancelled"))
        if mode == "polar":
            self._start_polar(sim)
            self._draw_polar_points()
            res = self.jobs.result(sim.id) if sim.status in ("done", "cancelled") else None
            if isinstance(res, PolarResult):
                self._show_polar_result(sim, res)
            else:
                self.polar_caption.setText(
                    f"Polar sweep · {self._polar_total} angles · not run" if sim.status != "failed"
                    else "The last sweep failed — see Messages")
        elif mode == "wing":
            self._start_wing(sim)
            res = self.jobs.result(sim.id) if sim.status in ("done", "cancelled") else None
            if isinstance(res, WingResult):
                self._show_wing_result(sim, res)
            else:
                self.wing_caption.setText(
                    f"Wing polar · {self._polar_total} angles · not run" if sim.status != "failed"
                    else "The last sweep failed — see Messages")
        elif mode == "field3d":
            self._reset_slice_controls()
            if sim.status in ("done", "cancelled") and self._show_last_result(sim):
                return
            self._show_slice_preview(sim, design)
        elif sim.status in ("done", "cancelled"):
            self._show_last_result(sim)

    def _set_status_from_sim(self, sim: Simulation) -> None:
        text, kind = STATUS_TEXT.get(sim.status, (sim.status.capitalize(), None))
        if sim.status == "done":
            secs = getattr(self.jobs, "job_seconds", lambda _sid: None)(sim.id)
            if secs is None:
                cached = getattr(self.jobs, "cached_result", None)
                res = (cached(sim.id) if cached is not None else None) or self.jobs.result(sim.id)
                secs = self._run_seconds(sim.id, res)
            if secs is not None and math.isfinite(secs):
                text = f"Done in {fmt_duration(secs)}"
        self.status_text.set_status(text, kind, (sim.error or "") if sim.status == "failed" else "")

    def _show_last_result(self, sim: Simulation) -> bool:
        """A finished simulation opens on its own flow, not on an empty tunnel.

        Shows the stored (time-averaged if available) field in the selected
        field, the averaged coefficients and the force history with its
        averaging window, so coming back to this page is not a step backwards.
        A 3D run shows its stored mean slice at the slice strip's position.
        Returns whether a field was put on screen.
        """
        res = self.jobs.result(sim.id)
        if isinstance(res, CFD3DResult):
            return self._show_last_result3d(sim, res)
        if not isinstance(res, CFDResult):
            return False
        shown = False
        fd = res.mean_fields or res.instant_fields
        design = self._design_of(sim)
        ground = ground_of(sim)
        self.view.set_ground(ground)
        if fd is not None and fd.fields:
            name = self.field_combo.currentData()
            name = name if name in fd.fields else ("speed" if "speed" in fd.fields else next(iter(fd.fields)))
            lines = [np.asarray(b.points) for b in res.bodies_wind] if res.bodies_wind else None
            if not lines:
                lines = wind_bodies(design, sim.conditions.alpha_deg, ground=ground is not None) \
                    if design is not None else None
            if lines:
                self.view.set_bodies(lines)   # first, so the default framing is around them
            self._frame_closed(sim)
            # same colour rule as the Results page: sampled around the body, speed from 0
            self.view.set_field(fd.fields[name], fd.extent, field=name, cmap=self.cmap_combo.currentData(),
                                mask=fd.mask, window=body_window(lines, fd.extent))
            self.view.set_caption("Time-averaged · last run" if fd.label == "mean" else "Final snapshot · last run")
            self._shown_sim_id = sim.id
            shown = True
        self.monitor_note.setText("Mean of the last run")
        if self._mine(sim.id):           # the Run rows describe the selected simulation only
            self._done_rows(sim, res)
        self.history.set_x_limits(None)
        if self._closed_mon:
            self.monitor_note.setText("End of the last run")
            self._closed_monitor_result(res)
            return shown
        self._show_mean_coefficients(res)
        self.history.set_history(res.history or None,
                                 (res.t_avg_start, res.t_end) if math.isfinite(res.t_avg_start) else None)
        return shown

    def _planned_t_end(self, sim: Simulation) -> float | None:
        """The end time [L/U] a run of ``sim`` was planned (or last extended) to, if known."""
        t = self._run_t_end.get(sim.id) or getattr(self._job_plans.get(sim.id), "t_end", None)
        if t is None:
            design = self._design_of(sim)
            plan = self._plans.get(plan_key(sim, design)) if design is not None else None
            t = getattr(plan, "t_end", None)
        if t is None:
            t = getattr(getattr(sim, "cfd", None), "duration", None)
        try:
            t = float(t)
        except (TypeError, ValueError):
            return None
        return t if math.isfinite(t) and t > 0 else None

    def _done_rows(self, sim: Simulation, res) -> None:
        """Phase and Time of a finished run, saying when it stopped early because it had converged.

        "Done · 16.3 L/U" hid that a run planned for 20 L/U ended at 16.3 because
        its averages were already known to the target accuracy, which is
        exactly what tells the user the numbers are settled.
        """
        self._rows_done_sim = sim.id
        t_end = getattr(res, "t_end", float("nan"))
        planned = self._planned_t_end(sim)
        stopped = sim.status != "done"
        early = (not stopped and bool(getattr(res, "converged", False)) and planned is not None
                 and math.isfinite(t_end) and t_end < 0.98 * planned)
        if early:
            self.phase_value.setText("Converged (stopped early)")
            self.time_value.setText(f"{sig(t_end, 3)} of {sig(planned, 3)} L/U")
            tip = (f"Converged at {sig(t_end, 3)} of {sig(planned, 3)} L/U: the run stopped early once its "
                   "averages were known to the target accuracy.")
        else:
            self.phase_value.setText("Stopped" if stopped else "Done")
            self.time_value.setText(f"{sig(t_end, 3)} L/U" if math.isfinite(t_end) else "—")
            tip = ""
        self.phase_value.setToolTip(tip)
        self.time_value.setToolTip(tip or "Simulated time in convective units (reference length / speed)")

    def _run_seconds(self, sim_id: str, res=None) -> float | None:
        """The run's duration: the job's wall time in this session (one source with the status bar),
        else the stored result's own ``wall_seconds``."""
        fn = getattr(self.jobs, "job_seconds", None)
        secs = fn(sim_id) if callable(fn) else None
        if secs is None or not math.isfinite(secs):
            secs = getattr(res, "wall_seconds", None) if res is not None else None
        return secs if secs is not None and math.isfinite(secs) and secs > 0 else None

    def _show_mean_coefficients(self, res) -> None:
        """The averaged coefficients, rounded to their standard errors as in the Results table."""
        keys = ("cd", "cs", "cl") if isinstance(res, CFD3DResult) else ("cd", "cl")
        for key in keys:
            st = getattr(res, key)
            d = decimals_from_error(st.stderr)
            self.coef_values[key].setText(fmt_num(st.mean, 4) if d is None else
                                          f"{fmt_fixed(st.mean, d)}  ± {fmt_fixed(st.stderr, d)}")
        self.coef_values["ld"].setText(fmt_num(res.lift_to_drag, 3))

    def _fill_sim_combo(self) -> None:
        project = self.state.project
        self.sim_combo.blockSignals(True)
        self.sim_combo.clear()
        if project is not None:
            for sid, sim in project.simulations.items():
                d = project.designs.get(sim.design_id)
                self.sim_combo.addItem(f"{sim.name} · {d.name if d else '?'}", sid)
            self.sim_combo.setCurrentIndex(self.sim_combo.findData(self.state.simulation_id))
        self.sim_combo.setEnabled(self.sim_combo.count() > 0)
        self.sim_combo.blockSignals(False)
        sim = self.state.simulation
        self.sim_title.setText(sim.name if sim is not None else "")
        if sim is not None and sim.id != self.jobs.running_id() and not self.live.is_running():
            self._set_status_from_sim(sim)

    # ---------------------------------------------------------------- plans
    def _run_plan(self, sim_id: str):
        """The plan the running job of ``sim_id`` executes, if it has reported one."""
        plan_of = getattr(self.jobs, "plan_of", None)
        return (plan_of(sim_id) if plan_of is not None else None) or self._job_plans.get(sim_id)

    def _job_inputs(self, sim_id: str):
        """(design, simulation) the running job of ``sim_id`` works on, (None, None) if unknown."""
        fn = getattr(self.jobs, "job_inputs", None)
        return fn(sim_id) if fn is not None else (None, None)

    def _request_plan(self, sim: Simulation, design) -> None:
        """Show the plan: the running job's own, else a cached or freshly made one (off the GUI thread).

        While explore runs, the Run panel describes the explore session instead.
        """
        if self.live.is_running() and getattr(self.live, "plan", lambda: None)() is not None:
            self._cancel_plan_call()
            self._show_live_plan()
            return
        self.plan_note.hide()
        self.plan_error.hide()
        running = self.jobs.running_id() == sim.id
        if running:
            # the runner works on copies made when the job started: describe those
            job_design, job_sim = self._job_inputs(sim.id)
            if job_sim is not None:
                sim, design = job_sim, job_design
        if design is None:
            self._cancel_plan_call()
            self._show_plan_rows("—", "—", "—")
            self.plan_error.set_text("The design of this simulation is missing.")
            self.plan_error.show()
            return
        c = sim.conditions
        re_real = c.reynolds(design.length_ref)
        if sim.analysis == "polar":
            self._cancel_plan_call()
            a = sim.airfoil
            n = len(a.alphas())
            self._row_grid.set_label("Angles")
            self._row_est.set_label("Ncrit")
            self._row_re.set_label("Reynolds number")
            angles = f"{n}, {deg_text(a.alpha_start)} to {deg_text(a.alpha_end)}, step {deg_text(a.alpha_step)}"
            self._show_plan_rows(angles, fmt_re(re_real), f"{a.ncrit:g}")
            return
        if sim.analysis == "wing":
            self._cancel_plan_call()
            w = sim.wing
            n = len(w.alphas())
            self._row_grid.set_label("Angles")
            self._row_est.set_label("Method")
            self._row_re.set_label("Re (MAC)")
            angles = f"{n}, {deg_text(w.alpha_start)} to {deg_text(w.alpha_end)}, step {deg_text(w.alpha_step)}"
            method = ("Vortex lattice" if w.method == "vlm" else "Lifting line") + \
                     (f", viscous (Ncrit {w.ncrit:g})" if w.viscous else ", inviscid")
            self._show_plan_rows(angles, fmt_re(re_real), method)
            self.estimate_value.setToolTip("Viscous runs compute 2D section polars first (cached for later runs); "
                                           "the lattice solves take seconds")
            return
        self._row_grid.set_label("Grid")
        self._row_est.set_label("Estimate")
        self._row_re.set_label("Reynolds number")
        self.estimate_value.setToolTip("Estimated wall-clock time")
        plan = self._run_plan(sim.id) if running else None
        key = plan_key(sim, design)
        if plan is None:
            plan = self._plans.get(key)
            if plan is not None:
                self._plans.move_to_end(key)
        if plan is not None:
            self._cancel_plan_call()
            self._apply_plan(sim, plan)
            return
        self._show_plan_rows(f"{sim.cfd.quality.capitalize()} quality", f"{fmt_re(re_real)} (real)", "—")
        if self._plan_call is not None and self._plan_call_key == key:
            return      # the same plan is already being made (a status change re-asked for it)
        self._cancel_plan_call()
        plan_mod = lazy_import("aerolume.simulation.plan")
        if plan_mod is None:
            return
        sid = sim.id
        three_d = getattr(design, "dimension", 2) == 3
        # copies with their mesh arrays: Setup may edit the simulation or the design while the worker plans
        plan_sim = clone_simulation(sim)
        plan_design = clone_design(design)

        def still_wanted() -> bool:
            cur = self.state.simulation
            return cur is not None and cur.id == sid and self._plan_call_key == key

        def done(out) -> None:
            self._plan_call = None
            new_plan = out
            if three_d and isinstance(out, tuple) and len(out) == 2:
                new_plan, sections = out
                self._sections[key] = sections
                while len(self._sections) > PLAN_CACHE_SIZE:
                    self._sections.pop(next(iter(self._sections)))
            self._plans[key] = new_plan
            while len(self._plans) > PLAN_CACHE_SIZE:
                self._plans.popitem(last=False)
            if still_wanted() and not self.live.is_running():     # explore owns the panel then
                self._apply_plan(plan_sim, new_plan)
                if three_d and not self.jobs.running_id() and not self.view.has_image():
                    self._show_slice_preview(self.state.simulation, self._design_of(self.state.simulation))

        def failed(msg: str) -> None:
            self._plan_call = None
            if still_wanted():
                self.plan_error.set_text(f"Cannot plan this setup: {pretty(msg)}")
                self.plan_error.show()

        self._plan_call_key = key
        if three_d:
            airfoils = airfoil_designs(self.state.project, design)
            self._plan_call = run_in_background(_plan_with_sections, plan_mod.plan_simulation, plan_design,
                                                plan_sim, airfoils, on_done=done, on_error=failed)
        else:
            self._plan_call = run_in_background(plan_mod.plan_simulation, plan_design, plan_sim,
                                                on_done=done, on_error=failed)

    def _cancel_plan_call(self) -> None:
        if self._plan_call is not None:
            self._plan_call.cancelled = True
            self._plan_call = None
        self._plan_call_key = ""

    def _show_plan_rows(self, grid: str, re_text: str, estimate: str) -> None:
        self.grid_value.setText(grid)
        self.re_value.setText(re_text)
        self.estimate_value.setText(estimate)

    @staticmethod
    def _grid_text(plan, suffix: str = "") -> str:
        """'320 × 160 · 51,200 cells' (2D) or '268 × 80 × 88 · 1.89 M cells' (3D)."""
        cfg = plan.solver_config
        nz = getattr(cfg, "nz", None)
        cells = getattr(plan, "cells", 0)
        if nz:
            return f"{cfg.nx} × {cfg.ny} × {nz} · {cells / 1e6:.2f} M cells{suffix}"
        return f"{cfg.nx} × {cfg.ny} · {cells:,} cells{suffix}"

    def _apply_plan(self, sim: Simulation, plan) -> None:
        try:
            grid = self._grid_text(plan)
        except AttributeError:
            grid = f"{getattr(plan, 'cells', 0):,} cells"
        re_sim, re_real = getattr(plan, "re_sim", float("nan")), getattr(plan, "re_real", float("nan"))
        if math.isfinite(re_sim) and math.isfinite(re_real) and abs(re_sim / re_real - 1) > 0.01:
            re_text = f"{fmt_re(re_sim)} simulated ({fmt_re(re_real)} real)"
        else:
            re_text = fmt_re(re_real)
        self._show_plan_rows(grid, re_text, fmt_duration(getattr(plan, "est_seconds", None)))
        self.grid_value.setToolTip("")
        self.sym_value.setToolTip("")
        self.plan_note.set_warnings(list(getattr(plan, "warnings", []) or []))
        if getattr(plan, "dimension", 2) == 3:
            self.sym_value.setText("Plane y = 0 (half domain)" if getattr(plan, "symmetry_plane", False)
                                   else "None (full domain)")
            if self._is3d():
                self._update_slice_range()
        # a finished run shown before its plan was known: now its planned end time is, so the
        # Run rows can say whether it stopped early at convergence
        cur = self._sim(sim.id)
        if (cur is not None and self._rows_done_sim == cur.id and self._mine(cur.id)
                and cur.id != self.jobs.running_id() and not self.live.is_running()):
            cached = getattr(self.jobs, "cached_result", None)
            res = cached(cur.id) if cached is not None else None
            if isinstance(res, (CFDResult, CFD3DResult)):
                self._done_rows(cur, res)

    # --------------------------------------------------------- run control
    def run_current(self) -> None:
        sim = self.state.simulation
        if sim is None:
            self.state.message("Select a simulation in the Project panel first.")
            return
        if self.live.is_running():
            self.live.stop()
        self._finished_id = ""
        try:
            self.jobs.submit(sim.id)
        except (RuntimeError, KeyError) as exc:
            self._log(f"Could not start “{sim.name}”: {exc}", "error")
            return
        if self.jobs.running_id() != sim.id:
            pos = self.jobs.queued_ids().index(sim.id) + 1 if sim.id in self.jobs.queued_ids() else 0
            self.status_text.set_status(f"Queued ({pos} waiting)" if pos > 1 else "Queued", None)
            self._log(f"“{sim.name}” queued; it starts when the current run ends")

    def stop(self) -> None:
        """The panel's Stop button: acts on the simulation the panel names, never on another.

        Explore stops explore. A queued simulation leaves the queue (the button
        reads "Remove from queue"). The running one is stopped. A run of some
        other simulation is left alone: pressing Stop under "CFD α 4° — Queued"
        used to kill the polar that happened to be running. (The window's own
        Stop, Shift+F5, stops whatever runs; its tooltip names it.)
        """
        if self.live.is_running():
            self.live.stop()
            return
        sim = self.state.simulation
        if sim is None:
            return
        if sim.id in self.jobs.queued_ids():
            self.jobs.cancel(sim.id)
            return
        if sim.id == self.jobs.running_id():
            self.jobs.cancel(sim.id)
            self.phase_value.setText(PHASE_TEXT["stopping"])
            self.status_text.set_status("Stopping…", "warn")

    def _dequeue_selected(self) -> None:
        """Remove the simulations selected in the Queue list (or the selected simulation, if queued)."""
        picked = [i.data(Qt.ItemDataRole.UserRole) for i in self.queue_list.selectedItems()]
        sim = self.state.simulation
        if not picked and sim is not None:
            picked = [sim.id]
        for sid in picked:
            if sid in self.jobs.queued_ids():
                self.jobs.cancel(sid)

    def _update_buttons(self) -> None:
        sim = self.state.simulation
        busy = sim is not None and self.jobs.is_busy(sim.id)
        rid = self.jobs.running_id()
        queued = self.jobs.queued_ids()
        self.run_btn.setEnabled(sim is not None and not busy)
        self.run_btn.setText("Run again" if sim is not None and sim.status in ("done", "cancelled") else "Run")
        if self.live.is_running():
            text, enabled, tip = "Stop", True, "Stop the explore session"
        elif sim is not None and sim.id in queued:
            text, enabled, tip = "Remove from queue", True, f"Take “{sim.name}” out of the queue; it does not run"
        elif sim is not None and sim.id == rid:
            text, enabled, tip = "Stop", True, "Stop this simulation (Shift+F5)"
        elif rid:
            text, enabled, tip = "Stop", False, (f"Running: “{self._sim_name(rid)}”. Select it to stop it, or use "
                                                 "Simulation ▸ Stop (Shift+F5).")
        else:
            text, enabled, tip = "Stop", False, "Nothing is running"
        self.stop_btn.setText(text)
        self.stop_btn.setEnabled(enabled)
        self.stop_btn.setToolTip(tip)
        sel = [i.data(Qt.ItemDataRole.UserRole) for i in self.queue_list.selectedItems()]
        self.dequeue_btn.setEnabled(any(s in queued for s in sel) or (sim is not None and sim.id in queued))
        self.promote_btn.setEnabled(sim is not None)
        self.live_btn.setEnabled(sim is not None and not rid)

    def _refresh_queue(self) -> None:
        project = self.state.project
        self.queue_list.clear()
        queued = self.jobs.queued_ids()
        for sid in queued:
            sim = project.simulations.get(sid) if project is not None else None
            item = QListWidgetItem(sim.name if sim else sid)
            item.setData(Qt.ItemDataRole.UserRole, sid)
            self.queue_list.addItem(item)
        self.queue_group.setVisible(bool(queued))
        self.queue_group.header.setText(f"Queue ({len(queued)})" if queued else "Queue")

    # ------------------------------------------------------- job signals
    def _sim(self, sim_id: str) -> Simulation | None:
        p = self.state.project
        return p.simulations.get(sim_id) if p is not None else None

    def _sim_name(self, sim_id: str) -> str:
        s = self._sim(sim_id)
        return s.name if s else sim_id

    def _t_end(self, sim_id: str) -> float | None:
        """End time of the run: what the runner last reported, else its plan's."""
        t = self._run_t_end.get(sim_id)
        if t is None:
            t = getattr(self._run_plan(sim_id), "t_end", None)
        return t if t is not None and math.isfinite(t) else None

    def _update_eta(self, sim_id: str, frac: float, measured: float | None) -> None:
        """'Remaining': the planner's estimate early on, then the measured value, smoothed.

        The first percent of a run is dominated by building the solver and
        capturing CUDA graphs, so a rate measured there promises far too little
        time (2 min 54 s against 5 min 30 s in practice). Until
        :data:`ETA_MEASURED_FROM` of the run the planner's estimate minus the
        elapsed time is shown; afterwards the solver's measurement, through an
        exponential average that starts from the planner's value so the
        hand-over does not jump.
        """
        _, _, elapsed = self.jobs.progress_of(sim_id)
        est = getattr(self._run_plan(sim_id), "est_seconds", None)
        planned = max(est - elapsed, 0.0) if est is not None and math.isfinite(est) else None
        now = time.monotonic()
        if self._eta is not None:
            self._eta = max(self._eta - (now - self._eta_at), 0.0)   # count down between updates
        self._eta_at = now
        if frac < ETA_MEASURED_FROM or measured is None or not math.isfinite(measured):
            if frac < ETA_MEASURED_FROM and planned is not None:
                self._eta = planned
            elif self._eta is None and 0.02 < frac < 1.0 and elapsed > 0:
                self._eta = elapsed * (1 - frac) / frac
        elif self._eta is None:
            self._eta = float(measured)
        else:
            self._eta += ETA_SMOOTHING * (float(measured) - self._eta)
        self.eta_value.setText(fmt_duration(self._eta) if self._eta is not None else "—")

    def _set_time_axis(self, sim_id: str) -> None:
        """Fix the monitor's time axis to the run (0 .. end time) once the end time is known."""
        t_end = self._t_end(sim_id)
        self.history.set_x_limits((0.0, t_end) if t_end else None)

    def _on_job_started(self, sim_id: str) -> None:
        self._rows_done_sim = ""
        self._shown_sim_id = sim_id
        self._finished_id = ""
        self._milestone[sim_id] = 0
        self._phase_seen.pop(sim_id, None)
        self._run_t_end.pop(sim_id, None)
        self._job_plans.pop(sim_id, None)
        self._eta = None
        self._reset_live_range()
        sim = self._sim(sim_id)
        design = self._design_of(sim)
        mode = self._mode_for(sim)
        self._set_mode(mode, design, sim)
        self.view.clear()
        ground = ground_of(sim)
        self.view.set_ground(ground)
        self._set_placeholder("Starting the solver…", "")
        if design is not None and mode == "field":
            self.view.set_bodies(wind_bodies(design, sim.conditions.alpha_deg, ground=ground is not None))
            self._frame_closed(sim)
        self._show_placements(sim, design)
        if self._diagnostics_shown():
            self._bind_diagnostics()
        if mode == "field3d":
            self._live_volume = self._live_lines = self._last_frame3d = None
            self._reset_slice_controls()
            self._push_flow_settings()
            self._show_slice_preview(sim, design, placeholder=("Starting the solver…",
                                                               "Building the 3D grid and the body masks"))
        self.history.set_x_limits(None)
        self._clear_monitors("Latest angle" if mode in ("polar", "wing") else "Latest time step")
        mine = self._mine(sim_id)
        if mine:
            self.status_text.set_status("Running 0 %", "accent")
            self.phase_value.setText(PHASE_TEXT["starting"])
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)
            for w in (self.time_value, self.rate_value, self.eta_value):
                w.setText("—")
            self.results_btn.hide()
        elif self.state.simulation is not None:
            self._blank_run_rows(self.state.simulation)
        self._update_buttons()
        if mode == "polar":
            self._start_polar(sim)
            self._draw_polar_points()
            # The first angle is cold-started and takes most of the sweep's start-up
            # (8.5 s of a 30 s NACA 2412 sweep): without a sign of life, "0 of 19 angles"
            # over four empty plots read as "it does not run".
            first = self._first_polar_angle(sim)
            self.polar_caption.setText(f"Polar sweep · solving α = {first} first (the first angle takes "
                                       f"longest) · 0 of {self._polar_total} angles")
            if mine:
                self.time_value.setText(f"0 of {self._polar_total} angles")
                self.status_text.set_status(f"Running · solving α = {first}", "accent")
                self.progress.setRange(0, 0)            # busy until the first angle is in
        elif mode == "wing":
            self._start_wing(sim)
            viscous = bool(getattr(sim.wing, "viscous", True)) if sim is not None else True
            self.wing_caption.setText("Wing polar · computing the section polars first" if viscous
                                      else "Wing polar · solving")
            if mine:
                self.time_value.setText(f"0 of {self._polar_total} angles")
                self.phase_value.setText(PHASE_TEXT["polars"] if viscous else PHASE_TEXT["solve"])
        self._update_other_note()
        if sim is not None and sim.id == self.state.simulation_id:
            self._request_plan(sim, design)

    def _adopt_running(self, sim_id: str) -> None:
        """Take on a run that is already in progress as if this page had seen it start.

        Without this a page that first meets a run mid-way (built after the job
        started, or showing something else when it did) kept whatever mode it
        had: a 3D run started from the Script console or by the assistant was
        drawn with the 2D live UI (a full-size y = 0 slice, Particles instead of
        Smoke, C_d / C_l monitors) until it finished.
        """
        milestone = self._milestone.get(sim_id)
        self._on_job_started(sim_id)
        if milestone is not None:
            self._milestone[sim_id] = milestone        # no second "25 % done" line in Messages
        plan = self._run_plan(sim_id)
        if plan is not None:
            self._job_plans[sim_id] = plan
            if sim_id == self.state.simulation_id:
                self._apply_plan(self._job_inputs(sim_id)[1] or self._sim(sim_id), plan)
            self._set_time_axis(sim_id)
            if getattr(plan, "dimension", 2) == 3 and self._is3d():
                self._update_slice_range()
                self._plan_to_3d(plan, self._sim(sim_id))
        self._restore_running_rows(sim_id)

    def _frame_is_3d(self, sim_id: str, frame) -> bool:
        """Whether a run's frames are 3D: the frame says so (a slice axis or a volume), or its plan does."""
        if getattr(frame, "slice_axis", None) or getattr(frame, "volume", None) is not None:
            return True
        return getattr(self._run_plan(sim_id), "dimension", 2) == 3

    @staticmethod
    def _first_polar_angle(sim: Simulation | None) -> str:
        """The angle a sweep solves first: the one closest to zero (solvers.airfoil.polar)."""
        alphas = sim.airfoil.alphas() if sim is not None else np.zeros(0)
        if not len(alphas):
            return "0°"
        return deg_text(float(alphas[int(np.argmin(np.abs(alphas)))]))

    def _on_job_planned(self, sim_id: str, plan) -> None:
        """The runner's own plan: the Run group, the time axis and the run log describe exactly it."""
        self._job_plans[sim_id] = plan
        if sim_id == self.state.simulation_id:
            job_sim = self._job_inputs(sim_id)[1] or self._sim(sim_id)
            self._apply_plan(job_sim, plan)
            self.plan_error.hide()
        if (sim_id == self._shown_sim_id and getattr(plan, "dimension", 2) == 3 and not self._is3d()
                and self._mode not in ("polar", "wing") and not self.live.is_running()):
            self._adopt_running(sim_id)             # a 3D plan for a run shown with the 2D live UI
        if sim_id == self._shown_sim_id:
            self._set_time_axis(sim_id)
            if getattr(plan, "dimension", 2) == 3 and self._is3d():
                self._update_slice_range()
                self._plan_to_3d(plan, self._sim(sim_id))
        try:
            cfg = plan.solver_config
            nz = getattr(cfg, "nz", None)
            cells = f"{cfg.nx} × {cfg.ny} × {nz} cells" if nz else f"{cfg.nx} × {cfg.ny} cells"
            if nz and getattr(plan, "symmetry_plane", False):
                cells += " (half domain, symmetry plane y = 0)"
            self._log(f"“{self._sim_name(sim_id)}”: {cells}, Re {fmt_re(plan.re_sim)} "
                      f"simulated, {plan.turbulence}, t = {sig(plan.t_end, 3)} L/U "
                      f"(averaging from {sig(plan.t_avg_start, 3)}); estimated {fmt_duration(plan.est_seconds)}")
        except AttributeError:
            pass

    def _on_job_progress(self, sim_id: str, fraction: float, message: str) -> None:
        if sim_id != self.jobs.running_id():
            return
        frac = max(0.0, min(1.0, fraction))
        mine = self._mine(sim_id)        # the Run rows describe the selected simulation only
        sim = self._sim(sim_id)
        polar = sim is not None and sim.analysis == "polar"
        wing = sim is not None and sim.analysis == "wing"
        if mine:
            if not (polar and self.progress.maximum() == 0 and not self._polar_points):
                self.progress.setValue(round(1000 * frac))
            stopping = self.jobs.progress_of(sim_id)[1] == "Stopping…"
            if stopping:
                self.status_text.set_status("Stopping…", "warn")
                self.phase_value.setText(PHASE_TEXT["stopping"])
            elif not (polar and not self._polar_points):
                self.status_text.set_status(f"Running {frac * 100:.0f} %", "accent", message)
        if polar:
            self._on_polar_progress(sim_id, frac, message)
            # a polar sweep sends no frames; a Navier-Stokes run logs its milestones from the
            # frames instead, which also carry the simulated time and the solver rate
            self._log_milestone(sim_id, frac)
            return
        if wing:
            self._on_wing_progress(sim_id, frac, message)
            self._log_milestone(sim_id, frac)
            return
        m = _T_END_RE.search(message or "")
        if m is not None:
            # one source with the status bar: the end time the runner is actually running to
            try:
                t_now, t_end = float(m.group(1)), float(m.group(2))
            except ValueError:
                t_now = t_end = float("nan")
            if math.isfinite(t_end) and t_end > 0:
                if self._run_t_end.get(sim_id) != t_end:
                    self._run_t_end[sim_id] = t_end
                    if sim_id == self._shown_sim_id:
                        self._set_time_axis(sim_id)
                if mine:
                    self.time_value.setText(f"{sig(t_now, 3)} of {sig(t_end, 3)} L/U")
        if mine and 0.0 < frac < 1.0 and not self.view.has_image():
            self._update_eta(sim_id, frac, None)

    # -- polar sweeps
    def _on_polar_progress(self, sim_id: str, frac: float, message: str) -> None:
        """One more angle of a polar sweep: add it to the plots and the monitors."""
        pt = parse_polar_point(message)
        if pt is not None:
            alpha, cl, cd, ok = pt
            self._polar_points[alpha] = (cl, cd, ok)
            self._draw_polar_points()
            if ok:
                self.monitor_note.setText(f"Latest angle, α = {fmt_num(alpha, 3)}°")
                self.coef_values["cd"].setText(fmt_num(cd, 4))
                self.coef_values["cl"].setText(fmt_num(cl, 4))
                self.coef_values["ld"].setText(fmt_num(_ld(cl, cd), 3))
        n_total = self._polar_total or len(self._polar_points)
        # the runner scales the solver's fraction by 0.99 (saving is the last percent)
        n_done = max(len(self._polar_points), min(n_total, round(frac / 0.99 * n_total)))
        if n_done == 0:
            return          # still on the first angle: keep the "solving α = …" busy state
        mine = self._mine(sim_id)
        if mine:
            if self.progress.maximum() == 0:         # the first angle is in: a real bar from here
                self.progress.setRange(0, 1000)
                self.progress.setValue(round(1000 * frac))
                self.status_text.set_status(f"Running {frac * 100:.0f} %", "accent", message)
            self.time_value.setText(f"{n_done} of {n_total} angles")
        self.polar_caption.setText(f"Polar sweep · {n_done} of {n_total} angles")
        _, _, elapsed = self.jobs.progress_of(sim_id)
        if mine and 0 < n_done < n_total and elapsed > 0:
            self.eta_value.setText(fmt_duration(elapsed / n_done * (n_total - n_done)))

    def _start_polar(self, sim: Simulation | None) -> None:
        """Forget the points of an earlier sweep; the angle axes span the sweep about to run."""
        self._polar_points = {}
        alphas = sim.airfoil.alphas() if sim is not None else np.zeros(0)
        self._polar_total = len(alphas)
        self._polar_alpha_range = (float(np.min(alphas)), float(np.max(alphas))) if len(alphas) else None

    def _fix_alpha_axes(self) -> None:
        """Angle axes span the whole sweep, so the curves fill in from left to right."""
        rng = self._polar_alpha_range
        if rng is None:
            return
        lo, hi = rng
        pad = max(0.03 * (hi - lo), 0.5)
        g = self.polar_view
        for p in (g.cl_alpha, g.cm_alpha, g.ld_alpha):
            p.enableAutoRange(x=False, y=True)
            p.setXRange(lo - pad, hi + pad, padding=0.0)
        for pi in self.polar_monitor.panels.values():
            pi.enableAutoRange(x=False)
            pi.setXRange(lo - pad, hi + pad, padding=0.0)

    def _draw_polar_points(self) -> None:
        """The angles computed so far, in the four polar plots and the angle monitors."""
        g = self.polar_view
        for p in g.plots():
            p.clear_series()
        ok = sorted((a, cl, cd) for a, (cl, cd, conv) in self._polar_points.items() if conv)
        if not ok:
            self.polar_monitor.set_history(None)
            self._fix_alpha_axes()
            return
        a = np.array([p[0] for p in ok])
        cl = np.array([p[1] for p in ok])
        cd = np.array([p[2] for p in ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(cd > 0, cl / cd, np.nan)
        g.cl_alpha.set_series("run", a, cl, color=0, symbol="o")
        g.drag_polar.set_series("run", cd, cl, color=0, symbol="o")
        g.ld_alpha.set_series("run", a, ld, color=0, symbol="o")
        g.drag_polar.autoRange()
        self.polar_monitor.set_history({"t": a, "cd": cd, "cl": cl})
        self._fix_alpha_axes()

    def _show_polar_result(self, sim: Simulation, res: PolarResult) -> None:
        """A finished sweep: its four plots, the angle monitors and the best-L/D numbers."""
        tab = res.table()
        self.polar_view.set_polars([(sim.name, tab, 0)])
        for p in self.polar_view.plots():
            p.autoRange()
        a = np.asarray(tab.get("alpha", []), dtype=float)
        if a.size:
            self.polar_monitor.set_history({"t": a, "cd": np.asarray(tab["cd"], dtype=float),
                                            "cl": np.asarray(tab["cl"], dtype=float)})
        n_ok = sum(1 for p in res.points if p.converged)
        stopped = sim.status == "cancelled"
        self.polar_caption.setText(f"{'Stopped' if stopped else 'Done'}: {n_ok} of {len(res.points)} angles "
                                   f"converged · Re {fmt_re(res.re)}")
        if self._mine(sim.id):
            self.time_value.setText(f"{len(res.points)} of {self._polar_total or len(res.points)} angles")
        best = res.best_ld()
        if best is not None:
            self.monitor_note.setText(f"At best L/D, α = {fmt_num(best.alpha, 3)}°")
            self.coef_values["cd"].setText(fmt_num(best.cd, 4))
            self.coef_values["cl"].setText(fmt_num(best.cl, 4))
            self.coef_values["ld"].setText(fmt_num(_ld(best.cl, best.cd), 3))
        self._shown_sim_id = sim.id

    # -- wing sweeps
    def _start_wing(self, sim: Simulation | None) -> None:
        """Forget the last wing run's points; the run view lists this one's section polars."""
        alphas = sim.wing.alphas() if sim is not None else np.zeros(0)
        self._polar_total = len(alphas)
        self._wing_angles = set()
        self._polar_alpha_range = (float(np.min(alphas)), float(np.max(alphas))) if len(alphas) else None
        self.wing_view.reset(self._polar_total, viscous=bool(getattr(getattr(sim, "wing", None), "viscous", True)))
        self.polar_monitor.set_history(None)

    def _on_wing_progress(self, sim_id: str, frac: float, message: str) -> None:
        """A section-polar angle (table + live cl-α) or one more angle of the wing solve."""
        info = parse_wing_message(message)
        mine = self._mine(sim_id)
        n_total = self._polar_total or 1
        if info is not None and info[0] == "section":
            _kind, airfoil, re_, alpha, cl, cd, ok = info
            self.wing_view.add_section_point(airfoil, re_, alpha, cl, cd, ok)
            nodes = self.wing_view.nodes()
            n_pts = sum(len(v) for v in nodes.values())
            self.wing_caption.setText(f"Section polars · {len(nodes)} airfoil/Re node"
                                      f"{'s' if len(nodes) != 1 else ''} · {n_pts} angles computed")
            if mine:
                self.phase_value.setText(PHASE_TEXT["polars"])
        elif info is not None and info[0] == "angle":
            self._wing_angles.add(info[1])
            self.wing_view.set_angles_done(len(self._wing_angles))
            if mine:
                self.phase_value.setText(PHASE_TEXT["solve"])
        # the solver reports 90 % of a viscous run for the section polars, 10 % for the angles
        done = len(self._wing_angles) or (round(max(0.0, frac - 0.9) / 0.1 * n_total) if frac > 0.9 else 0)
        done = min(done, n_total)
        if mine:
            self.time_value.setText(f"{done} of {n_total} angles")
            _, _, elapsed = self.jobs.progress_of(sim_id)
            if 0.05 < frac < 0.99 and elapsed > 0:
                self.eta_value.setText(fmt_duration(elapsed * (1 - frac) / frac))
        if done:
            self.wing_caption.setText(f"Wing polar · {done} of {n_total} angles solved")

    def _show_wing_result(self, sim: Simulation, res: WingResult) -> None:
        """A finished wing sweep: its polar over the section curves, the monitors at best L/D."""
        self.wing_view.show_result(sim.name, res)
        tab = res.table()
        a = np.asarray(tab.get("alpha", []), dtype=float)
        if a.size:
            order = np.argsort(a)
            self.polar_monitor.set_history({"t": a[order], "cd": np.asarray(tab["cd"], float)[order],
                                            "cl": np.asarray(tab["cl"], float)[order]})
            for pi in self.polar_monitor.panels.values():
                pi.enableAutoRange(x=True)
        n_ok = sum(1 for p in res.points if p.converged)
        stopped = sim.status == "cancelled"
        self.wing_caption.setText(f"{'Stopped' if stopped else 'Done'}: {n_ok} of {len(res.points)} angles "
                                  f"converged · Re {fmt_re(res.re_mac)} on the MAC")
        if self._mine(sim.id):
            self.time_value.setText(f"{len(res.points)} of {self._polar_total or len(res.points)} angles")
            self.phase_value.setText("Done" if not stopped else "Stopped")
        pts = [p for p in res.points if p.converged and p.cd and p.cd > 0]
        best = max(pts, key=lambda p: p.cl / p.cd) if pts and res.viscous else None
        if best is None and pts:
            best = max(pts, key=lambda p: p.cl)
        if best is not None:
            what = "At best L/D" if res.viscous else "At the highest CL"
            self.monitor_note.setText(f"{what}, α = {fmt_num(best.alpha, 3)}°")
            self.coef_values["cd"].setText(fmt_num(best.cd, 4))
            self.coef_values["cl"].setText(fmt_num(best.cl, 4))
            self.coef_values["ld"].setText(fmt_num(_ld(best.cl, best.cd), 3) if res.viscous else "—")
        self._shown_sim_id = sim.id

    def _log_milestone(self, sim_id: str, frac: float, frame=None) -> None:
        done = self._milestone.get(sim_id, 0)
        if done >= len(MILESTONES) or frac < MILESTONES[done]:
            return
        while done < len(MILESTONES) and frac >= MILESTONES[done]:
            done += 1
        self._milestone[sim_id] = done
        parts = [f"{round(MILESTONES[done - 1] * 100)} % done"]
        t = getattr(frame, "t", None) if frame is not None else None
        if t is not None and math.isfinite(t):
            t_end = self._t_end(sim_id)
            parts.append(f"t = {sig(t, 3)}" + (f" of {sig(t_end, 3)} L/U" if t_end else " L/U"))
        sps = getattr(frame, "steps_per_s", None) if frame is not None else None
        if sps is not None and math.isfinite(sps) and sps > 0:
            parts.append(f"{sps:,.0f} steps/s")
        self._log(f"“{self._sim_name(sim_id)}”: " + " · ".join(parts))

    def _on_job_frame(self, sim_id: str, frame) -> None:
        if self.live.is_running():
            return
        if not self._is3d() and self._mode not in ("polar", "wing") and self._frame_is_3d(sim_id, frame):
            # a 3D run on the 2D live UI: its mode was resolved without its design (see _adopt_running)
            self._adopt_running(sim_id)
        was = self._shown_sim_id
        self._shown_sim_id = sim_id
        if was != sim_id:
            self._update_other_note()
        self._show_frame(frame)
        mine = self._mine(sim_id)        # the Run rows describe the selected simulation only
        if mine:
            self._rows_done_sim = ""     # the rows follow the run's frames now
        phase = getattr(frame, "phase", "")
        if phase:
            if mine:
                self.phase_value.setText(PHASE_TEXT.get(phase, phase.capitalize()))
            prev = self._phase_seen.get(sim_id)
            if prev == "transient" and phase == "averaging":
                t = getattr(frame, "t", float("nan"))
                self._log(f"“{self._sim_name(sim_id)}”: transient over at t = {sig(t, 3)} L/U; "
                          "averaging forces and fields")
            self._phase_seen[sim_id] = phase
        prog = getattr(frame, "progress", None)
        frac = float("nan")
        if prog is not None and math.isfinite(prog):
            frac = max(0.0, min(1.0, prog))
            if mine:
                self.progress.setValue(round(1000 * frac))
                if self.status_text.kind == "accent":
                    self.status_text.set_status(f"Running {frac * 100:.0f} %", "accent", self.status_text.toolTip())
            self._log_milestone(sim_id, frac, frame)
        if not mine:
            return
        sps = getattr(frame, "steps_per_s", float("nan"))
        if phase == "done":
            self.eta_value.setText("—")
        elif math.isfinite(frac):
            self._update_eta(sim_id, frac, getattr(frame, "eta_s", None))
        self.rate_value.setText(f"{sps:,.0f} steps/s" if sps and math.isfinite(sps) else "—")
        t = getattr(frame, "t", float("nan"))
        t_end = self._t_end(sim_id)
        if math.isfinite(t):
            # t_end is the runner's own (extended) end time once a progress message named it;
            # a frame can still be a moment ahead of that message, so never print "23.7 of 20.0"
            within = t_end is not None and t <= t_end * 1.001
            self.time_value.setText(f"{sig(t, 3)} of {sig(t_end, 3)} L/U" if within else f"{sig(t, 3)} L/U")

    def _on_job_finished(self, sim_id: str, result) -> None:
        self._finished_id = sim_id
        name = self._sim_name(sim_id)
        sim = self._sim(sim_id)
        if self._mine(sim_id):
            self.phase_value.setText(PHASE_TEXT["done"])
            self.progress.setRange(0, 1000)
            self.progress.setValue(1000)
            secs = self._run_seconds(sim_id, result)
            self.status_text.set_status(f"Done in {fmt_duration(secs)}" if secs else "Done", "good")
            self.eta_value.setText("—")
            self.results_btn.setText("Show results")
            self.results_btn.show()
        self._update_other_note()
        self._update_buttons()
        shown_here = sim_id == self._shown_sim_id
        if shown_here and not self.view.has_image():
            # never leave "Starting the solver…" behind a finished run, whatever its analysis
            self._set_placeholder("Done", "A polar sweep has no flow field." if
                                      isinstance(result, (PolarResult, WingResult)) else "")
        if isinstance(result, PolarResult) and sim is not None and shown_here:
            self._show_polar_result(sim, result)
        if isinstance(result, WingResult):
            if sim is not None and shown_here:
                self._show_wing_result(sim, result)
            self._log_wing_result(name, result)
        if isinstance(result, (CFDResult, CFD3DResult)):
            self.monitor_note.setText("Mean of the run")
            self._show_mean_coefficients(result)
            if self._mine(sim_id) and sim is not None:
                self._done_rows(sim, result)
            if shown_here:
                if isinstance(result, CFD3DResult) and sim is not None and self._is3d():
                    # the 3D view moves from the live flow to the stored result: mean-flow streamlines,
                    # the smoke snapshot, surface Cp (a live run has no surface data to colour)
                    self._live_volume = self._live_lines = None
                    self._flow_timer.stop()
                    self._show_last_result3d(sim, result)
                elif not self.view.has_image() and sim is not None:
                    # no frame arrived (a very short run): show the saved field, never "Starting…"
                    self._show_last_result(sim)
                if not self.view.has_image():
                    self._set_placeholder("Done", "This run stored no flow field.")
                if result.history:
                    self.history.set_x_limits(None)
                    self.history.set_history(result.history, (result.t_avg_start, result.t_end)
                                             if math.isfinite(result.t_avg_start) else None)
            window = (f" (mean over t = {sig(result.t_avg_start, 3)}–{sig(result.t_end, 3)} L/U)"
                      if math.isfinite(result.t_avg_start) and math.isfinite(result.t_end) else "")
            ld = f", L/D {fmt_num(result.lift_to_drag, 3)}" if is_lifting(self._design_of(sim)) else ""
            measured = physics_summary(result)
            if not math.isfinite(result.cd.mean):
                # a closed box (cavity, room, enclosure) has no forces: lead with what it measured, never
                # "Cd —, Cl —"
                self._log(f"“{name}”: {measured or 'no forces (closed box)'}{window}")
            elif isinstance(result, CFD3DResult):
                cs = (f", CS {mean_error_text(result.cs)}"
                      if not result.symmetry_plane and math.isfinite(result.cs.mean) and result.cs.mean else "")
                self._log(f"“{name}”: {measured + ', ' if measured else ''}CD {mean_error_text(result.cd)}, "
                          f"CL {mean_error_text(result.cl)}{cs}{ld}{window}; "
                          f"drag {fmt_num(result.drag, 4)} N")
            else:
                self._log(f"“{name}”: {measured + ', ' if measured else ''}Cd {mean_error_text(result.cd)}, "
                          f"Cl {mean_error_text(result.cl)}{ld}{window}")
            warned = any("converged" in str(w).lower() for w in getattr(result, "warnings", []) or [])
            if not result.converged and not warned:   # the runner's own warning says it better
                self._log(f"“{name}”: averages not statistically converged; run longer for "
                          "ranking-grade numbers", "warning")
        for w in getattr(result, "warnings", []) or []:
            self._log(f"“{name}”: {pretty(w)}", "warning")

    def _log_wing_result(self, name: str, res: WingResult) -> None:
        n_ok = sum(1 for p in res.points if p.converged)
        an = lazy_import("aerolume.analysis")
        m = {}
        if an is not None and hasattr(an, "wing_metrics"):
            try:
                m = an.wing_metrics(res)
            except Exception:  # noqa: BLE001 - the log line is a convenience
                m = {}
        parts = [f"{n_ok} of {len(res.points)} angles"]
        slope = m.get("cl_alpha_per_rad", float("nan"))
        if math.isfinite(slope):
            parts.append(f"CLα {slope:.3f}/rad")
        if res.viscous and math.isfinite(m.get("ld_max", float("nan"))):
            parts.append(f"(L/D)max {m['ld_max']:.1f} at α = {m.get('alpha_ld_max', float('nan')):g}°")
        e = m.get("e_cruise", float("nan"))
        if math.isfinite(e):
            parts.append(f"e {e:.3f}")
        self._log(f"“{name}”: wing polar done, " + ", ".join(parts))

    def _on_job_failed(self, sim_id: str, message: str) -> None:
        if self._mine(sim_id):
            self.status_text.set_status(STATUS_TEXT["failed"][0], "bad", message)
            self.phase_value.setText("—")
            self.eta_value.setText("—")
            self.progress.setRange(0, 1000)
        if sim_id == self._shown_sim_id:
            self._set_placeholder("The run failed", message)
            if self._mode == "polar":
                self.polar_caption.setText("The sweep failed — see Messages")
            elif self._mode == "wing":
                self.wing_caption.setText("The wing sweep failed — see Messages")
        self._update_other_note()
        self._update_buttons()

    def _on_job_cancelled(self, sim_id: str) -> None:
        sim = self._sim(sim_id)
        mine = self._mine(sim_id)
        if sim is not None and sim.status == "done" and sim_id == self._shown_sim_id:
            # a stopped "Run again": the job manager discarded the partial run and kept the
            # complete result the simulation already had, so that is what goes back on screen
            self._finished_id = sim_id
            if mine:
                self._shown_sim_id = ""
                self._show_preview(sim, self._design_of(sim))
                self.status_text.set_status("Stopped; previous result kept", "warn",
                                            "The re-run was stopped and discarded; the earlier complete "
                                            "result is unchanged.")
            self._update_other_note()
            self._update_buttons()
            return
        if sim_id == self._shown_sim_id:
            has = getattr(self.state.project, "has_result", None)
            kept = sim is not None and sim.status == "cancelled" and callable(has) and bool(has(sim_id))
            self._finished_id = sim_id
            if mine:
                self.status_text.set_status("Stopped", "warn", "The partial result was kept." if kept else "")
                self.phase_value.setText("Stopped")
                self.eta_value.setText("—")
                self.progress.setRange(0, 1000)
                self.results_btn.setText("Show partial results")
                self.results_btn.setVisible(kept)
            if self._mode == "polar":
                res = self.jobs.result(sim_id) if kept else None
                if isinstance(res, PolarResult) and sim is not None:
                    self._show_polar_result(sim, res)
                else:
                    self.polar_caption.setText("Stopped before the first angle")
            elif self._mode == "wing":
                res = self.jobs.result(sim_id) if kept else None
                if isinstance(res, WingResult) and sim is not None:
                    self._show_wing_result(sim, res)
                else:
                    self.wing_caption.setText("Stopped before the first angle")
            elif not self.view.has_image():
                self._set_placeholder("Stopped", "The run was stopped before the first frame.")
        self._update_other_note()
        self._update_buttons()

    def _open_results(self) -> None:
        sid = self._finished_id or self.state.simulation_id
        if sid:
            self.state.select_simulation(sid)
        self.state.navigate("results")

    # ------------------------------------------------------------- frames
    def _reset_live_range(self) -> None:
        """Hand the colour range back to the solver's automatic range (new run, new field)."""
        self._range.reset()
        v = self.jobs.view
        if getattr(v, "range_mode", "auto") != "auto":
            v.range_mode, v.vmin, v.vmax = "auto", None, None
            if self.live.is_running():
                self.live.post(range_mode="auto", vmin=None, vmax=None)

    def _track_range(self, frame) -> None:
        """Steer the solver's colour range for speed frames (see :class:`LiveRangeTracker`).

        The solver colours live frames itself, over the whole domain; the
        tracker measures the near-body percentile in each frame and the result
        goes back as a fixed range: through the shared ``ViewSettings`` for a
        run, through ``post`` for explore mode.
        """
        try:
            rng = self._range.update(frame)
        except (AttributeError, TypeError, ValueError):
            rng = None
        v = self.jobs.view
        if rng is None:
            if getattr(v, "range_mode", "auto") == "fixed" and getattr(frame, "field", "") not in \
                    LiveRangeTracker.TRACKED:
                self._reset_live_range()
            return
        lo, hi = rng
        v.range_mode, v.vmin, v.vmax = "fixed", lo, hi
        if self.live.is_running():
            self.live.post(range_mode="fixed", vmin=lo, vmax=hi)

    def _show_frame(self, frame) -> None:
        self.monitor_note.setText("Latest time step")
        self._update_monitor_note()
        self._track_range(frame)
        self.view.set_frame(frame)
        self._sync_cmap_for_smoke(frame)
        t = getattr(frame, "t", float("nan"))
        step = getattr(frame, "step", 0)
        axis = getattr(frame, "slice_axis", None)
        where = ""
        base = f"t = {sig(t, 3)} L/U · step {step:,}"
        if axis:
            pos = getattr(frame, "slice_pos", None)
            where = f" · {axis} = {sig(pos, 3)} m".replace("-", MINUS) if pos is not None else ""
            self._slice_where = where      # the 3D caption names the slice only while one is drawn
            self._sync_slice_slider(axis, pos)
            self._frame_to_3d(frame, base)
        self.view.set_caption(base + where)
        self._orient_split()                 # the first frame of a run gives the slice its shape
        if self._closed_mon:
            self._closed_monitor_frame(frame)
            return
        coeffs = getattr(frame, "coeffs", {}) or {}
        cd, cl = coeffs.get("cd", float("nan")), coeffs.get("cl", float("nan"))
        self.coef_values["cd"].setText(fmt_num(cd, 4))
        self.coef_values["cl"].setText(fmt_num(cl, 4))
        self.coef_values["cs"].setText(fmt_num(coeffs.get("cs", float("nan")), 4))
        self.coef_values["ld"].setText(fmt_num(_ld(cl, cd), 3))
        self.history.set_history(getattr(frame, "history", None))

    # ------------------------------------------------------------ view opts
    #: Tooltip of the Colormap combo while the smoke is a multi-colour composite.
    _SMOKE_COLOURS_TIP = ("Each smoke channel is drawn in its emitter's colour (see the legend); a colour map "
                          "applies to a single field")

    def _sync_cmap_for_smoke(self, frame) -> None:
        """A multi-colour smoke picture (2D, several emitter colours) takes no colour map: the combo is
        disabled and says why, instead of offering "Greyscale" for a picture in orange and white."""
        if self._is3d():
            return
        composite = dye_legend(frame) is not None
        if composite == (not self.cmap_combo.isEnabled() and
                         self.cmap_combo.toolTip() == self._SMOKE_COLOURS_TIP):
            return
        self.cmap_combo.setEnabled(not composite)
        self.cmap_combo.setToolTip(self._SMOKE_COLOURS_TIP if composite else self._cmap_tip)

    def _on_view_changed(self, *_) -> None:
        v = self.jobs.view
        if v.field != self.field_combo.currentData():
            self._reset_live_range()   # a speed range must not colour the first vorticity frame
        v.field = self.field_combo.currentData()
        v.cmap = self.cmap_combo.currentData()
        v.particles = self.particles_btn.isChecked() and not self._is3d()
        # 3D: one Streamlines toggle (the slice strip's) draws them in the 3D view and in the 2D slice
        v.streamlines = self.flow_toggles.wanted("streamlines") if self._is3d() else self.streamlines_btn.isChecked()
        self.view.show_particles = v.particles
        self.view.show_streamlines = v.streamlines
        self.view.update()
        if self.live.is_running():
            self.live.post(field=v.field, cmap=v.cmap, particles=v.particles, streamlines=v.streamlines)
            return
        sim = self.state.simulation
        if (sim is not None and not self.jobs.running_id() and self._shown_sim_id == sim.id
                and sim.status in ("done", "cancelled") and sim.analysis not in ("polar", "wing")):
            self._show_last_result(sim)   # the stored result follows the field / colour choice too

    # -------------------------------------------------------------- 3D view
    def _ensure_view3d(self) -> None:
        """Build the 3D view the first time a 3D simulation is shown (it owns an OpenGL context)."""
        if self.view3d is not None:
            return
        from ..widgets.view3d import View3D
        self.view3d = View3D(overlay_toolbar=True)
        self.view3d.set_placeholder("No geometry", "")
        self.view3d_host.layout().addWidget(self.view3d)
        # layers without a toggle (the device glyphs) keep their live default; the toggles decide the rest
        shown = dict(DEFAULT_SHOWN_LIVE)
        shown.update({k: self.flow_toggles.wanted(k) for k in self.flow_toggles.buttons})
        self.flow = FlowScene(self.view3d, shown=shown)
        # the rake dragged by its handle in the view: the rows follow it, the release counts as an edit
        self.flow.rakeMoving.connect(self.rake_editor.show_rake)
        self.flow.rakeEdited.connect(self._on_rake_dragged)

    def _view3d_shown(self) -> bool:
        """Whether the 3D view is on screen (3D runs, View 3D or Split): 3D work is skipped otherwise."""
        return self.flow is not None and self._is3d() and self.view_mode.current() in ("3d", "split")

    def _apply_view_mode(self) -> None:
        three_d = self._is3d()
        mode = self.view_mode.current() if three_d else "slice"
        show3d = three_d and mode in ("3d", "split")
        self.view3d_host.setVisible(show3d)
        self.view.setVisible(mode in ("slice", "split"))
        self.flow_toggles.set_shown_keys([k for k in self.flow_toggles.buttons
                                          if show3d and k not in self._flow_absent])
        self._layer_strip.setVisible(show3d)
        self._sync_slice_controls()
        if self.fit_btn is not None:            # the 2D view's whole-domain toggle (the 3D view has its own Fit)
            self.fit_btn.setVisible(mode in ("slice", "split"))
        if mode == "split":
            self._orient_split(force=True)
        if three_d:
            self._push_flow_settings()

    def _slice_shown(self) -> bool:
        """Whether a slice is on screen: the 2D slice view (Slice, Split) or the 3D view's slice plane."""
        if not self._is3d():
            return True
        mode = self.view_mode.current()
        return mode in ("slice", "split") or self.flow_toggles.wanted("slice")

    def _colours_used(self) -> bool:
        """Whether Field and Colormap colour anything on screen (a slice, or streamlines)."""
        return self._slice_shown() or self.flow_toggles.wanted("streamlines")

    #: Tooltips of the Field / Colormap combos when nothing on screen takes their colours.
    _COLOURS_IDLE_TIP = ("Colours the slice plane and the streamlines; switch one of them on to choose. "
                         "Smoke is drawn in grey.")

    def _sync_slice_controls(self) -> None:
        """Show the slice controls only while a slice is on screen, and enable Field / Colormap only while
        something takes their colours. Smoke alone is grey: a live "Side · y" control, a "0 m" slider
        and an enabled Speed / Viridis pair read as if the view were sliced and coloured when it was not."""
        three_d = self._is3d()
        sliced = self._slice_shown()
        self._slice_row.setVisible(three_d and sliced)
        for w in (self.slice_axis, self.slice_slider, self.slice_value, self.slice_reset):
            w.setEnabled(sliced)
        used = (not three_d) or self._colours_used()
        for combo, tip in ((self.field_combo, "Field shown on the slice and the streamlines"),
                           (self.cmap_combo, self._cmap_tip)):
            combo.setEnabled(used)
            combo.setToolTip(tip if used else self._COLOURS_IDLE_TIP)
        self._reflow_slice_strip()
        self._refresh_caption3d()

    def _orient_split(self, *, force: bool = False) -> None:
        """Split view: the slice beside the 3D view, or under it when that frames the slice better.

        A side slice of a car (about 15 x 5 m) beside the 3D view filled a
        90-140 px band of a portrait pane with blank space above and below; the
        orientation whose pane aspect is closer (in log ratio) to the slice's
        framed window is chosen (:func:`split_orientation`).
        """
        if not self._is3d() or self.view_mode.current() != "split":
            return
        w, h = max(self.view_split.width(), 2), max(self.view_split.height(), 2)
        want = split_orientation(w, h, self.view.framing_aspect())
        if want != self.view_split.orientation() or force:
            self.view_split.setOrientation(want)
            n = h if want == Qt.Orientation.Vertical else w
            self.view_split.setSizes([int(0.55 * n), n - int(0.55 * n)])

    def _live_on_screen(self) -> bool:
        """A run (or explore) is what the viewport shows, rather than a stored result or a preview."""
        rid = self.jobs.running_id()
        return self.live.is_running() or bool(rid and rid == self._shown_sim_id)

    def _refresh_caption3d(self) -> None:
        """The 3D view's caption: time and step, plus the slice position only when a slice is drawn.

        A stored result says what is on screen: "Last run", then "time-averaged
        flow" only while streamlines or the slice show it and "smoke at the end
        of the run" only while the smoke does (smoke alone used to be captioned
        "Time-averaged", which it is not).
        """
        if self.flow is None:
            return
        if self._result_on_3d:
            parts = ["Stopped run" if self._result_on_3d == "stopped" else "Last run"]
            if self.flow.is_shown("streamlines") or self.flow.is_shown("slice"):
                parts.append("time-averaged flow")
            if self.flow.is_shown("smoke"):
                parts.append("smoke at the end of the run")
            if self.flow.is_shown("heat"):
                label_ = getattr(self.flow.heat, "label", "mean")
                parts.append("time-averaged temperature" if label_ == "mean" else "temperature at the end of the run")
            self.flow.set_caption(" · ".join(parts))
            return
        if not self._last_caption3d or not self._live_on_screen():
            return
        where = self._slice_where if self.flow_toggles.wanted("slice") else ""
        self.flow.set_caption(self._last_caption3d + where)

    def _on_view_mode(self, _key: str) -> None:
        self._apply_view_mode()
        if not self._view3d_shown():
            return
        # the 3D view was not updated while hidden: catch up with the newest frame or the stored result
        live = self.live.is_running() or bool(self.jobs.running_id() and self.jobs.running_id() == self._shown_sim_id)
        if live and self._last_frame3d is not None:
            self._flow_at = dict.fromkeys(self._flow_at, 0.0)
            self._frame_to_3d(self._last_frame3d, self._last_caption3d)
        elif not live:
            self._redraw_slice_static()

    def _on_flow_toggle(self, key: str, on: bool) -> None:
        if self.flow is not None:
            self.flow.show(key, on)
        if key == "streamlines":
            self._on_view_changed()          # the 2D slice's in-plane streamlines follow the same toggle
        if key in ("smoke", "streamlines", "heat", "particles"):
            self._flow_at["streamlines"] = 0.0
            self._apply_live_volume(force=True)
        self._sync_slice_controls()          # slice row, Field / Colormap and the caption follow the layers
        self._push_flow_settings()

    def _push_flow_settings(self) -> None:
        """Tell the solver what the 3D view needs: ``ViewSettings.view3d``, ``smoke`` and ``streamlines3d``.

        The settings object is shared with the running job and the explore
        session (both read it every frame), so plain attributes reach them; no
        ``post`` is needed, and a solver that predates these names ignores them.
        The rake (``seed_rake``) is not written here: it belongs to one
        simulation (see :meth:`_on_rake_edited`).
        """
        # temperature3d / particles3d (DESIGN.md 10, real ViewSettings fields since v0.3): the 3D view wants
        # the temperature in its volumes and the tracer positions in its frames
        values = {"view3d": self._view3d_shown(), "smoke": self.flow_toggles.wanted("smoke"),
                  "streamlines3d": self.flow_toggles.wanted("streamlines"),
                  "temperature3d": self._view3d_shown() and self.flow_toggles.wanted("heat")
                  and "heat" not in self._flow_absent,
                  "particles3d": self._view3d_shown() and self.flow_toggles.wanted("particles")
                  and "particles" not in self._flow_absent}
        for k, val in values.items():
            self._set_view(k, val)

    def _set_view(self, key: str, value) -> None:
        try:
            setattr(self.jobs.view, key, value)
        except (AttributeError, TypeError):          # a settings object that refuses new attributes
            logging.getLogger(__name__).debug("view settings refuse %s", key)

    def _cell_size(self) -> float | None:
        """The solver's cell size [m] of the plan on screen (it sets the default smoke nozzle radius)."""
        plan = self._current_plan3d()
        try:
            h = float(plan.solver_config.h) * float(plan.length_ref)
        except (AttributeError, TypeError, ValueError):
            return None
        return h if math.isfinite(h) and h > 0 else None

    def _apply_setup_smoke(self, sim) -> None:
        """A newly shown simulation: the Smoke toggle starts as Setup ▸ Flow visualisation says."""
        fv = setup_flow_viz(sim)
        if fv is None or "smoke" not in self.flow_toggles.buttons:
            return
        if self.flow_toggles.wanted("smoke") != bool(fv["smoke"]):
            self.flow_toggles.buttons["smoke"].setChecked(bool(fv["smoke"]))     # through _on_flow_toggle

    def _sync_flow_toggles(self, context: str) -> None:
        """Enable the flow toggles that have data; a disabled one says why ("live", "result" or "preview")."""
        if self.flow is None:
            return
        expected = context in ("live", "preview")       # a run (or explore) will release smoke from the rake
        if self.flow.smoke_expected != expected:
            self.flow.smoke_expected = expected
            self.flow.rebuild("rake")
        vol = self._live_volume
        surface = "surface pressure is stored with the result when the run ends"
        if context == "live":
            have = {"smoke": vol is not None and bool(smoke_channels(vol)),
                    "streamlines": has_velocity(vol) or self._live_lines is not None,
                    "heat": self.flow.heat is not None or (vol is not None and bool(temperature_field_of(vol))),
                    "particles": self.flow.particles is not None}
            why = {"smoke": "the solver has not sent a smoke volume yet",
                   "streamlines": "the solver has not sent a velocity volume yet", "surface": surface,
                   "heat": "the solver has not sent a temperature volume yet",
                   "particles": "the solver has not sent tracer particles yet"}
        elif context == "result":
            have = {}
            why = {"smoke": "this run stored no smoke snapshot", "streamlines": "this run stored no 3D velocity volume",
                   "surface": "this run stored no surface pressure", "heat": "this run stored no temperature field",
                   "particles": "a stored run keeps no tracer particles (Results replays them)"}
        else:
            have = {"smoke": False, "streamlines": False, "heat": False, "particles": False}
            why = {"smoke": "smoke is released from the rake once the run starts",
                   "streamlines": "streamlines are traced once the run sends its flow", "surface": surface,
                   "heat": "the plume is drawn once the run sends its temperature field",
                   "particles": "particles are released once the run starts"}
        for key in self.flow_toggles.buttons:
            ok = have[key] if key in have else self.flow.available(key)[0]
            self.flow_toggles.set_available(key, ok, why.get(key, "no slice yet"))

    def _render_ms(self) -> float:
        r = getattr(self.view3d, "last_render_ms", float("nan")) if self.view3d is not None else float("nan")
        return r if math.isfinite(r) else 0.0

    def _flow_interval(self, key: str, base: float) -> float:
        """Shortest wait [s] before the next rebuild of a 3D layer: its base interval, stretched so the
        layer's measured cost (plus one render) stays under FLOW_DUTY of the GUI thread's time."""
        cost = self.flow.cost_ms.get(key, 0.0) if self.flow is not None else 0.0
        return max(base, (cost + self._render_ms()) / 1000.0 / FLOW_DUTY)

    def _plane_window(self, axis: str) -> tuple[float, float, float, float] | None:
        """(col0, col1, row0, row1) of the slice drawn in 3D: the bodies' near field, not the whole domain."""
        if self.flow is None:
            return None
        return plane_window(near_field_box(self.flow.body_bounds(), ground=self.flow.ground), axis)

    def _bodies_to_3d(self, meshes, key, sim, *, domain=None) -> None:
        """Bodies of the 3D view (wind frame); the rake follows them (the user's own rake is kept)."""
        if self.flow is None:
            return
        ground = ground_of(sim) if sim is not None else None
        meshes = [m for m in meshes if len(getattr(m, "faces", []))]
        focus = flow_focus_box(bounds_of(meshes), ground=ground)
        room = closed_box3(sim, domain)
        if room is not None:
            # a room is about the whole box (its fans, heaters, vents and smoke), not the block inside:
            # the near-body framing opened the heated room on the inner block and a huge fan disc
            focus = room
        changed = self.flow.set_data(bodies=meshes, body_key=key, ground=ground, focus=focus)
        if hasattr(self.flow, "set_room"):
            self.flow.set_room(room)          # the walls, as Setup draws them (None for a tunnel)
        if room is not None and self.view3d is not None and self._framed_room != (sim.id, tuple(room)):
            self._framed_room = (sim.id, tuple(room))
            self.view3d.set_view("iso")
        sid = sim.id if sim is not None else ""
        # the rake is placed once per simulation (or again when the domain becomes known): an
        # attitude change in explore or the switch to the stored result must not move it, nor
        # the smoke nozzles the solver places from it
        if self.rake_editor.rake() is None or sid != self._rake_sim or (changed and domain is not None
                                                                         and self._rake_box is None):
            new_sim = sid != self._rake_sim
            self._rake_sim, self._rake_box = sid, domain
            self._set_rake_for(sid, self.flow.body_bounds(), domain)
            if new_sim and sim is not None:
                self._apply_setup_smoke(sim)

    def _set_rake_for(self, sim_id: str, bounds, domain) -> None:
        """The rake of a simulation: the one the user placed, else Setup's emitter placed for its bodies.

        Setup ▸ Flow visualisation chooses the emitter (rake, vertical rake,
        grid, plane band; heights; nozzle counts) and is resolved here by the
        solver's own rule (:meth:`RakeSpec.resolve`), so the rake drawn before
        the run is where the run releases its smoke.
        """
        if bounds is None:
            self.rake_editor.set_rake(None)
            return
        x0, x1, y0, y1, z0, z1 = bounds
        size = max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
        box = tuple(domain) if domain is not None else (x0 - 2 * size, x1 + 3 * size, y0 - size, y1 + size,
                                                        z0 - size, z1 + size)
        fv = setup_flow_viz(self._sim(sim_id) or self.state.simulation)
        default = RakeSpec.resolve(fv["rake"] if fv else None, bounds, domain=box, cell=self._cell_size())
        rake = self._rakes.get(sim_id, default).clamped(box)
        self.rake_editor.set_rake(rake, default=default, box=box)
        if self.flow is not None:
            self.flow.set_data(rake=rake)
        self._push_flow_settings()

    def _on_rake_dragged(self, spec: RakeSpec) -> None:
        self.rake_editor.show_rake(spec)
        self._on_rake_edited()

    def _on_rake_edited(self) -> None:
        """The user moved the rake (rows, drag or Reset): the simulation keeps it, and a solver running it
        (a run, or explore) moves its smoke nozzles there at once."""
        rake = self.rake_editor.rake()
        sid = self._rake_sim or self._shown_sim_id or self.state.simulation_id
        reset = self.rake_editor.is_default()
        if sid:
            if rake is None or reset:
                self._rakes.pop(sid, None)
            else:
                self._rakes[sid] = rake
            register = getattr(self.jobs, "set_seed_rake", None)
            if callable(register):
                # back on its default the run follows Setup's emitter again (resolved by the solver)
                register(sid, None if rake is None or reset else rake.to_view())
            if self.live.is_running() and self._live_sim_id == sid:
                self._set_view("seed_rake", rake.to_view() if rake is not None else None)
        if self.flow is not None:
            self.flow.set_data(rake=rake)
        self._push_flow_settings()

    def _explore_bodies_to_3d(self) -> bool:
        """Explore: the body at the sliders' attitude (the session turns its own bodies the same way,
        within the same limits), so the 3D view follows a pitch or yaw change at once."""
        if not (self.live.is_running() and self._is3d() and self.flow is not None):
            return False
        plan = getattr(self.live, "plan", lambda: None)()
        meshes = list(getattr(plan, "meshes_design", []) or [])
        p3 = lazy_import("aerolume.simulation.plan3d")
        fn = getattr(p3, "wind_frame_meshes", None) if p3 is not None else None
        if not meshes or fn is None:
            return False
        ground = bool(getattr(plan, "ground", False))
        lim_a = getattr(p3, "LIVE_PITCH_GROUND3" if ground else "LIVE_PITCH3", 90.0)
        lim_y = getattr(p3, "LIVE_YAW3", 90.0)
        a = max(-lim_a, min(lim_a, self.slider_alpha()))
        y = max(-lim_y, min(lim_y, self.slider_yaw()))
        try:
            wind = fn(meshes, a, y, tuple(plan.moment_ref3), ground=ground)
        except Exception:
            logging.getLogger(__name__).debug("explore attitude in 3D failed", exc_info=True)
            return False
        self._bodies_to_3d(wind, ("explore", id(plan), round(a, 4), round(y, 4)), self.state.simulation,
                           domain=getattr(plan, "extent3_full", None))
        return True

    def _plan_to_3d(self, plan, sim) -> None:
        """A 3D plan's wind-frame bodies (the attitude the solver uses) in the 3D view."""
        meshes = list(getattr(plan, "bodies_wind3", []) or [])
        if meshes:
            self._bodies_to_3d(meshes, ("plan", id(plan)), sim, domain=getattr(plan, "extent3_full", None))

    def _preview3d(self, sim, design, plan, caption: str) -> None:
        """A 3D simulation before its first frame: the body, the rake and a grey plane where the slice will be."""
        if self.flow is None:
            return
        if plan is not None and getattr(plan, "bodies_wind3", None):
            self._plan_to_3d(plan, sim)
        elif design is not None:
            self._bodies_to_3d(list(getattr(design, "meshes", []) or []),
                               ("design", id(design), _mesh_digest(design)), sim)
        if not self.flow.bodies and self.view3d is not None:
            # a wing is lofted by the planner: nothing to draw until the plan arrives
            self.view3d.set_placeholder(caption, "Planning the 3D grid…" if plan is None else "")
        axis, pos = self.slice_axis.current(), self.slice_position()
        win = self._plane_window(axis)
        plane = neutral_plane(axis, pos, win) if win is not None and pos is not None else None
        self._result_on_3d = ""
        self._sync_physics3d(sim, design)
        self._heat_live_rng = None
        self.flow.set_heat(None)
        self.flow.set_particles(None)
        self.flow.set_data(volume=None, smoke=None, lines=None, slice=plane, neutral=True)
        self.flow.set_caption(caption)
        self._sync_flow_toggles("preview")

    def _frame_to_3d(self, frame, caption: str = "") -> None:
        """A live 3D frame in the 3D view: the slice plane (throttled) and, when the frame carries a
        volume (``LiveFrame.volume``), the smoke and streamlines (see :meth:`_apply_live_volume`)."""
        self._last_frame3d = frame
        self._result_on_3d = ""                  # a live frame replaces a stored result on screen
        vol = getattr(frame, "volume", None)
        serial = getattr(frame, "volume_serial", None)
        if vol is not None and (serial is None or serial != self._live_serial or self._live_volume is None):
            self._live_volume, self._live_serial = vol, serial
        lines = getattr(frame, "streamlines3d", None)
        if lines is not None:
            self._live_lines = lines
        if caption:
            self._last_caption3d = caption
        if not self._view3d_shown():
            return
        done = getattr(frame, "phase", "") == "done"
        wait = self._tick_s - (time.monotonic() - self._flow_at["tick"])
        if done or wait <= 0:
            self._flow_timer.stop()
            self._tick3d(frame, force=done)
        elif not self._flow_timer.isActive():
            self._flow_timer.start(int(1000 * wait) + 1)      # the newest frame then, not this one

    def _deferred_tick(self) -> None:
        if self._last_frame3d is not None and self._view3d_shown():
            self._tick3d(self._last_frame3d)

    def _tick3d(self, frame, *, force: bool = False) -> None:
        """One live 3D update: the plan's bodies, the slice plane, the newest smoke and streamlines and the
        caption, so they cost one render together; the next tick is spaced by what this one cost."""
        t0 = time.perf_counter()
        if not self._explore_bodies_to_3d():
            plan = self._current_plan3d()
            sim = self._sim(self._shown_sim_id) or self.state.simulation
            if plan is not None:
                self._plan_to_3d(plan, sim)
            self._sync_physics3d(sim, self._design_of(sim))
        else:                                # explore: the devices turn with the sliders, as the body does
            sim = getattr(self, "_live_sim", None) or self.state.simulation
            self._sync_physics3d(sim, self._design_of(self.state.simulation), alpha=self.slider_alpha(),
                                 yaw=self.slider_yaw())
        sd = frame_to_slice(frame)
        if sd is not None:
            field = next(iter(sd.fields))
            self.flow.set_data(slice=crop_slice(sd, self._plane_window(sd.axis)), field=field,
                               cmap=getattr(frame, "cmap", None) or default_cmap3d(field),
                               rng=(float(frame.vmin), float(frame.vmax)))
        self._apply_live_volume(force=True, trace_force=force)
        self._refresh_caption3d()
        cost_ms = 1000.0 * (time.perf_counter() - t0) + self._render_ms()
        self._tick_s = max(LIVE3D_INTERVAL_S, cost_ms / 1000.0 / FLOW_DUTY)
        self._flow_at["tick"] = time.monotonic()

    def _apply_live_volume(self, force: bool = False, *, trace_force: bool = False) -> None:
        """Draw the newest live flow data (a tick calls this; so does a toggle, at once).

        Smoke and streamlines the run traced on its own thread
        (``LiveFrame.streamlines3d``) are only uploaded. Streamlines traced here
        from a velocity volume (a run that sends velocity but no lines) are the
        expensive case and keep their own spacing (STREAM_INTERVAL_S, stretched
        by their cost) unless ``trace_force``.
        """
        vol, lines = self._live_volume, self._live_lines
        live = self.live.is_running() or bool(self.jobs.running_id() and self.jobs.running_id() == self._shown_sim_id)
        if (vol is None and lines is None) or self.flow is None or not self._view3d_shown() or not live:
            return
        changes: dict = {}
        smoke = vol if vol is not None and smoke_channels(vol) else None
        if smoke is not None and smoke is not self.flow.smoke and self.flow_toggles.wanted("smoke") and force:
            changes["smoke"] = smoke
        # the thermal plume (a volume with temperature) and the tracer particles of the newest frame
        if vol is not None and temperature_field_of(vol) and vol is not self.flow.heat and force \
                and self.flow_toggles.wanted("heat"):
            self.flow.set_heat(vol, rng=self._live_heat_range(vol))
        frame = self._last_frame3d
        if frame is not None and self.flow_toggles.wanted("particles") and force:
            pts = self._frame_particles3d(frame)
            if pts is not None or self.flow.particles is not None:
                self.flow.set_particles(pts, colors=particle_colors(frame) if pts is not None else None,
                                        trails=getattr(frame, "particle_trails", None),
                                        mirror_y=bool(getattr(vol, "mirrored", False)))
        if self.flow_toggles.wanted("streamlines"):
            if lines is not None:
                if lines is not self.flow.lines:
                    changes["lines"] = lines
            elif has_velocity(vol) and vol is not self.flow.volume:
                now = time.monotonic()
                left = self._flow_interval("streamlines", STREAM_INTERVAL_S) - (now - self._flow_at["streamlines"])
                if trace_force or left <= 0:
                    changes["volume"] = vol
                    self._flow_at["streamlines"] = now
        if lines is not None and vol is not None and "volume" not in changes and self.flow.volume is not vol:
            changes["volume"] = vol            # kept for the toggles; the lines came traced
        if changes:
            self.flow.set_data(**changes)
        self._sync_flow_toggles("live")

    # -- placed physics in the 3D view (DESIGN.md 10) ---------------------------
    def _sync_physics3d(self, sim, design=None, *, alpha: float | None = None, yaw: float | None = None) -> None:
        """What the 3D view shows of a simulation's placed physics: the Thermal plume and Particles toggles
        (only for runs that can have them), the smoke channels' colours, and the device glyphs at the
        attitude the view is drawn at (explore passes its sliders). Glyphs are rebuilt only when the
        devices or the attitude changed, so a live tick costs nothing here."""
        if self.flow is None or sim is None:
            return
        thermal = bool(getattr(getattr(sim, "thermal", None), "enabled", False))
        particles = any(getattr(e, "species", "") == "particles" and getattr(e, "enabled", True)
                        for e in (getattr(sim, "emitters", None) or []))
        absent = {k for k, have in (("heat", thermal), ("particles", particles)) if not have}
        if absent != self._flow_absent:
            self._flow_absent = absent
            show3d = self._is3d() and self.view_mode.current() in ("3d", "split")
            self.flow_toggles.set_shown_keys([k for k in self.flow_toggles.buttons if show3d and k not in absent])
            self._reflow_slice_strip()
            self._push_flow_settings()          # the solver hears whether the view wants temperature / particles
        self.flow.set_dye_colors(dye_colours_of(sim))
        try:
            digest = json.dumps([[x.to_dict() for x in getattr(sim, a, None) or []]
                                 for a in ("emitters", "fans", "heat_sources", "ports")], sort_keys=True, default=str)
        except (AttributeError, TypeError):
            digest = str(id(sim))
        key = (digest, None if alpha is None else round(float(alpha), 3), None if yaw is None else round(float(yaw), 3),
               id(design), round(float(self.flow.size), 6))
        if key == self._devices_key:
            return
        self._devices_key = key
        placed = lazy_import("aerolume.viz3d.placed")
        glyphs = []
        if placed is not None:
            try:
                glyphs = placed.placed_glyphs(sim, design, alpha_deg=alpha, yaw_deg=yaw, scale=self.flow.size)
            except Exception:
                logging.getLogger(__name__).exception("3D device glyphs failed")
        self.flow.set_devices(glyphs)

    def _live_heat_range(self, vol) -> tuple[float, float] | None:
        """A fixed colour range for a live plume, so its colours do not drift as it develops.

        The case's own temperatures when it has them (ambient, heated bodies,
        hot walls and jets: the physical bounds of the field, which stay put),
        else the data's :func:`heat_display_range`, only ever widened during the
        run. "theta" frames (non-dimensional) always use the widening data range.
        """
        f = temperature_field_of(vol)
        sim = self._live_sim if self.live.is_running() and getattr(self, "_live_sim", None) is not None \
            else (self._sim(self._shown_sim_id) or self.state.simulation)
        if f == "temperature" and sim is not None and getattr(sim.thermal, "enabled", False):
            design = self._design_of(sim)
            temps = [float(sim.thermal.t_ambient)]
            items = list(getattr(design, "meshes", None) or []) + list(getattr(design, "bodies", None) or [])
            temps += [float(m.thermal_value) for m in items
                      if getattr(m, "thermal", "") == "temperature" and m.thermal_value is not None]
            temps += [float(w.value) for w in sim.thermal.walls if w.mode == "temperature"]
            temps += [float(p.temperature) for p in sim.ports if p.enabled and p.temperature is not None]
            if max(temps) - min(temps) > 1e-6:
                return min(temps), max(temps)
        vals = np.asarray(vol.fields[f], dtype=float)
        if getattr(vol, "mask", None) is not None:
            vals = vals[np.asarray(vol.mask) <= 0.5]
        lo, hi, _mode = heat_display_range(vals)
        if self._heat_live_rng is not None:
            lo, hi = min(lo, self._heat_live_rng[0]), max(hi, self._heat_live_rng[1])
        self._heat_live_rng = (lo, hi)
        return lo, hi

    @staticmethod
    def _frame_particles3d(frame):
        """(N, 3) tracer positions of a 3D frame (``particles3d``, or ``particles`` with three columns)."""
        for name in ("particles3d", "particles"):
            p = getattr(frame, name, None)
            if p is None:
                continue
            a = np.asarray(p, dtype=float)
            if a.ndim == 2 and a.shape[1] == 3:
                return a
        return None

    def _result_to_3d(self, sim, res: CFD3DResult, sd, field: str, cmap: str | None, rng) -> None:
        """A finished 3D run in the 3D view: its surfaces, mean-flow streamlines, smoke snapshot and mean slice."""
        if self.flow is None:
            return
        self.flow.set_data(lines=None)
        self._bodies_to_3d(res.surface, ("result", id(res)), sim, domain=volume_box(res.volume))
        vol = res.volume if has_velocity(res.volume) else None
        plane = crop_slice(sd, self._plane_window(sd.axis)) if sd is not None else None
        self._sync_physics3d(sim, self._design_of(sim))
        heat = heat_volume_of(res)
        self.flow.set_heat(heat, rng=None)
        self.flow.set_particles(None)
        self.flow.set_data(volume=vol, smoke=smoke_volume_of(res), slice=plane, field=field, cmap=cmap, rng=rng)
        self._result_on_3d = "stopped" if sim.status == "cancelled" else "last"
        self._refresh_caption3d()
        self._sync_flow_toggles("result")

    def _current_plan3d(self):
        """The 3D plan describing what the viewport shows: explore's, the running job's or the cached one."""
        if self.live.is_running():
            plan = getattr(self.live, "plan", lambda: None)()
            if getattr(plan, "dimension", 2) == 3:
                return plan
        rid = self.jobs.running_id()
        if rid and rid == self._shown_sim_id:
            plan = self._run_plan(rid)
            if getattr(plan, "dimension", 2) == 3:
                return plan
        sim = self.state.simulation
        design = self._design_of(sim)
        if sim is None or design is None:
            return None
        plan = self._plans.get(plan_key(sim, design))
        return plan if getattr(plan, "dimension", 2) == 3 else None

    def _slice_range(self, axis: str) -> tuple[float, float] | None:
        """[lo, hi] metres the slice position can take along ``axis``: the domain, or the stored volume."""
        k = {"x": 0, "y": 1, "z": 2}[axis]
        plan = self._current_plan3d()
        if plan is not None:
            ex = plan.extent3_full
            return float(ex[2 * k]), float(ex[2 * k + 1])
        sim = self.state.simulation
        res = self.jobs.cached_result(sim.id) if sim is not None and hasattr(self.jobs, "cached_result") else None
        if isinstance(res, CFD3DResult) and res.volume is not None:
            from ..widgets.results3d import volume_bounds
            b = volume_bounds(res.volume)
            return float(b[2 * k]), float(b[2 * k + 1])
        design = self._design_of(sim)
        if design is not None and getattr(design, "meshes", None):
            b = design.bounds3()
            size = max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6)
            return float(b[2 * k] - size), float(b[2 * k + 1] + size)
        return None

    def _default_slice_pos(self, axis: str) -> float | None:
        plan = self._current_plan3d()
        if plan is not None and axis in getattr(plan, "slice_positions", {}):
            return float(plan.slice_positions[axis])
        sim = self.state.simulation
        res = self.jobs.cached_result(sim.id) if sim is not None and hasattr(self.jobs, "cached_result") else None
        if isinstance(res, CFD3DResult):
            for s in res.slices:
                if s.axis == axis:
                    return float(s.position)
        return None

    def slice_position(self) -> float | None:
        """The slice position the strip shows [m] (None before the range is known)."""
        rng = self._slice_range(self.slice_axis.current())
        if rng is None:
            return None
        lo, hi = rng
        return lo + (hi - lo) * self.slice_slider.value() / 1000.0

    def _set_slider_to(self, pos: float | None) -> None:
        axis = self.slice_axis.current()
        rng = self._slice_range(axis)
        self._slice_lock = True
        try:
            if rng is None or pos is None:
                self.slice_slider.setValue(500)
            else:
                lo, hi = rng
                self.slice_slider.setValue(round(1000 * (pos - lo) / (hi - lo)) if hi > lo else 500)
        finally:
            self._slice_lock = False
        p = self.slice_position()
        self._set_slice_value(axis, p)

    def _reset_slice_controls(self) -> None:
        """Back to the shared view's slice (axis kept; position = the plan's default unless set)."""
        v = self.jobs.view
        axis = str(getattr(v, "slice_axis", "y") or "y")
        self.slice_axis.blockSignals(True)
        self.slice_axis.set_current(axis if axis in ("x", "y", "z") else "y")
        self.slice_axis.blockSignals(False)
        self._update_slice_range()

    def _update_slice_range(self) -> None:
        """Re-place the slider after the range became known (a plan arrived) or changed."""
        v = self.jobs.view
        pos = getattr(v, "slice_pos", None)
        axis = self.slice_axis.current()
        self._set_slider_to(pos if pos is not None else self._default_slice_pos(axis))

    def _sync_slice_slider(self, axis: str, pos: float | None) -> None:
        """A frame reports where it was cut: the strip follows (unless the user is dragging it)."""
        if self.slice_slider.isSliderDown():
            return
        if axis != self.slice_axis.current():
            self.slice_axis.blockSignals(True)
            self.slice_axis.set_current(axis)
            self.slice_axis.blockSignals(False)
        if pos is not None:
            self._set_slider_to(float(pos))

    def _on_slice_axis(self, axis: str) -> None:
        """A new slice normal: the default position through the body; the solver cuts there next frame."""
        v = self.jobs.view
        v.slice_axis = axis
        v.slice_pos = None
        self._set_slider_to(self._default_slice_pos(axis))
        if self.live.is_running():
            self.live.post(slice_axis=axis, slice_pos=None)
        self._flow_at["tick"] = 0.0
        self._redraw_slice_static()

    def _on_slice_pos(self, *_) -> None:
        if self._slice_lock:
            return
        pos = self.slice_position()
        axis = self.slice_axis.current()
        self._set_slice_value(axis, pos)
        if pos is None:
            return
        v = self.jobs.view
        v.slice_axis, v.slice_pos = axis, float(pos)
        if self.live.is_running():
            self.live.post(slice_axis=axis, slice_pos=float(pos))
        self._redraw_slice_static()

    def _redraw_slice_static(self) -> None:
        """No frames are coming (preview or a finished run): redraw the slice from what is stored."""
        if not self._is3d() or self.live.is_running():
            return
        sim = self.state.simulation
        if sim is None or (self.jobs.running_id() and self.jobs.running_id() == self._shown_sim_id):
            return
        if sim.status in ("done", "cancelled") and self._show_last_result(sim):
            return
        self._show_slice_preview(sim, self._design_of(sim))

    def _plan_sections(self, plan, axis: str, pos: float | None) -> list:
        """Cross-sections of a plan's wind-frame meshes at the slice (cached per plan and position)."""
        if plan is None or pos is None:
            return []
        sim = self.state.simulation
        design = self._design_of(sim)
        key = plan_key(sim, design) if sim is not None and design is not None else ""
        cache = self._sections.setdefault(key, {})
        k = (axis, round(float(pos), 9))
        if k not in cache:
            try:
                cache[k] = mesh_sections(list(getattr(plan, "bodies_wind3", []) or []), axis, float(pos))
            except Exception:  # noqa: BLE001 - an outline must never break the page
                cache[k] = []
            while len(cache) > 24:
                cache.pop(next(iter(cache)))
        return cache[k]

    def _show_slice_preview(self, sim: Simulation | None, design, *, placeholder: tuple[str, str] | None = None) -> None:
        """A 3D simulation before its first frame: the body's cross-section and the plane in 3D."""
        if sim is None:
            return
        axis = self.slice_axis.current()
        pos = self.slice_position()
        plan = self._current_plan3d()
        self.view.clear(reset_view=False)
        self.view.set_axis_names(SLICE_AXIS_NAMES[axis])
        ground = ground_of(sim)
        self.view.set_ground(ground if axis in ("y", "x") else None)
        self.view.set_sections(self._plan_sections(plan, axis, pos))
        if placeholder is not None:
            self._set_placeholder(*placeholder)
        elif sim.status == "failed":
            self._set_placeholder("The last run failed", sim.error or "Details are in Messages.")
        elif plan is None:
            self._set_placeholder("Not run", "Planning the 3D grid…")
        else:
            self._set_placeholder("Not run", "Run (F5) starts the simulation.")
        if placeholder is not None:
            caption = placeholder[0]
        elif sim.status == "failed":
            caption = "The last run failed"
        else:
            caption = "Not run · Run (F5) starts the simulation" if plan is not None else "Not run · planning the 3D grid"
        self._preview3d(sim, design, plan, caption)

    def _show_last_result3d(self, sim: Simulation, res: CFD3DResult) -> bool:
        """A finished 3D run: its stored mean slice at the strip's axis / position, coefficients, history."""
        axis = self.slice_axis.current()
        if self.slice_position() is None or getattr(self.jobs.view, "slice_pos", None) is None:
            self._set_slider_to(self._default_slice_pos(axis))
        pos = self.slice_position()
        field = self.field_combo.currentData() or "speed"
        sd = None
        if res.volume is not None and (field in res.volume.fields or field == "speed"):
            data = lazy_import("aerolume.viz3d.data")
            if data is not None:
                try:
                    sd = data.slice_from_volume(res.volume, axis, pos, field)
                except (KeyError, ValueError):
                    sd = None
        if sd is None:
            sd = next((s for s in res.slices if s.axis == axis and field in s.fields), None)
        if sd is None:
            sd = next((s for s in res.slices if s.axis == axis), None)
            field = next((f for f in ("speed", "cp") if sd is not None and f in sd.fields), field)
        shown = False
        if sd is not None and field in sd.fields:
            vals = np.asarray(sd.fields[field], dtype=np.float32)
            sections = mesh_sections(res.surface, sd.axis, float(sd.position))
            self.view.set_axis_names(SLICE_AXIS_NAMES[sd.axis])
            self.view.set_ground(ground_of(sim) if sd.axis in ("y", "x") else None)
            self.view.set_sections(sections)
            plan = self._current_plan3d()
            win = plan.focus_window(sd.axis) if plan is not None else None
            vmin, vmax = display_range3d(field, vals, extent=sd.extent, mask=sd.mask, window=win)
            lab, unit, _div, default = field_info3d(field)
            cmap = self.cmap_combo.currentData() or default
            self.view.set_field(vals, sd.extent, field=field, cmap=cmap, vmin=vmin, vmax=vmax, mask=sd.mask,
                                label=lab, units=unit)
            self.view.set_caption(f"Time-averaged · last run · {sd.axis} = {sig(float(sd.position), 3)} m"
                                  .replace("-", MINUS))
            # the strip names the plane actually shown (a half domain's centre slice sits half a cell off 0)
            self._set_slice_value(sd.axis, float(sd.position))
            self._shown_sim_id = sim.id
            shown = True
            self._result_to_3d(sim, res, sd, field, cmap, (vmin, vmax))
        elif res.surface:
            self._shown_sim_id = sim.id
            self._result_to_3d(sim, res, None, field, None, None)
            shown = self._view3d_shown()
        self.monitor_note.setText("Mean of the last run")
        self._show_mean_coefficients(res)
        if self._mine(sim.id):
            self._done_rows(sim, res)
            self.sym_value.setText("Plane y = 0 (half domain)" if res.symmetry_plane else "None (full domain)")
        self.history.set_x_limits(None)
        self.history.set_history(res.history or None,
                                 (res.t_avg_start, res.t_end) if math.isfinite(res.t_avg_start) else None)
        return shown

    # --------------------------------------------------------- live editing
    def _explore2d(self) -> bool:
        """Explore runs a 2D session: drawing and dragging act on the running flow."""
        return self.live.is_running() and not self._is3d() and self._mode == "field"

    def _show_placements(self, sim, design, alpha: float | None = None, *, refit: bool = True) -> None:
        """Glyphs of the simulation's emitters, fans, heaters and ports in the 2D view (wind frame).

        The default framing takes them in (``refit``); a drag or a pitch change passes False so the
        view does not move under the mouse or with every slider step."""
        if sim is None or self._is3d() or self._mode != "field":
            self.view.set_placements([])
            self.move_tool.set_placements([])
            return
        pls = placements_of(sim, design, alpha)
        self.view.set_placements(pls, refit=refit)
        self.move_tool.set_placements(pls)

    def _supports(self, key: str) -> bool | None:
        fn = getattr(self.live, "supports", None)
        return fn(key) if callable(fn) else None

    #: The lists a placement drag can change, and the live key that carries each to the solver.
    PLACEMENT_KEYS = ("emitters", "fans", "heat_sources", "ports")

    def _sync_live_controls(self) -> None:
        """Enable explore's controls by what runs and what the running session supports.

        A control the session cannot honour stays visible but disabled, with
        the reason in its tooltip (frame skip and moving placements need the
        live solver's support; stepping, the viscosity and drawn 2D obstacles
        work with any session through the explore worker).
        """
        on = self.live.is_running()
        for w in (self.pause_btn, self.reset_btn, self.fps_spin):
            w.setEnabled(on)
        self.step_btn.setEnabled(on and self.pause_btn.isChecked())
        self.step_n.setEnabled(on and self.pause_btn.isChecked())
        spf = self._supports("steps_per_frame") if on else None
        self.spf_spin.setEnabled(bool(on and spf is not False))
        self.spf_spin.setToolTip("Fixed solver steps between frames (frame skip); Automatic fits as many as the target "
                                 "rate allows" if spf is not False else
                                 "This explore solver paces its steps itself (no fixed steps per frame)")
        for w in (self.re_follow, self.nu_edit, self.re_edit):
            w.setEnabled(on and self._mode in ("field", "field3d"))
        two_d = self._explore2d()
        self.draw_strip.setVisible(two_d)
        movable = any(self._supports(k) is not False for k in self.PLACEMENT_KEYS)
        has_items = bool(self.move_tool.placements)
        self.move_tool.enabled = bool(two_d and has_items and movable)
        b = self.draw_strip.buttons.get("move")
        if b is not None:
            b.setToolTip("Drag an emitter, fan or heater; drag the empty view to pan" if self.move_tool.enabled else
                         ("This explore solver cannot move emitters, fans or heaters while it runs; edit them in Setup"
                          if has_items and not movable else "Drag the view to pan"))
        if not two_d and self.view.tool() is not self.move_tool:
            self.draw_strip.set_tool("move")
            self.view.set_tool(self.move_tool)
        self.draw_strip.set_history(self.sketch.can_undo(), self.sketch.can_redo(), len(self.sketch.ops))

    def _on_draw_tool(self, key: str) -> None:
        if key == "move":
            self.sketch_tool.cancel()
            self.view.set_tool(self.move_tool)
        else:
            self.sketch_tool.set_mode(key)
            self.sketch_tool.radius = self.draw_strip.radius()
            self.view.set_tool(self.sketch_tool)
        self.view.setFocus()

    def _on_brush_size(self, radius: float) -> None:
        self.sketch_tool.radius = float(radius)
        self.view.update()

    def _edit_sketch(self, what: str) -> None:
        changed = {"undo": self.sketch.undo, "redo": self.sketch.redo}.get(what, lambda: (self.sketch.clear(), True)[1])()
        if changed:
            self._on_sketch_changed(True)

    def _sketch_grid(self):
        """(extent [m], (ny, nx)) of the explore session's grid, None before it is known."""
        plan = getattr(self.live, "plan", lambda: None)()
        try:
            cfg = plan.solver_config
            L = float(plan.length_ref)
            ext = tuple(float(e) for e in plan.extent)
            return ext, (int(cfg.ny), int(cfg.nx)), float(cfg.h) * L
        except (AttributeError, TypeError, ValueError):
            return None

    def _refresh_sketch_overlay(self) -> None:
        """Outlines of the committed drawing and the filled stroke in progress, on a display grid."""
        grid = self._sketch_grid()
        ext = grid[0] if grid is not None else self.view.extent()
        if ext is None or not self.sketch.all_ops():
            self.view.set_sketch(None)
            return
        x0, x1, y0, y1 = ext
        h = grid[2] if grid is not None else (x1 - x0) / 1024.0              # the solver's own cells
        # committed outlines change only when an operation is added or undone: traced once per change
        key = (tuple(id(op) for op in self.sketch.ops), ext, h)
        if getattr(self, "_outline_cache", (None, None))[0] != key:
            committed = ObstacleSketch()
            committed.ops = list(self.sketch.ops)
            shape = (max(8, min(2048, round((y1 - y0) / h))), max(8, min(4096, round((x1 - x0) / h))))
            self._outline_cache = (key, committed.polygons(ext, shape) if committed.ops else [])
        outlines = self._outline_cache[1]
        pending = None
        op = self.sketch.pending
        if op is not None and len(op.points):
            # the stroke in progress: traced in a window around it only (it changes with every mouse move)
            bx0, bx1, by0, by1 = op.bounds()
            hl = h / 2.0
            win = (max(x0, bx0 - 3 * hl), min(x1, bx1 + 3 * hl), max(y0, by0 - 3 * hl), min(y1, by1 + 3 * hl))
            if win[1] > win[0] and win[3] > win[2]:
                only = ObstacleSketch()
                only.ops = [SketchOp(op.kind, op.points, op.radius, False)]
                shape = (max(4, round((win[3] - win[2]) / hl)), max(4, round((win[1] - win[0]) / hl)))
                pending = only.polygons(win, shape, min_cells=0.0)
        self.view.set_sketch(outlines, pending)

    def _on_sketch_changed(self, final: bool) -> None:
        """A stroke grows (``final`` False: post throttled) or an operation is committed (post now)."""
        self._refresh_sketch_overlay()
        self.draw_strip.set_history(self.sketch.can_undo(), self.sketch.can_redo(), len(self.sketch.ops))
        self._dirty_edits.add("obstacles")
        if final:
            self._post_timer.stop()
            self._flush_live_edits()
        elif not self._post_timer.isActive():
            self._post_timer.start()

    def _on_placement_moved(self, key: str, x: float, y: float, final: bool) -> None:
        """A glyph was dragged: move it in explore's copy of the simulation and send its list to the solver."""
        sim = self._live_sim
        if sim is None:
            return
        design = self._design_of(self.state.simulation)
        if apply_move(sim, key, x, y, design, self.slider_alpha()):
            lst = key.split(":")[0]
            self._dirty_edits.add(lst)
            self._moved_lists.add(lst)
        if final:
            self._post_timer.stop()
            self._flush_live_edits()
            self._show_placements(sim, design, self.slider_alpha(), refit=False)
        elif not self._post_timer.isActive():
            self._post_timer.start()

    def _flush_live_edits(self) -> None:
        """Send what changed since the last post: the drawing and / or the moved placement lists."""
        dirty, self._dirty_edits = self._dirty_edits, set()
        if not self.live.is_running():
            return
        if "obstacles" in dirty and not self._is3d():
            grid = self._sketch_grid()
            if grid is not None:
                ext, shape, _h = grid
                payload = self.sketch.payload(ext, shape) if not self.sketch.is_empty() else None
                self.live.post(obstacles=payload)
                self._sketch_posted = self.sketch.revision
        sim = self._live_sim
        for lst in self.PLACEMENT_KEYS:
            if lst in dirty and sim is not None:
                self.live.post(**{lst: copy.deepcopy(getattr(sim, lst))})
        self._update_monitor_note()

    def _update_monitor_note(self) -> None:
        """Say when the explored coefficients include drawn obstacles (they are bodies in the flow too)."""
        if self.live.is_running() and not self.sketch.is_empty() and not self._is3d():
            self.monitor_note.setText("Latest time step · includes the drawn obstacles")
            self.monitor_note.setToolTip("Drawn obstacles are solid bodies in the explored flow; their forces are "
                                         "part of the coefficients shown")
        elif self.live.is_running():
            self.monitor_note.setText("Latest time step")
            self.monitor_note.setToolTip("")

    # -- playback
    def _on_pause(self, on: bool) -> None:
        self.pause_btn.setText("Resume" if on else "Pause")
        self.pause_btn.setIcon(theme.icon("play" if on else "pause", "text2"))
        if self.live.is_running():
            self.live.post(paused=bool(on))
            self.phase_value.setText("Paused" if on else PHASE_TEXT["live"])
        self._sync_live_controls()

    def _on_step(self) -> None:
        if self.live.is_running() and self.pause_btn.isChecked():
            step = getattr(self.live, "step", None)
            if callable(step):
                step(self.step_n.value())
            else:
                self.live.post(step=self.step_n.value())

    def _on_reset(self) -> None:
        if self.live.is_running():
            self.live.post(reset=True)
            self.diag_panel.log.clear()
            self.history.set_history(None)

    def _on_fps(self, fps: int) -> None:
        if self.live.is_running():
            fn = getattr(self.live, "set_fps", None)
            if callable(fn):
                fn(float(fps))
            else:
                self.live.post(fps=float(fps))

    def _on_spf(self, n: int) -> None:
        if self.live.is_running():
            self.live.post(steps_per_frame=int(n))

    # -- the Reynolds lock
    def _length_ref(self) -> float:
        design = self._design_of(self.state.simulation)
        L = float(getattr(design, "length_ref", 1.0) or 1.0) if design is not None else 1.0
        return L if L > 0 else 1.0

    def _follows(self) -> str:
        return self.re_follow.currentData() or "re"

    def _init_re_lock(self, sim) -> None:
        """Start the lock from the simulation's fluid: ν = μ/ρ, Re = U L / ν."""
        try:
            nu = float(sim.conditions.fluid.kinematic_viscosity)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            nu = float("nan")
        self._nu = nu
        self._show_re_lock()

    def _show_re_lock(self) -> None:
        """Write ν and Re from the held values; the derived editor is read-only."""
        self._re_lock_busy = True
        try:
            U, nu, L = self.slider_speed(), self._nu, self._length_ref()
            re_ = U * L / nu if nu and math.isfinite(nu) and nu > 0 else float("nan")
            self.nu_edit.setText(f"{nu:.4g}" if math.isfinite(nu) else "")
            self.re_edit.setText(f"{re_:.5g}" if math.isfinite(re_) else "")
            follows = self._follows()
            self.nu_edit.setReadOnly(follows == "nu")
            self.re_edit.setReadOnly(follows == "re")
            self.speed_slider.setEnabled(follows != "speed")
            for w, derived in ((self.nu_edit, follows == "nu"), (self.re_edit, follows == "re")):
                w.setProperty("derived", "true" if derived else "false")
                w.setToolTip("Follows the held values" if derived else w.toolTip().replace("Follows the held values", ""))
            sim = self.state.simulation
            capped = sim is not None and getattr(sim.cfd, "reynolds_mode", "real") == "capped"
            cap = getattr(sim.cfd, "reynolds_cap", None) if sim is not None else None
            if capped and cap and math.isfinite(re_) and re_ > float(cap):
                self.re_note.setText(f"Simulated at Re {fmt_re(float(cap))} (capped in Setup)")
                self.re_note.show()
            else:
                self.re_note.hide()
        finally:
            self._re_lock_busy = False

    def _on_follow_changed(self, *_) -> None:
        self._show_re_lock()

    def _parse(self, edit: QLineEdit) -> float | None:
        try:
            v = float(edit.text().replace("×10", "e").replace(MINUS, "-"))
        except ValueError:
            return None
        return v if v > 0 and math.isfinite(v) else None

    def _set_speed_from_lock(self, U: float) -> None:
        """Move the speed slider to a derived speed (re-centred when it leaves the slider's 0.25x .. 2x)."""
        lo, hi = self._base_speed * 0.25, self._base_speed * 2.0
        if not lo <= U <= hi:
            self._base_speed = U
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(self._speed_to_slider(U))
        self.speed_slider.blockSignals(False)
        self._label_sliders()
        if self.live.is_running():
            self.live.post(speed=self.slider_speed())

    def _post_nu(self) -> None:
        if self.live.is_running() and math.isfinite(self._nu):
            self.live.post(nu=float(self._nu))
        if self._live_sim is not None and math.isfinite(self._nu):
            f = self._live_sim.conditions.fluid
            f.viscosity = self._nu * f.density

    def _on_nu_edited(self) -> None:
        if self._re_lock_busy or self._follows() == "nu":
            return
        nu = self._parse(self.nu_edit)
        if nu is None:
            self._show_re_lock()
            return
        re_ = self._parse(self.re_edit)
        self._nu = nu
        self._post_nu()
        if self._follows() == "speed" and re_ is not None:
            self._set_speed_from_lock(re_ * nu / self._length_ref())
        self._show_re_lock()

    def _on_re_edited(self) -> None:
        if self._re_lock_busy or self._follows() == "re":
            return
        re_ = self._parse(self.re_edit)
        if re_ is None:
            self._show_re_lock()
            return
        L = self._length_ref()
        if self._follows() == "nu":
            self._nu = self.slider_speed() * L / re_
            self._post_nu()
        elif math.isfinite(self._nu):
            self._set_speed_from_lock(re_ * self._nu / L)
        self._show_re_lock()

    def _re_lock_after_speed(self) -> None:
        """The speed slider moved: with ν following, keep Re by changing ν; otherwise Re follows."""
        if self._follows() == "nu":
            re_ = self._parse(self.re_edit)
            if re_ is not None:
                self._nu = self.slider_speed() * self._length_ref() / re_
                self._post_nu()
        self._show_re_lock()

    # -- solver inspection: diagnostics and traces
    def _on_diag_toggled(self, on: bool) -> None:
        """Show the diagnostics: the window's Diagnostics dock when the shell docked the tabs, else the
        pane under the viewport."""
        m = self._dock_manager()
        if m is not None:
            self._wire_dock_switch(m)
            m.set_wanted("diagnostics", bool(on), raise_=True)
        else:
            self.bottom_tabs.setVisible(bool(on))
            if on:
                h = self.vp_split.height()
                self.vp_split.setSizes([max(200, h - 260), 260])
        if on:
            self._bind_diagnostics()
            self.diag_panel.refresh(force=True)

    def _on_dock_visibility(self, key: str, shown: bool) -> None:
        """The Diagnostics dock was shown or hidden elsewhere (View ▸ Panels): the toolbar toggle follows."""
        if key == "diagnostics" and self.diag_btn.isChecked() != bool(shown):
            self.diag_btn.blockSignals(True)
            self.diag_btn.setChecked(bool(shown))
            self.diag_btn.blockSignals(False)
            if shown:
                self._bind_diagnostics()

    def _diagnostics_shown(self) -> bool:
        m = self._dock_manager()
        if m is not None:
            return bool(m.is_shown("diagnostics")) if hasattr(m, "is_shown") else self.bottom_tabs.isVisible()
        return self.bottom_tabs.isVisible()

    def _bind_diagnostics(self) -> None:
        """Point the panel at the log of what the viewport shows: explore's, or the running / last job's."""
        log_ = None
        if self.live.is_running() or self._live_sim_id:
            log_ = getattr(self.live, "diagnostics", None) if self.live.is_running() else None
        sid = self.jobs.running_id() or self.state.simulation_id or self._shown_sim_id
        if log_ is None and sid and hasattr(self.jobs, "diagnostics_for"):
            log_ = self.jobs.diagnostics_for(sid)
            if (log_ is None or not len(log_)) and sid != self.jobs.running_id():
                # a run from an earlier session: its stored history, else its end-of-run values (the Live
                # workspace showed 650 px of "No diagnostics yet" beside a result that keeps them)
                log_ = stored_diagnostics(self.jobs, sid) or log_
        if log_ is None:
            log_ = self._diag_bound or DiagnosticsLog()
        self._diag_bound = log_
        self.diag_panel.set_log(log_)
        # θ and heat columns only for a run with heat transfer (the 3D solver reports them regardless)
        sim = (getattr(self, "_live_sim", None) if self.live.is_running() else None) or self._sim(sid) \
            or self.state.simulation
        th = getattr(sim, "thermal", None) if sim is not None else None
        self.diag_panel.set_thermal(bool(getattr(th, "enabled", False)) if th is not None else None)
        p = self.state.project
        if p is not None and getattr(p, "path", None) is not None:
            self.diag_panel.set_export_dir(p.path)
            self.trace_view.set_export_dir(p.path)
        running = self.live.is_running() or bool(self.jobs.running_id())
        refused = self.live.is_running() and self._supports("capture_trace") is False
        self.trace_view.set_capture_enabled(running and not self._is3d() and not refused,
                                            "Start a run or explore to capture a step" if not running else
                                            "This solver does not capture step traces" if refused else
                                            "Traces are captured on 2D flows")

    def _on_capture_trace(self) -> None:
        ok = False
        if self.live.is_running():
            ok = bool(getattr(self.live, "request_trace", lambda: False)())
        elif self.jobs.running_id() and hasattr(self.jobs, "request_trace"):
            ok = bool(self.jobs.request_trace(self.jobs.running_id()))
        self._trace_wanted = ok
        self.trace_view.set_waiting(ok)
        if not ok:
            self.trace_view.set_capture_enabled(False, "This solver does not capture step traces")
            self.trace_view.info.setText("This solver does not capture step traces")
        else:
            QTimer.singleShot(4000, self._trace_timeout)

    def _trace_timeout(self) -> None:
        if self._trace_wanted:
            self._trace_wanted = False
            self.trace_view.info.setText("No trace arrived: this solver build does not capture step traces")

    def _trace_extent(self):
        grid = self._sketch_grid()
        if grid is not None:
            return grid[0]
        plan = self._run_plan(self.jobs.running_id() or "")
        return getattr(plan, "extent", None)

    def _on_trace(self, trace) -> None:
        self._trace_wanted = False
        self.trace_view.set_trace(trace, extent=self._trace_extent())
        if not self._diagnostics_shown():
            self.diag_btn.setChecked(True)
        self.bottom_tabs.setCurrentWidget(self.trace_view)

    def _on_job_trace(self, sim_id: str, trace) -> None:
        if sim_id == self._shown_sim_id or sim_id == self.state.simulation_id:
            self._on_trace(trace)

    def dock_panels(self) -> dict[str, QWidget]:
        """Widgets for the window's dock panels (``widgets.docks``): the Diagnostics dock gets the
        Diagnostics / Solver trace tabs. Called once by the shell when the page is built; without a
        shell the tabs stay in the page, under the viewport."""
        self._diag_docked = True
        return {"diagnostics": self.bottom_tabs}

    def _dock_manager(self):
        if not getattr(self, "_diag_docked", False):
            return None
        try:
            from ..widgets import docks
        except ImportError:
            return None
        m = docks.manager_for(self)
        return m if m is not None and "diagnostics" in getattr(m, "panels", {}) else None

    def _wire_dock_switch(self, m) -> None:
        """The dock header's on/off switch is the Record switch (Lumen: the panel's effect, kept values)."""
        if getattr(self, "_dock_switch_wired", False):
            return
        self._dock_switch_wired = True
        try:
            panel = m.panel("diagnostics") if hasattr(m, "panel") else m.panels["diagnostics"]
            panel.enable_switch(self.diag_panel.record_btn.isChecked(), "Record the solver's diagnostics",
                                on_toggled=self.diag_panel.record_btn.setChecked)
            self.diag_panel.record_btn.toggled.connect(panel.set_switch_checked)
            self.diag_panel.record_btn.hide()          # the header switch is the Record switch now
            if hasattr(m, "panelVisibilityChanged"):
                m.panelVisibilityChanged.connect(self._on_dock_visibility)
        except Exception:
            logging.getLogger(__name__).debug("no dock header switch for diagnostics", exc_info=True)

    # -------------------------------------------------------------- explore
    def _sync_sliders(self, sim: Simulation) -> None:
        """Centre the sliders on the simulation's own conditions (speed: 0.25x .. 2x)."""
        if self.live.is_running() and self._live_sim_id == sim.id:
            return
        self._base_speed = max(sim.conditions.speed, 1e-6)
        for sl in (self.alpha_slider, self.speed_slider, self.yaw_slider):
            sl.blockSignals(True)
        self.alpha_slider.setValue(round(sim.conditions.alpha_deg * 2))
        self.speed_slider.setValue(self._speed_to_slider(sim.conditions.speed))
        self.yaw_slider.setValue(round(float(getattr(sim.conditions, "yaw_deg", 0.0) or 0.0) * 2))
        for sl in (self.alpha_slider, self.speed_slider, self.yaw_slider):
            sl.blockSignals(False)
        design = self.state.design
        self._row_alpha.set_label("Pitch angle" if design is not None and design.kind == "vehicle"
                                  else "Angle of attack")
        self._label_sliders()

    def _speed_to_slider(self, speed: float) -> int:
        f = math.log2(max(speed, 1e-9) / self._base_speed)
        return round((f + 2.0) * self.SPEED_STEPS_PER_DOUBLING)

    def slider_speed(self) -> float:
        return self._base_speed * 2.0 ** (self.speed_slider.value() / self.SPEED_STEPS_PER_DOUBLING - 2.0)

    def slider_alpha(self) -> float:
        return self.alpha_slider.value() / 2.0

    def slider_yaw(self) -> float:
        return self.yaw_slider.value() / 2.0

    def _label_sliders(self) -> None:
        self.alpha_value.setText(f"{self.slider_alpha():+.1f}°".replace("-", "−"))
        self.yaw_value.setText(f"{self.slider_yaw():+.1f}°".replace("-", "−"))
        self.speed_value.setText(fmt_speed(self.slider_speed(), both=False))
        self.speed_value.setToolTip(fmt_speed(self.slider_speed()))

    def _on_alpha(self, *_) -> None:
        self._label_sliders()
        if self.live.is_running():
            self.live.post(alpha_deg=self.slider_alpha())
            if self._is3d():
                self._attitude_timer.start()
            elif self._live_sim is not None:
                # body-fixed emitters, fans and heaters turn with the geometry
                self._show_placements(self._live_sim, self._design_of(self.state.simulation), self.slider_alpha(),
                                      refit=False)

    def _on_yaw(self, *_) -> None:
        self._label_sliders()
        if self.live.is_running() and self._is3d():
            self.live.post(yaw_deg=self.slider_yaw())
            self._attitude_timer.start()

    def _on_speed(self, *_) -> None:
        self._label_sliders()
        if self.live.is_running():
            self.live.post(speed=self.slider_speed())
        self._re_lock_after_speed()

    def toggle_live(self) -> None:
        if self.live.is_running():
            self.live.stop()
            return
        project, sim = self.state.project, self.state.simulation
        if project is None or sim is None:
            self.state.message("Select a simulation in the Project panel first.")
            return
        if self.jobs.running_id():
            self.state.message("Explore is unavailable while a simulation runs.")
            return
        design = project.designs.get(sim.design_id)
        if design is None:
            self._on_live_failed("the design of this simulation is missing")
            return
        if sim.analysis == "wing":
            self._on_live_failed("a wing sweep has no flow to explore; explore runs the 3D flow solver, so "
                                 "make a CFD simulation of this wing in Setup")
            return
        if lazy_import("aerolume.simulation.live") is None:
            self._on_live_failed("the real-time solver is not installed")
            return
        self.live_error.hide()
        self._live_sim_id = sim.id
        # explore edits its own copy: moving a heater while exploring must not edit the simulation itself
        self._live_sim = clone_simulation(sim)
        self._moved_lists = set()
        if self._sketch_sim != sim.id:            # a drawing belongs to the case it was drawn on
            self.sketch = ObstacleSketch()
            self.sketch_tool.sketch = self.sketch
            self._sketch_sim = sim.id
        self._init_re_lock(sim)
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.pause_btn.setIcon(theme.icon("pause", "text2"))
        self.pause_btn.blockSignals(False)
        three_d = getattr(design, "dimension", 2) == 3
        # explore shows the flow, also for a polar's airfoil
        self._set_mode("field3d" if three_d else "field", design, sim)
        self.view.clear()
        ground = ground_of(sim)
        self.view.set_ground(ground)
        if not three_d:
            self.view.set_bodies(wind_bodies(design, self.slider_alpha(), ground=ground is not None))
            self._frame_closed(sim)
            self._show_placements(self._live_sim, design, self.slider_alpha())
        else:
            self._live_volume = self._live_lines = self._last_frame3d = None
            self._push_flow_settings()
            # the session releases its smoke where this simulation's run would
            rake_of = getattr(self.jobs, "seed_rake_for", None)
            mine = self._rakes.get(sim.id)
            self._set_view("seed_rake", mine.to_view() if mine is not None else
                           (rake_of(sim) if callable(rake_of) else None))
            self._show_slice_preview(sim, design)
        self._set_placeholder("Starting the real-time solver…", "Building the explore grid" if three_d else "")
        self.history.set_x_limits(None)
        # until the first frame the Run rows and monitors describe nothing: a 3D explore takes seconds to
        # build its grid, and "Time 11.1 of 20.0 L/U" with the last run's mean C_D stood under "Live"
        self._reset_run_rows()
        self._clear_monitors()
        self._reset_live_range()
        v = self.jobs.view
        self.live.start(design, sim, device=self._device_for(sim), view=v,
                        airfoils=airfoil_designs(project, design))
        posts = {"alpha_deg": self.slider_alpha(), "speed": self.slider_speed(), "field": v.field, "cmap": v.cmap,
                 "particles": v.particles, "streamlines": v.streamlines}
        if three_d:
            posts.update(yaw_deg=self.slider_yaw(), slice_axis=self.slice_axis.current(),
                         slice_pos=getattr(v, "slice_pos", None))
        self.live.post(**posts)
        self.live_btn.setText("Stop explore")
        self.status_text.set_status("Explore", "accent")
        self.phase_value.setText(PHASE_TEXT["starting"])
        self._update_buttons()
        self._bind_diagnostics()
        self._sync_live_controls()

    @staticmethod
    def _device_for(sim: Simulation) -> str | None:
        """The simulation's own device choice, else the user's preference; None = automatic."""
        dev = sim.cfd.device
        if dev in (None, "", "auto"):
            try:
                from .. import settings
                dev = settings.get().device
            except Exception:  # noqa: BLE001 - preferences unavailable: let the solver choose
                dev = "auto"
        return None if dev in (None, "", "auto") else dev

    def _on_live_started(self) -> None:
        self.phase_value.setText(PHASE_TEXT["live"])
        self._log(f"Explore started for “{self._sim_name(self._live_sim_id)}”")
        self._show_live_plan()
        self._update_buttons()
        if self.fps_spin.value() != 30:
            self._on_fps(self.fps_spin.value())
        if self.spf_spin.value():
            self._on_spf(self.spf_spin.value())
        if self.sketch.ops and not self._is3d():
            # the drawing of an earlier explore of this case goes back into the new flow
            self._dirty_edits.add("obstacles")
            self._flush_live_edits()
            self._refresh_sketch_overlay()
        self.draw_strip.set_reference_length(self._length_ref())
        grid = self._sketch_grid()
        if grid is not None:
            self.draw_strip.set_min_radius(1.5 * grid[2])      # thinner strokes break into beads on the grid
        self._bind_diagnostics()
        self._sync_live_controls()

    def _show_live_plan(self) -> None:
        """Describe the explore session's own grid in the Run panel.

        Explore runs a smaller grid than the production plan (sized for real-time
        rates, up to 10 % blockage) and has no end, so the production grid, its
        run-time estimate and its warnings would describe something not on screen.
        """
        plan_fn = getattr(self.live, "plan", None)
        plan = plan_fn() if callable(plan_fn) else None
        if plan is None:
            return
        try:
            # a 3D grid text is long: the panel's Status ("Explore") already says whose grid it is
            grid = self._grid_text(plan, "" if getattr(plan, "dimension", 2) == 3 else " (explore)")
        except AttributeError:
            return
        self._show_plan_rows(grid, fmt_re(getattr(plan, "re_sim", float("nan"))), "— (runs until stopped)")
        self.grid_value.setToolTip("The explore grid: coarser than the full run's, sized for real-time rates")
        self.plan_note.hide()
        if getattr(plan, "dimension", 2) == 3:
            self.sym_value.setText("None (full domain)")
            self.sym_value.setToolTip("Explore always solves both halves: a yaw change breaks the mirror symmetry")
            self._update_slice_range()
            self._plan_to_3d(plan, self.state.simulation)

    def _on_live_frame(self, frame) -> None:
        self._show_frame(frame)
        # edits the explore session could not carry out (a speed change in a closed box, an item
        # moved out of the domain, heat added to a session started without it): without this the
        # control would simply appear to do nothing. Each notice arrives on one frame only.
        for text in getattr(frame, "notices", None) or ():
            if isinstance(text, str) and text.strip():
                self._log(text.strip(), "warning")
        sps = getattr(frame, "steps_per_s", float("nan"))
        self.rate_value.setText(f"{sps:,.0f} steps/s" if sps and math.isfinite(sps) else "—")
        t = getattr(frame, "t", float("nan"))
        self.time_value.setText(f"{sig(t, 3)} L/U" if math.isfinite(t) else "—")

    def _on_live_failed(self, message: str) -> None:
        text = f"Explore could not run: {message}"
        self.live_error.set_text(text)
        self.live_error.show()
        self._log(text, "error")

    def _on_live_stopped(self) -> None:
        self.live_btn.setText("Start explore")
        self._post_timer.stop()
        self._dirty_edits.clear()
        self.sketch_tool.cancel()
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.pause_btn.setIcon(theme.icon("pause", "text2"))
        self.pause_btn.blockSignals(False)
        self.view.set_sketch(None)                # the drawing lived in explore's flow only
        self._sync_live_controls()
        sim = self.state.simulation
        if sim is not None and self._mode == "field":
            self._show_placements(sim, self._design_of(sim))       # the simulation's own positions again
        if not self.jobs.running_id() and sim is not None:
            self._set_status_from_sim(sim)
            self.phase_value.setText("—")
            self._request_plan(sim, self._design_of(sim))     # back to the production plan
            if self._mode_for(sim) != self._mode:     # a polar's page returns to its plots
                self._shown_sim_id = ""
                self._show_preview(sim, self._design_of(sim))
        self._update_buttons()

    def promote_live_settings(self) -> None:
        """Turn the explored angle / speed (and yaw, in 3D) into a full simulation and queue it.

        A simulation that already has a result is never overwritten: a new one
        named after the settings is created next to it, so the comparison of the
        two is one click away on the Compare page.
        """
        project, sim = self.state.project, self.state.simulation
        if project is None or sim is None:
            return
        alpha, speed = self.slider_alpha(), self.slider_speed()
        three_d = self._is3d()
        yaw = self.slider_yaw() if three_d else None
        live_sim = self._live_sim if self._live_sim_id == sim.id else None
        if self.live.is_running():
            self.live.stop()

        def explored(target: Simulation) -> None:
            """The explored viscosity and moved emitters / fans / heaters / ports go into the full run."""
            if live_sim is None:
                return
            nu = self._nu
            if math.isfinite(nu) and nu > 0:
                f = target.conditions.fluid
                if abs(f.kinematic_viscosity / nu - 1.0) > 1e-9:
                    f.viscosity = nu * f.density
            for lst in self._moved_lists:
                setattr(target, lst, copy.deepcopy(getattr(live_sim, lst)))

        if sim.status in ("new", "failed") and not self.jobs.is_busy(sim.id):
            sim.conditions.alpha_deg = alpha
            sim.conditions.speed = speed
            if yaw is not None:
                sim.conditions.yaw_deg = yaw
            explored(sim)
            self.state.update_simulation(sim)
            target = sim
        else:
            d = sim.to_dict()
            for k in ("id", "status", "finished", "error", "created"):
                d.pop(k, None)
            target = Simulation.from_dict(d)
            target.created = now_iso()
            yaw_txt = f", yaw {yaw:g}°" if yaw else ""
            target.name = f"{sim.name} · α {alpha:g}°{yaw_txt}, {fmt_speed(speed, both=False)}"
            target.conditions.alpha_deg = alpha
            target.conditions.speed = speed
            if yaw is not None:
                target.conditions.yaw_deg = yaw
            explored(target)
            self.state.add_simulation(target, select=True)
        self.jobs.submit(target.id)
        self._log(f"Queued “{target.name}” from the explored settings")

    def _on_sim_chosen(self, index: int) -> None:
        sid = self.sim_combo.itemData(index)
        if sid and sid != self.state.simulation_id:
            self.state.select_simulation(sid)
