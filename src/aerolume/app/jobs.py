"""Background execution: the simulation queue (``JobManager``) and the explore loop (``LiveWorker``).

DESIGN.md 5.8 requires that the window never blocks: every solve runs on a
worker thread and only results cross back to the GUI thread. The mechanics:

* **Threads.** Plain ``threading.Thread`` workers. The solver is PyTorch, which
  releases the GIL inside its kernels, so a Python thread gives the GUI all the
  time it needs; a QThread would add nothing but a second object lifetime.
* **Crossing threads.** Workers never touch widgets or ``AppState``. They emit
  signals of a private ``_Bridge`` QObject that lives in the GUI thread, connected
  with ``Qt.QueuedConnection``, so every slot below runs in the GUI thread.
* **Frames are coalesced, not queued.** A frame is ~2 MB of RGBA. The worker
  drops each new frame into a one-slot mailbox and only signals when the slot was
  empty; the GUI always paints the newest frame, so a slow GUI (a modal dialog, a
  laptop on battery) can never build a backlog or fall seconds behind the solver.
  3D runs attach a flow volume (smoke and velocity, DESIGN.md 7.13) to only some
  frames; a frame without one that replaces an unread frame with one inherits
  that volume (:func:`carry_volume`), and the rate limit never drops a frame
  that carries one, so the 3D view always gets the newest smoke.
* **Copies in, results out.** The worker receives deep copies of the Design and
  Simulation made when the job starts, so editing the setup while a job runs
  cannot race with the solver. They are real deep copies (:func:`clone_design`),
  not a ``to_dict`` round trip: ``Mesh3D.to_dict`` leaves the vertex and face
  arrays out (the project keeps them in an .npz), so such a "copy" of a 3D
  design has no triangles and every 3D run failed at planning.
* **Three analyses, one dispatcher.** ``execute_simulation`` hands every run to
  :func:`aerolume.simulation.runner.run_simulation` (as the command line and
  the scripting API do): a polar sweep goes to the airfoil solver, a
  finite-wing sweep (``analysis == "wing"``) to ``run_wing`` and everything
  else, 2D or 3D, to the Navier-Stokes planner and runner (which dispatch on
  ``Design.dimension``). Wing sections may name a 2D airfoil design of the
  project ("design:<id>"); the job takes a snapshot of those designs on the GUI
  thread (:func:`airfoil_designs`) so the worker never reads a project the user
  is editing.
* **The smoke rake belongs to its simulation.** ``view`` (``ViewSettings``) is
  shared by every run and read by the solver every frame, so a move of the
  rake on the Simulate page reaches a running solver at once. Where a run
  *starts* its smoke is per simulation (:meth:`JobManager.seed_rake_for`: the
  rake the user placed for it, else Setup's emitter), set on the view when the
  job starts -- a run started from the Project tree never inherits the rake of
  another case the Simulate page happened to show.
* **Saving.** ``project.save_result`` (an ``.npz`` of fields, up to tens of MB)
  runs on the worker thread so the GUI does not stall for a second at the end of
  every run. It writes only ``sims/<id>/``; ``project.json`` and the simulation
  status are updated afterwards on the GUI thread through ``AppState``.
* **Cancellation** is a ``threading.Event`` handed to the runner, which returns a
  partial result (its warnings say so); that partial result is kept, marked
  "cancelled", because a half-finished run is often all a user needed -- unless
  the simulation already had a complete result when the run started ("Run
  again", then Stop, a window close or a project switch): the partial run is
  then discarded and the previous result and its "done" status are kept.
* **A disk that refuses a write never costs a result or the queue.** OneDrive,
  an antivirus scanner or a full disk can block a file for longer than the
  store's retry. A result that cannot be saved is still delivered (pages show
  it, marked "not saved yet"), held in memory and saved again in the
  background with a growing delay (:data:`SAVE_RETRY_DELAYS_S`); a status that
  cannot be saved stays in memory (it is written with the next successful
  save) and is reported in Messages. The queue moves on in every case.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import logging
import math
import threading
import time
import traceback
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Qt, QTimer, Signal

from ..model import Design, Simulation, now_iso

log = logging.getLogger(__name__)

FRAME_MIN_INTERVAL_S = 1.0 / 60.0     # never deliver frames faster than the display can show
PROGRESS_MIN_INTERVAL_S = 1.0 / 15.0  # progress text/bars do not need more than this
#: Delays [s] between the background attempts to save a result whose first save failed.
#: OneDrive uploads of a tens-of-MB arrays.npz hold the file for seconds; together the
#: attempts cover about half a minute before the user is asked to retry by hand.
SAVE_RETRY_DELAYS_S = (1.0, 2.0, 4.0, 8.0, 15.0)
#: Messages-dock source of the job manager's own lines (save problems, kept results).
LOG_SOURCE = "Solver"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@dataclass
class FallbackViewSettings:
    """Mirror of ``simulation.live.ViewSettings`` used while that module is unavailable."""
    field: str = "speed"
    cmap: str | None = None
    range_mode: str = "auto"
    vmin: float | None = None
    vmax: float | None = None
    particles: bool = False
    streamlines: bool = False
    fps: float = 20.0
    slice_axis: str = "y"
    slice_pos: float | None = None
    view3d: bool = False
    smoke: bool = True
    streamlines3d: bool = False
    seed_rake: dict | None = None
    capture_trace: bool = False


#: What the 3D view asks the solver for (DESIGN.md 7.13), with defaults. Set on the shared
#: settings object even when the installed ``ViewSettings`` predates them: the GUI reads and
#: writes them all the same, and a solver that does not know them ignores them.
#: ``capture_trace`` (DESIGN.md 10.4) asks a running solver to record the stages of its next
#: step into the next frame (``LiveFrame.trace``); the producer sets it back to False.
VIEW3D_DEFAULTS = {"slice_axis": "y", "slice_pos": None, "view3d": False, "smoke": True, "streamlines3d": False,
                   "seed_rake": None, "capture_trace": False}


def make_view_settings():
    """A ``ViewSettings`` instance (the real class when importable) carrying every 3D view setting."""
    try:
        from ..simulation.live import ViewSettings
        v = ViewSettings()
    except Exception:  # noqa: BLE001 - any import failure means: use the mirror
        return FallbackViewSettings()
    for k, default in VIEW3D_DEFAULTS.items():
        if not hasattr(v, k):
            try:
                setattr(v, k, default)
            except (AttributeError, TypeError):        # a slotted class: the GUI falls back to defaults
                log.debug("ViewSettings has no %s and refuses it", k)
    return v


#: Flow data a 3D frame may carry that the next frames do not repeat (DESIGN.md 7.13): the
#: volume with its serial number, and streamlines traced on the solver thread.
CARRIED_FRAME_DATA = (("volume", "volume_serial"), ("streamlines3d",))


def carry_volume(old, new):
    """``new`` with ``old``'s flow data where ``new`` has none (both frames; either may be None).

    Runs on the worker thread while the GUI has not read ``old`` yet: the
    newer frame replaces it in the mailbox, and its slice is newer, but the
    volume (and the traced streamlines) came with the older one only.
    LiveFrame is a dataclass, so a shallow copy with those fields set costs
    nothing (no array is copied).
    """
    if old is None or new is None:
        return new
    take: dict = {}
    for names in CARRIED_FRAME_DATA:
        if getattr(old, names[0], None) is not None and getattr(new, names[0], None) is None:
            take.update({n: getattr(old, n) for n in names if hasattr(old, n)})
    if not take:
        return new
    try:
        import dataclasses
        if dataclasses.is_dataclass(new):
            fields = {f.name for f in dataclasses.fields(new)}
            return dataclasses.replace(new, **{k: v for k, v in take.items() if k in fields})
        for k, v in take.items():
            setattr(new, k, v)
    except (TypeError, ValueError, AttributeError):    # a frame type without those fields
        log.debug("could not carry the flow data over to a newer frame", exc_info=True)
    return new


def friendly_error(exc: BaseException) -> str:
    """One readable sentence for the UI; the traceback goes to the log."""
    name = type(exc).__name__
    text = str(exc).strip()
    if isinstance(exc, ModuleNotFoundError):
        return (f"A required component is missing ({exc.name or text}). "
                "The installation may be incomplete; see the log for details.")
    if name == "OutOfMemoryError" or "out of memory" in text.lower():
        return ("The GPU ran out of memory. Choose a lower quality preset, close other GPU programs, "
                "or switch the device to CPU in Setup.")
    if "CUDA" in text and "error" in text.lower():
        return f"The GPU reported an error: {text[:300]}. Try again, or run on the CPU."
    if not text:
        return f"{name} (no message). See the log for the traceback."
    return f"{text[:400]}" if name in ("RuntimeError", "ValueError") else f"{name}: {text[:400]}"


#: Modules the runtime pages need that are slow to import (matplotlib's colour-map
#: registry: about 0.6 s). Imported once on a background thread when the first
#: JobManager is made, i.e. while the user is still on the start page, so the first
#: visit to Simulate / Results does not stall the window for that long.
PREWARM_MODULES = ("aerolume.viz.colormaps",)
_prewarmed = threading.Event()


def prewarm(modules=PREWARM_MODULES) -> None:
    """Import ``modules`` on a daemon thread (once per process); failures are left to the user."""
    if _prewarmed.is_set():
        return
    _prewarmed.set()

    def work() -> None:
        for name in modules:
            try:
                importlib.import_module(name)
            except Exception:  # the page that needs the module reports the failure
                log.debug("prewarming %s failed", name, exc_info=True)

    threading.Thread(target=work, name="aerolume-prewarm", daemon=True).start()


class _Mailbox:
    """Thread-safe single slot: ``put`` overwrites, ``take`` empties.

    ``put`` returns True when the slot was empty, i.e. when the consumer has to
    be woken up; later puts before the consumer runs just replace the item.
    ``merge(old, new)``, when given, builds the item that replaces an unread one
    (frames keep the newest flow volume, see :func:`carry_volume`).
    """

    def __init__(self, merge=None) -> None:
        self._lock = threading.Lock()
        self._item: Any = None
        self._full = False
        self._merge = merge

    def put(self, item: Any) -> bool:
        with self._lock:
            was_empty = not self._full
            if not was_empty and self._merge is not None:
                item = self._merge(self._item, item)
            self._item, self._full = item, True
            return was_empty

    def take(self) -> Any:
        with self._lock:
            item, self._item, self._full = self._item, None, False
            return item


def _carry_job_volume(old, new):
    """:func:`carry_volume` for the job mailbox, whose items are (sim id, frame) pairs of one run."""
    if old is None or new is None or old[0] != new[0]:
        return new
    return new[0], carry_volume(old[1], new[1])


class _Bridge(QObject):
    """Signals emitted by worker threads, delivered (queued) in the GUI thread."""
    progress = Signal(str, float, str)
    frame = Signal()
    planned = Signal(str, object)               # sim id, SimulationPlan
    done = Signal(str, object, object)          # sim id, result | None, (message, traceback) | None
    trace = Signal(str, object)                 # sim id ("" for explore), a captured solver step
    live_frame = Signal()
    live_started = Signal()
    live_failed = Signal(str)
    live_stopped = Signal()


def _new_diagnostics_log(source: str = ""):
    """A :class:`~aerolume.app.widgets.diagnostics.DiagnosticsLog` (imported lazily: no Qt widgets needed)."""
    from .widgets.diagnostics import DiagnosticsLog
    log_ = DiagnosticsLog()
    log_.source = source
    return log_


def attach_diagnostics_history(result, log_) -> bool:
    """Store a decimated copy of the run's diagnostics (``DiagnosticsLog.decimated``, about 200 rows) as
    ``result.diagnostics_history`` before the result is saved; returns whether it did.

    The runner keeps only the solver's last diagnostics row, so a reopened result showed "1 sample ·
    step 7,040" and empty history plots. The project store writes this table to its own key
    (``project.results._pack_diagnostics``), Results ▸ Diagnostics and the Diagnostics dock read it,
    and the research export writes it as CSV. A table the runner already attached is kept, and a
    log with fewer than two rows adds nothing."""
    if log_ is None or result is None or getattr(result, "diagnostics_history", None):
        return False
    try:
        table = log_.decimated()
        if len(table.get("step", ())) < 2:
            return False
        result.diagnostics_history = table
        return True
    except Exception:  # a result without its history is still a result
        log.debug("diagnostics history not attached", exc_info=True)
        return False


def frame_trace(frame):
    """The captured solver step a frame carries (``LiveFrame.trace``, DESIGN.md 10.4), or None."""
    return getattr(frame, "trace", None) if frame is not None else None


def clone_design(design: Design) -> Design:
    """A private deep copy of a design, mesh arrays included (see the module docstring)."""
    return copy.deepcopy(design)


def clone_simulation(sim: Simulation) -> Simulation:
    """A private deep copy of a simulation (its settings may be edited while a job runs)."""
    return copy.deepcopy(sim)


def airfoil_designs(project, design: Design | None) -> dict[str, Design]:
    """Copies of the 2D airfoil designs a wing's sections name ("design:<id>"), by id.

    Made on the GUI thread when a job (or explore session) starts: the worker
    resolves those airfoils from this snapshot, never from the live project.
    Empty for anything but a wing that uses project airfoils.
    """
    wing = getattr(design, "wing", None) if design is not None else None
    designs = getattr(project, "designs", None)
    if wing is None or not isinstance(designs, Mapping):
        return {}
    out: dict[str, Design] = {}
    for s in getattr(wing, "sections", []) or []:
        name = str(getattr(s, "airfoil", "")).strip()
        if name.lower().startswith("design:"):
            did = name.split(":", 1)[1].strip()
            d = designs.get(did)
            if d is not None and did not in out:
                out[did] = clone_design(d)
    return out


def _airfoil_coords(airfoils: dict | None):
    """The "design:<id>" resolver for a snapshot from :func:`airfoil_designs` (None when empty)."""
    if not airfoils:
        return None
    from ..simulation.wing import airfoil_resolver
    return airfoil_resolver(airfoils)


def execute_simulation(design: Design, sim: Simulation, *, progress=None, frame=None,
                       cancel: threading.Event | None = None, view=None, planned=None,
                       airfoils: dict | None = None):
    """Run one simulation synchronously (called on a worker thread).

    Polar sweeps go to the viscous-inviscid airfoil solver, finite-wing sweeps
    (``analysis == "wing"``) to the vortex-lattice / lifting-line runner;
    everything else is planned and run with the Navier-Stokes runner, which
    returns a ``CFDResult`` or, for a 3D design, a ``CFD3DResult``.
    ``planned(plan)``, when given, receives the plan the runner is about to
    execute, so the GUI shows that plan rather than one it made itself from
    settings that may have changed since. ``airfoils`` ({id: Design}) resolves
    wing sections that name a project airfoil. Module-level so tests (and a
    future remote-execution back end) can swap it out.

    The dispatch itself is :func:`aerolume.simulation.runner.run_simulation`,
    the one the command line and the scripting API use too (DESIGN.md 5.4,
    7.7), so a simulation gives the same result whichever of them runs it. The
    airfoil snapshot goes in as its ``project`` (a mapping id -> Design is a
    valid one), never the live project the user is editing. A runner module
    without that function (an older build, a test fake) is dispatched here.
    """
    runner = importlib.import_module("aerolume.simulation.runner")
    dispatch = getattr(runner, "run_simulation", None)
    if callable(dispatch):
        return dispatch(design, sim, project=airfoils or None, progress=progress, frame=frame, cancel=cancel,
                        view=view, planned=planned)
    if sim.analysis == "polar":
        return runner.run_polar(design, sim, progress=progress, cancel=cancel)
    if sim.analysis == "wing":
        return runner.run_wing(design, sim, progress=progress, cancel=cancel, project=airfoils or None)
    plan_mod = importlib.import_module("aerolume.simulation.plan")
    kw = {}
    if getattr(design, "dimension", 2) == 3:
        coords = _airfoil_coords(airfoils)
        if coords is not None:
            kw["airfoil_coords"] = coords
    plan = plan_mod.plan_simulation(design, sim, **kw)
    if planned is not None:
        planned(plan)
    return runner.run_cfd(plan, progress=progress, frame=frame, cancel=cancel, view=view)


def require_api(jobs, names, user: str) -> None:
    """Raise TypeError naming what ``jobs`` lacks, before ``user`` builds a single widget.

    Pages take the JobManager by duck typing (tests hand them small fakes). A
    page that found a missing method only half-way through its constructor
    left dozens of half-built pyqtgraph widgets to the garbage collector; on
    PySide6 6.11 that was followed by heap corruption a few tests later. Failing
    up front leaves nothing behind (the window shows its placeholder page).
    """
    missing = [n for n in names if not hasattr(jobs, n)]
    if missing:
        raise TypeError(f"{user} needs a JobManager; the object given lacks {', '.join(missing)}")


def _accepts(fn, name: str) -> bool:
    """Whether a callable takes the keyword ``name`` (custom runners and test fakes may not)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _accepts_planned(fn) -> bool:
    """Whether a run function takes the ``planned`` callback (custom runners may not)."""
    return _accepts(fn, "planned")


@dataclass
class _Job:
    sim_id: str
    project: Any
    design: Design | None = None
    sim: Simulation | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    started: float = 0.0
    fraction: float = 0.0
    message: str = ""
    #: (result, error) set by the worker just before it signals ``done``, so ``shutdown``
    #: can finish the job synchronously when the queued signal would arrive too late
    outcome: tuple | None = None
    handled: bool = False
    #: whether the stop request came before the worker finished (a complete run that
    #: ends just before a window close is "done", not "cancelled")
    stopped: bool = False
    #: the plan the runner executes (Navier-Stokes runs; set once planning is done)
    plan: Any = None
    #: the simulation's status before it was queued (restored when it leaves the queue unrun)
    prior_status: str = "new"
    #: it had a complete, saved result when queued: a stopped re-run must not replace it
    keep_previous: bool = False
    #: the run was stopped and its partial result discarded, keeping the previous one
    kept_previous: bool = False
    #: why the result could not be saved ("" when it was)
    save_error: str = ""
    #: copies of the project airfoils a wing's sections name (see airfoil_designs)
    airfoils: dict = field(default_factory=dict)


@dataclass
class _Unsaved:
    """A finished result whose save failed, held until a background retry succeeds."""
    project: Any
    result: Any
    attempts: int = 0          # background attempts made so far
    busy: bool = False         # an attempt is running
    superseded: bool = False   # a newer run of the simulation replaced it: never save it


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

class JobManager(QObject):
    """Runs CFD (2D / 3D), polar and wing jobs one at a time on a worker thread, FIFO queue for the rest.

    All signals are emitted in the GUI thread. ``jobFinished`` fires after the
    result has been saved and ``sim.status`` updated. A cancelled run that still
    produced a partial result saves it and emits ``jobCancelled`` (the Results
    page then shows it with its "cancelled" warning) -- except a stopped re-run of
    a simulation that already had a complete result, which keeps that result
    (status "done") and discards the partial one. A result whose save fails is
    delivered all the same and saved again in the background
    (:meth:`unsaved_ids`, :meth:`retry_saves`).
    """

    jobQueued = Signal(str)
    jobStarted = Signal(str)
    jobProgress = Signal(str, float, str)       # sim id, fraction 0..1, message
    jobFrame = Signal(str, object)              # sim id, LiveFrame
    #: sim id, the result (already saved): CFDResult | CFD3DResult | PolarResult | WingResult
    jobFinished = Signal(str, object)
    jobFailed = Signal(str, str)                # sim id, readable message
    jobCancelled = Signal(str)
    #: sim id, the SimulationPlan the runner executes (Navier-Stokes runs, shortly after jobStarted)
    jobPlanned = Signal(str, object)
    queueChanged = Signal()                     # extension: any change to running / queued ids
    #: sim id, one captured solver step (``LiveFrame.trace``, DESIGN.md 10.4) after :meth:`request_trace`
    jobTrace = Signal(str, object)

    RESULT_CACHE_SIZE = 4   # results hold full flow fields (tens of MB each)

    def __init__(self, state, parent: QObject | None = None) -> None:
        super().__init__(parent if parent is not None else state)
        self.state = state
        self.view = make_view_settings()
        self.runner = None                      # override for execute_simulation (tests, remote)
        self._queue: deque[_Job] = deque()
        self._running: _Job | None = None
        self._frames = _Mailbox(merge=_carry_job_volume)
        self._last_progress_emit = 0.0
        self._last_frame_post = 0.0
        self._keep_cached: str | None = None
        self._results: OrderedDict[str, Any] = OrderedDict()
        self._result_errors: dict[str, str] = {}
        #: results whose save failed, by sim id. Never evicted like the LRU cache (it may be
        #: the only copy) until a save succeeds.
        self._unsaved: dict[str, _Unsaved] = {}
        #: the seed rake the user placed for a simulation (Simulate page), by sim id
        self._seed_rakes: dict[str, dict] = {}
        #: wall-clock seconds of each job that ended in this session (start of planning to the end), by sim id
        self._job_seconds: dict[str, float] = {}
        #: solver diagnostics of each run in this session, by sim id (fed on the worker thread)
        self._diagnostics: OrderedDict[str, Any] = OrderedDict()
        self._bridge = _Bridge()
        q = Qt.ConnectionType.QueuedConnection
        self._bridge.progress.connect(self._on_progress, q)
        self._bridge.frame.connect(self._on_frame, q)
        self._bridge.planned.connect(self._on_planned, q)
        self._bridge.done.connect(self._on_done, q)
        self._bridge.trace.connect(self.jobTrace, q)
        state.projectChanged.connect(self._on_project_changed)
        state.resultChanged.connect(self._invalidate_result)
        prewarm()

    # -- queries ----------------------------------------------------------
    def running_id(self) -> str | None:
        return self._running.sim_id if self._running is not None else None

    def queued_ids(self) -> list[str]:
        return [j.sim_id for j in self._queue]

    def is_busy(self, sim_id: str) -> bool:
        """True if the simulation is running or waiting in the queue."""
        return sim_id == self.running_id() or sim_id in self.queued_ids()

    def plan_of(self, sim_id: str):
        """The plan the running job of ``sim_id`` executes; None before it is made or when not running."""
        j = self._running
        return j.plan if j is not None and j.sim_id == sim_id else None

    def job_inputs(self, sim_id: str) -> tuple[Design | None, Simulation | None]:
        """(design, simulation) copies the running job of ``sim_id`` was started with, else (None, None)."""
        j = self._running
        return (j.design, j.sim) if j is not None and j.sim_id == sim_id else (None, None)

    def job_seconds(self, sim_id: str) -> float | None:
        """Wall-clock seconds of the job of ``sim_id``: so far while it runs, in total once it ended.

        This is the one duration the app quotes for a run ("Done in 1 min 00 s"
        in the Run panel and the status bar): from the job's start, planning
        and grid building included, to its end. The solver's own
        ``result.wall_seconds`` covers the time stepping only and is shorter,
        so quoting both made one run last 58 s in one place and 1 min 00 s in
        another. None when this session has not run it (the caller may fall
        back to the stored ``wall_seconds``).
        """
        j = self._running
        if j is not None and j.sim_id == sim_id and j.started:
            return time.monotonic() - j.started
        return self._job_seconds.get(sim_id)

    def progress_of(self, sim_id: str) -> tuple[float, str, float]:
        """(fraction, message, elapsed seconds) of the running job, zeros otherwise."""
        j = self._running
        if j is None or j.sim_id != sim_id:
            return 0.0, "", 0.0
        return j.fraction, j.message, time.monotonic() - j.started

    # -- flow visualisation (DESIGN.md 7.13) ------------------------------
    def set_seed_rake(self, sim_id: str, rake: dict | None) -> None:
        """Remember where the user placed the seed rake of ``sim_id`` (a ``viz3d.flow`` rake dict, metres;
        None forgets it). A run of that simulation releases its smoke there; while it runs, the shared
        view settings carry the move to the solver at once (it re-reads them every frame)."""
        if rake is None:
            self._seed_rakes.pop(sim_id, None)
        else:
            self._seed_rakes[sim_id] = dict(rake)
        if sim_id and sim_id == self.running_id():
            job = self._running
            sim = job.sim if job is not None and job.sim is not None else \
                getattr(self.state.project, "simulations", {}).get(sim_id)
            self._set_view_rake(self.seed_rake_for(sim))

    def seed_rake_for(self, sim) -> dict | None:
        """Where a run of ``sim`` releases its smoke and starts its streamlines.

        The rake the user placed on the Simulate page, else the emitter of Setup
        ▸ Flow visualisation (:func:`~aerolume.app.widgets.view3d_panel.setup_flow_viz`),
        else None -- the solver's default rake. Per simulation: the view settings
        are shared by every run, so a run started from the Project tree while
        the Simulate page shows another case must not inherit that case's rake.
        """
        if sim is None:
            return None
        rake = self._seed_rakes.get(getattr(sim, "id", ""))
        if rake is not None:
            return dict(rake)
        try:
            from .widgets.view3d_panel import setup_flow_viz
            fv = setup_flow_viz(sim)
        except Exception:  # an unreadable setting falls back to the default rake
            log.debug("no Setup rake for %s", getattr(sim, "id", "?"), exc_info=True)
            fv = None
        rake = fv.get("rake") if fv else None
        return dict(rake) if rake else None

    def _set_view_rake(self, rake: dict | None) -> None:
        try:
            self.view.seed_rake = rake
        except (AttributeError, TypeError):         # a settings object that refuses the name
            log.debug("the view settings refuse seed_rake")

    # -- solver diagnostics and traces (DESIGN.md 10.4) ---------------------
    #: Diagnostics logs kept (one per run of this session; a log is at most a few MB).
    DIAGNOSTICS_KEPT = 8

    def diagnostics_for(self, sim_id: str):
        """The :class:`~aerolume.app.widgets.diagnostics.DiagnosticsLog` of the last run of ``sim_id`` in
        this session (created empty on first use, so a page can bind it before the run starts)."""
        d = self._diagnostics.get(sim_id)
        if d is None:
            sim = getattr(self.state.project, "simulations", {}).get(sim_id) if self.state.project else None
            d = self._diagnostics[sim_id] = _new_diagnostics_log(getattr(sim, "name", ""))
            while len(self._diagnostics) > self.DIAGNOSTICS_KEPT:
                self._diagnostics.popitem(last=False)
        self._diagnostics.move_to_end(sim_id)
        return d

    def request_trace(self, sim_id: str | None = None) -> bool:
        """Ask the running solver to capture the stages of its next step (``jobTrace`` delivers them).

        Goes through the shared view settings (``capture_trace``), which the
        runner reads every frame; returns False when nothing (or another
        simulation) runs. A solver without trace support simply never answers.
        """
        rid = self.running_id()
        if rid is None or (sim_id is not None and sim_id != rid):
            return False
        try:
            self.view.capture_trace = True
        except (AttributeError, TypeError):
            return False
        return True

    # -- control ----------------------------------------------------------
    @staticmethod
    def _has_result(project, sim_id: str) -> bool:
        try:
            return bool(getattr(project, "has_result", lambda _s: False)(sim_id))
        except Exception:  # noqa: BLE001 - a store that cannot tell has nothing to protect
            return False

    def submit(self, sim_id: str) -> None:
        """Queue a simulation of the open project; starts at once if nothing is running."""
        project = self.state.project
        if project is None:
            raise RuntimeError("no project is open")
        sim = project.simulations.get(sim_id)
        if sim is None:
            raise KeyError(f"unknown simulation {sim_id!r}")
        if self.is_busy(sim_id):
            return
        # recorded before the status becomes "queued": a stopped "Run again" must keep the
        # complete result this simulation already has (see the module docstring)
        prior = sim.status
        keep = prior == "done" and self._has_result(project, sim_id) and sim_id not in self._unsaved
        self._queue.append(_Job(sim_id, project, prior_status=prior, keep_previous=keep))
        sim.status = "queued"
        sim.error = None
        self._update_sim(project, sim)
        self.jobQueued.emit(sim_id)
        self.queueChanged.emit()
        self._start_next()

    def cancel(self, sim_id: str) -> None:
        """Remove a queued job, or ask the running one to stop (it returns a partial result).

        A job removed from the queue never ran, so its simulation gets back the
        status it had before it was queued ("done" with its result, "new", ...).
        """
        for job in list(self._queue):
            if job.sim_id == sim_id:
                self._queue.remove(job)
                sim = job.project.simulations.get(sim_id)
                if sim is not None:
                    sim.status = self._unqueued_status(job)
                    self._update_sim(job.project, sim)
                self.jobCancelled.emit(sim_id)
                self.queueChanged.emit()
                return
        if self._running is not None and self._running.sim_id == sim_id:
            self._running.cancel.set()
            self._running.message = "Stopping…"
            self.jobProgress.emit(sim_id, self._running.fraction, "Stopping…")

    def _unqueued_status(self, job: _Job) -> str:
        """The status a simulation goes back to when it leaves the queue without running."""
        prior = job.prior_status if job.prior_status not in ("queued", "running") else "new"
        if prior in ("done", "cancelled") and not self._has_result(job.project, job.sim_id):
            return "new"
        return prior

    def cancel_all(self) -> None:
        for job in list(self._queue):
            self.cancel(job.sim_id)
        if self._running is not None:
            self._running.cancel.set()

    def _drop_queue(self) -> None:
        """Empty the queue, giving each waiting simulation back its status before it was queued."""
        if not self._queue:
            return
        for job in list(self._queue):
            sim = job.project.simulations.get(job.sim_id)
            if sim is not None and sim.status == "queued":
                sim.status = self._unqueued_status(job)
                self._update_sim(job.project, sim)
        self._queue.clear()
        self.queueChanged.emit()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel everything, join the worker and record its outcome (call on window close).

        The worker reports through a queued signal, which is only delivered
        when the event loop runs again; during application exit it never does.
        The partial result was already saved by the worker, but the simulation
        would stay "running" in project.json, and project recovery would later
        promote it to "done" - a run stopped after a second showing a green
        check. So once the worker has finished, its outcome is processed here
        synchronously (status "cancelled", ``jobCancelled`` emitted) before the
        caller saves the project; the queued signal that arrives afterwards, if
        any, is ignored. Queued simulations get back their earlier status, and
        results still waiting for a save get one last synchronous attempt.
        """
        self._drop_queue()
        job = self._running
        if job is not None:
            job.cancel.set()
            if job.thread is not None:
                job.thread.join(timeout)
                if job.thread.is_alive():
                    log.warning("simulation %s did not stop within %.0f s", job.sim_id, timeout)
            if job.outcome is not None and not job.handled and self._running is job:
                result, error = job.outcome
                self._on_done(job.sim_id, result, error)
        self._final_saves()

    # -- results that could not be saved ------------------------------------
    def unsaved_ids(self) -> list[str]:
        """Simulations whose finished result is held in memory because saving it failed."""
        return list(self._unsaved)

    def retry_saves(self) -> None:
        """Try again, now, to save every result that could not be saved (Simulation menu)."""
        for sid, entry in list(self._unsaved.items()):
            entry.attempts = 0
            self._save_again(sid)

    def _schedule_save_retry(self, sim_id: str) -> None:
        entry = self._unsaved.get(sim_id)
        if entry is None:
            return
        if entry.attempts >= len(SAVE_RETRY_DELAYS_S):
            sim = entry.project.simulations.get(sim_id)
            name = sim.name if sim is not None else sim_id
            self.state.log(f"The result of “{name}” is still not saved: the project folder refuses the write. "
                           "Close programs that may hold the file (OneDrive, a viewer, a virus scanner) and use "
                           "Simulation ▸ Save unsaved results; it is lost if Aerolume quits first.",
                           "error", LOG_SOURCE)
            return
        delay = SAVE_RETRY_DELAYS_S[entry.attempts]
        QTimer.singleShot(int(1000 * delay), self, lambda s=sim_id: self._save_again(s))

    def _save_again(self, sim_id: str) -> None:
        """One background attempt to save an unsaved result (never on the GUI thread:
        packing and compressing the fields takes about half a second, and the store's
        lock is shared with the GUI thread's own saves)."""
        entry = self._unsaved.get(sim_id)
        if entry is None or entry.busy:
            return
        entry.busy = True
        entry.attempts += 1

        def save() -> bool:
            # a newer run of the same simulation supersedes this result: never write over it
            if entry.superseded:
                return False
            entry.project.save_result(sim_id, entry.result)
            return True

        run_in_background(save, on_done=lambda saved, e=entry: self._saved_late(sim_id, e, saved),
                          on_error=lambda msg, e=entry: self._save_failed_again(sim_id, e, msg))

    def _drop_unsaved(self, sim_id: str) -> None:
        """Forget a held result because a newer run of the simulation replaces it."""
        entry = self._unsaved.pop(sim_id, None)
        if entry is not None:
            entry.superseded = True

    def _saved_late(self, sim_id: str, entry: _Unsaved, saved: bool = True) -> None:
        if self._unsaved.get(sim_id) is not entry or not saved:
            return
        self._unsaved.pop(sim_id, None)
        project = entry.project
        sim = project.simulations.get(sim_id)
        if sim is not None:
            sim.error = None
            self._update_sim(project, sim)
            self.state.log(f"The result of “{sim.name}” is now saved.", "info", LOG_SOURCE)
        if project is self.state.project:
            self._results[sim_id] = entry.result
            self._keep_cached = sim_id
            try:
                self.state.notify_result(sim_id)
            finally:
                self._keep_cached = None

    def _save_failed_again(self, sim_id: str, entry: _Unsaved, message: str) -> None:
        if self._unsaved.get(sim_id) is not entry:
            return
        entry.busy = False
        log.warning("saving the result of %s failed again: %s", sim_id, message)
        if sim_id not in entry.project.simulations:
            self._unsaved.pop(sim_id, None)    # the simulation was deleted: nothing to save into
            return
        self._schedule_save_retry(sim_id)

    def _final_saves(self) -> None:
        """At shutdown: one synchronous save of each unsaved result; say what was lost."""
        for sid, entry in list(self._unsaved.items()):
            project, result = entry.project, entry.result
            sim = project.simulations.get(sid)
            try:
                project.save_result(sid, result)
            except Exception as exc:  # noqa: BLE001 - the window is closing either way
                log.error("the result of %s could not be saved at shutdown: %s", sid, exc)
                if sim is not None:
                    sim.error = ("The result of the last run could not be saved; "
                                 + ("the stored result is from an earlier run." if self._has_result(project, sid)
                                    else "run it again."))
                    self._update_sim(project, sim)
                continue
            self._unsaved.pop(sid, None)
            if sim is not None:
                sim.error = None
                self._update_sim(project, sim)

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until the queue is empty, processing Qt events (tests and the CLI)."""
        end = time.monotonic() + timeout
        while (self._running is not None or self._queue) and time.monotonic() < end:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        QCoreApplication.processEvents()
        return self._running is None and not self._queue

    # -- results cache ----------------------------------------------------
    def result(self, sim_id: str):
        """Load (and cache) the stored result of a simulation; None if there is none.

        Shared by the Results, Compare and Report pages so switching between them
        does not re-read tens of MB of fields. ``result_error(sim_id)`` tells why a
        load failed.
        """
        pending = self._unsaved.get(sim_id)
        if pending is not None and pending.project is self.state.project:
            return pending.result               # newer than anything on disk (its save failed)
        if sim_id in self._results:
            self._results.move_to_end(sim_id)
            return self._results[sim_id]
        project = self.state.project
        if project is None or sim_id not in project.simulations:
            return None
        try:
            res = project.load_result(sim_id)
        except (FileNotFoundError, KeyError):
            res = None
        except Exception as exc:
            log.exception("could not load result of %s", sim_id)
            self._result_errors[sim_id] = friendly_error(exc)
            return None
        self._result_errors.pop(sim_id, None)
        if res is not None:
            self._results[sim_id] = res
            while len(self._results) > self.RESULT_CACHE_SIZE:
                self._results.popitem(last=False)
        return res

    def cached_result(self, sim_id: str):
        """The result if it is already in memory, else None (never reads the disk)."""
        pending = self._unsaved.get(sim_id)
        if pending is not None and pending.project is self.state.project:
            return pending.result
        return self._results.get(sim_id)

    def result_error(self, sim_id: str) -> str:
        return self._result_errors.get(sim_id, "")

    def _invalidate_result(self, sim_id: str) -> None:
        if sim_id == self._keep_cached:  # our own notify_result: the cache holds the new result
            return
        self._results.pop(sim_id, None)
        self._result_errors.pop(sim_id, None)

    # -- internals (GUI thread) -------------------------------------------
    def _update_sim(self, project, sim: Simulation) -> bool:
        """Record a status change and save it; never raises (returns False when the save failed).

        The status is bookkeeping around a run, and an exception here used to
        stop the queue for good: ``_on_done`` raised before it emitted
        jobFinished or started the next job, so the UI never learnt that the run
        had finished and the next simulation stayed "queued" forever. Saving
        fails when OneDrive or a virus scanner holds simulation.json for more
        than the store's retry, the disk is full or the folder was moved. The new
        status is already in memory then and goes to disk with the next
        successful save; the user is told in Messages.
        """
        try:
            if project is self.state.project:
                self.state.update_simulation(sim)
            else:  # the user switched projects meanwhile: write to the job's own project
                project.update_simulation(sim)
                project.save()
            return True
        except Exception as exc:
            log.exception("could not save the status of %s", getattr(sim, "id", "?"))
            self.state.log(f"The status of “{getattr(sim, 'name', '?')}” could not be saved: "
                           f"{friendly_error(exc)}. It is kept in memory and written with the next save.",
                           "error", LOG_SOURCE)
            if project is self.state.project:
                # AppState.update_simulation skips this when its save() raises
                self.state.simulationsChanged.emit()
            return False

    def _on_project_changed(self) -> None:
        # Results of queued jobs would land in a project that is no longer open.
        self._results.clear()
        self._result_errors.clear()
        self._drop_queue()          # leave the old project consistent: these never ran
        if self._running is not None:
            self._running.cancel.set()

    def _start_next(self) -> None:
        if self._running is not None:
            return
        while self._queue:
            job = self._queue.popleft()
            project = job.project
            sim = project.simulations.get(job.sim_id)
            design = project.designs.get(sim.design_id) if sim is not None else None
            if sim is None or design is None:
                self.jobFailed.emit(job.sim_id, "The simulation or its design was deleted.")
                continue
            job.sim = clone_simulation(sim)
            job.design = clone_design(design)
            job.airfoils = airfoil_designs(project, design)
            job.started = time.monotonic()
            job.message = "Starting…"
            if getattr(design, "dimension", 2) == 3 and getattr(sim, "analysis", "cfd") == "cfd":
                # this case's smoke rake, not the one of whatever case a page showed last
                self._set_view_rake(self.seed_rake_for(sim))
            # a new run: its own diagnostics from step 0, and no trace request left over from the last one
            try:
                self.diagnostics_for(job.sim_id).clear(source=sim.name)
            except Exception:
                log.debug("no diagnostics log for %s", job.sim_id, exc_info=True)
            with contextlib.suppress(AttributeError, TypeError):
                self.view.capture_trace = False
            # running and started before the status is saved: a save that fails must never
            # drop a job that has already left the queue
            self._running = job
            self._frames.take()
            job.thread = threading.Thread(target=self._work, args=(job,),
                                          name=f"aerolume-job-{job.sim_id}", daemon=True)
            job.thread.start()
            sim.status = "running"
            self._update_sim(project, sim)
            self.jobStarted.emit(job.sim_id)
            self.queueChanged.emit()
            return
        self.queueChanged.emit()

    def _on_progress(self, sim_id: str, fraction: float, message: str) -> None:
        if self._running is not None and self._running.sim_id == sim_id:
            self._running.fraction, self._running.message = fraction, message
        self.jobProgress.emit(sim_id, fraction, message)

    def _on_planned(self, sim_id: str, plan) -> None:
        if self._running is not None and self._running.sim_id == sim_id:
            self._running.plan = plan
            self.jobPlanned.emit(sim_id, plan)

    def _on_frame(self) -> None:
        item = self._frames.take()
        if item is not None:
            self.jobFrame.emit(*item)

    def _on_done(self, sim_id: str, result, error) -> None:
        job = self._running
        if job is None or job.sim_id != sim_id or job.handled:
            # already finished synchronously by shutdown(): this is the late queued signal
            log.debug("ignoring a stale completion of %s", sim_id)
            return
        job.handled = True
        self._running = None
        if job.started:
            self._job_seconds[sim_id] = time.monotonic() - job.started
        try:
            self._finish(job, sim_id, result, error)
        except Exception:
            log.exception("finishing the job of %s failed", sim_id)
        finally:
            self.queueChanged.emit()
            self._start_next()

    def _finish(self, job: _Job, sim_id: str, result, error) -> None:
        """Status, cache and signals for a job that has ended (see :meth:`_on_done`)."""
        project = job.project
        cancelled = job.stopped if job.outcome is not None else job.cancel.is_set()
        sim = project.simulations.get(sim_id) if project is not None else None
        name = sim.name if sim is not None else sim_id
        if error is not None:
            message, _tb = error  # the traceback was logged on the worker thread
            if sim is not None:
                if cancelled:
                    # stopped: back to the complete result it had, else "cancelled"
                    sim.status = "done" if job.keep_previous and self._has_result(project, sim_id) else "cancelled"
                else:
                    sim.status = "failed"
                sim.error = None if cancelled else message
                self._update_sim(project, sim)
            if cancelled:
                self.jobCancelled.emit(sim_id)
            else:
                self.jobFailed.emit(sim_id, message)
            return
        if job.kept_previous:
            # a stopped re-run: the partial result was discarded and the complete one kept
            if sim is not None:
                sim.status = "done"
                sim.error = None
                self._update_sim(project, sim)
            self._invalidate_result(sim_id)
            self.state.log(f"“{name}” stopped; its previous result was kept.", "info", LOG_SOURCE)
            if project is self.state.project:
                self.state.notify_result(sim_id)     # pages drop the partial run's frames
            self.jobCancelled.emit(sim_id)
            return
        self._drop_unsaved(sim_id)              # this run's result replaces any result still held
        if sim is not None:
            sim.status = "cancelled" if cancelled else "done"
            sim.finished = now_iso()
            sim.error = f"Result not saved yet: {job.save_error}" if job.save_error else None
            self._update_sim(project, sim)
        self._invalidate_result(sim_id)
        if job.save_error:
            self._unsaved[sim_id] = _Unsaved(project, result)
            self.state.log(f"The run of “{name}” finished but its result could not be saved "
                           f"({job.save_error}). It is kept in memory and saved again in the background.",
                           "warning", LOG_SOURCE)
            self._schedule_save_retry(sim_id)
        if project is self.state.project:
            # pages reload on resultChanged: hand them the in-memory result instead of
            # re-reading the file that was just written
            self._results[sim_id] = result
            self._keep_cached = sim_id
            try:
                self.state.notify_result(sim_id)
            finally:
                self._keep_cached = None
        if cancelled:
            self.jobCancelled.emit(sim_id)
        else:
            self.jobFinished.emit(sim_id, result)

    # -- worker thread ----------------------------------------------------
    def _work(self, job: _Job) -> None:
        sim_id = job.sim_id

        def progress(fraction: float, message: str = "") -> None:
            now = time.monotonic()
            if fraction >= 1.0 or now - self._last_progress_emit >= PROGRESS_MIN_INTERVAL_S:
                self._last_progress_emit = now
                self._bridge.progress.emit(sim_id, float(fraction), str(message))

        diag = self._diagnostics.get(sim_id)

        def frame(f) -> None:
            # diagnostics and a captured trace are taken from *every* frame, before the rate limit and the
            # one-slot mailbox drop any: a sample or a trace lost there would be lost for good
            if diag is not None:
                try:
                    diag.add_frame(f)
                except Exception:
                    log.debug("diagnostics sample rejected", exc_info=True)
            tr = frame_trace(f)
            if tr is not None:
                self._bridge.trace.emit(sim_id, tr)
            now = time.monotonic()
            if (now - self._last_frame_post < FRAME_MIN_INTERVAL_S and getattr(f, "phase", "") != "done"
                    and getattr(f, "volume", None) is None and getattr(f, "streamlines3d", None) is None):
                # the final frame is never dropped (it is what stays on screen), nor one carrying a flow volume
                return
            self._last_frame_post = now
            if self._frames.put((sim_id, f)):
                self._bridge.frame.emit()

        fn = self.runner or execute_simulation
        extra = {}
        if _accepts_planned(fn):
            extra["planned"] = lambda plan: self._bridge.planned.emit(sim_id, plan)
        if job.airfoils and _accepts(fn, "airfoils"):
            extra["airfoils"] = job.airfoils

        def fail(exc: BaseException) -> None:
            tb = traceback.format_exc()
            log.error("simulation %s failed:\n%s", sim_id, tb)
            job.stopped = job.cancel.is_set()
            job.outcome = (None, (friendly_error(exc), tb))
            self._bridge.done.emit(sim_id, None, job.outcome[1])

        try:
            result = fn(job.design, job.sim, progress=progress, frame=frame, cancel=job.cancel,
                        view=self.view, **extra)
            if result is None:
                raise RuntimeError("The solver finished without returning a result.")
        except Exception as exc:  # noqa: BLE001 - every failure becomes a readable job error
            fail(exc)
            return
        # Checked right before saving, so a run that completed just before the stop
        # request still saves normally: a stopped re-run of a simulation that had a
        # complete result must not overwrite it with its partial one.
        if job.cancel.is_set() and job.keep_previous:
            job.kept_previous = job.stopped = True
            job.outcome = (result, None)
            self._bridge.done.emit(sim_id, result, None)
            return
        attach_diagnostics_history(result, diag)
        try:
            job.project.save_result(sim_id, result)
        except KeyError:
            # the simulation was deleted while it ran: there is nothing to keep the result in
            try:
                raise RuntimeError("The simulation was deleted while it ran; its result was discarded.") from None
            except RuntimeError as exc:
                fail(exc)
            return
        except Exception as exc:
            log.exception("saving the result of %s failed", sim_id)
            job.save_error = friendly_error(exc)
        job.stopped = job.cancel.is_set()
        job.outcome = (result, None)
        self._bridge.done.emit(sim_id, result, None)


# ---------------------------------------------------------------------------
# one-off background calls
# ---------------------------------------------------------------------------

_ALIVE: set = set()


class BackgroundCall(QObject):
    """Run ``fn(*args, **kw)`` on a thread; ``finished(result)`` / ``failed(message)`` in the GUI thread.

    For short, self-contained work that would otherwise stall the GUI: planning
    a run (imports torch the first time), computing streamlines, writing a
    report. The object keeps itself alive until it has delivered its outcome, so
    callers can fire and forget.
    """

    finished = Signal(object)
    failed = Signal(str)
    _done = Signal(object, object)

    def __init__(self, fn, *args, **kw) -> None:
        super().__init__()
        self._fn, self._args, self._kw = fn, args, kw
        self._done.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        self.cancelled = False

    def start(self) -> BackgroundCall:
        _ALIVE.add(self)
        threading.Thread(target=self._run, name="aerolume-bg", daemon=True).start()
        return self

    def _run(self) -> None:
        try:
            out, err = self._fn(*self._args, **self._kw), None
        except Exception as exc:  # noqa: BLE001 - reported through failed(message)
            log.debug("background call failed:\n%s", traceback.format_exc())
            out, err = None, friendly_error(exc)
        try:
            self._done.emit(out, err)
        except RuntimeError:
            # the application quit while this ran (Qt deleted the object): nobody is left to tell
            _ALIVE.discard(self)

    def _deliver(self, out, err) -> None:
        _ALIVE.discard(self)
        if self.cancelled:
            return
        if err is None:
            self.finished.emit(out)
        else:
            self.failed.emit(err)


def run_in_background(fn, *args, on_done=None, on_error=None, **kw) -> BackgroundCall:
    """Start a :class:`BackgroundCall`; set ``.cancelled = True`` to drop its outcome."""
    call = BackgroundCall(fn, *args, **kw)
    if on_done is not None:
        call.finished.connect(on_done)
    if on_error is not None:
        call.failed.connect(on_error)
    return call.start()


# ---------------------------------------------------------------------------
# LiveWorker (explore mode)
# ---------------------------------------------------------------------------

def _default_session_factory(design, sim, *, device=None, view=None, airfoils=None):
    """A ``LiveSession`` (a ``LiveSession3D`` for 3D designs; wings resolve project airfoils)."""
    from ..simulation.live import LiveSession
    kw = {}
    if getattr(design, "dimension", 2) == 3:
        coords = _airfoil_coords(airfoils)
        if coords is not None:
            kw["airfoil_coords"] = coords
    return LiveSession(design, sim, device=device, view=view, **kw)


#: Live settings the explore worker can carry out itself when a session does not accept them (older
#: sessions, DESIGN.md 10.4): one solver step while paused, a new kinematic viscosity (the Reynolds
#: lock), and drawn obstacles (2D) as extra immersed bodies.
WORKER_FALLBACK_KEYS = frozenset({"step", "nu", "obstacles"})


def accepted_post_keys(session) -> frozenset | None:
    """The ``post`` keys a live session accepts, or None when it does not say.

    A session may list them (``POST_KEYS``); the explore sessions of
    :mod:`aerolume.simulation.live` / ``live3d`` keep them in module constants
    (``_POST_KEYS``, ``_POST_KEYS3``), which are read as a fallback. Knowing
    them up front lets the page disable a control the running solver cannot
    honour, with the reason in its tooltip, instead of failing on use.
    """
    if session is None:
        return None
    keys = getattr(session, "POST_KEYS", None) or getattr(type(session), "POST_KEYS", None)
    if keys:
        return frozenset(keys)
    import sys
    found: set = set()
    for cls in type(session).__mro__:
        mod = sys.modules.get(getattr(cls, "__module__", ""))
        for name in ("_POST_KEYS", "_POST_KEYS3", "POST_KEYS"):
            v = getattr(mod, name, None) if mod is not None else None
            if isinstance(v, (set, frozenset)):
                found |= set(v)
        if found:
            break
    return frozenset(found) if found else None


class LiveWorker(QObject):
    """Runs a ``LiveSession`` tick loop on a worker thread for explore mode.

    The GUI calls :meth:`post` with changes (``alpha_deg``, ``speed``, ``field`` ...)
    at any time; they are forwarded to the session, which is thread-safe for
    ``post`` by contract. Changes posted before the session finished building
    (CUDA initialisation can take a second) are buffered and replayed. Frames
    arrive through ``frameReady`` in the GUI thread, newest only.

    Live editing (DESIGN.md 10.4) goes through the same ``post``. Keys the
    session refuses are split off one by one (:func:`accepted_post_keys` says
    which up front, :meth:`supports` tells the page), so one unknown key never
    costs the others; :data:`WORKER_FALLBACK_KEYS` are then carried out here,
    on the ticking thread (the only thread that touches the solver):

    * ``step=n`` while paused: ``solver.advance(n)``, then a render;
    * ``nu`` (m²/s): the session's own fluid gets that viscosity and its speed
      is re-posted, so the session recomputes the Reynolds number by its own
      rules (a capped run stays capped);
    * ``obstacles`` (a :meth:`~aerolume.app.widgets.draw_tools.ObstacleSketch.payload`)
      on a 2D session: the traced outlines become extra immersed bodies next to
      the design's, re-applied after the session itself rebuilds its bodies (a
      pitch change, a road switch). The coefficients then include the drawn
      obstacles, as they do in AeroJAX; the page says so.

    ``diagnostics`` (a :class:`~aerolume.app.widgets.diagnostics.DiagnosticsLog`)
    is fed from every tick before frames are coalesced; a frame carrying a
    captured step (``LiveFrame.trace``) is also delivered through ``traceReady``.
    """

    frameReady = Signal(object)
    started = Signal()
    failed = Signal(str)
    stopped = Signal()
    traceReady = Signal(object)

    #: How long :meth:`stop` waits for the loop by default [s]. A ticking loop ends within
    #: one tick (~33 ms); only a session still being built (CUDA initialisation, seconds
    #: on a slow machine) takes longer, and the GUI must not freeze for that. Such a
    #: thread is left to finish on its own: it sees that it is stale and closes its
    #: session without touching the worker's state.
    STOP_WAIT_S = 0.5
    #: Largest single step request [steps].
    MAX_STEP = 10_000

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.session_factory = _default_session_factory
        self.fps = 30.0
        self._thread: threading.Thread | None = None
        self._detached: list[threading.Thread] = []    # stopped loops that had not ended yet
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._session = None
        self._pending: list[dict] = []
        self._requests: deque = deque()                 # worker-side fallbacks, run on the ticking thread
        self._refused: set[str] = set()                 # keys the current session refused
        self._paused = False
        self._obstacles: dict | None = None             # the drawing last applied through the fallback
        self._obstacle_state: tuple | None = None       # (solver id, alpha) it was applied to
        self.obstacle_error = ""                        # why the fallback could not apply it ("" = fine)
        self._frames = _Mailbox(merge=carry_volume)
        self.diagnostics = _new_diagnostics_log("Explore")
        self.last_trace = None
        self._bridge = _Bridge()
        q = Qt.ConnectionType.QueuedConnection
        self._bridge.live_frame.connect(self._deliver_frame, q)
        self._bridge.live_started.connect(self.started, q)
        self._bridge.live_failed.connect(self.failed, q)
        self._bridge.live_stopped.connect(self._on_stopped, q)
        self._bridge.trace.connect(self._on_trace, q)
        app = QCoreApplication.instance()
        if app is not None:  # close the session (and free the GPU) even if no page asks
            app.aboutToQuit.connect(self.shutdown)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def plan(self):
        """The running explore session's :class:`SimPlan` (its own, smaller grid), or None.

        The Simulate page shows it while explore runs: the production plan of the
        same simulation (grid, run-time estimate, its warnings) does not describe
        what is on screen then.
        """
        with self._lock:
            session = self._session
        return getattr(session, "plan", None) if session is not None and self.is_running() else None

    def session_dimension(self) -> int | None:
        """2 or 3 for the running session (its plan's dimension), None before it exists."""
        plan = self.plan()
        return int(getattr(plan, "dimension", 2) or 2) if plan is not None else None

    def start(self, design: Design, sim: Simulation, *, device: str | None = None, view=None,
              airfoils: dict | None = None) -> None:
        """Build a session for (copies of) design + sim and start ticking. Restarts if running.

        ``airfoils`` ({id: Design}, :func:`airfoil_designs`) resolves the project
        airfoils of a 3D wing; it is handed to factories that take it.
        """
        self.stop()
        with self._lock:
            self._stop = threading.Event()
            self._pending = []
            self._requests.clear()
            self._refused = set()
            self._paused = False
            self._obstacles = None
            self._obstacle_state = None
            self.obstacle_error = ""
        self._frames.take()  # a frame of the previous session must not flash up
        name = str(getattr(sim, "name", ""))
        self.diagnostics.clear(source=name if "explore" in name.lower() else f"Explore · {name}")
        self.last_trace = None
        d = clone_design(design)
        s = clone_simulation(sim)
        extra = {"airfoils": airfoils} if airfoils and _accepts(self.session_factory, "airfoils") else {}
        self._thread = threading.Thread(target=self._loop, args=(d, s, device, view, self._stop, extra),
                                        name="aerolume-live", daemon=True)
        self._thread.start()

    # -- posting ------------------------------------------------------------
    def supports(self, key: str) -> bool | None:
        """Whether the running session honours ``post(key=...)``: True / False, None while unknown.

        Worker fallbacks count as supported (``obstacles`` only on a 2D session).
        """
        with self._lock:
            session = self._session
        acc = accepted_post_keys(session)
        if key == "obstacles" and self.session_dimension() == 3:
            return acc is not None and key in acc
        if key in WORKER_FALLBACK_KEYS:
            return True
        if key in self._refused:
            return False
        if session is None:
            return None
        return None if acc is None else key in acc

    def post(self, **changes) -> None:
        """Forward parameter / view changes to the session (buffered until it exists)."""
        if "paused" in changes:
            self._paused = bool(changes["paused"])
        if "fps" in changes:
            try:
                self.fps = max(1.0, min(240.0, float(changes["fps"])))
            except (TypeError, ValueError):
                changes.pop("fps")
        with self._lock:
            session = self._session
            if session is None:
                self._pending.append(changes)
                return
        self._post_to(session, changes)

    def _post_to(self, session, changes: dict) -> None:
        """Post what the session takes; refused fallback keys go to the worker queue, others to the log."""
        acc = accepted_post_keys(session)
        direct = {k: v for k, v in changes.items() if (acc is None or k in acc) and k not in self._refused}
        rest = {k: v for k, v in changes.items() if k not in direct}
        if direct:
            try:
                session.post(**direct)
            except TypeError:
                # some key is unknown to this session: try them one by one, keep what it takes
                for k, v in direct.items():
                    try:
                        session.post(**{k: v})
                    except TypeError:
                        self._refused.add(k)
                        rest[k] = v
                    except Exception:
                        log.exception("live session rejected %s=%r", k, v)
            except Exception:
                log.exception("live session rejected %s", changes)
        for k, v in rest.items():
            if k in WORKER_FALLBACK_KEYS:
                self._requests.append((k, v))
            else:
                log.debug("the live session does not support %r; ignored", k)

    def step(self, n: int = 1) -> None:
        """Advance ``n`` steps and render (meant for a paused session: frame-by-frame inspection)."""
        self.post(step=int(max(1, min(int(n), self.MAX_STEP))))

    def set_fps(self, fps: float) -> None:
        """Target frame rate of the explore loop (the session paces its steps to it)."""
        self.post(fps=float(fps))

    def request_trace(self) -> bool:
        """Ask the session to capture its next step (``traceReady`` delivers it); False if it cannot."""
        if self.supports("capture_trace") is False:
            return False
        self.post(capture_trace=True)
        return self.supports("capture_trace") is not False

    def stop(self, timeout: float | None = None) -> None:
        """Stop the loop and close the session.

        Waits at most ``timeout`` seconds (default :data:`STOP_WAIT_S`) for the loop
        to end; a loop still building its session is detached instead and closes
        that session itself once it exists (see :meth:`_loop`).
        """
        th = self._thread
        if th is None:
            return
        self._stop.set()
        if th is not threading.current_thread():
            th.join(self.STOP_WAIT_S if timeout is None else timeout)
        if th.is_alive():
            self._detached.append(th)
        self._thread = None

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop and wait for every loop, detached ones included (application exit)."""
        self.stop(timeout)
        end = time.monotonic() + timeout
        for th in self._detached:
            if th is not threading.current_thread():
                th.join(max(0.0, end - time.monotonic()))
        self._detached = [th for th in self._detached if th.is_alive()]

    def _deliver_frame(self) -> None:
        f = self._frames.take()
        if f is not None:
            self.frameReady.emit(f)

    def _on_stopped(self) -> None:
        self.stopped.emit()

    def _on_trace(self, _sid: str, trace) -> None:
        self.last_trace = trace
        self.traceReady.emit(trace)

    # -- worker-side fallbacks (ticking thread only) --------------------------
    def _run_requests(self, session) -> None:
        """Carry out the queued fallback requests (see the class docstring)."""
        while self._requests:
            key, value = self._requests.popleft()
            try:
                if key == "step":
                    if self._paused and getattr(session, "solver", None) is not None:
                        session.solver.advance(int(value))
                elif key == "nu":
                    nu = float(value)
                    fluid = session.sim.conditions.fluid
                    if nu > 0 and math.isfinite(nu):
                        fluid.viscosity = nu * float(fluid.density)
                        session.post(speed=float(getattr(session, "speed", session.sim.conditions.speed)))
                elif key == "obstacles":
                    self._apply_obstacles(session, value)
            except Exception as exc:
                log.exception("live %s fallback failed", key)
                if key == "obstacles":
                    self.obstacle_error = friendly_error(exc)

    #: Name prefix of the immersed bodies the obstacle fallback adds (how it finds them again).
    DRAWN_PREFIX = "Drawn "

    def _apply_obstacles(self, session, payload: dict | None) -> None:
        """Drawn obstacles as extra immersed bodies of a 2D session (see the class docstring).

        The design's own bodies are whatever the solver has now minus the ones
        this fallback added (:data:`DRAWN_PREFIX`), so the session stays the
        single authority on where the design is (pitch, road, spinning wheels).
        """
        self._obstacles = payload if payload and payload.get("polygons") else None
        L = float(session.plan.length_ref)
        solver = session.solver
        current = list(getattr(getattr(solver, "cfg", None), "bodies", None) or getattr(solver, "_bodies", []) or [])
        base = [b for b in current if not str(getattr(b, "name", "")).startswith(self.DRAWN_PREFIX)]
        outlines = [np.asarray(p, dtype=np.float64) for p in list(getattr(session, "_bodies", []) or [])[:len(base)]]
        if base:
            make = type(base[0])
        else:
            from ..solvers.ns2d import ImmersedBody as make
        bodies = list(base)
        for i, poly in enumerate((self._obstacles or {}).get("polygons", []) or []):
            p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            if len(p) >= 3:
                bodies.append(make(points=p / L, name=f"{self.DRAWN_PREFIX}{i + 1}", role="body"))
                outlines.append(p)
        solver.set_bodies(bodies)
        session._bodies = outlines
        session.renderer.invalidate_geometry()
        self.obstacle_error = ""
        self._obstacle_state = (id(session.solver), float(session.alpha_deg))

    def _keep_obstacles(self, session) -> None:
        """Re-apply drawn obstacles after the session rebuilt its own bodies (pitch change, road switch)."""
        if self._obstacles is None or self.session_dimension() == 3:
            return
        solver = getattr(session, "solver", None)
        state = (id(solver), float(getattr(session, "alpha_deg", 0.0)))
        bodies = getattr(getattr(solver, "cfg", None), "bodies", None)
        lost = bodies is not None and not any(str(getattr(b, "name", "")).startswith(self.DRAWN_PREFIX)
                                              for b in bodies)
        if state != self._obstacle_state or lost:
            try:
                self._apply_obstacles(session, self._obstacles)
            except Exception as exc:
                log.exception("re-applying drawn obstacles failed")
                self.obstacle_error = friendly_error(exc)
                self._obstacles = None

    def _loop(self, design, sim, device, view, stop: threading.Event, extra: dict | None = None) -> None:
        # ``stop`` is this run's own event: after a restart, self._stop is a new one, and a
        # finishing old loop must not announce "stopped" / "failed" over the new session
        def current() -> bool:
            return stop is self._stop

        try:
            session = self.session_factory(design, sim, device=device, view=view, **(extra or {}))
        except Exception as exc:  # noqa: BLE001 - reported through failed(message)
            log.error("explore mode could not start:\n%s", traceback.format_exc())
            if current():
                self._bridge.live_failed.emit(friendly_error(exc))
                self._bridge.live_stopped.emit()
            return
        with self._lock:
            # A loop stopped (or replaced by a restart) while its session was being built
            # must not install it: it would take the new session's slot and its pending
            # slider changes, and its own clean-up would then leave the slot empty, so
            # every later post() was buffered forever while the new session kept running.
            stale = stop.is_set() or not current()
            if not stale:
                self._session = session
                pending, self._pending = self._pending, []
        if stale:
            try:
                session.close()
            except Exception:
                log.exception("closing a stale live session failed")
            if current():               # stopped, not replaced: the page waits for "stopped"
                self._bridge.live_stopped.emit()
            return
        for changes in pending:
            self._post_to(session, changes)
        if current():
            self._bridge.live_started.emit()
        diag = self.diagnostics
        try:
            while not stop.is_set():
                t0 = time.perf_counter()
                budget_ms = 1000.0 / max(self.fps, 1.0)
                if self._requests:
                    self._run_requests(session)
                frame = session.tick(max_ms=budget_ms * 0.8)
                self._keep_obstacles(session)
                if frame is not None:
                    try:
                        diag.add_frame(frame)
                    except Exception:
                        log.debug("diagnostics sample rejected", exc_info=True)
                    tr = frame_trace(frame)
                    if tr is not None and current():
                        self._bridge.trace.emit("", tr)
                if frame is not None and self._frames.put(frame):
                    self._bridge.live_frame.emit()
                # yield the GIL to the GUI thread every tick, and pace to the target fps
                spare = budget_ms / 1000.0 - (time.perf_counter() - t0)
                stop.wait(max(spare, 0.002))
        except Exception as exc:  # noqa: BLE001 - reported through failed(message)
            log.error("explore mode crashed:\n%s", traceback.format_exc())
            if current():
                self._bridge.live_failed.emit(friendly_error(exc))
        finally:
            with self._lock:
                if self._session is session:
                    self._session = None
            try:
                session.close()
            except Exception:
                log.exception("closing the live session failed")
            if current():
                self._bridge.live_stopped.emit()
