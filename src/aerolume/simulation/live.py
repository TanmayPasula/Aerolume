"""Live view types and the real-time explore session.

* :class:`ViewSettings` -- what the user wants to see (field, colour map, range,
  overlays, frame rate). One instance is shared between the GUI and a running
  job; the GUI mutates it and the solver thread reads it every frame.
* :class:`LiveFrame` -- one rendered frame plus everything the Simulate page
  shows next to it (time, coefficients, history, progress, ETA).
* :class:`FrameRenderer` -- turns a :class:`FlowSolver` state into a
  :class:`LiveFrame`. Shared by production runs (:mod:`.runner`) and explore
  mode, so both look the same.
* :class:`LiveSession` -- explore mode: a small, fast grid; angle of attack,
  speed and road settings change while it runs.

Threading model: :meth:`LiveSession.post` may be called from any thread (the
GUI); it only records the request under a lock. :meth:`LiveSession.tick` runs on
one worker thread, applies the pending requests between solver steps, advances
as many steps as fit in its time budget and renders one frame. The solver is
only ever touched from the ticking thread.
"""
from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry import polygon as P
from ..model import Design, Simulation
from .plan import (
    CPU_SOLVER_THREADS,
    LIVE_PITCH_DEG,
    LIVE_PITCH_GROUND_DEG,
    SimPlan,
    plan_live,
    wind_frame_bodies,
)

__all__ = ["FrameRenderer", "LiveFrame", "LiveSession", "ViewSettings", "cpu_threads"]

#: Fields the live view can show (FlowSolver.field names; "temperature" is theta shown in °C).
VIEW_FIELDS = ("speed", "u", "v", "cp", "p", "vorticity", "dye", "nut", "mask")
#: ... and with placed physics: the temperature, the other smoke channels, the divergence
#: (DESIGN.md 10.4: the continuity residual is a diagnostic worth seeing as a field)
PHYSICS_VIEW_FIELDS = ("temperature", "theta", "dye1", "dye2", "dye3", "divergence")
#: Diagnostics columns never sent as rows (bookkeeping the log keeps by itself).
_DIAG_SKIP = ("perf_steps",)
#: Placed-physics readouts (LiveFrame.physics) are refreshed at most this often [s].
READOUT_PERIOD_S = 0.5
#: Points in LiveFrame.history (the plot does not need more).
HISTORY_POINTS = 2000
#: Explore mode keeps this many convective times of force history.
LIVE_HISTORY_T = 40.0
#: Streamlines are pure-Python (~50-85 ms at 1024 x 512): recompute at most this often.
STREAMLINE_PERIOD_S = 1.0
#: Tracer particles in the live view.
N_PARTICLES = 3000
#: Solver threads on CPU (see plan.CPU_SOLVER_THREADS for the measurement).
CPU_THREADS = CPU_SOLVER_THREADS
#: Steps per tick until the step cost has been measured (the first tick also
#: captures the CUDA graphs, so it is not timed).
WARMUP_STEPS = 10


@contextlib.contextmanager
def cpu_threads(device: str):
    """Use :data:`CPU_THREADS` intra-op threads while the solver runs on the CPU.

    ``torch.set_num_threads`` is process-wide, so the previous value is restored
    afterwards. GPU runs are left alone.

    The thread's subnormal mode is restored on exit as well. The CPU solvers
    flush subnormals to zero only while they step and restore the mode
    themselves (:mod:`aerolume.solvers.fpmode`); this is the outer guarantee for
    runs, live sessions and sensitivity sweeps, because flush-to-zero left on
    crashes scipy's ``cKDTree`` (native stack overflow, not an exception) in
    later geometry code on the same thread.
    """
    if not str(device).startswith("cpu"):
        yield
        return
    import torch

    from ..solvers.fpmode import flushing_subnormals
    old = torch.get_num_threads()
    flush_before = flushing_subnormals()
    torch.set_num_threads(max(1, min(CPU_THREADS, old)))
    try:
        yield
    finally:
        torch.set_num_threads(old)
        if flushing_subnormals() != flush_before:
            torch.set_flush_denormal(flush_before)


# ---------------------------------------------------------------------------
# data types (DESIGN.md 5.4)
# ---------------------------------------------------------------------------

@dataclass
class ViewSettings:
    """What the live view shows; mutable, the GUI edits it while a run is going."""
    field: str = "speed"              # "speed","u","v","cp","p","vorticity","dye","nut","mask"
    cmap: str | None = None           # None -> default per field (viz.DEFAULT_CMAP)
    range_mode: str = "auto"          # "auto" (robust, per frame, smoothed) | "fixed"
    vmin: float | None = None         # display units (SI), used when range_mode == "fixed"
    vmax: float | None = None
    particles: bool = False
    streamlines: bool = False
    fps: float = 20.0                 # target frame rate of LiveFrame callbacks
    # 3D runs (DESIGN.md 7.7): the frame is a slice normal to this axis ("y": side view,
    # "z": plan view, "x": cross-flow view) at this position [m] (None: the plan's default)
    slice_axis: str = "y"
    slice_pos: float | None = None
    # 3D view of a 3D run (DESIGN.md 7.13): with view3d the frames also carry a down-sampled
    # volume (LiveFrame.volume) -- the smoke ("dye") when ``smoke``, the velocity and traced
    # streamlines (LiveFrame.streamlines3d) when ``streamlines3d`` -- and ``seed_rake`` (a dict,
    # metres, see aerolume.viz3d.flow; None = the default rake upstream of the body) is where
    # the smoke is released and the streamlines start. Ignored by 2D runs.
    view3d: bool = False
    smoke: bool = True
    streamlines3d: bool = False
    seed_rake: dict | None = None
    # DESIGN.md 10.4: ask the running solver to record the stages of its next step; the renderer
    # puts the capture into the next frame (LiveFrame.trace) and sets this back to False
    capture_trace: bool = False
    # 3D runs with placed physics (DESIGN.md 10): with ``view3d`` the volume also carries the
    # temperature (°C) when ``temperature3d``, and frames carry the emitter tracer particles when
    # ``particles3d``. Ignored by 2D runs, whose frames always carry their particles.
    temperature3d: bool = True
    particles3d: bool = True


@dataclass
class LiveFrame:
    """One rendered frame of a running simulation (all lengths in metres, wind frame)."""
    rgba: np.ndarray                  # (H, W, 4) uint8, row 0 = top (max y)
    extent: tuple[float, float, float, float]
    field: str
    cmap: str
    vmin: float                       # display units
    vmax: float
    units: str
    t: float                          # convective times
    step: int
    steps_per_s: float
    coeffs: dict[str, float]          # latest {"cd", "cl", "cm"}
    history: dict[str, np.ndarray]    # {"t", "cd", "cl", "cm"}, <= HISTORY_POINTS each
    progress: float
    eta_s: float | None
    phase: str                        # "transient" | "averaging" | "done" | "live"
    bodies: list[np.ndarray]
    particles: np.ndarray | None = None
    streamlines: list[np.ndarray] | None = None
    #: suggested default view window (metres): the bodies and their near wake.
    #: Vehicle domains are sized for 5-8 % blockage, so ``extent`` is very tall.
    focus: tuple[float, float, float, float] | None = None
    #: 3D runs only: the slice shown (normal axis and position [m]); ``extent`` is then
    #: (col_min, col_max, row_min, row_max) in the SliceData orientation and ``bodies``
    #: are the meshes' cross-sections in that plane. None for 2D frames.
    slice_axis: str | None = None
    slice_pos: float | None = None
    #: 3D runs with ``ViewSettings.view3d`` (DESIGN.md 7.13): the most recent down-sampled flow
    #: block (VolumeData, SI units, float16; "dye" with ``smoke``, "u", "v", "w" with
    #: ``streamlines3d``, the solid fraction as ``mask``, ``mirrored`` for a half domain), at most
    #: ~128^3 cells and ~20 MB. Volumes are taken less often than frames (their cost is kept to a
    #: small share of the run), so consecutive frames share one object: ``volume_serial`` grows
    #: by one with every new volume, and a viewer updates only when it changed.
    volume: Any = None
    volume_serial: int = 0
    #: traced 3D streamlines (viz3d.flow.StreamlineSet, metres) from the rake, refreshed about
    #: once a second; None without ``streamlines3d``
    streamlines3d: Any = None
    #: the rake the smoke and the streamlines use, resolved (dict, metres; viz3d.flow keys)
    seed_rake: dict | None = None
    # --- placed physics (DESIGN.md 10) ---------------------------------------------------
    #: the smoke channels composited in a "dye" frame (``viz.colormaps.dye_channels`` entries:
    #: field, colour, emitter names); None for a single-channel or non-smoke frame
    dye_channels: list[dict] | None = None
    #: emitter tracer particles: ``particles`` holds their positions (metres; (N, 2) in 2D,
    #: (N, 3) in 3D), ``particle_ids`` the particle emitter of each, ``particle_palette`` one
    #: '#rrggbb' per particle emitter and ``particle_trails`` the streak / time lines (metres)
    particle_ids: np.ndarray | None = None
    particle_palette: list[str] | None = None
    particle_trails: list[np.ndarray] | None = None
    #: the solver diagnostics recorded since the previous frame: column arrays with a "step"
    #: column (``app.widgets.diagnostics.DiagnosticsLog.add_rows`` takes them as they are)
    diagnostics: dict[str, np.ndarray] | None = None
    #: a captured solver step (``ViewSettings.capture_trace``): {"stages": {key: array}, "extent",
    #: "step", "t", "notes"} in solver orientation and units (``physics_run.trace_payload``)
    trace: dict | None = None
    #: the latest placed-physics readouts in SI: "nu", "heat_rate" [W or W/m], "t_max" / "t_min"
    #: [°C], per fan "fan<i>_thrust" / "fan<i>_velocity"
    physics: dict[str, float] | None = None
    #: explore sessions: what the edits since the previous frame could not do, in words (a closed box
    #: has no speed slider, an obstacle outside the domain ...), for the page's message log
    notices: list[str] | None = None


def downsample_history(h: dict[str, np.ndarray], n: int = HISTORY_POINTS) -> dict[str, np.ndarray]:
    """At most ``n`` samples of each series, evenly spaced in index (the solver's
    step is nearly constant, so this is evenly spaced in time), always keeping
    the latest sample so the plot ends at "now"."""
    t = np.asarray(h.get("t", np.zeros(0)))
    if t.size <= n:
        return {k: np.asarray(v).copy() for k, v in h.items()}
    idx = np.unique(np.linspace(0, t.size - 1, n).round().astype(np.int64))
    return {k: np.asarray(v)[idx] for k, v in h.items()}


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

class FrameRenderer:
    """Renders solver states into :class:`LiveFrame` objects.

    Keeps the state that must persist between frames: the smoothed colour
    range (so colours do not flicker), the particle population, the last
    streamlines, the cached solid mask. ``speed``, ``density`` and ``length``
    convert the solver's non-dimensional fields into SI for the colour bar.
    """

    def __init__(self, extent: tuple[float, float, float, float], *, length: float, speed: float,
                 density: float, h: float, focus: tuple[float, float, float, float] | None = None,
                 physics=None, placed=None):
        self.extent = tuple(float(e) for e in extent)
        self.focus = tuple(float(e) for e in focus) if focus is not None else None
        self.length, self.speed, self.density = float(length), float(speed), float(density)
        self.h = float(h)                       # cell size, solver units
        #: placed physics (simulation.physics): the temperature scale and the smoke / particle colours
        self.physics = physics
        self.placed = placed
        self._range: tuple[float, float] | None = None
        self._range_key: tuple | None = None
        self._mask = None
        self._particles = None
        self._particle_t: float | None = None
        self._stream: list[np.ndarray] | None = None
        self._stream_wall = -math.inf
        self._stream_key: tuple | None = None
        self._diag_seen = -math.inf            # last diagnostics step already sent
        self._readout: tuple[float, dict] | None = None   # (wall time, readouts) of the last refresh
        self.plan = None                       # set by the owner: the trace payload needs the plan

    def invalidate_geometry(self) -> None:
        """Call after the bodies changed (new mask, stale streamlines)."""
        self._mask = None
        self._stream = None
        self._stream_wall = -math.inf

    def reset(self) -> None:
        """Forget all inter-frame state (after a solver reset)."""
        self.invalidate_geometry()
        self._range = None
        self._range_key = None
        self._particle_t = None
        self._diag_seen = -math.inf
        self._readout = None
        if self._particles is not None:
            self._particles.seed(fill=True)

    def invalidate_readouts(self) -> None:
        """Refresh the SI readouts on the next frame (after an edit: a new fan has no value yet)."""
        self._readout = None

    def view_fields(self) -> tuple[str, ...]:
        """The fields this run can show: the flow, and with placed physics the temperature and the
        smoke channels it carries."""
        extra: tuple[str, ...] = ("divergence",)
        if self.physics is not None and getattr(self.physics, "thermal", False):
            extra += ("temperature", "theta")
        n = len(getattr(self.placed, "dye_channels", None) or [])
        extra += tuple(f"dye{k}" for k in range(1, n))
        return VIEW_FIELDS + extra

    def _field(self, solver, view: ViewSettings) -> tuple[str, object]:
        # unknown names fall back to speed; "nut" of a laminar run is honestly all zero
        name = view.field if view.field in self.view_fields() else "speed"
        if name == "temperature":
            return name, self.physics.celsius(solver.field("theta"))
        return name, solver.field(name)

    def render(self, solver, view: ViewSettings, *, bodies: list[np.ndarray], phase: str,
               progress: float, eta_s: float | None, steps_per_s: float,
               history: dict[str, np.ndarray] | None = None, coeffs: dict[str, float] | None = None) -> LiveFrame:
        """Render the solver's current state as a frame."""
        from ..viz import auto_range, default_cmap, field_info, render_field
        name, f = self._field(solver, view)
        if self._mask is None:
            self._mask = solver.field("mask")
        mask = self._mask
        info = field_info(name)
        scale = 1.0 if name == "temperature" else (info.scale(self.speed, self.density, self.length) or 1.0)
        cmap = view.cmap or default_cmap(name)
        key = (name, view.range_mode)
        chans = list(getattr(self.placed, "dye_channels", None) or [])
        composite = name == "dye" and len(chans) > 1
        if view.range_mode == "fixed" and view.vmin is not None and view.vmax is not None \
                and view.vmax > view.vmin:
            lo, hi = float(view.vmin) / scale, float(view.vmax) / scale
        elif name in ("dye", "mask", "dye1", "dye2", "dye3"):
            lo, hi = 0.0, 1.0
        else:
            prev = self._range if self._range_key == key else None
            lo, hi = auto_range(f, diverging=info.diverging, mask=mask, prev=prev, smoothing=0.8)
        self._range, self._range_key = (lo, hi), key
        if composite:
            # every smoke colour over the dark background (DESIGN.md 10.1), one channel per colour
            from ..viz import render_dye
            rgba = render_dye([solver.field(ch["field"]) for ch in chans], [ch["color"] for ch in chans], mask=mask)
        else:
            rgba = render_field(f, cmap, lo, hi, mask=None if name == "mask" else mask)

        stream = self._streamlines(solver, mask) if view.streamlines else None
        extra = self._emitter_particles(solver)
        if extra is not None:
            particles = extra[0]
            self._particles, self._particle_t = None, None
        else:
            particles = self._advance_particles(solver, mask) if view.particles else None
            if not view.particles:
                self._particles, self._particle_t = None, None

        if coeffs is None:
            c = solver.coefficients()
            coeffs = {"cd": float(c[0]), "cl": float(c[1]), "cm": float(c[2])}
        hist = downsample_history(history if history is not None else solver.history())
        return LiveFrame(rgba=rgba, extent=self.extent, field=name, cmap=cmap, vmin=lo * scale, vmax=hi * scale,
                         units=info.unit, t=float(solver.t), step=int(solver.steps),
                         steps_per_s=float(steps_per_s), coeffs=coeffs, history=hist, progress=float(progress),
                         eta_s=eta_s, phase=phase, bodies=[np.asarray(b, dtype=np.float64) for b in bodies],
                         particles=particles, streamlines=stream, focus=self.focus,
                         dye_channels=chans if composite else None,
                         particle_ids=extra[1] if extra is not None else None,
                         particle_palette=list(self.placed.particle_palette) if extra is not None else None,
                         particle_trails=extra[2] if extra is not None else None,
                         diagnostics=self.new_diagnostics(solver), trace=self.take_trace(solver, view),
                         physics=self.readouts(solver))

    # -------------------------------------------------- placed physics in frames
    def _emitter_particles(self, solver):
        """(positions (N, 2) m, emitter ids, streak / time lines in metres) of the emitter tracer
        particles, or None when the run has no particle emitters."""
        if self.placed is None or not self.placed.particle_palette or not hasattr(solver, "particles"):
            return None
        try:
            p = solver.particles()
            L = self.length
            pts = np.c_[np.asarray(p["x"]), np.asarray(p["y"])] * L
            lines = [np.asarray(ln, dtype=np.float64) * L for ln in solver.particle_lines()]
            return pts, np.asarray(p["emitter"], dtype=np.int16), lines
        except Exception:  # noqa: BLE001 - an overlay must never stop the run
            return None

    def new_diagnostics(self, solver) -> dict[str, np.ndarray] | None:
        """The diagnostics rows recorded since the previous frame (column arrays with "step")."""
        if not hasattr(solver, "diagnostics"):
            return None
        try:
            d = solver.diagnostics()
        except Exception:  # noqa: BLE001
            return None
        step = d.get("step")
        if step is None or not len(step):
            return None
        step = np.asarray(step, dtype=np.float64)
        sel = step > self._diag_seen
        if not sel.any():
            return None
        n = len(step)
        out = {k: np.asarray(v, dtype=np.float64)[sel] for k, v in d.items()
               if k not in _DIAG_SKIP and np.ndim(v) == 1 and len(v) == n}
        self._diag_seen = float(step[-1])
        return out

    def take_trace(self, solver, view: ViewSettings) -> dict | None:
        """A captured solver step when the view asked for one (it then clears the request)."""
        if not getattr(view, "capture_trace", False) or not hasattr(solver, "trace_step") or self.plan is None:
            return None
        try:
            view.capture_trace = False
        except (AttributeError, TypeError):
            pass
        try:
            from .physics_run import trace_payload
            tr = solver.trace_step(commit=False)
            return trace_payload(tr, self.plan, step=int(solver.steps), t=float(solver.t),
                                 axis=getattr(view, "slice_axis", None), pos=getattr(view, "slice_pos", None))
        except Exception:  # noqa: BLE001 - an inspection tool must never stop the run
            return None

    def readouts(self, solver) -> dict[str, float] | None:
        """The latest heat-transfer and fan numbers in SI (cheap: the solvers keep them on the device)."""
        if self.placed is None or self.plan is None:
            return None
        now = time.perf_counter()
        last = self._readout
        if last is not None and now - last[0] < READOUT_PERIOD_S:
            return last[1]                       # the per-step history is copied whole: not every frame
        out: dict[str, float] = {}
        self._readout = (now, out)
        try:
            from .physics_run import chunk_samples, physics_history_si
            if self.physics is not None and self.physics.thermal:
                th = solver.field("theta") if getattr(self.plan, "dimension", 2) == 2 else solver.field("temperature")
                out["t_max"] = float(self.physics.celsius(float(th.max())))
                out["t_min"] = float(self.physics.celsius(float(th.min())))
            ph = physics_history_si(solver, self.plan)
            for k, v in ph.items():
                if k != "t" and len(v):
                    out[k] = float(v[-1])
            for k, v in chunk_samples(solver, self.plan).items():
                out.setdefault({"nu_wall": "nu", "heat_wall": "heat_rate"}.get(k, k), float(v))
        except Exception:  # noqa: BLE001
            return out or None
        return out or None

    def _advance_particles(self, solver, mask) -> np.ndarray:
        """Advect the tracers over the simulated time since the previous frame.

        Velocities are non-dimensional and positions in metres, so the step is
        ``dt_nd * length`` (viz.Particles docs). One frame may span many solver
        steps; the advection is sub-stepped so no particle jumps more than ~1.5
        cells per sub-step (bilinear RK2 near a body needs that).
        """
        from ..viz import Particles
        t = float(solver.t)
        if self._particles is None:
            span = self.extent[1] - self.extent[0]
            # lifetime of about two flow-throughs keeps recirculation zones from hoarding dots
            self._particles = Particles(self.extent, n=N_PARTICLES, lifetime=2.0 * span / self.length,
                                        device=solver.device, seed=1)
            self._particle_t = t
        dt_nd = t - (self._particle_t if self._particle_t is not None else t)
        self._particle_t = t
        if dt_nd > 0:
            u, v = solver.field("u"), solver.field("v")
            n_sub = int(min(24, max(1, math.ceil(3.0 * dt_nd / (1.5 * self.h)))))
            for _ in range(n_sub):
                # lifetime is in the time unit of dt: here metres of free-stream travel
                self._particles.advect(u, v, dt_nd * self.length / n_sub, mask=mask)
        return self._particles.positions()

    def _streamlines(self, solver, mask) -> list[np.ndarray]:
        from ..viz import streamlines
        now = time.perf_counter()
        key = (round(float(solver.t), 3),)
        if self._stream is not None and (now - self._stream_wall < STREAMLINE_PERIOD_S or key == self._stream_key):
            return self._stream
        u = solver.field("u").float().cpu().numpy()
        v = solver.field("v").float().cpu().numpy()
        m = mask.float().cpu().numpy()
        try:
            self._stream = streamlines(u, v, self.extent, density=1.0, mask=m)
        except Exception:  # noqa: BLE001 - an overlay must never stop the run
            self._stream = []
        self._stream_wall, self._stream_key = time.perf_counter(), key
        return self._stream


# ---------------------------------------------------------------------------
# explore mode
# ---------------------------------------------------------------------------

#: 3D-view keys a 2D session accepts and ignores, so one GUI can post them to any session
VIEW3D_KEYS = {"view3d", "smoke", "streamlines3d", "seed_rake", "temperature3d", "particles3d"}
#: Notices a session keeps (the oldest are dropped; repeats are not added twice in a row).
MAX_NOTICES = 50
#: Placed-physics lists a session takes live (model objects, SI, design frame; DESIGN.md 10.4)
PLACEMENT_KEYS = {"emitters", "fans", "heat_sources", "ports"}
_POST_KEYS = ({"alpha_deg", "speed", "ground_moving", "paused", "reset", "field", "cmap", "particles",
               "streamlines", "range_mode", "vmin", "vmax", "fps", "capture_trace", "nu", "step", "steps_per_frame",
               "obstacles"} | VIEW3D_KEYS | PLACEMENT_KEYS)
_VIEW_KEYS = {"field", "cmap", "particles", "streamlines", "range_mode", "vmin", "vmax", "fps",
              "capture_trace"} | VIEW3D_KEYS
#: Largest frame-skip (solver steps per rendered frame) and single step request a session accepts.
MAX_STEPS_PER_FRAME = 2000


def _trim_history(solver, keep: float) -> dict[str, np.ndarray]:
    """The force history, after dropping samples older than ``keep`` convective times.

    Explore sessions run for minutes to hours; without trimming, the history
    (and the per-frame copy of it) would grow without bound. The solver's own
    ``trim_history`` also trims its placed-physics and diagnostics rings; an older
    solver without it has its host-side list replaced through the private field.
    """
    h = solver.history()
    t = h["t"]
    if t.size < 2 or t[-1] - t[0] <= 1.5 * keep:
        return h
    if hasattr(solver, "trim_history"):
        solver.trim_history(keep)
        return solver.history()
    sel = t >= t[-1] - keep
    h = {k: v[sel] for k, v in h.items()}
    if isinstance(getattr(solver, "_host_hist", None), list):
        solver._host_hist[:] = [np.c_[h["t"], h["cd"], h["cl"], h["cm"]]]
    return h


#: Point-segment pairs evaluated at once by :func:`_polyline_sdf` (bounds its temporary memory).
SDF_PAIRS_PER_CHUNK = 1 << 22
#: Device memory the explore drawing may spend on Undo checkpoints of the paint layer.
PAINT_CHECKPOINT_BYTES = 64 << 20


def _polyline_sdf(points: np.ndarray, radius: float):
    """Signed distance (torch, solver units) to a brush stroke: the polyline ``points`` thickened
    by ``radius``. The segments of a chunk are evaluated in one broadcast (segments x points, at
    most :data:`SDF_PAIRS_PER_CHUNK` pairs), so a 40-point stroke costs a handful of kernels
    instead of ten per segment."""
    import torch
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    def sdf(x, y):
        p = torch.as_tensor(pts, device=x.device, dtype=x.dtype)
        if len(p) == 1:
            return torch.hypot(x - p[0, 0], y - p[0, 1]) - radius
        a_all, b_all = p[:-1], p[1:]
        step = max(1, SDF_PAIRS_PER_CHUNK // max(x.numel(), 1))
        xs, ys = x[None], y[None]
        best = None
        for i0 in range(0, len(a_all), step):
            a, b = a_all[i0:i0 + step], b_all[i0:i0 + step]
            shape = (len(a),) + (1,) * x.dim()
            ax, ay = a[:, 0].reshape(shape), a[:, 1].reshape(shape)
            dx, dy = (b[:, 0] - a[:, 0]).reshape(shape), (b[:, 1] - a[:, 1]).reshape(shape)
            ll = (dx * dx + dy * dy).clamp_min(1e-30)
            t = (((xs - ax) * dx + (ys - ay) * dy) / ll).clamp(0.0, 1.0)
            d = torch.hypot(xs - (ax + t * dx), ys - (ay + t * dy)).amin(dim=0)
            best = d if best is None else torch.minimum(best, d)
        return best - radius
    return sdf


def _store_checkpoint(solver, checkpoints: dict, n: int) -> None:
    """Keep the paint layer after the first ``n`` operations (Undo restores it instead of replaying),
    dropping the oldest checkpoints beyond :data:`PAINT_CHECKPOINT_BYTES`."""
    snap = solver.paint_snapshot()
    if snap is None:
        return
    checkpoints[n] = snap
    size = sum(t.numel() * t.element_size() for t in snap)
    keep = max(4, PAINT_CHECKPOINT_BYTES // max(size, 1))
    while len(checkpoints) > keep:
        del checkpoints[min(checkpoints)]


def apply_obstacles_2d(solver, payload: dict | None, *, length: float, applied: list | None = None,
                       checkpoints: dict | None = None) -> list:
    """Draw (or erase) the obstacles of the Simulate page's sketch into a 2D solver (DESIGN.md 10.4).

    ``payload`` is ``app.widgets.draw_tools.ObstacleSketch.payload``: wind-frame metres, its
    operations ("ops": brush strokes and polygons, each adding or erasing solid, in order) and,
    as a fallback, the traced outlines ("polygons"). The operations are replayed onto the
    solver's paint layer (``FlowSolver.paint_*``: penalised to rest with the bodies' calibrated
    wall, insulating, clearing smoke, never part of the differentiable geometry), so undo and
    erase are exact. When the new list extends the ``applied`` one, only the new operations are
    painted. Otherwise (Undo, an edited or in-progress stroke) the layer goes back to the latest
    ``checkpoints`` entry (a dict the caller keeps: paint snapshots after each of the first n
    operations) that the new list still shares, and only the rest is replayed -- replaying every
    stroke from an empty layer made Undo cost 64 ms per stroke on the room grid (1.3 s at 20
    strokes, critic round 5). Without a usable checkpoint the layer is cleared and replayed.
    Returns the list now applied.
    """
    ops = list((payload or {}).get("ops") or [])
    pending = (payload or {}).get("pending")
    if pending:
        ops.append(pending)
    L = float(length)
    applied = list(applied or [])
    if not ops and payload and payload.get("polygons"):
        if checkpoints is not None:
            checkpoints.clear()
        solver.clear_paint()
        for poly in payload["polygons"]:
            pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            if len(pts) >= 3:
                solver.paint_polygon(pts / L)
        return []
    keyed = [_op_key(o) for o in ops]
    common = 0
    for k_new, k_old in zip(keyed, applied):
        if k_new != k_old:
            break
        common += 1
    use_ckpt = checkpoints is not None and hasattr(solver, "paint_snapshot") and hasattr(solver, "restore_paint")
    if use_ckpt:
        for n in [n for n in checkpoints if n > common]:     # a history the new list no longer shares
            del checkpoints[n]
    if applied and common == len(applied):
        start = common                                      # a pure extension: paint the new ones only
    else:
        best = max((n for n in checkpoints if n <= common), default=0) if use_ckpt else 0
        if best > 0:
            solver.restore_paint(checkpoints[best])
            start = best
        else:
            start = 0
            solver.clear_paint()
    for i, op in enumerate(ops[start:], start):
        kind = str(op.get("kind", "brush"))
        pts = np.asarray(op.get("points", []), dtype=np.float64).reshape(-1, 2) / L
        erase = bool(op.get("erase", False))
        if not len(pts):
            continue
        if kind == "polygon" and len(pts) >= 3:
            solver.paint_polygon(pts, erase=erase)
        else:
            r = max(float(op.get("radius", 0.0)) / L, 1e-9)
            if hasattr(solver, "_apply_paint"):
                # the exact capsule of the stroke, through the solver's own paint pass
                solver._apply_paint(_polyline_sdf(pts, r), erase)
            else:
                step = 0.5 * r
                for a, b in itertools.pairwise(pts):
                    n = max(1, math.ceil(float(np.hypot(*(b - a))) / step))
                    for s in np.linspace(0.0, 1.0, n + 1)[:-1]:
                        solver.paint_disc(tuple(a + s * (b - a)), r, erase=erase)
                solver.paint_disc(tuple(pts[-1]), r, erase=erase)
        if use_ckpt:
            _store_checkpoint(solver, checkpoints, i + 1)
    return keyed


def _op_key(op: dict) -> tuple:
    pts = np.asarray(op.get("points", []), dtype=np.float64).round(12)
    return (str(op.get("kind")), bool(op.get("erase")), round(float(op.get("radius", 0.0)), 12), pts.tobytes())


class LiveSession:
    """Real-time explore mode (DESIGN.md 5.4, 10.4).

    Built from a design and a simulation (whose conditions are the starting
    point). :meth:`post` changes parameters from any thread; :meth:`tick`
    advances the flow for about ``max_ms`` and returns one :class:`LiveFrame`.
    The grid is planned by :func:`aerolume.simulation.plan.plan_live`; angle
    changes rotate the bodies in place (the solver keeps its flow field and its
    CUDA graphs), speed changes only change the Reynolds number (the
    non-dimensional flow field stays valid), and switching the road between
    moving and fixed rebuilds the solver with the current flow state.

    Placed physics (DESIGN.md 10.4): moving, adding or retuning emitters, fans,
    heaters and jets (``post(emitters=[...])`` etc., model objects in SI) rewrites the
    solver's sources in place; drawn obstacles (``post(obstacles=payload)``) are
    painted into the flow; the Reynolds number follows a new kinematic viscosity
    (``nu``, the Reynolds lock: two of U, nu and Re held); ``paused`` with ``step``
    advances a given number of steps; ``steps_per_frame`` fixes the steps per
    rendered frame (frame skip) instead of pacing them to the frame budget; and
    ``capture_trace`` puts one step's stages into the next frame.
    """

    POST_KEYS = frozenset(_POST_KEYS)

    def __new__(cls, design: Design, sim: Simulation, *args, **kwargs):
        # a 3D design gets the 3D session (DESIGN.md 7.7); it subclasses this one, so
        # isinstance checks and the GUI's LiveWorker work unchanged
        if cls is LiveSession and getattr(design, "dimension", 2) == 3:
            from .live3d import LiveSession3D
            return super().__new__(LiveSession3D)
        return super().__new__(cls)

    def __init__(self, design: Design, sim: Simulation, *, device: str | None = None,
                 view: ViewSettings | None = None):
        from ..solvers.ns2d import FlowSolver
        self.design = design
        self.sim = Simulation.from_dict(sim.to_dict())         # private copy: posts edit it
        self.view = view if view is not None else ViewSettings()
        self.plan: SimPlan = plan_live(design, self.sim, device=device)
        self._threads = cpu_threads(self.plan.device)
        self._threads.__enter__()
        try:
            self.solver = FlowSolver(self.plan.solver_config)
            from .runner import start_from_rest, starts_unbounded
            if starts_unbounded(self.plan.solver_config) and math.isfinite(self.plan.dt) and self.plan.dt > 0:
                # a heated box at rest: the solver's first step is unbounded (runner.starts_unbounded)
                start_from_rest(self.solver, self.plan.solver_config, self.plan.dt)
        except Exception:
            self._threads.__exit__(None, None, None)
            raise
        cfg = self.plan.solver_config
        self.renderer = FrameRenderer(self.plan.extent, length=self.plan.length_ref, speed=self.plan.speed,
                                      density=self.plan.fluid.density, h=cfg.h, focus=self.plan.focus_extent,
                                      physics=self.plan.physics, placed=self.plan.placed)
        self.renderer.plan = self.plan
        self._init_common()
        self._bodies = [np.asarray(b.points, dtype=np.float64) for b in self.plan.bodies_wind]
        self.pitch_limit = LIVE_PITCH_GROUND_DEG if self.plan.ground else LIVE_PITCH_DEG

    def _init_common(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict = {}
        self._paused = False
        self._closed = False
        self._sec_per_step: float | None = None
        self._timed = False                    # has a tick after the warm-up been measured?
        self._render_timed = False
        self._render_s = 0.01
        self._rate = float("nan")
        self._steps_per_frame: int | None = None
        self._obstacles = None                 # the drawing last posted (replayed after a solver rebuild)
        self._obstacles_applied: list = []
        self._paint_checkpoints: dict = {}     # paint-layer snapshots after n operations (fast Undo)
        self._physics_dirty = False
        self.notices: list[str] = []           # what a post could not do, for the GUI's message log
        self._notices_sent = 0                 # notices already handed out in a frame
        self.alpha_deg = float(self.sim.conditions.alpha_deg)
        self.speed = float(self.sim.conditions.speed)

    def _notice(self, text: str) -> None:
        """Record what an edit could not do. The same text is not repeated while it is among the
        recent ones (a slider dragged over a closed box would otherwise log it per tick), and the
        list is bounded (explore sessions run for hours)."""
        if text in self.notices[-10:]:
            return
        self.notices.append(text)
        if len(self.notices) > MAX_NOTICES:
            drop = len(self.notices) - MAX_NOTICES
            del self.notices[:drop]
            self._notices_sent = max(0, self._notices_sent - drop)

    def _take_notices(self) -> list[str] | None:
        """The notices recorded since the previous frame (None when there are none)."""
        new = self.notices[self._notices_sent:]
        self._notices_sent = len(self.notices)
        return list(new) or None

    # ------------------------------------------------------------------ input
    def post(self, **changes) -> None:
        """Request parameter / view changes (thread-safe; applied on the next tick).

        Keys: ``alpha_deg`` (deg), ``speed`` (m/s), ``ground_moving`` (bool),
        ``paused`` (bool), ``reset`` (True), ``nu`` (kinematic viscosity, m²/s),
        ``step`` (steps to run while paused), ``steps_per_frame`` (frame skip, None
        or 0 = paced to the frame budget), ``emitters`` / ``fans`` / ``heat_sources``
        / ``ports`` (lists of model objects or dicts), ``obstacles`` (the sketch
        payload, or None to clear), and the view keys ``field``, ``cmap``,
        ``particles``, ``streamlines``, ``range_mode``, ``vmin``, ``vmax``, ``fps``,
        ``capture_trace``. View keys take effect immediately; the others at the
        start of the next tick, the latest value winning (``step`` requests add up).
        """
        unknown = set(changes) - self.POST_KEYS
        if unknown:
            raise TypeError(f"unknown live setting(s): {', '.join(sorted(unknown))}")
        self._check(changes)
        for k in _VIEW_KEYS & set(changes):
            setattr(self.view, k, changes[k])
        with self._lock:
            for k, v in changes.items():
                if k in _VIEW_KEYS:
                    continue
                if k == "reset":
                    self._pending["reset"] = self._pending.get("reset", False) or bool(v)
                elif k == "step":
                    self._pending["step"] = min(self._pending.get("step", 0) + int(v), MAX_STEPS_PER_FRAME)
                else:
                    self._pending[k] = v

    def _check(self, changes: dict) -> None:
        """Validate a post in the caller's thread (a bad value raises there, not in the tick)."""
        if "speed" in changes and not (float(changes["speed"]) > 0 and math.isfinite(float(changes["speed"]))):
            raise ValueError("speed must be positive")
        if "alpha_deg" in changes and not math.isfinite(float(changes["alpha_deg"])):
            raise ValueError("alpha_deg must be finite")
        if "nu" in changes and not (float(changes["nu"]) > 0 and math.isfinite(float(changes["nu"]))):
            raise ValueError("nu (kinematic viscosity) must be positive")
        if "step" in changes and int(changes["step"]) < 1:
            raise ValueError("step must be at least 1")
        spf = changes.get("steps_per_frame")
        if spf is not None and not 0 <= int(spf) <= MAX_STEPS_PER_FRAME:
            raise ValueError(f"steps_per_frame must be 0 (paced) to {MAX_STEPS_PER_FRAME}")
        for k in PLACEMENT_KEYS & set(changes):
            if not isinstance(changes[k], (list, tuple)):
                # ValueError, not TypeError: the explore worker reads a TypeError from post() as
                # "this session does not know the key" and stops sending it (app.jobs.LiveWorker)
                raise ValueError(f"{k} must be a list")  # noqa: TRY004

    def _take_pending(self) -> dict:
        with self._lock:
            p, self._pending = self._pending, {}
        return p

    # -------------------------------------------------------------- changes
    def _immersed(self, bodies_wind, moving: bool | None = None) -> list:
        """Wind-frame bodies (metres) -> solver bodies. Wheels on a moving road roll
        with it (omega = U / r) when the simulation asks for rotating wheels; heated
        bodies keep their thermal condition (solver units, from the plan)."""
        from ..solvers.ns2d import ImmersedBody
        L, U = self.plan.length_ref, self.plan.speed
        if moving is None:
            moving = bool(self.sim.ground.moving)
        moving = self.plan.ground and moving
        spin = moving and bool(self.sim.ground.rotate_wheels)
        thermal = [(b.thermal, b.thermal_value) for b in self.plan.solver_config.bodies]
        out = []
        for i, (b, planned) in enumerate(zip(bodies_wind, self.plan.bodies_wind)):
            omega = planned.omega
            if b.role == "wheel" and self.plan.ground:
                pts = np.asarray(planned.points)
                c = planned.center or (0.0, 0.0)
                r = float(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]).mean())
                omega = (self.plan.speed / r) if (spin and r > 0) else 0.0
            # a body without an explicit centre spins about its own centroid, which
            # moves when the body is pitched
            center = b.center if b.center is not None else tuple(P.centroid(b.points))
            th_mode, th_val = thermal[i] if i < len(thermal) else ("adiabatic", 0.0)
            out.append(ImmersedBody(points=np.asarray(b.points) / L, name=planned.name, role=b.role,
                                    omega=omega * L / U, center=(center[0] / L, center[1] / L),
                                    thermal=th_mode, thermal_value=th_val))
        return out

    def _valid_alpha(self, alpha: float) -> float:
        """Clamp to the planned pitch range and, on a road, to angles that keep the
        body above it (bisection on the angle)."""
        a = max(-self.pitch_limit, min(self.pitch_limit, float(alpha)))
        if not self.plan.ground or self.plan.closed:
            return a

        def ok(x: float) -> bool:
            bw = wind_frame_bodies(self.design, x, ground=True)
            return min(float(np.asarray(b.points)[:, 1].min()) for b in bw) >= 0.5 * self.plan.solver_config.h \
                * self.plan.length_ref or x == 0.0
        if ok(a):
            return a
        lo, hi = 0.0, a
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    def _set_alpha(self, alpha: float) -> None:
        a = self._valid_alpha(alpha)
        if abs(a - self.alpha_deg) < 1e-9 or not self.design.bodies:
            return
        bw = wind_frame_bodies(self.design, a, ground=self.plan.ground and not self.plan.closed)
        self.solver.set_bodies(self._immersed(bw))
        self._bodies = [np.asarray(b.points, dtype=np.float64) for b in bw]
        self.alpha_deg = a
        self.sim.conditions.alpha_deg = a
        self.renderer.invalidate_geometry()
        self._physics_dirty = True               # body-frame placements turn with the body

    def _reynolds(self) -> float:
        """The simulated Reynolds number of the session's current speed and fluid."""
        if self.plan.closed:
            return self.plan.speed * self.plan.length_ref / float(self.sim.conditions.fluid.kinematic_viscosity)
        re = self.sim.conditions.reynolds(self.plan.length_ref)
        if self.sim.cfd.reynolds_mode == "capped":
            re = min(re, float(self.sim.cfd.reynolds_cap or self.plan.re_sim))
        return re

    def _set_speed(self, speed: float) -> None:
        """New free-stream speed: in solver units only the Reynolds number changes
        (velocities are scaled by U), so the current flow field stays a valid state.
        Placed physics scaled by U (fan thrust, jet speeds, heater power, buoyancy)
        are re-expressed in the new units on the same tick."""
        if self.plan.closed:
            self._notice("A closed box has no free stream: its reference velocity comes from what drives the flow, "
                         "so the speed slider does not apply.")
            return
        self.speed = float(speed)
        self.sim.conditions.speed = self.speed
        self.solver.set_inflow(re=self._reynolds())
        self.renderer.speed = self.speed
        self._follow_units()

    def _set_nu(self, nu: float) -> None:
        """New kinematic viscosity (the Reynolds lock): the fluid's viscosity changes, the speed
        stays, so Re = U L / nu follows (and the Peclet number with it, at the same Prandtl number)."""
        fl = self.sim.conditions.fluid
        fl.viscosity = float(nu) * float(fl.density)
        self.solver.set_inflow(re=self._reynolds())
        self._follow_units()

    def _follow_units(self) -> None:
        """After a speed or viscosity change: the solver's velocity unit is now the session's speed,
        so the plan's speed and scales follow (the SI readouts -- heat rate, fan thrust -- and the
        placed physics are converted with them), and the placed physics is re-expressed."""
        if not self.plan.closed:
            self.plan.speed = self.speed
        if self.plan.physics is not None:
            self.plan.physics = self._current_scales()
            self.renderer.physics = self.plan.physics
        self._physics_dirty = True

    def _set_ground_moving(self, moving: bool) -> None:
        if not self.plan.ground or self.plan.closed or bool(moving) == bool(self.sim.ground.moving):
            return
        from ..solvers.ns2d import FlowSolver
        state = self.solver.state_dict()
        bw = wind_frame_bodies(self.design, self.alpha_deg, ground=True)
        cfg = dataclasses.replace(self.solver.cfg, bc_bottom="moving" if moving else "noslip",
                                  bodies=self._immersed(bw, moving=bool(moving)))
        new = FlowSolver(cfg)                   # the old solver stays usable if this fails
        new.load_state_dict(state)
        self.solver = new
        self.sim.ground.moving = bool(moving)
        self.renderer.invalidate_geometry()
        self._sec_per_step, self._timed = None, False   # new CUDA graphs: re-measure the pace
        self._obstacles_applied = []            # the new solver's paint layer is empty: replay the drawing
        self._paint_checkpoints = {}
        if self._obstacles is not None:
            self._apply_obstacles(self._obstacles)

    # ------------------------------------------------------ placed physics
    def _current_scales(self):
        """The plan's scales at the session's current speed and kinematic viscosity.

        ΔT_ref stays (theta keeps its meaning) and so does the Prandtl number: the solver holds
        Pr and takes Pe = Re Pr, so a new viscosity behaves as a fluid whose diffusivity follows
        it. A closed box keeps its reference velocity (no free stream); its Reynolds, Grashof and
        Rayleigh numbers follow the viscosity.
        """
        s = self.plan.physics
        if s is None:
            return None
        U = s.speed if self.plan.closed else float(self.speed)
        nu = float(self.sim.conditions.fluid.kinematic_viscosity)
        if abs(U - s.speed) <= 1e-12 * max(1.0, s.speed) and abs(nu - s.nu) <= 1e-12 * max(nu, s.nu):
            return s
        re = U * s.length / nu
        visc = (s.nu / nu) ** 2                  # Gr and Ra go as 1 / nu^2 at a fixed Pr
        forced = U if not self.plan.closed else s.u_forced
        ri_flow = (s.g_beta * s.delta_t * s.length_ra / forced ** 2
                   if (s.ri_flow is not None and forced > 0) else s.ri_flow)
        return dataclasses.replace(s, speed=U, u_forced=forced, nu=nu, re=re, pe=re * s.pr, gr=s.gr * visc,
                                   ra=s.ra * visc, ri_flow=ri_flow,
                                   ri=s.g_beta * s.delta_t * s.length / (U * U) if s.g_beta > 0 else 0.0)

    def _placed_now(self):
        from .physics import placed_physics_2d
        return placed_physics_2d(self.design, self.sim, self._current_scales(), alpha_deg=self.alpha_deg,
                                 extent=self.plan.extent, h_m=self.plan.solver_config.h * self.plan.length_ref,
                                 bodies_wind=self.plan.bodies_wind, closed=self.plan.closed)

    def _push_placed(self, placed) -> None:
        """Hand rebuilt specs to the solver; the solver keeps its tensors (and CUDA graphs) when only
        positions or values changed and re-captures lazily when the structure changed."""
        s = self.solver
        if s.cfg.thermal is None and placed.thermal is not None and self.plan.physics is not None:
            self._notice("Heat transfer was off when explore mode started; restart explore mode to add it.")
        s.set_emitters(placed.emitters)
        s.set_sources(heat_sources=placed.heat_sources if s.cfg.thermal is not None else [], fans=placed.fans,
                      ports=placed.ports)
        if s.cfg.thermal is not None and placed.thermal is not None:
            s.set_thermal(richardson=placed.thermal.richardson)

    def _apply_physics(self) -> None:
        """Rebuild the placed physics after a placement, speed or attitude change (once per tick)."""
        if self.plan.physics is None and not any(getattr(self.sim, k) for k in PLACEMENT_KEYS):
            return
        try:
            if self.plan.physics is None:
                from .physics import physics_scales
                self.plan.physics = physics_scales(self.design, self.sim, 2)
                self.renderer.physics = self.plan.physics
            placed = self._placed_now()
            self._push_placed(placed)
            self.plan.placed = placed
            self.renderer.placed = placed
            self.renderer.invalidate_readouts()
            for w in placed.warnings:
                if w not in self.notices:
                    self._notice(w)
        except (ValueError, TypeError, RuntimeError) as exc:
            self._notice(f"The placed physics could not be updated: {exc}")

    def _set_list(self, key: str, items) -> None:
        from ..model import Emitter, Fan, HeatSource, Port
        cls = {"emitters": Emitter, "fans": Fan, "heat_sources": HeatSource, "ports": Port}[key]
        setattr(self.sim, key, [x if isinstance(x, cls) else cls.from_dict(dict(x)) for x in items])
        self._physics_dirty = True

    def _apply(self, p: dict) -> None:
        self._physics_dirty = False
        if p.get("reset"):
            self.solver.reset()                 # same buffers: the CUDA graphs and the pace stay valid
            self.renderer.reset()
        if "paused" in p:
            self._paused = bool(p["paused"])
        if "steps_per_frame" in p:
            n = int(p["steps_per_frame"] or 0)
            self._steps_per_frame = n if n > 0 else None
        if "ground_moving" in p:
            self._set_ground_moving(bool(p["ground_moving"]))
        if "alpha_deg" in p:
            self._set_alpha(float(p["alpha_deg"]))
        if "speed" in p:
            self._set_speed(float(p["speed"]))
        if "nu" in p:
            self._set_nu(float(p["nu"]))
        for k in PLACEMENT_KEYS & set(p):
            self._set_list(k, p[k])
        if self._physics_dirty:
            self._apply_physics()
        if "obstacles" in p:
            self._apply_obstacles(p["obstacles"])

    def _apply_obstacles(self, payload) -> None:
        self._obstacles = payload
        try:
            self._obstacles_applied = apply_obstacles_2d(self.solver, payload, length=self.plan.length_ref,
                                                         applied=self._obstacles_applied,
                                                         checkpoints=self._paint_checkpoints)
        except Exception as exc:  # noqa: BLE001 - a drawing must never stop the run
            self._notice(f"The drawn obstacles could not be applied: {exc}")
        self.renderer.invalidate_geometry()

    # ------------------------------------------------------------------ run
    def _steps_this_tick(self, budget: float) -> int:
        if self._steps_per_frame is not None:
            n = self._steps_per_frame
        elif self._sec_per_step is None:
            n = WARMUP_STEPS
        else:
            n = int(budget / self._sec_per_step)
        n = max(1, min(n, MAX_STEPS_PER_FRAME))
        if self.solver.device.type == "cuda":
            # whole CUDA-graph blocks, and never fewer than one: a remainder would
            # capture a separate one-step graph (~70 ms, measured), which then
            # throws the pace off for several ticks
            n = max(WARMUP_STEPS, n - n % 10)
        return n

    def _advance(self, n: int) -> None:
        ts = time.perf_counter()
        self.solver.advance(n)
        el = time.perf_counter() - ts
        sps = el / n
        if not self._timed:
            # the first tick after a (re)build captures the CUDA graphs and builds
            # the cuFFT plans: far slower than a normal step, so it must not set the
            # pace (it would make the next ticks run a single step each)
            self._timed = True
        else:
            self._sec_per_step = sps if self._sec_per_step is None else 0.7 * self._sec_per_step + 0.3 * sps
            self._rate = 1.0 / self._sec_per_step
        c = self.solver.coefficients()
        if not np.all(np.isfinite(c)):
            # an extreme edit (huge speed jump at a coarse grid) can blow the flow up:
            # restart from the free stream instead of showing garbage
            self.solver.reset()
            self.renderer.reset()
            self._notice("The flow became unstable after an edit and was restarted.")

    def tick(self, max_ms: float = 33.0) -> LiveFrame:
        """Apply pending changes, run as many steps as fit in ``max_ms`` (minus the
        expected render time; or ``steps_per_frame`` steps) and render one frame.
        While paused only a posted ``step`` advances the flow."""
        if self._closed:
            raise RuntimeError("the live session is closed")
        p = self._take_pending()
        self._apply(p)
        budget = max(0.0, max_ms / 1000.0 - self._render_s)
        if not self._paused:
            self._advance(self._steps_this_tick(budget))
        elif p.get("step"):
            self._advance(int(p["step"]))
        tr = time.perf_counter()
        hist = _trim_history(self.solver, LIVE_HISTORY_T)
        frame = self.renderer.render(self.solver, self.view, bodies=self._bodies, phase="live", progress=0.0,
                                     eta_s=None, steps_per_s=self._rate, history=hist)
        frame.notices = self._take_notices()
        el = time.perf_counter() - tr
        if self._render_timed:                  # the first render builds masks and caches: not typical
            self._render_s = 0.7 * self._render_s + 0.3 * el
        self._render_timed = True
        return frame

    def close(self) -> None:
        """Release the solver (and GPU memory); idempotent."""
        if self._closed:
            return
        self._closed = True
        dev = getattr(self.solver, "device", None)
        self.solver = None
        self.renderer = None
        with contextlib.suppress(Exception):
            self._threads.__exit__(None, None, None)
        if dev is not None and getattr(dev, "type", "") == "cuda":
            with contextlib.suppress(Exception):
                import torch
                torch.cuda.empty_cache()
