"""Stateful flow solver for the application: fast path, live edits, outputs.

:class:`FlowSolver` wraps the functional core (:mod:`.core`) with everything a
GUI or a batch runner needs and that a pure function cannot hold:

* **Static buffers + CUDA graphs.** The state lives in fixed tensors; on CUDA a
  block of K steps is captured once in a ``torch.cuda.CUDAGraph`` and replayed.
  A step launches a few hundred small kernels, and at ~27 us per launch under
  Windows WDDM launch overhead alone would cap the solver at ~100 steps/s;
  replaying a graph costs one launch. Nothing inside a step synchronises with
  the host (dt comes from a device-side CFL reduction), so capture is legal.
  Any failure to capture or replay falls back to plain eager execution.
* **Device-side history.** [t, Cd, Cl, Cm] goes into a ring buffer on the
  device every step and is copied to the host once per :meth:`advance`; the
  placed-physics outputs (heat rates, wall heat, fan thrust and disc velocity,
  painted-obstacle forces) and the diagnostics rows go into rings of their own.
* **Live edits.** :meth:`set_inflow` writes the viscosity tensor in place and
  :meth:`set_bodies` rebuilds the masks into the existing tensors whenever the
  bodies still fit the current window, so captured graphs stay valid while the
  user drags an angle-of-attack slider. :meth:`set_emitters`,
  :meth:`set_sources` and :meth:`set_thermal` do the same for the placed physics
  (DESIGN.md 10.4) and the paint API (:meth:`paint_disc`, :meth:`paint_polygon`,
  :meth:`clear_paint`) draws obstacles with the mouse without touching the
  differentiable body geometry. A change of *structure* (a first fan, another
  smoke channel) rebuilds what it must and re-captures the graphs lazily.
* **Outputs.** Cell-centred fields, running means / rms, surface distributions
  of cp and cf, heat-transfer rates and Nusselt numbers, the fan report, tracer
  particles, diagnostics (CSV) and a one-step solver trace.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import time
import warnings

import numpy as np
import torch

from ..fpmode import flush_denormal
from . import physics as phx
from .config import EmitterSpec, FanSpec, HeatSourceSpec, ImmersedBody, PortSpec, SolverConfig, WallBC
from .core import (
    State,
    _gradients,
    advect_dye,
    divergence,
    eddy_viscosity,
    face_coordinates,
    init_state,
    make_operators,
    pad_u,
    pad_v,
    set_reynolds,
    step_full,
)
from .diagnostics import DIAG_COLUMNS, DIAG_EVERY, write_csv
from .geometry import Geometry, body_window, build_geometry, signed_distance, window_contains
from .particles import TracerParticles, particle_emitters

FIELDS = ("u", "v", "speed", "p", "cp", "vorticity", "mask", "dye", "nut", "divergence")
#: extra cell-centred fields of the placed physics (zeros when the feature is off)
PHYSICS_FIELDS = ("theta", "dye1", "dye2", "dye3", "paint", "heat_source")
HISTORY_CAPACITY = 4096         # device ring rows; flushed to the host before it can wrap
GRAPH_STEPS = 10                # steps captured per CUDA graph
LIVE_PITCH_DEG = 25.0           # the body window covers +- this rotation about moment_ref


# ---------------------------------------------------------------------------
# small field helpers (shared with mean_fields)
# ---------------------------------------------------------------------------

def centre_u(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * (u[:, 1:] + u[:, :-1])


def centre_v(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * (v[1:] + v[:-1])


def vorticity(u: torch.Tensor, v: torch.Tensor, ops) -> torch.Tensor:
    """dv/dx - du/dy at cell centres (corner values averaged, BC-aware)."""
    _, g12, g21, _, _ = _gradients(pad_u(u, ops), pad_v(v, ops), ops)
    return (g21 - g12)[1:-1, 1:-1]


def bilinear(field: torch.Tensor, x: torch.Tensor, y: torch.Tensor, ops,
             xoff: float = 0.5, yoff: float = 0.5) -> torch.Tensor:
    """Bilinear interpolation of a grid field at points (x, y).

    ``field[j, i]`` sits at ``(x0 + (i + xoff) h, y0 + (j + yoff) h)``: the
    defaults are cell centres; (0, 0.5) are the u-faces and (0.5, 0) the
    v-faces, so staggered velocities are interpolated where they live instead
    of after an extra averaging pass. Points outside are clamped to the edge.
    """
    ny, nx = field.shape
    fx = ((x - ops.x0) / ops.h - xoff).clamp(0.0, nx - 1.0)
    fy = ((y - ops.y0) / ops.h - yoff).clamp(0.0, ny - 1.0)
    i0 = fx.floor().clamp(max=nx - 2).long()
    j0 = fy.floor().clamp(max=ny - 2).long()
    tx, ty = fx - i0, fy - j0
    f00 = field[j0, i0]
    f01 = field[j0, i0 + 1]
    f10 = field[j0 + 1, i0]
    f11 = field[j0 + 1, i0 + 1]
    return (1 - ty) * ((1 - tx) * f00 + tx * f01) + ty * ((1 - tx) * f10 + tx * f11)


def resample_closed(poly: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n points equally spaced in arc length along a closed polygon, from vertex 0.

    Returns (points (n, 2), s (n,)) with s the arc length of each point.
    """
    p = np.asarray(poly, dtype=np.float64)
    q = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(q, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], n, endpoint=False)
    x = np.interp(s, cum, q[:, 0])
    y = np.interp(s, cum, q[:, 1])
    return np.c_[x, y], s


class _Ring:
    """A device ring buffer of fixed-width rows, flushed to a host list.

    Rows are written inside captured graphs (``index_copy_`` at a device
    counter); the host only knows how many rows it asked for, which is enough
    to copy the unread ones in order.
    """

    def __init__(self, columns: tuple[str, ...], device, dtype, capacity: int = HISTORY_CAPACITY):
        self.columns = tuple(columns)
        self.capacity = capacity
        self.buf = torch.zeros(capacity, len(columns), device=device, dtype=dtype)
        self.count = torch.zeros((), device=device, dtype=torch.long)
        self.host: list[np.ndarray] = []
        self.rows = 0
        self.flushed = 0

    def write(self, row: torch.Tensor) -> None:
        self.buf.index_copy_(0, torch.remainder(self.count, self.capacity).reshape(1), row.reshape(1, -1).to(
            self.buf.dtype))
        self.count.add_(1)

    def flush(self) -> None:
        k = self.rows - self.flushed
        if k <= 0:
            return
        a = self.flushed % self.capacity
        b = a + k
        rows = self.buf[a:b] if b <= self.capacity else torch.cat([self.buf[a:], self.buf[:b - self.capacity]])
        self.host.append(rows.detach().to("cpu", torch.float64).numpy())
        self.flushed = self.rows

    def table(self) -> np.ndarray:
        self.flush()
        return np.concatenate(self.host) if self.host else np.zeros((0, len(self.columns)))

    def clear(self) -> None:
        self.count.zero_()
        self.host.clear()
        self.rows = self.flushed = 0

    def trim(self, keep_from_index: int) -> None:
        """Drop host rows before ``keep_from_index`` (after a flush)."""
        t = self.table()
        self.host[:] = [t[keep_from_index:]] if len(t) > keep_from_index else []


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------

class FlowSolver:
    """Incompressible 2D Navier-Stokes around immersed bodies (non-dimensional).

    Typical use::

        solver = FlowSolver(cfg)
        solver.advance(500)                 # fast path, no autograd
        cd, cl, cm = solver.coefficients()
        speed = solver.field("speed")       # (ny, nx) tensor on the solver device
    """

    def __init__(self, cfg: SolverConfig):
        self.cfg = dataclasses.replace(cfg, bodies=list(cfg.bodies), emitters=list(cfg.emitters),
                                       heat_sources=list(cfg.heat_sources), fans=list(cfg.fans),
                                       ports=list(cfg.ports))
        self.ops = make_operators(self.cfg)
        self.device = self.ops.device
        self.dtype = self.ops.dtype
        self._bodies: list[ImmersedBody] = list(self.cfg.bodies)
        with torch.no_grad():
            self._geom = build_geometry(self._bodies, self.ops, window=self._live_window(self._bodies))
            self._state = init_state(self.cfg, self.ops)
        nb = max(len(self._bodies), 0)
        self._body_c = torch.zeros(nb, 3, device=self.device, dtype=self.dtype)
        self._hist = torch.zeros(HISTORY_CAPACITY, 4, device=self.device, dtype=self.dtype)
        self._hist_n = torch.zeros((), device=self.device, dtype=torch.long)
        self._host_hist: list[np.ndarray] = []
        self._rows = 0                 # rows written to the device ring since reset
        self._flushed = 0              # rows already copied to the host
        self.steps = 0
        self._avg: dict[str, torch.Tensor] | None = None
        self._avg_on = False
        self._use_graphs = bool(self.cfg.use_cuda_graphs) and self.device.type == "cuda"
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._pool = None
        self.graph_error: str | None = None
        # placed physics: outputs, rings, particles, diagnostics
        self._perf: list[tuple[int, int, float]] = []          # (steps after, n, wall seconds)
        self._diag = _Ring(DIAG_COLUMNS, self.device, torch.float64 if self.device.type != "mps" else torch.float32,
                           capacity=max(64, HISTORY_CAPACITY // DIAG_EVERY + 8)) if self.cfg.diagnostics else None
        self._step_dev = torch.zeros((), device=self.device, dtype=self._diag.buf.dtype) if self._diag else None
        self._particles: TracerParticles | None = None
        self._solid_c: torch.Tensor | None = None
        self._out: dict[str, torch.Tensor] = {}
        self._aux: _Ring | None = None
        self._phys_avg: torch.Tensor | None = None
        self._setup_outputs()
        self._setup_particles()

    # ------------------------------------------------------------- set-up parts
    def _out_layout(self) -> list[tuple[str, int, list[str]]]:
        """(key, length, column names) of the per-step physics outputs."""
        ph, g = self.ops.phys, self._geom
        lay: list[tuple[str, int, list[str]]] = []
        if ph is None:
            return lay
        tg = g.thermal
        if ph.thermal:
            if tg is not None and tg.temp_ids:
                lay.append(("heat", len(tg.temp_ids), [f"heat:{self._bodies[i].name or i}" for i in tg.temp_ids]))
            if tg is not None and tg.flux_ids:
                lay.append(("theta_wall", len(tg.flux_ids),
                            [f"theta_wall:{self._bodies[i].name or i}" for i in tg.flux_ids]))
            lay.append(("wall_heat", 4, ["heat:left", "heat:right", "heat:bottom", "heat:top"]))
            lay.append(("heat_storage", 1, ["heat:storage"]))
            lay.append(("flow_heat", 2, ["heat:inflow", "heat:outflow"]))
            if ph.ports:
                lay.append(("vent_heat", 1, ["heat:vents"]))
            if ph.pt_chi is not None:
                lay.append(("jet_heat", 1, ["heat:jets"]))
        if ph.nfan:
            names = [f.name or f"fan{k}" for k, f in enumerate(ph.specs.get("fans", []))]
            lay.append(("fan_thrust", ph.nfan, [f"thrust:{n}" for n in names]))
            lay.append(("fan_ud", ph.nfan, [f"disc_velocity:{n}" for n in names]))
        if ph.paint:
            lay.append(("paint_force", 2, ["paint:cd", "paint:cl"]))
        return lay

    def _setup_outputs(self) -> None:
        """(Re)allocate the static output buffers and the physics history ring."""
        lay = self._out_layout()
        self._out = {k: torch.zeros(n, device=self.device, dtype=self.dtype) for k, n, _ in lay}
        self._out_keys = tuple(k for k, _, _ in lay)
        cols = ("t", *[c for _, _, cs in lay for c in cs])
        if lay and (self._aux is None or self._aux.columns != cols):
            if self._aux is not None:
                self._aux.flush()
            self._aux = _Ring(cols, self.device, self.dtype)
            self._phys_avg = torch.zeros(len(cols) - 1, device=self.device, dtype=torch.float64)
        elif not lay:
            self._aux = None
            self._phys_avg = None
        self._rebuild_solid()

    def _setup_particles(self) -> None:
        before = self._particles
        if particle_emitters(self.cfg.emitters) and self.cfg.particle_capacity > 0:
            if self._particles is None or not self._particles.update_emitters(self.cfg.emitters):
                self._particles = TracerParticles(self.cfg.emitters, self.ops, self.cfg.particle_capacity)
        else:
            self._particles = None
        if self._particles is not before:
            self._solid_c = None
            self._invalidate_graphs()          # captured graphs hold the old pool's buffers
        self._rebuild_solid()

    def _rebuild_solid(self) -> None:
        """Full-field solid fraction for killing particles (bodies + paint), in place when possible."""
        if self._particles is None:
            self._solid_c = None
            return
        m = self._mask_field()
        if self._solid_c is None or self._solid_c.shape != m.shape:
            self._solid_c = m
        else:
            self._solid_c.copy_(m)

    @property
    def _want_out(self) -> bool:
        return bool(self._out) or self._diag is not None

    def _live_window(self, bodies: list[ImmersedBody]) -> tuple[int, int, int, int]:
        """Body window large enough for the bodies pitched by +-LIVE_PITCH_DEG about
        the moment reference, so angle-of-attack edits in the GUI reuse the
        mask tensors (and any captured CUDA graph). The window is still a small
        fraction of the domain, and penalisation cost scales with its area."""
        if not bodies:
            return (0, 0, 0, 0)
        xr, yr = self.cfg.moment_ref
        pts = []
        for ang in np.radians(np.linspace(-LIVE_PITCH_DEG, LIVE_PITCH_DEG, 11)):
            c, sn = math.cos(ang), math.sin(ang)
            for b in bodies:
                p = b.points - (xr, yr)
                pts.append(np.c_[c * p[:, 0] - sn * p[:, 1], sn * p[:, 0] + c * p[:, 1]] + (xr, yr))
        return body_window(pts, self.ops)

    # ----------------------------------------------------------------- props
    @property
    def t(self) -> float:
        """Simulated time in convective units (syncs with the device)."""
        return float(self._state.t)

    @property
    def dt(self) -> float:
        """The most recent time step (syncs); before the first step, the step the CFL rule would take."""
        d = float(self._state.dt)
        if d > 0:
            return d
        if self.cfg.dt is not None:
            return float(self.cfg.dt)
        from .core import cfl_dt
        return float(cfl_dt(self._state.u, self._state.v, self.ops))

    @property
    def state(self) -> State:
        """The live state tensors (read-only use, please)."""
        return self._state

    @property
    def geometry(self) -> Geometry:
        return self._geom

    @property
    def bodies(self) -> list[ImmersedBody]:
        return list(self._bodies)

    @property
    def uses_cuda_graphs(self) -> bool:
        return self._use_graphs

    @property
    def averaging_time(self) -> float:
        """Convective time covered by the running means (0 when not averaging)."""
        return float(self._avg["time"]) if self._avg is not None else 0.0

    @property
    def physics(self):
        """The placed-physics tensors (``ops.phys``) or None."""
        return self.ops.phys

    @property
    def peclet(self) -> float:
        """Pe = Re Pr of the temperature equation (nan without heat transfer)."""
        th = self.cfg.thermal
        return self.cfg.re * th.prandtl if th is not None else float("nan")

    # ---------------------------------------------------------------- stepping
    def _buffers(self) -> list[torch.Tensor]:
        s = self._state
        bufs = [s.u, s.v, s.p, s.t, s.dt, self._body_c, self._hist, self._hist_n]
        for t in (s.dye, s.theta, s.dye_extra, s.ctrl):
            if t is not None:
                bufs.append(t)
        if self._avg is not None:
            bufs.extend(self._avg.values())
        bufs.extend(self._out.values())
        for r in (self._aux, self._diag):
            if r is not None:
                bufs.extend([r.buf, r.count])
        if self._step_dev is not None:
            bufs.append(self._step_dev)
        if self._phys_avg is not None:
            bufs.append(self._phys_avg)
        if self._particles is not None:
            bufs.extend(self._particles.buffers())
        return bufs

    def _step_static(self, diag: bool = False) -> None:
        """One step written into the static buffers (graph-capturable)."""
        st = self._state
        out = {} if self._want_out else None
        new, bc = step_full(st, self._geom, self.ops, None, out=out, diag=diag and self._diag is not None)
        st.u.copy_(new.u)
        st.v.copy_(new.v)
        st.p.copy_(new.p)
        for name in ("dye", "theta", "dye_extra", "ctrl"):
            dst = getattr(st, name)
            if dst is not None:
                dst.copy_(getattr(new, name))
        st.t.copy_(new.t)
        st.dt.copy_(new.dt)
        if self._particles is not None:
            self._particles.step(st.u, st.v, st.dt, st.t, self._solid_c)
        self._record(new, bc, out)

    def _step_eager(self, diag: bool = False) -> None:
        out = {} if self._want_out else None
        new, bc = step_full(self._state, self._geom, self.ops, None, out=out, diag=diag and self._diag is not None)
        self._state = new
        if self._particles is not None:
            self._particles.step(new.u, new.v, new.dt, new.t, self._solid_c)
        self._record(new, bc, out)

    def _record(self, new: State, bc: torch.Tensor, out: dict | None = None) -> None:
        if bc.shape[0]:
            self._body_c.copy_(bc)
        row = torch.cat([new.t.reshape(1), bc.sum(0)]).reshape(1, 4)
        self._hist.index_copy_(0, torch.remainder(self._hist_n, HISTORY_CAPACITY).reshape(1), row)
        self._hist_n.add_(1)
        if out is not None:
            for k, buf in self._out.items():
                if k in out:
                    buf.copy_(out[k])
            if self._aux is not None:
                arow = torch.cat([new.t.reshape(1), *[self._out[k] for k in self._out_keys]])
                self._aux.write(arow)
                if self._avg_on and self._phys_avg is not None:
                    self._phys_avg.add_(arow[1:].to(torch.float64) * new.dt.to(torch.float64))
            if self._diag is not None:
                self._step_dev.add_(1)
                if "diag" in out:
                    self._diag.write(torch.cat([self._step_dev.reshape(1), out["diag"].to(self._diag.buf.dtype)]))
        if self._avg_on:
            a = self._avg
            dt = new.dt.to(torch.float64)
            u, v, p = new.u.to(torch.float64), new.v.to(torch.float64), new.p.to(torch.float64)
            a["u"].addcmul_(u, dt)
            a["v"].addcmul_(v, dt)
            a["p"].addcmul_(p, dt)
            a["uu"].addcmul_(u * u, dt)
            a["vv"].addcmul_(v * v, dt)
            a["pp"].addcmul_(p * p, dt)
            if "th" in a and new.theta is not None:
                th = new.theta.to(torch.float64)
                a["th"].addcmul_(th, dt)
                a["thth"].addcmul_(th * th, dt)
            a["time"].add_(dt)

    def _capture(self, k: int, diag_last: bool) -> torch.cuda.CUDAGraph:
        """Capture k steps. Warm-up steps run on a side stream (cuFFT plans,
        allocator) and are then undone, so capturing never changes the run."""
        snap = [b.clone() for b in self._buffers()]
        try:
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for i in range(2):
                    self._step_static(diag=diag_last and i == 1)
            torch.cuda.current_stream(self.device).wait_stream(side)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                for i in range(k):
                    self._step_static(diag=diag_last and i == k - 1)
        finally:                              # also on failure: the warm-up must not advance the run
            for b, c in zip(self._buffers(), snap):
                b.copy_(c)
        return g

    def _run_graph(self, k: int, times: int, diag_last: bool = False) -> bool:
        key = (k, self._avg_on, diag_last)
        try:
            g = self._graphs.get(key)
            if g is None:
                g = self._graphs[key] = self._capture(k, diag_last)
            for _ in range(times):
                g.replay()
            return True
        except Exception as exc:  # noqa: BLE001 - any capture/replay failure must fall back to eager
            self.graph_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"CUDA graph path disabled, falling back to eager execution ({self.graph_error})",
                          RuntimeWarning, stacklevel=3)
            self._use_graphs = False
            self._graphs.clear()
            with contextlib.suppress(Exception):     # best effort: drain the stream before going eager
                torch.cuda.synchronize(self.device)
            return False

    def _invalidate_graphs(self) -> None:
        self._graphs.clear()

    def advance(self, n: int) -> None:
        """Advance ``n`` time steps on the fast path (no autograd)."""
        n = int(n)
        if n <= 0:
            return
        t_wall = time.perf_counter()
        diag_on = self._diag is not None
        # on the CPU, flush subnormals to zero while stepping: decaying tails of smooth
        # fields (eddy viscosity, dye, wakes) otherwise reach the subnormal range, where
        # x86 arithmetic is ~100x slower; FTZ cannot change any result that matters at
        # single or double precision. The thread's previous mode is restored afterwards
        # (left on, it crashes scipy's k-d trees: solvers/fpmode.py).
        with flush_denormal(self.device.type == "cpu"), torch.no_grad():
            left = n
            while left > 0:
                room = HISTORY_CAPACITY - (self._rows - self._flushed)
                chunk = min(left, room)
                done = 0
                if self._use_graphs:
                    big, rest = divmod(chunk, GRAPH_STEPS)
                    if big and self._run_graph(GRAPH_STEPS, big, diag_on):
                        done += big * GRAPH_STEPS
                        if diag_on:
                            self._diag.rows += big
                    if self._use_graphs and rest and self._run_graph(1, rest):
                        done += rest
                for i in range(chunk - done):
                    k = self.steps + done + i
                    d = diag_on and (k % DIAG_EVERY == DIAG_EVERY - 1)
                    self._step_eager(d) if not self._use_graphs else self._step_static(d)
                    if d:
                        self._diag.rows += 1
                self._rows += chunk
                if self._aux is not None:
                    self._aux.rows += chunk
                self.steps += chunk
                left -= chunk
                if self._rows - self._flushed >= HISTORY_CAPACITY:
                    self._flush()
            self._flush()
        self._perf.append((self.steps, n, time.perf_counter() - t_wall))
        if len(self._perf) > 10000:
            del self._perf[:5000]

    def _flush(self) -> None:
        """Copy unread history rows from the device rings to the host (one sync)."""
        k = self._rows - self._flushed
        if k > 0:
            a = self._flushed % HISTORY_CAPACITY
            b = a + k
            if b <= HISTORY_CAPACITY:
                rows = self._hist[a:b]
            else:
                rows = torch.cat([self._hist[a:], self._hist[:b - HISTORY_CAPACITY]])
            self._host_hist.append(rows.detach().to("cpu", torch.float64).numpy())
            self._flushed = self._rows
        for r in (self._aux, self._diag):
            if r is not None:
                r.flush()

    # ----------------------------------------------------------- coefficients
    def coefficients(self) -> np.ndarray:
        """Latest [Cd, Cl, Cm] summed over all bodies."""
        return self._body_c.sum(0).detach().to("cpu", torch.float64).numpy()

    def body_coefficients(self) -> np.ndarray:
        """Latest [Cd, Cl, Cm] per body, shape (nbody, 3)."""
        return self._body_c.detach().to("cpu", torch.float64).numpy().copy()

    def history(self) -> dict[str, np.ndarray]:
        """t, cd, cl, cm for every step since the last reset."""
        self._flush()
        h = np.concatenate(self._host_hist) if self._host_hist else np.zeros((0, 4))
        return {"t": h[:, 0].copy(), "cd": h[:, 1].copy(), "cl": h[:, 2].copy(), "cm": h[:, 3].copy()}

    def physics_history(self) -> dict[str, np.ndarray]:
        """Per-step history of the placed-physics outputs since the last reset.

        Keys: "t", "heat:<body>" (heat rate of each temperature body into the
        fluid), "theta_wall:<body>" (mean wall temperature of each heat-flux
        body), "heat:left|right|bottom|top" (domain walls), "thrust:<fan>",
        "disc_velocity:<fan>", "paint:cd|cl". Empty without placed physics.
        Non-dimensional (heat rates per unit span in rho c_p U dT L).
        """
        if self._aux is None:
            return {}
        t = self._aux.table()
        return {c: t[:, i].copy() for i, c in enumerate(self._aux.columns)}

    def trim_history(self, keep: float) -> None:
        """Forget host history older than ``keep`` convective times (long explore sessions)."""
        h = self.history()
        if not len(h["t"]):
            return
        sel = h["t"] >= h["t"][-1] - keep
        i0 = int(np.argmax(sel))
        self._host_hist[:] = [np.c_[h["t"], h["cd"], h["cl"], h["cm"]][i0:]]
        if self._aux is not None:
            ta = self._aux.table()
            if len(ta):
                self._aux.trim(int(np.argmax(ta[:, 0] >= h["t"][-1] - keep)))
        if self._diag is not None:
            td = self._diag.table()
            if len(td):
                self._diag.trim(int(np.argmax(td[:, 1] >= h["t"][-1] - keep)))

    # ------------------------------------------------------------------ fields
    def _p_ref(self, p: torch.Tensor) -> torch.Tensor:
        """Reference static pressure: mean over the two inlet columns (domain
        mean in periodic and closed domains). cp is relative to it."""
        if self.ops.periodic or self.ops.closed:
            return p.mean()
        return p[:, :2].mean()

    def _mask_field(self) -> torch.Tensor:
        m = torch.zeros(self.ops.ny, self.ops.nx, device=self.device, dtype=self.dtype)
        if not self._geom.empty:
            j0, j1, i0, i1 = self._geom.window
            m[j0:j1, i0:i1] = self._geom.chi_c
        ph = self.ops.phys
        if ph is not None and ph.paint:
            m = torch.maximum(m, ph.paint_c)
        return m

    def _nut_field(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.ops.turbulence == "none":
            return torch.zeros(self.ops.ny, self.ops.nx, device=self.device, dtype=self.dtype)
        g11, g12, g21, g22, _ = _gradients(pad_u(u, self.ops), pad_v(v, self.ops), self.ops)
        nut = eddy_viscosity(g11, g12, g21, g22, self.ops)[1:-1, 1:-1]
        return nut * (1.0 - self._mask_field())

    def field(self, name: str) -> torch.Tensor:
        """A cell-centred (ny, nx) field of the current state (a new tensor).

        "u", "v", "speed": velocity (units of U); "p": pressure relative to the
        inlet (units of rho U^2); "cp" = 2 p; "vorticity" (U/L); "mask": solid
        fraction (bodies and painted obstacles); "dye": smoke concentration 0..1
        (channel 0); "nut": eddy viscosity (units of U L); "divergence": discrete
        div u (U/L, round-off level; jets / vents show their mass source).
        Placed physics: "theta" temperature (T - T_amb) / dT_ref, "dye1".."dye3"
        the other smoke channels, "paint" the painted solid fraction,
        "heat_source" the volumetric heater density. Missing features give zeros.
        """
        s = self._state
        with torch.no_grad():
            if name == "u":
                return centre_u(s.u)
            if name == "v":
                return centre_v(s.v)
            if name == "speed":
                return torch.hypot(centre_u(s.u), centre_v(s.v))
            if name == "p":
                return s.p - self._p_ref(s.p)
            if name == "cp":
                return 2.0 * (s.p - self._p_ref(s.p))
            if name == "vorticity":
                return vorticity(s.u, s.v, self.ops)
            if name == "mask":
                return self._mask_field()
            if name == "dye":
                return s.dye.clone() if s.dye is not None else torch.zeros_like(s.p)
            if name == "nut":
                return self._nut_field(s.u, s.v)
            if name == "divergence":
                return divergence(s.u, s.v, self.ops.h)
            if name == "theta":
                return s.theta.clone() if s.theta is not None else torch.zeros_like(s.p)
            if name in ("dye1", "dye2", "dye3"):
                k = int(name[-1]) - 1
                if s.dye_extra is not None and k < s.dye_extra.shape[0]:
                    return s.dye_extra[k].clone()
                return torch.zeros_like(s.p)
            ph = self.ops.phys
            if name == "paint":
                return ph.paint_c.clone() if ph is not None and ph.paint else torch.zeros_like(s.p)
            if name == "heat_source":
                return ph.heat.clone() if ph is not None and ph.heat is not None else torch.zeros_like(s.p)
        raise KeyError(f"unknown field {name!r}; available: {', '.join(FIELDS + PHYSICS_FIELDS)}")

    def set_temperature(self, theta) -> None:
        """Overwrite the temperature field (an (ny, nx) array or tensor), e.g. an initial perturbation."""
        s = self._state
        if s.theta is None:
            raise RuntimeError("this solver carries no temperature (SolverConfig.thermal is None)")
        with torch.no_grad():
            s.theta.copy_(torch.as_tensor(np.asarray(theta) if not torch.is_tensor(theta) else theta,
                                          device=self.device, dtype=self.dtype).reshape(s.theta.shape))

    # --------------------------------------------------------------- averaging
    def start_averaging(self) -> None:
        """(Re)start time-weighted running means of u, v, p (and theta) and their squares."""
        with torch.no_grad():
            s, f64 = self._state, torch.float64
            if self._avg is None or (s.theta is not None and "th" not in self._avg):
                self._avg = {
                    "u": torch.zeros_like(s.u, dtype=f64), "v": torch.zeros_like(s.v, dtype=f64),
                    "p": torch.zeros_like(s.p, dtype=f64), "uu": torch.zeros_like(s.u, dtype=f64),
                    "vv": torch.zeros_like(s.v, dtype=f64), "pp": torch.zeros_like(s.p, dtype=f64),
                    "time": torch.zeros((), device=self.device, dtype=f64),
                }
                if s.theta is not None:
                    self._avg["th"] = torch.zeros_like(s.theta, dtype=f64)
                    self._avg["thth"] = torch.zeros_like(s.theta, dtype=f64)
                # graphs captured without averaging do not touch these buffers;
                # graphs keyed with averaging on are captured lazily
                self._graphs = {k: g for k, g in self._graphs.items() if not k[1]}
            else:
                for a in self._avg.values():
                    a.zero_()
            if self._phys_avg is not None:
                self._phys_avg.zero_()
            self._avg_on = True

    def stop_averaging(self) -> None:
        """Freeze the running means (they stay readable)."""
        self._avg_on = False

    def mean_fields(self) -> dict[str, torch.Tensor]:
        """Time averages since :meth:`start_averaging`, cell-centred (ny, nx).

        Keys: u, v, speed, p, cp, vorticity (of the mean flow), u_rms, v_rms,
        p_rms, cp_rms, mask, and with heat transfer theta, theta_rms. Falls back
        to the instantaneous fields if no time has been averaged yet.
        """
        if self._avg is None:
            raise RuntimeError("call start_averaging() first")
        with torch.no_grad():
            T = float(self._avg["time"])
            if T <= 0:
                out = {k: self.field(k) for k in ("u", "v", "speed", "p", "cp", "vorticity", "mask")}
                for k in ("u_rms", "v_rms", "p_rms", "cp_rms"):
                    out[k] = torch.zeros_like(out["u"])
                if self._state.theta is not None:
                    out["theta"] = self.field("theta")
                    out["theta_rms"] = torch.zeros_like(out["u"])
                return out
            a, dt = self._avg, self.dtype
            mu, mv, mp = a["u"] / T, a["v"] / T, a["p"] / T
            ru = (a["uu"] / T - mu * mu).clamp_min(0).sqrt()
            rv = (a["vv"] / T - mv * mv).clamp_min(0).sqrt()
            rp = (a["pp"] / T - mp * mp).clamp_min(0).sqrt()
            mu_, mv_, mp_ = mu.to(dt), mv.to(dt), mp.to(dt)
            pg = mp_ - self._p_ref(mp_)
            uc, vc = centre_u(mu_), centre_v(mv_)
            out = {"u": uc, "v": vc, "speed": torch.hypot(uc, vc), "p": pg, "cp": 2 * pg,
                   "vorticity": vorticity(mu_, mv_, self.ops),
                   "u_rms": centre_u(ru.to(dt)), "v_rms": centre_v(rv.to(dt)), "p_rms": rp.to(dt),
                   "cp_rms": 2 * rp.to(dt), "mask": self._mask_field()}
            if "th" in a:
                mt = a["th"] / T
                out["theta"] = mt.to(dt)
                out["theta_rms"] = (a["thth"] / T - mt * mt).clamp_min(0).sqrt().to(dt)
            return out

    # ----------------------------------------------------------------- surface
    def _probe_distances(self) -> tuple[float, float]:
        """Wall-normal probe distances (solver units) used by :meth:`surface`.

        The smoothed mask ends ``(mask_width - MASK_SHIFT) h`` = 0.68 h outside
        the surface, so probes at 1.5 h and 2.5 h sit in the fluid while staying
        well inside any resolved boundary layer.
        """
        d1 = max(1.5, self.cfg.mask_width + 0.5) * self.ops.h
        return d1, d1 + self.ops.h

    def _staggered_means(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Time-averaged u, v, p on their own (staggered) grids, solver dtype."""
        if self._avg is None:
            raise RuntimeError("call start_averaging() first")
        T = float(self._avg["time"])
        if T <= 0:
            s = self._state
            return s.u, s.v, s.p
        a = self._avg
        return (a["u"] / T).to(self.dtype), (a["v"] / T).to(self.dtype), (a["p"] / T).to(self.dtype)

    def surface(self, n_per_body: int = 400, mean: bool = False) -> list[dict]:
        """Surface distributions of cp and cf per body (instantaneous or time-averaged).

        For each body: ``s`` (arc length from vertex 0, counter-clockwise),
        ``x``, ``y``, outward normal ``nx``, ``ny``, ``cp`` and ``cf`` (numpy
        float64, ``n_per_body`` points equally spaced in arc length), plus
        ``name``. Positive cf means wall shear along increasing s. With heat
        transfer also ``nu_local``: the local Nusselt number Pe dtheta/dn /
        (theta_wall - theta_ref) from the same two-probe quadratic (Nu based
        on the reference length; for heat-flux bodies theta_wall is the local
        extrapolated wall temperature).

        The wall itself is diffuse (the penalised layer reaches 0.68 h into the
        fluid), so wall values are *extrapolated* from two probes along the
        outward normal at d1 = 1.5 h and d2 = 2.5 h, just outside the layer:

        * cp: linear extrapolation of the probe pressures to d = 0 (it removes
          the centrifugal pressure gradient across the probe distance);
        * cf = (2/Re) du_t/dn at the wall from the quadratic through the wall
          velocity (zero, or omega x r on a spinning wheel) and the tangential
          velocity at both probes (second order; molecular viscosity only).

        Velocities are interpolated from their staggered faces and the pressure
        is the step-averaged pressure of the momentum balance, so integrating
        ``-cp n + cf t`` around a resolved body reproduces the penalisation
        force: cylinder Re 40 at 32 / 48 cells/D gives Cd 1.619 / 1.617 from the
        surface vs 1.619 / 1.616 from the impulse; Re 100 (time mean) 2.1 / 0.7 %
        low at 32 / 64 cells/D (docs/theory_ns2d.md, sections 6.5 and 11). Thin
        boundary layers (fewer than ~3 cells, i.e. high Re) make cf a lower
        bound; probes of a body that land inside another body, or outside the
        domain, return whatever is there (clamped), so treat contact regions
        (wheel on road) as undefined.
        """
        if mean:
            U, V, P = self._staggered_means()
        else:
            U, V, P = self._state.u, self._state.v, self._state.p
        TH = None
        if self._state.theta is not None:
            if mean and self._avg is not None and "th" in self._avg and float(self._avg["time"]) > 0:
                TH = (self._avg["th"] / float(self._avg["time"])).to(self.dtype)
            else:
                TH = self._state.theta
        ops = self.ops
        d1, d2 = self._probe_distances()
        with torch.no_grad():
            p_ref = self._p_ref(P)
        out = []
        th_ref = float(self.cfg.thermal.theta_inlet) if self.cfg.thermal is not None else 0.0
        for b in self._bodies:
            pts, s = resample_closed(b.points, n_per_body)
            tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
            tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-30)[:, None]
            nrm = np.c_[tang[:, 1], -tang[:, 0]]                 # outward for a CCW outline
            q = torch.as_tensor(np.concatenate([pts + d1 * nrm, pts + d2 * nrm]),
                                device=self.device, dtype=self.dtype)
            with torch.no_grad():
                pq = bilinear(P, q[:, 0], q[:, 1], ops) - p_ref
                uq = bilinear(U, q[:, 0], q[:, 1], ops, 0.0, 0.5)
                vq = bilinear(V, q[:, 0], q[:, 1], ops, 0.5, 0.0)
                tq = bilinear(TH, q[:, 0], q[:, 1], ops) if TH is not None else None
            pq, uq, vq = (t.cpu().double().numpy().reshape(2, -1) for t in (pq, uq, vq))
            cp = 2.0 * (pq[0] + (pq[0] - pq[1]) * d1 / (d2 - d1))
            ut = uq * tang[None, :, 0] + vq * tang[None, :, 1]
            # wall tangential velocity of a spinning body: (omega x r) . t
            rx, ry = pts[:, 0] - b.center[0], pts[:, 1] - b.center[1]
            uw = b.omega * (-ry * tang[:, 0] + rx * tang[:, 1])
            dudn = ((ut[0] - uw) * d2 * d2 - (ut[1] - uw) * d1 * d1) / (d1 * d2 * (d2 - d1))
            cf = 2.0 / self.cfg.re * dudn
            rec = {"name": b.name, "s": s, "x": pts[:, 0].copy(), "y": pts[:, 1].copy(),
                   "nx": nrm[:, 0].copy(), "ny": nrm[:, 1].copy(), "cp": cp, "cf": cf}
            if tq is not None:
                tq = tq.cpu().double().numpy().reshape(2, -1)
                if b.thermal == "temperature":
                    tw = np.full(len(s), b.thermal_value)
                else:                                   # extrapolate the wall value linearly
                    tw = tq[0] + (tq[0] - tq[1]) * d1 / (d2 - d1)
                # outward gradient from the quadratic through the wall value and both probes
                dtdn = ((tq[0] - tw) * d2 * d2 - (tq[1] - tw) * d1 * d1) / (d1 * d2 * (d2 - d1))
                rec["theta_wall"] = tw
                # local Nu = h L / k = -(dtheta/dn) / (theta_wall - theta_ref) with n outward, L = 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    rec["nu_local"] = -dtdn / np.where(np.abs(tw - th_ref) > 1e-12, tw - th_ref, np.nan)
            out.append(rec)
        return out

    # ------------------------------------------------------- physics reports
    def heat_transfer(self, mean: bool = False, theta_ref: float | None = None) -> dict:
        """Heat transfer of the bodies and the domain walls (non-dimensional).

        Per body: ``heat_rate`` (heat into the fluid per unit span, units of
        rho c_p U dT_ref L; from the penalisation impulse for "temperature"
        bodies, the prescribed G P / Pe for "heat_flux" bodies, 0 for adiabatic),
        ``theta_wall`` (the held temperature, or the mean skin temperature of a
        heat-flux body), and ``nusselt`` = Pe Q / ((theta_wall - theta_ref) P),
        the mean Nusselt number based on the reference length L (DESIGN.md
        10.2: Nu = Q L / (k dT A)). ``walls`` gives each side's heat rate and
        ``nusselt_mean`` averages the heated bodies. ``mean`` uses the running
        averages (since :meth:`start_averaging`).
        """
        if self._state.theta is None:
            return {"bodies": [], "walls": {}, "nusselt_mean": float("nan"), "pe": float("nan")}
        vals = self._phys_values(mean)
        pe = self.peclet
        ref = float(self.cfg.thermal.theta_inlet) if theta_ref is None else float(theta_ref)
        tg = self._geom.thermal
        perims = tg.perimeter.detach().cpu().double().numpy() if tg is not None else None
        bodies, nus = [], []
        for i, b in enumerate(self._bodies):
            # heat-exchanging outline: the part of the surface inside the domain
            per = float(perims[i]) if perims is not None else b.perimeter
            rec = {"name": b.name, "mode": b.thermal, "perimeter": per, "heat_rate": 0.0,
                   "theta_wall": float("nan"), "nusselt": float("nan")}
            if tg is not None and i in tg.temp_ids:
                rec["heat_rate"] = vals.get(f"heat:{b.name or i}", float("nan"))
                rec["theta_wall"] = b.thermal_value
            elif tg is not None and i in tg.flux_ids:
                rec["heat_rate"] = b.thermal_value * per / pe
                rec["theta_wall"] = vals.get(f"theta_wall:{b.name or i}", float("nan"))
            dth = rec["theta_wall"] - ref
            if b.thermal != "adiabatic" and abs(dth) > 1e-12 and per > 0:
                rec["nusselt"] = pe * rec["heat_rate"] / (dth * per)
                nus.append(rec["nusselt"])
            bodies.append(rec)
        walls = {s: vals.get(f"heat:{s}", 0.0) for s in ("left", "right", "bottom", "top")}
        return {"bodies": bodies, "walls": walls, "pe": pe,
                "nusselt_mean": float(np.mean(nus)) if nus else float("nan")}

    def heat_balance(self, mean: bool = False) -> dict[str, float]:
        """Energy balance of the latest step (or of the running means), per unit span in the
        solver's heat-rate unit rho c_p U dT_ref L (multiply by Pe for conduction units k dT_ref).

        Keys (the ones ``aerolume.analysis.physics`` reads; positive "in" adds heat to the
        fluid, positive "out" removes it): ``heat_in_bodies`` (temperature bodies from the
        penalisation impulse, heat-flux bodies G P / Pe), ``heat_in_walls`` (domain walls
        other than the inlet and outlet), ``heat_in_sources`` (heaters), ``heat_in_ports``
        (the jets' enthalpy), ``heat_in_inlet`` (advection plus conduction through the
        inlet), ``heat_out_outlet`` (advection through the outlet), ``heat_out_ports``
        (vents), ``heat_storage_rate`` (d/dt of the heat content over the step) and
        ``heat_balance_residual`` = in - out - storage. The terms are the scheme's own
        fluxes evaluated at the end of the step, so the residual is round-off in a steady
        state and O(dt) during transients (the Runge-Kutta stages see intermediate fields).
        Empty without heat transfer.
        """
        if self._state.theta is None:
            return {}
        vals = self._phys_values(mean)
        ph = self.ops.phys
        ht = self.heat_transfer(mean=mean)
        bodies = float(sum(b["heat_rate"] for b in ht["bodies"] if b["mode"] != "adiabatic"))
        modes = dict(zip(("left", "right", "bottom", "top"), ph.wall_modes))
        walls = sum(vals.get(f"heat:{side}", 0.0) for side, m in modes.items() if m not in ("inlet", "outlet"))
        inlet = vals.get("heat:inflow", 0.0) + sum(vals.get(f"heat:{side}", 0.0) for side, m in modes.items()
                                                    if m == "inlet")
        sources = float(ph.heat.double().sum()) * self.ops.h ** 2 if ph.heat is not None else 0.0
        out = {"heat_in_bodies": bodies, "heat_in_walls": float(walls), "heat_in_sources": sources,
               "heat_in_ports": vals.get("heat:jets", 0.0), "heat_in_inlet": float(inlet),
               "heat_out_outlet": vals.get("heat:outflow", 0.0), "heat_out_ports": vals.get("heat:vents", 0.0),
               "heat_storage_rate": vals.get("heat:storage", float("nan"))}
        tot_in = sum(v for k, v in out.items() if k.startswith("heat_in"))
        tot_out = out["heat_out_outlet"] + out["heat_out_ports"]
        out["heat_balance_residual"] = tot_in - tot_out - out["heat_storage_rate"]
        return out

    def fans_report(self, mean: bool = False) -> list[dict]:
        """Per fan: thrust, pressure jump, flow rate, disc and induced velocity, ideal power.

        Non-dimensional: thrust per unit span (rho U^2 L), pressure jump
        (rho U^2), volume flow per unit span (U L), velocities (U), power
        thrust x disc velocity (rho U^3 L). The induced velocity is the disc
        velocity minus the free stream's component along the fan axis (tunnel)
        or the disc velocity itself (closed domains).
        """
        ph = self.ops.phys
        if ph is None or not ph.nfan:
            return []
        vals = self._phys_values(mean)
        out = []
        for k, f in enumerate(ph.specs["fans"]):
            name = f.name or f"fan{k}"
            T = vals.get(f"thrust:{name}", float("nan"))
            ud = vals.get(f"disc_velocity:{name}", float("nan"))
            u_inf = 0.0 if (self.ops.closed or self.ops.periodic) else float(f.direction[0])
            out.append({"name": name, "mode": f.mode, "thrust": T, "pressure_jump": T / f.width,
                        "flow_rate": ud * f.width, "disc_velocity": ud, "induced_velocity": ud - u_inf,
                        "power": T * ud, "target": f.value if f.mode == "velocity" else float("nan")})
        return out

    def painted_force(self, mean: bool = False) -> np.ndarray:
        """[Cd, Cl] of the painted obstacles (zeros without paint)."""
        vals = self._phys_values(mean)
        return np.array([vals.get("paint:cd", 0.0), vals.get("paint:cl", 0.0)])

    def _phys_values(self, mean: bool) -> dict[str, float]:
        if self._aux is None:
            return {}
        cols = self._aux.columns[1:]
        if mean and self._avg is not None and float(self._avg["time"]) > 0 and self._phys_avg is not None:
            v = (self._phys_avg / self._avg["time"]).cpu().numpy()
        else:
            v = torch.cat([self._out[k] for k in self._out_keys]).detach().cpu().double().numpy() \
                if self._out else np.zeros(0)
        return {c: float(x) for c, x in zip(cols, v)}

    def particles(self) -> dict[str, np.ndarray]:
        """Living tracer particles: x, y, age, nozzle, emitter, birth (empty without particle emitters)."""
        if self._particles is None:
            return {k: np.zeros(0) for k in ("x", "y", "age", "nozzle", "emitter", "birth")}
        return self._particles.snapshot(self.t)

    def particle_lines(self) -> list[np.ndarray]:
        """Streaklines (continuous particle emitters) and timelines (pulsed) as (n, 2) polylines."""
        return self._particles.lines(self.t) if self._particles is not None else []

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict[str, np.ndarray]:
        """Recorded diagnostics (every DIAG_EVERY steps): the columns of
        :data:`~.diagnostics.DIAG_COLUMNS`, plus "perf_steps" / "steps_per_s"
        (wall-clock throughput of each :meth:`advance` call, host-measured,
        including the one host synchronisation per call)."""
        out: dict[str, np.ndarray] = {}
        if self._diag is not None:
            t = self._diag.table()
            out = {c: t[:, i].copy() for i, c in enumerate(DIAG_COLUMNS)}
        if self._perf:
            p = np.asarray(self._perf, dtype=np.float64)
            out["perf_steps"] = p[:, 0]
            out["steps_per_s"] = p[:, 1] / np.maximum(p[:, 2], 1e-12)
        return out

    def latest_diagnostics(self) -> dict[str, float]:
        """The last diagnostics row as plain floats (for CFDResult.diagnostics)."""
        d = self.diagnostics()
        out = {k: float(v[-1]) for k, v in d.items() if len(v) and k not in ("perf_steps",)}
        return out

    def diagnostics_csv(self, path) -> str:
        """Write the diagnostics history as CSV (one row per recorded step)."""
        if self._diag is None:
            raise RuntimeError("diagnostics are off (SolverConfig.diagnostics=False)")
        t = self._diag.table()
        return str(write_csv(path, list(DIAG_COLUMNS), t))

    def trace_step(self, commit: bool = False) -> dict[str, torch.Tensor]:
        """Run one step eagerly and return its stages as cell-centred fields.

        Keys "stage1..3/explicit_u|v" (after advection-diffusion and forces),
        "stage*/penalised_u|v" (after the solids), "stage*/divergence" (the
        Poisson right-hand side), "stage*/pressure_potential", "stage*/correction_u|v"
        (minus the gradient), "stage*/projected_u|v", with temperature
        "stage*/explicit_theta" and "stage*/theta", and "final_u|v|p|theta".
        ``commit=False`` (default) leaves the run untouched (the step is computed
        on the side); ``commit=True`` keeps it as a normal step.
        """
        tr: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            out = {} if self._want_out else None
            new, bc = step_full(self._state, self._geom, self.ops, None, out=out, trace=tr)
            if commit:
                st = self._state
                if self._use_graphs:
                    for name in ("u", "v", "p", "dye", "t", "dt", "theta", "dye_extra", "ctrl"):
                        dst = getattr(st, name)
                        if dst is not None:
                            dst.copy_(getattr(new, name))
                else:
                    self._state = new
                self._record(new, bc, out)
                self._rows += 1
                if self._aux is not None:
                    self._aux.rows += 1
                self.steps += 1
        return tr

    # -------------------------------------------------------------- live edits
    def set_inflow(self, *, re: float | None = None) -> None:
        """Change the Reynolds number live (in place: graphs stay valid). With
        heat transfer the Peclet number follows (Pe = Re Pr)."""
        if re is not None:
            if not (re > 0 and math.isfinite(re)):
                raise ValueError("Reynolds number must be positive and finite")
            self.cfg.re = float(re)
            with torch.no_grad():
                set_reynolds(self.ops, float(re))

    def set_bodies(self, bodies: list[ImmersedBody]) -> None:
        """Replace the bodies live; the flow field is kept.

        If the new bodies fit in the current window (same body count, same
        thermal kinds), the masks are rebuilt into the existing tensors and
        captured graphs remain valid; otherwise the window grows and graphs are
        re-captured lazily.
        """
        bodies = [b if isinstance(b, ImmersedBody) else ImmersedBody(np.asarray(b)) for b in bodies]
        win = body_window([b.points for b in bodies], self.ops)
        old = self._geom
        with torch.no_grad():
            g = None
            if (len(bodies) == len(self._bodies) and not old.empty and win[1] > win[0]
                    and window_contains(old.window, win)):
                g = build_geometry(bodies, self.ops, window=old.window)
                if g.signature() != old.signature():
                    g = None
            if g is not None:
                for dst, src in zip(old.tensors(), g.tensors()):
                    dst.copy_(src)
            else:
                self._geom = build_geometry(bodies, self.ops, window=self._live_window(bodies))
                if len(bodies) != self._body_c.shape[0]:
                    self._body_c = torch.zeros(len(bodies), 3, device=self.device, dtype=self.dtype)
                self._invalidate_graphs()
        self._bodies = bodies
        self.cfg.bodies = list(bodies)
        if g is None:
            self._setup_outputs()
        else:
            self._rebuild_solid()

    def set_emitters(self, emitters: list[EmitterSpec]) -> None:
        """Replace the smoke and particle emitters live (positions, rates, timing).

        Moving or retiming emitters rewrites their footprints in place; adding a
        smoke channel or a timing group re-captures the graphs once. Smoke already
        in the flow is kept.
        """
        emitters = [e if isinstance(e, EmitterSpec) else EmitterSpec(**e) for e in emitters]
        self._rebuild_physics(dataclasses.replace(self.cfg, emitters=emitters))
        self._setup_particles()

    def set_sources(self, *, heat_sources: list[HeatSourceSpec] | None = None, fans: list[FanSpec] | None = None,
                    ports: list[PortSpec] | None = None) -> None:
        """Replace heaters, fans and / or ports live (None leaves a kind unchanged).

        Moving a source or changing its value rewrites its tensors in place;
        adding the first fan, port or heater (or changing the number of fans)
        rebuilds the physics and re-captures the graphs lazily.
        """
        kw = {}
        if heat_sources is not None:
            kw["heat_sources"] = [q if isinstance(q, HeatSourceSpec) else HeatSourceSpec(**q) for q in heat_sources]
        if fans is not None:
            kw["fans"] = [f if isinstance(f, FanSpec) else FanSpec(**f) for f in fans]
        if ports is not None:
            kw["ports"] = [p if isinstance(p, PortSpec) else PortSpec(**p) for p in ports]
        if kw:
            self._rebuild_physics(dataclasses.replace(self.cfg, **kw))

    def set_thermal(self, *, richardson: float | None = None, prandtl: float | None = None,
                    walls: list[WallBC] | None = None, theta_inlet: float | None = None) -> None:
        """Change heat-transfer parameters live (in place unless buoyancy is switched on or off)."""
        th = self.cfg.thermal
        if th is None:
            raise RuntimeError("this solver carries no temperature (SolverConfig.thermal is None)")
        kw = {}
        if richardson is not None:
            kw["richardson"] = float(richardson)
        if prandtl is not None:
            kw["prandtl"] = float(prandtl)
        if walls is not None:
            kw["walls"] = list(walls)
        if theta_inlet is not None:
            kw["theta_inlet"] = float(theta_inlet)
        if kw:
            self._rebuild_physics(dataclasses.replace(self.cfg, thermal=dataclasses.replace(th, **kw)))

    def _rebuild_physics(self, cfg: SolverConfig) -> None:
        """Rebuild ``ops.phys`` for a new config; copy in place when the structure is unchanged."""
        old = self.ops.phys
        with torch.no_grad():
            new = phx.build_physics(cfg, self.ops)
            if old is not None and old.paint:
                if new is None:
                    new = phx.empty_physics(self.ops)
                phx.ensure_paint(new, self.ops)
                for name in ("paint_c", "paint_u", "paint_v"):
                    getattr(new, name).copy_(getattr(old, name))
                phx.combine_penalisation(new)
            same = old is not None and new is not None and old.signature() == new.signature()
            self.cfg = cfg
            self.ops.cfg = cfg
            if same:
                old.copy_from(new)
                old.buoyant = new.buoyant
                return
            self.ops.phys = new
            # state tensors whose existence or shape follows the structure
            st = init_state(cfg, self.ops)
            s = self._state
            theta = s.theta if (s.theta is not None and st.theta is not None) else st.theta
            dye_extra = st.dye_extra
            if dye_extra is not None and s.dye_extra is not None:
                k = min(dye_extra.shape[0], s.dye_extra.shape[0])
                dye_extra[:k] = s.dye_extra[:k]
            ctrl = st.ctrl
            if ctrl is not None and s.ctrl is not None:
                k = min(ctrl.shape[0], s.ctrl.shape[0])
                ctrl[:k] = s.ctrl[:k]
            dye = s.dye if s.dye is not None else st.dye
            self._state = State(s.u, s.v, s.p, dye, s.t, s.dt, theta, dye_extra, ctrl)
            if self._avg is not None and theta is not None and "th" not in self._avg:
                self._avg = None
                self._avg_on = False
            thermal_changed = (old is not None and old.thermal) != (new is not None and new.thermal)
            if thermal_changed:
                self._geom = build_geometry(self._bodies, self.ops, window=self._geom.window
                                            if not self._geom.empty else self._live_window(self._bodies))
            self._invalidate_graphs()
            self._setup_outputs()

    # ------------------------------------------------------------------ paint
    def _paint_layer(self):
        ph = self.ops.phys
        if ph is None:
            self.ops.phys = ph = phx.empty_physics(self.ops)
        if not ph.paint:
            phx.ensure_paint(ph, self.ops)
            self._invalidate_graphs()
            self._setup_outputs()
        return ph

    def _apply_paint(self, sdf, erase: bool) -> None:
        ph = self._paint_layer()
        with torch.no_grad():
            phx.paint_brush(ph, self.ops, sdf, self.cfg.mask_width, erase)
            phx.combine_penalisation(ph)
        self._rebuild_solid()

    def paint_disc(self, center: tuple[float, float], radius: float, *, erase: bool = False) -> None:
        """Add (or with ``erase`` remove) a solid disc - one brush dab of explore mode.

        The painted solid uses the bodies' calibrated interface (C1 step one cell
        wide, centred MASK_SHIFT cells inside the outline, evaluated on each grid),
        so its effective no-slip wall sits on the brush outline like a body's.
        Painted obstacles are penalised to rest, insulate heat and clear smoke; they
        live outside the differentiable body geometry (sensitivities never see
        them), and their total force is :meth:`painted_force`. Brush strokes rewrite
        tensors in place: no graph re-capture after the first stroke.
        """
        if not radius > 0:
            raise ValueError("brush radius must be positive")
        cx, cy, r = float(center[0]), float(center[1]), float(radius)
        self._apply_paint(lambda x, y: torch.hypot(x - cx, y - cy) - r, erase)

    def paint_polygon(self, points, *, erase: bool = False) -> None:
        """Add (or erase) a solid polygon (CCW (N, 2) outline in solver units)."""
        pts = torch.as_tensor(np.asarray(points, dtype=np.float64).reshape(-1, 2), device=self.device,
                              dtype=self.dtype)
        self._apply_paint(lambda x, y: signed_distance([pts], x, y), erase)

    def paint_snapshot(self) -> tuple[torch.Tensor, ...] | None:
        """Device copies of the paint layer (centres, u-, v-faces), or None without paint: a
        checkpoint that :meth:`restore_paint` puts back exactly (explore Undo, simulation.live)."""
        ph = self.ops.phys
        if ph is None or not ph.paint:
            return None
        with torch.no_grad():
            return tuple(t.clone() for t in (ph.paint_c, ph.paint_u, ph.paint_v))

    def restore_paint(self, snapshot) -> None:
        """Put back a :meth:`paint_snapshot` (None clears the layer); in place, graphs stay valid."""
        if snapshot is None:
            self.clear_paint()
            return
        ph = self._paint_layer()
        with torch.no_grad():
            for dst, src in zip((ph.paint_c, ph.paint_u, ph.paint_v), snapshot):
                dst.copy_(src)
            phx.combine_penalisation(ph)
        self._rebuild_solid()

    def clear_paint(self) -> None:
        """Remove every painted obstacle (the layer stays allocated, graphs stay valid)."""
        ph = self.ops.phys
        if ph is None or not ph.paint:
            return
        with torch.no_grad():
            for t in (ph.paint_c, ph.paint_u, ph.paint_v):
                t.zero_()
            phx.combine_penalisation(ph)
        self._rebuild_solid()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        """Back to the initial condition at t = 0; history and means cleared."""
        with torch.no_grad():
            fresh = init_state(self.cfg, self.ops)
            if self._use_graphs:
                for dst, src in zip(self._state, fresh):
                    if dst is not None and src is not None:
                        dst.copy_(src)
            else:
                self._state = fresh
            self._body_c.zero_()
            self._hist_n.zero_()
            if self._avg is not None:
                for a in self._avg.values():
                    a.zero_()
            for buf in self._out.values():
                buf.zero_()
            for r in (self._aux, self._diag):
                if r is not None:
                    r.clear()
            if self._step_dev is not None:
                self._step_dev.zero_()
            if self._phys_avg is not None:
                self._phys_avg.zero_()
            if self._particles is not None:
                self._particles.clear()
        self._avg_on = False
        self._host_hist.clear()
        self._rows = self._flushed = 0
        self.steps = 0
        self._perf.clear()

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        """Everything needed to resume: state tensors (CPU), counters, history."""
        s = self._state

        def snap(x: torch.Tensor) -> torch.Tensor:      # .cpu() aliases on CPU: always copy
            return x.detach().to("cpu", copy=True)
        d = {"version": 2, "u": snap(s.u), "v": snap(s.v), "p": snap(s.p), "t": snap(s.t), "dt": snap(s.dt),
             "steps": self.steps, "re": self.cfg.re, "history": self.history(),
             "body_coefficients": snap(self._body_c)}
        for name in ("dye", "theta", "dye_extra", "ctrl"):
            if getattr(s, name) is not None:
                d[name] = snap(getattr(s, name))
        ph = self.ops.phys
        if ph is not None and ph.paint:
            d["paint"] = snap(ph.paint_c)
            d["paint_u"] = snap(ph.paint_u)
            d["paint_v"] = snap(ph.paint_v)
        return d

    def load_state_dict(self, d: dict) -> None:
        """Restore a :meth:`state_dict` taken from a solver with the same grid."""
        s = self._state
        if tuple(d["u"].shape) != tuple(s.u.shape) or tuple(d["v"].shape) != tuple(s.v.shape):
            raise ValueError("state_dict grid does not match this solver")
        with torch.no_grad():
            for name in ("u", "v", "p", "t", "dt"):
                getattr(s, name).copy_(d[name].to(self.device, self.dtype))
            for name in ("dye", "theta", "dye_extra", "ctrl"):
                dst = getattr(s, name)
                if dst is not None and name in d and tuple(d[name].shape) == tuple(dst.shape):
                    dst.copy_(d[name].to(self.device, self.dtype))
            if "body_coefficients" in d and d["body_coefficients"].shape == self._body_c.shape:
                self._body_c.copy_(d["body_coefficients"].to(self.device, self.dtype))
        if "paint" in d:
            ph = self._paint_layer()
            with torch.no_grad():
                ph.paint_c.copy_(d["paint"].to(self.device, self.dtype))
                if "paint_u" in d and "paint_v" in d:
                    ph.paint_u.copy_(d["paint_u"].to(self.device, self.dtype))
                    ph.paint_v.copy_(d["paint_v"].to(self.device, self.dtype))
                else:
                    phx.paint_faces(ph, self.ops)
                phx.combine_penalisation(ph)
            self._rebuild_solid()
        if "re" in d:
            self.set_inflow(re=float(d["re"]))
        self.steps = int(d.get("steps", 0))
        hist = d.get("history")
        self._host_hist.clear()
        self._rows = self._flushed = 0
        if hist is not None and len(hist.get("t", [])):
            self._host_hist.append(np.c_[hist["t"], hist["cd"], hist["cl"], hist["cm"]].astype(np.float64))
        with torch.no_grad():
            self._hist_n.zero_()
        for r in (self._aux, self._diag):
            if r is not None:
                r.clear()


__all__ = [
    "FIELDS",
    "PHYSICS_FIELDS",
    "FlowSolver",
    "advect_dye",
    "bilinear",
    "centre_u",
    "centre_v",
    "face_coordinates",
    "resample_closed",
    "vorticity",
]
