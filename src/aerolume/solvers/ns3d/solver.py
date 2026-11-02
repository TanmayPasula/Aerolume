"""Stateful 3D flow solver for the application: fast path, live edits, outputs.

:class:`FlowSolver3D` wraps the functional core (:mod:`.core`) with everything a
GUI or a batch runner needs, mirroring the validated 2D ``FlowSolver``:

* **Fused kernels + static buffers + CUDA graphs.** On CUDA a step runs the
  fused jiterator kernels of :mod:`.fused` (same arithmetic as the reference
  core, ~3.5x less memory traffic); the state lives in fixed tensors and blocks
  of 10 steps are captured once in a ``torch.cuda.CUDAGraph`` and replayed.
  Nothing inside a step synchronises with the host (dt comes from a device-side
  CFL reduction). Any failure to compile, capture or replay falls back to the
  reference core / eager execution with a warning.
* **Device-side history.** [t, CD, CS, CL, C_roll, Cm, Cn] goes into a ring
  buffer on the device every step and is copied to the host once per
  :meth:`advance`.
* **Running means** in float32 with a weighted Welford update (mean and the sum
  of squared deviations), which keeps the variance accurate where a plain sum
  of squares would cancel catastrophically, at half the memory of float64
  accumulators (the 2D solver's choice) - it matters at 12 M cells. The flow is
  sampled once every ``AVERAGE_EVERY`` steps with the weight of the elapsed
  time (rectangle rule over ~0.1 L/U, far below any flow time scale), which
  cuts the averaging cost tenfold; reading the means folds in the time since
  the last sample, so they always cover exactly the averaged interval.
* **Live edits.** :meth:`set_inflow` and :meth:`set_bodies` rebuild in place
  whenever shapes allow, so captured graphs stay valid (a Reynolds change on
  the fused path re-captures: the kernels take the viscosity as a launch
  argument).
* **Outputs.** Cell-centred fields, slices, Q-criterion and vorticity, surface
  cp / cf / wall-shear vectors per mesh vertex.
* **Smoke.** With ``cfg.dye`` a passive scalar released by the configured
  :class:`~aerolume.solvers.ns3d.config.DyeRake` is advected after every step
  (:mod:`.dye`; fused kernels and inside the CUDA graphs on the fast path), so
  the 3D view can show smoke streams around the body (DESIGN.md 7.13).

Placed physics (DESIGN.md section 10, ``docs/theory_ns3d.md`` 15-18) is built
only when the configuration asks for it (:attr:`SolverConfig3D.has_placed_physics`):

* heat transfer (temperature field, buoyancy, heated bodies, wall conditions,
  heat sources), fans, jets and vents through :class:`~.physics.Physics3D`;
* up to four smoke channels and GPU tracer particles from
  :class:`~.config.Emitter3D` emitters (:mod:`.emitters`);
* live edits without reallocating the grid: :meth:`set_emitters`,
  :meth:`set_sources`, :meth:`set_thermal`, and mouse-painted obstacles
  (:meth:`paint_solid`);
* diagnostics recorded on the device every 10-step block
  (:meth:`diagnostics`), a one-step solver trace (:meth:`trace`) and the PID
  time-step controller.
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
from .config import WALL_SIDES, DyeRake, Emitter3D, ImmersedMesh, SolverConfig3D
from .core import (
    COEFF_NAMES,
    State3D,
    StepOut3D,
    cfl_dt3d,
    divergence3d,
    init_state3d,
    make_operators3d,
    set_reynolds,
    step3d_full,
    step3d_phys,
)
from .dye import advect_dye3d, apply_solids_and_sources, build_dye_sources, default_dye_rake
from .fields import (
    GRADIENT_FIELDS,
    centre_u,
    centre_v,
    centre_w,
    gradient_field,
    probe_distances,
    trilinear,
    vertex_normals,
)
from .geometry import Geometry3D, body_window3d, build_geometry3d, window_contains

FIELDS3D = ("u", "v", "w", "speed", "p", "cp", "vorticity", "q", "mask", "nut", "divergence", "dye",
            "dye1", "dye2", "dye3", "temperature")
MEAN_FIELDS3D = ("u", "v", "w", "speed", "p", "cp", "vorticity", "q", "u_rms", "v_rms", "w_rms", "p_rms",
                 "cp_rms", "mask")
#: extra running means of a heat-transfer run
MEAN_FIELDS3D_THERMAL = ("temperature", "temperature_rms")
HISTORY_KEYS = ("t",) + COEFF_NAMES
HISTORY_CAPACITY = 4096         # device ring rows; flushed to the host before it can wrap
GRAPH_STEPS = 10                # steps captured per CUDA graph
AVERAGE_EVERY = GRAPH_STEPS     # running means take one sample per this many steps (time-weighted)
#: diagnostics recorded on the last step of every GRAPH_STEPS block (``cfg.diagnostics``): time,
#: step, CFL number of the step taken, sum of the velocity maxima, continuity residual (RMS and
#: max, units U/L), relative Poisson residual of the last projection, and per field u, v, w, p,
#: theta the change over that step (discrete L2 norm, RMS, max, relative L2) and the heat stored
_CHANGE_FIELDS = ("u", "v", "w", "p", "theta")
DIAG_KEYS = (("t", "dt", "cfl", "umax", "div_rms", "div_max", "poisson_residual")
             + tuple(f"d{f}_{n}" for f in _CHANGE_FIELDS for n in ("l2", "rms", "max", "rel"))
             + ("heat_storage",))
DIAG_CAPACITY = 1024
_AXES = {"x": 2, "y": 1, "z": 0}
_DYE_NAMES = ("dye", "dye1", "dye2", "dye3")
#: peak device bytes per cell of FlowSolver3D.advance in float32, measured on CUDA with the fused
#: kernels and graphs (bench_ns3d.py perf: 256x128x128 and 400x176x172, running means on), plus a margin
BYTES_PER_CELL_F32 = {"none": 210, "les": 260}
#: extra bytes per cell with heat transfer (theta, its stage copies and flux temporaries; bench_ns3d.py
#: thermal), per further smoke channel, and per tracer particle
BYTES_PER_CELL_THERMAL_F32 = 48
BYTES_PER_CELL_DYE_CHANNEL_F32 = 8
BYTES_PER_PARTICLE_F32 = 24


def estimate_memory_bytes(nx: int, ny: int, nz: int, *, dtype: str = "float32", turbulence: str = "none",
                          mask_window_cells: int = 0, thermal: bool = False, dye_channels: int = 1,
                          particles: int = 0) -> int:
    """Device memory a run of this grid needs (state, step temporaries, CUDA-graph pool,
    running means), for planners that must refuse grids that do not fit.

    Measured peaks: 134 B/cell laminar without means, ~196-200 B/cell with
    running means (4.2 M and 12.1 M cells), 179 B/cell with WALE (no means), all
    float32; float64 doubles everything. The mask window adds ~30 floats per
    window cell (masks, solid velocities, force weights, penalisation
    temporaries). Heat transfer, extra smoke channels and particles add the
    ``BYTES_PER_*`` figures.
    """
    per = BYTES_PER_CELL_F32["none" if turbulence in ("none", "laminar") else "les"]
    if thermal:
        per += BYTES_PER_CELL_THERMAL_F32
    per += BYTES_PER_CELL_DYE_CHANNEL_F32 * max(int(dye_channels) - 1, 0)
    scale = 2 if dtype == "float64" else 1
    return int(scale * (per * nx * ny * nz + 30 * 4 * mask_window_cells + BYTES_PER_PARTICLE_F32 * int(particles))
               + 64 * 2**20)


class FlowSolver3D:
    """Incompressible 3D Navier-Stokes around immersed meshes (non-dimensional).

    Typical use::

        solver = FlowSolver3D(cfg)
        solver.advance(500)                       # fast path, no autograd
        cd, cs, cl, c_roll, cm, cn = solver.coefficients()
        q = solver.field("q")                     # (nz, ny, nx) tensor on the solver device
    """

    def __init__(self, cfg: SolverConfig3D):
        self.cfg = dataclasses.replace(cfg, bodies=list(cfg.bodies), heat_sources=list(cfg.heat_sources),
                                       fans=list(cfg.fans), ports=list(cfg.ports),
                                       emitters=None if cfg.emitters is None else list(cfg.emitters))
        self.ops = make_operators3d(self.cfg)
        self.device = self.ops.device
        self.dtype = self.ops.dtype
        self._bodies: list[ImmersedMesh] = list(self.cfg.bodies)
        self._kern = None
        self.fused_error: str | None = None
        if self.cfg.fused_kernels and self.device.type == "cuda":
            try:
                from .fused import load_kernels
                self._kern = load_kernels(self.cfg.advection, self.device, self.dtype)
            except Exception as exc:  # noqa: BLE001 - any failure: run the reference core
                self.fused_error = str(exc)
        with torch.no_grad():
            self._geom = build_geometry3d(self._bodies, self.ops, window=self._live_window(self._bodies))
            self._state = init_state3d(self.cfg, self.ops)
        self._body_c = torch.zeros(len(self._bodies), 6, device=self.device, dtype=self.dtype)
        self._hist = torch.zeros(HISTORY_CAPACITY, 7, device=self.device, dtype=self.dtype)
        self._hist_n = torch.zeros((), device=self.device, dtype=torch.long)
        self._host_hist: list[np.ndarray] = []
        self._rows = 0
        self._flushed = 0
        self.steps = 0
        self._avg: dict[str, torch.Tensor] | None = None
        self._avg_on = False
        self._avg_t0 = 0.0                  # solver time of start_averaging (mean physics reports)
        self._avg_stop: float | None = None  # solver time of stop_averaging, None while averaging
        self._use_graphs = bool(self.cfg.use_cuda_graphs) and self.device.type == "cuda"
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}   # (steps, averaging, sample, diagnostics)
        self._pool = None
        self.graph_error: str | None = None
        self.notes: list[str] = []
        # placed physics (DESIGN.md 10): none unless asked for, so the default step is the v0.2 one
        self._phys = None
        self._theta: torch.Tensor | None = None
        self._theta_fused = True
        self.theta_error: str | None = None
        if self.cfg.has_placed_physics:
            self._build_physics()
        # passive smoke (DESIGN.md 7.13 / 10.1): channel buffers, advected after every step
        self._dye: torch.Tensor | None = None
        self._dye_extra: list[torch.Tensor] = []
        self._dye_src = None                       # the v0.2 rake source (legacy mode)
        self._smoke_src: list = []                 # Emitter3D smoke sources (emitter mode)
        self._dye_rake: DyeRake | None = None
        self._dye_fused = False
        self.dye_error: str | None = None
        self._particles = None
        if self.cfg.dye:
            self._dye = torch.zeros(self.ops.nz, self.ops.ny, self.ops.nx, device=self.device, dtype=self.dtype)
            if self.cfg.emitters is None:
                self._dye_rake = self.cfg.dye_rake if self.cfg.dye_rake is not None else (
                    None if self.cfg.closed else self.default_dye_rake())
                self._dye_src = build_dye_sources(self._dye_rake, self.ops)
            else:
                self._dye_rake = self.cfg.dye_rake
                self._rebuild_smoke(self.cfg.emitters)
            if self._kern is not None:
                try:
                    from .fused import check_dye_kernels
                    check_dye_kernels(self._kern, self.device, self.dtype)
                    self._dye_fused = True
                except Exception as exc:  # noqa: BLE001 - smoke falls back to the reference scheme
                    self.dye_error = str(exc)
        if self.cfg.emitters and any(e.species == "particles" for e in self.cfg.emitters):
            self._particles = self._make_particles(self.cfg.emitters)
        # diagnostics (device ring, one row per 10-step block) and the placed-physics history
        self._diag = None
        self._diag_n = torch.zeros((), device=self.device, dtype=torch.long)
        self._diag_host: list[np.ndarray] = []
        self._diag_steps: list[int] = []
        self._diag_rows = 0
        self._diag_flushed = 0
        if self.cfg.diagnostics:
            self._diag = torch.zeros(DIAG_CAPACITY, len(DIAG_KEYS), device=self.device, dtype=self.dtype)
        self._phist = None
        self._phist_host: list[np.ndarray] = []
        self._phist_cols = 0
        self._phist_walls = False
        self._setup_physics_history()
        self._perf: list[tuple[int, int, float]] = []

    # ------------------------------------------------------------ physics setup
    def _build_physics(self) -> None:
        """Create the placed-physics operators (and the temperature field) for the current
        configuration, keeping the flow state."""
        from .physics import Physics3D
        with torch.no_grad():
            phys = Physics3D(self.cfg, self.ops, self._geom, self._bodies)
            if self._phys is not None and self._phys.paint is not None:
                phys.paint = self._phys.paint
        self._phys = phys
        self.notes = list(dict.fromkeys(self.notes + phys.notes))
        if phys.thermal is not None and self._theta is None:
            self._theta = torch.full((self.ops.nz, self.ops.ny, self.ops.nx), phys.thermal.initial,
                                     device=self.device, dtype=self.dtype)
            if self._kern is not None:
                try:
                    from .fused import check_theta_kernels
                    check_theta_kernels(self._kern, self.device, self.dtype)
                    self._theta_fused = True
                except Exception as exc:  # noqa: BLE001 - theta falls back to the reference arithmetic
                    self._theta_fused = False
                    self.theta_error = str(exc)
        elif phys.thermal is None:
            self._theta = None
        self._invalidate_graphs()

    def _ensure_physics(self) -> None:
        if self._phys is None:
            self._build_physics()

    def _setup_physics_history(self) -> None:
        """A device ring of [t, heat rate per body, (wall temperature per body, when a body
        has a set heat flux), thrust and disc velocity per fan], aligned with the force
        history (the runner's ``nu`` / ``heat_rate`` / ``fan<i>_*`` series)."""
        nb = len(self._bodies) if (self._phys is not None and self._phys.thermal is not None) else 0
        nf = len(self._phys.fans) if (self._phys is not None and self._phys.fans is not None) else 0
        self._phist_walls = bool(nb) and self._phys.thermal.geom.wall_idx is not None
        cols = 1 + nb * (2 if self._phist_walls else 1) + 2 * nf if (nb or nf) else 0
        if cols != self._phist_cols:
            self._flush()
            self._phist_host.clear()
            self._phist = torch.zeros(HISTORY_CAPACITY, cols, device=self.device, dtype=self.dtype) if cols else None
            self._phist_cols = cols

    def _make_particles(self, emitters):
        from .emitters import ParticleSystem3D
        dt0 = self.cfg.dt if self.cfg.dt is not None else min(self.cfg.cfl * self.ops.h,
                                                              self.ops.h ** 2 * self.cfg.re / 6.0)
        return ParticleSystem3D(emitters, self.ops, self.cfg.extent, capacity=self.cfg.particle_capacity,
                                dt_estimate=dt0)

    def _live_window(self, bodies: list[ImmersedMesh]) -> tuple[int, int, int, int, int, int]:
        """Mask window: the bodies plus a margin, widened to cover the configured live
        pitch / yaw range about the moment reference (keeps graphs across attitude edits)."""
        if not bodies:
            return (0, 0, 0, 0, 0, 0)
        return body_window3d([b.vertices_numpy() for b in bodies], self.ops,
                             pitch_deg=self.cfg.window_pitch_deg, yaw_deg=self.cfg.window_yaw_deg)

    # ----------------------------------------------------------------- props
    @property
    def t(self) -> float:
        """Simulated time in convective units (syncs with the device)."""
        return float(self._state.t)

    @property
    def dt(self) -> float:
        """The most recent time step (syncs); before the first step, the CFL step."""
        d = float(self._state.dt)
        if d > 0:
            return d
        if self.cfg.dt is not None:
            return float(self.cfg.dt)
        s = self._state
        return float(cfl_dt3d(s.u, s.v, s.w, self.ops))

    @property
    def state(self) -> State3D:
        return self._state

    @property
    def geometry(self) -> Geometry3D:
        return self._geom

    @property
    def bodies(self) -> list[ImmersedMesh]:
        return list(self._bodies)

    @property
    def uses_cuda_graphs(self) -> bool:
        return self._use_graphs

    @property
    def uses_fused_kernels(self) -> bool:
        return self._kern is not None

    @property
    def averaging_time(self) -> float:
        """Convective time covered by the running means (0 when not averaging)."""
        if self._avg is None:
            return 0.0
        return float(self._avg["time"] + self._avg["pending"])

    @property
    def cells(self) -> int:
        return self.ops.nx * self.ops.ny * self.ops.nz

    @property
    def physics(self):
        """The placed-physics operators (:class:`~.physics.Physics3D`), or None."""
        return self._phys

    @property
    def thermal_enabled(self) -> bool:
        return self._theta is not None

    @property
    def theta(self) -> torch.Tensor | None:
        """The temperature field theta (nz, ny, nx) on the device (the buffer itself), or None."""
        return self._theta

    # ------------------------------------------------------------------- smoke
    @property
    def dye_enabled(self) -> bool:
        return self._dye is not None

    @property
    def dye_channels(self) -> int:
        """Number of smoke channels held (0 without smoke)."""
        return 0 if self._dye is None else 1 + len(self._dye_extra)

    @property
    def dye_rake(self) -> DyeRake | None:
        """The rake in use (solver units): the configured rake, or the automatic one."""
        return self._dye_rake

    def _channels(self) -> list[torch.Tensor]:
        return [] if self._dye is None else [self._dye] + self._dye_extra

    def default_dye_rake(self, kind: str = "rake") -> DyeRake:
        """The automatic emitters for the current bodies (:func:`.dye.default_dye_rake`)."""
        ops = self.ops
        if self._bodies:
            v = np.vstack([b.vertices_numpy() for b in self._bodies])
            lo, hi = v.min(axis=0), v.max(axis=0)
            bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        else:                                   # no body: a rake across the middle of the inlet third
            ex = self.cfg.extent
            cx, cy, cz = ex[0] + 0.3 * (ex[1] - ex[0]), 0.5 * (ex[2] + ex[3]), 0.5 * (ex[4] + ex[5])
            q = 0.15 * min(ex[3] - ex[2], ex[5] - ex[4])
            bounds = (cx, cx + q, cy - q, cy + q, cz - q, cz + q)
        return default_dye_rake(bounds, self.cfg.extent, ops.h, kind=kind, symmetry=ops.symmetry,
                                ground=self.cfg.bc_zmin in ("noslip", "moving"))

    def set_dye_rake(self, rake: DyeRake | dict | None) -> None:
        """Move or replace the smoke rake on channel 0 (None = the automatic rake, or no rake
        when the configuration uses :class:`Emitter3D` emitters); the smoke already released
        keeps flowing. Captured graphs stay valid when the new emitters cover the same cells
        as the old ones (a pulse change, a nudge within the window); otherwise they are
        re-captured on the next advance."""
        if self._dye is None:
            return
        rake = DyeRake.coerce(rake)
        if self.cfg.emitters is not None:              # emitter mode: the rake is one more source
            self._dye_rake = rake
            self.cfg.dye_rake = rake
            self._rebuild_smoke(self.cfg.emitters)
            return
        rake = rake if rake is not None else self.default_dye_rake()
        new = build_dye_sources(rake, self.ops)
        old = self._dye_src
        with torch.no_grad():
            if old is not None and old.window == new.window and old.weight.shape == new.weight.shape:
                old.weight.copy_(new.weight)
                old.pulse.copy_(new.pulse)
            else:
                self._dye_src = new
                self._invalidate_graphs()
        self._dye_rake = rake
        self.cfg.dye_rake = rake

    # ---------------------------------------------------------------- emitters
    def _channel_count(self, emitters) -> int:
        n = 1
        for e in emitters or ():
            if e.species == "dye":
                n = max(n, e.channel + 1)
        for p in self.cfg.ports:
            if p.dye_channel is not None:
                n = max(n, int(p.dye_channel) + 1)
        return n

    def _rebuild_smoke(self, emitters) -> None:
        """Smoke sources of the emitters (+ the channel-0 rake if one is set); in place when
        the new sources cover the same cells as the old ones (graphs kept)."""
        from .emitters import build_smoke_sources
        with torch.no_grad():
            srcs = []
            if self._dye_rake is not None:
                srcs += build_smoke_sources(None, self._dye_rake, self.ops, self.cfg.extent)
            srcs += build_smoke_sources(list(emitters or ()), None, self.ops, self.cfg.extent)
            need = self._channel_count(emitters)
            grow = need > self.dye_channels
            while self.dye_channels < need:
                self._dye_extra.append(torch.zeros_like(self._dye))
            old = self._smoke_src
            same = (not grow and len(old) == len(srcs)
                    and all(a.window == b.window and a.weight.shape == b.weight.shape and a.channel == b.channel
                            for a, b in zip(old, srcs)))
            if same:
                for a, b in zip(old, srcs):
                    a.weight.copy_(b.weight)
                    a.params.copy_(b.params)
            else:
                self._smoke_src = srcs
                self._invalidate_graphs()

    def set_emitters(self, emitters: list[Emitter3D] | list[dict] | None) -> None:
        """Replace the smoke and particle emitters live (solver units); the grid, the flow and
        the smoke already released are kept. ``None`` returns to the v0.2 automatic rake.
        Captured graphs stay valid when the smoke sources keep their cells and the particle
        emitters are unchanged in number and kind; otherwise they are re-captured lazily."""
        if emitters is not None:
            emitters = [e if isinstance(e, Emitter3D) else Emitter3D(**e) for e in emitters]
        old_parts = [e for e in (self.cfg.emitters or ()) if e.species == "particles"]
        self.cfg.emitters = None if emitters is None else list(emitters)
        if self._dye is not None:
            if emitters is None:
                self._smoke_src = []
                self.set_dye_rake(self.cfg.dye_rake)
            else:
                self._dye_src = None
                self._rebuild_smoke(emitters)
        parts = [e for e in (emitters or ()) if e.species == "particles"]
        if parts != old_parts:
            old = self._particles
            self._particles = self._make_particles(parts) if parts else None
            if old is not None and self._particles is not None and old.capacity == self._particles.capacity:
                with torch.no_grad():
                    for a, b in zip(self._particles.buffers()[:4], old.buffers()[:4]):   # pos, alive, birth, nozzle
                        a.copy_(b)
            self._invalidate_graphs()

    # ------------------------------------------------------------ live sources
    def set_sources(self, *, heat_sources=None, fans=None, ports=None) -> None:
        """Replace heat sources, fans and / or ports live (each None = unchanged; solver
        units). The grid and the flow are kept; the fans' controller state is kept for
        fans that only changed their setting. Graphs are re-captured lazily when the
        source structure changed."""
        from .config import Fan3D, HeatSource3D, Port3D
        from .config import _coerce as coerce
        if heat_sources is not None:
            self.cfg.heat_sources = [coerce(HeatSource3D, s) for s in heat_sources]
            if self.cfg.heat_sources and self.cfg.thermal is None:
                raise ValueError("heat sources need heat transfer (set_thermal / cfg.thermal)")
        if fans is not None:
            self.cfg.fans = [coerce(Fan3D, f) for f in fans]
        if ports is not None:
            self.cfg.ports = [coerce(Port3D, p) for p in ports]
        old = self._phys
        if not self.cfg.has_placed_physics and (old is None or old.paint is None):
            self._phys = None
            self._invalidate_graphs()
            self._setup_physics_history()
            return
        keep = None
        if old is not None and old.fans is not None:
            keep = (old.fans.thrust.clone(), old.fans.e_prev.clone(), [f.mode for f in old.fans.fans])
        self._build_physics()
        new = self._phys
        if keep is not None and new.fans is not None and [f.mode for f in new.fans.fans] == keep[2]:
            with torch.no_grad():
                vel = new.fans.velocity_mode
                new.fans.thrust.copy_(torch.where(vel, keep[0], new.fans.thrust))
                new.fans.e_prev.copy_(keep[1])
        if ports is not None and self._dye is not None and self._channel_count(self.cfg.emitters) > self.dye_channels:
            self._rebuild_smoke(self.cfg.emitters)
        self._setup_physics_history()

    def set_thermal(self, *, walls=None, theta_inlet: float | None = None, richardson: float | None = None,
                    prandtl: float | None = None) -> None:
        """Change the heat-transfer settings live: wall conditions (same form as
        ``ThermalConfig3D.walls``), the inlet temperature, the Richardson or Prandtl number.
        Wall values and the inlet temperature are device values (graphs kept when the wall
        modes are unchanged); everything else re-captures the graphs lazily."""
        if self.cfg.thermal is None:
            raise ValueError("heat transfer is off (cfg.thermal is None)")
        tc = self.cfg.thermal
        changes = {}
        if walls is not None:
            changes["walls"] = walls
        if theta_inlet is not None:
            changes["theta_inlet"] = float(theta_inlet)
        if richardson is not None:
            changes["richardson"] = float(richardson)
        if prandtl is not None:
            changes["prandtl"] = float(prandtl)
        new_tc = dataclasses.replace(tc, **changes)
        th = self._phys.thermal
        old_kinds = th.side_kinds
        self.cfg.thermal = new_tc
        if richardson is None and prandtl is None:
            kinds, vals = _side_rules(self.cfg, new_tc)
            if tuple(kinds) == old_kinds:
                with torch.no_grad():
                    th.side_vals.copy_(torch.tensor(vals, device=self.device, dtype=self.dtype))
                th.cfg = new_tc
                return
        theta = self._theta
        self._build_physics()
        self._theta = theta

    # ----------------------------------------------------------- paint (explore)
    def paint_solid(self, center, radius: float = 0.0, *, erase: bool = False, shape: str = "sphere",
                    size=None) -> bool:
        """Add (``erase=False``) or remove solid in a brush footprint (solver units): a ball of
        ``radius`` or a box of ``size`` around ``center``, with the mask's smooth edge. Painted
        solids are at rest, block smoke and particles and are kept across steps; the first
        stroke adds the paint layer (graphs re-captured once), later strokes edit it in place.
        Returns False when the brush misses the domain. The differentiable core never sees
        paint.

        Paint uses the bodies' calibrated interface, so a stroke's effective wall sits on the
        brush outline; erasing (as in the 2D solver) cuts a hole whose wall sits on the
        eraser's outline: ``chi <- min(chi, 1 - chi_eraser)``. A stroke's smooth edge reaches
        ~0.7 cells beyond its radius and the eraser clears completely only ~1.3 cells inside
        its own, so erasing a dab with the same radius leaves a sub-cell film (solid
        fraction <= ~0.15); an eraser two cells larger removes it entirely."""
        from .physics import PaintLayer
        self._ensure_physics()
        if self._phys.paint is None:
            self._phys.paint = PaintLayer(self.ops)
            self._invalidate_graphs()
        with torch.no_grad():
            return self._phys.paint.stroke(self.ops, center, radius, erase=erase, shape=shape, size=size)

    def clear_paint(self) -> None:
        """Remove every painted solid (the layer stays allocated, graphs stay valid)."""
        if self._phys is not None and self._phys.paint is not None:
            with torch.no_grad():
                self._phys.paint.clear()

    @property
    def painted(self) -> bool:
        return self._phys is not None and self._phys.paint is not None

    # ---------------------------------------------------------------- stepping
    def _step_fn(self, state: State3D, diag: bool = False) -> StepOut3D:
        if self._phys is None:
            if self._kern is not None:
                from .fused import fast_step
                new, bc = fast_step(state, self._geom, self.ops, self._kern)
            else:
                new, bc = step3d_full(state, self._geom, self.ops)
            return StepOut3D(new, bc, None)
        if self._kern is not None:
            from .fused import fast_step_phys
            return fast_step_phys(state, self._geom, self.ops, self._kern, phys=self._phys, theta=self._theta,
                                  diag=diag, theta_kernels=self._theta_fused)
        return step3d_phys(state, self._geom, self.ops, phys=self._phys, theta=self._theta, diag=diag)

    def _buffers(self) -> list[torch.Tensor]:
        s = self._state
        bufs = [s.u, s.v, s.w, s.p, s.t, s.dt, self._body_c, self._hist, self._hist_n]
        bufs.extend(self._channels())
        if self._avg is not None:
            bufs.extend(self._avg.values())
        if self._theta is not None:
            bufs.append(self._theta)
        if self._phys is not None:
            bufs.extend(self._phys.buffers())
        if self._particles is not None:
            bufs.extend(self._particles.buffers())
        if self._diag is not None:
            bufs.extend([self._diag, self._diag_n])
        if self._phist is not None:
            bufs.append(self._phist)
        return bufs

    def _step_static(self, sample: bool, diag: bool = False) -> None:
        """One step written into the static buffers (graph-capturable)."""
        st = self._state
        out = self._step_fn(st, diag)
        if diag and self._diag is not None:
            self._diag_record(st, self._theta, out)
        for dst, src in zip(st, out.state):
            dst.copy_(src)
        if self._theta is not None and out.theta is not None:
            self._theta.copy_(out.theta)
        self._advance_tracers(out.state, static=True)
        self._record(out.state, out.coeffs, sample)

    def _step_eager(self, sample: bool, diag: bool = False) -> None:
        out = self._step_fn(self._state, diag)
        if diag and self._diag is not None:
            self._diag_record(self._state, self._theta, out)
        self._state = out.state
        if self._theta is not None and out.theta is not None:
            self._theta = out.theta
        self._advance_tracers(out.state, static=False)
        self._record(out.state, out.coeffs, sample)

    def _advance_dye(self, new: State3D, static: bool) -> None:
        """Back-compatible name of :meth:`_advance_tracers`."""
        self._advance_tracers(new, static)

    def _solid_cleanup(self, c: torch.Tensor) -> None:
        if self._phys is not None and self._phys.paint is not None:
            c.masked_fill_(self._phys.paint.chi_c > 0.5, 0.0)

    def _advance_tracers(self, new: State3D, static: bool) -> None:
        """Advect the smoke with the step's final (divergence-free) velocity, clear solids,
        emit from the nozzles; then move the tracer particles. ``static``: write into the
        fixed buffers (graph capture)."""
        ports = self._phys.ports if self._phys is not None else None
        if self._dye is not None:
            chans = self._channels()
            outs = []
            for ch, c0 in enumerate(chans):
                if self._dye_fused:
                    from .fused import fast_dye
                    c = fast_dye(c0, new.u, new.v, new.w, new.dt, self.ops, self._kern)
                else:
                    c = advect_dye3d(c0, new.u, new.v, new.w, new.dt, self.ops)
                if ports is not None:
                    ports.scalar_sources(c, c0, new.dt, ch)
                    c.clamp_(0.0, 1.0)
                outs.append(c)
            if self._dye_src is not None:                       # v0.2 rake on channel 0
                apply_solids_and_sources(outs[0], new.t, self._geom, self._dye_src)
                for c in outs[1:]:
                    apply_solids_and_sources(c, new.t, self._geom, None)
            else:
                from .emitters import apply_smoke_sources
                for c in outs:
                    apply_solids_and_sources(c, new.t, self._geom, None)
                apply_smoke_sources(outs, new.t, self._smoke_src)
            for c in outs:
                self._solid_cleanup(c)
            if static:
                for dst, c in zip(chans, outs):
                    dst.copy_(c)
            else:
                self._dye = outs[0]
                self._dye_extra = outs[1:]
        if self._particles is not None:
            # eager CPU steps advect only the live slots (a host sync, free there; emitters.py)
            self._particles.step(new.u, new.v, new.w, new.t, new.dt, solid=self._solid_at,
                                 compact=not static and new.u.device.type == "cpu")

    def _solid_at(self, p: torch.Tensor) -> torch.Tensor:
        """True where positions (P, 3) lie in a solid (body or paint; solid fraction > 1/2)."""
        ops = self.ops
        idx = []
        for a, (o, n) in enumerate(((ops.x0, ops.nx), (ops.y0, ops.ny), (ops.z0, ops.nz))):
            idx.append(((p[:, a] - o) / ops.h).floor().long().clamp(0, n - 1))
        i, j, k = idx
        out = torch.zeros(p.shape[0], device=p.device, dtype=torch.bool)
        g = self._geom
        if not g.empty:
            k0, k1, j0, j1, i0, i1 = g.window
            inside = (i >= i0) & (i < i1) & (j >= j0) & (j < j1) & (k >= k0) & (k < k1)
            ci = (i - i0).clamp(0, i1 - i0 - 1)
            cj = (j - j0).clamp(0, j1 - j0 - 1)
            ck = (k - k0).clamp(0, k1 - k0 - 1)
            out = out | (inside & (g.chi_c[ck, cj, ci] > 0.5))
        if self._phys is not None and self._phys.paint is not None:
            out = out | (self._phys.paint.chi_c[k, j, i] > 0.5)
        return out

    def _record(self, new: State3D, bc: torch.Tensor, sample: bool) -> None:
        if bc.shape[0]:
            self._body_c.copy_(bc)
        row = torch.cat([new.t.reshape(1), bc.sum(0)]).reshape(1, 7)
        slot = torch.remainder(self._hist_n, HISTORY_CAPACITY).reshape(1)
        self._hist.index_copy_(0, slot, row)
        if self._phist is not None:
            parts = [new.t.reshape(1)]
            ph = self._phys
            if ph.thermal is not None:
                parts.append(ph.thermal.heat_rate)
                if self._phist_walls:
                    parts.append(ph.thermal.geom.wall_theta(self._theta))
            if ph.fans is not None:
                parts += [ph.fans.thrust, ph.fans.ud]
            self._phist.index_copy_(0, slot, torch.cat(parts).reshape(1, -1).to(self.dtype))
        self._hist_n.add_(1)
        if self._avg_on:
            a = self._avg
            a["pending"].add_(new.dt.to(torch.float64))
            if sample:
                fields = [new.u, new.v, new.w, new.p]
                if "t" in a and self._theta is not None:
                    fields.append(self._theta)
                self._welford(a, fields)

    def _welford(self, a: dict, fields) -> None:
        """Fold the current fields into the running means with the weight of the time
        elapsed since the last sample (weighted Welford update, in place)."""
        w = a["pending"]
        a["time"].add_(w)
        r = (w / a["time"]).to(self.dtype)                    # w / T_new
        wf = w.to(self.dtype)
        for name, x in zip("uvwpt", fields):
            mean, m2 = a[name], a[name + "2"]
            d = x - mean
            mean.addcmul_(d, r)
            m2.addcmul_(d * (x - mean), wf)
        w.zero_()

    # ------------------------------------------------------------- diagnostics
    def _diag_record(self, old: State3D, theta_old, out: StepOut3D) -> None:
        """One diagnostics row (device only; see DIAG_KEYS) for the step old -> out."""
        new = out.state
        h = self.ops.h
        dV = h ** 3
        div = divergence3d(new.u, new.v, new.w, h)
        if self._phys is not None and self._phys.ports is not None:
            div = self._phys.continuity(div)
        d2 = (div * div).sum()
        n = div.numel()
        umax_old = old.u.abs().amax() + old.v.abs().amax() + old.w.abs().amax()
        umax = new.u.abs().amax() + new.v.abs().amax() + new.w.abs().amax()
        tiny = torch.finfo(self.dtype).tiny
        if self._phys is not None:
            res = torch.sqrt(d2 / torch.clamp(self._phys.div_pre, min=tiny))
        else:
            res = torch.full_like(d2, float("nan"))
        vals = [new.t, new.dt, new.dt * umax_old / h, umax, torch.sqrt(d2 / n), div.abs().amax(), res]
        pairs = [(new.u, old.u), (new.v, old.v), (new.w, old.w), (new.p, old.p)]
        if self._theta is not None and out.theta is not None:
            pairs.append((out.theta, theta_old))
        for a, b in pairs:
            d = a - b
            s2 = (d * d).sum()
            vals += [torch.sqrt(s2 * dV), torch.sqrt(s2 / d.numel()), d.abs().amax(),
                     torch.sqrt(s2 / torch.clamp((a * a).sum(), min=tiny))]
        if len(pairs) < 5:
            vals += [torch.full_like(d2, float("nan"))] * 4
        if self._theta is not None and out.theta is not None:
            th = self._phys.thermal
            sym = 2.0 if self.ops.symmetry else 1.0
            vals.append((out.theta - theta_old).sum() * (dV * th.pe * sym) / new.dt)
        else:
            vals.append(torch.full_like(d2, float("nan")))
        row = torch.stack([v.reshape(()).to(self.dtype) for v in vals]).reshape(1, -1)
        self._diag.index_copy_(0, torch.remainder(self._diag_n, DIAG_CAPACITY).reshape(1), row)
        self._diag_n.add_(1)

    def diagnostics(self) -> dict[str, np.ndarray]:
        """The diagnostics history (``cfg.diagnostics``): one row per 10-step block with the
        keys of :data:`DIAG_KEYS` plus ``step`` (the step it was taken on), ``steps_per_s``
        (measured on the host for the :meth:`advance` call that ran it; the first call
        includes kernel compilation and graph capture) and ``p_residual`` (the 2D solver's
        name of ``poisson_residual``, which the analysis layer reads). Empty arrays without
        diagnostics."""
        self._flush()
        keys = DIAG_KEYS + ("step", "steps_per_s", "p_residual")
        if self._diag is None or not self._diag_host:
            return {k: np.zeros(0) for k in keys}
        d = np.concatenate(self._diag_host)
        out = {k: d[:, i].copy() for i, k in enumerate(DIAG_KEYS)}
        steps = np.asarray(self._diag_steps[:len(d)], dtype=np.float64)
        out["step"] = steps
        rate = np.full(len(steps), np.nan)
        for a, b, sec in self._perf:
            m = (steps > a) & (steps <= b)
            if sec > 0:
                rate[m] = (b - a) / sec
        out["steps_per_s"] = rate
        out["p_residual"] = out["poisson_residual"].copy()
        return out

    def diagnostics_csv(self, path) -> str:
        """Write the diagnostics history as CSV (one row per recorded block, the columns of
        :meth:`diagnostics` with ``step`` first); returns the path written."""
        if self._diag is None:
            raise RuntimeError("diagnostics are off (SolverConfig3D.diagnostics=False)")
        from ..ns2d.diagnostics import write_csv
        d = self.diagnostics()
        cols = ["step"] + [k for k in d if k != "step"]
        rows = np.stack([d[k] for k in cols], axis=1) if len(d["step"]) else np.zeros((0, len(cols)))
        return str(write_csv(path, cols, rows))

    def latest_diagnostics(self) -> dict[str, float]:
        """The last diagnostics row as plain floats (empty without diagnostics), plus the
        throughput of the last :meth:`advance` call."""
        d = self.diagnostics()
        if not len(d["t"]):
            return {}
        out = {k: float(v[-1]) for k, v in d.items()}
        if self._perf:
            a, b, sec = self._perf[-1]
            out["steps_per_s"] = (b - a) / sec if sec > 0 else float("nan")
        return out

    def trace(self) -> dict:
        """Capture one time step stage by stage (the AeroJAX solver trace) without advancing
        the run: the current state is copied, stepped once on the reference core and every
        placed-physics accumulator restored. Returns ``{"stages": [...], "dt": float}`` with
        per Runge-Kutta stage cell-centred CPU tensors -- ``tendency`` (the explicit
        right-hand side, x / y / z components), ``u_star`` (the stage value after
        advection-diffusion, fans and ports), ``u_penalised`` (after the bodies' and the
        paint's penalisation and the buoyancy impulse, which is added there),
        ``divergence`` (the projection's right-hand side), ``psi`` (the pressure
        correction potential), ``grad_psi`` (the correction), ``u_projected`` and
        ``divergence_after`` -- and norms (``max_rhs``, ``max_div_after``,
        ``poisson_residual`` = |L psi - rhs| / |rhs|, ``correction_max``)."""
        rec: list = []
        phys = self._phys
        snap = [b.clone() for b in phys.buffers()] if phys is not None else []
        with torch.no_grad():
            s = self._state
            st = State3D(*(x.clone() for x in s))
            theta = self._theta.clone() if self._theta is not None else None
            step3d_phys(st, self._geom, self.ops, phys=phys, theta=theta, trace=rec)
            if phys is not None:
                for b, c in zip(phys.buffers(), snap):
                    b.copy_(c)
            out = {"stages": [], "dt": float(rec[0]["dt"]) if rec else float("nan")}
            for r in rec:
                psi = r["psi"]
                rhs = r["rhs"]
                from .core import pressure_gradient
                gx, gy, gz = pressure_gradient(psi, self.ops)
                up, vp, wp = r["u_projected"]
                after = divergence3d(up, vp, wp, self.ops.h)
                if phys is not None and phys.ports is not None:
                    after = phys.continuity(after)
                lap = self.ops.poisson.laplacian(psi)
                rhs0 = rhs - rhs.mean()

                def cen(t3):
                    return tuple(f(x).float().cpu() for f, x in zip((centre_u, centre_v, centre_w), t3))
                stage = {"stage": r["stage"], "tendency": cen(r["tendency"]), "u_star": cen(r["u_star"]),
                         "u_penalised": cen(r["u_penalised"]), "divergence": rhs.float().cpu(),
                         "psi": psi.float().cpu(), "grad_psi": cen((gx, gy, gz)), "u_projected": cen(r["u_projected"]),
                         "divergence_after": after.float().cpu(),
                         "max_rhs": float(rhs.abs().max()), "max_div_after": float(after.abs().max()),
                         "poisson_residual": float((lap - rhs0).norm() / max(float(rhs0.norm()), 1e-300)),
                         "correction_max": float(max(gx.abs().max(), gy.abs().max(), gz.abs().max()))}
                if "theta" in r:
                    stage["theta"] = r["theta"].float().cpu()
                out["stages"].append(stage)
        return out

    def trace_step(self, commit: bool = False) -> dict[str, torch.Tensor]:
        """The solver trace in the 2D solver's flat form: ``stage1..3/explicit_u|v|w`` (the
        stage value after advection, diffusion and forces), ``stage*/penalised_u|v|w``,
        ``stage*/divergence``, ``stage*/pressure_potential``, ``stage*/correction_u|v|w``
        (minus the gradient), ``stage*/projected_u|v|w``, ``stage*/theta`` with heat
        transfer, and ``final_u|v|w|p`` (``final_theta``) -- cell-centred CPU tensors.

        ``commit=False`` leaves the run untouched. ``commit=True`` then advances the run
        by one step on its normal path; that step agrees with the traced one (computed on
        the reference core) to round-off (fused vs reference: 1e-15 in float64, the
        documented float32 agreement otherwise), not bit for bit."""
        tr = self.trace()
        out: dict[str, torch.Tensor] = {}
        for st in tr["stages"]:
            k = f"stage{st['stage'] + 1}"
            for name, key in (("explicit", "u_star"), ("penalised", "u_penalised"), ("projected", "u_projected")):
                for comp, x in zip("uvw", st[key]):
                    out[f"{k}/{name}_{comp}"] = x
            for comp, x in zip("uvw", st["grad_psi"]):
                out[f"{k}/correction_{comp}"] = -x
            out[f"{k}/divergence"] = st["divergence"]
            out[f"{k}/pressure_potential"] = st["psi"]
            if "theta" in st:
                out[f"{k}/theta"] = st["theta"]
        if tr["stages"]:
            last = tr["stages"][-1]
            for comp, x in zip("uvw", last["u_projected"]):
                out[f"final_{comp}"] = x
            if "theta" in last:
                out["final_theta"] = last["theta"]
            from ..ns2d.core import FORCE_WEIGHTS  # the step-averaged pressure (core.step3d_phys)
            psi_w = sum(w * st["psi"].double() for w, st in zip(FORCE_WEIGHTS, tr["stages"]))
            p = psi_w / tr["dt"]
            out["final_p"] = (p - (p.mean() if (self.ops.periodic_axes[0] or self.ops.closed)
                                   else p[..., :2].mean())).float()
        if commit:
            self.advance(1)
        return out

    # -------------------------------------------------------------- graphs
    def _capture(self, k: int, sample_last: bool, diag_last: bool = False) -> torch.cuda.CUDAGraph:
        """Capture k steps (the last one taking an averaging sample and / or a diagnostics row
        when asked). Warm-up steps run on a side stream (NVRTC, cuBLAS, allocator) and are
        then undone, so capturing never changes the run."""
        flags = [(False, False)] * (k - 1) + [(sample_last, diag_last)]
        snap = [b.clone() for b in self._buffers()]
        try:
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._step_static(sample_last, diag_last)
            torch.cuda.current_stream(self.device).wait_stream(side)
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                for f, d in flags:
                    self._step_static(f, d)
        finally:
            for b, c in zip(self._buffers(), snap):
                b.copy_(c)
        return g

    def _run_graph(self, k: int, times: int, sample_last: bool, diag_last: bool = False) -> bool:
        key = (k, self._avg_on, sample_last and self._avg_on, diag_last)
        try:
            g = self._graphs.get(key)
            if g is None:
                g = self._graphs[key] = self._capture(k, key[2], diag_last)
            for _ in range(times):
                g.replay()
            return True
        except Exception as exc:  # noqa: BLE001 - any capture/replay failure must fall back to eager
            self.graph_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"CUDA graph path disabled, falling back to eager execution ({self.graph_error})",
                          RuntimeWarning, stacklevel=3)
            self._use_graphs = False
            self._graphs.clear()
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(self.device)
            return False

    def _invalidate_graphs(self) -> None:
        self._graphs.clear()

    def _singles(self, m: int) -> None:
        """m single steps; a step that brings the step count to a multiple of
        AVERAGE_EVERY takes the averaging sample (the same rule as the blocks)."""
        diag_on = self._diag is not None
        for _ in range(m):
            sample = (self.steps + 1) % AVERAGE_EVERY == 0
            diag = diag_on and sample
            if not (self._use_graphs and self._run_graph(1, 1, sample, diag)):
                self._step_static(sample, diag) if self._use_graphs else self._step_eager(sample, diag)
            self.steps += 1
            self._rows += 1
            if diag:
                self._diag_steps.append(self.steps)
                self._diag_rows += 1

    def advance(self, n: int) -> None:
        """Advance ``n`` time steps on the fast path (no autograd).

        Steps are run as single steps up to the next multiple of GRAPH_STEPS, then
        as whole captured 10-step blocks, then single steps again; the running
        means sample the flow (and the diagnostics take their row) at every
        multiple of AVERAGE_EVERY steps on every path, so CUDA-graph and eager runs
        stay bit-identical.
        """
        n = int(n)
        if n <= 0:
            return
        first, wall0 = self.steps, time.perf_counter()
        diag_on = self._diag is not None
        # CPU: subnormals flushed while stepping, the previous mode restored (see the 2D solver)
        with flush_denormal(self.device.type == "cpu"), torch.no_grad():
            left = n
            while left > 0:
                room = HISTORY_CAPACITY - (self._rows - self._flushed)
                if diag_on:
                    room = min(room, (DIAG_CAPACITY - (self._diag_rows - self._diag_flushed)) * GRAPH_STEPS)
                chunk = min(left, room)
                head = min(chunk, (-self.steps) % GRAPH_STEPS)
                self._singles(head)
                big = (chunk - head) // GRAPH_STEPS
                if big and self._use_graphs and self._run_graph(GRAPH_STEPS, big, True, diag_on):
                    if diag_on:
                        self._diag_steps.extend(self.steps + GRAPH_STEPS * (b + 1) for b in range(big))
                        self._diag_rows += big
                    self.steps += big * GRAPH_STEPS
                    self._rows += big * GRAPH_STEPS
                else:
                    self._singles(big * GRAPH_STEPS)
                self._singles(chunk - head - big * GRAPH_STEPS)
                left -= chunk
                if self._rows - self._flushed >= HISTORY_CAPACITY or (
                        diag_on and self._diag_rows - self._diag_flushed >= DIAG_CAPACITY):
                    self._flush()
            self._flush()
        self._perf.append((first, self.steps, time.perf_counter() - wall0))
        if len(self._perf) > 10000:
            del self._perf[:5000]

    def _flush(self) -> None:
        k = self._rows - self._flushed
        if k > 0:
            a = self._flushed % HISTORY_CAPACITY
            b = a + k

            def take(buf):
                return buf[a:b] if b <= HISTORY_CAPACITY else torch.cat([buf[a:], buf[:b - HISTORY_CAPACITY]])
            self._host_hist.append(take(self._hist).detach().to("cpu", torch.float64).numpy())
            if self._phist is not None:
                self._phist_host.append(take(self._phist).detach().to("cpu", torch.float64).numpy())
            self._flushed = self._rows
        if self._diag is not None:
            k = self._diag_rows - self._diag_flushed
            if k > 0:
                a = self._diag_flushed % DIAG_CAPACITY
                b = a + k
                rows = self._diag[a:b] if b <= DIAG_CAPACITY else torch.cat(
                    [self._diag[a:], self._diag[:b - DIAG_CAPACITY]])
                self._diag_host.append(rows.detach().to("cpu", torch.float64).numpy())
                self._diag_flushed = self._diag_rows

    # ----------------------------------------------------------- coefficients
    def coefficients(self) -> np.ndarray:
        """Latest [CD, CS, CL, C_roll, Cm, Cn] summed over all bodies (full-body values)."""
        return self._body_c.sum(0).detach().to("cpu", torch.float64).numpy()

    def body_coefficients(self) -> np.ndarray:
        """Latest coefficients per body, shape (nbody, 6)."""
        return self._body_c.detach().to("cpu", torch.float64).numpy().copy()

    def history(self) -> dict[str, np.ndarray]:
        """t, cd, cs, cl, c_roll, cm, cn for every step since the last reset."""
        self._flush()
        h = np.concatenate(self._host_hist) if self._host_hist else np.zeros((0, 7))
        return {k: h[:, i].copy() for i, k in enumerate(HISTORY_KEYS)}

    def trim_history(self, keep: float) -> None:
        """Forget host history older than ``keep`` convective times (mirrors the 2D solver).

        Long explore sessions copy the force history, the placed-physics history and the
        diagnostics every frame, so all three host logs grow without bound unless trimmed
        together. Each is cut by its own time column (the physics and diagnostics rows are not
        recorded on every step), and the diagnostics' step list, which :meth:`diagnostics`
        pairs with the rows by position, is cut with its rows. The device ring buffers are
        untouched: they only hold what has not been flushed yet.
        """
        self._flush()
        if not self._host_hist:
            return
        h = np.concatenate(self._host_hist)
        if not len(h):
            return
        t_cut = float(h[-1, 0]) - float(keep)
        self._host_hist[:] = [h[h[:, 0] >= t_cut]]
        if self._phist_host:
            d = np.concatenate(self._phist_host)
            self._phist_host[:] = [d[d[:, 0] >= t_cut]]
        if self._diag_host:
            d = np.concatenate(self._diag_host)
            n = min(len(d), len(self._diag_steps))
            later = d[:n, DIAG_KEYS.index("t")] >= t_cut
            k = int(np.argmax(later)) if later.any() else n
            self._diag_host[:] = [d[k:]]
            del self._diag_steps[:k]

    def physics_history(self) -> dict[str, np.ndarray]:
        """Placed-physics series per step, aligned with :meth:`history` (from the step at which
        the physics was last restructured): ``t``; with heat transfer ``heat_rate_<i>`` per
        body (conduction units k dT L, full body), ``heat_rate`` (their sum),
        ``theta_wall_<i>`` per heated body (the held temperature, or the measured skin
        temperature of a "heat_flux" body, :meth:`.physics.ThermalGeometry.wall_theta`) and
        ``nu`` (the mean Nusselt number of the heated bodies, sum Q / sum A theta_w); per fan
        ``fan<i>_thrust`` (rho U^2 L^2), ``fan<i>_velocity`` (disc-averaged axial velocity)
        and ``fan<i>_flow`` (U L^2). Empty without placed-physics outputs."""
        self._flush()
        if self._phist is None or not self._phist_host:
            return {}
        d = np.concatenate(self._phist_host)
        out = {"t": d[:, 0].copy()}
        c = 1
        ph = self._phys
        if ph.thermal is not None:
            nb = len(self._bodies)
            q = d[:, c:c + nb]
            c += nb
            for i in range(nb):
                out[f"heat_rate_{i}"] = q[:, i].copy()
            g = ph.thermal.geom
            out["heat_rate"] = q.sum(axis=1)
            heated = [b for b, m in enumerate(g.modes) if m == "temperature" or (m == "heat_flux" and self._phist_walls)]
            tw = np.tile(np.where(np.array(g.modes) == "temperature", g.values, np.nan), (len(q), 1))
            if self._phist_walls:
                tw = d[:, c:c + nb].copy()
                c += nb
                for i, m in enumerate(g.modes):
                    if m == "adiabatic":
                        tw[:, i] = np.nan
            for i, m in enumerate(g.modes):
                if m != "adiabatic":
                    out[f"theta_wall_{i}"] = tw[:, i].copy()
            if heated:
                # sum Q / sum A theta_w over the heated bodies (theta_ref = the inlet / ambient 0)
                den = (tw[:, heated] * g.area[heated]).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["nu"] = np.where(np.abs(den) > 1e-300, q[:, heated].sum(axis=1) / den, np.nan)
        if ph.fans is not None:
            nf = len(ph.fans)
            thrust, ud = d[:, c:c + nf], d[:, c + nf:c + 2 * nf]
            for i, f in enumerate(ph.fans.fans):
                mult = (2.0 if self.ops.symmetry else 1.0) * ph.fans.items[i]["frac"]
                out[f"fan{i}_thrust"] = thrust[:, i] * mult
                out[f"fan{i}_velocity"] = ud[:, i].copy()
                out[f"fan{i}_flow"] = ud[:, i] * f.area * mult
        return out

    # --------------------------------------------------------- physics reports
    def heat_rates(self) -> list[dict]:
        """Per body: ``name``, ``mode``, ``heat_rate`` (into the fluid, conduction units k dT L,
        full body), ``area`` (L^2), ``theta`` (the held temperature, or the measured skin
        temperature of a "heat_flux" body) and for both ``nusselt`` = Q / (A theta)."""
        if self._phys is None or self._phys.thermal is None:
            return []
        th = self._phys.thermal
        q = th.heat_rate.detach().double().cpu().numpy()
        g = th.geom
        with torch.no_grad():
            tw = g.wall_theta(self._theta).double().cpu().numpy()
        out = []
        for i, b in enumerate(self._bodies):
            mode = getattr(b, "thermal", "adiabatic")
            th_b = float(getattr(b, "thermal_value", 0.0)) if mode == "temperature" else float(tw[i])
            measured = mode == "temperature" or (mode == "heat_flux" and g.wall_idx is not None)
            nu = q[i] / (g.area[i] * th_b) if measured and th_b != 0 and g.area[i] > 0 else float("nan")
            out.append({"name": b.name, "mode": mode, "heat_rate": float(q[i]), "area": float(g.area[i]),
                        "theta": th_b if measured else float("nan"), "nusselt": float(nu)})
        return out

    def wall_heat_rates(self) -> dict[str, float]:
        """Heat into the fluid through each domain side now (conduction units, full body);
        a tunnel's "left" includes the heat the inflow carries in, "right" the heat the
        outflow carries out (negative)."""
        if self._theta is None:
            return {}
        with torch.no_grad():
            return self._phys.thermal.wall_heat_rates(self._theta, self._state, self.ops)

    def heat_balance(self) -> dict[str, float]:
        """Energy balance of the current state in conduction units (k dT L), with the keys the
        analysis layer reads (``analysis.physics.HEAT_IN_TERMS`` / ``HEAT_OUT_TERMS``): heat in
        from bodies, walls, sources, jets and the inlet, out through the outlet and vents,
        the storage rate (from the last diagnostics row, NaN without diagnostics) and the
        residual in - out - storage."""
        if self._theta is None:
            return {}
        th = self._phys.thermal
        sym = 2.0 if self.ops.symmetry else 1.0
        walls = self.wall_heat_rates()
        tunnel = not self.cfg.closed and not self.cfg.periodic
        inlet = walls.pop("left") if tunnel else 0.0
        outlet = -walls.pop("right") if tunnel else 0.0
        out = {"heat_in_bodies": float(th.heat_rate.double().sum()), "heat_in_walls": float(sum(walls.values())),
               "heat_in_sources": th.sources.total * sym, "heat_in_inlet": inlet, "heat_out_outlet": outlet,
               "heat_in_ports": 0.0, "heat_out_ports": 0.0}
        ports = self._phys.ports
        if ports is not None:
            h3 = self.ops.h ** 3
            with torch.no_grad():
                for it in ports.items:
                    sl = (slice(it["win"][0], it["win"][1]), slice(it["win"][2], it["win"][3]),
                          slice(it["win"][4], it["win"][5]))
                    if it["kind"] == "jet":
                        out["heat_in_ports"] += float(it["m"].sum()) * h3 * it["theta"] * th.pe * sym
                    else:
                        out["heat_out_ports"] -= float((it["m"] * self._theta[sl]).sum()) * h3 * th.pe * sym
                if ports.leak:
                    # the uniform leak of an unbalanced closed box carries the mean fluid's heat out
                    out["heat_out_ports"] += ports.leak * float(self._theta.double().sum()) * h3 * th.pe * sym
        d = self.latest_diagnostics()
        out["heat_storage_rate"] = d.get("heat_storage", float("nan"))
        tot_in = sum(v for k, v in out.items() if k.startswith("heat_in"))
        tot_out = out["heat_out_outlet"] + out["heat_out_ports"]
        out["heat_balance_residual"] = tot_in - tot_out - out["heat_storage_rate"]
        return out

    def fan_report(self) -> list[dict]:
        """Per fan (solver units; full-body / mirrored values with a symmetry plane): ``thrust``,
        ``pressure_jump``, ``disc_velocity``, ``flow`` (also as ``flow_rate``, the 2D
        spelling), ``induced_velocity``, ``power`` (ideal, T x disc velocity), ``target``
        (velocity mode) and ``in_domain`` (the share of the disc inside the solved domain)."""
        if self._phys is None or self._phys.fans is None:
            return []
        rep = self._phys.fans.report(self.ops.symmetry)
        for r in rep:
            r["flow_rate"] = r["flow"]
        return rep

    def _history_window(self, ph: dict) -> np.ndarray | None:
        """Time weights of the physics-history rows inside the averaging window (None when
        nothing has been averaged): each row stands for the step that ended at its time."""
        if self._avg is None or self.averaging_time <= 0 or not len(ph.get("t", ())):
            return None
        t = ph["t"]
        dt = np.diff(np.concatenate([[t[0] - (t[1] - t[0]) if len(t) > 1 else 0.0], t]))
        w = np.where(t > self._avg_t0 + 1e-12 * max(1.0, abs(self._avg_t0)), dt, 0.0)
        if self._avg_stop is not None:
            w = np.where(t <= self._avg_stop + 1e-12 * max(1.0, abs(self._avg_stop)), w, 0.0)
        return w if w.sum() > 0 else None

    def fans_report(self, mean: bool = False) -> list[dict]:
        """:meth:`fan_report`, or with ``mean`` its time average over the running-means window
        (since :meth:`start_averaging`; the instantaneous report when nothing is averaged).
        The same keys as the 2D solver's ``fans_report`` plus the 3D ones."""
        rep = self.fan_report()
        if not rep or not mean:
            return rep
        ph = self.physics_history()
        w = self._history_window(ph)
        if w is None:
            return rep
        fans = self._phys.fans
        for i, (r, f) in enumerate(zip(rep, fans.fans)):
            mult = (2.0 if self.ops.symmetry else 1.0) * fans.items[i]["frac"]
            thrust, ud = ph[f"fan{i}_thrust"], ph[f"fan{i}_velocity"]
            T = float(np.average(thrust, weights=w))
            u = float(np.average(ud, weights=w))
            r.update({"thrust": T, "pressure_jump": T / (mult * f.area) if mult > 0 else float("nan"),
                      "disc_velocity": u, "flow": u * f.area * mult, "flow_rate": u * f.area * mult,
                      "induced_velocity": u - fans.u_inf * f.direction[0],
                      "power": float(np.average(thrust * ud, weights=w))})
        return rep

    def heat_transfer(self, mean: bool = False, theta_ref: float | None = None) -> dict:
        """Heat transfer of the bodies and the domain walls in the 2D solver's structure.

        ``bodies``: per body ``name``, ``mode``, ``area`` (L^2), ``heat_rate`` (into the
        fluid, conduction units k dT_ref L, full body), ``theta_wall`` (the held
        temperature; for a "heat_flux" body the measured temperature of the skin its heat
        is released into, :meth:`.physics.ThermalGeometry.wall_theta`; for an adiabatic body
        the area-weighted surface temperature extrapolated from the flow) and ``nusselt`` =
        Q / (A (theta_wall - theta_ref)), the mean Nusselt number
        on the reference length L (DESIGN.md 10.2), NaN for adiabatic bodies. ``walls``:
        heat into the fluid through each side; ``nusselt_mean``: the mean over the heated
        bodies; ``pe``; ``units`` = "conduction". ``theta_ref`` defaults to the inlet
        temperature. ``mean`` averages the heat rates over the running-means window and
        takes the walls and surface temperatures from the mean fields (the walls' advected
        inlet / outlet terms then use the mean velocity: a product of means, not the mean
        product)."""
        empty = {"bodies": [], "walls": {}, "nusselt_mean": float("nan"), "pe": float("nan"), "units": "conduction"}
        if self._theta is None:
            return empty
        th = self._phys.thermal
        ref = float(self.cfg.thermal.theta_inlet) if theta_ref is None else float(theta_ref)
        q = th.heat_rate.detach().double().cpu().numpy()
        g = th.geom
        use_mean = mean and self._avg is not None and self.averaging_time > 0
        theta_now = self._theta
        if use_mean:
            ph = self.physics_history()
            w = self._history_window(ph)
            if w is not None:
                q = np.array([float(np.average(ph[f"heat_rate_{i}"], weights=w)) for i in range(len(self._bodies))])
            u, _, _, _, rms = self._staggered_means()
            theta_now = rms.get("t_mean", self._theta)
            from types import SimpleNamespace
            with torch.no_grad():
                walls = th.wall_heat_rates(theta_now, SimpleNamespace(u=u), self.ops)
        else:
            walls = self.wall_heat_rates()
        with torch.no_grad():
            skin_t = g.wall_theta(theta_now).double().cpu().numpy()     # measured "heat_flux" walls
        need_probe = any(getattr(b, "thermal", "adiabatic") == "adiabatic"
                         or (getattr(b, "thermal", "adiabatic") == "heat_flux" and g.wall_idx is None)
                         for b in self._bodies)
        surf = self.surface(mean=use_mean) if need_probe else []
        bodies, nus = [], []
        for i, b in enumerate(self._bodies):
            mode = getattr(b, "thermal", "adiabatic")
            if mode == "temperature":
                tw = float(b.thermal_value)
            elif mode == "heat_flux" and g.wall_idx is not None:
                tw = float(skin_t[i])
            else:
                s = surf[i]
                v_, f_ = s["vertices"], s["faces"]
                area = 0.5 * np.linalg.norm(np.cross(v_[f_[:, 1]] - v_[f_[:, 0]], v_[f_[:, 2]] - v_[f_[:, 0]]), axis=1)
                tw = float((s["theta_wall"][f_].mean(axis=1) * area).sum() / max(area.sum(), 1e-300))
            rec = {"name": b.name, "mode": mode, "area": float(g.area[i]), "heat_rate": float(q[i]),
                   "theta_wall": tw, "nusselt": float("nan")}
            if mode != "adiabatic" and abs(tw - ref) > 1e-12 and g.area[i] > 0:
                rec["nusselt"] = rec["heat_rate"] / (g.area[i] * (tw - ref))
                nus.append(rec["nusselt"])
            bodies.append(rec)
        return {"bodies": bodies, "walls": walls, "pe": th.pe, "units": "conduction",
                "nusselt_mean": float(np.mean(nus)) if nus else float("nan")}

    def set_temperature(self, theta) -> None:
        """Overwrite the temperature field (a number, or an array / tensor of shape
        (nz, ny, nx)); e.g. a conduction profile to start a natural-convection run from.
        The flow is untouched; captured graphs stay valid (the buffer is filled in place)."""
        if self._theta is None:
            raise ValueError("heat transfer is off (cfg.thermal is None)")
        with torch.no_grad():
            if isinstance(theta, (int, float)):
                self._theta.fill_(float(theta))
                return
            t = torch.as_tensor(np.asarray(theta) if not torch.is_tensor(theta) else theta)
            if tuple(t.shape) != tuple(self._theta.shape):
                raise ValueError(f"temperature field must have shape {tuple(self._theta.shape)}")
            self._theta.copy_(t.to(self.device, self.dtype))

    def particles(self) -> dict[str, np.ndarray]:
        """The live tracer particles: ``positions`` (n, 3) float32, ``age`` (n,) since release,
        ``birth`` (n,), ``emitter`` (n,) index among the particle emitters and ``nozzle``
        (n,) the rake nozzle it left from (solver units)."""
        if self._particles is None:
            return {"positions": np.zeros((0, 3), np.float32), "age": np.zeros(0), "birth": np.zeros(0),
                    "emitter": np.zeros(0, np.int16), "nozzle": np.zeros(0, np.int16)}
        snap = self._particles.snapshot()
        return {"positions": snap["positions"], "age": self.t - snap["birth"], "birth": snap["birth"],
                "emitter": snap["emitter"], "nozzle": snap["nozzle"]}

    def particle_lines(self) -> list[np.ndarray]:
        """Streak lines of the "point" and "rake" particle emitters as (n, 3) polylines from
        the nozzle outwards, cut at the pauses of pulsed emitters
        (:meth:`.emitters.ParticleSystem3D.lines`); empty without particle emitters."""
        return self._particles.lines() if self._particles is not None else []

    def particle_count(self) -> int:
        return 0 if self._particles is None else self._particles.count()

    # ------------------------------------------------------------------ fields
    def _p_ref(self, p: torch.Tensor) -> torch.Tensor:
        """Reference static pressure: mean over the two inlet planes (domain mean if
        periodic in x or closed). cp is relative to it."""
        if self.ops.periodic_axes[0] or self.ops.closed:
            return p.mean()
        return p[..., :2].mean()

    def _mask_field(self) -> torch.Tensor:
        m = torch.zeros(self.ops.nz, self.ops.ny, self.ops.nx, device=self.device, dtype=self.dtype)
        if not self._geom.empty:
            k0, k1, j0, j1, i0, i1 = self._geom.window
            m[k0:k1, j0:j1, i0:i1] = self._geom.chi_c
        if self._phys is not None and self._phys.paint is not None:
            m = torch.maximum(m, self._phys.paint.chi_c)
        return m

    def _centred(self, name: str, u, v, w, p, *, rms: dict | None = None, theta=None) -> torch.Tensor:
        """A cell-centred field from staggered (u, v, w, p) (and rms arrays for *_rms)."""
        if name == "u":
            return centre_u(u)
        if name == "v":
            return centre_v(v)
        if name == "w":
            return centre_w(w)
        if name == "speed":
            uc, vc, wc = centre_u(u), centre_v(v), centre_w(w)
            return torch.sqrt(uc * uc + vc * vc + wc * wc)
        if name == "p":
            return p - self._p_ref(p)
        if name == "cp":
            return 2.0 * (p - self._p_ref(p))
        if name == "mask":
            return self._mask_field()
        if name in _DYE_NAMES:      # smoke is always the current snapshot (a mean would be a blur)
            ch = _DYE_NAMES.index(name)
            chans = self._channels()
            return chans[ch].clone() if ch < len(chans) else torch.zeros_like(p)
        if name in ("temperature", "theta"):
            if theta is not None:
                return theta.clone()
            return self._theta.clone() if self._theta is not None else torch.zeros_like(p)
        if name == "divergence":
            div = divergence3d(u, v, w, self.ops.h)
            if self._phys is not None and self._phys.ports is not None:
                div = self._phys.continuity(div)
            return div
        if name in GRADIENT_FIELDS:
            mask = self._mask_field() if name == "nut" else None
            return gradient_field(name, u, v, w, self.ops, mask=mask)
        if rms is not None and name in ("u_rms", "v_rms", "w_rms", "p_rms", "cp_rms", "temperature_rms"):
            if name == "u_rms":
                return centre_u(rms["u"])
            if name == "v_rms":
                return centre_v(rms["v"])
            if name == "w_rms":
                return centre_w(rms["w"])
            if name == "temperature_rms":
                return rms["t"] if "t" in rms else torch.zeros_like(p)
            return rms["p"] * (2.0 if name == "cp_rms" else 1.0)
        raise KeyError(f"unknown field {name!r}; available: {', '.join(FIELDS3D)}")

    def field(self, name: str) -> torch.Tensor:
        """A cell-centred (nz, ny, nx) field of the current state (a new tensor).

        "u", "v", "w", "speed": velocity (units of U); "p": pressure relative to
        the inlet (rho U^2; the domain mean in a closed domain); "cp" = 2 p;
        "vorticity": |curl u| (U/L); "q": Q-criterion (U^2/L^2); "mask": solid
        fraction (bodies and paint); "nut": eddy viscosity (U L); "divergence": the
        continuity residual div u - m (round-off level; m the ports' sources);
        "dye", "dye1".."dye3": smoke concentration 0..1 of each channel (zero when
        absent); "temperature": theta = (T - T_amb) / dT_ref (zero without heat
        transfer).
        """
        s = self._state
        with torch.no_grad():
            return self._centred(name, s.u, s.v, s.w, s.p)

    # --------------------------------------------------------------- averaging
    def start_averaging(self) -> None:
        """(Re)start time-weighted running means of u, v, w, p (theta with heat transfer)
        and their variances."""
        with torch.no_grad():
            if self._avg is None:
                s = self._state
                self._avg = {"time": torch.zeros((), device=self.device, dtype=torch.float64),
                             "pending": torch.zeros((), device=self.device, dtype=torch.float64)}
                items = [("u", s.u), ("v", s.v), ("w", s.w), ("p", s.p)]
                if self._theta is not None:
                    items.append(("t", self._theta))
                for name, x in items:
                    self._avg[name] = torch.zeros_like(x)
                    self._avg[name + "2"] = torch.zeros_like(x)
            else:
                for a in self._avg.values():
                    a.zero_()
            self._avg_on = True
        self._avg_t0 = self.t
        self._avg_stop = None

    def stop_averaging(self) -> None:
        """Freeze the running means (they stay readable)."""
        if self._avg_on:
            self._avg_stop = self.t
        self._avg_on = False

    def _staggered_means(self):
        """Time-averaged (u, v, w, p) on their own grids and their rms (dict), or the
        instantaneous state (rms zero) before any time has been averaged."""
        if self._avg is None:
            raise RuntimeError("call start_averaging() first")
        a = self._avg
        s = self._state
        T, e = float(a["time"]), float(a["pending"])
        names = "uvwpt" if "t" in a else "uvwp"
        current = {"u": s.u, "v": s.v, "w": s.w, "p": s.p, "t": self._theta}
        if T + e <= 0:
            return s.u, s.v, s.w, s.p, {k: torch.zeros_like(current[k]) for k in names}
        means, rms = {}, {}
        for k in names:
            mean, m2 = a[k], a[k + "2"]
            if e > 0:          # the time since the last sample, folded in without touching the buffers
                x = current[k]
                d = x - mean
                mean = mean + d * (e / (T + e))
                m2 = m2 + e * d * (x - mean)
            means[k] = mean
            rms[k] = (m2 / (T + e)).clamp_min(0.0).sqrt()
        if "t" in means:
            rms["t_mean"] = means["t"]
        return means["u"], means["v"], means["w"], means["p"], rms

    def mean_fields(self, names: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
        """Time averages since :meth:`start_averaging`, cell-centred (nz, ny, nx).

        Keys (all by default): u, v, w, speed, p, cp, vorticity and q (of the
        mean flow), u_rms, v_rms, w_rms, p_rms, cp_rms, mask; with heat transfer
        also temperature and temperature_rms. Falls back to the instantaneous
        fields if no time has been averaged yet.
        """
        if names is None:
            names = MEAN_FIELDS3D + (MEAN_FIELDS3D_THERMAL if self._theta is not None else ())
        with torch.no_grad():
            u, v, w, p, rms = self._staggered_means()
            theta = rms.get("t_mean")
            return {k: self._centred(k, u, v, w, p, rms=rms, theta=theta) for k in names}

    # ------------------------------------------------------------------ slices
    def slice(self, axis: str, position: float, name: str, *, mean: bool = False) -> torch.Tensor:
        """A planar cut of a cell-centred field at ``position`` (solver units) normal to ``axis``.

        Orientation (SliceData): "y" -> (nz, nx) rows along z; "z" -> (ny, nx)
        rows along y; "x" -> (nz, ny) rows along z, columns along y. Linear
        interpolation between the two nearest cell planes (clamped to the
        domain). ``mean=True`` cuts the running means (``*_rms`` names allowed).
        """
        if axis not in _AXES:
            raise ValueError("axis must be 'x', 'y' or 'z'")
        dim = _AXES[axis]
        n = (self.ops.nz, self.ops.ny, self.ops.nx)[dim]
        origin = (self.ops.z0, self.ops.y0, self.ops.x0)[dim]
        f = min(max((position - origin) / self.ops.h - 0.5, 0.0), n - 1.0)
        i0 = min(math.floor(f), max(n - 2, 0))
        t = f - i0
        i1 = min(i0 + 1, n - 1)
        with torch.no_grad():
            theta = None
            if mean:
                u, v, w, p, rms = self._staggered_means()
                theta = rms.get("t_mean")
            else:
                s = self._state
                u, v, w, p, rms = s.u, s.v, s.w, s.p, None
            if name in GRADIENT_FIELDS:
                mask = None
                if name == "nut":
                    mask = self._mask_field().narrow(dim, i0, i1 - i0 + 1)
                block = gradient_field(name, u, v, w, self.ops, mask=mask, axis=dim, lo=i0, hi=i1 + 1)
                a, b = block.narrow(dim, 0, 1), block.narrow(dim, i1 - i0, 1)
            else:
                full = self._centred(name, u, v, w, p, rms=rms, theta=theta)
                a, b = full.narrow(dim, i0, 1), full.narrow(dim, i1, 1)
            return ((1.0 - t) * a + t * b).squeeze(dim)

    # ----------------------------------------------------------------- surface
    def surface(self, mean: bool = False) -> list[dict]:
        """Surface pressure and friction per mesh vertex (instantaneous or time-averaged).

        For each body: ``name``, ``vertices`` (N, 3) and ``faces`` as solved,
        outward unit ``normals`` (N, 3), ``cp`` (N,), ``cf`` (N,) the skin-friction
        magnitude and ``tau`` (N, 3) the wall-shear direction scaled by cf, numpy
        float64. As in 2D the wall is diffuse, so wall values are extrapolated
        from probes 1.5 h and 2.5 h off the wall along the vertex normal: cp
        linearly, ``tau = (2/Re) du_t/dn`` from the quadratic through the wall
        velocity (zero, or omega x r on a wheel) and the tangential velocity at
        both probes (molecular viscosity only).

        With heat transfer also ``theta_wall`` (the body temperature; for a "heat_flux"
        body theta sampled on the surface, inside the skin its heat is released into; for
        an adiabatic body theta extrapolated linearly to the wall) and ``heat_flux`` (into the
        fluid, conduction units: the local Nusselt number times theta_wall; from
        the quadratic through the wall value and both probes for "temperature"
        bodies, the set flux G for "heat_flux" bodies, 0 for adiabatic ones).

        Normals are area-weighted and oriented per vertex towards the fluid (the
        side where the solid fraction at the first probe is lower), so meshes of
        any winding work; on a sheet with fluid on both sides the mesh's own
        orientation is kept. With a symmetry plane, probes of vertices on the
        missing half are sampled at their mirror image (v reversed). Probes that
        land inside another body or outside the domain return what is there
        (clamped): treat contact regions (wheel on road) as undefined.
        """
        ops = self.ops
        theta = self._theta
        if mean:
            U, V, W, P, rms = self._staggered_means()
            theta = rms.get("t_mean", theta)
        else:
            s = self._state
            U, V, W, P = s.u, s.v, s.w, s.p
        d1, d2 = probe_distances(ops)
        with torch.no_grad():
            p_ref = float(self._p_ref(P))
            mask = self._mask_field()
        out = []
        for b in self._bodies:
            verts = b.vertices_numpy()
            faces = np.asarray(b.faces, dtype=np.int64)
            nrm = vertex_normals(verts, faces)

            def sample(points: np.ndarray, field: torch.Tensor, off, flip_v: bool = False) -> np.ndarray:
                pts = points.copy()
                sgn = np.ones(len(pts))
                if ops.symmetry:
                    low = pts[:, 1] < ops.y0
                    pts[low, 1] = 2.0 * ops.y0 - pts[low, 1]
                    if flip_v:
                        sgn[low] = -1.0
                q = torch.as_tensor(pts, device=self.device, dtype=self.dtype)
                with torch.no_grad():
                    val = trilinear(field, q, ops, off).double().cpu().numpy()
                return val * sgn

            centre = (0.5, 0.5, 0.5)
            # orient each normal towards the fluid
            solid_out = sample(verts + d1 * nrm, mask, centre)
            solid_in = sample(verts - d1 * nrm, mask, centre)
            nrm = np.where((solid_out > solid_in + 1e-6)[:, None], -nrm, nrm)
            q1, q2 = verts + d1 * nrm, verts + d2 * nrm
            p1, p2 = sample(q1, P, centre) - p_ref, sample(q2, P, centre) - p_ref
            cp = 2.0 * (p1 + (p1 - p2) * d1 / (d2 - d1))
            vel = []
            for q in (q1, q2):
                vel.append(np.stack([sample(q, U, (0.0, 0.5, 0.5)), sample(q, V, (0.5, 0.0, 0.5), flip_v=True),
                                     sample(q, W, (0.5, 0.5, 0.0))], axis=1))
            uw = b.omega * np.cross(np.asarray(b.axis), verts - np.asarray(b.center))     # wall velocity
            tang = []
            for vq in vel:
                rel = vq - uw
                tang.append(rel - np.sum(rel * nrm, axis=1, keepdims=True) * nrm)
            dudn = (tang[0] * d2 * d2 - tang[1] * d1 * d1) / (d1 * d2 * (d2 - d1))
            tau = 2.0 / self.cfg.re * dudn
            entry = {"name": b.name, "vertices": verts.copy(), "faces": faces.copy(), "normals": nrm,
                     "cp": cp, "cf": np.linalg.norm(tau, axis=1), "tau": tau}
            if theta is not None:
                t1, t2 = sample(q1, theta, centre), sample(q2, theta, centre)
                mode = getattr(b, "thermal", "adiabatic")
                if mode == "temperature":
                    tw = np.full(len(verts), float(b.thermal_value))
                    # d theta / dn at the wall from the quadratic through (0, tw), (d1, t1), (d2, t2)
                    dtdn = ((t1 - tw) * d2 * d2 - (t2 - tw) * d1 * d1) / (d1 * d2 * (d2 - d1))
                    qf = -dtdn
                elif mode == "heat_flux":
                    # theta on the surface itself, inside the skin the heat is released into (the
                    # probes' extrapolation reads ~ambient when the boundary layer is ~1 cell thick)
                    tw = sample(verts, theta, centre)
                    qf = np.full(len(verts), float(b.thermal_value))
                else:
                    tw = t1 + (t1 - t2) * d1 / (d2 - d1)
                    qf = np.zeros(len(verts))
                entry["theta_wall"] = tw
                entry["heat_flux"] = qf
            out.append(entry)
        return out

    # -------------------------------------------------------------- live edits
    def set_inflow(self, *, re: float | None = None) -> None:
        """Change the Reynolds number live. On the reference path graphs stay valid
        (nu and eta are tensors); the fused kernels take nu as a launch argument,
        so their graphs are re-captured lazily, and so does heat transfer (Pe = Re Pr
        enters as a launch value too)."""
        if re is not None:
            if not (re > 0 and math.isfinite(re)):
                raise ValueError("Reynolds number must be positive and finite")
            self.cfg.re = float(re)
            with torch.no_grad():
                set_reynolds(self.ops, float(re))
                if self._phys is not None and self._phys.thermal is not None:
                    self._phys.thermal.set_reynolds(float(re), self.ops)
            if self._kern is not None or (self._phys is not None and self._phys.thermal is not None):
                self._invalidate_graphs()

    def set_bodies(self, bodies: list[ImmersedMesh]) -> None:
        """Replace the bodies live; the flow field is kept.

        If the new bodies fit in the current window (same body count), the masks
        are rebuilt into the existing tensors and captured graphs remain valid;
        otherwise the window is rebuilt and graphs are re-captured lazily. With heat
        transfer the bodies' thermal conditions are rebuilt as well (graphs kept when
        the window and every body's thermal mode are unchanged).
        """
        bodies = [b if isinstance(b, ImmersedMesh) else ImmersedMesh(*b) for b in bodies]
        win = body_window3d([b.vertices_numpy() for b in bodies], self.ops)
        old = self._geom
        same_window = False
        with torch.no_grad():
            if (len(bodies) == len(self._bodies) and not old.empty and win[1] > win[0]
                    and window_contains(old.window, win)):
                g = build_geometry3d(bodies, self.ops, window=old.window)
                for dst, src in zip(old[1:-1], g[1:-1]):
                    dst.copy_(src)
                same_window = True
            else:
                self._geom = build_geometry3d(bodies, self.ops, window=self._live_window(bodies))
                if len(bodies) != self._body_c.shape[0]:
                    self._body_c = torch.zeros(len(bodies), 6, device=self.device, dtype=self.dtype)
                self._invalidate_graphs()
        modes_same = [getattr(b, "thermal", "adiabatic") for b in bodies] == \
            [getattr(b, "thermal", "adiabatic") for b in self._bodies]
        self._bodies = bodies
        self.cfg.bodies = list(bodies)
        if self._phys is not None and self._phys.thermal is not None:
            from .physics import ThermalGeometry
            th = self._phys.thermal
            with torch.no_grad():
                new = ThermalGeometry(bodies, self.ops, self._geom.window)
                if same_window and modes_same and _copy_thermal_geometry(th.geom, new):
                    return
                th.geom = new
                th.heat_acc = torch.zeros(len(bodies), device=self.device, dtype=self.dtype)
                th.heat_rate = torch.zeros(len(bodies), device=self.device, dtype=self.dtype)
                th.t_index = torch.as_tensor(new.t_index, device=self.device, dtype=torch.long)
            self._invalidate_graphs()
            self._setup_physics_history()

    def reset(self) -> None:
        """Back to the initial condition at t = 0; history and means cleared."""
        with torch.no_grad():
            fresh = init_state3d(self.cfg, self.ops)
            if self._use_graphs:
                for dst, src in zip(self._state, fresh):
                    dst.copy_(src)
            else:
                self._state = fresh
            self._body_c.zero_()
            self._hist_n.zero_()
            for c in self._channels():
                c.zero_()
            if self._theta is not None:
                self._theta.fill_(self._phys.thermal.initial)
            if self._phys is not None:
                self._phys.reset()
            if self._particles is not None:
                self._particles.reset()
            if self._avg is not None:
                for a in self._avg.values():
                    a.zero_()
            if self._diag is not None:
                self._diag_n.zero_()
        self._avg_on = False
        self._avg_t0, self._avg_stop = 0.0, None
        self._host_hist.clear()
        self._phist_host.clear()
        self._diag_host.clear()
        self._diag_steps.clear()
        self._diag_rows = self._diag_flushed = 0
        self._rows = self._flushed = 0
        self.steps = 0
        self._perf.clear()

    # ------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        """Everything needed to resume: state tensors (CPU), counters, history."""
        s = self._state

        def snap(x: torch.Tensor) -> torch.Tensor:
            return x.detach().to("cpu", copy=True)
        d = {"version": 1, "u": snap(s.u), "v": snap(s.v), "w": snap(s.w), "p": snap(s.p), "t": snap(s.t),
             "dt": snap(s.dt), "steps": self.steps, "re": self.cfg.re, "history": self.history(),
             "body_coefficients": snap(self._body_c)}
        if self._dye is not None:
            d["dye"] = snap(self._dye)
            for i, c in enumerate(self._dye_extra, start=1):
                d[f"dye{i}"] = snap(c)
        if self._theta is not None:
            d["theta"] = snap(self._theta)
        if self._phys is not None and self._phys.fans is not None:
            d["fan_thrust"] = snap(self._phys.fans.thrust)
            d["fan_error"] = snap(self._phys.fans.e_prev)
        return d

    def load_state_dict(self, d: dict) -> None:
        """Restore a :meth:`state_dict` taken from a solver with the same grid."""
        s = self._state
        for name in ("u", "v", "w"):
            if tuple(d[name].shape) != tuple(getattr(s, name).shape):
                raise ValueError("state_dict grid does not match this solver")
        with torch.no_grad():
            for name in ("u", "v", "w", "p", "t", "dt"):
                getattr(s, name).copy_(d[name].to(self.device, self.dtype))
            if "body_coefficients" in d and d["body_coefficients"].shape == self._body_c.shape:
                self._body_c.copy_(d["body_coefficients"].to(self.device, self.dtype))
            for i, c in enumerate(self._channels()):
                key = "dye" if i == 0 else f"dye{i}"
                if key in d and tuple(d[key].shape) == tuple(c.shape):
                    c.copy_(d[key].to(self.device, self.dtype))
            if self._theta is not None and "theta" in d and tuple(d["theta"].shape) == tuple(self._theta.shape):
                self._theta.copy_(d["theta"].to(self.device, self.dtype))
            fans = self._phys.fans if self._phys is not None else None
            if fans is not None and "fan_thrust" in d and tuple(d["fan_thrust"].shape) == tuple(fans.thrust.shape):
                fans.thrust.copy_(d["fan_thrust"].to(self.device, self.dtype))
                fans.e_prev.copy_(d["fan_error"].to(self.device, self.dtype))
        if "re" in d:
            self.set_inflow(re=float(d["re"]))
        self.steps = int(d.get("steps", 0))
        hist = d.get("history")
        self._host_hist.clear()
        self._phist_host.clear()
        self._rows = self._flushed = 0
        if hist is not None and len(hist.get("t", [])):
            self._host_hist.append(np.stack([np.asarray(hist[k], dtype=np.float64) for k in HISTORY_KEYS], axis=1))
        with torch.no_grad():
            self._hist_n.zero_()


def _side_rules(cfg: SolverConfig3D, tc) -> tuple[list[str], list[float]]:
    """(kinds, values) of the six domain sides for the heat equation (see ThermalOps)."""
    sides = cfg.side_walls
    kinds, vals = [], []
    for side in WALL_SIDES:
        vel = sides[side]
        if vel == "periodic":
            kind, val = "periodic", 0.0
        elif vel == "inflow":
            kind, val = "inflow", tc.theta_inlet
        elif vel == "outflow":
            kind, val = "outflow", 0.0
        elif vel == "symmetry":
            kind, val = "adiabatic", 0.0
        else:
            kind, val = tc.wall(side)
        kinds.append(kind)
        vals.append(val)
    return kinds, vals


def _copy_thermal_geometry(dst, src) -> bool:
    """Copy a ThermalGeometry into another of identical structure in place (graphs kept)."""
    names = ("chi_t", "a_t", "w_t", "skin", "flux_rate", "vol_w", "wall_idx", "wall_w", "wall_seg")
    for n in names:
        a, b = getattr(dst, n), getattr(src, n)
        if (a is None) != (b is None) or (a is not None and a.shape != b.shape):
            return False
    if (dst.adia is None) != (src.adia is None):
        return False
    for n in names:
        a, b = getattr(dst, n), getattr(src, n)
        if a is not None:
            a.copy_(b)
    if dst.adia is not None:
        for a, b in zip(dst.adia, src.adia):
            if a.shape != b.shape:
                return False
            a.copy_(b)
    dst.values, dst.area, dst.modes = src.values, src.area, src.modes
    return True


__all__ = ["BYTES_PER_CELL_F32", "DIAG_KEYS", "FIELDS3D", "HISTORY_KEYS", "MEAN_FIELDS3D", "MEAN_FIELDS3D_THERMAL",
           "FlowSolver3D", "estimate_memory_bytes"]
