"""Live view of 3D runs: slice frames and the 3D explore session (DESIGN.md 7.7).

* :class:`FrameRenderer3D` -- turns a :class:`~aerolume.solvers.ns3d.FlowSolver3D`
  state into the same :class:`~aerolume.simulation.live.LiveFrame` the 2D runs
  produce, so the Simulate page shows a 3D run without knowing it is one: the
  image is a planar *slice* through the flow, chosen by ``ViewSettings.slice_axis``
  ("y": the side view, "z": the plan view, "x": the cross-flow view behind the
  body) and ``ViewSettings.slice_pos`` (metres, None = the plan's default). The
  bodies are the meshes' cross-sections in that plane. With a symmetry plane the
  half domain is mirrored, so the user sees the whole body.
* :class:`LiveSession3D` -- explore mode in 3D. ``LiveSession(design, sim)``
  returns one when ``design.dimension == 3``; it takes the 2D keys (alpha_deg,
  speed, ground_moving, paused, reset, step, steps_per_frame, nu, capture_trace,
  the placed-physics lists, field, cmap, ...) plus ``yaw_deg``, ``slice_axis``,
  ``slice_pos``, the 3D-view keys ``view3d``, ``smoke``, ``streamlines3d`` and
  ``seed_rake``, and ``obstacles`` as solid primitives (balls and boxes,
  :func:`apply_obstacles_3d`).
* **Placed physics** (DESIGN.md 10). Frames carry the temperature (°C) as a
  slice field and, with ``ViewSettings.temperature3d``, in the volume; every
  smoke channel (``dye`` .. ``dye3``, composited in their colours on a slice);
  the emitter tracer particles (``LiveFrame.particles`` (N, 3) metres, with their
  emitter ids, colours and streak lines); the diagnostics rows recorded since the
  previous frame; a captured solver step on request; and the heat-transfer and
  fan readouts in SI. The parts that do not depend on the dimension are the 2D
  renderer's (:class:`~aerolume.simulation.live.FrameRenderer`), so both say the
  same thing.
* **Flow in the volume** (DESIGN.md 7.13). With ``ViewSettings.view3d`` the
  frames also carry a down-sampled volume (``LiveFrame.volume``: the smoke and,
  for streamlines, the velocity) and streamlines traced on this thread from the
  rake (``LiveFrame.streamlines3d``), so the GUI thread only uploads them. Both
  are throttled to a small share of the run's wall time (volumes at most every
  0.1 s and at most 10 % of the time; streamlines about once a second and at
  most 10 %), so a production run slows by little and the viewer never builds a
  backlog. ``ViewSettings.seed_rake`` (a dict in metres, keys in
  :mod:`aerolume.viz3d.flow`) is where the solver releases its smoke
  (:func:`dye_rake_for`) and where the streamlines start; moving it moves both.

The threading model is the 2D one: :meth:`LiveSession3D.post` may be called from
any thread; :meth:`LiveSession3D.tick` runs on one worker thread and is the only
code that touches the solver.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, NamedTuple

import numpy as np

from ..model import Design, Mesh3D, Simulation, VolumeData
from .live import (
    HISTORY_POINTS,
    LIVE_HISTORY_T,
    MAX_STEPS_PER_FRAME,
    PLACEMENT_KEYS,
    STREAMLINE_PERIOD_S,
    FrameRenderer,
    LiveFrame,
    LiveSession,
    ViewSettings,
    cpu_threads,
    downsample_history,
)
from .plan3d import LIVE_PITCH3, LIVE_PITCH_GROUND3, LIVE_YAW3, SimPlan3D, plan_live3d, wind_frame_meshes

__all__ = ["LIVE_VOLUME_MAX_BYTES", "LIVE_VOLUME_MAX_CELLS", "VIEW_FIELDS3D", "FieldMeta3D", "FrameRenderer3D",
           "LiveSession3D", "apply_obstacles_3d", "dye_rake_for", "field_meta3d", "mesh_section",
           "resolve_seed_rake", "slice_extent"]

#: Fields a 3D slice can show (FlowSolver3D.field names; "dye" is the smoke).
VIEW_FIELDS3D = ("speed", "u", "v", "w", "cp", "p", "vorticity", "q", "nut", "mask", "dye")
#: ... and with placed physics (DESIGN.md 10): the temperature (°C), the other smoke channels and
#: the continuity residual (offered when the run carries them, FrameRenderer3D.view_fields)
PHYSICS_VIEW_FIELDS3D = ("temperature", "dye1", "dye2", "dye3", "divergence")
_SMOKE_FIELDS = ("dye", "dye1", "dye2", "dye3")
#: Live volumes hold at most this many cells (128^3) ...
LIVE_VOLUME_MAX_CELLS = 128 ** 3
#: ... and at most this many bytes (all fields and the mask, float16)
LIVE_VOLUME_MAX_BYTES = 20 * 2 ** 20
#: shortest interval between two live volumes [s], and the largest share of wall time spent taking them
VOLUME_MIN_PERIOD_S = 0.1
VOLUME_DUTY = 0.10
#: shortest interval between two live 3D streamline tracings [s], and their largest share of wall time
STREAMLINE3D_PERIOD_S = 1.0
STREAMLINE3D_DUTY = 0.10
#: Slice axes and their (column, row) world axes (SliceData orientation).
SLICE_AXES = {"y": (0, 2), "z": (0, 1), "x": (1, 2)}
_AXIS = {"x": 0, "y": 1, "z": 2}
#: Fields that change sign under the mirror y -> -y.
Y_ODD = frozenset({"v"})


class FieldMeta3D(NamedTuple):
    """How a 3D field is shown: label, SI unit, diverging colour scale, default colour map."""
    label: str
    unit: str
    diverging: bool
    cmap: str
    kind: str                         # "speed" | "pressure" | "rate" | "rate2" | "viscosity" | "none"

    def scale(self, speed: float, density: float, length: float) -> float:
        """Factor turning the solver's non-dimensional value into SI."""
        if self.kind == "speed":
            return float(speed)
        if self.kind == "pressure":
            return float(density) * float(speed) ** 2
        if self.kind == "rate":
            return float(speed) / float(length)
        if self.kind == "rate2":
            return (float(speed) / float(length)) ** 2
        if self.kind == "viscosity":
            return float(speed) * float(length)
        return 1.0


_META = {
    "speed": FieldMeta3D("Speed |V|", "m/s", False, "viridis", "speed"),
    "u": FieldMeta3D("Streamwise velocity u", "m/s", False, "viridis", "speed"),
    "v": FieldMeta3D("Spanwise velocity v", "m/s", True, "coolwarm", "speed"),
    "w": FieldMeta3D("Vertical velocity w", "m/s", True, "coolwarm", "speed"),
    "p": FieldMeta3D("Pressure (gauge)", "Pa", True, "coolwarm", "pressure"),
    "cp": FieldMeta3D("Pressure coefficient Cp", "", True, "coolwarm", "none"),
    # 3D vorticity is a magnitude (non-negative), so a sequential map
    "vorticity": FieldMeta3D("Vorticity |ω|", "1/s", False, "magma", "rate"),
    "q": FieldMeta3D("Q-criterion (vortex cores > 0)", "1/s²", True, "RdBu_r", "rate2"),
    "nut": FieldMeta3D("Eddy viscosity νt", "m²/s", False, "magma", "viscosity"),
    "mask": FieldMeta3D("Solid fraction", "", False, "grayscale", "none"),
    # smoke on a slice reads like the 2D smoke view: light streaks on a dark field
    "dye": FieldMeta3D("Smoke", "", False, "grayscale", "none"),
    "dye1": FieldMeta3D("Smoke, channel 2", "", False, "grayscale", "none"),
    "dye2": FieldMeta3D("Smoke, channel 3", "", False, "grayscale", "none"),
    "dye3": FieldMeta3D("Smoke, channel 4", "", False, "grayscale", "none"),
    # the solver's theta shown in °C: FrameRenderer3D converts, so the factor here is 1
    "temperature": FieldMeta3D("Temperature T", "°C", False, "thermal", "none"),
    "divergence": FieldMeta3D("Divergence ∇·u", "1/s", True, "RdBu_r", "rate"),
}


def field_meta3d(name: str) -> FieldMeta3D:
    """Presentation of a 3D field (``*_rms`` fields share their base field's units)."""
    base = name.removesuffix("_rms")
    m = _META.get(base) or FieldMeta3D(name, "", False, "viridis", "none")
    if name.endswith("_rms"):
        return m._replace(label=f"{m.label} (rms)", diverging=False, cmap="magma")
    return m


def slice_extent(axis: str, extent3: tuple[float, ...]) -> tuple[float, float, float, float]:
    """(col_min, col_max, row_min, row_max) of a slice normal to ``axis`` through ``extent3``."""
    ci, ri = SLICE_AXES[axis]
    return (float(extent3[2 * ci]), float(extent3[2 * ci + 1]), float(extent3[2 * ri]), float(extent3[2 * ri + 1]))


def mirror_slice(a: Any, axis: str, name: str) -> Any:
    """A half-domain slice (y >= 0) completed by its mirror image (numpy or torch, SliceData order).

    "z" slices have rows along y, "x" slices columns along y; "y" slices need no
    mirror. ``v`` changes sign under the reflection, every other field does not.
    """
    if axis == "y":
        return a
    sgn = -1.0 if name in Y_ODD else 1.0
    dim = 0 if axis == "z" else 1
    if hasattr(a, "flip"):                                   # torch
        import torch
        return torch.cat([sgn * a.flip(dim), a], dim=dim)
    return np.concatenate([sgn * np.flip(a, axis=dim), a], axis=dim)


def mesh_section(meshes: list[tuple[np.ndarray, np.ndarray]], axis: str, position: float) -> list[np.ndarray]:
    """Cross-section polylines of triangle meshes with the plane ``axis = position``.

    Returns (N, 2) arrays in slice coordinates (SliceData column, row). Each
    triangle that straddles the plane contributes one segment between the
    crossing points of its two cut edges; segments are chained through those
    edges (a cut edge is shared by its two triangles), so a closed manifold gives
    closed loops and an open sheet open chains. Vertices exactly on the plane
    are nudged to one side, so every crossing is a proper one.
    """
    k = _AXIS[axis]
    ci, ri = SLICE_AXES[axis]
    keys_a: list[np.ndarray] = []
    keys_b: list[np.ndarray] = []
    pts_a: list[np.ndarray] = []
    pts_b: list[np.ndarray] = []
    offset = 0
    for V, F in meshes:
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
        nv = len(V)
        if nv == 0 or len(F) == 0:
            continue
        span = float(np.ptp(V, axis=0).max()) or 1.0
        d = V[:, k] - float(position)
        d = np.where(np.abs(d) < 1e-12 * span, 1e-12 * span, d)
        s = d > 0
        sf = s[F]
        cut = ~(sf.all(axis=1) | (~sf).all(axis=1))
        T = F[cut]
        if not len(T):
            offset += nv * nv
            continue
        edges = ((0, 1), (1, 2), (2, 0))
        ch = np.stack([s[T[:, a]] != s[T[:, b]] for a, b in edges], axis=1)        # (K, 3), two True per row
        P = np.empty((len(T), 3, 2))
        K = np.empty((len(T), 3), dtype=np.int64)
        for e, (a, b) in enumerate(edges):
            ia, ib = T[:, a], T[:, b]
            with np.errstate(divide="ignore", invalid="ignore"):     # uncut edges: discarded below
                t = d[ia] / (d[ia] - d[ib])
                p = V[ia] + t[:, None] * (V[ib] - V[ia])
            P[:, e] = p[:, [ci, ri]]
            lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
            K[:, e] = offset + lo * nv + hi
        sel = np.argsort(~ch, axis=1, kind="stable")[:, :2]
        rows = np.arange(len(T))
        keys_a.append(K[rows, sel[:, 0]])
        keys_b.append(K[rows, sel[:, 1]])
        pts_a.append(P[rows, sel[:, 0]])
        pts_b.append(P[rows, sel[:, 1]])
        offset += nv * nv
    if not keys_a:
        return []
    ka, kb = np.concatenate(keys_a), np.concatenate(keys_b)
    pa, pb = np.concatenate(pts_a), np.concatenate(pts_b)
    adj: dict[int, list[int]] = {}
    point: dict[int, np.ndarray] = {}
    for i, (a, b) in enumerate(zip(ka.tolist(), kb.tolist())):
        adj.setdefault(a, []).append(i)
        adj.setdefault(b, []).append(i)
        point[a] = pa[i]
        point[b] = pb[i]
    used = np.zeros(len(ka), dtype=bool)
    kal, kbl = ka.tolist(), kb.tolist()
    lines: list[np.ndarray] = []
    starts = [key for key, segs in adj.items() if len(segs) == 1] + list(adj)
    for k0 in starts:
        while True:
            free = [s for s in adj[k0] if not used[s]]
            if not free:
                break
            line = [point[k0]]
            key, seg = k0, free[0]
            while seg is not None:
                used[seg] = True
                key = kbl[seg] if kal[seg] == key else kal[seg]
                line.append(point[key])
                nxt = [s for s in adj[key] if not used[s]]
                seg = nxt[0] if nxt else None
            if len(line) >= 2:
                lines.append(np.asarray(line))
    return lines


# ---------------------------------------------------------------------------
# the rake: smoke nozzles in the solver, seeds of the streamlines
# ---------------------------------------------------------------------------

def resolve_seed_rake(plan: SimPlan3D, spec: dict | None, *, purpose: str = "smoke"):
    """The rake dict (``ViewSettings.seed_rake``, metres) resolved for this plan's body and
    domain (:func:`aerolume.viz3d.flow.resolve_rake`); the full (mirrored) domain, so a
    rake on a symmetric body is drawn and specified across the whole span."""
    from ..viz3d.flow import resolve_rake
    return resolve_rake(spec, plan.body_bounds, plan.extent3_full, purpose=purpose,
                        h=float(plan.solver_config.h) * float(plan.length_ref))


def dye_rake_for(plan: SimPlan3D, spec: dict | None):
    """(DyeRake in solver units, the resolved SeedRake in metres) for a rake dict.

    Metres become solver units (divided by the reference length; the solver grid
    shares the wind frame's origin) and pulse seconds convective times (times
    U / L). A half-domain run keeps the nozzles on its side of the symmetry
    plane; the solver's mirror boundary supplies the others, and completes a
    nozzle centred on the plane.
    """
    from ..solvers.ns3d import DyeRake
    sr = resolve_seed_rake(plan, spec, purpose="smoke")
    L, U = float(plan.length_ref), float(plan.speed)
    if sr.kind == "plane":
        y0 = max(sr.y[0], 0.0) if plan.symmetry_plane else sr.y[0]
        pos = np.array([[sr.x, y0, sr.z[0]], [sr.x, sr.y[1], sr.z[1]]])
        kind = "plane"
    else:
        pos = np.asarray(sr.points, dtype=np.float64)
        if plan.symmetry_plane:
            pos = pos[pos[:, 1] >= -1e-9 * max(1.0, abs(plan.extent3[3]))]
        kind = "grid" if sr.kind == "grid" else "rake"
    on, off = sr.pulse if sr.pulse is not None else (0.0, 0.0)
    rake = DyeRake(kind, pos, radius=float(sr.radius or 0.0), pulse_on=float(on), pulse_off=float(off))
    return rake.scaled(1.0 / L, U / L), sr


def _rake_key(spec) -> str:
    import json
    try:
        return json.dumps(spec, sort_keys=True, default=lambda o: np.asarray(o).tolist())
    except (TypeError, ValueError):
        return repr(spec)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

class FrameRenderer3D:
    """Renders slices of a FlowSolver3D state into :class:`LiveFrame` objects.

    Keeps what must persist between frames: the smoothed colour range (per field,
    axis and range mode, so colours do not flicker), the cached solid mask and
    mesh cross-sections of the current slice, the last streamlines.
    """

    def __init__(self, plan: SimPlan3D):
        self.plan = plan
        self.length, self.speed, self.density = float(plan.length_ref), float(plan.speed), float(plan.fluid.density)
        cfg = plan.solver_config
        self.h = float(cfg.h)                    # solver units
        self.symmetry = bool(plan.symmetry_plane)
        self.extent3 = tuple(float(e) for e in plan.extent3)              # solved domain, metres
        self.meshes = [(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64))
                       for m in plan.bodies_wind3]
        self._range: tuple[float, float] | None = None
        self._range_key: tuple | None = None
        self._mask_key: tuple | None = None
        self._mask = None
        self._sections: dict[tuple, list[np.ndarray]] = {}
        self._stream: list[np.ndarray] | None = None
        self._stream_wall = -math.inf
        self._stream_key: tuple | None = None
        # flow in the volume (DESIGN.md 7.13)
        self._rake_key: str | None = None
        self._rake = None                          # viz3d.flow.SeedRake (metres)
        self._seeds: np.ndarray | None = None      # streamline seeds of the rake
        self._vol: VolumeData | None = None
        self._vol_key: tuple | None = None
        self._vol_serial = 0
        self._vol_wall = -math.inf
        self._vol_cost = 0.0
        self._vol_mask: tuple[int, np.ndarray] | None = None
        self._s3d = None
        self._s3d_wall = -math.inf
        self._s3d_cost = 0.0
        self._s3d_serial = -1
        # placed physics (DESIGN.md 10): the temperature scale, the smoke and particle colours
        self.physics = getattr(plan, "physics", None)
        self.placed = getattr(plan, "placed", None)
        self._diag_seen = -math.inf
        self._readout: tuple[float, dict] | None = None

    # the parts of a frame that do not depend on the dimension are the 2D renderer's (they read
    # self.plan / self.placed / self.physics), so 2D and 3D frames carry the same readouts
    new_diagnostics = FrameRenderer.new_diagnostics
    take_trace = FrameRenderer.take_trace
    readouts = FrameRenderer.readouts
    invalidate_readouts = FrameRenderer.invalidate_readouts

    def view_fields(self) -> tuple[str, ...]:
        """The fields this run can show on a slice: the flow and, with placed physics, the
        temperature, the smoke channels it carries and the continuity residual."""
        extra: tuple[str, ...] = ("divergence",)
        if self.physics is not None and getattr(self.physics, "thermal", False):
            extra += ("temperature",)
        n = len(getattr(self.placed, "dye_channels", None) or [])
        return VIEW_FIELDS3D + extra + tuple(f"dye{k}" for k in range(1, min(n, 4)))

    def rake_smoke(self, solver) -> bool:
        """Whether the solver's smoke leaves from the seed rake (the v0.2 rake): not once placed
        emitters replace it (the solver's live configuration says), and never in a closed box."""
        return getattr(getattr(solver, "cfg", None), "emitters", None) is None and not getattr(self.plan, "closed",
                                                                                               False)

    # ----------------------------------------------------------------- state
    def invalidate_geometry(self, meshes: list[Mesh3D] | None = None) -> None:
        """Call after the bodies changed (new masks, sections and streamlines)."""
        if meshes is not None:
            self.meshes = [(np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64))
                           for m in meshes]
        self._mask, self._mask_key = None, None
        self._sections.clear()
        self._stream = None
        self._stream_wall = -math.inf
        self._vol_mask = None
        self._s3d, self._s3d_serial = None, -1
        self._vol_wall = -math.inf

    def reset(self) -> None:
        """Forget all inter-frame state (after a solver reset)."""
        self.invalidate_geometry()
        self._range = None
        self._range_key = None
        self._vol, self._vol_key = None, None
        self._diag_seen = -math.inf
        self._readout = None

    # ------------------------------------------------------------ the rake
    def apply_view(self, solver, view: ViewSettings) -> None:
        """Bring the solver's smoke nozzles to ``view.seed_rake`` when it changed (call on
        the solver thread; cheap when nothing changed). An invalid rake keeps the old one."""
        key = _rake_key(getattr(view, "seed_rake", None))
        if key == self._rake_key and self._rake is not None:
            return
        try:
            dye, sr = dye_rake_for(self.plan, getattr(view, "seed_rake", None))
            seeds = resolve_seed_rake(self.plan, getattr(view, "seed_rake", None), purpose="streamlines").points
        except (ValueError, TypeError):
            if self._rake is not None:
                self._rake_key = key            # do not retry the same bad dict every frame
                return
            dye, sr = dye_rake_for(self.plan, None)
            seeds = resolve_seed_rake(self.plan, None, purpose="streamlines").points
        if getattr(solver, "dye_enabled", False) and self.rake_smoke(solver):
            solver.set_dye_rake(dye)
        self._rake, self._seeds, self._rake_key = sr, seeds, key
        self._s3d, self._s3d_serial = None, -1

    @property
    def seed_rake(self):
        """The resolved rake (viz3d.flow.SeedRake, metres) of the last applied view."""
        return self._rake

    # ---------------------------------------------------------- the volume
    def volume_factor(self, n_fields: int) -> int:
        """Block size of a live volume with ``n_fields`` arrays (plus the mask)."""
        nx, ny, nz = self.plan.grid3
        f = 1
        while True:
            cells = (nx // f) * (ny // f) * (nz // f)
            if cells <= LIVE_VOLUME_MAX_CELLS and cells * (n_fields + 1) * 2 <= LIVE_VOLUME_MAX_BYTES:
                return f
            f += 1

    def volume_block(self, solver, fields: tuple[str, ...]) -> VolumeData:
        """A down-sampled copy of the current flow: SI units, block averages of whole cells (the
        far ends cropped), the solid fraction as the mask. Float16 except the temperature, which
        is float32: in °C, a plume a tenth of a kelvin warmer than a 15 °C ambient (the Ra 10^4
        cavity) spans a few float16 steps there (2^-6 K), so it would show as a few bands."""
        import torch
        import torch.nn.functional as F
        # the float32 temperature costs two float16 fields of the byte budget
        f = self.volume_factor(len(fields) + (1 if "temperature" in fields else 0))
        nx, ny, nz = self.plan.grid3

        def pooled(t, dtype=torch.float16) -> np.ndarray:
            if f > 1:
                t = t[: (nz // f) * f, : (ny // f) * f, : (nx // f) * f]
                t = F.avg_pool3d(t[None, None].float(), f, stride=f)[0, 0]
            return t.to(dtype).cpu().numpy()

        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for name in fields:
                t = solver.field(name)
                if name in ("u", "v", "w"):
                    t = t * self.speed
                elif name == "temperature" and self.physics is not None:
                    out[name] = pooled(self.physics.celsius(t.double()), torch.float32)   # °C, as stored volumes
                    continue
                out[name] = pooled(t)
            if self._vol_mask is None or self._vol_mask[0] != f:
                self._vol_mask = (f, pooled(solver.field("mask")))
        L = self.length
        h_m = self.h * L
        x0, _, y0, _, z0, _ = self.extent3
        return VolumeData(origin=(float(x0), float(y0), float(z0)), spacing=(f * h_m, f * h_m, f * h_m),
                          fields=out, mask=self._vol_mask[1], label="instant", mirrored=self.symmetry)

    def _live_volume(self, solver, view: ViewSettings, *, force: bool):
        smoke = getattr(view, "smoke", True) and getattr(solver, "dye_enabled", False)
        n_dye = max(1, min(int(getattr(solver, "dye_channels", 1) or 1), len(_SMOKE_FIELDS)))
        heat = (getattr(view, "temperature3d", True) and getattr(solver, "thermal_enabled", False)
                and self.physics is not None)
        fields = (_SMOKE_FIELDS[:n_dye] if smoke else ()) + \
                 (("u", "v", "w") if getattr(view, "streamlines3d", False) else ()) + \
                 (("temperature",) if heat else ())
        if not fields:
            self._vol, self._vol_key = None, None
            return None
        now = time.perf_counter()
        period = max(VOLUME_MIN_PERIOD_S, 1.0 / max(float(getattr(view, "fps", 20.0) or 20.0), 1.0),
                     self._vol_cost / VOLUME_DUTY)
        if not force and self._vol is not None and fields == self._vol_key and now - self._vol_wall < period:
            return self._vol
        try:
            vol = self.volume_block(solver, fields)
        except Exception:  # noqa: BLE001 - a viewer extra must never stop the run
            return self._vol
        el = time.perf_counter() - now
        self._vol_cost = el if self._vol is None else 0.7 * self._vol_cost + 0.3 * el
        self._vol, self._vol_key, self._vol_wall = vol, fields, time.perf_counter()
        self._vol_serial += 1
        return vol

    def _live_streamlines3d(self, view: ViewSettings, *, force: bool):
        vol = self._vol
        if vol is None or not all(k in vol.fields for k in ("u", "v", "w")) or self._seeds is None:
            return None
        now = time.perf_counter()
        period = max(STREAMLINE3D_PERIOD_S, self._s3d_cost / STREAMLINE3D_DUTY)
        if self._s3d is not None and (self._s3d_serial == self._vol_serial or (not force and now - self._s3d_wall
                                                                               < period)):
            return self._s3d
        try:
            from ..viz3d.flow import trace_streamlines
            s3d = trace_streamlines(vol, self._seeds, direction="forward")
        except Exception:  # noqa: BLE001 - a viewer extra must never stop the run
            return self._s3d
        el = time.perf_counter() - now
        self._s3d_cost = el if self._s3d is None else 0.7 * self._s3d_cost + 0.3 * el
        self._s3d, self._s3d_wall, self._s3d_serial = s3d, time.perf_counter(), self._vol_serial
        return s3d

    # --------------------------------------------------------------- slicing
    def slice_axis(self, view: ViewSettings) -> str:
        axis = str(getattr(view, "slice_axis", "y") or "y").lower()
        return axis if axis in SLICE_AXES else "y"

    def slice_position(self, axis: str, view: ViewSettings | None = None) -> float:
        """The slice position [m] for ``axis``: the view's, else the plan's default, inside the domain."""
        pos = getattr(view, "slice_pos", None) if view is not None else None
        if pos is None or not math.isfinite(float(pos)):
            pos = self.plan.slice_positions.get(axis, 0.0)
        pos = float(pos)
        lo, hi = self.full_extent3()[2 * _AXIS[axis]: 2 * _AXIS[axis] + 2]
        return min(max(pos, lo), hi)

    def full_extent3(self) -> tuple[float, float, float, float, float, float]:
        x0, x1, _, y1, z0, z1 = self.extent3
        return (x0, x1, -y1, y1, z0, z1) if self.symmetry else self.extent3

    def frame_extent(self, axis: str) -> tuple[float, float, float, float]:
        """Extent of a (mirrored, when a symmetry plane is used) slice image, metres."""
        return slice_extent(axis, self.full_extent3())

    def cut(self, solver, axis: str, position: float, name: str, *, mean: bool = False):
        """Slice tensor (solver units) of ``name`` at ``position`` [m], mirrored to the full width.

        With a symmetry plane a y-slice at y < 0 is the mirror image of the one at
        -y (``v`` changes sign), and z / x slices are completed by their mirror.
        """
        L = self.length
        pos, sgn = position, 1.0
        if self.symmetry and axis == "y" and position < 0:
            pos, sgn = -position, (-1.0 if name in Y_ODD else 1.0)
        f = solver.slice(axis, pos / L, name, mean=mean)
        if sgn != 1.0:
            f = sgn * f
        if self.symmetry:
            f = mirror_slice(f, axis, name)
        return f

    def _mask_for(self, solver, axis: str, position: float):
        key = (axis, round(position, 9))
        if self._mask is None or self._mask_key != key:
            self._mask = self.cut(solver, axis, position, "mask")
            self._mask_key = key
        return self._mask

    def sections(self, axis: str, position: float) -> list[np.ndarray]:
        key = (axis, round(position, 9))
        if key not in self._sections:
            try:
                self._sections[key] = mesh_section(self.meshes, axis, position)
            except Exception:  # noqa: BLE001 - an outline overlay must never stop the run
                self._sections[key] = []
            while len(self._sections) > 8:
                self._sections.pop(next(iter(self._sections)))
        return self._sections[key]

    def focus(self, axis: str) -> tuple[float, float, float, float]:
        return self.plan.focus_window(axis)

    # --------------------------------------------------------------- render
    def render(self, solver, view: ViewSettings, *, phase: str, progress: float, eta_s: float | None,
               steps_per_s: float, history: dict[str, np.ndarray] | None = None,
               coeffs: dict[str, float] | None = None) -> LiveFrame:
        """Render the solver's current state as a slice frame (plus, with ``view.view3d``, the
        volume, the 3D streamlines and the rake for the 3D view)."""
        from ..viz import auto_range, render_field
        self.apply_view(solver, view)
        axis = self.slice_axis(view)
        pos = self.slice_position(axis, view)
        # unknown names (the 2D "theta") fall back to speed; "nut" of a laminar run is honestly zero
        name = view.field if view.field in self.view_fields() else "speed"
        f = self.cut(solver, axis, pos, name)
        if name == "temperature":
            f = self.physics.celsius(f)
        mask = self._mask_for(solver, axis, pos)
        meta = field_meta3d(name)
        scale = meta.scale(self.speed, self.density, self.length) or 1.0
        cmap = view.cmap or meta.cmap
        key = (name, axis, view.range_mode)
        chans = list(getattr(self.placed, "dye_channels", None) or [])[:len(_SMOKE_FIELDS)]
        composite = name == "dye" and len(chans) > 1
        if view.range_mode == "fixed" and view.vmin is not None and view.vmax is not None and view.vmax > view.vmin:
            lo, hi = float(view.vmin) / scale, float(view.vmax) / scale
        elif name == "mask" or name in _SMOKE_FIELDS:
            lo, hi = 0.0, 1.0
        else:
            prev = self._range if self._range_key == key else None
            lo, hi = auto_range(f, diverging=meta.diverging, mask=mask, prev=prev, smoothing=0.8)
        self._range, self._range_key = (lo, hi), key
        if composite:
            # every smoke colour over the dark background (DESIGN.md 10.1), one channel per colour
            from ..viz import render_dye
            rgba = render_dye([self.cut(solver, axis, pos, ch["field"]) for ch in chans],
                              [ch["color"] for ch in chans], mask=mask)
        else:
            rgba = render_field(f, cmap, lo, hi, mask=None if name == "mask" else mask)
        stream = self._streamlines(solver, axis, pos, mask) if view.streamlines else None
        if coeffs is None:
            c = solver.coefficients()
            coeffs = {k: float(v) for k, v in zip(("cd", "cs", "cl", "c_roll", "cm", "cn"), c)}
        hist = downsample_history(history if history is not None else solver.history(), HISTORY_POINTS)
        vol = s3d = None
        if getattr(view, "view3d", False):
            force = phase == "done"
            vol = self._live_volume(solver, view, force=force)
            if getattr(view, "streamlines3d", False):
                s3d = self._live_streamlines3d(view, force=force)
        parts = self._emitter_particles(solver) if getattr(view, "particles3d", True) else None
        return LiveFrame(rgba=rgba, extent=self.frame_extent(axis), field=name, cmap=cmap, vmin=lo * scale,
                         vmax=hi * scale, units=meta.unit, t=float(solver.t), step=int(solver.steps),
                         steps_per_s=float(steps_per_s), coeffs=coeffs, history=hist, progress=float(progress),
                         eta_s=eta_s, phase=phase, bodies=self.sections(axis, pos),
                         particles=parts[0] if parts is not None else None,
                         streamlines=stream, focus=self.focus(axis), slice_axis=axis, slice_pos=pos,
                         volume=vol, volume_serial=self._vol_serial if vol is not None else 0, streamlines3d=s3d,
                         seed_rake=self._rake.spec() if self._rake is not None else None,
                         dye_channels=chans if composite else None,
                         particle_ids=parts[1] if parts is not None else None,
                         particle_palette=list(self.placed.particle_palette) if parts is not None else None,
                         particle_trails=parts[2] if parts is not None else None,
                         diagnostics=self.new_diagnostics(solver), trace=self.take_trace(solver, view),
                         physics=self.readouts(solver))

    def _emitter_particles(self, solver):
        """((N, 3) positions [m], emitter ids, streak lines [m]) of the emitter tracer particles, or
        None without particle emitters. A half-domain run is mirrored like its slices."""
        if self.placed is None or not self.placed.particle_palette or not hasattr(solver, "particles"):
            return None
        try:
            p = solver.particles()
            L = self.length
            pts = np.asarray(p["positions"], dtype=np.float64).reshape(-1, 3) * L
            ids = np.asarray(p["emitter"], dtype=np.int16).reshape(-1)
            lines = [np.asarray(ln, dtype=np.float64) * L for ln in solver.particle_lines()]
            if self.symmetry and len(pts):
                flip = np.array([1.0, -1.0, 1.0])
                pts = np.vstack([pts, pts * flip])
                ids = np.concatenate([ids, ids])
                lines = lines + [ln * flip for ln in lines]
            return pts, ids, lines
        except Exception:  # noqa: BLE001 - an overlay must never stop the run
            return None

    def _streamlines(self, solver, axis: str, pos: float, mask) -> list[np.ndarray]:
        """Streamlines of the in-plane velocity (u, w on a side view; u, v on a plan view; v, w behind)."""
        from ..viz import streamlines
        now = time.perf_counter()
        key = (round(float(solver.t), 3), axis, round(pos, 9))
        if (self._stream is not None and self._stream_key is not None and self._stream_key[1:] == key[1:]
                and (now - self._stream_wall < STREAMLINE_PERIOD_S or key == self._stream_key)):
            return self._stream
        comps = {"y": ("u", "w"), "z": ("u", "v"), "x": ("v", "w")}[axis]
        try:
            a = self.cut(solver, axis, pos, comps[0]).float().cpu().numpy()
            b = self.cut(solver, axis, pos, comps[1]).float().cpu().numpy()
            m = mask.float().cpu().numpy()
            self._stream = streamlines(a, b, self.frame_extent(axis), density=1.0, mask=m)
        except Exception:  # noqa: BLE001 - an overlay must never stop the run
            self._stream = []
        self._stream_wall, self._stream_key = time.perf_counter(), key
        return self._stream


# ---------------------------------------------------------------------------
# explore mode
# ---------------------------------------------------------------------------

#: View keys of a 3D session: applied to the shared ViewSettings at once (DESIGN.md 7.7, 7.13, 10.4).
#: ``temperature3d`` / ``particles3d`` say whether the 3D view wants the temperature in its volumes
#: and the tracer particles in its frames.
_VIEW_KEYS3 = {"field", "cmap", "particles", "streamlines", "range_mode", "vmin", "vmax", "fps", "slice_axis",
               "slice_pos", "view3d", "smoke", "streamlines3d", "seed_rake", "capture_trace", "temperature3d",
               "particles3d"}
#: Every key a 3D session accepts: the view keys, the 2D session's flow and placed-physics keys, and yaw.
_POST_KEYS3 = ({"alpha_deg", "yaw_deg", "speed", "ground_moving", "paused", "reset", "nu", "step", "steps_per_frame",
                "obstacles"} | _VIEW_KEYS3 | PLACEMENT_KEYS)


def _trim_history3(solver, keep: float) -> dict[str, np.ndarray]:
    """The force history without samples older than ``keep`` convective times (see live._trim_history).

    Explore sessions run for hours, and every frame copies the force history, the placed-physics
    history (the readouts) and the new diagnostics rows, so all three host-side logs are trimmed
    together: through the solver's ``trim_history`` when it has one, else through its private lists
    (the force rows, the physics rows by their own time column, and the diagnostics rows together
    with their step list, which :meth:`FlowSolver3D.diagnostics` pairs by position).
    """
    h = solver.history()
    t = h["t"]
    if t.size < 2 or t[-1] - t[0] <= 1.5 * keep:
        return h
    if hasattr(solver, "trim_history"):
        solver.trim_history(keep)
        return solver.history()
    t_cut = float(t[-1]) - keep
    sel = t >= t_cut
    h = {k: v[sel] for k, v in h.items()}
    if isinstance(getattr(solver, "_host_hist", None), list):
        from ..solvers.ns3d.solver import HISTORY_KEYS
        solver._host_hist[:] = [np.stack([h[k] for k in HISTORY_KEYS], axis=1)]
    ph = getattr(solver, "_phist_host", None)
    if isinstance(ph, list) and ph:
        d = np.concatenate(ph)
        ph[:] = [d[d[:, 0] >= t_cut]]
    dh, ds = getattr(solver, "_diag_host", None), getattr(solver, "_diag_steps", None)
    if isinstance(dh, list) and isinstance(ds, list) and dh:
        d = np.concatenate(dh)
        if len(d) == len(ds) and d.shape[1] >= 1:
            from ..solvers.ns3d.solver import DIAG_KEYS
            tcol = DIAG_KEYS.index("t") if "t" in DIAG_KEYS else 0
            k = int(np.argmax(d[:, tcol] >= t_cut)) if (d[:, tcol] >= t_cut).any() else len(d)
            dh[:] = [d[k:]]
            del ds[:k]
    return h


def apply_obstacles_3d(solver, payload: dict | None, *, length: float, applied: list | None = None) -> list:
    """Place (or erase) solid primitives in a 3D solver's paint layer (DESIGN.md 10.4: "3D: place
    primitives").

    ``payload`` is ``{"ops": [op, ...]}`` (or None to clear), each op ``{"kind": "sphere" | "box",
    "center": (x, y, z), "radius": r, "size": (sx, sy, sz), "erase": bool}`` in wind-frame metres
    (``radius`` for balls, ``size`` the box's full extents). The operations are replayed in order
    onto ``FlowSolver3D.paint_solid`` (penalised to rest with the bodies' calibrated wall, outside
    the differentiable geometry). As in 2D, an op list that extends the ``applied`` one paints only
    the new ops; anything else clears the layer and replays. Returns the list now applied.
    """
    ops = list((payload or {}).get("ops") or [])
    pending = (payload or {}).get("pending")
    if pending:
        ops.append(pending)
    L = float(length)
    applied = list(applied or [])
    keyed = [_op_key3(o) for o in ops]
    start = len(applied) if applied and keyed[:len(applied)] == applied else 0
    # check every new operation before painting any, so a bad one leaves the layer as it was
    strokes = []
    for op in ops[start:]:
        kind = str(op.get("kind", "sphere"))
        if kind not in ("sphere", "box"):
            raise ValueError(f"3D obstacles are balls ('sphere') or boxes ('box'), not {kind!r}")
        c = np.asarray(op.get("center", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(-1)
        if c.size != 3 or not np.all(np.isfinite(c)):
            raise ValueError("a 3D obstacle needs a finite centre (x, y, z) in metres")
        r = float(op.get("radius", 0.0) or 0.0)
        size = op.get("size")
        if kind == "sphere" and not (r > 0 and math.isfinite(r)):
            raise ValueError("a ball obstacle needs a positive radius")
        if kind == "box":
            s = np.asarray(size if size is not None else (2 * r,) * 3, dtype=np.float64).reshape(-1)
            if s.size != 3 or not np.all(np.isfinite(s) & (s > 0)):
                raise ValueError("a box obstacle needs three positive extents (size)")
            size = tuple(s / L)
        strokes.append((tuple(c / L), r / L, bool(op.get("erase", False)), kind, size))
    if start == 0 and hasattr(solver, "clear_paint"):
        solver.clear_paint()
    for center, radius, erase, kind, size in strokes:
        solver.paint_solid(center, radius, erase=erase, shape=kind, size=size)
    return keyed


def _op_key3(op: dict) -> tuple:
    def arr(v) -> bytes:
        return np.asarray(v if v is not None else (), dtype=np.float64).round(12).tobytes()
    return (str(op.get("kind")), bool(op.get("erase")), round(float(op.get("radius", 0.0) or 0.0), 12),
            arr(op.get("center")), arr(op.get("size")))


class LiveSession3D(LiveSession):
    """Real-time 3D explore mode (DESIGN.md 7.7, 10.4).

    The grid comes from :func:`aerolume.simulation.plan3d.plan_live3d`: a small,
    full-domain grid (a yaw change breaks any mirror symmetry) sized so every
    attitude the sliders reach fits, with the solver's mask window widened to
    cover them, so attitude edits rebuild only the masks and keep the CUDA
    graphs. Speed changes only change the Reynolds number; switching the road
    between moving and fixed rebuilds the solver with the current flow.

    Live editing is the 2D session's (:class:`~aerolume.simulation.live.LiveSession`):
    ``post(emitters=[...])`` (and ``fans`` / ``heat_sources`` / ``ports``, model
    objects in SI) rewrites the solver's sources in place; ``nu`` is the Reynolds
    lock; ``paused`` with ``step`` advances a given number of steps; ``steps_per_frame``
    is the frame skip; ``capture_trace`` puts one step's stages, cut on the slice
    shown, into the next frame; ``obstacles`` places balls and boxes
    (:func:`apply_obstacles_3d`).
    """

    POST_KEYS = frozenset(_POST_KEYS3)

    def __init__(self, design: Design, sim: Simulation, *, device: str | None = None,
                 view: ViewSettings | None = None, airfoil_coords=None):
        from ..solvers.ns3d import FlowSolver3D
        self.design = design
        self.sim = Simulation.from_dict(sim.to_dict())           # private copy: posts edit it
        self.view = view if view is not None else ViewSettings()
        self.plan: SimPlan3D = plan_live3d(design, self.sim, device=device, airfoil_coords=airfoil_coords)
        self._threads = cpu_threads(self.plan.device)
        self._threads.__enter__()
        try:
            self.solver = FlowSolver3D(self.plan.solver_config)
        except Exception:
            self._threads.__exit__(None, None, None)
            raise
        self.renderer = FrameRenderer3D(self.plan)
        self._init_common()
        self.yaw_deg = float(getattr(self.sim.conditions, "yaw_deg", 0.0) or 0.0)
        self.pitch_limit = LIVE_PITCH_GROUND3 if self.plan.ground else LIVE_PITCH3
        self.yaw_limit = LIVE_YAW3

    # ------------------------------------------------------------------ input
    def post(self, **changes) -> None:
        """Request parameter / view changes (thread-safe; applied on the next tick).

        Keys: those of :meth:`LiveSession.post` (``alpha_deg``, ``speed``, ``ground_moving``,
        ``paused``, ``reset``, ``nu``, ``step``, ``steps_per_frame``, ``emitters`` / ``fans`` /
        ``heat_sources`` / ``ports``, ``obstacles`` -- here a dict of solid primitives, see
        :func:`apply_obstacles_3d` -- and the view keys) plus ``yaw_deg`` (deg) and the 3D view
        keys ``slice_axis`` ("x" | "y" | "z"), ``slice_pos`` (m or None), ``view3d``, ``smoke``,
        ``streamlines3d``, ``temperature3d``, ``particles3d`` (bools) and ``seed_rake`` (a rake
        dict in metres or None). Bad values raise ValueError here, in the caller's thread.
        ``particles`` (the 2D view's passive tracers) is accepted and ignored: 3D tracers come
        from particle emitters.
        """
        unknown = set(changes) - _POST_KEYS3
        if unknown:
            raise TypeError(f"unknown live setting(s): {', '.join(sorted(unknown))}")
        self._check(changes)
        if "yaw_deg" in changes and not math.isfinite(float(changes["yaw_deg"])):
            raise ValueError("yaw_deg must be finite")
        if "slice_axis" in changes and str(changes["slice_axis"]).lower() not in SLICE_AXES:
            raise ValueError("slice_axis must be 'x', 'y' or 'z'")
        if changes.get("seed_rake") is not None:
            if not isinstance(changes["seed_rake"], dict):
                raise ValueError("seed_rake must be a dict (see aerolume.viz3d.flow) or None")
            resolve_seed_rake(self.plan, changes["seed_rake"])          # raises ValueError when invalid
        if changes.get("obstacles") is not None and not isinstance(changes["obstacles"], dict):
            raise ValueError("obstacles must be a dict {'ops': [...]} (see apply_obstacles_3d) or None")
        for k in _VIEW_KEYS3 & set(changes):
            v = changes[k]
            setattr(self.view, k, str(v).lower() if k == "slice_axis" else v)
        with self._lock:
            for k, v in changes.items():
                if k in _VIEW_KEYS3:
                    continue
                if k == "reset":
                    self._pending["reset"] = self._pending.get("reset", False) or bool(v)
                elif k == "step":
                    self._pending["step"] = min(self._pending.get("step", 0) + int(v), MAX_STEPS_PER_FRAME)
                else:
                    self._pending[k] = v

    # -------------------------------------------------------------- changes
    def _immersed(self, wind: list[Mesh3D], moving: bool | None = None) -> list:
        """Wind-frame meshes -> solver bodies; wheels roll with a moving belt; heated bodies keep
        their thermal condition (solver units, from the plan)."""
        from ..solvers.ns3d import ImmersedMesh
        L, U = self.plan.length_ref, self.plan.speed
        if moving is None:
            moving = bool(self.sim.ground.moving)
        spin = self.plan.ground and moving and bool(self.sim.ground.rotate_wheels)
        thermal = [(getattr(b, "thermal", "adiabatic"), float(getattr(b, "thermal_value", 0.0)))
                   for b in self.plan.solver_config.bodies]
        out = []
        for i, (m, planned) in enumerate(zip(wind, self.plan.bodies_wind3)):
            omega = planned.omega
            center = m.center if m.center is not None else planned.center
            axis = m.axis if m.axis is not None else planned.axis
            if m.role == "wheel" and self.plan.ground:
                from .plan3d import _wheel_frame
                c, ax, r = _wheel_frame(m)
                center, axis = c, ax
                axy2 = ax[0] ** 2 + ax[1] ** 2
                omega = (-U * ax[1] / (r * axy2)) if (spin and r > 0 and axy2 > 1e-6) else 0.0
            th_mode, th_val = thermal[i] if i < len(thermal) else ("adiabatic", 0.0)
            out.append(ImmersedMesh(vertices=np.asarray(m.vertices) / L, faces=np.asarray(m.faces), name=planned.name,
                                    role=m.role, omega=float(omega) * L / U,
                                    center=tuple(float(v) / L for v in (center or (0.0, 0.0, 0.0))),
                                    axis=tuple(axis or (0.0, 1.0, 0.0)), thermal=th_mode, thermal_value=th_val))
        return out

    def _wind(self, alpha: float, yaw: float) -> list[Mesh3D]:
        return wind_frame_meshes(self.plan.meshes_design, alpha, yaw, self.plan.moment_ref3, ground=self.plan.ground)

    def _valid_attitude(self, alpha: float, yaw: float) -> tuple[float, float]:
        """Clamp to the planned ranges and, on a road, to pitch angles that keep the body above it."""
        a = max(-self.pitch_limit, min(self.pitch_limit, float(alpha)))
        y = max(-self.yaw_limit, min(self.yaw_limit, float(yaw)))
        if not self.plan.ground or self.plan.closed:
            return a, y
        tol = 0.5 * self.plan.solver_config.h * self.plan.length_ref

        def ok(x: float) -> bool:
            low = min(float(np.asarray(m.vertices)[:, 2].min()) for m in self._wind(x, y) if m.role != "wheel") \
                if any(m.role != "wheel" for m in self.plan.meshes_design) else 1.0
            return low >= tol or x == 0.0
        if ok(a):
            return a, y
        lo, hi = 0.0, a
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo, y

    def _set_attitude(self, alpha: float, yaw: float) -> None:
        a, y = self._valid_attitude(alpha, yaw)
        if (abs(a - self.alpha_deg) < 1e-9 and abs(y - self.yaw_deg) < 1e-9) or not self.plan.meshes_design:
            return
        wind = self._wind(a, y)
        self.solver.set_bodies(self._immersed(wind))
        self.alpha_deg, self.yaw_deg = a, y
        self.sim.conditions.alpha_deg = a
        self.sim.conditions.yaw_deg = y
        self.renderer.invalidate_geometry(wind)
        self._physics_dirty = True               # body-frame placements turn with the body

    def _set_speed(self, speed: float) -> None:
        if self.plan.closed:
            self._notice("A closed box has no free stream: its reference velocity comes from what drives the flow, "
                         "so the speed slider does not apply.")
            return
        self.speed = float(speed)
        self.sim.conditions.speed = self.speed
        self.solver.set_inflow(re=self._reynolds())
        self.renderer.speed = self.speed
        self._timed = False                       # fused path: graphs are re-captured
        self._follow_units()                      # sources scaled by U are re-expressed

    def _set_nu(self, nu: float) -> None:
        super()._set_nu(nu)
        self._timed = False

    def _set_ground_moving(self, moving: bool) -> None:
        if not self.plan.ground or self.plan.closed or bool(moving) == bool(self.sim.ground.moving):
            return
        from ..solvers.ns3d import FlowSolver3D
        state = self.solver.state_dict()
        wind = self._wind(self.alpha_deg, self.yaw_deg)
        cfg = dataclasses.replace(self.solver.cfg, bc_zmin="moving" if moving else "noslip",
                                  bodies=self._immersed(wind, moving=bool(moving)))
        new = FlowSolver3D(cfg)                    # the old solver stays usable if this fails
        new.load_state_dict(state)
        self.solver = new
        self.sim.ground.moving = bool(moving)
        self.renderer.invalidate_geometry(wind)
        self._sec_per_step, self._timed = None, False
        self._obstacles_applied = []              # the new solver has no paint: replay the drawing
        if self._obstacles is not None:
            self._apply_obstacles(self._obstacles)

    # ------------------------------------------------------ placed physics
    def _placed_now(self):
        from .physics import placed_physics_3d
        return placed_physics_3d(self.design, self.sim, self._current_scales(), alpha_deg=self.alpha_deg,
                                 yaw_deg=self.yaw_deg, extent3=self.plan.extent3,
                                 h_m=self.plan.solver_config.h * self.plan.length_ref,
                                 bodies_wind=self.plan.bodies_wind3, closed=self.plan.closed)

    def _push_placed(self, placed) -> None:
        """Hand rebuilt specs to the 3D solver. Without emitters a tunnel goes back to the seed rake
        (``set_emitters(None)``, the planner's rule, :func:`.plan3d._solver_physics_3d`); a closed
        box has no rake, so it gets an empty emitter list instead (None would bring the automatic
        rake into the room)."""
        s = self.solver
        if s.cfg.thermal is None and placed.thermal is not None and self.plan.physics is not None:
            self._notice("Heat transfer was off when explore mode started; restart explore mode to add it.")
        emitters = list(placed.emitters)
        if any(e.species == "dye" for e in emitters) and not getattr(s, "dye_enabled", False):
            self._notice("Smoke was off when explore mode started (no smoke emitters); restart explore mode to "
                         "show it.")
        rake = not emitters and not self.plan.closed
        s.set_emitters(None if rake else emitters)
        s.set_sources(heat_sources=placed.heat_sources if s.cfg.thermal is not None else [], fans=placed.fans,
                      ports=placed.ports)
        if s.cfg.thermal is not None and placed.thermal is not None:
            ri = float(placed.thermal.richardson)
            if abs(ri - float(s.cfg.thermal.richardson)) > 1e-12 * max(1.0, abs(ri)):
                s.set_thermal(richardson=ri)       # re-captures the graphs: only when it changed
        if rake:
            self.renderer._rake_key = None         # the rake's nozzles go back in on the next frame
        self._timed = False

    def _apply_physics(self) -> None:
        if self.plan.physics is None and any(getattr(self.sim, k) for k in PLACEMENT_KEYS):
            try:
                from .physics import physics_scales
                self.plan.physics = physics_scales(self.design, self.sim, 3)
                self.renderer.physics = self.plan.physics
            except (ValueError, TypeError) as exc:
                self._notice(f"The placed physics could not be updated: {exc}")
                return
        super()._apply_physics()

    def _apply_obstacles(self, payload) -> None:
        self._obstacles = payload
        try:
            self._obstacles_applied = apply_obstacles_3d(self.solver, payload, length=self.plan.length_ref,
                                                         applied=self._obstacles_applied)
        except Exception as exc:  # noqa: BLE001 - a drawing must never stop the run
            self._notice(f"The obstacles could not be placed: {exc}")
        self.renderer.invalidate_geometry()
        self._timed = False

    def _apply(self, p: dict) -> None:
        self._physics_dirty = False
        if p.get("reset"):
            self.solver.reset()
            self.renderer.reset()
        if "paused" in p:
            self._paused = bool(p["paused"])
        if "steps_per_frame" in p:
            n = int(p["steps_per_frame"] or 0)
            self._steps_per_frame = n if n > 0 else None
        if "ground_moving" in p:
            self._set_ground_moving(bool(p["ground_moving"]))
        if "alpha_deg" in p or "yaw_deg" in p:
            self._set_attitude(float(p.get("alpha_deg", self.alpha_deg)), float(p.get("yaw_deg", self.yaw_deg)))
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

    # ------------------------------------------------------------------ run
    def tick(self, max_ms: float = 33.0) -> LiveFrame:
        """Apply pending changes, run as many steps as fit in ``max_ms`` (minus the expected render
        time; or ``steps_per_frame`` steps) and render one slice frame. While paused only a posted
        ``step`` advances the flow."""
        if self._closed:
            raise RuntimeError("the live session is closed")
        p = self._take_pending()
        self._apply(p)
        self.renderer.apply_view(self.solver, self.view)      # the smoke rake, before the steps
        budget = max(0.0, max_ms / 1000.0 - self._render_s)
        if not self._paused:
            self._advance(self._steps_this_tick(budget))
        elif p.get("step"):
            self._advance(int(p["step"]))
        tr = time.perf_counter()
        hist = _trim_history3(self.solver, LIVE_HISTORY_T)
        frame = self.renderer.render(self.solver, self.view, phase="live", progress=0.0, eta_s=None,
                                     steps_per_s=self._rate, history=hist)
        frame.notices = self._take_notices()
        el = time.perf_counter() - tr
        if self._render_timed:
            self._render_s = 0.7 * self._render_s + 0.3 * el
        self._render_timed = True
        return frame
