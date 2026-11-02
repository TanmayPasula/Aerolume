"""3D production runs: :func:`run_cfd3d` drives a FlowSolver3D through a SimPlan3D.

The loop is the 2D runner's (:mod:`.runner`), deliberately: the same chunking
to the frame rate, the same averaging start, the same convergence verdict
(MSER transient, batch-means standard errors on Cd and Cl, no drift, settled
amplitude, whole cycles), the same one-time averaging restart, the same early
stop, cancel and blow-up handling. A user who knows what "converged" means in
2D knows what it means here. :func:`run_cfd` dispatches to it.

What is new is the output (:class:`~aerolume.model.CFD3DResult`, DESIGN.md 7.7):

* six coefficients with statistics -- CD, CS, CL, C_roll, Cm, Cn -- and the
  dimensional forces [N] and moments [N m];
* surface cp, cf and the wall-shear vector per vertex of the user's own meshes
  (the solver probes each vertex; vertex order is the design's, so the values
  map straight back; with a symmetry plane the vertices of the missing half are
  probed at their mirror image, so the whole body is covered);
* three mean slices: the symmetry (or centre) plane y, mid-height z and a
  cross-flow plane just behind the body -- mirrored to the full width when a
  half domain was simulated;
* a down-sampled mean volume (<= VOLUME_MAX_CELLS cells: u, v, w, cp, Q and the
  solid fraction; ``mirrored`` when only y >= 0 was simulated) for the 3D viewer;
* the smoke (DESIGN.md 7.13): the solver releases its passive scalar from
  ``view.seed_rake`` (moved live when the GUI moves the rake), and the result
  keeps an *instantaneous* snapshot of it at the end of the run on the same block
  grid -- a VolumeData labelled "instant" in ``result.extra_volumes`` with "dye"
  and the velocity "u", "v", "w" of that moment (``result.find_volume("dye")``;
  ``find_volume("u", "instant")`` for instantaneous streamlines), or, for a
  result type without that list, the field "dye" of ``result.volume``
  (``aerolume.viz3d.flow.smoke_volume`` finds it either way). An average of
  smoke is a blur, so the snapshot is the picture a smoke photograph would give;
* when ``sim.cfd.compute_sensitivity`` is set, the 3D sensitivity map from
  :mod:`aerolume.optimize.sensitivity3d` (guarded: a missing module, too large a
  grid, too little GPU memory, a failure or a cancel become warnings and the
  forces are always kept).

Frames are slice images (:class:`~aerolume.simulation.live3d.FrameRenderer3D`,
``ViewSettings.slice_axis`` / ``slice_pos``) in the 2D LiveFrame contract.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import threading
import time
from collections.abc import Callable

import numpy as np

from ..core import units
from ..model import CFD3DResult, SliceData, Stat, SurfaceData3D, VolumeData
from . import plan3d as P3
from . import runner as R2
from .live import LiveFrame, ViewSettings, cpu_threads
from .live3d import FrameRenderer3D, dye_rake_for, field_meta3d, mirror_slice, slice_extent
from .plan import CL_RTOL_FACTOR
from .plan3d import SimPlan3D, estimate_steps_per_second3d, record_throughput3d

__all__ = ["SLICE_FIELDS", "VOLUME_FIELDS", "VOLUME_MAX_CELLS", "run_cfd3d"]

ProgressFn = Callable[[float, str], None]
FrameFn = Callable[[LiveFrame], None]

#: Largest volume kept in a result (cells): the 3D viewer's streamlines and Q
#: iso-surfaces need the field, the project stores it in float16 (2 M cells x 6
#: arrays ~ 24 MB), and the solver grid (up to ~14 M cells) is down-sampled by
#: whole-cell block averages to fit.
VOLUME_MAX_CELLS = 2_000_000
#: Fields of every stored slice (SI units; "q" in 1/s², "vorticity" the magnitude).
SLICE_FIELDS = ("u", "v", "w", "speed", "p", "cp", "vorticity", "q")
#: Fields of the stored volume (plus the solid fraction as ``mask``).
VOLUME_FIELDS = ("u", "v", "w", "cp", "q")
#: Coefficient names, history keys and model fields in solver order.
COEFFS = ("cd", "cs", "cl", "c_roll", "cm", "cn")


def _diverged_message3(t: float, plan: SimPlan3D, why: str = "non-finite forces") -> str:
    nx, ny, nz = plan.grid3
    return (f"The flow solution became unstable at t = {t:.3g} L/U ({why}). This happens when the grid is too "
            "coarse for the flow: choose a higher quality, set the flow model to LES, or use the 'capped' "
            f"Reynolds-number mode. (Grid {nx:,} × {ny:,} × {nz:,}, Re {units.fmt_reynolds(plan.re_sim)}.)")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_cfd3d(plan: SimPlan3D, *, progress: ProgressFn | None = None, frame: FrameFn | None = None,
              cancel: threading.Event | None = None, view: ViewSettings | None = None) -> CFD3DResult:
    """Run a planned 3D Navier-Stokes simulation and return its :class:`CFD3DResult`.

    Same contract as :func:`aerolume.simulation.runner.run_cfd`: ``progress``
    after every chunk, ``frame`` (slice images) at about ``view.fps``, the last
    one with ``phase="done"``; ``cancel`` returns the partial result with a
    warning; ``ValueError`` for an explore plan, ``RuntimeError`` for a blow-up
    or a device failure (readable messages).
    """
    from ..solvers.ns3d import FlowSolver3D
    if plan.live or not math.isfinite(plan.t_end):
        raise ValueError("This plan is for an explore session (no end time); plan the run with plan_simulation().")
    wall0 = time.perf_counter()
    view = view if view is not None else ViewSettings()
    report = R2._Progress(progress)
    cfg = plan.solver_config
    notes, warnings = list(plan.notes), list(plan.warnings)
    t_end_planned = t_end = float(plan.t_end)
    t_avg = float(plan.t_avg_start)
    user_duration = bool(plan.sim.cfd.duration)
    want_sens = bool(getattr(plan.sim.cfd, "compute_sensitivity", False))
    sens_fits = want_sens and plan.cells <= P3.sensitivity_max_cells(plan.device)
    share = R2.STEPPING_SHARE_SENS if sens_fits else R2.STEPPING_SHARE
    laminar = cfg.turbulence == "none"
    rtol = float(getattr(plan, "conv_rtol", 0.05))
    is_cuda = not plan.device.startswith("cpu")
    nx, ny, nz = plan.grid3

    if getattr(cfg, "dye", False) and getattr(cfg, "emitters", None) is None and not getattr(plan, "closed", False):
        # the smoke leaves from the rake the user sees (or the default one upstream of the body);
        # placed emitters (DESIGN.md 10.1) replace the rake, and a closed box has none
        try:
            cfg = dataclasses.replace(cfg, dye_rake=dye_rake_for(plan, getattr(view, "seed_rake", None))[0])
        except (ValueError, TypeError):
            cfg = dataclasses.replace(cfg, dye_rake=dye_rake_for(plan, None)[0])
    report(0.0, f"Building the solver ({nx:,} × {ny:,} × {nz:,} cells on {plan.device_name})")
    from .physics_run import MonitorLog, chunk_samples
    monlog = MonitorLog()
    bodies_on = bool(cfg.bodies)
    with cpu_threads(plan.device):
        solver = R2._retry_oom(lambda: FlowSolver3D(cfg), plan)
        if is_cuda and getattr(solver, "fused_error", None):
            notes.append("The fused GPU kernels could not be compiled here, so the solver ran its reference path "
                         f"(about 3.5× slower; {str(solver.fused_error).splitlines()[0][:160]}).")
        renderer = FrameRenderer3D(plan) if frame is not None else None
        rate = estimate_steps_per_second3d(plan.device, plan.cells, advection=cfg.advection,
                                           turbulence=cfg.turbulence, dtype=cfg.dtype) / float(getattr(plan, "step_cost", 1.0))
        averaging = False
        t_field_avg = math.nan
        samples_t: list[float] = []
        samples_b: list[np.ndarray] = []
        step_wall = 0.0
        steps_timed = 0
        loop_wall = 0.0
        first = True
        dt_last = plan.dt
        last_check = -math.inf
        last_frame = -math.inf
        render_s = 0.0
        wall_rate = math.nan
        mark = None
        verdict: dict | None = None
        stopped_early = cancelled = restarted = False
        t_first = None
        prog_base = (0.0, 0.0)

        while solver.t < t_end - 1e-12:
            it0 = time.perf_counter()
            timed = not first
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            if frame is not None:
                interval = max(1.0 / max(float(view.fps or 20.0), 1.0), render_s / R2.RENDER_DUTY)
                budget = interval - render_s
            else:
                interval = budget = R2.HEADLESS_CHUNK_S
            n = int(max(1, min(5000, rate * budget)))
            if first:
                n = min(n, R2.FIRST_CHUNK_STEPS)
            target = t_avg if not averaging else t_end
            if dt_last > 0:
                n = min(n, max(1, math.ceil((target - solver.t) / dt_last)))
            if is_cuda and n >= 20:
                n -= n % 10
            ts = time.perf_counter()
            if first:                           # NVRTC compile + graph capture: the riskiest allocation
                R2._retry_oom(lambda k=n: solver.advance(k), plan)
            else:
                try:
                    solver.advance(n)
                except Exception as exc:
                    gpu = R2._gpu_failure(exc, plan) or (RuntimeError(R2._oom_message(plan)) if R2._is_oom(exc)
                                                         else None)
                    if gpu is not None:
                        raise gpu from exc
                    raise
            el = max(time.perf_counter() - ts, 1e-9)
            t = solver.t
            if t_first is None:
                t_first = t
            dt_last = solver.dt
            if not first:
                step_wall += el
                steps_timed += n
                rate = 0.7 * rate + 0.3 * (n / el)
            first = False
            c = solver.coefficients()
            if not np.all(np.isfinite(c)) or np.abs(c).max() > 1e6:
                raise RuntimeError(_diverged_message3(t, plan))
            if not (dt_last > R2.DT_COLLAPSE * plan.dt):
                raise RuntimeError(_diverged_message3(t, plan, "the time step collapsed: the velocity is growing "
                                                               "without bound"))
            sample = chunk_samples(solver, plan)
            if sample:
                if not all(math.isfinite(v) for v in sample.values()):
                    raise RuntimeError(_diverged_message3(t, plan, "non-finite flow or temperature"))
                monlog.add(t, sample)
            if averaging:
                samples_t.append(t)
                samples_b.append(solver.body_coefficients())
            elif t >= t_avg - 1e-12:
                solver.start_averaging()
                averaging, t_field_avg = True, t

            now = time.perf_counter()
            win = t_end - max(t_avg, t_field_avg) if averaging else t_end - t_avg
            if averaging and now - last_check >= R2.CHECK_PERIOD_S and t - t_field_avg >= R2.MIN_AVG_FRACTION * win:
                last_check = now
                verdict = R2._combined_verdict(R2.history_now(solver, plan, monlog), plan, t_field_avg, rtol=rtol,
                                               laminar=laminar)
                if verdict["stop"]:
                    stopped_early = True
                    loop_wall += time.perf_counter() - it0 if timed else 0.0
                    break
                if (verdict["drifting"] and not restarted and not user_duration
                        and t - t_avg < 0.6 * win and t_end < R2.MAX_EXTENSION * t_end_planned):
                    solver.start_averaging()
                    restarted = True
                    verdict = None
                    lost = t - t_field_avg
                    t_field_avg = t
                    samples_t.clear()
                    samples_b.clear()
                    prog_base = (report.last, t)
                    t_end = min(t_end + lost, R2.MAX_EXTENSION * t_end_planned)
                    notes.append(f"The start-up transient lasted longer than planned (the "
                                 f"{'mean drag was' if bodies_on else 'monitored values were'} still drifting at "
                                 f"t = {t:.3g}), so averaging restarted there and the run was extended to "
                                 f"t = {t_end:.4g}.")

            if mark is not None and now > mark[0] and t > mark[1]:
                r = (t - mark[1]) / (now - mark[0])
                wall_rate = r if not math.isfinite(wall_rate) else 0.8 * wall_rate + 0.2 * r
            mark = (now, t)
            R2.refine_estimate(plan, elapsed=now - wall0, t=t, t_end=t_end, wall_rate=wall_rate)
            f0, t0p = prog_base
            frac = f0 + (share - f0) * (t - t0p) / max(t_end - t0p, 1e-12)
            eta = (t_end - t) / wall_rate + 1.0 if math.isfinite(wall_rate) and wall_rate > 0 else None
            report(frac, R2.progress_message(t, t_end, averaging, verdict, bodies_on))
            if renderer is not None and now - last_frame >= 0.8 * interval:
                last_frame = now
                tr = time.perf_counter()
                frame(renderer.render(solver, view, phase="averaging" if averaging else "transient", progress=frac,
                                      eta_s=eta, steps_per_s=rate))
                render_s = 0.7 * render_s + 0.3 * (time.perf_counter() - tr)
            if timed:
                loop_wall += time.perf_counter() - it0

        # ----------------------------------------------------------------- outputs
        report(share, "Collecting forces, surface, slices and the volume")
        t_final = solver.t
        h = R2.history_now(solver, plan, monlog)
        _record_measurements3(plan, solver=solver, steps_timed=steps_timed, step_wall=step_wall, loop_wall=loop_wall,
                              t_first=t_first, t_final=t_final, t_end_planned=t_end_planned,
                              live_view=frame is not None)
        result = _assemble3(plan, solver, h, averaging=averaging, t_field_avg=t_field_avg, samples_t=samples_t,
                            samples_b=samples_b, notes=notes, warnings=warnings, cancelled=cancelled,
                            stopped_early=stopped_early, t_end=t_end, rtol=rtol, laminar=laminar)
        final = (renderer.render(solver, view, phase="done", progress=1.0, eta_s=0.0, steps_per_s=rate, history=h)
                 if renderer is not None else None)
        if want_sens:
            if cancelled:
                result.warnings.append("The shape-sensitivity map was skipped because the run was stopped.")
            else:
                _attach_sensitivity3(plan, solver, result, report, cancel, share)
        result.wall_seconds = time.perf_counter() - wall0
        if final is not None:
            frame(final)
    solver = None                               # release the GPU memory before returning
    if is_cuda:
        import torch
        torch.cuda.empty_cache()
    report(1.0, R2.final_message(result, cancelled))
    return result


def _record_measurements3(plan: SimPlan3D, *, steps_timed: int, step_wall: float, loop_wall: float,
                          t_first: float | None, t_final: float, t_end_planned: float, live_view: bool,
                          solver=None) -> None:
    """Feed the run's throughput, loop share and CFL speed factor into the 3D estimates
    (the 2D rules of :func:`.runner._record_measurements`, including the diagnostics-based speed
    factor of runs with placed physics)."""
    if steps_timed <= 0 or step_wall < R2.MIN_TIMED_S or t_first is None:
        return
    cfg = plan.solver_config
    mean_dt = (t_final - t_first) / steps_timed
    whole = (t_final - t_first) >= 0.5 * t_end_planned
    sf = cfg.cfl * cfg.h / mean_dt if (mean_dt > 0 and whole) else None
    if getattr(plan, "placed", None) is not None:
        from .physics_run import measured_speed_factor
        sf = measured_speed_factor(solver, plan, t_first) if (whole and solver is not None) else None
    if sf is not None:
        sf = sf / max(float(getattr(plan, "speed_ratio", 1.0) or 1.0), 1.0)     # per peak speed, as planned
    eff = step_wall / loop_wall if loop_wall > step_wall > 0 else None
    kind = getattr(plan, "calibration_kind", "") or plan.geometry_class
    record_throughput3d(plan.device, plan.cells, steps_timed / step_wall * float(getattr(plan, "step_cost", 1.0)),
                        advection=cfg.advection, turbulence=cfg.turbulence, dtype=cfg.dtype, kind=kind,
                        speed_factor=sf, cells_per_ref=plan.cells_per_ref, loop_efficiency=eff, live_view=live_view)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _strouhal3(plan: SimPlan3D, h: dict[str, np.ndarray], t0: float, cd_mean: float) -> tuple[float | None, str | None]:
    """St of the stronger of the lift and side-force oscillations (2D rules, :func:`.runner._strouhal`).

    A 3D wake may shed in any plane; with a symmetry plane only in it (lift).
    For a wing whose boundary layers the grid does not resolve the oscillation
    belongs to the under-resolved flow, as for a 2D airfoil, and is not reported.
    """
    t = h["t"]
    if not t.size:
        return None, None
    sel = t >= t0
    key = "cl"
    if "cs" in h and sel.any() and np.nanstd(h["cs"][sel]) > np.nanstd(h["cl"][sel]):
        key = "cs"
    if plan.geometry_class == "wing" and not plan.bl_resolved:
        st, note = R2._strouhal(plan, t, h[key], t0, cd_mean=cd_mean)
        if st is None:
            return None, note
        return None, ("The forces oscillate, but at this Reynolds number the grid does not resolve the wing's "
                      "boundary layers, so the oscillation belongs to the under-resolved flow; no Strouhal number "
                      "is reported.")
    return R2._strouhal(plan, t, h[key], t0, cd_mean=cd_mean)


def _partner_fix(plan: SimPlan3D, per_body: dict[str, dict[str, float]], notes: list[str],
                 warnings: list[str]) -> None:
    """Per-part values of a half-domain run, shared between a part and its mirror image.

    The solver doubles each part's half-domain force to the full body. A part
    lying wholly on the y < 0 side (a left wheel) is outside the half domain and
    reads zero, while its mirror image (the right wheel) reads both: each gets
    half. Side force, roll and yaw of a single off-plane part are not separable
    from a half domain (the pair's sum is zero) and stay 0.
    """
    if not plan.symmetry_plane or len(plan.bodies_wind3) < 2:
        return
    size = max(plan.body_bounds[1] - plan.body_bounds[0], 1e-9)
    tol = 1e-3 * size
    bounds = [m.bounds() for m in plan.bodies_wind3]
    names = [m.name for m in plan.bodies_wind3]
    shared = 0
    for i, bi in enumerate(bounds):
        if bi[3] > tol:                                   # not wholly on the missing side
            continue
        mirror = (bi[0], bi[1], -bi[3], -bi[2], bi[4], bi[5])
        for j, bj in enumerate(bounds):
            if j == i or bj[2] < -tol:
                continue
            if max(abs(a - b) for a, b in zip(mirror, bj)) <= 10 * tol:
                both = per_body.get(names[j])
                if both is None:
                    break
                half = {k: (0.5 * v if k in ("cd", "cl", "cm") else 0.0) for k, v in both.items()}
                per_body[names[j]] = dict(half)
                per_body[names[i]] = dict(half)
                shared += 1
                break
        else:
            warnings.append(f"'{names[i]}' lies wholly on the mirrored side of the symmetry plane and has no mirror "
                            "image among the parts; its forces are counted only through the symmetry.")
    if shared:
        notes.append(f"Per-part values with a symmetry plane: {shared} part(s) off the plane share the force "
                     "measured for them and their mirror image half and half; their individual side force, roll and "
                     "yaw moments are not separable in a half domain and are given as 0.")


def _thermal(plan: SimPlan3D, solver) -> bool:
    """Whether the run carries a temperature (heat transfer on in the plan and the solver)."""
    s = getattr(plan, "physics", None)
    return bool(s is not None and s.thermal and getattr(solver, "thermal_enabled", False))


def _scale3(name: str, plan: SimPlan3D) -> float:
    return field_meta3d(name).scale(plan.speed, plan.fluid.density, plan.length_ref)


def _surfaces3(plan: SimPlan3D, solver, mean: bool) -> list[SurfaceData3D]:
    """Per-vertex surface data on every mesh, in metres, in the design's vertex order."""
    L = plan.length_ref
    out = []
    for d in solver.surface(mean=mean):
        out.append(SurfaceData3D(body=str(d["name"]), vertices=np.asarray(d["vertices"], dtype=np.float64) * L,
                                 faces=np.asarray(d["faces"], dtype=np.int64),
                                 cp=np.asarray(d["cp"], dtype=np.float64), cf=np.asarray(d["cf"], dtype=np.float64),
                                 tau=np.asarray(d["tau"], dtype=np.float64)))
    return out


def _slices3(plan: SimPlan3D, solver, mean: bool) -> list[SliceData]:
    """The three default slices, SI fields, mirrored to the full width with a symmetry plane."""
    L = plan.length_ref
    ext = plan.extent3
    h_m = plan.solver_config.h * L
    full = plan.extent3_full
    out = []
    for axis in ("y", "z", "x"):
        k = {"x": 0, "y": 1, "z": 2}[axis]
        pos = float(plan.slice_positions.get(axis, 0.5 * (ext[2 * k] + ext[2 * k + 1])))
        # the solver samples between the two nearest cell-centre planes, clamped to the domain
        pos = min(max(pos, ext[2 * k] + 0.5 * h_m), ext[2 * k + 1] - 0.5 * h_m)
        fields = {}
        for name in SLICE_FIELDS + (("temperature",) if _thermal(plan, solver) else ()):
            a = solver.slice(axis, pos / L, name, mean=mean).double()
            # temperature in °C (theta * ΔT + T_amb; FieldData's "temperature" is in °C)
            a = plan.physics.celsius(a) if name == "temperature" else a * _scale3(name, plan)
            a = a.float().cpu().numpy()
            fields[name] = mirror_slice(a, axis, name) if plan.symmetry_plane else a
        m = solver.slice(axis, pos / L, "mask").float().cpu().numpy()
        if plan.symmetry_plane:
            m = mirror_slice(m, axis, "mask")
        out.append(SliceData(axis=axis, position=pos, extent=slice_extent(axis, full), fields=fields, mask=m,
                             label="mean" if mean else "instant"))
    return out


def _volume_factor(plan: SimPlan3D) -> int:
    """Block size of the stored volume: the smallest whole number of cells per block that
    keeps it under VOLUME_MAX_CELLS."""
    nx, ny, nz = plan.grid3
    f = 1
    while (nx // f) * (ny // f) * (nz // f) > VOLUME_MAX_CELLS:
        f += 1
    return f


def _smoke3(plan: SimPlan3D, solver, mean_volume: VolumeData | None = None) -> VolumeData | None:
    """The instantaneous snapshot of 7.13 on the stored volume's block grid: the smoke
    ("dye") and, when the stored volume is a time average, the instantaneous velocity
    (u, v, w [m/s]), so a result can show the streamlines of the moment the smoke
    photograph was taken (they differ from the mean flow's in a shedding wake). None
    without smoke. The solid fraction is shared with ``mean_volume`` (same grid)."""
    import torch
    import torch.nn.functional as F
    if not getattr(solver, "dye_enabled", False):
        return None
    nx, ny, nz = plan.grid3
    f = _volume_factor(plan)
    h_m = plan.solver_config.h * plan.length_ref

    def pooled(t: torch.Tensor) -> np.ndarray:
        if f > 1:
            t = t[: (nz // f) * f, : (ny // f) * f, : (nx // f) * f]
            t = F.avg_pool3d(t[None, None].float(), f, stride=f)[0, 0]
        return t.float().cpu().numpy()

    fields: dict[str, np.ndarray] = {}
    with torch.no_grad():
        fields["dye"] = np.clip(pooled(solver.field("dye")), 0.0, 1.0)
        for k in range(1, int(getattr(solver, "dye_channels", 1) or 1)):
            fields[f"dye{k}"] = np.clip(pooled(solver.field(f"dye{k}")), 0.0, 1.0)
        if _thermal(plan, solver):
            fields["temperature"] = pooled(plan.physics.celsius(solver.field("temperature").double())).astype(np.float32)
        # an instantaneous result volume already holds this velocity: no second copy
        if mean_volume is not None and mean_volume.label == "mean":
            for name in ("u", "v", "w"):
                fields[name] = pooled(solver.field(name).double() * _scale3(name, plan)).astype(np.float32)
    mask = mean_volume.mask if mean_volume is not None else None
    x0, _, y0, _, z0, _ = plan.extent3
    return VolumeData(origin=(float(x0), float(y0), float(z0)), spacing=(f * h_m, f * h_m, f * h_m),
                      fields=fields, mask=mask, label="instant", mirrored=bool(plan.symmetry_plane))


def _attach_smoke(result: CFD3DResult, smoke: VolumeData | None) -> None:
    """Put the smoke snapshot where the result keeps it (see the module docstring)."""
    if smoke is None:
        return
    extra = getattr(result, "extra_volumes", None)
    if isinstance(extra, list):
        extra.append(smoke)
        return
    vol = result.volume
    if vol is not None and vol.fields and next(iter(vol.fields.values())).shape == smoke.fields["dye"].shape:
        vol.fields["dye"] = smoke.fields["dye"]


def _volume3(plan: SimPlan3D, solver, mean: bool) -> VolumeData:
    """A block-averaged copy of the flow (<= VOLUME_MAX_CELLS cells), SI units."""
    import torch
    import torch.nn.functional as F
    nx, ny, nz = plan.grid3
    f = _volume_factor(plan)
    L = plan.length_ref
    h_m = plan.solver_config.h * L

    def pooled(t: torch.Tensor) -> np.ndarray:
        if f > 1:
            t = t[: (nz // f) * f, : (ny // f) * f, : (nx // f) * f]
            t = F.avg_pool3d(t[None, None].float(), f, stride=f)[0, 0]
        return t.float().cpu().numpy()

    fields: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name in VOLUME_FIELDS + (("temperature",) if _thermal(plan, solver) else ()):
            src = solver.mean_fields((name,))[name] if mean else solver.field(name)
            val = plan.physics.celsius(src.double()) if name == "temperature" else src.double() * _scale3(name, plan)
            fields[name] = pooled(val).astype(np.float32)
            del src, val
        mask = pooled(solver.field("mask")).astype(np.float32)
    x0, _, y0, _, z0, _ = plan.extent3
    return VolumeData(origin=(float(x0), float(y0), float(z0)), spacing=(f * h_m, f * h_m, f * h_m), fields=fields,
                      mask=mask, label="mean" if mean else "instant", mirrored=bool(plan.symmetry_plane))


def _assemble3(plan: SimPlan3D, solver, h: dict[str, np.ndarray], *, averaging: bool, t_field_avg: float,
               samples_t: list[float], samples_b: list[np.ndarray], notes: list[str], warnings: list[str],
               cancelled: bool, stopped_early: bool, t_end: float, rtol: float, laminar: bool) -> CFD3DResult:
    """Statistics, dimensional values, surface, slices and volume of a finished (or stopped) run."""
    from ..analysis import coefficient_stats
    cfg = plan.solver_config
    L, U, rho = plan.length_ref, plan.speed, plan.fluid.density
    t = h["t"]
    t_now = float(t[-1]) if t.size else 0.0
    bodies_on = bool(cfg.bodies)
    if averaging:
        v = R2._combined_verdict(h, plan, t_field_avg, rtol=rtol, laminar=laminar)
        t0 = v["t0"]
    else:
        v = None
        t0 = float(t[0] + 0.5 * (t[-1] - t[0])) if t.size else 0.0
    if bodies_on:
        stats = {k: (coefficient_stats(t, h[k], t0) if t.size else Stat()) for k in COEFFS}
        R2._mark_steady_errors(stats, v, rtol)
        if plan.symmetry_plane:
            for k in ("cs", "c_roll", "cn"):              # exactly zero by construction: no error bar to doubt
                stats[k].stderr_reliable = True
        st, st_note = _strouhal3(plan, h, t0, stats["cd"].mean)
    else:
        stats = {k: Stat() for k in COEFFS}          # a closed box without bodies has no forces
        st, st_note = None, None
    converged = bool(v is not None and v["stop"])
    R2._monitor_messages(v, plan, notes, warnings, converged=converged, cancelled=cancelled, t_now=t_now,
                         t_end=t_end, rtol=rtol)

    if cancelled:
        warnings.append(f"Stopped by the user at t = {t_now:.3g} of {t_end:.4g} L/U: "
                        + ("these values are averaged over a shorter window than planned and are not converged."
                           if averaging else "the flow had not finished developing, so the values are a rough "
                                             "indication only (last half of the record)."))
    if v is not None and v.get("cd"):
        jcd, jcl = v["cd"], v["cl"]
        if converged:
            cyc = f" over {v['cycles']:.0f} shedding cycles" if v.get("periodic") and v.get("cycles") else ""
            notes.append(f"{'Stopped early at' if stopped_early else 'Converged by'} t = {t_now:.3g} of {t_end:.4g} "
                         f"L/U: mean Cd known to ±{R2._pct(jcd['rel_stderr'])} and mean Cl to "
                         f"±{units.sig(jcl['stderr'], 2)}{cyc} (targets {R2.target_text(jcd, 100 * rtol)} and "
                         f"{R2.target_text(jcl, 100 * CL_RTOL_FACTOR * rtol)}, one standard error from batch means); no drift between "
                         "the halves of the averaging window.")
        elif not cancelled and not (jcd["state"] == "converged" and jcl["state"] == "converged"
                                    and v.get("enough_cycles", True) and v.get("monitors")):
            worst = jcd if jcd["state"] != "converged" else jcl
            what = "Cd" if worst is jcd else "Cl"
            target = R2.target_text(worst, 100 * rtol * (1.0 if worst is jcd else CL_RTOL_FACTOR))
            reason = worst["reason"] if worst["state"] != "converged" else (
                f"Fewer than {R2.MIN_CYCLES:.0f} shedding cycles averaged.")
            if jcd.get("reliable") and jcl.get("reliable"):
                advice = ("The averages are usable with the error bars shown; run longer (Duration) before relying "
                          "on differences smaller than about twice them.")
            else:
                n_eff = min(jcd.get("ess", 0.0), jcl.get("ess", 0.0))
                advice = (f"The error bars are rough estimates, not measurements: only {n_eff:.0f} effectively "
                          "independent samples were averaged, and the mean may still contain start-up transient. "
                          "Run longer (Duration) before comparing designs with this result.")
            warnings.append(f"Not converged by t = {t_now:.3g} L/U ({what}, target {target}: {reason}) " + advice)
        notes.append(f"Forces averaged over t = {t0:.3g} … {t_now:.3g} L/U"
                     + (f" (the start-up transient was found to end at t = {t0:.3g}; field averages start at "
                        f"t = {t_field_avg:.3g})" if abs(t0 - t_field_avg) > 1e-6 else "") + ".")
    elif v is not None and not cancelled and bodies_on:
        # too few samples in the window for any verdict (a very short Duration)
        warnings.append(f"Not converged by t = {t_now:.3g} L/U: the averaging window holds too few time steps "
                        f"(fewer than {R2.MIN_SAMPLES}) to judge the mean forces; run longer (Duration).")
    if st is not None:
        notes.append(f"Strouhal number based on {units.sig(plan.strouhal_length, 3)} m "
                     + ("(the vehicle height)." if plan.geometry_class == "vehicle" else
                        "(the mean chord)." if plan.geometry_class == "wing" else "(the frontal size)."))
    elif st_note:
        notes.append(st_note)

    q = 0.5 * rho * U * U
    A, b = plan.ref_area, plan.span_ref
    m = {k: stats[k].mean for k in COEFFS}
    drag, side, lift = m["cd"] * q * A, m["cs"] * q * A, m["cl"] * q * A
    moments = (m["c_roll"] * q * A * b, m["cm"] * q * A * L, m["cn"] * q * A * b)

    names = [bd.name for bd in cfg.bodies]
    per_body: dict[str, dict[str, float]] = {}
    if len(names) == 1:
        per_body[names[0]] = {k: float(m[k]) for k in COEFFS}
    elif names:
        ts = np.asarray(samples_t)
        sel = ts >= t0 if ts.size else np.zeros(0, bool)
        if sel.sum() >= 2:
            mb = np.asarray(samples_b)[sel].mean(axis=0)
            notes.append(f"Per-part coefficients are means of {int(sel.sum())} samples spread over the averaging "
                         "window; the totals use every time step.")
        else:
            mb = solver.body_coefficients()
            warnings.append("Per-part coefficients are instantaneous values (too few samples to average).")
        for i, nm in enumerate(names):
            per_body[nm] = {k: float(mb[i, j]) for j, k in enumerate(COEFFS)}
        _partner_fix(plan, per_body, notes, warnings)

    surface: list[SurfaceData3D] = []
    slices: list[SliceData] = []
    volume = None
    smoke = None
    try:
        surface = _surfaces3(plan, solver, mean=averaging)
    except Exception as exc:  # noqa: BLE001 - outputs are best effort, forces are what matter
        warnings.append(f"Surface distributions could not be extracted ({type(exc).__name__}: {exc}).")
    try:
        slices = _slices3(plan, solver, mean=averaging)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Flow slices could not be extracted ({type(exc).__name__}: {exc}).")
    try:
        volume = _volume3(plan, solver, mean=averaging)
        nzv, nyv, nxv = next(iter(volume.fields.values())).shape
        if volume.spacing[0] > 1.5 * plan.solver_config.h * L:
            notes.append(f"The stored flow volume is a {nxv} × {nyv} × {nzv} block average of the solver grid "
                         f"(cells {units.sig(volume.spacing[0] * 1000, 3)} mm, {round(volume.spacing[0] / (cfg.h * L))}× "
                         "the solver's) for the 3D viewer; forces, surface data and slices use the full grid.")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"The flow volume could not be extracted ({type(exc).__name__}: {exc}).")
    try:
        smoke = _smoke3(plan, solver, volume)
    except Exception as exc:  # noqa: BLE001 - the smoke picture is an extra
        warnings.append(f"The smoke snapshot could not be extracted ({type(exc).__name__}: {exc}).")

    result = CFD3DResult(
        sim_id=plan.sim.id, cd=stats["cd"], cs=stats["cs"], cl=stats["cl"], c_roll=stats["c_roll"], cm=stats["cm"],
        cn=stats["cn"], drag=float(drag), side=float(side), lift=float(lift),
        moments=tuple(float(x) for x in moments), drag_area=float(m["cd"] * A), power=float(drag * U),
        strouhal=st, strouhal_length=float(plan.strouhal_length), re_real=float(plan.re_real),
        re_sim=float(solver.cfg.re), grid=tuple(int(x) for x in plan.grid3), cells=int(plan.cells),
        cells_per_ref=float(plan.cells_per_ref), symmetry_plane=bool(plan.symmetry_plane), t_end=t_now,
        t_avg_start=float(t0), converged=converged, device=plan.device_name, precision=plan.precision,
        turbulence=plan.turbulence, notes=notes, warnings=warnings, per_body=per_body,
        history={k: np.asarray(x, dtype=np.float64) for k, x in h.items()}, surface=surface, slices=slices,
        volume=volume)
    _attach_smoke(result, smoke)
    R2.attach_physics(result, solver, plan, h, t0, averaging)
    return result


# ---------------------------------------------------------------------------
# sensitivity (aerolume.optimize.sensitivity3d)
# ---------------------------------------------------------------------------

def _call_accepting(fn, *args, **kwargs):
    """Call ``fn`` with only the keyword arguments its signature accepts (the 3D
    sensitivity module is written in parallel; optional extras must not break it)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _attach_sensitivity3(plan: SimPlan3D, solver, result: CFD3DResult, report: R2._Progress,
                         cancel: threading.Event | None, share: float) -> None:
    """Compute the 3D shape-sensitivity map and attach it to ``result.surface``.

    Guarded like 2D (:func:`.runner._attach_sensitivity`), plus a size check: the
    reverse sweep holds checkpoints of the whole state, so a grid above the
    module's cell budget (:func:`.plan3d.sensitivity_max_cells`, DESIGN.md 7.1)
    is skipped with a warning before any work is done, and so is a sweep the
    module itself finds too large for the free device memory
    (``SensitivityBudgetError``).
    """
    limit = P3.sensitivity_max_cells(plan.device)
    if plan.cells > limit:
        result.warnings.append(f"The shape-sensitivity map was skipped: it is limited to grids of "
                               f"{limit / 1e6:.2g} M cells and this one has {plan.cells / 1e6:.2f} M (Draft quality "
                               "or a symmetry plane makes it fit); the forces are unaffected.")
        return
    try:
        # import_module consults sys.modules (a disabled or replaced module is honoured even
        # when the package already holds the attribute from an earlier import)
        S3 = importlib.import_module("aerolume.optimize.sensitivity3d")
        compute, attach = S3.compute_sensitivity3d, S3.attach_sensitivity3d
    except Exception as exc:  # noqa: BLE001 - ImportError, AttributeError, or a broken install
        result.warnings.append("The shape-sensitivity map was requested but the 3D sensitivity module is not "
                               f"available in this installation ({type(exc).__name__}: {exc}); the forces are "
                               "unaffected.")
        return
    from ..optimize.common import Cancelled
    budget_error = getattr(S3, "SensitivityBudgetError", ())
    objective = str(getattr(plan.sim.cfd, "sensitivity_objective", "cd") or "cd")
    label = {"cd": "Cd", "cl": "Cl", "ld": "L/D", "cm": "Cm"}.get(objective.lower(), objective)
    report(share, f"Computing the shape sensitivity of {label}")

    def fwd(frac: float, msg: str = "") -> None:
        report(share + (1.0 - share) * 0.98 * max(0.0, min(1.0, float(frac))),
               f"Sensitivity of {label}: {msg}" if msg else f"Computing the shape sensitivity of {label}")

    try:
        arrays = _call_accepting(compute, plan.design, plan.sim, objective=objective, plan=plan, solver=solver,
                                 progress=fwd, cancel=cancel)
        attach(result, arrays)
    except Cancelled:
        result.warnings.append("The shape-sensitivity map was skipped: the run was stopped while computing it (the "
                               "forces are complete).")
        return
    except budget_error as exc:                     # too large for the cell or memory budget: said before any work
        text = str(exc).strip() or "it does not fit the device"
        result.warnings.append(f"The shape-sensitivity map was skipped: {text}")
        return
    except Exception as exc:  # noqa: BLE001 - a failed adjoint must not cost the user the forces
        if R2._is_oom(exc):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            msg = "the GPU ran out of memory for the reverse sweep; try Draft quality for the map"
        else:
            msg = f"{type(exc).__name__}: {exc}"
        result.warnings.append(f"The shape-sensitivity map could not be computed ({msg}); the forces are unaffected.")
        return
    if not any(getattr(s, "sensitivity", None) is not None for s in result.surface):
        result.warnings.append("The sensitivity computation finished but returned no surface map.")

