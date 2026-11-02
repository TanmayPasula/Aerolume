"""Production runs: :func:`run_cfd` (Navier-Stokes) and :func:`run_polar` (viscous-inviscid sweep).

``run_cfd`` drives a :class:`~aerolume.solvers.ns2d.FlowSolver` through a
:class:`~aerolume.simulation.plan.SimPlan` and assembles a
:class:`~aerolume.model.CFDResult` in SI units. The loop:

1. **Transient.** Advance in chunks sized so that stepping plus rendering one
   frame takes one frame interval (``view.fps``), rendering a
   :class:`LiveFrame` per chunk and reporting progress with a message that
   reads well on its own ("Developing the flow (t = 3.2 of 30)").
2. **Averaging** from ``plan.t_avg_start``: the solver accumulates mean fields;
   per-body coefficients are sampled every chunk. About every 1.5 s of wall
   time the force history is judged (:func:`_verdict`):

   * the averaging window starts where the start-up transient really ends
     (MSER-5 on Cd and Cl, never before ``t_avg_start``);
   * the mean of Cd must be known to the plan's tolerance ``conv_rtol`` (one
     standard error, batch means over whole shedding periods / integral time
     scales), the mean of Cl to ``CL_RTOL_FACTOR`` times it -- with a floor tied
     to Cd, so a body without lift is not held to a relative error of zero;
   * the two halves of the window must agree: a difference larger than the
     tolerance *and* than three standard errors is a drift;
   * the fluctuation amplitude must not be changing. For laminar (deterministic)
     flows that is the classic +-25 % between the halves; a large-eddy
     simulation is chaotic, its rms has its own sampling error, so there the
     change must also exceed three standard errors of the log-rms. A fluctuation
     that is not growing and whose peak (√2 rms) fits inside the tolerance is
     exempt: it cannot move the mean by more than the target (the start-up
     residue on the steady Re 40 cylinder's lift). A growing one never is, and
     is not called "steady" however small: that is how shedding sets in;
   * a periodic flow needs at least four averaged cycles.

   When all hold the run stops early and the result is ``converged``. If the
   drag is still drifting early in the window, the field averages are restarted
   once and the run is extended by up to 50 % (unless the user fixed the
   duration). A run that ends without meeting the target is reported as not
   converged, with the uncertainty it did reach.
3. **Outputs**: statistics with standard errors, Strouhal number, dimensional
   forces, surface cp / cf from the mean flow, mean and instantaneous fields,
   and -- when ``sim.cfd.compute_sensitivity`` is set -- the shape-sensitivity
   map from :mod:`aerolume.optimize`.
4. **A cross-check for airfoils** (:func:`_polar_reference`): a single airfoil
   at a real Re of POLAR_CHECK_RE or more is also analysed with the viscous
   polar solver at the same Re and angle (~2 s), and the result states both
   values side by side -- as its first warning when they disagree. Without it a
   user copies "Cl 0.16, Cd 0.096" for a NACA 2412 at Re 1e6 into a report,
   where the boundary-layer-resolving answer is 0.72 / 0.007.

``cancel`` (a ``threading.Event``) stops the loop between chunks; the partial
result is returned with a warning that says so (the GUI saves it). A flow that
blows up raises a ``RuntimeError`` with advice instead of returning garbage.
"""
from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable

import numpy as np

from ..core import units
from ..core.fluid import display_name
from ..model import CFDResult, Design, FieldData, PolarResult, Simulation, Stat, SurfaceData
from .live import FrameRenderer, LiveFrame, ViewSettings, cpu_threads
from .plan import (
    CL_RTOL_FACTOR,
    POLAR_CHECK_RE,
    SimPlan,
    estimate_steps_per_second,
    fmt_duration,
    record_throughput,
)

__all__ = ["run_cfd", "run_polar", "run_simulation", "run_wing"]

ProgressFn = Callable[[float, str], None]
FrameFn = Callable[[LiveFrame], None]

#: Share of the progress bar for time stepping; the rest covers the outputs.
STEPPING_SHARE = 0.97
#: ... and when a sensitivity map is computed afterwards (an adjoint costs about
#: as much as a short forward run).
STEPPING_SHARE_SENS = 0.75
#: Wall-time interval between convergence checks.
CHECK_PERIOD_S = 1.5
#: Early stop needs at least this fraction of the planned averaging window ...
MIN_AVG_FRACTION = 0.3
#: ... and, for a shedding flow, at least this many averaged cycles.
MIN_CYCLES = 4.0
#: A drifting run may be extended to this multiple of its planned duration.
MAX_EXTENSION = 1.5
#: Chunk length without a frame callback [s of wall time] (fewer host syncs).
HEADLESS_CHUNK_S = 0.25
#: Largest share of the wall time spent rendering live frames.
RENDER_DUTY = 0.2
#: A Strouhal number is reported only when the spectral peak holds this share of
#: the lift fluctuation energy (measured: 0.93-0.98 for laminar shedding behind a
#: cylinder or a stalled airfoil, ~0.45 for the LES of a stalled airfoil) ...
ST_MIN_PEAK_SHARE = 0.35
#: ... and the lift rms is at least this fraction of max(|mean Cl|, mean Cd).
ST_MIN_RMS = 0.02
#: Steps in the first chunk (two CUDA-graph blocks).
FIRST_CHUNK_STEPS = 20
#: Runs with less stepping wall time than this do not update the run-time estimate.
MIN_TIMED_S = 1.0
#: Below this many samples in the window nothing is judged.
MIN_SAMPLES = 50
#: Fewer effectively independent samples than this cannot support a verdict.
MIN_ESS = 20.0
#: Laminar flows: fluctuation amplitude ratio between the window halves that
#: counts as settled.
AMPLITUDE_BAND = (0.8, 1.25)
#: Peak over rms of a fluctuation (√2, a sinusoid's): a fluctuation whose peak fits inside the
#: tolerance and whose amplitude is not growing is judged by its standard error alone.
SMALL_AMPLITUDE_PEAK = math.sqrt(2.0)
#: A fluctuation whose rms is below this share of the tolerance is round-off, whatever its trend
#: (the growth test of a "steady" signal ignores it).
GROWTH_FLOOR = 1e-3
#: The CFD and polar values "agree" when Cl is within this fraction (or 0.05
#: absolute) and Cd within this factor; otherwise the comparison is a warning.
POLAR_CL_RTOL = 0.2
POLAR_CD_FACTOR = 1.5
#: A time step below this fraction of the planned one means the velocity is
#: blowing up (the CFL rule shrinks dt as |u| grows) long before forces go NaN.
DT_COLLAPSE = 1e-3
#: The placed-physics monitors in words (the convergence messages and the "Stops early when" row
#: name what the verdict was taken on; fans are worded by :func:`_monitor_messages`).
MONITOR_NAMES = {"nu": "the Nusselt number", "ke": "the kinetic energy",
                 "t_rise_max": "the warmest fluid's rise above the ambient",
                 "t_rise_mean": "the mean fluid temperature's rise above the ambient"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    """Whether ``exc`` is the device (GPU or CPU) running out of memory.

    CUDA raises ``torch.OutOfMemoryError`` ("CUDA out of memory"); the CPU
    allocator raises a plain RuntimeError "[enforce fail at alloc_cpu.cpp] ...
    DefaultCPUAllocator: not enough memory", and numpy / Python a MemoryError.
    All of them must become the readable "ran out of memory" message rather than
    the allocator's own text.
    """
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    text = str(exc).lower()
    return (name == "OutOfMemoryError" or "out of memory" in text or "not enough memory" in text
            or "defaultcpuallocator" in text)


def _oom_message(plan: SimPlan) -> str:
    gpu = not plan.device.startswith("cpu")
    return (f"The {'GPU' if gpu else 'computer'} ran out of memory for the {plan.cells:,}-cell grid. Choose a "
            f"lower quality, close other programs using the {'GPU' if gpu else 'memory'}, or run on another device.")


def _gpu_failure(exc: BaseException, plan: SimPlan) -> RuntimeError | None:
    """A readable error for a GPU that fails for reasons other than memory.

    A driver reset, a device another process holds exclusively, or a broken
    CUDA context surface as RuntimeErrors full of CUDA jargon; the user needs
    to know it is the GPU, not their geometry, and what to do about it.
    """
    if plan.device.startswith("cpu") or _is_oom(exc):
        return None
    text = str(exc)
    if not any(k in text for k in ("CUDA", "cuda", "cuFFT", "CUBLAS", "cuDNN", "device-side")):
        return None
    first = text.strip().splitlines()[0][:200] if text.strip() else type(exc).__name__
    return RuntimeError(f"The GPU could not run the solver ({first}). Close other programs using the GPU and "
                        "try again; if it keeps failing, restart Aerolume or choose the CPU under Device in Setup.")


def _retry_oom(fn, plan: SimPlan):
    """Run ``fn``; on a GPU out-of-memory error free the cache and retry once.

    On a shared GPU an allocation can fail transiently while another process
    holds memory (observed during development with 6.5 GB nominally free).
    Other GPU failures are re-raised with a readable message.
    """
    try:
        return fn()
    except Exception as exc:
        if not _is_oom(exc):
            gpu = _gpu_failure(exc, plan)
            if gpu is not None:
                raise gpu from exc
            raise
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.5)
    try:
        return fn()
    except Exception as exc:
        if _is_oom(exc):
            raise RuntimeError(_oom_message(plan)) from exc
        gpu = _gpu_failure(exc, plan)
        if gpu is not None:
            raise gpu from exc
        raise


def _diverged_message(t: float, plan: SimPlan, why: str = "non-finite forces") -> str:
    return (f"The flow solution became unstable at t = {t:.3g} L/U ({why}). This happens when the grid is too "
            "coarse for the flow: choose a higher quality, set the flow model to LES, or use the 'capped' "
            f"Reynolds-number mode. (Grid {plan.grid[0]:,} × {plan.grid[1]:,}, "
            f"Re {units.fmt_reynolds(plan.re_sim)}.)")


def _uniform(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample onto a uniform time grid (same sample count) so that means are
    time-weighted and batch means / spectra are valid with a CFL-variable dt."""
    if t.size < 3:
        return t, x
    d = np.diff(t)
    if d.max() <= 1.05 * d.min():
        return t, x
    tu = np.linspace(t[0], t[-1], t.size)
    return tu, np.interp(tu, t, x)


def _judge(t: np.ndarray, x: np.ndarray, t0: float, *, rtol: float, atol: float, laminar: bool) -> dict:
    """Statistics of ``x`` over ``t >= t0`` and a verdict on its mean.

    Returns ``mean, stderr, rel_stderr, tol, ess, reliable, drift,
    amplitude_ratio, periodic, cycles, state`` ("insufficient" | "transient" |
    "converging" | "converged"), ``drifting`` (bool) and ``reason`` (one
    sentence). See the module docstring for the rules; ``tol = max(rtol |mean|,
    atol)``. ``reliable`` is batch means' own verdict on its error bar (False:
    the window is too short and ``stderr`` is a conservative estimate).
    """
    from ..analysis import batch_means, dominant_frequency
    nan = float("nan")
    ok = np.isfinite(t) & np.isfinite(x) & (t >= t0)
    tw, xw = _uniform(np.asarray(t, dtype=np.float64)[ok], np.asarray(x, dtype=np.float64)[ok])
    n = int(xw.size)
    out = {"n": n, "mean": nan, "stderr": nan, "rel_stderr": nan, "tol": nan, "ess": 0.0, "reliable": False,
           "drift": nan, "amplitude_ratio": nan, "periodic": False, "cycles": None, "state": "insufficient",
           "drifting": False, "reason": "Too few samples in the averaging window; keep running."}
    if n < MIN_SAMPLES:
        return out
    mean, rms = float(xw.mean()), float(xw.std())
    tol = max(rtol * abs(mean), atol)
    # the floor governs a mean near zero (the lift of a symmetric body): its target is then an
    # absolute ±atol, and a relative error of a near-zero mean says nothing
    out["absolute"] = bool(atol >= rtol * abs(mean))
    bm = batch_means(xw)
    se, ess = float(bm["stderr"]), float(bm["ess"])
    h = n // 2
    a, b = xw[:h], xw[n - h:]
    # "steady" (no fluctuation worth averaging) must not swallow an oscillation that is growing out
    # of the noise: below Re ~ 47 a cylinder's lift decays, above it the same tiny wiggle grows into
    # shedding, and a run that stopped on it would report the wrong flow. Round-off is exempt.
    growing = float(b.std()) > AMPLITUDE_BAND[1] * float(a.std()) and float(b.std()) > GROWTH_FLOOR * tol
    steady = rms <= 0.1 * tol and not growing
    out.update(mean=mean, stderr=se, tol=tol, ess=ess, reliable=bool(bm.get("reliable", True)) or steady,
               rel_stderr=(se / abs(mean)) if mean else math.inf)

    bma, bmb = batch_means(a), batch_means(b)
    drift = float(b.mean() - a.mean())
    se_d = math.hypot(bma["stderr"], bmb["stderr"])
    out["drift"] = drift
    if steady:
        drifting = abs(drift) > tol
    else:
        gross = abs(drift) > max(a.std(), b.std())      # a ramp: its own spread is smaller than the trend
        significant = abs(drift) > 3.0 * se_d if math.isfinite(se_d) else gross
        drifting = abs(drift) > tol and (significant or gross)
    out["drifting"] = drifting
    if drifting:
        out.update(state="transient", reason=f"The running mean is still drifting ({drift:+.3g} between the "
                                             "two halves of the window).")
        return out
    if not steady and ess < MIN_ESS:
        out["reason"] = (f"Only {ess:.0f} effectively independent samples so far (need {MIN_ESS:.0f}); "
                         "keep running.")
        return out
    if not steady:
        ra, rb = float(a.std()), float(b.std())
        ratio = rb / ra if ra > 0 else math.inf
        out["amplitude_ratio"] = ratio
        if laminar:
            changing = not (AMPLITUDE_BAND[0] <= ratio <= AMPLITUDE_BAND[1])
        else:
            # standard error of ln(rms) from n_eff samples is ~ 1 / sqrt(2 n_eff)
            se_ln = math.sqrt(0.5 / max(bma["ess"], 1.0) + 0.5 / max(bmb["ess"], 1.0))
            changing = abs(math.log(ratio)) > max(math.log(AMPLITUDE_BAND[1]), 3.0 * se_ln) \
                if 0 < ratio < math.inf else True
        if changing and ratio <= AMPLITUDE_BAND[1] and SMALL_AMPLITUDE_PEAK * max(ra, rb) <= tol:
            # A decaying wiggle whose whole envelope fits inside the tolerance cannot move the mean
            # by more than the tolerance, whatever its trend: the residue of the start-up transient
            # on a steady flow (Cl of the Re 40 cylinder at Draft: rms 0.0099 -> 0.0063 between the
            # halves against a target of ±0.083). Its standard error decides. A *growing* amplitude
            # stays a transient however small: that is how vortex shedding sets in.
            changing = False
            out["small_decaying"] = True
        # a dying wiggle is not shedding: it needs no whole cycles averaged (MIN_CYCLES), and the
        # Re 40 lift's 3.7 "cycles" of start-up residue would otherwise hold the run to the end
        f = None if out.get("small_decaying") and ratio < AMPLITUDE_BAND[0] else \
            dominant_frequency(tw, xw, t_start=float(tw[0]))
        if f is not None:
            out.update(periodic=True, cycles=float(f * (tw[-1] - tw[0])))
        if changing:
            out.update(state="transient", reason=f"The fluctuation amplitude is still changing (×{ratio:.2f} "
                                                 "between the two halves of the window).")
            return out
    known = (f"Mean known to ±{units.sig(se, 2)}" if out["absolute"] else
             f"Mean known to ±{units.sig(se, 2)} ({100 * out['rel_stderr']:.1f} %)")
    if not steady and not (se <= tol):
        out.update(state="converging", reason=f"{known}; the target ±{units.sig(tol, 2)} needs a longer average.")
        return out
    if steady:
        reason = "Steady: no fluctuation, mean settled."
    elif out.get("small_decaying"):
        reason = f"{known}; the remaining fluctuation is decaying and smaller than the target ±{units.sig(tol, 2)}."
    else:
        reason = f"{known}."
    out.update(state="converged", reason=reason)
    return out


def target_text(j: dict, rel_pct: float) -> str:
    """The tolerance a judgement (:func:`_judge`) was held to, in words: "±10 %", or for a mean near
    zero, whose target is the absolute floor, "±0.083 (absolute: the mean is near zero)"."""
    if j.get("absolute") and math.isfinite(j.get("tol", math.nan)):
        return f"±{units.sig(j['tol'], 2)} (absolute: the mean is near zero)"
    return f"±{rel_pct:g} %"


def _combined_verdict(h: dict[str, np.ndarray], plan, t_min: float, *, rtol: float, laminar: bool) -> dict:
    """The forces' verdict (:func:`_verdict`) and, for a plan with placed-physics monitors, theirs
    (:func:`.physics_run.monitor_verdict`): the run converges when both do.

    A closed box is never judged on its forces (``cd`` / ``cl`` None), with or without bodies:
    force coefficients on the box's own reference velocity mean nothing there, and the Summary
    hides them, so a verdict taken on them reads "Not converged" with no visible reason (the
    Electronics enclosure's circuit board, Cd known to only 12 %; critic round 3.4). Its monitors
    alone decide: temperatures, heat rates, fans and the kinetic energy (:func:`.physics.plan_monitors`)."""
    bodies = bool(getattr(plan.solver_config, "bodies", None))
    closed = getattr(plan, "domain_mode", "tunnel") == "closed"
    monitors = tuple(getattr(plan, "monitors", ()) or ())
    if (bodies and not closed) or not monitors:
        v = _verdict(h, t_min, rtol=rtol, laminar=laminar)
    else:
        v = {"t0": t_min, "stop": True, "drifting": False, "cd": None, "cl": None, "periodic": False,
             "cycles": None, "enough_cycles": True}
    if monitors:
        from .physics_run import monitor_verdict
        mv = monitor_verdict(h, plan, v["t0"], rtol=rtol, laminar=laminar)
        v["monitors"] = mv
        v["stop"] = bool(v["stop"] and mv["stop"])
        v["drifting"] = bool(v["drifting"] or mv["drifting"])
    return v


def _verdict(h: dict[str, np.ndarray], t_min: float, *, rtol: float, laminar: bool) -> dict:
    """Where the averaging window starts and whether its statistics are converged.

    ``stop`` is the early-stop decision and, at the end of a run, the meaning
    of ``CFDResult.converged``.
    """
    from ..analysis import transient_end
    t, cd, cl = h["t"], h["cd"], h["cl"]
    blank = {"t0": t_min, "stop": False, "drifting": False, "cd": None, "cl": None, "periodic": False,
             "cycles": None}
    if int((t >= t_min).sum()) < MIN_SAMPLES:
        return blank
    t0 = max(transient_end(t, cd, t_min=t_min), transient_end(t, cl, t_min=t_min))
    if not math.isfinite(t0) or (t[-1] - t0) < 0.25 * (t[-1] - t_min):
        t0 = t_min                              # transient "never ends": judge the whole window
    jcd = _judge(t, cd, t0, rtol=rtol, atol=1e-3, laminar=laminar)
    cd_scale = abs(jcd["mean"]) if math.isfinite(jcd["mean"]) else 0.0
    # Cl of a body without lift averages to ~0: hold it to the same absolute
    # precision as the drag instead of a relative error of zero
    jcl = _judge(t, cl, t0, rtol=CL_RTOL_FACTOR * rtol, atol=max(1e-3, rtol * cd_scale), laminar=laminar)
    periodic = bool(jcl["periodic"] or jcd["periodic"])
    cycles = jcl["cycles"] if jcl["periodic"] else jcd["cycles"]
    enough_cycles = (not periodic) or (cycles is not None and cycles >= MIN_CYCLES)
    stop = jcd["state"] == "converged" and jcl["state"] == "converged" and enough_cycles
    return {"t0": float(t0), "stop": bool(stop), "drifting": bool(jcd["drifting"]), "cd": jcd, "cl": jcl,
            "periodic": periodic, "cycles": cycles, "enough_cycles": enough_cycles}


def _scale(name: str, plan: SimPlan) -> float:
    from ..viz import field_info
    base = name.removesuffix("_rms")
    return field_info(base).scale(plan.speed, plan.fluid.density, plan.length_ref)


def _field_data(fields: dict, names: tuple[str, ...], plan: SimPlan, mask, label: str) -> FieldData:
    """Solver fields -> FieldData in SI, solver row order (row 0 = bottom, y-min)."""
    out = {}
    for k in names:
        if k in fields:
            out[k] = (fields[k].double() * _scale(k, plan)).float().cpu().numpy()
    m = mask.float().cpu().numpy() if mask is not None else None
    return FieldData(extent=tuple(float(e) for e in plan.extent), fields=out, mask=m, label=label)


def _surface(solver, plan: SimPlan, mean: bool) -> list[SurfaceData]:
    """Surface cp / cf per body in metres (about two samples per grid cell)."""
    L = plan.length_ref
    h = plan.solver_config.h
    out = []
    per = [float(np.hypot(*np.diff(np.vstack([b.points, b.points[:1]]), axis=0).T).sum())
           for b in plan.solver_config.bodies]
    n = int(np.clip(2.0 * max(per, default=0.0) / h, 200, 2000))
    for d in solver.surface(n, mean=mean):
        out.append(SurfaceData(body=str(d["name"]), s=d["s"] * L, x=d["x"] * L, y=d["y"] * L,
                               nx=d["nx"], ny=d["ny"], cp=d["cp"], cf=d["cf"]))
    return out


def _pct(rel: float) -> str:
    """A relative error as '4.3 %' ('—' when undefined)."""
    return f"{units.sig(100 * rel, 2)} %" if math.isfinite(rel) else "—"


class _Progress:
    """Wraps the caller's progress callback (None-safe, clamped, monotone)."""

    def __init__(self, fn: ProgressFn | None):
        self.fn = fn
        self.last = 0.0

    def __call__(self, frac: float, msg: str) -> None:
        if self.fn is None:
            return
        frac = max(self.last, min(1.0, max(0.0, float(frac))))
        self.last = frac
        self.fn(frac, msg)


# ---------------------------------------------------------------------------
# CFD
# ---------------------------------------------------------------------------

def run_cfd(plan: SimPlan, *, progress: ProgressFn | None = None, frame: FrameFn | None = None,
            cancel: threading.Event | None = None, view: ViewSettings | None = None) -> CFDResult:
    """Run a planned Navier-Stokes simulation and return its :class:`CFDResult`.

    ``progress(fraction, message)`` is called after every chunk; ``frame`` gets a
    :class:`LiveFrame` per chunk at about ``view.fps`` frames per second (chunks
    are sized so stepping plus rendering fill one frame interval), the last one
    with ``phase="done"``. ``view`` may be mutated by another thread while the
    run goes. ``cancel`` stops the run between chunks and returns the partial
    result (``warnings`` say so). Raises ``ValueError`` for a live plan and
    ``RuntimeError`` when the flow blows up or the device runs out of memory.

    A 3D plan (:class:`~aerolume.simulation.plan3d.SimPlan3D`) is run by
    :func:`aerolume.simulation.runner3d.run_cfd3d` and returns a
    :class:`~aerolume.model.CFD3DResult` (DESIGN.md 7.7).
    """
    if getattr(plan, "dimension", 2) == 3:
        from .runner3d import run_cfd3d
        return run_cfd3d(plan, progress=progress, frame=frame, cancel=cancel, view=view)  # type: ignore[return-value]
    from ..solvers.ns2d import FlowSolver
    if plan.live or not math.isfinite(plan.t_end):
        raise ValueError("This plan is for an explore session (no end time); plan the run with plan_simulation().")
    wall0 = time.perf_counter()
    view = view if view is not None else ViewSettings()
    report = _Progress(progress)
    cfg = plan.solver_config
    L, U, rho = plan.length_ref, plan.speed, plan.fluid.density
    notes, warnings = list(plan.notes), list(plan.warnings)
    t_end_planned = t_end = float(plan.t_end)
    t_avg = float(plan.t_avg_start)
    user_duration = bool(plan.sim.cfd.duration)
    want_sens = bool(getattr(plan.sim.cfd, "compute_sensitivity", False))
    share = STEPPING_SHARE_SENS if want_sens else STEPPING_SHARE
    laminar = cfg.turbulence == "none"
    rtol = float(getattr(plan, "conv_rtol", 0.01))
    bodies_xy = [np.asarray(b.points, dtype=np.float64) for b in plan.bodies_wind]
    is_cuda = not plan.device.startswith("cpu")

    report(0.0, f"Building the solver ({cfg.nx:,} × {cfg.ny:,} cells on {plan.device_name})")
    from .physics_run import MonitorLog, chunk_samples
    monlog = MonitorLog()
    bodies_on = bool(cfg.bodies)
    with cpu_threads(plan.device):
        solver = _retry_oom(lambda: FlowSolver(cfg), plan)
        if starts_unbounded(cfg):
            start_from_rest(solver, cfg, plan.dt)
        renderer = FrameRenderer(plan.extent, length=L, speed=U, density=rho, h=cfg.h, focus=plan.focus_extent,
                                 physics=getattr(plan, "physics", None), placed=getattr(plan, "placed", None))
        renderer.plan = plan
        rate = estimate_steps_per_second(plan.device, plan.cells, advection=cfg.advection,
                                         turbulence=cfg.turbulence, dtype=cfg.dtype) / float(getattr(plan, "step_cost", 1.0))
        averaging = False
        t_field_avg = math.nan
        samples_t: list[float] = []
        samples_b: list[np.ndarray] = []
        step_wall = 0.0
        steps_timed = 0
        loop_wall = 0.0                         # whole loop iterations after the first (for the estimate)
        first = True
        dt_last = plan.dt
        last_check = -math.inf
        last_frame = -math.inf
        render_s = 0.0                          # smoothed wall time of one frame render
        wall_rate = math.nan                    # convective times per wall second, all included
        mark = None                             # (wall, t) of the previous chunk, for wall_rate
        verdict: dict | None = None
        stopped_early = cancelled = restarted = False
        t_first = None
        prog_base = (0.0, 0.0)                  # (fraction, t) the bar is extrapolated from

        while solver.t < t_end - 1e-12:
            it0 = time.perf_counter()
            timed = not first
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            if frame is not None:
                # leave room for the render so frames really arrive at view.fps, but never
                # let rendering take more than RENDER_DUTY of the wall time (a slow render
                # on a big grid lowers the frame rate instead of halving the solver speed)
                interval = max(1.0 / max(float(view.fps or 20.0), 1.0), render_s / RENDER_DUTY)
                budget = interval - render_s
            else:
                interval = budget = HEADLESS_CHUNK_S
            n = int(max(1, min(5000, rate * budget)))
            if first:
                # short: it pays for the graph capture, and the planned dt it is sized with is
                # only an estimate (the real start-up step is larger), so a long first chunk
                # could run straight past the start of the averaging window
                n = min(n, FIRST_CHUNK_STEPS)
            target = t_avg if not averaging else t_end
            if dt_last > 0:
                n = min(n, max(1, math.ceil((target - solver.t) / dt_last)))
            if is_cuda and n >= 20:
                n -= n % 10                     # whole CUDA-graph blocks
            ts = time.perf_counter()
            if first:                           # includes graph capture: the riskiest allocation
                _retry_oom(lambda k=n: solver.advance(k), plan)
            else:
                try:
                    solver.advance(n)
                except Exception as exc:
                    gpu = _gpu_failure(exc, plan) or (RuntimeError(_oom_message(plan)) if _is_oom(exc) else None)
                    if gpu is not None:
                        raise gpu from exc
                    raise
            el = max(time.perf_counter() - ts, 1e-9)
            t = solver.t
            if t_first is None:
                t_first = t
            dt_last = solver.dt
            if not first:                       # the first chunk includes the graph capture
                step_wall += el
                steps_timed += n
                rate = 0.7 * rate + 0.3 * (n / el)
            first = False
            c = solver.coefficients()
            if not np.all(np.isfinite(c)) or np.abs(c).max() > 1e6:
                raise RuntimeError(_diverged_message(t, plan))
            if not (dt_last > DT_COLLAPSE * plan.dt):
                raise RuntimeError(_diverged_message(t, plan, "the time step collapsed: the velocity is "
                                                              "growing without bound"))
            sample = chunk_samples(solver, plan)
            if sample:
                if not all(math.isfinite(v) for v in sample.values()):
                    raise RuntimeError(_diverged_message(t, plan, "non-finite flow or temperature"))
                monlog.add(t, sample)
            if averaging:
                samples_t.append(t)
                samples_b.append(solver.body_coefficients())
            elif t >= t_avg - 1e-12:
                solver.start_averaging()
                averaging, t_field_avg = True, t

            now = time.perf_counter()
            win = t_end - max(t_avg, t_field_avg) if averaging else t_end - t_avg
            if averaging and now - last_check >= CHECK_PERIOD_S and t - t_field_avg >= MIN_AVG_FRACTION * win:
                last_check = now
                verdict = _combined_verdict(history_now(solver, plan, monlog), plan, t_field_avg, rtol=rtol,
                                            laminar=laminar)
                if verdict["stop"]:
                    stopped_early = True
                    loop_wall += time.perf_counter() - it0 if timed else 0.0
                    break
                if (verdict["drifting"] and not restarted and not user_duration
                        and t - t_avg < 0.6 * win and t_end < MAX_EXTENSION * t_end_planned):
                    # the start-up transient lasted longer than planned: restart the field
                    # averages now and give the averaging window the time it lost
                    solver.start_averaging()
                    restarted = True
                    verdict = None                  # its numbers describe the discarded window
                    lost = t - t_field_avg
                    t_field_avg = t
                    samples_t.clear()
                    samples_b.clear()
                    # the bar continues from where it is instead of freezing until t catches up
                    prog_base = (report.last, t)
                    t_end = min(t_end + lost, MAX_EXTENSION * t_end_planned)
                    notes.append(f"The start-up transient lasted longer than planned (the "
                                 f"{'mean drag was' if bodies_on else 'monitored values were'} still drifting at "
                                 f"t = {t:.3g}), so averaging restarted there and the run was extended to "
                                 f"t = {t_end:.4g}.")

            # progress + frame
            if mark is not None and now > mark[0] and t > mark[1]:
                r = (t - mark[1]) / (now - mark[0])
                wall_rate = r if not math.isfinite(wall_rate) else 0.8 * wall_rate + 0.2 * r
            mark = (now, t)
            refine_estimate(plan, elapsed=now - wall0, t=t, t_end=t_end, wall_rate=wall_rate)
            f0, t0p = prog_base
            frac = f0 + (share - f0) * (t - t0p) / max(t_end - t0p, 1e-12)
            eta = (t_end - t) / wall_rate + 1.0 if math.isfinite(wall_rate) and wall_rate > 0 else None
            msg = progress_message(t, t_end, averaging, verdict, bodies_on)
            report(frac, msg)
            if frame is not None and now - last_frame >= 0.8 * interval:
                last_frame = now
                tr = time.perf_counter()
                frame(renderer.render(solver, view, bodies=bodies_xy,
                                      phase="averaging" if averaging else "transient", progress=frac,
                                      eta_s=eta, steps_per_s=rate))
                render_s = 0.7 * render_s + 0.3 * (time.perf_counter() - tr)
            if timed:
                loop_wall += time.perf_counter() - it0

        # ----------------------------------------------------------------- outputs
        report(share, "Collecting forces, surface and field data")
        t_final = solver.t
        h = history_now(solver, plan, monlog)
        _record_measurements(plan, steps_timed=steps_timed, step_wall=step_wall, loop_wall=loop_wall,
                             t_first=t_first, t_final=t_final, t_end_planned=t_end_planned,
                             live_view=frame is not None, solver=solver)
        result = _assemble(plan, solver, h, averaging=averaging, t_field_avg=t_field_avg,
                           samples_t=samples_t, samples_b=samples_b, notes=notes, warnings=warnings,
                           cancelled=cancelled, stopped_early=stopped_early, t_end=t_end, rtol=rtol,
                           laminar=laminar)
        if not cancelled and _wants_polar_reference(plan):
            report(share, "Comparing with the viscous polar at the same Reynolds number")
            _polar_reference(plan, result, cancel)
        # the final frame shows the flow the result describes: render it before the
        # sensitivity sweep touches the GPU, deliver it last
        final = (renderer.render(solver, view, bodies=bodies_xy, phase="done", progress=1.0, eta_s=0.0,
                                 steps_per_s=rate, history=h) if frame is not None else None)
        if want_sens:
            if cancelled:
                result.warnings.append("The shape-sensitivity map was skipped because the run was stopped.")
            else:
                _attach_sensitivity(plan, solver, result, report, cancel, share)
        result.wall_seconds = time.perf_counter() - wall0
        if final is not None:
            frame(final)
    report(1.0, final_message(result, cancelled))
    return result


def _stat_value(st: Stat, digits: int) -> str:
    """'2.251' with its error bar when it means something: none for a steady value (no fluctuation
    to average) or an error far below the digits shown."""
    s = units.sig(st.mean, digits)
    se = float(st.stderr)
    if math.isfinite(se) and se > 1e-6 * abs(st.mean):
        s += f" ± {units.sig(se, 2)}"
    return s


#: Steps a box that nothing bounds at rest (:func:`starts_unbounded`) takes at the planned step.
START_STEPS = 10


def starts_unbounded(cfg) -> bool:
    """Whether the 2D solver's first step from rest has no stability bound but the viscous one: a
    closed, buoyant box heated by sources or heat fluxes, with no jet, vent or speed- or pressure-
    controlled fan (the solver's source speed floor). At rest the advective CFL bound is
    infinite, the buoyant-parcel bound ``cfl sqrt(h / (Ri |θ|max))`` needs θ > 0, and nothing
    bounds the heat a source adds in one step: the Draft heater-only room (96 × 72, Re 2×10⁴) took
    a first step of 56 L/U, heated its heater cells to θ 10⁵ and diverged (fix round 3.4; the
    solver's own bound is a contract request to the 2D solver)."""
    if getattr(cfg, "domain", "tunnel") != "closed" or getattr(cfg, "dt", None) is not None:
        return False
    th = getattr(cfg, "thermal", None)
    if th is None or not float(getattr(th, "richardson", 0.0) or 0.0):
        return False
    if getattr(cfg, "ports", None) or any(getattr(f, "mode", "thrust") != "thrust" for f in getattr(cfg, "fans", ())):
        return False
    walls = getattr(th, "walls", None) or []
    flux = any(getattr(w, "mode", "") == "heat_flux" for w in walls)
    flux = flux or any(getattr(b, "thermal", "adiabatic") == "heat_flux" for b in getattr(cfg, "bodies", ()))
    return bool(getattr(cfg, "heat_sources", None)) or flux


def start_from_rest(solver, cfg, dt: float, steps: int = START_STEPS) -> None:
    """Take the first ``steps`` steps of ``solver`` at the fixed step ``dt`` (the plan's, sized for the
    flow the box will reach), then hand it the state: once the heaters have warmed their cells the
    solver's own buoyant-parcel bound holds the step. Uses a separate eager solver with
    ``dt`` fixed, because a solver's step rule is fixed when it is built."""
    import dataclasses

    from ..solvers.ns2d import FlowSolver
    warm = FlowSolver(dataclasses.replace(cfg, dt=float(dt), use_cuda_graphs=False))
    warm.advance(int(steps))
    solver.load_state_dict(warm.state_dict())
    del warm


def final_message(result, cancelled: bool) -> str:
    """The last progress line of a run (2D and 3D): the headline number of what was simulated --
    the drag, else the Nusselt number, else the first fan's thrust -- or why there is none."""
    head = ("Stopped" if cancelled else "Done") + f" after {fmt_duration(result.wall_seconds)}"
    cd = result.cd
    if math.isfinite(cd.mean):
        return f"{head}: Cd = {_stat_value(cd, 4)}"
    nu = getattr(result, "nusselt", None)
    if nu is not None and math.isfinite(nu.mean):
        return f"{head}: Nu = {_stat_value(nu, 4)}"
    fans = getattr(result, "fans_report", None) or []
    if fans and math.isfinite(float(fans[0].get("thrust", math.nan))):
        f = fans[0]
        unit = "N" if len(getattr(result, "grid", ()) or ()) == 3 else "N/m"
        return f"{head}: fan “{f.get('name', '')}” thrust {units.sig(float(f['thrust']), 3)} {unit}"
    return f"{head}: " + ("no bodies, so no forces" if not result.per_body else "no averaged forces")


def history_now(solver, plan, monlog=None) -> dict[str, np.ndarray]:
    """The run's history so far: the forces and, with placed physics, the physics series in SI
    aligned to the force samples (:func:`.physics_run.merge_series`)."""
    h = solver.history()
    if getattr(plan, "placed", None) is None:
        return h
    from .physics_run import merge_series, physics_history_si
    return merge_series(h, physics_history_si(solver, plan), monlog.series() if monlog is not None else {})


def refine_estimate(plan, *, elapsed: float, t: float, t_end: float, wall_rate: float) -> None:
    """Replace ``plan.est_seconds`` by the measured re-estimate: the wall time so far, the
    remaining convective time at the measured rate (``wall_rate``, convective units per wall
    second over the timed blocks: the steps per second and the current time step together, the
    frames included), and what the plan expects after the loop (outputs, the sensitivity map).

    The page's "Remaining" shows the plan's estimate minus the elapsed time until the run has
    measured itself, so a plan whose time step was too optimistic promised 5 min 30 s for a 3D
    room that needed about 20 min (critic round 3.2); from the second timed block on it reads
    this one. ``est_seconds_planned`` keeps the planner's own number.
    """
    if getattr(plan, "live", False) or not (math.isfinite(wall_rate) and wall_rate > 0):
        return
    after = float(getattr(plan, "est_after_seconds", 0.0) or 0.0)
    plan.est_seconds = float(elapsed) + max(float(t_end) - float(t), 0.0) / wall_rate + after + 1.0


def progress_message(t: float, t_end: float, averaging: bool, verdict: dict | None, bodies: bool) -> str:
    """The progress line: what the run is doing and, once judged, the value it is converging to."""
    if not averaging:
        return f"Developing the flow (t = {t:.1f} of {t_end:.4g})"
    msg = (f"Averaging forces (t = {t:.1f} of {t_end:.4g})" if bodies else
           f"Settling (t = {t:.1f} of {t_end:.4g})")
    cdv = verdict.get("cd") if verdict is not None else None
    if cdv and math.isfinite(cdv["mean"]):
        rel = cdv["rel_stderr"]
        msg += f" · Cd ≈ {units.sig(cdv['mean'], 3)}" + (f" ± {100 * rel:.1f} %" if math.isfinite(rel) else "")
    mv = (verdict or {}).get("monitors") or {}
    nu = (mv.get("judged") or {}).get("nu")
    if nu and math.isfinite(nu.get("mean", math.nan)):
        msg += f" · Nu ≈ {units.sig(nu['mean'], 4)}"
    return msg


def _strouhal(plan: SimPlan, t: np.ndarray, cl: np.ndarray, t0: float, *,
              cd_mean: float) -> tuple[float | None, str | None]:
    """The Strouhal number of the lift oscillation when it is a real result, else
    None -- and, when a periodic oscillation was rejected for a reason the user
    should know, a note saying why.

    Every run's lift wiggles, and the dominant spectral peak of any wiggle has a
    frequency; quoting it as "St" invites comparisons of noise (Compare showed a
    -58 % "change" between two airfoil runs). A frequency is reported only when

    * the peak holds at least ST_MIN_PEAK_SHARE of the fluctuation energy (a
      clear line, not a broadband hump; the detector alone accepts 20 %);
    * the oscillation is not negligible: its rms is at least ST_MIN_RMS of
      max(|mean Cl|, mean Cd) (a laminar airfoil settling to a steady state
      still rings at ~0.5 % of its lift);
    * at least MIN_CYCLES cycles were averaged (a frequency read off two
      cycles of a slow drift, the Draft sedan's "St 0.034", is not a
      measurement);
    * the flow it comes from is meaningful: for an airfoil whose boundary
      layers the grid does not resolve (the plan's "not reliable" warning) the
      lift oscillation belongs to the under-resolved flow -- NACA 2412 at Re 1e6
      and 4° rings at St 1.66 per chord at Draft, where the real attached
      airfoil sheds nothing coherent -- so it is not reported either.
    """
    from ..analysis import dominant_frequency
    ok = np.isfinite(t) & np.isfinite(cl) & (t >= t0)
    tw, xw = _uniform(np.asarray(t, dtype=np.float64)[ok], np.asarray(cl, dtype=np.float64)[ok])
    if tw.size < MIN_SAMPLES:
        return None, None
    f = dominant_frequency(tw, xw, t_start=float(tw[0]), min_fraction=ST_MIN_PEAK_SHARE)
    if f is None or not (f > 0):
        return None, None
    resid = xw - np.polyval(np.polyfit(tw - tw[0], xw, 1), tw - tw[0])
    scale = max(abs(float(xw.mean())), abs(cd_mean) if math.isfinite(cd_mean) else 0.0, 1e-6)
    if float(resid.std()) < ST_MIN_RMS * scale:
        return None, None
    if plan.geometry_class == "airfoil" and not getattr(plan, "bl_resolved", True):
        return None, ("The lift oscillates, but at this Reynolds number the grid does not resolve the boundary "
                      "layers, so the oscillation belongs to the under-resolved flow, not to the real airfoil; "
                      "no Strouhal number is reported. A capped Reynolds number gives a flow the grid resolves.")
    cycles = f * float(tw[-1] - tw[0])
    if cycles < MIN_CYCLES:
        return None, (f"The lift oscillates, but only {cycles:.1f} cycles were averaged (at least "
                      f"{MIN_CYCLES:.0f} are needed to measure a frequency); no Strouhal number is reported. "
                      "A longer Duration would measure it.")
    return float(f * plan.strouhal_length / plan.length_ref), None


def _mark_steady_errors(stats: dict[str, Stat], verdict: dict | None, rtol: float) -> None:
    """Keep the error bars of a settled coefficient marked reliable.

    ``coefficient_stats`` flags a standard error as unreliable whenever the
    window holds too few correlation lengths for batch means. A steady flow
    always looks like that (its residual relaxation is one long correlation), but
    there the conservative error estimate (half the drift between the window
    halves, at most) is exactly the uncertainty that remains, so flagging it
    would make the Compare page and the report doubt a converged steady run. Cd
    and Cl take the convergence verdict's own judgement (which treats a
    negligible fluctuation as steady); Cm, which the verdict does not judge,
    uses the same steadiness rule directly.
    """
    judged = {k: verdict.get(k) for k in ("cd", "cl")} if verdict else {}
    for key, st in stats.items():
        if st.stderr_reliable:
            continue
        j = judged.get(key)
        if j is not None:
            st.stderr_reliable = bool(j.get("reliable"))
        elif math.isfinite(st.rms) and math.isfinite(st.mean):
            st.stderr_reliable = bool(st.rms <= 0.1 * max(rtol * abs(st.mean), 1e-3))


def _record_measurements(plan: SimPlan, *, steps_timed: int, step_wall: float, loop_wall: float,
                         t_first: float | None, t_final: float, t_end_planned: float, live_view: bool,
                         solver=None) -> None:
    """Feed this run's throughput, loop share and speed factor back into the planner.

    Only measurements that describe a whole run are kept: a run shorter than
    MIN_TIMED_S of stepping says more about start-up than about speed, and the
    speed factor of a run that covered less than half its planned duration
    misses the developed wake (where the time step is smallest: 3.5 while the
    Standard cylinder's wake forms, 4.0-4.3 once it sheds), so it is not kept.
    """
    if steps_timed <= 0 or step_wall < MIN_TIMED_S or t_first is None or getattr(plan, "live", False):
        # (explore sessions never feed the calibration: their obstacles, sliders and placed items
        # change while they run, so their speeds describe no plan)
        return
    cfg = plan.solver_config
    mean_dt = (t_final - t_first) / steps_timed
    whole = (t_final - t_first) >= 0.5 * t_end_planned
    sf = cfg.cfl * cfg.h / mean_dt if (mean_dt > 0 and whole) else None
    if getattr(plan, "placed", None) is not None:
        # another bound may set the step: read the speeds themselves (physics_run.measured_speed_factor)
        from .physics_run import measured_speed_factor
        sf = measured_speed_factor(solver, plan, t_first) if (whole and solver is not None) else None
    if sf is not None:
        # the plan multiplies the class factor by its peak-speed ratio; record it per peak speed
        sf = sf / max(float(getattr(plan, "speed_ratio", 1.0) or 1.0), 1.0)
    eff = step_wall / loop_wall if loop_wall > step_wall > 0 else None
    # the throughput is stored as the plain solver's (the placed physics' measured cost divided
    # out), and the speed factor under the plan's own key: a heated or closed run's time step is
    # set by its physics, and must not move the estimate of plain runs of the same class
    kind = getattr(plan, "calibration_kind", "") or getattr(plan, "geometry_class", plan.design.kind)
    record_throughput(plan.device, plan.cells, steps_timed / step_wall * float(getattr(plan, "step_cost", 1.0)),
                      advection=cfg.advection, turbulence=cfg.turbulence, dtype=cfg.dtype, kind=kind,
                      speed_factor=sf, cells_per_ref=plan.cells_per_ref, loop_efficiency=eff, live_view=live_view)


def _assemble(plan: SimPlan, solver, h: dict[str, np.ndarray], *, averaging: bool, t_field_avg: float,
              samples_t: list[float], samples_b: list[np.ndarray], notes: list[str], warnings: list[str],
              cancelled: bool, stopped_early: bool, t_end: float, rtol: float, laminar: bool) -> CFDResult:
    """Statistics, dimensional values, surface data and fields of a finished (or stopped) run."""
    from ..analysis import coefficient_stats
    cfg = plan.solver_config
    L, U, rho = plan.length_ref, plan.speed, plan.fluid.density
    t = h["t"]
    t_now = float(t[-1]) if t.size else 0.0
    bodies_on = bool(cfg.bodies)
    if averaging:
        v = _combined_verdict(h, plan, t_field_avg, rtol=rtol, laminar=laminar)
        t0 = v["t0"]
    else:
        v = None
        t0 = float(t[0] + 0.5 * (t[-1] - t[0])) if t.size else 0.0
    if bodies_on:
        stats = {k: (coefficient_stats(t, h[k], t0) if t.size else Stat()) for k in ("cd", "cl", "cm")}
        _mark_steady_errors(stats, v, rtol)
        st, st_note = _strouhal(plan, t, h["cl"], t0, cd_mean=stats["cd"].mean) if t.size else (None, None)
    else:
        # a closed box without bodies has no forces: nan, not a measured zero
        stats = {k: Stat() for k in ("cd", "cl", "cm")}
        st, st_note = None, None
    converged = bool(v is not None and v["stop"])
    _monitor_messages(v, plan, notes, warnings, converged=converged, cancelled=cancelled, t_now=t_now,
                      t_end=t_end, rtol=rtol)

    # what the numbers are worth, in words
    if cancelled:
        warnings.append(f"Stopped by the user at t = {t_now:.3g} of {t_end:.4g} L/U: "
                        + ("these values are averaged over a shorter window than planned and are not converged."
                           if averaging else "the flow had not finished developing, so the values are "
                                             "a rough indication only (last half of the record)."))
    if v is not None and v.get("cd"):
        jcd, jcl = v["cd"], v["cl"]
        if converged:
            cyc = f" over {v['cycles']:.0f} shedding cycles" if v.get("periodic") and v.get("cycles") else ""
            notes.append(f"{'Stopped early at' if stopped_early else 'Converged by'} t = {t_now:.3g} of "
                         f"{t_end:.4g} L/U: mean Cd known to ±{_pct(jcd['rel_stderr'])} and mean Cl to "
                         f"±{units.sig(jcl['stderr'], 2)}{cyc} (targets {target_text(jcd, 100 * rtol)} and "
                         f"{target_text(jcl, 100 * CL_RTOL_FACTOR * rtol)}, one standard error from batch means); no drift "
                         "between the halves of the averaging window.")
        elif not cancelled and not (jcd["state"] == "converged" and jcl["state"] == "converged"
                                    and v.get("enough_cycles", True) and v.get("monitors")):
            # (forces settled but a placed-physics monitor did not: _monitor_messages says which)
            worst = jcd if jcd["state"] != "converged" else jcl
            what = "Cd" if worst is jcd else "Cl"
            target = target_text(worst, 100 * rtol * (1.0 if worst is jcd else CL_RTOL_FACTOR))
            reason = worst["reason"] if worst["state"] != "converged" else (
                f"Fewer than {MIN_CYCLES:.0f} shedding cycles averaged.")
            if jcd.get("reliable") and jcl.get("reliable"):
                advice = ("The averages are usable with the error bars shown; run longer (Duration) before "
                          "relying on differences smaller than about twice them.")
            else:
                # batch means had too few correlation lengths: the error bars are estimates
                # (analysis.stats.batch_means), and a Draft NACA 4412 averaged over 20 L/U
                # missed its converged mean by 4 of the error bars this sentence used to vouch for
                n_eff = min(jcd.get("ess", 0.0), jcl.get("ess", 0.0))
                advice = (f"The error bars are rough estimates, not measurements: only {n_eff:.0f} effectively "
                          "independent samples were averaged, and the mean may still contain start-up "
                          "transient. Run longer (Duration) before comparing designs with this result.")
            warnings.append(f"Not converged by t = {t_now:.3g} L/U ({what}, target {target}: {reason}) "
                            + advice)
        notes.append(f"Forces averaged over t = {t0:.3g} … {t_now:.3g} L/U"
                     + (f" (the start-up transient was found to end at t = {t0:.3g}; field averages start at "
                        f"t = {t_field_avg:.3g})" if abs(t0 - t_field_avg) > 1e-6 else "") + ".")
    if st is not None:
        ref = "the chord" if plan.ref_label == "chord" else "the frontal height"
        notes.append(f"Strouhal number based on {ref} ({units.sig(plan.strouhal_length, 3)} m).")
    elif st_note:
        notes.append(st_note)

    q = 0.5 * rho * U * U
    fr = plan.design.force_ref
    cdm, clm, cmm = stats["cd"].mean, stats["cl"].mean, stats["cm"].mean
    drag = cdm * q * fr
    names = [b.name for b in cfg.bodies]
    per_body: dict[str, dict[str, float]] = {}
    if len(names) == 1:
        per_body[names[0]] = {"cd": cdm, "cl": clm, "cm": cmm}
    elif names:
        ts = np.asarray(samples_t)
        sel = ts >= t0 if ts.size else np.zeros(0, bool)
        if sel.sum() >= 2:
            m = np.asarray(samples_b)[sel].mean(axis=0)
            notes.append(f"Per-body coefficients are means of {int(sel.sum())} samples spread over the averaging "
                         "window; the totals use every time step.")
        else:
            m = solver.body_coefficients()
            warnings.append("Per-body coefficients are instantaneous values (too few samples to average).")
        for i, nm in enumerate(names):
            per_body[nm] = {"cd": float(m[i, 0]), "cl": float(m[i, 1]), "cm": float(m[i, 2])}

    surface: list[SurfaceData] = []
    mean_fd = inst_fd = None
    try:
        surface = _surface(solver, plan, mean=averaging)
    except Exception as exc:  # noqa: BLE001 - outputs are best effort, forces are what matter
        warnings.append(f"Surface distributions could not be extracted ({type(exc).__name__}: {exc}).")
    scales = getattr(plan, "physics", None)
    thermal_on = scales is not None and scales.thermal and getattr(solver.state, "theta", None) is not None
    n_dye = int(getattr(cfg, "dye_channels", 1) or 0)
    try:
        mask = solver.field("mask")
        if averaging:
            mf = solver.mean_fields()
            names_m = ("u", "v", "speed", "p", "cp", "vorticity", "u_rms", "v_rms", "p_rms", "cp_rms")
            if thermal_on and "theta" in mf:
                # °C and K: FieldData's "temperature" is in °C already (viz.colormaps.FieldInfo)
                mf["temperature"] = scales.celsius(mf["theta"].double())
                mf["temperature_rms"] = mf["theta_rms"].double() * scales.delta_t
                names_m += ("temperature", "temperature_rms")
            mean_fd = _field_data(mf, names_m, plan, mask, "mean")
        names_i = ("u", "v", "speed", "p", "cp", "vorticity", "dye") + (("nut",) if cfg.turbulence != "none" else ())
        names_i += tuple(f"dye{k}" for k in range(1, n_dye))
        inst = {k: solver.field(k) for k in names_i}
        if thermal_on:
            inst["temperature"] = scales.celsius(solver.field("theta").double())
            names_i += ("temperature",)
        inst_fd = _field_data(inst, names_i, plan, mask, "instant")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Flow fields could not be extracted ({type(exc).__name__}: {exc}).")

    result = CFDResult(
        sim_id=plan.sim.id, cd=stats["cd"], cl=stats["cl"], cm=stats["cm"],
        drag=drag, lift=clm * q * fr, moment=cmm * q * fr * L, drag_area=cdm * fr, power=drag * U,
        strouhal=st, strouhal_length=float(plan.strouhal_length), re_real=plan.re_real,
        re_sim=float(solver.cfg.re), grid=(cfg.nx, cfg.ny),
        cells_per_ref=plan.cells_per_ref, t_end=t_now, t_avg_start=float(t0), converged=converged,
        device=plan.device_name, precision=plan.precision, turbulence=plan.turbulence,
        notes=notes, warnings=warnings, per_body=per_body, bodies_wind=list(plan.bodies_wind),
        history={k: np.asarray(v, dtype=np.float64) for k, v in h.items()},
        surface=surface, mean_fields=mean_fd, instant_fields=inst_fd)
    attach_physics(result, solver, plan, h, t0, averaging)
    return result


#: Rows of the solver's diagnostics table a result keeps (the app's live log stores the same number,
#: ``app.widgets.diagnostics.STORED_ROWS``): enough for the history plots, a few kB on disk.
DIAG_HISTORY_ROWS = 200


def attach_diagnostics_history(result, solver, n: int = DIAG_HISTORY_ROWS) -> bool:
    """Store the solver's diagnostics table, decimated to at most ``n`` evenly spaced rows (first and last
    kept), as ``result.diagnostics_history``; returns whether it did.

    Without it a result run from a script or the CLI kept only the last diagnostics row, so a reopened
    result showed "1 sample" and empty history plots (critic round 3.4); the app attached its own live
    log, the API did not. The project store saves the table (``project.results._pack_diagnostics``).
    Only columns as long as ``step`` are kept (the 2D solver's per-call throughput columns are not
    rows of the table). A table already on the result is kept; fewer than two rows add nothing."""
    if getattr(result, "diagnostics_history", None) or not hasattr(solver, "diagnostics"):
        return False
    try:
        table = solver.diagnostics() or {}
        m = len(table.get("step", ()))
        if m < 2:
            return False
        idx = np.unique(np.linspace(0, m - 1, min(int(n), m)).round().astype(int))
        result.diagnostics_history = {k: np.asarray(v, dtype=np.float64)[idx] for k, v in table.items()
                                      if len(v) == m}
        return True
    except Exception:  # noqa: BLE001 - a result without its history is still a result
        return False


def attach_physics(result, solver, plan, h: dict, t0: float, averaging: bool) -> None:
    """Fill a result's placed-physics outputs (2D and 3D): the Nusselt number, the heat rate, the
    per-body heat transfer, the fans in SI and the diagnostics (the solver's last row plus the
    heat balance in W or W/m). Best effort: a failure becomes a warning, the forces stay."""
    from .physics_run import fans_report_si, heat_outputs
    try:
        diag: dict[str, float] = {}
        if hasattr(solver, "latest_diagnostics"):
            for k, v in (solver.latest_diagnostics() or {}).items():
                with contextlib.suppress(TypeError, ValueError):
                    fv = float(v)
                    if math.isfinite(fv) and k not in ("perf_steps",):
                        diag[k] = fv
        if getattr(plan, "placed", None) is not None:
            ho = heat_outputs(solver, plan, h, t0, averaging)
            result.nusselt = ho["nusselt"]
            result.heat_rate = float(ho["heat_rate"])
            for name, vals in ho["per_body"].items():
                result.per_body.setdefault(name, {}).update(vals)
            diag.update(ho["diagnostics"])
            result.notes.extend(ho["notes"])
            result.warnings.extend(w for w in ho.get("warnings", ()) if w not in result.warnings)
            result.fans_report = fans_report_si(solver, plan, h, t0, averaging)
            nu = result.nusselt
            if math.isfinite(nu.mean):
                steady = not (math.isfinite(nu.stderr) and nu.stderr > 1e-6 * abs(nu.mean))
                bar = "" if steady else (f" ± {units.sig(nu.stderr, 2)}"
                                         + ("" if nu.stderr_reliable else " (a rough estimate: few independent samples)"))
                result.notes.append(f"Mean Nusselt number {units.sig(nu.mean, 4)}{bar}"
                                    + (" (steady)" if steady else "")
                                    + f" over t = {t0:.3g} … {float(h['t'][-1]):.3g}; heat rate "
                                    f"{units.sig(result.heat_rate, 4)} {'W' if getattr(plan, 'dimension', 2) == 3 else 'W/m'}.")
        result.diagnostics = diag
        attach_diagnostics_history(result, solver)
        if getattr(plan, "placed", None) is not None:
            # the plan's ΔT_ref (and β·ΔT, Ra, Gr) came from an estimated rise; the run measured the real one
            from .physics_run import apply_measured_thermal
            measured = apply_measured_thermal(result, plan)
            # a buoyancy-dominated tunnel run: the coefficients on the plume's own dynamic pressure
            from .physics import buoyant_coefficients_note
            bnote = buoyant_coefficients_note(result, plan, scales=measured)
            if bnote:
                result.notes.append(bnote)
    except Exception as exc:  # noqa: BLE001 - outputs are best effort
        result.warnings.append(f"The heat-transfer and fan outputs could not be extracted ({type(exc).__name__}: {exc}).")


def _monitor_messages(v: dict | None, plan, notes: list[str], warnings: list[str], *, converged: bool,
                      cancelled: bool, t_now: float, t_end: float, rtol: float) -> None:
    """What the placed-physics monitors' verdict means, in words (the forces' own sentences stay)."""
    mv = (v or {}).get("monitors")
    if not mv:
        return
    names = MONITOR_NAMES
    judged = mv.get("judged", {})

    def label(m: str) -> str:
        if m.startswith("fan"):
            i, _, q = m[3:].partition("_")
            return f"fan {int(i) + 1}'s {'thrust' if q == 'thrust' else 'disc velocity'}"
        return names.get(m, m)
    if converged:
        parts = []
        for m, j in judged.items():
            rel, mean = j.get("rel_stderr", math.nan), j.get("mean", math.nan)
            if not math.isfinite(rel):
                continue
            if rel < 1e-6 and math.isfinite(mean) and mean:
                # a steady value has no fluctuation to average: say how much it still moved instead
                parts.append(f"{label(m)} steady (moved {_pct(abs(j.get('drift', 0.0)) / abs(mean))} between the "
                             "halves of the window)")
            else:
                parts.append(f"{label(m)} ±{_pct(rel)} (one standard error)")
        if parts:
            notes.append("Settled: " + ", ".join(parts) + ".")
    elif not cancelled and mv.get("worst"):
        j = judged.get(mv["worst"], {})
        others = [label(m) for m in getattr(plan, "monitors", ()) if m != mv["worst"]]
        if getattr(plan, "domain_mode", "tunnel") == "closed":
            # a closed box has no force verdict: say everything the run was judged on
            judged_on = (f" This closed box is judged on {label(mv['worst'])}"
                         + (f", {', '.join(others[:-1])} and {others[-1]}" if len(others) > 1 else
                            f" and {others[0]}" if others else "")
                         + "; force coefficients mean nothing inside it and are not used.")
        else:
            judged_on = ""
        warnings.append(f"Not converged by t = {t_now:.3g} of {t_end:.4g} L/U: {label(mv['worst'])} — "
                        f"{j.get('reason', 'too few samples to judge')} Run longer (Duration) before relying on it."
                        + judged_on)


# ---------------------------------------------------------------------------
# cross-check against the viscous polar (airfoils)
# ---------------------------------------------------------------------------

def _wants_polar_reference(plan: SimPlan) -> bool:
    """A single airfoil in free flight at a Reynolds number the polar solver handles."""
    d = plan.design
    return (d.kind == "airfoil" and len(d.bodies) == 1 and not plan.ground
            and math.isfinite(plan.re_real) and plan.re_real >= POLAR_CHECK_RE)


def _polar_reference(plan: SimPlan, result: CFDResult, cancel: threading.Event | None) -> None:
    """Analyse the same airfoil, Re and angle with the viscous polar solver and
    put both answers side by side in ``result`` (see the module docstring).

    Best effort: the forces are what the run is for, so any failure here
    becomes a note, never an error.
    """
    sim = plan.sim
    alpha = float(sim.conditions.alpha_deg)
    try:
        from ..solvers.airfoil import analyze, prepare_coords
        coords, _chord, info = prepare_coords(plan.design)
        # the CFD measures alpha from the design's x-axis, the polar from the chord line
        alpha_chord = alpha + float(info.get("incidence_deg", 0.0) or 0.0)
        a = sim.airfoil
        pt = analyze(coords, plan.re_real, alpha_chord, ncrit=a.ncrit, xtr_upper=a.xtr_upper,
                     xtr_lower=a.xtr_lower, panels=a.panels, cancel=cancel)
    except Exception as exc:  # noqa: BLE001 - a failed cross-check must not cost the forces
        result.notes.append(f"The comparison with the viscous polar could not be made ({type(exc).__name__}: "
                            f"{exc}).")
        return
    if cancel is not None and cancel.is_set():
        return
    where = f"Re {units.fmt_reynolds(plan.re_real)}, α {alpha:g}°"
    if not (pt.converged and math.isfinite(pt.cl) and math.isfinite(pt.cd) and pt.cd > 0):
        result.notes.append(f"The viscous polar did not converge at {where} (typically past the stall), so "
                            "there is no polar value to compare with.")
        return
    cl_c, cd_c = result.cl.mean, result.cd.mean
    if not (math.isfinite(cl_c) and math.isfinite(cd_c)):
        return
    sig = units.sig
    # the Results page shows the first clause of the first warning on one line, so the
    # numbers come before any ':' / ';' / '. ' outside parentheses
    pair = (f"Viscous polar at {where}, gives Cl {sig(pt.cl, 3)}, Cd {sig(pt.cd, 3)} "
            f"(this CFD run: Cl {sig(cl_c, 3)}, Cd {sig(cd_c, 3)})")
    agree = (abs(cl_c - pt.cl) <= max(0.05, POLAR_CL_RTOL * abs(pt.cl))
             and 1.0 / POLAR_CD_FACTOR <= cd_c / pt.cd <= POLAR_CD_FACTOR)
    if agree:
        result.notes.append(pair + ". The two methods agree.")
        return
    if plan.capped:
        why = (f"The CFD ran at the capped Re = {units.fmt_reynolds(plan.re_sim)}, a far more viscous flow; "
               "its numbers describe that flow.")
    else:
        why = ("The CFD grid cannot resolve boundary layers this thin and has no wall or transition model; "
               "the polar models both.")
    result.warnings.insert(0, f"{pair}. {why} Quote the polar values for this airfoil's lift and drag; use the "
                              "CFD for the flow field and for comparing designs run the same way.")


# ---------------------------------------------------------------------------
# sensitivity (aerolume.optimize)
# ---------------------------------------------------------------------------

def _attach_sensitivity(plan: SimPlan, solver, result: CFDResult, report: _Progress,
                        cancel: threading.Event | None, share: float) -> None:
    """Compute the shape-sensitivity map with :mod:`aerolume.optimize` and attach it.

    The optimize module is optional at run time (it may be missing from a
    trimmed install, or fail on an unusual geometry): every failure becomes a
    warning on the result, never a failed run -- the forces are still valid.
    """
    try:
        from ..optimize import Cancelled, attach_sensitivity, compute_sensitivity
    except Exception as exc:  # noqa: BLE001 - ImportError, AttributeError, or a broken install
        result.warnings.append("The shape-sensitivity map was requested but the sensitivity module is not "
                               f"available in this installation ({type(exc).__name__}: {exc}); the forces are "
                               "unaffected.")
        return
    objective = str(getattr(plan.sim.cfd, "sensitivity_objective", "cd") or "cd")
    label = {"cd": "Cd", "cl": "Cl", "ld": "L/D", "cm": "Cm"}.get(objective.lower(), objective)
    report(share, f"Computing the shape sensitivity of {label}")

    def fwd(frac: float, msg: str = "") -> None:
        report(share + (1.0 - share) * 0.98 * max(0.0, min(1.0, float(frac))),
               f"Sensitivity of {label}: {msg}" if msg else f"Computing the shape sensitivity of {label}")

    try:
        # reusing the run's solver skips developing the flow again and makes the map
        # belong to exactly the flow this result describes
        maps = compute_sensitivity(plan.design, plan.sim, plan=plan, state=solver, objective=objective,
                                   n_per_body=[len(sd.x) for sd in result.surface] or None,
                                   progress=fwd, cancel=cancel)
        attach_sensitivity(result, maps)
    except Cancelled:
        result.warnings.append("The shape-sensitivity map was skipped: the run was stopped while computing it "
                               "(the forces are complete).")
        return
    except Exception as exc:  # noqa: BLE001 - a failed adjoint must not cost the user the forces
        if _is_oom(exc):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            msg = "the GPU ran out of memory for the reverse sweep; try Draft quality for the map"
        else:
            msg = f"{type(exc).__name__}: {exc}"
        result.warnings.append(f"The shape-sensitivity map could not be computed ({msg}); the forces are "
                               "unaffected.")
        return
    if not any(getattr(s, "sensitivity", None) is not None for s in result.surface):
        result.warnings.append("The sensitivity computation finished but returned no surface map.")


# ---------------------------------------------------------------------------
# wing (re-exported, so a caller that runs every analysis through this module works)
# ---------------------------------------------------------------------------

def run_wing(design: Design, sim: Simulation, *, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None, **kwargs):
    """Finite-wing analysis (DESIGN.md 7.7); see :func:`aerolume.simulation.wing.run_wing`."""
    from .wing import run_wing as _run_wing
    return _run_wing(design, sim, progress=progress, cancel=cancel, **kwargs)


def run_simulation(design: Design, sim: Simulation, *, project=None, airfoil_coords=None,
                   device: str | None = None, progress: ProgressFn | None = None, frame: FrameFn | None = None,
                   cancel: threading.Event | None = None, view: ViewSettings | None = None,
                   planned: Callable[[object], None] | None = None):
    """Run any simulation the way the application does, whatever its analysis and dimension.

    ``sim.analysis`` "polar" goes to :func:`run_polar`, "wing" to :func:`run_wing`,
    and "cfd" is planned (:func:`~aerolume.simulation.plan.plan_simulation`, which
    returns a 3D plan for a 3D design) and run by :func:`run_cfd`, returning a
    ``CFDResult`` or ``CFD3DResult``. Wing sections naming a project airfoil
    ("design:<id>") are resolved from ``project`` (a Project or a mapping
    id -> Design) or ``airfoil_coords`` -- for the wing analysis and for a wing
    lofted to a mesh for CFD alike. ``device`` overrides ``sim.cfd.device`` for
    CFD; ``planned(plan)`` receives the plan before the run starts (the GUI shows
    that plan rather than one made from settings that may have changed since).

    The GUI's job manager, the command line and the scripting API dispatch
    through one function, so a simulation gives the same result whichever of
    them runs it. Errors are those of the runners: ``ValueError`` with a
    readable message for impossible set-ups, ``RuntimeError`` for a blow-up.
    """
    if sim.analysis == "polar":
        return run_polar(design, sim, progress=progress, cancel=cancel)
    from .wing import airfoil_resolver
    resolver = airfoil_resolver(project, airfoil_coords)
    if sim.analysis == "wing":
        return run_wing(design, sim, progress=progress, cancel=cancel, airfoil_coords=resolver)
    from .plan import plan_simulation
    kw = {"airfoil_coords": resolver} if resolver is not None else {}
    plan = plan_simulation(design, sim, device=device, **kw)
    if planned is not None:
        planned(plan)
    return run_cfd(plan, progress=progress, frame=frame, cancel=cancel, view=view)


# ---------------------------------------------------------------------------
# polar
# ---------------------------------------------------------------------------

def run_polar(design: Design, sim: Simulation, *, progress: ProgressFn | None = None,
              cancel: threading.Event | None = None) -> PolarResult:
    """Viscous-inviscid polar sweep of an airfoil design (DESIGN.md 5.4).

    Geometry goes through :func:`aerolume.solvers.airfoil.prepare_coords`
    (which rejects shapes that are not single airfoils with a readable
    message); the Reynolds number is the real one, based on the chord, the
    simulation's speed and fluid. ``progress``/``cancel`` are forwarded; a
    cancelled sweep returns the angles finished so far.
    """
    from ..solvers.airfoil import polar, prepare_coords
    report = _Progress(progress)
    coords, chord_in, info = prepare_coords(design)
    chord = float(design.length_ref) if design.kind == "airfoil" else float(chord_in)
    cond, a = sim.conditions, sim.airfoil
    if not (cond.speed > 0):
        raise ValueError("The flow speed must be positive.")
    re = cond.reynolds(chord)
    alphas = a.alphas()
    report(0.0, f"Polar sweep: {len(alphas)} angles at Re = {units.fmt_reynolds(re)}")

    def fwd(frac: float, msg: str) -> None:
        report(0.99 * frac, f"Polar: {msg}")

    res = polar(coords, re, alphas, mach=a.mach, ncrit=a.ncrit, xtr_upper=a.xtr_upper, xtr_lower=a.xtr_lower,
                panels=a.panels, sim_id=sim.id, progress=fwd if progress is not None else None, cancel=cancel)
    res.notes.insert(0, f"Reynolds number {units.fmt_reynolds(re)} from the chord ({units.sig(chord, 4)} m), "
                        f"{units.sig(cond.speed, 4)} m/s and {display_name(cond.fluid.name)}.")
    if abs(chord_in - chord) > 0.01 * chord:
        res.notes.append(f"The section's leading-to-trailing-edge chord is {units.sig(chord_in, 4)} m; the "
                         f"Reynolds number uses the design's reference length {units.sig(chord, 4)} m.")
    inc = float(info.get("incidence_deg", 0.0) or 0.0)
    if abs(inc) > 0.05:
        res.notes.append(f"Angles are measured from the section's chord line, which is inclined {inc:+.2f}° "
                         "in the design; the CFD analysis measures them from the design's x-axis.")
    nc = sum(1 for p in res.points if p.converged)
    report(1.0, ("Polar stopped" if cancel is not None and cancel.is_set() else "Polar finished")
           + f": {nc} of {len(res.points)} angles converged")
    return res
