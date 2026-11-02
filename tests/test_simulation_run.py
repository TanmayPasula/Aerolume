"""Production runs: run_cfd mechanics, outputs, cancel, frames, convergence, sensitivity hook, run_polar.

The fast tests run tiny grids on the CPU (a few seconds in all). The ``gpu``
tests are short Draft runs of the three kinds of design; the vehicle one takes
about a minute on an idle mid-range GPU and is also marked ``slow``.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest
import torch

from aerolume.core import fluid as F
from aerolume.geometry import design_from_airfoil, naca, templates
from aerolume.model import AirfoilSettings, CFDResult, CFDSettings, FlowConditions, GroundConfig, Simulation
from aerolume.simulation import runner as runmod
from aerolume.simulation.live import LiveFrame, ViewSettings
from aerolume.simulation.plan import plan_live, plan_simulation
from aerolume.simulation.runner import run_cfd, run_polar

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
AIR = F.air_isa(0.0)


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    """Runs record throughput into the planner's module state; keep each test's
    measurements (and any calibration file) to itself."""
    from aerolume.simulation import plan as planmod
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def speed_for(re: float, length: float = 1.0) -> float:
    return re * AIR.viscosity / (AIR.density * length)


def airfoil_case(*, re: float = 200.0, alpha: float = 4.0, device: str = "cpu", **cfd):
    d = design_from_airfoil(naca("0012", 160, 1.0), "NACA 0012", 1.0)
    cfd.setdefault("cells_per_ref", 16)
    sim = Simulation("af", d.id, conditions=FlowConditions(AIR, speed_for(re), alpha),
                     cfd=CFDSettings(quality=cfd.pop("quality", "draft"), device=device, **cfd))
    return d, sim, plan_simulation(d, sim)


def vehicle_case(**cfd):
    d = templates.make("ahmed")
    cfd.setdefault("cells_per_ref", 24)
    cfd.setdefault("duration", 1.5)
    cfd.setdefault("average_from", 0.75)
    sim = Simulation("car", d.id, conditions=FlowConditions(AIR, 20.0, 0.0),
                     ground=GroundConfig(enabled=True, moving=True), cfd=CFDSettings(quality="draft", device="cpu",
                                                                                    **cfd))
    return d, sim, plan_simulation(d, sim)


# ---------------------------------------------------------------------------
# outputs (CPU, tiny grids)
# ---------------------------------------------------------------------------

def test_result_is_in_si_units_and_solver_row_order():
    d, sim, plan = vehicle_case()
    res = run_cfd(plan)
    assert isinstance(res, CFDResult) and res.sim_id == sim.id
    nx, ny = plan.grid
    assert res.grid == (nx, ny) and res.re_sim == pytest.approx(plan.re_sim)
    assert res.turbulence == plan.turbulence and res.cells_per_ref == plan.cells_per_ref
    h = res.history
    assert set(h) >= {"t", "cd", "cl", "cm"} and np.all(np.diff(h["t"]) > 0)
    assert h["t"][-1] == pytest.approx(res.t_end)
    q = 0.5 * AIR.density * 20.0 ** 2
    assert res.drag == pytest.approx(res.cd.mean * q * d.force_ref)
    assert res.lift == pytest.approx(res.cl.mean * q * d.force_ref)
    assert res.moment == pytest.approx(res.cm.mean * q * d.force_ref * d.length_ref)
    assert res.power == pytest.approx(res.drag * 20.0) and res.drag_area == pytest.approx(res.cd.mean * d.force_ref)

    for fd in (res.mean_fields, res.instant_fields):
        assert fd is not None and fd.extent == pytest.approx(plan.extent)
        for k, a in fd.fields.items():
            assert a.shape == (ny, nx), k
        assert fd.mask.shape == (ny, nx)
    # row 0 is the bottom (the road): the car's solid cells sit in the lowest rows
    hm = (plan.extent[3] - plan.extent[2]) / ny
    rows = np.nonzero(res.mean_fields.mask.max(axis=1) > 0.5)[0]
    ytop = max(float(b.points[:, 1].max()) for b in plan.bodies_wind)
    assert rows.size and rows.min() < 3 or rows.min() * hm < 0.1
    assert (rows.max() + 0.5) * hm <= ytop + 2 * hm
    # SI: free-stream speed far above the car, pressure in Pa consistent with cp
    u = res.mean_fields.fields["u"]
    assert u[-1].mean() == pytest.approx(20.0, rel=0.1)
    inst = res.instant_fields.fields
    assert np.allclose(inst["p"], inst["cp"] * q, rtol=1e-3, atol=1e-3 * q)
    assert "nut" in inst and float(inst["nut"].max()) >= 0.0
    # surface data in metres on the bodies
    assert len(res.surface) == 1
    sd = res.surface[0]
    pts = np.vstack([b.points for b in plan.bodies_wind])
    pad = 3 * hm
    assert pts[:, 0].min() - pad <= sd.x.min() and sd.x.max() <= pts[:, 0].max() + pad
    assert sd.cp.shape == sd.x.shape == sd.cf.shape
    assert set(res.per_body) == {plan.solver_config.bodies[0].name}
    assert [b.name for b in res.bodies_wind] == [b.name for b in plan.bodies_wind]
    assert res.device == plan.device_name and math.isfinite(res.wall_seconds)


def test_progress_is_monotone_with_readable_messages():
    _, _, plan = airfoil_case(duration=3.0)
    seen: list[tuple[float, str]] = []
    run_cfd(plan, progress=lambda f, m: seen.append((f, m)))
    fr = [f for f, _ in seen]
    assert fr == sorted(fr) and fr[-1] == 1.0 and fr[0] == 0.0
    msgs = [m for _, m in seen]
    assert msgs[0].startswith("Building the solver")
    assert any(m.startswith("Developing the flow (t = ") for m in msgs)
    assert any(m.startswith("Averaging forces (t = ") for m in msgs)
    assert msgs[-1].startswith("Done after")
    assert all(isinstance(m, str) and m and "{" not in m for m in msgs)


def test_cancel_returns_a_partial_result_with_a_warning():
    _, _, plan = airfoil_case(duration=200.0, average_from=1.0)
    cancel = threading.Event()

    def progress(frac: float, msg: str) -> None:
        if msg.startswith("Averaging") and not cancel.is_set():
            threading.Timer(0.3, cancel.set).start()

    t0 = time.perf_counter()
    res = run_cfd(plan, progress=progress, cancel=cancel)
    assert time.perf_counter() - t0 < 60
    assert res.t_end < 100.0 and not res.converged
    assert any(w.startswith("Stopped by the user") for w in res.warnings)
    assert math.isfinite(res.cd.mean) and res.mean_fields is not None and res.surface


def test_cancel_before_the_first_step_still_returns_a_result():
    _, _, plan = airfoil_case(duration=5.0)
    cancel = threading.Event()
    cancel.set()
    res = run_cfd(plan, cancel=cancel)
    assert res.t_end == 0.0 and not res.converged and res.mean_fields is None
    assert any(w.startswith("Stopped by the user at t = 0") for w in res.warnings)
    assert res.instant_fields is not None


def test_frames_arrive_at_the_requested_rate():
    _, _, plan = airfoil_case(re=100.0, duration=400.0, average_from=200.0)
    frames: list[tuple[float, LiveFrame]] = []
    cancel = threading.Event()
    view = ViewSettings(field="vorticity", fps=10.0)

    def on_frame(f: LiveFrame) -> None:
        frames.append((time.perf_counter(), f))
        if len(frames) >= 25:
            cancel.set()

    run_cfd(plan, frame=on_frame, cancel=cancel, view=view)
    times = np.array([t for t, _ in frames[1:-1]])        # skip the capture frame and the final one
    dt = np.diff(times)
    assert 0.07 <= float(np.median(dt)) <= 0.16, dt
    last = frames[-1][1]
    assert last.phase == "done" and last.progress == 1.0 and last.eta_s == 0.0
    assert {f.phase for _, f in frames[:-1]} <= {"transient", "averaging"}
    nx, ny = plan.grid
    for _, f in frames:
        assert f.rgba.shape == (ny, nx, 4) and f.rgba.dtype == np.uint8
        assert f.field == "vorticity" and f.vmin < f.vmax and f.units == "1/s"
        assert len(f.history["t"]) <= 2000 and f.focus is not None
        assert set(f.coeffs) == {"cd", "cl", "cm"}


def test_live_plan_is_rejected():
    d, sim, _ = airfoil_case()
    with pytest.raises(ValueError, match="explore"):
        run_cfd(plan_live(d, sim, device="cpu"))


def test_blow_up_raises_a_readable_error(monkeypatch):
    _, _, plan = airfoil_case(duration=2.0)
    from aerolume.solvers.ns2d import FlowSolver
    monkeypatch.setattr(FlowSolver, "coefficients", lambda self: np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(RuntimeError, match="became unstable"):
        run_cfd(plan)


# ---------------------------------------------------------------------------
# convergence verdict (synthetic signals)
# ---------------------------------------------------------------------------

def _series(x: np.ndarray, t_end: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(0.0, t_end, x.size), x


def _ar1(n: int, phi: float, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = phi * acc + e[i]
        out[i] = acc
    return out


def test_verdict_stationary_chaotic_signal_converges_to_tolerance():
    n = 40_000
    x = 0.6 + _ar1(n, 0.995, 0.02)                # rms ~0.2, integral time ~0.4 time units
    t, cd = _series(x, 400.0)
    cl = 2.0 + 0.5 * (x - 0.6) / 0.2
    v = runmod._verdict({"t": t, "cd": cd, "cl": cl}, 10.0, rtol=0.05, laminar=False)
    assert v["cd"]["state"] in ("converged", "converging")
    assert v["cd"]["rel_stderr"] < 0.05
    assert v["stop"]


def test_verdict_flags_a_drifting_mean():
    t = np.linspace(0.0, 40.0, 8000)
    cd = 1.0 + 0.02 * t + 0.01 * np.sin(6 * t)
    v = runmod._verdict({"t": t, "cd": cd, "cl": 0 * t}, 5.0, rtol=0.05, laminar=True)
    assert not v["stop"] and v["drifting"] and v["cd"]["state"] == "transient"


def test_verdict_amplitude_rule_depends_on_the_flow_model():
    """A shedding amplitude that is still growing is a transient for a laminar flow;
    the same modest change is sampling noise for a chaotic LES signal with few
    independent samples."""
    t = np.linspace(0.0, 60.0, 12_000)
    cl = (1.0 + 0.03 * t) * np.sin(2 * np.pi * 0.2 * t)       # amplitude x1.3 between the halves
    j_lam = runmod._judge(t, cl, 20.0, rtol=0.05, atol=0.05, laminar=True)
    assert j_lam["state"] == "transient" and "amplitude" in j_lam["reason"]
    noise = _ar1(12_000, 0.999, 0.03, seed=3)
    x = noise * np.where(t < 40, 1.0, 1.35)
    j_les = runmod._judge(t, x, 20.0, rtol=0.05, atol=0.05, laminar=False)
    assert "amplitude" not in j_les["reason"]


def test_a_small_decaying_wiggle_on_a_steady_flow_converges():
    """The Draft heated cylinder at Re 40 (steady below Re ≈ 47): Cl = −0.0006 ± 0.0023 with a
    start-up residue whose rms fell from 0.0099 to 0.0063 between the window halves. The relative
    amplitude test called that a transient although the whole wiggle sits inside the ±0.083
    target; it is judged by its standard error, as an absolute target, and is not shedding."""
    t = np.linspace(0.0, 60.0, 10_000)
    cl = 0.02 * np.exp(-(t - 30.0) / 35.0) * np.sin(2 * np.pi * 0.123 * t)       # x0.65 between the halves
    cd = 1.652 + 0.002 * np.exp(-t / 10.0)
    v = runmod._verdict({"t": t, "cd": cd, "cl": cl}, 30.0, rtol=0.05, laminar=True)
    jcl = v["cl"]
    assert jcl["amplitude_ratio"] < runmod.AMPLITUDE_BAND[0]          # it is decaying ...
    assert jcl["state"] == "converged" and jcl["absolute"]             # ... and harmless
    assert "decaying" in jcl["reason"] and "%" not in jcl["reason"]    # no relative error of a zero mean
    assert not v["periodic"] and v["stop"]
    assert runmod.target_text(jcl, 10.0) == "±0.083 (absolute: the mean is near zero)"
    assert runmod.target_text(v["cd"], 5.0) == "±5 %"


def test_a_small_but_growing_lift_is_still_a_transient():
    """Vortex shedding sets in as a tiny oscillation that grows: however small, a growing
    amplitude must keep the run going (the exemption is for decaying wiggles only)."""
    t = np.linspace(0.0, 60.0, 10_000)
    cl = 0.002 * np.exp((t - 30.0) / 15.0) * np.sin(2 * np.pi * 0.17 * t)        # x2.7 between the halves
    j = runmod._judge(t, cl, 30.0, rtol=0.1, atol=0.083, laminar=True)
    assert j["state"] != "converged" and j["rel_stderr"] > 0          # its rms is below a tenth of the target,
    v = runmod._verdict({"t": t, "cd": 1.4 + 0 * t, "cl": cl}, 30.0, rtol=0.05, laminar=True)
    assert not v["stop"]                                               # yet it is not "steady"
    # a decaying one that does not fit inside the target is still judged by the amplitude rule
    big = 0.3 * np.exp(-(t - 30.0) / 20.0) * np.sin(2 * np.pi * 0.17 * t)
    assert runmod._judge(t, big, 30.0, rtol=0.1, atol=0.083, laminar=True)["state"] == "transient"


def test_verdict_needs_whole_cycles_for_a_periodic_flow():
    t = np.linspace(0.0, 10.0, 4000)
    cl = 0.3 * np.sin(2 * np.pi * 0.25 * t)          # 2.5 cycles from t = 0
    cd = 1.4 + 0.01 * np.sin(2 * np.pi * 0.5 * t)
    v = runmod._verdict({"t": t, "cd": cd, "cl": cl}, 0.0, rtol=0.05, laminar=True)
    assert not v["stop"]


def test_verdict_with_too_few_samples_does_not_stop():
    t = np.linspace(0.0, 1.0, 30)
    v = runmod._verdict({"t": t, "cd": 1 + 0 * t, "cl": 0 * t}, 0.0, rtol=0.05, laminar=True)
    assert not v["stop"] and v["cd"] is None


def test_verdict_says_when_its_error_bar_is_only_an_estimate():
    """A window of a few correlation lengths: the judge marks the error bar unreliable, a long
    window of the same noise is reliable, and a steady signal always is."""
    x = 0.16 + 0.004 * _ar1(2240, 319.0 / 321.0, 1.0, seed=5) / 12.7
    t, cd = _series(x, 20.0)
    j = runmod._judge(t, cd, 0.0, rtol=0.05, atol=1e-3, laminar=True)
    assert j["reliable"] is False and j["ess"] < 20
    x = 0.16 + 0.004 * _ar1(40_000, 0.9, 1.0, seed=6)
    t, cd = _series(x, 400.0)
    assert runmod._judge(t, cd, 0.0, rtol=0.05, atol=1e-3, laminar=True)["reliable"] is True
    t = np.linspace(0.0, 20.0, 2000)
    assert runmod._judge(t, 0.14 + 1e-7 * np.exp(-t), 0.0, rtol=0.05, atol=1e-3, laminar=True)["reliable"]


@pytest.mark.gpu
@pytest.mark.slow
@needs_cuda
def test_gpu_short_run_does_not_vouch_for_its_error_bars():
    """The review case: Draft NACA 4412, Re 1000, 6 deg, 20 L/U, about 7 effective samples. The
    warning used to say 'The averages are usable with the error bars shown' while both means were
    4.2 of those error bars from the converged run."""
    d = design_from_airfoil(naca("4412", 200, 1.0), "NACA 4412", 1.0)
    sim = Simulation("af", d.id, conditions=FlowConditions(AIR, speed_for(1000.0), 6.0),
                     cfd=CFDSettings(quality="draft", device="cuda", duration=20.0))
    res = run_cfd(plan_simulation(d, sim))
    if res.converged:
        pytest.skip("converged within 20 L/U on this machine: nothing to check")
    text = " ".join(res.warnings)
    assert "usable with the error bars shown" not in text and "rough estimates" in text
    assert res.cd.stderr_reliable is False
    long = run_cfd(plan_simulation(d, Simulation("af", d.id, conditions=sim.conditions,
                                                 cfd=CFDSettings(quality="draft", device="cuda",
                                                                 duration=60.0))))
    assert long.converged
    for key in ("cd", "cl"):
        short, ref = getattr(res, key), getattr(long, key)
        assert abs(short.mean - ref.mean) <= 3.0 * short.stderr, key     # was 4.2 standard errors


# ---------------------------------------------------------------------------
# sensitivity hook
# ---------------------------------------------------------------------------

def test_sensitivity_hook_reuses_the_solver_and_attaches_maps(monkeypatch):
    import aerolume.optimize as opt
    calls = {}

    def fake_compute(design, sim, *, plan=None, state=None, objective=None, n_per_body=None, progress=None,
                     cancel=None, **kw):
        calls.update(design=design, sim=sim, plan=plan, state=state, objective=objective, n=n_per_body)
        progress(0.5, "reverse sweep")
        return [np.full(n, 0.25) for n in n_per_body]

    def fake_attach(result, maps):
        for sd, m in zip(result.surface, maps):
            sd.sensitivity = m
        result.notes.append("fake sensitivity")
        return result

    monkeypatch.setattr(opt, "compute_sensitivity", fake_compute, raising=False)
    monkeypatch.setattr(opt, "attach_sensitivity", fake_attach, raising=False)
    _, _, plan = airfoil_case(duration=2.0, compute_sensitivity=True, sensitivity_objective="cl")
    msgs = []
    res = run_cfd(plan, progress=lambda f, m: msgs.append((f, m)))
    from aerolume.solvers.ns2d import FlowSolver
    assert isinstance(calls["state"], FlowSolver) and calls["plan"] is plan and calls["objective"] == "cl"
    assert calls["n"] == [len(sd.x) for sd in res.surface]
    assert all(sd.sensitivity is not None for sd in res.surface)
    assert "fake sensitivity" in res.notes
    assert any("sensitivity of Cl" in m for _, m in msgs)
    fr = [f for f, _ in msgs]
    assert fr == sorted(fr) and fr[-1] == 1.0


def test_sensitivity_failure_keeps_the_forces(monkeypatch):
    import aerolume.optimize as opt

    def boom(*a, **k):
        raise RuntimeError("adjoint exploded")

    monkeypatch.setattr(opt, "compute_sensitivity", boom, raising=False)
    _, _, plan = airfoil_case(duration=2.0, compute_sensitivity=True)
    res = run_cfd(plan)
    assert math.isfinite(res.cd.mean)
    assert any("sensitivity map could not be computed" in w and "adjoint exploded" in w for w in res.warnings)


def test_sensitivity_module_missing_is_a_warning(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (level == 2 and name == "optimize") or name == "aerolume.optimize":
            raise ImportError("no optimize here")
        return real_import(name, globals, locals, fromlist, level)

    _, _, plan = airfoil_case(duration=2.0, compute_sensitivity=True)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = run_cfd(plan)
    assert math.isfinite(res.cd.mean)
    assert any("not available" in w for w in res.warnings)


def test_sensitivity_is_skipped_after_a_cancel(monkeypatch):
    import aerolume.optimize as opt
    monkeypatch.setattr(opt, "compute_sensitivity", lambda *a, **k: pytest.fail("must not run"), raising=False)
    _, _, plan = airfoil_case(duration=5.0, compute_sensitivity=True)
    cancel = threading.Event()
    cancel.set()
    res = run_cfd(plan, cancel=cancel)
    assert any("sensitivity map was skipped" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# polar
# ---------------------------------------------------------------------------

def test_run_polar_sweep_and_notes():
    d = design_from_airfoil(naca("0012", 200, 1.0), "NACA 0012", 1.0)
    sim = Simulation("p", d.id, analysis="polar", conditions=FlowConditions(AIR, speed_for(1e6), 0.0),
                     airfoil=AirfoilSettings(alpha_start=0.0, alpha_end=4.0, alpha_step=2.0))
    seen = []
    res = run_polar(d, sim, progress=lambda f, m: seen.append((f, m)))
    assert res.re == pytest.approx(1e6, rel=1e-6) and res.sim_id == sim.id
    assert [p.alpha for p in res.points] == [0.0, 2.0, 4.0] and all(p.converged for p in res.points)
    cl = [p.cl for p in res.points]
    assert abs(cl[0]) < 0.02 and cl[0] < cl[1] < cl[2]
    assert res.notes[0].startswith("Reynolds number")
    assert seen[-1][0] == 1.0 and "3 of 3 angles converged" in seen[-1][1]


def test_run_polar_cancel_returns_what_is_done():
    d = design_from_airfoil(naca("0012", 200, 1.0), "NACA 0012", 1.0)
    sim = Simulation("p", d.id, analysis="polar", conditions=FlowConditions(AIR, speed_for(1e6), 0.0),
                     airfoil=AirfoilSettings(alpha_start=-4.0, alpha_end=12.0, alpha_step=1.0))
    cancel = threading.Event()
    cancel.set()
    seen = []
    res = run_polar(d, sim, cancel=cancel, progress=lambda f, m: seen.append(m))
    assert len([p for p in res.points if p.converged]) < 17
    assert seen[-1].startswith("Polar stopped")


# ---------------------------------------------------------------------------
# GPU draft runs
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@needs_cuda
def test_gpu_draft_airfoil_low_reynolds():
    """NACA 0012, Re 1000, 5 deg, Draft: steady laminar flow (theory_ns2d.md: Cd ~0.13,
    Cl 0.20-0.23 at 80-160 cells per chord; Draft has 64)."""
    d = design_from_airfoil(naca("0012", 240, 1.0), "NACA 0012", 1.0)
    sim = Simulation("af", d.id, conditions=FlowConditions(AIR, speed_for(1000.0), 5.0),
                     cfd=CFDSettings(quality="draft", device="cuda"))
    plan = plan_simulation(d, sim)
    assert plan.turbulence == "laminar"
    res = run_cfd(plan)
    assert 0.11 < res.cd.mean < 0.17 and 0.13 < res.cl.mean < 0.30
    assert res.converged and res.wall_seconds < 60


@pytest.mark.gpu
@needs_cuda
def test_gpu_draft_cylinder_re100_sheds():
    """Re 100 at Draft (24 cells/D, 5 % blockage): validation table 1.43 / St 0.169."""
    d = templates.make("cylinder")
    sim = Simulation("cyl", d.id, conditions=FlowConditions(AIR, speed_for(100.0), 0.0),
                     cfd=CFDSettings(quality="draft", device="cuda"))
    res = run_cfd(plan_simulation(d, sim))
    assert 1.38 < res.cd.mean < 1.50
    assert res.strouhal is not None and 0.160 < res.strouhal < 0.178
    assert 0.18 < res.cl.rms < 0.32                         # amplitude ~0.35 -> rms ~0.25
    assert abs(res.cl.mean) < 0.05 and res.wall_seconds < 120


@pytest.mark.gpu
@needs_cuda
def test_gpu_run_streams_frames_and_stops_on_cancel():
    d = templates.make("cylinder")
    sim = Simulation("cyl", d.id, conditions=FlowConditions(AIR, speed_for(100.0), 0.0),
                     cfd=CFDSettings(quality="draft", device="cuda"))
    plan = plan_simulation(d, sim)
    frames = []
    cancel = threading.Event()
    threading.Timer(4.0, cancel.set).start()
    res = run_cfd(plan, frame=frames.append, cancel=cancel, view=ViewSettings(fps=20.0))
    assert any(w.startswith("Stopped by the user") for w in res.warnings)
    assert 20 <= len(frames) <= 200 and frames[-1].phase == "done"


@pytest.mark.gpu
@pytest.mark.slow
@needs_cuda
def test_gpu_draft_sedan_on_a_moving_road():
    """The Draft sedan: ~1 minute on an idle RTX 4060 Ti; a 2D profile gives Cd ~0.6 and
    a strongly positive Cl ~2.8 per frontal height (see the vehicle note in plan.py)."""
    d = templates.make("sedan")
    sim = Simulation("car", d.id, conditions=FlowConditions(AIR, 30.0, 0.0),
                     ground=GroundConfig(enabled=True, moving=True), cfd=CFDSettings(quality="draft", device="cuda"))
    plan = plan_simulation(d, sim)
    assert plan.blockage <= 0.08
    res = run_cfd(plan)
    assert 0.4 < res.cd.mean < 0.85 and 1.5 < res.cl.mean < 4.0
    assert math.isfinite(res.cd.stderr) and res.cd.stderr / res.cd.mean < 0.12
    assert res.mean_fields is not None and res.wall_seconds < 400


# ---------------------------------------------------------------------------
# the viscous-polar cross-check of airfoil CFD results
# ---------------------------------------------------------------------------

def _naca2412_plan(re: float, **cfd):
    d = design_from_airfoil(naca("2412", 160, 1.0), "NACA 2412", 1.0)
    sim = Simulation("af", d.id, conditions=FlowConditions(AIR, speed_for(re), 4.0),
                     cfd=CFDSettings(quality="draft", device="cpu", **cfd))
    return plan_simulation(d, sim)


def _result(cl: float, cd: float) -> CFDResult:
    from aerolume.model import Stat
    return CFDResult(sim_id="af", cl=Stat(mean=cl), cd=Stat(mean=cd))


def test_polar_reference_leads_the_warnings_when_cfd_and_polar_disagree():
    """The measured case: NACA 2412, Re 1e6, 4 deg, Draft CFD gave Cl 0.16 / Cd 0.096."""
    plan = _naca2412_plan(1e6, reynolds_mode="capped")
    assert runmod._wants_polar_reference(plan)
    res = _result(0.152, 0.0985)
    res.warnings.append("an older warning")
    runmod._polar_reference(plan, res, None)
    first = res.warnings[0]
    assert first.startswith("Viscous polar at Re 1.00×10^6, α 4°, gives Cl 0.7")
    assert "(this CFD run: Cl 0.152, Cd 0.0985)." in first and "capped Re" in first
    # the Results page shows only the first clause of the first warning
    from aerolume.app.widgets.kpi import headline
    assert "Cd 0.0985)" in headline(first) and len(headline(first)) < 140
    assert res.warnings[1] == "an older warning"


def test_polar_reference_is_a_note_when_the_methods_agree():
    plan = _naca2412_plan(1e6)
    from aerolume.solvers.airfoil import analyze, prepare_coords
    pt = analyze(prepare_coords(plan.design)[0], plan.re_real, 4.0)
    res = _result(pt.cl * 1.05, pt.cd * 1.2)
    runmod._polar_reference(plan, res, None)
    assert not res.warnings and res.notes[-1].endswith("The two methods agree.")


def test_no_polar_reference_for_low_reynolds_airfoils_vehicles_or_ground_effect():
    assert not runmod._wants_polar_reference(_naca2412_plan(1e3))
    _, _, vplan = vehicle_case()
    assert not runmod._wants_polar_reference(vplan)
    wing = templates.make("front_wing")
    sim = Simulation("fw", wing.id, conditions=FlowConditions(AIR, 30.0, 0.0),
                     ground=GroundConfig(enabled=True, moving=True), cfd=CFDSettings(quality="draft", device="cpu"))
    assert not runmod._wants_polar_reference(plan_simulation(wing, sim))


# ---------------------------------------------------------------------------
# Strouhal number only when it is a result (UI critique round 2)
# ---------------------------------------------------------------------------

def _st_plan(geometry_class: str = "generic", bl_resolved: bool = True):
    from types import SimpleNamespace
    return SimpleNamespace(geometry_class=geometry_class, bl_resolved=bl_resolved, strouhal_length=0.5,
                           length_ref=1.0)


def _signal(freq: float, amp: float, mean: float = 0.0, span: float = 40.0, noise: float = 0.0, seed: int = 0):
    t = np.linspace(0.0, span, 4000)
    rng = np.random.default_rng(seed)
    return t, mean + amp * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(t.size)


def test_strouhal_of_clean_shedding_is_reported():
    t, cl = _signal(0.4, 0.3)
    st, note = runmod._strouhal(_st_plan(), t, cl, 0.0, cd_mean=1.4)
    assert st == pytest.approx(0.4 * 0.5, rel=0.01) and note is None


def test_a_negligible_or_broadband_wiggle_has_no_strouhal_number():
    t, cl = _signal(1.7, 1e-3, mean=0.5)                       # 0.2 % of the lift: a ringing steady flow
    assert runmod._strouhal(_st_plan(), t, cl, 0.0, cd_mean=0.1) == (None, None)
    t, cl = _signal(0.0, 0.0, mean=0.5, noise=0.05)              # white noise: no line in the spectrum
    assert runmod._strouhal(_st_plan(), t, cl, 0.0, cd_mean=0.1) == (None, None)


def test_too_few_cycles_give_no_strouhal_number_but_say_why():
    t, cl = _signal(0.06, 0.8, mean=2.5, span=40.0)             # 2.4 cycles (the Draft sedan's "St 0.034")
    st, note = runmod._strouhal(_st_plan("vehicle"), t, cl, 0.0, cd_mean=0.6)
    assert st is None and "cycles" in note


def test_under_resolved_airfoil_oscillation_is_not_a_strouhal_number():
    """NACA 2412, Re 1e6, 4 deg at Draft rings at St 1.66 per chord with 40 % of its
    lift: a property of the under-resolved flow, not of the attached airfoil."""
    t, cl = _signal(1.66, 0.09, mean=0.16, span=4.0)
    st, note = runmod._strouhal(_st_plan("airfoil", bl_resolved=False), t, cl, 0.0, cd_mean=0.097)
    assert st is None and "boundary layers" in note
    st, _ = runmod._strouhal(_st_plan("airfoil", bl_resolved=True), t, cl, 0.0, cd_mean=0.097)
    assert st == pytest.approx(1.66 * 0.5, rel=0.02)


# ---------------------------------------------------------------------------
# run-time measurements fed back to the planner
# ---------------------------------------------------------------------------

def _meas_plan():
    from types import SimpleNamespace
    cfg = SimpleNamespace(cfl=0.5, h=1 / 64, advection="upwind3", turbulence="wale", dtype="float32")
    return SimpleNamespace(solver_config=cfg, device="cuda", cells=300_000, cells_per_ref=64.0,
                           geometry_class="airfoil", design=SimpleNamespace(kind="airfoil"))


def test_whole_runs_record_throughput_loop_share_and_speed_factor():
    from aerolume.simulation import plan as planmod
    runmod._record_measurements(_meas_plan(), steps_timed=10_000, step_wall=20.0, loop_wall=25.0, t_first=0.1,
                                t_final=20.1, t_end_planned=20.0, live_view=True)
    assert planmod.loop_efficiency("cuda", 300_000, live_view=True) == pytest.approx(0.8)
    # mean dt 2e-3 with h = 1/64 and CFL 0.5 -> (max|u|+max|v|)/U = 3.906
    assert planmod._speed_factor("airfoil", "les", 64) == pytest.approx(0.5 / 64 / 2e-3)
    assert planmod.estimate_steps_per_second("cuda", 300_000, turbulence="wale") == pytest.approx(500.0)


def test_partial_runs_do_not_teach_the_planner_a_speed_factor():
    """The wake develops late (the time step shrinks), so the first third of a run
    under-states the steps a whole run takes."""
    from aerolume.simulation import plan as planmod
    runmod._record_measurements(_meas_plan(), steps_timed=3_000, step_wall=6.0, loop_wall=6.5, t_first=0.1,
                                t_final=6.1, t_end_planned=20.0, live_view=False)
    assert planmod._SPEED_CACHE == {} and planmod._RATE_CACHE and planmod._LOOP_CACHE
    runmod._record_measurements(_meas_plan(), steps_timed=100, step_wall=0.2, loop_wall=0.3, t_first=0.1,
                                t_final=20.1, t_end_planned=20.0, live_view=False)      # too short to time
    assert len(planmod._RATE_CACHE[("gpu", "les", "float32")]) == 1
