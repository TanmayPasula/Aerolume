"""The simulation planner: domains, grids, Reynolds handling, wheels, rationale.

Planning never touches the GPU (it only asks the device for its name and
memory), so these tests are fast; GPU-class plans are skipped without CUDA.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from aerolume.core import fluid as F
from aerolume.geometry import design_from_airfoil, naca, templates
from aerolume.model import CFDSettings, FlowConditions, GroundConfig, Simulation
from aerolume.simulation import defaults as defaults_module
from aerolume.simulation import plan as planmod
from aerolume.simulation.plan import (
    BLOCKAGE_MAX,
    CONVERGENCE_RTOL,
    QUALITY_ORDER,
    fft_friendly,
    plan_live,
    plan_simulation,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]
AIR = F.air_isa(0.0)


def speed_for(re: float, length: float = 1.0) -> float:
    """Air speed [m/s] that gives Reynolds number ``re`` on ``length``."""
    return re * AIR.viscosity / (AIR.density * length)


def airfoil(code: str = "0012"):
    return design_from_airfoil(naca(code, 200, 1.0), f"NACA {code}", 1.0)


def make_sim(design, *, speed: float = 30.0, alpha: float = 0.0, ground: bool | None = None,
             moving: bool = True, rotate: bool = True, **cfd) -> Simulation:
    on_road = design.kind == "vehicle" if ground is None else ground
    return Simulation("t", design.id, conditions=FlowConditions(AIR, speed, alpha),
                      ground=GroundConfig(enabled=on_road, moving=moving, rotate_wheels=rotate),
                      cfd=CFDSettings(**cfd))


def is_fft_friendly(n: int) -> bool:
    if n % 2:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_fft_friendly_is_smallest_even_5_smooth():
    for n in (1, 7, 97, 211, 1023, 1250, 4097):
        m = fft_friendly(n)
        assert m >= n and is_fft_friendly(m)
        assert not any(is_fft_friendly(k) for k in range(max(n, 8), m))


def test_plan_is_fast_enough_for_the_gui_thread():
    d = templates.make("sedan")
    t0 = time.perf_counter()
    plan_simulation(d, make_sim(d, quality="fine"), device="cpu")
    assert time.perf_counter() - t0 < 0.5


# ---------------------------------------------------------------------------
# domains and blockage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quality", QUALITY_ORDER)
@pytest.mark.parametrize("name", ["sedan", "suv", "box_truck", "ahmed"])
def test_vehicle_blockage_limits(name, quality, device):
    """With a road the domain only needs height above it, but enough of it: <= 8 %
    blockage at Draft, <= 5 % at Standard and Fine, on every device."""
    d = templates.make(name)
    p = plan_simulation(d, make_sim(d, quality=quality), device=device)
    limit = BLOCKAGE_MAX["vehicle"][QUALITY_ORDER.index(quality)]
    assert p.blockage <= limit + 1e-9
    assert limit == (0.08 if quality == "draft" else 0.05)
    ymin = min(float(b.points[:, 1].min()) for b in p.bodies_wind)
    above = max(float(b.points[:, 1].max()) for b in p.bodies_wind) - ymin
    assert p.extent[2] == 0.0 and p.solver_config.bc_bottom == "moving"
    assert above / (p.extent[3] - p.extent[2]) == pytest.approx(p.blockage, rel=1e-9)
    xmin = min(float(b.points[:, 0].min()) for b in p.bodies_wind)
    xmax = max(float(b.points[:, 0].max()) for b in p.bodies_wind)
    assert xmin - p.extent[0] >= 1.0 * d.length_ref - 1e-9            # >= 1 length of road ahead
    assert p.extent[1] - xmax >= 2.5 * d.length_ref - 1e-9            # >= 2.5 lengths of wake
    nx, ny = p.grid
    assert is_fft_friendly(nx) and is_fft_friendly(ny)
    assert p.cells == nx * ny
    assert p.cells <= planmod._cell_budget("gpu" if device == "cuda" else "cpu", "vehicle",
                                           QUALITY_ORDER.index(quality), live=False)


@needs_cuda
def test_draft_sedan_on_gpu_runs_in_about_a_minute_and_resolves_the_clearance():
    d = templates.make("sedan")
    p = plan_simulation(d, make_sim(d, quality="draft"), device="cuda")
    clearance = min(float(b.points[:, 1].min()) for b in p.bodies_wind)
    cells_under = clearance * p.cells_per_ref / d.length_ref
    assert cells_under >= planmod.THIN_SLACK * planmod.THIN_CELLS[0]
    assert p.cells < 400_000
    # the reference throughput model (not this machine's calibration) puts it near a minute
    assert 30 < p.steps_estimate / planmod._model_rate("gpu", "les", "float32", p.cells) < 120
    assert not any("ground clearance spans only" in w for w in p.warnings)


def test_free_flight_vehicle_and_airfoil_blockage():
    d = templates.make("sedan")
    p = plan_simulation(d, make_sim(d, ground=False, quality="draft"), device="cpu")
    assert p.solver_config.bc_bottom == "slip" and p.blockage <= 0.08 + 1e-9
    assert any("road is switched off" in n for n in p.notes)
    a = airfoil()
    for alpha in (0.0, 10.0):
        p = plan_simulation(a, make_sim(a, alpha=alpha), device="cpu")
        assert p.blockage <= BLOCKAGE_MAX["airfoil"][1] + 1e-9
        assert p.extent[3] - p.extent[2] >= 3.0                         # chords
        assert p.geometry_class == "airfoil" and p.ref_label == "chord"


def test_wing_on_a_road_is_laid_out_like_an_airfoil():
    d = templates.make("front_wing")
    p = plan_simulation(d, make_sim(d, quality="draft"), device="cpu")
    assert p.ground and p.geometry_class == "airfoil" and p.ref_label == "chord"
    assert p.extent[3] - p.extent[2] >= 4.0 * d.length_ref - 1e-9      # ceiling >= 4 chords up
    assert p.strouhal_length == pytest.approx(d.length_ref)


def test_user_domain_height_that_is_too_low_is_warned_about():
    d = templates.make("sedan")
    p = plan_simulation(d, make_sim(d, quality="draft", domain_height=1.0), device="cpu")
    assert p.blockage > 0.2
    assert any("blockage" in w and "as tall as the vehicle" in w for w in p.warnings)


def test_focus_extent_frames_the_body_inside_the_domain():
    d = templates.make("sedan")
    p = plan_simulation(d, make_sim(d, quality="draft"), device="cpu")
    fx0, fx1, fy0, fy1 = p.focus_extent
    ex0, ex1, ey0, ey1 = p.extent
    assert ex0 <= fx0 < fx1 <= ex1 and ey0 <= fy0 < fy1 <= ey1
    pts = np.vstack([b.points for b in p.bodies_wind])
    assert fx0 <= pts[:, 0].min() and pts[:, 0].max() <= fx1 and pts[:, 1].max() <= fy1
    assert (fy1 - fy0) < 0.5 * (ey1 - ey0)                   # much tighter than the tall domain


# ---------------------------------------------------------------------------
# resolution, budgets, devices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("quality", QUALITY_ORDER)
def test_airfoil_resolution_presets_and_thin_features(quality):
    a = airfoil("0006")
    p = plan_simulation(a, make_sim(a, quality=quality), device="cpu")
    qi = QUALITY_ORDER.index(quality)
    assert p.cells_per_ref >= planmod.THIN_SLACK * planmod.CELLS_PER_SIZE["airfoil"]["cpu"][qi] or \
        any("cell budget" in w for w in p.warnings)
    t_cells = 0.06 * p.cells_per_ref
    assert t_cells >= planmod.THIN_SLACK * planmod.THIN_CELLS[qi] - 1e-6 or any("thickness" in w for w in p.warnings)


def test_user_cells_per_ref_is_respected_and_reported():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, quality="custom", cells_per_ref=12), device="cpu")
    assert p.cells_per_ref == 12 and p.solver_config.h == pytest.approx(1 / 12)
    assert any("set by hand" in n for n in p.notes)


def test_unavailable_device_falls_back_with_a_warning():
    a = airfoil()
    p = plan_simulation(a, make_sim(a), device="mps" if not torch.backends.mps.is_available() else "cuda:7")
    assert any("not available" in w for w in p.warnings)


def test_cpu_plan_names_the_solver_threads_and_uses_float64():
    a = airfoil()
    p = plan_simulation(a, make_sim(a), device="cpu")
    assert p.precision == "float64" and "solver thread" in p.device_name
    assert p.summary()["Device"].endswith("float64")


# ---------------------------------------------------------------------------
# Reynolds number and flow model
# ---------------------------------------------------------------------------

def test_low_reynolds_is_laminar_and_central_when_the_cells_allow():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=speed_for(100.0), alpha=5.0, quality="standard"), device="cpu")
    assert p.turbulence == "laminar" and p.solver_config.turbulence == "none"
    assert p.advection == "central"                           # cell Re = 100 / 64 < 2
    assert p.re_sim == pytest.approx(p.re_real) and not p.capped


def test_high_reynolds_real_mode_uses_les():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=30.0, alpha=4.0), device="cpu")
    assert p.re_real > 1e6 and p.re_sim == p.re_real
    assert p.turbulence == "les-wale" and p.solver_config.turbulence == "wale" and p.advection == "upwind3"
    assert any("Polar analysis" in w for w in p.warnings)


def test_capped_reynolds_without_a_cap_uses_the_resolvable_value():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=30.0, reynolds_mode="capped"), device="cpu")
    h = 1.0 / p.cells_per_ref
    assert p.capped and p.re_sim == pytest.approx((5.0 / (planmod.BL_CELLS_MIN * h)) ** 2)
    assert p.turbulence == "laminar"
    assert p.summary()["Simulated Reynolds number"].endswith("(capped)")
    assert any("Reynolds number capped" in n for n in p.notes)


def test_capped_reynolds_with_an_explicit_cap():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=30.0, reynolds_mode="capped", reynolds_cap=1000.0), device="cpu")
    assert p.re_sim == 1000.0 and p.solver_config.re == 1000.0
    p2 = plan_simulation(a, make_sim(a, speed=speed_for(500.0), reynolds_mode="capped", reynolds_cap=1000.0),
                         device="cpu")
    assert p2.re_sim == pytest.approx(500.0) and not p2.capped           # a cap above the real Re does nothing


def test_forced_flow_models():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=30.0, turbulence="laminar"), device="cpu")
    assert p.turbulence == "laminar" and any("Laminar flow was requested" in w for w in p.warnings)
    p = plan_simulation(a, make_sim(a, speed=speed_for(100.0), turbulence="les"), device="cpu")
    assert p.turbulence == "les-wale" and any("as requested" in n for n in p.notes)


# ---------------------------------------------------------------------------
# wheels and road
# ---------------------------------------------------------------------------

def test_rotating_wheels_roll_with_a_moving_road():
    d = templates.make("sedan", wheels=True, wheel_gap=0.08)
    U = 30.0
    # 200 cells per length: the 8 cm wheel gap is 3.4 cells, open (on a coarse CPU grid
    # the planner rightly reports it as touching)
    p = plan_simulation(d, make_sim(d, speed=U, quality="draft", cells_per_ref=200), device="cpu")
    wheels = [(b, ib) for b, ib in zip(p.bodies_wind, p.solver_config.bodies) if b.role == "wheel"]
    assert len(wheels) == 2
    for b, ib in wheels:
        r = float(np.hypot(*(b.points - np.asarray(b.center)).T).mean())
        assert b.omega == pytest.approx(U / r)                       # SI, counter-clockwise positive
        assert ib.omega == pytest.approx(d.length_ref / r)           # solver units: ground speed / r
        assert ib.center == pytest.approx((b.center[0] / d.length_ref, b.center[1] / d.length_ref))
    assert any("counter-clockwise" in n for n in p.notes)
    assert not any("wheels touch the road" in w for w in p.warnings)


@pytest.mark.parametrize("moving,rotate", [(False, True), (True, False)])
def test_wheels_do_not_spin_on_a_fixed_floor_or_when_switched_off(moving, rotate):
    d = templates.make("sedan", wheels=True, wheel_gap=0.08)
    p = plan_simulation(d, make_sim(d, moving=moving, rotate=rotate), device="cpu")
    assert all(ib.omega == 0.0 for ib in p.solver_config.bodies)
    assert p.solver_config.bc_bottom == ("moving" if moving else "noslip")


def test_touching_wheels_are_flagged():
    d = templates.make("sedan", wheels=True)
    p = plan_simulation(d, make_sim(d), device="cpu")
    assert any("wheels touch the road" in w for w in p.warnings)


def test_pitch_into_the_road_is_refused_and_wheels_stay_put():
    d = templates.make("sedan", wheels=True, wheel_gap=0.08)
    with pytest.raises(ValueError, match="below the road"):
        plan_simulation(d, make_sim(d, alpha=8.0), device="cpu")
    p = plan_simulation(d, make_sim(d, alpha=1.0), device="cpu")
    for b0, b in zip(d.bodies, p.bodies_wind):
        if b.role == "wheel":
            assert np.allclose(b0.points, b.points)


def test_impossible_setups_raise_readable_errors():
    a = airfoil()
    with pytest.raises(ValueError, match="speed"):
        plan_simulation(a, make_sim(a, speed=0.0), device="cpu")
    empty = design_from_airfoil(naca("0012", 100, 1.0), "x", 1.0)
    empty.bodies = []
    with pytest.raises(ValueError, match="no bodies"):
        plan_simulation(empty, make_sim(a), device="cpu")


# ---------------------------------------------------------------------------
# summary, notes, durations
# ---------------------------------------------------------------------------

def test_summary_rows_are_complete_and_formatted():
    d = templates.make("sedan")
    p = plan_simulation(d, make_sim(d, quality="draft"), device="cpu")
    s = p.summary()
    for key in ("Real Reynolds number", "Simulated Reynolds number", "Flow model", "Grid", "Cells",
                "Cells per vehicle length", "Domain", "Blockage", "Road", "Time step", "Duration",
                "Averaging from", "Stops early when", "Steps (estimate)", "Estimated run time", "Device"):
        assert key in s and isinstance(s[key], str) and s[key]
    nx, ny = p.grid
    assert s["Grid"] == f"{nx:,} × {ny:,}" and s["Cells"] == f"{p.cells:,}"
    assert s["Road"] == "moving belt" and s["Blockage"].endswith("%")
    assert s["Stops early when"] == (f"Cd known to ±{100 * CONVERGENCE_RTOL[0]:g} %, "
                                     f"Cl to ±{100 * planmod.CL_RTOL_FACTOR * CONVERGENCE_RTOL[0]:g} %")
    assert s["Flow model"].startswith("LES (WALE")


def test_notes_explain_the_plan():
    d = templates.make("sedan")
    p = plan_simulation(d, make_sim(d, quality="draft"), device="cpu")
    text = " ".join(p.notes)
    for fragment in ("Domain:", "blockage", "Grid:", "Flow model:", "Duration:", "Time step", "2D side profile"):
        assert fragment in text
    assert all(isinstance(n, str) and n.strip() and "{" not in n for n in p.notes + p.warnings)


def test_durations_and_convergence_targets_follow_the_quality():
    d = templates.make("sedan")
    plans = [plan_simulation(d, make_sim(d, quality=q), device="cpu") for q in QUALITY_ORDER]
    assert [p.t_end for p in plans] == [planmod.DURATION["vehicle"][i][0] for i in range(3)]
    assert [p.conv_rtol for p in plans] == list(CONVERGENCE_RTOL)
    assert all(0 < p.t_avg_start < p.t_end for p in plans)
    assert all(math.isfinite(p.est_seconds) and p.est_seconds > 0 and p.steps_estimate > 0 for p in plans)


def test_user_duration_and_averaging_start():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, duration=10.0), device="cpu")
    assert p.t_end == 10.0 and p.t_avg_start <= 5.0
    p = plan_simulation(a, make_sim(a, duration=10.0, average_from=9.5), device="cpu")
    assert p.t_avg_start == 5.0 and any("too close to the end" in w for w in p.warnings)


def test_time_step_estimate_respects_the_viscous_limit():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=speed_for(5.0), quality="custom", cells_per_ref=64), device="cpu")
    h = 1.0 / 64
    assert p.dt <= 0.25 * h * h * p.re_sim + 1e-15


def test_live_plan_has_no_end_and_fits_every_pitch():
    a = airfoil()
    p = plan_live(a, make_sim(a, speed=speed_for(500.0), alpha=0.0), device="cpu")
    assert p.live and math.isinf(p.t_end) and p.precision == "float32"
    assert "Duration" not in p.summary() and "Stops early when" not in p.summary()
    assert p.cells <= planmod.LIVE_BUDGET["cpu"]
    for alpha in (-planmod.LIVE_PITCH_DEG, planmod.LIVE_PITCH_DEG):
        for b in planmod.wind_frame_bodies(a, alpha):
            assert p.extent[2] < b.points[:, 1].min() and b.points[:, 1].max() < p.extent[3]


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    """Every test plans from the reference model: runs elsewhere in the session
    (and a calibration file) must not leak measured corrections in."""
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def test_throughput_measurements_correct_the_estimate():
    base = planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale")
    planmod.record_throughput("cuda", 3e5, 2 * base, turbulence="wale", kind="vehicle", speed_factor=3.8)
    assert planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale") == pytest.approx(2 * base)
    assert planmod._speed_factor("vehicle", "les") == pytest.approx(3.8)
    planmod.record_throughput("cuda", 3e5, float("nan"), turbulence="wale")      # ignored
    assert planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale") == pytest.approx(2 * base)


# ---------------------------------------------------------------------------
# first-run defaults, Draft cost, honest warnings (UI critique round 1)
# ---------------------------------------------------------------------------

def test_new_simulations_start_at_draft_with_a_capped_airfoil_reynolds_number():
    a = airfoil("2412")
    cfd = planmod.default_cfd_settings(a, FlowConditions(AIR, speed_for(1e6), 4.0))
    assert cfd.quality == "draft" and cfd.reynolds_mode == "capped"
    assert cfd.reynolds_cap == defaults_module.AIRFOIL_DEFAULT_CAP_RE       # fixed, not left to the grid
    p = plan_simulation(a, make_sim(a, speed=speed_for(1e6), alpha=4.0, **cfd.to_dict()), device="cpu")
    assert p.capped and p.turbulence == "laminar" and p.re_sim < 1e5
    assert any(w.startswith("Cl and Cd describe the capped flow") for w in p.warnings)
    # a low-Re airfoil, a car and a bluff body keep the real Reynolds number
    assert planmod.default_cfd_settings(a, FlowConditions(AIR, speed_for(1e3), 4.0)).reynolds_mode == "real"
    for name in ("sedan", "cylinder"):
        cfd = planmod.default_cfd_settings(templates.make(name), FlowConditions(AIR, 30.0, 0.0))
        assert cfd.quality == "draft" and cfd.reynolds_mode == "real"
    assert planmod.default_cfd_settings(a, device="cpu", precision="float64").device == "cpu"


def test_every_way_of_creating_a_simulation_gets_the_same_defaults():
    """New study, Setup's "New simulation" and the planner share one definition (torch-free)."""
    import inspect

    from aerolume.simulation import defaults
    assert planmod.default_cfd_settings is defaults.default_cfd_settings
    assert planmod.AIRFOIL_CAP_RE == defaults.AIRFOIL_CAP_RE
    a = airfoil("2412")
    for design, mode in ((a, "capped"), (templates.make("sedan"), "real")):
        cfd = defaults.default_cfd_settings(design, FlowConditions(AIR, 30.0, 0.0))
        assert (cfd.quality, cfd.reynolds_mode) == ("draft", mode)
    assert defaults.geometry_class(templates.make("front_wing")) == "airfoil"   # a wing on a road
    src = inspect.getsource(defaults)                 # the GUI imports it on the event thread
    assert "import torch" not in src and "from .plan" not in src and "solvers" not in src


def test_the_sensitivity_map_is_part_of_the_run_time_estimate():
    a = airfoil("0012")
    sim = make_sim(a, speed=speed_for(1e3), alpha=5.0, quality="draft")
    base = plan_simulation(a, sim, device="cpu")
    sim.cfd.compute_sensitivity = True
    with_map = plan_simulation(a, sim, device="cpu")
    extra = with_map.est_seconds - base.est_seconds
    assert extra > 5.0, extra                         # measured: ~60 s on a CPU for this case
    assert with_map.est_seconds_headless - base.est_seconds_headless == pytest.approx(extra)
    assert any(n.startswith("Sensitivity map: about") for n in with_map.notes)
    assert not any(n.startswith("Sensitivity map: about") for n in base.notes)


def test_real_reynolds_airfoil_warning_does_not_call_the_numbers_reliable():
    a = airfoil("2412")
    p = plan_simulation(a, make_sim(a, speed=speed_for(1e6), alpha=4.0, quality="draft"), device="cpu")
    text = " ".join(p.warnings)
    assert "Lift and drag are not reliable" in text and "Polar analysis" in text
    assert "are reliable" not in text
    # a capped run that the grid resolves gets no "boundary layers thinner than the grid" claim
    p = plan_simulation(a, make_sim(a, speed=speed_for(1e6), alpha=4.0, quality="standard",
                                    reynolds_mode="capped"), device="cpu")
    assert not any("thinner than a grid cell" in w for w in p.warnings)


@pytest.mark.parametrize("device", DEVICES)
def test_underbody_gap_warning_leads_vehicle_plans_until_fine(device):
    d = templates.make("sedan")
    for q in ("draft", "standard"):
        p = plan_simulation(d, make_sim(d, quality=q), device=device)
        assert p.warnings[0].startswith("Underbody gap under-resolved (the 150 mm ground clearance spans")
        assert not any("ground clearance spans only" in w for w in p.warnings)      # said once, not twice
    if device == "cuda":
        p = plan_simulation(d, make_sim(d, quality="fine"), device=device)
        assert not any(w.startswith("Underbody gap") for w in p.warnings)
    # free flight has no underbody channel
    p = plan_simulation(d, make_sim(d, ground=False, quality="draft"), device=device)
    assert not any(w.startswith("Underbody gap") for w in p.warnings)


@needs_cuda
def test_gpu_draft_stays_small_for_airfoils_and_bluff_bodies():
    for name in ("naca", "cylinder", "square", "ellipse", "flat_plate", "rounded_rectangle"):
        d = templates.make(name)
        p = plan_simulation(d, make_sim(d, ground=False, quality="draft"), device="cuda")
        assert p.cells <= planmod.CELL_BUDGET["gpu"][0] <= 250_000, name
        assert p.t_end <= planmod.DRAFT_T_END_MAX, name
        # the reference throughput model (no calibration) keeps them under a minute (the thin
        # flat plate, which needs fine cells across its thickness, is the slowest at ~52 s)
        assert p.steps_estimate / planmod._model_rate("gpu", "les", "float32", p.cells) < 60, name


def test_draft_duration_is_capped_for_tall_bluff_bodies():
    t_end, t_avg = planmod._durations("generic", 0, 2.0)          # s = 2 L would give 120 / 60
    assert t_end == planmod.DRAFT_T_END_MAX and t_avg == pytest.approx(0.5 * t_end)
    assert planmod._durations("generic", 1, 2.0) == (160.0, 80.0)  # Standard keeps its long window


@pytest.mark.parametrize("seconds, text", [
    (None, ""), (float("nan"), ""), (0.0, ""), (4.0, "< 10 s"), (12.0, "~10 s"), (41.0, "~40 s"),
    (58.0, "~1 min"), (308.0, "~5 min"), (3600.0, "~1 h"), (4200.0, "~1 h 10 min"),
])
def test_fmt_estimate_for_the_run_button(seconds, text):
    assert planmod.fmt_estimate(seconds) == text


def test_plan_offers_its_estimate_as_a_button_label():
    a = airfoil()
    p = plan_simulation(a, make_sim(a, speed=speed_for(500.0), quality="draft"), device="cpu")
    assert p.estimate_label == planmod.fmt_estimate(p.est_seconds) and p.estimate_label.startswith(("~", "<"))
    assert plan_live(a, make_sim(a, speed=speed_for(500.0)), device="cpu").estimate_label == ""


# ---------------------------------------------------------------------------
# run-time estimate calibration (UI critique round 2: estimates 1.3-2.3x low)
# ---------------------------------------------------------------------------

def test_throughput_corrections_are_local_in_grid_size():
    """A ratio measured on a small grid must not set the estimate of a large one
    (launch overhead vs bandwidth), but one run still calibrates every size."""
    small, large = 1.2e5, 1.0e6
    m = planmod._model_rate
    planmod.record_throughput("cuda", small, 2.0 * m("gpu", "les", "float32", small), turbulence="wale")
    # one measurement only: it is also the machine-wide prior, so it applies everywhere
    assert planmod.estimate_steps_per_second("cuda", large, turbulence="wale") == pytest.approx(
        2.0 * m("gpu", "les", "float32", large))
    planmod.record_throughput("cuda", large, 0.5 * m("gpu", "les", "float32", large), turbulence="wale")
    at_small = planmod.estimate_steps_per_second("cuda", small, turbulence="wale") / m("gpu", "les", "float32", small)
    at_large = planmod.estimate_steps_per_second("cuda", large, turbulence="wale") / m("gpu", "les", "float32", large)
    assert at_small > 1.6 and at_large < 0.65
    # a sibling flavour without measurements borrows the machine-wide median
    lam = planmod.estimate_steps_per_second("cuda", small) / m("gpu", "upwind3", "float32", small)
    assert 0.5 < lam < 2.0


def test_newer_measurements_outweigh_older_ones_at_the_same_size():
    m = planmod._model_rate("gpu", "les", "float32", 3e5)
    planmod.record_throughput("cuda", 3e5, 1.0 * m, turbulence="wale")
    planmod.record_throughput("cuda", 3e5, 3.0 * m, turbulence="wale")
    r = planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale") / m
    assert 2.0 < r < 3.0


def test_speed_factor_defaults_per_class_and_measurements_per_resolution():
    assert planmod._speed_factor("airfoil", "les") < planmod._speed_factor("vehicle", "les")
    assert planmod._speed_factor("unknown", "les") == planmod.SPEED_FACTOR["les"]
    planmod.record_throughput("cuda", 2e5, None, turbulence="wale", kind="generic", speed_factor=3.0,
                              cells_per_ref=24)
    planmod.record_throughput("cuda", 1e6, None, turbulence="wale", kind="generic", speed_factor=5.0,
                              cells_per_ref=96)
    assert planmod._speed_factor("generic", "les", 24) < 3.5
    assert planmod._speed_factor("generic", "les", 96) > 4.5


def test_the_estimate_includes_the_live_view_and_its_measured_cost():
    """The app always shows the live view, which costs ~20 % of the loop; the CLI does not.
    (Steps cost ``step_cost`` times a plain step: the diagnostics are recorded by default.)"""
    d = templates.make("cylinder")
    p = plan_simulation(d, make_sim(d, ground=False, quality="draft"), device="cpu")
    step_s = p.steps_estimate * p.step_cost / planmod.estimate_steps_per_second(
        p.device, p.cells, advection=p.advection, turbulence=p.turbulence, dtype=p.precision)
    assert p.est_seconds - p.est_seconds_headless == pytest.approx(
        step_s * (1 / planmod.LOOP_EFFICIENCY["live"] - 1 / planmod.LOOP_EFFICIENCY["headless"]), rel=1e-6)
    planmod.record_throughput("cpu", p.cells, None, loop_efficiency=0.5, live_view=True)
    p2 = plan_simulation(d, make_sim(d, ground=False, quality="draft"), device="cpu")
    assert p2.est_seconds - p2.est_seconds_headless == pytest.approx(
        step_s * (1 / 0.5 - 1 / planmod.LOOP_EFFICIENCY["headless"]), rel=1e-6)


def test_airfoil_estimate_includes_the_polar_cross_check():
    a = airfoil("2412")
    hi = plan_simulation(a, make_sim(a, speed=speed_for(1e6), alpha=4.0, quality="draft"), device="cpu")
    lo = plan_simulation(a, make_sim(a, speed=speed_for(1e3), alpha=4.0, quality="draft"), device="cpu")

    def overhead(p):
        sps = planmod.estimate_steps_per_second(p.device, p.cells, advection=p.advection,
                                                turbulence=p.turbulence, dtype=p.precision) / p.step_cost
        return p.est_seconds_headless - p.steps_estimate / (sps * planmod.LOOP_EFFICIENCY["headless"])
    assert overhead(hi) - overhead(lo) == pytest.approx(planmod.POLAR_CHECK_S, abs=0.5)
    assert not hi.bl_resolved and lo.bl_resolved


def test_calibration_persists_in_a_file(tmp_path):
    path = tmp_path / "cfg" / planmod.CALIBRATION_FILE
    planmod.use_calibration_file(path)
    m = planmod._model_rate("gpu", "les", "float32", 3e5)
    planmod.record_throughput("cuda", 3e5, 1.7 * m, turbulence="wale", kind="vehicle", speed_factor=3.9,
                              cells_per_ref=126, loop_efficiency=0.85, live_view=True)
    assert path.is_file()
    planmod.use_calibration_file(None)
    planmod.reset_calibration()
    assert planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale") == pytest.approx(m)
    planmod.use_calibration_file(path)                        # a new session
    try:
        assert planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale") == pytest.approx(1.7 * m)
        assert planmod._speed_factor("vehicle", "les", 126) == pytest.approx(3.9)
        assert planmod.loop_efficiency("cuda", 3e5, live_view=True) == pytest.approx(0.85)
    finally:
        planmod.use_calibration_file(None)


@pytest.mark.parametrize("content", [
    "{not json", '{"version": 99}', '{"version": 1, "rate": {"gpu|les|float32": [["x", 1]]}}'])
def test_a_broken_calibration_file_is_ignored(tmp_path, content):
    path = tmp_path / planmod.CALIBRATION_FILE
    path.write_text(content, encoding="utf-8")
    planmod.use_calibration_file(path)
    try:
        m = planmod._model_rate("gpu", "les", "float32", 3e5)
        assert planmod.estimate_steps_per_second("cuda", 3e5, turbulence="wale") == pytest.approx(m)
    finally:
        planmod.use_calibration_file(None)


# ---------------------------------------------------------------------------
# review round: a fixed Reynolds cap, the airfoil domain, memory
# ---------------------------------------------------------------------------

def test_the_default_cap_is_resolved_by_every_quality_on_every_device():
    """AIRFOIL_DEFAULT_CAP_RE re-derived from the planner's own tables: the coarsest
    airfoil preset still puts BL_CELLS_MIN cells across the Blasius layer at that Re."""
    coarsest = min(min(v) for v in planmod.CELLS_PER_SIZE["airfoil"].values())
    resolvable = (5.0 * coarsest / planmod.BL_CELLS_MIN) ** 2
    assert defaults_module.AIRFOIL_DEFAULT_CAP_RE <= resolvable + 1e-9
    assert defaults_module.default_reynolds_cap(airfoil("2412")) == defaults_module.AIRFOIL_DEFAULT_CAP_RE
    assert defaults_module.default_reynolds_cap(templates.make("sedan")) is None


@pytest.mark.parametrize("device", DEVICES)
def test_a_capped_airfoil_simulates_the_same_flow_at_every_quality(device):
    """The review case: NACA 2412 at 30 m/s and 4 deg used to simulate Re 6,400 at Draft and
    25,600 at Standard on a GPU (2,500 and 6,202 on a CPU): stepping up the quality changed
    the flow, not just the grid."""
    a = airfoil("2412")
    cond = FlowConditions(AIR, 30.0, 4.0)
    cfd = planmod.default_cfd_settings(a, cond)
    res = set()
    for q in QUALITY_ORDER:
        sim = make_sim(a, speed=30.0, alpha=4.0, **{**cfd.to_dict(), "quality": q})
        p = plan_simulation(a, sim, device=device)
        res.add(round(p.re_sim))
        assert p.capped and p.bl_resolved and p.turbulence == "laminar", q
        assert not any("No Reynolds cap" in w or "is above what this grid resolves" in w for w in p.warnings)
    assert res == {round(defaults_module.AIRFOIL_DEFAULT_CAP_RE)}


def test_a_capped_simulation_without_a_cap_says_its_reynolds_number_follows_the_grid():
    a = airfoil("2412")
    sim = make_sim(a, speed=30.0, alpha=4.0, quality="draft", reynolds_mode="capped")
    p = plan_simulation(a, sim, device="cpu")
    assert p.capped and p.re_sim == pytest.approx((5.0 * p.cells_per_ref / planmod.BL_CELLS_MIN) ** 2)
    assert any(w.startswith("No Reynolds cap is set") for w in p.warnings)


def test_a_cap_the_grid_cannot_resolve_is_warned_about_and_gets_a_model():
    a = airfoil("2412")
    sim = make_sim(a, speed=30.0, alpha=4.0, quality="draft", reynolds_mode="capped", reynolds_cap=20_000.0)
    p = plan_simulation(a, sim, device="cpu")
    assert p.re_sim == pytest.approx(20_000.0) and not p.bl_resolved
    assert p.turbulence.startswith("les")
    assert any("is above what this grid resolves" in w for w in p.warnings)


@pytest.mark.parametrize("device", DEVICES)
def test_airfoil_domain_is_larger_and_its_bias_is_stated(device):
    """The 2-chord inlet and 4-chord height biased the NACA 2412 Draft run by +4 % Cd, +6 % Cl
    and +7 % Cm; Draft and Standard now share about 4 chords ahead and 6 of height, and every
    airfoil plan states what bias remains (AIRFOIL_DOMAIN_NOTE)."""
    a = airfoil("2412")
    for q in ("draft", "standard"):
        p = plan_simulation(a, make_sim(a, speed=speed_for(1e6), alpha=4.0, quality=q,
                                        reynolds_mode="capped", reynolds_cap=2500.0), device=device)
        x_le = min(float(np.asarray(b.points)[:, 0].min()) for b in p.bodies_wind)
        upstream = (x_le - p.extent[0]) / p.length_ref
        height = (p.extent[3] - p.extent[2]) / p.length_ref
        dclass = "gpu" if device == "cuda" else "cpu"
        if q == "draft":
            assert upstream >= 3.99 and height >= 5.99, (q, upstream, height)
        elif dclass == "gpu":
            assert upstream >= 3.5 and height >= 5.5, (q, upstream, height)
        assert p.cells <= planmod.CELL_BUDGET[dclass][QUALITY_ORDER.index(q)]
        assert p.cells_per_ref >= planmod.THIN_SLACK * planmod.CELLS_PER_SIZE["airfoil"][dclass][
            QUALITY_ORDER.index(q)]
        assert planmod.AIRFOIL_DOMAIN_NOTE in p.notes


def test_a_grid_that_cannot_fit_in_memory_is_refused(monkeypatch):
    """'Cells per chord' at the old spin-box maximum planned 806 M cells with only a warning."""
    from aerolume.core import device as devmod
    a = airfoil("2412")
    with pytest.raises(ValueError, match="needs about .* GB"):
        plan_simulation(a, make_sim(a, quality="custom", cells_per_ref=4096), device="cpu")
    # the guard follows the machine: 1 GB of RAM refuses what 64 GB accepts
    sim = make_sim(a, quality="custom", cells_per_ref=600, precision="float64")
    monkeypatch.setattr(devmod, "system_memory_gb", lambda: 64.0)
    assert plan_simulation(a, sim, device="cpu").cells > 5_000_000
    monkeypatch.setattr(devmod, "system_memory_gb", lambda: 1.0)
    with pytest.raises(ValueError, match="Lower the cells per chord"):
        plan_simulation(a, sim, device="cpu")
    # presets never trip it
    monkeypatch.setattr(devmod, "system_memory_gb", lambda: 1.0)
    plan_simulation(a, make_sim(a, quality="fine"), device="cpu")


def test_a_cuda_index_that_does_not_exist_falls_back_with_a_warning():
    from aerolume.core import device as devmod
    with pytest.raises(RuntimeError):
        devmod.resolve("cuda:x" if torch.cuda.is_available() else "cuda")
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="not available"):
            devmod.resolve(f"cuda:{torch.cuda.device_count()}")
        assert devmod.resolve("cuda:0") == "cuda:0"
    a = airfoil("0012")
    p = plan_simulation(a, make_sim(a, speed=speed_for(1e3), quality="draft", device="cuda:7"))
    assert any("'cuda:7' is not available" in w for w in p.warnings)
