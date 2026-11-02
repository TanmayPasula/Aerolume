"""Critic round 3.2 (sim): time steps and estimates from the speeds the placed physics produces,
buoyancy-dominated tunnel runs, one symmetry decision for one body, one ΔT_ref per plan.

The measured numbers these tests pin (enclosure 121,433 steps in 5 min 06 s, the 3D room
62 steps/s at 0.0019 L/U, the hot sphere's plume at 4.7 u_b) are in the comments of
``simulation.plan.SPEED_FACTOR_KIND`` and ``simulation.plan3d.SPEED_FACTOR3``.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aerolume.core import fluid as F
from aerolume.geometry import templates
from aerolume.geometry3d import design_from_meshes, sphere
from aerolume.model import (
    CFDSettings,
    Design,
    Emitter,
    Fan,
    FlowConditions,
    Fluid,
    HeatSource,
    Port,
    Simulation,
    Stat,
    ThermalSettings,
)
from aerolume.simulation import physics as PH
from aerolume.simulation import plan as planmod
from aerolume.simulation import plan3d as P3
from aerolume.simulation import runner as R2
from aerolume.simulation.plan import plan_simulation

AIR = F.air_isa(0.0)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
#: what a review measured for the Draft enclosure template on the reference machine (RTX 4060 Ti,
#: shared GPU) with an earlier development build: 121,433 steps at a CFL-limited 0.00028 L/U in
#: 5 min 06 s (the plan then said 28,321 steps and 1 min 05 s)
ENCLOSURE_MEASURED_STEPS = 121_433
#: the same Draft enclosure run headless on the reference machine with the GPU otherwise idle: 153.1 and
#: 151.9 s on 31 October 2026 (validation/records/planner-timing-2026-10-31.json; the planner's GPU
#: throughput table is an idle GPU's since then, so a shared GPU, like that review's 306 s run, runs slower)
ENCLOSURE_MEASURED_S = 152.0


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    """Keep run-time measurements and any calibration file out of these tests."""
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def enclosure():
    """The Start page's electronics enclosure (app.dialogs.physics_templates), as the user gets it."""
    pytest.importorskip("PySide6")
    from aerolume.app.dialogs.physics_templates import build_template
    d, sims = build_template("enclosure", device="cpu")
    return d, sims[0]


def hot_sphere(**sim_kw):
    """The critic's hot sphere: D 0.1 m at Re 100, 60 °C in 20 °C air with buoyancy (Ri 627)."""
    des = design_from_meshes([sphere(0.1, 32)], "Hot sphere", kind="generic")
    for m in des.meshes:
        m.thermal, m.thermal_value = "temperature", 60.0
    air = F.air(20.0)
    U = 100.0 * air.kinematic_viscosity / des.length_ref
    sim = Simulation("s", des.id, conditions=FlowConditions(air, U, 0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", reynolds_mode="real"),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True), **sim_kw)
    return des, sim


def closed_box(dim: int = 2, **kw):
    if dim == 2:
        des = Design(name="Box", kind="generic", bodies=[], length_ref=0.3, force_ref=0.1, moment_ref=(0.15, 0.06))
        box = (0.0, 0.3, 0.0, 0.12)
    else:
        des = Design(name="Box3", kind="generic", bodies=[], length_ref=0.3, force_ref=0.1, dimension=3,
                     ref_area=0.09, span_ref=0.3, moment_ref3=(0.15, 0.0, 0.06))
        box = (0.0, 0.3, -0.1, 0.1, 0.0, 0.12)
    cfd = CFDSettings(quality="draft", device="cpu", domain_mode="closed", domain_box=box)
    return des, Simulation("c", des.id, conditions=FlowConditions(AIR, 1.0, 0.0), cfd=cfd, **kw)


# ---------------------------------------------------------------------------
# the peak speed
# ---------------------------------------------------------------------------

def test_the_peak_speed_is_the_fastest_source():
    rho = AIR.density
    fan_v = Fan(name="F", position=(0.05, 0.08), diameter=0.04, mode="velocity", value=2.0, frame="wind")
    jet = Port(name="J", kind="jet", position=(0.004, 0.06), size=0.03, speed=1.5, frame="wind")
    des, sim = closed_box(fans=[fan_v], ports=[jet])
    s = PH.physics_scales(des, sim, 2, box=sim.cfd.domain_box)
    assert s.speed == pytest.approx(2.0)                             # the reference stays the fan's speed
    u, basis = PH.peak_speed(s, sim, 2)
    assert u == pytest.approx(PH.DISC_WAKE_FACTOR * 2.0) and PH.peak_source(basis) == "fan"
    # a thrust fan: twice its static momentum-theory disc velocity sqrt(T / (2 rho A))
    fan_t = Fan(name="T", position=(0.05, 0.08), diameter=0.04, mode="thrust", value=0.5, frame="wind")
    des, sim = closed_box(fans=[fan_t])
    s = PH.physics_scales(des, sim, 2, box=sim.cfd.domain_box)
    assert PH.peak_speed(s, sim, 2)[0] == pytest.approx(2.0 * math.sqrt(0.5 / (2 * rho * 0.04)))
    # a jet faster than everything else
    des, sim = closed_box(ports=[Port(name="Fast", kind="jet", position=(0.004, 0.06), size=0.03, speed=6.0,
                                      frame="wind")], fans=[fan_v])
    s = PH.physics_scales(des, sim, 2, box=sim.cfd.domain_box)
    u, basis = PH.peak_speed(s, sim, 2)
    assert u == pytest.approx(6.0) and PH.peak_source(basis) == "jet"
    # no placed physics, no buoyancy: the reference itself
    des = templates.make("cylinder", diameter=0.1)
    sim = Simulation("t", des.id, conditions=FlowConditions(AIR, 3.0, 0.0), cfd=CFDSettings(device="cpu"),
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False))
    s = PH.physics_scales(des, sim, 2)
    assert PH.peak_speed(s, sim, 2) == (pytest.approx(3.0), "free-stream speed")


def test_the_buoyant_speed_is_free_fall_or_viscous():
    des, sim = hot_sphere()
    s = PH.physics_scales(des, sim, 3)
    u_ff = math.sqrt(s.g_beta * s.delta_t * s.length_ra)
    assert s.gr > 1 and PH.buoyant_speed(s) == pytest.approx(u_ff, rel=1e-12)
    assert s.ri_flow == pytest.approx((u_ff / s.speed) ** 2, rel=1e-9)
    # a tiny body: Gr < 1, the Stokes balance sqrt(Gr) u_ff is the smaller scale
    small = design_from_meshes([sphere(0.0002, 16)], "Speck", kind="generic")
    for m in small.meshes:
        m.thermal, m.thermal_value = "temperature", 21.0
    sim2 = Simulation("s", small.id, conditions=FlowConditions(F.air(20.0), 0.01, 0.0), cfd=CFDSettings(device="cpu"),
                      thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True))
    s2 = PH.physics_scales(small, sim2, 3)
    assert s2.gr < 1
    assert PH.buoyant_speed(s2) == pytest.approx(math.sqrt(s2.gr * s2.g_beta * s2.delta_t * s2.length_ra), rel=1e-9)
    sim.thermal.buoyancy = False
    assert PH.buoyant_speed(PH.physics_scales(des, sim, 3)) == 0.0


# ---------------------------------------------------------------------------
# the enclosure: planned against measured
# ---------------------------------------------------------------------------

def test_the_enclosure_plans_the_step_its_run_measured():
    """The blocker: the enclosure planned 28,321 steps (dt 0.0011 L/U) and took 121,433 (dt 0.00028).
    Its fan's slipstream (2 × 2 m/s) sets the peak, and the closed-fan speed factor is the one
    this very case measured: the planned step count is now within 1.5× of the measured one."""
    d, sim = enclosure()
    p = plan_simulation(d, sim)
    assert planmod.base_kind(p.calibration_kind) == "closed-fan" and p.speed == pytest.approx(2.0)
    assert p.speed_ratio == pytest.approx(2.0) and PH.peak_source(p.peak_basis) == "fan"
    cfg = p.solver_config
    kfac = planmod._speed_factor(p.calibration_kind, "laminar", p.cells_per_ref) * p.speed_ratio
    assert p.dt <= planmod.CFL * cfg.h / kfac * (1 + 1e-12)
    ratio = p.steps_estimate / ENCLOSURE_MEASURED_STEPS
    assert 1 / 1.5 <= ratio <= 1.5, (p.steps_estimate, ENCLOSURE_MEASURED_STEPS)
    assert any("sized for the slipstream of fan" in n for n in p.notes)
    assert p.est_seconds_planned == pytest.approx(p.est_seconds)


@needs_cuda
def test_the_enclosure_estimate_on_the_reference_gpu_matches_the_measured_wall_time():
    """On the GPU the plan's own estimate (the reference machine's throughput table) is within
    1.5× of the 152 s the run takes on an idle GPU (it said 1 min 05 s for a 5 min run before round 3.2)."""
    d, sim = enclosure()
    sim.cfd.device = "cuda"
    p = plan_simulation(d, sim)
    if "4060 Ti" not in p.device_name:
        pytest.skip("the throughput table is the reference machine's")
    ratio = p.est_seconds / ENCLOSURE_MEASURED_S
    assert 1 / 1.5 <= ratio <= 1.5, (p.est_seconds, ENCLOSURE_MEASURED_S)


@pytest.mark.slow
@needs_cuda
def test_the_enclosure_run_takes_about_its_estimate():
    """The full Draft enclosure, headless: the wall time is within 1.5× of the plan's headless
    estimate (about 2 min 30 s on an idle RTX 4060 Ti; a GPU shared with other work runs slower)."""
    d, sim = enclosure()
    sim.cfd.device = "cuda"
    p = plan_simulation(d, sim)
    t0 = time.perf_counter()
    R2.run_cfd(p)
    wall = time.perf_counter() - t0
    assert 1 / 1.5 <= wall / p.est_seconds_headless <= 1.5, (wall, p.est_seconds_headless)


def test_a_closed_box_without_fans_keeps_its_jet_reference():
    des, sim = closed_box(ports=[Port(name="J", kind="jet", position=(0.004, 0.06), size=0.03, speed=1.5,
                                      frame="wind"),
                                 Port(name="V", kind="vent", position=(0.296, 0.06), size=0.03, speed=1.5,
                                      frame="wind")])
    p = plan_simulation(des, sim)
    assert planmod.base_kind(p.calibration_kind) == "closed-forced" and p.speed_ratio == pytest.approx(1.0)


def test_a_3d_room_with_a_fan_plans_on_the_fans_slipstream():
    des, sim = closed_box(3, fans=[Fan(name="Ceiling fan", position=(0.15, 0.0, 0.1), direction=(0, 0, -1),
                                       diameter=0.05, value=1.0, frame="wind")])
    p = P3.plan_simulation3d(des, sim)
    assert planmod.base_kind(p.calibration_kind) == "closed-fan" and p.speed_ratio == pytest.approx(2.0)
    cfg = p.solver_config
    kfac = P3._speed_factor3(p.calibration_kind, "laminar", p.cells_per_ref) * 2.0
    assert p.dt <= planmod.CFL * cfg.h / kfac * (1 + 1e-12)


# ---------------------------------------------------------------------------
# buoyancy-dominated tunnel runs
# ---------------------------------------------------------------------------

def test_a_buoyancy_dominated_tunnel_run_is_planned_on_its_plume_and_says_so():
    des, sim = hot_sphere()
    p = P3.plan_simulation3d(des, sim)
    s = p.physics
    assert s.ri_flow > 500 and s.regime == "Natural"             # 590 in 20 °C air (627 in ISA air)
    assert p.calibration_kind == "generic-plume"
    assert p.speed_ratio == pytest.approx(math.sqrt(s.ri_flow), rel=1e-9)        # u_b / U = sqrt(Ri)
    kfac = P3.SPEED_FACTOR3[("generic-plume", "laminar")] * p.speed_ratio
    assert p.dt <= planmod.CFL * p.solver_config.h / kfac * (1 + 1e-12)
    bd = [w for w in p.warnings if w.startswith("Buoyancy dominates this run")]
    assert len(bd) == 1 and f"Ri = Gr/Re² = {PH.fmt_number(s.ri_flow)}" in bd[0]
    assert "not aerodynamic coefficients" in bd[0]
    assert p.warnings[0] == bd[0]                                    # first: a results banner shows the first
    assert any(n.startswith("The plume rises") for n in p.notes)
    # buoyancy off: a forced run on the free stream, as before
    sim.thermal.buoyancy = False
    p0 = P3.plan_simulation3d(des, sim)
    assert p0.speed_ratio == 1.0 and p0.calibration_kind == "generic-physics"
    assert not any(w.startswith("Buoyancy dominates") for w in p0.warnings)


def test_mixed_convection_is_not_called_buoyancy_dominated():
    des = templates.make("cylinder", diameter=0.02)
    des.bodies[0].thermal, des.bodies[0].thermal_value = "temperature", 30.0
    air = F.air(20.0)
    sim = Simulation("t", des.id, conditions=FlowConditions(air, 40 * air.kinematic_viscosity / 0.02, 0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", reynolds_mode="real"),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True))
    p = plan_simulation(des, sim)
    assert 1 < p.physics.ri_flow < PH.RI_BUOYANT
    assert p.calibration_kind == "generic-plume" and p.speed_ratio > 1    # the plume still outruns U
    assert not any(w.startswith("Buoyancy dominates") for w in p.warnings)


def test_the_result_reports_coefficients_on_the_buoyant_scale():
    des, sim = hot_sphere()
    p = P3.plan_simulation3d(des, sim)
    res = SimpleNamespace(cd=Stat(mean=5.78), cl=Stat(mean=120.3), cs=Stat(mean=0.0))
    note = PH.buoyant_coefficients_note(res, p)
    f = 1.0 / p.physics.ri_flow
    assert note.startswith("On the buoyant scale q_b = ½ρu_b²")
    assert f"Cd {PH.units.sig(5.78 * f, 3)}" in note and f"Cl {PH.units.sig(120.3 * f, 3)}" in note
    assert "Cs" not in note                                        # zero by construction on the symmetry plane
    sim.thermal.buoyancy = False
    assert PH.buoyant_coefficients_note(res, P3.plan_simulation3d(des, sim)) == ""


# ---------------------------------------------------------------------------
# one symmetry decision for one body
# ---------------------------------------------------------------------------

def test_a_smoke_rake_on_the_centre_plane_keeps_the_symmetry_plane():
    """The critic planned the hot sphere without its rake (320×52×104, half domain) and ran it with
    one (240×88×88, full domain, 3.7 % blockage): the rake lies in y = 0 and is its own mirror image."""
    rake = Emitter(name="Rake", kind="rake", position=(-0.15, 0.0, -0.04), end=(-0.15, 0.0, 0.04), count=5,
                   color="#f05040")
    des, plain = hot_sphere()
    _, with_rake = hot_sphere(emitters=[rake])
    a, b = P3.plan_simulation3d(des, plain), P3.plan_simulation3d(des, with_rake)
    assert a.symmetry_plane and b.symmetry_plane
    assert a.grid3 == b.grid3 and a.extent3 == pytest.approx(b.extent3) and a.blockage == pytest.approx(b.blockage)
    assert any("own mirror images" in n for n in b.notes)
    # items that are not their own mirror images still need both halves
    off = Emitter(name="Off", kind="point", position=(-0.15, 0.02, 0.0))
    swirl = Fan(name="Swirl", position=(0.3, 0.0, 0.0), direction=(1, 0, 0), diameter=0.05, value=0.5, swirl=0.3,
                frame="wind")
    across = Port(name="Across", kind="jet", position=(0.3, 0.0, 0.0), direction=(0, 1, 0), size=0.02, speed=0.1,
                  frame="wind")
    heater = HeatSource(name="H", kind="sphere", position=(0.3, 0.0, 0.0), size=(0.01,), power=1.0, frame="wind")
    for kw, sym in (({"emitters": [off]}, False), ({"fans": [swirl]}, False), ({"ports": [across]}, False),
                    ({"heat_sources": [heater]}, True)):
        _, sim = hot_sphere(**kw)
        p = P3.plan_simulation3d(des, sim)
        assert p.symmetry_plane is sym, kw
        if not sym:
            assert any("off the centre plane" in n for n in p.notes)


def test_the_run_uses_the_plans_grid():
    """What runs is the plan's solver config: the result reports the plan's cells and grid."""
    des, sim = hot_sphere(emitters=[Emitter(name="Rake", kind="rake", position=(-0.15, 0.0, -0.04),
                                            end=(-0.15, 0.0, 0.04), count=5)])
    p = P3.plan_simulation3d(des, sim)
    q = P3.plan_simulation3d(des, sim)
    assert p.grid3 == q.grid3 and p.symmetry_plane == q.symmetry_plane      # deterministic for one input
    cfg = p.solver_config
    assert (cfg.nx, cfg.ny, cfg.nz) == p.grid3 and cfg.symmetry is p.symmetry_plane


# ---------------------------------------------------------------------------
# one ΔT_ref, one line per quantity
# ---------------------------------------------------------------------------

def test_warnings_keep_one_line_per_quantity():
    plan_line = "β·ΔT = 0.15 (ΔT 43.4 K): above about 0.1 the Boussinesq approximation loses accuracy."
    page_line = "β·ΔT = 0.20 (ΔT 57.8 K): above about 0.1 the Boussinesq approximation loses accuracy."
    other = "Something else entirely."
    assert PH.warning_key(plan_line) == PH.warning_key(page_line) == "boussinesq"
    assert PH.merge_warnings([plan_line, other], [page_line, other]) == [plan_line, other]
    assert PH.warning_key("Ra = 1.30×10^11: above about 10⁸ ...") == PH.warning_key("Ra = 1.7×10¹¹: above")


def test_a_room_plan_quotes_one_delta_t_and_one_rayleigh_number():
    des = Design(name="Room", kind="generic", bodies=[], length_ref=0.5, force_ref=0.5, moment_ref=(2, 1.5))
    sim = Simulation("c", des.id, conditions=FlowConditions(F.air(20.0), 1.0, 0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", domain_mode="closed", domain_box=(0, 4, 0, 3)),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True),
                     fans=[Fan(name="Fan", position=(4 / 3, 1.5), direction=(1, 0), diameter=0.5, value=1.0,
                               frame="wind")],
                     heat_sources=[HeatSource(name="Heater", kind="rect", position=(3.5, 0.3), size=(0.3, 0.2),
                                              power=500.0, frame="wind")])
    p = plan_simulation(des, sim)
    keys = [PH.warning_key(w) for w in p.warnings]
    assert len(keys) == len(set(keys))
    dts = {w.split("(ΔT ")[1].split(" K")[0] for w in p.warnings if w.startswith("β·ΔT")}
    assert dts <= {PH.units.sig(p.physics.delta_t, 3)}
    assert not any(w.startswith("The fluid is") for w in p.warnings)        # fluid and ambient agree
    assert p.summary()["Rayleigh number"] == PH.fmt_number(p.physics.ra)


def test_the_default_ambient_follows_the_fluid():
    assert PH.default_ambient(F.air_isa(0.0)) == pytest.approx(15.0)
    assert PH.default_ambient(F.air(25.0)) == pytest.approx(25.0)
    assert PH.default_ambient(F.water(40.0)) == pytest.approx(40.0)
    assert PH.default_ambient(Fluid("Oil", 900.0, 0.09)) == 20.0


# ---------------------------------------------------------------------------
# the runner: measured re-estimate, calibration per peak speed
# ---------------------------------------------------------------------------

def test_the_runner_re_estimates_from_its_measured_rate():
    plan = SimpleNamespace(live=False, est_seconds=60.0, est_seconds_planned=60.0, est_after_seconds=5.0)
    R2.refine_estimate(plan, elapsed=30.0, t=2.0, t_end=10.0, wall_rate=0.02)      # 0.02 L/U per second
    assert plan.est_seconds == pytest.approx(30.0 + 8.0 / 0.02 + 5.0 + 1.0)
    assert plan.est_seconds_planned == 60.0
    for bad in (math.nan, 0.0, -1.0):
        R2.refine_estimate(plan, elapsed=1.0, t=0.0, t_end=1.0, wall_rate=bad)
        assert plan.est_seconds == pytest.approx(436.0)
    live = SimpleNamespace(live=True, est_seconds=math.nan)
    R2.refine_estimate(live, elapsed=1.0, t=0.0, t_end=1.0, wall_rate=1.0)
    assert math.isnan(live.est_seconds)


def test_the_measured_speed_factor_is_recorded_per_peak_speed(monkeypatch):
    seen = {}
    monkeypatch.setattr(R2, "record_throughput", lambda *a, **kw: seen.update(kw))
    cfg = SimpleNamespace(cfl=0.5, h=0.01, advection="upwind3", turbulence="none", dtype="float32")
    plan = SimpleNamespace(solver_config=cfg, placed=None, speed_ratio=2.0, calibration_kind="closed-fan",
                           device="cpu", cells=1000, cells_per_ref=64.0, step_cost=1.0,
                           design=SimpleNamespace(kind="generic"))
    # 1000 steps over t = 0 .. 1: mean dt 0.001, so cfl h / dt = 5.0, i.e. 2.5 per peak speed
    R2._record_measurements(plan, steps_timed=1000, step_wall=10.0, loop_wall=11.0, t_first=0.0, t_final=1.0,
                            t_end_planned=1.0, live_view=False, solver=None)
    assert seen["kind"] == "closed-fan" and seen["speed_factor"] == pytest.approx(2.5)


def test_a_plain_plan_is_unchanged_by_the_peak_speed():
    des = templates.make("cylinder", diameter=0.1)
    sim = Simulation("t", des.id, conditions=FlowConditions(AIR, 1.0, 0.0), cfd=CFDSettings(device="cpu"))
    p = plan_simulation(des, sim)
    assert p.speed_ratio == 1.0 and p.calibration_kind == "generic" and p.peak_basis == "free-stream speed"
    kfac = planmod._speed_factor("generic", "laminar" if p.turbulence == "laminar" else "les", p.cells_per_ref)
    h = p.solver_config.h
    nu_max = 1.0 / p.re_sim + (planmod.NUT_PEAK * h if p.turbulence != "laminar" else 0.0)
    assert p.dt == pytest.approx(min(planmod.CFL * h / kfac, 0.25 * h * h / nu_max), rel=1e-12)
    assert np.isfinite(p.est_seconds_planned)
