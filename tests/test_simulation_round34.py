"""Critic round 3.4 (sim): the run-time calibration, buoyant thermal layers, closed-box monitors and the
heater's estimated rise.

* The CFL speed factor is kept per case signature (a fan alone, with jets or heaters, with bodies, the
  grid class), each run moves it by at most 1.5 times, and the planner takes the weighted median: other
  rooms' runs no longer push the critic's room from 52 s to 164 s (it ran in 49 s).
* Explore sessions never feed the calibration.
* A buoyancy-dominated run sizes its thermal boundary layer on the plume (the hot sphere: about 2 mm,
  under half a cell, not "12 cells"), and natural or mixed convection around a body says it is not
  validated (docs/validation.md §10).
* A closed box is never judged on force coefficients; it is judged on its temperatures, heat rates,
  fans and kinetic energy, and the messages name them.
* A heater's rise is the air its plume carries through it (4.1 K for the 300 W/m room the old
  q L / (k Nu) put at 44.1 K, measured 3.3 K); a heater-only room gets a box-sized grid and a Draft
  run of tens of free-fall times; the duration label is built from the duration used.
"""
from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_simulation_peak_speed import closed_box, enclosure, hot_sphere
from test_simulation_physics import closed_sim, room2d

from aerolume.core import fluid as F
from aerolume.model import Fan, HeatSource, Port, ThermalSettings, WallThermal
from aerolume.simulation import live, live3d
from aerolume.simulation import physics as PH
from aerolume.simulation import physics_run as PR
from aerolume.simulation import plan as planmod
from aerolume.simulation import plan3d as P3
from aerolume.simulation import runner as R2
from aerolume.simulation.plan import plan_simulation

#: Measured on the reference machine (RTX 4060 Ti, 31 October 2026; validation/records/planner-timing-2026-10-31.json):
#: the speed factors (relative to the peak speed) the Draft runs of the review's room (fan, heater, supply
#: jet, vent; 62.0 s), the plain fan room and the electronics enclosure recorded under their own keys.
ROOM_FACTOR = 3.87
FAN_ROOM_FACTOR = 5.09
ENCLOSURE_FACTOR = 2.68


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def fan(**kw):
    return Fan(name="Fan", position=(4 / 3, 1.5), direction=(1, 0), diameter=0.5, value=1.0, frame="wind", **kw)


def heater(power: float = 300.0, position=(3.5, 0.3), size=(0.3, 0.2)):
    return HeatSource(name="Heater", kind="rect", position=position, size=size, power=power, frame="wind")


def critic_room(device: str = "cpu", **cfd_kw):
    """The critic's Draft room: fan_room_case (1 N/m in 4 × 3 m of ISA air) with a 300 W/m heater, a supply
    jet and a vent, as the review ran it."""
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True),
                     fans=[fan()], heat_sources=[heater()],
                     ports=[Port(name="Supply", kind="jet", position=(0.05, 2.7), direction=(1, 0), size=0.2,
                                 speed=0.5, frame="wind"),
                            Port(name="Vent", kind="vent", position=(3.95, 2.7), direction=(1, 0), size=0.2,
                                 speed=0.5, frame="wind")], cfd_kw=cfd_kw)
    sim.cfd.device = device
    return des, sim


def plain_fan_room():
    des = room2d()
    return des, closed_sim(des, (0, 4, 0, 3), fans=[fan()])


def heater_only_room(**cfd_kw):
    des = room2d()
    return des, closed_sim(des, (0, 4, 0, 3), thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True),
                           heat_sources=[heater(position=(2.0, 0.3), size=(0.4, 0.2))], cfd_kw=cfd_kw)


def model_of(p) -> str:
    return "laminar" if p.turbulence == "laminar" else "les"


# ---------------------------------------------------------------------------
# run-time calibration
# ---------------------------------------------------------------------------

def test_closed_boxes_are_calibrated_per_case_signature():
    room, plain = plan_simulation(*critic_room()), plan_simulation(*plain_fan_room())
    encl = plan_simulation(*enclosure())
    assert planmod.base_kind(room.calibration_kind) == "closed-fan"
    assert "+jet" in room.calibration_kind and "+heat" in room.calibration_kind and "@" in room.calibration_kind
    assert "+jet" not in plain.calibration_kind and "+heat" not in plain.calibration_kind
    assert "+body" in encl.calibration_kind and "+body" not in room.calibration_kind
    assert len({room.calibration_kind, plain.calibration_kind, encl.calibration_kind}) == 3
    # the grid class follows the cells across the smaller side
    assert planmod.calibration_kind("generic", None, closed=True, natural=True, speed_ratio=1.0,
                                    cells_across=72) == "closed-natural@64"
    assert planmod.base_kind("closed-fan+jet+heat+body@128") == "closed-fan"
    # a signature without a measured default starts from its class's
    assert planmod._speed_factor("closed-fan+body@64", "les") == planmod.SPEED_FACTOR_KIND[("closed-fan", "les")]
    assert planmod.signature_defaults("closed-fan+jet+heat@64") == ["closed-fan+jet+heat", "closed-fan+jet",
                                                                     "closed-fan"]


def test_other_rooms_runs_leave_the_rooms_estimate_alone():
    """The finding's regression: plan the room after its own run, record a fan_room run and an enclosure run
    with the factors they measured, and the room's plan does not move (it moved 3.1 times before, critic
    round 3.4: 52 s planned after its own 49 s run, then 164 s)."""
    room = plan_simulation(*critic_room())
    planmod.record_throughput("cuda", room.cells, None, turbulence="wale", kind=room.calibration_kind,
                              speed_factor=ROOM_FACTOR, cells_per_ref=room.cells_per_ref)
    calibrated = plan_simulation(*critic_room()).steps_estimate
    plain = plan_simulation(*plain_fan_room())
    encl = plan_simulation(*enclosure())
    for _ in range(3):
        planmod.record_throughput("cuda", plain.cells, None, turbulence="wale", kind=plain.calibration_kind,
                                  speed_factor=FAN_ROOM_FACTOR, cells_per_ref=plain.cells_per_ref)
        planmod.record_throughput("cuda", encl.cells, None, turbulence="none", kind=encl.calibration_kind,
                                  speed_factor=ENCLOSURE_FACTOR, cells_per_ref=encl.cells_per_ref)
    assert plan_simulation(*critic_room()).steps_estimate == calibrated
    # and the class defaults are the factors these cases measured, so even a first plan is close
    assert planmod._speed_factor(room.calibration_kind, "les") == pytest.approx(ROOM_FACTOR, rel=0.05)
    assert planmod._speed_factor("closed-fan@64", "les") == pytest.approx(FAN_ROOM_FACTOR, rel=0.05)
    assert planmod._speed_factor("closed-fan+jet+heat+body@128", "laminar", 400.0) == pytest.approx(
        ENCLOSURE_FACTOR, rel=0.05)


def test_one_run_moves_a_speed_factor_by_at_most_1_5_times_and_the_median_ignores_an_outlier():
    kind, model = "closed-fan+jet+heat@64", "les"
    default = planmod._speed_factor(kind, model, 12.0)
    planmod.record_throughput("cuda", 7000, None, turbulence="wale", kind=kind, speed_factor=10 * default,
                              cells_per_ref=12.0)
    assert planmod._speed_factor(kind, model, 12.0) == pytest.approx(planmod.SPEED_UPDATE_MAX * default)
    # three consistent runs on another key settle it exactly; one far-off run after them moves nothing
    other = "closed-fan+jet@64"
    for _ in range(3):
        planmod.record_throughput("cuda", 7000, None, turbulence="wale", kind=other, speed_factor=0.9 * default,
                                  cells_per_ref=12.0)
    settled = planmod._speed_factor(other, model, 12.0)
    assert settled == pytest.approx(0.9 * default)
    planmod.record_throughput("cuda", 7000, None, turbulence="wale", kind=other, speed_factor=5 * default,
                              cells_per_ref=12.0)
    assert planmod._speed_factor(other, model, 12.0) == pytest.approx(settled)
    assert planmod.clamp_update(9.0, 2.0) == pytest.approx(3.0)
    assert planmod.clamp_update(0.5, 2.0) == pytest.approx(2.0 / 1.5)
    assert planmod.clamp_update(2.5, None) == 2.5


def test_3d_speed_factors_are_clamped_and_median_too():
    kind = "closed-fan+jet+heat+body@32"
    default = P3._speed_factor3(kind, "laminar", 20.0)
    assert default == P3.SPEED_FACTOR3[("closed-fan", "laminar")]
    P3.record_throughput3d("cuda", 1e6, None, kind=kind, speed_factor=20 * default, cells_per_ref=20.0)
    assert P3._speed_factor3(kind, "laminar", 20.0) == pytest.approx(1.5 * default)


def test_explore_sessions_never_feed_the_calibration(monkeypatch):
    seen = []
    monkeypatch.setattr(R2, "record_throughput", lambda *a, **kw: seen.append(kw))
    cfg = SimpleNamespace(cfl=0.5, h=0.01, advection="upwind3", turbulence="none", dtype="float32")
    plan = SimpleNamespace(solver_config=cfg, placed=None, speed_ratio=1.0, calibration_kind="generic", live=True,
                           device="cpu", cells=1000, cells_per_ref=64.0, step_cost=1.0,
                           design=SimpleNamespace(kind="generic"))
    R2._record_measurements(plan, steps_timed=1000, step_wall=10.0, loop_wall=11.0, t_first=0.0, t_final=1.0,
                            t_end_planned=1.0, live_view=True, solver=None)
    assert seen == []
    plan.live = False
    R2._record_measurements(plan, steps_timed=1000, step_wall=10.0, loop_wall=11.0, t_first=0.0, t_final=1.0,
                            t_end_planned=1.0, live_view=True, solver=None)
    assert len(seen) == 1
    # the explore sessions (with their drawn obstacles and live edits) do not call the recorder at all
    for mod in (live, live3d):
        assert "record_throughput" not in inspect.getsource(mod)


# ---------------------------------------------------------------------------
# heaters, durations and grids of rooms
# ---------------------------------------------------------------------------

def test_a_heaters_rise_is_the_air_its_plume_carries_through_it():
    p = plan_simulation(*critic_room())
    s = p.physics
    assert s.delta_t_basis.startswith(PH.ESTIMATED_RISE + "heater")
    assert PH.HEATER_RISE_BASIS in s.delta_t_basis
    # measured: the warmest fluid 3.31 K (time-averaged), 3.97 K at any instant (the review run of critic round 3.4)
    assert 3.31 / 1.5 < s.delta_t < 3.97 * 1.5
    assert "boussinesq" not in [PH.warning_key(w) for w in p.warnings]
    # the formula: P / (ρ c_p u_p w) with u_p the line plume's (g β P / ρ c_p)^(1/3), plus one turnover
    props = s.props
    g_beta = PH.G0 * props.beta
    rc = props.density * props.cp
    u_p = (g_beta * 300.0 / rc) ** (1 / 3)
    local = 300.0 / (rc * u_p * 0.3)
    bulk = 300.0 * (4.0 / max(u_p, s.u_forced)) / (rc * 12.0)
    assert PH.heater_rise(300.0, "rect", (0.3, 0.2), props, g_beta, 0.0, 2, box=(0, 4, 0, 3),
                          u_box=s.u_forced) == pytest.approx(local + bulk, rel=1e-12)
    # a tunnel's stream carries it away too; without any flow the conduction cap is all there is
    assert PH.heater_rise(300.0, "rect", (0.3, 0.2), props, g_beta, 5.0, 2) < local
    cap = PH.flux_rise(300.0 / 1.0, 0.3, props, 0.0, 0.0)
    assert PH.heater_rise(300.0, "rect", (0.3, 0.2), props, 0.0, 0.0, 2) == pytest.approx(cap)


def test_a_heater_only_room_gets_a_box_grid_and_a_short_draft_run():
    p = plan_simulation(*heater_only_room())
    # no heated wall: no boundary-layer rule (it spent the whole budget on 576 × 432 before)
    n = planmod.CLOSED_CELLS["cpu"][0]                      # these tests plan on the CPU: 48 cells (GPU 64)
    assert p.grid == (4 * n // 3, n)
    assert not any("thermal boundary layer of the heated walls" in w for w in p.warnings)
    n_ff = p.t_end / (p.physics.length_ra / p.length_ref)
    assert n_ff == pytest.approx(planmod.NATURAL_DRAFT_FF)
    assert any(n.startswith("Draft runs this room for 40 free-fall times") for n in p.notes)
    assert not any("capped at 1500" in w for w in p.warnings)
    # sealed and adiabatic: it warms without end, says so, and its temperatures are not monitors
    assert any(n.startswith("Nothing in this box takes heat out") for n in p.notes)
    assert not set(PH.TEMPERATURE_MONITORS) & set(p.monitors)
    # Standard keeps the long rule
    std = plan_simulation(*heater_only_room())
    std.sim.cfd.quality = "standard"
    assert plan_simulation(std.design, std.sim).t_end > p.t_end


def test_the_duration_label_follows_the_duration_used():
    p = plan_simulation(*heater_only_room(duration=6.0))
    note = next(n for n in p.notes if n.startswith("Duration: "))
    n_ff = 6.0 / (p.physics.length_ra / p.length_ref)
    assert note.startswith(f"Duration: 6 L/U ({PH.units.sig(n_ff, 3)} free-fall times H/U, set in Setup, ")
    assert "1500 free-fall" not in note
    room = plan_simulation(*critic_room(duration=100.0))
    note = next(n for n in room.notes if n.startswith("Duration: "))
    assert "box turnovers" in note and "set in Setup" in note and note.startswith("Duration: 100 L/U (12.5 ")


# ---------------------------------------------------------------------------
# closed boxes are judged on their own monitors
# ---------------------------------------------------------------------------

def test_the_enclosure_is_never_judged_on_force_coefficients():
    p = plan_simulation(*enclosure())
    assert "cd" not in p.monitors and "cl" not in p.monitors
    assert set(PH.TEMPERATURE_MONITORS) <= set(p.monitors) and "ke" in p.monitors
    assert any(m.startswith("fan0_") for m in p.monitors)
    rule = p.summary()["Stops early when"]
    assert "Cd" not in rule and "Cl" not in rule
    assert "fluid temperature rises" in rule and "fans' flow" in rule and "kinetic energy" in rule
    # the verdict: forces are not judged even though the box holds a body
    t = np.linspace(0.0, 10.0, 400)
    rng = np.random.default_rng(3)
    h = {"t": t, "cd": 1.0 + rng.normal(0, 0.5, t.size), "cl": rng.normal(0, 0.5, t.size), "cm": 0 * t}
    for m in p.monitors:
        h[m] = np.full(t.size, 2.0)
    v = R2._combined_verdict(h, p, 2.0, rtol=0.05, laminar=True)
    assert v["cd"] is None and v["cl"] is None
    assert v["stop"]                                        # the monitors are steady: done
    # a tunnel with a body still judges its forces
    tunnel = plan_simulation(*hot_sphere_2d())
    vt = R2._combined_verdict(h, tunnel, 2.0, rtol=0.05, laminar=True)
    assert vt["cd"] is not None


def hot_sphere_2d():
    from test_simulation_physics import cylinder, tunnel_sim
    des = cylinder(0.01, t_body=25.0)
    return des, tunnel_sim(des, 0.2, thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False))


def test_the_not_converged_message_of_a_closed_box_names_its_monitors():
    p = plan_simulation(*enclosure())
    worst = PH.TEMPERATURE_MONITORS[0]
    v = {"monitors": {"stop": False, "judged": {worst: {"state": "transient", "reason": "Still settling."}},
                      "worst": worst}}
    notes, warns = [], []
    R2._monitor_messages(v, p, notes, warns, converged=False, cancelled=False, t_now=30.0, t_end=37.0, rtol=0.05)
    w = warns[0]
    assert "the warmest fluid's rise above the ambient" in w and "kinetic energy" in w and "fan 1's" in w
    assert "force coefficients mean nothing inside it" in w


def test_the_temperature_monitors_need_heat_in_and_a_way_out():
    des = room2d()
    th = ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True)
    vent = [Port(name="J", kind="jet", position=(0.05, 2.7), size=0.2, speed=0.5, frame="wind"),
            Port(name="V", kind="vent", position=(3.95, 2.7), size=0.2, speed=0.5, frame="wind")]
    vented = plan_simulation(des, closed_sim(des, (0, 4, 0, 3), thermal=th, heat_sources=[heater()], ports=vent))
    assert set(PH.TEMPERATURE_MONITORS) <= set(vented.monitors)
    cooled = plan_simulation(des, closed_sim(des, (0, 4, 0, 3), heat_sources=[heater()],
                                             thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True,
                                                                     walls=[WallThermal("left", "temperature", 10.0)])))
    assert set(PH.TEMPERATURE_MONITORS) <= set(cooled.monitors)
    # a cavity held at set temperatures is judged on its Nusselt number, not on temperatures
    from test_simulation_physics import cavity_sim
    cav = plan_simulation(*cavity_sim(1e4)[:2])
    assert "nu" in cav.monitors and not set(PH.TEMPERATURE_MONITORS) & set(cav.monitors)


def test_chunk_samples_measure_the_temperature_rises():
    p = plan_simulation(*critic_room())
    theta = torch.zeros(p.solver_config.ny, p.solver_config.nx, dtype=torch.float64)
    theta[0, 0] = 2.0
    theta[1, 1] = 50.0                                      # inside a body: not fluid
    mask = torch.zeros_like(theta)
    mask[1, 1] = 1.0
    fields = {"theta": theta, "mask": mask, "speed": torch.zeros_like(theta)}
    solver = SimpleNamespace(field=lambda k: fields[k])
    out = PR.chunk_samples(solver, p)
    dt_ref = p.physics.delta_t
    assert out["t_rise_max"] == pytest.approx(2.0 * dt_ref)
    assert out["t_rise_mean"] == pytest.approx(2.0 * dt_ref / (theta.numel() - 1))


# ---------------------------------------------------------------------------
# buoyant thermal layers and unvalidated regimes
# ---------------------------------------------------------------------------

def test_a_buoyancy_dominated_run_sizes_its_thermal_layer_on_the_plume():
    des, sim = hot_sphere()
    p = P3.plan_simulation3d(des, sim)
    s = p.physics
    delta, how = PH.buoyant_layer(s)
    assert "Ra^(-1/4)" in how
    assert delta == pytest.approx(s.length_ra * s.ra ** -0.25)
    assert 1.5e-3 < delta < 3e-3                            # about 2 mm (Ra 4.2×10⁶)
    w = next(x for x in p.warnings if PH.warning_key(x) == "thermal-resolution")
    cells = delta / p.solver_config.h / p.length_ref
    assert w.startswith(f"The grid gives {cells:.1f} cells across the thermal boundary layers (about")
    assert cells < 1
    assert not any(n.startswith("Thermal resolution: about 12") for n in p.notes)
    # mixed convection: the forced scaling on the buoyant velocity
    mixed = SimpleNamespace(thermal=True, buoyancy=True, ri_flow=3.0, gr=1e6, ra=7.1e5, length_ra=0.1, length=0.1,
                            g_beta=0.0335, delta_t=10.0, nu=1.5e-5, pr=0.71)
    d_mix, how_mix = PH.buoyant_layer(mixed)
    u_b = PH.buoyant_speed(mixed)
    assert "mixed" in how_mix and d_mix == pytest.approx(0.1 / math.sqrt(u_b * 0.1 / 1.5e-5 * 0.71))
    assert PH.buoyant_layer(SimpleNamespace(**{**vars(mixed), "ri_flow": 0.5})) is None


def test_natural_and_mixed_convection_around_a_body_say_they_are_not_validated():
    des, sim = hot_sphere()
    p = P3.plan_simulation3d(des, sim)
    w = [x for x in p.warnings if PH.warning_key(x) == "regime-not-validated"]
    assert len(w) == 1 and "regime not validated (docs/validation.md §10)" in w[0]
    # forced convection (the validated sphere, buoyancy off) and a wall-heated cavity carry none
    _, forced = hot_sphere()
    forced.thermal.buoyancy = False
    assert "regime-not-validated" not in [PH.warning_key(x) for x in P3.plan_simulation3d(des, forced).warnings]
    from test_simulation_physics import cavity_sim
    cav = plan_simulation(*cavity_sim(1e5)[:2])
    assert "regime-not-validated" not in [PH.warning_key(x) for x in cav.warnings]


def test_a_thermal_run_still_storing_its_heat_says_its_error_bars_are_rough():
    assert PR.storage_share({"heat_in_bodies": 17.2, "heat_storage_rate": 11.9}) == pytest.approx(11.9 / 17.2)
    assert PR.storage_share({"heat_in_bodies": 17.2}) is None
    assert PR.storage_share({"heat_in_bodies": 0.0, "heat_storage_rate": 1.0}) is None
    assert PR.storage_share({"heat_in_bodies": 10.0, "heat_in_sources": 5.0, "heat_storage_rate": 0.0}) == 0.0


def test_a_closed_3d_room_plans_with_its_signature_and_names_its_duration():
    des, sim = closed_box(3, fans=[Fan(name="Ceiling fan", position=(0.15, 0.0, 0.1), direction=(0, 0, -1),
                                       diameter=0.05, value=1.0, frame="wind")])
    p = P3.plan_simulation3d(des, sim)
    assert planmod.base_kind(p.calibration_kind) == "closed-fan" and "@" in p.calibration_kind
    note = next(n for n in p.notes if n.startswith("Duration: "))
    assert "box turnovers max(W, D, H)/U" in note


def test_the_air_is_isa_in_the_critic_room():
    """(guards the fixtures: the critic's room is ISA sea-level air at a 15 °C ambient)"""
    _, sim = critic_room()
    assert sim.conditions.fluid.name == F.air_isa(0.0).name
    assert sim.thermal.t_ambient == 15.0


# ---------------------------------------------------------------------------
# a heated box at rest, and the Setup page's warnings after a run
# ---------------------------------------------------------------------------

def test_a_sealed_heated_box_starts_at_the_planned_step():
    """At rest nothing bounds the 2D solver's first step in a sealed heated box (no jet, vent or
    speed-controlled fan): the Draft heater-only room took a first step of 56 L/U and diverged. The runner
    takes its first steps at the planned step, then the solver's own bounds hold."""
    p = plan_simulation(*heater_only_room(duration=30.0))
    assert R2.starts_unbounded(p.solver_config)
    assert not R2.starts_unbounded(plan_simulation(*critic_room()).solver_config)          # a jet: a floor
    from test_simulation_physics import cavity_sim
    assert not R2.starts_unbounded(plan_simulation(*cavity_sim(1e4)[:2]).solver_config)    # no source
    res = R2.run_cfd(p)
    assert res.t_end == pytest.approx(30.0, rel=0.2)
    d = res.diagnostics
    assert 15.0 < d["t_fluid_max"] < 15.0 + 10 * p.physics.delta_t                       # warmed, not blown up


def test_the_setup_warnings_after_a_run_use_its_measured_rise():
    des, sim = critic_room()
    sim.heat_sources = []
    sim.thermal.walls = [WallThermal("bottom", "heat_flux", 100.0)]       # q L / (k Nu): a Boussinesq warning
    p = plan_simulation(des, sim)
    assert "boussinesq" in [PH.warning_key(w) for w in p.warnings]
    res = SimpleNamespace(diagnostics={"t_fluid_max": 18.3, "t_fluid_min": 15.0, "t_fluid_dev_max": 3.97})
    after = PR.warnings_after_run(p, res)
    assert "boussinesq" not in [PH.warning_key(w) for w in after]
    assert p.warnings and after is not p.warnings
    assert PR.warnings_after_run(p, SimpleNamespace(diagnostics={})) == list(p.warnings)


def test_the_2d_solver_bounds_a_heater_at_rest_on_its_own():
    """The solver's own source-aware bound (ns2d.physics.physics_dt_limit), without the runner's warm start:
    the sealed heater room's first step from rest stays small and its temperatures finite. Before the bound
    the first step was only viscous-limited (56 L/U) and the heater cells reached theta 1e5."""
    from aerolume.solvers.ns2d import FlowSolver
    p = plan_simulation(*heater_only_room(duration=30.0))
    cfg = p.solver_config
    s = FlowSolver(cfg)
    ph, ops = s.ops.phys, s.ops
    s_max, ri = float(ph.heat_max), abs(float(ph.ri))
    bound = min(1.0 / s_max, (6.0 * ops.cfl * ops.h / (ri * s_max)) ** (1.0 / 3.0))
    s.advance(1)
    first = float(s.t)
    s.advance(40)
    theta = s.state_dict()["theta"]
    # the first step is the source bound (1/S_max = 1.33 L/U on the Draft 64 x 48 grid, measured), not the
    # viscous limit (56 L/U before the bound): one step raises a heater cell by at most one ΔT scale
    assert 0 < first <= bound * (1 + 1e-5) and first < 2.0
    assert torch.isfinite(torch.as_tensor(theta)).all()
    assert float(torch.as_tensor(theta).abs().max()) < 10.0   # warmed by a few ΔT scales at most, not blown up
