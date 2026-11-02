"""Critic round 3.3 (sim): what a 2D fan room, a heater's estimated ΔT, a spanwise smoke rake and a
buoyancy-dominated Draft run state, and what they no longer state.

* A 2D closed domain with a fan carries the "qualitative and grid-dependent" warning from the plan
  onto the result, first, so the Results banner and the report show it (theory_ns2d.md §18.4).
* A run re-states ΔT, Gr, Ra and the ΔT-based warnings on the temperature rise it measured when
  the plan's ΔT was only q L / (k Nu); the planned values stay, labelled "Planned estimate".
* The plan says which length each dimensionless group is on.
* A smoke rake spanning y = 0 symmetrically is its own mirror image (the symmetry plane stays).
* A Draft tunnel run whose plume outruns its free stream is shortened, with a note, before Run.
* ``aerolume-cli validate --full`` checks the cavity against Nu0 2.238, not the cavity average 2.243.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from test_simulation_peak_speed import hot_sphere
from test_simulation_physics import closed_sim, room2d

from aerolume.core import fluid as F
from aerolume.model import Emitter, Fan, HeatSource, ThermalSettings, WallThermal
from aerolume.simulation import physics as PH
from aerolume.simulation import physics_run as PR
from aerolume.simulation import plan as planmod
from aerolume.simulation import plan3d as P3
from aerolume.simulation.plan import plan_simulation
from aerolume.simulation.runner import run_cfd

ROOM_KEY = "room-2d-qualitative"


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def fan(**kw):
    return Fan(name="Fan", position=(4 / 3, 1.5), direction=(1, 0), diameter=0.5, value=1.0, frame="wind", **kw)


def heater(power: float = 300.0):
    return HeatSource(name="Heater", kind="rect", position=(3.5, 0.3), size=(0.3, 0.2), power=power, frame="wind")


def heated_room(**cfd_kw):
    """The critic's Draft room: a 1 N/m fan and a 300 W/m heater in a 4 × 3 m box of 20 °C air."""
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fluid=F.air(20.0),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True),
                     fans=[fan()], heat_sources=[heater()], cfd_kw=cfd_kw)
    return des, sim


# ---------------------------------------------------------------------------
# 2D rooms with fans are qualitative
# ---------------------------------------------------------------------------

def test_a_2d_closed_room_with_a_fan_warns_that_its_speeds_are_qualitative():
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fans=[fan()])
    p = plan_simulation(des, sim)
    keys = [PH.warning_key(w) for w in p.warnings]
    assert keys[0] == ROOM_KEY                                     # first: the banner shows the first warning
    w = p.warnings[0]
    assert w.startswith(PH.ROOM_2D_FAN_WARNING)
    assert "qualitative and grid-dependent" in w and "2.4 m/s" in w and "3.4 m/s" in w and "§18.4" in w
    assert f"{p.t_end:.4g} L/U end inside that start-up transient" in w          # Draft: 240 < 500 L/U
    # no fan, no warning; a 3D room is not affected (its jet breaks down into 3D turbulence)
    no_fan = closed_sim(des, (0, 4, 0, 3), thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True),
                        heat_sources=[heater()])
    assert ROOM_KEY not in [PH.warning_key(x) for x in plan_simulation(des, no_fan).warnings]
    # a run long enough to settle drops the duration sentence but keeps the warning
    assert PH.room_2d_fan_warning(900.0) == PH.ROOM_2D_FAN_WARNING
    assert PH.warning_key(PH.ROOM_2D_FAN_WARNING) == ROOM_KEY
    assert "qualitative" in PH.ROOM_2D_FAN_NOTE and "dimension=3" in PH.ROOM_2D_FAN_NOTE


def test_a_2d_closed_fan_run_carries_the_qualitative_warning_onto_the_result():
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fans=[fan()], cfd_kw={"cells_per_ref": 6, "duration": 4.0})
    res = run_cfd(plan_simulation(des, sim))
    keys = [PH.warning_key(w) for w in res.warnings]
    assert ROOM_KEY in keys and keys[0] == ROOM_KEY
    assert res.fans_report and res.fans_report[0]["thrust"] == pytest.approx(1.0, rel=1e-6)   # the thrust is exact


def test_the_3d_room_has_no_2d_room_warning():
    from test_simulation_physics import room3d
    des = room3d()
    sim = closed_sim(des, (0, 4, 0, 3, 0, 2.5), fans=[Fan(name="F", position=(1.3, 1.5, 1.25), direction=(1, 0, 0),
                                                          diameter=0.5, value=1.0, frame="wind")])
    p = P3.plan_simulation3d(des, sim)
    assert ROOM_KEY not in [PH.warning_key(w) for w in p.warnings]


# ---------------------------------------------------------------------------
# a heater's ΔT: planned estimate, then measured
# ---------------------------------------------------------------------------

def test_the_plan_names_the_length_of_each_group():
    des, sim = heated_room()
    p = plan_simulation(des, sim)
    s = p.physics
    assert PH.delta_t_is_estimate(s) and s.surfaces == 0
    rows = p.summary()
    assert rows["Length of Gr, Ra and Ri"] == "3.00 m (height of the closed box)"
    note = next(n for n in p.notes if n.startswith("Buoyancy: Boussinesq"))
    assert f"Ri = gβΔT L/U² = {PH.units.sig(s.ri, 3)} on the reference length L = 0.500 m" in note
    assert f"Ri = Gr/Re² = {PH.fmt_number(s.ri_flow)} on the height of the closed box 3.00 m" in note
    assert s.ri_flow == pytest.approx(s.ri * 3.0 / 0.5, rel=1e-9)            # one ΔT and U, two lengths


def test_a_room_heated_only_by_heaters_has_no_thermal_boundary_layer_warning():
    """No wall or body exchanges heat, so "0.4 cells across the thermal boundary layers ... Nusselt
    numbers under-resolved" (critic round 3.3) described layers the room does not have."""
    des, sim = heated_room()
    p = plan_simulation(des, sim)
    assert "thermal-resolution" not in [PH.warning_key(w) for w in p.warnings]
    assert any(n.startswith("Thermal resolution: no body or wall exchanges heat") for n in p.notes)
    # a heated wall brings the check back
    sim.thermal.walls = [WallThermal("left", "temperature", 30.0)]
    q = plan_simulation(des, sim)
    assert q.physics.surfaces == 1
    assert not any(n.startswith("Thermal resolution: no body") for n in q.notes)


def test_measured_delta_t_restates_the_groups_and_replaces_the_planned_warnings():
    # a heat-flux floor keeps the q L / (k Nu) estimate (39 K here: β·ΔT and Ra above their limits);
    # heaters are now planned on their through-flow rise, close to what runs measure
    des, sim = heated_room()
    sim.heat_sources = []
    sim.thermal.walls = [WallThermal("bottom", "heat_flux", 100.0)]
    p = plan_simulation(des, sim)
    s = p.physics
    planned_ra = next(w for w in p.warnings if PH.warning_key(w) == "rayleigh")
    planned_bq = next(w for w in p.warnings if PH.warning_key(w) == "boussinesq")
    assert s.ra > 1e8 and s.beta_dt > PH.BOUSSINESQ_LIMIT
    # what the critic's run measured: warmest fluid 22.96 °C at a 20 °C ambient (3.48 K above it)
    res = SimpleNamespace(warnings=list(p.warnings), notes=list(p.notes),
                          diagnostics={"t_fluid_max": 22.96, "t_fluid_min": 20.0, "t_fluid_mean": 21.42,
                                       "t_fluid_dev_max": 3.48})
    ms = PR.apply_measured_thermal(res, p)
    assert ms is not None and ms.delta_t == pytest.approx(2.96, rel=1e-9)
    assert ms.ra == pytest.approx(s.ra * 2.96 / s.delta_t, rel=1e-9)        # Ra ∝ ΔT
    assert ms.speed == s.speed and ms.ri == s.ri                             # the solver's own scales stay
    assert planned_ra not in res.warnings and planned_bq not in res.warnings
    fresh_ra = [w for w in res.warnings if PH.warning_key(w) == "rayleigh"]
    assert (ms.ra > 1e8) == bool(fresh_ra)                                   # re-derived, not dropped blindly
    if fresh_ra:
        assert PH.units.fmt_reynolds(ms.ra) in fresh_ra[0]
    assert PH.warning_key(res.warnings[0]) == ROOM_KEY                       # still first
    assert any(n.startswith(PR.PLANNED_PREFIX + "Temperature scale ΔT = ") for n in res.notes)
    assert any(n.startswith(PR.PLANNED_PREFIX + "Buoyancy: Boussinesq") for n in res.notes)
    m = next(n for n in res.notes if n.startswith("Measured temperature difference ΔT = 2.96 K"))
    assert "22.96 °C" in m and f"planned estimate was {PH.units.sig(s.delta_t, 3)} K" in m
    assert f"Ra {PH.fmt_number(ms.ra)}" in m and f"planned estimate Gr {PH.fmt_number(s.gr)}" in m
    # the Conditions panel's helper labels which ΔT it states
    sc, how = PH.result_scales(s, res)
    assert how == "measured" and sc.delta_t == pytest.approx(2.96)
    assert PH.result_scales(s, SimpleNamespace(diagnostics={}))[1] == "planned estimate"


def test_a_set_temperature_difference_is_not_restated():
    """The cavity's ΔT is T_hot − T_cold, held exactly: nothing to re-derive."""
    from test_simulation_physics import cavity_sim
    des, sim, d_t = cavity_sim(1e4)
    p = plan_simulation(des, sim)
    assert not PH.delta_t_is_estimate(p.physics)
    res = SimpleNamespace(warnings=list(p.warnings), notes=list(p.notes),
                          diagnostics={"t_fluid_max": 15.0 + d_t / 2, "t_fluid_min": 15.0 - d_t / 2,
                                       "t_fluid_dev_max": d_t / 2})
    assert PR.apply_measured_thermal(res, p) is None
    assert not any(n.startswith(PR.PLANNED_PREFIX) for n in res.notes)
    assert PH.result_scales(p.physics, res) == (p.physics, "")


def test_a_heater_room_run_states_its_measured_delta_t():
    """Through the runner: the measured ΔT note, the planned notes relabelled, no planned Ra line."""
    des, sim = heated_room(cells_per_ref=6, duration=4.0)
    p = plan_simulation(des, sim)
    res = run_cfd(p)
    d = res.diagnostics
    rise = max(d["t_fluid_max"] - 20.0, 20.0 - d["t_fluid_min"])
    # the heater's through-flow estimate (physics.heater_rise) is the right size: this coarse 4 L/U run
    # reached 6.4 K against 4.2 K planned, where q L / (k Nu) had planned 44 K (critic round 3.4)
    assert 1 / 3 < rise / p.physics.delta_t < 3
    assert any(n.startswith(f"Measured temperature difference ΔT = {PH.units.sig(rise, 3)} K") for n in res.notes)
    assert any(n.startswith(PR.PLANNED_PREFIX + "Temperature scale") for n in res.notes)
    ra_lines = [w for w in res.warnings if PH.warning_key(w) == "rayleigh"]
    planned = next(w for w in p.warnings if PH.warning_key(w) == "rayleigh")
    assert planned not in ra_lines
    assert PH.warning_key(res.warnings[0]) == ROOM_KEY


# ---------------------------------------------------------------------------
# 3D: a spanwise rake symmetric about y = 0 keeps the symmetry plane
# ---------------------------------------------------------------------------

def test_a_rake_spanning_the_centre_plane_symmetrically_keeps_the_symmetry_plane():
    """smoke_3d.py's amber rake runs from y = −0.08 to +0.08 (7 nozzles): it is its own mirror image,
    so it must not switch the half domain off (critic round 3.3: CS 1.7e-7, twice the cells)."""
    des, plain = hot_sphere()
    span = Emitter(name="Span", kind="rake", position=(-0.15, -0.08, 0.0), end=(-0.15, 0.08, 0.0), count=7)
    slot = Emitter(name="Slot", kind="line", position=(-0.15, 0.05, 0.01), end=(-0.15, -0.05, 0.01))
    for em in ([span], [slot], [span, slot]):
        _, sim = hot_sphere(emitters=em)
        p = P3.plan_simulation3d(des, sim)
        assert p.symmetry_plane, em
        assert p.grid3 == P3.plan_simulation3d(des, plain).grid3
        assert any("span it symmetrically" in n for n in p.notes)
    # not mirror images: lopsided, shifted in x between the ends, or tracer particles (which the
    # solver reflects into the solved half, doubling the mirrored streak)
    lopsided = Emitter(name="L", kind="rake", position=(-0.15, -0.08, 0.0), end=(-0.15, 0.06, 0.0), count=7)
    skewed = Emitter(name="S", kind="rake", position=(-0.15, -0.08, 0.0), end=(-0.10, 0.08, 0.0), count=7)
    parts = Emitter(name="P", kind="rake", species="particles", position=(-0.15, -0.08, 0.0), end=(-0.15, 0.08, 0.0),
                    count=7, rate=100.0)
    tilted = Emitter(name="T", kind="rake", position=(-0.15, -0.08, 0.0), end=(-0.15, 0.08, 0.0), count=7,
                     normal=(1.0, 0.5, 0.0))
    for em in (lopsided, skewed, parts, tilted):
        _, sim = hot_sphere(emitters=[em])
        p = P3.plan_simulation3d(des, sim)
        assert not p.symmetry_plane, em.name
        assert any("off the centre plane" in n for n in p.notes)


def test_mirror_pair_is_a_reflection_across_y():
    assert P3._mirror_pair((1.0, -0.2, 0.3), (1.0, 0.2, 0.3), 1e-9)
    assert not P3._mirror_pair((1.0, -0.2, 0.3), (1.0, 0.2, 0.31), 1e-9)
    assert not P3._mirror_pair((1.0, -0.2, 0.3), (1.0, -0.2, 0.3), 1e-9)
    assert P3._normal_symmetric(None, 1e-9) and P3._normal_symmetric((1, 0, 0), 1e-9)
    assert P3._normal_symmetric((0, 1, 0), 1e-9) and not P3._normal_symmetric((1, 1, 0), 1e-9)


# ---------------------------------------------------------------------------
# Draft runs of buoyancy-dominated tunnel cases
# ---------------------------------------------------------------------------

def test_a_buoyancy_dominated_draft_tunnel_run_is_shortened_before_run():
    """The Draft hot sphere (Ri 627) was planned at 40 L/U and about 40 minutes (189,682 steps on the
    critic's GPU) for a quality meant for pictures."""
    des, sim = hot_sphere()
    p = P3.plan_simulation3d(des, sim)
    assert PH.buoyancy_dominated_warning(p.physics, closed=False)
    ratio = p.speed_ratio
    assert PH.DRAFT_BUOYANT_T_MIN <= p.t_end <= PH.DRAFT_BUOYANT_T_MAX
    assert p.t_end == pytest.approx(max(PH.DRAFT_BUOYANT_T_MIN, float(f"{40.0 / ratio:.2g}")), rel=1e-9)
    assert p.t_avg_start == pytest.approx(0.5 * p.t_end)
    note = next(n for n in p.notes if n.startswith("Draft runs this buoyancy-dominated case"))
    assert f"for {p.t_end:g} L/U instead of 40" in note and "Standard" in note and "closed box" in note
    assert p.steps_estimate == math.ceil(p.t_end / p.dt)                    # the estimate shown is the capped run's
    # a Duration typed in Setup wins; Standard keeps the full length
    sim.cfd.duration = 12.0
    assert P3.plan_simulation3d(des, sim).t_end == 12.0
    sim.cfd.duration = None
    sim.cfd.quality = "standard"
    q = P3.plan_simulation3d(des, sim)
    assert q.t_end > PH.DRAFT_BUOYANT_T_MAX and not any(n.startswith("Draft runs this") for n in q.notes)
    # without buoyancy nothing changes
    sim.cfd.quality = "draft"
    sim.thermal.buoyancy = False
    assert P3.plan_simulation3d(des, sim).t_end == 40.0


def test_draft_buoyant_duration_rule():
    s = SimpleNamespace(thermal=True, buoyancy=True, ri_flow=600.0, speed=0.015, g_beta=1.0, delta_t=40.0,
                        length_ra=0.1, gr=1e6, props=SimpleNamespace(density=1.2))
    s_ok = SimpleNamespace(**{**vars(s), "ri_flow": 2.0})
    assert PH.draft_buoyant_duration(s_ok, 40.0, 0.03) == (40.0, "")        # not buoyancy-dominated
    t, note = PH.draft_buoyant_duration(s, 40.0, 0.36)
    assert t == pytest.approx(2.0) and note                                  # 40 / 24 = 1.7 -> the 2 L/U floor
    t, _ = PH.draft_buoyant_duration(s, 400.0, 0.06)
    assert t == PH.DRAFT_BUOYANT_T_MAX                                       # 400 / 4 = 100 -> the 10 L/U cap
    assert PH.draft_buoyant_duration(None, 40.0, 1.0) == (40.0, "")


# ---------------------------------------------------------------------------
# the command line's cavity reference
# ---------------------------------------------------------------------------

def test_cli_validate_full_uses_the_hot_wall_reference():
    import inspect

    from aerolume import api, cli
    src = inspect.getsource(cli.cmd_validate)
    assert "CAVITY_NU_2D[1e4]" in src and "(de Vahl Davis 2.243)" not in src and "2.20, 2.29" not in src
    assert api.CAVITY_NU_2D[1e4] == 2.238
