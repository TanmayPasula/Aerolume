"""Runs with placed physics (DESIGN.md 10): SI outputs, monitors, convergence, validation.

The fast tests run tiny CPU grids (a few seconds each) and check what a result holds and that
its numbers agree with each other and with their definitions: the Nusselt number against the
heat rate (Q = Nu k dT A / H), temperatures in °C between the set ones, the delivered fan thrust
against the set thrust, the SI conversions of :mod:`aerolume.simulation.physics_run`. The
``slow`` GPU tests are the end-to-end validation of the planner's non-dimensionalisation: the
benchmark numbers come back through the whole chain (model in SI -> plan -> solver -> SI).
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest
import torch
from test_simulation_physics import (
    AIR,
    air_props,
    cavity_sim,
    closed_sim,
    cylinder,
    room2d,
    room3d,
    tunnel_sim,
)

from aerolume.geometry3d import design_from_meshes, sphere
from aerolume.model import CFD3DResult, CFDSettings, Emitter, Fan, HeatSource, Stat, ThermalSettings
from aerolume.simulation import physics_run as PR
from aerolume.simulation import plan as planmod
from aerolume.simulation.plan import plan_simulation
from aerolume.simulation.runner import final_message, run_cfd

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def run(design, sim, **kw):
    msgs: list[str] = []
    res = run_cfd(plan_simulation(design, sim), progress=lambda f, m: msgs.append(m), **kw)
    return res, msgs


# ---------------------------------------------------------------------------
# fast runs (CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cavity_run():
    des, sim, d_t = cavity_sim(1e3, cells_per_ref=16, duration=6.0)
    sim.cfd.device = "cpu"
    res, msgs = run(des, sim)
    return res, msgs, d_t


def test_a_cavity_reports_the_hot_walls_nusselt_number_and_its_heat(cavity_run):
    res, msgs, d_t = cavity_run
    nu = res.nusselt
    assert math.isfinite(nu.mean) and 0.9 < nu.mean < 1.4                  # de Vahl Davis: 1.118 at Ra 10^3
    k, *_ = air_props(15.0)
    # Nu = (Q / A) H / (k dT) with A = H (the wall, per metre of span): Q = Nu k dT
    assert res.heat_rate == pytest.approx(nu.mean * k * d_t, rel=2e-2)
    assert {"nu", "ke", "heat_rate"} <= set(res.history)
    assert math.isnan(res.cd.mean) and res.per_body == {}                    # no bodies, no forces: nan, not 0
    t = res.mean_fields.fields["temperature"]
    assert t.min() >= 15.0 - d_t / 2 - 1e-4 * d_t and t.max() <= 15.0 + d_t / 2 + 1e-4 * d_t
    assert "temperature" in res.instant_fields.fields and "temperature_rms" in res.mean_fields.fields
    d = res.diagnostics
    assert {"div_rms", "div_max", "p_residual", "heat_balance_residual", "cfl", "dt"} <= set(d)
    assert d["div_max"] < 1e-3
    assert msgs[-1].startswith("Done after") and "Nu = " in msgs[-1]
    assert any("Nusselt number of the left wall" in n for n in res.notes)


def test_a_heated_cylinder_carries_temperature_smoke_channels_and_per_body_heat():
    des = cylinder(0.01, t_body=25.0)
    sim = tunnel_sim(des, 40 * AIR.kinematic_viscosity / 0.01,
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False),
                     cfd=CFDSettings(quality="draft", device="cpu", cells_per_ref=8, duration=3.0, average_from=1.5),
                     emitters=[Emitter(name="a", species="dye", position=(-0.01, 0.002), color="#e04040"),
                               Emitter(name="b", species="dye", position=(-0.01, -0.002), color="#4060e0"),
                               Emitter(name="p", species="particles", position=(-0.01, 0.0), rate=200.0)])
    res, msgs = run(des, sim)
    body = res.per_body["Cylinder"]
    assert body["t_wall"] == pytest.approx(25.0) and body["heat_rate"] > 0
    assert math.isfinite(body["cd"]) and body["nu"] == pytest.approx(res.nusselt.mean, rel=2e-2)
    k, *_ = air_props(15.0)
    # one body: Q = Nu k dT pi (the perimeter pi d over the reference length d)
    assert res.heat_rate == pytest.approx(res.nusselt.mean * k * 10.0 * math.pi, rel=3e-2)
    t = res.mean_fields.fields["temperature"]
    assert 15.0 - 1e-3 <= float(t.min()) and float(t.max()) <= 25.0 + 1e-3
    assert {"dye", "dye1", "temperature"} <= set(res.instant_fields.fields)
    assert msgs[-1].startswith("Done after") and "Cd = " in msgs[-1]
    assert not any("e-5" in n and "domain sides" in n for n in res.notes)   # no round-off "heat" listed


def test_a_thrust_fan_in_a_room_delivers_its_thrust():
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fans=[Fan(name="F", position=(1.3, 1.5), direction=(1, 0), diameter=0.5,
                                                   value=1.0, frame="wind")],
                     cfd_kw={"cells_per_ref": 6, "duration": 10.0})
    res, msgs = run(des, sim)
    (f,) = res.fans_report
    assert f["name"] == "F" and f["mode"] == "thrust" and f["target"] == 1.0
    assert f["thrust"] == pytest.approx(1.0, rel=1e-6)                       # N per metre of span, as set
    assert f["pressure_jump"] == pytest.approx(1.0 / 0.5, rel=1e-6)          # dp = T / D
    assert f["volume_flow"] == pytest.approx(f["disc_velocity"] * 0.5, rel=1e-6)
    assert f["ideal_power"] == pytest.approx(f["thrust"] * f["disc_velocity"], rel=5e-2)
    assert f["induced_velocity"] == pytest.approx(f["disc_velocity"])      # no free stream in a room
    assert {"fan0_thrust", "fan0_velocity", "fan0_flow", "ke"} <= set(res.history)
    assert "thrust 1.00 N/m" in msgs[-1]


def test_a_room_warmed_by_a_heater_reports_its_heat_balance_and_temperatures():
    """Round 3 critique: a room with a heater (no heated body or wall) finished with nusselt and
    heat_rate NaN, so the Results page had no heat-transfer output at all. The heat rate is now
    the heat the heater puts into the fluid, measured from the solver's balance, with the fluid's
    mean / min / max temperature; there is honestly no Nusselt number."""
    from aerolume.analysis import physics as AP
    from aerolume.core import fluid as F
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fluid=F.air(20.0),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True),
                     heat_sources=[HeatSource(name="Heater", kind="rect", position=(3.5, 0.3), size=(0.3, 0.2),
                                              power=200.0, frame="wind")],
                     fans=[Fan(name="F", position=(1.3, 1.5), direction=(1, 0), diameter=0.5, value=1.0,
                               frame="wind")],
                     cfd_kw={"cells_per_ref": 6, "duration": 6.0})
    res, _ = run(des, sim)
    d = res.diagnostics
    assert math.isnan(res.nusselt.mean)                                       # nothing defines one
    assert d["heat_sources_set"] == 200.0
    assert d["heat_in_sources"] == pytest.approx(200.0, rel=1e-4)              # the heater delivers its power
    assert res.heat_rate == pytest.approx(d["heat_in_sources"], rel=1e-9)
    assert AP.has_thermal(res)                                                # the Results page's test
    hb = AP.heat_balance(res)
    assert abs(hb["relative"]) < 1e-2                                         # in = out + stored
    assert 20.0 - 1e-6 <= d["t_fluid_min"] <= d["t_fluid_mean"] < d["t_fluid_max"]
    assert d["t_fluid_dev_max"] >= d["t_fluid_max"] - 20.0 - 1e-9
    assert any(n.startswith("Heat rate ") and "No Nusselt number" in n for n in res.notes)
    assert any(n.startswith("Fluid temperature (") for n in res.notes)
    assert not any("ambient temperature is" in w for w in res.warnings)       # air at 20 °C in a 20 °C room


def test_a_measured_temperature_rise_replaces_the_planned_boussinesq_warning():
    from aerolume.simulation.physics import BOUSSINESQ_LIMIT, boussinesq_warning
    props = types.SimpleNamespace(beta=1 / 293.15)
    s = types.SimpleNamespace(thermal=True, buoyancy=True, props=props, delta_t=72.3,
                              beta_dt=72.3 / 293.15)
    plan = types.SimpleNamespace(physics=s)
    planned = boussinesq_warning(s)
    res = types.SimpleNamespace(warnings=[planned, "other"], diagnostics={"t_fluid_dev_max": 5.6})
    drop, notes, warns = PR.measured_boussinesq(res, plan)
    assert drop == [planned] and not warns and "0.019 measured" in notes[0] and "72.3 K" in notes[0]
    res = types.SimpleNamespace(warnings=["other"], diagnostics={"t_fluid_dev_max": 0.2 / props.beta})
    drop, notes, warns = PR.measured_boussinesq(res, plan)               # not warned, but it got hot
    assert not drop and not notes and warns[0].startswith("β·ΔT = 0.20 measured")
    assert 0.2 > BOUSSINESQ_LIMIT
    s.buoyancy = False                                                    # no buoyancy: nothing to judge
    assert PR.measured_boussinesq(res, plan) == ([], [], [])


def test_a_tiny_3d_cavity_carries_its_temperature_into_slices_and_the_volume():
    des, sim, d_t = cavity_sim(1e4, dim=3, cells_per_ref=12, duration=3.0)
    res, msgs = run(des, sim)
    assert isinstance(res, CFD3DResult) and math.isfinite(res.nusselt.mean) and res.nusselt.mean > 0.5
    k, *_ = air_props(15.0)
    # 3D: Q = Nu k dT A / H with A = H^2 (the hot wall)
    assert res.heat_rate == pytest.approx(res.nusselt.mean * k * d_t * 0.1, rel=3e-2)
    for sl in res.slices:
        t = sl.fields["temperature"]
        assert np.nanmin(t) >= 15.0 - d_t / 2 - 1e-3 * d_t and np.nanmax(t) <= 15.0 + d_t / 2 + 1e-3 * d_t
    assert "temperature" in res.volume.fields
    d = res.diagnostics                                     # from the full-resolution 3D field
    assert 15.0 - d_t / 2 - 1e-3 * d_t <= d["t_fluid_min"] <= d["t_fluid_mean"] <= d["t_fluid_max"]
    assert d["t_fluid_max"] <= 15.0 + d_t / 2 + 1e-3 * d_t
    assert {"nu", "ke"} <= set(res.history)
    assert "Nu = " in msgs[-1]


# ---------------------------------------------------------------------------
# physics_run helpers
# ---------------------------------------------------------------------------

def test_merge_series_aligns_per_step_series_and_never_extrapolates_chunk_samples():
    h = {"t": np.array([0.0, 1.0, 2.0, 3.0]), "cd": np.ones(4)}
    phys = {"t": h["t"].copy(), "fan0_thrust": np.array([1.0, 2.0, 3.0, 4.0])}
    chunks = {"t": np.array([1.0, 3.0]), "ke": np.array([10.0, 30.0]), "nu_wall": np.array([2.0, 4.0])}
    out = PR.merge_series(h, phys, chunks)
    assert out["fan0_thrust"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(out["ke"][0]) and out["ke"][1:].tolist() == [10.0, 20.0, 30.0]
    assert out["nu"][2] == pytest.approx(3.0)                        # a 3D hot wall becomes the "nu" series
    phys["nu"] = np.array([5.0, 5.0, 5.0, 5.0])
    assert PR.merge_series(h, phys, chunks)["nu"].tolist() == [5.0] * 4   # per-step values win


def test_a_steady_monitor_must_also_stop_drifting():
    """A cavity settling exponentially sits within its tolerance of the current value long before
    it reaches the asymptote: the verdict waits for the drift between the window halves."""
    t = np.linspace(0.0, 100.0, 4001)
    plan = types.SimpleNamespace(monitors=("nu",))
    settling = {"t": t, "nu": 2.0 + 0.2 * np.exp(-t / 40.0)}
    v = PR.monitor_verdict(settling, plan, 50.0, rtol=0.05, laminar=True)
    assert not v["stop"] and v["worst"] == "nu"
    settled = {"t": t, "nu": 2.0 + 0.2 * np.exp(-t / 4.0)}
    assert PR.monitor_verdict(settled, plan, 50.0, rtol=0.05, laminar=True)["stop"]
    missing = PR.monitor_verdict({"t": t}, types.SimpleNamespace(monitors=("ke",)), 50.0, rtol=0.05, laminar=True)
    assert not missing["stop"]


def test_the_hot_wall_is_the_hottest_fixed_temperature_side():
    walls = [types.SimpleNamespace(side="left", mode="temperature", value=0.5),
             types.SimpleNamespace(side="top", mode="temperature", value=0.8),
             types.SimpleNamespace(side="right", mode="heat_flux", value=3.0)]
    plan = types.SimpleNamespace(physics=types.SimpleNamespace(thermal=True), closed=True,
                                 placed=types.SimpleNamespace(thermal=types.SimpleNamespace(walls=walls),
                                                              body_thermal=[]))
    assert PR.hot_wall(plan) == "top"
    plan.placed.body_thermal = [("temperature", 1.0)]               # a heated body: its Nusselt number instead
    assert PR.hot_wall(plan) is None
    plan.placed.body_thermal, plan.closed = [], False                # a tunnel's walls are not the benchmark
    assert PR.hot_wall(plan) is None


def test_fans_report_converts_solver_units_to_si():
    rho, U, L = 1.2, 2.0, 0.5
    fan = types.SimpleNamespace(name="F", mode="velocity", value=3.0, diameter=0.4)
    plan = types.SimpleNamespace(dimension=3, speed=U, length_ref=L, fluid=types.SimpleNamespace(density=rho),
                                 placed=types.SimpleNamespace(fans=[object()], fans_si=[fan]))
    solver = types.SimpleNamespace(fans_report=lambda mean=False: [
        {"thrust": 0.5, "pressure_jump": 0.25, "flow": 0.3, "disc_velocity": 1.5, "induced_velocity": 1.5,
         "power": 0.75, "in_domain": 1.0}])
    (e,) = PR.fans_report_si(solver, plan, {}, 0.0, False)
    assert e["thrust"] == pytest.approx(0.5 * rho * U * U * L * L)
    assert e["pressure_jump"] == pytest.approx(0.25 * rho * U * U)
    assert e["volume_flow"] == pytest.approx(0.3 * U * L * L) and e["disc_velocity"] == pytest.approx(3.0)
    assert e["ideal_power"] == pytest.approx(0.75 * rho * U ** 3 * L * L) and e["in_domain"] == 1.0
    plan.dimension = 2
    (e2,) = PR.fans_report_si(solver, plan, {}, 0.0, False)
    assert e2["thrust"] == pytest.approx(0.5 * rho * U * U * L) and e2["volume_flow"] == pytest.approx(0.3 * U * L)


def test_the_final_message_names_the_headline_number():
    def result(**kw):
        base = {"cd": Stat(), "nusselt": Stat(), "fans_report": [], "per_body": {}, "wall_seconds": 12.0,
                "grid": (10, 10)}
        base.update(kw)
        return types.SimpleNamespace(**base)
    assert final_message(result(cd=Stat(mean=1.5, stderr=0.01)), False).endswith("Cd = 1.500 ± 0.010")
    assert final_message(result(nusselt=Stat(mean=2.2508, stderr=1e-16)), False).endswith("Nu = 2.251")
    fan = [{"name": "F", "thrust": 4.0}]
    assert final_message(result(fans_report=fan, grid=(4, 4, 4)), True).startswith("Stopped after")
    assert final_message(result(fans_report=fan, grid=(4, 4, 4)), False).endswith("thrust 4.00 N")
    assert final_message(result(), False).endswith("no bodies, so no forces")


# ---------------------------------------------------------------------------
# validation through the whole chain (GPU, slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@needs_cuda
def test_draft_square_cavity_ra_1e4_reproduces_de_vahl_davis():
    des, sim, _ = cavity_sim(1e4)
    sim.cfd.device = "cuda"
    res, _ = run(des, sim)
    assert res.converged
    assert res.nusselt.mean == pytest.approx(2.238, rel=1e-2)        # de Vahl Davis (1983) Nu0; measured +0.58 %


@pytest.mark.slow
@needs_cuda
def test_draft_cubic_cavity_ra_1e4_reproduces_tric_et_al():
    des, sim, _ = cavity_sim(1e4, dim=3)
    sim.cfd.device = "cuda"
    res, _ = run(des, sim)
    assert res.converged
    assert res.nusselt.mean == pytest.approx(2.0542, rel=2e-2)       # Tric et al. (2000); measured +1.2 % at 32^3


@pytest.mark.slow
@needs_cuda
def test_draft_heated_cylinder_re_40_matches_the_lange_correlation():
    re_ = 40.0
    des = cylinder(0.01, t_body=25.0)
    sim = tunnel_sim(des, re_ * AIR.kinematic_viscosity / 0.01,
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False, prandtl=0.7),
                     cfd=CFDSettings(quality="draft", device="cuda"))
    res, _ = run(des, sim)
    x = 0.05 + 0.226 * re_ ** 0.085                                    # Lange, Durst & Breuer (1998), Pr 0.7
    lange = 0.082 * math.sqrt(re_) + 0.734 * re_ ** x
    assert res.nusselt.mean == pytest.approx(lange, rel=4e-2)          # Draft: 23 cells/d, 5 % blockage
    # steady below Re ~ 47: the decaying start-up wiggle on the lift (rms ~0.008 against a ±0.083
    # target) must not hold the run to its end as "not converged" (round 3 critique; measured:
    # converged, stopped at t = 53 of 60, Nu +1.8 % from Lange)
    assert res.converged and not any(w.startswith("Not converged") for w in res.warnings)


@pytest.mark.slow
@needs_cuda
def test_a_speed_controlled_fan_reaches_its_target_in_a_room():
    des = room3d()
    sim = closed_sim(des, (0, 4, 0, 3, 0, 2.5), fans=[Fan(name="F", position=(1.3, 1.5, 1.25), direction=(1, 0, 0),
                                                         diameter=0.8, mode="velocity", value=1.0, frame="wind")])
    sim.cfd.device = "cuda"
    res, _ = run(des, sim)
    (f,) = res.fans_report
    assert f["disc_velocity"] == pytest.approx(1.0, rel=2e-2)
    assert f["volume_flow"] == pytest.approx(f["disc_velocity"] * math.pi * 0.4 ** 2, rel=5e-2)


@pytest.mark.slow
@needs_cuda
def test_a_heated_sphere_with_smoke_and_particles_runs_in_3d():
    des = design_from_meshes([sphere(0.02, 32, name="Sphere")], "Hot sphere", kind="generic")
    for m in des.meshes:
        m.thermal, m.thermal_value = "temperature", 25.0
    sim = tunnel_sim(des, 50 * AIR.kinematic_viscosity / 0.02,
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False, prandtl=0.71),
                     cfd=CFDSettings(quality="draft", device="cuda"),
                     emitters=[Emitter(name="smoke", species="dye", position=(-0.02, 0.0, 0.0)),
                               Emitter(name="dots", species="particles", kind="rake", position=(-0.02, -0.01, 0.0),
                                       end=(-0.02, 0.01, 0.0), count=5, rate=200.0)])
    res, _ = run(des, sim)
    re_, pr = 50.0, 0.71
    whitaker = 2 + (0.4 * re_ ** 0.5 + 0.06 * re_ ** (2 / 3)) * pr ** 0.4           # 5.18 (mu / mu_s = 1)
    ranz = 2 + 0.6 * re_ ** 0.5 * pr ** (1 / 3)                                       # 5.78
    assert whitaker * 0.97 < res.nusselt.mean < ranz * 1.03
    assert any("dye" in v.fields for v in res.volumes())
