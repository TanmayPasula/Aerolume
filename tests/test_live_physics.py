"""Explore mode with placed physics (DESIGN.md 10.4): live edits, readouts, inspection, 2D and 3D.

CPU sessions on the explore budget keep these fast. They check what the Simulate page relies
on: every post key is accepted and validated in the caller's thread, edits reach the solver in
solver units on the next tick, frames carry the temperature (°C), the smoke channels, the
emitter particles, the new diagnostics rows, a captured step and SI readouts, and what an edit
cannot do comes back as a notice instead of an exception.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from test_simulation_physics import AIR, cavity_sim, closed_sim, cylinder, room2d, tunnel_sim

from aerolume.geometry3d import design_from_meshes, sphere
from aerolume.model import CFDSettings, Emitter, Fan, ThermalSettings
from aerolume.simulation.live import MAX_NOTICES, LiveFrame, LiveSession
from aerolume.simulation.live3d import LiveSession3D, apply_obstacles_3d


def heated_cylinder_sim(d: float = 0.01):
    des = cylinder(d, t_body=25.0)
    U = 40 * AIR.kinematic_viscosity / d
    sim = tunnel_sim(des, U, thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False),
                     cfd=CFDSettings(quality="draft", device="cpu"),
                     emitters=[Emitter(name="red", species="dye", position=(-d, 0.2 * d), color="#e04040"),
                               Emitter(name="blue", species="dye", position=(-d, -0.2 * d), color="#4060e0"),
                               Emitter(name="dots", species="particles", kind="rake", position=(-d, -0.5 * d),
                                       end=(-d, 0.5 * d), count=4, rate=300.0, color="#ffaa00")],
                     fans=[Fan(name="F", position=(3 * d, 0.0), direction=(1.0, 0.0), diameter=d, mode="velocity",
                               value=U, frame="wind")])
    return des, sim


@pytest.fixture(scope="module")
def session():
    des, sim = heated_cylinder_sim()
    s = LiveSession(des, sim, device="cpu")
    s.tick(30.0)
    yield s
    s.close()


def test_the_session_accepts_every_live_editing_key(session):
    keys = session.POST_KEYS
    assert {"emitters", "fans", "heat_sources", "ports", "nu", "step", "steps_per_frame", "capture_trace",
            "obstacles", "paused", "reset", "temperature3d", "particles3d"} <= keys
    with pytest.raises(TypeError, match="unknown live setting"):
        session.post(warp=9)
    with pytest.raises(ValueError):
        session.post(nu=-1.0)
    with pytest.raises(ValueError):
        session.post(step=0)
    with pytest.raises(ValueError):
        session.post(steps_per_frame=10 ** 6)
    with pytest.raises(ValueError):                       # not TypeError: the worker would drop the key
        session.post(fans="F")


def test_frames_carry_the_placed_physics(session):
    f = session.tick(30.0)
    assert isinstance(f, LiveFrame)
    ph = f.physics
    assert ph["t_max"] == pytest.approx(25.0, abs=0.05) and ph["t_min"] == pytest.approx(15.0, abs=0.05)
    assert math.isfinite(ph["nu"]) and ph["heat_rate"] > 0 and "fan0_velocity" in ph
    assert f.particles is not None and f.particles.shape[1] == 2
    assert f.particle_palette == ["#ffaa00"] and len(f.particle_ids) == len(f.particles)
    assert f.diagnostics is not None and "step" in f.diagnostics and "div_rms" in f.diagnostics
    session.post(field="temperature")
    f = session.tick(30.0)
    assert f.field == "temperature" and f.units == "°C" and 14.9 <= f.vmin < f.vmax <= 25.1
    session.post(field="dye")
    f = session.tick(30.0)
    assert f.dye_channels is not None and [c["color"] for c in f.dye_channels] == ["#e04040", "#4060e0"]
    assert f.rgba.shape[2] == 4


def test_pause_step_frame_skip_and_trace_capture(session):
    session.post(paused=True)
    f0 = session.tick(30.0)
    assert session.tick(30.0).step == f0.step                # paused: nothing moves
    session.post(step=5)
    session.post(step=2)                                     # step requests add up
    assert session.tick(30.0).step == f0.step + 7
    session.post(step=1, capture_trace=True)
    f = session.tick(30.0)
    assert f.trace is not None and f.trace["stages"] and session.view.capture_trace is False
    assert all(a.ndim == 2 for a in f.trace["stages"].values())
    session.post(paused=False, steps_per_frame=6)
    a = session.tick(30.0)
    b = session.tick(30.0)
    assert b.step - a.step == 6
    session.post(steps_per_frame=0)                          # back to the frame budget
    session.tick(30.0)
    assert session._steps_per_frame is None


def test_moving_a_fan_and_retuning_it_reaches_the_solver_in_solver_units(session):
    _, sim = heated_cylinder_sim()
    fan = Fan.from_dict(sim.fans[0].to_dict())
    fan.position = (0.04, 0.005)
    fan.value = 2.0 * sim.fans[0].value
    session.post(fans=[fan])
    session.tick(30.0)
    spec = session.solver.cfg.fans[0]
    L, U = session.plan.length_ref, session.plan.speed
    assert spec.position == pytest.approx((0.04 / L, 0.005 / L))
    assert spec.value == pytest.approx(fan.value / U)


def test_the_reynolds_lock_and_a_speed_change_keep_the_units_consistent(session):
    re0 = session.solver.cfg.re
    nu0 = session.sim.conditions.fluid.kinematic_viscosity
    session.post(nu=2.0 * nu0)
    session.tick(30.0)
    assert session.solver.cfg.re == pytest.approx(re0 / 2.0)
    assert session.plan.physics.nu == pytest.approx(2.0 * nu0) and session.plan.physics.re == pytest.approx(re0 / 2)
    u0 = session.speed
    fan_si = session.sim.fans[0].value
    session.post(speed=1.5 * u0)
    session.tick(30.0)
    assert session.plan.speed == pytest.approx(1.5 * u0) and session.plan.physics.speed == pytest.approx(1.5 * u0)
    assert session.solver.cfg.re == pytest.approx(re0 / 2.0 * 1.5)
    # the fan keeps its set speed in m/s, so in units of the new U it is 1/1.5 of before
    assert session.solver.cfg.fans[0].value == pytest.approx(fan_si / (1.5 * u0))
    session.post(nu=nu0, speed=u0)
    session.tick(30.0)


def test_drawn_obstacles_are_painted_and_erased(session):
    h_m = session.plan.solver_config.h * session.plan.length_ref
    m0 = float(session.solver.field("mask").sum())
    # a stroke 20 cells long and 5 cells in radius (well resolved: the calibrated wall offset of the
    # smooth mask, a fraction of a cell along the outline, is then a few % of its area)
    r, half = 5 * h_m, 10 * h_m
    stroke = {"kind": "brush", "points": [[0.02, -half], [0.02, half]], "radius": r}
    session.post(obstacles={"ops": [stroke]})
    session.tick(30.0)
    m1 = float(session.solver.field("mask").sum())
    area = (2 * half * 2 * r + math.pi * r * r) / h_m ** 2                   # the capsule in cells
    assert m1 - m0 == pytest.approx(area, rel=0.12)
    session.post(obstacles={"ops": [stroke, dict(stroke, erase=True, radius=r + 2 * h_m)]})
    session.tick(30.0)
    assert float(session.solver.field("mask").sum()) == pytest.approx(m0, rel=1e-3, abs=1.0)
    session.post(obstacles=None)
    session.tick(30.0)


def test_a_closed_box_refuses_the_speed_slider_with_a_notice():
    des, sim, _ = cavity_sim(1e4)
    sim.cfd.device = "cpu"
    s = LiveSession(des, sim, device="cpu")
    try:
        s.tick(20.0)
        s.post(speed=3.0)
        f = s.tick(20.0)
        assert f.notices and "closed box has no free stream" in f.notices[0]
        s.post(speed=4.0)
        assert s.tick(20.0).notices is None                  # the same notice is not repeated
        for i in range(MAX_NOTICES + 20):
            s._notice(f"n{i}")
        assert len(s.notices) == MAX_NOTICES
        assert len(s.tick(20.0).notices) == MAX_NOTICES      # what was not sent yet, bounded
    finally:
        s.close()


def test_adding_the_first_fan_live_to_a_room_without_one():
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fans=[Fan(name="A", position=(1.0, 1.5), diameter=0.5, value=0.5,
                                                   frame="wind")])
    s = LiveSession(des, sim, device="cpu")
    try:
        s.tick(20.0)
        two = sim.fans + [Fan(name="B", position=(3.0, 1.0), direction=(-1.0, 0.0), diameter=0.5, value=0.5,
                              frame="wind")]
        s.post(fans=two)
        f = s.tick(20.0)
        assert [x.name for x in s.solver.cfg.fans] == ["A", "B"] and "fan1_thrust" in f.physics
    finally:
        s.close()


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session3d():
    des = design_from_meshes([sphere(0.02, 24, name="Sphere")], "Hot sphere", kind="generic")
    for m in des.meshes:
        m.thermal, m.thermal_value = "temperature", 25.0
    sim = tunnel_sim(des, 50 * AIR.kinematic_viscosity / 0.02,
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False),
                     cfd=CFDSettings(quality="draft", device="cpu"),
                     emitters=[Emitter(name="smoke", species="dye", position=(-0.02, 0.0, 0.0)),
                               Emitter(name="dots", species="particles", kind="rake", position=(-0.02, -0.005, 0.0),
                                       end=(-0.02, 0.005, 0.0), count=3, rate=300.0, color="#ffaa00")])
    s = LiveSession(des, sim, device="cpu")
    s.tick(30.0)
    yield s
    s.close()


def test_the_3d_session_takes_the_same_live_editing_keys(session3d):
    s = session3d
    assert isinstance(s, LiveSession3D)
    assert {"emitters", "fans", "nu", "step", "steps_per_frame", "capture_trace", "obstacles", "yaw_deg",
            "slice_axis", "temperature3d", "particles3d"} <= s.POST_KEYS
    with pytest.raises(ValueError):
        s.post(obstacles=[1, 2])
    with pytest.raises(ValueError):
        s.post(slice_axis="w")


def test_3d_frames_carry_temperature_particles_and_volumes(session3d):
    s = session3d
    s.post(field="temperature", view3d=True, temperature3d=True)
    f = s.tick(30.0)
    assert f.field == "temperature" and f.units == "°C" and 14.9 <= f.vmin < f.vmax <= 25.1
    assert f.particles is not None and f.particles.shape[1] == 3 and f.particle_palette == ["#ffaa00"]
    assert f.volume is not None and {"dye", "temperature"} <= set(f.volume.fields)
    assert f.volume.fields["temperature"].dtype == np.float32                 # °C needs more than float16
    assert f.physics["t_max"] == pytest.approx(25.0, abs=0.05) and f.physics["heat_rate"] > 0
    s.post(temperature3d=False, particles3d=False)
    f = s.tick(30.0)
    assert "temperature" not in f.volume.fields and f.particles is None
    s.post(view3d=False, temperature3d=True, particles3d=True, field="speed")


def test_3d_pause_step_and_trace_on_the_slice_shown(session3d):
    s = session3d
    s.post(paused=True)
    f0 = s.tick(30.0)
    s.post(step=3, capture_trace=True, slice_axis="z")
    f = s.tick(30.0)
    assert f.step == f0.step + 3 and f.trace is not None
    nx, ny, _ = s.plan.grid3
    assert all(a.shape == (ny, nx) for a in f.trace["stages"].values())     # cut on the z slice
    s.post(paused=False, slice_axis="y")
    s.tick(30.0)


def test_3d_obstacles_are_balls_and_boxes(session3d):
    s = session3d
    m0 = float(s.solver.field("mask").sum())
    h_m = s.plan.solver_config.h * s.plan.length_ref
    r = 4 * h_m                                                           # resolved: four cells in radius
    s.post(obstacles={"ops": [{"kind": "sphere", "center": [0.03, 0.0, 0.0], "radius": r},
                              {"kind": "box", "center": [0.03, 0.0, 0.0], "size": [r, r, r], "erase": True}]})
    s.tick(30.0)
    m1 = float(s.solver.field("mask").sum())
    assert m1 - m0 == pytest.approx((4 / 3 * math.pi * r ** 3 - r ** 3) / h_m ** 3, rel=0.2)
    s.post(obstacles=None)
    s.tick(30.0)
    assert float(s.solver.field("mask").sum()) == pytest.approx(m0, rel=1e-3)
    with pytest.raises(ValueError, match="balls"):
        apply_obstacles_3d(s.solver, {"ops": [{"kind": "cone", "center": [0, 0, 0]}]}, length=s.plan.length_ref)
