"""Explore mode: LiveSession post / tick / alpha and speed changes / road / close.

CPU sessions on the explore budget (~45 k cells) keep these fast; one GPU test
checks the CUDA-graph pacing.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from aerolume.core import fluid as F
from aerolume.geometry import design_from_airfoil, naca, templates
from aerolume.model import CFDSettings, FlowConditions, GroundConfig, Simulation
from aerolume.simulation.live import LiveFrame, LiveSession, ViewSettings, downsample_history

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
AIR = F.air_isa(0.0)


def speed_for(re: float, length: float = 1.0) -> float:
    return re * AIR.viscosity / (AIR.density * length)


@pytest.fixture
def airfoil_session():
    d = design_from_airfoil(naca("0012", 160, 1.0), "NACA 0012", 1.0)
    sim = Simulation("live", d.id, conditions=FlowConditions(AIR, speed_for(300.0), 2.0),
                     cfd=CFDSettings(quality="draft"))
    s = LiveSession(d, sim, device="cpu")
    yield s
    s.close()


def test_tick_returns_a_live_frame(airfoil_session):
    s = airfoil_session
    f = s.tick(30.0)
    assert isinstance(f, LiveFrame) and f.phase == "live" and f.progress == 0.0 and f.eta_s is None
    nx, ny = s.plan.grid
    assert f.rgba.shape == (ny, nx, 4) and f.rgba.dtype == np.uint8
    assert f.extent == pytest.approx(s.plan.extent) and f.focus is not None
    assert f.step > 0 and f.t > 0 and set(f.coeffs) == {"cd", "cl", "cm"}
    assert len(f.bodies) == 1 and f.bodies[0].shape[1] == 2


def test_first_tick_does_not_set_the_pace(airfoil_session):
    s = airfoil_session
    s.tick(30.0)                                    # warm-up (graph capture on CUDA): not timed
    assert s._sec_per_step is None
    s.tick(30.0)
    assert s._sec_per_step is not None and s._sec_per_step > 0
    assert np.isfinite(s.tick(30.0).steps_per_s)


def test_alpha_change_rotates_the_bodies_and_keeps_the_flow(airfoil_session):
    s = airfoil_session
    f0 = s.tick(20.0)
    s.post(alpha_deg=10.0)
    f1 = s.tick(20.0)
    assert s.alpha_deg == pytest.approx(10.0) and f1.step > f0.step     # flow kept, not reset
    te0, te1 = f0.bodies[0][0], f1.bodies[0][0]                        # trailing edge goes down
    assert te1[1] < te0[1] - 0.05
    s.post(alpha_deg=80.0)                                              # beyond the slider range
    s.tick(10.0)
    assert s.alpha_deg == pytest.approx(25.0)


def test_speed_change_only_changes_the_reynolds_number(airfoil_session):
    s = airfoil_session
    s.tick(10.0)
    re0 = s.solver.cfg.re
    s.post(speed=2 * s.speed)
    f = s.tick(10.0)
    assert s.solver.cfg.re == pytest.approx(2 * re0) and np.all(np.isfinite(list(f.coeffs.values())))
    with pytest.raises(ValueError):
        s.post(speed=-1.0)
    with pytest.raises(TypeError, match="unknown live setting"):
        s.post(colour="red")


def test_view_changes_apply_immediately(airfoil_session):
    s = airfoil_session
    s.post(field="vorticity", cmap="coolwarm", particles=True, streamlines=True)
    assert s.view.field == "vorticity" and s.view.cmap == "coolwarm"
    f = s.tick(20.0)
    f = s.tick(20.0)
    assert f.field == "vorticity" and f.cmap == "coolwarm" and f.units == "1/s"
    assert f.particles is not None and f.particles.shape[1] == 2
    assert isinstance(f.streamlines, list)
    s.post(field="no-such-field")
    assert s.tick(10.0).field == "speed"                                # unknown names fall back


def test_pause_and_reset(airfoil_session):
    s = airfoil_session
    s.tick(20.0)
    s.post(paused=True)
    a = s.tick(20.0).step
    b = s.tick(20.0).step
    assert a == b
    s.post(paused=False, reset=True)
    f = s.tick(20.0)
    assert f.step <= 2000 and f.t < 0.5


def test_post_is_thread_safe(airfoil_session):
    s = airfoil_session
    errors = []

    def poster():
        try:
            for a in np.linspace(-5, 5, 50):
                s.post(alpha_deg=float(a), speed=s.speed)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=poster)
    th.start()
    for _ in range(5):
        s.tick(10.0)
    th.join()
    s.tick(10.0)
    assert not errors and s.alpha_deg == pytest.approx(5.0)


def test_close_is_idempotent_and_final(airfoil_session):
    s = airfoil_session
    s.tick(10.0)
    s.close()
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.tick(10.0)


def test_road_switch_rebuilds_with_the_same_flow():
    d = templates.make("ahmed")
    sim = Simulation("car", d.id, conditions=FlowConditions(AIR, 5.0, 0.0),
                     ground=GroundConfig(enabled=True, moving=True), cfd=CFDSettings(quality="draft"))
    s = LiveSession(d, sim, device="cpu")
    try:
        assert s.plan.ground and s.plan.blockage <= 0.10 + 1e-9
        s.tick(20.0)
        t0 = s.solver.t
        s.post(ground_moving=False)
        f = s.tick(20.0)
        assert s.solver.cfg.bc_bottom == "noslip" and not s.sim.ground.moving
        assert f.t >= t0                                               # state carried over
        s.post(alpha_deg=-20.0)                                        # would dig into the road
        s.tick(10.0)
        assert -5.0 - 1e-9 <= s.alpha_deg <= 0.0
        ymin = min(float(b[:, 1].min()) for b in s._bodies)
        assert ymin >= 0.0
    finally:
        s.close()


def test_downsample_history_keeps_the_latest_sample():
    t = np.arange(10_000, dtype=float)
    h = downsample_history({"t": t, "cd": t * 2}, 100)
    assert h["t"].size <= 100 and h["t"][-1] == t[-1] and np.allclose(h["cd"], 2 * h["t"])


@pytest.mark.gpu
@needs_cuda
def test_gpu_session_runs_many_steps_per_tick():
    d = design_from_airfoil(naca("0012", 160, 1.0), "NACA 0012", 1.0)
    sim = Simulation("live", d.id, conditions=FlowConditions(AIR, speed_for(1000.0), 4.0))
    s = LiveSession(d, sim, device="cuda", view=ViewSettings(field="speed"))
    try:
        steps = [s.tick(33.0).step for _ in range(12)]
        per_tick = np.diff(steps)[-5:]
        assert per_tick.min() >= 10 and all(n % 10 == 0 for n in per_tick)   # whole CUDA-graph blocks
        assert s.solver.uses_cuda_graphs
    finally:
        s.close()
