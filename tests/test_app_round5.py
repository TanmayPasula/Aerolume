"""Application findings of critic round 5 (each measured before the fix).

* Results offered a Forces tab (six coefficient histories, C_D swinging to -1.2) and a
  Sensitivity tab for a closed room, whose Summary deliberately hides coefficients; both are
  hidden for closed domains now (the histories stay in Data / CSV).
* Explore drawing: Undo cleared the paint layer and replayed every stroke (1.3 s at 20
  strokes on the room grid); it now restores a checkpoint, which must give exactly the layer
  a full replay gives.
* The first tour step listed the electronics enclosure as if it were hidden behind Ctrl+N.
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _pump(app, seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


@pytest.mark.gui
def test_results_hide_forces_and_sensitivity_for_a_closed_room(qapp, tmp_path):
    pytest.importorskip("pyqtgraph")
    from test_app_runtime_physics import circle, thermal_result

    from aerolume.app.jobs import JobManager
    from aerolume.app.pages.results import TAB_KEYS, ResultsPage
    from aerolume.app.state import AppState
    from aerolume.model import Body, Design, FlowConditions, Simulation, ThermalSettings
    from aerolume.project import Project
    pr = Project.create(tmp_path / "Room.aerolume", "Room")
    d = Design(name="Cabinet", kind="generic", bodies=[Body("Cabinet", circle(0, 0, 0.05), thermal="temperature",
                                                            thermal_value=60.0)],
               length_ref=0.1, force_ref=0.1, moment_ref=(0.0, 0.0))
    pr.add_design(d)
    tunnel = Simulation(name="Tunnel", design_id=d.id, conditions=FlowConditions(speed=0.5), status="done", id="tun",
                        thermal=ThermalSettings(enabled=True, t_ambient=20.0))
    room = Simulation(name="Room", design_id=d.id, conditions=FlowConditions(speed=0.5), status="done", id="room",
                      thermal=ThermalSettings(enabled=True, t_ambient=20.0))
    room.cfd.domain_mode = "closed"
    pr.add_simulation(tunnel)
    pr.add_simulation(room)
    pr.save()
    pr.save_result("tun", thermal_result("tun"))
    pr.save_result("room", thermal_result("room"))
    state = AppState()
    state.set_project(pr)
    jobs = JobManager(state)
    page = ResultsPage(state, jobs)
    try:
        page.resize(1400, 900)
        page.show()
        state.select_simulation("tun")
        _pump(qapp, 0.2)
        for key in ("forces", "sensitivity", "heat"):
            assert page.tabs.isTabVisible(TAB_KEYS.index(key)), key
        state.select_simulation("room")
        _pump(qapp, 0.2)
        for key in ("forces", "sensitivity"):
            assert not page.tabs.isTabVisible(TAB_KEYS.index(key)), key
        assert page.tabs.isTabVisible(TAB_KEYS.index("heat"))
        # named for screen readers (the tab bar and the tables were nameless Tab stops)
        assert page.tabs.tabBar().accessibleName() == "Result views"
        assert page.heat_table.accessibleName() and page.balance_table.accessibleName() == "Heat balance"
    finally:
        page.close()
        page.deleteLater()
        jobs.shutdown(timeout=5)


def test_explore_undo_restores_a_checkpoint_equal_to_a_full_replay():
    from aerolume.simulation.live import apply_obstacles_2d
    from aerolume.solvers.ns2d import FlowSolver
    from aerolume.solvers.ns2d.config import SolverConfig

    def solver():
        return FlowSolver(SolverConfig(nx=96, ny=64, h=1 / 32, x0=0.0, y0=0.0, re=100.0, domain="closed",
                                       bc_top="noslip", bc_bottom="noslip", dye=False, device="cpu",
                                       dtype="float64", use_cuda_graphs=False))

    def stroke(k, erase=False):
        pts = [(0.3 + 0.05 * i, 0.4 + 0.25 * k + 0.03 * np.sin(i)) for i in range(25)]
        return {"kind": "brush", "points": pts, "radius": 0.05, "erase": erase}

    ops = [stroke(0), stroke(1), stroke(2), stroke(1, erase=True), stroke(3)]
    s = solver()
    ckpt: dict = {}
    applied: list = []
    for n in range(1, len(ops) + 1):                 # drawn one by one: pure extensions
        applied = apply_obstacles_2d(s, {"ops": ops[:n]}, length=1.0, applied=applied, checkpoints=ckpt)
    assert sorted(ckpt) == [1, 2, 3, 4, 5]
    calls = []
    real_clear = s.clear_paint
    s.clear_paint = lambda: (calls.append("clear"), real_clear())[1]
    applied = apply_obstacles_2d(s, {"ops": ops[:3]}, length=1.0, applied=applied, checkpoints=ckpt)   # Undo x2
    assert calls == [] and sorted(ckpt) == [1, 2, 3]          # restored from the checkpoint, no replay
    ref = solver()
    apply_obstacles_2d(ref, {"ops": ops[:3]}, length=1.0)      # a replay from an empty layer
    for a, b in zip(s.paint_snapshot(), ref.paint_snapshot()):
        assert torch.equal(a, b)
    # an edited stroke in the middle: back to the last shared checkpoint, then replay the rest
    edited = [ops[0], stroke(4), ops[2]]
    apply_obstacles_2d(s, {"ops": edited}, length=1.0, applied=applied, checkpoints=ckpt)
    ref2 = solver()
    apply_obstacles_2d(ref2, {"ops": edited}, length=1.0)
    for a, b in zip(s.paint_snapshot(), ref2.paint_snapshot()):
        assert torch.equal(a, b)
    assert calls == []


def test_vectorised_stroke_distance_matches_the_segment_loop():
    from aerolume.simulation.live import _polyline_sdf
    pts = np.array([(0.1 * i, 0.05 * np.sin(i)) for i in range(30)])
    y, x = torch.meshgrid(torch.linspace(-0.5, 0.5, 41, dtype=torch.float64),
                          torch.linspace(-0.2, 3.2, 70, dtype=torch.float64), indexing="ij")
    got = _polyline_sdf(pts, 0.04)(x, y)
    best = None
    for a, b in itertools.pairwise(pts):
        d = b - a
        t = (((x - a[0]) * d[0] + (y - a[1]) * d[1]) / float(d @ d)).clamp(0, 1)
        dist = torch.hypot(x - (a[0] + t * d[0]), y - (a[1] + t * d[1]))
        best = dist if best is None else torch.minimum(best, dist)
    assert torch.allclose(got, best - 0.04, atol=1e-12)


def test_first_tour_step_names_the_enclosure_as_a_visible_tile():
    from aerolume.app.widgets.tour import START_TOUR_STEPS
    text = START_TOUR_STEPS[0].text
    assert "a heated room or an electronics enclosure" in text and "including the electronics" not in text
    assert "Ctrl+N also offers an empty project" in text
