"""``aerolume.simulation.SUPPORTED_PHYSICS``: what the planners and solvers really run (DESIGN.md 10).

The scripting API (``al.supported_physics``), the AI assistant's run tools and the Setup page
refuse or flag any placed physics missing from this table, so it must list every feature the
planner carries into the solver -- and nothing it does not. Each feature is checked by planning
a case that uses it and finding it in the solver configuration, in 2D and in 3D. (Round 3's
critique found the table missing altogether: every heated, fan or smoke run from a script was
refused while the GUI ran the same cases.)
"""
from __future__ import annotations

import subprocess
import sys

import pytest
from test_simulation_physics import AIR, closed_sim, cylinder, room2d, room3d, tunnel_sim

import aerolume as al
from aerolume.api import PHYSICS_FEATURES, physics_features
from aerolume.geometry3d import design_from_meshes, sphere
from aerolume.model import CFDSettings, Emitter, Fan, HeatSource, Port, ThermalSettings, WallThermal
from aerolume.simulation import SUPPORTED_PHYSICS
from aerolume.simulation import plan as planmod
from aerolume.simulation.plan import plan_simulation


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def _everything(dim: int, *, closed: bool):
    """A case with every placed-physics feature on, in a closed room or a tunnel."""
    th = ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True,
                         walls=[WallThermal("left", "temperature", 20.0)] if closed else [])
    if dim == 2:
        pos, d, box = (2.0, 1.5), (1.0, 0.0), (0.0, 4.0, 0.0, 3.0)
        heater = HeatSource(name="H", kind="rect", position=(3.0, 0.5), size=(0.3, 0.2), power=100.0, frame="wind")
    else:
        pos, d, box = (2.0, 1.5, 1.25), (1.0, 0.0, 0.0), (0.0, 4.0, 0.0, 3.0, 0.0, 2.5)
        heater = HeatSource(name="H", kind="box", position=(3.0, 1.5, 0.5), size=(0.3, 0.3, 0.3), power=100.0,
                            frame="wind")
    kw = {"thermal": th, "heat_sources": [heater],
          "emitters": [Emitter(name="smoke", species="dye", position=pos, frame="wind")],
          "fans": [Fan(name="F", position=pos, direction=d, diameter=0.5, mode="velocity", value=0.5, frame="wind")],
          "ports": [Port(name="J", kind="jet", position=tuple(0.2 if i == 0 else c for i, c in enumerate(pos)),
                         direction=d, size=0.3, speed=0.3, frame="wind"),
                    Port(name="V", kind="vent", position=tuple(3.8 if i == 0 else c for i, c in enumerate(pos)),
                         direction=d, size=0.3, speed=0.3, frame="wind")]}
    if closed:
        des = room2d() if dim == 2 else room3d()
        sim = closed_sim(des, box, cfd_kw={"adaptive_dt": "pid"}, **kw)
    else:
        des = cylinder(0.2, t_body=25.0) if dim == 2 else design_from_meshes([sphere(0.2, 24)], "S", kind="generic")
        sim = tunnel_sim(des, 1.0, cfd=CFDSettings(quality="draft", device="cpu", adaptive_dt="pid"), **kw)
        sim.fans[0].position = tuple(-0.4 if i == 0 else 0.0 for i in range(dim))
        sim.heat_sources[0].position = tuple(0.6 if i == 0 else 0.0 for i in range(dim))
        sim.emitters[0].position = tuple(-0.6 if i == 0 else 0.0 for i in range(dim))
        for p, x in zip(sim.ports, (-0.5, 0.8), strict=True):
            p.position = tuple(x if i == 0 else 0.15 for i in range(dim))
    return des, sim


def _carried(plan) -> set[str]:
    """The features a plan's solver configuration really holds."""
    cfg = plan.solver_config
    got = set()
    if getattr(cfg, "emitters", None):
        got.add("emitters")
    if getattr(cfg, "thermal", None) is not None:
        got.add("thermal")
    if getattr(cfg, "heat_sources", None):
        got.add("heat_sources")
    if getattr(cfg, "fans", None):
        got.add("fans")
    if getattr(cfg, "ports", None):
        got.add("ports")
    if getattr(cfg, "domain", "tunnel") == "closed":
        got.add("closed_domain")
    if getattr(cfg, "adaptive_dt", "cfl") == "pid":
        got.add("adaptive_dt")
    return got


def test_the_table_names_only_features_the_api_knows():
    for dim in (2, 3):
        assert set(SUPPORTED_PHYSICS[dim]) <= set(PHYSICS_FEATURES)
    assert set(SUPPORTED_PHYSICS) == {2, 3}


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("closed", [True, False])
def test_every_supported_feature_reaches_the_solver(dim, closed):
    des, sim = _everything(dim, closed=closed)
    used = set(physics_features(sim))
    plan = plan_simulation(des, sim)
    carried = _carried(plan)
    assert used <= carried, f"planned without {used - carried}"
    assert used <= set(SUPPORTED_PHYSICS[dim])


@pytest.mark.parametrize("dim", [2, 3])
def test_the_api_runs_what_the_gui_planner_accepts(dim):
    """api.supported_physics covers every feature the planner carries, so nothing the GUI runs
    is refused from a script (the critique's blocker)."""
    have = al.supported_physics(dim)
    for closed in (True, False):
        des, sim = _everything(dim, closed=closed)
        assert _carried(plan_simulation(des, sim)) <= have
    assert have == frozenset(PHYSICS_FEATURES)
    assert al.supported_physics() == frozenset(PHYSICS_FEATURES)


def test_reading_the_table_does_not_import_torch():
    """The API, the Setup page and the AI tools read it on every check: it must stay cheap."""
    code = ("import sys, aerolume.simulation as S; S.SUPPORTED_PHYSICS; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True)
    assert out.stdout.strip() == "False"


def test_a_heated_cavity_and_a_fan_room_plan_from_a_script():
    """The two examples the critique saw refused (examples/scripts/heated_cavity.py, fan_room.py)."""
    p = al.new_project()
    cav = al.cavity_case(ra=1e4)
    assert "Rayleigh" in str(p.plan(cav))
    room = al.fan_room_case(thrust=1.0)
    smoke = [{"kind": "rake", "position": (0.4, 0.4), "end": (0.4, 2.6), "count": 8}]
    assert p.plan(room, emitters=smoke) is not None
    assert AIR.name                                                     # (shared fixture module imported)
