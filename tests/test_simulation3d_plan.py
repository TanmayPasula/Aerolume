"""The 3D planner (DESIGN.md 7.7): dispatch, frame, symmetry plane, domain, grid, budgets, rationale.

Planning never runs the solver, so these tests are fast; GPU-class plans are
skipped without CUDA (the budgets differ per device class).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from aerolume.core import device as devmod
from aerolume.core import fluid as F
from aerolume.geometry import design_from_airfoil, naca
from aerolume.geometry3d import box, cylinder, design_from_meshes, design_from_wing, sphere
from aerolume.model import (
    CFDSettings,
    FlowConditions,
    GroundConfig,
    Mesh3D,
    Simulation,
    WingSection,
    WingSpec,
)
from aerolume.simulation import defaults as D
from aerolume.simulation import plan as planmod
from aerolume.simulation import plan3d as P3
from aerolume.simulation.plan import SimPlan, plan_live, plan_simulation
from aerolume.simulation.plan3d import SimPlan3D, rotation3, wind_frame_meshes

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = ["cpu", pytest.param("cuda", marks=needs_cuda)]
AIR = F.air_isa(0.0)


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    """Keep run-time measurements (2D and 3D) and any calibration file out of these tests."""
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def speed_for(re: float, length: float = 1.0) -> float:
    return re * AIR.viscosity / (AIR.density * length)


def sphere_design(d: float = 1.0):
    return design_from_meshes([sphere(d, 32)], "Sphere", kind="generic")


def car_design(wheels: bool = True):
    body = box(4.5, 1.8, 1.2, center=(2.25, 0.0, 0.8), name="body")
    parts = [body]
    if wheels:
        parts += [cylinder(0.64, 0.25, axis="y", center=(x, y, 0.32), name=f"wheel {i}")
                  for i, (x, y) in enumerate([(0.9, 0.75), (0.9, -0.75), (3.6, 0.75), (3.6, -0.75)])]
    return design_from_meshes(parts, "Box car", kind="vehicle")


def wing_design(airfoil: str = "2412"):
    w = WingSpec([WingSection(0.0, 1.0, airfoil=airfoil), WingSection(5.0, 0.6, x_le=0.1, airfoil=airfoil)])
    return design_from_wing(w, "Wing")


def make_sim(design, *, speed: float = 30.0, alpha: float = 0.0, yaw: float = 0.0, ground: bool | None = None,
             moving: bool = True, rotate: bool = True, **cfd) -> Simulation:
    on_road = design.kind == "vehicle" if ground is None else ground
    return Simulation("t", design.id, conditions=FlowConditions(AIR, speed, alpha, yaw),
                      ground=GroundConfig(enabled=on_road, moving=moving, rotate_wheels=rotate),
                      cfd=CFDSettings(**cfd))


# ---------------------------------------------------------------------------
# dispatch, frame
# ---------------------------------------------------------------------------

def test_plan_simulation_dispatches_on_dimension():
    d2 = design_from_airfoil(naca("0012", 160, 1.0), "NACA 0012", 1.0)
    p2 = plan_simulation(d2, make_sim(d2, speed=speed_for(1000.0), quality="draft", device="cpu"))
    assert isinstance(p2, SimPlan) and not isinstance(p2, SimPlan3D) and len(p2.grid) == 2
    d3 = sphere_design()
    p3 = plan_simulation(d3, make_sim(d3, speed=speed_for(100.0), quality="draft", device="cpu"))
    assert isinstance(p3, SimPlan3D) and p3.dimension == 3
    assert p3.grid == p3.grid3 and len(p3.grid3) == 3 and p3.cells == math.prod(p3.grid3)
    assert p3.bodies_wind == [] and len(p3.bodies_wind3) == 1
    rows = p3.summary()
    for key in ("Real Reynolds number", "Simulated Reynolds number", "Flow model", "Grid", "Cells", "Cells (3D)",
                "Symmetry plane", "Blockage", "Domain", "Time step", "Duration", "Stops early when",
                "Estimated run time", "Device"):
        assert key in rows, key
    assert "half domain" in rows["Cells (3D)"] and rows["Symmetry plane"].startswith("y = 0")
    assert p3.estimate_label.startswith("~") or p3.estimate_label == "< 10 s"


def test_plan_live_dispatches_and_a_wing_analysis_is_not_a_cfd_run():
    d = sphere_design()
    lp = plan_live(d, make_sim(d, speed=speed_for(100.0), device="cpu"))
    assert isinstance(lp, SimPlan3D) and lp.live and math.isinf(lp.t_end)
    assert "Duration" not in lp.summary() and lp.estimate_label == ""
    sim = make_sim(d, speed=10.0, device="cpu")
    sim.analysis = "wing"
    with pytest.raises(ValueError, match="wing analysis"):
        plan_simulation(d, sim)


def test_rotation_follows_design_7_2():
    nose = np.array([-1.0, 0.0, 0.0])
    up = rotation3(10.0, 0.0) @ nose
    assert up[2] > 0 and abs(up[1]) < 1e-12               # +alpha lifts the nose (which points to -x)
    left = rotation3(0.0, 10.0) @ nose
    assert left[1] > 0 and abs(left[2]) < 1e-12           # +yaw: nose to +y, wind from the left
    # yaw first, then pitch
    both = rotation3(10.0, 20.0)
    assert np.allclose(both, rotation3(10.0, 0.0) @ rotation3(0.0, 20.0))
    # on a road the wheels only yaw; the body pitches
    car = car_design()
    wind = wind_frame_meshes(car.meshes, 3.0, 5.0, car.moment_ref3, ground=True)
    for m0, m1 in zip(car.meshes, wind):
        ref = np.asarray(car.moment_ref3)
        expect = rotation3(0.0, 5.0) if m0.role == "wheel" else rotation3(3.0, 5.0)
        assert np.allclose(m1.vertices, (m0.vertices - ref) @ expect.T + ref)
        assert np.array_equal(m1.faces, m0.faces)          # vertex order kept: results map back


def test_pitched_body_into_the_road_is_refused():
    car = car_design(wheels=False)
    with pytest.raises(ValueError, match="below the road"):
        plan_simulation(car, make_sim(car, alpha=20.0, quality="draft", device="cpu"))


# ---------------------------------------------------------------------------
# symmetry plane
# ---------------------------------------------------------------------------

def test_symmetry_plane_rules():
    d = sphere_design()
    assert d.symmetric
    half = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu"))
    cfg = half.solver_config
    assert half.symmetry_plane and cfg.bc_y == "symmetry_ymin" and cfg.y0 == 0.0 and half.extent3[2] == 0.0
    assert half.extent3_full[2] == pytest.approx(-half.extent3[3])
    yawed = plan_simulation(d, make_sim(d, speed=speed_for(100.0), yaw=5.0, quality="draft", device="cpu"))
    assert not yawed.symmetry_plane and yawed.solver_config.bc_y == "slip"
    assert any("yaw of 5" in n for n in yawed.notes)
    full = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu", symmetry="full"))
    assert not full.symmetry_plane and any("as set in Setup" in n for n in full.notes)
    # a design flagged symmetric that is not (moved after import) gets both halves and a warning
    moved = sphere_design()
    moved.meshes = [Mesh3D(name=m.name, vertices=m.vertices + [0.0, 0.3, 0.0], faces=m.faces) for m in moved.meshes]
    p = plan_simulation(moved, make_sim(moved, speed=speed_for(100.0), quality="draft", device="cpu"))
    assert not p.symmetry_plane and any("marked mirror-symmetric" in w for w in p.warnings)


def test_full_domain_is_centred_and_twice_as_wide_at_the_same_resolution():
    d = sphere_design()
    kw = {"speed": speed_for(100.0), "quality": "draft", "device": "cpu", "cells_per_ref": 8}
    half = plan_simulation(d, make_sim(d, **kw))
    full = plan_simulation(d, make_sim(d, symmetry="full", **kw))
    assert full.grid3[1] == 2 * half.grid3[1] and full.grid3[0] == half.grid3[0] and full.grid3[2] == half.grid3[2]
    assert full.extent3 == pytest.approx(half.extent3_full)
    assert full.blockage == pytest.approx(half.blockage)


# ---------------------------------------------------------------------------
# domain, grid, budgets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("make", [sphere_design, car_design, wing_design])
def test_grids_respect_the_budget_and_refine_with_quality(device, make):
    d = make()
    dclass = "cpu" if device == "cpu" else "gpu"
    speed = speed_for(100.0) if d.kind == "generic" else 30.0
    cpl = []
    for qi, q in enumerate(("draft", "standard", "fine")):
        p = plan_simulation(d, make_sim(d, speed=speed, quality=q, device=device))
        budget = P3.CELL_BUDGET3[dclass][qi]
        if dclass == "gpu" and devmod.info(p.device).memory_gb < 7.5:
            budget = int(budget * devmod.info(p.device).memory_gb / 8.0)
        assert p.cells <= budget, (q, p.cells, budget)
        assert all(n % P3.GRID_MULTIPLE == 0 for n in p.grid3)
        x0, x1, y0, y1, z0, z1 = p.extent3
        bx0, bx1, by0, by1, bz0, bz1 = p.body_bounds
        assert x0 < bx0 and x1 > bx1 and z1 > bz1 and (z0 == 0.0 if p.ground else z0 < bz0)
        assert y1 > by1 and (y0 == 0.0 if p.symmetry_plane else y0 < by0)
        h = p.solver_config.h * p.length_ref
        assert x1 - x0 == pytest.approx(p.grid3[0] * h)
        cpl.append(p.cells_per_ref)
        if dclass == "gpu":
            assert p.t_end > p.t_avg_start > 0 and p.steps_estimate > 0 and p.est_seconds > 0
    assert cpl[0] <= cpl[1] <= cpl[2]
    if dclass == "gpu":
        assert cpl[0] < cpl[2]


@pytest.mark.parametrize("make", [sphere_design, car_design, wing_design])
def test_blockage_is_limited_by_area_and_in_each_direction(make):
    d = make()
    speed = speed_for(100.0) if d.kind == "generic" else 30.0
    p = plan_simulation(d, make_sim(d, speed=speed, quality="draft", device="cpu"))
    geo = p.geometry_class
    if geo == "vehicle":                         # never traded for cells (as in 2D)
        assert p.blockage <= 1.02 * P3.BLOCKAGE_MAX3[geo][0]
        assert p.blockage_y <= P3.DIRECTION_MAX3[geo][0] + 1e-9
        assert p.blockage_z <= P3.DIRECTION_MAX3[geo][0] + 1e-9
    else:                                        # may trade up to 1.6x for the budget, with a warning
        assert p.blockage <= 1.6 * 1.02 * P3.BLOCKAGE_MAX3[geo][0]
        assert p.blockage_y <= P3.DIRECTION_MAX3[geo][0] + 1e-9
        if p.blockage > 1.05 * P3.BLOCKAGE_MAX3[geo][0]:
            assert any(w.startswith("Blockage") for w in p.warnings)
    if geo == "wing":
        # a 10 m wing: the tunnel is at least span / 0.6 wide (frictionless walls bind the tip vortices)
        _, _, y0, y1, _, _ = p.extent3_full
        assert y1 - y0 >= 10.0 / P3.DIRECTION_MAX3["wing"][0] - 1e-6


def test_vehicle_layout_on_a_moving_road_with_rolling_wheels():
    car = car_design()
    p = plan_simulation(car, make_sim(car, speed=30.0, quality="draft", device="cpu"))
    cfg = p.solver_config
    assert p.ground and p.ground_moving and cfg.bc_zmin == "moving" and cfg.z0 == 0.0
    assert p.geometry_class == "vehicle" and p.turbulence == "les-wale" and p.advection == "upwind3"
    wheels = [b for b in cfg.bodies if b.role == "wheel"]
    assert len(wheels) == 4
    for w in wheels:
        # omega = -U / r about +y (right-hand rule, ns3d config docstring), nondimensional: -L / r
        assert w.axis == pytest.approx((0.0, 1.0, 0.0))
        assert w.omega == pytest.approx(-car.length_ref / 0.32, rel=0.02)
    fixed = plan_simulation(car, make_sim(car, speed=30.0, moving=False, quality="draft", device="cpu"))
    assert fixed.solver_config.bc_zmin == "noslip" and all(b.omega == 0 for b in fixed.solver_config.bodies)
    free = plan_simulation(car, make_sim(car, ground=False, quality="draft", device="cpu"))
    assert free.solver_config.bc_zmin == "slip" and any("road is switched off" in n for n in free.notes)
    assert any("Reading the numbers" in n for n in p.notes)
    assert any("boundary layers are much thinner" in w for w in p.warnings)


def test_user_overrides_resolution_domain_and_duration():
    d = sphere_design()
    p = plan_simulation(d, make_sim(d, speed=speed_for(100.0), device="cpu", cells_per_ref=10, domain_upstream=3.0,
                                    domain_downstream=5.0, duration=12.0, average_from=4.0))
    assert p.cells_per_ref == 10 and p.solver_config.h == pytest.approx(0.1)
    assert p.body_bounds[0] - p.extent3[0] == pytest.approx(3.0, abs=1e-9)
    assert p.extent3[1] - p.body_bounds[1] >= 5.0 - 1e-9
    assert p.t_end == 12.0 and p.t_avg_start == 4.0
    assert "set by hand" in " ".join(p.notes)


def test_thin_feature_raises_the_resolution_but_only_so_far():
    plate = box(0.4, 0.4, 0.02, name="plate")             # 2 cm thick, 40 cm wide
    d = design_from_meshes([plate], "Plate", kind="generic")
    p = plan_simulation(d, make_sim(d, speed=5.0, quality="draft", device="cpu"))
    target = P3.CELLS_PER_SIZE3["generic"]["cpu"][0] * d.length_ref / (math.sqrt(4 * d.ref_area / math.pi))
    assert p.cells_per_ref <= P3.THIN_RAISE_MAX["generic"] * target + 1.0
    assert any("thick" in w and "spans only" in w for w in p.warnings)


def test_gap_between_parts_that_the_grid_closes_is_reported():
    a = box(0.5, 0.5, 0.5, center=(0.0, 0.0, 0.0), name="front")
    b = box(0.5, 0.5, 0.5, center=(0.52, 0.0, 0.0), name="back")         # 2 cm gap
    d = design_from_meshes([a, b], "Pair", kind="generic")
    p = plan_simulation(d, make_sim(d, speed=5.0, quality="draft", device="cpu"))
    msg = [w for w in p.warnings if "gap between 'front' and 'back'" in w]
    assert msg and "closed" in msg[0]
    assert P3._FACTS_CACHE                                                 # measured once, then cached


def test_memory_check_refuses_grids_that_do_not_fit(monkeypatch):
    d = sphere_design()
    monkeypatch.setattr(devmod, "system_memory_gb", lambda: 0.05)
    with pytest.raises(ValueError, match="GB of memory"):
        plan_simulation(d, make_sim(d, speed=speed_for(100.0), device="cpu", cells_per_ref=40))


# ---------------------------------------------------------------------------
# Reynolds number, flow model, estimate
# ---------------------------------------------------------------------------

def test_reynolds_and_flow_model_follow_the_2d_rules():
    d = sphere_design()
    lam = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu"))
    assert lam.turbulence == "laminar" and lam.re_sim == pytest.approx(100.0) and lam.bl_resolved
    les = plan_simulation(d, make_sim(d, speed=30.0, quality="draft", device="cpu"))
    assert les.turbulence == "les-wale" and not les.bl_resolved and les.solver_config.turbulence == "wale"
    capped = plan_simulation(d, make_sim(d, speed=30.0, quality="draft", device="cpu", reynolds_mode="capped",
                                         reynolds_cap=200.0))
    assert capped.capped and capped.re_sim == pytest.approx(200.0) and capped.solver_config.re == pytest.approx(200.0)
    forced = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu", turbulence="les"))
    assert forced.turbulence == "les-wale"


def test_estimate_uses_3d_measurements_only():
    d = sphere_design()
    sim = make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu")
    before = plan_simulation(d, sim).est_seconds_headless
    est2d = planmod.estimate_steps_per_second("cpu", 1e5)
    P3.record_throughput3d("cpu", 130_000, 10 * P3._model_rate3("cpu", "upwind3", "float32", 130_000))
    after = plan_simulation(d, sim).est_seconds_headless
    assert after < before                                         # 10x faster measured -> shorter estimate
    assert planmod.estimate_steps_per_second("cpu", 1e5) == pytest.approx(est2d)   # 2D untouched
    assert planmod._RATE_CACHE == {} and planmod._RATE_CACHE3D


def test_sensitivity_map_time_is_in_the_estimate_and_its_budget_warned_at_plan_time(monkeypatch):
    """compute_sensitivity: the map's own time (optimize.sensitivity3d.estimate_seconds) is added to
    the estimate with a note; a grid above the map's cell budget is warned about before the run."""
    from aerolume.optimize import sensitivity3d as S3

    d = sphere_design()
    base = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu"))
    with_map = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu",
                                           compute_sensitivity=True))
    assert with_map.cells == base.cells and with_map.cells <= S3.DEFAULT_MAX_CELLS["cpu"]
    extra = S3.estimate_seconds(with_map)
    assert extra > 0 and with_map.est_seconds == pytest.approx(base.est_seconds + extra)
    assert with_map.est_seconds_headless == pytest.approx(base.est_seconds_headless + extra)
    assert any("Sensitivity map" in n for n in with_map.notes)
    assert not any("sensitivity map is limited" in w for w in with_map.warnings)
    # a budget below this grid: skipped by the runner, and said so now, without adding its time
    monkeypatch.setitem(S3.DEFAULT_MAX_CELLS, "cpu", base.cells - 1)
    over = plan_simulation(d, make_sim(d, speed=speed_for(100.0), quality="draft", device="cpu",
                                       compute_sensitivity=True))
    assert any("sensitivity map is limited" in w for w in over.warnings)
    assert over.est_seconds == pytest.approx(base.est_seconds)
    # explore plans never carry a map
    live = plan_live(d, make_sim(d, speed=speed_for(100.0), device="cpu", compute_sensitivity=True))
    assert not any("ensitivity" in t for t in live.notes + live.warnings)


def test_3d_calibration_persists_in_the_shared_file(tmp_path, monkeypatch):
    path = tmp_path / "cal.json"
    planmod.use_calibration_file(path)
    try:
        P3.record_throughput3d("cpu", 50_000, 40.0, kind="generic", speed_factor=2.0, cells_per_ref=20)
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["version"] == 1 and doc["rate3d"] and doc["speed3d"] and doc["rate"] == {}
        for name in ("_RATE_CACHE3D", "_SPEED_CACHE3D"):
            monkeypatch.setattr(planmod, name, {})
        planmod.use_calibration_file(path)
        assert ("3d", "cpu", "upwind3", "float32") in planmod._RATE_CACHE3D
    finally:
        planmod.use_calibration_file(None)


# ---------------------------------------------------------------------------
# live plans, wings, defaults
# ---------------------------------------------------------------------------

def test_live_plan_is_full_domain_and_sized_for_the_sliders():
    car = car_design()
    lp = plan_live(car, make_sim(car, speed=30.0, device="cpu"))
    cfg = lp.solver_config
    assert not lp.symmetry_plane and cfg.bc_y == "slip"
    assert cfg.window_yaw_deg == P3.LIVE_YAW3 and cfg.window_pitch_deg == P3.LIVE_PITCH_GROUND3
    assert lp.cells <= P3.LIVE_BUDGET3["cpu"] or lp.cells_per_ref <= 4
    # every yawed attitude fits inside the domain
    for yaw in (-P3.LIVE_YAW3, P3.LIVE_YAW3):
        bb = P3._bbox(wind_frame_meshes(lp.meshes_design, 0.0, yaw, lp.moment_ref3, ground=True))
        assert lp.extent3[2] < bb[2] and lp.extent3[3] > bb[3]


def test_wing_planform_is_lofted_and_the_wing_analysis_recommended():
    d = wing_design()
    p = plan_simulation(d, make_sim(d, speed=30.0, alpha=4.0, quality="draft", device="cpu"))
    assert p.geometry_class == "wing" and p.ref_label == "mean chord" and len(p.bodies_wind3) == 1
    assert p.symmetry_plane                                       # a symmetric planform at zero yaw
    assert any("Wing analysis" in w for w in p.warnings)
    with_project = wing_design("design:abc123")
    with pytest.raises(ValueError, match="needs the project"):
        plan_simulation(with_project, make_sim(with_project, speed=30.0, device="cpu"))
    af = naca("2412", 160, 1.0)
    p2 = plan_simulation(with_project, make_sim(with_project, speed=30.0, device="cpu"), airfoil_coords=lambda n: af)
    assert isinstance(p2, SimPlan3D) and len(p2.bodies_wind3[0].vertices) > 100


def test_defaults_for_3d_designs_and_wings():
    car = car_design()
    cfd = D.default_cfd_settings(car, FlowConditions(AIR, 30.0))
    assert cfd.quality == "draft" and cfd.reynolds_mode == "real" and cfd.symmetry == "auto"
    # a compact 3D body whose force reference equals its length is NOT a 2D wing on a road
    cube = design_from_meshes([box(1.0, 1.0, 1.0)], "Cube", kind="vehicle")
    assert D.geometry_class(cube) == "vehicle"
    assert D.default_cfd_settings(cube, FlowConditions(AIR, 50.0)).reynolds_mode == "real"
    assert D.default_reynolds_cap(cube) is None
    w = wing_design()
    assert D.geometry_class(w) == "wing" and D.default_analysis(w) == "wing" and D.default_analysis(car) == "cfd"
    sim = D.default_simulation(w, conditions=FlowConditions(AIR, 25.0))
    assert sim.analysis == "wing" and sim.wing.method == "vlm" and sim.wing.viscous and sim.cfd.quality == "draft"
    s_car = D.default_simulation(car)
    assert s_car.analysis == "cfd" and s_car.ground.enabled and s_car.ground.rotate_wheels
    # 2D unchanged: airfoils above Re 1e5 still start capped at 2,500
    af = design_from_airfoil(naca("2412", 160, 1.0), "NACA 2412", 1.0)
    c2 = D.default_cfd_settings(af, FlowConditions(AIR, 30.0))
    assert c2.reynolds_mode == "capped" and c2.reynolds_cap == D.AIRFOIL_DEFAULT_CAP_RE
    assert D.geometry_class(af) == "airfoil"


def test_focus_and_default_slices_lie_inside_the_domain():
    car = car_design()
    p = plan_simulation(car, make_sim(car, speed=30.0, quality="draft", device="cpu"))
    f = p.focus_extent3
    full = p.extent3_full
    for i in range(3):
        assert full[2 * i] - 1e-9 <= f[2 * i] < f[2 * i + 1] <= full[2 * i + 1] + 1e-9
    assert p.focus_extent == (f[0], f[1], f[4], f[5])
    assert p.slice_positions["y"] == 0.0 and full[0] < p.slice_positions["x"] < full[1]
    assert 0.0 < p.slice_positions["z"] < p.body_bounds[5]
    assert p.focus_window("x") == (f[2], f[3], f[4], f[5])
