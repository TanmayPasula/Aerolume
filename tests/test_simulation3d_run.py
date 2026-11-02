"""3D production runs and the 3D live session (DESIGN.md 7.7).

The fast tests run tiny grids on the CPU (a few seconds each): result structure,
SI units, per-vertex surface data on the user's mesh, mirrored slices and
volume, frames, cancel, the sensitivity guard and the explore session. The
``gpu`` tests are the end-to-end checks: the Draft sphere at Re 100 against the
solver's validated grid study, and a half-domain run against the full domain.
"""
from __future__ import annotations

import itertools
import math
import sys
import threading
import types

import numpy as np
import pytest
import torch

from aerolume.core import fluid as F
from aerolume.geometry3d import box, design_from_meshes, sphere
from aerolume.model import CFD3DResult, CFDResult, CFDSettings, FlowConditions, GroundConfig, Simulation
from aerolume.simulation import LiveSession, ViewSettings, run_cfd
from aerolume.simulation import plan as planmod
from aerolume.simulation import runner3d as R3
from aerolume.simulation.live3d import LiveSession3D, mesh_section
from aerolume.simulation.plan import plan_live, plan_simulation

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
AIR = F.air_isa(0.0)


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def speed_for(re: float, length: float = 1.0) -> float:
    return re * AIR.viscosity / (AIR.density * length)


def sphere_case(*, re: float = 50.0, device: str = "cpu", alpha: float = 0.0, **cfd):
    d = design_from_meshes([sphere(1.0, 24)], "Sphere", kind="generic")
    cfd.setdefault("cells_per_ref", 6)
    cfd.setdefault("duration", 2.5)
    cfd.setdefault("average_from", 1.0)
    sim = Simulation("s", d.id, conditions=FlowConditions(AIR, speed_for(re), alpha),
                     cfd=CFDSettings(quality=cfd.pop("quality", "draft"), device=device, **cfd))
    return d, sim, plan_simulation(d, sim)


@pytest.fixture(scope="module")
def sphere_run():
    """One tiny half-domain sphere run shared by the structure tests (with frames)."""
    for name in ("_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        setattr(planmod, name, {})
    d, sim, plan = sphere_case()
    frames: list = []
    progress: list = []
    res = run_cfd(plan, frame=frames.append, view=ViewSettings(fps=50.0),
                  progress=lambda f, m: progress.append((f, m)))
    for name in ("_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        setattr(planmod, name, {})
    return d, sim, plan, res, frames, progress


# ---------------------------------------------------------------------------
# result structure
# ---------------------------------------------------------------------------

def test_run_cfd_dispatches_to_the_3d_runner_and_returns_si_values(sphere_run):
    d, sim, plan, res, _, progress = sphere_run
    assert isinstance(res, CFD3DResult) and not isinstance(res, CFDResult) and res.sim_id == sim.id
    assert res.grid == plan.grid3 and res.cells == plan.cells and res.symmetry_plane
    assert set(res.history) == {"t", "cd", "cs", "cl", "c_roll", "cm", "cn"}
    assert np.all(np.diff(res.history["t"]) > 0) and res.history["t"][-1] == pytest.approx(res.t_end)
    U = sim.conditions.speed
    q = 0.5 * AIR.density * U * U
    A, L, b = d.ref_area, d.length_ref, d.span_ref
    assert res.drag == pytest.approx(res.cd.mean * q * A)
    assert res.lift == pytest.approx(res.cl.mean * q * A)
    assert res.side == 0.0 and res.cs.mean == 0.0              # zero by construction with a symmetry plane
    assert res.moments[1] == pytest.approx(res.cm.mean * q * A * L)
    assert res.moments[0] == pytest.approx(res.c_roll.mean * q * A * b)
    assert res.drag_area == pytest.approx(res.cd.mean * A) and res.power == pytest.approx(res.drag * U)
    assert 1.0 < res.cd.mean < 4.0                             # a very coarse sphere at Re 50, early in the run
    assert res.turbulence == plan.turbulence and res.precision == "float32"
    assert progress[-1][0] == 1.0 and progress[-1][1].startswith("Done")
    assert all(b >= a for (a, _), (b, _) in itertools.pairwise(progress))


def test_surface_is_per_vertex_on_the_users_mesh(sphere_run):
    d, _, _, res, _, _ = sphere_run
    assert len(res.surface) == 1
    s = res.surface[0]
    m = d.meshes[0]
    assert s.body == m.name
    # same vertices (alpha = 0: the wind frame is the design frame), same faces, the whole body
    assert np.allclose(s.vertices, m.vertices) and np.array_equal(s.faces, m.faces)
    assert s.cp.shape == (len(m.vertices),) and s.cf.shape == s.cp.shape and s.tau.shape == (len(m.vertices), 3)
    assert np.all(np.isfinite(s.cp)) and np.allclose(s.cf, np.linalg.norm(s.tau, axis=1))
    front = int(np.argmin(m.vertices[:, 0]))
    assert s.cp[front] > 0.5                                   # stagnation point
    # the vertices of the mirrored half were probed at their mirror image: symmetric values
    v = m.vertices
    for i in np.flatnonzero(v[:, 1] < -0.2)[:20]:
        j = int(np.argmin(np.linalg.norm(v - v[i] * [1, -1, 1], axis=1)))
        assert s.cp[i] == pytest.approx(s.cp[j], abs=0.02)      # the triangulation is not exactly mirrored


def test_pitched_body_surface_is_in_the_wind_frame():
    d, _, plan = sphere_case(alpha=30.0, duration=0.3, average_from=0.1)
    res = run_cfd(plan)
    s = res.surface[0]
    expect = plan.bodies_wind3[0].vertices
    assert np.allclose(s.vertices, expect) and not np.allclose(s.vertices, d.meshes[0].vertices)


def test_slices_are_mirrored_si_and_follow_the_slicedata_orientation(sphere_run):
    _, sim, plan, res, _, _ = sphere_run
    nx, ny, nz = plan.grid3
    by_axis = {s.axis: s for s in res.slices}
    assert set(by_axis) == {"x", "y", "z"} and all(s.label == "mean" for s in res.slices)
    assert set(by_axis["y"].fields) == set(R3.SLICE_FIELDS)
    assert by_axis["y"].fields["u"].shape == (nz, nx)                  # rows z, columns x
    assert by_axis["z"].fields["u"].shape == (2 * ny, nx)              # rows y (mirrored), columns x
    assert by_axis["x"].fields["u"].shape == (nz, 2 * ny)              # rows z, columns y (mirrored)
    x0, x1, _, y1, z0, z1 = plan.extent3
    assert by_axis["z"].extent == pytest.approx((x0, x1, -y1, y1))
    assert by_axis["x"].extent == pytest.approx((-y1, y1, z0, z1))
    assert by_axis["y"].extent == pytest.approx((x0, x1, z0, z1))
    zs = by_axis["z"]
    assert np.allclose(zs.fields["u"], zs.fields["u"][::-1])          # even across the plane
    assert np.allclose(zs.fields["v"], -zs.fields["v"][::-1])         # v is odd
    U = sim.conditions.speed
    assert np.nanmax(np.abs(zs.fields["u"])) == pytest.approx(U, rel=0.6)    # m/s, not units of U
    assert zs.mask.shape == zs.fields["u"].shape and zs.mask.max() > 0.9


def test_volume_is_mirrored_and_block_averaged_under_the_cap(sphere_run, monkeypatch):
    _, _, plan, res, _, _ = sphere_run
    v = res.volume
    # the mean fields, plus the instantaneous smoke when the result keeps it in the volume (no extra_volumes)
    assert v.mirrored and set(v.fields) - {"dye"} == set(R3.VOLUME_FIELDS) and v.mask is not None
    nx, ny, nz = plan.grid3
    assert v.fields["u"].shape == (nz, ny, nx) and v.origin[1] == 0.0
    h = plan.solver_config.h * plan.length_ref
    assert v.spacing == pytest.approx((h, h, h))
    # a smaller cap block-averages (whole cells, the far ends cropped)
    _, _, plan2 = sphere_case(duration=0.2, average_from=0.1)
    monkeypatch.setattr(R3, "VOLUME_MAX_CELLS", plan2.cells // 7)
    res2 = run_cfd(plan2)
    v2 = res2.volume
    f = round(v2.spacing[0] / (plan2.solver_config.h * plan2.length_ref))
    assert f == 2 and v2.fields["cp"].shape == (nz // 2, ny // 2, nx // 2)
    assert v2.fields["u"].size <= plan2.cells // 7
    assert any("block average" in n for n in res2.notes)


def test_frames_are_slice_images_in_the_livframe_contract(sphere_run):
    _, _, plan, _, frames, _ = sphere_run
    assert frames and frames[-1].phase == "done" and frames[-1].progress == 1.0
    fr = frames[-1]
    nx, _, nz = plan.grid3
    assert fr.rgba.shape == (nz, nx, 4) and fr.rgba.dtype == np.uint8 and fr.slice_axis == "y"
    assert fr.extent == pytest.approx((plan.extent3[0], plan.extent3[1], plan.extent3[4], plan.extent3[5]))
    assert set(fr.coeffs) >= {"cd", "cl", "cm", "cs", "c_roll", "cn"} and fr.units == "m/s"
    assert fr.bodies and all(b.shape[1] == 2 for b in fr.bodies)          # the sphere's cross-section
    assert set(fr.history) >= {"t", "cd", "cl"} and fr.focus is not None


def test_frames_follow_the_slice_axis_and_field():
    _, _, plan = sphere_case(duration=0.3, average_from=0.1)
    frames: list = []
    run_cfd(plan, frame=frames.append, view=ViewSettings(field="q", slice_axis="x", fps=50.0))
    fr = frames[-1]
    _, ny, nz = plan.grid3
    assert fr.slice_axis == "x" and fr.field == "q" and fr.rgba.shape == (nz, 2 * ny, 4) and fr.units == "1/s²"
    assert fr.slice_pos == pytest.approx(plan.slice_positions["x"])


def test_cancel_returns_a_partial_result_with_a_warning():
    _, _, plan = sphere_case(duration=50.0, average_from=40.0)
    stop = threading.Event()

    def prog(f, m):
        if f > 0:
            stop.set()
    res = run_cfd(plan, progress=prog, cancel=stop)
    assert isinstance(res, CFD3DResult) and res.t_end < 50.0 and not res.converged
    assert any(w.startswith("Stopped by the user") for w in res.warnings)
    assert res.surface and res.slices and res.volume is not None


def test_a_live_plan_is_refused_and_a_too_short_run_says_so():
    d, sim, _ = sphere_case()
    with pytest.raises(ValueError, match="explore"):
        run_cfd(plan_live(d, sim))
    _, _, plan = sphere_case(duration=0.4, average_from=0.2)
    res = run_cfd(plan)
    assert not res.converged and any("too few time steps" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# sensitivity guard
# ---------------------------------------------------------------------------

def test_sensitivity_skipped_with_a_warning_when_the_module_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "aerolume.optimize.sensitivity3d", None)     # import fails
    _, _, plan = sphere_case(duration=0.3, average_from=0.1, compute_sensitivity=True)
    res = run_cfd(plan)
    assert any("3D sensitivity module is not available" in w for w in res.warnings)
    assert res.surface and res.surface[0].sensitivity is None


def test_sensitivity_skipped_above_the_grid_limit(monkeypatch):
    called = []
    fake = types.ModuleType("aerolume.optimize.sensitivity3d")
    fake.compute_sensitivity3d = lambda *a, **k: called.append(1)
    fake.attach_sensitivity3d = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "aerolume.optimize.sensitivity3d", fake)
    monkeypatch.setattr(R3.P3, "sensitivity_max_cells", lambda device: 1000)
    _, _, plan = sphere_case(duration=0.3, average_from=0.1, compute_sensitivity=True)
    res = run_cfd(plan)
    assert not called and any("limited to grids" in w for w in res.warnings)


def test_sensitivity_is_computed_and_attached_through_the_contract(monkeypatch):
    seen = {}
    fake = types.ModuleType("aerolume.optimize.sensitivity3d")

    def compute(design, sim, *, objective="cd", plan=None, solver=None, progress=None, cancel=None):
        seen.update(objective=objective, plan=plan, solver=solver)
        progress(0.5, "reverse sweep")
        return [np.ones(len(m.vertices)) for m in plan.bodies_wind3]

    def attach(result, arrays):
        for s, a in zip(result.surface, arrays):
            s.sensitivity = a
        return result
    fake.compute_sensitivity3d, fake.attach_sensitivity3d = compute, attach
    monkeypatch.setitem(sys.modules, "aerolume.optimize.sensitivity3d", fake)
    _, _, plan = sphere_case(duration=0.3, average_from=0.1, compute_sensitivity=True, sensitivity_objective="cl")
    msgs = []
    res = run_cfd(plan, progress=lambda f, m: msgs.append(m))
    assert seen["objective"] == "cl" and seen["plan"] is plan and seen["solver"] is not None
    assert res.surface[0].sensitivity is not None and len(res.surface[0].sensitivity) == len(res.surface[0].cp)
    assert any("Sensitivity of Cl" in m for m in msgs)


def test_a_failing_sensitivity_map_keeps_the_forces(monkeypatch):
    fake = types.ModuleType("aerolume.optimize.sensitivity3d")

    def boom(*a, **k):
        raise RuntimeError("adjoint exploded")
    fake.compute_sensitivity3d, fake.attach_sensitivity3d = boom, lambda *a: None
    monkeypatch.setitem(sys.modules, "aerolume.optimize.sensitivity3d", fake)
    _, _, plan = sphere_case(duration=0.3, average_from=0.1, compute_sensitivity=True)
    res = run_cfd(plan)
    assert math.isfinite(res.cd.mean) and any("adjoint exploded" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# multi-part bodies, sections
# ---------------------------------------------------------------------------

def test_off_plane_parts_share_their_mirrored_force_half_and_half():
    body = box(1.0, 0.4, 0.3, center=(0.5, 0.0, 0.25), name="body")
    left = box(0.2, 0.1, 0.1, center=(0.3, 0.35, 0.1), name="pod L")
    right = box(0.2, 0.1, 0.1, center=(0.3, -0.35, 0.1), name="pod R")
    d = design_from_meshes([body, left, right], "Pods", kind="vehicle", clearance=0.05)
    assert d.symmetric
    sim = Simulation("p", d.id, conditions=FlowConditions(AIR, 2.0, 0.0), ground=GroundConfig(True, True, False),
                     cfd=CFDSettings(quality="draft", device="cpu", cells_per_ref=10, duration=0.4, average_from=0.2))
    plan = plan_simulation(d, sim)
    assert plan.symmetry_plane
    res = run_cfd(plan)
    pb = res.per_body
    assert set(pb) == {"body", "pod L", "pod R"}
    assert pb["pod L"] == pytest.approx(pb["pod R"]) and pb["pod L"]["cs"] == 0.0
    total = sum(v["cd"] for v in pb.values())
    assert total == pytest.approx(res.cd.mean, rel=0.2)        # per-part means sample the same window
    assert any("half and half" in n for n in res.notes)


def test_mesh_section_of_a_sphere_is_a_closed_circle():
    m = sphere(2.0, 48)
    lines = mesh_section([(m.vertices, m.faces)], "x", 0.3)
    assert len(lines) == 1
    pts = lines[0]
    assert np.allclose(pts[0], pts[-1])                        # closed loop
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert np.allclose(r, math.sqrt(1.0 - 0.3 ** 2), rtol=0.02)
    assert mesh_section([(m.vertices, m.faces)], "z", 5.0) == []


# ---------------------------------------------------------------------------
# explore mode
# ---------------------------------------------------------------------------

def test_live_session_dispatches_to_3d_and_follows_posts():
    d, sim, _ = sphere_case(re=100.0)
    sim.cfd.cells_per_ref = 5
    ls = LiveSession(d, sim, device="cpu")
    try:
        assert isinstance(ls, LiveSession3D) and isinstance(ls, LiveSession)
        assert not ls.plan.symmetry_plane                      # yaw needs both halves
        f0 = ls.tick(20)
        assert f0.phase == "live" and f0.slice_axis == "y" and f0.rgba.ndim == 3
        ls.post(alpha_deg=5.0, yaw_deg=8.0, slice_axis="z", field="cp")
        f1 = ls.tick(20)
        assert ls.alpha_deg == 5.0 and ls.yaw_deg == 8.0 and f1.slice_axis == "z" and f1.field == "cp"
        assert f1.step > f0.step and f1.units == ""
        ls.post(alpha_deg=90.0)                                # clamped to the planned range
        ls.tick(20)
        assert ls.alpha_deg == pytest.approx(ls.pitch_limit)
        ls.post(speed=2 * sim.conditions.speed)
        ls.tick(20)
        assert ls.solver.cfg.re == pytest.approx(200.0, rel=1e-6)
        ls.post(reset=True)
        f2 = ls.tick(20)
        assert f2.step <= 2 * 2000
        with pytest.raises(TypeError):
            ls.post(colour="red")
        with pytest.raises(ValueError):
            ls.post(slice_axis="w")
    finally:
        ls.close()
        ls.close()                                             # idempotent
    with pytest.raises(RuntimeError):
        ls.tick()


def test_live_session_on_a_road_switches_the_belt():
    body = box(1.0, 0.4, 0.3, center=(0.5, 0.0, 0.25), name="body")
    d = design_from_meshes([body], "Box", kind="vehicle", clearance=0.05)
    sim = Simulation("b", d.id, conditions=FlowConditions(AIR, 2.0, 0.0), ground=GroundConfig(True, True, False),
                     cfd=CFDSettings(quality="draft", device="cpu", cells_per_ref=8))
    ls = LiveSession(d, sim, device="cpu")
    try:
        ls.tick(20)
        assert ls.solver.cfg.bc_zmin == "moving"
        ls.post(ground_moving=False)
        ls.tick(20)
        assert ls.solver.cfg.bc_zmin == "noslip" and not ls.sim.ground.moving
        ls.post(alpha_deg=-3.0)                               # would push the body into the road
        ls.tick(20)
        assert ls.alpha_deg >= -3.0
    finally:
        ls.close()


# ---------------------------------------------------------------------------
# end-to-end on the GPU
# ---------------------------------------------------------------------------

@needs_cuda
@pytest.mark.gpu
@pytest.mark.slow
def test_draft_sphere_re100_matches_the_validated_grid_study():
    """Draft sphere at Re 100 through the planner (half domain, 20 cells/D, 2.9 % blockage).

    The solver's grid study at 2.85 % blockage gives Cd 1.172 (16 cells/D) and
    1.148 (24 cells/D) (docs/theory_ns3d.md 10); the confinement adds ~3-4 %
    over an unconfined sphere (1.087). Measured here: 1.156 in ~16 s.
    """
    d = design_from_meshes([sphere(1.0, 64)], "Sphere", kind="generic")
    sim = Simulation("s", d.id, conditions=FlowConditions(AIR, speed_for(100.0), 0.0), cfd=CFDSettings(quality="draft"))
    plan = plan_simulation(d, sim, device="cuda")
    assert plan.symmetry_plane and plan.turbulence == "laminar"
    res = run_cfd(plan)
    assert res.converged and 1.13 <= res.cd.mean <= 1.18
    assert abs(res.cl.mean) < 1e-3 and res.wall_seconds < 120


@needs_cuda
@pytest.mark.gpu
@pytest.mark.slow
def test_symmetry_plane_run_equals_the_full_domain_run():
    """DESIGN 7.12: the half-domain run equals the full-domain one (same grid) within 1 %."""
    d = design_from_meshes([sphere(1.0, 48)], "Sphere", kind="generic")
    out = {}
    for sym in ("auto", "full"):
        sim = Simulation("s", d.id, conditions=FlowConditions(AIR, speed_for(100.0), 0.0),
                         cfd=CFDSettings(quality="draft", cells_per_ref=12, duration=20.0, average_from=12.0,
                                         symmetry=sym))
        plan = plan_simulation(d, sim, device="cuda")
        out[sym] = (plan, run_cfd(plan))
    (ph, rh), (pf, rf) = out["auto"], out["full"]
    assert ph.symmetry_plane and not pf.symmetry_plane and pf.cells == 2 * ph.cells
    assert rh.cd.mean == pytest.approx(rf.cd.mean, rel=0.01)
    assert rf.cs.mean == pytest.approx(0.0, abs=1e-3)
