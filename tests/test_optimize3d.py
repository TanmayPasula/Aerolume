"""3D shape sensitivity (optimize.sensitivity3d): mesh helpers, autograd vs finite differences,
symmetry-plane handling, the driver, the budget guard and attaching maps to results.

Fast tests run a tiny float64 sphere on the CPU (a few seconds each). The verification cases of
docs/theory_sensitivity.md section 9 (sphere and box on the GPU) are in
tests/test_optimize3d_validation.py (``-m slow``).
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aerolume.model import CFD3DResult, Mesh3D, SurfaceData3D
from aerolume.optimize import Cancelled
from aerolume.optimize.sensitivity3d import (
    MAGNITUDE_WARNING,
    SensitivityBudgetError,
    SensitivityMaps3D,
    WindowedObjective3D,
    attach_sensitivity3d,
    compute_sensitivity3d,
    estimate_memory_bytes,
    estimate_seconds,
    finite_difference_check3d,
    normal_modes3d,
    refine_mesh,
    surface_smoother,
    vertex_frame,
    visible_dual_area,
)
from aerolume.solvers.ns3d import FlowSolver3D, ImmersedMesh, SolverConfig3D, make_operators3d

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")


# ---------------------------------------------------------------------------
# meshes and set-ups
# ---------------------------------------------------------------------------

def icosphere(d: float = 1.0, level: int = 2, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic sphere (outward winding), mirror-symmetric about every coordinate plane."""
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.asarray(p, dtype=np.float64) / np.linalg.norm(p) for p in v]
    faces = list(f)
    for _ in range(level):
        cache: dict = {}

        def mid(a, b, cache=cache):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]
        faces = [g for a, b, c in faces for g in ((a, mid(a, b), mid(c, a)), (b, mid(b, c), mid(a, b)),
                                                  (c, mid(c, a), mid(b, c)), (mid(a, b), mid(b, c), mid(c, a)))]
    return np.asarray(verts) * 0.5 * d + np.asarray(center, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def box_mesh(lx=1.0, ly=1.0, lz=1.0) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    v = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]])
    return v, f


def signed_volume(v, f) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def total_area(v, f) -> float:
    return float(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum())


SPHERE = icosphere(1.0, 2)


def tiny_cfg(half: bool = False, *, mesh=SPHERE, **kw) -> SolverConfig3D:
    """A 5-cells-per-diameter sphere at Re 20 (float64, CPU): the solver's own gradient-check scale."""
    v, f = mesh
    base = {"nx": 28, "nz": 14, "h": 1 / 5, "x0": -2.0, "z0": -1.4, "re": 20.0,
            "bodies": [ImmersedMesh(v, f, name="ball")], "ref_area": math.pi / 4, "device": "cpu",
            "dtype": "float64", "perturb": False}
    base.update(kw)
    if half:
        return SolverConfig3D(ny=7, y0=0.0, bc_y="symmetry_ymin", **base)
    return SolverConfig3D(ny=14, y0=-1.4, **base)


def developed(cfg: SolverConfig3D, steps: int = 40) -> FlowSolver3D:
    s = FlowSolver3D(cfg)
    s.advance(steps)
    return s


@pytest.fixture(scope="module")
def full_case():
    cfg = tiny_cfg(False)
    s = developed(cfg)
    prob = WindowedObjective3D(cfg, s, objective="cd", lead=0.2, average=0.2)
    val, grad = prob.gradient()
    return cfg, s, prob, val, grad


@pytest.fixture(scope="module")
def half_case():
    cfg = tiny_cfg(True)
    s = developed(cfg)
    prob = WindowedObjective3D(cfg, s, objective="cd", lead=0.2, average=0.2)
    val, grad = prob.gradient()
    return cfg, s, prob, val, grad


MODE_CENTRES = [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.5, 0.0)]


# ---------------------------------------------------------------------------
# mesh helpers
# ---------------------------------------------------------------------------

def test_refine_mesh_keeps_the_shape_the_original_vertices_and_closure():
    v, f = box_mesh(1.0, 0.6, 0.4)
    rv, rf = refine_mesh(v, f, 0.15)
    assert np.array_equal(rv[:len(v)], v)                         # user's vertices keep their indices
    e = np.concatenate([rf[:, [0, 1]], rf[:, [1, 2]], rf[:, [2, 0]]])
    assert np.linalg.norm(rv[e[:, 1]] - rv[e[:, 0]], axis=1).max() <= 0.15 + 1e-12
    # every new vertex lies on the original surface: one coordinate at its face's half-extent
    on = (np.isclose(np.abs(rv[:, 0]), 0.5) | np.isclose(np.abs(rv[:, 1]), 0.3) | np.isclose(np.abs(rv[:, 2]), 0.2))
    assert on.all()
    assert math.isclose(total_area(rv, rf), total_area(v, f), rel_tol=1e-12)
    assert math.isclose(signed_volume(rv, rf), 1.0 * 0.6 * 0.4, rel_tol=1e-12)      # winding kept
    _, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)                                    # still closed and conforming


def test_refine_mesh_on_a_sphere_is_conforming_and_exact():
    v, f = icosphere(1.0, 1)
    rv, rf = refine_mesh(v, f, 0.05)
    e = np.sort(np.concatenate([rf[:, [0, 1]], rf[:, [1, 2]], rf[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert np.all(counts == 2)
    assert math.isclose(signed_volume(rv, rf), signed_volume(v, f), rel_tol=1e-12)
    assert len(rv) > 10 * len(v)


def test_vertex_frame_areas_sum_and_normals_are_radial():
    v, f = icosphere(1.0, 3)
    n, dual = vertex_frame(v, f)
    assert math.isclose(dual.sum(), total_area(v, f), rel_tol=1e-12)
    radial = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert np.min(np.sum(n * radial, axis=1)) > 0.999


def test_surface_smoother_is_a_kernel_ratio():
    """S^k (g A) / S^k A recovers a smooth density, and a constant one exactly."""
    v, f = icosphere(1.0, 3)
    _, dual = vertex_frame(v, f)
    sm = surface_smoother(v, f, 0.08)
    assert 1 <= sm.iterations <= 400
    const = sm(3.0 * dual) / sm(dual)
    assert np.allclose(const, 3.0, rtol=1e-12)
    g = 1.0 + 0.5 * v[:, 0] / 0.5                                  # varies on the body scale
    rec = sm(g * dual) / sm(dual)
    assert np.max(np.abs(rec - g)) < 0.05
    noisy = g * (1.0 + 0.5 * np.random.default_rng(1).standard_normal(len(g)))
    assert np.std(sm(noisy * dual) / sm(dual) - g) < 0.5 * np.std(noisy - g)   # it does smooth


def test_surface_smoother_links_repeated_vertices():
    """A triangle soup (every face with its own corners) still smooths across faces."""
    v, f = icosphere(1.0, 1)
    soup_v = v[f].reshape(-1, 3)
    soup_f = np.arange(len(soup_v)).reshape(-1, 3)
    x = np.zeros(len(soup_v))
    x[0] = 1.0
    y = surface_smoother(soup_v, soup_f, 0.3)(x)
    assert np.count_nonzero(y > 1e-6) > 30                        # spread well beyond the first triangle


def test_visible_dual_area_with_a_symmetry_plane():
    cfg_h, cfg_f = tiny_cfg(True), tiny_cfg(False)
    v, f = refine_mesh(*SPHERE, 0.1)
    _, dual = vertex_frame(v, f)
    vis_f = visible_dual_area(v, f, make_operators3d(cfg_f))
    vis_h = visible_dual_area(v, f, make_operators3d(cfg_h))
    assert np.allclose(vis_f, dual, rtol=1e-12)
    assert math.isclose(vis_h.sum(), 0.5 * dual.sum(), rel_tol=2e-3)
    assert np.all(vis_h[v[:, 1] > 0.1] == pytest.approx(dual[v[:, 1] > 0.1]))
    assert np.all(vis_h[v[:, 1] < -0.1] == 0.0)


# ---------------------------------------------------------------------------
# autograd vs finite differences (the discrete windowed objective)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["full_case", "half_case"])
def test_gradient_matches_central_differences(case, request):
    _, _, prob, _, grad = request.getfixturevalue(case)
    modes = normal_modes3d(prob, MODE_CENTRES, 0.25, mirror_y=0.0)
    modes = np.vstack([modes, np.ones(len(prob.surfaces[0].vertices))])
    res = finite_difference_check3d(prob, modes, eps=1e-5, vertex_grad=grad)
    for r in res:
        assert abs(r["fd"]) > 1e-2
        assert r["rel_err"] < 1e-6, r


def test_half_domain_equals_full_domain(full_case, half_case):
    """Same J and the same derivative along every mirror-symmetric mode; the map agrees too."""
    _, _, pf, vf, gf = full_case
    _, _, ph, vh, gh = half_case
    assert vh.value == pytest.approx(vf.value, rel=1e-10)
    modes = normal_modes3d(pf, MODE_CENTRES, 0.25, mirror_y=0.0)
    assert np.allclose(modes @ gh[0], modes @ gf[0], rtol=1e-9)
    df, dh = pf.density(gf)[0], ph.density(gh)[0]
    big = np.abs(df) > 0.3 * np.abs(df).max()
    assert np.max(np.abs(dh[big] - df[big]) / np.abs(df[big])) < 1e-3          # measured 1.4e-5
    # vertices on the missing half carry their mirror image's value exactly
    s = ph.surfaces[0]
    low = s.mirror >= 0
    assert low.sum() > 0 and np.array_equal(dh[low], dh[s.mirror[low]])
    mirrored = s.vertices[s.mirror[low]] * np.array([1.0, -1.0, 1.0])
    assert np.allclose(mirrored, s.vertices[low], atol=1e-9)


def test_drag_map_pattern_on_the_sphere(full_case):
    """Front stagnation region below the equator band, top/bottom and side bands alike."""
    _, _, prob, _, grad = full_case
    d = prob.density(grad)[0]
    x = prob.surfaces[0].vertices

    def at(p):
        return d[np.argmin(np.sum((x - np.asarray(p)) ** 2, axis=1))]
    eq = [at(p) for p in ((0, 0, 0.5), (0, 0, -0.5), (0, 0.5, 0), (0, -0.5, 0))]
    assert min(eq) > 0 and (max(eq) - min(eq)) < 0.1 * np.mean(eq)
    assert at((-0.5, 0, 0)) < 0.7 * np.mean(eq)


@pytest.mark.parametrize("objective", ["cl", "cm", "ld"])
def test_other_objectives_match_central_differences(objective):
    """A pitched box: non-zero lift and pitching moment, sharp edges."""
    v, f = box_mesh(0.6, 1.0, 0.8)
    a = math.radians(10.0)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    v = v @ rot.T + np.array([0.0137, 0.0071, 0.0093]) / 5       # faces off the grid planes (see the docs)
    cfg = tiny_cfg(False, mesh=(v, f))
    prob = WindowedObjective3D(cfg, developed(cfg, 30), objective=objective, lead=0.2, average=0.2)
    val, grad = prob.gradient()
    assert math.isfinite(val.value)
    modes = normal_modes3d(prob, [(-0.3, 0.0, 0.4), (0.0, 0.0, -0.4), (0.3, 0.3, 0.0)], 0.25)
    for r in finite_difference_check3d(prob, modes, eps=1e-5, vertex_grad=grad):
        assert r["rel_err"] < 1e-5, r


def test_two_bodies_with_a_spinning_wheel():
    """Per-body derivatives (each body's modes) and a rotating body's solid velocity are differentiated."""
    # both bodies sit off the grid lattice: centred on a cell centre, the faceted sphere put a grid
    # point within the voxeliser's ray jitter of its surface (2.2e-4 on the wheel's top mode)
    wc = (0.9137, 0.0071, 0.0093)
    sv, sf = icosphere(0.8, 2, center=wc)
    bv, bf = box_mesh(0.6, 0.8, 0.6)
    bv = bv + np.array([-0.7137, 0.0071, 0.0193])
    base = tiny_cfg(False)
    cfg = dataclasses.replace(base, bodies=[ImmersedMesh(bv, bf, name="block"),
                                            ImmersedMesh(sv, sf, name="wheel", role="wheel", omega=-2.5,
                                                         center=wc, axis=(0.0, 1.0, 0.0))])
    prob = WindowedObjective3D(cfg, developed(cfg, 30), objective="cl", lead=0.2, average=0.2)
    val, grad = prob.gradient()
    assert len(grad) == 2 and [len(g) for g in grad] == [len(s.vertices) for s in prob.surfaces]
    assert abs(val.coefficients[2]) > 1e-3                         # the spin makes lift
    for body, centre in ((0, (-1.0, 0.0, 0.3)), (1, (0.9, 0.0, 0.4)), (1, (0.5, 0.0, 0.0))):  # top, front
        modes = normal_modes3d(prob, [centre], 0.25, body=body)
        (r,) = finite_difference_check3d(prob, modes, eps=1e-5, body=body, vertex_grad=grad)
        assert abs(r["fd"]) > 1e-3 and r["rel_err"] < 1e-5, (body, r)
    maps = prob.density(grad)
    assert len(maps) == 2 and all(np.all(np.isfinite(m)) for m in maps)


def test_orientation_ignores_the_mesh_winding():
    """Randomly flipped faces: every face is turned towards the fluid, normals as for the clean mesh."""
    v, f = SPHERE
    rng = np.random.default_rng(7)
    bad = f.copy()
    flip = rng.random(len(f)) < 0.4
    bad[flip] = bad[flip][:, ::-1]
    cfg_ok, cfg_bad = tiny_cfg(False), tiny_cfg(False, mesh=(v, bad))
    s = developed(cfg_ok, 5)
    ok = WindowedObjective3D(cfg_ok, s, lead=0.0, average=0.05)
    worse = WindowedObjective3D(cfg_bad, s, lead=0.0, average=0.05)
    assert ok.surfaces[0].flipped_faces == 0 and worse.surfaces[0].flipped_faces > 0
    assert np.min(np.sum(ok.surfaces[0].normals * worse.surfaces[0].normals, axis=1)) > 0.999
    radial = ok.surfaces[0].vertices / np.linalg.norm(ok.surfaces[0].vertices, axis=1, keepdims=True)
    assert np.min(np.sum(ok.surfaces[0].normals * radial, axis=1)) > 0.99


def test_window_rejects_bad_input(full_case):
    cfg, s, *_ = full_case
    with pytest.raises(ValueError, match="objective"):
        WindowedObjective3D(cfg, s, objective="drag")
    with pytest.raises(ValueError, match="lead"):
        WindowedObjective3D(cfg, s, lead=-1.0)
    with pytest.raises(ValueError, match="grid"):
        WindowedObjective3D(cfg, (s.state.u[:, :, :-1], s.state.v, s.state.w))
    with pytest.raises(ValueError, match="no body"):
        WindowedObjective3D(dataclasses.replace(cfg, bodies=[]), s)


def test_cancel_raises(full_case):
    cfg, s, *_ = full_case
    prob = WindowedObjective3D(cfg, s, lead=0.2, average=0.2)
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        prob.gradient(cancel=ev)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

def fake_plan(cfg: SolverConfig3D, length: float = 2.0, *, bodies_wind=True) -> SimpleNamespace:
    """The members of a SimPlan3D that compute_sensitivity3d reads (metres = solver units * L)."""
    wind = [Mesh3D(b.name, b.vertices_numpy() * length, b.faces) for b in cfg.bodies] if bodies_wind else []
    return SimpleNamespace(solver_config=cfg, length_ref=length, strouhal_length=length, t_avg_start=0.4,
                           dt=0.05, device="cpu", bodies_wind3=wind, design=None, sim=None)


def test_compute_sensitivity3d_end_to_end(half_case):
    cfg, s, prob, _, grad = half_case
    plan = fake_plan(cfg, 2.0)
    msgs = []
    maps = compute_sensitivity3d(None, None, plan=plan, solver=s, objective="cd", window=(0.2, 0.2),
                                 progress=lambda f, m: msgs.append(f))
    assert isinstance(maps, SensitivityMaps3D) and len(maps) == 1
    nv = len(cfg.bodies[0].vertices)
    assert maps[0].shape == (nv,) and np.all(np.isfinite(maps[0]))
    # 1/m: the non-dimensional density divided by L_ref, at the body's own vertices
    assert np.allclose(maps[0], prob.density(grad)[0][:nv] / 2.0, rtol=1e-9)
    assert maps.symmetry and maps.symmetry_y == 0.0 and maps.objective == "cd"
    assert maps.steps == prob.steps and maps.window == pytest.approx((prob.lead, prob.average))
    assert set(maps.coefficients) == {"cd", "cs", "cl", "c_roll", "cm", "cn"}
    assert maps.value == pytest.approx(maps.coefficients["cd"])
    assert np.allclose(maps.points[0], cfg.bodies[0].vertices_numpy() * 2.0)
    assert MAGNITUDE_WARNING in maps.warnings and maps.notes[0] == maps.summary()
    assert "3D shape sensitivity of CD" in maps.summary()
    assert msgs and msgs[-1] == 1.0 and all(b >= a for a, b in itertools.pairwise(msgs))
    assert maps.peak_memory_bytes is None and maps.estimated_memory_bytes > 0


def test_compute_sensitivity3d_develops_the_flow_itself():
    cfg = tiny_cfg(True)
    plan = fake_plan(cfg, 1.0, bodies_wind=False)
    plan.t_avg_start = 0.3
    maps = compute_sensitivity3d(None, None, plan=plan, window=(0.1, 0.1), objective="cd")
    assert len(maps) == 1 and np.all(np.isfinite(maps[0]))
    assert np.allclose(maps.points[0], cfg.bodies[0].vertices_numpy())
    assert maps.period is None and maps.ensemble == 1


def test_budget_refusal_happens_before_any_work():
    cfg = tiny_cfg(True)
    with pytest.raises(SensitivityBudgetError, match="M cells") as err:
        compute_sensitivity3d(None, None, plan=fake_plan(cfg), max_cells=1000)
    assert isinstance(err.value, ValueError) and "Draft" in str(err.value)


def test_memory_refusal_names_the_shortfall(monkeypatch, half_case):
    """Too little free device memory even with the checkpoints on the host: a readable refusal."""
    import aerolume.optimize.sensitivity3d as s3d
    cfg, s, *_ = half_case
    monkeypatch.setattr(s3d, "_available_bytes", lambda device: 64 * 2**20)
    with pytest.raises(SensitivityBudgetError, match=r"GB of GPU memory .* is free"):
        compute_sensitivity3d(None, None, plan=fake_plan(cfg), solver=s, objective="cd", window=(0.1, 0.1))


def test_2d_plans_are_rejected():
    plan = SimpleNamespace(solver_config=object(), length_ref=1.0)
    with pytest.raises(TypeError, match="3D plan"):
        compute_sensitivity3d(None, None, plan=plan)


def test_memory_and_time_estimates():
    a = estimate_memory_bytes(1_000_000, steps=1000)
    assert estimate_memory_bytes(2_000_000, steps=1000) > a > estimate_memory_bytes(500_000, steps=1000)
    assert estimate_memory_bytes(1_000_000, steps=1000, offload=True) < a
    assert estimate_memory_bytes(1_000_000, steps=1000, dtype="float64") > 1.9 * (a - 2**28)
    cfg = tiny_cfg(True)
    plan = fake_plan(cfg)
    assert 0 < estimate_seconds(plan) < 60
    big = fake_plan(dataclasses.replace(cfg, nx=200, ny=100, nz=100))   # a config is cheap: no tensors
    gpu = fake_plan(dataclasses.replace(cfg, nx=200, ny=100, nz=100))
    gpu.device = "cuda"
    assert estimate_seconds(gpu) < estimate_seconds(big)
    assert estimate_seconds(big) > 10 * estimate_seconds(plan) - 50


# ---------------------------------------------------------------------------
# attaching to a result
# ---------------------------------------------------------------------------

def _surface(name, v, f):
    return SurfaceData3D(body=name, vertices=v, faces=f, cp=np.zeros(len(v)))


def test_attach_by_name_and_same_length():
    v, f = SPHERE
    res = CFD3DResult(sim_id="s", surface=[_surface("other", v, f), _surface("ball", v, f)],
                      notes=["kept"], warnings=[MAGNITUDE_WARNING])
    vals = np.arange(len(v), dtype=float)
    maps = SensitivityMaps3D([vals], bodies=["ball"], points=[v.copy()], notes=["a note"],
                             warnings=[MAGNITUDE_WARNING, "w2"])
    attach_sensitivity3d(res, maps)
    assert res.surface[0].sensitivity is None
    assert np.array_equal(res.surface[1].sensitivity, vals)
    assert res.notes == ["kept", "a note"] and res.warnings == [MAGNITUDE_WARNING, "w2"]
    attach_sensitivity3d(res, maps)                               # idempotent: no duplicated lines
    assert res.notes == ["kept", "a note"]


def test_attach_maps_a_half_mesh_onto_the_mirrored_full_surface():
    """A map on the y >= 0 half of a body fills the full (mirrored) result surface by mirror lookup."""
    v, f = icosphere(1.0, 2, center=(1.0, 0.0, 0.3))
    keep = v[:, 1] >= -1e-12
    half = v[keep]
    vals = half[:, 0] + 3.0 * half[:, 2] + 0.1 * half[:, 1]       # depends on |y| through the mirror
    maps = SensitivityMaps3D([vals], bodies=["b"], points=[half], symmetry=True, symmetry_y=0.0)
    res = CFD3DResult(sim_id="s", surface=[_surface("b", v, f)])
    attach_sensitivity3d(res, maps)
    sens = res.surface[0].sensitivity
    assert sens.shape == (len(v),)
    expect = v[:, 0] + 3.0 * v[:, 2] + 0.1 * np.abs(v[:, 1])
    assert np.allclose(sens, expect, atol=1e-12)


def test_attach_resamples_by_nearest_vertex_and_by_order_for_plain_lists():
    v, f = icosphere(1.0, 3)
    rv, rf = refine_mesh(v, f, 0.03)                               # a finer surface than the map
    maps = SensitivityMaps3D([v[:, 2].copy()], bodies=["b"], points=[v])
    res = CFD3DResult(sim_id="s", surface=[_surface("b", rv, rf)])
    attach_sensitivity3d(res, maps)
    assert np.max(np.abs(res.surface[0].sensitivity - rv[:, 2])) < 0.05
    res2 = CFD3DResult(sim_id="s", surface=[_surface("x", v, f)])
    attach_sensitivity3d(res2, [v[:, 0]])                          # plain list: by order, same length
    assert np.array_equal(res2.surface[0].sensitivity, v[:, 0])


def test_package_exports_are_lazy():
    import aerolume.optimize as opt
    assert opt.compute_sensitivity3d is compute_sensitivity3d
    assert opt.attach_sensitivity3d is attach_sensitivity3d
    assert opt.SensitivityBudgetError is SensitivityBudgetError


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------

@needs_cuda
@pytest.mark.gpu
def test_cuda_graphs_match_eager():
    cfg = tiny_cfg(True, device="cuda")
    s = developed(cfg, 20)
    g = WindowedObjective3D(cfg, s, lead=0.1, average=0.1)
    e = WindowedObjective3D(cfg, s, lead=0.1, average=0.1, use_graphs=False)
    vg, gg = g.gradient()
    ve, ge = e.gradient()
    assert g.cuda_graphs and not e.cuda_graphs
    assert vg.value == pytest.approx(ve.value, rel=1e-12)
    assert np.allclose(gg[0], ge[0], rtol=1e-9, atol=1e-12 * np.abs(ge[0]).max())
