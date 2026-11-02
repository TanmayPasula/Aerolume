"""3D sensitivity verification (DESIGN 7.12, docs/theory_sensitivity.md section 9): slow, GPU.

* Sphere, Re 30, 12 cells per diameter, half domain, float64: autograd against central
  differences of the same windowed objective along smooth displacement modes (front, rear, top,
  bottom, side, shoulders, aft, uniform dilation). Measured agreement 1e-5; the bar is the sign on
  every mode and 15 % on the dominant ones, checked here at 1e-3. The drag map must look
  axisymmetric: the top and side bands agree to 10 % (measured 7 %), top ~ bottom, +y = -y.
* Short box (0.5 x 1 x 1, pitched 8 degrees), Re 30, 12 cells per height, float64, CD and CL,
  with modes on faces, edges and a corner. Its side faces lie exactly on grid planes, where a
  signed distance taken as +-sqrt(d^2) loses its derivative; the voxeliser's distance is
  differentiable through the surface (the docs explain it), so every mode is checked at 3e-3:
  measured CD <= 2.1e-5 and CL <= 6.2e-4 (1.5 % and 14 % before that fix).
* The driver end to end at 24 cells per diameter (1.56 M cells, float32, the default 6 + 2
  window): peak memory below the estimate, the map's pattern, the wall time.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_optimize3d import box_mesh, icosphere

from aerolume.optimize.sensitivity3d import (
    WindowedObjective3D,
    compute_sensitivity3d,
    finite_difference_check3d,
    normal_modes3d,
)
from aerolume.solvers.ns3d import FlowSolver3D, ImmersedMesh, SolverConfig3D

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def sphere_cfg(cpd: int, *, level: int, dtype: str, re: float = 30.0) -> SolverConfig3D:
    """Sphere of diameter 1 at the origin, box [-3, 6] x [0, 2.5] x [-2.5, 2.5] (half domain)."""
    h = 1.0 / cpd
    nx, nw = round(9.0 / h), round(2.5 / h)
    v, f = icosphere(1.0, level)
    return SolverConfig3D(nx=nx, ny=nw, nz=2 * nw, h=h, x0=-3.0, y0=0.0, z0=-nw * h, re=re,
                          bc_y="symmetry_ymin", bodies=[ImmersedMesh(v, f, name="sphere")],
                          ref_area=math.pi / 4, length_ref=1.0, span_ref=1.0, dtype=dtype, perturb=False)


def develop(cfg: SolverConfig3D, t_end: float) -> FlowSolver3D:
    s = FlowSolver3D(cfg)
    while s.t < t_end:
        s.advance(500)
    return s


@pytest.fixture(scope="module")
def sphere_problem():
    cfg = sphere_cfg(12, level=3, dtype="float64")
    s = develop(cfg, 12.0)
    prob = WindowedObjective3D(cfg, s, objective="cd", lead=1.0, average=1.0)
    del s
    torch.cuda.empty_cache()
    val, grad = prob.gradient()
    yield prob, val, grad
    del prob
    torch.cuda.empty_cache()


def test_sphere_autograd_matches_finite_differences(sphere_problem):
    prob, _, grad = sphere_problem
    r, c = 0.5, 0.5 / math.sqrt(2.0)
    centres = [(-r, 0, 0), (r, 0, 0), (0, 0, r), (0, 0, -r), (0, r, 0), (-c, 0, c), (-c, c, 0), (c, 0, c)]
    modes = np.vstack([normal_modes3d(prob, centres, 0.15, mirror_y=0.0), np.ones(len(prob.surfaces[0].vertices))])
    res = finite_difference_check3d(prob, modes, eps=1e-4, vertex_grad=grad)
    for rr in res:
        assert np.sign(rr["ad"]) == np.sign(rr["fd"]), rr
        assert rr["rel_err"] < 1e-3, rr                           # measured <= 1.3e-5


def test_sphere_drag_map_is_axisymmetric(sphere_problem):
    prob, _, grad = sphere_problem
    d = prob.density(grad)[0]
    x = prob.surfaces[0].vertices
    th = np.degrees(np.arccos(np.clip(-x[:, 0] / np.linalg.norm(x, axis=1), -1, 1)))
    ph = np.degrees(np.arctan2(x[:, 1], x[:, 2]))
    peak = np.abs(d).max()
    for lo in range(0, 180, 15):
        band = (th >= lo) & (th < lo + 15)
        means = []
        for centre in (0, 90, 180, -90):
            m = band & (np.abs((ph - centre + 180) % 360 - 180) < 20)
            means.append(d[m].mean())
        assert means[1] == pytest.approx(means[3], rel=1e-9)        # -y carries the mirror image's values
        if abs(np.mean(means)) > 0.3 * peak:
            # measured <= 7.3 %: the facets meet the flow differently at +-y and +-z (docs 9.4)
            assert np.ptp(means) < 0.10 * abs(np.mean(means)), (lo, means)
            # top and bottom: the same to ~1e-4 here (1.3 % in the small front band, where mirrored
            # faces got different bisection diagonals)
            assert means[0] == pytest.approx(means[2], rel=1e-2), (lo, means)
    # the pattern: negative near the front stagnation point, peak around the equator, small at the rear
    front = d[th < 15].mean()
    eq = d[(th > 75) & (th < 105)].mean()
    rear = d[th > 165].mean()
    assert front < 0 < eq and abs(rear) < 0.05 * eq and eq > 0.8 * peak


@pytest.fixture(scope="module")
def box_case():
    lx, ly, lz = 0.5, 1.0, 1.0
    v, f = box_mesh(lx, ly, lz)
    a = math.radians(8.0)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    h = 1.0 / 12
    nx, nw = round(9.0 / h), round(2.5 / h)
    cfg = SolverConfig3D(nx=nx, ny=nw, nz=2 * nw, h=h, x0=-3.0, y0=0.0, z0=-nw * h, re=30.0,
                         bc_y="symmetry_ymin", bodies=[ImmersedMesh(v @ rot.T, f, name="box")], ref_area=ly * lz,
                         dtype="float64", perturb=False)
    s = develop(cfg, 12.0)
    pts = [(-lx / 2, 0, 0), (lx / 2, 0, 0), (0, 0, lz / 2), (0, 0, -lz / 2), (0, ly / 2, 0), (-lx / 2, 0, lz / 2),
           (-lx / 2, 0, -lz / 2), (-lx / 2, ly / 2, 0), (-lx / 2, ly / 2, lz / 2), (lx / 2, 0, lz / 2)]
    yield cfg, s, [tuple(rot @ np.asarray(p)) for p in pts]
    del s
    torch.cuda.empty_cache()


@pytest.mark.parametrize("objective", ["cd", "cl"])
def test_box_autograd_matches_finite_differences(box_case, objective):
    cfg, s, centres = box_case
    prob = WindowedObjective3D(cfg, s, objective=objective, lead=1.0, average=1.0)
    _, grad = prob.gradient()
    modes = np.vstack([normal_modes3d(prob, centres, 0.15, mirror_y=0.0), np.ones(len(prob.surfaces[0].vertices))])
    res = finite_difference_check3d(prob, modes, eps=1e-4, vertex_grad=grad)
    for rr in res:
        assert np.sign(rr["ad"]) == np.sign(rr["fd"]), rr
        # every mode, including those on the side faces, which lie exactly on a plane of grid faces
        # (y = +-0.5 D at 12 cells/D): the signed distance is differentiable through the surface
        # (docs/theory_sensitivity.md 9.4). Measured CD <= 2.1e-5, CL <= 6.2e-4 (before the
        # voxeliser fix: 1.5 % and 14 %).
        assert rr["rel_err"] < 3e-3, rr


def test_driver_at_draft_size_fits_and_keeps_the_pattern():
    """1.56 M cells, float32, default window: measured 2 min, ~1.5 GB extra on the development GPU."""
    cfg = sphere_cfg(24, level=4, dtype="float32")
    s = develop(cfg, 10.0)
    plan = SimpleNamespace(solver_config=cfg, length_ref=1.0, strouhal_length=1.0, t_avg_start=6.0, dt=s.dt,
                           device="cuda", bodies_wind3=[])
    t0 = time.perf_counter()
    maps = compute_sensitivity3d(None, None, plan=plan, solver=s, objective="cd")
    wall = time.perf_counter() - t0
    assert maps.cuda_graphs and maps.window == pytest.approx((6.0, 2.0), abs=0.01)
    assert maps.extra_memory_bytes < maps.estimated_memory_bytes
    assert wall < 900
    v = cfg.bodies[0].vertices_numpy()
    sens = maps[0]
    front = sens[v[:, 0] < -0.45].mean()
    eq = sens[np.abs(v[:, 0]) < 0.05].mean()
    assert front < 0 < eq
    # vertices on the missing half carry their mirror image's value
    lo = v[:, 1] < -1e-9
    assert np.allclose(sens[lo].mean(), sens[v[:, 1] > 1e-9].mean(), rtol=1e-6)
