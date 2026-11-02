"""Verification of the functional 3D Navier-Stokes core (fast, CPU, float64 mostly)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns3d import (
    ImmersedMesh,
    SolverConfig3D,
    build_geometry3d,
    core,
    init_state3d,
    make_operators3d,
    rollout3d,
    step3d,
    step3d_full,
)
from aerolume.solvers.ns3d.geometry import (
    body_window3d,
    grid_signed_distance,
    mesh_signed_distance,
    smooth_step,
)
from aerolume.solvers.ns3d.poisson import PoissonSolver3D, ieee_matmul


def _load_cases():
    """validation/ns3d_cases.py (not a package), loaded once per session under one name."""
    path = Path(__file__).resolve().parents[1] / "validation" / "ns3d_cases.py"
    mod = sys.modules.get("ns3d_cases")
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("ns3d_cases", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ns3d_cases"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("ns3d_cases", None)
        raise
    return mod


cases = _load_cases()
CPU = torch.device("cpu")


def _sphere(level: int = 1, d: float = 1.0, center=(0.0, 0.0, 0.0), **kw) -> ImmersedMesh:
    v, f = cases.icosphere(d, level, center)
    return ImmersedMesh(v, f, **kw)


def _cfg(**kw) -> SolverConfig3D:
    base = {"nx": 24, "ny": 16, "nz": 16, "h": 1 / 6, "x0": -1.5, "y0": -4 / 3, "z0": -4 / 3, "re": 50.0,
            "device": "cpu", "dtype": "float64", "perturb": False}
    base.update(kw)
    return SolverConfig3D(**base)


def _random_state(cfg, ops, seed=3, amp=0.3):
    g = torch.Generator().manual_seed(seed)
    s = init_state3d(cfg, ops)
    u, v, w = (x + amp * torch.randn(x.shape, generator=g, dtype=x.dtype) for x in (s.u, s.v, s.w))
    u, v, w = core.apply_boundary_values(u, v, w, ops)
    u, v, w, _ = core.project3d(u, v, w, ops)
    return s._replace(u=u, v=v, w=w)


# ---------------------------------------------------------------------------
# Poisson and projection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periodic", [(False, False, False), (True, True, True), (False, True, False),
                                      (True, False, True)])
@pytest.mark.parametrize("shape", [(8, 6, 5), (16, 9, 7), (12, 12, 12)])
def test_poisson_is_exact(periodic, shape):
    nx, ny, nz = shape
    ps = PoissonSolver3D(nx, ny, nz, 0.37, periodic=periodic, device=CPU, dtype=torch.float64)
    f = torch.randn(nz, ny, nx, dtype=torch.float64, generator=torch.Generator().manual_seed(nx * ny))
    f -= f.mean()
    psi = ps.solve(f)
    assert (ps.laplacian(psi) - f).abs().max() < 1e-10 * f.abs().max()
    assert abs(float(psi.mean())) < 1e-12 * float(psi.abs().max())
    # the transforms are orthonormal: forward then inverse is the identity
    assert torch.allclose(ps.inverse(ps.forward(f)), f, atol=1e-12)


def test_ieee_matmul_guard_restores_the_setting():
    m = torch.backends.cuda.matmul
    if not hasattr(m, "fp32_precision"):
        pytest.skip("this PyTorch has no per-backend fp32_precision switch")
    old = m.fp32_precision
    try:
        m.fp32_precision = "tf32"
        with ieee_matmul():
            assert m.fp32_precision == "ieee"
        assert m.fp32_precision == "tf32"
    finally:
        m.fp32_precision = old


def test_projection_divergence_float64():
    """White noise is projected to round-off; boundary-normal velocities are untouched."""
    cfg = _cfg(bc_zmin="moving", bc_zmax="noslip")
    ops = make_operators3d(cfg)
    g = torch.Generator().manual_seed(3)
    u = torch.randn(cfg.nz, cfg.ny, cfg.nx + 1, generator=g, dtype=torch.float64)
    v = torch.randn(cfg.nz, cfg.ny + 1, cfg.nx, generator=g, dtype=torch.float64)
    w = torch.randn(cfg.nz + 1, cfg.ny, cfg.nx, generator=g, dtype=torch.float64)
    u, v, w = core.apply_boundary_values(u, v, w, ops)
    up, vp, wp, _ = core.project3d(u, v, w, ops)
    assert float(core.divergence3d(up, vp, wp, cfg.h).abs().max()) * cfg.h < 1e-10
    assert torch.equal(up[..., 0], u[..., 0]) and torch.equal(up[..., -1], u[..., -1])
    assert torch.equal(vp[:, 0], v[:, 0]) and torch.equal(vp[:, -1], v[:, -1])
    assert torch.equal(wp[0], w[0]) and torch.equal(wp[-1], w[-1])


@pytest.mark.parametrize("dtype,tol", [("float64", 1e-10), ("float32", 1e-5)])
def test_solver_divergence(dtype, tol):
    """Projected divergence of real solver states relative to U/h (DESIGN 7.12)."""
    cfg = _cfg(dtype=dtype, bodies=[_sphere(2)], bc_zmin="moving", bc_zmax="noslip", perturb=True, re=200.0)
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    worst = 0.0
    with torch.no_grad():
        for _ in range(12):
            s, _ = step3d(s, geom, ops)
            worst = max(worst, float(core.divergence3d(s.u, s.v, s.w, cfg.h).abs().max()) * cfg.h)
    assert worst < tol, worst


# ---------------------------------------------------------------------------
# Taylor-Green vortex: order of accuracy, decay, energy conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_taylor_green_second_order_central(plane):
    errs = [cases.taylor_green_error(n, advection="central", plane=plane)[0] for n in (16, 32, 64)]
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(1.9 < o < 2.1 for o in orders), (errs, orders)


def test_taylor_green_upwind3_converges_and_decays_at_the_right_rate():
    up = [cases.taylor_green_error(n, advection="upwind3")[0] for n in (32, 64)]
    cen = [cases.taylor_green_error(n, advection="central")[0] for n in (32, 64)]
    assert up[1] < up[0] / 2.5 and up[0] < cen[0] and up[1] < cen[1]
    _, ratio = cases.taylor_green_error(64, advection="central", t_end=1.0)
    assert abs(ratio - 1.0) < 2e-3                        # discrete Laplacian error ~ h^2/12


def test_central_advection_conserves_energy():
    """Inviscid periodic flow: the divergence-form central scheme loses energy only
    through the O(dt^3) Runge-Kutta error."""
    n = 16
    cfg = SolverConfig3D(nx=n, ny=n, nz=n, h=2 * math.pi / n, x0=0, y0=0, z0=0, re=1e12, periodic=True,
                         advection="central", device="cpu", dtype="float64", perturb=False)
    ops = make_operators3d(cfg)
    s = _random_state(cfg, ops, amp=1.0)
    geom = build_geometry3d([], ops)

    def energy(st):
        return float((st.u[..., :-1] ** 2).sum() + (st.v[:, :-1] ** 2).sum() + (st.w[:-1] ** 2).sum())
    e0 = energy(s)
    with torch.no_grad():
        for _ in range(30):
            s, _ = step3d(s, geom, ops, 0.02)
    assert abs(energy(s) / e0 - 1) < 1e-4


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------

def test_ghost_cells_impose_the_wall_values():
    cfg = _cfg(bc_zmin="moving", bc_zmax="noslip", ground_speed=0.7)
    ops = make_operators3d(cfg)
    g = torch.Generator().manual_seed(1)
    u = torch.rand(cfg.nz, cfg.ny, cfg.nx + 1, generator=g, dtype=torch.float64)
    v = torch.rand(cfg.nz, cfg.ny + 1, cfg.nx, generator=g, dtype=torch.float64)
    w = torch.rand(cfg.nz + 1, cfg.ny, cfg.nx, generator=g, dtype=torch.float64)
    u, v, w = core.apply_boundary_values(u, v, w, ops)
    U, V, W = core.pad_velocity(u, v, w, ops)
    m = slice(2, -2)
    road_u = 0.5 * (U[1, m, m] + U[2, m, m])                     # u interpolated onto z = z0
    top_u = 0.5 * (U[-2, m, m] + U[-3, m, m])
    road_v = 0.5 * (V[1, m, m] + V[2, m, m])
    assert torch.allclose(road_u, torch.full_like(road_u, 0.7)) and torch.allclose(top_u, torch.zeros_like(top_u))
    assert torch.allclose(road_v, torch.zeros_like(road_v))
    # slip side walls: zero normal gradient of u and w, v = 0 on the wall and odd across it
    assert torch.equal(U[m, 1, m], U[m, 2, m]) and torch.equal(W[m, 1, m], W[m, 2, m])
    assert torch.equal(V[m, 1, m], -V[m, 3, m]) and torch.all(V[m, 2, m] == 0)
    # inlet: v and w vanish on the inlet plane, u repeats the inlet value
    assert torch.allclose(0.5 * (V[m, m, 1] + V[m, m, 2]), torch.zeros_like(V[m, m, 1]))
    assert torch.all(U[m, m, 0] == 1.0) and torch.all(U[m, m, 1] == 1.0)


def test_periodic_padding_wraps():
    cfg = SolverConfig3D(nx=6, ny=5, nz=4, h=0.5, x0=0, y0=0, z0=0, re=10, periodic=True, device="cpu",
                         dtype="float64", perturb=False)
    ops = make_operators3d(cfg)
    u = torch.randn(4, 5, 7, dtype=torch.float64)
    u, _, _ = core.sync_periodic(u, torch.zeros(4, 6, 6, dtype=torch.float64), torch.zeros(5, 5, 6), ops)
    U = core.pad3(u, ops.ghost["u"])
    assert torch.equal(U[2:-2, 2:-2, 0], u[..., 4]) and torch.equal(U[2:-2, 2:-2, -1], u[..., 2])
    assert torch.equal(U[0, 2:-2, 2:-2], u[2]) and torch.equal(U[2:-2, 1, 2:-2], u[:, 4])


def test_pad3_is_differentiable():
    cfg = SolverConfig3D(nx=5, ny=4, nz=4, h=0.3, x0=0, y0=0, z0=0, re=10.0, device="cpu", dtype="float64",
                         bc_zmin="moving", bc_zmax="noslip", perturb=False)
    ops = make_operators3d(cfg)
    for comp, shape in (("u", (4, 4, 6)), ("v", (4, 5, 5)), ("w", (5, 4, 5))):
        x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda t, c=comp: core.pad3(t, ops.ghost[c]), (x,))


def test_inlet_walls_outlet_mass_balance():
    cfg = _cfg(bodies=[_sphere(2)], bc_zmin="moving", bc_zmax="noslip", perturb=True)
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    with torch.no_grad():
        for _ in range(12):
            s, _ = step3d(s, geom, ops)
    assert torch.all(s.u[..., 0] == 1.0)
    assert torch.all(s.v[:, 0] == 0) and torch.all(s.v[:, -1] == 0)
    assert torch.all(s.w[0] == 0) and torch.all(s.w[-1] == 0)
    assert abs(float(s.u[..., -1].sum() - s.u[..., 0].sum())) < 1e-10


def test_periodic_y_keeps_duplicate_faces():
    cfg = _cfg(bodies=[_sphere(2)], bc_y="periodic", perturb=True)
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    with torch.no_grad():
        for _ in range(8):
            s, _ = step3d(s, geom, ops)
    assert torch.equal(s.v[:, 0], s.v[:, -1])
    assert float(core.divergence3d(s.u, s.v, s.w, cfg.h).abs().max()) * cfg.h < 1e-10


def test_cfl_time_step():
    cfg = _cfg(re=1e4)
    ops = make_operators3d(cfg)
    s = init_state3d(cfg, ops)
    assert float(core.cfl_dt3d(s.u, s.v, s.w, ops)) == pytest.approx(cfg.cfl * cfg.h / 1.0)
    cfg = _cfg(re=0.5)
    ops = make_operators3d(cfg)
    assert float(core.cfl_dt3d(s.u, s.v, s.w, ops)) == pytest.approx(cfg.h ** 2 / 6 * 0.5)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_mesh_signed_distance_sphere_and_box():
    v, f = cases.icosphere(2.0, 3)
    pts = torch.tensor([[0, 0, 0], [0.5, 0, 0], [2.0, 0, 0], [0, -3.0, 0]], dtype=torch.float64)
    d = mesh_signed_distance(torch.as_tensor(v), torch.as_tensor(f), pts)
    # level-3 facets sit up to 0.5 % of the radius inside the true sphere
    assert torch.allclose(d, torch.tensor([-1.0, -0.5, 1.0, 2.0], dtype=torch.float64), atol=6e-3)
    v, f = cases.box_mesh(1.0, 2.0, 3.0, n=2)
    pts = torch.tensor([[0, 0, 0], [0.6, 0, 0], [0, 1.25, 0], [0.75, 1.25, 0]], dtype=torch.float64)
    d = mesh_signed_distance(torch.as_tensor(v), torch.as_tensor(f), pts)
    assert torch.allclose(d, torch.tensor([-0.5, 0.1, 0.25, math.hypot(0.25, 0.25)], dtype=torch.float64))


def test_signed_distance_ignores_global_winding_and_small_holes():
    """The private fallback: a mesh wound the other way and a mesh with two missing
    triangles classify inside / outside like the clean mesh (|winding number| > 1/2)."""
    v, f = cases.icosphere(1.0, 3)
    pts = torch.tensor([[0, 0, 0], [0.3, 0.1, 0.0], [0.8, 0, 0], [0, 0, -0.7]], dtype=torch.float64)
    ref = mesh_signed_distance(torch.as_tensor(v), torch.as_tensor(f), pts)
    assert torch.equal(ref < 0, torch.tensor([True, True, False, False]))
    for bad in (f[:, ::-1].copy(), np.delete(f, [5, 400], axis=0)):
        d = mesh_signed_distance(torch.as_tensor(v), torch.as_tensor(bad), pts)
        assert torch.equal(d < 0, ref < 0)


def test_shared_voxeliser_ignores_random_winding():
    """With the shared voxeliser (ray-parity voting) even randomly flipped triangles plus
    holes leave the masks as for the clean mesh (AirShaper behaviour)."""
    pytest.importorskip("aerolume.geometry3d.voxel")
    v, f = cases.icosphere(1.0, 3)
    rng = np.random.default_rng(0)
    bad = f.copy()
    flip = rng.random(len(f)) < 0.3
    bad[flip] = bad[flip][:, ::-1]
    bad = np.delete(bad, [5, 400], axis=0)
    cfg = _cfg(nx=16, ny=16, nz=16, h=1 / 8, x0=-1.0, y0=-1.0, z0=-1.0)
    ops = make_operators3d(cfg)
    ga = build_geometry3d([ImmersedMesh(v, f)], ops)
    gb = build_geometry3d([ImmersedMesh(v, bad)], ops)
    assert float((ga.chi_c - gb.chi_c).abs().max()) < 0.05


def test_shared_voxeliser_matches_the_exact_distance():
    """The solver uses aerolume.geometry3d.voxel.sdf_band when importable; within the band
    it must equal the exact distance - also on a window that cuts the body at a symmetry
    plane, and for an open mesh cut by a window across x (the voxeliser once took faces
    whose probes left its grid for open sheets and thickened them by half a cell, and the
    bridge evaluated a covering grid to avoid it)."""
    pytest.importorskip("aerolume.geometry3d.voxel")
    v, f = cases.icosphere(1.0, 3)
    vt, ft = torch.as_tensor(v), torch.as_tensor(f)
    h = 1 / 8
    band = 2.5 * h
    grid = (18, 7, 14, h, -1.1, 0.0, -0.875)                    # y >= 0 only: half the sphere outside
    a = grid_signed_distance([(vt, ft)], grid, band, use_shared=True)
    b = grid_signed_distance([(vt, ft)], grid, band, use_shared=False).clamp(-band, band)
    assert float((a - b).abs().max()) < 1e-6
    # the open y >= 0 half of the sphere through a window across x: compare with a grid that
    # holds all of it (the exact fallback needs a closed mesh, the voxeliser does not)
    half = ft[(vt[ft][:, :, 1] >= -1e-12).all(1)]
    win = (4, 7, 14, h, -0.25, 0.0, -0.875)
    big = (20, 11, 18, h, -1.25, -0.25, -1.125)
    c = grid_signed_distance([(vt, half)], win, band, use_shared=True)
    d = grid_signed_distance([(vt, half)], big, band, use_shared=True)[2:16, 2:9, 8:12]
    assert float((c - d).abs().max()) < 1e-9


def test_private_distance_is_differentiable_through_the_surface():
    """A point exactly on a face: the fallback distance is the face's signed plane distance
    (value 0, the derivative of moving that face), not sqrt(0) with no derivative."""
    v, f = cases.icosphere(1.0, 2)
    vt = torch.as_tensor(v).clone().requires_grad_(True)
    tri = vt.detach()[torch.as_tensor(f)[7]]
    p = (tri.mean(0))[None]                                       # the centroid of face 7
    d = mesh_signed_distance(vt, torch.as_tensor(f), p, plane_zone=1e-9)
    assert abs(float(d.detach())) < 1e-12
    d.sum().backward()
    n = torch.linalg.cross(tri[1] - tri[0], tri[2] - tri[0])
    n = n / n.norm()
    n = n if float(n @ tri.mean(0)) > 0 else -n                   # outward
    g = vt.grad[torch.as_tensor(f)[7]]
    # pushing the face outward by s along n makes the point inside by s: dd/ds = -1
    assert float((g * n).sum()) == pytest.approx(-1.0, rel=1e-9)


def test_smooth_step_is_compact_and_monotone():
    d = torch.linspace(-2, 2, 401, dtype=torch.float64)
    chi = smooth_step(d, 1.0)
    assert float(chi[d <= -1].min()) == 1.0 and float(chi[d >= 1].max()) == 0.0
    assert torch.all(chi[1:] <= chi[:-1])


def test_masks_volume_window_and_boundary_faces():
    cfg = _cfg(nx=20, ny=16, nz=16, h=1 / 8, x0=-1.25, y0=-1.0, z0=-1.0, bodies=[_sphere(3)])
    ops = make_operators3d(cfg)
    g = build_geometry3d(cfg.bodies, ops)
    vol = float(g.chi_c.sum()) * cfg.h ** 3
    # radial integral of the shifted half-sine ramp around the faceted sphere's equivalent radius
    # (the ramp's first moment adds ~0.19 * 4 pi r w^2 to the volume of the 0.5 iso-surface)
    v, f = cases.icosphere(1.0, 3)
    radius = (3 * cases.mesh_volume(v, f) / (4 * math.pi)) ** (1 / 3)
    r = torch.linspace(0.0, 1.0, 200001, dtype=torch.float64)
    chi = smooth_step(r - radius + core.MASK_SHIFT * cfg.h, cfg.h)
    expected = float(torch.trapezoid(4 * math.pi * r * r * chi, r))
    assert vol == pytest.approx(expected, rel=5e-3)
    k0, k1, j0, j1, i0, i1 = g.window
    assert g.chi_u.shape == (k1 - k0, j1 - j0, i1 - i0 + 1) and g.chi_w.shape == (k1 - k0 + 1, j1 - j0, i1 - i0)
    # a body touching the domain: boundary faces never penalised
    cfg = _cfg(nx=20, ny=16, nz=16, h=1 / 8, x0=-1.25, y0=0.0, z0=-1.0, bodies=[_sphere(3)],
               bc_y="symmetry_ymin")
    ops = make_operators3d(cfg)
    g = build_geometry3d(cfg.bodies, ops)
    assert g.window[2] == 0 and float(g.chi_v[:, 0].abs().max()) == 0.0
    assert float(g.chi_v[:, 1].max()) > 0.9


def test_live_window_covers_rotations():
    cfg = _cfg(bodies=[ImmersedMesh(*cases.box_mesh(1.0, 0.4, 0.2), name="plate")])
    ops = make_operators3d(cfg)
    verts = [cfg.bodies[0].vertices]
    w0 = body_window3d(verts, ops)
    w1 = body_window3d(verts, ops, pitch_deg=20.0, yaw_deg=10.0)
    assert w1[0] < w0[0] and w1[2] < w0[2]                       # grows in z (pitch) and y (yaw)


def test_rolling_wheel_sign():
    """omega = -U/r about +y: the contact point below the axle moves at +U with the road."""
    r = 0.25
    wheel = ImmersedMesh(*cases.cylinder_mesh(2 * r, 0.2, center=(0.0, 0.0, r), axis=1, n=48), name="wheel",
                         role="wheel", omega=-1.0 / r, center=(0.0, 0.0, r), axis=(0.0, 1.0, 0.0))
    cfg = _cfg(nx=16, ny=12, nz=12, h=1 / 16, x0=-0.5, y0=-0.375, z0=0.0, bodies=[wheel], bc_zmin="moving")
    ops = make_operators3d(cfg)
    g = build_geometry3d(cfg.bodies, ops)
    k0, _, j0, j1, i0, i1 = g.window
    xs = cfg.x0 + (torch.arange(i0, i1 + 1, dtype=torch.float64)) * cfg.h
    i = int(torch.argmin(xs.abs()))
    us = g.a_u[0, (j1 - j0) // 2, i] / g.chi_u[0, (j1 - j0) // 2, i]     # bottom layer, centre row, x = 0
    zc = cfg.z0 + (k0 + 0.5) * cfg.h
    assert float(us) == pytest.approx(-(-1.0 / r) * (r - zc), rel=1e-9)   # = +U (r - z)/r > 0
    assert float(us) > 0


# ---------------------------------------------------------------------------
# forces, moments, symmetry
# ---------------------------------------------------------------------------

def test_sphere_drag_positive_and_no_side_force_or_moments():
    cfg = _cfg(bodies=[_sphere(2)])
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    with torch.no_grad():
        for _ in range(20):
            s, c = step3d(s, geom, ops)
    cd, cs, cl, cr, cm, cn = c.tolist()
    assert cd > 0.5
    assert max(abs(cs), abs(cl), abs(cr), abs(cm), abs(cn)) < 1e-3 * cd


def test_moment_reference_transfer():
    """M(r2) = M(r1) + (r1 - r2) x F for the same discrete forces."""
    body = ImmersedMesh(*cases.box_mesh(0.8, 0.3, 0.2, center=(0.1, 0.05, 0.08)), name="b")
    out = []
    for ref in ((0.0, 0.0, 0.0), (0.3, -0.2, 0.1)):
        cfg = _cfg(bodies=[body], moment_ref=ref, perturb=True, ref_area=0.7, length_ref=1.3, span_ref=0.9)
        ops = make_operators3d(cfg)
        geom = build_geometry3d(cfg.bodies, ops)
        s = init_state3d(cfg, ops)
        with torch.no_grad():
            for _ in range(6):
                s, c = step3d(s, geom, ops, 0.02)
        out.append(c.numpy())
    q = 0.5 * 0.7
    f = out[0][:3] * q
    m1 = out[0][3:] * q * np.array([0.9, 1.3, 0.9])
    m2 = out[1][3:] * q * np.array([0.9, 1.3, 0.9])
    r1, r2 = np.zeros(3), np.array([0.3, -0.2, 0.1])
    assert np.allclose(m2, m1 + np.cross(r1 - r2, f), atol=1e-10)
    assert abs(out[0][2]) > 1e-3 and abs(out[0][4]) > 1e-4           # a lifting, pitching body


def test_pitch_sign_nose_up():
    """A flat plate pitched nose-up (+alpha about +y, DESIGN 7.2) makes positive lift."""
    from aerolume.solvers.ns3d.geometry import rotation_matrix
    v, f = cases.box_mesh(1.0, 0.8, 0.25, n=3)
    v = v @ rotation_matrix(10.0, 0.0).T
    cfg = _cfg(nx=28, bodies=[ImmersedMesh(v, f, name="plate")], re=100.0, ref_area=0.8)
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    acc = []
    with torch.no_grad():
        for _ in range(40):
            s, c = step3d(s, geom, ops)
            acc.append(c.numpy())
    cl = np.mean(acc[-10:], axis=0)[2]
    assert cl > 0.05


def test_symmetry_plane_equals_full_domain():
    """A half-domain run reports the full body's coefficients: same CD and CL as the
    full domain, zero side force, roll and yaw (moment reference on the plane)."""
    body = ImmersedMesh(*cases.box_mesh(0.6, 0.5, 0.3, center=(0.0, 0.0, 0.1), n=2), name="b")
    res = {}
    for name, y0, ny, bc in (("full", -4 / 3, 16, "slip"), ("half", 0.0, 8, "symmetry_ymin")):
        cfg = _cfg(bodies=[body], y0=y0, ny=ny, bc_y=bc, moment_ref=(0.1, 0.0, 0.0), perturb=True)
        ops = make_operators3d(cfg)
        geom = build_geometry3d(cfg.bodies, ops)
        s = init_state3d(cfg, ops)
        with torch.no_grad():
            for _ in range(10):
                s, c = step3d(s, geom, ops, 0.02)
        res[name] = c.numpy()
    full, half = res["full"], res["half"]
    assert np.allclose(half, full, rtol=1e-9, atol=1e-10), (half, full)
    assert half[1] == 0.0 and half[3] == 0.0 and half[5] == 0.0
    assert abs(full[2]) > 1e-3 and abs(full[4]) > 1e-4


def test_multiple_bodies_sum():
    a = _sphere(1, 0.6, (-0.4, 0.0, 0.0), name="a")
    b = _sphere(1, 0.6, (0.5, 0.0, 0.3), name="b")
    cfg = _cfg(bodies=[a, b])
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    with torch.no_grad():
        for _ in range(5):
            s, cb = step3d_full(s, geom, ops)
    assert cb.shape == (2, 6) and torch.all(cb[:, 0] > 0)
    _, tot = step3d(s, geom, ops)
    assert tot.shape == (6,)


def test_rotating_body_drives_flow():
    """A sphere spinning about +z in a periodic box sets the fluid around it rotating."""
    n = 16
    cfg = SolverConfig3D(nx=n, ny=n, nz=n, h=2.0 / n, x0=-1, y0=-1, z0=-1, re=20.0, periodic=True,
                         device="cpu", dtype="float64", perturb=False,
                         bodies=[_sphere(2, 0.8, omega=2.0, center=(0, 0, 0), axis=(0, 0, 1))])
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    with torch.no_grad():
        for _ in range(30):
            s, _ = step3d(s, geom, ops)
    vc = 0.5 * (s.v[n // 2, :-1] + s.v[n // 2, 1:])           # v at cell centres, mid-plane
    i = int((0.55 - cfg.x0) / cfg.h)                           # just outside the sphere at +x
    assert float(vc[n // 2, i]) > 0.05                          # counter-clockwise about +z


# ---------------------------------------------------------------------------
# LES models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
def test_eddy_viscosity_models(model):
    ops = make_operators3d(_cfg(turbulence=model))
    g = torch.Generator().manual_seed(0)
    gr = [[torch.randn(50, generator=g, dtype=torch.float64) for _ in range(3)] for _ in range(3)]
    tr = (gr[0][0] + gr[1][1] + gr[2][2]) / 3
    for i in range(3):
        gr[i][i] = gr[i][i] - tr                                   # traceless (incompressible)
    nut = core.eddy_viscosity3d(gr, ops)
    assert torch.all(nut >= 0) and torch.all(torch.isfinite(nut))
    z = torch.zeros(5, dtype=torch.float64)
    shear = [[z, z + 1.0, z], [z, z, z], [z, z, z]]               # pure shear du/dy
    nut = core.eddy_viscosity3d(shear, ops)
    if model == "smagorinsky":
        assert torch.all(nut > 0)
    else:
        assert torch.all(nut.abs() < 1e-12)


@pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
def test_les_step_runs_zero_in_solid_and_differentiable(model):
    cfg = _cfg(bodies=[_sphere(2)], turbulence=model, re=1e5, perturb=True)
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = _random_state(cfg, ops)
    U, V, W = core.pad_velocity(s.u, s.v, s.w, ops)
    fu, fv, fw, nut_max = core.sgs_divergence3d(U, V, W, geom, ops)
    assert all(bool(torch.isfinite(t).all()) for t in (fu, fv, fw)) and float(nut_max) > 0
    chi_c = geom.chi_c.clone().requires_grad_(True)
    chi_u = geom.chi_u.clone().requires_grad_(True)
    st = init_state3d(cfg, ops)
    for _ in range(2):
        st, c = step3d(st, geom._replace(chi_c=chi_c, chi_u=chi_u), ops, 0.01)
    c[0].backward()
    for t in (chi_c, chi_u):
        assert t.grad is not None and bool(torch.isfinite(t.grad).all())
    assert float(chi_u.grad.abs().sum()) > 0


# ---------------------------------------------------------------------------
# differentiability
# ---------------------------------------------------------------------------

def _mean_cd(eps: torch.Tensor, nsteps: int = 10, use_shared: bool = False) -> torch.Tensor:
    v, f = cases.icosphere(1.0, 1)
    base = torch.as_tensor(v, dtype=torch.float64)
    mode = torch.stack([base[:, 0], torch.zeros_like(base[:, 1]), 0.5 * base[:, 2]], dim=1)   # smooth stretch
    verts = base + eps * mode
    cfg = _cfg(nx=20, ny=12, nz=12, h=1 / 5, x0=-1.6, y0=-1.2, z0=-1.2, re=20.0)
    ops = make_operators3d(cfg)
    body = ImmersedMesh(verts, f, name="s")
    geom = build_geometry3d([body], ops, window=(1, 11, 1, 11, 2, 14), use_shared=use_shared)
    s = init_state3d(cfg, ops)
    _, coeffs = rollout3d(s, geom, ops, nsteps, dt=0.04, checkpoint_every=4)
    return coeffs[:, 0].mean()


@pytest.mark.parametrize("use_shared", [False, True])
def test_gradient_matches_finite_difference(use_shared):
    """d(mean CD over 10 steps)/d(vertex displacement mode): autograd vs central FD, float64."""
    if use_shared:
        pytest.importorskip("aerolume.geometry3d.voxel")
    eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
    j = _mean_cd(eps, use_shared=use_shared)
    (g,) = torch.autograd.grad(j, eps)
    with torch.no_grad():
        e = 1e-5
        fd = (_mean_cd(torch.tensor(e, dtype=torch.float64), use_shared=use_shared)
              - _mean_cd(torch.tensor(-e, dtype=torch.float64), use_shared=use_shared)) / (2 * e)
    assert abs(float(fd)) > 1e-2                                      # a real, non-trivial sensitivity
    assert abs(float(g) - float(fd)) < 1e-4 * abs(float(fd)), (float(g), float(fd))


def test_gradient_wrt_inflow_and_initial_state():
    cfg = _cfg(nx=16, ny=10, nz=10, h=1 / 4, x0=-1.5, y0=-1.25, z0=-1.25, re=30.0, bodies=[_sphere(1)])
    ops = make_operators3d(cfg)
    geom = build_geometry3d(cfg.bodies, ops)
    s = init_state3d(cfg, ops)
    u0 = s.u.clone().requires_grad_(True)
    ops.u_inlet = ops.u_inlet.clone().requires_grad_(True)
    _, c = rollout3d(s._replace(u=u0), geom, ops, 3, dt=0.04, checkpoint_every=0)
    c[:, 0].sum().backward()
    assert u0.grad is not None and float(u0.grad.abs().sum()) > 0
    assert ops.u_inlet.grad is not None and float(ops.u_inlet.grad.abs().sum()) > 0
