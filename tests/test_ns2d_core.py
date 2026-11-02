"""Verification of the functional Navier-Stokes core (fast, CPU, float64 mostly)."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns2d import (
    ImmersedBody,
    SolverConfig,
    build_geometry,
    core,
    init_state,
    make_operators,
    step,
    step_full,
)
from aerolume.solvers.ns2d.geometry import polygon_sdf, smooth_step
from aerolume.solvers.ns2d.poisson import PoissonSolver
from aerolume.solvers.ns2d.rollout import rollout


def _load_cases():
    """validation/ns2d_cases.py (not a package), loaded once per session.

    Every ns2d test module shares one module object under one name: re-executing
    the file per test module would leave dataclasses pointing at a stale module,
    and a failed load must not leave a half-initialised entry in sys.modules.
    """
    path = Path(__file__).resolve().parents[1] / "validation" / "ns2d_cases.py"
    mod = sys.modules.get("ns2d_cases")
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("ns2d_cases", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ns2d_cases"] = mod           # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("ns2d_cases", None)
        raise
    return mod


cases = _load_cases()

CPU = torch.device("cpu")


def _cfg(**kw) -> SolverConfig:
    base = {"nx": 64, "ny": 32, "h": 1 / 8, "x0": -2.0, "y0": -2.0, "re": 50.0, "device": "cpu",
            "dtype": "float64", "dye": False, "perturb": False}
    base.update(kw)
    return SolverConfig(**base)


# ---------------------------------------------------------------------------
# Poisson and projection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("shape", [(16, 8), (15, 9), (64, 33), (128, 96)])
def test_poisson_is_exact(periodic, shape):
    nx, ny = shape
    ps = PoissonSolver(nx, ny, 0.37, periodic=periodic, device=CPU, dtype=torch.float64)
    f = torch.randn(ny, nx, dtype=torch.float64, generator=torch.Generator().manual_seed(nx))
    f -= f.mean()
    psi = ps.solve(f)
    assert (ps.laplacian(psi) - f).abs().max() < 1e-10 * f.abs().max()
    assert abs(float(psi.mean())) < 1e-12 * float(psi.abs().max())


def test_projection_divergence_float64():
    """Even a white-noise field is projected to round-off in float64."""
    cfg = _cfg()
    ops = make_operators(cfg)
    g = torch.Generator().manual_seed(3)
    u = torch.randn(cfg.ny, cfg.nx + 1, generator=g, dtype=torch.float64)
    v = torch.randn(cfg.ny + 1, cfg.nx, generator=g, dtype=torch.float64)
    u, v = core._apply_boundary_values(u, v, ops)
    up, vp, _ = core.project(u, v, ops)
    div = core.divergence(up, vp, cfg.h)
    assert float(div.abs().max()) * cfg.h < 1e-10           # relative to U/h with |u| ~ 1
    # boundary-normal velocities are untouched by the projection
    assert torch.equal(up[:, 0], u[:, 0]) and torch.equal(up[:, -1], u[:, -1])
    assert torch.equal(vp[0], v[0]) and torch.equal(vp[-1], v[-1])


@pytest.mark.parametrize("dtype,tol", [("float64", 1e-10), ("float32", 1e-5)])
def test_solver_divergence(dtype, tol):
    """Projected divergence of real solver states, relative to U/h (DESIGN quality bar)."""
    cfg = _cfg(dtype=dtype, nx=128, ny=64, h=1 / 16, x0=-2.0, y0=-2.0, re=200.0, perturb=True,
               bodies=[ImmersedBody(cases.circle(96))], bc_bottom="moving", bc_top="noslip")
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    worst = 0.0
    with torch.no_grad():
        for _ in range(40):
            s, _ = step(s, geom, ops)
            worst = max(worst, float(core.divergence(s.u, s.v, cfg.h).abs().max()) * cfg.h)
    assert worst < tol, worst


def test_step_keeps_divergence_free_with_body():
    cfg = _cfg(bodies=[ImmersedBody(cases.circle(64))])
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    with torch.no_grad():
        for _ in range(20):
            s, _ = step(s, geom, ops)
    assert float(core.divergence(s.u, s.v, cfg.h).abs().max()) * cfg.h < 1e-12


# ---------------------------------------------------------------------------
# Taylor-Green vortex (periodic): order of accuracy and decay rate
# ---------------------------------------------------------------------------

def test_taylor_green_second_order_central():
    errs = [cases.taylor_green_error(n, advection="central")[0] for n in (16, 32, 64)]
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert all(1.9 < o < 2.1 for o in orders), (errs, orders)


def test_taylor_green_upwind3_converges():
    """upwind3 = 2nd-order diffusion error + 3rd-order upwind dissipation of the
    opposite sign: the observed order rises from ~1.4 towards 2 as the 3rd-order
    term dies out, and the error stays below the central scheme's."""
    up = [cases.taylor_green_error(n, advection="upwind3")[0] for n in (32, 64, 128)]
    cen = [cases.taylor_green_error(n, advection="central")[0] for n in (32, 64)]
    assert up[0] < cen[0] and up[1] < cen[1]
    assert math.log2(up[1] / up[2]) > 1.6


def test_taylor_green_decay_rate():
    _, ratio = cases.taylor_green_error(64, advection="central")
    assert abs(ratio - 1.0) < 2e-3                     # discrete Laplacian error ~ h^2/12


def test_central_advection_conserves_energy():
    """Inviscid periodic flow: the energy-conserving central scheme loses energy
    only through the O(dt^3) Runge-Kutta error."""
    n = 32
    cfg = SolverConfig(nx=n, ny=n, h=2 * math.pi / n, x0=0, y0=0, re=1e12, periodic=True,
                       advection="central", device="cpu", dtype="float64", dye=False, perturb=False)
    ops = make_operators(cfg)
    g = torch.Generator().manual_seed(1)
    u = torch.randn(n, n + 1, generator=g, dtype=torch.float64)
    v = torch.randn(n + 1, n, generator=g, dtype=torch.float64)
    u, v = core._apply_boundary_values(u, v, ops)
    u, v, _ = core.project(u, v, ops)
    s = init_state(cfg, ops)._replace(u=u, v=v)
    geom = build_geometry([], ops)
    e0 = float((u[:, :-1] ** 2).sum() + (v[:-1] ** 2).sum())
    with torch.no_grad():
        for _ in range(50):
            s, _ = step(s, geom, ops, 0.02)
    e1 = float((s.u[:, :-1] ** 2).sum() + (s.v[:-1] ** 2).sum())
    assert abs(e1 / e0 - 1) < 1e-4


# ---------------------------------------------------------------------------
# boundary conditions
# ---------------------------------------------------------------------------

def test_inlet_walls_outlet_mass_balance():
    cfg = _cfg(bodies=[ImmersedBody(cases.circle(64))], bc_top="noslip", bc_bottom="moving", perturb=True)
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    with torch.no_grad():
        for _ in range(30):
            s, _ = step(s, geom, ops)
    assert torch.all(s.u[:, 0] == 1.0)                                  # inlet held exactly
    assert torch.all(s.v[0] == 0) and torch.all(s.v[-1] == 0)           # impermeable walls
    assert abs(float(s.u[:, -1].sum() - s.u[:, 0].sum())) < 1e-10       # global mass balance


def test_wall_ghosts():
    cfg = _cfg(bc_top="noslip", bc_bottom="moving", ground_speed=0.7)
    ops = make_operators(cfg)
    u = torch.rand(cfg.ny, cfg.nx + 1, dtype=torch.float64)
    U = core.pad_u(u, ops)
    wall_bottom = 0.5 * (U[1, 2:-2] + U[2, 2:-2])          # u interpolated onto y = y0
    wall_top = 0.5 * (U[-2, 2:-2] + U[-3, 2:-2])
    assert torch.allclose(wall_bottom, torch.full_like(wall_bottom, 0.7))
    assert torch.allclose(wall_top, torch.zeros_like(wall_top))


def test_parabolic_inlet_has_unit_mean():
    prof = core.inlet_profile(_cfg(inlet="parabolic"))
    assert abs(prof.mean() - 1.0) < 1e-12 and abs(prof.max() - 1.5) < 0.01


# ---------------------------------------------------------------------------
# geometry, forces and signs
# ---------------------------------------------------------------------------

def test_polygon_sdf_circle():
    poly = torch.as_tensor(cases.circle(400, 2.0), dtype=torch.float64)
    x = torch.tensor([0.0, 0.5, 1.0, 2.0, -3.0], dtype=torch.float64)
    y = torch.zeros_like(x)
    d = polygon_sdf(poly, x, y)
    assert torch.allclose(d, torch.tensor([-1.0, -0.5, 0.0, 1.0, 2.0], dtype=torch.float64), atol=1e-4)


def test_shared_sdf_matches_private_fallback():
    """The solver uses aerolume.geometry.sdf when present; it must agree with the
    private fallback (exact distance to edges, negative inside)."""
    shared = pytest.importorskip("aerolume.geometry.sdf").signed_distance
    for poly in (cases.naca4("4412", 200), cases.circle(256), cases.rounded_box()):
        p = torch.as_tensor(poly, dtype=torch.float64)
        ys, xs = torch.meshgrid(torch.linspace(-0.6, 0.7, 90, dtype=torch.float64),
                                torch.linspace(-0.6, 1.2, 120, dtype=torch.float64), indexing="ij")
        a, b = shared([p], xs, ys), polygon_sdf(p, xs, ys)
        assert float((a - b).abs().max()) < 1e-12 and bool(torch.all((a < 0) == (b < 0)))


def test_smooth_step_is_compact_and_monotone():
    d = torch.linspace(-2, 2, 401, dtype=torch.float64)
    chi = smooth_step(d, 1.0)
    assert float(chi[d <= -1].min()) == 1.0 and float(chi[d >= 1].max()) == 0.0
    assert torch.all(chi[1:] <= chi[:-1]) and abs(float(smooth_step(torch.tensor(0.0), 1.0)) - 0.5) < 1e-12


def test_symmetric_body_drag_positive_no_lift():
    cfg = _cfg(bodies=[ImmersedBody(cases.circle(64))], moment_ref=(0.0, 0.0))
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    with torch.no_grad():
        for _ in range(40):
            s, c = step(s, geom, ops)
    cd, cl, cm = c.tolist()
    assert cd > 0.5 and abs(cl) < 1e-8 and abs(cm) < 1e-8


def test_cambered_airfoil_lift_and_nose_down_moment():
    """NACA 4412 vs 0012 at 4 deg, Re = 1000: both lift, and camber adds a
    nose-down (negative) moment about c/4, as aeronautics expects (Cm positive
    nose-up). At this Re thick boundary layers almost cancel the camber's lift
    at alpha = 0 (Cl ~ 0.03 at 160 cells/chord), so the sign test uses 4 deg."""
    out = {}
    for code in ("4412", "0012"):
        h = 1 / 40
        poly = cases.rotate(cases.naca4(code, 160), -4.0, (0.25, 0.0))     # nose up by 4 deg
        cfg = SolverConfig(nx=int(4.5 / h), ny=int(3.0 / h), h=h, x0=-1.0, y0=-1.5, re=1000.0,
                           bodies=[ImmersedBody(poly)], moment_ref=(0.25, 0.0),
                           device="cpu", dtype="float32", dye=False, perturb=False)
        ops = make_operators(cfg)
        geom = build_geometry(cfg.bodies, ops)
        s = init_state(cfg, ops)
        acc = []
        with torch.no_grad():
            while float(s.t) < 6.0:
                s, c = step(s, geom, ops)
                if float(s.t) > 4.5:
                    acc.append(c.numpy())
        out[code] = np.mean(acc, axis=0)
    cd, cl, cm = out["4412"]
    assert cd > 0 and cl > 0.1 and cm < -0.01, out
    assert out["0012"][1] > 0.1 and out["0012"][2] > cm + 0.02, out


def test_moment_reference_transfer():
    """Cm about two reference points differs by the lever arm of the forces."""
    poly = cases.rotate(cases.naca4("0012", 120), -8.0, (0.25, 0.0))
    res = []
    for ref in ((0.25, 0.0), (0.0, 0.0)):
        cfg = _cfg(bodies=[ImmersedBody(poly)], moment_ref=ref, nx=72, ny=40, h=1 / 16, x0=-1.5, y0=-1.25)
        ops = make_operators(cfg)
        geom = build_geometry(cfg.bodies, ops)
        s = init_state(cfg, ops)
        with torch.no_grad():
            for _ in range(10):
                s, c = step(s, geom, ops, 0.01)
        res.append(c.numpy())
    (_cd, cl, cm_q), (_, _, cm_le) = res
    # nose-up moment about the LE = about c/4 minus (0.25) * lift (lift acts behind the LE)
    assert abs(cm_le - (cm_q - 0.25 * cl)) < 1e-9


def test_multiple_bodies_sum():
    a = ImmersedBody(cases.circle(48, 0.6, -0.5, 0.0), name="a")
    b = ImmersedBody(cases.circle(48, 0.6, 0.6, 0.4), name="b")
    cfg = _cfg(bodies=[a, b])
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    with torch.no_grad():
        for _ in range(10):
            s, cb = step_full(s, geom, ops)
    assert cb.shape == (2, 3) and torch.all(cb[:, 0] > 0)
    _, tot = step(s, geom, ops)
    assert tot.shape == (3,)


def test_rotating_body_drives_flow():
    """A spinning cylinder in a periodic box sets the fluid around it in rotation."""
    n = 48
    cfg = SolverConfig(nx=n, ny=n, h=4.0 / n, x0=-2.0, y0=-2.0, re=20.0, periodic=True, device="cpu",
                       dtype="float64", dye=False, perturb=False,
                       bodies=[ImmersedBody(cases.circle(96), omega=2.0, center=(0.0, 0.0))])
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    with torch.no_grad():
        for _ in range(60):
            s, _ = step(s, geom, ops)
    j = n // 2
    i_right = int((0.75 - cfg.x0) / cfg.h)         # cell centre just outside the cylinder at +x
    vc = 0.5 * (s.v[j, i_right] + s.v[j + 1, i_right])
    assert float(vc) > 0.1                           # counter-clockwise rotation -> v > 0 at +x


# ---------------------------------------------------------------------------
# LES models, dye
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
def test_eddy_viscosity_models(model):
    ops = make_operators(_cfg(turbulence=model))
    g = torch.Generator().manual_seed(0)
    a, b, c = (torch.randn(50, generator=g, dtype=torch.float64) for _ in range(3))
    nut = core.eddy_viscosity(a, b, c, -a, ops)
    assert torch.all(nut >= 0) and torch.all(torch.isfinite(nut))
    shear = core.eddy_viscosity(torch.zeros(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64),
                                torch.zeros(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64), ops)
    if model == "smagorinsky":
        assert torch.all(shear > 0)
    else:
        assert torch.all(shear.abs() < 1e-12)       # no eddy viscosity in pure shear


@pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
def test_les_runs_and_is_zero_in_solid(model):
    from aerolume.solvers.ns2d import FlowSolver
    cfg = _cfg(bodies=[ImmersedBody(cases.circle(64))], turbulence=model, re=1e5, perturb=True)
    s = FlowSolver(cfg)
    s.advance(30)
    nut, mask = s.field("nut"), s.field("mask")
    assert torch.all(torch.isfinite(nut)) and float(nut.max()) > 0
    assert float(nut[mask > 0.999].abs().max()) == 0.0


def test_dye_bounded_and_injected():
    cfg = _cfg(dye=True, bodies=[ImmersedBody(cases.circle(64))])
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    with torch.no_grad():
        for _ in range(60):
            s, _ = step(s, geom, ops)
    assert float(s.dye.min()) >= 0.0 and float(s.dye.max()) <= 1.0
    assert float(s.dye[:, :4].max()) > 0.5                    # streaks enter at the inlet


# ---------------------------------------------------------------------------
# differentiability
# ---------------------------------------------------------------------------

def _mean_cd(eps: torch.Tensor, nsteps: int = 20) -> torch.Tensor:
    base = torch.as_tensor(cases.circle(32), dtype=torch.float64)
    direction = torch.stack([base[:, 0], torch.zeros_like(base[:, 1])], dim=1)   # stretch in x
    poly = base + eps * direction
    cfg = _cfg(nx=48, ny=32, h=1 / 8, x0=-2.0, y0=-2.0, re=20.0)
    ops = make_operators(cfg)
    geom = build_geometry([poly], ops, window=(10, 22, 10, 22))
    s = init_state(cfg, ops)
    _, coeffs = rollout(s, geom, ops, nsteps, dt=0.02, checkpoint_every=5)
    return coeffs[:, 0].mean()


def test_gradient_matches_finite_difference():
    eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
    j = _mean_cd(eps)
    (g,) = torch.autograd.grad(j, eps)
    with torch.no_grad():
        e = 1e-4
        fd = (_mean_cd(torch.tensor(e, dtype=torch.float64)) - _mean_cd(torch.tensor(-e, dtype=torch.float64))) / (2 * e)
    assert abs(float(g) - float(fd)) < 0.05 * abs(float(fd)), (float(g), float(fd))
    assert abs(float(fd)) > 1e-3                                      # a real, non-trivial sensitivity


def test_gradient_wrt_inflow_and_initial_state():
    cfg = _cfg(nx=32, ny=24, h=1 / 6, re=30.0, bodies=[ImmersedBody(cases.circle(32))])
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    s = init_state(cfg, ops)
    u0 = s.u.clone().requires_grad_(True)
    ops.u_inlet = ops.u_inlet.clone().requires_grad_(True)
    _, c = rollout(s._replace(u=u0), geom, ops, 5, dt=0.02, checkpoint_every=0)
    c[:, 0].sum().backward()
    assert u0.grad is not None and float(u0.grad.abs().sum()) > 0
    assert ops.u_inlet.grad is not None and float(ops.u_inlet.grad.abs().sum()) > 0


@pytest.mark.parametrize("model", ["smagorinsky", "wale", "vreman"])
def test_les_step_is_differentiable_including_the_solid_fraction(model):
    """Shape gradients through every LES model are finite, and chi_c can carry a gradient.

    Two details used to break this: the eddy viscosity was switched off inside solids by an
    in-place write that autograd rejected when chi_c required grad, and Smagorinsky / Vreman
    took a square root at exactly zero strain (infinite derivative -> NaN). The impulsive
    start leaves most of the domain at zero strain, so one step from rest exercises both.
    """
    cfg = _cfg(nx=40, ny=24, h=1 / 6, re=1e5, turbulence=model, bodies=[ImmersedBody(cases.circle(32))])
    ops = make_operators(cfg)
    geom = build_geometry(cfg.bodies, ops)
    chi_c = geom.chi_c.clone().requires_grad_(True)
    chi_u = geom.chi_u.clone().requires_grad_(True)
    s = init_state(cfg, ops)
    for _ in range(2):
        s, c = step(s, geom._replace(chi_c=chi_c, chi_u=chi_u), ops, 0.01)
    c[0].backward()
    for t in (chi_c, chi_u):
        assert t.grad is not None and bool(torch.isfinite(t.grad).all())
    assert float(chi_u.grad.abs().sum()) > 0
