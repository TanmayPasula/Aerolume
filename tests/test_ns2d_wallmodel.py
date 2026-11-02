"""Werner-Wengle wall model of the 2D solver's domain walls (theory doc 18.4).

Fast CPU tests. They check the law itself (zero excess in the viscous sublayer,
the power law above it, continuity at y+ = 11.81), that the force reaches exactly
the faces next to the no-slip walls with the stress / h it should, that it is
opt-in (default off: every validated run is unchanged), that it only ever removes
kinetic energy, that it is differentiable, and that a fan room runs stably with it.
The grid study of the 2D fan room with and without it - and why it stays opt-in -
is ``run_fan_room`` in validation/thermal2d_cases.py and theory doc 18.4.
"""
from __future__ import annotations

import math

import pytest
import torch

from aerolume.solvers.ns2d import FanSpec, FlowSolver, SolverConfig, make_operators
from aerolume.solvers.ns2d.core import (
    WW_A,
    WW_B,
    WW_YPLUS_CROSS,
    momentum_rhs,
    wall_model_force,
    wall_shear_excess,
)
from aerolume.solvers.ns2d.geometry import build_geometry


def room(n: int = 8, *, re: float = 3e4, turbulence: str = "wale", **kw) -> SolverConfig:
    """A closed 4 x 3 box with n cells per unit length (CPU float64)."""
    base = {"nx": 4 * n, "ny": 3 * n, "h": 1.0 / n, "x0": 0.0, "y0": 0.0, "re": re, "domain": "closed",
            "turbulence": turbulence, "device": "cpu", "dtype": "float64", "dye": False, "perturb": False,
            "use_cuda_graphs": False, "bc_top": "noslip", "bc_bottom": "noslip", "wall_model": "werner_wengle"}
    base.update(kw)
    return SolverConfig(**base)


def test_law_is_zero_in_the_sublayer_and_the_power_law_above():
    nu, y = 1e-4, 0.05
    # u+ = y+ in the sublayer: a velocity whose y+ (from the linear law) is below 11.81
    yplus = torch.tensor([0.5, 3.0, 11.0, 20.0, 100.0, 1000.0], dtype=torch.float64)
    # velocity that the *power law* assigns to these y+ (u_tau = y+ nu / y)
    ut = yplus * nu / y
    u_pow = ut * WW_A * yplus ** WW_B
    u_lin = ut * yplus
    us = torch.where(yplus < WW_YPLUS_CROSS, u_lin, u_pow)
    ex = wall_shear_excess(us, y, nu)
    tau_total = ex + us.abs() * nu / y
    sub = yplus < WW_YPLUS_CROSS
    assert torch.all(ex[sub] == 0.0)
    # above the crossover the total stress is u_tau^2 of the power law, to round-off
    torch.testing.assert_close(tau_total[~sub], (ut * ut)[~sub], rtol=1e-12, atol=0.0)
    # sign follows the velocity, magnitude is even in it
    torch.testing.assert_close(wall_shear_excess(-us, y, nu), -ex)


def test_power_law_meets_the_linear_law_at_the_crossover():
    assert WW_YPLUS_CROSS == pytest.approx(11.81, abs=0.01)
    nu, y = 2e-5, 0.03
    ut = WW_YPLUS_CROSS * nu / y
    u = torch.tensor([ut * WW_YPLUS_CROSS], dtype=torch.float64)
    assert float(wall_shear_excess(u, y, nu)) == pytest.approx(0.0, abs=1e-15)


def test_force_acts_only_next_to_the_no_slip_walls_with_stress_over_h():
    cfg = room(re=1e5)
    ops = make_operators(cfg)
    assert ops.wall_model
    g = torch.Generator().manual_seed(3)
    u = torch.randn(cfg.ny, cfg.nx + 1, generator=g, dtype=torch.float64)
    v = torch.randn(cfg.ny + 1, cfg.nx, generator=g, dtype=torch.float64)
    fu, fv = wall_model_force(u, v, ops)
    y = 0.5 * cfg.h
    torch.testing.assert_close(fu[0], -wall_shear_excess(u[0], y, ops.nu) / cfg.h)
    torch.testing.assert_close(fu[-1], -wall_shear_excess(u[-1], y, ops.nu) / cfg.h)
    torch.testing.assert_close(fv[:, 0], -wall_shear_excess(v[:, 0], y, ops.nu) / cfg.h)
    torch.testing.assert_close(fv[:, -1], -wall_shear_excess(v[:, -1], y, ops.nu) / cfg.h)
    assert torch.all(fu[1:-1] == 0) and torch.all(fv[:, 1:-1] == 0)
    assert float(fu[0].abs().sum()) > 0           # Re 1e5 on 8 cells/L: far into the log layer
    # wired into the tendency: on minus off is the force on the interior faces (walls stay Dirichlet)
    geom = build_geometry([], ops)
    lu_on, lv_on, _ = momentum_rhs(u, v, geom, ops)
    ops.wall_model = False
    lu_off, lv_off, _ = momentum_rhs(u, v, geom, ops)
    torch.testing.assert_close(lu_on - lu_off, torch.cat([torch.zeros_like(fu[:, :1]), fu[:, 1:-1],
                                                          torch.zeros_like(fu[:, :1])], dim=1))
    torch.testing.assert_close(lv_on - lv_off, torch.cat([torch.zeros_like(fv[:1]), fv[1:-1],
                                                          torch.zeros_like(fv[:1])], dim=0))


def test_slip_walls_and_moving_road():
    cfg = room(re=1e5, bc_left="slip", bc_top="slip")
    ops = make_operators(cfg)
    u = torch.full((cfg.ny, cfg.nx + 1), 0.7, dtype=torch.float64)
    v = torch.full((cfg.ny + 1, cfg.nx), 0.4, dtype=torch.float64)
    fu, fv = wall_model_force(u, v, ops)
    assert torch.all(fu[-1] == 0) and torch.all(fv[:, 0] == 0)
    assert torch.all(fu[0] < 0) and torch.all(fv[:, -1] < 0)
    # a road moving with the air exerts no excess shear; one moving faster pushes it along
    t = SolverConfig(nx=16, ny=8, h=0.25, x0=0.0, y0=0.0, re=1e5, bc_bottom="moving", ground_speed=0.7,
                     turbulence="wale", wall_model="werner_wengle", device="cpu", dtype="float64", dye=False)
    ot = make_operators(t)
    fu, fv = wall_model_force(u, v[:, :16], ot)
    assert fv is None and torch.all(fu[0] == 0)
    fu, _ = wall_model_force(u * 0.5, v[:, :16], ot)
    assert torch.all(fu[0] > 0)


def test_opt_in_only():
    """Off unless asked for: the planner's configs and every validated case run exactly as before."""
    assert SolverConfig(nx=8, ny=8, h=0.1, x0=0, y0=0, re=1e5, domain="closed", turbulence="wale").wall_model == "none"
    assert not make_operators(room(wall_model="none")).wall_model
    assert make_operators(room()).wall_model
    assert make_operators(room(domain="tunnel")).wall_model         # explicit: tunnel walls too
    with pytest.raises(ValueError, match="wall_model"):
        room(wall_model="auto")


def test_only_removes_kinetic_energy():
    """The force opposes the wall-relative velocity face by face, so its power is never positive."""
    ops = make_operators(room(re=1e6))
    g = torch.Generator().manual_seed(11)
    for _ in range(5):
        u = 3.0 * torch.randn(ops.ny, ops.nx + 1, generator=g, dtype=torch.float64)
        v = 3.0 * torch.randn(ops.ny + 1, ops.nx, generator=g, dtype=torch.float64)
        fu, fv = wall_model_force(u, v, ops)
        assert torch.all(fu * u <= 0) and torch.all(fv * v <= 0)
        assert float((fu * u).sum() + (fv * v).sum()) < 0


def test_differentiable():
    ops = make_operators(room(n=4, re=1e5))
    u = torch.randn(ops.ny, ops.nx + 1, dtype=torch.float64, requires_grad=True)
    v = torch.randn(ops.ny + 1, ops.nx, dtype=torch.float64)
    torch.autograd.gradcheck(lambda x: wall_model_force(x, v, ops)[0], (u,))


def test_fan_room_runs_and_the_wall_takes_power():
    """A fan room with the model: stable, momentum-free walls stay Dirichlet, and the modelled
    wall shear does real (negative) work on the spun-up flow."""
    fan = FanSpec(position=(4 / 3, 1.5), direction=(1.0, 0.0), width=1.0, mode="thrust", value=0.6)
    s = FlowSolver(room(n=8, re=3e4, fans=[fan]))
    while s.t < 40.0:
        s.advance(200)
    st = s.state
    assert torch.isfinite(st.u).all() and torch.isfinite(st.v).all()
    assert torch.all(st.u[:, 0] == 0) and torch.all(st.u[:, -1] == 0)
    assert torch.all(st.v[0] == 0) and torch.all(st.v[-1] == 0)
    fu, fv = wall_model_force(st.u, st.v, s.ops)
    power = float((fu * st.u).sum() + (fv * st.v).sum()) * s.cfg.h ** 2
    fan_power = s.fans_report()[0]["power"]
    assert power < 0 and math.isfinite(fan_power)
