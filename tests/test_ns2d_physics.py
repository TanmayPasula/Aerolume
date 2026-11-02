"""Placed physics of the 2D solver (DESIGN.md 10): heat transfer, buoyancy, emitters,
tracer particles, fans, ports, painted obstacles, live edits and diagnostics.

Fast tests (CPU, float64 unless stated, seconds each). Most of them check an
*exact* discrete identity - a linear conduction profile, an energy or momentum
budget, a mass balance, a particle path in uniform flow - so their tolerances are
round-off, not modelling error. The accuracy of the models against published
benchmarks is in ``tests/test_ns2d_physics_validation.py`` (slow, GPU) and
docs/theory_ns2d.md sections 15-20.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aerolume.solvers.ns2d import (
    DIAG_COLUMNS,
    EmitterSpec,
    FanSpec,
    FlowSolver,
    HeatSourceSpec,
    ImmersedBody,
    PortSpec,
    SolverConfig,
    ThermalConfig,
    WallBC,
    make_operators,
)
from aerolume.solvers.ns2d import physics as phx

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def circle(n: int = 96, d: float = 1.0, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[cx + 0.5 * d * np.cos(th), cy + 0.5 * d * np.sin(th)]


def box(n: int, *, length: float = 1.0, height: float | None = None, re: float = 1.0 / 0.71, **kw) -> SolverConfig:
    """A closed box [0, length] x [0, height] with n cells per unit length (CPU float64, no dye)."""
    height = length if height is None else height
    base = {"nx": round(n * length), "ny": round(n * height), "h": 1.0 / n, "x0": 0.0, "y0": 0.0, "re": re,
            "domain": "closed", "device": "cpu", "dtype": "float64", "dye": False, "perturb": False}
    base.update(kw)
    return SolverConfig(**base)


def tunnel(**kw) -> SolverConfig:
    base = {"nx": 64, "ny": 32, "h": 1 / 8, "x0": -2.0, "y0": -2.0, "re": 100.0, "device": "cpu",
            "dtype": "float64", "dye": False, "perturb": False}
    base.update(kw)
    return SolverConfig(**base)


def run_until(s: FlowSolver, t_end: float, chunk: int = 200) -> None:
    while s.t < t_end:
        s.advance(chunk)


# ---------------------------------------------------------------------------
# off means absent
# ---------------------------------------------------------------------------

def test_default_config_builds_no_physics():
    cfg = tunnel(bodies=[ImmersedBody(circle())], dye=True)
    assert not cfg.has_physics and cfg.dye_channels == 1
    ops = make_operators(cfg)
    assert ops.phys is None
    s = FlowSolver(cfg)
    st = s.state
    assert st.theta is None and st.dye_extra is None and st.ctrl is None
    assert s.physics_history() == {} and s.fans_report() == [] and s.diagnostics().keys() <= {"perf_steps",
                                                                                              "steps_per_s"}
    assert s.heat_transfer()["bodies"] == []
    for name in ("theta", "dye1", "paint", "heat_source"):
        assert float(s.field(name).abs().max()) == 0.0


def test_spec_validation():
    with pytest.raises(ValueError):
        EmitterSpec((0, 0), kind="cone")
    with pytest.raises(ValueError):
        EmitterSpec((0, 0), kind="line")                      # needs an end point
    with pytest.raises(ValueError):
        EmitterSpec((0, 0), channel=4)
    with pytest.raises(ValueError):
        EmitterSpec((0, 0), start=2.0, stop=1.0)
    with pytest.raises(ValueError):
        FanSpec((0, 0), direction=(0, 0))
    with pytest.raises(ValueError):
        PortSpec((0, 0), speed=-1.0)
    with pytest.raises(ValueError):
        WallBC("north")
    with pytest.raises(ValueError):
        SolverConfig(nx=8, ny=8, h=0.1, x0=0, y0=0, re=1, periodic=True, domain="closed")
    rake = EmitterSpec((0, 0), kind="rake", end=(0, 1), count=5)
    assert rake.nozzles().shape == (5, 2) and np.allclose(rake.nozzles()[:, 1], np.linspace(0, 1, 5))
    rect = EmitterSpec((0, 0), kind="rect", size=(2.0, 1.0), count=8)
    noz = rect.nozzles()
    assert len(noz) >= 8 and np.all(np.abs(noz[:, 0]) < 1.0) and np.all(np.abs(noz[:, 1]) < 0.5)
    cfg = tunnel(emitters=[EmitterSpec((0, 0), channel=2)], ports=[PortSpec((0, 1), dye_channel=3)])
    assert cfg.dye_channels == 4 and cfg.has_physics


def _validation_cases():
    """``validation/thermal2d_cases.py`` (not a package), loaded under its own name."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "validation" / "thermal2d_cases.py"
    mod = sys.modules.get("thermal2d_cases")
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("thermal2d_cases", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["thermal2d_cases"] = mod          # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_cylinder_reference_data_and_lange_correlation():
    """The heated-cylinder references the validation quotes are self-consistent: the
    Lange, Durst & Breuer (1998) correlation in its implemented form reproduces the
    value tabulated for them at Re 20 (Notus CFD validation, Pr 0.70) to its four
    decimals, and at Re 40 it lies between the two Pr 0.70 solutions tabulated by
    Sharma & Dhiman (2012). A wrong exponent or coefficient fails both."""
    cases = _validation_cases()
    ref = cases.CYLINDER_NU_REF_PR070
    assert cases.lange_nusselt(20.0) == pytest.approx(ref[20.0]["Lange et al. 1998"], abs=5e-5)
    lo, hi = min(ref[40.0].values()), max(ref[40.0].values())
    assert lo <= cases.lange_nusselt(40.0) <= hi
    assert cases.lange_nusselt(20.0) < cases.lange_nusselt(30.0) < cases.lange_nusselt(40.0)


# ---------------------------------------------------------------------------
# heat transfer: exact discrete identities
# ---------------------------------------------------------------------------

def test_conduction_between_walls_is_exactly_linear():
    """Hot left wall +1/2, cold right -1/2, insulated top / bottom, no buoyancy: the
    discrete steady state is the exact linear profile and the wall heat rates are
    exactly kappa dT / L per unit height (Pe = 1 here, so Nu = 1)."""
    walls = [WallBC("left", "temperature", 0.5), WallBC("right", "temperature", -0.5)]
    s = FlowSolver(box(4, thermal=ThermalConfig(prandtl=0.71, walls=walls)))
    run_until(s, 3.5)
    th = s.field("theta").numpy()
    x = (np.arange(4) + 0.5) / 4
    assert np.allclose(th, (0.5 - x)[None, :], atol=1e-12)
    w = s.heat_transfer()["walls"]
    assert w["left"] == pytest.approx(1.0, rel=1e-10) and w["right"] == pytest.approx(-1.0, rel=1e-10)
    assert w["top"] == 0.0 and w["bottom"] == 0.0


def test_heat_flux_wall_profile_and_balance():
    """Left wall releases the flux G (dimensionless gradient), right wall held at 0:
    theta = G (1 - x) exactly, and both walls carry kappa G per unit height."""
    g = 2.0
    walls = [WallBC("left", "heat_flux", g), WallBC("right", "temperature", 0.0)]
    s = FlowSolver(box(4, thermal=ThermalConfig(prandtl=0.71, walls=walls)))
    run_until(s, 13.0)
    x = (np.arange(4) + 0.5) / 4
    assert np.allclose(s.field("theta").numpy(), g * (1.0 - x)[None, :], atol=1e-10)
    w = s.heat_transfer()["walls"]
    assert w["left"] == pytest.approx(g, rel=1e-12) and w["right"] == pytest.approx(-g, rel=1e-9)


def test_heater_energy_budget_is_exact():
    """An adiabatic closed box with a volumetric heater of power P: the heat content
    grows at exactly P (the source footprint integrates to P, walls are insulated)."""
    p = 0.3
    cfg = box(16, thermal=ThermalConfig(prandtl=0.71),
              heat_sources=[HeatSourceSpec((0.4, 0.55), kind="rect", size=(0.2, 0.3), power=p),
                            HeatSourceSpec((0.7, 0.3), kind="disc", size=(0.1,), power=-0.1)])
    s = FlowSolver(cfg)
    q = s.field("heat_source")
    assert float(q.sum()) * s.ops.h ** 2 == pytest.approx(p - 0.1, rel=1e-12)
    s.advance(37)
    total = float(s.field("theta").sum()) * s.ops.h ** 2
    assert total == pytest.approx((p - 0.1) * s.t, rel=1e-10)


def test_hot_body_heat_rate_balances_the_walls():
    """A penalised cylinder at theta = 1 in a box with cold walls (pure conduction):
    in the steady state the body's heat rate - from the penalisation impulse - equals
    the heat leaving through the four walls (discrete conservation), and its Nusselt
    number is positive and consistent with the reported rate."""
    walls = [WallBC(side, "temperature", 0.0) for side in ("left", "right", "bottom", "top")]
    cyl = ImmersedBody(circle(64, 0.5, 0.5, 0.5), name="hot", thermal="temperature", thermal_value=1.0)
    s = FlowSolver(box(12, bodies=[cyl], thermal=ThermalConfig(prandtl=0.71, walls=walls)))
    run_until(s, 0.9, chunk=400)
    ht = s.heat_transfer()
    body = ht["bodies"][0]
    walls_out = sum(ht["walls"].values())
    assert body["heat_rate"] > 0 and walls_out < 0
    assert body["heat_rate"] == pytest.approx(-walls_out, rel=1e-6)
    assert body["nusselt"] == pytest.approx(ht["pe"] * body["heat_rate"] / (1.0 * body["perimeter"]), rel=1e-12)
    th = s.field("theta")
    assert float(th.min()) >= -1e-12 and float(th.max()) <= 1.0 + 1e-12          # maximum principle
    bal = s.heat_balance()
    assert bal["heat_in_bodies"] == pytest.approx(body["heat_rate"])
    assert abs(bal["heat_balance_residual"]) < 1e-6 * bal["heat_in_bodies"]


def test_energy_balance_closes_in_a_tunnel():
    """Every heat path at once - a hot body, a heat-flux body, a heater, a hot jet, a
    vent, a heated and a cooled wall, inflow and outflow, buoyancy: once the run is
    steady the reported terms balance (in = out + storage) to ~1e-6 of the heat input."""
    hot = ImmersedBody(circle(64, 0.6, -0.5, 0.0), name="hot", thermal="temperature", thermal_value=1.0)
    chip = ImmersedBody(circle(48, 0.4, 1.0, 0.8), name="chip", thermal="heat_flux", thermal_value=2.0)
    cfg = tunnel(nx=48, ny=24, h=1 / 6, re=50.0, bodies=[hot, chip],
                 thermal=ThermalConfig(prandtl=0.71, richardson=0.3, walls=[WallBC("bottom", "temperature", 0.5),
                                                                              WallBC("top", "heat_flux", -0.5)]),
                 heat_sources=[HeatSourceSpec((2.0, -1.0), size=(0.4, 0.4), power=0.2)],
                 ports=[PortSpec((0.5, -1.9), direction=(0.0, 1.0), size=0.3, speed=1.0, theta=1.5),
                        PortSpec((3.0, 1.9), direction=(0.0, 1.0), size=0.3, speed=0.5, kind="vent")])
    s = FlowSolver(cfg)
    s.advance(1100)
    b = s.heat_balance()
    total_in = sum(v for k, v in b.items() if k.startswith("heat_in"))
    assert b["heat_in_sources"] == pytest.approx(0.2, rel=1e-12)
    assert b["heat_in_bodies"] == pytest.approx(s.heat_transfer()["bodies"][0]["heat_rate"] + 2.0 * float(
        s.geometry.thermal.perimeter[1]) / s.peclet)
    assert b["heat_in_ports"] > 0 and b["heat_out_outlet"] > 0
    assert abs(b["heat_balance_residual"]) < 1e-5 * total_in


def test_temperature_stays_bounded_in_a_tunnel():
    """Regression: with the momentum limits alone (advective CFL and 0.25 h^2/kappa
    separately) a heated cylinder's temperature grew without bound once both were
    active; the positivity limit keeps theta within [0, 1] (maximum principle)."""
    cyl = ImmersedBody(circle(96), name="cyl", thermal="temperature", thermal_value=1.0)
    s = FlowSolver(tunnel(nx=96, ny=64, h=1 / 16, re=40.0, bodies=[cyl], dtype="float32",
                          thermal=ThermalConfig(prandtl=0.71)))
    s.advance(300)
    th = s.field("theta")
    assert torch.isfinite(th).all()
    assert float(th.min()) >= -1e-5 and float(th.max()) <= 1.0 + 1e-5
    assert s.heat_transfer()["bodies"][0]["nusselt"] > 1.0


def test_diffusion_coefficient_rule():
    assert phx.diffusion_coefficient(("adiabatic",) * 4) == 4.0
    assert phx.diffusion_coefficient(("inlet", "outlet", "adiabatic", "adiabatic")) == 5.0
    assert phx.diffusion_coefficient(("temperature", "temperature", "adiabatic", "adiabatic")) == 5.0
    assert phx.diffusion_coefficient(("temperature", "adiabatic", "temperature", "adiabatic")) == 6.0
    assert phx.diffusion_coefficient(("periodic",) * 4) == 4.0


def test_heat_flux_body_releases_its_heat():
    """A heat-flux body releases exactly G P / Pe: in an adiabatic box the heat content
    grows at that rate from the first step (the skin source is normalised per body)."""
    g = 1.5
    body = ImmersedBody(circle(64, 0.3, 0.5, 0.5), name="chip", thermal="heat_flux", thermal_value=g)
    s = FlowSolver(box(24, bodies=[body], thermal=ThermalConfig(prandtl=0.71)))
    s.advance(25)
    rate = g * s.geometry.thermal.perimeter[0].item() / s.peclet
    assert s.heat_transfer()["bodies"][0]["heat_rate"] == pytest.approx(rate, rel=1e-12)
    total = float(s.field("theta").sum()) * s.ops.h ** 2
    assert total == pytest.approx(rate * s.t, rel=1e-9)
    assert s.heat_transfer()["bodies"][0]["theta_wall"] > 0


# ---------------------------------------------------------------------------
# buoyancy
# ---------------------------------------------------------------------------

def test_stable_stratification_stays_at_rest():
    """Hot on top, cold below: the Boussinesq force is a discrete gradient that the
    projection removes exactly, so the fluid stays at rest to round-off."""
    walls = [WallBC("bottom", "temperature", -0.5), WallBC("top", "temperature", 0.5)]
    s = FlowSolver(box(16, re=50.0, thermal=ThermalConfig(prandtl=0.71, richardson=1.0, walls=walls)))
    y = (np.arange(16) + 0.5) / 16
    s.set_temperature(np.repeat((y - 0.5)[:, None], 16, axis=1))
    s.advance(100)
    assert float(s.state.u.abs().max()) < 1e-12 and float(s.state.v.abs().max()) < 1e-12


@pytest.mark.parametrize("gravity,axis,sign", [((0.0, -1.0), 1, +1), ((1.0, 0.0), 0, -1)])
def test_hot_blob_rises_against_gravity(gravity, axis, sign):
    s = FlowSolver(box(32, re=200.0, thermal=ThermalConfig(prandtl=0.71, richardson=1.0, gravity=gravity)))
    xc = (np.arange(32) + 0.5) / 32
    X, Y = np.meshgrid(xc, xc)
    s.set_temperature(np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.01))
    s.advance(20)
    vel = (s.field("u"), s.field("v"))[axis].numpy()
    assert sign * vel[15:17, 15:17].mean() > 1e-3
    other = (s.field("v"), s.field("u"))[axis].numpy()
    assert abs(other[15:17, 15:17].mean()) < 1e-10                 # symmetric about the blob's axis


def test_live_thermal_edits_in_place():
    walls = [WallBC("left", "temperature", 0.5), WallBC("right", "temperature", -0.5)]
    s = FlowSolver(box(16, re=40.0, thermal=ThermalConfig(prandtl=0.71, richardson=1.0, walls=walls)))
    ph = s.physics
    s.advance(5)
    s.set_thermal(richardson=2.0)
    s.set_inflow(re=80.0)
    assert s.physics is ph                                           # same structure: tensors rewritten
    assert float(ph.ri) == 2.0 and float(ph.kappa) == pytest.approx(1.0 / (80.0 * 0.71))
    s.set_thermal(walls=[WallBC("left", "temperature", 1.0), WallBC("right", "temperature", -1.0)])
    assert s.physics is ph and np.allclose(ph.wall_values.numpy()[:2], [1.0, -1.0])
    s.advance(5)
    assert torch.isfinite(s.field("theta")).all()


# ---------------------------------------------------------------------------
# smoke emitters and tracer particles
# ---------------------------------------------------------------------------

def test_dye_emitters_channels_timing_and_solids():
    em = [EmitterSpec((-1.5, 0.0), kind="point", channel=0, rate=0.8),
          EmitterSpec((-1.5, -1.0), kind="line", end=(-1.5, -0.6), channel=1, start=0.5),
          EmitterSpec((-1.5, 1.0), kind="point", channel=2, pulse_period=0.4, pulse_duty=0.5, stop=10.0)]
    cyl = ImmersedBody(circle(64, 0.6, 0.0, 0.0))
    s = FlowSolver(tunnel(bodies=[cyl], emitters=em, dye=True, dye_inlet=False))
    assert s.state.dye_extra.shape[0] == 2                          # channels 1 and 2
    s.advance(5)
    assert s.t < 0.5
    j, i = 16, 4                                                    # the cell at (-1.5, 0)
    assert float(s.field("dye")[j, i]) == pytest.approx(0.8, abs=0.05)
    assert float(s.field("dye1").max()) == 0.0                      # not started yet
    run_until(s, 3.0, chunk=50)
    for name in ("dye", "dye1", "dye2"):
        f = s.field(name)
        assert float(f.min()) >= 0.0 and float(f.max()) <= 1.0
        assert float(f.max()) > 0.3, name
    assert float(s.field("dye")[:, : i - 2].max()) == 0.0           # no inlet comb, nothing upstream
    solid = s.field("mask") > 0.5
    for name in ("dye", "dye1", "dye2"):
        assert float(s.field(name)[solid].max()) == 0.0
    # emitter clock: on during the first half of every period, off before start / after stop
    timing = torch.tensor([1.0, 3.0, 0.4, 0.5], dtype=torch.float64)
    on = [float(phx.emitter_on(torch.tensor(t, dtype=torch.float64), timing)) for t in (0.5, 1.1, 1.3, 2.9, 3.1)]
    assert on == [0.0, 1.0, 0.0, 0.0, 0.0]


def test_particles_follow_uniform_flow_exactly():
    """In the uniform tunnel flow (no bodies) RK2 is exact: x = x_e + age, y = y_e."""
    em = [EmitterSpec((-1.0, 0.25), kind="point", species="particles", rate=20.0)]
    s = FlowSolver(tunnel(emitters=em))
    s.advance(60)
    p = s.particles()
    released = math.floor(20.0 * s.t + 1e-9)
    assert len(p["x"]) == pytest.approx(released, abs=1)
    assert np.allclose(p["x"], -1.0 + p["age"], atol=1e-9) and np.allclose(p["y"], 0.25, atol=1e-12)
    lines = s.particle_lines()
    assert len(lines) == 1 and lines[0].shape[0] == len(p["x"])
    assert np.all(np.diff(lines[0][:, 0]) > 0)                       # ordered from the nozzle outwards


def test_particles_recycle_and_die_in_solids_and_outlet():
    em = [EmitterSpec((-1.8, 0.0), kind="rake", end=(-1.8, 0.5), count=3, species="particles", rate=60.0)]
    cyl = ImmersedBody(circle(64, 0.8, 0.0, 0.0))
    s = FlowSolver(tunnel(bodies=[cyl], emitters=em, particle_capacity=48))
    assert s._particles.ring == 16 and s._particles.capacity == 48
    s.advance(400)
    p = s.particles()
    assert 0 < len(p["x"]) <= 48
    assert np.all(p["x"] < 6.0) and np.all(p["x"] > -2.0)
    m = s.field("mask").numpy()
    i = np.clip(((p["x"] + 2.0) * 8).astype(int), 0, 63)
    j = np.clip(((p["y"] + 2.0) * 8).astype(int), 0, 31)
    assert np.all(m[j, i] <= 0.5)                                    # nothing lives inside the solid
    assert np.max(p["age"]) < s.t                                    # old particles were recycled


def test_pulsed_particle_rake_draws_timelines():
    em = [EmitterSpec((-1.5, -0.5), kind="rake", end=(-1.5, 0.5), count=5, species="particles",
                      pulse_period=0.5, pulse_duty=0.2)]
    s = FlowSolver(tunnel(emitters=em))
    s.advance(26)                                                    # dt = 1/16: t = 1.625
    p = s.particles()
    assert s.t == pytest.approx(1.625)
    assert len(p["x"]) == 5 * 4                                      # one batch per period: t ~ 0, 0.5, 1, 1.5
    assert len(np.unique(np.round(p["birth"], 9))) == 4
    lines = s.particle_lines()
    assert len(lines) == 4 and all(len(line) == 5 for line in lines)
    for line in lines:                                               # a timeline is a straight vertical line here
        assert np.ptp(line[:, 0]) < 1e-9


def test_live_emitter_edits_keep_the_structure():
    em = [EmitterSpec((-1.5, 0.0), channel=1), EmitterSpec((-1.5, 0.5), species="particles", rate=10.0)]
    s = FlowSolver(tunnel(emitters=em, dye=True, dye_inlet=False))
    s.advance(10)
    ph, pool = s.physics, s._particles
    moved = [EmitterSpec((-1.0, 0.0), channel=1), EmitterSpec((-1.0, 0.5), species="particles", rate=10.0)]
    s.set_emitters(moved)
    assert s.physics is ph and s._particles is pool                  # rewritten in place
    s.advance(3)
    assert float(s.field("dye1")[16, 8]) > 0.5                       # the cell at (-1, 0)
    s.set_emitters(moved + [EmitterSpec((-1.0, -1.0), channel=3)])   # a new channel: rebuilt
    assert s.state.dye_extra.shape[0] == 3
    s.advance(3)
    assert float(s.field("dye3").max()) > 0.5


# ---------------------------------------------------------------------------
# fans
# ---------------------------------------------------------------------------

def test_fan_adds_exactly_its_thrust_in_a_periodic_box():
    """Periodic box (no walls, no outlet): the mean momentum changes only by the fan
    force, so d/dt int u dA = T n exactly (advection, diffusion and pressure all
    integrate to zero on the periodic grid)."""
    fan = FanSpec((2.0, 1.0), direction=(0.8, 0.6), width=0.8, mode="thrust", value=0.3)
    cfg = SolverConfig(nx=32, ny=16, h=1 / 8, x0=0.0, y0=0.0, re=50.0, periodic=True, device="cpu",
                       dtype="float64", dye=False, perturb=False, fans=[fan])
    s = FlowSolver(cfg)
    s.advance(25)
    h2 = s.ops.h ** 2
    px = float(s.state.u[:, :-1].sum()) * h2
    py = float(s.state.v[:-1].sum()) * h2
    assert px == pytest.approx(0.3 * 0.8 * s.t, rel=1e-10)
    assert py == pytest.approx(0.3 * 0.6 * s.t, rel=1e-10)
    rep = s.fans_report()[0]
    assert rep["thrust"] == pytest.approx(0.3) and rep["pressure_jump"] == pytest.approx(0.3 / 0.8)


def test_velocity_regulated_fan_reaches_its_target():
    fan = FanSpec((-0.5, 0.0), direction=(1.0, 0.0), width=1.0, mode="velocity", value=1.5, name="blower")
    s = FlowSolver(tunnel(fans=[fan], dtype="float32"))
    run_until(s, 15.0, chunk=250)
    rep = s.fans_report()[0]
    assert rep["name"] == "blower" and rep["mode"] == "velocity"
    assert rep["disc_velocity"] == pytest.approx(1.5, rel=0.005)
    assert rep["thrust"] > 0 and rep["induced_velocity"] == pytest.approx(0.5, abs=0.01)
    ud = s.physics_history()["disc_velocity:blower"]
    assert ud.max() < 1.5 * 1.05                                     # no large overshoot
    hist = s.physics_history()
    assert "thrust:blower" in hist and len(hist["t"]) == s.steps


# ---------------------------------------------------------------------------
# ports (jets and vents)
# ---------------------------------------------------------------------------

def test_jet_and_vent_in_a_closed_box_conserve_mass():
    jet = PortSpec((0.3, 0.5), direction=(1.0, 0.0), size=0.2, speed=1.0, kind="jet", theta=1.0, dye_channel=1)
    vent = PortSpec((0.8, 0.5), direction=(1.0, 0.0), size=0.4, speed=0.5, kind="vent")
    cfg = box(32, re=100.0, ports=[jet, vent], dye=True, thermal=ThermalConfig(prandtl=0.71), diagnostics=True)
    s = FlowSolver(cfg)
    assert float(s.physics.q_net) == pytest.approx(0.0, abs=1e-12)
    s.advance(60)
    div = s.field("divergence").numpy()
    src = s.physics.src.numpy()
    assert np.abs(div - src).max() < 1e-10                           # div u = the port sources, exactly
    assert div.sum() * s.ops.h ** 2 == pytest.approx(0.0, abs=1e-10)  # closed: what the jet adds, the vent removes
    # the jet delivers exactly speed x size through the first face line past its open face
    # (x = 0.3125), at about its speed, and carries its temperature and smoke
    u = s.state.u
    assert float(u[:, 10].sum()) * s.ops.h == pytest.approx(1.0 * 0.2, rel=1e-9)
    assert u[16, 10].item() == pytest.approx(1.0, abs=0.03)
    assert float(s.field("theta")[16, 8]) == pytest.approx(1.0, abs=0.03)
    assert float(s.field("dye1")[16, 11]) > 0.9
    assert s.latest_diagnostics()["p_residual"] < 1e-10


def test_unbalanced_port_in_a_closed_box_keeps_the_projection_exact():
    """A lone jet in a closed box breaks the compatibility of the Neumann Poisson
    problem; the solver spreads the excess as a uniform sink, so the solve stays exact."""
    jet = PortSpec((0.3, 0.5), direction=(0.0, 1.0), size=0.2, speed=1.0)
    s = FlowSolver(box(32, re=100.0, ports=[jet], diagnostics=True))
    s.advance(40)
    d = s.latest_diagnostics()
    assert d["p_residual"] < 1e-10 and d["div_rms"] < 1e-10
    assert torch.isfinite(s.state.u).all()


def test_jet_in_a_tunnel_balances_at_the_outlet():
    jet = PortSpec((0.0, -1.0), direction=(0.0, 1.0), size=0.25, speed=2.0)
    s = FlowSolver(tunnel(ports=[jet]))
    s.advance(30)
    u = s.state.u
    inflow, outflow = float(u[:, 0].sum()) * s.ops.h, float(u[:, -1].sum()) * s.ops.h
    assert outflow - inflow == pytest.approx(2.0 * 0.25, rel=1e-10)


# ---------------------------------------------------------------------------
# painted obstacles (explore mode)
# ---------------------------------------------------------------------------

def test_paint_erase_and_clear():
    """A painted disc is the same obstacle as an ImmersedBody of that outline: same
    calibrated interface on every grid, same drag (painted_force uses the same
    penalisation impulse), the same small residual velocity inside."""
    s = FlowSolver(tunnel(dtype="float32"))
    geom_before = s.geometry
    s.paint_disc((0.0, 0.0), 0.4)
    m = s.field("mask")
    assert float(m[16, 16]) == 1.0 and float(m[16, 4]) == 0.0
    ref = FlowSolver(tunnel(dtype="float32", bodies=[ImmersedBody(circle(256, 0.8))]))
    assert torch.allclose(m, ref.field("mask"), atol=0.02)
    s.advance(80)
    ref.advance(80)
    cd, cl = s.painted_force()
    assert cd == pytest.approx(ref.coefficients()[0], rel=0.01) and abs(cl) < 0.05
    speed = torch.hypot(s.field("u"), s.field("v"))
    assert float(speed[16, 16]) < 1.2 * float(torch.hypot(ref.field("u"), ref.field("v"))[16, 16]) + 1e-3
    s.paint_disc((0.0, 0.0), 0.3, erase=True)
    assert float(s.field("mask")[16, 16]) == 0.0 and float(s.field("paint").max()) > 0.3   # a thin ring is left
    s.paint_polygon(np.array([[1.0, -0.5], [1.5, -0.5], [1.5, 0.5], [1.0, 0.5]]))
    assert float(s.field("mask")[16, 26]) == 1.0
    s.clear_paint()
    assert float(s.field("paint").max()) == 0.0
    assert s.geometry is geom_before                                 # the differentiable bodies are untouched
    s.advance(5)
    assert torch.isfinite(s.field("u")).all()


def test_paint_survives_a_state_dict_roundtrip():
    s = FlowSolver(tunnel(dtype="float64", thermal=ThermalConfig(prandtl=0.71),
                          emitters=[EmitterSpec((-1.5, 0.0), channel=1)], dye=True))
    s.paint_disc((0.5, 0.0), 0.3)
    s.advance(15)
    d = s.state_dict()
    t = FlowSolver(tunnel(dtype="float64", thermal=ThermalConfig(prandtl=0.71),
                          emitters=[EmitterSpec((-1.5, 0.0), channel=1)], dye=True))
    t.load_state_dict(d)
    for name in ("u", "theta", "dye1", "paint"):
        assert torch.equal(s.field(name), t.field(name)), name
    s.advance(3)
    t.advance(3)
    assert torch.allclose(s.state.u, t.state.u, atol=1e-12)


def test_live_source_edits_keep_totals_exact():
    heater = HeatSourceSpec((0.5, 0.5), size=(0.2, 0.2), power=0.5)
    fan = FanSpec((0.5, 0.3), direction=(0.0, 1.0), width=0.4, value=0.1)
    s = FlowSolver(box(16, re=50.0, thermal=ThermalConfig(prandtl=0.71), heat_sources=[heater], fans=[fan]))
    s.advance(3)
    ph = s.physics
    s.set_sources(heat_sources=[HeatSourceSpec((0.37, 0.61), size=(0.2, 0.2), power=0.8)],
                  fans=[FanSpec((0.6, 0.3), direction=(0.0, 1.0), width=0.4, value=0.2)])
    assert s.physics is ph                                           # moved and retuned in place
    assert float(s.field("heat_source").sum()) * s.ops.h ** 2 == pytest.approx(0.8, rel=1e-12)
    s.advance(3)
    assert s.fans_report()[0]["thrust"] == pytest.approx(0.2)
    s.set_sources(fans=[])                                           # removing the fan changes the structure
    assert s.physics is not ph and s.fans_report() == []
    s.advance(3)
    assert torch.isfinite(s.field("u")).all()


# ---------------------------------------------------------------------------
# diagnostics, solver trace, PID time step
# ---------------------------------------------------------------------------

def test_diagnostics_history_and_csv(tmp_path):
    walls = [WallBC("left", "temperature", 0.5), WallBC("right", "temperature", -0.5)]
    s = FlowSolver(box(16, re=60.0, diagnostics=True,
                       thermal=ThermalConfig(prandtl=0.71, richardson=1.0, walls=walls)))
    s.advance(45)
    d = s.diagnostics()
    assert set(DIAG_COLUMNS) <= set(d)
    assert list(d["step"]) == [10, 20, 30, 40]
    assert np.all(d["div_rms"] < 1e-12) and np.all(d["p_residual"] < 1e-10)
    assert np.all(d["cfl"] <= 0.5 + 1e-9) and np.all(d["dt"] > 0)
    assert np.all(np.isfinite(d["dtheta_rel"])) and np.all(d["du_l2"] >= 0)
    assert len(d["steps_per_s"]) == 1 and d["steps_per_s"][0] > 0
    path = s.diagnostics_csv(tmp_path / "diag.csv")
    assert path == str(tmp_path / "diag.csv")
    lines = (tmp_path / "diag.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == list(DIAG_COLUMNS) and len(lines) == 5
    s.reset()
    assert len(s.diagnostics().get("t", [])) == 0


def test_trace_step_stages_are_consistent():
    cyl = ImmersedBody(circle(64, 0.6), thermal="temperature", thermal_value=1.0)
    s = FlowSolver(tunnel(bodies=[cyl], thermal=ThermalConfig(prandtl=0.71, richardson=0.5)))
    s.advance(5)
    u_before, t_before = s.state.u.clone(), s.t
    tr = s.trace_step()
    assert s.t == t_before and torch.equal(s.state.u, u_before)      # commit=False: the run is untouched
    for k in (1, 2, 3):
        p = f"stage{k}/"
        for key in ("explicit_u", "penalised_v", "divergence", "pressure_potential", "correction_u",
                    "projected_u", "explicit_theta", "theta"):
            assert p + key in tr, p + key
        assert torch.allclose(tr[p + "projected_u"], tr[p + "penalised_u"] + tr[p + "correction_u"], atol=1e-12)
    assert torch.allclose(tr["stage3/projected_u"], tr["final_u"])
    s2 = FlowSolver(tunnel(bodies=[cyl], thermal=ThermalConfig(prandtl=0.71, richardson=0.5)))
    s2.advance(5)
    s2.advance(1)
    s.trace_step(commit=True)
    assert s.steps == 6 and torch.allclose(s.state.u, s2.state.u, atol=1e-12)


def test_pid_time_step_is_smooth_and_stable():
    cyl = ImmersedBody(circle(64, 0.8))
    s = FlowSolver(tunnel(bodies=[cyl], adaptive_dt="pid", perturb=True, dtype="float32", diagnostics=True))
    s.advance(300)
    dt = np.diff(np.r_[0.0, s.history()["t"]])
    growth = dt[1:] / dt[:-1]
    assert growth.max() <= 1.03 + 1e-4                               # PID_MAX_GROWTH
    assert np.all(s.diagnostics()["cfl"] <= 0.5 * (1 + 1e-4))       # never above the stability limit
    plain = FlowSolver(tunnel(bodies=[cyl], perturb=True, dtype="float32"))
    plain.advance(300)
    assert s.t == pytest.approx(plain.t, rel=0.15)                   # about 0.9 x the limit, on average


# ---------------------------------------------------------------------------
# CUDA graphs with every feature on
# ---------------------------------------------------------------------------

def _everything(graphs: bool) -> FlowSolver:
    bodies = [ImmersedBody(circle(96), name="hot", thermal="temperature", thermal_value=1.0),
              ImmersedBody(circle(64, 0.5, 2.0, 1.0), name="flux", thermal="heat_flux", thermal_value=2.0)]
    em = [EmitterSpec((-1.5, -0.5), kind="rake", end=(-1.5, 0.5), count=5, channel=0),
          EmitterSpec((-1.5, 1.2), kind="point", channel=1, pulse_period=1.0, pulse_duty=0.4),
          EmitterSpec((-1.5, -1.0), kind="line", end=(-1.5, -1.4), species="particles", rate=40.0, count=4)]
    cfg = SolverConfig(nx=256, ny=128, h=1 / 16, x0=-2.0, y0=-4.0, re=100.0, bodies=bodies, device="cuda",
                       dtype="float32", use_cuda_graphs=graphs, moment_ref=(0.0, 0.0), dye_inlet=False,
                       emitters=em, thermal=ThermalConfig(prandtl=0.71, richardson=0.5,
                                                          walls=[WallBC("bottom", "temperature", 0.2)]),
                       heat_sources=[HeatSourceSpec((5.0, -1.0), size=(0.5, 0.5), power=0.3)],
                       fans=[FanSpec((6.0, 1.5), width=1.0, mode="velocity", value=1.6),
                             FanSpec((6.0, -2.5), width=0.8, mode="thrust", value=0.3)],
                       ports=[PortSpec((9.0, 2.0), direction=(0.0, -1.0), size=0.4, speed=1.0, theta=1.0,
                                       dye_channel=2),
                              PortSpec((9.0, -3.0), direction=(0.0, -1.0), size=0.4, speed=0.5, kind="vent")],
                       diagnostics=True, adaptive_dt="pid")
    return FlowSolver(cfg)


@pytest.mark.gpu
@needs_cuda
def test_cuda_graphs_match_eager_with_all_physics():
    g, e = _everything(True), _everything(False)
    for s in (g, e):
        s.advance(23)
        s.start_averaging()
        s.advance(17)
    assert g.uses_cuda_graphs and g.graph_error is None
    for name in ("u", "v", "p", "theta", "dye", "dye1", "dye2"):
        assert torch.allclose(g.field(name), e.field(name), atol=1e-5), name
    assert np.allclose(g.coefficients(), e.coefficients(), atol=1e-5)
    hg, he = g.heat_transfer(mean=True), e.heat_transfer(mean=True)
    assert hg["bodies"][0]["nusselt"] == pytest.approx(he["bodies"][0]["nusselt"], rel=1e-5)
    assert len(g.particles()["x"]) == len(e.particles()["x"]) > 0
    # live edits rewrite tensors in place: no re-capture
    n = len(g._graphs)
    g.set_sources(heat_sources=[HeatSourceSpec((5.3, -1.0), size=(0.5, 0.5), power=0.6)])
    g.set_inflow(re=150.0)
    g.set_thermal(richardson=0.8)
    g.paint_disc((3.5, -2.0), 0.3)                                   # first stroke: a structure change
    g.advance(20)
    n = len(g._graphs)
    g.paint_disc((3.5, -1.5), 0.3)
    g.advance(20)
    assert len(g._graphs) == n and g.uses_cuda_graphs
    assert torch.isfinite(g.field("u")).all() and g.painted_force()[0] != 0.0
