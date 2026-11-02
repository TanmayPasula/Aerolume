"""Regression tests for the physics findings of critic round 5 (each measured before the fix).

* 3D heaters warming still fluid: the step rule had no bound from the sources, so a
  300 W box heater in a still Draft room took a 68.7 L/U first step and was NaN on
  step 2. The rule now caps dt by 1 / S_max and by the buoyant parcel heated from rest
  (``ns3d.physics.Physics3D.dt_rule``), as the 2D solver does.
* Capped-Reynolds 2D heat rates were in the more conductive simulated fluid's unit:
  53x the rate the reported Nusselt number gives for a NACA 0012 capped at Re 2500.
  Conducted heat now converts with the solver's own Pe (``PhysicsScales.conduction_unit``).
* An unbalanced closed box removed the excess fluid evenly but left its heat behind:
  a theta = 1 jet with no vent climbed to theta 2.77 in 2D. The uniform leak now carries
  the local temperature (and smoke) out, in 2D and 3D, and the heat balance books it.
* 2D buoyancy was added before the body penalisation, which booked the fictitious inner
  fluid's buoyancy as a force (Cl +1.77 on a hot cylinder in still cold fluid) and left a
  uniformly warm fluid around a body moving at 1e-2. It is now added after the
  penalisation with the Archimedes share subtracted (the 3D rule).
* Body-frame box heaters turn with the body (2D rect, 3D box) instead of staying
  aligned with the grid.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aerolume.solvers.ns2d import FlowSolver
from aerolume.solvers.ns2d.config import (
    HeatSourceSpec,
    ImmersedBody,
    PortSpec,
    SolverConfig,
    ThermalConfig,
)
from aerolume.solvers.ns3d import (
    FlowSolver3D,
    HeatSource3D,
    ImmersedMesh,
    Port3D,
    SolverConfig3D,
    ThermalConfig3D,
)
from aerolume.solvers.ns3d import physics as P3


def _circle(n=256, r=0.5, c=(0.0, 0.0)):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)


def _box3(n=12, **kw) -> SolverConfig3D:
    base = {"nx": n, "ny": n, "nz": n, "h": 1.0 / n, "x0": 0.0, "y0": 0.0, "z0": 0.0, "re": 20.0,
            "domain": "closed", "device": "cpu", "dtype": "float64", "dye": False, "diagnostics": True}
    base.update(kw)
    return SolverConfig3D(**base)


# ---------------------------------------------------------------------------
# 3D: the heat sources bound the step
# ---------------------------------------------------------------------------

def test_3d_heater_in_still_air_bounds_its_first_step():
    """A strong heater in a still, buoyant box: the first step is at most 1 / S_max (S = kappa max|q|)
    and the free-fall time of a parcel heated from rest; theta stays bounded (was NaN on step 2)."""
    heater = HeatSource3D(kind="box", center=(0.5, 0.5, 0.25), size=(0.25, 0.25, 0.17), power=4000.0)
    cfg = _box3(16, re=2.4e4, dtype="float32", heat_sources=[heater],
                thermal=ThermalConfig3D(prandtl=0.71, richardson=0.2))
    s = FlowSolver3D(cfg)
    th = s.physics.thermal
    s_max = float(th.heat_max)
    assert s_max == pytest.approx(th.kappa * th.q_max, rel=1e-6) and s_max > 0
    free_fall = (6.0 * s.ops.cfl * s.ops.h / (0.2 * s_max)) ** (1.0 / 3.0)
    s.advance(1)
    assert s.state.dt.item() <= min(1.0 / s_max, free_fall) * (1 + 1e-5)
    # each step adds at most S dt <= 1 to the hottest cell (it was 6e5 in one step), until the plume
    # carries the heat away (this strong heater peaks near theta 12.7 at step 19, then cools)
    prev = float(s.theta.max())
    for _ in range(29):
        s.advance(1)
        assert math.isfinite(s.state.dt.item()) and s.state.dt.item() > 0
        now = float(s.theta.max())
        assert now - prev <= 1.0 + 1e-3
        prev = now
    assert torch.isfinite(s.theta).all() and torch.isfinite(s.state.w).all()
    # a live Reynolds-number change follows into the bound (kappa changes, in place)
    th.set_reynolds(1.2e4, s.ops)
    assert float(th.heat_max) == pytest.approx(th.kappa * th.q_max, rel=1e-6)


def test_3d_heat_flux_body_from_rest_is_bounded_by_its_skin_rate():
    """The heat-flux skin is a source too: its rate kappa max|skin| caps the first step."""
    v, f = P3_icosphere(0.3, (0.5, 0.5, 0.4))
    body = ImmersedMesh(v, f, name="hot", thermal="heat_flux", thermal_value=200.0)
    s = FlowSolver3D(_box3(16, re=2.4e4, bodies=[body], thermal=ThermalConfig3D(prandtl=0.71, richardson=0.2)))
    th = s.physics.thermal
    assert th.geom.skin is not None and th.heat_max is not None
    assert th.q_max == pytest.approx(float(th.geom.skin.abs().max()), rel=1e-9)
    s.advance(1)
    assert s.state.dt.item() <= 1.0 / float(th.heat_max) * (1 + 1e-9)
    s.advance(10)
    assert torch.isfinite(s.theta).all()


def P3_icosphere(d, center):
    """An icosphere from the validation cases (as the 3D physics tests use)."""
    import importlib.util
    import sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "validation" / "ns3d_cases.py"
    mod = sys.modules.get("ns3d_cases")
    if mod is None:
        spec = importlib.util.spec_from_file_location("ns3d_cases", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ns3d_cases"] = mod
        spec.loader.exec_module(mod)
    return mod.icosphere(d, 2, center)


def test_3d_overlapping_heaters_are_summed_for_the_bound():
    """Two heaters on the same cells add their densities in the heat equation, so the bound uses the
    summed density, not the larger of the two."""
    a = HeatSource3D(kind="box", center=(0.5, 0.5, 0.5), size=(0.3, 0.3, 0.3), power=10.0)
    one = FlowSolver3D(_box3(12, heat_sources=[a], thermal=ThermalConfig3D()))
    two = FlowSolver3D(_box3(12, heat_sources=[a, a], thermal=ThermalConfig3D()))
    assert two.physics.thermal.q_max == pytest.approx(2.0 * one.physics.thermal.q_max, rel=1e-9)


def test_planned_3d_heater_only_room_runs_from_rest():
    """The critic's case through the public API on the CPU: al.fan_room_case(dimension=3) with no fan,
    buoyancy on and one 300 W box heater, planned at Draft; 20 steps from rest stay finite."""
    import dataclasses

    import aerolume as al
    p = al.new_project()
    case = al.fan_room_case(thrust=1.0, dimension=3)
    pl = p.plan(case, quality="draft", device="cpu", fans=[], thermal={"buoyancy": True},
                heat_sources=[{"name": "Heater", "kind": "box", "position": (2.0, 1.5, 0.3),
                               "size": (0.4, 0.4, 0.2), "power": 300.0}])
    cfg = dataclasses.replace(pl.raw.solver_config, use_cuda_graphs=False)
    s = FlowSolver3D(cfg)
    th = s.physics.thermal
    s.advance(1)
    first = s.state.dt.item()
    assert first <= 1.0 / float(th.heat_max) * (1 + 1e-5)       # was 68.7 L/U (1 / S_max = 1.03 here)
    peak = float(s.theta.max())
    for _ in range(19):
        s.advance(1)
        assert math.isfinite(s.state.dt.item())
        peak = max(peak, float(s.theta.max()))
    # theta peaks near 2.3 in the heater's cells on this grid, then the plume carries it away
    assert torch.isfinite(s.theta).all() and peak < 3.0


# ---------------------------------------------------------------------------
# closed boxes with unbalanced ports: the uniform leak carries heat out
# ---------------------------------------------------------------------------

def test_2d_unbalanced_closed_box_keeps_theta_bounded_and_books_the_leak():
    """A theta = 1 jet into a closed adiabatic box with no vent: theta never exceeds 1 (it reached 2.77),
    the leak's heat is reported as heat out of the ports, and the balance closes."""
    n = 32
    jet = PortSpec(position=(0.1, 0.8), direction=(1.0, 0.0), size=0.1, speed=1.0, kind="jet", theta=1.0)
    cfg = SolverConfig(nx=n, ny=n, h=1.0 / n, x0=0.0, y0=0.0, re=200.0, domain="closed", bc_top="noslip",
                       bc_bottom="noslip", dye=False, thermal=ThermalConfig(prandtl=0.71, richardson=0.0),
                       ports=[jet], use_cuda_graphs=False, device="cpu", dtype="float64")
    s = FlowSolver(cfg)
    ph = s.ops.phys
    q = float(ph.q_net)
    assert q > 0 and float(ph.src_neg.max()) == pytest.approx(-q / 1.0, rel=1e-9)   # the leak covers every cell
    s.advance(1500)
    th = s.state.theta
    assert float(th.max()) <= 1.0 + 1e-6
    hb = s.heat_balance()
    assert hb["heat_out_ports"] == pytest.approx(q * float(th.sum()) * s.ops.h ** 2, rel=1e-6)
    assert abs(hb["heat_balance_residual"]) < 1e-3 * hb["heat_in_ports"]
    # mean theta approaches 1 from below (well-mixed estimate 1 - exp(-q t / V))
    mean = float(th.mean())
    assert 0.5 * (1 - math.exp(-q * s.t)) < mean < 1.0


def test_2d_net_sink_is_unchanged_by_the_leak():
    """Only vents (q_net < 0): the evenly created fluid arrives at 0, so src_neg is the vents' own sink."""
    vent = PortSpec(position=(0.9, 0.2), direction=(1.0, 0.0), size=0.1, speed=1.0, kind="vent")
    cfg = SolverConfig(nx=24, ny=24, h=1.0 / 24, x0=0.0, y0=0.0, re=100.0, domain="closed", bc_top="noslip",
                       bc_bottom="noslip", dye=False, thermal=ThermalConfig(prandtl=0.71, richardson=0.0),
                       ports=[vent], use_cuda_graphs=False, device="cpu", dtype="float64")
    ph = FlowSolver(cfg).ops.phys
    assert float(ph.q_net) < 0 and float(ph.src_neg.max()) == 0.0


def test_3d_unbalanced_closed_box_keeps_theta_bounded_and_books_the_leak():
    jet = Port3D(kind="jet", center=(0.2, 0.5, 0.5), direction=(1, 0, 0), diameter=0.3, speed=1.0, theta=1.0)
    s = FlowSolver3D(_box3(12, re=100.0, ports=[jet], thermal=ThermalConfig3D(prandtl=0.71)))
    ports = s.physics.ports
    assert ports.leak == pytest.approx(ports.volume_flow / 1.0, rel=1e-12)
    assert any("takes its heat and smoke" in n for n in s.notes)
    s.advance(300)
    assert float(s.theta.max()) <= 1.0 + 1e-6
    hb = s.heat_balance()
    expect = ports.leak * float(s.theta.double().sum()) * s.ops.h ** 3 * s.physics.thermal.pe
    assert hb["heat_out_ports"] == pytest.approx(expect, rel=1e-9) and hb["heat_out_ports"] > 0
    assert abs(hb["heat_balance_residual"]) < 0.02 * hb["heat_in_ports"]
    balanced = FlowSolver3D(_box3(8, ports=[jet, Port3D(kind="vent", center=(0.8, 0.5, 0.5), diameter=0.3,
                                                        speed=1.0)], thermal=ThermalConfig3D()))
    assert balanced.physics.ports.leak == 0.0


# ---------------------------------------------------------------------------
# 2D buoyancy after the penalisation (Archimedes)
# ---------------------------------------------------------------------------

def _hot_box(theta0: float, mode: str = "temperature", n: int = 16) -> FlowSolver:
    cfg = SolverConfig(nx=8 * n, ny=8 * n, h=1.0 / n, x0=-4.0, y0=-4.0, re=100.0, domain="closed",
                       bc_top="noslip", bc_bottom="noslip", dye=False, perturb=False, force_ref=1.0,
                       moment_ref=(0.0, 0.0), device="cpu", dtype="float64", use_cuda_graphs=False,
                       bodies=[ImmersedBody(_circle(), name="cyl", thermal=mode, thermal_value=1.0)],
                       thermal=ThermalConfig(prandtl=0.71, richardson=1.0, theta_initial=theta0))
    s = FlowSolver(cfg)
    if theta0:
        s.set_temperature(torch.full_like(s.state.theta, theta0))
    return s


@pytest.mark.parametrize("mode", ["temperature", "adiabatic"])
def test_2d_uniformly_warm_fluid_around_a_body_stays_at_rest_and_feels_archimedes(mode):
    """theta = 1 everywhere is an exact hydrostatic rest state: |u| at round-off (was 1e-2) and the force
    is -Ri A, A the body's area (Cl = -Ri A / (0.5 ref)), for every kind of body."""
    s = _hot_box(1.0, mode)
    s.advance(10)
    st = s.state
    assert max(float(st.u.abs().max()), float(st.v.abs().max())) < 1e-10
    tg = s._geom.thermal
    area = float(tg.vol_v.sum()) * s.ops.h ** 2
    assert area == pytest.approx(math.pi / 4, rel=5e-3)                  # the unshifted indicator's area
    cl = float(s.coefficients()[1])
    assert cl == pytest.approx(-1.0 * area / 0.5, rel=1e-6)
    assert cl == pytest.approx(-math.pi / 4 / 0.5, rel=5e-3)


def test_2d_hot_body_in_still_cold_fluid_has_no_archimedes_lift():
    """The fluid outside is cold and still, so the true force starts at ~0. Before the fix the step booked
    the inner fluid's buoyancy, Cl = +Ri theta_s A / 0.5 = +1.57 at once (+1.77 after 20 steps). Now the
    first steps show a short start-up transient (the penalisation removes the interface fluid's buoyancy
    over a few stages while the share is booked at once) that has died out by step 5."""
    s = _hot_box(0.0)
    cls = []
    for _ in range(10):
        s.advance(1)
        cls.append(float(s.coefficients()[1]))
    assert all(abs(c) < 0.3 for c in cls[4:]), cls
    assert abs(float(np.mean(cls[1:]))) < 0.3


def test_2d_stratified_fluid_around_an_insulated_body_stays_at_rest():
    """A linear stratification (non-conducting fluid) around an insulated body is an exact rest state."""
    s = _hot_box(0.0, "adiabatic")
    s.ops.phys.kappa.fill_(0.0)
    y = s.ops.y0 + (torch.arange(s.ops.ny, dtype=torch.float64) + 0.5) * s.ops.h
    s.set_temperature((y[:, None] * 0.25).expand(s.ops.ny, s.ops.nx).clone())
    s.advance(10)
    st = s.state
    assert max(float(st.u.abs().max()), float(st.v.abs().max())) < 1e-9


# ---------------------------------------------------------------------------
# capped 2D heat rates
# ---------------------------------------------------------------------------

def _capped_naca(mode: str, value: float):
    import aerolume as al
    from aerolume.simulation.physics import thermal_props
    p = al.new_project()
    d = p.add_naca("0012", chord=0.2, name="Heated NACA")
    d.set_thermal(mode, value)
    kw = {"speed": 10.0, "alpha": 0.0, "quality": "draft", "device": "cpu",
          "thermal": {"t_ambient": 20.0, "buoyancy": False}}
    pl = p.plan(d, **kw)
    raw = pl.raw
    assert raw.capped and raw.re_real / raw.re_sim > 10
    assert any("Heat transfer in the capped flow" in n for n in raw.notes)
    r = p.run_cfd(d, name="capped heated airfoil", duration=0.3, **kw)
    pts = np.asarray(raw.design.bodies[0].points)
    per = float(np.sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)))
    return r, raw, thermal_props(raw.sim), per


def test_capped_2d_heat_rate_is_the_nusselt_number_with_the_real_conductivity():
    r, raw, props, per = _capped_naca("temperature", 40.0)
    q_from_nu = r.nusselt * props.conductivity * 20.0 * per / raw.length_ref
    assert r.heat_rate == pytest.approx(q_from_nu, rel=1e-6)          # the ratio was Re_real / Re_sim = 53
    assert any("Reynolds number capped" in n for n in r.raw.notes) or any(
        "Reynolds number capped" in n for n in (r.raw.notes or []))


def test_capped_2d_heat_flux_body_reports_its_set_flux():
    r, _raw, _props, per = _capped_naca("heat_flux", 500.0)
    assert r.heat_rate == pytest.approx(500.0 * per, rel=1e-4)        # q'' P [W/m]


# ---------------------------------------------------------------------------
# body-frame box heaters turn with the body
# ---------------------------------------------------------------------------

def test_2d_body_frame_rect_heater_turns_with_the_angle_of_attack():
    """The critic's strip: a 0.6 x 0.02 chord-wise heater at (0.5, 0) on a NACA 0012 at alpha 30 kept
    its box horizontal while its centre turned; its ends sat up to 0.17 c off the surface."""
    import aerolume as al
    from aerolume.model import HeatSource
    from aerolume.simulation.defaults import default_simulation
    from aerolume.simulation.physics import physics_scales, placed_physics_2d
    raw_d = al.new_project().add_naca("0012", chord=1.0, name="NACA").raw
    s = default_simulation(raw_d, "heater")
    s.thermal.enabled = True
    s.heat_sources = [HeatSource(name="strip", kind="rect", position=(0.5, 0.0), size=(0.6, 0.02), power=100.0,
                                 frame="body")]
    sc = physics_scales(raw_d, s, 2)
    pl = placed_physics_2d(raw_d, s, sc, alpha_deg=30.0, extent=(-5, 10, -5, 5), h_m=0.01,
                           bodies_wind=[], closed=False)
    spec = pl.heat_sources[0]
    assert spec.direction == pytest.approx((math.cos(math.radians(-30)), math.sin(math.radians(-30))), abs=1e-12)
    # the footprint follows: its second moments are those of a strip turned by -30 degrees
    from aerolume.solvers.ns2d.physics import heat_source_field
    cfg = SolverConfig(nx=200, ny=200, h=0.01, x0=-0.5, y0=-1.0, re=100.0, dye=False, device="cpu",
                       dtype="float64", thermal=ThermalConfig(), heat_sources=[spec])
    ops = FlowSolver(cfg).ops
    q = heat_source_field([spec], ops)
    assert float(q.sum()) * ops.h ** 2 == pytest.approx(spec.power, rel=1e-9)        # exact total
    x = ops.x0 + (torch.arange(ops.nx, dtype=torch.float64) + 0.5) * ops.h
    y = ops.y0 + (torch.arange(ops.ny, dtype=torch.float64) + 0.5) * ops.h
    Y, X = torch.meshgrid(y, x, indexing="ij")
    w = q / q.sum()
    mx, my = float((w * X).sum()), float((w * Y).sum())
    cxy = float((w * (X - mx) * (Y - my)).sum())
    cxx = float((w * (X - mx) ** 2).sum())
    cyy = float((w * (Y - my) ** 2).sum())
    angle = 0.5 * math.degrees(math.atan2(2 * cxy, cxx - cyy))
    assert angle == pytest.approx(-30.0, abs=1.0)


def test_3d_box_shape_with_axes_integrates_exactly_and_covers_the_turned_box():
    r = P3_rotation(30.0)
    fn, lo, hi = P3.box_shape((0.0, 0.0, 0.0), (0.6, 0.1, 0.2), 0.02, axes=r)
    ext = np.abs(r) @ np.array([0.3, 0.05, 0.1])
    assert np.allclose(hi, ext + 0.02) and np.allclose(lo, -ext - 0.02)
    # a point along the box's own x axis, 0.25 out, is inside; the same distance along grid x is not
    px = r[:, 0] * 0.25
    assert fn(px[0], px[1], px[2]) == pytest.approx(1.0)
    assert fn(0.25, 0.0, 0.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        HeatSource3D(kind="box", center=(0, 0, 0), size=(1, 1, 1), axes=np.diag([1.0, 2.0, 1.0]))


def P3_rotation(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def test_2d_heater_direction_defaults_to_x():
    assert HeatSourceSpec(position=(0, 0)).direction == (1.0, 0.0)
