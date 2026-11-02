"""Placed physics in the planners (DESIGN.md 10): scales, frames, solver units, checks, rationale.

Planning never runs a solver, so these tests are fast. Every expected number is computed
here from its definition (DESIGN.md 10.2 / 10.3 and the solver configs' docstrings: theta =
(T - T_amb) / dT, Ri = g beta dT L / U^2, G = q'' L / (k dT), heat in rho c_p U dT L per metre
of span in 2D and k dT L in 3D, thrust in rho U^2 L^(d-1) ...), not copied from the code under
test, so a wrong conversion fails here before it can bias a run.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from aerolume.core import fluid as F
from aerolume.core import units
from aerolume.geometry import templates
from aerolume.geometry3d import design_from_meshes, sphere
from aerolume.model import (
    CFDSettings,
    Design,
    Emitter,
    Fan,
    FlowConditions,
    Fluid,
    HeatSource,
    Port,
    Simulation,
    ThermalSettings,
    WallThermal,
)
from aerolume.simulation import physics as PH
from aerolume.simulation import plan as planmod
from aerolume.simulation.plan import plan_live, plan_simulation
from aerolume.simulation.plan3d import rotation3 as plan_rotation3

AIR = F.air_isa(0.0)                 # 15 °C, the fluid the CLI and the API cases use
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
G0 = 9.80665


@pytest.fixture(autouse=True)
def fresh_calibration(monkeypatch):
    """Keep run-time measurements and any calibration file out of these tests."""
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def cylinder(d: float = 0.01, *, t_body: float | None = None, mode: str = "temperature"):
    des = templates.make("cylinder", diameter=d)
    if t_body is not None:
        des.bodies[0].thermal, des.bodies[0].thermal_value = mode, float(t_body)
    return des


def tunnel_sim(design, speed: float, *, alpha: float = 0.0, fluid=AIR, **kw) -> Simulation:
    cfd = kw.pop("cfd", None) or CFDSettings(quality="draft", device="cpu")
    return Simulation("t", design.id, conditions=FlowConditions(fluid, speed, alpha), cfd=cfd, **kw)


def room2d(width: float = 4.0, height: float = 3.0, ref: float = 0.5) -> Design:
    return Design(name="Room", kind="generic", bodies=[], length_ref=ref, force_ref=ref,
                  moment_ref=(width / 2, height / 2))


def room3d(size=(4.0, 3.0, 2.5), ref: float = 0.5) -> Design:
    return Design(name="Room 3D", kind="generic", bodies=[], length_ref=ref, force_ref=ref, dimension=3,
                  ref_area=ref * ref, span_ref=ref, moment_ref3=tuple(c / 2 for c in size))


def closed_sim(design, box, *, fluid=AIR, quality="draft", **kw) -> Simulation:
    cfd = CFDSettings(quality=quality, device="cpu", domain_mode="closed", domain_box=tuple(box),
                      **kw.pop("cfd_kw", {}))
    return Simulation("c", design.id, conditions=FlowConditions(fluid, 1.0, 0.0), cfd=cfd, **kw)


def cavity_sim(ra: float, *, size: float = 0.1, pr: float = 0.71, dim: int = 2, **cfd_kw):
    """The de Vahl Davis square / Tric cubic cavity, built here from the definition of Ra."""
    nu = AIR.kinematic_viscosity
    beta = 1.0 / 288.15
    d_t = ra * nu * nu / (pr * G0 * beta * size ** 3)
    th = ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True, beta=beta, prandtl=pr,
                         walls=[WallThermal("left", "temperature", 15.0 + d_t / 2),
                                WallThermal("right", "temperature", 15.0 - d_t / 2)])
    if dim == 2:
        des = Design(name="cavity", kind="generic", bodies=[], length_ref=size, force_ref=size,
                     moment_ref=(size / 2, size / 2))
        box = (0.0, size, 0.0, size)
    else:
        des = Design(name="cavity3", kind="generic", bodies=[], length_ref=size, force_ref=size, dimension=3,
                     ref_area=size * size, span_ref=size, moment_ref3=(size / 2,) * 3)
        box = (0.0, size, 0.0, size, 0.0, size)
    return des, closed_sim(des, box, thermal=th, cfd_kw=cfd_kw), d_t


def air_props(t_c: float = 15.0) -> tuple[float, float, float, float]:
    """(k, cp, beta, Pr) of air: Sutherland's law for k (White, Table 1-3), ideal-gas beta."""
    tk = t_c + 273.15
    k = 0.0241 * (tk / 273.15) ** 1.5 * (273.15 + 194.0) / (tk + 194.0)
    mu = F.sutherland_viscosity(tk)
    return k, 1006.0, 1.0 / tk, mu * 1006.0 / k


# ---------------------------------------------------------------------------
# defaults: nothing changes for v0.1 / v0.2 simulations
# ---------------------------------------------------------------------------

def test_a_simulation_without_placed_physics_plans_exactly_as_before():
    des = cylinder()
    p = plan_simulation(des, tunnel_sim(des, 0.05))
    cfg = p.solver_config
    assert p.placed is None and p.physics is None and p.monitors == () and not p.closed
    assert cfg.thermal is None and cfg.emitters == [] and cfg.fans == [] and cfg.ports == [] and cfg.heat_sources == []
    assert cfg.domain == "tunnel" and cfg.dye_inlet and cfg.dye
    assert not PH.has_placed_physics(tunnel_sim(des, 0.05))
    rows = p.summary()
    assert not any(k in rows for k in ("Temperature scale", "Reference velocity", "Domain type", "Fans", "Emitters"))
    # the recorded diagnostics (on by default, a few % of a step) are the only addition
    assert cfg.diagnostics and p.step_cost == pytest.approx(1.0 + PH.PHYSICS_COST[2]["diagnostics"])
    off = tunnel_sim(des, 0.05, cfd=CFDSettings(quality="draft", device="cpu", record_diagnostics=False))
    assert plan_simulation(des, off).step_cost == 1.0


def test_record_diagnostics_and_the_pid_step_reach_the_solver():
    des = cylinder()
    cfd = CFDSettings(quality="draft", device="cpu", record_diagnostics=False, adaptive_dt="pid")
    cfg = plan_simulation(des, tunnel_sim(des, 0.05, cfd=cfd)).solver_config
    assert cfg.diagnostics is False and cfg.adaptive_dt == "pid"


# ---------------------------------------------------------------------------
# fluid properties and scales
# ---------------------------------------------------------------------------

def test_air_and_water_properties_come_from_their_references():
    k, cp, beta, pr = air_props(15.0)
    d, basis = PH.fluid_defaults(AIR, 15.0)
    assert d["conductivity"] == pytest.approx(k, rel=1e-12) and d["cp"] == cp
    assert d["beta"] == pytest.approx(beta) and d["prandtl"] == pytest.approx(pr, rel=1e-12)
    assert 0.70 < pr < 0.72 and 0.0250 < k < 0.0256 and "Sutherland" in basis
    w, basis_w = PH.fluid_defaults(F.water(20.0), 20.0)
    assert (w["conductivity"], w["cp"], w["beta"]) == pytest.approx((0.598, 4182.0, 2.07e-4))   # Incropera A.6
    assert w["prandtl"] == pytest.approx(F.water(20.0).viscosity * 4182.0 / 0.598)
    assert "Incropera" in basis_w
    none, why = PH.fluid_defaults(Fluid("Glycerol", 1260.0, 1.41), 20.0)
    assert none is None and "custom" in why


def test_dimensionless_groups_read_like_the_setup_page():
    assert [PH.fmt_number(v) for v in (0.71, 4.5, 84.0, 1e5, 2.3e-3, 0.0, 9.96e4, float("nan"))] ==         ["0.71", "4.50", "84.0", "1.0×10⁵", "2.3×10⁻³", "0", "1.0×10⁵", "—"]


def test_forced_convection_scales_of_a_heated_cylinder():
    des = cylinder(0.01, t_body=25.0)
    U = 0.3                                      # Ri = g beta dT d / U^2 = 0.038: forced convection
    sim = tunnel_sim(des, U, thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=True))
    p = plan_simulation(des, sim)
    s = p.physics
    _, _, beta, pr = air_props(15.0)
    nu = AIR.kinematic_viscosity
    assert s.thermal and not s.closed and not s.natural
    assert s.delta_t == pytest.approx(10.0) and "Cylinder" in s.delta_t_basis
    assert s.speed == U and s.re == pytest.approx(U * 0.01 / nu) and s.pe == pytest.approx(s.re * pr)
    assert s.ri == pytest.approx(G0 * beta * 10.0 * 0.01 / U ** 2, rel=1e-12)         # Boussinesq coefficient
    assert s.gr == pytest.approx(G0 * beta * 10.0 * 0.01 ** 3 / nu ** 2) and s.ra == pytest.approx(s.gr * pr)
    assert s.ri_flow == pytest.approx(s.gr / s.re ** 2) and s.regime == "Forced"       # Ri 0.038 < 0.1
    th = p.solver_config.thermal
    assert th.prandtl == pytest.approx(pr) and th.richardson == pytest.approx(s.ri) and tuple(th.gravity) == (0.0, -1.0)
    body = p.solver_config.bodies[0]
    assert (body.thermal, body.thermal_value) == ("temperature", pytest.approx(1.0))   # theta of the hot body
    assert p.monitors == ("nu",)
    rows = p.summary()
    assert rows["Temperature scale"] == "10.0 K" and rows["Prandtl number"] == PH.fmt_number(pr) == "0.71"
    assert rows["Péclet number"] == PH.fmt_number(s.pe) and rows["Richardson number"] == PH.fmt_number(s.ri_flow)
    assert rows["Convection"] == "Forced" and "Rayleigh number" in rows and "Grashof number" in rows
    # without buoyancy the temperature is passive: Ri 0 in the solver, the regime says so
    sim.thermal.buoyancy = False
    p2 = plan_simulation(des, sim)
    assert p2.solver_config.thermal.richardson == 0.0 and p2.physics.regime == "Forced (no buoyancy)"
    assert any("Buoyancy off" in n for n in p2.notes)


def test_mixed_and_natural_regimes_follow_the_richardson_number():
    des = cylinder(0.05, t_body=65.0)
    nu = AIR.kinematic_viscosity
    beta = 1.0 / 288.15
    gr = G0 * beta * 50.0 * 0.05 ** 3 / nu ** 2
    for re_, regime in ((math.sqrt(gr / 1.0), "Mixed"), (math.sqrt(gr / 20.0), "Natural")):
        U = re_ * nu / 0.05
        s = plan_simulation(des, tunnel_sim(des, U, thermal=ThermalSettings(enabled=True, t_ambient=15.0))).physics
        assert s.ri_flow == pytest.approx(gr / re_ ** 2, rel=1e-9) and s.regime == regime


def test_typed_properties_win_over_the_fluid_defaults():
    des = cylinder(t_body=25.0)
    th = ThermalSettings(enabled=True, t_ambient=15.0, prandtl=0.9, conductivity=0.05, cp=1200.0, beta=0.002)
    p = plan_simulation(des, tunnel_sim(des, 0.1, thermal=th))
    props = p.physics.props
    assert (props.prandtl, props.conductivity, props.cp, props.beta) == (0.9, 0.05, 1200.0, 0.002)
    assert set(props.typed) == {"prandtl", "conductivity", "cp", "beta"}
    assert p.solver_config.thermal.prandtl == 0.9
    assert p.physics.ri == pytest.approx(G0 * 0.002 * 10.0 * 0.01 / 0.01)


def test_heat_flux_sets_the_temperature_scale_and_becomes_a_gradient():
    """A flux body: dT_ref is the estimated rise q L / (k Nu) and the solver gets G = q'' L / (k dT)."""
    q = 500.0
    des = cylinder(0.01, t_body=q, mode="heat_flux")
    U = 0.2
    p = plan_simulation(des, tunnel_sim(des, U, thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False)))
    s = p.physics
    k, _, _, pr = air_props(15.0)
    nu_f = 0.664 * math.sqrt(U * 0.01 / AIR.kinematic_viscosity) * pr ** (1 / 3)
    assert s.delta_t == pytest.approx(q * 0.01 / (k * max(nu_f, 1.0)), rel=1e-9)
    assert "q·L/(k·Nu)" in s.delta_t_basis
    body = p.solver_config.bodies[0]
    assert body.thermal == "heat_flux" and body.thermal_value == pytest.approx(q * 0.01 / (k * s.delta_t))


def test_custom_fluid_warns_and_a_large_delta_t_breaks_boussinesq():
    des = cylinder(t_body=25.0)
    oil = Fluid("Test oil", 900.0, 0.09)
    p = plan_simulation(des, tunnel_sim(des, 2.0, fluid=oil, thermal=ThermalSettings(enabled=True)))
    assert not p.physics.props.known and any("Custom fluid" in w for w in p.warnings)
    hot = cylinder(t_body=215.0)                                      # beta dT = 200 / 288 = 0.69
    p2 = plan_simulation(hot, tunnel_sim(hot, 0.1, thermal=ThermalSettings(enabled=True, t_ambient=15.0)))
    assert any("Boussinesq" in w for w in p2.warnings)


def test_heat_transfer_with_nothing_heated_says_so():
    des = cylinder()
    p = plan_simulation(des, tunnel_sim(des, 0.1, thermal=ThermalSettings(enabled=True)))
    assert p.physics.delta_t == 1.0 and any("nothing is heated" in w for w in p.warnings)


def test_heaters_and_hot_jets_without_heat_transfer_warn():
    des = cylinder()
    sim = tunnel_sim(des, 0.1, heat_sources=[HeatSource(name="H", kind="rect", position=(0.03, 0.0),
                                                        size=(0.005, 0.005), power=5.0)])
    p = plan_simulation(des, sim)
    assert p.solver_config.heat_sources == [] and any("Heat transfer is off" in w for w in p.warnings)


# ---------------------------------------------------------------------------
# closed boxes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ra", [1e4, 1e5])
def test_a_cavity_runs_at_the_free_fall_velocity_with_its_rayleigh_number(ra):
    des, sim, d_t = cavity_sim(ra)
    p = plan_simulation(des, sim)
    s, cfg = p.physics, p.solver_config
    beta = 1.0 / 288.15
    assert p.closed and s.closed and s.natural and s.regime == "Natural"
    assert s.ra == pytest.approx(ra, rel=1e-9) and s.length_ra == pytest.approx(0.1)
    assert s.speed == pytest.approx(math.sqrt(G0 * beta * d_t * 0.1), rel=1e-12)
    assert s.re == pytest.approx(math.sqrt(ra / 0.71), rel=1e-9)                       # Re = sqrt(Gr)
    assert s.ri == pytest.approx(1.0) and cfg.thermal.richardson == pytest.approx(1.0)  # L = H
    walls = {w.side: (w.mode, w.value) for w in cfg.thermal.walls}
    assert walls == {"left": ("temperature", pytest.approx(0.5)), "right": ("temperature", pytest.approx(-0.5))}
    assert cfg.domain == "closed" and cfg.bc_left == cfg.bc_right == cfg.bc_top == cfg.bc_bottom == "noslip"
    assert cfg.nx * cfg.h == pytest.approx(1.0) and cfg.ny * cfg.h == pytest.approx(1.0)   # walls on the box
    assert p.monitors == ("nu", "ke") and math.isnan(p.blockage)
    rows = p.summary()
    assert rows["Domain type"].startswith("Closed box") and "Blockage" not in rows
    assert rows["Rayleigh number"] == ("1.0×10⁴" if ra == 1e4 else "1.0×10⁵") and "free-fall" in rows["Reference velocity"]
    assert "Nu to ±" in rows["Stops early when"] and "Real Reynolds number" not in rows
    # the run length: (45 + 0.225 Pe_H) free-fall times (NATURAL_T), measured from the validation runs
    a, b, cap = planmod.NATURAL_T
    assert p.t_end == pytest.approx(min(a + b * s.pe, cap)) and p.t_avg_start == pytest.approx(0.5 * p.t_end)


def test_the_thermal_boundary_layer_sets_the_cavity_grid():
    des, sim, _ = cavity_sim(1e6)
    p = plan_simulation(des, sim)
    nu_c = PH.cavity_nusselt(1e6, 0.71)
    assert nu_c == pytest.approx(0.18 * (0.71 * 1e6 / 0.91) ** 0.29)                   # Incropera eq. 9.50
    h_bl = 0.1 / (2 * nu_c) / planmod.THIN_CELLS[0]
    assert p.solver_config.h * 0.1 <= h_bl * (1 + 1e-9)
    assert any("thermal boundary layer" in n for n in p.notes)
    coarse_des, coarse, _ = cavity_sim(1e6, cells_per_ref=16)
    assert any("thermal boundary layer" in w for w in plan_simulation(coarse_des, coarse).warnings)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_a_strongly_heated_room_plans_promptly_within_its_budget(device):
    """Regression: the budget loop once re-fitted the same 384 x 288 grid forever for this room (the
    Setup page plans on the GUI thread, so the app froze)."""
    import time
    room = room2d()
    th = ThermalSettings(enabled=True, t_ambient=15.0, walls=[WallThermal("left", "temperature", 50.0),
                                                            WallThermal("top", "heat_flux", 200.0)])
    sim = closed_sim(room, (0, 4, 0, 3), thermal=th)
    sim.cfd.device = device
    t0 = time.perf_counter()
    p = plan_simulation(room, sim, device=device)
    assert time.perf_counter() - t0 < 5.0
    dclass = "gpu" if device == "cuda" else "cpu"
    assert p.cells <= planmod.CELL_BUDGET[dclass][0] and p.physics.ra > 1e10
    assert any("cell budget" in w for w in p.warnings)


def test_a_conduction_box_moves_at_kappa_over_h():
    des, sim, _ = cavity_sim(300.0)
    p = plan_simulation(des, sim)
    s = p.physics
    kappa = AIR.kinematic_viscosity / 0.71
    assert s.regime == "Conduction" and "κ/H" in s.speed_basis
    assert s.speed == pytest.approx(kappa / 0.1)
    assert p.t_end * p.length_ref / p.speed == pytest.approx(planmod.CONDUCTION_T * 0.1 ** 2 / kappa)


def test_a_closed_box_that_nothing_drives_is_refused():
    des = room2d()
    with pytest.raises(ValueError, match="Nothing drives the flow"):
        plan_simulation(des, closed_sim(des, (0, 4, 0, 3)))
    hot_wall = ThermalSettings(enabled=True, walls=[WallThermal("left", "temperature", 30.0)])
    with pytest.raises(ValueError, match="needs its box"):        # no bodies, no items, no box: no walls
        plan_simulation(des, Simulation("x", des.id, cfd=CFDSettings(domain_mode="closed", device="cpu"),
                                        thermal=hot_wall))
    # placed items alone are enough: the box is them with one size of room around
    p = plan_simulation(des, Simulation("x", des.id, cfd=CFDSettings(domain_mode="closed", device="cpu"),
                                        fans=[Fan(position=(1, 1), diameter=0.5, value=1.0, frame="wind")]))
    # the fan's disc spans 0.75 .. 1.25 m on both axes; one size (0.5 m) of room on every side
    assert p.extent[0] == pytest.approx(0.25) and p.extent[2] == pytest.approx(0.25)


def test_a_closed_box_around_the_bodies_has_one_size_of_room():
    des = cylinder(0.1, t_body=40.0)
    sim = Simulation("x", des.id, conditions=FlowConditions(AIR, 1.0, 0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", domain_mode="closed"),
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0))
    p = plan_simulation(des, sim)
    x0, x1, y0, y1 = p.extent
    assert x0 == pytest.approx(-0.15) and y0 == pytest.approx(-0.15)
    assert x1 == pytest.approx(0.15, abs=p.solver_config.h * 0.1) and y1 == pytest.approx(0.15)
    assert any("one size of room" in n for n in p.notes)


@pytest.mark.parametrize("dim", [2, 3])
def test_a_thrust_fan_drives_a_closed_room_at_its_momentum_theory_speed(dim):
    rho = AIR.density
    T, D = 2.0, 0.5
    if dim == 2:
        des, box, pos, direction = room2d(), (0, 4, 0, 3), (1.3, 1.5), (1.0, 0.0)
        area = D
    else:
        des, box, pos, direction = room3d(), (0, 4, 0, 3, 0, 2.5), (1.3, 1.5, 1.25), (1.0, 0.0, 0.0)
        area = math.pi * D * D / 4
    sim = closed_sim(des, box, fans=[Fan(name="F", position=pos, direction=direction, diameter=D, value=T,
                                          frame="wind")])
    p = plan_simulation(des, sim)
    U = math.sqrt(T / (2 * rho * area))
    assert p.speed == pytest.approx(U, rel=1e-12) and "momentum theory" in p.physics.speed_basis
    fan = p.solver_config.fans[0]
    L = 0.5
    assert fan.value == pytest.approx(T / (rho * U * U * L ** (dim - 1)), rel=1e-12) and fan.mode == "thrust"
    assert p.monitors == ("fan0_velocity", "ke")
    assert p.summary()["Fans"].startswith("F (thrust 2.00 " + ("N/m" if dim == 2 else "N"))


def _heated_room(fluid=AIR, t_ambient: float = 20.0):
    """The critique's room: a 1 N/m thrust fan, a 500 W/m heater, a jet and a vent, heat on."""
    des = room2d()
    sim = closed_sim(des, (0, 4, 0, 3), fluid=fluid,
                     thermal=ThermalSettings(enabled=True, t_ambient=t_ambient, buoyancy=True),
                     fans=[Fan(name="Fan", position=(4 / 3, 1.5), direction=(1, 0), diameter=0.5, value=1.0,
                               frame="wind")],
                     heat_sources=[HeatSource(name="Heater", kind="rect", position=(3.5, 0.3), size=(0.3, 0.2),
                                              power=500.0, frame="wind")],
                     ports=[Port(name="Supply", kind="jet", position=(0.05, 2.7), direction=(1, 0), size=0.2,
                                 speed=0.5, frame="wind"),
                            Port(name="Vent", kind="vent", position=(3.95, 2.7), direction=(1, 0), size=0.2,
                                 speed=0.5, frame="wind")])
    return des, sim


def test_the_room_states_one_temperature_scale_everywhere():
    """Round 3 critique: the room's planner gave β·ΔT 0.25 (ΔT 72.3 K) while the page said 0.30
    (ΔT 86.9 K) and Ra 2.6×10¹¹: the page ignored the thrust fan's momentum-theory speed, which
    sets the heater's estimated rise. One derivation (thermal_estimate = the planner's scales)
    feeds every message."""
    des, sim = _heated_room()
    p = plan_simulation(des, sim)
    est = PH.thermal_estimate(des, sim)
    assert est is not None
    for k in ("delta_t", "ra", "gr", "pe", "re", "speed", "beta_dt"):
        assert getattr(est, k) == pytest.approx(getattr(p.physics, k), rel=1e-12), k
    assert est.beta_dt == pytest.approx(est.props.beta * est.delta_t, rel=1e-12)
    text = PH.boussinesq_warning(est)
    assert text.startswith(f"β·ΔT = {est.beta_dt:.2f} (ΔT {units.sig(est.delta_t, 3)} K)")
    # the plan quotes the very same line when β·ΔT is out of range (the heater's through-flow rise,
    # physics.heater_rise, puts this 500 W/m room at about 5.6 K: inside it)
    assert (text in p.warnings) == (est.beta_dt > PH.BOUSSINESQ_LIMIT)
    dt = units.sig(p.physics.delta_t, 3)
    assert sum(f"ΔT = {dt} K" in n for n in p.notes) == 1           # and its notes the same ΔT
    assert p.summary()["Temperature scale"] == f"{dt} K"
    assert p.summary()["Rayleigh number"] == PH.fmt_number(est.ra)
    # heat off, or a setup that cannot be scaled: no numbers rather than an exception
    sim.thermal.enabled = False
    assert PH.thermal_estimate(des, sim) is None


def test_the_fluid_and_the_ambient_temperature_must_agree():
    assert PH.fluid_temperature(F.air(20.0)) == pytest.approx(20.0)
    assert PH.fluid_temperature(F.water(35.0)) == pytest.approx(35.0)
    assert PH.fluid_temperature(F.air_isa(0.0)) == pytest.approx(15.0)
    assert PH.fluid_temperature(F.air_isa(1000.0)) == pytest.approx(15.0 - 6.5, abs=1e-6)
    assert PH.fluid_temperature(Fluid("Test oil", 900.0, 0.09)) is None
    des, sim = _heated_room(fluid=AIR, t_ambient=20.0)               # ISA air (15 °C) in a 20 °C room
    w = PH.fluid_ambient_warning(sim)
    assert w.startswith("The fluid is air at 15 °C but the ambient temperature is 20 °C")
    assert w in plan_simulation(des, sim).warnings
    des, sim = _heated_room(fluid=F.air(20.0), t_ambient=20.0)
    assert PH.fluid_ambient_warning(sim) == ""
    assert not any("ambient temperature is" in x for x in plan_simulation(des, sim).warnings)
    sim.thermal.enabled = False
    sim.conditions.fluid = AIR
    assert PH.fluid_ambient_warning(sim) == ""                       # heat off: the temperatures do not matter


# ---------------------------------------------------------------------------
# solver units and frames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,value", [("thrust", 0.6), ("pressure_jump", 3.0), ("velocity", 1.5)])
def test_fan_settings_become_solver_units_2d(mode, value):
    des = cylinder(0.2)
    U, L, rho = 2.0, 0.2, AIR.density
    sim = tunnel_sim(des, U, fans=[Fan(position=(0.6, 0.1), direction=(1.0, 1.0), diameter=0.1, mode=mode,
                                       value=value, thickness=0.02, frame="wind")])
    fan = plan_simulation(des, sim).solver_config.fans[0]
    expect = {"thrust": value / (rho * U * U * L), "pressure_jump": value / (rho * U * U), "velocity": value / U}[mode]
    assert fan.value == pytest.approx(expect, rel=1e-12) and fan.mode == mode
    assert fan.width == pytest.approx(0.1 / L) and fan.thickness == pytest.approx(0.02 / L)
    assert fan.position == pytest.approx((0.6 / L, 0.1 / L))
    assert fan.direction == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))


def test_fan_settings_become_solver_units_3d():
    des = design_from_meshes([sphere(0.2, 24)], "S", kind="generic")
    U, L, rho = 2.0, des.length_ref, AIR.density
    sim = tunnel_sim(des, U, fans=[Fan(position=(0.5, 0.0, 0.0), direction=(0, 0, 2), diameter=0.1, mode="thrust",
                                       value=0.3, swirl=0.4, frame="wind")])
    p = plan_simulation(des, sim)
    fan = p.solver_config.fans[0]
    assert fan.value == pytest.approx(0.3 / (rho * U * U * L * L), rel=1e-12) and fan.swirl == 0.4
    assert fan.diameter == pytest.approx(0.1 / L) and tuple(fan.direction) == pytest.approx((0.0, 0.0, 1.0))
    assert not p.symmetry_plane and any("symmetry plane is not used" in n for n in p.notes)


def test_body_frame_items_turn_with_the_angle_of_attack_and_wind_frame_items_stay():
    des = cylinder(0.2)
    alpha = 12.0
    fans = [Fan(name="body", position=(0.5, 0.1), direction=(1.0, 0.0), diameter=0.1, value=0.2, frame="body"),
            Fan(name="wind", position=(0.5, 0.1), direction=(1.0, 0.0), diameter=0.1, value=0.2, frame="wind")]
    p = plan_simulation(des, tunnel_sim(des, 2.0, alpha=alpha, fans=fans))
    f_body, f_wind = p.solver_config.fans
    a = math.radians(-alpha)                     # nose up = the geometry turns clockwise in the tunnel
    ref = np.array(des.moment_ref)
    d = np.array([0.5, 0.1]) - ref
    expect = ref + np.array([math.cos(a) * d[0] - math.sin(a) * d[1], math.sin(a) * d[0] + math.cos(a) * d[1]])
    assert f_body.position == pytest.approx(tuple(expect / 0.2))
    assert f_body.direction == pytest.approx((math.cos(a), math.sin(a)))
    assert f_wind.position == pytest.approx((0.5 / 0.2, 0.1 / 0.2)) and f_wind.direction == pytest.approx((1.0, 0.0))


@pytest.mark.parametrize("alpha,yaw", [(0.0, 0.0), (8.0, 0.0), (0.0, 15.0), (5.0, -10.0)])
def test_the_3d_attitude_matrix_is_the_planners(alpha, yaw):
    r = PH.rotation3(alpha, yaw)
    assert r == pytest.approx(plan_rotation3(alpha, yaw), abs=1e-15)
    assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
    des = design_from_meshes([sphere(0.2, 16)], "S", kind="generic")
    p = np.array([0.3, 0.1, -0.05])
    got = PH.wind_points3(p, des, alpha, yaw, "body")[0]
    ref = np.asarray(des.moment_ref3)
    assert got == pytest.approx((p - ref) @ r.T + ref)
    assert PH.wind_points3(p, des, alpha, yaw, "wind")[0] == pytest.approx(p)


def test_emitters_heaters_and_ports_in_solver_units_2d():
    des = cylinder(0.2, t_body=35.0)
    U, L = 2.0, 0.2
    th = ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False)
    sim = tunnel_sim(des, U, thermal=th,
                     emitters=[Emitter(name="smoke", species="dye", kind="point", position=(-0.2, 0.05), rate=1.4,
                                       pulse_period=0.5, pulse_duty=0.3, start=0.1, stop=0.9, color="#e04040"),
                               Emitter(name="dots", species="particles", kind="rake", position=(-0.2, -0.1),
                                       end=(-0.2, 0.1), count=5, rate=250.0, color="#40c040")],
                     heat_sources=[HeatSource(name="H", kind="rect", position=(0.6, 0.0), size=(0.1, 0.05),
                                              power=40.0, frame="wind")],
                     ports=[Port(name="J", kind="jet", position=(0.8, 0.2), direction=(0.0, -1.0), size=0.05, speed=3.0,
                                 temperature=55.0, dye=True, frame="wind")])
    p = plan_simulation(des, sim)
    cfg, s = p.solver_config, p.physics
    dye, dots = cfg.emitters
    assert dye.species == "dye" and dye.rate == 1.0 and any("clipped" in n for n in p.notes)    # concentration <= 1
    assert dye.pulse_period == pytest.approx(0.5 * U / L) and dye.start == pytest.approx(0.1 * U / L)
    assert dye.stop == pytest.approx(0.9 * U / L) and dye.pulse_duty == 0.3
    assert dye.position == pytest.approx((-1.0, 0.25))
    assert dots.species == "particles" and dots.rate == pytest.approx(250.0 * L / U) and dots.count == 5
    assert p.placed.particle_palette == ["#40c040"] and p.placed.particle_names == ["dots"]
    _, cp, _, _ = air_props(15.0)
    assert s.delta_t == pytest.approx(40.0)                                            # the 55 °C jet
    heater = cfg.heat_sources[0]
    assert heater.power == pytest.approx(40.0 / (AIR.density * cp * U * 40.0 * L), rel=1e-12)
    assert heater.size == pytest.approx((0.5, 0.25)) and heater.kind == "rect"
    jet = cfg.ports[0]
    assert jet.speed == pytest.approx(3.0 / U) and jet.size == pytest.approx(0.25)
    # smoky jets release the default (first) colour: here a channel after the red emitter's
    assert dye.channel == 0 and jet.theta == pytest.approx((55.0 - 15.0) / 40.0) and jet.dye_channel == 1
    assert [c["color"] for c in p.placed.dye_channels] == ["#e04040", "#f5f5f5"]
    assert cfg.bodies[0].thermal_value == pytest.approx(20.0 / 40.0)
    assert not cfg.dye_inlet                                    # the placed smoke replaces the inlet comb
    rows = p.summary()
    assert rows["Emitters"] == "1 smoke (2 colours) · 1 particle"          # the jet's smoke is a colour too
    assert rows["Jets and vents"] == "1 jet, 0 vents" and rows["Heat sources"].startswith("1 (40.0 W/m")


def test_heaters_in_3d_are_in_conduction_units():
    des = design_from_meshes([sphere(0.2, 24)], "S", kind="generic")
    for m in des.meshes:
        m.thermal, m.thermal_value = "temperature", 45.0
    U, L = 1.0, des.length_ref
    sim = tunnel_sim(des, U, thermal=ThermalSettings(enabled=True, t_ambient=15.0, buoyancy=False),
                     heat_sources=[HeatSource(name="H", kind="box", position=(0.5, 0.0, 0.0), size=(0.1, 0.1, 0.1),
                                              power=20.0, frame="wind")])
    p = plan_simulation(des, sim)
    k, *_ = air_props(15.0)
    h = p.solver_config.heat_sources[0]
    assert h.power == pytest.approx(20.0 / (k * 30.0 * L), rel=1e-12) and h.kind == "box"
    assert p.solver_config.bodies[0].thermal_value == pytest.approx(1.0)
    assert p.physics.heat_unit == pytest.approx(k * 30.0 * L)


def test_wall_conditions_in_a_tunnel_and_in_a_closed_box():
    des = cylinder(0.2, t_body=35.0)
    walls = [WallThermal("left", "temperature", 50.0), WallThermal("top", "heat_flux", 200.0),
             WallThermal("bottom", "temperature", 5.0)]
    p = plan_simulation(des, tunnel_sim(des, 1.0, thermal=ThermalSettings(enabled=True, t_ambient=15.0,
                                                                          walls=walls)))
    s = p.physics
    k, *_ = air_props(15.0)
    got = {w.side: (w.mode, w.value) for w in p.solver_config.thermal.walls}
    assert "left" not in got and any("inlet" in n for n in p.notes)                    # the inlet holds ambient
    assert s.delta_t == pytest.approx(30.0)                                            # body 35 to wall 5
    assert got["bottom"] == ("temperature", pytest.approx(-10.0 / 30.0))
    assert got["top"] == ("heat_flux", pytest.approx(200.0 * 0.2 / (k * 30.0)))
    # in a closed box the sides are walls: the left one counts
    room = room2d()
    p2 = plan_simulation(room, closed_sim(room, (0, 4, 0, 3), thermal=ThermalSettings(enabled=True, t_ambient=15.0,
                                                                                        walls=walls)))
    got2 = {w.side: (w.mode, w.value) for w in p2.solver_config.thermal.walls}
    # the 200 W/m² ceiling on a 3 m room: its natural-convection rise q H / (k Nu), Nu = 0.60 Ra*^(1/5)
    # (Sparrow & Gregg), exceeds the 45 K spread of the set temperatures, so it is the scale
    k, _, beta, pr = air_props(15.0)
    nu = AIR.kinematic_viscosity
    ra_star = G0 * beta * 200.0 * 3.0 ** 4 / (k * nu * nu / pr)
    rise = 200.0 * 3.0 / (k * 0.60 * ra_star ** 0.2)
    assert rise > 45.0 and p2.physics.delta_t == pytest.approx(rise, rel=1e-9)
    assert got2["left"] == ("temperature", pytest.approx(35.0 / rise))
    assert p2.physics.natural and p2.physics.length_ra == pytest.approx(3.0)


def test_resolution_warnings_for_fans_ports_and_heaters():
    des = cylinder(0.2)
    sim = tunnel_sim(des, 2.0, thermal=ThermalSettings(enabled=True, t_ambient=15.0),
                     fans=[Fan(name="tiny", position=(0.6, 0.0), diameter=0.004, value=0.01, frame="wind")],
                     ports=[Port(name="pin", position=(0.6, 0.1), size=0.002, speed=1.0, frame="wind")],
                     heat_sources=[HeatSource(name="dot", kind="disc", position=(0.6, -0.1), size=(0.001,), power=1.0,
                                              frame="wind")])
    p = plan_simulation(des, sim)
    assert any("Fan “tiny” spans" in w for w in p.warnings)
    assert any("Jet “pin” is" in w and "grid cells wide" in w for w in p.warnings)
    assert any("Heat source “dot”" in w and "under two grid cells" in w for w in p.warnings)


def test_items_outside_the_domain_are_left_out_with_a_warning():
    des = cylinder(0.2)
    p = plan_simulation(des, tunnel_sim(des, 2.0, fans=[Fan(name="far", position=(50.0, 0.0), diameter=0.1,
                                                            value=0.1, frame="wind")]))
    assert p.solver_config.fans == [] and any("“far” is outside the domain" in w for w in p.warnings)


def test_a_closed_box_whose_jets_and_vents_do_not_balance_is_reported():
    des = room2d()
    jet = Port(name="in", kind="jet", position=(0.5, 2.5), direction=(1.0, 0.0), size=0.2, speed=1.0, frame="wind")
    vent = Port(name="out", kind="vent", position=(3.5, 0.5), direction=(1.0, 0.0), size=0.2, speed=0.5, frame="wind")
    p = plan_simulation(des, closed_sim(des, (0, 4, 0, 3), ports=[jet, vent]))
    assert any("must balance" in w and "0.200" in w and "0.100" in w for w in p.warnings)
    vent.speed = 1.0
    p2 = plan_simulation(des, closed_sim(des, (0, 4, 0, 3), ports=[jet, vent]))
    assert not any("must balance" in w for w in p2.warnings)
    assert p2.speed == pytest.approx(1.0) and "jet “in”" in p2.physics.speed_basis


def test_smoke_maps_to_channels_by_colour():
    des = cylinder(0.2)
    cols = ["#f5f5f5", "#e04040", "#e04040", "#4060e0"]
    ems = [Emitter(name=f"e{i}", species="dye", position=(-0.2, 0.05 * i), color=c) for i, c in enumerate(cols)]
    p = plan_simulation(des, tunnel_sim(des, 2.0, emitters=ems))
    chans = [e.channel for e in p.solver_config.emitters]
    assert chans[1] == chans[2] and len(set(chans)) == 3 and len(p.placed.dye_channels) == 3
    assert [c["field"] for c in p.placed.dye_channels] == ["dye", "dye1", "dye2"]
    many = [Emitter(name=f"m{i}", species="dye", position=(-0.2, 0.02 * i), color=c)
            for i, c in enumerate(["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff"])]
    p2 = plan_simulation(des, tunnel_sim(des, 2.0, emitters=many))
    assert any(w.startswith("5 smoke colours, but the solver carries 4") for w in p2.warnings)
    assert max(e.channel for e in p2.solver_config.emitters) <= 3


def test_physics_costs_and_time_step_limits():
    c2 = PH.PHYSICS_COST[2]
    des = cylinder(0.2, t_body=40.0)
    sim = tunnel_sim(des, 1.0, thermal=ThermalSettings(enabled=True, t_ambient=15.0),
                     emitters=[Emitter(species="dye", position=(-0.2, 0.0))],
                     ports=[Port(name="J", position=(0.6, 0.0), size=0.05, speed=4.0, frame="wind")])
    p = plan_simulation(des, sim)
    assert p.step_cost == pytest.approx(1 + c2["diagnostics"] + c2["thermal"] + c2["buoyancy"] + c2["smoke"]
                                        + c2["ports"])
    # the jet's own speed (4 U) enters the advective bound, and the temperature step its positivity bound
    h = p.solver_config.h
    assert p.dt <= planmod.CFL * h / 4.0 * (1 + 1e-12)
    s = p.physics
    # the 4 U jet outruns the stream: the step is planned on the jet's speed (critic round 3.2)
    assert p.calibration_kind == "generic-jet" and p.speed_ratio == pytest.approx(4.0)
    kfac = planmod._speed_factor(p.calibration_kind, "laminar") * p.speed_ratio
    rate = kfac * 2.0 / h + PH._diffusion_coef2(s, sim, False) / s.pe / h ** 2
    assert PH.physics_dt(h, kfac, s, p.placed, sim, closed=False, dim=2) <= 1.0 / rate * (1 + 1e-12)


def test_plan_live_keeps_the_placed_physics():
    des = cylinder(0.2, t_body=40.0)
    sim = tunnel_sim(des, 1.0, thermal=ThermalSettings(enabled=True, t_ambient=15.0),
                     fans=[Fan(position=(0.6, 0.0), diameter=0.1, mode="velocity", value=1.0, frame="wind")])
    p = plan_live(des, sim)
    assert p.live and p.physics is not None and p.placed.fans and p.solver_config.thermal is not None
    rows = p.summary()
    assert "Duration" not in rows and "Temperature scale" in rows and "Fans" in rows
    assert p.monitors == ("nu", "fan0_thrust")                    # a speed-controlled fan: its thrust


def test_3d_closed_cubic_cavity_plan():
    des, sim, _ = cavity_sim(1e4, dim=3)
    p = plan_simulation(des, sim)
    cfg = p.solver_config
    assert p.dimension == 3 and p.closed and not p.symmetry_plane
    assert cfg.domain == "closed" and cfg.thermal.richardson == pytest.approx(1.0)
    assert cfg.thermal.walls == {"left": ("temperature", pytest.approx(0.5)), "right": ("temperature", pytest.approx(-0.5))}
    assert p.physics.ra == pytest.approx(1e4, rel=1e-9) and p.monitors == ("nu", "ke")
    nx, ny, nz = p.grid3
    assert nx == ny == nz and nz * cfg.h == pytest.approx(1.0)
    assert any("gravity along −z" in n for n in p.notes)


def test_3d_live_plan_with_placed_physics_runs_the_full_domain():
    des = design_from_meshes([sphere(0.2, 24)], "S", kind="generic")
    sim = tunnel_sim(des, 1.0, emitters=[Emitter(species="particles", kind="point", position=(-0.2, 0.0, 0.0),
                                                 rate=50.0)])
    p = plan_live(des, sim)
    assert p.live and not p.symmetry_plane and p.placed.particle_palette
    assert p.solver_config.emitters and p.solver_config.emitters[0].species == "particles"
