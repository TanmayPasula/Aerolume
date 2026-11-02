"""simulation.defaults: the names new simulations get (one rule for the API, the AI tools and the examples)."""
from __future__ import annotations

import pytest

import aerolume.geometry3d as G
from aerolume.geometry import design_from_airfoil, naca
from aerolume.model import FlowConditions, Fluid
from aerolume.simulation.defaults import (
    NAME_BY_RE_BELOW_SPEED,
    default_simulation,
    names_by_reynolds,
    reynolds_text,
    simulation_name,
)


def _sphere(d: float = 0.2):
    return G.design_from_meshes([G.sphere(d, 32)], "Sphere 0.2 m", "generic")


def _at_re(design, re_: float) -> FlowConditions:
    fluid = Fluid()
    return FlowConditions(fluid=fluid, speed=re_ * fluid.viscosity / (fluid.density * design.length_ref))


@pytest.mark.parametrize("re_, text", [(100, "100"), (2500, "2,500"), (1.0e6, "1.0×10⁶"), (9.96e5, "1.0×10⁶"),
                                       (3.2e7, "3.2×10⁷"), (0, "—"), (float("nan"), "—"), ("x", "—")])
def test_reynolds_text(re_, text):
    assert reynolds_text(re_) == text


def test_a_low_reynolds_number_run_is_named_by_re_not_by_a_tiny_speed():
    """The scripting example's sphere at Re 100 flies at 0.0073 m/s in air; "CFD 3D ·
    0.0073 m/s" looked like a typo in the tree, Compare and the report."""
    d = _sphere()
    sim = default_simulation(d, conditions=_at_re(d, 100.0))
    assert sim.conditions.speed < NAME_BY_RE_BELOW_SPEED
    assert names_by_reynolds(d, sim)
    assert simulation_name(d, sim) == "CFD 3D · Re 100 · α 0°"
    # the caller knows best: a speed given as a speed stays a speed, an Re stays an Re
    assert simulation_name(d, sim, by_reynolds=False) == "CFD 3D · 0.0073 m/s · α 0°"
    fast = default_simulation(d, conditions=FlowConditions(speed=30.0))
    assert simulation_name(d, fast) == "CFD 3D · 30 m/s · α 0°"
    assert simulation_name(d, fast, by_reynolds=True) == "CFD 3D · Re 4.1×10⁵ · α 0°"


def test_names_match_the_gui_style_for_every_analysis():
    car = G.design_from_meshes(G.sedan(), "Sedan", "vehicle")
    sim = default_simulation(car, conditions=FlowConditions(speed=30.0, alpha_deg=-1.5, yaw_deg=5.0))
    sim.cfd.quality = "standard"
    sim.cfd.compute_sensitivity = True
    assert simulation_name(car, sim) == "CFD 3D · 30 m/s · α -1.5° · yaw 5° · Standard · sensitivity cd"

    wing = G.design_from_wing(G.tapered_wing(3.0, 0.4, 0.2), "Wing")
    ws = default_simulation(wing, conditions=FlowConditions(speed=25.0))
    assert ws.analysis == "wing"
    assert simulation_name(wing, ws) == "Wing polar · 25 m/s"
    ws.wing.method, ws.wing.viscous = "llt", False
    assert simulation_name(wing, ws) == "Wing polar · 25 m/s · LLT · inviscid"

    foil = design_from_airfoil(naca("2412"), "NACA 2412", 1.0)
    polar = default_simulation(foil, conditions=_at_re(foil, 1.0e6))
    polar.analysis = "polar"
    assert simulation_name(foil, polar) == "Polar · Re 1.0×10⁶"
    cfd2d = default_simulation(foil, conditions=FlowConditions(speed=20.0, alpha_deg=4.0))
    assert simulation_name(foil, cfd2d) == "CFD · 20 m/s · α 4°"
