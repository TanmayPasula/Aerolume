"""Setup of the placed physics (DESIGN.md section 10): editors, placement on the canvases, templates.

The pure helpers of :mod:`aerolume.app.widgets.physics_editor` are checked against the
definitions they implement (the planner's attitude rotation, the dimensionless groups of the
benchmarks); the editors and the case previews are driven like a user drives them (widgets,
mouse drags, the keyboard) against a real temporary project, and every assertion is made on the
:class:`~aerolume.model.Simulation` / :class:`~aerolume.model.Design` the runner reads.
"""
from __future__ import annotations

import itertools
import math
import os
import re

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from aerolume.core import fluid as fluids
from aerolume.model import (
    Body,
    Design,
    Emitter,
    Fan,
    FlowConditions,
    HeatSource,
    Port,
    Simulation,
    WallThermal,
)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    from aerolume.app import theme
    theme.apply(app)
    yield app


@pytest.fixture
def state(qapp, tmp_path):
    from aerolume.app.state import AppState
    from aerolume.project import Project
    st = AppState()
    st.set_project(Project.create(tmp_path / "Physics", "Physics"))
    yield st


@pytest.fixture
def planner():
    from aerolume.app.pages import setup as setup_mod
    loader = setup_mod._PlannerLoader.get()
    if not loader.wait(180):
        pytest.skip(f"planner not importable: {loader.error}")
    return loader


def _process(qapp, n=6):
    for _ in range(n):
        qapp.processEvents()


def _square(side=0.1, n=8):
    t = np.linspace(0, 1, n, endpoint=False)
    s = side / 2
    return np.vstack([np.c_[-s + 2 * s * t, np.full(n, -s)], np.c_[np.full(n, s), -s + 2 * s * t],
                      np.c_[s - 2 * s * t, np.full(n, s)], np.c_[np.full(n, -s), s - 2 * s * t]])


def _design2d():
    return Design(name="Block", kind="generic", bodies=[Body("Block", _square(0.1))], length_ref=0.1, force_ref=0.1,
                  moment_ref=(0.0, 0.0))


def _box3d():
    from aerolume.geometry3d import box, design_from_meshes
    return design_from_meshes([box(0.4, 0.2, 0.2, center=(0.0, 0.0, 0.3), name="Box")], "Box", "generic")


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def test_wind_frame_matches_the_planners_rotation_in_2d_and_3d():
    from aerolume.app.widgets import physics_editor as pe
    from aerolume.simulation.plan import rotate_points
    from aerolume.simulation.plan3d import rotation3
    d2 = _design2d()
    d2.moment_ref = (0.02, -0.01)
    sim = Simulation(name="s", design_id=d2.id, conditions=FlowConditions(alpha_deg=12.0))
    p = np.array([[0.3, 0.1], [-0.2, 0.05]])
    np.testing.assert_allclose(pe.to_wind(p, d2, sim, "body"), rotate_points(p, -12.0, d2.moment_ref), atol=1e-12)
    np.testing.assert_allclose(pe.to_wind(p, d2, sim, "wind"), p)                    # fixed in the tunnel
    np.testing.assert_allclose(pe.from_wind(pe.to_wind(p, d2, sim), d2, sim), p, atol=1e-12)
    d3 = _box3d()
    d3.moment_ref3 = (0.1, 0.0, 0.2)
    sim3 = Simulation(name="s", design_id=d3.id, conditions=FlowConditions(alpha_deg=8.0, yaw_deg=-5.0))
    q = np.array([[0.5, 0.2, 0.1], [-0.3, -0.1, 0.4]])
    ref = np.asarray(d3.moment_ref3)
    np.testing.assert_allclose(pe.to_wind(q, d3, sim3), (q - ref) @ rotation3(8.0, -5.0).T + ref, atol=1e-12)
    np.testing.assert_allclose(pe.from_wind(pe.to_wind(q, d3, sim3), d3, sim3), q, atol=1e-12)
    v = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(pe.to_wind(v, d3, sim3, direction=True), v @ rotation3(8.0, -5.0).T, atol=1e-12)


def test_snapping_steps_are_round():
    from aerolume.app.widgets import physics_editor as pe
    assert pe.snap_step(0.0013) == pytest.approx(0.002)
    assert pe.snap_step(0.004) == pytest.approx(0.005)
    assert pe.snap_step(0.009) == pytest.approx(0.01)
    assert pe.snap(0.1234, 0.005) == pytest.approx(0.125)
    assert pe.snap(-0.0001, 0.01) == 0.0 and str(pe.snap(-0.0001, 0.01)) == "0.0"


# ---------------------------------------------------------------------------
# heat transfer: properties and the dimensionless groups
# ---------------------------------------------------------------------------

def test_fluid_defaults_air_and_water():
    from aerolume.app.widgets import physics_editor as pe
    air, basis = pe.fluid_defaults(fluids.air(20.0), 20.0)
    # textbook air at 20 C: k = 0.0257 W/(m K), Pr = 0.71, beta = 1/T
    assert air["conductivity"] == pytest.approx(0.0257, rel=0.01)
    assert air["prandtl"] == pytest.approx(0.71, rel=0.01)
    assert air["beta"] == pytest.approx(1 / 293.15)
    assert "Sutherland" in basis
    water, basis_w = pe.fluid_defaults(fluids.water(20.0), 20.0)
    assert water["prandtl"] == pytest.approx(7.0, rel=0.02)              # Pr of water at 20 C
    assert water["beta"] == pytest.approx(2.07e-4, rel=1e-6)
    assert "Incropera" in basis_w
    none, why = pe.fluid_defaults(fluids.Fluid(name="Custom fluid", density=900.0, viscosity=0.1), 20.0)
    assert none is None and "custom" in why


def test_cavity_template_is_de_vahl_davis_at_the_rayleigh_number_asked_for():
    from aerolume.app.dialogs.physics_templates import (
        CAVITY_NU_AVERAGE,
        CAVITY_NU_HOT_WALL,
        build_template,
        cavity_side,
    )
    from aerolume.app.widgets import physics_editor as pe
    for ra in (1e3, 1e4, 1e5, 1e6):
        d, sims = build_template("cavity", ra=ra)
        sim = sims[0]
        assert d.bodies == [] and sim.cfd.domain_mode == "closed"
        side = d.length_ref
        assert sim.cfd.domain_box == pytest.approx((0.0, side, 0.0, side))
        walls = {w.side: (w.mode, w.value) for w in sim.thermal.walls}
        assert walls == {"left": ("temperature", 25.0), "right": ("temperature", 15.0)}
        tn = pe.thermal_numbers(d, sim)
        assert tn.delta_t == pytest.approx(10.0)                 # T_hot - T_cold, the benchmark's definition
        assert tn.length == pytest.approx(side) and tn.pr == pytest.approx(0.71)
        assert tn.ra == pytest.approx(ra, rel=1e-9)
        assert d.source["nu_hot_wall_reference"] == CAVITY_NU_HOT_WALL[ra]      # Nu0: what the run measures
        assert d.source["nu_cavity_average"] == CAVITY_NU_AVERAGE[ra]
        assert f"Nu₀ = {CAVITY_NU_HOT_WALL[ra]:.3f}" in sim.notes
        assert tn.regime == ("Conduction" if ra < 1e3 else "Natural")
        # the free-fall velocity is the time scale of the run
        assert sim.conditions.speed == pytest.approx(math.sqrt(9.80665 / 293.15 * 10.0 * side), rel=1e-3)
    air = fluids.air(20.0)
    assert cavity_side(1e5, air, t_mean_c=20.0, delta_t=10.0) == pytest.approx(0.0457, rel=0.01)


def test_heated_cylinder_is_forced_convection_at_re_40():
    from aerolume.app.dialogs.physics_templates import build_template
    from aerolume.app.widgets import physics_editor as pe
    d, sims = build_template("heated_cylinder")
    sim = sims[0]
    assert sim.conditions.reynolds(d.length_ref) == pytest.approx(40.0, rel=1e-9)
    assert all(b.thermal == "temperature" and b.thermal_value == 30.0 for b in d.bodies)
    assert sim.thermal.enabled and not sim.thermal.buoyancy
    tn = pe.thermal_numbers(d, sim)
    assert tn.delta_t == pytest.approx(10.0) and tn.re == pytest.approx(40.0, rel=1e-9)
    assert tn.pe == pytest.approx(40.0 * tn.pr) and tn.ri is None
    assert tn.regime == "Forced (no buoyancy)"
    keys = [r[0] for r in pe.thermal_rows(tn)]
    assert "Rayleigh number" not in keys and "Péclet number" in keys          # no buoyancy: no Gr / Ra
    # the published range and the correlation beside it, not one number (docs/validation.md does the same)
    assert d.source["nu_reference_range"] == (3.2, 3.255)
    assert d.source["nu_correlation"] == pytest.approx(3.2805, abs=1e-4)
    assert "3.200–3.255" in sim.notes and "gives 3.28 " in sim.notes and "0.8–2.5 % above" in sim.notes


def test_template_references_match_the_api_and_the_validation_suite():
    """One measured quantity, one reference: the template, the API and the validation suite agree.

    The critique found the template quoting de Vahl Davis' cavity-average Nū (4.519) where the API
    and the validation compare the hot wall's Nu0 (4.509), the quantity the solver measures."""
    import importlib.util
    import sys
    from pathlib import Path

    from aerolume import api
    from aerolume.app.dialogs import physics_templates as pt
    assert pt.CAVITY_NU_HOT_WALL == api.CAVITY_NU_2D
    for re_ in (1.0, 20.0, 40.0, 150.0):
        assert pt.lange_nusselt(re_) == pytest.approx(api.lange_nusselt(re_), rel=1e-15)
    path = Path(__file__).resolve().parents[1] / "validation" / "thermal2d_cases.py"
    if path.is_file():                         # the validation suite ships with the source tree only
        spec = importlib.util.spec_from_file_location("_thermal2d_cases_ref", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod             # its dataclasses look their module up while it loads
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(spec.name, None)
        assert pt.CAVITY_NU_HOT_WALL == {ra: v["nu_0"] for ra, v in mod.DE_VAHL_DAVIS.items()}
        assert pt.CAVITY_NU_AVERAGE == {ra: v["nu_bar"] for ra, v in mod.DE_VAHL_DAVIS.items()}
        assert pt.CYLINDER_NU_PUBLISHED == mod.CYLINDER_NU_REF[40.0]
    texts = " ".join(t.description for t in pt.PHYSICS_TEMPLATES)
    assert "4.519" not in texts and "Nu₀ 4.509" in texts and "3.200–3.255" in texts


def test_richardson_number_sets_the_regime():
    from aerolume.app.widgets import physics_editor as pe
    d = _design2d()
    d.bodies[0].thermal, d.bodies[0].thermal_value = "temperature", 40.0
    sim = Simulation(name="s", design_id=d.id, conditions=FlowConditions(fluid=fluids.air(20.0), speed=5.0))
    sim.thermal.enabled = True
    tn = pe.thermal_numbers(d, sim)
    g_beta = 9.80665 / 293.15
    assert tn.ri == pytest.approx(g_beta * 20.0 * 0.1 / 25.0)              # g beta dT L / U^2
    assert tn.gr / tn.re ** 2 == pytest.approx(tn.ri)
    assert tn.regime == "Forced"
    sim.conditions.speed = 0.05
    assert pe.thermal_numbers(d, sim).regime == "Natural"
    sim.thermal.enabled = False
    assert pe.thermal_numbers(d, sim) is None and pe.thermal_rows(None) == []


def test_heat_flux_scale_is_the_planners_with_a_correlation_fallback(monkeypatch):
    """A heater's temperature scale is the planner's (the air carried through the heater), so the
    Estimate rows, the Planner list and the run quote one ΔT; without the planner it is the page's
    own q L / (k Nu) from a laminar correlation. Never the conduction bound q L / k."""
    from aerolume.app.widgets import physics_editor as pe
    from aerolume.simulation import physics as sim_physics
    d = _design2d()
    sim = Simulation(name="s", design_id=d.id, conditions=FlowConditions(fluid=fluids.air(20.0), speed=1.0))
    sim.thermal.enabled = True
    sim.heat_sources = [HeatSource(kind="rect", position=(-0.3, 0.0), size=(0.02, 0.02), power=10.0)]
    tn = pe.thermal_numbers(d, sim)
    props = tn.props
    q = 10.0 / 0.08
    conduction = q * 0.02 / props.conductivity
    planned = sim_physics.thermal_estimate(d, sim)
    assert tn.delta_t == pytest.approx(planned.delta_t) and tn.delta_t_basis == planned.delta_t_basis
    assert 0 < tn.delta_t < conduction / 5

    def unavailable(*_a, **_k):
        raise RuntimeError("planner unavailable")
    monkeypatch.setattr(sim_physics, "thermal_estimate", unavailable)
    own = pe.thermal_numbers(d, sim)
    assert 0 < own.delta_t < conduction / 5 and "correlation" in own.delta_t_basis


# ---------------------------------------------------------------------------
# planner-style warnings
# ---------------------------------------------------------------------------

def test_closed_box_balance_and_drivers():
    from aerolume.app.widgets import physics_editor as pe
    d = _design2d()
    sim = Simulation(name="s", design_id=d.id)
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (-0.5, 0.5, -0.3, 0.3)
    w, _ = pe.physics_warnings(d, sim)
    assert any("Nothing drives the flow" in x for x in w)
    sim.ports = [Port(kind="jet", position=(-0.45, 0.0), size=0.05, speed=2.0)]
    w, _ = pe.physics_warnings(d, sim)
    assert any("supply 0.100 m²/s per metre of span and the vents remove 0" in x for x in w)
    sim.ports.append(Port(kind="vent", position=(0.45, 0.0), size=0.1, speed=1.0))
    w, _ = pe.physics_warnings(d, sim)
    assert not any("must balance" in x for x in w) and not any("Nothing drives" in x for x in w)
    sim.fans = [Fan(position=(2.0, 0.0), direction=(1.0, 0.0))]                # outside the box
    w, _ = pe.physics_warnings(d, sim)
    assert any("Fan “Fan” is outside the domain" in x for x in w)
    sim.fans[0].enabled = False
    w, _ = pe.physics_warnings(d, sim)
    assert not any("outside the domain" in x for x in w)


def test_boussinesq_and_custom_fluid_warnings():
    from aerolume.app.widgets import physics_editor as pe
    d = _design2d()
    d.bodies[0].thermal, d.bodies[0].thermal_value = "temperature", 120.0
    sim = Simulation(name="s", design_id=d.id, conditions=FlowConditions(fluid=fluids.air(20.0), speed=1.0))
    sim.thermal.enabled = True
    w, _ = pe.physics_warnings(d, sim)
    assert any(x.startswith("β·ΔT = 0.34") for x in w)
    d.bodies[0].thermal_value = 20.0
    w, _ = pe.physics_warnings(d, sim)
    assert any("nothing is heated or cooled" in x for x in w)
    sim.conditions.fluid = fluids.Fluid(name="Custom fluid", density=900.0, viscosity=0.01)
    w, _ = pe.physics_warnings(d, sim)
    assert any(x.startswith("Custom fluid: type the Prandtl number") for x in w)


def test_planner_gaps_are_reported_until_the_planner_carries_the_settings():
    from types import SimpleNamespace

    from aerolume.app.pages.setup import planner_gaps
    sim = Simulation(name="s", design_id="d")
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (0.0, 1.0, 0.0, 0.5)
    sim.thermal.enabled = True
    old = SimpleNamespace(dimension=2, summary=dict)
    # an old planner of a simulation layer that publishes no physics (the installed one carries heat now)
    gaps = planner_gaps(sim, old, (-1.0, 3.0, -1.0, 1.0), [("Grid", "10 × 10")], supported=frozenset())
    assert len(gaps) == 3 and "4 × 2 m domain" in gaps[0]
    new = SimpleNamespace(dimension=2, domain_mode="closed")
    assert planner_gaps(sim, new, (0.0, 1.0, 0.0, 0.5), [("Rayleigh number", "1e5")]) == []


def test_placed_items_the_planner_would_drop_are_named():
    """Fans, jets and vents, heaters and 2D emitters that the installed planner does not hand to the
    solver are named in one line; once the simulation layer publishes a feature (or the plan carries
    it) the line drops it. 3D emitters ride the seed rake, which keeps only their nozzles."""
    from types import SimpleNamespace

    from aerolume.app.pages.setup import planner_gaps
    sim = Simulation(name="s", design_id="d")
    sim.fans = [Fan(position=(0.0, 0.0))]
    sim.ports = [Port(position=(0.1, 0.0)), Port(kind="vent", position=(0.2, 0.0))]
    sim.emitters = [Emitter(position=(-0.1, 0.0))]
    sim.thermal.enabled = True
    sim.heat_sources = [HeatSource(position=(0.0, 0.1), size=(0.01, 0.01))]
    plan2 = SimpleNamespace(dimension=2, domain_mode="tunnel")
    rows = [("Rayleigh number", "1e5")]
    gaps = planner_gaps(sim, plan2, None, rows, supported=frozenset())
    assert gaps == [("The installed planner does not pass fans, jets, vents, heat sources and emitters to the "
                     "solver yet: this run would leave them out.")]
    every = frozenset({"fans", "ports", "heat_sources", "emitters", "thermal"})
    assert planner_gaps(sim, plan2, None, rows, supported=every) == []
    # a plan that holds the fans carries them, whatever the published set says
    with_fans = SimpleNamespace(dimension=2, domain_mode="tunnel", fans=[object()])
    assert "pass jets, vents, heat sources and emitters" in planner_gaps(sim, with_fans, None, rows,
                                                                          supported=frozenset())[0]
    # heat transfer not planned: its own line says the run solves the flow only (heaters included)
    gaps = planner_gaps(sim, plan2, None, [], supported=frozenset())
    assert gaps[0].startswith("The installed planner cannot plan heat transfer yet")
    assert "heat sources" not in gaps[1]
    # 3D: emitters ride the seed rake; what it cannot keep is said, a plain smoke rake needs no line
    sim3 = Simulation(name="s3", design_id="d")
    sim3.emitters = [Emitter(kind="rake", position=(-0.5, -0.2, 0.1), end=(-0.5, 0.2, 0.1))]
    plan3 = SimpleNamespace(dimension=3, domain_mode="tunnel")
    assert planner_gaps(sim3, plan3, None, [], supported=frozenset()) == []
    sim3.emitters.append(Emitter(species="particles", position=(-0.5, 0.0, 0.3), color="#f2a33a", start=1.0))
    (line,) = planner_gaps(sim3, plan3, None, [], supported=frozenset())
    assert line == ("The installed planner passes the emitters as one smoke rake at their nozzles: particles "
                    "become smoke, all colours become one and rates and start / stop times are ignored.")


def test_heat_without_heat_transfer_is_flagged():
    from aerolume.app.widgets import physics_editor as pe
    d = _design2d()
    sim = Simulation(name="s", design_id=d.id)
    sim.heat_sources = [HeatSource(name="Chip", position=(0.0, 0.2), size=(0.02, 0.02), power=5.0)]
    w, _ = pe.physics_warnings(d, sim)
    assert any(x.startswith("Heat transfer is off, so “Chip” adds no heat") for x in w)
    d.bodies[0].thermal = "temperature"
    w, _ = pe.physics_warnings(d, sim)
    assert any("“Chip”, “Block” add no heat" in x for x in w)
    sim.thermal.enabled = True
    w, _ = pe.physics_warnings(d, sim)
    assert not any(x.startswith("Heat transfer is off") for x in w)


# ---------------------------------------------------------------------------
# drawing specs and drags
# ---------------------------------------------------------------------------

def test_specs_and_drags_keep_body_items_on_the_pitched_body():
    from aerolume.app.widgets import physics_editor as pe
    d = _design2d()
    sim = Simulation(name="s", design_id=d.id, conditions=FlowConditions(alpha_deg=10.0))
    sim.heat_sources = [HeatSource(kind="rect", position=(0.1, 0.0), size=(0.02, 0.02), frame="body")]
    sim.emitters = [Emitter(kind="rake", position=(-0.3, -0.1), end=(-0.3, 0.1), count=5, frame="wind")]
    specs = pe.scene_specs(d, sim, selected=("heat_sources", 0))
    heat = next(s for s in specs if s["type"] == "heat_source")
    assert heat["selected"]
    np.testing.assert_allclose(heat["centre"], pe.to_wind([0.1, 0.0], d, sim)[0])    # turned with the body
    em = next(s for s in specs if s["type"] == "emitter")
    np.testing.assert_allclose(em["points"][[0, -1]], [[-0.3, -0.1], [-0.3, 0.1]])    # the tunnel keeps it
    assert len(em["points"]) == 5 and set(em["handles"]) == {"centre", "start", "end"}
    dom = specs[-1]
    assert dom["type"] == "domain" and not dom["explicit"]
    # a drag in the preview (wind frame) stores the design-frame position that draws there
    assert pe.apply_drag(d, sim, ("heat_sources", 0), "centre", (0.2, 0.05))
    np.testing.assert_allclose(pe.to_wind(sim.heat_sources[0].position, d, sim)[0], (0.2, 0.05), atol=1e-12)
    assert pe.apply_drag(d, sim, ("emitters", 0), "end", (-0.25, 0.2))
    assert sim.emitters[0].end == pytest.approx((-0.25, 0.2)) and sim.emitters[0].position == pytest.approx((-0.3, -0.1))
    assert pe.apply_drag(d, sim, ("emitters", 0), "centre", (0.0, 0.0))     # the whole rake moves
    e = sim.emitters[0]
    np.testing.assert_allclose(0.5 * (np.array(e.position) + np.array(e.end)), (0.0, 0.0), atol=1e-12)
    # a domain wall drag fixes the extent at the drawn box, with that wall moved
    planned = (-1.0, 2.0, -0.6, 0.6)
    assert pe.apply_drag(d, sim, ("domain",), "x1", (2.5, 0.0), extent=planned)
    assert sim.cfd.domain_box == pytest.approx((-1.0, 2.5, -0.6, 0.6))
    assert pe.apply_drag(d, sim, ("domain",), "x0y1", (-1.2, 0.8))
    assert sim.cfd.domain_box == pytest.approx((-1.2, 2.5, -0.6, 0.8))
    assert not pe.apply_drag(d, sim, ("fans", 3), "centre", (0.0, 0.0))        # no such item


def test_new_items_have_sensible_defaults():
    from aerolume.app.widgets import physics_editor as pe
    d = _design2d()
    sim = Simulation(name="s", design_id=d.id)
    e = pe.new_item("emitters", d, sim)
    assert e.kind == "rake" and e.position[0] < -0.05 and e.end[1] > e.position[1] and e.frame == "wind"
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (-0.5, 0.5, -0.3, 0.3)
    jet = pe.new_item("ports", d, sim)
    sim.ports.append(jet)
    vent = pe.new_item("ports", d, sim)                        # the pair: a vent follows a jet
    assert jet.kind == "jet" and vent.kind == "vent" and vent.position[0] > jet.position[0]
    sim.ports.append(vent)
    assert not any("must balance" in w for w in pe.physics_warnings(d, sim)[0])
    fan = pe.new_item("fans", d, sim)
    assert fan.direction == (1.0, 0.0) and -0.5 < fan.position[0] < 0.0
    heat = pe.new_item("heat_sources", d, sim)
    assert heat.kind == "rect" and heat.position[1] < 0.0
    dup = pe.copy_item(heat, {"Heater"})
    assert dup.name == "Heater copy" and dup.position[1] > heat.position[1]
    d3 = _box3d()
    s3 = Simulation(name="s3", design_id=d3.id)
    e3 = pe.new_item("emitters", d3, s3)
    b = d3.bounds3()
    assert len(e3.position) == 3 and e3.position[0] < b[0] and e3.position[2] == pytest.approx(b[4] + 0.4 * (b[5] - b[4]))


# ---------------------------------------------------------------------------
# the hand-over to the 3D runner's seed rake
# ---------------------------------------------------------------------------

def test_emitters_hand_over_as_a_seed_rake():
    from aerolume.app.pages import setup as S
    sim = Simulation(name="s", design_id="d")
    assert "spec" not in S.flow_viz_of(sim)                        # no emitters: the automatic rake
    sim.emitters = [Emitter(kind="rake", position=(-0.5, -0.3, 0.2), end=(-0.5, 0.3, 0.2), count=7, radius=0.01,
                            pulse_period=0.5, pulse_duty=0.4)]
    out = S.stored_flow_viz(S.flow_viz_of(sim))
    assert out == {"smoke": True, "rake": {"kind": "rake", "x": -0.5, "y": [-0.3, 0.3], "z": 0.2, "n": 7,
                                           "radius": 0.01, "pulse": [pytest.approx(0.2), pytest.approx(0.3)]}}
    sim.emitters[0].end = (-0.5, -0.3, 0.9)
    assert S.stored_flow_viz(S.flow_viz_of(sim))["rake"]["kind"] == "vertical"
    sim.emitters = [Emitter(kind="plane", position=(-0.4, 0.0, 0.5), size=(0.6, 0.4), normal=(1.0, 0.0, 0.0))]
    rake = S.stored_flow_viz(S.flow_viz_of(sim))["rake"]
    assert rake == {"kind": "plane", "x": -0.4, "y": [-0.3, 0.3], "z": [pytest.approx(0.3), pytest.approx(0.7)]}
    sim.emitters.append(Emitter(kind="point", position=(-0.4, 0.1, 0.1)))
    rake = S.stored_flow_viz(S.flow_viz_of(sim))["rake"]
    assert rake["kind"] == "points" and len(rake["points"]) == 26        # a 5 x 5 band and one nozzle
    for e in sim.emitters:
        e.enabled = False
    assert S.stored_flow_viz(S.flow_viz_of(sim)) == {"smoke": False, "rake": None}
    # the Simulate page and the job manager read it through view3d_panel
    from aerolume.app.widgets.view3d_panel import RakeSpec, setup_flow_viz
    sim.emitters = [Emitter(kind="rake", position=(-0.5, -0.3, 0.2), end=(-0.5, 0.3, 0.2), count=7)]
    fv = setup_flow_viz(sim)
    spec = RakeSpec.from_view(fv["rake"])
    assert spec is not None and spec.kind == "rake" and spec.n == 7 and spec.x == pytest.approx(-0.5)


def test_automatic_rake_converts_to_emitters_where_it_stands():
    from aerolume.app.pages.setup import emitters_from_layout
    from aerolume.app.widgets.geometry_canvas import smoke_rake_layout
    box = (0.0, 4.0, -0.9, 0.9, 0.0, 1.4)
    domain = (-8.0, 20.0, -4.0, 4.0, 0.0, 5.0)
    lay = smoke_rake_layout({"kind": "rake"}, box, extent3=domain, cell=0.03)
    (e,) = emitters_from_layout(lay)
    np.testing.assert_allclose(np.vstack([np.linspace(e.position, e.end, e.count)]), lay["points"], atol=1e-12)
    vert = emitters_from_layout(smoke_rake_layout({"kind": "vertical"}, box, extent3=domain, cell=0.03))
    assert len(vert) == 1 and vert[0].position[2] < vert[0].end[2]
    grid_lay = smoke_rake_layout({"kind": "grid"}, box, extent3=domain, cell=0.03)
    rows = emitters_from_layout(grid_lay)
    assert len(rows) == grid_lay["n"][1] and sum(r.count for r in rows) == grid_lay["count"]
    (plane,) = emitters_from_layout(smoke_rake_layout({"kind": "plane"}, box, extent3=domain), smoke_on=False)
    assert plane.kind == "plane" and not plane.enabled and plane.normal == (1.0, 0.0, 0.0)
    assert emitters_from_layout(None) == [] and emitters_from_layout({"error": "x"}) == []


# ---------------------------------------------------------------------------
# templates and new simulations
# ---------------------------------------------------------------------------

def test_templates_build_round_trip_and_add_through_appstate(state):
    from aerolume.app.dialogs.physics_templates import PHYSICS_TEMPLATES, add_template, build_template
    for t in PHYSICS_TEMPLATES:
        d, sims = build_template(t.key)
        assert d.dimension == t.dimension and sims and all(s.design_id == d.id for s in sims)
        for s in sims:
            again = Simulation.from_dict(s.to_dict())
            assert again.to_dict() == s.to_dict()
            assert s.thermal.enabled
    d, sims = add_template(state, "enclosure")
    assert state.design is d and state.simulation is sims[0]
    sim = state.simulation
    assert [p.kind for p in sim.ports] == ["jet", "vent"] and sim.fans and sim.heat_sources
    d2, _sims2 = add_template(state, "enclosure")
    assert d2.name == "Electronics enclosure 2"
    # saved: a reopened project has every item
    from aerolume.project import Project
    reopened = Project.open(state.project.path)
    s = reopened.simulations[sim.id]
    assert s.to_dict() == sim.to_dict() and reopened.designs[d.id].bodies[0].name == "Circuit board"


def test_room_template_is_balanced_and_inside_the_boussinesq_range():
    from aerolume.app.dialogs.physics_templates import build_template
    from aerolume.app.widgets import physics_editor as pe
    d, (sim,) = build_template("room")
    warns, _notes = pe.physics_warnings(d, sim)
    assert warns == []
    tn = pe.thermal_numbers(d, sim)
    assert tn.beta_dt < 0.1 and tn.length == pytest.approx(2.5)


def test_room_template_cabinet_stands_on_the_floor_clear_of_the_radiator():
    """design_from_meshes centres a generic shape on the origin; the room's cabinet must be moved back
    into the room's own coordinates (it used to straddle the window wall and sink under the floor)."""
    from aerolume.app.dialogs.physics_templates import ROOM_CABINET_CENTRE, build_template
    d, (sim,) = build_template("room")
    v = np.vstack([m.vertices for m in d.meshes])
    lo, hi = v.min(axis=0), v.max(axis=0)
    box = np.asarray(sim.cfd.domain_box).reshape(3, 2)
    assert np.all(lo >= box[:, 0]) and np.all(hi <= box[:, 1])            # inside the room
    assert lo[2] == pytest.approx(0.0, abs=1e-12)                          # standing on the floor
    np.testing.assert_allclose(0.5 * (lo + hi), ROOM_CABINET_CENTRE, atol=1e-12)
    rad = sim.heat_sources[0]
    r_lo = np.asarray(rad.position) - 0.5 * np.asarray(rad.size)
    r_hi = np.asarray(rad.position) + 0.5 * np.asarray(rad.size)
    assert np.any(hi < r_lo) or np.any(lo > r_hi)                          # no overlap with the radiator
    assert d.moment_ref3 == pytest.approx(ROOM_CABINET_CENTRE) and not d.symmetric
    assert d.source["symmetry_error"] > 0.1                                # measured about y = 0, off-centre


def test_planner_count_and_the_closed_cavity_refusal():
    from aerolume.app.pages.setup import plan_failure_text, planner_count_text
    assert planner_count_text([("error", "a")]) == "1 error"
    assert planner_count_text([("warning", "a"), ("warning", "b"), ("info", "c")]) == "2 warnings"
    assert planner_count_text([("error", "a"), ("warning", "b")]) == "1 error, 1 warning"
    assert planner_count_text([("info", "a")]) == ""
    cavity = Design(name="Cavity", kind="generic", bodies=[], length_ref=0.05, force_ref=0.05)
    sim = Simulation(name="s", design_id=cavity.id)
    exc = ValueError("The design has no bodies to simulate.")
    assert plan_failure_text(cavity, sim, exc).startswith("This setup cannot be planned")    # a tunnel: a gap
    sim.cfd.domain_mode = "closed"
    text = plan_failure_text(cavity, sim, exc)
    assert text.startswith("The installed planner needs a body") and text.endswith("to simulate).")
    assert plan_failure_text(_design2d(), sim, exc).startswith("This setup cannot be planned")


def test_frame_caption_and_the_physics_layer_description():
    from aerolume.app.pages.setup import frame_text
    from aerolume.app.widgets.geometry_canvas import physics_description
    assert frame_text(False) == "Wind frame · flow from the left"
    assert frame_text(True, closed=True) == "Closed box · z up" and frame_text(False, True) == "Closed box · y up"
    fan = {"key": ("fans", 0), "type": "fan", "name": "Fan", "enabled": True, "selected": True, "dim": 3}
    dom = {"key": ("domain",), "type": "domain", "name": "Domain", "selected": False, "dim": 3}
    text = physics_description([fan, dom])
    assert text.startswith("1 placed item: Fan (fan).") and "Page Up / Page Down" in text
    fan2 = {**fan, "selected": False, "dim": 2}
    text = physics_description([fan2, {**dom, "selected": True}])
    assert "the domain" in text and "arrow keys" not in text               # the walls do not nudge


def test_new_simulation_copies_the_physics_of_its_template():
    from aerolume.app.pages.setup import copy_physics
    d = _design2d()
    a = Simulation(name="a", design_id=d.id)
    a.thermal.enabled, a.thermal.t_ambient = True, 30.0
    a.thermal.walls = [WallThermal(side="top", mode="temperature", value=10.0)]
    a.cfd.domain_mode, a.cfd.domain_box = "closed", (0.0, 1.0, 0.0, 1.0)
    a.fans = [Fan(position=(0.5, 0.5), direction=(1.0, 0.0))]
    same = Simulation(name="b", design_id=d.id)
    copy_physics(a, same, same_design=True)
    assert same.thermal.to_dict() == a.thermal.to_dict() and same.fans[0].to_dict() == a.fans[0].to_dict()
    assert same.fans[0] is not a.fans[0] and same.cfd.domain_box == a.cfd.domain_box
    other = Simulation(name="c", design_id="other")
    copy_physics(a, other, same_design=False)
    assert other.thermal.enabled and other.cfd.domain_mode == "closed"
    assert other.fans == [] and other.cfd.domain_box is None              # positions belong to that design


# ---------------------------------------------------------------------------
# the Setup page
# ---------------------------------------------------------------------------

def _setup_page(qapp, state, design, sim=None):
    from aerolume.app.pages.setup import SetupPage
    state.add_design(design)
    if sim is None:
        sim = Simulation(name="run", design_id=design.id, conditions=FlowConditions(fluid=fluids.air(20.0), speed=2.0))
    state.add_simulation(sim, select=True)
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    _process(qapp, 10)
    page.update_summary()
    _process(qapp)
    return page, sim


def test_heat_transfer_editor_updates_the_simulation_and_the_design(qapp, state, planner):
    d = _design2d()
    page, sim = _setup_page(qapp, state, d)
    hs = page.heat_section
    assert not hs.rows["ambient"].label.isVisibleTo(page)               # off: only the switch shows
    hs.enabled.setChecked(True)
    assert sim.thermal.enabled and hs.rows["ambient"].label.isVisibleTo(page)
    hs.ambient.set_si_value(35.0, emit=True)
    assert sim.thermal.t_ambient == 35.0
    page.update_summary()                                   # air at 20 C, ambient 35 C: said, with the fix
    assert any("set Flow ▸ Fluid to that temperature" in t for _lv, t in page.planner_lines)
    box, auto = hs.props["prandtl"]
    assert auto.isChecked() and box.value() == pytest.approx(0.71, rel=0.01)      # the default is shown
    box.setValue(0.9)
    assert sim.thermal.prandtl == 0.9 and not auto.isChecked()
    auto.click()
    assert sim.thermal.prandtl is None and box.value() == pytest.approx(0.71, rel=0.01)
    mode, temp, _flux = hs.walls["bottom"]
    mode.setCurrentIndex(mode.findData("temperature"))
    temp.set_si_value(60.0, emit=True)
    assert [(w.side, w.mode, w.value) for w in sim.thermal.walls] == [("bottom", "temperature", 60.0)]
    mode.setCurrentIndex(mode.findData("adiabatic"))
    assert sim.thermal.walls == []                                       # adiabatic sides are not listed
    left_row = hs.rows["wall_left"]
    assert not left_row.editor.isEnabled()                                 # the inlet of a tunnel
    hs.body_mode.setCurrentIndex(hs.body_mode.findData("temperature"))
    hs.body_temp.set_si_value(80.0, emit=True)
    assert d.bodies[0].thermal == "temperature" and d.bodies[0].thermal_value == 80.0
    page.flush()
    from aerolume.project import Project
    again = Project.open(state.project.path)
    assert again.designs[d.id].bodies[0].thermal == "temperature"
    assert again.simulations[sim.id].thermal.t_ambient == 35.0
    page.update_summary()
    assert page.estimate.value_text("Temperature scale") == "45.0 K"
    assert page.estimate.value_text("Rayleigh number") is not None
    page.close()


def test_placed_item_lists_add_edit_duplicate_remove(qapp, state, planner):
    d = _design2d()
    page, sim = _setup_page(qapp, state, d)
    fans = page.item_sections["fans"]
    assert fans.empty.isVisibleTo(page) and not fans.list.isVisibleTo(page)
    fans.add_button.click()
    assert len(sim.fans) == 1 and fans.list.count() == 1 and page._physics_sel == ("fans", 0)
    spec = next(s for s in page.preview.physics() if s["type"] == "fan")
    assert spec["selected"] and spec["name"] == "Fan"
    fans.mode.setCurrentIndex(fans.mode.findData("thrust"))
    assert sim.fans[0].mode == "thrust" and fans.rows["value"].unit.text() == "N/m"
    fans.value.setValue(3.5)
    assert sim.fans[0].value == 3.5 and "3.50 N/m" in fans.list.item(0).text()
    fans.direction.angle.set_si_value(90.0, emit=True)
    assert sim.fans[0].direction == pytest.approx((0.0, 1.0), abs=1e-12)
    fans.position.boxes[0].set_si_value(-0.4, emit=True)
    assert sim.fans[0].position[0] == pytest.approx(-0.4)
    fans.dup_button.click()
    assert len(sim.fans) == 2 and sim.fans[1].name == "Fan copy"
    fans.remove_button.click()
    assert len(sim.fans) == 1
    em = page.emitters_section
    em.add_button.click()
    e = sim.emitters[0]
    assert e.kind == "rake" and em.rows["end"].label.isVisibleTo(page)
    em.shape.setCurrentIndex(em.shape.findData("point"))
    assert e.kind == "point" and not em.rows["end"].label.isVisibleTo(page)
    em.pulse.setValue(0.4)
    assert e.pulse_period == 0.4 and em.rows["duty"].label.isVisibleTo(page)
    em.colour.setCurrentIndex(em.colour.findData("#4aa3df"))
    assert e.color == "#4aa3df"
    ports = page.item_sections["ports"]
    ports.add_button.click()
    ports.add_button.click()
    assert [p.kind for p in sim.ports] == ["jet", "jet"]                  # a tunnel: jets until told otherwise
    ports.port_kind.setCurrentIndex(ports.port_kind.findData("vent"))
    assert sim.ports[1].kind == "vent" and sim.ports[1].name == "Vent"
    page.flush()
    from aerolume.project import Project
    again = Project.open(state.project.path).simulations[sim.id]
    assert again.to_dict() == sim.to_dict()
    page.close()


def test_domain_and_diagnostics_editors(qapp, state, planner):
    d = _design2d()
    page, sim = _setup_page(qapp, state, d)
    ds = page.domain_section
    ds.mode.buttons["closed"].click()
    assert sim.cfd.domain_mode == "closed"
    assert page._speed_row.label.text() == "Reference speed"
    assert "closed box has no inflow" in page.emitters_section.empty.text()
    lo, hi = ds.ranges["x"]
    hi.set_si_value(0.8, emit=True)
    assert sim.cfd.domain_box is not None and sim.cfd.domain_box[1] == pytest.approx(0.8)
    assert not ds.auto.isChecked()
    lo.set_si_value(0.9, emit=True)                                       # past the other edge: refused
    assert sim.cfd.domain_box[1] > sim.cfd.domain_box[0] and lo.property("invalid") == "true"
    ds.auto.click()
    assert sim.cfd.domain_box is None
    dg = page.diagnostics_section
    dg.record.setChecked(False)
    assert sim.cfd.record_diagnostics is False
    dg.dt.setCurrentIndex(dg.dt.findData("pid"))
    assert sim.cfd.adaptive_dt == "pid"
    page.close()


def test_typing_one_edge_or_component_keeps_the_others_exact(qapp, state, planner):
    """The de Vahl Davis cavity is 45.7 mm for Ra = 10⁵: its edges show three significant figures,
    and typing one edge writes the untouched ones back unrounded (a 3-decimal box once turned the
    height into 46 mm, +2 % in Ra). The same holds for a placed item's position."""
    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    _d, (sim,) = add_template(state, "cavity")
    _process(qapp, 12)
    side = sim.cfd.domain_box[1]
    assert 0.04 < side < 0.05 and abs(side - round(side, 3)) > 1e-6        # not a round number of mm
    _lo, hi = page.domain_section.ranges["x"]
    _y0, y1 = page.domain_section.ranges["y"]
    assert y1.text() == f"{side:.4f}"                                       # three significant figures
    hi.lineEdit().selectAll()
    QTest.keyClicks(hi, "0.05")
    QTest.keyClick(hi, Qt.Key.Key_Return)
    _process(qapp)
    assert sim.cfd.domain_box[1] == pytest.approx(0.05) and sim.cfd.domain_box[3] == side   # exact, not 0.046
    from aerolume.app.widgets.physics_editor import VectorField
    v = VectorField(2)
    got = []
    v.valueChanged.connect(got.append)
    v.set_value((0.03051, 0.08072))
    v.boxes[0].lineEdit().selectAll()
    QTest.keyClicks(v.boxes[0], "0.04")
    QTest.keyClick(v.boxes[0], Qt.Key.Key_Return)
    assert got[-1] == (pytest.approx(0.04), 0.08072)
    page.close()


def _handle_px(canvas, key, handle="centre") -> QPoint:
    p = canvas.physics_handle_point(key, handle)
    q = canvas.to_screen(*p)
    return QPoint(round(q.x()), round(q.y()))


def test_dragging_an_item_on_the_2d_preview_moves_it_with_snapping(qapp, state, planner):
    d = _design2d()
    sim = Simulation(name="run", design_id=d.id, conditions=FlowConditions(fluid=fluids.air(20.0), speed=2.0))
    sim.heat_sources = [HeatSource(kind="rect", position=(-0.25, 0.0), size=(0.04, 0.04), power=50.0, frame="wind")]
    page, sim = _setup_page(qapp, state, d, sim)
    cv = page.preview
    key = ("heat_sources", 0)
    start = _handle_px(cv, key)
    assert cv.physics_hit(QPointF(start)) == (key, "centre")
    step = cv.snap_length()
    QTest.mousePress(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for k in range(1, 6):
        QTest.mouseMove(cv, start + QPoint(8 * k, -5 * k))
    QTest.mouseRelease(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(40, -25))
    _process(qapp)
    x, y = sim.heat_sources[0].position
    want = cv.to_world(QPointF(start + QPoint(40, -25)))
    assert x == pytest.approx(round(want[0] / step) * step, abs=1e-9)       # snapped to the grid step
    assert y == pytest.approx(round(want[1] / step) * step, abs=1e-9)
    assert page._physics_sel == key and page.item_sections["heat_sources"].current_index() == 0
    assert page.item_sections["heat_sources"].position.value() == pytest.approx((x, y))     # fields follow
    # arrow keys nudge the selected item by one step (Shift: ten)
    cv.setFocus()
    QTest.keyClick(cv, Qt.Key.Key_Right)
    assert sim.heat_sources[0].position[0] == pytest.approx(x + step)
    QTest.keyClick(cv, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier)
    assert sim.heat_sources[0].position[1] == pytest.approx(y + 10 * step)
    # Shift while dragging: free placement
    start = _handle_px(cv, key)
    QTest.mousePress(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(cv, start + QPoint(3, 0))
    end = start + QPoint(13, 0)
    from PySide6.QtGui import QMouseEvent
    ev = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(end), QPointF(cv.mapToGlobal(end)),
                     Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    QApplication.sendEvent(cv, ev)
    QTest.mouseRelease(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, end)
    free = sim.heat_sources[0].position[0]
    assert abs(free / step - round(free / step)) > 1e-6                     # not on the grid
    # a click on empty space clears the selection; the layer toggle hides and disables it
    QTest.mouseClick(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(cv.width() - 60, 60))
    assert page._physics_sel is None
    page.physics_toggle.click()
    assert not cv.show_physics and cv.physics_hit(QPointF(_handle_px(cv, key))) is None
    page.close()


def test_dragging_the_domain_wall_fixes_the_box(qapp, state, planner):
    d = _design2d()
    page, sim = _setup_page(qapp, state, d)
    cv = page.preview
    page.domain_toggle.setChecked(True)                                    # frame the whole domain
    _process(qapp)
    assert sim.cfd.domain_box is None
    start = _handle_px(cv, ("domain",), "x1")
    assert cv.physics_hit(QPointF(start)) == (("domain",), "x1")
    planned = page.physics_extent()
    QTest.mousePress(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(cv, start + QPoint(-10, 0))
    QTest.mouseMove(cv, start + QPoint(-30, 0))
    QTest.mouseRelease(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(-30, 0))
    _process(qapp)
    box = sim.cfd.domain_box
    assert box is not None and box[1] < planned[1] and box[0] == pytest.approx(planned[0])
    assert box[2:] == pytest.approx(planned[2:]) and not page.domain_section.auto.isChecked()
    assert page.domain_section.ranges["x"][1].si_value() == pytest.approx(box[1], abs=1e-6)
    page.close()


def test_clicking_an_item_in_the_preview_opens_its_editor(qapp, state, planner):
    d = _design2d()
    sim = Simulation(name="run", design_id=d.id)
    sim.ports = [Port(name="Jet", kind="jet", position=(-0.3, 0.05), size=0.03),
                 Port(name="Vent", kind="vent", position=(0.4, 0.0), size=0.03)]
    page, sim = _setup_page(qapp, state, d, sim)
    cv = page.preview
    QTest.mouseClick(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _handle_px(cv, ("ports", 1)))
    _process(qapp)
    ports = page.item_sections["ports"]
    assert page._physics_sel == ("ports", 1) and ports.current_index() == 1
    assert ports.name_edit.text() == "Vent" and page.item_groups["ports"].is_expanded()
    page.close()


def test_long_names_show_their_start(qapp, state, planner):
    """A name longer than its field shows how it begins, not its tail.

    QLineEdit.setText leaves the cursor at the end, so the critique saw
    "eated cylinder · d 10 mm · ΔT 10 K" in the Analysis name field. The
    simulation name and a placed item's name load with the cursor at 0 and go
    back there when focus leaves after an edit."""
    long_sim = "Heated cylinder · d 10 mm · ΔT 10 K · second refinement study"
    long_port = "Exhaust vent on the right-hand wall, upper position, second variant"
    d = _design2d()
    sim = Simulation(name=long_sim, design_id=d.id)
    sim.ports = [Port(name=long_port, kind="vent", position=(0.4, 0.0), size=0.03)]
    page, sim = _setup_page(qapp, state, d, sim)
    ne = page.name_edit
    assert ne.fontMetrics().horizontalAdvance(long_sim) > ne.width()     # really too long for the field
    assert ne.text() == long_sim and ne.cursorPosition() == 0
    ports = page.item_sections["ports"]
    ports.select(0)
    _process(qapp)
    assert ports.name_edit.text() == long_port and ports.name_edit.cursorPosition() == 0
    # type at the end, then leave the field: the rename sticks and the start comes back into view
    page.activateWindow()
    ne.setFocus(Qt.FocusReason.MouseFocusReason)
    _process(qapp)
    ne.end(False)
    QTest.keyClicks(ne, " B")
    assert ne.cursorPosition() == len(long_sim) + 2
    page.preview.setFocus(Qt.FocusReason.MouseFocusReason)
    _process(qapp)
    assert sim.name == long_sim + " B" and ne.cursorPosition() == 0
    page.close()


def test_import_dialog_names_show_their_start(qapp, tmp_path):
    """The mesh import dialog names the design after the file; a long file name shows its start."""
    import shutil
    from pathlib import Path

    from aerolume.app.dialogs.mesh_import import MeshImportDialog
    src = Path(__file__).resolve().parents[1] / "examples" / "geometry3d" / "sphere_d100mm_binary.stl"
    if not src.is_file():
        pytest.skip("example mesh not shipped")
    path = tmp_path / "wind_tunnel_model_sphere_100mm_final_revision_for_the_validation_campaign.stl"
    shutil.copyfile(src, path)
    dlg = MeshImportDialog(None, path=str(path))
    try:
        assert not dlg.load_error
        assert dlg.name.text().startswith("wind tunnel model sphere") and dlg.name.cursorPosition() == 0
    finally:
        dlg.done(0)


def test_heated_bodies_are_drawn_and_named_but_not_dragged(qapp, state, planner):
    """A heated body is part of the case: the preview outlines it in the heater colour with its
    condition beside it ("Cylinder · 30 °C", so colour is not the only cue) and the screen-reader
    description says it, but it is no placed item: no handles, no hit, not in physics()."""
    from aerolume.app.widgets import physics_editor as pe
    from aerolume.app.widgets.geometry_canvas import item_label
    d = _design2d()
    d.bodies[0].thermal, d.bodies[0].thermal_value = "temperature", 30.0
    sim = Simulation(name="run", design_id=d.id, conditions=FlowConditions(alpha_deg=10.0))
    assert not any(s["type"] == "body_thermal" for s in pe.scene_specs(d, sim))    # heat transfer off
    sim.thermal.enabled = True
    spec = next(s for s in pe.scene_specs(d, sim) if s["type"] == "body_thermal")
    assert spec["key"] == ("bodies", 0) and spec["sign"] == 1 and spec["handles"] == {}
    np.testing.assert_allclose(spec["points"], pe.to_wind(d.bodies[0].points, d, sim, "body"))  # pitched
    assert item_label(spec) == "Block · 30 °C"
    page, sim = _setup_page(qapp, state, d, sim)
    cv = page.preview
    assert all(s["type"] != "body_thermal" for s in cv.physics()) and len(cv._thermal_bodies) == 1
    assert "Block: held at 30 °C." in cv.accessibleDescription()
    top = cv.to_screen(*np.asarray(spec["points"])[np.argmax(np.asarray(spec["points"])[:, 1])])
    assert cv.physics_hit(top) is None or cv.physics_hit(top)[0] == ("domain",)
    cv.grab()                                            # paints the outline and the label without error
    page.close()


def test_closed_cavity_without_bodies_shows_its_box(qapp, state, planner):
    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1280, 800)
    page.show()
    _d, (sim,) = add_template(state, "cavity")
    _process(qapp, 10)
    page.update_summary()
    _process(qapp)
    assert page.estimate.value_text("Rayleigh number") == "1.0×10⁵"
    if page.plan_status == "error":            # a tunnel-only planner: said as its limit, not as a bad case
        assert any(lv == "error" and "closed box without one" in t for lv, t in page.planner_lines)
    dom = next(s for s in page.preview.physics() if s["type"] == "domain")
    assert dom["explicit"] and dom["closed"] and dom["box"] == pytest.approx(sim.cfd.domain_box)
    assert page.preview.design() is not None and not page.preview.design().bodies
    b, box = page.preview._content_bounds(), sim.cfd.domain_box                       # the box is framed
    assert b[0] <= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] >= box[3]
    assert (b[1] - b[0]) < 2 * (box[1] - box[0])
    img = page.preview.grab()
    assert not img.isNull()
    page.close()


def test_auto_rake_converts_to_an_emitter_on_the_3d_page(qapp, state, planner):
    from aerolume.app.pages.setup import flow_viz_of, stored_flow_viz
    d = _box3d()
    sim = Simulation(name="run", design_id=d.id, conditions=FlowConditions(speed=10.0))
    page, sim = _setup_page(qapp, state, d, sim)
    assert page.smoke_row.label.isVisibleTo(page) and page.last_smoke is not None   # the automatic rake
    layout = page.last_smoke
    made = page.convert_rake_to_emitter()
    assert len(made) == 1 and sim.emitters[0].count == layout["n"][0]
    assert not page.smoke_row.label.isVisibleTo(page)                             # the emitter rules now
    rake = stored_flow_viz(flow_viz_of(sim))["rake"]
    assert rake["kind"] == "rake" and rake["x"] == pytest.approx(layout["x"]) and rake["n"] == layout["n"][0]
    specs = page.preview3d.physics()
    assert any(s["type"] == "emitter" for s in specs) and page.preview3d.smoke() is None
    page.emitters_section.remove_button.click()
    assert sim.emitters == [] and page.smoke_row.label.isVisibleTo(page)
    page.close()


def test_3d_preview_draws_and_drags_placed_items(qapp, state, planner):
    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    _d, (sim,) = add_template(state, "room")
    _process(qapp, 10)
    page.update_summary()
    _process(qapp, 10)
    v = page.preview3d
    if not getattr(v.scene, "available", False):
        pytest.skip("no OpenGL context for the 3D view")
    v.grab()                                     # renders once, so projections use the real camera
    keys = {s["key"] for s in v.physics()}
    assert {("heat_sources", 0), ("fans", 0), ("ports", 0), ("ports", 1), ("emitters", 0), ("domain",)} <= keys
    fan = next(s for s in v.physics() if s["type"] == "fan")
    px = v._project(np.asarray([fan["centre"]]))[0]
    pos = QPointF(float(px[0]), float(px[1]))
    assert v.physics_hit3(pos) == (("fans", 0), "centre")
    before = np.array(sim.fans[0].position)
    QTest.mousePress(v, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos.toPoint())
    QTest.mouseMove(v, pos.toPoint() + QPoint(5, 0))
    QTest.mouseMove(v, pos.toPoint() + QPoint(40, 0))
    QTest.mouseRelease(v, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pos.toPoint() + QPoint(40, 0))
    _process(qapp)
    after = np.array(sim.fans[0].position)
    moved = after - before
    assert np.linalg.norm(moved) > 0.05 and page._physics_sel == ("fans", 0)
    # the move stays in the plane facing the camera's dominant axis: one coordinate is unchanged
    assert np.sum(np.abs(moved) < 1e-9) >= 1
    step = v.snap_length3()
    assert np.allclose(np.round(after / step) * step, after, atol=1e-9)
    # the keyboard alternative: arrows move one step along the world axis nearest screen-right / up,
    # Page Up toward the viewer, Shift ten steps; Escape clears the selection
    er, eu, et = v.nudge_axes3()
    assert sorted(int(np.argmax(np.abs(e))) for e in (er, eu, et)) == [0, 1, 2]
    v.setFocus()
    QTest.keyClick(v, Qt.Key.Key_Right)
    _process(qapp)
    np.testing.assert_allclose(np.array(sim.fans[0].position) - after, step * er, atol=1e-9)
    QTest.keyClick(v, Qt.Key.Key_PageUp, Qt.KeyboardModifier.ShiftModifier)
    _process(qapp)
    np.testing.assert_allclose(np.array(sim.fans[0].position) - after, step * (er + 10 * et), atol=1e-9)
    assert page.item_sections["fans"].position.value() == pytest.approx(tuple(sim.fans[0].position))
    QTest.keyClick(v, Qt.Key.Key_Escape)
    _process(qapp)
    assert page._physics_sel is None
    # a closed room: the caption says so, there is no free stream to pitch into, no upstream to place
    # smoke in, and the frame of a placed item is the box's, not a tunnel's
    assert page.frame_label.text() == "Closed box · z up" and not page.attitude_group.isVisibleTo(page)
    em = page.emitters_section
    em.select(0)
    assert not em.rows["place"].editor.isVisibleTo(page)
    assert em.frame.itemText(em.frame.findData("wind")) == "Fixed in the box"
    page.domain_section.mode.buttons["tunnel"].click()
    _process(qapp)
    assert page.frame_label.text().startswith("Wind frame") and page.attitude_group.isVisibleTo(page)
    fans = page.item_sections["fans"]
    fans.select(0)
    assert fans.frame.itemText(fans.frame.findData("wind")) == "Fixed in the tunnel"
    page.close()


def test_geometry_page_physics_layer_shows_and_moves_the_selected_simulation(qapp, state):
    from aerolume.app.pages.geometry import GeometryPage
    d = _design2d()
    state.add_design(d)
    sim = Simulation(name="run", design_id=d.id, conditions=FlowConditions(alpha_deg=10.0))
    sim.fans = [Fan(position=(-0.3, 0.0), direction=(1.0, 0.0), diameter=0.05, frame="body")]
    state.add_simulation(sim, select=True)
    page = GeometryPage(state, None)
    page.resize(1600, 1000)
    page.show()
    _process(qapp, 10)
    specs = page.canvas.physics()
    fan = next(s for s in specs if s["type"] == "fan")
    np.testing.assert_allclose(fan["centre"], (-0.3, 0.0), atol=1e-12)     # design frame: as stored
    assert all(s["type"] != "domain" for s in specs)                       # a tunnel's extent is Setup's
    start = _handle_px(page.canvas, ("fans", 0))
    QTest.mousePress(page.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(page.canvas, start + QPoint(0, -10))
    QTest.mouseMove(page.canvas, start + QPoint(0, -30))
    QTest.mouseRelease(page.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(0, -30))
    _process(qapp)
    assert sim.fans[0].position[1] > 0.0 and sim.fans[0].position[0] == pytest.approx(-0.3, abs=page.canvas.snap_length())
    from aerolume.project import Project
    assert Project.open(state.project.path).simulations[sim.id].fans[0].position == pytest.approx(sim.fans[0].position)
    page.physics_toggle.click()
    assert page.canvas.physics() == [] and not page.physics_toggle3d.isChecked()
    page.close()


def test_physics_editors_are_named_reachable_and_show_whole_numbers(qapp, state, planner):
    """The accessibility review's measured checks (WCAG 4.1.2, 2.1.1 / 2.4.3, and legibility): every
    visible control in the physics groups has an accessible name, the Tab chain reaches each of them
    top to bottom, and no number box clips its value (a "-1.500" domain edge once read "-1.5")."""
    from PySide6.QtWidgets import (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
        QListWidget,
        QWidget,
    )

    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1280, 800)
    page.show()
    add_template(state, "room")
    _process(qapp, 12)
    page.emitters_section.select(0)
    kinds = (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit, QListWidget)
    groups = [page.flowviz_group, page.heat_group, page.domain_group, page.diagnostics_group,
              *page.item_groups.values()]
    controls = [w for g in groups if g.isVisibleTo(page) for w in g.findChildren(QWidget)
                if isinstance(w, kinds) and w.objectName() != "qt_spinbox_lineedit" and w.isVisibleTo(page)
                and w.isEnabled() and w.focusPolicy() != Qt.FocusPolicy.NoFocus]
    assert len(controls) > 60
    unnamed = [w.objectName() for w in controls
               if not (w.accessibleName() or (isinstance(w, QAbstractButton) and w.text()))]
    assert unnamed == []
    want, seq, w = {id(c) for c in controls}, [], controls[0]
    for _ in range(5000):
        if id(w) in want:
            seq.append(w)
        w = w.nextInFocusChain()
        if w is controls[0]:
            break
    assert {id(c) for c in seq} == want
    ys = [c.mapTo(page, c.rect().center()).y() for c in seq]
    assert all(b >= a - 6 for a, b in itertools.pairwise(ys))
    assert _clipped(controls) == []
    # the cavity's 45.7 mm edges ("0.0457") and its wall temperatures, the narrowest values shown
    add_template(state, "cavity")
    _process(qapp, 12)
    boxes = [w for g in groups if g.isVisibleTo(page) for w in g.findChildren(QAbstractSpinBox)
             if w.isVisibleTo(page)]
    assert any(b.lineEdit().text() == "0.0457" for b in boxes) and _clipped(boxes) == []
    page.close()


def _clipped(controls) -> list:
    """Number boxes whose text does not fit: a QLineEdit draws inside its contents rect, less its
    text margins and 2 px on each side (the old ``advance + 4 > width`` let "0.0457" lose its 7)."""
    from PySide6.QtWidgets import QAbstractSpinBox
    out = []
    for c in controls:
        if not isinstance(c, QAbstractSpinBox):
            continue
        le = c.lineEdit()
        m = le.textMargins()
        room = le.contentsRect().width() - m.left() - m.right() - 4
        if le.fontMetrics().horizontalAdvance(le.text()) > room:
            out.append((c.accessibleName(), le.text()))
    return out


def test_inspectors_fit_their_column_for_every_starter_case(qapp, state, planner):
    """No starter case widens an inspector past its scroll area (the column has no horizontal
    scroll bar, so an over-wide grid hides its editors behind the window's edge: a one-line
    "Starter case · Natural-convection cavity (2D, validation)" did exactly that)."""
    from aerolume.app.dialogs.physics_templates import PHYSICS_TEMPLATES, add_template
    from aerolume.app.pages.geometry import GeometryPage
    from aerolume.app.pages.setup import SetupPage
    geo, setup = GeometryPage(state, None), SetupPage(state, None)
    for page in (geo, setup):
        page.resize(1280, 800)
        page.show()
    for t in PHYSICS_TEMPLATES:
        add_template(state, t.key)
        _process(qapp, 12)
        for page in (geo, setup):
            grid = page.grid if t.dimension == 2 or page is setup else page.mesh_inspector.grid
            view = grid.parentWidget()
            if grid.isVisibleTo(page) and view is not None:
                assert grid.width() <= view.width() + 1, (t.key, type(page).__name__, grid.width(), view.width())
    geo.close()
    setup.close()


def test_new_case_from_template_menu(qapp, state, planner):
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1280, 800)
    page.show()
    titles = [a.text() for a in page.template_menu.actions()]
    assert titles == ["Electronics enclosure (2D)", "Closed room (3D)", "Heated cylinder (2D)",
                      "Natural-convection cavity (2D, validation)"]
    page.handle_action("template:heated_cylinder")
    _process(qapp, 10)
    assert state.design.name == "Heated cylinder" and state.simulation.thermal.enabled
    assert page.heat_section.enabled.isChecked()
    assert page.heat_section.body_mode.currentData() == "temperature"
    page.close()


# ---------------------------------------------------------------------------
# critique round 3.1: a new run of a cavity, the Run gate and the fix button
# ---------------------------------------------------------------------------

def _api_cavity(state, ra=1e4):
    """The scripting API's heated cavity (no body) and its closed-box run, added through ``state``."""
    from aerolume import api
    from aerolume.app.pages.setup import _case_settings_sim
    case = api.cavity_case(ra)
    d = case.design
    state.add_design(d)
    sim = _case_settings_sim(d, case)
    sim.name = "Cavity run"
    state.add_simulation(sim, select=True)
    return d, sim


def test_a_bodiless_tunnel_is_blocked_with_the_case_as_its_fix(state):
    from aerolume.app.pages.setup import apply_run_fix, has_body, inherit_design_physics, run_blocker
    d, cav = _api_cavity(state)
    assert not has_body(d) and has_body(_design2d())
    assert run_blocker(d, cav, state.project) is None                       # the closed run itself is fine
    # the critic's path: AppState.new_simulation gives the model default, a tunnel without heat transfer
    new = state.new_simulation(d.id, "cfd")
    if new.cfd.domain_mode == "closed":
        pytest.skip("AppState.new_simulation inherits the case already")
    b = run_blocker(d, new, state.project)
    assert b is not None and b.action == "Switch to Closed box"
    assert b.text.startswith("This design is a closed box") and "“Cavity run”" in b.text.replace(chr(0xA0), " ")
    assert any(c.startswith("Heat transfer on") for c in b.changes)
    apply_run_fix(new, b, 2)
    assert new.cfd.domain_mode == "closed" and new.cfd.domain_box == pytest.approx(cav.cfd.domain_box)
    assert new.thermal.to_dict() == cav.thermal.to_dict()
    assert new.conditions.speed == pytest.approx(cav.conditions.speed)
    assert run_blocker(d, new, state.project) is None
    # a brand-new simulation object inherits the same setup up front
    fresh = Simulation(name="fresh", design_id=d.id)
    assert inherit_design_physics(state.project, d, fresh) == f"“{new.name}”"       # the latest closed run
    assert fresh.cfd.domain_mode == "closed" and fresh.thermal.enabled
    # with no run to copy and no recorded case, the fix is a box of the reference length
    bare = Design(name="Empty", kind="generic", bodies=[], length_ref=0.5, force_ref=0.5)
    s = Simulation(name="s", design_id=bare.id)
    b = run_blocker(bare, s, None)
    assert b is not None and "(0.500 × 0.500 m)" in b.text
    apply_run_fix(s, b, 2)
    assert s.cfd.domain_box == (0.0, 0.5, 0.0, 0.5)


def test_the_case_a_design_was_built_from_is_rebuilt():
    from aerolume import api
    from aerolume.app.dialogs.physics_templates import build_template
    from aerolume.app.pages.setup import case_source_template
    case = api.cavity_case(1e5, size=0.05)
    t = case_source_template(case.design)
    assert t is not None and not t.from_run
    assert t.sim.cfd.domain_mode == "closed" and t.sim.cfd.domain_box == pytest.approx((0.0, 0.05, 0.0, 0.05))
    hot, cold = (w.value for w in t.sim.thermal.walls)
    assert hot - cold == pytest.approx(case.design.source["delta_t"])
    u_ff = math.sqrt(t.sim.thermal.gravity * t.sim.thermal.beta * (hot - cold) * 0.05)
    assert t.sim.conditions.speed == pytest.approx(u_ff, rel=1e-3)            # the free-fall time unit
    room = api.fan_room_case(2.0)
    t = case_source_template(room.design)
    assert t.sim.fans and t.sim.fans[0].value == 2.0 and t.sim.cfd.domain_box == (0.0, 4.0, 0.0, 3.0)
    d, (sim,) = build_template("enclosure")
    t = case_source_template(d)
    assert t.sim.cfd.domain_box == sim.cfd.domain_box and len(t.sim.ports) == len(sim.ports)
    assert case_source_template(_design2d()) is None


def test_setup_page_refuses_the_run_and_the_button_fixes_it(qapp, state, planner):
    from aerolume.app.pages.setup import PlannerLine, SetupPage
    d, _cav = _api_cavity(state)
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    tunnel = Simulation(name="tunnel copy", design_id=d.id)            # what a plain default gives
    state.add_simulation(tunnel, select=True)
    _process(qapp, 10)
    page.update_summary()
    _process(qapp)
    assert page.run_blocker is not None and page.plan_status == "error"
    assert not page.run_button.isEnabled()
    page.run_simulation()                                               # never reaches a job manager
    buttons = [ln.action_button for ln in page.findChildren(PlannerLine) if ln.action_button is not None]
    assert [b.text() for b in buttons] == ["Switch to Closed box"]
    assert "Heat transfer on" in buttons[0].toolTip()
    buttons[0].click()
    for _ in range(20):
        _process(qapp)
    saved = state.project.simulations[tunnel.id]
    assert saved.cfd.domain_mode == "closed" and saved.thermal.enabled
    assert page.run_blocker is None and page.plan_status != "error"
    # Setup's own New simulation of the cavity starts as the closed box
    new = page.new_simulation()
    assert new.cfd.domain_mode == "closed" and new.thermal.enabled
    page.close()


def test_closed_box_is_framed_with_a_tenth_of_margin(qapp):
    from aerolume.app.widgets.geometry_canvas import CasePreview
    pv = CasePreview()
    pv.resize(980, 410)
    pv.show()
    room = Design(name="Room", kind="generic", bodies=[], length_ref=0.5, force_ref=0.5)
    pv.set_case(room, extent=(0.0, 4.0, 0.0, 3.0), inflow=False)
    _process(qapp)
    x0, x1, y0, y1 = pv._content_bounds()
    assert (x0, x1, y0, y1) == pytest.approx((-0.4, 4.4, -0.4, 3.4))
    s = pv.scale()
    # the 3 m room fills the height but for the margin, a tenth of the longer side: 3 of 3.8 m
    assert 3.0 * s == pytest.approx((pv.height() - sum(pv.CLOSED_FIT_MARGIN[1::2])) * 3.0 / 3.8, rel=1e-6)
    pv.set_case(room, extent=(0.0, 4.0, 0.0, 3.0), inflow=True)                # a tunnel keeps its context
    _process(qapp)
    assert pv._content_bounds() != pytest.approx((-0.4, 4.4, -0.4, 3.4))
    pv.close()


def test_device_labels_never_overlap_and_lead_back_when_moved():
    from aerolume.app.widgets.geometry_canvas import LEADER_PX, leader_end, place_labels
    # the 3D room's three labels wanted the same pixels: radiator name, wall value, emitter name
    reqs = [{"anchor": (240, 180), "size": (34, 15), "mode": "centre"},
            {"anchor": (232, 176), "size": (99, 15)},
            {"anchor": (250, 182), "size": (120, 15)}]
    lay = place_labels(reqs, (700, 330))
    boxes = [r["box"] for r in lay]
    for a, b in itertools.combinations(boxes, 2):
        assert (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]), boxes
    assert lay[0]["box"] == (223.0, 172.5, 34.0, 15.0) and not lay[0]["leader"]     # the first keeps its spot
    for r in lay:                                                   # a moved label leads back to its anchor
        x, y = leader_end(r["box"], r["anchor"])
        far = math.hypot(x - r["anchor"][0], y - r["anchor"][1]) > LEADER_PX
        assert far == r["leader"]
    # labels stay inside the view, and an obstacle (the corner notes) is avoided
    lay = place_labels([{"anchor": (690, 10), "size": (80, 15)}], (700, 330), taken=[(600, 0, 100, 20)])
    x, y, w, h = lay[0]["box"]
    assert 0 <= x and x + w <= 700 and 0 <= y and y + h <= 330
    assert y >= 20 + 2 or x + w <= 600


# ---------------------------------------------------------------------------
# round 4: heat colour, screen-reader text, keyboard focus (design:accessibility-review)
# ---------------------------------------------------------------------------

def _contrast(a: str, b: str) -> float:
    def lum(h):
        r, g, b_ = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        f = (lambda v: v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b_)
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_heat_marks_are_the_highlight_token_and_heat_text_stays_readable():
    from aerolume.app import theme
    from aerolume.app.widgets.geometry_canvas import heat_mark, heat_text
    for name in ("light", "dark"):
        tok = theme.PALETTES[name]
        field = tok["field_bg"]
        # marks: the one "heat" token (docs/ui_guidelines.md 2), cold in the first series colour
        assert heat_mark(tok, 1).name() == tok["highlight"]
        assert heat_mark(tok, -1).name() == tok["series"][0] and heat_mark(tok, 0).name() == tok["fg2"]
        assert _contrast(heat_mark(tok, 1).name(), field) >= 3.0                  # WCAG 1.4.11
        # text never in the highlight amber: warm values in warn, cold in the series blue, >= 4.5:1
        assert heat_text(tok, 1).name() == tok["warn"] != tok["highlight"]
        for sign in (1, -1, 0):
            assert _contrast(heat_text(tok, sign).name(), field) >= 4.5, (name, sign)


def test_walls_alone_are_described_to_screen_readers():
    from aerolume.app.widgets.geometry_canvas import physics_description
    dom = {"key": ("domain",), "type": "domain", "name": "Domain", "dim": 2, "selected": False, "closed": True,
           "explicit": True, "box": (0.0, 0.0457, 0.0, 0.0457),
           "walls": [{"side": "left", "mode": "temperature", "value": 25.0, "sign": 1},
                     {"side": "right", "mode": "heat_flux", "value": -40.0, "sign": -1}]}
    text = physics_description([dom])
    assert text == ("Closed box 0.0457 × 0.0457 m; left (x min) wall held at 25 °C, right (x max) wall heat flux "
                    "-40 W/m² into the fluid.")
    assert physics_description([{**dom, "selected": True}]).endswith("edit its edges in Setup ▸ Domain. Esc clears.")
    # a planned tunnel says nothing about its extent (Setup's Estimate does)
    assert physics_description([{**dom, "closed": False, "explicit": False, "walls": []}]) == ""


def test_the_fluid_list_shows_degrees_and_keeps_its_keys(qapp, state):
    from aerolume.app.pages.setup import SetupPage, fluid_display_name
    assert fluid_display_name("Air, 20 C") == "Air, 20 °C" and fluid_display_name("Water, -2.5 C") == "Water, -2.5 °C"
    assert fluid_display_name("Air, ISA 1,000 m") == "Air, ISA 1,000 m"
    page = SetupPage(state, None)
    texts = [page.fluid_combo.itemText(i) for i in range(page.fluid_combo.count())]
    assert "Air, 20 °C" in texts and not any(t.endswith(" C") for t in texts)
    assert page.fluid_combo.findData("preset:Air, 20 C") >= 0                 # the stored name is the key
    page.close()


def test_fluid_names_read_either_degree_spelling():
    """The core writes "Air, 25 C"; the screen shows "Air, 25 °C". Both spellings must parse back
    to the same fluid row, so a saved project (or a core that adopts the symbol) keeps working."""
    from aerolume.app.pages.setup import fluid_display_name, fluid_mode, mach_number, preset_key
    from aerolume.app.widgets.physics_editor import fluid_defaults, fluid_temperature
    from aerolume.model import FlowConditions, Fluid

    assert fluid_display_name("Air, 25 C") == "Air, 25 °C"
    assert fluid_display_name("Air, 25 °C") == "Air, 25 °C"                 # idempotent
    assert fluid_display_name("Custom fluid") == "Custom fluid"
    assert preset_key("Air, 20 C") == preset_key("Air, 20 °C") == "Air, 20 C"
    assert preset_key("Air, 21 °C") is None
    for spelled in ("Air, 25 C", "Air, 25 °C"):
        f = fluids.air(25.0)
        f = Fluid(name=spelled, density=f.density, viscosity=f.viscosity)
        assert fluid_mode(f) == ("air_t", 25.0)
        assert fluid_temperature(f) == 25.0
        assert math.isclose(mach_number(FlowConditions(fluid=f, speed=100.0)),
                            100.0 / fluids.speed_of_sound(298.15))
    ref = fluids.PRESETS["Air, 20 C"]()
    assert fluid_mode(Fluid(name="Air, 20 °C", density=ref.density, viscosity=ref.viscosity)) == ("preset", None)
    assert fluid_mode(Fluid(name="Water, 30 °C", density=995.6, viscosity=7.97e-4)) == ("water_t", 30.0)
    # the thermal-defaults note is on screen too: no bare "C" after a number
    _, note = fluid_defaults(fluids.air(25.0), 25.0)
    assert "°C" in note and not re.search(r"\d C\b", note)


def test_a_fluid_saved_with_the_degree_sign_selects_its_preset(qapp, state):
    from aerolume.app.pages.setup import SetupPage
    from aerolume.model import Fluid
    page = SetupPage(state, None)
    _d, sim = _api_cavity(state)
    ref = fluids.PRESETS["Air, 20 C"]()
    sim.conditions.fluid = Fluid(name="Air, 20 °C", density=ref.density, viscosity=ref.viscosity)
    page._load_form(sim, state.design)
    assert page.simulation is sim
    assert page.fluid_combo.currentText() == "Air, 20 °C"
    assert page.fluid_combo.currentData() == "preset:Air, 20 C"
    page.close()


def test_every_part_of_a_composite_row_carries_its_explanation(qapp, state, planner):
    from PySide6.QtGui import QAccessible

    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    add_template(state, "cavity")
    _process(qapp, 12)
    sec = page.heat_section
    mode, _temp, _flux = sec.walls["left"]
    # the condition selector sits beside its value in one row: it used to be announced bare
    assert "Thermal condition on this side of the domain." in mode.accessibleDescription()
    box, auto = sec.props["prandtl"]
    assert box.accessibleDescription().startswith("Prandtl number ν/α")
    iface = QAccessible.queryAccessibleInterface(mode)
    assert "Thermal condition" in iface.text(QAccessible.Text.Description)
    assert auto.toolTip() and "Prandtl" not in auto.accessibleDescription()  # its own tooltip is kept
    # the copy read aloud (design:ux-copy): articles and plurals of the item nouns
    assert page.emitters_section.add_button.toolTip().startswith("Add an emitter ")
    assert page.item_sections["ports"].list.accessibleDescription().startswith("The jets and vents of this")
    assert "tunnel's (or box's) axes" in page.item_sections["fans"].position.toolTip() or \
        "tunnel's (or box's) axes" in page.item_sections["fans"].rows["position"].label.toolTip()
    page.close()


def test_the_previews_show_a_focus_ring_for_the_keyboard_only(qapp):
    from aerolume.app.widgets.geometry_canvas import CasePreview, keyboard_focus
    assert keyboard_focus(Qt.FocusReason.TabFocusReason) and keyboard_focus(Qt.FocusReason.BacktabFocusReason)
    assert not keyboard_focus(Qt.FocusReason.MouseFocusReason)
    from PySide6.QtWidgets import QLineEdit, QVBoxLayout, QWidget

    from aerolume.app import theme
    tok = theme.tokens()
    win = QWidget()
    lay = QVBoxLayout(win)
    field, cv = QLineEdit(), CasePreview()
    lay.addWidget(field)
    lay.addWidget(cv)
    win.resize(400, 340)
    win.show()
    assert QTest.qWaitForWindowActive(win, 2000)
    field.setFocus()
    QTest.keyClick(field, Qt.Key.Key_Tab)                      # Tab from the inspector onto the preview
    _process(qapp)
    assert cv.hasFocus() and cv.focus_visible
    img = cv.grab().toImage()
    edge = img.pixelColor(int(img.width() * 0.5), 1).name()   # the ring's 2 px inside the top edge
    assert edge == tok["focus"]
    QTest.mousePress(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 150))
    QTest.mouseRelease(cv, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 150))
    _process(qapp)
    assert not cv.focus_visible
    assert cv.hasFocus() and cv.grab().toImage().pixelColor(200, 1).name() != tok["focus"]
    # the 3D preview (VTK image under a painted overlay) does the same
    from aerolume.app.widgets.geometry_canvas import CasePreview3D
    v3 = CasePreview3D()
    lay.addWidget(v3)
    _process(qapp)
    field.setFocus()
    QTest.keyClick(field, Qt.Key.Key_Tab)
    QTest.keyClick(cv, Qt.Key.Key_Tab)
    _process(qapp, 10)
    assert v3.hasFocus() and v3.focus_visible and not cv.focus_visible
    assert v3.grab().toImage().pixelColor(v3.width() // 2, 1).name() == tok["focus"]
    win.close()


def test_a_closed_box_names_its_axes_without_a_stream(qapp, state, planner):
    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    page.resize(1280, 800)
    page.show()
    add_template(state, "room")                         # a closed 3D box: no upstream or downstream
    _process(qapp, 12)
    ports = page.item_sections["ports"]
    ports.select(0)
    _process(qapp)
    texts = [ports.direction.preset.itemText(i) for i in range(ports.direction.preset.count())]
    assert texts[:2] == ["+x", "−x"] and "+z (up)" in texts and not any("stream" in t for t in texts)
    # the same editor in a tunnel says where the stream goes
    state.simulation.cfd.domain_mode = "tunnel"
    ports.reload()
    ports.select(0)
    texts = [ports.direction.preset.itemText(i) for i in range(ports.direction.preset.count())]
    assert texts[0] == "+x (downstream)" and texts[1] == "−x (upstream)"
    page.close()


def test_a_fact_the_planner_states_is_not_repeated_by_the_page(qapp, state, planner):
    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.setup import SetupPage, said_by_planner
    planner_says = ["Ra = 3.91×10¹⁰: above about 10⁸ the thermal boundary layers become unsteady ..."]
    assert said_by_planner("Ra = 3.9×10¹⁰: above about 10⁸ ... treat the result as indicative.", planner_says)
    assert not said_by_planner("Ra = 3.9×10¹⁰ ...", [])                     # an older planner: keep ours
    assert not said_by_planner("Fan swirl acts in 3D only; ...", planner_says)  # another topic
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    add_template(state, "room")                         # Ra ~ 4e10: both used to say so
    _process(qapp, 12)
    page.update_summary()
    _process(qapp)
    ra = [text for _lv, text in page.planner_lines if text.startswith("Ra = ")]
    assert len(ra) == 1, ra
    page.close()
