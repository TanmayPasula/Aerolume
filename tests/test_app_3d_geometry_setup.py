"""Setup in 3D: the page's 3D CFD and finite-wing set-ups, and the helpers behind them.

Real 3D designs (the ``examples/geometry3d`` samples, a lofted wing) are planned
by the real planner (``aerolume.simulation.plan`` -> ``plan3d``); the tests
assert on the project's :class:`~aerolume.model.Simulation` objects, which is
what the runner reads.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from aerolume.model import FlowConditions, Simulation, WingSection, WingSpec

pytestmark = pytest.mark.gui

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"


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
    st.set_project(Project.create(tmp_path / "Setup3D", "Setup 3D"))
    yield st


@pytest.fixture
def planner():
    from aerolume.app.pages import setup as setup_mod
    loader = setup_mod._PlannerLoader.get()
    if not loader.wait(180):
        pytest.skip(f"planner not importable: {loader.error}")
    return loader


def _vehicle():
    import aerolume.geometry3d as g
    return g.design_from_meshes(g.read_mesh(EXAMPLES / "body_and_wheel.obj"), "Sample car", "vehicle")


def _sphere():
    import aerolume.geometry3d as g
    return g.design_from_meshes(g.read_mesh(EXAMPLES / "sphere_d100mm_binary.stl"), "Sphere", "generic",
                                units="mm")


def _wing(swept: bool = False):
    import aerolume.geometry3d as g
    tip_x = 0.6 if swept else 0.03
    return g.design_from_wing(WingSpec(sections=[WingSection(0.0, 0.3, airfoil="2412"),
                                                 WingSection(1.0, 0.18, x_le=tip_x, airfoil="2412")]), "Wing")


def _process(qapp, n=5):
    for _ in range(n):
        qapp.processEvents()


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_default_simulation_3d_and_comparable_template(state):
    from aerolume.app.pages.setup import comparable_template, default_simulation
    from aerolume.geometry.templates import make
    car3, car2, wing = _vehicle(), make("sedan"), _wing()
    for d in (car3, car2, wing):
        state.project.add_design(d)
    sim = default_simulation(car3, "run 1")
    assert sim.analysis == "cfd" and sim.cfd.quality == "draft" and sim.cfd.symmetry == "auto"
    assert sim.ground.enabled and sim.ground.moving and sim.ground.rotate_wheels   # the sample has a wheel
    ws = default_simulation(wing, "w")
    assert ws.analysis == "wing" and ws.wing.method == "vlm" and not ws.ground.enabled
    # a 2D car run is not a template for a 3D car, and the other way round
    s2 = Simulation(name="2D", design_id=car2.id, conditions=FlowConditions(speed=40.0))
    state.project.add_simulation(s2)
    assert comparable_template(state.project, car3) is None
    s3 = default_simulation(car3, "3D")
    s3.conditions.speed = 22.0
    s3.conditions.yaw_deg = 5.0
    s3.cfd.symmetry = "full"
    s3.cfd.domain_width = 6.0
    state.project.add_simulation(s3)
    assert comparable_template(state.project, car3) is s3
    assert comparable_template(state.project, car2) is s2
    nxt = default_simulation(car3, "next", comparable_template(state.project, car3))
    assert nxt.conditions.speed == 22.0 and nxt.conditions.yaw_deg == 5.0
    assert nxt.cfd.symmetry == "full" and nxt.cfd.domain_width == 6.0
    w1 = default_simulation(wing, "w1")
    w1.wing.method, w1.wing.alpha_end = "llt", 8.0
    state.project.add_simulation(w1)
    w2 = default_simulation(wing, "w2", comparable_template(state.project, wing))
    assert w2.analysis == "wing" and w2.wing.method == "llt" and w2.wing.alpha_end == 8.0


def test_analysis_offers_ground_wheels_and_symmetry_reason():
    from aerolume.app.dialogs.import_dialogs import naca_design
    from aerolume.app.pages.setup import (
        analyses_for,
        has_wheels,
        polar_allowed,
        shows_ground,
        symmetry_reason,
    )
    car, sph, wing = _vehicle(), _sphere(), _wing()
    assert analyses_for(naca_design("2412")) == ["cfd", "polar"]
    assert analyses_for(car) == ["cfd"] and analyses_for(wing) == ["wing", "cfd"]
    assert not polar_allowed(wing) and not polar_allowed(car)
    assert has_wheels(car) and not has_wheels(sph)
    assert shows_ground(car) and not shows_ground(sph) and not shows_ground(wing)
    sph.source["on_road"] = True
    assert shows_ground(sph)
    sim = Simulation(name="s", design_id=car.id)
    sim.cfd.symmetry = "auto"
    half, why = symmetry_reason(car, sim)
    assert half == bool(car.symmetric) and why
    sim.conditions.yaw_deg = 4.0
    half, why = symmetry_reason(car, sim)
    assert not half and "yaw" in why
    sim.cfd.symmetry = "full"
    assert symmetry_reason(car, sim) == (False, "Off: both halves are simulated.")


def test_compact_rows_shorten_3d_summary_values():
    from aerolume.app.pages.setup import compact_rows
    rows = {k: (v, t) for k, v, t in compact_rows([
        ("Cells (3D)", "1.83 M solved (half domain; 3.66 M for both halves)"),
        ("Blockage", "4.9 % (height 28 %, width 20 %)"),
        ("Symmetry plane", "y = 0 (half domain simulated)"),
        ("Grid", "260 × 80 × 88")])}
    assert rows["Cells (3D)"][0] == "1.83 M (half domain)" and "3.66 M" in rows["Cells (3D)"][1]
    assert rows["Blockage"] == ("4.9 %", "Height 28 %, width 20 %")
    assert rows["Symmetry plane"][0] == "y = 0"
    assert rows["Grid"] == ("260 × 80 × 88", "")


def test_wing_estimate_counts_section_polars():
    from aerolume.app.pages.setup import wing_estimate
    d = _wing()
    sim = Simulation(name="w", design_id=d.id, analysis="wing")
    rows, _notes, warnings = wing_estimate(d, sim)
    r = dict(rows)
    assert r["Section polars"].startswith("1 airfoil × ") and r["Section polars"].endswith(" Re")
    assert int(r["Polar points, first run"].lstrip("≈ ").replace(",", "")) > 10
    assert r["Method"] == "Vortex lattice" and r["Panels per half"] == "8 × 24"
    assert "s" in r["Estimated run time"] and not warnings
    assert any("Ncrit 9" in n for n in _notes)
    # every value reads in full in the narrow Estimate column at 1280 px (~170 px, ~28 characters)
    assert max(len(v) for v in r.values()) <= 28, r
    sim.wing.viscous = False
    r = dict(wing_estimate(d, sim)[0])
    assert r["Profile drag"].startswith("None") and "Section polars" not in r
    sim.wing.method = "llt"
    _, _, warns = wing_estimate(_wing(swept=True), sim)
    assert any("lifting line" in w for w in warns)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def _page(state, design):
    from aerolume.app.pages.setup import SetupPage
    state.add_design(design, select=True)
    page = SetupPage(state, None)
    page.resize(1600, 1000)
    page.show()
    page.on_shown()
    return page


def test_setup_3d_vehicle_estimate_preview_and_edits(qapp, state, planner):
    page = _page(state, _vehicle())
    sim = state.simulation
    assert sim is not None and sim.design_id == state.design.id and sim.analysis == "cfd"
    page.update_summary()
    assert page.plan_status == "ok", page.planner_lines
    rows = dict(page.estimate.items())
    for key in ("Cells (3D)", "Symmetry plane", "Estimated run time", "Grid", "Attitude"):
        assert key in rows, key
    assert "Cells" not in rows                                           # "Cells (3D)" replaces it
    assert page._preview_stack.currentIndex() == 1
    assert page.preview3d.extent3() is not None and page.preview3d.meshes()
    assert page.preview3d.symmetry() == bool(page.last_plan.symmetry_plane)
    assert "M cells" in page.quality.itemText(0)                         # the presets' 3D cell counts
    assert page.analysis_fixed.isVisibleTo(page) and not page.analysis.isVisibleTo(page)
    assert page._yaw_row.label.isVisibleTo(page) and page._dom_rows["domain_width"].label.isVisibleTo(page)
    assert page.ground_group.isVisibleTo(page) and page.ground_wheels.isVisibleTo(page)
    assert page.symmetry_note.text()
    # yaw: the run needs the whole tunnel, and says so
    page.yaw.set_si_value(6.0, emit=True)
    page.flush()
    assert state.simulation.conditions.yaw_deg == 6.0
    page.update_summary()
    assert not page.last_plan.symmetry_plane and "yaw" in page.symmetry_note.text()
    assert not page.preview3d.symmetry()
    # domain width and symmetry setting go to the model (custom quality for the width)
    page.dom_width.setValue(5.0)
    page.symmetry.setCurrentIndex(page.symmetry.findData("full"))
    page.flush()
    cfd = state.simulation.cfd
    assert cfd.domain_width == 5.0 and cfd.quality == "custom" and cfd.symmetry == "full"
    # the sensitivity map states its cost (or why it cannot run)
    page.sensitivity.setChecked(True)
    page.update_summary()
    assert page.sensitivity_note.text()
    page.domain_toggle.setChecked(True)
    assert page.preview3d.fit_domain
    for k in ("iso", "front", "side", "top"):
        page.view_buttons[k].click()
    assert not page.grab().isNull()


def test_setup_wing_analysis_and_3d_cfd_of_the_wing(qapp, state, planner):
    page = _page(state, _wing())
    sim = state.simulation
    assert sim.analysis == "wing"
    assert page.analysis.visible_keys() == ["wing", "cfd"]
    assert page.analysis.buttons["cfd"].text() == "3D CFD"
    assert page.wing_group.isVisibleTo(page) and not page.mesh_group.isVisibleTo(page)
    assert not page._alpha_row.label.isVisibleTo(page) and page._yaw_row.label.text() == "Sideslip angle"
    page.update_summary()
    assert page.plan_status == "wing"
    rows = dict(page.estimate.items())
    assert "Section polars" in rows and "Estimated run time" in rows and "Reynolds number (MAC)" in rows
    assert page.preview3d.meshes()                                       # the lofted wing
    assert not page.live_button.isEnabled()
    page.w_alpha_end.set_si_value(8.0, emit=True)
    page.w_method.setCurrentIndex(page.w_method.findData("llt"))
    page.w_viscous.setChecked(False)
    page.flush()
    w = state.simulation.wing
    assert w.alpha_end == 8.0 and w.method == "llt" and w.viscous is False
    assert not page.w_ncrit.isEnabled()
    # the lofted wing in 3D CFD
    page.analysis.buttons["cfd"].click()
    page.flush()
    assert state.simulation.analysis == "cfd"
    page.update_summary()
    assert page.plan_status == "ok", page.planner_lines
    assert "Cells (3D)" in dict(page.estimate.items())
    assert page.mesh_group.isVisibleTo(page) and not page.wing_group.isVisibleTo(page)
    # polar is never offered for a wing
    page._set_analysis("polar")
    assert state.simulation.analysis == "cfd"


def test_setup_3d_generic_body_no_ground_and_2d_unchanged(qapp, state, planner):
    from aerolume.app.dialogs.import_dialogs import naca_design
    page = _page(state, _sphere())
    page.update_summary()
    assert page.plan_status == "ok"
    assert not page.ground_group.isVisibleTo(page)
    assert page._alpha_row.label.text() == "Pitch angle"
    # a 2D airfoil next to it keeps the 2D page
    state.add_design(naca_design("0012"), select=True)
    page.on_shown()
    page.update_summary()
    assert page._preview_stack.currentIndex() == 0
    assert page.analysis.visible_keys() == ["cfd", "polar"] and page.analysis.isVisibleTo(page)
    assert not page._yaw_row.label.isVisibleTo(page)
    assert not page._dom_rows["domain_width"].label.isVisibleTo(page)
    assert all("M cells" not in page.quality.itemText(i) for i in range(page.quality.count()))
    assert "Cells" in dict(page.estimate.items())
