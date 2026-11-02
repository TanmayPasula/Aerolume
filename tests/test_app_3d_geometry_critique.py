"""Critique fixes on the 3D Geometry and Setup pages (round 1).

* the Estimate reads the cost of the run first (grid, cells, run time, device),
  then the Reynolds numbers;
* the Flow visualisation group sits right under Flow, and a click on the smoke
  emitter in the 3D case preview brings it into view with a field focused;
* a new wing is not named after its kind ("Wing | Wing" in the step bar);
* "clearance" means one thing: the import dialog's placement field is the
  model's lowest point, and the body's ground clearance is shown with the
  Geometry page's definition.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

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
    st.set_project(Project.create(tmp_path / "Critique3D", "Critique 3D"))
    yield st


@pytest.fixture
def planner():
    from aerolume.app.pages import setup as setup_mod
    loader = setup_mod._PlannerLoader.get()
    if not loader.wait(180):
        pytest.skip(f"planner not importable: {loader.error}")
    return loader


def _process(qapp, n: int = 5) -> None:
    for _ in range(n):
        qapp.processEvents()


def _vehicle():
    import aerolume.geometry3d as g
    return g.design_from_meshes(g.read_mesh(EXAMPLES / "body_and_wheel.obj"), "Sample car", "vehicle")


def _setup(state, design, size=(1600, 1000)):
    from aerolume.app.pages.setup import SetupPage
    state.add_design(design, select=True)
    page = SetupPage(state, None)
    page.resize(*size)
    page.show()
    page.on_shown()
    return page


# ---------------------------------------------------------------------------
# the Estimate's reading order
# ---------------------------------------------------------------------------

def test_order_estimate_puts_the_cost_first_and_keeps_the_rest():
    from aerolume.app.pages.setup import order_estimate
    rows = [("Real Reynolds number", "a"), ("Simulated Reynolds number", "b"), ("Dynamic pressure", "q"),
            ("Mach number", "m"), ("Flow model", "f"), ("Grid", "g"), ("Cells (3D)", "c"), ("Domain", "d"),
            ("Estimated run time", "t"), ("Device", "v"), ("Smoke", "s")]
    assert [k for k, _ in order_estimate(rows)] == [
        "Grid", "Cells (3D)", "Estimated run time", "Device", "Real Reynolds number",
        "Simulated Reynolds number", "Dynamic pressure", "Mach number", "Flow model", "Domain", "Smoke"]
    # a partial table (planner still loading) keeps the same order for what it has
    assert order_estimate([("Dynamic pressure", "q"), ("Real Reynolds number", "a")]) == \
        [("Real Reynolds number", "a"), ("Dynamic pressure", "q")]
    assert order_estimate([]) == []


def test_setup_estimate_leads_with_grid_cells_time_and_device(qapp, state, planner):
    page = _setup(state, _vehicle())
    try:
        page.update_summary()
        assert page.plan_status == "ok", page.planner_lines
        keys = [row[0] for row in page.estimate.items()]
        assert keys[:4] == ["Grid", "Cells (3D)", "Estimated run time", "Device"]
        assert keys[4:6] == ["Real Reynolds number", "Simulated Reynolds number"]
        assert "Smoke" in keys and "Dynamic pressure" in keys
    finally:
        page.release_3d()


# ---------------------------------------------------------------------------
# Flow visualisation: right under Flow, reached by a click on the emitter
# ---------------------------------------------------------------------------

def test_flow_visualisation_group_follows_flow(qapp, state):
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    try:
        order = list(page.grid.groups)
        assert order.index("flowviz") == order.index("flow") + 1
        lay = page.grid.layout_
        idx = [lay.indexOf(page.grid.group(k)) for k in order]
        assert idx == sorted(idx), "the dict order is the on-screen order"
    finally:
        page.release_3d()


def _nozzle_px(qapp, page) -> QPoint:
    """A nozzle of the painted rake, in the preview's logical px."""
    view = page.preview3d
    view.grab()                                    # paints the overlays and records the emitter
    _process(qapp)
    hp = view._smoke_px
    assert hp is not None and hp["points"] is not None and len(hp["points"])
    x, y = hp["points"][len(hp["points"]) // 2]
    return QPoint(round(x), round(y))


def test_click_on_the_smoke_rake_brings_its_settings_into_view(qapp, state, planner):
    page = _setup(state, _vehicle(), size=(1280, 800))
    try:
        page.update_summary()
        _process(qapp)
        view = page.preview3d
        assert view.smoke() is not None and page.flowviz_group.isVisibleTo(page)
        page.flowviz_group.set_expanded(False)
        bar = page.inspector_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())                # scrolled away, the group collapsed
        at = _nozzle_px(qapp, page)
        assert view.smoke_hit(QPointF(at))
        assert not view.smoke_hit(QPointF(2, 2))   # the corner of the view is not the rake
        clicks = []
        view.smokeClicked.connect(lambda: clicks.append(1))
        # a drag that starts on the rake orbits the camera and does not count as a click
        QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
        QTest.mouseMove(view, at + QPoint(40, 0))
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at + QPoint(40, 0))
        _process(qapp)
        assert clicks == []
        at = _nozzle_px(qapp, page)                # the camera moved: find the rake again
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
        _process(qapp)
        assert clicks == [1]
        g = page.flowviz_group
        assert g.is_expanded()
        port = page.inspector_scroll.viewport()
        top = g.mapTo(port, QPoint(0, 0)).y()
        # the group's header at the top of the inspector, or as near as the scroll range allows
        assert 0 <= top <= 8 or bar.value() == bar.maximum(), (top, bar.value(), bar.maximum())
        assert top >= 0 and top + min(g.height(), port.height()) <= port.height(), (top, g.height())
        assert page.smoke_kind.hasFocus() or QApplication.focusWidget() in (page.smoke_kind, None)
    finally:
        page.release_3d()


def test_focus_flow_viz_falls_back_to_the_smoke_switch_when_off(qapp, state, planner, monkeypatch):
    page = _setup(state, _vehicle(), size=(1280, 800))
    try:
        page.update_summary()
        page.smoke.setChecked(False)
        _process(qapp)
        focused = []
        monkeypatch.setattr(page.smoke, "setFocus", lambda *a: focused.append("smoke"))
        monkeypatch.setattr(page.smoke_kind, "setFocus", lambda *a: focused.append("kind"))
        page.focus_flow_viz()
        assert focused == ["smoke"]
        page.smoke.setChecked(True)
        _process(qapp)
        page.focus_flow_viz()
        assert focused == ["smoke", "kind"]
    finally:
        page.release_3d()


def test_no_smoke_nothing_to_click(qapp):
    from aerolume.app.widgets.geometry_canvas import CasePreview3D
    car = _vehicle()
    view = CasePreview3D()
    view.resize(700, 400)
    view.show()
    try:
        view.set_case(car, meshes=car.meshes, ground="moving", smoke=None)
        view.grab()
        assert view._smoke_px is None and not view.smoke_hit(QPointF(350, 200))
    finally:
        view.scene.close()


# ---------------------------------------------------------------------------
# names: a new wing, and what "clearance" means
# ---------------------------------------------------------------------------

def test_new_wing_is_named_tapered_wing_and_unique(qapp, state):
    from aerolume.app.pages.geometry import NEW_WING_NAME, GeometryPage
    page = GeometryPage(state, None)
    try:
        first = page.new_wing()
        second = page.new_wing()
        assert NEW_WING_NAME == "Tapered wing"
        assert first.name == "Tapered wing" and second.name == "Tapered wing 2"
        # the same rule as the main window's fallback (AppState.unique_design_name)
        assert state.unique_design_name("Tapered wing") == "Tapered wing 3"
    finally:
        page.release_3d()


def test_import_dialog_separates_lowest_point_from_ground_clearance(qapp):
    import aerolume.geometry3d as g
    from aerolume.app.dialogs.mesh_import import MeshImportDialog
    from aerolume.app.widgets.mesh_inspector import CLEARANCE_TIP, clearance
    dlg = MeshImportDialog(None, path=str(EXAMPLES / "body_and_wheel.obj"))
    try:
        dlg.show()
        dlg.kind.setCurrentIndex(dlg.kind.findData("vehicle"))
        _process(qapp)
        labels = {w.text() for w in dlg.findChildren(type(dlg.contents))}
        assert "Lowest point" in labels and "Clearance" not in labels
        assert dlg._ground_clearance_label.toolTip() == CLEARANCE_TIP
        assert dlg.ground_clearance.isVisibleTo(dlg)
        wheels = [m for m in dlg.preview.meshes() if m.role == "wheel"]
        made = dlg.designs()[0]
        # the dialog states the clearance the Geometry page will show for the imported design
        assert dlg.ground_clearance.text().startswith(f"{clearance(made):.3f} m")
        assert len(wheels) == sum(m.role == "wheel" for m in made.meshes)
        if wheels:
            assert "recognised" in dlg.ground_clearance.text()
            assert clearance(made) > float(g.bbox(made.meshes)[4]) - 1e-12
        # a generic body: the lowest point is its clearance, so the second row goes
        dlg.kind.setCurrentIndex(dlg.kind.findData("generic"))
        _process(qapp)
        assert not dlg.ground_clearance.isVisibleTo(dlg)
    finally:
        dlg.done(0)
        dlg.preview.scene.close()


def test_inspector_clearance_carries_the_shared_definition(qapp):
    from aerolume.app.widgets.mesh_inspector import CLEARANCE_TIP, body_clearance, clearance
    car = _vehicle()
    assert clearance(car) == pytest.approx(body_clearance(car.meshes))
    body = [m for m in car.meshes if m.role != "wheel"]
    import aerolume.geometry3d as g
    assert body_clearance(car.meshes) == pytest.approx(g.bbox(body)[4])
    assert "wheels excluded" in CLEARANCE_TIP
