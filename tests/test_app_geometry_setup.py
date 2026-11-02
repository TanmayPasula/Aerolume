"""Geometry and Setup pages: helpers, widgets, page behaviour (offscreen Qt).

The pages are driven the way a user would drive them - through their widgets
and actions - against a real temporary :class:`aerolume.project.Project`, and
the assertions are made on the project's model objects, which is what the
rest of the application reads. Modal dialogs are bypassed through the pages'
``_exec_dialog`` / ``_confirm`` funnels.
"""
from __future__ import annotations

import os
import types
from dataclasses import dataclass, field

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from aerolume.core import units
from aerolume.model import Body, Design, FlowConditions, Fluid

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def geo(monkeypatch):
    """The geometry package with its documented API (DESIGN.md 5.1).

    While ``geometry/__init__.py`` is still being written, expose the
    submodules' functions on the package for the duration of a test.
    """
    pkg = pytest.importorskip("aerolume.geometry")
    names = ("naca", "read_airfoil_dat", "read_points_csv", "read_svg", "read_dxf", "read_image_silhouette",
             "design_from_airfoil", "design_from_polygons", "check_design", "resample", "smooth",
             "is_self_intersecting")
    missing = [n for n in names if not hasattr(pkg, n)]
    if missing:
        try:
            from aerolume.geometry import airfoils, csvio, design, dxf, image, polygon, svg
        except ImportError as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"geometry package incomplete: {exc}")
        for n in missing:
            for m in (polygon, airfoils, csvio, design, dxf, image, svg):
                if hasattr(m, n):
                    monkeypatch.setattr(pkg, n, getattr(m, n), raising=False)
                    break
    return pkg


@dataclass
class _Spec:
    name: str
    label: str
    unit_kind: str
    default: float
    min: float
    max: float
    step: float


@dataclass
class _Info:
    name: str
    kind: str
    description: str
    params: list = field(default_factory=list)


def _box_design(length=4.0, height=1.2, clearance=0.2, **_):
    pts = np.array([[0, clearance], [length, clearance], [length, clearance + height],
                    [0, clearance + height]], float)
    return Design(name="Box car", kind="vehicle", bodies=[Body("Body", pts)], length_ref=length,
                  force_ref=height, moment_ref=(length / 2, 0.0),
                  source={"type": "template", "template": "boxcar",
                          "params": {"length": length, "height": height, "clearance": clearance}})


FAKE_TEMPLATES = types.SimpleNamespace(
    list_templates=lambda: [_Info("boxcar", "vehicle", "A box on the road.", [
        _Spec("length", "Length", "length", 4.0, 1.0, 10.0, 0.05),
        _Spec("height", "Height", "length", 1.2, 0.5, 3.0, 0.01),
        _Spec("clearance", "Ground clearance", "length", 0.2, 0.0, 1.0, 0.005)])],
    make=lambda name, **p: _box_design(**p),
)


@pytest.fixture
def fake_templates(monkeypatch):
    import aerolume.app.dialogs.import_dialogs as dlg
    import aerolume.app.pages.geometry as gpage
    monkeypatch.setattr(gpage, "templates_module", lambda: FAKE_TEMPLATES)
    monkeypatch.setattr(dlg, "templates_module", lambda: FAKE_TEMPLATES)
    dlg._THUMB_CACHE.clear()
    return FAKE_TEMPLATES


@pytest.fixture
def state(qapp, tmp_path):
    project_mod = pytest.importorskip("aerolume.project")
    from aerolume.app.state import AppState
    proj = project_mod.Project.create(tmp_path / "Test", "Test")
    st = AppState()
    st.set_project(proj)
    yield st
    st.set_project(None)


def _settle(app, n=3):
    for _ in range(n):
        app.processEvents()


def _naca(geo, code="0012", chord=1.0):
    from aerolume.app.dialogs.import_dialogs import naca_design
    return naca_design(code, chord, 120)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_validate_naca_messages():
    from aerolume.app.dialogs.import_dialogs import validate_naca
    assert validate_naca("2412") is None
    assert validate_naca("NACA 0012") is None
    assert validate_naca("23012") is None
    assert "thickness" in validate_naca("2400").lower()
    assert "camber position" in validate_naca("2012").lower()
    assert validate_naca("12") is not None
    assert validate_naca("abcd") is not None
    assert validate_naca("26012") is not None        # camber position 6 does not exist


def test_classify_path():
    from aerolume.app.dialogs.import_dialogs import classify_path
    assert classify_path("wing.DAT") == "dat"
    assert classify_path("car.svg") == "svg"
    assert classify_path("car.png") == "image"
    assert classify_path("pts.csv") == "csv"
    assert classify_path("x.step") is None


def _square_vehicle():
    pts = np.array([[0, 0.1], [2, 0.1], [2, 1.1], [0, 1.1]], float)
    wheel = np.column_stack([0.5 + 0.2 * np.cos(np.linspace(0, 2 * np.pi, 32, endpoint=False)),
                             0.2 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, 32, endpoint=False))])
    return Design(name="v", kind="vehicle", bodies=[Body("b", pts), Body("w", wheel, role="wheel",
                                                                          center=(0.5, 0.2))],
                  length_ref=2.0, force_ref=1.1, moment_ref=(1.0, 0.0))


def test_auto_references_conventions():
    from aerolume.app.pages.geometry import auto_references
    d = _square_vehicle()
    length, force, mref = auto_references(d)
    assert length == pytest.approx(2.0)
    assert force == pytest.approx(1.1)          # union of [0.1,1.1] and [0,0.4] = [0, 1.1]
    assert mref == pytest.approx((1.0, 0.0))
    af = Design(name="a", kind="airfoil", bodies=[Body("a", np.array([[1, 0], [0.5, 0.05], [0, 0],
                                                                       [0.5, -0.05]], float))],
                length_ref=1, force_ref=1)
    c, f, m = auto_references(af)
    assert c == pytest.approx(1.0) and f == pytest.approx(1.0)
    assert m == pytest.approx((0.25, 0.0))


def test_apply_edit_mirror_keeps_ccw_and_scale_to():
    from aerolume.app.pages.geometry import apply_edit

    def area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    d = _square_vehicle()
    assert area(d.bodies[0].points) > 0
    apply_edit(d, {"op": "mirror", "axis": "x"})
    assert area(d.bodies[0].points) > 0
    assert d.bodies[1].center[0] == pytest.approx(1.5)        # wheel moved to the other end
    apply_edit(d, {"op": "scale_to", "length": 4.0})
    x0, x1, y0, _ = d.bounds()
    assert x1 - x0 == pytest.approx(4.0)
    assert y0 == pytest.approx(0.0)                           # scaled about the road origin
    apply_edit(d, {"op": "lift", "height": 0.05})
    assert d.bounds()[2] == pytest.approx(0.05)
    apply_edit(d, {"op": "rotate", "deg": 90})
    x0, x1, y0, y1 = d.bounds()
    assert y1 - y0 == pytest.approx(4.0)
    with pytest.raises(ValueError):
        apply_edit(d, {"op": "explode"})


def test_refresh_references_respects_manual_fields():
    from aerolume.app.pages.geometry import apply_edit, refresh_references
    d = _square_vehicle()
    d.force_ref = 0.5
    d.source = {"manual_refs": ["force_ref"]}
    apply_edit(d, {"op": "scale", "factor": 2.0})
    refresh_references(d, scale=2.0)
    assert d.length_ref == pytest.approx(4.0)                # auto
    assert d.force_ref == pytest.approx(1.0)                 # manual, scaled with the shape


def test_fluid_mode_roundtrip():
    from aerolume.app.pages.setup import fluid_mode, make_fluid
    from aerolume.core import fluid as fl
    for name in fl.PRESETS:
        f = fl.PRESETS[name]()
        assert fluid_mode(f) == ("preset", None)
        assert make_fluid("preset", None, preset=name).density == pytest.approx(f.density)
    assert fluid_mode(Fluid()) == ("preset", None)          # the model default is a preset
    f = make_fluid("isa", 2500.0)
    assert fluid_mode(f) == ("isa", 2500.0)
    assert f.density == pytest.approx(fl.air_isa(2500.0).density)
    f = make_fluid("air_t", 35.0)
    assert fluid_mode(f) == ("air_t", 35.0)
    f = make_fluid("water_t", 10.0)
    assert fluid_mode(f) == ("water_t", 10.0)
    f = make_fluid("custom", None, density=2.0, viscosity=3e-5)
    assert fluid_mode(f) == ("custom", None)
    assert (f.density, f.viscosity) == (2.0, 3e-5)


def test_mach_and_duration_helpers():
    from aerolume.app.pages.setup import format_duration, mach_number
    m = mach_number(FlowConditions(speed=34.03))
    assert m == pytest.approx(0.1, rel=1e-3)
    assert mach_number(FlowConditions(fluid=Fluid("Water, 20 C", 998, 1e-3), speed=3)) is None
    assert format_duration(42) == "~42 s"
    assert format_duration(600) == "~10 min"
    assert format_duration(4000) == "~1 h 07 min"


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------

def test_quantity_spinbox_accepts_any_unit(qapp):
    from PySide6.QtGui import QValidator

    from aerolume.app.widgets.param_form import QuantitySpinBox
    w = QuantitySpinBox("length", "m", minimum=0, maximum=100, step=0.01)
    w.set_si_value(4.5)
    assert w.si_value() == pytest.approx(4.5)
    assert w.text().endswith(" m")
    assert w.valueFromText("180 mm") == pytest.approx(0.18)
    assert w.validate("180 mm", 0)[0] == QValidator.State.Acceptable
    assert w.validate("abc", 0)[0] == QValidator.State.Intermediate
    got = []
    w.siValueChanged.connect(got.append)
    w.set_unit("mm")
    assert w.si_value() == pytest.approx(4.5)                # unit switch keeps SI
    assert w.value() == pytest.approx(4500)
    assert not got                                           # and does not emit an edit
    w.set_si_value(1.0, emit=True)
    assert got and got[-1] == pytest.approx(1.0)


def test_param_form_builds_fields_and_reports_si(qapp):
    from aerolume.app.widgets.param_form import ParamForm, QuantitySpinBox
    specs = [_Spec("length", "Length", "length", 4.0, 1, 10, 0.05),
             _Spec("n", "Spokes", "count", 5, 3, 12, 1),
             _Spec("ratio", "Taper", "", 0.5, 0, 1, 0.01)]
    form = ParamForm(specs, preferred_units={"length": "mm"})
    assert isinstance(form.field("length"), QuantitySpinBox)
    assert form.values() == pytest.approx({"length": 4.0, "n": 5, "ratio": 0.5})
    seen = []
    form.valuesChanged.connect(seen.append)
    form.set_values({"length": 2.5})
    assert not seen                                          # programmatic loads are silent
    assert form.values()["length"] == pytest.approx(2.5)
    form.field("length").setValue(3000.0)                    # user types 3000 mm
    assert seen and seen[-1]["length"] == pytest.approx(3.0)


def test_canvas_paints_and_navigates(qapp, geo):
    from aerolume.app.widgets.geometry_canvas import GeometryCanvas, max_thickness, nice_step
    c = GeometryCanvas()
    c.resize(800, 500)
    c.show()
    _settle(qapp)
    assert not c.grab().isNull()                             # empty state paints
    d = _square_vehicle()
    c.set_design(d)
    c.set_show_vertices(True)
    img = c.grab().toImage()
    assert not img.isNull()
    # the body fill is visible at the centre of the car
    p = c.to_screen(1.0, 0.6)
    assert img.pixelColor(int(p.x()), int(p.y())) != img.pixelColor(5, 5)
    s0 = c.scale()
    ev = QWheelEvent(QPointF(400, 250), QPointF(400, 250), QPoint(0, 0), QPoint(0, 240),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                     False)
    c.wheelEvent(ev)
    assert c.scale() > s0
    c.fit_view(animate=False)
    assert c.scale() == pytest.approx(s0)
    # world <-> screen round trip
    x, y = c.to_world(c.to_screen(1.234, -0.5))
    assert (x, y) == pytest.approx((1.234, -0.5))
    assert nice_step(0.13) == pytest.approx(0.2)
    af = _naca(geo, "0012")
    c.set_design(af)
    assert not c.grab().isNull()
    tc, _ = max_thickness(af)
    assert tc == pytest.approx(0.12, abs=0.004)


def test_canvas_stays_fitted_until_the_user_navigates(qapp):
    from aerolume.app.widgets.geometry_canvas import GeometryCanvas
    c = GeometryCanvas()
    c.resize(400, 300)
    c.show()
    c.set_design(_square_vehicle())
    _settle(qapp)
    s_small = c.scale()
    c.resize(1200, 800)                                      # e.g. the window laid out after show
    _settle(qapp)
    assert c.scale() > 1.5 * s_small                         # refitted
    c.zoom(2.0)
    s_user = c.scale()
    c.resize(1000, 700)
    _settle(qapp)
    assert c.scale() == pytest.approx(s_user)                # the user's view is kept
    c.fit_view(animate=False)
    c.resize(800, 600)
    _settle(qapp)
    assert c.scale() < s_user                                # fitted again after Fit


def test_canvas_reports_dropped_files(qapp, tmp_path):
    from aerolume.app.widgets.geometry_canvas import GeometryCanvas
    c = GeometryCanvas()
    got = []
    c.filesDropped.connect(got.append)
    f = tmp_path / "wing.dat"
    f.write_text("x\n")
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(f))])
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier)
    c.dropEvent(ev)
    assert got and os.path.samefile(got[0][0], f)


def test_naca_dialog_validates_and_builds(qapp, geo):
    from aerolume.app.dialogs.import_dialogs import NacaDialog
    dlg = NacaDialog(code="4412", chord=0.3)
    d = dlg.design()
    assert d is not None and d.kind == "airfoil"
    assert d.length_ref == pytest.approx(0.3)
    x0, x1, _, _ = d.bounds()
    assert x1 - x0 == pytest.approx(0.3, rel=1e-6)
    assert d.source["type"] == "naca" and d.source["code"] == "4412"
    dlg.code_edit.setText("4400")
    assert dlg.design() is None
    from PySide6.QtWidgets import QDialogButtonBox
    assert not dlg.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()


def test_template_gallery_lists_and_builds(qapp, fake_templates):
    from aerolume.app.dialogs.import_dialogs import TemplateGalleryDialog
    g = TemplateGalleryDialog()
    assert g.template_names() == ["boxcar"]
    assert g.selected_name() == "boxcar"
    assert not g.items["boxcar"].icon(0).isNull()           # thumbnail of the default shape
    assert g.params_view.topLevelItemCount() == 3            # default parameters listed
    g.search.setText("nothing-matches")
    assert not g.is_listed("boxcar")
    g.search.setText("box")
    assert g.is_listed("boxcar")
    d = g.design()
    assert d.kind == "vehicle"


# ---------------------------------------------------------------------------
# Geometry page
# ---------------------------------------------------------------------------

def test_geometry_page_without_project(qapp):
    pytest.importorskip("aerolume.project")
    from aerolume.app.pages.geometry import GeometryPage
    from aerolume.app.state import AppState
    st = AppState()
    page = GeometryPage(st, None)
    assert page._pages.currentIndex() == 0                   # friendly empty state
    nav = []
    st.navigateRequested.connect(nav.append)
    page.handle_action("import:nothing.dat")                 # tolerated without a project
    page.on_shown()


def test_geometry_page_edit_undo_redo(qapp, state, geo):
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    page.resize(1600, 1000)
    page.show()
    assert page._inspector_stack.currentIndex() == 0         # "add your first design"
    assert not hasattr(page, "design_list")                  # the Project tree owns selection
    page.add_design(_naca(geo, "2412", 1.0))
    d = state.design
    assert d is not None and len(state.project.designs) == 1
    assert page._inspector_stack.currentIndex() == 1
    assert page.canvas.design() is d
    assert page.title_label.text() == "NACA 2412"
    assert page.naca_box.isVisibleTo(page) and not page.template_box.isVisibleTo(page)
    assert not page._scale_row.editor.isVisibleTo(page)      # the chord lives in Source for NACA
    assert not page.grab().isNull()

    # scale to a 0.5 m chord: geometry, auto references and the recipe all follow
    page.scale_length.set_si_value(0.5)
    page.apply_scale()
    d2 = state.design
    assert d2.length_ref == pytest.approx(0.5, rel=1e-6)
    assert d2.moment_ref[0] == pytest.approx(0.125, rel=1e-4)
    assert d2.source["edits"][-1]["op"] == "scale_to"
    page._undo_group.undo()
    assert state.design.length_ref == pytest.approx(1.0, rel=1e-6)
    page._undo_group.redo()
    assert state.design.length_ref == pytest.approx(0.5, rel=1e-6)

    # NACA code change regenerates and replays the scale edit
    page._naca_changed({"code": "0015"}, "naca_code", rename=True)
    d3 = state.design
    assert d3.source["code"] == "0015" and d3.name == "NACA 0015"
    assert d3.length_ref == pytest.approx(0.5, rel=1e-6)
    from aerolume.app.widgets.geometry_canvas import max_thickness
    assert max_thickness(d3)[0] == pytest.approx(0.15, abs=0.005)

    # a typed reference becomes manual; Auto releases it
    page.force_ref.set_si_value(0.7, emit=True)
    assert state.design.force_ref == pytest.approx(0.7)
    assert "force_ref" in state.design.source["manual_refs"]
    assert not page._auto_buttons["force_ref"].isChecked()
    page._auto_buttons["force_ref"].click()
    assert state.design.force_ref == pytest.approx(0.5, rel=1e-6)

    # duplicate / rename / delete
    dup = page.duplicate_design()
    assert dup is not None and len(state.project.designs) == 2 and state.design_id == dup.id
    page._ask_text = lambda *a: "Renamed"
    page.rename_design()
    assert state.project.designs[dup.id].name == "Renamed"
    page._confirm = lambda *a: True
    page.delete_design()
    assert dup.id not in state.project.designs
    assert state.design_id                                   # another design got selected


def test_geometry_page_template_params_regenerate(qapp, state, fake_templates, geo):
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    page.add_design(FAKE_TEMPLATES.make("boxcar"))
    assert page.template_box.isVisibleTo(page)
    assert page.param_form.names() == ["length", "height", "clearance"]
    # a transform edit first ...
    page.apply_edit({"op": "translate", "dx": 1.0, "dy": 0.0}, "Move")
    stack = page._stacks[state.design_id]
    n = stack.count()
    # ... then a parameter change: the shape is rebuilt and the move replayed
    page.param_form.field("height").set_si_value(2.0, emit=True)
    page._flush_regeneration()
    d = state.design
    x0, _, y0, y1 = d.bounds()
    assert y1 - y0 == pytest.approx(2.0)
    assert x0 == pytest.approx(1.0)
    assert d.source["params"]["height"] == pytest.approx(2.0)
    assert d.force_ref == pytest.approx(2.0)                 # frontal height follows the shape
    # quick successive edits of one field merge into a single undo step ...
    page.param_form.field("height").set_si_value(2.1, emit=True)
    page._flush_regeneration()
    page.param_form.field("height").set_si_value(2.2, emit=True)
    page._flush_regeneration()
    assert stack.count() == n + 1
    stack.undo()
    assert state.design.bounds()[3] - state.design.bounds()[2] == pytest.approx(1.2)
    stack.redo()
    # ... but not edits made after a pause
    from aerolume.app.pages.geometry import DesignEditCommand
    monkey_window = DesignEditCommand.MERGE_WINDOW_S
    DesignEditCommand.MERGE_WINDOW_S = -1.0
    try:
        page.param_form.field("height").set_si_value(2.4, emit=True)
        page._flush_regeneration()
    finally:
        DesignEditCommand.MERGE_WINDOW_S = monkey_window
    assert stack.count() == n + 2
    page._reset_params()
    assert state.design.source["params"]["height"] == pytest.approx(1.2)


def test_geometry_page_import_and_diagnostics(qapp, state, geo, tmp_path, fake_templates):
    from aerolume.app.pages.geometry import GeometryPage, check_messages
    page = GeometryPage(state, None)
    page._exec_dialog = lambda dlg: True
    # airfoil .dat (Selig format) and a CSV outline
    coords = np.asarray(geo.naca("2412", n=100, chord=1.0))
    dat = tmp_path / "my_foil.dat"
    dat.write_text("MY FOIL\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in coords))
    added = page.import_files([str(dat)])
    assert len(added) == 1 and added[0].kind == "airfoil"
    assert added[0].source["format"] == "dat" and added[0].source["file"] == "my_foil.dat"
    csv = tmp_path / "shape.csv"
    sq = np.array([[0, 0], [3, 0], [3, 1], [0, 1]], float) * 100      # drawing in "mm"
    csv.write_text("x,y\n" + "\n".join(f"{x},{y}" for x, y in sq))
    added = page.import_files([str(csv)])
    assert len(added) == 1
    assert page.import_info.isVisibleTo(page)
    assert len(state.project.designs) == 2
    # unsupported files are reported, not raised
    errors = []
    page._error = lambda t, m: errors.append(m)
    page.import_files([str(tmp_path / "model.step")])
    assert errors
    # diagnostics: a vehicle sunk into the road is an error
    bad = _square_vehicle()
    for b in bad.bodies:
        b.points = b.points - [0.0, 0.5]
    msgs = check_messages(bad)
    assert msgs and all(lv in ("error", "warning", "info") for lv, _ in msgs)
    page.add_design(bad)
    assert page.diag_badge.text() != "OK"


# ---------------------------------------------------------------------------
# Setup page
# ---------------------------------------------------------------------------

class _FakeJobs:
    def __init__(self):
        self.submitted = []

    def submit(self, sim_id):
        self.submitted.append(sim_id)

    def cancel(self, sim_id):
        pass

    def running_id(self):
        return None

    def queued_ids(self):
        return []


def test_setup_page_empty_states(qapp, state):
    from aerolume.app.pages.setup import SetupPage
    page = SetupPage(state, None)
    assert page._pages.currentIndex() == 0                   # no design yet
    assert page.empty.title.text() == "No design selected"
    nav = []
    state.navigateRequested.connect(nav.append)
    page.empty.button.click()
    assert nav == ["geometry"]


def test_setup_page_edits_update_simulation(qapp, state, geo):
    from aerolume.app.pages.setup import SetupPage
    from aerolume.project import Project
    state.add_design(_naca(geo, "2412", 0.5))
    jobs = _FakeJobs()
    page = SetupPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    page.on_shown()                                          # creates a first simulation
    sim = state.simulation
    assert sim is not None and sim.design_id == state.design_id
    assert sim.conditions.alpha_deg == pytest.approx(4.0)    # airfoil default
    assert page._pages.currentIndex() == 1
    assert page.title_label.text() == sim.name and page.status_text() == "Not run"
    assert not hasattr(page, "sim_list")                     # the Project tree selects simulations

    page.speed.set_si_value(40.0, emit=True)
    page.alpha_slider.setValue(13)                           # half-degree slider -> 6.5 deg
    assert sim.conditions.speed == pytest.approx(40.0)
    assert sim.conditions.alpha_deg == pytest.approx(6.5)
    assert page.alpha.si_value() == pytest.approx(6.5)
    # unit selector changes the display only
    page.speed_unit.setCurrentIndex(page.speed_unit.findData("km/h"))
    assert page.speed.unit == "km/h" and page.speed.text() == "144.0"    # the combo carries the unit
    assert page.speed.value() == pytest.approx(144.0)
    assert sim.conditions.speed == pytest.approx(40.0)
    page.speed.lineEdit().setText("90 mph")                  # any unit is still accepted when typed
    page.speed.interpretText()
    assert sim.conditions.speed == pytest.approx(90 * 0.44704)
    page.speed.set_si_value(40.0, emit=True)

    # fluid: ISA altitude; density follows and is read-only
    page.fluid_combo.setCurrentIndex(page.fluid_combo.findData("isa"))
    assert page._alt_row.editor.isVisibleTo(page) and not page._temp_row.editor.isVisibleTo(page)
    page.altitude.set_si_value(3000.0, emit=True)
    from aerolume.core import fluid as fl
    assert sim.conditions.fluid.density == pytest.approx(fl.air_isa(3000).density)
    assert page.density.isReadOnly()

    # quality and an override that implies Custom
    page.quality.setCurrentIndex(page.quality.findData("fine"))
    assert sim.cfd.quality == "fine"
    page.cells_per_ref.setValue(128)
    assert sim.cfd.cells_per_ref == 128 and sim.cfd.quality == "custom"
    assert page.quality.currentData() == "custom"
    page.cells_per_ref.setValue(0)
    assert sim.cfd.cells_per_ref is None                     # "Auto"
    assert not page.solver_group.is_expanded()               # Solver starts collapsed
    page.turbulence.setCurrentIndex(page.turbulence.findData("les"))
    assert sim.cfd.turbulence == "les"
    # a new airfoil simulation above Re 1e5 starts capped, with a cap fixed when it is created
    # (simulation.defaults), so every quality and device simulates the same flow; the cap
    # editor shows it instead of "Auto"
    from aerolume.simulation.defaults import AIRFOIL_DEFAULT_CAP_RE
    assert sim.cfd.reynolds_mode == "capped" and sim.cfd.reynolds_cap == AIRFOIL_DEFAULT_CAP_RE
    assert page.re_cap.value() == pytest.approx(AIRFOIL_DEFAULT_CAP_RE)
    page.re_mode.setCurrentIndex(page.re_mode.findData("real"))
    assert sim.cfd.reynolds_mode == "real" and not page._re_cap_row.editor.isEnabled()
    page.re_mode.setCurrentIndex(page.re_mode.findData("capped"))
    assert sim.cfd.reynolds_mode == "capped" and page._re_cap_row.editor.isEnabled()
    assert sim.cfd.reynolds_cap == AIRFOIL_DEFAULT_CAP_RE    # the cap survived the round trip
    page.re_cap.setValue(20000.0)
    assert sim.cfd.reynolds_cap == 20000.0
    page.re_cap.setValue(0.0)
    assert sim.cfd.reynolds_cap is None and page.re_cap.specialValueText() == "Auto"
    page.re_mode.setCurrentIndex(page.re_mode.findData("real"))
    page.re_mode.setCurrentIndex(page.re_mode.findData("capped"))
    assert sim.cfd.reynolds_cap == AIRFOIL_DEFAULT_CAP_RE    # switching to Capped fixes a cap again
    assert page.re_cap.value() == pytest.approx(AIRFOIL_DEFAULT_CAP_RE)

    # the name field renames the simulation (and saves at once, for the Project tree)
    page.name_edit.setText("Cruise")
    page.name_edit.editingFinished.emit()
    assert state.project.simulations[sim.id].name == "Cruise" and page.title_label.text() == "Cruise"

    # polar is allowed for a single airfoil; its group replaces the CFD groups
    page.analysis.buttons["polar"].click()
    assert sim.analysis == "polar"
    assert page.polar_group.isVisibleTo(page)
    assert not page.mesh_group.isVisibleTo(page) and not page.attitude_group.isVisibleTo(page)
    assert not page.live_button.isEnabled()
    page.ncrit.setValue(11.0)
    page.alpha_end.set_si_value(10.0, emit=True)
    assert sim.airfoil.ncrit == 11.0 and sim.airfoil.alpha_end == pytest.approx(10.0)

    # Estimate shows the real Reynolds number for the edited state
    page.update_summary()
    re_real = sim.conditions.reynolds(state.design.length_ref)
    assert page.last_summary.get("Reynolds number") == units.fmt_reynolds(re_real)
    assert page.estimate.value_text("Reynolds number") == units.fmt_reynolds(re_real)
    assert page.plan_status == "polar"

    # debounced save reaches disk
    page.flush()
    reopened = Project.open(state.project.path)
    s2 = reopened.simulations[sim.id]
    assert s2.analysis == "polar" and s2.conditions.speed == pytest.approx(40.0)
    assert s2.cfd.turbulence == "les" and s2.airfoil.ncrit == 11.0 and s2.name == "Cruise"

    # run -> JobManager + navigation
    nav = []
    state.navigateRequested.connect(nav.append)
    page.run_button.click()
    assert jobs.submitted == [sim.id] and nav[-1] == "simulate:run"
    page.analysis.buttons["cfd"].click()
    assert page.live_button.isEnabled()
    page.live_button.click()
    assert nav[-1] == "simulate:live"
    assert not page.grab().isNull()


def test_setup_page_vehicle_ground_and_actions(qapp, state):
    from aerolume.app.pages.setup import SetupPage
    state.add_design(_square_vehicle())
    page = SetupPage(state, None)
    page.on_shown()
    sim = state.simulation
    assert sim.ground.enabled and sim.ground.moving and sim.ground.rotate_wheels
    assert page.ground_group.isVisibleTo(page)
    assert page.ground_combo.currentData() == "moving" and page.ground_wheels.isEnabled()
    assert page._alpha_row.label.text() == "Pitch angle"
    assert page._cells_row.label.text() == "Cells per length"
    assert not page.analysis.buttons["polar"].isEnabled()    # polars are airfoil-only
    page.ground_combo.setCurrentIndex(page.ground_combo.findData("fixed"))
    assert sim.ground.enabled and not sim.ground.moving
    assert not sim.ground.rotate_wheels and not page.ground_wheels.isEnabled()
    page.ground_combo.setCurrentIndex(page.ground_combo.findData("none"))
    assert not sim.ground.enabled
    page._set_analysis("polar")
    assert sim.analysis == "cfd"
    # simulation actions (also in the Project tree)
    dup = page.duplicate_simulation()
    assert dup is not None and state.simulation_id == dup.id
    assert len(page._sims_of(state.design_id)) == 2
    assert dup.ground.enabled is False                       # setup copied
    page._confirm = lambda *a: True
    page.delete_simulation()
    assert dup.id not in state.project.simulations
    assert state.simulation_id == sim.id
    new = page.new_simulation()
    assert new is not None and len(page._sims_of(state.design_id)) == 2
    page.handle_action("new")
    assert len(page._sims_of(state.design_id)) == 3


def test_setup_header_status_follows_jobs(qapp, state):
    from aerolume.app.pages.setup import SetupPage
    state.add_design(_square_vehicle())
    jobs = _FakeJobs()
    page = SetupPage(state, jobs)
    page.on_shown()
    sim = state.simulation
    jobs.running_id = lambda: sim.id
    page._on_job_progress(sim.id, 0.43, "")
    assert page.status_text() == "Running 43 %"
    jobs.running_id = lambda: None
    sim.status = "done"
    page._on_job_event(sim.id)
    assert page.status_text() == "Done"
    assert any("next run" in text for _lv, text in page._context_lines(sim))


def test_setup_summary_uses_planner_when_available(qapp, state, monkeypatch):
    """With a planner present the Estimate shows its rows, the Planner strip its
    notes and warnings, and each line is logged to Messages once."""
    from aerolume.app.pages import setup as setup_mod

    class _Plan:
        extent = (-5.0, 15.0, 0.0, 6.0)
        bodies_wind = None

        def __init__(self, notes=("Re capped for the grid",)):
            self.notes = list(notes)
            self.warnings = ["Thin feature"]

        def summary(self):
            return {"Real Reynolds number": "1.00×10^6", "Simulated Reynolds number": "5,000",
                    "Cells": "131,072"}

    loader = setup_mod._PlannerLoader.get()
    monkeypatch.setattr(loader, "state", "ready")
    monkeypatch.setattr(loader, "module", types.SimpleNamespace(plan_simulation=lambda d, s: _Plan()))
    state.add_design(_square_vehicle())
    logged = []
    state.messageLogged.connect(lambda level, source, text: logged.append((level, source, text)))
    page = setup_mod.SetupPage(state, None)
    page.on_shown()
    page.update_summary()
    assert page.plan_status == "ok"
    assert page.last_summary["Cells"] == "131,072"
    assert "Dynamic pressure" in page.last_summary and "Mach number" in page.last_summary
    assert page.estimate.value_text("Real Reynolds number") == "1.00×10^6"
    kinds = [page.planner_box.itemAt(i).widget().property("summary_kind")
             for i in range(page.planner_box.count()) if page.planner_box.itemAt(i).widget() is not None]
    assert kinds.count("warning") == 1 and kinds.count("note") == 1
    # logged to Messages once, however often the summary is recomputed
    page.flush_planner_log()
    page.update_summary()
    page.flush_planner_log()
    assert sorted(lv for lv, _src, _t in logged) == ["info", "warning"]
    assert all(src == "Planner" for _lv, src, _t in logged)
    # a changed note is logged; the unchanged warning is not repeated
    monkeypatch.setattr(loader, "module", types.SimpleNamespace(
        plan_simulation=lambda d, s: _Plan(notes=("Grid refined",))))
    page.update_summary()
    page.flush_planner_log()
    assert [t for _lv, _src, t in logged[2:]] == ["Grid refined"]
    # the case preview shows the planned domain
    assert page.preview._extent == pytest.approx(_Plan.extent)
    # a planner error is shown, not raised
    monkeypatch.setattr(loader, "module", types.SimpleNamespace(
        plan_simulation=lambda d, s: (_ for _ in ()).throw(ValueError("bad domain"))))
    page.update_summary()
    assert page.plan_status == "error"
    assert any(lv == "error" and "bad domain" in t for lv, t in page.planner_lines)


def test_setup_with_the_real_planner(qapp, state, geo):
    """The shipped planner fills the Estimate for an airfoil (plans in ~50 ms)."""
    from aerolume.app.pages import setup as setup_mod
    loader = setup_mod._PlannerLoader.get()
    if not loader.wait(120):
        pytest.skip(f"planner not importable: {loader.error}")
    state.add_design(_naca(geo, "0012", 1.0))
    page = setup_mod.SetupPage(state, None)
    page.on_shown()
    page.update_summary()
    assert page.plan_status == "ok", page.planner_lines
    rows = dict(page.estimate.items())
    for key in ("Real Reynolds number", "Simulated Reynolds number", "Grid", "Cells", "Estimated run time"):
        assert key in rows, key
    assert page.planner_lines                                # the planner explains its choices
    assert page.preview._extent is not None


def test_setup_defers_planning_while_hidden(qapp, state, monkeypatch):
    """Design edits made on the Geometry page must not run the planner on a hidden Setup page."""
    from aerolume.app.pages import setup as setup_mod
    calls = []
    loader = setup_mod._PlannerLoader.get()
    monkeypatch.setattr(loader, "state", "ready")
    monkeypatch.setattr(loader, "module", types.SimpleNamespace(
        plan_simulation=lambda d, s: calls.append(1) or types.SimpleNamespace(notes=[], warnings=[],
                                                                              summary=lambda: {"Cells": "1"})))
    state.add_design(_square_vehicle())
    page = setup_mod.SetupPage(state, None)
    page.on_shown()
    n = len(calls)
    state.update_design(state.design)                        # e.g. a parameter edit on Geometry
    _settle(qapp)
    assert len(calls) == n and page._summary_stale
    page.show()
    _settle(qapp)
    assert len(calls) == n + 1 and not page._summary_stale
    page.hide()


def test_pretty_planner_text():
    from aerolume.app.pages.setup import pretty
    assert pretty("Re = 6.16×10^5") == "Re = 6.16×10⁵"
    assert pretty("1×10^-3 -- coarse") == "1×10⁻³ — coarse"
    assert pretty("plain") == "plain"


def test_image_silhouette_dialog_threshold_and_invert(qapp, geo, tmp_path):
    from PIL import Image, ImageDraw

    from aerolume.app.dialogs.import_dialogs import ImageSilhouetteDialog
    img = Image.new("L", (400, 160), 235)
    ImageDraw.Draw(img).rounded_rectangle((40, 40, 360, 120), radius=25, fill=30)
    path = tmp_path / "car.png"
    img.save(path)
    dlg = ImageSilhouetteDialog(str(path))
    d = dlg.design()
    assert d is not None and dlg.kind() == "vehicle"          # 4:1 outline -> vehicle guess
    x0, x1, _, _ = d.bounds()
    assert x1 - x0 == pytest.approx(dlg.length.si_value(), rel=1e-6)
    assert d.source["format"] == "image" and d.source["threshold"] is None
    # manual threshold: the slider value is a 0..255 grey level
    dlg.auto_threshold.setChecked(False)
    dlg.threshold.setValue(90)
    dlg._reload()
    assert dlg.threshold_value_arg() == 90.0
    assert dlg.design() is not None and dlg.design().source["threshold"] == 90.0
    # inverting traces the background instead: a different outline (or none)
    dlg.invert.setChecked(True)
    dlg._reload()
    assert dlg.design() is None or dlg.design().source["invert"] is True
    assert not dlg.preview.grab().isNull()


def test_svg_import_is_scaled_to_real_length(qapp, geo, tmp_path):
    from aerolume.app.dialogs.import_dialogs import PolygonImportDialog
    svg = tmp_path / "shape.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">'
                   '<path d="M 10 10 L 290 10 L 290 90 L 10 90 Z"/></svg>')
    dlg = PolygonImportDialog(str(svg))
    assert dlg.polygons(), dlg.status.text()
    dlg.kind_combo.setCurrentIndex(dlg.kind_combo.findData("generic"))
    dlg.length.set_si_value(2.0, emit=True)
    d = dlg.design()
    x0, x1, y0, y1 = d.bounds()
    assert x1 - x0 == pytest.approx(2.0, rel=1e-6)
    assert y1 - y0 == pytest.approx(2.0 * 80 / 280, rel=1e-3)


def test_wheel_does_not_change_unfocused_fields(qapp):
    from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

    from aerolume.app.widgets.param_form import QuantitySpinBox, guard_wheel
    area = QScrollArea()
    inner = QWidget()
    lay = QVBoxLayout(inner)
    box = QuantitySpinBox("speed", "m/s", minimum=0, maximum=100, step=1)
    box.set_si_value(30.0)
    lay.addWidget(box)
    area.setWidget(inner)
    guard_wheel(inner)
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, 120),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                     False)
    QApplication.sendEvent(box, ev)
    assert box.si_value() == pytest.approx(30.0)


def test_real_templates_build_forms_and_regenerate(qapp, state, geo):
    """Every shipped template gets a working form; editing one parameter rebuilds it."""
    templates = pytest.importorskip("aerolume.geometry.templates")
    from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit

    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    for info in templates.list_templates():
        page.add_design(templates.make(info.name))
        d = state.design
        assert page.template_box.isVisibleTo(page), info.name
        assert page.param_form.names() == [p.name for p in info.params]
        # change the first parameter the way a user would
        spec = info.params[0]
        w = page.param_form.field(spec.name)
        if isinstance(w, QCheckBox):
            w.setChecked(not w.isChecked())
        elif isinstance(w, QComboBox):
            w.setCurrentIndex((w.currentIndex() + 1) % w.count())
        elif isinstance(w, QLineEdit):
            continue                                   # e.g. a NACA code; covered elsewhere
        else:
            v = page.param_form.values()[spec.name]
            target = v + (spec.step or 0.01) if spec.max is None or v + (spec.step or 0.01) <= spec.max \
                else v - (spec.step or 0.01)
            if hasattr(w, "set_si_value"):
                w.set_si_value(target, emit=True)
            else:
                w.setValue(target)
        page._flush_regeneration()
        d2 = state.design
        assert d2.id == d.id and d2.source["template"] == info.name
        assert d2.source["params"][spec.name] == page.param_form.values()[spec.name], info.name


def test_naca_template_code_field_is_validated(qapp, state, geo):
    templates = pytest.importorskip("aerolume.geometry.templates")
    names = [i.name for i in templates.list_templates()]
    if "naca" not in names:
        pytest.skip("no NACA template")
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    page.add_design(templates.make("naca"))
    before = state.design.to_dict()["bodies"][0]["points"]
    field = page.param_form.field("naca")
    field.setText("2400")                                    # 0 % thickness: invalid
    assert field.property("invalid") == "true"
    field.editingFinished.emit()
    page._flush_regeneration()
    assert state.design.to_dict()["bodies"][0]["points"] == before     # nothing rebuilt
    field.setText("0015")
    assert field.property("invalid") == "false"
    field.editingFinished.emit()
    page._flush_regeneration()
    assert state.design.source["params"]["naca"] == "0015"


def test_pages_tolerate_a_design_without_bodies(qapp, state):
    from aerolume.app.pages.geometry import GeometryPage
    from aerolume.app.pages.setup import SetupPage
    page = GeometryPage(state, None)
    page.add_design(Design(name="Empty", kind="generic", bodies=[], length_ref=1.0, force_ref=1.0))
    assert not page.grab().isNull()
    sp = SetupPage(state, None)
    sp.on_shown()
    sp.update_summary()
    assert not sp.grab().isNull()


# ---------------------------------------------------------------------------
# UI critique round 1: units, switches, import placement, compact Setup text
# ---------------------------------------------------------------------------

def test_quantity_spinbox_one_decimal_convention_per_quantity(qapp):
    from aerolume.app.widgets.param_form import QuantitySpinBox
    # a 0.01 m step and a 0.001 m step read the same: "1.000 m" twice, not "1.00" beside "1.000"
    a = QuantitySpinBox("length", "m", minimum=0, maximum=10, step=0.01)
    b = QuantitySpinBox("length", "m", minimum=0, maximum=10, step=0.001)
    a.set_si_value(1.0)
    b.set_si_value(1.0)
    assert a.text() == b.text() == "1.000 m"
    assert QuantitySpinBox("length", "m", step=0.0005).decimals() == 4      # a finer step still shows
    assert QuantitySpinBox("angle", "deg", step=1.0).decimals() == 1
    assert QuantitySpinBox("angle", "deg", step=0.25).decimals() == 2
    mph = QuantitySpinBox("speed", "mph", step=1.0)                        # 1 m/s = 2.2369... mph
    assert mph.decimals() == 1
    bare = QuantitySpinBox("speed", "m/s", step=1.0, show_suffix=False)
    bare.set_si_value(30.0)
    assert bare.text() == "30.00" and bare.suffix() == ""
    assert bare.valueFromText("108 km/h") == pytest.approx(30.0)          # typed units still work


def test_quantity_spinbox_keeps_significant_figures_of_slow_speeds(qapp):
    """The Setup speed box keeps three significant figures: a Re = 40 cylinder of 20 mm runs at
    0.0301 m/s, which a two-decimal box showed as 0.03 and would not accept typed in."""
    from PySide6.QtTest import QTest

    from aerolume.app.widgets.param_form import QuantitySpinBox
    box = QuantitySpinBox("speed", "m/s", minimum=0.01, maximum=400.0, step=1.0, show_suffix=False, significant=3)
    box.show()
    box.set_si_value(0.0301)
    assert box.text() == "0.0301" and box.si_value() == pytest.approx(0.0301)
    box.set_si_value(30.0)
    assert box.text() == "30.00"                                          # the convention for normal speeds
    got = []
    box.siValueChanged.connect(got.append)
    box.lineEdit().selectAll()
    QTest.keyClicks(box, "0.0452")
    QTest.keyClick(box, Qt.Key.Key_Return)
    assert got[-1] == pytest.approx(0.0452) and box.text() == "0.0452"    # typed, not rounded to 0.05
    box.set_si_value(0.0301)
    box.set_unit("km/h")
    box.set_unit("m/s")
    assert box.text() == "0.0301"                                         # no drift through a unit switch
    QTest.keyClick(box, Qt.Key.Key_Up)
    assert got[-1] == pytest.approx(1.03) and box.text() == "1.03"        # the value stored is the one shown
    plain = QuantitySpinBox("speed", "m/s", step=1.0, show_suffix=False)
    plain.set_si_value(0.0301)
    assert plain.text() == "0.03"                                         # opt-in: other boxes unchanged


def test_param_form_switch_disables_its_parameters(qapp):
    from aerolume.app.widgets.param_form import ParamForm, switch_dependents
    specs = [_Spec("length", "Length", "length", 4.0, 1, 10, 0.05),
             _Spec("wheels", "Wheels", "bool", False, None, None, None),
             _Spec("wheel_radius", "Wheel radius", "length", 0.3, 0.2, 0.6, 0.005),
             _Spec("wheel_gap", "Wheel gap to road", "length", 0.0, 0.0, 0.1, 0.002),
             _Spec("closed_te", "Sharp trailing edge", "bool", True, None, None, None)]
    assert switch_dependents(specs) == {"wheels": ["wheel_radius", "wheel_gap"]}
    form = ParamForm(specs)
    assert form.is_gated_off("wheel_radius") and form.is_gated_off("wheel_gap")
    assert not form.label_for("wheel_gap").isEnabled()
    assert not form.is_gated_off("length")
    form.field("wheels").setChecked(True)                    # the user switches wheels on
    assert not form.is_gated_off("wheel_radius") and not form.is_gated_off("wheel_gap")
    form.set_values({"wheels": False})                       # loading the model gates them again
    assert form.is_gated_off("wheel_radius")


def test_real_sedan_template_gates_wheel_parameters(qapp, state, geo):
    templates = pytest.importorskip("aerolume.geometry.templates")
    if "sedan" not in [i.name for i in templates.list_templates()]:
        pytest.skip("no sedan template")
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    page.add_design(templates.make("sedan"))
    assert not state.design.source["params"].get("wheels", False)
    assert page.param_form.is_gated_off("wheel_radius") and page.param_form.is_gated_off("wheel_gap")
    page.param_form.field("wheels").setChecked(True)
    page._flush_regeneration()
    assert state.design.source["params"]["wheels"] is True
    assert not page.param_form.is_gated_off("wheel_radius")


def _circle(cx, cy, r, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _car_outlines():
    """A drawing in mm: body 4100 long standing 140 above the road, two wheels on
    the road (y = 0), and a stray circle above the roof."""
    body = np.array([[0, 140], [4100, 140], [4100, 900], [3000, 900], [2600, 1480], [1200, 1480],
                     [800, 900], [0, 800]], float)
    return [_circle(700, 310, 310), _circle(3300, 310, 310), _circle(1600, 1800, 100), body]


def test_classify_outlines_and_place_on_road(geo):
    from aerolume.app.dialogs.import_dialogs import classify_outlines, drawing_clearance, place_on_road
    polys = _car_outlines()
    roles = [o.role for o in classify_outlines(polys)]
    assert roles == ["wheel", "wheel", "separate", "main"]
    info = {o.index: o for o in classify_outlines(polys)}
    assert not info[2].include and info[0].include and info[3].include
    # a window inside the body is listed but cannot be included
    window = np.array([[1300, 950], [2400, 950], [2400, 1350], [1300, 1350]], float)
    inner = classify_outlines([*polys, window])[-1]
    assert inner.role == "inner" and not inner.enabled and not inner.include
    # the drawing's own body clearance, scaled to a 4.5 m car
    sel = [polys[0], polys[1], polys[3]]
    s = 4.5 / 4100
    assert drawing_clearance(sel, s) == pytest.approx(140 * s)
    assert drawing_clearance([polys[3]], s) is None
    # clearance is measured to the body; the wheels stand on the road plus the gap
    d = geo.design_from_polygons([p.copy() for p in sel], "car", "vehicle", length=4.5, ground_clearance=0.2)
    place_on_road(d, 0.10, 0.01)
    body = [b for b in d.bodies if b.role != "wheel"]
    wheels = [b for b in d.bodies if b.role == "wheel"]
    assert len(wheels) == 2 and len(body) == 1
    assert body[0].points[:, 1].min() == pytest.approx(0.10)
    for w in wheels:
        assert w.points[:, 1].min() == pytest.approx(0.01)
        assert w.center[1] == pytest.approx(0.01 + 310 * s, rel=1e-3)
    assert d.force_ref == pytest.approx(d.bounds()[3] - 0.01)


def test_dxf_import_dialog_outline_list_and_body_clearance(qapp, geo, tmp_path):
    ezdxf = pytest.importorskip("ezdxf")
    from aerolume.app.dialogs.import_dialogs import PolygonImportDialog, vehicle_height
    from aerolume.app.widgets.geometry_canvas import fmt_len
    doc = ezdxf.new()
    msp = doc.modelspace()
    for p in _car_outlines():
        msp.add_lwpolyline([tuple(map(float, q)) for q in p], close=True)
    path = tmp_path / "car.dxf"
    doc.saveas(path)
    dlg = PolygonImportDialog(str(path))
    assert dlg.kind() == "vehicle" and dlg.outline_list.isVisibleTo(dlg)
    infos = dlg.outline_infos()
    stray = next(o.index for o in infos if o.role == "separate")
    assert stray not in dlg.included_indices()
    d = dlg.design()
    assert d is not None
    assert d.source["excluded_outlines"] == [stray + 1]
    assert len(dlg.preview.ghosts) == 1                      # drawn dashed in the preview
    s = dlg.length.si_value() / 4100
    # the clearance follows the drawing: body 140 mm above wheels standing on the road
    assert dlg.clearance.si_value() == pytest.approx(140 * s, abs=1e-3)
    assert dlg.wheel_gap.isVisibleTo(dlg)
    body = next(b for b in d.bodies if b.role != "wheel")
    assert body.points[:, 1].min() == pytest.approx(dlg.clearance.si_value())
    assert all(b.points[:, 1].min() == pytest.approx(0.0, abs=1e-9) for b in d.bodies if b.role == "wheel")
    # a typed clearance moves the body, not the wheels
    dlg.clearance.set_si_value(0.10, emit=True)
    d = dlg.design()
    assert next(b for b in d.bodies if b.role != "wheel").points[:, 1].min() == pytest.approx(0.10)
    assert min(b.points[:, 1].min() for b in d.bodies if b.role == "wheel") == pytest.approx(0.0, abs=1e-9)
    # the footer reports the same length and height as the preview's dimension lines
    x0, x1, _, _ = d.bounds()
    assert f"{fmt_len(x1 - x0)} × {fmt_len(vehicle_height(d))} m" in dlg.status.text()
    h_without = vehicle_height(d)
    dlg.set_outline_included(stray, True)                    # the user includes the stray circle
    assert vehicle_height(dlg.design()) > h_without + 0.1
    assert "excluded_outlines" not in dlg.design().source
    for o in infos:
        dlg.set_outline_included(o.index, False)
    assert dlg.design() is None and dlg.status.level() == "error"
    assert not dlg.buttons.button(dlg.buttons.StandardButton.Ok).isEnabled()


def test_setup_estimate_and_planner_text_are_compact():
    from aerolume.app.pages.setup import compact_flow_model, compact_rows, planner_headline
    short, tip = compact_flow_model("LES (WALE sub-grid model), upwind3 advection")
    assert short == "LES-WALE"
    assert "WALE" in tip and "third-order upwind" in tip and "upwind3" not in tip
    assert compact_flow_model("Laminar (all scales resolved), central advection")[0] == "Laminar"
    assert compact_flow_model("something new") == ("something new", "")
    rows = compact_rows([("Cells", "131,072"), ("Flow model", "LES (WALE sub-grid model), upwind3 advection")])
    assert rows[0] == ("Cells", "131,072", "") and rows[1][1] == "LES-WALE"
    # one sentence per note: the decision and its number, the rationale in the tooltip
    note = ("Grid: 1,250 × 640 = 800,000 cells of 7.81 mm; 128 cells per chord (Standard quality on the GPU). "
            "Sizes are rounded to FFT-friendly numbers for the pressure solver.")
    assert planner_headline(note) == ("Grid: 1,250 × 640 = 800,000 cells of 7.81 mm; 128 cells per chord "
                                      "(Standard quality on the GPU).")
    assert planner_headline("Flow model: at Re = 1.03×10^6 a layer is 0.0049 chords thick, i.e. 0.6 cells. "
                            "More.") == "Flow model: at Re = 1.03×10⁶ a layer is 0.0049 chords thick, i.e. 0.6 cells."
    assert planner_headline("One sentence only.") == "One sentence only."


def test_setup_page_shows_compact_flow_model_and_one_line_notes(qapp, state, monkeypatch):
    from aerolume.app.pages import setup as setup_mod

    class _Plan:
        extent = (-5.0, 15.0, 0.0, 6.0)
        bodies_wind = None
        notes = ("Domain: 2.5 chords upstream (2.6 % blockage). Frictionless walls top and bottom.",)
        warnings = ()

        def summary(self):
            return {"Flow model": "LES (WALE sub-grid model), upwind3 advection", "Cells": "1"}

    loader = setup_mod._PlannerLoader.get()
    monkeypatch.setattr(loader, "state", "ready")
    monkeypatch.setattr(loader, "module", types.SimpleNamespace(plan_simulation=lambda d, s: _Plan()))
    state.add_design(_square_vehicle())
    page = setup_mod.SetupPage(state, None)
    page.on_shown()
    page.update_summary()
    assert page.estimate.value_text("Flow model") == "LES-WALE"
    assert "WALE" in page.estimate.tooltip("Flow model")
    lines = [page.planner_box.itemAt(i).widget() for i in range(page.planner_box.count())
             if isinstance(page.planner_box.itemAt(i).widget(), setup_mod.PlannerLine)]
    assert [ln.headline for ln in lines] == ["Domain: 2.5 chords upstream (2.6 % blockage)."]
    assert "Frictionless walls" in lines[0].text.toolTip()
    assert page.planner_lines[0][1] == _Plan.notes[0]        # the full text is what gets logged


def test_clearance_dimension_avoids_the_wheels():
    from aerolume.app.widgets.geometry_canvas import clearance_station
    floor = np.array([[0.0, 0.15], [4.5, 0.15], [4.5, 1.2], [0.0, 1.0]])     # flat floor, 4 vertices
    wheels = [_circle(0.8, 0.31, 0.31), _circle(3.6, 0.31, 0.31)]
    clear, x = clearance_station([floor], wheels)
    assert clear == pytest.approx(0.15)
    assert x == pytest.approx(0.62 * 4.5)                    # under the floor, clear of both wheels
    sloped = np.array([[0.0, 0.3], [4.0, 0.1], [4.5, 0.4], [4.5, 1.2], [0.0, 1.0]])
    clear, x = clearance_station([sloped])
    assert (clear, x) == (pytest.approx(0.1), pytest.approx(4.0))           # the lowest vertex itself


def test_clearance_label_never_sits_on_a_wheel():
    """The boxed "clearance" label goes to the first free slot: in the gap if it
    fits, else under the road; a wheel's bounding box rules a slot out."""
    from aerolume.app.widgets.geometry_canvas import clearance_label_slot
    view = (0.0, 0.0, 800.0, 500.0)
    size = (100.0, 16.0)
    rear_wheel = (360.0, 280.0, 420.0, 340.0)               # starts 60 px right of the leader

    def hits(r, o):
        return r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3]

    # gap too small (15 px): under the road, right of the leader - clear of the wheel
    r = clearance_label_slot(300.0, 340.0, 325.0, size, [rear_wheel], view)
    assert r[1] > 340.0 + 12.0 and r[0] > 300.0 and not hits(r, rear_wheel)
    # a tall gap: in the gap, but on the left because the wheel is on the right
    r = clearance_label_slot(300.0, 340.0, 300.0, size, [rear_wheel], view)
    assert r[3] <= 340.0 and r[2] <= 300.0
    # something below the road on the right too: left, under the road
    below = (305.0, 350.0, 500.0, 380.0)
    r = clearance_label_slot(300.0, 340.0, 325.0, size, [rear_wheel, below], view)
    assert r[2] <= 300.0 and r[1] > 340.0


def test_template_design_has_one_editor_per_quantity(qapp, state, fake_templates):
    """A template owns its size and its height above the road: Transform shows
    neither (they were a second 'Length' and a second clearance); an imported
    vehicle keeps both."""
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    page.add_design(FAKE_TEMPLATES.make("boxcar"))
    assert page.template_box.isVisibleTo(page)
    assert not page._scale_row.editor.isVisibleTo(page)
    assert not page._lift_row.editor.isVisibleTo(page)
    imported = _square_vehicle()
    imported.source = {"type": "import", "format": "dxf", "file": "car.dxf"}
    page.add_design(imported)
    assert page._scale_row.editor.isVisibleTo(page) and page._lift_row.editor.isVisibleTo(page)
    assert page._scale_row.label.text() == "Length"


def test_size_parameter_supersedes_older_scale_edits(fake_templates):
    """Typing the template's Length after a recorded 'scale to' must give that
    length, not replay the old scale; unrelated edits are kept."""
    from aerolume.app.pages.geometry import apply_edit, regenerate, supersede_edits
    edits = [{"op": "scale_to", "length": 3.0}, {"op": "translate", "dx": 1.0, "dy": 0.0},
             {"op": "lift", "height": 0.5}]
    assert [e["op"] for e in supersede_edits(edits, {"height"})] == ["scale_to", "translate", "lift"]
    assert [e["op"] for e in supersede_edits(edits, {"length"})] == ["translate", "lift"]
    assert [e["op"] for e in supersede_edits(edits, {"clearance"})] == ["scale_to", "translate"]
    d = FAKE_TEMPLATES.make("boxcar")
    for e in edits:
        apply_edit(d, e)
    d.source["edits"] = list(edits)
    out = regenerate(d, params={"length": 5.0})
    x0, x1, y0, _ = out.bounds()
    assert x1 - x0 == pytest.approx(5.0) and x0 == pytest.approx(1.0)
    assert y0 == pytest.approx(0.5)                          # the lift edit still applies
    assert [e["op"] for e in out.source["edits"]] == ["translate", "lift"]
    # an unchanged size value (the whole form re-sent) drops nothing
    same = regenerate(d, params={"length": 4.0, "height": 1.3})
    assert [e["op"] for e in same.source["edits"]] == ["scale_to", "translate", "lift"]


def test_estimate_values_are_split_short_and_readable():
    from aerolume.app.pages.setup import compact_device, compact_rows
    rows = {k: v for k, v, _t in compact_rows([
        ("Time step", "0.000882 L/U (138 µs)"), ("Duration", "30 L/U (4.70 s of flow)"),
        ("Stops early when", "Cd known to ±5 %, Cl to ±10 %"),
        ("Device", "GPU · NVIDIA GeForce RTX 4060 Ti (8.0 GB) · float32")])}
    assert rows == {"Time step": "0.000882 L/U", "Real time step": "138 µs", "Duration": "30 L/U",
                    "Real flow time": "4.70 s", "Stops early when": "Cd ±5 %, Cl ±10 %",
                    "Device": "RTX 4060 Ti (8.0 GB) · float32"}
    cpu = "CPU · Intel64 Family 6 Model 154, GenuineIntel · 8 solver threads · float64"
    assert compact_device(cpu) == "CPU · 8 threads · float64"
    assert compact_device("something else") == "something else"
    tips = {k: t for k, _v, t in compact_rows([("Device", "GPU · NVIDIA GeForce RTX 4060 Ti · float32")])}
    assert "NVIDIA" in tips["Device"]                        # the long form is the tooltip


def test_elided_label_wraps_to_two_lines_before_eliding(qapp):
    from aerolume.app.widgets.param_form import ElidedLabel, wrap_elided
    text = ("Grid: 720 × 450 = 324,000 cells of 37.3 mm; 126 cells per vehicle length "
            "(Draft quality on the GPU), chosen from the clearance and the Reynolds number.")
    lab = ElidedLabel(text, None, max_lines=2)
    lab.resize(260, 40)
    # A hidden top-level widget gets its resize event only when it is shown (Qt queues it
    # until the native window exists); once another test has created an OpenGL context
    # (a 3D view) the timing differs, so make the event happen as it does on screen.
    lab.show()
    qapp.processEvents()
    shown = lab.text().split("\n")
    assert len(shown) == 2 and shown[1].endswith("…")
    assert lab.toolTip() == text                             # elided: the whole note on hover
    line = lab.fontMetrics().lineSpacing()
    assert lab.heightForWidth(260) == 2 * line
    assert lab.heightForWidth(4000) == line                  # one line when wide enough
    assert wrap_elided("short", lab.font(), 400, 2) == ["short"]
    one = ElidedLabel(text)                                  # the default stays one line
    one.resize(200, 20)
    assert "\n" not in one.text() and one.text().endswith("…")


def test_wheels_row_shows_what_will_run(qapp, state):
    """No wheel bodies: a muted note, never a checked-but-disabled box. A fixed
    floor shows the box unchecked; the user's choice returns with the belt."""
    from aerolume.app.pages.setup import SetupPage
    state.add_design(_square_vehicle())
    page = SetupPage(state, None)
    page.on_shown()
    sim = state.simulation
    assert page.ground_wheels.isVisibleTo(page) and not page.no_wheels_label.isVisibleTo(page)
    assert page.ground_wheels.isEnabled() and page.ground_wheels.isChecked()
    page.ground_wheels.setChecked(False)                     # the user turns rotation off ...
    assert not sim.ground.rotate_wheels
    page.ground_combo.setCurrentIndex(page.ground_combo.findData("fixed"))
    assert not page.ground_wheels.isEnabled() and not page.ground_wheels.isChecked()
    page.ground_combo.setCurrentIndex(page.ground_combo.findData("moving"))
    assert page.ground_wheels.isEnabled() and not page.ground_wheels.isChecked()   # ... and it stays off
    page.ground_wheels.setChecked(True)
    page.ground_combo.setCurrentIndex(page.ground_combo.findData("fixed"))
    page.ground_combo.setCurrentIndex(page.ground_combo.findData("moving"))
    assert page.ground_wheels.isChecked() and sim.ground.rotate_wheels
    # a design without wheels
    d = _square_vehicle()
    d.bodies = [b for b in d.bodies if b.role != "wheel"]
    d.name = "no wheels"
    state.add_design(d, select=True)
    page.on_shown()
    assert state.design.id == d.id
    assert not page.ground_wheels.isVisibleTo(page) and page.no_wheels_label.isVisibleTo(page)
    assert page.no_wheels_label.text() == "No wheels in this design"
    assert not page.ground_wheels.isChecked() and not state.simulation.ground.rotate_wheels


def test_setup_info_strip_shows_whole_estimate_rows(qapp, state, monkeypatch):
    """The Estimate never ends in a half-clipped row: the strip is sized to the
    whole table when the preview keeps enough room, else to whole rows."""
    from aerolume.app.pages import setup as setup_mod

    class _Plan:
        extent = (-5.0, 15.0, 0.0, 6.0)
        bodies_wind = None
        notes = ()
        warnings = ()

        def summary(self):
            return {f"Row {i}": f"{i}" for i in range(18)}

    loader = setup_mod._PlannerLoader.get()
    monkeypatch.setattr(loader, "state", "ready")
    monkeypatch.setattr(loader, "module", types.SimpleNamespace(plan_simulation=lambda d, s: _Plan()))
    state.add_design(_square_vehicle())
    page = setup_mod.SetupPage(state, None)
    fitted = []
    for height in (1000, 560):
        page.resize(1200, height)
        page.show()
        page.on_shown()
        page.update_summary()
        _settle(qapp, 5)
        page.fit_info_height()
        n = len(page.estimate.items())
        chrome = page._info_head_h + 2 * page._info.frameWidth()
        body = page._vsplit.sizes()[1] - chrome
        whole = [page.estimate.height_for_rows(k) for k in range(1, n + 1)]
        assert body in whole or body == whole[-1] + 8, (height, body, whole)
        assert page._vsplit.sizes()[0] >= setup_mod.PREVIEW_MIN_PX
        fitted.append(body == whole[-1] + 8)
    assert fitted == [True, False]                           # all rows when tall, whole rows when not
    # a splitter the user dragged is left alone
    page._info_split_moved()
    page._vsplit.setSizes([300, 200])
    page.fit_info_height()
    assert page._vsplit.sizes()[1] == 200
    page.close()
