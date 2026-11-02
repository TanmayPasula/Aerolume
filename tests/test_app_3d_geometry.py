"""Geometry in 3D: the 3D edit helpers, the mesh inspector, the mesh import dialog,
the wing planform editor and the Geometry page's 3D workspace (offscreen Qt).

Everything runs on real :mod:`aerolume.geometry3d` designs made from the sample
files in ``examples/geometry3d`` (and a lofted wing); modal dialogs are bypassed
through the page's ``_exec_dialog`` funnel. The 3D views render with VTK
off-screen; where OpenGL is unavailable they show a placeholder, which these
tests tolerate.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from aerolume.model import Design, WingSection, WingSpec

pytestmark = pytest.mark.gui

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

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
    proj = Project.create(tmp_path / "Study3D", "Study 3D")
    st = AppState()
    st.set_project(proj)
    yield st


def _process(qapp, n: int = 5) -> None:
    for _ in range(n):
        qapp.processEvents()


def _vehicle() -> Design:
    """The two-part sample (body + wheel) as a vehicle design, as the importer makes it."""
    import aerolume.geometry3d as g
    meshes = g.read_mesh(EXAMPLES / "body_and_wheel.obj")
    return g.design_from_meshes(meshes, "Sample car", "vehicle",
                                source={"type": "import", "format": "obj", "file": "body_and_wheel.obj"})


def _sphere() -> Design:
    import aerolume.geometry3d as g
    return g.design_from_meshes(g.read_mesh(EXAMPLES / "sphere_d100mm_binary.stl"), "Sphere", "generic",
                                units="mm", source={"type": "import", "format": "stl",
                                                    "file": "sphere_d100mm_binary.stl"})


def _box_generic(on_road: bool = False) -> Design:
    import aerolume.geometry3d as g
    return g.design_from_meshes([g.box(2.0, 1.0, 0.5, name="Box")], "Box", "generic", on_road=on_road,
                                source={"type": "primitive", "shape": "box"})


def _wing_design(**kw) -> Design:
    import aerolume.geometry3d as g
    spec = WingSpec(sections=[WingSection(0.0, 0.3, airfoil="2412"),
                              WingSection(1.0, 0.18, x_le=0.03, airfoil="2412")], **kw)
    return g.design_from_wing(spec, "Wing")


# ---------------------------------------------------------------------------
# pure helpers: 3D edits and references
# ---------------------------------------------------------------------------

def test_clone_design_shares_mesh_arrays():
    from aerolume.app.widgets.mesh_inspector import clone_design
    d = _vehicle()
    c = clone_design(d)
    assert c is not d and c.id == d.id and len(c.meshes) == len(d.meshes)
    for a, b in zip(c.meshes, d.meshes, strict=True):
        assert a.vertices is b.vertices and a.faces is b.faces        # shared, never copied
        assert a is not b and a.role == b.role and a.center == b.center
    c.source["edits"] = ["x"]
    assert "edits" not in d.source                                      # the JSON part is a copy


def test_auto_references_follow_importer_conventions():
    from aerolume.app.widgets.mesh_inspector import auto_references3d
    d = _vehicle()
    auto = auto_references3d(d)
    assert auto["ref_area"] == pytest.approx(d.ref_area, rel=1e-6)
    assert auto["length_ref"] == pytest.approx(d.length_ref, rel=1e-9)
    assert auto["span_ref"] == pytest.approx(d.span_ref, rel=1e-9)
    assert auto["moment_ref3"] == pytest.approx(d.moment_ref3, abs=1e-9)
    s = _sphere()                                                       # compact body: L_ref = diameter
    assert auto_references3d(s)["length_ref"] == pytest.approx(0.1, rel=0.02)
    assert auto_references3d(s)["compact"]


def test_scale_edit_scales_references_and_keeps_typed_ones_consistent():
    from aerolume.app.widgets.mesh_inspector import edit_design3d, size3
    d = _vehicle()
    d.ref_area = 1.0                                                    # typed by the user
    d.source["manual_refs"] = ["ref_area"]
    L0 = size3(d)[0]
    after, info = edit_design3d(d, {"op": "scale_to", "length": 2 * L0})
    assert info["scale"] == pytest.approx(2.0)
    assert size3(after)[0] == pytest.approx(2 * L0)
    assert after.ref_area == pytest.approx(4.0)                         # typed area scales with k²
    assert after.length_ref == pytest.approx(2 * d.length_ref)          # auto length follows
    assert after.meshes[0].faces is d.meshes[0].faces                   # faces re-shared, not copied
    assert d.meshes[0].vertices is not after.meshes[0].vertices         # the input is untouched
    assert after.source["edits"][-1]["op"] == "scale_to"


def test_rotate_keeps_a_vehicle_on_the_road_and_nose_at_zero():
    from aerolume.app.widgets.mesh_inspector import _bbox, clearance, edit_design3d
    d = _vehicle()
    c0 = clearance(d)
    after, _ = edit_design3d(d, {"op": "rotate", "axis": "y", "deg": 3.0})
    lo, _hi = _bbox(after.meshes)
    assert lo[0] == pytest.approx(0.0, abs=1e-9)                        # nose back at x = 0
    assert lo[2] >= -1e-9                                               # nothing below the road
    assert clearance(after) == pytest.approx(max(c0, 0.0), abs=1e-6) or lo[2] == pytest.approx(0.0, abs=1e-9)


def test_orientation_preset_and_place_on_road_and_free():
    import aerolume.geometry3d as g
    from aerolume.app.widgets.mesh_inspector import _bbox, edit_design3d, on_road
    d = _box_generic()
    lo, _hi = _bbox(d.meshes)
    assert lo[2] == pytest.approx(-0.25) and not on_road(d)             # a free body is centred
    up, _ = edit_design3d(d, {"op": "orient", "preset": "x-up->z-up"})  # 2 m along x becomes the height
    lo2, hi2 = _bbox(up.meshes)
    assert hi2[2] - lo2[2] == pytest.approx(2.0, abs=1e-9)
    placed, _ = edit_design3d(d, {"op": "place_on_road", "clearance": 0.1})
    assert _bbox(placed.meshes)[0, 2] == pytest.approx(0.1) and on_road(placed)
    assert placed.moment_ref3[2] == pytest.approx(0.35)                 # generic: the box centre
    free, _ = edit_design3d(placed, {"op": "free"})
    assert not on_road(free) and _bbox(free.meshes)[:, 2].sum() == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        edit_design3d(d, {"op": "place_on_road", "clearance": -1.0})
    with pytest.raises(ValueError):
        edit_design3d(d, {"op": "spin"})
    assert g.ORIENT_PRESETS                                             # presets come from geometry3d


def test_mirror_centre_and_symmetry_flag():
    from aerolume.app.widgets.mesh_inspector import edit_design3d
    d = _box_generic()
    assert d.symmetric
    moved, info = edit_design3d(d, {"op": "translate", "dy": 0.3})
    assert info["symmetry"] and not moved.symmetric                     # re-measured: off the plane
    assert moved.source["symmetry_error"] > 0.01
    back, _ = edit_design3d(moved, {"op": "centre_y"})
    assert back.symmetric and back.source["symmetry_error"] < 1e-6
    # a typed flag is kept even when the measurement disagrees
    moved.source["manual_refs"] = ["symmetric"]
    moved.symmetric = True
    again, _ = edit_design3d(moved, {"op": "translate", "dy": 0.1})
    assert again.symmetric
    mirrored, _ = edit_design3d(d, {"op": "mirror", "axis": "x"})
    assert np.allclose(np.sort(mirrored.meshes[0].vertices[:, 0]), np.sort(d.meshes[0].vertices[:, 0]))


def test_role_and_kind_edits():
    from aerolume.app.widgets.mesh_inspector import _bbox, edit_design3d
    d = _vehicle()
    wheel = next(i for i, m in enumerate(d.meshes) if m.role == "wheel")
    body, _ = edit_design3d(d, {"op": "role", "index": wheel, "role": "body"})
    assert body.meshes[wheel].role == "body" and body.meshes[wheel].center is None
    back, _ = edit_design3d(body, {"op": "role", "index": wheel, "role": "wheel"})
    m = back.meshes[wheel]
    assert m.role == "wheel" and m.center is not None and abs(m.axis[1]) == pytest.approx(1.0)
    assert m.vertices is d.meshes[wheel].vertices                       # arrays shared through the edit
    generic = _box_generic()
    veh, _ = edit_design3d(generic, {"op": "kind", "kind": "vehicle"})
    lo, _hi = _bbox(veh.meshes)
    assert veh.kind == "vehicle" and lo[0] == pytest.approx(0.0) and lo[2] == pytest.approx(0.0, abs=1e-9)
    assert veh.moment_ref3[2] == pytest.approx(0.0)                     # vehicles: moments about the road


def test_bounds_cache_and_merged_winding_notes():
    import aerolume.geometry3d as g
    from aerolume.app.widgets.geometry_canvas import mesh_lohi, meshes_bounds
    from aerolume.app.widgets.mesh_inspector import classify_check, merge_checks
    s = g.sphere(1.0, n=24)
    lo, hi = mesh_lohi(s)
    assert mesh_lohi(s)[0] is lo                                         # cached per array
    moved = g.translate(s, (1.0, 0.0, 0.0))
    assert mesh_lohi(moved)[0][0] == pytest.approx(lo[0] + 1.0)         # new arrays, new bounds
    assert meshes_bounds([s, moved])[1] == pytest.approx(hi[0] + 1.0)
    assert meshes_bounds([]) is None
    notes = ["'body': 100 face(s) re-wound so the normals point outward.", "Mesh 'a' is not closed: 1 hole(s)",
             "'w1': 20 face(s) re-wound so the normals point outward."]
    merged = merge_checks(notes)
    assert merged[0] == "Face winding made consistent on 2 parts (120 faces re-wound outward)."
    assert len(merged) == 2 and classify_check(merged[0]) == "info" and classify_check(merged[1]) == "warning"
    assert classify_check("The vehicle extends 2 cm below the road (z < 0); raise it.") == "error"


def test_pick_dimension_edges_choose_silhouette_edges():
    from aerolume.app.widgets.geometry_canvas import pick_dimension_edges

    def side(pts):           # orthographic side view: x right, z up, 100 px per metre
        p = np.asarray(pts, float)
        return np.column_stack([100 * p[:, 0], -100 * p[:, 2]])

    edges = pick_dimension_edges((0, 4, -1, 1, 0, 1.5), side)
    axes = {e["axis"]: e for e in edges}
    assert set(axes) == {0, 2}                                          # the width is seen end-on
    assert axes[0]["length"] == pytest.approx(4.0)
    assert axes[0]["normal"][1] > 0.9                                   # the length goes below the model
    assert axes[2]["length"] == pytest.approx(1.5) and axes[2]["normal"][0] < -0.9   # the height left of it

    def iso(pts):
        p = np.asarray(pts, float)
        return np.column_stack([80 * (p[:, 0] - p[:, 1]), -40 * (p[:, 0] + p[:, 1]) - 90 * p[:, 2]])

    assert {e["axis"] for e in pick_dimension_edges((0, 4, -1, 1, 0, 1.5), iso)} == {0, 1, 2}
    assert pick_dimension_edges((0, 1, 0, 1, 0, 1), lambda p: np.full((8, 2), np.nan)) == []


# ---------------------------------------------------------------------------
# pure helpers: wings
# ---------------------------------------------------------------------------

def test_generate_planform_shapes_and_metrics():
    import aerolume.geometry3d as g
    from aerolume.app.widgets.wing_editor import generate_planform, generator_values
    rect = generate_planform("rectangular", span=2.0, root_chord=0.25, tip_chord=0.1)
    pf = g.planform(WingSpec(sections=rect))
    assert pf["area"] == pytest.approx(0.5) and pf["aspect_ratio"] == pytest.approx(8.0)
    tap = generate_planform("tapered", span=2.0, root_chord=0.3, tip_chord=0.15, sweep_c4_deg=10.0,
                            dihedral_deg=3.0, washout_deg=2.0, airfoil="4412")
    pf = g.planform(WingSpec(sections=tap))
    assert pf["sweep_c4_deg"] == pytest.approx(10.0, abs=1e-6)
    assert pf["dihedral_deg"] == pytest.approx(3.0, abs=1e-6)
    assert tap[-1].twist_deg == pytest.approx(-2.0) and tap[0].x_le == 0.0 and tap[-1].airfoil == "4412"
    ell = generate_planform("elliptic", span=2.0, root_chord=0.3, tip_chord=0.03, n_sections=11)
    pf = g.planform(WingSpec(sections=ell))
    assert len(ell) == 11 and pf["area"] == pytest.approx(math.pi * 1.0 * 0.3 / 2, rel=0.04)
    gv = generator_values(WingSpec(sections=tap))
    assert gv["shape"] == "swept" and gv["span"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        generate_planform("tapered", span=2.0, root_chord=0.3, tip_chord=0.0)


def test_panel_angles_insert_remove_sections():
    from aerolume.app.widgets.wing_editor import (
        insert_section,
        panel_angles,
        remove_section,
        set_panel_angle,
        validate_spec,
    )
    secs = [WingSection(0.0, 0.3), WingSection(0.5, 0.25, x_le=0.05), WingSection(1.0, 0.2, x_le=0.1)]
    ang = panel_angles(secs)
    assert ang[0] == (0.0, 0.0) and ang[1][0] == pytest.approx(math.degrees(math.atan(0.1)))
    swept = set_panel_angle(secs, 1, sweep_deg=30.0)
    shift = swept[1].x_le - secs[1].x_le
    assert panel_angles(swept)[1][0] == pytest.approx(30.0)
    assert swept[2].x_le - secs[2].x_le == pytest.approx(shift)          # outboard sections move along
    assert panel_angles(swept)[2] == pytest.approx(ang[2])               # and keep their own angle
    raised = set_panel_angle(secs, 2, dihedral_deg=5.0)
    assert panel_angles(raised)[2][1] == pytest.approx(5.0) and raised[1].z_le == 0.0
    with pytest.raises(ValueError):
        set_panel_angle(secs, 0, sweep_deg=10.0)
    mid = insert_section(secs, 0)
    assert len(mid) == 4 and mid[1].y == pytest.approx(0.25) and mid[1].chord == pytest.approx(0.275)
    tip = insert_section(secs, 2)
    assert tip[-1].y == pytest.approx(1.25) and tip[-1].chord == pytest.approx(0.175)
    assert validate_spec(WingSpec(sections=tip)) is None
    assert len(remove_section(secs, 1)) == 2
    with pytest.raises(ValueError):
        remove_section(secs[:2], 0)
    assert validate_spec(WingSpec(sections=[WingSection(0.5, 0.3), WingSection(0.2, 0.2)])) is not None


def test_airfoil_names_and_wing_references():
    from aerolume.app.widgets.wing_editor import (
        airfoil_label,
        parse_airfoil,
        refresh_wing_references,
        update_wing,
    )
    names = {"abc123": "My foil"}
    assert airfoil_label("2412") == "NACA 2412" and airfoil_label("design:abc123", names) == "My foil"
    assert parse_airfoil("NACA 4415") == "4415" and parse_airfoil("My foil", names) == "design:abc123"
    assert parse_airfoil("9999") is None and parse_airfoil("banana") is None
    d = _wing_design()
    d.ref_area = 1.0
    d.source["manual_refs"] = ["ref_area"]
    spec = WingSpec(sections=[WingSection(0.0, 0.4), WingSection(1.5, 0.2)])
    after = update_wing(d, spec)
    assert after.ref_area == 1.0                                         # typed: kept
    assert after.span_ref == pytest.approx(3.0)                          # auto: follows the planform
    assert after.length_ref == pytest.approx(after.force_ref)
    after.source["manual_refs"] = []
    refresh_wing_references(after)
    assert after.ref_area == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# the mesh import dialog
# ---------------------------------------------------------------------------

def test_mesh_import_dialog_units_orientation_and_parts(qapp):
    from aerolume.app.dialogs.mesh_import import MeshImportDialog, guess_kind, is_mesh_path
    assert is_mesh_path("a.STL") and is_mesh_path("b.obj") and not is_mesh_path("c.dat")
    dlg = MeshImportDialog(None, path=str(EXAMPLES / "sphere_d100mm_binary.stl"))
    try:
        assert not dlg.load_error
        assert dlg.units.currentData() == "mm"                           # the guess: a 100-unit sphere is mm
        assert "millimetres" in dlg.units_reason.text()
        assert dlg.kind.currentData() == "generic"
        assert dlg.size_label.text().startswith("0.100 × 0.100 × 0.100")
        designs = dlg.designs()
        assert len(designs) == 1 and designs[0].dimension == 3
        assert designs[0].source["scale_to_m"] == pytest.approx(0.001)
        assert designs[0].length_ref == pytest.approx(0.1, rel=0.02)
        dlg.units.setCurrentIndex(dlg.units.findData("m"))                # the preview follows the units
        assert dlg.size_label.text().startswith("100.000")
        assert dlg.status.level() == "warning"
    finally:
        dlg.done(0)
    dlg = MeshImportDialog(None, path=str(EXAMPLES / "box_and_cylinder_ascii.stl"))
    try:
        assert len(dlg._raw) == 2 and dlg.parts.isEnabled()
        dlg.parts.setCurrentIndex(dlg.parts.findData("each"))
        dlg.name.setText("Parts")
        designs = dlg.designs()
        assert len(designs) == 2 and all(d.name.startswith("Parts – ") for d in designs)
        dlg.orientation.setCurrentIndex(dlg.orientation.findData("x-up->z-up"))
        rotated = dlg.designs()
        assert rotated is not designs                                     # options changed: rebuilt
    finally:
        dlg.done(0)
    dlg = MeshImportDialog(None, path=str(EXAMPLES / "nothing_here.stl"))
    try:
        assert dlg.load_error and dlg.status.level() == "error"
        with pytest.raises(ValueError):
            dlg.designs()
    finally:
        dlg.done(0)
    import aerolume.geometry3d as g
    assert guess_kind(g.read_mesh(EXAMPLES / "body_and_wheel.obj")) in ("vehicle", "generic")
    assert guess_kind([], "my_car_v2") == "vehicle"


def test_import_mesh_dialog_adds_designs_through_state(qapp, state):
    from aerolume.app.dialogs.mesh_import import import_mesh_dialog
    seen = []

    def run(dlg):
        seen.append(dlg)
        dlg.kind.setCurrentIndex(dlg.kind.findData("vehicle"))
        return True

    added = import_mesh_dialog(None, state, str(EXAMPLES / "body_and_wheel.obj"), exec_dialog=run)
    assert added and len(state.project.designs) == 1
    d = state.design
    assert d is added[0] and d.kind == "vehicle" and d.dimension == 3 and d.source["file"] == "body_and_wheel.obj"
    assert import_mesh_dialog(None, state, str(EXAMPLES / "wedge_binary.ply"), exec_dialog=lambda dlg: False) is None
    assert len(state.project.designs) == 1


# ---------------------------------------------------------------------------
# the Geometry page in 3D
# ---------------------------------------------------------------------------

def _page(state, size=(1600, 1000)):
    from aerolume.app.pages.geometry import GeometryPage
    page = GeometryPage(state, None)
    page._exec_dialog = lambda dlg: True
    page.resize(*size)
    page.show()
    return page


def test_geometry_page_imports_a_mesh_and_shows_the_3d_workspace(qapp, state):
    page = _page(state)
    added = page.import_files([str(EXAMPLES / "body_and_wheel.obj")])
    _process(qapp)
    assert len(added) == 1 and state.design is added[0]
    assert page._viewport_stack.currentIndex() == 1 and page._content3d.currentIndex() == 0
    assert page._inspector_stack.currentIndex() == 2
    assert page.view3d.design() is state.design and page.view3d.bounds() is not None
    assert page.mesh_inspector.design() is state.design
    assert "triangles" in page.summary_label3d.text() and page.kind_label.text().endswith("3D")
    assert page.mesh_inspector.parts.rowCount() == len(state.design.meshes)
    assert page.mesh_inspector.wait_analysis(30)
    assert page.mesh_inspector.diag_summary.text()                        # checks arrived from the thread
    assert page.mesh_inspector.src_surface.text() not in ("", "…")
    for k in ("iso", "front", "side", "top"):
        page.view_buttons[k].click()
    page.edges_toggle.setChecked(True)
    assert page.view3d.edges()
    page.dims_toggle3d.setChecked(False)
    assert not page.view3d.show_dimensions
    assert not page.grab().isNull()
    # 2D designs still get the 2D canvas
    from aerolume.app.dialogs.import_dialogs import naca_design
    page.add_design(naca_design("2412", 1.0))
    assert page._viewport_stack.currentIndex() == 0 and page._inspector_stack.currentIndex() == 1


def test_geometry_page_3d_edits_undo_and_redo(qapp, state):
    from aerolume.app.widgets.mesh_inspector import size3
    page = _page(state)
    page.add_design(_vehicle())
    d0 = state.design
    L0, A0 = size3(d0)[0], d0.ref_area
    v0 = d0.meshes[0].vertices
    insp = page.mesh_inspector
    insp.scale_length.set_si_value(2 * L0, emit=True)                     # type a length, press Enter
    d1 = state.design
    assert size3(d1)[0] == pytest.approx(2 * L0) and d1.ref_area == pytest.approx(4 * A0, rel=1e-3)
    insp.rot_axis.setCurrentIndex(insp.rot_axis.findData("z"))
    insp.rot_deg.set_si_value(180.0)
    from PySide6.QtWidgets import QPushButton
    insp.findChild(QPushButton, "rotate_button3").click()
    assert page._undo_group.canUndo()
    page._undo_group.undo()
    assert size3(state.design)[0] == pytest.approx(2 * L0)
    page._undo_group.undo()
    d = state.design
    assert size3(d)[0] == pytest.approx(L0) and d.meshes[0].vertices is v0    # the original arrays back
    page._undo_group.redo()
    assert size3(state.design)[0] == pytest.approx(2 * L0)
    # references: typed value, then back to Auto
    insp.refEdited.emit("ref_area", 0.5)
    assert state.design.ref_area == 0.5 and "ref_area" in state.design.source["manual_refs"]
    insp.autoRefToggled.emit("ref_area", True)
    assert state.design.ref_area == pytest.approx(4 * A0, rel=1e-3)
    insp.autoRefToggled.emit("symmetric", False)
    insp.refEdited.emit("symmetric", False)
    assert state.design.symmetric is False
    insp.autoRefToggled.emit("symmetric", True)
    assert state.design.symmetric == bool(state.design.source["symmetry_error"] <= 0.005 * max(size3(state.design)))
    # a part's role from the Parts table
    wheel = next(i for i, m in enumerate(state.design.meshes) if m.role == "wheel")
    combo = insp.parts.cellWidget(wheel, 1)
    combo.setCurrentIndex(combo.findData("body"))
    assert state.design.meshes[wheel].role == "body"
    # the edit survives a save and a reopen with its arrays
    from aerolume.project import Project
    state.save()
    again = Project.open(state.project.path)
    dd = again.designs[state.design.id]
    assert np.allclose(dd.meshes[0].vertices, state.design.meshes[0].vertices)
    assert dd.meshes[wheel].role == "body"


def test_geometry_page_duplicate_3d_keeps_arrays_and_new_primitives(qapp, state):
    page = _page(state)
    page.add_design(_sphere())
    dup = page.duplicate_design()
    assert dup is not None and dup.id != state.project.designs[next(iter(state.project.designs))].id
    assert dup.meshes[0].vertices.shape[0] > 0
    for shape in ("sphere", "box", "cylinder"):
        d = page.new_primitive(shape)
        assert d is not None and d.dimension == 3 and d.kind == "generic" and d.source["type"] == "primitive"
    assert page._viewport_stack.currentIndex() == 1 and page._inspector_stack.currentIndex() == 2
    assert "3D mesh (STL, OBJ, PLY)…" in page.import_actions
    assert set(page.new3d_actions) == {"wing", "sphere", "box", "cylinder"}


def test_geometry_page_drop_mesh_on_the_3d_view(qapp, state):
    page = _page(state)
    page.add_design(_sphere())
    _process(qapp)
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(EXAMPLES / "wedge_binary.ply"))])
    ev = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, md, Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier)
    page.view3d.dropEvent(ev)
    _process(qapp)
    assert len(state.project.designs) == 2
    assert state.design.source.get("file") == "wedge_binary.ply"
    # a drop anywhere on the page (e.g. over the inspector) imports too
    page.dropEvent(QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, md, Qt.MouseButton.NoButton,
                              Qt.KeyboardModifier.NoModifier))
    assert len(state.project.designs) == 3


def test_geometry_page_wing_editor(qapp, state):
    page = _page(state)
    d = page.new_wing()
    _process(qapp)
    assert d is not None and d.kind == "wing" and state.design is d
    assert page._content3d.currentIndex() == 1 and page._inspector_stack.currentIndex() == 3
    ed, insp = page.wing_editor, page.wing_inspector
    assert ed.table.rowCount() == len(d.wing.sections) == 2
    assert ed.view3d.meshes(), "the planform is lofted for the preview"
    rows = dict(insp.metrics.items())
    assert rows["Aspect ratio"] == f"{d.span_ref ** 2 / d.ref_area:.2f}"
    # edit the tip chord in the table (as typed: "150 mm")
    item = ed.table.item(1, 1)
    item.setText("150 mm")
    _process(qapp)
    assert state.design.wing.sections[1].chord == pytest.approx(0.15)
    assert state.design.ref_area == pytest.approx(0.45)                  # references follow the planform
    # sweep in angle mode moves the tip
    ed.le_mode.buttons["angles"].click()
    ed.le_mode.currentChanged.emit("angles")
    assert ed.table.angles and ed.table.item(0, 2).text() == "—"
    ed.table.item(1, 2).setText("20")
    _process(qapp)
    assert math.degrees(math.atan2(state.design.wing.sections[1].x_le, 1.0)) == pytest.approx(20.0)
    # an invalid value is refused with a reason and nothing changes
    before = state.design.wing.to_dict()
    ed.table.item(1, 1).setText("-3")
    _process(qapp)
    assert state.design.wing.to_dict() == before and ed.status.level() == "error"
    # airfoil by name
    ed.table.item(0, 5).setText("NACA 4412")
    _process(qapp)
    assert state.design.wing.sections[0].airfoil == "4412"
    # add / insert / remove
    ed.add_button.click()
    assert len(state.design.wing.sections) == 3
    ed.select_section(0)
    ed.insert_button.click()
    assert len(state.design.wing.sections) == 4
    ed.select_section(3)
    ed.remove_button.click()
    assert len(state.design.wing.sections) == 3
    # the inspector: generator, options, references
    insp.shape.setCurrentIndex(insp.shape.findData("elliptic"))
    insp.gen_span.set_si_value(3.0)
    insp.gen_root.set_si_value(0.3)
    insp.generate_button.click()
    w = state.design.wing
    assert len(w.sections) >= 5 and state.design.span_ref == pytest.approx(3.0)
    insp.n_span.setValue(32)
    assert state.design.wing.n_span == 32
    insp.symmetric.setChecked(False)
    assert state.design.wing.symmetric is False and state.design.symmetric is False
    insp.refEdited.emit("ref_area", 2.0)
    assert state.design.ref_area == 2.0
    insp.autoRefToggled.emit("ref_area", True)
    assert state.design.ref_area != 2.0
    # undo walks back through the planform edits
    n = page._stack_for(d.id).count()
    assert n >= 8
    for _ in range(n):
        page._undo_group.undo()
    assert state.design.wing.to_dict() == d.wing.to_dict()
    assert not page.grab().isNull()


def test_wing_editor_uses_project_airfoils(qapp, state):
    from aerolume.app.dialogs.import_dialogs import naca_design
    from aerolume.app.widgets.wing_editor import airfoil_resolver, project_airfoils
    page = _page(state)
    foil = naca_design("4415", 1.0, name="Root foil")
    page.add_design(foil)
    wing = page.new_wing()
    names = project_airfoils(state.project)
    assert names == {foil.id: "Root foil"}
    resolve = airfoil_resolver(state.project)
    assert resolve(f"design:{foil.id}").shape[1] == 2
    with pytest.raises(ValueError):
        resolve("design:missing")
    page.wing_editor.table.item(0, 5).setText("Root foil")
    _process(qapp)
    assert state.design.wing.sections[0].airfoil == f"design:{foil.id}"
    assert page.wing_editor.table.item(0, 5).text() == "Root foil"
    assert page.wing_editor.view3d.meshes()                             # lofted through the resolver
    assert state.design.id == wing.id


def test_geometry_page_3d_screens_light_and_dark(qapp, state, tmp_path):
    """Both themes render the 3D workspace (VTK colours follow the tokens)."""
    from aerolume.app import theme
    page = _page(state, (1280, 800))
    page.add_design(_vehicle())
    _process(qapp)
    out = []
    try:
        for name in ("dark", "light"):
            theme.apply(QApplication.instance(), name)
            _process(qapp, 10)
            img = page.grab().toImage()
            out.append(img.pixelColor(img.width() // 2, img.height() // 2).lightness())
    finally:
        theme.apply(QApplication.instance(), "light")
    assert len(out) == 2


def test_wing_editor_sizes_the_table_to_its_rows(qapp, state):
    """At 1280 x 800 a two-section wing must not leave the 3D preview a strip above
    a table of empty rows; once the user drags a handle the sizes are theirs."""
    page = _page(state, (1280, 800))
    page.new_wing()
    _process(qapp, 10)
    ed = page.wing_editor
    view_h, sketch_h, table_h = ed._split.sizes()
    rows_h = (ed.table.rowCount() + 1) * ed.table.verticalHeader().defaultSectionSize()
    assert table_h <= ed.TABLE_MAX_SHARE * sum(ed._split.sizes()) + 1
    assert table_h < rows_h + 120 and view_h > table_h and view_h > sketch_h
    ed._split.splitterMoved.emit(100, 1)                     # the user takes over
    ed._split.setSizes([200, 200, 300])
    ed.fit_panes()
    assert ed._split.sizes()[2] == 300
