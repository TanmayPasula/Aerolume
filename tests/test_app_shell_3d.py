"""Tests for the 3D side of the desktop shell (DESIGN.md 7.11, docs/ui_guidelines.md).

The shell has to make 3D visible without a mode switch: the Start page lists
3D and wing studies next to the 2D ones, the New study dialog asks for a mesh
file, File ▸ Import geometry splits into a 2D outline and a 3D mesh, the
Project tree tells 2D outlines, 3D meshes and wings apart (glyph and tag) and
labels each simulation's analysis, the step bar's breadcrumb carries a "3D" /
"Wing" tag, the Simulation menu follows the selected design, and the status
bar shows cells and steps/s of a running job.

The mesh import dialog belongs to the Geometry page's owner; it is reached
through ``main_window.MESH_IMPORT_HOOK``, so these tests replace it with fakes
and also check the fallback used when a build has no dialog. Everything runs
offscreen with private settings (the ``isolated`` fixture of
``test_app_shell``), so the user's preferences and recent list are untouched.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import test_app_shell as shell
from PySide6.QtCore import Signal
from test_app_shell import DummyPage, FakeJobs, _pump, _window, needs_store

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


@pytest.fixture()
def isolated(tmp_path, monkeypatch, qapp):
    """Private settings + recent list, light theme, a projects folder (as in test_app_shell)."""
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    projects = tmp_path / "projects"
    projects.mkdir()
    yield projects
    shell._destroy_windows(qapp)
    settings.use_file(tmp_path / "settings-after.ini")

REPO = Path(__file__).resolve().parents[1]
SPHERE_STL = REPO / "examples" / "geometry3d" / "sphere_d100mm_binary.stl"


class FakeJobs3D(FakeJobs):
    """FakeJobs plus the plan / frame signals the status bar reads."""

    jobPlanned = Signal(str, object)


def _box_design(name: str = "Box body", kind: str = "vehicle"):
    from aerolume import geometry3d as g3
    return g3.design_from_meshes([g3.box(1.0, 0.4, 0.3)], name, kind=kind)


def _wing_design(name: str = "Main wing"):
    from aerolume.app.pages.start import default_wing_spec
    from aerolume.geometry3d import design_from_wing
    return design_from_wing(default_wing_spec(), name)


def _plate_design(name: str = "Plate"):
    import numpy as np

    from aerolume.model import Body, Design
    sq = np.array([[0, 0], [1, 0], [1, 0.2], [0, 0.2]], float)
    return Design(name, "generic", [Body(name.lower(), sq)], 1.0, 1.0)


def _menu_texts(menu) -> list[str]:
    return [a.text().replace("&", "") for a in menu.actions() if a.text() and a.isVisible()]


# ---------------------------------------------------------------------------
# words and icons
# ---------------------------------------------------------------------------

def test_designs_and_analyses_are_named_the_same_everywhere():
    from aerolume.app import state as st
    from aerolume.model import Simulation
    plate, box, wing = _plate_design(), _box_design(), _wing_design()
    assert [st.design_tag(d) for d in (plate, box, wing, None)] == ["2D", "3D", "Wing", ""]
    assert [st.design_icon(d) for d in (plate, box, wing)] == ["layers", "cube", "wing"]
    assert st.design_icon(_box_design(kind="generic")) == "cube"
    assert st.is_3d(box) and st.is_3d(wing) and not st.is_3d(plate)
    assert st.is_wing(wing) and not st.is_wing(box)
    assert "3D mesh" in st.design_description(box) and "12 triangles" in st.design_description(box)
    assert st.design_description(wing).startswith("Wing · planform · ")
    assert st.design_description(wing).endswith("span 2 m")
    assert st.design_description(plate) == "Body · 2D outline"
    cfd, polar, wsim = (Simulation("a", "x"), Simulation("b", "x", analysis="polar"),
                        Simulation("c", "x", analysis="wing"))
    assert [st.analysis_label(cfd, plate), st.analysis_label(cfd, box), st.analysis_label(cfd, wing)] == \
        ["CFD 2D", "CFD 3D", "CFD 3D"]
    assert st.analysis_label(polar) == "Polar" and st.analysis_label(wsim, wing) == "Wing"
    assert st.analysis_label(cfd) == "CFD"
    assert st.analysis_description(cfd, box) == "CFD (Navier–Stokes, 3D)"
    assert st.analysis_description(wsim) == "Wing (vortex lattice, viscous)"
    wsim.wing.method, wsim.wing.viscous = "llt", False
    assert st.analysis_description(wsim) == "Wing (lifting line, inviscid)"
    assert st.analyses_for(wing) == ["wing", "cfd"] and st.analyses_for(box) == ["cfd"]


def test_new_icons_belong_to_the_set(qapp):
    from aerolume.app import theme
    for name in ("cube", "mesh", "wing", "slice"):
        assert name in theme.icon_names(), name
        img = theme.render_icon(name, 16, "text", 2.0).toImage()
        assert any(img.pixelColor(x, y).alpha() > 0 for x in range(32) for y in range(32)), name


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------

@needs_store
def test_wing_study_starts_with_a_wing_polar(isolated):
    from aerolume.app import settings
    from aerolume.app.pages import start
    from aerolume.project import Project
    settings.get().device = "cpu"
    proj, msgs = start.create_study("wing", "Glider", isolated)
    assert not msgs
    (d,) = proj.designs.values()
    assert d.kind == "wing" and d.dimension == 3 and d.wing is not None and d.symmetric
    assert d.span_ref == pytest.approx(2.0) and 0.3 < d.ref_area < 0.6       # a 2 m tapered wing
    (sim,) = proj.simulations.values()
    assert sim.analysis == "wing" and sim.name == "Wing polar · 30 m/s"
    assert sim.wing.method == "vlm" and sim.wing.viscous and sim.conditions.speed == 30.0
    assert sim.cfd.device == "cpu"
    again = Project.open(start.project_path(proj))           # a 3D design survives the round trip
    assert [s.chord for s in next(iter(again.designs.values())).wing.sections] ==         [s.chord for s in d.wing.sections]
    s = start.summarize(start.project_path(proj))
    assert s.tags == {"Wing"} and s.type_text == "Wing" and s.icon == "wing"


@needs_store
def test_mesh_study_gives_each_imported_design_a_3d_run(isolated):
    from aerolume.app.pages import start
    car, body = _box_design("Car"), _box_design("Pod", kind="generic")
    proj, msgs = start.create_study("mesh", "Bodies", isolated, designs=[car, body])
    assert start.summarize(start.project_path(proj)).type_text == "3D"     # a car and a pod: no one kind
    assert not msgs and set(proj.designs) == {car.id, body.id}
    sims = {s.design_id: s for s in proj.simulations.values()}
    assert sims[car.id].analysis == "cfd" and sims[car.id].name == "CFD 3D · 30 m/s"
    assert sims[car.id].ground.enabled and not sims[body.id].ground.enabled   # vehicles sit on a road
    assert sims[car.id].cfd.quality == "draft"
    cars, _ = start.create_study("mesh", "Cars", isolated, designs=[_box_design("A"), _box_design("B")])
    s = start.summarize(start.project_path(cars))
    assert s.tags == {"3D"} and s.icon == "cube" and s.type_text == "Vehicle (3D)"
    empty, _ = start.create_study("mesh", "Nothing yet", isolated)
    assert not empty.designs and start.summarize(start.project_path(empty)).type_text == "Empty"


@needs_store
def test_summary_types_for_2d_and_mixed_projects(isolated):
    from aerolume.app.pages import start
    proj, _ = start.create_study("airfoil", "Section", isolated)
    s = start.summarize(start.project_path(proj))
    assert s.type_text == "Airfoil (2D)" and s.icon == "airfoil"
    proj.add_design(_box_design())
    proj.save()
    (start.project_path(proj) / "project.json").touch()       # summaries are cached per mtime
    s = start.summarize(start.project_path(proj))
    assert s.tags == {"2D", "3D"} and s.type_text == "2D and 3D" and s.icon == "cube"


def test_new_study_dialog_offers_every_study_type(isolated, qapp, tmp_path):
    from aerolume.app import settings
    from aerolume.app.pages.start import STUDY_TYPES, NewStudyDialog
    settings.get().last_dir = tmp_path
    dlg = NewStudyDialog("wing")
    assert dlg.kind.keys() == ["airfoil", "vehicle", "mesh", "wing", "room", "enclosure", "blank"] == [
        t.key for t in STUDY_TYPES]
    assert dlg.kind.current() == "wing" and dlg.name.text() == "Wing study"
    assert "vortex-lattice" in dlg.hint.text() and dlg.type_title.text() == "Wing (3D)"
    assert not dlg._mesh_row.isVisibleTo(dlg) and dlg.ok.text() == "Create"
    dlg.kind.set_current("mesh")
    assert dlg._mesh_row.isVisibleTo(dlg) and dlg.name.text() == "3D study"
    assert dlg.ok.text() == "Create and import…"
    dlg._accept()                                             # no mesh file yet
    assert dlg.error.isVisibleTo(dlg) and "mesh file" in dlg.error.text()
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    dlg.mesh.setText(str(bad))
    dlg._accept()
    assert "STL, OBJ or PLY" in dlg.error.text()
    dlg.mesh.setText(str(SPHERE_STL))                         # the name follows the file
    assert dlg.name.text() == "Sphere d100mm binary"
    assert dlg.mesh_path() == str(SPHERE_STL)
    dlg._accept()
    assert dlg.result() == dlg.DialogCode.Accepted
    assert dlg.values()[0] == "mesh"
    dlg.kind.set_current("airfoil")                           # a suggested name follows the type
    assert dlg.name.text() == "Airfoil study" and dlg.mesh_path() == ""
    dlg.name.setText("My own name")
    dlg.kind.set_current("vehicle")
    assert dlg.name.text() == "My own name"                   # a typed name is never replaced
    dlg.close()


@needs_store
def test_start_page_lists_3d_studies(isolated, qapp):
    from aerolume.app.pages import start
    page = start.StartPage(None)
    asked = []
    page.newStudyRequested.connect(asked.append)
    for link in (page.link_airfoil, page.link_vehicle, page.link_mesh, page.link_wing):
        link.click()
    page.handle_action("new-wing")
    page.handle_action("new-blank")
    page.handle_action("new-nonsense")
    assert asked == ["airfoil", "vehicle", "mesh", "wing", "wing", "blank"]
    assert "3D" in page.version.text()
    start.create_study("wing", "Wing one", isolated)
    page.refresh()
    item = page.recent.topLevelItem(0)
    assert item.text(0) == "Wing one" and item.text(1) == "Wing"
    assert page.recent.headerItem().text(1) == "Type"
    page.close()


def test_start_lists_refit_when_the_style_changes(isolated, qapp):
    """The lists are fitted to their rows once, at refresh. Fitted before the style sheet arrived (a
    window built before the theme, as the first screenshot run did), the examples list kept its
    unstyled 106 px when the sheet's header and row padding needed 117, and hid its last row behind a
    scroll bar (measured at 1600 x 1000, light). They now refit after a theme or font change."""
    from aerolume.app import theme
    from aerolume.app.pages import start

    def pump():
        for _ in range(6):
            qapp.processEvents()

    def all_rows_fit():
        out = []
        for lst in (page.recent, page.examples):
            n = lst.topLevelItemCount()
            if n and lst.isVisible():
                last = lst.visualItemRect(lst.topLevelItem(min(n, 8) - 1))
                out.append(last.bottom() <= lst.viewport().height())
        return out

    for name in ("one", "two", "three"):
        start.create_study("airfoil", f"Recent {name}", isolated)
    qapp.setStyleSheet("")                       # no theme yet
    try:
        page = start.StartPage(None)
        page.resize(1600, 1000)
        page.refresh()
        page.show()
        pump()
        theme.apply(qapp, "light")               # the sheet arrives: taller header and rows
        pump()
        assert all_rows_fit() and all(all_rows_fit())
        theme.apply(qapp, "dark")
        pump()
        assert all(all_rows_fit())
    finally:
        theme.apply(qapp, "light")
        page.close()


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

def _fake_hook(designs, calls):
    def import_mesh_dialog(parent, state, *, path=None):
        calls.append((parent, state, path))
        return designs
    return import_mesh_dialog


@needs_store
def test_new_mesh_study_runs_the_import_hook(isolated, qapp, monkeypatch):
    from aerolume.app import main_window as mw
    from aerolume.app.widgets.project_tree import TAG_ROLE
    box = _box_design("Ahmed body")
    calls = []
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: _fake_hook([box], calls))
    w = _window(jobs=FakeJobs3D())
    w.resize(1600, 1000)
    w.show()
    proj = w.new_study("mesh", "Ahmed", isolated, mesh_path=str(SPHERE_STL))
    _pump(qapp)
    assert calls and calls[0][0] is w and calls[0][1] is w.state and calls[0][2] == str(SPHERE_STL)
    assert box.id in proj.designs
    (sim,) = proj.simulations.values()
    assert sim.design_id == box.id and sim.analysis == "cfd" and sim.ground.enabled
    assert w.state.design_id == box.id and w.state.simulation_id == sim.id
    assert w.current_page_key() == "geometry"
    tree = w.project_tree
    # the design row carries the dimension; the simulation row only its name (its analysis is in the tooltip)
    assert tree.item(box.id).data(TAG_ROLE) == "3D" and tree.item(sim.id).data(TAG_ROLE) == ""
    assert tree.item(sim.id).toolTip().splitlines()[1].startswith("CFD 3D")
    assert w.step_bar.tag() == "3D" and "Ahmed body" in w.step_bar.context()
    assert w.step_bar.full_context().startswith("3D · Ahmed body")
    assert any("Imported “Ahmed body”" in text for *_x, text in
               (w.messages.row(r) for r in range(w.messages.row_count())))
    # a hook that adds the design itself is not added twice; a cancelled import adds nothing
    calls.clear()
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: _fake_hook(None, calls))
    assert w.import_mesh() == [] and calls and calls[0][2] is None
    n = len(proj.designs)
    other = _box_design("Second")

    def adding_hook(parent, state):                          # an older signature: no path keyword
        state.add_design(other, select=False)
        return [other]
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: adding_hook)
    assert w.import_mesh("ignored.stl") == [other] and len(proj.designs) == n + 1
    assert not proj.simulations_of(other.id)                 # plain imports add no simulation
    # the real dialog's signature: it adds through the shell's adder, which keeps names unique
    same = _box_design("Ahmed body")

    def real_like(parent, state, path=None, *, exec_dialog=None, add=None):
        (add or state.add_design)(same)
        return [same]
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: real_like)
    assert w.import_mesh() == [same] and same.name == "Ahmed body 2" and same.id in proj.designs
    assert w.state.design_id == same.id
    w.close()


@needs_store
def test_mesh_import_without_a_dialog_still_works(isolated, qapp, monkeypatch):
    from aerolume.app import main_window as mw
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: None)
    w = _window()
    proj = w.new_study("mesh", "Sphere", isolated, mesh_path=str(SPHERE_STL))
    (d,) = proj.designs.values()
    assert d.dimension == 3 and d.kind == "generic" and len(d.meshes) == 1
    lo = d.bounds3()
    assert 0.09 < lo[1] - lo[0] < 0.11                        # 100 mm sphere: millimetres guessed
    logged = [w.messages.row(r) for r in range(w.messages.row_count())]
    assert any(lv == "warning" and "dialog is not part of this build" in text for _t, lv, _s, text in logged)
    # nothing chosen: nothing added, the study stays valid
    monkeypatch.setattr(w, "_ask_mesh_file", lambda: "")
    assert w.import_mesh() == []
    w.close()


@needs_store
def test_a_broken_import_dialog_is_reported(isolated, qapp, monkeypatch):
    from aerolume.app import main_window as mw

    def broken():
        raise ImportError("vtk is missing")
    monkeypatch.setattr(mw, "mesh_import_hook", broken)
    w = _window()
    w.new_study("blank", "Broken", isolated)
    assert w.import_mesh("x.stl") == []
    assert w.messages.counts()["error"] >= 1

    def raising(parent, state, path=None):
        raise ValueError("no triangles in the input")
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: raising)
    assert w.import_mesh("x.stl") == []
    logged = [w.messages.row(r) for r in range(w.messages.row_count())]
    assert any("no triangles" in text for *_x, text in logged)
    w.close()


def test_hook_lookup_tells_missing_from_broken(monkeypatch):
    import importlib

    from aerolume.app import main_window as mw
    real = importlib.import_module

    def missing(name, *a, **k):
        if name == "aerolume.app.dialogs.mesh_import":
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real(name, *a, **k)
    monkeypatch.setattr(mw.importlib, "import_module", missing)
    # falls through to the alternative module, which may or may not define the function
    expected = getattr(real("aerolume.app.dialogs.import_dialogs"), "import_mesh_dialog", None)
    assert mw.mesh_import_hook() is expected

    alt_mod = SimpleNamespace(import_mesh_dialog=lambda parent, state: [])

    def alternative(name, *a, **k):
        if name == "aerolume.app.dialogs.import_dialogs":
            return alt_mod
        return missing(name, *a, **k)
    monkeypatch.setattr(mw.importlib, "import_module", alternative)
    assert mw.mesh_import_hook() is alt_mod.import_mesh_dialog

    def dep_missing(name, *a, **k):
        if name == "aerolume.app.dialogs.mesh_import":
            raise ModuleNotFoundError("No module named 'vtkmodules'", name="vtkmodules")
        return real(name, *a, **k)
    monkeypatch.setattr(mw.importlib, "import_module", dep_missing)
    with pytest.raises(ModuleNotFoundError):
        mw.mesh_import_hook()
    assert mw._accepts_keyword(lambda parent, state, *, path=None: None, "path")
    assert mw._accepts_keyword(lambda parent, state, **kw: None, "path")
    assert not mw._accepts_keyword(lambda parent, state: None, "path")


@needs_store
def test_tree_and_menus_follow_the_kind_of_design(isolated, qapp):
    from PySide6.QtCore import QModelIndex

    from aerolume.app.widgets.project_tree import TAG_ROLE
    from aerolume.model import Simulation
    w = _window(jobs=FakeJobs3D())
    w.resize(1600, 1000)
    w.show()
    proj = w.new_study("blank", "Kinds", isolated)
    w._pages["setup"] = setup_page = DummyPage()
    w.stack.addWidget(setup_page)
    plate, box, wing = _plate_design(), _box_design(), _wing_design()
    for d in (plate, box, wing):
        w.state.add_design(d)
    tree = w.project_tree
    assert [tree.item(d.id).data(TAG_ROLE) for d in (plate, box, wing)] == ["", "3D", "Wing"]
    assert "3D mesh" in tree.item(box.id).toolTip()
    # context menus by kind
    texts = [a.text() for a in tree.build_menu(tree.item(plate.id).index()).actions() if a.text()]
    assert texts[0] == "New simulation"
    texts = [a.text() for a in tree.build_menu(tree.item(box.id).index()).actions() if a.text()]
    assert texts[:2] == ["New 3D CFD", "Open in Geometry"]
    menu = tree.build_menu(tree.item(wing.id).index())
    texts = [a.text() for a in menu.actions() if a.text()]
    assert texts[:3] == ["New wing analysis", "New 3D CFD", "Open in Geometry"]
    # "New 3D CFD" of a wing: a CFD simulation of the wing, at the speed of its latest one, then Setup
    w.state.add_simulation(Simulation("Polar", wing.id, analysis="wing"))
    proj.simulations_of(wing.id)[0].conditions.speed = 25.0
    next(a for a in menu.actions() if a.text() == "New 3D CFD").trigger()
    cfd = proj.simulations_of(wing.id)[-1]
    assert cfd.analysis == "cfd" and cfd.name == "CFD 3D · 25 m/s" and cfd.conditions.speed == 25.0
    assert w.state.simulation_id == cfd.id and w.current_page_key() == "setup"
    assert tree.item(cfd.id).data(TAG_ROLE) == "" and "CFD 3D" in tree.item(cfd.id).toolTip()
    from PySide6.QtCore import Qt
    # a name elides at its end ("CFD 3D · Re 100 · α 0°" -> "CFD 3D · Re 100…"): it reads from its
    # start, and the tooltip carries the whole name
    assert tree.view.textElideMode() == Qt.TextElideMode.ElideRight
    assert w.step_bar.tag() == "Wing"
    # the empty area offers both importers and a new wing
    texts = [a.text() for a in tree.build_menu(QModelIndex()).actions() if a.text()]
    assert texts == ["Import 2D outline…", "Import 3D mesh…", "New design from template…", "New wing"]
    # the Simulation menu follows the selection
    w._sync_simulation_actions()
    assert w.act_new_wing_sim.isVisible() and w.act_new_cfd3d.isVisible()
    assert not w.act_new_sim.isVisible() and "wing" in w.act_new_cfd3d.text()
    assert w.act_explore.isEnabled()                        # a CFD run is selected
    w.state.select_simulation(proj.simulations_of(wing.id)[0].id)
    assert not w.act_explore.isEnabled() and "no live flow" in w.act_explore.toolTip()
    w.act_new_wing_sim.trigger()
    assert proj.simulations_of(wing.id)[-1].analysis == "wing"
    w.state.select_design(box.id)
    assert w.act_new_cfd3d.isVisible() and not w.act_new_wing_sim.isVisible()
    assert w.act_new_cfd3d.text().replace("&", "") == "New 3D CFD"
    w.state.select_design(plate.id)
    assert w.act_new_sim.isVisible() and not w.act_new_cfd3d.isVisible()
    assert w.step_bar.tag() == "" and w.step_bar.full_context() == "Plate"
    plate.name = "Plate X"                                  # a rename reaches the breadcrumb at once
    w.state.update_design(plate)
    assert w.step_bar.context() == "Plate X"
    # duplicating a 3D body copies its mesh arrays (to_dict leaves them out), and they are saved
    from aerolume.app.pages import start
    from aerolume.project import Project
    dup = tree.duplicate(box.id)
    copy_ = proj.designs[dup]
    assert copy_.dimension == 3 and len(copy_.meshes[0].vertices) == len(box.meshes[0].vertices) > 0
    assert copy_.meshes[0].vertices is not box.meshes[0].vertices
    again = Project.open(start.project_path(proj))
    assert len(again.designs[dup].meshes[0].faces) == len(box.meshes[0].faces)
    assert tree.duplicate(wing.id) and len(proj.designs) == 5
    # an analysis the design cannot have is refused, not created
    with pytest.raises(ValueError):
        w.state.new_simulation(box.id, "wing")
    with pytest.raises(ValueError):
        w.state.new_simulation(box.id, "polar")
    w.close()


@needs_store
def test_commands_and_import_menus(isolated, qapp, monkeypatch):
    from aerolume.app import main_window as mw
    w = _window()
    assert w.act_import.shortcut().toString() == "Ctrl+I"
    assert w.act_import_mesh.shortcut().toString() == "Ctrl+Shift+I"
    assert _menu_texts(w.import_menu) == ["2D outline…", "3D mesh (STL, OBJ, PLY)…"]
    file_menu = w.menuBar().actions()[0].menu()
    assert "Import geometry" in _menu_texts(file_menu)
    assert not w.act_import_mesh.isEnabled()                  # no project yet
    w.state.request("new-wing")                               # refused without a project
    assert "project first" in w.statusBar().currentMessage()
    proj = w.new_study("blank", "Commands", isolated)
    assert w.act_import_mesh.isEnabled() and w.act_import_geometry.isEnabled()
    w.state.request("new-wing")                               # through the Geometry page when it has one
    (d,) = proj.designs.values()
    assert d.kind == "wing" and d.dimension == 3 and w.state.design_id == d.id
    assert w.current_page_key() == "geometry"
    w.act_new_wing.trigger()
    assert len(proj.designs) == 2 and len({x.name for x in proj.designs.values()}) == 2
    calls = []
    monkeypatch.setattr(mw, "mesh_import_hook", lambda: _fake_hook([_box_design("Dropped")], calls))
    w.state.request(f"import-mesh:{SPHERE_STL}")
    assert calls[-1][2] == str(SPHERE_STL) and "Dropped" in {x.name for x in proj.designs.values()}
    w.act_import_mesh.trigger()
    assert calls[-1][2] is None
    w.state.request("frobnicate")
    assert "Unknown command" in w.statusBar().currentMessage()
    w.close()


@needs_store
def test_dropped_meshes_go_to_the_mesh_import(isolated, qapp, tmp_path):
    from PySide6.QtCore import QMimeData, QUrl
    w = _window()
    w.new_study("blank", "Drops 3D", isolated)
    dat = tmp_path / "foil.dat"
    dat.write_text("x\n1 0\n0 0\n1 0\n", encoding="utf-8")
    stl = tmp_path / "car.STL"
    stl.write_bytes(SPHERE_STL.read_bytes())
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(dat)), QUrl.fromLocalFile(str(stl))])
    nav, req = [], []
    w.state.navigate = nav.append
    w.state.request = req.append
    accepted = []
    w.dropEvent(SimpleNamespace(mimeData=lambda: md, acceptProposedAction=lambda: accepted.append(1),
                                ignore=lambda: None))
    assert accepted and len(nav) == 1 and nav[0].startswith("geometry:import:")
    assert len(req) == 1 and req[0].startswith("import-mesh:") and req[0].endswith("car.STL")
    w.close()


@needs_store
def test_status_bar_shows_cells_and_throughput(isolated, qapp):
    from aerolume.app.main_window import format_cells
    from aerolume.model import Simulation
    assert format_cells(1_250_000) == "1.25 M cells" and format_cells(524_288) == "524 k cells"
    assert format_cells(800) == "800 cells"
    jobs = FakeJobs3D()
    w = _window(jobs=jobs)
    w.new_study("blank", "Status 3D", isolated)
    d = _box_design()
    w.state.add_design(d)
    sim = Simulation("Car CFD", d.id)
    w.state.add_simulation(sim)
    rs = w.run_status
    jobs.running = sim.id
    jobs.jobStarted.emit(sim.id)
    jobs.jobProgress.emit(sim.id, 0.1, "Developing the flow")
    assert rs.summary() == "Running Car CFD  10 %  ·  Developing the flow"
    jobs.jobPlanned.emit(sim.id, SimpleNamespace(cells=1_250_000, grid=(250, 100, 50)))
    jobs.jobFrame.emit(sim.id, SimpleNamespace(steps_per_s=61.3))
    assert rs.summary() == "Running Car CFD  10 %  ·  Developing the flow  ·  1.25 M cells · 61 steps/s"
    jobs.jobFrame.emit(sim.id, SimpleNamespace(steps_per_s=None))   # a frame without a rate changes nothing
    assert rs.metrics(sim.id) == "1.25 M cells · 61 steps/s"
    jobs.jobFrame.emit(sim.id, SimpleNamespace(steps_per_s=4.26))          # a CPU run
    assert rs.metrics(sim.id) == "1.25 M cells · 4.3 steps/s"
    jobs.running = None
    jobs.jobFinished.emit(sim.id, None)
    assert rs.metrics(sim.id) == "" and "done" in rs.summary()
    w.close()


# ---------------------------------------------------------------------------
# screenshots (review artefacts, written outside the repository)
# ---------------------------------------------------------------------------

@needs_store
@pytest.mark.parametrize("name", ["light", "dark"])
def test_render_3d_shell_screenshots(isolated, qapp, name, tmp_path):
    """Start page with 3D studies and a shell with a mesh and a wing in the tree."""
    from aerolume.app import theme
    from aerolume.app.pages import start
    shots = Path(os.environ.get("AEROLUME_SHOTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    theme.apply(qapp, name)
    try:
        start.create_study("wing", "Glider wing", isolated)
        start.create_study("mesh", "Ahmed body", isolated, designs=[_box_design("Ahmed body")])
        w = _window()
        w._device_ready("cuda", "CUDA · RTX 4060 Ti", "test")
        w.resize(1600, 1000)
        w.show()
        _pump(qapp, 10)
        start_png = shots / f"test3d_start_{name}.png"
        assert w.grab().save(str(start_png))
        proj = w.new_study("airfoil", "Mixed", isolated)
        for d in (_box_design("Ahmed body"), _wing_design("Main wing")):
            w.state.add_design(d)
            w.state.new_simulation(d.id)
        w.state.navigate("setup")
        _pump(qapp, 10)
        shell_png = shots / f"test3d_shell_{name}.png"
        assert w.grab().save(str(shell_png))
        assert len(proj.designs) == 3
        for p in (start_png, shell_png):
            assert p.stat().st_size > 15_000
        w.close()
    finally:
        theme.apply(qapp, "light")


@needs_store
def test_new_wing_without_a_geometry_page(isolated, qapp):
    """The shell adds the default wing itself when the Geometry page offers no "New wing"."""
    w = _window()
    proj = w.new_study("blank", "No page", isolated)
    w._pages["geometry"] = page = DummyPage()
    w.stack.addWidget(page)
    d = w.new_wing()
    assert d is not None and d.kind == "wing" and d.name == "Tapered wing"
    assert w.new_wing().name == "Tapered wing 2" and len(proj.designs) == 2
    assert w.state.design_id != d.id and w.current_page_key() == "geometry"
    w.close()


def test_default_wing_falls_back_when_the_editor_is_missing(monkeypatch):
    import sys

    from aerolume.app.pages import start
    monkeypatch.setitem(sys.modules, "aerolume.app.pages.geometry", None)    # import raises ImportError
    spec = start.default_wing_spec()
    assert spec.symmetric and [s.chord for s in spec.sections] == [0.25, 0.15]
    assert spec.sections[1].x_le == pytest.approx(0.025)                       # straight quarter-chord line
