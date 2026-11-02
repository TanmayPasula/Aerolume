"""Fixes from the second critique of the v4 restyle.

* the Assistant's example goals and Script hint get every wrapped line (they
  showed one line plus the top half of the next in a short panel);
* the Assistant footer (goal box, hint, Run button) is never clipped when the
  panel shares the window's height with the Script panel;
* the toolbar's "Import geometry" and "Workspace" buttons are window-owned
  QActions, not cached ``menuAction()`` wrappers that went stale, and one dead
  action cannot abort ``_project_changed``;
* ``MeshImportDialog.size()`` is QWidget.size() again (an attribute shadowed it);
* the frozen self-test checks that the bundled font is the one Qt resolved;
* the Results tab bar shows the result's tabs after it was refilled while hidden.

Offscreen, private settings, as the other shell tests.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


_WINDOWS: list = []


@pytest.fixture()
def isolated(tmp_path, monkeypatch, qapp):
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    yield tmp_path
    while _WINDOWS:
        w = _WINDOWS.pop()
        with contextlib.suppress(Exception):
            w.jobs.shutdown()
        try:
            w._quit_confirmed = True
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    from PySide6.QtCore import QEvent
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    theme.apply(qapp, "light")
    settings.use_file(tmp_path / "settings-after.ini")


def _wait(app, sec: float = 0.3) -> None:
    t = time.time()
    while time.time() - t < sec:
        app.processEvents()
        time.sleep(0.01)


def _window(app, size=(1280, 800)):
    from aerolume.app.main_window import MainWindow
    w = MainWindow()
    _WINDOWS.append(w)
    w.resize(*size)
    w.show()
    _wait(app, 0.3)
    return w


def _open_example(w, app, tmp_path, name: str):
    from aerolume.app.pages.start import example_projects
    from aerolume.project import Project
    src = {p.name.removesuffix(".aerolume"): p for p in example_projects()}.get(name)
    if src is None:
        pytest.skip(f"example {name!r} not bundled")
    dst = tmp_path / "examples" / src.name
    if not dst.exists():
        shutil.copytree(src, dst)
    w.set_project(Project.open(dst))
    _wait(app, 0.3)
    sim = next(iter(w.state.project.simulations.values()))
    w.state.select_design(sim.design_id)
    w.state.select_simulation(sim.id)
    _wait(app, 0.2)
    return sim


# ---------------------------------------------------------------------------
# Assistant panel
# ---------------------------------------------------------------------------

def _fully_shown(lab) -> bool:
    return lab.height() >= lab.heightForWidth(lab.width()) > 0


def test_assistant_links_get_every_wrapped_line(isolated, qapp):
    """The example goals and the Script hint have height-for-width, like the introduction above them."""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from aerolume.app.state import AppState
    from aerolume.app.widgets.assistant_panel import AssistantPanel
    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    panel = AssistantPanel(AppState())
    lay.addWidget(panel)
    for ex in panel.example_buttons:
        ex.setText(ex.text())                    # the generic goals: long enough to wrap at 320 px
    host.resize(320, 520)                        # narrow and short: every goal wraps to two or three lines
    host.show()
    _wait(qapp, 0.3)
    labels = [b for b in panel.example_buttons if b.isVisibleTo(panel)] + [panel.script_hint]
    assert len(labels) >= 2
    for lab in labels:
        assert lab.hasHeightForWidth() and lab.wordWrap()
        assert lab.heightForWidth(lab.width()) > lab.fontMetrics().height() * 1.5     # it does wrap here
        assert _fully_shown(lab), (lab.text()[:60], lab.height(), lab.heightForWidth(lab.width()))
    host.close()
    host.deleteLater()


def test_assistant_footer_is_never_clipped(isolated, qapp):
    """Assistant + Script share the height at 1280 x 800: the hint and the Run button stay whole."""
    w = _window(qapp, (1280, 800))
    _open_example(w, qapp, isolated, "Wing section NACA 2412")
    w.show_page("setup")
    w.show_assistant()
    w.show_script()
    _wait(qapp, 0.6)
    panel = w.assistant
    run = panel.run_button
    assert run.isVisibleTo(panel)
    # the Run button and the hint lie inside the visible part of the dock
    vp = panel
    while vp.parentWidget() is not None and type(vp.parentWidget()).__name__ not in ("DockPanel",):
        vp = vp.parentWidget()
    dock = vp.parentWidget() or panel
    bottom = run.mapTo(dock, run.rect().bottomLeft()).y()
    assert bottom <= dock.height(), (bottom, dock.height())
    assert panel.height() >= panel.minimumSizeHint().height() - 1
    assert _fully_shown(panel.hint)
    # the minimum is header + footer + a few transcript lines, so the splitter cannot squeeze the Run row
    foot = panel._foot
    assert panel.minimumSizeHint().height() >= foot.sizeHint().height() + panel.MIN_TRANSCRIPT


# ---------------------------------------------------------------------------
# toolbar actions
# ---------------------------------------------------------------------------

def test_toolbar_menu_actions_are_window_owned(isolated, qapp):
    w = _window(qapp)
    for name, menu in (("act_import_geometry", w.import_menu), ("act_workspace", w.workspace_menu)):
        act = getattr(w, name)
        # owned by the window (Qt 6's setMenu also makes it the menu's menuAction, so File > Import
        # geometry and the toolbar button are one action, enabled together)
        assert act.parent() is w and act.menu() is menu and menu.menuAction() is act
        assert w.main_toolbar.widgetForAction(act) is not None
    assert w.act_workspace.iconText() == w.docks.workspace


def test_project_changed_survives_a_dead_action(isolated, qapp):
    """One deleted action must not abort the title / step-bar update (it raised at every project change)."""
    from PySide6.QtGui import QAction
    w = _window(qapp)
    doomed = QAction("doomed", None)
    w.act_import_geometry = doomed
    import shiboken6
    shiboken6.delete(doomed)
    _open_example(w, qapp, isolated, "Wing section NACA 2412")
    w._project_changed()
    assert w.windowTitle().endswith("— Aerolume") and w.windowTitle() != "Aerolume"


# ---------------------------------------------------------------------------
# mesh import dialog, self-test fonts
# ---------------------------------------------------------------------------

def test_mesh_import_dialog_keeps_qwidget_size(isolated, qapp):
    from aerolume.app.dialogs.mesh_import import MeshImportDialog
    dlg = MeshImportDialog(None)
    dlg.show()
    _wait(qapp, 0.1)
    s = dlg.size()
    assert s.width() > 100 and s.height() > 100
    assert dlg.size_label.objectName() == "mesh_size"
    dlg.close()
    dlg.deleteLater()


def test_self_test_checks_the_resolved_font(isolated, qapp):
    from PySide6.QtGui import QFont

    from aerolume.app import platform, theme
    if not theme.bundled_fonts_available():
        pytest.skip("bundled fonts not available in this environment")
    detail = platform._font_status(qapp)
    assert theme.BUNDLED_UI_FAMILY in detail and "registered: True" in detail
    old = qapp.font()
    try:
        qapp.setFont(QFont("Arial"))           # a build that shipped the files but did not register them
        with pytest.raises(RuntimeError, match="interface font"):
            platform._font_status(qapp)
    finally:
        qapp.setFont(old)


# ---------------------------------------------------------------------------
# Results tab bar
# ---------------------------------------------------------------------------

def test_results_tabs_after_refill_while_hidden(isolated, qapp):
    """3D CFD result -> polar -> (Setup) -> 2D heated CFD: the tab bar lays out every visible tab.

    The critic's dark "docks" scene showed only Summary, Surface and Data for a
    heated-cylinder result; measured, ``isTabVisible`` was True for all eight tabs
    while the bar still held the polar's layout (Forces, Fields ... at zero width).
    """
    w = _window(qapp, (1600, 1000))

    def check(rp) -> list[str]:
        t, bar = rp.tabs, rp.tabs.tabBar()
        shown = [i for i in range(t.count()) if t.isTabVisible(i)]
        rects = {t.tabText(i): bar.tabRect(i).width() for i in shown}
        assert all(v > 0 for v in rects.values()), rects
        return [t.tabText(i) for i in shown]

    _open_example(w, qapp, isolated, "Sedan in 3D")
    w.show_page("results")
    _wait(qapp, 1.0)
    _open_example(w, qapp, isolated, "Wing section NACA 2412")
    rp = w.show_page("results")
    _wait(qapp, 0.5)
    assert "Polar" in check(rp)
    w.show_page("setup")
    _wait(qapp, 0.3)
    _open_example(w, qapp, isolated, "Heated cylinder")
    rp = w.show_page("results")
    _wait(qapp, 0.8)
    assert {"Forces", "Fields", "Heat transfer"} <= set(check(rp))
