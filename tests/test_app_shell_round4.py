"""Shell fixes from the fourth critique round (v3.4 audit).

* the Start page's study tiles show their whole description on the first show, at
  1600 x 1000 and 1280 x 800 (they lost the third line until a later relayout);
* a new study never opens in the Scripting workspace, and Scripting gives the Script
  panel 40 % of the height, not 54 %;
* Tab goes from the step bar straight into the page, the panel chrome is out of the
  Tab chain, and F6 / Shift+F6 cycle the regions and reach each panel's header
  (WCAG 2.4.3);
* segmented controls, toolbar buttons and check boxes are at least 24 px tall
  (WCAG 2.2 SC 2.5.8).

Offscreen, private settings, as the other shell tests.
"""
from __future__ import annotations

import os

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
    projects = tmp_path / "projects"
    projects.mkdir()
    yield projects
    while _WINDOWS:
        w = _WINDOWS.pop()
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


def _pump(app, n: int = 5) -> None:
    for _ in range(n):
        app.processEvents()


def _window(size=(1600, 1000)):
    from aerolume.app.main_window import MainWindow
    w = MainWindow(probe_device=False, load_jobs=False)
    _WINDOWS.append(w)
    w.resize(*size)
    w.show()
    return w


# ---------------------------------------------------------------------------
# Start page tiles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme_name", ["light", "dark"])
@pytest.mark.parametrize("size", [(1600, 1000), (1280, 800)])
def test_study_tiles_show_their_whole_description_on_first_show(isolated, qapp, size, theme_name):
    """A fresh window at each size: what the user sees first is what failed (the tiles were
    fitted to the old widths before the grid handed out the new ones)."""
    from PySide6.QtGui import QColor

    from aerolume.app import theme
    theme.apply(qapp, theme_name)
    w = _window(size)
    _pump(qapp, 12)
    page = w.page("start")
    surface = QColor(theme.qcolor("surface"))
    for key, t in page.tiles.items():
        assert t.isVisible(), key
        # measured at the width the tile has now, the whole description fits its height
        assert t.required_height(t.width()) <= t.height(), (key, t.width(), t.height())
        # and nothing is painted in the bottom padding: a cut line left half a row of glyphs there
        t.clearFocus()
        img = t.grab().toImage()
        y0 = t.height() - t.PAD + 2
        for y in range(y0, t.height() - 3):
            for x in range(t.PAD, t.width() - t.PAD):
                c = img.pixelColor(x, y)
                assert (abs(c.red() - surface.red()) + abs(c.green() - surface.green())
                        + abs(c.blue() - surface.blue())) < 30, (key, x, y, c.name())
    # one height for every tile, and the heightForWidth the layout asks for agrees
    heights = {t.height() for t in page.tiles.values()}
    assert len(heights) == 1
    t = next(iter(page.tiles.values()))
    assert t.hasHeightForWidth() and t.heightForWidth(t.width()) <= t.height()


def test_tiles_refit_after_a_resize_without_the_page_resizing_first(isolated, qapp):
    """Narrowing the window re-fits the tiles once the grid has given them their new widths."""
    w = _window((1600, 1000))
    _pump(qapp, 10)
    page = w.page("start")
    w.resize(1100, 800)
    _pump(qapp, 12)
    for key, t in page.tiles.items():
        assert t.required_height(t.width()) <= t.height(), key
    w.resize(1600, 1000)
    _pump(qapp, 12)
    for key, t in page.tiles.items():
        assert t.required_height(t.width()) <= t.height(), key


# ---------------------------------------------------------------------------
# workspaces
# ---------------------------------------------------------------------------

def test_a_new_study_opens_in_the_last_study_workspace_not_scripting(isolated, qapp):
    w = _window()
    _pump(qapp, 4)
    w.set_workspace("Analysis")
    w.set_workspace("Scripting")
    assert w.docks.study_workspace() == "Analysis"
    w.new_study("room", "Heated room", isolated)
    _pump(qapp, 10)
    assert w.docks.workspace == "Analysis"
    assert not w.script_dock.isVisible()
    # from Default (no other workspace chosen yet) it is Default; a study workspace is kept as is
    w.set_workspace("Live")
    assert w.docks.study_workspace() == "Live"


def test_the_last_study_workspace_survives_a_restart(isolated, qapp):
    w = _window()
    w.set_workspace("Live")
    w.set_workspace("Scripting")
    w.docks.save()
    w2 = _window()
    assert w2.docks.workspace == "Scripting" and w2.docks.study_workspace() == "Live"


def test_scripting_gives_the_script_panel_40_percent_and_the_page_the_rest(isolated, qapp):
    from aerolume.app.widgets import docks
    assert docks.WORKSPACES["Scripting"]["shares"]["bottom"] == pytest.approx(0.40)
    assert docks.preset_rev("Scripting") == 3
    w = _window()
    w.new_study("room", "Heated room", isolated)
    w.show_page("setup")
    _pump(qapp, 6)
    w.set_workspace("Scripting")
    _pump(qapp, 14)
    assert w.script_dock.isVisible()
    assert 0.37 * w.height() <= w.script_dock.height() <= 0.43 * w.height()
    assert w.stack.height() >= 420                      # the Setup page kept about 300 px at 54 %


def test_a_dragged_share_is_capped_at_60_percent(isolated, qapp):
    import json

    from aerolume.app import settings
    from aerolume.app.widgets import docks
    raw = json.loads(json.dumps(docks.WORKSPACES["Scripting"]))
    raw["shares"]["bottom"] = 0.75
    settings.get().set_value("workspaces/Scripting", json.dumps(raw))
    w = _window()
    w.set_workspace("Scripting")
    assert w.docks.bottom_share() == pytest.approx(0.40)        # out of range: back to the preset


# ---------------------------------------------------------------------------
# keyboard: Tab order and F6
# ---------------------------------------------------------------------------

def _setup_window(qapp, isolated):
    w = _window()
    w.new_study("room", "Heated room", isolated)
    w.set_workspace("Default")
    w.show_page("setup")
    _pump(qapp, 12)
    w.activateWindow()
    return w


def test_tab_goes_from_the_step_bar_straight_into_the_page(isolated, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from aerolume.app.widgets.common import IconButton
    w = _setup_window(qapp, isolated)
    w.step_bar.setFocus()
    _pump(qapp, 2)
    QTest.keyClick(w.step_bar, Qt.Key.Key_Tab)
    _pump(qapp, 2)
    f = qapp.focusWidget()
    assert f is not None and w.centralWidget().isAncestorOf(f), type(f).__name__
    chain = w.tab_chain()
    assert chain[0] is w.step_bar
    regions = []
    for x in chain:
        r = "page" if w.centralWidget().isAncestorOf(x) else ("steps" if x is w.step_bar else "dock")
        if not regions or regions[-1] != r:
            regions.append(r)
    assert regions[:2] == ["steps", "page"] and regions.count("page") == 1   # the page is one run
    # no panel chrome in the chain
    for p in w.docks.panels.values():
        for b in (p.header.collapse_button, p.header.menu_button, p.header.close_button):
            assert b not in chain and b.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert not [b for b in chain if isinstance(b, IconButton) and "panel" in b.toolTip().lower()
                and b.parentWidget() in [p.header for p in w.docks.panels.values()]]
    # Shift+Tab walks back the same chain
    QTest.keyClick(qapp.focusWidget(), Qt.Key.Key_Backtab)
    _pump(qapp, 2)
    assert qapp.focusWidget() is w.step_bar


def test_f6_cycles_the_regions_and_lands_on_panel_headers(isolated, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = _setup_window(qapp, isolated)
    w.step_bar.setFocus()
    _pump(qapp, 2)
    seen = [w.cycle_region(1) for _ in range(len(w.focus_regions()))]
    assert seen[0] == "page" and seen[-1] == "steps"
    assert "projectDock" in seen and "messagesDock" in seen
    # on the Messages header: Enter expands, Enter folds, Tab goes into the panel
    w.step_bar.setFocus()
    while w.cycle_region(1) != "messagesDock":
        pass
    hdr = w.docks.panel("messages").header
    assert hdr.hasFocus()
    before = w.docks.is_collapsed("messages")
    QTest.keyClick(hdr, Qt.Key.Key_Return)
    _pump(qapp, 4)
    assert w.docks.is_collapsed("messages") != before
    hdr.focus_header()
    QTest.keyClick(hdr, Qt.Key.Key_Tab)
    _pump(qapp, 2)
    assert w.docks.panel("messages").isAncestorOf(qapp.focusWidget())
    # Shift+F6 goes back; F6 from the page returns to the field that had focus there
    field = next(x for x in w.tab_chain() if w.centralWidget().isAncestorOf(x) and x.isVisible())
    field.setFocus()
    _pump(qapp, 2)
    assert w.cycle_region(-1) == "steps"
    assert w.cycle_region(1) == "page" and qapp.focusWidget() is field
    # the shortcuts are on the window and listed in the Keyboard shortcuts dialog
    assert w.act_next_region.shortcut().toString() == "F6"
    assert w.act_prev_region.shortcut().toString() == "Shift+F6"
    from aerolume.app.main_window import SHORTCUTS
    assert any(k.startswith("F6") for k, _ in SHORTCUTS)


def test_header_keys_open_the_menu_and_hide(isolated, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = _setup_window(qapp, isolated)
    hdr = w.docks.panel("project").header
    asked = []
    hdr.menuRequested.disconnect()          # the manager's slot exec()s the menu, which would block here
    hdr.menuRequested.connect(lambda pos: asked.append(pos))
    hdr.focus_header()
    QTest.keyClick(hdr, Qt.Key.Key_F10, Qt.KeyboardModifier.ShiftModifier)
    assert len(asked) == 1
    assert hdr.accessibleName() == "Project panel" and "Enter" in hdr.accessibleDescription()
    QTest.keyClick(hdr, Qt.Key.Key_Delete)
    _pump(qapp, 4)
    assert not w.docks.is_shown("project")


# ---------------------------------------------------------------------------
# target sizes
# ---------------------------------------------------------------------------

def test_segmented_toolbar_and_check_box_targets_are_24_px(isolated, qapp):
    from PySide6.QtWidgets import QAbstractButton, QCheckBox

    from aerolume.app.widgets.common import SegmentedControl
    w = _setup_window(qapp, isolated)
    seg = SegmentedControl([("a", "Step"), ("b", "Time")], "a")
    seg.show()
    _pump(qapp, 2)
    assert all(b.height() >= 24 for b in seg.findChildren(QAbstractButton))
    seg.close()
    tb = [b for b in w.main_toolbar.findChildren(QAbstractButton) if b.isVisible()]
    assert tb and all(b.height() >= 24 and b.width() >= 24 for b in tb)
    boxes = [b for b in w.findChildren(QCheckBox) if b.isVisible()]
    assert boxes and all(b.height() >= 24 for b in boxes)
