"""Pre-release fixes in the shell (0.1.0), from the third critique round.

* every image the README and the docs link exists (the front page opened with three
  broken images);
* the breadcrumb names a simulation once when it shares its design's name;
* keyboard focus is visible on every focusable widget of the Setup and Start pages,
  scroll areas are not nameless Tab stops, and every button meets the 24 px target
  size or its spacing exception (WCAG 2.4.7, 4.1.2, 2.2 SC 2.5.8);
* the Scripting workspace gives the Script editor 40 % of the window (rev 3, v3.4 audit);
* the page area reports the current page's minimum only;
* the Start page offers the 2D enclosure and lists the bundled heat-transfer examples.

Offscreen, private settings, as the other shell tests.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.gui

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


@pytest.fixture()
def isolated(tmp_path, monkeypatch, qapp):
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    projects = tmp_path / "projects"
    projects.mkdir()
    yield projects
    _destroy_windows(qapp)
    theme.apply(qapp, "light")
    settings.use_file(tmp_path / "settings-after.ini")


_WINDOWS: list = []


def _pump(app, n: int = 5) -> None:
    for _ in range(n):
        app.processEvents()


def _window(**kw):
    from aerolume.app.main_window import MainWindow
    kw.setdefault("probe_device", False)
    kw.setdefault("load_jobs", False)
    w = MainWindow(**kw)
    _WINDOWS.append(w)
    return w


def _destroy_windows(app) -> None:
    from PySide6.QtCore import QEvent
    while _WINDOWS:
        w = _WINDOWS.pop()
        try:
            w._quit_confirmed = True
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    _pump(app, 2)


def _shown_window(qapp, isolated, size=(1600, 1000), **kw):
    w = _window(**kw)
    w.resize(*size)
    w.show()
    w.new_study("airfoil", f"Release {size[0]}", isolated)
    _pump(qapp, 8)
    return w


# ---------------------------------------------------------------------------
# documentation images
# ---------------------------------------------------------------------------

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_HTML_IMAGE = re.compile(r"<img\s[^>]*src=[\"']([^\"']+)[\"']", re.IGNORECASE)


def _documents() -> list[Path]:
    docs = [REPO / "README.md", REPO / "validation" / "README.md", REPO / "examples" / "README.md"]
    docs += sorted((REPO / "docs").glob("*.md"))
    return [d for d in docs if d.is_file()]


def test_every_image_the_readme_and_docs_link_exists():
    """The repository front page (and the website built from it) must not open on broken images."""
    missing, seen = [], 0
    for doc in _documents():
        text = doc.read_text(encoding="utf-8")
        for target in _MD_IMAGE.findall(text) + _HTML_IMAGE.findall(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue                                       # http(s):, data: and anchors
            seen += 1
            path = (doc.parent / target.split("#")[0].split("?")[0]).resolve()
            if not path.is_file():
                missing.append(f"{doc.relative_to(REPO)} -> {target}")
    assert seen >= 3, "no local images found: the pattern no longer matches the docs"
    assert missing == []


def test_readme_headline_screenshots_are_real_screenshots():
    """The three 0.3 headline images are full-window captures, not placeholders."""
    from PySide6.QtGui import QImage
    for name in ("ui_results3d.png", "ui_heat.png", "ui_explore.png"):
        img = QImage(str(REPO / "docs" / "img" / name))
        assert not img.isNull(), name
        assert img.width() >= 1280 and img.height() >= 800, (name, img.width(), img.height())


# ---------------------------------------------------------------------------
# breadcrumb
# ---------------------------------------------------------------------------

def test_breadcrumb_names_a_simulation_once_when_it_shares_the_design_name():
    from aerolume.app.main_window import breadcrumb_text
    d = SimpleNamespace(name="Heated cylinder · d 10 mm · ΔT 10 K")
    same = SimpleNamespace(name="heated  cylinder · d 10 mm · ΔT 10 K")
    other = SimpleNamespace(name="CFD · Re 40 · heated")
    assert breadcrumb_text(d, same) == "Heated cylinder · d 10 mm · ΔT 10 K"
    assert breadcrumb_text(d, other) == "Heated cylinder · d 10 mm · ΔT 10 K  ›  CFD · Re 40 · heated"
    assert breadcrumb_text(d, None) == "Heated cylinder · d 10 mm · ΔT 10 K"
    assert breadcrumb_text(None, None) == ""


def test_window_breadcrumb_follows_a_renamed_simulation(isolated, qapp):
    w = _shown_window(qapp, isolated)
    d, s = w.state.design, w.state.simulation
    assert d is not None and s is not None
    w.step_bar.set_context("", "")
    s.name = d.name
    w._selection_changed()
    assert w.step_bar.context() == d.name                       # once, no "›"
    s.name = "CFD · α 4°"
    w._selection_changed()
    assert w.step_bar.context() == f"{d.name}  ›  CFD · α 4°"


# ---------------------------------------------------------------------------
# keyboard: visible focus, no nameless stops, target size
# ---------------------------------------------------------------------------

def _focus_changes_pixels(app, w, x) -> bool:
    from PySide6.QtCore import QPoint, Qt
    x.clearFocus()
    w.setFocus()
    _pump(app, 3)
    r = x.rect().translated(x.mapTo(w, QPoint(0, 0))).adjusted(-3, -3, 3, 3)
    before = w.grab(r).toImage()
    x.setFocus(Qt.FocusReason.TabFocusReason)
    _pump(app, 3)
    assert x.hasFocus(), (type(x).__name__, x.objectName())
    return w.grab(r).toImage() != before


def _tab_stops(w):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget
    return [x for x in w.findChildren(QWidget)
            if x.isEnabled() and x.isVisible() and x.focusPolicy() & Qt.FocusPolicy.TabFocus
            and x.visibleRegion().boundingRect().width() > 4
            and not x.objectName().startswith("qt_spinbox")]


@pytest.mark.parametrize("page", ["setup", "start"])
def test_every_focusable_widget_changes_pixels_on_focus(isolated, qapp, page):
    from aerolume.app import theme
    w = _shown_window(qapp, isolated)
    w.show_page(page)
    _pump(qapp, 10)
    w.activateWindow()
    for name in ("light", "dark"):
        theme.apply(qapp, name)
        _pump(qapp, 4)
        stops = _tab_stops(w)
        assert len(stops) >= 15
        unchanged = [(type(x).__name__, x.objectName(), x.accessibleName()) for x in stops
                     if not _focus_changes_pixels(qapp, w, x)]
        assert unchanged == [], name


def test_group_header_draws_a_two_pixel_focus_ring(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from aerolume.app import theme
    from aerolume.app.widgets.common import CollapsibleSection
    theme.apply(qapp, "light")
    host = QWidget()
    lay = QVBoxLayout(host)
    sec = CollapsibleSection("Heat transfer")
    lay.addWidget(sec)
    host.resize(360, 80)
    host.show()
    host.activateWindow()
    _pump(qapp, 4)
    head = sec.header
    head.clearFocus()                       # activating the window gave it focus (its only Tab stop)
    _pump(qapp, 2)
    assert not head.hasFocus()
    before = head.grab().toImage()
    head.setFocus(Qt.FocusReason.TabFocusReason)
    _pump(qapp, 4)
    assert head.hasFocus()
    after = head.grab().toImage()
    assert after != before
    focus = theme.qcolor("focus")
    ring = head.focus_rect()
    # the left edge of the ring, two pixels wide, in the focus colour; the rule starts after the ring
    y = head.height() // 2
    for x in (0, 1):
        c = after.pixelColor(x, y)
        assert abs(c.red() - focus.red()) + abs(c.green() - focus.green()) + abs(c.blue() - focus.blue()) < 40
    assert ring.right() < head.width()
    host.close()


def test_scroll_areas_are_not_nameless_tab_stops(isolated, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QScrollArea
    w = _shown_window(qapp, isolated)
    w.show_page("setup")
    _pump(qapp, 10)
    w.activateWindow()
    w.setFocus()
    stops = []
    for _ in range(70):
        QTest.keyClick(qapp.focusWidget() or w, Qt.Key.Key_Tab)
        _pump(qapp, 1)
        f = qapp.focusWidget()
        if f is not None:
            stops.append(f)
    scrolls = [f for f in stops if isinstance(f, QScrollArea)]
    assert [type(f).__name__ for f in scrolls if not f.accessibleName()] == []
    # the panel holders and the property grid's area never take focus themselves
    assert all(p.scroll.focusPolicy() == Qt.FocusPolicy.NoFocus for p in w.docks.panels.values())


def _targets_ok(buttons) -> list:
    """WCAG 2.2 SC 2.5.8: 24 x 24 px, or a 24 px circle on the target's centre that meets no other's."""
    bad = []
    centres = [(b, b.mapToGlobal(b.rect().center())) for b in buttons]
    for b, c in centres:
        if b.width() >= 24 and b.height() >= 24:
            continue
        for o, oc in centres:
            if o is b:
                continue
            if math.hypot(c.x() - oc.x(), c.y() - oc.y()) < 24:
                bad.append((type(b).__name__, b.objectName(), b.toolTip()[:30], b.width(), b.height()))
                break
    return bad


def test_buttons_meet_the_target_size_or_its_spacing_exception(isolated, qapp):
    from PySide6.QtWidgets import QAbstractButton

    from aerolume.app.widgets.common import IconButton
    w = _shown_window(qapp, isolated)
    for page in ("setup", "geometry"):
        w.show_page(page)
        _pump(qapp, 8)
        buttons = [b for b in w.findChildren(QAbstractButton) if b.isVisible()]
        assert _targets_ok(buttons) == [], page
        icon_buttons = [b for b in buttons if isinstance(b, IconButton)]
        assert icon_buttons and all(b.width() >= 24 and b.height() >= 24 for b in icon_buttons)
    filters = list(w.messages.filter_buttons.values()) + [w.messages.scope_button, w.messages.clear_button]
    assert all(b.height() >= 24 for b in filters)
    assert IconButton("close", "Close", size=18).width() == 24          # never below the minimum


# ---------------------------------------------------------------------------
# Scripting workspace
# ---------------------------------------------------------------------------

def _editor_lines(w) -> float:
    ed = w.script_console.editor
    return ed.viewport().height() / ed.fontMetrics().lineSpacing()


@pytest.mark.parametrize(("size", "lines"), [((1600, 1000), 18), ((1280, 800), 12)])
def test_scripting_workspace_puts_the_editor_first(isolated, qapp, size, lines):
    w = _shown_window(qapp, isolated, size=size)
    w.show_page("setup")
    _pump(qapp, 4)
    w.set_workspace("Scripting")
    _pump(qapp, 12)
    assert w.script_dock.isVisible()
    assert _editor_lines(w) >= lines
    if size == (1600, 1000):
        # 40 %, never the old 150 px, and never the 54 % that left the Setup page 300 px (v3.4 audit)
        assert 0.37 * w.height() <= w.script_dock.height() <= 0.43 * w.height()
    # a window resize keeps the share (the console and the workspace agree on it)
    w.resize(size[0], size[1] - 100)
    _pump(qapp, 10)
    w.resize(*size)
    _pump(qapp, 10)
    assert _editor_lines(w) >= lines


def test_a_squeezed_scripting_rail_is_not_learnt_and_old_layouts_are_replaced(isolated, qapp):
    from aerolume.app import settings
    from aerolume.app.widgets import docks
    stale = json.loads(json.dumps(docks.WORKSPACES["Scripting"]))
    stale.pop("shares")
    stale["rev"] = 1
    stale["sizes"]["bottom"] = 173                              # captured after a 1280 x 800 session
    settings.get().set_value("workspaces/Scripting", json.dumps(stale))
    w = _shown_window(qapp, isolated, size=(1280, 800))
    w.set_workspace("Scripting")
    _pump(qapp, 12)
    assert w.docks.bottom_share() == pytest.approx(0.40)
    cap = w.docks.capture()
    assert cap["shares"]["bottom"] == pytest.approx(0.40) and cap["rev"] == 3
    w.resize(1600, 1000)
    _pump(qapp, 12)
    assert _editor_lines(w) >= 18


# ---------------------------------------------------------------------------
# the page area
# ---------------------------------------------------------------------------

def test_page_area_reports_the_current_pages_minimum(isolated, qapp):
    w = _shown_window(qapp, isolated)
    for key in ("geometry", "setup", "start"):
        w.show_page(key)
        _pump(qapp, 3)
    w.show_page("setup")
    _pump(qapp, 3)
    tallest = max(w.stack.widget(i).minimumSizeHint().height() for i in range(w.stack.count()))
    assert w.stack.minimumSizeHint().height() == w.page("setup").minimumSizeHint().height()
    assert w.stack.minimumSizeHint().height() <= tallest


# ---------------------------------------------------------------------------
# Start page: the enclosure tile and the heat-transfer examples
# ---------------------------------------------------------------------------

def test_start_page_offers_the_enclosure_and_wraps_tiles_in_even_rows(isolated, qapp):
    from aerolume.app.pages import start
    if not start.physics_templates_available():
        pytest.skip("app/dialogs/physics_templates.py is not part of this build")
    page = start.StartPage(None)
    page.resize(1600, 960)
    page.show()
    _pump(qapp, 6)
    assert list(page.tiles)[-2:] == ["room", "enclosure"]
    assert page.tile_columns() == len(page.tiles)
    assert "minutes on a GPU" in page.tiles["enclosure"].accessibleDescription()
    page.resize(900, 760)                                        # beside an open Assistant at 1280 px
    _pump(qapp, 6)
    cols = page.tile_columns()
    assert cols < len(page.tiles) and len(page.tiles) % cols in (0, cols - 1)
    for t in page.tiles.values():
        assert t.width() >= t.minimumWidth() and t.isVisible()
        assert t.width() <= page.width()
    page.close()


def test_bundled_heat_transfer_examples_have_finished_results():
    from aerolume.app.pages import start
    found = {p.name: p for p in start.example_projects()}
    for name in ("Heated cavity Ra 1e5.aerolume", "Heated cylinder.aerolume", "Room with a fan and a heater.aerolume"):
        assert name in found, sorted(found)
        sims = list((found[name] / "sims").glob("*/result.json"))
        assert sims, f"{name} has no stored result"
        summary = start.summarize(found[name])
        assert summary.icon == "thermometer" and summary.simulations >= 1


def test_checked_boxes_and_kept_scroll_stops_show_focus(qapp):
    """A checked box is filled with the focus blue, and a frameless read-only area has no border:
    both need their own focus mark. A kept scroll stop is named after the heading above it."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QCheckBox, QLabel, QScrollArea, QVBoxLayout, QWidget

    from aerolume.app import theme
    from aerolume.app.widgets.common import unfocus_scroll_areas
    theme.apply(qapp, "light")
    host = QWidget()
    lay = QVBoxLayout(host)
    box = QCheckBox("Heat transfer")
    box.setChecked(True)
    lay.addWidget(box)
    head = QLabel("Estimate")
    head.setProperty("role", "h3")
    lay.addWidget(head)
    area = QScrollArea()
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    text = QLabel("\n".join(f"row {i}" for i in range(40)))
    area.setWidget(text)
    area.setFixedHeight(80)
    lay.addWidget(area)
    host.resize(300, 240)
    host.show()
    host.activateWindow()
    _pump(qapp, 4)
    assert unfocus_scroll_areas(host) == 1
    assert area.focusPolicy() != Qt.FocusPolicy.NoFocus             # overflowing read-only text: keyboard scroll
    assert area.accessibleName() == "Estimate (scrollable)"
    for w in (box, area):
        host.setFocus()
        _pump(qapp, 2)
        before = host.grab().toImage()
        w.setFocus(Qt.FocusReason.TabFocusReason)
        _pump(qapp, 3)
        assert w.hasFocus() and host.grab().toImage() != before, type(w).__name__
    host.close()
