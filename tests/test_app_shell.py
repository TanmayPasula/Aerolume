"""Tests for the desktop shell: theme, settings, shared widgets, start page, main window.

All run on Qt's ``offscreen`` platform. Every test gets a private settings INI
and a private project-store config folder, so nothing touches the real user's
preferences or recent-projects list. Screenshots go to a temporary folder
unless ``AEROLUME_SHOTS`` names one (keep them out of the repository).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    app = create_app(["aerolume-tests"])
    yield app


@pytest.fixture()
def isolated(tmp_path, monkeypatch, qapp):
    """Private settings + recent list, light theme, and a projects folder."""
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    projects = tmp_path / "projects"
    projects.mkdir()
    yield projects
    _destroy_windows(qapp)
    settings.use_file(tmp_path / "settings-after.ini")


def _have_project_store() -> bool:
    try:
        from aerolume.project import Project  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


needs_store = pytest.mark.skipif(not _have_project_store(), reason="aerolume.project not available")


class FakeJobs(QObject):
    """Just enough of JobManager for the shell: signals + queue queries."""

    jobQueued = Signal(str)
    jobStarted = Signal(str)
    jobProgress = Signal(str, float, str)
    jobFrame = Signal(str, object)
    jobFinished = Signal(str, object)
    jobFailed = Signal(str, str)
    jobCancelled = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.shutdown_called = 0
        self.cancelled: list[str] = []
        self.running: str | None = None
        self.queue: list[str] = []

    def running_id(self):
        return self.running

    def queued_ids(self):
        return list(self.queue)

    def cancel(self, sim_id: str) -> None:
        self.cancelled.append(sim_id)

    def shutdown(self) -> None:
        self.shutdown_called += 1


class DummyPage(QWidget):
    """Records the page protocol calls the window makes."""

    def __init__(self, state=None, jobs=None) -> None:
        super().__init__()
        self.shown = 0
        self.actions: list[str] = []
        self.shutdowns = 0

    def on_shown(self) -> None:
        self.shown += 1

    def handle_action(self, action: str) -> None:
        self.actions.append(action)

    def shutdown(self) -> None:
        self.shutdowns += 1


def _pump(app, n: int = 5) -> None:
    for _ in range(n):
        app.processEvents()


_WINDOWS: list = []


def _window(**kw):
    """A MainWindow without the device probe / real JobManager, destroyed after the test.

    Closed-but-alive windows would otherwise pile up, and every later theme switch
    re-polishes all of them (seconds per test).
    """
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
            w.jobs = None if not isinstance(w.jobs, FakeJobs) else w.jobs
            w._quit_confirmed = True            # never ask a test harness about running jobs
            w.close()
            w.deleteLater()
        except RuntimeError:        # already deleted on the C++ side
            pass
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    _pump(app, 2)


def _plate_design(name: str):
    from aerolume.model import Body, Design
    sq = np.array([[0, 0], [1, 0], [1, 0.2], [0, 0.2]], float)
    return Design(name, "generic", [Body(name.lower(), sq)], 1.0, 1.0)


# ---------------------------------------------------------------------------
# theme
# ---------------------------------------------------------------------------

GUIDE_TOKENS = {   # docs/ui_guidelines.md section 2 (v4 semantic tokens, the website's palette), light / dark
    "bg": ("#efefef", "#0a0a0a"), "bg_subtle": ("#e9e9e9", "#151515"), "surface": ("#ffffff", "#1a1a1a"),
    "surface_2": ("#f6f6f6", "#111111"), "fg": ("#0e0e0e", "#f1f1f1"), "fg2": ("#474747", "#b1b1b1"),
    "muted": ("#5c5c5c", "#a3a3a3"), "faint": ("#8e8e8e", "#6b6b6b"), "line": ("#d6d6d6", "#2b2b2b"),
    "line_strong": ("#b8b8b8", "#3d3d3d"), "control": ("#858585", "#7a7a7a"),
    "primary": ("#0e0e0e", "#f1f1f1"), "primary_fg": ("#ffffff", "#0a0a0a"),
    "primary_soft": ("#e2e2e2", "#2e2e2e"), "accent": ("#2463c4", "#7fb1f7"), "highlight": ("#b87508", "#e2a640"),
    "success": ("#17703c", "#6dcb93"), "warn": ("#8f5a0b", "#e6b765"), "danger": ("#b02d27", "#f5928a"),
    "focus": ("#0e0e0e", "#f1f1f1"),
    # the v1/v2 names are aliases of the v3 tokens
    "panel": ("#f6f6f6", "#111111"), "border": ("#d6d6d6", "#2b2b2b"), "text": ("#0e0e0e", "#f1f1f1"),
    "text2": ("#474747", "#b1b1b1"), "good": ("#17703c", "#6dcb93"), "bad": ("#b02d27", "#f5928a"),
}
SERIES = ["#2463c4", "#d9730d", "#2f855a", "#b8322b", "#6b46c1", "#5f6b7a"]
SERIES_DARK = ["#5e9df4", "#f0943f", "#4fb47b", "#f0716a", "#a68af2", "#a3acb9"]
COMPAT_KEYS = ("bg2", "panel2", "input", "border2", "faint", "accent2", "on_accent", "grid", "plot_bg", "series")

DESIGN_HOOKS = (
    'QPushButton[variant="primary"]', 'QPushButton[variant="ghost"]', 'QPushButton[variant="danger"]',
    'QPushButton[variant="link"]', 'QFrame[card="true"]', 'QLabel[role="h1"]', 'QLabel[role="h2"]',
    'QLabel[role="h3"]', 'QLabel[role="muted"]', 'QLabel[role="secondary"]', 'QLabel[role="kpi-value"]',
    'QLabel[role="kpi-label"]', 'QLabel[role="kpi-delta-good"]', 'QLabel[role="kpi-delta-bad"]',
    'QLabel[role="mono"]', 'QLabel[role="badge"]', 'QLabel[role="badge-warn"]', 'QLabel[role="badge-ok"]',
    'QLabel[role="pg-label"]', 'QAbstractItemView[flat="true"]', 'QAbstractItemView[pane="true"]',
    'QFrame[chrome="toolbar"]', 'QToolButton[role="section"]',
    # v3
    'QToolButton[variant="layer"]', 'QFrame[chrome="dock-header"]', 'QFrame[notice="warn"]',
    'QFrame[notice="bad"]', 'QFrame[notice="good"]', 'QFrame[notice="info"]', 'QFrame[callout="true"]',
    'QLabel[role="title"]', 'QFrame[splash="true"]',
)


@pytest.mark.parametrize("name", ["light", "dark"])
def test_tokens_match_guidelines(qapp, name):
    from aerolume.app import theme
    t = theme.tokens(name)
    col = 0 if name == "light" else 1
    for k, vals in GUIDE_TOKENS.items():
        assert t[k].lower() == vals[col], k
    assert t["series"] == (SERIES if name == "light" else SERIES_DARK)
    hexre = re.compile(r"^#[0-9a-fA-F]{6}$")
    for k in COMPAT_KEYS:
        vals = t[k] if k == "series" else [t[k]]
        assert all(hexre.match(v) for v in vals), (k, vals)
    assert t["plot_bg"] == t["surface"] and t["input"] == t["surface"] and t["border2"] == t["border_strong"]
    # every v1/v2 compatibility key is an opaque colour (the v2 soft keys were rgba() strings)
    for k in ("good_soft", "warn_soft", "bad_soft", "accent2_soft"):
        assert hexre.match(t[k]), (k, t[k])
    # tokens() hands out copies: mutating one must not leak into the palette
    t["series"].append("#000000")
    assert len(theme.tokens(name)["series"]) == 6
    assert theme.DEFAULT_THEME == "light" and theme.THEMES[0] == "light"


def test_text_contrast(qapp):
    """Text stays readable in both themes (WCAG AA for body text, 3:1 for large/UI)."""
    from PySide6.QtGui import QColor

    from aerolume.app import theme

    def lum(hexc):
        c = QColor(hexc)
        ch = [x / 255 for x in (c.red(), c.green(), c.blue())]
        ch = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    def contrast(a, b):
        la, lb = sorted((lum(a), lum(b)), reverse=True)
        return (la + 0.05) / (lb + 0.05)

    for name in theme.THEMES:
        t = theme.tokens(name)
        assert contrast(t["text"], t["panel"]) >= 7.0
        assert contrast(t["text2"], t["bg"]) >= 7.0
        assert contrast(t["muted"], t["surface"]) >= 4.5             # hints are text (v2 allowed 4.2)
        assert contrast(t["on_accent"], t["accent_fill"]) >= 4.5     # primary button label
        assert contrast(t["accent"], t["surface"]) >= 4.5            # link text (v2 allowed 3.0)


@pytest.mark.parametrize("name", ["light", "dark"])
def test_stylesheet_covers_every_hook(qapp, name):
    from aerolume.app import theme
    qss = theme.stylesheet(name)
    for hook in DESIGN_HOOKS:
        assert hook in qss, hook
    for widget in ("QComboBox", "QAbstractSpinBox", "QSlider", "QTabBar", "QScrollBar", "QTableView",
                   "QTreeView", "QHeaderView", "QToolTip", "QMenu", "QToolBar", "QDockWidget", "QSplitter",
                   "QGroupBox", "QProgressBar", "QStatusBar"):
        assert widget in qss, widget
    assert "$" not in qss                       # every template placeholder was substituted
    assert "gradient" not in qss and "box-shadow" not in qss
    for r in re.findall(r"border-radius:\s*(\d+)px", qss):
        assert int(r) <= 8, r                   # 6 px controls, 8 px framed blocks (the website's radii)
    # sub-control images exist on disk (combo arrows, check marks)
    for url in re.findall(r'url\("([^"]+)"\)', qss):
        assert Path(url).exists(), url


def test_apply_switches_theme_and_emits(qapp):
    from aerolume.app import theme
    seen = []
    theme.changed.connect(seen.append)
    try:
        theme.apply(qapp, "dark")
        assert theme.current() == "dark" and theme.is_dark()
        assert qapp.styleSheet() == theme.stylesheet("dark")
        assert abs(qapp.font().pointSizeF() - 9.0) < 0.01
        assert theme.toggle() == "light"
        assert seen[-2:] == ["dark", "light"]
    finally:
        theme.changed.disconnect(seen.append)
        theme.apply(qapp, "light")


def test_pyqtgraph_follows_theme(qapp):
    import pyqtgraph as pg

    from aerolume.app import theme
    theme.apply(qapp, "dark")
    assert pg.getConfigOption("background") == theme.DARK["surface"]
    assert pg.getConfigOption("foreground") == theme.DARK["text2"] == theme.DARK["fg2"]
    theme.apply(qapp, "light")
    assert pg.getConfigOption("background") == theme.LIGHT["surface"]
    ps = theme.plot_style()
    assert ps["series"] == SERIES and 0.1 <= ps["grid_alpha"] <= 0.2
    assert theme.plot_style("dark")["series"] == SERIES_DARK


def test_status_colours(qapp):
    from aerolume.app import theme
    assert theme.status_colour("done") == "good"
    assert theme.status_colour("failed") == "bad"
    assert theme.status_colour("running") == "accent"
    assert theme.status_colour("whatever") == "muted"


def test_icons_are_one_coherent_set(qapp):
    from aerolume.app import theme
    names = theme.icon_names()
    nav = ["home", "geometry", "setup", "simulate", "results", "compare", "report"]
    actions = ["new", "open", "save", "import", "run", "stop", "settings", "check", "airfoil", "car",
               "warning", "error", "info", "chip", "circle", "clock", "check-circle", "messages", "project",
               "cube", "mesh", "wing", "slice"]
    for n in nav + actions:
        assert n in names, n
    for n in names:
        svg = (theme.ICON_DIR / f"{n}.svg").read_text(encoding="utf-8")
        assert "currentColor" in svg, n           # tintable
        assert 'viewBox="0 0 16 16"' in svg, n    # one 16 px grid
        assert 'stroke-width="1.25"' in svg, n    # one stroke weight
        assert "#" not in svg.replace("currentColor", ""), n   # monochrome: no baked-in colours
        assert QSvgRenderer(svg.replace("currentColor", "#ffffff").encode()).isValid(), n
    assert QSvgRenderer(str(theme.APP_ICON_PATH)).isValid()
    app_svg = theme.APP_ICON_PATH.read_text(encoding="utf-8")
    assert "gradient" not in app_svg.lower()
    pm = theme.render_icon("airfoil", 16, "accent", 2.0)
    assert pm.width() == 32 and not pm.toImage().isNull()
    img = pm.toImage()
    assert any(img.pixelColor(x, y).alpha() > 0 for x in range(32) for y in range(32))
    ic = theme.icon("run", "accent")
    assert not ic.pixmap(16, 16).isNull()
    assert not theme.app_icon().isNull()


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def test_settings_roundtrip(isolated, tmp_path):
    from aerolume.app import settings
    s = settings.get()
    assert (s.theme, s.unit_system, s.device, s.precision) == ("light", "SI", "auto", "auto")
    keys = []
    settings.signals.changed.connect(keys.append)
    try:
        s.theme, s.unit_system, s.device, s.precision = "dark", "imperial", "cpu", "float64"
    finally:
        settings.signals.changed.disconnect(keys.append)
    assert keys == ["appearance/theme", "units/system", "compute/device", "compute/precision"]
    again = settings.Settings(s.path)
    assert (again.theme, again.unit_system, again.device, again.precision) == ("dark", "imperial", "cpu",
                                                                               "float64")
    assert again.display_unit("speed") == "mph" and again.display_unit("length") == "ft"
    with pytest.raises(ValueError):
        s.theme = "neon"
    with pytest.raises(ValueError):
        s.device = "tpu"
    # a hand-edited bad value falls back to the default instead of crashing
    s.set_value("compute/device", "quantum")
    assert s.device == "auto"


def test_settings_recent_and_last_dir(isolated, tmp_path):
    from aerolume.app import settings
    s = settings.get()
    a, b = tmp_path / "a.aerolume", tmp_path / "b.aerolume"
    a.mkdir()
    b.mkdir()
    s.add_recent(a)
    s.add_recent(b)
    s.add_recent(a)
    assert [Path(p) for p in s.recent_projects()] == [a, b]
    b.rmdir()                                   # vanished projects are hidden
    assert [Path(p) for p in s.recent_projects()] == [a]
    s.clear_recent()
    assert s.recent_projects() == []
    s.last_dir = a / "project.json"             # a file path stores its folder
    assert Path(s.last_dir) == a
    s.set_value("paths/last_dir", str(tmp_path / "missing"))
    assert Path(s.last_dir) == settings.default_projects_dir()


# ---------------------------------------------------------------------------
# shared widgets
# ---------------------------------------------------------------------------

def test_unit_line_edit(isolated):
    from aerolume.app.widgets.common import UnitLineEdit
    w = UnitLineEdit("speed", "km/h", 30.0, minimum=0.0)
    assert w.text() == "108.0 km/h"
    got = []
    w.valueChanged.connect(got.append)
    w.setText("55 mph")
    w.editingFinished.emit()
    assert got and abs(got[-1] - 55 * 0.44704) < 1e-9
    assert w.text().endswith("km/h") and w.is_valid()
    w.setText("12 furlongs")
    w.editingFinished.emit()
    assert not w.is_valid() and abs(w.value() - 55 * 0.44704) < 1e-9   # last good value kept
    w.setText("-3")
    w.editingFinished.emit()
    assert not w.is_valid()
    w.setText("36")                             # bare number = display unit
    w.editingFinished.emit()
    assert w.is_valid() and abs(w.value() - 10.0) < 1e-12
    w.set_unit("m/s")
    assert w.text() == "10.00 m/s"
    with pytest.raises(ValueError):
        UnitLineEdit("colour")


def test_common_widgets_behave(isolated, qapp):
    from aerolume.app.widgets.common import (
        Badge,
        Card,
        CollapsibleSection,
        EmptyState,
        IconButton,
        KpiTile,
        SectionHeader,
        SegmentedControl,
        Spinner,
    )
    seg = SegmentedControl([("SI", "SI"), ("imperial", "Imperial")], "imperial")
    picked = []
    seg.changed.connect(picked.append)
    assert seg.current() == "imperial"
    seg.set_current("SI")
    assert seg.current() == "SI" and picked == ["SI"]

    k = KpiTile("Drag coefficient", "0.3120", "", hint="Cd")
    k.set_delta("-4.1 %", good=True)
    assert k.delta.property("role") == "kpi-delta-good" and k.delta.isVisibleTo(k)
    k.set_delta("+2 %", good=False)
    assert k.delta.property("role") == "kpi-delta-bad"
    assert k.label.text() == "Drag coefficient"         # sentence case, no shouting
    assert k.value.property("role") == "kpi-value"

    assert Badge("Converged", "ok").property("role") == "badge-ok"
    b = Badge("x", "warn", glyph=True)
    b.set_kind("bad")
    assert b.property("role") == "badge-bad" and b.indent() > 0
    b.resize(80, 20)
    assert not b.grab().isNull()

    hits = []
    e = EmptyState("results", "No results", "Run a simulation first.", "Run", lambda: hits.append(1))
    e.button.click()
    assert hits == [1]
    assert not hasattr(e, "disc")                       # no illustration
    e.set_action(None)
    assert not e.button.isVisibleTo(e)

    c = CollapsibleSection("Advanced", QWidget(), expanded=False)
    assert not c.is_expanded()
    c.set_expanded(True)
    assert c.is_expanded() and c.content.isVisibleTo(c)

    card = Card(interactive=True)
    clicks = []
    card.clicked.connect(lambda: clicks.append(1))
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    card.resize(200, 100)
    card.show()
    QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=QPoint(50, 50))
    assert clicks == [1]
    card.hide()

    hdr = SectionHeader("Results", "subtitle")
    hdr.add_action(IconButton("save", "Save"))
    assert hdr.actions.count() == 1
    assert IconButton("save", "Save").width() == 24
    sp = Spinner()
    sp.show()
    _pump(qapp)
    sp.hide()


def test_property_grid(isolated, qapp):
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

    from aerolume.app.widgets.common import KeyValueList, MessageLine, PropertyGrid, ToolStrip, UnitLineEdit
    grid = PropertyGrid(label_width=120)
    flow = grid.add_group("Flow")
    speed = UnitLineEdit("speed", "m/s", 30.0)
    r1 = flow.add_row("Speed", speed, tooltip="Free-stream speed")
    spin = QDoubleSpinBox()
    r2 = flow.add_row("Angle of attack", spin, unit="°")
    solver = grid.add_group("Solver", expanded=False)
    solver.add_row("Scheme", QComboBox(), key="scheme")
    grid.add_stretch()
    scroll = grid.scrollable()
    scroll.resize(360, 400)
    scroll.show()
    _pump(qapp)
    assert r1.label.property("role") == "pg-label" and r1.label.toolTip() == "Free-stream speed"
    assert speed.toolTip() != ""                         # the editor keeps its own tooltip
    assert r2.unit is not None and r2.unit.text() == "°"
    assert flow.body.isVisible() and not solver.body.isVisible()
    assert grid.expanded_state() == {"Flow": True, "Solver": False}
    grid.restore_expanded({"Solver": True})
    assert solver.is_expanded() and solver.row("scheme").editor.isVisibleTo(solver)
    r2.set_visible(False)
    assert not spin.isVisibleTo(flow)
    # labels share one column: the editors line up across groups
    assert abs(speed.mapTo(grid, speed.rect().topLeft()).x()
               - solver.row("scheme").editor.mapTo(grid, solver.row("scheme").editor.rect().topLeft()).x()) <= 1
    scroll.hide()

    kv = KeyValueList([("Cells", "655 360"), ("Device", "CUDA · RTX 4060 Ti", "tooltip")])
    assert kv.value_text("Cells") == "655 360"
    kv.set_items([("Re (simulated)", "2.0×10⁴")])
    assert kv.value_text("Cells") is None and kv.value_text("Re (simulated)") == "2.0×10⁴"
    # replacement is atomic: the old labels leave the widget at once, not at the next DeferredDelete
    # (under a nested processEvents the old rows used to paint underneath the new ones)
    from PySide6.QtWidgets import QLabel
    kv.show()
    kv.set_items([("Design", "NACA 2412"), ("Angle of attack", "4°"), ("Reynolds", "1e6")])
    kv.set_items([("Design", "Sedan"), ("Pitch angle", "0°")])
    children = [lab.text() for lab in kv.findChildren(QLabel)]      # before any event processing
    assert sorted(children) == sorted(["Design", "Sedan", "Pitch angle", "0°"])
    assert kv.items() == [("Design", "Sedan"), ("Pitch angle", "0°")]
    assert kv.value_text("Reynolds") is None
    kv.hide()

    m = MessageLine("Reynolds number capped", "warning")
    assert m.text.text() == "Reynolds number capped" and not m.glyph.pixmap().isNull()

    strip = ToolStrip()
    b = strip.add_button("grid", "", "Show grid", checkable=True)
    strip.add_separator()
    strip.add_stretch()
    assert b.isCheckable() and b.toolTip() == "Show grid" and strip.property("chrome") == "toolbar"


def test_step_bar(isolated, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from aerolume.app.widgets.step_bar import STEP_KEYS, StepBar
    assert STEP_KEYS == ("geometry", "setup", "simulate", "results", "compare", "report")
    bar = StepBar()
    bar.resize(900, 30)
    bar.show()
    _pump(qapp)
    got = []
    bar.stepRequested.connect(got.append)
    r = bar.step_rect("results")
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(int(r.center().x()), int(r.center().y())))
    assert got == ["results"]
    bar.set_step_enabled("results", False)
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton, pos=QPoint(int(r.center().x()), int(r.center().y())))
    assert got == ["results"]                   # disabled steps ignore clicks
    w0 = bar.step_rect("geometry").width()
    bar.set_states({"geometry": True})
    assert bar.is_done("geometry") and bar.step_rect("geometry").width() > w0   # room for the check
    bar.set_current("setup")
    assert bar.current() == "setup"
    bar.set_current("start")                    # not a step: no underline
    assert bar.current() == ""
    bar.set_context("NACA 2412  ›  CFD · α 4°")
    assert not bar.grab().isNull()
    bar.hide()


def test_messages_panel(isolated, qapp):
    from aerolume.app.state import AppState
    from aerolume.app.widgets.messages import MessagesPanel, format_duration, normalise_level
    assert normalise_level("warn") == "warning" and normalise_level("bogus") == "info"
    assert format_duration(42) == "42 s" and format_duration(204) == "3 min 24 s"
    assert format_duration(3900) == "1 h 05 min"
    st = AppState()
    jobs = FakeJobs()
    panel = MessagesPanel(st, jobs)
    counts = []
    panel.countsChanged.connect(lambda e, w, i: counts.append((e, w, i)))
    st.log("Grid 1280×512", "info", "Planner")
    st.log("Re capped", "warn", "Planner")
    st.log("Import failed:\nbad file", "error", "Geometry")
    assert panel.counts() == {"error": 1, "warning": 1, "info": 1}
    assert counts[-1] == (1, 1, 1)
    _t, lv, src, text = panel.row(2)
    assert lv == "error" and src == "Geometry" and text == "Import failed: bad file"   # one line
    assert panel.filter_buttons["error"].text() == "1 error"
    panel.filter_buttons["info"].setChecked(False)
    assert panel.proxy.rowCount() == 2
    jobs.jobStarted.emit("sim-1")
    jobs.jobFinished.emit("sim-1", None)
    jobs.jobFailed.emit("sim-2", "diverged")
    rows = [panel.row(r) for r in range(panel.row_count())]
    assert rows[-3][3].startswith("Started") and "finished in" in rows[-2][3]
    assert rows[-1][1] == "error" and "diverged" in rows[-1][3]
    panel.view.selectAll()
    assert "diverged" in panel.copy_selection()
    panel.clear()
    assert panel.row_count() == 0 and panel.counts()["error"] == 0


@needs_store
def test_messages_follow_the_selected_simulation(isolated, qapp):
    """Rows are tied to a simulation; the default scope hides other simulations' rows."""
    from aerolume.app import settings
    from aerolume.app.widgets.messages import MessagesPanel, in_scope
    from aerolume.model import Simulation
    assert in_scope("a", "warning", "Planner", True, "b") is False
    assert in_scope("a", "error", "Run", True, "b") is True       # a failed run always shows
    assert in_scope("", "warning", "Project", True, "b") is True  # general lines always show
    assert in_scope("a", "warning", "Planner", True, "") is True  # nothing selected: show all
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.new_study("blank", "Scope", isolated)
    d = _plate_design("Plate")
    w.state.add_design(d)
    a, b = Simulation("Wing A", d.id), Simulation("Wing B", d.id)
    w.state.add_simulation(a)
    w.state.add_simulation(b)
    panel: MessagesPanel = w.messages
    panel.clear()
    assert panel.current_only() and panel.scope_button.isChecked()
    w.state.select_simulation(a.id)
    w.state.log("Grid is coarse for this Re", "warning", "Planner")     # no name: the selected one
    w.state.log("“Wing B”: 25 % done", "info", "Solver")                # a quoted name wins
    w.state.log("Opened Scope", "info", "Project")                      # general
    assert [panel.row_simulation(r) for r in range(3)] == [a.id, b.id, ""]
    assert panel.counts() == {"error": 0, "warning": 1, "info": 1}
    assert panel.filter_buttons["warning"].text() == "1 warning"
    w.state.select_simulation(b.id)
    assert panel.counts() == {"error": 0, "warning": 0, "info": 2}      # A's warning is not B's
    texts = [t for *_x, t in panel.visible_rows()]
    assert texts == ["“Wing B”: 25 % done", "Opened Scope"]
    # a failure of A still shows while B is selected, and names A in its Source cell
    jobs.jobFailed.emit(a.id, "diverged")
    assert panel.counts()["error"] == 1
    src = panel.proxy.index(panel.proxy.rowCount() - 1, 2).data()
    assert src == "Run · Wing A"
    assert panel.latest.level() == "error"
    # the filter off shows everything, and remembers that choice
    panel.set_current_only(False)
    assert panel.counts() == {"error": 1, "warning": 1, "info": 2}
    assert panel.proxy.index(0, 2).data() == "Planner · Wing A"
    assert settings.get().messages_current_only is False
    panel.set_current_only(True)
    assert settings.get().messages_current_only is True
    w.close()


def test_message_filters_are_flat_in_both_themes(isolated, qapp):
    """The level toggles must not render as filled chips (they did in the dark theme)."""
    from aerolume.app import theme
    for name in ("light", "dark"):
        qss = theme.stylesheet(name)
        m = re.search(r'QToolButton\[variant="toggle"\]:checked \{([^}]*)\}', qss)
        assert m and "background: transparent" in m.group(1), name


# ---------------------------------------------------------------------------
# start page helpers
# ---------------------------------------------------------------------------

def test_project_naming_helpers(tmp_path):
    from aerolume.app.pages import start
    assert start.safe_name('a<b>:c"/d|e?*') == "abcde"
    assert start.safe_name("   ") == "Untitled"
    assert start.project_folder(tmp_path, "Wing").name == "Wing.aerolume"
    (tmp_path / "Wing.aerolume").mkdir()
    assert start.unique_project_name(tmp_path, "Wing") == "Wing 2"
    assert start.format_modified(0) == ""
    assert start.display_folder(Path.home() / "Documents" / "Aerolume") == str(Path("~/Documents/Aerolume"))
    assert re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d$", start.format_modified(1.7e9))


@needs_store
def test_create_study_airfoil_and_vehicle(isolated):
    from aerolume.app import settings
    from aerolume.app.pages import start
    settings.get().device = "cpu"
    proj, msgs = start.create_study("airfoil", "Wing", isolated)
    root = start.project_path(proj)
    assert root.name == "Wing.aerolume" and (root / "project.json").exists()
    if proj.designs:
        d = next(iter(proj.designs.values()))
        sims = list(proj.simulations.values())
        # the CFD run first: the first simulation is the one selected when the project opens,
        # so the first F5 shows a live flow instead of ~9 s of an empty polar sweep
        assert [s.analysis for s in sims] == ["cfd", "polar"]
        re_ = sims[1].conditions.reynolds(d.length_ref)
        assert abs(re_ - 1e6) / 1e6 < 1e-6
        assert sims[0].cfd.device == "cpu"
        assert sims[0].cfd.quality == "draft"          # a first run that finishes in seconds
        assert sims[0].cfd.reynolds_mode == "capped" and sims[0].cfd.reynolds_cap == 2500.0
        assert all(s.design_id == d.id for s in sims)
    else:
        assert msgs and "Geometry" in msgs[0]
    with pytest.raises(FileExistsError):
        start.create_study("airfoil", "Wing", isolated)

    veh, msgs = start.create_study("vehicle", "Car", isolated)
    if veh.designs:
        (sim,) = veh.simulations.values()
        assert sim.analysis == "cfd" and sim.conditions.speed == 30.0
        assert sim.ground.enabled and sim.ground.moving
        assert sim.cfd.quality == "draft"
    else:
        assert msgs
    blank, msgs = start.create_study("blank", "Empty", isolated)
    assert not blank.designs and not msgs
    # every created project is remembered; remove_recent forgets one
    recents = {Path(p).resolve() for p in start.recent_projects(10)}
    assert start.project_path(proj).resolve() in recents
    start.remove_recent(start.project_path(proj))
    assert start.project_path(proj).resolve() not in {Path(p).resolve() for p in start.recent_projects(10)}


@needs_store
def test_open_project_and_examples(isolated, tmp_path, monkeypatch):
    from aerolume.app.pages import start
    proj, _ = start.create_study("blank", "Example One", tmp_path / "examples" / "projects")
    src = start.project_path(proj)
    monkeypatch.setattr(start, "examples_dirs", lambda: [tmp_path / "examples" / "projects"])
    assert [p.name for p in start.example_projects()] == [src.name]
    opened = start.open_example(src, dest_root=tmp_path / "copies")
    copy = start.project_path(opened)
    assert copy.parent == (tmp_path / "copies").resolve() or copy.parent == tmp_path / "copies"
    assert copy != src and (src / "project.json").exists()
    again = start.open_project(copy / "project.json")       # the index file is accepted too
    assert start.project_name(again) == "Example One"
    s = start.summarize(copy)
    assert s.name == "Example One" and s.designs == 0


@needs_store
def test_start_page_lists(isolated, qapp, tmp_path, monkeypatch):
    from aerolume.app.pages import start
    page = start.StartPage(None)
    assert not page.recent.isVisibleTo(page) and page.recent_empty.isVisibleTo(page)
    proj, _ = start.create_study("airfoil", "Listed", isolated)
    page.refresh()
    assert page.recent_paths() == [str(start.project_path(proj))]
    item = page.recent.topLevelItem(0)
    assert item.text(0) == "Listed" and item.text(2) == start.display_folder(isolated)
    assert item.text(1) == "Airfoil (2D)"                   # the type column: 2D, 3D or wing
    assert str(isolated) in item.toolTip(0)
    opened = []
    page.openPathRequested.connect(opened.append)
    page._recent_clicked(item)
    assert opened == [str(start.project_path(proj))]
    asked = []
    page.newStudyRequested.connect(asked.append)
    page.link_vehicle.click()
    page.handle_action("new-blank")
    assert asked == ["vehicle", "blank"]
    helps = []
    page.helpRequested.connect(helps.append)
    for topic in ("guide", "validation", "licence"):
        page.footer_links[topic].click()
    assert helps == ["guide", "validation", "licence"]
    # ui_guidelines 4: Documentation · Validation · Licence, each help link once on the page
    assert [b.text() for b in page.footer_links.values()] == ["Documentation", "Validation", "Licence"]
    from PySide6.QtWidgets import QPushButton
    texts = [b.text() for b in page.findChildren(QPushButton)]
    assert "User guide" not in texts and "Validation cases" not in texts and "Scripting guide" in texts
    # secondary columns follow the theme (they were left in the light theme's grey on dark)
    from aerolume.app import theme
    theme.apply(qapp, "dark")
    try:
        assert page.recent.topLevelItem(0).foreground(1).color() == theme.qcolor("text2")
    finally:
        theme.apply(qapp, "light")
    assert page.recent.topLevelItem(0).foreground(1).color() == theme.qcolor("text2")
    monkeypatch.setattr(start, "examples_dirs", lambda: [isolated / "no-examples"])
    page.refresh()
    assert not page.examples.isVisibleTo(page)            # no bundled examples in this setup
    assert not page.link_examples.isVisibleTo(page)       # hidden, not greyed out with no reason
    monkeypatch.setattr(start, "examples_dirs", lambda: [isolated])
    page.refresh()
    assert page.examples.topLevelItemCount() == 1
    assert page.link_examples.isVisibleTo(page)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

def test_window_without_project(isolated, qapp):
    from aerolume.app.main_window import STEP_KEYS
    w = _window()
    w.resize(1600, 1000)
    w.show()
    _pump(qapp)
    assert w.current_page_key() == "start"
    assert not any(w.step_bar.is_step_enabled(k) for k in STEP_KEYS)
    assert not any(w.page_actions[k].isEnabled() for k in STEP_KEYS)
    # the welcome page reads like an IDE's: docks and step bar step aside
    assert not w.project_dock.isVisible() and not w.messages_dock.isVisible()
    assert not w.step_toolbar.isVisible() and w.main_toolbar.isVisible()
    w.state.navigate("results")                 # blocked until a project is open
    assert w.current_page_key() == "start"
    assert "project first" in w.statusBar().currentMessage()
    assert not w.act_save.isEnabled() and not w.act_run.isEnabled() and not w.act_stop.isEnabled()
    w.state.message("hello", 0)
    assert w.statusBar().currentMessage() == "hello"
    # no navigation rail, no top-bar combos, no device chip any more
    for gone in ("rail", "topbar"):
        assert not hasattr(w, gone)
    w.close()


@needs_store
def test_window_navigates_every_page(isolated, qapp):
    from aerolume.app.main_window import STEP_KEYS, PlaceholderPage
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.resize(1600, 1000)
    w.show()
    proj = w.new_study("airfoil", "Nav test", isolated)
    _pump(qapp)
    assert w.state.project is proj
    assert w.current_page_key() == "geometry"
    assert all(w.step_bar.is_step_enabled(k) for k in STEP_KEYS)
    assert w.project_dock.isVisible() and w.messages_dock.isVisible() and w.step_toolbar.isVisible()
    assert w.windowTitle().startswith("Nav test")
    for i, key in enumerate(STEP_KEYS, start=1):
        w.page_actions[key].trigger()           # the Ctrl+<n> action
        _pump(qapp, 2)
        assert w.current_page_key() == key and w.step_bar.current() == key
        assert w.stack.currentWidget() is w.page(key)
        assert w.page_actions[key].shortcut().toString() == f"Ctrl+{i}"
        assert isinstance(w.page(key), (QWidget, PlaceholderPage))
    # "page:action" calls on_shown then handle_action
    dummy = DummyPage()
    w._pages["compare"] = dummy
    w.stack.addWidget(dummy)
    w.state.navigate("compare:rank")
    assert w.current_page_key() == "compare" and dummy.shown == 1 and dummy.actions == ["rank"]
    # F5 runs the selected simulation from any step (through the Simulate page)
    w._pages["simulate"] = sim = DummyPage()
    w.stack.addWidget(sim)
    w.show_page("setup")
    assert w.act_run.shortcut().toString() == "F5" and w.act_run.isEnabled()
    w.act_run.trigger()
    assert w.current_page_key() == "simulate" and sim.actions == ["run"]
    # step bar clicks navigate
    w.step_bar.stepRequested.emit("results")
    assert w.current_page_key() == "results"
    w.state.navigate("nowhere")
    assert "Unknown page" in w.statusBar().currentMessage()
    # the welcome page hides the workspace; coming back restores it
    w.act_welcome.trigger()
    assert w.current_page_key() == "start" and not w.project_dock.isVisible()
    w.state.navigate("geometry")
    assert w.project_dock.isVisible()
    w.close()
    assert jobs.shutdown_called == 1
    assert sim.shutdowns == 1                   # pages with shutdown() are stopped on close


def test_broken_page_becomes_placeholder(isolated, qapp, monkeypatch):
    import importlib

    from aerolume.app import main_window as mw
    real = importlib.import_module

    def fake(name, *a, **k):
        if name == "aerolume.app.pages.report":
            raise RuntimeError("boom")
        return real(name, *a, **k)

    monkeypatch.setattr(mw.importlib, "import_module", fake)
    w = _window()
    page = w.show_page("report")
    assert isinstance(page, mw.PlaceholderPage)
    assert "boom" in page.empty.text.text()
    assert w.messages.counts()["error"] == 1           # logged to Messages too
    monkeypatch.setattr(mw.importlib, "import_module", lambda name, *a, **k: type(
        "M", (), {"ReportPage": DummyPage}))
    page.retryRequested.emit("report")
    assert isinstance(w.page("report"), DummyPage)
    assert w.stack.currentWidget() is w.page("report")
    w.close()


@needs_store
def test_project_tree_follows_state(isolated, qapp):
    from aerolume.app.widgets.project_tree import KIND_ROLE
    from aerolume.model import Simulation
    w = _window()
    w.resize(1600, 1000)
    w.show()
    proj = w.new_study("blank", "Tree", isolated)
    tree = w.project_tree
    assert tree._stack.currentWidget() is tree.empty            # no designs yet
    d1, d2 = _plate_design("Plate A"), _plate_design("Plate B")
    w.state.add_design(d1)
    w.state.add_design(d2)
    s1, s2 = Simulation("A run", d1.id), Simulation("B run", d2.id)
    w.state.add_simulation(s1)
    w.state.add_simulation(s2)
    _pump(qapp)
    assert tree._stack.currentWidget() is tree.view
    assert tree.model.rowCount() == 2 and tree.item(d1.id).rowCount() == 1
    assert tree.item(s1.id).data(KIND_ROLE) == "sim"
    # AppState -> tree
    assert tree.selected() == ("sim", s2.id)
    w.state.select_simulation(s1.id)
    assert tree.selected() == ("sim", s1.id)
    # tree -> AppState: picking design B selects its first simulation
    tree.view.setCurrentIndex(tree.item(d2.id).index())
    assert w.state.design_id == d2.id and w.state.simulation_id == s2.id
    tree.view.setCurrentIndex(tree.item(s1.id).index())
    assert w.state.simulation_id == s1.id and w.state.design_id == d1.id
    assert "Plate A" in w.step_bar.context() and "A run" in w.step_bar.context()
    # inline rename writes through AppState
    tree.item(s1.id).setText("Renamed run")
    _pump(qapp)                                                  # the write-through is deferred
    assert proj.simulations[s1.id].name == "Renamed run"
    assert tree.item(s1.id).text() == "Renamed run"
    tree.item(d1.id).setText("   ")                              # blank names are refused
    _pump(qapp)
    assert proj.designs[d1.id].name == "Plate A" and tree.item(d1.id).text() == "Plate A"
    # duplicate a simulation and a design
    dup = tree.duplicate(s1.id)
    assert dup in proj.simulations and proj.simulations[dup].name == "Renamed run (copy)"
    assert w.state.simulation_id == dup
    ddup = tree.duplicate(d1.id)
    assert proj.designs[ddup].name == "Plate A (copy)"
    # delete asks first
    tree._confirm = lambda *_a: False
    assert not tree.delete(dup) and dup in proj.simulations
    tree._confirm = lambda *_a: True
    assert tree.delete(dup) and dup not in proj.simulations
    assert tree.delete(d2.id) and d2.id not in proj.designs and s2.id not in proj.simulations
    # context menus offer the documented actions
    texts = [a.text() for a in tree.build_menu(tree.item(s1.id).index()).actions() if a.text()]
    assert texts == ["Run", "Open setup", "Show results", "Rename", "Duplicate", "Delete"]
    texts = [a.text() for a in tree.build_menu(tree.item(d1.id).index()).actions() if a.text()]
    assert texts == ["New simulation", "Open in Geometry", "Rename", "Duplicate", "Delete"]
    from PySide6.QtCore import QModelIndex
    texts = [a.text() for a in tree.build_menu(QModelIndex()).actions() if a.text()]
    assert texts[0] == "Import 2D outline…" and texts[1] == "Import 3D mesh…"
    # step check marks follow the data
    steps = w.step_states()
    assert steps["geometry"] and steps["setup"] and not steps["simulate"]
    s1.status = "done"
    w.state.update_simulation(s1)
    w.state.select_simulation(s1.id)
    assert w.step_bar.is_done("simulate") and w.step_bar.is_done("results")
    assert not w.step_bar.is_done("compare")
    w.close()


@needs_store
def test_jobs_drive_tree_status_and_messages(isolated, qapp):
    from aerolume.app.widgets.project_tree import STATUS_ROLE, SUFFIX_ROLE
    from aerolume.model import Simulation
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.new_study("blank", "Jobs", isolated)
    d = _plate_design("Plate")
    w.state.add_design(d)
    a, b = Simulation("Run A", d.id), Simulation("Run B", d.id)
    w.state.add_simulation(a)
    w.state.add_simulation(b)
    w.show_page("setup")
    rs = w.run_status
    assert not rs.isVisibleTo(w)
    jobs.queue = [a.id, b.id]
    jobs.jobQueued.emit(a.id)
    jobs.jobQueued.emit(b.id)
    assert w.project_tree.item(b.id).data(SUFFIX_ROLE) == "queued"
    jobs.queue, jobs.running = [b.id], a.id
    jobs.jobStarted.emit(a.id)
    jobs.jobProgress.emit(a.id, 0.42, "t = 17.2 / 40")
    assert rs.isVisibleTo(w)
    assert rs.summary() == "Running Run A  42 %  ·  t = 17.2 / 40  ·  1 queued"
    item = w.project_tree.item(a.id)
    assert item.data(STATUS_ROLE) == "running" and item.data(SUFFIX_ROLE) == "42 %"
    assert w.act_stop.isEnabled()
    w.act_stop.trigger()
    assert jobs.cancelled == [a.id]
    assert w.statusBar().currentMessage() == "Stopping…"
    jobs.running = None
    a.status = "done"
    jobs.jobFinished.emit(a.id, None)
    assert w.project_tree.item(a.id).data(STATUS_ROLE) == "done"
    jobs.queue, jobs.running = [], b.id
    jobs.jobStarted.emit(b.id)
    assert rs.summary().startswith("Running Run B  0 %")
    assert w.statusBar().currentMessage() == ""         # "Stopping…" does not outlive the stop
    jobs.running = None
    b.status = "failed"
    w.messages_dock.hide()
    jobs.jobFailed.emit(b.id, "diverged")
    assert rs.summary() == "Run B: failed — see Messages" and rs.text.property("role") == "status-bad"
    assert not w.messages_dock.isHidden()       # a failure brings the log back
    logged = [w.messages.row(r) for r in range(w.messages.row_count())]
    assert any(lv == "error" and "diverged" in text for _t, lv, _s, text in logged)
    assert any(text.startswith("“Run A” finished") for *_x, text in logged)
    assert not w.act_stop.isEnabled()
    w.close()


@needs_store
def test_run_follows_the_design_on_screen(isolated, qapp):
    """Selecting or adding a design never leaves another design's simulation current.

    Regression: adding a Cylinder from a template kept NACA 2412's polar
    current, the breadcrumb read "Cylinder › Polar · Re 1.0×10⁶" and F5 ran
    the NACA polar.
    """
    from aerolume.model import Simulation
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.new_study("blank", "Run target", isolated)
    w._pages["simulate"] = sim_page = DummyPage()
    w.stack.addWidget(sim_page)
    w._pages["setup"] = setup_page = DummyPage()
    w.stack.addWidget(setup_page)
    st = w.state
    assert not w.act_run.isEnabled()                         # no design yet
    a = _plate_design("Plate A")
    st.add_design(a)
    sa1, sa2 = Simulation("A one", a.id), Simulation("A two", a.id)
    st.add_simulation(sa1)
    st.add_simulation(sa2)
    assert (st.design_id, st.simulation_id) == (a.id, sa2.id)
    assert w.act_run.isEnabled() and w.act_run.iconText() == "Run" and "A two" in w.act_run.toolTip()
    # a new design (template / import / NACA all go through add_design -> select_design)
    b = _plate_design("Plate B")
    st.add_design(b)
    assert st.design_id == b.id and st.simulation_id == ""
    assert w.step_bar.context() == "Plate B"
    assert w.project_tree.selected() == ("design", b.id)
    assert w.act_run.isEnabled() and w.act_run.iconText() == "Set up simulation…"
    w.act_run.trigger()
    assert w.current_page_key() == "setup" and setup_page.actions == ["new"]
    assert sim_page.actions == []                            # nothing of Plate A was run
    # back to A by design: its first simulation becomes current, Run runs it
    st.select_design(a.id)
    assert st.simulation_id == sa1.id and w.act_run.iconText() == "Run"
    w.act_run.trigger()
    assert w.current_page_key() == "simulate" and sim_page.actions == ["run"]
    # "Open in Geometry" from the tree's context menu goes through the same rule
    menu = w.project_tree.build_menu(w.project_tree.item(b.id).index())
    next(x for x in menu.actions() if x.text() == "Open in Geometry").trigger()
    assert st.design_id == b.id and st.simulation_id == ""
    # a selection that is somehow incoherent is refused rather than run
    st.add_simulation(Simulation("B one", b.id), select=False)
    st.simulation_id = sa1.id
    w._update_run_actions()
    assert not w.act_run.isEnabled()
    n = len(sim_page.actions)
    w.run_current()
    assert len(sim_page.actions) == n and "Plate B" in w.statusBar().currentMessage()
    w.close()


@needs_store
def test_messages_dock_folds_until_needed(isolated, qapp):
    from aerolume.app import settings
    from aerolume.app.main_window import DockTitle
    w = _window()
    w.resize(1280, 800)
    w.show()
    w.new_study("blank", "Messages", isolated)
    _pump(qapp)
    dock = w.messages_dock
    # starts as its title row: counters + the newest line, no table
    assert not w.messages_expanded() and dock.height() <= DockTitle.HEIGHT
    assert w.messages.latest.isVisibleTo(w) and "Opened Messages" in w.messages.latest.full_text()
    assert w.messages.latest.width() > 300                    # the line gets the free width
    assert not w.messages.view.isVisibleTo(w)
    w.state.log("just a note", "info", "Test")
    assert not w.messages_expanded()
    # a warning waits in the title row (counted, in the warning colour): it no longer takes the page's height,
    # and the planner's notes, which Setup / Simulate / Results show themselves, never open the log
    w.state.log("grid is coarse for this Reynolds number", "warning", "Setup")
    w.state.log("Reynolds number capped", "warning", "Planner")
    w.state.log("refused: the grid does not fit", "error", "Planner")
    _pump(qapp)
    assert not w.messages_expanded() and dock.height() <= DockTitle.HEIGHT
    assert w.messages.latest.property("role") == "status-bad"
    # an error opens it
    w.state.log("solver diverged", "error", "Run")
    _pump(qapp)
    assert w.messages_expanded() and dock.height() > DockTitle.HEIGHT + 60
    assert w.messages.view.isVisibleTo(w) and not w.messages.latest.isVisibleTo(w)
    # the user drags it taller; that height is kept for this window size
    from PySide6.QtCore import Qt
    w.resizeDocks([dock], [230], Qt.Orientation.Vertical)
    _pump(qapp)
    assert w.messages_height() == dock.height()
    w.messages_toggle.click()                                 # the user folds it
    _pump(qapp)
    assert not w.messages_expanded() and dock.height() <= DockTitle.HEIGHT
    w.state.log("grid is coarse", "warning", "Setup")         # folded by the user: a warning waits
    assert not w.messages_expanded()
    assert w.messages.latest.property("role") == "status-warn"
    w.messages.latest.clicked.emit()                          # clicking the line opens the log
    _pump(qapp)
    assert w.messages_expanded() and abs(dock.height() - w.messages_height()) <= 2
    w.save_window_state()
    stored = str(settings.get().value("window/messages_heights", ""))
    assert f"{w._messages_bucket()}:" in stored
    w.close()


def test_item_view_check_boxes_are_styled(isolated, qapp):
    """Check boxes inside lists match QCheckBox instead of falling back to Fusion."""
    from aerolume.app import theme
    for name in ("light", "dark"):
        qss = theme.stylesheet(name)
        assert "QAbstractItemView::indicator {" in qss
        assert "QAbstractItemView::indicator:checked" in qss and "QAbstractItemView::indicator:disabled" in qss
        assert "$" not in qss.split("QAbstractItemView::indicator", 1)[1].split("}", 1)[0]


def test_preferences_dialog_applies(isolated, qapp):
    from aerolume.app import settings, theme
    w = _window()
    dlg = w.open_preferences(exec_=False)
    assert dlg.values() == {"theme": "light", "ui_font": "bundled", "unit_system": "SI", "device": "auto",
                            "precision": "auto", "update_url": ""}
    assert dlg.device.findData("cpu") >= 0
    dlg.theme.set_current("dark")
    dlg.units.set_current("imperial")
    dlg.precision.setCurrentIndex(dlg.precision.findData("float32"))
    changed = dlg.apply()
    assert changed == {"theme": "dark", "unit_system": "imperial", "precision": "float32"}
    s = settings.get()
    assert (s.theme, s.unit_system, s.precision) == ("dark", "imperial", "float32")
    assert theme.current() == "dark" and w.theme_actions["dark"].isChecked()
    assert w._units_label.text() == "imperial" and "mph" in w._units_label.toolTip()
    w.set_theme("light")
    assert theme.current() == "light" and settings.get().theme == "light"
    # the interface font is a preference too: System gives the platform font back
    dlg = w.open_preferences(exec_=False)
    dlg.font.set_current("system")
    assert dlg.apply() == {"ui_font": "system"}
    assert settings.get().ui_font == "system" and theme.font_choice() == "system"
    assert theme.font_family() != theme.BUNDLED_UI_FAMILY
    theme.apply(qapp, "light", font="bundled")
    w.close()


def test_device_label():
    from aerolume.app.main_window import device_label
    assert device_label("cuda", "NVIDIA GeForce RTX 4060 Ti (8.0 GB)") == "CUDA · RTX 4060 Ti"
    assert device_label("cpu", "AMD Ryzen 7") == "CPU"
    assert device_label("mps", "Apple M2") == "MPS · Apple M2"


def test_about_and_help(isolated, qapp):
    w = _window()
    about = w.show_about(exec_=False)
    names = dict(about.version_rows)
    assert names["Aerolume"] and "PySide6 / Qt" in names
    assert "mfoil" in about._third_party_text() and "MIT" in about._third_party_text()
    assert "Apache License" in about._licence_text()
    for topic in ("guide", "validation", "licence"):
        p = w.help_path(topic)
        assert p is None or p.exists()
    assert w.help_path("licence") is not None          # LICENSE ships with the repo
    from PySide6.QtWidgets import QTreeWidget

    from aerolume.app.main_window import SHORTCUTS
    tree = w.shortcuts_dialog().findChild(QTreeWidget)
    assert tree.topLevelItemCount() == len(SHORTCUTS)
    assert dict(SHORTCUTS)["Delete"] == "Delete the selected design or simulation"
    w.close()


def test_menus_and_shortcuts(isolated, qapp):
    w = _window()
    titles = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Simulation", "Tools", "Help"]
    assert w.act_new.shortcut().toString() == "Ctrl+N"
    assert w.act_open.shortcut().toString() == "Ctrl+O"
    assert w.act_save.shortcut().toString() == "Ctrl+S"
    assert w.act_import.shortcut().toString() == "Ctrl+I"
    assert w.act_run.shortcut().toString() == "F5"
    assert w.act_stop.shortcut().toString() == "Shift+F5"
    # sentence-case menu items
    for a in (w.act_new, w.act_open, w.act_import, w.act_new_sim, w.act_guide):
        words = a.text().replace("&", "").rstrip("…").split()
        assert all(x[0].islower() or x in ("NACA",) for x in words[1:]), a.text()
    # toolbar: New Open Save | Import geometry | Run Stop | Report | Assistant Script (panel toggles),
    # then (after a spacer) the workspace switcher at the right end
    tb = [a.iconText() if not a.isSeparator() else "|" for a in w.main_toolbar.actions()]
    assert tb == ["New study", "Open project", "Save", "|", "Import geometry", "|", "Run", "Stop", "|", "Report",
                  "|", "Assistant", "Script", "", "Default"]
    # View: the steps, then every panel under a "Panels" heading (Alt+1 ...), workspaces, theme
    view = w.view_menu
    texts = [a.text() for a in view.actions()]
    assert "Panels" in texts and "&Workspace" in texts
    shortcuts = {a.text(): a.shortcut().toString() for a in view.actions()}
    assert shortcuts["Project panel"] == "Alt+1" and shortcuts["Messages panel"] == "Alt+2"
    assert shortcuts["Diagnostics panel"] == "Alt+7"
    helps = [a.text() for a in w.help_menu.actions()]
    assert "Driving Aerolume from &Claude (MCP)" in helps and "Welcome &tour" in helps
    w._fill_recent_menu()
    w._fill_examples_menu()
    assert w.recent_menu.actions() and w.examples_menu.actions()
    w.close()


@needs_store
def test_close_saves_geometry_and_project(isolated, qapp):
    from aerolume.app import settings
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.resize(1300, 900)
    w.show()
    w.new_study("blank", "Close me", isolated)
    _pump(qapp)
    w.messages_dock.titleBarWidget().close_button.click()     # the user hides Messages
    assert not w.panel_wanted("messagesDock")
    w.close()
    assert jobs.shutdown_called == 1
    assert settings.get().window_geometry is not None
    stored = settings.get().window_geometry
    w2 = _window()
    # the offscreen screen is tiny, so Qt clamps the restored size; check the bytes round-trip
    assert w2.restoreGeometry(stored)
    assert not w2.panel_wanted("messagesDock") and w2.panel_wanted("projectDock")
    w2.close()


@needs_store
def test_main_build_opens_project_argument(isolated, qapp):
    from aerolume.app import theme
    from aerolume.app.main import build
    from aerolume.app.pages import start
    proj, _ = start.create_study("blank", "From CLI", isolated)
    app, win = build(["aerolume", str(start.project_path(proj)), "--theme", "dark"])
    _WINDOWS.append(win)
    try:
        assert app is qapp
        assert theme.current() == "dark"
        assert win.state.project is not None and start.project_name(win.state.project) == "From CLI"
        assert win.current_page_key() == "geometry"
    finally:
        if win.jobs is not None and hasattr(win.jobs, "shutdown"):
            win.jobs.shutdown()
        win.jobs = None
        win.close()
        theme.apply(qapp, "light")


# ---------------------------------------------------------------------------
# screenshots (visual review artefacts, written outside the repository)
# ---------------------------------------------------------------------------

@needs_store
@pytest.mark.parametrize("name", ["light", "dark"])
def test_render_screenshots(isolated, qapp, name, tmp_path):
    """Render the start page and a project shell at 1600x1000."""
    from PySide6.QtGui import QImage

    from aerolume.app import theme
    shots = Path(os.environ.get("AEROLUME_SHOTS") or tmp_path / "shots")
    shots.mkdir(parents=True, exist_ok=True)
    theme.apply(qapp, name)
    w = _window()
    w._device_ready("cuda", "CUDA · RTX 4060 Ti", "test")
    w.resize(1600, 1000)
    w.show()
    _pump(qapp, 10)
    start_png = shots / f"test_start_{name}.png"
    assert w.grab().save(str(start_png))
    w.new_study("airfoil", "Screenshot study", isolated)
    w.state.navigate("start")
    _pump(qapp, 10)
    recent_png = shots / f"test_start_recent_{name}.png"
    assert w.grab().save(str(recent_png))
    w.state.navigate("geometry")
    _pump(qapp, 10)
    shell_png = shots / f"test_shell_{name}.png"
    assert w.grab().save(str(shell_png))
    for p in (start_png, recent_png, shell_png):
        assert p.stat().st_size > 15_000          # a real render, not a blank surface
    img = QImage(str(start_png))
    assert (img.width(), img.height()) == (1600, 1000)
    w.close()
    theme.apply(qapp, "light")


# ---------------------------------------------------------------------------
# review round: creating, switching, closing
# ---------------------------------------------------------------------------

def _airfoil_design(name: str = "NACA 4412"):
    from aerolume.geometry import design_from_airfoil, naca
    return design_from_airfoil(naca(name.split()[-1], 120, 1.0), name, 1.0)


@needs_store
@pytest.mark.parametrize("how", ["f5", "menu", "tree"])
def test_setting_up_a_design_without_simulations_creates_exactly_one(isolated, qapp, how):
    """navigate('setup:new') ran on_shown (which creates one) and then handle_action('new')
    (which created a second): 'NACA 4412 · run 1' was left behind next to 'run 2'."""
    w = _window()
    w.new_study("airfoil", "One sim", isolated)
    d = _airfoil_design()
    w.state.add_design(d)
    assert not w.state.project.simulations_of(d.id)
    if how == "f5":
        w.run_current()
    elif how == "menu":
        w.act_new_sim.trigger()
    else:
        w.project_tree.new_simulation(d.id)
    sims = w.state.project.simulations_of(d.id)
    assert [s.name for s in sims] == ["NACA 4412 · run 1"] and w.state.simulation_id == sims[0].id
    # with one already there, "New simulation" still adds a second
    w.act_new_sim.trigger()
    assert len(w.state.project.simulations_of(d.id)) == 2
    w.close()


@needs_store
def test_a_second_design_starts_at_the_conditions_of_the_study(isolated, qapp):
    """The starter NACA 2412 runs at 14.6 m/s (Re 1e6); a NACA 4412 added next to it used to get
    the model default of 30 m/s, and Compare showed 4.5x the drag with no hint why."""
    w = _window()
    w.new_study("airfoil", "Two designs", isolated)
    starter = next(s for s in w.state.project.simulations.values() if s.analysis == "cfd")
    d = _airfoil_design()
    w.state.add_design(d)
    w.run_current()                                     # sets up its first simulation
    (sim,) = w.state.project.simulations_of(d.id)
    assert sim.conditions.speed == pytest.approx(starter.conditions.speed)
    assert sim.conditions.alpha_deg == pytest.approx(4.0)
    assert sim.conditions.fluid.density == pytest.approx(starter.conditions.fluid.density)
    assert sim.cfd.quality == starter.cfd.quality == "draft"
    assert (sim.cfd.reynolds_mode, sim.cfd.reynolds_cap) == (starter.cfd.reynolds_mode, starter.cfd.reynolds_cap)
    w.close()


@needs_store
def test_a_new_airfoil_study_selects_its_cfd_run(isolated, qapp):
    """The first F5 of a new study runs the CFD case (a live flow within a second), not the polar."""
    w = _window()
    w.new_study("airfoil", "First run", isolated)
    assert w.state.simulation is not None and w.state.simulation.analysis == "cfd"
    assert w.run_target() == ("none", "The simulation runner is unavailable (see Messages).") or \
        w.run_target()[1] == w.state.simulation.name
    w.close()


@needs_store
def test_switching_projects_stops_runs_only_when_the_switch_happens(isolated, qapp, monkeypatch, tmp_path):
    """r4: answering Yes and then cancelling the folder dialog stopped the run for nothing; so did
    a damaged project and re-opening the open one."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    proj = w.new_study("blank", "Busy", isolated)
    jobs.running = "sim-a"
    asked = []

    def answer(*a, **k):
        asked.append(a[2] if len(a) > 2 else "")
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    w.open_project_dialog()                                    # cancelled folder dialog
    assert jobs.cancelled == [] and asked == []
    bad = tmp_path / "Broken.aerolume"
    bad.mkdir()
    (bad / "project.json").write_text("{ not json", encoding="utf-8")
    assert w.open_project_path(bad) is None                    # a damaged project
    assert jobs.cancelled == [] and asked == []
    from aerolume.app.pages import start
    assert w.open_project_path(start.project_path(proj)) is proj   # the project that is open
    assert jobs.cancelled == [] and asked == [] and w.state.project is proj
    other, _ = start.create_study("blank", "Other", isolated)
    w.open_project_path(start.project_path(other))
    assert asked and "running" in asked[0] and jobs.cancelled == ["sim-a"]
    assert start.project_name(w.state.project) == "Other"
    w.close()


@needs_store
def test_quitting_asks_before_stopping_a_simulation(isolated, qapp, monkeypatch):
    """r4: closing the window stopped a running simulation without asking (project switching asked)."""
    from PySide6.QtWidgets import QMessageBox
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.new_study("blank", "Quit", isolated)
    w.show()
    jobs.running, jobs.queue = "sim-a", ["sim-b"]
    asked = []
    reply = {"v": QMessageBox.StandardButton.No}
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: asked.append(a[2]) or reply["v"])
    w.act_exit.trigger()                                       # File > Exit / Ctrl+Q
    assert asked and "Stop it and quit" in asked[0] and "1 more is queued" in asked[0]
    assert w.isVisible() and jobs.shutdown_called == 0         # No: nothing was stopped
    w.close()                                                  # the title bar's close button
    assert len(asked) == 2 and w.isVisible() and jobs.shutdown_called == 0
    reply["v"] = QMessageBox.StandardButton.Yes
    w.act_exit.trigger()
    assert not w.isVisible() and jobs.shutdown_called == 1


@needs_store
def test_a_failed_save_on_close_leaves_the_runs_alone(isolated, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    jobs = FakeJobs()
    w = _window(jobs=jobs)
    w.new_study("blank", "Save fails", isolated)
    w.show()

    def broken() -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(w.state, "save", broken)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.No)
    w.close()                                                   # "Close anyway?" -> No
    assert w.isVisible() and jobs.shutdown_called == 0
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Yes)
    w.close()
    assert not w.isVisible() and jobs.shutdown_called == 1


@needs_store
def test_status_message_never_paints_over_ready_and_reports_tick_the_step(isolated, qapp):
    w = _window()
    proj = w.new_study("blank", "Status", isolated)            # "Created …" before the window shows
    assert w.statusBar().currentMessage() and w._ready_label.isHidden()
    w.statusBar().clearMessage()
    assert not w._ready_label.isHidden()
    from aerolume.app.pages import start
    assert not w.step_bar.is_done("report")
    reports = start.project_path(proj) / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "r.pdf").write_bytes(b"%PDF-1.4\n")
    w.state.reportsChanged.emit()                              # what ReportPage._done emits
    assert w.step_bar.is_done("report")
    w.close()


def test_user_guide_images_are_scaled_to_the_viewer(isolated, qapp, tmp_path):
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QColor, QImage, QTextDocument

    from aerolume.app.main_window import DocViewer
    img = QImage(1600, 400, QImage.Format.Format_RGB32)
    img.fill(QColor("#2463c4"))
    assert img.save(str(tmp_path / "wide.png"))
    md = tmp_path / "guide.md"
    md.write_text("# Guide\n\n![wide](wide.png)\n", encoding="utf-8")
    dlg = DocViewer(md, "Guide")
    dlg.resize(880, 720)
    dlg.show()
    _pump(qapp, 30)
    view = dlg.view
    view._reload()                                             # what the resize debounce does
    res = view.document().resource(QTextDocument.ResourceType.ImageResource.value, QUrl("wide.png"))
    w = res.width() if hasattr(res, "width") else QImage.fromData(bytes(res)).width()
    assert 0 < w <= view.viewport().width()
    dlg.close()


@needs_store
def test_geometry_files_dropped_anywhere_are_imported(isolated, qapp, tmp_path):
    from types import SimpleNamespace

    from PySide6.QtCore import QMimeData, QUrl
    w = _window()
    w.new_study("blank", "Drops", isolated)
    dat = tmp_path / "foil.dat"
    dat.write_text("NACA test\n1 0\n0.5 0.05\n0 0\n0.5 -0.05\n1 0\n", encoding="utf-8")
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(dat)), QUrl.fromLocalFile(str(tmp_path / "notes.docx"))])
    ev = SimpleNamespace(mimeData=lambda: md)
    assert w._dropped_geometry(ev) == [str(dat).replace("\\", "/")] or \
        [str(p) for p in map(Path, w._dropped_geometry(ev))] == [str(dat)]
    nav = []
    w.state.navigate = nav.append                  # the import dialog itself is modal: record only
    accepted = []
    ev2 = SimpleNamespace(mimeData=lambda: md, acceptProposedAction=lambda: accepted.append(True),
                          ignore=lambda: None)
    w.dropEvent(ev2)
    assert accepted and nav and nav[0].startswith("geometry:import:") and nav[0].endswith("foil.dat")
    w.close()


@needs_store
def test_editing_a_design_keeps_it_selected(isolated, qapp):
    """Regression: the tree rebuild after any design edit moved Qt's current index row by row
    while it cleared the model, and the tree passed those moves on to AppState: renaming or
    transforming the first of several designs selected another one."""
    from aerolume.model import Simulation
    w = _window()
    w.resize(1280, 800)
    w.show()
    w.new_study("blank", "Keep selection", isolated)
    a, b, c = _plate_design("Plate A"), _plate_design("Plate B"), _plate_design("Plate C")
    for d in (a, b, c):
        w.state.add_design(d)
    w.state.add_simulation(Simulation("C run", c.id))
    w.state.select_design(a.id)
    assert (w.state.design_id, w.state.simulation_id) == (a.id, "")
    a.name = "Plate A2"
    w.state.update_design(a)
    assert w.state.design_id == a.id and w.project_tree.selected() == ("design", a.id)
    assert w.step_bar.context() == "Plate A2"
    w.state.add_simulation(Simulation("B run", b.id), select=False)
    assert w.state.design_id == a.id
    w.close()
