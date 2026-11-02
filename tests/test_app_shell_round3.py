"""UI critique round 3 fixes in the shell: Preferences ▸ Account and System theme, rail caps,
Diagnostics beside Messages, tool windows on the Start page, the tour box and the stored
diagnostics fallback. Offscreen, private settings, as the other shell tests."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QRect, Signal

pytestmark = pytest.mark.gui


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
    w.new_study("airfoil", f"Round3 {size[0]}", isolated)
    _pump(qapp, 8)
    return w


# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------

def test_preferences_has_account_tab_system_theme_and_fits_its_content(isolated, qapp):
    from aerolume.app import settings, theme
    w = _window()
    dlg = w.open_preferences(exec_=False, tab="account")
    assert dlg.TABS == ("general", "assistant", "account")
    assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == ["General", "Assistant", "Account"]
    assert dlg.tabs.currentIndex() == 2
    from aerolume.app.dialogs.account_dialogs import AccountSettingsPage
    assert isinstance(dlg.account, AccountSettingsPage)
    # sized to the tab on screen: General no longer sits above 420 px of nothing
    heights = {}
    for i, key in enumerate(dlg.TABS):
        dlg.tabs.setCurrentIndex(i)
        _pump(qapp, 2)
        heights[key] = dlg.height()
        need = max(dlg.sizeHint().height(), dlg.minimumSizeHint().height())
        assert dlg.height() <= need + 2 and dlg.width() >= 580, key
    assert dlg.page_widget("account") is dlg.account and dlg.page_widget("nope") is None
    spare = heights["general"] - dlg.page_widget("general").sizeHint().height()
    assert spare < 200, heights                    # tab bar, footer, buttons: no empty field of space
    # System is a theme choice
    from PySide6.QtWidgets import QPushButton
    assert "System" in [b.text() for b in dlg.theme.findChildren(QPushButton)]
    dlg.theme.set_current("system")
    changed = dlg.apply()
    assert changed.get("theme") == "system" and settings.get().theme == "system"
    assert theme.preference() == "system" and theme.current() in ("light", "dark")
    assert w.theme_actions["system"].isChecked()
    w.set_theme("light")
    assert theme.preference() == "light" and w.theme_actions["light"].isChecked()
    dlg.close()


def test_an_invalid_account_entry_keeps_preferences_open_on_its_tab(isolated, qapp):
    w = _window()
    dlg = w.open_preferences(exec_=False)
    dlg.account.url.setText("https://abc.supabase.co")        # a URL without its key is refused
    dlg.account.key.clear()
    dlg.show()
    dlg.accept()
    _pump(qapp)
    assert dlg.isVisible() and dlg.tabs.currentIndex() == 2 and dlg.account.last_error
    dlg.account.url.clear()
    dlg.accept()                                               # both empty: offline, accepted
    _pump(qapp)
    assert not dlg.isVisible() and not dlg.account.last_error


# ---------------------------------------------------------------------------
# rails and the bottom rail
# ---------------------------------------------------------------------------

def test_side_rails_are_capped_below_1400_px(isolated, qapp):
    from aerolume.app.widgets.docks import RAIL_CAP_SHARE, RAIL_CAP_WIDTH
    w = _shown_window(qapp, isolated, size=(1280, 800))
    d = w.docks
    d.set_workspace("Live")
    _pump(qapp, 8)
    cap = d.rail_cap(1280)
    assert cap == round(RAIL_CAP_SHARE * 1280)
    for key in ("flowvis", "diagnostics", "project", "properties"):
        p = d.panel(key)
        if not p.isHidden() and d.area_of(key) != "bottom":
            assert p.width() <= max(cap, p.minimumWidth()) + 1, (key, p.width())
    assert d.panel("messages").maximumWidth() > 10_000           # the bottom rail is never capped
    assert d.panel("diagnostics").maximumWidth() > 10_000        # Live rev 2: Diagnostics is in it
    d.show_panel("assistant")                                     # a right-rail panel on every page
    _pump(qapp, 6)
    d.set_rail_size("right", 600)                                 # a drag wider than the cap is held to it
    _pump(qapp, 4)
    shown = [k for k in d.order("right") if not d.panel(k).isHidden()]
    assert shown and d.panel(shown[0]).width() <= cap + 1
    w.resize(RAIL_CAP_WIDTH + 200, 900)                           # wide window: no cap
    _pump(qapp, 4)
    assert d.rail_cap() is None and d.panel("flowvis").maximumWidth() > 10_000


def test_live_rev2_puts_diagnostics_under_the_page_and_replaces_a_stale_saved_layout(isolated, qapp):
    """Live rev 2: Diagnostics' 592 px content goes to the bottom rail; a layout saved from rev 1
    (Diagnostics on the right) is replaced by the new preset when the settings are read back,
    while a rev-2 layout the user arranged is kept."""
    import json

    from aerolume.app import settings
    from aerolume.app.widgets.docks import WORKSPACES, preset_rev
    assert preset_rev("Live") == 2 and preset_rev("Default") == 1
    assert WORKSPACES["Live"]["panels"]["diagnostics"][1] == "bottom"
    stale = json.loads(json.dumps(WORKSPACES["Live"]))
    stale.pop("rev")
    stale["panels"]["diagnostics"] = [True, "right", False]
    stale["order"]["right"].append("diagnostics")
    stale["order"]["bottom"].remove("diagnostics")
    settings.get().set_value("workspaces/Live", json.dumps(stale))
    settings.get().set_value("workspaces/current", "Live")
    w = _shown_window(qapp, isolated, size=(1600, 1000))
    d = w.docks
    assert d.workspace == "Live" and d.area_of("diagnostics") == "bottom"
    assert d.capture()["rev"] == 2
    d.move_to("flowvis", "left")                       # a rev-2 arrangement of the user's own ...
    d.save()
    w2 = _window()
    assert w2.docks.workspace == "Live" and w2.docks.area_of("flowvis") == "left"   # ... survives a restart


def test_a_wide_panel_in_a_side_rail_never_squeezes_the_page_below_its_minimum(isolated, qapp):
    """Above 1400 px no share cap applies; the page guard still leaves the page its minimum width."""
    w = _shown_window(qapp, isolated, size=(1440, 900))
    d = w.docks
    if not d.panel("diagnostics").available():
        pytest.skip("no Diagnostics panel in this build")
    w.show_page("results")
    d.move_to("diagnostics", "right")
    d.show_panel("diagnostics")
    _pump(qapp, 10)
    from aerolume.app.widgets.docks import WORKSPACES
    central = w.centralWidget()
    page = w.stack.currentWidget()
    diag = d.panel("diagnostics")
    # the guard measures the page on screen, not the stack (whose minimum is its widest page's)
    assert d.page_min_width() < w.stack.minimumSizeHint().width() or \
        page.minimumSizeHint().width() == w.stack.minimumSizeHint().width()
    # Diagnostics with a history needs 592 px (its list beside its plots); an empty one less, so
    # give the content that minimum, and the page one that does not fit beside it at 1440 px
    diag.content().setMinimumWidth(600)
    _pump(qapp, 6)
    natural = d.natural_min_width("diagnostics")
    assert natural >= 600
    need = w.width() - natural - d.natural_min_width("project") + 120
    page.setMinimumWidth(need)
    try:
        d.cap_rails()
        _pump(qapp, 10)
        guard = d.page_guard_caps()
        assert guard["right"] is not None and guard["right"] < natural
        floor = max(diag.header.minimumSizeHint().width(), WORKSPACES[d.workspace]["sizes"]["right"])
        assert guard["right"] > floor                  # this case is the guard's, not the floor's
        assert diag.width() <= guard["right"] + 1
        assert central.width() >= need - 1, (central.width(), need)
        for width in (1400, 1520, 1600):
            w.resize(width, 900)
            _pump(qapp, 6)
            assert central.width() >= need - 1, (width, central.width(), need)
    finally:
        page.setMinimumWidth(0)
        diag.content().setMinimumWidth(0)
    w.resize(1600, 900)
    d.cap_rails()
    _pump(qapp, 6)
    assert d.page_guard_caps()["right"] is None        # room again: the guard lets go


def test_opening_diagnostics_folds_the_log(isolated, qapp):
    w = _shown_window(qapp, isolated, size=(1280, 800))
    w.set_messages_expanded(True, user=True)
    _pump(qapp)
    assert w.messages_expanded() and w.docks.area_of("diagnostics") == "bottom"
    if not w.docks.panel("diagnostics").available():
        pytest.skip("no Diagnostics panel in this build")
    w.docks.show_panel("diagnostics")
    _pump(qapp, 4)
    assert w.diagnostics_dock.isVisible() and not w.messages_expanded()
    assert w.messages.latest.isVisibleTo(w)                      # counters and newest line stay


# ---------------------------------------------------------------------------
# the Start page shows the tool windows it lights up
# ---------------------------------------------------------------------------

def test_tool_windows_stay_on_the_start_page(isolated, qapp):
    w = _shown_window(qapp, isolated)
    d = w.docks
    d.show_panel("script")
    _pump(qapp, 3)
    w.show_page("start")
    _pump(qapp, 3)
    assert w.project_dock.isHidden() and w.messages_dock.isHidden()     # project panels step aside
    assert d.actions["script"].isChecked() and w.script_dock.isVisible()  # a lit toggle = a visible panel
    d.hide_panel("script")
    assert not d.actions["script"].isChecked() and w.script_dock.isHidden()
    d.actions["assistant"].trigger()                                     # the toolbar toggle works there too
    _pump(qapp, 3)
    assert w.assistant_dock.isVisible()


# ---------------------------------------------------------------------------
# tour
# ---------------------------------------------------------------------------

def test_tour_box_never_covers_the_toolbar_it_points_at(isolated, qapp):
    from PySide6.QtCore import QPoint
    for size in ((1600, 1000), (1280, 800)):
        w = _shown_window(qapp, isolated, size=size)
        tour = w.start_tour()
        seen = {}
        for _ in range(len(tour.steps)):
            _pump(qapp, 2)
            seen[tour.current_step().key] = (tour.target_rect(), tour.box_rect())
            tour.next()
        tb = w.main_toolbar
        band = QRect(tb.mapTo(w, QPoint(0, 0)), tb.size())
        for key in ("run", "panels", "help"):
            target, box = seen[key]
            assert not target.isNull(), key
            assert not box.intersects(target) and not box.intersects(band), (key, size, box, band)
        run = seen["run"][0]
        stop = w.main_toolbar.widgetForAction(w.act_stop)
        assert run.contains(QRect(stop.mapTo(w, QPoint(0, 0)), stop.size()))   # Run and Stop outlined together
        _destroy_windows(qapp)


# ---------------------------------------------------------------------------
# Diagnostics: a finished run from an earlier session shows its stored history
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, n: int = 40) -> None:
        step = np.arange(n, dtype=float) * 10
        self.history = {"step": step, "t": step * 0.01, "cd": np.ones(n),
                        "div_rms": np.geomspace(1e-3, 1e-6, n), "cfl": np.full(n, 0.5)}


class _Jobs(QObject):
    jobStarted = Signal(str)
    jobFrame = Signal(str, object)
    jobFinished = Signal(str, object)

    def __init__(self, result=None, session=None) -> None:
        super().__init__()
        self._result = result
        self._session = session
        self.running = None

    def running_id(self):
        return self.running

    def queued_ids(self):
        return []

    def diagnostics_for(self, sim_id):
        from aerolume.app.widgets.diagnostics import DiagnosticsLog
        if self._session is None:
            self._session = DiagnosticsLog()
        return self._session

    def result(self, sim_id):
        return self._result


def test_diagnostics_log_prefers_the_session_then_the_stored_history():
    pytest.importorskip("aerolume.app.widgets.diagnostics")
    from aerolume.app.main_window import diagnostics_log_for
    from aerolume.app.widgets.diagnostics import DiagnosticsLog
    assert diagnostics_log_for(None, "s1") is None
    stored = diagnostics_log_for(_Jobs(result=_Result()), "s1")
    assert stored is not None and len(stored) == 40 and "div_rms" in set(stored.keys())
    live = DiagnosticsLog()
    live.add(1, 0.01, {"div_rms": 1e-4})
    assert diagnostics_log_for(_Jobs(result=_Result(), session=live), "s1") is live
    running = _Jobs(result=_Result(), session=DiagnosticsLog())
    running.running = "s1"                       # a run that just started: its (empty) live log
    assert diagnostics_log_for(running, "s1") is running._session
    nothing = diagnostics_log_for(_Jobs(result=None), "s1")
    assert nothing is not None and len(nothing) == 0
    # a result with no diagnostic history but end-of-run values (what the solvers store today)
    end = _Result()
    end.history = {"t": end.history["t"], "cd": end.history["cd"]}
    end.diagnostics = {"step": 10658.0, "t": 59.99, "div_rms": 1.03e-6, "cfl": 0.232, "note": "text is skipped"}
    last = diagnostics_log_for(_Jobs(result=end), "s1")
    assert last is not None and len(last) == 1 and "div_rms" in set(last.keys())
    assert last.latest()["cfl"] == pytest.approx(0.232) and "End of the run" in last.source


def test_the_window_panel_loads_the_selected_result(isolated, qapp):
    pytest.importorskip("aerolume.app.widgets.diagnostics")
    jobs = _Jobs(result=_Result())
    w = _window(jobs=jobs)
    w.resize(1600, 1000)
    w.show()
    w.new_study("airfoil", "Diag", isolated)
    _pump(qapp, 4)
    w.show_page("results")
    w.docks.show_panel("diagnostics")
    _pump(qapp, 6)
    if w.diagnostics is None or w.docks.panel("diagnostics").content() is not w.diagnostics:
        pytest.skip("the Results page fills the Diagnostics panel itself")
    assert len(w.diagnostics.log) == 40
