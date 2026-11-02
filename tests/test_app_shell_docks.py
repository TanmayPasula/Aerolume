"""UI v3 shell: the dock system and workspaces, the tour, the splash, one instance, crash reports.

DESIGN.md sections 12 (app polish) and 13 (Lumen-style docks). Everything runs on
Qt's offscreen platform with private settings, as tests/test_app_shell.py does.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QLabel, QWidget

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
    settings.use_file(tmp_path / "settings-after.ini")


class FakeJobs(QObject):
    jobQueued = Signal(str)
    jobStarted = Signal(str)
    jobProgress = Signal(str, float, str)
    jobFrame = Signal(str, object)
    jobFinished = Signal(str, object)
    jobFailed = Signal(str, str)
    jobCancelled = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.running: str | None = None
        self.queue: list[str] = []
        self.seconds: dict[str, float] = {}

    def running_id(self):
        return self.running

    def queued_ids(self):
        return list(self.queue)

    def cancel(self, sim_id: str) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def job_seconds(self, sim_id: str):
        return self.seconds.get(sim_id)


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
    w.new_study("airfoil", f"Docks {size[0]}", isolated)
    _pump(qapp, 8)
    return w


# ---------------------------------------------------------------------------
# the panels
# ---------------------------------------------------------------------------

def test_every_side_panel_is_a_managed_dock(isolated, qapp):
    from aerolume.app.widgets.docks import PANELS, DockHeader, DockPanel
    w = _window()
    keys = [s.key for s in PANELS]
    assert keys == ["project", "messages", "assistant", "script", "properties", "flowvis", "diagnostics"]
    for spec in PANELS:
        p = w.docks.panel(spec.key)
        assert isinstance(p, DockPanel) and p.objectName() == spec.object_name
        assert isinstance(p.titleBarWidget(), DockHeader)
        assert not (p.features() & p.DockWidgetFeature.DockWidgetFloatable)     # panels never float
        assert w.docks.actions[spec.key].shortcut().toString() == spec.shortcut
    assert w.project_dock is w.docks.panel("project") and w.messages_dock is w.docks.panel("messages")
    assert w.assistant_dock is w.docks.panel("assistant") and w.script_dock is w.docks.panel("script")
    # the shell's own panels have content; page panels wait for a page
    for key in ("project", "messages", "assistant", "script"):
        assert w.docks.panel(key).available()
    assert not w.docks.panel("properties").available()
    a = w.docks.actions["properties"]
    assert not a.isEnabled() and "no properties panel" in a.toolTip()


def test_collapse_move_reorder_hide_and_restore(isolated, qapp):
    from aerolume.app.widgets.docks import DockHeader
    w = _shown_window(qapp, isolated)
    d = w.docks
    project = d.panel("project")
    assert d.is_shown("project") and not d.is_collapsed("project")
    # collapse from the header (the chevron), then expand by double-clicking the header
    project.header.collapse_button.click()
    _pump(qapp)
    assert d.is_collapsed("project") and project.height() <= DockHeader.HEIGHT
    assert "Expand" in project.header.collapse_button.toolTip()
    project.header.collapseRequested.emit()
    _pump(qapp)
    assert not d.is_collapsed("project") and project.height() > 200
    # the Assistant moves to the left rail, under the tree, then above it
    d.show_panel("assistant")
    _pump(qapp)
    assert d.area_of("assistant") == "right"
    d.move_to("assistant", "left")
    _pump(qapp)
    assert d.area_of("assistant") == "left" and d.order("left")[-1] == "assistant"
    assert w.dockWidgetArea(d.panel("assistant")) == w.dockWidgetArea(project)
    d.move_by("assistant", -10)
    assert d.order("left")[0] == "assistant"
    _pump(qapp)
    assert d.panel("assistant").y() < project.y()                 # Qt's stacking follows the model
    # the header menu offers the same moves, solo mode and Hide
    texts = [a.text() for a in d.panel_menu("assistant").actions()]
    for t in ("Collapse", "Move up", "Move down", "Move to the right rail", "Move to the bottom rail",
              "Solo mode in the left rail", "Hide panel"):
        assert t in texts, (t, texts)
    # hide from the header, restore from View ▸ Panels
    d.panel("assistant").header.close_button.click()
    _pump(qapp)
    assert not d.wanted("assistant") and d.panel("assistant").isHidden()
    assert not d.actions["assistant"].isChecked()
    d.actions["assistant"].trigger()
    _pump(qapp)
    assert d.wanted("assistant") and not d.panel("assistant").isHidden()
    assert d.area_of("assistant") == "left"                       # restored where it was


def test_solo_mode_keeps_one_panel_open_per_rail(isolated, qapp):
    w = _shown_window(qapp, isolated)
    d = w.docks
    d.show_panel("assistant")
    d.show_panel("diagnostics")
    d.move_to("diagnostics", "right")
    _pump(qapp)
    assert {"assistant", "diagnostics"} <= set(d.order("right"))
    d.set_solo("right", True)
    open_right = [k for k in d.order("right") if d.is_shown(k) and not d.is_collapsed(k)]
    assert len(open_right) == 1
    closed = next(k for k in ("assistant", "diagnostics") if d.is_collapsed(k))
    other = "assistant" if closed == "diagnostics" else "diagnostics"
    d.set_collapsed(closed, False, user=True)                    # opening one closes the other
    _pump(qapp)
    assert not d.is_collapsed(closed) and d.is_collapsed(other)
    d.set_solo("right", False)
    d.set_collapsed(other, False, user=True)
    assert not d.is_collapsed(closed) and not d.is_collapsed(other)


def test_header_switch_toggles_an_effect_and_mirrors_the_panel(isolated, qapp):
    from aerolume.app.widgets.common import Switch
    w = _shown_window(qapp, isolated)
    panel = w.docks.panel("diagnostics")
    if w.diagnostics is None:
        pytest.skip("widgets/diagnostics.py is not part of this build")
    sw = panel.header.switch
    assert isinstance(sw, Switch) and sw.isChecked() and w.diagnostics.record_btn.isChecked()
    seen = []
    w.docks.switchToggled.connect(lambda k, on: seen.append((k, on)))
    sw.click()                                               # off: recording pauses, values stay
    assert not w.diagnostics.record_btn.isChecked() and seen[-1] == ("diagnostics", False)
    w.diagnostics.record_btn.setChecked(True)                # and the other way round
    assert sw.isChecked() and panel.switch_checked() is True
    # any panel can carry one (API for the pages)
    other = w.docks.panel("assistant")
    s2 = other.enable_switch(False, "Turn the suggestions on or off")
    assert other.switch_checked() is False and s2.accessibleName() == "Turn the suggestions on or off"
    other.set_switch_checked(True)
    assert seen[-1] == ("assistant", True)


def test_a_page_puts_its_own_widgets_in_the_page_panels(isolated, qapp):
    from aerolume.app.widgets import docks
    w = _shown_window(qapp, isolated)

    class Page(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.props = QLabel("page properties")

        def dock_panels(self):
            return {"properties": self.props, "nonsense": QLabel("ignored")}

    page = Page()
    w._pages["compare"] = page
    w.stack.addWidget(page)
    assert w.docks.adopt_page_panels("compare", page) == ["properties"]
    w.show_page("compare")
    _pump(qapp)
    props = w.docks.panel("properties")
    assert props.available() and props.content() is page.props and w.docks.is_shown("properties")
    assert w.docks.actions["properties"].isEnabled()
    w.show_page("geometry")                                   # the Geometry page offers none
    _pump(qapp)
    assert not props.available() and props.isHidden() and w.docks.wanted("properties")
    assert docks.manager_for(page.props) is w.docks and docks.manager_for(w.project_tree) is w.docks


# ---------------------------------------------------------------------------
# workspaces
# ---------------------------------------------------------------------------

def _model(layout: dict) -> tuple:
    return layout["panels"], layout["order"], sorted(layout["solo"])


def test_workspaces_round_trip_and_persist(isolated, qapp, tmp_path):
    from aerolume.app import settings
    from aerolume.app.widgets.docks import WORKSPACE_NAMES, WORKSPACES, _normalise_layout
    w = _shown_window(qapp, isolated)
    d = w.docks
    assert WORKSPACE_NAMES == ("Default", "Analysis", "Live", "Scripting")
    for name in WORKSPACE_NAMES:
        w.set_workspace(name)
        _pump(qapp, 3)
        preset = _normalise_layout(WORKSPACES[name], WORKSPACES[name])
        assert _model(d.capture()) == _model(preset), name
        assert w.act_workspace.iconText() == name
        # the window's own fold state of the log follows the workspace
        assert w.messages_expanded() == (not preset["panels"]["messages"][2])
    # apply(capture()) is an exact round trip
    snap = d.capture()
    d.apply(snap)
    assert _model(d.capture()) == _model(snap)
    # a change is kept in its workspace, across a switch and across a restart
    w.set_workspace("Analysis")
    d.hide_panel("assistant")
    d.move_to("messages", "left")
    w.set_workspace("Default")
    assert d.is_shown("project") and not d.wanted("assistant")
    w.set_workspace("Analysis")
    assert not d.wanted("assistant") and d.area_of("messages") == "left"
    w.save_window_state()
    assert settings.get().value("workspaces/current") == "Analysis"
    w2 = _window()
    assert w2.docks.workspace == "Analysis" and not w2.docks.wanted("assistant")
    assert w2.docks.area_of("messages") == "left"
    # Reset puts the preset back
    w2.docks.reset_workspace()
    assert w2.docks.wanted("assistant") and w2.docks.area_of("messages") == "bottom"
    # a damaged stored layout falls back to the preset instead of breaking the window
    settings.get().set_value("workspaces/Live", "{not json")
    settings.get().set_value("workspaces/current", "Live")
    w3 = _window()
    assert w3.docks.workspace == "Live" and _model(w3.docks.capture()) == _model(
        _normalise_layout(WORKSPACES["Live"], WORKSPACES["Live"]))


def test_v2_dock_preferences_are_carried_over(isolated, qapp):
    from aerolume.app import settings
    settings.get().set_value("window/docks", "projectDock=1,messagesDock=0,stepToolbar=0,assistantDock=1,"
                                             "scriptDock=0")
    w = _window()
    assert w.panel_wanted("projectDock") and not w.panel_wanted("messagesDock")
    assert w.panel_wanted("assistantDock") and not w.panel_wanted("stepToolbar")
    assert settings.get().value("workspaces/current") == "Default"     # migrated once


def test_narrow_window_folds_the_right_rail_into_the_left(isolated, qapp):
    from aerolume.app.widgets.docks import COMPACT_WIDTH
    w = _shown_window(qapp, isolated)
    d = w.docks
    d.show_panel("assistant")
    _pump(qapp)
    assert not d.is_compact() and d.area_of("assistant") == "right"
    w.resize(COMPACT_WIDTH - 140, 800)
    _pump(qapp, 6)
    assert d.is_compact() and "assistant" in d.folded()
    assert w.dockWidgetArea(d.panel("assistant")) == w.dockWidgetArea(w.project_dock)
    assert d.panel("assistant") in w.tabifiedDockWidgets(w.project_dock)
    assert d.area_of("assistant") == "right"                    # the model remembers where it belongs
    w.resize(1400, 900)
    _pump(qapp, 6)
    assert not d.is_compact() and not d.folded()
    assert w.dockWidgetArea(d.panel("assistant")) != w.dockWidgetArea(w.project_dock)
    assert d.is_shown("assistant")


def test_rail_widths_follow_the_workspace_and_reset(isolated, qapp):
    w = _shown_window(qapp, isolated)
    d = w.docks
    _pump(qapp, 4)
    assert abs(d.rail_size("left") - 230) <= 12                 # Default: a 230 px left rail
    d.set_rail_size("left", 330)
    _pump(qapp, 3)
    assert abs(d.rail_size("left") - 330) <= 12
    d.reset_rail_sizes()                                         # the rail-edge double-click
    _pump(qapp, 3)
    assert abs(d.rail_size("left") - 230) <= 12


# ---------------------------------------------------------------------------
# Messages keeps its own rules inside the dock system
# ---------------------------------------------------------------------------

def test_messages_fold_is_the_panel_collapse(isolated, qapp):
    from aerolume.app.main_window import DockTitle
    w = _shown_window(qapp, isolated, size=(1280, 800))
    assert not w.messages_expanded() and w.docks.is_collapsed("messages")
    assert w.messages_dock.height() <= DockTitle.HEIGHT
    w.messages_toggle.click()                                    # the header chevron opens the log
    _pump(qapp)
    assert w.messages_expanded() and not w.docks.is_collapsed("messages")
    assert w.messages.view.isVisibleTo(w)
    w.messages_dock.header.collapseRequested.emit()              # a double-click on the header folds it
    _pump(qapp)
    assert not w.messages_expanded() and w.messages.latest.isVisibleTo(w)


# ---------------------------------------------------------------------------
# tour
# ---------------------------------------------------------------------------

def test_tour_on_the_start_page_explains_only_the_start_page(isolated, qapp):
    """Started before a project is open, every step outlines a Start page widget (no centred boxes)."""
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtGui import QKeyEvent

    from aerolume.app import settings
    from aerolume.app.widgets.tour import PROJECT_TOUR_STEPS, START_TOUR_STEPS, TOUR_STEPS
    assert [s.key for s in START_TOUR_STEPS] == ["start", "examples", "learn"]
    assert [s.key for s in PROJECT_TOUR_STEPS] == ["steps", "project", "run", "panels", "help"]
    assert TOUR_STEPS == START_TOUR_STEPS + PROJECT_TOUR_STEPS
    w = _window()
    w.resize(1600, 1000)
    w.show()
    _pump(qapp, 4)
    tour = w.start_tour()
    _pump(qapp, 3)
    assert tour.part == "start" and tour.steps == START_TOUR_STEPS
    assert tour.isVisible() and tour.current_index() == 0
    page = w.page("start")
    expect = {"start": [page.tiles_widget],
              "examples": [page.examples_header, page.examples],
              "learn": [page.learn_header, *page.help_links.values()]}
    for i, step in enumerate(START_TOUR_STEPS):
        _pump(qapp, 2)
        assert tour.current_index() == i and tour.current_step().key == step.key
        r = tour.target_rect()
        assert not r.isNull() and not tour.box.absent.isVisibleTo(tour), step.key   # a spotlight, every step
        for wid in expect[step.key]:
            if wid.isVisible():
                assert r.contains(QRect(wid.mapTo(w, QPoint(0, 0)), wid.size())), (step.key, wid)
        assert not tour.box_rect().intersects(r) and tour.rect().contains(tour.box_rect()), step.key
        if i < len(START_TOUR_STEPS) - 1:
            tour.next()
    assert tour.box.next.text() == "Finish"
    tour.back()
    tour.back()
    assert tour.current_index() == 0 and not tour.box.back.isEnabled()
    finished = []
    tour.finished.connect(finished.append)
    tour.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert finished == [True]
    assert settings.get().flag(w.TOUR_KEY) is True and w.maybe_show_first_run_tour() is False
    assert not w.project_tour_owed()                             # skipped: not shown again on a project


def test_finished_start_tour_continues_on_the_first_project(isolated, qapp):
    """The project part is owed after the Start part and runs once, when a project page shows."""
    from aerolume.app import settings
    w = _window()
    w.resize(1600, 1000)
    w.show()
    _pump(qapp, 4)
    tour = w.start_tour(first_run=True)
    _pump(qapp, 2)
    for _ in range(3):
        tour.next()
    _pump(qapp, 2)
    assert settings.get().flag(w.TOUR_KEY) is True and w.project_tour_owed()
    w.new_study("airfoil", "Tour continues", isolated)

    end = time.monotonic() + 3.0
    while time.monotonic() < end and not (getattr(w, "tour", None) is not None
                                           and w.tour.isVisible() and w.tour.part == "project"):
        _pump(qapp, 2)
        time.sleep(0.05)
    tour = w.tour
    assert tour.part == "project" and tour.isVisible() and tour.current_step().key == "steps"
    assert not tour.target_rect().isNull()
    tour.skip()
    _pump(qapp, 2)
    assert not w.project_tour_owed()
    w.show_page("setup")
    _pump(qapp, 2)
    assert getattr(w, "_tour_continue_pending", False) is False   # settled: never runs again by itself


def test_tour_points_at_each_panel_once_a_project_is_open(isolated, qapp):
    w = _shown_window(qapp, isolated)
    tour = w.start_tour()
    assert tour.part == "project"
    rects = []
    for _ in range(len(tour.steps)):
        _pump(qapp, 2)
        rects.append((tour.current_step().key, tour.target_rect()))
        tour.next()
    assert [k for k, _ in rects] == ["steps", "project", "run", "panels", "help"]
    for key, r in rects:
        assert not r.isNull(), key                               # every step highlights its widget
    box = tour.box.geometry()
    assert tour.rect().contains(box)


# ---------------------------------------------------------------------------
# splash, one instance, crash reports, accounts hook
# ---------------------------------------------------------------------------

def test_splash_names_each_step_and_survives_a_failing_one(qapp):
    from aerolume.app.widgets.splash import Splash, preload
    splash = Splash()
    seen = []
    clock = iter([0.0, 0.5, 1.0, 1.25]).__next__

    def fail():
        raise ImportError("no VTK here")

    steps = [("Loading one…", lambda: seen.append(splash.message.text())), ("Loading two…", fail)]
    done = preload(splash, steps, clock=clock)
    assert seen == ["Loading one…"]
    assert [s.error for s in done] == ["", "ImportError: no VTK here"]
    assert [round(s.seconds, 2) for s in done] == [0.5, 0.25]
    assert splash.message.text() == "Opening the window…" and splash.bar.value() == 1000
    assert splash.title.text() == "Aerolume" and "Apache" in splash.footer.text()
    splash.close()


class _FakeInstance(QObject):
    message_received = Signal(list)

    def __init__(self, primary: bool) -> None:
        super().__init__()
        self.primary = primary
        self.argv = None

    def start(self, argv):
        self.argv = list(argv)
        return self.primary


def test_a_second_launch_hands_its_project_to_the_first(isolated, qapp, monkeypatch):
    from aerolume.app import main as main_mod
    from aerolume.app import main_window as mw
    from aerolume.app.pages import start
    built = []

    class NoWindow:
        def __init__(self, *a, **k) -> None:
            built.append(1)

    monkeypatch.setattr(mw, "MainWindow", NoWindow)
    fake = _FakeInstance(primary=False)
    assert main_mod.main(["aerolume", "C:/somewhere/Wing.aerolume", "--no-splash"], instance=fake) == 0
    assert fake.argv == ["C:/somewhere/Wing.aerolume", "--no-splash"] and built == []
    monkeypatch.undo()
    # the first instance opens what it is handed and comes to the front
    w = _window()
    w.resize(1280, 800)
    w.show()
    proj, _ = start.create_study("blank", "Forwarded", isolated)
    marker = start.project_path(proj) / "project.aerolume"
    marker.write_text("", encoding="utf-8")
    opened = w.handle_forwarded_argv([str(marker)])            # a double-click passes the marker file
    assert opened is not None and start.project_name(w.state.project) == "Forwarded"


def test_crash_dialog_shows_a_copyable_report(isolated, qapp, tmp_path):
    from dataclasses import dataclass

    from PySide6.QtWidgets import QApplication

    @dataclass
    class Report:
        text: str
        path: Path | None
        summary: str

    w = _window()
    r1 = Report("ValueError: bad\n\nTraceback:\n  ...", tmp_path / "crash-1.log", "ValueError: bad")
    dlg = w.show_crash(r1)
    assert dlg.isVisible() and dlg.summary.text() == "ValueError: bad"
    assert "crash-1.log" in dlg.notice.text.text() and "keeps running" in dlg.notice.text.text()
    assert dlg.copy_report() == r1.text and QApplication.clipboard().text() == r1.text
    r2 = Report("KeyError: 'x'", None, "KeyError: 'x'")
    assert w.show_crash(r2) is dlg                               # one dialog collects a burst
    assert dlg.summary.text().startswith("2 unexpected errors") and "KeyError" in dlg.details.toPlainText()
    assert w.messages.counts()["error"] >= 1                     # and the log says it happened
    dlg.close()


def test_platform_crash_handler_reaches_the_window(isolated, qapp, monkeypatch):
    from aerolume.app import main as main_mod
    plat = pytest.importorskip("aerolume.app.platform")
    got = []
    monkeypatch.setattr(plat, "install_crash_handler", lambda cb: got.append(cb))
    w = _window()
    assert main_mod._install_crash_reports(w) is True and callable(got[0])
    shown = []
    monkeypatch.setattr(w, "show_crash", shown.append)
    main_mod._hook_window[:] = [w]
    got[0]("report")
    assert shown == ["report"]


def test_account_menu_hook_is_optional_and_guarded(isolated, qapp, monkeypatch):
    from aerolume.app import main_window as mw
    monkeypatch.setattr(mw.MainWindow, "ACCOUNT_MENU_HOOK", "aerolume.app.widgets.no_such_menu:install")
    w = _window()
    _pump(qapp)
    assert w.messages.counts()["error"] == 0                     # absent: no menu, no complaint
    calls = []
    monkeypatch.setattr(mw.MainWindow, "ACCOUNT_MENU_HOOK", "aerolume.app.widgets.docks:placeholder")
    monkeypatch.setattr("aerolume.app.widgets.docks.placeholder", lambda win, state: calls.append((win, state)))
    w2 = _window()
    assert calls == [(w2, w2.state)]
    monkeypatch.setattr("aerolume.app.widgets.docks.placeholder",
                        lambda win, state: (_ for _ in ()).throw(RuntimeError("no network")))
    w3 = _window()
    _pump(qapp, 3)
    assert w3.messages.counts()["error"] == 1                    # broken: said once, never raised


# ---------------------------------------------------------------------------
# run control and the status bar
# ---------------------------------------------------------------------------

def test_run_is_not_offered_for_a_simulation_that_already_runs(isolated, qapp):
    jobs = FakeJobs()
    w = _shown_window(qapp, isolated, jobs=jobs)
    sim = w.state.simulation
    assert w.run_target() == ("run", sim.name) and w.act_run.isEnabled()
    jobs.running = sim.id
    jobs.jobStarted.emit(sim.id)
    _pump(qapp)
    kind, text = w.run_target()
    assert kind == "none" and "already running" in text and not w.act_run.isEnabled()
    assert "already running" in w.act_run.toolTip()
    jobs.running, jobs.queue = None, [sim.id]
    jobs.jobQueued.emit(sim.id)
    assert w.run_target()[0] == "none" and "is queued" in w.run_target()[1]


def test_status_bar_quotes_the_job_managers_duration(isolated, qapp):
    jobs = FakeJobs()
    w = _shown_window(qapp, isolated, jobs=jobs)
    sim = w.state.simulation
    jobs.jobStarted.emit(sim.id)
    jobs.seconds[sim.id] = 204.0
    jobs.jobFinished.emit(sim.id, None)
    _pump(qapp)
    assert w.run_status.summary().endswith("done in 3 min 24 s")


# ---------------------------------------------------------------------------
# Start page: the study tiles and the heat-transfer studies
# ---------------------------------------------------------------------------

def test_start_page_tiles_start_each_study(isolated, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from aerolume.app.pages import start
    page = start.StartPage(None)
    page.resize(1280, 760)
    page.show()
    _pump(qapp)
    asked = []
    page.newStudyRequested.connect(asked.append)
    keys = list(page.tiles)
    expect = ["airfoil", "vehicle", "mesh", "wing"] + (["room", "enclosure"] if start.physics_templates_available()
                                                        else [])
    assert keys == expect
    for k in keys:
        page.tiles[k].click()
    assert asked == keys
    tile = page.tiles["wing"]
    tile.setFocus()
    tile.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert asked[-1] == "wing"
    assert tile.accessibleName() == "New wing (3d) study"
    assert "vortex-lattice" in tile.accessibleDescription()
    for t in page.tiles.values():
        assert t.width() >= t.MIN_W and t.isVisible()
    page.link_more.click()
    assert asked[-1] == "blank"                                  # More study types -> the New study dialog
    assert set(page.help_links) == {"tour", "scripting", "assistant", "mcp", "shortcuts"}
    page.close()


def test_heated_room_and_enclosure_studies(isolated):
    from aerolume.app.pages import start
    if not start.physics_templates_available():
        pytest.skip("app/dialogs/physics_templates.py is not part of this build")
    proj, msgs = start.create_study("room", "Room", isolated)
    assert msgs == []
    (design,) = proj.designs.values()
    (sim,) = proj.simulations.values()
    assert design.dimension == 3 and sim.design_id == design.id
    assert sim.thermal.enabled and sim.thermal.buoyancy and sim.cfd.domain_mode == "closed"
    assert sim.heat_sources and sim.fans and sim.ports
    assert start.summarize(start.project_path(proj)).type_text == "Room (3D)"
    proj2, msgs2 = start.create_study("enclosure", "Box", isolated)
    assert msgs2 == []
    (d2,) = proj2.designs.values()
    assert d2.dimension == 2 and start.summarize(start.project_path(proj2)).type_text == "Enclosure (2D)"


def test_wing_and_3d_projects_are_3d_in_the_recent_list(isolated):
    from aerolume.app.pages.start import ProjectSummary
    s = ProjectSummary(Path("."), "Mixed", 0.0, 2, 2, "", [], "wing", frozenset({"3D", "Wing"}),
                       frozenset({"wing", "vehicle"}))
    assert s.type_text == "3D"


def test_project_tree_elides_names_at_the_end(isolated, qapp):
    from PySide6.QtCore import Qt

    from aerolume.app.widgets.project_tree import ProjectTree
    w = _window()
    assert isinstance(w.project_tree, ProjectTree)
    assert w.project_tree.view.textElideMode() == Qt.TextElideMode.ElideRight


# ---------------------------------------------------------------------------
# accessibility, the jump list, the update check
# ---------------------------------------------------------------------------

def _acc_name(w) -> str:
    from PySide6.QtGui import QAccessible
    return QAccessible.queryAccessibleInterface(w).text(QAccessible.Text.Name)


def test_property_rows_name_their_fields_for_screen_readers(qapp):
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QPushButton

    from aerolume.app.widgets.common import PropertyGroup, SegmentedControl
    g = PropertyGroup("Flow")
    speed = QDoubleSpinBox()
    row = g.add_row("Speed", speed, unit="m/s")
    assert _acc_name(speed) == "Speed" and speed.accessibleDescription() == "in m/s"
    assert row.label.buddy() is speed
    # a composite editor: the spin box is the buddy, the unit selector says what it is
    box = QWidget()
    lay = QHBoxLayout(box)
    value, unit = QDoubleSpinBox(), QComboBox()
    unit.setToolTip("Display unit (the value is kept)")
    lay.addWidget(value)
    lay.addWidget(unit)
    g.add_row("Reference length", box)
    assert _acc_name(value) == "Reference length"
    assert _acc_name(unit) == "Reference length: Display unit (the value is kept)"
    # rich text is read as text; a renamed row renames its field; a page's own name is kept
    cd = QDoubleSpinBox()
    r2 = g.add_row("C<sub>d</sub>", cd)
    assert _acc_name(cd) == "Cd"
    r2.set_label("Drag coefficient")
    assert _acc_name(cd) == "Drag coefficient"
    named = QDoubleSpinBox()
    named.setAccessibleName("Angle of attack")
    g.add_row("α", named)
    assert _acc_name(named) == "Angle of attack"
    # a plain "&" stays visible text (no buddy, so no mnemonic) and the field is still named
    both = QDoubleSpinBox()
    r3 = g.add_row("Lift & drag", both)
    assert r3.label.buddy() is None and _acc_name(both) == "Lift & drag"
    # a segmented control's buttons keep their own text; the group carries the label
    seg = SegmentedControl([("light", "Light"), ("dark", "Dark")], "light")
    g.add_row("Theme", seg)
    assert seg.accessibleName() == "Theme"
    assert sorted(_acc_name(b) for b in seg.findChildren(QPushButton)) == ["Dark", "Light"]


def test_preferences_fields_are_named(isolated, qapp):
    from PySide6.QtWidgets import QComboBox, QLineEdit
    w = _window()
    dlg = w.open_preferences(exec_=False)
    names = {_acc_name(c) for c in dlg.findChildren(QComboBox) if c.isVisibleTo(dlg) or True}
    assert {"Compute device", "Precision"} <= names
    assert _acc_name(dlg.update_url) == "Release feed" and isinstance(dlg.update_url, QLineEdit)
    # the hosted Assistant page builds its rows with addRow(QLabel, field); the window names them
    if dlg.assistant is not None:
        from PySide6.QtWidgets import QAbstractSpinBox
        fields = [c for cls in (QComboBox, QLineEdit, QAbstractSpinBox) for c in dlg.assistant.findChildren(cls)
                  if not (isinstance(c, QLineEdit) and isinstance(c.parent(), (QComboBox, QAbstractSpinBox)))]
        assert fields
        unnamed = [type(c).__name__ for c in fields if not _acc_name(c).strip()]
        assert not unnamed, unnamed
    dlg.close()


def test_tie_form_labels_names_label_widget_rows(qapp):
    from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QWidget

    from aerolume.app.widgets.common import label, tie_form_labels
    page = QWidget()
    form = QFormLayout(page)
    edit, spin, own = QLineEdit(), QSpinBox(), QLineEdit()
    own.setAccessibleName("Kept")
    row = QHBoxLayout()
    pick, refresh = QLineEdit(), QPushButton("Refresh")
    pick2 = QLineEdit()
    pick2.setToolTip("Second part\nmore")
    row.addWidget(pick)
    row.addWidget(pick2)
    row.addWidget(refresh)
    form.addRow(label("Ollama address", "pg-label"), edit)
    form.addRow(label("Context", "pg-label"), spin)
    form.addRow(label("Own", "pg-label"), own)
    form.addRow(label("Model", "pg-label"), row)
    form.addRow("String label", QLineEdit())          # Qt reads string labels itself: left alone
    assert tie_form_labels(page) == 4
    assert _acc_name(edit) == "Ollama address" and _acc_name(spin) == "Context"
    assert _acc_name(own) == "Kept"
    assert _acc_name(pick) == "Model" and _acc_name(pick2) == "Model: Second part"
    assert _acc_name(refresh) == "Refresh"            # a button keeps its own text


def test_opening_a_project_notes_it_for_the_desktop(isolated, monkeypatch):
    plat = pytest.importorskip("aerolume.app.platform")
    from aerolume.app.pages import start
    noted = []
    monkeypatch.setattr(plat, "note_project_opened", lambda p: noted.append(Path(p)))
    proj, _ = start.create_study("blank", "Jump list", isolated)
    folder = start.project_path(proj)
    start.open_project(folder)
    assert noted == [folder, folder]
    # the program's own files (bundled examples opened in place) are never written to
    noted.clear()
    examples = Path(plat.examples_dir())
    if examples.is_dir() and any(examples.iterdir()):
        start.open_project(next(p for p in examples.iterdir() if p.is_dir()))
        assert noted == []
    # a failing desktop helper never stops a project opening
    monkeypatch.setattr(plat, "note_project_opened", lambda p: (_ for _ in ()).throw(OSError("read-only")))
    assert start.project_name(start.open_project(folder)) == "Jump list"


def test_update_check_is_off_until_a_feed_is_set(isolated, qapp):
    from aerolume.app import settings
    w = _window()
    assert settings.get().update_url == ""
    assert w.check_for_updates() is None
    assert w._update_box.windowTitle() == "Update checks are off" and "Release feed" in w._update_box.text()
    w._update_box.close()
    assert w.act_updates in w.help_menu.actions()


def test_update_check_reads_the_feed_in_the_background(isolated, qapp, tmp_path):
    import json

    from aerolume import __version__
    from aerolume.app import main_window as mw
    from aerolume.app import settings
    feed = tmp_path / "latest.json"
    feed.write_text(json.dumps({"tag_name": "v99.0.0", "html_url": "https://example.org/r/99"}), encoding="utf-8")
    settings.get().update_url = feed.as_uri()
    w = _window()

    def finish(thread):
        thread.join(5)
        deadline = time.monotonic() + 5
        while getattr(w, "last_update_check", None) is None and time.monotonic() < deadline:
            _pump(qapp, 2)
        return w.last_update_check

    assert finish(w.check_for_updates()) == ("v99.0.0", "https://example.org/r/99", "")
    assert w._update_box.windowTitle() == "Update available" and "99.0.0" in w._update_box.text()
    assert any(b.text() == "Open release page" for b in w._update_box.buttons())
    w._update_box.close()
    # the same version: up to date; a broken feed: said so, with the reason
    w.last_update_check = None
    assert finish(w.check_for_updates(fetch=lambda url: (__version__, ""))) == (__version__, "", "")
    assert w._update_box.windowTitle() == "Aerolume is up to date"
    w._update_box.close()
    w.last_update_check = None
    feed.write_text("[1, 2]", encoding="utf-8")
    version, _page, error = finish(w.check_for_updates())
    assert version == "" and "release feed" in error
    assert w._update_box.windowTitle() == "Could not check for updates"
    w._update_box.close()
    # feeds: the two shapes, versions compared as numbers, other schemes refused
    assert mw.parse_release_feed({"version": "0.3.1", "url": "https://x/y"}) == ("0.3.1", "https://x/y")
    assert mw.version_key("v0.10.0") > mw.version_key("0.9.9") and mw.version_key("0.3") == mw.version_key("0.3.0")
    with pytest.raises(ValueError):
        mw.fetch_release_feed("ftp://example.org/feed.json")


def test_header_switch_belongs_to_the_content_on_screen(qapp):
    from PySide6.QtWidgets import QCheckBox, QMainWindow

    from aerolume.app.widgets.docks import DockManager
    win = QMainWindow()
    m = DockManager(win, None)
    panel = m.panel("diagnostics")
    own, page = QLabel("window"), QLabel("page")
    m.set_content("diagnostics", own)
    own_btn, page_btn = QCheckBox("record"), QCheckBox("record")
    own_btn.setChecked(True)
    s_own = panel.enable_switch(True, "Record (window)", on_toggled=own_btn.setChecked)
    m.set_page_content("simulate", "diagnostics", page)
    m.show_page("simulate")
    s_page = panel.enable_switch(False, "Record (page)", on_toggled=page_btn.setChecked)
    page_btn.toggled.connect(panel.set_switch_checked)
    own_btn.toggled.connect(panel.set_switch_checked)
    assert panel.header.switch is s_page and s_page.isVisibleTo(panel.header) and not s_own.isVisibleTo(panel.header)
    m.show_page("results")                                   # back to the window's own content
    assert panel.header.switch is s_own and panel.switch_checked() is True
    s_own.click()
    assert not own_btn.isChecked() and not s_page.isChecked()
    # the page's button, toggled while another page is on screen, still sets the page's own switch
    page_btn.setChecked(True)
    assert s_page.isChecked() and not s_own.isChecked()
    m.show_page("simulate")
    assert panel.switch_checked() is True
    panel.disable_switch()
    assert panel.header.switch is None                       # the window's switch is not the page's
    m.show_page("results")
    assert panel.header.switch is s_own
    win.deleteLater()
