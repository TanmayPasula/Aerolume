"""The in-app Script console and its engine (DESIGN.md 8), offscreen.

Scripts run on a worker thread against the open project through
``aerolume.app.scripting.AppHost``: project edits happen on the GUI thread
through AppState, simulations go through the real JobManager (with a fake
runner, so no solver runs here), Stop cancels runs and interrupts plain
Python, output is routed to the console only.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from aerolume.model import PolarPoint, PolarResult

pytestmark = pytest.mark.gui

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "scripts"


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    """AppState with a fresh project, a real JobManager whose runner is a fast fake, a console."""
    from aerolume.app import settings, theme
    from aerolume.app.jobs import JobManager
    from aerolume.app.state import AppState
    from aerolume.app.widgets.script_console import ScriptConsole
    from aerolume.project import Project
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    state = AppState()
    jobs = JobManager(state)
    runs: list[str] = []

    def fake_runner(design, sim, *, progress=None, frame=None, cancel=None, view=None, **_kw):
        """A polar in a blink -- or, for a simulation named 'slow…', one that runs until cancelled."""
        runs.append(sim.name)
        if sim.name.startswith("slow"):
            while cancel is not None and not cancel.is_set():
                if progress:
                    progress(0.3, "Developing the flow")
                time.sleep(0.02)
        if progress:
            progress(1.0, "Polar finished")
        pts = [PolarPoint(alpha=a, cl=0.1 * a, cd=0.01, cm=-0.05, converged=True) for a in sim.airfoil.alphas()]
        return PolarResult(sim_id=sim.id, re=1e6, points=pts)
    jobs.runner = fake_runner
    state.set_project(Project.create(tmp_path / "Console", "Console"))
    console = ScriptConsole(state, jobs)
    console.resize(1400, 400)
    console.show()
    log: list[tuple] = []
    state.messageLogged.connect(lambda *a: log.append(a))
    yield {"state": state, "jobs": jobs, "console": console, "runs": runs, "log": log}
    console.shutdown()
    jobs.shutdown()
    console.close()
    console.deleteLater()


def _run(console, text: str, timeout: float = 30.0):
    outcomes = []
    console.runFinished.connect(outcomes.append)
    console.set_text(text)
    assert console.run()
    assert console.engine.wait(timeout), "the script did not finish"
    console.runFinished.disconnect(outcomes.append)
    return outcomes[-1]


def test_script_records_through_appstate_and_runs_through_the_job_queue(env):
    state, console = env["state"], env["console"]
    changed = []
    state.simulationsChanged.connect(lambda: changed.append(1))
    outcome = _run(console, (
        'print("hello")\n'
        'd = project.add_naca("2412")\n'
        'r = project.run_polar(d, alpha=(0, 4, 2))\n'
        'print(r.cl.tolist(), r.status)\n'
        'log("polar done")\n'))
    assert outcome.ok and outcome.status == "finished" and outcome.sim_ids
    text = console.output.plain_text()
    assert "hello" in text and "[0.0, 0.2, 0.4] done" in text and "polar done" in text
    assert "Finished in" in console.status.text() and "1 simulation recorded" in text
    (sim,) = state.project.simulations.values()
    assert sim.status == "done" and state.project.has_result(sim.id) and env["runs"] == [sim.name]
    assert changed, "the Project tree was not told about the new simulation"
    assert ("info", "Script", "polar done") in env["log"]
    # re-running reuses the stored result: the job queue is not used again
    again = _run(console, 'r = project.run_polar(project.design("NACA 2412"), alpha=(0, 4, 2))\nprint(r.reused)\n')
    assert again.ok and console.output.plain_text().rstrip().splitlines()[-2] == "True"
    assert env["runs"] == [sim.name]
    # ... and the console says so: "Finished in 0.1 s" alone read as a fresh run (critic round 3.2)
    assert again.reused_ids == [sim.id] and not again.sim_ids
    last = console.output.plain_text().rstrip().splitlines()[-1]
    assert "1 stored result of an identical run reused, not run again (rerun=True runs it again)" in last
    assert console.status.text().endswith("· stored result reused")
    printed = _run(console, 'print(project.run_polar(project.design("NACA 2412"), alpha=(0, 4, 2)))\n')
    assert printed.ok and "Stored result of an identical run, reused (not run again" in console.output.plain_text()


def test_stop_interrupts_plain_python(env):
    console = env["console"]
    console.set_text("i = 0\nwhile True:\n    i += 1\n")
    console.run()
    from PySide6.QtCore import QCoreApplication
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.3:
        QCoreApplication.processEvents()
    assert console.is_running()
    outcomes = []
    console.runFinished.connect(outcomes.append)
    console.stop()
    assert console.engine.wait(10)
    assert outcomes[-1].status == "stopped" and time.monotonic() - t0 < 6
    assert console.status.text().startswith("Stopped after") and console.run_button.isEnabled()


def test_stop_cancels_a_running_simulation_which_keeps_its_partial_result(env):
    console, state = env["console"], env["state"]
    outcomes = []
    console.runFinished.connect(outcomes.append)
    console.set_text('d = project.add_naca("0012")\n'
                     'project.run_polar(d, alpha=(0, 2, 1), name="slow polar")\n'
                     'print("not reached")\n')
    console.run()
    sim = None
    from PySide6.QtCore import QCoreApplication
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10:
        QCoreApplication.processEvents()
        sims = list(state.project.simulations.values())
        if sims and sims[0].status == "running":
            sim = sims[0]
            break
        time.sleep(0.01)
    assert sim is not None, "the simulation never started"
    console.stop()
    assert console.engine.wait(10)
    assert outcomes[-1].status == "stopped" and "not reached" not in console.output.plain_text()
    assert state.project.simulations[sim.id].status == "cancelled"
    assert state.project.has_result(sim.id) and env["jobs"].running_id() is None


def test_errors_mark_the_line_and_link_it(env):
    console = env["console"]
    outcome = _run(console, "x = 1\n\nproject.run_cfd('nope')\n")
    assert outcome.status == "failed" and outcome.line == 3 and "NotFoundError" in outcome.error
    assert console.editor._error_line == 3 and console.status.text() == "Failed on line 3"
    assert "inside Aerolume or other libraries not shown" in outcome.traceback
    html = console.output.toHtml()
    assert 'href="line:3"' in html
    console.output.lineRequested.emit(3)
    assert console.editor.textCursor().blockNumber() == 2
    assert any(level == "error" and "line 3" in text for level, _src, text in env["log"])
    syntax = _run_failing_syntax(console)
    assert syntax.status == "failed" and syntax.line == 2


def _run_failing_syntax(console):
    outcomes = []
    console.runFinished.connect(outcomes.append)
    console.set_text("x = 1\ndef broken(:\n")
    assert console.run() is False                     # compiled on the GUI thread, never started
    return outcomes[-1]


def test_output_is_routed_to_the_console_only(env, capsys):
    console = env["console"]
    outcome = _run(console, 'import sys\nprint("to console")\nprint("err line", file=sys.stderr)\n')
    out, err = capsys.readouterr()
    assert "to console" not in out and "err line" not in err
    assert "to console" in outcome.output and "err line" in console.output.plain_text()
    import sys
    assert type(sys.stdout).__name__ != "_Router"           # the router is gone after the run


def test_variables_persist_until_reset_and_selection_runs_keep_line_numbers(env):
    console = env["console"]
    assert _run(console, "value = 41\n").ok
    assert _run(console, "print(value + 1)\n").ok and "42" in console.output.plain_text()
    console.reset_variables()
    failed = _run(console, "print(value)\n")
    assert failed.status == "failed" and "NameError" in failed.error
    console.set_text("a = 1\nb = 2\nc = undefined_name")
    cur = console.editor.textCursor()
    cur.movePosition(cur.MoveOperation.End)
    console.editor.setTextCursor(cur)
    outcomes = []
    console.runFinished.connect(outcomes.append)
    assert console.run_selection()
    assert console.engine.wait(10)
    assert outcomes[-1].line == 3 and console.editor._error_line == 3


def test_show_puts_figures_inline(env):
    console = env["console"]
    outcome = _run(console, (
        "from matplotlib.figure import Figure\n"
        "fig = Figure(figsize=(3, 2)); fig.add_subplot().plot([0, 1], [0, 1])\n"
        "show(fig)\nshow('plain text')\n"))
    assert outcome.ok and len(outcome.figures) == 1 and outcome.figures[0][:4] == b"\x89PNG"
    assert "plain text" in console.output.plain_text() and "<img" in console.output.toHtml()
    # pyplot draws off-screen (a Qt window from the worker thread would crash the app) and its
    # figures appear when the script ends, as in a notebook
    outcome = _run(console, "import matplotlib.pyplot as plt\nplt.plot([0, 1], [1, 0])\nplt.show()\n")
    import matplotlib
    assert outcome.ok and len(outcome.figures) == 1 and matplotlib.get_backend().lower() == "agg"
    assert "non-interactive" not in console.output.plain_text()


def test_inline_figures_stay_readable_and_open_full_size(env, tmp_path):
    from PySide6.QtCore import QCoreApplication

    from aerolume.app.widgets import script_console as sc
    console = env["console"]
    console.resize(1400, 220)                        # a short bottom dock
    QCoreApplication.processEvents()
    outcome = _run(console, (
        "from matplotlib.figure import Figure\n"
        "for n in range(2):\n"
        "    fig = Figure(figsize=(6.4, 4.8)); fig.add_subplot().plot([0, 1], [n, 1 - n])\n"
        "    show(fig)\n"))
    out = console.output
    assert outcome.ok and out.figure_count() == 2
    # a 6.4 x 4.8 in figure is not squeezed to the pane's ~180 px height: it keeps a readable width
    assert out.figure_width(1024, 768) == pytest.approx(min(sc.FIGURE_MIN_W, out._available_width()))
    console.resize(1400, 900)
    QCoreApplication.processEvents()
    assert out.figure_width(1024, 768) == pytest.approx(min(sc.FIGURE_MAX_W, out._available_width()))
    html = out.toHtml()
    assert 'href="figure:1"' in html and 'href="figure:2"' in html
    # a click opens the figure full size; Save writes the script's own PNG
    out.anchorClicked.emit(sc.QUrl("figure:2"))
    (viewer,) = out.viewers
    assert viewer.windowTitle() == "Figure 2" and viewer.image.width() == 1024
    assert viewer.canvas.pixmap() is not None and not viewer.canvas.pixmap().isNull()
    saved = viewer.save(tmp_path / "fig")
    assert saved == tmp_path / "fig.png" and saved.read_bytes() == outcome.figures[1]
    viewer._ask_save_path = lambda: None
    assert viewer.save() is None                     # cancelled dialog: nothing written
    assert out.save_dir == env["state"].project.path / "exports"
    # re-rendering (theme switch) keeps the numbering, clearing restarts it
    from aerolume.app import theme
    theme.apply(QCoreApplication.instance(), "dark")
    assert out.figure_count() == 2 and 'href="figure:2"' in out.toHtml()
    theme.apply(QCoreApplication.instance(), "light")
    console.clear_output()
    assert out.figure_count() == 0 and out.open_figure(1) is None
    console.shutdown()
    assert not out.viewers


def test_a_scripts_own_cancel_is_a_failure_not_a_stop(env):
    outcome = _run(env["console"], ("import threading\nev = threading.Event(); ev.set()\n"
                                    "project.run_polar(project.add_naca('0012'), cancel=ev)\n"))
    assert outcome.status == "failed" and outcome.error.startswith("Cancelled") and outcome.line == 3


def test_scripts_are_saved_in_the_project_and_listed(env, tmp_path):
    console, state = env["console"], env["state"]
    console.set_text("print('saved script')\n")
    path = console.save_script("my study")
    assert path == state.project.path / "scripts" / "my study.py" and path.read_text() == "print('saved script')\n"
    assert console.combo.currentText() == "my study.py"
    console.set_text("print('edited')\n")
    assert console.combo.currentText() == "my study.py *"
    assert console.run() and console.engine.wait(10)
    assert path.read_text() == "print('edited')\n"            # a script with a file is saved before running
    console.new_script()
    assert console.current_path is None and console.combo.findText("my study.py") >= 0
    assert console.open_script(path) and console.text() == "print('edited')\n"
    other = tmp_path / "elsewhere.py"
    other.write_text("print(1)\n", encoding="utf-8")
    assert console.open_script(other) and console.combo.currentText() == "elsewhere.py"


def test_examples_menu_and_files_match_the_snippets(env):
    from aerolume.app import scripting
    console = env["console"]
    keys = [s.key for s in scripting.SNIPPETS]
    assert {"polar_sweep", "alpha_sweep", "compare_designs", "mesh_3d", "wing_taper"} <= set(keys)
    for s in scripting.SNIPPETS:
        compile(s.code, s.key, "exec")
        assert (EXAMPLES / f"{s.key}.py").read_text(encoding="utf-8") == s.code, s.key
    titles = [a.text() for a in console.examples_button.menu().actions()]
    assert titles == [s.title for s in scripting.SNIPPETS]
    console.new_script()
    console.insert_snippet("wing_taper")
    assert console.text() == scripting.snippet("wing_taper").code and console.current_path is None
    console.insert_snippet("polar_sweep")                       # replaces an unchanged example
    assert console.text() == scripting.snippet("polar_sweep").code


def test_keys_win_over_the_window_only_while_the_console_has_focus(env):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtTest import QTest
    console = env["console"]
    console.set_text("print('from keyboard')\n")
    override = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_F5, Qt.KeyboardModifier.NoModifier)
    assert console.eventFilter(console.editor, override) and override.isAccepted()
    QTest.keyClick(console.editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert console.engine.wait(10) and "from keyboard" in console.output.plain_text()
    shift_f5 = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_F5, Qt.KeyboardModifier.ShiftModifier)
    assert not console.eventFilter(console.editor, shift_f5)        # Stop only while a script runs


def test_editor_highlights_indents_and_numbers_lines(env):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    console = env["console"]
    ed = console.editor
    console.set_text('def f(x):  # comment\n    return "text" + 3\n')
    formats = {fr.format.foreground().color().name() for fr in ed.document().firstBlock().layout().formats()}
    from aerolume.app import theme
    tok = theme.tokens()
    assert tok["accent"] in formats and tok["muted"] in formats         # keyword, comment
    second = {fr.format.foreground().color().name() for fr in ed.document().findBlockByNumber(1).layout().formats()}
    assert tok["good"] in second and tok["warn"] in second              # string, number
    console.set_text("if x:")
    cur = ed.textCursor()
    cur.movePosition(cur.MoveOperation.End)
    ed.setTextCursor(cur)
    QTest.keyClick(ed, Qt.Key.Key_Return)
    assert ed.toPlainText() == "if x:\n    "
    ed.selectAll()
    ed.toggle_comment()
    assert ed.toPlainText().startswith("# if x:")
    assert ed.gutter_width() > 20


def test_the_editor_is_not_a_keyboard_trap(env, qapp):
    """WCAG 2.1.2: Tab indents, so Ctrl+Tab, Esc-then-Tab and the "Tab key moves
    focus" toggle must each move focus out of the editor (and Shift back in)."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    console = env["console"]
    ed = console.editor
    console.activateWindow()

    def focus_editor():
        ed.setFocus(Qt.FocusReason.OtherFocusReason)
        qapp.processEvents()
        assert QApplication.focusWidget() is ed

    console.set_text("x = 1")
    cur = ed.textCursor()
    cur.movePosition(cur.MoveOperation.End)
    ed.setTextCursor(cur)
    focus_editor()
    QTest.keyClick(ed, Qt.Key.Key_Tab)                   # plain Tab still indents
    assert QApplication.focusWidget() is ed and ed.toPlainText().endswith("    ")
    QTest.keyClick(ed, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier)
    after_next = QApplication.focusWidget()
    assert after_next is not None and after_next is not ed
    assert ed.toPlainText() == "x = 1    "                # Ctrl+Tab inserted nothing
    QTest.keyClick(after_next, Qt.Key.Key_Tab)             # and the next control lets Tab go on
    assert QApplication.focusWidget() not in (None, after_next, ed)

    focus_editor()
    QTest.keyClick(ed, Qt.Key.Key_Backtab, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert QApplication.focusWidget() not in (None, ed, after_next)   # Ctrl+Shift+Tab goes back

    focus_editor()
    QTest.keyClick(ed, Qt.Key.Key_Escape)
    QTest.keyClick(ed, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is not ed           # Esc then Tab leaves
    assert ed.toPlainText() == "x = 1    "

    focus_editor()
    QTest.keyClick(ed, Qt.Key.Key_Escape)
    QTest.keyClick(ed, Qt.Key.Key_A)                       # another key cancels the escape
    QTest.keyClick(ed, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is ed

    console.act_tab_focus.setChecked(True)                 # More > Tab key moves focus
    QTest.keyClick(ed, Qt.Key.Key_Tab)
    assert QApplication.focusWidget() is not ed
    console.act_tab_focus.setChecked(False)
    assert "Ctrl+Tab" in ed.toolTip() and "Esc then Tab" in ed.toolTip()


def test_without_a_project_scripts_still_run(env):
    console, state = env["console"], env["state"]
    state.set_project(None)
    outcome = _run(console, "print(project is None)\n")
    assert outcome.ok and "True" in console.output.plain_text()
    assert console.combo.currentText() == "Untitled"


def test_engine_can_be_used_without_the_widget(env):
    """The assistant's run_python tool drives the engine directly."""
    from aerolume.app.scripting import ScriptEngine
    engine = ScriptEngine(env["state"])
    done = []
    engine.finished.connect(done.append)
    assert engine.run("print(2 + 2)\nresult = 'kept'\n")
    assert engine.wait(10) and done[0].ok and done[0].output == "4\n" and engine.namespace["result"] == "kept"
    assert engine.run("raise ValueError('bad input')\n")
    assert engine.wait(10) and done[1].error == "ValueError: bad input" and done[1].line == 1
    assert threading.active_count() < 50


def test_run_message_names_the_simulation_and_clears_when_the_script_ends(env):
    """The line beside the status is a simulation's progress, so it says so, and it goes when
    the script ends (a "Done after 22 s" beside "Running · 23 s" read as a contradiction)."""
    console = env["console"]
    outcome = _run(console, (
        'd = project.add_naca("2412")\n'
        'r = project.run_polar(d, alpha=(0, 4, 2))\n'))
    assert outcome.ok and console.message.full_text() == ""
    console.set_text("import time\nfor _ in range(400):\n    time.sleep(0.01)\n")
    assert console.run()
    console._on_progress(1.0, "Done after 22 s: Cd = 1.156")
    assert console.message.full_text() == "Simulation: Done after 22 s: Cd = 1.156"
    assert console.engine.wait(20)
    assert console.message.full_text() == "" and "Finished" in console.status.text()
    console._on_progress(1.0, "a late report")            # after the end: ignored
    assert console.message.full_text() == ""


def _figure_top(out, number):
    from PySide6.QtGui import QTextCursor
    cur = QTextCursor(out.document())
    cur.setPosition(out._figure_pos[number])
    return out.cursorRect(cur).top()


def test_output_scrolls_to_the_runs_first_figure_and_holds_there(env):
    """A run that prints, plots three figures and prints more ends with its first figure's
    top edge at the top of the output, not the tail; the next run follows the tail again."""
    from PySide6.QtCore import QCoreApplication
    console = env["console"]
    console.resize(1400, 380)
    QCoreApplication.processEvents()
    out = console.output
    outcome = _run(console, (
        "from matplotlib.figure import Figure\n"
        "for i in range(60):\n    print('line', i)\n"
        "for n in range(3):\n"
        "    fig = Figure(figsize=(6.4, 4.8)); fig.add_subplot().plot([0, 1], [n, 1 - n])\n"
        "    show(fig)\n"
        "for i in range(40):\n    print('after', i)\n"))
    QCoreApplication.processEvents()
    assert outcome.ok and out.figure_count() == 3 and out.is_holding_figure()
    bar = out.verticalScrollBar()
    assert 0 <= _figure_top(out, 1) <= 12 and bar.value() < bar.maximum()
    # a theme switch re-lays the output out: the view stays on the same figure
    from aerolume.app import theme
    theme.apply(QCoreApplication.instance(), "dark")
    QCoreApplication.processEvents()
    assert out.is_holding_figure() and 0 <= _figure_top(out, 1) <= 12
    theme.apply(QCoreApplication.instance(), "light")
    # the next run starts at the tail and, without figures, keeps following it -- also when the
    # pane is made shorter mid-run (the range grows under the view; it used to stop following)
    console.set_text("import time\nfor i in range(80):\n    print('second', i)\n    time.sleep(0.005)\n")
    assert console.run()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.2:
        QCoreApplication.processEvents()
    console.resize(1400, 240)
    assert console.engine.wait(20)
    QCoreApplication.processEvents()
    assert "second 79" in out.plain_text()
    assert not out.is_holding_figure() and bar.value() == bar.maximum()
    console.resize(1400, 380)
    QCoreApplication.processEvents()
    # scrolling to the end by hand during a held run lets go of the figure
    _run(console, (
        "from matplotlib.figure import Figure\n"
        "fig = Figure(figsize=(6.4, 4.8)); fig.add_subplot().plot([0, 1], [0, 1])\nshow(fig)\n"
        "for i in range(40):\n    print('tail', i)\n"))
    QCoreApplication.processEvents()
    assert out.is_holding_figure()
    bar.setValue(bar.maximum())
    assert not out.is_holding_figure()
    # a failure shows the traceback, not the figure
    outcome = _run(console, (
        "from matplotlib.figure import Figure\n"
        "fig = Figure(figsize=(6.4, 4.8)); fig.add_subplot().plot([0, 1], [0, 1])\nshow(fig)\n"
        "raise ValueError('boom')\n"))
    QCoreApplication.processEvents()
    assert not outcome.ok and not out.is_holding_figure() and bar.value() == bar.maximum()


def test_script_dock_keeps_its_share_of_the_window(env, qapp):
    """Shrinking the window caps the dock (the page keeps its minimum height); growing it
    again gives the dock its share back instead of handing every new pixel to the page."""
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QDockWidget, QFrame, QMainWindow, QVBoxLayout, QWidget

    from aerolume.app.widgets.script_console import ScriptConsole

    class Window(QMainWindow):
        SCRIPT_SHARE = 0.38

    win = Window()
    page = QWidget()
    page.setMinimumHeight(430)
    win.setCentralWidget(page)
    console = ScriptConsole(env["state"], env["jobs"])
    dock = QDockWidget("Script")
    frame = QFrame()                                        # the window frames dock content
    QVBoxLayout(frame).addWidget(console)
    dock.setWidget(frame)
    win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    win.resize(1600, 1000)
    win.show()
    QCoreApplication.processEvents()
    win.resizeDocks([dock], [380], Qt.Orientation.Vertical)
    QCoreApplication.processEvents()
    try:
        assert console.dock_share() is None
        win.resize(1280, 800)
        QCoreApplication.processEvents()
        QCoreApplication.processEvents()
        assert console.dock_share() == pytest.approx(0.38)
        squeezed = dock.height()
        assert squeezed < 0.38 * 800 + 4                    # capped by the page's minimum
        win.resize(1600, 1000)
        QCoreApplication.processEvents()
        QCoreApplication.processEvents()
        assert dock.height() == pytest.approx(0.38 * 1000, abs=12) and dock.height() > squeezed + 60
        # the user drags the separator: that height is the share kept from now on
        win.resizeDocks([dock], [300], Qt.Orientation.Vertical)
        QCoreApplication.processEvents()
        console._remember_share()
        assert console.dock_share() == pytest.approx(dock.height() / win.height())
        win.resize(1600, 900)
        QCoreApplication.processEvents()
        QCoreApplication.processEvents()
        assert dock.height() == pytest.approx(console.dock_share() * 900, abs=12)
    finally:
        console.shutdown()
        win.close()
        win.deleteLater()


def test_hosts_never_take_the_main_qthread_with_them(env, qapp):
    """Regression: AppHost once asked ``self._invoker.thread()`` from the script's worker thread,
    which tied the main QThread's wrapper to the invoker's; collecting an old host deleted the
    application's main QThread and the next style-sheet icon crashed Qt (an access violation in
    the fast suite). The host now records the GUI thread with ``threading.get_ident``."""
    import gc

    from aerolume.app.scripting import AppHost
    seen: list[bool] = []
    for _ in range(5):
        host = AppHost(env["state"], env["jobs"])
        cycle = {"host": host}
        cycle["self"] = cycle                         # collected by the cyclic GC, like a closed panel's
        assert host._on_gui_thread()
        worker = threading.Thread(target=lambda h=host: seen.append(h._on_gui_thread()))
        worker.start()
        worker.join()
        assert host.call(lambda: 41 + 1) == 42
        del host, cycle
        gc.collect()
        qapp.processEvents()
    assert seen == [False] * 5
    assert qapp.thread() is not None and qapp.thread().objectName() == "Qt mainThread"
