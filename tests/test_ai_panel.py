"""The Assistant dock and its place in the main window (offscreen Qt).

A scripted model drives the real panel inside the real window: project edits
go through ``AppState`` and the polar runs through the window's JobManager
(the run shows in the Project tree like any other), approvals are answered
with the card's own buttons, run_python goes through the Script engine, and
Preferences > Assistant writes the user config folder. The Script dock,
Tools > Run script… and the Project tree's Experiments group are covered too.
"""
from __future__ import annotations

import json
import time

import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


_WINDOWS: list = []


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    """Private config folder and settings, light theme, a projects folder; windows closed afterwards."""
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    (tmp_path / "projects").mkdir()
    yield tmp_path
    from PySide6.QtCore import QEvent
    while _WINDOWS:
        w = _WINDOWS.pop()
        try:
            w._quit_confirmed = True
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qapp.processEvents()
    settings.use_file(tmp_path / "settings-after.ini")


def _window(env, *, jobs: bool = False, project: bool = True):
    from aerolume.app.main_window import MainWindow
    w = MainWindow(probe_device=False, load_jobs=jobs)
    _WINDOWS.append(w)
    w.resize(1600, 1000)
    if project:
        w.new_study("airfoil", "Assistant test", env / "projects")
    return w


def _wait(qapp, cond, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return bool(cond())


def _config(**kw) -> None:
    from aerolume.ai.config import AssistantConfig, save_config
    save_config(AssistantConfig(**kw))


def _ready(qapp, panel, provider) -> None:
    panel.provider_factory = lambda _cfg: provider
    panel.refresh_provider()
    assert _wait(qapp, lambda: panel._provider_ready is not None, 10)


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

def test_window_offers_the_assistant_and_the_script_docks(env, qapp):
    from aerolume.app.widgets.assistant_panel import AssistantPanel
    from aerolume.app.widgets.script_console import ScriptConsole
    w = _window(env, project=False)
    assert isinstance(w.assistant, AssistantPanel) and isinstance(w.script_console, ScriptConsole)
    assert w.assistant_dock.windowTitle() == "Assistant" and w.script_dock.windowTitle() == "Script"
    assert w.assistant_dock.isHidden() and w.script_dock.isHidden()          # hidden until asked for
    view = next(a.menu() for a in w.menuBar().actions() if a.text() == "&View")
    shortcuts = {a.text(): a.shortcut().toString() for a in view.actions()}
    assert shortcuts["Assistant panel"] == "Alt+3" and shortcuts["Script panel"] == "Alt+4"
    tools = next(a.menu() for a in w.menuBar().actions() if a.text() == "&Tools")
    names = [a.text() for a in tools.actions()]
    assert "Experiment &assistant" in names and "Run &script…" in names
    help_menu = next(a.menu() for a in w.menuBar().actions() if a.text() == "&Help")
    assert {"Experiment &assistant guide", "&Scripting guide"} <= {a.text() for a in help_menu.actions()}
    assert w.show_assistant() is False                                       # experiments need a project
    assert not w.assistant.run_button.isEnabled()
    w.new_study("airfoil", "Docks", env / "projects")
    assert w.show_assistant("Find the best L/D") is True
    assert not w.assistant_dock.isHidden() and w.assistant.goal.toPlainText() == "Find the best L/D"
    assert w.assistant.run_button.isEnabled()
    w.show_script()
    assert not w.script_dock.isHidden()
    dlg = w.open_preferences(exec_=False, tab="assistant")
    assert dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Assistant" and dlg.assistant is not None
    assert w.help_path("assistant").name == "ai_assistant.md" and w.help_path("scripting").name == "scripting.md"


def test_empty_state_examples_and_script_link(env, qapp):
    from aerolume.app.widgets.assistant_panel import EXAMPLES
    w = _window(env)
    panel = w.assistant
    w.show_assistant()
    d = w.state.design
    # every example is drawn from the open project (critic round 3.4: an enclosure project was offered
    # NACA 4412, Ahmed bodies and a wing); none of the generic ones, which are for a project without designs
    shown = [b for b in panel.example_buttons if b.isVisibleTo(panel)]
    goals = [b.property("goal") for b in shown]
    assert len(shown) >= 2 and all(f"“{d.name}”" in g for g in goals) and not set(goals) & set(EXAMPLES)
    shown[1].linkActivated.emit("goal")
    assert panel.goal.toPlainText() == goals[1]
    assert all(b.wordWrap() for b in panel.example_buttons)                  # long goals wrap, never clip
    # the first example is about the design selected in the window, worded for its kind
    assert d is not None and panel.selection_example.isVisibleTo(panel)
    goal = panel.selection_example.property("goal")
    assert f"“{d.name}”" in goal and "best L/D" in goal
    panel.selection_example.linkActivated.emit("goal")
    assert panel.goal.toPlainText() == goal
    # the introduction gets every line it needs at the dock's width (it was clipped at both ends), and the
    # empty state reads from its heading: showing the selection example used to scroll it to the bottom
    w.resize(1600, 1000)
    _wait(qapp, lambda: False, 0.3)
    intro = panel.intro
    assert intro.wordWrap() and intro.height() >= intro.heightForWidth(intro.width()) >= intro.fontMetrics().height()
    w.assistant_dock.resize(300, w.assistant_dock.height())                  # a narrow dock: more lines, all shown
    _wait(qapp, lambda: False, 0.3)
    assert intro.height() >= intro.heightForWidth(intro.width())
    assert panel.scroll.verticalScrollBar().value() == 0
    panel.script_hint.linkActivated.emit("script")
    assert not w.script_dock.isHidden()


def test_script_dock_opens_tall_enough_and_folds_messages(env, qapp):
    """Script opens at about 38 % of the window (it was 150 px: five lines and a sliver of each figure),
    Messages folds to its title row (both share the bottom), and the Project tree is wide enough to read."""
    from PySide6.QtCore import Qt

    from aerolume.app import settings
    from aerolume.app.main_window import DockTitle
    for size in ((1600, 1000), (1280, 800)):
        settings.use_file(env / f"settings-{size[0]}.ini")                  # a first start at each size
        w = _window(env, project=False)
        w.resize(*size)
        w.show()
        w.new_study("airfoil", f"Script {size[0]}", env / "projects")
        _wait(qapp, lambda: False, 0.3)
        assert w.project_dock.width() >= w.PROJECT_DOCK_MIN_KEPT
        w.state.log("solver diverged", "error", "Run")             # the log is open when Script appears
        _wait(qapp, lambda: False, 0.2)
        assert w.messages_expanded()
        w.show_script()
        _wait(qapp, lambda: False, 0.4)
        want = w.script_height()
        # the page above keeps its minimum height, so a small window gives Script what is left
        room = w.script_dock.height() + w.centralWidget().height() - w.centralWidget().minimumSizeHint().height()
        assert want >= 300 and abs(w.script_dock.height() - min(want, room)) <= 24, (size, w.script_dock.height())
        if size == (1600, 1000):
            assert abs(w.script_dock.height() - want) <= 24                 # 38 % of the window
        assert not w.messages_expanded() and w.messages_dock.height() <= DockTitle.HEIGHT
        # the user's later choice is kept: hiding and showing again does not resize it
        w.resizeDocks([w.script_dock], [260], Qt.Orientation.Vertical)
        _wait(qapp, lambda: False, 0.2)
        h = w.script_dock.height()
        w.script_dock.hide()
        w.show_script()
        _wait(qapp, lambda: False, 0.3)
        assert abs(w.script_dock.height() - h) <= 24
        w.close()


def test_a_repeated_answer_is_shown_once(qapp):
    """After "record the conclusion, then answer again" a model repeats itself: the earlier copy is hidden."""
    from aerolume.app.widgets.assistant_panel import _Turn
    t = _Turn("goal", "Local · qwen3:8b")
    t.append_text("Best L/D 29.51 at α = 3° ± 0.5°.")
    t.close_segment()
    t.append_text("Something else first.")
    t.close_segment()
    t.append_text("Best L/D 29.51  at α = 3° ± 0.5°.\n")
    assert t.drop_repeated_answers() == 1
    assert t.answers[0].isHidden() and not t.answers[1].isHidden() and not t.answers[2].isHidden()


def test_selection_example_follows_the_kind_of_design():
    from types import SimpleNamespace as NS

    from aerolume.app.widgets.assistant_panel import selection_example
    car = NS(name="sedan mm", kind="vehicle", dimension=3)
    assert selection_example(car, 30.0) == ("How does yaw 0–10° change the drag of “sedan mm” at 30 m/s? "
                                            "Estimate the cost first.")
    assert "how much should I trust it" in selection_example(NS(name="Sphere", kind="generic", dimension=3))
    assert "how confident" in selection_example(NS(name="Wing", kind="wing", dimension=3), 25.0)
    assert "polar analysis" in selection_example(NS(name="NACA 2412", kind="airfoil", dimension=2))
    assert selection_example(None) == ""


def test_markdown_tables_follow_the_theme(env, qapp):
    from aerolume.app import theme
    from aerolume.app.widgets.assistant_panel import markdown_html
    md = "Best **L/D**:\n\n| Quantity | Value |\n|---|---:|\n| L/D | 108.7 |\n"
    for mode in ("light", "dark"):
        theme.apply(qapp, mode)
        out = markdown_html(md)
        assert f"border-color: {theme.tokens()['border']}" in out and "108.7" in out
    theme.apply(qapp, "light")


# ---------------------------------------------------------------------------
# an experiment in the dock
# ---------------------------------------------------------------------------

def test_dock_runs_an_experiment_through_the_job_manager(env, qapp):
    _config(approval_threshold_s=1e6)
    w = _window(env, jobs=True)
    panel = w.assistant
    did = next(d.id for d in w.state.project.designs.values())
    seen: dict = {}

    def plot(_s, _u, results):
        data = json.loads(results[1].content)["data"]
        seen.update(data)
        return tools_turn(call("c4", "make_plot", {"kind": "polar", "simulation_ids": [data["simulation_id"]]}))

    def answer(_s, _u, _r):
        m = seen["metrics"]
        return final(f"Best L/D **{m['ld_max']:.4g}** at α = {m['alpha_ld_max']:g}°.\n\n"
                     f"| Quantity | Value |\n|---|---:|\n| Best L/D | {m['ld_max']:.4g} |\n")

    prov = fakes.ScriptedProvider([
        tools_turn(call("c1", "project_overview"), text="Reading the project."),
        tools_turn(call("c2", "record_plan", {"objective": "α of best L/D", "method": "polar",
                                              "variables": ["α 2..4°"], "expected_seconds": 10}),
                   call("c3", "run_polar", {"design_id": did, "speed_m_s": 20, "alpha_start": 2, "alpha_end": 4,
                                            "alpha_step": 1})),
        plot, answer])
    w.show_assistant()
    _ready(qapp, panel, prov)
    # the header names the model; the check's text (server version and so on) is the tooltip
    assert panel.model_label.text() == "Scripted · test" and panel.model_label.toolTip() == "Scripted · test · ready"
    panel.set_goal("Find the angle of attack with the best L/D of NACA 2412 at 20 m/s.")
    assert panel.submit() is True
    assert panel.stop_button.isVisibleTo(panel) and not panel.run_button.isVisibleTo(panel)
    assert _wait(qapp, lambda: not panel.is_running(), 240), "the experiment did not finish"
    _wait(qapp, lambda: False, 0.3)
    turn = panel._turn
    assert list(turn.rows) == ["c1", "c2", "c3", "c4"]
    assert {r.status for r in turn.rows.values()} == {"ok"}
    assert not turn.rows["c2"].extra.isHidden()                              # the plan is shown under its row
    assert turn.figures and not turn.figures[0].pixmap.isNull()
    text = panel.transcript_text()
    assert "Best L/D" in text and f"{seen['metrics']['ld_max']:.4g}" in text
    assert "Done in" in text
    # the run went through the JobManager and is an ordinary simulation of the project
    sid = seen["simulation_id"]
    sim = w.state.project.simulations[sid]
    assert sim.status == "done" and sim.analysis == "polar"
    assert sid in w.project_tree._items
    # the log is listed in the Project tree and opens in a document window
    exps = w.project_tree.experiments()
    assert len(exps) == 1 and exps[0]["status"] == "done"
    from aerolume.app.widgets.project_tree import EXPERIMENTS_GROUP
    assert EXPERIMENTS_GROUP in w.project_tree._items
    dlg = w.open_experiment(exps[0]["path"], show=False)
    assert dlg is not None and "Experiment" in dlg.windowTitle()
    _assert_document_follows_the_guidelines(dlg.view.document())
    assert panel.run_button.isVisibleTo(panel) and panel.new_button.isEnabled()
    panel.new_experiment()
    assert panel._turn is None and panel.empty.isVisibleTo(panel)


def test_approval_cards_code_and_stop(env, qapp):
    _config(approval_threshold_s=0.0)
    w = _window(env)
    panel = w.assistant
    did = next(d.id for d in w.state.project.designs.values())
    polar = {"design_id": did, "speed_m_s": 20, "alpha_start": 2, "alpha_end": 4, "alpha_step": 1}
    prov = fakes.ScriptedProvider([
        tools_turn(call("r1", "run_polar", polar)),
        tools_turn(call("p1", "run_python", {"code": "result = {'designs': len(project.designs)}",
                                             "purpose": "count the designs"})),
        final("Counted."),
        tools_turn(call("r2", "run_polar", polar)),
        final("never")])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal("count")
    panel.submit()
    turn = panel._turn
    assert _wait(qapp, lambda: "r1" in turn.cards, 20)
    card = turn.cards["r1"]
    # the estimate is a row of the card, so the heading leaves it out (the full title is the tooltip)
    assert card.title.text() == "Polar of NACA 2412" and card.code is None
    assert card.title.toolTip().startswith("Polar of NACA 2412: about")
    # the card lists exactly what will run, row by row, not only how long it takes
    assert card.items is not None
    rows = {k.text(): v.text() for k, v in card.items._rows}
    assert rows["Design"].startswith("NACA 2412") and rows["Analysis"].startswith("Airfoil polar")
    assert rows["Speed"] == "20 m/s" and rows["Angles"] == "α 2° to 4°, step 1° (3 angles)"
    assert rows["Transition"] == "ncrit 9" and rows["Estimated time"].startswith("about ")
    card.decline.click()
    assert _wait(qapp, lambda: "p1" in turn.cards, 20)
    code_card = turn.cards["p1"]
    assert code_card.code is not None and "len(project.designs)" in code_card.code.toPlainText()
    assert code_card.approve.text() == "Run"
    code_card.approve.click()
    assert _wait(qapp, lambda: not panel.is_running(), 60)
    assert turn.rows["r1"].status == "declined" and turn.rows["p1"].status == "ok"
    assert prov.sessions[0].last_result("run_python")["data"]["result"] == {"designs": 1}
    assert not card.outcome.isHidden() and card.outcome.text() == "Declined"
    # a follow-up in the same experiment; Stop while its approval card waits
    panel.set_goal("now the polar")
    panel.submit()
    turn2 = panel._turn
    assert _wait(qapp, lambda: "r2" in turn2.cards, 20)
    panel.stop()
    assert _wait(qapp, lambda: not panel.is_running(), 20)
    assert "Stopped" in panel.transcript_text()
    assert len(prov.sessions) == 1                                           # same conversation, same log
    from aerolume.ai.experiment import load_record
    rec = load_record(panel.agent.experiment_path)
    assert rec.status == "stopped" and [a.status for a in rec.actions] == ["declined", "ok"]


def _assert_document_follows_the_guidelines(doc) -> None:
    """Headings at most 11 pt semibold, tables with a 1 px border in the border token (docs/ui_guidelines.md)."""
    from PySide6.QtGui import QTextFormat, QTextTable

    from aerolume.app import theme
    block, headings = doc.begin(), 0
    while block.isValid():
        if block.blockFormat().headingLevel() > 0:
            headings += 1
            fmt = block.begin().fragment().charFormat()
            assert 0 < fmt.fontPointSize() <= 11.0 and fmt.fontWeight() >= 600, block.text()
            # Qt's relative heading size would otherwise be applied on top of the point size
            assert not fmt.intProperty(QTextFormat.Property.FontSizeAdjustment), block.text()
        block = block.next()
    assert headings >= 2
    tables = [f for f in doc.rootFrame().childFrames() if isinstance(f, QTextTable)]
    assert tables
    for t in tables:
        tf = t.format()
        assert tf.border() == 1 and tf.cellSpacing() == 0 and tf.borderCollapse()
        assert tf.borderBrush().color().name() == theme.tokens()["border"]


def test_an_answer_claiming_an_action_that_did_not_happen_is_withheld(env, qapp):
    _config(approval_threshold_s=1e6)
    w = _window(env)
    panel = w.assistant
    claim = "The design NACA 2412 has been deleted from the project."
    prov = fakes.ScriptedProvider([final(claim), final(claim)])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal("Delete the design NACA 2412")
    panel.submit()
    assert _wait(qapp, lambda: not panel.is_running(), 30)
    _wait(qapp, lambda: False, 0.2)
    turn = panel._turn
    assert turn.answers and all(b.isHidden() for b in turn.answers)      # the false answers left the flow
    assert turn.answers[0].property("role") == "muted"                  # superseded when sent back
    text = panel.transcript_text()
    assert "Not verified" in text and "Nothing was deleted" in text and "Done in" not in text
    from aerolume.app.widgets.common import CollapsibleSection
    folded = [c for c in turn.findChildren(CollapsibleSection) if "Withheld" in c.header.text()]
    assert folded and not folded[0].is_expanded()
    assert w.project_tree.experiments()[0]["status"] == "unverified"
    assert any(d.name == "NACA 2412" for d in w.state.project.designs.values())


def test_unavailable_provider_is_explained(env, qapp):
    w = _window(env)
    panel = w.assistant
    w.show_assistant()
    _ready(qapp, panel, fakes.ScriptedProvider([], ready=False))
    assert not panel.setup_line.isHidden() and "unavailable" in panel.setup_line.text.text()
    assert not panel.setup_button.isHidden()
    asked = []
    panel.preferencesRequested.connect(lambda: asked.append(True))
    panel.setup_button.click()
    assert asked == [True]


def test_provider_errors_offer_the_settings(env, qapp):
    from aerolume.ai.providers.base import ProviderError

    def boom(*_a):
        raise ProviderError("Ollama is not reachable at http://localhost:11434.", "connection")
    w = _window(env)
    panel = w.assistant
    w.show_assistant()
    _ready(qapp, panel, fakes.ScriptedProvider([boom]))
    panel.set_goal("anything")
    panel.submit()
    assert _wait(qapp, lambda: not panel.is_running(), 20)
    text = panel.transcript_text()
    assert "not reachable" in text and "Failed" in text


# ---------------------------------------------------------------------------
# Preferences > Assistant
# ---------------------------------------------------------------------------

def test_assistant_preferences_store_settings_and_key_apart(env, qapp):
    from aerolume.ai import config as C
    from aerolume.app.widgets.assistant_panel import AssistantSettingsPage
    with fakes.FakeOllama() as srv:
        page = AssistantSettingsPage(list_models=False)
        page.endpoint.setText(srv.url)
        page.refresh_models()
        assert _wait(qapp, lambda: page.local_model.count() > 0, 10)
        assert [page.local_model.itemText(i) for i in range(page.local_model.count())] == ["qwen3:8b"]
        assert page.models_status.text() == "Ollama 0.40.2 · 1 model with tool calling"
    page.provider.setCurrentIndex(page.provider.findData("ollama"))
    page.threshold.setValue(30)
    page.think.setChecked(True)
    secret = "sk-ant-api03-not-a-real-key-0000-WXYZ"
    page.key.setText(secret)
    assert page.apply() is True
    saved = json.loads(C.settings_path().read_text(encoding="utf-8"))
    assert saved["provider"] == "ollama" and saved["approval_threshold_s"] == 30 and saved["ollama_think"] is True
    assert saved["ollama_endpoint"] == srv.url
    assert secret not in C.settings_path().read_text(encoding="utf-8")
    assert C.stored_key() == secret and page.key.text() == ""                # never shown again
    assert "sk-ant-…WXYZ" in page.key_status.text() and secret not in page.key_status.text()
    from aerolume.app import settings
    settings.get().sync()
    ini = (env / "settings.ini")
    assert not ini.exists() or secret not in ini.read_text(encoding="utf-8", errors="replace")
    assert page.apply() is False                                             # nothing changed
    page.remove_key.click()
    assert page.apply() is True and C.stored_key() == ""
    again = AssistantSettingsPage(list_models=False)
    assert again.values() == C.load_config()


# ---------------------------------------------------------------------------
# Script dock and experiment logs in the tree
# ---------------------------------------------------------------------------

def test_tools_run_script_runs_a_file_in_the_script_dock(env, qapp):
    w = _window(env)
    path = env / "hello.py"
    path.write_text("print('designs:', len(project.designs))\n", encoding="utf-8")
    assert w.run_script_file(path) is True
    console = w.script_console
    assert not w.script_dock.isHidden()
    assert _wait(qapp, lambda: not console.is_running(), 30)
    assert "designs: 1" in console.output.plain_text()


def test_project_tree_lists_and_deletes_experiment_logs(env, qapp, monkeypatch):
    from aerolume.ai.experiment import ExperimentLog
    w = _window(env)
    root = w.state.project.path
    log = ExperimentLog(root, "Compare two airfoils", provider="ollama", model="Local · qwen3:8b")
    log.set_status("done")
    tree = w.project_tree
    tree.refresh_experiments()
    exps = tree.experiments()
    assert [e["goal"] for e in exps] == ["Compare two airfoils"] and exps[0]["status"] == "done"
    item = tree._items[str(log.path)]
    assert item.text() == "Compare two airfoils"
    opened = []
    tree.experimentActivated.connect(opened.append)
    tree.open_experiment(str(log.path))
    assert opened == [str(log.path)]
    monkeypatch.setattr(tree, "_confirm", lambda *_a: True)
    assert tree.delete_experiment(str(log.path)) is True
    assert not log.path.exists() and tree.experiments() == []
    assert tree.delete_experiment(str(env)) is False                         # only logs inside experiments/


# ---------------------------------------------------------------------------
# the end of a run: busy until its outcome is shown
# ---------------------------------------------------------------------------

class _InstantAgent:
    """An agent stand-in whose request emits its outcome and returns at once.

    ``session`` is set, so the panel asks no consent; the events go through the panel's own queued
    bridge, exactly as the real agent's do, so they wait in the event queue after the worker exits.
    """

    session = object()

    def __init__(self, panel, exc: BaseException | None = None) -> None:
        self.panel, self.exc, self.stopped = panel, exc, False

    def run(self, text: str) -> None:
        if self.exc is not None:
            raise self.exc
        emit = self.panel._bridge.event.emit
        emit("status", {"text": "Waiting for the model"})
        emit("text", {"delta": f"Answer to: {text}"})
        emit("done", {"status": "done", "text": "", "untraced": [], "path": ""})

    def stop(self) -> None:
        self.stopped = True


def _submit_and_let_the_worker_exit(qapp, panel, agent, goal: str = "first goal") -> None:
    """Submit with ``agent`` and wait for the worker thread to end WITHOUT handling its events."""
    _ready(qapp, panel, fakes.ScriptedProvider([]))
    panel.agent = agent
    panel.set_goal(goal)
    assert panel.submit()
    thread = panel._thread
    thread.join(10)
    assert not thread.is_alive()                 # the worker is gone; its "done" is still queued


def test_the_panel_stays_busy_until_its_runs_outcome_is_shown(env, qapp):
    """The worker exits before the GUI thread handles its queued "done"; in that gap the run is not over.

    Found in the full suite: a test waited for ``not is_running()`` and then read "Waiting for the model"
    in the status line, because ``is_running`` followed the thread. In the same gap ``submit`` accepted a
    second goal, and the first run's late "done" then ended the new turn.
    """
    w = _window(env)
    w.show_assistant()
    panel = w.assistant
    _submit_and_let_the_worker_exit(qapp, panel, _InstantAgent(panel))
    first = panel._turn
    assert panel.is_running()
    assert panel.stop_button.isVisibleTo(panel) and not panel.run_button.isVisibleTo(panel)
    panel.set_goal("second goal")
    assert panel.submit() is False and panel._turn is first        # refused: the first run is not over
    assert panel.goal.toPlainText() == "second goal"                # and the goal is kept for later
    assert _wait(qapp, lambda: not panel.is_running(), 10)
    assert panel.run_button.isVisibleTo(panel) and not panel.stop_button.isVisibleTo(panel)
    assert panel.status_text.full_text() == "Waiting for the model"  # the last status, then the outcome row
    assert "Done" in panel.transcript_text() and "Answer to: first goal" in panel.transcript_text()


def test_a_run_whose_conversation_was_reset_still_ends(env, qapp):
    """Another project opened while a request ran: its "done" has no turn to draw into, yet ends the run.

    ``_project_changed`` stops the request and resets the conversation; the late "done" used to be
    dropped with the turn, which left Stop and the status row on screen for good.
    """
    w = _window(env)
    w.show_assistant()
    panel = w.assistant
    agent = _InstantAgent(panel)
    _submit_and_let_the_worker_exit(qapp, panel, agent)
    panel._project_changed()
    assert agent.stopped and panel._turn is None and panel.is_running()
    assert _wait(qapp, lambda: not panel.is_running(), 10)
    assert panel.run_button.isVisibleTo(panel) and not panel.stop_button.isVisibleTo(panel)
    assert not panel.status_row.isVisibleTo(panel)


def test_a_worker_that_dies_of_a_base_exception_still_ends_the_run(env, qapp, monkeypatch):
    """``is_running`` waits for a "done", so every way out of the worker must send one."""
    import threading

    class _Abort(BaseException):
        pass

    escaped: list[type] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: escaped.append(args.exc_type))
    w = _window(env)
    w.show_assistant()
    panel = w.assistant
    _submit_and_let_the_worker_exit(qapp, panel, _InstantAgent(panel, exc=_Abort()))
    assert escaped == [_Abort]                   # re-raised after the "done": not swallowed
    assert _wait(qapp, lambda: not panel.is_running(), 10)
    assert "_Abort" in panel.transcript_text() and "Failed" in panel.transcript_text()
