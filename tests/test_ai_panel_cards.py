"""The Assistant dock's cards (offscreen Qt): suggestions, the claims check, consent.

Lumen's assistant rules, as the panel enforces them in the real window with a
scripted model:

* a suggestion arrives as a card listing every change old -> new; nothing changes
  until Apply; values are checked again by the API on Apply; one Undo takes the
  whole suggestion back; Dismiss changes nothing;
* every final answer's statements are listed with the claims check's verdict
  (folded when all are verified, open and marked "not verified" when not);
* nothing is sent to a model off this computer until the user agrees on a card.
"""
from __future__ import annotations

import pytest
import test_ai_fakes as fakes
import test_ai_panel as panel_tests
from test_ai_fakes import call, final, tools_turn
from test_ai_panel import _config, _ready, _wait, _window

# the dock's fixtures (session QApplication, private config folder and settings), shared with test_ai_panel
qapp = panel_tests.qapp
env = panel_tests.env

pytestmark = pytest.mark.gui


def _cfd_sim(w, panel):
    """A new (not run) 2D CFD simulation of the window's airfoil at 20 m/s, Draft quality."""
    did = next(d.id for d in w.state.project.designs.values())
    return panel._api_project().add_simulation(did, "cfd", speed=20, alpha=2)


# ---------------------------------------------------------------------------
# suggestions
# ---------------------------------------------------------------------------

def test_a_suggestion_changes_nothing_until_apply_and_undo_restores_it(env, qapp):
    _config(approval_threshold_s=1e6)
    w = _window(env)
    panel = w.assistant
    sim = _cfd_sim(w, panel)
    sid = sim.id
    assert sim.cfd.quality == "draft"
    prov = fakes.ScriptedProvider([
        tools_turn(call("d1", "describe_simulation", {"simulation_id": sid})),
        tools_turn(call("s1", "suggest_settings", {
            "simulation_id": sid, "reason": "A finer grid resolves the boundary layer better.",
            "changes": [{"setting": "quality", "value": "standard"}, {"setting": "speed", "value": 5000}]})),
        final("I suggest the Standard quality; nothing changes until you click Apply on the card.")])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal("How could this simulation be made more accurate?")
    panel.submit()
    assert _wait(qapp, lambda: not panel.is_running(), 30)
    turn = panel._turn
    assert len(turn.suggestions) == 1
    card = turn.suggestions[0]
    store_sim = w.state.project.simulations[sid]
    assert store_sim.cfd.quality == "draft" and store_sim.conditions.speed == 20     # proposing changed nothing
    rows = {r["setting"]: r for r in card.rows}
    assert rows["quality"]["valid"] and rows["quality"]["old"] == "draft" and rows["quality"]["new"] == "standard"
    assert not rows["speed"]["valid"]
    assert card.apply.text() == "Apply 1 change" and card.apply.isEnabled()
    # the refused row is struck out with its reason, and is never applied
    # (as rich text: a font set in code is lost when the theme's stylesheet polishes the label)
    struck = [lab for lab in card.findChildren(type(card.title)) if lab.property("struck")]
    assert len(struck) == 3 and any(lab.text() == "<s>5000 m/s</s>" for lab in struck)
    assert any(lab.text().startswith("Not applied:") for lab in card.findChildren(type(card.title)))
    assert card.changes() == {"quality": "standard"}

    card.apply.click()
    qapp.processEvents()
    applied = w.state.project.simulations[sid]
    assert applied.cfd.quality == "standard" and applied.conditions.speed == 20     # only the valid change
    assert card.undo.isVisibleTo(card) and not card.apply.isVisibleTo(card)
    assert card.outcome.text().startswith("Applied to")
    card.undo.click()                                                           # one step takes it all back
    qapp.processEvents()
    assert w.state.project.simulations[sid].cfd.quality == "draft"
    assert card.outcome.text().startswith("Undone") and not card.undo.isVisibleTo(card)
    assert "Done in" in panel.transcript_text()


def test_dismiss_changes_nothing_and_a_refused_apply_is_shown(qapp):
    from aerolume.app import theme
    from aerolume.app.widgets.assistant_panel import _SuggestionCard
    theme.apply(qapp, "light")
    applied: list = []

    def apply_fn(sim_id, changes):
        applied.append((sim_id, changes))
        raise ValueError("duration must be between 1 and 1000 L/U")

    data = {"simulation_id": "abc", "simulation": "CFD · 20 m/s · α 2°", "reason": "Average longer.",
            "has_result": True,
            "changes": [{"setting": "duration", "label": "Duration", "unit": "L/U", "old": 8.0, "new": 40.0,
                         "valid": True}]}
    card = _SuggestionCard(data, apply_fn, lambda token: None)
    labels = [lab.text() for lab in card.findChildren(type(card.title))]
    assert "8 L/U" in labels and "40 L/U" in labels and "→" in labels
    assert any("makes an edited copy" in t for t in labels)                  # a result is never re-labelled
    card.apply.click()
    assert applied == [("abc", {"duration": 40.0})]
    assert card.outcome.text().startswith("Not applied: duration must be between")  # the API's refusal, shown
    assert card.apply.isVisibleTo(card)                                       # still there to try again
    card.dismiss.click()
    assert card.outcome.text() == "Dismissed: nothing changed" and not card.apply.isVisibleTo(card)
    card.deleteLater()


# ---------------------------------------------------------------------------
# the claims check in the panel
# ---------------------------------------------------------------------------

def test_unsupported_statements_are_listed_open_and_the_answer_ends_not_verified(env, qapp):
    _config(approval_threshold_s=1e6)
    w = _window(env)
    panel = w.assistant
    sid = _cfd_sim(w, panel).id
    wrong = "The simulation runs at 20 m/s. Its grid has 48213 cells."
    prov = fakes.ScriptedProvider([tools_turn(call("d1", "describe_simulation", {"simulation_id": sid})),
                                   final(wrong), final(wrong)])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal("Describe the selected simulation.")
    panel.submit()
    assert _wait(qapp, lambda: not panel.is_running(), 30)
    _wait(qapp, lambda: False, 0.2)
    turn = panel._turn
    # the first answer was sent back once with the problem listed, then shown superseded
    sent = prov.sessions[0].sent[2]["user_text"]
    assert "48213" in sent
    assert turn.answers[0].property("role") == "muted"
    block = turn.claims
    assert block is not None and block.has_problems()
    verdicts = [s["status"] for s in block.statements]
    assert verdicts == ["verified", "unsupported"]
    assert block.section.is_expanded()                                        # open when something is wrong
    assert block.section.header.text() == "Checked statements · 1 not verified"
    assert block.summary.text() == "2 statements checked: 1 verified, 1 not supported"
    # the block never sets the transcript's width: it wraps inside the dock instead of running off its edge
    assert block.minimumSizeHint().width() <= 260
    text = panel.transcript_text()
    assert "1 statement not verified" in text and "Done in" not in text
    assert "Not supported: 48213 is in no tool result" in text
    assert w.project_tree.experiments()[0]["status"] == "unverified"


def test_verified_statements_are_folded_away_with_their_evidence(env, qapp):
    _config(approval_threshold_s=1e6)
    w = _window(env)
    panel = w.assistant
    sid = _cfd_sim(w, panel).id
    prov = fakes.ScriptedProvider([tools_turn(call("d1", "describe_simulation", {"simulation_id": sid})),
                                   final("The simulation runs at 20 m/s and has no result yet.")])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal("Describe the selected simulation.")
    panel.submit()
    assert _wait(qapp, lambda: not panel.is_running(), 30)
    block = panel._turn.claims
    assert block is not None and not block.has_problems()
    assert [s["status"] for s in block.statements] == ["verified"]
    assert not block.section.is_expanded()
    assert block.section.header.text() == "Checked statements · all verified"
    assert block.summary.text() == "1 statement checked: 1 verified"
    assert "Done in" in panel.transcript_text()
    assert len(prov.sessions[0].sent) == 2                                    # nothing was sent back


# ---------------------------------------------------------------------------
# consent before anything leaves the computer
# ---------------------------------------------------------------------------

class _RemoteProvider(fakes.ScriptedProvider):
    """A scripted model that counts as Claude: its conversation would leave this computer."""
    name = "claude"


def test_nothing_is_sent_off_the_computer_until_the_user_agrees(env, qapp, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _config(approval_threshold_s=1e6)
    w = _window(env)
    panel = w.assistant
    prov = _RemoteProvider([final("Hello."), final("Hello again.")])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal("What is in this project?")
    assert panel.submit() is True
    card = panel._consent_card
    assert card is not None and "Anthropic (Claude API)" in panel.transcript_text()
    facts = {k.text(): v.text() for k, v in card.facts._rows}
    assert "tools' results" in facts["Sent"] and "meshes" in facts["Not sent"]
    assert prov.sessions == [] and not panel.is_running()
    card.cancel.click()
    qapp.processEvents()
    assert prov.sessions == [] and "Nothing was sent" in panel.transcript_text()
    assert panel.goal.toPlainText() == "What is in this project?"             # the goal is kept for later

    assert panel.submit() is True
    card2 = panel._consent_card
    assert card2 is not card
    card2.remember.setChecked(True)
    card2.allow.click()
    assert _wait(qapp, lambda: len(prov.sessions) == 1 and not panel.is_running(), 30)
    from aerolume.ai.config import load_config
    assert "claude" in load_config().consent                                  # remembered in Preferences
    panel.new_experiment()
    panel.set_goal("And again?")
    panel.submit()
    assert _wait(qapp, lambda: len(prov.sessions) == 2 and not panel.is_running(), 30)
    assert panel._consent_card is card2                                       # no new card was needed


# ---------------------------------------------------------------------------
# answers the loop sent back
# ---------------------------------------------------------------------------

def test_every_answer_sent_back_is_shown_superseded_and_the_reference_row_is_titled(env, qapp):
    """Critic round 3's Lange transcript in the real dock: no comparison, then a Boussinesq claim, then the answer.

    A correction notice queued right behind its answer's text used to find the new
    block still "hidden" (Qt shows a widget added to a visible layout a moment
    later) and left the answer it sent back in the normal colour.
    """
    import test_ai_reference as ref

    from aerolume import api
    from aerolume.app.widgets.assistant_panel import TOOL_TITLES
    w = _window(env)
    panel = w.assistant
    proj = panel._api_project()
    sim = proj.add_simulation(api.heated_cylinder_case(re=40), quality="draft")
    ref._store(proj, sim.id, ref._cfd(sim.id, 3.3402, 0.001, re=40.0))
    lange = round(api.lange_nusselt(40.0), 4)
    diff = 100.0 * (3.3402 - lange) / lange
    prov = ref.LocalScripted([
        tools_turn(call("g1", "get_results", {"simulation_ids": [sim.id]})),
        tools_turn(call("c1", "record_conclusion", {"summary": "Nu", "findings": []})),
        final(ref.CYL_ANSWER),
        final(f"Nu is 3.3402 ± 0.001 against {lange:.4g} (Lange): {diff:+.1f} %. The assumptions made in the "
              "Boussinesq energy equation limit it."),
        final(f"Nu is 3.3402 ± 0.001 (standard error) against {lange:.4g} from the Lange et al. correlation: "
              f"{diff:+.1f} %. Buoyancy was off (forced convection).")])
    w.show_assistant()
    _ready(qapp, panel, prov)
    panel.set_goal(ref.CYL_GOAL)
    panel.submit()
    assert _wait(qapp, lambda: not panel.is_running(), 30)
    turn = panel._turn
    assert [b.property("role") for b in turn.answers] == ["muted", "muted", None]
    titles = [r.title.text() for r in turn.rows.values()]
    assert TOOL_TITLES["compare_reference"] in titles and "compare_reference" not in titles
    assert "Done" in panel.status_text.full_text()
