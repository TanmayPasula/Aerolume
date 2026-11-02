"""The assistant's honesty checks, each pinned to the live failure that motivated it.

A review drove qwen3:8b through the real app and found answers that every
earlier check passed:

* "The simulation ... has been deleted from the project": nothing was deleted
  (no delete tool is offered to a local model), yet the panel said "Done";
* "best L/D at 15 m/s is 26.47, matching the stored 25 m/s polar's best L/D of
  26.47": the 25 m/s run's figure is 29.52, and record_conclusion accepted a
  finding citing that run with 26.47 and an uncertainty of "+/- 0.5" (the
  angle resolution, in degrees, applied to an L/D);
* a goal comparing 15 and 25 m/s answered without ever running 15 m/s, after
  reading the same results twice into an 8k context;
* "Cd is 0.51377 ± 0.00455 ... I am confident within ±0.00455": a standard
  error given as the accuracy of a run whose drag is not validated;
* an approval card that did not say what would run, and an estimate failure
  that started the run without asking at all;
* "Controls (held fixed): speed_m_s" for an experiment that varied the speed.

The model is :class:`test_ai_fakes.ScriptedProvider` (as a local "ollama" model
where the local profile matters); tools, API, solvers and logs are real.
"""
from __future__ import annotations

import json

import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

from aerolume.ai.agent import Agent, claimed_actions, treats_stderr_as_accuracy
from aerolume.ai.config import AssistantConfig
from aerolume.ai.experiment import (
    SimFacts,
    attribution_problems,
    elide,
    goal_targets,
    load_record,
    plan_controls_label,
    render_markdown,
)
from aerolume.ai.prompts import (
    CLAIM_HINT_DELETE,
    CLAIM_NUDGE,
    CLAIMS_NUDGE,
    COVERAGE_NUDGE,
    FOLLOW_UP,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_LOCAL,
    VALIDITY,
)
from aerolume.ai.providers.base import ToolCall
from aerolume.ai.tools import (
    InputError,
    Toolbox,
    ToolContext,
    compact_summary,
    round_to_stderr,
    run_detail,
    stderr_meaning,
    summarize_result,
)
from aerolume.ai.workspace import Workspace


class LocalScripted(fakes.ScriptedProvider):
    """A scripted provider the agent treats as a small local model (the Ollama profile)."""
    name = "ollama"


@pytest.fixture(scope="module")
def wing_project(tmp_path_factory):
    """A tapered wing with viscous polars at 15 and 25 m/s (about 10 s, once for the module)."""
    p = fakes.make_project(tmp_path_factory.mktemp("checks"))
    w = p.add_wing(2.0, 0.3, 0.18, airfoil="2412", name="Tapered wing")
    r15 = p.run_wing(w.id, speed=15, alpha=(-2, 8, 2))
    r25 = p.run_wing(w.id, speed=25, alpha=(-2, 8, 2))
    return p, w.id, r15.sim_id, r25.sim_id


def _metrics(project, sid: str) -> dict:
    r = project.result(sid)
    return summarize_result(r.raw, r.simulation, r.design.raw)["metrics"]


def _agent(provider, project, *, threshold=1e6, approve=None, rec=None):
    cfg = AssistantConfig(approval_threshold_s=threshold, max_steps=40)
    return Agent(provider, Workspace(project), config=cfg, approve=approve, on_event=rec)


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]


# ---------------------------------------------------------------------------
# claimed actions
# ---------------------------------------------------------------------------

def test_claimed_actions_are_found_and_negations_are_not():
    claim = 'The simulation "Polar · Re 3.0×10⁵" has been deleted from the project.'
    assert claimed_actions(claim) == [("delete", claim)]          # the point inside 3.0 is not a sentence end
    assert claimed_actions("I ran a wing analysis at 15 m/s.")[0][0] == "run"
    assert claimed_actions("I've created a plot of the polar.")[0][0] == "create"
    for honest in ("Nothing was deleted.", "I cannot delete simulations; use Project tree > Delete.",
                   "I have not run anything yet.", "If I deleted it, it would be gone.",
                   "The stored 25 m/s polar has been run before.", "The simulation was not deleted."):
        assert claimed_actions(honest) == [], honest


def test_a_false_deletion_claim_is_corrected_once_then_withheld(tmp_path):
    project = fakes.make_project(tmp_path)
    sid = project.run_polar(next(iter(project.raw.designs)), speed=20, alpha=(2, 4, 1)).sim_id
    name = project.raw.simulations[sid].name
    claim = f"The simulation “{name}” has been deleted from the project."
    prov = LocalScripted([final(claim), final(claim + " It is gone.")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run(f"Delete the simulation “{name}”")
    # one correction request, with the local model's way to delete
    nudge = prov.sessions[0].sent[1]["user_text"]
    assert nudge == CLAIM_NUDGE.format(claims=f"“{claim}”", hint=CLAIM_HINT_DELETE)
    # repeated: the answer is replaced, the status is not "done", and the original stays in the log
    assert out.status == "unverified"
    assert "withheld" in out.text and "Nothing was deleted" in out.text and "Project tree" in out.text
    done = rec.of("done")[-1]
    assert done["status"] == "unverified" and done["withheld"][-1].endswith("It is gone.")
    assert any(e["text"].startswith("Answer not verified") for e in rec.of("status"))
    record = load_record(out.path)
    assert record.status == "unverified" and record.conclusions == out.text
    assert any(c.startswith("Action check:") for c in record.checks)
    assert [m["role"] for m in record.conversation] == ["user", "assistant", "aerolume", "assistant", "aerolume"]
    assert sid in project.raw.simulations                               # and indeed nothing was deleted
    assert "| Status | Finished, answer not verified" in render_markdown(record)


def test_a_corrected_claim_ends_normally(tmp_path):
    project = fakes.make_project(tmp_path)
    prov = LocalScripted([final("The design has been deleted."),
                          final("I cannot delete designs. Select it in the Project tree and press Delete.")])
    out = _agent(prov, project).run("Delete NACA 2412")
    assert out.status == "done" and "cannot delete" in out.text


def test_a_configuration_is_not_an_action_but_a_past_deletion_is():
    for configuration in ("Drag falls by 8 % when the spoiler is removed.",
                          "With the rear wing removed the lift is negative.",
                          "Where the wheels are removed, the wake is shorter."):
        assert claimed_actions(configuration) == [], configuration
    assert claimed_actions("After the old run was deleted, the polar was repeated.")[0][0] == "delete"
    assert claimed_actions("When I deleted the simulation, its result went too.")[0][0] == "delete"


def test_record_conclusion_refuses_a_claimed_action(tmp_path):
    """Live (qwen3:8b, the prompt's delete rule removed): the delete goal's conclusion said it was deleted.

    That call was refused only because a finding's value was the word "done";
    here the value is a number, so the action check alone must refuse it.
    """
    from aerolume.ai.experiment import ExperimentLog
    project = fakes.make_project(tmp_path)
    sid = project.run_polar(next(iter(project.raw.designs)), speed=20, alpha=(2, 4, 1)).sim_id
    log = ExperimentLog(tmp_path, "Delete the simulation")
    ctx = ToolContext(Workspace(project), log)
    tb = Toolbox()
    tb.run("project_overview", {}, ctx)
    live = {"summary": "The simulation 'Wing polar · 25 m/s' has been deleted and is no longer available.",
            "limitations": ["Simulation has been deleted."],
            "findings": [{"quantity": "Simulations left", "simulation_id": sid, "value": 1, "unit": ""}]}
    out = tb.run("record_conclusion", live, ctx)
    assert out.is_error and "claims an action that did not happen" in out.content["error"]
    assert "no tool call in this experiment deleted anything" in out.content["error"]
    rec = load_record(log.path)
    assert rec.conclusions == "" and rec.findings == []                     # nothing reached the log
    # a claimed run is refused in an experiment that ran nothing, and recorded after a real run
    assert tb.run("record_conclusion", {"summary": "I ran the polar at 20 m/s.", "findings": []}, ctx).is_error
    ran = ExperimentLog(tmp_path, "run it")
    ctx2 = ToolContext(Workspace(project), ran)
    tb.run("run_polar", {"design_id": next(iter(project.raw.designs)), "speed_m_s": 20, "alpha_start": 2,
                         "alpha_end": 4, "alpha_step": 1}, ctx2)
    ok = tb.run("record_conclusion", {"summary": "I ran the polar at 20 m/s.", "findings": []}, ctx2)
    assert not ok.is_error, ok.content


def test_an_empty_reply_is_asked_about_once_then_ends_failed(tmp_path):
    from aerolume.ai.prompts import EMPTY_NUDGE, NO_ANSWER
    project = fakes.make_project(tmp_path)
    prov = LocalScripted([final(""), final("I cannot delete simulations; use Project tree > Delete.")])
    out = _agent(prov, project).run("Delete NACA 2412")
    assert prov.sessions[0].sent[1]["user_text"] == EMPTY_NUDGE
    assert out.status == "done" and "cannot delete" in out.text
    prov2 = LocalScripted([final(""), final("  ")])
    rec = Recorder()
    out2 = _agent(prov2, project, rec=rec).run("Delete NACA 2412")
    assert out2.status == "failed" and out2.text == NO_ANSWER                  # never "Done" with nothing to read
    assert rec.of("done")[-1]["status"] == "failed"
    assert any(n["text"] == NO_ANSWER for n in rec.of("notice"))
    record = load_record(out2.path)
    assert record.status == "failed"
    assert record.conversation[-1]["role"] == "aerolume"                    # not put in the model's mouth


def test_the_local_prompt_says_what_the_model_cannot_do():
    assert "You cannot delete designs or simulations or run Python code" in SYSTEM_PROMPT_LOCAL
    assert "Project tree > Delete" in SYSTEM_PROMPT_LOCAL
    for p in (SYSTEM_PROMPT, SYSTEM_PROMPT_LOCAL):
        assert 'never write "confident within ±" a standard error' in p
    assert "stderr_meaning" in VALIDITY
    assert len(SYSTEM_PROMPT_LOCAL) < 0.75 * len(SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# findings must hold for the simulation they cite
# ---------------------------------------------------------------------------

def test_record_conclusion_checks_each_finding_against_its_simulation(wing_project, tmp_path):
    """The live case: the 15 m/s figure cited as the 25 m/s run's, with the angle step as its error bar."""
    from aerolume.ai.experiment import ExperimentLog
    project, _wid, s15, s25 = wing_project
    m15, m25 = _metrics(project, s15), _metrics(project, s25)
    assert abs(m15["ld_max"] - m25["ld_max"]) > 0.1                       # the two runs differ
    ctx = ToolContext(Workspace(project), ExperimentLog(tmp_path, "compare 15 and 25 m/s"))
    tb = Toolbox()
    ld15 = float(f"{m15['ld_max']:.4g}")
    bad = tb.run("record_conclusion", {"summary": "Same best L/D.", "findings": [
        {"quantity": "best L/D", "simulation_id": s15, "value": ld15, "uncertainty": "+/- 0.5"},
        {"quantity": "best L/D (25 m/s)", "simulation_id": s25, "value": ld15, "uncertainty": ""}]}, ctx)
    assert bad.is_error
    err = bad.content["error"]
    assert err.startswith("Conclusion not recorded.")
    # finding 2: names the cited run's actual values
    assert f"Finding 2 (“best L/D (25 m/s)” = {ld15!r})" in err
    assert f"ld_max {m25['ld_max']:.4g} at α {m25['alpha_ld_max']:g}°" in err
    # finding 1: a deterministic result has no ± error bar; ±0.5 is the angle resolution
    assert "Finding 1" in err and "no statistical error bar" in err
    assert load_record(ctx.log.path).findings == []                       # nothing was recorded
    # a degree-unit uncertainty on a quantity that is not an angle is refused too
    deg = tb.run("record_conclusion", {"summary": "s", "findings": [
        {"quantity": "best L/D", "simulation_id": s15, "value": ld15, "uncertainty": "± 1°"}]}, ctx)
    assert deg.is_error and "is in degrees" in deg.content["error"]
    # correct findings (an angle may carry the half-step resolution) are recorded
    ok = tb.run("record_conclusion", {"summary": "Best L/D at both speeds.", "findings": [
        {"quantity": "best L/D", "simulation_id": s15, "value": ld15, "uncertainty": ""},
        {"quantity": "best L/D", "simulation_id": s25, "value": float(f"{m25['ld_max']:.4g}")},
        {"quantity": "α at best L/D", "simulation_id": s25, "value": m25["alpha_ld_max"], "unit": "deg",
         "uncertainty": "± 1° (half the step)"},
        {"quantity": "change in best L/D", "simulation_id": s25, "value": 12.3, "unit": "%"}]}, ctx)
    assert not ok.is_error and len(load_record(ctx.log.path).findings) == 4


def test_attribution_check_finds_a_number_given_for_the_wrong_run():
    s15 = SimFacts("a", "“Wing polar · 15 m/s”", "Tapered wing", 15.0, [26.47, 3.0, 0.98], [26.47, 3.0])
    s25 = SimFacts("b", "“Wing polar · 25 m/s”", "Tapered wing", 25.0, [29.52, 3.0, 0.99], [29.52, 3.0])
    live = ("The best L/D for the Tapered wing at 15 m/s is **26.47**, matching the stored 25 m/s polar's best "
            "L/D of **26.47**. Both results are validated for CL_alpha.")
    assert attribution_problems(live, [s15, s25]) == [
        "26.47 is given for “Wing polar · 25 m/s”, but it is a value of “Wing polar · 15 m/s”"]
    assert attribution_problems("At 15 m/s the best L/D is 26.47; at 25 m/s it is 29.52.", [s15, s25]) == []
    # prose numbers that belong to no run are the plain number check's business, not this one's
    assert attribution_problems("At 25 m/s the lift slope is validated within 2.5 %.", [s15, s25]) == []
    assert attribution_problems(live, [s15]) == []                       # one run: nothing to confuse


def test_the_final_answer_is_checked_for_attribution(wing_project):
    project, _wid, s15, s25 = wing_project
    m15 = _metrics(project, s15)
    wrong = (f"At 15 m/s the best L/D is {m15['ld_max']:.4g}; the stored 25 m/s polar's best L/D is "
             f"{m15['ld_max']:.4g} too.")
    prov = fakes.ScriptedProvider([tools_turn(call("g", "get_results", {"simulation_ids": [s15, s25]})),
                                   final(wrong), final(wrong)])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("Compare the best L/D of the Tapered wing at 15 and 25 m/s")
    nudge = prov.sessions[0].sent[2]["user_text"]
    m25 = _metrics(project, s25)
    # the claims check sends it back once, with the value the number should have been
    assert nudge.startswith(CLAIMS_NUDGE.split("{")[0]) and "Wing polar · 25 m/s" in nudge
    assert f"ld_max of “{project.raw.simulations[s25].name}” is {m25['ld_max']:.4g}" in nudge
    record = load_record(out.path)
    # the repeated misattribution is not verified: flagged in the log, no green check
    assert out.status == "unverified" and any(c.startswith("Attribution check:") for c in record.checks)
    assert any(c.startswith("Claims check:") for c in record.checks)
    md = render_markdown(record)
    assert "Attribution check" in md and "### Verification" in md and "| Contradicted |" in md
    assert rec.of("done")[-1]["claims"]["counts"]["contradicted"] >= 1


# ---------------------------------------------------------------------------
# covering the goal, compact and repeated reads
# ---------------------------------------------------------------------------

def test_goal_targets_pair_conditions_with_designs():
    names = ["Tapered wing", "NACA 2412", "sedan mm"]
    t = goal_targets('How much does the best L/D of the wing "Tapered wing" change between 15 and 25 m/s?', names)
    assert [(x.design, x.speed) for x in t] == [("Tapered wing", 15.0), ("Tapered wing", 25.0)]
    t = goal_targets("Compare NACA 2412 at 20 m/s with the sedan mm at 30 m/s", names)
    assert [(x.design, x.speed) for x in t] == [("NACA 2412", 20.0), ("sedan mm", 30.0)]
    assert goal_targets('Delete the simulation "Polar · Re 3.0×10⁵"', names) == []


def test_a_goal_condition_that_was_never_run_is_asked_for(wing_project):
    project, _wid, s15, s25 = wing_project
    prov = LocalScripted([
        tools_turn(call("g", "get_results", {"simulation_ids": [s25]})),
        tools_turn(call("g2", "get_results", {"simulation_ids": [s25]})),     # the same call again
        final("The best L/D at 25 m/s is as read; 15 m/s was not needed."),
        final("I did not run 15 m/s.")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run(
        'How much does the best L/D of the wing "Tapered wing" change between 15 and 25 m/s?')
    sent = prov.sessions[0].sent
    # compact payload for a local model: metrics without the table, short validity codes, plain metric names
    first = json.loads(sent[1]["tool_results"][0].content)["data"]["results"][0]
    assert "table" not in first and "confidence_note" not in first
    assert first["validity"][0].startswith("CD, L/D and stall not validated")
    assert {"span_efficiency_at_best_LD", "oswald_e_parabolic_fit"} <= set(first["metrics"])
    assert "e_cruise" not in first["metrics"] and "oswald" not in first["metrics"]
    # the identical second read is answered with a pointer, not the payload
    second = json.loads(sent[2]["tool_results"][0].content)["data"]
    assert second["repeated_call"] is True and "action" in second["note"]
    # the goal's 15 m/s was neither run nor read: one request to cover it (the project holds a 15 m/s run)
    nudge = sent[3]["user_text"]
    name15 = project.raw.simulations[s15].name
    assert nudge == COVERAGE_NUDGE.format(targets="Tapered wing at 15 m/s", it="it",
                                          calls=f"Read simulation {s15} (“{name15}”) with get_results")
    record = load_record(out.path)
    assert out.status == "done" and record.actions[-1].summary.startswith("Same call as action")
    assert sum(1 for x in sent if (x["user_text"] or "").startswith("Aerolume note: the goal asks")) == 1
    # without a stored match the request is to run it
    from aerolume.ai.experiment import GoalTarget
    agent = _agent(LocalScripted([]), project)
    assert agent._call_for(GoalTarget("Tapered wing", speed=18.0)) == "Call run_wing on Tapered wing at 18 m/s"


def test_compact_summary_is_much_smaller_and_keeps_the_facts(wing_project):
    project, _wid, s15, _s25 = wing_project
    r = project.result(s15)
    full = summarize_result(r.raw, r.simulation, r.design.raw)
    small = compact_summary(full)
    assert len(json.dumps(small)) < 0.75 * len(json.dumps(full))
    for key in ("metrics", "validated_quantities", "not_validated", "optimum_resolution_deg", "conditions"):
        assert small[key] == full[key]


# ---------------------------------------------------------------------------
# CFD standard errors
# ---------------------------------------------------------------------------

def test_cfd_values_are_rounded_to_their_standard_error():
    # the Particle Data Group rule of the Results tables (455 -> one figure, 950-999 -> 1000 at two)
    assert round_to_stderr(0.51377, 0.004553) == (0.514, 0.005)
    assert round_to_stderr(3.34079, 0.00095) == (3.3408, 0.001)
    assert round_to_stderr(0.0965, 0.0016) == (0.0965, 0.0016)
    assert round_to_stderr(640.23, 5.71) == (640.0, 6.0)
    assert round_to_stderr(221.4, 13.2) == (221.0, 13.0)
    assert round_to_stderr(0.513771234, float("nan")) == (0.51377, None)


def test_stderr_meaning_says_what_it_leaves_out():
    class R:
        re_sim = 9.24e6

    class D:
        dimension = 3
        kind = "vehicle"
    text = stderr_meaning(R(), D())
    assert text.startswith("statistical averaging uncertainty of this run only")
    assert "no wall model" in text and "Re 9.2e6" in text and "not included" in text


def test_a_standard_error_given_as_accuracy_gets_a_follow_up(tmp_path):
    from aerolume.ai.prompts import STDERR_NUDGE
    assert treats_stderr_as_accuracy("Cd is 0.51377 ± 0.00455. I am confident within ±0.00455 due to the "
                                     "standard error.")
    assert not treats_stderr_as_accuracy("Cd 0.5138 ± 0.0045 (standard error; modelling error not included).")
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    prov = fakes.ScriptedProvider([
        tools_turn(call("r", "run_polar", {"design_id": did, "speed_m_s": 20, "alpha_start": 2, "alpha_end": 4,
                                           "alpha_step": 1})),
        final("The result is accurate to ±0.5."), final("Corrected.")])
    out = _agent(prov, project).run("Polar of NACA 2412 at 20 m/s")
    item = prov.sessions[0].sent[2]["user_text"]
    assert out.status == "done"
    assert item == FOLLOW_UP.format(items=STDERR_NUDGE.format(not_validated="Cl_max, values near and past stall"))


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------

def test_the_approval_card_says_what_will_run(wing_project):
    project, wid, _s15, _s25 = wing_project
    ctx = ToolContext(Workspace(project))
    tb = Toolbox()
    args = {"design_id": wid, "speed_m_s": 18, "alpha_start": -4, "alpha_end": 14, "alpha_step": 1}
    a = tb.approval("run_wing", args, ctx, 0.0)
    assert a is not None and a.title.startswith("Wing analysis of Tapered wing: about")
    assert a.what == "18 m/s · α −4° to 14°, step 1° (19 angles) · VLM, viscous strips"
    did = next(d.id for d in project.raw.designs.values() if d.name == "NACA 2412")
    cfd = tb.approval("run_cfd", {"design_id": did, "speed_m_s": 20, "alpha_deg": 4}, ctx, 0.0)
    assert cfd is not None and cfd.what.startswith("20 m/s · α 4° · draft quality · ")
    assert " cells · Re " in cfd.what
    sweep = run_detail("run_sweep", {"design_id": wid, "analysis": "wing", "variable": "speed",
                                     "values": [10, 15, 20]})
    assert sweep.startswith("speed = 10, 15, 20 m/s · α −4° to 12°")


def test_an_estimate_failure_asks_instead_of_running(wing_project, monkeypatch):
    project, wid, _s15, _s25 = wing_project
    ctx = ToolContext(Workspace(project))
    tb = Toolbox()

    def boom(_ctx, _args):
        raise RuntimeError("planner exploded")
    monkeypatch.setattr(tb.tools["run_wing"], "estimate", boom)
    a = tb.approval("run_wing", {"design_id": wid, "speed_m_s": 18}, ctx, 1e9)
    assert a is not None and a.kind == "run" and a.title == "Wing analysis of Tapered wing: cost unknown"
    assert "could not estimate this run (planner exploded); approve to run it anyway" in a.detail
    assert a.estimate_s is None and a.what.startswith("18 m/s")
    # wrong input is not put to the user: the call is refused (InputError) and nothing runs
    monkeypatch.undo()
    with pytest.raises(InputError):
        tb.approval("run_wing", {"design_id": "no such wing", "speed_m_s": 18}, ctx, 0.0)


def test_an_unapproved_unknown_cost_run_does_not_start(wing_project, monkeypatch):
    project, wid, _s15, _s25 = wing_project
    n = len(project.raw.simulations)
    tb = Toolbox()

    def boom(_ctx, _args):
        raise RuntimeError("no estimate")
    monkeypatch.setattr(tb.tools["run_wing"], "estimate", boom)
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_wing", {"design_id": wid, "speed_m_s": 33})),
                                   final("Not run.")])
    agent = Agent(prov, Workspace(project), config=AssistantConfig(approval_threshold_s=1e9), toolbox=tb)
    agent.run("wing at 33 m/s")                                             # nobody approves
    assert "Declined by the user" in prov.sessions[0].last_result("run_wing")["error"]
    assert len(project.raw.simulations) == n


# ---------------------------------------------------------------------------
# the plan and the log
# ---------------------------------------------------------------------------

def test_a_varied_quantity_is_not_accepted_as_a_control(tmp_path):
    from aerolume.ai.experiment import ExperimentLog
    project = fakes.make_project(tmp_path)
    goal = "Compare the best L/D of the Tapered wing at 15 m/s with the stored 25 m/s wing polar."
    ctx = ToolContext(Workspace(project), ExperimentLog(tmp_path, goal))
    tb = Toolbox()

    def plan(**kw):
        a, problems = tb.prepare(ToolCall("p", "record_plan", {"objective": "o", "method": "wing", **kw}))
        assert problems == []
        return tb.run("record_plan", a, ctx)
    live = plan(controls=["speed_m_s", "alpha_deg"])
    assert live.is_error and "compares 15 and 25 m/s, so speed is varied" in live.content["error"]
    both = plan(variables=["alpha_deg -4..14"], controls=["alpha range"])
    assert both.is_error and "also among the variables" in both.content["error"]
    ok = plan(variables=["speed 15, 25 m/s"], controls=["alpha -4..14 deg step 1", "viscous strips"])
    assert not ok.is_error
    md = (ctx.log.path / "experiment.md").read_text(encoding="utf-8")
    assert "**Controls (held fixed):** alpha -4..14 deg step 1; viscous strips" in md
    assert plan_controls_label({"controls": ["speed_m_s"]}) == "Settings"


def test_the_log_title_is_elided_and_the_goal_kept_in_full():
    goal = ('Run a wing analysis of the wing "Tapered wing" at 15 m/s from -4 to 14 degrees in 1 degree steps, '
            "and compare its best L/D with the stored 25 m/s wing polar.")
    short = elide(goal)
    assert len(short) <= 80 and short.endswith("…") and not short[:-1].endswith(" ")
    from aerolume.ai.experiment import ExperimentRecord
    md = render_markdown(ExperimentRecord(id="x", goal=goal, created="2026-10-11T21:19:41+02:00",
                                          updated="2026-10-11T21:19:41+02:00", status="done"))
    assert md.startswith(f"# Experiment: {short}\n") and f"## Goal\n\n{goal}\n" in md
