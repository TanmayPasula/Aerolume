"""The experiment loop with a scripted model on a real (headless) project.

The model is :class:`test_ai_fakes.ScriptedProvider`; everything else is real:
the tools, the scripting API, the solvers (one small polar, computed once for
the module and then reused by the API, as identical runs are), the experiment
log. The tests pin the rules DESIGN.md section 9 puts on the loop: validated
inputs, all results of a turn answered together, approvals, Stop, honest
endings (refusal, provider failure, step limit) and the log on disk.
"""
from __future__ import annotations

import json
import threading
import time

import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

from aerolume.ai.agent import Agent
from aerolume.ai.config import AssistantConfig
from aerolume.ai.experiment import list_experiments, load_record
from aerolume.ai.prompts import NO_TOOLS_NUDGE, SYSTEM_PROMPT, SYSTEM_PROMPT_LOCAL
from aerolume.ai.providers.base import ProviderError, ToolCall, Turn
from aerolume.ai.workspace import Workspace

POLAR = {"speed_m_s": 20, "alpha_start": 2, "alpha_end": 6, "alpha_step": 1}
#: angles no stored polar covers yet (a covered request reuses the stored result and asks nothing)
NEW_POLAR = {"speed_m_s": 20, "alpha_start": 2, "alpha_end": 7, "alpha_step": 1}
UNRUN_POLAR = {"speed_m_s": 20, "alpha_start": 0, "alpha_end": 9, "alpha_step": 1}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project whose NACA 2412 polar (20 m/s, α 2..6°) was computed once; the agent's runs reuse it."""
    p = fakes.make_project(tmp_path_factory.mktemp("agent"))
    did = next(iter(p.raw.designs))
    p.run_polar(did, speed=20, alpha=(2, 6, 1))
    return p


@pytest.fixture()
def did(project):
    return next(iter(project.raw.designs))


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]


def _agent(provider, project, *, threshold=1e6, approve=None, max_steps=40, rec=None, **kw):
    cfg = AssistantConfig(approval_threshold_s=threshold, max_steps=max_steps)
    return Agent(provider, Workspace(project), config=cfg, approve=approve, on_event=rec, **kw)


# ---------------------------------------------------------------------------
# a whole experiment
# ---------------------------------------------------------------------------

def test_a_complete_experiment_is_planned_run_traced_and_logged(project, did):
    seen: dict = {}

    def analyse(session, _user, results):
        # both calls of the previous turn come back together, in call order
        seen["ids"] = [r.call_id for r in results]
        polar = json.loads(results[1].content)["data"]
        seen["polar"] = polar
        m = polar["metrics"]
        return tools_turn(call("c4", "record_conclusion", {
            "summary": f"Best L/D {m['ld_max']:.4g} at α {m['alpha_ld_max']:g}°.",
            "findings": [{"quantity": "Best L/D", "value": m["ld_max"], "unit": "", "uncertainty": "",
                          "simulation_id": polar["simulation_id"]}],
            "limitations": polar["validity"]}))

    def answer(session, _user, _results):
        m = seen["polar"]["metrics"]
        return final(f"The best L/D of NACA 2412 at 20 m/s is {m['ld_max']:.4g} at α = {m['alpha_ld_max']:g}° "
                     f"(Re {seen['polar']['reynolds']:.3g}, within the 1° step).")

    prov = fakes.ScriptedProvider([
        tools_turn(call("c1", "project_overview"), text="Reading the project."),
        tools_turn(call("c2", "record_plan", {"objective": "α of best L/D", "method": "polar at the real Re",
                                              "variables": ["α 2..6°"], "expected_runs": 1, "expected_seconds": 20}),
                   call("c3", "run_polar", {"design_id": did, **POLAR})),
        analyse, answer])
    rec = Recorder()
    agent = _agent(prov, project, rec=rec, context=f"design “NACA 2412” (id {did})")
    out = agent.run("Find the angle of attack with the best L/D of NACA 2412 at 20 m/s")
    assert out.status == "done" and out.untraced == []
    assert seen["ids"] == ["c2", "c3"]
    polar = seen["polar"]
    assert polar["reused_stored_result"] is True and polar["analysis"] == "polar"
    assert polar["validity"] and polar["table_columns"][0] == "alpha_deg"
    assert 50 < polar["metrics"]["ld_max"] < 150
    # the session got the honest prompt and the full tool set; the goal carried the selection
    s = prov.sessions[0]
    assert s.system == SYSTEM_PROMPT and "run_python" in {t.name for t in s.tools}
    assert "NACA 2412" in s.sent[0]["user_text"] and did in s.sent[0]["user_text"]
    # events: every tool started and ended, then "done" with the log path
    starts, ends = rec.of("tool_start"), rec.of("tool_end")
    assert [e["id"] for e in starts] == [e["id"] for e in ends] == ["c1", "c2", "c3", "c4"]
    assert all(e["status"] == "ok" for e in ends)
    assert "(stored result reused)" in ends[2]["summary"]
    assert rec.of("done")[-1]["path"] == out.path
    # the log
    record = load_record(out.path)
    assert record.status == "done" and record.plan["method"] == "polar at the real Re"
    assert [a.tool for a in record.actions] == ["project_overview", "record_plan", "run_polar", "record_conclusion"]
    assert record.actions[2].simulation_ids == [polar["simulation_id"]]
    assert any(r["label"] == "Best L/D" for r in record.results)
    assert record.findings[0]["value"] == polar["metrics"]["ld_max"]
    md = (agent.log.path / "experiment.md").read_text(encoding="utf-8")
    assert "## Plan" in md and "`run_polar`" in md and "Every number in the conclusions appears" in md
    assert any(e.path == agent.log.path for e in list_experiments(project.path))


def test_a_follow_up_continues_the_same_experiment(project):
    prov = fakes.ScriptedProvider([final("Noted."), final("Still noted.")])
    agent = _agent(prov, project)
    first = agent.run("Remember the chord is 1 m.")
    second = agent.run("And the span?")
    assert first.path == second.path and len(prov.sessions) == 1
    rec = load_record(second.path)
    assert [m["role"] for m in rec.conversation] == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def test_invalid_inputs_never_run_and_go_back_as_errors(project):
    n_sims = len(project.raw.simulations)
    prov = fakes.ScriptedProvider([
        tools_turn(call("a", "run_polar", {"speed_m_s": "fast", "colour": "red"}),
                   ToolCall("b", "run_polar", "{oops", raw="{oops"),
                   ToolCall("c", "run_polar", ["not", "an", "object"]),
                   call("d", "no_such_tool", {})),
        final("I could not run it.")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("run something")
    assert out.status == "done"
    results = prov.sessions[0].sent[1]["tool_results"]
    assert [r.call_id for r in results] == ["a", "b", "c", "d"] and all(r.is_error for r in results)
    a = json.loads(results[0].content)["error"]
    assert "design_id: missing (required)" in a and "speed_m_s: expected a number" in a and "colour" in a
    # input that is not a JSON object at all goes back in the documented INVALID_JSON wrapper
    assert json.loads(results[1].content) == {"INVALID_JSON": "{oops"}
    assert "INVALID_JSON" in json.loads(results[2].content)
    assert "Unknown tool 'no_such_tool'" in json.loads(results[3].content)["error"]
    assert {e["status"] for e in rec.of("tool_end")} == {"invalid"}
    assert len(project.raw.simulations) == n_sims                    # nothing ran
    assert [a.status for a in load_record(out.path).actions] == ["invalid"] * 4


def test_small_model_slips_are_repaired_not_rejected(project, did):
    prov = fakes.ScriptedProvider([
        tools_turn(call("a", "run_polar", {"design_id": did, "speed_m_s": "20", "alpha_start": "2",
                                           "alpha_end": 6, "alpha_step": 1, "mach": None})),
        final("ok")])
    _agent(prov, project).run("polar")
    res = prov.sessions[0].last_result("run_polar")
    assert "data" in res and res["data"]["reused_stored_result"] is True


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------

def test_long_runs_ask_and_a_decline_runs_nothing(project, did):
    asked: list = []
    answers = iter([False, True])

    def approve(approval, call_id):
        asked.append((approval.kind, call_id, approval.title))
        return next(answers)

    prov = fakes.ScriptedProvider([
        tools_turn(call("r1", "run_polar", {"design_id": did, **NEW_POLAR})),
        tools_turn(call("r2", "run_polar", {"design_id": did, **NEW_POLAR})),
        final("done")])
    rec = Recorder()
    out = _agent(prov, project, threshold=0.0, approve=approve, rec=rec).run("polar please")
    assert [a[:2] for a in asked] == [("run", "r1"), ("run", "r2")]
    assert asked[0][2].startswith("Polar of NACA 2412: about")
    r1 = json.loads(prov.sessions[0].sent[1]["tool_results"][0].content)
    assert "Declined by the user" in r1["error"]
    assert [e["status"] for e in rec.of("tool_end")] == ["declined", "ok"]
    assert [e["id"] for e in rec.of("approval")] == ["r1", "r2"]
    assert [a.status for a in load_record(out.path).actions] == ["declined", "ok"]


def test_code_and_deletion_always_ask(project):
    asked: list = []

    def approve(approval, _cid):
        asked.append(approval)
        return False

    extra = project.add_naca("0012")
    prov = fakes.ScriptedProvider([
        tools_turn(call("p", "run_python", {"code": "result = 6 * 7", "purpose": "multiply"}),
                   call("x", "delete_item", {"kind": "design", "id": extra.id})),
        final("ok")])
    _agent(prov, project, threshold=1e9, approve=approve).run("do it")
    assert [a.kind for a in asked] == ["code", "delete"]
    assert asked[0].code == "result = 6 * 7" and "NACA 0012" in asked[1].title
    assert extra.id in project.raw.designs                           # declined: still there


def test_approved_code_runs_with_the_project_bound(project):
    prov = fakes.ScriptedProvider([
        tools_turn(call("p", "run_python", {"code": "print(len(project.designs) > 0)\nresult = {'answer': 6 * 7}",
                                            "purpose": "check"})),
        final("ok")])
    _agent(prov, project, approve=lambda a, c: True).run("compute")
    res = prov.sessions[0].last_result("run_python")["data"]
    assert res["output"].strip() == "True" and res["result"] == {"answer": 42}


# ---------------------------------------------------------------------------
# Stop and other endings
# ---------------------------------------------------------------------------

def test_stop_while_waiting_for_approval(project, did):
    holder: dict = {}

    def approve(_a, _c):
        holder["agent"].stop()            # the user pressed Stop instead of answering
        return True

    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_polar", {"design_id": did, **UNRUN_POLAR})),
                                   final("never")])
    agent = _agent(prov, project, threshold=0.0, approve=approve)
    holder["agent"] = agent
    out = agent.run("polar")
    assert out.status == "stopped" and load_record(out.path).status == "stopped"
    assert len(prov.sessions[0].sent) == 1                            # the model was not asked again


def test_stop_cancels_a_running_solver(project):
    d = project.add_naca("4412")
    prov = fakes.ScriptedProvider([
        tools_turn(call("r", "run_polar", {"design_id": d.id, "speed_m_s": 25, "alpha_start": -10, "alpha_end": 16,
                                           "alpha_step": 0.5})),
        final("never")])
    rec = Recorder()
    holder: dict = {}

    def on_event(kind, data):
        rec(kind, data)
        if kind == "tool_progress" and "stopped" not in holder:
            holder["stopped"] = time.monotonic()
            holder["agent"].stop()

    agent = _agent(prov, project, rec=on_event)
    holder["agent"] = agent
    t0 = time.monotonic()
    out = agent.run("long polar")
    assert out.status == "stopped"
    assert rec.of("tool_end")[-1]["status"] == "cancelled"
    assert time.monotonic() - holder["stopped"] < 20                 # stopped, not run to the end
    sims = [s for s in project.raw.simulations.values() if s.design_id == d.id]
    assert sims and sims[0].status == "cancelled"
    assert time.monotonic() - t0 < 120


def test_provider_failure_refusal_and_step_limit(project):
    def boom(*_a):
        raise ProviderError("Ollama is not reachable at http://localhost:11434.", "connection", retryable=True)

    rec = Recorder()
    out = _agent(fakes.ScriptedProvider([boom]), project, rec=rec).run("goal")
    assert out.status == "failed" and rec.of("error")[0]["kind"] == "connection"
    assert load_record(out.path).status == "failed"

    refused = Turn(stop_reason="refusal", refusal="Claude declined this request (category: cyber).")
    out = _agent(fakes.ScriptedProvider([refused]), project).run("goal")
    assert out.status == "refused" and "declined" in out.text

    loop = [tools_turn(call(f"c{i}", "project_overview")) for i in range(10)]
    rec = Recorder()
    out = _agent(fakes.ScriptedProvider(loop), project, max_steps=4, rec=rec).run("goal")
    assert out.status == "stopped" and len(load_record(out.path).actions) == 4
    assert any("4 tool rounds" in n["text"] for n in rec.of("notice"))


def test_numbers_without_tools_get_one_reminder(project):
    prov = fakes.ScriptedProvider([final("The best L/D is 87.3 at 6.5°."),
                                   tools_turn(call("c", "project_overview")),
                                   final("I ran the overview; no polar yet.")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("best L/D?")
    assert prov.sessions[0].sent[1]["user_text"] == NO_TOOLS_NUDGE
    assert out.status == "done" and any("without running anything" in n["text"] for n in rec.of("notice"))


def test_an_answer_about_a_design_that_was_not_run_is_challenged(project, did):
    from aerolume.ai.prompts import DESIGN_NUDGE
    run = tools_turn(call("r", "run_polar", {"design_id": did, **POLAR}))
    prov = fakes.ScriptedProvider([run, final("NACA 4412 has its best L/D at 5°."),
                                   final("Correction: these numbers are for NACA 2412; NACA 4412 was not run.")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("best L/D of NACA 4412?")
    nudge = prov.sessions[0].sent[2]["user_text"]
    assert nudge == DESIGN_NUDGE.format(named="NACA 4412", used="NACA 2412")
    assert any("asking the model to correct it" in n["text"] for n in rec.of("notice"))
    # the corrected answer still names NACA 4412 (as not run): the mismatch stays on record, once
    record = load_record(out.path)
    assert out.status == "done" and record.checks == [
        "Design check: the answer names NACA 4412, but this experiment's runs used NACA 2412."]
    assert [m["role"] for m in record.conversation] == ["user", "assistant", "aerolume", "assistant"]
    from pathlib import Path
    md = (Path(out.path) / "experiment.md").read_text(encoding="utf-8")
    assert md.count("Design check") == 1 and "**Aerolume**" in md
    # a correct answer passes without a second request
    prov2 = fakes.ScriptedProvider([run, final("NACA 2412 has its best L/D at 5°.")])
    out2 = _agent(prov2, project).run("best L/D of NACA 2412?")
    assert len(prov2.sessions[0].sent) == 2 and load_record(out2.path).checks == []


class LocalScripted(fakes.ScriptedProvider):
    """A scripted provider the agent treats as a small local model (the Ollama profile)."""
    name = "ollama"


def test_a_local_model_is_given_the_overview_and_held_to_plan_and_conclusion(project, did):
    from aerolume.ai.prompts import CONCLUSION_NUDGE, FOLLOW_UP, PLAN_FIRST
    seen: dict = {}

    def runs_first(_s, user, _r):
        seen["goal"] = user
        return tools_turn(call("r1", "run_polar", {"design_id": did, **POLAR}))

    def plans(_s, _u, results):
        seen["deferred"] = json.loads(results[0].content)
        return tools_turn(call("p", "record_plan", {"objective": "best L/D", "method": "polar"}),
                          call("r2", "run_polar", {"design_id": did, **POLAR}))

    def answers(_s, _u, results):
        seen["polar"] = json.loads(results[1].content)["data"]
        m = seen["polar"]["metrics"]
        return final(f"Best L/D {m['ld_max']:.4g} at α = {m['alpha_ld_max']:g}° ± 0.5° (step).")

    def concludes(_s, user, _r):
        seen["follow_up"] = user
        m = seen["polar"]["metrics"]
        return tools_turn(call("c", "record_conclusion", {
            "summary": f"Best L/D {m['ld_max']:.4g}.", "findings": [
                {"quantity": "Best L/D", "value": m["ld_max"], "simulation_id": seen["polar"]["simulation_id"]}]}))

    prov = LocalScripted([runs_first, plans, answers, concludes, final("Recorded: best L/D as above.")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("Find the best L/D of NACA 2412 at 20 m/s")
    assert out.status == "done"
    # the overview was made by the loop and handed over in the goal message
    assert "Project overview" in seen["goal"] and did in seen["goal"]
    # the first run waited for the plan: the model got PLAN_FIRST and the user saw no row for it
    assert seen["deferred"] == {"error": PLAN_FIRST}
    assert [e["id"] for e in rec.of("tool_start")] == ["aerolume-overview", "p", "r2", "c"]
    # the answer without record_conclusion got one follow-up, then the conclusion was recorded
    assert seen["follow_up"] == FOLLOW_UP.format(items=CONCLUSION_NUDGE)
    assert any("record its conclusion" in n["text"] for n in rec.of("notice"))
    record = load_record(out.path)
    assert [a.tool for a in record.actions] == ["project_overview", "record_plan", "run_polar", "record_conclusion"]
    assert record.findings and record.plan["method"] == "polar"
    assert agent_roles(record) == ["user", "aerolume", "assistant", "aerolume", "assistant"]
    assert prov.sessions[0].system == SYSTEM_PROMPT_LOCAL


def agent_roles(record) -> list[str]:
    return [m["role"] for m in record.conversation]


def test_a_confidence_question_needs_an_uncertainty_figure(project, did):
    from aerolume.ai.agent import asks_confidence, states_uncertainty
    from aerolume.ai.prompts import CONFIDENCE_NUDGE, FOLLOW_UP
    assert asks_confidence("…and say how confident that answer is") and asks_confidence("how much should I trust it?")
    assert not asks_confidence("Find the best L/D of NACA 2412")
    assert states_uncertainty("α = 3° ± 0.5° (step)") and states_uncertainty("Cd 0.514 with a standard error of 0.005")
    assert states_uncertainty("known to half the 1° step") and not states_uncertainty("validated within a few percent")

    prov = fakes.ScriptedProvider([
        tools_turn(call("r", "run_polar", {"design_id": did, **POLAR})),
        final("The best angle is 5°. The result is confident because the solver is reliable."),
        final("Still confident."),                         # asked once only: this ends the request
    ])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("Find the angle of best L/D of NACA 2412 at 20 m/s, and how confident "
                                             "is that?")
    assert out.status == "done" and out.text == "Still confident."
    assert prov.sessions[0].sent[2]["user_text"] == FOLLOW_UP.format(items=CONFIDENCE_NUDGE)
    assert len(prov.sessions[0].sent) == 3
    # Claude-class models are not sent the local model's plan or conclusion requests
    assert not any("record its conclusion" in n["text"] for n in rec.of("notice"))
    res = prov.sessions[0].last_result("run_polar")["data"]
    assert res["optimum_resolution_deg"] == 0.5 and "cl" in res["validated_quantities"]


def test_selection_context_defers_to_the_named_design():
    from aerolume.ai.prompts import goal_message
    msg = goal_message("Best L/D of NACA 4412", selected="design “NACA 2412” (id d1)")
    assert msg.startswith("Best L/D of NACA 4412") and "only when the goal refers to the selected" in msg


def test_gpu_is_released_before_a_cfd_run(project, did, monkeypatch):
    from aerolume import api

    def busy(*_a, **_k):
        raise api.RunError("the GPU is busy")
    monkeypatch.setattr(project, "run_cfd", busy)
    prov = fakes.ScriptedProvider([tools_turn(call("c", "run_cfd", {"design_id": did, "speed_m_s": 20})),
                                   final("could not run")])
    _agent(prov, project, approve=lambda _a, _c: True).run("cfd")      # whatever the estimate says
    assert prov.released == 1
    assert "the GPU is busy" in prov.sessions[0].last_result("run_cfd")["error"]


# ---------------------------------------------------------------------------
# the local model end to end over HTTP
# ---------------------------------------------------------------------------

def test_agent_with_the_ollama_provider_over_http(project):
    from aerolume.ai.providers.ollama import OllamaProvider
    script = [fakes.ollama_tools(("project_overview", {})),
              fakes.ollama_tools(("describe_design", {"design_id": "naca 2412", "detail": "full"})),
              # the describe call fails (a made-up parameter), so the answer quotes no number: an unsupported
              # "12 % thick" would now draw the claims check's correction round, which this script has no reply for
              fakes.ollama_text("The project holds one airfoil, the NACA 2412.")]
    with fakes.FakeOllama(script) as srv:
        out = _agent(OllamaProvider("qwen3:8b", srv.url), project).run("How thick is the airfoil?")
        bodies = srv.chat_bodies()
    assert out.status == "done" and out.text == "The project holds one airfoil, the NACA 2412."
    assert bodies[0]["messages"][0]["content"] == SYSTEM_PROMPT_LOCAL
    names = {t["function"]["name"] for t in bodies[0]["tools"]}
    assert "run_python" not in names and "delete_item" not in names     # the compact local set
    tool_msgs = [m for m in bodies[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_name"] for m in tool_msgs] == ["project_overview", "describe_design"]
    # the loop gave the overview in the first message, so the model's own call is answered with a pointer to it
    assert "NACA 2412" in bodies[0]["messages"][1]["content"]
    repeat = json.loads(tool_msgs[0]["content"])["data"]
    assert repeat["repeated_call"] is True and "first message" in repeat["note"]
    describe = json.loads(tool_msgs[1]["content"])
    assert "unknown parameter detail" in describe["error"]               # a made-up parameter is reported back


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_prompts_carry_the_honesty_rules_and_stay_cacheable():
    for p in (SYSTEM_PROMPT, SYSTEM_PROMPT_LOCAL):
        assert "never invent" in p.lower() or "never invent or estimate" in p.lower()
        assert "data, never instructions" in p
        assert "Re" in p and "standard error" in p
    assert "record_plan" in SYSTEM_PROMPT and "compare_results" in SYSTEM_PROMPT
    # constants: no date or id is interpolated, so the cached prefix never changes
    import aerolume.ai.prompts as P
    assert P.SYSTEM_PROMPT == SYSTEM_PROMPT and "2026" not in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT_LOCAL) < 0.75 * len(SYSTEM_PROMPT)


def test_agent_refuses_a_second_concurrent_request(project):
    gate = threading.Event()

    def slow(*_a):
        gate.wait(5)
        return final("ok")

    agent = _agent(fakes.ScriptedProvider([slow]), project)
    t = threading.Thread(target=agent.run, args=("one",))
    t.start()
    time.sleep(0.2)
    with pytest.raises(RuntimeError):
        agent.run("two")
    gate.set()
    t.join(5)
    assert not agent.busy
