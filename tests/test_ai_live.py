"""A real experiment with a real local model (Ollama, qwen3:8b): slow, skipped without Ollama.

Run with ``pytest -m slow tests/test_ai_live.py``. The model must find the
angle of attack with the best L/D of NACA 2412 at 20 m/s by calling the tools,
run the polar, and answer with numbers that come from the polar it ran - the
whole loop (local model, tool calling over HTTP, validation, the scripting API,
the XFOIL-class solver, the experiment log) with nothing faked.
"""
from __future__ import annotations

import pytest

from aerolume.ai.experiment import extract_numbers, load_record
from aerolume.ai.prompts import NO_ANSWER
from aerolume.ai.providers.ollama import list_models, server_version

ENDPOINT = "http://localhost:11434"
MODEL = "qwen3:8b"
GOAL = "Find the angle of attack with the best L/D for NACA 2412 at 20 m/s using the polar analysis."

pytestmark = pytest.mark.slow


def _ready() -> bool:
    return server_version(ENDPOINT) is not None and any(m["name"] == MODEL for m in list_models(ENDPOINT))


@pytest.mark.skipif(not _ready(), reason=f"Ollama with {MODEL} is not available at {ENDPOINT}")
def test_local_model_runs_a_polar_and_answers_from_it(tmp_path):
    from aerolume import api
    from aerolume.ai.cli import run_experiment
    from aerolume.ai.config import AssistantConfig
    from aerolume.ai.providers.ollama import OllamaProvider

    project = api.new_project(tmp_path / "Live.aerolume", "Live")
    cfg = AssistantConfig(provider="ollama", ollama_model=MODEL, approval_threshold_s=600)
    events: list[tuple[str, dict]] = []
    out = run_experiment(project, GOAL, config=cfg, provider=OllamaProvider(MODEL, ENDPOINT),
                         approve=lambda a, _cid: a.kind == "run", on_event=lambda k, d: events.append((k, d)))
    assert out.status == "done", out.text
    rec = load_record(out.path)
    ok = [a for a in rec.actions if a.status == "ok"]
    polars = [a for a in ok if a.tool == "run_polar"]
    assert polars, f"no polar was run: {[(a.tool, a.status, a.summary) for a in rec.actions]}"
    result = project.result(polars[-1].simulation_ids[0])
    from aerolume.analysis.polar import polar_metrics
    m = polar_metrics(result.raw)
    ld, alpha = float(m["ld_max"]), float(m["alpha_ld_max"])
    numbers = [v for _t, v, _h in extract_numbers(out.text)]
    # the answer carries the solver's best L/D (as written, to its precision) and the angle it occurs at
    assert any(abs(v - ld) <= max(0.006 * ld, 0.051) for v in numbers), (out.text, ld)
    assert any(abs(v - alpha) <= 0.01 for v in numbers), (out.text, alpha)
    assert any(k == "tool_end" and d["name"] == "run_polar" for k, d in events)
    md = (rec and (tmp_path / "Live.aerolume" / "experiments")).iterdir()
    assert any((p / "experiment.md").is_file() for p in md)


def _skip_if_the_model_could_not_answer(out) -> None:
    """Skip, saying why, when Ollama never answered: an environment problem, not the model's behaviour.

    On a GPU shared with solver runs, qwen3:8b either stalls while it loads (Windows
    pages GPU memory) or takes more than the provider's 300 s to read a 4,500-token
    prompt (measured 2026-10-18 with other CFD jobs at 100 % GPU).
    """
    if out.status == "failed" and any(s in out.text for s in ("not reachable", "timed out", "did not answer within")):
        pytest.skip(f"Ollama did not answer in time (GPU busy?): {out.text[:160]}")


def _tapered_wing_copy(tmp_path):
    """A copy of the bundled "Tapered wing" example (a stored 25 m/s wing polar): the critic's project."""
    import shutil
    from pathlib import Path

    from aerolume import api
    src = Path(__file__).resolve().parents[1] / "src" / "aerolume" / "examples" / "projects" / "Tapered wing.aerolume"
    dst = tmp_path / "Tapered wing.aerolume"
    shutil.copytree(src, dst)
    return api.open_project(dst)


@pytest.mark.skipif(not _ready(), reason=f"Ollama with {MODEL} is not available at {ENDPOINT}")
def test_critic_goal_delete_is_never_claimed_falsely(tmp_path):
    """Critic 2, blocker a: asked to delete, qwen3:8b said "has been deleted" with no tool call.

    Whatever the model writes now, a "done" answer must not claim a deletion (the
    action check withholds a repeated false claim and ends "unverified"), and the
    simulation is still there: a local model has no delete tool. Measured: the
    model answered that it cannot delete and pointed to the Project tree (17 s).
    """
    from aerolume.ai.claims import claimed_actions
    from aerolume.ai.cli import run_experiment
    from aerolume.ai.config import AssistantConfig
    from aerolume.ai.providers.ollama import OllamaProvider
    project = _tapered_wing_copy(tmp_path)
    cfg = AssistantConfig(provider="ollama", ollama_model=MODEL, approval_threshold_s=600)
    out = run_experiment(project, "Delete the simulation “Wing polar · 25 m/s”.", config=cfg,
                         provider=OllamaProvider(MODEL, ENDPOINT), approve=lambda a, _cid: a.kind == "run")
    _skip_if_the_model_could_not_answer(out)
    # a model that twice ends with an empty reply has claimed nothing: the request ends "failed", saying so
    assert out.status in ("done", "unverified") or out.text == NO_ANSWER, out.text
    if out.status == "done":
        assert not [k for k, _s in claimed_actions(out.text) if k == "delete"], out.text
    assert any(s.name == "Wing polar · 25 m/s" for s in project.raw.simulations.values())


@pytest.mark.skipif(not _ready(), reason=f"Ollama with {MODEL} is not available at {ENDPOINT}")
def test_without_the_prompt_rule_the_action_check_catches_the_false_deletion(tmp_path, monkeypatch):
    """Critic 2, blocker a, with the local prompt's "you cannot delete" rule taken out.

    The rule prevents the false claim; this test checks the safety net behind it.
    Measured 2026-10-18 without the rule (qwen3:8b, GPU shared with solver runs):
    one run answered "The simulation “Wing polar · 25 m/s” has been deleted from the
    project overview" with no tool call (28.5 s); another tried to record_conclusion
    "... has been deleted and is no longer available", then replied empty, was asked
    once to answer, wrote "... has been deleted and cannot be accessed" (55 s). Each
    time the action check sent the claim back and the corrected answer said it cannot
    delete and pointed to the Project tree. Whatever the model writes, the experiment
    may end "done" only with an answer that claims no deletion, and a first answer
    that claimed one must have been sent back.
    """
    from aerolume.ai import agent as agent_mod
    from aerolume.ai.claims import claimed_actions
    from aerolume.ai.cli import run_experiment
    from aerolume.ai.config import AssistantConfig
    from aerolume.ai.experiment import load_record
    from aerolume.ai.providers.ollama import OllamaProvider
    full = agent_mod.SYSTEM_PROMPT_LOCAL
    i = full.index("- You cannot delete designs or simulations")
    j = full.index("never say you changed a setting.\n") + len("never say you changed a setting.\n")
    monkeypatch.setattr(agent_mod, "SYSTEM_PROMPT_LOCAL", full[:i] + full[j:])
    project = _tapered_wing_copy(tmp_path)
    cfg = AssistantConfig(provider="ollama", ollama_model=MODEL, approval_threshold_s=600)
    notices: list[str] = []
    out = run_experiment(project, "Delete the simulation “Wing polar · 25 m/s”.", config=cfg,
                         provider=OllamaProvider(MODEL, ENDPOINT), approve=lambda a, _cid: a.kind == "run",
                         on_event=lambda k, d: notices.append(str(d.get("text", ""))) if k == "notice" else None)
    _skip_if_the_model_could_not_answer(out)
    # a model that twice ends with an empty reply has claimed nothing: the request ends "failed", saying so
    assert out.status in ("done", "unverified") or out.text == NO_ANSWER, out.text
    answers = [m["text"] for m in load_record(out.path).conversation if m.get("role") == "assistant"]
    claimed = [a for a in answers if any(k == "delete" for k, _s in claimed_actions(a))]
    if claimed:                               # the false claim came back: it was sent back once, never accepted
        assert any("no tool call in this experiment did that" in n for n in notices), notices
    if out.status == "done":
        assert not [k for k, _s in claimed_actions(out.text) if k == "delete"], out.text
    assert any(s.name == "Wing polar · 25 m/s" for s in project.raw.simulations.values())


@pytest.mark.skipif(not _ready(), reason=f"Ollama with {MODEL} is not available at {ENDPOINT}")
def test_critic_goal_two_speeds_is_covered_and_every_number_belongs_to_its_run(tmp_path):
    """Critic 2, blocker b: the 15 vs 25 m/s question once answered with the 15 m/s value for 25 m/s.

    Now the 15 m/s run must be made (the coverage check), and a "done" answer may
    hold no unsupported or contradicted statement (the claims check). Measured in
    four live runs: 159-268 s on a shared GPU, both speeds covered every time;
    one answer's "L/D is validated within 2-2.5 %" and one's "trustworthy for
    differences above ±1.5 %" passed the checker then and are caught now
    (test_ai_claims replays both). In the fifth run (198 s, final code) the model
    wrote "Best L/D is validated within 2-2.5% of reference solutions" again, and
    the claims check sent it back live.
    """
    from aerolume.ai.cli import run_experiment
    from aerolume.ai.config import AssistantConfig
    from aerolume.ai.experiment import load_record
    from aerolume.ai.providers.ollama import OllamaProvider
    project = _tapered_wing_copy(tmp_path)
    cfg = AssistantConfig(provider="ollama", ollama_model=MODEL, approval_threshold_s=600)
    goal = "How much does the best L/D of the wing “Tapered wing” change between 15 and 25 m/s?"
    out = run_experiment(project, goal, config=cfg, provider=OllamaProvider(MODEL, ENDPOINT),
                         approve=lambda a, _cid: a.kind == "run")
    _skip_if_the_model_could_not_answer(out)
    assert out.status in ("done", "unverified"), out.text
    speeds = {round(s.conditions.speed) for s in project.raw.simulations.values() if s.analysis == "wing"}
    assert {15, 25} <= speeds
    if out.status == "done":
        counts = load_record(out.path).claims.get("counts", {})
        assert counts.get("unsupported", 0) == 0 and counts.get("contradicted", 0) == 0, out.claims
