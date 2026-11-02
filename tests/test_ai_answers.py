"""What the assistant shows and says about its runs (critic round 3.2).

* answers are plain text: LaTeX math from the model is Unicode in the stream, the
  final answer, the claims check and the log;
* the local model's "context is nearly full" warning appears once per experiment;
* "run a new simulation" runs it again instead of returning a stored result, and a
  reused result is said to be one when the answer does not say it;
* a CFD run card leads with Nu, heat rate and fan thrust, not "Cd n/a, Cl n/a";
* the default context window is 16k tokens, and an old file's saved 8k default is raised;
* the example goals follow the project's physics.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

from aerolume.ai.agent import Agent, asks_new_run
from aerolume.ai.config import (
    DEFAULT_OLLAMA_NUM_CTX,
    AssistantConfig,
    load_config,
    save_config,
    settings_path,
)
from aerolume.ai.experiment import load_record
from aerolume.ai.providers.base import Turn
from aerolume.ai.tools import REUSED_NOTE, ToolContext, cfd_headline
from aerolume.ai.workspace import Workspace

POLAR = {"speed_m_s": 20, "alpha_start": 2, "alpha_end": 6, "alpha_step": 1}
CONTEXT = ("The local model's context is nearly full (7,309 of 8,192 tokens): older messages may be dropped. Start a "
           "new experiment, or raise the context size in Preferences > Assistant.")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    p = fakes.make_project(tmp_path_factory.mktemp("answers"))
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


def _agent(provider, project, rec):
    return Agent(provider, Workspace(project), config=AssistantConfig(approval_threshold_s=1e6), on_event=rec)


def _polar(project, did):
    r = project.run_polar(did, speed=20, alpha=(2, 6, 1))
    return r, float(r.metrics["ld_max"]), float(r.metrics["alpha_ld_max"])


# ---------------------------------------------------------------------------
# plain-text answers
# ---------------------------------------------------------------------------

def test_a_latex_answer_is_plain_text_in_the_stream_the_answer_and_the_log(project, did):
    _r, ld, a = _polar(project, did)
    latex = (f"The best L/D of NACA 2412 at 20 m/s is ${ld:.4g}$ at $\\alpha = {a:g}^\\circ$, known to "
             "$\\pm 0.5^\\circ$ (the step), from the stored result.")
    plain = f"The best L/D of NACA 2412 at 20 m/s is {ld:.4g} at α = {a:g}°, known to ±0.5° (the step), from the " \
            "stored result."
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_polar", {"design_id": did, **POLAR})), final(latex)])
    rec = Recorder()
    out = _agent(prov, project, rec).run("Find the best L/D of NACA 2412 at 20 m/s")
    assert out.text == plain
    streamed = "".join(e["delta"] for e in rec.of("text"))
    assert streamed == plain and "$" not in streamed and "\\" not in streamed
    assert rec.of("done")[-1]["text"] == plain
    record = load_record(out.path)
    said = [m["text"] for m in record.conversation if m["role"] == "assistant"]
    assert said and all("$" not in t for t in said) and record.conclusions == plain
    statements = rec.of("done")[-1]["claims"].get("statements", [])
    assert statements and all("\\" not in s["text"] for s in statements)


def test_the_context_warning_is_shown_once_per_experiment(project):
    turns = [Turn(text="", tool_calls=[call("o", "project_overview")], stop_reason="tool_use", notices=[CONTEXT]),
             Turn(text="", tool_calls=[call("p", "record_plan", {"objective": "nothing to run", "method": "read",
                                                                 "expected_runs": 0, "expected_seconds": 0})],
                  stop_reason="tool_use", notices=[CONTEXT]),
             Turn(text="The project holds one polar; nothing needed running.", stop_reason="end_turn",
                  notices=[CONTEXT])]
    rec = Recorder()
    _agent(fakes.ScriptedProvider(turns), project, rec).run("What does the project hold?")
    shown = [n for n in rec.of("notice") if "context is nearly full" in n["text"]]
    assert len(shown) == 1 and shown[0]["level"] == "warning"


# ---------------------------------------------------------------------------
# new runs and reused results
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("goal, new", [
    ("Run a new draft simulation of a differentially heated square cavity at Ra 1e5", True),
    ("Run a fresh CFD of the car", True),
    ("Re-run the cavity and compare", True),
    ("Run it again at standard quality", True),
    ("Compare the drag of the car at 20 and 30 m/s", False),
    ("Create a new NACA 4412 airfoil and find its best L/D", False),
    ("What is new in the stored results?", False),
])
def test_asks_new_run(goal, new):
    assert asks_new_run(goal) is new


def test_a_goal_that_asks_for_a_new_run_runs_it_again(project, did):
    sim_id = _polar(project, did)[0].sim_id
    ctx = ToolContext(Workspace(project), fresh=set())
    assert ctx.wants_fresh(sim_id) and not ToolContext(Workspace(project)).wants_fresh(sim_id)
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_simulation", {"simulation_id": sim_id})),
                                   tools_turn(call("r2", "run_simulation", {"simulation_id": sim_id})),
                                   final("Ran the polar again; the numbers are in the run card.")])
    rec = Recorder()
    _agent(prov, project, rec).run("Run a new polar simulation of NACA 2412 at 20 m/s")
    first, second = rec.of("tool_end")
    assert first["status"] == "ok" and "(stored result reused)" not in first["summary"]
    data = json.loads(prov.sessions[0].sent[1]["tool_results"][0].content)["data"]
    assert "reused_stored_result" not in data and "computed again" in data["ran_again"]
    # once per simulation: the model's second call reads the result it just made
    assert "(stored result reused)" in second["summary"]


def test_a_reused_result_is_said_to_be_one(project, did):
    sim_id = _polar(project, did)[0].sim_id
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_simulation", {"simulation_id": sim_id})),
                                   final("The polar of NACA 2412 is complete.")])
    rec = Recorder()
    _agent(prov, project, rec).run("Run the NACA 2412 polar")
    data = json.loads(prov.sessions[0].sent[1]["tool_results"][0].content)["data"]
    assert data["reused_stored_result"] is True and data["reused_note"] == REUSED_NOTE
    notes = [n["text"] for n in rec.of("notice") if n["text"].startswith("Not run again")]
    assert len(notes) == 1 and "is the stored result of an identical run computed earlier" in notes[0]
    # an answer that says so needs no note
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_simulation", {"simulation_id": sim_id})),
                                   final("The stored result of an identical polar was reused.")])
    rec = Recorder()
    _agent(prov, project, rec).run("Run the NACA 2412 polar")
    assert not [n for n in rec.of("notice") if n["text"].startswith("Not run again")]


# ---------------------------------------------------------------------------
# run cards
# ---------------------------------------------------------------------------

def test_a_cfd_card_leads_with_the_physics_it_measured():
    none = {"mean": None}
    cavity = {"analysis": "cfd2d", "coefficients": {"cd": none, "cl": none},
              "nusselt": {"mean": 4.5624, "stderr": 1.9e-5}, "heat_rate_W_per_m": 0.114}
    # 1.9e-5 on 4.56 is 4e-6 of it: steady, as the Results page says (critic round 3.3)
    assert cfd_headline(cavity) == "Nu 4.562 (steady), Q 0.114 W/m"
    room = {"analysis": "cfd3d", "coefficients": {"cd": none, "cl": none},
            "fans": [{"name": "Fan", "thrust": 1.002, "volume_flow": 0.4}]}
    assert cfd_headline(room) == "Fan thrust 1 N"
    cylinder = {"analysis": "cfd2d", "coefficients": {"cd": {"mean": 1.52, "stderr": 0.004}, "cl": none},
                "nusselt": {"mean": 3.28, "stderr": 0.01}}
    assert cfd_headline(cylinder) == "Nu 3.280 ± 0.010, Cd 1.520 ± 0.004"     # the Results rounding rule
    assert cfd_headline({"analysis": "cfd2d", "coefficients": {"cd": none, "cl": none}}) == "Cd n/a, Cl n/a"


# ---------------------------------------------------------------------------
# the local model's context window
# ---------------------------------------------------------------------------

def test_the_default_context_is_16k_and_an_old_saved_default_is_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path))
    assert AssistantConfig().ollama_num_ctx == DEFAULT_OLLAMA_NUM_CTX == 16384
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # a file from before: every field written, the then-default 8192 among them
    path.write_text(json.dumps({"provider": "ollama", "ollama_num_ctx": 8192}), encoding="utf-8")
    assert load_config().ollama_num_ctx == 16384
    path.write_text(json.dumps({"provider": "ollama", "ollama_num_ctx": 12288}), encoding="utf-8")
    assert load_config().ollama_num_ctx == 12288            # a choice, not the old default: kept
    # chosen after the change, 8192 stays 8192
    save_config(AssistantConfig(provider="ollama", ollama_num_ctx=8192))
    assert json.loads(path.read_text(encoding="utf-8"))["defaults_version"] == 2
    assert load_config().ollama_num_ctx == 8192


# ---------------------------------------------------------------------------
# example goals
# ---------------------------------------------------------------------------

def _sim(name, did, *, thermal=False, fans=()):
    return NS(name=name, design_id=did, analysis="cfd", thermal=NS(enabled=thermal), fans=list(fans))


def test_examples_follow_the_projects_physics():
    from aerolume.app.widgets.assistant_panel import physics_examples, selection_example
    designs = {"c": NS(name="Heated cavity", source={"type": "case", "case": "cavity"}),
               "f": NS(name="Fan in a room", source={"type": "case", "case": "fan_room"}),
               "b": NS(name="Heated box", source=None, bodies=[NS(thermal="temperature")], meshes=[])}
    sims = {"1": _sim("Heated cavity · Ra 1.0×10⁵", "c", thermal=True),
            "2": _sim("Fan in a room · 1 N/m", "f", fans=[NS(enabled=True)]),
            "3": _sim("Heated box", "b", thermal=True),
            "4": _sim("Second cavity", "c", thermal=True)}
    project = NS(designs=designs, simulations=sims)
    ex = physics_examples(project)
    assert len(ex) == 3
    assert ex[0].startswith("Compare the mean Nusselt number of “Second cavity” with de Vahl Davis (1983)")
    assert ex[1] == "What are the Nusselt number and heat rate of “Heated box”, and how far can they be trusted?"
    assert ex[2].startswith("What thrust and volume flow does the fan in “Fan in a room · 1 N/m” deliver")
    assert not any("drag" in e or "NACA" in e for e in ex)
    assert physics_examples(NS(designs={}, simulations={})) == [] and physics_examples(None) == []
    # the selected design's newest run decides its example; a heated design with no run asks for Nu
    assert "Nusselt" in selection_example(designs["c"], None, sims["1"])
    heated = selection_example(NS(name="Heated box", kind="generic", dimension=2, bodies=[NS(thermal="temperature")],
                                  meshes=[]), 0.5)
    assert heated.startswith("What are the Nusselt number and heat rate of the heated “Heated box” at 0.5 m/s")
