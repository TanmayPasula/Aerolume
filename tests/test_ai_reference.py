"""Reference comparisons, the keyword router and the physics-model claims check (critic round 3).

Two live qwen3:8b experiments ended "Done" with answers that did not do what
the goal asked, and the loop accepted both:

1. "... mean Nusselt number of the heated cylinder ... compare it with the Lange
   correlation": the answer gave "Nu 3.3402 ± 0.001" and no comparison (Lange
   3.28, +1.8 %), and cited "the assumptions made in the Boussinesq energy
   equation" for a run that had buoyancy off;
2. "Run a new draft simulation of a differentially heated square cavity at
   Rayleigh number 1e5 and compare ... with de Vahl Davis": after an empty reply
   the model answered "The tools cannot run a simulation for a differentially
   heated square cavity" -- the local tool list really lacked create_case.

Both transcripts are replayed here with the scripted provider in the local-model
profile on a real headless project; the solver is replaced by stored results
(the numbers of the live run) so the test needs no GPU.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

from aerolume import api
from aerolume import model as M
from aerolume.ai import router
from aerolume.ai.agent import Agent
from aerolume.ai.claims import SimRecord, check_answer
from aerolume.ai.config import AssistantConfig
from aerolume.ai.experiment import load_record
from aerolume.ai.prompts import EMPTY_NUDGE
from aerolume.ai.tools import Toolbox, ToolContext, ToolError, ToolOutcome, _outcome, summarize_result
from aerolume.ai.workspace import Workspace

CYL_GOAL = ("Measure the mean Nusselt number of the heated cylinder in this project at Re 40 with a draft run and "
            "compare it with the Lange correlation. Report the number with its uncertainty.")
#: the live answer of the review run (critic round 3.1), word for word
CYL_ANSWER = ("The mean Nusselt number of the heated cylinder at Re 40 with a draft run is **3.3402 ± 0.001 "
              "(uncertain)**. The result is not validated for the absolute value of the Nusselt number or the heat "
              "rate. The simulation did not converge, and the validity limits include the lack of convergence and "
              "the assumptions made in the Boussinesq energy equation.")
CAV_GOAL = ("Run a new draft simulation of a differentially heated square cavity at Rayleigh number 1e5 and compare "
            "its mean Nusselt number with de Vahl Davis (1983). Give the difference in percent and its uncertainty.")
#: the live answer of the review run (critic round 3.1), word for word
CAV_CANNOT = ("The tools cannot run a simulation for a differentially heated square cavity at Rayleigh number $10^5$ "
              "because it is not listed in the project overview. The available simulations are for other designs, "
              "such as a heated cylinder, heated cavity, fan in a room, car body, hot sphere, and heated box. To "
              "proceed, you would need to create a new design for the differentially heated square cavity and then "
              "run the simulation.")


class LocalScripted(fakes.ScriptedProvider):
    """The scripted provider in the local-model (Ollama) profile: compact tools, the loop's overview."""
    name = "ollama"


def _cfd(sim_id: str, nu: float, se: float, *, re: float, converged: bool = False) -> M.CFDResult:
    t = np.linspace(0.0, 30.0, 301)
    return M.CFDResult(sim_id=sim_id, cd=M.Stat(1.52, 0.01, 1.5, 1.54, 0.002), cl=M.Stat(0.0, 0.01, -0.01, 0.01, 0.001),
                       cm=M.Stat(0.0, 0.0, 0.0, 0.0, 1e-6), re_real=re, re_sim=re, grid=(320, 160), cells_per_ref=16,
                       t_end=30.0, t_avg_start=10.0, converged=converged, wall_seconds=60.0, device="cpu",
                       precision="float32", turbulence="laminar", nusselt=M.Stat(nu, 0.01, nu - 0.02, nu + 0.02, se),
                       heat_rate=1.25, history={"t": t, "cd": 1.52 + 0 * t, "cl": 0 * t, "cm": 0 * t, "nu": nu + 0 * t})


def _store(project, sim_id: str, res: M.CFDResult) -> None:
    project.raw.save_result(sim_id, res)
    sim = project.raw.simulations[sim_id]
    sim.status = "done"
    project.raw.update_simulation(sim)


@pytest.fixture()
def cylinder(tmp_path):
    """A project holding the heated-cylinder case at Re 40 with the live run's Nu (3.3402 ± 0.001, not converged)."""
    p = fakes.make_project(tmp_path, "Reference", naca=None)
    sim = p.add_simulation(api.heated_cylinder_case(re=40), quality="draft")
    _store(p, sim.id, _cfd(sim.id, 3.3402, 0.001, re=40.0))
    return p, sim.id


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]


def _agent(provider, project, *, toolbox=None, rec=None) -> Agent:
    cfg = AssistantConfig(approval_threshold_s=1e9, max_steps=20)
    return Agent(provider, Workspace(project), config=cfg, toolbox=toolbox, approve=lambda *_a: True, on_event=rec)


# ---------------------------------------------------------------------------
# transcript 1: the heated cylinder against Lange et al.
# ---------------------------------------------------------------------------

def test_a_lange_goal_gets_the_comparison_and_the_buoyancy_claim_is_contradicted(cylinder):
    project, sid = cylinder
    lange = round(api.lange_nusselt(40.0), 4)
    diff = 100.0 * (3.3402 - lange) / lange
    seen: dict = {}

    def reads(_s, user, _r):
        seen["goal"] = user
        return tools_turn(call("g1", "get_results", {"simulation_ids": [sid]}))

    def concludes(_s, _u, results):
        seen["read"] = json.loads(results[0].content)["data"]["results"][0]
        return tools_turn(call("c1", "record_conclusion", {
            "summary": "Nu 3.3402", "findings": [{"quantity": "Nu", "value": 3.3402, "simulation_id": sid}]}))

    def compares(_s, user, _r):
        seen["reference_note"] = user
        return final(f"The mean Nusselt number is 3.3402 ± 0.001 (standard error), against {lange:.4g} from the "
                     f"Lange et al. correlation: {diff:+.1f} %. The run did not converge, and the assumptions made in "
                     "the Boussinesq energy equation limit it.")

    def corrects(_s, user, _r):
        seen["claims_note"] = user
        return final(f"The mean Nusselt number of the heated cylinder at Re 40 is 3.3402 ± 0.001 (standard error of "
                     f"the averaging), against {lange:.4g} from the Lange et al. (1998) correlation: {diff:+.1f} %. "
                     "The draft run did not converge and ran with buoyancy off (forced convection); the "
                     "discretisation error is not included.")

    prov = LocalScripted([reads, concludes, final(CYL_ANSWER), compares, corrects])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run(CYL_GOAL)

    # the result the model read already carried the comparison and the models that ran
    read = seen["read"]
    assert read["reference_check"]["reference"] == pytest.approx(lange)
    assert read["reference_check"]["difference_pct"] == pytest.approx(diff, abs=0.01)
    assert read["physics_models"] == {"heat_transfer": True, "buoyancy": False, "domain": "tunnel",
                                      "turbulence": "laminar"}
    assert not any("Boussinesq" in v for v in read["validity"])          # the sentence the model had copied
    assert any("buoyancy OFF" in v for v in read["validity"])

    # 1. the answer without a comparison: the loop called compare_reference itself and sent the numbers back
    assert "Lange et al. (1998) correlation" in seen["reference_note"]
    assert f"reference {lange:g}" in seen["reference_note"] and f"{diff:+.3g} %" in seen["reference_note"]
    starts = [e["name"] for e in rec.of("tool_start")]
    assert starts == ["project_overview", "get_results", "record_conclusion", "compare_reference"]
    # 2. the Boussinesq sentence was contradicted by the claims check (buoyancy was off)
    assert "buoyancy was off" in seen["claims_note"] and "Boussinesq" in seen["claims_note"]
    # 3. the corrected answer passes every check
    assert out.status == "done", out
    assert out.claims["counts"]["contradicted"] == 0 and out.claims["counts"]["unsupported"] == 0
    record = load_record(out.path)
    assert [a.tool for a in record.actions] == ["project_overview", "get_results", "record_conclusion",
                                                "compare_reference"]
    notices = " ".join(n["text"] for n in rec.of("notice"))
    assert "comparison with the Lange et al. (1998) correlation" in notices


def test_an_answer_that_still_leaves_the_comparison_out_ends_unverified(cylinder):
    project, sid = cylinder
    prov = LocalScripted([tools_turn(call("g1", "get_results", {"simulation_ids": [sid]})),
                          tools_turn(call("c1", "record_conclusion", {"summary": "Nu", "findings": []})),
                          final("Nu is 3.3402 ± 0.001 (standard error)."),
                          final("Nu is 3.3402 ± 0.001 (standard error).")])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run(CYL_GOAL)
    assert out.status == "unverified"
    record = load_record(out.path)
    assert any(c.startswith("Completeness check: the goal asks for a comparison with the Lange") for c in record.checks)
    assert any("Completeness check" in n["text"] for n in rec.of("notice"))


# ---------------------------------------------------------------------------
# transcript 2: a new cavity at Ra 1e5 against de Vahl Davis
# ---------------------------------------------------------------------------

def _fake_run_toolbox(project, nu: float, se: float) -> Toolbox:
    """A toolbox whose run_simulation stores a result with this Nu instead of starting the solver."""
    tb = Toolbox()

    def fake_run(ctx: ToolContext, args: dict) -> ToolOutcome:
        sid = args["simulation_id"]
        _store(project, sid, _cfd(sid, nu, se, re=float("nan"), converged=True))
        return _outcome(project.result(sid), project.raw.simulations[sid].name)

    tb.tools["run_simulation"] = dataclasses.replace(tb.tools["run_simulation"], handler=fake_run, estimate=None)
    return tb


def test_a_cannot_answer_is_routed_to_create_case_and_the_cavity_is_compared(cylinder):
    project, _sid = cylinder
    seen: dict = {}

    def routed(_s, user, _r):
        seen["router_note"] = user
        return tools_turn(call("k1", "create_case", {"case": "cavity", "rayleigh": 1e5}))

    def runs(_s, _u, results):
        seen["case"] = json.loads(results[0].content)["data"]
        cid = seen["case"]["simulation_id"]
        return tools_turn(call("p1", "record_plan", {"objective": "Nu at Ra 1e5", "method": "cavity case"}),
                          call("r1", "run_simulation", {"simulation_id": cid}))

    def answers(_s, _u, results):
        seen["run"] = json.loads(results[1].content)["data"]
        return final("The cavity ran at Ra 1e5 and its mean Nusselt number is 4.53 ± 0.004.")

    def compares(_s, user, _r):
        seen["reference_note"] = user
        return final("The mean Nusselt number of the hot wall is 4.53 ± 0.004 (standard error), against 4.509 from "
                     "de Vahl Davis (1983): +0.47 %. The discretisation error is not included.")

    def concludes(_s, _u, _r):
        cid = seen["case"]["simulation_id"]
        return tools_turn(call("c1", "record_conclusion", {
            "summary": "Nu 4.53 against 4.509", "findings": [{"quantity": "Nu", "value": 4.53, "simulation_id": cid}]}))

    again = ("The mean Nusselt number of the hot wall is 4.53 ± 0.004 (standard error), against 4.509 from de Vahl "
             "Davis (1983): +0.47 %. The discretisation error is not included.")
    steps = [tools_turn(call("o2", "project_overview", {})), final(""), final(CAV_CANNOT), routed, runs, answers,
             compares, concludes, final(again)]
    prov = LocalScripted(steps)
    rec = Recorder()
    out = _agent(prov, project, toolbox=_fake_run_toolbox(project, 4.53, 0.004), rec=rec).run(CAV_GOAL)

    session = prov.sessions[0]
    assert session.sent[2]["user_text"] == EMPTY_NUDGE                      # the empty reply was asked about
    # the refusal was sent back once with the concrete calls, parameters read from the goal
    assert 'create_case with case="cavity", rayleigh=100000' in seen["router_note"]
    assert "run_simulation" in seen["router_note"] and "compare_reference" in seen["router_note"]
    assert any("cannot be done, but create_case" in n["text"] for n in rec.of("notice"))
    # the case is de Vahl Davis' hot-wall Nu0 at Ra 1e5, which is what the run measures
    assert seen["case"]["reference"]["value"] == 4.509 and "Nu₀" in seen["case"]["reference"]["source"]
    assert seen["run"]["reference_check"]["reference"] == 4.509
    # the answer without the comparison got the loop's compare_reference numbers
    assert "de Vahl Davis (1983)" in seen["reference_note"] and "reference 4.509" in seen["reference_note"]
    assert out.status == "done", out
    assert out.claims["counts"]["contradicted"] == 0 and out.claims["counts"]["unsupported"] == 0
    tools = [a.tool for a in load_record(out.path).actions]
    assert tools[:4] == ["project_overview", "project_overview", "create_case", "record_plan"]
    assert "run_simulation" in tools and "compare_reference" in tools and tools[-1] == "record_conclusion"


def test_a_cannot_after_trying_the_route_is_accepted(cylinder):
    """A refusal after create_case / run_simulation failed is the honest answer: no second push."""
    project, _sid = cylinder
    tb = Toolbox()

    def failing(_ctx, _args):
        raise ToolError("This version's 2D solver does not implement heat transfer yet.")

    tb.tools["create_case"] = dataclasses.replace(tb.tools["create_case"], handler=failing)
    prov = LocalScripted([tools_turn(call("k1", "create_case", {"case": "cavity", "rayleigh": 1e5})),
                          final("The cavity cannot be run: this version's solver does not implement heat transfer "
                                "yet (create_case failed).")])
    out = _agent(prov, project, toolbox=tb).run(CAV_GOAL)
    assert out.status in ("done", "unverified")
    assert all("cannot be done, but" not in str(s.get("user_text")) for s in prov.sessions[0].sent)


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("goal", "case", "args"), [
    (CAV_GOAL, "cavity", {"rayleigh": 1e5}),
    ("Heated cavity at Ra = 1.0×10⁶ against de Vahl Davis", "cavity", {"rayleigh": 1e6}),
    ("natural convection in a cubic cavity at Ra 10^4", "cavity", {"rayleigh": 1e4, "dimension": 3}),
    ("Nusselt number of a heated cylinder at Re 20 vs Lange", "heated_cylinder", {"reynolds": 20.0}),
    ("How much flow does a 2 N fan move in my room?", "fan_room", {"thrust_n": 2.0}),
])
def test_the_router_maps_keywords_to_reference_cases(goal, case, args):
    r = router.route(goal)
    assert r is not None and r.case == case and r.args == args


def test_the_router_ignores_goals_without_a_reference_case():
    assert router.route("Find the best L/D of NACA 2412 at 20 m/s") is None
    assert router.asks_reference("Compare the drag of the two Ahmed bodies") == ("", "")
    assert router.asks_reference(CYL_GOAL)[1] == "heated_cylinder"
    assert router.says_cannot(CAV_CANNOT) and not router.says_cannot(CYL_ANSWER)
    assert router.parse_value("10⁵") == 1e5 and router.parse_value("10^5") == 1e5 and router.parse_value("1e5") == 1e5


def test_a_comparison_needs_the_reference_or_a_percent_difference():
    checks = [{"reference": 3.2823, "difference_pct": 1.764}]
    assert not router.gives_comparison(CYL_ANSWER, checks)
    assert router.gives_comparison("Nu 3.34 against 3.28 (Lange).", checks)
    assert router.gives_comparison("Nu 3.34, +1.8 % above the correlation.", checks)
    assert not router.gives_comparison("Nu 3.34 from 2 runs.", checks)      # a count is not a difference


def _rec(**physics) -> SimRecord:
    return SimRecord("s1", "“Heated cylinder”", "Heated cylinder", 0.06, [3.3402], {"nusselt": 3.3402},
                     physics=physics)


@pytest.mark.parametrize(("text", "physics", "bad"), [
    ("The limits include the assumptions made in the Boussinesq energy equation.", {"buoyancy": False}, True),
    ("Natural convection lifts the plume above the cylinder.", {"buoyancy": False}, True),
    ("Buoyancy was off, so this is forced convection.", {"buoyancy": False}, False),
    ("Buoyancy would raise Nu if it were switched on.", {"buoyancy": False}, False),
    ("The Boussinesq buoyancy drives the cavity flow.", {"buoyancy": True}, False),
    ("Buoyancy was neglected in this run.", {"buoyancy": True}, True),
    ("The LES subgrid model resolves the wake.", {"buoyancy": False, "turbulence": "laminar"}, True),
    ("It was a laminar simulation.", {"buoyancy": False, "turbulence": "les"}, True),
    ("It was a laminar simulation.", {"buoyancy": False, "turbulence": "laminar"}, False),
])
def test_statements_about_physics_models_are_checked_against_the_settings(text, physics, bad):
    report = check_answer(text, sources=[], sims=[_rec(**physics)], actions=[])
    st = report.statements[0]
    assert (st.status == "contradicted") is bad, (text, st)


def test_get_results_states_the_models_and_a_changed_case_loses_its_reference(cylinder):
    project, sid = cylinder
    r = project.result(sid)
    s = summarize_result(r.raw, r.simulation, r.design.raw)
    assert s["reference_check"]["applies"] and s["reference_check"]["kind"] == "validation"
    # buoyancy switched on: the forced-convection correlation no longer applies, and the result says why
    sim = project.raw.simulations[sid]
    sim.thermal.buoyancy = True
    s2 = summarize_result(r.raw, sim, r.design.raw)
    assert s2["reference_check"] == {"applies": False, "why": pytest.approx(s2["reference_check"]["why"])}
    assert "forced convection" in s2["reference_check"]["why"]


@pytest.mark.parametrize(("text", "refusal"), [
    (CAV_CANNOT, True),
    ("I can't set up that case.", True),
    ("It is not possible to simulate a cavity with these tools.", True),
    ("The tools do not support natural convection.", True),
    ("There is no tool to run a cavity.", True),
    ("The difference isn't significant.", False),
    ("The simulation did not converge.", False),
    ("This cannot be trusted without a grid study.", False),
    (CYL_ANSWER, False),
])
def test_only_a_refusal_of_the_work_counts_as_cannot(text, refusal):
    """Any negation would push answers that merely say "did not converge" back to create_case."""
    assert router.says_cannot(text) is refusal
