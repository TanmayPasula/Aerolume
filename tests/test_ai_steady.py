"""Steady results, validated reference set-ups and the plan's expected time (critic round 3.3).

The live qwen3:8b experiment "run a new draft cavity at Ra 1e4 and compare its
hot-wall Nu with de Vahl Davis" (a review run) ended with three
statements the checks passed:

1. the run card read "Nu 2.251 ± 4.4e-16" and the answer "2.25 ± 0.00
   (4.4e-16)": a steady run's batch-means error is round-off, not an uncertainty;
2. "The Nusselt number's absolute value and heat rate are not validated", right
   after matching de Vahl Davis to 0.575 %: the cavity is a validated set-up
   (docs/validation.md, +0.1 to +0.6 %);
3. the plan card showed the model's "Expected time 5 min 00 s" for an 11 s run.

The transcript is replayed with the scripted provider on a real headless project;
the solver is replaced by a stored result with the live run's numbers.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

from aerolume import api
from aerolume import model as M
from aerolume.ai import tools as T
from aerolume.ai.agent import Agent, states_uncertainty
from aerolume.ai.claims import check_answer
from aerolume.ai.config import AssistantConfig
from aerolume.ai.experiment import load_record
from aerolume.ai.tools import Toolbox, ToolContext, ToolOutcome, _outcome, cfd_headline, summarize_result
from aerolume.ai.workspace import Workspace

GOAL = ("Run a new draft simulation of the differentially heated square cavity at Ra 1e4 in this project and compare "
        "its hot-wall mean Nusselt number with de Vahl Davis (1983). Report the number with its uncertainty and say "
        "what limits it.")
#: the live answer, word for word (markdown kept)
LIVE_ANSWER = ("The hot-wall mean Nusselt number for the heated cavity at Ra 1.0×10⁴ is **2.25 ± 0.00 (4.4e-16)**, "
               "with a grid of [64, 64]. This result differs from the de Vahl Davis (1983) reference value of 2.238 by "
               "**0.575%**.\n\n**Validity limits:** The Nusselt number's absolute value and heat rate are not "
               "validated. The result is based on a 2D CFD simulation with laminar turbulence, and the discretisation "
               "error requires a grid study for trust.")
NU, SE = 2.2509, 4.4e-16                       # the live run (steady to round-off)


class LocalScripted(fakes.ScriptedProvider):
    name = "ollama"


def _cavity_result(sim_id: str, nu: float, se: float) -> M.CFDResult:
    t = np.linspace(0.0, 30.0, 301)
    nan = M.Stat(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    return M.CFDResult(sim_id=sim_id, cd=nan, cl=nan, cm=nan, re_real=float("nan"), re_sim=float("nan"),
                       grid=(64, 64), cells_per_ref=64, t_end=30.0, t_avg_start=10.0, converged=True,
                       wall_seconds=11.0, device="cpu", precision="float32", turbulence="laminar",
                       nusselt=M.Stat(nu, 1e-12, nu, nu, se), heat_rate=0.00504,
                       history={"t": t, "cd": 0 * t, "cl": 0 * t, "cm": 0 * t, "nu": nu + 0 * t})


def _store(project, sim_id: str, res: M.CFDResult) -> None:
    project.raw.save_result(sim_id, res)
    sim = project.raw.simulations[sim_id]
    sim.status = "done"
    project.raw.update_simulation(sim)


@pytest.fixture()
def project(tmp_path):
    return fakes.make_project(tmp_path, "Steady", naca=None)


def _toolbox(project, nu: float = NU, se: float = SE) -> Toolbox:
    tb = Toolbox()

    def fake_run(ctx: ToolContext, args: dict) -> ToolOutcome:
        sid = args["simulation_id"]
        _store(project, sid, _cavity_result(sid, nu, se))
        return _outcome(project.result(sid), project.raw.simulations[sid].name)

    tb.tools["run_simulation"] = dataclasses.replace(tb.tools["run_simulation"], handler=fake_run, estimate=None)
    return tb


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]


# ---------------------------------------------------------------------------
# one rule for "steady", shared with the Results page
# ---------------------------------------------------------------------------

def test_the_steady_rule_and_the_run_card_texts_are_the_results_pages():
    kpi = pytest.importorskip("aerolume.app.widgets.kpi")
    assert T.STEADY_REL == kpi.STEADY_REL
    for mean, se in ((NU, SE), (4.5624, 1.9e-5), (3.28, 0.01), (1.52, 0.004), (0.51377, 0.004553),
                     (-0.0016, 0.0028), (1.002, float("nan")), (0.0, 0.0)):
        assert T.is_steady(mean, se) == kpi.is_steady(mean, se)
        assert T.value_error_texts(mean, se) == kpi.value_error_texts(mean, se)


def test_without_qt_the_card_applies_the_same_rule(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_qt(name, *a, **k):
        if "widgets.kpi" in name:
            raise ImportError("no Qt in this build")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_qt)
    assert T.value_error_texts(NU, SE) == ("2.251", "steady")
    assert T.value_error_texts(3.28, 0.01) == ("3.280", "0.010")
    assert T.value_error_texts(1.002, float("nan")) == ("1.002", "—")


def test_a_steady_statistic_is_sent_without_its_round_off():
    st = T._stat(M.Stat(NU, 0.0, NU, NU, SE))
    assert st == {"mean": 2.251, "steady": True}
    noisy = T._stat(M.Stat(3.3402, 0.01, 3.32, 3.36, 0.001))
    assert noisy == {"mean": 3.3402, "stderr": 0.001}
    assert cfd_headline({"analysis": "cfd2d", "coefficients": {}, "nusselt": st,
                         "heat_rate_W_per_m": 0.00504}) == "Nu 2.251 (steady), Q 0.00504 W/m"


# ---------------------------------------------------------------------------
# a validated reference set-up and the steady result in the summary
# ---------------------------------------------------------------------------

def test_a_validated_cavity_run_lists_nu_as_validated_with_its_deviation(project):
    sim = project.add_simulation(api.cavity_case(ra=1e4), quality="draft")
    _store(project, sim.id, _cavity_result(sim.id, NU, SE))
    r = project.result(sim.id)
    summ = summarize_result(r.raw, r.simulation, r.design.raw)
    assert summ["nusselt"] == {"mean": 2.251, "steady": True}
    assert summ["steady_quantities"] == ["nusselt"] and "run_grid_study (GCI)" in summ["steady_meaning"]
    ref = summ["reference_check"]
    assert ref["steady"] is True and "stderr" not in ref and "reference_within_2_stderr" not in ref
    assert ref["reference"] == 2.238 and ref["difference_pct"] == pytest.approx(100 * (NU - 2.238) / 2.238, abs=0.01)
    assert "Nusselt number (absolute value)" not in summ.get("not_validated", [])
    assert "heat rate" in summ["not_validated"]                 # no benchmark checks it directly
    entry = next(v for v in summ["validated_quantities"] if v.startswith("Nusselt number"))
    assert "against 2.238" in entry and "+0.58 %" in entry and "de Vahl Davis" in entry
    # the log's Results table says "steady", not "± 4.4e-16"
    rows = T.key_numbers(summ)
    assert next(r for r in rows if r["label"] == "Nusselt number")["uncertainty"] == "steady"


def test_a_cavity_whose_walls_left_the_benchmark_keeps_nu_unvalidated(project):
    case = api.cavity_case(ra=1e4)
    sim = project.add_simulation(case, quality="draft")
    raw = project.raw.simulations[sim.id]
    raw.thermal.walls = [w if w.side != "left" else dataclasses.replace(w, value=w.value + 5.0)
                         for w in raw.thermal.walls]
    project.raw.update_simulation(raw)
    _store(project, sim.id, _cavity_result(sim.id, NU, SE))
    r = project.result(sim.id)
    summ = summarize_result(r.raw, r.simulation, r.design.raw)
    assert summ["reference_check"]["applies"] is False
    assert "Nusselt number (absolute value)" in summ["not_validated"]
    assert not any(v.startswith("Nusselt number") for v in summ["validated_quantities"])


# ---------------------------------------------------------------------------
# the claims check and the uncertainty statement of a steady value
# ---------------------------------------------------------------------------

def test_a_plus_minus_for_a_steady_value_is_contradicted():
    def problems(text: str, steady=("nusselt",)) -> list[str]:
        report = check_answer(text, sources=[], sims=[], actions=[], steady=list(steady))
        # only this check's problems: the bare sentences have no tool results to trace their numbers to
        return [p for st in report.problems for p in st.problems if "round-off" in p]

    assert any("steady" in p for p in problems("The hot-wall mean Nusselt number is 2.25 ± 0.00 (4.4e-16)."))
    assert any("steady" in p for p in problems("Nu is 2.251 ± 4.4e-16."))
    # a ± of zero is round-off whatever it is given for
    assert any("± 0" in p for p in problems("The drag is 0.51 ± 0.000.", steady=()))
    # the GCI of a grid study is the right uncertainty of a steady value
    assert problems("Nu is 2.251 ± 0.012 (GCI of the grid study).") == []
    assert problems("Cd is 1.52 ± 0.004 (standard error).") == []
    assert states_uncertainty("Nu is 2.251 (steady); its discretisation error needs a grid study (GCI).")
    # the two halves of the statement may be sentences apart
    assert states_uncertainty("Nu is 2.251 (steady: the run settled), against 2.238 from de Vahl Davis (1983): "
                              "+0.57 %. The cavity is a validated set-up; this run's discretisation error needs a "
                              "grid study (GCI).")
    assert not states_uncertainty("Nu is 2.251 and the flow is steady.")


ENTRY = ("Nusselt number: validated set-up (docs/validation.md); this run 2.2509 against 2.238, +0.57 % "
         "(de Vahl Davis (1983), Nu₀: mean Nu of the hot wall)")


def test_calling_a_validated_set_ups_nu_not_validated_is_contradicted():
    def problems(text: str) -> list[str]:
        report = check_answer(text, sources=[], sims=[], actions=[], validated=[ENTRY])
        return [p for st in report.problems for p in st.problems if "lists it as validated" in p]

    live = problems("The Nusselt number's absolute value and heat rate are not validated.")
    assert live == [("it says Nusselt number is not validated, but the result lists it as validated: this run "
                     "2.2509 against 2.238, +0.57 % (de Vahl Davis (1983), Nu₀: mean Nu of the hot wall)")]
    assert problems("Nu is not validated.") and problems("The heat rate is not validated.") == []
    assert problems("Nu is validated against de Vahl Davis.") == []          # not negated


# ---------------------------------------------------------------------------
# the live experiment, replayed
# ---------------------------------------------------------------------------

def test_the_live_cavity_experiment_now_reports_steady_validated_and_the_planners_time(project):
    seen: dict = {}

    def creates(_s, _u, _r):
        return tools_turn(call("k1", "create_case", {"case": "cavity", "rayleigh": 1e4}))

    def plans(_s, _u, results):
        seen["case"] = json.loads(results[0].content)["data"]
        return tools_turn(call("p1", "record_plan", {
            "objective": "Compare the hot-wall mean Nusselt number with de Vahl Davis (1983).",
            "method": "CFD simulation of the cavity", "variables": ["Rayleigh number"],
            "controls": ["Geometry", "boundary conditions"], "expected_runs": 1, "expected_seconds": 300}))

    def runs(_s, _u, results):
        seen["plan"] = json.loads(results[0].content)["data"]
        return tools_turn(call("r1", "run_simulation", {"simulation_id": seen["case"]["simulation_id"]}))

    def compares(_s, _u, results):
        seen["run"] = json.loads(results[0].content)["data"]
        return tools_turn(call("c1", "compare_reference", {"simulation_ids": [seen["case"]["simulation_id"]]}))

    def concludes(_s, _u, _r):
        sid = seen["case"]["simulation_id"]
        return tools_turn(call("f1", "record_conclusion", {
            "summary": "Nu 2.251 against 2.238", "findings": [{"quantity": "Nu", "value": 2.251, "simulation_id": sid}]}))

    def corrects(_s, user, _r):
        seen["claims_note"] = user
        return final("The hot-wall mean Nusselt number of the heated cavity at Ra 1.0×10⁴ is 2.251 (steady: the run "
                     "settled, so it has no averaging uncertainty), against 2.238 from de Vahl Davis (1983): +0.58 %. "
                     "Its discretisation error needs a grid study (GCI); the heat rate is not validated.")

    prov = LocalScripted([creates, plans, runs, compares, concludes, final(LIVE_ANSWER), corrects])
    rec = Recorder()
    cfg = AssistantConfig(approval_threshold_s=1e9, max_steps=20)
    agent = Agent(prov, Workspace(project), config=cfg, toolbox=_toolbox(project), approve=lambda *_a: True,
                  on_event=rec)
    out = agent.run(GOAL)

    sid = seen["case"]["simulation_id"]
    planner = float(project.plan_of(sid).est_seconds)
    # 3. the plan's expected time is the planner's, and the card is given the recorded plan
    assert seen["plan"]["expected_seconds"] == pytest.approx(planner, rel=1e-2)
    assert seen["plan"]["expected_time_basis"] == "planner estimate, 1 run" and "5 min" in seen["plan"]["note"]
    plan_end = next(e for e in rec.of("tool_end") if e["name"] == "record_plan")
    assert plan_end["plan"]["expected_seconds"] == pytest.approx(planner, rel=1e-2)
    assert math.isfinite(planner) and abs(planner - 300.0) > 1.0
    # 1. the run card reads "steady", as Results does
    run_end = next(e for e in rec.of("tool_end") if e["name"] == "run_simulation")
    assert "Nu 2.251 (steady)" in run_end["summary"] and "e-16" not in run_end["summary"]
    # 2. the cavity's Nu is validated, with this run's deviation
    assert "Nusselt number (absolute value)" not in seen["run"].get("not_validated", [])
    assert any(v.startswith("Nusselt number: validated set-up") for v in seen["run"]["validated_quantities"])
    # the live answer's "± 0.00 (4.4e-16)" was sent back, and the corrected answer passes
    assert "steady" in seen["claims_note"] and "grid study" in seen["claims_note"]
    assert "it says Nusselt number is not validated, but the result lists it as validated" in seen["claims_note"]
    assert out.status == "done", out
    assert out.claims["counts"]["contradicted"] == 0 and out.claims["counts"]["unsupported"] == 0
    md = (Path(out.path) / "experiment.md").read_text(encoding="utf-8")
    assert "(planner estimate, 1 run)" in md and "5 min 00 s" not in md


def test_a_plan_without_a_planner_estimate_shows_no_expected_time(project):
    prov = LocalScripted([tools_turn(call("p1", "record_plan", {"objective": "Look at the project", "method": "read",
                                                                "expected_seconds": 300})),
                          final("Nothing to run.")])
    rec = Recorder()
    cfg = AssistantConfig(approval_threshold_s=1e9, max_steps=5)
    out = Agent(prov, Workspace(project), config=cfg, approve=lambda *_a: True, on_event=rec).run("Plan only")
    record = load_record(out.path)
    assert record.plan is not None and "expected_seconds" not in record.plan
    end = next(e for e in rec.of("tool_end") if e["name"] == "record_plan")
    assert "expected_seconds" not in end["plan"] and "about" not in end["summary"]
