"""The claims checker, fail-closed approvals, data consent and the new assistant tools.

The two blockers of the second critique are replayed with a scripted model on a
real project (real tools, API, solvers and logs):

* asked to delete a simulation, a local model answered "has been deleted"
  without any tool call (it had no delete tool);
* it wrote "the stored 25 m/s polar's best L/D of 26.47" when that run's value
  is 29.52 (26.47 belongs to the 15 m/s run).

Every final answer now goes through :func:`aerolume.ai.claims.check_answer`; the
panel shows each checked statement and the log records the verdicts.
"""
from __future__ import annotations

import json

import pytest
import test_ai_fakes as fakes
from test_ai_fakes import call, final, tools_turn

from aerolume.ai.agent import Agent, consent_key, data_destination
from aerolume.ai.claims import (
    ActionRecord,
    ClaimReport,
    NumberSource,
    SimRecord,
    check_answer,
    claimed_actions,
    split_statements,
)
from aerolume.ai.config import AssistantConfig
from aerolume.ai.experiment import load_record, render_markdown
from aerolume.ai.prompts import CLAIMS_NUDGE
from aerolume.ai.tools import TOOLS, Approval, InputError, Toolbox, ToolContext, physics_settings, run_items
from aerolume.ai.workspace import Workspace


class LocalScripted(fakes.ScriptedProvider):
    """A scripted provider the agent treats as a small local model (the Ollama profile, on localhost)."""
    name = "ollama"
    endpoint = "http://127.0.0.1:11434"


class CloudScripted(fakes.ScriptedProvider):
    """A scripted provider that stands for Claude: its data would leave the computer."""
    name = "claude"


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, data))

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]


def _agent(provider, project, *, threshold=1e6, approve=None, rec=None, consent=None, cfg=None):
    cfg = cfg or AssistantConfig(approval_threshold_s=threshold, max_steps=40)
    return Agent(provider, Workspace(project), config=cfg, approve=approve, on_event=rec, consent=consent)


@pytest.fixture(scope="module")
def wing_project(tmp_path_factory):
    """A tapered wing with viscous polars at 15 and 25 m/s (about 10 s, once for the module)."""
    p = fakes.make_project(tmp_path_factory.mktemp("claims"))
    w = p.add_wing(2.0, 0.3, 0.18, airfoil="2412", name="Tapered wing")
    r15 = p.run_wing(w.id, speed=15, alpha=(-2, 8, 2))
    r25 = p.run_wing(w.id, speed=25, alpha=(-2, 8, 2))
    return p, w.id, r15, r25


# ---------------------------------------------------------------------------
# the checker itself (no project)
# ---------------------------------------------------------------------------

S15 = SimRecord("a", "“Wing polar · 15 m/s”", "Tapered wing", 15.0, [26.47, 3.0, 0.98, 1.31],
                {"ld_max": 26.47, "alpha_ld_max": 3.0, "cl_max": 1.31})
S25 = SimRecord("b", "“Wing polar · 25 m/s”", "Tapered wing", 25.0, [29.52, 3.0, 0.99, 1.33],
                {"ld_max": 29.52, "alpha_ld_max": 3.0, "cl_max": 1.33})
SOURCES = [NumberSource(26.47, "result", "ld_max in “Wing polar · 15 m/s” (run_wing)", "a", "ld_max"),
           NumberSource(29.52, "result", "ld_max in “Wing polar · 25 m/s” (get_results)", "b", "ld_max"),
           NumberSource(15.0, "goal", "the goal"), NumberSource(25.0, "goal", "the goal"),
           NumberSource(0.5, "result", "optimum_resolution_deg in “Wing polar · 15 m/s” (run_wing)", "a")]
ACTIONS = [ActionRecord("project_overview", "read", "ok"), ActionRecord("run_wing", "run", "ok", ["Tapered wing"])]
DESIGNS = ["Tapered wing", "NACA 2412"]


#: what a wing result lists under validated_quantities (tools.VALIDATED["wing"])
WING_VALIDATED = ["CL_alpha", "e (span efficiency)", "CDi"]


def _check(text: str, actions=ACTIONS) -> ClaimReport:
    return check_answer(text, sources=SOURCES, sims=[S15, S25], actions=actions, designs=DESIGNS,
                        model_validated=WING_VALIDATED)


def test_the_misattributed_best_ld_is_contradicted_with_the_value_it_should_be():
    live = ("The best L/D for the Tapered wing at 15 m/s is **26.47**, matching the stored 25 m/s polar's best "
            "L/D of **26.47**. Both results are validated for **CL_alpha**.")
    rep = _check(live)
    first = rep.statements[0]
    assert first.status == "contradicted"
    bad = [n for n in first.numbers if n.status == "contradicted"]
    assert bad and bad[0].token == "26.47" and bad[0].correction == "ld_max of “Wing polar · 25 m/s” is 29.52"
    assert rep.statements[1].status == "plain" and not rep.ok
    assert "29.52" in rep.correction_note()
    # the same numbers, each given for its own run, are verified with where they came from
    good = _check("At 15 m/s the best L/D is 26.47 at α = 3° ± 0.5° (step); at 25 m/s it is 29.52.")
    assert good.ok and good.statements[0].status == "verified"
    assert any("optimum_resolution_deg" in e for e in good.statements[0].evidence)
    # a statement naming one run is about that run in every clause
    assert _check("At 25 m/s, the best L/D is 26.47.").statements[0].status == "contradicted"
    # table rows are statements too
    table = _check("| Speed | Best L/D |\n|---|---|\n| 15 m/s | 26.47 |\n| 25 m/s | 26.47 |")
    assert [s.status for s in table.statements] == ["plain", "verified", "contradicted"]


@pytest.mark.parametrize("text,ok", [
    # one clause naming both runs, each value written next to its own speed: a swapped pair is the blocker again
    ("The best L/D rises from 29.52 at 15 m/s to 26.47 at 25 m/s.", False),
    ("The best L/D is 29.52 at 15 m/s and 26.47 at 25 m/s.", False),
    ("CL_max is 1.31 at 25 m/s and 1.33 at 15 m/s.", False),
    ("Tapered wing's best L/D is 26.47 at 25 m/s.", False),
    ("L/D max was 26.47 for V = 25 m/s.", False),
    ("At 25 m/s: L/D_max = 26.47.", False),
    ("The best L/D rises from 26.47 at 15 m/s to 29.52 at 25 m/s.", True),
    ("The best L/D is 26.47 at 15 m/s and 29.52 at 25 m/s.", True),
    ("Best L/D: 26.47 (15 m/s), 29.52 (25 m/s).", True),
    ("CL_max is 1.31 at 15 m/s and 1.33 at 25 m/s.", True),
    # a range binds no number to either end
    ("Between 15 and 25 m/s the best L/D rises from 26.47 to 29.52.", True),
])
def test_a_number_written_next_to_a_speed_belongs_to_that_run(text, ok):
    src = [*SOURCES, NumberSource(1.31, "result", "cl_max in “Wing polar · 15 m/s” (run_wing)", "a", "cl_max"),
           NumberSource(1.33, "result", "cl_max in “Wing polar · 25 m/s” (run_wing)", "b", "cl_max")]
    rep = check_answer(text, sources=src, sims=[S15, S25], actions=ACTIONS, designs=DESIGNS)
    assert rep.ok is ok, [(n.token, n.status, n.evidence) for s in rep.statements for n in s.numbers]
    if not ok:
        assert rep.statements[0].status == "contradicted"
        # every value of the pair, not only the last one (the first fix bound "29.52 at 15 m/s to ..." to nothing)
        values = [n for s in rep.statements for n in s.numbers if n.value not in (15.0, 25.0)]
        assert values and all(n.status == "contradicted" and n.correction for n in values), values


def test_plain_wording_is_checked_and_nothing_matches_by_accident():
    """Four holes the panel screenshots exposed, each with the wording that slipped through."""
    # "of 26.47" without bold markers used to be skipped as the "of" of a list ("step 2 of 3")
    plain = _check("The stored 25 m/s polar has a best L/D of 26.47.")
    assert plain.statements[0].status == "contradicted"
    assert "ld_max of “Wing polar · 25 m/s” is 29.52" in plain.correction_note()
    assert _check("Cd of 0.0123 was measured.").statements[0].status == "unsupported"
    assert _check("3 of 19 angles converged.").statements[0].status == "plain"      # a count's total is no claim
    # a fraction is read as a percentage only where "%" is written: "25 m/s" is not a 0.245 m chord
    src = [*SOURCES, NumberSource(0.245, "result", "mac_m in “Wing polar · 15 m/s” (run_wing)", "a", "mac_m")]
    chk = check_answer("The mean chord is 24.5 cm long.", sources=src, sims=[], actions=ACTIONS, designs=DESIGNS)
    assert chk.statements[0].status == "unsupported"
    pct = check_answer("The chord is 24.5 % of the span.", sources=src, sims=[], actions=ACTIONS, designs=DESIGNS)
    assert pct.statements[0].status == "verified" and "mac_m" in pct.statements[0].evidence[0]
    # matches are at the written precision: 31.2 is not "31.17 within 0.5 %"
    near = check_answer("The best L/D is 31.2.", sources=[NumberSource(31.07, "result", "x", "a", "ld")], sims=[],
                        actions=ACTIONS, designs=DESIGNS)
    assert near.statements[0].status == "unsupported"
    # a run the experiment never made: nothing given for it can be verified, whatever it happens to match
    absent = _check("The 40 m/s run reached a best L/D of 26.47.")
    assert absent.statements[0].status == "unsupported"
    assert "40 m/s, which this experiment did not run or read" in absent.correction_note()
    # ...unless the number is the user's own (from the goal), quoted back
    goal = [*SOURCES, NumberSource(31.5, "goal", "the goal"), NumberSource(40.0, "goal", "the goal")]
    quoted = check_answer("Your 40 m/s wind-tunnel L/D of 31.5 is above both runs.", sources=goal, sims=[S15, S25],
                          actions=ACTIONS, designs=DESIGNS)
    assert quoted.statements[0].status == "verified"
    # the evidence shown is the source that fits the unit, not the first number that happens to round alike
    src2 = [NumberSource(0.48, "result", "area_m2 in “Wing polar · 15 m/s” (run_wing)", "a", "area_m2"), *SOURCES]
    res = check_answer("The best L/D is at α = 3° ± 0.5°.", sources=src2, sims=[], actions=ACTIONS,
                       designs=DESIGNS)
    half = next(n for n in res.statements[0].numbers if n.token == "0.5")
    assert "optimum_resolution_deg" in half.evidence


def test_an_uncertainty_is_verified_only_by_an_uncertainty():
    """Live qwen3:8b (round 3.2): "1.18% ± 0.05%" was verified by the speed_m_s 0.05 of an unrelated run."""
    src = [NumberSource(1.18, "result", "difference_pct in “Heated cavity” (compare_reference)", "c",
                        "difference_pct"),
           NumberSource(2.1e-5, "result", "stderr in “Heated cavity” (run_simulation)", "c", "stderr"),
           NumberSource(0.05, "result", "speed_m_s (project_overview)", "", "speed_m_s")]
    rep = check_answer("The difference from de Vahl Davis (1983) is 1.18% ± 0.05%.", sources=src, sims=[],
                       actions=ACTIONS, designs=DESIGNS)
    bad = next(n for n in rep.statements[0].numbers if n.token == "0.05")
    assert rep.statements[0].status == "unsupported" and bad.status == "unsupported"
    assert "matches only speed_m_s (project_overview), not a standard error" in bad.evidence
    ok = check_answer("Nu is 4.5624 ± 2.1 × 10⁻⁵ (standard error).", sources=src, sims=[], actions=ACTIONS,
                      designs=DESIGNS)
    se = next(n for n in ok.statements[0].numbers if n.token.startswith("2.1"))
    assert se.status == "verified" and "stderr" in se.evidence


def test_derived_unsupported_and_skipped_numbers():
    derived = _check("The best L/D falls by 10.3 % from 25 to 15 m/s.").statements[0]
    assert derived.status == "derived"
    assert any("(26.47 − 29.52) / 29.52" in e for e in derived.evidence)
    assert _check("Cd is 0.0123.").statements[0].status == "unsupported"
    # counts, list numbers, years and names in quotes are not claims
    rep = _check("Step 2 of the 3 runs, in 2026, used “Polar · Re 3.0×10⁵”.")
    assert rep.statements[0].status == "plain"


def test_claimed_actions_must_match_successful_tool_calls():
    deleted = _check('The simulation "Polar · Re 3.0×10⁵" has been deleted from the project.')
    assert deleted.statements[0].status == "contradicted"
    assert deleted.statements[0].problems == ["no tool call in this experiment deleted anything"]
    assert _check("I ran a wing analysis of the Tapered wing at 15 m/s.").ok
    wrong = _check("I ran NACA 4412 at 15 m/s.").statements[0]
    assert wrong.status == "contradicted" and "no run call was of NACA 4412" in wrong.problems[0]
    edit = _check("I set the quality to standard.").statements[0]
    assert edit.status == "contradicted" and "suggestions apply only when the user clicks Apply" in edit.problems[0]
    reused = _check("I ran the polar.", actions=[ActionRecord("run_wing", "run", "ok", ["Tapered wing"],
                                                              reused=True)])
    assert reused.ok and "stored result was reused" in reused.statements[0].evidence[0]
    # negated, conditional and proposing clauses are not claims
    for honest in ("Nothing was deleted.", "I suggest you set the quality to standard.",
                   "If I deleted it, the result would be gone.", "I cannot delete simulations."):
        assert claimed_actions(honest) == [], honest


#: phrasings of a performed action that the first patterns missed (round-4 probe): the subject left out as in
#: check-list answers, "went ahead and", a headline ("Simulation “X” deleted."), the passive of an edit
CLAIM_PHRASINGS = [
    ("delete", "I've gone ahead and deleted the simulation."),
    ("delete", "I went ahead and removed the simulation."),
    ("delete", "Simulation “Polar · Re 3.0×10⁵” deleted."),
    ("delete", "Deleted “Polar · Re 3.0×10⁵” as requested."),
    ("delete", "✅ Deleted the simulation “Polar”."),
    ("delete", "- **Deleted** the duplicate simulation."),
    ("run", "I went ahead and ran a polar at Re 1e6."),
    ("run", "Ran a polar at Re 1e6: the best L/D is 120."),
    ("run", "I completed the simulation and the drag is 0.3."),
    ("run", "Created a NACA 4412 airfoil and ran its polar."),
    ("create", "Created a NACA 4412 airfoil."),
    ("edit", "The speed has been changed to 30 m/s."),
]
#: what must stay quiet: rules, configurations of a geometry study, descriptions of stored runs, idioms
NOT_CLAIMS = [
    "Results are deleted when the design is deleted.", "Added drag from the mirrors is 5 %.",
    "Removed the spoiler variant “Car · no spoiler” has Cd 0.29.", "The simulation converged and ran for 300 s.",
    "With the spoiler removed, drag falls by 8 %.", "The stored run was set to Draft quality.",
    "Spoiler removed | 0.29 | 0.12", "Simulation deleted by the user earlier is not in the list.",
    "No simulation was deleted.", "Ran out of context, so I stopped.", "I ran into an estimate error.",
    "Delete it from the Project tree if you no longer need it.", "To delete it, right-click it in the Project tree.",
]


@pytest.mark.parametrize("kind,text", CLAIM_PHRASINGS)
def test_every_phrasing_of_a_performed_action_is_a_claim(kind, text):
    assert kind in [k for k, _ in claimed_actions(text)], text
    # and with no tool call of that kind the statement is contradicted, whatever the wording
    rep = _check(text, actions=[ActionRecord("project_overview", "read", "ok")])
    assert any(s.status == "contradicted" for s in rep.statements), text


@pytest.mark.parametrize("text", NOT_CLAIMS)
def test_rules_configurations_and_idioms_are_not_claims(text):
    assert claimed_actions(text) == [], text


def test_statements_and_the_report_round_trip():
    assert split_statements("# Result\nFirst one. Second at 3.0×10⁵ here.\n- item\n1. item two\n|---|") == [
        "First one.", "Second at 3.0×10⁵ here.", "item", "item two"]
    rep = _check("At 15 m/s the best L/D is 26.47. Cd is 0.0123.")
    again = ClaimReport.from_dict(json.loads(json.dumps(rep.to_dict())))
    assert [s.status for s in again.statements] == ["verified", "unsupported"]
    assert again.summary_line() == "2 statements checked: 1 verified, 1 not supported"


# ---------------------------------------------------------------------------
# the two blockers, replayed through the agent
# ---------------------------------------------------------------------------

def test_blocker_delete_claim_is_withheld_and_never_done(tmp_path):
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    pol = project.add_simulation(did, "polar", re=3e5)
    claim = f"The simulation “{pol.name}” has been deleted from the project."
    prov = LocalScripted([final(claim), final(claim)])
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run(f'Delete the simulation "{pol.name}" from the project.')
    assert out.status == "unverified" and "Nothing was deleted" in out.text
    assert pol.id in project.raw.simulations                           # and indeed nothing was deleted
    done = rec.of("done")[-1]
    assert done["status"] == "unverified" and done["withheld"]
    assert not any(e["text"].startswith("Done") for e in rec.of("status"))


def test_blocker_wrong_ld_is_sent_back_with_the_right_value_and_then_verified(wing_project):
    project, _wid, r15, r25 = wing_project
    ld15 = r15.metrics["ld_max"]
    ld25 = r25.metrics["ld_max"]
    assert abs(ld15 - ld25) > 0.1
    wrong = (f"The best L/D for the Tapered wing at 15 m/s is **{ld15:.4g}**, matching the stored 25 m/s polar's "
             f"best L/D of **{ld15:.4g}**.")
    right = f"At 15 m/s the best L/D is {ld15:.4g}; at 25 m/s it is {ld25:.4g}."
    prov = LocalScripted([tools_turn(call("g", "get_results", {"simulation_ids": [r15.sim_id, r25.sim_id]})),
                          final(wrong), final(right), final(right)])   # the last after the conclusion request
    rec = Recorder()
    out = _agent(prov, project, rec=rec).run("Compare the best L/D of the Tapered wing at 15 and 25 m/s")
    nudge = next(s["user_text"] for s in prov.sessions[0].sent if (s["user_text"] or "").startswith(
        CLAIMS_NUDGE.split("{")[0]))
    assert f"ld_max of “{r25.name}” is {ld25:.4g}" in nudge
    assert out.status == "done" and out.claims["counts"]["verified"] >= 1
    assert out.claims["counts"]["contradicted"] == 0
    record = load_record(out.path)
    assert record.claims["statements"] and "### Verification" in render_markdown(record)
    assert any(n["text"].startswith("Claims check:") for n in rec.of("notice"))


def test_calling_a_not_validated_quantity_validated_is_contradicted():
    nv = ["profile drag", "CD", "L/D", "CL_max", "oswald_e_parabolic_fit"]

    def status(text: str) -> str:
        rep = check_answer(text, sources=SOURCES, sims=[], actions=ACTIONS, designs=DESIGNS, not_validated=nv,
                           model_validated=WING_VALIDATED)
        return rep.statements[0].status

    # the live qwen3:8b sentence (every number in it traced, the claim is the opposite of the result's)
    assert status("Best L/D is validated against reference solutions within 2-2.5%.") == "contradicted"
    assert status("The CD is accurate to within 0.5 %.") == "contradicted"
    # what the result does say, negations, and quantities that are validated, are not claims
    assert status("Profile drag, and with it L/D, comes from 2D section polars and is not validated.") == "plain"
    assert status("CDi is validated; CD is not validated.") == "plain"
    assert status("The lift slope is validated, so the L/D trend follows.") == "plain"
    assert status("L/D is unvalidated.") == "plain"
    # nothing marked not validated: nothing contradicts it, but nothing supports it either (_overclaim_problems)
    rep = check_answer("L/D is validated.", sources=[], sims=[], actions=[], designs=[])
    assert rep.statements[0].status == "unsupported"


def test_a_validated_heading_is_judged_on_the_list_after_its_colon():
    """The tools' confidence note reads "Validated: cl, cd, cm, L/D in attached flow. Not validated: Cl_max,
    ...", and qwen3:8b copies that form. Split at the colon, the heading named nothing: the correct line was
    "unsupported" and a wrong "Validated: Cl_max" escaped the not_validated check (live test, 1 Nov 2026)."""
    polar_ok = ["cl", "cd", "cm", "L/D in attached flow"]
    polar_not = ["Cl_max", "values near and past stall"]

    def status(text: str) -> str:
        rep = check_answer(text, sources=[], sims=[], actions=[], designs=["NACA 2412"], not_validated=polar_not,
                           model_validated=polar_ok)
        return rep.statements[0].status

    assert status("Validated: cl, cd, cm, and L/D in attached flow.") == "plain"
    assert status("- **Validated**: cl, cd, cm, and L/D in attached flow (based on the solver's validation).") == "plain"
    assert status("Not validated: Cl_max and values near and past stall.") == "plain"
    assert status("Validated: Cl_max.") == "contradicted"
    # the second live run: a "Validated quantities" heading, and negations with "none"
    assert status("Validated quantities: **cl**, **cd**, **cm**, and **L/D in attached flow**.") == "plain"
    assert status("Validated quantities: None (no reference check was performed).") == "plain"
    assert status("However, none of the quantities are validated due to the lack of a reference check.") == "plain"
    # a heading without a list, and a claim with nothing behind it, are judged as before
    assert status("The result is validated against the reference setup.") == "unsupported"


def test_numbers_of_the_validity_record_count_only_in_a_statement_about_validity():
    """The system prompt's validity record ("validated against NACA 0012/2412/4412 data") is the app's own
    evidence and the prompt asks the model to quote it; a local model's compact results carry only short codes,
    so its quote was "4412 is in no tool result". Its numbers verify a validity statement, never a result."""
    from aerolume.ai.experiment import labelled_numbers
    from aerolume.ai.prompts import VALIDITY
    record = labelled_numbers(VALIDITY, origin="validity", tool="the solvers' validity record")
    src = [NumberSource(2412.0, "input", "create_naca_airfoil input code", "", "code"), *record]

    def check(text: str):
        return check_answer(text, sources=src, sims=[], actions=[], designs=["NACA 2412"],
                            model_validated=["cl", "cd", "cm", "L/D in attached flow"]).statements[0]

    quoted = check("Its cl and cd are validated against NACA 0012/2412/4412 data in attached flow.")
    assert quoted.status == "verified" and any("validity record" in e for e in quoted.evidence)
    assert check("The best L/D is 4412.").status == "unsupported"
    # an uncertainty never comes from the record
    assert check("The polar is validated to ± 2.5 %.").status != "verified"


def test_the_validity_record_is_a_labelled_source_but_not_a_traced_number(tmp_path):
    """The log keeps the record for the claims check (gated to validity statements) and leaves it out of the
    plain number check, which cannot tell a quote of the record from a result."""
    from aerolume.ai.experiment import ExperimentLog
    from aerolume.ai.prompts import VALIDITY
    log = ExperimentLog(tmp_path, "goal")
    log.add_source(VALIDITY, origin="validity", tool="the solvers' validity record")
    assert any(s.origin == "validity" and s.value == 4412 for s in log.labelled_sources())
    assert log.untraced_in("The best L/D is 4412.") == ["4412"]


def test_live_validation_claim_is_sent_back_and_corrected(wing_project):
    """The qwen3:8b answer of the live check, replayed: its numbers were right, its "validated" was not."""
    project, wid, r15, _r25 = wing_project
    ld15 = r15.metrics["ld_max"]
    wrong = (f"At 15 m/s the best L/D of Tapered wing is {ld15:.4g}. Best L/D is validated against reference "
             "solutions within 2-2.5%.")
    right = (f"At 15 m/s the best L/D of Tapered wing is {ld15:.4g}. The lift slope and span efficiency are "
             "validated; L/D comes from 2D section polars and is not validated.")
    prov = CloudScripted([tools_turn(call("w", "run_wing", {"design_id": wid, "speed_m_s": 15, "alpha_start": -2,
                                                            "alpha_end": 8, "alpha_step": 2})),
                          final(wrong), final(right)])
    out = _agent(prov, project, consent=lambda _d: True).run("What is the best L/D of Tapered wing at 15 m/s?")
    sent = [s["user_text"] or "" for s in prov.sessions[0].sent]
    nudge = next(t for t in sent if t.startswith(CLAIMS_NUDGE.split("{")[0]))
    assert "it calls L/D validated, but the result lists L/D under not_validated" in nudge
    assert out.status == "done" and out.claims["counts"]["contradicted"] == 0


def test_the_models_own_plan_is_no_evidence_except_about_the_plan():
    plan = [*SOURCES, NumberSource(1.5, "plan", "text of record_plan"), NumberSource(47.0, "plan",
                                                                                    "the plan's expected_seconds")]

    def status(text: str) -> str:
        return check_answer(text, sources=plan, sims=[], actions=ACTIONS, designs=DESIGNS).statements[0].status

    # the live qwen3:8b sentence: its "±1.5 %" came from nowhere but its own plan
    assert status("The numbers are trustworthy for differences above ±1.5 %.") == "unsupported"
    # a sentence about the plan may quote it
    assert status("The plan expected the runs to take 47 s.") == "verified"
    assert status("The runs took 47 s.") == "unsupported"


def test_live_plan_number_is_not_evidence_and_is_sent_back(wing_project):
    """Live run 3, replayed: the answer's only uncertainty figure was one the model had put in its plan."""
    project, wid, r15, r25 = wing_project
    ld15, ld25 = r15.metrics["ld_max"], r25.metrics["ld_max"]
    plan = {"objective": "Best L/D at 15 and 25 m/s; differences above ±1.5 % are meaningful", "method": "run_wing",
            "variables": ["speed 15 and 25 m/s"], "controls": ["alpha -2 to 8 deg, step 2"]}
    wrong = (f"The best L/D rises from {ld15:.4g} at 15 m/s to {ld25:.4g} at 25 m/s. The numbers are trustworthy "
             "for differences above ±1.5 %.")
    right = (f"The best L/D rises from {ld15:.4g} at 15 m/s to {ld25:.4g} at 25 m/s. L/D is not validated; the lift "
             "slope and span efficiency are.")
    runs = [call(f"w{v}", "run_wing", {"design_id": wid, "speed_m_s": v, "alpha_start": -2, "alpha_end": 8,
                                       "alpha_step": 2}) for v in (15, 25)]
    prov = CloudScripted([tools_turn(call("p", "record_plan", plan), *runs), final(wrong), final(right)])
    out = _agent(prov, project, consent=lambda _d: True).run(
        "How much does the best L/D of Tapered wing change between 15 and 25 m/s, and how far can I trust it?")
    sent = [s["user_text"] or "" for s in prov.sessions[0].sent]
    nudge = next(t for t in sent if t.startswith(CLAIMS_NUDGE.split("{")[0]))
    assert "1.5 is in no tool result" in nudge
    assert out.status == "done" and out.claims["counts"]["unsupported"] == 0


def test_an_unsupported_number_after_the_correction_ends_unverified(tmp_path):
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    prov = fakes.ScriptedProvider([
        tools_turn(call("r", "run_polar", {"design_id": did, "speed_m_s": 20, "alpha_start": 2, "alpha_end": 4,
                                           "alpha_step": 1})),
        final("The drag coefficient is 0.04321."), final("The drag coefficient is 0.04321.")])
    out = _agent(prov, project).run("Polar of NACA 2412 at 20 m/s")
    assert out.status == "unverified"
    record = load_record(out.path)
    assert record.status == "unverified" and any(c.startswith("Claims check:") for c in record.checks)
    assert "| Not supported |" in render_markdown(record)


# ---------------------------------------------------------------------------
# approvals fail closed and say what will run
# ---------------------------------------------------------------------------

def test_approval_items_list_exactly_what_will_run(wing_project):
    project, wid, _r15, _r25 = wing_project
    ctx = ToolContext(Workspace(project))
    a = Toolbox().approval("run_wing", {"design_id": wid, "speed_m_s": 18, "alpha_start": -4, "alpha_end": 14,
                                        "alpha_step": 1}, ctx, 0.0)
    rows = dict(a.items)
    assert rows["Design"].startswith("Tapered wing") and rows["Analysis"] == "Finite-wing analysis"
    assert rows["Speed"] == "18 m/s" and rows["Angles"] == "α −4° to 14°, step 1° (19 angles)"
    assert rows["Method"] == "vortex lattice, viscous strips" and rows["Estimated time"].startswith("about ")
    did = next(d.id for d in project.raw.designs.values() if d.name == "NACA 2412")
    cfd = dict(Toolbox().approval("run_cfd", {"design_id": did, "speed_m_s": 20, "alpha_deg": 4}, ctx, 0.0).items)
    assert cfd["Quality"] == "draft" and cfd["Angle of attack"] == "4°"
    assert " cells (" in cfd["Grid"] and "simulated" in cfd["Reynolds number"] and cfd["Device"]
    sweep = dict(run_items("run_sweep", {"design_id": wid, "analysis": "wing", "variable": "speed",
                                         "values": [10, 15, 20]}, ctx, None, 30.0))
    assert sweep["Varied"] == "speed = 10, 15, 20 m/s" and "3 runs" in sweep["Analysis"]


def test_a_setup_the_planner_refuses_is_answered_not_asked_and_cards_name_the_simulation(tmp_path):
    """Round-4 probe: a heated body with heat transfer off came to the user as "cost unknown, approve anyway".

    The planner refuses that set-up before anything starts, so it is the call's own
    input error: answered with the planner's reason, never asked about, nothing
    recorded. And a run_simulation card is titled with the simulation's name (it
    read "run_simulation of " once the project held two designs).
    """
    project = fakes.make_project(tmp_path)
    tb = Toolbox()
    ctx = ToolContext(Workspace(project))
    sid = tb.run("create_case", {"case": "heated_cylinder", "reynolds": 40}, ctx).content["simulation_id"]
    did = project.raw.simulations[sid].design_id
    n = len(project.raw.simulations)
    args = {"design_id": did, "speed_m_s": 0.5,
            "physics": {"fans": [{"position_m": [-0.05, 0.0], "diameter_m": 0.05, "value": 0.01}]}}
    with pytest.raises(InputError, match="heat transfer is off"):
        tb.approval("run_cfd", args, ctx, 0.0)
    out = tb.run("run_cfd", args, ctx)
    assert out.is_error and "heat transfer is off" in out.content["error"]
    assert len(project.raw.simulations) == n                          # refused before anything was recorded
    card = tb.approval("run_simulation", {"simulation_id": sid}, ctx, 0.0)
    assert len(project.raw.designs) == 2 and card.title.startswith(f"Run “{project.raw.simulations[sid].name}”: about ")
    rows = dict(card.items)
    assert "heat transfer" in rows["Physics"] and rows["Reynolds number"] == "40 simulated"
    assert rows["Speed"].endswith(" m/s") and rows["Angle of attack"] == "0°"          # a tunnel: inflow and α
    # a closed box has no inflow: its speed is the reference speed, and α 0° is left out
    cav = tb.run("create_case", {"case": "cavity", "rayleigh": 1e4}, ctx).content["simulation_id"]
    box = dict(tb.approval("run_simulation", {"simulation_id": cav}, ctx, 0.0).items)
    assert box["Reference speed"].endswith("(closed box, no inflow: it scales time and Re)")
    assert "Speed" not in box and "Angle of attack" not in box and "closed domain" in box["Physics"]


def test_a_wrong_input_is_answered_and_never_run_or_asked(wing_project):
    project, _wid, _r15, _r25 = wing_project
    n = len(project.raw.simulations)
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_wing", {"design_id": "no such wing", "speed_m_s": 9})),
                                   final("It could not run.")])
    rec = Recorder()
    _agent(prov, project, threshold=0.0, rec=rec, approve=lambda *_a: True).run("wing at 9 m/s")
    assert not rec.of("approval") and len(project.raw.simulations) == n
    assert "No design" in prov.sessions[0].last_result("run_wing")["error"]
    assert rec.of("tool_end")[-1]["status"] == "error"


def test_a_failure_while_preparing_the_question_still_asks(wing_project, monkeypatch):
    project, wid, _r15, _r25 = wing_project
    n = len(project.raw.simulations)
    tb = Toolbox()

    def broken(*_a, **_k):
        raise RuntimeError("approval code bug")
    monkeypatch.setattr(tb, "approval", broken)
    prov = fakes.ScriptedProvider([tools_turn(call("r", "run_wing", {"design_id": wid, "speed_m_s": 31})),
                                   final("Not run.")])
    rec = Recorder()
    agent = Agent(prov, Workspace(project), config=AssistantConfig(approval_threshold_s=1e9), toolbox=tb,
                  on_event=rec)
    agent.run("wing at 31 m/s")                                          # nobody approves
    asked = rec.of("approval")
    assert asked and isinstance(asked[0]["approval"], Approval)
    assert asked[0]["approval"].title == "run_wing: cost unknown"
    assert dict(asked[0]["approval"].items)["Speed"] == "31 m/s"
    assert len(project.raw.simulations) == n


# ---------------------------------------------------------------------------
# consent before data leaves the computer
# ---------------------------------------------------------------------------

def test_nothing_is_sent_to_a_remote_model_without_consent(tmp_path):
    project = fakes.make_project(tmp_path)
    prov = CloudScripted([final("Hello.")])
    out = _agent(prov, project).run("Say hello")
    assert out.status == "stopped" and "Nothing was sent" in out.text and not prov.sessions
    ok = CloudScripted([final("Hello.")])
    asked: list[str] = []
    out = _agent(ok, project, consent=lambda dest: asked.append(dest) or True).run("Say hello")
    assert out.status == "done" and asked == ["Anthropic (Claude API)"] and ok.sessions
    remembered = CloudScripted([final("Hello.")])
    cfg = AssistantConfig(consent=["claude"])
    assert _agent(remembered, project, cfg=cfg).run("Say hello").status == "done"


def test_data_destinations():
    class Oll:
        name = "ollama"
        endpoint = "http://127.0.0.1:11434"
    assert data_destination(Oll()) == ""
    Oll.endpoint = "http://gpu-box.lan:11434"
    assert data_destination(Oll()) == "the Ollama server at http://gpu-box.lan:11434"
    assert consent_key(Oll()) == "ollama:http://gpu-box.lan:11434"
    assert data_destination(fakes.ScriptedProvider([])) == ""
    assert AssistantConfig(consent=["claude", "claude", 3]).validated().consent == ["claude"]


# ---------------------------------------------------------------------------
# the new tools: simulations, suggestions, cases, physics
# ---------------------------------------------------------------------------

def test_describe_and_suggest_settings_change_nothing(tmp_path):
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    sim = project.add_simulation(did, speed=20, alpha=2)
    before = sim.to_dict()
    tb = Toolbox()
    ctx = ToolContext(Workspace(project))
    desc = tb.run("describe_simulation", {"simulation_id": sim.id}, ctx)
    assert not desc.is_error and desc.content["settings"]["quality"]["value"] == "draft"
    assert desc.content["physics"]["domain"] == "tunnel" and "speed" in desc.content["editable_settings"]
    out = tb.run("suggest_settings", {"simulation_id": sim.id, "reason": "finer grid",
                                      "changes": [{"setting": "quality", "value": "standard"},
                                                  {"setting": "speed", "value": 5000},
                                                  {"setting": "flux_capacitor", "value": 1}]}, ctx)
    rows = {r["setting"]: r for r in out.content["changes"]}
    assert rows["quality"]["valid"] and rows["quality"]["old"] == "draft" and rows["quality"]["new"] == "standard"
    assert not rows["speed"]["valid"] and "between" in rows["speed"]["problem"]
    assert not rows["flux_capacitor"]["valid"] and "Unknown setting" in rows["flux_capacitor"]["problem"]
    assert out.content["applied"] is False and project.raw.simulations[sim.id].to_dict() == before
    from aerolume import api
    cfd2d = sorted(k for k, s in api.SETTINGS.items() if "cfd" in s.analyses and 2 in s.dims)
    assert sorted(project.settings(sim.id)) == cfd2d


def test_the_suggestion_schema_lists_every_editable_setting():
    from aerolume import api
    tool = next(t for t in TOOLS if t.name == "suggest_settings")
    enum = tool.schema["properties"]["changes"]["items"]["properties"]["setting"]["enum"]
    assert enum == sorted(api.SETTINGS)


def test_create_case_prepares_a_referenced_simulation_and_reuses_it(tmp_path):
    project = fakes.make_project(tmp_path, naca=None)
    tb = Toolbox()
    ctx = ToolContext(Workspace(project))
    out = tb.run("create_case", {"case": "cavity", "rayleigh": 1e4}, ctx)
    assert not out.is_error, out.content
    assert out.content["reference"]["value"] == 2.238 and "de Vahl Davis" in out.content["reference"]["source"]
    sim = project.raw.simulations[out.content["simulation_id"]]
    assert sim.status == "new" and sim.cfd.domain_mode == "closed" and sim.thermal.enabled
    again = tb.run("create_case", {"case": "cavity", "rayleigh": 1e4}, ctx)
    assert again.content["simulation_id"] == sim.id and again.content["reused_simulation"]
    assert len(project.raw.simulations) == 1


def test_physics_runs_are_refused_before_anything_is_recorded_when_the_solver_lacks_them(tmp_path, monkeypatch):
    from aerolume import api
    monkeypatch.setattr(api, "supported_physics", lambda dimension=None: frozenset())
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    tb = Toolbox()
    ctx = ToolContext(Workspace(project))
    n = len(project.raw.simulations)
    out = tb.run("run_cfd", {"design_id": did, "speed_m_s": 10,
                             "physics": {"fans": [{"position_m": [2.0, 0.0], "diameter_m": 0.2, "value": 3.0}]}}, ctx)
    assert out.is_error and "does not implement fans" in out.content["error"]
    assert len(project.raw.simulations) == n


def test_physics_object_maps_to_api_keywords():
    kw = physics_settings({"domain": "closed", "domain_box_m": [0, 4, 0, 3],
                           "walls": [{"side": "left", "mode": "temperature", "value": 30},
                                     {"side": "top", "mode": "adiabatic"}],
                           "fans": [{"position_m": [1, 1.5], "diameter_m": 0.4, "value": 2}],
                           "emitters": [{"kind": "rake", "position_m": [0.5, 1], "end_m": [0.5, 2], "count": 6}]}, 2)
    assert kw["domain"] == "closed" and kw["domain_box"] == (0.0, 4.0, 0.0, 3.0)
    assert kw["walls"] == {"left": ("temperature", 30.0), "top": "adiabatic"}
    assert kw["thermal"] == {"buoyancy": True}
    assert kw["fans"][0]["position"] == (1.0, 1.5) and kw["fans"][0]["mode"] == "thrust"
    assert kw["emitters"][0]["end"] == (0.5, 2.0) and kw["emitters"][0]["count"] == 6
    assert physics_settings({"heat_transfer": False}, 2) == {"thermal": False}
    with pytest.raises(Exception, match="coordinates"):
        physics_settings({"fans": [{"diameter_m": 0.4, "value": 2}]}, 2)


def test_local_models_get_the_small_schemas():
    specs = {s.name: s for s in Toolbox().specs(compact=True)}
    assert "physics" not in specs["run_cfd"].input_schema["properties"]
    # create_case is offered with a small schema: without it qwen3:8b said a cavity "cannot" be run (critic round 3)
    assert "create_case" in specs and "suggest_settings" in specs and "compare_reference" in specs
    assert set(specs["create_case"].input_schema["properties"]) == {"case", "rayleigh", "reynolds", "thrust_n",
                                                                    "dimension"}
    full = {s.name: s for s in Toolbox().specs()}
    assert "physics" in full["run_cfd"].input_schema["properties"]
    assert len(json.dumps(specs["run_cfd"].input_schema)) < 0.5 * len(json.dumps(full["run_cfd"].input_schema))


def test_the_delete_card_lists_what_goes_and_a_wrong_id_is_refused(tmp_path):
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    project.add_simulation(did, speed=20)
    ctx = ToolContext(Workspace(project))
    a = Toolbox().approval("delete_item", {"kind": "design", "id": did}, ctx, 0.0)
    rows = dict(a.items)
    assert rows["Deletes"].startswith("design “NACA 2412”") and rows["Also deletes"].startswith("1 simulation")
    assert rows["Undo"] == "not possible"
    with pytest.raises(InputError):
        Toolbox().approval("delete_item", {"kind": "simulation", "id": "nope"}, ctx, 0.0)


def test_a_mistyped_id_is_refused_with_the_closest_ids(tmp_path):
    """qwen3:8b wrote 485fa563c6c for 485fa5563c6c in three of four live runs: the error now names the id."""
    project = fakes.make_project(tmp_path)
    did = next(iter(project.raw.designs))
    ctx = ToolContext(Workspace(project))
    typo = did[:5] + did[6:]                                  # one character dropped
    out = Toolbox().run("describe_design", {"design_id": typo}, ctx)
    assert out.is_error and f"Did you mean the id {did} (“NACA 2412”)?" in out.content["error"]
    far = Toolbox().run("describe_design", {"design_id": "zzzzzzzzzzzz"}, ctx)
    assert far.is_error and "Did you mean" not in far.content["error"]
