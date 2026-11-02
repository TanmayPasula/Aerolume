"""Critic round 3.4 fixes of the assistant and the scripting API, each pinned to the evidence.

* the plan card read "cavity at Ra , about 28 s": a 90-character slice cut
  between "Ra" and "1e4" (plaintext.brief keeps the number);
* the answer read "3.3407 ± 0.00095" beside a table showing ± 0.0010
  (plaintext.round_pm, the Results tables' PDG rule, in the stream and the answer);
* "The result is validated against the reference setup" passed the claims check
  after one draft run (claims._overclaim_problems: a reference check inside its
  stated band is needed);
* an enclosure project was offered NACA 4412, Ahmed bodies and a wing as
  examples (assistant_panel.project_examples);
* a mesh imported with ``import_mesh`` showed File "—" and Format "—"
  (api.file_source records name, format, size and SHA-256).
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from aerolume.ai.claims import ActionRecord, check_answer
from aerolume.ai.plaintext import AnswerStream, brief, pdg_round, plain_answer, plain_math, round_pm

EXAMPLES3D = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"

# ---------------------------------------------------------------------------
# plan summaries keep their numbers
# ---------------------------------------------------------------------------

OBJECTIVE = ("Compare the hot-wall mean Nusselt number of the differentially heated square cavity at Ra 1e4 with "
             "de Vahl Davis (1983).")


def test_a_cut_summary_keeps_the_number_after_the_cut():
    assert OBJECTIVE[:90].endswith("at Ra ")                            # the old slice dropped the number
    out = brief(OBJECTIVE, 90)
    assert "Ra 1e4" in out and out.endswith("…") and "  " not in out
    assert len(out) <= 90 + 16 + 1
    # maths first becomes text, then the cut is made between words
    latex = OBJECTIVE.replace("Ra 1e4", "Ra $10^4$")
    assert "Ra 10⁴" in brief(latex, 90) and "$" not in brief(latex, 90)
    # a number with its unit is kept together; a short text is untouched
    assert brief("Measure the drag of the wing at 20 m/s and 30 m/s", 33).endswith("at 20 m/s…")
    assert brief("short goal", 90) == "short goal"


def test_a_number_too_long_to_keep_drops_its_label_too():
    text = "Compare the drag at speed " + "1" * 40 + " exactly"
    out = brief(text, 30)
    assert out == "Compare the drag at…"


def test_the_plan_card_summary_keeps_ra(tmp_path):
    """record_plan's summary, the line the card shows: through the real tool."""
    import test_ai_fakes as fakes

    from aerolume.ai.tools import TOOLS, Toolbox, ToolContext
    from aerolume.ai.workspace import Workspace
    project = fakes.make_project(tmp_path)
    ws = Workspace(project)
    box = Toolbox(TOOLS)
    ctx = ToolContext(ws, None)
    out = box.run("record_plan", {"objective": OBJECTIVE, "method": "CFD of the cavity", "variables": ["Ra"],
                                  "controls": ["grid"], "runs": 1}, ctx)
    assert not out.is_error, out.summary
    assert "Ra 1e4" in out.summary


# ---------------------------------------------------------------------------
# ± rounded by the Results tables' rule
# ---------------------------------------------------------------------------

def test_pdg_round_is_the_results_tables_rule():
    kpi = pytest.importorskip("aerolume.app.widgets.kpi")
    for e in (0.00095, 0.000949, 0.0046, 0.0016, 0.0008, 31.4, 5.71, 13.2, 9.996, 0.1, 1.9e-5, 0.0, float("nan")):
        assert pdg_round(e) == kpi.round_error(e), e


@pytest.mark.parametrize("src, want", [
    ("Nu is **3.3407 ± 0.00095** (standard error).", "Nu is **3.3407 ± 0.0010** (standard error)."),
    ("3.34079 ± 0.00095", "3.3408 ± 0.0010"),
    ("Cd 0.51377 ± 0.004553", "Cd 0.514 ± 0.005"),
    ("184.27 ± 31.4", "184 ± 31"),
    ("640.23 +/- 5.71", "640 +/- 6"),
    ("3.34 ± 0.01", "3.34 ± 0.01"),                    # never padded
    ("2.6988 ± 0.0008", "2.6988 ± 0.0008"),
    ("4.5624 ± 0.000019", "4.5624 ± 0.000019"),
    ("2.5 ± 1.2e-3", "2.5 ± 1.2e-3"),                  # scientific notation is left alone
    ("4.2 ± 3.1 × 10⁻⁴", "4.2 ± 3.1 × 10⁻⁴"),
    ("1.18% ± 0.0532%", "1.18% ± 0.05%"),             # 532: one figure
    ("no uncertainty here: 3.14159", "no uncertainty here: 3.14159"),
])
def test_round_pm(src, want):
    assert round_pm(src) == want
    assert round_pm(want) == want                       # idempotent


def test_the_critic_answer_is_rounded_where_it_is_read():
    answer = ("The mean Nusselt number of the heated cylinder at Re 40 is **3.3407 ± 0.00095** (standard error). "
              "This value is **1.8% higher** than the Lange correlation value of 3.2805.")
    out = plain_answer(answer)
    assert "3.3407 ± 0.0010" in out and "0.00095" not in out and "1.8% higher" in out


@pytest.mark.parametrize("seed", range(60))
def test_streamed_answer_equals_the_whole(seed):
    """Whatever the chunking, the streamed text is exactly plain_answer of the whole (no ± split mid-way)."""
    rng = random.Random(seed)
    texts = ["Nu is **3.34079 ± 0.00095** (standard error), Lange 3.2805 (+1.8 %).",
             "Cd $0.51377 \\pm 0.004553$ and 640.23 +/- 5.71 N; 2.5 ± 1.2e-3 and 4.2 ± 3.1 × 10⁻⁴.",
             "Ends with a value ± 0.00095", "Ends 3.3407", "x ± 0.0046 then $1 \\times 10^5$ end."]
    text = texts[seed % len(texts)]
    ms = AnswerStream()
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 6)
        out.append(ms.feed(text[i:i + n]))
        i += n
    out.append(ms.flush())
    assert "".join(out) == plain_answer(text)


def test_the_stream_releases_prose_promptly():
    ms = AnswerStream()
    assert ms.feed("The mean Nusselt number is ") == "The mean Nusselt number is "
    assert ms.feed("3.3407 ± 0.000") == ""              # the error may still grow
    assert ms.feed("95 (standard error).") == "3.3407 ± 0.0010 (standard error)."
    assert ms.flush() == ""
    assert plain_math("3.3407 ± 0.00095") == "3.3407 ± 0.00095"   # plain_math itself does not round


def test_tool_results_send_the_pdg_rounded_error():
    from aerolume.ai.tools import _stat, value_error_texts
    st = NS(mean=3.34079, stderr=0.00095, stderr_reliable=True)
    assert _stat(st) == {"mean": 3.3408, "stderr": 0.001}
    assert value_error_texts(3.34079, 0.00095) == ("3.3408", "0.0010")


def test_without_qt_the_card_uses_the_pdg_rule(monkeypatch):
    import builtins

    from aerolume.ai import tools as T
    real = builtins.__import__

    def no_qt(name, *a, **k):
        if "widgets.kpi" in name:
            raise ImportError("no Qt in this build")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_qt)
    assert T.value_error_texts(3.34079, 0.00095) == ("3.3408", "0.0010")
    assert T.value_error_texts(0.51377, 0.004553) == ("0.514", "0.005")


# ---------------------------------------------------------------------------
# "validated" needs a reference check inside its stated band
# ---------------------------------------------------------------------------

ACTIONS = [ActionRecord("create_case", "create", "ok"), ActionRecord("run_simulation", "run", "ok"),
           ActionRecord("compare_reference", "read", "ok")]
CRITIC = "The result is validated against the reference setup, and the difference is within the expected range."


def _ref(diff, band=2.0, kind="validation", within=None):
    r = {"applies": True, "kind": kind, "quantity": "nusselt", "measured": 2.2509, "reference": 2.238,
         "difference_pct": diff, "source": "de Vahl Davis (1983), Nu₀: mean Nu of the hot wall"}
    if band is not None:
        r.update(band_pct=band, within_band=abs(diff) <= band if within is None else within)
    return r


def _status(text, refs, model_validated=()):
    rep = check_answer(text, sources=[], sims=[], actions=ACTIONS, designs=[], references=refs,
                       model_validated=list(model_validated))
    return rep.statements[0].status, rep.statements[0].problems


def test_validated_without_a_reference_check_is_not_supported():
    status, problems = _status(CRITIC, [])
    assert status == "unsupported" and "no reference check" in problems[0]


def test_validated_inside_the_band_passes_and_outside_is_contradicted():
    rep = check_answer(CRITIC, sources=[], sims=[], actions=ACTIONS, designs=[], references=[_ref(0.57)])
    st = rep.statements[0]
    assert st.status == "verified" and not st.problems
    assert st.evidence == [("reference check: +0.57 % against de Vahl Davis (1983), Nu₀: mean Nu of the hot wall, "
                            "inside its ± 2 % band")]
    status, problems = _status(CRITIC, [_ref(3.4)])
    assert status == "contradicted" and "+3.4 %" in problems[0] and "± 2 %" in problems[0]
    # no stated band: agreement cannot be called validated
    status, problems = _status(CRITIC, [_ref(0.57, band=None)])
    assert status == "unsupported" and "no tolerance band" in problems[0]
    # a consistency check (the fan room's imposed thrust) validates nothing
    fan = dict(_ref(0.1, band=None, kind="consistency"), quantity="fan thrust")
    assert _status("The thrust is validated.", [fan])[0] == "contradicted"


def test_verified_is_a_claim_too_and_hedges_negations_and_documented_quantities_are_not():
    assert _status("The Nusselt number is verified against de Vahl Davis.", [])[0] == "unsupported"
    assert _status("The Nusselt number is verified against de Vahl Davis.", [_ref(0.57)])[0] == "verified"
    for text in ("The absolute value must be validated by a grid study.",
                 "The result is not validated.",
                 "The discretisation error needs a grid study for further validation.",
                 "All numbers are verified against the tool results."):
        assert _status(text, [])[0] == "plain", text
    # a quantity the result lists as validated from docs/validation.md (a polar's cl, a wing's lift slope)
    assert _status("The lift slope is validated.", [], ["CL_alpha"])[0] == "plain"
    # a quantity no reference check covers, while the Nusselt check exists
    status, problems = _status("The drag is validated.", [_ref(0.57)])
    assert status == "unsupported" and "drag" in problems[0]


def test_the_cavity_case_states_its_band_and_the_tool_passes_it_on():
    from aerolume import api
    from aerolume.ai.tools import physics_validated
    case = api.cavity_case(1e4)
    assert case.reference["band_pct"] == 2.0 and "64²" in case.reference["band_source"]
    assert api.cavity_case(1e6).reference["band_pct"] == 5.0
    assert "band_pct" not in api.cavity_case(1e5, dimension=3).reference       # no measured band in 3D at 1e5
    assert api.heated_cylinder_case(40).reference["band_pct"] == 4.0
    assert "band_pct" not in api.heated_cylinder_case(100).reference          # checked at Re 20 and 40 only
    # a stored run finds the same band again (reference_case rebuilds the reference from the design)
    cyl = api.heated_cylinder_case(40)
    from aerolume.model import Simulation
    sim = Simulation(name="cyl", design_id=cyl.design.id, analysis="cfd", thermal=cyl.settings["thermal"])
    sim.conditions.fluid = cyl.settings["fluid"]
    again, why = api.reference_case(cyl.design, sim, re=40.0)
    assert again is not None and again.reference["band_pct"] == 4.0, why
    entry = physics_validated({"applies": True, "kind": "validation", "quantity": "nusselt", "measured": 2.2509,
                               "reference": 2.238, "difference_pct": 0.575, "band_pct": 2.0, "within_band": True,
                               "source": "de Vahl Davis"})
    assert entry and entry[0].endswith("inside its ± 2 % band")


def test_the_agent_collects_reference_checks_for_the_claims_check():
    from aerolume.ai.agent import Agent
    agent = Agent.__new__(Agent)
    agent._not_validated, agent._steady, agent._validated, agent._model_validated, agent._references = \
        [], [], [], [], []
    agent.observe({"checks": [dict(_ref(0.57), simulation_id="s1")],
                   "validated_quantities": ["cl", "Nusselt number: validated set-up (docs/validation.md); ..."]})
    agent.observe({"reference_check": dict(_ref(0.57), simulation_id="s1")})      # the same check again
    assert len(agent._references) == 1 and agent._model_validated == ["cl"]
    assert agent._validated and agent._validated[0].startswith("Nusselt number")


# ---------------------------------------------------------------------------
# example goals come from the project
# ---------------------------------------------------------------------------

def _sim(name, did, *, thermal=False, fans=(), speed=1.0):
    return NS(name=name, design_id=did, analysis="cfd", thermal=NS(enabled=thermal), fans=list(fans),
              conditions=NS(speed=speed))


def test_an_enclosure_project_gets_only_its_own_examples():
    pytest.importorskip("PySide6")
    from aerolume.app.widgets.assistant_panel import EXAMPLES, project_examples
    enclosure = NS(name="Enclosure", kind="generic", dimension=2, source=None,
                   bodies=[NS(thermal="temperature")], meshes=[])
    project = NS(designs={"e": enclosure},
                 simulations={"1": _sim("Enclosure · fans", "e", thermal=True, fans=[NS(enabled=True)])})
    ex = project_examples(project, 3)
    assert len(ex) == 3 and not set(ex) & set(EXAMPLES)
    assert all("Enclosure" in e for e in ex)
    assert not any(w in e for e in ex for w in ("NACA", "Ahmed", "aspect ratio", "wing"))
    assert any("thrust" in e for e in ex) and any("Nusselt" in e and "heat rate" in e for e in ex)
    assert any("grid study" in e for e in ex)
    # the selected design's example is not repeated in the slots
    assert ex[0] not in project_examples(project, 3, (ex[0],))
    # a project without designs gets the generic ones; one with a lone airfoil gets its own questions only
    assert project_examples(NS(designs={}, simulations={}), 3) == list(EXAMPLES)
    foil = NS(designs={"a": NS(name="NACA 0012", kind="airfoil", dimension=2, bodies=[], meshes=[])},
              simulations={})
    ex = project_examples(foil, 3)
    assert len(ex) == 3 and all("NACA 0012" in e for e in ex)


# ---------------------------------------------------------------------------
# an imported mesh keeps its provenance
# ---------------------------------------------------------------------------

def test_import_mesh_records_file_format_and_sha256(tmp_path, monkeypatch):
    import aerolume as al
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    src = EXAMPLES3D / "sphere_d100mm_binary.stl"
    proj = al.new_project(tmp_path / "Prov")
    d = proj.import_mesh(src, kind="generic")
    s = d.raw.source
    assert s["type"] == "import" and s["file"] == src.name and s["format"] == "stl"
    assert s["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest() and s["bytes"] == src.stat().st_size
    assert Path(s["path"]) == src.resolve()
    # the Mesh inspector reads exactly these keys (file, path; the format from the file's extension)
    pytest.importorskip("PySide6")
    from aerolume.app.widgets.mesh_inspector import FORMAT3
    assert FORMAT3.get(Path(s["file"]).suffix.lower())
