"""validation/run_all.py: the registry, the checks, the reports and the command line.

The physics of each case is asserted by the slow validation tests and by the run
itself; these tests make sure the runner reports honestly: a failing check fails the
run, a missing feature is "unavailable" (never a pass), every check cites a
reference, the generated documentation replaces only its own section, and two
real CPU cases (2D Taylor-Green verification, finite-wing theory) pass.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ra():
    spec = importlib.util.spec_from_file_location("aerolume_run_all_test", ROOT / "validation" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_registry_is_complete_and_cited(ra):
    keys = [c.key for c in ra.CASES]
    assert len(keys) == len(set(keys))
    for c in ra.CASES:
        assert c.area in ra.AREAS, c.key
        assert c.sources and all(s in ra.REFERENCES for s in c.sources), c.key
        assert callable(c.run) and c.est_seconds > 0
    quick_areas = {c.area for c in ra.CASES if c.quick}
    assert quick_areas == set(ra.AREAS)                          # --quick touches every area
    assert {"cavity2d", "heated_cylinder", "actuator2d", "cyl40", "sphere100", "ladson",
            "wing_theory"} <= {c.key for c in ra.CASES if c.quick}


def test_check_bands_and_deviation(ra):
    c = ra.Check("x", 1.04, *ra.band(1.0, 0.05), "exact", ref=1.0)
    assert c.passed and c.deviation == pytest.approx(0.04)
    assert not ra.Check("x", 1.06, *ra.band(1.0, 0.05), "exact", ref=1.0).passed
    assert not ra.Check("x", float("nan"), 0.0, 1.0, "exact").passed
    assert ra.band(-2.0, 0.1) == pytest.approx((-2.2, -1.8))
    assert ra.widen(5.57, 5.59, 0.01) == pytest.approx((5.57 * 0.99, 5.59 * 1.01))
    d = ra.Check("q", 2.0, 1.0, 3.0, "exact").to_dict()
    assert d["passed"] is True and d["deviation"] is None


def test_select(ra):
    cases = ra.CASES
    quick = ra.select(cases, quick=True)
    assert all(c.quick for c in quick) and not any(c.stretch for c in quick)
    full = ra.select(cases, quick=False)
    assert len(full) > len(quick) and not any(c.stretch for c in full)
    assert any(c.stretch for c in ra.select(cases, quick=False, stretch=True))
    assert [c.key for c in ra.select(cases, quick=True, names=["cavity"])] == ["cavity2d", "cavity3d"]
    assert [c.key for c in ra.select(cases, quick=True, names=["cavity2d"])] == ["cavity2d"]
    assert [c.key for c in ra.select(cases, quick=True, names=["sphere"])] == ["sphere100", "sphere_symmetry",
                                                                             "sphere300"]
    assert {c.area for c in ra.select(cases, quick=False, areas=["wing"])} == {"wing"}
    with pytest.raises(KeyError, match="nope"):
        ra.select(cases, quick=True, names=["nope"])


def _fake_cases(ra):
    def good(ctx):
        return [ra.Check("good quantity", 1.0, 0.9, 1.1, "exact", ref=1.0, ref_text="1", tolerance="± 10 %")]

    def bad(ctx):
        return [ra.Check("bad quantity", 2.0, 0.9, 1.1, "devahl", ref=1.0)], {"cavity": [{"ra": 1e3, "nu": 1.12}]}

    def missing(ctx):
        raise ra.Unavailable("feature X is not in this version")

    def broken(ctx):
        raise ZeroDivisionError("boom")
    return [ra.Case("fake_good", "Good", "verification", good, ("exact",), quick=True),
            ra.Case("fake_bad", "Bad", "thermal", bad, ("devahl",), quick=True),
            ra.Case("fake_missing", "Missing", "fans", missing, ("froude",), quick=True),
            ra.Case("fake_broken", "Broken", "flow2d", broken, ("exact",), quick=True)]


def test_run_case_statuses(ra):
    ctx = ra.Context(device="cpu", quick=True)
    out = {c.key: ra.run_case(c, ctx) for c in _fake_cases(ra)}
    assert out["fake_good"].status == "pass"
    assert out["fake_bad"].status == "fail" and out["fake_bad"].data["cavity"][0]["nu"] == 1.12
    assert out["fake_missing"].status == "unavailable" and "feature X" in out["fake_missing"].error
    assert out["fake_broken"].status == "error" and "ZeroDivisionError" in out["fake_broken"].error
    json.dumps([r.to_dict() for r in out.values()], allow_nan=False)


def test_markdown_and_docs_section(ra, tmp_path):
    ctx = ra.Context(device="cpu", quick=True)
    results = [ra.run_case(c, ctx) for c in _fake_cases(ra)]
    meta = {"date": "2026-10-18 12:00", "mode": "quick", "wall_minutes": 0.1, "gpu": "Test GPU", "os": "Test OS",
            "python": "3.11", "torch": "2.x", "cuda": "13", "aerolume": "0.3"}
    md = ra.markdown(results, meta)
    assert md.startswith(ra.SECTION_TITLE) and ra.SECTION_TITLE.startswith("## 9. ")
    assert "(**1 failed**)" in md                                            # failures are counted up front
    assert "| Verification (exact answers) | Good | 1 / 1 | pass |" in md
    assert "**FAIL**" in md and "unavailable: feature X" in md and "**ERROR**" in md
    assert "[devahl](#ref-devahl)" in md and "de Vahl Davis" in md          # every check cites its source
    doc = tmp_path / "validation.md"
    doc.write_text("# Validation report\n\nHand-written text.\n\n## 11. Later section\n", encoding="utf-8")
    ra.write_docs(md, doc)
    first = doc.read_text(encoding="utf-8")
    assert first.count(ra.BEGIN) == 1 and "Hand-written text." in first and "## 11. Later section" in first
    ra.write_docs(md.replace("Good", "Fine"), doc)                         # replaced in place, not appended
    second = doc.read_text(encoding="utf-8")
    assert second.count(ra.BEGIN) == 1 and "| Fine |" in second and "| Good |" not in second
    assert second.startswith("# Validation report\n\nHand-written text.")
    assert ra.figure_deviations(results, tmp_path / "dev.png").stat().st_size > 5000
    assert ra.figure_cavity(results, tmp_path / "cav.png") is None           # only the real cavity case
    cav = ra.CaseResult("cavity2d", "Cavity", "thermal", "pass",
                        data={"cavity": [{"ra": 1e3, "nu": 1.12, "nu_ref": 1.117}, {"ra": 1e4, "nu": 2.25,
                                                                                    "nu_ref": 2.238}]})
    assert ra.figure_cavity([cav], tmp_path / "cav.png").stat().st_size > 5000


def test_main_exit_codes_and_outputs(ra, tmp_path, monkeypatch, capsys):
    fakes = _fake_cases(ra)
    monkeypatch.setattr(ra, "CASES", fakes)
    assert ra.main(["--list"]) == 0
    assert "fake_missing" in capsys.readouterr().out
    out = tmp_path / "out"
    doc = tmp_path / "v.md"
    common = ["--out", str(out), "--device", "cpu"]
    assert ra.main(["--cases", "fake_good,fake_missing", *common]) == 0       # unavailable is not a failure
    assert ra.main(["--cases", "fake_good,fake_missing", "--strict", *common]) == 1
    assert ra.main(["--cases", "fake_bad", *common, "--write-docs", "--docs", str(doc), "--img",
                    str(tmp_path / "img"), "--record", str(tmp_path / "kept.json")]) == 1
    assert ra.BEGIN in doc.read_text(encoding="utf-8")
    kept = json.loads((tmp_path / "kept.json").read_text(encoding="utf-8"))   # the record behind the docs
    assert kept["results"][0]["key"] == "fake_bad" and kept["meta"]["source"]["sha256"]
    assert (tmp_path / "img" / "validation_run_all.png").is_file()
    records = sorted(out.glob("validation-quick-*.json"))
    assert records and json.loads(records[-1].read_text(encoding="utf-8"))["results"][0]["key"] == "fake_bad"
    assert ra.main(["--cases", "nope", *common]) == 2
    assert ra.main(["--bogus-flag"]) == 2


def test_discover_plugin_cases(ra, tmp_path, monkeypatch):
    plug = tmp_path / "extra_cases.py"
    plug.write_text(
        "import importlib, sys\n"
        "ra = sys.modules['aerolume_run_all_test']\n"
        "def run(ctx):\n"
        "    return [ra.Check('plug', 1.0, 0.0, 2.0, 'exact')]\n"
        "RUN_ALL_CASES = [ra.Case('plugin_case', 'Plug-in', 'thermal', run, ('exact',), quick=True)]\n",
        encoding="utf-8")
    (tmp_path / "broken_cases.py").write_text("RUN_ALL_CASES = [\nraise\n", encoding="utf-8")
    monkeypatch.setattr(ra, "HERE", tmp_path)
    found = ra.discover(list(ra.CASES))
    assert "plugin_case" in [c.key for c in found]
    assert len(found) == len(ra.CASES) + 1                                  # the broken one is skipped


def test_confined_disc_theory_both_signs(ra):
    """The signed streamtube model reproduces Froude for B -> 0 (propeller and turbine), the case
    module's turbine branch at finite blockage, and confinement pushes the disc velocity the right way."""
    for ct in (0.2, 0.5, 0.8):
        a_prop = 0.5 * (math.sqrt(1.0 + ct) - 1.0)                          # C_T = 4 a (1 + a)
        a_turb = 0.5 * (1.0 - math.sqrt(1.0 - ct))                          # C_T = 4 a (1 - a)
        assert ra.confined_disc_theory(ct, 1e-6)["u_t"] == pytest.approx(1.0 + a_prop, rel=1e-5)
        assert ra.confined_disc_theory(-ct, 1e-6)["u_t"] == pytest.approx(1.0 - a_turb, rel=1e-5)
        assert ra.froude_disc(ct)["u_disc"] == pytest.approx(1.0 + a_prop)
        assert ra.froude_disc(-ct)["u_wake"] == pytest.approx(1.0 - 2.0 * a_turb)
    # a turbine in a channel slows the flow less than in open air, a propeller accelerates it less
    assert ra.confined_disc_theory(-0.5, 0.1)["u_t"] > ra.froude_disc(-0.5)["u_disc"]
    assert ra.confined_disc_theory(0.5, 0.1)["u_t"] < ra.froude_disc(0.5)["u_disc"]
    spec = importlib.util.spec_from_file_location("aerolume_t2d_theory", ROOT / "validation" / "thermal2d_cases.py")
    if spec is not None and (ROOT / "validation" / "thermal2d_cases.py").is_file():
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001 - the case module is another area's; its absence is not ours to test
            mod = None
        if mod is not None and hasattr(mod, "channel_disc_theory"):
            for b in (0.05, 0.1, 0.2):
                assert ra.confined_disc_theory(-0.5, b)["u_t"] == pytest.approx(
                    mod.channel_disc_theory(0.5, b)["u_t"], rel=1e-9)
    with pytest.raises(ValueError):
        ra.confined_disc_theory(0.5, 0.0)


def test_fan_and_jet_cases_run_what_their_labels_say(ra):
    """The 2D actuator case runs both directions and compares each with its own branch of the
    momentum theory (a turbine checked against the propeller theory once failed a correct
    solver), and the jet's labels name the span the case module actually measured."""
    from types import SimpleNamespace
    calls: list[dict] = []

    def run_actuator_disc(ct, cpd, *, propeller=False, device="auto"):
        calls.append({"ct": ct, "cpd": cpd, "propeller": propeller})
        u = ra.confined_disc_theory(ct if propeller else -ct, 0.1)["u_t"] * 1.004     # 0.4 % off: inside ± 3 %
        return SimpleNamespace(values={"u_disc": u, "blockage": 0.1, "thrust": 0.5 * ct}, grid=(8, 4), steps=1,
                               wall_s=0.1)

    def run_plane_jet(*, device="auto", x_fit=(12.0, 40.0)):
        calls.append({"x_fit": tuple(x_fit)})
        return SimpleNamespace(values={"slope": 1.0, "slope_bickley": 1.02, "K_drift_pct": 6.2, "K_mean": 0.9,
                                       "Kp_mean": -0.01}, grid=(8, 8), steps=1, wall_s=0.1)

    ctx = ra.Context(device="cpu", quick=True)
    ctx._modules["thermal2d_cases"] = SimpleNamespace(run_actuator_disc=run_actuator_disc, run_plane_jet=run_plane_jet)
    checks, data = ra.case_actuator2d(ctx)
    assert sorted(c["propeller"] for c in calls) == [False, True]
    assert all(c.passed for c in checks) and len(checks) == 4
    assert set(data) == {"propeller", "turbine"}
    assert data["turbine"]["u_disc_theory"] < 1.0 < data["propeller"]["u_disc_theory"]
    assert any("turbine" in c.quantity for c in checks) and any("propeller" in c.quantity for c in checks)
    calls.clear()
    checks, data = ra.case_plane_jet(ctx)
    assert calls == [{"x_fit": ra.JET_SPAN}] and data["span"] == list(ra.JET_SPAN)
    span = f"{ra.JET_SPAN[0]:g} ≤ x/b ≤ {ra.JET_SPAN[1]:g}"
    assert len(checks) == 2 and all(span in c.quantity for c in checks)      # no budget, no half-width checks
    drift = checks[1]
    # the band written before the first run (5 %), not one widened after it: the closed box's 6.2 % fails
    assert not drift.passed and drift.value == pytest.approx(0.062) and drift.hi == ra.JET_DRIFT_BAND == 0.05
    assert "-1.1 % of K" in drift.note


def test_plane_jet_holds_the_momentum_budget_not_the_raw_drift(ra):
    """The conservation statement is the core's momentum budget (residual < 1 % of K); the raw
    drift of K keeps its pre-registered 5 % band and fails visibly (the closed box drains it)."""
    from types import SimpleNamespace

    def run_plane_jet(*, device="auto", x_fit=(12.0, 40.0)):
        return SimpleNamespace(values={"slope": 1.0, "slope_bickley": 1.02, "K_drift_pct": 6.2, "K_mean": 0.9,
                                       "Kp_mean": -0.01, "K_budget_residual_pct": -0.27,
                                       "K_budget_shear_pct": -3.6, "K_budget_pressure_pct": -2.5},
                               grid=(8, 8), steps=1, wall_s=0.1)

    ctx = ra.Context(device="cpu", quick=True)
    ctx._modules["thermal2d_cases"] = SimpleNamespace(run_plane_jet=run_plane_jet)
    checks, _ = ra.case_plane_jet(ctx)
    assert len(checks) == 3 and [c.passed for c in checks] == [True, True, False]
    assert "fails until a free jet is validated" in checks[2].note
    budget = next(c for c in checks if "budget" in c.quantity)
    assert budget.value == pytest.approx(0.0027) and budget.hi == 0.01
    assert "-3.60 % is the return flow's shear" in budget.note
    leaky = {"K_budget_residual_pct": 1.5}

    def run_leaky(*, device="auto", x_fit=(12.0, 40.0)):
        r = run_plane_jet(device=device, x_fit=x_fit)
        r.values.update(leaky)
        return r

    ctx._modules["thermal2d_cases"] = SimpleNamespace(run_plane_jet=run_leaky)
    checks, _ = ra.case_plane_jet(ctx)
    assert not next(c for c in checks if "budget" in c.quantity).passed      # a scheme that leaks fails


def test_actuator3d_runs_both_branches_and_jet3d_checks_the_budget(ra):
    """The 3D disc runs as a propeller (C_T 0.5) and a turbine (C_T 0.4), each against its own
    theory branch; the round jet is held to its budget residual, the k0 decay law and the shape."""
    from types import SimpleNamespace
    calls: list[tuple] = []
    blockage = (ra.math.pi / 4.0) / 36.0

    def run_actuator_disc(ct, cpd, *, propeller=True, device="auto"):
        calls.append((ct, propeller))
        u = ra.confined_disc_theory(ct if propeller else -ct, blockage)["u_t"] * 0.995
        return SimpleNamespace(values={"u_disc": u, "ct": ct, "u_axis_2d": 1.1, "u_axis_4d": 1.1}, grid=(8, 4, 4),
                               steps=1, wall_s=0.1)

    def run_round_jet(re_j, cpd, *, device="auto"):
        calls.append(("jet", re_j, cpd))
        return SimpleNamespace(values={"budget_residual_pct": -0.12, "budget_dK_pct": -5.0,
                                       "budget_shear_pct": -3.0, "budget_pressure_pct": -1.9,
                                       "inv_uc_err_k0": 0.01, "shape_rms": 0.008, "notes": 0},
                               grid=(128, 97, 97), steps=10, wall_s=1.0)

    ctx = ra.Context(device="cpu", quick=True)
    ctx._modules["thermal3d_cases"] = SimpleNamespace(run_actuator_disc=run_actuator_disc,
                                                      run_round_jet=run_round_jet)
    checks, data = ra.case_actuator3d(ctx)
    assert calls == [(0.5, True), (0.4, False)] and set(data) == {"propeller", "turbine"}
    assert len(checks) == 4 and all(c.passed for c in checks)
    assert data["turbine"]["u_disc_theory"] < 1.0 < data["propeller"]["u_disc_theory"]
    calls.clear()
    checks, data = ra.case_jet3d(ctx)
    assert calls == [("jet", 100.0, 4)] and len(checks) == 3 and all(c.passed for c in checks)
    assert "jet3d" in {c.key for c in ra.CASES} and "schlichting" in ra.REFERENCES


def test_published_values_in_either_table_layout(ra):
    """The cubic cavity's references, one value per quantity or one per source (with positions)."""
    old = {"nu": 2.302, "u_max": 0.2013}
    new = {"nu_tric": 2.25, "nu_fusegi": 2.302, "u_max": {"wakashima": (0.198, 0.825), "fusegi": (0.2013, 0.817)}}
    assert ra._published(old, "nu") == [2.302] and ra._published(old, "u_max") == [0.2013]
    assert sorted(ra._published(new, "nu")) == [2.25, 2.302]
    assert sorted(ra._published(new, "u_max")) == [0.198, 0.2013]
    assert ra._published(new, "w_max") == []
    assert ra._pow10(1e4) == "10⁴" and ra._pow10(1500.0) == "1500"


def test_numpy_values_serialise_and_pipes_are_escaped(ra):
    """A band built from NumPy reference data must not lose the run's record at the end
    (numpy.bool_ is not JSON), and |∇·u| in a quantity must not break the Markdown table."""
    import numpy as np
    c = ra.Check("max |∇·u| h/U", np.float64(1.0), np.float64(0.5), np.float64(1.5), "exact",
                 ref=np.float64(1.0))
    assert type(c.passed) is bool
    json.dumps(c.to_dict(), allow_nan=False)
    res = ra.CaseResult("x", "X", "verification", "pass", [c], data={"a": np.arange(3), "b": np.bool_(True)})
    json.dumps(ra._jsonable(res.to_dict()), allow_nan=False)
    md = ra.markdown([res], {"date": "d", "mode": "quick"})
    row = next(line for line in md.splitlines() if "h/U" in line)
    assert r"max \|∇·u\| h/U" in row and row.count("|") - row.count(r"\|") == 7     # six cells
    nan_check = ra.Check("q", float("nan"), 0.0, 1.0, "exact")
    assert ra._value_text(nan_check) == "NaN"


def test_digits_resolve_narrow_bands(ra):
    wide = ra.Check("w", 1.12, 1.09, 1.14, "exact")
    narrow = ra.Check("n", 1.117793, 1.1165, 1.1185, "devahl")
    assert ra._digits(wide) == 4 and ra._digits(narrow) >= 5
    assert "1.1178" in ra._fmt(narrow.value, ra._digits(narrow))
    assert ra._digits(ra.Check("z", 0.0, 0.0, 0.0, "exact")) == 4


def test_records_merge_and_document(ra, tmp_path, monkeypatch, capsys):
    """--from-json writes the documentation from earlier records: later runs of a case win,
    run times add up, a failure in any record fails the report."""
    fakes = _fake_cases(ra)
    monkeypatch.setattr(ra, "CASES", fakes)
    out = tmp_path / "out"
    common = ["--out", str(out), "--device", "cpu"]
    assert ra.main(["--cases", "fake_good", *common]) == 0
    first = sorted(out.glob("*.json"))
    assert ra.main(["--cases", "fake_bad", *common, "--all"]) == 1
    records = sorted(out.glob("*.json"), key=lambda p: p.stat().st_mtime)
    assert len(records) == 2 and records[0] == first[0]
    results, meta = ra.load_records([str(p) for p in records])
    assert [r.key for r in results] == ["fake_good", "fake_bad"]            # registry order
    assert meta["mode"] == "quick + all" and meta["wall_minutes"] >= 0
    assert results[1].checks[0].source == "devahl" and not results[1].checks[0].passed
    doc, img = tmp_path / "v.md", tmp_path / "img"
    code = ra.main(["--from-json", *map(str, records), "--write-docs", "--docs", str(doc), "--img", str(img),
                    "--record", str(tmp_path / "kept.json")])
    assert code == 1
    assert (tmp_path / "kept.json").is_file()
    text = doc.read_text(encoding="utf-8")
    assert "| Good |" in text and "| Bad |" in text and "quick + all" in text
    assert ra.main(["--from-json", str(tmp_path / "missing.json")]) == 2
    capsys.readouterr()


def test_gci_figure_from_case_data(ra, tmp_path):
    data = {"gci": {"ra": 1e3, "n": [128, 64, 32, 16], "nu": [1.11789, 1.11837, 1.12015, 1.12745], "ref": 1.117,
                    "ref_spread": [1.117, 1.118, 1.118], "p": 2.036, "phi_ext": 1.11779, "u_num": 0.00072}}
    res = ra.CaseResult("gci_cavity", "GCI", "thermal", "pass", data=data)
    assert ra.figure_gci([res], tmp_path / "gci.png").stat().st_size > 5000
    assert ra.figure_gci([ra.CaseResult("other", "x", "thermal", "pass")], tmp_path / "none.png") is None


def test_real_cpu_cases_pass(ra):
    ctx = ra.Context(device="cpu", quick=True)
    for key in ("tg2d", "wing_theory"):
        case = next(c for c in ra.CASES if c.key == key)
        res = ra.run_case(case, ctx)
        assert res.status == "pass", (key, res.error, [(c.quantity, c.value, c.lo, c.hi) for c in res.checks])


def test_source_digest_tracks_the_measured_code(ra, tmp_path):
    """The digest changes with a solver or a band, ignores line endings and code outside the scope."""
    (tmp_path / "src" / "aerolume" / "solvers").mkdir(parents=True)
    (tmp_path / "src" / "aerolume" / "app").mkdir(parents=True)
    (tmp_path / "validation").mkdir()
    solver = tmp_path / "src" / "aerolume" / "solvers" / "core.py"
    solver.write_bytes(b"x = 1\n")
    (tmp_path / "validation" / "case.py").write_bytes(b"BAND = 0.05\n")
    scope = ("src/aerolume/solvers", "validation")
    d0 = ra.source_digest(tmp_path, scope)
    assert d0["files"] == 2 and len(d0["sha256"]) == 64
    solver.write_bytes(b"x = 1\r\n")                                     # a Windows checkout: same code
    assert ra.source_digest(tmp_path, scope)["sha256"] == d0["sha256"]
    (tmp_path / "src" / "aerolume" / "app" / "page.py").write_text("ui = 2\n")   # outside the scope
    assert ra.source_digest(tmp_path, scope)["sha256"] == d0["sha256"]
    (tmp_path / "validation" / "case.py").write_bytes(b"BAND = 0.08\n")   # a widened band is a change
    assert ra.source_digest(tmp_path, scope)["sha256"] != d0["sha256"]
    # the default scope covers the solvers and the cases but not the GUI or the report tables
    assert "src/aerolume/solvers" in ra.SOURCE_SCOPE and "validation" in ra.SOURCE_SCOPE
    assert not any(x.startswith(("src/aerolume/app", "src/aerolume/report")) for x in ra.SOURCE_SCOPE)


def test_check_docs_says_when_section_9_is_stale(ra, tmp_path, monkeypatch, capsys):
    now = ra.source_digest()["sha256"]
    meta = {"date": "2026-10-30 12:00", "mode": "all", "wall_minutes": 1.0, "aerolume": "0.3.0",
            "source": {"sha256": now, "files": 80, "scope": list(ra.SOURCE_SCOPE)}}
    res = [ra.CaseResult("x", "X", "verification", "pass", [ra.Check("q", 1.0, 0.9, 1.1, "exact", ref=1.0)])]
    doc = tmp_path / "v.md"
    ra.write_docs(ra.markdown(res, meta), doc)
    text = doc.read_text(encoding="utf-8")
    assert f"`{now[:ra.DIGEST_CHARS]}`" in text and "Aerolume 0.3.0" in text and ra.RECORD_REL in text
    assert ra.check_docs(doc) == 0 and "up to date" in capsys.readouterr().out
    meta["source"]["sha256"] = "0" * 64
    ra.write_docs(ra.markdown(res, meta), doc)
    assert ra.check_docs(doc) == 1 and "STALE" in capsys.readouterr().out
    assert ra.main(["--check-docs", "--docs", str(doc)]) == 1
    doc.write_text("# Validation report\n", encoding="utf-8")
    assert ra.check_docs(doc) == 1                                         # no generated section at all


def test_records_of_different_code_are_marked_mixed(ra, tmp_path):
    recs = []
    for i, digest in enumerate(("a" * 64, "b" * 64)):
        rec = {"meta": {"date": f"2026-10-30 1{i}:00", "mode": "all", "source": {"sha256": digest, "files": 1}},
               "results": [{"key": f"k{i}", "title": "T", "area": "verification", "status": "pass", "checks": []}]}
        path = tmp_path / f"r{i}.json"
        path.write_text(json.dumps(rec), encoding="utf-8")
        recs.append(path)
    _, meta = ra.load_records(recs)
    assert meta["source_mixed"] == ["a" * 64, "b" * 64]
    assert "measured on different code" in ra.markdown([], meta)
    _, same = ra.load_records([recs[0], recs[0]])
    assert "source_mixed" not in same


def test_package_version_is_the_imported_one(ra):
    """An editable install reports the version it was installed with (0.1.0 for months while the
    package said 0.3.0): the record must carry the imported package's own version."""
    import aerolume
    assert ra.package_version() == aerolume.__version__
    assert ra.environment("cpu")["aerolume"] == aerolume.__version__


def test_square_band_uses_the_rule_of_the_other_2d_bands(ra):
    """In early development the square cylinder's drag band was the published range +8 %, written after the +5.9 %
    deviation at 32 cells/D was known. It is now range ± 5 %, measured on 32 and 48 cells/D."""
    class FakeRun:
        def __init__(self, cd, st):
            self.values, self.grid, self.wall_s = {"cd": cd, "st": st}, (1, 1), 0.0

    class FakeModule:
        def __init__(self):
            self.SQUARE_RE100_REF = {"cd": (1.477, 1.494), "st": (0.145, 0.149)}

        @staticmethod
        def run_square(re, cpd, device="auto"):
            return FakeRun({32: 1.5818, 48: 1.5489}[cpd], 0.146)

    ctx = ra.Context(device="cpu", quick=False)
    ctx._modules["ns2d_cases"] = FakeModule()
    checks, data = ra.case_square(ctx)
    cd = {c.quantity: c for c in checks if "mean Cd" in c.quantity}
    c32, c48 = cd["Square cylinder Re 100, mean Cd (32 cells/D)"], cd["Square cylinder Re 100, mean Cd (48 cells/D)"]
    assert c32.hi == pytest.approx(1.494 * 1.05) and c32.lo == pytest.approx(1.477 * 0.95)
    assert not c32.passed and c48.passed                 # 0.1.0's +5.9 % fails, its +3.7 % passes
    assert "8 %" not in c32.tolerance and "expected to fail" in c32.note
    assert set(data) == {"32", "48"}
