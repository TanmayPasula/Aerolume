"""polar_metrics on a synthetic polar with known answers; compare_rows deltas and flags."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.analysis import compare_rows, polar_metrics
from aerolume.model import CFDResult, PolarPoint, PolarResult, Stat

A0 = -2.0          # zero-lift angle [deg]
CM0 = -0.053


def synthetic_polar(step: float = 1.0) -> PolarResult:
    """Linear lift (2 pi per rad) to 12 deg, then stall; parabolic drag; constant Cm about c/4."""
    pts = []
    for a in np.arange(-6.0, 18.0 + 1e-9, step):
        if a <= 12.0:
            cl = 2 * math.pi * math.radians(a - A0)
        else:
            cl = 2 * math.pi * math.radians(12.0 - A0) - 0.08 * (a - 12.0) ** 2 - 0.05 * (a - 12.0)
        cd = 0.0062 + 0.0045 * (cl - 0.4) ** 2 + (0.002 * (a - 12.0) ** 2 if a > 12 else 0.0)
        pts.append(PolarPoint(alpha=float(a), cl=cl, cd=cd, cdp=cd * 0.3, cdf=cd * 0.7, cm=CM0,
                              converged=True))
    # a failed point must be ignored, even with absurd values
    pts.append(PolarPoint(alpha=5.5, cl=9.0, cd=1e-6, cm=1.0, converged=False))
    return PolarResult(sim_id="p1", re=1e6, points=pts)


def test_polar_metrics_known_values():
    pol = synthetic_polar()
    m = polar_metrics(pol)
    assert m["n_points"] == len(pol.points) and m["n_converged"] == len(pol.points) - 1
    assert m["lift_slope_per_rad"] == pytest.approx(2 * math.pi, rel=1e-9)
    assert m["lift_slope_per_deg"] == pytest.approx(2 * math.pi * math.pi / 180, rel=1e-9)
    assert m["alpha_zero_lift"] == pytest.approx(A0, abs=1e-9)
    assert m["cm0"] == pytest.approx(CM0)
    assert m["alpha_cl_max"] == pytest.approx(12.0)
    assert m["cl_max"] == pytest.approx(2 * math.pi * math.radians(14.0))
    # brute-force checks of the extremes over converged points
    tab = pol.table()
    i = int(np.argmin(tab["cd"]))
    assert m["cd_min"] == tab["cd"][i] and m["cl_at_cd_min"] == tab["cl"][i]
    ld = tab["cl"] / tab["cd"]
    j = int(np.argmax(ld))
    assert m["ld_max"] == pytest.approx(ld[j]) and m["alpha_ld_max"] == tab["alpha"][j]
    assert (m["alpha_min"], m["alpha_max"]) == (-6.0, 18.0)


def test_polar_metrics_accepts_table_and_handles_gaps():
    pol = synthetic_polar(0.5)
    assert polar_metrics(pol.table())["alpha_zero_lift"] == pytest.approx(A0, abs=1e-9)
    # sweep that never reaches zero lift: slope still found, alpha0 extrapolated
    sel = pol.table()["alpha"] >= 0
    tab = {k: v[sel] for k, v in pol.table().items()}
    m = polar_metrics(tab)
    assert m["lift_slope_per_rad"] == pytest.approx(2 * math.pi, rel=1e-6)
    assert m["alpha_zero_lift"] == pytest.approx(A0, abs=1e-6)
    assert m["cm0"] == pytest.approx(CM0)        # falls back to Cm at alpha = 0
    empty = polar_metrics(PolarResult(sim_id="x", re=1e5))
    assert empty["n_converged"] == 0 and math.isnan(empty["cl_max"])


def cfd(sim_id: str, cd: float, cl: float, se: float, converged: bool = True) -> CFDResult:
    q = 0.5 * 1.225 * 30.0 ** 2
    return CFDResult(sim_id=sim_id, cd=Stat(mean=cd, rms=0.01, stderr=se), cl=Stat(mean=cl, rms=0.2, stderr=se),
                     cm=Stat(mean=-0.1, stderr=se), drag=cd * q * 1.4, lift=cl * q * 1.4,
                     drag_area=cd * 1.4, power=cd * q * 1.4 * 30, strouhal=0.21, re_real=2.8e6,
                     re_sim=2e4, converged=converged)


def test_compare_rows_deltas_and_flags():
    res = {"base": cfd("base", 0.40, 0.10, 0.002), "a": cfd("a", 0.36, 0.12, 0.002),
           "b": cfd("b", 0.401, 0.10, 0.002, converged=False)}
    rows = {r["key"]: r for r in compare_rows(res, "base")}
    cd = rows["cd"]
    assert cd["baseline"] == 0.40 and cd["better"] == "lower"
    c_base, c_a, c_b = cd["cells"]
    assert c_base["is_baseline"] and c_base["delta"] == 0 and c_base["improved"] is None
    assert c_a["delta"] == pytest.approx(-0.04) and c_a["pct"] == pytest.approx(-10.0)
    assert c_a["improved"] is True and c_a["significant"] is True
    assert c_b["improved"] is False and c_b["significant"] is False      # 0.25 % is within noise
    ld = rows["ld"]["cells"][1]
    assert ld["value"] == pytest.approx(0.12 / 0.36) and ld["improved"] is True
    assert ld["stderr"] > 0
    assert rows["re_real"]["cells"][1]["improved"] is None               # context, not a result
    assert [c["value"] for c in rows["converged"]["cells"]] == [True, True, False]
    # sequence input, default baseline = first
    rows2 = compare_rows(list(res.values()))
    assert rows2[0]["cells"][0]["sim_id"] == "base"
    with pytest.raises(KeyError):
        compare_rows(res, "nope")
    assert compare_rows({}, None) == []


def test_compare_rows_polars_and_missing_values():
    p1 = synthetic_polar()
    p2 = synthetic_polar()
    p2.sim_id = "p2"
    for p in p2.points:
        p.cd *= 0.9
    rows = {r["key"]: r for r in compare_rows([p1, p2], "p1")}
    assert rows["cd_min"]["cells"][1]["pct"] == pytest.approx(-10.0)
    assert rows["cd_min"]["cells"][1]["improved"] is True
    assert rows["ld_max"]["cells"][1]["pct"] == pytest.approx(100 / 0.9 - 100)
    assert "cd" not in rows                       # no CFD result in the set
    mixed = {r["key"]: r for r in compare_rows([cfd("c", 0.3, 0.0, 0.001), p1], "c")}
    assert mixed["cd"]["cells"][1]["value"] is None and mixed["cd"]["cells"][1]["delta"] is None
    assert mixed["cl"]["cells"][0]["pct"] is None   # baseline Cl is exactly 0


# ---------------------------------------------------------------------------
# review round: Cl,max that is only the end of the sweep; rough error bars;
# runs that are not comparable
# ---------------------------------------------------------------------------

def _linear_polar(alpha_end: float, sim_id: str = "lin", peak_at: float | None = None) -> PolarResult:
    """Cl = 0.1 alpha + 0.2 up to ``peak_at`` (default: never), then falling."""
    pts = []
    for a in np.arange(-4.0, alpha_end + 1e-9, 1.0):
        cl = 0.1 * a + 0.2
        if peak_at is not None and a > peak_at:
            cl = 0.1 * peak_at + 0.2 - 0.15 * (a - peak_at)
        pts.append(PolarPoint(alpha=float(a), cl=cl, cd=0.008 + 0.01 * cl ** 2, cm=-0.05, converged=True))
    return PolarResult(sim_id=sim_id, re=2e6, points=pts)


def test_cl_max_still_rising_at_the_last_angle_is_not_reached():
    """The default NACA 2412 polar at Re 2e6 still gains 0.05 per degree at 14 deg."""
    m = polar_metrics(_linear_polar(14.0))
    assert m["cl_max_reached"] == 0.0
    assert m["cl_max"] == pytest.approx(1.6) and m["alpha_cl_max"] == 14.0   # kept: a lower bound
    m = polar_metrics(_linear_polar(16.0, peak_at=12.0))
    assert m["cl_max_reached"] == 1.0 and m["alpha_cl_max"] == 12.0
    # a curve that has flattened at the last angle counts as stalled
    flat = _linear_polar(14.0)
    flat.points[-1].cl = flat.points[-2].cl + 0.01          # 0.01 per deg << 25 % of 0.1
    assert polar_metrics(flat)["cl_max_reached"] == 1.0
    assert math.isnan(polar_metrics(PolarResult(sim_id="x", re=1e5))["cl_max_reached"])


def test_compare_never_ranks_polars_on_an_unreached_cl_max():
    a, b = _linear_polar(14.0, "a"), _linear_polar(14.0, "b")
    for p in b.points:
        p.cl *= 1.05
    rows = {r["key"]: r for r in compare_rows([a, b], "a")}
    cell = rows["cl_max"]["cells"][1]
    assert cell["lower_bound"] and cell["improved"] is None
    assert rows["alpha_cl_max"]["label"] == "Angle of Cl,max"
    # a real stall on both sides is compared as before
    c, d = _linear_polar(16.0, "c", peak_at=12.0), _linear_polar(16.0, "d", peak_at=12.0)
    for p in d.points:
        p.cl *= 1.05
    rows = {r["key"]: r for r in compare_rows([c, d], "c")}
    assert rows["cl_max"]["cells"][1]["improved"] is True and not rows["cl_max"]["cells"][1]["lower_bound"]


def test_significance_is_unknown_for_rough_or_missing_error_bars():
    base, var = cfd("base", 0.40, 0.10, 0.002), cfd("v", 0.36, 0.10, 0.002)
    var.cd.stderr_reliable = False
    cells = {r["key"]: r["cells"][1] for r in compare_rows([base, var], "base")}
    assert cells["cd"]["significant"] is None and cells["cd"]["stderr_reliable"] is False
    assert cells["drag"]["significant"] is None              # derived from Cd's error bar
    assert cells["cl"]["significant"] is False               # Cl's error bar is a measured one
    nan_se = cfd("n", 0.36, 0.10, float("nan"))
    cells = {r["key"]: r["cells"][1] for r in compare_rows([base, nan_se], "base")}
    assert cells["cd"]["significant"] is None                # nan used to read as "not significant"


def test_comparison_warnings_name_conditions_and_reynolds_numbers():
    from aerolume.analysis import comparison_warnings, conditions_differ, reynolds_mismatch
    from aerolume.model import FlowConditions
    a, b = cfd("a", 0.1, 0.2, 0.001), cfd("b", 0.1, 0.2, 0.001)
    same = {"a": FlowConditions(speed=14.6, alpha_deg=4.0), "b": FlowConditions(speed=14.6, alpha_deg=4.0)}
    assert comparison_warnings([a, b], same) == [] and reynolds_mismatch([a, b]) == {}
    b.re_sim = 2.56e4                                        # a Standard run with a grid-chosen cap
    msgs = comparison_warnings([a, b], same)
    assert len(msgs) == 1 and "different Reynolds numbers" in msgs[0] and "not design effects" in msgs[0]
    other = {"a": same["a"], "b": FlowConditions(speed=30.0, alpha_deg=4.0)}
    assert conditions_differ(other) == ["speed"]
    msgs = comparison_warnings([a, b], other)                # the speed explains the Reynolds number
    assert len(msgs) == 1 and "different speeds" in msgs[0]
    other["b"] = FlowConditions(speed=14.6, alpha_deg=6.0)
    assert conditions_differ(other) == ["angle"]
    # a polar sweeps the angle by design: not judged
    assert comparison_warnings([a, synthetic_polar()], {"a": same["a"], "p1": other["b"]}) == []
