"""3D analysis: wing figures of merit, six-coefficient history statistics and 2D/3D-aware comparison."""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.analysis import (
    HISTORY_KEYS_3D,
    WING_METRIC_KEYS,
    compare_rows,
    comparison_groups,
    comparison_warnings,
    conditions_differ,
    result_dimension,
    reynolds_mismatch,
    summarize_history3d,
    wing_metrics,
)
from aerolume.model import (
    CFD3DResult,
    CFDResult,
    FlowConditions,
    Fluid,
    PolarPoint,
    PolarResult,
    Stat,
    WingPoint,
    WingResult,
)

AR = 8.0
CD0 = 0.012
E_OSWALD = 0.85
A0 = -2.0                      # zero-lift angle [deg]
SLOPE = 0.08                   # dCL/dalpha [1/deg]


def make_wing(sim_id: str = "w1", *, viscous: bool = True, stall_at: float | None = None,
              alphas: np.ndarray | None = None, e_span: float = 0.97) -> WingResult:
    """A synthetic sweep with an exactly parabolic drag polar CD = CD0 + CL^2 / (pi e AR).

    With ``stall_at`` the lift curve turns over above that angle (LLT with section
    stall); ``viscous=False`` keeps induced drag only (CDi with span efficiency e_span).
    """
    pts = []
    for a in (np.arange(-4.0, 13.0, 1.0) if alphas is None else alphas):
        cl = SLOPE * (a - A0)
        if stall_at is not None and a > stall_at:
            cl = SLOPE * (stall_at - A0) - 0.06 * (a - stall_at) ** 2
        cdi = cl ** 2 / (math.pi * e_span * AR)
        cd = CD0 + cl ** 2 / (math.pi * E_OSWALD * AR) if viscous else cdi
        if stall_at is not None and a > stall_at:
            cd += 0.02 * (a - stall_at)
        y = np.linspace(-5, 5, 12)
        pts.append(WingPoint(alpha=float(a), cl=cl, cd=cd, cdi=cdi, cdv=cd - cdi if viscous else 0.0,
                             cm=-0.05 - 0.01 * (a - A0), e=e_span, converged=True,
                             y=y, cl_local=cl * np.sqrt(1 - (y / 5.5) ** 2), c_cl=np.ones(12),
                             alpha_i=np.full(12, -1.0), cd_local=None))
    return WingResult(sim_id=sim_id, method="llt" if stall_at else "vlm", viscous=viscous, area=12.5, span=10.0,
                      aspect_ratio=AR, mac=1.25, re_mac=1.0e6, points=pts)


# --- wing_metrics -------------------------------------------------------------------

def test_wing_metrics_recovers_a_known_linear_wing():
    m = wing_metrics(make_wing())
    assert set(m) == set(WING_METRIC_KEYS)
    assert m["cl_alpha_per_deg"] == pytest.approx(SLOPE, rel=1e-9)
    assert m["cl_alpha_per_rad"] == pytest.approx(SLOPE * 180 / math.pi, rel=1e-9)
    assert m["alpha_zero_lift"] == pytest.approx(A0, abs=1e-9)
    # exact parabola: the fit recovers CD0, k and the Oswald factor
    assert m["cd0"] == pytest.approx(CD0, rel=1e-9)
    assert m["k_parabolic"] == pytest.approx(1 / (math.pi * E_OSWALD * AR), rel=1e-9)
    assert m["oswald"] == pytest.approx(E_OSWALD, rel=1e-9)
    # linear lift to the last angle: CL,max is only a lower bound
    assert m["cl_max_reached"] == 0.0 and m["alpha_cl_max"] == 12.0
    # best L/D at a computed point, close to the analytic optimum CL* = sqrt(CD0 pi e AR)
    cl_star = math.sqrt(CD0 * math.pi * E_OSWALD * AR)
    assert abs(m["cl_at_ld_max"] - cl_star) <= SLOPE / 2 + 1e-9
    assert m["ld_max"] == pytest.approx(cl_star / (2 * CD0), rel=0.01)
    # e at cruise = the solver's span efficiency at best L/D
    assert m["e_cruise"] == pytest.approx(0.97) and m["alpha_cruise"] == m["alpha_ld_max"]
    assert m["cm_alpha_per_deg"] == pytest.approx(-0.01, rel=1e-9)
    assert m["cm0"] == pytest.approx(-0.05, abs=1e-12)
    assert (m["n_points"], m["n_converged"], m["aspect_ratio"]) == (17.0, 17.0, AR)


def test_wing_metrics_with_stall_and_cruise_cl():
    w = make_wing(stall_at=10.0)
    m = wing_metrics(w)
    assert m["cl_max_reached"] == 1.0
    assert m["alpha_cl_max"] == 10.0 and m["cl_max"] == pytest.approx(SLOPE * 12.0)
    # post-stall drag rise is outside the fit window: the parabola is still exact
    assert m["oswald"] == pytest.approx(E_OSWALD, rel=1e-9) and m["cd0"] == pytest.approx(CD0, rel=1e-9)
    # explicit cruise CL is interpolated in the attached range
    mc = wing_metrics(w, cl_cruise=0.5)
    assert mc["cl_cruise"] == 0.5 and mc["e_cruise"] == pytest.approx(0.97)
    assert mc["alpha_cruise"] == pytest.approx(A0 + 0.5 / SLOPE)
    assert math.isnan(wing_metrics(w, cl_cruise=3.0)["e_cruise"])          # outside the sweep


def test_wing_metrics_inviscid_reports_no_profile_drag_figures():
    m = wing_metrics(make_wing(viscous=False, e_span=0.99))
    for k in ("cd_min", "ld_max", "alpha_ld_max", "cl_at_ld_max", "cd0"):
        assert math.isnan(m[k]), k
    assert m["oswald"] == pytest.approx(0.99, rel=1e-9)      # induced only: the span efficiency
    assert m["e_cruise"] == pytest.approx(0.99) and m["alpha_cruise"] == 12.0


def test_wing_metrics_on_a_table_mapping_and_degenerate_input():
    w = make_wing()
    m = wing_metrics(w.table(), aspect_ratio=AR)
    assert m["oswald"] == pytest.approx(E_OSWALD, rel=1e-9)
    assert math.isnan(wing_metrics(w.table())["oswald"])                   # no AR, no Oswald factor
    empty = wing_metrics(WingResult(sim_id="e"))
    assert empty["n_points"] == 0.0 and all(math.isnan(empty[k]) for k in ("cl_max", "oswald", "e_cruise"))
    # AR derived from span and area when the result does not carry it
    w2 = make_wing()
    w2.aspect_ratio = float("nan")
    assert wing_metrics(w2)["aspect_ratio"] == pytest.approx(10.0 ** 2 / 12.5)
    # two points cannot define a parabola
    w3 = make_wing(alphas=np.array([0.0, 4.0]))
    assert math.isnan(wing_metrics(w3)["oswald"])
    # unconverged points are ignored but counted
    w4 = make_wing()
    w4.points[3].converged = False
    m4 = wing_metrics(w4)
    assert (m4["n_points"], m4["n_converged"]) == (17.0, 16.0)


def test_oswald_of_an_untwisted_rectangular_wing_matches_its_span_efficiency():
    """Inviscid vortex lattice, rectangular AR 8, no twist: the drag polar is CDi = CL^2 / (pi AR e)
    with one e, so the parabolic-fit Oswald factor must be the solver's span efficiency."""
    from aerolume.model import WingSection, WingSpec
    from aerolume.solvers.wing import wing_polar

    wing = WingSpec(sections=[WingSection(y=0.0, chord=1.0, airfoil="0012"),
                              WingSection(y=4.0, chord=1.0, airfoil="0012")], symmetric=True,
                    n_chord=4, n_span=16)
    res = wing_polar(wing, [float(a) for a in range(-4, 13)], speed=30.0, viscous=False)
    assert res.aspect_ratio == pytest.approx(8.0)
    m = wing_metrics(res)
    e_span = float(np.nanmedian([p.e for p in res.points if abs(p.cl) > 0.1]))
    assert 0.9 < e_span <= 1.0
    assert m["oswald"] == pytest.approx(e_span, rel=0.02)
    assert m["oswald_r2"] > 0.999 and m["oswald_cl_lo"] < 0.05 and m["oswald_cl_hi"] > 0.6


# The critic's viscous VLM sweep (AR 8.33, taper 0.6, NACA 2412, 25 m/s): the strips reach their
# section cl_max at alpha 12.7 deg (CL 1.222), and at 13-14 deg the lattice lift keeps rising while
# their profile drag is held at cl_max. Fitted over the whole sweep this gave Oswald e = 0.54.
_CRITIC_SWEEP = np.array([
    # alpha, CL, CD, CDi, e
    (-4.0, -0.1531, 0.01063, 0.00090, 0.997), (-3.0, -0.0702, 0.00910, 0.00019, 0.997),
    (-2.0, 0.0128, 0.00807, 0.00001, np.nan), (-1.0, 0.0958, 0.00766, 0.00036, 0.976),
    (0.0, 0.1789, 0.00815, 0.00124, 0.985), (1.0, 0.2619, 0.00957, 0.00265, 0.988),
    (2.0, 0.3449, 0.01174, 0.00459, 0.989), (3.0, 0.4278, 0.01449, 0.00706, 0.990),
    (4.0, 0.5106, 0.01778, 0.01006, 0.990), (5.0, 0.5934, 0.02162, 0.01358, 0.990),
    (6.0, 0.6760, 0.02618, 0.01763, 0.990), (7.0, 0.7585, 0.03143, 0.02219, 0.990),
    (8.0, 0.8409, 0.03761, 0.02728, 0.990), (9.0, 0.9230, 0.04497, 0.03288, 0.990),
    (10.0, 1.0050, 0.05332, 0.03899, 0.989), (11.0, 1.0867, 0.06254, 0.04561, 0.989),
    (12.0, 1.1682, 0.07426, 0.05274, 0.988), (13.0, 1.2494, 0.09509, 0.06036, 0.988),
    (14.0, 1.3303, 0.11005, 0.06848, 0.987)])


def _critic_wing() -> WingResult:
    pts = [WingPoint(alpha=float(a), cl=float(cl), cd=float(cd), cdi=float(cdi), cdv=float(cd - cdi), e=float(e),
                     cm=-0.05, converged=True) for a, cl, cd, cdi, e in _CRITIC_SWEEP]
    return WingResult(sim_id="w", method="vlm", viscous=True, area=0.48, span=2.0, aspect_ratio=2.0 ** 2 / 0.48,
                      mac=0.245, re_mac=4.19e5, points=pts, stall_alpha=12.6674, stall_cl=1.2224)


def test_oswald_fit_uses_only_the_linear_attached_polar():
    m = wing_metrics(_critic_wing())
    # window: from the lift of minimum drag to 0.8 x the critical-section CL_max, below stall onset
    assert m["oswald_cl_lo"] == pytest.approx(0.0958) and m["cl_at_cd_min"] == pytest.approx(0.0958)
    assert m["oswald_cl_hi"] == pytest.approx(0.9230) and m["oswald_cl_hi"] <= 0.8 * 1.2224
    assert m["oswald_r2"] > 0.99
    # a normal viscous Oswald factor, well below the span efficiency it includes
    assert 0.72 < m["oswald"] < 0.80 and m["e_cruise"] == pytest.approx(0.990)
    assert m["cd0"] == pytest.approx(0.0077, abs=3e-4)
    # a result saved before the stall estimate existed: the lattice's "beyond cl_max" warning still
    # keeps the strips held at cl_max out of the fit (CL,max was not reached, so no 0.8 cap)
    old = _critic_wing()
    old.stall_alpha = old.stall_cl = float("nan")
    old.warnings = [("Strips asked for more lift than their section polar reaches (beyond cl_max) at alpha = "
                     "13, 14 deg: the vortex lattice's lift is linear and does not stall.")]
    m_old = wing_metrics(old)
    assert m_old["oswald_cl_hi"] == pytest.approx(1.1682) and 0.6 < m_old["oswald"] < m["oswald"]
    old.warnings = []                                     # nothing tells the clipped points apart
    assert wing_metrics(old)["oswald_cl_hi"] == pytest.approx(1.3303)


def test_coarse_inviscid_sweep_still_reports_its_span_efficiency():
    """Five angles, lift still rising, no profile drag: no CL cap applies, the fit uses them all."""
    m = wing_metrics(make_wing(viscous=False, e_span=0.95, alphas=np.arange(0.0, 9.0, 2.0)))
    assert m["oswald"] == pytest.approx(0.95, rel=1e-9) and m["oswald_cl_hi"] == pytest.approx(SLOPE * 10.0)


def test_oswald_is_omitted_when_the_polar_is_not_parabolic():
    w = make_wing()
    for i, p in enumerate(w.points):                      # scattered drag: no parabola describes it
        p.cd = CD0 + 0.02 * p.cl ** 2 + (0.006 if i % 2 else -0.006)
    m = wing_metrics(w)
    assert m["oswald_r2"] < 0.98
    assert all(math.isnan(m[k]) for k in ("oswald", "cd0", "k_parabolic"))
    # too few points in the window: no fit at all
    m3 = wing_metrics(make_wing(alphas=np.array([0.0, 2.0, 4.0])))
    assert math.isnan(m3["oswald"]) and math.isnan(m3["oswald_r2"])


# --- six-coefficient history -------------------------------------------------------

def test_summarize_history3d_stats_strouhal_and_symmetry_zeros():
    t = np.linspace(0.0, 60.0, 6001)
    rng = np.random.default_rng(3)
    transient = np.exp(-t / 2.0)
    st_true = 0.2
    hist = {"t": t,
            "cd": 0.65 + 0.5 * transient + 0.002 * rng.standard_normal(t.size),
            "cs": 0.08 * np.sin(2 * np.pi * st_true * t),                    # side-force shedding
            "cl": 0.01 * np.sin(2 * np.pi * 0.05 * t) + 0.001 * rng.standard_normal(t.size),
            "c_roll": np.zeros_like(t), "cm": -0.02 + 0.1 * transient, "cn": np.zeros_like(t)}
    out = summarize_history3d(hist, length=1.0, speed=1.0)
    assert set(out["stats"]) == set(HISTORY_KEYS_3D)
    assert out["t_start"] > 5.0                                               # transient discarded
    assert out["stats"]["cd"].mean == pytest.approx(0.65, abs=0.002)
    assert out["stats"]["cs"].rms == pytest.approx(0.08 / math.sqrt(2), rel=0.02)
    assert out["strouhal_from"] == "cs" and out["strouhal"] == pytest.approx(st_true, rel=0.03)
    for k in ("c_roll", "cn"):                                                # symmetry: identically zero
        assert out["stats"][k].mean == 0.0 and out["convergence"][k]["converged"]


# --- comparison ----------------------------------------------------------------------

def stat(mean: float, se: float = 0.002, ok: bool = True) -> Stat:
    return Stat(mean=mean, rms=0.01, min=mean - 0.02, max=mean + 0.02, stderr=se, stderr_reliable=ok)


def cfd3d(sid: str, cd: float, cl: float = -0.1, re_sim: float = 2e5, converged: bool = True) -> CFD3DResult:
    q_area = 0.5 * 1.225 * 30.0 ** 2 * 2.0
    return CFD3DResult(sim_id=sid, cd=stat(cd), cs=stat(0.0), cl=stat(cl), c_roll=stat(0.0), cm=stat(0.03),
                       cn=stat(0.0), drag=cd * q_area, side=0.0, lift=cl * q_area, moments=(0.0, 12.0, 0.0),
                       drag_area=cd * 2.0, power=cd * q_area * 30.0, re_real=8e6, re_sim=re_sim, converged=converged,
                       grid=(256, 64, 96), cells=256 * 64 * 96, symmetry_plane=True)


def cfd2d(sid: str, cd: float, re_sim: float = 1e4) -> CFDResult:
    return CFDResult(sim_id=sid, cd=stat(cd), cl=stat(0.5), cm=stat(-0.05), drag=cd * 100, lift=50.0,
                     re_real=1e6, re_sim=re_sim, converged=True)


def polar(sid: str) -> PolarResult:
    pts = [PolarPoint(alpha=a, cl=0.1 * a, cd=0.01 + 0.001 * a * a, cm=-0.05, converged=True) for a in range(-2, 9)]
    return PolarResult(sim_id=sid, re=1e6, points=pts)


def test_result_dimension_and_groups():
    assert result_dimension(cfd2d("a", 1.0)) == 2 and result_dimension(polar("p")) == 2
    assert result_dimension(cfd3d("b", 0.3)) == 3 and result_dimension(make_wing("w")) == 3
    with pytest.raises(TypeError):
        result_dimension("not a result")  # type: ignore[arg-type]
    res = [cfd2d("a", 1.0), cfd3d("b", 0.3), polar("p"), make_wing("w")]
    assert comparison_groups(res) == {2: ["a", "p"], 3: ["b", "w"]}


def test_compare_rows_3d_cfd_like_for_like():
    rows = compare_rows([cfd3d("base", 0.30), cfd3d("var", 0.27)])
    keys = [r["key"] for r in rows]
    for k in ("cd", "cs", "cl", "c_roll", "cm", "cn", "ld", "drag", "side", "lift", "pitch_moment", "drag_area",
              "power", "re_sim", "converged"):
        assert k in keys, k
    assert not {"cl_max", "cd0", "oswald"} & set(keys)                       # no wing in the set
    cd = next(r for r in rows if r["key"] == "cd")
    assert cd["kind"] == "cfd3d" and cd["label"].endswith("CD")
    c = cd["cells"][1]
    assert c["delta"] == pytest.approx(-0.03) and c["pct"] == pytest.approx(-10.0)
    assert c["improved"] is True and c["significant"] is True and c["comparable"] is True
    drag = next(r for r in rows if r["key"] == "drag")
    assert drag["unit"] == "N" and drag["cells"][1]["stderr"] == pytest.approx(
        abs(drag["cells"][1]["value"] * 0.002 / 0.27))
    conv = rows[-1]
    assert conv["key"] == "converged" and conv["kind"] == "cfd3d" and conv["cells"][0]["value"] is True


def test_compare_rows_3d_cfd_and_wing_share_reynolds_only():
    w = make_wing("wing")
    rows = compare_rows({"cfd": cfd3d("cfd", 0.05, cl=0.6), "wing": w}, baseline_id="wing")
    by = {r["key"]: r for r in rows}
    assert by["oswald"]["cells"][1]["value"] == pytest.approx(E_OSWALD)
    assert by["oswald"]["cells"][0]["value"] is None and by["cd"]["cells"][1]["value"] is None
    assert by["re_real"]["cells"][0]["value"] == 8e6 and by["re_real"]["cells"][1]["value"] == 1e6
    # CL,max of a sweep still rising at its last angle is a lower bound: never an "improvement"
    assert by["cl_max"]["cells"][1]["lower_bound"] is True
    assert all(c["comparable"] for r in rows for c in r["cells"])
    rows2 = compare_rows([make_wing("a"), make_wing("b", stall_at=10.0)])
    clmax = next(r for r in rows2 if r["key"] == "cl_max")
    assert clmax["cells"][0]["lower_bound"] is True and clmax["cells"][1]["improved"] is None


@pytest.mark.parametrize("baseline", ["two_d", "three_d"])
def test_compare_rows_never_mixes_dimensions(baseline):
    res = {"two_d": cfd2d("two_d", 1.2), "pol": polar("pol"), "three_d": cfd3d("three_d", 0.3),
           "wing": make_wing("wing")}
    rows = compare_rows(res, baseline_id=baseline)
    other = {"pol", "two_d"} if baseline == "three_d" else {"three_d", "wing"}
    for r in rows:
        assert [c["sim_id"] for c in r["cells"]] == list(res)               # every column kept, in order
        for c in r["cells"]:
            if c["sim_id"] in other:
                assert c["comparable"] is False
                assert c["value"] is None and c["delta"] is None and c["improved"] is None
                assert c["significant"] is None
            else:
                assert c["comparable"] is True
    kinds = {r["kind"] for r in rows}
    assert kinds <= ({"cfd", "polar"} if baseline == "two_d" else {"cfd3d", "wing"})
    labels = {r["key"]: r["label"] for r in rows}
    assert labels["cd"] == ("Drag coefficient Cd" if baseline == "two_d" else "Drag coefficient CD")


def test_comparison_warnings_for_3d_and_mixed_selections():
    res = {"a": cfd2d("a", 1.0), "b": cfd3d("b", 0.3), "c": cfd3d("c", 0.32)}
    conds = {"a": FlowConditions(speed=30.0), "b": FlowConditions(speed=30.0, yaw_deg=0.0),
             "c": FlowConditions(speed=30.0, yaw_deg=5.0)}
    msgs = comparison_warnings(res, conds)
    assert any("mixes 1 2D and 2 3D results" in m and "not comparable" in m for m in msgs)
    assert any("different yaw angles" in m for m in msgs)
    assert conditions_differ([FlowConditions(), FlowConditions(yaw_deg=3.0)]) == ["yaw"]
    # 2D-only conditions never differ in yaw (objects without the attribute too)

    class Old:
        speed, alpha_deg, fluid = 30.0, 0.0, Fluid()
    assert conditions_differ([Old(), Old()]) == []
    # Reynolds numbers are judged within a dimension, never across
    assert reynolds_mismatch({"a": cfd2d("a", 1.0, re_sim=1e4), "b": cfd3d("b", 0.3, re_sim=1e6)}) == {}
    mism = reynolds_mismatch({"b": cfd3d("b", 0.3, re_sim=1e5), "c": cfd3d("c", 0.3, re_sim=2e5),
                              "a": cfd2d("a", 1.0)})
    assert mism == {"b": 1e5, "c": 2e5}


def test_gui_and_script_air_are_the_same_fluid():
    """A GUI-made simulation (default_simulation -> Fluid() defaults) and a script/assistant one
    (core.fluid.air_isa) must store the same "Air, ISA sea level" numbers, and Compare must not
    call them different fluids -- not even for projects saved with the old rounded constants."""
    from aerolume.analysis.compare import FLUID_REL_TOL
    from aerolume.core.fluid import PRESETS, air_isa
    from aerolume.model import Design
    from aerolume.simulation.defaults import default_simulation

    car = Design(name="car", kind="vehicle", bodies=[], length_ref=4.5, force_ref=1.4)
    gui = default_simulation(car, "Run 1").conditions.fluid
    assert gui == air_isa(0.0) == PRESETS["Air, ISA sea level"]()          # identical, not just close
    assert Fluid() == air_isa(0.0)
    # the two values measured in the critic's project (GUI car vs API sphere)
    old_gui = FlowConditions(fluid=Fluid("Air, ISA sea level", 1.225, 1.7894e-05))
    api = FlowConditions(fluid=Fluid("Air, ISA sea level", 1.225000018, 1.7892976e-05))
    assert conditions_differ([old_gui, api]) == []
    assert not any("fluid" in m for m in comparison_warnings(
        {"a": cfd3d("a", 0.3), "b": cfd3d("b", 0.31)}, {"a": old_gui, "b": api}))
    # a real change of fluid is still reported
    assert conditions_differ([api, FlowConditions(fluid=air_isa(1000.0))]) == ["fluid"]
    assert conditions_differ([api, FlowConditions(fluid=Fluid("x", 1.225 * (1 + 3 * FLUID_REL_TOL),
                                                               1.7893e-5))]) == ["fluid"]
    assert any("different Reynolds numbers" in m for m in comparison_warnings(
        {"b": cfd3d("b", 0.3, re_sim=1e5), "c": cfd3d("c", 0.3, re_sim=2e5)}, None))
    # wing sweeps are not judged on conditions (they sweep the angle by design)
    assert comparison_warnings({"w1": make_wing("w1"), "w2": make_wing("w2")},
                               {"w1": FlowConditions(alpha_deg=0), "w2": FlowConditions(alpha_deg=5)}) == []
