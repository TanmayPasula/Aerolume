"""SectionPolars: node placement, interpolation, stall detection, extension, cancel, real-solver integration.

Most tests replace :func:`aerolume.solvers.airfoil.polar` with an analytic stand-in
(instant, exact), so they check the cache logic itself; one test runs the real
XFOIL-class solver end to end on a small wing.
"""
from __future__ import annotations

import math
import threading

import numpy as np
import pytest

import aerolume.solvers.airfoil as airfoil_pkg
from aerolume.model import PolarPoint, PolarResult, WingSection, WingSpec
from aerolume.solvers.wing import SectionPolars, airfoil_key, wing_polar
from aerolume.solvers.wing.polars import (
    RE_GRID_PER_DECADE,
    SNAP,
    PolarCancelled,
    _make_table,
    blend_tables,
)


def _section(re, a, stall=12.0, fail_above=15.0):
    cl = 0.1 * (a + 2.0) if a <= stall else 0.1 * (stall + 2.0) - 0.04 * (a - stall)
    cd = 0.006 * (1e6 / re) ** 0.3 * (1.0 + cl * cl)
    return cl, cd, -0.05, a <= fail_above


class FakeSolver:
    """Stand-in for solvers.airfoil.polar: records every call."""

    def __init__(self, stall=12.0, fail_above=15.0, warn=None):
        self.calls = []
        self.stall, self.fail_above, self.warn = stall, fail_above, warn

    def __call__(self, coords, re, alphas, *, mach=0.0, ncrit=9.0, xtr_upper=1.0, xtr_lower=1.0, panels=199,
                 distributions=True, sim_id="", progress=None, cancel=None):
        alphas = [float(a) for a in alphas]
        self.calls.append((float(re), alphas))
        res = PolarResult(sim_id=sim_id, re=re, mach=mach, ncrit=ncrit)
        for k, a in enumerate(alphas):
            if cancel is not None and cancel.is_set():
                res.warnings.append("Cancelled after some angles.")
                break
            cl, cd, cm, ok = _section(re, a, self.stall, self.fail_above)
            res.points.append(PolarPoint(alpha=a, cl=cl if ok else math.nan, cd=cd if ok else math.nan,
                                         cm=cm if ok else math.nan, converged=ok))
            if progress:
                progress((k + 1) / len(alphas), f"alpha = {a:g}")
        if self.warn:
            res.warnings.append(self.warn)
        return res


@pytest.fixture
def fake(monkeypatch):
    f = FakeSolver()
    monkeypatch.setattr(airfoil_pkg, "polar", f)
    return f


def test_airfoil_keys():
    assert airfoil_key("NACA 2412") == airfoil_key("naca2412") == airfoil_key(" 2412") == "2412"
    assert airfoil_key("23012") == "23012"
    assert airfoil_key("design:ab12") == "design:ab12"


def _on_grid(re: float) -> bool:
    k = math.log10(re) * RE_GRID_PER_DECADE
    return abs(k - round(k)) < 1e-4


def test_nodes_cover_the_reynolds_range_and_are_reused(fake):
    sp = SectionPolars(re_ratio=2.0)
    wanted = (1e5, 1.5e5, 4e5, 2.2e5, 3e5)                   # strips spanning a factor 4 in Re
    sp.ensure({"2412": [(r, -2, 4) for r in wanted]})
    res = sorted({c[0] for c in fake.calls})
    # three grid nodes (the fewest a factor 4 allows), 1e5 being one; the top one is within SNAP of 4e5
    assert res[0] == pytest.approx(1e5) and len(res) == 3 and all(_on_grid(r) for r in res)
    assert abs(math.log(res[-1] / 4e5)) <= SNAP
    # every requested Re is within SNAP of a node or between two nodes at most re_ratio apart
    for r in wanted:
        near = [x for x in res if abs(math.log(x / r)) <= SNAP]
        lo, hi = max((x for x in res if x <= r), default=None), min((x for x in res if x >= r), default=None)
        assert near or (lo and hi and hi / lo <= 2.0 + 1e-9)
    assert all(c[1] == [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0] for c in fake.calls)
    n = len(fake.calls)
    assert sp.ensure({"NACA 2412": [(1.5e5, -1, 3), (3.9e5, 0, 4)]}) == 0 and len(fake.calls) == n
    assert sp.cost({"2412": [(3e5, -2, 4)]}) == 0
    assert sp.cost({"2412": [(3e5, -2, 6)]}) == 4            # 5, 6 at both bracketing nodes
    assert sp.node_res("2412") == pytest.approx(res)


def test_reynolds_interpolation_is_log_linear(fake):
    sp = SectionPolars()
    sp.ensure({"0012": [(1e5, -2, 4), (2e5, -2, 4)]})
    re = math.sqrt(1e5 * 2e5)
    t = sp.table("0012", re)
    cl, cd, _, _ = _section(re, 2.0)
    assert t.re == pytest.approx(re)
    assert float(t.drag(2.0)) == pytest.approx(cd, rel=1e-9)         # power law in Re: exact in log space
    assert float(t.lift(2.0)[0][0]) == pytest.approx(cl, rel=1e-9)
    # a Reynolds number within 5 % of a node uses that node alone
    assert sp.table("0012", 1.03e5).re == pytest.approx(1e5)
    # beyond the nodes: the nearest node (no extrapolation)
    assert sp.table("0012", 5e5).re == pytest.approx(sp.node_res("0012")[-1])
    assert sp.node_res("0012")[-1] == pytest.approx(2e5, rel=0.025)


def test_reynolds_nodes_lie_on_a_grid_and_a_new_speed_adds_only_what_it_needs(fake):
    """A tapered wing's strip Re range at 27 m/s, then at 33 and 25 m/s (the critic's study):
    each new speed adds one grid node, never a second node beside an existing one."""
    sp = SectionPolars()
    lo, hi = 3.33e5, 5.47e5                                   # chords 0.18-0.30 m at 27 m/s
    span = [lo * (hi / lo) ** (k / 23) for k in range(24)]
    sp.ensure({"2412": [(r, -6, 15) for r in span]})
    first = sp.node_res("2412")
    assert len(first) == 2 and all(_on_grid(r) for r in first)
    for v in (33.0, 25.0):
        before = sp.node_res("2412")
        sp.ensure({"2412": [(r * v / 27.0, -6, 15) for r in span]})
        after = sp.node_res("2412")
        assert len(after) == len(before) + 1 and set(before) <= set(after)
        assert all(_on_grid(r) for r in after)
    # every strip of every run is bracketed by nodes at most re_ratio apart, or within SNAP of one
    for v in (25.0, 27.0, 33.0):
        for r in span:
            nodes = sp._bracket("2412", r * v / 27.0)
            assert len(nodes) in (1, 2)
            if len(nodes) == 2:
                assert nodes[1][0] / nodes[0][0] <= 2.0 + 1e-9
            else:
                assert abs(math.log(nodes[0][0] / (r * v / 27.0))) <= SNAP + 1e-12
    # one chord: a single grid node within half a grid step
    sp1 = SectionPolars()
    sp1.ensure({"0012": [(6.16e5, -2, 4)]})
    (only,) = sp1.node_res("0012")
    assert _on_grid(only) and abs(math.log10(only / 6.16e5)) <= 0.5 / RE_GRID_PER_DECADE + 1e-9


def test_attached_branch_and_stall_detection(fake):
    sp = SectionPolars()
    sp.ensure({"2412": [(1e6, -4, 18)]})
    t = sp.table("2412", 1e6)
    assert t.top_limit == "stall" and t.alpha_cl_max == pytest.approx(12.0) and t.cl_max == pytest.approx(1.4)
    a, clip = t.alpha_at_cl(np.array([0.5, 1.4, 2.0, -5.0]))
    assert a[0] == pytest.approx(3.0) and list(clip) == [0, 0, 1, -1]
    cl, d, clip = t.lift(np.array([3.0, 14.0, -30.0]))
    assert cl[0] == pytest.approx(0.5) and d[0] == pytest.approx(0.1)
    assert cl[1] == pytest.approx(1.4) and d[1] == 0 and list(clip) == [0, 1, -1]
    assert t.bottom_limit == "range"                         # lift still falling at the lowest angle
    # still rising at the top of the computed range -> "range"
    sp2 = SectionPolars()
    sp2.ensure({"2412": [(1e6, -2, 6)]})
    assert sp2.table("2412", 1e6).top_limit == "range"


def test_extension_is_seeded_from_the_converged_edge(fake):
    sp = SectionPolars()
    sp.ensure({"2412": [(1e6, -2, 4)]})
    sp.ensure({"2412": [(1e6, -2, 8)]})
    assert fake.calls[-1][1] == [4.0, 5.0, 6.0, 7.0, 8.0]   # starts from the known top angle
    sp.ensure({"2412": [(1e6, -5, 8)]})
    assert fake.calls[-1][1] == [-5.0, -4.0, -3.0, -2.0]
    t = sp.table("2412", 1e6)
    assert t.alpha[0] == -5 and t.alpha[-1] == 8


def test_failed_points_mark_stall_and_warnings_are_prefixed(monkeypatch):
    f = FakeSolver(stall=30.0, fail_above=10.0, warn="Re is very low")
    monkeypatch.setattr(airfoil_pkg, "polar", f)
    sp = SectionPolars()
    sp.ensure({"2412": [(1e6, 0, 13)]})
    t = sp.table("2412", 1e6)
    assert t.alpha[-1] == 10 and t.top_limit == "stall"     # the solver failed above: a real limit
    assert sp.warnings == ["Section 2412 at Re 1e+06: Re is very low"]


def test_blend_tables():
    a = np.arange(-4.0, 9.0)
    t1 = _make_table(a, 0.1 * a, np.full_like(a, 0.01), np.full_like(a, -0.1), 1e6)
    t2 = _make_table(a, 0.1 * a + 0.2, np.full_like(a, 0.04), np.zeros_like(a), 4e6)
    assert blend_tables([t1, t2], [1.0, 0.0]) is t1
    b = blend_tables([t1, t2], [0.5, 0.5])
    assert float(b.lift(2.0)[0][0]) == pytest.approx(0.3)
    assert float(b.drag(2.0)) == pytest.approx(0.02)        # geometric mean of 0.01 and 0.04
    assert float(b.moment(2.0)) == pytest.approx(-0.05)
    assert b.re == pytest.approx(2e6)


def test_cancel_raises_and_keeps_finished_points(fake):
    sp = SectionPolars()
    ev = threading.Event()

    def stop(frac, msg):
        if frac > 0.3:
            ev.set()

    with pytest.raises(PolarCancelled):
        sp.ensure({"2412": [(1e6, -4, 12)]}, progress=stop, cancel=ev)
    kept = sp._nodes["2412"][1e6].converged()
    assert 0 < len(kept) < 17
    ev.clear()
    sp.ensure({"2412": [(1e6, -4, 12)]})
    assert len(sp._nodes["2412"][1e6].converged()) == 17


def test_coords_resolution():
    sp = SectionPolars()
    assert sp.coords("NACA 23012").shape[1] == 2
    with pytest.raises(ValueError, match="coords_provider"):
        sp.coords("design:abc")
    with pytest.raises(ValueError, match="Unknown airfoil"):
        sp.coords("clark-y")
    calls = []
    sp2 = SectionPolars(coords_provider=lambda n: calls.append(n) or sp.coords("2412"))
    sp2.coords("design:abc")
    sp2.coords("design:abc")
    assert calls == ["design:abc"]                           # resolved once, then cached
    with pytest.raises(ValueError):
        SectionPolars(re_ratio=1.0)


def test_wing_polar_uses_the_cache_and_reports_nodes(fake):
    w = WingSpec(sections=[WingSection(y=0, chord=0.3, airfoil="2412"),
                           WingSection(y=1.0, chord=0.15, x_le=0.0375, airfoil="2412")], n_span=12)
    sp = SectionPolars()
    r1 = wing_polar(w, [0.0, 4.0], speed=20.0, polar_provider=sp)
    n1 = len(fake.calls)
    r2 = wing_polar(w, [0.0, 4.0], speed=20.0, polar_provider=sp)
    assert len(fake.calls) == n1 and n1 >= 2
    assert r1.points[1].cd == pytest.approx(r2.points[1].cd)
    assert any("all section polars were cached" in n for n in r2.notes)
    note = next(n for n in r1.notes if n.startswith("Viscous strip"))
    assert "2412 at Re" in note
    # a cache built for other conditions is not reused silently
    r3 = wing_polar(w, [0.0], speed=20.0, polar_provider=sp, ncrit=5.0)
    assert any("separate cache" in n for n in r3.notes)


def test_real_section_polars_end_to_end():
    """Real mfoil polars: a constant-chord NACA 0012 wing at one Reynolds number (a few seconds)."""
    w = WingSpec(sections=[WingSection(y=0, chord=0.3, airfoil="0012"),
                           WingSection(y=0.9, chord=0.3, airfoil="0012")], n_span=8, n_chord=4)
    sp = SectionPolars()
    r = wing_polar(w, [0.0, 3.0], speed=30.0, polar_provider=sp)
    assert all(p.converged for p in r.points)
    p0, p3 = r.points
    assert abs(p0.cl) < 1e-6 and p0.cdi < 1e-9
    assert 0.004 < p0.cdv < 0.008                           # NACA 0012, Re 6e5, free transition
    assert p3.cdv > p0.cdv and p3.cd == pytest.approx(p3.cdi + p3.cdv)
    (node,) = sp.node_res("0012")                           # one grid node, within half a grid step
    assert node == pytest.approx(30.0 * 0.3 * 1.225 / 1.7894e-5, rel=0.025)
    t = sp.table("0012", sp.node_res("0012")[0])
    j = int(np.argmin(np.abs(p3.y)))
    a_eff = float(t.alpha_at_cl(np.array([p3.cl_local[j]]))[0][0])
    assert p3.cd_local[j] == pytest.approx(float(t.drag(a_eff)), rel=1e-9)
    n = sp.points_computed
    wing_polar(w, [3.0], speed=30.0, polar_provider=sp)
    assert sp.points_computed == n


def test_state_round_trip(fake):
    import json

    sp = SectionPolars()
    sp.ensure({"2412": [(1e5, -2, 4), (2e5, -2, 4)]})
    state = json.loads(json.dumps(sp.state()))
    sp2 = SectionPolars()
    sp2.load_state(state)
    n = len(fake.calls)
    assert sp2.ensure({"2412": [(1.4e5, -2, 4)]}) == 0 and len(fake.calls) == n
    assert float(sp2.table("2412", 1.4e5).drag(1.0)) == pytest.approx(float(sp.table("2412", 1.4e5).drag(1.0)))
    with pytest.raises(ValueError, match="settings|computed with"):
        SectionPolars(ncrit=5.0).load_state(state)
