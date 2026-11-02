"""Grid-convergence studies (analysis.gci): Celik et al. (2008) procedure, noise, studies from results.

The reference numbers are the worked example of Celik et al. (2008), J. Fluids Eng.
130(7) 078001, Table 1 (first column: N = 18,000 / 8,000 / 4,500 cells, phi = 6.063 /
5.972 / 5.863), which every implementation of the procedure must reproduce.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.analysis import gci as G
from aerolume.model import CFD3DResult, CFDResult, CFDSettings, Simulation, Stat


def test_celik_table1_example():
    h = [(1.0 / n) ** 0.5 for n in (18000, 8000, 4500)]            # 2D: h = (1/N)^(1/2)
    r = G.gci_study(h, [6.063, 5.972, 5.863], quantity="reattachment length")
    assert r.r21 == pytest.approx(1.5) and r.r32 == pytest.approx(4 / 3)
    assert r.convergence == "monotone"
    assert r.p == pytest.approx(1.53, abs=0.005)
    assert r.phi_ext == pytest.approx(6.1685, abs=5e-4)
    assert r.e_a21 == pytest.approx(0.015, abs=5e-4)
    assert r.e_ext21 == pytest.approx(0.017, abs=5e-4)
    assert r.gci_fine == pytest.approx(0.022, abs=5e-4)
    assert r.u_num == pytest.approx(r.gci_fine * 6.063)
    assert r.asymptotic is False and any("formal order" in n for n in r.notes)   # 1.53 is not 2 +- 10 %
    assert r.noise_limited is False                                 # deterministic: no standard errors


def test_exact_power_law_recovers_order_and_limit():
    # constant ratio: p explicit; non-uniform ratios: the fixed-point iteration of eq. 3-5
    for h in ([1.0, 2.0, 4.0], [1.0, 1.4, 2.3], [0.5, 0.9, 1.0]):
        phi = [2.0 + 0.5 * x ** 1.7 for x in h]
        r = G.gci_study(h, phi, formal_order=1.7)
        assert r.p == pytest.approx(1.7, rel=1e-8)
        assert r.phi_ext == pytest.approx(2.0, abs=1e-10)
        assert r.asymptotic is True
    # input order does not matter: sorted fine to coarse
    r = G.gci_study([4.0, 1.0, 2.0], [2 + 0.5 * 16, 2.5, 2 + 0.5 * 4])
    assert [lv.h for lv in r.levels] == [1.0, 2.0, 4.0] and r.p == pytest.approx(2.0)


def test_convergence_types():
    assert G.convergence_type(1.0, 1.1, 1.3)[0] == "monotone"
    assert G.convergence_type(1.0, 1.1, 0.95)[0] == "oscillatory"
    assert G.convergence_type(1.0, 1.3, 1.4)[0] == "divergent"
    assert G.convergence_type(1.0, 0.4, 0.7)[0] == "oscillatory-divergent"
    # unequal ratios: a small coarse step after a large fine one still converges
    assert G.convergence_type(*(2 + 0.5 * x ** 1.7 for x in (0.5, 0.9, 1.0)), h=(0.5, 0.9, 1.0))[0] == "monotone"
    assert G.convergence_type(*(2 + 0.5 * x ** 1.7 for x in (0.5, 0.9, 1.0)))[0] == "divergent"   # Celik's R
    assert G.convergence_type(1.0, 1.0, 1.0)[0] == "insensitive"
    assert G.convergence_type(1.0, 1.0, 1.2)[0] == "converged-fine"
    div = G.gci_study([1, 2, 4], [1.0, 1.3, 1.4])
    assert div.convergence == "divergent" and not math.isfinite(div.u_num) and not div.usable
    assert any("do not converge" in w for w in div.warnings)
    osc = G.gci_study([1, 2, 4], [1.0, 1.1, 0.95])
    assert osc.convergence == "oscillatory" and any("oscillate" in w for w in osc.warnings)
    flat = G.gci_study([1, 2, 4], [0.0, 0.0, 0.0])
    assert flat.convergence == "insensitive" and flat.u_num == 0.0 and flat.usable


def test_noise_limited_studies_are_flagged_and_reproducible():
    h = [1.0, 1.5, 2.25]
    phi = [1.500, 1.506, 1.519]
    clean = G.gci_study(h, phi, [1e-4, 1e-4, 1e-4])
    assert clean.noise_limited is False and clean.usable
    assert clean.monte_carlo["fraction_monotone"] > 0.99
    assert clean.monte_carlo["p_p16"] < clean.p < clean.monte_carlo["p_p84"]
    assert clean.u_stat == pytest.approx(2e-4) and clean.u_total == pytest.approx(math.hypot(clean.u_num, 2e-4))
    noisy = G.gci_study(h, phi, [0.004, 0.004, 0.004])
    assert noisy.noise_limited is True and not noisy.usable and noisy.asymptotic is None
    assert 0.0 < noisy.monte_carlo["fraction_monotone"] < 0.9
    again = G.gci_study(h, phi, [0.004, 0.004, 0.004])
    assert again.monte_carlo == noisy.monte_carlo                    # fixed seed: the numbers repeat
    rough = G.gci_study(h, phi, [1e-4, 1e-4, 1e-4], reliable=[True, False, True])
    assert any("rough estimate" in w for w in rough.warnings)


def test_roy_method_falls_back_to_fs3_away_from_the_formal_order():
    h = [1.0, 2.0, 4.0]
    phi = [1.0 + 0.1 * x ** 1.0 for x in h]                          # first order, formal 2
    celik = G.gci_study(h, phi, method="celik")
    roy = G.gci_study(h, phi, method="roy")
    assert celik.fs == 1.25 and celik.p_used == pytest.approx(1.0)
    assert roy.fs == 3.0 and roy.p_used == pytest.approx(1.0)
    assert roy.u_num == pytest.approx(celik.u_num * 3.0 / 1.25)
    near = G.gci_study(h, [1.0 + 0.1 * x ** 1.95 for x in h], method="roy")
    assert near.fs == 1.25
    with pytest.raises(ValueError):
        G.gci_study(h, phi, method="bogus")


def test_two_grid_and_four_grid_studies():
    two = G.gci_two_grid([1.0, 2.0], [1.0 + 0.1, 1.0 + 0.4])            # exact second order
    assert two.fs == 3.0 and two.phi_ext == pytest.approx(1.0)
    assert two.u_num == pytest.approx(3.0 * 0.3 / 3.0)
    assert any("Two grids" in w for w in two.warnings)
    h = [1.0, 1.5, 2.25, 3.375]
    four = G.gci_study(h, [3.0 + 0.2 * x ** 2 for x in h])
    assert len(four.triplet_orders) == 2 and all(p == pytest.approx(2.0) for p in four.triplet_orders)
    assert four.fit["p"] == pytest.approx(2.0, abs=1e-3) and four.fit["phi0"] == pytest.approx(3.0, abs=1e-4)
    assert four.asymptotic is True
    bent = G.gci_study(h, [3.0 + 0.2 * h[0] ** 2, 3.0 + 0.2 * h[1] ** 2, 3.0 + 0.2 * h[2] ** 2, 3.0 + 3.0])
    assert bent.asymptotic is False                                  # the coarse triplet's order differs
    with pytest.raises(ValueError):
        G.gci_study([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        G.gci_study([1.0, 1.0, 2.0], [1.0, 1.0, 2.0])


def test_summary_and_json():
    r = G.gci_study([1, 2, 4], [1.0, 1.04, 1.16], [1e-4] * 3, quantity="Cd")
    text = r.summary()
    assert text.startswith("Cd: 1 ± ") and "observed order 1.58" in text
    d = r.to_dict()
    import json
    json.dumps(d, allow_nan=False)                                   # nan / inf become None
    assert d["levels"][0]["value"] == 1.0


def _cfd(sid: str, cpr: float, cd: float, se: float, *, grid_scale: float = 1.0, re_sim: float = 2e4) -> CFDResult:
    n = (round(cpr * 30 * grid_scale), round(cpr * 20))
    return CFDResult(sim_id=sid, cd=Stat(cd, 0.01, cd, cd, se), cl=Stat(0.3, 0.01, 0.3, 0.3, 1e-3),
                     cm=Stat(-0.05, 0.0, -0.05, -0.05, 1e-4), re_sim=re_sim, re_real=1e6, grid=n, cells_per_ref=cpr)


def test_grid_study_from_results():
    rs = [_cfd("c", 48.0, 1.546, 4e-4), _cfd("f", 108.0, 1.5115, 4e-4), _cfd("m", 72.0, 1.5231, 4e-4)]
    st = G.grid_study(rs)
    assert st.sim_ids == ["f", "m", "c"] and st.domain_fixed is True and st.dimension == 2
    assert set(st.quantities) == {"cd", "cl", "cm"}
    q = st.quantities["cd"]
    assert q.convergence == "monotone" and q.levels[0].sim_id == "f" and q.levels[0].cells == 108 * 30 * 108 * 20
    assert st.quantities["cl"].convergence == "insensitive"
    st2 = G.grid_study({"f": rs[1], "m": rs[2], "c": _cfd("c", 48.0, 1.546, 4e-4, grid_scale=1.3)})
    assert st2.domain_fixed is False and any("domain size changes" in w for w in st2.warnings)
    with pytest.raises(ValueError, match="Reynolds"):
        G.grid_study([rs[0], rs[1], _cfd("x", 72.0, 1.52, 4e-4, re_sim=4e4)])
    with pytest.raises(ValueError, match="same resolution"):
        G.grid_study([rs[0], rs[1], _cfd("y", 108.0, 1.52, 4e-4)])
    r3 = CFD3DResult(sim_id="z", cells_per_ref=10.0, grid=(10, 10, 10))
    with pytest.raises(ValueError, match="mix"):
        G.grid_study([rs[0], rs[1], r3])


def _sim(sid: str, design: str, cpr: int, *, speed: float = 30.0, quality: str = "custom") -> Simulation:
    s = Simulation(name=f"run {sid}", design_id=design, id=sid,
                   cfd=CFDSettings(quality=quality, cells_per_ref=cpr))
    s.conditions.speed = speed
    return s


def test_find_grid_studies_groups_by_setup():
    sims = {s.id: s for s in (_sim("a", "d1", 32), _sim("b", "d1", 48), _sim("c", "d1", 64),
                              _sim("d", "d1", 48, speed=40.0), _sim("e", "d2", 32), _sim("f", "d2", 48))}
    res = {sid: _cfd(sid, float(s.cfd.cells_per_ref), 1.0, 1e-3) for sid, s in sims.items()}
    groups = G.find_grid_studies(sims, res)
    assert groups == [["c", "b", "a"]]                     # d differs in speed, d2 has two grids only
    assert G.find_grid_studies(sims, {k: v for k, v in res.items() if k != "b"}) == []
    # quality presets differ only in resolution settings and are grouped (the domain check then decides)
    sims2 = {s.id: s for s in (_sim("p", "d3", 32, quality="draft"), _sim("q", "d3", 48, quality="standard"),
                               _sim("r", "d3", 64, quality="fine"))}
    res2 = {sid: _cfd(sid, float(s.cfd.cells_per_ref), 1.0, 1e-3) for sid, s in sims2.items()}
    assert G.find_grid_studies(sims2, res2) == [["r", "q", "p"]]


def test_observed_order_bisection_fallback():
    # a strongly non-uniform pair of ratios, solved by iteration or the bisection fallback
    h = [1.0, 1.05, 4.0]
    phi = [1 + 0.3 * x ** 1.3 for x in h]
    assert G.observed_order(h, phi) == pytest.approx(1.3, rel=1e-6)
    assert math.isnan(G.observed_order([1, 2, 4], [1.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        G.observed_order([2, 1, 4], [1, 2, 3])
    assert np.isnan(G.richardson(1.0, 2.0, 2.0, float("nan")))
