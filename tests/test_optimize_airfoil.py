"""Airfoil optimisation (CST perturbations + SLSQP).

Fast tests drive the optimiser with a cheap analytic stand-in for the panel
solver (thin-airfoil lift, a quadratic drag polar), which exercises the
parametrisation, constraints, failure handling, cancellation and determinism
in well under a second per run. The slow test runs the real viscous solver on
NACA 0012 at Re 1e6, 4 degrees.
"""
from __future__ import annotations

import itertools
import math
import threading

import numpy as np
import pytest

from aerolume.geometry import naca
from aerolume.model import PolarPoint
from aerolume.optimize import OptimizationResult, airfoil_metrics, optimize_airfoil
from aerolume.optimize.airfoil import LOCAL_RANGE, LOCAL_THICKNESS, _Shape


def surrogate(*, fail_camber: float | None = None, counter: list | None = None, cancel_after: int | None = None,
              event: threading.Event | None = None):
    """Thin-airfoil Cl plus a drag polar with an optimum camber; optionally fails above a camber."""
    def fn(coords, re, alpha):
        if counter is not None:
            counter.append(1)
            if cancel_after is not None and len(counter) >= cancel_after and event is not None:
                event.set()
        m = airfoil_metrics(coords)
        camber, t = m["max_camber"], m["max_thickness"]
        if fail_camber is not None and camber > fail_camber:
            return PolarPoint(alpha=alpha, converged=False)
        cl = 2 * math.pi * math.radians(alpha) * (1 + 0.77 * t) + 4 * math.pi * camber
        # a drag bucket centred on the camber's design lift: camber pays at a fixed Cl too
        cd = 0.0055 + 0.12 * (t - 0.09) ** 2 + 0.01 * (cl - 12.0 * camber) ** 2 + 0.3 * camber ** 2
        return PolarPoint(alpha=alpha, cl=cl, cd=cd, cm=-math.pi * camber, converged=True,
                          xtr_upper=0.5, xtr_lower=0.6)
    return fn


def local_thickness_ok(coords0, coords1) -> bool:
    s0, s1 = _Shape(coords0, 2), _Shape(coords1, 2)
    sel = s0.local
    return bool(np.all(s1.t0[sel] >= LOCAL_THICKNESS * s0.t0[sel] - 1e-9))


def test_surrogate_ld_improves_within_constraints():
    c0 = naca("0012")
    r = optimize_airfoil(c0, 1e6, alpha=4.0, objective="ld", n_iter=6, analyze_fn=surrogate())
    assert isinstance(r, OptimizationResult) and r.method == "airfoil-cst-slsqp"
    assert r.improvement > 0.05
    assert r.best["ld"] > r.initial["ld"] * 1.05
    t0 = r.initial["max_thickness"]
    assert r.best["max_thickness"] >= t0 - 1e-9                   # default min thickness: the original's
    assert local_thickness_ok(c0, r.best_coords)
    assert r.history[0]["iter"] == 0 and len(r.history) >= 2
    assert r.cst is not None and r.cst.residual < 2e-3
    q = r.best_coords
    assert q[0, 0] == pytest.approx(1.0) and np.hypot(*q[np.argmin(q[:, 0])]) < 1e-12   # TE and LE fixed
    assert r.message.startswith("L/D")


def test_deterministic():
    a = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, n_iter=4, analyze_fn=surrogate())
    b = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, n_iter=4, analyze_fn=surrogate())
    assert np.array_equal(a.x, b.x) and np.array_equal(a.best_coords, b.best_coords)
    assert [h["objective"] for h in a.history] == [h["objective"] for h in b.history]


def test_target_cl_minimises_drag_at_the_required_lift():
    r = optimize_airfoil(naca("0012"), 1e6, target_cl=0.6, objective="ld", n_iter=8, analyze_fn=surrogate())
    assert r.objective == "cd"
    assert r.best["cl"] == pytest.approx(0.6, abs=2e-3)
    assert r.initial["cl"] == pytest.approx(0.6, abs=2e-3)       # the starting angle was found by secant
    assert r.best["cd"] < r.initial["cd"]
    assert r.best["alpha"] != r.initial["alpha"]                   # the angle is a variable


def test_failed_analyses_are_never_returned():
    calls: list = []
    r = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, n_iter=6, analyze_fn=surrogate(fail_camber=0.008,
                                                                                      counter=calls))
    assert r.failed_evaluations > 0 and r.evaluations == len(calls)
    assert airfoil_metrics(r.best_coords)["max_camber"] <= 0.008 + 1e-12
    assert r.improvement > 0
    assert any("did not converge" in w for w in r.warnings)


def test_keep_area_and_min_thickness():
    c0 = naca("0012")
    a0 = airfoil_metrics(c0)["area"]
    r = optimize_airfoil(c0, 1e6, alpha=4.0, n_iter=5, keep_area=True, min_thickness=0.13,
                         analyze_fn=surrogate())
    assert r.best["area"] >= a0 * (1 - 1e-9)
    assert airfoil_metrics(r.best_coords)["area"] == pytest.approx(r.best["area"], rel=1e-3)
    assert r.best["max_thickness"] >= 0.13 - 1e-9
    assert any("above the starting section" in w for w in r.warnings)


def test_cancel_returns_best_so_far():
    ev = threading.Event()
    calls: list = []
    r = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, n_iter=20, cancel=ev,
                         analyze_fn=surrogate(counter=calls, cancel_after=25, event=ev))
    assert r.extra["cancelled"] and not r.converged
    assert any("Stopped by the user" in w for w in r.warnings)
    assert len(calls) <= 26
    assert r.best["ld"] >= r.initial["ld"]


def test_progress_is_monotone_and_bounded():
    seen: list[float] = []
    optimize_airfoil(naca("2412"), 5e5, alpha=2.0, n_iter=2, analyze_fn=surrogate(),
                     progress=lambda f, m: seen.append(f))
    assert seen and all(0.0 <= f <= 1.0 for f in seen)
    assert all(a <= b for a, b in itertools.pairwise(seen))


def test_argument_checks():
    c = naca("0012")
    with pytest.raises(ValueError):
        optimize_airfoil(c, 1e6, alpha=2.0, target_cl=0.5, analyze_fn=surrogate())
    with pytest.raises(ValueError):
        optimize_airfoil(c, 1e6, analyze_fn=surrogate())
    with pytest.raises(ValueError):
        optimize_airfoil(c, 1e6, target_cl=0.5, objective="cl", analyze_fn=surrogate())
    with pytest.raises(ValueError):
        optimize_airfoil(c, 1e6, alpha=2.0, objective="moment", analyze_fn=surrogate())
    with pytest.raises(ValueError):                                # the start must converge
        optimize_airfoil(naca("4412"), 1e6, alpha=2.0, analyze_fn=surrogate(fail_camber=0.0))


def test_to_dict_is_json_safe():
    import json
    r = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, n_iter=1, analyze_fn=surrogate())
    d = json.loads(json.dumps(r.to_dict()))
    assert d["method"] == "airfoil-cst-slsqp" and len(d["best_coords"]) == len(r.best_coords)
    assert d["cst"]["upper"] and LOCAL_RANGE[0] < LOCAL_RANGE[1]


@pytest.mark.slow
def test_naca0012_ld_improves_with_the_viscous_solver():
    """Real mfoil analyses: L/D of NACA 0012 at Re 1e6, 4 deg rises within two SLSQP iterations
    (measured: 57.9 -> 75.1 after 2, 86.8 after 3 iterations; ~11 analyses per iteration)."""
    r = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, objective="ld", n_iter=2)
    assert r.initial["ld"] == pytest.approx(57.9, rel=0.02)
    assert r.best["ld"] > 1.10 * r.initial["ld"]
    assert r.best["max_thickness"] >= r.initial["max_thickness"] - 1e-9
    assert local_thickness_ok(naca("0012"), r.best_coords)
    assert r.failed_evaluations <= r.evaluations // 4
    # deterministic: the solver and SLSQP have no randomness
    r2 = optimize_airfoil(naca("0012"), 1e6, alpha=4.0, objective="ld", n_iter=1)
    assert r2.history[1]["objective"] == pytest.approx(r.history[1]["objective"], rel=1e-12)
