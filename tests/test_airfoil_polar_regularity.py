"""Shape of the lift and moment curves: regular where the model is linear, XFOIL's kink where it is not.

A NACA 2412 polar at Re 1e6 has an uneven lift slope between 2 and 5 deg
(dCl/dalpha 0.11 -> 0.15 -> 0.09 /deg) and a Cm dip of ~0.01 at 4 deg. It
looks like a numerical artefact, but it is not one: the pristine mfoil run
cold at each angle and XFOIL 6.99 run on the same coordinates give the same
curve to within 0.006 in Cl. The cause is the pressure-side transition point
running off the trailing edge: just before it does, that boundary layer
separates laminar a few per cent of chord ahead of the trailing edge and its
displacement thickness acts as extra aft camber (docs/validation_airfoil.md).

So these tests pin the curve *shape* against XFOIL (``XFOIL_REF`` in
validation/airfoil_validation.py, computed on the identical geometry) instead
of demanding a constant slope everywhere, and check the lift slope is near
constant below the run-off, Cm is continuous, warm and cold starts agree, and
the polar notes name the run-off angles. A real artefact (a spurious Newton
branch, warm-start contamination, a transition point stuck on the wrong panel)
shows up as a jump of 0.02-0.1 in Cl at one angle and fails the XFOIL
comparison. The full 3 sections x 2 Reynolds numbers matrix is ``slow``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from aerolume.model import PolarPoint
from aerolume.solvers.airfoil import polar
from aerolume.solvers.airfoil import solver as S

_PATH = Path(__file__).resolve().parents[1] / "validation" / "airfoil_validation.py"
_spec = importlib.util.spec_from_file_location("airfoil_validation", _PATH)
V = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("airfoil_validation", V)      # dataclasses resolve their module by name
_spec.loader.exec_module(V)

# Tolerances against XFOIL on identical geometry. Measured worst cases over the
# six polars at 1-deg steps (-4..12 deg): Cl 0.012, Cm 0.0025, Cd 2.4 %, interval
# slope 0.008 /deg; the residual difference grows smoothly with alpha (XFOIL's
# wake and TE treatment differ in detail from mfoil's).
TOL_CL, TOL_CM, TOL_CD, TOL_SLOPE = 0.02, 0.004, 0.035, 0.015
RUNOFF_START = 0.85          # lower-surface transition aft of this x/c: run-off stretch begins


def _check_against_xfoil(c) -> None:
    rows = V.compare_xfoil(c)
    assert len(rows) >= 0.7 * len(c.result.points), "too few angles in common with the XFOIL reference"
    for a, clx, cl, cdx, cd, cmx, cm, _xbx, _xb in rows:
        assert abs(cl - clx) < TOL_CL, (c.name, a, cl, clx)
        assert abs(cm - cmx) < TOL_CM, (c.name, a, cm, cmx)
        assert abs(cd / cdx - 1) < TOL_CD, (c.name, a, cd, cdx)
    sl = V.xfoil_slopes(c)
    assert len(sl) >= 3
    for am, sx, sc in sl:
        assert abs(sc - sx) < TOL_SLOPE, (c.name, am, sc, sx)


def _regular_range(c) -> tuple[float, float]:
    """Angles between the upper- and lower-surface run-off stretches (where Cl is linear)."""
    t = c.table
    a, xu, xl = t["alpha"], t["xtr_upper"], t["xtr_lower"]
    hi = a[(a >= 0) & (xl >= RUNOFF_START)]
    lo = a[(a <= 0) & (xu >= RUNOFF_START)]
    return (float(lo.max()) if len(lo) else float(a.min()), float(hi.min()) if len(hi) else float(a.max()))


def _check_regular(c) -> None:
    t = c.table
    a, cl, cm = t["alpha"], t["cl"], t["cm"]
    assert np.all(np.diff(a) > 0)
    s = np.diff(cl) / np.diff(a)
    # Cl rises monotonically in the attached range; Cm has no jumps
    assert np.all(s[a[1:] <= 10.0 + 1e-9] > 0.04), (c.name, s)
    assert np.all(np.abs(np.diff(cm)) < 0.012 * np.diff(a)), (c.name, np.diff(cm))
    # near-constant slope below the run-off stretch, at a thin-airfoil-like value
    lo, hi = _regular_range(c)
    m = (a[:-1] >= lo - 1e-9) & (a[1:] <= hi + 1e-9)
    assert m.sum() >= 3, (c.name, lo, hi)
    med = float(np.median(s[m]))
    assert 0.095 < med < 0.115, (c.name, med)
    assert np.all(np.abs(s[m] / med - 1) < 0.15), (c.name, s[m], med)


# ---------------------------------------------------------------------------
# fast: one polar through the run-off stretch
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def n2412_1e6():
    return V.run("NACA 2412, Re 1e6, 0..6 deg", "2412", 1e6, np.arange(0.0, 6.01, 1.0))


def test_2412_re1e6_follows_xfoil_through_the_kink(n2412_1e6):
    c = n2412_1e6
    assert c.n_failed() == 0
    _check_against_xfoil(c)
    t = c.table
    s = np.diff(t["cl"])
    # the kink is there (as in XFOIL): steeper over 2..4 deg than over 0..2 and 4..6 deg
    assert s[2:4].mean() > 1.2 * s[0:2].mean() and s[2:4].mean() > 1.2 * s[4:6].mean(), s
    # ... and Cm is most nose-down at 4 deg, by about 0.01, not a jump
    cm = t["cm"]
    assert int(np.argmin(cm)) == 4 and cm[2] - cm[4] < 0.02
    assert any("transition reaches the trailing edge between" in n for n in c.result.notes)


def test_cold_start_in_the_kink_matches_the_sweep(n2412_1e6):
    """No warm-start contamination: each angle solved from scratch gives the swept value."""
    t = n2412_1e6.table
    xy = V.naca4("2412", 200)
    for a in (4.0,):            # the kink peak; 3 and 5 deg agree too (~5 s each, so not repeated)
        p = polar(xy, 1e6, [a], distributions=False).points[0]
        i = int(np.flatnonzero(t["alpha"] == a)[0])
        assert p.converged
        assert abs(p.cl - t["cl"][i]) < 2e-3 and abs(p.cm - t["cm"][i]) < 5e-4, (a, p.cl, t["cl"][i])


def _pt(a: float, xu: float, xl: float) -> PolarPoint:
    return PolarPoint(alpha=a, cl=0.1 * a, cd=0.01, cm=-0.05, xtr_upper=xu, xtr_lower=xl, converged=True)


def test_runoff_note_names_the_angles():
    pts = [_pt(0, 0.6, 0.70), _pt(1, 0.55, 0.85), _pt(2, 0.5, 0.96), _pt(3, 0.45, 0.99), _pt(4, 0.4, 1.0)]
    notes = S._transition_runoff_notes(pts[::-1])        # order of the input does not matter
    assert len(notes) == 1 and notes[0].startswith("Lower-surface transition reaches the trailing edge "
                                                   "between 2° and 4°")
    # upper surface on the negative side, a failed point in between is skipped
    pts = [_pt(-4, 1.0, 0.1), PolarPoint(alpha=-3, converged=False), _pt(-2, 0.97, 0.2), _pt(0, 0.6, 0.6)]
    notes = S._transition_runoff_notes(pts)
    assert len(notes) == 1 and "Upper-surface" in notes[0] and "between -4° and -2°" in notes[0]


def test_no_runoff_note_without_runoff():
    assert S._transition_runoff_notes([_pt(0, 0.6, 0.7), _pt(2, 0.5, 0.9)]) == []
    assert S._transition_runoff_notes([_pt(0, 1.0, 1.0), _pt(2, 1.0, 1.0)]) == []    # laminar throughout
    assert S._transition_runoff_notes([]) == []


# ---------------------------------------------------------------------------
# slow: the 0012 / 2412 / 4412 x Re 1e6 / 3e6 matrix
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("code, re", V.XFOIL_CASES, ids=[f"{c}-Re{r:.0e}" for c, r in V.XFOIL_CASES])
def test_polar_shape_matches_xfoil_and_is_regular(code, re):
    c = V.case_xfoil(code, re, step=1.0)
    assert c.n_failed() == 0, c.result.warnings
    _check_against_xfoil(c)
    _check_regular(c)
    assert any("transition reaches the trailing edge between" in n for n in c.result.notes), c.result.notes
