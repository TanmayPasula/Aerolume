"""Finite-wing solvers vs theory, lifting-surface results and wind-tunnel data.

The cases and the transcribed reference values live in validation/wing_validation.py
(which also renders the figures and tables of docs/validation_wing.md); these tests
assert on the same numbers with the tolerances of its ``TOL`` table. The inviscid
cases run in a few seconds; the viscous NACA TN 1270 wings and the timing target
need real section polars and are marked slow.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "validation" / "wing_validation.py"
_spec = importlib.util.spec_from_file_location("wing_validation", _PATH)
V = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("wing_validation", V)      # dataclasses resolve their module by name
_spec.loader.exec_module(V)
TOL = V.TOL


@pytest.fixture(scope="module")
def elliptic():
    return V.case_elliptic()


@pytest.fixture(scope="module")
def lifting_surface():
    return V.case_lifting_surface()


def test_elliptic_wings_lifting_line_theory(elliptic):
    for r in elliptic:
        assert abs(r["llt"] / r["formula"] - 1) < TOL["elliptic_llt_cla"], r
        assert r["e_llt"] >= TOL["elliptic_e"] and r["e_vlm"] >= TOL["elliptic_e"], r


def test_elliptic_wings_lattice_tends_to_lifting_line_with_aspect_ratio(elliptic):
    """A lifting surface carries less lift than a lifting line, less so as AR grows (the 2 % bar on
    2 pi AR/(AR+2) is met by the lifting line; the lattice is checked against lifting-surface data)."""
    gaps = [1 - r["vlm"] / r["formula"] for r in elliptic]
    assert all(g > 0 for g in gaps) and gaps == sorted(gaps, reverse=True)
    assert gaps[-1] < 0.03                                   # AR 12


def test_lattice_vs_lifting_surface_references(lifting_surface):
    ls = lifting_surface
    assert abs(ls["circle"] / V.MULTHOPP_CIRCLE - 1) < TOL["lifting_surface"]
    for ar, vals in V.FALKNER_RECT.items():
        tol = TOL["lifting_surface"] if ar >= 4 else TOL["lifting_surface_low_ar"]
        assert abs(ls[f"rect{ar}"] / np.mean(vals) - 1) < tol, (ar, ls[f"rect{ar}"])
    assert abs(ls["rect6_llt"] / V.FALKNER_LL_AR6 - 1) < 0.01


def test_swept_wings_vs_falkner(lifting_surface):
    for (ar, sw), vals in V.FALKNER_SWEPT.items():
        assert abs(V.range_deviation(lifting_surface[f"swept{ar}_{sw}"], vals)) < TOL["swept"], (ar, sw)


def test_weber_brebner_swept_wing():
    wb = V.case_weber()
    rel = np.abs(wb["cl"] / V.WEBER_CL - 1)
    assert np.all(rel < TOL["weber_cl"]), rel
    inner = (V.WEBER_Y2B >= 0.04) & (V.WEBER_Y2B <= 0.9)
    assert np.max(np.abs(wb["span_cl"] - V.WEBER_CL_42)[inner]) < TOL["weber_span"]


def test_bertin_smith_coarse_lattice_is_reproduced():
    assert abs(V.case_bertin_smith() / V.BERTIN_SMITH_CLA - 1) < TOL["bertin_smith"]


def test_tn1270_planforms_match_table_one():
    for w in V.TN1270.values():
        m = V.planform_metrics(V.tn1270_wing(w))
        assert m["area"] / V.FT ** 2 == pytest.approx(w.area_ft2, rel=1e-6)
        assert m["mac"] / V.FT == pytest.approx(w.mac_ft, rel=0.01)
        assert m["aspect_ratio"] == pytest.approx(w.aspect_ratio, rel=0.003)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(V.TN1270))
def test_tn1270_viscous_wings(name):
    c = V.case_tn1270(name)
    w = c.wing
    sv, sl = c.summary("vlm"), c.summary("llt")
    assert sv["converged"] == sv["total"] and sl["converged"] == sl["total"]
    assert abs(sv["cl_alpha"] / w.cl_alpha - 1) < TOL["tn_cla_vlm"]
    assert abs(sl["cl_alpha"] / w.cl_alpha - 1) < TOL["tn_cla_llt"]
    for s in (sv, sl):
        assert abs(s["alpha_l0"] - w.alpha_l0) < TOL["tn_alpha_l0"]
        assert abs(s["cd_min"] / w.cd_min - 1) < TOL["tn_cd_min"]
        assert abs(s["ld_max"] / w.ld_max - 1) < TOL["tn_ld_max"]
    assert abs(sv["cm0"] - w.cm0) < TOL["tn_cm0_vlm"]
    assert abs(sl["cm0"] - w.cm0) < TOL["tn_cm0_llt"]
    # CL_max is NOT validated: the section polars stall late, so the critical-section estimate is high
    lo, hi = TOL["tn_cl_max"]
    assert math.isfinite(sv["cl_max"]) and lo < sv["cl_max"] - w.cl_max < hi
    assert all(p.cl > w.cl_max for p in c.llt.points if p.alpha >= 18)


@pytest.mark.slow
def test_performance_target():
    p = V.case_performance()
    assert p["converged"] == p["n"] == 17
    assert p["cold"] < 45.0                     # target ~30 s on the development machine (shared CPU)
    assert p["warm"] < 3.0 and p["same"]
