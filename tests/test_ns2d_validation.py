"""Validation against published cylinder data, and the moving-ground LES smoke test.

Slow (tens of seconds each on an RTX 4060 Ti), so they are marked ``slow`` and
run only with ``pytest -m slow``. They need a CUDA device (CPU would take
hours at these resolutions). The bands are those of the Aerolume solver
specification; reference values are listed in validation/ns2d_cases.py and the
measured numbers at several resolutions in docs/theory_ns2d.md.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns2d import FlowSolver


def _load_cases():
    """validation/ns2d_cases.py (not a package), loaded once per session.

    Every ns2d test module shares one module object under one name: re-executing
    the file per test module would leave dataclasses pointing at a stale module,
    and a failed load must not leave a half-initialised entry in sys.modules.
    """
    path = Path(__file__).resolve().parents[1] / "validation" / "ns2d_cases.py"
    mod = sys.modules.get("ns2d_cases")
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("ns2d_cases", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ns2d_cases"] = mod           # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("ns2d_cases", None)
        raise
    return mod


cases = _load_cases()

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_cylinder_re40_steady():
    """Wide domain (blockage 5 %, slip walls), 32 cells per diameter."""
    r = cases.run_cylinder(40.0, 32, 60.0, 50.0, device="cuda")
    v = r.values
    assert 1.50 <= v["cd"] <= 1.62, r.row()
    assert 2.1 <= v["L/D"] <= 2.3, r.row()
    assert abs(v["cl_mean"]) < 0.01, r.row()
    # surface() distributions integrate to the solver force and look physical:
    # viscous stagnation cp slightly above 1, separation ~53 deg from the rear
    assert abs(v["cd_surface"] / v["cd"] - 1.0) < 0.02, r.row()
    assert 1.0 < v["cp_front"] < 1.25 and -0.75 < v["cp_rear"] < -0.4, r.row()
    assert 122.0 < v["sep_deg"] < 131.0, r.row()


def test_cylinder_re100_shedding():
    r = cases.run_cylinder(100.0, 32, 200.0, 100.0, device="cuda")
    v = r.values
    assert 1.30 <= v["cd"] <= 1.42, r.row()
    assert 0.30 <= v["cl_amp"] <= 0.36, r.row()
    assert 0.160 <= v["st"] <= 0.172, r.row()
    assert abs(v["cd_surface"] / v["cd"] - 1.0) < 0.04, r.row()      # thinner layers: see theory doc


def test_shedding_starts_promptly():
    """The deterministic start-up perturbation locks in shedding within ~40
    convective times at Re = 100 (round-off alone takes 100+)."""
    cfg = cases.cylinder_config(100.0, 24, device="cuda")
    s = FlowSolver(cfg)
    while s.t < 60.0:
        s.advance(500)
    h = s.history()
    late = h["t"] > 40.0
    assert 0.5 * (h["cl"][late].max() - h["cl"][late].min()) > 0.25


def test_moving_ground_les_re1e6_stable():
    """Rounded box with rolling wheels on a moving road, upwind3 + WALE at Re = 1e6,
    256 cells per length, 6 convective times.

    A robustness test. 2D flow at this Re is under-resolved and dominated by
    intense 2D vortices (instantaneous peaks of 3-4.5 U above the roof and in
    the wake, theory doc section 11), and the touching wheels seal the
    underbody, so only boundedness and sanity are asserted, not values.
    """
    cfg = cases.moving_ground_config(256, device="cuda")
    s = FlowSolver(cfg)
    peak = 0.0
    while s.t < 6.0:
        s.advance(500)
        peak = max(peak, float(torch.hypot(s.field("u"), s.field("v")).max()))
    assert np.isfinite(peak) and peak < 6.0, peak
    h = s.history()
    late = h["t"] > 3.0
    assert np.all(np.isfinite(h["cd"])) and np.all(np.isfinite(h["cl"]))
    assert 0.1 < h["cd"][late].mean() < 6.0, h["cd"][late].mean()
    assert float(s.field("divergence").abs().max()) * cfg.h < 1e-5
    assert float(s.field("nut").max()) > 0.0
    assert np.all(np.isfinite(s.body_coefficients()))
    for surf in s.surface(200):
        assert np.all(np.isfinite(surf["cp"])) and np.all(np.isfinite(surf["cf"]))


def test_schafer_turek_2d1_steady_channel():
    """Benchmark 2D-1 (Re 20, no-slip channel, parabolic inflow) at 40 cells per diameter.

    Measured (docs/validation.md section 3): Cd 5.585, Cl 0.0103, La 0.0849 m, dp 0.1154 Pa
    against 5.57-5.59, 0.0104-0.0110, 0.0842-0.0852 and 0.1172-0.1176. The bands allow
    for the immersed (not body-fitted) wall at this resolution.
    """
    r = cases.run_schafer_turek("2D-1", 40, device="cuda")
    v = r.values
    assert 5.50 <= v["cd"] <= 5.66, r.row()
    assert 0.0080 <= v["cl"] <= 0.0130, r.row()
    assert 0.0830 <= v["La"] <= 0.0865, r.row()
    assert 0.112 <= v["dp"] <= 0.121, r.row()
    assert v["cd_drift"] < 1e-3, r.row()                      # steady by the end of the run


def test_schafer_turek_2d2_periodic_channel():
    """Benchmark 2D-2 (Re 100, periodic shedding) at 40 cells per diameter.

    Measured: St 0.300, Cd,max 3.306, Cl,max 1.034 against 0.295-0.305, 3.22-3.24 and
    0.99-1.01 (the peaks are 2-3 % high at this resolution, see the validation report).
    """
    r = cases.run_schafer_turek("2D-2", 40, device="cuda")
    v = r.values
    assert 0.290 <= v["st"] <= 0.310, r.row()
    assert 3.15 <= v["cd_max"] <= 3.40, r.row()
    assert 0.95 <= v["cl_max"] <= 1.08, r.row()
