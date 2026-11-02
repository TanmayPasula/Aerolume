"""3D validation against published data (slow, GPU). Run with ``pytest -m slow tests/test_ns3d_validation.py``.

Numbers and their discussion: docs/theory_ns3d.md section 10. The bands are the
DESIGN 7.12 quality bars where it states them, otherwise the reference value
with the margin the confined domain (2.9 % blockage) and the grid explain.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def _load_cases():
    path = Path(__file__).resolve().parents[1] / "validation" / "ns3d_cases.py"
    mod = sys.modules.get("ns3d_cases")
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location("ns3d_cases", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ns3d_cases"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("ns3d_cases", None)
        raise
    return mod


cases = _load_cases()


@pytest.fixture(autouse=True)
def _free_gpu():
    yield
    torch.cuda.empty_cache()


def test_sphere_re100_drag_and_wake_at_24_cells_per_diameter():
    """DESIGN 7.12: Re 100, >= 24 cells/D, <= ~3 % blockage: Cd 1.05-1.15, L/D 0.8-0.95."""
    r = cases.run_sphere(100.0, 24, 40.0, device="cuda")
    v = r.values
    assert v["blockage"] <= 0.031
    assert 1.05 <= v["cd"] <= 1.15, r.row()
    assert 0.80 <= v["L/D"] <= 0.95, r.row()
    assert abs(v["sep_deg"] - cases.separation_angle_ref(100.0)) < 3.0     # 126.5 deg (Taneda fit)
    assert v["cd_drift"] < 2e-3                                   # steady
    assert abs(v["cd_cv"] / v["cd"] - 1) < 0.01                    # control volume = impulse
    assert abs(v["cl"]) < 1e-3                                     # axisymmetric


@pytest.mark.parametrize("re", [20.0, 50.0])
def test_sphere_low_re_drag_is_grid_converged_and_near_the_standard_curve(re):
    """Re 20 and 50 at 16 and 24 cells/D (same domain): the drag changes by < 1 % with the
    grid (measured 0.1 % and 0.4 %) and lies within 6 % of the unconfined Clift et al. curve;
    the measured excess (~5 %) shrinks as the box is widened (docs/theory_ns3d.md 10), i.e.
    it is the confinement of a 2.85 %-blockage box, which matters most at low Re."""
    lo = cases.run_sphere(re, 16, 30.0, device="cuda").values
    hi = cases.run_sphere(re, 24, 30.0, device="cuda").values
    ref = cases.clift_cd(re)
    assert abs(hi["cd"] / lo["cd"] - 1) < 0.01, (lo["cd"], hi["cd"])
    assert 0.0 < hi["cd"] / ref - 1 < 0.06, (lo["cd"], hi["cd"], ref)
    assert abs(hi["cd_cv"] / hi["cd"] - 1) < 0.01
    if re == 50.0:
        assert 0.30 < hi["L/D"] < 0.55                             # Taneda / Johnson & Patel: ~0.4


def test_symmetry_plane_run_equals_full_domain_run():
    """DESIGN 7.12: a half-domain run with the symmetry plane matches the full domain within 1 %."""
    half = cases.run_sphere(100.0, 16, 30.0, device="cuda", half=True).values
    full = cases.run_sphere(100.0, 16, 30.0, device="cuda", half=False).values
    assert abs(half["cd"] / full["cd"] - 1) < 0.01
    assert abs(half["L/D"] - full["L/D"]) < 0.02
    assert abs(half["cd_surface"] / full["cd_surface"] - 1) < 0.01


def test_moving_road_wheels_wale_re1e6_is_stable():
    """Moving road, rolling wheels, WALE, Re 1e6, symmetry plane: no blow-up, divergence at
    round-off, eddy viscosity active, bounded velocities."""
    r = cases.run_moving_road(64, 6.0, device="cuda")
    v = r.values
    assert v["finite"], r.row()
    assert v["divergence_h"] < 1e-5
    assert v["nut_max"] > 0
    assert v["max_speed"] < 3.0
    assert 0.2 < v["cd_mean"] < 2.0 and math.isfinite(v["cl_mean"])


def test_sphere_re300_periodic_wake():
    """Stretch case (Johnson & Patel 1999): Re 300 sheds periodically in the symmetry plane with
    mean Cd 0.656, mean lift 0.069 and St 0.137. At 24 cells/D the boundary layer has ~1.5
    cells and the box adds ~3 %: measured Cd +11 %, St -4 %, lift +4 % (32 cells/D: +8 %,
    -1.5 %, +4 %; docs/theory_ns3d.md 10)."""
    r = cases.run_sphere_unsteady(300.0, 24, device="cuda")
    v = r.values
    assert 0.0 < v["cd"] / 0.656 - 1 < 0.15, r.row()
    assert abs(v["st"] / cases.SPHERE_RE300_ST - 1) < 0.08, r.row()
    assert abs(v["cl_mean"] / 0.069 - 1) < 0.20, r.row()
    assert v["cl_amp"] > 0.005                                     # it does shed
