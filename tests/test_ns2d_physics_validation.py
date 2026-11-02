"""Validation of the 2D placed physics against published benchmarks and theory.

Slow (seconds to a minute each on an RTX 4060 Ti), so marked ``slow`` and ``gpu``;
run with ``pytest -m slow tests/test_ns2d_physics_validation.py``. The cases live in
``validation/thermal2d_cases.py`` (which also holds the reference values and their
sources); docs/theory_ns2d.md section 21 lists the measured numbers at several
resolutions, with grid-convergence estimates. Each tolerance below is the band the
measured value must stay in at the resolution used here, chosen from those tables
(the converged results are much closer than the bands).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load(name: str):
    """A validation/ module (not a package), loaded once per session under its own name."""
    path = Path(__file__).resolve().parents[1] / "validation" / f"{name}.py"
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                   # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


cases = _load("thermal2d_cases")

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.mark.parametrize("ra,n,tol_nu,tol_vel", [(1e3, 64, 0.005, 0.01), (1e4, 64, 0.008, 0.01),
                                                 (1e5, 128, 0.006, 0.006), (1e6, 128, 0.012, 0.012)])
def test_de_vahl_davis_cavity(ra, n, tol_nu, tol_vel):
    """Differentially heated square cavity, Pr 0.71: mean Nusselt number, peak
    mid-plane velocities and their positions against de Vahl Davis (1983), Table V."""
    r = cases.run_cavity(ra, n, device="cuda")
    v, ref = r.values, cases.DE_VAHL_DAVIS[ra]
    assert abs(v["energy_balance"]) < 1e-5, r.row()                    # hot-wall heat = cold-wall heat
    assert v["nu_hot"] == pytest.approx(ref["nu_bar"], rel=tol_nu), r.row()
    assert v["nu_half"] == pytest.approx(ref["nu_half"], rel=tol_nu), r.row()
    assert v["u_max"] == pytest.approx(ref["u_max"], rel=tol_vel), r.row()
    assert v["v_max"] == pytest.approx(ref["v_max"], rel=tol_vel), r.row()
    assert v["u_max_y"] == pytest.approx(ref["u_max_y"], abs=1.5 / n), r.row()
    assert v["v_max_x"] == pytest.approx(ref["v_max_x"], abs=1.5 / n), r.row()


@pytest.mark.parametrize("re", [20.0, 40.0])
def test_heated_cylinder_nusselt(re):
    """Isothermal cylinder in cross flow, Pr 0.71, 16 cells/D, 5 % blockage: the mean
    Nusselt number from the penalisation heat rate against the unconfined references
    tabulated by Singhal et al. (2023). The 5 % blockage raises Nu by about 1.3 %
    (measured, theory doc 21.2), so the band is +-4 % around Biswas & Sarkar (2009)."""
    r = cases.run_heated_cylinder(re, 16, device="cuda")
    v = r.values
    ref = cases.CYLINDER_NU_REF[re]["Biswas & Sarkar 2009"]
    assert v["nu"] == pytest.approx(ref, rel=0.04), r.row()
    assert v["nu_surface"] == pytest.approx(v["nu"], rel=0.05), r.row()     # wall-gradient Nu agrees
    assert v["nu_front"] > v["nu"] > v["nu_rear"] > 0, r.row()              # thin layer at the front


@pytest.mark.parametrize("re,source", [(20.0, "Bharti et al. 2007"), (40.0, "Lange et al. 1998")])
def test_heated_cylinder_pr070_references(re, source):
    """The same cylinder at Pr 0.70 on the 2.5 % blockage domain (16 cells/D) against the
    Pr 0.70 references (theory doc 21.2): Re 20 against Bharti et al. (2007), Re 40
    against the Lange, Durst & Breuer (1998) correlation, each within 1.5 % (measured
    -0.07 % and +0.48 %; the domain's remaining blockage adds about +0.4 %). Steady:
    the heat rate changes by < 1e-3 between the halves of the averaging window."""
    r = cases.run_heated_cylinder(re, 16, pr=0.70, device="cuda", **cases.CYLINDER_DOMAINS[0.025])
    v = r.values
    ref = cases.lange_nusselt(re) if source.startswith("Lange") else cases.CYLINDER_NU_REF_PR070[re][source]
    assert v["nu"] == pytest.approx(ref, rel=0.015), r.row()
    assert abs(v["nu_drift"]) < 1e-3, r.row()


def test_rayleigh_benard_onset():
    """Linear growth rates of one convection roll bracket the onset between rigid plates:
    the interpolated critical Rayleigh number is within 3 % of 1707.76 at 16 cells
    across the layer (0.4 % at 32, theory doc 21.3)."""
    r = cases.rb_onset(1600.0, 1800.0, 16, device="cuda")
    assert r["sigma_lo"] < 0 < r["sigma_hi"], r
    assert abs(r["err_pct"]) < 3.0, r


@pytest.mark.parametrize("propeller", [False, True])
def test_actuator_disc_momentum_theory(propeller):
    """An actuator strip in a channel of 10 % blockage, C_T = 0.5: the disc velocity
    against confined momentum theory (Garrett & Cummins 2007 model), 16 cells/D."""
    r = cases.run_actuator_disc(0.5, 16, propeller=propeller, device="cuda")
    v = r.values
    assert v["thrust"] == pytest.approx(0.25, rel=1e-6), r.row()
    assert abs(v["err_pct"]) < 1.0, r.row()                                  # disc velocity within 1 %
    assert abs(v["a_err_pct"]) < 6.0, r.row()                                # induction within 6 %


def test_plane_jet_momentum_flux():
    """A jet port blowing into a closed box (Re_j 25, 8 cells per port width).

    Momentum conservation is tested where it is exact: the x-momentum budget of the
    jet core over 8 <= x/b <= 24 (change of int (u^2 + p - nu du/dx) dy = edge shear +
    the box's pressure-gradient force; ``jet_measures``) must close to 1 % of K
    (measured 0.27 %). The core flux K = int u^2 dy itself drifts by 6 % over that
    span because the closed box returns the entrained fluid as a counter-flow that
    shears the jet (3.6 %) and needs an adverse pressure gradient (2.5 %) — the
    confinement, quantified, not a loss of the scheme (theory doc 21.5) — so it is
    held to < 8 %. The centreline decay and half-width follow Bickley (1937) within
    10 % (measured -4.7 % and -3.9 %)."""
    r = cases.run_plane_jet(device="cuda")
    v = r.values
    assert abs(v["K_budget_residual_pct"]) < 1.0, r.row()
    assert v["K_budget_shear_pct"] < 0 and v["K_budget_pressure_pct"] < 0, r.row()   # both drain the jet
    assert v["K_drift_pct"] < 8.0, r.row()
    assert abs(v["slope_err_pct"]) < 10.0, r.row()
    assert abs(v["b_half_err_pct_x20"]) < 10.0, r.row()
    assert v["shape_rms_x20"] < 0.03, r.row()                                 # sech^2 profile
