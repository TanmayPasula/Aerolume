"""Validation of the 3D placed physics against published references
(docs/theory_ns3d.md sections 15-17; the full tables come from
``validation/bench_ns3d.py thermal``). All slow: run with ``-m slow``.

Tolerances are set from the measured grid studies (the documented values), with a
margin for GPU and float32 round-off -- never wider than the references' own scatter:

* differentially heated cubic cavity, Ra 1e4 at 32^3: mean Nu 2.0785 against the
  pseudo-spectral 2.0542 of Tric et al. (2000), +1.2 %, approaching it at second order;
* the same solver in its 2D limit (square cavity, de Vahl Davis 1983, Ra 1e4 at 64^2):
  Nu and both velocity maxima within 0.4 %;
* a body in a stratified fluid at rest: the Archimedes force within 2.5 % at ~10 cells
  per diameter, and the fluid at rest;
* forced convection from an isothermal sphere, Re 50 at 16 cells per diameter, inside
  the band of the published correlations, energy balanced;
* actuator discs pushing the flow (a propeller, C_T 0.8) and holding it back (a turbine, C_T
  0.4) in a tunnel of 2.2 % blockage: disc and wake velocities against confined momentum
  theory (Garrett & Cummins 2007);
* a laminar round jet against Schlichting's (1933) similarity solution, with its core
  momentum budget closed;
* heat-flux bodies: a laminar plate at uniform flux against 0.680 Re^1/2 Pr^1/3 at Draft
  density (+13.3 %), the measured skin wall temperature holding where the old probe
  extrapolation fails, and a uniform-flux sphere next to the isothermal band;
* a buoyancy-dominated sphere (Ri 627) stepping as the planning rule of ns3d.physics predicts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.slow
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "validation" / f"{name}.py"
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    vdir = str(path.parent)
    if vdir not in sys.path:
        sys.path.insert(0, vdir)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


_load("ns3d_cases")
tc = _load("thermal3d_cases")


def test_cubic_cavity_ra1e4_nusselt():
    r = tc.run_cavity(1e4, 32, device=DEVICE)
    v = r.values
    assert v["drift"] < 2e-5
    assert v["nu_hot"] == pytest.approx(tc.CAVITY_NU[1e4]["tric"], rel=0.02)
    assert v["nu_hot"] > tc.CAVITY_NU[1e4]["tric"]                      # converges from above
    assert abs(v["balance"]) < 1e-4 * v["nu_hot"]                       # hot wall = cold wall
    assert v["nu_mid"] == pytest.approx(v["nu_hot"], rel=1e-4)          # same flux through the mid-plane
    mp = tc.CAVITY_MIDPLANE[1e4]
    assert v["u_max"] == pytest.approx(mp["u_max"]["wakashima"][0], rel=0.01)
    assert v["z_umax"] == pytest.approx(mp["u_max"]["wakashima"][1], abs=0.01)
    assert v["w_max"] == pytest.approx(mp["w_max"]["wakashima"][0], rel=0.01)
    assert v["div_rms"] < 1e-5


def test_square_cavity_limit_matches_de_vahl_davis():
    r = tc.run_cavity_2d(1e4, 64, device=DEVICE)
    v = r.values
    assert abs(v["nu_err"]) < 0.004 and abs(v["u_err"]) < 0.004 and abs(v["w_err"]) < 0.004
    assert v["z_umax"] == pytest.approx(v["z_ref"], abs=0.005)
    assert v["x_wmax"] == pytest.approx(v["x_ref"], abs=0.005)


def test_archimedes_force_on_a_body_in_a_stratified_fluid():
    r = tc.run_archimedes(24, device=DEVICE)
    assert abs(r.values["rel_err"]) < 0.025
    assert r.values["max_speed"] < 1e-6


def test_heated_sphere_forced_convection_re50():
    r = tc.run_heated_sphere(50.0, 16, device=DEVICE)
    v = r.values
    lo = min(v["nu_whitaker"], v["nu_clift"], v["nu_yuge"], v["nu_ranz"])
    hi = max(v["nu_whitaker"], v["nu_clift"], v["nu_yuge"], v["nu_ranz"])
    assert lo * 0.97 < v["nu"] < hi * 1.03
    assert v["nu_drift"] < 0.01
    assert v["balance_ratio"] == pytest.approx(1.0, abs=0.02)          # what the body adds leaves the outlet


@pytest.mark.parametrize(("ct", "propeller", "tol"), [(0.8, True, 0.01), (0.4, False, 0.015)],
                         ids=["propeller_ct0.8", "turbine_ct0.4"])
def test_actuator_disc_momentum_theory(ct, propeller, tol):
    """Disc velocity against the confined (2.2 % blockage) momentum theory and the wake on the
    axis 4 D behind the disc against its far-wake value, 12 cells/D (measured: propeller -0.57 %
    / -0.1 %, turbine +0.77 % / +0.2 %; theory doc section 17)."""
    r = tc.run_actuator_disc(ct, 12, propeller=propeller, device=DEVICE)
    v = r.values
    assert v["ct"] == pytest.approx(ct, rel=1e-3), r.row()             # the set thrust is delivered
    assert abs(v["err_confined"]) < tol, r.row()
    assert v["u_axis_4d"] == pytest.approx(v["u_wake_confined"], rel=0.01), r.row()
    if propeller:
        assert 1.0 < v["u_disc"] < v["u_axis_4d"], r.row()              # the wake keeps accelerating
    else:
        assert v["u_axis_4d"] < v["u_disc"] < 1.0, r.row()              # ... or decelerating
    assert v["drift"] < 1e-4, r.row()                                   # steady


def test_round_jet_conserves_momentum_and_follows_schlichting():
    """A laminar round jet (Re_j 100) in a closed box, 4 cells/d, in its steady state: the core's
    momentum budget closes (measured -0.12 % of K), so the drift of K is the box's return flow;
    the profile inside 2 r_1/2 is Schlichting's (1.1 % rms) and the similarity slopes follow it
    within what the confinement explains (measured -6.0 % 1/u_c, -7.9 % r_1/2, -10.8 % inner
    flux; a box of four times the section halves these, theory doc section 17)."""
    r = tc.run_round_jet(100.0, 4, device=DEVICE)
    v = r.values
    assert v["notes"] == 0, r.row()                                     # jet and vent balance
    assert v["uc_unsteady"] < 1e-4 and v["div_rms"] < 1e-6, r.row()
    assert abs(v["budget_residual_pct"]) < 0.5, r.row()
    assert v["shape_rms"] < 0.02, r.row()
    assert abs(v["inv_uc_err_k0"]) < 0.02, r.row()                      # decay for the injected flux
    assert -0.08 < v["inv_uc_err"] < 0.0, r.row()
    assert -0.10 < v["r_half_err"] < 0.0, r.row()
    assert -0.15 < v["q2_err"] < 0.0, r.row()


def test_flux_plate_at_draft_resolution_matches_the_uniform_flux_correlation():
    """A laminar plate at uniform heat flux (Re_L 400, 10 % thick, 32 cells per chord -- the
    Draft density): the mean Nusselt number from the measured wall temperature against the
    similarity solution 0.680 Re^1/2 Pr^1/3 (measured +13.3 %; +8 to +14 % over 32-128 cells per
    chord, not monotone, theory doc 15.7), the skin and surface wall temperatures agree (8 cells
    across the boundary layer), heat balanced."""
    r = tc.run_flux_plate(400.0, 32, t_end=20.0, device=DEVICE)
    v = r.values
    assert abs(v["nu_err"]) < 0.30, r.row()
    assert 0.0 < v["nu_err"] < 0.20, r.row()                            # the documented +13.3 %
    assert v["tw_drift"] < 1e-3, r.row()
    assert v["nu_surface"] == pytest.approx(v["nu"], rel=0.05), r.row()
    assert v["balance_ratio"] == pytest.approx(1.0, abs=0.01), r.row()
    assert 0.8 < v["local_ratio_mean"] < 1.1, r.row()                   # theta_w(x) ~ x^1/2 along the top


def test_flux_plate_wall_temperature_holds_when_the_boundary_layer_is_one_cell():
    """Re_L 3000 at 32 cells per chord (2.9 cells across the boundary layer): the skin
    temperature still gives Nu within 30 % of the correlation (measured -9.2 %), while the probe
    extrapolation that the solver used before reads +51 % -- the critic's box (Nu 1,821) was the
    same bias at ~1 cell."""
    r = tc.run_flux_plate(3000.0, 32, t_end=20.0, device=DEVICE)
    v = r.values
    assert abs(v["nu_err"]) < 0.30, r.row()
    assert v["nu_probe"] > 1.3 * v["nu_ref"], r.row()


def test_flux_sphere_re100_sits_with_the_isothermal_band():
    """Uniform-flux sphere, Re 100, 16 cells/D: Nu = G / theta_w (measured 7.74; the isothermal
    sphere on the same grid 7.15, correlations 6.62-7.35) -- slightly above the isothermal band,
    as an isoflux body's mean-temperature Nusselt number is; energy balanced."""
    r = tc.run_flux_sphere(100.0, 16, device=DEVICE)
    v = r.values
    lo = min(v["nu_whitaker"], v["nu_clift"], v["nu_yuge"], v["nu_ranz"])
    hi = max(v["nu_whitaker"], v["nu_clift"], v["nu_yuge"], v["nu_ranz"])
    assert lo < v["nu"] < 1.15 * hi, r.row()
    assert v["nu_history"] == pytest.approx(v["nu"], rel=1e-6), r.row()
    assert v["tw_drift"] < 0.005, r.row()
    assert v["balance_ratio"] == pytest.approx(1.0, abs=0.01), r.row()


def test_buoyancy_dominated_sphere_steps_as_planned():
    """A heated sphere at Re 100, Ri 627 (the critic's hot sphere: the plume, not the 0.0146 m/s
    stream, sets the flow) over 20 buoyant times: the solver's step count within 2x of the
    planning rule thermal_dt_estimate(h, planning_speed_sum(Ri, 2.1), Pe) (measured over 40 buoyant times: 0.92 at 8 cells/D, 1.06 at 16),
    and the free-fall bound the planner used before was ~10x too optimistic."""
    r = tc.run_buoyant_sphere_steps(100.0, 627.0, 8, 20.0, device=DEVICE)
    v = r.values
    assert 0.5 < v["steps_ratio"] < 2.0, r.row()
    assert v["free_fall_dt"] > 5.0 * v["mean_dt"], r.row()
    assert 2.5 < v["speed_sum_late"] < 6.0, r.row()
