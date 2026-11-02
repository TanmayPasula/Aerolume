"""Experimental CFD shape optimiser (autograd sensitivity + smooth normal modes)."""
from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest
import torch

from aerolume.geometry import design_from_airfoil, design_from_polygons, naca
from aerolume.geometry.polygon import signed_area
from aerolume.model import CFDSettings, Design, FlowConditions, Fluid, Simulation
from aerolume.optimize import Cancelled, OptimizationResult, optimize_shape_cfd
from aerolume.optimize.shape import corner_taper

CUDA = torch.cuda.is_available()


def circle(n: int = 64) -> np.ndarray:
    th = 2 * np.pi * np.arange(n) / n
    return 0.5 * np.c_[np.cos(th), np.sin(th)]


def test_corner_taper_pins_sharp_corners_only():
    foil = naca("0012")
    w = corner_taper(foil, 0.02)
    assert w[0] == pytest.approx(0.0, abs=1e-12)                 # sharp trailing edge held
    assert w[len(foil) // 2] == pytest.approx(1.0, abs=1e-6)     # the round nose moves freely
    assert np.all((w >= 0) & (w <= 1))
    assert np.allclose(corner_taper(circle(), 0.05), 1.0)        # no corners on a circle


def _tiny_case():
    d = design_from_polygons([circle()], "Cylinder", kind="generic")
    sim = Simulation("s", d.id, conditions=FlowConditions(fluid=Fluid("unit", 1.0, 0.05), speed=1.0),
                     cfd=CFDSettings(quality="draft", cells_per_ref=8, domain_upstream=1.5, domain_downstream=3.0,
                                     domain_height=4.0, duration=2.0, average_from=1.0, device="cpu",
                                     precision="float64", sensitivity_objective="cd"))
    return d, sim


def test_tiny_cfd_optimisation_runs_and_keeps_area():
    d, sim = _tiny_case()
    r = optimize_shape_cfd(d, sim, n_modes=8, n_iter=2, window=(0.1, 0.1))
    assert isinstance(r, OptimizationResult) and r.method == "cfd-autograd-modes" and r.objective == "cd"
    nd = r.extra["design"]
    assert isinstance(nd, Design) and len(nd.bodies) == 1 and nd.source["from"] == d.id
    assert len(r.best_coords) == 1 and r.best_coords[0].shape[1] == 2
    a0, a1 = signed_area(d.bodies[0].points), signed_area(nd.bodies[0].points)
    assert abs(a1 / a0 - 1.0) < 1e-3                              # area held (projection + correction)
    assert r.history[0]["iter"] == 0 and r.best["value"] <= r.initial["value"] + 1e-12
    if r.improvement > 0:
        assert r.extra["max_movement_m"] > 0
    assert np.isfinite(r.improvement)


def test_cancel_before_first_gradient_raises():
    d, sim = _tiny_case()
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        optimize_shape_cfd(d, sim, n_iter=1, window=(0.1, 0.1), cancel=ev)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.gpu
def test_naca0012_re500_ld_gain_confirmed_by_cfd():
    """Two gradient steps raise L/D at Re 500, 4 deg, and an independent run_cfd confirms it
    (measured with three steps: 1.035 -> 1.518 from run_cfd, +47 %)."""
    from aerolume.simulation import plan_simulation, run_cfd
    d = design_from_airfoil(naca("0012"), "NACA 0012", 1.0)
    sim = Simulation("s", d.id, conditions=FlowConditions(fluid=Fluid("unit", 1.0, 1.0 / 500), speed=1.0,
                                                          alpha_deg=4.0),
                     cfd=CFDSettings(quality="draft", sensitivity_objective="ld"))
    r = optimize_shape_cfd(d, sim, n_iter=2)
    assert r.improvement > 0.1
    nd = r.extra["design"]
    ld = []
    for des in (d, nd):
        res = run_cfd(plan_simulation(des, dataclasses.replace(sim, design_id=des.id)))
        ld.append(res.cl.mean / res.cd.mean)
    assert ld[1] > 1.1 * ld[0], ld
