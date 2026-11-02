"""Shape sensitivity by autograd through the solver: verification against finite differences.

Fast tests run a tiny float64 cylinder on the CPU (seconds). The slow ones are
the verification cases of docs/theory_sensitivity.md: a steady cylinder at
Re 30 (16 cells per diameter) and a NACA 0012 at Re 500, 4 degrees (GPU), with
autograd checked against central differences of the same windowed objective
along smooth displacement modes.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import threading

import numpy as np
import pytest
import torch

from aerolume.geometry import design_from_airfoil, design_from_polygons, naca
from aerolume.model import CFDResult, CFDSettings, FlowConditions, Fluid, Simulation, SurfaceData
from aerolume.optimize import Cancelled, attach_sensitivity, compute_sensitivity
from aerolume.optimize.sensitivity import (
    SensitivityMaps,
    WindowedObjective,
    choose_window,
    density_on_outline,
    finite_difference_check,
    normal_modes,
    refine_polygon,
)
from aerolume.solvers.ns2d import FlowSolver, ImmersedBody, SolverConfig

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def circle(n: int = 48, d: float = 1.0) -> np.ndarray:
    th = 2 * np.pi * np.arange(n) / n
    return 0.5 * d * np.c_[np.cos(th), np.sin(th)]


def tiny_cfg(device: str = "cpu", **kw) -> SolverConfig:
    base = {"nx": 48, "ny": 32, "h": 1 / 8, "x0": -2.0, "y0": -2.0, "re": 20.0,
            "bodies": [ImmersedBody(circle(48), name="cyl")], "device": device, "dtype": "float64",
            "dye": False, "perturb": False}
    base.update(kw)
    return SolverConfig(**base)


@pytest.fixture(scope="module")
def tiny_problem():
    cfg = tiny_cfg()
    s = FlowSolver(cfg)
    s.advance(60)
    return cfg, s.state.u.clone(), s.state.v.clone()


# ---------------------------------------------------------------------------
# outline helpers
# ---------------------------------------------------------------------------

def test_refine_polygon_keeps_vertices_and_shape():
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    r, idx = refine_polygon(sq, 0.3)
    assert np.allclose(r[idx], sq)
    seg = np.hypot(*np.diff(np.vstack([r, r[:1]]), axis=0).T)
    assert seg.max() <= 0.3 + 1e-12 and math.isclose(seg.sum(), 4.0)
    # every inserted point lies on an original edge (|y| or |x| on the unit square boundary)
    on_edge = np.isclose(r[:, 0], 0) | np.isclose(r[:, 0], 1) | np.isclose(r[:, 1], 0) | np.isclose(r[:, 1], 1)
    assert on_edge.all()


def test_density_recovers_a_known_density():
    p, _ = refine_polygon(circle(64), 0.02)
    seg = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    dual = 0.5 * (seg + np.roll(seg, 1))
    per = seg.sum()
    g_true = 1.0 + 0.5 * np.cos(2 * np.pi * arc / per)        # smooth density
    grad = g_true * dual                                       # dJ/dd_i ~ g(s_i) * dual_i
    sq = np.linspace(0, per, 200, endpoint=False)
    expect = 1.0 + 0.5 * np.cos(2 * np.pi * sq / per)
    for sigma in (0.0, 0.03):
        got = density_on_outline(arc, grad, dual, per, sq, sigma)
        assert np.allclose(got, expect, atol=0.02), sigma


def test_choose_window_defaults():
    assert choose_window(1.0, "gpu", None) == (6.0, 2.0, 1)
    assert choose_window(1.0, "cpu", None) == (3.0, 1.0, 1)
    lead, avg, m = choose_window(1.0, "gpu", 0.8)                 # short period: lead >= 1 unit
    assert lead == 1.0 and m == 3 and avg >= 4.0 and abs(avg / 0.8 - round(avg / 0.8)) < 1e-9
    lead, avg, m = choose_window(0.3, "gpu", 1.8)                 # long period: lead 2 units, 1 period
    assert lead == pytest.approx(0.6) and avg == pytest.approx(1.8) and m == 3
    assert choose_window(1.0, "gpu", None, (1.5, 0.5), 2) == (1.5, 0.5, 2)


# ---------------------------------------------------------------------------
# autograd vs finite differences (tiny, CPU, float64)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ["cd", "cl", "ld"])
def test_gradient_matches_central_differences(tiny_problem, objective):
    cfg, u, v = tiny_problem
    prob = WindowedObjective(cfg, u, v, objective=objective, lead=0.12, average=0.12)
    assert prob.steps >= 6
    modes = normal_modes(prob, 6)
    res = finite_difference_check(prob, modes[[0, 1, 2]], eps=1e-5)
    for r in res:
        scale = max(abs(x["fd"]) for x in res)
        assert abs(r["ad"] - r["fd"]) <= 1e-4 * scale + 1e-12, r
    if objective == "cd":
        assert max(abs(r["fd"]) for r in res) > 1e-2           # a real, non-trivial sensitivity


def test_drag_sensitivity_is_mirror_symmetric(tiny_problem):
    """Symmetric flow: Cd sensitivity mirror-symmetric, Cl antisymmetric (top vs bottom).

    Per vertex the symmetry holds to ~1e-3 of the peak, not to round-off: grid
    points on the symmetry line are exactly equidistant from two mirrored
    segments, and the distance gradient (a subgradient there) goes to one of
    them. Smooth modes average that out (measured 3e-5).
    """
    cfg, u, v = tiny_problem
    for objective, sign in (("cd", 1.0), ("cl", -1.0)):
        prob = WindowedObjective(cfg, u, v, objective=objective, lead=0.12, average=0.12)
        _, g = prob.gradient()
        g = g[0]
        n = len(g)
        mirror = (-np.arange(n)) % n                   # vertex k <-> vertex n - k (reflection in y = 0)
        assert np.abs(g - sign * g[mirror]).max() <= 2e-3 * np.abs(g).max(), objective
        a = normal_modes(prob, 8) @ g                  # mode k centred at 45 k degrees from the rear
        for k in (1, 2, 3):
            assert abs(a[k] - sign * a[8 - k]) <= 1e-4 * np.abs(a).max(), (objective, k, a)


def test_gradient_is_deterministic(tiny_problem):
    cfg, u, v = tiny_problem
    a = WindowedObjective(cfg, u, v, lead=0.06, average=0.06).gradient()[1][0]
    b = WindowedObjective(cfg, u, v, lead=0.06, average=0.06).gradient()[1][0]
    assert np.array_equal(a, b)


def test_cancel_raises(tiny_problem):
    cfg, u, v = tiny_problem
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        WindowedObjective(cfg, u, v, lead=0.06, average=0.06).gradient(cancel=ev)


@needs_cuda
@pytest.mark.gpu
def test_cuda_graph_reverse_sweep_matches_eager():
    cfg = tiny_cfg(device="cuda", nx=96, ny=64, h=1 / 16, bodies=[ImmersedBody(circle(96), name="cyl")])
    s = FlowSolver(cfg)
    s.advance(100)
    g = WindowedObjective(cfg, s.state.u, s.state.v, lead=0.1, average=0.1, use_graphs=True)
    e = WindowedObjective(cfg, s.state.u, s.state.v, lead=0.1, average=0.1, use_graphs=False)
    vg, gg = g.gradient()
    ve, ge = e.gradient()
    assert g.cuda_graphs and not e.cuda_graphs
    assert vg.value == pytest.approx(ve.value, rel=1e-13)
    # the geometry backward (SDF index_add) uses atomics: equal up to summation order
    assert np.allclose(gg[0], ge[0], rtol=1e-9, atol=1e-12 * np.abs(ge[0]).max())


@needs_cuda
@pytest.mark.gpu
def test_checkpoints_offloaded_to_host_give_the_same_gradient(monkeypatch):
    import aerolume.optimize.sensitivity as sens
    cfg = tiny_cfg(device="cuda")
    s = FlowSolver(cfg)
    s.advance(60)
    ref = WindowedObjective(cfg, s.state.u, s.state.v, lead=0.12, average=0.12).gradient()[1][0]
    monkeypatch.setattr(sens, "OFFLOAD_FRACTION", 0.0)              # always offload
    prob = WindowedObjective(cfg, s.state.u, s.state.v, lead=0.12, average=0.12)
    assert prob._offload_target(prob.steps, 3) == "cpu"
    off = prob.gradient()[1][0]
    assert np.allclose(off, ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())


# ---------------------------------------------------------------------------
# public driver + attach
# ---------------------------------------------------------------------------

def _tiny_generic():
    d = design_from_polygons([circle(64)], "Cylinder", kind="generic")
    sim = Simulation("s", d.id, conditions=FlowConditions(fluid=Fluid("unit", 1.0, 0.05), speed=1.0),
                     cfd=CFDSettings(quality="draft", cells_per_ref=8, domain_upstream=1.5, domain_downstream=3.0,
                                     domain_height=4.0, device="cpu", precision="float64"))
    return d, sim


def test_compute_sensitivity_api_and_units():
    from aerolume.simulation import plan_simulation
    d, sim = _tiny_generic()
    plan = plan_simulation(d, sim)
    solver = FlowSolver(dataclasses.replace(plan.solver_config, dye=False))
    solver.advance(40)
    t_before = solver.t
    msgs: list[float] = []
    maps = compute_sensitivity(d, sim, plan=plan, state=solver, objective="cd", window=(0.1, 0.1),
                               n_per_body=120, progress=lambda f, m: msgs.append(f))
    assert solver.t == t_before                              # the caller's solver is only read
    assert isinstance(maps, SensitivityMaps) and len(maps) == 1 and maps[0].shape == (120,)
    assert np.all(np.isfinite(maps[0])) and np.abs(maps[0]).max() > 0
    assert maps.objective == "cd" and maps.bodies == [plan.solver_config.bodies[0].name]
    assert maps.steps > 0 and maps.window[0] >= 0.1 - 1e-12 and maps.peak_memory_bytes is None
    assert maps.points[0].shape == (120, 2) and len(maps.s[0]) == 120
    assert msgs and msgs[-1] == pytest.approx(1.0) and all(a <= b for a, b in itertools.pairwise(msgs))
    # units: 1/m = non-dimensional shape-derivative density / L_ref
    L = plan.length_ref
    vp = maps.vertex_points[0]
    seg = np.hypot(*(np.roll(vp, -1, axis=0) - vp).T)
    arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    dual = 0.5 * (seg + np.roll(seg, 1))
    dens = density_on_outline(arc, maps.vertex_gradient[0], dual, seg.sum(), maps.s[0] / L,
                              1.5 * plan.solver_config.h)
    assert np.allclose(maps[0] * L, dens, rtol=1e-9, atol=0)
    assert maps.summary().startswith("Shape sensitivity of Cd")


def test_attach_sensitivity_matches_and_resamples():
    s = np.linspace(0, 1, 40, endpoint=False)
    sd = [SurfaceData("a", s, s, s, s, s, s, s), SurfaceData("b", s[:20], s[:20], s[:20], s[:20], s[:20],
                                                              s[:20], s[:20])]
    r = CFDResult(sim_id="x", surface=sd)
    maps = SensitivityMaps([np.arange(10.0), np.full(40, 2.0)], bodies=["b", "a"], notes=["note"],
                           warnings=["warn"])
    attach_sensitivity(r, maps)
    assert np.allclose(r.surface[0].sensitivity, 2.0)                  # matched by name, same length
    assert r.surface[1].sensitivity.shape == (20,)                     # resampled 10 -> 20
    assert r.surface[1].sensitivity[0] == 0.0 and r.surface[1].sensitivity[2] == pytest.approx(1.0)
    assert "note" in r.notes and "warn" in r.warnings
    attach_sensitivity(r, maps)
    assert r.notes.count("note") == 1                                  # idempotent
    r2 = CFDResult(sim_id="y", surface=[SurfaceData("z", s, s, s, s, s, s, s)])
    attach_sensitivity(r2, [np.ones(40)])                              # plain list: by order
    assert np.allclose(r2.surface[0].sensitivity, 1.0)


def test_compute_sensitivity_cancel():
    from aerolume.simulation import plan_simulation
    d, sim = _tiny_generic()
    plan = plan_simulation(d, sim)
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        compute_sensitivity(d, sim, plan=plan, window=(0.1, 0.1), cancel=ev)


def test_unknown_objective_rejected():
    d, sim = _tiny_generic()
    with pytest.raises(ValueError):
        compute_sensitivity(d, sim, objective="downforce")


class _HistorySolver:
    """Just enough of a FlowSolver for shedding_period: a force history and a turbulence flag."""

    def __init__(self, t, cd, cl, turbulence: str = "none") -> None:
        self._h = {"t": np.asarray(t, float), "cd": np.asarray(cd, float), "cl": np.asarray(cl, float),
                   "cm": np.zeros(len(t))}
        self.cfg = type("Cfg", (), {"turbulence": turbulence})()

    def history(self):
        return self._h


def test_shedding_period_uses_the_runners_rules():
    """The steady Re-40 cylinder: a start-up ripple of Cl (0.57 % of Cd, decaying x0.59) was taken
    for shedding, so the map ran the 3x-costlier unsteady branch and called the flow unsteady."""
    from aerolume.optimize.sensitivity import shedding_period
    t = np.linspace(0.0, 60.0, 6000)
    cd = 1.6 + 0.0 * t
    # real shedding: 8 cycles of +-0.3 about zero lift
    p = shedding_period(_HistorySolver(t, cd, 0.3 * np.sin(2 * np.pi * t / 6.0)), 30.0)
    assert p == pytest.approx(6.0, rel=0.02)
    # a small decaying ripple of a steady flow (rms well under 2 % of Cd)
    ripple = 0.013 * np.exp(-(t - 30.0) / 60.0) * np.sin(2 * np.pi * t / 8.2)
    assert shedding_period(_HistorySolver(t, cd, ripple), 30.0) is None
    # a strong oscillation that is dying out (laminar): not shedding either
    dying = 0.3 * np.exp(-(t - 30.0) / 15.0) * np.sin(2 * np.pi * t / 6.0)
    assert shedding_period(_HistorySolver(t, cd, dying), 30.0) is None
    # too few cycles to measure a frequency
    slow = 0.3 * np.sin(2 * np.pi * t / 20.0)
    assert shedding_period(_HistorySolver(t, cd, slow), 30.0) is None


def test_every_map_says_its_size_is_indicative():
    from aerolume.optimize.sensitivity import MAGNITUDE_WARNING
    from aerolume.simulation import plan_simulation
    d, sim = _tiny_generic()
    plan = plan_simulation(d, sim)
    solver = FlowSolver(dataclasses.replace(plan.solver_config, dye=False))
    solver.advance(40)
    maps = compute_sensitivity(d, sim, plan=plan, state=solver, objective="cd", window=(0.1, 0.1),
                               n_per_body=60)
    assert MAGNITUDE_WARNING in maps.warnings and "pattern" in MAGNITUDE_WARNING


# ---------------------------------------------------------------------------
# slow: the verification cases of docs/theory_sensitivity.md
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fd_verification_cylinder_re30():
    """Steady cylinder, Re 30, 16 cells/D, float64: autograd vs central differences on 8 modes."""
    dev = "cuda" if CUDA else "cpu"
    cfg = SolverConfig(nx=320, ny=160, h=1 / 16, x0=-5.0, y0=-5.0, re=30.0,
                       bodies=[ImmersedBody(circle(128), name="cyl")], device=dev, dtype="float64",
                       dye=False, perturb=False)
    s = FlowSolver(cfg)
    s.advance(2600)                                   # t ~ 40 D/U: steady
    prob = WindowedObjective(cfg, s.state.u, s.state.v, objective="cd", lead=2.0, average=2.0)
    modes = normal_modes(prob, 8)
    res = finite_difference_check(prob, modes, eps=1e-4)
    fd = np.array([r["fd"] for r in res])
    ad = np.array([r["ad"] for r in res])
    assert np.all(np.sign(fd) == np.sign(ad))
    big = np.abs(fd) > 0.1 * np.abs(fd).max()
    assert np.all(np.abs(ad - fd)[big] <= 0.01 * np.abs(fd)[big]), (ad, fd)     # measured: <= 3e-5
    # top/bottom symmetry: mode k (centred at angle 2 pi k / 8 from the rear) mirrors mode 8 - k
    for k in (1, 2, 3):
        assert ad[k] == pytest.approx(ad[8 - k], rel=1e-6)


@pytest.mark.slow
@needs_cuda
@pytest.mark.gpu
def test_fd_verification_naca0012_re500():
    """NACA 0012, Re 500, 4 deg, draft plan (float64 on the GPU): Cd and Cl sensitivities vs FD (~100 s)."""
    from aerolume.simulation import plan_simulation
    d = design_from_airfoil(naca("0012"), "NACA 0012", 1.0)
    sim = Simulation("s", d.id, conditions=FlowConditions(fluid=Fluid("unit", 1.0, 1.0 / 500), speed=1.0,
                                                          alpha_deg=4.0),
                     cfd=CFDSettings(quality="draft", precision="float64"))
    cfg = plan_simulation(d, sim).solver_config
    s = FlowSolver(dataclasses.replace(cfg, dye=False))
    s.advance(3000)                                   # t ~ 12 c/U: steady at this Re
    for objective in ("cd", "cl"):
        prob = WindowedObjective(cfg, s.state.u, s.state.v, objective=objective, lead=2.0, average=2.0)
        res = finite_difference_check(prob, normal_modes(prob, 8), eps=1e-4)
        fd = np.array([r["fd"] for r in res])
        ad = np.array([r["ad"] for r in res])
        assert np.all(np.sign(fd) == np.sign(ad)), (objective, ad, fd)
        big = np.abs(fd) > 0.2 * np.abs(fd).max()
        # measured: Cd <= 1 %; Cl <= 0.1 % except the three modes touching the sharp trailing
        # edge (3-5.4 %), where grid points inside the thin TE switch between the upper and
        # lower surface as the nearest one (the distance has a kink on the camber line)
        assert np.all(np.abs(ad - fd)[big] <= 0.10 * np.abs(fd)[big]), (objective, ad, fd)


@pytest.mark.slow
@needs_cuda
@pytest.mark.gpu
def test_compute_sensitivity_develops_flow_on_gpu():
    """Full driver on a planned draft airfoil run: develops the flow itself, reports GPU memory."""
    d = design_from_airfoil(naca("0012"), "NACA 0012", 1.0)
    sim = Simulation("s", d.id, conditions=FlowConditions(fluid=Fluid("unit", 1.0, 1.0 / 1000), speed=1.0,
                                                          alpha_deg=5.0),
                     cfd=CFDSettings(quality="draft", sensitivity_objective="ld"))
    maps = compute_sensitivity(d, sim)
    assert maps.cuda_graphs and maps.peak_memory_bytes and maps.peak_memory_bytes < 2 * 2**30
    assert maps.window == pytest.approx((6.0, 2.0), abs=0.01) and maps.period is None
    s = maps[0]
    assert np.all(np.isfinite(s))
    # L/D: the trailing edge dominates (camber / Gurney effect: pushing the lower TE region out raises L/D)
    x = maps.points[0][:, 0]
    assert np.abs(s[x > 0.9]).max() == pytest.approx(np.abs(s).max())
