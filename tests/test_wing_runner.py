"""run_wing (DESIGN.md 7.7): the finite-wing analysis of a wing design through the simulation layer.

Inviscid sweeps take a second or two; a cold viscous sweep computes 2D section
polars (~30 s for one airfoil on the development machine) and is marked slow.
"""
from __future__ import annotations

import math
import threading
import types

import numpy as np
import pytest

from aerolume.core import fluid as F
from aerolume.geometry import design_from_airfoil, naca
from aerolume.geometry3d import design_from_meshes, design_from_wing, sphere
from aerolume.model import FlowConditions, Simulation, WingResult, WingSection, WingSettings, WingSpec
from aerolume.simulation import defaults as D
from aerolume.simulation import runner
from aerolume.simulation import wing as W
from aerolume.simulation.plan import plan_simulation

AIR = F.air_isa(0.0)


def rect_wing(airfoil: str = "0012", span: float = 6.0, chord: float = 1.0):
    w = WingSpec([WingSection(0.0, chord, airfoil=airfoil), WingSection(0.5 * span, chord, airfoil=airfoil)])
    return design_from_wing(w, "Rectangular wing")


def wing_sim(design, *, viscous: bool = False, method: str = "vlm", alphas=(0.0, 2.0, 4.0), speed: float = 30.0,
             yaw: float = 0.0) -> Simulation:
    a0, a1 = alphas[0], alphas[-1]
    step = alphas[1] - alphas[0] if len(alphas) > 1 else 1.0
    return Simulation("w", design.id, analysis="wing", conditions=FlowConditions(AIR, speed, 0.0, yaw),
                      wing=WingSettings(alpha_start=a0, alpha_end=a1, alpha_step=step, method=method, viscous=viscous))


@pytest.fixture(autouse=True)
def _fresh_polar_cache():
    W.clear_polar_cache()
    yield
    W.clear_polar_cache()


def test_inviscid_vortex_lattice_polar_of_a_rectangular_wing():
    d = rect_wing()
    sim = wing_sim(d)
    msgs = []
    res = W.run_wing(d, sim, progress=lambda f, m: msgs.append((f, m)))
    assert isinstance(res, WingResult) and res.sim_id == sim.id and res.method == "vlm" and not res.viscous
    assert [p.alpha for p in res.points] == [0.0, 2.0, 4.0] and all(p.converged for p in res.points)
    assert res.area == pytest.approx(6.0) and res.span == pytest.approx(6.0) and res.aspect_ratio == pytest.approx(6.0)
    t = res.table()
    assert t["cl"][0] == pytest.approx(0.0, abs=1e-6)                   # symmetric section
    slope = (t["cl"][2] - t["cl"][0]) / math.radians(4.0)
    # lifting-line estimate for AR 6: 2 pi AR / (AR + 2) = 4.71 per radian (a rectangular wing is a little lower)
    assert 4.0 < slope < 4.9
    assert np.all(t["cd"] >= 0) and t["cd"][2] > t["cd"][1]
    assert res.re_mac == pytest.approx(AIR.density * 30.0 * 1.0 / AIR.viscosity)
    assert res.notes[0].startswith("Reynolds number") and "mean aerodynamic chord" in res.notes[0]
    assert msgs[-1][0] == 1.0 and msgs[-1][1].startswith("Wing polar finished")


def test_lifting_line_and_sideslip_are_passed_through():
    d = rect_wing()
    res = W.run_wing(d, wing_sim(d, method="llt", alphas=(4.0,), yaw=5.0))
    assert res.method == "llt" and res.points[0].converged
    assert any("Sideslip 5" in n for n in res.notes)
    assert res.points[0].cy == pytest.approx(res.points[0].cy) and math.isfinite(res.points[0].cy)


def test_project_airfoils_are_resolved_from_the_project():
    af = design_from_airfoil(naca("2412", 200, 1.0), "My 2412", 1.0)
    d = rect_wing(airfoil=f"design:{af.id}")
    ref = rect_wing(airfoil="2412")
    res = W.run_wing(d, wing_sim(d), project={af.id: af})
    want = W.run_wing(ref, wing_sim(ref))
    assert all(p.converged for p in res.points)
    # the same section, generated in two ways: the same lift to a few thousandths
    assert np.allclose(res.table()["cl"], want.table()["cl"], atol=0.02)

    proj = types.SimpleNamespace(designs={af.id: af})                    # anything with a ``designs`` mapping
    assert W.run_wing(d, wing_sim(d), project=proj).points[0].converged
    # an explicit callback wins
    calls = []
    W.run_wing(d, wing_sim(d, alphas=(2.0,)), airfoil_coords=lambda n: calls.append(n) or naca("2412", 200, 1.0))
    assert calls and calls[0] == f"design:{af.id}"


def test_bad_inputs_raise_readable_errors_before_any_work():
    d = rect_wing(airfoil="design:missing")
    with pytest.raises(ValueError, match="no design with id 'missing'"):
        W.run_wing(d, wing_sim(d), project={})
    blob = design_from_meshes([sphere(1.0, 16)], "Ball", kind="generic")
    with pytest.raises(ValueError, match="needs a wing design"):
        W.run_wing(blob, wing_sim(blob))
    d3 = design_from_meshes([sphere(1.0, 16)], "Ball", kind="generic")
    d_wing_of_3d = rect_wing(airfoil=f"design:{d3.id}")
    with pytest.raises(ValueError, match="3D design"):
        W.run_wing(d_wing_of_3d, wing_sim(d_wing_of_3d), project={d3.id: d3})
    ok = rect_wing()
    bad = wing_sim(ok)
    bad.conditions.speed = 0.0
    with pytest.raises(ValueError, match="speed"):
        W.run_wing(ok, bad)


def test_cancel_returns_the_angles_finished_so_far():
    d = rect_wing()
    stop = threading.Event()
    stop.set()
    res = W.run_wing(d, wing_sim(d, alphas=(0.0, 1.0, 2.0, 3.0)), cancel=stop)
    assert len(res.points) < 4 and any("Cancelled" in w for w in res.warnings)


def test_runner_reexport_defaults_and_planner_refusal():
    d = rect_wing()
    res = runner.run_wing(d, wing_sim(d, alphas=(2.0,)))
    assert isinstance(res, WingResult)
    sim = D.default_simulation(d, conditions=FlowConditions(AIR, 25.0))
    assert sim.analysis == "wing" and sim.wing.method == "vlm" and sim.wing.viscous
    with pytest.raises(ValueError, match="run_wing"):
        plan_simulation(d, sim)


def test_run_simulation_dispatches_every_analysis(monkeypatch):
    """runner.run_simulation: the one dispatcher the GUI, the CLI and the scripting API share.
    Wings run for real (inviscid, a second); polar and CFD dispatch is checked with stand-ins."""
    af = design_from_airfoil(naca("2412", 200, 1.0), "My 2412", 1.0)
    d = rect_wing(airfoil=f"design:{af.id}")
    project = types.SimpleNamespace(designs={af.id: af})
    res = runner.run_simulation(d, wing_sim(d, alphas=(2.0,)), project=project)
    assert isinstance(res, WingResult) and res.points[0].converged
    with pytest.raises(ValueError, match="no design with id"):
        runner.run_simulation(d, wing_sim(d, alphas=(2.0,)), project={})

    calls: dict = {}
    monkeypatch.setattr(runner, "run_polar", lambda design, sim, **kw: calls.setdefault("polar", (design, kw)))
    polar = Simulation("p", af.id, analysis="polar")
    runner.run_simulation(af, polar, project=project)
    assert calls["polar"][0] is af and set(calls["polar"][1]) == {"progress", "cancel"}

    from aerolume.simulation import plan as planmod

    def fake_plan(design, sim, *, device=None, airfoil_coords=None):
        calls["plan"] = (design, device, airfoil_coords)
        return "PLAN"
    monkeypatch.setattr(planmod, "plan_simulation", fake_plan)
    monkeypatch.setattr(runner, "run_cfd", lambda plan, **kw: calls.setdefault("cfd", (plan, kw)) and "RESULT")
    seen = []
    cfd = Simulation("c", d.id, analysis="cfd")
    out = runner.run_simulation(d, cfd, project=project, device="cpu", planned=seen.append)
    assert out == "RESULT" and seen == ["PLAN"] and calls["cfd"][0] == "PLAN"
    design, device, coords = calls["plan"]
    assert design is d and device == "cpu"
    xy = coords(f"design:{af.id}")
    assert xy.ndim == 2 and xy.shape[1] == 2 and np.ptp(xy[:, 0]) == pytest.approx(1.0, rel=1e-3)
    # no project, no callback: the planner gets no resolver (NACA sections need none)
    calls.clear()
    runner.run_simulation(rect_wing(), Simulation("c", "x", analysis="cfd"))
    assert calls["plan"][2] is None
    assert "run_simulation" in runner.__all__


@pytest.mark.slow
def test_viscous_polar_reuses_section_polars_between_runs():
    """Viscous strips: profile drag on top of the induced drag, and the second run of the
    same airfoils (another speed or planform would do the same) reuses the section polars."""
    d = rect_wing("2412")
    sim = wing_sim(d, viscous=True, alphas=(0.0, 4.0))
    first = W.run_wing(d, sim)
    assert first.viscous and all(p.converged for p in first.points)
    p = first.points[-1]
    assert p.cdv > 0.004 and p.cd == pytest.approx(p.cdi + p.cdv, rel=1e-6)
    second = W.run_wing(d, sim)
    assert second.wall_seconds < 0.5 * first.wall_seconds
    assert any("reused" in n for n in second.notes)
    assert second.table()["cd"] == pytest.approx(first.table()["cd"])


class _FakePolar:
    """Instant stand-in for solvers.airfoil.polar (thin-airfoil lift, power-law drag); counts points."""

    def __init__(self):
        self.points = 0

    def __call__(self, coords, re, alphas, *, mach=0.0, ncrit=9.0, xtr_upper=1.0, xtr_lower=1.0, panels=199,
                 distributions=True, sim_id="", progress=None, cancel=None):
        from aerolume.model import PolarPoint, PolarResult
        res = PolarResult(sim_id=sim_id, re=re, mach=mach, ncrit=ncrit)
        for a in alphas:
            cl = 0.11 * min(float(a), 14.0)
            res.points.append(PolarPoint(alpha=float(a), cl=cl, cd=0.006 * (1e6 / re) ** 0.3 * (1 + cl * cl),
                                         cm=-0.0, converged=float(a) <= 16.0))
            self.points += 1
        return res


@pytest.fixture
def fake_polar(monkeypatch, tmp_path):
    import aerolume.solvers.airfoil as airfoil_pkg
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AEROLUME_NO_POLAR_CACHE", raising=False)
    W.clear_polar_cache()
    f = _FakePolar()
    monkeypatch.setattr(airfoil_pkg, "polar", f)
    return f


def test_section_polars_are_kept_between_sessions(fake_polar, monkeypatch):
    """A new process (here: an empty in-memory cache) reads the polars an earlier session saved, so
    a script or assistant run of a wing analysed before does not repeat ~25 s of section polars."""
    d = rect_wing("2412")
    sim = wing_sim(d, viscous=True, alphas=(0.0, 4.0))
    first = W.run_wing(d, sim)
    n_first = fake_polar.points
    files = list(W.polar_cache_dir().glob("*.json"))
    assert n_first > 0 and len(files) == 1
    W.clear_polar_cache(disk=False)                                   # "next session"
    second = W.run_wing(d, sim)
    assert fake_polar.points == n_first                                # nothing recomputed
    assert any("saved between sessions" in n and n.endswith("were reused.") for n in second.notes)
    assert second.table()["cd"] == pytest.approx(first.table()["cd"], rel=1e-12)
    # another speed: only the new Reynolds node is computed, and the file grows
    W.clear_polar_cache(disk=False)
    third = W.run_wing(d, wing_sim(d, viscous=True, alphas=(0.0, 4.0), speed=40.0))
    assert fake_polar.points > n_first
    assert any("new section points were added" in n for n in third.notes)
    # a damaged file is ignored (and removed), never an error
    files[0].write_text("{not json", encoding="utf-8")
    W.clear_polar_cache(disk=False)
    W.run_wing(d, sim)
    assert json_ok(files[0])
    # clear_polar_cache() forgets the saved polars too; the switch turns the files off
    W.clear_polar_cache()
    assert not list(W.polar_cache_dir().glob("*.json"))
    monkeypatch.setenv("AEROLUME_NO_POLAR_CACHE", "1")
    assert W.polar_cache_dir() is None
    W.run_wing(d, sim)
    monkeypatch.delenv("AEROLUME_NO_POLAR_CACHE")
    assert not list(W.polar_cache_dir().glob("*.json"))


def json_ok(path) -> bool:
    import json
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("nodes"))
    except (OSError, ValueError):
        return False


def test_saved_polars_are_keyed_by_the_airfoil_shape(fake_polar):
    """Editing a project airfoil changes its coordinates hash: the saved polars of the old shape
    are not served for the new one."""
    af = design_from_airfoil(naca("2412", 200, 1.0), "My airfoil", 1.0)
    d = rect_wing(airfoil=f"design:{af.id}")
    sim = wing_sim(d, viscous=True, alphas=(0.0,))
    W.run_wing(d, sim, project={af.id: af})
    n = fake_polar.points
    W.clear_polar_cache(disk=False)
    af2 = design_from_airfoil(naca("4412", 200, 1.0), "My airfoil", 1.0)
    af2.id = af.id
    W.run_wing(d, sim, project={af.id: af2})
    assert fake_polar.points > n
    assert len(list(W.polar_cache_dir().glob("*.json"))) == 2
