"""Live editing, diagnostics, the solver trace and the PID time step of the 3D solver
(DESIGN.md 10.4, docs/theory_ns3d.md section 18).

What is proven here (float64 on the CPU unless marked):

* ``set_emitters``, ``set_sources``, ``set_thermal`` and ``paint_solid`` change a running
  solver without rebuilding its grid (same operators, same Poisson solver), keep the
  flow, and edit device tensors in place when the structure is unchanged (on CUDA the
  captured graphs survive);
* painted obstacles stop the flow inside the brush, block smoke and particles, can be
  erased and cleared, and never touch the differentiable core;
* diagnostics take one device-side row per 10-step block with the continuity residual
  at round-off, the CFL number of the step taken and per-field change norms; they
  export to CSV;
* the trace captures the three Runge-Kutta stages (projected divergence at round-off,
  relative Poisson residual ~1e-12) without changing the run;
* the PID controller keeps the CFL number at or below the limit and changes dt
  smoothly (at most x1.2 per step), settling near its target;
* state_dict / load_state_dict resume a thermal run exactly, reset restores the
  initial state, and time-averaged reports average the per-step history.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns3d import (
    Emitter3D,
    Fan3D,
    FlowSolver3D,
    HeatSource3D,
    ImmersedMesh,
    SolverConfig3D,
    ThermalConfig3D,
    core,
)
from aerolume.solvers.ns3d.solver import DIAG_KEYS


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
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sphere(d=0.6, center=(0.0, 0.0, 0.0), **kw) -> ImmersedMesh:
    v, f = cases.icosphere(d, 2, center)
    return ImmersedMesh(v, f, name=kw.pop("name", "sphere"), **kw)


def _tunnel(**kw) -> SolverConfig3D:
    base = {"nx": 32, "ny": 12, "nz": 12, "h": 1 / 6, "x0": -1.5, "y0": -1.0, "z0": -1.0, "re": 60.0,
            "device": "cpu", "dtype": "float64", "perturb": False, "ref_area": math.pi * 0.09}
    base.update(kw)
    return SolverConfig3D(**base)


def _hot_tunnel(**kw) -> SolverConfig3D:
    base = {"bodies": [_sphere(name="hot", thermal="temperature", thermal_value=1.0)],
            "thermal": ThermalConfig3D(prandtl=0.71, richardson=0.2), "diagnostics": True,
            "fans": [Fan3D(center=(-1.0, 0.0, 0.3), direction=(1, 0, 0), diameter=0.5, mode="velocity", value=1.3)],
            "emitters": [Emitter3D(kind="point", position=(-1.2, 0.0, 0.0))]}
    base.update(kw)
    return _tunnel(**base)


# ---------------------------------------------------------------------------
# live edits
# ---------------------------------------------------------------------------

def test_set_emitters_edits_in_place_when_the_cells_are_unchanged():
    s = FlowSolver3D(_tunnel(emitters=[Emitter3D(kind="point", position=(-1.0, 0.0, 0.0), rate=1.0)]))
    ops, poisson = s.ops, s.ops.poisson
    s.advance(20)
    src = s._smoke_src[0]
    s.set_emitters([Emitter3D(kind="point", position=(-1.0, 0.0, 0.0), rate=0.4)])
    assert s._smoke_src[0] is src                         # same tensors: a captured graph stays valid
    assert float(src.params[0]) == pytest.approx(0.4)
    s.advance(30)
    k0, k1, j0, j1, i0, i1 = src.window
    core_val = float(s.field("dye")[k0:k1, j0:j1, i0:i1].max())
    assert core_val == pytest.approx(0.4, abs=0.02)        # the nozzle now releases at the new rate
    s.set_emitters([Emitter3D(kind="disc", position=(-0.5, 0.2, 0.0), size=(0.3,), channel=1)])
    assert s._smoke_src[0] is not src and s.dye_channels == 2
    assert s.ops is ops and s.ops.poisson is poisson      # the grid is never rebuilt
    s.set_emitters(None)                                  # back to the v0.2 automatic rake
    assert s._smoke_src == [] and s._dye_src is not None


def test_set_sources_keeps_the_flow_and_the_controller_state():
    s = FlowSolver3D(_hot_tunnel())
    s.advance(40)
    u_before = s.field("u").clone()
    thrust = s.fan_report()[0]["thrust"]
    s.set_sources(fans=[Fan3D(center=(-1.0, 0.0, 0.3), direction=(1, 0, 0), diameter=0.5, mode="velocity",
                              value=1.5)])
    assert torch.equal(s.field("u"), u_before)                    # the flow is kept
    assert s.fan_report()[0]["thrust"] == pytest.approx(thrust)   # the PI integrator carries over
    assert s.fan_report()[0]["target"] == pytest.approx(1.5)
    s.set_sources(heat_sources=[HeatSource3D(center=(1.0, 0.0, 0.0), size=(0.3, 0.3, 0.3), power=1.0)])
    assert s.physics.thermal.sources.total == pytest.approx(1.0, rel=1e-12)
    s.advance(10)
    s.set_sources(fans=[])
    assert s.physics.fans is None and s.fan_report() == []
    cold = FlowSolver3D(_tunnel())
    with pytest.raises(ValueError):
        cold.set_sources(heat_sources=[HeatSource3D()])


def test_set_thermal_changes_wall_values_in_place():
    s = FlowSolver3D(_hot_tunnel(thermal=ThermalConfig3D(prandtl=0.71, walls={"bottom": ("temperature", 0.0)})))
    s.advance(10)
    phys = s.physics
    s.set_thermal(walls={"bottom": ("temperature", 0.8)})
    assert s.physics is phys                                       # same operators (graphs kept)
    assert float(phys.thermal.side_vals[4]) == pytest.approx(0.8)
    s.set_thermal(richardson=0.5)
    assert s.physics is not phys and s.physics.thermal.ri == 0.5   # a structural change rebuilds
    assert s.theta is not None and s.thermal_enabled


def test_painted_obstacle_stops_the_flow_and_can_be_erased():
    em = [Emitter3D(kind="point", position=(-1.2, 0.0, 0.0), radius=0.3),
          Emitter3D(species="particles", kind="disc", position=(-1.2, 0.0, 0.0), size=(0.4,), rate=300.0)]
    s = FlowSolver3D(_tunnel(emitters=em))
    geom_before = [t.clone() for t in s.geometry[1:-1]]
    assert s.paint_solid((0.5, 0.0, 0.0), 0.45)
    assert not s.paint_solid((50.0, 0.0, 0.0), 0.3)               # misses the domain
    assert s.painted
    s.advance(120)
    i, j, k = int((0.5 + 1.5) * 6), 6, 6
    assert float(s.field("speed")[k, j, i]) < 0.05                 # at rest inside the brush (penalisation leak)
    assert float(s.field("mask")[k, j, i]) == pytest.approx(1.0)
    assert float(s.field("dye")[k, j, i]) == 0.0                    # smoke blocked
    pos = torch.as_tensor(s.particles()["positions"], dtype=torch.float64)
    assert pos.shape[0] > 0 and not bool(s._solid_at(pos).any())   # particles retired in paint
    for a, b in zip(s.geometry[1:-1], geom_before):                 # the body geometry is untouched
        assert torch.equal(a, b)
    s.paint_solid((0.5, 0.0, 0.0), 0.45 + 2 / 6, erase=True)             # two cells larger: no film left
    assert float(s.field("mask").max()) == 0.0
    s.paint_solid((0.0, 0.0, 0.0), 0.0, shape="box", size=(0.6, 0.6, 0.6))   # >= 2.7 cells thick to be fully solid
    assert float(s.field("mask").max()) > 0.99
    s.clear_paint()
    assert float(s.field("mask").max()) == 0.0
    s.advance(60)
    assert float(s.field("speed")[k, j, i]) > 0.8                    # the flow came back


def test_paint_never_enters_the_differentiable_core():
    cfg = _tunnel(bodies=[_sphere()], dye=False)
    ops = core.make_operators3d(cfg)
    geom = core.build_geometry3d(cfg.bodies, ops)
    st = core.init_state3d(cfg, ops)
    a, _ = core.step3d_full(st, geom, ops)
    s = FlowSolver3D(cfg)
    s.paint_solid((1.5, 0.0, 0.0), 0.4)
    b, _ = core.step3d_full(st, geom, ops)                          # the functional step knows no paint
    assert all(torch.equal(x, y) for x, y in zip(a, b))


@needs_cuda
def test_live_edits_keep_the_cuda_graphs():
    cfg = _hot_tunnel(device="cuda", dtype="float32", thermal=ThermalConfig3D(
        prandtl=0.71, richardson=0.2, walls={"bottom": ("temperature", 0.0)}))
    s = FlowSolver3D(cfg)
    s.advance(30)
    assert s.uses_cuda_graphs and s._graphs
    graphs = dict(s._graphs)
    ptr = s.state.u.data_ptr()
    s.set_emitters([Emitter3D(kind="point", position=(-1.2, 0.0, 0.0), rate=0.5, pulse_period=0.3)])
    s.set_thermal(walls={"bottom": ("temperature", 0.5)})
    assert s._graphs == graphs                                     # device values edited in place
    s.advance(10)
    assert s._graphs == graphs and float(s.physics.thermal.side_vals[4]) == 0.5
    s.paint_solid((1.0, 0.0, 0.0), 0.3)                            # the first stroke adds the layer (re-capture)
    s.advance(20)
    graphs = dict(s._graphs)
    s.paint_solid((1.5, 0.3, 0.0), 0.2)                            # later strokes edit it in place
    s.advance(20)
    assert s._graphs == graphs and s.state.u.data_ptr() == ptr and s.graph_error is None


# ---------------------------------------------------------------------------
# diagnostics, trace, PID
# ---------------------------------------------------------------------------

def test_diagnostics_take_one_row_per_block(tmp_path):
    s = FlowSolver3D(_hot_tunnel())
    s.advance(35)
    d = s.diagnostics()
    assert set(DIAG_KEYS) <= set(d) and {"step", "steps_per_s", "p_residual"} <= set(d)
    np.testing.assert_array_equal(d["step"], [10, 20, 30])
    assert np.all(d["div_rms"] < 1e-11) and np.all(d["div_max"] < 1e-10)
    assert np.all(d["cfl"] <= 0.5 + 1e-12) and np.all(d["cfl"] > 0.3)
    assert np.all(np.isfinite(d["dtheta_rel"])) and np.all(d["du_rel"] > 0)
    assert np.all(d["poisson_residual"] < 1e-10) and np.all(d["steps_per_s"] > 0)
    last = s.latest_diagnostics()
    assert last["step"] == 30 and last["t"] == pytest.approx(d["t"][-1])
    path = s.diagnostics_csv(tmp_path / "diag.csv")
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[0] == "step" and "div_rms" in lines[0] and len(lines) == 4
    plain = FlowSolver3D(_tunnel(diagnostics=True, bodies=[_sphere()]))
    plain.advance(10)
    assert math.isnan(plain.latest_diagnostics()["dtheta_rel"])   # no temperature: no theta change
    with pytest.raises(RuntimeError):
        FlowSolver3D(_tunnel()).diagnostics_csv(tmp_path / "x.csv")


def test_trim_history_cuts_all_three_host_logs_together():
    """Long explore sessions trim the force, physics and diagnostics logs by time; the
    diagnostics' step list stays paired with its rows and later rows keep coming."""
    s = FlowSolver3D(_hot_tunnel())
    s.advance(60)
    t_end = float(s.history()["t"][-1])
    keep = 0.4 * t_end
    s.trim_history(keep)
    h, ph, d = s.history(), s.physics_history(), s.diagnostics()
    assert h["t"][0] >= t_end - keep - 1e-12 and h["t"][-1] == pytest.approx(t_end)
    assert len(h["t"]) < 60 and ph["t"][0] >= t_end - keep - 1e-12 and len(ph["t"]) == len(h["t"])
    assert len(d["step"]) == len(d["t"]) and np.all(d["t"] >= t_end - keep - 1e-12)
    n = len(d["step"])
    assert 0 < n < 6 and list(d["step"]) == [10 * k for k in range(7 - n, 7)]    # the newest rows, still paired
    s.advance(10)
    d2 = s.diagnostics()
    assert d2["step"][-1] == 70 and len(d2["step"]) == len(d["step"]) + 1
    FlowSolver3D(_tunnel()).trim_history(1.0)                     # nothing recorded yet: a no-op


def test_trace_captures_the_stages_without_changing_the_run():
    s = FlowSolver3D(_hot_tunnel())
    s.advance(12)
    u0, th0 = s.field("u").clone(), s.theta.clone()
    thrust0, steps0 = s.fan_report()[0]["thrust"], s.steps
    tr = s.trace()
    assert len(tr["stages"]) == 3 and tr["dt"] > 0
    for st in tr["stages"]:
        assert st["max_div_after"] < 1e-11 and st["poisson_residual"] < 1e-10
        assert st["max_rhs"] > st["max_div_after"]
        assert st["divergence"].shape == (12, 12, 32) and len(st["u_projected"]) == 3 and "theta" in st
    assert torch.equal(s.field("u"), u0) and torch.equal(s.theta, th0)
    assert s.fan_report()[0]["thrust"] == thrust0 and s.steps == steps0
    flat = s.trace_step()
    assert {"stage1/explicit_u", "stage3/projected_w", "stage2/divergence", "stage1/pressure_potential",
            "stage3/correction_v", "stage2/theta", "final_u", "final_p", "final_theta"} <= set(flat)
    s.trace_step(commit=True)
    assert s.steps == steps0 + 1
    np.testing.assert_allclose(s.field("u").numpy(), flat["final_u"].numpy(), atol=1e-6)


def test_pid_keeps_the_cfl_limit_and_changes_dt_smoothly():
    cfg = _tunnel(bodies=[_sphere()], adaptive_dt="pid", perturb=True, dye=False, diagnostics=True)
    s = FlowSolver3D(cfg)
    s.advance(200)
    t = s.history()["t"]
    dt = np.diff(np.concatenate([[0.0], t]))
    growth = dt[1:] / dt[:-1]
    assert growth.max() <= 1.2 + 1e-9 and growth.min() >= 0.5 - 1e-9
    d = s.diagnostics()
    assert np.all(d["cfl"] <= 0.5 * (1 + 1e-9))
    assert d["cfl"][-1] == pytest.approx(0.85 * 0.5, rel=0.1)      # settles near pid_target x cfl
    plain = FlowSolver3D(_tunnel(bodies=[_sphere()], perturb=True, dye=False, diagnostics=True))
    plain.advance(200)
    assert plain.diagnostics()["cfl"][-1] == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# persistence, reset, mean reports
# ---------------------------------------------------------------------------

def test_state_dict_resumes_a_thermal_run_exactly():
    a = FlowSolver3D(_hot_tunnel())
    a.advance(25)
    d = a.state_dict()
    assert "theta" in d and "fan_thrust" in d
    b = FlowSolver3D(_hot_tunnel())
    b.load_state_dict(d)
    assert torch.equal(b.theta, a.theta) and b.steps == a.steps
    a.advance(7)
    b.advance(7)
    for name in ("u", "w", "temperature", "dye"):
        assert torch.equal(a.field(name), b.field(name)), name
    assert a.fan_report()[0]["thrust"] == b.fan_report()[0]["thrust"]


def test_reset_restores_the_initial_physics_state():
    s = FlowSolver3D(_hot_tunnel(emitters=[Emitter3D(species="particles", kind="point", position=(-1.2, 0, 0),
                                                      rate=100.0)]))
    s.advance(30)
    assert s.particle_count() > 0
    s.reset()
    assert s.steps == 0 and s.t == 0.0 and s.particle_count() == 0
    assert float(s.theta.abs().max()) == 0.0
    assert s.diagnostics()["t"].size == 0 and s.physics_history() == {}
    assert s.fan_report()[0]["thrust"] == pytest.approx(float(s.physics.fans.thrust0[0]))


def test_mean_reports_average_the_history():
    s = FlowSolver3D(_hot_tunnel())
    s.advance(30)
    s.start_averaging()
    s.advance(40)
    ph = s.physics_history()
    m = ph["t"] > s._avg_t0
    w = np.diff(ph["t"])[-int(m.sum()):]
    mean_t = float(np.average(ph["fan0_thrust"][m], weights=w))
    rep = s.fans_report(mean=True)[0]
    assert rep["thrust"] == pytest.approx(mean_t, rel=1e-12)
    assert rep["thrust"] != pytest.approx(s.fans_report()[0]["thrust"], rel=1e-9)
    ht = s.heat_transfer(mean=True)
    q = float(np.average(ph["heat_rate_0"][m], weights=w))
    assert ht["bodies"][0]["heat_rate"] == pytest.approx(q, rel=1e-12)
    assert ht["bodies"][0]["nusselt"] == pytest.approx(q / ht["bodies"][0]["area"], rel=1e-12)
    assert set(ht["walls"]) == {"left", "right", "front", "back", "bottom", "top"} and ht["units"] == "conduction"
