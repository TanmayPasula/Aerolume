"""FlowSolver API: fields, history, averaging, surface data, live edits, persistence, CUDA graphs."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns2d import FIELDS, FlowSolver, ImmersedBody, SolverConfig
from aerolume.solvers.ns2d.solver import bilinear


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

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cfg(**kw) -> SolverConfig:
    base = {"nx": 96, "ny": 48, "h": 1 / 12, "x0": -2.0, "y0": -2.0, "re": 100.0, "device": "cpu",
            "dtype": "float32", "bodies": [ImmersedBody(cases.circle(96), name="cyl")], "moment_ref": (0.0, 0.0)}
    base.update(kw)
    return SolverConfig(**base)


def test_fields_shapes_and_values():
    s = FlowSolver(_cfg())
    s.advance(20)
    for name in FIELDS:
        f = s.field(name)
        assert f.shape == (48, 96), name
        assert torch.all(torch.isfinite(f)), name
    assert float(s.field("divergence").abs().max()) * s.ops.h < 1e-5
    m = s.field("mask")
    assert float(m.max()) == 1.0 and float(m.min()) == 0.0
    cp = s.field("cp")
    assert abs(float(cp[:, :2].mean())) < 1e-5                   # referenced to the inlet
    assert float(cp.max()) > 0.8                                 # stagnation point
    with pytest.raises(KeyError):
        s.field("nonsense")


def test_history_coefficients_and_time():
    s = FlowSolver(_cfg())
    s.advance(7)
    s.advance(5)
    h = s.history()
    assert s.steps == 12 and len(h["t"]) == 12
    assert np.all(np.diff(h["t"]) > 0) and abs(h["t"][-1] - s.t) < 1e-6
    assert np.allclose(s.coefficients(), [h["cd"][-1], h["cl"][-1], h["cm"][-1]], atol=1e-6)
    assert s.body_coefficients().shape == (1, 3)
    assert s.dt > 0


def test_history_survives_ring_wrap(monkeypatch):
    from aerolume.solvers.ns2d import solver as mod
    monkeypatch.setattr(mod, "HISTORY_CAPACITY", 16)
    s = FlowSolver(_cfg(nx=48, ny=24, h=1 / 6))
    s.advance(50)
    h = s.history()
    assert len(h["t"]) == 50 and np.all(np.diff(h["t"]) > 0)


def test_averaging_and_mean_fields():
    s = FlowSolver(_cfg())
    s.advance(10)
    s.start_averaging()
    s.advance(20)
    mf = s.mean_fields()
    for k in ("u", "v", "speed", "p", "cp", "vorticity", "u_rms", "v_rms", "p_rms", "cp_rms", "mask"):
        assert mf[k].shape == (48, 96) and torch.all(torch.isfinite(mf[k])), k
    assert s.averaging_time > 0
    assert float(mf["u_rms"].min()) >= 0
    # the mean of a short window is close to the instantaneous field
    assert float((mf["u"] - s.field("u")).abs().max()) < 0.5


def test_surface_output():
    s = FlowSolver(_cfg())
    s.advance(30)
    (surf,) = s.surface(200)
    for k in ("s", "x", "y", "nx", "ny", "cp", "cf"):
        assert surf[k].shape == (200,) and np.all(np.isfinite(surf[k])), k
    # outward normals of a CCW circle point away from the centre
    assert np.all(surf["x"] * surf["nx"] + surf["y"] * surf["ny"] > 0.99 * 0.5)
    i_front = int(np.argmin(surf["x"]))
    assert surf["cp"][i_front] > 0.6                              # stagnation region
    assert np.all(np.diff(surf["s"]) > 0)
    s.start_averaging()
    s.advance(5)
    assert len(s.surface(50, mean=True)) == 1


def _control_volume_drag(s: FlowSolver, r: float, n: int = 1000) -> float:
    """Cd from the steady momentum balance on a circle of radius r around the
    origin, using the solver's own p (field) and staggered velocities."""
    ops, st, h, nu = s.ops, s.state, s.ops.h, 1.0 / s.cfg.re
    u, v, p = st.u, st.v, st.p
    ux, uy = (u[:, 1:] - u[:, :-1]) / h, (u[1:] - u[:-1]) / h        # centres, corners
    vx = (v[:, 1:] - v[:, :-1]) / h                                  # corners
    th = torch.arange(n, dtype=u.dtype) * (2 * math.pi / n)
    cx, cy = torch.cos(th), torch.sin(th)
    x, y = r * cx, r * cy
    P = bilinear(p, x, y, ops)
    U, V = bilinear(u, x, y, ops, 0.0, 0.5), bilinear(v, x, y, ops, 0.5, 0.0)
    UX, UY, VX = bilinear(ux, x, y, ops), bilinear(uy, x, y, ops, 0.0, 1.0), bilinear(vx, x, y, ops, 1.0, 0.0)
    fx = -P * cx - U * (U * cx + V * cy) + nu * (2 * UX * cx + (UY + VX) * cy)
    return 2.0 * float(fx.sum()) * (2 * math.pi * r / n)


def test_pressure_closes_the_momentum_balance():
    """Steady Re = 10 cylinder: the drag from a control-volume momentum balance
    with the output pressure equals the penalisation impulse on every circle
    around the body. This pins the output p to the step-averaged pressure of the
    scheme (with the last-stage pressure alone it was 6 % low at Re 40), and surface() data
    integrate to the same force within the coarse-grid sampling error."""
    h = 0.1
    cfg = SolverConfig(nx=120, ny=80, h=h, x0=-4.0, y0=-4.0, re=10.0, device="cpu", dtype="float64",
                       bodies=[ImmersedBody(cases.circle(128))], moment_ref=(0.0, 0.0), dye=False, perturb=False)
    s = FlowSolver(cfg)
    while s.t < 10.0:
        s.advance(100)
    cd = float(s.coefficients()[0])
    for r in (0.75, 1.0, 1.5, 2.5):
        assert abs(_control_volume_drag(s, r) / cd - 1.0) < 3e-3, (r, _control_volume_drag(s, r), cd)
    surf = s.surface(400)[0]
    assert abs(cases.surface_drag(surf)["cd_surface"] / cd - 1.0) < 0.07
    assert float(surf["cp"][np.argmin(surf["x"])]) > 1.0             # viscous stagnation pressure


def test_set_inflow_and_set_bodies_live():
    s = FlowSolver(_cfg())
    s.advance(10)
    s.set_inflow(re=400.0)
    assert abs(float(s.ops.nu) - 1 / 400) < 1e-9 and s.cfg.re == 400.0
    win = s.geometry.window
    rotated = ImmersedBody(cases.rotate(cases.naca4("0012", 120) - [0.5, 0], -12.0), name="foil")
    s.set_bodies([rotated])
    assert s.geometry.window == win                               # fits: tensors reused
    s.advance(10)
    assert np.all(np.isfinite(s.coefficients()))
    big = ImmersedBody(cases.circle(96, 2.5), name="big")
    s.set_bodies([big, ImmersedBody(cases.circle(48, 0.4, 1.5, 0.0))])
    assert s.geometry.window != win and s.body_coefficients().shape == (2, 3)
    s.advance(5)
    assert np.all(np.isfinite(s.body_coefficients()))


def test_reset_and_state_dict_roundtrip():
    s = FlowSolver(_cfg())
    s.advance(15)
    d = s.state_dict()
    c_before = s.coefficients()
    s.advance(10)
    s2 = FlowSolver(_cfg())
    s2.load_state_dict(d)
    assert s2.steps == 15 and abs(s2.t - float(d["t"])) < 1e-9
    assert np.allclose(s2.coefficients(), c_before)
    s2.advance(10)
    assert np.allclose(s2.field("u").numpy(), s.field("u").numpy(), atol=1e-6)
    s.reset()
    assert s.steps == 0 and s.t == 0.0 and len(s.history()["t"]) == 0


def test_cpu_eager_reproducible():
    a, b = FlowSolver(_cfg()), FlowSolver(_cfg())
    a.advance(12)
    b.advance(5)
    b.advance(7)
    assert torch.equal(a.state.u, b.state.u)


def test_moving_ground_les_smoke_cpu():
    """Rounded box with rolling wheels above a moving road, upwind3 + WALE at
    Re = 1e6: stays bounded (short CPU version of the slow GPU test)."""
    cfg = cases.moving_ground_config(48, device="cpu", dtype="float32")
    s = FlowSolver(cfg)
    s.advance(150)
    u, v = s.field("u"), s.field("v")
    assert torch.all(torch.isfinite(u)) and float(torch.hypot(u, v).max()) < 3.0
    assert np.all(np.isfinite(s.body_coefficients()))
    assert float(s.field("nut").max()) > 0


@pytest.mark.gpu
@needs_cuda
def test_cuda_graph_matches_eager():
    kw = {"device": "cuda", "nx": 256, "ny": 128, "h": 1 / 16, "x0": -4.0, "y0": -4.0, "perturb": True}
    g = FlowSolver(_cfg(use_cuda_graphs=True, **kw))
    e = FlowSolver(_cfg(use_cuda_graphs=False, **kw))
    for s in (g, e):
        s.advance(23)            # 2 graph blocks of 10 + 3 single-step replays
        s.start_averaging()
        s.advance(11)
    assert g.uses_cuda_graphs and g.graph_error is None
    assert torch.allclose(g.state.u, e.state.u, atol=1e-6)
    assert np.allclose(g.history()["cd"], e.history()["cd"], atol=1e-5)
    assert torch.allclose(g.mean_fields()["u"], e.mean_fields()["u"], atol=1e-6)


@pytest.mark.gpu
@needs_cuda
def test_cuda_graph_live_edits_keep_graphs():
    s = FlowSolver(_cfg(device="cuda", nx=256, ny=128, h=1 / 16, x0=-4.0, y0=-4.0))
    s.advance(20)
    n_graphs = len(s._graphs)
    s.set_inflow(re=250.0)
    s.set_bodies([ImmersedBody(cases.rotate(cases.circle(96), 3.0))])
    s.advance(20)
    assert len(s._graphs) == n_graphs and s.uses_cuda_graphs
    assert np.all(np.isfinite(s.coefficients()))
    s.reset()
    s.advance(10)
    assert s.steps == 10


@pytest.mark.gpu
@needs_cuda
def test_gpu_float32_and_float64_agree():
    s32 = FlowSolver(_cfg(device="cuda", dtype="float32"))
    s64 = FlowSolver(_cfg(device="cuda", dtype="float64"))
    s32.advance(40)
    s64.advance(40)
    assert np.allclose(s32.coefficients(), s64.coefficients(), atol=2e-3)
    assert math.isclose(s32.t, s64.t, rel_tol=1e-5)
