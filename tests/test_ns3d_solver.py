"""FlowSolver3D: API, outputs, live edits, persistence; fused kernels and CUDA graphs on the GPU."""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns3d import FIELDS3D, FlowSolver3D, ImmersedMesh, SolverConfig3D, core
from aerolume.solvers.ns3d.geometry import build_geometry3d
from aerolume.solvers.ns3d.solver import HISTORY_KEYS, MEAN_FIELDS3D


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


def _sphere(level: int = 2, d: float = 1.0, center=(0.0, 0.0, 0.0), flip: bool = False, **kw) -> ImmersedMesh:
    v, f = cases.icosphere(d, level, center)
    return ImmersedMesh(v, f[:, ::-1].copy() if flip else f, name=kw.pop("name", "sphere"), **kw)


def _cfg(**kw) -> SolverConfig3D:
    base = {"nx": 24, "ny": 16, "nz": 16, "h": 1 / 6, "x0": -1.5, "y0": -4 / 3, "z0": -4 / 3, "re": 50.0,
            "bodies": [_sphere()], "device": "cpu", "dtype": "float64", "perturb": False,
            "ref_area": math.pi / 4}
    base.update(kw)
    return SolverConfig3D(**base)


# ---------------------------------------------------------------------------
# API and outputs (CPU)
# ---------------------------------------------------------------------------

def test_api_shapes_history_and_fields():
    s = FlowSolver3D(_cfg())
    assert not s.uses_cuda_graphs and not s.uses_fused_kernels
    s.advance(6)
    assert s.coefficients().shape == (6,) and s.body_coefficients().shape == (1, 6)
    h = s.history()
    assert tuple(h) == HISTORY_KEYS and all(len(h[k]) == 6 for k in h)
    assert np.all(np.diff(h["t"]) > 0) and h["t"][-1] == pytest.approx(s.t)
    assert h["cd"][-1] == pytest.approx(s.coefficients()[0])
    for name in FIELDS3D:
        f = s.field(name)
        assert f.shape == (16, 16, 24) and bool(torch.isfinite(f).all()), name
    assert float(s.field("divergence").abs().max()) * s.ops.h < 1e-10
    m = s.field("mask")
    assert float(m.min()) >= 0.0 and float(m.max()) == pytest.approx(1.0)
    assert torch.allclose(s.field("cp"), 2 * s.field("p"))
    with pytest.raises(KeyError):
        s.field("nope")


def test_q_criterion_and_vorticity_of_solid_body_rotation():
    """u = omega x r about +z: Q = Omega^2 and |curl u| = 2 Omega in the interior."""
    s = FlowSolver3D(_cfg(bodies=[]))
    c = core.face_coordinates3d(s.ops)
    om = 0.7
    st = s.state
    st.u.copy_((-om * c["yc"])[None, :, None].expand_as(st.u))
    st.v.copy_((om * c["xc"])[None, None, :].expand_as(st.v))
    st.w.zero_()
    q, vort = s.field("q"), s.field("vorticity")
    inner = (slice(2, -2), slice(2, -2), slice(2, -2))
    assert torch.allclose(q[inner], torch.full_like(q[inner], om * om))
    assert torch.allclose(vort[inner], torch.full_like(vort[inner], 2 * om))
    # slices of gradient fields are computed on the cut only, and agree with the full field
    sl = s.slice("x", 0.3, "q")
    assert sl.shape == (16, 16) and torch.allclose(sl[2:-2, 2:-2], torch.full_like(sl[2:-2, 2:-2], om * om))


def test_slice_orientation_and_interpolation():
    """A linear velocity field is reproduced exactly at any cut position, in the SliceData layout."""
    s = FlowSolver3D(_cfg(bodies=[]))
    c = core.face_coordinates3d(s.ops)
    st = s.state
    a, b, d = 0.3, -0.2, 0.5
    st.u.copy_(a * c["xu"][None, None, :] + b * c["yc"][None, :, None] + d * c["zc"][:, None, None])
    for axis, pos, shape in (("y", 0.11, (16, 24)), ("z", -0.37, (16, 24)), ("x", 0.52, (16, 16))):
        sl = s.slice(axis, pos, "u")
        assert sl.shape == shape
        if axis == "y":
            exact = a * c["xc"][None, :] + b * pos + d * c["zc"][:, None]
        elif axis == "z":
            exact = a * c["xc"][None, :] + b * c["yc"][:, None] + d * pos
        else:
            exact = a * pos + b * c["yc"][None, :] + d * c["zc"][:, None]
        assert torch.allclose(sl, exact, atol=1e-12), axis
    with pytest.raises(ValueError):
        s.slice("q", 0.0, "u")


@pytest.mark.parametrize("flip", [False, True])
def test_surface_data(flip):
    s = FlowSolver3D(_cfg(nx=30, ny=20, nz=20, h=1 / 10, x0=-1.5, y0=-1.0, z0=-1.0, bodies=[_sphere(3, flip=flip)],
                          re=50.0))
    s.advance(25)
    (surf,) = s.surface()
    n = len(surf["vertices"])
    assert surf["name"] == "sphere" and surf["faces"].shape[1] == 3
    assert surf["cp"].shape == (n,) and surf["cf"].shape == (n,) and surf["tau"].shape == (n, 3)
    nrm = surf["normals"]
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    assert np.all(np.sum(nrm * surf["vertices"], axis=1) > 0)          # outward, whatever the winding
    assert np.allclose(np.sum(surf["tau"] * nrm, axis=1), 0.0, atol=1e-12)   # wall shear is tangential
    front = int(np.argmin(surf["vertices"][:, 0]))
    assert surf["cp"][front] > 0.7                                      # stagnation pressure
    assert np.allclose(surf["cf"], np.linalg.norm(surf["tau"], axis=1))


def test_surface_mirrors_the_missing_half_of_a_symmetry_run():
    s = FlowSolver3D(_cfg(y0=0.0, ny=8, bc_y="symmetry_ymin", bodies=[_sphere(3)]))
    s.advance(20)
    (surf,) = s.surface()
    v = surf["vertices"]
    # pair each vertex with its mirror image (the icosphere is symmetric about y = 0)
    key = {tuple(np.round(p, 9)): i for i, p in enumerate(v)}
    pairs = [(i, key[tuple(np.round(p * [1, -1, 1], 9))]) for i, p in enumerate(v) if p[1] > 1e-6]
    i, j = np.array(pairs).T
    assert np.allclose(surf["cp"][i], surf["cp"][j], atol=1e-12)
    assert np.allclose(surf["tau"][i] * [1, -1, 1], surf["tau"][j], atol=1e-12)


def test_running_means_match_an_explicit_time_average():
    """The means sample the flow every AVERAGE_EVERY steps with the elapsed time as the
    weight, and fold in the time since the last sample when read."""
    from aerolume.solvers.ns3d.solver import AVERAGE_EVERY
    s = FlowSolver3D(_cfg(perturb=True, re=200.0))
    s.advance(5)
    s.start_averaging()
    samples, pending = [], 0.0
    for _ in range(2 * AVERAGE_EVERY + 2):
        s.advance(1)
        pending += s.dt
        if s.steps % AVERAGE_EVERY == 0:
            samples.append((pending, s.state.u.clone(), s.state.p.clone()))
            pending = 0.0
    samples.append((pending, s.state.u.clone(), s.state.p.clone()))      # folded in at read time
    assert len(samples) == 3
    w = torch.tensor([x[0] for x in samples], dtype=torch.float64)
    T = float(w.sum())
    mu = sum(wi * x[1] for wi, x in zip(w, samples)) / T
    var = sum(wi * (x[1] - mu) ** 2 for wi, x in zip(w, samples)) / T
    mp = sum(wi * x[2] for wi, x in zip(w, samples)) / T
    m = s.mean_fields()
    assert set(m) == set(MEAN_FIELDS3D)
    assert s.averaging_time == pytest.approx(T)
    assert torch.allclose(m["u"], 0.5 * (mu[..., 1:] + mu[..., :-1]), atol=1e-12)
    rms = var.clamp_min(0).sqrt()
    assert torch.allclose(m["u_rms"], 0.5 * (rms[..., 1:] + rms[..., :-1]), atol=1e-9)
    assert torch.allclose(m["p"], mp - mp[..., :2].mean(), atol=1e-12)
    assert float(m["u_rms"].max()) > 0
    again = s.mean_fields(("u",))["u"]                                  # reading changes nothing
    assert torch.equal(again, m["u"])
    sl = s.slice("y", 0.0, "cp", mean=True)
    assert sl.shape == (16, 24) and bool(torch.isfinite(sl).all())


def test_state_dict_round_trip_and_reset():
    a = FlowSolver3D(_cfg(perturb=True))
    a.advance(4)
    d = a.state_dict()
    a.advance(4)
    b = FlowSolver3D(_cfg(perturb=True))
    b.load_state_dict(d)
    b.advance(4)
    for n in ("u", "v", "w", "p"):
        assert torch.equal(getattr(a.state, n), getattr(b.state, n))
    assert np.array_equal(a.history()["cd"], b.history()["cd"]) and b.steps == 8
    a.reset()
    assert a.t == 0.0 and a.steps == 0 and len(a.history()["t"]) == 0
    fresh = FlowSolver3D(_cfg(perturb=True))
    assert torch.equal(a.state.u, fresh.state.u)


def test_set_inflow_and_set_bodies():
    s = FlowSolver3D(_cfg())
    s.advance(2)
    s.set_inflow(re=400.0)
    assert float(s.ops.nu) == pytest.approx(1 / 400) and s.ops.nu_value == pytest.approx(1 / 400)
    with pytest.raises(ValueError):
        s.set_inflow(re=-1.0)
    geom = s.geometry
    s.set_bodies([_sphere(center=(0.02, 0.0, 0.01))])          # fits the window: rebuilt in place
    assert s.geometry is geom
    s.set_bodies([_sphere(name="a"), _sphere(1, 0.4, (1.2, 0, 0), name="b")])
    assert s.geometry is not geom and s.body_coefficients().shape == (2, 6)
    s.advance(2)
    assert np.all(np.isfinite(s.coefficients()))


def test_live_window_option():
    plate = ImmersedMesh(*cases.box_mesh(1.0, 0.5, 0.1), name="plate")
    tight = FlowSolver3D(_cfg(bodies=[plate])).geometry.window
    wide = FlowSolver3D(_cfg(bodies=[plate], window_pitch_deg=20.0, window_yaw_deg=20.0)).geometry.window
    assert wide[0] <= tight[0] and wide[2] <= tight[2] and (wide[1] - wide[0]) > (tight[1] - tight[0])


# ---------------------------------------------------------------------------
# GPU: fused kernels, CUDA graphs
# ---------------------------------------------------------------------------

_BC_CASES = [{}, {"advection": "central"}, {"turbulence": "wale", "re": 1e5}, {"turbulence": "vreman", "re": 1e5},
             {"turbulence": "smagorinsky", "re": 1e5},
             {"bc_zmin": "moving", "bc_zmax": "noslip"}, {"bc_y": "symmetry_ymin", "y0": 0.0},
             {"bc_y": "periodic"}, {"periodic": True}]


@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("kw", _BC_CASES)
def test_fused_step_matches_reference_float64(kw):
    from aerolume.solvers.ns3d import fused
    args = {"nx": 32, "ny": 16, "nz": 16, "h": 1 / 8, "x0": -1.5, "y0": -1.0, "z0": -1.0, "device": "cuda",
            "perturb": True, "re": 100.0}
    args.update(kw)
    cfg = _cfg(**args)
    ops = core.make_operators3d(cfg)
    kern = fused.load_kernels(cfg.advection, ops.device, ops.dtype)
    with torch.no_grad():
        geom = build_geometry3d(cfg.bodies, ops)
        s0 = core.init_state3d(cfg, ops)
        g = torch.Generator(device="cuda").manual_seed(1)
        u, v, w = (x + 0.1 * torch.randn(x.shape, generator=g, device="cuda", dtype=x.dtype) for x in (s0.u, s0.v, s0.w))
        u, v, w = core.apply_boundary_values(u, v, w, ops)
        u, v, w, _ = core.project3d(u, v, w, ops)
        a = b = s0._replace(u=u, v=v, w=w)
        for _ in range(4):
            a, ca = core.step3d_full(a, geom, ops)
            b, cb = fused.fast_step(b, geom, ops, kern)
    for n in ("u", "v", "w", "p"):
        assert float((getattr(a, n) - getattr(b, n)).abs().max()) < 1e-11, n
    assert float((ca - cb).abs().max()) < 1e-11


@pytest.mark.gpu
@needs_cuda
def test_graphs_match_eager_and_fused_matches_reference():
    base = _cfg(nx=32, ny=16, nz=16, h=1 / 8, x0=-1.5, y0=-1.0, z0=-1.0, device="cuda", dtype="float32",
                perturb=True, re=200.0)
    a = FlowSolver3D(base)
    b = FlowSolver3D(dataclasses.replace(base, use_cuda_graphs=False))
    c = FlowSolver3D(dataclasses.replace(base, use_cuda_graphs=False, fused_kernels=False))
    assert a.uses_fused_kernels and a.uses_cuda_graphs and not c.uses_fused_kernels
    for s in (a, b, c):
        s.advance(13)
        s.start_averaging()
        s.advance(21)
    for n in ("u", "v", "w", "p"):
        assert torch.equal(getattr(a.state, n), getattr(b.state, n)), n          # same kernels, replayed
        assert float((getattr(a.state, n) - getattr(c.state, n)).abs().max()) < 2e-5, n
    assert np.array_equal(a.history()["cd"], b.history()["cd"])
    assert torch.equal(a.mean_fields(("u",))["u"], b.mean_fields(("u",))["u"])
    assert a.graph_error is None


@pytest.mark.gpu
@needs_cuda
def test_set_inflow_recaptures_the_fused_graphs():
    base = _cfg(nx=32, ny=16, nz=16, h=1 / 8, x0=-1.5, y0=-1.0, z0=-1.0, device="cuda", dtype="float32",
                perturb=True, re=100.0)
    a = FlowSolver3D(base)
    b = FlowSolver3D(dataclasses.replace(base, use_cuda_graphs=False))
    for s in (a, b):
        s.advance(10)
        s.set_inflow(re=37.0)
        s.advance(10)
    for n in ("u", "v", "w"):
        assert torch.equal(getattr(a.state, n), getattr(b.state, n))


@pytest.mark.gpu
@needs_cuda
def test_projection_stays_exact_when_tf32_is_enabled_globally():
    m = torch.backends.cuda.matmul
    if not hasattr(m, "fp32_precision"):
        pytest.skip("no per-backend fp32_precision switch")
    old = m.fp32_precision
    try:
        m.fp32_precision = "tf32"
        s = FlowSolver3D(_cfg(nx=32, ny=16, nz=16, h=1 / 8, x0=-1.5, y0=-1.0, z0=-1.0, device="cuda",
                              dtype="float32", perturb=True))
        s.advance(20)
        assert float(s.field("divergence").abs().max()) * s.ops.h < 1e-5
    finally:
        m.fp32_precision = old


@pytest.mark.gpu
@needs_cuda
def test_falls_back_to_the_reference_core_without_fused_kernels(monkeypatch):
    from aerolume.solvers.ns3d import fused

    def broken(*_a, **_k):
        raise RuntimeError("fused kernels unavailable (test)")
    monkeypatch.setattr(fused, "load_kernels", broken)
    s = FlowSolver3D(_cfg(nx=32, ny=16, nz=16, h=1 / 8, x0=-1.5, y0=-1.0, z0=-1.0, device="cuda",
                          dtype="float32"))
    assert not s.uses_fused_kernels and "test" in s.fused_error
    s.advance(12)
    assert np.all(np.isfinite(s.coefficients()))
