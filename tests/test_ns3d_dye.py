"""Passive smoke ("dye") of the 3D solver (DESIGN.md 7.13, docs/theory_ns3d.md section 14).

What is proven here:

* the advection conserves the smoke exactly away from emitters and solids (finite
  volume with the discretely divergence-free face velocities);
* it creates no new extrema (van Leer MUSCL is TVD at the solver's CFL 0.5), and
  the stored field is always in [0, 1] and zero inside solids;
* clean air enters at the inlet; nozzles emit where they are put, with pulses;
* a half-domain run with a symmetry plane gives the mirror half of the full run;
* the fused CUDA kernels compute the reference arithmetic (float64 to 1e-12), CUDA
  graphs replay it bit for bit, and moving the emitters keeps the graphs;
* the solver API: field("dye"), state_dict / reset, default and custom rakes.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns3d import DyeRake, FlowSolver3D, ImmersedMesh, SolverConfig3D, core
from aerolume.solvers.ns3d import dye as D


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


def _sphere(level: int = 2, d: float = 1.0, center=(0.0, 0.0, 0.0)) -> ImmersedMesh:
    v, f = cases.icosphere(d, level, center)
    return ImmersedMesh(v, f, name="sphere")


def _cfg(**kw) -> SolverConfig3D:
    base = {"nx": 32, "ny": 16, "nz": 16, "h": 1 / 6, "x0": -1.5, "y0": -4 / 3, "z0": -4 / 3, "re": 50.0,
            "bodies": [_sphere()], "device": "cpu", "dtype": "float64", "perturb": False,
            "ref_area": math.pi / 4}
    base.update(kw)
    return SolverConfig3D(**base)


def _periodic_flow(n: int = 16, dtype=torch.float64):
    """A periodic, discretely divergence-free 3D velocity (Taylor-Green-like plus a uniform drift)
    on [0, 2 pi)^3, and the ops of that grid."""
    cfg = SolverConfig3D(nx=n, ny=n, nz=n, h=2 * math.pi / n, x0=0, y0=0, z0=0, re=100.0, periodic=True,
                         device="cpu", dtype="float64" if dtype == torch.float64 else "float32", dye=False)
    ops = core.make_operators3d(cfg)
    c = core.face_coordinates3d(ops)
    xu, yc, zc = c["xu"][None, None, :], c["yc"][None, :, None], c["zc"][:, None, None]
    xc, yv, zw = c["xc"][None, None, :], c["yv"][None, :, None], c["zw"][:, None, None]
    u = 0.6 + torch.sin(xu) * torch.cos(yc) * torch.cos(zc)
    v = 0.3 - torch.cos(xc) * torch.sin(yv) * torch.cos(zc)
    w = (0.2 + 0.0 * zw) * torch.ones(n + 1, n, n, dtype=u.dtype)
    u, v, w = (a.expand(*s).clone() for a, s in ((u, (n, n, n + 1)), (v, (n, n + 1, n)), (w, (n + 1, n, n))))
    u, v, w = core.sync_periodic(u, v, w, ops)
    u, v, w, _ = core.project3d(u, v, w, ops)
    return ops, u, v, w


def _cfl_dt(ops, u, v, w, cfl=0.5):
    return float(cfl * ops.h / (u.abs().max() + v.abs().max() + w.abs().max()))


# ---------------------------------------------------------------------------
# the scheme
# ---------------------------------------------------------------------------

def test_advection_conserves_smoke_exactly_away_from_sources():
    ops, u, v, w = _periodic_flow(16)
    co = core.face_coordinates3d(ops)
    x, y, z = co["xc"][None, None, :], co["yc"][None, :, None], co["zc"][:, None, None]
    c = (0.3 + 0.2 * torch.sin(x) * torch.sin(2 * y) * torch.cos(z)).expand(16, 16, 16).clone()
    total = float(c.sum())
    dt = _cfl_dt(ops, u, v, w)
    for _ in range(60):
        c = D.advect_dye3d(c, u, v, w, dt, ops)
    assert abs(float(c.sum()) - total) / total < 1e-13
    assert float(c.min()) > 0.0 and float(c.max()) < 1.0       # nothing was clipped


def test_advection_is_bounded_and_creates_no_new_extrema():
    ops, u, v, w = _periodic_flow(16)
    g = torch.Generator().manual_seed(3)
    c = 0.1 + 0.7 * torch.rand(16, 16, 16, generator=g, dtype=torch.float64)       # rough field in [0.1, 0.8]
    c[4:8, 4:8, 4:8] = 0.8
    c[10:12, :, :] = 0.1
    lo, hi = float(c.min()), float(c.max())
    dt = _cfl_dt(ops, u, v, w, cfl=0.5)
    for _ in range(80):
        c = D.advect_dye3d(c, u, v, w, dt, ops)
        assert float(c.max()) <= hi + 1e-12
        assert float(c.min()) >= lo - 1e-12


def test_a_uniform_field_stays_uniform():
    ops, u, v, w = _periodic_flow(12)
    c = torch.full((12, 12, 12), 0.5, dtype=torch.float64)
    for _ in range(10):
        c = D.advect_dye3d(c, u, v, w, _cfl_dt(ops, u, v, w), ops)
    assert float((c - 0.5).abs().max()) < 1e-13


def test_floor_and_clip():
    ops, u, v, w = _periodic_flow(8)
    c = torch.zeros(8, 8, 8, dtype=torch.float64)
    c[2, 3, 4] = 1e-9                       # below the floor: removed
    c[5, 5, 5] = 1.0
    out = D.advect_dye3d(c, u, v, w, 1e-6, ops)
    assert float(out[2, 3, 4]) == 0.0
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


# ---------------------------------------------------------------------------
# emitters
# ---------------------------------------------------------------------------

def test_rake_validation_scaling_and_coercion():
    with pytest.raises(ValueError):
        DyeRake("spray", [(0, 0, 0)])
    with pytest.raises(ValueError):
        DyeRake("plane", [(0, 0, 0)])
    with pytest.raises(ValueError):
        DyeRake("rake", [(0, 0, 0)], radius=-1)
    with pytest.raises(ValueError):
        DyeRake("rake", [(0, float("nan"), 0)])
    r = DyeRake("rake", [(1.0, 2.0, 3.0)], radius=0.5, pulse_on=1.0, pulse_off=2.0)
    s = r.scaled(0.5, 4.0, shift=(1.0, 0.0, 0.0))
    assert np.allclose(s.positions, [[0.0, 1.0, 1.5]]) and s.radius == 0.25
    assert (s.pulse_on, s.pulse_off) == (4.0, 8.0) and s.pulsed
    assert DyeRake.coerce(r.to_dict()) == r and DyeRake.coerce(r) is r
    assert r != DyeRake("rake", [(1.0, 2.0, 3.5)], radius=0.5, pulse_on=1.0, pulse_off=2.0)
    assert DyeRake.coerce(None) is None
    with pytest.raises(TypeError):
        DyeRake.coerce("rake")
    cfg = _cfg(dye_rake={"kind": "grid", "positions": [(-1.0, 0.0, 0.0), (-1.0, 0.3, 0.0)]})
    assert isinstance(cfg.dye_rake, DyeRake) and cfg.dye_rake.kind == "grid"


def test_nozzle_weights_are_anti_aliased_and_follow_sub_cell_moves():
    kw = {"nx": 20, "ny": 20, "nz": 20, "h": 0.1, "x0": 0.0, "y0": 0.0, "z0": 0.0}

    def on_grid(rake):
        (k0, k1, j0, j1, i0, i1), w = D.source_weights(rake, **kw)
        full = np.zeros((20, 20, 20))
        full[k0:k1, j0:j1, i0:i1] = w
        return full

    w0 = on_grid(DyeRake("rake", [(0.5, 1.0, 1.0)], radius=0.2))
    w1 = on_grid(DyeRake("rake", [(0.5, 1.03, 1.0)], radius=0.2))
    assert w0.max() == 1.0 and ((w0 > 0) & (w0 < 1)).any()        # a soft edge, not a staircase
    assert not np.allclose(w0, w1)                 # a 0.3-cell move changes the emission
    assert (w1 * np.arange(20)[None, :, None]).sum() / w1.sum() > (w0 * np.arange(20)[None, :, None]).sum() / w0.sum()
    # the emitted cross-section is about the nozzle area (pi r^2 = 0.126 m^2 vs cells of 0.01 m^2)
    win, w = D.source_weights(DyeRake("rake", [(0.5, 1.0, 1.0)], radius=0.2), **kw)
    _k0, _k1, _j0, _j1, i0, i1 = win
    xc = (np.arange(i0, i1) + 0.5) * 0.1
    plane = w[:, :, int(np.argmin(np.abs(xc - 0.55)))]
    assert abs(plane.sum() * 0.01 - math.pi * 0.2 ** 2) < 0.25 * math.pi * 0.2 ** 2
    win, w = D.source_weights(DyeRake("plane", [(0.5, 0.4, 0.4), (0.5, 1.6, 1.2)]), **kw)
    assert w.max() == 1.0
    missed = D.source_weights(DyeRake("rake", [(9.0, 9.0, 9.0)]), **kw)
    assert missed[0] == (0, 0, 0, 0, 0, 0)


def test_default_rake_is_upstream_and_the_same_in_half_and_full_domains():
    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    full = D.default_dye_rake(bounds, (-3, 6, -2, 2, -2, 2), 0.05, kind="rake")
    half = D.default_dye_rake(bounds, (-3, 6, 0, 2, -2, 2), 0.05, kind="rake", symmetry=True)
    assert np.all(full.positions[:, 0] < -0.5)
    assert np.allclose(np.sort(full.positions[full.positions[:, 1] >= -1e-9, 1]), np.sort(half.positions[:, 1]))
    grid = D.default_dye_rake(bounds, (-3, 6, -2, 2, -2, 2), 0.05, kind="grid")
    assert len(grid.positions) >= 6 and len(np.unique(grid.positions[:, 2])) >= 2
    road = D.default_dye_rake((0.0, 4.0, -0.9, 0.9, 0.05, 1.4), (-4, 16, -4, 4, 0, 5), 0.06, kind="grid",
                              ground=True)
    assert road.positions[:, 2].min() >= 2 * 0.06


def test_solver_emits_from_the_nozzles_and_keeps_solids_clean():
    s = FlowSolver3D(_cfg(nx=40, dye_rake=DyeRake("rake", [(-1.0, 0.0, 0.0), (-1.0, 0.0, 0.6)], radius=0.2)))
    s.advance(80)
    d = s.field("dye")
    assert float(d.min()) >= 0.0 and float(d.max()) <= 1.0
    m = s.field("mask")
    assert float(d[m > 0.5].abs().max()) == 0.0
    # the stream aimed at the sphere spreads over it; the upper one passes above it, downstream
    zc = s.ops.z0 + (torch.arange(s.ops.nz, dtype=torch.float64) + 0.5) * s.ops.h
    xc = s.ops.x0 + (torch.arange(s.ops.nx, dtype=torch.float64) + 0.5) * s.ops.h
    behind = d[:, :, xc > 1.0]
    assert float(behind.max()) > 0.05                                 # smoke reached the wake
    upper = d[zc > 0.45][:, :, (xc > -0.3) & (xc < 0.3)]
    assert float(upper.max()) > 0.3                                    # over the top of the sphere


def test_no_smoke_enters_from_the_inlet_without_emitters():
    s = FlowSolver3D(_cfg(dye_rake=DyeRake("rake", np.zeros((0, 3)))))
    s.advance(30)
    assert float(s.field("dye").abs().max()) == 0.0


def test_pulsed_nozzles_cut_the_stream_into_time_lines():
    rake = DyeRake("rake", [(-1.2, 0.0, 0.5)], radius=0.25, pulse_on=0.6, pulse_off=0.6)
    s = FlowSolver3D(_cfg(nx=48, ny=12, nz=16, bodies=[], dye_rake=rake))
    s.advance(int(4.0 / s.dt))
    d = s.field("dye")
    zc = s.ops.z0 + (torch.arange(s.ops.nz, dtype=torch.float64) + 0.5) * s.ops.h
    line = d[int(torch.argmin((zc - 0.5).abs())), s.ops.ny // 2].numpy()
    on = line > 0.5 * line.max()
    starts = np.flatnonzero(on[1:] & ~on[:-1])
    assert line.max() > 0.5 and len(starts) >= 2                      # separate puffs along the stream
    steady = FlowSolver3D(_cfg(nx=48, ny=12, nz=16, bodies=[], dye_rake=DyeRake("rake", [(-1.2, 0.0, 0.5)],
                                                                                    radius=0.25)))
    steady.advance(int(4.0 / steady.dt))
    sl = steady.field("dye")[int(torch.argmin((zc - 0.5).abs())), s.ops.ny // 2].numpy()
    ok = sl > 0.5 * sl.max()
    assert len(np.flatnonzero(ok[1:] & ~ok[:-1])) <= 1                 # a continuous stream has one front


def test_symmetry_plane_gives_the_mirror_half_of_the_full_run():
    rake_full = DyeRake("rake", [(-1.0, -0.4, 0.2), (-1.0, 0.0, 0.2), (-1.0, 0.4, 0.2)], radius=0.25)
    rake_half = DyeRake("rake", [(-1.0, 0.0, 0.2), (-1.0, 0.4, 0.2)], radius=0.25)
    full = FlowSolver3D(_cfg(ny=16, y0=-4 / 3, dye_rake=rake_full))
    half = FlowSolver3D(_cfg(ny=8, y0=0.0, bc_y="symmetry_ymin", dye_rake=rake_half))
    for s in (full, half):
        s.advance(40)
    a = full.field("dye")[:, 8:, :]
    b = half.field("dye")
    assert float(b.max()) > 0.5
    assert float((a - b).abs().max()) < 1e-9
    mirror = full.field("dye")[:, :8, :].flip(1)
    assert float((mirror - b).abs().max()) < 1e-9


def test_state_dict_reset_and_field_list():
    s = FlowSolver3D(_cfg())
    s.advance(20)
    d = s.state_dict()
    assert "dye" in d and float(d["dye"].max()) > 0.5
    t = FlowSolver3D(_cfg())
    t.load_state_dict(d)
    assert torch.equal(t.field("dye"), s.field("dye"))
    s.reset()
    assert float(s.field("dye").abs().max()) == 0.0
    off = FlowSolver3D(_cfg(dye=False))
    off.advance(3)
    assert not off.dye_enabled and float(off.field("dye").abs().max()) == 0.0


def test_set_dye_rake_moves_the_stream():
    s = FlowSolver3D(_cfg(nx=36, bodies=[], dye_rake=DyeRake("rake", [(-1.0, 0.0, -0.6)], radius=0.25)))
    s.advance(30)
    s.set_dye_rake(DyeRake("rake", [(-1.0, 0.0, 0.6)], radius=0.25))
    s.reset()
    s.advance(30)
    d = s.field("dye")
    zc = s.ops.z0 + (torch.arange(s.ops.nz, dtype=torch.float64) + 0.5) * s.ops.h
    assert float(d[zc > 0.3].sum()) > 10 * float(d[zc < -0.3].sum())
    s.set_dye_rake(None)                       # back to the automatic rake
    assert s.dye_rake is not None and len(s.dye_rake.positions) >= 3


# ---------------------------------------------------------------------------
# GPU: fused kernels and CUDA graphs
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@needs_cuda
@pytest.mark.parametrize("kw", [{}, {"bc_y": "symmetry_ymin", "y0": 0.0, "ny": 8}, {"bc_zmin": "moving"},
                                {"periodic": True, "bodies": []}])
def test_fused_smoke_matches_the_reference_float64(kw):
    from aerolume.solvers.ns3d import fused
    args = {"device": "cuda", "perturb": True, "re": 100.0, "dye_rake": DyeRake("grid", [(-1.0, y, z) for y in (0.1, 0.5)
                                                                                    for z in (-0.3, 0.4)], radius=0.2)}
    args.update(kw)
    cfg = _cfg(**args)
    ops = core.make_operators3d(cfg)
    kern = fused.load_kernels(cfg.advection, ops.device, ops.dtype)
    g = torch.Generator(device="cuda").manual_seed(5)
    c = torch.rand(ops.nz, ops.ny, ops.nx, generator=g, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        s0 = core.init_state3d(cfg, ops)
        u, v, w = (x + 0.3 * torch.randn(x.shape, generator=g, device="cuda", dtype=x.dtype) for x in (s0.u, s0.v, s0.w))
        u, v, w = core.apply_boundary_values(u, v, w, ops)
        u, v, w, _ = core.project3d(u, v, w, ops)
        dt = torch.tensor(_cfl_dt(ops, u, v, w), device="cuda", dtype=torch.float64)
        a = b = c
        for _ in range(5):
            a = D.advect_dye3d(a, u, v, w, dt, ops)
            b = fused.fast_dye(b, u, v, w, dt, ops, kern)
    assert float((a - b).abs().max()) < 1e-12


@pytest.mark.gpu
@needs_cuda
def test_gpu_solver_agrees_with_cpu_and_graphs_replay_bit_for_bit():
    base = _cfg(nx=32, ny=16, nz=16, h=1 / 8, x0=-1.5, y0=-1.0, z0=-1.0, perturb=True, re=100.0,
                dye_rake=DyeRake("rake", [(-1.0, -0.3, 0.1), (-1.0, 0.3, 0.1)], radius=0.2))
    cpu = FlowSolver3D(base)
    gpu = FlowSolver3D(dataclasses.replace(base, device="cuda"))
    eager = FlowSolver3D(dataclasses.replace(base, device="cuda", use_cuda_graphs=False))
    assert gpu.uses_fused_kernels and gpu._dye_fused and gpu.dye_error is None
    for s in (cpu, gpu, eager):
        s.advance(40)
    assert torch.equal(gpu.field("dye"), eager.field("dye"))
    assert float((gpu.field("dye").cpu() - cpu.field("dye")).abs().max()) < 1e-9
    assert float(gpu.field("dye").max()) > 0.5


@pytest.mark.gpu
@needs_cuda
def test_moving_the_rake_within_its_window_keeps_the_cuda_graphs():
    base = _cfg(nx=32, ny=16, nz=16, h=1 / 8, x0=-1.5, y0=-1.0, z0=-1.0, device="cuda", dtype="float32",
                dye_rake=DyeRake("plane", [(-1.0, -0.8, -0.8), (-1.0, 0.8, 0.8)]))
    s = FlowSolver3D(base)
    s.advance(20)
    graphs = dict(s._graphs)
    assert graphs
    s.set_dye_rake(DyeRake("plane", [(-1.0, -0.8, -0.8), (-1.0, 0.8, 0.8)], pulse_on=0.1, pulse_off=0.1))
    assert s._graphs == graphs                          # same window: weights and pulses updated in place
    s.advance(20)
    s.set_dye_rake(DyeRake("rake", [(0.5, 0.0, 0.0)]))
    assert not s._graphs                                # another window: re-captured on the next advance
    s.advance(10)
    assert s.graph_error is None
