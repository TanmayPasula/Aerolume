"""Flow-visualisation emitters of the 3D solver: smoke channels and tracer particles
(DESIGN.md 10.1, docs/theory_ns3d.md section 16).

What is proven here (float64 on the CPU unless marked):

* every footprint kind builds weights in [0, 1] where it was put, and the release
  switch honours rate, start, stop and pulses (on the device, inside graphs);
* emitters map to the smoke channels by ``channel``; ``emitters=[]`` releases
  nothing, ``None`` keeps the v0.2 automatic rake;
* smoke is conserved after an emitter stops, and the closed-domain release
  saturates at the emitter's rate;
* particles are released at exactly ``rate`` per unit time, move with the flow
  (RK2 is exact in uniform flow), are retired at the outlet, in solids and at
  their lifetime, recycle through a fixed ring, reflect at a symmetry plane, and
  give one streak line per rake nozzle (cut at pulse pauses);
* a half-domain run with a symmetry plane is the mirror half of the full run
  (smoke channels, temperature, heat rate).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from aerolume.solvers.ns3d import Emitter3D, FlowSolver3D, ImmersedMesh, SolverConfig3D, core
from aerolume.solvers.ns3d import emitters as E


def _load(name: str):
    path = Path(__file__).resolve().parents[1] / "validation" / f"{name}.py"
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "") or ".").resolve() == path:
        return mod
    vdir = str(path.parent)
    if vdir not in sys.path:
        sys.path.insert(0, vdir)                  # thermal3d_cases imports its sibling ns3d_cases
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


cases = _load("ns3d_cases")
thermal_cases = _load("thermal3d_cases")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
#: half-depth of a nozzle along its normal on the tunnel grid (h = 1/6)
NOZZLE_DEPTH = E.NOZZLE_DEPTH_CELLS * (1 / 6)


def _tunnel(**kw) -> SolverConfig3D:
    base = {"nx": 32, "ny": 12, "nz": 12, "h": 1 / 6, "x0": -1.0, "y0": -1.0, "z0": -1.0, "re": 50.0,
            "device": "cpu", "dtype": "float64", "perturb": False}
    base.update(kw)
    return SolverConfig3D(**base)


def _room(**kw) -> SolverConfig3D:
    base = {"nx": 12, "ny": 12, "nz": 12, "h": 1 / 12, "x0": 0.0, "y0": 0.0, "z0": 0.0, "re": 20.0,
            "domain": "closed", "device": "cpu", "dtype": "float64"}
    base.update(kw)
    return SolverConfig3D(**base)


# ---------------------------------------------------------------------------
# footprints and the release switch
# ---------------------------------------------------------------------------

KINDS = [
    {"kind": "point", "position": (0.0, 0.0, 0.0)},
    {"kind": "line", "position": (0.0, -0.5, 0.0), "end": (0.0, 0.5, 0.2)},
    {"kind": "rake", "position": (0.0, -0.5, 0.0), "end": (0.0, 0.5, 0.0), "count": 5},
    {"kind": "rect", "position": (0.0, 0.0, 0.0), "size": (0.0, 0.6, 0.3)},
    {"kind": "disc", "position": (0.0, 0.0, 0.0), "size": (0.3,), "normal": (1.0, 0.0, 1.0)},
    {"kind": "plane", "position": (0.0, 0.0, 0.0), "size": (0.8, 0.5)},
    {"kind": "plane", "position": (0.0, 0.0, 0.0)},
]


@pytest.mark.parametrize("spec", KINDS, ids=lambda s: s["kind"] + ("_full" if s["kind"] == "plane" and
                                                                    "size" not in s else ""))
def test_every_footprint_kind_builds_bounded_weights_where_it_was_put(spec):
    cfg = _tunnel(emitters=[Emitter3D(**spec)])
    ops = core.make_operators3d(cfg)
    src = E.build_smoke_sources(cfg.emitters, None, ops, cfg.extent)
    assert len(src) == 1
    w = src[0].weight
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0 and float(w.max()) > 0.5
    k0, k1, j0, j1, i0, i1 = src[0].window
    xc = ops.x0 + (np.arange(i0, i1) + 0.5) * ops.h
    centre = xc[int(np.argmax(w.sum(dim=(0, 1)).numpy()))]
    assert abs(centre - spec["position"][0]) <= ops.h          # the smoke sits where it was put
    if spec["kind"] == "plane" and "size" not in spec:
        assert (j1 - j0, k1 - k0) == (ops.ny, ops.nz)          # the whole cross-section


def test_release_switch_honours_start_stop_and_pulses():
    e = Emitter3D(rate=0.7, start=1.0, stop=5.0, pulse_period=1.0, pulse_duty=0.25)
    p = E.emitter_params(e, torch.device("cpu"), torch.float64)
    on = lambda t: float(E.release_switch(torch.tensor(t, dtype=torch.float64), p))
    assert on(0.5) == 0.0                      # before start
    assert on(1.1) == 1.0 and on(1.3) == 0.0   # first quarter of each period
    assert on(3.2) == 1.0 and on(3.9) == 0.0
    assert on(5.1) == 0.0                      # after stop
    off = E.emitter_params(Emitter3D(enabled=False), torch.device("cpu"), torch.float64)
    assert float(off[0]) == 0.0                # disabled = rate 0, the source stays allocated


def test_emitter_out_of_the_domain_is_dropped():
    cfg = _tunnel(emitters=[Emitter3D(position=(50.0, 0.0, 0.0))])
    ops = core.make_operators3d(cfg)
    assert E.build_smoke_sources(cfg.emitters, None, ops, cfg.extent) == []


# ---------------------------------------------------------------------------
# smoke channels
# ---------------------------------------------------------------------------

def test_channels_follow_the_emitters():
    em = [Emitter3D(kind="point", position=(-0.5, 0.0, 0.0), channel=0),
          Emitter3D(kind="disc", position=(-0.5, 0.3, 0.3), size=(0.2,), channel=2)]
    s = FlowSolver3D(_tunnel(emitters=em))
    assert s.dye_channels == 3
    s.advance(30)
    sums = [float(s.field(n).sum()) for n in ("dye", "dye1", "dye2", "dye3")]
    assert sums[0] > 0 and sums[1] == 0.0 and sums[2] > 0 and sums[3] == 0.0
    assert 0.0 <= float(s.field("dye2").min()) and float(s.field("dye2").max()) <= 1.0


def test_no_emitters_releases_nothing_and_none_keeps_the_v02_rake():
    s = FlowSolver3D(_tunnel(emitters=[]))
    s.advance(20)
    assert float(s.field("dye").abs().max()) == 0.0
    legacy = FlowSolver3D(_tunnel(bodies=[ImmersedMesh(*cases.icosphere(0.5, 2), name="s")]))
    assert legacy.dye_rake is not None
    legacy.advance(20)
    assert float(legacy.field("dye").max()) > 0.1


def test_smoke_saturates_at_the_rate_and_is_conserved_after_stop():
    em = [Emitter3D(kind="point", position=(0.5, 0.5, 0.5), radius=0.15, rate=0.6, stop=0.5)]
    s = FlowSolver3D(_room(emitters=em, dt=0.01))
    s.advance(50)                                      # t = 0.5: the release has just stopped
    c = s.field("dye")
    assert float(c.max()) == pytest.approx(0.6, abs=1e-6)        # relaxed to the rate inside the nozzle
    total = float(c.sum())
    s.advance(40)                                      # at rest and no source: nothing changes
    assert float(s.field("dye").sum()) == pytest.approx(total, rel=1e-13)


# ---------------------------------------------------------------------------
# tracer particles
# ---------------------------------------------------------------------------

def test_particles_are_released_at_exactly_the_rate():
    em = [Emitter3D(species="particles", kind="disc", position=(0.5, 0.5, 0.5), size=(0.2,), rate=300.0)]
    s = FlowSolver3D(_room(emitters=em, dt=0.01, dye=False))
    s.advance(73)
    assert abs(s.particle_count() - 300.0 * s.t) <= 1.0
    p = s.particles()
    assert p["positions"].shape == (s.particle_count(), 3)
    assert np.all(p["age"] >= -1e-12) and float(p["age"].max()) <= s.t + 1e-12
    r = np.linalg.norm(p["positions"][:, 1:] - 0.5, axis=1)
    assert float(r.max()) <= 0.2 + 1e-6                          # inside the disc


def test_particles_move_with_the_flow_and_leave_at_the_outlet():
    em = [Emitter3D(species="particles", kind="point", position=(-0.6, 0.0, 0.0), rate=100.0)]
    s = FlowSolver3D(_tunnel(emitters=em, dye=False))                   # uniform flow U = 1
    s.advance(200)
    p = s.particles()
    x = p["positions"][:, 0].astype(np.float64)
    assert np.abs(x - (-0.6 + p["age"])).max() < 1.5 * NOZZLE_DEPTH                # RK2 is exact here
    assert float(x.max()) <= s.cfg.extent[1] + 1e-9
    released = 100.0 * s.t
    assert s.particle_count() < 0.9 * released                       # the oldest have left


def test_particles_are_retired_in_solids_and_by_lifetime():
    body = ImmersedMesh(*cases.icosphere(0.6, 2, (0.2, 0.0, 0.0)), name="s")
    em = [Emitter3D(species="particles", kind="rect", position=(-0.6, 0.0, 0.0), size=(0.0, 0.5, 0.5), rate=400.0,
                    lifetime=1.5)]
    s = FlowSolver3D(_tunnel(bodies=[body], emitters=em, dye=False))
    s.advance(150)
    pos = torch.as_tensor(s.particles()["positions"], dtype=torch.float64)
    assert not bool(s._solid_at(pos).any())
    assert float(s.particles()["age"].max()) <= 1.5 + 1e-9
    assert s.particle_count() <= 400.0 * 1.5 + 2


def test_particle_ring_recycles_the_oldest():
    em = [Emitter3D(species="particles", kind="disc", position=(0.5, 0.5, 0.5), size=(0.2,), rate=2000.0)]
    s = FlowSolver3D(_room(emitters=em, dt=0.01, dye=False, particle_capacity=64))
    s.advance(20)
    assert s.particle_count() == 64
    # at most per/4 = 16 per step are released, so the ring holds the last four steps' particles
    assert float(s.particles()["age"].max()) <= 4 * 0.01 + 1e-12


def test_cpu_steps_advect_only_live_particles_and_match_the_full_ring():
    """Eager CPU steps advect only the live slots of the ring (the full ring, which graph capture
    needs, cost 14x the flow step on the CPU): the live particles must be the full path's bit for
    bit -- positions, birth times, emitters and nozzles -- with bodies, lifetimes and recycling."""
    body = ImmersedMesh(*cases.icosphere(0.6, 2, (0.2, 0.0, 0.0)), name="s")
    em = [Emitter3D(species="particles", kind="disc", position=(-0.6, 0.0, 0.0), size=(0.3,), rate=400.0,
                    lifetime=1.0),
          Emitter3D(species="particles", kind="rake", position=(-0.6, -0.4, 0.0), end=(-0.6, 0.4, 0.0), count=3,
                    rate=150.0)]
    runs = []
    for compact in (True, False):
        torch.manual_seed(7)
        s = FlowSolver3D(_tunnel(bodies=[body], emitters=em, dye=False, particle_capacity=1024))
        if not compact:
            s._particles.step = lambda *a, _full=s._particles.step, **k: _full(*a, **{**k, "compact": False})
        s.advance(60)
        runs.append(s.particles())
    a, b = runs
    assert len(a["positions"]) > 100
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_symmetry_plane_reflects_particles():
    em = [Emitter3D(species="particles", kind="line", position=(-0.6, 0.0, 0.0), end=(-0.6, 0.3, 0.0), rate=300.0)]
    body = ImmersedMesh(*cases.icosphere(0.6, 2), name="s")
    s = FlowSolver3D(_tunnel(ny=6, y0=0.0, bc_y="symmetry_ymin", bodies=[body], emitters=em, dye=False))
    s.advance(120)
    assert s.particle_count() > 0
    assert float(s.particles()["positions"][:, 1].min()) >= 0.0


def test_rake_gives_one_streak_line_per_nozzle_cut_at_pulses():
    em = [Emitter3D(species="particles", kind="rake", position=(-0.6, -0.4, 0.0), end=(-0.6, 0.4, 0.0), count=3,
                    rate=150.0, radius=0.02)]
    s = FlowSolver3D(_tunnel(emitters=em, dye=False))
    s.advance(120)
    lines = s.particle_lines()
    assert len(lines) == 3
    for ln in lines:
        assert ln.shape[1] == 3 and len(ln) > 5
        assert np.all(np.diff(ln[:, 0]) > 0)                   # from the nozzle downstream
    assert sorted(round(float(ln[0, 1]), 1) for ln in lines) == [-0.4, 0.0, 0.4]
    pulsed = FlowSolver3D(_tunnel(emitters=[Emitter3D(**{**em[0].__dict__, "pulse_period": 0.5,
                                                          "pulse_duty": 0.4})], dye=False))
    pulsed.advance(120)
    assert len(pulsed.particle_lines()) > 3                    # each nozzle's line is cut into dashes
    area = FlowSolver3D(_tunnel(emitters=[Emitter3D(species="particles", kind="disc", position=(-0.6, 0, 0),
                                                    size=(0.3,), rate=100.0)], dye=False))
    area.advance(10)
    assert area.particle_lines() == [] and area.particle_count() > 0


# ---------------------------------------------------------------------------
# symmetry plane consistency (smoke, temperature, heat rate)
# ---------------------------------------------------------------------------

def test_half_domain_is_the_mirror_half_of_the_full_domain():
    out = thermal_cases.emitter_symmetry(steps=30)
    for name, err in out.items():
        assert err < 1e-10, (name, err)


@needs_cuda
def test_particles_and_smoke_run_inside_cuda_graphs():
    em = [Emitter3D(kind="rake", position=(-0.6, -0.4, 0.0), end=(-0.6, 0.4, 0.0), count=4, pulse_period=0.4),
          Emitter3D(species="particles", kind="rake", position=(-0.6, -0.4, 0.0), end=(-0.6, 0.4, 0.0), count=4,
                    rate=200.0)]
    s = FlowSolver3D(_tunnel(device="cuda", dtype="float32", emitters=em))
    s.advance(40)
    assert s.uses_cuda_graphs and s.graph_error is None and s._graphs
    assert s.particle_count() == pytest.approx(200.0 * s.t, abs=2.0)
    assert float(s.field("dye").max()) > 0.1
    assert math.isfinite(float(s.particles()["positions"].sum()))
