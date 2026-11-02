"""Performance benchmark and validation runner for the 2D Navier-Stokes solver.

Usage (from the repository root)::

    python validation/bench_ns2d.py perf                 # steps/s, GPU and CPU
    python validation/bench_ns2d.py verify               # Poisson, divergence, Taylor-Green, gradient
    python validation/bench_ns2d.py cylinder --cpd 32 48 64 [--re 40 100]
    python validation/bench_ns2d.py ground               # moving-ground LES smoke test
    python validation/bench_ns2d.py channel --cpd 40 80  # Schaefer-Turek 2D-1 and 2D-2
    python validation/bench_ns2d.py square --cpd 32 48   # square cylinder, Re 100
    python validation/bench_ns2d.py physics-perf         # steps/s with each placed-physics feature on
    python validation/bench_ns2d.py default-cost         # what the placed physics costs the default path
    python validation/bench_ns2d.py thermal [--quick]    # cavity, heated cylinder, RB onset, discs, jet
    python validation/bench_ns2d.py fan-room --cpd 6 12 24   # 2D fan-room grid study (theory doc 18.4; ~2 h)
    python validation/bench_ns2d.py all --json validation/out/ns2d.json

Numbers printed here are the ones quoted in docs/theory_ns2d.md. Performance
is measured on the fast path (FlowSolver.advance) after a warm-up, in float32,
with the GPU synchronised before and after the timed block. Other processes
on the machine (shared GPU, busy CPU) lower the numbers; run on an idle box
for representative figures.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ns2d_cases", HERE / "ns2d_cases.py")
cases = importlib.util.module_from_spec(_spec)
sys.modules["ns2d_cases"] = cases
_spec.loader.exec_module(cases)
_tspec = importlib.util.spec_from_file_location("thermal2d_cases", HERE / "thermal2d_cases.py")
tcases = importlib.util.module_from_spec(_tspec)
sys.modules["thermal2d_cases"] = tcases
_tspec.loader.exec_module(tcases)

from aerolume.solvers.ns2d import (
    EmitterSpec,
    FanSpec,
    FlowSolver,
    HeatSourceSpec,
    ImmersedBody,
    PortSpec,
    SolverConfig,
    ThermalConfig,
    WallBC,
    build_geometry,
    core,
    init_state,
    make_operators,
)
from aerolume.solvers.ns2d.poisson import PoissonSolver
from aerolume.solvers.ns2d.rollout import rollout


def _sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def steps_per_second(nx: int, ny: int, device: str, *, graphs: bool = True, advection: str = "upwind3",
                     turbulence: str = "none", dtype: str = "float32", n: int | None = None,
                     repeats: int = 3, body: ImmersedBody | None = None, **extra) -> float:
    """Best-of-``repeats`` throughput of FlowSolver.advance on a cylinder case
    (``extra`` goes to SolverConfig: placed physics, diagnostics, adaptive dt)."""
    h = 24.0 / nx
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=-6.0, y0=-0.5 * ny * h, re=1e5 if turbulence != "none" else 200.0,
                       bodies=[body or ImmersedBody(cases.circle(512))], device=device, dtype=dtype,
                       use_cuda_graphs=graphs, advection=advection, turbulence=turbulence, **extra)
    s = FlowSolver(cfg)
    n = n or (400 if device.startswith("cuda") else 60)
    s.advance(min(n, 40))                      # warm-up (graph capture happens here)
    best = 0.0
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        s.advance(n)
        _sync(device)
        best = max(best, n / (time.perf_counter() - t0))
    return best


def perf(out: dict) -> None:
    rows = []
    if torch.cuda.is_available():
        for kw in ({}, {"graphs": False}, {"advection": "central"}, {"turbulence": "wale"},
                   {"turbulence": "smagorinsky"}):
            n = 100 if kw.get("graphs") is False else None
            r = steps_per_second(1024, 512, "cuda", n=n, **kw)
            rows.append(("cuda", "1024x512", kw, r))
        rows.append(("cuda", "2048x1024", {}, steps_per_second(2048, 1024, "cuda", n=200)))
    threads0 = torch.get_num_threads()
    for th in sorted({1, 4, threads0}):
        torch.set_num_threads(th)
        rows.append(("cpu", "512x256", {"threads": th}, steps_per_second(512, 256, "cpu")))
    torch.set_num_threads(threads0)
    print("\n== performance (float32, steps/s, best of 3) ==")
    for dev, grid, kw, r in rows:
        print(f"  {dev:<5} {grid:<10} {kw!s:<34} {r:8.1f}")
    out["perf"] = [{"device": d, "grid": g, "options": k, "steps_per_s": r} for d, g, k, r in rows]


def physics_variants() -> list[tuple[str, dict]]:
    """The placed-physics feature sets timed by :func:`physics_perf` (same cylinder case)."""
    hot = ImmersedBody(cases.circle(512), name="cyl", thermal="temperature", thermal_value=1.0)
    rake = EmitterSpec((-5.0, -1.0), kind="rake", end=(-5.0, 1.0), count=8, channel=1)
    return [
        ("default", {}),
        ("diagnostics", {"diagnostics": True}),
        ("adaptive_dt=pid", {"adaptive_dt": "pid"}),
        ("2 smoke channels (rake)", {"emitters": [EmitterSpec((-5.0, 0.0), channel=0), rake], "dye_inlet": False}),
        ("particles (8 nozzles)", {"emitters": [EmitterSpec((-5.0, -1.0), kind="rake", end=(-5.0, 1.0), count=8,
                                                            species="particles", rate=200.0)]}),
        ("heat transfer (forced)", {"body": hot, "thermal": ThermalConfig(prandtl=0.71)}),
        ("heat transfer + buoyancy", {"body": hot, "thermal": ThermalConfig(
            prandtl=0.71, richardson=0.5, walls=[WallBC("bottom", "temperature", 0.0)])}),
        ("fan (velocity mode)", {"fans": [FanSpec((-3.0, 2.0), width=1.0, mode="velocity", value=1.5)]}),
        ("jet + vent", {"ports": [PortSpec((-2.0, -3.0), direction=(0.0, 1.0), size=0.5, speed=1.0),
                                  PortSpec((6.0, 3.0), direction=(0.0, 1.0), size=0.5, speed=1.0, kind="vent")]}),
        ("everything", {"body": hot, "diagnostics": True,
                        "thermal": ThermalConfig(prandtl=0.71, richardson=0.5),
                        "emitters": [EmitterSpec((-5.0, 0.0), channel=0), rake,
                                     EmitterSpec((-5.0, 2.0), kind="line", end=(-5.0, 2.5), species="particles",
                                                 rate=100.0, count=4)],
                        "heat_sources": [HeatSourceSpec((3.0, -2.0), size=(0.5, 0.5), power=0.2)],
                        "fans": [FanSpec((-3.0, 2.0), width=1.0, mode="velocity", value=1.5)],
                        "ports": [PortSpec((-2.0, -3.0), direction=(0.0, 1.0), size=0.5, speed=1.0)]}),
    ]


def physics_perf(out: dict) -> None:
    """Throughput with each placed-physics feature on (1024 x 512, float32, CUDA graphs).

    "default" is the plain solver (no placed physics: ``ops.phys`` is None); the
    other rows give the cost of each feature. The variants are interleaved over
    three rounds and the median of the per-round results is reported, which damps
    the drift of a shared GPU.
    """
    if not torch.cuda.is_available():
        print("physics-perf needs CUDA")
        return
    variants = physics_variants()
    runs: dict[str, list[float]] = {name: [] for name, _ in variants}
    for _ in range(3):
        for name, kw in variants:
            runs[name].append(steps_per_second(1024, 512, "cuda", repeats=1, **kw))
    rows = [(name, sorted(r)[1]) for name, r in runs.items()]
    base = rows[0][1]
    print("\n== placed-physics throughput (cuda, 1024x512, float32, graphs; median of 3 rounds) ==")
    for name, r in rows:
        print(f"  {name:<28} {r:8.1f} steps/s   {100.0 * (r / base - 1.0):+6.1f} %")
    out["physics_perf"] = [{"variant": n, "steps_per_s": r, "relative": r / base} for n, r in rows]


def default_cost(out: dict) -> None:
    """What the placed-physics work costs the *default* path (theory doc section 15).

    Counts kernels per step with torch.profiler (eager, the 1024 x 512 cylinder) and
    measures the GPU kernel time of the default path's one addition - smoke cleared
    inside solids - as a fraction of the step's kernel time, over five interleaved
    rounds. Kernel time, not wall clock: a lone eager call of the clean-up is dominated
    by launch overhead that a captured CUDA graph does not pay, and two whole-step
    timings taken separately differ by more than the effect on a shared GPU.
    """
    if not torch.cuda.is_available():
        print("default-cost needs CUDA")
        return
    from torch.profiler import ProfilerActivity, profile

    def kernels(fn, n: int) -> tuple[float, float]:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        return len(evs) / n, sum(e.device_time for e in evs) / n

    nx, ny = 1024, 512
    h = 24.0 / nx
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=-6.0, y0=-0.5 * ny * h, re=200.0,
                       bodies=[ImmersedBody(cases.circle(512))], device="cuda", dtype="float32",
                       use_cuda_graphs=False)
    s = FlowSolver(cfg)
    assert s.physics is None, "the default config must build no placed physics"
    s.advance(20)
    dye, geom = s.state.dye, s.geometry
    ratios, per_step = [], []
    for _ in range(5):
        _, clean_us = kernels(lambda: [core._zero_dye_in_solids(dye, geom) for _ in range(50)], 50)
        k, step_us = kernels(lambda: s.advance(20), 20)
        ratios.append(clean_us / step_us)
        per_step.append(k)
    ratios.sort()
    print("\n== default-path cost (cuda, 1024x512, eager, kernel time) ==")
    print(f"  kernels per step {per_step[0]:.1f}; smoke clean-up / step kernel time "
          f"{100 * ratios[2]:.3f} % (range {100 * ratios[0]:.3f}-{100 * ratios[-1]:.3f} %)")
    out["default_cost"] = {"kernels_per_step": per_step[0], "cleanup_fraction_median": ratios[2],
                           "cleanup_fraction_range": [ratios[0], ratios[-1]]}


def thermal(out: dict, device: str, quick: bool = False) -> None:
    """The placed-physics validation of docs/theory_ns2d.md section 21 (validation/thermal2d_cases.py)."""
    res: dict = {}

    def show(r):
        print(r.row() if hasattr(r, "row") else r, flush=True)
        return r

    def rec(r) -> dict:
        return {"name": r.name, "resolution": r.resolution, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s,
                **r.values}

    grids = ({1e3: [32, 64], 1e4: [32, 64], 1e5: [64, 128], 1e6: [64, 128]} if quick else
             {1e3: [32, 64, 128], 1e4: [32, 64, 128], 1e5: [64, 128, 256], 1e6: [64, 128, 256]})
    cav = []
    for ra, ns in grids.items():
        rows = [show(tcases.run_cavity(ra, n, device=device)) for n in ns]
        entry = {"ra": ra, "rows": [rec(r) for r in rows]}
        if len(rows) >= 3:
            entry["richardson_nu"] = tcases.richardson([r.values["nu_hot"] for r in rows])
            print("   Richardson extrapolation of Nu:", entry["richardson_nu"], flush=True)
        cav.append(entry)
    res["cavity"] = cav
    cpds = (16,) if quick else (16, 32)
    res["heated_cylinder"] = [rec(show(tcases.run_heated_cylinder(re_, cpd, device=device)))
                              for re_ in (20.0, 40.0) for cpd in cpds]
    # Pr 0.70 against Lange et al.'s correlation and the Pr 0.70 solutions, with the blockage study
    pr07 = [(20.0, 0.05), (30.0, 0.05), (40.0, 0.05)]
    if not quick:
        pr07 += [(20.0, 0.025), (40.0, 0.025), (20.0, 0.0125), (40.0, 0.0125)]
    res["heated_cylinder_pr070"] = []
    for re_, blk in pr07:
        r = show(tcases.run_heated_cylinder(re_, 16, pr=0.70, device=device, **tcases.CYLINDER_DOMAINS[blk]))
        cmp_ = tcases.cylinder_reference_comparison(re_, r.values["nu"])
        print("   ", {k: round(v, 4) for k, v in cmp_.items()}, flush=True)
        res["heated_cylinder_pr070"].append({**rec(r), **cmp_})
    res["rb_onset"] = [show(tcases.rb_onset(1600.0, 1800.0, n, device=device)) for n in cpds]
    res["actuator_disc"] = [rec(show(tcases.run_actuator_disc(ct, cpd, propeller=prop, device=device)))
                            for prop in (False, True) for ct in (0.5, 0.8) for cpd in cpds]
    res["plane_jet"] = rec(show(tcases.run_plane_jet(device=device)))
    res["conduction_slabs"] = [show(tcases.run_conduction_slabs(16, frac=f, stiffness=st, device=device))
                               for st in (60.0, 250.0, 1000.0) for f in (0.0, 0.25, 0.5, 0.75)]
    res["flux_slab"] = [show(tcases.run_conduction_slabs(n, frac=f, flux_g=1.0, device=device))
                        for n in (16, 32) for f in (0.0, 0.5)]
    out["thermal"] = res


def fan_room(out: dict, cpls: list[int], device: str) -> None:
    """Grid study of the app's 2D fan room with and without the wall model (theory doc 18.4).

    Not a validation: it measures how far the room's mean speed is from grid-independent,
    which is why the 2D closed fan room is reported as qualitative. 1200 L/U per run,
    averaged over [800, 1200] (the rooms take 500-900 L/U to settle)."""
    rows = []
    for cpl in cpls:
        for wm in ("none", "werner_wengle"):
            r = tcases.run_fan_room(cpl, wall_model=wm, device=device)
            print(r.row(), flush=True)
            rows.append({"name": r.name, "cells_per_l": cpl, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s,
                         **{k: v for k, v in r.values.items() if k != "series"}})
    out["fan_room"] = rows


def verify(out: dict) -> None:
    res = {}
    # Poisson exactness
    worst = 0.0
    for periodic in (False, True):
        for nx, ny in ((128, 96), (255, 129)):
            ps = PoissonSolver(nx, ny, 0.1, periodic=periodic, device=torch.device("cpu"), dtype=torch.float64)
            f = torch.randn(ny, nx, dtype=torch.float64)
            f -= f.mean()
            worst = max(worst, float((ps.laplacian(ps.solve(f)) - f).abs().max() / f.abs().max()))
    res["poisson_rel_residual_f64"] = worst
    # projected divergence of solver states
    for dt_, dev in (("float64", "cpu"), ("float32", "cuda" if torch.cuda.is_available() else "cpu")):
        cfg = cases.cylinder_config(100, 16, upstream=4, downstream=8, half_height=4, device=dev, dtype=dt_)
        s = FlowSolver(cfg)
        worst = 0.0
        for _ in range(20):
            s.advance(5)
            worst = max(worst, float(s.field("divergence").abs().max()) * cfg.h)
        res[f"divergence_h_{dt_}"] = worst
    # Taylor-Green
    for adv in ("central", "upwind3"):
        errs = [cases.taylor_green_error(n, advection=adv)[0] for n in (16, 32, 64, 128)]
        res[f"tg_{adv}_errors"] = errs
        res[f"tg_{adv}_orders"] = [math.log2(errs[i] / errs[i + 1]) for i in range(3)]
    res["tg_decay_rate_ratio_64"] = cases.taylor_green_error(64)[1]
    # gradient vs central finite differences
    base = torch.as_tensor(cases.circle(32), dtype=torch.float64)
    direction = torch.stack([base[:, 0], torch.zeros_like(base[:, 1])], 1)

    def J(eps):
        cfg = SolverConfig(nx=48, ny=32, h=1 / 8, x0=-2.0, y0=-2.0, re=20.0, device="cpu", dtype="float64",
                           dye=False, perturb=False)
        ops = make_operators(cfg)
        geom = build_geometry([base + eps * direction], ops, window=(10, 22, 10, 22))
        _, c = rollout(init_state(cfg, ops), geom, ops, 20, dt=0.02, checkpoint_every=5)
        return c[:, 0].mean()
    eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(J(eps), eps)
    with torch.no_grad():
        e = 1e-4
        fd = float((J(torch.tensor(e, dtype=torch.float64)) - J(torch.tensor(-e, dtype=torch.float64))) / (2 * e))
    res["grad_autograd"] = float(g)
    res["grad_fd"] = fd
    res["grad_rel_err"] = abs(float(g) - fd) / abs(fd)
    print("\n== verification ==")
    for k, v in res.items():
        print(f"  {k:<26} {np.round(v, 6) if isinstance(v, list) else f'{v:.3e}'}")
    out["verify"] = res


def cylinder(out: dict, cpds: list[int], res_list: list[float], device: str) -> None:
    rows = []
    for re_ in res_list:
        for cpd in cpds:
            t_end, t_avg = (60.0, 50.0) if re_ <= 50 else (200.0, 100.0)
            r = cases.run_cylinder(re_, cpd, t_end, t_avg, device=device)
            print(r.row(), flush=True)
            rows.append({"name": r.name, "cells_per_d": cpd, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s,
                         **r.values})
    out["cylinder"] = rows


def ground(out: dict, device: str) -> None:
    cfg = cases.moving_ground_config(256 if device.startswith("cuda") else 64, device=device)
    s = FlowSolver(cfg)
    _sync(device)
    t0 = time.perf_counter()
    while s.t < 6.0:
        s.advance(500)
    _sync(device)
    wall = time.perf_counter() - t0
    u, v = s.field("u"), s.field("v")
    hst = s.history()
    m = hst["t"] > 3.0
    res = {"grid": (cfg.nx, cfg.ny), "steps": s.steps, "wall_s": wall, "t": s.t,
           "max_speed": float(torch.hypot(u, v).max()), "finite": bool(torch.all(torch.isfinite(u))),
           "cd_mean": float(hst["cd"][m].mean()), "cl_mean": float(hst["cl"][m].mean()),
           "nut_max": float(s.field("nut").max())}
    print("\n== moving ground, Re 1e6, upwind3 + WALE ==")
    for k, v in res.items():
        print(f"  {k:<10} {v}")
    out["ground"] = res


def channel(out: dict, cpds: list[int], device: str) -> None:
    rows = []
    for case in ("2D-1", "2D-2"):
        for cpd in cpds:
            r = cases.run_schafer_turek(case, cpd, device=device)
            print(r.row(), flush=True)
            rows.append({"name": r.name, "cells_per_d": cpd, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s,
                         **r.values})
    out["schafer_turek"] = rows


def square(out: dict, cpds: list[int], device: str) -> None:
    rows = []
    for cpd in cpds:
        r = cases.run_square(100.0, cpd, device=device)
        print(r.row(), flush=True)
        rows.append({"name": r.name, "cells_per_d": cpd, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s,
                     **r.values})
    out["square"] = rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["perf", "verify", "cylinder", "ground", "channel", "square", "physics-perf",
                                     "default-cost", "thermal", "fan-room", "all"])
    ap.add_argument("--quick", action="store_true", help="thermal: coarser grid studies (minutes, not an hour)")
    ap.add_argument("--cpd", type=int, nargs="+", default=None,
                    help="cells per diameter (default 32 48 64; channel 40 80; square 32 48)")
    ap.add_argument("--re", type=float, nargs="+", default=[40.0, 100.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    out: dict = {"torch": torch.__version__, "platform": platform.platform(),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "date": time.strftime("%Y-%m-%d %H:%M"), "mask_shift": core.MASK_SHIFT}
    if a.what in ("perf", "all"):
        perf(out)
    if a.what in ("verify", "all"):
        verify(out)
    if a.what in ("cylinder", "all"):
        cylinder(out, a.cpd or [32, 48, 64], a.re, a.device)
    if a.what in ("ground", "all"):
        ground(out, a.device)
    if a.what in ("channel", "all"):
        channel(out, a.cpd or [40, 80], a.device)
    if a.what in ("square", "all"):
        square(out, a.cpd or [32, 48], a.device)
    if a.what in ("physics-perf", "all"):
        physics_perf(out)
    if a.what in ("default-cost", "all"):
        default_cost(out)
    if a.what in ("thermal", "all"):
        thermal(out, a.device, quick=a.quick)
    if a.what == "fan-room":                    # hours, so not part of "all"
        fan_room(out, a.cpd or [6, 12, 24], a.device)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
