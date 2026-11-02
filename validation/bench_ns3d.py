"""Performance benchmark and validation runner for the 3D Navier-Stokes solver.

Usage (from the repository root)::

    python validation/bench_ns3d.py perf                       # steps/s and peak memory
    python validation/bench_ns3d.py verify                     # Poisson, divergence, TG, gradient
    python validation/bench_ns3d.py sphere --re 20 50 100 --cpd 16 24
    python validation/bench_ns3d.py domain --re 20 100 --cpd 16   # blockage study (wider boxes)
    python validation/bench_ns3d.py symmetry --cpd 16          # half domain vs full domain
    python validation/bench_ns3d.py unsteady --cpd 24          # sphere Re 300 (stretch case)
    python validation/bench_ns3d.py ground                     # moving road + wheels + WALE, Re 1e6
    python validation/bench_ns3d.py physics                    # cost / memory of each placed-physics feature
    python validation/bench_ns3d.py thermal --parts cavity sphere disc jet archimedes symmetry cavity2d
    python validation/bench_ns3d.py thermal --parts flux plume  # uniform-flux plate / sphere, Ri >> 1 steps
    python validation/bench_ns3d.py all --json out.json

Numbers printed here are the ones quoted in docs/theory_ns3d.md. Throughput is
measured on the fast path (FlowSolver3D.advance) after a warm-up that includes
the CUDA-graph capture, in float32, with the GPU synchronised around the timed
block. The development GPU is shared with other work: other processes lower
the numbers, so representative figures need an idle machine.
"""
from __future__ import annotations

import argparse
import dataclasses
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
_spec = importlib.util.spec_from_file_location("ns3d_cases", HERE / "ns3d_cases.py")
cases = importlib.util.module_from_spec(_spec)
sys.modules["ns3d_cases"] = cases
_spec.loader.exec_module(cases)

from aerolume.solvers.ns3d import (
    FlowSolver3D,
    ImmersedMesh,
    SolverConfig3D,
    build_geometry3d,
    core,
    init_state3d,
    make_operators3d,
    rollout3d,
)
from aerolume.solvers.ns3d.poisson import PoissonSolver3D


def _sync(dev) -> None:
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def bench_config(nx: int, ny: int, nz: int, device: str, **kw) -> SolverConfig3D:
    """Sphere (diameter 1, level-4 mesh) in a 12-diameter-long box with nx cells along x."""
    h = 12.0 / nx
    v, f = cases.icosphere(1.0, 4)
    base = {"nx": nx, "ny": ny, "nz": nz, "h": h, "x0": -4.0, "y0": -0.5 * ny * h, "z0": -0.5 * nz * h,
            "re": 200.0, "bodies": [ImmersedMesh(v, f, name="sphere")], "device": device, "dtype": "float32",
            "ref_area": math.pi / 4}
    base.update(kw)
    return SolverConfig3D(**base)


def steps_per_second(cfg: SolverConfig3D, *, n: int = 200, repeats: int = 3, averaging: bool = False) -> dict:
    """Best-of-``repeats`` throughput of FlowSolver3D.advance, and the peak memory above what
    was allocated before the solver was built (a previous solver's CUDA-graph pool is only
    released once it is garbage-collected, so collect first and subtract the baseline)."""
    base_alloc = 0
    if cfg.device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
    s = FlowSolver3D(cfg)
    if averaging:
        s.start_averaging()
    s.advance(min(n, 30))                     # warm-up: NVRTC compile, cuBLAS, graph capture
    best = 0.0
    for _ in range(repeats):
        _sync(cfg.device)
        t0 = time.perf_counter()
        s.advance(n)
        _sync(cfg.device)
        best = max(best, n / (time.perf_counter() - t0))
    out = {"steps_per_s": best, "fused": s.uses_fused_kernels, "graphs": s.uses_cuda_graphs,
           "cells": s.cells, "cell_steps_per_s": best * s.cells}
    if cfg.device.startswith("cuda"):
        out["peak_memory_gb"] = (torch.cuda.max_memory_allocated() - base_alloc) / 1e9
    del s
    return out


def perf(out: dict) -> None:
    rows = []
    if torch.cuda.is_available():
        for name, kw, avg in (("default (upwind3, laminar, fused, graphs)", {}, False),
                              ("default + running means", {}, True),
                              ("eager (no graphs)", {"use_cuda_graphs": False}, False),
                              ("reference core (no fused kernels)", {"fused_kernels": False}, False),
                              ("central advection", {"advection": "central"}, False),
                              ("upwind3 + WALE", {"turbulence": "wale", "re": 1e6}, False)):
            r = steps_per_second(bench_config(256, 128, 128, "cuda", **kw),
                                 n=100 if "no" in name else 200, averaging=avg)
            rows.append(("cuda", "256x128x128", name, r))
            print(f"  cuda 256x128x128 {name:<44} {r['steps_per_s']:7.1f} steps/s  "
                  f"{r.get('peak_memory_gb', 0):.2f} GB", flush=True)
        for grid in ((400, 176, 172),):              # 12.1 M cells
            r = steps_per_second(bench_config(*grid, "cuda"), n=60, averaging=True)
            rows.append(("cuda", "x".join(map(str, grid)), "default + running means", r))
            print(f"  cuda {'x'.join(map(str, grid))} default + means {r['steps_per_s']:7.1f} steps/s  "
                  f"peak {r['peak_memory_gb']:.2f} GB ({r['cells'] / 1e6:.1f} M cells)", flush=True)
    threads0 = torch.get_num_threads()
    for th in sorted({4, threads0}):
        torch.set_num_threads(th)
        r = steps_per_second(bench_config(96, 48, 48, "cpu"), n=10, repeats=2)
        rows.append(("cpu", "96x48x48", f"{th} threads", r))
        print(f"  cpu  96x48x48 {th} threads {r['steps_per_s']:7.2f} steps/s", flush=True)
    torch.set_num_threads(threads0)
    out["perf"] = [{"device": d, "grid": g, "options": k, **r} for d, g, k, r in rows]


def verify(out: dict) -> None:
    res: dict = {}
    worst = 0.0
    for per in ((False, False, False), (True, True, True), (False, True, False)):
        for nx, ny, nz in ((64, 48, 40), (63, 33, 17)):
            ps = PoissonSolver3D(nx, ny, nz, 0.1, periodic=per, device=torch.device("cpu"), dtype=torch.float64)
            f = torch.randn(nz, ny, nx, dtype=torch.float64)
            f -= f.mean()
            worst = max(worst, float((ps.laplacian(ps.solve(f)) - f).abs().max() / f.abs().max()))
    res["poisson_rel_residual_f64"] = worst
    if torch.cuda.is_available():
        ps = PoissonSolver3D(256, 128, 128, 0.05, periodic=(False, False, False), device=torch.device("cuda"),
                             dtype=torch.float32)
        f = torch.randn(128, 128, 256, device="cuda")
        f -= f.mean()
        res["poisson_rel_residual_f32_gpu"] = float((ps.laplacian(ps.solve(f)) - f).abs().max() / f.abs().max())
    for dt_, dev in (("float64", "cpu"), ("float32", "cuda" if torch.cuda.is_available() else "cpu")):
        cfg = cases.sphere_config(100.0, 8, device=dev, dtype=dt_, half=False, bc_zmin="moving", perturb=True)
        s = FlowSolver3D(cfg)
        w = 0.0
        for _ in range(10):
            s.advance(4)
            w = max(w, float(s.field("divergence").abs().max()) * cfg.h)
        res[f"divergence_h_{dt_}"] = w
    for adv in ("central", "upwind3"):
        for plane in ("xy", "xz", "yz"):
            errs = [cases.taylor_green_error(n, advection=adv, plane=plane)[0] for n in (16, 32, 64, 128)]
            res[f"tg_{adv}_{plane}_errors"] = errs
            res[f"tg_{adv}_{plane}_orders"] = [math.log2(errs[i] / errs[i + 1]) for i in range(3)]
    res["tg_decay_rate_ratio_64"] = cases.taylor_green_error(64, t_end=1.0)[1]
    # gradient of a short rollout w.r.t. a smooth vertex-displacement mode vs central FD (float64)
    v, fc = cases.icosphere(1.0, 1)
    base = torch.as_tensor(v, dtype=torch.float64)
    mode = torch.stack([base[:, 0], torch.zeros_like(base[:, 1]), 0.5 * base[:, 2]], dim=1)

    def J(eps, shared):
        cfg = SolverConfig3D(nx=20, ny=12, nz=12, h=0.2, x0=-1.6, y0=-1.2, z0=-1.2, re=20.0, device="cpu",
                             dtype="float64", perturb=False)
        ops = make_operators3d(cfg)
        geom = build_geometry3d([ImmersedMesh(base + eps * mode, fc)], ops, window=(1, 11, 1, 11, 2, 14),
                                use_shared=shared)
        _, c = rollout3d(init_state3d(cfg, ops), geom, ops, 10, dt=0.04, checkpoint_every=4)
        return c[:, 0].mean()
    for shared in (False, True):
        eps = torch.zeros((), dtype=torch.float64, requires_grad=True)
        (g,) = torch.autograd.grad(J(eps, shared), eps)
        with torch.no_grad():
            e = 1e-5
            fd = float((J(torch.tensor(e, dtype=torch.float64), shared)
                        - J(torch.tensor(-e, dtype=torch.float64), shared)) / (2 * e))
        key = "shared_sdf" if shared else "exact_sdf"
        res[f"grad_{key}_autograd"] = float(g)
        res[f"grad_{key}_fd"] = fd
        res[f"grad_{key}_rel_err"] = abs(float(g) - fd) / abs(fd)
    if torch.cuda.is_available():
        from aerolume.solvers.ns3d import fused
        cfg = cases.sphere_config(100.0, 8, device="cuda", dtype="float64", perturb=True, bc_zmin="moving",
                                  turbulence="wale")
        ops = make_operators3d(cfg)
        kern = fused.load_kernels(cfg.advection, ops.device, ops.dtype)
        with torch.no_grad():
            geom = build_geometry3d(cfg.bodies, ops)
            a = b = init_state3d(cfg, ops)
            for _ in range(20):
                a, _ = core.step3d_full(a, geom, ops)
                b, _ = fused.fast_step(b, geom, ops, kern)
        res["fused_vs_reference_f64_20_steps"] = max(float((getattr(a, n) - getattr(b, n)).abs().max())
                                                     for n in ("u", "v", "w", "p"))
    print("\n== verification ==")
    for k, v in res.items():
        print(f"  {k:<36} {np.round(v, 4) if isinstance(v, list) else f'{v:.4e}'}")
    out["verify"] = res


def _row(r) -> dict:
    return {"name": r.name, "cells_per_d": r.cells_per_d, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s,
            **r.values}


def sphere(out: dict, res_list: list[float], cpds: list[int], device: str) -> None:
    rows = []
    for re_ in res_list:
        for cpd in cpds:
            t_end = 40.0 if re_ >= 100 else 30.0
            r = cases.run_sphere(re_, cpd, t_end, device=device)
            print(r.row(), flush=True)
            rows.append(_row(r))
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    out["sphere"] = rows


def symmetry(out: dict, cpds: list[int], device: str) -> None:
    rows = []
    for cpd in cpds:
        for half in (True, False):
            r = cases.run_sphere(100.0, cpd, 30.0, device=device, half=half)
            print(r.row(), flush=True)
            rows.append(_row(r))
    out["symmetry"] = rows


def domain(out: dict, res_list: list[float], cpds: list[int], device: str) -> None:
    """Confinement study: the same sphere in wider boxes (blockage 2.85, 1.2, 0.7 %)."""
    rows = []
    for re_ in res_list:
        for cpd in cpds:
            for w in (2.625, 4.0, 5.25):
                t_end = 40.0 if re_ >= 100 else 30.0
                r = cases.run_sphere(re_, cpd, t_end, device=device, half_width=w)
                print(r.row(), flush=True)
                rows.append({**_row(r), "half_width": w})
    out["domain"] = rows


def unsteady(out: dict, cpds: list[int], device: str) -> None:
    rows = []
    for cpd in cpds:
        r = cases.run_sphere_unsteady(300.0, cpd, device=device)
        print(r.row(), flush=True)
        rows.append(_row(r))
    out["unsteady"] = rows


def ground(out: dict, device: str, cpl: int) -> None:
    r = cases.run_moving_road(cpl, device=device)
    print(r.row(), flush=True)
    out["ground"] = _row(r)


# ---------------------------------------------------------------------------
# placed physics (DESIGN.md section 10): cost, memory and validation
# ---------------------------------------------------------------------------

def _physics_variants() -> list[tuple[str, dict]]:
    """Configurations of the placed-physics cost table (on top of bench_config's sphere)."""
    from aerolume.solvers.ns3d import Emitter3D, Fan3D, Port3D, ThermalConfig3D
    v, f = cases.icosphere(1.0, 4)
    hot = [ImmersedMesh(v, f, name="sphere", thermal="temperature", thermal_value=1.0)]
    th = ThermalConfig3D(prandtl=0.71, richardson=0.3)
    fans = [Fan3D(center=(-2.0, 0.0, 0.8), direction=(1, 0, 0), diameter=0.8, mode="velocity", value=1.3)]
    ports = [Port3D(kind="jet", center=(1.0, 0.0, -1.5), direction=(0, 0, 1), diameter=0.4, speed=1.0, theta=0.5),
             Port3D(kind="vent", center=(6.0, 0.0, 1.5), diameter=0.4, speed=0.5)]
    smoke = [Emitter3D(kind="rake", position=(-2.5, -1.0, z), end=(-2.5, 1.0, z), count=8, channel=c)
             for c, z in enumerate((-0.6, -0.2, 0.2, 0.6))]
    parts = [Emitter3D(species="particles", kind="rake", position=(-2.5, -1.0, z), end=(-2.5, 1.0, z), count=16,
                       rate=4000.0) for z in (-0.4, 0.0, 0.4)]
    return [("default (no placed physics)", {}),
            ("diagnostics (one row / 10 steps)", {"diagnostics": True}),
            ("PID time step", {"adaptive_dt": "pid"}),
            ("heat transfer, heated body, Ri 0.3", {"bodies": hot, "thermal": th}),
            ("heat transfer + WALE (Re 1e5)", {"bodies": hot, "thermal": th, "turbulence": "wale", "re": 1e5}),
            ("fan (PI) + jet + vent", {"fans": fans, "ports": ports}),
            ("4 smoke channels", {"emitters": smoke}),
            ("3 particle rakes (3 x 65536)", {"emitters": parts}),
            ("everything", {"bodies": hot, "thermal": th, "fans": fans, "ports": ports, "emitters": smoke + parts,
                            "diagnostics": True})]


def _block_kernel_ms(s: FlowSolver3D, blocks: int, block: int = 10) -> tuple[float, float]:
    """(min-over-blocks kernel ms per step, launches per step) of ``blocks`` x ``block`` eager steps.

    The step sequence of an eager run repeats every ``block`` steps (one diagnostics / averaging
    step per block), so the k-th kernel of every block is the same launch; its smallest duration
    over the blocks is the least disturbed one, and their sum the block's cost. A remainder of
    events that are not part of a block (a host flush at the end of ``advance``) is trimmed."""
    from torch.profiler import ProfilerActivity, profile
    _sync(s.device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        s.advance(blocks * block)
        _sync(s.device)
    ev = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                key=lambda e: e.time_range.start)
    names = [e.name for e in ev]
    dur = np.array([e.time_range.elapsed_us() for e in ev], dtype=np.float64)
    r = len(ev) % blocks
    for a, b in ((0, len(ev) - r), (r, len(ev))):
        per = (b - a) // blocks
        seq = [names[a + i * per:a + (i + 1) * per] for i in range(blocks)]
        if per and all(x == seq[0] for x in seq):
            d = dur[a:b].reshape(blocks, per)
            return float(d.min(axis=0).sum()) / block / 1000.0, per / block
    return float(dur.sum()) / (blocks * block) / 1000.0, len(ev) / (blocks * block)


def kernel_ms_per_step(cfg: SolverConfig3D, *, blocks: int = 6, repeats: int = 3) -> dict:
    """GPU kernel time per step of this process (torch.profiler, eager so every kernel is
    attributed; graphs remove launch overhead, not kernel work): the median over ``repeats``
    of :func:`_block_kernel_ms`, and the spread between repeats.

    Why not wall time: on a GPU shared with other processes a step's wall time measures the
    other work as much as ours. Kernel durations are less exposed, but not immune -- on the
    shared development GPU (another process at 100 %) even they drifted by up to 2x between
    windows a minute apart -- so ``kernel_ms_spread`` (relative) is reported: a spread above a
    few per cent means the numbers describe the contention, and only an idle GPU gives the
    feature costs."""
    s = FlowSolver3D(dataclasses.replace(cfg, use_cuda_graphs=False))
    s.advance(30)                               # warm-up: NVRTC compile, cuBLAS handles, allocator
    vals, counts = [], []
    for _ in range(repeats):
        ms, k = _block_kernel_ms(s, blocks)
        vals.append(ms)
        counts.append(k)
    del s
    torch.cuda.empty_cache()
    med = float(np.median(vals))
    return {"kernel_ms_per_step": med, "kernel_ms_spread": float(max(vals) - min(vals)) / med,
            "kernels_per_step": float(np.median(counts))}


def physics_perf(out: dict, passes: int = 1) -> None:
    """Cost and peak memory of each placed-physics feature on the 256 x 128 x 128 sphere (fused
    kernels, float32), relative to the default path. The cost is the GPU kernel time per step
    (:func:`kernel_ms_per_step`, robust to other work on the GPU); the wall-clock throughput
    of the CUDA-graph path (best of ``passes``) and its peak memory are reported beside it --
    the wall clock is only representative on an idle GPU. The extra memory per cell of heat
    transfer feeds solver.BYTES_PER_CELL_THERMAL_F32."""
    if not torch.cuda.is_available():
        print("  physics perf needs CUDA")
        return
    variants = _physics_variants()
    best: dict[str, dict] = {}
    for _ in range(passes):                     # the best of several passes: other GPU work only slows a run
        for name, kw in variants:
            r = steps_per_second(bench_config(256, 128, 128, "cuda", **kw), n=100, repeats=2)
            if name not in best or r["steps_per_s"] > best[name]["steps_per_s"]:
                best[name] = r
            torch.cuda.empty_cache()
    for name, kw in variants:
        best[name].update(kernel_ms_per_step(bench_config(256, 128, 128, "cuda", **kw)))
    base = best[variants[0][0]]
    rows = []
    for name, _ in variants:
        r = best[name]
        r["relative_cost"] = r["kernel_ms_per_step"] / base["kernel_ms_per_step"]
        r["relative_wall"] = base["steps_per_s"] / r["steps_per_s"]
        r["extra_bytes_per_cell"] = (r["peak_memory_gb"] - base["peak_memory_gb"]) * 1e9 / r["cells"]
        rows.append({"options": name, **r})
        print(f"  cuda 256x128x128 {name:<38} kernels {r['kernel_ms_per_step']:7.2f} ms/step "
              f"(x{r['relative_cost']:.3f}, spread {100 * r['kernel_ms_spread']:.0f} %, {r['kernels_per_step']:.0f} "
              f"launches)  wall {r['steps_per_s']:6.1f} steps/s  peak {r['peak_memory_gb']:.2f} GB "
              f"(+{r['extra_bytes_per_cell']:.0f} B/cell)", flush=True)
    out["physics_perf"] = rows


def _thermal_cases():
    spec = importlib.util.spec_from_file_location("thermal3d_cases", HERE / "thermal3d_cases.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["thermal3d_cases"] = mod
    spec.loader.exec_module(mod)
    return mod


def _trow(r) -> dict:
    return {"name": r.name, "grid": r.grid, "steps": r.steps, "wall_s": r.wall_s, **r.values}


def thermal(out: dict, parts: list[str], device: str, args) -> None:
    """The validation cases of docs/theory_ns3d.md sections 15-17 (``parts`` selects them)."""
    tc = _thermal_cases()
    res: dict = {}
    if "cavity" in parts:
        for ra, ns in ((1e4, args.cavity_n4), (1e5, args.cavity_n5)):
            rows = []
            for n in ns:
                r = tc.run_cavity(ra, n, device=device)
                print(r.row(), flush=True)
                rows.append(_trow(r))
            if len(rows) >= 3:
                for key in ("nu_hot", "nu_sym_plane", "u_max", "w_max"):
                    rows[-1][f"richardson_{key}"] = tc.richardson([x[key] for x in rows], [1.0 / n for n in ns])
            res[f"cavity_{ra:g}"] = rows
    if "cavity2d" in parts:
        res["cavity2d"] = []
        for ra in (1e4, 1e5):
            for n in args.cavity2d_n:
                r = tc.run_cavity_2d(ra, n, device=device)
                print(r.row(), flush=True)
                res["cavity2d"].append(_trow(r))
    if "sphere" in parts:
        res["heated_sphere"] = []
        for re_ in args.sphere_re:
            for cpd in args.sphere_cpd:
                r = tc.run_heated_sphere(re_, cpd, device=device)
                print(r.row(), flush=True)
                res["heated_sphere"].append({**_trow(r), "re": re_, "cells_per_d": cpd})
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
    if "disc" in parts:
        res["actuator_disc"] = []
        for propeller in (True, False):
            for ct in args.disc_ct:
                for cpd in args.disc_cpd:
                    r = tc.run_actuator_disc(ct, cpd, propeller=propeller, device=device)
                    print(r.row(), flush=True)
                    res["actuator_disc"].append({**_trow(r), "cells_per_d": cpd})
    if "jet" in parts:
        res["round_jet"] = []
        for spec in args.jet:
            cpd, hw = (float(x) for x in spec.split(":"))
            r = tc.run_round_jet(100.0, int(cpd), half_width=hw, device=device)
            print(r.row(), flush=True)
            res["round_jet"].append({**_trow(r), "re_j": 100.0, "cells_per_d": int(cpd), "half_width": hw,
                                     "x_fit": [10.0, 22.0]})
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    if "archimedes" in parts:
        res["archimedes"] = []
        for n in args.archimedes_n:
            r = tc.run_archimedes(n, device=device, dtype="float64")
            print(r.row(), flush=True)
            res["archimedes"].append(_trow(r))
    if "flux" in parts:
        res["flux_plate"] = []
        for re_ in args.plate_re:
            for n in args.plate_n:
                for t in args.plate_t:
                    r = tc.run_flux_plate(re_, n, t_end=20.0, thickness=t, device=device)
                    print(r.row(), flush=True)
                    res["flux_plate"].append({**_trow(r), "re": re_, "cells_per_l": n, "thickness": t})
        res["flux_sphere"] = []
        for cpd in args.sphere_cpd:
            r = tc.run_flux_sphere(100.0, cpd, device=device)
            print(r.row(), flush=True)
            res["flux_sphere"].append({**_trow(r), "re": 100.0, "cells_per_d": cpd})
    if "plume" in parts:
        res["buoyant_sphere"] = []
        for spec in args.plume:
            re_, ri, cpd = (float(x) for x in spec.split(":"))
            r = tc.run_buoyant_sphere_steps(re_, ri, int(cpd), 40.0, device=device)
            print(r.row(), flush=True)
            res["buoyant_sphere"].append({**_trow(r), "re": re_, "cells_per_d": int(cpd)})
    if "symmetry" in parts:
        res["emitter_symmetry"] = tc.emitter_symmetry()
        print("  emitter symmetry (half vs full, max rel. difference):", res["emitter_symmetry"], flush=True)
    out["thermal"] = res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["perf", "verify", "sphere", "domain", "symmetry", "unsteady", "ground", "physics",
                                     "thermal", "all"])
    ap.add_argument("--cpd", type=int, nargs="+", default=None, help="cells per diameter (sphere 16 24)")
    ap.add_argument("--re", type=float, nargs="+", default=[20.0, 50.0, 100.0])
    ap.add_argument("--cpl", type=int, default=64, help="cells per length for the moving-road case")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=Path, default=None)
    th = ap.add_argument_group("thermal (placed physics validation)")
    th.add_argument("--parts", nargs="+",
                    default=["cavity", "cavity2d", "sphere", "disc", "jet", "archimedes", "symmetry"],
                    choices=["cavity", "cavity2d", "sphere", "disc", "jet", "archimedes", "symmetry", "flux",
                             "plume"])
    th.add_argument("--cavity-n4", type=int, nargs="+", default=[24, 48, 96], help="cube grids at Ra 1e4")
    th.add_argument("--cavity-n5", type=int, nargs="+", default=[32, 64, 128], help="cube grids at Ra 1e5")
    th.add_argument("--cavity2d-n", type=int, nargs="+", default=[64, 128])
    th.add_argument("--sphere-re", type=float, nargs="+", default=[50.0, 100.0])
    th.add_argument("--sphere-cpd", type=int, nargs="+", default=[12, 16, 24])
    th.add_argument("--disc-ct", type=float, nargs="+", default=[0.4, 0.8])
    th.add_argument("--disc-cpd", type=int, nargs="+", default=[12, 16])
    th.add_argument("--archimedes-n", type=int, nargs="+", default=[16, 24, 32])
    th.add_argument("--plate-re", type=float, nargs="+", default=[400.0, 1000.0],
                    help="uniform-flux plate Reynolds numbers (part flux)")
    th.add_argument("--plate-n", type=int, nargs="+", default=[32, 48, 64, 96, 128], help="plate cells per chord")
    th.add_argument("--plate-t", type=float, nargs="+", default=[0.1], help="plate thickness / chord")
    th.add_argument("--plume", nargs="+", default=["100:1:8", "100:10:8", "100:100:8", "100:627:8", "100:627:16",
                                                   "20:627:8"],
                    help="buoyant sphere step-count runs as re:ri:cells_per_d (part plume)")
    th.add_argument("--jet", nargs="+", default=["4:12", "6:12", "4:24"],
                    help="round jet runs as cells_per_d:box_half_width (grid and confinement study)")
    a = ap.parse_args(argv)
    out: dict = {"torch": torch.__version__, "platform": platform.platform(),
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "date": time.strftime("%Y-%m-%d %H:%M")}
    if a.what in ("perf", "all"):
        perf(out)
    if a.what in ("verify", "all"):
        verify(out)
    if a.what in ("sphere", "all"):
        sphere(out, a.re, a.cpd or [16, 24], a.device)
    if a.what in ("domain", "all"):
        domain(out, a.re, a.cpd or [16], a.device)
    if a.what in ("symmetry", "all"):
        symmetry(out, a.cpd or [16], a.device)
    if a.what in ("unsteady", "all"):
        unsteady(out, a.cpd or [24], a.device)
    if a.what in ("ground", "all"):
        ground(out, a.device, a.cpl)
    if a.what in ("physics", "all"):
        physics_perf(out)
    if a.what in ("thermal", "all"):
        thermal(out, a.parts, a.device, a)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
