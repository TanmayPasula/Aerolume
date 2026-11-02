"""Headless command line: ``aerolume-cli <command>``.

Commands::

    doctor                          versions, devices, CUDA, a 1-second solver check
    bench    [--device D] [--quick] solver throughput on this machine
    polar    --naca 2412 --re 1e6 [--alpha -4:14:1] [--csv out.csv]
    quick    (--template sedan | --naca 0012) [--speed 30] [--alpha 4] [--quality draft]
             [--report out.pdf] [--save dir] [--sensitivity cd]
                                    plan + run one CFD simulation, print the results
             [--heated 60] [--fan X,Y:D:THRUST] [--closed X0,X1,Y0,Y1]
                                    (heat transfer, fans and closed rooms: see "Placed physics" below)
    quick3d  (--stl car.stl | --sphere 0.2) [--speed 30] [--yaw 5] [--quality draft] [--save dir]
             [--heated 60] [--fan X,Y,Z:D:THRUST] [--closed X0,X1,Y0,Y1,Z0,Z1]
                                    plan + run one 3D CFD simulation (STL / OBJ / PLY or a sphere)
    cavity   [--ra 1e4] [--3d] [--quality draft]
                                    the differentially heated cavity (natural convection) against
                                    de Vahl Davis (1983, 2D) / Tric et al. (2000, 3D)
    wing     --naca 2412 --span 10 --chord 1 [--taper 0.5] [--sweep 10] --alpha -4:12:1
             [--method vlm|llt] [--csv out.csv]
                                    finite-wing polar (vortex lattice / lifting line, viscous strips)
    run      <project dir> <simulation id or name>     run a saved simulation (2D, 3D or wing)
    script   FILE [--project P [--create]] [-- args]   run a Python script against the scripting API
                                    (``project``, ``al``, ``args``, ``log``, ``show`` bound; docs/scripting.md)
    validate [--full]               quick verification / validation subset

Placed physics (DESIGN.md 10) on ``quick`` / ``quick3d``: ``--heated T`` holds every
body at T °C and switches heat transfer on (with buoyancy unless ``--no-buoyancy``;
``--ambient`` sets the air temperature, default 15 °C, the standard sea-level air
the commands use); ``--fan X,Y[,Z]:DIAMETER:VALUE[:MODE[:DX,DY[,DZ]]]`` (repeatable)
places an actuator-disc fan in the tunnel frame, MODE ``thrust`` (N, or N per metre of
span in 2D; the default), ``pressure`` (Pa across the disc) or ``velocity`` (m/s
through it), blowing along +x unless a direction is given; ``--closed`` puts the case
in a closed box with walls on every side (a room). The results then add the Nusselt
number and heat rate, each fan's delivered thrust and flow, and the solver
diagnostics.

Every command prints plain, aligned tables (numbers through ``core.units``) and
returns a process exit code: 0 on success, 1 when a check failed, the run could
not be done or was stopped, 2 for bad arguments. During ``quick`` and ``run`` the
first Ctrl+C stops the simulation at the next chunk and keeps the partial result
(``run`` saves it, marked cancelled); a second Ctrl+C aborts. Heavy modules
(torch, the solvers) are imported only by the commands that need them, so
``--help`` is instant.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import math
import os
import platform
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .core import units

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def _out(text: str = "") -> None:
    print(text, flush=True)


def _err(text: str) -> None:
    # stdout first: a script's own print()s are block-buffered when piped to a log file, and
    # would otherwise land after the error or "finished" line that follows them
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    print(text, file=sys.stderr, flush=True)


def table(headers: Sequence[str], rows: Sequence[Sequence], align: str | None = None) -> str:
    """Plain-text table; ``align`` is one 'l'/'r' per column (default: first left, rest right)."""
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    n = len(headers)
    align = align or ("l" + "r" * (n - 1))
    widths = [max(len(r[i]) for r in cells) for i in range(n)]

    def line(r: list[str]) -> str:
        return "  ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(r, widths, align)).rstrip()
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(cells[0]), sep] + [line(r) for r in cells[1:]])


def kv(rows: Sequence[tuple[str, str]], indent: int = 2) -> str:
    """Aligned 'label   value' lines."""
    if not rows:
        return ""
    w = max(len(k) for k, _ in rows)
    return "\n".join(" " * indent + f"{k.ljust(w)}  {v}" for k, v in rows)


def _stat_text(stat, digits: int = 4) -> str:
    """'1.234 ± 0.0056'; a value whose error is below a millionth of it (a steady flow: nothing
    fluctuates to average) reads '1.234 (steady)' instead of '± 0'."""
    if stat is None or not math.isfinite(stat.mean):
        return "—"
    s = units.sig(stat.mean, digits)
    if math.isfinite(stat.stderr):
        if stat.stderr <= 1e-6 * abs(stat.mean):
            return s + " (steady)"
        s += f" ± {units.sig(stat.stderr, 2)}"
    return s


class ProgressPrinter:
    """Progress for a terminal: one updating line on a TTY, sparse lines otherwise."""

    def __init__(self, stream=None, *, every_s: float = 5.0):
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.every_s = every_s
        self._last = -math.inf
        self._last_head = ""
        self._width = 0

    def __call__(self, fraction: float, message: str) -> None:
        now = time.monotonic()
        head = message.split("(")[0].strip()
        if self.tty:
            if now - self._last < 0.1 and fraction < 1.0:
                return
            bar = round(24 * fraction)
            text = f"[{'#' * bar}{'.' * (24 - bar)}] {100 * fraction:5.1f} %  {message}"
            pad = max(0, self._width - len(text))
            self.stream.write("\r" + text + " " * pad)
            self._width = len(text)
            if fraction >= 1.0:
                self.stream.write("\n")
                self._width = 0
            self.stream.flush()
        elif fraction >= 1.0 or now - self._last >= self.every_s or head != self._last_head:
            self.stream.write(f"  {100 * fraction:5.1f} %  {message}\n")
            self.stream.flush()
        else:
            return
        self._last, self._last_head = now, head

    def close(self) -> None:
        if self.tty and self._width:
            self.stream.write("\n")
            self.stream.flush()
            self._width = 0


class StopOnInterrupt:
    """Context manager: the first Ctrl+C sets an Event (the run stops at its next
    chunk and returns a partial result), a second one raises KeyboardInterrupt.

    A simulation that has run for twenty minutes should not lose everything to
    one keystroke; the runners already stop cleanly on a ``cancel`` event. Signal
    handlers can only be installed from the main thread, so elsewhere (tests
    calling :func:`main` from a worker) this is a plain Event.
    """

    def __init__(self) -> None:
        self.event = threading.Event()
        self._old = None

    def __enter__(self) -> threading.Event:
        if threading.current_thread() is threading.main_thread():
            self._old = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handler)
        return self.event

    def _handler(self, signum, frame) -> None:
        if self.event.is_set():
            raise KeyboardInterrupt
        self.event.set()
        _err("\nStopping at the next step block and keeping the partial result (Ctrl+C again to abort)…")

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
            self._old = None


# ---------------------------------------------------------------------------
# argument helpers
# ---------------------------------------------------------------------------

def parse_alpha_range(text: str) -> tuple[float, float, float]:
    """'-4:14:1' -> (-4, 14, 1); '4' -> (4, 4, 1); '0:10' -> step 1."""
    parts = [p for p in str(text).replace(",", ":").split(":") if p.strip() != ""]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad angle range {text!r}; use start:end:step, e.g. -4:14:1") from None
    if len(vals) == 1:
        return vals[0], vals[0], 1.0
    if len(vals) == 2:
        return vals[0], vals[1], 1.0
    if len(vals) == 3 and vals[2] > 0:
        return vals[0], vals[1], vals[2]
    raise argparse.ArgumentTypeError(f"bad angle range {text!r}; use start:end:step with step > 0")


def _speed(text: str) -> float:
    try:
        v = units.parse(text, "speed")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if not v > 0:
        raise argparse.ArgumentTypeError("speed must be positive")
    return v


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not v > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def _param(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"template parameter {text!r} must be name=value")
    k, v = text.split("=", 1)
    return k.strip(), v.strip()


def _numbers(text: str, what: str, counts: tuple[int, ...]) -> tuple[float, ...]:
    """'1.5,0.4' -> (1.5, 0.4): comma-separated finite numbers, as many as one of ``counts``."""
    try:
        vals = tuple(float(p) for p in str(text).replace(" ", "").split(",") if p != "")
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad {what} {text!r}: use comma-separated numbers") from None
    if len(vals) not in counts or not all(math.isfinite(v) for v in vals):
        need = " or ".join(str(c) for c in counts)
        raise argparse.ArgumentTypeError(f"bad {what} {text!r}: give {need} finite numbers separated by commas")
    return vals


#: ``--fan`` modes and the model's names for them
_FAN_MODES = {"thrust": "thrust", "pressure": "pressure_jump", "pressure_jump": "pressure_jump",
              "velocity": "velocity"}


def parse_fan(text: str) -> dict:
    """``X,Y[,Z]:DIAMETER:VALUE[:MODE[:DX,DY[,DZ]]]`` -> {"position", "diameter", "value", "mode", "direction"}.

    The position and direction are in the tunnel (wind) frame, metres; MODE is ``thrust``
    (N, or N/m in 2D; the default), ``pressure`` (Pa across the disc) or ``velocity`` (m/s through
    it). Examples: ``1.5,0.4:0.3:10`` (10 N/m along +x), ``0.4,0,0.3:0.2:3:velocity:0,0,1``.
    """
    parts = str(text).split(":")
    if not 3 <= len(parts) <= 5:
        raise argparse.ArgumentTypeError(f"bad fan {text!r}: use X,Y[,Z]:DIAMETER:VALUE[:MODE[:DX,DY[,DZ]]], "
                                         "e.g. 1.5,0.4:0.3:10")
    pos = _numbers(parts[0], "fan position", (2, 3))
    try:
        diameter, value = float(parts[1]), float(parts[2])
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad fan {text!r}: the diameter and the value must be numbers") from None
    if not (diameter > 0 and math.isfinite(diameter)) or not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"bad fan {text!r}: the diameter must be positive and the value finite")
    mode = parts[3].strip().lower() if len(parts) > 3 and parts[3].strip() else "thrust"
    if mode not in _FAN_MODES:
        raise argparse.ArgumentTypeError(f"bad fan mode {mode!r}: use thrust, pressure or velocity")
    direction = _numbers(parts[4], "fan direction", (len(pos),)) if len(parts) > 4 else (1.0,) + (0.0,) * (len(pos) - 1)
    if math.hypot(*direction) < 1e-12:
        raise argparse.ArgumentTypeError(f"bad fan {text!r}: the direction must not be zero")
    return {"position": pos, "diameter": diameter, "value": value, "mode": _FAN_MODES[mode], "direction": direction}


def parse_box(text: str) -> tuple[float, ...]:
    """``X0,X1,Y0,Y1[,Z0,Z1]`` -> the walls of a closed box [m] (min < max on every axis)."""
    vals = _numbers(text, "box", (4, 6))
    if any(vals[2 * i + 1] <= vals[2 * i] for i in range(len(vals) // 2)):
        raise argparse.ArgumentTypeError(f"bad box {text!r}: every axis needs min < max (X0,X1,Y0,Y1[,Z0,Z1])")
    return vals


def _celsius(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a temperature: {text!r} (°C)") from None
    if not (math.isfinite(v) and v > -273.15):
        raise argparse.ArgumentTypeError("the temperature must be above absolute zero (-273.15 °C)")
    return v


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

_PACKAGES = (("numpy", "numpy"), ("scipy", "scipy"), ("torch", "torch"), ("matplotlib", "matplotlib"),
             ("PySide6", "PySide6-Essentials"), ("pyqtgraph", "pyqtgraph"), ("ezdxf", "ezdxf"),
             ("Pillow", "pillow"), ("contourpy", "contourpy"), ("VTK (3D viewer)", "vtk"))


def _version(dist: str) -> str:
    from importlib import metadata
    for name in (dist, dist.split("-")[0]):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "not installed"


def _solver_check(device: str) -> tuple[bool, str]:
    """A tiny cylinder case: does the solver build, step and give a sane drag?"""
    import numpy as np

    from .solvers.ns2d import FlowSolver, ImmersedBody, SolverConfig
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.c_[0.5 * np.cos(th), 0.5 * np.sin(th)]
    cfg = SolverConfig(nx=160, ny=96, h=1 / 8, x0=-5.0, y0=-6.0, re=40.0, bodies=[ImmersedBody(circle)],
                       device=device, dtype="float32", dye=False)
    t0 = time.perf_counter()
    s = FlowSolver(cfg)
    s.advance(20)
    t1 = time.perf_counter()
    s.advance(100)
    rate = 100 / (time.perf_counter() - t1)
    cd = float(s.coefficients()[0])
    ok = math.isfinite(cd) and 0.5 < cd < 10.0
    return ok, f"{'OK' if ok else 'FAILED'}  ({rate:,.0f} steps/s on a 160 × 96 grid, set-up {time.perf_counter() - t0:.1f} s)"


def _solver_check3d(device: str) -> tuple[bool, str]:
    """A tiny 3D sphere: does the 3D solver build (fused kernels on CUDA), step and give a sane drag?"""
    from .geometry3d import sphere
    from .solvers.ns3d import FlowSolver3D, ImmersedMesh, SolverConfig3D
    m = sphere(1.0, 24)
    cfg = SolverConfig3D(nx=64, ny=16, nz=32, h=1 / 8, x0=-3.0, y0=0.0, z0=-2.0, re=50.0, bc_y="symmetry_ymin",
                         bodies=[ImmersedMesh(m.vertices, m.faces, name="sphere")], ref_area=math.pi / 4,
                         device=device, dtype="float32")
    t0 = time.perf_counter()
    s = FlowSolver3D(cfg)
    s.advance(20)
    t1 = time.perf_counter()
    s.advance(100)
    rate = 100 / (time.perf_counter() - t1)
    cd = float(s.coefficients()[0])
    ok = math.isfinite(cd) and 0.5 < cd < 20.0
    path = ("fused kernels" if s.uses_fused_kernels else "reference path") if device.startswith("cuda") else "reference path"
    return ok, (f"{'OK' if ok else 'FAILED'}  ({rate:,.0f} steps/s on a 64 × 16 × 32 grid, {path}, set-up "
                f"{time.perf_counter() - t0:.1f} s)")


def _vtk_report(probe: bool) -> tuple[bool, list[tuple[str, str]]]:
    """VTK version and, with ``probe``, whether it renders off-screen here (the 3D viewer and the report
    figures need that). A missing VTK is reported, not fatal: 2D work does not need it."""
    ver = _version("vtk")
    if ver == "not installed":
        return False, [("VTK", ("not installed: the 3D viewer and 3D report figures are unavailable "
                                "(pip install vtk)"))]
    rows = [("VTK", ver)]
    if not probe:
        return True, rows
    try:
        from .viz3d import backend_info
        info = backend_info()
    except Exception as exc:  # noqa: BLE001 - a broken install is a finding
        return False, rows + [("Off-screen rendering", f"FAILED ({type(exc).__name__}: {exc})")]
    if info.get("ok"):
        gl = ", ".join(x for x in (info.get("renderer", ""), info.get("version", "")) if x)
        rows.append(("Off-screen rendering", f"OK ({info.get('window', '')}{'; ' + gl if gl else ''})"))
        return True, rows
    return False, rows + [("Off-screen rendering", f"FAILED ({info.get('error') or 'no OpenGL context'})")]


def cmd_doctor(args: argparse.Namespace) -> int:
    _out(f"Aerolume {__version__} — system check")
    _out()
    _out(kv([("Python", f"{platform.python_version()} ({platform.python_implementation()})"),
             ("Platform", platform.platform()),
             ("Executable", sys.executable)]))
    _out()
    _out("Packages")
    _out(kv([(name, _version(dist)) for name, dist in _PACKAGES]))
    _out()
    ok_all = True
    try:
        import torch

        from .core import device as devmod
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        _out(f"PyTorch could not be imported: {type(exc).__name__}: {exc}")
        return 1
    rows = [("torch build", f"{torch.__version__}; CUDA {torch.version.cuda or 'none'}"),
            ("CUDA available", "yes" if torch.cuda.is_available() else "no"),
            ("CPU threads (torch)", str(torch.get_num_threads()))]
    if torch.cuda.is_available():
        rows.append(("CUDA devices", str(torch.cuda.device_count())))
    _out("Compute")
    _out(kv(rows))
    _out()
    _out("Devices")
    for kind in devmod.available():
        info = devmod.info(kind)
        mem = f", {info.memory_gb:.1f} GB" if info.memory_gb else ""
        _out(f"  {kind:<5} {info.name}{mem}")
        if args.no_solver:
            continue
        try:
            ok, msg = _solver_check(kind)
        except Exception as exc:  # noqa: BLE001 - a broken device is a finding, not a crash
            ok, msg = False, f"FAILED  ({type(exc).__name__}: {exc})"
        ok_all &= ok
        _out(f"        solver check: {msg}")
        try:
            from .simulation.live import cpu_threads
            with cpu_threads(kind):                  # the thread count a real run uses
                ok, msg = _solver_check3d(kind)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"FAILED  ({type(exc).__name__}: {exc})"
        ok_all &= ok
        _out(f"        3D solver check: {msg}")
    _out()
    _out("3D viewer")
    vtk_ok, vtk_rows = _vtk_report(probe=not args.no_solver)
    _out(kv(vtk_rows))
    ok_all &= vtk_ok
    _out()
    _out("All checks passed." if ok_all else "Some checks FAILED (see above).")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------

def _bench_rate(nx: int, ny: int, device: str, turbulence: str, steps: int) -> float:
    import numpy as np
    import torch

    from .solvers.ns2d import FlowSolver, ImmersedBody, SolverConfig
    th = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    h = 24.0 / nx
    cfg = SolverConfig(nx=nx, ny=ny, h=h, x0=-6.0, y0=-0.5 * ny * h, re=1e5 if turbulence != "none" else 200.0,
                       bodies=[ImmersedBody(np.c_[0.5 * np.cos(th), 0.5 * np.sin(th)])], device=device,
                       dtype="float32", turbulence=turbulence)
    s = FlowSolver(cfg)
    s.advance(min(steps, 40))                        # warm-up incl. CUDA-graph capture

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    best = 0.0
    for _ in range(2):
        sync()
        t0 = time.perf_counter()
        s.advance(steps)
        sync()
        best = max(best, steps / (time.perf_counter() - t0))
    return best


def cmd_bench(args: argparse.Namespace) -> int:
    from .core import device as devmod
    from .simulation.live import cpu_threads
    from .simulation.plan import estimate_steps_per_second, record_throughput
    devices = devmod.available() if args.device in (None, "auto", "all") else (devmod.resolve(args.device),)
    rows = []
    _out(f"Solver throughput (float32, cylinder benchmark, best of 2){' — quick' if args.quick else ''}")
    for dev in devices:
        if dev.startswith("cpu"):
            grids = [(256, 128), (512, 256)]
            steps = 30
        else:
            grids = [(512, 256), (1024, 512)] + ([] if args.quick else [(2048, 1024)])
            steps = 200
        flavours = ("none",) if args.quick else ("none", "wale")
        with cpu_threads(dev):
            for nx, ny in grids:
                for turb in flavours:
                    try:
                        r = _bench_rate(nx, ny, dev, turb, steps)
                    except Exception as exc:  # noqa: BLE001
                        rows.append((dev, f"{nx} × {ny}", "laminar" if turb == "none" else "LES", "error",
                                     type(exc).__name__, ""))
                        continue
                    est = estimate_steps_per_second(dev, nx * ny, turbulence=turb, dtype="float32")
                    record_throughput(dev, nx * ny, r, turbulence=turb, dtype="float32")
                    rows.append((dev, f"{nx} × {ny}", "laminar" if turb == "none" else "LES (WALE)",
                                 f"{r:,.0f}", f"{est:,.0f}", f"{r / est:.2f}"))
    _out(table(["device", "grid", "flow model", "steps/s", "model", "ratio"], rows, "lllrrr"))
    _out()
    _out("'model' is the planner's run-time model before this measurement; ratios far from 1 mean "
         "this machine is faster (> 1) or slower than the reference machine.")
    return 0


# ---------------------------------------------------------------------------
# polar
# ---------------------------------------------------------------------------

def _airfoil_design(args: argparse.Namespace):
    from .geometry import design_from_airfoil, naca, read_airfoil_dat
    chord = float(getattr(args, "chord", 1.0) or 1.0)
    if getattr(args, "dat", None):
        name, coords = read_airfoil_dat(Path(args.dat))
        return design_from_airfoil(coords, name or Path(args.dat).stem, chord=chord,
                                   source={"type": "dat", "path": str(args.dat)})
    code = str(args.naca).strip()
    return design_from_airfoil(naca(code, 240, 1.0), f"NACA {code}", chord=chord,
                               source={"type": "naca", "code": code})


def cmd_polar(args: argparse.Namespace) -> int:
    from .analysis import polar_metrics
    from .core import fluid as F
    from .model import AirfoilSettings, FlowConditions, Simulation
    from .simulation.runner import run_polar
    design = _airfoil_design(args)
    air = F.air_isa(0.0)
    speed = args.re * air.viscosity / (air.density * design.length_ref)
    a0, a1, da = args.alpha
    sim = Simulation(name="polar", design_id=design.id, analysis="polar",
                     conditions=FlowConditions(air, speed, 0.0),
                     airfoil=AirfoilSettings(alpha_start=a0, alpha_end=a1, alpha_step=da, ncrit=args.ncrit,
                                             xtr_upper=args.xtr, xtr_lower=args.xtr, mach=args.mach))
    _out(f"{design.name}: Re = {units.fmt_reynolds(args.re)}, Mach {args.mach:g}, Ncrit {args.ncrit:g}, "
         f"alpha {a0:g} to {a1:g} step {da:g}")
    pp = ProgressPrinter()
    try:
        res = run_polar(design, sim, progress=pp)
    finally:
        pp.close()
    rows = []
    for p in res.points:
        ld = p.cl / p.cd if p.converged and p.cd > 0 else float("nan")
        rows.append((f"{p.alpha:g}", units.sig(p.cl, 4), units.sig(p.cd, 3), units.sig(p.cdp, 3),
                     units.sig(p.cdf, 3), units.sig(p.cm, 3), units.sig(ld, 3), units.sig(p.xtr_upper, 3),
                     units.sig(p.xtr_lower, 3), "yes" if p.converged else "NO"))
    _out(table(["alpha°", "Cl", "Cd", "Cdp", "Cdf", "Cm c/4", "L/D", "xtr up", "xtr lo", "conv."], rows))
    m = polar_metrics(res)
    _out()
    if m.get("cl_max_reached") == 0.0:          # still rising at the last angle: not a maximum
        cl_max = (f">= {units.sig(m['cl_max'], 4)} (still rising at {units.sig(m['alpha_cl_max'], 3)}°; "
                  "stall not reached)")
    else:
        cl_max = f"{units.sig(m['cl_max'], 4)} at {units.sig(m['alpha_cl_max'], 3)}°"
    _out(kv([("Cl max", cl_max),
             ("Best L/D", f"{units.sig(m['ld_max'], 4)} at {units.sig(m['alpha_ld_max'], 3)}°"),
             ("Cd min", f"{units.sig(m['cd_min'], 3)} at Cl {units.sig(m['cl_at_cd_min'], 3)}"),
             ("Lift slope", f"{units.sig(m['lift_slope_per_deg'], 4)} per degree"),
             ("Zero-lift angle", f"{units.sig(m['alpha_zero_lift'], 3)}°"),
             ("Run time", f"{res.wall_seconds:.1f} s")], indent=0))
    for w in res.warnings:
        _out(f"! {w}")
    if args.csv:
        from .export import polar_csv
        path = polar_csv(res, args.csv)
        _out(f"Wrote {path}")
    return 0 if any(p.converged for p in res.points) else 1


# ---------------------------------------------------------------------------
# CFD (quick / run)
# ---------------------------------------------------------------------------

def _print_plan(plan, *, verbose: bool) -> None:
    rows = plan.summary()
    headless = getattr(plan, "est_seconds_headless", math.nan)
    if "Estimated run time" in rows and math.isfinite(headless):
        # the summary's estimate includes the app's live view, which the CLI does not render
        from .simulation.plan import fmt_duration
        rows["Estimated run time"] = fmt_duration(headless)
    _out(kv(list(rows.items())))
    if verbose:
        for n in plan.notes:
            _out(f"  · {n}")
    for w in plan.warnings:
        _out(f"  ! {w}")


def _rel_text(stat) -> str:
    """'(±4.3 %)' -- the relative standard error, when it means something."""
    if stat is None or not (math.isfinite(stat.stderr) and math.isfinite(stat.mean)) or stat.mean == 0:
        return ""
    if stat.stderr <= 1e-6 * abs(stat.mean):
        return ""                                   # steady: _stat_text says so
    return f" (±{units.sig(100 * abs(stat.stderr / stat.mean), 2)} %)"


def _print_cfd_result(res, design, *, verbose: bool = False) -> None:
    sens = [s for s in res.surface if getattr(s, "sensitivity", None) is not None]
    rows = [("Cd", _stat_text(res.cd) + _rel_text(res.cd)), ("Cl", _stat_text(res.cl) + _rel_text(res.cl)),
            ("Cm", _stat_text(res.cm)),
            ("L/D", units.sig(res.lift_to_drag, 3)),
            ("Drag", f"{units.fmt(res.drag, 'force_per_span')} (per metre of span)"),
            ("Lift", f"{units.fmt(res.lift, 'force_per_span')}" + (" (downforce)" if res.lift < 0 else "")),
            ("Drag area Cd·ref", f"{units.sig(res.drag_area, 4)} m²/m"),
            ("Power to overcome drag", units.fmt(res.power, "power") + " per metre"),
            ("Strouhal number", units.sig(res.strouhal, 3) if res.strouhal else "— (no periodic shedding)"),
            ("Reynolds (real / simulated)", f"{units.fmt_reynolds(res.re_real)} / {units.fmt_reynolds(res.re_sim)}"),
            ("Averaged over", f"t = {units.sig(res.t_avg_start, 3)} … {units.sig(res.t_end, 3)} L/U"),
            ("Converged", "yes" if res.converged else "NO (see below)"),
            ("Sensitivity map", f"{len(sens)} bod{'y' if len(sens) == 1 else 'ies'}" if sens else "—"),
            ("Run time", f"{res.wall_seconds:.1f} s on {res.device} ({res.precision})")]
    _out(kv(rows[:-1] + _physics_rows(res, 2) + rows[-1:]))
    _print_physics_tables(res, 2)
    if len(res.per_body) > 1:
        _out()
        _out(table(["body", "Cd", "Cl", "Cm"],
                   [(k, units.sig(v.get("cd", math.nan), 4), units.sig(v.get("cl", math.nan), 4),
                     units.sig(v.get("cm", math.nan), 4)) for k, v in res.per_body.items()]))
    if verbose:
        for n in res.notes:
            _out(f"  · {n}")
    for w in res.warnings:
        _out(f"! {w}")


def _run_with_progress(fn, *a, **kw):
    pp = ProgressPrinter()
    try:
        return fn(*a, progress=pp, **kw)
    finally:
        pp.close()


def _report(project, sim_ids: list[str], path: Path) -> Path:
    from . import report
    fmt = "html" if path.suffix.lower() in (".html", ".htm") else "pdf"
    return report.generate(project, sim_ids, path, fmt=fmt)


def _fluid_at(args: argparse.Namespace):
    """Standard sea-level air (15 °C), or air at ``--ambient`` when it is given."""
    from .core import fluid as F
    t = getattr(args, "ambient", None)
    return F.air_isa(0.0) if t is None else F.air(float(t))


def _apply_physics_args(args: argparse.Namespace, design, sim, dim: int) -> list[str]:
    """Put ``--heated``, ``--fan`` and ``--closed`` into the design and the simulation (DESIGN.md 10).

    Returns one short phrase per feature for the header line. Raises ValueError for options that
    do not fit the case (a 3D fan position on a 2D design, a box of the wrong dimension).
    """
    from .model import Fan, ThermalSettings
    said: list[str] = []
    t_amb = float(args.ambient) if getattr(args, "ambient", None) is not None else 15.0
    heated = getattr(args, "heated", None)
    if heated is not None:
        objs = design.meshes if dim == 3 else design.bodies
        if not objs:
            raise ValueError("--heated needs a body to heat.")
        for b in objs:
            b.thermal, b.thermal_value = "temperature", float(heated)
        sim.thermal = ThermalSettings(enabled=True, t_ambient=t_amb, buoyancy=not args.no_buoyancy)
        said.append(f"bodies at {float(heated):.4g} °C in {t_amb:.4g} °C air"
                    + ("" if args.no_buoyancy else ", buoyancy on"))
    elif getattr(args, "no_buoyancy", False) or getattr(args, "ambient", None) is not None:
        said.append(f"{t_amb:.4g} °C air (no heated bodies: heat transfer stays off)")
    fans = []
    for i, f in enumerate(getattr(args, "fan", None) or []):
        if len(f["position"]) != dim:
            raise ValueError(f"--fan {i + 1}: a {dim}D case needs {dim} coordinates (X,Y{',Z' if dim == 3 else ''}), "
                             f"got {len(f['position'])}.")
        fans.append(Fan(name=f"Fan {i + 1}", position=f["position"], direction=f["direction"], diameter=f["diameter"],
                        mode=f["mode"], value=f["value"], frame="wind"))
    if fans:
        sim.fans = fans
        unit = {"thrust": "N" if dim == 3 else "N/m", "pressure_jump": "Pa", "velocity": "m/s"}
        said.append(", ".join(f"{f.name} {units.sig(f.value, 3)} {unit[f.mode]}"
                              + ("" if f.mode == "thrust" else f" ({f.mode.replace('_', ' ')})") for f in fans))
    box = getattr(args, "closed", None)
    if box is not None:
        if len(box) != 2 * dim:
            raise ValueError(f"--closed: a {dim}D box needs {2 * dim} numbers (X0,X1,Y0,Y1{',Z0,Z1' if dim == 3 else ''}).")
        sim.cfd.domain_mode = "closed"
        sim.cfd.domain_box = tuple(box)
        said.append("closed box " + " × ".join(units.sig(box[2 * i + 1] - box[2 * i], 3) for i in range(dim)) + " m")
    return said


def _diag_rows(diag: dict, result=None, *, dim: int = 2, sim=None, design=None) -> list[tuple[str, str]]:
    """The run's last diagnostics row, in words: continuity, the pressure solve, the heat balance.

    The heat-balance residual's share is taken of the heat passing *through* the domain (the
    larger of the gross heat in, the gross heat out and the run's heat rate), as the Results page
    does (``analysis.physics.heat_balance``, critic round 3.4). Against the largest single term, as
    before, a wall-heated cavity's net walls term of ~0 made a 4e-8 round-off residual read
    "118 %" (release gate 3, 2 November 2026).
    """
    rows: list[tuple[str, str]] = []
    if not diag:
        return rows
    if "div_max" in diag or "div_rms" in diag:
        rows.append(("Divergence (rms / max)", (f"{units.sig(diag.get('div_rms', math.nan), 2)} / "
                                                f"{units.sig(diag.get('div_max', math.nan), 2)} U/L")))
    res = diag.get("p_residual", diag.get("poisson_residual"))
    if res is not None:
        rows.append(("Pressure-solve residual", units.sig(res, 2)))
    if "cfl" in diag:
        rows.append(("CFL number (last step)", units.sig(diag["cfl"], 3)))
    resid = diag.get("heat_balance_residual")
    if resid is not None and math.isfinite(resid):
        from .analysis.physics import heat_balance
        if result is None:
            result = argparse.Namespace(diagnostics=diag, heat_rate=math.nan)
        hb = heat_balance(result, sim, design) or {}
        through = float(hb.get("throughput", math.nan))
        text = units.sig(resid, 2)
        if math.isfinite(through) and through > 0:
            rel = abs(resid) / through
            # a round-off share reads as a fraction ("5.2e-8 of"), not as "5.2e-6 %" (as on the Results page)
            part = f"{units.sig(100 * rel, 2)} %" if rel >= 1e-4 else units.sig(rel, 2)
            text += f" ({part} of the {units.sig(through, 4)} {'W' if dim == 3 else 'W/m'} through the domain)"
        rows.append(("Heat balance residual", text))
    return rows


def _physics_rows(res, dim: int) -> list[tuple[str, str]]:
    """Result rows of the placed physics (DESIGN.md 10): heat transfer and the solver diagnostics."""
    rows: list[tuple[str, str]] = []
    nu = getattr(res, "nusselt", None)
    if nu is not None and math.isfinite(nu.mean):
        rows.append(("Nusselt number", _stat_text(nu) + _rel_text(nu)))
        rows.append(("Heat rate", f"{units.sig(res.heat_rate, 4)} {'W' if dim == 3 else 'W per metre of span'}"))
    return rows + _diag_rows(getattr(res, "diagnostics", {}) or {}, res, dim=dim)


def _print_physics_tables(res, dim: int) -> None:
    """Per-body heat transfer (several heated bodies) and the fans' delivered thrust and flow."""
    heated = {k: v for k, v in (res.per_body or {}).items() if math.isfinite(float(v.get("nu", math.nan)))}
    if len(heated) > 1:
        _out()
        _out(table(["body", "Nu", "heat rate", "wall °C"],
                   [(k, units.sig(v["nu"], 4), units.sig(v.get("heat_rate", math.nan), 4),
                     units.sig(v.get("t_wall", math.nan), 4)) for k, v in heated.items()]))
    fans = getattr(res, "fans_report", None) or []
    if fans:
        tu, qu = ("N", "m³/s") if dim == 3 else ("N/m", "m²/s")
        _out()
        _out(table(["fan", f"thrust {tu}", "Δp Pa", f"flow {qu}", "disc m/s", "induced m/s"],
                   [(f.get("name", ""), units.sig(f.get("thrust", math.nan), 4),
                     units.sig(f.get("pressure_jump", math.nan), 3), units.sig(f.get("volume_flow", math.nan), 3),
                     units.sig(f.get("disc_velocity", math.nan), 3), units.sig(f.get("induced_velocity", math.nan), 3))
                    for f in fans]))


def cmd_quick(args: argparse.Namespace) -> int:
    from .model import CFDSettings, FlowConditions, GroundConfig, Simulation
    from .simulation.plan import default_cfd_settings, plan_simulation
    from .simulation.runner import run_cfd
    if args.template:
        from .geometry import templates
        params = dict(args.param or [])
        try:
            design = templates.make(args.template, **params)
        except KeyError as exc:
            _err(str(exc).strip("'\""))
            return 2
    else:
        design = _airfoil_design(args)
    fluid = _fluid_at(args)
    on_road = design.kind == "vehicle" or bool((design.source or {}).get("ground"))
    ground = GroundConfig(enabled=on_road and not args.no_ground, moving=not args.fixed_ground,
                          rotate_wheels=any(b.role == "wheel" for b in design.bodies))
    if args.cells_per_ref is not None and args.cells_per_ref < 4:
        _err("--cells-per-ref must be at least 4")
        return 2
    conditions = FlowConditions(fluid, args.speed, args.alpha)
    re_mode = args.re_mode
    if re_mode == "auto":
        # an explicit cap means "capped"; otherwise the same choice a new simulation
        # gets in the app (capped for airfoils above Re 1e5, see default_cfd_settings)
        re_mode = "capped" if args.re_cap else default_cfd_settings(design, conditions).reynolds_mode
    re_cap = args.re_cap
    if re_mode == "capped" and not re_cap:
        # the same fixed cap a new simulation gets in the app: a cap left to the planner
        # follows the grid, so --quality would change the flow, not just the resolution
        from .simulation.defaults import default_reynolds_cap
        re_cap = default_reynolds_cap(design)
    cfd = CFDSettings(quality=args.quality, device=args.device, reynolds_mode=re_mode,
                      reynolds_cap=re_cap, duration=args.duration, cells_per_ref=args.cells_per_ref,
                      compute_sensitivity=bool(args.sensitivity), sensitivity_objective=args.sensitivity or "cd")
    sim = Simulation(name=f"{design.name} quick", design_id=design.id,
                     conditions=conditions, ground=ground, cfd=cfd)
    try:
        physics = _apply_physics_args(args, design, sim, 2)
        plan = plan_simulation(design, sim)
    except ValueError as exc:
        _err(f"Cannot plan this simulation: {exc}")
        return 1
    _out(f"{design.name} at {units.fmt(args.speed, 'speed')}, alpha {args.alpha:g}°, "
         f"{args.quality} quality" + "".join(f"; {p}" for p in physics))
    _print_plan(plan, verbose=args.verbose)
    _out()
    if args.plan_only:
        return 0
    from .project import Project
    proj = None
    if args.save:
        # create the project first: a bad path should fail now, not after the run
        proj = Project.create(Path(args.save), design.name)
        proj.add_design(design)
        sim.status = "running"
        proj.add_simulation(sim)
        proj.save()
    with StopOnInterrupt() as stop:
        res = _run_with_progress(run_cfd, plan, cancel=stop)
    cancelled = stop.is_set()
    _out()
    _print_cfd_result(res, design, verbose=args.verbose)
    tmp = None
    try:
        if proj is not None or args.report:
            if proj is None:
                tmp = Path(tempfile.mkdtemp(prefix="aerolume-quick-"))
                proj = Project.create(tmp / "quick", design.name)
                proj.add_design(design)
                proj.add_simulation(sim)
            from .model import now_iso
            proj.save_result(sim.id, res)
            sim.status, sim.finished = ("cancelled" if cancelled else "done"), now_iso()
            proj.update_simulation(sim)
            proj.save()
            if args.save:
                _out(f"Saved project {proj.path}")
            if args.report:
                path = _report(proj, [sim.id], Path(args.report))
                _out(f"Wrote report {path}")
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    if cancelled:
        _err("Stopped by the user: the values above come from a partial run.")
        return 1
    return 0


def _find_simulation(project, key: str):
    sims = project.simulations
    if key in sims:
        return sims[key]
    hits = [s for s in sims.values() if s.name.strip().lower() == key.strip().lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise LookupError(f"several simulations are named {key!r}; use the id: "
                          + ", ".join(f"{s.id}" for s in hits))
    starts = [s for s in sims.values() if s.id.startswith(key)]
    if len(starts) == 1:
        return starts[0]
    listing = "\n".join(f"  {s.id}  {s.name}  ({s.analysis}, {s.status})" for s in sims.values()) or "  (none)"
    raise LookupError(f"no simulation {key!r} in this project. Available:\n{listing}")


def cmd_run(args: argparse.Namespace) -> int:
    from .model import now_iso
    from .project import Project
    try:
        project = Project.open(args.project)
        sim = _find_simulation(project, args.simulation)
    except Exception as exc:  # noqa: BLE001 - user-facing
        _err(str(exc))
        return 1
    design = project.designs.get(sim.design_id)
    if design is None:
        _err(f"The design of simulation {sim.name!r} is missing from the project.")
        return 1
    _out(f"Project {project.name}: running {sim.name!r} ({sim.analysis}) on design {design.name!r}")
    sim.status, sim.error = "running", None
    project.update_simulation(sim)
    project.save()
    with StopOnInterrupt() as stop:
        try:
            if sim.analysis == "polar":
                from .simulation.runner import run_polar
                res = _run_with_progress(run_polar, design, sim, cancel=stop)
            elif sim.analysis == "wing":
                from .simulation.wing import run_wing
                res = _run_with_progress(run_wing, design, sim, cancel=stop, project=project)
            else:
                from .simulation.plan import plan_simulation
                from .simulation.runner import run_cfd
                from .simulation.wing import airfoil_resolver
                plan = plan_simulation(design, sim, device=args.device, airfoil_coords=airfoil_resolver(project))
                _print_plan(plan, verbose=args.verbose)
                res = _run_with_progress(run_cfd, plan, cancel=stop)
        except KeyboardInterrupt:              # the second Ctrl+C: abort without a result
            sim.status = "cancelled"
            project.update_simulation(sim)
            project.save()
            _err("Aborted; no result was saved.")
            return 1
        except Exception as exc:  # noqa: BLE001 - recorded on the simulation, then reported
            sim.status, sim.error = "failed", str(exc)
            project.update_simulation(sim)
            project.save()
            _err(f"Failed: {exc}")
            return 1
    cancelled = stop.is_set()
    project.save_result(sim.id, res)
    sim.status, sim.finished = ("cancelled" if cancelled else "done"), now_iso()
    project.update_simulation(sim)
    project.save()
    _out()
    if sim.analysis == "polar":
        rows = [(f"{p.alpha:g}", units.sig(p.cl, 4), units.sig(p.cd, 3), units.sig(p.cm, 3),
                 "yes" if p.converged else "NO") for p in res.points]
        _out(table(["alpha°", "Cl", "Cd", "Cm c/4", "conv."], rows))
        for w in res.warnings:
            _out(f"! {w}")
    elif sim.analysis == "wing":
        _print_wing_result(res, verbose=args.verbose)
    elif getattr(design, "dimension", 2) == 3:
        _print_cfd3d_result(res, verbose=args.verbose)
    else:
        _print_cfd_result(res, design, verbose=args.verbose)
    _out(f"Result saved in {project.sim_dir(sim.id)}" + (" (partial: stopped by the user)" if cancelled else ""))
    return 1 if cancelled else 0


# ---------------------------------------------------------------------------
# 3D CFD (quick3d) and wings
# ---------------------------------------------------------------------------

def _print_cfd3d_result(res, *, verbose: bool = False) -> None:
    """The six coefficients with error bars, dimensional forces and the run's validity context."""
    sens = [s for s in res.surface if getattr(s, "sensitivity", None) is not None]
    mx, my, mz = res.moments
    rows = [("CD (drag)", _stat_text(res.cd) + _rel_text(res.cd)),
            ("CL (lift)", _stat_text(res.cl) + _rel_text(res.cl) + (" (downforce)" if res.cl.mean < 0 else "")),
            ("CS (side force)", _stat_text(res.cs)),
            ("Cm (pitch, + nose up)", _stat_text(res.cm)),
            ("C_roll / Cn (yaw)", f"{_stat_text(res.c_roll)} / {_stat_text(res.cn)}"),
            ("L/D", units.sig(res.lift_to_drag, 3)),
            ("Drag / lift / side force", (f"{units.sig(res.drag, 4)} N / {units.sig(res.lift, 4)} N / "
                                          f"{units.sig(res.side, 4)} N")),
            ("Moments (roll, pitch, yaw)", f"{units.sig(mx, 3)} / {units.sig(my, 3)} / {units.sig(mz, 3)} N m"),
            ("Drag area CD·A", f"{units.sig(res.drag_area, 4)} m²"),
            ("Power to overcome drag", units.fmt(res.power, "power")),
            ("Strouhal number", units.sig(res.strouhal, 3) if res.strouhal else "— (no periodic shedding)"),
            ("Reynolds (real / simulated)", f"{units.fmt_reynolds(res.re_real)} / {units.fmt_reynolds(res.re_sim)}"),
            ("Grid", f"{' × '.join(f'{n:,}' for n in res.grid)} = {res.cells:,} cells"
                     + (" (half domain, symmetry plane)" if res.symmetry_plane else "")),
            ("Averaged over", f"t = {units.sig(res.t_avg_start, 3)} … {units.sig(res.t_end, 3)} L/U"),
            ("Converged", "yes" if res.converged else "NO (see below)"),
            ("Sensitivity map", f"{len(sens)} part{'' if len(sens) == 1 else 's'}" if sens else "—"),
            ("Run time", f"{res.wall_seconds:.1f} s on {res.device} ({res.precision})")]
    _out(kv(rows[:-1] + _physics_rows(res, 3) + rows[-1:]))
    _print_physics_tables(res, 3)
    if len(res.per_body) > 1:
        _out()
        _out(table(["part", "CD", "CL", "CS", "Cm"],
                   [(k, units.sig(v.get("cd", math.nan), 4), units.sig(v.get("cl", math.nan), 4),
                     units.sig(v.get("cs", math.nan), 3), units.sig(v.get("cm", math.nan), 3))
                    for k, v in res.per_body.items()]))
    if verbose:
        for n in res.notes:
            _out(f"  · {n}")
    for w in res.warnings:
        _out(f"! {w}")


def _design3d(args: argparse.Namespace):
    """The 3D design of ``quick3d``: an imported mesh file (with the provenance a script's ``import_mesh``
    records: name, format, size and SHA-256) or a sphere."""
    from .api import file_source
    from .geometry3d import design_from_meshes, read_mesh, sphere
    if args.sphere is not None:
        d = float(args.sphere)
        return design_from_meshes([sphere(d, 64, name="Sphere")], f"Sphere {d:g} m", kind="generic",
                                  source={"type": "sphere", "diameter": d})
    path = Path(args.stl)
    notes: list[str] = []
    meshes = read_mesh(path, notes=notes)
    units_arg = None if args.units in (None, "m") else args.units
    kind = args.kind or "vehicle"
    return design_from_meshes(meshes, path.stem, kind=kind, length=args.length, units=units_arg,
                              clearance=args.clearance, notes=notes, source=file_source(path))


def cmd_quick3d(args: argparse.Namespace) -> int:
    from .model import CFDSettings, FlowConditions, GroundConfig, Simulation
    from .simulation.plan import plan_simulation
    from .simulation.runner import run_cfd
    if args.cells_per_ref is not None and args.cells_per_ref < 4:
        _err("--cells-per-ref must be at least 4")
        return 2
    try:
        design = _design3d(args)
    except (OSError, ValueError) as exc:
        _err(f"Cannot read the geometry: {exc}")
        return 1
    fluid = _fluid_at(args)
    on_road = bool((design.source or {}).get("on_road"))
    ground = GroundConfig(enabled=on_road and not args.no_ground, moving=not args.fixed_ground,
                          rotate_wheels=any(m.role == "wheel" for m in design.meshes))
    conditions = FlowConditions(fluid, args.speed, args.alpha, args.yaw)
    re_mode = "capped" if (args.re_mode == "capped" or (args.re_mode == "auto" and args.re_cap)) else "real"
    cfd = CFDSettings(quality=args.quality, device=args.device, reynolds_mode=re_mode, reynolds_cap=args.re_cap,
                      duration=args.duration, cells_per_ref=args.cells_per_ref,
                      symmetry="full" if args.full_domain else "auto",
                      compute_sensitivity=bool(args.sensitivity), sensitivity_objective=args.sensitivity or "cd")
    sim = Simulation(name=f"{design.name} 3D", design_id=design.id, conditions=conditions, ground=ground, cfd=cfd)
    try:
        physics = _apply_physics_args(args, design, sim, 3)
        plan = plan_simulation(design, sim)
    except ValueError as exc:
        _err(f"Cannot plan this simulation: {exc}")
        return 1
    L = design.length_ref
    n_parts = len(design.meshes)
    _out(f"{design.name}: {n_parts} part{'' if n_parts == 1 else 's'}, "
         f"length {units.sig(design.bounds3()[1] - design.bounds3()[0], 4)} m, "
         f"frontal area {units.sig(design.ref_area, 4)} m², reference length {units.sig(L, 4)} m"
         + (", mirror-symmetric" if design.symmetric else ""))
    _out(f"At {units.fmt(args.speed, 'speed')}, alpha {args.alpha:g}°, yaw {args.yaw:g}°, {args.quality} quality"
         + "".join(f"; {p}" for p in physics))
    if args.verbose:
        for n in (design.source or {}).get("import_notes", []):
            _out(f"  · {n}")
    _print_plan(plan, verbose=args.verbose)
    _out()
    if args.plan_only:
        return 0
    from .project import Project
    proj = None
    if args.save:
        proj = Project.create(Path(args.save), design.name)
        proj.add_design(design)
        sim.status = "running"
        proj.add_simulation(sim)
        proj.save()
    with StopOnInterrupt() as stop:
        res = _run_with_progress(run_cfd, plan, cancel=stop)
    cancelled = stop.is_set()
    _out()
    _print_cfd3d_result(res, verbose=args.verbose)
    if proj is not None:
        from .model import now_iso
        proj.save_result(sim.id, res)
        sim.status, sim.finished = ("cancelled" if cancelled else "done"), now_iso()
        proj.update_simulation(sim)
        proj.save()
        _out(f"Saved project {proj.path}")
    if cancelled:
        _err("Stopped by the user: the values above come from a partial run.")
        return 1
    return 0


def cmd_cavity(args: argparse.Namespace) -> int:
    """The differentially heated cavity through the normal planner and runner (DESIGN.md 10.2).

    The case (box, wall temperatures, the ΔT that gives the Rayleigh number, the reference
    Nusselt number) is the scripting API's :func:`aerolume.api.cavity_case`, so the command,
    a script and the app run the same set-up and compare with the same table.
    """
    from .api import cavity_case
    from .model import CFDSettings, FlowConditions, Simulation
    from .simulation.plan import plan_simulation
    from .simulation.runner import run_cfd
    if args.cells_per_ref is not None and args.cells_per_ref < 8:
        _err("--cells-per-ref must be at least 8 (cells across the box)")
        return 2
    dim = 3 if args.three_d else 2
    try:
        case = cavity_case(ra=args.ra, pr=args.pr, size=args.size, dimension=dim)
    except ValueError as exc:
        _err(f"Cannot set up the cavity: {exc}")
        return 2
    design, st = case.design, case.settings
    th = st["thermal"]
    d_t = float((design.source or {}).get("delta_t", math.nan))
    u_ff = math.sqrt(th.gravity * th.beta * d_t * args.size) if d_t > 0 else 1.0
    cfd = CFDSettings(quality=args.quality, device=args.device, duration=args.duration,
                      cells_per_ref=args.cells_per_ref, domain_mode="closed", domain_box=tuple(st["domain_box"]))
    sim = Simulation(name=case.name, design_id=design.id, conditions=FlowConditions(st["fluid"], u_ff, 0.0), cfd=cfd,
                     thermal=th)
    try:
        plan = plan_simulation(design, sim)
    except ValueError as exc:
        _err(f"Cannot plan this simulation: {exc}")
        return 1
    ref = case.reference
    _out(f"{case.name} ({'cubic' if dim == 3 else 'square'} cavity of {units.sig(args.size, 3)} m, Pr "
         f"{units.sig(args.pr, 3)}, {args.quality} quality): left wall {units.sig(d_t, 3)} K warmer than the right")
    _out("Reference: " + (f"Nu = {ref['value']:g} ({ref['source']})" if ref else
                          "none published at these parameters (de Vahl Davis: 2D, Ra 1e3-1e6; Tric et al.: 3D, "
                          "Ra 1e4-1e6; both Pr 0.71)"))
    if args.verbose:
        for n in case.notes:
            _out(f"  · {n}")
    _print_plan(plan, verbose=args.verbose)
    _out()
    if args.plan_only:
        return 0
    from .project import Project
    proj = None
    if args.save:
        proj = Project.create(Path(args.save), design.name)
        proj.add_design(design)
        sim.status = "running"
        proj.add_simulation(sim)
        proj.save()
    with StopOnInterrupt() as stop:
        res = _run_with_progress(run_cfd, plan, cancel=stop)
    cancelled = stop.is_set()
    _out()
    nu = res.nusselt
    rows = [("Nusselt number (hot wall)", _stat_text(nu) if math.isfinite(nu.mean) else "—")]
    if ref and math.isfinite(nu.mean):
        rows.append((f"Reference ({ref['source'].split(',')[0]})", f"{ref['value']:g}"))
        rows.append(("Difference", f"{100 * (nu.mean - ref['value']) / ref['value']:+.2f} %"))
    rows += [("Heat rate", f"{units.sig(res.heat_rate, 4)} {'W' if dim == 3 else 'W per metre of span'}"),
             ("Grid", " × ".join(f"{n:,}" for n in res.grid) + f" ({res.cells_per_ref:.0f} cells across)"),
             ("Simulated", f"t = {units.sig(res.t_end, 3)} free-fall times (averaged from {units.sig(res.t_avg_start, 3)})"),
             ("Converged", "yes" if res.converged else "NO (see below)"),
             ("Run time", f"{res.wall_seconds:.1f} s on {res.device} ({res.precision})")]
    rows += _diag_rows(res.diagnostics or {}, res, dim=dim, sim=sim, design=design)
    _out(kv(rows))
    if args.verbose:
        for n in res.notes:
            _out(f"  · {n}")
    for w in res.warnings:
        _out(f"! {w}")
    if proj is not None:
        from .model import now_iso
        proj.save_result(sim.id, res)
        sim.status, sim.finished = ("cancelled" if cancelled else "done"), now_iso()
        proj.update_simulation(sim)
        proj.save()
        _out(f"Saved project {proj.path}")
    if cancelled:
        _err("Stopped by the user: the values above come from a partial run.")
        return 1
    return 0


def _wing_design(args: argparse.Namespace):
    """A straight-tapered wing from the command line (root at y = 0, mirrored).

    ``--sweep`` is the quarter-chord sweep (the aerodynamic one), ``--dihedral``
    the leading-edge dihedral, ``--twist`` the tip twist (negative = washout).
    """
    from .geometry3d import design_from_wing, naca_code
    from .model import WingSection, WingSpec
    code = naca_code(args.naca)
    if code is None:
        raise ValueError(f"{args.naca!r} is not a NACA 4- or 5-digit code")
    semi, c_root = 0.5 * float(args.span), float(args.chord)
    c_tip = c_root * float(args.taper)
    x_tip = semi * math.tan(math.radians(args.sweep)) + 0.25 * (c_root - c_tip)
    z_tip = semi * math.tan(math.radians(args.dihedral))
    wing = WingSpec([WingSection(0.0, c_root, airfoil=code),
                     WingSection(semi, c_tip, x_le=x_tip, z_le=z_tip, twist_deg=float(args.twist), airfoil=code)],
                    symmetric=True)
    name = f"NACA {code} wing, span {units.sig(args.span, 3)} m"
    return design_from_wing(wing, name, source={"type": "cli", "naca": code})


def _print_wing_result(res, *, verbose: bool = False) -> None:
    import numpy as np
    rows = []
    for p in res.points:
        ld = p.cl / p.cd if p.converged and p.cd > 0 else float("nan")
        rows.append((f"{p.alpha:g}", units.sig(p.cl, 4), units.sig(p.cd, 3), units.sig(p.cdi, 3), units.sig(p.cdv, 3),
                     units.sig(p.cm, 3), units.sig(ld, 3), units.sig(p.e, 3), "yes" if p.converged else "NO"))
    _out(table(["alpha°", "CL", "CD", "CDi", "CDv", "Cm", "L/D", "e", "conv."], rows))
    tab = res.table()
    _out()
    summary = [("Planform", (f"area {units.sig(res.area, 4)} m², span {units.sig(res.span, 4)} m, "
                             f"AR {units.sig(res.aspect_ratio, 3)}, MAC {units.sig(res.mac, 4)} m")),
               ("Reynolds number (MAC)", units.fmt_reynolds(res.re_mac)),
               ("Method", ("vortex lattice" if res.method == "vlm" else "nonlinear lifting line")
                + (", viscous strips" if res.viscous else ", inviscid"))]
    a, cl = tab["alpha"], tab["cl"]
    lin = np.abs(a) <= 5.0
    if lin.sum() >= 2:
        slope = float(np.polyfit(a[lin], cl[lin], 1)[0])
        per_rad = units.sig(slope * 180 / math.pi, 4)
        summary.append(("Lift slope CL_α", f"{units.sig(slope, 4)} per degree ({per_rad} per radian)"))
    ok = tab["cd"] > 0
    if ok.any():
        ld = np.where(ok, tab["cl"] / np.where(ok, tab["cd"], 1.0), -np.inf)
        k = int(np.argmax(ld))
        summary.append(("Best L/D", f"{units.sig(ld[k], 4)} at {units.sig(a[k], 3)}° (CL {units.sig(cl[k], 3)})"))
    summary.append(("Run time", f"{res.wall_seconds:.1f} s"))
    _out(kv(summary, indent=0))
    for n in res.notes if verbose else [n for n in res.notes if n.startswith("Stall onset")]:
        _out(f"  · {n}")
    for w in res.warnings:
        _out(f"! {w}")


def cmd_wing(args: argparse.Namespace) -> int:
    from .core import fluid as F
    from .model import FlowConditions, Simulation, WingSettings
    from .simulation.wing import run_wing
    for name in ("span", "chord", "taper"):
        if not getattr(args, name) > 0:
            _err(f"--{name} must be positive")
            return 2
    try:
        design = _wing_design(args)
    except ValueError as exc:
        _err(f"Cannot build the wing: {exc}")
        return 2
    a0, a1, da = args.alpha
    sim = Simulation(name="wing polar", design_id=design.id, analysis="wing",
                     conditions=FlowConditions(F.air_isa(0.0), args.speed, 0.0, args.yaw),
                     wing=WingSettings(alpha_start=a0, alpha_end=a1, alpha_step=da, method=args.method,
                                       viscous=not args.inviscid, ncrit=args.ncrit, mach=args.mach))
    _out(f"{design.name}: chord {units.sig(args.chord, 3)} m, taper {units.sig(args.taper, 3)}, quarter-chord sweep "
         f"{args.sweep:g}°, {units.fmt(args.speed, 'speed')}, alpha {a0:g} to {a1:g} step {da:g}")
    proj = None
    if args.save:
        from .project import Project
        proj = Project.create(Path(args.save), design.name)
        proj.add_design(design)
        sim.status = "running"
        proj.add_simulation(sim)
        proj.save()
    with StopOnInterrupt() as stop:
        res = _run_with_progress(run_wing, design, sim, cancel=stop)
    _out()
    _print_wing_result(res, verbose=args.verbose)
    if args.csv:
        from .export import wing_csv
        _out(f"Wrote {wing_csv(res, args.csv)}")
    if proj is not None:
        from .model import now_iso
        proj.save_result(sim.id, res)
        sim.status, sim.finished = ("cancelled" if stop.is_set() else "done"), now_iso()
        proj.update_simulation(sim)
        proj.save()
        _out(f"Saved project {proj.path}")
    if stop.is_set():
        _err("Stopped by the user: only the angles above were computed.")
        return 1
    return 0 if any(p.converged for p in res.points) else 1


# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------

def cmd_script(args: argparse.Namespace) -> int:
    """Run a Python script with the scripting API bound, as the app's Script console does.

    Names in the script: ``al`` (the ``aerolume`` package), ``project`` (an
    ``aerolume.api.Project`` for ``--project``, else None), ``args`` (what
    followed ``--``; also ``sys.argv[1:]``), ``log(text, level)`` and
    ``show(figure)`` (saves ``<script>-figure-<n>.png`` in ``--figures``). The
    first Ctrl+C stops the running simulation, which keeps its partial result
    in the project, and ends the script; a second one aborts. An error prints
    the script's own lines of the traceback and returns 1.
    """
    import builtins

    import aerolume

    from . import api
    path = Path(args.file)
    if not path.is_file():
        _err(f"No script at {path}.")
        return 2
    try:
        source = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        _err(f"Cannot read {path}: {exc}")
        return 2
    filename = str(path.resolve())
    project = None
    if args.project:
        try:
            project = api.open_project(args.project, create=args.create)
        except api.AerolumeError as exc:
            _err(str(exc) + ("" if args.create else " (--create makes a new project there)"))
            return 1
    figures_dir = Path(args.figures) if args.figures else Path.cwd()
    count = 0

    def show(obj):
        nonlocal count
        if api._is_figure(obj):
            from .viz.figures import save_figure
            count += 1
            figures_dir.mkdir(parents=True, exist_ok=True)
            out = save_figure(obj, figures_dir / f"{path.stem}-figure-{count}.png", dpi=150)
            _out(f"Figure saved to {out}")
        else:
            _out(str(obj))
        return obj

    def log(text, level: str = "info") -> None:
        if level in ("warning", "error"):
            _err(f"{level.capitalize()}: {text}")
        else:
            _out(str(text))

    namespace = {"__name__": "__main__", "__file__": filename, "__builtins__": builtins, "al": aerolume,
                 "project": project, "args": list(args.script_args), "log": log, "show": show}
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as exc:
        _err(api.script_traceback(exc, filename)[2].rstrip())
        return 1
    old_argv, old_path = list(sys.argv), list(sys.path)
    sys.argv = [str(path), *args.script_args]
    sys.path.insert(0, str(path.resolve().parent))      # sibling modules import as with `python FILE`
    t0 = time.perf_counter()
    try:
        with StopOnInterrupt() as stop, api.context(cancel=stop, show=show, project=project,
                                                    verbose=True if args.verbose else None):
            try:
                exec(code, namespace)                   # noqa: S102 - running the user's script is the point
            except SystemExit as exc:
                if exc.code in (None, 0):
                    return 0
                if isinstance(exc.code, int):
                    return exc.code
                _err(str(exc.code))
                return 1
            except api.Cancelled as exc:
                _err(f"Stopped: {exc}")
                return 1
            except KeyboardInterrupt:
                _err("Aborted.")
                return 1
            except Exception as exc:  # noqa: BLE001 - the script's error, shown readably
                _err(api.script_traceback(exc, filename)[2].rstrip())
                return 1
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
    if args.verbose:
        _err(f"Script finished in {time.perf_counter() - t0:.1f} s.")
    return 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _find_validation_dir() -> Path | None:
    here = Path(__file__).resolve()
    for cand in (here.parents[2] / "validation", Path.cwd() / "validation"):
        if (cand / "ns2d_cases.py").is_file():
            return cand
    return None


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # dataclasses need the module registered
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _cavity_nusselt(ra: float, device: str) -> float:
    """Mean Nusselt number of the Draft square cavity at ``ra``, planned and run as the app does."""
    from .api import cavity_case
    from .model import CFDSettings, FlowConditions, Simulation
    from .simulation.plan import plan_simulation
    from .simulation.runner import run_cfd
    case = cavity_case(ra=ra)
    st = case.settings
    cfd = CFDSettings(quality="draft", device=device, domain_mode="closed", domain_box=tuple(st["domain_box"]))
    sim = Simulation(name=case.name, design_id=case.design.id, conditions=FlowConditions(st["fluid"], 1.0, 0.0),
                     cfd=cfd, thermal=st["thermal"])
    return float(run_cfd(plan_simulation(case.design, sim, device=device)).nusselt.mean)


def cmd_validate(args: argparse.Namespace) -> int:
    vdir = _find_validation_dir()
    if vdir is None:
        _err("validation/ns2d_cases.py was not found (it ships with the source repository, not the wheel). "
             "Run this command from the repository root.")
        return 1
    cases = _load_module(vdir / "ns2d_cases.py", "aerolume_ns2d_cases")
    from .core import device as devmod
    from .simulation.live import cpu_threads
    dev = devmod.resolve(args.device)
    rows: list[tuple] = []
    ok_all = True

    def check(name: str, value: float, lo: float, hi: float, fmt: str = "{:.4f}", secs: float | None = None) -> None:
        nonlocal ok_all
        ok = math.isfinite(value) and lo <= value <= hi
        ok_all &= ok
        rows.append((name, fmt.format(value), f"{fmt.format(lo)} … {fmt.format(hi)}",
                     "" if secs is None else f"{secs:.0f}", "PASS" if ok else "FAIL"))

    _out(f"Validation subset ({'full' if args.full else 'quick'}) on {devmod.info(dev).name}")
    # 1. verification: Taylor-Green spatial order (exact solution), CPU float64
    t0 = time.perf_counter()
    e16, _ = cases.taylor_green_error(16)
    e32, rate = cases.taylor_green_error(32)
    check("Taylor-Green spatial order (16→32)", math.log2(e16 / e32), 1.8, 2.2, "{:.2f}", time.perf_counter() - t0)
    check("Taylor-Green decay rate / exact (N=32)", rate, 0.98, 1.02, "{:.4f}")
    with cpu_threads(dev):
        # 2. cylinder Re 40, steady (16 cells/D: the doc's coarse row, Cd 1.63, L/D 2.32)
        t0 = time.perf_counter()
        r = cases.run_cylinder(40.0, 16, t_end=40.0, t_avg=30.0, device=dev)
        secs = time.perf_counter() - t0
        check("Cylinder Re 40: Cd (16 cells/D)", r.values["cd"], 1.55, 1.70, secs=secs)
        check("Cylinder Re 40: wake length L/D", r.values["L/D"], 2.0, 2.5, "{:.3f}")
        if args.full:
            t0 = time.perf_counter()
            r = cases.run_cylinder(100.0, 24, t_end=120.0, t_avg=80.0, device=dev)
            secs = time.perf_counter() - t0
            check("Cylinder Re 100: Cd (24 cells/D)", r.values["cd"], 1.33, 1.47, secs=secs)
            check("Cylinder Re 100: lift amplitude", r.values["cl_amp"], 0.30, 0.40, "{:.3f}")
            check("Cylinder Re 100: Strouhal number", r.values["st"], 0.160, 0.175, "{:.4f}")
    if args.full:
        # placed physics through the whole chain (planner -> solver -> SI): the Draft heated cavity.
        # The solver measures the hot wall's mean Nu, so the reference is de Vahl Davis' Nu0 (2.238,
        # api.CAVITY_NU_2D), not the paper's cavity average 2.243; the Draft run reads 2.251 on the
        # GPU, +0.58 % (docs/user_guide_simulation.md 13.9). Band: +-2 % of Nu0.
        from .api import CAVITY_NU_2D
        nu0 = float(CAVITY_NU_2D[1e4])
        t0 = time.perf_counter()
        nu = _cavity_nusselt(1e4, dev)
        check(f"Heated cavity Ra 1e4: hot-wall mean Nu (de Vahl Davis Nu0 {nu0:.3f})", nu, 0.98 * nu0, 1.02 * nu0,
              "{:.3f}", time.perf_counter() - t0)
    # 3. airfoil polar vs Ladson (NACA 0012, Re 6e6, M 0.15, tripped)
    af_path = vdir / "airfoil_validation.py"
    if af_path.is_file():
        af = _load_module(af_path, "aerolume_airfoil_validation")
        t0 = time.perf_counter()
        c = af.run("NACA 0012 Ladson", "0012", 6e6, [0.0, 2.0, 4.0, 6.0, 8.0], tripped=True, mach=0.15)
        secs = time.perf_counter() - t0
        ref = af.LADSON_0012_RE6E6
        for a_ref, cl_ref, cd_ref in ref[3:6]:
            cl = float(c.at("cl", a_ref))
            check(f"NACA 0012 Re 6e6: Cl at {a_ref:g}° (Ladson {cl_ref:.3f})", cl, cl_ref - 0.04, cl_ref + 0.04,
                  "{:.3f}", secs)
            secs = None
        cd = float(c.at("cd", 4.04))
        check("NACA 0012 Re 6e6: Cd at 4.04° (Ladson 0.00823)", cd, 0.0070, 0.0095, "{:.5f}")
    _out(table(["check", "value", "expected", "s", "result"], rows, "lrrrl"))
    _out()
    _out("All checks passed." if ok_all else "Some checks FAILED.")
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aerolume-cli", description="Aerolume headless tools: 2D and 3D CFD, finite "
                                "wings and airfoil polars from the command line.")
    p.add_argument("--version", action="version", version=f"Aerolume {__version__}")
    sub = p.add_subparsers(dest="command", metavar="command")
    sub.required = True

    d = sub.add_parser("doctor", help="show versions, devices and run a short solver check")
    d.add_argument("--no-solver", action="store_true", help="skip the solver check")
    d.set_defaults(fn=cmd_doctor)

    b = sub.add_parser("bench", help="measure solver throughput on this machine")
    b.add_argument("--device", default="auto", help="auto (all devices), cuda or cpu")
    b.add_argument("--quick", action="store_true", help="fewer grids, laminar only")
    b.set_defaults(fn=cmd_bench)

    def physics_args(sp: argparse.ArgumentParser, dim: int) -> None:
        g = sp.add_argument_group("placed physics (DESIGN.md 10)")
        xyz = "X,Y,Z" if dim == 3 else "X,Y"
        g.add_argument("--heated", type=_celsius, default=None, metavar="T",
                       help="hold every body at T °C and solve for the temperature (heat transfer on)")
        g.add_argument("--ambient", type=_celsius, default=None, metavar="T",
                       help="air temperature in °C (default 15, the standard sea-level air used otherwise)")
        g.add_argument("--no-buoyancy", action="store_true",
                       help="forced convection only: the temperature does not move the air")
        g.add_argument("--fan", type=parse_fan, action="append", metavar=f"{xyz}:D:VALUE[:MODE[:DIR]]",
                       help=f"an actuator-disc fan at {xyz} [m] of diameter D [m]: VALUE is the thrust "
                            f"({'N' if dim == 3 else 'N per metre of span'}; MODE thrust, the default), the pressure "
                            "jump (Pa; MODE pressure) or the speed through it (m/s; MODE velocity); DIR is its "
                            "direction (default +x). Repeatable")
        g.add_argument("--closed", type=parse_box, default=None,
                       metavar="X0,X1,Y0,Y1" + (",Z0,Z1" if dim == 3 else ""),
                       help="a closed box with walls on every side (a room) instead of a wind tunnel [m]")

    def airfoil_args(sp: argparse.ArgumentParser, required: bool) -> None:
        g = sp.add_mutually_exclusive_group(required=required)
        g.add_argument("--naca", help="NACA 4- or 5-digit code, e.g. 2412")
        g.add_argument("--dat", help="airfoil coordinate file (Selig or Lednicer .dat)")
        sp.add_argument("--chord", type=_positive_float, default=1.0, help="chord in metres (default 1)")

    pl = sub.add_parser("polar", help="viscous-inviscid polar of an airfoil (XFOIL-class)")
    airfoil_args(pl, True)
    pl.add_argument("--re", type=_positive_float, required=True, help="chord Reynolds number, e.g. 1e6")
    pl.add_argument("--alpha", type=parse_alpha_range, default=(-4.0, 14.0, 1.0),
                    help="angles start:end:step in degrees (default -4:14:1)")
    pl.add_argument("--ncrit", type=_positive_float, default=9.0, help="e^N transition criterion (default 9)")
    pl.add_argument("--xtr", type=float, default=1.0, help="forced transition x/c on both sides (1 = free)")
    pl.add_argument("--mach", type=float, default=0.0, help="Mach number for compressibility correction")
    pl.add_argument("--csv", help="also write the polar to this CSV file")
    pl.set_defaults(fn=cmd_polar)

    q = sub.add_parser("quick", help="plan and run one CFD simulation of a template or NACA section")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", help="geometry template (sedan, hatchback, suv, cylinder, ahmed, ...)")
    g.add_argument("--naca", help="NACA airfoil code, e.g. 0012")
    g.add_argument("--dat", help="airfoil coordinate file")
    q.add_argument("--chord", type=_positive_float, default=1.0, help="airfoil chord in metres (default 1)")
    q.add_argument("--param", type=_param, action="append", help="template parameter name=value (repeatable)")
    q.add_argument("--speed", type=_speed, default=30.0, help="flow speed, e.g. 30, '120 km/h' (default 30 m/s)")
    q.add_argument("--alpha", type=float, default=0.0, help="angle of attack / pitch in degrees")
    q.add_argument("--quality", choices=("draft", "standard", "fine"), default="draft")
    q.add_argument("--device", default="auto", help="auto, cuda or cpu")
    q.add_argument("--re-mode", choices=("auto", "real", "capped"), default="auto",
                   help="Reynolds-number treatment; auto = capped for airfoils above Re 1e5 (2D CFD cannot "
                        "resolve their boundary layers), real otherwise")
    q.add_argument("--re-cap", type=_positive_float, default=None, help="Reynolds cap for --re-mode capped")
    q.add_argument("--duration", type=_positive_float, default=None, help="run length in convective times L/U")
    q.add_argument("--cells-per-ref", type=int, default=None,
                   help="grid cells per reference length (overrides the quality preset)")
    q.add_argument("--no-ground", action="store_true", help="vehicles: no road (free air)")
    q.add_argument("--fixed-ground", action="store_true", help="vehicles: fixed floor instead of a moving belt")
    q.add_argument("--sensitivity", choices=("cd", "cl", "ld", "cm"), default=None,
                   help="also compute the shape-sensitivity map of this objective")
    q.add_argument("--report", help="write a PDF (or .html) report of the run")
    q.add_argument("--save", help="save the design, simulation and result as a project in this folder")
    q.add_argument("--plan-only", action="store_true", help="show the plan and stop")
    q.add_argument("-v", "--verbose", action="store_true", help="also print the planner's reasoning")
    physics_args(q, 2)
    q.set_defaults(fn=cmd_quick)

    q3 = sub.add_parser("quick3d", help="plan and run one 3D CFD simulation of a mesh file or a sphere")
    g3 = q3.add_mutually_exclusive_group(required=True)
    g3.add_argument("--stl", help="mesh file: STL (ASCII or binary), OBJ or PLY; holes are tolerated")
    g3.add_argument("--sphere", type=_positive_float, help="a sphere of this diameter in metres")
    q3.add_argument("--kind", choices=("vehicle", "generic"), default=None,
                    help="mesh files: vehicle (default: on the road, nose at x = 0) or generic (free flight)")
    q3.add_argument("--units", default="auto",
                    help="units of the mesh file: auto (guess from the size), m, cm, mm, in, ft (default auto)")
    q3.add_argument("--length", type=_positive_float, default=None, help="scale the shape to this length [m]")
    q3.add_argument("--clearance", type=float, default=None, help="vehicles: ground clearance [m] (default 0)")
    q3.add_argument("--speed", type=_speed, default=30.0, help="flow speed, e.g. 30, '120 km/h' (default 30 m/s)")
    q3.add_argument("--alpha", type=float, default=0.0, help="pitch / angle of attack in degrees")
    q3.add_argument("--yaw", type=float, default=0.0, help="yaw (sideslip) in degrees, + = wind from the left")
    q3.add_argument("--quality", choices=("draft", "standard", "fine"), default="draft")
    q3.add_argument("--device", default="auto", help="auto, cuda or cpu")
    q3.add_argument("--re-mode", choices=("auto", "real", "capped"), default="auto",
                    help="Reynolds-number treatment (auto = real unless --re-cap is given)")
    q3.add_argument("--re-cap", type=_positive_float, default=None, help="Reynolds cap for --re-mode capped")
    q3.add_argument("--duration", type=_positive_float, default=None, help="run length in convective times L/U")
    q3.add_argument("--cells-per-ref", type=int, default=None,
                    help="grid cells per reference length (overrides the quality preset)")
    q3.add_argument("--no-ground", action="store_true", help="vehicles: no road (free air)")
    q3.add_argument("--fixed-ground", action="store_true", help="vehicles: fixed floor instead of a moving belt")
    q3.add_argument("--full-domain", action="store_true",
                    help="simulate both halves even when the shape is mirror-symmetric")
    q3.add_argument("--sensitivity", choices=("cd", "cl", "ld"), default=None,
                    help="also compute the 3D shape-sensitivity map of this objective (coarse grids only)")
    q3.add_argument("--save", help="save the design, simulation and result as a project in this folder")
    q3.add_argument("--plan-only", action="store_true", help="show the plan and stop")
    q3.add_argument("-v", "--verbose", action="store_true", help="also print the planner's reasoning")
    physics_args(q3, 3)
    q3.set_defaults(fn=cmd_quick3d)

    cv = sub.add_parser("cavity", help="natural convection in a differentially heated cavity, against its benchmark",
                        description="The differentially heated square (2D) or cubic (3D) cavity: air in a closed box, "
                                    "the left wall hotter than the right by the temperature difference that gives the "
                                    "Rayleigh number asked for, gravity downwards. Prints the hot wall's mean Nusselt "
                                    "number next to de Vahl Davis (1983, 2D, Ra 1e3-1e6) or Tric, Labrosse & Betrouni "
                                    "(2000, 3D, Ra 1e4-1e6).")
    cv.add_argument("--ra", type=_positive_float, default=1e4, help="Rayleigh number (default 1e4)")
    cv.add_argument("--pr", type=_positive_float, default=0.71, help="Prandtl number (default 0.71, air)")
    cv.add_argument("--size", type=_positive_float, default=0.1, help="side of the box in metres (default 0.1)")
    cv.add_argument("--3d", dest="three_d", action="store_true", help="the cubic cavity (3D) instead of the square one")
    cv.add_argument("--quality", choices=("draft", "standard", "fine"), default="draft")
    cv.add_argument("--device", default="auto", help="auto, cuda or cpu")
    cv.add_argument("--duration", type=_positive_float, default=None,
                    help="run length in free-fall times H/U (default: the planner's; the run stops once Nu settles)")
    cv.add_argument("--cells-per-ref", type=int, default=None,
                    help="grid cells across the box (overrides the quality preset)")
    cv.add_argument("--save", help="save the case and its result as a project in this folder")
    cv.add_argument("--plan-only", action="store_true", help="show the plan and stop")
    cv.add_argument("-v", "--verbose", action="store_true", help="also print the planner's and runner's notes")
    cv.set_defaults(fn=cmd_cavity)

    w = sub.add_parser("wing", help="finite-wing polar (vortex lattice or lifting line on viscous section polars)")
    w.add_argument("--naca", required=True, help="NACA 4- or 5-digit section, e.g. 2412")
    w.add_argument("--span", type=_positive_float, required=True, help="tip-to-tip span in metres")
    w.add_argument("--chord", type=_positive_float, required=True, help="root chord in metres")
    w.add_argument("--taper", type=float, default=1.0, help="tip chord / root chord (default 1)")
    w.add_argument("--sweep", type=float, default=0.0, help="quarter-chord sweep in degrees (default 0)")
    w.add_argument("--dihedral", type=float, default=0.0, help="dihedral in degrees (default 0)")
    w.add_argument("--twist", type=float, default=0.0, help="tip twist in degrees, negative = washout (default 0)")
    w.add_argument("--alpha", type=parse_alpha_range, default=(-4.0, 12.0, 1.0),
                   help="angles start:end:step in degrees (default -4:12:1)")
    w.add_argument("--method", choices=("vlm", "llt"), default="vlm",
                   help="vlm: vortex lattice (any planform); llt: nonlinear lifting line (straight wings)")
    w.add_argument("--inviscid", action="store_true", help="no section polars: induced drag only (fast)")
    w.add_argument("--speed", type=_speed, default=30.0, help="flow speed (sets the section Reynolds numbers)")
    w.add_argument("--yaw", type=float, default=0.0, help="sideslip in degrees")
    w.add_argument("--ncrit", type=_positive_float, default=9.0, help="e^N transition criterion (default 9)")
    w.add_argument("--mach", type=float, default=0.0, help="Mach number (Prandtl-Glauert)")
    w.add_argument("--csv", help="also write the polar to this CSV file")
    w.add_argument("--save", help="save the wing, analysis and result as a project in this folder")
    w.add_argument("-v", "--verbose", action="store_true", help="also print the solver's notes")
    w.set_defaults(fn=cmd_wing)

    r = sub.add_parser("run", help="run a simulation saved in a project and store its result")
    r.add_argument("project", help="project folder (Name.aerolume)")
    r.add_argument("simulation", help="simulation id (or a unique prefix) or name")
    r.add_argument("--device", default=None, help="override the simulation's device")
    r.add_argument("-v", "--verbose", action="store_true", help="also print the planner's and runner's notes")
    r.set_defaults(fn=cmd_run)

    sc = sub.add_parser("script", help="run a Python script against the scripting API (docs/scripting.md)",
                        description="Run FILE with `al` (the aerolume package), `project` (with --project), "
                                    "`args` (what follows --), `log` and `show` bound. Example: aerolume-cli script "
                                    "study.py --project 'Wing study.aerolume' -- 0.3 0.6")
    sc.add_argument("file", help="the Python script")
    sc.add_argument("--project", help="project folder bound as `project` (Name.aerolume)")
    sc.add_argument("--create", action="store_true", help="create the --project folder when it does not exist")
    sc.add_argument("--figures", help="folder for figures passed to show() (default: the current folder)")
    sc.add_argument("-v", "--verbose", action="store_true", help="print plans, progress and results of every run")
    sc.set_defaults(fn=cmd_script, script_args=[])

    v = sub.add_parser("validate", help="run the quick verification / validation subset")
    vm = v.add_mutually_exclusive_group()
    # --quick is the default; it is accepted because validation/run_all.py spells its subset that way
    vm.add_argument("--quick", action="store_true", help="the quick subset (the default)")
    vm.add_argument("--full", action="store_true",
                    help="also the Re 100 shedding case and the heated cavity (~30 s on a GPU)")
    v.add_argument("--device", default="auto")
    v.set_defaults(fn=cmd_validate)
    return p


#: Commands whose runs feed (and read) the persisted run-time calibration.
_CALIBRATED = ("bench", "quick", "quick3d", "cavity", "run", "script")


@contextlib.contextmanager
def _calibration(command: str):
    """Use the run-time calibration file in the user config folder while a
    solver command runs, so its measurements refine later estimates (the app
    reads the same file). ``AEROLUME_CALIBRATION=off`` disables it."""
    if command not in _CALIBRATED or os.environ.get("AEROLUME_CALIBRATION", "").lower() in ("0", "off", "no"):
        yield
        return
    from .project import config_dir
    from .simulation.plan import CALIBRATION_FILE, use_calibration_file
    use_calibration_file(config_dir() / CALIBRATION_FILE)
    try:
        yield
    finally:
        use_calibration_file(None)


_NEGATIVE_VALUE = re.compile(r"^-\.?\d")


def attach_negative_values(argv: Sequence[str]) -> list[str]:
    """``--alpha -4:14:1`` -> ``--alpha=-4:14:1`` (the same for any ``--option -value``).

    argparse reads a token that starts with "-" as an option unless it is a plain
    negative number, so an angle range that starts below zero ("-4:14:1", the
    documented way to write a polar sweep) was rejected with "expected one
    argument". No option name starts with a digit, so a "-" followed by a digit
    (or ".digit") after a long option is always that option's value.
    """
    args, out, i = list(argv), [], 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" not in a and i + 1 < len(args) and _NEGATIVE_VALUE.match(args[i + 1]):
            out.append(f"{a}={args[i + 1]}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def split_script_args(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """``script f.py --project P -- a -b`` -> (``script f.py --project P``, ``a -b``).

    Everything after the first ``--`` belongs to the script (``args`` /
    ``sys.argv``), options that look like ours included; argparse would
    otherwise try to read them.
    """
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``aerolume-cli``; returns the process exit code."""
    for stream in (sys.stdout, sys.stderr):
        try:                                  # never die on a console that cannot show "×" or "≈"
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    argv, script_args = split_script_args(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(attach_negative_values(argv))
    if script_args:
        if args.command != "script":
            parser.error("arguments after -- are only for the script command")
        args.script_args = script_args
    try:
        with _calibration(args.command):
            return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        _err("Interrupted.")
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        _err(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
