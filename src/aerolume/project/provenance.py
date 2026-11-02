"""Provenance records: what produced a result, so it can be checked, cited and reproduced.

Every result the project store saves carries a provenance record (``result.json`` key
``"provenance"``, read back with :meth:`Project.provenance`). It answers the questions
a reviewer asks of a number:

* **which software** -- Aerolume, Python, numpy, PyTorch (with its CUDA build and the
  GPU when the solver ran on one), VTK, the operating system;
* **which configuration** -- the simulation's full settings and the design's geometry
  as they were run, with SHA-256 hashes of each (``settings_hash``, ``design_hash``) and
  of both together (``config_hash``: the same hash means the same computation was asked
  for, whatever the names and ids), plus the planner's full resolved settings (grid,
  domain, time step, Reynolds numbers, turbulence model, every ``SolverConfig`` field)
  and their hash (``solver_hash``) when the run provided its plan;
* **which run** -- device, precision, wall time, finish time, grid and cells;
* **what randomness** -- ``seed``. The flow computation draws no random numbers: the
  start-up perturbation is a fixed Lamb-Oseen vortex (docs/theory_ns2d.md section 9),
  so ``seed`` is None and ``deterministic_inputs`` True. (3D tracer particles are
  released at random points, which moves only their dots in the display.) GPU
  reductions are not bitwise reproducible across devices or driver versions, so a
  reproduction is checked within the statistical error (docs/methods.md sections 2.8
  and 10; ``project/reproduce.py``).

The record is captured twice, and the two are merged: the runner attaches the plan
it used to the result (``result.provenance``, captured at run start, so a simulation
edited during its run is still recorded as it ran), and the store fills in anything
missing at save time from the project's design and simulation (``source`` says which
parts came from where). Hashes use :func:`canonical_json` (sorted keys, ``repr``
floats, arrays as their exact bytes), so they do not depend on dict order or on how
a float was printed.

This module imports no torch, VTK or Qt: versions are read from the installed
distributions' metadata, and torch's CUDA build and the GPU name only from a torch
that the solver has already imported.
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import logging
import math
import platform
import sys
import time
from collections.abc import Mapping
from importlib import metadata
from typing import Any

import numpy as np

from .. import __version__
from ..model import Design, Simulation

PROVENANCE_FORMAT = "aerolume-provenance"
PROVENANCE_SCHEMA = 1

#: Simulation fields that do not change what is computed (names, ids, status, timestamps).
SIM_METADATA = ("id", "name", "status", "created", "finished", "notes", "error", "design_id")
#: Design fields that do not change what is computed.
DESIGN_METADATA = ("id", "name", "created", "notes", "source")
#: Distributions whose versions are recorded (when installed).
log = logging.getLogger(__name__)

DISTRIBUTIONS = ("numpy", "scipy", "torch", "vtk", "matplotlib", "PySide6", "pyqtgraph")

DETERMINISM_NOTE = ("The flow computation draws no random numbers (the start-up perturbation is a fixed Lamb-Oseen "
                    "vortex), so the inputs determine every field, force and heat rate; only the display positions "
                    "of 3D tracer particles are random. Floating-point reductions on a GPU are not bitwise "
                    "reproducible across devices or driver versions, which changes time-averaged values only "
                    "within their statistical uncertainty.")


# ---------------------------------------------------------------------------
# canonical JSON and hashes
# ---------------------------------------------------------------------------

def _plain(v: Any) -> Any:
    """``v`` as JSON-able values; arrays are kept as small lists or summarised by digest."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        if hasattr(v, "to_dict"):
            return _plain(v.to_dict())
        return {f.name: _plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, np.ndarray):
        if v.size <= 64:
            return _plain(v.tolist())
        return array_digest(v)
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)        # "nan" / "inf": JSON has no non-finite numbers
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, bool)) or v is None:
        return v
    return repr(v)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text of ``obj``: sorted keys, no spaces, non-finite floats as strings,
    large arrays replaced by their digest (:func:`array_digest`)."""
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def array_digest(a: Any) -> dict[str, Any]:
    """``{"shape", "dtype", "sha256"}`` of an array's exact little-endian bytes."""
    arr = np.ascontiguousarray(np.asarray(a))
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return {"shape": list(arr.shape), "dtype": arr.dtype.str, "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def simulation_settings(sim: Simulation | Mapping[str, Any]) -> dict[str, Any]:
    """The computational settings of a simulation: ``to_dict()`` without :data:`SIM_METADATA`."""
    d = dict(sim.to_dict() if hasattr(sim, "to_dict") else sim)
    for k in SIM_METADATA:
        d.pop(k, None)
    return d


def settings_digest(sim: Simulation | Mapping[str, Any]) -> str:
    """SHA-256 of a simulation's computational settings (two simulations asking for the same
    computation of the same design share it, whatever their names, ids and status)."""
    return sha256_text(canonical_json(simulation_settings(sim)))


def design_geometry(design: Design) -> dict[str, Any]:
    """The computational content of a design: ``to_dict()`` without :data:`DESIGN_METADATA`, with
    the exact digest of every 3D mesh's vertices and faces (which ``to_dict`` leaves out)."""
    d = design.to_dict()
    for k in DESIGN_METADATA:
        d.pop(k, None)
    for i, m in enumerate(getattr(design, "meshes", None) or []):
        if i < len(d.get("meshes") or []):
            d["meshes"][i]["vertices"] = array_digest(np.asarray(m.vertices, np.float64))
            d["meshes"][i]["faces"] = array_digest(np.asarray(m.faces, np.int64))
    return d


def design_digest(design: Design) -> str:
    """SHA-256 of a design's geometry and reference quantities (names and ids excluded)."""
    return sha256_text(canonical_json(design_geometry(design)))


def config_hash(design: Design | None, sim: Simulation | Mapping[str, Any]) -> str:
    """SHA-256 of design geometry + simulation settings: the identity of a computation."""
    return sha256_text(canonical_json({"design": design_digest(design) if design is not None else None,
                                       "settings": settings_digest(sim)}))


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - not installed, or broken metadata
        return None


@functools.lru_cache(maxsize=1)
def _static_environment() -> tuple[tuple[str, Any], ...]:
    """The parts of the environment that cannot change while the process runs (read once:
    the metadata scan takes tens of milliseconds, and the store saves under its lock)."""
    return (("aerolume", __version__), ("python", platform.python_version()),
            ("implementation", platform.python_implementation()),
            ("os", f"{platform.system()} {platform.release()}"), ("os_version", platform.version()),
            ("machine", platform.machine()), ("processor", platform.processor() or None),
            ("packages", tuple((n, v) for n in DISTRIBUTIONS if (v := _dist_version(n)) is not None)))


def software_environment() -> dict[str, Any]:
    """Versions of Aerolume, Python, the OS and the numerical libraries (no heavy imports).

    torch's CUDA / cuDNN build and the GPU are added only when the solver has already
    imported torch (and initialised CUDA): asking would otherwise start CUDA in the
    saving thread.
    """
    env: dict[str, Any] = dict(_static_environment())
    env["packages"] = dict(env["packages"])
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            env["torch_cuda"] = getattr(torch.version, "cuda", None)
            cudnn = getattr(torch.backends, "cudnn", None)
            env["cudnn"] = cudnn.version() if cudnn is not None and cudnn.is_available() else None
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and cuda.is_available() and cuda.is_initialized():
                idx = cuda.current_device()
                env["gpu"] = cuda.get_device_name(idx)
                cap = cuda.get_device_capability(idx)
                env["gpu_capability"] = f"{cap[0]}.{cap[1]}"
                env["gpu_count"] = cuda.device_count()
        except Exception:  # noqa: BLE001 - provenance must never fail a save
            env["torch_error"] = "could not query torch"
    return env


# ---------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------

def _solver_config_record(cfg: Any) -> dict[str, Any]:
    """Every field of a (2D or 3D) SolverConfig; bodies and meshes summarised by size and digest."""
    out: dict[str, Any] = {"type": type(cfg).__name__}
    if not dataclasses.is_dataclass(cfg):
        return out
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if f.name in ("bodies", "meshes") and isinstance(v, (list, tuple)):
            items = []
            for b in v:
                item: dict[str, Any] = {"name": getattr(b, "name", ""), "role": getattr(b, "role", "")}
                for arr_name in ("points", "vertices", "faces"):
                    a = getattr(b, arr_name, None)
                    if a is not None:
                        item[arr_name] = array_digest(a)
                for k in ("omega", "center", "axis"):
                    if hasattr(b, k):
                        item[k] = _plain(getattr(b, k))
                items.append(item)
            out[f.name] = items
        else:
            out[f.name] = _plain(v)
    return out


_MISSING = object()


def _readable(obj: Any, name: str) -> Any:
    """``_plain(getattr(obj, name))``, or :data:`_MISSING` when the attribute cannot be read
    (a property of one plan type that the other lacks, or one that fails)."""
    try:
        return _plain(getattr(obj, name))
    except Exception:
        log.debug("plan attribute %s not recorded", name, exc_info=True)
        return _MISSING


#: SimPlan fields that are the design / simulation themselves or large geometry: recorded elsewhere.
_PLAN_SKIP = ("design", "sim", "bodies_wind", "meshes_wind", "solver_config")


def plan_record(plan: Any) -> dict[str, Any]:
    """The planner's full resolved settings: every scalar field of the plan, its ``summary()``
    (the Setup page's text, verbatim), the solver configuration (:func:`_solver_config_record`)
    and ``solver_hash``, the SHA-256 of that configuration."""
    rec: dict[str, Any] = {"type": type(plan).__name__}
    names = ([f.name for f in dataclasses.fields(plan) if f.name not in _PLAN_SKIP]
             if dataclasses.is_dataclass(plan) else [])
    for name in [*names, "grid", "time_unit", "estimate_label"]:   # the last three: properties of either plan type
        value = _readable(plan, name)
        if value is not _MISSING:
            rec[name] = value
    try:
        rec["summary"] = {str(k): str(v) for k, v in plan.summary().items()}
    except Exception:  # noqa: BLE001 - summary() is informational only
        rec["summary"] = {}
    cfg = getattr(plan, "solver_config", None)
    if cfg is not None:
        rec["solver_config"] = _solver_config_record(cfg)
        rec["solver_hash"] = sha256_text(canonical_json(rec["solver_config"]))
    return rec


#: The planner module; a save-time re-plan happens only where it is already imported.
PLANNER_MODULE = "aerolume.simulation.plan"
#: ``source`` entry of a plan record made at save time rather than captured by the runner.
REPLANNED = "store (re-planned at save; grid and Reynolds number match the run)"


def _same_grid(plan_grid: Any, result_grid: Any) -> bool:
    try:
        return [int(v) for v in plan_grid] == [int(v) for v in result_grid]
    except (TypeError, ValueError):
        return False


def replan_record(design: Design | None, sim: Simulation | None, result: Any, *,
                  airfoil_coords: Any = None) -> tuple[dict[str, Any] | None, str]:
    """``(plan record, note)`` for a CFD result saved without the runner's plan.

    The runner is the right place to capture the plan (:func:`attach`, at run start). Until
    it does, the store calls this at save time: the planner is deterministic and side-effect
    free (``simulation.plan.plan_simulation``: the same design, settings and device give the
    same grid, time step and solver configuration, measured identical ``solver_hash``), so
    planning the saved design and settings again on the device the run used reproduces the
    plan -- provided they were not edited during the run. That proviso is checked, not
    assumed: the re-plan is kept only when its grid and simulated Reynolds number equal the
    result's; otherwise the record says why it has no plan.

    Plans only in a process that has already imported the planner (one that ran the
    simulation), so saving a result never pulls in torch. Returns ``(None, reason)`` when
    there is nothing to plan (a polar or wing sweep: their settings are the whole
    configuration) or the plan cannot be made or does not match.
    """
    if sim is None or design is None or getattr(sim, "analysis", "cfd") != "cfd":
        return None, ""
    if not hasattr(result, "grid") or not hasattr(result, "re_sim"):
        return None, ""
    mod = sys.modules.get(PLANNER_MODULE)
    if mod is None:
        return None, "plan not recorded: the planner was not loaded in the saving process"
    label = str(getattr(result, "device", "") or "")
    device = "cpu" if label.upper().startswith("CPU") else None      # None: the device the settings ask for
    try:
        kw = {"airfoil_coords": airfoil_coords} if airfoil_coords is not None else {}
        plan = mod.plan_simulation(design, sim, device=device, **kw)
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a save
        return None, f"plan not recorded: re-planning at save failed ({type(exc).__name__}: {exc})"
    grid = _readable(plan, "grid")
    if grid is _MISSING or not _same_grid(grid, getattr(result, "grid", ())):
        return None, (f"plan not recorded: re-planning at save gave grid {grid!r}, the run used "
                      f"{list(getattr(result, 'grid', ()))!r} (settings edited during the run?)")
    re_plan = _readable(plan, "re_sim")
    re_run = float(getattr(result, "re_sim", math.nan))
    if isinstance(re_plan, (int, float)) and math.isfinite(re_plan) and math.isfinite(re_run) and \
            not math.isclose(float(re_plan), re_run, rel_tol=1e-6):
        return None, (f"plan not recorded: re-planning at save gave Re_sim {re_plan:.6g}, the run used "
                      f"{re_run:.6g}")
    return plan_record(plan), ""


# ---------------------------------------------------------------------------
# capture and merge
# ---------------------------------------------------------------------------

def _run_record(result: Any, sim: Simulation | None) -> dict[str, Any]:
    keys = ("device", "precision", "wall_seconds", "grid", "cells", "cells_per_ref", "re_real", "re_sim",
            "t_end", "t_avg_start", "converged", "turbulence", "symmetry_plane", "method", "viscous", "re", "mach",
            "ncrit")
    rec = {k: _plain(getattr(result, k)) for k in keys if hasattr(result, k)}
    rec["result_type"] = type(result).__name__
    # the status and finish time are set by the caller after the save, so they are not
    # recorded here (the record's "saved" time stamps the result instead)
    if sim is not None and sim.finished:
        rec["finished"] = sim.finished
    return rec


def capture(design: Design | None, sim: Simulation | None, *, plan: Any = None, result: Any = None,
            seed: int | None = None, extra: Mapping[str, Any] | None = None, source: str = "runner") -> dict:
    """A provenance record for ``result`` of ``sim`` on ``design`` (any argument may be None).

    ``plan`` adds the planner's resolved settings (:func:`plan_record`); ``seed`` is the
    random seed of any stochastic part (None: the run draws no random numbers);
    ``extra`` is merged in last (a caller's own keys, e.g. ``{"command": ...}``).
    ``source`` names who captured it ("runner" at run start, "store" at save time).
    Never raises for an odd input: a part that cannot be recorded is left out and named
    in ``errors``.
    """
    rec: dict[str, Any] = {"format": PROVENANCE_FORMAT, "schema": PROVENANCE_SCHEMA,
                           "captured": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": {}, "errors": []}

    def part(name: str, fn: Any) -> None:
        try:
            rec[name] = fn()
            rec["source"][name] = source
        except Exception as exc:  # noqa: BLE001 - provenance must never fail a run or a save
            rec["errors"].append(f"{name}: {type(exc).__name__}: {exc}")

    part("software", software_environment)
    if sim is not None:
        part("simulation", sim.to_dict)       # as the model writes it (NaN kept), so it reloads
        part("settings_hash", lambda: settings_digest(sim))
    if design is not None:
        part("design", design.to_dict)
        part("design_hash", lambda: design_digest(design))
    if sim is not None:
        part("config_hash", lambda: config_hash(design, sim))
    if plan is not None:
        part("plan", lambda: plan_record(plan))
    if result is not None:
        part("run", lambda: _run_record(result, sim))
    rec["seed"] = seed
    rec["deterministic_inputs"] = seed is None
    rec["determinism"] = DETERMINISM_NOTE
    if extra:
        rec.update({str(k): _plain(v) for k, v in extra.items()})
    if not rec["errors"]:
        rec.pop("errors")
    return rec


def merge(primary: Mapping[str, Any] | None, fallback: Mapping[str, Any] | None) -> dict[str, Any]:
    """``primary`` (the runner's record) with every part it lacks taken from ``fallback`` (the
    store's); ``source`` keeps which record each part came from. The run record is merged key by
    key, because the runner captures it before the result exists."""
    a = dict(primary or {})
    b = dict(fallback or {})
    out = dict(b)
    out.update({k: v for k, v in a.items() if v is not None})
    src = dict(b.get("source") or {})
    src.update(a.get("source") or {})
    out["source"] = src
    if isinstance(b.get("run"), Mapping) or isinstance(a.get("run"), Mapping):
        run = dict(b.get("run") or {})
        run.update(a.get("run") or {})
        out["run"] = run
    errs = list(a.get("errors") or []) + list(b.get("errors") or [])
    if errs:
        out["errors"] = errs
    else:
        out.pop("errors", None)
    return out


def attach(result: Any, design: Design | None, sim: Simulation | None, *, plan: Any = None,
           seed: int | None = None, extra: Mapping[str, Any] | None = None) -> Any:
    """Attach a runner-side provenance record to ``result`` (``result.provenance``) and return it.

    For the simulation runner: call it with the plan the run used, before the result is
    saved, so the store records the settings *as run*. Results without the attribute
    are recorded from the store's design and simulation at save time instead.
    """
    rec = capture(design, sim, plan=plan, result=None, seed=seed, extra=extra, source="runner")
    try:
        result.provenance = merge(rec, getattr(result, "provenance", None))
    except AttributeError:          # a result type that refuses attributes still runs
        log.debug("result %s takes no provenance attribute", type(result).__name__)
    return result


def summary_rows(rec: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """(label, value) pairs of a provenance record for reports and the GUI, in reading order."""
    if not rec:
        return []
    sw = rec.get("software") or {}
    pk = sw.get("packages") or {}
    run = rec.get("run") or {}
    plan = rec.get("plan") or {}
    rows: list[tuple[str, str]] = [("Aerolume", str(sw.get("aerolume", "—")))]
    if pk.get("torch"):
        cuda = f", CUDA {sw['torch_cuda']}" if sw.get("torch_cuda") else ""
        rows.append(("PyTorch", f"{pk['torch']}{cuda}"))
    if sw.get("gpu"):
        rows.append(("GPU", str(sw["gpu"])))
    rows.append(("Python / OS", f"{sw.get('python', '—')} / {sw.get('os', '—')}"))
    if run.get("device"):
        rows.append(("Device, precision", f"{run.get('device')}, {run.get('precision') or '—'}"))
    if rec.get("config_hash"):
        rows.append(("Configuration hash", str(rec["config_hash"])[:16]))
    if plan.get("solver_hash"):
        rows.append(("Solver-settings hash", str(plan["solver_hash"])[:16]))
    rows.append(("Random seed", "none (deterministic inputs)" if rec.get("seed") is None else str(rec["seed"])))
    return rows


__all__ = ["DESIGN_METADATA", "DETERMINISM_NOTE", "PLANNER_MODULE", "PROVENANCE_FORMAT", "PROVENANCE_SCHEMA",
           "REPLANNED", "SIM_METADATA", "array_digest", "attach", "canonical_json", "capture", "config_hash",
           "design_digest", "design_geometry", "merge", "plan_record", "replan_record", "settings_digest",
           "sha256_text", "simulation_settings", "software_environment", "summary_rows"]
