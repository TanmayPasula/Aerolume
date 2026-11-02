"""Finite-wing analysis runs: :func:`run_wing` (DESIGN.md 7.7).

A ``kind == "wing"`` design carries a :class:`~aerolume.model.WingSpec`
planform; a simulation with ``analysis == "wing"`` sweeps the angle of attack
with :func:`aerolume.solvers.wing.wing_polar` (vortex lattice or nonlinear
lifting line on 2D viscous section polars, XFLR5-style) and returns a
:class:`~aerolume.model.WingResult`.

Two things live here rather than in the solver:

* **Project airfoils.** A section may name a 2D airfoil design of the same
  project ("design:<id>"). The solver cannot see the project, so the caller
  hands it in: ``project`` (anything with a ``designs`` mapping; the GUI's job
  manager passes the open project) or an ``airfoil_coords(name)`` callback.
  The outline goes through :func:`aerolume.solvers.airfoil.prepare_coords`,
  so a design that is not a single airfoil fails with that function's
  readable message.
* **Section-polar reuse.** A cold viscous sweep spends most of its time
  computing 2D section polars (one XFOIL-class polar per airfoil and Reynolds
  node); they depend only on the airfoil, Ncrit and Mach. A small cache of
  :class:`~aerolume.solvers.wing.SectionPolars` per (Ncrit, Mach, airfoil
  shapes) makes the second run of a study -- another speed, another planform,
  the other method -- take seconds. Project airfoils enter the key by a hash of
  their coordinates, so editing one is never served stale polars.
* **Polars kept between sessions.** The same caches are written to the user's
  configuration folder (``<config>/cache/section_polars``, a few KB per airfoil
  set) after every run that computed section points, and read back the first
  time a session needs them. A cold viscous polar of a two-section wing costs
  ~25 s of section polars on the development machine (validation_wing.md); the
  scripting API, the CLI and the assistant start a new process per task, so
  without this every script repeated that cost for a wing it had already
  analysed. The file key holds the solver settings, Ncrit, Mach, the airfoil
  fingerprint and :data:`POLAR_DISK_VERSION` (bumped whenever the 2D solver's
  numbers change), so a stale file is never read; unreadable files are ignored
  and removed. ``AEROLUME_NO_POLAR_CACHE=1`` turns the files off.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core import units
from ..core.fluid import display_name
from ..model import Design, Simulation, WingResult

__all__ = ["airfoil_resolver", "clear_polar_cache", "estimate_wing_seconds", "polar_cache_dir", "run_wing"]

ProgressFn = Callable[[float, str], None]
AirfoilCoords = Callable[[str], Any]

#: SectionPolars kept between runs (each holds a few polars per airfoil, ~KB).
POLAR_CACHE_SIZE = 4
_POLARS: OrderedDict[tuple, Any] = OrderedDict()
_POLARS_LOCK = threading.Lock()
#: Version of the on-disk section polars: bump it when the 2D solver's numbers change.
POLAR_DISK_VERSION = 1
#: Most cache files kept on disk (oldest dropped first); one per (settings, airfoil set).
POLAR_DISK_FILES = 64
#: Wall time of one 2D section point [s], measured on the development machine for a tapered
#: NACA 2412 wing (strip Re 3.1-5.5e5, -4..14 deg): 25.6 s for 44 points (0.58 s) when idle,
#: 40.0 s for 52 (0.77 s) with other jobs running; 0.55 s in validation_wing.md.
SECTION_POINT_SECONDS = 0.6
#: Wall time of the vortex solves per angle [s] (validation_wing.md: ~25 ms VLM, ~1 ms LLT)
#: plus a fixed overhead for building the lattice and the result.
WING_SOLVE_SECONDS = {"vlm": 0.03, "llt": 0.005}
WING_OVERHEAD_SECONDS = 0.3
#: Section points a run adds beyond its first pass (polars extended towards cl_max): the
#: same sweep at 25 m/s planned 46 points and computed 52.
SECTION_EXTENSION_FACTOR = 1.2


def polar_cache_dir() -> Path | None:
    """Folder of the section polars kept between sessions, or None when switched off."""
    if os.environ.get("AEROLUME_NO_POLAR_CACHE", "").strip() not in ("", "0"):
        return None
    from ..project.recent import config_dir
    return config_dir() / "cache" / "section_polars"


def _disk_path(key: tuple, settings: Mapping[str, float]) -> Path | None:
    folder = polar_cache_dir()
    if folder is None:
        return None
    ident = json.dumps([POLAR_DISK_VERSION, repr(key), sorted(settings.items())])
    return folder / (hashlib.sha1(ident.encode("utf-8")).hexdigest()[:24] + ".json")


def _load_disk(sp: Any, key: tuple) -> bool:
    """Merge the saved polars for ``key`` into ``sp``; True when some were read."""
    path = _disk_path(key, sp._settings())
    if path is None or not path.is_file():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        sp.load_state(state)
    except (OSError, ValueError, TypeError, KeyError):
        try:                                      # damaged or from other settings: recompute later
            path.unlink()
        except OSError:
            pass
        return False
    try:
        os.utime(path)                            # most recently used, for the pruning below
    except OSError:
        pass
    return bool(state.get("nodes"))


def _save_disk(sp: Any, key: tuple) -> None:
    """Write ``sp``'s polars for ``key`` (atomically) and keep at most POLAR_DISK_FILES files.

    Best effort: a read-only or full disk costs the next session the polars, never the run.
    """
    path = _disk_path(key, sp._settings())
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(sp.state(), separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
        files = sorted(path.parent.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for old in files[POLAR_DISK_FILES:]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


def clear_polar_cache(*, disk: bool = True) -> None:
    """Forget the section polars kept between wing runs (and, with ``disk``, between sessions)."""
    with _POLARS_LOCK:
        _POLARS.clear()
        folder = polar_cache_dir() if disk else None
        if folder is not None and folder.is_dir():
            for f in folder.glob("*.json"):
                try:
                    f.unlink()
                except OSError:
                    pass


def _designs_of(project: Any) -> Mapping[str, Design] | None:
    if project is None:
        return None
    if isinstance(project, Mapping):
        return project
    designs = getattr(project, "designs", None)
    return designs if isinstance(designs, Mapping) else None


def airfoil_resolver(project: Any = None, airfoil_coords: AirfoilCoords | None = None) -> AirfoilCoords | None:
    """A callable resolving "design:<id>" airfoil names to unit-chord outlines, or None.

    ``airfoil_coords`` wins when given; otherwise ``project`` (a Project, or a
    mapping id -> Design) is searched. Unknown ids and designs that are not a
    single airfoil raise ``ValueError`` with a message naming the section's
    airfoil, which the wing solver reports before any work is done.
    """
    if airfoil_coords is not None:
        return airfoil_coords
    designs = _designs_of(project)
    if designs is None:
        return None
    cache: dict[str, np.ndarray] = {}

    def resolve(name: str) -> np.ndarray:
        key = str(name).strip()
        if key in cache:
            return cache[key]
        if not key.lower().startswith("design:"):
            raise ValueError(f"Airfoil {name!r} is neither a NACA code nor a project airfoil ('design:<id>').")
        did = key.split(":", 1)[1].strip()
        d = designs.get(did)
        if d is None:
            raise ValueError(f"Airfoil {name!r}: there is no design with id {did!r} in this project (was it "
                             "deleted?). Choose another airfoil for that wing section.")
        if getattr(d, "dimension", 2) != 2:
            raise ValueError(f"Airfoil {name!r} refers to the 3D design {d.name!r}; a wing section needs a 2D "
                             "airfoil design.")
        from ..solvers.airfoil import prepare_coords
        try:
            coords, _chord, _info = prepare_coords(d)
        except ValueError as exc:
            raise ValueError(f"Airfoil {name!r} ({d.name}): {exc}") from exc
        cache[key] = np.asarray(coords, dtype=np.float64)
        return cache[key]
    return resolve


def _fingerprint(design: Design, resolver: AirfoilCoords | None) -> tuple:
    """Airfoil identities of a wing: NACA codes as they are, project airfoils by a coordinate hash."""
    out = []
    for s in design.wing.sections:
        name = str(s.airfoil).strip()
        if name.lower().startswith("design:"):
            if resolver is None:
                out.append((name, None))
                continue
            xy = np.ascontiguousarray(np.asarray(resolver(name), dtype=np.float64))
            out.append((name, hashlib.sha1(xy.tobytes()).hexdigest()))
        else:
            out.append((name, ""))
    return tuple(sorted(set(out)))


def _polar_key(ncrit: float, mach: float, fingerprint: tuple) -> tuple:
    return (round(float(ncrit), 6), round(float(mach), 6), fingerprint)


def _shared_polars(ncrit: float, mach: float, fingerprint: tuple, resolver: AirfoilCoords | None):
    """(SectionPolars, reused) for these settings and airfoils, kept across runs (LRU) and
    seeded from disk the first time a session needs them. ``reused`` is "memory", "disk" or ""."""
    from ..solvers.wing import SectionPolars
    key = _polar_key(ncrit, mach, fingerprint)
    with _POLARS_LOCK:
        sp = _POLARS.get(key)
        if sp is not None:
            _POLARS.move_to_end(key)
            sp.coords_provider = resolver or sp.coords_provider
            return sp, "memory"
        sp = SectionPolars(ncrit=ncrit, mach=mach, coords_provider=resolver)
        from_disk = _load_disk(sp, key)
        _POLARS[key] = sp
        while len(_POLARS) > POLAR_CACHE_SIZE:
            _POLARS.popitem(last=False)
        return sp, "disk" if from_disk else ""


def estimate_wing_seconds(design: Design, sim: Simulation, *, project: Any = None,
                          airfoil_coords: AirfoilCoords | None = None) -> float:
    """Expected wall time [s] of :func:`run_wing` for ``sim``, from what is already cached.

    A viscous run costs its section points (:data:`SECTION_POINT_SECONDS` each, times
    :data:`SECTION_EXTENSION_FACTOR` for the polars it extends towards cl_max); the
    shared cache, including the polars saved by earlier sessions, is asked which points
    are still missing (:func:`aerolume.solvers.wing.section_points_needed`), so a wing
    analysed before estimates about a second and a cold one the 20-60 s it takes. The
    vortex solves add milliseconds per angle. NaN when the inputs cannot be planned
    (the run itself would then raise a readable error).
    """
    from ..solvers.wing import section_points_needed
    if design.kind != "wing" or design.wing is None:
        return float("nan")
    ws, cond = sim.wing, sim.conditions
    alphas = [float(a) for a in ws.alphas()]
    secs = WING_OVERHEAD_SECONDS + WING_SOLVE_SECONDS.get(ws.method, 0.03) * len(alphas)
    if not ws.viscous:
        return secs
    try:
        resolver = airfoil_resolver(project, airfoil_coords)
        provider, _ = _shared_polars(ws.ncrit, ws.mach, _fingerprint(design, resolver), resolver)
        n = section_points_needed(design.wing, alphas, speed=float(cond.speed), fluid=cond.fluid,
                                  ncrit=float(ws.ncrit), mach=float(ws.mach),
                                  beta_deg=float(getattr(cond, "yaw_deg", 0.0) or 0.0),
                                  polar_provider=provider, coords_provider=resolver)
    except (ValueError, RuntimeError):
        return float("nan")
    return secs + SECTION_EXTENSION_FACTOR * SECTION_POINT_SECONDS * n


def run_wing(design: Design, sim: Simulation, *, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None, project: Any = None,
             airfoil_coords: AirfoilCoords | None = None, polar_provider: Any = None) -> WingResult:
    """Finite-wing polar of a wing design (DESIGN.md 7.6 / 7.7).

    Uses ``sim.wing`` (angles, method "vlm" | "llt", viscous, Ncrit, Mach),
    ``sim.conditions`` (speed and fluid set the strip Reynolds numbers; the yaw
    angle is the sideslip) and the design's references (``ref_area``, the mean
    chord ``length_ref``, ``span_ref``, ``moment_ref3``). "design:<id>"
    airfoils are resolved from ``project`` or ``airfoil_coords`` (see
    :func:`airfoil_resolver`). ``progress(fraction, message)`` and ``cancel``
    (a ``threading.Event``: the angles finished so far are returned, with a
    warning) work as for the other runs. Bad inputs raise ``ValueError`` with a
    readable message before any work is done.
    """
    from ..solvers.wing import wing_polar
    from .runner import _Progress
    if design.kind != "wing" or design.wing is None:
        raise ValueError("The wing analysis needs a wing design (a planform of sections); this design is a "
                         f"{design.kind!r} without one. Use CFD for other shapes.")
    cond, ws = sim.conditions, sim.wing
    if not (cond.speed > 0 and math.isfinite(cond.speed)):
        raise ValueError("The flow speed must be positive.")
    report = _Progress(progress)
    resolver = airfoil_resolver(project, airfoil_coords)
    reused, shared_key = "", None
    if polar_provider is None and ws.viscous:
        fp = _fingerprint(design, resolver)
        shared_key = _polar_key(ws.ncrit, ws.mach, fp)
        polar_provider, reused = _shared_polars(ws.ncrit, ws.mach, fp, resolver)
    computed0 = getattr(polar_provider, "points_computed", 0)
    alphas = [float(a) for a in ws.alphas()]
    method_txt = "vortex lattice" if ws.method == "vlm" else "lifting line"
    report(0.0, f"Wing polar ({method_txt}): {len(alphas)} angles at {units.sig(cond.speed, 4)} m/s")

    def fwd(frac: float, msg: str) -> None:
        report(0.99 * frac, f"Wing: {msg}")

    ref_area = float(design.ref_area) if design.ref_area > 0 else None
    ref_chord = float(design.length_ref) if design.length_ref > 0 else None
    ref_span = float(design.span_ref) if design.span_ref > 0 else None
    res = wing_polar(design.wing, alphas, speed=float(cond.speed), fluid=cond.fluid, method=ws.method,
                     viscous=bool(ws.viscous), ncrit=float(ws.ncrit), mach=float(ws.mach),
                     beta_deg=float(getattr(cond, "yaw_deg", 0.0) or 0.0), moment_ref=tuple(design.moment_ref3),
                     polar_provider=polar_provider, coords_provider=resolver, ref_area=ref_area, ref_chord=ref_chord,
                     ref_span=ref_span, sim_id=sim.id, progress=fwd if progress is not None else None, cancel=cancel)
    re_txt = units.fmt_reynolds(res.re_mac) if math.isfinite(res.re_mac) else "—"
    res.notes.insert(0, f"Reynolds number {re_txt} on the mean aerodynamic chord ({units.sig(res.mac, 4)} m) at "
                        f"{units.sig(cond.speed, 4)} m/s in {display_name(cond.fluid.name)}.")
    yaw = float(getattr(cond, "yaw_deg", 0.0) or 0.0)
    if abs(yaw) > 1e-9:
        res.notes.append(f"Sideslip {yaw:g}° (the simulation's yaw angle).")
    computed = getattr(polar_provider, "points_computed", 0) - computed0
    if shared_key is not None and computed > 0:
        _save_disk(polar_provider, shared_key)
    if reused and ws.viscous:
        res.notes.append("Section polars computed by an earlier wing run "
                         + ("of this session" if reused == "memory" else "(saved between sessions)")
                         + (" were reused." if computed == 0 else f" were reused; {computed} new section points "
                            "were added."))
    nc = sum(1 for p in res.points if p.converged)
    stopped = cancel is not None and cancel.is_set()
    report(1.0, ("Wing polar stopped" if stopped else "Wing polar finished")
           + f": {nc} of {len(res.points)} angles converged")
    return res
