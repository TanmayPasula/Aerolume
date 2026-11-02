"""The public scripting API: ``import aerolume as al`` (DESIGN.md section 8).

One synchronous, documented facade over the geometry tools, the planners, the
solvers, the project store, the figure builders, the exports and the report
writer. People use it from scripts and notebooks, ``aerolume-cli script`` runs
files against it, the in-app Script console binds it, and the AI experiment
assistant drives the app through it -- so a run started from any of them is
recorded in the project exactly as if it had been started in the GUI, and its
result opens on the Results page.

A short tour::

    import aerolume as al

    p = al.new_project("Wing study")                  # or al.open_project(path)
    d = p.add_naca("2412", chord=1.0)                 # or add_template / import_mesh / add_wing
    r = p.run_cfd(d, speed=30, alpha=4)               # one 2D or 3D Navier-Stokes run
    print(r.cd, r.cl, r.ld, r.converged)              # mean coefficients and the verdict
    pol = p.run_polar(d, re=1e6, alpha=(-4, 14, 1))   # viscous-inviscid polar (XFOIL-class)
    sw = p.sweep(d, alpha=[0, 4, 8])                  # one simulation per point -> Sweep
    print(sw.table()); sw.best("ld"); sw.plot("alpha", "cl", path="cl.png")
    p.compare([r, sw[2]]).table()                     # the Compare page's table
    p.report([r, pol], "study.pdf")                   # the Report page's PDF

Placed physics (DESIGN.md section 10) go into the same calls::

    case = al.cavity_case(ra=1e4)                     # de Vahl Davis' heated cavity, with its reference Nu
    r = p.run_case(case)                              # or p.run_cfd(case, quality="standard")
    print(r.nusselt, case.check(r))                   # measured against the published value
    r = p.run_cfd(d, speed=2, fans=[al.Fan(position=(1, 0.5), direction=(1, 0), diameter=0.3, value=5)],
                  emitters=[{"kind": "rake", "position": (-0.5, 0.2), "end": (-0.5, 0.8)}])
    gs = p.grid_study(d, speed=30, levels=3)          # Richardson extrapolation and the GCI

Rules every call follows (they are what makes scripts safe to re-run and to
hand to an assistant):

* **Synchronous.** A call returns when the work is done. It prints nothing
  unless ``verbose=True`` (or :func:`set_verbose`).
* **Progress and cancel.** Every call that runs a solver accepts
  ``progress(fraction, message)`` and ``cancel`` (a :class:`threading.Event`).
  A cancelled run keeps its partial result in the project, marked "cancelled",
  and the call raises :class:`Cancelled` (``exc.result`` holds the partial
  result). :func:`context` sets defaults for every call on the current thread:
  the Script console and the CLI use it, so Stop and Ctrl+C reach every run of
  a script without the script passing ``cancel=`` itself.
* **Readable errors.** Everything raised is an :class:`AerolumeError`
  subclass whose message is written for people: :class:`InputError` (a bad
  argument; also a ``ValueError``), :class:`NotFoundError` (an unknown design or
  simulation; also a ``LookupError``), :class:`RunError` (a run failed; also a
  ``RuntimeError``), :class:`ProjectFileError` (the project folder could not
  be read or written; also an ``OSError``) and :class:`Cancelled`. The
  original exception is chained (``exc.__cause__``).
* **Recorded like the GUI.** Designs, simulations and results go into the
  project with the GUI's defaults and names ("CFD · 30 m/s · α 4°"), so the
  app shows them in the Project tree and opens their results.
* **Re-runs are free.** A simulation with exactly the same design and settings
  that already has a complete result is not run again: its stored result is
  returned (``r.reused`` is True). ``rerun=True`` forces a new run of that same
  simulation. Adding a design that already exists with the same name and the
  same geometry returns the existing one. A script that is edited and run
  again therefore only computes what changed.
* **Lazy.** ``import aerolume`` and this module import only numpy and the data
  model; torch, VTK and matplotlib load when a solver, a 3D picture or a
  figure is first needed.

Who edits the project: a :class:`Host`
--------------------------------------

A :class:`Project` makes every change (add a design, record a status, run a
simulation) through a small :class:`Host` object. The default host edits the
project store directly on the calling thread and runs simulations there; this
is what scripts, notebooks and the CLI get. Inside the desktop app the Script
console and the assistant bind :class:`aerolume.app.scripting.AppHost`
instead, which performs each change on the GUI thread through ``AppState`` (so
every page refreshes at once) and runs simulations through the app's job queue
(so a script never competes with a GUI run for the GPU, and the Simulate page
shows its live view). Scripts do not see the difference.
"""
from __future__ import annotations

import atexit
import contextlib
import copy
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np

from . import __version__
from . import model as M
from .core import units
from .core.format import plain as plain_number
from .core.format import value_error_texts

__all__ = [
    "SETTINGS", "AerolumeError", "Cancelled", "Case", "Comparison", "Design", "Emitter", "Fan", "GridStudy",
    "HeatSource", "Host", "InputError", "NotFoundError", "Plan", "Port", "Project", "ProjectFileError", "Result",
    "RunError", "Setting", "Sweep", "ThermalSettings", "WallThermal", "cavity_case", "context", "fan_room_case",
    "file_source", "grid_convergence", "heated_cylinder_case", "new_project", "open_project", "physics_features",
    "plain", "plan",
    "reference_case", "script_traceback", "set_verbose", "show", "supported_physics", "version",
]

ProgressFn = Callable[[float, str], None]
DesignRef = "Design | M.Design | str"


def version() -> str:
    """The Aerolume version string."""
    return __version__


# ===========================================================================
# errors
# ===========================================================================

class AerolumeError(Exception):
    """Base class of every error the scripting API raises; the message is meant for people."""


class InputError(AerolumeError, ValueError):
    """A bad argument: an unknown NACA code, a negative speed, an analysis a design cannot have."""


class NotFoundError(AerolumeError, LookupError):
    """No design, simulation, template or file of that name; the message lists what exists."""

    def __str__(self) -> str:          # LookupError's KeyError-style quoting is not wanted
        return str(self.args[0]) if self.args else ""


class RunError(AerolumeError, RuntimeError):
    """A run failed: the planner refused the set-up, the solver diverged, the GPU ran out of memory."""


class ProjectFileError(AerolumeError, OSError):
    """The project folder could not be created, read or written."""


class Cancelled(AerolumeError):
    """The work was stopped (``cancel`` set, Stop pressed, Ctrl+C).

    ``result`` is what was finished before the stop -- a partial :class:`Result`
    (already saved in the project, marked "cancelled"), a partial :class:`Sweep`
    -- or None when nothing was.
    """

    def __init__(self, message: str = "Stopped.", result: Any = None) -> None:
        super().__init__(message)
        self.result = result


def _friendly(exc: BaseException) -> str:
    """One readable sentence for an exception from the layers below (the GUI's wording)."""
    name = type(exc).__name__
    text = str(exc).strip()
    if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], str):
        text = exc.args[0]
    if isinstance(exc, ModuleNotFoundError):
        return f"A required component is missing ({exc.name or text}); the installation may be incomplete."
    if name == "OutOfMemoryError" or "out of memory" in text.lower():
        return ("The GPU ran out of memory. Choose a lower quality preset, close other GPU programs, "
                "or run on the CPU (device='cpu').")
    if not text:
        return f"{name} (no message)."
    return text if name in ("RuntimeError", "ValueError", "KeyError", "LookupError") else f"{name}: {text}"


def _translate(exc: Exception) -> AerolumeError:
    """The API error class for an exception raised below the API."""
    from .project import ProjectError
    text = _friendly(exc)
    if isinstance(exc, (ProjectError, PermissionError, IsADirectoryError, NotADirectoryError)):
        return ProjectFileError(text)
    if isinstance(exc, FileNotFoundError):
        return NotFoundError(text)
    if isinstance(exc, OSError):
        return ProjectFileError(text)
    if isinstance(exc, (KeyError, LookupError)):
        return NotFoundError(text)
    if isinstance(exc, (ValueError, TypeError)):
        return InputError(text)
    if isinstance(exc, RuntimeError):
        return RunError(text)
    return AerolumeError(text)


def _api(fn: Callable) -> Callable:
    """Decorator: every exception that leaves a public call is an :class:`AerolumeError`.

    KeyboardInterrupt / SystemExit pass through untouched (the console's Stop and
    Ctrl+C rely on them); API errors raised by nested API calls are not wrapped twice.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except AerolumeError:
            raise
        except Exception as exc:
            raise _translate(exc) from exc
    return wrapper


# ===========================================================================
# per-thread defaults: cancel, progress, verbose, display
# ===========================================================================

@dataclasses.dataclass
class _Context:
    cancel: threading.Event | None = None
    progress: ProgressFn | None = None
    verbose: bool | None = None
    show: Callable[[Any], Any] | None = None
    project: Any = None


_LOCAL = threading.local()
_VERBOSE = False


def _ctx() -> _Context:
    return getattr(_LOCAL, "ctx", None) or _Context()


@contextlib.contextmanager
def context(*, cancel: threading.Event | None = None, progress: ProgressFn | None = None,
            verbose: bool | None = None, show: Callable[[Any], Any] | None = None,
            project: Project | None = None) -> Iterator[_Context]:
    """Defaults for every API call made on this thread inside the ``with`` block.

    ``cancel`` is used by every run that is not given its own (so one Stop
    reaches all of them); ``progress`` is called *in addition to* a call's own
    callback; ``verbose`` switches printing on or off; ``show`` is what
    :func:`show` does with a figure (the Script console displays it inline, the
    CLI saves a PNG); ``project`` is the project the caller has bound (the app's open
    project): :func:`open_project` of its folder returns it instead of a second copy
    of the same folder. Nested blocks inherit what they do not override.
    """
    old = getattr(_LOCAL, "ctx", None)
    base = old or _Context()
    _LOCAL.ctx = _Context(cancel if cancel is not None else base.cancel,
                          progress if progress is not None else base.progress,
                          verbose if verbose is not None else base.verbose,
                          show if show is not None else base.show,
                          project if project is not None else base.project)
    try:
        yield _LOCAL.ctx
    finally:
        _LOCAL.ctx = old


def set_verbose(on: bool = True) -> None:
    """Print plans, progress and results for every call (all threads) unless a call says otherwise."""
    global _VERBOSE
    _VERBOSE = bool(on)


def _verbose(flag: bool | None) -> bool:
    if flag is not None:
        return bool(flag)
    c = _ctx().verbose
    return _VERBOSE if c is None else bool(c)


def _cancel_event(cancel: threading.Event | None) -> threading.Event | None:
    return cancel if cancel is not None else _ctx().cancel


def _check_cancel(cancel: threading.Event | None, what: str = "") -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled(f"Stopped before {what}." if what else "Stopped.")


def _say(text: str) -> None:
    print(text, flush=True)


class _Progress:
    """Fan-out of one progress report: the call's callback, the context's, verbose printing.

    ``scale`` maps the run's own 0..1 onto a slice of a longer job (a sweep point);
    printing is throttled to one line every few seconds, so a log stays readable.
    """

    def __init__(self, fn: ProgressFn | None, verbose: bool, *, start: float = 0.0, span: float = 1.0,
                 prefix: str = "") -> None:
        self.fns = [f for f in (fn, _ctx().progress) if f is not None]
        self.verbose = verbose
        self.start, self.span, self.prefix = start, span, prefix
        self._last_print = -math.inf
        self._last_head = ""

    def sub(self, start: float, span: float, prefix: str) -> _Progress:
        p = _Progress(None, self.verbose, start=self.start + self.span * start, span=self.span * span,
                      prefix=prefix)
        p.fns = self.fns
        p._last_print = self._last_print
        return p

    def __call__(self, fraction: float, message: str = "") -> None:
        f = self.start + self.span * min(max(float(fraction), 0.0), 1.0)
        msg = f"{self.prefix}{message}" if self.prefix else str(message)
        for fn in self.fns:
            with contextlib.suppress(Exception):     # a broken progress callback must not stop a run
                fn(f, msg)
        if self.verbose:
            now = time.monotonic()
            head = msg.split("(")[0].strip()
            if fraction >= 1.0 or now - self._last_print >= 5.0 or head != self._last_head:
                _say(f"  {100 * f:5.1f} %  {msg}")
                self._last_print, self._last_head = now, head


@_api
def show(obj: Any) -> Any:
    """Display a figure or a result: inline in the Script console, a PNG file elsewhere.

    A matplotlib ``Figure`` (what ``plot()`` returns) is handed to the current
    :func:`context`'s display hook; without one it is saved as
    ``figure-<n>.png`` in the working folder and the path is printed. Anything
    else is printed. Returns ``obj`` so it can wrap an expression.
    """
    hook = _ctx().show
    if hook is not None:
        hook(obj)
        return obj
    if _is_figure(obj):
        global _FIGURE_COUNT
        _FIGURE_COUNT += 1
        path = Path.cwd() / f"figure-{_FIGURE_COUNT}.png"
        from .viz.figures import save_figure
        save_figure(obj, path, dpi=150)
        _say(f"Figure saved to {path}")
    else:
        _say(str(obj))
    return obj


_FIGURE_COUNT = 0


def _is_figure(obj: Any) -> bool:
    return type(obj).__name__ == "Figure" and hasattr(obj, "savefig")


def plain(obj: Any) -> Any:
    """A strict-JSON copy of what ``summary()`` / ``table()`` return: numpy numbers become
    Python numbers, NaN and infinities become None, tuples lists, paths strings.

    For tools that hand results to another program (the AI assistant's tool results,
    a web service): ``json.dumps(plain(result.summary()))`` never writes ``NaN``.
    """
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return plain(obj.to_dict() if hasattr(obj, "to_dict") else dataclasses.asdict(obj))
    return obj


def script_traceback(exc: BaseException, filename: str) -> tuple[str, int | None, str]:
    """How a script's error is shown (the Script console and ``aerolume-cli script``).

    Returns ``(one line, script line number, traceback text)``. The traceback
    keeps the frames of the script itself (``filename``) and counts the ones
    inside Aerolume and other libraries instead of listing them: a script
    author needs the line of their script and the readable message, not the
    planner's internals (``exc.__cause__`` keeps those for a bug report).
    """
    import traceback as tb
    frames = tb.extract_tb(exc.__traceback__)
    mine = [f for f in frames if f.filename == filename]
    hidden = len(frames) - len(mine)
    line = mine[-1].lineno if mine else None
    if isinstance(exc, SyntaxError) and exc.filename == filename:
        line = exc.lineno
    msg = str(exc).strip()
    one = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    parts = ["Traceback (most recent call last):\n", *tb.format_list(mine)]
    if hidden:
        parts.append(f"  ({hidden} frame{'s' if hidden != 1 else ''} inside Aerolume or other libraries not shown)\n")
    if isinstance(exc, SyntaxError):
        parts += tb.format_exception_only(type(exc), exc)
    else:       # "NotFoundError: ...", not "aerolume.api.NotFoundError: ..."
        parts.append(one + "\n")
    return one, line, "".join(parts)


# ===========================================================================
# small helpers
# ===========================================================================

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _re_text(re_: float) -> str:
    """'1.0×10⁶' -- the Reynolds number as the GUI writes it in simulation names."""
    if not (isinstance(re_, (int, float)) and math.isfinite(re_) and re_ > 0):
        return "—"
    if re_ < 1e4:
        return f"{re_:,.0f}"
    e = math.floor(math.log10(re_))
    m = re_ / 10 ** e
    if round(m, 1) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{m:.1f}×10{str(e).translate(_SUP)}"


def _mean_error(mean: float, stderr: float) -> str:
    """"0.751 ± 0.009", "≈0 ± 0.019" or "4.562 (steady)": a time average rounded to its standard error by
    the Results table's rule (:func:`aerolume.core.format.value_error_texts`), in plain notation.

    The headline a printed result starts with said "CL 0.06323 ± 0.019" while the table under it said
    "0.063 ± 0.019"; the digits beyond the error are noise, and the two lines disagreed.
    """
    value, err = (plain_number(t) for t in value_error_texts(mean, stderr))
    if err in ("—", ""):
        return value
    return f"{value} ({err})" if err == "steady" else f"{value} ± {err}"


def _num(x: float, digits: int = 3) -> str:
    """``x`` to ``digits`` significant figures without trailing zeros: 4 -> "4", 0.3405 -> "0.3405"."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v) < 1e-10:          # round-off (the lift of a symmetric section at zero incidence)
        v = 0.0
    s = units.sig(v, digits, grouping=False)
    if "." in s and "e" not in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _count(n: int, word: str, plural: str | None = None) -> str:
    """'1 simulation', '3 simulations': a count with the noun in the right number."""
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _float(name: str, value: Any, *, positive: bool = False, allow_zero: bool = True) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {value!r}.") from None
    if not math.isfinite(v):
        raise InputError(f"{name} must be a finite number, got {value!r}.")
    if positive and not (v > 0 or (allow_zero and v == 0)):
        raise InputError(f"{name} must be positive, got {value:g}.")
    return v


def _angle_range(value: Any, what: str = "alpha") -> tuple[float, float, float]:
    """(start, end, step) from ``(a, b, s)``, ``(a, b)``, a number, "a:b:s" or evenly spaced values."""
    if isinstance(value, str):
        parts = [p for p in value.replace(",", ":").split(":") if p.strip()]
        value = tuple(float(p) for p in parts)
    if isinstance(value, (int, float, np.floating, np.integer)):
        v = _float(what, value)
        return v, v, 1.0
    vals = [float(v) for v in np.asarray(list(value), dtype=float).ravel()]
    if isinstance(value, tuple) and len(vals) in (2, 3):
        a, b = vals[0], vals[1]
        s = vals[2] if len(vals) == 3 else 1.0
        if not (s > 0 and math.isfinite(a) and math.isfinite(b)):
            raise InputError(f"{what}=(start, end, step) needs a positive step, got {value!r}.")
        if b < a:
            raise InputError(f"{what}=(start, end, step) needs end >= start, got {value!r}.")
        return a, b, s
    if not vals:
        raise InputError(f"{what} is empty.")
    if len(vals) == 1:
        return vals[0], vals[0], 1.0
    steps = np.diff(vals)
    if np.all(steps > 0) and np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        return vals[0], vals[-1], float(steps[0])
    raise InputError(f"{what} must be evenly spaced angles or (start, end, step); got {value!r}. A polar sweeps "
                     "a regular range of angles.")


def _fluid(fluid: Any, altitude: float | None) -> M.Fluid:
    from .core import fluid as F
    if fluid is None:
        return F.air_isa(0.0 if altitude is None else _float("altitude", altitude))
    if altitude is not None:
        raise InputError("Give either fluid= or altitude= (ISA air at that altitude), not both.")
    if isinstance(fluid, M.Fluid):
        return fluid
    if isinstance(fluid, Mapping):
        return M.Fluid.from_dict(dict(fluid))
    if isinstance(fluid, str):
        key = fluid.strip().lower()
        if key in ("air", "isa", "air isa"):
            return F.air_isa(0.0)
        if key == "water":
            return F.water(20.0)
        raise InputError(f"Unknown fluid {fluid!r}: use 'air', 'water' or an aerolume.model.Fluid.")
    raise InputError(f"fluid must be 'air', 'water' or an aerolume.model.Fluid, got {type(fluid).__name__}.")


def _digest(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def _geometry_key(d: M.Design, *, thermal: bool = True) -> str:
    """A fingerprint of a design's shape and references (what makes two designs 'the same').

    ``thermal=False`` leaves the bodies' heating out: the shape alone.
    """
    parts: list[Any] = [d.kind, d.dimension, round(float(d.length_ref), 12), round(float(d.force_ref), 12),
                        round(float(d.ref_area), 12), round(float(d.span_ref), 12)]
    for b in d.bodies:
        parts += [b.name, b.role, round(float(b.omega), 12), np.round(np.asarray(b.points, float), 12)]
        if thermal and getattr(b, "thermal", "adiabatic") != "adiabatic":
            # a heated copy of a shape is a different design (its runs transfer heat); the condition is
            # added only when set, so every design made before heat transfer keeps its fingerprint
            parts += [b.thermal, round(float(b.thermal_value), 12)]
    for m in d.meshes:
        parts += [m.name, m.role, np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)]
        if thermal and getattr(m, "thermal", "adiabatic") != "adiabatic":
            parts += [m.thermal, round(float(m.thermal_value), 12)]
    if d.wing is not None:
        parts.append(d.wing.to_dict())
    if (d.source or {}).get("type") == "case":
        parts.append(d.source)          # two empty cavities of different size are different designs
    return _digest(*parts)


#: Simulation fields of the placed physics (DESIGN.md 10): part of what defines a CFD run.
_PHYSICS_FIELDS = ("emitters", "thermal", "heat_sources", "fans", "ports")


def _settings_key(sim: M.Simulation) -> str:
    """The settings that define a run (not its name, id or status): equal keys -> equal runs.

    For CFD the placed physics are part of the key: a run with a fan must never
    be answered by the stored result of the same case without it. Empty physics
    serialise identically for old and new simulations, so stored runs made
    before the physics existed are still found.
    """
    d = sim.to_dict()
    keep = ["design_id", "analysis", "conditions"]
    keep += {"cfd": ["ground", "cfd", *_PHYSICS_FIELDS], "polar": ["airfoil"], "wing": ["wing"]}[sim.analysis]
    return json.dumps({k: d.get(k) for k in keep}, sort_keys=True, default=str)


def _unique(name: str, taken: Iterable[str]) -> str:
    taken = set(taken)
    out, k = name, 2
    while out in taken:
        out, k = f"{name} {k}", k + 1
    return out


def _speed_text(v: float) -> str:
    return f"{_num(v, 3)} m/s"


# ===========================================================================
# the host: who edits the project, and where runs execute
# ===========================================================================

class Host:
    """How a :class:`Project` edits its store and runs simulations.

    This base class is the headless behaviour: every change goes straight to the
    project store on the calling thread and is saved at once, and
    :meth:`run_simulation` returns None so the API runs the solver itself. The
    desktop app's :class:`aerolume.app.scripting.AppHost` overrides the methods
    to do the same work on the GUI thread through ``AppState`` and to run
    simulations through the job queue. Every method receives the store
    (:class:`aerolume.project.Project`) it acts on.
    """

    #: whether runs render a live view (the app does; estimates include that cost)
    live_view: bool = False

    def call(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` where the project may be edited (here: at once, on this thread)."""
        return fn()

    def defaults(self) -> dict[str, str]:
        """Device and precision a new simulation gets ({"device": ..., "precision": ...})."""
        return {"device": "auto", "precision": "auto"}

    def check(self, store: Any) -> None:
        """Raise :class:`AerolumeError` when the store can no longer be edited (closed project)."""

    def add_design(self, store: Any, design: M.Design) -> None:
        store.add_design(design)
        store.save()

    def update_design(self, store: Any, design: M.Design) -> None:
        store.update_design(design)
        store.save()

    def remove_design(self, store: Any, design_id: str) -> None:
        store.remove_design(design_id)
        store.save()

    def add_simulation(self, store: Any, sim: M.Simulation) -> None:
        store.add_simulation(sim)
        store.save()

    def update_simulation(self, store: Any, sim: M.Simulation) -> None:
        store.update_simulation(sim)
        store.save()

    def remove_simulation(self, store: Any, sim_id: str) -> None:
        store.remove_simulation(sim_id)
        store.save()

    def result_saved(self, store: Any, sim_id: str) -> None:
        """A result of ``sim_id`` was written by the API (the app refreshes its pages)."""

    def result_reused(self, store: Any, sim_id: str) -> None:
        """A run returned the stored result of ``sim_id`` instead of running (the Script console says so)."""

    def report_written(self, store: Any, path: Path) -> None:
        """A report was written into the project's reports folder."""

    def run_simulation(self, store: Any, sim_id: str, *, progress: ProgressFn,
                       cancel: threading.Event | None) -> tuple[Any, str] | None:
        """Run a saved simulation and return ``(result, status)`` with status "done" or
        "cancelled" (the result already saved), raise :class:`RunError` when it fails, or
        return None to let the API run it on the calling thread (the default)."""
        return None

    def show(self, store: Any, sim_id: str) -> None:
        """Bring a simulation's result to the user's attention (the app selects it)."""


_DIRECT_HOST = Host()


# ===========================================================================
# plan (an estimate without running)
# ===========================================================================

#: Rough run-time model of the fast analyses [s] (measured on the reference machine:
#: a cold NACA 2412 polar spends ~12 s starting the viscous solver, then ~1.4 s per
#: angle; a cold viscous wing sweep ~24 s for its section polars, under 1 s warm).
POLAR_SECONDS = (12.0, 1.4)
WING_SECONDS = {"viscous": (24.0, 0.05), "inviscid": (0.5, 0.05)}


class Plan:
    """What a run would cost and do, without running it (:func:`plan`, :meth:`Project.plan`).

    ``est_seconds`` is the expected wall time (headless, or with the app's live
    view when the project is bound to the app); ``estimate`` the same as the
    Run button writes it ("~40 s"). CFD plans also carry the grid, cells, the
    real and simulated Reynolds numbers and the planner's ``notes`` and
    ``warnings`` (the Setup page's messages); ``raw`` is the planner's
    ``SimPlan`` / ``SimPlan3D``. Polar and wing estimates are rough.
    """

    def __init__(self, analysis: str, *, raw: Any = None, design: M.Design | None = None,
                 simulation: M.Simulation | None = None, est_seconds: float = math.nan,
                 notes: Sequence[str] = (), warnings: Sequence[str] = (), rows: Mapping[str, str] | None = None,
                 re_real: float = math.nan, re_sim: float = math.nan) -> None:
        self.analysis = analysis
        self.raw = raw
        self.design = design
        self.simulation = simulation
        self.est_seconds = float(est_seconds)
        self.notes = list(notes)
        self.warnings = list(warnings)
        self._rows = dict(rows or {})
        self.re_real = float(re_real)
        self.re_sim = float(re_sim)

    @property
    def estimate(self) -> str:
        """The estimate as the Run button shows it: '~40 s', '~5 min' ('' when unknown)."""
        s = self.est_seconds
        if not _finite(s) or s <= 0:
            return ""
        if s < 10:
            return "< 10 s"
        if s < 57.5:
            return f"~{5 * round(s / 5):.0f} s"
        if s < 3570:
            return f"~{max(1, round(s / 60))} min"
        h, m = divmod(round(s / 60), 60)
        return f"~{h} h" + (f" {m:02d} min" if m else "")

    @property
    def cells(self) -> int:
        return int(getattr(self.raw, "cells", 0) or 0)

    @property
    def grid(self) -> tuple[int, ...]:
        return tuple(int(n) for n in getattr(self.raw, "grid", ()) or ())

    @property
    def quality(self) -> str:
        return str(getattr(self.raw, "quality", "") or "")

    @property
    def device(self) -> str:
        return str(getattr(self.raw, "device_name", "") or getattr(self.raw, "device", "") or "")

    def summary(self) -> dict[str, str]:
        """Label -> value, as the Setup page's Estimate table shows it."""
        rows = dict(self._rows)
        if self.raw is not None and hasattr(self.raw, "summary"):
            rows = dict(self.raw.summary())
            if "Estimated run time" in rows and _finite(self.est_seconds):
                from .simulation.plan import fmt_duration
                rows["Estimated run time"] = fmt_duration(self.est_seconds)
        return rows

    def __repr__(self) -> str:
        bits = [self.analysis]
        if self.cells:
            bits.append(f"{self.cells:,} cells")
        if self.estimate:
            bits.append(self.estimate)
        if self.warnings:
            bits.append(_count(len(self.warnings), "warning"))
        return f"<Plan {' · '.join(bits)}>"

    def __str__(self) -> str:
        rows = self.summary()
        w = max((len(k) for k in rows), default=0)
        lines = [f"{k.ljust(w)}  {v}" for k, v in rows.items()]
        lines += [f"! {x}" for x in self.warnings]
        return "\n".join(lines)


def _make_plan(design: M.Design, sim: M.Simulation, store: Any, host: Host) -> Plan:
    if sim.analysis == "polar":
        n = len(sim.airfoil.alphas())
        est = POLAR_SECONDS[0] + POLAR_SECONDS[1] * n
        re_ = sim.conditions.reynolds(float(design.length_ref))
        return Plan("polar", design=design, simulation=sim, est_seconds=est, re_real=re_, re_sim=re_,
                    rows={"Reynolds number": _re_text(re_), "Angles": f"{n}",
                          "Estimated run time": f"about {est:.0f} s (rough)"},
                    notes=["The first polar of a session also starts the viscous solver (about 10 s)."])
    if sim.analysis == "wing":
        n = len(sim.wing.alphas())
        a, b = WING_SECONDS["viscous" if sim.wing.viscous else "inviscid"]
        est = a + b * n
        return Plan("wing", design=design, simulation=sim, est_seconds=est,
                    rows={"Method": "vortex lattice" if sim.wing.method == "vlm" else "lifting line",
                          "Angles": f"{n}", "Estimated run time": f"about {est:.0f} s (rough)"},
                    notes=[("Viscous wing runs reuse section polars computed earlier in the session "
                            "(then they take about a second).")] if sim.wing.viscous else [])
    from .simulation.plan import plan_simulation
    kw: dict[str, Any] = {}
    if design.dimension == 3 and store is not None:
        from .simulation.wing import airfoil_resolver
        kw["airfoil_coords"] = airfoil_resolver(store)
    raw = plan_simulation(design, sim, **kw)
    headless = float(getattr(raw, "est_seconds_headless", math.nan))
    est = raw.est_seconds if host.live_view or not _finite(headless) else headless
    warnings = list(raw.warnings)
    missing = _missing_physics(design, sim)
    if missing:
        warnings.insert(0, f"This version's solver does not implement {', '.join(_FEATURE_WORDS[f] for f in missing)} "
                           "yet: a run of this set-up is refused (DESIGN.md 10).")
    return Plan("cfd", raw=raw, design=design, simulation=sim, est_seconds=est, notes=raw.notes,
                warnings=warnings, re_real=raw.re_real, re_sim=raw.re_sim)


@_api
def plan(design: Design | M.Design, analysis: str | None = None, **settings: Any) -> Plan:
    """Plan a run without running it: cells, grid, run time, Reynolds numbers, the planner's notes.

    ``design`` is a :class:`Design` (from a project) or an ``aerolume.model.Design``;
    ``settings`` are the keywords of :meth:`Project.run_cfd` / :meth:`Project.run_polar`
    / :meth:`Project.run_wing` (``analysis`` picks which; default: what the GUI would
    start with, "wing" for wings and "cfd" otherwise). Nothing is recorded.
    """
    if isinstance(design, Design):
        return design.project.plan(design, analysis, **settings)
    if not isinstance(design, M.Design):
        raise InputError(f"plan() needs a design, got {type(design).__name__}.")
    sim = _build_simulation(design, analysis, settings, _DIRECT_HOST.defaults())
    return _make_plan(design, sim, None, _DIRECT_HOST)


# ===========================================================================
# simulation settings (shared by run_*, sweep, plan and add_simulation)
# ===========================================================================

_CONDITION_KEYS = ("speed", "re", "alpha", "yaw", "fluid", "altitude")
_CFD_KEYS = ("quality", "ground", "moving_ground", "rotate_wheels", "device", "precision", "sensitivity",
             "symmetry", "reynolds", "re_cap", "duration", "cells_per_ref", "turbulence", "average_from")
_POLAR_KEYS = ("ncrit", "xtr", "xtr_upper", "xtr_lower", "mach", "panels")
_WING_KEYS = ("method", "viscous", "ncrit", "mach")
_CFD_FIELDS = {f.name for f in dataclasses.fields(M.CFDSettings)}


def _analyses_for(design: M.Design) -> list[str]:
    """The analyses a design can have, default first (the shell's rule, ``app.state.analyses_for``)."""
    if design.kind == "wing":
        return ["wing", "cfd"]
    if design.dimension == 2 and design.kind == "airfoil" and len(design.bodies) == 1:
        return ["cfd", "polar"]
    return ["cfd"]


def _build_simulation(design: M.Design, analysis: str | None, settings: dict[str, Any],
                      defaults: Mapping[str, str], name: str = "") -> M.Simulation:
    """A Simulation of ``design`` with the GUI's defaults and ``settings`` applied (validated)."""
    from .simulation.defaults import default_cfd_settings, default_simulation
    allowed = _analyses_for(design)
    analysis = (analysis or allowed[0]).lower()
    if analysis not in M.ANALYSES:
        raise InputError(f"Unknown analysis {analysis!r}; use 'cfd', 'polar' or 'wing'.")
    if analysis not in allowed:
        why = {"polar": "a polar needs a single 2D airfoil", "wing": "a wing analysis needs a wing planform"}
        raise InputError(f"Design {design.name!r} cannot have a {analysis!r} analysis: "
                         f"{why.get(analysis, 'not available')}. It can have: {', '.join(allowed)}.")
    s = dict(settings)
    known = set(_CONDITION_KEYS)
    known |= set({"cfd": _CFD_KEYS + _PHYSICS_KEYS, "polar": _POLAR_KEYS, "wing": _WING_KEYS}[analysis])
    if analysis != "cfd":
        physics = [k for k in s if k in _PHYSICS_KEYS and s[k] not in (None, [], (), False)]
        if physics:
            raise InputError(f"{', '.join(physics)} {'is' if len(physics) == 1 else 'are'} placed physics of CFD runs; "
                             f"a {analysis!r} analysis has none (DESIGN.md 10). Use run_cfd().")
        for k in _PHYSICS_KEYS:
            s.pop(k, None)
    extra = {k: s.pop(k) for k in list(s) if k not in known}
    if analysis == "cfd":
        unknown = [k for k in extra if k not in _CFD_FIELDS]
    else:
        unknown = list(extra)
    if unknown:
        import difflib
        valid = sorted(known | (_CFD_FIELDS if analysis == "cfd" else set()))
        close = [m for k in unknown for m in difflib.get_close_matches(k, valid, n=1)]
        hint = f" Did you mean {', '.join(map(repr, close))}?" if close else ""
        raise InputError(f"Unknown setting{'s' if len(unknown) > 1 else ''} {', '.join(map(repr, unknown))} for a "
                         f"{analysis!r} analysis.{hint} Valid: {', '.join(valid)}.")
    # --- conditions ---------------------------------------------------------
    fluid_arg, altitude_arg = s.pop("fluid", None), s.pop("altitude", None)
    t_amb = _given_ambient(s.get("thermal")) if analysis == "cfd" else None
    if fluid_arg is None and altitude_arg is None and t_amb is not None:
        # thermal={"t_ambient": 25} and no fluid: air at 25 °C, so the density and viscosity belong to the
        # same temperature as the conductivity and β (built before re= turns into a speed)
        fluid = _air(t_amb)
    else:
        fluid = _fluid(fluid_arg, altitude_arg)
    speed, re_ = s.pop("speed", None), s.pop("re", None)
    speed_given = speed is not None or re_ is not None
    if speed is not None and re_ is not None:
        raise InputError("Give either speed= or re= (the speed is then computed from the reference length), "
                         "not both.")
    if analysis == "polar" and speed is None and re_ is None:
        re_ = 1.0e6
    if re_ is not None:
        re_v = _float("re", re_, positive=True, allow_zero=False)
        L = float(design.length_ref)
        if not L > 0:
            raise InputError(f"Design {design.name!r} has no reference length; give speed= instead of re=.")
        speed_v = re_v * fluid.viscosity / (fluid.density * L)
    else:
        speed_v = _float("speed", 30.0 if speed is None else speed, positive=True, allow_zero=False)
    alpha = s.pop("alpha", None)
    yaw = _float("yaw", s.pop("yaw", 0.0) or 0.0)
    if yaw and design.dimension != 3:
        raise InputError("yaw= needs a 3D design; a 2D section has no sideslip.")
    cond = M.FlowConditions(fluid=fluid, speed=speed_v, alpha_deg=0.0, yaw_deg=yaw)
    if analysis == "cfd":
        if alpha is not None and not isinstance(alpha, (int, float, np.floating, np.integer)):
            raise InputError("run_cfd() takes one angle; use sweep(design, alpha=[...]) for several.")
        cond.alpha_deg = _float("alpha", alpha or 0.0)
    device = str(s.pop("device", None) or defaults.get("device", "auto"))
    precision = str(s.pop("precision", None) or defaults.get("precision", "auto"))
    sim = default_simulation(design, name, cond, device=device, precision=precision)
    # how the caller set the flow names the run ("CFD 3D · Re 100" for re=100): not a model field,
    # so it never reaches the project files
    sim._named_by_re = re_ is not None  # type: ignore[attr-defined]
    if sim.analysis != analysis:
        sim.analysis = analysis
    if analysis == "cfd":
        sim.cfd = default_cfd_settings(design, cond, device=device, precision=precision)
        cfd = sim.cfd
        quality = s.pop("quality", None)
        if quality is not None:
            q = str(quality).lower()
            if q not in M.QUALITIES:
                raise InputError(f"Unknown quality {quality!r}; use 'draft', 'standard' or 'fine'.")
            cfd.quality = q
        for key in ("duration", "average_from"):
            if s.get(key) is not None:
                setattr(cfd, key, _float(key, s.pop(key), positive=True, allow_zero=False))
            s.pop(key, None)
        cpr = s.pop("cells_per_ref", None)
        if cpr is not None:
            n = round(_float("cells_per_ref", cpr, positive=True))
            if n < 4:
                raise InputError("cells_per_ref must be at least 4.")
            cfd.cells_per_ref = n
        turb = s.pop("turbulence", None)
        if turb is not None:
            if turb not in M.TURBULENCE:
                raise InputError(f"turbulence must be one of {', '.join(M.TURBULENCE)}, got {turb!r}.")
            cfd.turbulence = turb
        mode, cap = s.pop("reynolds", None), s.pop("re_cap", None)
        if cap is not None:
            cfd.reynolds_cap = _float("re_cap", cap, positive=True, allow_zero=False)
            mode = mode or "capped"
        if mode is not None:
            if mode not in M.REYNOLDS_MODES:
                raise InputError(f"reynolds must be 'real' or 'capped', got {mode!r}.")
            cfd.reynolds_mode = mode
            if mode == "capped" and not cfd.reynolds_cap:
                from .simulation.defaults import default_reynolds_cap
                cfd.reynolds_cap = default_reynolds_cap(design)
        sens = s.pop("sensitivity", None)
        if sens:
            obj = "cd" if sens is True else str(sens).lower()
            if obj not in ("cd", "cl", "ld", "cm"):
                raise InputError(f"sensitivity must be 'cd', 'cl', 'ld' or 'cm' (or True for 'cd'), got {sens!r}.")
            cfd.compute_sensitivity, cfd.sensitivity_objective = True, obj
        sym = s.pop("symmetry", None)
        if sym is not None:
            if sym not in ("auto", "full"):
                raise InputError(f"symmetry must be 'auto' or 'full', got {sym!r}.")
            cfd.symmetry = sym
        for k, v in extra.items():
            setattr(cfd, k, v)
        ground, moving, wheels = s.pop("ground", None), s.pop("moving_ground", None), s.pop("rotate_wheels", None)
        if ground is not None:
            sim.ground.enabled = bool(ground)
        if moving is not None:
            sim.ground.moving = bool(moving)
        if wheels is not None:
            sim.ground.rotate_wheels = bool(wheels)
        _apply_physics(design, sim, {k: s.pop(k) for k in _PHYSICS_KEYS if k in s}, explicit_ground=ground)
        try:
            M.CFDSettings.from_dict(cfd.to_dict())          # re-validates choices
        except (ValueError, TypeError) as exc:
            raise InputError(str(exc)) from None
        if cfd.domain_mode == "closed" and not speed_given:
            # no free stream: the reference speed is what drives the flow (buoyancy, fans, jets)
            sim.conditions.speed = _reference_speed(design, sim)
    elif analysis == "polar":
        a0, a1, da = _angle_range((-4.0, 14.0, 1.0) if alpha is None else alpha)
        xtr = s.pop("xtr", None)
        xu = s.pop("xtr_upper", None)
        xl = s.pop("xtr_lower", None)
        sim.airfoil = M.AirfoilSettings(
            alpha_start=a0, alpha_end=a1, alpha_step=da,
            ncrit=_float("ncrit", s.pop("ncrit", 9.0), positive=True, allow_zero=False),
            xtr_upper=_float("xtr_upper", xu if xu is not None else (xtr if xtr is not None else 1.0), positive=True),
            xtr_lower=_float("xtr_lower", xl if xl is not None else (xtr if xtr is not None else 1.0), positive=True),
            mach=_float("mach", s.pop("mach", 0.0), positive=True),
            panels=int(s.pop("panels", 199)))
        if not (0 <= sim.airfoil.mach < 1):
            raise InputError("mach must be between 0 and 1.")
    else:
        a0, a1, da = _angle_range((-4.0, 12.0, 1.0) if alpha is None else alpha)
        method = str(s.pop("method", "vlm")).lower()
        if method not in M.WING_METHODS:
            raise InputError(f"method must be 'vlm' (vortex lattice) or 'llt' (lifting line), got {method!r}.")
        sim.wing = M.WingSettings(alpha_start=a0, alpha_end=a1, alpha_step=da, method=method,
                                  viscous=bool(s.pop("viscous", True)),
                                  ncrit=_float("ncrit", s.pop("ncrit", 9.0), positive=True, allow_zero=False),
                                  mach=_float("mach", s.pop("mach", 0.0), positive=True))
    if s:   # every key above is consumed; anything left is a programming error here
        raise InputError(f"Unused settings: {', '.join(sorted(s))}.")
    return sim


# ===========================================================================
# placed physics: smoke, heat, fans, jets, closed domains (DESIGN.md section 10)
# ===========================================================================

#: The model types of the placed physics, so a script can write ``al.Fan(...)``. Every
#: call that takes them also takes plain dicts of their fields (``{"diameter": 0.3}``).
Emitter = M.Emitter
Fan = M.Fan
Port = M.Port
HeatSource = M.HeatSource
WallThermal = M.WallThermal
ThermalSettings = M.ThermalSettings

#: API keywords of the placed physics (CFD only; see :meth:`Project.run_cfd`)
_PHYSICS_KEYS = ("emitters", "thermal", "walls", "heat_sources", "fans", "ports", "jets", "domain", "domain_box",
                 "diagnostics", "adaptive_dt")
#: keywords whose list value is a set of objects, never a list of values to sweep
_NOT_SWEPT = ("emitters", "heat_sources", "fans", "ports", "jets", "domain_box", "walls")
#: the features :func:`supported_physics` can report (one per kind of placed physics)
PHYSICS_FEATURES = ("emitters", "thermal", "heat_sources", "fans", "ports", "closed_domain", "adaptive_dt")
_SMOKE_COLOURS_MAX = 4          # dye channels dye, dye1..dye3 (DESIGN.md 10.1)
_WALLS_2D = ("left", "right", "bottom", "top")


def supported_physics(dimension: int | None = None) -> frozenset[str]:
    """The placed-physics features this installation's planner and solvers implement.

    Read from ``aerolume.simulation.SUPPORTED_PHYSICS`` -- a set of
    :data:`PHYSICS_FEATURES` names, or a dict ``{2: {...}, 3: {...}}`` per
    dimension -- which the simulation layer publishes for what it really runs.
    When it publishes nothing the answer is empty, and every run that asks for a
    fan, heat transfer, smoke emitters or a closed domain is refused before it
    starts (:class:`InputError`) instead of silently running without them.
    ``dimension`` None gives the features both dimensions support.
    """
    raw: Any = None
    with contextlib.suppress(Exception):         # a broken or partial simulation layer supports nothing new
        from . import simulation as S
        raw = getattr(S, "SUPPORTED_PHYSICS", None)
        if raw is None:
            from .simulation import plan as P
            raw = getattr(P, "SUPPORTED_PHYSICS", None)
    if raw is None:
        return frozenset()
    if isinstance(raw, Mapping):
        sets = {int(k): frozenset(str(x) for x in v) for k, v in raw.items()}
        if dimension is None:
            return frozenset.intersection(*sets.values()) if sets else frozenset()
        return sets.get(int(dimension), frozenset())
    return frozenset(str(x) for x in raw)


def physics_features(sim: M.Simulation) -> list[str]:
    """The placed-physics features a simulation uses (names of :data:`PHYSICS_FEATURES`), in that order."""
    if sim.analysis != "cfd":
        return []
    used = {
        "emitters": any(e.enabled for e in sim.emitters),
        "thermal": bool(sim.thermal.enabled),
        "heat_sources": any(h.enabled for h in sim.heat_sources),
        "fans": any(f.enabled for f in sim.fans),
        "ports": any(p.enabled for p in sim.ports),
        "closed_domain": sim.cfd.domain_mode == "closed",
        "adaptive_dt": sim.cfd.adaptive_dt == "pid",
    }
    return [k for k in PHYSICS_FEATURES if used[k]]


_FEATURE_WORDS = {"emitters": "smoke and particle emitters", "thermal": "heat transfer", "heat_sources": "heat sources",
                  "fans": "fans", "ports": "jets and vents", "closed_domain": "closed domains",
                  "adaptive_dt": "the PID time-step control"}


def _missing_physics(design: M.Design, sim: M.Simulation) -> list[str]:
    used = physics_features(sim)
    if not used:                # nothing new asked for: no need to load the simulation layer to ask
        return []
    have = supported_physics(int(design.dimension))
    return [f for f in used if f not in have]


def _require_physics(design: M.Design, sim: M.Simulation, label: str | None = None) -> None:
    """Refuse a run that asks for physics this build's solver would ignore (nothing is started).

    ``label`` is the name the run would be recorded under: a new simulation still
    carries the GUI's default name until it is registered, and the refusal should
    name the run the user would have seen ("CFD · 2 m/s · α 0° · 1 fan").
    """
    missing = _missing_physics(design, sim)
    if missing:
        words = ", ".join(_FEATURE_WORDS[f] for f in missing)
        label = label or sim.name or _default_name(design, sim)
        raise InputError(f"This version's {design.dimension}D solver does not implement {words} yet (DESIGN.md 10): "
                         f"“{label}” would run without {'it' if len(missing) == 1 else 'them'}, so nothing was run.")


def _vector(what: str, value: Any, dim: int, *, unit: bool = False) -> tuple[float, ...]:
    """A point or direction with ``dim`` finite coordinates (``unit`` normalises a direction).

    A 2D design also accepts three coordinates whose z is 0 (the model's default
    ``(0, 0, 0)`` position and ``(1, 0, 0)`` direction), since that is the same point.
    """
    try:
        vals = tuple(float(c) for c in value)
    except TypeError:
        raise InputError(f"{what} must be a sequence of {dim} numbers, got {value!r}.") from None
    except ValueError:
        raise InputError(f"{what} must be numbers, got {value!r}.") from None
    if dim == 2 and len(vals) == 3 and vals[2] == 0.0:
        vals = vals[:2]
    if len(vals) != dim:
        axes = "x, y" if dim == 2 else "x, y, z"
        raise InputError(f"{what} needs {dim} coordinates ({axes}) for a {dim}D design, got {len(vals)}.")
    if not all(math.isfinite(c) for c in vals):
        raise InputError(f"{what} must be finite, got {value!r}.")
    if unit:
        n = math.sqrt(sum(c * c for c in vals))
        if n == 0.0:
            raise InputError(f"{what} must not be the zero vector.")
        vals = tuple(c / n for c in vals)
    return vals


def _items(name: str, value: Any, cls: type) -> list:
    """A list of ``cls`` from objects, dicts of its fields, or one of either (None/False: empty)."""
    if value is None or value is False:
        return []
    if isinstance(value, (cls, Mapping)):
        value = [value]
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise InputError(f"{name} must be a list of {cls.__name__} objects (or dicts of their fields), "
                         f"got {type(value).__name__}.")
    names = [f.name for f in dataclasses.fields(cls)]
    out = []
    for i, item in enumerate(value):
        if isinstance(item, cls):
            out.append(copy.deepcopy(item))
            continue
        if not isinstance(item, Mapping):
            raise InputError(f"{name}[{i}] must be a {cls.__name__} or a dict, got {type(item).__name__}.")
        unknown = [k for k in item if k not in names]
        if unknown:
            import difflib
            close = [m for k in unknown for m in difflib.get_close_matches(str(k), names, n=1)]
            hint = f" Did you mean {', '.join(map(repr, close))}?" if close else ""
            raise InputError(f"{name}[{i}]: unknown field{'s' if len(unknown) > 1 else ''} "
                             f"{', '.join(map(repr, unknown))} of {cls.__name__}.{hint} Fields: {', '.join(names)}.")
        try:
            out.append(cls(**dict(item)))
        except (TypeError, ValueError) as exc:
            raise InputError(f"{name}[{i}]: {exc}") from None
    return out


def _positive(what: str, v: Any, *, allow_zero: bool = False) -> float:
    x = _float(what, v)
    if not (x > 0 or (allow_zero and x == 0)):
        raise InputError(f"{what} must be {'zero or ' if allow_zero else ''}positive, got {x:g}.")
    return x


def _check_emitters(items: list[M.Emitter], dim: int) -> None:
    colours: list[str] = []
    for e in items:
        tag = f"emitter {e.name!r}"
        e.position = _vector(f"{tag} position", e.position, dim)
        if e.kind in ("line", "rake"):
            if e.end is None:
                raise InputError(f"{tag}: a {e.kind!r} emitter needs end= (the other end of the {e.kind}).")
            e.end = _vector(f"{tag} end", e.end, dim)
            if e.end == e.position:
                raise InputError(f"{tag}: end must differ from position.")
        elif e.end is not None:
            e.end = _vector(f"{tag} end", e.end, dim)
        if e.kind in ("disc", "plane") and dim == 2:
            raise InputError(f"{tag}: a {e.kind!r} emitter is 3D; a 2D design takes point, line, rake or rect.")
        if e.kind == "rect":
            if len(e.size) != dim or not all(c > 0 for c in e.size):
                raise InputError(f"{tag}: a 'rect' emitter needs size= with {dim} positive extents [m].")
        elif e.kind == "disc":
            if not e.size or not e.size[0] > 0:
                raise InputError(f"{tag}: a 'disc' emitter needs size=(radius,) with a positive radius [m].")
        elif e.kind == "plane" and (len(e.size) != 2 or not all(c > 0 for c in e.size)):
            raise InputError(f"{tag}: a 'plane' emitter needs size=(width, height) [m].")
        if e.normal is not None:
            e.normal = _vector(f"{tag} normal", e.normal, dim, unit=True)
        if e.kind == "rake" and int(e.count) < 1:
            raise InputError(f"{tag}: count must be at least 1.")
        _positive(f"{tag} radius", e.radius, allow_zero=True)
        _positive(f"{tag} rate", e.rate, allow_zero=True)
        _positive(f"{tag} pulse_period", e.pulse_period, allow_zero=True)
        if e.pulse_period > 0 and not 0 < float(e.pulse_duty) <= 1:
            raise InputError(f"{tag}: pulse_duty must be in (0, 1], got {e.pulse_duty:g}.")
        _positive(f"{tag} start", e.start, allow_zero=True)
        if e.stop is not None and not float(e.stop) > float(e.start):
            raise InputError(f"{tag}: stop must be later than start.")
        if not (isinstance(e.color, str) and len(e.color) == 7 and e.color.startswith("#")
                and all(c in "0123456789abcdefABCDEF" for c in e.color[1:])):
            raise InputError(f"{tag}: color must be '#rrggbb', got {e.color!r}.")
        if e.species == "dye" and e.enabled and e.color.lower() not in colours:
            colours.append(e.color.lower())
    if len(colours) > _SMOKE_COLOURS_MAX:
        raise InputError(f"At most {_SMOKE_COLOURS_MAX} smoke colours (dye channels); got {len(colours)}. Emitters "
                         "of one colour share a channel.")


def _check_fans(items: list[M.Fan], dim: int) -> None:
    for f in items:
        tag = f"fan {f.name!r}"
        f.position = _vector(f"{tag} position", f.position, dim)
        f.direction = _vector(f"{tag} direction", f.direction, dim, unit=True)
        f.diameter = _positive(f"{tag} diameter", f.diameter)
        if f.thickness is not None:
            f.thickness = _positive(f"{tag} thickness", f.thickness)
        v = _float(f"{tag} value", f.value)
        if f.mode == "velocity" and not v > 0:
            raise InputError(f"{tag}: a 'velocity' fan needs a positive target speed [m/s], got {v:g}.")
        if f.mode == "thrust" and v == 0:
            raise InputError(f"{tag}: a 'thrust' fan with zero thrust does nothing; give value= in N "
                             f"({'N per metre of span in 2D' if dim == 2 else 'N'}).")
        if dim == 2 and float(f.swirl or 0.0) != 0.0:
            raise InputError(f"{tag}: swirl needs a 3D design (a 2D actuator line has no tangential direction).")


def _check_ports(items: list[M.Port], dim: int, thermal_on: bool) -> None:
    for p in items:
        tag = f"{p.kind} {p.name!r}"
        p.position = _vector(f"{tag} position", p.position, dim)
        p.direction = _vector(f"{tag} direction", p.direction, dim, unit=True)
        p.size = _positive(f"{tag} size", p.size)
        p.speed = _positive(f"{tag} speed", p.speed)
        if p.temperature is not None:
            t = _float(f"{tag} temperature", p.temperature)
            if not thermal_on:
                raise InputError(f"{tag}: a jet temperature needs heat transfer: pass thermal=True "
                                 "(or thermal={'buoyancy': False}).")
            if p.kind == "vent":
                raise InputError(f"{tag}: a vent takes fluid out; only a jet carries a temperature.")
            p.temperature = t


def _check_heat_sources(items: list[M.HeatSource], dim: int) -> None:
    kinds = ("rect", "disc") if dim == 2 else ("box", "sphere")
    for h in items:
        tag = f"heat source {h.name!r}"
        if h.kind not in kinds:
            raise InputError(f"{tag}: a {dim}D heat source is {' or '.join(map(repr, kinds))}, got {h.kind!r}.")
        h.position = _vector(f"{tag} position", h.position, dim)
        if h.kind in ("rect", "box"):
            size = h.size[:dim] if len(h.size) >= dim else h.size
            if len(size) != dim or not all(c > 0 for c in size):
                raise InputError(f"{tag}: a {h.kind!r} needs size= with {dim} positive extents [m].")
            h.size = tuple(size)
        elif not h.size or not h.size[0] > 0:
            raise InputError(f"{tag}: a {h.kind!r} needs size=(radius,) with a positive radius [m].")
        _float(f"{tag} power", h.power)


def _walls(value: Any, dim: int) -> list[M.WallThermal]:
    """Wall conditions from ``{"left": 30.0, "right": ("temperature", 10), "top": "adiabatic",
    "bottom": ("heat_flux", 50)}`` or a list of :class:`WallThermal` (numbers are °C)."""
    if value is None:
        return []
    sides = _WALLS_2D if dim == 2 else M.WALL_SIDES
    out: dict[str, M.WallThermal] = {}
    items: list[tuple[str, Any]]
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, M.WallThermal):
        items = [(value.side, value)]
    else:
        try:
            items = [(w.side if isinstance(w, M.WallThermal) else dict(w).get("side"), w) for w in value]
        except (TypeError, ValueError):
            raise InputError("walls must be a dict {side: condition} or a list of WallThermal.") from None
    for side, spec in items:
        if side not in sides:
            raise InputError(f"Unknown wall side {side!r}; a {dim}D domain has {', '.join(sides)}.")
        if isinstance(spec, M.WallThermal):
            w = copy.deepcopy(spec)
        elif isinstance(spec, Mapping):
            try:
                w = M.WallThermal(side=side, mode=spec.get("mode", "temperature"), value=float(spec.get("value", 0)))
            except (TypeError, ValueError) as exc:
                raise InputError(f"wall {side!r}: {exc}") from None
        elif isinstance(spec, str):
            if spec != "adiabatic":
                raise InputError(f"wall {side!r}: a text condition can only be 'adiabatic'; give a temperature in °C "
                                 "or ('heat_flux', W/m²).")
            w = M.WallThermal(side=side)
        elif isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
            if spec[0] not in M.THERMAL_MODES:
                raise InputError(f"wall {side!r}: mode must be one of {', '.join(M.THERMAL_MODES)}, got {spec[0]!r}.")
            w = M.WallThermal(side=side, mode=spec[0], value=_float(f"wall {side!r} value", spec[1]))
        else:
            w = M.WallThermal(side=side, mode="temperature", value=_float(f"wall {side!r} temperature", spec))
        if w.mode == "temperature" and not w.value > -273.15:
            raise InputError(f"wall {side!r}: {w.value:g} °C is below absolute zero.")
        out[side] = w
    return list(out.values())


def _given_ambient(value: Any) -> float | None:
    """The ambient temperature [°C] a ``thermal=`` argument states itself, or None when it leaves it to the default.

    A dict states it with ``t_ambient`` (or ``ambient``); a ThermalSettings object
    always carries one, and the caller who built it chose it. ``True`` or a dict
    without it does not. A value that is no temperature is left to :func:`_thermal`,
    which refuses it with the field's own message.
    """
    if isinstance(value, M.ThermalSettings):
        raw: Any = value.t_ambient
    elif isinstance(value, Mapping) and ("t_ambient" in value or "ambient" in value):
        raw = value.get("t_ambient", value.get("ambient"))
    else:
        return None
    try:
        t = float(raw)
    except (TypeError, ValueError):
        return None
    return t if math.isfinite(t) and t > -273.15 else None


def _thermal(value: Any, walls: Any, dim: int, fluid: M.Fluid | None = None) -> M.ThermalSettings:
    """ThermalSettings from True / a dict of its fields (``ambient`` aliases ``t_ambient``) / the object.

    Without a stated ambient temperature it defaults to the one ``fluid`` was made
    at (:func:`aerolume.simulation.physics.default_ambient`: 15 °C for ISA sea-level
    air). The model's own default is 20 °C, and every API thermal run (room, heated
    sphere, heated box) began with the warning that the fluid is air at 15 °C but the
    ambient is 20 °C: a mismatch the caller never asked for (critic round 3.3).
    """
    if value is None and walls is None:
        return M.ThermalSettings()
    if value is False:
        if walls:
            raise InputError("walls= set wall temperatures or heat fluxes, which need heat transfer: drop "
                             "thermal=False or the walls.")
        return M.ThermalSettings()
    if isinstance(value, M.ThermalSettings):
        th = copy.deepcopy(value)
    elif value is True or value is None:
        th = M.ThermalSettings()
    elif isinstance(value, Mapping):
        d = dict(value)
        if "ambient" in d:
            d["t_ambient"] = d.pop("ambient")
        w = d.pop("walls", None)
        names = [f.name for f in dataclasses.fields(M.ThermalSettings)]
        unknown = [k for k in d if k not in names]
        if unknown:
            raise InputError(f"thermal: unknown field{'s' if len(unknown) > 1 else ''} {', '.join(map(repr, unknown))}; "
                             f"fields: {', '.join(n for n in names if n != 'walls')}, walls, ambient.")
        try:
            th = M.ThermalSettings(**d)
        except (TypeError, ValueError) as exc:
            raise InputError(f"thermal: {exc}") from None
        th.walls = _walls(w, dim)
    else:
        raise InputError(f"thermal must be True, False, a dict or a ThermalSettings, got {type(value).__name__}.")
    th.enabled = True
    th.walls = _walls(list(th.walls), dim) if th.walls else []
    if walls is not None:
        merged = {w.side: w for w in th.walls}
        merged.update({w.side: w for w in _walls(walls, dim)})
        th.walls = list(merged.values())
    stated = isinstance(value, M.ThermalSettings) or (
        isinstance(value, Mapping) and ("t_ambient" in value or "ambient" in value))
    if not stated and fluid is not None:
        from .simulation.physics import default_ambient
        th.t_ambient = default_ambient(fluid, fallback=th.t_ambient)
    th.t_ambient = _float("thermal t_ambient", th.t_ambient)
    if not th.t_ambient > -273.15:
        raise InputError(f"thermal t_ambient {th.t_ambient:g} °C is below absolute zero.")
    th.gravity = _positive("thermal gravity", th.gravity, allow_zero=True)
    for k in ("beta", "prandtl", "conductivity", "cp"):
        if getattr(th, k) is not None:
            setattr(th, k, _positive(f"thermal {k}", getattr(th, k)))
    th.buoyancy = bool(th.buoyancy)
    return th


def _box(value: Any, dim: int) -> tuple[float, ...]:
    try:
        b = tuple(float(c) for c in value)
    except (TypeError, ValueError):
        raise InputError(f"domain_box must be {2 * dim} numbers, got {value!r}.") from None
    if len(b) != 2 * dim:
        axes = "(xmin, xmax, ymin, ymax)" if dim == 2 else "(xmin, xmax, ymin, ymax, zmin, zmax)"
        raise InputError(f"domain_box of a {dim}D design is {axes} in metres, got {len(b)} numbers.")
    for i in range(dim):
        if not (math.isfinite(b[2 * i]) and math.isfinite(b[2 * i + 1]) and b[2 * i + 1] > b[2 * i]):
            raise InputError(f"domain_box: each max must exceed its min, got {b!r}.")
    return b


def _inside(box: tuple[float, ...], p: Sequence[float]) -> bool:
    return all(box[2 * i] - 1e-12 <= float(c) <= box[2 * i + 1] + 1e-12 for i, c in enumerate(p))


def _heated_bodies(design: M.Design) -> list[str]:
    parts = list(design.bodies) + list(design.meshes)
    return [b.name for b in parts if getattr(b, "thermal", "adiabatic") != "adiabatic"]


def _check_in_box(sim: M.Simulation) -> None:
    """Refuse placed physics outside the simulation's domain box (when it has one): it would act on nothing."""
    box = sim.cfd.domain_box
    if box is None:
        return
    for kind, objs in (("emitter", sim.emitters), ("fan", sim.fans), ("heat source", sim.heat_sources),
                       ("port", sim.ports)):
        for o in objs:
            if not _inside(box, o.position):
                raise InputError(f"The {kind} {o.name!r} at {tuple(round(c, 6) for c in o.position)} m lies "
                                 f"outside the domain box {box}.")


def _apply_physics(design: M.Design, sim: M.Simulation, kw: dict[str, Any], *,
                   explicit_ground: bool | None = None) -> None:
    """Put the placed-physics keywords on ``sim`` after checking every one (DESIGN.md 10).

    Everything is validated here, before a run is recorded: positions have the
    design's dimension and lie inside a given domain box, directions are
    normalised, sizes are positive, a vent carries no temperature, a closed
    domain has no moving road, and at most four smoke colours are used. The
    model's own enumerations (emitter kinds, fan modes, wall sides) are checked
    by the model types, so the API and the GUI accept the same values.
    """
    dim = int(design.dimension)
    cfd = sim.cfd
    sim.thermal = _thermal(kw.get("thermal"), kw.get("walls"), dim, sim.conditions.fluid)
    sim.emitters = _items("emitters", kw.get("emitters"), M.Emitter)
    sim.heat_sources = _items("heat_sources", kw.get("heat_sources"), M.HeatSource)
    sim.fans = _items("fans", kw.get("fans"), M.Fan)
    ports = kw.get("ports")
    jets = kw.get("jets")
    if ports is not None and jets is not None:
        raise InputError("Give jets and vents in one list, ports= (jets= is the same keyword).")
    sim.ports = _items("ports", ports if ports is not None else jets, M.Port)
    _check_emitters(sim.emitters, dim)
    _check_fans(sim.fans, dim)
    _check_heat_sources(sim.heat_sources, dim)
    _check_ports(sim.ports, dim, sim.thermal.enabled)
    if sim.heat_sources and not sim.thermal.enabled:
        raise InputError("heat_sources release heat, which needs heat transfer: pass thermal=True "
                         "(buoyancy on by default) or thermal={'buoyancy': False}.")
    heated = _heated_bodies(design)
    if heated and not sim.thermal.enabled:
        raise InputError(f"{design.name!r} has heated bodies ({', '.join(heated[:3])}), but heat transfer is off: "
                         "pass thermal=True (natural convection with buoyancy) or thermal={'buoyancy': False} "
                         "(forced convection), or make the bodies adiabatic.")
    if kw.get("domain") is not None:
        mode = str(kw["domain"]).lower()
        if mode not in M.DOMAIN_MODES:
            raise InputError(f"domain must be 'tunnel' (free stream from the left) or 'closed' (walls on every "
                             f"side), got {kw['domain']!r}.")
        cfd.domain_mode = mode
    if kw.get("domain_box") is not None:
        cfd.domain_box = _box(kw["domain_box"], dim)
    if cfd.domain_mode == "closed":
        if explicit_ground:
            raise InputError("A closed domain has walls on every side; its floor is a wall, not a moving road: "
                             "drop ground=True.")
        sim.ground.enabled = False
    elif sim.thermal.walls and any(w.side in ("left", "right") for w in sim.thermal.walls):
        raise InputError("In a wind tunnel the left side is the inlet (at t_ambient) and the right side the outlet: "
                         "only the bottom, top (and 3D front/back) walls take thermal conditions. Use "
                         "domain='closed' for a cavity heated from its sides.")
    _check_in_box(sim)
    if kw.get("diagnostics") is not None:
        cfd.record_diagnostics = bool(kw["diagnostics"])
    if kw.get("adaptive_dt") is not None:
        mode = str(kw["adaptive_dt"]).lower()
        if mode not in ("cfl", "pid"):
            raise InputError(f"adaptive_dt must be 'cfl' (fixed CFL target) or 'pid' (PID-controlled), got "
                             f"{kw['adaptive_dt']!r}.")
        cfd.adaptive_dt = mode


def _delta_t(design: M.Design, sim: M.Simulation) -> float:
    """The temperature difference that drives buoyancy [K] (DESIGN.md 10.2).

    Set temperatures (walls, bodies, hot jets) give it directly. Heat given as a
    power or a flux -- a volumetric heater, a heat-flux body or wall -- has no
    temperature of its own: its scale is the rise it produces, q″L/(k·Nu), which
    :func:`aerolume.simulation.physics.thermal_estimate` computes exactly as the
    planner does ("estimated rise at the heater"). Without that term a closed room
    heated only by a 300 W heater was refused as "nothing drives the flow".
    """
    th = sim.thermal
    amb = float(th.t_ambient)
    diffs = [abs(w.value - amb) for w in th.walls if w.mode == "temperature"]
    diffs += [abs(float(b.thermal_value) - amb) for b in list(design.bodies) + list(design.meshes)
              if getattr(b, "thermal", "") == "temperature"]
    diffs += [abs(float(p.temperature) - amb) for p in sim.ports if p.temperature is not None]
    temps = [w.value for w in th.walls if w.mode == "temperature"]
    if len(temps) >= 2:
        diffs.append(max(temps) - min(temps))
    fluxed = (any(h.enabled and h.power for h in sim.heat_sources)
              or any(w.mode == "heat_flux" and w.value for w in th.walls)
              or any(getattr(b, "thermal", "") == "heat_flux" and getattr(b, "thermal_value", 0.0)
                     for b in list(design.bodies) + list(design.meshes)))
    if th.enabled and fluxed:
        from .simulation.physics import thermal_estimate
        est = thermal_estimate(design, sim)
        if est is not None and str(est.delta_t_basis).startswith("estimated rise") and math.isfinite(est.delta_t):
            diffs.append(float(est.delta_t))
    return max(diffs, default=0.0)


def _fluid_beta(sim: M.Simulation) -> float:
    th = sim.thermal
    if th.beta:
        return float(th.beta)
    if "water" in sim.conditions.fluid.name.lower():
        return 2.07e-4                       # water at 20 °C [1/K]
    return 1.0 / (float(th.t_ambient) + 273.15)   # ideal gas


def _reference_speed(design: M.Design, sim: M.Simulation) -> float:
    """The speed that characterises a closed-domain flow, which has no free stream [m/s].

    Buoyancy: the planner's natural-convection velocity (free fall sqrt(g beta dT H)
    over the box height, or kappa / H below Ra 10^3; sqrt(g beta dT L) when it has
    no estimate), with dT from :func:`_delta_t`; a thrust fan: the
    momentum-theory disc velocity sqrt(T / (2 rho A)) of a fan in still air
    (A per metre of span in 2D); a velocity fan or a jet: its speed. The
    largest one names the simulation and sets its Reynolds number; the planner
    states the scales it actually uses.
    """
    rho = float(sim.conditions.fluid.density)
    L = float(design.length_ref) or 1.0
    cand: list[float] = []
    th = sim.thermal
    if th.enabled and th.buoyancy and th.gravity > 0:
        dT = _delta_t(design, sim)
        if dT > 0:
            # the planner's own natural-convection scale when it has one (free fall over the box height, or
            # kappa/H below Ra 10^3), so the run's reference speed and the plan's "Reference velocity" agree
            from .simulation.physics import thermal_estimate
            est = thermal_estimate(design, sim)
            if est is not None and est.natural and math.isfinite(est.speed) and est.speed > 0:
                cand.append(float(est.speed))
            else:
                cand.append(math.sqrt(th.gravity * _fluid_beta(sim) * dT * L))
    for f in sim.fans:
        if not f.enabled:
            continue
        if f.mode == "velocity":
            cand.append(abs(float(f.value)))
        else:
            area = math.pi * f.diameter ** 2 / 4.0 if design.dimension == 3 else f.diameter
            dp = abs(float(f.value)) if f.mode == "pressure_jump" else abs(float(f.value)) / area
            cand.append(math.sqrt(dp / (2.0 * rho)))
    cand += [float(p.speed) for p in sim.ports if p.enabled]
    cand = [c for c in cand if math.isfinite(c) and c > 0]
    if not cand:
        raise InputError("A closed domain has no free stream: something must drive the flow -- heat with "
                         "buoyancy (a heated wall or body, or a heater), a fan or a jet -- or give speed= as the "
                         "reference speed.")
    return max(cand)


def _physics_tags(design: M.Design, sim: M.Simulation) -> list[str]:
    """Words appended to a simulation's name for its placed physics ("closed", "heated", "2 fans")."""
    if sim.analysis != "cfd":
        return []
    tags: list[str] = []
    if sim.cfd.domain_mode == "closed":
        tags.append("closed")
    if sim.thermal.enabled:
        tags.append("heated" if sim.thermal.buoyancy else "heat transfer")
    n = sum(1 for f in sim.fans if f.enabled)
    if n:
        tags.append(_count(n, "fan"))
    n = sum(1 for p in sim.ports if p.enabled)
    if n:
        tags.append(_count(n, "jet" if all(p.kind == "jet" for p in sim.ports) else "port"))
    if any(e.enabled for e in sim.emitters):
        tags.append("smoke" if all(e.species == "dye" for e in sim.emitters) else "tracers")
    return tags


def _default_name(design: M.Design, sim: M.Simulation, *, by_reynolds: bool | None = None) -> str:
    """The GUI's simulation name (``simulation.defaults.simulation_name``) plus the run's physics.

    "CFD · 30 m/s · α 4°", "Polar · Re 1.0×10⁶", "CFD 3D · Re 100 · α 0°" (a run set by its
    Reynolds number is named by it), then "· closed · heated · 2 fans" for placed physics, so
    two runs of one design that differ only in a fan never share a name.
    """
    from .simulation.defaults import simulation_name
    if by_reynolds is None and getattr(sim, "_named_by_re", False):
        by_reynolds = True
    base = simulation_name(design, sim, by_reynolds=by_reynolds)
    tags = [t for t in _physics_tags(design, sim) if t not in base]
    return " · ".join([base, *tags])


# ===========================================================================
# results
# ===========================================================================

_COEFF_LABELS = {"cd": "Cd", "cl": "Cl", "cm": "Cm", "cs": "Cs", "c_roll": "C_roll", "cn": "Cn"}
_COEFF_LABELS_3D = {"cd": "CD", "cs": "CS", "cl": "CL", "c_roll": "C_roll", "cm": "Cm", "cn": "Cn"}
# coefficients and forces that are zero by construction on a half-domain (symmetry-plane) run
_ANTISYMMETRIC = ("cs", "c_roll", "cn")
_ANTISYMMETRIC_FORCES = ("side", "roll", "yaw")
_LOWER_BETTER = {"cd", "drag", "drag_area", "power", "cdi", "cdv", "cd_min", "cd0"}
_PLOT_KINDS = {
    "cfd2d": ("history", "cp", "field", "geometry", "sensitivity", "diagnostics"),
    "cfd3d": ("history", "surface", "streamlines", "smoke", "vortices", "slice", "geometry", "diagnostics"),
    "polar": ("polar", "cp"),
    "wing": ("polar", "span", "planform"),
}


#: units of the keys a fan report may carry (DESIGN.md 10.3: thrust, volume flow, induced velocity, power)
_FAN_UNITS_3D = {"thrust": "N", "volume_flow": "m³/s", "flow": "m³/s", "flow_rate": "m³/s",
                 "induced_velocity": "m/s", "disc_velocity": "m/s", "velocity": "m/s", "power": "W",
                 "ideal_power": "W", "pressure_jump": "Pa", "delta_p": "Pa", "diameter": "m"}
_FAN_UNITS_2D = {"thrust": "N/m", "volume_flow": "m²/s", "flow": "m²/s", "flow_rate": "m²/s",
                 "induced_velocity": "m/s", "disc_velocity": "m/s", "velocity": "m/s", "power": "W/m",
                 "ideal_power": "W/m", "pressure_jump": "Pa", "delta_p": "Pa", "diameter": "m"}


def _fan_unit(key: str, mode: str, units: Mapping[str, str]) -> str:
    """The unit of one fan-report value; the set ``target`` takes the unit of the fan's mode."""
    if key == "target":
        return {"thrust": units["thrust"], "pressure_jump": "Pa", "velocity": "m/s"}.get(mode, "")
    return units.get(key, "")


class Result:
    """A stored result with the numbers people ask for first.

    CFD results (2D and 3D) are one operating point: ``cd``, ``cl``, ``cm``
    (3D also ``cs``, ``c_roll``, ``cn``) are the time-averaged coefficients,
    ``stats[name]`` their :class:`~aerolume.model.Stat` (``stderr`` is the
    standard error of the mean), ``ld`` = Cl / Cd, ``forces`` the dimensional
    forces and moments (2D per metre of span), ``converged`` the runner's
    statistical verdict, ``re_sim`` / ``re_real`` the simulated and real
    Reynolds numbers. Polars and wing sweeps are curves: ``alpha``, ``cl``,
    ``cd``, ``cm`` and ``ld`` are arrays over the converged angles, ``metrics``
    the figures of merit (Cl,max, best L/D, lift slope, ...), :meth:`best` the
    point of best L/D and :meth:`at` an interpolated point.

    ``warnings`` and ``notes`` are the solver's (always read the warnings);
    ``raw`` is the model object (``CFDResult``, ``CFD3DResult``, ``PolarResult``
    or ``WingResult``), ``simulation`` the settings it ran with, ``reused`` True
    when a stored result was returned instead of running again, ``cancelled``
    True for a partial run.
    """

    def __init__(self, project: Project | None, simulation: M.Simulation, raw: Any, *, reused: bool = False) -> None:
        self.project = project
        self.simulation = simulation
        self.raw = raw
        self.reused = reused

    # -- identity --------------------------------------------------------------
    @property
    def sim_id(self) -> str:
        return self.simulation.id

    @property
    def name(self) -> str:
        return self.simulation.name

    @property
    def analysis(self) -> str:
        return self.simulation.analysis

    @property
    def kind(self) -> str:
        """"cfd2d", "cfd3d", "polar" or "wing"."""
        if isinstance(self.raw, M.CFD3DResult):
            return "cfd3d"
        if isinstance(self.raw, M.CFDResult):
            return "cfd2d"
        return "polar" if isinstance(self.raw, M.PolarResult) else "wing"

    @property
    def dimension(self) -> int:
        return 2 if self.kind in ("cfd2d", "polar") else 3

    @property
    def design(self) -> Design | None:
        if self.project is None:
            return None
        d = self.project.raw.designs.get(self.simulation.design_id)
        return Design(self.project, d) if d is not None else None

    @property
    def status(self) -> str:
        return self.simulation.status

    @property
    def cancelled(self) -> bool:
        return self.simulation.status == "cancelled"

    @property
    def conditions(self) -> dict[str, Any]:
        c = self.simulation.conditions
        return {"speed": c.speed, "alpha": c.alpha_deg, "yaw": c.yaw_deg, "fluid": c.fluid.name,
                "density": c.fluid.density, "viscosity": c.fluid.viscosity}

    @property
    def warnings(self) -> list[str]:
        return list(getattr(self.raw, "warnings", []) or [])

    @property
    def notes(self) -> list[str]:
        return list(getattr(self.raw, "notes", []) or [])

    @property
    def wall_seconds(self) -> float:
        return float(getattr(self.raw, "wall_seconds", math.nan))

    # -- coefficients ----------------------------------------------------------
    def _is_cfd(self) -> bool:
        return self.kind in ("cfd2d", "cfd3d")

    @property
    def lifting(self) -> bool:
        """Whether L/D means something for this result: airfoils and wings, not cars or bluff bodies.

        A sphere's "L/D 0.014" or a sedan's 4.0 is arithmetic nobody optimises, so the
        tables, :meth:`summary` and sweep tables leave it out for non-lifting designs
        (the Results and Compare pages' rule); :attr:`ld` still computes it on request.
        """
        if self.kind in ("polar", "wing"):
            return True
        d = self.project.raw.designs.get(self.simulation.design_id) if self.project is not None else None
        return d is None or d.kind in ("airfoil", "wing")

    def _by_symmetry(self, key: str) -> bool:
        """Is ``key`` zero only because a half domain was simulated (DESIGN.md 7.13)?

        On a symmetry-plane run the side force, rolling and yawing moments vanish by
        construction; printing "0  0" for them reads like a measured zero with a zero
        error, so the text says "0 (symmetry)", as the report does.
        """
        if self.kind != "cfd3d" or not self.raw.symmetry_plane:
            return False
        if key in _ANTISYMMETRIC:
            st = self.stats.get(key)
            return st is not None and not (st.mean or st.rms)
        if key in _ANTISYMMETRIC_FORCES:
            return not self.forces.get(key)
        return False

    @property
    def stats(self) -> dict[str, M.Stat]:
        """Coefficient statistics of a CFD result (mean, rms, min, max, stderr); {} for curves."""
        if not self._is_cfd():
            return {}
        keys = ("cd", "cs", "cl", "c_roll", "cm", "cn") if self.kind == "cfd3d" else ("cd", "cl", "cm")
        return {k: getattr(self.raw, k) for k in keys}

    def _curve(self, key: str) -> np.ndarray:
        tab = self.raw.table()
        if key == "ld":
            cd = tab["cd"]
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(cd > 0, tab["cl"] / cd, np.nan)
        return np.asarray(tab[key], float)

    def _coef(self, key: str) -> Any:
        if self._is_cfd():
            st = self.stats.get(key)
            return float(st.mean) if st is not None else math.nan
        return self._curve(key)

    @property
    def cd(self) -> Any:
        return self._coef("cd")

    @property
    def cl(self) -> Any:
        return self._coef("cl")

    @property
    def cm(self) -> Any:
        return self._coef("cm")

    @property
    def cs(self) -> float:
        return self._coef("cs") if self.kind == "cfd3d" else math.nan

    @property
    def c_roll(self) -> float:
        return self._coef("c_roll") if self.kind == "cfd3d" else math.nan

    @property
    def cn(self) -> float:
        return self._coef("cn") if self.kind == "cfd3d" else math.nan

    @property
    def ld(self) -> Any:
        """Lift-to-drag ratio (array over the angles for curves)."""
        if self._is_cfd():
            cd, cl = self.cd, self.cl
            return cl / cd if cd and _finite(cd) and _finite(cl) else math.nan
        return self._curve("ld")

    @property
    def alpha(self) -> Any:
        """The angle of attack [deg]: one number for CFD, the converged angles for curves."""
        if self._is_cfd():
            return float(self.simulation.conditions.alpha_deg)
        return self._curve("alpha")

    def stderr(self, key: str) -> float:
        """Standard error of a CFD coefficient's mean (nan for curves or without one)."""
        st = self.stats.get(key)
        return float(st.stderr) if st is not None else math.nan

    @property
    def forces(self) -> dict[str, float]:
        """Dimensional mean forces and moments: 2D N/m, N·m/m; 3D N, N·m ({} for curves)."""
        r = self.raw
        if self.kind == "cfd2d":
            return {"drag": r.drag, "lift": r.lift, "moment": r.moment, "drag_area": r.drag_area, "power": r.power}
        if self.kind == "cfd3d":
            mx, my, mz = r.moments
            return {"drag": r.drag, "side": r.side, "lift": r.lift, "roll": mx, "pitch": my, "yaw": mz,
                    "drag_area": r.drag_area, "power": r.power}
        return {}

    @property
    def units(self) -> dict[str, str]:
        """Unit of every key in :attr:`forces` and :meth:`summary`."""
        if self.kind == "cfd2d":
            return {"drag": "N/m", "lift": "N/m", "moment": "N·m/m", "drag_area": "m²/m", "power": "W/m",
                    "speed": "m/s", "alpha": "°", "wall_seconds": "s", "heat_rate": "W/m"}
        return {"drag": "N", "side": "N", "lift": "N", "roll": "N·m", "pitch": "N·m", "yaw": "N·m",
                "drag_area": "m²", "power": "W", "speed": "m/s", "alpha": "°", "wall_seconds": "s",
                "heat_rate": "W"}

    # -- placed physics (DESIGN.md 10) --------------------------------------------
    @property
    def physics(self) -> list[str]:
        """The placed-physics features the run used ("thermal", "fans", "closed_domain", ...)."""
        return physics_features(self.simulation)

    @property
    def nusselt_stat(self) -> M.Stat | None:
        """Statistics of the mean Nusselt number of the heated bodies or walls (None without heat transfer)."""
        st = getattr(self.raw, "nusselt", None) if self._is_cfd() else None
        return st if st is not None and _finite(getattr(st, "mean", math.nan)) else None

    @property
    def nusselt(self) -> float:
        """Mean Nusselt number Nu = Q L / (k ΔT A) of the heated surfaces (nan without heat transfer)."""
        st = self.nusselt_stat
        return float(st.mean) if st is not None else math.nan

    @property
    def heat_rate(self) -> float:
        """Heat flowing from the bodies into the fluid: W in 3D, W per metre of span in 2D (nan when unused)."""
        return float(getattr(self.raw, "heat_rate", math.nan)) if self._is_cfd() else math.nan

    @property
    def fans(self) -> list[dict[str, Any]]:
        """Per fan what it delivered: thrust, volume flow, mean induced velocity, ideal power (the solver's keys)."""
        return [dict(f) for f in (getattr(self.raw, "fans_report", None) or [])] if self._is_cfd() else []

    @property
    def diagnostics(self) -> dict[str, float]:
        """The solver's end-of-run diagnostics: RMS divergence, change norms, CFL, residuals ({} when not recorded)."""
        return dict(getattr(self.raw, "diagnostics", None) or {}) if self._is_cfd() else {}

    def reference_check(self) -> dict[str, Any] | None:
        """This result against the published reference its set-up reproduces, or None when it has none.

        The run's design and settings are matched to a reference case
        (:func:`reference_case`: the de Vahl Davis cavity, the Lange heated
        cylinder in forced convection, the fan room's consistency check), and
        :meth:`Case.check` compares the result with it. When the design is a
        reference set-up whose settings no longer match the published conditions,
        the answer is ``{"applies": False, "why": ...}`` -- never a comparison
        under conditions the reference was not published for.
        """
        d = self.design
        if d is None or not self._is_cfd():
            return None
        case, why = reference_case(d.raw, self.simulation, re=getattr(self.raw, "re_sim", None))
        if case is None:
            return {"applies": False, "why": why} if why else None
        out = {"applies": True, **case.check(self)}
        if case.notes:
            out["notes"] = list(case.notes)
        return out

    def diagnostics_history(self) -> dict[str, np.ndarray]:
        """The recorded diagnostic series against time ("t" plus every history series that is not a coefficient).

        The coefficient histories (cd, cl, cm, ...) are what the "history" plot shows;
        what remains in the run's history -- divergence, change norms, CFL, dt,
        pressure residuals, when the run recorded them -- is the solver's own health.
        """
        if not self._is_cfd():
            return {}
        hist = dict(getattr(self.raw, "history", None) or {})
        coeffs = {"t", "cd", "cl", "cm", "cs", "c_roll", "cn"}
        out = {k: np.asarray(v, float) for k, v in hist.items() if k not in coeffs and np.ndim(v) == 1}
        if not out:
            return {}
        t = np.asarray(hist.get("t", np.arange(max(len(v) for v in out.values()))), float)
        return {"t": t, **{k: v for k, v in out.items() if len(v) == len(t)}}

    @property
    def re_sim(self) -> float:
        """The Reynolds number that was simulated (capped CFD runs: below the real one)."""
        r = self.raw
        if self._is_cfd():
            return float(r.re_sim)
        return float(r.re if self.kind == "polar" else r.re_mac)

    @property
    def re_real(self) -> float:
        r = self.raw
        if self._is_cfd():
            return float(r.re_real)
        return self.re_sim

    @property
    def converged(self) -> bool:
        """CFD: the runner's statistical verdict. Curves: every angle converged."""
        if self._is_cfd():
            return bool(self.raw.converged)
        return bool(self.raw.points) and all(p.converged for p in self.raw.points)

    @property
    def n_converged(self) -> int:
        if self._is_cfd():
            return int(self.converged)
        return sum(1 for p in self.raw.points if p.converged)

    @property
    def metrics(self) -> dict[str, float]:
        """Figures of merit of a curve (``analysis.polar_metrics`` / ``wing_metrics``); {} for CFD."""
        if self.kind == "polar":
            from .analysis import polar_metrics
            return dict(polar_metrics(self.raw))
        if self.kind == "wing":
            from .analysis import wing_metrics
            return dict(wing_metrics(self.raw))
        return {}

    @_api
    def best(self) -> dict[str, float]:
        """The converged point of best L/D of a polar or wing sweep, as a dict."""
        if self._is_cfd():
            raise InputError("best() is for polars and wing sweeps; a CFD result is one point (see ld).")
        rows = [r for r in self.table() if r["converged"] and _finite(r["ld"]) and r["cd"] > 0]
        if not rows:
            raise RunError(f"No converged point with positive drag in {self.name!r}.")
        return max(rows, key=lambda r: r["ld"])

    @_api
    def at(self, alpha: float) -> dict[str, float]:
        """A polar or wing sweep interpolated (linearly, converged points) at ``alpha`` degrees."""
        if self._is_cfd():
            raise InputError("at() is for polars and wing sweeps.")
        a = self._curve("alpha")
        if a.size == 0:
            raise RunError(f"{self.name!r} has no converged points.")
        x = _float("alpha", alpha)
        if x < a.min() - 1e-9 or x > a.max() + 1e-9:
            raise InputError(f"alpha {x:g}° is outside the converged range {a.min():g}° to {a.max():g}°.")
        tab = self.raw.table()
        out = {"alpha": x}
        for k, v in tab.items():
            if k != "alpha":
                out[k] = float(np.interp(x, a, np.asarray(v, float)))
        out["ld"] = out["cl"] / out["cd"] if out.get("cd", 0) > 0 else math.nan
        return out

    # -- tables ----------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """The key facts as a flat dict (what a results table or an assistant reports)."""
        c = self.simulation.conditions
        out: dict[str, Any] = {"name": self.name, "sim_id": self.sim_id, "analysis": self.analysis,
                               "kind": self.kind, "design": getattr(self.design, "name", ""),
                               "speed": c.speed, "alpha": c.alpha_deg if self._is_cfd() else None}
        if self.kind == "cfd3d":
            out["yaw"] = c.yaw_deg
        if self._is_cfd():
            for k, st in self.stats.items():
                out[k] = float(st.mean)
                out[f"{k}_stderr"] = float(st.stderr)
            if self.lifting:
                out["ld"] = self.ld
            out.update(self.forces)
            if self.kind == "cfd3d":
                out["symmetry_plane"] = bool(self.raw.symmetry_plane)
            out.update({"re_real": self.re_real, "re_sim": self.re_sim, "converged": self.converged,
                        "grid": "×".join(str(n) for n in self.raw.grid), "strouhal": self.raw.strouhal})
            if self.physics:
                out["physics"] = list(self.physics)
            st = self.nusselt_stat
            if st is not None:
                out["nusselt"], out["nusselt_stderr"] = float(st.mean), float(st.stderr)
            if _finite(self.heat_rate):
                out["heat_rate"] = self.heat_rate
            if self.fans:
                out["fans"] = self.fans
            if self.diagnostics:
                out["diagnostics"] = self.diagnostics
        else:
            m = self.metrics
            keys = ("cl_max", "alpha_cl_max", "cl_max_reached", "cd_min", "ld_max", "alpha_ld_max",
                    "lift_slope_per_deg", "cl_alpha_per_deg", "alpha_zero_lift", "oswald", "cm0")
            out.update({k: m[k] for k in keys if k in m})
            out.update({"re_sim": self.re_sim, "points": len(self.raw.points), "converged_points": self.n_converged})
            if self.kind == "wing":
                out.update({"method": self.raw.method, "viscous": self.raw.viscous, "aspect_ratio": self.raw.aspect_ratio,
                            "stall_alpha": self.raw.stall_alpha, "stall_cl": self.raw.stall_cl})
        out.update({"status": self.status, "reused": self.reused, "wall_seconds": self.wall_seconds,
                    "warnings": len(self.warnings)})
        return out

    def table(self) -> list[dict[str, Any]]:
        """CFD: one row per coefficient and force (mean, ± std. error, rms, min, max, unit), the
        Results page's Summary table; L/D only for lifting designs (:attr:`lifting`), and
        ``by_symmetry`` marks the zeros a half-domain run has by construction. Curves: one
        row per angle."""
        if self._is_cfd():
            labels = _COEFF_LABELS_3D if self.kind == "cfd3d" else _COEFF_LABELS
            rows = [{"quantity": labels[k], "key": k, "mean": float(st.mean), "stderr": float(st.stderr),
                     "rms": float(st.rms), "min": float(st.min), "max": float(st.max), "unit": "",
                     "by_symmetry": self._by_symmetry(k)}
                    for k, st in self.stats.items()]
            if self.lifting:
                rows.append({"quantity": "L/D", "key": "ld", "mean": self.ld, "stderr": math.nan, "rms": math.nan,
                             "min": math.nan, "max": math.nan, "unit": "", "by_symmetry": False})
            u = self.units
            for k, v in self.forces.items():
                rows.append({"quantity": k.replace("_", " ").capitalize(), "key": k, "mean": float(v),
                             "stderr": math.nan, "rms": math.nan, "min": math.nan, "max": math.nan,
                             "unit": u.get(k, ""), "by_symmetry": self._by_symmetry(k)})
            rows += self._physics_rows()
            return rows
        rows = []
        for p in self.raw.points:
            row = {"alpha": float(p.alpha), "cl": float(p.cl), "cd": float(p.cd), "cm": float(p.cm)}
            if self.kind == "polar":
                row.update({"cdp": float(p.cdp), "cdf": float(p.cdf), "xtr_upper": float(p.xtr_upper),
                            "xtr_lower": float(p.xtr_lower)})
            else:
                row.update({"cdi": float(p.cdi), "cdv": float(p.cdv), "e": float(p.e)})
            row["ld"] = row["cl"] / row["cd"] if p.converged and row["cd"] > 0 else math.nan
            row["converged"] = bool(p.converged)
            rows.append(row)
        return rows

    def _physics_rows(self) -> list[dict[str, Any]]:
        """Table rows of the heat transfer and the fans (Results Summary order: after the forces)."""
        rows: list[dict[str, Any]] = []
        nan = math.nan

        def row(quantity: str, key: str, mean: float, unit: str = "", st: M.Stat | None = None) -> None:
            rows.append({"quantity": quantity, "key": key, "mean": float(mean),
                         "stderr": float(st.stderr) if st is not None else nan,
                         "rms": float(st.rms) if st is not None else nan, "min": float(st.min) if st is not None else nan,
                         "max": float(st.max) if st is not None else nan, "unit": unit, "by_symmetry": False})
        st = self.nusselt_stat
        if st is not None:
            row("Nusselt number", "nusselt", st.mean, "", st)
        if _finite(self.heat_rate):
            row("Heat rate", "heat_rate", self.heat_rate, self.units.get("heat_rate", ""))
        units = _FAN_UNITS_3D if self.kind == "cfd3d" else _FAN_UNITS_2D
        for i, fan in enumerate(self.fans):
            label = str(fan.get("name") or f"Fan {i + 1}")
            mode = str(fan.get("mode", "thrust"))
            series_se = self._fan_series_stderr(i)
            for k, v in fan.items():
                # a standard error is its quantity's "± std. error" cell, not a row of its own
                if k == "name" or k.endswith("_stderr") or isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                se = fan.get(f"{k}_stderr", series_se.get(k))
                st = M.Stat(mean=float(v), stderr=float(se)) if isinstance(se, (int, float)) and _finite(se) else None
                unit = _fan_unit(k, mode, units)
                row(f"{label}: {k.replace('_', ' ')}", f"fan{i}.{k}", float(v), unit, st)
                if st is not None:          # only the standard error is known: rms/min/max stay blank
                    rows[-1].update(rms=nan, min=nan, max=nan)
        return rows

    def _fan_series_stderr(self, i: int) -> dict[str, float]:
        """Standard errors of fan ``i``'s volume flow and disc velocity from their recorded histories.

        The solver reports those two as plain means; their averaging error is in the ``fan<i>_flow`` /
        ``fan<i>_velocity`` series, as the Results page's Fans tab and Summary quote it (0.079 ± 0.028
        there, a bare 0.07915 here before). Empty when the series were not recorded.
        """
        try:
            from .analysis.physics import series_stat
        except ImportError:  # pragma: no cover - analysis always ships with the API
            return {}
        out: dict[str, float] = {}
        for key, series in (("volume_flow", f"fan{i}_flow"), ("induced_velocity", f"fan{i}_velocity")):
            st = series_stat(self.raw, series)
            if st is not None and _finite(st.stderr):
                out[key] = float(st.stderr)
        return out

    # -- plots, exports, reports ----------------------------------------------
    def plot_kinds(self) -> tuple[str, ...]:
        """The ``kind`` values :meth:`plot` accepts for this result (the first is the default)."""
        return _PLOT_KINDS[self.kind]

    @_api
    def plot(self, kind: str | None = None, path: str | os.PathLike | None = None, *, style: str = "print",
             **options: Any) -> Any:
        """A figure of this result (matplotlib, never pyplot): returns the Figure, or the
        written Path when ``path`` is given (PNG, SVG or PDF by suffix).

        Kinds (:meth:`plot_kinds`; the first is the default):

        * 2D CFD -- "history" (coefficients against time with the averaging window),
          "cp" (surface pressure), "field" (``field="speed"|"cp"|"vorticity"|...``,
          ``streamlines=True``), "geometry", "sensitivity";
        * both CFD kinds -- "diagnostics" (the solver's divergence, change norms, CFL and
          residuals against time, on a log scale, when the run recorded them);
        * 3D CFD -- "history", "surface" (``field="cp"|"cf"|"sensitivity"``, views),
          "streamlines" (3D streamlines through the mean flow around the body),
          "smoke" (the instantaneous 3D smoke cloud, when the run recorded it;
          ``opacity=`` scales it, ``color=`` overrides the theme's grey, ``threshold=``
          hides concentrations below it),
          "vortices" (Q-criterion iso-surface), "slice" (``index=0..``, ``field=``),
          "geometry";
        * polar -- "polar" (Cl-α, drag polar, Cm, L/D), "cp" (``alpha=`` one or several);
        * wing -- "polar", "span" (span loading, ``alphas=``), "planform".

        ``style`` is "print" (white) or "dark". 3D pictures need an OpenGL context.
        """
        kinds = self.plot_kinds()
        kind = (kind or kinds[0]).lower()
        if kind not in kinds:
            raise InputError(f"A {self.kind} result has no {kind!r} plot; choose one of {', '.join(kinds)}.")
        if kind == "diagnostics":
            fig = _plot_diagnostics(self, style, dict(options))
        else:
            fig = _PLOTTERS[self.kind](self, kind, style, dict(options))
        if path is None:
            return fig
        from .viz.figures import save_figure
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return save_figure(fig, p, dpi=float(options.get("dpi", 200)))

    @_api
    def export(self, path: str | os.PathLike, what: str | None = None) -> Path:
        """Write the result to a file chosen by its suffix; returns the written path.

        * ``.csv`` -- ``what`` "summary" (default: the result table), "history", "surface"
          (2D), "polar" / "points" (curves), "span" (wing span loading), "slice" (3D, first
          stored slice), "diagnostics" (CFD: the solver's divergence and change norms over time);
        * ``.vtp`` -- 3D surface with Cp, Cf, wall shear, sensitivity (ParaView);
        * ``.vti`` -- flow fields (2D mean fields; 3D mean volume, or ``what="smoke"`` the smoke
          snapshot), mirrored to the full body for symmetry-plane runs;
        * ``.npz`` -- the same fields as numpy arrays;
        * ``.json`` -- :meth:`summary`;
        * ``.png`` / ``.svg`` -- :meth:`plot` with ``what`` as the plot kind;
        * ``.pdf`` / ``.html`` -- a report of this result (:meth:`report`).
        """
        p = Path(path)
        ext = p.suffix.lower()
        p.parent.mkdir(parents=True, exist_ok=True)
        from . import export as X
        r, kind = self.raw, self.kind
        if ext == ".csv":
            w = (what or "summary").lower()
            if w == "summary" or (w in ("polar", "points") and kind in ("polar", "wing")):
                return X.result_csv(r, p) if w == "summary" or kind == "wing" else X.polar_csv(r, p)
            if w == "history" and kind != "polar":
                return X.history_csv(r, p)
            if w == "surface" and kind == "cfd2d":
                return X.surface_csv(r, p)
            if w == "span" and kind == "wing":
                return X.wing_span_csv(r, p)
            if w == "slice" and kind == "cfd3d":
                return X.slice_csv(r, p)
            if w == "diagnostics" and kind in ("cfd2d", "cfd3d"):
                return _diagnostics_csv(self, p)
            raise InputError(f"A {kind} result cannot export {w!r} as CSV.")
        if ext == ".vtp":
            if kind != "cfd3d":
                raise InputError(".vtp (surface) export is for 3D CFD results.")
            return X.surface_vtp(r, p)
        if ext in (".vti", ".npz"):
            if kind == "cfd2d":
                fd = r.mean_fields or r.instant_fields
                if fd is None:
                    raise InputError("This result stores no flow fields.")
                return X.fields_vtk(fd, p) if ext == ".vti" else X.fields_npz(fd, p)
            if kind == "cfd3d":
                vol = _smoke_volume(r) if (what or "").lower() == "smoke" else _mean_volume(r)
                if vol is None:
                    raise InputError("This result stores no volume data" + (" with smoke." if what == "smoke" else "."))
                return X.volume_vti(vol, p) if ext == ".vti" else X.volume_npz(vol, p)
            raise InputError(f"{ext} export is for CFD results.")
        if ext == ".json":
            from .project.fileio import write_json
            write_json(p, plain(self.summary()))
            return p
        if ext in (".png", ".svg"):
            return self.plot(what, p)
        if ext in (".pdf", ".html", ".htm"):
            return self.report(p)
        raise InputError(f"Unknown export format {ext!r}; use .csv, .vtp, .vti, .npz, .json, .png, .svg, .pdf "
                         "or .html.")

    @_api
    def report(self, path: str | os.PathLike, **options: Any) -> Path:
        """A PDF (or .html) report of this result; see :meth:`Project.report`."""
        if self.project is None:
            raise InputError("This result is not in a project; reports are written from a project.")
        return self.project.report([self], path, **options)

    def show(self) -> Result:
        """Select this result in the app (no effect outside the app)."""
        if self.project is not None:
            self.project._host.show(self.project.raw, self.sim_id)
        return self

    # -- text ------------------------------------------------------------------
    def _headline(self) -> str:
        if self._is_cfd():
            labels = _COEFF_LABELS_3D if self.kind == "cfd3d" else _COEFF_LABELS
            bits = []
            for k in ("cd", "cl"):
                st = self.stats[k]
                if not _finite(st.mean):        # a closed cavity has no body: no forces to quote
                    continue
                bits.append(f"{labels[k]} {_mean_error(st.mean, st.stderr)}")
            nu = self.nusselt_stat
            if nu is not None:
                bits.append(f"Nu {_mean_error(nu.mean, nu.stderr)}")
            bits.append("converged" if self.converged else "not converged")
            re_txt = f"Re {_re_text(self.re_sim)}"
            if _finite(self.re_real) and abs(self.re_real - self.re_sim) > 0.01 * self.re_sim:
                re_txt += f" simulated ({_re_text(self.re_real)} real)"
            bits.append(re_txt)
            return ", ".join(bits)
        if not self.n_converged:
            return f"0 of {len(self.raw.points)} angles converged"
        m = self.metrics
        best = f"best L/D {_num(m.get('ld_max', math.nan), 3)} at {_num(m.get('alpha_ld_max', math.nan), 3)}°"
        clmax = f"Cl,max {_num(m.get('cl_max', math.nan), 3)}" + ("" if m.get("cl_max_reached", 1.0) else " (still rising)")
        return f"{clmax}, {best}, {self.n_converged} of {len(self.raw.points)} angles, Re {_re_text(self.re_sim)}"

    def __repr__(self) -> str:
        flags = " (reused)" if self.reused else (" (cancelled)" if self.cancelled else "")
        d = self.design
        of = f" of {d.name!r}" if d is not None else ""
        return f"<Result {self.name!r}{of}{flags}: {self._headline()}>"

    def __str__(self) -> str:
        tag = {"cfd2d": "CFD 2D", "cfd3d": "CFD 3D", "polar": "Polar", "wing": "Wing"}[self.kind]
        lines = [f"{self.name}  [{tag}]", self._headline()]
        if self.reused:
            # a script that prints its result must say when nothing ran: "Finished in 1.3 s" read as a new run
            lines.append("Stored result of an identical run, reused (not run again; rerun=True runs it again).")
        rows = self.table()
        if self._is_cfd():
            def cells(r: dict[str, Any]) -> tuple[str, ...]:
                if r["by_symmetry"]:
                    return (r["quantity"], "0 (symmetry)", "", r["unit"])
                if not _finite(r["stderr"]):
                    return (r["quantity"], _num(r["mean"], 4), "", r["unit"])
                # the Results page's rule (core.format): the error rounded by the PDG rule and the
                # mean to the same decimal place, "steady" for round-off, "≈0" inside the error bar;
                # "Cl -0.001462 ± 0.0027" printed digits the run did not measure
                mean_t, err_t = (plain_number(x) for x in value_error_texts(r["mean"], r["stderr"]))
                return (r["quantity"], mean_t, "" if err_t == "—" else err_t, r["unit"])
            shown = [r for r in rows if r["by_symmetry"] or _finite(r["mean"])]   # a cavity has no forces
            lines.append(_text_table(["quantity", "mean", "± std. error", "unit"], [cells(r) for r in shown]))
        else:
            cols = ["alpha", "cl", "cd", "cm", "ld"]
            lines.append(_text_table(["α°", "Cl", "Cd", "Cm", "L/D", "conv."],
                                     [tuple(_num(r[c], 4) for c in cols) + ("yes" if r["converged"] else "NO",)
                                      for r in rows]))
        lines += [f"! {w}" for w in self.warnings]
        return "\n".join(lines)


def _text_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]

    def line(r: list[str]) -> str:
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
    return "\n".join([line(cells[0]), "  ".join("-" * w for w in widths)] + [line(r) for r in cells[1:]])


def _smoke_volume(raw: M.CFD3DResult) -> M.VolumeData | None:
    """The instantaneous smoke block (DESIGN.md 7.13); None for runs that did not record one."""
    return raw.find_volume("dye")


def _mean_volume(raw: M.CFD3DResult) -> M.VolumeData | None:
    """The time-averaged flow block (streamlines, vortex cores, .vti export).

    Falls back to ``raw.volume`` for results whose mean block carries no label,
    i.e. results written before blocks were labelled.
    """
    return raw.find_volume("u", "mean") or raw.volume


def _on_road(result: Result) -> bool:
    return bool(result.simulation.ground.enabled)


def _plot_cfd2d(r: Result, kind: str, style: str, o: dict) -> Any:
    from .viz import figures as F
    raw, sim = r.raw, r.simulation
    q_inf = sim.conditions.dynamic_pressure()
    o.pop("dpi", None)
    if kind == "history":
        return F.fig_history(raw.history, raw.t_avg_start, raw, style=style, **o)
    if kind == "cp":
        d = r.design
        dk = d.kind if d is not None else "airfoil"
        return F.fig_cp(raw.surface, kind="airfoil" if dk == "airfoil" else "vehicle" if dk == "vehicle" else "body",
                        show_cf=o.pop("show_cf", True), style=style, **o)
    if kind == "field":
        fd = raw.mean_fields if o.pop("instant", False) is False else raw.instant_fields
        fd = fd or raw.mean_fields or raw.instant_fields
        if fd is None:
            raise InputError("This result stores no flow fields.")
        name = o.pop("field", "speed")
        return F.fig_field(fd, name, raw.bodies_wind, streamlines=o.pop("streamlines", name == "speed"),
                           ground=_on_road(r), q_inf=q_inf, style=style, **o)
    if kind == "geometry":
        d = r.design
        if d is None:
            raise InputError("The design of this result is no longer in the project.")
        return F.fig_geometry(d.raw, alpha_deg=sim.conditions.alpha_deg, style=style, **o)
    surfs = [s for s in raw.surface if s.sensitivity is not None]
    if not surfs:
        raise InputError("This result has no sensitivity map; run with sensitivity='cd' (or 'cl', 'ld').")
    return F.fig_sensitivity(surfs, objective=sim.cfd.sensitivity_objective, ground=_on_road(r), style=style, **o)


def _plot_cfd3d(r: Result, kind: str, style: str, o: dict) -> Any:
    from .report import content3d as C3
    from .viz import figures as F
    raw, sim = r.raw, r.simulation
    ground = 0.0 if _on_road(r) else None
    o.pop("dpi", None)
    if kind == "history":
        return F.fig_history3d(raw.history, raw.t_avg_start, raw, symmetry_plane=bool(raw.symmetry_plane),
                               style=style, **o)
    if kind == "geometry":
        d = r.design
        if d is None:
            raise InputError("The design of this result is no longer in the project.")
        return C3.figure_geometry3d(d.raw, style=style, **o)
    if not raw.surface:
        raise InputError("This result stores no surface data.")
    if kind == "surface":
        field = o.pop("field", "cp")
        relative = field == "sensitivity"
        return C3.figure_surface3d(raw.surface, field, ground=ground, relative=relative,
                                   label="relative sensitivity" if relative else None, style=style, **o)
    if kind == "slice":
        if not raw.slices:
            raise InputError("This result stores no slices.")
        index = int(o.pop("index", 0))
        if not 0 <= index < len(raw.slices):
            raise InputError(f"index must be 0..{len(raw.slices) - 1} (the stored slices).")
        return C3.figure_slice3d(raw, index, o.pop("field", "speed"), streamlines=o.pop("streamlines", True),
                                 ground=ground is not None, q_inf=sim.conditions.dynamic_pressure(), style=style, **o)
    if kind in ("streamlines", "vortices"):
        vol = _mean_volume(raw)
        if vol is None:
            raise InputError("This result stores no volume data for streamlines.")
        return C3.figure_flow3d(raw.surface, vol, "streamlines" if kind == "streamlines" else "q", ground=ground,
                                style=style, **o)
    return _figure_smoke3d(r, style, o)


def _figure_smoke3d(r: Result, style: str, o: dict) -> Any:
    """The instantaneous smoke cloud around the grey body (iso view), DESIGN.md 7.13."""
    from .report import content3d as C3
    from .report import render3d as R
    from .viz import figures as F
    raw = r.raw
    vol = _smoke_volume(raw)
    if vol is None:
        raise InputError("This result has no smoke snapshot. Smoke (a passive dye released upstream) is "
                         "recorded by 3D runs of this version; run the simulation again (rerun=True) to get it.")
    ground = 0.0 if _on_road(r) else None
    focus = C3._flow_focus(raw.surface, vol)

    # the look of the smoke (viz3d.Scene.add_smoke); everything else goes to the figure
    smoke_kw = {k: o.pop(k) for k in list(o) if k in ("opacity", "color", "threshold")}

    def build(scene: Any) -> None:
        scene.add_meshes(raw.surface)
        scene.add_smoke(vol, **smoke_kw)
    v = R.render_views(build, ("iso",), ground=ground, focus=focus, theme="dark" if style == "dark" else "light")
    return F.fig_views3d(v.images, v.names, v.titles, v.extents, style=style, **o)


def _plot_polar(r: Result, kind: str, style: str, o: dict) -> Any:
    from .viz import figures as F
    o.pop("dpi", None)
    if kind == "polar":
        return F.fig_polar(r.raw, labels=[o.pop("label", r.name)], style=style, **o)
    alphas = o.pop("alpha", None)
    pts = [p for p in r.raw.points if p.converged and p.cp is not None]
    if not pts:
        raise InputError("This polar stores no pressure distributions.")
    want = [0.0, 4.0, 8.0] if alphas is None else ([alphas] if np.isscalar(alphas) else list(alphas))
    chosen = []
    for a in want:
        p = min(pts, key=lambda q: abs(q.alpha - float(a)))
        if p not in chosen:
            chosen.append(p)
    return F.fig_cp(chosen, labels=[f"α = {p.alpha:g}°" for p in chosen], style=style, **o)


def _plot_wing(r: Result, kind: str, style: str, o: dict) -> Any:
    from .viz import figures as F
    o.pop("dpi", None)
    if kind == "polar":
        return F.fig_wing_polar(r.raw, labels=[o.pop("label", r.name)], style=style, **o)
    if kind == "span":
        return F.fig_span_loading(r.raw, alphas=o.pop("alphas", None), style=style, **o)
    d = r.design
    if d is None or d.raw.wing is None:
        raise InputError("The wing of this result is no longer in the project.")
    from .geometry3d import planform
    return F.fig_wing_planform(d.raw.wing, metrics=planform(d.raw.wing), style=style, **o)



def _plot_diagnostics(r: Result, style: str, o: dict) -> Any:
    """The solver's own health against time: every recorded diagnostic series, log scale where positive."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from .viz.figures import get_style, style_context
    hist = r.diagnostics_history()
    series = {k: v for k, v in hist.items() if k != "t" and np.isfinite(v).any()}
    if not series:
        raise InputError("This result recorded no diagnostics history (runs with diagnostics=True of this "
                         "version record divergence and change norms).")
    st = get_style(style)
    keys = list(series)[:8]
    with style_context(st):
        fig = Figure(figsize=(6.7, 1.0 + 1.25 * len(keys)), layout="constrained", facecolor=st.bg)
        FigureCanvasAgg(fig)
        fig._aerolume_style = st  # type: ignore[attr-defined]   # read back by save_figure
        axs = np.atleast_1d(fig.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0])
        t = hist["t"]
        for i, (ax, k) in enumerate(zip(axs, keys)):
            v = series[k]
            ax.plot(t, v, "-", color=st.series[i % len(st.series)], lw=1.0)
            finite = v[np.isfinite(v)]
            if finite.size and (finite > 0).all() and finite.max() / max(finite.min(), 1e-300) > 50:
                ax.set_yscale("log")
            ax.set_ylabel(k.replace("_", " "), fontsize=7.5)
        axs[-1].set_xlabel("t [L/U]")
        title = o.pop("title", None)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


def _diagnostics_csv(r: Result, path: Path) -> Path:
    import csv
    hist = r.diagnostics_history()
    if not hist:
        raise InputError("This result recorded no diagnostics history.")
    keys = list(hist)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(len(hist["t"])):
            w.writerow([f"{float(hist[k][i]):.9g}" for k in keys])
    return path

_PLOTTERS = {"cfd2d": _plot_cfd2d, "cfd3d": _plot_cfd3d, "polar": _plot_polar, "wing": _plot_wing}


# ===========================================================================
# sweeps and comparisons
# ===========================================================================

class Sweep:
    """Results of a parameter sweep: one simulation per combination of the swept values.

    ``params`` are the swept names (in sweep order), ``values[i]`` the
    combination of point ``i`` and ``results[i]`` its :class:`Result`.
    Iterating a sweep yields its results; ``sweep[i]`` is one of them.
    """

    def __init__(self, params: Sequence[str], points: Sequence[tuple[dict[str, Any], Result]]) -> None:
        self.params = list(params)
        self.values = [dict(v) for v, _ in points]
        self.results = [r for _, r in points]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[Result]:
        return iter(self.results)

    def __getitem__(self, i: int) -> Result:
        return self.results[i]

    @property
    def analysis(self) -> str:
        return self.results[0].analysis if self.results else ""

    def table(self) -> list[dict[str, Any]]:
        """One row per point: the swept values, then the key numbers of its result
        (CFD: coefficients with standard errors, L/D, Re, converged; curves: Cl,max, best L/D...)."""
        rows = []
        for vals, r in zip(self.values, self.results):
            row: dict[str, Any] = dict(vals)
            if r.kind in ("cfd2d", "cfd3d"):
                for k, st in r.stats.items():
                    if r.kind == "cfd3d" and k in ("c_roll", "cn"):
                        continue
                    row[k] = float(st.mean)
                    row[f"{k}_stderr"] = float(st.stderr)
                if r.lifting:
                    row["ld"] = r.ld
                row.update({"drag": r.forces.get("drag", math.nan), "re_sim": r.re_sim, "converged": r.converged})
            else:
                m = r.metrics
                for k in ("cl_max", "alpha_cl_max", "ld_max", "alpha_ld_max", "cd_min", "lift_slope_per_deg",
                          "cl_alpha_per_deg", "oswald", "e_cruise", "cm0"):
                    if k in m:
                        row[k] = m[k]
                row.update({"re_sim": r.re_sim, "converged_points": r.n_converged})
            row["name"] = r.name
            rows.append(row)
        return rows

    def _value(self, row: dict[str, Any], key: str) -> float:
        v = row.get(key, math.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    @_api
    def best(self, key: str | None = None, *, minimize: bool | None = None) -> Result:
        """The result with the best ``key`` (default: "ld" for CFD of a lifting design, "cd" for
        CFD of a car or bluff body, "ld_max" for curves).

        Lower is better for drag-like keys (cd, drag, cd_min, ...), higher for the
        rest, unless ``minimize`` says otherwise. Unconverged CFD points are skipped
        when at least one point converged.
        """
        if not self.results:
            raise RunError("The sweep has no results.")
        rows = self.table()
        if key is None:
            first = self.results[0]
            key = ("ld" if first.lifting else "cd") if first.kind in ("cfd2d", "cfd3d") else "ld_max"
        if all(key not in r for r in rows):
            raise InputError(f"No column {key!r} in this sweep; available: {', '.join(rows[0])}.")
        lower = key in _LOWER_BETTER if minimize is None else bool(minimize)
        cand = [i for i, r in enumerate(rows) if _finite(self._value(r, key))]
        conv = [i for i in cand if rows[i].get("converged", True)]
        cand = conv or cand
        if not cand:
            raise RunError(f"No point of the sweep has a finite {key!r}.")
        pick = min if lower else max
        return self.results[pick(cand, key=lambda i: self._value(rows[i], key))]

    @_api
    def plot(self, x: str | None = None, y: str | Sequence[str] | None = None, *, by: str | None = None,
             path: str | os.PathLike | None = None, style: str = "print", errorbars: bool = True,
             title: str | None = None) -> Any:
        """``y`` against the swept parameter ``x`` (defaults: the first swept parameter and
        "cd"/"ld_max"), one line per value of ``by`` (default: the second swept parameter).
        CFD points carry ±1 standard-error bars; unconverged points are drawn hollow.

        For a sweep of polars or wing sweeps, ``x`` and ``y`` may also be columns of
        the curves themselves ("alpha", "cl", "cd", "cm", "ld", "cdi", "e"): the
        curves are then overlaid, one per point (``plot("alpha", "cl")`` lift
        curves, ``plot("cd", "cl")`` drag polars). Returns the Figure, or the
        written Path when ``path`` is given."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from .viz.figures import get_style, save_figure, style_context
        if not self.results:
            raise RunError("The sweep has no results to plot.")
        rows = self.table()
        x = x or (self.params[0] if self.params else "name")
        ys = [y] if isinstance(y, str) else list(y or ["cd" if "cd" in rows[0] else "ld_max"])
        curves = self.results[0].kind in ("polar", "wing")
        curve_cols = set(self.results[0].table()[0]) if curves and self.results[0].raw.points else set()
        if curves and x in curve_cols and all(k in curve_cols for k in ys) and x not in self.params:
            fig = self._plot_curves(x, ys, style, title)
            if path is None:
                return fig
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return save_figure(fig, p)
        for k in [x, *ys]:
            if all(k not in r for r in rows):
                raise InputError(f"No column {k!r} in this sweep; available: {', '.join(rows[0])}.")
        by = by if by is not None else next((p for p in self.params if p != x), None)
        st = get_style(style)
        groups: dict[Any, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(r.get(by) if by else None, []).append(i)
        with style_context(st):
            fig = Figure(figsize=(6.7, 3.6 if len(ys) == 1 else 2.0 + 1.6 * len(ys)), layout="constrained",
                         facecolor=st.bg)
            FigureCanvasAgg(fig)
            fig._aerolume_style = st  # type: ignore[attr-defined]   # read back by save_figure
            axs = np.atleast_1d(fig.subplots(len(ys), 1, sharex=True, squeeze=False)[:, 0])
            numeric_x = all(isinstance(r.get(x), (int, float, np.floating, np.integer))
                            and not isinstance(r.get(x), bool) for r in rows)
            # a non-numeric x (a template name, a method) is drawn at evenly spaced positions
            cats = list(dict.fromkeys(str(r.get(x)) for r in rows))
            for ax, key in zip(axs, ys):
                for gi, (gval, idx) in enumerate(groups.items()):
                    idx = sorted(idx, key=lambda i: float(rows[i][x]) if numeric_x else cats.index(str(rows[i].get(x))))
                    xv = np.array([float(rows[i][x]) if numeric_x else float(cats.index(str(rows[i].get(x))))
                                   for i in idx])
                    yv = np.array([self._value(rows[i], key) for i in idx])
                    ev = np.array([self._value(rows[i], f"{key}_stderr") for i in idx])
                    conv = np.array([bool(rows[i].get("converged", True)) for i in idx])
                    color = st.series[gi % len(st.series)]
                    label = None
                    if by:
                        label = f"{by} = {_num(gval) if isinstance(gval, (int, float)) else gval}"
                    ax.plot(xv, yv, "-", color=color, lw=1.2, label=label, zorder=3)
                    ax.plot(xv[conv], yv[conv], "o", color=color, ms=4, zorder=4)
                    if (~conv).any():
                        ax.plot(xv[~conv], yv[~conv], "o", mfc=st.bg, mec=color, ms=4, zorder=4)
                    if errorbars and np.isfinite(ev).any():
                        ok = np.isfinite(ev) & np.isfinite(yv)
                        ax.errorbar(xv[ok], yv[ok], yerr=ev[ok], fmt="none", ecolor=color, elinewidth=0.8,
                                    capsize=2.5, zorder=3)
                ax.set_ylabel(_axis_label(key))
                if by:
                    ax.legend(loc="best")
            if not numeric_x:
                axs[-1].set_xticks(range(len(cats)), cats)
            axs[-1].set_xlabel(_axis_label(x))
            if title:
                fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
            if any(not r.get("converged", True) for r in rows):
                axs[-1].annotate("hollow markers: not statistically converged", (1, 0), xycoords="axes fraction",
                                 xytext=(0, -28), textcoords="offset points", ha="right", fontsize=7,
                                 color=st.muted)
        if path is None:
            return fig
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return save_figure(fig, p)

    def _label(self, i: int) -> str:
        """'re 1.0×10⁶, ncrit 9' -- a point's swept values for a legend."""
        bits = []
        for k, v in self.values[i].items():
            if k == "re" and isinstance(v, (int, float)):
                bits.append(f"Re {_re_text(float(v))}")
            else:
                bits.append(f"{k} {_num(v, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}")
        return ", ".join(bits) or self.results[i].name

    def _plot_curves(self, x: str, ys: list[str], style: str, title: str | None) -> Any:
        """Curves of a polar / wing sweep overlaid, one colour per sweep point (converged angles)."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from .viz.figures import get_style, style_context
        st = get_style(style)
        with style_context(st):
            fig = Figure(figsize=(6.7, 3.6 if len(ys) == 1 else 2.0 + 1.6 * len(ys)), layout="constrained",
                         facecolor=st.bg)
            FigureCanvasAgg(fig)
            fig._aerolume_style = st  # type: ignore[attr-defined]   # read back by save_figure
            axs = np.atleast_1d(fig.subplots(len(ys), 1, sharex=True, squeeze=False)[:, 0])
            for ax, key in zip(axs, ys):
                for i, r in enumerate(self.results):
                    pts = [row for row in r.table() if row["converged"]]
                    if x == "alpha":
                        pts.sort(key=lambda row: row["alpha"])
                    xv = np.array([row[x] for row in pts], float)
                    yv = np.array([row[key] for row in pts], float)
                    ax.plot(xv, yv, "-", color=st.series[i % len(st.series)], lw=1.3, marker="o", ms=2.5,
                            label=self._label(i))
                ax.set_ylabel(_axis_label(key))
            axs[0].legend(loc="best")
            axs[-1].set_xlabel(_axis_label(x))
            if title:
                fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
        return fig

    @_api
    def export(self, path: str | os.PathLike) -> Path:
        """Write :meth:`table` as CSV (UTF-8, header row)."""
        import csv
        rows = self.table()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        return p

    def __repr__(self) -> str:
        return f"<Sweep of {', '.join(self.params) or 'nothing'}: {_count(len(self), 'point')}>"

    def __str__(self) -> str:
        rows = self.table()
        if not rows:
            return repr(self)
        cfd = self.results[0].kind in ("cfd2d", "cfd3d")
        if cfd:
            cols = self.params + ["cd", "cl"] + (["ld"] if any("ld" in r for r in rows) else []) + ["converged"]
        else:       # the figures of merit this kind of sweep has (an inviscid wing has no L/D)
            cand = ["cl_max", "ld_max", "alpha_ld_max", "cd_min", "cl_alpha_per_deg", "lift_slope_per_deg", "oswald",
                    "e_cruise"]
            cols = self.params + [c for c in cand if any(_finite(r.get(c)) for r in rows)]
        def cell(c: str, v: Any) -> str:
            if isinstance(v, bool):
                return "yes" if v else "NO"
            if isinstance(v, (int, float)):
                return _re_text(float(v)) if c in ("re", "re_sim") else _num(v, 4)
            return "" if v is None else str(v)
        return _text_table(cols, [tuple(cell(c, r.get(c)) for c in cols) for r in rows])


_AXIS_LABELS = {"alpha": "α [°]", "speed": "speed [m/s]", "yaw": "yaw [°]", "re": "Re", "cd": "Cd", "cl": "Cl",
                "cm": "Cm", "ld": "L/D", "cs": "Cs", "drag": "drag", "ld_max": "best L/D", "cl_max": "Cl,max",
                "cd_min": "Cd,min", "oswald": "Oswald factor e", "e_cruise": "span efficiency e (cruise)",
                "re_sim": "Re (simulated)", "cl_alpha_per_deg": "lift slope CLα [1/°]",
                "lift_slope_per_deg": "lift slope Clα [1/°]", "alpha_ld_max": "α at best L/D [°]",
                "alpha_cl_max": "α at Cl,max [°]", "taper": "taper ratio", "cdi": "induced drag CDi",
                "cdv": "profile drag CDv", "e": "span efficiency e", "quality": "quality"}


def _axis_label(key: str) -> str:
    return _AXIS_LABELS.get(key, key.replace("_", " "))


def _adapt_lifting_rows(rows: list[dict], lifting: dict[str, bool]) -> list[dict]:
    """The Compare page's L/D rule (``app.pages.compare.adapt_rows``, repeated here so the
    API stays free of Qt): L/D is dropped when no compared design lifts, blanked for the
    non-lifting ones in a mixed comparison, and the lift row of cars and bluff bodies
    says how to read its sign."""
    if not lifting or all(lifting.values()):
        return rows
    out = []
    for row in rows:
        if row.get("key") == "ld":
            if not any(lifting.values()):
                continue
            row = {**row, "cells": [c if lifting.get(c.get("sim_id"), True) else
                                    {**c, "value": None, "delta": None, "pct": None, "stderr": None,
                                     "significant": None} for c in row["cells"]]}
        elif row.get("key") == "lift" and not any(lifting.values()):
            row = {**row, "label": "Lift (downforce if negative)"}
        out.append(row)
    return out


class Comparison:
    """The Compare page's table for a set of results (``analysis.compare_rows``).

    ``rows`` are the raw comparison rows (one per metric, cells with value,
    standard error, delta, % and significance); ``warnings`` say why the runs may
    differ for reasons other than the design (different speeds, Reynolds
    numbers, 2D mixed with 3D). :meth:`table` flattens them for printing or CSV.
    """

    def __init__(self, results: Sequence[Result], baseline: int = 0) -> None:
        from .analysis import compare_rows, comparison_warnings
        self.results = list(results)
        self.baseline = int(baseline)
        raw = {r.sim_id: r.raw for r in self.results}
        self.rows = _adapt_lifting_rows(compare_rows(raw, baseline_id=self.results[self.baseline].sim_id),
                                        {r.sim_id: r.lifting for r in self.results})
        # the Simulations (not only their conditions): two closed rooms are compared by their driving
        # inputs (heater power, fans, ...) rather than warned about "different speeds"
        self.warnings = comparison_warnings(raw, {r.sim_id: r.simulation for r in self.results})
        # the Compare page's rule: runs are named after their conditions ("CFD · 30 m/s · α 0°"),
        # so two designs compared at the same conditions get the design's name in front
        sim_names = [r.name for r in self.results]
        names: list[str] = []
        for r in self.results:
            label = r.name
            d = r.design
            if sim_names.count(r.name) > 1 and d is not None:
                label = f"{d.name} · {r.name}"
            names.append(_unique(label, names))
        self.names = names

    def table(self) -> list[dict[str, Any]]:
        """One row per metric: ``metric``, ``unit``, then each result's value and, for the
        others, its change from the baseline in % (``"<name> Δ%"``) and whether that change
        exceeds the statistical noise (``"<name> significant"``)."""
        out = []
        for row in self.rows:
            d: dict[str, Any] = {"metric": row["label"], "key": row["key"], "unit": row["unit"]}
            for name, cell in zip(self.names, row["cells"]):
                d[name] = cell["value"]
                if not cell["is_baseline"]:
                    d[f"{name} Δ%"] = cell["pct"]
                    d[f"{name} significant"] = cell["significant"]
            out.append(d)
        return out

    @_api
    def plot(self, metric: str | Sequence[str] = "cd", path: str | os.PathLike | None = None, *,
             style: str = "print") -> Any:
        """Bars of ``metric`` per result with ±1 standard error and Δ% against the baseline."""
        from .viz import figures as F
        metrics = [metric] if isinstance(metric, str) else list(metric)
        rows = []
        for name, r in zip(self.names, self.results):
            d: dict[str, Any] = {"label": name}
            for m in metrics:
                st = r.stats.get(m)
                if st is not None:
                    d[m] = st
                else:
                    v = r.summary().get(m, math.nan)
                    d[m] = float(v) if isinstance(v, (int, float)) and v is not None else math.nan
            rows.append(d)
        fig = F.fig_compare_bars(rows, metrics, baseline=self.baseline, style=style)
        if path is None:
            return fig
        return F.save_figure(fig, Path(path))

    def __repr__(self) -> str:
        return f"<Comparison of {_count(len(self.results), 'result')}, baseline {self.names[self.baseline]!r}>"

    def __str__(self) -> str:
        head = ["metric"] + [n + (" (baseline)" if i == self.baseline else "") for i, n in enumerate(self.names)]
        lines = []
        for row in self.rows:
            cells = []
            for c in row["cells"]:
                v = c["value"]
                if isinstance(v, bool):
                    cells.append("yes" if v else "NO")
                    continue
                t = "—" if v is None else (_re_text(v) if row["key"] in ("re_real", "re_sim") else _num(v, 4))
                if c["pct"] is not None and not c["is_baseline"]:
                    t += f" ({c['pct']:+.1f} %{'*' if c['significant'] else ''})"
                cells.append(t)
            lines.append([row["label"] + (f" [{row['unit']}]" if row["unit"] else "")] + cells)
        text = _text_table(head, lines)
        if any(c["significant"] for r in self.rows for c in r["cells"]):
            text += "\n* larger than twice the combined standard error"
        return "\n".join([text] + [f"! {w}" for w in self.warnings])


# ===========================================================================
# reference cases (a design, its run settings and the published value to check)
# ===========================================================================

def file_source(path: str | os.PathLike, fmt: str | None = None) -> dict[str, Any]:
    """The provenance of an imported file, as ``Design.source`` keeps it.

    ``{"type": "import", "file", "path", "format", "bytes", "sha256"}``: the name
    and format the Mesh inspector's Source group shows (a mesh imported with
    ``import_mesh`` used to show File "—" and Format "—", critic round 3.4), the
    full path, and the size and SHA-256 of the exact bytes read, so a result can
    be traced to the file even after the file is moved or edited. ``fmt``
    overrides the format taken from the extension ("stl", "obj", "dat").
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"type": "import", "file": p.name, "path": str(p.resolve()),
            "format": (fmt or p.suffix.lower().lstrip(".")) or "", "bytes": p.stat().st_size,
            "sha256": h.hexdigest()}


class Case:
    """A ready-made, referenced set-up: a design, the settings to run it with and the value to expect.

    ``design`` is an ``aerolume.model.Design`` (not yet in a project), ``settings``
    the :meth:`Project.run_cfd` keywords, ``reference`` the published number the
    run should reproduce ({"quantity", "value", "source"}; None when the chosen
    parameters have no published value). :meth:`Project.run_case` (or
    ``run_cfd(case)``) runs it; :meth:`check` compares a result with the
    reference. Built by :func:`cavity_case`, :func:`heated_cylinder_case` and
    :func:`fan_room_case`.
    """

    def __init__(self, name: str, design: M.Design, settings: Mapping[str, Any], *,
                 reference: Mapping[str, Any] | None = None, notes: Sequence[str] = ()) -> None:
        self.name = str(name)
        self.design = design
        self.settings = dict(settings)
        self.reference = dict(reference) if reference else None
        self.notes = list(notes)

    @property
    def kind(self) -> str:
        """"validation" (an independent published value) or "consistency" (the value is imposed by the set-up)."""
        return str((self.reference or {}).get("kind", "validation"))

    def check(self, result: Result) -> dict[str, Any]:
        """The result against the reference: measured value (± standard error), reference, difference in %.

        ``within_stderr`` says whether the reference lies inside ±2 standard errors
        of the measured mean (the statistical part only: the discretisation error
        comes from :meth:`Project.grid_study`). ``band_pct`` is the tolerance band
        docs/validation.md checks this reference with (``band_source`` says where
        and on which grid), and ``within_band`` whether this run's difference lies
        inside it; both are absent when no band is stated (then nothing may be
        called validated against it). ``kind`` is "validation" when the
        reference is an independent published value (de Vahl Davis, Lange et al.)
        and "consistency" when the set-up imposes it (the fan room's thrust): a
        consistency check shows the solver delivers what was set, never that the
        physics is right, and ``note`` says so.
        """
        if self.reference is None:
            raise InputError(f"The case {self.name!r} has no published reference value at these parameters.")
        q = str(self.reference["quantity"])
        se = math.nan
        if q == "nusselt":
            measured = result.nusselt
            st = result.nusselt_stat
            se = float(st.stderr) if st is not None else math.nan
        elif q == "fan thrust":
            fans = result.fans
            measured = float(fans[0].get("thrust", math.nan)) if fans else math.nan
            se = float(fans[0].get("thrust_stderr", math.nan)) if fans else math.nan
        else:
            measured = float(result.summary().get(q, math.nan) or math.nan)
            se = result.stderr(q) if q in result.stats else math.nan
        ref = float(self.reference["value"])
        diff = 100.0 * (measured - ref) / ref if _finite(measured) and ref else math.nan
        out = {"case": self.name, "quantity": q, "measured": measured, "stderr": se, "reference": ref,
               "difference_pct": diff, "source": self.reference.get("source", ""), "kind": self.kind,
               "within_stderr": bool(_finite(se) and abs(measured - ref) <= 2 * se) if _finite(measured) else False}
        band = self.reference.get("band_pct")
        if isinstance(band, (int, float)) and _finite(float(band)) and band > 0:
            out["band_pct"] = float(band)
            out["band_source"] = str(self.reference.get("band_source", ""))
            out["within_band"] = bool(_finite(diff) and abs(diff) <= float(band))
        if self.kind == "consistency":
            out["note"] = str(self.reference.get("note", "The set-up imposes this value: agreement shows the "
                                                         "solver delivers it, not that the physics is validated."))
        if not _finite(measured):
            out["note"] = f"The result has no {q} value: was the run made with this case's settings?"
        return out

    def __repr__(self) -> str:
        ref = ""
        if self.reference:
            word = "consistency" if self.kind == "consistency" else "reference"
            ref = f", {word} {self.reference['quantity']} {_num(self.reference['value'], 4)}"
        return f"<Case {self.name!r}{ref}>"


#: de Vahl Davis (1983), "Natural convection of air in a square cavity: a bench mark numerical
#: solution", Int. J. Numer. Meth. Fluids 3, Table V: Nu₀, the mean Nusselt number of the hot wall
#: (x = 0), Pr 0.71 -- the quantity the solver measures (``Result.nusselt`` of a cavity is the
#: hottest wall's). The paper's Nū (1.118, 2.243, 4.519, 8.800) is the average over the whole
#: cavity, a different quantity; the two differ by 0.1-0.4 %. Same values as
#: validation/thermal2d_cases.py ``DE_VAHL_DAVIS[ra]["nu_0"]`` and docs/validation.md (a test
#: keeps the three equal).
CAVITY_NU_2D = {1e3: 1.117, 1e4: 2.238, 1e5: 4.509, 1e6: 8.817}
#: Tric, Labrosse & Betrouni (2000), Int. J. Heat Mass Transfer 43: mean Nusselt number of the hot
#: wall of the differentially heated cubic cavity (air, Pr 0.71).
CAVITY_NU_3D = {1e4: 2.0542, 1e5: 4.3370, 1e6: 8.6407}
#: The tolerance bands docs/validation.md (section 9, written by validation/run_all.py) checks these
#: references with, in percent of the reference: 2D cavity ``_cavity_tolerance`` (2 % to Ra 10⁴, 3 % at
#: 10⁵, 5 % at 10⁶; on 64², 64², 128², 256² grids), 3D cavity at Ra 10⁴ the published range ± 3 % (48³),
#: heated cylinder ± 4 % (the allowance that section adds to the published range for 5 %
#: blockage at 32 cells/D, applied here about the Lange value). A reference without a measured
#: band (3D cavity at Ra 10⁵ and 10⁶) has none: a run there cannot be called validated.
CAVITY_BAND_2D = {1e3: (2.0, "64²"), 1e4: (2.0, "64²"), 1e5: (3.0, "128²"), 1e6: (5.0, "256²")}
CAVITY_BAND_3D = {1e4: (3.0, "48³")}
CYLINDER_BAND = (4.0, "32 cells/D, 5 % blockage")
#: the Reynolds numbers docs/validation.md checks the heated cylinder at (Re 20 and 40): the band is stated
#: only between them, since outside that range the solver's Nu has not been compared with anything
CYLINDER_BAND_RE = (20.0, 40.0)


#: Published mean Nusselt numbers of the heated cylinder (Pr 0.7) beside the Lange correlation: (low, high,
#: sources) at the Reynolds numbers docs/validation.md checks (the tables Singhal et al. 2023 collect)
CYLINDER_NU_PUBLISHED = {
    20.0: (2.433, 2.4483, "Jafroudi & Yang 1986; Biswas & Sarkar 2009"),
    40.0: (3.200, 3.255, "Jafroudi & Yang 1986; Biswas & Sarkar 2009; Apelt & Ledwich 1979"),
}


def _cylinder_band(re: float) -> dict[str, Any]:
    """``band_pct`` / ``band_source`` of the heated-cylinder reference at ``re`` ({} outside the checked range),
    and ``range`` = (low, high, sources) of the published values at Re 20 or 40 (within 1 %)."""
    lo, hi = CYLINDER_BAND_RE
    if not (lo * (1 - 1e-6) <= float(re) <= hi * (1 + 1e-6)):
        return {}
    out: dict[str, Any] = {
        "band_pct": CYLINDER_BAND[0],
        "band_source": (f"± {CYLINDER_BAND[0]:g} % about the correlation: the allowance docs/validation.md "
                        f"section 9 adds to the published range ({CYLINDER_BAND[1]}, checked at Re 20 and 40)")}
    for re_ref, rng in CYLINDER_NU_PUBLISHED.items():
        if abs(float(re) / re_ref - 1.0) <= 0.01:
            out["range"] = rng
    return out


def _lookup(table: Mapping[float, float], key: float) -> float | None:
    for k, v in table.items():
        if abs(key - k) <= 1e-9 * k:
            return v
    return None


def _air(t_ambient: float) -> M.Fluid:
    from .core import fluid as F
    if abs(t_ambient - 15.0) <= 1e-9:
        return F.air_isa(0.0)
    try:
        return F.air(t_ambient)                 # air at that temperature and sea-level pressure
    except (AttributeError, TypeError):
        return F.air_isa(0.0)


@_api
def cavity_case(ra: float = 1e4, pr: float = 0.71, *, size: float = 0.1, dimension: int = 2,
                t_ambient: float = 15.0, name: str | None = None) -> Case:
    """The differentially heated square (2D) or cubic (3D) cavity, the natural-convection benchmark.

    Air in a closed box of side ``size`` [m]: the left wall at t_ambient + ΔT/2,
    the right wall at t_ambient − ΔT/2, the others adiabatic, gravity along −y
    (2D) or −z (3D). ΔT is chosen so that the Rayleigh number
    Ra = g β ΔT L³ / (ν κ) is exactly ``ra``, with ν from the fluid, κ = ν / Pr
    and the ideal-gas β = 1 / T_ambient written into the settings (so the
    planner uses the same β). The reference is the mean Nusselt number of the hot
    wall, the quantity the run measures: de Vahl Davis' (1983) Nu₀ for Ra 10³–10⁶ in
    2D (1.117, 2.238, 4.509, 8.817), Tric et al. (2000) for Ra 10⁴–10⁶ in 3D
    (:data:`CAVITY_NU_2D`, :data:`CAVITY_NU_3D`).
    """
    ra = _positive("ra", ra)
    pr = _positive("pr", pr)
    L = _positive("size", size)
    if int(dimension) not in (2, 3):
        raise InputError("dimension must be 2 (square cavity) or 3 (cubic cavity).")
    dim = int(dimension)
    t_amb = _float("t_ambient", t_ambient)
    fluid = _air(t_amb)
    nu = fluid.kinematic_viscosity
    g = 9.80665
    beta = 1.0 / (t_amb + 273.15)
    dT = ra * nu * nu / (pr * g * beta * L ** 3)
    th = M.ThermalSettings(enabled=True, t_ambient=t_amb, buoyancy=True, gravity=g, beta=beta, prandtl=pr,
                           walls=[M.WallThermal("left", "temperature", t_amb + dT / 2),
                                  M.WallThermal("right", "temperature", t_amb - dT / 2)])
    label = name or f"Heated cavity{' 3D' if dim == 3 else ''} · Ra {_re_text(ra)}"
    src = {"type": "case", "case": "cavity", "ra": ra, "pr": pr, "size": L, "delta_t": dT}
    if dim == 2:
        d = M.Design(name=label, kind="generic", bodies=[], length_ref=L, force_ref=L, moment_ref=(L / 2, L / 2),
                     source=src)
        box: tuple[float, ...] = (0.0, L, 0.0, L)
    else:
        d = M.Design(name=label, kind="generic", bodies=[], length_ref=L, force_ref=L, dimension=3, ref_area=L * L,
                     span_ref=L, moment_ref3=(L / 2, L / 2, L / 2), source=src)
        box = (0.0, L, 0.0, L, 0.0, L)
    ref_value = _lookup(CAVITY_NU_2D if dim == 2 else CAVITY_NU_3D, ra) if abs(pr - 0.71) < 1e-9 else None
    reference = None
    if ref_value is not None:
        reference = {"quantity": "nusselt", "value": ref_value,
                     "kind": "validation",
                     "source": ("de Vahl Davis (1983), Nu₀: mean Nu of the hot wall" if dim == 2 else
                                "Tric, Labrosse & Betrouni (2000), mean Nu of the hot wall")}
        band = next((v for k, v in (CAVITY_BAND_2D if dim == 2 else CAVITY_BAND_3D).items()
                     if abs(ra - k) <= 1e-9 * k), None)
        if band is not None:
            reference["band_pct"] = band[0]
            reference["band_source"] = f"docs/validation.md section 9, ± {band[0]:g} % on the {band[1]} grid"
    u_ff = math.sqrt(g * beta * dT * L)
    notes = [(f"ΔT = {_num(dT, 4)} K gives Ra {_re_text(ra)} (ν {_num(nu, 4)} m²/s, Pr {_num(pr)}, "
              f"β 1/{_num(t_amb + 273.15, 5)} K)."),
             f"Free-fall velocity √(gβΔT L) = {_num(u_ff, 4)} m/s, √Gr = {_num(math.sqrt(ra / pr), 4)}."]
    return Case(label, d, {"domain": "closed", "domain_box": box, "thermal": th, "fluid": fluid},
                reference=reference, notes=notes)


def lange_nusselt(re: float) -> float:
    """Mean Nusselt number of a cylinder in laminar cross flow of air (Pr 0.7), Lange, Durst & Breuer (1998).

    Nu = 0.082 Re^0.5 + 0.734 Re^x with x = 0.05 + 0.226 Re^0.085, fitted to their
    simulations for 10⁻⁴ ≤ Re ≤ 200 (Int. J. Heat Mass Transfer 41, 3409-3430).
    """
    x = 0.05 + 0.226 * re ** 0.085
    return 0.082 * math.sqrt(re) + 0.734 * re ** x


@_api
def heated_cylinder_case(re: float = 40.0, *, d: float = 0.01, delta_t: float = 10.0, pr: float = 0.7,
                         t_ambient: float = 15.0, name: str | None = None) -> Case:
    """Forced convection from a heated circular cylinder in a free stream (2D).

    A cylinder of diameter ``d`` [m] held ``delta_t`` [K] above the free stream at
    Reynolds number ``re`` (the speed follows from d and the fluid). Buoyancy is
    off: it is the forced-convection benchmark (with it on, Ri = gβΔT d / U² is of
    order 1 at these Reynolds numbers and changes the answer). The reference is
    the correlation of Lange, Durst & Breuer (1998) at Pr 0.7 (:func:`lange_nusselt`),
    for 10⁻⁴ ≤ Re ≤ 200.
    """
    from .geometry import templates
    re = _positive("re", re)
    dia = _positive("d", d)
    dT = _float("delta_t", delta_t)
    if dT == 0:
        raise InputError("delta_t must not be zero: an unheated cylinder transfers no heat.")
    t_amb = _float("t_ambient", t_ambient)
    pr = _positive("pr", pr)
    design = templates.make("cylinder", diameter=dia)
    for b in design.bodies:
        b.thermal, b.thermal_value = "temperature", t_amb + dT
    label = name or f"Heated cylinder · d {_num(dia * 1000, 3)} mm · ΔT {_num(dT)} K"
    design.name = label
    design.source = {**dict(design.source or {}), "case": "heated_cylinder", "delta_t": dT}
    th = M.ThermalSettings(enabled=True, t_ambient=t_amb, buoyancy=False, prandtl=pr)
    reference = None
    if abs(pr - 0.7) < 1e-9 and 1e-4 <= re <= 200:
        reference = {"quantity": "nusselt", "value": round(lange_nusselt(re), 4), "kind": "validation",
                     "source": "Lange, Durst & Breuer (1998) correlation, Pr 0.7", **_cylinder_band(re)}
    return Case(label, design, {"re": re, "thermal": th, "fluid": _air(t_amb)}, reference=reference,
                notes=["Buoyancy off (forced convection); the cylinder wall is held at a fixed temperature."])


@_api
def fan_room_case(thrust: float = 1.0, *, room: Sequence[float] | None = None, diameter: float = 0.5,
                  dimension: int = 2, position: Sequence[float] | None = None, name: str | None = None) -> Case:
    """A fan blowing in a closed room: a consistency check of the actuator disc (DESIGN.md 10.3).

    A room of ``room`` metres ((4, 3) in 2D, (4, 3, 2.5) in 3D) with walls on every
    side and one thrust fan of ``diameter`` [m] blowing along +x from a third of
    the length (``position`` overrides). ``thrust`` is N (3D) or N per metre of
    span (2D).

    The "reference" is the set thrust, and it is a **consistency** check, not a
    validation (``case.kind == "consistency"``): in thrust mode the solver imposes
    that thrust, so agreement only shows the disc delivers what was set. Froude
    momentum theory does not apply either: it assumes an unbounded stream, while in
    a closed room the fan recirculates its own wake and the induced velocity is set
    by the room's losses, so it reads several times below the free-air value. The
    fan model is validated in a channel against confined momentum theory
    (docs/validation.md, "Fans and jets").
    """
    dim = int(dimension)
    if dim not in (2, 3):
        raise InputError("dimension must be 2 or 3.")
    size = tuple(float(c) for c in (room if room is not None else ((4.0, 3.0) if dim == 2 else (4.0, 3.0, 2.5))))
    if len(size) != dim or not all(c > 0 for c in size):
        raise InputError(f"room must hold {dim} positive lengths [m], got {room!r}.")
    dia = _positive("diameter", diameter)
    if dia >= min(size[1:]):
        raise InputError(f"A fan of {dia:g} m does not fit across a room {min(size[1:]):g} m wide.")
    t = _float("thrust", thrust)
    pos = _vector("position", position, dim) if position is not None else \
        tuple([size[0] / 3.0] + [c / 2.0 for c in size[1:]])
    direction = (1.0, 0.0) if dim == 2 else (1.0, 0.0, 0.0)
    label = name or f"Fan in a room{' 3D' if dim == 3 else ''} · {_num(t)} N{'/m' if dim == 2 else ''}"
    src = {"type": "case", "case": "fan_room", "room": list(size), "diameter": dia, "thrust": t}
    if dim == 2:
        d = M.Design(name=label, kind="generic", bodies=[], length_ref=dia, force_ref=dia, moment_ref=pos, source=src)
        box: tuple[float, ...] = (0.0, size[0], 0.0, size[1])
    else:
        d = M.Design(name=label, kind="generic", bodies=[], length_ref=dia, force_ref=dia, dimension=3,
                     ref_area=math.pi * dia * dia / 4, span_ref=dia, moment_ref3=pos, source=src)
        box = (0.0, size[0], 0.0, size[1], 0.0, size[2])
    fan = M.Fan(name="Fan", position=pos, direction=direction, diameter=dia, mode="thrust", value=t)
    note = fan_room_note(dim)
    return Case(label, d, {"domain": "closed", "domain_box": box, "fans": [fan]},
                reference={"quantity": "fan thrust", "value": t, "kind": "consistency",
                           "source": "consistency (imposed): thrust mode sets this thrust",
                           "note": note},
                notes=[note])


def fan_room_note(dimension: int) -> str:
    """:data:`FAN_ROOM_NOTE`, plus, for a 2D room, :data:`aerolume.simulation.physics.ROOM_2D_FAN_NOTE`.

    A 2D room's speeds have no grid-converged answer (theory_ns2d.md §18.4), so every place the
    case's note reaches (the printed case, the reference check, the assistant) says so."""
    if int(dimension) == 3:
        return FAN_ROOM_NOTE
    from .simulation.physics import ROOM_2D_FAN_NOTE  # lazily: the planner module is heavy
    return f"{FAN_ROOM_NOTE} {ROOM_2D_FAN_NOTE}"


#: Why the fan room's check is not a validation, and why momentum theory does not apply there.
FAN_ROOM_NOTE = ("Consistency check, not a validation: thrust mode imposes this thrust, so agreement only shows the "
                 "fan delivers what was set. Froude momentum theory assumes an unbounded stream and does not apply in "
                 "a closed room, where the fan recirculates its own wake; the fan model is validated in a channel "
                 "against confined momentum theory (docs/validation.md).")


def _run_prandtl(sim: M.Simulation) -> float:
    """The Prandtl number a thermal run used: the typed one, else the fluid's at the ambient temperature."""
    th = sim.thermal
    if th.prandtl is not None and _finite(th.prandtl):
        return float(th.prandtl)
    try:
        from .simulation.physics import thermal_props
        return float(thermal_props(sim).prandtl)
    except Exception:  # noqa: BLE001 - without the simulation layer the run's Pr is unknown
        return math.nan


def reference_case(design: Any, simulation: M.Simulation, *, re: float | None = None) -> tuple[Case | None, str]:
    """The reference :class:`Case` a stored simulation reproduces, rebuilt from its design: ``(case, why_not)``.

    A run made from :func:`cavity_case`, :func:`heated_cylinder_case` or
    :func:`fan_room_case` -- or a heated 2D cylinder template run in forced
    convection -- still carries the reference it can be checked against, after
    the case object itself is gone (a later session, the assistant, the app).
    Returns ``(None, "")`` for a design that is no reference set-up at all, and
    ``(None, reason)`` when it is one but the simulation's settings no longer
    match the published conditions (buoyancy switched on for the forced-convection
    cylinder, walls changed in the cavity, Re outside the correlation's range):
    a reference is only quoted for the conditions it was published for. ``re``
    is the Reynolds number the run simulated (the cylinder's correlation is
    evaluated there; default: from the simulation's conditions).
    """
    d = design.raw if isinstance(design, Design) else design
    if not isinstance(d, M.Design) or simulation.analysis != "cfd":
        return None, ""
    src = dict(d.source or {})
    kind = str(src.get("case", ""))
    dim = int(d.dimension)
    sim = simulation
    th = sim.thermal
    if kind == "cavity":
        if sim.cfd.domain_mode != "closed":
            return None, "the cavity benchmark needs the closed domain; this run used a free-stream tunnel"
        if not (th.enabled and th.buoyancy):
            return None, "the cavity benchmark is natural convection: this run had heat transfer or buoyancy off"
        walls = {w.side: w for w in th.walls}
        hot, cold = walls.get("left"), walls.get("right")
        dt = float(src.get("delta_t", math.nan))
        if (hot is None or cold is None or hot.mode != "temperature" or cold.mode != "temperature"
                or not _finite(dt) or abs((float(hot.value) - float(cold.value)) - dt) > 1e-6 * max(abs(dt), 1e-12)
                or any(w.mode != "adiabatic" for s, w in walls.items() if s not in ("left", "right"))):
            return None, "the cavity's wall temperatures were changed from the benchmark's"
        pr = _run_prandtl(sim)
        if not (_finite(pr) and abs(pr - float(src.get("pr", 0.71))) <= 1e-6):
            return None, "the Prandtl number differs from the benchmark's"
        case = cavity_case(float(src["ra"]), float(src.get("pr", 0.71)), size=float(src.get("size", d.length_ref)),
                           dimension=dim, t_ambient=float(th.t_ambient), name=d.name)
        if case.reference is None:
            return None, (f"no published value at Ra {_re_text(float(src['ra']))}, Pr {_num(float(src.get('pr', 0.71)))}"
                          " (published: Ra 10³–10⁶ in 2D, 10⁴–10⁶ in 3D, Pr 0.71)")
        return case, ""
    template = str(src.get("template", "")) if src.get("type") == "template" else ""
    if kind == "heated_cylinder" or (template == "cylinder" and dim == 2):
        parts = list(d.bodies)
        if dim != 2 or not parts or any(getattr(b, "thermal", "adiabatic") != "temperature" for b in parts):
            return (None, "the cylinder correlation needs a cylinder wall held at a fixed temperature"
                    if kind == "heated_cylinder" else "")
        if not th.enabled:
            return None, "heat transfer was off in this run"
        if th.buoyancy:
            return None, ("buoyancy was on: Lange et al.'s correlation is for forced convection (buoyancy off); "
                          "with it on the Richardson number changes the answer")
        if sim.cfd.domain_mode == "closed":
            return None, "the correlation is for a cylinder in a free stream, not in a closed domain"
        re_run = float(re) if re is not None and _finite(re) else float(sim.conditions.reynolds(float(d.length_ref)))
        if not (_finite(re_run) and 1e-4 <= re_run <= 200):
            return None, f"Re {_re_text(re_run)} is outside the correlation's range (10⁻⁴ to 200)"
        pr = _run_prandtl(sim)
        if not (_finite(pr) and abs(pr - 0.7) <= 0.011):
            return None, f"the run's Prandtl number {_num(pr, 3)} is not the correlation's 0.7"
        notes = ["Buoyancy off (forced convection); the cylinder wall is held at a fixed temperature."]
        if abs(pr - 0.7) > 1e-9:
            notes.append(f"Lange et al. fitted Pr 0.7; this run used Pr {_num(pr, 3)}, which raises Nu by about "
                         f"{_num(100 * ((pr / 0.7) ** (1 / 3) - 1), 2)} % (Nu ∝ Pr^⅓).")
        reference = {"quantity": "nusselt", "value": round(lange_nusselt(re_run), 4), "kind": "validation",
                     "source": f"Lange, Durst & Breuer (1998) correlation at Re {_num(re_run, 4)}, Pr 0.7",
                     **_cylinder_band(re_run)}
        return Case(d.name, d, {"re": re_run}, reference=reference, notes=notes), ""
    if kind == "fan_room":
        fans = [f for f in sim.fans if f.enabled]
        if sim.cfd.domain_mode != "closed" or not fans:
            return None, "the fan-room check needs the closed room and its fan"
        f0 = fans[0]
        if f0.mode != "thrust" or abs(float(f0.value) - float(src.get("thrust", math.nan))) > 1e-9 * max(
                abs(float(f0.value)), 1.0):
            return None, "the fan's mode or thrust was changed from the case's"
        return Case(d.name, d, {}, reference={"quantity": "fan thrust", "value": float(f0.value),
                                              "kind": "consistency",
                                              "source": "consistency (imposed): thrust mode sets this thrust",
                                              "note": fan_room_note(d.dimension)},
                    notes=[fan_room_note(d.dimension)]), ""
    return None, ""


# ===========================================================================
# grid convergence (Richardson extrapolation and the GCI)
# ===========================================================================

def grid_convergence(h: Sequence[float], phi: Sequence[float], *, stderr: Sequence[float] | None = None,
                     quantity: str = "", formal_order: float = 2.0, method: str = "celik",
                     safety: float | None = None) -> Any:
    """Discretisation uncertainty of ``phi`` from three or more grids (Celik et al. 2008).

    A thin front for :func:`aerolume.analysis.gci.gci_study`, the one
    implementation the Results page, the report and the CSV exports use, so a
    script quotes the same order, extrapolated value and GCI as the app. ``h``
    are representative cell sizes in any order (only ratios matter), ``phi`` the
    values on them and ``stderr`` their statistical standard errors (time
    averages of an unsteady run; None for deterministic values). Returns an
    :class:`aerolume.analysis.gci.GCIResult`: ``convergence`` ("monotone",
    "oscillatory", "divergent", "oscillatory-divergent", "insensitive",
    "converged-fine"), the observed order ``p``, the Richardson value ``phi_ext``,
    ``gci_fine`` (relative), ``u_num`` / ``u_stat`` / ``u_total`` (absolute
    half-widths of about 95 % bands on the fine value), ``noise_limited`` (the
    grid differences are within twice the combined standard errors, so the order
    measures the averaging noise and is not claimed), ``asymptotic`` (observed
    order within 10 % of ``formal_order``), ``warnings`` and ``summary()``.

    ``method="celik"`` uses the observed order with the safety factor 1.25;
    ``"roy"`` keeps that only near the formal order and otherwise uses Fs = 3
    (Oberkampf & Roy 2010). ``safety`` overrides the safety factor.
    """
    from .analysis import gci as G
    try:
        return G.gci_study([float(x) for x in h], [float(x) for x in phi],
                           None if stderr is None else [float(x) for x in stderr], quantity=str(quantity) or "value",
                           formal_order=float(formal_order), fs=None if safety is None else float(safety),
                           method=str(method))
    except (TypeError, ValueError) as exc:
        raise InputError(f"grid_convergence: {exc}.") from None


class GridStudy:
    """The same case on systematically refined grids, with Richardson extrapolation and the GCI.

    ``results`` coarse to fine (API :class:`Result` objects), ``cells_per_ref``
    the grids as solved, ``h`` the relative cell sizes (1 / cells per reference
    length), ``gci[q]`` the :class:`aerolume.analysis.gci.GCIResult` of each
    quantity ``q`` and ``study`` the :class:`aerolume.analysis.gci.GridStudy` the
    report and :func:`aerolume.export.grid_study_csv` take. The numbers are those of
    :mod:`aerolume.analysis.gci`, so the report of the same three runs agrees.
    """

    def __init__(self, results: Sequence[Result], cells_per_ref: Sequence[float], quantities: Sequence[str]) -> None:
        from .analysis import gci as G
        self.results = list(results)
        self.cells_per_ref = [float(c) for c in cells_per_ref]
        self.h = [1.0 / c for c in self.cells_per_ref]
        self.quantities = list(quantities)
        self.gci: dict[str, Any] = {}
        sim_ids = [r.sim_id for r in self.results]
        cells = [_cells(r) for r in self.results]
        self.notes: list[str] = []
        for q in self.quantities:
            vals, ses = zip(*(_quantity(r, q) for r in self.results), strict=True) if self.results else ((), ())
            try:
                self.gci[q] = G.gci_study(self.h, list(vals), list(ses), quantity=_axis_label(q), sim_ids=sim_ids,
                                          cells=cells, labels=[r.name for r in self.results])
            except ValueError as exc:
                self.notes.append(f"{_axis_label(q)}: {exc}.")
        dim = 3 if self.results and self.results[0].dimension == 3 else 2
        order = sorted(range(len(self.results)), key=lambda i: self.h[i])          # fine first, as analysis.gci
        self.study = G.GridStudy(sim_ids=[sim_ids[i] for i in order], h=[self.h[i] for i in order],
                                 cells=[cells[i] for i in order], dimension=dim, quantities=dict(self.gci),
                                 notes=list(self.notes))

    def table(self) -> list[dict[str, Any]]:
        """One row per grid: cells per reference length, cells, and every quantity with its standard error."""
        rows = []
        for c, r in zip(self.cells_per_ref, self.results, strict=True):
            row: dict[str, Any] = {"cells_per_ref": c, "cells": _cells(r), "name": r.name}
            for q in self.quantities:
                v, se = _quantity(r, q)
                row[q], row[f"{q}_stderr"] = v, se
            rows.append(row)
        return rows

    def summary(self) -> dict[str, str]:
        """``{quantity: sentence}``: the fine value ± its GCI band, the order and the extrapolated value."""
        return {q: _gci_sentence(g) for q, g in self.gci.items()}

    def export(self, path: str | os.PathLike) -> Path:
        """The study as CSV (one row per quantity and grid, ``export.grid_study_csv``)."""
        from .export import grid_study_csv
        p = Path(path)
        if p.suffix.lower() != ".csv":
            raise InputError(f"A grid study exports as .csv, got {p.suffix or 'no suffix'!r}.")
        p.parent.mkdir(parents=True, exist_ok=True)
        return Path(grid_study_csv(self.study, p))

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        bits = []
        for q, g in self.gci.items():
            if _finite(g.gci_fine):
                bits.append(f"{q} GCI {100 * g.gci_fine:.2g} % (p {g.p:.2g})" if _finite(g.p)
                            else f"{q} GCI {100 * g.gci_fine:.2g} %")
            else:
                bits.append(f"{q} {g.convergence or 'no estimate'}")
        return f"<GridStudy of {_count(len(self), 'grid')}: {'; '.join(bits)}>"

    def __str__(self) -> str:
        cols = ["cells_per_ref", *self.quantities]
        body = _text_table(["cells/ref", *self.quantities],
                           [tuple([_num(r["cells_per_ref"], 4)] + [
                               _num(r[q], 5) + (f" ± {_num(r[f'{q}_stderr'], 2)}" if _finite(r[f"{q}_stderr"]) else "")
                               for q in cols[1:]]) for r in self.table()])
        lines = [body, ""]
        for g in self.gci.values():
            lines.append(_gci_sentence(g))
            lines.extend(f"  {w}" for w in [*g.warnings, *g.notes])
        lines.extend(self.notes)
        return "\n".join(lines).rstrip()

    @_api
    def plot(self, quantity: str | None = None, path: str | os.PathLike | None = None, *, style: str = "print") -> Any:
        """``quantity`` against the cell size h (±1 standard error), with the extrapolated value at h = 0."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from .viz.figures import get_style, save_figure, style_context
        q = quantity or self.quantities[0]
        if q not in self.quantities:
            raise InputError(f"No quantity {q!r} in this study; it has {', '.join(self.quantities)}.")
        g = self.gci.get(q)
        st = get_style(style)
        rows = self.table()
        with style_context(st):
            fig = Figure(figsize=(5.6, 3.4), layout="constrained", facecolor=st.bg)
            FigureCanvasAgg(fig)
            fig._aerolume_style = st  # type: ignore[attr-defined]
            ax = fig.subplots()
            hv = np.array(self.h)
            yv = np.array([r[q] for r in rows], float)
            ev = np.array([r[f"{q}_stderr"] for r in rows], float)
            ax.errorbar(hv, yv, yerr=np.where(np.isfinite(ev), ev, 0.0), fmt="o-", color=st.series[0], ms=4,
                        lw=1.2, capsize=2.5, label="computed (±1 standard error)")
            if g is not None and _finite(g.phi_ext) and not g.noise_limited:
                lab = f"extrapolated (p = {g.p:.2g})" if _finite(g.p) else "extrapolated"
                ax.plot([0.0], [g.phi_ext], "s", color=st.series[1], ms=5, label=lab)
            ax.set_xlim(left=0.0)
            ax.set_xlabel("h = 1 / cells per reference length")
            ax.set_ylabel(_axis_label(q))
            ax.legend(loc="best")
        if path is None:
            return fig
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return save_figure(fig, p)


def _with_band(value: float, band: float) -> tuple[str, str]:
    """``value`` and ``band`` written to the band's second significant figure ("1.00028", "0.00039").

    A fixed 4 significant figures printed 1.000277 ± 0.000386 as "1 ± 0.0003858":
    the value lost the digits its own uncertainty supports.
    """
    if not _finite(value):
        return "—", "—"
    if not (_finite(band) and band > 0):
        return _num(value, 5), "—"
    places = max(0, 1 - math.floor(math.log10(band)))
    return f"{value:.{places}f}", f"{band:.{places}f}"


def _gci_sentence(g: Any) -> str:
    """One line for a :class:`~aerolume.analysis.gci.GCIResult`: fine value ± GCI band, order, extrapolation."""
    f1 = g.fine.value
    if not _finite(g.u_num) or g.convergence == "insensitive":
        return g.summary()
    v, u = _with_band(f1, g.u_num)
    ext, _ = _with_band(g.phi_ext, g.u_num)
    rel = f" ({100 * g.gci_fine:.2g} %)" if _finite(g.gci_fine) else ""
    order = f"observed order {g.p:.2f}" if _finite(g.p) else "order not determined"
    tail = (" Noise-limited: the grid differences are within the statistical error, so the order is not claimed."
            if g.noise_limited else "")
    return (f"{g.quantity}: {v} ± {u}{rel} (GCI of the fine grid, {g.convergence}; {order}; extrapolated "
            f"{ext}).{tail}")


def _cells(r: Result) -> int:
    """The cell count of a CFD result (its ``cells``, else the product of its grid)."""
    raw = r.raw
    n = int(getattr(raw, "cells", 0) or 0)
    if not n:
        grid = tuple(getattr(raw, "grid", ()) or ())
        n = int(np.prod(grid)) if grid else 0
    return n


_GRID_SUFFIX = re.compile(r"\s*·\s*\d+ cells/ref$")


def _grid_study_of(results: list[Result], grids: Sequence[float], quantity: str | Sequence[str] | None) -> GridStudy:
    """A :class:`GridStudy` of three results: the quantities asked for, or the defaults of :meth:`Project.grid_study`."""
    if quantity is None:
        first = results[0]
        if first.nusselt_stat is not None and not _finite(first.cd):
            qs = ["nusselt"]
        else:
            qs = ["cd", "cl"] if first.lifting else ["cd"]
            if first.nusselt_stat is not None:
                qs.append("nusselt")
    else:
        qs = [quantity] if isinstance(quantity, str) else list(quantity)
    solved = [float(r.raw.cells_per_ref) if _finite(getattr(r.raw, "cells_per_ref", math.nan)) else float(n)
              for r, n in zip(results, grids, strict=True)]
    return GridStudy(results, solved, qs)


def _quantity(r: Result, q: str) -> tuple[float, float]:
    """(value, standard error) of a quantity of a CFD result: a coefficient, "nusselt" or a summary key."""
    if q == "nusselt":
        st = r.nusselt_stat
        return (float(st.mean), float(st.stderr)) if st is not None else (math.nan, math.nan)
    if q in r.stats:
        st = r.stats[q]
        return float(st.mean), float(st.stderr)
    v = r.summary().get(q, math.nan)
    try:
        return float(v), math.nan
    except (TypeError, ValueError):
        return math.nan, math.nan


# ===========================================================================
# editable settings (suggestions, edit_simulation)
# ===========================================================================

@dataclasses.dataclass(frozen=True)
class Setting:
    """One editable setting of a simulation, with its label, unit, valid range and analyses.

    :meth:`Project.edit_simulation` and the assistant's suggestion cards go
    through these, so a value is checked the same way wherever it comes from: an
    unknown setting is refused (never guessed), a number outside ``lo``..``hi``
    is refused with the range, a choice must be one of ``choices``.
    """
    key: str
    label: str
    unit: str
    analyses: tuple[str, ...]
    kind: str                       # "float" | "int" | "bool" | "choice"
    lo: float | None = None
    hi: float | None = None
    choices: tuple[str, ...] = ()
    optional: bool = False          # None is allowed ("automatic": the planner chooses)
    dims: tuple[int, ...] = (2, 3)

    def check(self, value: Any) -> Any:
        """The value coerced to this setting's type, or InputError saying why it is not allowed."""
        if value is None:
            if self.optional:
                return None
            raise InputError(f"{self.label} needs a value.")
        if self.kind == "bool":
            if isinstance(value, str):
                low = value.strip().lower()
                if low not in ("true", "false", "on", "off", "yes", "no", "1", "0"):
                    raise InputError(f"{self.label} is on or off, got {value!r}.")
                return low in ("true", "on", "yes", "1")
            return bool(value)
        if self.kind == "choice":
            v = str(value).strip().lower()
            if v not in self.choices:
                raise InputError(f"{self.label} must be one of {', '.join(self.choices)}, got {value!r}.")
            return v
        x = _float(self.label, value)
        if self.kind == "int":
            if abs(x - round(x)) > 1e-9:
                raise InputError(f"{self.label} must be a whole number, got {value!r}.")
            x = round(x)
        if (self.lo is not None and x < self.lo) or (self.hi is not None and x > self.hi):
            unit = f" {self.unit}" if self.unit else ""
            raise InputError(f"{self.label} must be between {self.lo:g} and {self.hi:g}{unit}, got {x:g}.")
        return x

    def get(self, sim: M.Simulation) -> Any:
        return _SETTING_ACCESS[self.key][0](sim)

    def set(self, sim: M.Simulation, value: Any) -> None:
        _SETTING_ACCESS[self.key][1](sim, value)


def _setter(obj: Callable[[M.Simulation], Any], attr: str) -> Callable[[M.Simulation, Any], None]:
    return lambda sim, v: setattr(obj(sim), attr, v)


def _access(obj: Callable[[M.Simulation], Any], attr: str) -> tuple[Callable, Callable]:
    return (lambda sim: getattr(obj(sim), attr)), _setter(obj, attr)


def _sweep_part(sim: M.Simulation) -> Any:
    return sim.airfoil if sim.analysis == "polar" else sim.wing


_SETTING_ACCESS: dict[str, tuple[Callable, Callable]] = {
    "speed": _access(lambda s: s.conditions, "speed"),
    "alpha": _access(lambda s: s.conditions, "alpha_deg"),
    "yaw": _access(lambda s: s.conditions, "yaw_deg"),
    "quality": _access(lambda s: s.cfd, "quality"),
    "duration": _access(lambda s: s.cfd, "duration"),
    "average_from": _access(lambda s: s.cfd, "average_from"),
    "cells_per_ref": _access(lambda s: s.cfd, "cells_per_ref"),
    "turbulence": _access(lambda s: s.cfd, "turbulence"),
    "reynolds": _access(lambda s: s.cfd, "reynolds_mode"),
    "re_cap": _access(lambda s: s.cfd, "reynolds_cap"),
    "ground": _access(lambda s: s.ground, "enabled"),
    "moving_ground": _access(lambda s: s.ground, "moving"),
    "alpha_start": _access(_sweep_part, "alpha_start"),
    "alpha_end": _access(_sweep_part, "alpha_end"),
    "alpha_step": _access(_sweep_part, "alpha_step"),
    "ncrit": _access(_sweep_part, "ncrit"),
    "mach": _access(_sweep_part, "mach"),
    "method": _access(lambda s: s.wing, "method"),
    "viscous": _access(lambda s: s.wing, "viscous"),
    "thermal": _access(lambda s: s.thermal, "enabled"),
    "t_ambient": _access(lambda s: s.thermal, "t_ambient"),
    "buoyancy": _access(lambda s: s.thermal, "buoyancy"),
    "diagnostics": _access(lambda s: s.cfd, "record_diagnostics"),
    "adaptive_dt": _access(lambda s: s.cfd, "adaptive_dt"),
}

_CFD = ("cfd",)
_SWEEPS = ("polar", "wing")
#: Every setting :meth:`Project.edit_simulation` (and an assistant suggestion) may change.
SETTINGS: dict[str, Setting] = {s.key: s for s in (
    Setting("speed", "Speed", "m/s", ("cfd", "polar", "wing"), "float", 1e-4, 400.0),
    Setting("alpha", "Angle of attack", "°", _CFD, "float", -30.0, 30.0),
    Setting("yaw", "Yaw", "°", _CFD, "float", -45.0, 45.0, dims=(3,)),
    Setting("quality", "Quality", "", _CFD, "choice", choices=("draft", "standard", "fine")),
    Setting("duration", "Duration", "L/U", _CFD, "float", 1.0, 2000.0, optional=True),
    Setting("average_from", "Average from", "L/U", _CFD, "float", 0.0, 2000.0, optional=True),
    Setting("cells_per_ref", "Cells per reference length", "", _CFD, "int", 4, 2048, optional=True),
    Setting("turbulence", "Turbulence", "", _CFD, "choice", choices=M.TURBULENCE),
    Setting("reynolds", "Reynolds mode", "", _CFD, "choice", choices=M.REYNOLDS_MODES),
    Setting("re_cap", "Reynolds cap", "", _CFD, "float", 10.0, 1e9, optional=True),
    Setting("ground", "Road", "", _CFD, "bool"),
    Setting("moving_ground", "Moving road", "", _CFD, "bool"),
    Setting("alpha_start", "First angle", "°", _SWEEPS, "float", -30.0, 30.0),
    Setting("alpha_end", "Last angle", "°", _SWEEPS, "float", -30.0, 40.0),
    Setting("alpha_step", "Angle step", "°", _SWEEPS, "float", 0.05, 10.0),
    Setting("ncrit", "Transition N_crit", "", _SWEEPS, "float", 0.5, 15.0),
    Setting("mach", "Mach number", "", _SWEEPS, "float", 0.0, 0.7),
    Setting("method", "Wing method", "", ("wing",), "choice", choices=M.WING_METHODS),
    Setting("viscous", "Viscous strips", "", ("wing",), "bool"),
    Setting("thermal", "Heat transfer", "", _CFD, "bool"),
    Setting("t_ambient", "Ambient temperature", "°C", _CFD, "float", -80.0, 300.0),
    Setting("buoyancy", "Buoyancy", "", _CFD, "bool"),
    Setting("diagnostics", "Record diagnostics", "", _CFD, "bool"),
    Setting("adaptive_dt", "Time-step control", "", _CFD, "choice", choices=("cfl", "pid")),
)}


def setting_for(key: str, sim: M.Simulation, design: M.Design | None = None) -> Setting:
    """The :class:`Setting` ``key`` of ``sim``, or InputError (unknown, or not for this analysis / dimension)."""
    spec = SETTINGS.get(str(key))
    if spec is None:
        import difflib
        close = difflib.get_close_matches(str(key), list(SETTINGS), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise InputError(f"Unknown setting {key!r}.{hint} Settings: {', '.join(SETTINGS)}.")
    if sim.analysis not in spec.analyses:
        raise InputError(f"{spec.label} is not a setting of a {sim.analysis!r} simulation "
                         f"(it applies to {', '.join(spec.analyses)}).")
    if design is not None and int(design.dimension) not in spec.dims:
        raise InputError(f"{spec.label} applies to {' or '.join(f'{d}D' for d in spec.dims)} designs only.")
    return spec


# ===========================================================================
# designs
# ===========================================================================

class Design:
    """A design of a :class:`Project` (2D outline, 3D mesh or wing planform).

    Reads ``name``, ``kind``, ``dimension``, ``id`` and the references from the
    project's current copy, so a rename in the app shows here too. ``raw`` is
    the :class:`aerolume.model.Design`. Methods that change the design record
    the change in the project.
    """

    def __init__(self, project: Project, raw: M.Design) -> None:
        self.project = project
        self.id = raw.id
        self._raw = raw

    @property
    def raw(self) -> M.Design:
        d = self.project.raw.designs.get(self.id)
        if d is not None:
            self._raw = d
        return self._raw

    @property
    def name(self) -> str:
        return self.raw.name

    @property
    def kind(self) -> str:
        return self.raw.kind

    @property
    def dimension(self) -> int:
        return int(self.raw.dimension)

    @property
    def is_3d(self) -> bool:
        return self.dimension == 3

    @property
    def analyses(self) -> list[str]:
        """The analyses this design can have, default first ("cfd", "polar", "wing")."""
        return _analyses_for(self.raw)

    @property
    def references(self) -> dict[str, Any]:
        """Reference quantities: 2D length/force reference and moment point; 3D also area and span."""
        d = self.raw
        out: dict[str, Any] = {"length_ref": d.length_ref, "force_ref": d.force_ref}
        if d.dimension == 3:
            out.update({"ref_area": d.ref_area, "span_ref": d.span_ref, "moment_ref": tuple(d.moment_ref3),
                        "symmetric": d.symmetric})
        else:
            out["moment_ref"] = tuple(d.moment_ref)
        return out

    def summary(self) -> dict[str, Any]:
        """Kind, dimension, size, parts and references as a flat dict."""
        d = self.raw
        out: dict[str, Any] = {"name": d.name, "id": d.id, "kind": d.kind, "dimension": d.dimension}
        if d.kind == "wing" and d.wing is not None:
            from .geometry3d import planform
            pf = planform(d.wing)
            out.update({"sections": len(d.wing.sections), "span": pf["span"], "area": pf["area"],
                        "aspect_ratio": pf["aspect_ratio"], "mac": pf["mac"], "taper": pf["taper"],
                        "sweep_c4_deg": pf["sweep_c4_deg"]})
        elif d.dimension == 3 and d.meshes:
            x0, x1, y0, y1, z0, z1 = d.bounds3()
            out.update({"parts": len(d.meshes), "triangles": int(sum(len(m.faces) for m in d.meshes)),
                        "length": x1 - x0, "width": y1 - y0, "height": z1 - z0})
        elif d.bodies:
            x0, x1, y0, y1 = d.bounds()
            out.update({"bodies": len(d.bodies), "length": x1 - x0, "height": y1 - y0})
        out.update(self.references)
        heated = [b for b in list(d.bodies) + list(d.meshes) if getattr(b, "thermal", "adiabatic") != "adiabatic"]
        if heated:
            out["thermal"] = {b.name: (b.thermal, b.thermal_value) for b in heated}
        if (d.source or {}).get("type") == "case":
            out["case"] = {k: v for k, v in d.source.items() if k != "type"}
        out["simulations"] = len(self.project.raw.simulations_of(d.id))
        return out

    def simulations(self) -> list[M.Simulation]:
        """This design's simulations (project order)."""
        return self.project.raw.simulations_of(self.id)

    def results(self) -> list[Result]:
        """Stored results of this design's simulations."""
        return [r for s in self.simulations() if (r := self.project.result(s.id)) is not None]

    @_api
    def check(self) -> list[str]:
        """The Geometry page's diagnostics (thin trailing edges, holes, gaps, references...)."""
        d = self.raw
        if d.dimension == 3:
            from .geometry3d import check_design3d
            return list(check_design3d(d))
        from .geometry import check_design
        return list(check_design(d))

    @_api
    def plot(self, path: str | os.PathLike | None = None, *, style: str = "print", alpha: float = 0.0) -> Any:
        """The outline drawing (2D) or the grey iso/side/front/top views (3D)."""
        from .viz import figures as F
        d = self.raw
        if d.dimension == 3:
            from .report import content3d as C3
            from .simulation.wing import airfoil_resolver
            fig = C3.figure_geometry3d(d, airfoil_coords=airfoil_resolver(self.project.raw), style=style)
        else:
            fig = F.fig_geometry(d, alpha_deg=alpha, style=style)
        if path is None:
            return fig
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return F.save_figure(fig, p)

    @_api
    def export(self, path: str | os.PathLike, *, units: str = "m") -> Path:
        """Write the geometry: 2D ``.svg`` ``.dxf`` ``.dat`` (Selig airfoil); 3D ``.stl`` ``.obj``
        (a wing is lofted). ``units`` "m" or "mm" for meshes."""
        from . import export as X
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()
        d = self.raw
        if d.dimension == 3:
            meshes = list(d.meshes)
            if not meshes and d.wing is not None:
                from .geometry3d import loft
                from .simulation.wing import airfoil_resolver
                meshes = [loft(d.wing, airfoil_coords=airfoil_resolver(self.project.raw), name=d.name)]
            if ext == ".stl":
                return X.mesh_stl(meshes, p, units=units)
            if ext == ".obj":
                return X.mesh_obj(meshes, p, units=units)
            raise InputError(f"A 3D design exports as .stl or .obj, not {ext!r}.")
        if ext == ".svg":
            return X.design_svg(d, p)
        if ext == ".dxf":
            return X.design_dxf(d, p)
        if ext == ".dat":
            return X.design_dat(d, p)
        raise InputError(f"A 2D design exports as .svg, .dxf or .dat, not {ext!r}.")

    @_api
    def rename(self, name: str) -> Design:
        """Give the design a new name (recorded in the project)."""
        name = str(name).strip()
        if not name:
            raise InputError("A design name cannot be empty.")
        d = copy.deepcopy(self.raw)
        d.name = name
        self.project._edit("update_design", d)
        return self

    @_api
    def copy(self, name: str | None = None) -> Design:
        """A copy of this design (new id, geometry and references included) added to the project."""
        d = copy.deepcopy(self.raw)
        d.id = M.new_id()
        d.created = M.now_iso()
        d.name = _unique(name or f"{d.name} (copy)", (x.name for x in self.project.raw.designs.values()))
        self.project._edit("add_design", d)
        return Design(self.project, d)

    @_api
    def transform(self, *, scale: float | None = None, length: float | None = None,
                  rotate_deg: Sequence[float] | None = None, clearance: float | None = None) -> Design:
        """Scale and/or rotate a 3D mesh design (in place, recorded in the project).

        ``scale`` multiplies every size; ``length`` scales uniformly to that
        overall length [m]; ``rotate_deg`` = (rx, ry, rz) rotates about x, y, z (in
        that order, right-handed); ``clearance`` [m] sets the ground clearance of a
        vehicle. The design is then placed and its references re-derived by the
        importer's rules (vehicles: nose at x = 0, centred, on the road; others:
        centred), exactly as after an edit on the Geometry page.
        """
        d = self.raw
        if d.dimension != 3 or not d.meshes:
            raise InputError("transform() is for 3D mesh designs; build a 2D design at its size instead "
                             "(add_naca(chord=...), add_template(length=...)).")
        from .geometry3d import design_from_meshes, transform
        meshes = [copy.deepcopy(m) for m in d.meshes]
        if rotate_deg is not None:
            r = tuple(float(a) for a in rotate_deg)
            if len(r) != 3:
                raise InputError("rotate_deg must be (rx, ry, rz) in degrees.")
            meshes = transform(meshes, rotate_deg=r)
        k = 1.0 if scale is None else _float("scale", scale, positive=True, allow_zero=False)
        if k != 1.0:
            meshes = transform(meshes, scale=k)
        src = dict(d.source or {})
        on_road = bool(src.get("on_road", d.kind == "vehicle"))
        if clearance is None and on_road:
            body = [m for m in d.meshes if m.role != "wheel"] or d.meshes
            clearance = max(0.0, min(float(m.vertices[:, 2].min()) for m in body)) * k
        new = design_from_meshes(meshes, d.name, d.kind, length=length, units=None, on_road=on_road,
                                 clearance=clearance, fix_winding=False, source=src)
        new.id, new.created, new.notes = d.id, d.created, d.notes
        if d.symmetric != new.symmetric and "symmetric" in set(src.get("manual_refs", [])):
            new.symmetric = d.symmetric
        self.project._edit("update_design", new)
        return self

    @_api
    def set_thermal(self, mode: str, value: float = 0.0, *, body: str | None = None) -> Design:
        """Give the design's bodies a thermal condition (in place, recorded in the project).

        ``mode`` "temperature" (``value`` in °C: a heated or cooled wall),
        "heat_flux" (W/m², positive into the fluid) or "adiabatic". ``body`` names
        one body (2D outline or 3D mesh); default every body. The condition takes
        effect in runs with heat transfer on (``run_cfd(..., thermal=True)``).
        Use :meth:`copy` first to keep an unheated original.
        """
        if mode not in M.THERMAL_MODES:
            raise InputError(f"mode must be one of {', '.join(M.THERMAL_MODES)}, got {mode!r}.")
        v = _float("value", value)
        if mode == "temperature" and not v > -273.15:
            raise InputError(f"{v:g} °C is below absolute zero.")
        d = copy.deepcopy(self.raw)
        parts = list(d.bodies) + list(d.meshes)
        if not parts:
            raise InputError(f"{d.name!r} has no bodies to heat (heat a domain wall with walls= instead).")
        chosen = [b for b in parts if body is None or b.name == body]
        if not chosen:
            raise NotFoundError(f"No body {body!r} in {d.name!r}; bodies: {', '.join(b.name for b in parts)}.")
        for b in chosen:
            b.thermal, b.thermal_value = mode, (v if mode != "adiabatic" else 0.0)
        self.project._edit("update_design", d)
        return self

    def __repr__(self) -> str:
        d = self.raw
        tag = "Wing" if d.kind == "wing" else ("3D" if d.dimension == 3 else "2D")
        return f"<Design {d.name!r} ({tag} {d.kind})>"


# ===========================================================================
# the project
# ===========================================================================

_TEMP_DIRS: list[Path] = []


def _cleanup_temp() -> None:
    for p in _TEMP_DIRS:
        shutil.rmtree(p, ignore_errors=True)


atexit.register(_cleanup_temp)


class Project:
    """An Aerolume project: its designs, simulations and results, recorded on disk.

    Build one with :func:`open_project` / :func:`new_project` (or
    ``al.Project(path)``). Inside the app the Script console binds ``project``
    to the open project. ``raw`` is the :class:`aerolume.project.Project` store.
    Use it as a context manager to save on exit (every change is saved at once
    anyway).
    """

    def __init__(self, store: Any, *, host: Host | None = None) -> None:
        if isinstance(store, (str, os.PathLike)):
            from .project import Project as Store
            try:
                store = Store.open(store)
            except Exception as exc:
                raise _translate(exc) from exc
        self.raw = store
        self._host = host or _DIRECT_HOST

    # -- basics ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return Path(self.raw.path)

    @property
    def name(self) -> str:
        return self.raw.name

    @property
    def host(self) -> Host:
        return self._host

    def __repr__(self) -> str:
        return (f"<Project {self.name!r}: {_count(len(self.raw.designs), 'design')}, "
                f"{_count(len(self.raw.simulations), 'simulation')}>")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.save()

    @_api
    def save(self) -> None:
        """Write anything not yet on disk (changes made through the API are saved at once)."""
        self._host.check(self.raw)
        self._host.call(self.raw.save)

    def _edit(self, op: str, *args: Any) -> None:
        self._host.check(self.raw)
        getattr(self._host, op)(self.raw, *args)

    # -- lookup ------------------------------------------------------------------
    @property
    def designs(self) -> list[Design]:
        """All designs, in project order."""
        return [Design(self, d) for d in list(self.raw.designs.values())]

    @property
    def simulations(self) -> list[M.Simulation]:
        """All simulations, in project order (model objects; change them through the API)."""
        return list(self.raw.simulations.values())

    def _design(self, ref: Any) -> M.Design:
        if isinstance(ref, Design):
            if ref.project.raw is not self.raw:
                raise InputError(f"Design {ref.name!r} belongs to another project.")
            return ref.raw
        if isinstance(ref, M.Design):
            if ref.id in self.raw.designs:
                return self.raw.designs[ref.id]
            raise NotFoundError(f"Design {ref.name!r} is not in this project; add it with add_design().")
        if isinstance(ref, str):
            designs = dict(self.raw.designs)
            if ref in designs:
                return designs[ref]
            key = ref.strip().lower()
            hits = [d for d in designs.values() if d.name.strip().lower() == key]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise NotFoundError(f"Several designs are named {ref!r}; use the id: "
                                    + ", ".join(d.id for d in hits))
            starts = [d for d in designs.values() if d.id.startswith(ref)]
            if len(starts) == 1:
                return starts[0]
            names = ", ".join(repr(d.name) for d in designs.values()) or "none"
            raise NotFoundError(f"No design {ref!r} in this project. Designs: {names}.")
        raise InputError(f"Expected a design (Design, name or id), got {type(ref).__name__}.")

    @_api
    def design(self, ref: Any) -> Design:
        """A design by name (case-insensitive), id or unique id prefix."""
        return Design(self, self._design(ref))

    def _simulation(self, ref: Any) -> M.Simulation:
        if isinstance(ref, Result):
            ref = ref.sim_id
        if isinstance(ref, M.Simulation):
            ref = ref.id
        if not isinstance(ref, str):
            raise InputError(f"Expected a simulation (name, id or Result), got {type(ref).__name__}.")
        sims = dict(self.raw.simulations)
        if ref in sims:
            return sims[ref]
        key = ref.strip().lower()
        hits = [s for s in sims.values() if s.name.strip().lower() == key]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise NotFoundError(f"Several simulations are named {ref!r}; use the id: "
                                + ", ".join(f"{s.id} ({self.raw.designs[s.design_id].name if s.design_id in self.raw.designs else '?'})"
                                            for s in hits))
        starts = [s for s in sims.values() if s.id.startswith(ref)]
        if len(starts) == 1:
            return starts[0]
        names = ", ".join(repr(s.name) for s in sims.values()) or "none"
        raise NotFoundError(f"No simulation {ref!r} in this project. Simulations: {names}.")

    @_api
    def simulation(self, ref: Any) -> M.Simulation:
        """A simulation by name (case-insensitive), id, unique id prefix or Result."""
        return self._simulation(ref)

    @_api
    def result(self, ref: Any) -> Result | None:
        """The stored result of a simulation (None when it has none yet)."""
        sim = self._simulation(ref)
        raw = self.raw.load_result(sim.id)
        return None if raw is None else Result(self, sim, raw)

    @_api
    def results(self, design: Any = None) -> list[Result]:
        """Every stored result (of one design when given), in project order."""
        did = self._design(design).id if design is not None else None
        out = []
        for s in list(self.raw.simulations.values()):
            if (did is None or s.design_id == did) and self.raw.has_result(s.id):
                raw = self.raw.load_result(s.id)
                if raw is not None:
                    out.append(Result(self, s, raw))
        return out

    def summary(self) -> dict[str, Any]:
        """Designs and simulations with their status, as plain data (what an assistant lists)."""
        designs = []
        for d in list(self.raw.designs.values()):
            sims = []
            for s in self.raw.simulations_of(d.id):
                info: dict[str, Any] = {"name": s.name, "id": s.id, "analysis": s.analysis, "status": s.status,
                                        "has_result": self.raw.has_result(s.id)}
                try:
                    rs = self.raw.result_summary(s.id) if info["has_result"] else None
                except Exception:  # noqa: BLE001 - a damaged result costs its summary only
                    rs = None
                if rs:
                    for k in ("cd", "cl"):
                        v = rs.get(k)
                        if isinstance(v, dict) and _finite(v.get("mean")):
                            info[k] = v["mean"]
                    info["converged"] = rs.get("converged")
                sims.append(info)
            designs.append({"name": d.name, "id": d.id, "kind": d.kind, "dimension": d.dimension, "simulations": sims})
        return {"name": self.name, "path": str(self.path), "designs": designs}

    # -- adding designs ----------------------------------------------------------
    def _add(self, d: M.Design, *, reuse: bool = True) -> Design:
        """Add ``d`` -- or return the existing design with the same name and geometry.

        An existing design of the same name and shape that differs only in its
        bodies' heating and has no simulations is reused too, with the heating of
        ``d``: the pattern ``d = p.add_sphere(name="Hot sphere"); d.set_thermal(...)``
        heats the design after adding it, so a second run of the same script found
        the heated original "different" and added "Hot sphere 2" (critic round 3).
        Nothing was computed with the unused design, so resetting it loses nothing.
        """
        existing = list(self.raw.designs.values())
        if reuse:
            key = _geometry_key(d)
            for e in existing:
                if e.name == d.name and e.dimension == d.dimension and _geometry_key(e) == key:
                    return Design(self, e)
            shape = _geometry_key(d, thermal=False)
            for e in existing:
                if (e.name == d.name and e.dimension == d.dimension and not self.raw.simulations_of(e.id)
                        and _geometry_key(e, thermal=False) == shape):
                    same = copy.deepcopy(e)
                    for mine, theirs in zip(list(same.bodies) + list(same.meshes), list(d.bodies) + list(d.meshes),
                                            strict=False):
                        mine.thermal = getattr(theirs, "thermal", "adiabatic")
                        mine.thermal_value = getattr(theirs, "thermal_value", 0.0)
                    self._edit("update_design", same)
                    return Design(self, self.raw.designs.get(e.id, same))
        d.name = _unique(d.name, (e.name for e in existing))
        self._edit("add_design", d)
        return Design(self, d)

    @_api
    def add_design(self, design: M.Design, *, reuse: bool = True) -> Design:
        """Add an ``aerolume.model.Design`` built by hand (or by the geometry modules)."""
        if not isinstance(design, M.Design):
            raise InputError(f"add_design() takes an aerolume.model.Design, got {type(design).__name__}.")
        d = copy.deepcopy(design)
        if d.id in self.raw.designs:
            d.id = M.new_id()
        return self._add(d, reuse=reuse)

    @_api
    def add_naca(self, code: str, chord: float = 1.0, *, name: str | None = None, sharp_te: bool = True,
                 reuse: bool = True) -> Design:
        """A NACA 4- or 5-digit airfoil (e.g. "2412", "23012") of ``chord`` metres.

        Built like the Geometry page's NACA entry (the "naca" template: 240
        points, sharp trailing edge unless ``sharp_te=False``), so it can be edited
        there afterwards.
        """
        from .geometry import templates
        code = str(code).strip().upper().replace("NACA", "").strip()
        if not (code.isdigit() and len(code) in (4, 5)):
            raise InputError(f"{code!r} is not a NACA 4- or 5-digit code (e.g. '2412', '23012').")
        d = templates.make("naca", naca=code, chord=_float("chord", chord, positive=True, allow_zero=False),
                           closed_te=bool(sharp_te))
        if name:
            d.name = str(name)
            d.bodies[0].name = str(name)
        return self._add(d, reuse=reuse)

    @_api
    def add_airfoil(self, source: str | os.PathLike | np.ndarray, chord: float = 1.0, *, name: str | None = None,
                    reuse: bool = True) -> Design:
        """An airfoil from a Selig / Lednicer ``.dat`` file or an (N, 2) coordinate array."""
        from .geometry import design_from_airfoil, read_airfoil_dat
        c = _float("chord", chord, positive=True, allow_zero=False)
        if isinstance(source, np.ndarray):
            d = design_from_airfoil(source, name or "Airfoil", chord=c, source={"type": "coordinates"})
        else:
            p = Path(source)
            if not p.is_file():
                raise NotFoundError(f"No airfoil file at {p}.")
            nm, coords = read_airfoil_dat(p)
            d = design_from_airfoil(coords, name or nm or p.stem, chord=c, source=file_source(p, "dat"))
        return self._add(d, reuse=reuse)

    @_api
    def add_template(self, template: str, *, name: str | None = None, reuse: bool = True, **params: Any) -> Design:
        """A shape from the template gallery: "sedan", "hatchback", "fastback", "suv", "pickup",
        "van", "box_truck", "ahmed", "front_wing", "naca", "cylinder", "square", "ellipse",
        "flat_plate", "rounded_rectangle" (2D). ``params`` override the template's
        parameters (SI units, degrees), e.g. ``add_template("ahmed", slant_angle=30)``.
        Without ``name`` the design is named after the template and the changed parameters."""
        from .geometry import templates
        try:
            info = templates.get_template(template)
        except KeyError as exc:
            raise NotFoundError(str(exc.args[0])) from None
        d = templates.make(template, **params)
        if name:
            d.name = str(name)
        elif params:
            specs = {s.name: s for s in info.params}
            bits = []
            for k, v in params.items():
                spec = specs.get(k)
                lab = (spec.label if spec else k).lower()
                if spec is not None and spec.unit_kind == "angle":
                    bits.append(f"{lab} {_num(float(v))}°")
                elif spec is not None and spec.unit_kind == "length":
                    bits.append(f"{lab} {_num(float(v))} m")
                else:
                    bits.append(f"{lab} {v}")
            d.name = f"{info.label or d.name} · {', '.join(bits)}"
        return self._add(d, reuse=reuse)

    @_api
    def import_geometry(self, path: str | os.PathLike, *, kind: str | None = None, length: float | None = None,
                        name: str | None = None, reuse: bool = True) -> Design:
        """A 2D design from an outline file: SVG, DXF, image silhouette, point CSV or airfoil .dat.

        ``kind`` "airfoil" | "vehicle" | "generic" (default: the importer's guess);
        ``length`` [m] the real length (needed unless the file is in metres).
        """
        from .geometry import design_from_airfoil, design_from_polygons, import_file
        p = Path(path)
        if not p.is_file():
            raise NotFoundError(f"No file at {p}.")
        res = import_file(p)
        kind = kind or res.suggested_kind
        if kind not in M.KINDS_2D:
            raise InputError(f"kind must be one of {', '.join(M.KINDS_2D)}, got {kind!r}.")
        src = {**file_source(p), **dict(res.source), "import_notes": list(res.notes)}
        if kind == "airfoil" and len(res.polygons) == 1:
            d = design_from_airfoil(res.polygons[0], name or res.name, chord=length or 1.0, source=src)
        else:
            if length is None and res.units != "m":
                raise InputError(f"{p.name} is in {res.units}, not metres: give its real length= in metres.")
            d = design_from_polygons(res.polygons, name or res.name, kind, length=length)
            d.source = {**src, **(d.source or {})}
        return self._add(d, reuse=reuse)

    @_api
    def import_mesh(self, path: str | os.PathLike, *, units: str | float | None = "auto", kind: str = "vehicle",
                    on_road: bool | None = None, length: float | None = None, clearance: float | None = None,
                    name: str | None = None, reuse: bool = True) -> Design:
        """A 3D design from an STL (ASCII or binary), OBJ or PLY file; holes are tolerated.

        ``units``: "auto" (guess from the size, default), "m", "cm", "mm", "in",
        "ft", a metres-per-unit number, or None to keep the file's numbers as
        metres. ``kind`` "vehicle" (nose at x = 0, on the road) or "generic"
        (free flight); ``length`` [m] scales to that length. References (frontal
        area, length, width, symmetry) are measured as the import dialog does.
        ``design.source`` records the file's name, format, full path, size and
        SHA-256 (:func:`file_source`), so the Mesh inspector shows where the
        geometry came from and a report can prove which file it was.
        """
        from .geometry3d import design_from_meshes, read_mesh
        p = Path(path)
        if not p.is_file():
            raise NotFoundError(f"No mesh file at {p}.")
        if kind not in ("vehicle", "generic"):
            raise InputError("kind must be 'vehicle' or 'generic' for a mesh (wings: add_wing()).")
        notes: list[str] = []
        meshes = read_mesh(p, notes=notes)
        d = design_from_meshes(meshes, name or p.stem, kind=kind, length=length, units=units, on_road=on_road,
                               clearance=clearance, notes=notes, source=file_source(p))
        return self._add(d, reuse=reuse)

    @_api
    def add_primitive(self, shape: str, *, name: str | None = None, reuse: bool = True, **size: Any) -> Design:
        """A basic 3D body: "sphere" (``d``), "box" (``lx``, ``ly``, ``lz``) or "cylinder"
        (``d``, ``length``, ``axis``), metres; a generic free-flight design."""
        from . import geometry3d as g3
        defaults = {"sphere": ("Sphere", {"d": 0.2}), "box": ("Box", {"lx": 0.5, "ly": 0.3, "lz": 0.3}),
                    "cylinder": ("Cylinder", {"d": 0.2, "length": 0.6, "axis": "y"})}
        if shape not in defaults:
            raise InputError(f"shape must be one of {', '.join(defaults)}, got {shape!r}.")
        label, kw = defaults[shape]
        kw = {**kw, **size}
        mesh = getattr(g3, shape)(name=label, **kw)
        d = g3.design_from_meshes([mesh], name or label, "generic", source={"type": "primitive", "shape": shape,
                                                                            "size": M._jsonable(kw)})
        return self._add(d, reuse=reuse)

    @_api
    def add_sphere(self, d: float = 0.2, *, name: str | None = None, reuse: bool = True) -> Design:
        """A sphere of diameter ``d`` metres (the 3D validation case)."""
        return self.add_primitive("sphere", d=_float("d", d, positive=True, allow_zero=False),
                                  name=name or f"Sphere {_num(d)} m", reuse=reuse)

    @_api
    def add_wing(self, span: float, root_chord: float, tip_chord: float | None = None, *, sweep_deg: float = 0.0,
                 dihedral_deg: float = 0.0, twist_deg: float = 0.0, airfoil: str = "2412",
                 tip_airfoil: str | None = None, sections: Sequence[M.WingSection] | None = None,
                 name: str | None = None, reuse: bool = True) -> Design:
        """A finite wing (XFLR5-style planform), symmetric about y = 0.

        A straight-tapered wing from ``span`` (tip to tip) and ``root_chord`` /
        ``tip_chord`` [m]; ``sweep_deg`` is the quarter-chord sweep (the
        aerodynamic one, so lattice and lifting line agree), ``dihedral_deg`` the
        dihedral, ``twist_deg`` the tip twist (negative = washout), ``airfoil`` /
        ``tip_airfoil`` NACA codes or "design:<id>" of a 2D airfoil in this project.
        ``sections`` (``aerolume.model.WingSection``, root first, y >= 0) gives
        any other planform instead.
        """
        from .geometry3d import design_from_wing, tapered_wing
        if sections is not None:
            secs = [copy.deepcopy(s) for s in sections]
            if not secs:
                raise InputError("sections is empty.")
            wing = M.WingSpec(sections=secs, symmetric=True)
        else:
            # validated here for the API's wording; the planform itself comes from
            # geometry3d.tapered_wing, which the CLI and the planform editor use too, so
            # "sweep 10°" is the same wing whichever of them built it
            b = _float("span", span, positive=True, allow_zero=False)
            cr = _float("root_chord", root_chord, positive=True, allow_zero=False)
            ct = cr if tip_chord is None else _float("tip_chord", tip_chord, positive=True, allow_zero=False)
            wing = tapered_wing(b, cr, ct, sweep_deg=_float("sweep_deg", sweep_deg),
                                dihedral_deg=_float("dihedral_deg", dihedral_deg),
                                twist_deg=_float("twist_deg", twist_deg), airfoil=self._airfoil_name(airfoil),
                                tip_airfoil=self._airfoil_name(tip_airfoil or airfoil))
        if name is None:
            if sections is None:
                taper = ct / cr
                name = f"Wing · span {_num(span)} m, taper {_num(taper, 2)}"
                if sweep_deg:
                    name += f", sweep {_num(sweep_deg)}°"
            else:
                name = "Wing"
        d = design_from_wing(wing, str(name), source={"type": "wing", "script": True})
        return self._add(d, reuse=reuse)

    @_api
    def add_case(self, case: Case, *, reuse: bool = True) -> Design:
        """Add the design of a reference :class:`Case` (an empty cavity, a heated cylinder, a room)."""
        if not isinstance(case, Case):
            raise InputError(f"add_case() takes a Case (al.cavity_case(), ...), got {type(case).__name__}.")
        d = copy.deepcopy(case.design)
        d.id = M.new_id()
        return self._add(d, reuse=reuse)

    def _airfoil_name(self, airfoil: Any) -> str:
        if isinstance(airfoil, (Design, M.Design)):
            return f"design:{self._design(airfoil).id}"
        s = str(airfoil).strip()
        if s.lower().startswith("design:"):
            return s
        code = s.upper().replace("NACA", "").replace(" ", "")
        if code.isdigit() and len(code) in (4, 5):
            return code
        return f"design:{self._design(s).id}"     # a project airfoil by name

    @_api
    def remove_design(self, design: Any) -> None:
        """Delete a design and all its simulations and results (cannot be undone)."""
        d = self._design(design)
        self._edit("remove_design", d.id)

    @_api
    def remove_simulation(self, simulation: Any) -> None:
        """Delete a simulation and its result (cannot be undone)."""
        s = self._simulation(simulation)
        self._edit("remove_simulation", s.id)

    # -- simulations -------------------------------------------------------------
    def _new_sim(self, design: M.Design, analysis: str | None, settings: dict[str, Any],
                 name: str | None) -> M.Simulation:
        return _build_simulation(design, analysis, settings, self._host.defaults(), name or "")

    def _match(self, sim: M.Simulation) -> M.Simulation | None:
        key = _settings_key(sim)
        same = [s for s in list(self.raw.simulations.values())
                if s.design_id == sim.design_id and s.analysis == sim.analysis and _settings_key(s) == key]
        if not same:
            return None
        done = [s for s in same if s.status == "done" and self.raw.has_result(s.id)]
        return (done or same)[-1]

    def _resolve(self, design: Any, settings: dict[str, Any], name: str | None, *, runs: bool = True
                 ) -> tuple[M.Design, dict[str, Any], str | None, Case | None]:
        """(design, settings, name, case): a :class:`Case` is added and its settings merged under the caller's.

        The case's run is built and checked *before* its design is added: a bad
        keyword or physics this build's solver lacks used to be refused only after
        the design was recorded, which left an empty "Heated cavity" design in the
        project for every refused attempt (and a "… 2" on the next). ``runs``
        False (:meth:`add_simulation`: prepared, not run) skips the physics check,
        so a case can be set up for a build whose solver will run it later.
        """
        if isinstance(design, Case):
            merged = {**design.settings, **{k: v for k, v in settings.items() if v is not None}}
            probe = self._new_sim(design.design, "cfd", dict(merged), name)
            if runs:
                _require_physics(design.design, probe, name or self._case_name(design, probe))
            d = self._design(self.add_case(design))
            return d, merged, name, design
        return self._design(design), settings, name, None

    @staticmethod
    def _case_name(case: Case, sim: M.Simulation) -> str:
        """A case run's name: the case's own ("Heated cavity · Ra 1.0×10⁴"), plus quality and grid when not default."""
        bits = [case.name]
        if sim.cfd.quality != "draft":
            bits.append(sim.cfd.quality.capitalize())
        if sim.cfd.cells_per_ref:
            bits.append(f"{sim.cfd.cells_per_ref} cells/ref")
        return " · ".join(bits)

    @_api
    def add_simulation(self, design: Any, analysis: str | None = None, *, name: str | None = None,
                       reuse: bool = False, **settings: Any) -> M.Simulation:
        """Create a simulation without running it (it appears in the app, ready to Run).

        Same keywords as :meth:`run_cfd` / :meth:`run_polar` / :meth:`run_wing`; ``design``
        may also be a :class:`Case`. ``reuse`` returns an existing simulation of the same
        design and run settings instead of adding a second one.
        """
        d, settings, name, case = self._resolve(design, dict(settings), name, runs=False)
        sim = self._new_sim(d, "cfd" if case is not None else analysis, settings, name)
        if reuse:
            existing = self._match(sim)
            if existing is not None:
                return existing
        self._register(d, sim, name or (self._case_name(case, sim) if case is not None else None))
        return self.raw.simulations[sim.id]

    @_api
    def settings(self, simulation: Any) -> dict[str, Any]:
        """The editable settings of a simulation and their current values (the keys of :data:`SETTINGS`)."""
        sim = self._simulation(simulation)
        d = self.raw.designs.get(sim.design_id)
        out = {}
        for key, spec in SETTINGS.items():
            if sim.analysis in spec.analyses and (d is None or int(d.dimension) in spec.dims):
                out[key] = spec.get(sim)
        return out

    @_api
    def edit_simulation(self, simulation: Any, *, copy: bool | None = None, name: str | None = None,
                        **changes: Any) -> M.Simulation:
        """Change settings of a simulation (the keys of :data:`SETTINGS`), checked as a whole first.

        Every value is validated (unknown settings and out-of-range values are
        refused, not guessed or clamped) and the edited simulation is re-checked as
        a whole before anything is recorded. A simulation that has a result is
        copied rather than edited (``copy=None``, the default), so a stored result
        never sits under settings it was not computed with; ``copy=True`` always
        copies, ``copy=False`` edits in place and is refused when there is a result.
        A simulation with an automatic name is renamed after its new settings.
        Returns the edited (or new) simulation.
        """
        import copy as _copy
        sim0 = self._simulation(simulation)
        if sim0.status in ("running", "queued"):
            raise InputError(f"“{sim0.name}” is {sim0.status}; it can be changed when it has finished.")
        d = self._design(sim0.design_id)
        if not changes and name is None:
            raise InputError("edit_simulation() needs at least one setting to change, e.g. quality='standard'.")
        new = _copy.deepcopy(sim0)
        for key, value in changes.items():
            spec = setting_for(key, new, d)
            spec.set(new, spec.check(value))
        if new.analysis in ("polar", "wing"):
            part = _sweep_part(new)
            if part.alpha_end < part.alpha_start:
                raise InputError("The last angle must not be below the first.")
            if (part.alpha_end - part.alpha_start) / part.alpha_step > 200:
                raise InputError("At most 200 angles per sweep: use a larger angle step.")
        if new.analysis == "cfd":
            if new.cfd.domain_mode == "closed" and new.ground.enabled:
                raise InputError("A closed domain has walls on every side; it cannot have a moving road.")
            if not new.thermal.enabled and _heated_bodies(d):
                raise InputError(f"{d.name!r} has heated bodies: heat transfer cannot be switched off for it.")
            if not new.thermal.enabled and (new.heat_sources or any(p.temperature is not None for p in new.ports)):
                raise InputError("This simulation has heat sources or heated jets: heat transfer cannot be "
                                 "switched off for it.")
            if new.cfd.reynolds_mode == "capped" and not new.cfd.reynolds_cap:
                from .simulation.defaults import default_reynolds_cap
                new.cfd.reynolds_cap = default_reynolds_cap(d)
        return self._store_edit(sim0, new, d, copy=copy, name=name)

    def _store_edit(self, sim0: M.Simulation, new: M.Simulation, d: M.Design, *, copy: bool | None,
                    name: str | None) -> M.Simulation:
        """Record ``new`` (an edited deep copy of ``sim0``): in place, or as a copy when ``sim0`` has a result.

        Shared by :meth:`edit_simulation` and the placement calls (:meth:`add_emitter`, ...),
        so a stored result never sits under settings it was not computed with, whichever
        call changed them, and an automatic name follows the new settings.
        """
        try:
            M.Simulation.from_dict(new.to_dict())          # the model's own checks on the whole simulation
        except (TypeError, ValueError) as exc:
            raise InputError(f"The edited simulation is not valid: {exc}") from None
        has_result = self.raw.has_result(sim0.id)
        make_copy = has_result if copy is None else bool(copy)
        if has_result and not make_copy:
            raise InputError(f"“{sim0.name}” has a result: editing it in place would leave a result that no longer "
                             "matches its settings. Pass copy=True (the default does) to make an edited copy.")
        import re as _re
        stem = _re.sub(r" \d+$", "", sim0.name)          # without the " 2" that kept a name unique
        auto = stem == _default_name(d, sim0)
        taken = [s.name for s in self.raw.simulations_of(d.id) if s.id != sim0.id or make_copy]
        if name is not None:
            new.name = _unique(str(name).strip() or sim0.name, taken)
        elif auto:
            new.name = _unique(_default_name(d, new), taken)
        elif make_copy:
            new.name = _unique(f"{stem} (edited)", taken)
        if make_copy:
            new.id = M.new_id()
            new.created = M.now_iso()
            new.status, new.finished, new.error = "new", None, None
            self._edit("add_simulation", new)
        else:
            if new.status in ("failed", "cancelled"):
                new.status, new.error = "new", None
            self._edit("update_simulation", new)
        return self.raw.simulations[new.id]

    # -- placed physics on a simulation that is already in the project ---------------------------
    @_api
    def add_emitter(self, simulation: Any, emitter: M.Emitter | Mapping[str, Any] | None = None, *,
                    copy: bool | None = None, **fields: Any) -> M.Simulation:
        """Add a smoke (dye) or tracer-particle emitter to a CFD simulation; returns the simulation.

        ``emitter`` is an :class:`Emitter` or a dict of its fields, or the fields are
        given as keywords: ``p.add_emitter(sim, kind="rake", position=(-1, 0.2),
        end=(-1, 0.8), count=8, color="#f0a040")``. Checked like ``run_cfd(emitters=)``
        together with the emitters the simulation already has (so the four-colour limit
        counts them too). Like :meth:`edit_simulation`, a simulation that has a result is
        copied (``copy=None``), because its stored fields hold no smoke of the new
        emitter; ``copy=False`` edits in place and is refused when there is a result.
        Emitters never change the flow (DESIGN.md 10.1): the copy's forces will be the
        same, its pictures will show the smoke.
        """
        return self._place(simulation, "emitters", emitter, fields, copy=copy)

    @_api
    def add_fan(self, simulation: Any, fan: M.Fan | Mapping[str, Any] | None = None, *,
                copy: bool | None = None, **fields: Any) -> M.Simulation:
        """Add an actuator-disc fan (:class:`Fan`, a dict, or its fields as keywords) to a CFD simulation.

        ``p.add_fan(sim, position=(1, 1.5), direction=(1, 0), diameter=0.5, mode="thrust",
        value=2)`` (N, or N per metre of span in 2D). Checked like ``run_cfd(fans=)``;
        a simulation with a result is copied (see :meth:`add_emitter`).
        """
        return self._place(simulation, "fans", fan, fields, copy=copy)

    @_api
    def add_heat_source(self, simulation: Any, source: M.HeatSource | Mapping[str, Any] | None = None, *,
                        copy: bool | None = None, **fields: Any) -> M.Simulation:
        """Add a volumetric heater (:class:`HeatSource`: W in 3D, W/m in 2D; negative cools) to a CFD simulation.

        The simulation needs heat transfer on (``p.edit_simulation(sim, thermal=True)``
        first), since a heater in a flow without an energy equation would do nothing.
        """
        return self._place(simulation, "heat_sources", source, fields, copy=copy)

    @_api
    def add_port(self, simulation: Any, port: M.Port | Mapping[str, Any] | None = None, *,
                 copy: bool | None = None, **fields: Any) -> M.Simulation:
        """Add a jet (blows in) or a vent (sucks out) to a CFD simulation (:class:`Port`, a dict or keywords).

        ``p.add_port(sim, kind="jet", position=(0, 0.8), direction=(0, -1), size=0.05,
        speed=3)``; a jet ``temperature`` needs heat transfer on the simulation.
        """
        return self._place(simulation, "ports", port, fields, copy=copy)

    def _place(self, simulation: Any, kind: str, item: Any, fields: dict[str, Any], *,
               copy: bool | None) -> M.Simulation:
        """Append placed-physics objects to a copy of a simulation, check the whole list, and record it.

        The same checks as the ``run_cfd`` keywords (:func:`_apply_physics`) run on
        the existing objects plus the new one, so a placement never records a
        simulation that a run with the same keywords would have refused.
        """
        import copy as _copy
        cls = {"emitters": M.Emitter, "fans": M.Fan, "heat_sources": M.HeatSource, "ports": M.Port}[kind]
        if item is not None and fields:
            raise InputError(f"Give the {cls.__name__} either as an object (or dict) or as keywords, not both.")
        if item is None and not fields:
            raise InputError(f"Describe the {cls.__name__}: pass an al.{cls.__name__}, a dict, or its fields as "
                             f"keywords ({', '.join(f.name for f in dataclasses.fields(cls))}).")
        added = _items(kind, item if item is not None else dict(fields), cls)
        sim0 = self._simulation(simulation)
        if sim0.analysis != "cfd":
            raise InputError(f"{cls.__name__}s belong to CFD simulations; “{sim0.name}” is a {sim0.analysis} "
                             "analysis.")
        if sim0.status in ("running", "queued"):
            raise InputError(f"“{sim0.name}” is {sim0.status}; it can be changed when it has finished.")
        d = self._design(sim0.design_id)
        dim = int(d.dimension)
        new = _copy.deepcopy(sim0)
        objs = list(getattr(new, kind)) + added
        setattr(new, kind, objs)
        if kind == "emitters":
            _check_emitters(objs, dim)
        elif kind == "fans":
            _check_fans(objs, dim)
        elif kind == "heat_sources":
            if not new.thermal.enabled:
                raise InputError(f"A heat source releases heat, which needs heat transfer: switch it on first with "
                                 f"edit_simulation(“{sim0.name}”, thermal=True).")
            _check_heat_sources(objs, dim)
        else:
            _check_ports(objs, dim, new.thermal.enabled)
        _check_in_box(new)
        return self._store_edit(sim0, new, d, copy=copy, name=None)

    def _register(self, d: M.Design, sim: M.Simulation, name: str | None) -> None:
        taken = [s.name for s in self.raw.simulations_of(d.id)]
        sim.name = _unique(str(name) if name else _default_name(d, sim), taken)
        self._edit("add_simulation", sim)

    @_api
    def plan_of(self, simulation: Any) -> Plan:
        """The plan of a simulation already in the project (what :meth:`run` would compute), not run."""
        sim = self._simulation(simulation)
        return _make_plan(self._design(sim.design_id), sim, self.raw, self._host)

    @_api
    def plan(self, design: Any, analysis: str | None = None, **settings: Any) -> Plan:
        """Plan a run without running or recording it (see :func:`plan`); ``design`` may be a :class:`Case`
        (its design is then added to the project, as a run would)."""
        d, settings, _name, case = self._resolve(design, dict(settings), None)
        sim = self._new_sim(d, "cfd" if case is not None else analysis, settings, None)
        sim.name = self._case_name(case, sim) if case is not None else _default_name(d, sim)
        # a plan of physics this build cannot run would describe a run that cannot happen (and the planner
        # would fail on, e.g., an empty cavity with a less helpful message): the same refusal as a run
        _require_physics(d, sim, sim.name)
        return _make_plan(d, sim, self.raw, self._host)

    @_api
    def run_cfd(self, design: Any, *, speed: float | None = None, alpha: float = 0.0, yaw: float = 0.0,
                quality: str | None = None, ground: bool | None = None, name: str | None = None,
                rerun: bool = False, progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                verbose: bool | None = None, **settings: Any) -> Result:
        """One Navier-Stokes run (2D or 3D, by the design) and its :class:`Result`.

        ``speed`` [m/s] (default 30) or ``re`` (the Reynolds number on the design's
        reference length); ``alpha`` angle of attack / pitch [°]; ``yaw`` sideslip [°]
        (3D, wind from the left for positive yaw); ``quality`` "draft" (the GUI's
        start), "standard" or "fine"; ``ground`` road on/off (default: vehicles on a
        moving road). Further keywords: ``fluid`` ("air", "water" or a Fluid),
        ``altitude`` (ISA air), ``moving_ground``, ``rotate_wheels``, ``device``
        ("auto", "cuda", "cpu"), ``precision``, ``sensitivity`` ("cd", "cl", "ld":
        also compute the shape-sensitivity map), ``symmetry`` ("auto" | "full", 3D),
        ``reynolds`` ("real" | "capped") with ``re_cap``, ``duration`` and
        ``average_from`` [L/U], ``cells_per_ref``, ``turbulence``, and any other
        ``aerolume.model.CFDSettings`` field.

        Airfoils above Re 1e5 start, as in the app, in the capped-Reynolds mode
        (2D CFD cannot resolve their boundary layers): ``r.re_sim`` says what was
        simulated; the polar gives the real-Re answer.

        **Placed physics** (DESIGN.md 10; every one is checked before the run is
        recorded, and a run that asks for physics this build's solver does not
        implement is refused rather than run without it, :func:`supported_physics`):

        * ``emitters`` -- smoke (``species="dye"``) or tracer particles: a list of
          :class:`Emitter` or dicts, e.g. ``{"kind": "rake", "position": (-1, 0.2),
          "end": (-1, 0.8), "count": 8}``; up to four smoke colours;
        * ``thermal`` -- heat transfer: True, a dict (``t_ambient`` / ``ambient`` °C,
          ``buoyancy``, ``prandtl``, ``beta``, ...) or a :class:`ThermalSettings`;
          the ambient defaults to the fluid's temperature (15 °C for ISA air), and a
          stated ambient without ``fluid`` runs in air at that temperature; bodies are
          heated on the design (:meth:`Design.set_thermal`);
        * ``walls`` -- domain-side conditions ``{"left": 30.0, "bottom": ("heat_flux",
          50), "top": "adiabatic"}`` (numbers are °C; enables heat transfer);
        * ``heat_sources`` -- volumetric heaters (:class:`HeatSource`, W or W/m);
        * ``fans`` -- actuator discs (:class:`Fan`: thrust N or N/m, pressure jump Pa,
          or a target velocity m/s);
        * ``ports`` (or ``jets``) -- jets and vents (:class:`Port`);
        * ``domain`` "tunnel" (free stream from the left, default) or "closed" (walls
          on every side: cavities, rooms); ``domain_box`` the extent in metres,
          (xmin, xmax, ymin, ymax[, zmin, zmax]). A closed domain without ``speed``
          takes the speed that drives it (free-fall velocity, fan or jet speed) as
          its reference speed;
        * ``diagnostics`` (record the solver's divergence and change norms, default on)
          and ``adaptive_dt`` ("cfl" or "pid").

        ``design`` may be a :class:`Case` (:func:`cavity_case`, ...): its design is
        added and its settings used, overridden by any keyword given here.
        """
        d, merged, name, case = self._resolve(design, dict(settings, speed=speed, quality=quality, ground=ground),
                                              name)
        sim = self._new_sim(d, "cfd", dict(merged, alpha=alpha, yaw=yaw), name)
        if case is not None and not name:
            name = self._case_name(case, sim)
        return self._run_new(d, sim, name, rerun=rerun, progress=progress, cancel=cancel, verbose=verbose)

    @_api
    def run_case(self, case: Case, *, rerun: bool = False, progress: ProgressFn | None = None,
                 cancel: threading.Event | None = None, verbose: bool | None = None, **overrides: Any) -> Result:
        """Run a reference :class:`Case` (``overrides`` change any of its settings, e.g. ``quality="standard"``);
        compare with the published value with ``case.check(result)``."""
        if not isinstance(case, Case):
            raise InputError(f"run_case() takes a Case (al.cavity_case(), ...), got {type(case).__name__}.")
        return self.run_cfd(case, rerun=rerun, progress=progress, cancel=cancel, verbose=verbose, **overrides)

    @_api
    def run_polar(self, design: Any, *, re: float | None = None, alpha: Any = (-4.0, 14.0, 1.0),
                  name: str | None = None, rerun: bool = False, progress: ProgressFn | None = None,
                  cancel: threading.Event | None = None, verbose: bool | None = None, **settings: Any) -> Result:
        """Viscous-inviscid polar of a 2D airfoil (XFOIL-class, seconds).

        ``re`` the chord Reynolds number (default 1e6; or ``speed=``), ``alpha``
        = (start, end, step) degrees, ``ncrit`` (e^N, default 9), ``xtr`` forced
        transition x/c on both sides (or ``xtr_upper`` / ``xtr_lower``; 1 = free),
        ``mach`` (Karman-Tsien), ``fluid``.
        """
        d = self._design(design)
        sim = self._new_sim(d, "polar", dict(settings, re=re, alpha=alpha), name)
        return self._run_new(d, sim, name, rerun=rerun, progress=progress, cancel=cancel, verbose=verbose)

    @_api
    def run_wing(self, design: Any, *, speed: float | None = None, alpha: Any = (-4.0, 12.0, 1.0),
                 method: str = "vlm", viscous: bool = True, yaw: float = 0.0, name: str | None = None,
                 rerun: bool = False, progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                 verbose: bool | None = None, **settings: Any) -> Result:
        """Finite-wing polar of a wing design (seconds): vortex lattice ("vlm", any planform)
        or nonlinear lifting line ("llt", straight wings) with viscous strip drag from the
        section polars (``viscous=False``: induced drag only).

        ``speed`` [m/s] (default 30) sets the section Reynolds numbers, ``alpha`` =
        (start, end, step) degrees, ``yaw`` the sideslip; also ``ncrit``, ``mach``, ``fluid``.
        """
        d = self._design(design)
        sim = self._new_sim(d, "wing", dict(settings, speed=speed, alpha=alpha, method=method, viscous=viscous,
                                            yaw=yaw), name)
        return self._run_new(d, sim, name, rerun=rerun, progress=progress, cancel=cancel, verbose=verbose)

    @_api
    def run(self, simulation: Any, *, rerun: bool = False, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None, verbose: bool | None = None) -> Result:
        """Run a simulation that is already in the project (like the app's Run), or return
        its complete stored result unless ``rerun``."""
        sim = self._simulation(simulation)
        d = self._design(sim.design_id)
        if sim.status == "done" and self.raw.has_result(sim.id) and not rerun:
            return self._reuse(sim, progress, verbose)
        return self._execute(d, sim, progress=_Progress(progress, _verbose(verbose)), cancel=_cancel_event(cancel),
                             verbose=_verbose(verbose))

    def _reuse(self, sim: M.Simulation, progress: ProgressFn | _Progress | None, verbose: bool | None) -> Result:
        raw = self.raw.load_result(sim.id)
        if raw is None:
            raise NotFoundError(f"The result of {sim.name!r} disappeared.")
        r = Result(self, sim, raw, reused=True)
        hook = getattr(self._host, "result_reused", None)
        if callable(hook):
            with contextlib.suppress(Exception):     # a notification must never fail the run it reports
                hook(self.raw, sim.id)
        (progress if isinstance(progress, _Progress) else _Progress(progress, False))(
            1.0, f"Reused the stored result of “{sim.name}”")
        if _verbose(verbose):
            _say(f"Reused the stored result of “{sim.name}” (rerun=True runs it again).")
            _say(f"  {r._headline()}")
        return r

    def _run_new(self, d: M.Design, sim: M.Simulation, name: str | None, *, rerun: bool,
                 progress: ProgressFn | _Progress | None, cancel: threading.Event | None,
                 verbose: bool | None) -> Result:
        cancel = _cancel_event(cancel)
        verbose_on = _verbose(verbose)
        prog = progress if isinstance(progress, _Progress) else _Progress(progress, verbose_on)
        _check_cancel(cancel, f"running {name or _default_name(d, sim)!r}")
        # before anything is recorded: a refused run leaves no trace
        _require_physics(d, sim, name or _default_name(d, sim))
        existing = self._match(sim)
        if existing is not None:
            if existing.status == "done" and self.raw.has_result(existing.id) and not rerun:
                return self._reuse(existing, prog, verbose)
            sim = existing
        else:
            self._register(d, sim, name)
        return self._execute(d, self.raw.simulations[sim.id], progress=prog, cancel=cancel, verbose=verbose_on)

    def _execute(self, d: M.Design, sim: M.Simulation, *, progress: _Progress, cancel: threading.Event | None,
                 verbose: bool) -> Result:
        _check_cancel(cancel, f"running {sim.name!r}")
        _require_physics(d, sim)
        if verbose:
            _say(f"Running “{sim.name}” on {d.name}")
            if sim.analysis == "cfd":
                try:
                    pl = _make_plan(d, sim, self.raw, self._host)
                    _say(f"  {pl.summary().get('Grid', '')} cells, estimated {pl.estimate or '—'}, "
                         f"Re {_re_text(pl.re_sim)} simulated ({_re_text(pl.re_real)} real)")
                    for w in pl.warnings:
                        _say(f"  ! {w}")
                except Exception as exc:  # noqa: BLE001 - the run itself reports a bad plan
                    _say(f"  (no plan: {_friendly(exc)})")
        self._host.check(self.raw)
        out = self._host.run_simulation(self.raw, sim.id, progress=progress, cancel=cancel)
        if out is None:
            out = self._run_here(sim.id, progress, cancel)
        raw, status = out
        sim_now = self.raw.simulations.get(sim.id, sim)
        result = Result(self, sim_now, raw) if raw is not None else None
        if verbose and result is not None:
            _say(f"  {result._headline()}")
            for w in result.warnings:
                _say(f"  ! {w}")
        if status == "cancelled":
            raise Cancelled(f"“{sim.name}” was stopped" + (" (its partial result is kept)." if raw is not None
                                                            else "."), result)
        if result is None:
            raise RunError(f"“{sim.name}” finished without a result.")
        return result

    def _run_here(self, sim_id: str, progress: _Progress, cancel: threading.Event | None) -> tuple[Any, str]:
        """Run a saved simulation on this thread and record status and result (the default host)."""
        store = self.raw
        sim = store.simulations[sim_id]
        design = copy.deepcopy(store.designs[sim.design_id])
        had_result = sim.status == "done" and store.has_result(sim_id)
        work = copy.deepcopy(sim)

        def record(**changes: Any) -> None:
            s = copy.deepcopy(store.simulations.get(sim_id, sim))
            for k, v in changes.items():
                setattr(s, k, v)
            self._edit("update_simulation", s)

        record(status="running", error=None)
        try:
            res = _execute_simulation(design, work, store, progress, cancel)
        except BaseException as exc:
            stopped = isinstance(exc, KeyboardInterrupt) or (cancel is not None and cancel.is_set())
            if stopped:
                record(status="done" if had_result else "cancelled", error=None)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                return (store.load_result(sim_id) if had_result else None), "cancelled"
            record(status="failed", error=_friendly(exc))
            if isinstance(exc, Exception):
                raise RunError(f"“{sim.name}” failed: {_friendly(exc)}") from exc
            raise
        if res is None:
            record(status="failed", error="The solver finished without returning a result.")
            raise RunError(f"“{sim.name}” finished without a result.")
        stopped = cancel is not None and cancel.is_set()
        if stopped and had_result:
            record(status="done", error=None)       # the GUI's rule: a stopped re-run keeps the complete result
            return store.load_result(sim_id), "cancelled"
        try:
            store.save_result(sim_id, res)
        except Exception as exc:
            # never leave the simulation "running" in the project because the disk refused a write
            record(status="failed", error=f"The result could not be saved: {_friendly(exc)}")
            raise ProjectFileError(f"The result of “{sim.name}” could not be saved: {_friendly(exc)}") from exc
        record(status="cancelled" if stopped else "done", finished=M.now_iso(), error=None)
        self._host.result_saved(store, sim_id)
        return res, ("cancelled" if stopped else "done")

    # -- sweeps ------------------------------------------------------------------
    @_api
    def sweep(self, design: Any, *, analysis: str | None = None, name: str | None = None, rerun: bool = False,
              progress: ProgressFn | None = None, cancel: threading.Event | None = None,
              verbose: bool | None = None, **params: Any) -> Sweep:
        """Run one simulation per combination of the list-valued ``params`` (a full factorial).

        ``design`` is a design -- then the swept keywords are run settings
        (``alpha=[0, 4, 8]``, ``speed=[20, 30]``, ``yaw``, ``re``, ``quality``,
        ``method``, ...) -- or a function returning a design, whose own keyword
        arguments can be swept too (geometry studies)::

            p.sweep(lambda slant: p.add_template("ahmed", slant_angle=slant),
                    slant=[15, 25, 35], speed=40)

        Lists, tuples, ranges and arrays are swept; scalars are passed to every
        run. For polars and wing sweeps ``alpha`` is the sweep's own angle range,
        never a sweep parameter. Every point is recorded (and reused on a re-run)
        like a single :meth:`run_cfd`.
        """
        cancel = _cancel_event(cancel)
        verbose_on = _verbose(verbose)
        factory = design if callable(design) and not isinstance(design, (Design, M.Design)) else None
        fixed_design = None if factory is not None else self._design(design)
        if analysis is None:
            analysis = _analyses_for(fixed_design)[0] if fixed_design is not None else None
        factory_args: set[str] = set()
        if factory is not None:
            import inspect
            try:
                sig = inspect.signature(factory)
                factory_args = {n for n, p in sig.parameters.items()
                                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
            except (TypeError, ValueError):
                factory_args = set()
        if analysis is None and factory is not None:
            # whether alpha is the run's own range (polar, wing) or a sweep parameter depends on
            # the analysis, which depends on the design: build the first one to find out (adding a
            # design is idempotent, so the first point reuses it)
            first = {k: (next(iter(v)) if isinstance(v, (list, tuple, range, np.ndarray)) and not isinstance(v, str)
                         else v) for k, v in params.items() if k in factory_args}
            analysis = _analyses_for(self._design(factory(**first)))[0]
        swept: dict[str, list[Any]] = {}
        fixed: dict[str, Any] = {}
        for k, v in params.items():
            native_alpha = k == "alpha" and analysis in ("polar", "wing")
            # lists of fans, emitters, jets, wall conditions or a domain box are one set-up, not values to sweep
            if (not native_alpha and k not in _NOT_SWEPT and isinstance(v, (list, tuple, range, np.ndarray))
                    and not isinstance(v, str)):
                vals = [x.item() if isinstance(x, np.generic) else x for x in list(np.asarray(v, dtype=object).ravel())]
                if not vals:
                    raise InputError(f"{k}= is empty.")
                swept[k] = vals
            else:
                fixed[k] = v
        if not swept:
            raise InputError("Nothing to sweep: give at least one parameter as a list, e.g. alpha=[0, 4, 8].")
        names = list(swept)
        combos = list(itertools.product(*(swept[n] for n in names)))
        total = len(combos)
        prog = _Progress(progress, verbose_on)
        done: list[tuple[dict[str, Any], Result]] = []
        for i, combo in enumerate(combos):
            vals = dict(zip(names, combo))
            label = ", ".join(f"{k} {_num(v) if isinstance(v, (int, float)) else v}" for k, v in vals.items())
            try:
                _check_cancel(cancel, f"point {i + 1} of {total} ({label})")
                kw = dict(fixed, **vals)
                if factory is not None:
                    fargs = {k: kw.pop(k) for k in list(kw) if k in factory_args}
                    made = factory(**fargs)
                    d = self._design(made)
                else:
                    d = fixed_design
                assert d is not None
                if verbose_on:
                    _say(f"Point {i + 1} of {total}: {label}")
                sim = self._new_sim(d, analysis, kw, None)
                pname = f"{name} · {label}" if name else None
                r = self._run_new(d, sim, pname, rerun=rerun, progress=prog.sub(i / total, 1.0 / total,
                                                                                f"Point {i + 1} of {total} ({label}): "),
                                  cancel=cancel, verbose=False)
            except Cancelled as exc:
                if exc.result is not None:
                    done.append((vals, exc.result))
                raise Cancelled(f"The sweep was stopped after {len(done)} of {total} points.",
                                Sweep(names, done)) from None
            done.append((vals, r))
            if verbose_on:
                _say(f"  {r._headline()}" + (" (reused)" if r.reused else ""))
        prog(1.0, f"Sweep finished: {total} points")
        return Sweep(names, done)

    @_api
    def grid_study(self, design: Any, *, cells_per_ref: Sequence[float] | None = None, refinement: float = 1.5,
                   levels: int = 3, quantity: str | Sequence[str] | None = None, rerun: bool = False,
                   progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                   verbose: bool | None = None, **settings: Any) -> GridStudy:
        """The same CFD case on three systematically refined grids, with Richardson extrapolation and the GCI.

        ``cells_per_ref`` gives the grids (cells per reference length, three
        values); otherwise they are the planner's choice for the given quality and
        that times ``refinement`` and ``refinement²`` (cost grows with the cell
        count, about ``refinement``³ per level in 2D and ⁴ in 3D, since the time step
        shrinks too). ``quantity``: "cd", "cl", "cm", "nusselt" or a summary key
        (default: "nusselt" for heat-transfer runs without forces, "cd" and "cl" for
        lifting designs, else "cd"). ``design`` may be a :class:`Case`. Every grid is
        an ordinary recorded simulation (reused on a re-run); the analysis is
        :func:`grid_convergence` (``aerolume.analysis.gci``) on the cell sizes
        actually solved, with the standard errors of the time averages, so a
        difference within the averaging noise is reported as noise, not as an order.
        """
        if int(levels) != 3:
            raise InputError("A grid study uses three grids (the GCI procedure needs three); levels must be 3.")
        r_ = _float("refinement", refinement)
        if not r_ >= 1.1:
            raise InputError(f"refinement must be at least 1.1 (Celik et al. recommend 1.3 or more), got {r_:g}.")
        settings.pop("cells_per_ref", None)
        d, _merged, _name, case = self._resolve(design, dict(settings), None)
        if cells_per_ref is None:
            pl = self.plan(case if case is not None else d.id, "cfd", **settings)
            base = float(getattr(pl.raw, "cells_per_ref", math.nan))
            if not _finite(base) or base <= 0:
                raise RunError("The planner did not report its grid resolution; give cells_per_ref=[...].")
            grids = [round(base * r_ ** k) for k in range(3)]
        else:
            grids = sorted(round(_positive("cells_per_ref", c)) for c in cells_per_ref)
            if len(grids) != 3 or len(set(grids)) != 3:
                raise InputError(f"cells_per_ref must be three different resolutions, got {list(cells_per_ref)!r}.")
        cancel = _cancel_event(cancel)
        prog = _Progress(progress, _verbose(verbose))
        results: list[Result] = []
        for i, n in enumerate(grids):
            _check_cancel(cancel, f"grid {i + 1} of 3 ({n} cells per reference length)")
            sub = prog.sub(i / 3, 1 / 3, f"Grid {i + 1} of 3 ({n} cells/ref): ")
            target: Any = case if case is not None else d.id
            level_name = None
            if case is None:            # the grid is what tells the three runs apart in the Project tree
                level_sim = self._new_sim(d, "cfd", dict(settings, cells_per_ref=n), None)
                level_name = f"{_default_name(d, level_sim)} · {n} cells/ref"
            r = self.run_cfd(target, cells_per_ref=n, rerun=rerun, progress=sub, cancel=cancel, verbose=False,
                             name=level_name, **settings)
            results.append(r)
        prog(1.0, "Grid study finished")
        return _grid_study_of(results, grids, quantity)

    @_api
    def grid_study_of(self, simulation: Any, *, cells_per_ref: Sequence[float] | None = None,
                      refinement: float = 1.5, quantity: str | Sequence[str] | None = None, rerun: bool = False,
                      progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                      verbose: bool | None = None) -> GridStudy:
        """The grid study of a CFD simulation already in the project: that simulation on three grids.

        Everything but the resolution is kept -- conditions, quality preset (and with
        it the domain), placed physics (heat transfer, walls, fans, jets, a closed
        domain) -- so a reference case prepared by :meth:`add_simulation` or the
        assistant's ``create_case``, or a run set up on the Setup page, is studied
        as it is. The coarsest grid is the simulation itself (its stored result is
        reused); the others are copies at ``refinement`` and ``refinement²`` times
        its cells per reference length ("… · 36 cells/ref"), reused on a re-run.
        ``cells_per_ref`` gives the three grids instead. See :meth:`grid_study`.
        """
        base = self._simulation(simulation)
        if base.analysis != "cfd":
            raise InputError(f"A grid study needs a CFD simulation; “{base.name}” is a {base.analysis} analysis "
                             "(polars and wings are deterministic: their resolution is the angle step).")
        d = self._design(base.design_id)
        r_ = _float("refinement", refinement)
        if not r_ >= 1.1:
            raise InputError(f"refinement must be at least 1.1 (Celik et al. recommend 1.3 or more), got {r_:g}.")
        if cells_per_ref is None:
            n0 = base.cfd.cells_per_ref
            if not n0:
                pl = self.plan_of(base.id)
                n0 = float(getattr(pl.raw, "cells_per_ref", math.nan))
                if not _finite(n0) or n0 <= 0:
                    raise RunError("The planner did not report its grid resolution; give cells_per_ref=[...].")
            grids = [round(float(n0) * r_ ** k) for k in range(3)]
            explicit = False
        else:
            grids = sorted(round(_positive("cells_per_ref", c)) for c in cells_per_ref)
            if len(grids) != 3 or len(set(grids)) != 3:
                raise InputError(f"cells_per_ref must be three different resolutions, got {list(cells_per_ref)!r}.")
            explicit = True
        cancel = _cancel_event(cancel)
        prog = _Progress(progress, _verbose(verbose))
        stem = _GRID_SUFFIX.sub("", base.name) or base.name
        results: list[Result] = []
        for i, n in enumerate(grids):
            _check_cancel(cancel, f"grid {i + 1} of 3 ({n} cells per reference length)")
            sub = prog.sub(i / 3, 1 / 3, f"Grid {i + 1} of 3 ({n} cells/ref): ")
            if i == 0 and not explicit:
                # the simulation itself is the coarse grid (with its planner-chosen or own resolution)
                results.append(self.run(base.id, rerun=rerun, progress=sub, cancel=cancel, verbose=False))
                continue
            sim = M.Simulation.from_dict(base.to_dict())
            sim.id, sim.status, sim.error, sim.finished, sim.created = M.new_id(), "new", None, None, M.now_iso()
            sim.cfd.cells_per_ref = int(n)
            results.append(self._run_new(d, sim, f"{stem} · {n} cells/ref", rerun=rerun, progress=sub, cancel=cancel,
                                         verbose=False))
        prog(1.0, "Grid study finished")
        return _grid_study_of(results, grids, quantity)

    # -- comparisons and reports ------------------------------------------------
    def _results(self, items: Sequence[Any]) -> list[Result]:
        out = []
        for it in items:
            if isinstance(it, Result):
                out.append(it)
                continue
            r = self.result(it)
            if r is None:
                raise NotFoundError(f"{self._simulation(it).name!r} has no result yet; run it first.")
            out.append(r)
        return out

    @_api
    def compare(self, results: Sequence[Any] | Sweep, *, baseline: int = 0) -> Comparison:
        """Compare results (Result objects, simulation names or ids, or a Sweep) against
        ``results[baseline]`` -- the Compare page's table, warnings included."""
        items = list(results.results if isinstance(results, Sweep) else results)
        if len(items) < 2:
            raise InputError("compare() needs at least two results.")
        rs = self._results(items)
        if not 0 <= baseline < len(rs):
            raise InputError(f"baseline must be 0..{len(rs) - 1}.")
        return Comparison(rs, baseline)

    @_api
    def report(self, results: Sequence[Any] | Result | Sweep, path: str | os.PathLike | None = None, *,
               sections: Sequence[str] | None = None, title: str | None = None, author: str | None = None,
               fmt: str | None = None) -> Path:
        """Write a PDF or HTML report (by suffix, or ``fmt``) of the given results; the
        Report page's document. Without ``path`` it goes to the project's reports folder."""
        from . import report as rep
        items = [results] if isinstance(results, Result) else list(
            results.results if isinstance(results, Sweep) else results)
        rs = self._results(items)
        if not rs:
            raise InputError("report() needs at least one result.")
        if path is None:
            stamp = time.strftime("%Y-%m-%d %H%M")
            path = self.raw.reports_dir / f"{title or self.name} {stamp}.{fmt or 'pdf'}"
        p = Path(path)
        f = (fmt or ("html" if p.suffix.lower() in (".html", ".htm") else "pdf")).lower()
        p.parent.mkdir(parents=True, exist_ok=True)
        out = rep.generate(self.raw, [r.sim_id for r in rs], p, fmt=f, sections=sections, title=title,
                           author=author)
        try:
            inside = Path(out).resolve().is_relative_to(Path(self.raw.reports_dir).resolve())
        except (OSError, ValueError):
            inside = False
        if inside:
            self._host.report_written(self.raw, Path(out))
        return Path(out)


def _execute_simulation(design: M.Design, sim: M.Simulation, store: Any, progress: ProgressFn,
                        cancel: threading.Event | None) -> Any:
    """Run one simulation synchronously through the dispatcher the app and the CLI use.

    :func:`aerolume.simulation.runner.run_simulation` sends polars, wings and
    2D / 3D CFD to their runners and resolves wing sections that name a project
    airfoil from ``store``; going through it (rather than calling the runners
    here) is what keeps a script's result identical to the GUI's for the same
    simulation.
    """
    from .simulation import runner
    return runner.run_simulation(design, sim, project=store, progress=progress, cancel=cancel)


# ===========================================================================
# opening and creating projects
# ===========================================================================

@_api
def open_project(path: str | os.PathLike, *, create: bool = False) -> Project:
    """Open an existing project folder (``Name.aerolume``); with ``create`` make it when missing."""
    from .project import Project as Store
    p = Path(path).expanduser()
    cand = p if p.exists() else p.with_name(p.name + ".aerolume")
    bound = _ctx().project
    if bound is not None and cand.exists():
        with contextlib.suppress(OSError):
            if cand.resolve() == Path(bound.path).resolve():
                return bound            # never a second store on the folder the app has open
    if not cand.exists():
        if create:
            return new_project(p)
        raise NotFoundError(f"No project at {p}. Use al.new_project({str(p)!r}) to create one.")
    store = Store.open(cand)
    proj = Project(store)
    for w in store.load_warnings:
        if _verbose(None):
            _say(f"! {w}")
    return proj


@_api
def new_project(path: str | os.PathLike | None = None, name: str | None = None, *,
                exist_ok: bool = False) -> Project:
    """Create a project folder (".aerolume" is appended when missing) and return it.

    Without ``path`` the project lives in a temporary folder that is deleted
    when Python exits -- handy for quick experiments; give a path to keep it.
    ``exist_ok`` opens the project instead when it already exists.
    """
    from .project import Project as Store
    if path is None:
        tmp = Path(tempfile.mkdtemp(prefix="aerolume-"))
        _TEMP_DIRS.append(tmp)
        path = tmp / (name or "Scratch")
    p = Path(path).expanduser()
    target = p if p.suffix.lower() == ".aerolume" else p.with_name(p.name + ".aerolume")
    if target.exists() and (target / "project.json").is_file():
        if exist_ok:
            return Project(Store.open(target))
        raise ProjectFileError(f"A project already exists at {target}; open it with al.open_project() "
                               "(or pass exist_ok=True).")
    target.parent.mkdir(parents=True, exist_ok=True)
    return Project(Store.create(target, name or p.stem.removesuffix(".aerolume")))


def __dir__() -> list[str]:
    return sorted(__all__)

