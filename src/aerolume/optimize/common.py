"""Types shared by the sensitivity and optimisation drivers.

Kept free of torch (and of the GUI) so that :func:`aerolume.optimize.optimize_airfoil`,
which needs only numpy / scipy and the panel solver, can be imported cheaply.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Cancelled", "OptimizationResult", "Progress", "ProgressFn", "check_cancel"]

ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    """Raised by the sensitivity and optimisation drivers when the caller's cancel event is set.

    The drivers stop at the next safe point (between solver chunks or optimiser
    evaluations) and raise this instead of returning a half-finished object, so
    a caller can tell "stopped by the user" from "failed". The optimisers attach
    the best result found so far as ``exc.partial`` (an :class:`OptimizationResult`
    or ``None``).
    """

    def __init__(self, message: str = "cancelled", partial: OptimizationResult | None = None):
        super().__init__(message)
        self.partial = partial


def check_cancel(cancel: threading.Event | None, partial: OptimizationResult | None = None) -> None:
    """Raise :class:`Cancelled` when ``cancel`` is set."""
    if cancel is not None and cancel.is_set():
        raise Cancelled("cancelled by the user", partial)


class Progress:
    """None-safe, clamped, monotone wrapper around a ``progress(fraction, message)`` callback.

    ``span=(a, b)`` maps the wrapped driver's 0..1 onto a sub-range of the caller's
    bar, so nested drivers (an optimiser calling the sensitivity, which develops
    the flow first) report one smooth bar instead of several that restart.
    """

    def __init__(self, fn: ProgressFn | None, span: tuple[float, float] = (0.0, 1.0)):
        self.fn = fn
        self.a, self.b = float(span[0]), float(span[1])
        self.last = -math.inf

    def __call__(self, frac: float, msg: str) -> None:
        if self.fn is None:
            return
        f = self.a + (self.b - self.a) * min(1.0, max(0.0, float(frac)))
        f = max(f, self.last)
        self.last = f
        self.fn(f, msg)

    def sub(self, lo: float, hi: float) -> Progress:
        """A child reporter covering ``[lo, hi]`` of this reporter's range."""
        return Progress(self.fn, (self.a + (self.b - self.a) * lo, self.a + (self.b - self.a) * hi))


@dataclass
class OptimizationResult:
    """Outcome of a shape optimisation (airfoil polar-solver or CFD-adjoint driven).

    ``objective`` names what was optimised ("ld", "cd", "cl"); ``initial`` and
    ``best`` hold the aerodynamic numbers of the starting and the best feasible
    shape (keys cl, cd, cm, ld, alpha, max_thickness, area where available);
    ``improvement`` is the relative gain of the objective in the *improving*
    direction (+0.08 = 8 % better: L/D up 8 %, or Cd down 8 %). ``history`` has one
    dict per optimiser iteration (objective value, coefficients, constraint
    values, evaluations used so far, seconds). ``best_coords`` is the improved
    outline (airfoils: unit chord, Selig order, leading edge at the origin; CFD
    shapes: one (N, 2) array per body in the design frame, metres); ``cst`` the
    CST parameters of the best airfoil (fitted, order 8) and ``x`` the raw design
    variables. ``converged`` says whether the optimiser met its own stopping
    test (rather than the iteration limit); ``message`` is a human-readable
    summary for the GUI.
    """
    method: str
    objective: str
    initial: dict[str, float]
    best: dict[str, float]
    improvement: float
    best_coords: np.ndarray | list[np.ndarray]
    history: list[dict] = field(default_factory=list)
    cst: object | None = None                       # CSTParams (airfoils)
    x: np.ndarray | None = None
    converged: bool = False
    message: str = ""
    evaluations: int = 0
    failed_evaluations: int = 0
    wall_seconds: float = float("nan")
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe summary (arrays become lists; CST via its own ``to_dict``)."""
        def conv(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            return v
        d = {k: conv(getattr(self, k)) for k in ("method", "objective", "initial", "best", "improvement",
                                                 "best_coords", "history", "x", "converged", "message",
                                                 "evaluations", "failed_evaluations", "wall_seconds",
                                                 "notes", "warnings")}
        d["cst"] = self.cst.to_dict() if hasattr(self.cst, "to_dict") else None
        return d
