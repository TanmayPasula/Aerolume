"""Airfoil shape optimisation: CST perturbations, the viscous panel solver and SLSQP.

:func:`optimize_airfoil` improves an airfoil for one operating point -- best
L/D (or least drag, or most lift) at a fixed angle of attack, or least drag at
a required lift coefficient -- under geometric constraints a designer actually
has: a minimum thickness, optionally no loss of cross-section area, and no
part of the section thinner than half of what it was.

Why it is built this way
------------------------
* **Shape variables: CST perturbations of the original outline.** Each surface
  gets ``y += C(x) * sum_i da_i B_i(x)`` with Kulfan's class function
  ``C = sqrt(x) (1 - x)`` and Bernstein polynomials ``B_i`` (order 4 by
  default: 5 per surface). Added to the *original* coordinates rather than to a
  CST fit of them, so the baseline is reproduced exactly (a fit would already
  change the drag by its residual). C vanishes at the leading and trailing edge,
  so those points, the chord line and the angle-of-attack reference stay put;
  the sqrt(x) term moves the nose radius smoothly. Every coefficient set is a
  smooth, kink-free section.
* **Geometry constraints are linear** in these variables (thickness at fixed
  stations, area by the shoelace formula, with the x stations fixed), so SLSQP
  gets them exactly; only the aerodynamics needs the solver.
* **Aerodynamics: the XFOIL-class viscous-inviscid solver** (vendored mfoil).
  Gradients are forward differences with a step of 1e-3 in a Bernstein
  coefficient (a few 1e-4 chord of surface movement): measured Cd along such
  steps is smooth to the solver's transition-point quantisation (NACA 0012,
  Re 1e6, 4 deg), and larger steps would blur the optimum. One gradient costs
  one analysis per variable (~1-3 s each).
* **Robust to failed analyses.** A point that does not converge is
  infeasible: the objective returns a penalty worse than anything feasible
  that rises quadratically away from the best shape found, so SLSQP's line
  search backs off and its next step heads back (a flat penalty leaves it
  stranded there); if it still stalls, it is restarted from the best shape
  with the iterations left. A difference whose perturbed point fails is taken
  one-sided the other way, and if both fail that gradient component is zero.
  The best *feasible converged* point seen is what is returned, never an
  unconverged one.
* **Scaling.** The objective is normalised by its baseline value and scaled so
  that the first SLSQP step is about one design unit (0.01 in a Bernstein
  coefficient): the BFGS start Hessian is the identity, and an unscaled first
  step would be either negligible or wild.

``target_cl`` adds the angle of attack as a variable and an equality
constraint Cl = target; the objective is then drag (equivalently L/D at that
lift). Results are deterministic (the solver and SLSQP are).
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

import numpy as np

from ..geometry.design import normalize_airfoil
from ..geometry.params import cst_fit
from ..model import PolarPoint
from .common import Cancelled, OptimizationResult, Progress, ProgressFn, check_cancel

__all__ = ["AnalyzeFn", "airfoil_metrics", "optimize_airfoil"]

#: (coords, re, alpha_deg) -> PolarPoint. The default wraps aerolume.solvers.airfoil.analyze.
AnalyzeFn = Callable[[np.ndarray, float, float], PolarPoint]

OBJECTIVES = ("ld", "cd", "cl")
DEFAULT_ORDER = 4
#: One design-variable unit is this change of a Bernstein coefficient.
X_UNIT = 0.01
#: Forward-difference steps: shape (design units) and angle of attack (degrees).
FD_STEP = 0.1
ALPHA_FD_STEP = 0.05
#: Default bound on each coefficient change (in CST units; ~0.385 * 0.06 = 2.3 % chord peak).
MAX_CHANGE = 0.06
#: Target length of the first SLSQP step in design units.
FIRST_STEP = 1.0
#: The section may nowhere get thinner than this fraction of its original
#: thickness (between LOCAL_RANGE in x/c): keeps spars and trailing edges buildable.
LOCAL_THICKNESS = 0.5
LOCAL_RANGE = (0.02, 0.95)
#: Thickness stations (cosine spaced in x/c).
STATIONS = 121
#: Curvature of the penalty bowl returned at points the solver cannot converge.
FAIL_BOWL = 0.05
#: Restarts from the best shape when SLSQP stalls in a region that does not converge.
MAX_RESTARTS = 2
#: Feasibility tolerances for picking the best point.
TOL_GEOM = 1e-7
TOL_CL = 2e-3


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _bernstein(psi: np.ndarray, order: int) -> np.ndarray:
    i = np.arange(order + 1)
    k = np.array([math.comb(order, j) for j in i], dtype=float)
    return k * psi[:, None] ** i * (1.0 - psi[:, None]) ** (order - i)


def _interp_matrix(xs: np.ndarray, idx: np.ndarray, stations: np.ndarray, n: int) -> np.ndarray:
    """Rows of weights so that ``W @ y`` interpolates y (on points ``idx``, x = ``xs``,
    increasing) linearly at ``stations``."""
    W = np.zeros((stations.size, n))
    k = np.clip(np.searchsorted(xs, stations) - 1, 0, xs.size - 2)
    x0, x1 = xs[k], xs[k + 1]
    t = np.where(x1 > x0, (stations - x0) / np.where(x1 > x0, x1 - x0, 1.0), 0.0)
    rows = np.arange(stations.size)
    W[rows, idx[k]] += 1.0 - t
    W[rows, idx[k + 1]] += t
    return W


def airfoil_metrics(coords: np.ndarray) -> dict[str, float]:
    """Max thickness and its x/c, max camber and its x/c, and area of a section (unit chord)."""
    q = normalize_airfoil(coords, 1.0)
    ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
    up = q[:ile + 1][::-1]
    lo = q[ile:]
    xs = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, STATIONS)))
    yu = np.interp(xs, up[:, 0], up[:, 1])
    yl = np.interp(xs, lo[:, 0], lo[:, 1])
    t, c = yu - yl, 0.5 * (yu + yl)
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return {"max_thickness": float(t.max()), "x_thickness": float(xs[int(np.argmax(t))]),
            "max_camber": float(c[int(np.argmax(np.abs(c)))]), "x_camber": float(xs[int(np.argmax(np.abs(c)))]),
            "area": abs(area)}


class _Shape:
    """Linear map from design variables to the outline and its constraint quantities."""

    def __init__(self, coords: np.ndarray, order: int):
        q = normalize_airfoil(coords, 1.0)
        self.q0 = q
        n = len(q)
        self.ile = ile = int(np.argmin(np.hypot(q[:, 0], q[:, 1])))
        psi = np.clip(q[:, 0], 0.0, 1.0)
        cb = (np.sqrt(psi) * (1.0 - psi))[:, None] * _bernstein(psi, order)
        upper = np.zeros(n, bool)
        upper[:ile] = True
        lower = np.zeros(n, bool)
        lower[ile + 1:] = True
        m = order + 1
        M = np.zeros((n, 2 * m))
        M[upper, :m] = cb[upper]
        M[lower, m:] = cb[lower]
        self.M = M * X_UNIT                       # y = y0 + M x
        self.nvar = 2 * m
        up_idx = np.arange(ile, -1, -1)           # LE -> upper TE, x increasing
        lo_idx = np.arange(ile, n)                # LE -> lower TE
        st = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, STATIONS)))
        Wu = _interp_matrix(q[up_idx, 0], up_idx, st, n)
        Wl = _interp_matrix(q[lo_idx, 0], lo_idx, st, n)
        self.stations = st
        self.t0 = (Wu - Wl) @ q[:, 1]
        self.T = (Wu - Wl) @ self.M               # thickness = t0 + T x
        x = q[:, 0]
        a = 0.5 * (np.roll(x, 1) - np.roll(x, -1))   # A = a . y (shoelace, x fixed)
        self.area0 = float(a @ q[:, 1])
        self.a = a @ self.M
        self.local = (st >= LOCAL_RANGE[0]) & (st <= LOCAL_RANGE[1])

    def coords(self, xs: np.ndarray) -> np.ndarray:
        y = self.q0[:, 1] + self.M @ xs
        return np.c_[self.q0[:, 0], y]

    def thickness(self, xs: np.ndarray) -> np.ndarray:
        return self.t0 + self.T @ xs

    def area(self, xs: np.ndarray) -> float:
        return self.area0 + float(self.a @ xs)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _default_analyze(ncrit: float, mach: float, xtr_upper: float, xtr_lower: float) -> AnalyzeFn:
    from ..solvers.airfoil import analyze

    def fn(coords: np.ndarray, re: float, alpha: float) -> PolarPoint:
        return analyze(coords, re, alpha, mach=mach, ncrit=ncrit, xtr_upper=xtr_upper, xtr_lower=xtr_lower)
    return fn


class _Evaluator:
    """Cached, failure-tolerant aerodynamic evaluations of design vectors."""

    def __init__(self, shape: _Shape, re: float, analyze: AnalyzeFn, fixed_alpha: float | None,
                 cancel: threading.Event | None, on_eval: Callable[[], None]):
        self.shape, self.re, self.analyze = shape, re, analyze
        self.fixed_alpha = fixed_alpha
        self.cancel = cancel
        self.cache: dict[bytes, PolarPoint | None] = {}
        self.calls = 0
        self.failed = 0
        self.on_eval = on_eval

    def split(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        if self.fixed_alpha is None:
            return x[:-1], float(x[-1])
        return x, float(self.fixed_alpha)

    def __call__(self, x: np.ndarray) -> PolarPoint | None:
        x = np.asarray(x, dtype=float)
        key = np.round(x, 12).tobytes()
        if key in self.cache:
            return self.cache[key]
        check_cancel(self.cancel)
        xs, alpha = self.split(x)
        try:
            p = self.analyze(self.shape.coords(xs), self.re, alpha)
            ok = bool(p.converged) and math.isfinite(p.cl) and math.isfinite(p.cd) and p.cd > 0
        except Cancelled:
            raise
        except Exception:  # noqa: BLE001 - any solver failure is an infeasible point
            p, ok = None, False
        self.calls += 1
        if not ok:
            self.failed += 1
            p = None
        self.cache[key] = p
        self.on_eval()
        return p


def optimize_airfoil(coords: np.ndarray, re: float, *, alpha: float | None = None,
                     target_cl: float | None = None, objective: str = "ld",
                     min_thickness: float | None = None, keep_area: bool = False, n_iter: int = 8,
                     order: int = DEFAULT_ORDER, max_change: float = MAX_CHANGE, ncrit: float = 9.0,
                     mach: float = 0.0, xtr_upper: float = 1.0, xtr_lower: float = 1.0,
                     progress: ProgressFn | None = None, cancel: threading.Event | None = None,
                     analyze_fn: AnalyzeFn | None = None) -> OptimizationResult:
    """Optimise an airfoil section for one operating point; returns an :class:`OptimizationResult`.

    ``coords``: any airfoil outline (normalised to unit chord, Selig order).
    ``re``: chord Reynolds number. Give either ``alpha`` (degrees from the chord
    line; ``objective`` "ld" = maximise Cl/Cd, "cd" = minimise Cd, "cl" =
    maximise Cl) or ``target_cl`` (the angle becomes a variable, Cl is held at
    the target and Cd is minimised -- "ld" and "cd" are then the same problem).
    ``min_thickness``: lower bound on max t/c (default: the original's);
    ``keep_area``: cross-section area may not shrink. In addition the local
    thickness never drops below half the original between 2 % and 95 % chord.
    ``n_iter``: SLSQP iterations (each costs ~``2 * order + 3`` analyses);
    ``order``: Bernstein order per surface; ``max_change``: bound on each
    coefficient change. ``ncrit``, ``mach``, ``xtr_*``: solver settings.

    ``progress(fraction, message)`` after every analysis. ``cancel``: when set,
    the optimiser stops at the next analysis and returns the best shape found so
    far with a warning (``converged=False``). ``analyze_fn`` replaces the panel
    solver (tests, other solvers). Raises ValueError when the starting section
    itself does not converge.
    """
    wall0 = time.perf_counter()
    objective = objective.lower()
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; use one of {', '.join(OBJECTIVES)}")
    if (alpha is None) == (target_cl is None):
        raise ValueError("give exactly one of alpha (fixed angle) or target_cl (fixed lift)")
    if target_cl is not None and objective == "cl":
        raise ValueError("objective 'cl' needs a fixed alpha; with target_cl the drag is minimised")
    if not (re > 0 and math.isfinite(re)):
        raise ValueError("the Reynolds number must be positive")
    if int(n_iter) < 1:
        raise ValueError("n_iter must be at least 1")
    from scipy.optimize import minimize

    rep = Progress(progress)
    shape = _Shape(coords, int(order))
    nshape = shape.nvar
    analyze = analyze_fn or _default_analyze(ncrit, mach, xtr_upper, xtr_lower)
    est_total = (int(n_iter) + 1) * (nshape + 3) + 4
    state = {"evals": 0}

    def tick() -> None:
        state["evals"] += 1
        rep(min(0.99, state["evals"] / est_total), f"Analysis {state['evals']}: optimising the section")

    ev = _Evaluator(shape, float(re), analyze, alpha, cancel, tick)
    notes: list[str] = []
    warnings: list[str] = []

    # -- starting point -----------------------------------------------------------
    x0 = np.zeros(nshape)
    if target_cl is not None:
        a0 = _alpha_for_cl(shape, ev, float(target_cl), nshape)
        x0 = np.r_[x0, a0]
    p0 = ev(x0)
    if p0 is None:
        raise ValueError("The starting airfoil does not converge in the viscous solver at this operating point; "
                         "choose another angle or Reynolds number.")
    t_min = float(shape.t0.max()) if min_thickness is None else float(min_thickness)
    if t_min > shape.t0.max() + 1e-9:
        warnings.append(f"The required thickness {100 * t_min:.1f} % is above the starting section's "
                        f"{100 * shape.t0.max():.1f} %: the optimiser must thicken it first.")

    mode = "cd" if target_cl is not None else objective     # at fixed lift, best L/D = least drag

    def aero_value(p: PolarPoint) -> float:
        """The user-facing objective: Cd, L/D or Cl."""
        return {"cd": float(p.cd), "ld": float(p.cl) / float(p.cd), "cl": float(p.cl)}[mode]

    def raw_f(p: PolarPoint) -> float:
        """Normalised objective to minimise (the baseline gives +-1)."""
        v, v0 = aero_value(p), aero_value(p0)
        return v / v0 if mode == "cd" else -v / max(abs(v0), 0.1 if mode == "cl" else 1e-12)

    def feasible(x: np.ndarray, p: PolarPoint | None) -> bool:
        if p is None:
            return False
        xs, _ = ev.split(x)
        t = shape.thickness(xs)
        if t.max() < t_min - TOL_GEOM or np.any(t[shape.local] < LOCAL_THICKNESS * shape.t0[shape.local] - TOL_GEOM):
            return False
        if keep_area and shape.area(xs) < shape.area0 * (1 - 1e-9):
            return False
        if target_cl is not None and abs(float(p.cl) - target_cl) > TOL_CL:
            return False
        return bool(np.all(np.abs(xs) <= max_change / X_UNIT + 1e-9))

    best = {"x": x0.copy(), "p": p0, "f": raw_f(p0) if feasible(x0, p0) else math.inf}

    def note_point(x: np.ndarray, p: PolarPoint | None) -> None:
        if p is not None and feasible(x, p):
            f = raw_f(p)
            if f < best["f"] - 1e-12:
                best.update(x=np.array(x, float), p=p, f=f)

    f_start = raw_f(p0)

    def bowl(x: np.ndarray) -> tuple[float, np.ndarray]:
        """Value and gradient at a point the solver could not converge: worse than anything
        feasible, rising quadratically away from the best known point, so that both the line
        search and the next quasi-Newton step lead back into the region that converges."""
        d = np.asarray(x, float) - best["x"]
        ref = max(f_start, best["f"]) if math.isfinite(best["f"]) else f_start
        return ref + 0.5 + FAIL_BOWL * float(d @ d), 2.0 * FAIL_BOWL * d

    def fval(x: np.ndarray) -> float:
        p = ev(x)
        note_point(x, p)
        return raw_f(p) if p is not None else bowl(x)[0]

    def fd_grad(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward-difference gradients of the objective and of Cl (one-sided fallbacks)."""
        p = ev(x)
        if p is None:
            return bowl(x)[1], np.zeros_like(x)
        f0, c0 = raw_f(p), float(p.cl)
        gf, gc = np.zeros_like(x), np.zeros_like(x)
        for k in range(x.size):
            h = ALPHA_FD_STEP if (target_cl is not None and k == x.size - 1) else FD_STEP
            for sgn in (1.0, -1.0):
                xp = x.copy()
                xp[k] += sgn * h
                pp = ev(xp)
                note_point(xp, pp)
                if pp is not None:
                    gf[k] = sgn * (raw_f(pp) - f0) / h
                    gc[k] = sgn * (float(pp.cl) - c0) / h
                    break
        return gf, gc

    # -- scaling so the first step is ~FIRST_STEP design units ----------------------
    g0, _ = fd_grad(x0)
    gnorm = float(np.linalg.norm(g0[:nshape]))
    scale = float(np.clip(FIRST_STEP / gnorm, 1e-3, 1e4)) if gnorm > 0 else 1.0

    def fun(x):
        return scale * fval(x)

    def jac(x):
        return scale * fd_grad(x)[0]

    def pad(m: np.ndarray) -> np.ndarray:
        """Shape-only Jacobian rows extended with a zero column for alpha."""
        return np.c_[m, np.zeros(len(m))] if target_cl is not None else m

    loc = shape.local
    cons = [
        {"type": "ineq",
         "fun": lambda x: (shape.thickness(ev.split(x)[0])[loc] - LOCAL_THICKNESS * shape.t0[loc]) / 0.01,
         "jac": lambda x: pad(shape.T[loc]) / 0.01},
        {"type": "ineq",
         "fun": lambda x: np.array([(shape.thickness(ev.split(x)[0]).max() - t_min) / 0.01]),
         "jac": lambda x: pad(shape.T[[int(np.argmax(shape.thickness(ev.split(x)[0])))]]) / 0.01},
    ]
    if keep_area:
        cons.append({"type": "ineq",
                     "fun": lambda x: np.array([100.0 * (shape.area(ev.split(x)[0]) - shape.area0) / shape.area0]),
                     "jac": lambda x: pad(shape.a[None, :]) * 100.0 / shape.area0})
    if target_cl is not None:
        cl_scale = 10.0 / max(abs(target_cl), 0.1)

        def eq_fun(x):
            p = ev(x)
            return np.array([cl_scale * ((float(p.cl) if p is not None else target_cl + 1.0) - target_cl)])

        cons.append({"type": "eq", "fun": eq_fun, "jac": lambda x: cl_scale * fd_grad(x)[1][None, :]})
    lim = max_change / X_UNIT
    bounds = [(-lim, lim)] * nshape + ([(-15.0, 20.0)] if target_cl is not None else [])

    history: list[dict] = []

    def record(x: np.ndarray, it: int) -> None:
        p = ev(x)
        xs, a = ev.split(x)
        t = shape.thickness(xs)
        history.append({"iter": it, "objective": aero_value(p) if p is not None else float("nan"),
                        "cl": float(p.cl) if p is not None else float("nan"),
                        "cd": float(p.cd) if p is not None else float("nan"),
                        "cm": float(p.cm) if p is not None else float("nan"),
                        "alpha": a, "max_thickness": float(t.max()), "area": shape.area(xs),
                        "feasible": feasible(x, p), "best_objective": aero_value(best["p"]),
                        "evaluations": ev.calls, "seconds": time.perf_counter() - wall0})

    record(x0, 0)
    it = {"n": 0}

    def callback(xk, *_):
        it["n"] += 1
        record(np.asarray(xk, float), it["n"])

    cancelled = False
    res = None
    restarts = 0
    start = x0
    try:
        while True:
            left = int(n_iter) - it["n"]
            res = minimize(fun, start, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
                           callback=callback,
                           options={"maxiter": max(1, left), "ftol": 1e-6 * max(scale, 1e-12), "disp": False})
            stuck = ev(res.x) is None or res.status not in (0, 9)
            left = int(n_iter) - it["n"]
            # SLSQP gave up inside the region the solver cannot converge (or its line search
            # failed): restart from the best feasible shape with the iterations that are left,
            # unless that would repeat the previous attempt
            if not (stuck and left > 0 and restarts < MAX_RESTARTS and not np.array_equal(best["x"], start)):
                break
            restarts += 1
            start = best["x"].copy()
            notes.append(f"SLSQP was restarted from the best shape after: {res.message}")
    except Cancelled:
        cancelled = True

    # -- result -----------------------------------------------------------------
    xb, pb = best["x"], best["p"]
    xs_b, a_b = ev.split(xb)
    xs_0, a_0 = ev.split(x0)
    best_coords = shape.coords(xs_b)

    def summary(p: PolarPoint, xs: np.ndarray, a: float) -> dict[str, float]:
        t = shape.thickness(xs)
        return {"cl": float(p.cl), "cd": float(p.cd), "cm": float(p.cm), "ld": float(p.cl / p.cd),
                "alpha": float(a), "max_thickness": float(t.max()), "area": shape.area(xs),
                "xtr_upper": float(p.xtr_upper), "xtr_lower": float(p.xtr_lower)}

    initial, final = summary(p0, xs_0, a_0), summary(pb, xs_b, a_b)
    v0, v1 = aero_value(p0), aero_value(pb)
    improving = -1.0 if mode == "cd" else 1.0
    improvement = improving * (v1 - v0) / abs(v0) if v0 else 0.0
    sym = {"cd": "Cd", "ld": "L/D", "cl": "Cl"}[mode]
    digits = {"cd": 5, "ld": 2, "cl": 4}[mode]
    converged = bool(res is not None and res.success and not cancelled)
    stop = ("stopped by the user" if cancelled else
            "converged" if converged else
            f"iteration limit ({int(n_iter)}) reached" if res is not None and res.status == 9 else
            (res.message if res is not None else "stopped"))
    message = (f"{sym} {v0:.{digits}f} -> {v1:.{digits}f} ({100 * improvement:+.1f} %) after {it['n']} iteration(s), "
               f"{ev.calls} analyses; {stop}.")
    notes.append(f"Shape variables: order-{order} CST (Bernstein) perturbations of each surface added to the "
                 "original outline; the leading and trailing edges and the chord line are fixed. Gradients by "
                 "forward differences of the viscous panel solver (mfoil); optimiser SLSQP.")
    notes.append(f"Constraints: max thickness >= {100 * t_min:.2f} % chord; local thickness >= "
                 f"{100 * LOCAL_THICKNESS:.0f} % of the original between {100 * LOCAL_RANGE[0]:.0f} % and "
                 f"{100 * LOCAL_RANGE[1]:.0f} % chord" + ("; area not below the original" if keep_area else "")
                 + (f"; Cl = {target_cl:g} (angle free)" if target_cl is not None else f"; alpha = {alpha:g} deg")
                 + f"; each coefficient within +-{max_change:g}.")
    if ev.failed:
        warnings.append(f"{ev.failed} of {ev.calls} analyses did not converge; they were treated as infeasible "
                        "and are never returned as the result.")
    if re < 5e4:
        warnings.append(f"Re = {re:.3g} is very low for the integral boundary-layer model; treat the gains as "
                        "indicative and check them with a CFD run.")
    if cancelled:
        warnings.append("Stopped by the user: this is the best shape found so far.")
    if improvement <= 1e-6:
        warnings.append("No better feasible shape was found; the original is returned.")
    cst = cst_fit(best_coords, 8)
    return OptimizationResult(
        method="airfoil-cst-slsqp", objective=mode,
        initial=initial, best=final, improvement=float(improvement), best_coords=best_coords,
        history=history, cst=cst, x=np.array(xb, float), converged=converged, message=message,
        evaluations=ev.calls, failed_evaluations=ev.failed, wall_seconds=time.perf_counter() - wall0,
        notes=notes, warnings=warnings,
        extra={"re": float(re), "alpha": alpha, "target_cl": target_cl, "order": int(order),
               "min_thickness": t_min, "keep_area": bool(keep_area), "objective_scale": scale,
               "x_unit": X_UNIT, "cancelled": cancelled})


def _alpha_for_cl(shape: _Shape, ev: _Evaluator, target: float, nshape: int) -> float:
    """Angle of attack giving ``target`` Cl on the starting shape (secant from 2 and 4 deg)."""
    xs = np.zeros(nshape)
    a1, a2 = 2.0, 4.0
    p1, p2 = ev(np.r_[xs, a1]), ev(np.r_[xs, a2])
    for _ in range(6):
        if p1 is None or p2 is None or abs(p2.cl - p1.cl) < 1e-9:
            break
        a3 = a2 + (target - p2.cl) * (a2 - a1) / (p2.cl - p1.cl)
        a3 = float(np.clip(a3, -10.0, 16.0))
        p3 = ev(np.r_[xs, a3])
        a1, p1, a2, p2 = a2, p2, a3, p3
        if p3 is not None and abs(p3.cl - target) < 0.2 * TOL_CL:
            return a3
    if p2 is None:
        raise ValueError(f"Could not find an angle of attack for Cl = {target:g} on the starting airfoil.")
    return a2
