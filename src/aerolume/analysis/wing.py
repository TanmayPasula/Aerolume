"""Characteristic numbers of a finite-wing sweep (DESIGN.md 7.6 / 7.10).

The wing counterpart of :mod:`aerolume.analysis.polar`: the Results page, the
Compare page, the report and the CSV export all read a wing's figures of merit
from :func:`wing_metrics`, so they never disagree.

The lift-curve figures (lift slope, zero-lift angle, CL,max and whether the stall
was reached, CD,min, best L/D, Cm0) are computed exactly as for an airfoil polar,
by :func:`aerolume.analysis.polar.polar_metrics` on the wing's table, so a wing
and its section read the same way. Specific to a wing:

* **Parabolic drag polar.** ``CD = a + b CL + c CL^2`` is fitted by least squares
  over the *linear, attached* part of the polar only: converged points from the
  lift of minimum drag (``cl_at_cd_min``) up to :data:`PARABOLA_CL_FRACTION`
  (80 %) of CL,max, never at or beyond the angle where the first strip reaches its
  section cl_max (``WingResult.stall_alpha``, and the angles the lattice's
  "beyond cl_max" warning lists, which also covers results saved before
  ``stall_alpha`` existed). CL,max there is the critical-section estimate
  ``stall_cl`` when the solver gave one, else the sweep's CL,max when the stall
  was reached; a sweep that never reached section cl_max (and every inviscid
  one) is fitted up to its last attached point. The window matters: fitted over a whole -4..14 deg sweep of
  an AR 8.3 tapered wing, the viscous drag rise near section stall and the strips
  held at cl_max (whose lattice lift keeps rising) gave e = 0.54 for a wing whose
  span efficiency is 0.99; over CL 0.10-0.92 the same data give 0.76, a normal
  Oswald factor. The linear term absorbs camber and twist (minimum drag away from
  CL = 0); ``k_parabolic = c`` and ``oswald = 1 / (pi AR c)``, the classical
  aircraft-design definition, which includes the lift dependence of profile drag.
  ``cd0`` is the fitted CD at CL = 0 (zero-lift drag). ``oswald_cl_lo`` /
  ``oswald_cl_hi`` are the CL range actually fitted (quote them with the figure:
  "Oswald e (parabolic fit, CL a-b)") and ``oswald_r2`` the fit's R^2. With fewer
  than :data:`PARABOLA_MIN_POINTS` points in the window, or R^2 below
  :data:`PARABOLA_MIN_R2`, the drag polar is not parabolic there and ``cd0``,
  ``k_parabolic`` and ``oswald`` are nan rather than a misleading number. For an
  inviscid sweep (no profile drag) the fit sees induced drag only, so ``oswald``
  is the span efficiency of the lattice and ``cd0`` is not reported.
* **e at cruise** is the span efficiency ``CL^2 / (pi AR CDi)`` the solver
  reports per point (``WingPoint.e``), read at the cruise point: at
  ``cl_cruise`` when given (interpolated in CL over the attached range), else at
  the angle of best L/D (viscous sweeps), else, for an inviscid sweep, at the
  highest attached CL (where it is best conditioned: e is 0/0 at zero lift).
* **Inviscid sweeps** have no profile drag, so L/D grows without bound as CL
  goes to zero and "best L/D" would only say where the sweep began; CD,min,
  best L/D and CD0 are therefore ``nan`` when ``viscous`` is False.

numpy + stdlib only (DESIGN.md section 2).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

import numpy as np

from ..model import WingResult
from .polar import LINEAR_HALF_WIDTH, _zero_crossing, polar_metrics

#: The drag-polar fit uses points up to this fraction of CL,max (the critical-section
#: estimate when there is one): above it profile drag rises faster than CL^2 as the
#: sections approach stall.
PARABOLA_CL_FRACTION = 0.8
#: Fewest points the parabolic fit needs (three always fit exactly, so R^2 would say nothing).
PARABOLA_MIN_POINTS = 4
#: Below this R^2 the drag polar is not parabolic over the window: no CD0 / k / Oswald e.
PARABOLA_MIN_R2 = 0.98

WING_METRIC_KEYS = (
    "cl_alpha_per_deg", "cl_alpha_per_rad", "alpha_zero_lift", "cl_max", "alpha_cl_max", "cl_max_reached",
    "cd_min", "cl_at_cd_min", "alpha_cd_min", "ld_max", "alpha_ld_max", "cl_at_ld_max",
    "cd0", "k_parabolic", "oswald", "oswald_cl_lo", "oswald_cl_hi", "oswald_r2",
    "e_cruise", "cl_cruise", "alpha_cruise",
    "cm0", "cm_alpha_per_deg", "aspect_ratio", "alpha_min", "alpha_max", "n_points", "n_converged",
)
#: keys that need profile drag; nan for an inviscid sweep (see the module docstring)
_VISCOUS_ONLY = ("cd_min", "cl_at_cd_min", "alpha_cd_min", "ld_max", "alpha_ld_max", "cl_at_ld_max", "cd0")


def _sorted_table(tab: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    alpha = np.asarray(tab.get("alpha", []), dtype=float)
    order = np.argsort(alpha, kind="stable")
    return {k: np.asarray(v, dtype=float)[order] for k, v in tab.items() if len(v) == alpha.size}


def _aspect_ratio(wing: WingResult | None, given: float | None) -> float:
    if given is not None:
        return float(given)
    if wing is None:
        return float("nan")
    ar = float(wing.aspect_ratio)
    if math.isfinite(ar) and ar > 0:
        return ar
    if math.isfinite(wing.span) and math.isfinite(wing.area) and wing.area > 0:
        return float(wing.span ** 2 / wing.area)
    return float("nan")


#: When the stall was reached, the cruise lookup uses points up to this fraction of CL,max.
_CRUISE_CL_FRACTION = 0.9


def _attached(a: np.ndarray, cl: np.ndarray, alpha_cl_max: float, reached: float, cl_max: float) -> np.ndarray:
    """Mask of the attached-flow points used for the cruise lookup."""
    ok = np.isfinite(cl)
    if math.isfinite(alpha_cl_max):
        ok &= a <= alpha_cl_max
    if reached == 1.0 and math.isfinite(cl_max) and cl_max > 0:
        ok &= cl <= _CRUISE_CL_FRACTION * cl_max
    return ok


#: The vortex-lattice warning that lists the angles where strips exceeded their section cl_max
#: (solvers/wing/solver.py); parsed so that results saved before ``stall_alpha`` existed are fitted
#: without those points too.
_BEYOND_RE = re.compile(r"beyond cl_max\) at alpha = ([-\u2212\d., ]+?) deg")


def _clipped_angles(res: WingResult | None) -> list[float]:
    out: list[float] = []
    for w in getattr(res, "warnings", None) or []:
        m = _BEYOND_RE.search(str(w))
        if m:
            for tok in m.group(1).replace("\u2212", "-").split(","):
                try:
                    out.append(float(tok))
                except ValueError:
                    pass
    return out


def _fit_window(a: np.ndarray, cl: np.ndarray, cd: np.ndarray, out: Mapping[str, float],
                stall_alpha: float, stall_cl: float, clipped: list[float]) -> np.ndarray:
    """Mask of the points of the linear, attached polar used by the parabolic fit (module docstring)."""
    ok = np.isfinite(cl) & np.isfinite(cd)
    if math.isfinite(out["alpha_cl_max"]):
        ok &= a <= out["alpha_cl_max"]
    if math.isfinite(stall_alpha):
        ok &= a < stall_alpha - 1e-9                 # strips at or beyond their section cl_max
    if clipped:
        ok &= a < min(clipped) - 1e-9
    if math.isfinite(stall_cl) and stall_cl > 0:
        ok &= cl <= PARABOLA_CL_FRACTION * stall_cl
    elif out["cl_max_reached"] == 1.0 and math.isfinite(out["cl_max"]) and out["cl_max"] > 0:
        ok &= cl <= PARABOLA_CL_FRACTION * out["cl_max"]
    lo = out["cl_at_cd_min"]
    if math.isfinite(lo):
        ok &= cl >= lo - 1e-9
    return ok


def _parabola(cl: np.ndarray, cd: np.ndarray) -> tuple[float, float, float, float] | None:
    """Least-squares (a, b, c, R^2) of cd = a + b cl + c cl^2, or None without enough distinct CL values."""
    fin = np.isfinite(cl) & np.isfinite(cd)
    cl, cd = cl[fin], cd[fin]
    if np.unique(np.round(cl, 12)).size < max(3, PARABOLA_MIN_POINTS):
        return None
    c, b, a = np.polyfit(cl, cd, 2)
    resid = cd - (a + b * cl + c * cl * cl)
    ss_tot = float(np.sum((cd - cd.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(c), r2


def _slope_window(a: np.ndarray, y: np.ndarray, centre: float) -> float:
    """Least-squares dy/dalpha over |alpha - centre| <= LINEAR_HALF_WIDTH (widened to >= 3 points)."""
    fin = np.isfinite(y)
    a, y = a[fin], y[fin]
    if a.size < 2:
        return float("nan")
    half = LINEAR_HALF_WIDTH
    sel = np.abs(a - centre) <= half
    while sel.sum() < min(3, a.size) and half < 90.0:
        half *= 1.5
        sel = np.abs(a - centre) <= half
    if sel.sum() < 2 or np.ptp(a[sel]) == 0:
        return float("nan")
    return float(np.polyfit(a[sel], y[sel], 1)[0])


def wing_metrics(wing: WingResult | Mapping[str, np.ndarray], *, cl_cruise: float | None = None,
                 aspect_ratio: float | None = None, viscous: bool | None = None) -> dict[str, float]:
    """Key figures of a finite-wing sweep; ``nan`` for anything the data cannot support.

    Keys (:data:`WING_METRIC_KEYS`): ``cl_alpha_per_deg`` / ``cl_alpha_per_rad``
    (lift slope over the attached range), ``alpha_zero_lift`` [deg], ``cl_max`` /
    ``alpha_cl_max`` with ``cl_max_reached`` (1.0 a real peak, 0.0 CL still rising
    at the last angle so ``cl_max`` is a lower bound; see ``polar_metrics``),
    ``cd_min`` / ``cl_at_cd_min`` / ``alpha_cd_min``, ``ld_max`` / ``alpha_ld_max`` /
    ``cl_at_ld_max``, ``cd0`` / ``k_parabolic`` / ``oswald`` from the parabolic
    drag polar over the CL range ``oswald_cl_lo`` .. ``oswald_cl_hi`` with its
    ``oswald_r2``, ``e_cruise`` with the ``cl_cruise`` / ``alpha_cruise`` it was read
    at, ``cm0`` (Cm at zero lift) and ``cm_alpha_per_deg`` (dCm/dalpha, negative is
    statically stable about the moment reference), ``aspect_ratio``,
    ``alpha_min`` / ``alpha_max``, ``n_points`` / ``n_converged``. See the module
    docstring for the definitions and for what an inviscid sweep reports.

    Accepts a :class:`~aerolume.model.WingResult` or a table mapping like
    ``WingResult.table()`` (then pass ``aspect_ratio`` and ``viscous``; without
    them the AR-dependent keys are nan and the sweep counts as viscous).
    """
    if isinstance(wing, WingResult):
        tab, n_all, res = _sorted_table(wing.table()), len(wing.points), wing
        visc = bool(wing.viscous) if viscous is None else bool(viscous)
    else:
        tab = _sorted_table(wing)
        n_all, res = len(tab.get("alpha", [])), None
        visc = True if viscous is None else bool(viscous)
    nan = float("nan")
    out: dict[str, float] = {k: nan for k in WING_METRIC_KEYS}
    ar = _aspect_ratio(res, aspect_ratio)
    out["aspect_ratio"] = ar
    pm = polar_metrics({k: tab[k] for k in ("alpha", "cl", "cd", "cm") if k in tab})
    for k in ("alpha_zero_lift", "cl_max", "alpha_cl_max", "cl_max_reached", "cd_min", "cl_at_cd_min",
              "alpha_cd_min", "ld_max", "alpha_ld_max", "cl_at_ld_max", "cm0", "alpha_min", "alpha_max",
              "n_converged"):
        out[k] = pm[k]
    out["cl_alpha_per_deg"], out["cl_alpha_per_rad"] = pm["lift_slope_per_deg"], pm["lift_slope_per_rad"]
    out["n_points"] = float(n_all)
    a = tab.get("alpha", np.zeros(0))
    if a.size == 0:
        return out
    cl = tab.get("cl", np.full(a.size, nan))
    cd = tab.get("cd", np.full(a.size, nan))
    cm = tab.get("cm", np.full(a.size, nan))
    e = tab.get("e", np.full(a.size, nan))

    att = _attached(a, cl, out["alpha_cl_max"], out["cl_max_reached"], out["cl_max"])
    stall_alpha = float(getattr(res, "stall_alpha", nan)) if res is not None else nan
    stall_cl = float(getattr(res, "stall_cl", nan)) if res is not None else nan
    win = _fit_window(a, cl, cd, out, stall_alpha, stall_cl, _clipped_angles(res))
    fit = _parabola(cl[win], cd[win])
    if fit is not None:
        out["oswald_cl_lo"], out["oswald_cl_hi"] = float(np.min(cl[win])), float(np.max(cl[win]))
        out["oswald_r2"] = fit[3]
    if fit is not None and fit[2] > 0 and fit[3] >= PARABOLA_MIN_R2:
        out["k_parabolic"] = fit[2]
        out["cd0"] = fit[0]
        if math.isfinite(ar) and ar > 0:
            out["oswald"] = 1.0 / (math.pi * ar * fit[2])

    if not visc:
        for k in _VISCOUS_ONLY:
            out[k] = nan

    # cruise point (see the module docstring)
    ok = att & np.isfinite(e)
    if cl_cruise is not None:
        cc, ee, aa = cl[ok], e[ok], a[ok]
        order = np.argsort(cc)
        cc, ee, aa = cc[order], ee[order], aa[order]
        if cc.size >= 2 and cc[0] <= cl_cruise <= cc[-1] and np.all(np.diff(cc) > 0):
            out["e_cruise"] = float(np.interp(cl_cruise, cc, ee))
            out["alpha_cruise"] = float(np.interp(cl_cruise, cc, aa))
            out["cl_cruise"] = float(cl_cruise)
    else:
        i = None
        if visc and math.isfinite(out["alpha_ld_max"]):
            i = int(np.argmin(np.abs(a - out["alpha_ld_max"])))
        elif ok.any():
            idx = np.nonzero(ok)[0]
            i = int(idx[np.argmax(cl[idx])])
        if i is not None and math.isfinite(e[i]):
            out["e_cruise"], out["cl_cruise"], out["alpha_cruise"] = float(e[i]), float(cl[i]), float(a[i])

    # pitching-moment slope over the same attached window as the lift slope
    pre = np.isfinite(cl)
    if math.isfinite(out["alpha_cl_max"]):
        pre &= a <= out["alpha_cl_max"]
    if pre.sum() >= 2:
        a0 = _zero_crossing(a[pre], cl[pre])
        out["cm_alpha_per_deg"] = _slope_window(a[pre], cm[pre], a0 if math.isfinite(a0) else 0.0)
    return out


__all__ = ["PARABOLA_CL_FRACTION", "PARABOLA_MIN_POINTS", "PARABOLA_MIN_R2", "WING_METRIC_KEYS",
           "wing_metrics"]
