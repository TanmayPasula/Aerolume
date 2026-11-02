"""Characteristic numbers of an airfoil polar: Cl_max, Cd_min, (L/D)_max, lift slope, ...

These are the figures an airfoil designer reads off a polar first, so the Results
page, the Compare page and the report all take them from here rather than each
re-deriving them slightly differently.

Only converged points are used (:meth:`PolarResult.table`). Extremes (Cl_max,
Cd_min, L/D_max) are reported at the computed points, not interpolated: with the
usual 0.5-1 deg steps an interpolated stall peak would claim precision the sweep
does not have. The lift slope and zero-lift angle come from a straight-line fit
over the attached-flow range, which is where thin-airfoil theory (2 pi per rad)
is the yardstick.

**Was the stall reached?** The largest Cl of a sweep is only Cl,max when the lift
curve has turned over inside it. The default sweep stops at 14 deg, where a
NACA 2412 at Re 2e6 still gains 0.05 per degree (1.43, 1.50, 1.55 at 12, 13, 14
deg): "Cl,max 1.547 at 14 deg" there is just the lift at the last angle, and
ranking two airfoils on it ranks them on Cl at 14 deg. ``cl_max_reached`` says
which case it is (see :data:`STALL_SLOPE_FRACTION`); when it is 0 the value is a
lower bound and the Compare page, the report and the CLI say so.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ..model import PolarResult

# Half-width [deg] of the alpha window, centred on zero lift, used to fit the lift slope.
LINEAR_HALF_WIDTH = 4.0
#: When the highest Cl is the last converged angle, the peak still counts as reached
#: if the lift curve has flattened there: the slope over the final interval below this
#: fraction of the fitted lift slope (a stalling section's curve rounds over; one in its
#: linear range keeps ~100 %, the NACA 2412 case above 47 %).
STALL_SLOPE_FRACTION = 0.25

METRIC_KEYS = ("cl_max", "alpha_cl_max", "cl_max_reached", "cd_min", "cl_at_cd_min", "alpha_cd_min", "ld_max",
               "alpha_ld_max", "cl_at_ld_max", "lift_slope_per_deg", "lift_slope_per_rad",
               "alpha_zero_lift", "cm0", "alpha_min", "alpha_max", "n_points", "n_converged")


def _table(polar: PolarResult | Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    if isinstance(polar, PolarResult):
        tab, n_all = polar.table(), len(polar.points)
    else:
        tab = {k: np.asarray(v, dtype=float) for k, v in polar.items()}
        n_all = len(tab.get("alpha", []))
    alpha = np.asarray(tab.get("alpha", []), dtype=float)
    order = np.argsort(alpha, kind="stable")
    out = {k: np.asarray(v, dtype=float)[order] for k, v in tab.items() if len(v) == alpha.size}
    return out, n_all


def _zero_crossing(x: np.ndarray, y: np.ndarray) -> float:
    """x where y first changes sign (linear interpolation), nan if it never does."""
    s = np.sign(y)
    idx = np.nonzero(s[:-1] * s[1:] <= 0)[0]
    for i in idx:
        if y[i] == y[i + 1]:
            continue
        return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")


def _peak_reached(a: np.ndarray, cl: np.ndarray, alpha_peak: float, slope: float) -> float:
    """1.0 if the Cl maximum at ``alpha_peak`` is a real peak of the lift curve, 0.0 if Cl
    is still rising at the end of the sweep, nan without data (see :func:`polar_metrics`)."""
    fin = np.isfinite(cl)
    if not math.isfinite(alpha_peak) or fin.sum() < 2:
        return float("nan") if not fin.any() else 0.0
    aa, cc = a[fin], cl[fin]
    i = int(np.argmin(np.abs(aa - alpha_peak)))
    if i < aa.size - 1:
        return 1.0                         # a converged angle above the peak has less lift
    da = aa[i] - aa[i - 1]
    if not (da > 0 and math.isfinite(slope) and slope > 0):
        return 0.0
    return 1.0 if (cc[i] - cc[i - 1]) / da < STALL_SLOPE_FRACTION * slope else 0.0


def polar_metrics(polar: PolarResult | Mapping[str, np.ndarray]) -> dict[str, float]:
    """Key figures of a polar; ``nan`` for anything the data cannot support.

    Keys (:data:`METRIC_KEYS`): ``cl_max, alpha_cl_max, cl_max_reached, cd_min,
    cl_at_cd_min, alpha_cd_min, ld_max, alpha_ld_max, cl_at_ld_max,
    lift_slope_per_deg, lift_slope_per_rad, alpha_zero_lift, cm0, alpha_min,
    alpha_max, n_points, n_converged``. Angles in degrees. ``cm0`` is the moment
    coefficient at zero lift (the aerodynamic-centre moment for a linear polar),
    interpolated in alpha; when the sweep never reaches zero lift it falls back to
    Cm at alpha = 0. Accepts a :class:`PolarResult` or a table mapping like
    ``PolarResult.table()``.

    ``cl_max_reached`` is 1.0 when the lift curve peaks inside the sweep (a
    converged point at a higher angle has less lift, or the curve has flattened
    to below :data:`STALL_SLOPE_FRACTION` of the lift slope at the last point) and
    0.0 when Cl is still rising at the end: ``cl_max`` / ``alpha_cl_max`` are then
    the last point's values, a lower bound on Cl,max and not a stall angle. nan
    without data.
    """
    tab, n_all = _table(polar)
    nan = float("nan")
    out: dict[str, float] = {k: nan for k in METRIC_KEYS}
    a = tab.get("alpha", np.zeros(0))
    out["n_points"] = float(n_all)
    out["n_converged"] = float(a.size)
    if a.size == 0:
        return out
    cl = tab.get("cl", np.full(a.size, nan))
    cd = tab.get("cd", np.full(a.size, nan))
    cm = tab.get("cm", np.full(a.size, nan))
    out["alpha_min"], out["alpha_max"] = float(a[0]), float(a[-1])

    if np.isfinite(cl).any():
        i = int(np.nanargmax(cl))
        out["cl_max"], out["alpha_cl_max"] = float(cl[i]), float(a[i])
    pos = np.isfinite(cd) & (cd > 0)
    if pos.any():
        i = int(np.nanargmin(np.where(pos, cd, np.inf)))
        out["cd_min"], out["cl_at_cd_min"], out["alpha_cd_min"] = float(cd[i]), float(cl[i]), float(a[i])
        ld = np.where(pos & np.isfinite(cl), cl / np.where(pos, cd, 1.0), -np.inf)
        if np.isfinite(ld).any():
            i = int(np.argmax(ld))
            out["ld_max"], out["alpha_ld_max"], out["cl_at_ld_max"] = float(ld[i]), float(a[i]), float(cl[i])

    # Lift slope: fit over the attached-flow range only. Points past Cl_max are
    # post-stall; the window is centred on the zero-lift angle when the polar crosses
    # zero lift (else on alpha = 0) and widened until it holds at least 3 points.
    ok = np.isfinite(cl)
    if math.isfinite(out["alpha_cl_max"]):
        ok &= a <= out["alpha_cl_max"]
    aa, cc = a[ok], cl[ok]
    if aa.size >= 2:
        a0 = _zero_crossing(aa, cc)
        centre = a0 if math.isfinite(a0) else 0.0
        half = LINEAR_HALF_WIDTH
        sel = np.abs(aa - centre) <= half
        while sel.sum() < min(3, aa.size) and half < 90.0:
            half *= 1.5
            sel = np.abs(aa - centre) <= half
        if sel.sum() >= 2 and np.ptp(aa[sel]) > 0:
            slope, icpt = np.polyfit(aa[sel], cc[sel], 1)
            out["lift_slope_per_deg"] = float(slope)
            out["lift_slope_per_rad"] = float(slope * 180.0 / math.pi)
            out["alpha_zero_lift"] = a0 if math.isfinite(a0) else (
                float(-icpt / slope) if slope != 0 else nan)

    out["cl_max_reached"] = _peak_reached(a, cl, out["alpha_cl_max"], out["lift_slope_per_deg"])

    fin = np.isfinite(cm)
    if fin.any():
        az = out["alpha_zero_lift"]
        if math.isfinite(az) and a[fin][0] <= az <= a[fin][-1]:
            out["cm0"] = float(np.interp(az, a[fin], cm[fin]))
        else:
            out["cm0"] = float(np.interp(0.0, a[fin], cm[fin])) if a[fin][0] <= 0.0 <= a[fin][-1] else nan
    return out
