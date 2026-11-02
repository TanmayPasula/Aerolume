"""Number formatting for reports.

A report is read by people who act on its numbers, so formatting carries meaning:
a mean is printed only to the precision its standard error supports
("0.0255 ± 0.0006", never "0.025484 ± 0.000613"), Reynolds numbers read as
mantissa x power of ten, deltas carry an explicit sign, and anything that could
not be computed shows as an em dash rather than "nan".
"""
from __future__ import annotations

import math
import re
from typing import Any

from ..core import units

DASH = "—"
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def finite(x: Any) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def num(x: Any, digits: int = 4) -> str:
    """Significant-figure formatting, em dash for missing values."""
    return units.sig(float(x), digits) if finite(x) else DASH


def fixed(x: Any, decimals: int = 2) -> str:
    return f"{float(x):,.{decimals}f}" if finite(x) else DASH


def decimals_for(err: float) -> int:
    """Decimal places for quoting a standard error: one significant figure, two when the
    leading digit is 1 or 2 (the usual convention: 0.0024 keeps two, 0.00006 keeps one)."""
    e = math.floor(math.log10(abs(err)))
    lead = int(abs(err) / 10 ** e)
    return max(0, -e + (1 if lead < 3 else 0))


#: A coefficient smaller than this is printed as "≈ 0": a symmetric body's lift of 6×10⁻⁸ is
#: round-off of the solver, and eight decimals of it would read as a measurement.
ZERO_COEFF = 1e-6
#: Most significant digits a mean is quoted to, however small its standard error: a steady
#: laminar run is "known" to 10⁻⁶, but 1.1886079 says nothing 1.1886 does not.
MAX_SIG = 5


def _mean_decimals(mean: float, err: float) -> tuple[int, bool]:
    """(decimals for the mean, True when capped by :data:`MAX_SIG` rather than by the error)."""
    d = decimals_for(err)
    cap = max(0, MAX_SIG - 1 - math.floor(math.log10(abs(mean)))) if mean else d
    return (cap, True) if d > cap else (d, False)


def sci1(x: Any) -> str:
    """A small positive number as one or two significant figures times a power of ten: '2.0×10⁻⁶'."""
    if not finite(x):
        return DASH
    v = abs(float(x))
    e = math.floor(math.log10(v))
    m = v / 10 ** e
    return f"{m:.1f}×10{str(e).translate(_SUP)}"


def pm(mean: Any, err: Any, digits: int = 4) -> str:
    """'mean ± stderr' rounded to the stderr's precision; plain number when no stderr.

    Near-zero coefficients read "≈ 0"; a mean is never quoted to more than
    :data:`MAX_SIG` significant figures (the error is then given as a power of ten).
    """
    if not finite(mean):
        return DASH
    if abs(float(mean)) < ZERO_COEFF:
        return "≈ 0"
    if not (finite(err) and float(err) > 0):
        return num(mean, digits)
    d, capped = _mean_decimals(float(mean), float(err))
    if capped:
        return f"{float(mean):.{d}f} ± {sci1(err)}"
    return f"{float(mean):.{d}f} ± {float(err):.{d}f}"


def mean_only(mean: Any, err: Any, digits: int = 4) -> str:
    """The mean rounded to the precision its stderr supports (for KPI tiles)."""
    if not finite(mean):
        return DASH
    if abs(float(mean)) < ZERO_COEFF:
        return "≈ 0"
    if not (finite(err) and float(err) > 0):
        return num(mean, digits)
    return f"{float(mean):.{_mean_decimals(float(mean), float(err))[0]}f}"


def err_only(err: Any) -> str:
    if not (finite(err) and float(err) > 0):
        return DASH
    if float(err) < 1e-5:
        return sci1(err)
    return f"{float(err):.{decimals_for(float(err))}f}"


def solver_text(text: Any) -> str:
    """Solver and planner messages as report text: "1.00×10^6" -> "1.00×10⁶".

    Messages are written once for every front end; the console and the log use the
    caret and a plain-text dash ("--"), while a typeset report shows a proper superscript
    and an en dash.
    """
    text = re.sub(r"10\^(-?\d+)", lambda m: "10" + m.group(1).translate(_SUP), str(text))
    return text.replace(" -- ", " – ")


def reynolds(re_: Any) -> str:
    """'2.00 × 10⁶' (or plain digits below 10 000)."""
    if not finite(re_) or float(re_) <= 0:
        return DASH
    v = float(re_)
    if v < 1e4:
        return f"{v:,.0f}"
    e = math.floor(math.log10(v))
    return f"{v / 10 ** e:.2f} × 10{str(e).translate(_SUP)}"


def pct(x: Any, decimals: int = 1, signed: bool = True) -> str:
    if not finite(x):
        return DASH
    tiny = 10.0 ** -decimals
    if 0 < abs(float(x)) < 0.5 * tiny and not signed:
        return f"< {tiny:.{decimals}f} %"
    return f"{float(x):+.{decimals}f} %" if signed else f"{float(x):.{decimals}f} %"


def qty(x: Any, kind: str, unit: str | None = None, digits: int = 4) -> str:
    """SI value in a display unit via :mod:`aerolume.core.units` ("33.33 m/s")."""
    if not finite(x):
        return DASH
    return units.fmt(float(x), kind, unit, digits)


def with_unit(x: Any, unit: str, digits: int = 4) -> str:
    return f"{num(x, digits)} {unit}" if finite(x) else DASH


def seconds(s: Any) -> str:
    """Wall-clock duration: '42 s', '3 min 04 s', '1 h 12 min'."""
    if not finite(s):
        return DASH
    s = float(s)
    if s < 60:
        return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
    m, sec = divmod(round(s), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def angle(a: Any) -> str:
    return f"{float(a):g}°" if finite(a) else DASH


def yes_no(b: Any) -> str:
    return DASH if b is None else ("yes" if b else "no")


def eq(symbol: str, text: str) -> str:
    """'Cm = 0.131', or 'CL ≈ 0' when the value text is already an approximation."""
    return f"{symbol} {text}" if text.startswith("≈") else f"{symbol} = {text}"
