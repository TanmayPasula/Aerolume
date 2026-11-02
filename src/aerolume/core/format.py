"""Number formatting shared by every surface that prints a measured value.

The Results tables, the Compare page, the assistant's run cards and tool data, the scripting
API's printed tables and the MCP server must quote one number one way: a value rounded to its
standard error by the Particle Data Group rule (:func:`round_error`), "steady" for a settled
series (:func:`is_steady`), "≈0" for a mean inside its own error bar (:func:`within_error`).
These functions were written for the Results page (``aerolume.app.widgets.kpi`` re-exports
them); they live here, free of Qt, so the CLI, the MCP server and a build without PySide6 apply
the identical rule instead of keeping copies in step by tests.

* :func:`fmt_re`, :func:`fmt_err`, :func:`fmt_num`, :func:`fmt_fixed`, :func:`fmt_sci`,
  :func:`fmt_sig` - typographic forms (``8.22×10⁵``, ``± 0.00015``, a real minus sign).
* :func:`round_error`, :func:`decimals_from_error`, :func:`fmt_error`, :func:`stat_texts`,
  :func:`value_error_texts` - a value with its uncertainty, rounded alike.
* :func:`plain` - the same texts in plain ASCII-friendly notation (``-``, ``1.2e-06``) for
  terminals and logs, which the scripting API prints.
"""
from __future__ import annotations

import math
import re

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_FROM_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
MINUS = "−"


def fmt_re(re_: float) -> str:
    """Reynolds number as '8.22×10⁵' (fixed-point below 10⁴), an em dash when unknown."""
    try:
        re_ = float(re_)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(re_) or re_ <= 0:
        return "—"
    if re_ < 1e4:
        return f"{re_:,.0f}"
    e = math.floor(math.log10(re_))
    return f"{re_ / 10**e:.2f}×10{str(e).translate(_SUPERSCRIPT)}"


def fmt_err(err: float | None, prefix: str = "± ") -> str:
    """Standard error with two significant figures in plain decimals ('± 0.000061'), '' if unknown."""
    try:
        err = float(err)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(err) or err <= 0:
        return ""
    if err >= 100:
        return f"{prefix}{err:,.0f}"
    decimals = max(0, 1 - math.floor(math.log10(err)))
    return f"{prefix}{err:.{decimals}f}"


def fmt_num(v, digits: int = 4) -> str:
    """Significant-figure formatting with a typographic minus; an em dash for missing values."""
    from .units import sig
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    text = sig(f, digits)
    if "e" in text:                 # sig's scientific branch (|x| < 1e-4 or >= 1e7): "3.267e-05"
        return fmt_sci(f, digits)
    return text.replace("-", MINUS)


#: Below this magnitude a number is written as mantissa × 10ⁿ (:func:`fmt_sig`): fixed decimals
#: printed a sphere's 1.2 µN drag as "0.000001" and its lift as "0.000000".
SCI_BELOW = 1e-3
#: At or above this magnitude too (seven-digit integers stop being readable at a glance).
SCI_ABOVE = 1e7


def fmt_sci(v, digits: int = 4) -> str:
    """``v`` as a mantissa with ``digits`` significant figures times a power of ten: '1.183×10⁻⁶'.

    Typographic minus and superscript exponent, as the Reynolds number is
    written (:func:`fmt_re`); an em dash when missing.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    if f == 0:
        return "0"
    mant, _, exp = f"{f:.{max(1, int(digits)) - 1}e}".partition("e")
    e = int(exp)
    if e == 0:
        return mant.replace("-", MINUS)
    return f"{mant}×10{str(e).translate(_SUPERSCRIPT)}".replace("-", MINUS)


def fmt_sig(v, digits: int = 4) -> str:
    """``digits`` significant figures; × 10ⁿ notation when |v| < :data:`SCI_BELOW` or >= :data:`SCI_ABOVE`.

    The scripting API rounds to significant figures too (``aerolume.api._num``
    over :func:`aerolume.core.units.sig`), so the app and a script quote the
    same drag: 1.183×10⁻⁶ N here, 1.183e-06 in plain-text output.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    if f != 0 and not (SCI_BELOW <= abs(f) < SCI_ABOVE):
        return fmt_sci(f, digits)
    return fmt_num(f, digits)


def needs_sci(values) -> bool:
    """Whether a row of numbers is too small for fixed decimals: its largest finite non-zero |value|
    is below :data:`SCI_BELOW`, where the six decimals a table allows would print zeros."""
    finite = []
    for v in values:
        try:
            f = abs(float(v))
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0:
            finite.append(f)
    return bool(finite) and max(finite) < SCI_BELOW


def fmt_fixed(v, decimals: int) -> str:
    """A number at a fixed number of decimals with a typographic minus; an em dash when missing.

    Tables use one decimal count per column (or per row of one quantity), so the
    right-aligned monospace digits line up on the decimal point.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    text = f"{f:.{max(0, int(decimals))}f}"
    if float(text) == 0.0:          # "-0.00" reads as a sign error
        text = text.lstrip("-")
    return text.replace("-", MINUS)


def decimals_for(values, digits: int = 4, *, max_decimals: int = 6) -> int:
    """Decimals that give the largest finite |value| ``digits`` significant figures.

    One count for a whole column / row keeps the decimal points aligned;
    smaller values of the same group simply show fewer significant figures.
    """
    finite = [abs(float(v)) for v in values if v is not None and math.isfinite(float(v)) and float(v) != 0]
    if not finite:
        return min(digits - 1, max_decimals)
    mag = math.floor(math.log10(max(finite)))
    return int(min(max(digits - 1 - mag, 0), max_decimals))


def round_error(err) -> tuple[float, int, int] | None:
    """(rounded error, significant figures, decimals) of an uncertainty by the Particle Data Group rule.

    The rule (Review of Particle Physics, "Rounding"): take the error's three
    leading digits; 100-354 are quoted to two significant figures, 355-949 to
    one, and 950-999 round up to 1000 quoted to two (0.00095 -> 0.0010). The
    value is then given to the same decimal place. One fixed rule for every
    table: "3.34079 ± 0.00095" read as false precision next to "2.6988 ± 0.0008";
    under the rule they are 3.3408 ± 0.0010 and 2.6988 ± 0.0008. ``decimals``
    is never negative (an error of 31 gives 0 decimals, "184 ± 31"). None when
    the error is missing, zero or not finite.
    """
    try:
        e = abs(float(err))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(e) or e <= 0:
        return None
    exp = math.floor(math.log10(e))
    three = round(e / 10.0 ** exp * 100.0)
    if three >= 1000:                    # 9.996 at three digits is 1000: the next decade
        exp, three = exp + 1, 100
    if three >= 950:
        exp, three = exp + 1, 100        # rounds up to 1.0 of the next decade, two figures
    digits = 1 if 355 <= three < 950 else 2
    decimals = digits - 1 - exp
    rounded = round(three / 10.0 ** (3 - digits)) * 10.0 ** (exp - digits + 1)
    return rounded, digits, max(decimals, 0)


def decimals_from_error(err, sig_digits: int | None = None, *, max_decimals: int = 6) -> int | None:
    """Decimals that quote a standard error to its significant figures; None if unknown.

    The metrology convention: the uncertainty is rounded (by default with the
    Particle Data Group rule, :func:`round_error`: one or two significant
    figures) and the value to the same decimal place (0.0965 ± 0.0016,
    12.59 ± 0.21, 183.9 ± 3.1, 2.6988 ± 0.0008). A value's decimals then say
    how well it is known, instead of changing at random from row to row.
    ``sig_digits`` forces a fixed count instead of the rule.
    """
    try:
        e = abs(float(err))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(e) or e <= 0:
        return None
    if sig_digits is None:
        rule = round_error(e)
        return int(min(rule[2], max_decimals)) if rule is not None else None
    return int(min(max(sig_digits - 1 - math.floor(math.log10(e)), 0), max_decimals))


def fmt_error(err, *, max_decimals: int = 6) -> str:
    """A standard error rounded by :func:`round_error` in plain decimals ('0.0010', '0.0008', '31'),
    × 10ⁿ when it needs more than ``max_decimals`` decimals; an em dash when unknown."""
    rule = round_error(err)
    if rule is None:
        return "—"
    value, digits, decimals = rule
    if decimals > max_decimals:
        return fmt_sci(value, digits)
    return fmt_fixed(value, decimals)


#: A standard error below this share of the mean is far below anything else that limits the value (the
#: grid, the model): the series has settled, and the value is quoted to four significant figures.
STEADY_REL = 1e-5
#: What the error column says for a steady quantity (no averaging error worth quoting). The Summary's
#: footnote and the cell's tooltip spell it out: the whole phrase in a numeric column widened it by twenty
#: characters of monospace.
STEADY_TEXT = "steady"
STEADY_TIP = ("Steady: the averaging error is below 10⁻⁵ of the mean (0.001 %), far below the grid's error; "
              "the value is given to four significant figures")
#: Mean column of a value smaller than its own standard error: it cannot be told from zero. Short, with the
#: footnote and the cell's tooltip saying why: "≈0 (within error)" in the monospace Mean column widened the
#: Summary table by fifteen characters and pushed Conditions below it at 1600 px.
WITHIN_ERROR_TEXT = "≈0"


def is_steady(mean, stderr) -> bool:
    """True when a time average's standard error is below :data:`STEADY_REL` of its (non-zero) mean.

    A converged laminar cavity's Nusselt number has a batch-means error of
    4.4×10⁻¹⁶ (constant to float round-off) or 1.9×10⁻⁵ on 4.56 (4×10⁻⁶ of
    it): quoting that error and rounding the mean to it printed 4.562400 with
    seven figures, which the grid's percent-level error makes false precision.
    """
    try:
        m, e = float(mean), float(stderr)
    except (TypeError, ValueError):
        return False
    return math.isfinite(m) and math.isfinite(e) and m != 0.0 and 0.0 <= e < STEADY_REL * abs(m)


def within_error(mean, stderr) -> bool:
    """True when |mean| is below its (positive, finite) standard error: zero is inside one error bar.

    A cylinder's lift of −0.0016 ± 0.0028, or a pitching moment of
    −1.2×10⁻¹² ± 1.7×10⁻¹¹, is noise around zero; printing its digits invites
    reading a sign or a size that the run did not measure.
    """
    try:
        m, e = float(mean), float(stderr)
    except (TypeError, ValueError):
        return False
    return math.isfinite(m) and math.isfinite(e) and e > 0 and abs(m) < e


def stat_texts(mean, stderr, rms, lo, hi, *, digits: int = 4) -> tuple[str, str, str, str, str]:
    """Texts of one statistics row (mean, ± standard error, RMS, min, max).

    The usual rule rounds the error by :func:`round_error` and the row to the
    same decimal place, so the decimal points line up. A row whose values are
    all below :data:`SCI_BELOW` is written with ``digits`` significant figures
    in × 10ⁿ notation instead (six fixed decimals printed a sphere's drag as
    "0.000001" and its lift as "0.000000"); an RMS too small for two
    significant digits at the row's decimals is written the same way. A
    steady value (:func:`is_steady`) has four significant figures and "steady"
    for its error; a mean inside its own error bar reads
    :data:`WITHIN_ERROR_TEXT`.
    """
    def finite_pos(v) -> bool:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        return math.isfinite(f) and f > 0

    if is_steady(mean, stderr):
        # a steady run: the "error" is round-off of a settled value (4.4×10⁻¹⁶), and rounding the
        # mean to it printed 2.250865 next to a caption promising rounding to the standard error
        return fmt_sig(mean, digits), STEADY_TEXT, "—", fmt_sig(lo, digits), fmt_sig(hi, digits)
    if needs_sci([mean, lo, hi]):
        rule = round_error(stderr)
        err = fmt_sci(rule[0], rule[1]) if rule is not None else "—"
        mean_t = WITHIN_ERROR_TEXT if within_error(mean, stderr) else fmt_sig(mean, digits)
        return mean_t, err, fmt_sig(rms, digits), fmt_sig(lo, digits), fmt_sig(hi, digits)
    d = decimals_from_error(stderr)
    if d is None:
        d = decimals_for([mean, lo, hi], digits)

    def spread(v) -> str:
        # fewer than two significant digits at the row's decimals ("0.000000", "0.000001"): × 10ⁿ -- for
        # small spreads only; an RMS of 0.77 next to "0.6 ± 0.5" reads "0.8", not "7.7×10⁻¹"
        if finite_pos(v) and float(v) < 10.0 ** (1 - d) and float(v) < SCI_BELOW:
            return fmt_sci(v, 2)
        return fmt_fixed(v, d)

    err = fmt_error(stderr) if finite_pos(stderr) else "—"
    mean_t = WITHIN_ERROR_TEXT if within_error(mean, stderr) else fmt_fixed(mean, d)
    return mean_t, err, spread(rms), fmt_fixed(lo, d), fmt_fixed(hi, d)


def value_error_texts(value, stderr, *, digits: int = 4) -> tuple[str, str]:
    """(value, ± error) of a quantity without RMS / min / max, by the same rule as :func:`stat_texts`:
    the per-body Nusselt number, a fan's thrust, a heat rate. The error is "—" when unknown."""
    mean_t, err_t, _rms, _lo, _hi = stat_texts(value, stderr, float("nan"), value, value, digits=digits)
    return mean_t, err_t


_SCI_TEXT_RE = re.compile(r"×10([⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")


def plain(text: str) -> str:
    """A formatted number in plain notation for terminals, logs and CSV: "−1.183×10⁻⁶" -> "-1.183e-06".

    The typographic minus becomes "-" and a superscript power of ten becomes an ``e`` exponent with
    at least two digits (Python's own spelling); "≈0", "steady" and "—" pass through unchanged.
    """
    def exp(m: re.Match) -> str:
        e = int(m.group(1).translate(_FROM_SUPERSCRIPT))
        return f"e{'-' if e < 0 else '+'}{abs(e):02d}"
    return _SCI_TEXT_RE.sub(exp, str(text)).replace(MINUS, "-")


__all__ = ["MINUS", "SCI_ABOVE", "SCI_BELOW", "STEADY_REL", "STEADY_TEXT", "STEADY_TIP", "WITHIN_ERROR_TEXT",
           "decimals_for", "decimals_from_error", "fmt_err", "fmt_error", "fmt_fixed", "fmt_num", "fmt_re",
           "fmt_sci", "fmt_sig", "is_steady", "needs_sci", "plain", "round_error", "stat_texts",
           "value_error_texts", "within_error"]
