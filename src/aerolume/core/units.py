"""Unit parsing and formatting for user-facing values.

Users type "120 km/h", "55 mph", "4.5 m", "180 mm"; everything internal is SI.
``parse(text, kind)`` returns the SI value; ``fmt(value, kind, unit)`` formats one.

Commas. Half the world writes "2,5" for two and a half, the other half "1,500"
for fifteen hundred, and a parser that silently picks one reading turns the
other into a 10x-1000x error that sails through every range check (a speed of
"2,5" m/s was read as 25 m/s). :func:`parse` therefore accepts a comma only
where it has a single meaning -- a lone decimal comma ("2,5", "0,25", "1234,5")
or strict thousands groups ("12,345,678", "1,234.5") -- and refuses the
genuinely ambiguous "1,500" with a message that says why. Formatting for an
editable field uses ``sig(..., grouping=False)`` so that what the program
writes into a box always parses back.
"""
from __future__ import annotations

import math
import re

UNITS: dict[str, dict[str, float]] = {
    "speed": {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704, "kn": 0.514444, "ft/s": 0.3048},
    "length": {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3, "in": 0.0254, "ft": 0.3048},
    "pressure": {"Pa": 1.0, "kPa": 1e3, "bar": 1e5, "psi": 6894.757},
    "force": {"N": 1.0, "kN": 1e3, "lbf": 4.448222},
    "force_per_span": {"N/m": 1.0, "kN/m": 1e3},
    "power": {"W": 1.0, "kW": 1e3, "hp": 745.6999},
    "angle": {"deg": 1.0, "rad": 180.0 / math.pi},
    "temperature": {"C": 1.0},
}
DEFAULT_UNIT = {"speed": "m/s", "length": "m", "pressure": "Pa", "force": "N",
                "force_per_span": "N/m", "power": "W", "angle": "deg", "temperature": "C"}
_ALIASES = {"kph": "km/h", "kmh": "km/h", "knots": "kn", "kt": "kn", "kts": "kn", "mps": "m/s",
            "fps": "ft/s", "inch": "in", "inches": "in", '"': "in", "feet": "ft", "'": "ft",
            "°": "deg", "degrees": "deg", "degree": "deg"}
_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")
#: The number at the start of the text, commas allowed (validated by _decomma).
_TOKEN = re.compile(r"^\s*([-+]?[\d.,]*\d[\d.,]*(?:[eE][-+]?\d+)?|[-+]?[.,]\d+(?:[eE][-+]?\d+)?)(.*)$", re.DOTALL)
#: Thousands groups: 1-3 leading digits (not a leading zero) and groups of exactly three.
_GROUPED = re.compile(r"[1-9]\d{0,2}(?:,\d{3})+")
_DECIMAL_HINT = "use '.' as the decimal mark (or ',' alone, as in 2,5)"


def _decomma(number: str) -> str:
    """``number`` with its commas resolved, or ValueError when they are ambiguous.

    * no comma: unchanged;
    * a '.' as well: the commas must be strict thousands groups before it
      ("1,234.5" -> "1234.5"; "1.234,5" and "1,23.4" are refused);
    * one comma: a decimal comma ("2,5" -> "2.5", "0,250" -> "0.250"), except
      "d,ddd" / "dd,ddd" / "ddd,ddd" with a non-zero lead, which is a valid
      thousands grouping as much as a decimal one and is refused;
    * several commas: only strict thousands groups ("12,345,678").
    """
    if "," not in number:
        return number
    sign = number[0] if number[:1] in "+-" else ""
    body = number[len(sign):]
    mant, e, exp = re.split(r"([eE])", body, maxsplit=1) if re.search(r"[eE]", body) else (body, "", "")
    exponent = e + exp
    if "." in mant:
        whole, _, frac = mant.partition(".")
        if not _GROUPED.fullmatch(whole) or "," in frac:
            raise ValueError(f"cannot read {number!r}: {_DECIMAL_HINT}")
        return sign + whole.replace(",", "") + "." + frac + exponent
    if mant.count(",") == 1:
        left, right = mant.split(",")
        if not right.isdigit() or (left and not left.isdigit()):
            raise ValueError(f"cannot read {number!r}: {_DECIMAL_HINT}")
        if len(right) == 3 and _GROUPED.fullmatch(mant):
            raise ValueError(f"{number!r} is ambiguous (thousands separator or decimal comma?): "
                             f"{_DECIMAL_HINT}")
        return sign + (left or "0") + "." + right + exponent
    if _GROUPED.fullmatch(mant):
        return sign + mant.replace(",", "") + exponent
    raise ValueError(f"cannot read {number!r}: {_DECIMAL_HINT}")


def parse(text: str | float, kind: str, default_unit: str | None = None) -> float:
    """Parse a value with an optional unit into SI. Raises ValueError on nonsense.

    Commas follow the rules in the module docstring; infinities and NaN are
    refused (``1e400`` would otherwise reach the solver as ``inf``).
    """
    if isinstance(text, (int, float)):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"value must be finite, got {text!r}")
        return value * UNITS[kind][default_unit or DEFAULT_UNIT[kind]]
    raw = str(text)
    tok = _TOKEN.match(raw)
    if not tok:
        raise ValueError(f"not a number: {text!r}")
    m = _NUM.match(_decomma(tok.group(1)) + tok.group(2))
    if not m:
        raise ValueError(f"not a number: {text!r}")
    value, unit = float(m.group(1)), m.group(2).strip()
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {text!r}")
    unit = _ALIASES.get(unit.lower(), unit) if unit else (default_unit or DEFAULT_UNIT[kind])
    table = UNITS[kind]
    for k, f in table.items():
        if k.lower() == unit.lower():
            return value * f
    raise ValueError(f"unknown {kind} unit {unit!r}; use one of {', '.join(table)}")


def convert(value_si: float, kind: str, unit: str) -> float:
    """SI value expressed in ``unit``."""
    return value_si / UNITS[kind][unit]


def fmt(value_si: float, kind: str, unit: str | None = None, digits: int = 4) -> str:
    """Format an SI value in a unit with ``digits`` significant figures."""
    unit = unit or DEFAULT_UNIT[kind]
    return f"{sig(convert(value_si, kind, unit), digits)} {unit}"


def sig(x: float, digits: int = 4, *, grouping: bool = True) -> str:
    """Significant-figure formatting without scientific notation for everyday sizes.

    ``grouping`` adds thousands separators ("1,235") for reading; text that is
    put into an editable field must use ``grouping=False``, because
    :func:`parse` refuses "1,235" as ambiguous (see the module docstring).
    """
    if x is None or not math.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    if -4 <= mag < 7:
        decimals = max(digits - 1 - mag, 0)
        return f"{x:,.{decimals}f}" if grouping else f"{x:.{decimals}f}"
    return f"{x:.{digits - 1}e}"


def fmt_reynolds(re_: float) -> str:
    """Reynolds numbers read best as mantissa x 10^n: '1.2 x 10^6'."""
    if not math.isfinite(re_) or re_ <= 0:
        return "—"
    if re_ < 1e4:
        return f"{re_:,.0f}"
    e = math.floor(math.log10(re_))
    return f"{re_ / 10**e:.2f}×10^{e}"
