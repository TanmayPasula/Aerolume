"""NACA airfoil generators and airfoil coordinate files (.dat).

Coordinates follow the Aerolume polygon convention: an ``(N, 2)`` array in Selig
order -- upper trailing edge, forward over the upper surface to the leading edge,
back along the lower surface to the lower trailing edge. That order is
counter-clockwise, so the result is directly a valid solid outline. It is
implicitly closed: a sharp trailing edge appears once (as point 0), a blunt one as
two distinct points joined by the implicit closing edge.

NACA formulas are the published ones (Abbott & von Doenhoff, *Theory of Wing
Sections*, 1959; NASA TM-4741 for the 5-digit reflex constants). Thickness is
applied perpendicular to the mean camber line, which is what the published
coordinate tables use. The "closed trailing edge" option replaces the last
thickness coefficient (-0.1015) by -0.1036, the common modification that makes the
thickness exactly zero at x = 1 without changing anything visible elsewhere.
"""
from __future__ import annotations

import io
import math
import os
import re
from pathlib import Path

import numpy as np

from .polygon import as_polygon, orient_ccw

__all__ = [
    "cosine_spacing",
    "format_airfoil_dat",
    "naca",
    "naca4",
    "naca5",
    "naca_thickness",
    "read_airfoil_dat",
    "write_airfoil_dat",
]

_T_OPEN = (0.2969, -0.1260, -0.3516, 0.2843, -0.1015)
_T_CLOSED = (0.2969, -0.1260, -0.3516, 0.2843, -0.1036)

# 5-digit standard (non-reflex) mean lines, design CL = 0.3: P -> (r, k1)
_FIVE_STD = {1: (0.0580, 361.400), 2: (0.1260, 51.640), 3: (0.2025, 15.957),
             4: (0.2900, 6.643), 5: (0.3910, 3.230)}
# 5-digit reflex mean lines, design CL = 0.3: P -> (r, k1, k2/k1)
_FIVE_REFLEX = {2: (0.1300, 51.990, 0.000764), 3: (0.2170, 15.793, 0.00677),
                4: (0.3180, 6.520, 0.0303), 5: (0.4410, 3.191, 0.1355)}


def cosine_spacing(n_side: int) -> np.ndarray:
    """``n_side + 1`` stations from 0 to 1 clustered at both ends (x = (1 - cos b) / 2).

    Cosine spacing puts points where the curvature is: the leading-edge radius is
    resolved without wasting points on the flat mid-chord.
    """
    beta = np.linspace(0.0, math.pi, int(n_side) + 1)
    return 0.5 * (1.0 - np.cos(beta))


def naca_thickness(x: np.ndarray, t: float, closed_te: bool = True) -> np.ndarray:
    """Half-thickness distribution y_t(x) of the NACA 4/5-digit families (chord 1)."""
    a0, a1, a2, a3, a4 = _T_CLOSED if closed_te else _T_OPEN
    x = np.asarray(x, dtype=float)
    return 5.0 * t * (a0 * np.sqrt(np.clip(x, 0, None)) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)


def _clean_code(code: str | int) -> str:
    s = str(code).strip().upper().replace("NACA", "").replace(" ", "").replace("-", "")
    if not s.isdigit():
        raise ValueError(f"not a NACA designation: {code!r}")
    return s


def _assemble(x: np.ndarray, yc: np.ndarray, dyc: np.ndarray, yt: np.ndarray, chord: float,
              closed_te: bool) -> np.ndarray:
    """Thickness perpendicular to the camber line; Selig order; scale to chord."""
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    upper = np.column_stack([xu[::-1], yu[::-1]])          # TE -> LE (includes LE)
    lower = np.column_stack([xl[1:], yl[1:]])              # after LE -> TE
    if closed_te:
        lower = lower[:-1]                                  # TE already present as point 0
    pts = np.vstack([upper, lower]) * chord
    pts[np.abs(pts) < 1e-15 * chord] = 0.0
    return pts


def naca4(code: str, n: int = 160, chord: float = 1.0, closed_te: bool = True) -> np.ndarray:
    """NACA 4-digit section ``MPTT`` (e.g. "2412"), Selig order, CCW.

    ``n`` is the number of output points: ``2 * (n // 2)`` for a closed trailing edge
    (the shared TE point appears once) and one more for a blunt one. Stations use
    cosine spacing. M = max camber in % chord, P = its position in tenths of chord,
    TT = thickness in % chord.
    """
    s = _clean_code(code)
    if len(s) != 4:
        raise ValueError(f"NACA 4-digit code must have 4 digits, got {code!r}")
    m, p, t = int(s[0]) / 100.0, int(s[1]) / 10.0, int(s[2:]) / 100.0
    if t <= 0:
        raise ValueError("thickness must be positive")
    if m > 0 and not 0 < p < 1:
        raise ValueError("a cambered 4-digit section needs a camber position P between 1 and 9")
    x = cosine_spacing(max(int(n) // 2, 4))
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0:
        f = x < p
        yc[f] = m / p**2 * (2 * p * x[f] - x[f] ** 2)
        dyc[f] = 2 * m / p**2 * (p - x[f])
        b = ~f
        yc[b] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[b] - x[b] ** 2)
        dyc[b] = 2 * m / (1 - p) ** 2 * (p - x[b])
    return _assemble(x, yc, dyc, naca_thickness(x, t, closed_te), chord, closed_te)


def naca5(code: str, n: int = 160, chord: float = 1.0, closed_te: bool = True) -> np.ndarray:
    """NACA 5-digit section ``LPSTT`` (e.g. "23012" standard, "23112" reflex).

    L: design lift coefficient = 0.15 L; P: max-camber position = 0.05 P (P = 1..5);
    S: 0 = standard mean line, 1 = reflexed (P = 2..5); TT: thickness in % chord.
    Uses the published (r, k1, k2/k1) constants for design CL = 0.3, scaled
    linearly with the design CL as in the original NACA report.
    """
    s = _clean_code(code)
    if len(s) != 5:
        raise ValueError(f"NACA 5-digit code must have 5 digits, got {code!r}")
    L, P, S, t = int(s[0]), int(s[1]), int(s[2]), int(s[3:]) / 100.0
    if t <= 0:
        raise ValueError("thickness must be positive")
    if S not in (0, 1):
        raise ValueError("third digit of a 5-digit code must be 0 (standard) or 1 (reflex)")
    cl_scale = (0.15 * L) / 0.3
    x = cosine_spacing(max(int(n) // 2, 4))
    if S == 0:
        if P not in _FIVE_STD:
            raise ValueError(f"no standard 5-digit mean line for P = {P} (use 1..5)")
        r, k1 = _FIVE_STD[P]
        f = x < r
        yc = np.where(f, k1 / 6 * (x**3 - 3 * r * x**2 + r**2 * (3 - r) * x), k1 * r**3 / 6 * (1 - x))
        dyc = np.where(f, k1 / 6 * (3 * x**2 - 6 * r * x + r**2 * (3 - r)), -k1 * r**3 / 6)
    else:
        if P not in _FIVE_REFLEX:
            raise ValueError(f"no reflex 5-digit mean line for P = {P} (use 2..5)")
        r, k1, k21 = _FIVE_REFLEX[P]
        f = x < r
        yc = np.where(
            f,
            k1 / 6 * ((x - r) ** 3 - k21 * (1 - r) ** 3 * x - r**3 * x + r**3),
            k1 / 6 * (k21 * (x - r) ** 3 - k21 * (1 - r) ** 3 * x - r**3 * x + r**3))
        dyc = np.where(
            f,
            k1 / 6 * (3 * (x - r) ** 2 - k21 * (1 - r) ** 3 - r**3),
            k1 / 6 * (3 * k21 * (x - r) ** 2 - k21 * (1 - r) ** 3 - r**3))
    yc = yc * cl_scale
    dyc = dyc * cl_scale
    return _assemble(x, yc, dyc, naca_thickness(x, t, closed_te), chord, closed_te)


def naca(code: str, n: int = 160, chord: float = 1.0, closed_te: bool = True) -> np.ndarray:
    """Dispatch on the number of digits: 4 -> :func:`naca4`, 5 -> :func:`naca5`."""
    s = _clean_code(code)
    if len(s) == 4:
        return naca4(s, n, chord, closed_te)
    if len(s) == 5:
        return naca5(s, n, chord, closed_te)
    raise ValueError(f"unsupported NACA designation {code!r} (4 or 5 digits)")


# ---------------------------------------------------------------------------
# .dat files
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?$")


def _parse_numbers(line: str) -> list[float] | None:
    """Two-or-more numbers on a line, tolerating tabs, ';' and decimal commas.

    "0,5  0,25" (decimal commas, whitespace separated) and "0.5, 0.25" (comma as a
    separator) are both accepted; anything with a non-numeric token is a header.
    """
    s = line.strip()
    if not s:
        return None
    toks = [t for t in re.split(r"[\s;]+", s) if t]
    if len(toks) >= 2 and all("," in t and "." not in t for t in toks):
        toks = [t.replace(",", ".") for t in toks]           # decimal commas
    else:
        toks = [t for t in re.split(r"[\s;,]+", s) if t]
    vals = []
    for t in toks:
        if not _NUM_RE.match(t):
            return None
        vals.append(float(t.replace("d", "e").replace("D", "e")))
    return vals if len(vals) >= 2 else None


def _read_text(path_or_text) -> tuple[str, str]:
    """(text, fallback_name). Accepts a path, a Path, raw text, or a file object."""
    if hasattr(path_or_text, "read"):
        data = path_or_text.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return data, ""
    if isinstance(path_or_text, Path) or (
            isinstance(path_or_text, str) and "\n" not in path_or_text and len(path_or_text) < 1024
            and os.path.exists(path_or_text)):
        p = Path(path_or_text)
        return p.read_bytes().decode("utf-8", errors="replace"), p.stem
    if isinstance(path_or_text, str):
        return path_or_text, ""
    raise TypeError("pass a path, a file object or the file text")


def read_airfoil_dat(path_or_text) -> tuple[str, np.ndarray]:
    """Read an airfoil coordinate file in Selig or Lednicer format.

    *Selig*: optional name line(s), then x y pairs from the upper trailing edge
    around the leading edge to the lower trailing edge. *Lednicer*: name, a line
    with the point counts of the upper and lower surfaces (e.g. "61. 61."), then
    the upper surface LE -> TE, a blank line, and the lower surface LE -> TE. The
    format is recognised from that count line (both values >= 2 and integral).

    Returns ``(name, coords)`` with coords in Selig order, CCW, implicitly closed (a
    repeated TE point and the duplicated Lednicer LE point are removed). No
    normalisation is applied -- see :func:`aerolume.geometry.design_from_airfoil`.
    """
    text, fallback = _read_text(path_or_text)
    name_lines: list[str] = []
    rows: list[list[float]] = []
    for raw in io.StringIO(text):
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        vals = _parse_numbers(line)
        if vals is None:
            if not rows:
                name_lines.append(line)
            continue
        rows.append(vals[:2])
    if not rows:
        raise ValueError("no coordinate pairs found")
    name = (name_lines[0] if name_lines else fallback) or "Airfoil"
    first = rows[0]
    is_led = (first[0] >= 2 and first[1] >= 2 and float(first[0]).is_integer() and float(first[1]).is_integer()
              and len(rows) >= int(first[0]) + int(first[1]) + 1)
    if is_led:
        nu, nl = int(first[0]), int(first[1])
        pts = np.array(rows[1:], dtype=float)
        upper = pts[:nu]                       # LE -> TE
        lower = pts[nu:nu + nl]                # LE -> TE
        if len(lower) and np.allclose(lower[0], upper[0], atol=1e-12):
            lower = lower[1:]
        coords = np.vstack([upper[::-1], lower])
    else:
        coords = np.array(rows, dtype=float)
    coords = as_polygon(coords)                # drops a repeated closing TE point
    if len(coords) < 3:
        raise ValueError("airfoil file has fewer than 3 points")
    return name, orient_ccw(coords)


def format_airfoil_dat(name: str, coords: np.ndarray, fmt: str = "selig", digits: int = 7) -> str:
    """Text of a .dat file (``fmt`` "selig" or "lednicer") for Selig-ordered coords.

    A sharp trailing edge is written at both ends (the convention every airfoil
    tool expects); :func:`read_airfoil_dat` drops the repeat again. The leading
    edge (smallest x) splits the surfaces for the Lednicer layout.
    """
    p = as_polygon(coords)
    f = f"{{:.{digits}f}}"

    def line(q):
        return f"{f.format(q[0]).rjust(digits + 4)} {f.format(q[1]).rjust(digits + 4)}"

    if fmt == "selig":
        pts = p if _blunt(p) else np.vstack([p, p[:1]])
        return name.strip() + "\n" + "\n".join(line(q) for q in pts) + "\n"
    if fmt == "lednicer":
        ile = int(np.argmin(p[:, 0]))
        upper = p[:ile + 1][::-1]                       # LE -> upper TE
        lower = p[ile:]                                 # LE -> lower TE
        if not _blunt(p):
            lower = np.vstack([lower, p[:1]])           # sharp TE: close onto point 0
        out = [name.strip(), f"{len(upper):d}. {len(lower):d}.", ""]
        out += [line(q) for q in upper] + [""] + [line(q) for q in lower]
        return "\n".join(out) + "\n"
    raise ValueError("fmt must be 'selig' or 'lednicer'")


def _blunt(p: np.ndarray) -> bool:
    """True when the implicit closing edge is a blunt trailing-edge base.

    In Selig order the closing edge runs from the lower to the upper TE point. For a
    blunt TE it is roughly perpendicular to the chord; for a sharp TE (point 0
    shared) it is just the last lower-surface panel, roughly along the chord.
    """
    xr = p[:, 0].max() - p[:, 0].min()
    d = p[0] - p[-1]
    near_te = p[-1, 0] > p[:, 0].max() - 0.02 * xr and p[0, 0] > p[:, 0].max() - 0.02 * xr
    return bool(near_te and abs(d[1]) > abs(d[0]))


def write_airfoil_dat(path, name: str, coords: np.ndarray, fmt: str = "selig") -> Path:
    """Write :func:`format_airfoil_dat` output to ``path`` and return it."""
    path = Path(path)
    path.write_text(format_airfoil_dat(name, coords, fmt), encoding="utf-8")
    return path
