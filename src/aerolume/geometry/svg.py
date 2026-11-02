"""SVG import: every shape element becomes a polygon.

A small, dependency-free SVG reader built on :mod:`xml.etree`. It understands what
drawing programs (Inkscape, Illustrator, Figma, Affinity) actually write for an
outline: ``<path>`` with all commands M L H V C S Q T A Z in absolute and relative
form, ``<polygon>``, ``<polyline>``, ``<line>``, ``<rect>`` (with rounded corners),
``<circle>``, ``<ellipse>``, nested ``<g>`` / ``<svg>`` with ``transform`` lists
(matrix, translate, scale, rotate about a point, skewX, skewY) and the root
``viewBox``. Text, images, gradients, ``<defs>`` / ``<use>`` clones, clip paths and
hidden elements are ignored.

Curves are handled exactly up to the last step: transforms are affine, so they are
applied to the Bezier control points, arcs are converted to cubic Beziers first,
and only then is every curve flattened by recursive subdivision until it deviates
from its chord by less than ``tolerance`` (in output units). That keeps a nose
radius drawn as a tiny arc as smooth as the long roof line.

SVG's y axis points down; output is flipped to Aerolume's y-up frame, so a car
drawn facing left in the editor still faces left (into the flow) after import.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .polygon import close_gaps, orient_ccw, remove_duplicates, signed_area

__all__ = ["parse_path_data", "parse_transform", "read_svg"]

_SKIP = {"defs", "clipPath", "mask", "symbol", "marker", "pattern", "metadata", "title", "desc",
         "text", "image", "style", "script", "linearGradient", "radialGradient", "filter",
         "foreignObject", "use"}
_UNITS_M = {"": 0.0254 / 96, "px": 0.0254 / 96, "pt": 0.0254 / 72, "pc": 0.0254 / 6, "mm": 1e-3,
            "cm": 1e-2, "in": 0.0254, "m": 1.0}

Mat = np.ndarray   # 3x3 affine matrix


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def _mat(a, b, c, d, e, f) -> Mat:
    return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])


def parse_transform(text: str | None) -> Mat:
    """3x3 matrix of an SVG ``transform`` attribute (list applied right to left)."""
    m = np.eye(3)
    if not text:
        return m
    for name, args in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", text):
        v = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        name = name.strip()
        if name == "matrix" and len(v) == 6:
            t = _mat(*v)
        elif name == "translate":
            t = _mat(1, 0, 0, 1, v[0] if v else 0.0, v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            sx = v[0] if v else 1.0
            sy = v[1] if len(v) > 1 else sx
            t = _mat(sx, 0, 0, sy, 0, 0)
        elif name == "rotate":
            a = math.radians(v[0] if v else 0.0)
            c, s = math.cos(a), math.sin(a)
            t = _mat(c, s, -s, c, 0, 0)
            if len(v) >= 3:
                cx, cy = v[1], v[2]
                t = _mat(1, 0, 0, 1, cx, cy) @ t @ _mat(1, 0, 0, 1, -cx, -cy)
        elif name == "skewX":
            t = _mat(1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY":
            t = _mat(1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = m @ t
    return m


def _apply(m: Mat, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:2, :2].T + m[:2, 2]


# ---------------------------------------------------------------------------
# path data
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


class _Lexer:
    """Cursor over path data. Arc flags may be packed ("a1 1 0 00 1 1") so they are
    read character-wise instead of as numbers."""

    def __init__(self, d: str) -> None:
        self.s = d
        self.i = 0

    def _skip(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n,":
            self.i += 1

    def peek_command(self) -> str | None:
        self._skip()
        if self.i < len(self.s) and self.s[self.i].isalpha() and self.s[self.i] not in "eE":
            return self.s[self.i]
        return None

    def command(self) -> str | None:
        c = self.peek_command()
        if c is not None:
            self.i += 1
        return c

    def has_number(self) -> bool:
        self._skip()
        return self.i < len(self.s) and (self.s[self.i].isdigit() or self.s[self.i] in "+-.")

    def number(self) -> float:
        self._skip()
        m = _TOKEN.match(self.s, self.i)
        if not m or m.group(2) is None:
            raise ValueError(f"bad path data near {self.s[self.i:self.i + 20]!r}")
        self.i = m.end()
        return float(m.group(2))

    def flag(self) -> int:
        self._skip()
        ch = self.s[self.i: self.i + 1]
        if ch not in ("0", "1"):
            raise ValueError(f"bad arc flag near {self.s[self.i:self.i + 20]!r}")
        self.i += 1
        return int(ch)


def _arc_to_cubics(p0, rx, ry, phi_deg, large, sweep, p1) -> list[np.ndarray]:
    """SVG endpoint arc -> list of cubic Bezier control polygons (4x2), <= 22.5 deg each.

    A cubic matches a circular arc of angle a to a relative radial error of about
    (a/4)^6 / 54: 2.7e-4 for the common 90-degree split, 6.6e-8 at 22.5 degrees,
    which keeps the approximation far below any flattening tolerance. Implements the SVG 1.1 implementation notes (F.6.5 / F.6.6), including the
    radius correction when the given radii are too small to span the endpoints.
    """
    x1, y1 = p0
    x2, y2 = p1
    if (x1, y1) == (x2, y2):
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0:
        return [np.array([p0, p0 + (p1 - p0) / 3, p0 + 2 * (p1 - p0) / 3, p1])]
    phi = math.radians(phi_deg % 360.0)
    cphi, sphi = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2, (y1 - y2) / 2
    x1p = cphi * dx + sphi * dy
    y1p = -sphi * dx + cphi * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    coef = math.sqrt(max(num / den, 0.0)) if den > 0 else 0.0
    if large == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cphi * cxp - sphi * cyp + (x1 + x2) / 2
    cy = sphi * cxp + cphi * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi
    nseg = max(1, math.ceil(abs(dth) / (math.pi / 8) - 1e-9))
    d = dth / nseg
    k = 4.0 / 3.0 * math.tan(d / 4)
    out = []
    t = th1
    for _ in range(nseg):
        c1, s1 = math.cos(t), math.sin(t)
        c2, s2 = math.cos(t + d), math.sin(t + d)
        e = np.array([[c1, s1], [c1 - k * s1, s1 + k * c1], [c2 + k * s2, s2 - k * c2], [c2, s2]])
        e[:, 0] *= rx
        e[:, 1] *= ry
        pts = np.column_stack([cphi * e[:, 0] - sphi * e[:, 1] + cx, sphi * e[:, 0] + cphi * e[:, 1] + cy])
        out.append(pts)
        t += d
    out[0][0] = p0
    out[-1][-1] = p1
    return out


def parse_path_data(d: str) -> list[tuple[list[np.ndarray], bool]]:
    """Parse SVG path data into subpaths of Bezier pieces.

    Returns ``[(pieces, closed)]`` where each piece is a control polygon: 2 points
    for a line, 3 for a quadratic, 4 for a cubic, in the path's own (untransformed,
    y-down) coordinates.
    """
    lx = _Lexer(d)
    subpaths: list[tuple[list[np.ndarray], bool]] = []
    pieces: list[np.ndarray] = []
    cur = np.zeros(2)
    start = np.zeros(2)
    last_c2: np.ndarray | None = None      # reflected control point for S
    last_q: np.ndarray | None = None       # for T
    cmd: str | None = None

    def flush(closed: bool):
        nonlocal pieces
        if pieces:
            subpaths.append((pieces, closed))
        pieces = []

    while True:
        c = lx.command()
        if c is None:
            if cmd is None or not lx.has_number():
                break
            c = {"M": "L", "m": "l"}.get(cmd, cmd)    # implicit repetition
        cmd = c
        rel = c.islower()
        C = c.upper()
        base = cur if rel else np.zeros(2)
        if C == "Z":
            if pieces:
                if np.hypot(*(cur - start)) > 0:
                    pieces.append(np.array([cur, start]))
                flush(True)
            cur = start.copy()
            last_c2 = last_q = None
            cmd = None
            continue
        if C == "M":
            flush(False)
            cur = base + np.array([lx.number(), lx.number()])
            start = cur.copy()
            last_c2 = last_q = None
            continue
        if C == "L":
            p = base + np.array([lx.number(), lx.number()])
            pieces.append(np.array([cur, p]))
            cur = p
            last_c2 = last_q = None
        elif C == "H":
            x = lx.number() + (cur[0] if rel else 0.0)
            p = np.array([x, cur[1]])
            pieces.append(np.array([cur, p]))
            cur = p
            last_c2 = last_q = None
        elif C == "V":
            y = lx.number() + (cur[1] if rel else 0.0)
            p = np.array([cur[0], y])
            pieces.append(np.array([cur, p]))
            cur = p
            last_c2 = last_q = None
        elif C == "C":
            c1 = base + np.array([lx.number(), lx.number()])
            c2 = base + np.array([lx.number(), lx.number()])
            p = base + np.array([lx.number(), lx.number()])
            pieces.append(np.array([cur, c1, c2, p]))
            cur, last_c2, last_q = p, c2, None
        elif C == "S":
            c1 = 2 * cur - last_c2 if last_c2 is not None else cur.copy()
            c2 = base + np.array([lx.number(), lx.number()])
            p = base + np.array([lx.number(), lx.number()])
            pieces.append(np.array([cur, c1, c2, p]))
            cur, last_c2, last_q = p, c2, None
        elif C == "Q":
            q = base + np.array([lx.number(), lx.number()])
            p = base + np.array([lx.number(), lx.number()])
            pieces.append(np.array([cur, q, p]))
            cur, last_q, last_c2 = p, q, None
        elif C == "T":
            q = 2 * cur - last_q if last_q is not None else cur.copy()
            p = base + np.array([lx.number(), lx.number()])
            pieces.append(np.array([cur, q, p]))
            cur, last_q, last_c2 = p, q, None
        elif C == "A":
            rx, ry, phi = lx.number(), lx.number(), lx.number()
            large, sweep = lx.flag(), lx.flag()
            p = base + np.array([lx.number(), lx.number()])
            pieces.extend(_arc_to_cubics(cur, rx, ry, phi, large, sweep, p))
            cur = p
            last_c2 = last_q = None
        else:
            raise ValueError(f"unknown path command {c!r}")
    flush(False)
    return subpaths


# ---------------------------------------------------------------------------
# flattening
# ---------------------------------------------------------------------------

def _flatten_piece(ctrl: np.ndarray, tol: float, out: list[np.ndarray], depth: int = 0) -> None:
    """Append points of one Bezier piece (excluding its start) to ``out``.

    Subdivides at t = 1/2 (de Casteljau) until every control point lies within
    ``tol`` of the chord -- the control polygon bounds the curve, so the flattened
    polyline is then within ``tol`` of the true curve.
    """
    if len(ctrl) == 2:
        out.append(ctrl[1])
        return
    a, b = ctrl[0], ctrl[-1]
    ab = b - a
    L = math.hypot(*ab)
    inner = ctrl[1:-1]
    if L > 0:
        dev = np.abs(ab[0] * (inner[:, 1] - a[1]) - ab[1] * (inner[:, 0] - a[0])).max() / L
    else:
        dev = np.hypot(*(inner - a).T).max()
    if dev <= tol or depth >= 18:
        out.append(b)
        return
    # de Casteljau split
    levels = [ctrl]
    while len(levels[-1]) > 1:
        q = levels[-1]
        levels.append(0.5 * (q[:-1] + q[1:]))
    left = np.array([lv[0] for lv in levels])
    right = np.array([lv[-1] for lv in levels[::-1]])
    _flatten_piece(left, tol, out, depth + 1)
    _flatten_piece(right, tol, out, depth + 1)


# ---------------------------------------------------------------------------
# document walk
# ---------------------------------------------------------------------------

def _tag(el: ET.Element) -> str:
    t = el.tag
    return t.split("}", 1)[1] if isinstance(t, str) and "}" in t else str(t)


def _length(v: str | None, default: float = 0.0) -> float:
    if v is None or v.strip() == "":
        return default
    m = re.match(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", v)
    return float(m.group(1)) if m else default


def _style(el: ET.Element) -> dict[str, str]:
    st = {}
    for item in (el.get("style") or "").split(";"):
        if ":" in item:
            k, v = item.split(":", 1)
            st[k.strip()] = v.strip()
    return st


def _hidden(el: ET.Element) -> bool:
    st = _style(el)
    return (el.get("display") == "none" or st.get("display") == "none"
            or el.get("visibility") == "hidden" or st.get("visibility") == "hidden")


def _ellipse_pieces(cx, cy, rx, ry) -> list[tuple[list[np.ndarray], bool]]:
    """Full ellipse as two arcs (so it gets the same accurate Bezier split as paths)."""
    d = f"M{cx + rx},{cy} A{rx},{ry} 0 1 1 {cx - rx},{cy} A{rx},{ry} 0 1 1 {cx + rx},{cy} Z"
    return parse_path_data(d)


def _rect_pieces(x, y, w, h, rx, ry) -> list[tuple[list[np.ndarray], bool]]:
    if rx <= 0 and ry <= 0:
        P = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
        return [([np.array([P[i], P[(i + 1) % 4]]) for i in range(4)], True)]
    rx = rx if rx > 0 else ry
    ry = ry if ry > 0 else rx
    rx, ry = min(rx, w / 2), min(ry, h / 2)
    d = (f"M{x + rx},{y} H{x + w - rx} A{rx},{ry} 0 0 1 {x + w},{y + ry} V{y + h - ry} "
         f"A{rx},{ry} 0 0 1 {x + w - rx},{y + h} H{x + rx} A{rx},{ry} 0 0 1 {x},{y + h - ry} "
         f"V{y + ry} A{rx},{ry} 0 0 1 {x + rx},{y} Z")
    return parse_path_data(d)


def _points_attr(v: str | None) -> np.ndarray:
    nums = [float(t) for t in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", v or "")]
    return np.array(nums[: len(nums) // 2 * 2], float).reshape(-1, 2)


def _element_subpaths(el: ET.Element, tag: str) -> list[tuple[list[np.ndarray], bool]]:
    g = el.get
    if tag == "path":
        return parse_path_data(g("d") or "")
    if tag in ("polygon", "polyline"):
        P = _points_attr(g("points"))
        if len(P) < 2:
            return []
        segs = [np.array([P[i], P[i + 1]]) for i in range(len(P) - 1)]
        closed = tag == "polygon"
        if closed and np.hypot(*(P[-1] - P[0])) > 0:
            segs.append(np.array([P[-1], P[0]]))
        return [(segs, closed)]
    if tag == "line":
        a = np.array([_length(g("x1")), _length(g("y1"))])
        b = np.array([_length(g("x2")), _length(g("y2"))])
        return [([np.array([a, b])], False)]
    if tag == "rect":
        w, h = _length(g("width")), _length(g("height"))
        if w <= 0 or h <= 0:
            return []
        return _rect_pieces(_length(g("x")), _length(g("y")), w, h, _length(g("rx"), -1), _length(g("ry"), -1))
    if tag == "circle":
        r = _length(g("r"))
        return _ellipse_pieces(_length(g("cx")), _length(g("cy")), r, r) if r > 0 else []
    if tag == "ellipse":
        rx, ry = _length(g("rx")), _length(g("ry"))
        return _ellipse_pieces(_length(g("cx")), _length(g("cy")), rx, ry) if rx > 0 and ry > 0 else []
    return []


def _walk(el: ET.Element, m: Mat, out: list[tuple[list[np.ndarray], bool]], root: bool) -> None:
    tag = _tag(el)
    if tag in _SKIP or _hidden(el):
        return
    local = m @ parse_transform(el.get("transform"))
    if tag == "svg" and not root:
        local = local @ _mat(1, 0, 0, 1, _length(el.get("x")), _length(el.get("y")))
    for sub, closed in _element_subpaths(el, tag):
        out.append(([_apply(local, c) for c in sub], closed))
    for child in el:
        _walk(child, local, out, False)


def _root_matrix(root: ET.Element, units: str) -> Mat:
    """Map root user units to output units, flipping y up.

    ``units="user"`` keeps viewBox user units (the drawing's own numbers). ``"m"``
    converts with the root width/height (mm, cm, in, pt, px at 96 dpi), useful when a
    CAD-like SVG was drawn to scale.
    """
    vb = [float(v) for v in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()) if v]
    flip = _mat(1, 0, 0, -1, 0, 0)
    if units == "user":
        if len(vb) == 4:
            return _mat(1, 0, 0, -1, 0, vb[1] + vb[3])       # keep y >= 0 inside the viewBox
        return flip
    if units != "m":
        raise ValueError("units must be 'user' or 'm'")

    def phys(v: str | None) -> float | None:
        if not v:
            return None
        mm = re.match(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-z%]*)", v)
        if not mm or mm.group(2) == "%":
            return None
        return float(mm.group(1)) * _UNITS_M.get(mm.group(2), _UNITS_M[""])

    w, h = phys(root.get("width")), phys(root.get("height"))
    if len(vb) == 4 and w and h:
        sx, sy = w / vb[2], h / vb[3]
        return _mat(sx, 0, 0, -sy, -sx * vb[0], sy * (vb[1] + vb[3]))
    s = _UNITS_M[""]
    return _mat(s, 0, 0, -s, 0, 0)


def read_svg(path, *, tolerance: float | None = None, units: str = "user",
             notes: list[str] | None = None) -> list[np.ndarray]:
    """Read every shape of an SVG file as a CCW, implicitly closed polygon.

    ``tolerance``: maximum deviation of the flattened polyline from the true curves,
    in output units; default 2e-4 of the drawing's size (about 1 mm on a 4.5 m car).
    ``units``: "user" (viewBox units, y flipped up) or "m" (physical size from the
    root width/height). Open subpaths are chained end-to-end where their endpoints
    meet and any remaining gap is closed with a straight segment -- a message is
    appended to ``notes`` when that happens. Degenerate shapes (zero area) are
    dropped. ``path`` may also be the SVG text itself.
    """
    src = str(path)
    if isinstance(path, (str, Path)) and not src.lstrip().startswith("<"):
        tree_root = ET.parse(str(path)).getroot()
    else:
        tree_root = ET.fromstring(src)
    if _tag(tree_root) != "svg":
        raise ValueError("not an SVG document")
    raw: list[tuple[list[np.ndarray], bool]] = []
    _walk(tree_root, _root_matrix(tree_root, units), raw, True)
    raw = [(p, c) for p, c in raw if p]
    if not raw:
        return []
    allctrl = np.vstack([c for pieces, _ in raw for c in pieces])
    size = max(float(np.ptp(allctrl, axis=0).max()), 1e-300)
    tol = tolerance if tolerance and tolerance > 0 else 2e-4 * size

    closed_polys: list[np.ndarray] = []
    open_polys: list[np.ndarray] = []
    for pieces, closed in raw:
        pts = [pieces[0][0]]
        for c in pieces:
            _flatten_piece(np.asarray(c, float), tol, pts)
        P = np.array(pts)
        if not closed and len(P) > 2 and np.hypot(*(P[-1] - P[0])) <= 1e-6 * size:
            closed = True
        (closed_polys if closed else open_polys).append(P)
    polys = list(closed_polys)
    if open_polys:
        polys += close_gaps(open_polys, tol=max(1e-6 * size, 0.5 * tol), notes=notes)
    out = []
    for P in polys:
        P = remove_duplicates(P, tol=1e-9 * size)
        if len(P) >= 3 and abs(signed_area(P)) > 1e-12 * size * size:
            out.append(orient_ccw(P))
    return out
