"""Parametric starting shapes: vehicles, airfoils and basic bodies.

Each template is described by a :class:`TemplateInfo` -- name, kind, a sentence of
description and a list of :class:`ParamSpec` -- so the GUI can build its form
automatically, and ``make(name, **params)`` returns a ready :class:`Design` with
its reference quantities set (DESIGN.md section 3).

Vehicle side profiles are drawn like a designer would with a pen tool: a closed
chain of nodes joined by straight lines or cubic Bezier curves, with tangents
continuous at smooth nodes and broken only at deliberate sharp edges (a boot-lid
trailing edge, a spoiler lip, the rear edges of a box). Sharp rear edges matter
aerodynamically -- they fix where the flow separates -- so they are kept exact
when the outline is resampled. The result is smooth (no jaggies, no faceting
beyond the point spacing) and never self-intersecting for parameters inside the
published ranges.

Wheels are optional and **off by default**. In a 2D side-profile simulation a
wheel is an infinitely wide cylinder: if it touches the road it seals the
underbody channel completely, so no air flows under the car at all -- unlike a
real car, where air passes between and around four narrow wheels. Without
wheels the 2D underbody flow is a fairer stand-in. Turn them on to study wheel
arches or rotating-wheel effects, knowing the underbody is then blocked (a small
``wheel_gap`` above the road keeps a thin channel open instead).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..model import Body, Design
from . import polygon as P
from .airfoils import naca
from .design import design_from_airfoil

__all__ = ["UNIT_KINDS", "ParamSpec", "TemplateInfo", "get_template", "list_templates", "make"]

# unit_kind values understood by the GUI form builder:
#   "length", "angle"            -> SI value, unit conversion via aerolume.core.units
#   "ratio"                      -> dimensionless float
#   "count"                      -> integer
#   "bool"                       -> checkbox
#   "text"                       -> free text (e.g. a NACA code)
#   "choice"                     -> one of ParamSpec.choices
UNIT_KINDS = ("length", "angle", "ratio", "count", "bool", "text", "choice")


@dataclass(frozen=True)
class ParamSpec:
    """One template parameter. ``min`` / ``max`` / ``step`` are None for non-numeric kinds."""
    name: str
    label: str
    unit_kind: str
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    help: str = ""


@dataclass(frozen=True)
class TemplateInfo:
    """A template's catalogue entry. ``category`` groups the GUI gallery."""
    name: str
    kind: str
    description: str
    params: list[ParamSpec] = field(default_factory=list)
    label: str = ""
    category: str = ""

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


# ---------------------------------------------------------------------------
# pen-tool outline builder
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = math.hypot(v[0], v[1])
    return v / n if n > 0 else v


class _Pen:
    """Closed outline of nodes joined by lines or cubic Bezier curves.

    ``add(x, y, seg=..., corner=..., tangent=...)`` appends a node; ``seg`` is the
    type of the segment *leaving* the node ("line" or "curve"). Tangents at smooth
    nodes are continuous: an explicit ``tangent`` (degrees, direction of travel)
    wins, else the direction of an adjacent straight line, else the Catmull-Rom
    direction ``P[i+1] - P[i-1]``. At ``corner`` nodes each side uses its own chord
    direction. Curves get Bezier handles of one third of the chord.
    """

    def __init__(self) -> None:
        self.nodes: list[dict] = []

    def add(self, x: float, y: float, *, seg: str = "curve", corner: bool = False,
            tangent: float | None = None) -> _Pen:
        self.nodes.append({"p": np.array([x, y], float), "seg": seg, "corner": corner, "t": tangent})
        return self

    def _tangents(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.nodes)
        nd, prv, nxt = self.nodes[i], self.nodes[i - 1], self.nodes[(i + 1) % n]
        if nd["t"] is not None:
            a = math.radians(nd["t"])
            t = np.array([math.cos(a), math.sin(a)])
            return t, t
        d_in = _unit(nd["p"] - prv["p"])
        d_out = _unit(nxt["p"] - nd["p"])
        if nd["corner"]:
            return d_in, d_out
        if prv["seg"] == "line":
            return d_in, d_in
        if nd["seg"] == "line":
            return d_out, d_out
        t = _unit(nxt["p"] - prv["p"])
        return t, t

    def build(self, spacing: float) -> tuple[np.ndarray, list[int]]:
        """Dense polyline (spacing ~``spacing``) and the indices of corner nodes."""
        n = len(self.nodes)
        tans = [self._tangents(i) for i in range(n)]
        pts: list[np.ndarray] = []
        corners: list[int] = []
        for i in range(n):
            a = self.nodes[i]
            b = self.nodes[(i + 1) % n]
            if a["corner"]:
                corners.append(len(pts))
            p0, p3 = a["p"], b["p"]
            d = math.hypot(*(p3 - p0))
            m = max(2, math.ceil(d / spacing))
            t = np.linspace(0.0, 1.0, m + 1)[:-1, None]
            if a["seg"] == "line":
                seg = p0 + t * (p3 - p0)
            else:
                p1 = p0 + tans[i][1] * d / 3.0
                p2 = p3 - tans[(i + 1) % n][0] * d / 3.0
                seg = ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3)
            pts.extend(seg)
        return np.array(pts), corners


def _finish(pen: _Pen, n: int, *, curvature: float = 0.3) -> np.ndarray:
    """Sample the pen outline, resample to ``n`` points keeping its corners, CCW."""
    pts, corners = pen.build(spacing=P.perimeter(np.array([nd["p"] for nd in pen.nodes])) / (40 * n))
    out = P.resample(pts, n, curvature=curvature, corners=corners)
    return P.orient_ccw(out)


def _fit_box(poly: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Affinely map the outline's bounding box onto [x0, x1] x [y0, y1] (exact extents).

    The pen's key points put the extremes where they belong to within a few mm;
    this tiny correction makes length, height and clearance *exactly* the values
    the user typed, which is what the reference quantities are built from.
    """
    bx0, bx1, by0, by1 = P.bbox(poly)
    q = poly.copy()
    q[:, 0] = x0 + (q[:, 0] - bx0) * (x1 - x0) / (bx1 - bx0)
    q[:, 1] = y0 + (q[:, 1] - by0) * (y1 - y0) / (by1 - by0)
    return q


# ---------------------------------------------------------------------------
# vehicles
# ---------------------------------------------------------------------------

_VEH_POINTS = 480


def _nose(pen: _Pen, *, y_n: float, r_n: float, y_hood: float, L: float) -> None:
    """Front: hood front -> rounded nose of radius r_n at height y_n -> apex.

    Nodes are added in CCW order (the outline runs bottom -> rear -> roof -> front,
    so this is called last); the closing curve from the apex back to the first
    node (the start of the flat underbody, tangent horizontal) forms the lower,
    convex part of the bumper. The upper part is a circle arc of radius ``r_n``
    (apex and 45 deg point with circle tangents, which a cubic Bezier follows to
    ~0.03 % of the radius).
    """
    s45 = math.sqrt(0.5)
    y45 = y_n + r_n * s45
    # the hood front sits just far enough back for a gentle blend from the arc
    pen.add(r_n * (1 - s45) + max(0.03 * L, 2.5 * (y_hood - y45)), y_hood)
    pen.add(r_n * (1 - s45), y45, tangent=225.0)
    pen.add(0.0, y_n, tangent=270.0)


def _underbody_start(r_n: float, y_n: float, c: float) -> float:
    """x where the flat floor starts: far enough back that the lower bumper is convex."""
    return max(1.1 * r_n, 0.6 * (y_n - c))


def _clamp_nose(r_n: float, c: float, y_n: float, y_hood: float, x_hood: float) -> float:
    s45 = math.sqrt(0.5)
    room = min((y_n - c) / (s45 + 0.35), (y_hood - y_n) / (s45 + 0.3), x_hood / (1 - s45 + 0.5))
    return max(min(r_n, room), 0.01 * (y_hood - c))


def _wheels(L: float, c: float, xs: tuple[float, float], r: float, gap: float, n: int = 96) -> list[Body]:
    """Two wheel bodies (front, rear), bottom ``gap`` above the road, spinning off."""
    out = []
    for name, xc in (("Front wheel", xs[0]), ("Rear wheel", xs[1])):
        th = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        yc = r + gap
        pts = np.column_stack([xc + r * np.cos(th), yc + r * np.sin(th)])
        out.append(Body(name=name, points=pts, role="wheel", center=(float(xc), float(yc))))
    return out


def _vehicle_design(name: str, body_pts: np.ndarray, L: float, c: float, H: float, *, template: str,
                    params: dict, wheels: list[Body] | None = None, notes: list[str] | None = None) -> Design:
    bodies = [Body(name="Body", points=body_pts, role="body")] + list(wheels or [])
    polys = [b.points for b in bodies]
    x0, x1, _, _ = P.bbox(polys)
    src = {"type": "template", "template": template, "params": dict(params), "ground": True}
    if notes:
        src["import_notes"] = list(notes)       # surfaced by check_design on the Geometry page
    return Design(name=name, kind="vehicle", bodies=bodies, length_ref=float(x1 - x0),
                  force_ref=float(P.frontal_height(polys)), moment_ref=(0.5 * (x0 + x1), 0.0), source=src)


def _fit_glass(cowl_x: float, rise1: float, ws: float, anchor_x: float, rise2: float, rw: float,
               min_roof: float, notes: list[str]) -> tuple[float, float]:
    """x of the roof front and rear from the windscreen / rear-glass angles.

    If the two glass runs leave less than ``min_roof`` of roof (shallow angles on a
    short or tall car), both runs are shortened by the same factor -- the glass gets
    steeper -- instead of producing a crossed outline, and a note says so. This keeps
    every combination of in-range parameters valid.
    """
    run1, run2 = rise1 / math.tan(ws), rise2 / math.tan(rw)
    avail = anchor_x - cowl_x - min_roof
    if run1 + run2 > avail:
        f = max(avail, 0.05 * (anchor_x - cowl_x)) / (run1 + run2)
        run1, run2 = run1 * f, run2 * f
        notes.append(f"The windscreen and rear glass were steepened to "
                     f"{math.degrees(math.atan2(rise1, run1)):.0f} and "
                     f"{math.degrees(math.atan2(rise2, run2)):.0f} deg so that the roof fits "
                     "this length and height.")
    return cowl_x + run1, anchor_x - run2


def _car(style: str, p: dict) -> Design:
    """Passenger cars: sedan, hatchback, fastback, suv (one body + optional wheels)."""
    L, H, c, r_n = p["length"], p["height"], p["ground_clearance"], p["nose_radius"]
    ws = math.radians(p["windscreen_angle"])
    rw = math.radians(p["rear_angle"])
    hb = H - c
    if hb <= 0.2 * H:
        raise ValueError("ground clearance leaves too little body height")
    # shared front geometry (fractions tuned against real side views)
    y_n = c + (0.27 if style == "fastback" else 0.33) * hb
    hood_y = {"sedan": 0.55, "hatchback": 0.56, "fastback": 0.42, "suv": 0.58}[style]
    y_hood = c + hood_y * hb
    x_hood = 0.07 * L
    cowl_x = {"sedan": 0.30, "hatchback": 0.30, "fastback": 0.36, "suv": 0.29}[style] * L
    cowl_y = c + {"sedan": 0.68, "hatchback": 0.66, "fastback": 0.56, "suv": 0.66}[style] * hb
    r_n = _clamp_nose(r_n, c, y_n, y_hood, x_hood)
    notes: list[str] = []
    pen = _Pen()
    x_fb = _underbody_start(r_n, y_n, c)       # underbody starts behind the nose curve
    pen.add(x_fb, c, seg="line")               # underbody (straight, flat floor)
    if style == "sedan":
        x_diff = 0.84 * L
        y_rb = c + 0.10 * hb
        deck_len = min(p.get("deck_length", 0.20 * L), 0.30 * L)
        if deck_len < p.get("deck_length", 0.0):
            notes.append(f"Boot length limited to {deck_len:.2f} m (30 % of the length).")
        deck_x = L - deck_len
        deck_y = c + 0.72 * hb
        roof_front_x, roof_rear_x = _fit_glass(cowl_x, H - cowl_y, ws, deck_x, H - deck_y, rw, 0.12 * L, notes)
        pen.add(x_diff, c, seg="line", corner=True)         # diffuser start
        pen.add(L - 0.03 * L, y_rb, corner=True)             # rear bumper bottom edge
        pen.add(L, c + 0.42 * hb)                            # bumper
        pen.add(L - 0.004 * L, deck_y - 0.03 * hb, corner=True, seg="curve")  # boot-lid trailing edge
        pen.add(deck_x, deck_y)                              # rear-window base
        pen.add(roof_rear_x, H - 0.012 * hb, tangent=180.0 - 0.5 * math.degrees(rw))
        pen.add(0.5 * (roof_rear_x + roof_front_x), H, tangent=180.0)
        pen.add(roof_front_x, H - 0.01 * hb, tangent=180.0 + 0.45 * math.degrees(ws))
    elif style in ("hatchback", "suv"):
        x_diff = 0.86 * L
        y_rb = c + 0.10 * hb
        tail_y = c + (0.52 if style == "hatchback" else 0.50) * hb   # bottom of the hatch glass
        roof_front_x, roof_end_x = _fit_glass(cowl_x, H - cowl_y, ws, L - 0.03 * L, H - 0.02 * hb - tail_y,
                                              rw, 0.18 * L, notes)
        pen.add(x_diff, c, seg="line", corner=True)
        pen.add(L - 0.02 * L, y_rb, corner=True)
        pen.add(L, c + 0.30 * hb)
        pen.add(L - 0.012 * L, tail_y, tangent=90.0 + 0.5 * (90 - math.degrees(rw)))
        pen.add(roof_end_x + 0.004 * L, H - 0.02 * hb, corner=True)  # roof spoiler lip (separation)
        pen.add(roof_end_x - 0.02 * L, H - 0.004 * hb, tangent=178.0)
        pen.add(0.5 * (roof_end_x + roof_front_x), H, tangent=180.0)
        pen.add(roof_front_x, H - 0.01 * hb, tangent=180.0 + 0.45 * math.degrees(ws))
    elif style == "fastback":
        x_diff = 0.78 * L
        y_rb = c + 0.16 * hb
        tail_y = c + 0.62 * hb
        tail_x = L
        roof_front_x, roof_rear_x = _fit_glass(cowl_x, H - cowl_y, ws, tail_x, H - tail_y, rw, 0.04 * L, notes)
        pen.add(x_diff, c, seg="line", corner=True)
        pen.add(L - 0.035 * L, y_rb, corner=True)
        pen.add(L - 0.002 * L, c + 0.40 * hb)
        pen.add(tail_x, tail_y, corner=True)                 # ducktail lip
        pen.add(roof_rear_x, H - 0.015 * hb, tangent=180.0 - 0.6 * math.degrees(rw))
        pen.add(0.5 * (roof_rear_x + roof_front_x), H, tangent=180.0)
        pen.add(roof_front_x, H - 0.012 * hb, tangent=180.0 + 0.45 * math.degrees(ws))
    else:
        raise ValueError(style)
    pen.add(cowl_x, cowl_y, tangent=180.0 + math.degrees(ws) * 0.55)       # windscreen base (cowl)
    _nose(pen, y_n=y_n, r_n=r_n, y_hood=y_hood, L=L)
    body = _fit_box(_finish(pen, _VEH_POINTS), 0.0, L, c, H)
    wheels = None
    if p.get("wheels"):
        r = p["wheel_radius"]
        xs = {"sedan": (0.19, 0.78), "hatchback": (0.17, 0.80), "fastback": (0.20, 0.80),
              "suv": (0.18, 0.80)}[style]
        wheels = _wheels(L, c, (xs[0] * L, xs[1] * L), r, p.get("wheel_gap", 0.0))
    return _vehicle_design(style.capitalize() if style != "suv" else "SUV", body, L, c, H,
                           template=style, params=p, wheels=wheels, notes=notes)


def _pickup(p: dict) -> Design:
    """Pickup truck: cab with an open bed (a real cavity) and a tailgate."""
    L, H, c, r_n = p["length"], p["height"], p["ground_clearance"], p["nose_radius"]
    ws = math.radians(p["windscreen_angle"])
    hb = H - c
    bed_len = p["bed_length"]
    y_n = c + 0.36 * hb
    y_hood = c + 0.62 * hb
    x_hood = 0.06 * L
    r_n = _clamp_nose(r_n, c, y_n, y_hood, x_hood)
    cowl_x, cowl_y = 0.25 * L, c + 0.70 * hb
    notes: list[str] = []
    roof_front_x = cowl_x + (H - cowl_y) / math.tan(ws)
    max_bed = L - (roof_front_x + 0.10 * L)
    if bed_len > max_bed:
        bed_len = max(max_bed, 0.15 * L)
        roof_front_x = min(roof_front_x, L - bed_len - 0.10 * L)
        notes.append(f"Bed length limited to {bed_len:.2f} m so the cab fits.")
    cab_back_x = L - bed_len
    bed_top = c + 0.60 * hb                     # top of bed sides / tailgate
    floor = c + 0.22 * hb                       # bed floor
    wall = max(0.012 * L, 0.05)                 # tailgate / headboard thickness
    pen = _Pen()
    pen.add(_underbody_start(r_n, y_n, c), c, seg="line")
    pen.add(0.86 * L, c, seg="line", corner=True)
    pen.add(L - 0.015 * L, c + 0.10 * hb, seg="line", corner=True)
    pen.add(L, c + 0.20 * hb, seg="line", corner=True)
    pen.add(L, bed_top, seg="line", corner=True)                      # tailgate top, rear edge
    pen.add(L - wall, bed_top, seg="line", corner=True)               # tailgate top, inner edge
    pen.add(L - wall, floor, seg="line", corner=True)                 # bed floor, rear
    pen.add(cab_back_x + wall, floor, seg="line", corner=True)        # bed floor, front
    pen.add(cab_back_x + wall, bed_top, seg="line", corner=True)      # headboard
    pen.add(cab_back_x, bed_top + 0.02 * hb, seg="line", corner=True)
    pen.add(cab_back_x, H - 0.06 * hb, corner=True)                   # cab back, top edge
    pen.add(cab_back_x - 0.04 * L, H, tangent=180.0)
    pen.add(roof_front_x, H - 0.01 * hb, tangent=180.0 + 0.45 * math.degrees(ws))
    pen.add(cowl_x, cowl_y, tangent=180.0 + math.degrees(ws) * 0.5)
    _nose(pen, y_n=y_n, r_n=r_n, y_hood=y_hood, L=L)
    body = _fit_box(_finish(pen, _VEH_POINTS), 0.0, L, c, H)
    wheels = _wheels(L, c, (0.17 * L, 0.74 * L), p["wheel_radius"], p.get("wheel_gap", 0.0)) \
        if p.get("wheels") else None
    return _vehicle_design("Pickup", body, L, c, H, template="pickup", params=p, wheels=wheels, notes=notes)


def _van(p: dict) -> Design:
    """Panel van / minibus: short sloping bonnet, steep windscreen, boxy rear."""
    L, H, c, r_n = p["length"], p["height"], p["ground_clearance"], p["nose_radius"]
    ws = math.radians(p["windscreen_angle"])
    hb = H - c
    y_n = c + 0.25 * hb
    y_hood = c + 0.42 * hb
    x_hood = 0.05 * L
    r_n = _clamp_nose(r_n, c, y_n, y_hood, x_hood)
    cowl_x, cowl_y = 0.15 * L, c + 0.50 * hb
    rr = max(min(p["rear_radius"], 0.3 * hb), 0.01 * hb)
    notes: list[str] = []
    roof_front_x = cowl_x + (H - cowl_y) / math.tan(ws)
    if roof_front_x > 0.6 * L:                  # very raked glass on a short, tall van
        roof_front_x = 0.6 * L
        notes.append(f"The windscreen was steepened to "
                     f"{math.degrees(math.atan2(H - cowl_y, roof_front_x - cowl_x)):.0f} deg so the roof fits.")
    pen = _Pen()
    pen.add(_underbody_start(r_n, y_n, c), c, seg="line")
    pen.add(0.90 * L, c, seg="line", corner=True)
    pen.add(L - 0.01 * L, c + 0.08 * hb, seg="line", corner=True)
    pen.add(L, c + 0.12 * hb, seg="line", corner=True)
    pen.add(L, H - rr, tangent=90.0)
    pen.add(L - rr * (1 - math.sqrt(0.5)), H - rr * (1 - math.sqrt(0.5)), tangent=135.0)
    pen.add(L - rr, H, seg="line", tangent=180.0)
    pen.add(roof_front_x + 0.04 * L, H, tangent=180.0)
    pen.add(roof_front_x, H - 0.015 * hb, tangent=180.0 + 0.45 * math.degrees(ws))
    pen.add(cowl_x, cowl_y, tangent=180.0 + math.degrees(ws) * 0.7)
    _nose(pen, y_n=y_n, r_n=r_n, y_hood=y_hood, L=L)
    body = _fit_box(_finish(pen, _VEH_POINTS), 0.0, L, c, H)
    wheels = _wheels(L, c, (0.17 * L, 0.78 * L), p["wheel_radius"], p.get("wheel_gap", 0.0)) \
        if p.get("wheels") else None
    return _vehicle_design("Van", body, L, c, H, template="van", params=p, wheels=wheels, notes=notes)


def _box_truck(p: dict) -> Design:
    """Box truck: cab plus a taller cargo box, one body with a step at the box front."""
    L, H, c = p["length"], p["height"], p["ground_clearance"]
    cab_h, cab_l = p["cab_height"], p["cab_length"]
    r_n = p["nose_radius"]
    notes: list[str] = []
    lim = (c + 0.4 * (H - c), H - 0.08 * (H - c))
    if not lim[0] <= cab_h <= lim[1]:
        cab_h = min(max(cab_h, lim[0]), lim[1])
        notes.append(f"Cab height adjusted to {cab_h:.2f} m to stay below the box.")
    if not 0.12 * L <= cab_l <= 0.45 * L:
        cab_l = min(max(cab_l, 0.12 * L), 0.45 * L)
        notes.append(f"Cab length adjusted to {cab_l:.2f} m (12-45 % of the length).")
    hc = cab_h - c
    r_n = min(r_n, 0.25 * hc)
    gap_x = cab_l + 0.02 * L                      # box front
    rb = max(min(p["box_edge_radius"], 0.2 * (H - cab_h)), 0.002 * L)   # >0 keeps nodes distinct
    pen = _Pen()
    pen.add(r_n, c, seg="line")
    pen.add(L - 0.01 * L, c, seg="line", corner=True)
    pen.add(L, c + 0.05 * (H - c), seg="line", corner=True)
    pen.add(L, H, seg="line", corner=True)                            # sharp rear top edge
    pen.add(gap_x + rb, H, tangent=180.0)
    pen.add(gap_x + rb * (1 - math.sqrt(0.5)), H - rb * (1 - math.sqrt(0.5)), tangent=225.0)
    pen.add(gap_x, H - rb, seg="line", tangent=270.0)
    pen.add(gap_x, cab_h, seg="line", corner=True)                    # step down onto the cab roof
    pen.add(0.25 * cab_l, cab_h, corner=True)                         # cab roof front edge
    pen.add(0.0, c + 0.55 * hc, seg="line", corner=True)              # windscreen (steep) base
    pen.add(0.0, c + r_n, tangent=270.0)
    pen.add(r_n * (1 - math.sqrt(0.5)), c + r_n * (1 - math.sqrt(0.5)), tangent=315.0)
    body = _fit_box(_finish(pen, _VEH_POINTS), 0.0, L, c, H)
    return _vehicle_design("Box truck", body, L, c, H, template="box_truck", params=p, notes=notes)


def _ahmed(p: dict) -> Design:
    """2D Ahmed-like bluff body: rounded front, flat roof, rear slant, square base.

    Built from exact arcs and lines (no splines) so it matches the canonical
    geometry: front corners of radius ``nose_radius``, a slant of angle
    ``slant_angle`` from the horizontal and length ``slant_length`` (measured along
    the slant, as in Ahmed et al. 1984), sharp slant and base edges.
    """
    L, Hb, c = p["length"], p["body_height"], p["ground_clearance"]
    r = min(p["nose_radius"], 0.5 * Hb, 0.4 * L)
    phi = math.radians(p["slant_angle"])
    ls = p["slant_length"]
    notes: list[str] = []
    # keep the slant inside the body: at most 90 % of the height and half the length
    lim = min(0.9 * Hb / max(math.sin(phi), 1e-12), 0.5 * (L - r) / max(math.cos(phi), 1e-12))
    if ls > lim:
        ls = lim
        notes.append(f"Slant length limited to {ls:.3f} m to fit the body.")
    drop = ls * math.sin(phi)
    run = ls * math.cos(phi)
    y0, y1 = c, c + Hb
    pts: list[np.ndarray] = []
    corners: list[int] = []

    def arc(cx, cy, a0, a1, n=48):
        th = np.linspace(math.radians(a0), math.radians(a1), n, endpoint=False)
        pts.extend(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))

    def line(a, b, n):
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        pts.extend(np.asarray(a) + t * (np.asarray(b) - np.asarray(a)))

    sp = L / 400
    sharp = r <= 1e-6 * L                                        # r = 0: square front corners
    if sharp:
        r = 0.0
        corners.append(0)
    line((r, y0), (L, y0), max(2, int((L - r) / sp)))           # underbody
    corners.append(len(pts))
    line((L, y0), (L, y1 - drop), max(2, int((Hb - drop) / sp)))  # base
    if ls > 1e-6 * L:
        corners.append(len(pts))
        line((L, y1 - drop), (L - run, y1), max(2, int(ls / sp)))    # slant
    corners.append(len(pts))
    line((L - run, y1), (r, y1), max(2, int((L - run - r) / sp)))   # roof
    if sharp:
        corners.append(len(pts))
        line((0.0, y1), (0.0, y0), max(2, int(Hb / sp)))            # vertical front face
    else:
        arc(r, y1 - r, 90, 180)                                  # upper front radius
        if Hb - 2 * r > 1e-9:
            line((0.0, y1 - r), (0.0, y0 + r), max(2, int((Hb - 2 * r) / sp)))
        arc(r, y0 + r, 180, 270)                                 # lower front radius
    dense = np.array(pts)
    body = P.orient_ccw(P.resample(dense, _VEH_POINTS, curvature=0.3, corners=corners))
    # re-anchor exactly (resampling keeps corners, so this is a no-op to round-off)
    body = _fit_box(body, 0.0, L, c, c + Hb)
    return _vehicle_design("Ahmed body", body, L, c, c + Hb, template="ahmed", params=p, notes=notes)


# ---------------------------------------------------------------------------
# wings and basic shapes
# ---------------------------------------------------------------------------

def _front_wing(p: dict) -> Design:
    """Inverted cambered wing in ground effect (race-car front wing), optional flap.

    The section is the NACA profile flipped upside down (camber toward the road),
    pitched nose-down by ``incidence`` about its quarter chord (more downforce),
    then placed so its lowest point is ``ride_height`` above the road. Lift is
    therefore negative (downforce). Coefficients are referred to the main-element
    chord, as is usual for wings: ``length_ref = force_ref = chord``.
    """
    ch = p["chord"]
    main = naca(p["naca"], 200, ch, closed_te=True)
    main = P.transform(main, mirror="y")                             # invert, keeps CCW
    main = P.transform(main, rotate_deg=p["incidence"], about=(0.25 * ch, 0.0))   # +: nose down
    polys = [main]
    names = ["Main element"]
    if p.get("flap"):
        fc = p["flap_chord"] * ch
        flap = naca(p["naca"], 160, fc, closed_te=True)
        flap = P.transform(flap, mirror="y")
        flap = P.transform(flap, rotate_deg=p["flap_angle"], about=(0.0, 0.0))
        # slot: flap LE sits a gap below / overlap behind the main TE (the TE is index 0)
        te = main[0]
        gap, overlap = 0.02 * ch, 0.03 * ch
        flap = flap + (te + np.array([-overlap, gap]))
        polys.append(flap)
        names.append("Flap")
    # quarter chord is the rotation centre, so it only moves with the final shift
    shift = np.array([-min(float(q[:, 0].min()) for q in polys),
                      p["ride_height"] - min(float(q[:, 1].min()) for q in polys)])
    polys = [q + shift for q in polys]
    qc = np.array([0.25 * ch, 0.0]) + shift
    bodies = [Body(name=nm, points=q, role="element") for nm, q in zip(names, polys)]
    return Design(name="Front wing (ground effect)", kind="vehicle", bodies=bodies, length_ref=ch,
                  force_ref=ch, moment_ref=(float(qc[0]), float(qc[1])),
                  source={"type": "template", "template": "front_wing", "params": dict(p), "ground": True})


def _naca_t(p: dict) -> Design:
    code = str(p["naca"]).strip()
    pts = naca(code, int(p["points"]), 1.0, closed_te=bool(p["closed_te"]))
    return design_from_airfoil(pts, f"NACA {code}", chord=p["chord"],
                               source={"type": "template", "template": "naca", "params": dict(p),
                                       "naca": code})


def _generic(name: str, pts: np.ndarray, p: dict, template: str, *, lref: float, fref: float,
             mref=(0.0, 0.0)) -> Design:
    return Design(name=name, kind="generic", bodies=[Body(name=name, points=P.orient_ccw(pts))],
                  length_ref=float(lref), force_ref=float(fref), moment_ref=mref,
                  source={"type": "template", "template": template, "params": dict(p)})


def _circle(n: int, rx: float, ry: float) -> np.ndarray:
    th = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([rx * np.cos(th), ry * np.sin(th)])


def _rounded_rect(w: float, h: float, r: float, n: int) -> np.ndarray:
    """Rectangle centred on the origin with corner radius r (r = 0: sharp corners kept exact)."""
    r = max(0.0, min(r, 0.5 * min(w, h)))
    if r <= 1e-12 * max(w, h):
        sq = np.array([[w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2], [-w / 2, -h / 2]])
        return P.resample(sq, n, corners=[0, 1, 2, 3])
    pen = _Pen()
    for cx, cy, a0 in ((w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                       (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)):
        for k in range(3):
            a = math.radians(a0 + 45 * k)
            seg = "line" if k == 2 else "curve"
            pen.add(cx + r * math.cos(a), cy + r * math.sin(a), seg=seg, tangent=a0 + 45 * k + 90)
    pts, _ = pen.build(spacing=(2 * (w + h)) / (40 * n))
    return P.resample(pts, n, curvature=0.0, keep_corners=False)


def _cylinder(p):
    D = p["diameter"]
    return _generic("Cylinder", _circle(int(p["points"]), D / 2, D / 2), p, "cylinder", lref=D, fref=D)


def _ellipse(p):
    a, b = p["length"], p["height"]
    return _generic("Ellipse", _circle(int(p["points"]), a / 2, b / 2), p, "ellipse", lref=a, fref=b)


def _square(p):
    s = p["side"]
    return _generic("Square", _rounded_rect(s, s, p["corner_radius"], int(p["points"])), p, "square",
                    lref=s, fref=s)


def _rrect(p):
    w, h = p["length"], p["height"]
    return _generic("Rounded rectangle", _rounded_rect(w, h, p["corner_radius"], int(p["points"])), p,
                    "rounded_rectangle", lref=w, fref=h)


def _plate(p):
    """Flat plate. Parallel to the flow by default; 'normal' stands it across the flow.

    Referred to its length (chord) in both orientations, so Cd of the normal plate
    is directly comparable with the classic 2D value (~2).
    """
    Lp = p["length"]
    t = p["thickness_ratio"] * Lp
    n = int(p["points"])
    if p["edges"] == "round":
        pts = _rounded_rect(Lp, t, 0.5 * t, n)
    else:
        pts = _rounded_rect(Lp, t, 0.0, n)
    if p["orientation"] == "normal":
        pts = P.transform(pts, rotate_deg=90.0)
    return _generic("Flat plate", pts, p, "flat_plate", lref=Lp, fref=Lp)


# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------

def _veh_params(L, H, c, rn, ws, rw, wheel_r, rw_label="Rear window angle", extra=(),
                ws_range=(15.0, 75.0)) -> list[ParamSpec]:
    ps = [
        # ranges scale with the template's defaults so the form only offers plausible cars
        ParamSpec("length", "Length", "length", L, round(0.6 * L, 1), round(1.5 * L, 1), 0.05,
                  help="Overall length, nose to tail."),
        ParamSpec("height", "Height", "length", H, round(0.7 * H, 2), round(1.4 * H, 2), 0.01,
                  help="Road to highest point of the roof."),
        ParamSpec("ground_clearance", "Ground clearance", "length", c, 0.02, round(0.25 * H, 2), 0.005,
                  help="Gap between the road and the flat underbody."),
        ParamSpec("nose_radius", "Nose radius", "length", rn, 0.02, 0.6, 0.01,
                  help="Rounding of the front bumper; limited to what fits the front face."),
        ParamSpec("windscreen_angle", "Windscreen angle", "angle", ws, ws_range[0], ws_range[1], 1.0,
                  help="Rake of the windscreen from the horizontal."),
    ]
    if rw is not None:
        ps.append(ParamSpec("rear_angle", rw_label, "angle", rw, 10.0, 88.0, 1.0,
                            help="Slope of the rear glass from the horizontal."))
    ps += list(extra)
    ps += [ParamSpec("wheels", "Wheels", "bool", False,
                     help="Off by default: in 2D a wheel touching the road seals the underbody."),
           ParamSpec("wheel_radius", "Wheel radius", "length", wheel_r, 0.2, 0.6, 0.005),
           ParamSpec("wheel_gap", "Wheel gap to road", "length", 0.0, 0.0, 0.1, 0.002,
                     help="Lift the wheels to keep a thin underbody channel open.")]
    return ps


def _pts(n: int, lo: int = 32, hi: int = 2000) -> ParamSpec:
    return ParamSpec("points", "Points", "count", n, lo, hi, 1)


_REGISTRY: dict[str, tuple[TemplateInfo, Callable[[dict], Design]]] = {}


def _register(info: TemplateInfo, fn: Callable[[dict], Design]) -> None:
    _REGISTRY[info.name] = (info, fn)


_register(TemplateInfo("sedan", "vehicle", "Three-box saloon with a notchback boot and a sharp boot-lid edge.",
                       _veh_params(4.7, 1.45, 0.15, 0.22, 27.0, 25.0, 0.32,
                                   extra=[ParamSpec("deck_length", "Boot length", "length", 0.95, 0.3, 1.6,
                                                    0.01, help="Length of the boot lid behind the rear window.")]),
                       "Sedan", "Vehicles"), lambda p: _car("sedan", p))
_register(TemplateInfo("hatchback", "vehicle", "Compact two-box car with a steep hatch and roof spoiler lip.",
                       _veh_params(4.1, 1.48, 0.14, 0.2, 30.0, 58.0, 0.31, "Hatch angle"),
                       "Hatchback", "Vehicles"), lambda p: _car("hatchback", p))
_register(TemplateInfo("fastback", "vehicle", "Low sports car with a long fastback slope and a ducktail lip.",
                       _veh_params(4.5, 1.25, 0.11, 0.16, 22.0, 17.0, 0.33, "Fastback slope"),
                       "Fastback / sports car", "Vehicles"), lambda p: _car("fastback", p))
_register(TemplateInfo("suv", "vehicle", "Tall SUV with an almost vertical tailgate.",
                       _veh_params(4.8, 1.75, 0.21, 0.28, 33.0, 75.0, 0.37, "Tailgate angle"),
                       "SUV", "Vehicles"), lambda p: _car("suv", p))
_register(TemplateInfo("pickup", "vehicle", "Pickup truck with an open load bed (a real cavity) and tailgate.",
                       _veh_params(5.6, 1.85, 0.24, 0.25, 33.0, None, 0.4, extra=[
                           ParamSpec("bed_length", "Bed length", "length", 1.75, 0.8, 2.6, 0.01)]),
                       "Pickup", "Vehicles"), _pickup)
_register(TemplateInfo("van", "vehicle", "Panel van with a short bonnet, steep windscreen and boxy rear.",
                       _veh_params(5.3, 2.3, 0.2, 0.3, 55.0, None, 0.35, ws_range=(30.0, 80.0), extra=[
                           ParamSpec("rear_radius", "Rear roof radius", "length", 0.08, 0.0, 0.5, 0.01)]),
                       "Van", "Vehicles"), _van)
_register(TemplateInfo("box_truck", "vehicle", "Cab-over box truck: a cab with a taller cargo box behind it.", [
    ParamSpec("length", "Length", "length", 7.5, 4.0, 12.0, 0.1),
    ParamSpec("height", "Height", "length", 3.4, 2.4, 4.2, 0.05, help="Road to the top of the box."),
    ParamSpec("ground_clearance", "Ground clearance", "length", 0.35, 0.1, 0.8, 0.01),
    ParamSpec("cab_height", "Cab height", "length", 2.7, 1.8, 3.6, 0.05),
    ParamSpec("cab_length", "Cab length", "length", 1.9, 1.2, 3.0, 0.05),
    ParamSpec("nose_radius", "Cab edge radius", "length", 0.15, 0.02, 0.5, 0.01),
    ParamSpec("box_edge_radius", "Box front edge radius", "length", 0.05, 0.0, 0.3, 0.01,
              help="Rounding the box's leading edge strongly reduces drag."),
], "Box truck", "Vehicles"), _box_truck)
_register(TemplateInfo("ahmed", "vehicle", "2D Ahmed-like reference body with a rear slant (bluff-body benchmark).", [
    ParamSpec("length", "Length", "length", 1.044, 0.3, 5.0, 0.001),
    ParamSpec("body_height", "Body height", "length", 0.288, 0.1, 1.5, 0.001),
    ParamSpec("ground_clearance", "Ground clearance", "length", 0.05, 0.005, 0.3, 0.001),
    ParamSpec("nose_radius", "Front radius", "length", 0.1, 0.0, 0.5, 0.001),
    ParamSpec("slant_angle", "Slant angle", "angle", 25.0, 0.0, 50.0, 0.5,
              help="Classic cases: 25 deg (attached) and 35 deg (separated)."),
    ParamSpec("slant_length", "Slant length", "length", 0.222, 0.0, 0.6, 0.001, help="Measured along the slant."),
], "Ahmed body (2D)", "Vehicles"), _ahmed)
_register(TemplateInfo("front_wing", "vehicle", "Inverted wing in ground effect (race-car front wing), optional flap.", [
    ParamSpec("naca", "Section (NACA)", "text", "4412"),
    ParamSpec("chord", "Chord", "length", 0.3, 0.05, 2.0, 0.005),
    ParamSpec("ride_height", "Ride height", "length", 0.08, 0.005, 1.0, 0.001,
              help="Lowest point of the wing above the road."),
    ParamSpec("incidence", "Incidence (nose down)", "angle", 4.0, -10.0, 20.0, 0.5),
    ParamSpec("flap", "Flap", "bool", False),
    ParamSpec("flap_chord", "Flap chord / main chord", "ratio", 0.4, 0.15, 0.8, 0.01),
    ParamSpec("flap_angle", "Flap angle", "angle", 20.0, 0.0, 45.0, 0.5),
], "Front wing in ground effect", "Wings"), _front_wing)
_register(TemplateInfo("naca", "airfoil", "NACA 4- or 5-digit airfoil (e.g. 0012, 2412, 23012).", [
    ParamSpec("naca", "NACA code", "text", "2412"),
    ParamSpec("chord", "Chord", "length", 1.0, 0.01, 20.0, 0.01),
    ParamSpec("closed_te", "Sharp trailing edge", "bool", True,
              help="Off gives the original blunt trailing edge (0.25 % thick for 12 %)."),
    _pts(240, 40, 1000),
], "NACA airfoil", "Wings"), _naca_t)
_register(TemplateInfo("cylinder", "generic", "Circular cylinder (the classic vortex-street benchmark).", [
    ParamSpec("diameter", "Diameter", "length", 1.0, 1e-3, 100.0, 0.01), _pts(256)],
    "Cylinder", "Basic shapes"), _cylinder)
_register(TemplateInfo("square", "generic", "Square cylinder with optional corner radius.", [
    ParamSpec("side", "Side", "length", 1.0, 1e-3, 100.0, 0.01),
    ParamSpec("corner_radius", "Corner radius", "length", 0.0, 0.0, 50.0, 0.005), _pts(256)],
    "Square", "Basic shapes"), _square)
_register(TemplateInfo("ellipse", "generic", "Ellipse aligned with the flow.", [
    ParamSpec("length", "Length", "length", 1.0, 1e-3, 100.0, 0.01),
    ParamSpec("height", "Height", "length", 0.5, 1e-3, 100.0, 0.01), _pts(256)],
    "Ellipse", "Basic shapes"), _ellipse)
_register(TemplateInfo("flat_plate", "generic", "Thin flat plate, parallel or normal to the flow.", [
    ParamSpec("length", "Length", "length", 1.0, 1e-3, 100.0, 0.01),
    ParamSpec("thickness_ratio", "Thickness / length", "ratio", 0.02, 0.002, 0.2, 0.001),
    ParamSpec("edges", "Edges", "choice", "square", choices=("square", "round")),
    ParamSpec("orientation", "Orientation", "choice", "parallel", choices=("parallel", "normal")),
    _pts(256)], "Flat plate", "Basic shapes"), _plate)
_register(TemplateInfo("rounded_rectangle", "generic", "Rectangle with rounded corners.", [
    ParamSpec("length", "Length", "length", 2.0, 1e-3, 100.0, 0.01),
    ParamSpec("height", "Height", "length", 1.0, 1e-3, 100.0, 0.01),
    ParamSpec("corner_radius", "Corner radius", "length", 0.2, 0.0, 50.0, 0.005), _pts(256)],
    "Rounded rectangle", "Basic shapes"), _rrect)


def list_templates() -> list[TemplateInfo]:
    """All templates, in gallery order (vehicles, wings, basic shapes)."""
    return [info for info, _ in _REGISTRY.values()]


def get_template(name: str) -> TemplateInfo:
    """The :class:`TemplateInfo` of one template; KeyError lists the valid names."""
    try:
        return _REGISTRY[name][0]
    except KeyError:
        raise KeyError(f"unknown template {name!r}; available: {', '.join(_REGISTRY)}") from None


def _coerce(spec: ParamSpec, v: Any) -> Any:
    if spec.unit_kind == "bool":
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    if spec.unit_kind == "text":
        return str(v)
    if spec.unit_kind == "choice":
        if v not in (spec.choices or ()):
            raise ValueError(f"{spec.label}: choose one of {', '.join(spec.choices or ())}")
        return v
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{spec.label} must be a finite number")
    if spec.min is not None and x < spec.min - 1e-12:
        raise ValueError(f"{spec.label} must be at least {spec.min:g}")
    if spec.max is not None and x > spec.max + 1e-12:
        raise ValueError(f"{spec.label} must be at most {spec.max:g}")
    return round(x) if spec.unit_kind == "count" else x


def make(name: str, **params: Any) -> Design:
    """Build a template with defaults overridden by ``params`` (SI units, degrees).

    Unknown parameter names and out-of-range values raise ValueError with a message
    meant for the user. The design's ``source`` records the template and the full
    parameter set, so a project can rebuild or tweak it later.
    """
    info, fn = _REGISTRY.get(name, (None, None))
    if info is None:
        raise KeyError(f"unknown template {name!r}; available: {', '.join(_REGISTRY)}")
    specs = {s.name: s for s in info.params}
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"unknown parameter(s) for {name}: {', '.join(sorted(unknown))}")
    p = info.defaults()
    for k, v in params.items():
        p[k] = _coerce(specs[k], v)
    d = fn(p)
    d.source.setdefault("template", name)
    d.source["params"] = dict(p)
    return d
