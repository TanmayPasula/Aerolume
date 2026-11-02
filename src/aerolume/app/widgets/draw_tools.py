"""Live editing in the 2D flow view (DESIGN.md 10.4): draw obstacles, move emitters, fans and heaters.

AeroJAX lets you paint a wall into a running flow and watch the wake form; this
module is that, on top of :class:`~aerolume.app.widgets.field_view.FieldView`.
Three layers, from data to pixels:

* **Data, no Qt.** :class:`ObstacleSketch` is the drawing as an ordered list of
  operations -- brush strokes and polygons, each adding solid or erasing it --
  with undo / redo. It rasterises itself onto *any* grid as an anti-aliased
  solid fraction (:meth:`ObstacleSketch.rasterize`, the form a solver mask
  takes) and traces its outlines (:meth:`ObstacleSketch.polygons`, the form
  ``set_bodies`` takes), so the GUI, the live session and a test agree on
  exactly what was drawn. Erasing removes drawn solid only: the design's own
  bodies are geometry, edited on the Geometry page. :class:`Placement` is one
  emitter / fan / heater / port as drawn (wind frame), built from the
  simulation by :func:`placements_of` and moved back into it by
  :func:`apply_move` (body-frame placements turn with the geometry, exactly as
  the planner turns them).
* **Tools.** :class:`SketchTool` (brush, polygon, eraser) and
  :class:`MoveTool` (drag a glyph; the empty view still pans) implement the
  small tool protocol FieldView calls (``mouse_press`` / ``mouse_move`` /
  ``mouse_release`` / ``key_press`` / ``paint``).
* **Chrome.** :class:`DrawToolStrip` is the strip of tool buttons, the brush
  size and undo / redo / clear, with its own line icons (inline SVG, tinted
  from the theme at paint time like the bundled set).

Why a list of operations and not a painted bitmap: undo is then exact, the
drawing is resolution-independent (the explore grid and a later full run have
different cells), and it is tiny to post to the solver thread.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QIconEngine, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QLabel, QSlider, QToolButton, QWidget

from .common import ToolStrip, label
from .plots import active_icon_token, qcolor, tokens

__all__ = [
    "DRAW_TOOLS",
    "DrawToolStrip",
    "MoveTool",
    "ObstacleSketch",
    "Placement",
    "SketchOp",
    "SketchTool",
    "apply_move",
    "paint_placements",
    "placements_of",
    "tool_icon",
]

# ---------------------------------------------------------------------------
# the drawing (no Qt)
# ---------------------------------------------------------------------------

#: Tools of the strip: (key, label, tooltip).
DRAW_TOOLS = (
    ("move", "Move", "Drag an emitter, fan or heater; drag the empty view to pan"),
    ("brush", "Brush", "Paint solid obstacles into the flow (they act at once)"),
    ("polygon", "Polygon", "Click the corners of a solid obstacle; double-click or Enter closes it, Esc cancels"),
    ("erase", "Eraser", "Erase drawn obstacles (the design's own bodies are edited in Geometry)"),
)


@dataclass
class SketchOp:
    """One drawing operation: a brush stroke (polyline of ``radius``) or a filled polygon (metres, wind frame)."""
    kind: str                         # "brush" | "polygon"
    points: np.ndarray                # (N, 2)
    radius: float = 0.0               # brush half-width [m]
    erase: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("brush", "polygon"):
            raise ValueError(f"unknown sketch operation {self.kind!r}")
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        self.radius = float(self.radius)

    def bounds(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) the operation can touch, brush radius included."""
        p, r = self.points, self.radius if self.kind == "brush" else 0.0
        return (float(p[:, 0].min()) - r, float(p[:, 0].max()) + r, float(p[:, 1].min()) - r,
                float(p[:, 1].max()) + r)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "points": self.points.tolist(), "radius": self.radius, "erase": self.erase}

    @classmethod
    def from_dict(cls, d: dict) -> SketchOp:
        return cls(str(d["kind"]), np.asarray(d["points"], float), float(d.get("radius", 0.0)),
                   bool(d.get("erase", False)))


def _segment_distance(px: np.ndarray, py: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Distance from points (px, py) to a polyline ``pts`` (N >= 1 vertices), vectorised in chunks."""
    d = np.full(px.shape, np.inf)
    if len(pts) == 1:
        return np.hypot(px - pts[0, 0], py - pts[0, 1])
    a, b = pts[:-1], pts[1:]
    shape = (-1,) + (1,) * np.ndim(px)               # segments along a new leading axis
    for i0 in range(0, len(a), 64):                # bounded memory for long strokes
        ax, ay = a[i0:i0 + 64, 0].reshape(shape), a[i0:i0 + 64, 1].reshape(shape)
        bx, by = b[i0:i0 + 64, 0].reshape(shape), b[i0:i0 + 64, 1].reshape(shape)
        dx, dy = bx - ax, by - ay
        ll = np.maximum(dx * dx + dy * dy, 1e-300)
        t = np.clip(((px[None] - ax) * dx + (py[None] - ay) * dy) / ll, 0.0, 1.0)
        dd = np.hypot(px[None] - (ax + t * dx), py[None] - (ay + t * dy))
        d = np.minimum(d, dd.min(axis=0))
    return d


def _polygon_coverage(px: np.ndarray, py: np.ndarray, poly: np.ndarray, h: float) -> np.ndarray:
    """Anti-aliased coverage 0..1 of cell centres by a polygon: a one-cell ramp across its edge."""
    from ...geometry import polygon as P
    inside = np.asarray(P.point_in_polygon(poly, np.column_stack([px.ravel(), py.ravel()])),
                        dtype=bool).reshape(px.shape)
    closed = np.vstack([poly, poly[:1]])
    d = _segment_distance(px, py, closed)
    sd = np.where(inside, d, -d)                  # signed distance, > 0 inside
    return np.clip(0.5 + sd / h, 0.0, 1.0)


class ObstacleSketch:
    """The obstacles a user drew: an ordered list of :class:`SketchOp` with undo / redo.

    ``revision`` grows with every change, so viewers and the solver hand-over
    can tell a new drawing from a re-post of the same one. ``pending`` is the
    stroke being drawn right now (shown and, throttled, posted while the mouse
    is down; committed on release).
    """

    def __init__(self) -> None:
        self.ops: list[SketchOp] = []
        self._redo: list[SketchOp] = []
        self.pending: SketchOp | None = None
        self.revision = 0

    # -- editing ----------------------------------------------------------
    def _changed(self) -> None:
        self.revision += 1

    def add(self, op: SketchOp) -> None:
        """Commit an operation (clears the redo list, as every editor does)."""
        self.ops.append(op)
        self._redo.clear()
        self.pending = None
        self._changed()

    def add_stroke(self, points, radius: float, *, erase: bool = False) -> SketchOp:
        op = SketchOp("brush", points, radius, erase)
        self.add(op)
        return op

    def add_polygon(self, points, *, erase: bool = False) -> SketchOp | None:
        """A filled polygon; fewer than three distinct corners or a zero area is refused (None)."""
        from ...geometry import polygon as P
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3 or abs(P.signed_area(pts)) <= 1e-12:
            return None
        op = SketchOp("polygon", P.orient_ccw(pts), 0.0, erase)
        self.add(op)
        return op

    def set_pending(self, op: SketchOp | None) -> None:
        self.pending = op
        self._changed()

    def undo(self) -> bool:
        if not self.ops:
            return False
        self._redo.append(self.ops.pop())
        self._changed()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self.ops.append(self._redo.pop())
        self._changed()
        return True

    def clear(self) -> None:
        """Remove everything (one undo step brings it all back)."""
        if not self.ops and self.pending is None:
            return
        self._redo = list(reversed(self.ops)) + self._redo
        self.ops = []
        self.pending = None
        self._changed()

    def can_undo(self) -> bool:
        return bool(self.ops)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def is_empty(self) -> bool:
        """True when nothing solid remains drawn (erasing alone draws nothing)."""
        return not any(not op.erase for op in self.all_ops())

    def all_ops(self) -> list[SketchOp]:
        return self.ops + ([self.pending] if self.pending is not None else [])

    # -- output -----------------------------------------------------------
    def rasterize(self, extent: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
        """The drawing as a solid fraction on a cell grid: (ny, nx) float32, row 0 = y min.

        Operations apply in order -- drawing is a union, erasing a
        subtraction -- each with a one-cell anti-aliasing ramp across its
        edge (the solver's penalisation mask is fractional across the wall
        too). Only the cells inside an operation's bounds are evaluated, so a
        stroke costs its own footprint, not the domain.
        """
        x0, x1, y0, y1 = (float(e) for e in extent)
        ny, nx = int(shape[0]), int(shape[1])
        m = np.zeros((ny, nx), np.float32)
        hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
        h = max(min(hx, hy), 1e-12)
        for op in self.all_ops():
            bx0, bx1, by0, by1 = op.bounds()
            i0 = max(math.floor((bx0 - x0) / hx) - 1, 0)
            i1 = min(math.ceil((bx1 - x0) / hx) + 1, nx)
            j0 = max(math.floor((by0 - y0) / hy) - 1, 0)
            j1 = min(math.ceil((by1 - y0) / hy) + 1, ny)
            if i1 <= i0 or j1 <= j0:
                continue
            xs = x0 + (np.arange(i0, i1) + 0.5) * hx
            ys = y0 + (np.arange(j0, j1) + 0.5) * hy
            px, py = np.meshgrid(xs, ys)
            if op.kind == "brush":
                d = _segment_distance(px, py, op.points)
                f = np.clip(0.5 + (op.radius - d) / h, 0.0, 1.0)
            else:
                f = _polygon_coverage(px, py, op.points, h)
            block = m[j0:j1, i0:i1]
            m[j0:j1, i0:i1] = np.minimum(block, 1.0 - f) if op.erase else np.maximum(block, f)
        return m

    def polygons(self, extent: Sequence[float], shape: tuple[int, int], *,
                 min_cells: float = 2.0) -> list[np.ndarray]:
        """Outlines of the drawn solid (CCW, metres), traced at the 1/2 level of :meth:`rasterize`.

        Holes are not returned (a polygon body is all solid): what was erased
        *inside* a drawn region is filled again in this form. Regions smaller
        than ``min_cells`` cells are dropped (a click without a drag still gives
        a dot of one brush radius, well above that).
        """
        import contourpy

        from ...geometry import polygon as P
        m = self.rasterize(extent, shape)
        if not np.any(m > 0.5):
            return []
        x0, x1, y0, y1 = (float(e) for e in extent)
        ny, nx = m.shape
        hx, hy = (x1 - x0) / nx, (y1 - y0) / ny
        z = np.pad(m - 0.5, 1, constant_values=-0.5)
        cg = contourpy.contour_generator(z=z, fill_type=contourpy.FillType.OuterOffset)
        pts_list, offs_list = cg.filled(0.0, np.inf)
        out = []
        for pts, offs in zip(pts_list, offs_list):
            ring = pts[offs[0]:offs[1]]
            if len(ring) < 4:
                continue
            q = np.column_stack([x0 + (ring[:, 0] - 0.5) * hx, y0 + (ring[:, 1] - 0.5) * hy])
            q = P.remove_duplicates(q)
            if len(q) < 3 or abs(P.signed_area(q)) < min_cells * hx * hy:
                continue
            q = P.simplify_rdp(q, 0.25 * min(hx, hy)) if len(q) > 8 else q
            if len(q) >= 3:
                out.append(P.orient_ccw(q))
        return out

    def payload(self, extent: Sequence[float], shape: tuple[int, int]) -> dict:
        """What the live session receives (``post(obstacles=...)``): the solid fraction on its grid,
        the traced outlines, the operations themselves and the revision (see the module docstring)."""
        return {"revision": self.revision, "extent": tuple(float(e) for e in extent),
                "mask": self.rasterize(extent, shape), "polygons": self.polygons(extent, shape),
                "ops": [op.to_dict() for op in self.ops]}

    def to_dict(self) -> dict:
        return {"ops": [op.to_dict() for op in self.ops]}

    @classmethod
    def from_dict(cls, d: dict) -> ObstacleSketch:
        s = cls()
        s.ops = [SketchOp.from_dict(o) for o in (d or {}).get("ops", [])]
        s.revision = len(s.ops)
        return s


# ---------------------------------------------------------------------------
# placements: emitters, fans, heaters and ports as drawn
# ---------------------------------------------------------------------------

#: Simulation list attribute -> glyph kind.
PLACEMENT_LISTS = (("emitters", "emitter"), ("fans", "fan"), ("heat_sources", "heater"), ("ports", "port"))


@dataclass
class Placement:
    """One emitter, fan, heater or port of a 2D simulation, in the wind frame (metres), for drawing and dragging.

    ``key`` is ``"<list>:<index>"`` (``"fans:0"``) so a move finds its model
    object again. ``x, y`` is the handle (its position); ``end`` the other end
    of a line / rake emitter; ``direction`` a unit vector (fans, ports);
    ``width`` the fan diameter / port size; ``size`` a heater's box or radius.
    """
    key: str
    kind: str                         # "emitter" | "fan" | "heater" | "port" | "vent"
    name: str
    x: float
    y: float
    end: tuple[float, float] | None = None
    direction: tuple[float, float] | None = None
    width: float = 0.0
    size: tuple[float, ...] = ()
    shape: str = ""                   # emitter kind ("point", "rake" ...) or heater kind ("rect", "disc")
    color: str = ""
    count: int = 0
    enabled: bool = True
    frame: str = "wind"
    extra: dict = field(default_factory=dict)


def _rotation(design, alpha_deg: float):
    """(cos, sin, about) of the wind-frame rotation of body-frame geometry: -alpha about moment_ref."""
    a = -math.radians(float(alpha_deg or 0.0))
    about = tuple(getattr(design, "moment_ref", (0.0, 0.0)) or (0.0, 0.0))[:2] if design is not None else (0.0, 0.0)
    return math.cos(a), math.sin(a), (float(about[0]), float(about[1]))


def _to_wind(xy, frame: str, rot) -> tuple[float, float]:
    x, y = float(xy[0]), float(xy[1]) if len(xy) > 1 else 0.0
    if frame != "body":
        return x, y
    c, s, (cx, cy) = rot
    dx, dy = x - cx, y - cy
    return cx + c * dx - s * dy, cy + s * dx + c * dy


def _from_wind(xy, frame: str, rot) -> tuple[float, float]:
    x, y = float(xy[0]), float(xy[1])
    if frame != "body":
        return x, y
    c, s, (cx, cy) = rot
    dx, dy = x - cx, y - cy
    return cx + c * dx + s * dy, cy - s * dx + c * dy


def _turn(vec, frame: str, rot) -> tuple[float, float]:
    vx, vy = float(vec[0]), float(vec[1]) if len(vec) > 1 else 0.0
    if frame != "body":
        return vx, vy
    c, s, _ = rot
    return c * vx - s * vy, s * vx + c * vy


def placements_of(sim, design=None, alpha_deg: float | None = None) -> list[Placement]:
    """Every enabled-or-not emitter, fan, heater and port of a 2D simulation as a :class:`Placement`.

    Body-frame items are turned into the wind frame by -alpha about the
    design's moment reference (DESIGN.md 3), so a heater fixed to a pitched
    body is drawn where the solver puts it.
    """
    if sim is None:
        return []
    alpha = float(getattr(getattr(sim, "conditions", None), "alpha_deg", 0.0) or 0.0) if alpha_deg is None \
        else float(alpha_deg)
    rot = _rotation(design, alpha)
    out: list[Placement] = []
    for attr, kind in PLACEMENT_LISTS:
        for i, item in enumerate(getattr(sim, attr, None) or []):
            frame = str(getattr(item, "frame", "wind"))
            x, y = _to_wind(getattr(item, "position", (0.0, 0.0)), frame, rot)
            name = str(getattr(item, "name", kind.capitalize()))
            pl = Placement(key=f"{attr}:{i}", kind=kind, name=name, x=x, y=y, frame=frame,
                           enabled=bool(getattr(item, "enabled", True)))
            if kind == "emitter":
                pl.shape = str(getattr(item, "kind", "point"))
                end = getattr(item, "end", None)
                pl.end = _to_wind(end, frame, rot) if end is not None and pl.shape in ("line", "rake") else None
                pl.color = str(getattr(item, "color", "") or "")
                pl.count = int(getattr(item, "count", 0) or 0)
                pl.size = tuple(getattr(item, "size", ()) or ())
                pl.extra["species"] = str(getattr(item, "species", "dye"))
            elif kind == "fan":
                pl.direction = _turn(getattr(item, "direction", (1.0, 0.0)), frame, rot)
                pl.width = float(getattr(item, "diameter", 0.0) or 0.0)
                pl.extra.update(mode=str(getattr(item, "mode", "thrust")), value=float(getattr(item, "value", 0.0)))
            elif kind == "heater":
                pl.shape = str(getattr(item, "kind", "rect"))
                pl.size = tuple(float(v) for v in (getattr(item, "size", ()) or ()))
                pl.extra["power"] = float(getattr(item, "power", 0.0) or 0.0)
            else:
                pl.kind = "vent" if str(getattr(item, "kind", "jet")) == "vent" else "port"
                pl.direction = _turn(getattr(item, "direction", (1.0, 0.0)), frame, rot)
                pl.width = float(getattr(item, "size", 0.0) or 0.0)
                pl.extra["speed"] = float(getattr(item, "speed", 0.0) or 0.0)
            norm = math.hypot(*(pl.direction or (0.0, 0.0)))
            if pl.direction is not None and norm > 0:
                pl.direction = (pl.direction[0] / norm, pl.direction[1] / norm)
            out.append(pl)
    return out


def apply_move(sim, key: str, x: float, y: float, design=None, alpha_deg: float | None = None) -> bool:
    """Move the item ``key`` of ``sim`` so its handle sits at wind-frame (x, y); returns whether it moved.

    A line / rake emitter keeps its length and direction (both ends move).
    The z of a 3-component position is kept, so a 2D move of a 3D item never
    flattens it.
    """
    try:
        attr, idx = key.split(":")
        item = getattr(sim, attr)[int(idx)]
    except (ValueError, AttributeError, IndexError, TypeError):
        return False
    alpha = float(getattr(getattr(sim, "conditions", None), "alpha_deg", 0.0) or 0.0) if alpha_deg is None \
        else float(alpha_deg)
    rot = _rotation(design, alpha)
    frame = str(getattr(item, "frame", "wind"))
    old = tuple(getattr(item, "position", (0.0, 0.0)))
    nx_, ny_ = _from_wind((x, y), frame, rot)
    dx, dy = nx_ - float(old[0]), ny_ - float(old[1] if len(old) > 1 else 0.0)
    if abs(dx) < 1e-15 and abs(dy) < 1e-15:
        return False
    item.position = (nx_, ny_, *old[2:])
    end = getattr(item, "end", None)
    if end is not None:
        e = tuple(end)
        item.end = (float(e[0]) + dx, float(e[1]) + dy, *e[2:])
    return True


# ---------------------------------------------------------------------------
# painting
# ---------------------------------------------------------------------------

#: Glyph size in screen pixels (handles stay the same size at every zoom).
GLYPH_PX = 7.0
#: Hit radius of a glyph handle [px].
HIT_PX = 10.0


def _glyph_pen(color: QColor, width: float = 1.5) -> QPen:
    pen = QPen(color, width)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _halo(p: QPainter, path: QPainterPath, dark_image: bool) -> None:
    """A 3 px contrasting halo under a glyph so it reads on any colour of any map."""
    halo = QColor(16, 18, 22, 200) if not dark_image else QColor(250, 250, 250, 210)
    p.setPen(_glyph_pen(halo, 3.6))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


def paint_placements(p: QPainter, xf, placements: Sequence[Placement], *, dark_image: bool = True,
                     active: str | None = None, hover: str | None = None) -> None:
    """Draw emitter, fan, heater and port glyphs over a field (``xf``: world -> screen transform).

    Glyphs are line drawings in the accent colour over a contrasting halo, at a
    fixed pixel size (they are handles, not geometry), except where the item
    has a physical size worth seeing: a rake's nozzles and line, a fan's
    actuator strip (its true width), a heater's box. A disabled item is drawn
    dashed and muted; the hovered or dragged one gets a filled handle.
    """
    tk = tokens()
    accent = qcolor(tk["accent"])
    muted = qcolor(tk["muted"])
    warm = qcolor(tk["warn"])
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for pl in placements:
        h = xf.map(QPointF(pl.x, pl.y))
        col = accent if pl.enabled else muted
        if pl.kind == "heater":
            col = warm if pl.enabled else muted
        path = QPainterPath()
        if pl.kind == "emitter":
            if pl.end is not None:
                e = xf.map(QPointF(*pl.end))
                path.moveTo(h)
                path.lineTo(e)
                if pl.shape == "rake" and pl.count > 1:
                    for k in range(pl.count):
                        q = h + (e - h) * (k / (pl.count - 1))
                        path.addEllipse(q, 2.2, 2.2)
            path.addEllipse(h, GLYPH_PX * 0.55, GLYPH_PX * 0.55)
            if pl.color and pl.extra.get("species") != "particles":
                # the smoke's own colour in the handle: this is how the channel legend is read
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(pl.color))
                p.drawEllipse(h, GLYPH_PX * 0.42, GLYPH_PX * 0.42)
        elif pl.kind == "fan":
            d = pl.direction or (1.0, 0.0)
            half = 0.5 * pl.width
            a = xf.map(QPointF(pl.x - d[1] * half, pl.y + d[0] * half))
            b = xf.map(QPointF(pl.x + d[1] * half, pl.y - d[0] * half))
            path.moveTo(a)
            path.lineTo(b)                                       # the actuator strip, true width
            tip = xf.map(QPointF(pl.x, pl.y)) + QPointF(d[0], -d[1]) * (2.6 * GLYPH_PX)
            path.moveTo(h)
            path.lineTo(tip)
            left = tip + QPointF(-d[0] * 5 - d[1] * 3.5, d[1] * 5 - d[0] * 3.5)
            right = tip + QPointF(-d[0] * 5 + d[1] * 3.5, d[1] * 5 + d[0] * 3.5)
            path.moveTo(left)
            path.lineTo(tip)
            path.lineTo(right)
        elif pl.kind == "heater":
            if pl.shape in ("disc", "sphere"):
                r = pl.size[0] if pl.size else 0.0
                rr = max(abs(xf.map(QPointF(pl.x + r, pl.y)).x() - h.x()), GLYPH_PX)
                path.addEllipse(h, rr, rr)
            else:
                w = pl.size[0] if len(pl.size) > 0 else 0.0
                ht = pl.size[1] if len(pl.size) > 1 else w
                r = xf.mapRect(QRectF(pl.x - w / 2, pl.y - ht / 2, w, ht))
                if r.width() < 2 * GLYPH_PX or r.height() < 2 * GLYPH_PX:
                    r = r.adjusted(-max(0.0, GLYPH_PX - r.width() / 2), -max(0.0, GLYPH_PX - r.height() / 2),
                                   max(0.0, GLYPH_PX - r.width() / 2), max(0.0, GLYPH_PX - r.height() / 2))
                path.addRect(r)
            # the region a heater fills, hatched in the warm colour (the CAD convention for a heat source)
            _halo(p, path, dark_image)
            p.save()
            # intersect, not replace: the view clips its overlays to the plot area, and a replaced clip let a
            # heater just outside the window hatch the colour-bar strip beside it
            p.setClipPath(path, Qt.ClipOperation.IntersectClip)
            hatch = QColor(col)
            hatch.setAlpha(150)
            p.setPen(_glyph_pen(hatch, 1.0))
            br = path.boundingRect()
            step = 5.0
            x = br.left() - br.height()
            while x < br.right():
                p.drawLine(QPointF(x, br.bottom()), QPointF(x + br.height(), br.top()))
                x += step
            p.restore()
        else:                                                    # jet / vent
            d = pl.direction or (1.0, 0.0)
            half = max(0.5 * pl.width, 0.0)
            a = xf.map(QPointF(pl.x - d[1] * half, pl.y + d[0] * half))
            b = xf.map(QPointF(pl.x + d[1] * half, pl.y - d[0] * half))
            path.moveTo(a)
            path.lineTo(b)
            sgn = 1.0 if pl.kind == "port" else -1.0              # a vent's arrow points into it
            start = h if sgn > 0 else h + QPointF(d[0], -d[1]) * (2.6 * GLYPH_PX)
            tip = h + QPointF(d[0], -d[1]) * (2.6 * GLYPH_PX) if sgn > 0 else h
            path.moveTo(start)
            path.lineTo(tip)
            dd = QPointF(d[0], -d[1]) * sgn
            path.moveTo(tip - dd * 5 + QPointF(-dd.y(), dd.x()) * 3.5)
            path.lineTo(tip)
            path.lineTo(tip - dd * 5 - QPointF(-dd.y(), dd.x()) * 3.5)
        _halo(p, path, dark_image)
        pen = _glyph_pen(col, 1.6)
        if not pl.enabled:
            pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        if pl.key in (active, hover):
            p.setPen(_glyph_pen(col, 1.2))
            p.setBrush(col)
            p.drawEllipse(h, 3.2, 3.2)
    p.restore()


def hit_placement(placements: Sequence[Placement], xf, pos: QPointF, radius: float = HIT_PX) -> str | None:
    """Key of the placement whose handle is nearest to ``pos`` within ``radius`` px, else None."""
    best, best_d = None, radius
    for pl in placements:
        q = xf.map(QPointF(pl.x, pl.y))
        d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
        if d <= best_d:
            best, best_d = pl.key, d
    return best


# ---------------------------------------------------------------------------
# tools (the FieldView tool protocol)
# ---------------------------------------------------------------------------

class _Tool:
    """Base of the FieldView tool protocol: every handler returns True when it used the event."""

    def mouse_press(self, view, ev) -> bool:
        return False

    def mouse_move(self, view, ev) -> bool:
        return False

    def mouse_release(self, view, ev) -> bool:
        return False

    def double_click(self, view, ev) -> bool:
        return False

    def key_press(self, view, ev) -> bool:
        return False

    def leave(self, view) -> None:
        return None

    def paint(self, view, p: QPainter) -> None:
        return None

    def cursor(self, view):
        return None


class SketchTool(_Tool):
    """Brush, polygon and eraser on an :class:`ObstacleSketch` (left button; the middle button still pans).

    ``radius`` is the brush half-width in metres. ``on_change(final)`` is called
    while a stroke is drawn (``final=False``: the page posts a throttled
    preview) and when an operation is committed (``final=True``).
    """

    def __init__(self, sketch: ObstacleSketch, *, mode: str = "brush", radius: float = 0.05,
                 on_change=None) -> None:
        self.sketch = sketch
        self.mode = mode
        self.radius = float(radius)
        self.on_change = on_change
        self._stroke: list[tuple[float, float]] = []
        self._poly: list[tuple[float, float]] = []
        self._cursor_world: tuple[float, float] | None = None

    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.cancel()
        self.mode = mode

    def cancel(self) -> None:
        self._stroke, self._poly = [], []
        if self.sketch.pending is not None:
            self.sketch.set_pending(None)
            self._notify(True)

    def _notify(self, final: bool) -> None:
        if self.on_change is not None:
            self.on_change(final)

    def _pending_stroke(self) -> SketchOp:
        return SketchOp("brush", np.asarray(self._stroke), self.radius, self.mode == "erase")

    def mouse_press(self, view, ev) -> bool:
        if ev.button() != Qt.MouseButton.LeftButton:
            return False
        w = view.screen_to_world(ev.position())
        if self.mode in ("brush", "erase"):
            self._stroke = [w]
            self.sketch.set_pending(self._pending_stroke())
            self._notify(False)
        elif self.mode == "polygon":
            if not self._poly or math.hypot(w[0] - self._poly[-1][0], w[1] - self._poly[-1][1]) > 1e-9:
                self._poly.append(w)
        view.update()
        return True

    def mouse_move(self, view, ev) -> bool:
        w = view.screen_to_world(ev.position())
        self._cursor_world = w
        if self._stroke and self.mode in ("brush", "erase") and ev.buttons() & Qt.MouseButton.LeftButton:
            last = self._stroke[-1]
            # thin the input: a point every third of a radius keeps the capsule smooth and the op small
            if math.hypot(w[0] - last[0], w[1] - last[1]) >= max(self.radius / 3.0, 1e-12):
                self._stroke.append(w)
                self.sketch.set_pending(self._pending_stroke())
                self._notify(False)
        view.update()
        return bool(self._stroke) or self.mode == "polygon"

    def mouse_release(self, view, ev) -> bool:
        if ev.button() != Qt.MouseButton.LeftButton:
            return False
        if self._stroke and self.mode in ("brush", "erase"):
            self.sketch.add_stroke(np.asarray(self._stroke), self.radius, erase=self.mode == "erase")
            self._stroke = []
            self._notify(True)
            view.update()
            return True
        return self.mode == "polygon"

    def double_click(self, view, ev) -> bool:
        if self.mode != "polygon":
            return False
        self.close_polygon()
        view.update()
        return True

    def close_polygon(self) -> bool:
        """Commit the polygon being drawn (Enter / double-click)."""
        pts, self._poly = self._poly, []
        op = self.sketch.add_polygon(pts) if len(pts) >= 3 else None
        if op is not None:
            self._notify(True)
        return op is not None

    def key_press(self, view, ev) -> bool:
        k = ev.key()
        if self.mode == "polygon" and k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.close_polygon()
        elif k == Qt.Key.Key_Escape and (self._poly or self._stroke):
            self.cancel()
        elif self.mode == "polygon" and k == Qt.Key.Key_Backspace and self._poly:
            self._poly.pop()
        else:
            return False
        view.update()
        return True

    def leave(self, view) -> None:
        self._cursor_world = None
        view.update()

    def cursor(self, view):
        return Qt.CursorShape.CrossCursor

    def paint(self, view, p: QPainter) -> None:
        tk = tokens()
        xf = view.world_to_screen()
        accent = qcolor(tk["accent"])
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._poly:
            poly = QPolygonF([xf.map(QPointF(*q)) for q in self._poly])
            if self._cursor_world is not None:
                poly.append(xf.map(QPointF(*self._cursor_world)))
            fill = QColor(accent)
            fill.setAlpha(40)
            p.setBrush(fill)
            p.setPen(_glyph_pen(accent, 1.4))
            p.drawPolygon(poly)
            p.setBrush(accent)
            for q in self._poly:
                p.drawEllipse(xf.map(QPointF(*q)), 2.6, 2.6)
        if self._cursor_world is not None and self.mode in ("brush", "erase"):
            c = xf.map(QPointF(*self._cursor_world))
            r = abs(xf.map(QPointF(self._cursor_world[0] + self.radius, self._cursor_world[1])).x() - c.x())
            pen = _glyph_pen(accent if self.mode == "brush" else qcolor(tk["bad"]), 1.2)
            pen.setStyle(Qt.PenStyle.DashLine if self.mode == "erase" else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, max(r, 2.0), max(r, 2.0))
        p.restore()


class MoveTool(_Tool):
    """Drag a placement glyph by its handle; elsewhere the view pans as usual.

    ``on_move(key, x, y, final)`` is called while dragging (``final=False``)
    and on release (``final=True``) with the new wind-frame position.
    """

    def __init__(self, placements: Sequence[Placement] = (), *, on_move=None, enabled: bool = True) -> None:
        self.placements = list(placements)
        self.on_move = on_move
        self.enabled = enabled
        self.drag: str | None = None
        self.hover: str | None = None
        self._grab = (0.0, 0.0)

    def set_placements(self, placements: Sequence[Placement]) -> None:
        self.placements = list(placements)
        if self.drag is not None and not any(pl.key == self.drag for pl in self.placements):
            self.drag = None

    def _find(self, key: str) -> Placement | None:
        return next((pl for pl in self.placements if pl.key == key), None)

    def mouse_press(self, view, ev) -> bool:
        if not self.enabled or ev.button() != Qt.MouseButton.LeftButton:
            return False
        key = hit_placement(self.placements, view.world_to_screen(), ev.position())
        if key is None:
            return False
        pl = self._find(key)
        wx, wy = view.screen_to_world(ev.position())
        self.drag, self._grab = key, (pl.x - wx, pl.y - wy)
        view.setCursor(Qt.CursorShape.SizeAllCursor)
        return True

    def mouse_move(self, view, ev) -> bool:
        if self.drag is None:
            if self.enabled:
                hov = hit_placement(self.placements, view.world_to_screen(), ev.position())
                if hov != self.hover:
                    self.hover = hov
                    view.setToolTip(self._tip(hov))
                    if hov is not None:
                        view.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        view.unsetCursor()
                    view.update()
            return False
        wx, wy = view.screen_to_world(ev.position())
        pl = self._find(self.drag)
        if pl is not None:
            nx_, ny_ = wx + self._grab[0], wy + self._grab[1]
            if pl.end is not None:
                pl.end = (pl.end[0] + nx_ - pl.x, pl.end[1] + ny_ - pl.y)
            pl.x, pl.y = nx_, ny_
            if self.on_move is not None:
                self.on_move(pl.key, pl.x, pl.y, False)
        view.update()
        return True

    def mouse_release(self, view, ev) -> bool:
        if self.drag is None or ev.button() != Qt.MouseButton.LeftButton:
            return False
        pl = self._find(self.drag)
        self.drag = None
        view.unsetCursor()
        if pl is not None and self.on_move is not None:
            self.on_move(pl.key, pl.x, pl.y, True)
        view.update()
        return True

    def _tip(self, key: str | None) -> str:
        pl = self._find(key) if key else None
        if pl is None:
            return ""
        what = {"emitter": "Emitter", "fan": "Fan", "heater": "Heater", "port": "Jet", "vent": "Vent"}[pl.kind]
        return f"{what} “{pl.name}”: drag to move ({'fixed to the body' if pl.frame == 'body' else 'fixed in the tunnel'})"

    def leave(self, view) -> None:
        if self.hover is not None:
            self.hover = None
            view.update()


# ---------------------------------------------------------------------------
# icons and the strip
# ---------------------------------------------------------------------------

#: 16 px line icons in the bundled set's style (stroke = currentColor, 1.5 px). Kept here because they
#: belong to these tools only; the shell may later move them to resources/icons.
_SVG = {
    "move": '<path d="M8 1.5v13M1.5 8h13M8 1.5 6 3.5M8 1.5l2 2M8 14.5l-2-2M8 14.5l2-2M1.5 8l2-2M1.5 8l2 2'
            'M14.5 8l-2-2M14.5 8l-2 2"/>',
    "brush": '<path d="M13.8 2.2c-1-1-4.6 2.6-7 5.6l1.4 1.4c3-2.4 6.6-6 5.6-7Z"/><path d="M6.6 8.1c-1.6 0-2.7 '
             '1.1-2.9 2.6-.1 1.1-.7 1.9-1.9 2.3 2.6 1 5.9.5 6.2-2.5"/>',
    "polygon": '<path d="M3 4.5 9.5 2 14 7.5 10.5 14 3.5 11.5Z"/><circle cx="3" cy="4.5" r=".9"/>'
               '<circle cx="9.5" cy="2" r=".9"/><circle cx="14" cy="7.5" r=".9"/><circle cx="10.5" cy="14" r=".9"/>'
               '<circle cx="3.5" cy="11.5" r=".9"/>',
    "erase": '<path d="M9.2 2.6 14 7.4 7.6 13.8H4.4L2 11.4Z"/><path d="M5.9 6 10.6 10.7M7.6 13.8H14"/>',
    "undo": '<path d="M5 3.5 2 6.5l3 3"/><path d="M2.5 6.5h7a4 4 0 0 1 0 8H6"/>',
    "redo": '<path d="m11 3.5 3 3-3 3"/><path d="M13.5 6.5h-7a4 4 0 0 0 0 8H10"/>',
    "clear": '<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 9h5.6l.7-9"/>',
    "step": '<path d="M3 3.5v9l6-4.5Z"/><path d="M12 3.5v9"/>',
    "fan": '<circle cx="8" cy="8" r="1.4"/><path d="M8 6.6C8 3 9.9 1.8 11 2.6c1 .8.2 3-3 4.2M9.3 8.6c3.1 1.7 '
           '3.2 3.9 1.9 4.4-1.2.5-2.7-1.3-2.6-4.6M6.8 8.7c-3.1 1.8-5 .7-4.8-.6.2-1.3 2.5-1.7 4.8.6"/>',
    "heater": '<path d="M5 13.5c-1.5-1.3-1.6-3.4 0-5 1.4-1.4 1.2-2.8.4-4.5 2.6 1 4 3.2 3.3 5.7 1-.5 1.6-1.6 '
              '1.6-2.9 1.7 2 1.8 4.9.2 6.7"/><path d="M3 14.5h10"/>',
    "emitter": '<circle cx="3.5" cy="8" r="1.6"/><path d="M6 8h2.5M10 5.5c1 .8 1.5 1.6 1.5 2.5s-.5 1.7-1.5 2.5'
               'M12.5 3.5c1.4 1.3 2 2.8 2 4.5s-.6 3.2-2 4.5"/>',
}


class _InlineIconEngine(QIconEngine):
    """Renders one of :data:`_SVG` in a theme colour resolved at paint time (follows theme switches)."""

    def __init__(self, name: str, colour: str = "text2", active: str = "accent") -> None:
        super().__init__()
        self._name, self._colour, self._active = name, colour, active

    def _svg(self, hexc: str) -> QByteArray:
        body = _SVG.get(self._name, _SVG["move"])
        return QByteArray((f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" '
                           f'stroke="{hexc}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
                           f'{body}</svg>').encode())

    def paint(self, painter: QPainter, rect, mode, state) -> None:
        tk = tokens()
        key = active_icon_token(self._active) if state == QIcon.State.On else self._colour
        if mode == QIcon.Mode.Disabled:
            key = "faint"
        hexc = tk.get(key, tk["text2"])
        QSvgRenderer(self._svg(hexc)).render(painter, QRectF(rect))

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        pm = QPixmap(size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.paint(p, pm.rect(), mode, state)
        p.end()
        return pm

    def clone(self) -> QIconEngine:
        return _InlineIconEngine(self._name, self._colour, self._active)


def tool_icon(name: str, colour: str = "text2") -> QIcon:
    """A 16 px line icon of this module (``move``, ``brush``, ``polygon``, ``erase``, ``undo``, ...)."""
    return QIcon(_InlineIconEngine(name, colour))


class DrawToolStrip(ToolStrip):
    """Tool buttons for live editing: Move | Brush, Polygon, Eraser | size | Undo, Redo, Clear.

    ``toolChanged(key)`` when the tool changes, ``sizeChanged(radius_m)`` when
    the brush size changes, ``undoRequested`` / ``redoRequested`` /
    ``clearRequested`` for the edit buttons. The size slider is logarithmic
    in the reference length (1 % to 50 % of it), and its value is shown in
    metres, because a brush of "12" means nothing.
    """

    toolChanged = Signal(str)
    sizeChanged = Signal(float)
    undoRequested = Signal()
    redoRequested = Signal()
    clearRequested = Signal()

    SIZE_RANGE = (0.01, 0.5)          # brush radius as a share of the reference length

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._length = 1.0
        self.buttons: dict[str, QToolButton] = {}
        for key, text, tip in DRAW_TOOLS:
            b = self.add_button(None, "", tip, checkable=True)
            b.setIcon(tool_icon(key))
            b.setIconSize(QSize(16, 16))
            b.setProperty("variant", "icon")
            b.setFixedSize(26, 24)
            b.setAccessibleName(text)
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k, emit=True))
            self.buttons[key] = b
        self.add_separator()
        self.size_label = self.add_widget(label("Size", "secondary"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(0, 100)
        self.size_slider.setValue(55)
        self.size_slider.setFixedWidth(84)
        self.size_slider.setToolTip("Brush and eraser radius")
        self.add_widget(self.size_slider)
        self.size_value = QLabel()
        self.size_value.setProperty("role", "secondary")
        self.size_value.setMinimumWidth(52)
        self.add_widget(self.size_value)
        self.add_separator()
        self.undo_btn = self._edit_button("undo", "Undo the last drawing step (Ctrl+Z)", self.undoRequested)
        self.redo_btn = self._edit_button("redo", "Redo (Ctrl+Y)", self.redoRequested)
        self.clear_btn = self._edit_button("clear", "Remove every drawn obstacle (undo brings them back)",
                                           self.clearRequested)
        self.count_label = self.add_widget(label("", "muted"))
        self.add_stretch()
        self.size_slider.valueChanged.connect(self._size_moved)
        self._tool = "move"
        self.set_tool("move")
        self._size_moved()

    def _edit_button(self, icon: str, tip: str, signal) -> QToolButton:
        b = self.add_button(None, "", tip)
        b.setIcon(tool_icon(icon))
        b.setIconSize(QSize(16, 16))
        b.setProperty("variant", "icon")
        b.setFixedSize(26, 24)
        b.setAccessibleName(tip.split(" (")[0])
        b.clicked.connect(lambda _=False: signal.emit())
        return b

    def tool(self) -> str:
        return self._tool

    def set_tool(self, key: str, *, emit: bool = False) -> None:
        if key not in self.buttons:
            return
        self._tool = key
        for k, b in self.buttons.items():
            b.setChecked(k == key)
        sized = key in ("brush", "erase")
        for w in (self.size_label, self.size_slider, self.size_value):
            w.setEnabled(sized)
        if emit:
            self.toolChanged.emit(key)

    def set_reference_length(self, length: float) -> None:
        """Scale of the size slider (the design's reference length [m])."""
        if length > 0 and math.isfinite(length):
            self._length = float(length)
            self._size_moved()

    def set_min_radius(self, radius: float) -> None:
        """The smallest brush the grid resolves [m] (about 1.5 cells: a thinner stroke breaks into beads)."""
        self._min_radius = max(0.0, float(radius)) if math.isfinite(radius) else 0.0
        self._size_moved()

    def radius(self) -> float:
        lo, hi = self.SIZE_RANGE
        f = lo * (hi / lo) ** (self.size_slider.value() / 100.0)
        return max(f * self._length, getattr(self, "_min_radius", 0.0))

    def _size_moved(self, *_) -> None:
        from ...core.units import sig
        self.size_value.setText(f"{sig(2 * self.radius(), 2)} m")
        clamped = getattr(self, "_min_radius", 0.0) >= self.radius() > 0
        self.size_value.setToolTip("Brush diameter" + (" (the smallest the explore grid resolves)" if clamped else ""))
        self.sizeChanged.emit(self.radius())

    def set_history(self, can_undo: bool, can_redo: bool, n_shapes: int) -> None:
        """Enable undo / redo / clear and say how many drawing steps there are."""
        self.undo_btn.setEnabled(can_undo)
        self.redo_btn.setEnabled(can_redo)
        self.clear_btn.setEnabled(n_shapes > 0)
        self.count_label.setText(f"{n_shapes} drawing step{'s' if n_shapes != 1 else ''}" if n_shapes else "")
