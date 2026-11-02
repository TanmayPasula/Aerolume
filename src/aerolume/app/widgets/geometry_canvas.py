"""The geometry canvas: a CAD-style 2D view of one design, and the Setup case preview.

A custom ``QWidget`` painted with ``QPainter`` rather than a ``QGraphicsView``:
a design is a handful of polygons with at most a few thousand vertices, so a
full repaint per frame is far below a millisecond, and owning the world-to-screen
transform makes the engineering overlays (adaptive metric grid, dimension lines
that keep a constant pixel size, hatched road) straightforward and crisp.

The look follows ``docs/ui_guidelines.md`` (section 4, Geometry): the viewport
is the ``surface`` colour with a light grid, outlines are drawn in the text
colour over a light neutral fill, dimensions and their labels in ``text2``,
the road is a line with a hatched band, and every line is a 1 px cosmetic pen.
Colour is used for exactly two things: the first vertex (accent, to show the
point order) and the drop target highlight.

What it draws, back to front:

* an adaptive grid whose spacing follows the 1-2-5 sequence so that labels are
  always round metric numbers, with major lines every five minors, and a small
  x/y axis marker at the origin;
* for vehicles, the road at y = 0 with a hatched band below it;
* the bodies, filled and outlined (wheels a shade darker), optionally with
  their vertices;
* for airfoils the chord line (leading edge to trailing edge) with a
  quarter-chord tick; for every design the moment reference marker;
* dimensions: overall length and height plus ground clearance for vehicles;
  chord and maximum thickness for airfoils.

Interaction: wheel zooms about the cursor, left or middle drag pans, double
click (or :meth:`fit_view`) fits the design with a short animated transition.
Files dropped on the canvas are reported through ``filesDropped`` - the page
decides what to do with them. :class:`CasePreview` reuses all of this to show a
simulation case in the wind frame: rotated geometry, the flow domain, the
inflow and a moving road.

The 3D half (:class:`GeometryView3D`, :class:`CasePreview3D`) paints the same
kind of annotations over a VTK render: bounding-box dimensions on the
silhouette edges, the tunnel, the symmetry plane, the free-stream arrow placed
clear of the body with its speed label beside it (:func:`flow_arrow_layout`)
and the smoke emitters of a 3D run, resolved by
:func:`aerolume.viz3d.flow.resolve_rake` exactly as the runner resolves them
(:func:`smoke_rake_layout`).
"""
from __future__ import annotations

import contextlib
import itertools
import logging
import math
import weakref
from collections.abc import Callable

import numpy as np
import shiboken6
from PySide6.QtCore import QEasingCurve, QObject, QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...model import Body, Design, Mesh3D
from .physics_editor import SIDE_TEXT, thermal_text
from .view3d import View3D

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# theme access (theme.py is optional at import time)
# ---------------------------------------------------------------------------

_FALLBACK_TOKENS: dict[str, object] = {
    "bg": "#efefef", "bg2": "#efefef", "panel": "#f6f6f6", "surface": "#ffffff", "border": "#d6d6d6",
    "border_strong": "#b8b8b8", "text": "#0e0e0e", "text2": "#474747", "muted": "#5c5c5c",
    "faint": "#8e8e8e", "accent": "#2463c4", "accent2": "#1b4f9e", "accent_soft": "#e2e2e2",
    "good": "#17703c", "warn": "#8f5a0b", "bad": "#b02d27", "grid": "#e6e6e6", "plot_bg": "#ffffff",
    "highlight": "#b87508",
    "series": ["#2463c4", "#d9730d", "#2f855a", "#b8322b", "#6b46c1", "#5f6b7a"],
}


def theme_tokens() -> dict:
    """The active colour tokens (``theme.tokens()``), or the light palette.

    Looked up on every paint rather than cached so a theme switch takes effect
    on the next repaint even if the ``changed`` signal was not connected.
    """
    try:
        from .. import theme
        tok = dict(theme.tokens())
    except Exception:  # noqa: BLE001 - any failure means "use defaults"
        return dict(_FALLBACK_TOKENS)
    for k, v in _FALLBACK_TOKENS.items():
        tok.setdefault(k, v)
    return tok


def connect_theme_changed(callback: Callable[[], None]) -> bool:
    """Call ``callback`` when the theme switches. Returns False if no theme module.

    ``theme.changed`` is a bound signal; older builds exposed a holder object
    with a ``changed`` attribute, so accept either. The signal outlives every
    widget, so when ``callback`` is a bound method of a QObject (``self.update``)
    the connection is dropped as soon as that object's C++ side is gone - a
    closed dialog's preview must not be called on the next theme switch.
    """
    try:
        from .. import theme
        holder = theme.changed
    except Exception:  # noqa: BLE001
        return False
    owner = getattr(callback, "__self__", None)
    owner = owner if isinstance(owner, QObject) else None
    for sig in (getattr(holder, "changed", None), holder):
        connect = getattr(sig, "connect", None)
        if connect is None:
            continue

        def slot(*_args, _sig=sig) -> None:
            if owner is not None and not shiboken6.isValid(owner):
                with contextlib.suppress(RuntimeError, TypeError):
                    _sig.disconnect(slot)
                return
            callback()

        connect(slot)
        return True
    return False


def qcolor(value: object, alpha: int | None = None) -> QColor:
    """QColor from a hex string, with an optional 0-255 alpha."""
    c = QColor(str(value))
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def mix(a: object, b: object, t: float) -> QColor:
    """Opaque blend ``a + (b - a) t``: a lighter grid line that stays opaque, so
    overlapping lines do not darken where they cross."""
    ca, cb = QColor(str(a)), QColor(str(b))
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t), round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t))


def heat_mark(tok: dict, sign: float) -> QColor:
    """Colour of a *mark* that says hot (sign > 0), cold (< 0) or neither: a heated wall's line, a
    heater's hatch and outline, a hot body's outline, a label's swatch.

    Heat is the ``highlight`` amber (docs/ui_guidelines.md section 2: the one token that means
    "heat"; 3.8:1 light, 6.2:1 dark against the field, a graphics-only contrast), cold the first
    series blue, neither the line-work grey. ``warn`` is a status colour and is kept for text."""
    if sign > 0:
        return qcolor(tok.get("highlight", tok["warn"]))
    return qcolor(tok["series"][0] if sign < 0 else tok["text2"])


def heat_text(tok: dict, sign: float) -> QColor:
    """Colour of *text* that names a hot / cold value (a wall's "45 °C"): ``highlight`` never carries
    text, so warm text is ``warn`` (5.8:1 light, 7.2:1 dark on the field), cold the series blue
    (5.7:1, 4.9:1), neither ``text2``."""
    if sign > 0:
        return qcolor(tok["warn"])
    return qcolor(tok["series"][0] if sign < 0 else tok["text2"])


#: focus reasons that come from the keyboard: the canvases draw their focus ring only for these
_KEYBOARD_FOCUS = (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason,
                   Qt.FocusReason.ShortcutFocusReason)


def keyboard_focus(reason: Qt.FocusReason) -> bool:
    """Whether a focus change by ``reason`` should show the viewport's focus ring (WCAG 2.4.7).

    The previews take the keyboard (arrow keys move the selected item, Esc clears it) but drew no
    focus indicator, so Tab landed on them invisibly. A click focuses them too; a ring after every
    click on a viewport is noise, so - like CSS ``:focus-visible`` - only Tab, Shift+Tab and
    shortcuts show it."""
    return reason in _KEYBOARD_FOCUS


def paint_viewport_focus(p: QPainter, rect: QRectF, tok: dict) -> None:
    """The keyboard-focus ring of a viewport: 2 px of the ``focus`` token just inside its edge
    (5.7:1 light, 6.1:1 dark against the field, WCAG 1.4.11). Square, as panels are (radius 0);
    2 px because 1 px along the edge of a 900 px viewport is easy to miss."""
    p.save()
    p.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(qcolor(tok.get("focus", tok["accent"])), 2.0)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    p.setPen(pen)
    p.drawRect(QRectF(rect).adjusted(1.0, 1.0, -1.0, -1.0))
    p.restore()


def _pen(colour: QColor, width: float = 1.0, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(colour, width, style)
    pen.setCosmetic(True)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    return pen


# body fill and outline: neutral, readable in both themes (text colour at low alpha)
_FILL_ALPHA = 20          # main bodies
_WHEEL_FILL_ALPHA = 48    # wheels a shade darker so they read as separate parts


# ---------------------------------------------------------------------------
# geometry helpers shared with the page (pure numpy)
# ---------------------------------------------------------------------------

def airfoil_chord(design: Design) -> tuple[np.ndarray, np.ndarray]:
    """(leading edge, trailing edge) of an airfoil design.

    The trailing edge is the vertex farthest downstream; the leading edge is the
    vertex farthest from it. That is the classical chord definition and, unlike
    "min x", stays correct after the user rotates the section.
    """
    pts = np.vstack([b.points for b in design.bodies if b.role != "wheel"] or [design.bodies[0].points])
    te = pts[int(np.argmax(pts[:, 0]))]
    le = pts[int(np.argmax(np.hypot(*(pts - te).T)))]
    return le, te


def frontal_extent(polys: list[np.ndarray]) -> float:
    """Total projected height of several outlines (union of their y-intervals).

    Wheels overlap the body in a side view, so summing heights would double
    count; the union of intervals is the 2D analogue of frontal area.
    """
    iv = sorted((float(p[:, 1].min()), float(p[:, 1].max())) for p in polys if len(p))
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in iv:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


def thickness_station(design: Design) -> tuple[float, float, np.ndarray, np.ndarray]:
    """(t/c, x/c, lower point, upper point) at the maximum thickness of an airfoil.

    Points are projected onto the chord frame and the extent normal to the chord
    is sampled at 200 stations; adequate for an annotation, not for analysis.
    The two points are in design coordinates (metres), for the dimension line.
    """
    le, te = airfoil_chord(design)
    c = float(np.hypot(*(te - le)))
    if c <= 0:
        return 0.0, 0.0, le, le
    ex = (te - le) / c
    ey = np.array([-ex[1], ex[0]])
    pts = np.vstack([b.points for b in design.bodies if b.role != "wheel"])
    loc = np.column_stack([(pts - le) @ ex, (pts - le) @ ey]) / c
    poly = np.vstack([loc, loc[:1]])
    a, b = poly[:-1], poly[1:]
    best = (0.0, 0.0, 0.0, 0.0)          # t, x, y_lo, y_hi (chord units)
    for x in np.linspace(0.005, 0.995, 200):
        # intersections of the closed outline with the line x = const
        m = (a[:, 0] - x) * (b[:, 0] - x) <= 0
        m &= a[:, 0] != b[:, 0]
        if m.sum() < 2:
            continue
        t = (x - a[m, 0]) / (b[m, 0] - a[m, 0])
        ys = a[m, 1] + t * (b[m, 1] - a[m, 1])
        th = float(ys.max() - ys.min())
        if th > best[0]:
            best = (th, float(x), float(ys.min()), float(ys.max()))
    th, xs, ylo, yhi = best
    lo = le + c * (xs * ex + ylo * ey)
    hi = le + c * (xs * ex + yhi * ey)
    return th, xs, lo, hi


def max_thickness(design: Design) -> tuple[float, float]:
    """(t/c, x/c of max thickness) of an airfoil (see :func:`thickness_station`)."""
    th, xs, _, _ = thickness_station(design)
    return th, xs


def _lower_y_at(poly: np.ndarray, x: float) -> float | None:
    """Lowest crossing of the closed outline with the vertical line at ``x``."""
    a, b = poly, np.roll(poly, -1, axis=0)
    m = ((a[:, 0] - x) * (b[:, 0] - x) <= 0) & (a[:, 0] != b[:, 0])
    if not m.any():
        return None
    t = (x - a[m, 0]) / (b[m, 0] - a[m, 0])
    return float((a[m, 1] + t * (b[m, 1] - a[m, 1])).min())


def clearance_station(bodies: list[np.ndarray], wheels: list[np.ndarray] | None = None) -> tuple[float, float]:
    """(ground clearance, x at which to draw its dimension) for bodies on a road.

    The clearance is the lowest body point. Where to draw it matters as much:
    a flat floor reaches its minimum along its whole length, and the first
    such vertex is usually right above the front wheel, where the label would
    sit on the wheel. So a few stations clear of the wheels and of the moment
    marker at mid-length are tried (62 %, 38 %, 75 %, 25 % of the length), and
    the first where the underside is at the minimum wins; otherwise the
    lowest vertex itself.
    """
    pts = np.vstack(bodies)
    i = int(np.argmin(pts[:, 1]))
    clear = float(pts[i, 1])
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    span = max(x1 - x0, 1e-12)
    tol = 1e-4 * max(span, float(np.ptp(pts[:, 1])))
    busy = [(float(w[:, 0].min()) - 0.03 * span, float(w[:, 0].max()) + 0.03 * span) for w in wheels or []]
    for f in (0.62, 0.38, 0.75, 0.25):
        x = x0 + f * span
        if any(lo <= x <= hi for lo, hi in busy):
            continue
        ys = [y for y in (_lower_y_at(np.asarray(p, float), x) for p in bodies) if y is not None]
        if ys and min(ys) <= clear + tol:
            return clear, x
    return clear, float(pts[i, 0])


_Rect = tuple[float, float, float, float]          # (left, top, right, bottom) in pixels


def _overlaps(a: _Rect, b: _Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def clearance_label_slot(x: float, road_y: float, gap_top: float, size: tuple[float, float],
                         obstacles: list[_Rect], view: _Rect, band: float = 12.0) -> _Rect:
    """Where to put the boxed "clearance 0.15 m" label when the gap under the
    body is too small to hold it on the dimension line itself.

    The label used to sit at a fixed offset to the right of the leader line,
    which put it on the rear wheel of a car whose clearance station is close
    to that wheel (seen in the DXF import preview). Candidates, in order of
    preference: inside the gap to the right / left of the leader (only if the
    gap is tall enough), then under the road's hatched band to the right /
    left. The first that stays inside ``view`` and touches no ``obstacles``
    (screen bounding boxes of every body plus other labels) wins; if all
    collide, the under-the-road slot on the right does - no body is ever
    below the road, so it is the least bad.
    """
    w, h = size
    pad = 6.0
    in_gap = road_y - gap_top >= h + 2
    y_gap = road_y - 2 - h
    y_road = road_y + band + 3
    cands: list[tuple[float, float]] = []
    if in_gap:
        cands += [(x + pad, y_gap), (x - pad - w, y_gap)]
    cands += [(x + pad, y_road), (x - pad - w, y_road), (x - w / 2, y_road)]
    rects = [(cx, cy, cx + w, cy + h) for cx, cy in cands]
    for r in rects:
        inside = r[0] >= view[0] and r[1] >= view[1] and r[2] <= view[2] and r[3] <= view[3]
        if inside and not any(_overlaps(r, o) for o in obstacles):
            return r
    return rects[2 if in_gap else 0]


def nice_step(raw: float) -> float:
    """Smallest 1-2-5 x 10^n step >= ``raw``."""
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    e = math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * 10**e >= raw * (1 - 1e-9):
            return m * 10**e
    return 10.0 ** (e + 1)


def fmt_len(v: float, step: float | None = None) -> str:
    """Metre label with just enough decimals for the grid step (or 3 s.f.)."""
    if step is not None and step > 0:
        dec = max(0, -math.floor(math.log10(step) + 1e-9))
        s = f"{v:.{dec}f}"
        return "0" if float(s) == 0 else s          # never "-0.0"
    a = abs(v)
    if a == 0:
        return "0"
    dec = max(0, 2 - math.floor(math.log10(a) + 1e-9))     # 0.9999999 m reads "1.00"
    return f"{v:.{min(dec, 4)}f}"


def rotated_design(design: Design, alpha_deg: float) -> Design:
    """A copy of ``design`` in the wind frame: rotated by -alpha about the moment point.

    DESIGN.md section 3: a positive angle of attack is applied by rotating the
    geometry nose-up, i.e. by -alpha, about ``Design.moment_ref``; the flow
    always comes from the left. Used by the case preview when no planner is
    available to supply ``SimPlan.bodies_wind``.
    """
    a = math.radians(-float(alpha_deg))
    c, s = math.cos(a), math.sin(a)
    rot = np.array([[c, -s], [s, c]])
    ctr = np.asarray(design.moment_ref, float)
    bodies = []
    for b in design.bodies:
        pts = (np.asarray(b.points, float) - ctr) @ rot.T + ctr
        centre = None
        if b.center is not None:
            centre = tuple(((np.asarray(b.center, float) - ctr) @ rot.T + ctr).tolist())
        bodies.append(Body(b.name, pts, role=b.role, omega=b.omega, center=centre))
    return Design(id=design.id, name=design.name, kind=design.kind, bodies=bodies,
                  length_ref=design.length_ref, force_ref=design.force_ref,
                  moment_ref=tuple(design.moment_ref), source=dict(design.source or {}))


# ---------------------------------------------------------------------------
# thumbnails (template gallery, dialogs)
# ---------------------------------------------------------------------------

def render_thumbnail(design: Design | None, width: int, height: int, *, tokens: dict | None = None,
                     margin: float = 0.12, background: bool = False) -> QPixmap:
    """A small picture of a design: neutral fill and outline, road line for vehicles.

    Shares the colour logic of the canvas so thumbnails look like miniature
    viewports. ``background`` fills with the surface colour (else transparent).
    """
    tok = tokens or theme_tokens()
    pm = QPixmap(width, height)
    pm.fill(qcolor(tok.get("surface", tok["plot_bg"])) if background else Qt.GlobalColor.transparent)
    if design is None or not design.bodies:
        return pm
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    x0, x1, y0, y1 = design.bounds()
    ground = design.kind == "vehicle" or bool((design.source or {}).get("ground"))
    if ground:
        y0 = min(y0, 0.0)
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    s = min(width * (1 - 2 * margin) / w, height * (1 - 2 * margin) / h)
    ox = (width - s * w) / 2 - s * x0
    oy = (height + s * h) / 2 + s * y0

    def to_s(pts: np.ndarray) -> QPolygonF:
        return QPolygonF([QPointF(ox + s * x, oy - s * y) for x, y in pts])

    if ground:
        p.setPen(_pen(qcolor(tok["muted"])))
        p.drawLine(QPointF(2, oy + 0.5), QPointF(width - 2, oy + 0.5))
    for b in design.bodies:
        wheel = b.role == "wheel"
        p.setBrush(qcolor(tok["text"], _WHEEL_FILL_ALPHA if wheel else _FILL_ALPHA + 8))
        p.setPen(_pen(qcolor(tok["text"], 220)))
        p.drawPolygon(to_s(b.points))
    p.end()
    return pm


# ---------------------------------------------------------------------------
# the canvas
# ---------------------------------------------------------------------------

class GeometryCanvas(QWidget):
    """Interactive view of a :class:`~aerolume.model.Design` (metres, y up).

    Signals:
        ``hoverChanged(float, float, bool)`` - world x, y under the cursor and
        whether the cursor is inside the widget.
        ``filesDropped(list)`` - local file paths dropped on the canvas.
        ``physicsSelected(object)`` - the key of the placed item clicked (a
        :func:`~aerolume.app.widgets.physics_editor.scene_specs` key), None for a
        click on empty space.
        ``physicsMoved(object, str, object)`` - key, handle and the (x, y) world
        point [m] while a handle is dragged (snapped to the grid unless Shift is
        held); ``physicsDropped`` the same once, when the drag ends or an arrow key
        nudged the selected item.

    The **physics layer** (``set_physics(specs)``, toggled by
    :meth:`set_show_physics`) draws emitters, heat sources, fans, jets, vents and
    the domain box from plain drawing specs (wind frame, metres) and lets the user
    drag their handles: an item by its centre, a line or rake by either end, the
    domain by an edge or a corner. The canvas never edits the model: it reports
    where a handle was dragged and the page moves the item.
    """

    hoverChanged = Signal(float, float, bool)
    filesDropped = Signal(list)
    physicsSelected = Signal(object)
    physicsMoved = Signal(object, str, object)
    physicsDropped = Signal(object, str, object)

    #: how close [px] to a handle or a drawn line a press grabs it
    PHYS_HIT_PX = 6.0
    #: snapping step: the 1-2-5 world length nearest this many pixels (Shift disables it)
    SNAP_PX = 8.0

    # pixels kept free around a fitted design for dimensions and grid labels
    FIT_MARGIN = (64, 44, 70, 64)        # left, top, right, bottom

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 220)
        self.focus_visible = False      # focused from the keyboard: the ring is drawn (keyboard_focus)
        self._design: Design | None = None
        self._cx, self._cy, self._scale = 0.0, 0.0, 200.0    # view centre (m), px per metre
        self._pan_from: QPointF | None = None
        self._hover: QPointF | None = None
        self._drop_active = False
        self._needs_fit = True
        # True once the user zoomed or panned. Until then the view is "fitted" and
        # stays fitted through resizes (splitter drags, dock changes, the window
        # being laid out after the first show) - the CAD convention; afterwards a
        # resize keeps the user's view.
        self._user_view = False
        self.show_vertices = False
        self.show_dimensions = True
        self.show_grid = True
        self.show_origin = True
        self.show_hover_readout = False       # the Geometry page shows the cursor in its status line
        self.show_ground: bool | None = None    # None -> automatically for vehicles
        self.empty_text = "No design. Drop a .dat, SVG, DXF, CSV or image file here."
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim_from = (0.0, 0.0, 1.0)
        self._anim_to = (0.0, 0.0, 1.0)
        # the physics layer (see the class docstring)
        self.show_physics = True
        self.physics_editable = True
        self._physics: list[dict] = []
        self._thermal_bodies: list[dict] = []        # heated / cooled bodies: drawn, not edited here
        self._wall_labels: list[dict] = []           # heated / cooled walls' values of the paint in progress
        self.last_label_boxes: list[tuple] = []      # label boxes (x, y, w, h) px of the last paint
        self._phys_drag: dict | None = None
        self._phys_hover: tuple | None = None        # (key, handle) under the cursor
        self._press_pos: QPointF | None = None       # a left press on empty space (click = deselect)
        self._draws_domain_frame = True              # CasePreview draws the planned extent itself
        connect_theme_changed(self.update)

    # -- physics layer --------------------------------------------------------
    def set_physics(self, specs: list[dict] | None, *, editable: bool = True) -> None:
        """Show placed items and the domain (``physics_editor.scene_specs``: wind frame, metres).

        While the user drags a handle the canvas keeps drawing the specs the page sends back after
        each move, so the item follows the cursor; a new set mid-drag keeps the drag going. Heated
        and cooled bodies (``type="body_thermal"``) are kept apart: drawn and labelled, never hit
        or dragged (a body is edited on the Geometry page), and not in :meth:`physics`."""
        specs = list(specs or [])
        self._thermal_bodies = [s for s in specs if s["type"] == "body_thermal"]
        self._physics = [s for s in specs if s["type"] != "body_thermal"]
        self.physics_editable = bool(editable)
        if self._phys_hover is not None and self._spec(self._phys_hover[0]) is None:
            self._phys_hover = None
        self.setAccessibleDescription(physics_description(self._physics, self._thermal_bodies))
        sig = physics_signature(self._physics)
        if sig != getattr(self, "_phys_sig", None) and self._phys_drag is None:
            # a new set of items (a template, an item added): frame them unless the user zoomed
            self._phys_sig = sig
            if not self._user_view and self.isVisible() and self.width() > 10:
                self.fit_view(animate=False)
        self.update()

    def physics(self) -> list[dict]:
        """The drawing specs shown (see :meth:`set_physics`)."""
        return list(self._physics)

    def set_show_physics(self, on: bool) -> None:
        """Show or hide the physics layer (the page's "Physics" toggle)."""
        self.show_physics = bool(on)
        if not on:
            self._phys_hover = None
            self._phys_drag = None
        self.update()

    def selected_physics(self) -> tuple | None:
        return next((s["key"] for s in self._physics if s.get("selected")), None)

    def _spec(self, key) -> dict | None:
        return next((s for s in self._physics if s.get("key") == key), None)

    def snap_length(self) -> float:
        """The snapping step [m] at the current zoom: a 1-2-5 value about :attr:`SNAP_PX` pixels long."""
        return nice_step(self.SNAP_PX / max(self._scale, 1e-12))

    def physics_handle_point(self, key, handle: str) -> tuple[float, float] | None:
        """World point (x, y) of an item's handle as drawn (tests and keyboard nudges use it)."""
        s = self._spec(key)
        if s is None:
            return None
        if s["type"] == "domain":
            x0, x1, y0, y1 = (float(v) for v in s["box"][:4])
            xs = {"0": x0, "1": x1}
            ys = {"0": y0, "1": y1}
            hx = handle[1] if handle.startswith("x") else None
            hy = handle[handle.index("y") + 1] if "y" in handle else None
            return (xs[hx] if hx else 0.5 * (x0 + x1), ys[hy] if hy else 0.5 * (y0 + y1))
        h = s.get("handles", {}).get(handle)
        return None if h is None else (float(h[0]), float(h[1]))

    def _domain_handles(self, s: dict) -> dict[str, QPointF]:
        x0, x1, y0, y1 = (float(v) for v in s["box"][:4])
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        pts = {"x0": (x0, ym), "x1": (x1, ym), "y0": (xm, y0), "y1": (xm, y1),
               "x0y0": (x0, y0), "x1y0": (x1, y0), "x0y1": (x0, y1), "x1y1": (x1, y1)}
        return {k: self.to_screen(*v) for k, v in pts.items()}

    def physics_hit(self, pos: QPointF) -> tuple | None:
        """(key, handle) of the placed item or domain wall under ``pos`` (px), or None.

        Handles of the selected item win, then any item's handles, then an item's drawn shape
        (it moves by its centre), then the domain's corners and edges."""
        if not (self.show_physics and self._physics):
            return None
        q = np.array([pos.x(), pos.y()])
        r = self.PHYS_HIT_PX
        items = [s for s in self._physics if s["type"] != "domain"]
        items.sort(key=lambda s: not s.get("selected"))
        for s in items:
            for h, wp in s.get("handles", {}).items():
                sp = self.to_screen(float(wp[0]), float(wp[1]))
                if math.hypot(sp.x() - q[0], sp.y() - q[1]) <= r + 1:
                    return s["key"], h
        for s in items:
            if self._shape_hit2(s, q, r):
                return s["key"], "centre"
        dom = next((s for s in self._physics if s["type"] == "domain"), None)
        if dom is not None and self.physics_editable:
            hs = self._domain_handles(dom)
            for k in ("x0y0", "x1y0", "x0y1", "x1y1"):
                if math.hypot(hs[k].x() - q[0], hs[k].y() - q[1]) <= r + 2:
                    return dom["key"], k
            a, b = hs["x0y1"], hs["x1y0"]                  # top-left, bottom-right
            if a.y() - r <= q[1] <= b.y() + r:
                if abs(q[0] - a.x()) <= r:
                    return dom["key"], "x0"
                if abs(q[0] - b.x()) <= r:
                    return dom["key"], "x1"
            if a.x() - r <= q[0] <= b.x() + r:
                if abs(q[1] - b.y()) <= r:
                    return dom["key"], "y0"
                if abs(q[1] - a.y()) <= r:
                    return dom["key"], "y1"
        return None

    def _px_len(self, metres: float) -> float:
        return float(metres) * self._scale

    def _shape_hit2(self, s: dict, q: np.ndarray, r: float) -> bool:
        """Whether screen point ``q`` lies on the drawn shape of item spec ``s`` (2D)."""
        segs, polys = self._shape_geometry2(s)
        for a, b in segs:
            if _segment_distance(q, a, b) <= r:
                return True
        for poly in polys:
            if len(poly) >= 3 and _point_polygon_distance(q, np.asarray(poly)) <= r:
                return True
        return False

    def _shape_geometry2(self, s: dict) -> tuple[list, list]:
        """(segments, polygons) of an item spec in screen px: what is drawn and what is clickable."""
        t = s["type"]
        segs: list = []
        polys: list = []

        def sp(pt) -> np.ndarray:
            v = self.to_screen(float(pt[0]), float(pt[1]))
            return np.array([v.x(), v.y()])

        c = np.asarray(s.get("centre", (0.0, 0.0)), dtype=float)[:2]
        if t == "emitter":
            if "segment" in s:
                segs.append((sp(s["segment"][0]), sp(s["segment"][1])))
            for pt in np.asarray(s.get("points", np.zeros((0, 2))), dtype=float).reshape(-1, 2):
                q = sp(pt)
                polys.append([q + (-3, -3), q + (3, -3), q + (3, 3), q + (-3, 3)])
            if s.get("shape") in ("rect", "plane", "disc"):
                size = list(s.get("size") or (0.05,))
                w = float(size[0]) * (2 if s.get("shape") == "disc" else 1)
                h = float(size[1]) if len(size) > 1 and s.get("shape") != "disc" else w
                polys.append([sp(c + (-w / 2, -h / 2)), sp(c + (w / 2, -h / 2)), sp(c + (w / 2, h / 2)),
                              sp(c + (-w / 2, h / 2))])
        elif t == "heat_source":
            size = list(s.get("size") or (0.1,))
            if s.get("shape") in ("disc", "sphere"):
                rr = float(size[0])
                ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
                polys.append([sp(c + rr * np.array([math.cos(a), math.sin(a)])) for a in ang])
            else:
                w, h = float(size[0]), float(size[1] if len(size) > 1 else size[0])
                polys.append([sp(c + (-w / 2, -h / 2)), sp(c + (w / 2, -h / 2)), sp(c + (w / 2, h / 2)),
                              sp(c + (-w / 2, h / 2))])
        elif t in ("fan", "port"):
            d = np.asarray(s.get("direction", (1.0, 0.0)), dtype=float)[:2]
            d = d / max(float(np.linalg.norm(d)), 1e-12)
            n = np.array([-d[1], d[0]])
            half = 0.5 * float(s.get("diameter", s.get("size", 0.1)))
            segs.append((sp(c - half * n), sp(c + half * n)))
            a0, a1 = self._arrow2(s, c, d, half)
            segs.append((a0, a1))
        elif t == "body_thermal":
            polys.append([sp(q) for q in np.asarray(s.get("points", np.zeros((0, 2))), dtype=float).reshape(-1, 2)])
        return segs, polys

    def _arrow2(self, s: dict, c: np.ndarray, d: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
        """Screen end points of a fan's or port's arrow: from the disc downstream (fan, jet), or
        into the opening (vent); at least 26 px long so a small opening still shows its way."""
        cs = self.to_screen(float(c[0]), float(c[1]))
        c_px = np.array([cs.x(), cs.y()])
        dpx = np.array([d[0], -d[1]])                   # screen y points down
        length = max(self._px_len(2.4 * half), 26.0)
        if s["type"] == "port" and s.get("kind") == "vent":
            return c_px - dpx * length, c_px
        return c_px, c_px + dpx * length

    def design(self) -> Design | None:
        return self._design

    def set_design(self, design: Design | None, *, fit: bool | None = None) -> None:
        """Show a design. ``fit`` defaults to fitting when the design id changes,
        so live edits of the same design keep the user's zoom and pan."""
        same = (design is not None and self._design is not None and design.id == self._design.id)
        self._design = design
        if fit is None:
            fit = not same
        if fit:
            self.request_fit()
        self.update()

    def request_fit(self) -> None:
        """Fit now if the widget has a size, else as soon as it gets one."""
        if self.isVisible() and self.width() > 10:
            self.fit_view(animate=False)
        else:
            self._needs_fit = True

    def set_show_vertices(self, on: bool) -> None:
        self.show_vertices = bool(on)
        self.update()

    def set_show_dimensions(self, on: bool) -> None:
        self.show_dimensions = bool(on)
        self.update()

    def set_show_grid(self, on: bool) -> None:
        self.show_grid = bool(on)
        self.update()

    def scale(self) -> float:
        """Pixels per metre."""
        return self._scale

    def view_center(self) -> tuple[float, float]:
        return self._cx, self._cy

    def _ground_visible(self) -> bool:
        if self.show_ground is not None:
            return self.show_ground
        d = self._design
        return d is not None and (d.kind == "vehicle" or bool((d.source or {}).get("ground")))

    def _physics_points(self) -> np.ndarray:
        """World points (N, 2) of the placed items' handles (not the domain), for framing."""
        if not (self.show_physics and self._physics):
            return np.zeros((0, 2))
        pts = [np.asarray(h, dtype=float)[:2] for s in self._physics if s["type"] != "domain"
               for h in s.get("handles", {}).values()]
        return np.array(pts, dtype=float).reshape(-1, 2) if pts else np.zeros((0, 2))

    def _content_bounds(self) -> tuple[float, float, float, float] | None:
        d = self._design
        if d is None or not d.bodies:
            # no body (a natural-convection cavity, an empty room): frame the domain box
            dom = next((s for s in self._physics if s["type"] == "domain"), None) if self.show_physics else None
            if dom is None:
                return None
            x0, x1, y0, y1 = (float(v) for v in dom["box"][:4])
            return x0, x1, y0, y1
        x0, x1, y0, y1 = d.bounds()
        if self._ground_visible():
            y0 = min(y0, 0.0)
        pts = self._physics_points()
        if len(pts):                       # placed items stand around the body: keep them in view
            x0, x1 = min(x0, float(pts[:, 0].min())), max(x1, float(pts[:, 0].max()))
            y0, y1 = min(y0, float(pts[:, 1].min())), max(y1, float(pts[:, 1].max()))
        return x0, x1, y0, y1

    def _fit_margins(self) -> tuple[int, int, int, int]:
        """Pixels (left, top, right, bottom) a fit keeps free around the content."""
        return self.FIT_MARGIN

    def fit_view(self, animate: bool = True) -> None:
        """Zoom and pan so the design (and its road) fills the view with room for
        the dimension annotations."""
        b = self._content_bounds()
        if b is None:
            target = (0.0, 0.0, 200.0)
        else:
            x0, x1, y0, y1 = b
            w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
            ml, mt, mr, mb = self._fit_margins()
            avail_w = max(self.width() - ml - mr, 50)
            avail_h = max(self.height() - mt - mb, 50)
            s = min(avail_w / w, avail_h / h)
            # centre the content inside the margin box, not the widget
            cx = (x0 + x1) / 2 - (ml - mr) / 2 / s
            cy = (y0 + y1) / 2 - (mb - mt) / 2 / s
            target = (cx, cy, s)
        self._needs_fit = False
        self._user_view = False
        if animate and self.isVisible():
            self._anim.stop()
            self._anim_from = (self._cx, self._cy, self._scale)
            self._anim_to = target
            self._anim.setStartValue(0.0)
            self._anim.setEndValue(1.0)
            self._anim.start()
        else:
            self._cx, self._cy, self._scale = target
            self.update()

    def _on_anim(self, t: float) -> None:
        (ax, ay, a_s), (bx, by, b_s) = self._anim_from, self._anim_to
        t = float(t)
        # interpolate the scale geometrically so zooming feels uniform
        self._scale = a_s * (b_s / a_s) ** t if a_s > 0 and b_s > 0 else b_s
        self._cx = ax + (bx - ax) * t
        self._cy = ay + (by - ay) * t
        self.update()

    def zoom(self, factor: float, anchor: QPointF | None = None) -> None:
        """Zoom by ``factor`` keeping the world point under ``anchor`` fixed."""
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        wx, wy = self.to_world(anchor)
        self._user_view = True
        self._scale = float(np.clip(self._scale * factor, 1e-3, 1e7))
        # keep (wx, wy) under the anchor
        self._cx = wx - (anchor.x() - self.width() / 2) / self._scale
        self._cy = wy + (anchor.y() - self.height() / 2) / self._scale
        self.update()

    # -- transforms -------------------------------------------------------
    def to_screen(self, x: float, y: float) -> QPointF:
        return QPointF(self.width() / 2 + (x - self._cx) * self._scale,
                       self.height() / 2 - (y - self._cy) * self._scale)

    def to_world(self, p: QPointF) -> tuple[float, float]:
        return (self._cx + (p.x() - self.width() / 2) / self._scale,
                self._cy - (p.y() - self.height() / 2) / self._scale)

    def _poly(self, pts: np.ndarray) -> QPolygonF:
        sx = self.width() / 2 + (pts[:, 0] - self._cx) * self._scale
        sy = self.height() / 2 - (pts[:, 1] - self._cy) * self._scale
        return QPolygonF([QPointF(float(a), float(b)) for a, b in zip(sx, sy, strict=True)])

    # -- events -----------------------------------------------------------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._needs_fit:
            self.fit_view(animate=False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if (self._needs_fit or not self._user_view) and self.width() > 10:
            self._anim.stop()
            self.fit_view(animate=False)

    def wheelEvent(self, event: QWheelEvent) -> None:
        dy = event.angleDelta().y() or event.pixelDelta().y()
        if dy:
            self._anim.stop()
            self.zoom(1.0015 ** dy, event.position())
        event.accept()

    def focusInEvent(self, event) -> None:
        self.focus_visible = keyboard_focus(event.reason())
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self.focus_visible = False
        self.update()
        super().focusOutEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._press_pos = None
        if self.focus_visible:                 # the pointer took over: the ring goes, as on the web
            self.focus_visible = False
            self.update()
        if event.button() == Qt.MouseButton.LeftButton and self.physics_editable:
            hit = self.physics_hit(event.position())
            if hit is not None:
                key, handle = hit
                anchor = self.physics_handle_point(key, handle)
                if anchor is not None:
                    self._anim.stop()
                    w = self.to_world(event.position())
                    self._phys_drag = {"key": key, "handle": handle, "offset": (anchor[0] - w[0], anchor[1] - w[1]),
                                       "start": QPointF(event.position()), "moved": False, "last": anchor}
                    self.physicsSelected.emit(key)
                    self.setCursor(_handle_cursor(handle, dragging=True))
                    self.setFocus(Qt.FocusReason.MouseFocusReason)
                    event.accept()
                    return
            if self.show_physics and self._physics:
                self._press_pos = QPointF(event.position())
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._anim.stop()
            self._pan_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def _drag_target(self, pos: QPointF, snap_on: bool) -> tuple[float, float]:
        d = self._phys_drag
        w = self.to_world(pos)
        x, y = w[0] + d["offset"][0], w[1] + d["offset"][1]
        if snap_on:
            step = self.snap_length()
            x, y = round(x / step) * step + 0.0, round(y / step) * step + 0.0
        return float(x), float(y)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._phys_drag is not None:
            d = self._phys_drag
            if (pos - d["start"]).manhattanLength() >= 2 or d["moved"]:
                target = self._drag_target(pos, not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier))
                if target != tuple(d["last"]):
                    d["moved"], d["last"] = True, target
                    self.physicsMoved.emit(d["key"], d["handle"], target)
            self._hover = pos
            wx, wy = self.to_world(pos)
            self.hoverChanged.emit(wx, wy, True)
            self.update()
            event.accept()
            return
        if self._pan_from is None and event.buttons() == Qt.MouseButton.NoButton:
            hit = self.physics_hit(pos) if self.physics_editable else None
            if hit != self._phys_hover:
                self._phys_hover = hit
                if hit is None:
                    self.unsetCursor()
                    self.setToolTip("")
                else:
                    s = self._spec(hit[0])
                    self.setCursor(_handle_cursor(hit[1]))
                    name = s.get("name", "") if s else ""
                    what = ("Drag to move this wall of the domain" if hit[0] == ("domain",) else
                            f"{name}: drag to move" if hit[1] == "centre" else f"{name}: drag this end")
                    self.setToolTip(f"{what} (snaps to the grid; hold Shift for free placement)")
        if self._pan_from is not None:
            d = pos - self._pan_from
            if d.x() or d.y():
                self._user_view = True
            self._cx -= d.x() / self._scale
            self._cy += d.y() / self._scale
            self._pan_from = pos
        self._hover = pos
        wx, wy = self.to_world(pos)
        self.hoverChanged.emit(wx, wy, True)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._phys_drag is not None:
            d, self._phys_drag = self._phys_drag, None
            self.unsetCursor()
            if d["moved"]:
                self.physicsDropped.emit(d["key"], d["handle"], tuple(d["last"]))
            self.update()
            event.accept()
            return
        press, self._press_pos = self._press_pos, None
        if (press is not None and event.button() == Qt.MouseButton.LeftButton
                and (event.position() - press).manhattanLength() <= 3):
            self.physicsSelected.emit(None)            # a click on empty space clears the selection
        self._pan_from = None
        self.unsetCursor()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self.physics_hit(event.position()) is not None:
            event.accept()                              # a double click on an item does not refit
            return
        self.fit_view()

    def leaveEvent(self, event) -> None:
        self._hover = None
        self._phys_hover = None
        self.hoverChanged.emit(0.0, 0.0, False)
        self.update()
        super().leaveEvent(event)

    def nudge_selected(self, dx_steps: float, dy_steps: float) -> bool:
        """Move the selected item by whole snapping steps (the keyboard alternative to dragging)."""
        key = self.selected_physics()
        if key is None or key == ("domain",) or not self.physics_editable or not self.show_physics:
            return False
        p = self.physics_handle_point(key, "centre")
        if p is None:
            return False
        step = self.snap_length()
        target = (p[0] + dx_steps * step, p[1] + dy_steps * step)
        self.physicsMoved.emit(key, "centre", target)
        self.physicsDropped.emit(key, "centre", target)
        return True

    def keyPressEvent(self, event) -> None:
        k = event.key()
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, 1), Qt.Key.Key_Down: (0, -1)}
        if k in arrows and self.selected_physics() not in (None, ("domain",)):
            f = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            if self.nudge_selected(arrows[k][0] * f, arrows[k][1] * f):
                event.accept()
                return
        if k == Qt.Key.Key_Escape and self.selected_physics() is not None:
            self.physicsSelected.emit(None)
            event.accept()
            return
        if k in (Qt.Key.Key_F, Qt.Key.Key_Home):
            self.fit_view()
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom(1.25)
        elif k == Qt.Key.Key_Minus:
            self.zoom(0.8)
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:
        md = event.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            self._drop_active = True
            event.acceptProposedAction()
            self.update()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._drop_active = False
        self.update()

    def dropEvent(self, event) -> None:
        self._drop_active = False
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.update()
        if paths:
            event.acceptProposedAction()
            self.filesDropped.emit(paths)

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        tok = theme_tokens()
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            p.fillRect(self.rect(), qcolor(tok.get("surface", tok["plot_bg"])))
            if self.show_grid:
                self._paint_grid(p, tok)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._paint_underlay(p, tok)
            if self._ground_visible():
                self._paint_ground(p, tok)
            d = self._design
            if self.show_origin and d is not None and d.bodies:     # not under the empty-view text
                self._paint_origin(p, tok)
            physics = self.show_physics and bool(self._physics or self._thermal_bodies)
            if d is not None and d.bodies:
                self._paint_bodies(p, tok, d)
                if d.kind == "airfoil":
                    self._paint_chord(p, tok, d)
                self._paint_moment_ref(p, tok, d)
                if self.show_dimensions:
                    self._paint_dimensions(p, tok, d)
            elif not (physics and d is not None):
                self._paint_empty(p, tok)
            self.last_label_boxes = []
            if physics:
                self._paint_physics(p, tok)
            self._paint_overlay(p, tok)
            if self.show_hover_readout:
                self._paint_hover(p, tok)
            if self._drop_active:
                self._paint_drop(p, tok)
            if self.focus_visible and self.hasFocus():
                paint_viewport_focus(p, QRectF(self.rect()), tok)
        finally:
            p.end()

    def _paint_underlay(self, p: QPainter, tok: dict) -> None:
        """Hook for subclasses: drawn above the grid, below the road and bodies."""

    def _paint_overlay(self, p: QPainter, tok: dict) -> None:
        """Hook for subclasses: drawn above everything but the drop highlight."""

    def _font(self, size: float = 8.0, bold: bool = False) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(size)
        f.setBold(bold)
        return f

    def _paint_grid(self, p: QPainter, tok: dict) -> None:
        w, h = self.width(), self.height()
        minor = nice_step(12.0 / self._scale)          # minor lines >= 12 px apart
        major = nice_step(minor * 4.5)                  # labels on majors, ~5x minor
        x0, y1 = self.to_world(QPointF(0, 0))
        x1, y0 = self.to_world(QPointF(w, h))
        surface = tok.get("surface", tok["plot_bg"])
        major_c = qcolor(tok["grid"])
        minor_c = mix(surface, tok["grid"], 0.45)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for step, colour in ((minor, minor_c), (major, major_c)):
            p.setPen(_pen(colour))
            i0, i1 = math.floor(x0 / step), math.ceil(x1 / step)
            j0, j1 = math.floor(y0 / step), math.ceil(y1 / step)
            if i1 - i0 > 3000 or j1 - j0 > 3000:
                continue
            for i in range(i0, i1 + 1):
                sx = round(self.to_screen(i * step, 0).x())
                p.drawLine(sx, 0, sx, h)
            for j in range(j0, j1 + 1):
                sy = round(self.to_screen(0, j * step).y())
                p.drawLine(0, sy, w, sy)
        # labels along the left and bottom edges, on the surface colour
        p.setFont(self._font(7.5))
        fm = QFontMetricsF(p.font())
        p.setPen(qcolor(tok["muted"]))
        unit = "m"
        unit_w = fm.horizontalAdvance(unit) + 10
        for i in range(math.floor(x0 / major), math.ceil(x1 / major) + 1):
            x = i * major
            sx = self.to_screen(x, 0).x()
            text = fmt_len(x, major)
            tw = fm.horizontalAdvance(text)
            if 34 < sx - tw / 2 and sx + tw / 2 < w - unit_w - 4:
                p.drawText(QPointF(sx - tw / 2, h - 5), text)
        for j in range(math.floor(y0 / major), math.ceil(y1 / major) + 1):
            y = j * major
            sy = self.to_screen(0, y).y()
            if 12 < sy < h - 22:
                p.drawText(QPointF(5, sy + fm.ascent() / 2 - 1), fmt_len(y, major))
        p.drawText(QPointF(w - unit_w + 2, h - 5), unit)

    def _paint_origin(self, p: QPainter, tok: dict) -> None:
        o = self.to_screen(0.0, 0.0)
        if not (-20 < o.x() < self.width() + 20 and -20 < o.y() < self.height() + 20):
            return
        col = qcolor(tok["muted"])
        p.setPen(_pen(col))
        p.setBrush(col)
        n = 18.0
        p.drawLine(o, QPointF(o.x() + n, o.y()))
        p.drawLine(o, QPointF(o.x(), o.y() - n))
        p.drawPolygon(QPolygonF([QPointF(o.x() + n + 4, o.y()), QPointF(o.x() + n - 1, o.y() - 2.5),
                                 QPointF(o.x() + n - 1, o.y() + 2.5)]))
        p.drawPolygon(QPolygonF([QPointF(o.x(), o.y() - n - 4), QPointF(o.x() - 2.5, o.y() - n + 1),
                                 QPointF(o.x() + 2.5, o.y() - n + 1)]))
        p.setFont(self._font(7.5))
        p.drawText(QPointF(o.x() + n + 6, o.y() + 4), "x")
        p.drawText(QPointF(o.x() - 3, o.y() - n - 7), "y")

    def _paint_ground(self, p: QPainter, tok: dict) -> None:
        w, h = self.width(), self.height()
        sy = round(self.to_screen(0, 0).y()) + 0.5
        if sy > h + 1:
            return
        band = max(min(12.0, h - sy), 0.0)
        if band > 0:
            p.save()
            p.setClipRect(QRectF(0, sy, w, band))
            p.setPen(_pen(qcolor(tok["border_strong"])))
            step = 7.0
            for x in np.arange(-band, w + band, step):
                p.drawLine(QPointF(x, sy + band), QPointF(x + band, sy))
            p.restore()
        p.setPen(_pen(qcolor(tok["text2"])))
        p.drawLine(QPointF(0, sy), QPointF(w, sy))

    def _paint_bodies(self, p: QPainter, tok: dict, d: Design) -> None:
        outline = qcolor(tok["text"])
        for b in d.bodies:
            if len(b.points) < 2:
                continue
            poly = self._poly(b.points)
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            wheel = b.role == "wheel"
            p.setBrush(qcolor(tok["text"], _WHEEL_FILL_ALPHA if wheel else _FILL_ALPHA))
            p.setPen(_pen(outline))
            p.drawPath(path)
            if self.show_vertices:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qcolor(tok["text2"]))
                r = 1.5 if len(b.points) < 600 else 1.0
                for q in poly:
                    p.drawRect(QRectF(q.x() - r, q.y() - r, 2 * r, 2 * r))
                # first vertex marked so orientation and start are visible
                p.setBrush(qcolor(tok["accent"]))
                p.drawRect(QRectF(poly[0].x() - 2.5, poly[0].y() - 2.5, 5, 5))

    def _paint_chord(self, p: QPainter, tok: dict, d: Design) -> None:
        le, te = airfoil_chord(d)
        a, b = self.to_screen(*le), self.to_screen(*te)
        p.setPen(_pen(qcolor(tok["muted"]), 1.0, Qt.PenStyle.DashLine))
        p.drawLine(a, b)
        q = le + 0.25 * (te - le)
        qs = self.to_screen(*q)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        n = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / n, dx / n
        p.setPen(_pen(qcolor(tok["text2"])))
        p.drawLine(QPointF(qs.x() - 5 * nx, qs.y() - 5 * ny), QPointF(qs.x() + 5 * nx, qs.y() + 5 * ny))

    def _paint_moment_ref(self, p: QPainter, tok: dict, d: Design) -> None:
        """The conventional centre-of-gravity symbol: a small circle with two
        opposite quadrants filled."""
        c = self.to_screen(*d.moment_ref)
        r = 4.5
        col = qcolor(tok["text"])
        rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(tok.get("surface", tok["plot_bg"])))
        p.drawEllipse(rect)
        p.setBrush(col)
        p.drawPie(rect, 0 * 16, 90 * 16)
        p.drawPie(rect, 180 * 16, 90 * 16)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(col))
        p.drawEllipse(rect)

    def _dim_label(self, p: QPainter, tok: dict, centre: QPointF, text: str) -> QRectF:
        """Dimension text centred on ``centre`` on a surface-coloured gap (CAD style:
        the dimension line breaks around its value)."""
        p.setFont(self._font(8.0))
        fm = QFontMetricsF(p.font())
        tw, th = fm.horizontalAdvance(text) + 8, fm.height() + 2
        box = QRectF(centre.x() - tw / 2, centre.y() - th / 2, tw, th)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(tok.get("surface", tok["plot_bg"])))
        p.drawRect(box)
        p.setPen(qcolor(tok["text2"]))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        return box

    def _dim_line(self, p: QPainter, tok: dict, a: QPointF, b: QPointF, text: str,
                  ext_from: tuple[QPointF, QPointF] | None = None) -> None:
        """A dimension: extension lines, a line with arrowheads, centred label."""
        col = qcolor(tok["text2"])
        if ext_from is not None:
            p.setPen(_pen(qcolor(tok["muted"], 150)))
            for src, dst in ((ext_from[0], a), (ext_from[1], b)):
                # extension lines stop 3 px short of the object and run 4 px past the dimension
                dx, dy = dst.x() - src.x(), dst.y() - src.y()
                n = math.hypot(dx, dy)
                if n > 4:
                    ux, uy = dx / n, dy / n
                    p.drawLine(QPointF(src.x() + 3 * ux, src.y() + 3 * uy),
                               QPointF(dst.x() + 4 * ux, dst.y() + 4 * uy))
        p.setPen(_pen(col))
        p.drawLine(a, b)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        n = math.hypot(dx, dy)
        if n < 1:
            return
        ux, uy = dx / n, dy / n
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        for tip, sgn in ((a, 1), (b, -1)):
            back = QPointF(tip.x() + sgn * 7 * ux, tip.y() + sgn * 7 * uy)
            left = QPointF(back.x() - 2.2 * uy, back.y() + 2.2 * ux)
            right = QPointF(back.x() + 2.2 * uy, back.y() - 2.2 * ux)
            p.drawPolygon(QPolygonF([tip, left, right]))
        if text:        # an unlabelled dimension (the label sits elsewhere) needs no gap
            self._dim_label(p, tok, QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2), text)

    def _paint_dimensions(self, p: QPainter, tok: dict, d: Design) -> None:
        x0, x1, y0, y1 = d.bounds()
        tl, br = self.to_screen(x0, y1), self.to_screen(x1, y0)
        if br.x() - tl.x() < 40:
            return
        if d.kind == "airfoil":
            le, te = airfoil_chord(d)
            c = float(np.hypot(*(te - le)))
            les, tes = self.to_screen(*le), self.to_screen(*te)
            yb = br.y() + 30
            self._dim_line(p, tok, QPointF(les.x(), yb), QPointF(tes.x(), yb), f"c = {fmt_len(c)} m",
                           ext_from=(les, tes))
            tc, xc, lo, hi = thickness_station(d)
            if tc > 0:
                los, his = self.to_screen(*lo), self.to_screen(*hi)
                if abs(los.y() - his.y()) > 14:
                    self._dim_line(p, tok, his, los, "")
                lab = f"t/c = {100 * tc:.1f} % at x/c = {xc:.2f}"
                p.setFont(self._font(8.0))
                fm = QFontMetricsF(p.font())
                top = min(tl.y(), his.y())
                self._dim_label(p, tok, QPointF(his.x() + fm.horizontalAdvance(lab) / 2 - 6, top - 16), lab)
                p.setPen(_pen(qcolor(tok["muted"], 150)))
                p.drawLine(QPointF(his.x(), his.y() - 3), QPointF(his.x(), top - 10))
            return
        # length below the body (below the road band for vehicles)
        ground = self._ground_visible()
        base = self.to_screen(0, 0).y() if ground else br.y()
        yb = max(base, br.y()) + (34 if ground else 28)
        length_text = f"{fmt_len(x1 - x0)} m"
        self._dim_line(p, tok, QPointF(tl.x(), yb), QPointF(br.x(), yb), length_text,
                       ext_from=(QPointF(tl.x(), max(base, br.y())), QPointF(br.x(), max(base, br.y()))))
        fm = QFontMetricsF(self._font(8.0))
        lw, lh = fm.horizontalAdvance(length_text) + 8, fm.height() + 2
        xm = (tl.x() + br.x()) / 2
        length_box = (xm - lw / 2, yb - lh / 2, xm + lw / 2, yb + lh / 2)
        # height on the right (from the road for vehicles: what a tape measure gives)
        y_lo = 0.0 if ground and y0 >= -1e-9 else y0
        top = self.to_screen(x1, y1).y()
        bot = self.to_screen(x1, y_lo).y()
        xr = br.x() + 34
        ext_top = self._extreme_point(d, "top")
        self._dim_line(p, tok, QPointF(xr, top), QPointF(xr, bot), f"{fmt_len(y1 - y_lo)} m",
                       ext_from=(QPointF(self.to_screen(*ext_top).x(), top), QPointF(br.x(), bot)))
        if ground:
            main = [b.points for b in d.bodies if b.role != "wheel"] or [b.points for b in d.bodies]
            clear, x_dim = clearance_station(main, [b.points for b in d.bodies if b.role == "wheel"])
            if clear > 0:
                xs = self.to_screen(x_dim, 0).x()
                a, b = self.to_screen(0, clear).y(), self.to_screen(0, 0).y()
                if b - a > 22:
                    self._dim_line(p, tok, QPointF(xs, a), QPointF(xs, b), f"{fmt_len(clear)} m")
                else:
                    p.setPen(_pen(qcolor(tok["text2"])))
                    p.drawLine(QPointF(xs, a), QPointF(xs, b))
                    text = f"clearance {fmt_len(clear)} m"
                    size = (fm.horizontalAdvance(text) + 8, fm.height() + 2)
                    obstacles = [length_box]
                    for body in d.bodies:
                        lo = self.to_screen(*np.asarray(body.points, float).min(axis=0))
                        hi = self.to_screen(*np.asarray(body.points, float).max(axis=0))
                        obstacles.append((lo.x(), hi.y(), hi.x(), lo.y()))
                    m = self.to_screen(*d.moment_ref)
                    obstacles.append((m.x() - 6, m.y() - 6, m.x() + 6, m.y() + 6))
                    r = clearance_label_slot(xs, b, a, size, obstacles,
                                             (0.0, 0.0, float(self.width()), float(self.height())))
                    if r[1] > b:          # under the road: a muted leader from the road down to it
                        p.setPen(_pen(qcolor(tok["muted"], 150)))
                        p.drawLine(QPointF(xs, b), QPointF(xs, (r[1] + r[3]) / 2))
                        edge = r[0] if r[0] > xs else r[2]
                        p.drawLine(QPointF(xs, (r[1] + r[3]) / 2), QPointF(edge, (r[1] + r[3]) / 2))
                    self._dim_label(p, tok, QPointF((r[0] + r[2]) / 2, (r[1] + r[3]) / 2), text)

    @staticmethod
    def _extreme_point(d: Design, which: str) -> tuple[float, float]:
        pts = np.vstack([b.points for b in d.bodies])
        i = int(np.argmax(pts[:, 1])) if which == "top" else int(np.argmin(pts[:, 1]))
        return float(pts[i, 0]), float(pts[i, 1])

    def _paint_empty(self, p: QPainter, tok: dict) -> None:
        if not self.empty_text:
            return
        p.setPen(qcolor(tok["muted"]))
        p.setFont(self._font(9.0))
        p.drawText(QRectF(self.rect()).adjusted(40, 40, -40, -40),
                   Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_text)

    def _paint_hover(self, p: QPainter, tok: dict) -> None:
        if self._hover is None:
            return
        wx, wy = self.to_world(self._hover)
        dec = max(2, min(6, 1 - math.floor(math.log10(max(1.0 / self._scale, 1e-9)))))
        text = f"x {wx:.{dec}f} m   y {wy:.{dec}f} m"
        p.setFont(self._font(8.0))
        fm = QFontMetricsF(p.font())
        box = QRectF(self.width() - fm.horizontalAdvance(text) - 18, 6, fm.horizontalAdvance(text) + 12,
                     fm.height() + 6)
        p.setPen(_pen(qcolor(tok["border"])))
        p.setBrush(qcolor(tok.get("surface", tok["plot_bg"])))
        p.drawRect(box)
        p.setPen(qcolor(tok["text2"]))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_drop(self, p: QPainter, tok: dict) -> None:
        r = QRectF(self.rect()).adjusted(4.5, 4.5, -4.5, -4.5)
        p.setBrush(qcolor(tok["accent"], 22))
        p.setPen(_pen(qcolor(tok["accent"]), 1.0, Qt.PenStyle.DashLine))
        p.drawRect(r)
        p.setFont(self._font(9.0))
        p.setPen(qcolor(tok["accent"]))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Drop to import")

    # -- the physics layer -------------------------------------------------------
    def _paint_physics(self, p: QPainter, tok: dict) -> None:
        """Placed items over the geometry: neutral line work (text2) like the rest of the drawing,
        heaters in the heat amber (coolers in the series blue, :func:`heat_mark`), smoke in its own
        colour, the selected or hovered item in the accent with its handles; inactive items faded."""
        dom = next((s for s in self._physics if s["type"] == "domain"), None)
        self._wall_labels: list[dict] = []           # the walls' values, laid out with the names
        if dom is not None:
            self._paint_domain2(p, tok, dom)
        for s in self._thermal_bodies:
            # a heated (cooled) body: its outline in the heater (series blue) colour, as a wall is
            col = heat_mark(tok, s["sign"])
            _segs, polys = self._shape_geometry2(s)
            if polys and len(polys[0]) >= 2:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(_pen(col, 2.5))
                p.drawPolygon(QPolygonF([QPointF(*q) for q in polys[0]]))
        items = [s for s in self._physics if s["type"] != "domain"]
        items.sort(key=lambda s: bool(s.get("selected")))             # the selection on top
        hover_key = self._phys_hover[0] if self._phys_hover else None
        drag_key = self._phys_drag["key"] if self._phys_drag else None
        p.setFont(self._font(7.5))
        for s in items:
            hot = s.get("selected") or s["key"] in (hover_key, drag_key)
            p.save()
            if not s.get("enabled", True):
                p.setOpacity(0.45)
            try:
                self._paint_item2(p, tok, s, bool(hot))
            finally:
                p.restore()
        self._paint_labels2(p, tok, items + self._thermal_bodies)
        for s in items:
            if s.get("selected") or s["key"] in (hover_key, drag_key):
                self._paint_handles2(p, tok, s)

    def _paint_domain2(self, p: QPainter, tok: dict, s: dict) -> None:
        hs = self._domain_handles(s)
        hot = s.get("selected") or (self._phys_hover is not None and self._phys_hover[0] == s["key"]) or (
            self._phys_drag is not None and self._phys_drag["key"] == s["key"])
        r = QRectF(hs["x0y1"], hs["x1y0"])
        if self._draws_domain_frame or s.get("explicit"):
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(_pen(qcolor(tok["accent"] if hot else tok["text2"]), 1.0,
                          Qt.PenStyle.SolidLine if s.get("closed") else Qt.PenStyle.DashLine))
            p.drawRect(r)
        self._paint_walls2(p, tok, s, r)
        if not self.physics_editable:
            return
        size = 6.0
        for k, q in hs.items():
            active = (self._phys_hover is not None and self._phys_hover == (s["key"], k)) or (
                self._phys_drag is not None and self._phys_drag["key"] == s["key"] and self._phys_drag["handle"] == k)
            if not (hot or len(k) == 4):
                continue                           # corners always, edge midpoints when the domain is active
            p.setPen(_pen(qcolor(tok["accent"] if active else tok["text2"])))
            p.setBrush(qcolor(tok["accent"]) if active else qcolor(tok.get("surface", tok["plot_bg"])))
            p.drawRect(QRectF(q.x() - size / 2, q.y() - size / 2, size, size))

    def _paint_walls2(self, p: QPainter, tok: dict, s: dict, r: QRectF) -> None:
        """Heated and cooled domain walls: the side drawn in the heat amber (warmer than ambient)
        or the series blue (colder), with its temperature or heat flux just inside the box (the text
        in :func:`heat_text`, which stays readable where the amber is a mark only)."""
        walls = [w for w in s.get("walls", ()) if w["side"] in ("left", "right", "bottom", "top")]
        if not walls:
            return
        p.setFont(self._font(7.5))
        fm = QFontMetricsF(p.font())
        for w in walls:
            col = heat_text(tok, w["sign"])
            a, b = {"left": (r.topLeft(), r.bottomLeft()), "right": (r.topRight(), r.bottomRight()),
                    "bottom": (r.bottomLeft(), r.bottomRight()), "top": (r.topLeft(), r.topRight())}[w["side"]]
            p.setPen(_pen(heat_mark(tok, w["sign"]), 3.0))
            p.drawLine(a, b)
            text = (f"{w['value']:.4g} °C" if w["mode"] == "temperature" else f"{w['value']:.4g} W/m²")
            tw, th = fm.horizontalAdvance(text), fm.height()
            mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
            inward = {"left": (8.0, 0.0), "right": (-8.0 - tw, 0.0), "bottom": (-tw / 2, -8.0),
                      "top": (-tw / 2, 8.0 + th)}[w["side"]]
            at = QPointF(mid.x() + inward[0], mid.y() + inward[1])
            # just inside the wall, laid out with the item names (_paint_labels2) so none covers another
            getattr(self, "_wall_labels", []).append(
                label_request(fm, (mid.x(), mid.y()), text, col, mode="box",
                              at=(at.x() - 2, at.y() - fm.ascent() - 1)))

    def _paint_item2(self, p: QPainter, tok: dict, s: dict, hot: bool) -> None:
        t = s["type"]
        line = qcolor(tok["accent"] if hot else tok["text2"])
        width = 1.5 if hot else 1.0
        segs, polys = self._shape_geometry2(s)
        if t == "emitter":
            shape = s.get("shape")
            smoke = qcolor(s.get("colour") or "#f5f5f5")
            if shape in ("rect", "plane", "disc") and polys:
                fill = QColor(smoke)
                fill.setAlpha(90)
                p.setBrush(fill)
                p.setPen(_pen(line, width, Qt.PenStyle.DashLine))
                p.drawPolygon(QPolygonF([QPointF(*q) for q in polys[-1]]))
            if "segment" in s and segs:
                a, b = segs[0]
                if shape == "line":                      # a slot: a band in the smoke colour, outlined
                    p.setPen(_pen(line, 5.0 if hot else 4.0))
                    p.drawLine(QPointF(*a), QPointF(*b))
                    p.setPen(_pen(smoke, 2.5))
                    p.drawLine(QPointF(*a), QPointF(*b))
                else:
                    p.setPen(_pen(line, width))
                    p.drawLine(QPointF(*a), QPointF(*b))
            pts = np.asarray(s.get("points", np.zeros((0, 2))), dtype=float).reshape(-1, 2)
            r = 3.0 if len(pts) <= 40 else 2.0
            p.setPen(_pen(line, width))
            p.setBrush(smoke)
            for q in pts[:400]:
                c = self.to_screen(float(q[0]), float(q[1]))
                p.drawEllipse(c, r, r)
            if s.get("species") == "particles":         # particles: a dotted ring round the source
                c = self.to_screen(*np.asarray(s["centre"], dtype=float)[:2])
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(_pen(line, 1.0, Qt.PenStyle.DotLine))
                p.drawEllipse(c, 7.0, 7.0)
        elif t == "heat_source":
            cool = float(s.get("power", 0.0)) < 0
            col = heat_mark(tok, -1.0 if cool else 1.0)                # the hatch keeps saying heat
            poly = QPolygonF([QPointF(*q) for q in polys[0]]) if polys else QPolygonF()
            path = QPainterPath()
            path.addPolygon(poly)
            path.closeSubpath()
            p.save()
            p.setClipPath(path)
            hatch = QColor(col)
            hatch.setAlpha(110)
            p.setPen(_pen(hatch))
            br = path.boundingRect()
            span = br.width() + br.height()
            for k in np.arange(0.0, span + 6.0, 6.0):
                p.drawLine(QPointF(br.left() + k, br.top()), QPointF(br.left() + k - br.height(), br.bottom()))
            p.restore()
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(_pen(qcolor(tok["accent"]) if hot else col, 1.5 if hot else 1.25))
            p.drawPath(path)
        elif t in ("fan", "port"):
            (da, db), (aa, ab) = segs[0], segs[1]
            if t == "fan":
                # the actuator disc seen edge-on: a short bar with end ticks, and its push
                p.setPen(_pen(line, 3.0 if hot else 2.5))
                p.drawLine(QPointF(*da), QPointF(*db))
            else:
                p.setPen(_pen(line, width))
                n = (db - da) / max(float(np.hypot(*(db - da))), 1e-9)
                back = np.array([n[1], -n[0]]) * 4.0
                p.drawLine(QPointF(*da), QPointF(*db))           # the opening's edges: two short walls
                p.drawLine(QPointF(*(da - back)), QPointF(*(da + back)))
                p.drawLine(QPointF(*(db - back)), QPointF(*(db + back)))
            self._arrow_px(p, line, aa, ab, width)
            if t == "port" and s.get("kind") == "vent":
                p.setPen(_pen(line, 1.0, Qt.PenStyle.DotLine))     # where the fluid goes: out
                ext = ab + (ab - aa) * 0.35
                p.drawLine(QPointF(*ab), QPointF(*ext))

    def _label_anchor2(self, s: dict) -> tuple[float, float, str] | None:
        """Where an item's name goes (px, right of its drawing and level with its top: the text's
        baseline) and the text; the drawing's top-right corner is the name's leader anchor."""
        name = s.get("name", "")
        if not name:
            return None
        segs, polys = self._shape_geometry2(s)
        if s["type"] in ("fan", "port") and segs:
            # beside the disc or the opening, not at the arrow's tip: a jet's arrow ends next to the fan
            # in front of it, and a name at the tip read as the fan's
            segs, polys = segs[:1], []
        allpts = [q for seg in segs for q in seg] + [q for poly in polys for q in poly]
        if not allpts:
            return None
        arr = np.asarray(allpts, dtype=float)
        text = item_label(s)
        return float(arr[:, 0].max()) + 6, float(arr[:, 1].min()) - 2, text

    def _paint_labels2(self, p: QPainter, tok: dict, items: list[dict]) -> None:
        """Item names and wall values on a surface backing, laid out by :func:`paint_labels` so that
        none covers another (a fan right behind an intake must not print its name over the
        intake's): the selected item keeps its spot, then the walls, then the rest, each moved a
        line down or up, or to the left of its drawing, with a leader line once it is moved away."""
        fm = QFontMetricsF(p.font())
        hover_key = self._phys_hover[0] if self._phys_hover else None
        many = len(items) > 8
        first: list[dict] = []
        rest: list[dict] = []
        for s in sorted(items, key=lambda x: not x.get("selected")):
            hot = s.get("selected") or s["key"] == hover_key
            if many and not hot:
                continue
            a = self._label_anchor2(s)
            if a is None:
                continue
            x, y, text = a
            req = label_request(fm, (x - 6, y + 2), text, qcolor(tok["accent"] if hot else tok["text2"]),
                                mode="box", at=(x - 2, y - fm.ascent() - 1), backing=200)
            (first if hot else rest).append(req)
        self.last_label_boxes = paint_labels(p, tok, fm, first + getattr(self, "_wall_labels", []) + rest,
                                             (float(self.width()), float(self.height())))

    def _paint_handles2(self, p: QPainter, tok: dict, s: dict) -> None:
        size = 7.0
        for h, wp in s.get("handles", {}).items():
            q = self.to_screen(float(wp[0]), float(wp[1]))
            active = (self._phys_hover == (s["key"], h)) or (
                self._phys_drag is not None and self._phys_drag["key"] == s["key"] and self._phys_drag["handle"] == h)
            p.setPen(_pen(qcolor(tok["accent"])))
            p.setBrush(qcolor(tok["accent"]) if active else qcolor(tok.get("surface", tok["plot_bg"])))
            if h == "centre":
                p.drawRect(QRectF(q.x() - size / 2, q.y() - size / 2, size, size))
            else:
                p.drawEllipse(q, size / 2, size / 2)

    @staticmethod
    def _arrow_px(p: QPainter, col: QColor, a: np.ndarray, b: np.ndarray, width: float = 1.0) -> None:
        d = b - a
        n = float(np.hypot(*d))
        if n < 4:
            return
        u = d / n
        p.setPen(_pen(col, width))
        p.drawLine(QPointF(*a), QPointF(*(b - 5 * u)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        back = b - 7 * u
        nrm = np.array([-u[1], u[0]]) * 3.0
        p.drawPolygon(QPolygonF([QPointF(*b), QPointF(*(back + nrm)), QPointF(*(back - nrm))]))


def physics_description(specs: list[dict], thermal_bodies: list[dict] | None = None) -> str:
    """The physics layer in words, for screen readers: what is placed and what is selected (the
    painted items have no accessibility tree of their own; the inspector lists edit them), the
    domain when it is a closed or fixed box with its heated and cooled walls, then the heated and
    cooled bodies with their conditions.

    A case with walls but no placed item (the de Vahl Davis cavity: a hot and a cold wall) was read
    as an empty description; its walls carry the whole case, so they are said."""
    heated = "".join(f" {s.get('name', '')}: {_thermal_label(s)}." for s in thermal_bodies or ())
    items = [s for s in specs if s["type"] != "domain"]
    dom = next((s for s in specs if s["type"] == "domain"), None)
    sel = next((s for s in specs if s.get("selected")), None)
    if sel is None:
        tail = ""
    elif sel["type"] == "domain":            # the walls move with the mouse or the Domain fields
        tail = " Selected: the domain; edit its edges in Setup ▸ Domain. Esc clears."
    else:
        depth = ", Page Up / Page Down toward or away from you" if int(sel.get("dim", 2)) == 3 else ""
        tail = (f" Selected: {sel.get('name', '')}; arrow keys move it{depth}, Shift ten steps at a time, "
                "Esc clears.")
    text = ""
    if items:
        names = ", ".join(f"{s.get('name', '')} ({s['type'].replace('_', ' ')}"
                          f"{'' if s.get('enabled', True) else ', off'})" for s in items)
        text = f"{len(items)} placed item{'s' if len(items) != 1 else ''}: {names}."
    return " ".join(t for t in (text, _domain_words(dom), tail.strip(), heated.strip()) if t)


def _domain_words(dom: dict | None) -> str:
    """A closed or fixed domain and its heated / cooled walls in words ("" for a planned tunnel):
    "Closed box 0.0457 × 0.0457 m; left (x min) wall held at 25 °C, right (x max) wall held at
    15 °C." The sides are named as the Heat transfer group names them."""
    if dom is None or not (dom.get("closed") or dom.get("explicit") or dom.get("walls")):
        return ""
    what = "Closed box" if dom.get("closed") else "Domain"
    box, dim = dom.get("box"), int(dom.get("dim", 2))
    if box is not None and len(box) >= 2 * dim:
        what += " " + " × ".join(f"{float(box[2 * i + 1]) - float(box[2 * i]):.3g}" for i in range(dim)) + " m"
    sides = SIDE_TEXT.get(dim, SIDE_TEXT[2])
    walls = [f"{sides.get(w['side'], w['side']).lower()} wall {_thermal_label(w)}" for w in dom.get("walls", ())]
    return what + ("; " + ", ".join(walls) if walls else "") + "."


def _thermal_label(s: dict) -> str:
    """A heated or cooled body's condition in words: "held at 30 °C" / "heat flux 500 W/m² into the
    fluid" (the sign convention of Body.thermal_value, said, since a bare flux has no direction)."""
    text = thermal_text(s.get("mode", "temperature"), float(s.get("value", 0.0)))
    return f"held at {text}" if s.get("mode") == "temperature" else f"heat flux {text} into the fluid"


def physics_signature(specs: list[dict]) -> tuple:
    """What decides the framing of a physics layer: the item keys and the domain box (not where the
    items are, so dragging one never re-frames the view under the cursor)."""
    out = []
    for s in specs:
        if s["type"] == "domain":
            out.append(("domain", bool(s.get("explicit")), tuple(round(float(v), 9) for v in s["box"])))
        else:
            out.append(tuple(s["key"]))
    return tuple(out)


def item_label(s: dict) -> str:
    """The name drawn beside a placed item: heaters with their power, and "(off)" on an item that is
    kept in the setup but not run - the fading alone is a colour-only cue (WCAG 1.4.1)."""
    name = s.get("name", "")
    if not name:
        return ""
    if s.get("type") == "body_thermal":
        return f"{name} · {thermal_text(s.get('mode', 'temperature'), float(s.get('value', 0.0)))}"
    text = name if s.get("type") != "heat_source" else f"{name} · {units_power(s)}"
    return text if s.get("enabled", True) else f"{text} (off)"


#: gap [px] kept between two placed labels
LABEL_GAP_PX = 2.0
#: a label placed further than this [px] from what it names gets a leader line back to it
LEADER_PX = 12.0


def _rects_overlap(a: tuple, b: tuple, gap: float) -> float:
    """Overlap area [px²] of boxes ``a`` and ``b`` (x, y, w, h), ``b`` grown by ``gap`` on every side."""
    w = min(a[0] + a[2], b[0] + b[2] + gap) - max(a[0], b[0] - gap)
    h = min(a[1] + a[3], b[1] + b[3] + gap) - max(a[1], b[1] - gap)
    return w * h if w > 0 and h > 0 else 0.0


def _label_candidates(req: dict) -> list[tuple[float, float]]:
    """Top-left corners to try for a label, best first (see :func:`place_labels`)."""
    ax, ay = (float(v) for v in req["anchor"])
    w, h = (float(v) for v in req["size"])
    step = h + LABEL_GAP_PX
    mode = req.get("mode", "beside")
    ne, nw, se, sw = (ax + 6, ay - 4 - h), (ax - 6 - w, ay - 4 - h), (ax + 6, ay + 4), (ax - 6 - w, ay + 4)
    if mode == "box":                       # a preferred box, then a line up or down, then mirrored left
        px, py = (float(v) for v in req["at"])
        out = [(px, py)]
        for k in range(1, 5):
            out += [(px, py + k * step), (px, py - k * step)]
        mx = ax - (px - ax) - w
        return out + [(mx, py)] + [(mx, py + s * k * step) for k in range(1, 4) for s in (1, -1)]
    if mode == "centre":                    # centred on its point (a wall's temperature), then shifted
        cx, cy = ax - w / 2, ay - h / 2
        out = [(cx, cy)]
        for k in range(1, 5):
            out += [(cx, cy + k * step), (cx, cy - k * step)]
        return out + [(cx + w / 2 + 8, cy), (cx - w / 2 - 8, cy)]
    out = [(ax - w / 2, ay - 8 - h)] if mode == "above" else []
    out += [ne, nw, se, sw]
    for k in range(1, 6):                   # stacked further out: above the high ones, below the low ones
        out += [(ne[0], ne[1] - k * step), (se[0], se[1] + k * step),
                (nw[0], nw[1] - k * step), (sw[0], sw[1] + k * step)]
    return out


def place_labels(requests: list[dict], view: tuple[float, float], *, taken: list[tuple] | None = None
                 ) -> list[dict]:
    """Lay out device labels so that none covers another: the first free spot of each, in order.

    Each request is ``{"anchor": (x, y) px, "size": (w, h) px, "mode": ...}``; ``mode`` "beside"
    (a corner off the anchor, upper right first), "above" (centred above it first, a body's label),
    "centre" (on the anchor, a wall's temperature) or "box" with ``"at"`` the preferred top-left
    corner (the 2D canvas's place right of a drawing). Requests are placed in the order given -
    the selected item first, so it keeps its own spot - and a spot counts as free when it overlaps
    no label already placed and nothing in ``taken`` (other overlays, ``(x, y, w, h)`` px) and lies
    inside the ``view`` (width, height). With no free spot the least-covered one is used. A label
    that ends further than :data:`LEADER_PX` from its anchor is marked ``leader`` so the painter
    draws a line back to what it names: in the 3D room, "Radiator · 200 W", the window wall's
    "12 °C" and "Smoke at the radiator" all wanted the same few pixels, and three stacked labels
    without leaders would not say which is which.

    Returns, per request, ``{"box": (x, y, w, h), "leader": bool, "anchor": (x, y)}``.
    """
    vw, vh = float(view[0]), float(view[1])
    placed: list[tuple] = [tuple(float(v) for v in t) for t in (taken or ())]
    out = []
    for req in requests:
        w, h = (float(v) for v in req["size"])
        best, best_cost = None, math.inf
        for x, y in _label_candidates(req):
            box = (x, y, w, h)
            outside = (max(0.0, -x) + max(0.0, x + w - vw)) * h + (max(0.0, -y) + max(0.0, y + h - vh)) * w
            cost = sum(_rects_overlap(box, t, LABEL_GAP_PX) for t in placed) + 4.0 * outside
            if cost <= 0.0:
                best = box
                break
            if cost < best_cost:
                best, best_cost = box, cost
        x, y = min(max(best[0], 0.0), max(vw - w, 0.0)), min(max(best[1], 0.0), max(vh - h, 0.0))
        box = (x, y, w, h)
        placed.append(box)
        ax, ay = (float(v) for v in req["anchor"])
        dx = max(x - ax, 0.0, ax - (x + w))
        dy = max(y - ay, 0.0, ay - (y + h))
        out.append({"box": box, "leader": math.hypot(dx, dy) > LEADER_PX, "anchor": (ax, ay)})
    return out


def leader_end(box: tuple, anchor: tuple) -> tuple[float, float]:
    """The point of label ``box`` (x, y, w, h) nearest its ``anchor``: where its leader line starts."""
    x, y, w, h = box
    return min(max(anchor[0], x), x + w), min(max(anchor[1], y), y + h)


def label_request(fm: QFontMetricsF, anchor, text: str, colour: QColor, *, mode: str = "beside",
                  swatch: QColor | None = None, at=None, backing: int = 210) -> dict:
    """A :func:`place_labels` request for ``text`` in font ``fm``, with what :func:`paint_labels`
    needs to draw it (its colour, an optional colour swatch before the text, the backing's alpha)."""
    sw = fm.ascent() - 2 if swatch is not None else 0.0
    req = {"anchor": (float(anchor[0]), float(anchor[1])), "text": text, "colour": colour, "swatch": swatch,
           "swatch_px": sw, "backing": backing, "mode": mode,
           "size": (fm.horizontalAdvance(text) + 4 + (sw + 4 if sw else 0.0), fm.height() + 2)}
    if at is not None:
        req["at"] = (float(at[0]), float(at[1]))
    return req


def paint_labels(p: QPainter, tok: dict, fm: QFontMetricsF, requests: list[dict], view: tuple[float, float],
                 taken: list[tuple] | None = None) -> list[tuple]:
    """Lay out (:func:`place_labels`) and draw device labels on a surface backing, with a leader line
    to a label that had to move away from what it names. Returns the boxes drawn."""
    if not requests:
        return []
    lay = place_labels(requests, view, taken=taken)
    lead = qcolor(tok["text2"], 170)
    for req, res in zip(requests, lay, strict=True):
        if not res["leader"]:
            continue
        a = QPointF(*res["anchor"])
        p.setPen(_pen(lead, 1.0))
        p.drawLine(a, QPointF(*leader_end(res["box"], res["anchor"])))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(lead)
        p.drawEllipse(a, 1.8, 1.8)
    for req, res in zip(requests, lay, strict=True):
        x, y, w, h = res["box"]
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(tok.get("surface", tok["plot_bg"]), int(req.get("backing", 210))))
        p.drawRect(QRectF(x, y, w, h))
        tx = x + 2
        if req.get("swatch") is not None:
            sw = float(req["swatch_px"])
            p.setBrush(req["swatch"])
            p.drawRect(QRectF(x + 2, y + (h - sw) / 2, sw, sw))
            tx += sw + 4
        p.setPen(req["colour"])
        p.drawText(QPointF(tx, y + 1 + fm.ascent()), req["text"])
    return [res["box"] for res in lay]


def units_power(s: dict) -> str:
    """A heat source's power as its label shows it: "100 W" (3D) or "100 W/m" (2D, per metre of span)."""
    v = float(s.get("power", 0.0))
    text = f"{v:,.3g}" if abs(v) < 1e4 else f"{v:,.0f}"
    return f"{text} {'W' if s.get('dim', 2) == 3 else 'W/m'}"


def _segment_distance(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance [px] from point ``q`` to the segment a-b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ab = b - a
    t = float(np.clip(np.dot(q - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.hypot(*(q - (a + t * ab))))


def _handle_cursor(handle: str, *, dragging: bool = False) -> Qt.CursorShape:
    """The cursor over a handle: a domain wall shows which way it moves, an item a move cursor."""
    if handle in ("x0", "x1"):
        return Qt.CursorShape.SizeHorCursor
    if handle in ("y0", "y1", "z0", "z1"):
        return Qt.CursorShape.SizeVerCursor
    if handle in ("x0y1", "x1y0"):
        return Qt.CursorShape.SizeFDiagCursor
    if handle in ("x0y0", "x1y1"):
        return Qt.CursorShape.SizeBDiagCursor
    if handle == "centre":
        return Qt.CursorShape.ClosedHandCursor if dragging else Qt.CursorShape.SizeAllCursor
    return Qt.CursorShape.CrossCursor


# ---------------------------------------------------------------------------
# the Setup page's case preview
# ---------------------------------------------------------------------------

class CasePreview(GeometryCanvas):
    """A simulation case in the wind frame: rotated geometry, domain, inflow, road.

    ``set_case(design, alpha_deg=..., extent=..., speed_text=..., ground=...)``:
    the geometry is rotated by -alpha about the moment point (DESIGN.md 3) unless
    ``bodies_wind`` (from ``SimPlan.bodies_wind``) is given; ``extent`` is the
    flow domain in metres (``SimPlan.extent``) and is drawn as a dashed frame
    with inflow arrows on its left edge; ``ground`` is None (free flight),
    "fixed" or "moving" (arrows along the road). ``fit_domain`` chooses whether
    a fit frames the whole domain or just the body.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(False)
        self.show_dimensions = False
        self.show_origin = False
        self.empty_text = "No geometry"
        # left: room for the inflow arrows; bottom: the moving-road arrows and
        # their label sit between the road and the grid labels
        self.FIT_MARGIN = (72, 30, 40, 64)
        self.fit_domain = False
        self._extent: tuple[float, float, float, float] | None = None
        self._ground: str | None = None
        self._speed_text = ""
        self._alpha = 0.0
        self._draws_domain_frame = False      # the planned extent is drawn by _paint_underlay
        self._inflow = True

    def set_case(self, design: Design | None, *, alpha_deg: float = 0.0,
                 extent: tuple[float, float, float, float] | None = None, speed_text: str = "",
                 ground: str | None = None, bodies_wind: list | None = None, fit: bool | None = None,
                 inflow: bool = True) -> None:
        """Show ``design`` as the solver will see it (see the class docstring). ``inflow`` False
        (a closed box: every side a wall) leaves out the inflow arrows."""
        changed_mode = bool(inflow) != self._inflow          # tunnel <-> closed box: framed differently
        self._inflow = bool(inflow)
        shown = None
        if design is not None and design.bodies:
            if bodies_wind:
                shown = Design(id=design.id, name=design.name, kind=design.kind,
                               bodies=[Body(getattr(b, "name", ""), np.asarray(getattr(b, "points", b), float),
                                            role=getattr(b, "role", "body")) for b in bodies_wind],
                               length_ref=design.length_ref, force_ref=design.force_ref,
                               moment_ref=tuple(design.moment_ref), source=dict(design.source or {}))
            else:
                shown = rotated_design(design, alpha_deg)
        elif design is not None:
            shown = design                     # no body (a cavity, an empty room): the domain is the case
        changed_frame = (extent is not None) != (self._extent is not None)
        self._extent = tuple(float(v) for v in extent) if extent is not None else None
        self._draws_domain_frame = self._extent is None    # unplanned: the physics layer frames it
        self._ground = ground
        self.show_ground = ground is not None
        self._speed_text = speed_text
        self._alpha = float(alpha_deg)
        if fit is None:
            fit = (changed_frame or changed_mode or shown is None or self._design is None
                   or shown.id != self._design.id)
        self._design = shown
        if fit:
            self.request_fit()
        self.update()

    def set_fit_domain(self, on: bool) -> None:
        self.fit_domain = bool(on)
        self.fit_view()

    #: Room around the body when framing it (fraction of its larger extent):
    #: enough to see the inflow arrows, the road ahead and the near wake.
    BODY_CONTEXT = 0.35
    #: Room around a closed box (fraction of its larger side): the box is the whole case, so it
    #: fills the view; the 35 % kept round a body in a tunnel drew a 4 × 3 m room small inside
    #: an axis running from -5 to 10 m.
    CLOSED_CONTEXT = 0.10
    #: fit margins [px] of a closed box: no inflow arrows on the left and no road below
    CLOSED_FIT_MARGIN = (44, 22, 22, 30)

    def _closed_box(self) -> tuple[float, float, float, float] | None:
        """The walls of a closed box (the planned extent, else the physics layer's domain), or None
        for a tunnel."""
        if self._inflow:
            return None
        if self._extent is not None:
            return self._extent
        dom = next((s for s in self._physics if s["type"] == "domain"), None)
        if dom is None or len(dom.get("box", ())) < 4:
            return None
        return tuple(float(v) for v in dom["box"][:4])

    def _fit_margins(self) -> tuple[int, int, int, int]:
        return self.CLOSED_FIT_MARGIN if self._closed_box() is not None else self.FIT_MARGIN

    def _content_bounds(self) -> tuple[float, float, float, float] | None:
        box = self._closed_box()
        if box is not None:
            # a closed box is the case: its walls (and anything placed past them) plus 10 %
            x0, x1, y0, y1 = box
            d = self._design
            if d is not None and d.bodies:
                bx0, bx1, by0, by1 = d.bounds()
                x0, x1, y0, y1 = min(x0, bx0), max(x1, bx1), min(y0, by0), max(y1, by1)
            pad = self.CLOSED_CONTEXT * max(x1 - x0, y1 - y0)
            return x0 - pad, x1 + pad, y0 - pad, y1 + pad
        if self.fit_domain and self._extent is not None:
            return self._extent
        b = super()._content_bounds()
        if b is None:
            return None
        x0, x1, y0, y1 = b
        pad = self.BODY_CONTEXT * max(x1 - x0, y1 - y0)
        y_lo = y0 if self._ground is not None else y0 - pad    # the road is the floor of the view
        return x0 - pad, x1 + pad, y_lo, y1 + pad

    def _paint_underlay(self, p: QPainter, tok: dict) -> None:
        if self._extent is None:
            return
        x0, x1, y0, y1 = self._extent
        a, b = self.to_screen(x0, y1), self.to_screen(x1, y0)
        r = QRectF(a, b)
        # outside the domain is "not simulated": dim it slightly
        outside = qcolor(tok["text"], 10)
        full = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(outside)
        p.drawRect(QRectF(full.left(), full.top(), full.width(), max(r.top() - full.top(), 0)))
        p.drawRect(QRectF(full.left(), r.bottom(), full.width(), max(full.bottom() - r.bottom(), 0)))
        p.drawRect(QRectF(full.left(), r.top(), max(r.left() - full.left(), 0), r.height()))
        p.drawRect(QRectF(r.right(), r.top(), max(full.right() - r.right(), 0), r.height()))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(qcolor(tok["text2"]), 1.0, Qt.PenStyle.DashLine))
        p.drawRect(r)

    def _paint_overlay(self, p: QPainter, tok: dict) -> None:
        col = qcolor(tok["text2"])
        p.setFont(self._font(8.0))
        fm = QFontMetricsF(p.font())
        # inflow arrows: on the domain's left edge, else on the view's left edge
        if self._extent is not None:
            x0, x1, y0, y1 = self._extent
            left = self.to_screen(x0, 0).x()
            top, bot = self.to_screen(0, y1).y(), self.to_screen(0, y0).y()
        else:
            left, top, bot = 18.0, 24.0, self.height() - 24.0
        top, bot = max(top, 8.0), min(bot, self.height() - 8.0)
        if self._inflow and bot - top > 30 and left < self.width() - 60:
            # a domain edge left of the view clamps to the view edge: the arrows
            # say where the flow comes from even when the body is framed
            n = 5
            ax = max(left + 6, 36.0)       # clear of the grid's y labels
            for i in range(n):
                y = top + (i + 0.5) * (bot - top) / n
                self._arrow(p, col, QPointF(ax, y), QPointF(ax + 26, y))
                if i == n // 2 and self._speed_text:
                    p.setPen(col)
                    p.drawText(QPointF(ax + 32, y + fm.ascent() / 2 - 1), self._speed_text)
        # moving road: arrows along the road, pointing downstream (car frame)
        if self._ground == "moving":
            sy = self.to_screen(0, 0).y() + 20
            if sy < self.height() - 4:
                x_start = self.to_screen(self._extent[0], 0).x() if self._extent else 0.0
                x_end = self.to_screen(self._extent[1], 0).x() if self._extent else float(self.width())
                x_start, x_end = max(x_start, 0.0) + 30, min(x_end, float(self.width())) - 30
                for x in np.arange(x_start, x_end - 20, 120.0):
                    self._arrow(p, qcolor(tok["muted"]), QPointF(x, sy), QPointF(x + 22, sy))
                p.setPen(qcolor(tok["muted"]))
                p.drawText(QPointF(x_start, sy + fm.height()), "Moving road")
        # attitude and domain size, top left
        lines = []
        if self._design is not None and abs(self._alpha) > 1e-9:
            name = "Pitch" if self._design.kind == "vehicle" else "α"
            lines.append(f"{name} = {self._alpha:g}°")
        if self._extent is not None:
            x0, x1, y0, y1 = self._extent
            lines.append(f"Domain {fmt_len(x1 - x0)} × {fmt_len(y1 - y0)} m")
        p.setPen(col)
        for i, text in enumerate(lines):
            p.drawText(QPointF(self.width() - fm.horizontalAdvance(text) - 10, 16 + i * (fm.height() + 1)), text)

    @staticmethod
    def _arrow(p: QPainter, col: QColor, a: QPointF, b: QPointF) -> None:
        p.setPen(_pen(col))
        p.drawLine(a, QPointF(b.x() - 4, b.y()))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        p.drawPolygon(QPolygonF([b, QPointF(b.x() - 6, b.y() - 2.5), QPointF(b.x() - 6, b.y() + 2.5)]))


# ---------------------------------------------------------------------------
# 3D: the Geometry page's viewport and the Setup page's case preview
# ---------------------------------------------------------------------------

def stands_on_road(design: Design | None) -> bool:
    """Whether a 3D design sits on the road plane z = 0 (vehicles, and generic bodies
    imported "on the road"): the viewport then draws the road under it."""
    if design is None:
        return False
    src = design.source if isinstance(design.source, dict) else {}
    return design.kind == "vehicle" or bool(src.get("on_road")) or bool(src.get("ground"))


def design_meshes3d(design: Design | None, airfoil_coords: Callable | None = None) -> list[Mesh3D]:
    """The surfaces a 3D design shows: its meshes, or its wing planform lofted.

    A wing design carries a :class:`~aerolume.model.WingSpec`, not a mesh; the
    viewport lofts it on demand (a few ms) exactly as the planner will, so what
    the user sees is the geometry a 3D CFD run would get. A planform that cannot
    be lofted (an unresolvable airfoil, sections out of order) shows nothing
    rather than raising: the Diagnostics group says why.
    """
    if design is None or design.dimension != 3:
        return []
    if design.meshes:
        return [m for m in design.meshes if len(m.faces) and len(m.vertices)]
    if design.kind == "wing" and design.wing is not None:
        try:
            from ...geometry3d import loft
            return [loft(design.wing, airfoil_coords=airfoil_coords, name=design.name or "Wing")]
        except Exception:  # noqa: BLE001 - an invalid planform is reported by the inspector
            return []
    return []


#: (min, max) corners of the vertices a mesh's faces use, per vertex array (weakly held).
#: The page, the inspector and the viewport all need a design's box several times per
#: change; ``np.unique`` over the faces of a 200k-triangle mesh costs ~30 ms each time.
#: Safe because the application never modifies mesh arrays in place (every geometry
#: operation returns new arrays); the faces array is checked by identity as well.
#: Keyed by ``id(vertices)``; each entry holds weak references to both arrays, so a
#: recycled id (an array freed and another allocated at its address) never matches.
_LOHI_CACHE: dict[int, tuple] = {}


def mesh_lohi(m: Mesh3D) -> tuple[np.ndarray, np.ndarray] | None:
    """(lo, hi) corners of the vertices ``m``'s faces use, or None without faces."""
    if len(m.faces) == 0 or len(m.vertices) == 0:
        return None
    hit = _LOHI_CACHE.get(id(m.vertices))
    if hit is not None and hit[0]() is m.vertices and hit[1]() is m.faces:
        return hit[2], hit[3]
    v = m.vertices[np.unique(m.faces)]
    lo, hi = v.min(axis=0), v.max(axis=0)
    with contextlib.suppress(TypeError):          # arrays that cannot be weakly referenced
        _LOHI_CACHE[id(m.vertices)] = (weakref.ref(m.vertices), weakref.ref(m.faces), lo, hi)
    if len(_LOHI_CACHE) > 512:                    # forget arrays that no longer exist
        for k in [k for k, e in _LOHI_CACHE.items() if e[0]() is None or e[1]() is None]:
            del _LOHI_CACHE[k]
    return lo, hi


def meshes_bounds(meshes: list[Mesh3D]) -> tuple[float, float, float, float, float, float] | None:
    """(xmin, xmax, ymin, ymax, zmin, zmax) over the vertices faces use, or None."""
    boxes = [b for b in (mesh_lohi(m) for m in meshes) if b is not None]
    if not boxes:
        return None
    a = np.min([b[0] for b in boxes], axis=0)
    b = np.max([b[1] for b in boxes], axis=0)
    return float(a[0]), float(b[0]), float(a[1]), float(b[1]), float(a[2]), float(b[2])


def pick_dimension_edges(bounds, project: Callable[[np.ndarray], np.ndarray], *,
                         min_px: float = 36.0) -> list[dict]:
    """For each axis, the bounding-box edge to put a dimension on in the current view.

    ``project`` maps (N, 3) world points to (N, 2) screen pixels. Of the four box
    edges parallel to an axis, the one farthest from the projected box centre
    (measured across the edge on screen) lies on the silhouette, so a dimension
    line offset outwards from it never crosses the model - the rule a draughtsman
    follows on an isometric drawing. Ties (two edges that project onto one line in
    a straight view) prefer the lower edge for x and y and the left one for z, so
    dimensions sit under and beside the model as on a drawing. An edge seen end-on
    (shorter than ``min_px``: the height in a top view) gets no dimension.

    Returns dicts with ``axis`` (0, 1, 2), the screen end points ``a`` and ``b``,
    the unit ``normal`` pointing away from the model, and ``length`` [m].
    """
    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds)
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    corners = np.array([[hi[i] if (k >> i) & 1 else lo[i] for i in range(3)] for k in range(8)])
    px = np.asarray(project(corners), dtype=float)
    if px.shape != (8, 2) or not np.all(np.isfinite(px)):
        return []
    centre = px.mean(axis=0)
    out = []
    for axis in range(3):
        o1, o2 = [j for j in range(3) if j != axis]
        best, best_score = None, -math.inf
        for s1 in (0, 1):
            for s2 in (0, 1):
                k0 = (s1 << o1) | (s2 << o2)
                k1 = k0 | (1 << axis)
                a, b = px[k0], px[k1]
                d = b - a
                n = float(np.hypot(*d))
                if n < min_px:
                    continue
                u = d / n
                normal = np.array([-u[1], u[0]])
                mid = 0.5 * (a + b)
                off = float(np.dot(mid - centre, normal))
                if off < 0:
                    normal, off = -normal, -off
                tie = (mid[1] if axis < 2 else -mid[0]) * 1e-3
                if off + tie > best_score:
                    best_score = off + tie
                    best = {"axis": axis, "a": (float(a[0]), float(a[1])), "b": (float(b[0]), float(b[1])),
                            "normal": (float(normal[0]), float(normal[1])), "length": float(hi[axis] - lo[axis])}
        if best is not None:
            out.append(best)
    return out


def wheel_colour(tok: dict) -> str:
    """Tyres a clear step darker than the body grey, in both themes."""
    return mix(tok["text"], tok.get("surface", "#ffffff"), 0.5).name()


# ---------------------------------------------------------------------------
# the smoke rake of a 3D run (DESIGN.md 7.13), as the Setup preview shows it
# ---------------------------------------------------------------------------

#: Emitter kinds the Setup page offers (a subset of ``aerolume.viz3d.flow.RAKE_KINDS``):
#: a row of nozzles across the flow at one height, a column up the height on the
#: centre plane, a span x height grid of nozzles, or a continuous band.
RAKE_KINDS = ("rake", "vertical", "grid", "plane")
#: Above this many nozzles the streaks merge into a sheet (use "plane" instead); the
#: Setup page warns, and the preview draws at most this many.
MAX_NOZZLES = 400


def nice_round(raw: float) -> float:
    """The 1-2-5 x 10^n value nearest ``raw`` on a log scale (0.27 -> 0.2, 0.4 -> 0.5).

    :func:`nice_step` rounds *up*, right for grid lines but up to 2.5x too coarse
    for a step the user is offered as a starting value."""
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    e = math.floor(math.log10(raw))
    cands = [m * 10.0 ** k for k in (e - 1, e, e + 1) for m in (1.0, 2.0, 5.0)]
    return min(cands, key=lambda c: abs(math.log(c / raw)))


def seed_rake_spec(settings: dict | None) -> dict:
    """The Setup page's smoke settings as a rake dict of :mod:`aerolume.viz3d.flow`.

    That dict (``ViewSettings.seed_rake``, metres, wind frame) is the one
    definition of "the rake": the runner releases the solver's smoke from it
    (``simulation.live3d.dye_rake_for``) and the streamlines start there. Only
    what the user fixed is written - ``z`` (a height for a rake, a range
    otherwise) and ``n`` (nozzle counts) - so everything else follows the
    resolver's defaults for the body. ``spacing`` [m] rides along: the
    resolver works in counts, the page in a pitch, and keeps ``n`` in step
    with it (:func:`counts_for_spacing`).
    """
    s = settings or {}
    kind = str(s.get("kind") or "rake")
    kind = kind if kind in RAKE_KINDS else "rake"
    spec: dict = {"kind": kind}
    zmin, zmax = s.get("z_min"), s.get("z_max")
    if zmin is not None:
        if kind == "rake":
            spec["z"] = float(zmin)
        elif zmax is not None:
            spec["z"] = [float(min(zmin, zmax)), float(max(zmin, zmax))]
    n = s.get("n")
    if n is not None and kind != "plane":
        spec["n"] = [int(v) for v in n] if np.ndim(n) else int(n)
    if s.get("spacing") is not None and kind != "plane":
        spec["spacing"] = float(s["spacing"])
    return spec


#: Most nozzles along one direction a typed spacing may ask for: a slip of the finger
#: (0.1 mm on a car) must not make the resolver build millions of points.
MAX_NOZZLES_PER_AXIS = 200


def counts_for_spacing(kind: str, y: tuple[float, float], z: tuple[float, float], spacing: float):
    """Nozzle counts that give ``spacing`` [m] over the extents ``y`` and ``z``: an int
    for a rake (across) or a vertical rake (up), [n_span, n_height] for a grid; at
    least 2 and at most :data:`MAX_NOZZLES_PER_AXIS` per direction."""
    def n(span: float) -> int:
        k = round(max(span, 0.0) / max(spacing, 1e-9)) + 1 if math.isfinite(spacing) else 2
        return int(min(max(2, k), MAX_NOZZLES_PER_AXIS))
    if kind == "rake":
        return n(y[1] - y[0])
    if kind == "vertical":
        return n(z[1] - z[0])
    return [n(y[1] - y[0]), n(z[1] - z[0])]


def _resolve(spec: dict, bounds, domain, cell: float | None):
    """``viz3d.flow.resolve_rake`` for smoke, with only the keys it defines."""
    from ...viz3d.flow import resolve_rake
    clean = {k: v for k, v in spec.items() if k in ("kind", "x", "y", "z", "n", "radius", "pulse", "points")}
    return resolve_rake(clean, bounds, domain, purpose="smoke", h=cell)


def _spacings(sr) -> tuple[float, float]:
    (y0, y1), (z0, z1), (ny, nz) = sr.y, sr.z, sr.n
    return ((y1 - y0) / (ny - 1) if ny > 1 else 0.0, (z1 - z0) / (nz - 1) if nz > 1 else 0.0)


def smoke_rake_layout(settings: dict | None, bounds, *, on_road: bool = False, extent3=None,
                      cell: float | None = None) -> dict | None:
    """Where a 3D run releases its smoke, for the Setup preview, its fields and its note.

    ``settings`` is the flat Flow visualisation dict of the Setup page
    (``smoke``, ``kind``, ``z_min``, ``z_max``, ``spacing``, ``n``; None =
    automatic), ``bounds`` the wind-frame box of the body (x0, x1, y0, y1, z0,
    z1) [m], ``extent3`` the full (mirrored) flow domain when planned and
    ``cell`` the solver cell size [m] (it sets the default nozzle radius and
    so the default spacing). The nozzles come from
    :func:`aerolume.viz3d.flow.resolve_rake` with the plan's body and domain -
    the call the runner makes - so the preview shows the nozzles the run will
    use. Without a plan the domain is a generous box around the body (the
    nozzles may shift slightly once planned). A typed ``spacing`` becomes
    nozzle counts over the resolved extents (:func:`counts_for_spacing`).

    Returns None when the smoke is off or there is no body. When the rake
    cannot be resolved, a dict with ``error`` (the reason) and no points.
    Otherwise ``kind``, ``x``, the ranges ``y`` / ``z``, ``n`` (n_span,
    n_height), ``spacing`` (the pitch shown: across for a rake and a grid, up
    for a vertical rake; 0 for a plane), ``spacing_yz``, ``points`` (N, 3)
    (none for a plane), ``count``, ``spec`` (the rake dict resolved),
    ``body_x0`` and ``auto`` (the automatic z range and spacing, which the
    page shows while those fields are on Auto).
    """
    s = dict(settings or {})
    if not s.get("smoke", True) or bounds is None:
        return None
    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds)
    size = max(x1 - x0, y1 - y0, z1 - z0)
    if not size > 0:
        return None
    if extent3 is None:
        # no plan yet: a tunnel of a few body sizes, on the road if the body stands on it
        pad = 3.0 * size
        extent3 = (x0 - pad, x1 + 2 * pad, y0 - pad, y1 + pad, 0.0 if on_road else z0 - pad, z1 + pad)
    domain = tuple(float(v) for v in extent3)
    kind = str(s.get("kind") or "rake")
    kind = kind if kind in RAKE_KINDS else "rake"
    s["kind"] = kind
    try:
        auto = _resolve({"kind": kind}, bounds, domain, cell)
        spec = seed_rake_spec(s)
        if s.get("spacing") is not None and kind != "plane":
            ext = _resolve({k: v for k, v in spec.items() if k != "n"}, bounds, domain, cell)
            spec["n"] = counts_for_spacing(kind, ext.y, ext.z, float(s["spacing"]))
        sr = _resolve(spec, bounds, domain, cell)
    except Exception as exc:  # noqa: BLE001 - the preview is advisory; the reason is shown
        return {"kind": kind, "error": f"{type(exc).__name__}: {exc}", "points": np.zeros((0, 3)), "count": 0}
    sy, sz = _spacings(sr)
    ay, az = _spacings(auto)
    pts = np.zeros((0, 3)) if kind == "plane" else np.asarray(sr.points, dtype=float).reshape(-1, 3)
    return {"kind": kind, "x": float(sr.x), "y": (float(sr.y[0]), float(sr.y[1])),
            "z": (float(sr.z[0]), float(sr.z[1])), "n": (int(sr.n[0]), int(sr.n[1])),
            "spacing": float(sz if kind == "vertical" else sy), "spacing_yz": (float(sy), float(sz)),
            "points": pts, "count": len(pts), "spec": spec, "body_x0": x0,
            "auto": {"z_min": float(auto.z[0]), "z_max": float(auto.z[1]),
                     "spacing": float(az if kind == "vertical" else ay)}}


class GeometryView3D(View3D):
    """The Geometry page's 3D viewport: a design's meshes (or its lofted wing).

    On top of :class:`~aerolume.app.widgets.view3d.View3D` (VTK rendered
    off-screen, mouse camera) it adds what a CAD viewport shows: the road plane
    under vehicles, wheels in a darker grey so the parts read separately, the
    bounding-box dimensions (length, width, height, drawn on the silhouette
    edges of the current view and re-placed as the camera moves), the moment
    reference marker, and a drop target for mesh files (``filesDropped``).

    :meth:`set_design` rebuilds the scene only when the geometry changed (the
    arrays or roles of the meshes), so editing a reference value never re-uploads
    a 200k-triangle mesh; a new design resets the camera to the iso preset, an
    edit of the same design keeps the user's view.
    """

    filesDropped = Signal(list)
    #: the physics layer, as on the 2D canvas: the key clicked (None: empty space), and a handle
    #: dragged to an (x, y, z) world point [m] (snapped unless Shift is held) / released there
    physicsSelected = Signal(object)
    physicsMoved = Signal(object, str, object)
    physicsDropped = Signal(object, str, object)

    #: room around the bounding box when framing with dimensions shown, as a fraction
    #: of the largest size: the dimension lines stand ~20 px outside the silhouette
    DIMENSION_PAD = 0.08
    #: how close [px] to a projected handle or line a press grabs a placed item
    PHYS_HIT_PX = 7.0

    def __init__(self, parent: QWidget | None = None, *, scene=None) -> None:
        super().__init__(parent, scene=scene)
        self.setAcceptDrops(True)
        self.show_dimensions = True
        # the moment point is drawn on top of the model (no depth test): on a car it sits on
        # the road under the body and would read as a mark on the door, so it is opt-in
        self.show_reference = False
        self._design: Design | None = None
        self._meshes: list[Mesh3D] = []
        self._sig: tuple | None = None
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self._mref: tuple[float, float, float] | None = None
        self._drop_active = False
        # the physics layer (placed items drawn over the render; see set_physics)
        self.show_physics = True
        self.physics_editable = True
        self._physics: list[dict] = []
        self._thermal_bodies: list[dict] = []
        #: label boxes (x, y, w, h) px of the last paint, and overlay boxes labels must keep clear of
        self.last_label_boxes: list[tuple] = []
        self._label_obstacles: list[tuple] = []
        self._phys_drag: dict | None = None
        self._phys_hover: tuple | None = None
        self._phys_press: QPointF | None = None
        self._draws_domain_frame = True
        self.focus_visible = False      # focused from the keyboard: the ring is drawn (keyboard_focus)
        self.set_placeholder("No geometry", "Import an STL, OBJ or PLY file, or drop one here.")

    # -- keyboard focus ---------------------------------------------------------------
    def focusInEvent(self, ev) -> None:
        self.focus_visible = keyboard_focus(ev.reason())
        self.update()
        super().focusInEvent(ev)

    def focusOutEvent(self, ev) -> None:
        self.focus_visible = False
        self.update()
        super().focusOutEvent(ev)

    # -- the physics layer ----------------------------------------------------------
    def set_physics(self, specs: list[dict] | None, *, editable: bool = True) -> None:
        """Placed items and the domain to draw over the model (``physics_editor.scene_specs``, 3D,
        wind frame, metres); with ``editable`` their handles can be dragged in the view. Heated and
        cooled bodies (``type="body_thermal"``) are only labelled, as in the 2D canvas."""
        specs = list(specs or [])
        self._thermal_bodies = [s for s in specs if s["type"] == "body_thermal"]
        self._physics = [s for s in specs if s["type"] != "body_thermal"]
        self.physics_editable = bool(editable)
        self.setMouseTracking(True)
        if self._phys_hover is not None and self._spec3(self._phys_hover[0]) is None:
            self._phys_hover = None
        self.setAccessibleDescription(physics_description(self._physics, self._thermal_bodies))
        sig = physics_signature(self._physics)
        if sig != getattr(self, "_phys_sig", None) and self._phys_drag is None:
            self._phys_sig = sig
            self._apply_focus()                # the camera re-frames until the user moves it
        self.update()

    def _physics_focus(self, box: tuple | None) -> tuple | None:
        """``box`` grown to hold the placed items (and a closed box's walls: the room is the case)."""
        if not (self.show_physics and self._physics):
            return box
        pts = [vec3(h) for s in self._physics if s["type"] != "domain" for h in s.get("handles", {}).values()]
        for s in self._physics:
            if s["type"] == "domain" and s.get("closed") and s.get("explicit"):
                b = s["box"]
                pts += [np.array([b[0], b[2], b[4]]), np.array([b[1], b[3], b[5]])]
        if not pts:
            return box
        a = np.asarray(pts, dtype=float)
        lo, hi = a.min(axis=0), a.max(axis=0)
        if box is not None:
            lo = np.minimum(lo, [box[0], box[2], box[4]])
            hi = np.maximum(hi, [box[1], box[3], box[5]])
        return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])

    def physics(self) -> list[dict]:
        return list(self._physics)

    def set_show_physics(self, on: bool) -> None:
        self.show_physics = bool(on)
        if not on:
            self._phys_hover = self._phys_drag = None
        self.update()

    def selected_physics(self) -> tuple | None:
        return next((s["key"] for s in self._physics if s.get("selected")), None)

    def _spec3(self, key) -> dict | None:
        return next((s for s in self._physics if s.get("key") == key), None)

    def _overlays_possible(self) -> bool:
        return bool(getattr(self.scene, "available", False)) and (
            bool(self.scene.layers()) or bool(self.show_physics and (self._physics or self._thermal_bodies)))

    def physics_handles3(self, s: dict) -> dict[str, np.ndarray]:
        """World points of a spec's handles: an item's own, or the domain's six face centres."""
        if s["type"] == "domain":
            x0, x1, y0, y1, z0, z1 = (float(v) for v in s["box"][:6])
            c = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)])
            out = {}
            for ax, (lo, hi) in enumerate(((x0, x1), (y0, y1), (z0, z1))):
                for side, v in (("0", lo), ("1", hi)):
                    q = c.copy()
                    q[ax] = v
                    out["xyz"[ax] + side] = q
            return out
        return {k: vec3(v) for k, v in s.get("handles", {}).items()}

    def _phys_lines3(self, s: dict) -> tuple[list[np.ndarray], np.ndarray]:
        """(world polylines, world nozzle points) of an item spec, as drawn."""
        t = s["type"]
        lines: list[np.ndarray] = []
        pts = np.zeros((0, 3))
        c = vec3(s.get("centre", (0.0, 0.0, 0.0)))
        if t == "domain":
            lines = [np.array([a, b], dtype=float) for a, b in box_edges(tuple(s["box"][:6]))]
        elif t == "emitter":
            shape = s.get("shape")
            if "segment" in s:
                lines.append(np.array([vec3(s["segment"][0]), vec3(s["segment"][1])]))
            pts = np.asarray(s.get("points", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
            size = list(s.get("size") or (0.05,))
            if shape == "rect":
                size += [size[-1]] * (3 - len(size))
                half = np.asarray(size[:3], dtype=float) / 2
                lines += [np.array([a, b], dtype=float) for a, b in
                          box_edges((c[0] - half[0], c[0] + half[0], c[1] - half[1], c[1] + half[1],
                                     c[2] - half[2], c[2] + half[2]))]
            elif shape == "plane":
                u, v = _basis3(vec3(s.get("normal", (1, 0, 0))))
                w, h = float(size[0]), float(size[1] if len(size) > 1 else size[0])
                quad = [c + sx * 0.5 * w * u + sy * 0.5 * h * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
                lines.append(np.array(quad))
            elif shape == "disc":
                lines.append(_circle3(c, vec3(s.get("normal", (1, 0, 0))), float(size[0])))
        elif t == "heat_source":
            size = list(s.get("size") or (0.1,))
            if s.get("shape") in ("sphere", "disc"):
                r = float(size[0])
                for n in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    lines.append(_circle3(c, np.asarray(n, dtype=float), r))
            else:
                size += [size[-1]] * (3 - len(size))
                half = np.asarray(size[:3], dtype=float) / 2
                lines += [np.array([a, b], dtype=float) for a, b in
                          box_edges((c[0] - half[0], c[0] + half[0], c[1] - half[1], c[1] + half[1],
                                     c[2] - half[2], c[2] + half[2]))]
        elif t in ("fan", "port"):
            d = vec3(s.get("direction", (1, 0, 0)))
            d = d / max(float(np.linalg.norm(d)), 1e-12)
            r = 0.5 * float(s.get("diameter", s.get("size", 0.1)))
            lines.append(_circle3(c, d, r))
            if t == "fan":                                # the actuator disc's hub-to-rim spokes
                u, v = _basis3(d)
                lines += [np.array([c - r * u, c + r * u]), np.array([c - r * v, c + r * v])]
            length = 2.4 * r
            if t == "port" and s.get("kind") == "vent":
                lines.append(np.array([c - d * length, c]))
            else:
                lines.append(np.array([c, c + d * length]))
        return lines, pts

    def _paint_walls3(self, p: QPainter, tok: dict, s: dict, labels: list[dict] | None = None) -> None:
        """Heated and cooled domain faces outlined a little inside the box, in the heater colour or
        the series blue (the 2D layer's convention), each with its temperature or heat flux at the
        face centre: a blue outline alone does not say "12 °C window". The values join ``labels``
        (laid out with the items' names by :func:`paint_labels`) when given, else are drawn here."""
        x0, x1, y0, y1, z0, z1 = (float(v) for v in s["box"][:6])
        p.setFont(self._font(7.5))
        fm = QFontMetricsF(p.font())
        inset = 0.01 * max(x1 - x0, y1 - y0, z1 - z0)
        faces = {"left": [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
                 "right": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
                 "front": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                 "back": [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
                 "bottom": [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
                 "top": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]}
        for w in s.get("walls", ()):
            quad = np.asarray(faces.get(w["side"], []), dtype=float)
            if not len(quad):
                continue
            c = quad.mean(axis=0)
            quad = c + (quad - c) * (1.0 - inset / max(float(np.linalg.norm(quad[0] - c)), 1e-12))
            px = self._project(np.vstack([quad, quad[:1]]))
            if not np.all(np.isfinite(px)):
                continue
            col = heat_text(tok, w["sign"])
            p.setPen(_pen(heat_mark(tok, w["sign"]), 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(*q) for q in px]))
            mid = self._project(np.asarray([c]))[0]
            if np.all(np.isfinite(mid)):
                text = f"{w['value']:.4g} °C" if w["mode"] == "temperature" else f"{w['value']:.4g} W/m²"
                if labels is not None:
                    labels.append(label_request(fm, mid, text, col, mode="centre"))
                    continue
                tw = fm.horizontalAdvance(text)
                at = QPointF(mid[0] - tw / 2, mid[1] + fm.ascent() / 2)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qcolor(tok.get("surface", tok["plot_bg"]), 210))
                p.drawRect(QRectF(at.x() - 2, at.y() - fm.ascent() - 1, tw + 4, fm.height() + 2))
                p.setPen(col)
                p.drawText(at, text)

    def physics_hit3(self, pos: QPointF) -> tuple | None:
        """(key, handle) of the placed item or domain face under ``pos`` (logical px), or None."""
        if not (self.show_physics and self._physics) or not getattr(self.scene, "available", False):
            return None
        q = np.array([pos.x(), pos.y()])
        r = self.PHYS_HIT_PX
        specs = sorted(self._physics, key=lambda s: (s["type"] == "domain", not s.get("selected")))
        for s in specs:
            if s["type"] == "domain" and not (self.physics_editable and (s.get("explicit") or s.get("closed"))):
                continue
            for h, wp in self.physics_handles3(s).items():
                px = self._project(np.asarray([wp]))[0]
                if np.all(np.isfinite(px)) and math.hypot(px[0] - q[0], px[1] - q[1]) <= r + 1:
                    return s["key"], h
        for s in specs:
            if s["type"] == "domain":
                continue
            lines, pts = self._phys_lines3(s)
            for ln in lines:
                px = self._project(ln)
                for a, b in itertools.pairwise(px):
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and _segment_distance(q, a, b) <= r - 2:
                        return s["key"], "centre"
            if len(pts):
                px = self._project(pts)
                ok = np.all(np.isfinite(px), axis=1)
                if np.any(ok) and float(np.min(np.hypot(*(px[ok] - q).T))) <= r:
                    return s["key"], "centre"
        return None

    def _focus_size(self) -> float:
        b = self._bounds
        if b is None:
            dom = next((s for s in self._physics if s["type"] == "domain"), None)
            b = tuple(dom["box"][:6]) if dom is not None else None
        if b is None:
            return 1.0
        return max(b[1] - b[0], b[3] - b[2], b[5] - b[4], 1e-6)

    def snap_length3(self) -> float:
        """Snapping step [m] in 3D: the 1-2-5 value nearest a 1/100 of the body (or domain) size."""
        return nice_round(self._focus_size() / 100.0)

    def _ray_plane(self, pos: QPointF, normal: np.ndarray, point: np.ndarray) -> np.ndarray | None:
        dpr = self._dpr()
        try:
            o, d = self.scene.pick_ray(pos.x() * dpr, pos.y() * dpr)
        except Exception:  # noqa: BLE001 - a scene without picking cannot drag
            return None
        den = float(np.dot(d, normal))
        if abs(den) < 1e-9:
            return None
        t = float(np.dot(point - o, normal)) / den
        return o + t * d if t > 0 else None

    def _drag_normal(self, handle: str) -> np.ndarray:
        """Plane a handle moves in: the one facing the camera's dominant axis (a face handle's own
        axis excluded, so the face can still move along it)."""
        try:
            _, _, f = self.scene.camera.frame()
            f = np.abs(np.asarray(f, dtype=float))
        except Exception:  # noqa: BLE001 - no camera yet: the side view's plane
            f = np.array([0.0, 1.0, 0.0])
        if handle in ("x0", "x1", "y0", "y1", "z0", "z1"):
            f[{"x": 0, "y": 1, "z": 2}[handle[0]]] = -1.0
        n = np.zeros(3)
        n[int(np.argmax(f))] = 1.0
        return n

    def nudge_axes3(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World unit axes the arrow keys move along: the ones nearest the screen's right and up,
        and the remaining one pointing at the viewer (PageUp). Whole axes, not the camera's own
        vectors, so a nudge keeps the item on the snapping lattice and moves one coordinate."""
        try:
            r, u, f = (np.asarray(v, dtype=float) for v in self.scene.camera.frame())
        except Exception:  # noqa: BLE001 - no camera yet: the front view (x right, z up, y away)
            r, u, f = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        taken: list[int] = []
        out = []
        for v in (r, u, -f):
            a = np.abs(v)
            a[taken] = -1.0
            i = int(np.argmax(a))
            taken.append(i)
            e = np.zeros(3)
            e[i] = 1.0 if v[i] >= 0 else -1.0
            out.append(e)
        return out[0], out[1], out[2]

    def nudge_selected3(self, right: float, up: float, toward: float = 0.0) -> bool:
        """Move the selected item by whole snapping steps along :meth:`nudge_axes3` (the keyboard
        alternative to dragging; the 2D canvas's arrow keys). Returns whether a move was sent."""
        key = self.selected_physics()
        if key is None or key == ("domain",) or not self.physics_editable or not self.show_physics:
            return False
        s = self._spec3(key)
        p = self.physics_handles3(s).get("centre") if s is not None else None
        if p is None:
            return False
        er, eu, et = self.nudge_axes3()
        target = tuple(float(v) for v in vec3(p) + self.snap_length3() * (right * er + up * eu + toward * et))
        self.physicsMoved.emit(key, "centre", target)
        self.physicsDropped.emit(key, "centre", target)
        return True

    def keyPressEvent(self, ev) -> None:
        """Arrows move the selected placed item across the screen, PageUp / PageDown toward or away
        from the viewer, one snapping step (ten with Shift); Escape clears the selection."""
        k = ev.key()
        moves = {Qt.Key.Key_Left: (-1, 0, 0), Qt.Key.Key_Right: (1, 0, 0), Qt.Key.Key_Up: (0, 1, 0),
                 Qt.Key.Key_Down: (0, -1, 0), Qt.Key.Key_PageUp: (0, 0, 1), Qt.Key.Key_PageDown: (0, 0, -1)}
        if k in moves and self.selected_physics() not in (None, ("domain",)):
            f = 10.0 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            if self.nudge_selected3(*(f * m for m in moves[k])):
                ev.accept()
                return
        if k == Qt.Key.Key_Escape and self.selected_physics() is not None:
            self.physicsSelected.emit(None)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def mousePressEvent(self, ev) -> None:
        self._phys_press = None
        if self.focus_visible:                 # the pointer took over: the ring goes
            self.focus_visible = False
            self.update()
        if ev.button() == Qt.MouseButton.LeftButton and self.physics_editable:
            hit = self.physics_hit3(ev.position())
            if hit is not None:
                key, handle = hit
                s = self._spec3(key)
                wp = self.physics_handles3(s).get(handle) if s is not None else None
                if wp is not None:
                    normal = self._drag_normal(handle)
                    grab = self._ray_plane(ev.position(), normal, wp)
                    self._phys_drag = {"key": key, "handle": handle, "normal": normal, "point": wp.copy(),
                                       "offset": (wp - grab) if grab is not None else np.zeros(3),
                                       "start": QPointF(ev.position()), "moved": False, "last": tuple(wp)}
                    self.physicsSelected.emit(key)
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    ev.accept()
                    return
            if self.show_physics and self._physics:
                self._phys_press = QPointF(ev.position())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        d = self._phys_drag
        if d is not None:
            if (ev.position() - d["start"]).manhattanLength() >= 2 or d["moved"]:
                p = self._ray_plane(ev.position(), d["normal"], d["point"])
                if p is not None:
                    p = p + d["offset"]
                    if not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                        step = self.snap_length3()
                        p = np.round(p / step) * step + 0.0
                    h = d["handle"]
                    if h in ("x0", "x1", "y0", "y1", "z0", "z1"):
                        ax = {"x": 0, "y": 1, "z": 2}[h[0]]
                        q = d["point"].copy()
                        q[ax] = p[ax]
                        p = q
                    target = tuple(float(v) for v in p)
                    if target != tuple(d["last"]):
                        d["moved"], d["last"] = True, target
                        self.physicsMoved.emit(d["key"], h, target)
                        self.update()
            ev.accept()
            return
        if ev.buttons() == Qt.MouseButton.NoButton and self.physics_editable:
            hit = self.physics_hit3(ev.position())
            if hit != self._phys_hover:
                self._phys_hover = hit
                if hit is None:
                    self.unsetCursor()
                    self.setToolTip("")
                else:
                    s = self._spec3(hit[0])
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    name = s.get("name", "") if s else ""
                    self.setToolTip(("Drag to move this wall of the domain" if hit[0] == ("domain",) else
                                     f"{name}: drag to move it in the plane facing you, or select it and use "
                                     "the arrow keys")
                                    + " (snaps; hold Shift for free placement)")
                self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        d = self._phys_drag
        if d is not None:
            self._phys_drag = None
            self.unsetCursor()
            if d["moved"]:
                self.physicsDropped.emit(d["key"], d["handle"], tuple(d["last"]))
            self.update()
            ev.accept()
            return
        press, self._phys_press = self._phys_press, None
        super().mouseReleaseEvent(ev)
        if (press is not None and ev.button() == Qt.MouseButton.LeftButton
                and (ev.position() - press).manhattanLength() <= 3):
            self.physicsSelected.emit(None)

    def _paint_physics3(self, p: QPainter, tok: dict) -> None:
        """The placed items over the render: wire shapes projected on screen (discs as ellipses, boxes
        as their edges), the colours of the 2D layer, handles on the selected or hovered item."""
        hover = self._phys_hover[0] if self._phys_hover else None
        drag = self._phys_drag["key"] if self._phys_drag else None
        specs = sorted(self._physics, key=lambda s: (s["type"] != "domain", bool(s.get("selected"))))
        p.setFont(self._font(7.5))
        fm = QFontMetricsF(p.font())
        n_items = sum(1 for s in specs if s["type"] != "domain")
        # names and wall values are laid out together once everything is drawn, the selected item's
        # first (it keeps its own spot), then the walls, the items and the heated bodies
        first: list[dict] = []
        walls: list[dict] = []
        rest: list[dict] = []
        for s in specs:
            t = s["type"]
            hot = bool(s.get("selected")) or s["key"] in (hover, drag)
            lines, pts = self._phys_lines3(s)
            if t == "domain" and not self._draws_domain_frame and not s.get("explicit"):
                lines = []                         # the case preview draws the planned tunnel itself
            p.save()
            if not s.get("enabled", True):
                p.setOpacity(0.45)
            if t == "heat_source":
                col = heat_mark(tok, -1.0 if float(s.get("power", 0)) < 0 else 1.0)
            else:
                col = qcolor(tok["text2"])
            if hot:
                col = qcolor(tok["accent"])
            style = Qt.PenStyle.DashLine if t == "domain" and not s.get("closed") else Qt.PenStyle.SolidLine
            if t == "domain" and s.get("walls"):
                self._paint_walls3(p, tok, s, walls)
            width = 1.5 if hot else (1.25 if t in ("fan", "heat_source") else 1.0)
            drawn = []
            for ln in lines:
                px = self._project(ln)
                if not np.all(np.isfinite(px)):
                    continue
                drawn.append(px)
                is_arrow = t in ("fan", "port") and len(ln) == 2 and ln is lines[-1]
                if is_arrow:
                    GeometryCanvas._arrow_px(p, col, px[0], px[1], width)
                else:
                    p.setPen(_pen(col, width, style))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPolyline(QPolygonF([QPointF(*q) for q in px]))
            if len(pts):
                px = self._project(pts[:400])
                ok = np.all(np.isfinite(px), axis=1)
                p.setPen(_pen(col, width))
                p.setBrush(qcolor(s.get("colour") or "#f5f5f5"))
                rr = 2.8 if len(pts) <= 40 else 2.0
                for q in px[ok]:
                    p.drawEllipse(QPointF(*q), rr, rr)
                if np.any(ok):
                    drawn.append(px[ok])
            p.restore()
            if t != "domain" and drawn and (hot or n_items <= 8):
                allpx = np.vstack(drawn)
                top = allpx[int(np.argmin(allpx[:, 1]))]
                text = item_label(s)
                if text:
                    req = label_request(fm, top, text, qcolor(tok["accent"] if hot else tok["text2"]), backing=200)
                    (first if hot else rest).append(req)
            if hot and self.physics_editable and (t != "domain" or s.get("explicit") or s.get("closed")):
                for h, wp in self.physics_handles3(s).items():
                    q = self._project(np.asarray([wp]))[0]
                    if not np.all(np.isfinite(q)):
                        continue
                    active = self._phys_hover == (s["key"], h) or (
                        self._phys_drag is not None and self._phys_drag["key"] == s["key"]
                        and self._phys_drag["handle"] == h)
                    p.setPen(_pen(qcolor(tok["accent"])))
                    p.setBrush(qcolor(tok["accent"]) if active else qcolor(tok.get("surface", tok["plot_bg"])))
                    p.drawRect(QRectF(q[0] - 3.5, q[1] - 3.5, 7, 7))
        bodies = self._thermal_body_labels3(tok, fm)
        self.last_label_boxes = paint_labels(p, tok, fm, first + walls + rest + bodies,
                                             (float(self.width()), float(self.height())),
                                             taken=list(getattr(self, "_label_obstacles", ()) or ()))

    def _thermal_body_labels3(self, tok: dict, fm: QFontMetricsF) -> list[dict]:
        """Heated and cooled bodies: a label at the top of the body ("Cabinet · 40 °C") led by a
        swatch in the heater (series blue) colour; the text says the condition, so the colour is
        never the only cue (WCAG 1.4.1). The surface itself is the scene's, not repainted here.
        Returned as :func:`paint_labels` requests, laid out with the placed items' names."""
        out: list[dict] = []
        for s in self._thermal_bodies:
            pts = np.asarray(s.get("points", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
            if not len(pts):
                continue
            px = self._project(pts)
            ok = np.all(np.isfinite(px), axis=1)
            if not np.any(ok):
                continue
            px = px[ok]
            top = np.array([float(px[:, 0].mean()), float(px[:, 1].min())])
            text = item_label(s)
            out.append(label_request(fm, top, text, qcolor(tok["text2"]), mode="above",
                                     swatch=heat_mark(tok, s["sign"])))
        return out

    # -- content ------------------------------------------------------------
    def design(self) -> Design | None:
        return self._design

    def meshes(self) -> list[Mesh3D]:
        return list(self._meshes)

    def bounds(self) -> tuple[float, float, float, float, float, float] | None:
        """Bounding box of the shown meshes (what the dimensions measure), metres."""
        return self._bounds

    def set_show_dimensions(self, on: bool) -> None:
        self.show_dimensions = bool(on)
        self._apply_focus()
        self.update()

    def _apply_focus(self) -> None:
        """Frame the bounding box plus room for its dimensions (or the model tightly
        when they are off): a tight fit put the dimension lines outside the view."""
        b = self._bounds
        if b is None or not self.show_dimensions:
            self.scene.set_focus(self._physics_focus(b) if self._physics and self.show_physics else None)
        else:
            pad = self.DIMENSION_PAD * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
            self.scene.set_focus(self._physics_focus((b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad, b[4] - pad,
                                                      b[5] + pad)))
        self.refresh()

    def set_design(self, design: Design | None, *, meshes: list[Mesh3D] | None = None,
                   ground: bool | None = None, fit: bool | None = None) -> None:
        """Show ``design`` (``meshes`` overrides its own surfaces, e.g. an import preview)."""
        same = design is not None and self._design is not None and design.id == self._design.id
        shown = list(meshes) if meshes is not None else design_meshes3d(design)
        road = stands_on_road(design) if ground is None else bool(ground)
        sig = (tuple((id(m.vertices), id(m.faces), m.role, len(m.faces)) for m in shown), road)
        self._design = design
        self._mref = tuple(design.moment_ref3) if design is not None and shown else None
        if sig != self._sig or not same:
            self._meshes = shown
            self._sig = sig
            self._bounds = meshes_bounds(shown)
            self._rebuild(road)
            self._apply_focus()
            if fit or (fit is None and not same):
                self.set_view("iso")
                return
        elif fit:
            self.set_view("iso")
            return
        self.refresh()
        self.update()

    def _rebuild(self, road: bool) -> None:
        sc = self.scene
        sc.clear()
        tok = theme_tokens()
        for m in self._meshes:
            sc.add_mesh(m, color=wheel_colour(tok) if m.role == "wheel" else None)
        if road and self._meshes:
            sc.add_ground(0.0)

    def _on_theme(self, *args) -> None:
        super()._on_theme(*args)
        if self._meshes:                       # the wheel grey is derived from the tokens
            self._rebuild(bool(self._sig[1]) if self._sig else False)
            self.refresh()

    # -- overlays -----------------------------------------------------------
    def _project(self, pts: np.ndarray) -> np.ndarray:
        """World points -> widget pixels (the scene projects in device pixels)."""
        return np.asarray(self.scene.project(np.asarray(pts, dtype=float)), dtype=float) / self._dpr()

    def _overlays_possible(self) -> bool:
        return bool(getattr(self.scene, "available", False)) and bool(self.scene.layers())

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        ring = self.focus_visible and self.hasFocus()
        if not (self._overlays_possible() or self._drop_active or ring):
            return
        tok = theme_tokens()
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            if self._overlays_possible():
                self._paint_overlays(p, tok)
            if self._drop_active:
                self._paint_drop(p, tok)
            if ring:
                p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                paint_viewport_focus(p, QRectF(self.rect()), tok)
        except Exception:  # noqa: BLE001 - an annotation must never take the viewport down
            _log.debug("3D overlay painting failed", exc_info=True)
        finally:
            p.end()

    def _paint_overlays(self, p: QPainter, tok: dict) -> None:
        if self.show_dimensions and self._bounds is not None:
            self._paint_dimensions(p, tok, self._bounds)
        if self.show_reference and self._mref is not None:
            self._paint_reference(p, tok, self._mref)
        self.last_label_boxes = []
        if self.show_physics and (self._physics or self._thermal_bodies):
            self._paint_physics3(p, tok)

    def _font(self, size: float = 8.0) -> QFont:
        f = QFont(self.font())
        f.setPointSizeF(size)
        return f

    def _paint_dimensions(self, p: QPainter, tok: dict, bounds) -> None:
        """Length, width and height on the silhouette edges, CAD style (see
        :func:`pick_dimension_edges`): extension lines, arrowheads, the value
        centred on a gap in the line."""
        p.setFont(self._font(8.0))
        fm = QFontMetricsF(p.font())
        col = qcolor(tok["text2"])
        surface = qcolor(tok.get("surface", tok["plot_bg"]), 235)
        for e in pick_dimension_edges(bounds, self._project):
            nx, ny = e["normal"]
            gap, over = 18.0, 5.0
            a = QPointF(e["a"][0] + gap * nx, e["a"][1] + gap * ny)
            b = QPointF(e["b"][0] + gap * nx, e["b"][1] + gap * ny)
            # the measured edge of the bounding box, faint and dotted: a box corner is empty
            # space (most of all beside a sphere), and extension lines starting from nothing
            # read as stray ticks rather than as a dimension of the box
            p.setPen(_pen(qcolor(tok["muted"], 110), 1.0, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(*e["a"]), QPointF(*e["b"]))
            p.setPen(_pen(qcolor(tok["muted"], 160)))
            for src, dst in ((e["a"], a), (e["b"], b)):
                p.drawLine(QPointF(src[0] + 3 * nx, src[1] + 3 * ny),
                           QPointF(dst.x() + over * nx, dst.y() + over * ny))
            p.setPen(_pen(col))
            p.drawLine(a, b)
            dx, dy = b.x() - a.x(), b.y() - a.y()
            n = math.hypot(dx, dy)
            ux, uy = dx / n, dy / n
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            for tip, sgn in ((a, 1), (b, -1)):
                back = QPointF(tip.x() + sgn * 7 * ux, tip.y() + sgn * 7 * uy)
                p.drawPolygon(QPolygonF([tip, QPointF(back.x() - 2.2 * uy, back.y() + 2.2 * ux),
                                         QPointF(back.x() + 2.2 * uy, back.y() - 2.2 * ux)]))
            text = f"{fmt_len(e['length'])} m"
            tw, th = fm.horizontalAdvance(text) + 8, fm.height() + 2
            cx, cy = (a.x() + b.x()) / 2, (a.y() + b.y()) / 2
            if tw > n - 20:              # no room on the line: the value goes beside it
                cx, cy = cx + nx * (tw / 2 + 4) * abs(nx) + nx * 4, cy + ny * (th / 2 + 4)
            box = QRectF(cx - tw / 2, cy - th / 2, tw, th)
            p.setBrush(surface)
            p.drawRect(box)
            p.setPen(col)
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_reference(self, p: QPainter, tok: dict, mref) -> None:
        """The moment reference point: the centre-of-gravity symbol, as in 2D."""
        c = self._project(np.asarray([mref], dtype=float))[0]
        if not np.all(np.isfinite(c)):
            return
        r = 4.5
        col = qcolor(tok["text"])
        rect = QRectF(c[0] - r, c[1] - r, 2 * r, 2 * r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(tok.get("surface", tok["plot_bg"])))
        p.drawEllipse(rect)
        p.setBrush(col)
        p.drawPie(rect, 0, 90 * 16)
        p.drawPie(rect, 180 * 16, 90 * 16)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(_pen(col))
        p.drawEllipse(rect)

    def _paint_drop(self, p: QPainter, tok: dict) -> None:
        r = QRectF(self.rect()).adjusted(4.5, 4.5, -4.5, -4.5)
        p.setBrush(qcolor(tok["accent"], 22))
        p.setPen(_pen(qcolor(tok["accent"]), 1.0, Qt.PenStyle.DashLine))
        p.drawRect(r)
        p.setFont(self._font(9.0))
        p.setPen(qcolor(tok["accent"]))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Drop to import")

    # -- drag and drop --------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        md = event.mimeData()
        if self.acceptDrops() and md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            self._drop_active = True
            event.acceptProposedAction()
            self.update()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._drop_active = False
        self.update()

    def dropEvent(self, event) -> None:
        self._drop_active = False
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.update()
        if paths:
            event.acceptProposedAction()
            self.filesDropped.emit(paths)


def vec3(v) -> np.ndarray:
    """``v`` as a float 3-vector (a 2D point gets z = 0)."""
    out = np.zeros(3)
    a = np.asarray(v, dtype=float).ravel()[:3]
    out[: len(a)] = a
    return out


def _basis3(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to ``n`` (u horizontal where possible)."""
    n = np.asarray(n, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return u, np.cross(n, u)


def _circle3(c: np.ndarray, n: np.ndarray, r: float, k: int = 40) -> np.ndarray:
    """(k + 1, 3) closed polyline of a circle of radius ``r`` about ``c`` in the plane normal to ``n``."""
    u, v = _basis3(n)
    t = np.linspace(0.0, 2.0 * np.pi, k + 1)
    return np.asarray(c, dtype=float) + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)


def box_edges(b) -> list[tuple[tuple, tuple]]:
    """The 12 edges of an axis-aligned box (x0, x1, y0, y1, z0, z1) as point pairs."""
    x0, x1, y0, y1, z0, z1 = b
    c = [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(c[i], c[j]) for i, j in pairs]


def _point_polygon_distance(p: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance [px] from point ``p`` to a convex polygon (negative inside).

    Inside means on the same side of every edge, whichever way the polygon
    is wound (screen y points down, so "counter-clockwise" is ambiguous)."""
    d = math.inf
    sides: set[bool] = set()
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
        d = min(d, float(np.hypot(*(p - (a + t * ab)))))
        c = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
        if abs(c) > 1e-9:
            sides.add(bool(c > 0))
    return -d if len(sides) == 1 else d


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """2D convex hull of points (Andrew's monotone chain)."""
    p = sorted({(float(x), float(y)) for x, y in pts})
    if len(p) < 3:
        return np.asarray(p, dtype=float)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def arrow_length(bounds) -> float:
    """Length [m] of the free-stream arrow for a body box: half its length, at least a
    quarter of its largest size (a sphere's arrow is not a stub)."""
    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds)
    return max(0.5 * (x1 - x0), 0.25 * max(x1 - x0, y1 - y0, z1 - z0, 1e-6))


def arrow_label_centre(a, b, size: tuple[float, float], gap: float = 5.0, *, side: int = 1) -> np.ndarray:
    """Screen centre [px] of a label of ``size`` (width, height) beside the arrow a -> b.

    The label sits off the arrow's middle on the side facing up the screen (left for
    a vertical arrow; ``side=-1``: the other side), pushed out by its own
    half-extent along that normal plus ``gap``, so its box never covers the shaft
    or the head whatever the arrow's slope. (A label simply centred some pixels
    above the middle covers half of a steep arrow - the iso view of a wing put the
    arrow head under its backing.)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    n = float(np.hypot(*d))
    t = d / n if n > 0 else np.array([1.0, 0.0])
    nrm = np.array([t[1], -t[0]])
    if nrm[1] > 1e-9 or (abs(nrm[1]) <= 1e-9 and nrm[0] > 0):
        nrm = -nrm
    if side < 0:
        nrm = -nrm
    w, h = float(size[0]), float(size[1])
    reach = 0.5 * w * abs(nrm[0]) + 0.5 * h * abs(nrm[1])
    return 0.5 * (a + b) + nrm * (reach + gap)


def _smoke_pads(sm: dict) -> tuple[float, float]:
    """Half the nozzle pitch across and up (for frames drawn around the nozzles); a line
    of nozzles uses its own pitch for both."""
    sy, sz = sm.get("spacing_yz", (sm.get("spacing", 0.0), sm.get("spacing", 0.0)))
    sy = float(sy) or float(sz)
    sz = float(sz) or sy
    return 0.5 * sy, 0.5 * sz


def rake_bar(sm: dict) -> np.ndarray:
    """World end points (2, 3) of the bar a rake's nozzles hang from, as the Setup
    preview draws it: a little upstream of the nozzles (1.2 pitches, at most half the
    gap to the body) and half a pitch longer than the row at each end."""
    py, pz = _smoke_pads(sm)
    x = float(sm["x"])
    (yl, yh), (zl, zh) = sm["y"], sm["z"]
    kind = sm["kind"]
    pitch = 2.0 * (py if kind == "rake" else pz)
    gap = max(float(sm.get("body_x0", x)) - x, 0.0)
    back = x - min(1.2 * pitch, 0.5 * gap) if gap > 0 else x - 1.2 * pitch
    bar = ([[back, yl - py, zl], [back, yh + py, zl]] if kind != "vertical"
           else [[back, yl, zl - pz], [back, yl, zh + pz]])
    return np.asarray(bar, dtype=float)


def _rect_distance(points: np.ndarray, centre: np.ndarray, w: float, h: float) -> np.ndarray:
    """Signed distance [px] of screen points from an axis-aligned box (negative inside)."""
    dx = np.abs(points[:, 0] - centre[0]) - 0.5 * w
    dy = np.abs(points[:, 1] - centre[1]) - 0.5 * h
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return np.where((dx < 0) & (dy < 0), np.maximum(dx, dy), outside)


def flow_arrow_placement(bounds, project: Callable[[np.ndarray], np.ndarray], **kw) -> tuple | None:
    """World end points (tail, head) of the free-stream arrow (:func:`flow_arrow_layout`
    without the label side), or None when nothing projects."""
    lay = flow_arrow_layout(bounds, project, **kw)
    return (lay[0], lay[1]) if lay is not None else None


def flow_arrow_layout(bounds, project: Callable[[np.ndarray], np.ndarray], *,
                      avoid: np.ndarray | None = None, road: bool = False,
                      avoid_px: np.ndarray | None = None, sheets: list | None = None,
                      view: tuple[float, float] | None = None,
                      label: tuple[float, float] | None = None,
                      clear_px: float = 14.0) -> tuple[np.ndarray, np.ndarray, int] | None:
    """The free-stream arrow, placed clear of the body: (tail, head, label side).

    The arrow points along +x upstream of the body. Its natural place is in
    line with the body's centre, but from the iso camera a slender body (a
    wing, whose span runs along the same screen diagonal as the flow) lies
    across that line, and a smoke rake (``avoid``, world points) stands
    there too. So placements are tried in order - in line with the centre;
    in line but ending upstream of the rake; above; beside either side;
    below (never through a road) - and the first whose projection stays
    ``clear_px`` from the body's projected box and from the ``avoid`` points,
    and lies inside the ``view`` (width, height in px) with a margin, wins.
    ``sheets`` are further world point sets (a smoke grid's frame, a plane's
    band) whose projected outline is kept clear like the body's.
    ``label`` (width, height in px) is the speed label drawn beside the arrow's
    middle (:func:`arrow_label_centre`), on whichever side has more room (1:
    the upper side, -1: the lower); it must keep clear too, also of
    ``avoid_px`` (screen points, e.g. the corners of the smoke label).
    Otherwise the placement with the most room is used. Returns None when
    nothing projects.
    """
    x0, x1, y0, y1, z0, z1 = (float(v) for v in bounds)
    size = max(x1 - x0, y1 - y0, z1 - z0, 1e-6)
    yc, zc = 0.5 * (y0 + y1), 0.5 * (z0 + z1)
    span = arrow_length(bounds)
    head = x0 - 0.12 * size
    ranges = [(head - span, head)]
    av_world = np.asarray(avoid, dtype=float).reshape(-1, 3) if avoid is not None and len(avoid) else None
    cands = [(ranges[0], yc, zc)]
    if av_world is not None:
        # ends just before the rake (full or shorter), then a little further off it: a small
        # body's tenth is only a few pixels, and a head on the rake bar reads as touching it
        for gap in (0.1, 0.25):
            h2 = float(av_world[:, 0].min()) - gap * size
            cands += [((h2 - span, h2), yc, zc), ((h2 - 0.6 * span, h2), yc, zc)]
    cands += [(ranges[0], yc, z1 + 0.15 * size), (ranges[0], y0 - 0.2 * size, zc),
              (ranges[0], y1 + 0.2 * size, zc)]
    if not road:
        cands.append((ranges[0], yc, z0 - 0.15 * size))
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)
    pc = np.asarray(project(corners), dtype=float)
    if pc.shape != (8, 2) or not np.all(np.isfinite(pc)):
        return None
    hulls = [_convex_hull(pc)]
    for sh in sheets or ():
        ps = np.asarray(project(np.asarray(sh, dtype=float).reshape(-1, 3)), dtype=float)
        if ps.ndim == 2 and len(ps) >= 3 and np.all(np.isfinite(ps)):
            hulls.append(_convex_hull(ps))
    av = None
    if av_world is not None:
        av = np.asarray(project(av_world), dtype=float)
        av = av[np.all(np.isfinite(av), axis=1)] if av.ndim == 2 else None
    if avoid_px is not None and len(avoid_px):
        extra = np.asarray(avoid_px, dtype=float).reshape(-1, 2)
        av = extra if av is None else np.vstack([av, extra])
    def clearance(samples: np.ndarray) -> float:
        room = min((_point_polygon_distance(q, h) for h in hulls if len(h) >= 3 for q in samples),
                   default=math.inf)
        if av is not None and len(av):
            d = np.hypot(samples[:, None, 0] - av[None, :, 0], samples[:, None, 1] - av[None, :, 1])
            room = min(room, float(d.min()) - 4.0)
        return room

    def label_room(pe: np.ndarray, side: int) -> float:
        lw, lh = label
        c = arrow_label_centre(pe[0], pe[1], label, side=side)
        # the box's outline every ~12 px: a rake bar crossing it lands near one of them
        fx = np.linspace(-0.5, 0.5, max(3, int(lw // 12) + 1))
        fy = np.array([-0.5, 0.0, 0.5])
        box = np.array([(c[0] + a * lw, c[1] + b * lh) for a in fx for b in fy])
        room = clearance(box)
        if av is not None and len(av):                       # an emitter point inside the box
            room = min(room, float(_rect_distance(av, c, lw, lh).min()) - 4.0)
        if view is not None:
            m = 6.0
            out = max(float(np.max(m - box[:, 0])), float(np.max(box[:, 0] - (view[0] - m))),
                      float(np.max(m - box[:, 1])), float(np.max(box[:, 1] - (view[1] - m))), 0.0)
            room -= 4.0 * out
        return room

    best, best_room = None, -math.inf
    for (tx, hx), y, z in cands:
        ends = np.array([[tx, y, z], [hx, y, z]], dtype=float)
        pe = np.asarray(project(ends), dtype=float)
        if pe.shape != (2, 2) or not np.all(np.isfinite(pe)) or float(np.hypot(*(pe[1] - pe[0]))) < 24:
            continue
        room = clearance(pe[0] + np.linspace(0.0, 1.0, 9)[:, None] * (pe[1] - pe[0]))
        if view is not None:
            m = 12.0
            out = max(float(np.max(m - pe[:, 0])), float(np.max(pe[:, 0] - (view[0] - m))),
                      float(np.max(m + 16 - pe[:, 1])), float(np.max(pe[:, 1] - (view[1] - m))), 0.0)
            room -= 4.0 * out                             # off-screen is worse than crowded
        side = 1
        if label is not None:
            up, down = label_room(pe, 1), label_room(pe, -1)
            # the upper side reads best (a caption above its arrow): it wins unless crowded
            side = 1 if up >= min(clear_px, down) else -1
            room = min(room, up if side == 1 else down)
        if room >= clear_px:
            return ends[0], ends[1], side
        if room > best_room:
            best, best_room = (ends[0], ends[1], side), room
    return best


class CasePreview3D(GeometryView3D):
    """The Setup page's 3D case: the body as the solver sees it, in its tunnel.

    ``set_case`` takes the wind-frame meshes of the plan (rotated by pitch and
    yaw about the moment point, ``SimPlan3D.bodies_wind3``), the full domain
    ``extent3`` (a dashed box; with a road, the road is the tunnel floor), whether
    a symmetry plane is used (its outline at y = 0 in the accent colour: only the
    y >= 0 half is solved), the free stream as an arrow along +x placed clear of
    the body, the rake, the smoke label and the axes triad, with the speed label
    beside it (:func:`flow_arrow_layout`), and the smoke emitters of a 3D run
    (``smoke``, a :func:`smoke_rake_layout` dict: nozzles as small grey dots on
    their rake, or the inlet band of a plane; greyscale like the smoke itself).
    ``fit_domain`` frames the whole tunnel instead of the body and its
    surroundings. Dimensions are off here: the Geometry page shows them.

    The smoke emitter is an object, not decoration: hovering it shows a pointing
    hand, and a click on it (its nozzles, bar, frame, band or label, without
    dragging) emits :attr:`smokeClicked`, which the Setup page answers by bringing
    its Flow visualisation settings into view. A press that turns into a drag
    still orbits the camera.
    """

    #: emitted on a click (press and release within :attr:`CLICK_SLOP_PX`) on the smoke emitter
    smokeClicked = Signal()

    #: room around the body when framing it, as a fraction of its largest size (the
    #: preview is a wide, short strip: more context made the body a small speck)
    BODY_CONTEXT = 0.12
    #: how far [px] from a nozzle or a drawn line a click still counts as on the emitter
    #: (the dots are 5 px across; a smaller target is hard to hit on a laptop pad)
    HIT_PX = 6.0
    #: largest press-to-release travel [px] that is a click rather than an orbit
    CLICK_SLOP_PX = 4.0

    def __init__(self, parent: QWidget | None = None, *, scene=None) -> None:
        super().__init__(parent, scene=scene)
        self.setAcceptDrops(False)
        self.show_dimensions = False
        self.show_reference = False
        self.fit_domain = False
        self._extent3: tuple | None = None
        self._symmetry = False
        self._ground_mode: str | None = None
        self._speed_text = ""
        self._attitude_text = ""
        self._smoke: dict | None = None
        #: where the emitter was last painted, in logical px (see :meth:`smoke_hit`), or None
        self._smoke_px: dict | None = None
        self._smoke_press: QPointF | None = None
        self._smoke_hover = False
        self._inflow = True
        #: the free-stream arrow last painted: (tail, head, label side) in world metres, or None
        self.last_arrow: tuple | None = None
        self.set_placeholder("No geometry", "")

    def extent3(self) -> tuple | None:
        return self._extent3

    def symmetry(self) -> bool:
        return self._symmetry

    def smoke(self) -> dict | None:
        """The smoke emitters shown (a :func:`smoke_rake_layout` dict), or None."""
        return self._smoke

    def set_case(self, design: Design | None, *, meshes: list[Mesh3D] | None = None, extent3=None,
                 symmetry: bool = False, ground: str | None = None, speed_text: str = "",
                 attitude_text: str = "", smoke: dict | None = None, fit: bool | None = None,
                 inflow: bool = True) -> None:
        """Show a case (see the class docstring); ``ground`` is None, "fixed" or "moving";
        ``inflow`` False (a closed box) leaves out the free-stream arrow."""
        self._inflow = bool(inflow)
        ext = tuple(float(v) for v in extent3) if extent3 is not None else None
        if ext != self._extent3 or bool(symmetry) != self._symmetry or ground != self._ground_mode:
            self._sig = None                   # the tunnel layers change: rebuild them
        self._extent3, self._symmetry, self._ground_mode = ext, bool(symmetry), ground
        self._draws_domain_frame = ext is None             # planned: the dashed tunnel is the domain
        self._speed_text, self._attitude_text = speed_text, attitude_text
        # a rake that could not be resolved carries its reason, not a position: the page says why
        self._smoke = smoke if smoke is not None and "error" not in smoke and "x" in smoke else None
        if self._smoke is None:
            self._smoke_px = None              # nothing left to click until the next paint
        super().set_design(design, meshes=meshes, ground=ground is not None, fit=fit)
        self._apply_focus()
        self.update()                          # the overlays changed even when the scene did not

    def set_fit_domain(self, on: bool) -> None:
        self.fit_domain = bool(on)
        self._apply_focus()
        self.fit()

    # -- the smoke emitter as a clickable object ---------------------------------------
    def smoke_hit(self, pos: QPointF) -> bool:
        """Whether ``pos`` (logical px) is on the smoke emitter as last painted: within
        :attr:`HIT_PX` of a nozzle or of a drawn line (bar, tubes, frame), inside a
        plane's band, or on the label."""
        hp = self._smoke_px
        if hp is None:
            return False
        x, y = float(pos.x()), float(pos.y())
        lab = hp.get("label")
        if lab is not None and lab.adjusted(-2, -2, 2, 2).contains(QPointF(x, y)):
            return True
        band = hp.get("band")
        if band is not None and band.containsPoint(QPointF(x, y), Qt.FillRule.OddEvenFill):
            return True
        q = np.array([x, y])
        pts = hp.get("points")
        if pts is not None and len(pts) and np.min(np.hypot(*(pts - q).T)) <= self.HIT_PX:
            return True
        segs = hp.get("segments")
        if segs is not None and len(segs):
            a, b = segs[:, 0], segs[:, 1]
            ab = b - a
            t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
            d = np.hypot(*(a + t[:, None] * ab - q).T)
            if np.min(d) <= self.HIT_PX:
                return True
        return False

    def mousePressEvent(self, ev) -> None:
        on = ev.button() == Qt.MouseButton.LeftButton and self.smoke_hit(ev.position())
        self._smoke_press = QPointF(ev.position()) if on else None
        super().mousePressEvent(ev)            # a press that becomes a drag still orbits

    def mouseReleaseEvent(self, ev) -> None:
        press, self._smoke_press = self._smoke_press, None
        super().mouseReleaseEvent(ev)          # resets the drag cursor ...
        if self._smoke_hover:                  # ... which over the emitter is the pointing hand
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        if press is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        moved = ev.position() - press
        if abs(moved.x()) + abs(moved.y()) <= self.CLICK_SLOP_PX and self.smoke_hit(ev.position()):
            self.smokeClicked.emit()

    def mouseMoveEvent(self, ev) -> None:
        if ev.buttons() == Qt.MouseButton.NoButton:
            hover = self.smoke_hit(ev.position())
            if hover != self._smoke_hover:
                self._smoke_hover = hover
                if hover:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    self.setToolTip("Smoke emitter: click to edit it under Flow visualisation")
                else:
                    self.unsetCursor()
                    self.setToolTip("")
        super().mouseMoveEvent(ev)

    def leaveEvent(self, ev) -> None:
        if self._smoke_hover:
            self._smoke_hover = False
            self.unsetCursor()
            self.setToolTip("")
        super().leaveEvent(ev)

    def _apply_focus(self) -> None:
        if self.fit_domain and self._extent3 is not None:
            self.scene.set_focus(self._extent3)
        elif self._bounds is not None:
            x0, x1, y0, y1, z0, z1 = self._bounds
            pad = self.BODY_CONTEXT * max(x1 - x0, y1 - y0, z1 - z0)
            sm = self._smoke
            if sm is not None:
                # frame the emitters too: they stand upstream, and a rake cut off at the
                # edge of the view reads as a drawing error
                x0 = min(x0, float(sm["x"]))
                y0, y1 = min(y0, sm["y"][0]), max(y1, sm["y"][1])
                z0, z1 = min(z0, sm["z"][0]), max(z1, sm["z"][1])
            zlo = min(z0, 0.0) if self._ground_mode is not None else z0 - pad
            self.scene.set_focus(self._physics_focus((x0 - pad, x1 + pad, y0 - pad, y1 + pad, zlo, z1 + pad)))
        else:
            self.scene.set_focus(self._physics_focus(None))
        self.refresh()

    def _rebuild(self, road: bool) -> None:
        sc = self.scene
        sc.clear()
        tok = theme_tokens()
        for m in self._meshes:
            sc.add_mesh(m, color=wheel_colour(tok) if m.role == "wheel" else None)
        if not self._meshes:
            return
        ext = self._extent3
        if road:
            sc.add_ground(0.0, extent=(ext[0], ext[1], ext[2], ext[3]) if ext is not None else None)

    def _paint_overlays(self, p: QPainter, tok: dict) -> None:
        p.setFont(self._font(8.0))
        fm = QFontMetricsF(p.font())
        col = qcolor(tok["text2"])
        if self._extent3 is not None:
            p.setPen(_pen(qcolor(tok["text2"], 170), 1.0, Qt.PenStyle.DashLine))
            for a, b in box_edges(self._extent3):
                pa, pb = self._project(np.array([a, b], dtype=float))
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    p.drawLine(QPointF(*pa), QPointF(*pb))
            if self._symmetry:
                # the symmetry plane: an outline only - a translucent sheet tinted the whole view
                x0, x1, _, _, z0, z1 = self._extent3
                rect = self._project(np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1], [x0, 0, z0]],
                                              dtype=float))
                if np.all(np.isfinite(rect)):
                    p.setPen(_pen(qcolor(tok["accent"], 200), 1.0, Qt.PenStyle.DashDotLine))
                    p.drawPolyline(QPolygonF([QPointF(*q) for q in rect]))
        sm = self._smoke
        smoke_label = None
        self._smoke_px = None                  # set again by _paint_smoke when it draws
        if sm is not None and self._bounds is not None:
            smoke_label = self._paint_smoke(p, tok, sm)
        # the corner notes (drawn last, on top) and the smoke label are taken: device labels keep clear
        lines = [t for t in (self._attitude_text,) if t]
        if self._extent3 is not None:
            x0, x1, y0, y1, z0, z1 = self._extent3
            lines.append(f"Domain {fmt_len(x1 - x0)} × {fmt_len(y1 - y0)} × {fmt_len(z1 - z0)} m")
            if self._symmetry:
                lines.append("Half domain: symmetry plane y = 0")
        corner = [(self.width() - fm.horizontalAdvance(t) - 13, 16 + i * (fm.height() + 2) - fm.ascent() - 1,
                   fm.horizontalAdvance(t) + 6, fm.height() + 2) for i, t in enumerate(lines)]
        self.last_label_boxes = []
        self._label_obstacles = corner + ([(smoke_label.x(), smoke_label.y(), smoke_label.width(),
                                            smoke_label.height())] if smoke_label is not None else [])
        if self.show_physics and (self._physics or self._thermal_bodies):
            self._paint_physics3(p, tok)
            p.setFont(self._font(8.0))
        # the free stream: an arrow upstream of the body along +x, clear of the body and the rake
        self.last_arrow = None
        if self._bounds is not None and self._inflow:
            avoid, sheets = None, None
            placed = [vec3(h) for s in (self._physics if self.show_physics else ()) if s["type"] != "domain"
                      for h in s.get("handles", {}).values()]
            if sm is not None:
                # what the arrow must not cross: the nozzles, and a grid's frame or a plane's
                # whole band (an arrow through a sheet reads as if it started behind it)
                yl, yh = sm["y"]
                zl, zh = sm["z"]
                py, pz = _smoke_pads(sm)
                pts = np.asarray(sm["points"], dtype=float)
                avoid = pts[:: max(1, len(pts) // 64)] if len(pts) else None
                if sm["kind"] in ("rake", "vertical") and avoid is not None:
                    # the bar the nozzles hang from is drawn too, beyond the end nozzles
                    bar = rake_bar(sm)
                    avoid = np.vstack([avoid, bar[0] + np.linspace(0.0, 1.0, 17)[:, None] * (bar[1] - bar[0])])
                if sm["kind"] in ("grid", "plane"):
                    sheets = [np.array([[sm["x"], y, z] for y in (yl - py, yh + py) for z in (zl - pz, zh + pz)])]
                    if avoid is None:            # a plane: its band still counts as the emitter
                        avoid = sheets[0]
            if placed:                             # nor the placed items (an emitter rake stands there)
                avoid = np.asarray(placed, dtype=float) if avoid is None else np.vstack([avoid, placed])
            lab = (fm.horizontalAdvance(self._speed_text) + 6, fm.height() + 2) if self._speed_text else None
            box_px = self._triad_px()
            # the smoke label's and the device labels' outlines, sampled every ~12 px
            rects = [smoke_label] if smoke_label is not None else []
            rects += [QRectF(*b) for b in (self.last_label_boxes if self.show_physics else ())]
            for r in rects:
                xs = np.linspace(r.left(), r.right(), max(2, int(r.width() // 12) + 1))
                box_px = np.vstack([box_px, [(x, y) for x in xs for y in (r.top(), r.center().y(), r.bottom())]])
            lay = flow_arrow_layout(self._bounds, self._project, avoid=avoid, avoid_px=box_px, sheets=sheets,
                                    road=self._ground_mode is not None,
                                    view=(float(self.width()), float(self.height())), label=lab)
            self.last_arrow = lay
            if lay is not None:
                a, b = self._project(np.asarray(lay[:2]))
                self._arrow3(p, col, QPointF(*a), QPointF(*b))
                if self._speed_text:           # beside the arrow's middle, never over its shaft or head
                    tw = fm.horizontalAdvance(self._speed_text)
                    c = arrow_label_centre(a, b, lab, side=lay[2])
                    # _label's backing spans ascent + 1 above the baseline and height + 2 in all
                    self._label(p, tok, QPointF(c[0] - tw / 2, c[1] + fm.ascent() + 1 - 0.5 * lab[1]),
                                self._speed_text)
        for i, text in enumerate(lines):
            self._label(p, tok, QPointF(self.width() - fm.horizontalAdvance(text) - 10,
                                        16 + i * (fm.height() + 2)), text)

    def _triad_px(self) -> np.ndarray:
        """Points (logical px) covering the axes triad the scene draws in the bottom-left
        corner (``viz3d.overlay``: centred ``TRIAD_INSET`` px in, 30 px axes plus their
        letters): the free-stream arrow and its label keep clear of it as of any label."""
        try:
            from ...viz3d.overlay import TRIAD_INSET as inset
        except ImportError:                       # pragma: no cover - the overlay module is part of viz3d
            inset = 48.0
        reach = 30.0 + 16.0
        h = float(self.height())
        xs = np.linspace(max(inset - reach, 0.0), inset + reach, 8)
        ys = np.linspace(h - inset - reach, min(h - inset + reach, h), 8)
        return np.array([(x, y) for x in xs for y in ys], dtype=float)

    def _paint_smoke(self, p: QPainter, tok: dict, sm: dict) -> QRectF | None:
        """The smoke emitters, greyscale like the smoke the Simulate and Results pages draw:
        a rake (across the flow) or a vertical rake (up the height) as the real thing, a bar
        with a short tube per nozzle pointing downstream, which also shows which way the
        smoke leaves; a grid as its nozzles in a thin frame; a plane's band as a light grey
        sheet with a dashed edge. Returns the label's box (the free-stream arrow keeps
        clear of it) and records what was drawn, in px, for :meth:`smoke_hit`."""
        dot = qcolor(tok["text2"])
        frame = qcolor(tok["muted"], 170)
        x = float(sm["x"])
        (yl, yh), (zl, zh) = sm["y"], sm["z"]
        kind = sm["kind"]
        hit: dict = {"points": None, "segments": [], "band": None, "label": None}
        if kind == "plane":
            quad = self._project(np.array([[x, yl, zl], [x, yh, zl], [x, yh, zh], [x, yl, zh]], dtype=float))
            if not np.all(np.isfinite(quad)):
                return None
            p.setPen(_pen(frame, 1.0, Qt.PenStyle.DashLine))
            p.setBrush(qcolor(tok["text"], 22))
            hit["band"] = QPolygonF([QPointF(*q) for q in quad])
            p.drawPolygon(hit["band"])
            # a band seen edge-on is a line: its edges stay clickable
            hit["segments"] = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
            top = quad[np.argmin(quad[:, 1])]
            text = "Smoke plane"
        else:
            pts = np.asarray(sm["points"], dtype=float)[:MAX_NOZZLES]
            px = self._project(pts) if len(pts) else np.zeros((0, 2))
            ok = np.all(np.isfinite(px), axis=1) if len(px) else np.zeros(0, bool)
            if not np.any(ok):
                return None
            py, pz = _smoke_pads(sm)
            p.setPen(_pen(frame, 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            if kind in ("rake", "vertical"):
                bar = rake_bar(sm)
                back = float(bar[0, 0])
                ends = self._project(bar)
                tubes = self._project(np.column_stack([np.full(len(pts), back), pts[:, 1], pts[:, 2]]))
                for a, b in zip(tubes, px, strict=True):
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                        p.drawLine(QPointF(*a), QPointF(*b))
                        hit["segments"].append((a, b))
            else:
                ends = self._project(np.array([[x, yl - py, zl - pz], [x, yh + py, zl - pz], [x, yh + py, zh + pz],
                                               [x, yl - py, zh + pz], [x, yl - py, zl - pz]], dtype=float))
            if np.all(np.isfinite(ends)):
                p.drawPolyline(QPolygonF([QPointF(*q) for q in ends]))
                hit["segments"].extend(itertools.pairwise(ends))
            r = 2.6 if len(pts) <= 60 else 2.0
            p.setPen(_pen(qcolor(tok.get("surface", tok["plot_bg"])), 1.0))
            p.setBrush(dot)
            for q in px[ok]:
                p.drawEllipse(QPointF(*q), r, r)
            vis = px[ok]
            hit["points"] = np.asarray(vis, dtype=float)
            top = vis[np.argmin(vis[:, 1])]
            n = int(sm.get("count", len(pts)))
            noun = "vertical rake" if kind == "vertical" else kind
            text = f"Smoke {noun} · {n:,} nozzle{'s' if n != 1 else ''}"
        fm = QFontMetricsF(p.font())
        box = self._label(p, tok, QPointF(top[0] - fm.horizontalAdvance(text) / 2, top[1] - 9), text)
        segs = [s for s in hit["segments"] if np.all(np.isfinite(s[0])) and np.all(np.isfinite(s[1]))]
        hit["segments"] = np.array(segs, dtype=float).reshape(-1, 2, 2)
        hit["label"] = box
        self._smoke_px = hit
        return box

    @staticmethod
    def _label(p: QPainter, tok: dict, at: QPointF, text: str) -> QRectF:
        """Overlay text on a surface-coloured backing, readable over the floor grid and the
        tunnel lines; returns the backing's box."""
        fm = QFontMetricsF(p.font())
        box = QRectF(at.x() - 3, at.y() - fm.ascent() - 1, fm.horizontalAdvance(text) + 6, fm.height() + 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcolor(tok.get("surface", tok["plot_bg"]), 215))
        p.drawRect(box)
        p.setPen(qcolor(tok["text2"]))
        p.drawText(at, text)
        return box

    @staticmethod
    def _arrow3(p: QPainter, col: QColor, a: QPointF, b: QPointF) -> None:
        dx, dy = b.x() - a.x(), b.y() - a.y()
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        p.setPen(_pen(col, 1.25))
        p.drawLine(a, QPointF(b.x() - 6 * ux, b.y() - 6 * uy))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(col)
        back = QPointF(b.x() - 8 * ux, b.y() - 8 * uy)
        p.drawPolygon(QPolygonF([b, QPointF(back.x() - 3 * uy, back.y() + 3 * ux),
                                 QPointF(back.x() + 3 * uy, back.y() - 3 * ux)]))
