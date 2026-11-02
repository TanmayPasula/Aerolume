"""FieldView: the flow-field viewport used by the Simulate, Results and Compare pages.

It paints a colour image (a ``LiveFrame.rgba`` straight from the solver's GPU
renderer, or a stored ``FieldData`` array colourised here) mapped onto its
physical extent in metres, with equal aspect ratio, and draws vector overlays
(bodies, particles, streamlines) in the same world coordinates.

Design notes (the WHY):

* **One world transform.** Everything is stored in metres and drawn through a
  single world->screen ``QTransform``; overlays are built once per data update as
  ``QPainterPath`` / ``QPolygonF`` and re-used on every pan / zoom repaint, so
  interaction costs one ``drawImage`` plus a few path strokes (well under the
  16 ms budget of 60 fps at 1024 x 512).
* **The colour bar owns a strip, it does not float.** A vertical bar with title,
  units and ticks sits in a reserved strip at the right edge (the ParaView /
  Fluent convention), and the field is fitted into the remaining area, so the
  legend never hides the wake it explains.
* **Neutral solids.** Bodies are mid-grey with a 1 px dark outline in both
  themes (docs/ui_guidelines.md section 5): a solid must never be mistaken for a
  value of the colour map, and grey sits outside every map Aerolume offers.
* **Hover values without raw data.** Live frames carry only RGBA. The readout
  inverts the colour map (nearest entry of the 256-colour LUT) to recover the
  value under the cursor, accurate to 1/255 of the colour range, which is what
  the eye can resolve anyway. When raw values are supplied they are used instead.
* **Overlay contrast follows the image.** Streamlines and particles are drawn
  white on dark images and near-black on light ones (decided from the image's
  mean luminance), because no single colour reads on both ends of viridis.
* **Auto-fit until the user takes over.** New extents re-fit the view only while
  the user has not zoomed / panned; double-click or ``F`` returns to auto-fit.
  A new field of another study (:func:`extent_jumped`: a span changed by more
  than 2x, or no overlap) always re-fits and drops the old body box: a 4 m room
  after a 10 mm cylinder was framed at "2 mm" as one flat rectangle.
* **One default framing, filling the viewport** (:func:`near_field_window`):
  the bodies' box plus 1 reference length upstream, 3 downstream and 1 on
  each side - on Simulate, Results and Compare alike. Fitting the whole
  domain (8 x 4 to 31 x 20 m) drew the body at 80-130 px with white bands
  around it, which hides the flow the page exists to show. The window is
  cropped to the viewport's aspect ratio (no letterboxing) but never so far
  that the body and its near wake (:data:`ESSENTIAL_MARGINS`) leave the view,
  and never past the domain's edges. The toolbar's Fit toggle switches to the
  whole domain and back.
* **The road is drawn, not implied.** Ground runs end at y = 0; below it the
  view shows a 1 px text2 road line over a hatched band, as on the Geometry
  and Setup pages, instead of a white area that reads as a cropped field.
* **One colour-range rule** (:func:`display_range`): percentiles sampled inside
  the bodies' box grown by one reference length, speed from exactly 0 to the
  99.5th percentile, diverging fields symmetric. Over the whole domain the free
  stream is most of the sample and the range collapses to U-infinity +-20 %,
  which paints boundary layers, stagnation point and wake one flat colour.
  Clipped ends get extend triangles on the colour bar. Live frames are coloured
  by the solver, so :class:`LiveRangeTracker` applies the same rule by feeding a
  fixed range back to it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pyqtgraph.functions as pg_functions
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import QComboBox, QSizePolicy, QToolButton, QWidget

from ...core.units import sig
from .common import ToolStrip, label
from .plots import connect_theme, lazy_import, nice_ticks, qcolor, tokens

# ---------------------------------------------------------------------------
# colour maps and field metadata (viz is authoritative; these are fallbacks)
# ---------------------------------------------------------------------------

_ANCHORS = {
    # perceptually ordered anchors, interpolated to 256 entries when viz is missing
    "viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    "coolwarm": ["#3b4cc0", "#7b9ff9", "#dddddd", "#f49a7b", "#b40426"],
    "RdBu_r": ["#053061", "#4393c3", "#f7f7f7", "#d6604d", "#67001f"],
    "magma": ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
    "grayscale": ["#000000", "#ffffff"],
}
_FALLBACK_DEFAULT_CMAP = {"speed": "viridis", "u": "viridis", "v": "coolwarm", "cp": "coolwarm",
                          "p": "coolwarm", "vorticity": "RdBu_r", "dye": "grayscale", "nut": "magma",
                          "mask": "grayscale", "sensitivity": "RdBu_r", "temperature": "magma",
                          "theta": "magma", "divergence": "RdBu_r"}
_FALLBACK_FIELD_INFO = {
    "speed": ("Speed |V|", "speed", False),
    "u": ("Streamwise velocity u", "speed", False),
    "v": ("Vertical velocity v", "speed", True),
    "cp": ("Pressure coefficient Cp", "none", True),
    "p": ("Pressure (gauge)", "pressure", True),
    "vorticity": ("Vorticity ωz", "vorticity", True),
    "dye": ("Smoke", "none", False),
    "nut": ("Eddy viscosity νt", "kinematic_viscosity", False),
    "mask": ("Solid fraction", "none", False),
    "divergence": ("Divergence", "frequency", True),
    "temperature": ("Temperature T", "temperature", False),
    "theta": ("Temperature excess θ", "none", False),
}
# SI unit string for each FIELD_INFO units kind
UNIT_OF_KIND = {"speed": "m/s", "velocity": "m/s", "pressure": "Pa", "vorticity": "1/s",
                "frequency": "1/s", "rate": "1/s", "kinematic_viscosity": "m²/s", "viscosity": "m²/s",
                "length": "m", "none": "", "": "", "dimensionless": "", "fraction": "", "temperature": "°C",
                "temperature_difference": "K", "heat_flux": "W/m²"}
#: Display names for the colour-map combo (keys are viz.COLORMAPS names).
CMAP_TITLES = {"viridis": "Viridis", "magma": "Magma", "inferno": "Inferno", "plasma": "Plasma",
               "cividis": "Cividis", "thermal": "Thermal", "fire": "Fire", "turbo": "Turbo (rainbow)",
               "coolwarm": "Cool–warm", "RdBu_r": "Red–blue", "bwr": "Blue–white–red", "icefire": "Ice–fire",
               "grayscale": "Greyscale"}
#: Fields offered in the viewport toolbars, in menu order: (key, menu text).
VIEW_FIELDS = (("speed", "Speed"), ("cp", "Pressure coefficient"), ("vorticity", "Vorticity"),
               ("dye", "Smoke"), ("nut", "Eddy viscosity"))
#: Fields only a run with placed physics has (DESIGN.md 10): added by :func:`view_fields_for`.
THERMAL_FIELDS = (("temperature", "Temperature"),)
#: AeroJAX-style solver diagnostics as a field (the discrete divergence of the velocity).
DIAGNOSTIC_FIELDS = (("divergence", "Divergence ∇·u"),)


def view_fields_for(sim, *, three_d: bool = False, diagnostics: bool = False) -> tuple[tuple[str, str], ...]:
    """The field menu of a simulation: the base list, plus Temperature when heat transfer is on.

    Smoke stays one entry: with several emitter colours its picture is the
    multi-colour composite (the legend names the channels). ``diagnostics``
    appends the divergence field for the solver-inspection view.
    """
    base = list(VIEW_FIELDS_3D if three_d else VIEW_FIELDS)
    thermal = getattr(sim, "thermal", None) if sim is not None else None
    if thermal is not None and bool(getattr(thermal, "enabled", False)):
        i = next((k for k, (key, _t) in enumerate(base) if key == "dye"), len(base))
        base[i:i] = list(THERMAL_FIELDS)
    if diagnostics and not three_d:
        base += list(DIAGNOSTIC_FIELDS)
    return tuple(base)
#: Fields of a 3D slice (live frames and stored results). Vorticity is the magnitude |ω| in
#: 3D and "q" the Q-criterion (vortex cores where Q > 0). "dye" is the smoke concentration in
#: the slice; the smoke itself is drawn as a volume in the 3D view (widgets.view3d_panel,
#: DESIGN.md 7.13).
VIEW_FIELDS_3D = (("speed", "Speed"), ("cp", "Pressure coefficient"), ("vorticity", "Vorticity |ω|"),
                  ("q", "Q-criterion"), ("u", "Velocity u"), ("v", "Velocity v"), ("w", "Velocity w"),
                  ("dye", "Smoke"), ("nut", "Eddy viscosity"))
#: 3D field metadata (label, unit, diverging, default map) when viz3d.colors is unavailable;
#: the same table as :data:`aerolume.viz3d.colors.FIELD_INFO_3D`.
_FALLBACK_FIELD_INFO_3D = {
    "speed": ("Speed |V|", "m/s", False, "viridis"), "u": ("Velocity u", "m/s", False, "viridis"),
    "v": ("Velocity v", "m/s", True, "coolwarm"), "w": ("Velocity w", "m/s", True, "coolwarm"),
    "p": ("Pressure (gauge)", "Pa", True, "coolwarm"), "cp": ("Pressure coefficient Cp", "", True, "coolwarm"),
    "cf": ("Skin friction coefficient Cf", "", False, "viridis"),
    "vorticity": ("Vorticity |ω|", "1/s", False, "magma"), "q": ("Q-criterion", "1/s²", True, "RdBu_r"),
    "sensitivity": ("Sensitivity dJ/dn", "1/m", True, "RdBu_r"), "nut": ("Eddy viscosity νt", "m²/s", False, "magma"),
    "mask": ("Solid fraction", "", False, "grayscale"), "dye": ("Smoke concentration", "", False, "grayscale"),
    "temperature": ("Temperature T", "°C", False, "thermal"),
}
#: 3D magnitudes whose colour range starts at exactly 0 (viz3d.colors.ZERO_BASED).
NON_NEGATIVE_FIELDS_3D = frozenset({"speed", "vorticity", "cf", "tau", "nut", "mask", "dye"})
#: In-plane (column, row) axis names of a slice normal to x, y or z (SliceData orientation).
SLICE_AXIS_NAMES = {"y": ("x", "z"), "z": ("x", "y"), "x": ("y", "z")}

#: Solids: neutral mid-grey with a dark 1 px outline, identical in both themes.
SOLID_FILL = (143, 148, 156)
SOLID_EDGE = (42, 45, 50)

_LUT_CACHE: dict[str, np.ndarray] = {}


def _viz():
    """The module that carries ``COLORMAPS`` (package re-export or ``viz.colormaps``), or None."""
    for name in ("aerolume.viz.colormaps", "aerolume.viz"):
        mod = lazy_import(name)
        if mod is not None and hasattr(mod, "COLORMAPS"):
            return mod
    return None


def _interp_lut(hexes: Sequence[str]) -> np.ndarray:
    cols = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=float)
    pos = np.linspace(0, 1, len(cols))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, pos, cols[:, c]) for c in range(3)], axis=1).round().astype(np.uint8)


def colormap_names() -> list[str]:
    """Colour maps offered in the UI (from ``viz.COLORMAPS`` when available)."""
    viz = _viz()
    names = list(getattr(viz, "COLORMAPS", {}) or {}) if viz is not None else []
    return names or list(_ANCHORS)


def cmap_title(name: str) -> str:
    return CMAP_TITLES.get(name, name)


def colormap_families() -> list[tuple[str, list[str]]]:
    """The offered maps grouped by family ([("Sequential", [...]), ("Diverging", [...]), ...]).

    From ``viz.colormaps.CMAP_FAMILIES``; maps it does not list go last, so a
    newly added map is always offered.
    """
    names = colormap_names()
    viz = _viz()
    fams = getattr(viz, "CMAP_FAMILIES", None) if viz is not None else None
    if not fams:
        return [("", names)]
    out, seen = [], set()
    for fam in fams:
        title, members = fam[0], [n for n in fam[1] if n in names]
        if members:
            out.append((title, members))
            seen.update(members)
    rest = [n for n in names if n not in seen]
    if rest:
        out.append(("Other", rest))
    return out


def default_cmap(field: str) -> str:
    viz = _viz()
    table = getattr(viz, "DEFAULT_CMAP", None) if viz is not None else None
    if table and field in table:
        return table[field]
    name = _FALLBACK_DEFAULT_CMAP.get(field, "viridis")
    names = colormap_names()
    return name if name in names else names[0]


def colormap_lut(name: str | None, field: str = "speed") -> np.ndarray:
    """(256, 3) uint8 lookup table for a colour-map name (default per field when None)."""
    name = name or default_cmap(field)
    viz = _viz()
    maps = getattr(viz, "COLORMAPS", None) if viz is not None else None
    if maps and name in maps:
        lut = np.asarray(maps[name])
        if lut.dtype != np.uint8:
            lut = np.clip(lut * (255.0 if lut.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return lut[:, :3]
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = _interp_lut(_ANCHORS.get(name, _ANCHORS["viridis"]))
    return _LUT_CACHE[name]


def field_info(field: str) -> tuple[str, str, bool]:
    """(label, SI unit string, diverging) for a field name."""
    viz = _viz()
    table = getattr(viz, "FIELD_INFO", None) if viz is not None else None
    info = (table or {}).get(field) or _FALLBACK_FIELD_INFO.get(field) or (field, "", False)
    lab, kind, div = info[0], info[1] if len(info) > 1 else "", bool(info[2]) if len(info) > 2 else False
    try:
        unit = info.unit  # viz.FieldInfo knows its own SI symbol
    except AttributeError:  # a plain (label, kind, diverging) tuple
        unit = UNIT_OF_KIND.get(kind, kind if ("/" in str(kind) or len(str(kind)) <= 3) else "")
    return lab, unit, div


def field_info3d(field: str) -> tuple[str, str, bool, str]:
    """(label, SI unit, diverging, default colour map) of a 3D field.

    3D differs from 2D for two names: ``vorticity`` is the magnitude |ω| (a
    sequential map from 0) and ``v`` is the lateral velocity. The table is
    :data:`aerolume.viz3d.colors.FIELD_INFO_3D` (numpy only), so a slice in the
    Simulate page, the Results page and the 3D viewer colour the same value
    the same way.
    """
    mod = lazy_import("aerolume.viz3d.colors")
    table = getattr(mod, "FIELD_INFO_3D", None) if mod is not None else None
    if table and field in table:
        fi = table[field]
        return fi.label, fi.unit, bool(fi.diverging), fi.cmap
    base = field.removesuffix("_rms") if field else field
    if base in _FALLBACK_FIELD_INFO_3D:
        lab, unit, div, cmap = _FALLBACK_FIELD_INFO_3D[base]
        if base != field:
            return f"{lab} (rms)", unit, False, "magma"
        return lab, unit, div, cmap
    return field or "", "", False, "viridis"


def default_cmap3d(field: str) -> str:
    """The recommended colour map of a 3D field (falls back to the first available map)."""
    name = field_info3d(field)[3]
    names = colormap_names()
    return name if name in names else names[0]


#: Fields that are magnitudes (never negative): their colour range starts at exactly 0,
#: so a stagnation point or a dead-water region reads as "no flow", not as "low end of
#: whatever the free stream happened to span".
NON_NEGATIVE_FIELDS = frozenset({"speed", "nut", "dye", "mask", "u_rms", "v_rms", "w_rms", "p_rms", "cp_rms",
                                 "temperature_rms", "theta_rms"})
#: Upper percentile of the automatic range. 99.5 rather than 99: the sample is the near
#: field, where the suction peak and the stagnation region are a few per cent of the cells.
RANGE_HI_PCT = 99.5
#: Cp is centred on 0 and spans at least +-1 (stagnation is exactly Cp = 1), at most +-3
#: (a sharp leading edge's suction peak of -6 would wash every other pressure out).
CP_LIMITS = (1.0, 3.0)


def body_window(polys: Sequence[np.ndarray] | None, extent: Sequence[float] | None = None, *,
                margin: float = 1.0) -> tuple[float, float, float, float] | None:
    """The bodies' bounding box grown by ``margin`` reference lengths on every side.

    The reference length is the larger side of the box (chord, vehicle length,
    cylinder diameter). This is where the flow differs from the free stream: the
    automatic colour range is sampled here, because over the whole domain the
    free-stream cells are the overwhelming majority and the percentiles collapse
    to U-infinity +-20 %, which saturates the boundary layers, the stagnation
    point and the wake. Clipped to ``extent`` when given; None without bodies.
    """
    pts = [np.asarray(p, dtype=float).reshape(-1, 2) for p in (polys or []) if len(p) >= 3]
    if not pts:
        return None
    a = np.vstack(pts)
    x0, x1 = float(a[:, 0].min()), float(a[:, 0].max())
    y0, y1 = float(a[:, 1].min()), float(a[:, 1].max())
    pad = margin * max(x1 - x0, y1 - y0, 1e-9)
    win = [x0 - pad, x1 + pad, y0 - pad, y1 + pad]
    if extent is not None:
        ex0, ex1, ey0, ey1 = (float(e) for e in extent)
        win = [max(win[0], ex0), min(win[1], ex1), max(win[2], ey0), min(win[3], ey1)]
        if win[1] <= win[0] or win[3] <= win[2]:
            return None
    return tuple(win)  # type: ignore[return-value]


#: Default framing (:func:`near_field_window`), in reference lengths around the bodies' box.
NEAR_FIELD_UPSTREAM, NEAR_FIELD_DOWNSTREAM, NEAR_FIELD_SIDE = 1.0, 3.0, 1.0
#: (upstream, downstream, side) margins, in reference lengths, that cropping the default
#: window to the viewport's aspect ratio must keep in view: the body, the flow arriving at
#: it and one and a half lengths of wake. Without this a tall, narrow viewport would crop
#: the nose, or show a car's roof and no wake.
ESSENTIAL_MARGINS = (0.5, 1.5, 0.5)
#: (upstream, downstream) margins, in reference lengths, of the tighter window a ``keep_body``
#: view zooms to when the essential window leaves blank bands above and below the domain
#: (1.5 body lengths across: the body, its nose flow and the start of the wake).
KEEP_BODY_MARGINS = (0.15, 0.35)
#: Height of the hatched road band under the road line (px), as on the Geometry page.
ROAD_BAND_PX = 12.0


def split_orientation(width: float, height: float, aspect: float | None) -> Qt.Orientation:
    """How to split a ``width`` x ``height`` viewport between a 3D view and a 2D slice of ``aspect``.

    Side by side (``Qt.Horizontal``) gives each pane ``width / 2`` x ``height``;
    stacked (``Qt.Vertical``) gives ``width`` x ``height / 2``. The arrangement
    whose pane shape is closer to the slice's (in log ratio, so 3:1 against
    1.5:1 counts the same as 1:3 against 1:1.5) wastes the least of the pane
    on blank bands. A car's side slice (about 3:1) is stacked under the 3D
    view; a cross-flow slice (about 1:1) sits beside it.
    """
    if aspect is None or not math.isfinite(aspect) or aspect <= 0 or width <= 0 or height <= 0:
        return Qt.Orientation.Horizontal
    side = (width / 2.0) / height
    stacked = width / (height / 2.0)
    d_side = abs(math.log(aspect / side))
    d_stacked = abs(math.log(aspect / stacked))
    return Qt.Orientation.Vertical if d_stacked < d_side else Qt.Orientation.Horizontal


def _bbox(polys: Sequence[np.ndarray] | None) -> tuple[float, float, float, float] | None:
    pts = [np.asarray(p, dtype=float).reshape(-1, 2) for p in (polys or []) if len(p) >= 3]
    if not pts:
        return None
    a = np.vstack(pts)
    return float(a[:, 0].min()), float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 1].max())


def placements_box(placements: Sequence | None) -> tuple[float, float, float, float] | None:
    """Bounding box [m] of placed emitters, fans, heaters and ports (``draw_tools.Placement``-like).

    Each item counts with its handle, a line or rake emitter's far end, a heater's box (or
    disc radius) and half a fan's or port's width on every side of the handle (they may
    point any way). None without finite placements.
    """
    xs: list[float] = []
    ys: list[float] = []
    for pl in placements or []:
        try:
            x, y = float(pl.x), float(pl.y)
        except (AttributeError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        sizes = [abs(float(s)) for s in tuple(getattr(pl, "size", ()) or ())[:2]]
        half = 0.5 * abs(float(getattr(pl, "width", 0.0) or 0.0))
        if sizes and str(getattr(pl, "shape", "")) in ("disc", "sphere"):
            hx = hy = max(half, sizes[0])                    # a disc heater's size is its radius
        elif sizes:
            hx, hy = max(half, 0.5 * sizes[0]), max(half, 0.5 * (sizes[1] if len(sizes) > 1 else sizes[0]))
        else:
            hx = hy = half
        xs += [x - hx, x + hx]
        ys += [y - hy, y + hy]
        end = getattr(pl, "end", None)
        if end is not None and len(end) >= 2 and math.isfinite(float(end[0])) and math.isfinite(float(end[1])):
            xs.append(float(end[0]))
            ys.append(float(end[1]))
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def near_field_window(polys: Sequence[np.ndarray] | None, extent: Sequence[float] | None = None, *,
                      upstream: float = NEAR_FIELD_UPSTREAM, downstream: float = NEAR_FIELD_DOWNSTREAM,
                      side: float = NEAR_FIELD_SIDE,
                      ground: float | None = None) -> tuple[float, float, float, float] | None:
    """The default view window: the bodies' box grown by multiples of the reference length.

    The reference length is the larger side of the box (chord, vehicle length,
    cylinder diameter). With a ``ground`` (road height) the road is the floor
    of the window. Clipped to ``extent`` when given; None without bodies.
    Simulate, Results and Compare all frame their fields with this, so the
    same flow looks the same on every page.
    """
    box = _bbox(polys)
    if box is None:
        return None
    x0, x1, y0, y1 = box
    L = max(x1 - x0, y1 - y0, 1e-9)
    win = [x0 - upstream * L, x1 + downstream * L, y0 - side * L, y1 + side * L]
    if ground is not None:
        win[2] = max(win[2], float(ground))
    if extent is not None:
        ex0, ex1, ey0, ey1 = (float(e) for e in extent)
        clipped = [max(win[0], ex0), min(win[1], ex1), max(win[2], ey0), min(win[3], ey1)]
        if clipped[1] > clipped[0] and clipped[3] > clipped[2]:
            win = clipped
    return tuple(win)  # type: ignore[return-value]


def window_slices(shape: tuple[int, int], extent: Sequence[float],
                  window: Sequence[float] | None, *, image_order: bool = False) -> tuple[slice, slice]:
    """(row, column) slices of a cell-centred (ny, nx) array that cover ``window``.

    ``image_order=False`` is solver order (row 0 = y min, FieldData); True is
    image order (row 0 = top, LiveFrame.rgba). The whole array without a window.
    """
    ny, nx = int(shape[0]), int(shape[1])
    if window is None:
        return slice(0, ny), slice(0, nx)
    x0, x1, y0, y1 = (float(e) for e in extent)
    wx0, wx1, wy0, wy1 = (float(w) for w in window)
    i0 = int(np.clip(math.floor((wx0 - x0) / (x1 - x0) * nx), 0, nx - 1))
    i1 = int(np.clip(math.ceil((wx1 - x0) / (x1 - x0) * nx), i0 + 1, nx))
    j0 = int(np.clip(math.floor((wy0 - y0) / (y1 - y0) * ny), 0, ny - 1))
    j1 = int(np.clip(math.ceil((wy1 - y0) / (y1 - y0) * ny), j0 + 1, ny))
    if image_order:
        j0, j1 = ny - j1, ny - j0
    return slice(j0, j1), slice(i0, i1)


def robust_range(values: np.ndarray, *, diverging: bool = False, mask: np.ndarray | None = None,
                 lo: float = 1.0, hi: float = 99.0, extent: Sequence[float] | None = None,
                 window: Sequence[float] | None = None, zero_floor: bool = False) -> tuple[float, float]:
    """Percentile colour range over fluid cells; symmetric about zero for diverging fields.

    Percentiles rather than min / max because a handful of cells next to sharp
    corners carry extreme pressure and vorticity that would wash out the rest.
    With ``extent`` and ``window`` (metres) only the cells inside the window are
    sampled (see :func:`body_window` for why). ``zero_floor`` pins the lower
    end at 0 for magnitudes.
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if mask is not None and np.shape(mask) == v.shape:
        ok &= np.asarray(mask) < 0.5
    if window is not None and extent is not None and v.ndim == 2:
        rows, cols = window_slices(v.shape, extent, window)
        if np.any(ok[rows, cols]):
            v, ok = v[rows, cols], ok[rows, cols]
    sample = v[ok]
    if sample.size == 0:
        return 0.0, 1.0
    if sample.size > 400_000:  # percentiles of a strided sample are indistinguishable and 10x cheaper
        sample = sample[:: sample.size // 200_000]
    a, b = np.percentile(sample, [lo, hi])
    if diverging:
        m = max(abs(a), abs(b)) or 1.0
        return -m, m
    if zero_floor:
        a = 0.0
    if b <= a:
        b = a + (abs(b) or abs(a) or 1.0) * 1e-3
    return float(a), float(b)


def display_range(field: str, values: np.ndarray, *, extent: Sequence[float] | None = None,
                  mask: np.ndarray | None = None,
                  window: Sequence[float] | None = None) -> tuple[float, float]:
    """The automatic colour range for a stored field: the one rule every page uses.

    * magnitudes (speed, eddy viscosity): 0 to the 99.5th percentile;
    * smoke / solid fraction: 0 to 1;
    * Cp: symmetric about 0, at least +-1, at most +-3 (:data:`CP_LIMITS`);
    * other diverging fields (vorticity, v, p): symmetric about 0 at the
      larger of the 0.5th / 99.5th percentile magnitudes;
    * anything else: 0.5th to 99.5th percentile.

    Percentiles are taken inside ``window`` (normally :func:`body_window`).
    """
    if field in ("dye", "mask") or field.startswith("dye"):
        return 0.0, 1.0
    if field in ("temperature", "theta"):
        # bounded by the maximum principle (no spike cells to clip): the full fluid range, so the hottest
        # fluid at a heated wall reads as the top colour rather than as "beyond the scale"
        return robust_range(values, mask=mask, lo=0.0, hi=100.0, extent=extent, window=None)
    div = field_info(field)[2]
    lo_pct = 100.0 - RANGE_HI_PCT
    a, b = robust_range(values, diverging=div, mask=mask, lo=lo_pct, hi=RANGE_HI_PCT, extent=extent,
                        window=window, zero_floor=field in NON_NEGATIVE_FIELDS)
    if field == "cp":
        m = min(max(abs(a), CP_LIMITS[0]), CP_LIMITS[1])
        return -m, m
    return a, b


def display_range3d(field: str, values: np.ndarray, *, extent: Sequence[float] | None = None,
                    mask: np.ndarray | None = None,
                    window: Sequence[float] | None = None) -> tuple[float, float]:
    """:func:`display_range` for a slice of a 3D result (3D field semantics, same percentiles).

    Magnitudes (speed, |ω|, νt) from 0; Cp centred, at least +-1, at most +-3;
    v, w, p and Q symmetric about 0.
    """
    if field in ("mask", "dye"):           # fractions: the full scale, so "no smoke" is always black
        return 0.0, 1.0
    div = field_info3d(field)[2]
    lo_pct = 100.0 - RANGE_HI_PCT
    a, b = robust_range(values, diverging=div, mask=mask, lo=lo_pct, hi=RANGE_HI_PCT, extent=extent,
                        window=window, zero_floor=field in NON_NEGATIVE_FIELDS_3D)
    if field == "cp":
        m = min(max(abs(a), CP_LIMITS[0]), CP_LIMITS[1])
        return -m, m
    return a, b


def section_paths(lines: Sequence[np.ndarray] | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split mesh cross-sections into (closed loops, open chains).

    A 3D slice cuts each mesh into polylines (``simulation.live3d.mesh_section``):
    a closed surface gives loops whose last point repeats the first, an open
    sheet (a wing without tip caps, a spoiler) gives open chains. Loops are
    solids and are filled; chains are walls of zero thickness and are only
    stroked -- closing them would paint a triangle of "solid" across the flow.
    """
    closed, open_ = [], []
    for ln in lines or []:
        p = np.asarray(ln, dtype=float).reshape(-1, 2)
        if len(p) < 2:
            continue
        span = float(np.ptp(p, axis=0).max()) if len(p) else 0.0
        if len(p) >= 4 and np.allclose(p[0], p[-1], atol=1e-9 * max(span, 1e-12) + 1e-12):
            closed.append(p[:-1])
        else:
            open_.append(p)
    return closed, open_


def clipped_ends(values: np.ndarray, vmin: float, vmax: float, *,
                 mask: np.ndarray | None = None) -> tuple[bool, bool]:
    """(below, above): whether any fluid value lies outside [vmin, vmax].

    Drives the extend triangles of the colour bar (the matplotlib convention):
    a saturated end colour then reads as "this much or more", not "exactly this".
    """
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if mask is not None and np.shape(mask) == v.shape:
        ok &= np.asarray(mask) < 0.5
    sample = v[ok]
    if sample.size == 0:
        return False, False
    tol = 1e-6 * max(abs(vmax - vmin), 1e-30)
    return bool(np.any(sample < vmin - tol)), bool(np.any(sample > vmax + tol))


def lut_indices(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Colour-map index (0..255) of each pixel of an (..., 3) image; -1 for colours not in the map.

    Exact matching is right here: the solver's renderer writes LUT entries
    unchanged for fluid cells, and everything else (solids, their anti-aliased
    edge) is a colour the map does not contain and must be ignored anyway.
    """
    lut = np.asarray(lut, dtype=np.int64)[:, :3]
    px = np.asarray(rgb, dtype=np.int64)[..., :3]
    packed = (px[..., 0] << 16) | (px[..., 1] << 8) | px[..., 2]
    keys = (lut[:, 0] << 16) | (lut[:, 1] << 8) | lut[:, 2]
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    pos = np.clip(np.searchsorted(sk, packed), 0, len(sk) - 1)
    return np.where(sk[pos] == packed, order[pos], -1)


#: Live fields whose range starts at a physical floor, so the first colour means "at it", not "below":
#: a live temperature range starts at the coldest temperature of the case (the ambient around a heated
#: body), which by the maximum principle nothing undershoots without a heat sink, and most of the
#: fluid sits exactly there. Read from colours alone that looked like clipping and drew a misleading
#: extend triangle under every live temperature colour bar.
LIVE_FLOOR_FIELDS = frozenset({"temperature", "theta", "dye1", "dye2", "dye3"})


def frame_clipped_ends(rgba: np.ndarray, lut: np.ndarray, *, field: str = "",
                       threshold: float = 0.002) -> tuple[bool, bool]:
    """Extend flags for a live frame, which carries colours but no values.

    A pixel in the first / last colour of the map is at or beyond that end of
    the range; when more than ``threshold`` of the fluid pixels are, the end is
    taken as clipped. Magnitudes drawn from 0 never extend below.
    """
    a = np.asarray(rgba)
    step = max(1, int(math.sqrt(a.shape[0] * a.shape[1] / 60_000)))
    idx = lut_indices(a[::step, ::step], lut)
    fluid = idx >= 0
    n = int(fluid.sum())
    if n == 0:
        return False, False
    top = len(lut) - 1
    below = field not in NON_NEGATIVE_FIELDS and field not in LIVE_FLOOR_FIELDS \
        and float(np.sum(idx == 0)) / n > threshold
    return below, float(np.sum(idx == top)) / n > threshold


class LiveRangeTracker:
    """Keeps the live speed colour range at 0 .. the 99.5th percentile near the body.

    Live frames are coloured on the solver side, whose automatic range is the
    1st-99th percentile of the whole domain: dominated by free-stream cells, it
    collapses to about U-infinity +-20 % and saturates everything that matters.
    The GUI cannot re-colour a frame (it carries RGBA, not values), so this
    closes the loop instead: it reads the colour indices inside the body window
    of each frame, works out the range the rule asks for, and the caller hands
    it back to the solver as a fixed range for the next frames.

    * When the top colour covers more than 0.5 % of the window, the 99.5th
      percentile lies beyond the range and is unknown: the range grows by
      :attr:`GROW` per frame (a few frames at 20 fps).
    * Otherwise the target is the measured 99.5th percentile; the range moves
      to it at once when growing and by :attr:`SHRINK` of the gap per frame
      when shrinking, with a 6 % dead band, so colours do not pump.

    Only magnitudes (:data:`TRACKED`) are tracked; for other fields
    :meth:`update` returns None and the solver's own range stays in charge.
    """

    TRACKED = frozenset({"speed"})
    GROW = 1.3
    SHRINK = 0.25
    DEAD_BAND = 0.06
    MAX_SAMPLES = 40_000

    def __init__(self) -> None:
        self.vmax: float | None = None
        self.field = ""

    def reset(self) -> None:
        self.vmax = None
        self.field = ""

    def update(self, frame) -> tuple[float, float] | None:
        """The (vmin, vmax) to request for the next frames, or None to leave the solver's range."""
        field = str(getattr(frame, "field", "") or "")
        if field not in self.TRACKED:
            self.reset()
            return None
        if field != self.field:
            self.reset()
            self.field = field
        rgba = np.asarray(frame.rgba)
        vmin, vmax = float(frame.vmin), float(frame.vmax)
        if rgba.ndim != 3 or not (math.isfinite(vmax) and vmax > vmin):
            return None
        lut = colormap_lut(getattr(frame, "cmap", None), field)
        win = body_window(getattr(frame, "bodies", None), frame.extent)
        rows, cols = window_slices(rgba.shape[:2], frame.extent, win, image_order=True)
        crop = rgba[rows, cols]
        step = max(1, int(math.sqrt(crop.shape[0] * crop.shape[1] / self.MAX_SAMPLES)))
        idx = lut_indices(crop[::step, ::step], lut)
        idx = idx[idx >= 0]
        if idx.size < 16:
            return None
        top = len(lut) - 1
        if float(np.mean(idx == top)) > (100.0 - RANGE_HI_PCT) / 100.0:
            target = vmax * self.GROW
        else:
            values = vmin + (vmax - vmin) * idx.astype(np.float64) / top
            target = float(np.percentile(values, RANGE_HI_PCT))
        target = max(target, 1e-9)
        if self.vmax is None or target > self.vmax:
            self.vmax = target
        elif target < (1.0 - self.DEAD_BAND) * self.vmax:
            self.vmax += self.SHRINK * (target - self.vmax)
        return 0.0, float(self.vmax)


def colorize(values: np.ndarray, lut: np.ndarray, vmin: float, vmax: float, *,
             mask: np.ndarray | None = None, solid: Sequence[int] = (*SOLID_FILL, 255),
             flip: bool = True) -> np.ndarray:
    """Map a 2-D array to RGBA uint8 through a LUT.

    ``flip=True`` converts solver order (row 0 = y min, as stored in FieldData) to
    image order (row 0 = top), the convention of ``LiveFrame.rgba``. NaNs and
    solid cells (mask >= 0.5) get the ``solid`` colour.
    """
    v = np.asarray(values, dtype=np.float32)
    span = (vmax - vmin) or 1.0
    idx = np.nan_to_num((v - vmin) * (255.0 / span), nan=0.0, posinf=255.0, neginf=0.0)
    idx = np.rint(np.clip(idx, 0, 255)).astype(np.uint8)  # round, so the readout inverts exactly
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = lut[idx]
    rgba[..., 3] = 255
    bad = ~np.isfinite(v)
    if mask is not None and np.shape(mask) == v.shape:
        bad |= np.asarray(mask) >= 0.5
    if bad.any():
        rgba[bad] = np.asarray(solid, dtype=np.uint8)
    return np.ascontiguousarray(rgba[::-1] if flip else rgba)


def format_length(metres: float) -> str:
    """'0.5 m', '200 mm', '2 km': the unit a scale bar should read in."""
    if metres >= 1000:
        return f"{metres / 1000:g} km"
    if metres >= 1:
        return f"{metres:g} m"
    if metres >= 0.01:
        return f"{metres * 100:g} cm" if metres < 0.1 else f"{metres * 1000:g} mm"
    return f"{metres * 1000:g} mm"


def to_qpolygonf(points: np.ndarray) -> QPolygonF:
    """(N, 2) array -> QPolygonF without a Python loop.

    pyqtgraph exposes the polygon's memory as a numpy view; at 5000 particles per
    frame that is ~0.01 ms instead of ~5 ms for building QPointF objects one by one.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    try:
        poly = pg_functions.create_qpolygonf(len(pts))
        pg_functions.ndarray_from_qpolygonf(poly)[:] = pts
        return poly
    except Exception:  # noqa: BLE001 - pyqtgraph internals moved: fall back to the slow path
        return QPolygonF([QPointF(x, y) for x, y in pts])


def boxes_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """True when two (xmin, xmax, ymin, ymax) boxes share some area (touching counts)."""
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def extent_jumped(old: Sequence[float], new: Sequence[float], *, ratio: float = 2.0) -> bool:
    """True when a new field extent belongs to a different study rather than the same domain.

    A mesh refinement or a re-run keeps the domain within a few per cent; a switch
    from a 10 mm cylinder to a 4 m room changes its size 400-fold. A factor of
    ``ratio`` in either span, or boxes that do not overlap, is a different study,
    whose old zoom must not survive (the room was framed at "2 mm").
    """
    if not boxes_overlap(old, new):
        return True
    for a, b in ((old[1] - old[0], new[1] - new[0]), (old[3] - old[2], new[3] - new[2])):
        if a <= 0 or b <= 0:
            return True
        if max(a / b, b / a) > ratio:
            return True
    return False


def nice_length(target: float) -> float:
    """Largest 1/2/5 x 10^k not exceeding ``target`` (> 0)."""
    if target <= 0 or not math.isfinite(target):
        return 1.0
    mag = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * mag <= target:
            return m * mag
    return mag


def tick_labels(vmin: float, vmax: float, n: int = 6) -> list[tuple[float, str]]:
    """Round tick values over [vmin, vmax] with labels at a common number of decimals."""
    span = vmax - vmin
    ticks = nice_ticks(vmin, vmax, n) if span > 0 else [vmin]
    step = abs(ticks[1] - ticks[0]) if len(ticks) > 1 else abs(span) or 1.0
    decimals = max(0, -math.floor(math.log10(step) + 1e-9)) if step > 0 else 0
    out = []
    for tv in ticks:
        txt = f"{tv:.{decimals}f}" if abs(tv) < 1e5 and decimals <= 6 else sig(tv, 3)
        out.append((tv, txt.replace("-", "−")))
    return out


def _ui_font(pt: float, bold: bool = False) -> QFont:
    theme = lazy_import("aerolume.app.theme")
    f = theme.ui_font(pt=pt) if theme is not None and hasattr(theme, "ui_font") else QFont()
    f.setPointSizeF(pt)
    if bold:
        f.setWeight(QFont.Weight.DemiBold)
    return f


def colorbar_width(title: str, units: str, vmin: float, vmax: float) -> float:
    """Width the vertical colour bar needs for these labels (px).

    Matches :func:`paint_colorbar`: 12 px of padding each side, the 12 px bar, a 6 px gap to the
    tick label, and 2 px of rounding slack. With 4 px for that gap the strip was 2 px short of what
    the painter draws in, and a label such as "−1" lost its last glyph to the clip.
    """
    fm = QFontMetricsF(_ui_font(8.0))
    fm_t = QFontMetricsF(_ui_font(8.0, bold=True))
    labels = [t for _, t in tick_labels(vmin, vmax)]
    w_ticks = 12 + 6 + 2 + max((fm.horizontalAdvance(t) for t in labels), default=20)
    w_title = max(fm_t.horizontalAdvance(title), fm.horizontalAdvance(units))
    return float(min(max(w_ticks, w_title) + 24, 168))


#: Height of an extend triangle at either end of the colour bar (px).
EXTEND_H = 9.0


def colorbar_ticks(vmin: float, vmax: float, bar_h: float, label_h: float) -> list[tuple[float, str, float]]:
    """(value, label, px above the bar's bottom) of the colour-bar ticks that fit a ``bar_h`` px bar.

    As many ticks as the height has room for (one per 1.5 label heights, 2 to 6), and any that would
    still overlap the one below dropped: six labels on a 40 px bar (a solver-trace panel) piled
    "0.04 0.02 0.00 −0.02 −0.04" on top of each other.
    """
    span = vmax - vmin
    n_fit = int(bar_h // (1.5 * label_h)) + 1 if label_h > 0 else 6

    def place(n: int) -> list[tuple[float, str, float]]:
        out: list[tuple[float, str, float]] = []
        last = -math.inf
        for tv, txt in tick_labels(vmin, vmax, n):
            off = ((tv - vmin) / span if span > 0 else 0.5) * bar_h
            if off - last < label_h:
                continue
            last = off
            out.append((tv, txt, off))
        return out

    best: list[tuple[float, str, float]] = []
    # fewest ticks first, but at least two (a 1, 2, 5 step can leave a single "0.00" on a short bar)
    for n in range(max(2, min(6, n_fit)), 7):
        best = place(n)
        if len(best) >= 2 or span <= 0:
            return best
    return best


def paint_colorbar(p: QPainter, area: QRectF, lut: np.ndarray, vmin: float, vmax: float,
                   title: str, units: str, tk: dict, extend: tuple[bool, bool] = (False, False)) -> None:
    """Draw a vertical colour bar (title, units, gradient, ticks) inside ``area``.

    High values are at the top. The title is elided to the strip width; the
    gradient takes about 60 % of the height, centred, so the bar reads the same
    in a tall viewport and in a compact comparison thumbnail. ``extend`` =
    (below, above) adds a triangle in the end colour at that end: values exist
    beyond the range and are drawn in the end colour (matplotlib's convention).
    """
    pad = 12.0
    f_title = _ui_font(8.0, bold=True)
    f_tick = _ui_font(8.0)
    fm_t = QFontMetricsF(f_title)
    fm = QFontMetricsF(f_tick)
    x = area.left() + pad
    w = area.width() - 2 * pad
    ext_top = EXTEND_H + 2 if extend[1] else 0.0
    ext_bot = EXTEND_H + 2 if extend[0] else 0.0
    head = fm_t.height() + (fm.height() if units else 0) + 6 + ext_top
    bar_h = max(40.0, min(area.height() * 0.62, area.height() - head - 2 * pad - fm.height() - ext_bot))
    top = area.top() + max(pad + head, (area.height() - bar_h) / 2)
    p.setFont(f_title)
    p.setPen(qcolor(tk["text"]))
    t_rect = QRectF(x, top - head, w, fm_t.height())
    p.drawText(t_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               fm_t.elidedText(title, Qt.TextElideMode.ElideRight, w))
    if units:
        p.setFont(f_tick)
        p.setPen(qcolor(tk["text2"]))
        p.drawText(QRectF(x, t_rect.bottom(), w, fm.height()),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, units)
    bar = QRectF(x, top, 12.0, bar_h)
    grad = QLinearGradient(0, bar.bottom(), 0, bar.top())
    n = len(lut)
    for i in range(0, n, 8):
        grad.setColorAt(i / (n - 1), QColor(int(lut[i][0]), int(lut[i][1]), int(lut[i][2])))
    grad.setColorAt(1.0, QColor(int(lut[-1][0]), int(lut[-1][1]), int(lut[-1][2])))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRect(bar)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(qcolor(tk["text2"]), 1))
    p.drawRect(bar.adjusted(0.5, 0.5, -0.5, -0.5))
    for on, base_y, tip_y, rgb in ((extend[1], bar.top() - 1, bar.top() - 1 - EXTEND_H, lut[-1]),
                                   (extend[0], bar.bottom() + 1, bar.bottom() + 1 + EXTEND_H, lut[0])):
        if not on:
            continue
        tri = QPolygonF([QPointF(bar.left() + 0.5, base_y), QPointF(bar.right() - 0.5, base_y),
                         QPointF(bar.center().x(), tip_y)])
        p.setBrush(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        p.setPen(QPen(qcolor(tk["text2"]), 1))
        p.drawPolygon(tri)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setFont(f_tick)
    for tv, txt, off in colorbar_ticks(vmin, vmax, bar.height(), fm.height()):
        ty = bar.bottom() - off
        p.setPen(QPen(qcolor(tk["text2"]), 1))
        p.drawLine(QPointF(bar.right(), ty), QPointF(bar.right() + 3, ty))
        p.setPen(qcolor(tk["text2"]))
        p.drawText(QRectF(bar.right() + 6, ty - fm.height() / 2, w - 18, fm.height()),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)


def paint_legend(p: QPainter, area: QRectF, items: Sequence[tuple[QColor, str]], title: str, tk: dict) -> None:
    """A swatch legend in the colour-bar strip: one row per smoke channel (colour square, its emitters).

    A multi-colour smoke picture has no single scale: each colour is one
    channel (DESIGN.md 10.1), so the strip names the channels instead of
    drawing a gradient that would not mean anything.
    """
    pad = 12.0
    f_title = _ui_font(8.0, bold=True)
    f = _ui_font(8.0)
    fm_t, fm = QFontMetricsF(f_title), QFontMetricsF(f)
    x, w = area.left() + pad, area.width() - 2 * pad
    row = max(fm.height() + 6, 16.0)
    top = area.top() + max(pad, (area.height() - (len(items) * row + fm_t.height() + 6)) / 2)
    p.setFont(f_title)
    p.setPen(qcolor(tk["text"]))
    p.drawText(QRectF(x, top, w, fm_t.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
               fm_t.elidedText(title or "Smoke", Qt.TextElideMode.ElideRight, w))
    y = top + fm_t.height() + 6
    p.setFont(f)
    for colour, text in items:
        sw = QRectF(x, y + (row - 10) / 2, 10, 10)
        p.setPen(QPen(qcolor(tk["text2"]), 1))
        p.setBrush(colour)
        p.drawRect(sw.adjusted(0.5, 0.5, -0.5, -0.5))
        p.setPen(qcolor(tk["text2"]))
        p.drawText(QRectF(x + 16, y, w - 16, row), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   fm.elidedText(text, Qt.TextElideMode.ElideRight, w - 16))
        y += row
    p.setBrush(Qt.BrushStyle.NoBrush)


def particle_colors(frame) -> np.ndarray | None:
    """(N, 3) uint8 colours of a frame's particles, from ``particle_colors`` or ``particle_ids`` +
    ``particle_palette`` (one '#rrggbb' per emitter); None when the frame gives neither."""
    cols = getattr(frame, "particle_colors", None)
    if cols is not None:
        a = np.asarray(cols)
        return a.astype(np.uint8) if a.ndim == 2 and a.shape[1] >= 3 else None
    ids, pal = getattr(frame, "particle_ids", None), getattr(frame, "particle_palette", None)
    if ids is None or not pal:
        return None
    try:
        from ...viz.colormaps import hex_rgb
        table = np.array([hex_rgb(c) for c in pal], dtype=np.uint8)
    except (ValueError, ImportError):
        return None
    idx = np.clip(np.asarray(ids, dtype=np.int64), 0, len(table) - 1)
    return table[idx]


#: (column, row, normal) world axes of a slice normal to x, y or z (``SLICE_AXIS_NAMES`` as indices).
_SLICE_INDICES = {"y": (0, 2, 1), "z": (0, 1, 2), "x": (1, 2, 0)}
#: Half-thickness of the band of a slice whose tracer particles are drawn on it, in image pixels
#: (about grid cells: a live slice image has one pixel per cell).
SLICE_PARTICLE_BAND_PX = 3.0


def frame_particles2d(frame) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """(positions (N, 2), colours, trails) of a frame's tracer particles in its 2D view.

    A 2D frame's particles are drawn as they are. A 3D frame carries them in the volume, (N, 3)
    metres: its slice shows the ones within :data:`SLICE_PARTICLE_BAND_PX` cells of the plane, in
    the slice's own (column, row) axes -- a particle far behind the plane drawn on it would put
    smoke where there is none. (N, 3) positions once crashed every 3D frame here (a reshape to
    two columns).
    """
    pts = getattr(frame, "particles", None)
    if pts is None:
        return None, None, None
    a = np.asarray(pts, dtype=float)
    cols = particle_colors(frame)
    trails = getattr(frame, "particle_trails", None)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] not in (2, 3):
        return None, None, None
    if a.shape[1] == 2:
        return a, cols, trails
    axis = str(getattr(frame, "slice_axis", "") or "")
    if axis not in _SLICE_INDICES:
        return a[:, :2], cols, None                  # a 2D run that sends z = 0
    ci, ri, ni = _SLICE_INDICES[axis]
    pos = getattr(frame, "slice_pos", None)
    keep = np.ones(len(a), bool)
    if pos is not None and math.isfinite(float(pos)):
        ext = getattr(frame, "extent", None)
        rgba = getattr(frame, "rgba", None)
        width = np.shape(rgba)[1] if rgba is not None and np.ndim(rgba) >= 2 else 0
        cell = (float(ext[1]) - float(ext[0])) / width if ext is not None and width else 0.0
        band = SLICE_PARTICLE_BAND_PX * cell if cell > 0 else math.inf
        keep = np.abs(a[:, ni] - float(pos)) <= band
    out = a[keep][:, [ci, ri]]
    if cols is not None and len(cols) == len(a):
        cols = cols[keep]
    else:
        cols = None
    return (out if len(out) else None), (cols if len(out) else None), None


def dye_legend(frame, labels: dict | None = None) -> list[tuple[str, str]] | None:
    """Swatch legend of a multi-channel smoke frame: [(colour, "Rake 1, Jet"), ...]; None for a plain field.

    A frame carries ``dye_channels`` (``viz.colormaps.dye_channels`` entries)
    when its image is the composite of several smoke channels. ``labels``
    ({channel field: [names]}, :func:`~aerolume.app.widgets.view3d_panel.smoke_channel_labels`)
    names emitters left at the default name by kind and colour ("Rake, orange"),
    as the 3D legend does: two rows of "Emitter" told the channels apart not at all.
    """
    chans = getattr(frame, "dye_channels", None)
    if not chans or len(chans) < 2 or str(getattr(frame, "field", "")) not in ("dye", "smoke"):
        return None
    out = []
    for ch in chans:
        names = list(ch.get("names", []) or []) if isinstance(ch, dict) else []
        if labels and isinstance(ch, dict) and labels.get(ch.get("field")):
            names = list(labels[ch.get("field")])
        text = names[0] if len(names) == 1 else (f"{names[0]} +{len(names) - 1}" if names else str(ch.get("field")))
        out.append((str(ch.get("color", "#f5f5f5")), text))
    return out


class ColorBar(QWidget):
    """A standalone vertical colour bar (used beside pyqtgraph plots, e.g. sensitivity)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lut = colormap_lut(None)
        self._vmin, self._vmax = 0.0, 1.0
        self._title, self._units = "", ""
        self._extend = (False, False)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setFixedWidth(90)
        connect_theme(self.update)

    def set_scale(self, lut: np.ndarray, vmin: float, vmax: float, title: str, units: str = "", *,
                  extend: tuple[bool, bool] = (False, False)) -> None:
        self._lut = np.asarray(lut, dtype=np.uint8)
        self._vmin, self._vmax, self._title, self._units = float(vmin), float(vmax), title, units
        self._extend = (bool(extend[0]), bool(extend[1]))
        self.setFixedWidth(int(colorbar_width(title, units, vmin, vmax)))
        self.update()

    def paintEvent(self, _e) -> None:
        tk = tokens()
        p = QPainter(self)
        p.fillRect(self.rect(), qcolor(tk["surface"]))
        paint_colorbar(p, QRectF(self.rect()), self._lut, self._vmin, self._vmax, self._title, self._units, tk,
                       self._extend)
        p.end()


# ---------------------------------------------------------------------------
# the viewport
# ---------------------------------------------------------------------------

class FieldView(QWidget):
    """Zoomable, pannable flow-field viewport with colour bar, scale bar and hover readout.

    Feed it either :meth:`set_frame` (a LiveFrame-like object) or
    :meth:`set_field` (a raw 2-D array + extent). Interaction: wheel zooms about the
    cursor (animated), drag pans, double-click and ``F`` return to the default
    framing, ``+`` / ``-`` zoom. :meth:`set_whole_domain` shows the whole field.
    """

    hovered = Signal(float, float, object)       # x [m], y [m], value (float) or None
    viewChanged = Signal()
    #: True while the view shows the whole domain (the toolbar's Fit toggle follows it)
    wholeDomainChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False,
                 interactive: bool = True) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if interactive else Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.compact = compact
        self.interactive = interactive
        self.show_colorbar = True
        self.show_scalebar = not compact
        self.show_readout = interactive
        self.body_fill = True
        #: keep the body and its near wake in view even if that shows blank space beyond the
        #: domain (3D slices: a flat domain in a portrait viewport would otherwise be cropped
        #: to a fraction of the body by the "no blank space" rule)
        self.keep_body = False
        self.show_particles = True
        self.show_streamlines = True
        # image state
        self._img: QImage | None = None
        self._rgba: np.ndarray | None = None      # keeps the QImage buffer alive
        self._values: np.ndarray | None = None    # raw values, solver order (row 0 = y min)
        self._extent: tuple[float, float, float, float] | None = None
        self._lut: np.ndarray = colormap_lut(None)
        self._vmin, self._vmax = 0.0, 1.0
        self._extend = (False, False)            # colour-bar extend triangles (below, above)
        self._label, self._units = "", ""
        self._caption = ""
        self._title = ""
        self._dark_image = True                   # overlay contrast (white on dark images)
        # overlays (world coordinates)
        self._bodies_path = QPainterPath()
        self._open_lines: list[QPolygonF] = []    # open sheets of a 3D cross-section (stroked only)
        self._axis_names = ("x", "y")             # readout names of the horizontal / vertical axis
        self._stream_lines: list[QPolygonF] = []
        self._particles = QPolygonF()
        self._particle_xy: np.ndarray | None = None       # world positions (N, 2), for coloured drawing
        self._particle_rgb: np.ndarray | None = None      # (N, 3) uint8 per-particle colour (by emitter)
        self._trails: np.ndarray | None = None            # (N, K, 2) streak trails, NaN before release
        self._legend: list[tuple[QColor, str]] = []       # smoke channels: shown instead of the colour bar
        # live editing (widgets.draw_tools): the active tool, glyphs and the drawn obstacles
        self._tool = None
        self._placements: list = []
        self._placement_box: tuple[float, float, float, float] | None = None   # what the framing keeps in view
        self._sketch_lines: list[QPolygonF] = []          # outlines of committed drawn obstacles (world)
        self._sketch_pending = QPainterPath()             # the stroke being drawn (world)
        self._body_extent: tuple[float, float, float, float] | None = None
        self._focus: tuple[float, float, float, float] | None = None
        #: smoke channel -> emitter labels for the frame legend (set by the page that knows the simulation)
        self.channel_labels: dict | None = None
        #: show the whole focus rect (letterboxed) instead of filling the view with it: a closed room
        self._focus_contain = False
        self._ground: float | None = None         # road height (m) for ground runs
        self._whole = False                       # showing the whole domain (Fit toggle)
        # view
        self._scale = 0.0            # pixels per metre (0 = not fitted yet)
        self._cx = self._cy = 0.0    # world point at the centre of the plot area
        self._user_view = False
        self._target_scale = 0.0
        self._anchor_world: tuple[float, float] | None = None
        self._anchor_screen: QPointF | None = None
        self._drag_from: QPointF | None = None
        self._hover: QPointF | None = None
        self._placeholder = ("No flow field", "")
        self._cb_w = 0.0                         # width of the reserved colour-bar strip
        self._anim = QTimer(self)
        self._anim.setInterval(15)
        self._anim.timeout.connect(self._animate_zoom)
        connect_theme(self.update)

    # -- data -------------------------------------------------------------
    def sizeHint(self) -> QSize:
        return QSize(640, 360)

    def has_image(self) -> bool:
        return self._img is not None

    def set_placeholder(self, title: str, subtitle: str = "") -> None:
        """Text shown when no field has been set (e.g. before the first frame)."""
        self._placeholder = (title, subtitle)
        self.update()

    def set_caption(self, text: str) -> None:
        """Small status text in the top-right corner (time, step ...)."""
        self._caption = text
        self.update()

    def caption(self) -> str:
        return self._caption

    def set_title(self, text: str) -> None:
        """Small title in the top-left corner (used by comparison thumbnails)."""
        self._title = text
        self.update()

    def title(self) -> str:
        return self._title

    def clear(self, *, reset_view: bool = True) -> None:
        """Forget the image and overlays; by default the next data re-fits the view."""
        self._img = None
        self._rgba = None
        self._values = None
        self._bodies_path = QPainterPath()
        self._open_lines = []
        self._stream_lines = []
        self._particles = QPolygonF()
        self._particle_xy = self._particle_rgb = self._trails = None
        self._legend = []
        self._caption = ""
        self._update_reserve()
        if reset_view:
            self._focus = None
            self._extent = None
            self._body_extent = None
            self._user_view = False
            self._scale = 0.0
        self.update()

    def set_ground(self, y: float | None) -> None:
        """Road height in metres for ground runs (drawn as a line over a hatched band), None for none."""
        self._ground = None if y is None else float(y)
        if not self._user_view:
            self._apply_fit()
        self.update()

    def ground(self) -> float | None:
        return self._ground

    def set_frame(self, frame) -> None:
        """Show a ``LiveFrame`` (rgba image, extent, colour range, overlays).

        A 3D frame (``frame.slice_axis`` set) is a planar slice: its bodies are
        mesh cross-sections (:meth:`set_sections`), its field names have their
        3D meaning (|ω|, Q) and the readout names the slice's own axes.
        """
        field = getattr(frame, "field", "speed") or ""
        axis = getattr(frame, "slice_axis", None)
        if axis:
            label, unit, _div, default = field_info3d(field)
            lut = colormap_lut(getattr(frame, "cmap", None) or default, field or "speed")
        else:
            label, unit, _ = field_info(field)
            lut = colormap_lut(getattr(frame, "cmap", None), field or "speed")
        units = getattr(frame, "units", None)
        self.set_image(frame.rgba, frame.extent, lut=lut, vmin=frame.vmin, vmax=frame.vmax,
                       label=label, units=unit if units is None else units,
                       extend=frame_clipped_ends(frame.rgba, lut, field=field))
        if axis:
            self.set_axis_names(SLICE_AXIS_NAMES.get(str(axis), ("x", "y")))
            self.set_sections(getattr(frame, "bodies", None) or [])
        else:
            self.set_axis_names(("x", "y"))
            self.set_bodies(getattr(frame, "bodies", None) or [])
        pts, cols, trails = frame_particles2d(frame)
        self.set_particles(pts, colors=cols, trails=trails)
        self.set_streamlines(getattr(frame, "streamlines", None))
        self.set_legend(dye_legend(frame, self.channel_labels))

    def set_axis_names(self, names: Sequence[str]) -> None:
        """Names of the horizontal and vertical world axes in the hover readout ("x", "z" for a side slice)."""
        names = (str(names[0]), str(names[1]))
        if names != self._axis_names:
            self._axis_names = names
            self.update()

    def axis_names(self) -> tuple[str, str]:
        return self._axis_names

    def set_image(self, rgba: np.ndarray, extent: Sequence[float], *, lut: np.ndarray | None = None,
                  vmin: float = 0.0, vmax: float = 1.0, label: str = "", units: str = "",
                  values: np.ndarray | None = None, extend: tuple[bool, bool] = (False, False)) -> None:
        """Show an (H, W, 4) uint8 image (row 0 = top) covering ``extent`` in metres.

        ``extend`` = (below, above) draws the colour bar's extend triangles.
        """
        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
            raise ValueError(f"expected (H, W, 4) uint8 image, got shape {rgba.shape}")
        if rgba.shape[2] == 3:
            rgba = np.ascontiguousarray(np.concatenate(
                [rgba, np.full(rgba.shape[:2] + (1,), 255, np.uint8)], axis=2))
        h, w = rgba.shape[:2]
        self._rgba = rgba
        self._img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        self._values = values
        if lut is not None:
            self._lut = np.asarray(lut, dtype=np.uint8)
        self._vmin, self._vmax = float(vmin), float(vmax)
        self._extend = (bool(extend[0]), bool(extend[1]))
        self._label, self._units = label, units
        step = max(1, (h * w) // 4096)
        sample = rgba.reshape(-1, 4)[::step, :3].astype(np.float32)
        self._dark_image = float(np.mean(sample @ np.array([0.2126, 0.7152, 0.0722], np.float32))) < 150.0
        reserve_changed = self._update_reserve()
        new_extent = tuple(float(e) for e in extent)
        changed = self._extent is None or not np.allclose(new_extent, self._extent, rtol=1e-6, atol=1e-12)
        if self._extent is not None and extent_jumped(self._extent, new_extent):
            # a different study, not a refinement of the same one (a 10 mm cylinder's domain, then a
            # 4 m room): the old zoom and the old body's near-field window mean nothing here. Kept,
            # they framed the room at "2 mm" as one flat rectangle, so the run looked empty.
            self._user_view = False
            if self._body_extent is not None and not boxes_overlap(self._body_extent, new_extent):
                self._body_extent = None
            if self._placement_box is not None and not boxes_overlap(self._placement_box, new_extent):
                self._placement_box = None
            if self._focus is not None and not boxes_overlap(self._focus, new_extent):
                self._focus = None
        self._extent = new_extent  # type: ignore[assignment]
        if (changed or reserve_changed) and (not self._user_view or self._scale == 0.0):
            self._apply_fit()
        self.update()

    def set_field(self, values: np.ndarray, extent: Sequence[float], *, field: str = "",
                  cmap: str | None = None, vmin: float | None = None, vmax: float | None = None,
                  mask: np.ndarray | None = None, label: str | None = None,
                  units: str | None = None, window: Sequence[float] | None = None) -> tuple[float, float]:
        """Colourise a solver-ordered (ny, nx) array and show it. Returns the (vmin, vmax) used.

        Missing limits come from :func:`display_range`, sampled in ``window``
        (metres), else around the bodies already set (:func:`body_window`).
        """
        values = np.asarray(values, dtype=np.float32)
        lab, unit, _div = field_info(field)
        if vmin is None or vmax is None:
            if window is None and self._body_extent is not None:
                bx0, bx1, by0, by1 = self._body_extent
                window = body_window([np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])], extent)
            a, b = display_range(field, values, extent=extent, mask=mask, window=window)
            vmin = a if vmin is None else vmin
            vmax = b if vmax is None else vmax
        lut = colormap_lut(cmap, field)
        rgba = colorize(values, lut, vmin, vmax, mask=mask)
        extend = clipped_ends(values, vmin, vmax, mask=mask)
        if mask is not None and np.shape(mask) == values.shape:
            values = np.where(np.asarray(mask) >= 0.5, np.nan, values)
        self.set_image(rgba, extent, lut=lut, vmin=vmin, vmax=vmax,
                       label=lab if label is None else label, units=unit if units is None else units,
                       values=values, extend=extend)
        return float(vmin), float(vmax)

    def set_bodies(self, polygons: Sequence[np.ndarray]) -> None:
        """Solid outlines (metres) drawn filled on top of the field (2D bodies: implicitly closed)."""
        self._set_solids(polygons, [])

    def set_sections(self, lines: Sequence[np.ndarray]) -> None:
        """Cross-sections of 3D meshes with the slice plane (see :func:`section_paths`).

        Closed loops are drawn like 2D bodies (filled grey, dark outline); open
        chains of sheet-like parts are drawn as dark 1.5 px lines.
        """
        closed, open_ = section_paths(lines)
        self._set_solids(closed, open_)

    def _set_solids(self, polygons: Sequence[np.ndarray], chains: Sequence[np.ndarray]) -> None:
        path = QPainterPath()
        pts_all = []
        for poly in polygons or []:
            p = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(p) < 3:
                continue
            path.addPolygon(to_qpolygonf(p))
            path.closeSubpath()
            pts_all.append(p)
        path.setFillRule(Qt.FillRule.WindingFill)
        self._bodies_path = path
        self._open_lines = []
        for ln in chains or []:
            p = np.asarray(ln, dtype=float).reshape(-1, 2)
            if len(p) >= 2:
                self._open_lines.append(to_qpolygonf(p))
                pts_all.append(p)
        if pts_all:
            a = np.vstack(pts_all)
            box = (float(a[:, 0].min()), float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 1].max()))
            # the default framing follows the bodies; re-fit only when they actually moved
            # (explore mode turns them every frame the angle changes, runs never do)
            moved = self._body_extent is None or not np.allclose(box, self._body_extent, rtol=0, atol=1e-9)
            self._body_extent = box
            if moved and not self._user_view:
                self._apply_fit()
        elif self._body_extent is not None:
            # no bodies now (a closed room or cavity after a cylinder): the stale body box kept the
            # default framing on a 10 mm window inside a 4 m room. Without bodies it is the whole domain.
            self._body_extent = None
            if not self._user_view:
                self._apply_fit()
        self.update()

    def set_particles(self, positions: np.ndarray | None, *, colors: np.ndarray | None = None,
                      trails: np.ndarray | None = None) -> None:
        """(N, 2) tracer positions in metres, or None.

        ``colors`` (N, 3) uint8 draws each particle in its emitter's colour
        (DESIGN.md 10.1) instead of the image-contrast overlay colour;
        ``trails`` (N, K, 2), oldest first and NaN before release, adds a short
        fading streak behind each particle.
        """
        arr = None if positions is None else np.asarray(positions, dtype=float)
        if arr is not None and arr.ndim == 2 and arr.shape[1] > 2:
            arr = arr[:, :2]                 # (N, 3) world points: this view draws their first two axes
        if arr is None or arr.size == 0 or arr.size % 2:
            self._particles = QPolygonF()
            self._particle_xy = self._particle_rgb = self._trails = None
        else:
            positions = arr
            pts = arr.reshape(-1, 2)[:20000]
            self._particles = to_qpolygonf(pts)
            self._particle_xy = pts
            ok = colors is not None and np.shape(colors)[:1] == (len(np.asarray(positions).reshape(-1, 2)),)
            self._particle_rgb = np.asarray(colors, dtype=np.uint8)[:20000] if ok else None
            tr = np.asarray(trails, dtype=float) if trails is not None else None
            self._trails = tr[:20000] if tr is not None and tr.ndim == 3 and tr.shape[0] >= len(pts) \
                and tr.shape[1] >= 2 else None
        self.update()

    def set_legend(self, items: Sequence[tuple] | None) -> None:
        """Swatches shown in place of the colour bar: [(colour, text), ...] (smoke channels), None for none."""
        legend = [(QColor(c) if not isinstance(c, QColor) else c, str(t)) for c, t in (items or [])]
        if [(c.name(), t) for c, t in legend] != [(c.name(), t) for c, t in self._legend]:
            self._legend = legend
            if self._update_reserve() and (not self._user_view or self._scale == 0.0):
                self._apply_fit()
            self.update()

    # -- live editing (widgets.draw_tools) ---------------------------------
    def set_tool(self, tool) -> None:
        """Route the left mouse button through a draw_tools tool (None: plain pan / zoom)."""
        if tool is self._tool:
            return
        if self._tool is not None:
            self._tool.leave(self)
        self._tool = tool
        self.unsetCursor()
        cur = tool.cursor(self) if tool is not None else None
        if cur is not None:
            self.setCursor(cur)
        self.update()

    def tool(self):
        return self._tool

    def set_placements(self, placements: Sequence, *, refit: bool = False) -> None:
        """Emitter / fan / heater / port glyphs (:class:`~aerolume.app.widgets.draw_tools.Placement`).

        The default framing keeps them in view (:meth:`default_window`): a smoke rake placed a
        body length upstream, or a heater in the wake, sat outside the near-field window, so a
        user saw the smoke but not where it came from. ``refit`` re-frames now (a new
        simulation); without it the view stays put (a glyph being dragged must not move the view
        under the mouse) and the next automatic fit includes the new positions.
        """
        self._placements = list(placements or [])
        self._placement_box = placements_box(self._placements)
        if refit and not self._user_view:
            self._apply_fit()
        self.update()

    def placements(self) -> list:
        return list(self._placements)

    def set_sketch(self, outlines: Sequence[np.ndarray] | None, pending: np.ndarray | None = None) -> None:
        """Drawn obstacles: committed ones as outlines (metres), the stroke in progress as a filled preview.

        The solver turns committed drawings into solid cells, which the frame
        then paints grey like any body; the dashed outline keeps "drawn here"
        distinguishable from the design's own geometry. The pending stroke is
        filled translucently so the brush responds before the solver has seen it.
        """
        self._sketch_lines = [to_qpolygonf(np.asarray(p, float).reshape(-1, 2)) for p in (outlines or [])
                              if len(p) >= 3]
        path = QPainterPath()
        if pending is not None:
            for poly in pending:
                q = np.asarray(poly, float).reshape(-1, 2)
                if len(q) >= 3:
                    path.addPolygon(to_qpolygonf(q))
                    path.closeSubpath()
        path.setFillRule(Qt.FillRule.WindingFill)
        self._sketch_pending = path
        self.update()

    def set_streamlines(self, lines: Sequence[np.ndarray] | None) -> None:
        """Polylines in metres, or None."""
        polys = []
        for line in lines or []:
            ln = np.asarray(line, dtype=float).reshape(-1, 2)
            if len(ln) >= 2:
                polys.append(to_qpolygonf(ln))
        self._stream_lines = polys
        self.update()

    def colour_range(self) -> tuple[float, float]:
        return self._vmin, self._vmax

    def extend(self) -> tuple[bool, bool]:
        """(below, above): whether the colour bar shows extend triangles."""
        return self._extend

    def extent(self) -> tuple[float, float, float, float] | None:
        return self._extent

    # -- layout -----------------------------------------------------------
    def _update_reserve(self) -> bool:
        """Recompute the colour-bar strip width; True if it changed (the plot area moved)."""
        w = colorbar_width(self._label, self._units, self._vmin, self._vmax) \
            if (self.show_colorbar and self._img is not None) else 0.0
        if w and self._legend:
            fm = QFontMetricsF(_ui_font(8.0))
            text_w = max(fm.horizontalAdvance(t) for _c, t in self._legend) + 24 + 18
            title_w = QFontMetricsF(_ui_font(8.0, bold=True)).horizontalAdvance(self._label or "Smoke") + 24
            w = float(min(max(text_w, title_w, 90.0), 220.0))
        if self.compact and w:
            w = min(w, 120.0)
        changed = abs(w - self._cb_w) > 0.5
        self._cb_w = w
        return changed

    def plot_rect(self) -> QRectF:
        """The part of the widget the field is drawn in (left of the colour-bar strip)."""
        return QRectF(0, 0, max(self.width() - self._cb_w, 8.0), self.height())

    def colorbar_reserve(self) -> float:
        """Width [px] the colour bar (or smoke legend) takes at the right; containers that size a view to
        its field's aspect ratio add it to the image width."""
        return float(self._cb_w)

    # -- view transform ---------------------------------------------------
    def world_to_screen(self) -> QTransform:
        """World (metres, y up) -> widget pixels (y down)."""
        s = self._scale or 1.0
        r = self.plot_rect()
        return QTransform(s, 0, 0, -s, r.center().x() - self._cx * s, r.center().y() + self._cy * s)

    def screen_to_world(self, p: QPointF) -> tuple[float, float]:
        s = self._scale or 1.0
        c = self.plot_rect().center()
        return (self._cx + (p.x() - c.x()) / s, self._cy - (p.y() - c.y()) / s)

    def set_focus_rect(self, rect: Sequence[float] | None, *, contain: bool = False) -> None:
        """Window (xmin, xmax, ymin, ymax) [m] that the default framing uses.

        Without one the window is :func:`near_field_window` of the bodies, so
        every page frames the same flow the same way; pages only set this to
        frame something else. ``contain`` shows all of the window, letterboxed
        when its aspect differs from the view's (a closed room: its walls, fan,
        intake and exhaust all in view), instead of filling the view with it
        and cropping its longer side. Returns the view to the default framing.
        """
        self._focus = tuple(float(v) for v in rect) if rect is not None else None
        self._focus_contain = bool(contain) and self._focus is not None
        self.fit()

    def fit(self) -> None:
        """Return to the default framing: the body's near field, filling the plot area."""
        was_whole = self._whole
        self._whole = False
        self._apply_fit()
        if was_whole:
            self.wholeDomainChanged.emit(False)

    def set_whole_domain(self, on: bool) -> None:
        """Show the whole computational domain (letterboxed if need be), or return to the default framing."""
        on = bool(on)
        changed = on != self._whole
        self._whole = on
        self._apply_fit()
        if changed:
            self.wholeDomainChanged.emit(on)

    def whole_domain(self) -> bool:
        return self._whole

    def view_state(self) -> tuple[float, float, float]:
        """(pixels per metre, centre x, centre y) of the current view (linked views copy it)."""
        return self._scale, self._cx, self._cy

    def is_user_view(self) -> bool:
        """True after the user zoomed or panned (the view no longer auto-fits)."""
        return bool(self._user_view)

    def set_view_state(self, state: Sequence[float]) -> None:
        """Show the same window as another view (:meth:`view_state`); counts as a user view (no auto-fit)."""
        s, cx, cy = (float(v) for v in state)
        if not (s > 0 and math.isfinite(s) and math.isfinite(cx) and math.isfinite(cy)):
            return
        self._anim.stop()
        self._scale = self._target_scale = s
        self._cx, self._cy = cx, cy
        self._user_view = True
        self.update()

    def default_window(self) -> tuple[float, float, float, float] | None:
        """The window the default framing shows: the focus rect, else the bodies' near field grown to
        take in the placed emitters, fans and heaters (:meth:`set_placements`)."""
        if self._focus is not None:
            return self._focus
        if self._body_extent is None:
            return None
        bx0, bx1, by0, by1 = self._body_extent
        box = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
        win = near_field_window([box], self._extent, ground=self._ground)
        pb = self._padded_placements()
        if win is None or pb is None:
            return win
        return (min(win[0], pb[0]), max(win[1], pb[1]), min(win[2], pb[2]), max(win[3], pb[3]))

    def _padded_placements(self) -> tuple[float, float, float, float] | None:
        """The placements' box with room for their arrows and handles: a quarter body length around it."""
        pb = self._placement_box
        if pb is None:
            return None
        if self._body_extent is not None:
            bx0, bx1, by0, by1 = self._body_extent
            pad = 0.25 * max(bx1 - bx0, by1 - by0, 1e-9)
        else:
            pad = 0.05 * max(pb[1] - pb[0], pb[3] - pb[2], 1e-9)
        return pb[0] - pad, pb[1] + pad, pb[2] - pad, pb[3] + pad

    def framing_aspect(self) -> float | None:
        """Width / height of what the default framing shows: the near-field window inside the domain,
        else the domain; None before any data. Pages use it to shape the pane (:func:`split_orientation`)."""
        ext = self._extent
        win = self.default_window()
        if win is not None and ext is not None:
            win = (max(win[0], ext[0]), min(win[1], ext[1]), max(win[2], ext[2]), min(win[3], ext[3]))
        rect = win if win is not None and win[1] > win[0] and win[3] > win[2] else ext
        if rect is None or rect[3] <= rect[2]:
            return None
        return float((rect[1] - rect[0]) / (rect[3] - rect[2]))

    def _road_px(self) -> float:
        """Pixels the view may show below the domain: the road band, for ground runs whose domain ends there."""
        if self._ground is None or self._extent is None:
            return 0.0
        return ROAD_BAND_PX + 4.0 if abs(self._extent[2] - self._ground) < 1e-6 * max(
            self._extent[3] - self._extent[2], 1e-9) + 1e-9 else 0.0

    def _apply_fit(self) -> None:
        """Set scale and centre for the current mode (whole domain or default framing)."""
        r = self.plot_rect()
        if r.width() < 4 or r.height() < 4:
            return
        ext = self._extent
        win = None if self._whole else self.default_window()
        if win is not None and ext is not None:
            clipped = (max(win[0], ext[0]), min(win[1], ext[1]), max(win[2], ext[2]), min(win[3], ext[3]))
            win = clipped if clipped[1] > clipped[0] and clipped[3] > clipped[2] else None
        if self._img is None or ext is None:
            # geometry-only preview (or no data yet): contain the window with a little air
            if win is None and ext is None and self._body_extent is not None:
                bx0, bx1, by0, by1 = self._body_extent
                span = max(bx1 - bx0, by1 - by0, 1e-9)
                win = (bx0 - 0.6 * span, bx1 + 0.9 * span, by0 - 0.45 * span, by1 + 0.45 * span)
            target = win or ext
            if target is None:
                return
            self._set_view(*self._contain(target, r))
            return
        if win is None:
            # the whole domain: contained, letterboxed if its aspect ratio differs from the view's
            self._set_view(*self._contain(ext, r))
            return
        if self._focus_contain and self._focus is not None:
            self._set_view(*self._contain(win, r))
            return
        self._set_view(*self._cover(win, ext, r))

    def _contain(self, rect, r: QRectF) -> tuple[float, float, float]:
        x0, x1, y0, y1 = rect
        pad = 0.98 if self.compact else 0.96
        s = min(r.width() * pad / max(x1 - x0, 1e-9), r.height() * pad / max(y1 - y0, 1e-9))
        return s, (x0 + x1) / 2, (y0 + y1) / 2

    def _cover(self, win, ext, r: QRectF) -> tuple[float, float, float]:
        """Scale and centre that fill the plot area with ``win`` cropped to its aspect ratio.

        Three rules, in order of priority: never show blank space beyond the
        domain (except the road band under a ground run); keep the body with
        :data:`ESSENTIAL_MARGINS` in view; otherwise crop the window's longer
        side (cover rather than contain, so there is no letterboxing).
        """
        rw, rh = r.width(), r.height()
        wx0, wx1, wy0, wy1 = win
        s = max(rw / max(wx1 - wx0, 1e-9), rh / max(wy1 - wy0, 1e-9))      # cover the window
        cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
        ess = None
        if self._body_extent is not None:
            bx0, bx1, by0, by1 = self._body_extent
            L = max(bx1 - bx0, by1 - by0, 1e-9)
            up, down, side = ESSENTIAL_MARGINS
            ess = (bx0 - up * L, bx1 + down * L, by0 - side * L, by1 + side * L)
            if self._ground is not None:
                ess = (ess[0], ess[1], max(ess[2], self._ground), ess[3])
            pb = self._padded_placements() if self._focus is None else None
            if pb is not None:           # placed devices are essential too: cropping must not cut them off
                ess = (min(ess[0], pb[0]), max(ess[1], pb[1]), min(ess[2], pb[2]), max(ess[3], pb[3]))
            # only what lies inside the requested window: a caller framing a detail gets that detail
            ess = (max(ess[0], wx0), min(ess[1], wx1), max(ess[2], wy0), min(ess[3], wy1))
            if ess[1] > ess[0] and ess[3] > ess[2]:
                s = min(s, rw / (ess[1] - ess[0]), rh / (ess[3] - ess[2]))
            else:
                ess = None
        below = self._road_px()
        x0, x1, y0, y1 = ext
        if self.keep_body and ess is not None and s * (y1 - y0) + below < rh - 1:
            # the essential window leaves blank bands above and below a flat domain (a car's
            # side slice in a portrait pane filled a 90-140 px band): zoom in towards
            # KEEP_BODY_MARGINS around the body, no further than the domain filling the height
            bx0, bx1, by0, by1 = self._body_extent
            L = max(bx1 - bx0, by1 - by0, 1e-9)
            up, down = KEEP_BODY_MARGINS
            tight = (bx0 - up * L, bx1 + down * L, by0 - up * L, by1 + up * L)
            if self._ground is not None:
                tight = (tight[0], tight[1], max(tight[2], self._ground), tight[3])
            tight = (max(tight[0], wx0, x0), min(tight[1], wx1, x1), max(tight[2], wy0, y0), min(tight[3], wy1, y1))
            if tight[1] > tight[0] and tight[3] > tight[2]:
                s_tight = min(rw / (tight[1] - tight[0]), rh / (tight[3] - tight[2]))
                s_fill = (rh - below) / max(y1 - y0, 1e-9)
                s_new = max(s, min(s_tight, s_fill))
                if s_new > s:
                    s, ess = s_new, tight
        # the domain must cover the plot area: a lower bound on the scale (unless the body has priority)
        if not (self.keep_body and ess is not None):
            s = max(s, rw / max(x1 - x0, 1e-9), (rh - below) / max(y1 - y0, 1e-9))
        hw, hh = rw / (2 * s), rh / (2 * s)
        lo_y = y0 - below / s
        if ess is not None:   # move the window just enough that the essential box is in view
            e_lo = lo_y if below else ess[2]       # a ground run shows its road band, not just the road
            cx = min(max(cx, ess[1] - hw), ess[0] + hw)
            cy = min(max(cy, ess[3] - hh), e_lo + hh)
        cx = min(max(cx, x0 + hw), x1 - hw) if x1 - x0 >= 2 * hw else (x0 + x1) / 2
        cy = min(max(cy, lo_y + hh), y1 - hh) if y1 - lo_y >= 2 * hh else (lo_y + y1) / 2
        return s, cx, cy

    def _set_view(self, s: float, cx: float, cy: float) -> None:
        self._scale = s
        self._target_scale = s
        self._cx, self._cy = cx, cy
        self._user_view = False
        self._anim.stop()
        self.update()
        self.viewChanged.emit()

    def _user_took_over(self) -> None:
        """A zoom or pan by the user: stop auto-fitting, and the whole-domain toggle no longer holds."""
        self._user_view = True
        if self._whole:
            self._whole = False
            self.wholeDomainChanged.emit(False)

    def zoom_by(self, factor: float, anchor: QPointF | None = None, *, animate: bool = True) -> None:
        """Zoom by ``factor`` keeping the world point under ``anchor`` (widget px) fixed."""
        if self._scale == 0.0:
            return
        anchor = anchor or self.plot_rect().center()
        base = self._target_scale if self._anim.isActive() else self._scale
        self._target_scale = float(np.clip(base * factor, 1e-6, 1e9))
        self._anchor_screen = anchor
        self._anchor_world = self.screen_to_world(anchor)
        self._user_took_over()
        if animate:
            if not self._anim.isActive():
                self._anim.start()
        else:
            self._apply_scale(self._target_scale)

    def _apply_scale(self, s_new: float) -> None:
        if self._anchor_world is None or self._anchor_screen is None:
            self._scale = s_new
        else:
            ax, ay = self._anchor_world
            p = self._anchor_screen
            c = self.plot_rect().center()
            self._scale = s_new
            self._cx = ax - (p.x() - c.x()) / s_new
            self._cy = ay + (p.y() - c.y()) / s_new
        self.update()
        self.viewChanged.emit()

    def _animate_zoom(self) -> None:
        # exponential approach in log-scale: smooth and frame-rate independent enough
        r = self._target_scale / self._scale
        if abs(math.log(r)) < 0.004:
            self._apply_scale(self._target_scale)
            self._anim.stop()
            return
        self._apply_scale(self._scale * r ** 0.35)

    # -- events -----------------------------------------------------------
    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if not self._user_view or self._scale == 0.0:
            self._apply_fit()

    def wheelEvent(self, ev) -> None:
        if not self.interactive:
            return ev.ignore()
        steps = ev.angleDelta().y() / 120.0
        if steps:
            self.zoom_by(1.2 ** steps, ev.position())
        ev.accept()

    def mousePressEvent(self, ev) -> None:
        if self.interactive and self._tool is not None and self._scale and self._tool.mouse_press(self, ev):
            ev.accept()
            return
        if self.interactive and ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_from = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        pos = ev.position()
        if self._drag_from is None and self._tool is not None and self._scale and self._tool.mouse_move(self, ev):
            self._hover = pos if self.plot_rect().contains(pos) else None
            x, y = self.screen_to_world(pos)
            self.hovered.emit(x, y, self.value_at(x, y))
            self.update()
            return
        if self._drag_from is not None and self._scale:
            d = pos - self._drag_from
            self._drag_from = pos
            self._cx -= d.x() / self._scale
            self._cy += d.y() / self._scale
            self._user_took_over()
            if self._anchor_world is not None and self._anim.isActive():
                # keep an in-flight zoom anchored to the same screen point while panning
                self._anchor_world = self.screen_to_world(self._anchor_screen)
            self.viewChanged.emit()
        self._hover = pos if self.plot_rect().contains(pos) else None
        x, y = self.screen_to_world(pos)
        self.hovered.emit(x, y, self.value_at(x, y))
        self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag_from is None and self._tool is not None and self._tool.mouse_release(self, ev):
            ev.accept()
            return
        self._drag_from = None
        self.unsetCursor()
        cur = self._tool.cursor(self) if self._tool is not None else None
        if cur is not None:
            self.setCursor(cur)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        if self.interactive and self._tool is not None and self._tool.double_click(self, ev):
            return
        if self.interactive:
            self.fit()

    def leaveEvent(self, ev) -> None:
        self._hover = None
        if self._tool is not None:
            self._tool.leave(self)
        self.update()
        super().leaveEvent(ev)

    def keyPressEvent(self, ev) -> None:
        if self._tool is not None and self._tool.key_press(self, ev):
            ev.accept()
            return
        if ev.key() == Qt.Key.Key_F:
            self.fit()
        elif ev.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.25)
        elif ev.key() == Qt.Key.Key_Minus:
            self.zoom_by(0.8)
        else:
            super().keyPressEvent(ev)

    # -- readout ----------------------------------------------------------
    def value_at(self, x: float, y: float) -> float | None:
        """Field value at a world point: raw data if present, else inverted from the colour."""
        if self._extent is None or self._img is None:
            return None
        x0, x1, y0, y1 = self._extent
        if not (x0 <= x < x1 and y0 <= y < y1):
            return None
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        if self._values is not None:
            ny, nx = self._values.shape
            v = float(self._values[min(int(fy * ny), ny - 1), min(int(fx * nx), nx - 1)])
            return v if math.isfinite(v) else None
        h, w = self._rgba.shape[:2]
        rgb = self._rgba[min(int((1 - fy) * h), h - 1), min(int(fx * w), w - 1), :3].astype(int)
        d = np.sum((self._lut.astype(int) - rgb) ** 2, axis=1)
        i = int(np.argmin(d))
        if d[i] > 3 * 12 ** 2:  # not a colour of this map: solid body or overlay
            return None
        return self._vmin + (self._vmax - self._vmin) * i / (len(self._lut) - 1)

    # -- painting ---------------------------------------------------------
    def _overlay_colour(self, alpha: int) -> QColor:
        return QColor(255, 255, 255, alpha) if self._dark_image else QColor(15, 17, 20, alpha)

    def paintEvent(self, ev) -> None:
        tk = tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), qcolor(tk["surface"]))
        area = self.plot_rect()
        if self._img is None and self._bodies_path.isEmpty() and not self._open_lines:
            self._paint_placeholder(p, tk, area)
            p.end()
            return
        if self._img is None:  # geometry preview: bodies plus the hint underneath
            self._paint_placeholder(p, tk, area, low=True)
        if self._scale == 0.0:
            self._apply_fit()
        xf = self.world_to_screen()
        p.save()
        p.setClipRect(area)
        if self._img is not None and self._extent is not None:
            x0, x1, y0, y1 = self._extent
            target = xf.mapRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            p.drawImage(target, self._img)
            p.setPen(QPen(qcolor(tk["border_strong"]), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(target.adjusted(-0.5, -0.5, 0.5, 0.5))
        if self.show_streamlines and self._stream_lines:
            # mapped to pixels and drawn as polylines: stroking one big world-space path with a
            # cosmetic pen costs ~300 ms for 80 lines, this costs a few ms
            p.setPen(QPen(self._overlay_colour(150), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in self._stream_lines:
                p.drawPolyline(xf.map(poly))
        if self._ground is not None:
            self._paint_road(p, tk, area, xf)
        if not self._bodies_path.isEmpty():
            p.save()
            p.setTransform(xf, combine=True)
            pen = QPen(QColor(*SOLID_EDGE), 1.0)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(QColor(*SOLID_FILL) if self.body_fill else Qt.BrushStyle.NoBrush)
            p.drawPath(self._bodies_path)
            p.restore()
        if self._open_lines:
            # sheets have no inside: a dark line of constant pixel width, mapped to pixels first
            pen = QPen(QColor(*SOLID_EDGE), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in self._open_lines:
                p.drawPolyline(xf.map(poly))
        if self.show_particles and not self._particles.isEmpty():
            self._paint_particles(p, xf)
        if self._sketch_lines or not self._sketch_pending.isEmpty():
            self._paint_sketch(p, tk, xf)
        if self._placements:
            from .draw_tools import paint_placements
            mv = self._tool if hasattr(self._tool, "placements") else None
            paint_placements(p, xf, self._placements, dark_image=self._dark_image,
                             active=getattr(mv, "drag", None), hover=getattr(mv, "hover", None))
        if self._tool is not None:
            self._tool.paint(self, p)
        p.restore()
        if self._title:
            self._paint_label(p, tk, self._title, area, "top-left", bold=True)
        if self._caption:
            self._paint_label(p, tk, self._caption, area, "top-right")
        if self.show_scalebar and self._scale > 0:
            self._paint_scalebar(p, tk, area)
        if self.show_readout and self._hover is not None and self._scale > 0:
            self._paint_readout(p, tk, area)
        if self._cb_w > 0 and self._img is not None:
            strip = QRectF(area.right(), 0, self._cb_w, self.height())
            if self._legend:
                paint_legend(p, strip, self._legend, self._label, tk)
            else:
                paint_colorbar(p, strip, self._lut, self._vmin, self._vmax, self._label, self._units, tk,
                               self._extend)
        p.end()

    def _paint_particles(self, p: QPainter, xf: QTransform) -> None:
        """Tracer dots (and their streak trails), in their emitters' colours when the frame gives them.

        Points are mapped to pixels first: a cosmetic pen under the flipped,
        scaled world transform draws each point as a short smear instead of a
        dot. Trails are one path per colour built from numpy arrays
        (``arrayToQPath``), so a few thousand trails cost a few milliseconds.
        """
        size = 2.0 if not self.compact else 1.4
        pts = self._particle_xy
        rgb = self._particle_rgb
        if self._trails is not None and pts is not None:
            self._paint_trails(p, xf, rgb)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)   # 2 px dots: AA doubles the cost
        if rgb is None or pts is None:
            pen = QPen(self._overlay_colour(210), size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawPoints(xf.map(self._particles))
        else:
            keys = rgb[:, 0].astype(np.int64) << 16 | rgb[:, 1].astype(np.int64) << 8 | rgb[:, 2]
            for k in np.unique(keys):
                sel = keys == k
                pen = QPen(QColor(int(k >> 16) & 255, int(k >> 8) & 255, int(k) & 255, 230), size + 0.4)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawPoints(xf.map(to_qpolygonf(pts[sel])))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _paint_trails(self, p: QPainter, xf: QTransform, rgb: np.ndarray | None) -> None:
        tr = self._trails[: len(self._particle_xy)]
        n, k = tr.shape[:2]
        if n == 0 or k < 2:
            return
        sx, sy, dx, dy = xf.m11(), xf.m22(), xf.dx(), xf.dy()
        x = tr[:, :, 0] * sx + dx
        y = tr[:, :, 1] * sy + dy
        valid = np.isfinite(x) & np.isfinite(y)
        # connect point i to i+1 of the same particle when both are valid; the last of each never connects
        conn = valid[:, :-1] & valid[:, 1:]
        conn = np.concatenate([conn, np.zeros((n, 1), bool)], axis=1)
        x, y = np.where(valid, x, 0.0), np.where(valid, y, 0.0)
        groups = [(None, np.ones(n, bool))] if rgb is None else []
        if rgb is not None:
            keys = rgb[:, 0].astype(np.int64) << 16 | rgb[:, 1].astype(np.int64) << 8 | rgb[:, 2]
            groups = [(int(c), keys == c) for c in np.unique(keys)]
        for c, sel in groups:
            if not sel.any():
                continue
            path = pg_functions.arrayToQPath(x[sel].ravel(), y[sel].ravel(), connect=conn[sel].ravel())
            col = self._overlay_colour(90) if c is None else QColor((c >> 16) & 255, (c >> 8) & 255, c & 255, 110)
            pen = QPen(col, 1.0)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

    def _paint_sketch(self, p: QPainter, tk: dict, xf: QTransform) -> None:
        """Drawn obstacles: a dashed accent outline per committed shape, the pending stroke filled."""
        accent = qcolor(tk["accent"])
        if not self._sketch_pending.isEmpty():
            fill = QColor(accent)
            fill.setAlpha(70)
            p.save()
            p.setTransform(xf, combine=True)
            pen = QPen(accent, 1.0)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(fill)
            p.drawPath(self._sketch_pending)
            p.restore()
        if self._sketch_lines:
            pen = QPen(accent, 1.4)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for poly in self._sketch_lines:
                p.drawPolygon(xf.map(poly))

    def _box(self, p: QPainter, rect: QRectF, tk: dict) -> None:
        """A small annotation box: surface fill (slightly translucent), 1 px border, 2 px radius."""
        p.setPen(QPen(qcolor(tk["border"]), 1))
        p.setBrush(qcolor(tk["surface"], 230))
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)

    def _overlay_bottom(self, area: QRectF) -> float:
        """Lowest y for the scale bar and readout boxes: above the road band of a ground run."""
        if self._ground is None or self._scale <= 0:
            return area.bottom()
        sy = self.world_to_screen().map(QPointF(0.0, self._ground)).y()
        return min(area.bottom(), sy) if sy > area.top() + 0.5 * area.height() else area.bottom()

    def _paint_road(self, p: QPainter, tk: dict, area: QRectF, xf: QTransform) -> None:
        """The road at the ground height: a 1 px text2 line over a hatched band (Geometry's style).

        Everything below the road is covered with the surface colour first, so
        a ground run never shows field colours (or white) under the road, and
        the band is clipped to the image's width when the domain is narrower
        than the view.
        """
        sy = round(xf.map(QPointF(0.0, self._ground)).y()) + 0.5
        if sy > area.bottom() + 1 or sy < area.top() - 1:
            return
        left, right = area.left(), area.right()
        if self._extent is not None and self._img is not None:
            x0, x1 = self._extent[0], self._extent[1]
            left = max(left, xf.map(QPointF(x0, 0.0)).x())
            right = min(right, xf.map(QPointF(x1, 0.0)).x())
        if right <= left:
            return
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.fillRect(QRectF(left, sy, right - left, area.bottom() - sy + 1), qcolor(tk["surface"]))
        band = max(min(ROAD_BAND_PX, area.bottom() - sy), 0.0)
        if band > 0:
            p.setClipRect(QRectF(left, sy, right - left, band), Qt.ClipOperation.IntersectClip)
            p.setPen(QPen(qcolor(tk["border_strong"]), 1))
            for x in np.arange(left - band - (left % 7.0), right + band, 7.0):
                p.drawLine(QPointF(x, sy + band), QPointF(x + band, sy))
            p.setClipping(False)
        p.setClipRect(area)
        p.setPen(QPen(qcolor(tk["text2"]), 1))
        p.drawLine(QPointF(left, sy), QPointF(right, sy))
        p.restore()

    def _paint_placeholder(self, p: QPainter, tk: dict, area: QRectF, low: bool = False) -> None:
        title, sub = self._placeholder
        r = area.adjusted(24, 0, -24, 0)
        f_title = _ui_font(9.0, bold=True)
        f_sub = _ui_font(9.0)
        fm = QFontMetricsF(f_title)
        mid = r.center().y() - (fm.height() if sub else fm.height() / 2)
        if low:
            mid = r.top() + r.height() * 0.82 - fm.height()
        p.setFont(f_title)
        p.setPen(qcolor(tk["text2"]))
        p.drawText(QRectF(r.left(), mid, r.width(), fm.height() * 1.3),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, title)
        if sub:
            p.setFont(f_sub)
            p.setPen(qcolor(tk["muted"]))
            p.drawText(QRectF(r.left(), mid + fm.height() * 1.4, r.width(), 60),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                       sub)

    def _paint_label(self, p: QPainter, tk: dict, text: str, area: QRectF, corner: str,
                     bold: bool = False) -> None:
        f = _ui_font(8.0, bold=bold)
        p.setFont(f)
        fm = QFontMetricsF(f)
        w = min(fm.horizontalAdvance(text) + 12, area.width() - 16)
        h = fm.height() + 6
        x = area.left() + 8 if corner == "top-left" else area.right() - 8 - w
        r = QRectF(x, area.top() + 8, w, h)
        self._box(p, r, tk)
        p.setPen(qcolor(tk["text"] if bold else tk["text2"]))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, fm.elidedText(text, Qt.TextElideMode.ElideRight, w - 10))

    def _scalebar_corners(self, area: QRectF, w: float, h: float) -> list[QRectF]:
        """Candidate boxes for the scale bar, in order of preference: bottom-left, bottom-right,
        top-left and top-right (the top ones below the title and caption labels)."""
        label_h = QFontMetricsF(_ui_font(8.0)).height() + 6
        bottom = self._overlay_bottom(area) - 8 - h
        top_l = area.top() + 8 + (label_h + 6 if self._title else 0)
        top_r = area.top() + 8 + (label_h + 6 if self._caption else 0)
        left, right = area.left() + 8, area.right() - 8 - w
        return [QRectF(left, bottom, w, h), QRectF(right, bottom, w, h),
                QRectF(left, top_l, w, h), QRectF(right, top_r, w, h)]

    def _solid_share(self, rect: QRectF) -> float:
        """How much of a screen rectangle the body covers: 1 when it touches the body outline, else
        the share of masked (solid) cells under it, from the raw values (NaN = solid)."""
        if rect.width() <= 0 or rect.height() <= 0:
            return 0.0
        xf = self.world_to_screen()
        if not self._bodies_path.isEmpty() and xf.map(self._bodies_path).intersects(rect):
            return 1.0
        for poly in self._open_lines:
            if xf.map(poly).boundingRect().intersects(rect):
                return 1.0
        vals, ext = self._values, self._extent
        if vals is None or ext is None or vals.ndim != 2:
            return 0.0
        ny, nx = vals.shape
        xa, ya = self.screen_to_world(rect.topLeft())
        xb, yb = self.screen_to_world(rect.bottomRight())
        x0, x1, y0, y1 = ext
        i0 = int(np.clip((min(xa, xb) - x0) / (x1 - x0) * nx, 0, nx))
        i1 = int(np.clip(math.ceil((max(xa, xb) - x0) / (x1 - x0) * nx), 0, nx))
        j0 = int(np.clip((min(ya, yb) - y0) / (y1 - y0) * ny, 0, ny))
        j1 = int(np.clip(math.ceil((max(ya, yb) - y0) / (y1 - y0) * ny), 0, ny))
        if i1 <= i0 or j1 <= j0:
            return 0.0
        return float(np.mean(~np.isfinite(vals[j0:j1, i0:i1])))

    def _scalebar_rect(self, area: QRectF, w: float, h: float) -> QRectF:
        """The scale bar's box: the first corner (bottom-left first) that keeps clear of the body.

        A fixed bottom-left box sat on a car's body in a short Split pane; the
        box goes to the corner with the fewest solid cells under it, never over
        the body outline while a clear corner exists.
        """
        corners = self._scalebar_corners(area, w, h)
        best, best_share = corners[0], 2.0
        for r in corners:
            share = self._solid_share(r)
            if share < best_share - 1e-9:
                best, best_share = r, share
            if share <= 0.0:
                break
        return best

    def _scalebar_px(self) -> float:
        """Longest the scale bar may be [px]: 15 % of the plot width, 100 to 240 px.

        A fixed 100 px read "0.2 m" across a 4 m room 1200 px wide, a twentieth of what is on
        screen; a share of the width keeps the bar a readable fraction of the view at any size."""
        return float(min(240.0, max(100.0, 0.15 * self.plot_rect().width())))

    def scalebar_length(self) -> float | None:
        """The length the scale bar reads now, in metres; None when there is none (for tests)."""
        if not self.show_scalebar or self._scale <= 0:
            return None
        return nice_length(self._scalebar_px() / self._scale)

    def scalebar_rect(self) -> QRectF | None:
        """Where the scale bar is drawn now (widget px), None when there is none (for tests)."""
        if not self.show_scalebar or self._scale <= 0:
            return None
        length = nice_length(self._scalebar_px() / self._scale)
        fm = QFontMetricsF(_ui_font(8.0))
        w = max(length * self._scale, fm.horizontalAdvance(format_length(length))) + 16
        return self._scalebar_rect(self.plot_rect(), w, fm.height() + 14)

    def _paint_scalebar(self, p: QPainter, tk: dict, area: QRectF) -> None:
        length = nice_length(self._scalebar_px() / self._scale)
        px = length * self._scale
        f = _ui_font(8.0)
        p.setFont(f)
        fm = QFontMetricsF(f)
        txt = format_length(length)
        w = max(px, fm.horizontalAdvance(txt)) + 16
        h = fm.height() + 14
        box = self._scalebar_rect(area, w, h)
        self._box(p, box, tk)
        y = box.bottom() - 5.5
        x0 = box.left() + 8
        p.setPen(QPen(qcolor(tk["text"]), 1.0))
        p.drawLine(QPointF(x0, y), QPointF(x0 + px, y))
        p.drawLine(QPointF(x0, y - 3), QPointF(x0, y))
        p.drawLine(QPointF(x0 + px, y - 3), QPointF(x0 + px, y))
        p.setPen(qcolor(tk["text2"]))
        p.drawText(QRectF(x0, box.top() + 2, w - 16, fm.height()), Qt.AlignmentFlag.AlignLeft, txt)

    def _paint_readout(self, p: QPainter, tk: dict, area: QRectF) -> None:
        x, y = self.screen_to_world(self._hover)
        ax, ay = self._axis_names

        def num(v: float) -> str:          # a true minus, as on the colour bar ("y −0.84 m", not "-0.84")
            return sig(v, 4).replace("-", "−")

        parts = [f"{ax} {num(x)} m", f"{ay} {num(y)} m"]
        if self._img is not None:
            v = self.value_at(x, y)
            last = self._label.split()[-1] if self._label else ""
            name = last if len(last) <= 4 else self._label  # "Cp", "|V|", "ωz" read better alone
            if v is not None:
                parts.append(f"{name} {num(v)}{(' ' + self._units) if self._units else ''}")
        txt = "    ".join(parts)
        f = _ui_font(8.0)
        p.setFont(f)
        fm = QFontMetricsF(f)
        w = fm.horizontalAdvance(txt) + 14
        r = QRectF(area.right() - w - 8, self._overlay_bottom(area) - fm.height() - 14, w, fm.height() + 6)
        self._box(p, r, tk)
        p.setPen(qcolor(tk["text"]))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)


# ---------------------------------------------------------------------------
# viewport toolbar
# ---------------------------------------------------------------------------

def fill_cmap_combo(combo: QComboBox, default: str, current: str | None = None) -> None:
    """Fill a colour-map combo: every map grouped by family (separators between families, the family
    in each item's tooltip), the field's recommended map marked "(default)" with data None.

    The marked entry means "the field's default" and keeps following the field;
    any other pick sticks. ``current`` is re-selected when it is still offered.
    """
    fams = colormap_families()
    if default not in {n for _t, ns in fams for n in ns}:
        fams = [("", [default])] + fams
    viz = _viz()
    purpose = {f[0]: f[2] for f in (getattr(viz, "CMAP_FAMILIES", ()) or ()) if len(f) > 2}
    combo.blockSignals(True)
    combo.clear()
    default_index = 0
    for k, (title, names) in enumerate(fams):
        if k and combo.count():
            combo.insertSeparator(combo.count())     # (a separator's data is None too: never looked up by data)
        for name in names:
            text = f"{cmap_title(name)} (default)" if name == default else cmap_title(name)
            combo.addItem(text, None if name == default else name)
            if name == default:
                default_index = combo.count() - 1
            tip = f"{title}: {purpose[title]}" if title in purpose else title
            if tip:
                combo.setItemData(combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
    i = combo.findData(current) if current is not None and current != default else -1
    combo.setCurrentIndex(i if i >= 0 else default_index)
    combo.blockSignals(False)



class ElidingComboBox(QComboBox):
    """A combo box whose current text ends in "…" when the box is narrower than the text.

    The viewport toolbars let their combos shrink (their minimum width is below
    the text's) so a 1280 px window with a dock open still fits the View
    control; a plain QComboBox then cuts the label mid-letter ("Viridis
    (defaul"), which reads as a rendering fault. The popup lists full names.

    Its size hint is the widest item plus the theme's padding and arrow
    (:attr:`TEXT_PAD`), whenever the items change: QComboBox measures its
    contents once, on first show, and with the theme's padding that hint was a
    few pixels short of "Thermal (default)", so a wide window still elided it.
    """

    #: Frame, padding and drop-down arrow of the theme's combo box around the text [px].
    TEXT_PAD = 52

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        m = self.model()
        for sig_ in (m.rowsInserted, m.rowsRemoved, m.modelReset):
            sig_.connect(lambda *_: self.updateGeometry())

    def sizeHint(self) -> QSize:
        s = super().sizeHint()
        fm = self.fontMetrics()
        widest = max((fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())), default=0)
        return QSize(max(s.width(), widest + self.TEXT_PAD), s.height())

    def paintEvent(self, _e) -> None:
        from PySide6.QtWidgets import QStyle, QStyleOptionComboBox, QStylePainter
        p = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        field = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                            QStyle.SubControl.SC_ComboBoxEditField, self)
        room = max(field.width() - 2, 0)
        if self.fontMetrics().horizontalAdvance(opt.currentText) > room:
            opt.currentText = self.fontMetrics().elidedText(opt.currentText, Qt.TextElideMode.ElideRight, room)
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)


class ViewportToolbar(ToolStrip):
    """The compact strip above a FieldView: Field, Colormap, Particles, Streamlines, Fit.

    ``changed`` fires on any user change of field, colour map or overlay toggles.
    Pages may append their own controls with ``add_widget`` / ``add_button``
    before or after :meth:`finish` (which adds the stretch and the Fit button).

    The strip has two widths (:meth:`fit_labels`): full, with the "Field" and
    "Colormap" captions and the text of every ``compactable`` button, and
    compact, without them. Its :meth:`minimumSizeHint` is the compact width, so
    a narrow viewport can actually make it compact: reporting the full width
    (Qt's default, the row's minimum with every caption shown) kept the page
    from ever shrinking below it, and the captions never went.
    """

    changed = Signal()

    def __init__(self, *, fields: Sequence[tuple[str, str]] = VIEW_FIELDS, particles: bool = True,
                 streamlines: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: 3D slices: field names and default colour maps have their 3D meaning (|ω|, Q)
        self.three_d = False
        self.field_label = self.add_widget(label("Field", "secondary"))
        self.field_combo = ElidingComboBox()
        self.field_combo.setMinimumWidth(150)
        self.field_combo.setToolTip("Field shown in the view")
        for key, text in fields:
            self.field_combo.addItem(text, key)
        self.add_widget(self.field_combo)
        self.row.addSpacing(8)
        self.cmap_label = self.add_widget(label("Colormap", "secondary"))
        self.cmap_combo = ElidingComboBox()
        # wide enough for the longest name with "(default)", so no label is clipped and the
        # toolbar does not shift when the mark moves to another map with the field
        fm = self.cmap_combo.fontMetrics()
        widest = max((fm.horizontalAdvance(f"{cmap_title(n)} (default)") for n in colormap_names()), default=0)
        self.cmap_combo.setMinimumWidth(max(140, widest + 52))      # padding, frame and arrow of the theme
        self.cmap_combo.setToolTip("Colour map; the one marked (default) is the recommended map for this field "
                                   "and follows the field when it changes")
        self.fill_cmaps()
        self.add_widget(self.cmap_combo)
        self.particles_btn: QToolButton | None = None
        self.streamlines_btn: QToolButton | None = None
        if particles or streamlines:
            self.add_separator()
        if particles:
            self.particles_btn = self._toggle("Particles", "Tracer particles carried by the flow")
        if streamlines:
            self.streamlines_btn = self._toggle("Streamlines", "Evenly spaced streamlines")
        # relabel the colour maps before anyone hears of the new field: the "(default)" mark moves
        self.field_combo.currentIndexChanged.connect(lambda *_: (self.fill_cmaps(), self.changed.emit()))
        self.cmap_combo.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.fit_btn: QToolButton | None = None

    def _toggle(self, text: str, tip: str) -> QToolButton:
        b = self.add_button(None, text, tip, checkable=True)
        b.setProperty("variant", "toggle")
        b.toggled.connect(lambda *_: self.changed.emit())
        return b

    def fill_cmaps(self, field: str | None = None) -> None:
        """List every colour map by name, the field's recommended one marked "(default)".

        The marked entry carries ``None`` as its data, meaning "the field's
        default", so it keeps following the field (speed -> viridis, Cp ->
        coolwarm ...) until the user picks another map, which then sticks.
        """
        field = field or self.field() or "speed"
        cur = self.cmap_combo.currentData()
        default = default_cmap3d(field) if self.three_d else default_cmap(field)
        fill_cmap_combo(self.cmap_combo, default, cur)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.fit_labels()

    def _labels(self) -> list[QWidget]:
        return [w for w in (getattr(self, "field_label", None), getattr(self, "cmap_label", None)) if w is not None]

    def minimumSizeHint(self) -> QSize:
        """The compact width (see the class docstring): captions and button texts are optional."""
        s = super().minimumSizeHint()
        spacing = max(self.row.spacing(), 0)
        fm = self.fontMetrics()
        shown = (sum(w.sizeHint().width() + spacing for w in self._labels() if not w.isHidden())
                 + sum(fm.horizontalAdvance(text) + 6 for b, text in getattr(self, "compactable", []) if b.text()))
        return QSize(max(s.width() - shown, 0), s.height())

    def fit_labels(self) -> None:
        """Drop the "Field" and "Colormap" captions when the strip is too narrow for them.

        At 1280 px with a dock open the viewport is about 430 px wide: with the
        captions the strip's minimum exceeded it and the View control beside
        them was squeezed ("Slice Split" ran together, the colour map was cut
        off). The combos say what they are in their tooltips and their text.
        The captions go before the combos are squeezed below their natural
        width: "Viridis (default)" reads better than "Colormap Viridis (de…".
        """
        labels = self._labels()
        if not labels:
            return
        spacing = max(self.row.spacing(), 0)
        hidden = sum(w.sizeHint().width() + spacing for w in labels if w.isHidden())
        # page buttons registered in ``compactable`` (button, text) lose their text together with the
        # captions; while compact they count at their full width, so the decision does not flicker
        compact = getattr(self, "compactable", [])
        fm = self.fontMetrics()
        saved = sum(fm.horizontalAdvance(text) + 6 for b, text in compact if not b.text())
        # what the combos lack of their natural width when laid out at their minimum (the explicit
        # minimum the page set, which is below the text's width)
        squeeze = sum(max(c.sizeHint().width() - c.minimumWidth(), 0)
                      for c in (self.field_combo, self.cmap_combo) if not c.isHidden() and c.minimumWidth() > 0)
        need = self.row.minimumSize().width() + hidden + saved + squeeze
        show = self.width() >= need
        for w in labels:
            if w.isHidden() == show:
                w.setVisible(show)
        for b, text in compact:
            if show and not b.text():
                b.setText(text)
                b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            elif not show and b.text():
                b.setText("")
                b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

    def finish(self, view: FieldView | None = None) -> QToolButton:
        """Add the stretch and the Fit toggle (bound to ``view``'s whole-domain mode when given).

        Fit is a toggle because the view has two framings: the default one
        around the body, and the whole computational domain. Checked shows the
        whole domain; unchecking (or ``F`` / double-click in the view) returns to
        the body. Zooming or panning by hand unchecks it.
        """
        self.add_stretch()
        self.fit_btn = self.add_button("fullscreen", "Fit", "Show the whole computational domain; "
                                       "off frames the body and its wake (F, or double-click)", checkable=True)
        self.fit_btn.setProperty("variant", "toggle")
        if view is not None:
            btn = self.fit_btn
            btn.toggled.connect(lambda on: view.set_whole_domain(on) if on != view.whole_domain() else None)

            def follow(on: bool) -> None:
                btn.blockSignals(True)
                btn.setChecked(on)
                btn.blockSignals(False)

            view.wholeDomainChanged.connect(follow)
        return self.fit_btn

    def set_fields(self, fields: Sequence[tuple[str, str]], *, three_d: bool = False) -> None:
        """Offer another field list (2D <-> 3D slices), keeping the current field when it is offered.

        Emits ``changed`` only when the selected field actually changes. In 3D
        the particle toggle is hidden (3D runs have no tracer particles) and the
        "(default)" colour map follows the 3D table.
        """
        cur = self.field()
        same = [(self.field_combo.itemData(i), self.field_combo.itemText(i))
                for i in range(self.field_combo.count())] == [tuple(f) for f in fields]
        mode_changed = bool(three_d) != self.three_d
        self.three_d = bool(three_d)
        if self.particles_btn is not None:
            self.particles_btn.setVisible(not self.three_d)
        if same and not mode_changed:
            return
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for key, text in fields:
            self.field_combo.addItem(text, key)
        i = self.field_combo.findData(cur)
        self.field_combo.setCurrentIndex(i if i >= 0 else max(self.field_combo.findData("speed"), 0))
        self.field_combo.blockSignals(False)
        self.fill_cmaps()
        if self.field() != cur or mode_changed:
            self.changed.emit()

    def field(self) -> str:
        return self.field_combo.currentData()

    def cmap(self) -> str | None:
        return self.cmap_combo.currentData()


__all__ = [
    "CMAP_TITLES",
    "CP_LIMITS",
    "DIAGNOSTIC_FIELDS",
    "ESSENTIAL_MARGINS",
    "KEEP_BODY_MARGINS",
    "NON_NEGATIVE_FIELDS",
    "NON_NEGATIVE_FIELDS_3D",
    "ROAD_BAND_PX",
    "SLICE_AXIS_NAMES",
    "SOLID_EDGE",
    "SOLID_FILL",
    "THERMAL_FIELDS",
    "VIEW_FIELDS",
    "VIEW_FIELDS_3D",
    "ColorBar",
    "ElidingComboBox",
    "FieldView",
    "LiveRangeTracker",
    "ViewportToolbar",
    "body_window",
    "clipped_ends",
    "cmap_title",
    "colorbar_width",
    "colorize",
    "colormap_families",
    "colormap_lut",
    "colormap_names",
    "default_cmap",
    "default_cmap3d",
    "display_range",
    "display_range3d",
    "dye_legend",
    "field_info",
    "field_info3d",
    "fill_cmap_combo",
    "format_length",
    "frame_clipped_ends",
    "lut_indices",
    "near_field_window",
    "nice_length",
    "paint_colorbar",
    "paint_legend",
    "particle_colors",
    "robust_range",
    "section_paths",
    "split_orientation",
    "tick_labels",
    "to_qpolygonf",
    "view_fields_for",
    "window_slices",
]
