"""2D overlays of the 3D view: colour bar, axes triad and caption, drawn by VTK in pixels.

Why not ``vtkScalarBarActor`` / ``vtkOrientationMarkerWidget``? The scalar bar
scales its fonts with the bar size and places labels by its own rules, so it
cannot match the 2D colour bar of :mod:`aerolume.app.widgets.field_view` (title,
units line, ticks at round values with a common number of decimals, extend
triangles, 8 pt text); the orientation widget needs an interactor, which an
off-screen renderer does not have. Both are rebuilt here from plain 2D actors
laid out in pixels, so the 3D view, the 2D view and the report all use one
legend style.

Layout is computed by pure functions (:func:`colorbar_layout`,
:func:`tick_labels`, :func:`triad_segments`) that tests can check without
OpenGL; :class:`Overlay` turns the layout into actors and caches them, because
building a text texture costs ~0.2 ms and the triad changes on every orbit.
Coordinates are VTK display pixels: origin bottom-left, y up.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.units import sig
from ..viz.colormaps import get_lut

__all__ = ["BAR_FRACTION", "ColorbarSpec", "Overlay", "colorbar_layout", "colorbar_strip_width",
           "font_file", "nice_ticks", "tick_labels", "triad_segments"]

#: Fraction of the available height the colour gradient takes (the 2D bar uses 62 %).
BAR_FRACTION = 0.55
_FONT_PX = 11.0          # 8 pt at 96 dpi, the 2D colour bar's size
_PAD = 12.0
_BAR_W = 12.0
_EXTEND_H = 9.0
#: Distance of the triad origin from the bottom-left corner (px at scale 1).
TRIAD_INSET = 48.0
#: Opacity of the background-coloured plate behind the colour bar: lines passing under the
#: legend fade to a hint, so they neither hide its labels nor vanish abruptly at its edge.
PLATE_OPACITY = 0.86


def nice_ticks(vmin: float, vmax: float, n: int = 6) -> list[float]:
    """Round tick values (1, 2, 2.5, 5 x 10^k) covering [vmin, vmax] (the 2D plots' rule)."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return [vmin] if math.isfinite(vmin) else []
    raw = (vmax - vmin) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    start = math.ceil(vmin / step - 1e-9) * step
    out, v = [], start
    while v <= vmax + 1e-9 * step:
        out.append(0.0 if abs(v) < 1e-12 * step else v)
        v += step
    return out


def tick_labels(vmin: float, vmax: float, n: int = 6) -> list[tuple[float, str]]:
    """Ticks with labels at a common number of decimals and a true minus sign."""
    ticks = nice_ticks(vmin, vmax, n) if vmax > vmin else [vmin]
    step = abs(ticks[1] - ticks[0]) if len(ticks) > 1 else (abs(vmax - vmin) or 1.0)
    decimals = max(0, -math.floor(math.log10(step) + 1e-9)) if step > 0 else 0
    out = []
    for tv in ticks:
        txt = f"{tv:.{decimals}f}" if abs(tv) < 1e5 and decimals <= 6 else sig(tv, 3)
        out.append((tv, txt.replace("-", "\u2212")))
    return out


@dataclass(frozen=True)
class ColorbarSpec:
    """What the colour bar shows: map, range, title, units and clipped ends."""
    cmap: str
    vmin: float
    vmax: float
    title: str
    units: str = ""
    extend: tuple[bool, bool] = (False, False)     # (below, above)


def _text_w(text: str, px: float) -> float:
    """Width estimate of UI-font text: 0.56 em per character (Segoe UI average)."""
    return 0.56 * px * len(text)


def colorbar_strip_width(spec: ColorbarSpec | None, scale: float = 1.0) -> float:
    """Pixel width the colour bar reserves at the right edge (0 without a bar).

    The 3D view is rendered in the remaining width, so the legend never hides
    the model (the 2D FieldView rule).
    """
    if spec is None:
        return 0.0
    px = _FONT_PX * scale
    labels = [t for _, t in tick_labels(spec.vmin, spec.vmax)]
    w_ticks = (_BAR_W + 4 + 6) * scale + max((_text_w(t, px) for t in labels), default=20.0)
    w_title = max(_text_w(spec.title, px * 1.04), _text_w(spec.units, px))
    return float(min(max(w_ticks, w_title) + 2 * _PAD * scale, 190.0 * scale))


def colorbar_layout(spec: ColorbarSpec, width: float, height: float, scale: float = 1.0, *,
                    right: float = 0.0) -> dict:
    """Pixel geometry of the colour bar inside a (width x height) image (y up).

    Returns ``bar`` = (x0, y0, x1, y1), ``ticks`` = [(y, label)], ``title`` and
    ``units`` anchor points (top-left of each line) and the extend triangles.
    ``right`` is the width [px] already taken at the right edge by other bars
    (a second colour map's bar sits left of the first).
    """
    strip = colorbar_strip_width(spec, scale)
    px = _FONT_PX * scale
    line = px * 1.45
    x0 = width - right - strip + _PAD * scale
    head = line + (line if spec.units else 0.0) + 6 * scale + (_EXTEND_H + 2) * scale * spec.extend[1]
    tail = (_EXTEND_H + 2) * scale * spec.extend[0]
    bar_h = max(40.0 * scale, min(height * BAR_FRACTION, height - head - tail - 2 * _PAD * scale - line))
    y_top = height - max(_PAD * scale + head, (height - bar_h) / 2)
    y_bot = y_top - bar_h
    bar = (x0, y_bot, x0 + _BAR_W * scale, y_top)
    span = spec.vmax - spec.vmin
    ticks = []
    for tv, txt in tick_labels(spec.vmin, spec.vmax):
        f = (tv - spec.vmin) / span if span > 0 else 0.5
        ticks.append((y_bot + f * bar_h, txt))
    title_y = y_top + head
    tri = []
    right_edge = width - right
    if spec.extend[1]:
        tri.append(("above", y_top + 1 * scale, y_top + (1 + _EXTEND_H) * scale))
    if spec.extend[0]:
        tri.append(("below", y_bot - 1 * scale, y_bot - (1 + _EXTEND_H) * scale))
    return {"strip": strip, "bar": bar, "ticks": ticks, "title": (x0, title_y),
            "units": (x0, title_y - line) if spec.units else None, "font_px": px, "extend": tri,
            "right_edge": right_edge}


def triad_segments(right: np.ndarray, up: np.ndarray, forward: np.ndarray, *, origin=(40.0, 40.0),
                   length: float = 28.0) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """2D segments of the x/y/z axes for a camera frame, far axes first.

    Each entry is (label, start, end, depth) in display pixels (y up); depth > 0
    means the axis points away from the viewer, and those are drawn first so
    the near axes overlap them, as in a real 3D triad.
    """
    o = np.asarray(origin, float)
    segs = []
    for name, e in (("x", np.array([1.0, 0, 0])), ("y", np.array([0, 1.0, 0])), ("z", np.array([0, 0, 1.0]))):
        d = np.array([float(np.dot(e, right)), float(np.dot(e, up))])
        segs.append((name, o, o + d * length, float(np.dot(e, forward))))
    segs.sort(key=lambda s: -s[3])
    return segs


def font_file(bold: bool = False) -> str | None:
    """Path of the UI font for VTK text (Segoe UI on Windows), or None for VTK's Arial."""
    windir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = ("seguisb.ttf", "segoeuib.ttf") if bold else ("segoeui.ttf",)
    cands = [windir / n for n in names]
    cands += [Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for c in cands:
        if c.exists():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# VTK actors
# ---------------------------------------------------------------------------

class Overlay:
    """Owns the 2D actors of one overlay renderer and keeps them in sync with the view."""

    def __init__(self, renderer) -> None:
        self.ren = renderer
        self._bar_key = None
        self._bar_actors: list = []
        self._triad_actors: list = []
        self._triad_texts: dict = {}
        self._caption = None
        self._caption_key = None

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _text(txt: str, px: float, rgb, *, bold: bool = False, halign: str = "left", valign: str = "bottom"):
        from vtkmodules.vtkRenderingCore import vtkTextActor
        a = vtkTextActor()
        a.SetInput(txt)
        tp = a.GetTextProperty()
        f = font_file(bold)
        if f:
            tp.SetFontFamily(4)              # VTK_FONT_FILE
            tp.SetFontFile(f)
        else:
            tp.SetFontFamilyToArial()
            tp.SetBold(bold)
        tp.SetFontSize(max(6, round(px)))
        tp.SetColor(*rgb)
        tp.SetJustification({"left": 0, "center": 1, "right": 2}[halign])
        tp.SetVerticalJustification({"bottom": 0, "center": 1, "top": 2}[valign])
        a.SetTextScaleModeToNone()
        a.PickableOff()
        return a

    @staticmethod
    def _poly_actor(points: np.ndarray, lines=None, polys=None, colors=None, rgb=(0, 0, 0), width=1.0):
        from vtkmodules.util.numpy_support import numpy_to_vtk
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D

        pts = vtkPoints()
        p3 = np.zeros((len(points), 3))
        p3[:, :2] = points
        pts.SetData(numpy_to_vtk(p3, deep=True))
        pd = vtkPolyData()
        pd.SetPoints(pts)
        if lines:
            ca = vtkCellArray()
            for ln in lines:
                ca.InsertNextCell(len(ln), list(ln))
            pd.SetLines(ca)
        if polys:
            ca = vtkCellArray()
            for pl in polys:
                ca.InsertNextCell(len(pl), list(pl))
            pd.SetPolys(ca)
        m = vtkPolyDataMapper2D()
        m.SetInputData(pd)
        if colors is not None:
            c = numpy_to_vtk(np.ascontiguousarray(colors, dtype=np.uint8), deep=True)
            c.SetName("rgb")
            pd.GetCellData().SetScalars(c)
            m.SetColorModeToDirectScalars()
            m.ScalarVisibilityOn()
            m.SetScalarModeToUseCellData()
        else:
            m.ScalarVisibilityOff()
        a = vtkActor2D()
        a.SetMapper(m)
        a.GetProperty().SetColor(*rgb)
        a.GetProperty().SetLineWidth(width)
        a.PickableOff()
        return a

    def _clear(self, actors: list) -> None:
        for a in actors:
            self.ren.RemoveViewProp(a)
        actors.clear()

    # ---- colour bar ----------------------------------------------------
    def set_colorbar(self, spec: ColorbarSpec | None, width: int, height: int, scale: float, theme) -> None:
        """Rebuild the colour bar if its spec, the image size or the theme changed."""
        self.set_colorbars([spec] if spec is not None else [], width, height, scale, theme)

    def set_colorbars(self, specs, width: int, height: int, scale: float, theme) -> None:
        """Rebuild the colour bars (the first at the right edge, each further one left of the previous)
        if the specs, the image size or the theme changed."""
        specs = tuple(s for s in specs if s is not None)
        key = (specs, width, height, round(scale, 3), theme)
        if key == self._bar_key:
            return
        self._bar_key = key
        self._clear(self._bar_actors)
        acts: list = []
        right = 0.0
        for spec in specs:
            acts += self._bar_actors_of(spec, width, height, scale, theme, right)
            right += colorbar_strip_width(spec, scale)
        for a in acts:
            self.ren.AddViewProp(a)
        self._bar_actors = acts

    def _bar_actors_of(self, spec: ColorbarSpec, width: int, height: int, scale: float, theme,
                       right: float) -> list:
        """The 2D actors of one colour bar whose strip ends ``right`` px from the image's right edge."""
        lay = colorbar_layout(spec, width, height, scale, right=right)
        x0, y0, x1, y1 = lay["bar"]
        lut = get_lut(spec.cmap)
        n = lut.shape[0]
        px = lay["font_px"]
        line_w = max(1.0, round(scale))
        # A boxed legend (the plots' "small boxed legend"): a flat plate of the background colour
        # with a hairline frame, no radius or shadow. The camera frames the model left of the bar,
        # but streamlines and smoke running on into the far wake pass under it, and without the
        # plate they cross the title and the tick labels; the frame makes the plate's edge read as
        # a legend box rather than as a cut through the floor.
        m = 8.0 * scale
        tail = max((y0 - tip for side, _, tip in lay["extend"] if side == "below"), default=0.0)
        bx0, by0 = round(x0 - m) + 0.5, round(y0 - max(tail, 0.6 * px) - m) + 0.5
        bx1, by1 = round(lay["right_edge"] - _PAD * scale + m) - 0.5, round(lay["title"][1] + m) - 0.5
        plate = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]])
        back = self._poly_actor(plate, polys=[(0, 1, 2, 3)], rgb=theme.background)
        back.GetProperty().SetOpacity(PLATE_OPACITY)
        rim = self._poly_actor(plate, lines=[(0, 1, 2, 3, 0)], rgb=theme.border, width=line_w)
        rim.GetProperty().SetOpacity(0.6)
        ys = np.linspace(y0, y1, n + 1)
        pts = np.concatenate([np.stack([np.full(n + 1, x0), ys], 1), np.stack([np.full(n + 1, x1), ys], 1)])
        quads = [(i, i + 1, n + 2 + i, n + 1 + i) for i in range(n)]
        acts = [back, rim, self._poly_actor(pts, polys=quads, colors=lut)]
        # 1 px frame at pixel centres
        h = 0.5
        frame = np.array([[x0 + h, y0 + h], [x1 - h, y0 + h], [x1 - h, y1 - h], [x0 + h, y1 - h]])
        acts.append(self._poly_actor(frame, lines=[(0, 1, 2, 3, 0)], rgb=theme.text2, width=line_w))
        for side, base, tip in lay["extend"]:
            rgb = lut[-1] if side == "above" else lut[0]
            tri = np.array([[x0 + h, base], [x1 - h, base], [0.5 * (x0 + x1), tip]])
            acts.append(self._poly_actor(tri, polys=[(0, 1, 2)], colors=np.array([rgb])))
            acts.append(self._poly_actor(tri, lines=[(0, 1, 2, 0)], rgb=theme.text2, width=line_w))
        tick_pts, tick_lines = [], []
        for y, txt in lay["ticks"]:
            yy = round(y) + 0.5
            tick_pts += [[x1, yy], [x1 + 3 * scale, yy]]
            tick_lines.append((len(tick_pts) - 2, len(tick_pts) - 1))
            t = self._text(txt, px, theme.text2, valign="center")
            t.SetPosition(x1 + 6 * scale, y)
            acts.append(t)
        if tick_pts:
            acts.append(self._poly_actor(np.array(tick_pts), lines=tick_lines, rgb=theme.text2, width=line_w))
        tx, ty = lay["title"]
        t = self._text(spec.title, px * 1.04, theme.text, bold=True, valign="top")
        t.SetPosition(tx, ty)
        acts.append(t)
        if lay["units"] is not None:
            ux, uy = lay["units"]
            t = self._text(spec.units, px, theme.text2, valign="top")
            t.SetPosition(ux, uy)
            acts.append(t)
        return acts

    # ---- triad ---------------------------------------------------------
    def set_triad(self, visible: bool, frame, scale: float, theme) -> None:
        """Axes triad in the bottom-left corner, oriented like the camera."""
        if not visible:
            self._clear(self._triad_actors)
            self._triad_texts = {}
            return
        r, u, f = frame
        length = 30.0 * scale
        # far enough in that a label of an axis pointing left / down stays inside the image
        origin = (TRIAD_INSET * scale, TRIAD_INSET * scale)
        segs = triad_segments(r, u, f, origin=origin, length=length)
        cols = {"x": theme.axis_x, "y": theme.axis_y, "z": theme.axis_z}
        self._clear(self._triad_actors)
        acts = []
        lw = max(1.0, 1.5 * scale)
        for name, a, b, _ in segs:
            acts.append(self._poly_actor(np.array([a, b]), lines=[(0, 1)], rgb=cols[name], width=lw))
        for name, a, b, _ in segs:
            d = b - a
            n = float(np.linalg.norm(d))
            pos = b + (d / n * 8.0 * scale if n > 1e-6 else np.array([0.0, 8.0 * scale]))
            key = (name, round(scale, 3), theme)
            t = self._triad_texts.get(key)
            if t is None:
                t = self._text(name, _FONT_PX * scale, cols[name], bold=True, halign="center", valign="center")
                self._triad_texts[key] = t
            t.SetPosition(float(pos[0]), float(pos[1]))
            t.SetVisibility(n > 0.18 * length)       # an axis seen end-on gets no label
            acts.append(t)
        for a in acts:
            self.ren.AddViewProp(a)
        self._triad_actors = acts

    # ---- caption -------------------------------------------------------
    def set_caption(self, text: str, height: int, scale: float, theme, *, top: float = 0.0) -> None:
        """One line of text in the top-left corner (field name, case, time), ``top`` px lower."""
        key = (text, height, round(scale, 3), theme, top)
        if key == self._caption_key:
            return
        self._caption_key = key
        if self._caption is not None:
            self.ren.RemoveViewProp(self._caption)
            self._caption = None
        if not text:
            return
        t = self._text(text, _FONT_PX * 1.09 * scale, theme.text2, valign="top")
        t.SetPosition(_PAD * scale, height - (_PAD * 0.75 + top) * scale)
        self.ren.AddViewProp(t)
        self._caption = t
