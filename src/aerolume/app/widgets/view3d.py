"""View3D: the Qt viewport of the 3D viewer (DESIGN.md 7.9).

The widget owns a :class:`aerolume.viz3d.Scene` and paints its off-screen
render; it never embeds a native VTK window. Pages build the picture through
:attr:`View3D.scene` (``add_mesh``, ``add_slice``, ...) and call
:meth:`View3D.refresh`; the widget turns mouse input into camera moves.

Design notes (the WHY):

* **Paint an image, do not embed VTK.** ``QVTKRenderWindowInteractor`` puts a
  native OpenGL window inside the Qt hierarchy, which breaks ``grab()``, the
  offscreen platform used by the tests and screenshot tools, and high-DPI
  scaling on Windows. Rendering off-screen and painting the numpy image costs
  ~4 ms per frame at 1600 x 900 on the target GPU, so the viewport still
  follows the mouse at the display rate, and ``grab()`` returns exactly the
  pixels of :meth:`Scene.render`.
* **Render on demand, in paint.** Every change only marks the view dirty and
  starts a 4 ms single-shot timer that calls ``update()``; bursts of mouse or
  wheel events collapse into one render, and nothing renders while nothing
  changes. ``paintEvent`` renders only if the scene revision, the size or the
  device pixel ratio differ from the cached image, so repainting an
  unchanged view (an overlapping dialog, a focus change) costs one blit.
* **Device pixels.** The scene renders at width x dpr by height x dpr with
  ``scale=dpr`` (overlay text and lines scale too) and the image carries that
  ratio, so text in the colour bar is crisp on 150 % displays.
* **CAD mouse conventions** (ParaView, Fusion, Onshape): left drag orbits
  (turntable about world z), right or middle drag (or Shift + left) pans,
  the wheel zooms towards the cursor, double-click or F fits. 0.35 deg per
  pixel orbits a full turn over about 1000 px.
* **Theme.** Background, text and floor follow the active theme tokens
  (``theme.changed``), as the 2D field view does.
* **Flow in the volume** (DESIGN.md 7.13). :meth:`View3D.set_smoke`,
  :meth:`View3D.set_streamlines` and :meth:`View3D.set_seed_rake` keep one layer
  each and update it in place (the same VTK actors: a live run's smoke and lines
  change without flashing); :meth:`View3D.show_live_frame` applies a
  ``LiveFrame`` (its volume only when ``volume_serial`` changed). The seed rake
  can be dragged by its round handle: it moves in the plane facing the camera's
  dominant axis (the y-z plane from the front, x-z from the side, x-y from the
  top), the grabbed point stays under the cursor, ``rakeMoving`` reports the
  rake while dragging and ``rakeChanged`` on release (the page then writes it to
  ``ViewSettings.seed_rake``). The handle is grabbed within 10 px.
* **One legend in a split view.** When the 3D view sits in a ``QSplitter``
  next to a slice view (a ``FieldView``: Simulate and Results, View: Split),
  the slice already shows the colour bar and the time label. Two identical
  bars side by side are clutter, and the 3D bar took a fifth of a pane that is
  only ~350 px wide at 1280 px. So the 3D view leaves out its colour bar when
  the partner shows a bar *of the same range* (the same legend; a different
  field keeps its own bar, it would otherwise be read against the wrong scale)
  and its caption when the partner shows the same text. Without the bar the
  camera frames the model over the full width. The partner is found
  automatically (the visible sibling in the splitter that draws a colour bar)
  or set with :meth:`View3D.set_legend_partner`; :meth:`View3D.set_legend`
  forces both on or off.
* **Overlay toolbar** (guideline 5b): borderless, 24 px icon buttons with
  tooltips (iso cube, front / side / top / back faces, fit, edges), so the view
  does not get a third row of text controls under the page strips.
"""
from __future__ import annotations

import logging
import re
import weakref
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QIcon, QIconEngine, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QFrame, QSizePolicy, QSplitter, QToolButton, QWidget

from .common import CONTROL_H, SPACING, ToolStrip
from .plots import active_icon_token, lazy_import, qcolor, tokens

log = logging.getLogger(__name__)

__all__ = ["CAMERA_PRESETS", "PRESET_ICONS", "View3D", "View3DToolbar", "preset_icon"]

#: Presets offered in toolbars, in menu order: (key, button text, tooltip).
CAMERA_PRESETS: tuple[tuple[str, str, str], ...] = (
    ("iso", "Iso", "Isometric view from upstream, left and above"),
    ("front", "Front", "Look downstream at the frontal area (along +x)"),
    ("side", "Side", "Side view, flow from left to right"),
    ("top", "Top", "Plan view from above, flow from left to right"),
    ("back", "Back", "Look upstream at the base and wake (along -x)"),
)

ORBIT_DEG_PER_PX = 0.35
WHEEL_ZOOM_PER_NOTCH = 1.18
COALESCE_MS = 4
#: the seed rake's handle is grabbed within this many (logical) pixels
RAKE_GRAB_PX = 10.0
#: an unbounded box for drawing a rake that was already resolved (nothing to clamp)
_NO_CLAMP = (-1e9, 1e9, -1e9, 1e9, -1e9, 1e9)
#: two colour bars are "the same legend" when both ends agree within this share of the span
LEGEND_MATCH = 0.02

# 16 px line icons of the camera presets, drawn like the bundled set (resources/icons: 1.25 px
# stroke in currentColor, round caps). Iso is the isometric cube; the others are a cabinet
# cube with the face the camera looks at filled, the CAD convention (SolidWorks, Fusion).
_CUBE = "M2.5 6.5h7v7h-7z M2.5 6.5l4-4h7l-4 4 M9.5 13.5l4-4v-7"
_FACE = '<path d="{d}" fill="currentColor" fill-opacity=".5" stroke="none"/>'
_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" fill="none" '
        'stroke="currentColor" stroke-width="1.25" stroke-linecap="round" stroke-linejoin="round">{}</svg>')
PRESET_ICONS: dict[str, str] = {
    "iso": _SVG.format('<path d="M8 1.9 13.6 5v6.1L8 14.1 2.4 11.1V5z M2.4 5 8 8.1 13.6 5 M8 8.1v6"/>'),
    "front": _SVG.format(_FACE.format(d="M2.5 6.5h7v7h-7z") + f'<path d="{_CUBE}"/>'),
    "side": _SVG.format(_FACE.format(d="M9.5 6.5l4-4v7l-4 4z") + f'<path d="{_CUBE}"/>'),
    "top": _SVG.format(_FACE.format(d="M2.5 6.5l4-4h7l-4 4z") + f'<path d="{_CUBE}"/>'),
    "back": _SVG.format(_FACE.format(d="M6.5 2.5h7v7h-7z")
                        + '<path d="M6.5 2.5v7h7 M6.5 9.5l-4 4" stroke-dasharray="1.4 1.3"/>'
                        + f'<path d="{_CUBE}"/>'),
}
PRESET_ICONS["bottom"] = PRESET_ICONS["top"]


class _InlineIconEngine(QIconEngine):
    """A QIcon engine for an inline SVG, tinted with theme tokens at paint time.

    It mirrors ``theme._SvgIconEngine`` (the vector drawn at the exact size and
    device pixel ratio, colours looked up when painted so the icons follow a
    theme switch) for the preset icons above, which belong to the 3D view
    rather than to the shared icon folder.
    """

    def __init__(self, svg: str, colour: str = "text2", active: str | None = "accent") -> None:
        super().__init__()
        self._svg, self._colour, self._active = svg, colour, active

    def _renderer(self, mode, state):
        from PySide6.QtSvg import QSvgRenderer
        t = tokens()
        if mode == QIcon.Mode.Disabled:
            c = t["faint"]
        elif self._active and state == QIcon.State.On:
            active = active_icon_token(self._active)        # ink since theme v4, as the theme's icons
            c = t.get(active, active)
        else:
            c = t.get(self._colour, self._colour)
        return QSvgRenderer(QByteArray(self._svg.replace("currentColor", c).encode()))

    def paint(self, painter: QPainter, rect, mode, state) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(rect.width(), rect.height())
        r = QRectF(rect.x() + (rect.width() - side) / 2, rect.y() + (rect.height() - side) / 2, side, side)
        self._renderer(mode, state).render(painter, r)
        painter.restore()

    def pixmap(self, size: QSize, mode, state) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size: QSize, mode, state, scale: float) -> QPixmap:
        pm = QPixmap(QSize(max(1, int(size.width() * scale)), max(1, int(size.height() * scale))))
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        self.paint(p, pm.rect(), mode, state)
        p.end()
        pm.setDevicePixelRatio(scale)
        return pm

    def clone(self) -> QIconEngine:
        return _InlineIconEngine(self._svg, self._colour, self._active)


def preset_icon(key: str) -> QIcon:
    """The 16 px line icon of a camera preset ('iso', 'front', 'side', 'top', 'back', 'bottom')."""
    return QIcon(_InlineIconEngine(PRESET_ICONS.get(key, PRESET_ICONS["iso"])))


def _theme_icon(name: str) -> QIcon:
    theme = lazy_import("aerolume.app.theme")
    return theme.icon(name, "text2", active="accent") if theme is not None else QIcon()


def _shows_legend(w) -> bool:
    """Is ``w`` a visible view drawing a colour bar over an image (a ``FieldView``-like partner)?"""
    try:
        return (w is not None and w.isVisible() and bool(getattr(w, "show_colorbar", False))
                and callable(getattr(w, "has_image", None)) and bool(w.has_image())
                and callable(getattr(w, "colour_range", None)))
    except RuntimeError:                          # its C++ side is gone
        return False


_NUMBER = re.compile(r"\d[\d.,\u2009]*")   # digits with decimal / thousands (thin space) separators


def _caption_shape(text: str) -> str:
    """A caption with its numbers blanked out.

    During a run the slice gets every frame and the 3D view only the frames
    that carry a volume, so their time labels differ by a step or two
    ("t = 11.0 L/U" against "t = 11.1 L/U"). They are still the same label,
    shown twice; a caption that says something else ("smoke at the end of
    the run") has another shape and is kept.
    """
    return _NUMBER.sub("#", text.strip())


def _scene_class():
    from ...viz3d import Scene
    return Scene


class View3D(QWidget):
    """A 3D viewport: a :class:`~aerolume.viz3d.Scene` painted as an image, with mouse camera control.

    ``scene`` may be passed in (e.g. one built by a page in advance);
    otherwise the widget creates one in the active theme. Signals:
    ``cameraChanged`` after any camera move, ``rendered(float)`` with the
    render time in ms.
    """

    cameraChanged = Signal()
    rendered = Signal(float)
    rakeMoving = Signal(object)      # rake dict (metres) while the handle is dragged
    rakeChanged = Signal(object)     # rake dict when the drag ends

    def __init__(self, parent: QWidget | None = None, *, scene=None, overlay_toolbar: bool = False) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(160, 120)
        self._scene = scene if scene is not None else _scene_class()(tokens())
        self._image: QImage | None = None
        self._buf: np.ndarray | None = None
        self._key: tuple | None = None
        self._drag: tuple[str, QPointF] | None = None
        self._placeholder: tuple[str, str] = ("", "")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(COALESCE_MS)
        self._timer.timeout.connect(self.update)
        self._overlay: View3DToolbar | None = None
        self._toolbars: weakref.WeakSet = weakref.WeakSet()     # kept in sync with the edge toggle
        self.last_render_ms = float("nan")
        # flow in the volume: one layer each, updated in place
        self._flow: dict[str, str] = {}
        self._flow_serial: int | None = None
        self._flow_lines = None
        self._rake_editable = False
        self._rake_drag: dict | None = None
        # one legend in a split view (see the module notes)
        self._legend: bool | None = None            # None: shared with a partner; True / False: forced
        self._partner: QWidget | None = None        # explicit partner (None: the splitter sibling)
        self._watched: QWidget | None = None        # the partner whose show / hide events we follow
        # Connect the bound method itself (not a lambda, as plots.connect_theme does): Qt then
        # drops the connection when this widget is destroyed, so a theme switch after a page
        # closed never calls into a deleted view.
        theme = lazy_import("aerolume.app.theme")
        sig = getattr(theme, "changed", None) if theme is not None else None
        if sig is not None:
            sig.connect(self._on_theme)
        if overlay_toolbar:
            self.set_overlay_toolbar(True)

    # ------------------------------------------------------------------
    # API for pages
    # ------------------------------------------------------------------
    @property
    def scene(self):
        """The :class:`~aerolume.viz3d.Scene` this view paints (edit it, then :meth:`refresh`)."""
        return self._scene

    def set_scene(self, scene) -> None:
        """Show another scene (the old one is not closed: the caller owns it)."""
        self._scene = scene
        scene.set_theme(tokens())
        scene.set_top_inset(self._overlay.height() + SPACING if self._overlay is not None else 0.0)
        self.refresh()

    def refresh(self) -> None:
        """Schedule a repaint; renders only if the scene changed. Cheap to call often."""
        if not self._timer.isActive():
            self._timer.start()

    def set_placeholder(self, title: str, subtitle: str = "") -> None:
        """Text shown while the scene has no layers (e.g. before a geometry is imported)."""
        self._placeholder = (title, subtitle)
        self.update()

    def set_view(self, preset: str, fit: bool = True) -> None:
        """Camera preset: 'iso', 'front', 'side', 'top', 'back' (or 'bottom')."""
        self._scene.set_camera(preset, fit=fit)
        self._camera_moved()

    def fit(self) -> None:
        """Frame the bodies, keeping the view direction."""
        self._scene.fit()
        self._camera_moved()

    def set_edges(self, on: bool) -> None:
        """Show or hide mesh edges (the toolbar's Edges toggle)."""
        self._scene.set_edges(on)
        for bar in list(self._toolbars):
            try:
                bar.sync(self)
            except RuntimeError:            # a toolbar whose C++ side is already gone
                self._toolbars.discard(bar)
        self.refresh()

    def edges(self) -> bool:
        return bool(self._scene.edges)

    def toolbar(self, parent: QWidget | None = None) -> View3DToolbar:
        """A page-level tool strip (camera presets, Fit, Edges) bound to this view."""
        return View3DToolbar(self, parent)

    def set_overlay_toolbar(self, on: bool) -> None:
        """Float a borderless row of icon buttons (presets, Fit, Edges) in the view's top-left corner."""
        if on and self._overlay is None:
            self._overlay = View3DToolbar(self, self, compact=True)
            self._overlay.move(SPACING, SPACING)
            self._overlay.show()
            self._scene.set_top_inset(self._overlay.height() + SPACING)   # caption below the icons
        elif not on and self._overlay is not None:
            self._overlay.deleteLater()
            self._overlay = None
            self._scene.set_top_inset(0.0)
        self.refresh()

    def overlay_toolbar(self) -> View3DToolbar | None:
        return self._overlay

    # ------------------------------------------------------------------
    # one legend in a split view
    # ------------------------------------------------------------------
    def set_legend(self, on: bool | None) -> None:
        """Force the colour bar and caption on (True) or off (False); None shares them with a partner."""
        self._legend = None if on is None else bool(on)
        self.update()

    def set_legend_partner(self, view: QWidget | None) -> None:
        """The view whose matching colour bar and caption this one leaves out (None: automatic).

        ``view`` is duck-typed like ``FieldView``: ``has_image()``,
        ``show_colorbar``, ``colour_range()`` and ``caption()``.
        """
        self._partner = view
        self.update()

    def legend_partner(self) -> QWidget | None:
        """The partner carrying the legend now (visible, with an image), or None."""
        cand = self._partner
        if cand is not None:
            try:
                cand.isVisible()
            except RuntimeError:                  # its C++ side is gone
                self._partner = cand = None
        if cand is None:
            # the pane holding this view: the view itself or a host frame around it (Simulate
            # puts the view in a frame), at most three levels up, directly in a QSplitter
            branch, par = self, self.parentWidget()
            for _ in range(3):
                if par is None or isinstance(par, QSplitter):
                    break
                branch, par = par, par.parentWidget()
            if isinstance(par, QSplitter):
                for i in range(par.count()):
                    w = par.widget(i)
                    if w is not branch and callable(getattr(w, "colour_range", None)):
                        cand = w
                        break
        self._watch(cand)
        return cand if _shows_legend(cand) and self.isVisible() else None

    def annotations_shown(self) -> dict[str, bool]:
        """Whether the next render draws the colour bar and the caption (after legend sharing)."""
        bar, caption = self._annotations()
        return {"colorbar": bar, "caption": caption}

    def _annotations(self) -> tuple[bool, bool]:
        if self._legend is not None:
            return self._legend, self._legend
        partner = self.legend_partner()
        if partner is None:
            return True, True
        bar = True
        spec = self._scene.colorbar_spec() if hasattr(self._scene, "colorbar_spec") else None
        if spec is not None:
            try:
                lo, hi = (float(v) for v in partner.colour_range())
            except Exception:                     # noqa: BLE001 - a partner without a usable range
                lo = hi = float("nan")
            tol = LEGEND_MATCH * max(abs(spec.vmax - spec.vmin), abs(hi - lo), 1e-30)
            bar = not (abs(lo - spec.vmin) <= tol and abs(hi - spec.vmax) <= tol)
        mine = str(self._scene.caption() if hasattr(self._scene, "caption") else "").strip()
        try:
            theirs = str(partner.caption() or "").strip()
        except Exception:                         # noqa: BLE001 - a partner without a caption
            theirs = ""
        return bar, not (mine and _caption_shape(mine) == _caption_shape(theirs))

    def _watch(self, w: QWidget | None) -> None:
        """Follow the partner's show / hide (View: Split -> 3D) so the bar comes back at once."""
        if w is self._watched:
            return
        if self._watched is not None:
            try:
                self._watched.removeEventFilter(self)
            except RuntimeError:
                pass
        self._watched = w
        if w is not None:
            w.installEventFilter(self)

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if obj is self._watched and ev.type() in (QEvent.Type.Show, QEvent.Type.Hide):
            self.update()
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------
    # flow in the volume (DESIGN.md 7.13)
    # ------------------------------------------------------------------
    def _alive(self, key: str) -> str | None:
        h = self._flow.get(key)
        if h is not None and h not in self._scene.layers():
            self._flow.pop(key, None)            # the page cleared the scene
            return None
        return h

    def set_smoke(self, volume, *, field: str = "dye", opacity: float = 1.0) -> None:
        """Show ``volume``'s smoke (None removes it); updates the existing smoke in place."""
        h = self._alive("smoke")
        if volume is None or field not in getattr(volume, "fields", {}):
            if h is not None:
                self._scene.remove(h)
                self._flow.pop("smoke", None)
            self.refresh()
            return
        if h is None:
            self._flow["smoke"] = self._scene.add_smoke(volume, field, opacity=opacity)
        else:
            self._scene.update_smoke(h, volume, field)
            self._scene.set_smoke_opacity(h, opacity)
        self.refresh()

    def set_streamlines(self, data, *, seeds=None, color_by: str | None = "speed", tubes: bool = True,
                        **kw) -> None:
        """Show streamlines: ``data`` a VolumeData (traced from ``seeds``: a rake dict, a preset
        name or points) or a traced ``viz3d.flow.StreamlineSet``; None removes them. Updates
        in place (same actor, colour range and tube radius) when they are already shown."""
        h = self._alive("streamlines")
        if data is None:
            if h is not None:
                self._scene.remove(h)
                self._flow.pop("streamlines", None)
            self._flow_lines = None
            self.refresh()
            return
        if h is None:
            self._flow["streamlines"] = self._scene.add_streamlines(data, seeds if seeds is not None else "rake",
                                                                    color_by=color_by, tubes=tubes, **kw)
        else:
            self._scene.update_streamlines(h, data, seeds=seeds)
        self._flow_lines = data
        self.refresh()

    def set_seed_rake(self, rake, *, editable: bool = True, body_bounds=None, domain_bounds=None) -> None:
        """Draw the seed rake (a rake dict or ``viz3d.flow.SeedRake``; None removes it).
        ``editable`` lets the user drag it by its handle (``rakeMoving`` / ``rakeChanged``)."""
        h = self._alive("rake")
        self._rake_editable = bool(editable) and rake is not None
        self.setMouseTracking(self._rake_editable)
        if rake is None:
            if h is not None:
                self._scene.remove(h)
                self._flow.pop("rake", None)
            self.refresh()
            return
        if self._rake_drag is not None:          # the user is holding it: their position wins
            return
        if h is None:
            self._flow["rake"] = self._scene.add_seed_rake(rake, body_bounds=body_bounds, domain_bounds=domain_bounds,
                                                           handle_marker=self._rake_editable)
        else:
            self._scene.update_seed_rake(h, rake, body_bounds=body_bounds, domain_bounds=domain_bounds)
        self.refresh()

    def seed_rake(self):
        """The rake shown (``viz3d.flow.SeedRake``) or None."""
        h = self._alive("rake")
        return self._scene.rake_of(h) if h is not None else None

    def clear_flow(self) -> None:
        """Remove the smoke, the streamlines and the rake."""
        for key in ("smoke", "streamlines", "rake"):
            h = self._flow.pop(key, None)
            if h is not None:
                self._scene.remove(h)
        self._flow_serial, self._flow_lines = None, None
        self._rake_editable = False
        self.refresh()

    def show_live_frame(self, frame, *, smoke: bool = True, streamlines: bool = True, rake: bool = True,
                        editable: bool = True) -> None:
        """Apply a 3D run's ``LiveFrame`` (DESIGN.md 7.13): its volume's smoke when the volume
        is new (``volume_serial``), its traced streamlines when new, and its rake. Frames
        without a volume leave what is shown in place, so a slow viewer never flashes."""
        vol = getattr(frame, "volume", None)
        serial = getattr(frame, "volume_serial", 0)
        if vol is not None and serial != self._flow_serial:
            self._flow_serial = serial
            if smoke and "dye" in vol.fields:
                self.set_smoke(vol)
        if not smoke and self._alive("smoke") is not None:
            self.set_smoke(None)
        lines = getattr(frame, "streamlines3d", None)
        if streamlines and lines is not None and lines is not self._flow_lines:
            self.set_streamlines(lines)
        elif not streamlines and self._alive("streamlines") is not None:
            self.set_streamlines(None)
        spec = getattr(frame, "seed_rake", None)
        if rake and spec is not None:
            self.set_seed_rake(spec, editable=editable, domain_bounds=_NO_CLAMP)
        elif not rake and self._alive("rake") is not None:
            self.set_seed_rake(None)

    def render_image(self, width: int | None = None, height: int | None = None, *, scale: float = 1.0) -> np.ndarray:
        """Render the current view off-screen at any size (exports, reports)."""
        w = width or max(1, self.width())
        h = height or max(1, self.height())
        return self._scene.render(w, h, scale=scale)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _camera_moved(self) -> None:
        self.cameraChanged.emit()
        self.refresh()

    def _on_theme(self, *_args) -> None:
        try:
            self._scene.set_theme(tokens())
        except Exception:
            log.exception("3D scene theme update failed")
        self.refresh()

    def _dpr(self) -> float:
        return float(self.devicePixelRatioF() or 1.0)

    def _ensure_image(self) -> QImage | None:
        dpr = self._dpr()
        pw, ph = max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr))
        if hasattr(self._scene, "set_annotations"):
            bar, caption = self._annotations()
            self._scene.set_annotations(colorbar=bar, caption=caption)    # bumps the revision only on change
        key = (self._scene.revision, pw, ph, round(dpr, 4), id(self._scene))
        if key != self._key or self._image is None:
            img = self._scene.render(pw, ph, scale=dpr)
            self._buf = np.ascontiguousarray(img)
            qimg = QImage(self._buf.data, pw, ph, 3 * pw, QImage.Format.Format_RGB888)
            qimg.setDevicePixelRatio(dpr)
            self._image = qimg
            self._key = key
            self.last_render_ms = float(getattr(self._scene, "last_render_ms", float("nan")))
            self.rendered.emit(self.last_render_ms)
        return self._image

    def sizeHint(self) -> QSize:
        return QSize(800, 520)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        tk = tokens()
        try:
            if not getattr(self._scene, "available", False):
                p.fillRect(self.rect(), qcolor(tk["surface"]))
                self._paint_text(p, tk, "3D view unavailable", getattr(self._scene, "error", ""))
                return
            img = self._ensure_image()
            if img is not None:
                p.drawImage(QRectF(0, 0, self.width(), self.height()), img)
            if not self._scene.layers() and self._placeholder[0]:
                self._paint_text(p, tk, *self._placeholder)
        except Exception as exc:
            log.exception("3D render failed")
            p.fillRect(self.rect(), qcolor(tk["surface"]))
            self._paint_text(p, tk, "3D render failed", f"{type(exc).__name__}: {exc}")
        finally:
            p.end()

    def _paint_text(self, p: QPainter, tk: dict, title: str, sub: str = "") -> None:
        from .. import theme
        f_title = theme.ui_font(weight=QFont.Weight.DemiBold, pt=9.0)
        f_sub = theme.ui_font(pt=9.0)
        fm = QFontMetricsF(f_title)
        r = QRectF(self.rect()).adjusted(24, 0, -24, 0)
        mid = r.center().y() - (fm.height() if sub else fm.height() / 2)
        p.setFont(f_title)
        p.setPen(qcolor(tk["text2"]))
        p.drawText(QRectF(r.left(), mid, r.width(), fm.height() * 1.3),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, title)
        if sub:
            p.setFont(f_sub)
            p.setPen(qcolor(tk["muted"]))
            p.drawText(QRectF(r.left(), mid + fm.height() * 1.4, r.width(), 80),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap, sub)

    # ------------------------------------------------------------------
    # mouse and keyboard
    # ------------------------------------------------------------------
    def _rake_hit(self, pos: QPointF) -> bool:
        """Is the cursor on the rake's handle?"""
        h = self._alive("rake")
        if not self._rake_editable or h is None:
            return False
        sr = self._scene.rake_of(h)
        if sr is None:
            return False
        dpr = self._dpr()
        px = self._scene.project(sr.centre[None, :])[0]
        if not np.all(np.isfinite(px)):
            return False
        return float(np.hypot(px[0] - pos.x() * dpr, px[1] - pos.y() * dpr)) <= RAKE_GRAB_PX * dpr

    def _rake_plane_point(self, pos: QPointF, normal: np.ndarray, point: np.ndarray) -> np.ndarray | None:
        dpr = self._dpr()
        o, d = self._scene.pick_ray(pos.x() * dpr, pos.y() * dpr)
        den = float(d @ normal)
        if abs(den) < 1e-9:
            return None
        t = float((point - o) @ normal) / den
        return o + t * d if t > 0 else None

    def _start_rake_drag(self, pos: QPointF) -> None:
        sr = self._scene.rake_of(self._flow["rake"])
        _, _, f = self._scene.camera.frame()
        normal = np.zeros(3)
        normal[int(np.argmax(np.abs(f)))] = 1.0        # move in the plane facing the camera's dominant axis
        grab = self._rake_plane_point(pos, normal, sr.centre)
        self._rake_drag = {"rake": sr, "normal": normal, "grab": grab if grab is not None else sr.centre.copy(),
                           "current": sr}

    def _drag_rake_to(self, pos: QPointF) -> None:
        d = self._rake_drag
        p = self._rake_plane_point(pos, d["normal"], d["grab"])
        if p is None:
            return
        delta = p - d["grab"]
        d["current"] = d["rake"].moved(*delta)
        self._scene.update_seed_rake(self._flow["rake"], d["current"])
        self.refresh()
        self.rakeMoving.emit(d["current"].spec())

    def mousePressEvent(self, ev) -> None:
        b = ev.button()
        mods = ev.modifiers()
        if b == Qt.MouseButton.LeftButton and self._rake_hit(ev.position()):
            self._start_rake_drag(ev.position())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        if b == Qt.MouseButton.LeftButton and not (mods & Qt.KeyboardModifier.ShiftModifier):
            self._drag = ("orbit", ev.position())
        elif b in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._drag = ("pan", ev.position())
        else:
            return super().mousePressEvent(ev)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        ev.accept()

    def mouseMoveEvent(self, ev) -> None:
        if self._rake_drag is not None:
            self._drag_rake_to(ev.position())
            ev.accept()
            return
        if self._drag is None:
            if self._rake_editable:            # hover feedback: the rake handle can be grabbed
                if self._rake_hit(ev.position()):
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.unsetCursor()
            return super().mouseMoveEvent(ev)
        mode, last = self._drag
        pos = ev.position()
        dx, dy = pos.x() - last.x(), pos.y() - last.y()
        self._drag = (mode, pos)
        if dx == 0 and dy == 0:
            return
        if mode == "orbit":
            self._scene.orbit(dx * ORBIT_DEG_PER_PX, dy * ORBIT_DEG_PER_PX)
        else:
            dpr = self._dpr()
            self._scene.pan(dx * dpr, dy * dpr)
        self._camera_moved()
        ev.accept()

    def mouseReleaseEvent(self, ev) -> None:
        if self._rake_drag is not None:
            d, self._rake_drag = self._rake_drag, None
            self.unsetCursor()
            self.rakeChanged.emit(d["current"].spec())
            ev.accept()
            return
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = None
            self.unsetCursor()
            self.fit()
            ev.accept()
            return
        super().mouseDoubleClickEvent(ev)

    def wheelEvent(self, ev) -> None:
        steps = ev.angleDelta().y() / 120.0
        if not steps:
            return super().wheelEvent(ev)
        dpr = self._dpr()
        pos = ev.position()
        self._scene.zoom(WHEEL_ZOOM_PER_NOTCH ** steps, at=(pos.x() * dpr, pos.y() * dpr))
        self._camera_moved()
        ev.accept()

    def keyPressEvent(self, ev) -> None:
        key = ev.key()
        if key == Qt.Key.Key_F:
            self.fit()
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._scene.zoom(WHEEL_ZOOM_PER_NOTCH)
            self._camera_moved()
        elif key == Qt.Key.Key_Minus:
            self._scene.zoom(1.0 / WHEEL_ZOOM_PER_NOTCH)
            self._camera_moved()
        else:
            return super().keyPressEvent(ev)
        ev.accept()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self.refresh()

    def closeEvent(self, ev) -> None:
        self._timer.stop()
        super().closeEvent(ev)


class View3DToolbar(ToolStrip):
    """Camera presets, Fit and an Edges toggle for a :class:`View3D`.

    As a page-level strip it sits above the viewport like the 2D
    ``ViewportToolbar`` (text buttons). With ``compact=True`` it floats inside
    the view as a borderless row of 24 px icon buttons with tooltips and no
    background of its own (guideline 5b), so it reads as part of the viewport
    rather than as another tool strip. Pages may add their own widgets with
    ``add_widget`` / ``add_separator``.
    """

    def __init__(self, view: View3D, parent: QWidget | None = None, *, compact: bool = False,
                 presets: Sequence[tuple[str, str, str]] = CAMERA_PRESETS) -> None:
        super().__init__(parent)
        self.view = view
        self.compact = bool(compact)
        self.preset_buttons: dict[str, QToolButton] = {}
        for key, text, tip in presets:
            b = self._button(text, f"{text} view: {tip[0].lower()}{tip[1:]}" if compact else tip,
                             preset_icon(key), on_click=lambda k=key: view.set_view(k))
            b.setProperty("preset", key)
            self.preset_buttons[key] = b
        if compact:
            self.row.addSpacing(SPACING + 2)
        else:
            self.add_separator()
        self.fit_button = self._button("Fit", "Fit: frame the model (double-click in the view, or F)",
                                       _theme_icon("fullscreen"), on_click=view.fit)
        self.edges_button = self._button("Edges", "Edges: show the mesh triangles", _theme_icon("mesh"),
                                         checkable=True)
        self.edges_button.setChecked(view.edges())
        self.edges_button.toggled.connect(view.set_edges)
        view._toolbars.add(self)
        if compact:
            self.setProperty("chrome", "")
            self.setFrameShape(QFrame.Shape.NoFrame)
            self.setAutoFillBackground(False)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.row.setContentsMargins(0, 0, 0, 0)
            self.row.setSpacing(0)
            self.setFixedHeight(CONTROL_H)
            self.adjustSize()
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        else:
            self.add_stretch()

    def _button(self, text: str, tip: str, icon: QIcon, *, checkable: bool = False, on_click=None) -> QToolButton:
        """A text button in a page strip, an icon-only 24 px button in the overlay."""
        if not self.compact:
            return self.add_button(None, text, tip, checkable=checkable, on_click=on_click)
        b = QToolButton()
        b.setProperty("variant", "icon")
        b.setIcon(icon)
        b.setIconSize(QSize(16, 16))
        b.setFixedSize(CONTROL_H, CONTROL_H)
        b.setAutoRaise(True)
        b.setToolTip(tip)
        b.setAccessibleName(text)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.FocusPolicy.TabFocus)     # a click must not take the keyboard from the view
        if on_click is not None:
            b.clicked.connect(lambda _=False: on_click())
        self.row.addWidget(b)
        return b

    def sync(self, view: View3D) -> None:
        """Reflect the view's edge state without re-triggering it."""
        b = self.edges_button
        if b.isChecked() != view.edges():
            b.blockSignals(True)
            b.setChecked(view.edges())
            b.blockSignals(False)
