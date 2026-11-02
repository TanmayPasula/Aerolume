"""pyqtgraph plots for the runtime pages, styled like publication figures.

Why pyqtgraph and not matplotlib here: the Simulate page redraws a 2000-point
coefficient history 20+ times a second while the solver runs, and every plot
in the app must pan / zoom interactively. pyqtgraph does both without leaving
the Qt event loop. matplotlib stays reserved for the static figures in
:mod:`aerolume.viz.figures` (report, PNG export).

Style (docs/ui_guidelines.md section 5): surface background, 1 px axes in
text2 drawn as a closed box, ticks pointing inwards, a faint grid (alpha
~0.15), 1.5 px lines in the six series colours, a small boxed legend only when
there is more than one series, axis labels with units, and averaging windows
as a light grey band. Colours come from :func:`aerolume.app.theme.tokens`;
every plot re-reads them on ``theme.changed``.

Force histories are drawn as *stacked* panels that share the time axis (Cd on
top, Cl, then Cm) rather than as three lines on one axis: Cd is an order of
magnitude smaller than Cl, so on a shared axis its fluctuations, which are what
tells a user whether the run settled, would be a flat line.
"""
from __future__ import annotations

import importlib
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

log = logging.getLogger(__name__)


def _guard_empty_axis_paint() -> None:
    """Paint pyqtgraph axes directly instead of through a cached QPicture.

    pyqtgraph 0.14 records every axis into a QPicture and replays it. With
    PySide6 6.11 that recording crashes with an access violation on the offscreen
    platform (tests, CI, headless screenshot rendering) and when an axis has been
    squeezed to zero size. Drawing the same specs straight onto the widget's
    painter produces identical pixels; only the per-axis cache is lost, which is
    negligible for the handful of axes on a page.
    """
    if getattr(pg.AxisItem.paint, "_aerolume_guard", False):
        return

    def paint(self, p, opt, widget):
        rect = self.geometry()
        if rect.width() < 1.0 or rect.height() < 1.0:
            return
        p.save()
        try:
            if self.style["tickFont"]:
                p.setFont(self.style["tickFont"])
            specs = self.generateDrawSpecs(p)
            if specs is not None:
                self.drawPicture(p, *specs)
        finally:
            p.restore()

    paint._aerolume_guard = True  # type: ignore[attr-defined]
    pg.AxisItem.paint = paint


_guard_empty_axis_paint()

pg.setConfigOptions(antialias=True)

# Light guideline palette (v4, the website's ink on paper), used only if app/theme.py cannot be
# imported (tests of this module in isolation). Same keys as theme.tokens().
FALLBACK_TOKENS: dict = {
    "bg": "#efefef", "bg2": "#efefef", "panel": "#f6f6f6", "surface": "#ffffff", "border": "#d6d6d6",
    "border_strong": "#b8b8b8", "text": "#0e0e0e", "text2": "#474747", "muted": "#5c5c5c",
    "faint": "#8e8e8e", "accent": "#2463c4", "accent2": "#1b4f9e", "accent_soft": "#e2e2e2",
    "good": "#17703c", "warn": "#8f5a0b", "bad": "#b02d27", "grid": "#e6e6e6", "plot_bg": "#ffffff",
    "hover": "#ebebeb", "pressed": "#dcdcdc",
    "series":["#2463c4", "#d9730d", "#2f855a", "#b8322b", "#6b46c1", "#5f6b7a"],
}

GRID_ALPHA = 0.15
LINE_W = 1.5
AXIS_PT = 8.0
LABEL_PT = 8.5


# ---------------------------------------------------------------------------
# lazy imports and theme access
# ---------------------------------------------------------------------------

_IMPORT_ERRORS: dict[str, str] = {}
_FAILED_AT: dict[str, float] = {}
_RETRY_S = 5.0


def lazy_import(name: str):
    """Import a sibling module on demand; ``None`` (and a remembered reason) if it fails.

    The runtime pages depend on solver, viz, report and export modules that are
    developed independently. Importing them lazily means a missing or broken
    module degrades one feature with a readable message instead of preventing
    the whole application from starting. Failures are remembered for a few
    seconds because this is called from paint paths at frame rate, and a failed
    import searches the file system every time.
    """
    mod = sys.modules.get(name)
    # a module another thread is still executing (jobs.prewarm) is already in sys.modules,
    # half-initialised; import_module below then waits for it on the import lock
    if mod is not None and not getattr(getattr(mod, "__spec__", None), "_initializing", False):
        return mod
    failed = _FAILED_AT.get(name)
    if failed is not None and time.monotonic() - failed < _RETRY_S:
        return None
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - ImportError, but also errors raised while importing
        _IMPORT_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        _FAILED_AT[name] = time.monotonic()
        log.debug("optional module %s unavailable: %s", name, exc)
        return None
    _IMPORT_ERRORS.pop(name, None)
    _FAILED_AT.pop(name, None)
    return mod


def import_error(name: str) -> str:
    """Why the last :func:`lazy_import` of ``name`` failed (empty if it did not)."""
    return _IMPORT_ERRORS.get(name, "")


def tokens() -> dict:
    """Active theme colours (``theme.tokens()`` merged over the fallback palette)."""
    theme = lazy_import("aerolume.app.theme")
    if theme is not None and hasattr(theme, "tokens"):
        try:
            merged = dict(FALLBACK_TOKENS)
            merged.update(theme.tokens())
            if not merged.get("series"):
                merged["series"] = FALLBACK_TOKENS["series"]
            return merged
        except Exception:
            log.exception("theme.tokens() failed; using fallback colours")
    return dict(FALLBACK_TOKENS)


def active_icon_token(key: str) -> str:
    """The token a checked icon is drawn in: ``theme.ACTIVE_ICON_TOKEN`` applied to ``key``.

    Theme v4 draws checked chrome icons in ink rather than the blue ``accent`` most call sites
    still ask for. The theme's own icon engine applies that mapping; the inline icon engines of
    the 3D view and of the live-edit tools call this, so a checked tool there is ink like every
    other checked button. Without the theme module the key comes back unchanged.
    """
    theme = lazy_import("aerolume.app.theme")
    mapping = getattr(theme, "ACTIVE_ICON_TOKEN", None) if theme is not None else None
    return str(mapping.get(key, key)) if isinstance(mapping, dict) else key


class _ThemeRelay(QObject):
    """Forwards ``theme.changed`` to one callback for as long as its owner lives.

    A plain ``sig.connect(lambda: widget.apply_theme())`` keeps the lambda
    connected after the widget is destroyed, and the next theme switch calls
    into a deleted C++ object ("Internal C++ object already deleted"). The relay
    is a child of the widget, so Qt deletes it -- and with it the connection --
    together with the widget.
    """

    def __init__(self, owner: QObject, callback: Callable[[], None]) -> None:
        super().__init__(owner)
        self._callback = callback

    def fire(self, *_args) -> None:
        try:
            self._callback()
        except RuntimeError:            # the owner's C++ side is already gone (deleteLater pending)
            log.debug("theme callback of a deleted widget skipped")


def adopt_plot_menu(plot_item, host: QWidget) -> None:
    """Hand pyqtgraph's "Plot Options" menu of ``plot_item`` to ``host``, so Qt deletes it with the host.

    pyqtgraph (0.14) builds ``PlotItem.ctrlMenu`` as a parentless QMenu with six
    submenus and frees it only in ``PlotItem.close()``, which nothing calls when a
    page is deleted with its plots inside. Each plot then left seven top-level
    menus behind: 6,440 after the run-area tests, and every later theme switch
    restyled them (1.0 s in the full suite against 0.22 s with the same pages and
    no leftovers). Our plots disable that menu anyway (``setMenuEnabled(False)``).
    The window flags are passed on so the menu stays a popup; ``PlotItem.close()``
    still works, because it re-parents the menu to None before dropping it.
    """
    menu = getattr(plot_item, "ctrlMenu", None)
    if menu is None or host is None:
        return
    try:
        menu.setParent(host, menu.windowFlags())
    except RuntimeError:        # the plot or the host is already deleted on the C++ side
        pass


def release_plot_menus(plot_items) -> None:
    """Undo :func:`adopt_plot_menu` for plots about to be discarded from a host that lives on (a layout
    cleared and rebuilt): their menus go back to Python and are freed with their plot, instead of
    piling up under the host until it is deleted."""
    for pi in plot_items:
        menu = getattr(pi, "ctrlMenu", None)
        if menu is not None:
            try:
                menu.setParent(None)
            except RuntimeError:
                pass


def connect_theme(callback: Callable[[], None]) -> bool:
    """Call ``callback()`` whenever the theme switches. Returns False if there is no hook.

    A bound method of a QObject (``self.update``, ``self.apply_theme``) is
    connected through a :class:`_ThemeRelay` owned by that object, so the
    connection ends when the object is destroyed (see the relay's docstring).
    """
    theme = lazy_import("aerolume.app.theme")
    sig = getattr(theme, "changed", None) if theme is not None else None
    if sig is None or not hasattr(sig, "connect"):
        return False
    owner = getattr(callback, "__self__", None)
    try:
        if isinstance(owner, QObject):
            sig.connect(_ThemeRelay(owner, callback).fire)
        else:
            sig.connect(lambda *_: callback())
        return True
    except (TypeError, RuntimeError, AttributeError):
        log.debug("theme hook %r is not connectable", sig)
        return False


def qcolor(c: str | QColor | Sequence[int], alpha: int | None = None) -> QColor:
    """QColor from a hex string / QColor / tuple, optionally with alpha 0..255."""
    col = QColor(c) if not isinstance(c, (tuple, list)) else QColor(*c)
    if alpha is not None:
        col.setAlpha(int(alpha))
    return col


def series_color(i: int, tk: dict | None = None) -> QColor:
    """The i-th categorical series colour of the theme (cycles)."""
    s = (tk or tokens())["series"]
    return QColor(s[i % len(s)])


def make_pen(color, width: float = LINE_W, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
    """Cosmetic (pixel-width) pen, so line weights do not change with zoom."""
    pen = QPen(qcolor(color) if not isinstance(color, QColor) else color)
    pen.setWidthF(width)
    pen.setStyle(style)
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap if style != Qt.PenStyle.SolidLine else Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    """Round tick values (1, 2, 2.5, 5 x 10^k) covering [vmin, vmax]."""
    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return [vmin] if math.isfinite(vmin) else []
    raw = (vmax - vmin) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), 10 * mag)
    start = math.ceil(vmin / step - 1e-9) * step
    ticks = []
    v = start
    while v <= vmax + 1e-9 * step:
        ticks.append(0.0 if abs(v) < 1e-12 * step else v)
        v += step
    return ticks


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def decade_scale(values, *, low: float = 1e-2, high: float = 1e4) -> tuple[float, str]:
    """(factor, label suffix) that brings data of tiny or huge magnitude to plain tick numbers.

    pyqtgraph labels an axis spanning 0..8e-5 with "8e-05", "6e-05" ..., which
    reads as noise. Multiplying the data by ``factor`` and appending the suffix
    (" ×10⁻⁵") to the axis label gives ticks 0, 2, 4, 6, 8 instead, the
    convention of printed plots. Data whose largest magnitude lies in
    [low, high) is left alone: (1.0, "").
    """
    v = np.asarray(values, dtype=float)
    v = np.abs(v[np.isfinite(v)])
    m = float(v.max()) if v.size else 0.0
    if m <= 0.0 or low <= m < high:
        return 1.0, ""
    e = math.floor(math.log10(m))
    return 10.0 ** (-e), f" ×10{str(e).translate(_SUPERSCRIPT)}"


def _font(pt: float) -> QFont:
    theme = lazy_import("aerolume.app.theme")
    f = theme.ui_font(pt=pt) if theme is not None and hasattr(theme, "ui_font") else QFont()
    f.setPointSizeF(pt)
    return f


def _typographic_ticks(ax: pg.AxisItem) -> None:
    """Make one axis label negative ticks with a real minus sign (U+2212), not a hyphen.

    The tables, the colour bar and the readouts all print "−0.5"; pyqtgraph's
    "-0.5" next to them reads as a different, shorter dash, and in a Cp plot,
    where the axis is inverted, the sign is the whole point. Wrapped per axis
    instance (not on the class) so pyqtgraph axes elsewhere stay untouched.
    """
    if getattr(ax, "_aerolume_minus", False):
        return
    plain = ax.tickStrings

    def tick_strings(values, scale, spacing):
        return [str(t).replace("-", "−") for t in plain(values, scale, spacing)]

    ax.tickStrings = tick_strings
    ax._aerolume_minus = True


def style_plot_item(pi: pg.PlotItem, tk: dict, *, xlabel: str = "", ylabel: str = "", title: str | None = None,
                    grid: tuple[bool, bool] = (True, True), box: bool = True) -> None:
    """Apply the publication style to one PlotItem (axes box, inward ticks, faint grid, labels).

    Units go into the label text ("Drag [N/m]") rather than pyqtgraph's ``units=``
    argument, which would silently rescale values with SI prefixes.
    """
    tick_font = _font(AXIS_PT)
    # The closing box is the ViewBox border, not shown top/right axes: a value-less
    # axis collapses to zero height and pyqtgraph 0.14 then records its ticks into a
    # zero-sized QPicture, which crashes PySide6 6.11 with an access violation the
    # first time the plot is painted.
    for name in ("top", "right"):
        pi.hideAxis(name)
    pi.getViewBox().setBorder(make_pen(tk["text2"], 1.0) if box else None)
    for name in ("left", "bottom"):
        ax = pi.getAxis(name)
        ax.setPen(make_pen(tk["text2"], 1.0))
        ax.setTextPen(qcolor(tk["text2"]))
        ax.setTickFont(tick_font)
        ax.setStyle(tickLength=-4, tickTextOffset=4)
        _typographic_ticks(ax)
    style = {"color": tk["text2"], "font-size": f"{LABEL_PT}pt"}
    if xlabel:
        pi.setLabel("bottom", xlabel, **style)
    if ylabel:
        pi.setLabel("left", ylabel, **style)
    for name in ("left", "bottom"):
        # No SI-prefix rescaling, ever. pyqtgraph 0.14 ignores enableAutoSIPrefix(False) when it
        # recomputes the prefix: that call itself stores a x1000 scale if a labelled axis spans
        # 0..1 at that moment, and the ticks then silently read 50, 100, 150 for Cd = 0.05..0.15.
        # Empty enable-ranges (re-applied after every setLabel, which resets them) turn it off.
        ax = pi.getAxis(name)
        ax.setSIPrefixEnableRanges(())
        ax.enableAutoSIPrefix(False)
    if title:
        pi.setTitle(title, color=tk["text"], size=f"{LABEL_PT + 0.5}pt")
    pi.showGrid(x=grid[0], y=grid[1], alpha=GRID_ALPHA)
    pi.getViewBox().setBackgroundColor(None)


def style_legend(legend: pg.LegendItem, tk: dict) -> None:
    """Theme a legend, its existing entries included.

    pyqtgraph's ``setLabelTextColor`` only records the colour for later entries (``LabelItem.setAttr``
    does not re-render): after a switch to the dark theme the entries already there stayed dark grey on
    the dark plot, unreadable. Each label's text is set again with the new colour.
    """
    legend.setBrush(qcolor(tk["surface"], 235))
    legend.setPen(make_pen(tk["border"], 1.0))
    color = qcolor(tk["text"])
    legend.setLabelTextColor(color)
    legend.setLabelTextSize(f"{AXIS_PT}pt")
    for _sample, lab in list(getattr(legend, "items", []) or []):
        try:
            lab.setText(lab.text, color=color, size=f"{AXIS_PT}pt")
        except (AttributeError, TypeError, RuntimeError):
            continue


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

class ThemedPlot(pg.PlotWidget):
    """A PlotWidget in the publication style, with labelled axes and an optional legend."""

    #: Legend anchor offset in px; pyqtgraph reads the sign as the corner (x < 0: right, y < 0: bottom).
    LEGEND_OFFSET: ClassVar[tuple[int, int]] = (-6, 6)

    def __init__(self, parent: QWidget | None = None, *, title: str | None = None,
                 xlabel: str = "", ylabel: str = "", legend: bool = False,
                 equal_aspect: bool = False) -> None:
        super().__init__(parent)
        self._title = title
        self._xlabel, self._ylabel = xlabel, ylabel
        self.legend = (self.addLegend(offset=self.LEGEND_OFFSET, verSpacing=-6, labelTextSize=f"{AXIS_PT}pt")
                       if legend else None)
        if self.legend is not None:
            self.legend.setVisible(False)
        pi = self.getPlotItem()
        pi.setMenuEnabled(False)
        adopt_plot_menu(pi, self)
        pi.hideButtons()
        if equal_aspect:
            self.setAspectLocked(True)
        self.setMinimumHeight(120)
        self.apply_theme()
        connect_theme(self.apply_theme)

    def close(self) -> bool:
        """Close; a second call is a no-op. pyqtgraph's ``PlotWidget.close`` drops its plot item and a
        second call dereferenced None (a test's own close followed by the suite's clean-up raised)."""
        if self.getPlotItem() is None:
            return True
        return super().close()

    def apply_theme(self) -> None:
        """Re-read the theme tokens and restyle background, axes, grid and legend."""
        if self.getPlotItem() is None:      # PlotWidget.close() drops its plot item; the widget lives on
            return
        tk = tokens()
        self.setBackground(tk["surface"])
        style_plot_item(self.getPlotItem(), tk, xlabel=self._xlabel, ylabel=self._ylabel, title=self._title)
        if self.legend is not None:
            style_legend(self.legend, tk)

    def set_labels(self, xlabel: str | None = None, ylabel: str | None = None) -> None:
        if xlabel is not None:
            self._xlabel = xlabel
        if ylabel is not None:
            self._ylabel = ylabel
        self.apply_theme()


class LinePlot(ThemedPlot):
    """Named line series that can be replaced cheaply (``setData``) on every update."""

    def __init__(self, *args, **kw) -> None:
        kw.setdefault("legend", True)
        super().__init__(*args, **kw)
        self._items: dict[str, pg.PlotDataItem] = {}
        self._legend_count = 0

    def set_series(self, name: str, x, y, *, color=None, width: float = LINE_W,
                   style: Qt.PenStyle = Qt.PenStyle.SolidLine, symbol: str | None = None,
                   legend: bool = True) -> pg.PlotDataItem:
        """Create or update a series. ``color`` may be a hex string, QColor or series index."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if isinstance(color, int):
            color = series_color(color)
        color = color if color is not None else series_color(len(self._items))
        pen = make_pen(color, width, style)
        item = self._items.get(name)
        if item is None:
            item = self.plot(x, y, pen=pen, name=name if legend else None,
                             symbol=symbol, symbolSize=4, symbolPen=None,
                             symbolBrush=qcolor(color) if symbol else None)
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
            self._items[name] = item
            if legend:
                self._legend_count += 1
            if self.legend is not None:
                # a legend with a single entry only hides data; the axis label names it
                self.legend.setVisible(self._legend_count >= 2)
        else:
            item.setPen(pen)
            item.setData(x, y)
        return item

    def series_names(self) -> list[str]:
        return list(self._items)

    def clear_series(self) -> None:
        for item in self._items.values():
            self.removeItem(item)
        self._items.clear()
        self._legend_count = 0
        if self.legend is not None:
            self.legend.clear()
            self.legend.setVisible(False)  # an empty legend still paints its frame


#: Share of the run, from its start, that the vertical range of a history ignores.
STARTUP_SKIP = 0.05


def robust_history_range(t: np.ndarray, y: np.ndarray, *, skip: float = STARTUP_SKIP,
                         pad: float = 0.12, t_min: float | None = None) -> tuple[float, float] | None:
    """Vertical range for a force history that is not dominated by the start-up spike.

    An impulsively started solver sees an enormous force in its first steps
    (Cd of 20 or more for a body whose settled Cd is 0.1): auto-ranging on that
    squeezes the part of the trace an engineer actually reads - the settling
    and the oscillation in the averaging window - into a flat line. The range
    is therefore taken from the 0.5-99.5 percentiles of the samples after the
    first ``skip`` of the run, plus ``pad`` of the span on each side. The early
    samples are still drawn; they simply run off the top of the panel.

    A run in progress has only a few samples, all of them close to the spike,
    so a fraction of the samples seen so far does not get past it. ``t_min``
    (in the history's time unit) then skips by time instead: samples before
    it are left out while later ones exist; before that the 2nd-98th
    percentile of the newest half is used, which excludes the first steps.
    Returns None when there is not enough finite data.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(t), len(y))
    ok = np.isfinite(t[:n]) & np.isfinite(y[:n])
    if ok.sum() < 2:
        return None
    tt, yy = t[:n][ok], y[:n][ok]
    pct = (0.5, 99.5)
    if t_min is not None:
        sel = yy[tt >= max(t_min, tt[0] + skip * (tt[-1] - tt[0]))]
        if sel.size < 4:
            sel, pct = yy[len(yy) // 2:], (2.0, 98.0)
    else:
        sel = yy[tt >= tt[0] + skip * (tt[-1] - tt[0])]
        if sel.size < 8:
            sel = yy
    lo, hi = (float(v) for v in np.percentile(sel, pct))
    span = hi - lo
    if span <= 1e-12:
        span = max(abs(hi) * 0.1, 1e-3)
    return lo - pad * span, hi + pad * span


class HistoryPlot(pg.GraphicsLayoutWidget):
    """Force-coefficient histories as stacked panels sharing one time axis.

    ``keys`` picks the panels (``("cd", "cl")`` for the live monitor,
    ``("cd", "cl", "cm")`` on the Results page). The averaging window is a light
    grey band on every panel, with the mean over the window as a thin dashed line
    when ``show_mean`` is set: that window is what turns a noisy trace into the
    number in the Summary table, so the two are always shown together. The
    vertical ranges leave out the start-up spike (:func:`robust_history_range`).
    """

    #: Axis labels (HTML: pyqtgraph renders rich text, so the subscripts are real). Section
    #: (2D) coefficients are lower case; :data:`LABELS_3D` has the 3D / aircraft capitals.
    LABELS: ClassVar[dict[str, str]] = {"cd": "C<sub>d</sub>", "cl": "C<sub>l</sub>", "cm": "C<sub>m</sub>",
                                        "cs": "C<sub>s</sub>", "c_roll": "C<sub>roll</sub>", "cn": "C<sub>n</sub>"}
    LABELS_3D: ClassVar[dict[str, str]] = {"cd": "C<sub>D</sub>", "cs": "C<sub>S</sub>", "cl": "C<sub>L</sub>",
                                           "c_roll": "C<sub>roll</sub>", "cm": "C<sub>m</sub>",
                                           "cn": "C<sub>n</sub>"}
    #: Plain names, for code and tests that read which panels hold data.
    NAMES: ClassVar[dict[str, str]] = {"cd": "Cd", "cl": "Cl", "cm": "Cm", "cs": "CS", "c_roll": "C_roll",
                                       "cn": "Cn"}

    def __init__(self, parent: QWidget | None = None, *, keys: Sequence[str] = ("cd", "cl"),
                 xlabel: str = "t · U/L", show_mean: bool = False, show_cm: bool | None = None,
                 compact: bool = False, labels: dict[str, str] | None = None, **_ignored) -> None:
        super().__init__(parent)
        if show_cm is not None:
            keys = ("cd", "cl", "cm") if show_cm else ("cd", "cl")
        self.keys = tuple(keys)
        self.labels = dict(labels or self.LABELS)
        self.show_mean = show_mean
        self._xlabel = xlabel
        self._compact = compact
        self.ci.setContentsMargins(4, 6, 10, 4)
        self.ci.setSpacing(0 if compact else 4)
        self._xlimits: tuple[float, float] | None = None
        self.t_min: float | None = None           # see robust_history_range
        self.skip = STARTUP_SKIP                  # share of the x span left out of the y range
        self._build_panels()
        connect_theme(self.apply_theme)

    def _build_panels(self) -> None:
        """(Re)create one stacked panel per key, sharing the time axis."""
        if self.minimumHeight() != self.maximumHeight():      # a page that fixed the height keeps it
            self.setMinimumHeight(90 * len(self.keys) if not self._compact else 72 * len(self.keys))
        self.panels: dict[str, pg.PlotItem] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._bands: dict[str, pg.LinearRegionItem] = {}
        self._means: dict[str, pg.InfiniteLine] = {}
        self._yranges: dict[str, tuple[float, float]] = {}
        first = None
        for i, key in enumerate(self.keys):
            pi = self.ci.addPlot(row=i, col=0, axisItems=self._axis_items())
            pi.setMenuEnabled(False)
            adopt_plot_menu(pi, self)
            pi.hideButtons()
            pi.getAxis("left").setWidth(54)
            if first is None:
                first = pi
            else:
                pi.setXLink(first)
            band = pg.LinearRegionItem(movable=False)
            band.setZValue(-10)
            band.hide()
            pi.addItem(band, ignoreBounds=True)
            mean = pg.InfiniteLine(angle=0, movable=False)
            mean.hide()
            pi.addItem(mean, ignoreBounds=True)
            curve = pi.plot([], [])
            curve.setClipToView(True)
            curve.setDownsampling(auto=True, method="peak")
            self.panels[key], self._curves[key], self._bands[key], self._means[key] = pi, curve, band, mean
        self.apply_theme()
        if self._xlimits is not None:
            self.set_x_limits(self._xlimits)

    def _axis_items(self) -> dict | None:
        """Compact monitors (Simulate's 240 px column, three 70 px panels) get at most three ticks per axis
        with SI labels: pyqtgraph's own put "14 12 10 0.8" on top of each other."""
        if not self._compact:
            return None
        from .diagnostics import compact_axis_class
        axis = compact_axis_class()
        return {"left": axis("left"), "bottom": axis("bottom")}

    def set_keys(self, keys: Sequence[str], labels: dict[str, str] | None = None) -> None:
        """Show other panels (e.g. CD, CS, CL of a yawed 3D run); a no-op when nothing changes.

        The data is cleared: the caller sets the history again for the new panels.
        """
        keys = tuple(keys)
        labels = dict(labels or self.labels)
        if keys == self.keys and labels == self.labels:
            return
        self.keys, self.labels = keys, labels
        release_plot_menus(self.panels.values())
        self.ci.clear()
        self._build_panels()

    def apply_theme(self) -> None:
        tk = tokens()
        self.setBackground(tk["surface"])
        last = self.keys[-1]
        for i, key in enumerate(self.keys):
            pi = self.panels[key]
            style_plot_item(pi, tk, xlabel=self._xlabel if key == last else "", ylabel=self.labels.get(key, key))
            pi.getAxis("bottom").setStyle(showValues=key == last)
            self._curves[key].setPen(make_pen(series_color(i, tk), LINE_W))
            band = self._bands[key]
            band.setBrush(qcolor(tk["muted"], 30))
            for ln in band.lines:
                ln.setPen(make_pen(qcolor(tk["muted"], 0), 0.0))
            self._means[key].setPen(make_pen(qcolor(tk["text2"], 200), 1.0, Qt.PenStyle.DashLine))

    def set_x_limits(self, limits: tuple[float, float] | None) -> None:
        """Fix the shared time axis to ``(start, end)``, or None to follow the data.

        A run in progress knows its end time from the plan; a fixed axis lets
        the trace grow from left to right instead of auto-ranging around the
        first sample (-40..40 for a single point), and the reader sees at a
        glance how much of the run is done.
        """
        ok = limits is not None and all(map(math.isfinite, limits)) and limits[1] > limits[0]
        self._xlimits = (float(limits[0]), float(limits[1])) if ok else None
        for pi in self.panels.values():
            if self._xlimits is None:
                pi.enableAutoRange(x=True)
            else:
                pi.enableAutoRange(x=False)
                pi.setXRange(*self._xlimits, padding=0.0)

    def x_limits(self) -> tuple[float, float] | None:
        return self._xlimits

    def set_history(self, history: dict | None, avg_window: tuple[float, float] | None = None) -> None:
        """Update from a ``{"t","cd","cl","cm"}`` dict; ``avg_window`` in the same time units."""
        if not history or "t" not in history or len(history["t"]) == 0:
            self._yranges.clear()
            for key in self.keys:
                self._curves[key].setData([], [])
                self._bands[key].hide()
                self._means[key].hide()
                self.panels[key].enableAutoRange(x=self._xlimits is None, y=True)
            return
        t = np.asarray(history["t"], dtype=float)
        win_ok = (avg_window is not None and all(map(math.isfinite, avg_window))
                  and avg_window[1] > avg_window[0])
        for key in self.keys:
            if key not in history:
                self._curves[key].setData([], [])
                continue
            y = np.asarray(history[key], dtype=float)
            n = min(len(t), len(y))
            self._curves[key].setData(t[:n], y[:n])
            yr = robust_history_range(t[:n], y[:n], skip=self.skip, t_min=self.t_min)
            pi = self.panels[key]
            if self._xlimits is not None:
                pi.enableAutoRange(x=False)
                pi.setXRange(*self._xlimits, padding=0.0)
            if yr is not None:
                self._yranges[key] = yr
                pi.enableAutoRange(x=self._xlimits is None, y=False)
                pi.setYRange(*yr, padding=0)
            band, mean = self._bands[key], self._means[key]
            if win_ok:
                band.setRegion(avg_window)
                band.show()
                sel = (t[:n] >= avg_window[0]) & (t[:n] <= avg_window[1]) & np.isfinite(y[:n])
                if self.show_mean and sel.any():
                    mean.setValue(float(np.mean(y[:n][sel])))
                    mean.show()
                else:
                    mean.hide()
            else:
                band.hide()
                mean.hide()

    def series_names(self) -> list[str]:
        """Labels of the panels that currently hold data (for tests and legends)."""
        return [self.NAMES.get(k, k) for k in self.keys if self._curves[k].xData is not None
                and len(self._curves[k].xData)]

    def autoRange(self) -> None:
        """Fit the time axis to the data and restore the robust vertical ranges."""
        for key, pi in self.panels.items():
            yr = self._yranges.get(key)
            if yr is None:
                pi.enableAutoRange()
                pi.autoRange()
            else:
                pi.enableAutoRange(x=self._xlimits is None, y=False)
                pi.setYRange(*yr, padding=0)
            if self._xlimits is not None:
                pi.enableAutoRange(x=False)
                pi.setXRange(*self._xlimits, padding=0.0)

    def y_range(self, key: str) -> tuple[float, float] | None:
        """The vertical range last applied to a panel (None while it auto-ranges)."""
        return self._yranges.get(key)

    def clear_series(self) -> None:
        self.set_history(None)


class MultiHistoryPlot(pg.GraphicsLayoutWidget):
    """Several runs' force histories overlaid, one stacked panel per coefficient (Compare page).

    Each run keeps its series colour across panels; the legend sits in the top
    panel only. Every run's averaging window is drawn as a short thick tick of
    its own colour along the bottom of each panel (bands of several runs on top
    of each other would grey the whole plot out). The vertical ranges cover the
    robust ranges (:func:`robust_history_range`) of all runs, so one run's
    start-up spike does not flatten the others.
    """

    def __init__(self, parent: QWidget | None = None, *, keys: Sequence[str] = ("cd", "cl"),
                 labels: dict[str, str] | None = None, xlabel: str = "t · U/L") -> None:
        super().__init__(parent)
        self.keys = tuple(keys)
        self.labels = dict(labels or HistoryPlot.LABELS_3D)
        self._xlabel = xlabel
        self.ci.setContentsMargins(4, 6, 10, 4)
        self.ci.setSpacing(4)
        self._runs: list[tuple[str, dict, tuple | None, int]] = []
        self._build()
        connect_theme(self.apply_theme)

    def _build(self) -> None:
        self.setMinimumHeight(80 * len(self.keys))
        self.panels: dict[str, pg.PlotItem] = {}
        first = None
        for i, key in enumerate(self.keys):
            pi = self.ci.addPlot(row=i, col=0)
            pi.setMenuEnabled(False)
            adopt_plot_menu(pi, self)
            pi.hideButtons()
            pi.getAxis("left").setWidth(54)
            if first is None:
                first = pi
            else:
                pi.setXLink(first)
            self.panels[key] = pi
        self.legend = pg.LegendItem(offset=(-6, 6), verSpacing=-6, labelTextSize=f"{AXIS_PT}pt")
        self.legend.setParentItem(first.getViewBox())
        self.legend.setVisible(False)
        self.apply_theme()

    def set_keys(self, keys: Sequence[str], labels: dict[str, str] | None = None) -> None:
        keys = tuple(keys)
        if keys == self.keys and (labels is None or dict(labels) == self.labels):
            return
        self.keys = keys
        if labels is not None:
            self.labels = dict(labels)
        runs = list(self._runs)
        release_plot_menus(self.panels.values())
        self.ci.clear()
        self._runs = []
        self._build()
        for name, h, win, col in runs:
            self.add_run(name, h, win, col)

    def apply_theme(self) -> None:
        tk = tokens()
        self.setBackground(tk["surface"])
        last = self.keys[-1]
        for key in self.keys:
            pi = self.panels[key]
            style_plot_item(pi, tk, xlabel=self._xlabel if key == last else "", ylabel=self.labels.get(key, key))
            pi.getAxis("bottom").setStyle(showValues=key == last)
        style_legend(self.legend, tk)
        runs = list(self._runs)
        if runs:
            self.clear_runs()
            for name, h, win, col in runs:
                self.add_run(name, h, win, col)

    def clear_runs(self) -> None:
        for pi in self.panels.values():
            pi.clear()
        self.legend.clear()
        self.legend.setVisible(False)
        self._runs = []

    def add_run(self, name: str, history: dict | None, avg_window: tuple[float, float] | None = None,
                color: int | str | QColor = 0) -> None:
        """Overlay one run's ``{"t", key...}`` history in ``color`` (a series index or colour)."""
        if not history or "t" not in history or len(history["t"]) == 0:
            return
        self._runs.append((name, history, avg_window, color))
        col = series_color(color) if isinstance(color, int) else qcolor(color)
        t = np.asarray(history["t"], dtype=float)
        for i, key in enumerate(self.keys):
            pi = self.panels[key]
            if key not in history:
                continue
            y = np.asarray(history[key], dtype=float)
            n = min(len(t), len(y))
            item = pi.plot(t[:n], y[:n], pen=make_pen(col, LINE_W))
            item.setClipToView(True)
            item.setDownsampling(auto=True, method="peak")
            if i == 0:
                self.legend.addItem(item, name)
            ok = avg_window is not None and all(map(math.isfinite, avg_window)) and avg_window[1] > avg_window[0]
            if ok:
                # the averaging window as a tick along the bottom of the panel, in the run's colour
                tick = pg.PlotCurveItem([avg_window[0], avg_window[1]], [0, 0], pen=make_pen(col, 3.0))
                tick.setZValue(5)
                pi.addItem(tick, ignoreBounds=True)
                tick._aerolume_window = True          # repositioned by _fit_ranges
        self.legend.setVisible(len(self._runs) >= 2)
        self._fit_ranges()

    def _fit_ranges(self) -> None:
        for key, pi in self.panels.items():
            lo, hi = math.inf, -math.inf
            for _name, h, _w, _c in self._runs:
                if key in h:
                    yr = robust_history_range(np.asarray(h["t"]), np.asarray(h[key]))
                    if yr is not None:
                        lo, hi = min(lo, yr[0]), max(hi, yr[1])
            if math.isfinite(lo) and math.isfinite(hi):
                pi.enableAutoRange(x=True, y=False)
                pi.setYRange(lo, hi, padding=0)
                for item in pi.items:
                    if getattr(item, "_aerolume_window", False):
                        xs = item.xData
                        y0 = lo + 0.02 * (hi - lo)
                        item.setData(xs, [y0, y0])

    def run_names(self) -> list[str]:
        return [r[0] for r in self._runs]


class CpPlot(LinePlot):
    """Pressure coefficient along the chord. y is inverted (suction up), the aerodynamic convention.

    The legend sits in the bottom-right corner, and :meth:`autoRange` leaves
    room for it below the data: with suction up, the bottom right (positive Cp
    at the trailing edge) is the one corner no Cp curve ever reaches, whereas
    the default top-right corner is exactly where the upper-surface curves
    recover towards the trailing edge.
    """

    LEGEND_OFFSET: ClassVar[tuple[int, int]] = (-6, -6)
    #: Share of the data's Cp span added on the positive (bottom) side per legend row.
    LEGEND_ROOM_PER_ROW = 0.075

    def __init__(self, parent: QWidget | None = None, *, xlabel: str = "x / c", **kw) -> None:
        super().__init__(parent, xlabel=xlabel, ylabel="C<sub>p</sub>", **kw)
        self.getPlotItem().invertY(True)

    def autoRange(self, *args, **kw) -> None:
        # PlotWidget forwards autoRange through __getattr__, which super() does not see
        self.getPlotItem().getViewBox().autoRange(*args, **kw)
        rows = self._legend_count if self.legend is not None and self.legend.isVisible() else 0
        if not rows:
            return
        (_, _), (y0, y1) = self.getPlotItem().getViewBox().viewRange()
        span = (y1 - y0) or 1.0
        self.setYRange(y0, y1 + span * self.LEGEND_ROOM_PER_ROW * (rows + 1), padding=0)


class GeometryPlot(ThemedPlot):
    """Equal-aspect view of outlines with optional Cp arrows or a value-coloured outline."""

    def __init__(self, parent: QWidget | None = None, **kw) -> None:
        kw.setdefault("xlabel", "x [m]")
        kw.setdefault("ylabel", "y [m]")
        super().__init__(parent, equal_aspect=True, **kw)

    def clear_all(self) -> None:
        self.clear()
        if self.legend is not None:
            self.legend.clear()
            self.legend.setVisible(False)

    def add_outline(self, points: np.ndarray, color=None, *, width: float = LINE_W, name: str | None = None,
                    fill: bool = False, style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> None:
        """Closed polygon outline (the closing segment is added here)."""
        p = np.asarray(points, dtype=float)
        if len(p) < 2:
            return
        p = np.vstack([p, p[:1]])
        col = qcolor(color) if color is not None else qcolor(tokens()["text"])
        kw = {"fillLevel": None, "fillOutline": True, "brush": qcolor(col.name(), 30)} if fill else {}
        item = pg.PlotDataItem(p[:, 0], p[:, 1], pen=make_pen(col, width, style), name=name, **kw)
        self.addItem(item)
        if name and self.legend is not None:
            self.legend.setVisible(len(self.legend.items) >= 2)

    def add_arrows(self, x, y, dx, dy, color, *, width: float = 1.0) -> None:
        """Line segments from (x, y) to (x+dx, y+dy): a 'hedgehog' vector plot."""
        x, y, dx, dy = (np.asarray(a, dtype=float) for a in (x, y, dx, dy))
        if x.size == 0:
            return
        xs = np.empty(2 * x.size)
        ys = np.empty(2 * x.size)
        xs[0::2], xs[1::2] = x, x + dx
        ys[0::2], ys[1::2] = y, y + dy
        self.addItem(pg.PlotCurveItem(xs, ys, connect="pairs", pen=make_pen(color, width)))
        self.addItem(pg.ScatterPlotItem(x + dx, y + dy, size=2.5, pen=None, brush=qcolor(color)))

    def add_value_outline(self, x, y, values, lut: np.ndarray, vmin: float, vmax: float,
                          *, width: float = 5.0, bins: int = 48) -> None:
        """Outline whose segments are coloured by ``values`` through a (256, 3) colour LUT.

        pyqtgraph has no per-segment pen, so segments are grouped into ``bins`` colour
        bins and each bin is one ``connect="pairs"`` curve: 48 items instead of
        thousands keeps panning smooth on long outlines.
        """
        x, y, v = (np.asarray(a, dtype=float) for a in (x, y, values))
        if x.size < 2:
            return
        xs = np.append(x, x[0])
        ys = np.append(y, y[0])
        seg_v = 0.5 * (v + np.roll(v, -1))
        span = (vmax - vmin) or 1.0
        idx = np.clip(((seg_v - vmin) / span * (bins - 1)).round(), 0, bins - 1).astype(int)
        lut = np.asarray(lut)
        for b in np.unique(idx):
            sel = np.nonzero(idx == b)[0]
            px = np.empty(2 * sel.size)
            py = np.empty(2 * sel.size)
            px[0::2], px[1::2] = xs[sel], xs[sel + 1]
            py[0::2], py[1::2] = ys[sel], ys[sel + 1]
            rgb = lut[round(b / max(bins - 1, 1) * (len(lut) - 1))]
            pen = make_pen(QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])), width)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            self.addItem(pg.PlotCurveItem(px, py, connect="pairs", pen=pen))


class BarChart(ThemedPlot):
    """Vertical bars with category labels, standard-error whiskers and the value above each bar.

    The value labels and whiskers are drawn in the theme's primary text colour
    and recoloured on a theme change: labels made once in the light theme's
    grey stayed dark grey on the dark theme's graphite, where they could hardly
    be read.
    """

    def __init__(self, parent: QWidget | None = None, **kw) -> None:
        self._items: list = []
        self._value_labels: list = []
        self._whiskers: list = []
        super().__init__(parent, **kw)
        self.getPlotItem().showGrid(x=False, y=True, alpha=GRID_ALPHA)

    def apply_theme(self) -> None:
        super().apply_theme()
        if self.getPlotItem() is None:
            return
        self.getPlotItem().showGrid(x=False, y=True, alpha=GRID_ALPHA)
        tk = tokens()
        for ti in getattr(self, "_value_labels", []):
            ti.setColor(qcolor(tk["text"]))
        for eb in getattr(self, "_whiskers", []):
            eb.setData(pen=make_pen(tk["text"], 1.0))

    def value_label_colors(self) -> list[str]:
        """Colours (#rrggbb) of the value labels above the bars (for tests)."""
        return [ti.textItem.defaultTextColor().name() for ti in self._value_labels]

    def set_bars(self, labels: Sequence[str], values: Sequence[float], colors: Sequence | None = None,
                 *, errors: Sequence[float] | None = None, fmt: Callable[[float], str] | None = None) -> None:
        for it in self._items:
            self.removeItem(it)
        self._items.clear()
        self._value_labels.clear()
        self._whiskers.clear()
        tk = tokens()
        vals = np.asarray(values, dtype=float)
        n = len(labels)
        if n == 0:
            return
        xs = np.arange(n, dtype=float)
        brushes = [qcolor(colors[i]) if colors is not None else series_color(i, tk) for i in range(n)]
        safe = np.where(np.isfinite(vals), vals, 0.0)
        bars = pg.BarGraphItem(x=xs, height=safe, width=self.bar_width(n), brushes=brushes, pens=[None] * n)
        self.addItem(bars)
        self._items.append(bars)
        if errors is not None:
            err = np.nan_to_num(np.asarray(errors, dtype=float))
            eb = pg.ErrorBarItem(x=xs, y=safe, height=2 * err, beam=0.12, pen=make_pen(tk["text"], 1.0))
            self.addItem(eb)
            self._items.append(eb)
            self._whiskers.append(eb)
        fmt = fmt or (lambda v: f"{v:.4g}")
        span = float(np.nanmax(np.abs(safe))) if np.any(safe) else 1.0
        font = _font(AXIS_PT)
        for x, v in zip(xs, vals):
            if not math.isfinite(v):
                continue
            ti = pg.TextItem(fmt(v), color=qcolor(tk["text"]), anchor=(0.5, 1.0 if v >= 0 else 0.0))
            ti.setFont(font)
            ti.setPos(x, v + (0.03 * span if v >= 0 else -0.03 * span))
            self.addItem(ti)
            self._items.append(ti)
            self._value_labels.append(ti)
        ax = self.getPlotItem().getAxis("bottom")
        ax.setTicks([[(float(x), str(lab)) for x, lab in zip(xs, labels)]])
        lo = min(0.0, float(np.nanmin(safe)))
        hi = max(0.0, float(np.nanmax(safe)))
        pad = 0.18 * ((hi - lo) or 1.0)
        # the bars span the plot: [-0.6, n - 0.4] puts the first and last bar 0.6 slot from the
        # axes. Padding the range to five slots (so two bars were not half a screen wide each)
        # left a blank half beside two bars that read as a broken chart; narrower bars do that job
        self.setXRange(-0.6, n - 0.4, padding=0)
        self.setYRange(lo - (pad if lo < 0 else 0), hi + pad, padding=0)

    @staticmethod
    def bar_width(n: int) -> float:
        """Bar width in slots: half a slot, narrower for one or two bars so a pair does not fill the plot."""
        return 0.5 if n >= 3 else 0.4

    def bar_spans_px(self) -> list[tuple[float, float]]:
        """(left, right) of each bar in this widget's pixels, after layout (for tests and layout checks)."""
        from PySide6.QtCore import QPointF
        bars = next((it for it in self._items if isinstance(it, pg.BarGraphItem)), None)
        if bars is None:
            return []
        xs = np.asarray(bars.opts.get("x"), dtype=float)
        w = float(bars.opts.get("width", 0.5))
        vb = self.getPlotItem().getViewBox()
        out = []
        for x in xs:
            a = self.mapFromScene(vb.mapViewToScene(QPointF(x - w / 2, 0.0)))
            b = self.mapFromScene(vb.mapViewToScene(QPointF(x + w / 2, 0.0)))
            out.append((float(a.x()), float(b.x())))
        return out

    def plot_span_px(self) -> tuple[float, float]:
        """(left, right) of the plotting area (the view box) in this widget's pixels."""
        vb = self.getPlotItem().getViewBox()
        r = vb.mapRectToScene(vb.boundingRect())
        a, b = self.mapFromScene(r.topLeft()), self.mapFromScene(r.topRight())
        return float(a.x()), float(b.x())


class PolarGrid(QWidget):
    """The four standard airfoil polar views in a 2 x 2 grid, overlayable.

    Cl-alpha (lift curve), Cl-Cd (drag polar, Cl on the vertical axis as in XFOIL
    and most published data), Cm-alpha and L/D-alpha.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(1)
        cl, cd, cm = "C<sub>l</sub>", "C<sub>d</sub>", "C<sub>m</sub>"
        self.cl_alpha = LinePlot(self, xlabel="α (°)", ylabel=cl)
        self.drag_polar = LinePlot(self, xlabel=cd, ylabel=cl)
        self.cm_alpha = LinePlot(self, xlabel="α (°)", ylabel=f"{cm} (c/4)")
        self.ld_alpha = LinePlot(self, xlabel="α (°)", ylabel="L/D")
        for i, w in enumerate((self.cl_alpha, self.drag_polar, self.cm_alpha, self.ld_alpha)):
            grid.addWidget(w, i // 2, i % 2)

    def plots(self) -> tuple[LinePlot, ...]:
        return (self.cl_alpha, self.drag_polar, self.cm_alpha, self.ld_alpha)

    def set_polars(self, polars: Sequence[tuple[str, dict, int | str | QColor]]) -> None:
        """``polars`` = [(name, PolarResult.table(), colour), ...]; replaces all series."""
        for p in self.plots():
            p.clear_series()
        for name, tab, color in polars:
            a = np.asarray(tab.get("alpha", []), dtype=float)
            if a.size == 0:
                continue
            cl = np.asarray(tab["cl"], dtype=float)
            cd = np.asarray(tab["cd"], dtype=float)
            cm = np.asarray(tab.get("cm", np.full_like(a, np.nan)), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ld = np.where(cd > 0, cl / cd, np.nan)
            self.cl_alpha.set_series(name, a, cl, color=color, symbol="o")
            self.drag_polar.set_series(name, cd, cl, color=color, symbol="o")
            self.cm_alpha.set_series(name, a, cm, color=color, symbol="o")
            self.ld_alpha.set_series(name, a, ld, color=color, symbol="o")


class PlotPanel(QWidget):
    """A plot under a one-line semibold title, with an optional muted caption below the title."""

    def __init__(self, title: str, caption: str, plot: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.title = QLabel(title)
        self.title.setProperty("role", "h3")
        self.title.setVisible(bool(title))
        self.caption = QLabel(caption)
        self.caption.setProperty("role", "muted")
        self.caption.setWordWrap(True)
        self.caption.setVisible(bool(caption))
        lay.addWidget(self.title)
        lay.addWidget(self.caption)
        lay.addSpacing(2)
        lay.addWidget(plot, 1)
        self.plot = plot

    def set_caption(self, text: str) -> None:
        self.caption.setText(text)
        self.caption.setVisible(bool(text))
