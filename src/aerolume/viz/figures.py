"""Print-quality static figures (matplotlib, no pyplot).

These builders are shared by the PDF/HTML report and by the desktop app's
"export PNG" actions, so a plot looks identical wherever it ends up. Each one
returns a :class:`matplotlib.figure.Figure` that already carries an Agg canvas:
nothing here touches ``pyplot``, so no GUI backend is ever imported and the
functions are safe on worker threads and headless machines.

Style. One restrained, print-first look: white background, thin spines on the
left and bottom only, light grid, a short palette of well-separated colours and
SI units in every dimensional axis label. ``style="dark"`` swaps the tokens for
a dark background so the GUI can export figures that match its dark theme.
Styles are applied with :func:`matplotlib.rc_context` while a figure is *built*
(fonts, colours and tick parameters are captured by the artists at creation),
and :func:`save_figure` / :func:`figure_to_rgba` re-enter the same context
while *drawing*, because the layout engine reads a few rcParams at draw time.
Prefer those helpers over ``fig.savefig`` for pixel-exact results.

Conventions follow DESIGN.md: wind-frame geometry in metres, flow along +x,
Cp axes inverted (suction up) as aerodynamicists expect, forces per unit span.
"""
from __future__ import annotations

import functools
import io
import itertools
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.font_manager import findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

from ..model import Body, Design, FieldData, PolarPoint, PolarResult, Stat, SurfaceData

log = logging.getLogger(__name__)

__all__ = [
    "COEFF_LABELS_3D",
    "FIELD_LABELS",
    "FONT_FAMILY",
    "STYLES",
    "FigureStyle",
    "SurfaceCut",
    "fig_compare_bars",
    "fig_cp",
    "fig_diagnostics",
    "fig_field",
    "fig_geometry",
    "fig_grid_convergence",
    "fig_history",
    "fig_history3d",
    "fig_polar",
    "fig_sensitivity",
    "fig_smoke",
    "fig_span_loading",
    "fig_views3d",
    "fig_wing_planform",
    "fig_wing_polar",
    "figure_to_png",
    "figure_to_rgba",
    "get_style",
    "save_figure",
    "style_context",
]


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------

def _pick_font() -> str:
    """First installed family of a short list of clean sans-serifs.

    The report must render on any machine, so we never ship or require a font:
    Segoe UI (Windows), Helvetica (macOS), Arial / Liberation Sans (most Linux
    desktops) all look professional in print; DejaVu Sans is matplotlib's bundled
    last resort. Glyphs a font lacks (e.g. Greek in some fonts) fall back to the
    next family in ``font.sans-serif`` automatically.
    """
    for name in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans"):
        try:
            findfont(name, fallback_to_default=False)
        except Exception:  # noqa: BLE001 - font_manager raises ValueError; be defensive
            log.debug("font %s not installed", name)
        else:
            return name
    return "DejaVu Sans"


FONT_FAMILY: str = _pick_font()
FONT_STACK: list[str] = [FONT_FAMILY, "DejaVu Sans"] if FONT_FAMILY != "DejaVu Sans" else ["DejaVu Sans"]


@dataclass(frozen=True)
class FigureStyle:
    """Colour tokens of one figure style. Keys mirror ``app.theme.tokens()`` where they overlap."""
    name: str
    bg: str             # figure + axes background
    text: str           # titles, labels
    muted: str          # tick labels, annotations
    grid: str
    spine: str
    accent: str         # the one accent (docs/ui_guidelines.md): highlights, reference marks
    series: tuple[str, ...]
    body_fill: str
    body_edge: str
    ground: str
    good: str
    warn: str
    bad: str
    panel: str          # subtle fill for bands and boxes
    stream: str         # streamline colour over colour maps


STYLES: dict[str, FigureStyle] = {
    "print": FigureStyle(
        name="print", bg="#ffffff", text="#1d2733", muted="#5b6776", grid="#e5e8ed",
        spine="#8a94a3", accent="#2463c4",
        # docs/ui_guidelines.md section 2: the application's plot series, so a figure in a
        # report shows "the orange one" in the same colour as the Compare page
        series=("#2463c4", "#d9730d", "#2f855a", "#b8322b", "#6b46c1", "#5f6b7a"),
        body_fill="#d5dbe3", body_edge="#2b3440", ground="#6b7a8f",
        good="#1e8a4c", warn="#c27c0e", bad="#c23b3b", panel="#f3f5f8", stream="#ffffff",
    ),
    "dark": FigureStyle(
        name="dark", bg="#16191d", text="#e6e9ed", muted="#9aa4b1", grid="#2a3038",
        spine="#5b6470", accent="#4a8cf0",
        series=("#5b93ee", "#e98a2e", "#57b37c", "#e0675f", "#9f7aea", "#9aa4b1"),
        body_fill="#3a414b", body_edge="#c9d1db", ground="#8b95a3",
        good="#4cc38a", warn="#f0b429", bad="#ef6461", panel="#1e2328", stream="#ffffff",
    ),
}


def get_style(style: str | FigureStyle = "print") -> FigureStyle:
    """Resolve a style name ("print" | "dark") or pass a :class:`FigureStyle` through."""
    if isinstance(style, FigureStyle):
        return style
    try:
        return STYLES[style]
    except KeyError:
        raise ValueError(f"unknown figure style {style!r}; use one of {', '.join(STYLES)}") from None


def _rc(st: FigureStyle) -> dict[str, Any]:
    """rcParams of a style. Sizes are tuned for figures printed at ~1:1 on A4 (8-9 pt text)."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 7.0,
        "axes.titlecolor": st.text,
        "axes.labelsize": 8.5,
        "axes.labelcolor": st.text,
        "axes.edgecolor": st.spine,
        "axes.facecolor": st.bg,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=list(st.series)),
        "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-4, 5),
        "grid.color": st.grid,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": st.spine,
        "ytick.color": st.spine,
        "xtick.labelcolor": st.muted,
        "ytick.labelcolor": st.muted,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.6,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.labelcolor": st.text,
        "text.color": st.text,
        "figure.facecolor": st.bg,
        "figure.edgecolor": st.bg,
        "figure.dpi": 100,
        "figure.constrained_layout.h_pad": 0.04,
        "figure.constrained_layout.w_pad": 0.04,
        "savefig.facecolor": st.bg,
        "savefig.edgecolor": st.bg,
        "savefig.dpi": 200,
        "image.interpolation": "antialiased",
        "hatch.color": st.spine,
        "hatch.linewidth": 0.4,
        "pdf.fonttype": 42,     # embed TrueType: crisp, selectable text in PDFs
        "svg.fonttype": "none",
    }


def style_context(style: str | FigureStyle = "print"):
    """Context manager applying a figure style's rcParams (for custom plots that should match)."""
    return mpl.rc_context(_rc(get_style(style)))


def _new_figure(st: FigureStyle, figsize: tuple[float, float], layout: str | None = "constrained") -> Figure:
    fig = Figure(figsize=figsize, layout=layout, facecolor=st.bg)
    FigureCanvasAgg(fig)
    fig._aerolume_style = st  # type: ignore[attr-defined]  # read back by save_figure
    return fig


def _builder(default_size: tuple[float, float] | None):
    """Decorator: resolve ``style``/``figsize`` and build the figure inside the style's rc context.

    ``default_size=None`` marks builders that size themselves from their content (equal-aspect
    geometry and field plots), so they receive ``figsize=None`` when the caller gives none.
    """
    def deco(fn: Callable[..., Figure]) -> Callable[..., Figure]:
        @functools.wraps(fn)
        def wrapper(*args: Any, style: str | FigureStyle = "print",
                    figsize: tuple[float, float] | None = None, **kw: Any) -> Figure:
            st = get_style(style)
            size = tuple(figsize) if figsize is not None else default_size
            with mpl.rc_context(_rc(st)):
                return fn(*args, st=st, figsize=size, **kw)
        return wrapper
    return deco


def _fit_height(width_in: float, data_w: float, data_h: float, *, side_in: float = 0.75,
                extra_in: float = 0.75, lo: float = 2.0, hi: float = 5.0) -> float:
    """Figure height that lets an equal-aspect axes of ``data_w x data_h`` fill ``width_in``.

    ``side_in`` is the horizontal space taken by tick labels / colorbar and ``extra_in`` the
    vertical space of title and x label. Clamped so slender airfoils still get a usable plot.
    """
    avail = max(width_in - side_in, 1.0)
    return float(min(max(avail * data_h / max(data_w, 1e-12) + extra_in, lo), hi))


def save_figure(fig: Figure, path: str | Path, *, dpi: float = 200, **kw: Any) -> Path:
    """Save a figure (PNG, PDF, SVG by suffix) inside its own style context."""
    path = Path(path)
    st = getattr(fig, "_aerolume_style", STYLES["print"])
    with mpl.rc_context(_rc(st)):
        fig.savefig(path, dpi=dpi, facecolor=st.bg, **kw)
    return path


def figure_to_png(fig: Figure, *, dpi: float = 150) -> bytes:
    """PNG bytes of a figure (for HTML embedding and clipboard export)."""
    buf = io.BytesIO()
    st = getattr(fig, "_aerolume_style", STYLES["print"])
    with mpl.rc_context(_rc(st)):
        fig.savefig(buf, format="png", dpi=dpi, facecolor=st.bg)
    return buf.getvalue()


def figure_to_rgba(fig: Figure, *, dpi: float = 200) -> np.ndarray:
    """Rasterise a figure to an (H, W, 4) uint8 array, row 0 = top."""
    st = getattr(fig, "_aerolume_style", STYLES["print"])
    with mpl.rc_context(_rc(st)):
        fig.set_dpi(dpi)
        canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

COEFF_LABELS: dict[str, tuple[str, str]] = {
    "cd": ("Cd", "Drag coefficient"),
    "cl": ("Cl", "Lift coefficient"),
    "cm": ("Cm", "Moment coefficient"),
    # heat transfer histories (DESIGN.md section 10); fig_history takes labels= for others
    "nu": ("Nu", "Nusselt number"),
    "heat_rate": ("Q", "Heat rate"),
}

#: 3D coefficients (DESIGN.md 7.2, wind axes): capital C_D / C_S / C_L for forces on a
#: reference area, as in every finite-wing and vehicle text; "Croll" because "Cl" is the lift.
COEFF_LABELS_3D: dict[str, tuple[str, str]] = {
    "cd": ("CD", "Drag"),
    "cs": ("CS", "Side force"),
    "cl": ("CL", "Lift"),
    "c_roll": ("Croll", "Rolling moment"),
    "cm": ("Cm", "Pitching moment"),
    "cn": ("Cn", "Yawing moment"),
}

# field name -> (label, unit, diverging). Used when aerolume.viz does not provide FIELD_INFO,
# and always for the unit string (viz stores a unit *kind*, figures need the printed unit).
FIELD_LABELS: dict[str, tuple[str, str, bool]] = {
    "speed": ("Velocity magnitude |U|", "m/s", False),
    "u": ("Streamwise velocity u", "m/s", True),
    "v": ("Vertical velocity v", "m/s", True),
    "p": ("Static pressure p (gauge)", "Pa", True),
    "cp": ("Pressure coefficient Cp", "", True),
    "vorticity": ("Vorticity ωz", "1/s", True),
    "dye": ("Smoke concentration", "", False),
    "nut": ("Eddy viscosity νt", "m²/s", False),
    "mask": ("Solid fraction", "", False),
    "divergence": ("Velocity divergence", "1/s", True),
    # 3D plane cuts: the streamwise component under its own key, since "u" / "v" there carry
    # the in-plane velocity that the streamlines follow
    "ux": ("Streamwise velocity u", "m/s", False),
    "q": ("Q-criterion", "1/s²", True),
    # heat transfer and extra smoke channels (DESIGN.md section 10)
    "temperature": ("Temperature T", "°C", False),
    "theta": ("Temperature θ = (T − T∞)/ΔT", "", False),
    "dye1": ("Smoke concentration (channel 2)", "", False),
    "dye2": ("Smoke concentration (channel 3)", "", False),
    "dye3": ("Smoke concentration (channel 4)", "", False),
}

# Fallback colormaps (matplotlib names) when aerolume.viz is unavailable or lacks a field.
_FALLBACK_CMAP = {"speed": "viridis", "u": "viridis", "v": "RdBu_r", "p": "RdBu_r", "cp": "RdBu_r",
                  "vorticity": "RdBu_r", "dye": "bone_r", "nut": "magma", "mask": "Greys",
                  "divergence": "RdBu_r", "ux": "viridis", "q": "RdBu_r", "temperature": "inferno",
                  "theta": "inferno", "dye1": "bone_r", "dye2": "bone_r", "dye3": "bone_r"}


def _viz_attr(name: str) -> Any:
    """Optional attribute of :mod:`aerolume.viz` (owned by another module; may not exist yet)."""
    try:
        from .. import viz
        return getattr(viz, name, None)
    except Exception:  # noqa: BLE001 - viz may import torch or be half-written; figures must not care
        return None


def _resolve_cmap(field_name: str, cmap: str | Colormap | None) -> Colormap:
    """Colormap by explicit name/object, else the app's per-field default, else a matplotlib default.

    Aerolume's own colormaps (``viz.COLORMAPS``, (256, 3) uint8 tables) win over matplotlib's so
    exported figures match the live view in the app.
    """
    if isinstance(cmap, Colormap):
        return cmap
    if cmap is None:
        defaults = _viz_attr("DEFAULT_CMAP") or {}
        cmap = defaults.get(field_name) or _FALLBACK_CMAP.get(field_name, "viridis")
    tables = _viz_attr("COLORMAPS") or {}
    if cmap in tables:
        arr = np.asarray(tables[cmap], dtype=float)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return ListedColormap(arr[:, :3], name=str(cmap))
    try:
        return colormaps[str(cmap)]
    except KeyError:
        return colormaps[_FALLBACK_CMAP.get(field_name, "viridis")]


@dataclass
class _Outline:
    name: str
    role: str
    points: np.ndarray      # (N, 2) metres


def _outlines(obj: Any) -> list[_Outline]:
    """Normalise a Design, Body list, SurfaceData list or raw (N, 2) arrays into outlines."""
    if obj is None:
        return []
    if isinstance(obj, Design):
        items: Iterable[Any] = obj.bodies
    elif isinstance(obj, np.ndarray) and obj.ndim == 2:
        items = [obj]
    else:
        items = list(obj)
    out = []
    for i, b in enumerate(items):
        if isinstance(b, Body) or (hasattr(b, "points") and hasattr(b, "name")):
            out.append(_Outline(str(b.name), str(getattr(b, "role", "body")), np.asarray(b.points, float)))
        elif isinstance(b, SurfaceData) or (hasattr(b, "x") and hasattr(b, "body")):
            out.append(_Outline(str(b.body), "body", np.column_stack([b.x, b.y]).astype(float)))
        else:
            out.append(_Outline(f"body {i + 1}", "body", np.asarray(b, float).reshape(-1, 2)))
    return out


def _rotate(points: np.ndarray, alpha_deg: float, about: Sequence[float]) -> np.ndarray:
    """Apply the solver's attitude convention: rotate by -alpha about ``about`` (nose up for +alpha)."""
    if not alpha_deg:
        return points
    a = -math.radians(alpha_deg)
    c, s = math.cos(a), math.sin(a)
    p = points - np.asarray(about, float)
    return np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]]) + np.asarray(about, float)


def _draw_bodies(ax: Any, outlines: Sequence[_Outline], st: FigureStyle, *, lw: float = 0.8,
                 zorder: float = 3, fill: str | None = None) -> None:
    for o in outlines:
        face = st.spine if o.role == "wheel" else (fill or st.body_fill)
        ax.add_patch(MplPolygon(o.points, closed=True, facecolor=face, edgecolor=st.body_edge,
                                linewidth=lw, zorder=zorder, joinstyle="round"))


def _draw_ground(ax: Any, st: FigureStyle, x0: float, x1: float, depth: float) -> None:
    """Road line at y = 0 with a light hatched band beneath (vehicles)."""
    ax.add_patch(Rectangle((x0, -depth), x1 - x0, depth, facecolor=st.panel, edgecolor=st.grid,
                           hatch="//////", zorder=1, linewidth=0))
    ax.plot([x0, x1], [0, 0], color=st.ground, lw=1.0, zorder=2, solid_capstyle="butt")


def _no_data(ax: Any, st: FigureStyle, text: str = "No data") -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=st.muted)
    ax.set_xticks([])
    ax.set_yticks([])


def _stat_of(stats: Any, key: str) -> Stat | None:
    if stats is None:
        return None
    s = stats.get(key) if isinstance(stats, Mapping) else getattr(stats, key, None)
    return s if isinstance(s, Stat) or hasattr(s, "mean") else None


def _fmt(x: float, digits: int = 4) -> str:
    try:
        from ..core.units import sig
        return sig(float(x), digits)
    except Exception:  # noqa: BLE001
        return f"{x:.{digits}g}"


def _fmt_len(x: float) -> str:
    """Compact length for annotations: '0.3 m', '4.5 m', '1.42 m' (no padded zeros)."""
    return f"{x:.4g} m"


def _fmt_pm(mean: float, err: float) -> str:
    """'0.03124 ± 0.00041': mean rounded to the precision of a two-significant-figure stderr."""
    if not (math.isfinite(mean)):
        return "—"
    if abs(mean) < 1e-6:            # round-off of a coefficient that is zero by symmetry (report.fmt)
        return "≈ 0"
    if not (math.isfinite(err) and err > 0):
        return _fmt(mean)
    e = math.floor(math.log10(err))
    dec = max(0, -e + (1 if int(err / 10 ** e) < 3 else 0))   # 1 sig. fig., 2 if it starts with 1-2
    cap = max(0, 4 - math.floor(math.log10(abs(mean))))         # at most 5 significant figures
    if dec > cap:
        return f"{mean:.{cap}f} ± {err:.1e}"
    return f"{mean:.{dec}f} ± {err:.{dec}f}"


def _robust_limits(core: np.ndarray, whole: np.ndarray, pad: float = 0.5) -> tuple[float, float]:
    """y-limits that frame the statistically interesting part of a signal.

    Impulsive starts produce force spikes 10-100x the converged value; plain autoscaling
    would flatten the part a reader cares about. We frame the averaging window (``core``)
    generously and only include the transient when it is of comparable size.
    """
    core = core[np.isfinite(core)]
    whole = whole[np.isfinite(whole)]
    if core.size == 0:
        core = whole
    if core.size == 0:
        return -1.0, 1.0
    lo, hi = float(core.min()), float(core.max())
    span = hi - lo
    floor = max(abs(float(np.median(core))) * 0.02, 1e-6)
    span = max(span, floor)
    lo, hi = lo - pad * span, hi + pad * span
    if whole.size:
        wlo, whi = float(np.percentile(whole, 1)), float(np.percentile(whole, 99))
        if wlo > lo - 3 * span:
            lo = min(lo, wlo - 0.1 * span)
        if whi < hi + 3 * span:
            hi = max(hi, whi + 0.1 * span)
    return lo, hi


def _decimate(t: np.ndarray, *ys: np.ndarray, max_points: int = 6000) -> tuple[np.ndarray, ...]:
    """Min/max-preserving decimation so long histories stay light without hiding peaks."""
    n = t.size
    if n <= max_points:
        return (t, *ys)
    bins = max_points // 2
    edges = np.linspace(0, n, bins + 1).astype(int)
    idx = []
    ref = ys[0] if ys else t
    for a, b in itertools.pairwise(edges):
        if b <= a:
            continue
        seg = ref[a:b]
        i0, i1 = a + int(np.nanargmin(seg)), a + int(np.nanargmax(seg))
        idx.extend(sorted((i0, i1)))
    ii = np.unique(np.asarray(idx))
    return (t[ii], *(y[ii] for y in ys))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

@_builder(None)
def fig_geometry(design: Design | Sequence[Any], *, st: FigureStyle, figsize: tuple[float, float] | None,
                 alpha_deg: float = 0.0, title: str | None = None, show_reference: bool = True,
                 ground: bool | None = None, flow_arrow: bool = True, width: float = 6.7) -> Figure:
    """Outline drawing of a design (or a list of bodies) with reference dimensions.

    With a :class:`Design`, ``alpha_deg`` rotates it exactly as the solver will (about
    ``moment_ref`` by -alpha), the reference length is dimensioned underneath, the moment
    reference point is marked and vehicles get the road plane. The free-stream arrow makes
    the flow direction explicit, which non-specialist readers otherwise have to guess.
    Without ``figsize`` the height is fitted to the drawing at ``width`` inches, so a slender
    airfoil does not float in a tall empty frame.
    """
    outs = _outlines(design)
    is_design = isinstance(design, Design)
    about = design.moment_ref if is_design else (0.0, 0.0)
    for o in outs:
        o.points = _rotate(o.points, alpha_deg, about)
    if not outs:
        fig = _new_figure(st, figsize or (width, 2.4))
        _no_data(fig.add_subplot(), st, "No geometry")
        return fig
    pts = np.vstack([o.points for o in outs])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = max(xmax - xmin, 1e-9)
    h = max(ymax - ymin, 1e-9)
    if ground is None:
        ground = is_design and design.kind == "vehicle"
    unit = max(min(h, 0.25 * w), 0.12 * w)           # annotation spacing scale
    fx0, fx1 = xmin - 0.06 * w, xmax + 0.06 * w      # frame, grown by every annotation below
    fy0, fy1 = ymin - 0.25 * unit, ymax + 0.25 * unit
    gdepth = 0.035 * w
    if ground:
        fy0 = min(fy0, -gdepth)
    ops: list[Callable[[Any], None]] = []
    if show_reference and is_design:
        mx, my = design.moment_ref
        lref = design.length_ref
        yd = (-gdepth if ground else ymin) - 0.32 * unit
        sym = "c" if design.kind == "airfoil" else "L"

        def _dims(ax: Any) -> None:
            ax.plot([mx], [my], marker="+", color=st.accent, ms=8, mew=1.2, zorder=6)
            ax.plot([mx], [my], marker="o", mfc="none", mec=st.accent, ms=6, mew=0.9, zorder=6)
            ax.annotate("", xy=(xmin, yd), xytext=(xmin + lref, yd),
                        arrowprops={"arrowstyle": "<->", "color": st.muted, "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
            for xe in (xmin, xmin + lref):
                ax.plot([xe, xe], [yd - 0.12 * unit, yd + 0.12 * unit], color=st.muted, lw=0.6)
            ax.text(xmin + lref / 2, yd - 0.12 * unit, f"{sym} = {_fmt_len(lref)}", ha="center", va="top",
                    color=st.muted, fontsize=7.5)
        ops.append(_dims)
        fy0 = min(fy0, yd - 0.6 * unit)
        if design.kind == "vehicle":
            xh = xmax + 0.05 * w
            base = 0.0 if ground else ymin

            def _height(ax: Any) -> None:
                ax.annotate("", xy=(xh, base), xytext=(xh, base + design.force_ref),
                            arrowprops={"arrowstyle": "<->", "color": st.muted, "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
                ax.text(xh + 0.015 * w, base + design.force_ref / 2, f"h = {_fmt_len(design.force_ref)}",
                        ha="left", va="center", color=st.muted, fontsize=7.5)
            ops.append(_height)
            fx1 = max(fx1, xh + 0.15 * w)
    if flow_arrow:
        ya = ymax + 0.35 * unit if not ground else ymin + 0.6 * h
        xa1 = xmin - 0.04 * w
        xa0 = xa1 - max(0.10 * w, 0.4 * unit)

        def _arrow(ax: Any) -> None:
            ax.annotate("", xy=(xa1, ya), xytext=(xa0, ya),
                        arrowprops={"arrowstyle": "-|>", "color": st.accent, "lw": 1.2, "mutation_scale": 9})
            ax.text(0.5 * (xa0 + xa1), ya + 0.08 * unit, "U∞", ha="center", va="bottom",
                    color=st.accent, fontsize=8)
        ops.append(_arrow)
        fx0 = min(fx0, xa0 - 0.03 * w)
        fy1 = max(fy1, ya + 0.45 * unit)
    if figsize is None:
        figsize = (width, _fit_height(width, fx1 - fx0, fy1 - fy0, extra_in=0.8, lo=1.8, hi=4.6))
    fig = _new_figure(st, figsize)
    ax = fig.add_subplot()
    if ground:
        _draw_ground(ax, st, fx0, fx1, gdepth)
    _draw_bodies(ax, outs, st, lw=0.9)
    for op in ops:
        op(ax)
    ax.set_xlim(fx0, fx1)
    ax.set_ylim(fy0, fy1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, lw=0.4)
    label = title if title is not None else (design.name if is_design else "")
    if is_design and alpha_deg and title is None:
        label += f"  ·  α = {alpha_deg:g}°"
    if label:
        ax.set_title(label)
    return fig


# ---------------------------------------------------------------------------
# surface distributions
# ---------------------------------------------------------------------------

@dataclass
class _Branch:
    xc: np.ndarray          # chordwise coordinate 0 (front) .. 1 (rear)
    cp: np.ndarray
    cf: np.ndarray | None
    side: str               # "upper" | "lower"


@dataclass
class _CpSeries:
    label: str
    branches: list[_Branch]
    outline: np.ndarray | None       # (N, 2) in chord-normalised, de-rotated coordinates
    cp_inviscid: list[_Branch]


def _split_loop(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a closed surface loop at its front-most and rear-most points.

    Returns (index arrays of the upper and lower branches, chord coordinate t in [0, 1],
    normal coordinate) using the front->rear line as the chord, so a rotated (wind-frame)
    airfoil is still plotted against true x/c rather than a foreshortened x.
    """
    n = x.size
    i_le, i_te = int(np.argmin(x)), int(np.argmax(x))
    p_le = np.array([x[i_le], y[i_le]])
    d = np.array([x[i_te], y[i_te]]) - p_le
    c2 = float(d @ d) or 1.0
    rel = np.column_stack([x, y]) - p_le
    t = rel @ d / c2
    nrm = (rel[:, 0] * -d[1] + rel[:, 1] * d[0]) / c2   # left of chord = "up" in the chord frame
    a = [(i_te + k) % n for k in range(((i_le - i_te) % n) + 1)]   # rear -> front
    b = [(i_le + k) % n for k in range(((i_te - i_le) % n) + 1)]   # front -> rear
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if nrm[a_arr].mean() >= nrm[b_arr].mean():
        return a_arr, b_arr, t, nrm
    return b_arr, a_arr, t, nrm


@dataclass
class SurfaceCut:
    """Surface values along a closed plane section of a 3D body (e.g. the centre line).

    ``x``, ``y`` are the in-plane coordinates of the loop [m] (for a centre-line cut of a
    vehicle: x and the height z); ``cf`` is already the *streamwise* skin friction (the x
    component of the wall shear over q, negative in reversed flow), so :func:`fig_cp` plots
    it as given instead of signing it from an outline direction as for 2D ``SurfaceData``.
    """
    body: str
    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray
    cf: np.ndarray | None = None


def _cp_series(item: Any, label: str) -> _CpSeries:
    if isinstance(item, SurfaceCut):
        x, y, cp = np.asarray(item.x, float), np.asarray(item.y, float), np.asarray(item.cp, float)
        up, lo, t, nrm = _split_loop(x, y)
        cf = np.asarray(item.cf, float) if item.cf is not None and np.size(item.cf) == x.size else None
        br = [_Branch(t[i], cp[i], None if cf is None else cf[i], s) for i, s in ((up, "upper"), (lo, "lower"))]
        return _CpSeries(label, br, np.column_stack([t, nrm]), [])
    if isinstance(item, PolarPoint) or (hasattr(item, "cp_inviscid") and hasattr(item, "alpha")):
        x = np.asarray(item.x, float)
        y = np.asarray(item.y, float) if item.y is not None else np.zeros_like(x)
        cp = np.asarray(item.cp, float)
        up, lo, t, nrm = _split_loop(x, y)
        cf = np.asarray(item.cf, float) if item.cf is not None and np.size(item.cf) == x.size else None
        br = [_Branch(t[i], cp[i], None if cf is None else cf[i], s) for i, s in ((up, "upper"), (lo, "lower"))]
        inv = []
        if item.cp_inviscid is not None and np.size(item.cp_inviscid) == x.size:
            ci = np.asarray(item.cp_inviscid, float)
            inv = [_Branch(t[i], ci[i], None, s) for i, s in ((up, "upper"), (lo, "lower"))]
        return _CpSeries(label, br, np.column_stack([t, nrm]), inv)
    # SurfaceData: cf is signed along increasing s; for a CCW loop the tangent is (-ny, nx), so
    # multiplying by sign(t_x) gives Cf in the streamwise sense (negative = reversed flow).
    x, y, cp = np.asarray(item.x, float), np.asarray(item.y, float), np.asarray(item.cp, float)
    up, lo, t, nrm = _split_loop(x, y)
    cf = None
    if getattr(item, "cf", None) is not None and np.size(item.cf) == x.size:
        tx = -np.asarray(item.ny, float)
        cf = np.asarray(item.cf, float) * np.where(tx >= 0, 1.0, -1.0)
    br = [_Branch(t[i], cp[i], None if cf is None else cf[i], s) for i, s in ((up, "upper"), (lo, "lower"))]
    return _CpSeries(label, br, np.column_stack([t, nrm]), [])


@_builder((6.7, 4.2))
def fig_cp(data: Any, *, st: FigureStyle, figsize: tuple[float, float], labels: Sequence[str] | None = None,
           kind: str = "airfoil", show_geometry: bool = True, show_cf: bool = False,
           title: str | None = None, body_names: Sequence[str] | None = None) -> Figure:
    """Surface pressure (and optionally skin-friction) distribution, upper and lower surface.

    ``data`` is a :class:`SurfaceData`, a :class:`PolarPoint` with distributions, or a list of
    either (overlaid for comparisons; ``labels`` names them). The Cp axis is inverted so suction
    plots upward, the convention every airfoil reference uses. The chordwise coordinate is
    measured along the front-to-rear line of each body, so wind-frame (rotated) surfaces still
    read as x/c. ``kind="vehicle"`` relabels the axes as x/L with roof / underbody branches;
    ``kind="body"`` uses x/L with neutral upper / lower names (3D centre-line cuts).
    ``body_names`` restricts SurfaceData lists to the named bodies (e.g. skip wheels).
    """
    items = list(data) if isinstance(data, (list, tuple)) else [data]
    if body_names is not None:
        items = [it for it in items if not hasattr(it, "body") or it.body in set(body_names)]
    items = [it for it in items if getattr(it, "cp", None) is not None and np.size(it.cp) > 2]
    if labels is None:
        labels = [getattr(it, "body", None) or (f"α = {it.alpha:g}°" if hasattr(it, "alpha") else f"#{i + 1}")
                  for i, it in enumerate(items)]
    series = [_cp_series(it, str(lab)) for it, lab in zip(items, labels, strict=False)]
    has_cf = show_cf and any(b.cf is not None for s in series for b in s.branches)
    ratios = [4.0] + ([1.9] if has_cf else []) + ([0.75] if show_geometry and series else [])
    fig = _new_figure(st, figsize)
    axes = fig.subplots(len(ratios), 1, sharex=True, gridspec_kw={"height_ratios": ratios})
    axes = np.atleast_1d(axes)
    ax = axes[0]
    xlab = "x/c" if kind == "airfoil" else "x/L"
    # "body": a bluff body or other 3D shape cut along its centre plane (x/L, neutral side names)
    names = ({"upper": "upper (roof)", "lower": "lower (underbody)"} if kind == "vehicle"
             else {"upper": "upper surface", "lower": "lower surface"})
    if not series:
        _no_data(ax, st, "No surface data")
        return fig
    multi = len(series) > 1
    for k, s in enumerate(series):
        for b in s.branches:
            if multi:
                color = st.series[k % len(st.series)]
                ls = "-" if b.side == "upper" else (0, (4, 2))
                lab = s.label if b.side == "upper" else None
            else:
                color = st.series[0] if b.side == "upper" else st.series[1]
                ls = "-"
                lab = names[b.side]
            ax.plot(b.xc, b.cp, color=color, ls=ls, lw=1.3, label=lab, zorder=3)
        for b in s.cp_inviscid:
            ax.plot(b.xc, b.cp, color=st.muted, lw=0.8, ls=(0, (1.5, 1.5)), zorder=2,
                    label="inviscid" if (b.side == "upper" and k == 0) else None)
    ax.axhline(0.0, color=st.spine, lw=0.6, zorder=1)
    ax.axhline(1.0, color=st.spine, lw=0.5, ls=(0, (2, 3)), zorder=1)
    ax.text(1.0, 1.0, "stagnation  ", transform=ax.get_yaxis_transform(), ha="right", va="top",
            fontsize=6.5, color=st.muted)
    ax.invert_yaxis()
    ax.set_ylabel("Pressure coefficient Cp")
    if multi:
        # colour identifies the case, line style the surface: one entry each instead of 2n entries
        handles, labs = ax.get_legend_handles_labels()
        order = sorted(range(len(labs)), key=lambda i: labs[i] == "inviscid")   # cases first
        handles, labs = [handles[i] for i in order], [labs[i] for i in order]
        handles += [Line2D([], [], color=st.muted, lw=1.2, ls="-"),
                    Line2D([], [], color=st.muted, lw=1.2, ls=(0, (4, 2)))]
        labs += [names["upper"], names["lower"]]
        # above the plot: overlaid distributions leave no free corner inside the axes
        ax.legend(handles, labs, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=min(len(labs), 6),
                  borderaxespad=0.0, columnspacing=1.2)
    else:
        # lifted clear of the "stagnation" label at Cp = 1 in the lower right corner
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07))
    if title:
        if multi:   # the legend sits where an axes title would go
            fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
        else:
            ax.set_title(title)
    row = 1
    if has_cf:
        axf = axes[row]
        row += 1
        for k, s in enumerate(series):
            for b in s.branches:
                if b.cf is None:
                    continue
                color = st.series[k % len(st.series)] if multi else (st.series[0] if b.side == "upper"
                                                                       else st.series[1])
                ls = "-" if (not multi or b.side == "upper") else (0, (4, 2))
                axf.plot(b.xc, b.cf, color=color, ls=ls, lw=1.1)
        axf.axhline(0.0, color=st.spine, lw=0.6)
        # Frame the boundary layer, not the stagnation point: Cf changes sign there and its
        # leading-edge peak is several times the value anywhere else on the surface.
        vals = np.concatenate([b.cf[(b.xc > 0.03) & np.isfinite(b.cf)] for s in series for b in s.branches
                               if b.cf is not None] or [np.zeros(1)])
        if vals.size > 3:
            lo, hi = np.percentile(vals, [1, 99])
            span = max(hi - lo, 1e-6)
            axf.set_ylim(min(lo - 0.25 * span, -0.1 * span), hi + 0.25 * span)
        axf.set_ylabel("Skin friction Cf")
        axf.text(0.995, 0.04, "Cf < 0: reversed flow", transform=axf.transAxes, ha="right", va="bottom",
                 fontsize=6.5, color=st.muted)
    if show_geometry:
        axg = axes[row]
        o = series[0].outline
        if o is not None:
            axg.fill(o[:, 0], o[:, 1], facecolor=st.body_fill, edgecolor=st.body_edge, lw=0.7)
        # True aspect where the panel allows it (sized from its physical height, since sharex
        # rules out set_aspect); thick sections are squeezed rather than clipped.
        h_in = figsize[1] * ratios[-1] / sum(ratios) * 0.82
        w_in = figsize[0] * 0.86
        half = 0.5 * 1.06 * h_in / w_in
        if o is not None:
            mid = 0.5 * (o[:, 1].min() + o[:, 1].max())
            half = max(half, 0.6 * float(np.ptp(o[:, 1])))
        else:
            mid = 0.0
        axg.set_ylim(mid - half, mid + half)
        axg.set_yticks([])
        axg.grid(False)
        axg.spines["left"].set_visible(False)
        axg.set_xlabel(xlab)
    else:
        axes[-1].set_xlabel(xlab)
    axes[-1].set_xlim(-0.03, 1.03)
    return fig


# ---------------------------------------------------------------------------
# convergence history
# ---------------------------------------------------------------------------

@_builder((6.7, 4.4))
def fig_history(history: Mapping[str, np.ndarray], t_avg_start: float | None = None, stats: Any = None, *,
                st: FigureStyle, figsize: tuple[float, float], keys: Sequence[str] | None = None,
                title: str | None = None, labels: Mapping[str, str] | None = None) -> Figure:
    """Force-coefficient histories with the averaging window and the mean ± standard error.

    ``history`` holds ``t`` (convective times) and coefficient series (``cd``, ``cl``, ``cm``).
    ``stats`` maps each key to a :class:`Stat` (a :class:`CFDResult` works too: its ``cd``/``cl``/
    ``cm`` attributes are used). The shaded band is the window the reported means come from; the
    darker band is ±1 standard error of that mean, i.e. the statistical uncertainty of the number
    in the results table, not the size of the oscillation. ``keys`` picks other series (the
    Nusselt number "nu", a fan's "fan0_thrust"); ``labels`` gives their axis labels.
    """
    t = np.asarray(history.get("t", []), float)
    if keys is None:
        keys = [k for k in ("cd", "cl", "cm") if k in history and np.size(history[k]) == t.size and t.size]
    keys = list(keys)
    fig = _new_figure(st, figsize)
    if not keys:
        ax = fig.add_subplot()
        _no_data(ax, st, "No force history recorded")
        return fig
    axes = np.atleast_1d(fig.subplots(len(keys), 1, sharex=True))
    t_end = float(t[-1]) if t.size else 1.0
    for ax, key in zip(axes, keys, strict=True):
        y = np.asarray(history[key], float)
        td, yd = _decimate(t, y)
        color = st.series[{"cd": 0, "cl": 1, "cm": 3}.get(key, 4)]
        ax.plot(td, yd, color=color, lw=0.9, zorder=3)
        win = (t >= t_avg_start) if t_avg_start is not None and math.isfinite(t_avg_start) else None
        if win is not None and win.any():
            ax.axvspan(t_avg_start, t_end, color=st.accent, alpha=0.07, lw=0, zorder=0)
            ax.axvline(t_avg_start, color=st.accent, lw=0.8, ls=(0, (3, 2)), zorder=2)
        s = _stat_of(stats, key)
        if s is None and win is not None and win.any():
            s = Stat(mean=float(np.mean(y[win])))
        if s is not None and math.isfinite(s.mean):
            xa = t_avg_start if (t_avg_start is not None and math.isfinite(t_avg_start)) else float(t[0])
            if math.isfinite(s.stderr) and s.stderr > 0:
                ax.fill_between([xa, t_end], s.mean - s.stderr, s.mean + s.stderr, color=st.text,
                                alpha=0.18, lw=0, zorder=4)
            ax.plot([xa, t_end], [s.mean, s.mean], color=st.text, lw=1.0, zorder=5)
            ax.text(0.995, 0.94, f"mean {_fmt_pm(s.mean, s.stderr)}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=st.text,
                    bbox={"boxstyle": "round,pad=0.25", "fc": st.bg, "ec": st.grid, "lw": 0.5})
        core = y[win] if win is not None and win.any() else y[y.size // 2:]
        ax.set_ylim(*_robust_limits(core, y[y.size // 10:]))
        if labels and key in labels:
            ax.set_ylabel(labels[key])
        else:
            sym, name = COEFF_LABELS.get(key, (key, key))
            ax.set_ylabel(f"{name.split()[0]} {sym}")
    axes[-1].set_xlabel("Time t·U/L [convective units]")
    axes[-1].set_xlim(float(t[0]) if t.size else 0.0, t_end)
    if t_avg_start is not None and math.isfinite(t_avg_start):
        axes[0].text(t_avg_start, 1.02, " averaging window", transform=axes[0].get_xaxis_transform(),
                     ha="left", va="bottom", fontsize=7, color=st.accent)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


# ---------------------------------------------------------------------------
# polars
# ---------------------------------------------------------------------------

@_builder((6.7, 5.4))
def fig_polar(polars: PolarResult | Sequence[PolarResult], *, st: FigureStyle, figsize: tuple[float, float],
              labels: Sequence[str] | None = None, highlight_best: bool = True,
              title: str | None = None) -> Figure:
    """The four classic polar plots: Cl-α, drag polar Cl-Cd, Cm-α and L/D-α.

    Several polars are overlaid with one colour each (comparisons). Only converged points are
    plotted; the legend notes how many points failed so a gap is never mistaken for physics.
    ``highlight_best`` rings the best-L/D point in every panel.
    """
    plist = [polars] if isinstance(polars, PolarResult) or hasattr(polars, "points") else list(polars)
    if labels is None:
        labels = [f"Re {_fmt(p.re, 3)}" for p in plist]
    fig = _new_figure(st, figsize)
    axs = fig.subplots(2, 2)
    a_cl, a_dp, a_cm, a_ld = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    any_data = False
    for k, (p, lab) in enumerate(zip(plist, labels, strict=False)):
        tb = p.table()
        if tb["alpha"].size == 0:
            continue
        any_data = True
        o = np.argsort(tb["alpha"])
        a, cl, cd, cm = tb["alpha"][o], tb["cl"][o], tb["cd"][o], tb["cm"][o]
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(cd > 0, cl / cd, np.nan)
        n_bad = sum(1 for q in p.points if not q.converged)
        lab2 = lab + (f" ({n_bad} not converged)" if n_bad else "")
        c = st.series[k % len(st.series)]
        kw = {"color": c, "lw": 1.3, "marker": "o", "ms": 2.8, "mew": 0}
        a_cl.plot(a, cl, label=lab2, **kw)
        a_dp.plot(cd, cl, **kw)
        a_cm.plot(a, cm, **kw)
        a_ld.plot(a, ld, **kw)
        if highlight_best and np.isfinite(ld).any():
            i = int(np.nanargmax(ld))
            ring = {"marker": "o", "ms": 7, "mfc": "none", "mec": c, "mew": 1.1, "zorder": 6, "ls": "none"}
            a_cl.plot(a[i], cl[i], **ring)
            a_dp.plot(cd[i], cl[i], **ring)
            a_cm.plot(a[i], cm[i], **ring)
            a_ld.plot(a[i], ld[i], **ring)
            if len(plist) == 1:
                a_ld.text(0.98, 0.04, f"(L/D)max = {ld[i]:.1f} at α = {a[i]:g}°", transform=a_ld.transAxes,
                          ha="right", va="bottom", fontsize=7, color=st.text,
                          bbox={"boxstyle": "round,pad=0.25", "fc": st.bg, "ec": st.grid, "lw": 0.5})
    for ax in (a_cl, a_cm, a_ld):
        ax.axvline(0, color=st.spine, lw=0.5)
        ax.set_xlabel("Angle of attack α [°]")
    a_cl.axhline(0, color=st.spine, lw=0.5)
    a_cm.axhline(0, color=st.spine, lw=0.5)
    a_cl.set_ylabel("Lift coefficient Cl")
    a_dp.set_xlabel("Drag coefficient Cd")
    a_dp.set_ylabel("Lift coefficient Cl")
    a_cm.set_ylabel("Moment coefficient Cm (c/4)")
    a_ld.set_ylabel("Lift-to-drag ratio L/D")
    a_cl.set_title("Lift curve")
    a_dp.set_title("Drag polar")
    a_cm.set_title("Pitching moment")
    a_ld.set_title("Efficiency")
    if not any_data:
        for ax in axs.flat:
            _no_data(ax, st, "No converged points")
    elif len(plist) > 1 or any(not q.converged for p in plist for q in p.points):
        handles, labs = a_cl.get_legend_handles_labels()
        fig.legend(handles, labs, loc="outside upper center", ncols=min(len(labs), 3))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

def _field_array(fd: FieldData, name: str, q_inf: float | None) -> np.ndarray | None:
    """A named field, deriving speed / vorticity / cp from u, v, p when not stored."""
    f = fd.fields
    if name in f:
        return np.asarray(f[name], float)
    ny_nx = next(iter(f.values())).shape if f else None
    if ny_nx is None:
        return None
    if name == "speed" and "u" in f and "v" in f:
        return np.hypot(np.asarray(f["u"], float), np.asarray(f["v"], float))
    if name == "vorticity" and "u" in f and "v" in f:
        ny, nx = ny_nx
        dx = (fd.extent[1] - fd.extent[0]) / nx
        dy = (fd.extent[3] - fd.extent[2]) / ny
        u, v = np.asarray(f["u"], float), np.asarray(f["v"], float)
        return np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0)
    if name == "cp" and "p" in f and q_inf:
        return np.asarray(f["p"], float) / q_inf
    return None


def _auto_view(outs: Sequence[_Outline], extent: Sequence[float], ground: bool,
               aspect: float = 0.36) -> tuple[float, float, float, float]:
    """A view framing the bodies and the near wake: 0.6 lengths upstream, 1.6 downstream."""
    pts = np.vstack([o.points for o in outs])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    lx = max(xmax - xmin, 1e-9)
    x0, x1 = xmin - 0.6 * lx, xmax + 1.6 * lx
    if ground:
        # road at the bottom of the view; above the body only as much as the flow needs
        y0 = extent[2]
        y1 = y0 + max(2.4 * (ymax - y0), 0.25 * (x1 - x0))
    else:
        hh = max((x1 - x0) * aspect, (ymax - ymin) * 1.6)
        yc = 0.5 * (ymin + ymax)
        y0, y1 = yc - hh / 2, yc + hh / 2
    ex = extent
    x0, x1 = max(x0, ex[0]), min(x1, ex[1])
    y0, y1 = max(y0, ex[2]), min(y1, ex[3])
    return float(x0), float(x1), float(y0), float(y1)


# Dark-centred diverging maps suit the dark live view but print as a black slab; the print
# style swaps them for a light-centred equivalent unless the caller asked for a map explicitly.
_PRINT_SWAP = {"icefire": "RdBu_r", "grayscale": "Greys"}


def _field_labels(name: str) -> tuple[str, str, bool]:
    """(label, unit symbol, diverging) — from aerolume.viz when present so GUI and report agree."""
    label, unit, diverging = FIELD_LABELS.get(name, (name, "", False))
    info = (_viz_attr("FIELD_INFO") or {}).get(name)
    if info is not None:
        label, diverging = str(info[0]), bool(info[2])
        try:
            from .colormaps import UNIT_SYMBOL
            unit = UNIT_SYMBOL.get(str(info[1]), unit)
        except Exception:  # noqa: BLE001 - optional sibling module
            log.debug("viz.colormaps.UNIT_SYMBOL unavailable; using built-in units")
    return label, unit, diverging


@_builder(None)
def fig_field(fields: FieldData, name: str, bodies: Any = None, *, st: FigureStyle,
              figsize: tuple[float, float] | None, cmap: str | Colormap | None = None, vmin: float | None = None,
              vmax: float | None = None, streamlines: bool = False, view: str | Sequence[float] = "auto",
              ground: bool = False, title: str | None = None, q_inf: float | None = None,
              stream_density: float = 1.1, xlabel: str = "x [m]", ylabel: str = "y [m]",
              lines: Sequence[np.ndarray] | None = None, center: float | None = None) -> Figure:
    """A colour map of one flow field with the bodies, an optional streamline overlay and a colorbar.

    ``fields`` is a :class:`FieldData` (row 0 = lowest y). ``speed``, ``vorticity`` and ``cp``
    are derived from ``u``, ``v``, ``p`` when not stored (``cp`` needs ``q_inf`` [Pa]). The colour
    range defaults to robust percentiles inside the view, symmetric for diverging quantities and
    centred on zero for Cp, so one thin wall layer cannot wash out the whole picture.
    ``view`` is "auto" (frame the bodies and near wake), "full" or (xmin, xmax, ymin, ymax) in m.
    Without ``figsize`` the height is fitted to the view at 6.7 in wide. ``center`` draws the
    field with a diverging map centred on that value (a temperature field with hot and cold
    walls, centred on the ambient temperature); ``cmap`` then defaults to "RdBu_r".

    The same builder draws a plane cut of a 3D result: pass the slice's in-plane velocity
    components as ``u`` / ``v``, the axis names as ``xlabel`` / ``ylabel`` and the body's exact
    section as ``bodies`` (closed loops) and ``lines`` (open chains, e.g. a thin sheet).
    """
    arr = _field_array(fields, name, q_inf)
    outs = _outlines(bodies)
    if arr is None:
        fig = _new_figure(st, figsize or (6.7, 3.0))
        _no_data(fig.add_subplot(), st, f"Field '{name}' not available")
        return fig
    ny, nx = arr.shape
    ext = tuple(float(v) for v in fields.extent)
    if isinstance(view, str):
        vw = ext if (view == "full" or not outs) else _auto_view(outs, ext, ground)
    else:
        vw = tuple(float(v) for v in view)
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[2] + (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    ix = np.flatnonzero((xs >= vw[0]) & (xs <= vw[1]))
    iy = np.flatnonzero((ys >= vw[2]) & (ys <= vw[3]))
    if ix.size < 2 or iy.size < 2:
        ix, iy = np.arange(nx), np.arange(ny)
    sub = arr[np.ix_(iy, ix)]
    if figsize is None:
        vw_w = (ix[-1] - ix[0] + 1) * (ext[1] - ext[0]) / nx
        vw_h = (iy[-1] - iy[0] + 1) * (ext[3] - ext[2]) / ny
        figsize = (6.7, _fit_height(6.7, vw_w, vw_h, side_in=1.55, extra_in=0.75, lo=2.0, hi=6.0))
    fig = _new_figure(st, figsize)
    ax = fig.add_subplot()
    solid = None
    if fields.mask is not None:
        solid = np.asarray(fields.mask)[np.ix_(iy, ix)] > 0.5
        sub = np.where(solid, np.nan, sub)
    finite = sub[np.isfinite(sub)]
    if solid is not None and outs and solid.any() and not solid.all():
        # The polygons are drawn over the image, so solid cells are only visible where the
        # staircase mask pokes out of the smooth outline. Giving them the nearest fluid value
        # avoids a jagged pale fringe around every body.
        try:
            from scipy.ndimage import distance_transform_edt
            _, (ii, jj) = distance_transform_edt(solid, return_indices=True)
            sub = sub[ii, jj]
        except Exception:  # noqa: BLE001 - scipy missing: keep the NaN cells (drawn in body colour)
            log.debug("scipy unavailable; solid cells left blank")
    label, unit, diverging = _field_labels(name)
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    norm: Normalize
    if name == "cp":
        lo = vmin if vmin is not None else min(float(np.percentile(finite, 1)), -0.1)
        hi = vmax if vmax is not None else max(min(float(np.percentile(finite, 99.5)), 1.0), 0.1)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)
    elif center is not None and math.isfinite(float(center)):
        c = float(center)
        m = float(np.percentile(np.abs(finite - c), 99.5)) or 1.0
        lo = vmin if vmin is not None else c - m
        hi = vmax if vmax is not None else c + m
        norm = TwoSlopeNorm(vcenter=c, vmin=min(lo, c - 1e-9 * max(m, 1.0)), vmax=max(hi, c + 1e-9 * max(m, 1.0)))
        if cmap is None:
            cmap = "RdBu_r"
    elif diverging:
        q = 98.0 if name == "vorticity" else 99.5
        m = float(np.percentile(np.abs(finite), q)) or 1.0
        norm = Normalize(vmin if vmin is not None else -m, vmax if vmax is not None else m)
    else:
        lo = vmin if vmin is not None else (0.0 if name in ("speed", "dye", "dye1", "dye2", "dye3", "nut", "mask")
                                            else float(np.percentile(finite, 0.5)))
        hi = vmax if vmax is not None else float(np.percentile(finite, 99.5))
        norm = Normalize(lo, hi if hi > lo else lo + 1.0)
    if cmap is None and st.name != "dark":
        default = (_viz_attr("DEFAULT_CMAP") or {}).get(name)
        cmap = _PRINT_SWAP.get(default, default) if default else None
    cm = _resolve_cmap(name, cmap).with_extremes(bad=st.body_fill)
    dx = (ext[1] - ext[0]) / nx
    dy = (ext[3] - ext[2]) / ny
    sub_ext = (xs[ix[0]] - dx / 2, xs[ix[-1]] + dx / 2, ys[iy[0]] - dy / 2, ys[iy[-1]] + dy / 2)
    im = ax.imshow(np.ma.masked_invalid(sub), origin="lower", extent=sub_ext, cmap=cm, norm=norm,
                   interpolation="bilinear", aspect="equal", zorder=1)
    if streamlines:
        u = _field_array(fields, "u", None)
        v = _field_array(fields, "v", None)
        if u is not None and v is not None:
            us, vs = u[np.ix_(iy, ix)], v[np.ix_(iy, ix)]
            if solid is not None:
                us = np.ma.masked_where(solid, us)
                vs = np.ma.masked_where(solid, vs)
            try:
                # no arrowheads: streamplot puts them mid-line, which lines up into a distracting
                # column on uniform flow; the flow direction is shown by the U-inf convention instead
                ax.streamplot(xs[ix], ys[iy], us, vs, density=stream_density, color=st.stream,
                              linewidth=0.45, arrowstyle="-", broken_streamlines=True, zorder=2)
            except Exception:
                log.warning("streamlines skipped", exc_info=True)
    if outs:
        _draw_bodies(ax, outs, st, lw=0.8, zorder=4)
    elif solid is not None and solid.any() and not lines:
        ax.contour(xs[ix], ys[iy], solid.astype(float), levels=[0.5], colors=[st.body_edge], linewidths=0.8)
    for ln in lines or ():
        p = np.asarray(ln, float)
        ax.plot(p[:, 0], p[:, 1], color=st.body_edge, lw=0.9, zorder=4, solid_capstyle="round")
    if ground:
        ax.axhline(0.0, color=st.ground, lw=1.4, zorder=5)
    ax.set_xlim(sub_ext[0], sub_ext[1])
    ax.set_ylim(sub_ext[2], sub_ext[3])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=22, shrink=1.0)
    cb.set_label(f"{label} [{unit}]" if unit else label, fontsize=8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    if title is None:
        prefix = {"mean": "Mean", "instant": "Instantaneous"}.get(fields.label, "")
        title = f"{prefix} {label[0].lower()}{label[1:]}" if prefix else label
    ax.set_title(title)
    return fig


# ---------------------------------------------------------------------------
# comparison bars
# ---------------------------------------------------------------------------

def _row_value(row: Mapping[str, Any], metric: str) -> tuple[float, float]:
    v = row.get(metric, float("nan"))
    err = row.get(f"{metric}_err", row.get("err", float("nan")) if isinstance(metric, str) else float("nan"))
    if isinstance(v, Stat) or hasattr(v, "mean"):
        err = v.stderr
        v = v.mean
    try:
        return float(v), float(err) if err is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan"), float("nan")


METRIC_LABELS = {"cd": "Drag coefficient Cd", "cl": "Lift coefficient Cl", "cm": "Moment coefficient Cm",
                 "ld": "Lift-to-drag ratio L/D", "drag": "Drag D′ [N/m]", "lift": "Lift L′ [N/m]",
                 "drag_area": "Drag area CdA [m²/m]", "power": "Drag power [W/m]",
                 "strouhal": "Strouhal number St", "cl_max": "Max lift coefficient Clmax",
                 "ld_max": "Max lift-to-drag (L/D)max", "cd_min": "Min drag coefficient Cdmin"}


#: Longest line of a bar label before it wraps, and the most lines it may take: a simulation name
#: such as "Heated cavity · Ra 1,000 · 36 cells/ref" would otherwise take more than half the figure
#: width, and constrained layout gives up and clips the labels on the left.
BAR_LABEL_WIDTH = 26
BAR_LABEL_LINES = 2


def _bar_label(text: str, width: int = BAR_LABEL_WIDTH, lines: int = BAR_LABEL_LINES) -> str:
    """``text`` wrapped at word boundaries to at most ``lines`` lines of about ``width``
    characters, the last one shortened with an ellipsis."""
    import textwrap
    if len(text) <= width:
        return text
    out = textwrap.wrap(text, width=width, break_long_words=True) or [text]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: max(width - 1, 1)].rstrip(" ·") + "…"
    return "\n".join(out)


@_builder(None)
def fig_compare_bars(rows: Sequence[Mapping[str, Any]], metric: str | Sequence[str] = "cd", *,
                     st: FigureStyle, figsize: tuple[float, float] | None, baseline: int = 0,
                     metric_labels: Mapping[str, str] | None = None, title: str | None = None) -> Figure:
    """Horizontal bars of one or more metrics across variants, with Δ% against a baseline.

    ``rows`` are mappings with a ``label`` (or ``name``) and the metric value either as a float
    (with an optional ``<metric>_err`` standard error) or as a :class:`Stat`. One panel per
    metric. Error bars are ±1 standard error, so overlapping bars mean "not distinguishable at
    this simulation length" — the most common trap when ranking close variants.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    labels = [_bar_label(str(r.get("label", r.get("name", f"#{i + 1}")))) for i, r in enumerate(rows)]
    if figsize is None:
        lines = max((lab.count("\n") + 1 for lab in labels), default=1)
        figsize = (6.7, min(0.9 + (0.32 + 0.16 * (lines - 1)) * max(len(rows), 1), 7.5))
    fig = _new_figure(st, figsize)
    axs = np.atleast_1d(fig.subplots(1, len(metrics), sharey=True))
    ypos = np.arange(len(rows))[::-1]
    for ax, m in zip(axs, metrics, strict=True):
        vals = [_row_value(r, m) for r in rows]
        v = np.array([a for a, _ in vals])
        e = np.array([b for _, b in vals])
        colors = [st.spine if i == baseline else st.series[0] for i in range(len(rows))]
        ax.barh(ypos, np.nan_to_num(v), height=0.58, color=colors, zorder=3)
        ok = np.isfinite(e) & (e > 0)
        if ok.any():
            ax.errorbar(v[ok], ypos[ok], xerr=e[ok], fmt="none", ecolor=st.text, elinewidth=0.8,
                        capsize=2.5, zorder=4)
        span = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
        base = v[baseline] if 0 <= baseline < v.size else float("nan")
        # a baseline that is zero up to round-off (a symmetric body's lift) gives no percentages
        if math.isfinite(base) and abs(base) <= 1e-3 * max(span, 1e-300):
            base = float("nan")
        for i, (yy, val) in enumerate(zip(ypos, v, strict=True)):
            if not math.isfinite(val):
                ax.text(0, yy, " n/a", va="center", ha="left", fontsize=7, color=st.muted)
                continue
            txt = _fmt(val, 4)
            if i != baseline and math.isfinite(base) and base != 0:
                d = (val - base) / abs(base) * 100
                txt += f"  ({d:+.1f} %)"
            off = (e[i] if ok[i] else 0.0) + 0.02 * span
            ax.text(val + (off if val >= 0 else -off), yy, txt, va="center",
                    ha="left" if val >= 0 else "right", fontsize=7, color=st.text, zorder=5)
        lo = min(0.0, float(np.nanmin(v)) if np.isfinite(v).any() else 0.0)
        hi = max(0.0, float(np.nanmax(v)) if np.isfinite(v).any() else 1.0)
        ax.set_xlim(lo - (0.45 * span if lo < 0 else 0), hi + 0.45 * span)
        ax.axvline(0, color=st.spine, lw=0.6)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel((metric_labels or {}).get(m) or METRIC_LABELS.get(m, m))
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(ypos, labels)
    for tl in axs[0].get_yticklabels():
        tl.set_color(st.text)
    if 0 <= baseline < len(labels):
        axs[0].get_yticklabels()[baseline].set_fontweight("bold")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------

def _vertex_normals(p: np.ndarray) -> np.ndarray:
    """Outward unit normals of a CCW polygon at its vertices (average of adjacent edge normals)."""
    e = np.roll(p, -1, axis=0) - p
    en = np.column_stack([e[:, 1], -e[:, 0]])
    en /= np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-300)
    vn = en + np.roll(en, 1, axis=0)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)


OBJECTIVES = {"cd": ("Cd", "reduce"), "cl": ("Cl", "increase"), "ld": ("L/D", "increase"),
              "drag": ("drag", "reduce"), "lift": ("lift", "increase"), "downforce": ("downforce", "increase")}


@_builder(None)
def fig_sensitivity(bodies: Any, sensitivity: Sequence[np.ndarray] | None = None, *, st: FigureStyle,
                    figsize: tuple[float, float] | None, objective: str = "cd", arrows: bool = True,
                    n_arrows: int = 36, cmap: str | Colormap = "RdBu_r", ground: bool = False,
                    title: str | None = None) -> Figure:
    """Shape sensitivity map: the outline coloured by dJ/dn, with arrows showing the improving move.

    ``bodies`` is a list of :class:`SurfaceData` carrying ``sensitivity`` (dJ/dn per metre of
    outward displacement) or a list of bodies/polygons with ``sensitivity`` given separately, one
    array per body aligned with its vertices. Red means pushing the surface *outward* raises J,
    blue means it lowers J. Arrows point the way to move the surface to improve the objective
    ("reduce" Cd/drag, "increase" Cl, L/D, downforce), sized by local strength; tiny values are
    skipped so the picture shows where reshaping actually pays off. Colours are relative to the
    98th-percentile magnitude (the colour bar states that value in 1/m): the map's pattern is
    reliable, its absolute size only indicative (docs/theory_sensitivity.md section 5).
    """
    outs = _outlines(bodies)
    if sensitivity is None:
        sens = [None if getattr(b, "sensitivity", None) is None else np.asarray(b.sensitivity, float)
                for b in (bodies if isinstance(bodies, (list, tuple)) else [bodies])]
    else:
        sens = [None if s is None else np.asarray(s, float) for s in sensitivity]
    normals: list[np.ndarray] = []
    for b, o in zip((bodies if isinstance(bodies, (list, tuple)) else [bodies]), outs, strict=False):
        if hasattr(b, "nx") and getattr(b, "nx", None) is not None and np.size(b.nx) == len(o.points):
            normals.append(np.column_stack([b.nx, b.ny]).astype(float))
        else:
            normals.append(_vertex_normals(o.points))
    pairs = [(o, s, n) for o, s, n in zip(outs, sens, normals, strict=False)
             if s is not None and s.size == len(o.points)]
    if not pairs:
        fig = _new_figure(st, figsize or (6.7, 2.6))
        _no_data(fig.add_subplot(), st, "No sensitivity data")
        return fig
    pts_all = np.vstack([o.points for o in outs])
    lscale = 0.10 * float(np.ptp(pts_all[:, 0]) or 1.0)
    (bx0, by0), (bx1, by1) = pts_all.min(axis=0), pts_all.max(axis=0)
    fx0, fx1 = bx0 - 1.4 * lscale, bx1 + 1.4 * lscale
    fy0, fy1 = (min(by0, 0.0) - 0.6 * lscale if ground else by0 - 1.4 * lscale), by1 + 1.4 * lscale
    if figsize is None:
        figsize = (6.7, _fit_height(6.7, fx1 - fx0, fy1 - fy0, side_in=1.55, extra_in=1.0, lo=2.2, hi=5.0))
    fig = _new_figure(st, figsize)
    ax = fig.add_subplot()
    allv = np.concatenate([np.abs(s[np.isfinite(s)]) for _, s, _ in pairs])
    smax = float(np.percentile(allv, 98)) if allv.size else 1.0
    smax = smax or 1.0
    cm = _resolve_cmap("sensitivity", cmap)
    # colours relative to the 98th-percentile magnitude, not in 1/m: the pattern of the
    # map is right but its size can be about twice the physical dJ/dn
    # (docs/theory_sensitivity.md section 5)
    norm = Normalize(-1.0, 1.0)
    _draw_bodies(ax, outs, st, lw=0.0, zorder=2, fill=st.panel)
    sym, sense = OBJECTIVES.get(objective, (objective, "reduce"))
    sign = -1.0 if sense == "reduce" else 1.0
    lc = None
    for o, s, nrm in pairs:
        p = o.points
        seg = np.stack([p, np.roll(p, -1, axis=0)], axis=1)
        val = np.clip(0.5 * (s + np.roll(s, -1)) / smax, -1.0, 1.0)
        lc = LineCollection(seg, cmap=cm, norm=norm, linewidths=3.2, capstyle="round", zorder=4)
        lc.set_array(val)
        ax.add_collection(lc)
        ax.add_collection(LineCollection(seg, colors=st.body_edge, linewidths=0.4, zorder=5))
        if arrows and n_arrows > 0:
            ds = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
            sarc = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
            total = float(ds.sum()) or 1.0
            share = max(2, round(n_arrows * total / sum(
                float(np.linalg.norm(np.diff(np.vstack([q.points, q.points[:1]]), axis=0), axis=1).sum())
                for q, _, _ in pairs)))
            targets = (np.arange(share) + 0.5) * total / share
            idx = np.unique(np.searchsorted(sarc, targets).clip(0, len(p) - 1))
            mag = np.abs(s[idx]) / smax
            keep = idx[(mag > 0.12) & np.isfinite(mag)]
            if keep.size:
                d = (sign * np.clip(s[keep] / smax, -1, 1))[:, None] * nrm[keep] * lscale
                ax.quiver(p[keep, 0], p[keep, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1,
                          color=st.text, width=0.0028, headwidth=3.6, headlength=4.2, zorder=6)
    if ground:
        _draw_ground(ax, st, fx0, fx1, 0.5 * lscale)
    ax.set_xlim(fx0, fx1)
    ax.set_ylim(fy0, fy1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, lw=0.4)
    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02, aspect=22, extend="both")
        cb.set_label(f"∂{sym}/∂n, relative (1 = {smax:.3g} per m)", fontsize=8)
        cb.outline.set_linewidth(0.5)
        cb.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    ax.set_title(title if title is not None else f"Shape sensitivity of {sym}")
    fig.supxlabel(f"Arrows: move the surface this way to {sense} {sym}.  Red: growing outward raises "
                  f"{sym}; blue: growing outward lowers it.", x=0.02, ha="left", fontsize=7, color=st.muted)
    return fig


# ---------------------------------------------------------------------------
# 3D: rendered views
# ---------------------------------------------------------------------------

def _colorbar_from_spec(fig: Figure, cax: Any, color: Any, orientation: str) -> None:
    """A matplotlib colour bar for a range rendered elsewhere (viz3d): same LUT, report font."""
    from matplotlib.cm import ScalarMappable
    cm = _resolve_cmap("", getattr(color, "cmap", "viridis"))
    norm = Normalize(float(color.vmin), float(color.vmax))
    sm = ScalarMappable(norm=norm, cmap=cm)
    cb = fig.colorbar(sm, cax=cax, orientation=orientation, extend=getattr(color, "extend", "neither"),
                      extendfrac=0.04)
    label = getattr(color, "title", None) or getattr(color, "label", "")
    cb.set_label(label, fontsize=7.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7, width=0.5, length=2.5)


def _place_image(ax: Any, img: np.ndarray, extent: Sequence[float] | None) -> None:
    if extent is not None:
        ax.imshow(img, extent=tuple(extent), interpolation="antialiased", aspect="auto")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.imshow(img, interpolation="antialiased", aspect="auto")
    ax.set_axis_off()


def _outline(ax: Any, img: np.ndarray, extent: Sequence[float], st: FigureStyle) -> None:
    """A thin dark silhouette around the body in an orthographic view (engineering-drawing look).

    Flat-lit grey surfaces seen square-on lose their edge against a white page; the outline
    is traced on the rendered pixels (everything that differs from the corner colour), so it
    follows the true silhouette, holes included.
    """
    bg = img[0, 0].astype(np.int16)
    solid = np.abs(img.astype(np.int16) - bg).max(axis=2) > 10
    try:                            # a specular highlight as bright as the page is not a hole
        from scipy.ndimage import binary_fill_holes
        solid = binary_fill_holes(solid)
    except Exception:  # noqa: BLE001 - scipy missing: an occasional inner contour is harmless
        log.debug("scipy unavailable; outline holes not filled")
    mask = solid.astype(float)
    if mask.min() == mask.max():
        return
    h, w = mask.shape
    xs = np.linspace(extent[0], extent[1], w)
    ys = np.linspace(extent[3], extent[2], h)
    ax.contour(xs, ys, mask, levels=[0.5], colors=[st.body_edge], linewidths=0.6, zorder=4)


def _road(ax: Any, st: FigureStyle, extent: Sequence[float], z: float) -> None:
    """Road line at height ``z`` with a light hatched band beneath, in an orthographic view."""
    x0, x1 = sorted((float(extent[0]), float(extent[1])))
    bottom = min(float(extent[2]), float(extent[3]))
    if z <= bottom:
        return
    ax.add_patch(Rectangle((x0, bottom), x1 - x0, z - bottom, facecolor=st.panel, edgecolor=st.grid,
                           hatch="//////", zorder=0.5, linewidth=0))
    ax.plot([x0, x1], [z, z], color=st.ground, lw=0.9, zorder=3, solid_capstyle="butt")


@_builder(None)
def fig_views3d(images: Sequence[np.ndarray], names: Sequence[str], titles: Sequence[str] | None = None,
                extents: Sequence[Sequence[float] | None] | None = None, *, st: FigureStyle,
                figsize: tuple[float, float] | None, color: Any = None, ground: float | None = None,
                layout: str = "auto", width: float = 6.7, max_height: float = 8.6) -> Figure:
    """Rendered 3D views (``aerolume.viz3d``) arranged as a figure, with a matplotlib colour bar.

    ``images`` are RGB renders of the camera presets ``names`` ("iso", "side", "front", "top",
    "back"); ``extents`` give, for orthographic views, the image's world extent in metres
    (see ``report.render3d.Views``). With side, front and top views of a body longer than it
    is wide, ``layout="auto"`` draws them as an **engineering drawing**: one common scale, the
    side view above the top view (same x), the front view beside the side view (same z), the
    rear view (if given) beside the top view, and the iso view across the top; a reader can
    then compare the size of features between views. Orthographic views get a thin
    silhouette line. Other selections fall back to a grid of independently framed views.
    ``color`` (cmap, vmin, vmax, title/label, extend) adds a colour bar drawn in the report's
    font from the same lookup table the renderer used; ``ground`` (a height) draws the road in
    the side, front and rear views.
    """
    images = [np.asarray(i) for i in images]
    names = list(names)
    titles = list(titles) if titles is not None else names
    extents = list(extents) if extents is not None else [None] * len(images)
    idx = {n: i for i, n in enumerate(names)}
    pad, gap, tit = 0.04, 0.16, 0.2          # inches: outer pad, gap between views, title band
    drawing = False
    if layout in ("auto", "drawing") and all(n in idx and extents[idx[n]] is not None for n in ("side", "front", "top")):
        es, ef, et = (extents[idx[n]] for n in ("side", "front", "top"))
        lx, lz = abs(es[1] - es[0]), abs(es[3] - es[2])
        ly = abs(ef[1] - ef[0])
        drawing = layout == "drawing" or lx >= 0.9 * ly
    fig_w = float(figsize[0]) if figsize else width
    rects: list[tuple[int, float, float, float, float]] = []    # (image index, x, y_top, w, h) in inches
    cbar: tuple[str, float, float, float, float] | None = None
    y = pad
    if drawing:
        # one scale for all orthographic views, small enough that the two rows of views stay
        # under ~4.4 in (a compact body would otherwise fill the page), centred horizontally
        et_h = abs(et[3] - et[2])
        s = min((fig_w - 2 * pad - gap) / (lx + ly), 4.4 / max(lz + et_h, 1e-9))
        pad = max(pad, 0.5 * (fig_w - (lx + ly) * s - gap))
        cell_right = ly * s
        back = idx.get("back") if "back" in idx and extents[idx["back"]] is not None else None
        # keep the drawing inside max_height by shrinking the iso view first
        rows_h = tit + lz * s + gap + tit + abs(et[3] - et[2]) * s
        if "iso" in idx:
            ii = images[idx["iso"]]
            hi = min(fig_w * ii.shape[0] / ii.shape[1], 2.7, max(max_height - rows_h - tit - gap - 0.08, 1.2))
            wi = min(fig_w - 0.08, hi * ii.shape[1] / ii.shape[0])
            hi = wi * ii.shape[0] / ii.shape[1]
            rects.append((idx["iso"], (fig_w - wi) / 2, y + tit, wi, hi))
            y += tit + hi + gap
        rects.append((idx["side"], pad, y + tit, lx * s, lz * s))
        rects.append((idx["front"], pad + lx * s + gap, y + tit, ly * s, lz * s))
        y += tit + lz * s + gap
        ht = abs(et[3] - et[2]) * s
        rects.append((idx["top"], pad, y + tit, abs(et[1] - et[0]) * s, ht))
        if back is not None:
            eb = extents[back]
            rects.append((back, pad + lx * s + gap, y + tit, abs(eb[1] - eb[0]) * s, abs(eb[3] - eb[2]) * s))
            if color is not None:
                cbar = ("h", 0.0, 0.0, 0.0, 0.0)
        elif color is not None:
            if ht >= 1.0:
                cbar = ("v", pad + lx * s + gap + 0.08 * cell_right, y + tit + 0.1 * ht, 0.11, 0.8 * ht)
            else:
                cbar = ("h", 0.0, 0.0, 0.0, 0.0)
        y += tit + ht + pad
    else:
        n = len(images)
        ncol = 1 if n == 1 else 2
        cw = (fig_w - 2 * pad - (ncol - 1) * gap) / ncol
        for r0 in range(0, n, ncol):
            row = list(range(r0, min(r0 + ncol, n)))
            hmax = max(cw * images[i].shape[0] / images[i].shape[1] for i in row)
            hmax = min(hmax, 4.2 if n == 1 else 3.0)
            for k, i in enumerate(row):
                h_i = min(hmax, cw * images[i].shape[0] / images[i].shape[1])
                w_i = h_i * images[i].shape[1] / images[i].shape[0]
                x = pad + k * (cw + gap) + (cw - w_i) / 2
                rects.append((i, x, y + (tit if n > 1 else 0.0), w_i, h_i))
            y += (tit if n > 1 else 0.0) + hmax + gap
        y += pad - gap
        if color is not None:
            cbar = ("h", 0.0, 0.0, 0.0, 0.0)
    if cbar is not None and cbar[0] == "h":
        bw = min(3.4, 0.55 * fig_w)
        cbar = ("h", (fig_w - bw) / 2, y + 0.06, bw, 0.11)
        y += 0.06 + 0.11 + 0.42
    fig_h = float(figsize[1]) if figsize else y
    fig = _new_figure(st, (fig_w, fig_h), layout=None)

    def frac(x: float, yt: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x / fig_w, 1.0 - (yt + h) / fig_h, w / fig_w, h / fig_h)
    for i, x, yt, w, h in rects:
        ax = fig.add_axes(frac(x, yt, w, h))
        _place_image(ax, images[i], extents[i])
        if extents[i] is not None:
            _outline(ax, images[i], extents[i], st)
        if ground is not None and extents[i] is not None and names[i] in ("side", "front", "back"):
            _road(ax, st, extents[i], float(ground))
        if len(images) > 1:
            fig.text(x / fig_w, 1.0 - (yt - 0.05) / fig_h, titles[i], ha="left", va="bottom", fontsize=7.5,
                     color=st.muted)
    if cbar is not None and color is not None:
        kind, x, yt, w, h = cbar
        cax = fig.add_axes(frac(x, yt, w, h))
        _colorbar_from_spec(fig, cax, color, "vertical" if kind == "v" else "horizontal")
    return fig


# ---------------------------------------------------------------------------
# 3D: force histories
# ---------------------------------------------------------------------------

@_builder((6.7, 5.4))
def fig_history3d(history: Mapping[str, np.ndarray], t_avg_start: float | None = None, stats: Any = None, *,
                  st: FigureStyle, figsize: tuple[float, float], symmetry_plane: bool = False,
                  title: str | None = None) -> Figure:
    """The six force and moment coefficient histories of a 3D run (forces left, moments right).

    Same conventions as :func:`fig_history`: the shaded band is the averaging window, the black
    line the reported mean with its ±1 standard-error band. With a symmetry plane the side
    force, rolling and yawing moments are zero by construction, so only drag, lift and pitching
    moment are drawn (one column) and the figure says why, instead of plotting flat lines that
    look like results.
    """
    t = np.asarray(history.get("t", []), float)
    zero_lat = symmetry_plane and all(not np.any(np.abs(np.asarray(history.get(k, []), float)) > 0)
                                      for k in ("cs", "c_roll", "cn"))
    if zero_lat:
        keys: tuple = (("cd",), ("cl",), ("cm",))
        fig = _new_figure(st, (figsize[0], min(figsize[1], 4.6)))
        axs = np.asarray(fig.subplots(3, 1, sharex=True)).reshape(3, 1)
    else:
        keys = (("cd", "c_roll"), ("cs", "cm"), ("cl", "cn"))
        fig = _new_figure(st, figsize)
        axs = fig.subplots(3, 2, sharex=True)
    t_end = float(t[-1]) if t.size else 1.0
    win = (t >= t_avg_start) if (t_avg_start is not None and math.isfinite(t_avg_start) and t.size) else None
    colours = {"cd": 0, "cs": 4, "cl": 1, "c_roll": 5, "cm": 3, "cn": 2}
    for r, row in enumerate(keys):
        for c, key in enumerate(row):
            ax = axs[r, c]
            sym, name = COEFF_LABELS_3D[key]
            ax.set_ylabel(f"{sym}", rotation=0, ha="right", va="center", labelpad=6)
            ax.set_title(name, fontsize=8, loc="left", pad=3, fontweight="normal", color=st.muted)
            y = np.asarray(history.get(key, []), float)
            zero = y.size == t.size and t.size and not np.any(np.abs(y) > 0)
            if t.size == 0 or y.size != t.size:
                _no_data(ax, st, "not recorded")
                continue
            if zero and symmetry_plane and key in ("cs", "c_roll", "cn"):
                ax.text(0.5, 0.5, "zero by symmetry (half-domain run)", transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color=st.muted)
                ax.set_yticks([])
                ax.grid(False)
                continue
            td, yd = _decimate(t, y)
            ax.plot(td, yd, color=st.series[colours[key]], lw=0.8, zorder=3)
            if win is not None and win.any():
                ax.axvspan(t_avg_start, t_end, color=st.accent, alpha=0.07, lw=0, zorder=0)
                ax.axvline(t_avg_start, color=st.accent, lw=0.7, ls=(0, (3, 2)), zorder=2)
            s = _stat_of(stats, key)
            if s is not None and math.isfinite(s.mean):
                xa = t_avg_start if (t_avg_start is not None and math.isfinite(t_avg_start)) else float(t[0])
                if math.isfinite(s.stderr) and s.stderr > 0:
                    ax.fill_between([xa, t_end], s.mean - s.stderr, s.mean + s.stderr, color=st.text,
                                    alpha=0.18, lw=0, zorder=4)
                ax.plot([xa, t_end], [s.mean, s.mean], color=st.text, lw=0.9, zorder=5)
                ax.text(0.995, 0.92, f"mean {_fmt_pm(s.mean, s.stderr)}", transform=ax.transAxes, ha="right",
                        va="top", fontsize=6.8, color=st.text,
                        bbox={"boxstyle": "round,pad=0.2", "fc": st.bg, "ec": st.grid, "lw": 0.5})
            core = y[win] if win is not None and win.any() else y[y.size // 2:]
            ax.set_ylim(*_robust_limits(core, y[y.size // 10:]))
            ax.tick_params(labelsize=6.8)
    for ax in axs[-1]:
        ax.set_xlabel("Time t·U/L [convective units]")
        ax.set_xlim(float(t[0]) if t.size else 0.0, t_end)
    if zero_lat:
        fig.supxlabel("Side force, rolling and yawing moment are zero by construction (half-domain run with a "
                      "symmetry plane).", x=0.02, ha="left", fontsize=7, color=st.muted)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


# ---------------------------------------------------------------------------
# finite wings
# ---------------------------------------------------------------------------

def _wing_table(w: Any) -> dict[str, np.ndarray]:
    tb = w.table()
    o = np.argsort(tb["alpha"])
    return {k: np.asarray(v, float)[o] for k, v in tb.items()}


@_builder((6.7, 5.4))
def fig_wing_polar(wings: Any, *, st: FigureStyle, figsize: tuple[float, float],
                   labels: Sequence[str] | None = None, title: str | None = None) -> Figure:
    """Finite-wing polars: CL-α, drag polar CL-CD (with the induced part), Cm-α and L/D-α.

    ``wings`` is a :class:`~aerolume.model.WingResult` or a list of them (overlaid, one colour
    each). In the drag polar of a single viscous sweep the induced drag CDi is drawn dashed
    next to the total CD, so the profile-drag share reads directly as the gap between the two.
    Only converged points are plotted; the legend counts the others. The best-L/D point is
    ringed in every panel, as in the airfoil polar.
    """
    wl = [wings] if hasattr(wings, "points") else list(wings)
    if labels is None:
        labels = [f"{getattr(w, 'method', '').upper()} {'viscous' if getattr(w, 'viscous', False) else 'inviscid'}"
                  for w in wl]
    fig = _new_figure(st, figsize)
    axs = fig.subplots(2, 2)
    a_cl, a_dp, a_cm, a_ld = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    any_data = False
    for k, (w, lab) in enumerate(zip(wl, labels, strict=False)):
        tb = _wing_table(w)
        if tb["alpha"].size == 0:
            continue
        any_data = True
        a, cl, cd, cdi, cm = tb["alpha"], tb["cl"], tb["cd"], tb["cdi"], tb["cm"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(cd > 0, cl / cd, np.nan)
        viscous = bool(getattr(w, "viscous", True))
        n_bad = sum(1 for p in w.points if not p.converged)
        c = st.series[k % len(st.series)]
        kw = {"color": c, "lw": 1.3, "marker": "o", "ms": 2.8, "mew": 0}
        a_cl.plot(a, cl, label=lab + (f" ({n_bad} not converged)" if n_bad else ""), **kw)
        a_dp.plot(cd, cl, **kw)
        if viscous and len(wl) == 1 and np.isfinite(cdi).any():
            a_dp.plot(cdi, cl, color=c, lw=1.0, ls=(0, (3, 2)), label="induced CDi")
            a_dp.plot([], [], color=c, lw=1.3, label="total CD")
            a_dp.legend(loc="lower right")
        a_cm.plot(a, cm, **kw)
        if viscous:
            a_ld.plot(a, ld, **kw)
            if np.isfinite(ld).any():
                i = int(np.nanargmax(ld))
                ring = {"marker": "o", "ms": 7, "mfc": "none", "mec": c, "mew": 1.1, "zorder": 6, "ls": "none"}
                for ax_, xx, yy in ((a_cl, a[i], cl[i]), (a_dp, cd[i], cl[i]), (a_cm, a[i], cm[i]),
                                    (a_ld, a[i], ld[i])):
                    ax_.plot(xx, yy, **ring)
                if len(wl) == 1:
                    a_ld.text(0.98, 0.04, f"(L/D)max = {ld[i]:.1f} at α = {a[i]:g}°", transform=a_ld.transAxes,
                              ha="right", va="bottom", fontsize=7, color=st.text,
                              bbox={"boxstyle": "round,pad=0.25", "fc": st.bg, "ec": st.grid, "lw": 0.5})
        elif len(wl) == 1:
            _no_data(a_ld, st, "Inviscid sweep: no profile drag, L/D not meaningful")
    for ax in (a_cl, a_cm, a_ld):
        ax.axvline(0, color=st.spine, lw=0.5)
        ax.set_xlabel("Angle of attack α [°]")
    a_cl.axhline(0, color=st.spine, lw=0.5)
    a_cm.axhline(0, color=st.spine, lw=0.5)
    a_cl.set_ylabel("Lift coefficient CL")
    a_dp.set_xlabel("Drag coefficient CD")
    a_dp.set_ylabel("Lift coefficient CL")
    a_cm.set_ylabel("Pitching moment Cm")
    a_ld.set_ylabel("Lift-to-drag ratio L/D")
    a_cl.set_title("Lift curve")
    a_dp.set_title("Drag polar")
    a_cm.set_title("Pitching moment (about the reference point)")
    a_ld.set_title("Efficiency")
    if not any_data:
        for ax in axs.flat:
            _no_data(ax, st, "No converged points")
    elif len(wl) > 1 or any(not q.converged for w in wl for q in w.points):
        handles, labs = a_cl.get_legend_handles_labels()
        fig.legend(handles, labs, loc="outside upper center", ncols=min(len(labs), 3))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


def _pick_alphas(points: Sequence[Any], want: int = 3) -> list[Any]:
    """Up to ``want`` converged points with span data, spread over the positive-lift range."""
    pts = [p for p in points if p.converged and p.y is not None and p.c_cl is not None and np.size(p.y) > 2]
    pos = [p for p in pts if math.isfinite(p.cl) and p.cl > 0.05] or pts
    if len(pos) <= want:
        return sorted(pos, key=lambda p: p.alpha)
    pos = sorted(pos, key=lambda p: p.alpha)
    idx = np.unique(np.round(np.linspace(0, len(pos) - 1, want)).astype(int))
    return [pos[i] for i in idx]


@_builder((6.7, 3.1))
def fig_span_loading(wing: Any, *, st: FigureStyle, figsize: tuple[float, float],
                     alphas: Sequence[float] | None = None, title: str | None = None) -> Figure:
    """Span loading c·cl/c̄ against 2y/b at a few angles, with the elliptic loading of the same CL,
    and the section lift coefficient cl across the span.

    The elliptic distribution (c·cl/c̄ = 4/π · CL · √(1 − η²)) is the minimum-induced-drag
    loading of a planar wing; how far the computed curve departs from it is what the span
    efficiency e summarises. The section-cl panel shows where a wing will stall first (the
    station whose cl approaches the airfoil's cl,max), which the integrated CL hides.
    """
    pts = list(wing.points)
    if alphas is not None:
        sel = []
        for a in alphas:
            cand = [p for p in pts if p.converged and p.y is not None]
            if cand:
                p = min(cand, key=lambda q: abs(q.alpha - a))
                if all(p is not s for s in sel):
                    sel.append(p)
    else:
        sel = _pick_alphas(pts)
    fig = _new_figure(st, figsize)
    a_l, a_c = fig.subplots(1, 2)
    if not sel:
        _no_data(a_l, st, "No span distributions stored")
        _no_data(a_c, st, "")
        return fig
    span = float(getattr(wing, "span", float("nan")))
    for k, p in enumerate(sel):
        y = np.asarray(p.y, float)
        b = span if math.isfinite(span) and span > 0 else 2.0 * float(np.max(np.abs(y)))
        eta = 2.0 * y / b
        o = np.argsort(eta)
        c = st.series[k % len(st.series)]
        a_l.plot(eta[o], np.asarray(p.c_cl, float)[o], color=c, lw=1.3, label=f"α = {p.alpha:g}°, CL = {p.cl:.3f}")
        e = np.linspace(-1, 1, 201)
        a_l.plot(e, 4.0 / math.pi * p.cl * np.sqrt(np.clip(1 - e * e, 0, None)), color=c, lw=0.8,
                 ls=(0, (1.5, 1.5)))
        if p.cl_local is not None:
            a_c.plot(eta[o], np.asarray(p.cl_local, float)[o], color=c, lw=1.3)
    a_l.plot([], [], color=st.muted, lw=0.8, ls=(0, (1.5, 1.5)), label="elliptic, same CL")
    handles, labs = a_l.get_legend_handles_labels()
    # below the panels: the loadings fill both plots from tip to tip
    fig.legend(handles, labs, loc="outside lower center", ncols=min(len(labs), 4), fontsize=6.8)
    a_l.set_xlabel("Span station 2y/b")
    a_l.set_ylabel("Span loading c·cl / c̄")
    a_l.set_title("Span loading")
    a_l.set_xlim(-1.02, 1.02)
    a_l.set_ylim(bottom=min(0.0, a_l.get_ylim()[0]))
    a_c.set_xlabel("Span station 2y/b")
    a_c.set_ylabel("Section lift coefficient cl")
    a_c.set_title("Section lift")
    a_c.set_xlim(-1.02, 1.02)
    a_c.set_ylim(bottom=min(0.0, a_c.get_ylim()[0]))
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


@_builder(None)
def fig_wing_planform(wing: Any, *, st: FigureStyle, figsize: tuple[float, float] | None,
                      metrics: Mapping[str, float] | None = None, title: str | None = None,
                      width: float = 6.7) -> Figure:
    """Planform (top view) of a :class:`~aerolume.model.WingSpec`.

    Drawn as XFLR5 and most aircraft texts do: span horizontal, the leading edge at the top,
    the free stream coming from the top of the page. The quarter-chord line, the mean
    aerodynamic chord (at its spanwise station, both halves) and the section stations are
    marked, with span and root / tip chords dimensioned. ``metrics`` are
    ``geometry3d.planform`` values (MAC, its station and leading edge).
    """
    secs = sorted(wing.sections, key=lambda s: s.y)
    y = np.array([s.y for s in secs], float)
    c = np.array([s.chord for s in secs], float)
    xle = np.array([s.x_le for s in secs], float)
    if wing.symmetric:
        ys = np.concatenate([-y[::-1], y]) if y[0] > 0 else np.concatenate([-y[:0:-1], y])
        cs = np.concatenate([c[::-1], c]) if y[0] > 0 else np.concatenate([c[:0:-1], c])
        xs = np.concatenate([xle[::-1], xle]) if y[0] > 0 else np.concatenate([xle[:0:-1], xle])
    else:
        ys, cs, xs = y, c, xle
    # page frame: horizontal = span y, vertical = -x (leading edge on top, flow from the top)
    le = np.column_stack([ys, -xs])
    te = np.column_stack([ys, -(xs + cs)])
    outline = np.vstack([le, te[::-1]])
    b = float(np.ptp(ys)) or 1.0
    depth = float(np.ptp(np.concatenate([xs, xs + cs]))) or 1.0
    if figsize is None:
        figsize = (width, _fit_height(width, b * 1.08, depth * 1.9, side_in=0.5, extra_in=0.75, lo=1.6, hi=3.6))
    fig = _new_figure(st, figsize)
    ax = fig.add_subplot()
    ax.add_patch(MplPolygon(outline, closed=True, facecolor=st.body_fill, edgecolor=st.body_edge, lw=0.9,
                            zorder=3))
    for yy, xx, cc in zip(ys, xs, cs, strict=True):           # section stations
        ax.plot([yy, yy], [-xx, -(xx + cc)], color=st.body_edge, lw=0.5, zorder=4, alpha=0.6)
    ax.plot(ys, -(xs + 0.25 * cs), color=st.accent, lw=0.8, ls=(0, (5, 2, 1, 2)), zorder=5,
            label="quarter-chord line")
    if metrics:
        mac, mx, my = metrics.get("mac"), metrics.get("mac_le_x"), metrics.get("mac_y")
        if all(v is not None and math.isfinite(v) for v in (mac, mx, my)):
            for sgn in ((-1, 1) if wing.symmetric else (1,)):
                ax.plot([sgn * my, sgn * my], [-mx, -(mx + mac)], color=st.series[1], lw=1.6, zorder=6,
                        solid_capstyle="butt", label="mean aerodynamic chord" if sgn == 1 else None)
            ax.plot([0], [-(mx + 0.25 * mac)], marker="+", color=st.accent, ms=8, mew=1.2, zorder=7)
    unit = 0.12 * depth
    yd = -(np.max(xs + cs)) - 0.55 * depth
    ax.annotate("", xy=(ys.min(), yd), xytext=(ys.max(), yd),
                arrowprops={"arrowstyle": "<->", "color": st.muted, "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
    ax.text(0.0 if wing.symmetric else 0.5 * (ys.min() + ys.max()), yd - unit, f"b = {_fmt_len(b)}",
            ha="center", va="top", color=st.muted, fontsize=7.5)
    ax.text(ys.max() + 0.015 * b, -(xs[-1] + 0.5 * cs[-1]), f"c tip = {_fmt_len(cs[-1])}", ha="left",
            va="center", color=st.muted, fontsize=7)
    root_i = int(np.argmin(np.abs(ys)))
    ax.text(ys[root_i] + 0.01 * b, -(xs[root_i] + cs[root_i]) - 0.08 * depth, f"c root = {_fmt_len(cs[root_i])}",
            ha="left", va="top", color=st.muted, fontsize=7)
    ya = -np.min(xs) + 0.45 * depth
    ax.annotate("", xy=(ys.min() + 0.04 * b, ya - 0.35 * depth), xytext=(ys.min() + 0.04 * b, ya),
                arrowprops={"arrowstyle": "-|>", "color": st.accent, "lw": 1.1, "mutation_scale": 8})
    ax.text(ys.min() + 0.055 * b, ya - 0.17 * depth, "U∞", color=st.accent, fontsize=8, va="center")
    ax.set_xlim(ys.min() - 0.03 * b, ys.max() + 0.13 * b)
    ax.set_ylim(yd - 2.2 * unit, ya + 0.1 * depth)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Span y [m]")
    ax.set_ylabel("−x [m]")
    ax.grid(True, lw=0.4)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title if title is not None else "Planform")
    return fig


def with_style(st: FigureStyle, **changes: Any) -> FigureStyle:
    """A copy of a style with some tokens replaced (e.g. a brand accent for a report)."""
    return replace(st, **changes)


# ---------------------------------------------------------------------------
# research figures: grid convergence, smoke (DESIGN.md section 10)
# ---------------------------------------------------------------------------

@_builder(None)
def fig_grid_convergence(gcis: Any, *, st: FigureStyle, figsize: tuple[float, float] | None,
                         title: str | None = None, max_panels: int = 3) -> Figure:
    """Grid-convergence plot: each quantity against the cell size, with the Richardson curve.

    ``gcis`` is an ``analysis.gci.GridStudy``, a mapping of ``GCIResult`` or a sequence of
    them (at most ``max_panels`` drawn). Per panel: the values on every grid with ±2
    standard-error bars, the fitted curve φ = φ_ext + (φ₁ − φ_ext)(h/h₁)^p through the finest
    two grids when the convergence is monotone, the extrapolated value at h = 0 (ringed) and
    the GCI band on the finest grid (bracket). The x axis is h / h₁, so 1 is the finest grid
    and 0 the grid-converged limit.
    """
    items = list(getattr(gcis, "quantities", gcis).values()) if isinstance(
        getattr(gcis, "quantities", gcis), Mapping) else list(gcis)
    # a quantity that does not change with the grid (the lift of a symmetric body) has nothing to show
    items = [g for g in items if getattr(g, "levels", None) and g.convergence != "insensitive"][:max_panels]
    n = max(len(items), 1)
    if figsize is None:
        figsize = (6.7, 2.55) if n > 1 else (4.6, 3.0)
    fig = _new_figure(st, figsize)
    axes = np.atleast_1d(fig.subplots(1, n))
    if not items:
        _no_data(axes[0], st, "No grid study")
        return fig
    for ax, g in zip(axes, items, strict=False):
        lv = g.levels
        h1 = lv[0].h
        x = np.array([v.h / h1 for v in lv])
        y = np.array([v.value for v in lv])
        e = np.array([2.0 * v.stderr if math.isfinite(v.stderr) else 0.0 for v in lv])
        ax.errorbar(x, y, yerr=np.where(e > 0, e, np.nan), fmt="o", color=st.series[0], ms=4.5, capsize=2.5,
                    elinewidth=0.8, zorder=4, label="grids (±2 s.e.)")
        ok_curve = (g.convergence == "monotone" and math.isfinite(g.p_used) and math.isfinite(g.phi_ext))
        if ok_curve:
            xs = np.linspace(0.0, float(x.max()) * 1.05, 120)
            ax.plot(xs, g.phi_ext + (lv[0].value - g.phi_ext) * xs ** g.p_used, color=st.series[1], lw=1.1,
                    zorder=3, label=f"Richardson, p = {g.p_used:.2f}")
            ax.plot([0.0], [g.phi_ext], marker="o", mfc=st.bg, mec=st.series[1], mew=1.2, ms=6, zorder=5,
                    ls="none", label="extrapolated")
        if math.isfinite(g.u_num) and g.u_num > 0:
            ax.errorbar([1.0], [lv[0].value], yerr=[[g.u_num], [g.u_num]], fmt="none", ecolor=st.text, capsize=4,
                        elinewidth=1.2, zorder=6, label="GCI band")
        ax.set_xlim(-0.05 * float(x.max()), float(x.max()) * 1.08)
        ax.set_xlabel("Cell size h / h₁")
        ax.set_ylabel(g.quantity or "value")
        ttl = g.quantity or ""
        if math.isfinite(g.gci_fine):
            ttl += f"  GCI {100 * g.gci_fine:.2g} %"
        elif g.convergence:
            ttl += f"  ({g.convergence})"
        ax.set_title(ttl, fontsize=8.5)
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=6.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


@_builder((6.7, 4.2))
def fig_diagnostics(table: Mapping[str, Any], *, st: FigureStyle, figsize: tuple[float, float],
                    title: str | None = None) -> Figure:
    """The solver's health over a run (DESIGN.md 10.4): divergence, per-step change, CFL and Δt.

    ``table`` holds the diagnostics columns (``t``, ``div_rms``, ``div_max``, ``du_rel``,
    ``dv_rel``, ``dw_rel``, ``dp_rel``, ``dtheta_rel``, ``cfl``, ``dt``; any subset). Top:
    divergence left by the projection (log scale; round-off level means the projection is
    exact). Middle: the relative change of each field per step, the convergence signal of a
    steady run. Bottom: CFL number and time step.
    """
    t = np.asarray(table.get("t", []), float)
    fig = _new_figure(st, figsize)
    if t.size == 0:
        _no_data(fig.add_subplot(), st, "No diagnostics recorded")
        return fig
    axes = fig.subplots(3, 1, sharex=True)

    def series(ax: Any, keys: Sequence[tuple[str, str]], log: bool) -> None:
        drawn = False
        for i, (k, lab) in enumerate(keys):
            if k not in table:
                continue
            y = np.asarray(table[k], float)
            if y.size != t.size:
                continue
            if log:
                y = np.where(y > 0, y, np.nan)
            ax.plot(t, y, color=st.series[i % len(st.series)], lw=0.9, label=lab)
            drawn = True
        if log and drawn:
            ax.set_yscale("log")
        if drawn and len(keys) > 1:
            ax.legend(loc="upper right", fontsize=6.5, ncol=min(len(keys), 5))

    series(axes[0], [("div_rms", "rms"), ("div_max", "max")], True)
    axes[0].set_ylabel("|div u| [U/L]")
    series(axes[1], [("du_rel", "u"), ("dv_rel", "v"), ("dw_rel", "w"), ("dp_rel", "p"), ("dtheta_rel", "θ")], True)
    axes[1].set_ylabel("Relative change")
    series(axes[2], [("cfl", "CFL")], False)
    axes[2].set_ylabel("CFL")
    if "dt" in table and np.asarray(table["dt"]).size == t.size:
        ax2 = axes[2].twinx()
        ax2.plot(t, np.asarray(table["dt"], float), color=st.muted, lw=0.8, ls=(0, (3, 2)))
        ax2.set_ylabel("Δt [L/U] (dashed)", color=st.muted)
        ax2.grid(False)
    axes[-1].set_xlabel("Time t·U/L [convective units]")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=st.text)
    return fig


#: Background of smoke pictures: flow visualisation is read like a wind-tunnel photograph,
#: light smoke on a dark field, in print and on screen alike.
SMOKE_BACKGROUND = "#101418"


def _rgb(color: str) -> np.ndarray:
    from matplotlib.colors import to_rgb
    try:
        return np.array(to_rgb(color), float)
    except ValueError:
        return np.array([0.96, 0.96, 0.96])


@_builder(None)
def fig_smoke(fields: FieldData, channels: Sequence[tuple[str, str]] | None = None, bodies: Any = None, *,
              st: FigureStyle, figsize: tuple[float, float] | None, view: str | Sequence[float] = "auto",
              ground: bool = False, title: str | None = None, labels: Sequence[str] | None = None) -> Figure:
    """Smoke picture of one or more dye channels, composited in their emitters' colours.

    ``channels`` are ``(field name, colour)`` pairs, e.g. ``[("dye", "#f5f5f5"), ("dye1",
    "#e98a2e")]``; by default every stored "dye*" field in white. Each channel is laid over
    the dark :data:`SMOKE_BACKGROUND` with its concentration (clipped to 0..1) as opacity,
    in order, so a later channel shows on top. ``labels`` name the channels in the legend
    (the emitters; default the field names). Solid cells take the colour of the nearest
    fluid cell under the drawn outline, so the grid's staircase leaves no fringe.
    """
    names = [k for k in fields.fields if k == "dye" or re.match(r"^dye\d$", k)]
    chans = list(channels) if channels else [(k, "#f5f5f5") for k in names]
    chans = [(k, c) for k, c in chans if k in fields.fields]
    outs = _outlines(bodies)
    if not chans:
        fig = _new_figure(st, figsize or (6.7, 3.0))
        _no_data(fig.add_subplot(), st, "No smoke recorded")
        return fig
    first = np.asarray(fields.fields[chans[0][0]], float)
    ny, nx = first.shape
    ext = tuple(float(v) for v in fields.extent)
    vw = (ext if (view == "full" or not outs) else _auto_view(outs, ext, ground)) if isinstance(view, str) \
        else tuple(float(v) for v in view)
    xs = ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx
    ys = ext[2] + (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny
    ix = np.flatnonzero((xs >= vw[0]) & (xs <= vw[1]))
    iy = np.flatnonzero((ys >= vw[2]) & (ys <= vw[3]))
    if ix.size < 2 or iy.size < 2:
        ix, iy = np.arange(nx), np.arange(ny)
    img = np.empty((iy.size, ix.size, 3))
    img[:] = _rgb(SMOKE_BACKGROUND)
    for name, color in chans:
        a = np.clip(np.nan_to_num(np.asarray(fields.fields[name], float)[np.ix_(iy, ix)]), 0.0, 1.0)[..., None]
        img = img * (1.0 - a) + _rgb(color) * a
    if fields.mask is not None:
        solid = np.asarray(fields.mask)[np.ix_(iy, ix)] > 0.5
        filled = False
        if outs and solid.any() and not solid.all():
            try:
                from scipy.ndimage import distance_transform_edt
                _, (ii, jj) = distance_transform_edt(solid, return_indices=True)
                img = img[ii, jj]
                filled = True
            except Exception:  # noqa: BLE001 - scipy missing: body colour instead
                log.debug("scipy unavailable; solid cells painted in the body colour")
        if not filled:
            img[solid] = _rgb(st.body_fill)
    dx = (ext[1] - ext[0]) / nx
    dy = (ext[3] - ext[2]) / ny
    sub_ext = (xs[ix[0]] - dx / 2, xs[ix[-1]] + dx / 2, ys[iy[0]] - dy / 2, ys[iy[-1]] + dy / 2)
    if figsize is None:
        figsize = (6.7, _fit_height(6.7, sub_ext[1] - sub_ext[0], sub_ext[3] - sub_ext[2], side_in=0.6,
                                    extra_in=0.75, lo=2.0, hi=6.0))
    fig = _new_figure(st, figsize)
    ax = fig.add_subplot()
    ax.imshow(img, origin="lower", extent=sub_ext, interpolation="bilinear", aspect="equal", zorder=1)
    if outs:
        _draw_bodies(ax, outs, st, lw=0.8, zorder=4)
    if ground:
        ax.axhline(0.0, color=st.ground, lw=1.4, zorder=5)
    ax.set_xlim(sub_ext[0], sub_ext[1])
    ax.set_ylim(sub_ext[2], sub_ext[3])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(False)
    if len(chans) > 1:
        handles = [Rectangle((0, 0), 1, 1, color=c) for _, c in chans]
        names_ = list(labels) if labels is not None and len(labels) == len(chans) else [n for n, _ in chans]
        ax.legend(handles, names_, loc="upper right", fontsize=6.5, frameon=True,
                  facecolor=SMOKE_BACKGROUND, labelcolor="#e6e9ed", edgecolor="#5b6470")
    prefix = {"mean": "Mean", "instant": "Instantaneous"}.get(fields.label, "")
    ax.set_title(title if title is not None else f"{prefix} smoke".strip().capitalize())
    return fig
