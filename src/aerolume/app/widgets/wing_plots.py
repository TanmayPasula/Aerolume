"""Plots of finite-wing analyses: polars, span loading and the live view of a wing run.

A wing sweep (:class:`~aerolume.model.WingResult`) is read the way XFLR5 users
read one, so the views follow that lineage:

* :class:`WingPolarGrid` -- the four standard polars in a 2 x 2 grid: CL-α,
  the drag polar CL-CD, Cm-α and L/D-α. For a single wing the drag polar also
  splits CD into its induced part CDi (dashed) and profile part CDv (dotted):
  that split is the first thing a designer reads off a wing polar (is the drag
  from lift or from the sections?). Several wings overlay in their series
  colours (Compare page) and only the total CD is drawn then. Capital symbols
  (C<sub>L</sub>, C<sub>D</sub>) mark wing coefficients, as opposed to the
  lower-case section coefficients of an airfoil polar.
* :class:`SpanLoadingPlots` -- three stacked panels sharing the span axis: the
  span loading c·cl/c̄ with the elliptic loading of the same CL as a dashed
  reference (the ideal for induced drag; the gap between the two curves *is*
  the span efficiency), the local section cl (where the wing stalls first) and
  the induced angle αi.
* :class:`WingRunView` -- the Simulate page's viewport during a wing run. A cold
  viscous sweep spends ~90 % of its time computing 2D section polars, so the
  view lists those polars as they are computed and plots their cl(α) curves
  live; the wing's own CL-α (a few hundred milliseconds of lattice solves at the
  end) joins them, drawn in the accent colour, when the result arrives.

Inviscid sweeps have no profile drag: L/D grows without bound towards CL = 0,
so the L/D panel stays empty with a note instead of plotting a meaningless
curve (``analysis.wing_metrics`` reports the same quantities as NaN).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QSplitter, QVBoxLayout, QWidget

from .kpi import DataTable, fmt_fixed, fmt_re, label
from .plots import LinePlot, qcolor, series_color, tokens

__all__ = ["SpanLoadingPlots", "WingPolarGrid", "WingRunView", "elliptic_loading", "nearest_point",
           "span_order", "wing_table"]

CL, CD, CM = "C<sub>L</sub>", "C<sub>D</sub>", "C<sub>m</sub>"


# ---------------------------------------------------------------------------
# data helpers (pure, tested directly)
# ---------------------------------------------------------------------------

def wing_table(result) -> dict[str, np.ndarray]:
    """Converged points of a WingResult sorted by angle: alpha, cl, cd, cdi, cdv, cm, e, ld.

    ``ld`` is NaN where CD is not positive and for inviscid sweeps (see the
    module docstring).
    """
    tab = result.table()
    a = np.asarray(tab.get("alpha", []), dtype=float)
    order = np.argsort(a, kind="stable")
    out = {k: np.asarray(v, dtype=float)[order] for k, v in tab.items()}
    cd = out.get("cd", np.zeros(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ld = np.where(cd > 0, out.get("cl", np.zeros(0)) / cd, np.nan)
    out["ld"] = ld if getattr(result, "viscous", True) else np.full(ld.shape, np.nan)
    return out


def elliptic_loading(y: np.ndarray, span: float, cl: float) -> np.ndarray:
    """c·cl/c̄ of the elliptic span loading with lift coefficient ``cl``.

    Γ(y) ∝ sqrt(1 - (2y/b)²); normalised so its span average equals CL (the
    average of the square root over the span is π/4), which is how
    ``WingPoint.c_cl`` is normalised too: (4 CL / π) sqrt(1 - (2y/b)²).
    """
    y = np.asarray(y, dtype=float)
    if not (span > 0 and math.isfinite(span) and math.isfinite(cl)):
        return np.full(y.shape, np.nan)
    eta = np.clip(2.0 * y / span, -1.0, 1.0)
    return 4.0 * cl / math.pi * np.sqrt(1.0 - eta ** 2)


def nearest_point(result, alpha: float, *, with_span: bool = True):
    """The converged point closest to ``alpha`` (with span data when ``with_span``), or None."""
    pts = [p for p in getattr(result, "points", []) if p.converged and (not with_span or p.y is not None)]
    if not pts or alpha is None or not math.isfinite(float(alpha)):
        return pts[0] if pts else None
    return min(pts, key=lambda p: abs(p.alpha - float(alpha)))


def span_order(point) -> np.ndarray:
    """Index order of a point's strips by increasing y (strips may come per half wing)."""
    y = np.asarray(point.y if point is not None and point.y is not None else [], dtype=float)
    return np.argsort(y, kind="stable")


# ---------------------------------------------------------------------------
# polars
# ---------------------------------------------------------------------------

class WingPolarGrid(QWidget):
    """CL-α, CL-CD (with the CDi / CDv split for one wing), Cm-α and L/D-α of wing sweeps."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(1)
        self.cl_alpha = LinePlot(self, xlabel="α (°)", ylabel=CL)
        self.drag_polar = LinePlot(self, xlabel=CD, ylabel=CL)
        if self.drag_polar.legend is not None:
            # a drag polar climbs to the upper right, where the default legend would sit on its
            # high-lift end; the lower right stays empty unless a sweep goes far into negative lift
            self.drag_polar.legend.setOffset((-6, -6))
        self.cm_alpha = LinePlot(self, xlabel="α (°)", ylabel=f"{CM} (moment reference)")
        self.ld_alpha = LinePlot(self, xlabel="α (°)", ylabel="L/D")
        for i, w in enumerate(self.plots()):
            grid.addWidget(w, i // 2, i % 2)
        self._markers: list = []

    def plots(self) -> tuple[LinePlot, ...]:
        return (self.cl_alpha, self.drag_polar, self.cm_alpha, self.ld_alpha)

    def set_wings(self, wings: Sequence[tuple[str, object, int | str]], *, split: bool | None = None) -> None:
        """``wings`` = [(name, WingResult, colour), ...]; replaces every series.

        ``split`` draws CDi and CDv next to CD in the drag polar (default: when
        there is exactly one viscous wing).
        """
        for p in self.plots():
            p.clear_series()
        self._clear_markers()
        wings = list(wings)
        if split is None:
            split = len(wings) == 1 and bool(getattr(wings[0][1], "viscous", False))
        tk = tokens()
        any_viscous = False
        for name, res, color in wings:
            t = wing_table(res)
            a = t.get("alpha", np.zeros(0))
            if a.size == 0:
                continue
            many = len(wings) > 1
            self.cl_alpha.set_series(name, a, t["cl"], color=color, symbol="o")
            self.drag_polar.set_series(name if many else "CD (total)", t["cd"], t["cl"], color=color, symbol="o")
            if split:
                muted = qcolor(tk["text2"])
                self.drag_polar.set_series("CDi (induced)", t["cdi"], t["cl"], color=muted, width=1.2,
                                           style=Qt.PenStyle.DashLine)
                self.drag_polar.set_series("CDv (profile)", t["cdv"], t["cl"], color=muted, width=1.2,
                                           style=Qt.PenStyle.DotLine)
            self.cm_alpha.set_series(name, a, t["cm"], color=color, symbol="o")
            if getattr(res, "viscous", True):
                any_viscous = True
                self.ld_alpha.set_series(name, a, t["ld"], color=color, symbol="o")
        self.ld_alpha.setToolTip("" if any_viscous or not wings else
                                 "An inviscid sweep has no profile drag, so L/D is unbounded near CL = 0 "
                                 "and is not plotted.")
        for p in self.plots():
            p.autoRange()

    def mark_alpha(self, alpha: float | None) -> None:
        """A thin vertical line at the angle shown in the span-loading view (None removes it)."""
        import pyqtgraph as pg

        from .plots import make_pen
        self._clear_markers()
        if alpha is None or not math.isfinite(alpha):
            return
        tk = tokens()
        for p in (self.cl_alpha, self.cm_alpha, self.ld_alpha):
            line = pg.InfiniteLine(pos=float(alpha), angle=90, movable=False,
                                   pen=make_pen(qcolor(tk["muted"]), 1.0, Qt.PenStyle.DashLine))
            p.addItem(line, ignoreBounds=True)
            self._markers.append((p, line))

    def _clear_markers(self) -> None:
        for p, line in self._markers:
            try:
                p.removeItem(line)
            except RuntimeError:
                pass
        self._markers = []


# ---------------------------------------------------------------------------
# span loading
# ---------------------------------------------------------------------------

class SpanLoadingPlots(QWidget):
    """Span loading c·cl/c̄ (with the elliptic reference), local cl and induced angle against y."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        self.loading = LinePlot(self, xlabel="", ylabel="c·c<sub>l</sub> / c̄")
        self.local_cl = LinePlot(self, xlabel="", ylabel="Local c<sub>l</sub>")
        self.induced = LinePlot(self, xlabel="y [m]", ylabel="α<sub>i</sub> (°)")
        self._place_legend(True)
        for p in self.plots():
            lay.addWidget(p, 1)
            if p is not self.loading:
                p.setXLink(self.loading)
        self.loading.setToolTip("Span loading: local chord × section lift coefficient over the mean chord. "
                                "Its span average is CL; the dashed curve is the elliptic loading of the same "
                                "CL, the minimum-induced-drag ideal.")
        self.local_cl.setToolTip("Section lift coefficient along the span: where it first reaches the "
                                 "section's cl,max is where the wing starts to stall.")
        self.induced.setToolTip("Induced angle: how much the trailing vortices turn the local flow down.")

    def plots(self) -> tuple[LinePlot, ...]:
        return (self.loading, self.local_cl, self.induced)

    def _place_legend(self, lifting: bool) -> None:
        """Span loading peaks at mid-span and falls to zero at both tips, so the curves cross every
        corner of the plot; the one place they never reach is the middle edge on the far side of the
        peak: the bottom centre for positive lift, the top centre for negative lift."""
        lg = self.loading.legend
        if lg is None:
            return
        edge = 1.0 if lifting else 0.0
        lg.anchor(itemPos=(0.5, edge), parentPos=(0.5, edge), offset=(0, -6 if lifting else 6))

    def set_points(self, points: Sequence[tuple[str, object, float, int | str]], *, elliptic: bool = True) -> None:
        """``points`` = [(name, WingPoint, span [m], colour), ...]; replaces every series.

        The elliptic reference is drawn for the first point only (with several
        wings, one reference per wing would hide the data).
        """
        for p in self.plots():
            p.clear_series()
        tk = tokens()
        for k, (name, pt, span, color) in enumerate(points):
            if pt is None or pt.y is None:
                continue
            o = span_order(pt)
            y = np.asarray(pt.y, dtype=float)[o]
            if pt.c_cl is not None:
                self.loading.set_series(name, y, np.asarray(pt.c_cl, dtype=float)[o], color=color)
            if k == 0 and elliptic and pt.c_cl is not None and span and math.isfinite(pt.cl):
                yy = np.linspace(-span / 2, span / 2, 161)
                if np.nanmin(y) >= -1e-9:            # a half wing: reference over the same half
                    yy = yy[yy >= 0]
                self.loading.set_series("Elliptic (same CL)", yy, elliptic_loading(yy, span, pt.cl),
                                        color=qcolor(tk["muted"]), width=1.2, style=Qt.PenStyle.DashLine)
            if pt.cl_local is not None:
                self.local_cl.set_series(name, y, np.asarray(pt.cl_local, dtype=float)[o], color=color)
            if pt.alpha_i is not None:
                self.induced.set_series(name, y, np.asarray(pt.alpha_i, dtype=float)[o], color=color)
        first = next((pt for _n, pt, _s, _c in points if pt is not None and pt.c_cl is not None), None)
        if first is not None:
            self._place_legend(float(np.nanmean(np.asarray(first.c_cl, dtype=float))) >= 0.0)
        for p in self.plots():
            p.autoRange()


# ---------------------------------------------------------------------------
# the Simulate page's live view of a wing run
# ---------------------------------------------------------------------------

class WingRunView(QWidget):
    """Section polars being computed (table + live cl-α curves), then the wing's own polar.

    ``reset(total_angles)`` before a run; ``add_section_point(airfoil, re, alpha,
    cl, cd, converged)`` for every section-polar angle the solver reports;
    ``set_angles_done(n)`` while the wing itself is solved; ``show_result(name,
    result)`` when the sweep is done (or to show a stored one).
    """

    #: Section polars are context, the wing's CL is the result: sections are drawn in
    #: these muted series colours and thin lines, the wing in the accent colour.
    SECTION_WIDTH = 1.1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(6)
        self.table_title = label("Section polars", "h3")
        ll.addWidget(self.table_title)
        self.table_note = label("", "muted", wrap=True)
        ll.addWidget(self.table_note)
        self.table = DataTable(["Airfoil", "Re", "Angles", "Converged"])
        ll.addWidget(self.table, 1)
        left.setMinimumWidth(260)
        split.addWidget(left)
        plots = QWidget()
        pl = QGridLayout(plots)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(1)
        self.cl_alpha = LinePlot(plots, xlabel="α (°)", ylabel=f"{CL}, c<sub>l</sub>")
        self.drag_polar = LinePlot(plots, xlabel=f"{CD}, c<sub>d</sub>", ylabel=f"{CL}, c<sub>l</sub>")
        pl.addWidget(self.cl_alpha, 0, 0)
        pl.addWidget(self.drag_polar, 0, 1)
        split.addWidget(plots)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([320, 900])
        self._nodes: dict[tuple[str, float], dict[float, tuple[float, float, bool]]] = {}
        self._total = 0
        self._done = 0
        self._result_rows = False

    # -- state ------------------------------------------------------------
    def reset(self, total_angles: int = 0, *, viscous: bool = True) -> None:
        self._nodes = {}
        self._total = int(total_angles)
        self._done = 0
        self._result_rows = False
        for p in (self.cl_alpha, self.drag_polar):
            p.clear_series()
        self.table_title.setText("Section polars")
        self.table_note.setText("2D viscous polars of each section airfoil at its Reynolds numbers; they take "
                                "most of a cold run and are reused by later runs." if viscous else
                                "Inviscid sweep: no section polars are needed.")
        self.table.set_data(["Airfoil", "Re", "Angles", "Converged"], [])

    def nodes(self) -> dict[tuple[str, float], dict[float, tuple[float, float, bool]]]:
        return self._nodes

    def angles_done(self) -> int:
        return self._done

    def add_section_point(self, airfoil: str, re: float, alpha: float, cl: float, cd: float, ok: bool) -> None:
        if self._result_rows:
            return
        node = self._nodes.setdefault((str(airfoil), float(re)), {})
        node[float(alpha)] = (float(cl), float(cd), bool(ok))
        self._draw_sections()

    def set_angles_done(self, n: int) -> None:
        self._done = max(self._done, int(n))

    # -- drawing ----------------------------------------------------------
    def _node_name(self, key: tuple[str, float]) -> str:
        af, re = key
        return f"{af} · Re {fmt_re(re)}"

    def _draw_sections(self) -> None:
        rows = []
        for i, (key, pts) in enumerate(sorted(self._nodes.items())):
            ok = sorted((a, v[0], v[1]) for a, v in pts.items() if v[2])
            rows.append([key[0], fmt_re(key[1]), str(len(pts)), str(len(ok))])
            if ok:
                a = np.array([p[0] for p in ok])
                cl = np.array([p[1] for p in ok])
                cd = np.array([p[2] for p in ok])
                col = series_color(1 + i)
                self.cl_alpha.set_series(self._node_name(key), a, cl, color=col, width=self.SECTION_WIDTH,
                                         symbol="o")
                self.drag_polar.set_series(self._node_name(key), cd, cl, color=col, width=self.SECTION_WIDTH,
                                           symbol="o", legend=False)
        if not self._result_rows:
            self.table.set_data(["Airfoil", "Re", "Angles", "Converged"], rows, numeric=[2, 3])
        for p in (self.cl_alpha, self.drag_polar):
            p.autoRange()

    def show_result(self, name: str, result) -> None:
        """The wing's polar on top of the section curves; the table lists the wing's points."""
        t = wing_table(result)
        a = t.get("alpha", np.zeros(0))
        accent = qcolor(tokens()["accent"])
        if a.size:
            self.cl_alpha.set_series(f"Wing · {name}" if name else "Wing", a, t["cl"], color=accent, width=2.0,
                                     symbol="o")
            self.drag_polar.set_series(f"Wing · {name}" if name else "Wing", t["cd"], t["cl"], color=accent,
                                       width=2.0, symbol="o")
        rows = []
        for p in sorted(getattr(result, "points", []), key=lambda q: q.alpha):
            ld = p.cl / p.cd if p.converged and p.cd and p.cd > 0 and result.viscous else float("nan")
            rows.append([fmt_fixed(p.alpha, 2), fmt_fixed(p.cl, 4), fmt_fixed(p.cd, 5), fmt_fixed(ld, 1),
                         "yes" if p.converged else "no"])
        self._result_rows = True
        self.table_title.setText("Wing polar")
        self.table_note.setText("Coefficients on the reference area; the section polars it used are drawn "
                                "thin." if self._nodes else "Coefficients on the reference area.")
        self.table.set_data(["α (°)", "CL", "CD", "L/D", "Converged"], rows, numeric=[0, 1, 2, 3])
        for p in (self.cl_alpha, self.drag_polar):
            p.autoRange()
