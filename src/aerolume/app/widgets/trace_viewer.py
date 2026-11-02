"""Solver trace: one captured time step, stage by stage, side by side (DESIGN.md 10.4).

A projection step is a short pipeline -- advection and diffusion give a
tentative velocity u* that is not divergence-free; its divergence is the
source of the pressure Poisson equation; the pressure correction phi is
solved; the correction makes u divergence-free again. When a run misbehaves
the question is *which stage* went wrong, and the only way to see it is to
look at the intermediate fields. AeroJAX shows them; so does this viewer.

* :func:`normalize_trace` accepts what a solver sends (a ``TraceCapture``-like
  object with ``stages``, a dict ``{"stages": [...], "extent": ...}``, a plain
  ``{stage: array}`` or ``{stage: {field: array}}`` mapping) and returns a
  :class:`TraceCapture` of :class:`TraceStage` panels, so the producer is free
  in its form and the viewer stays simple.
* :class:`TraceViewer` shows every (stage, field) as a thumbnail, and the
  selected one enlarged beside them with its colour bar, scale and hover
  readout (click a thumbnail to enlarge it; the arrow keys step through them).
  A solver trace with Runge–Kutta stages is laid out as a table: one row per
  RK stage (and one for the end of the step), one column per quantity (u*
  after advection–diffusion, ∇·u*, φ, the corrected u ...), so a quantity is
  read down its column through the stages -- forty 45 x 35 px thumbnails in
  two rows, their labels repeating "1 u*" .. "6 ∇·u*" for every stage, could
  not be inspected. Panels of one quantity share one colour map and range
  ("Shared range", on by default; off scales each panel to its own values,
  for a divergence that falls by five orders of magnitude). Other traces keep
  a two-row strip. "Open in window" shows the same capture in a floating,
  resizable window; zoom and pan of the enlarged view apply to every
  thumbnail; a statistics table (min, max, RMS, L2) can sit under them.
  "Capture step" asks the running solver for a new trace; "Export NPZ…"
  saves every array.

Arrays are in solver orientation (row 0 = y min, like ``FieldData``) and in
the solver's units unless the producer says otherwise (``units``).
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

__all__ = ["STAGE_WORDS", "TraceCapture", "TraceStage", "TraceViewer", "TraceWindow", "column_title", "normalize_trace",
           "stage_group", "stage_label", "stage_stats", "table_layout"]


@dataclass
class TraceStage:
    """One panel: a field after one stage of the step."""
    stage: str                        # "Advection–diffusion", "Pressure solve" ...
    field: str                        # "u*", "div", "phi" ...
    data: np.ndarray                  # (ny, nx), solver orientation
    extent: tuple[float, float, float, float]
    units: str = ""
    note: str = ""
    group: str = ""                   # "RK 1" .. "RK 3", "End of step": the trace table's row ("" ungrouped)


@dataclass
class TraceCapture:
    """A captured step: its panels, the step number and time, and producer notes."""
    stages: list[TraceStage] = field(default_factory=list)
    step: int | None = None
    t: float | None = None
    notes: list[str] = field(default_factory=list)


def _get(obj, *names, default=None):
    for n in names:
        if isinstance(obj, Mapping) and n in obj:
            return obj[n]
        if not isinstance(obj, Mapping) and hasattr(obj, n):
            return getattr(obj, n)
    return default


def _arrays_of(entry) -> list[tuple[str, np.ndarray]]:
    """(field, array) pairs of one stage entry (an array, a {field: array} mapping or an object)."""
    if isinstance(entry, np.ndarray) or hasattr(entry, "cpu"):
        return [("", _to_numpy(entry))]
    fields = _get(entry, "fields")
    if isinstance(fields, Mapping):
        return [(str(k), _to_numpy(v)) for k, v in fields.items() if _ndim(v) == 2]
    data = _get(entry, "data", "array", "values")
    if data is not None:
        return [(str(_get(entry, "field", default="")), _to_numpy(data))]
    if isinstance(entry, Mapping):
        return [(str(k), _to_numpy(v)) for k, v in entry.items() if _ndim(v) == 2]
    return []


def _ndim(v) -> int:
    try:
        return int(np.ndim(v)) if not hasattr(v, "dim") else int(v.dim())
    except Exception:  # noqa: BLE001 - anything that is not an array is skipped
        return -1


def _to_numpy(v) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().float().cpu().numpy()
    return np.asarray(v, dtype=np.float32)


def normalize_trace(trace, *, extent: Sequence[float] | None = None) -> TraceCapture:
    """A :class:`TraceCapture` from any of the forms a solver may send (see the module docstring).

    ``extent`` (metres, (x0, x1, y0, y1)) is used when the trace gives none;
    without either, cell indices are used. Stages keep the producer's order.
    """
    if trace is None:
        return TraceCapture()
    if isinstance(trace, TraceCapture):
        return trace
    ext = _get(trace, "extent", default=None) or extent
    step = _get(trace, "step", default=None)
    t = _get(trace, "t", "time", default=None)
    notes = list(_get(trace, "notes", default=[]) or [])
    raw = _get(trace, "stages", default=None)
    entries: list[tuple[str, object]] = []
    if raw is None and isinstance(trace, Mapping):
        entries = [(str(k), v) for k, v in trace.items() if k not in ("extent", "step", "t", "time", "notes")]
    elif isinstance(raw, Mapping):
        entries = [(str(k), v) for k, v in raw.items()]
    elif raw is not None:
        for i, e in enumerate(raw):
            if isinstance(e, (tuple, list)) and len(e) == 2 and isinstance(e[0], str):
                entries.append((e[0], e[1]))
            else:
                entries.append((str(_get(e, "name", "stage", "label", default=f"Stage {i + 1}")), e))
    out = TraceCapture(step=int(step) if step is not None else None,
                       t=float(t) if t is not None else None, notes=notes)
    for name, e in entries:
        units = str(_get(e, "units", default="") or "") if not isinstance(e, np.ndarray) else ""
        note = str(_get(e, "description", "note", default="") or "") if not isinstance(e, np.ndarray) else ""
        e_ext = _get(e, "extent", default=None) if not isinstance(e, np.ndarray) else None
        for fname, arr in _arrays_of(e):
            if arr.ndim != 2:
                continue
            ex = e_ext or ext or (0.0, float(arr.shape[1]), 0.0, float(arr.shape[0]))
            stage, fld, group = name, fname, ""
            if not fname:                        # a bare solver key: its words (the key stays in the note)
                stage, fld, rk = stage_label(name)
                if (stage, fld) != (name, ""):
                    note = "; ".join(x for x in (note, rk, f"solver key {name}") if x)
                    group = stage_group(name)
            out.stages.append(TraceStage(stage, fld, arr, tuple(float(v) for v in ex), units, note, group))
    return out


#: What each stage of the projection step is called on screen, by the solver's key (``FlowSolver.trace_step``:
#: "stage3/explicit_u", "stage3/divergence", "final_p" ...); unknown keys keep their own name.
STAGE_WORDS = {
    "explicit": "Advection–diffusion",
    "penalised": "Solid penalisation",
    "divergence": "Divergence of u*",
    "pressure_potential": "Pressure solve",
    "correction": "Correction −∇φ",
    "projected": "Projection",
    "theta": "Temperature",
    "final": "End of step",
}
#: Field names of those keys: the tentative velocity before the projection is starred.
_FIELD_WORDS = {"u": "u", "v": "v", "w": "w", "p": "p", "theta": "θ"}
_STAGE_RE = re.compile(r"^(?:stage(\d+)/)?(explicit|penalised|divergence|pressure_potential|correction|projected|"
                       r"theta|final)(?:_(u|v|w|p|theta))?$")


def stage_label(key: str) -> tuple[str, str, str]:
    """(stage, field, note) of a solver trace key: ("Advection–diffusion", "u*", "Runge–Kutta stage 3")
    for "stage3/explicit_u"; a key the table does not know is its own stage name."""
    m = _STAGE_RE.match(str(key).strip())
    if m is None:
        return str(key), "", ""
    k, what, comp = m.group(1), m.group(2), m.group(3)
    stage = STAGE_WORDS[what]
    if what == "divergence":
        fld = "∇·u*"
    elif what == "pressure_potential":
        fld = "φ"
    elif what == "theta":
        fld = "θ"
    else:
        fld = _FIELD_WORDS.get(comp or "", comp or "")
        if what in ("explicit", "penalised") and fld:
            fld += "*"
    note = f"Runge–Kutta stage {k}" if k else ""
    return stage, fld, note


def stage_group(key: str) -> str:
    """The trace-table row of a solver key: "RK 3" for "stage3/explicit_u", "End of step" for "final_p",
    "" for a key the table does not know."""
    m = _STAGE_RE.match(str(key).strip())
    if m is None:
        return ""
    if m.group(1):
        return f"RK {m.group(1)}"
    return "End of step" if m.group(2) == "final" else ""


#: Short stage tags in the column headers (the full words are in the header's tooltip).
_STAGE_TAGS = {"Advection–diffusion": "adv.", "Solid penalisation": "pen.", "Correction −∇φ": "corr.",
               "Projection": "proj.", "End of step": "end"}


def column_title(stage: str, field: str) -> str:
    """A trace-table column header: the field with a short stage tag ("u* adv.", "∇·u*", "u proj.")."""
    tag = _STAGE_TAGS.get(stage, "")
    if not field:
        return stage
    return f"{field} {tag}".strip()


def table_layout(stages: Sequence[TraceStage]) -> tuple[list[str], list[tuple[str, str, int]], dict[int, tuple]]:
    """(rows, columns, cell of each panel) of the grouped trace table, or ([], [], {}) for a trace without
    groups. Rows are the groups in order of first appearance; a column is a (stage, field, n) quantity --
    ``n`` counts repeats of one quantity within a group, so nothing is overwritten. ``cell[i]`` is the
    (row, column) of panel i."""
    if not stages or not all(st.group for st in stages):
        return [], [], {}
    rows: list[str] = []
    cols: list[tuple[str, str, int]] = []
    cell: dict[int, tuple] = {}
    seen: dict[tuple[str, str, str], int] = {}
    ends = []
    for i, st in enumerate(stages):
        if st.group not in rows:
            rows.append(st.group)
        if st.group == "End of step":
            ends.append(i)                 # placed below, under the column of the same quantity
            continue
        n = seen.get((st.group, st.stage, st.field), 0)
        seen[(st.group, st.stage, st.field)] = n + 1
        key = (st.stage, st.field, n)
        if key not in cols:
            cols.append(key)
        cell[i] = (rows.index(st.group), cols.index(key))
    used: set[int] = set()
    for i in ends:
        st = stages[i]
        # the end of the step's u under the projected u (the last stage that has it), θ under θ ...
        bare = st.field.rstrip("*")
        match = [j for j, (_s, f, _n) in enumerate(cols) if f.rstrip("*") == bare and j not in used]
        if match:
            j = match[-1]
        else:
            cols.append((st.stage, st.field, 0))
            j = len(cols) - 1
        used.add(j)
        cell[i] = (rows.index(st.group), j)
    # the end-of-step row last, whatever order the producer used
    if "End of step" in rows and rows[-1] != "End of step":
        order = [r for r in rows if r != "End of step"] + ["End of step"]
        remap = {rows.index(r): k for k, r in enumerate(order)}
        cell = {i: (remap[r], c) for i, (r, c) in cell.items()}
        rows = order
    return rows, cols, cell


def stage_stats(data: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """min, max, RMS and L2 (root-sum-square times the cell count's square root, i.e. a discrete L2
    norm on unit cells) of the finite values -- the numbers a solver developer compares across stages."""
    a = np.asarray(data, dtype=np.float64)
    ok = np.isfinite(a)
    if mask is not None and np.shape(mask) == a.shape:
        ok &= np.asarray(mask) < 0.5
    v = a[ok]
    if v.size == 0:
        return {"min": math.nan, "max": math.nan, "rms": math.nan, "l2": math.nan, "n": 0}
    return {"min": float(v.min()), "max": float(v.max()), "rms": float(np.sqrt(np.mean(v * v))),
            "l2": float(np.sqrt(np.sum(v * v))), "n": int(v.size)}


def _base_name(name: str) -> str:
    """'u*' / 'u_star' / 'u^{n+1}' -> 'u'; 'div u*' -> 'div' (used to share ranges and pick maps)."""
    n = name.strip().lower()
    if "div" in n:
        return "divergence"
    n = re.sub(r"(\*|_star|\^.*|\(.*\)|_new|_old|_n1|_n)$", "", n)
    return {"phi": "phi", "φ": "phi", "pressure": "p", "p": "p", "theta": "temperature", "θ": "temperature",
            "∇·u": "divergence"}.get(n, n)


def _cmap_and_range(base: str, data: np.ndarray, shared: tuple[float, float] | None) -> tuple[str, float, float]:
    """A colour map and range for a trace panel: signed quantities centred on 0, others robust."""
    from .field_view import robust_range
    signed = base in ("divergence", "phi", "p", "v", "w") or base.startswith("d")
    if shared is not None:
        lo, hi = shared
    else:
        lo, hi = robust_range(data, diverging=signed, lo=0.5, hi=99.5)
    if signed:
        return "RdBu_r", lo, hi
    if base == "temperature":
        return "thermal", lo, hi
    return "viridis", lo, hi


class TraceViewer(QWidget):
    """The captured step's stages side by side (see the module docstring).

    ``captureRequested`` fires on Capture step; the page asks the solver and
    later calls :meth:`set_trace` with what arrives. :meth:`set_capture_enabled`
    disables the button with the reason (no run, or a solver without traces).
    """

    captureRequested = Signal()
    exported = Signal(str)

    #: Rows of thumbnails: two when they fit at :data:`THUMB_MIN_H` each, else one (a very short dock).
    ROWS = 2
    #: Smallest thumbnail image height [px] (its one-line caption comes on top); a taller viewer gets
    #: taller thumbnails, up to THUMB_MAX_H, and more rows when two leave them taller than that.
    THUMB_MIN_H = 30
    THUMB_MAX_H = 120
    #: Narrowest the enlarged view gets [px] (it takes about 40 % of a wide viewer).
    DETAIL_MIN_W = 260
    #: Below this height [px] the enlarged view leaves out its scale bar: in the 180 px Diagnostics dock
    #: the "2 m" box and the caption covered half of a 100 px field (Open in window shows it whole).
    DETAIL_SCALEBAR_MIN_H = 150

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from .common import ToolStrip, label
        from .kpi import DataTable, EmptyState
        self._trace = TraceCapture()
        self._views: list = []
        self._linking = False
        self._export_dir = Path.home()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = ToolStrip()
        self.capture_btn = bar.add_button("target", "Capture step", "Record the stages of the solver's next time "
                                          "step (advection–diffusion, divergence, pressure solve, correction)")
        self.shared_btn = bar.add_button(None, "Shared range", "Panels of the same quantity on one colour map and "
                                         "scale, so a column reads through the stages (off: each panel scaled "
                                         "to its own values)", checkable=True)
        self.shared_btn.setProperty("variant", "toggle")
        self.shared_btn.setChecked(True)
        self.stats_btn = bar.add_button(None, "Statistics", "Min, max, RMS and L2 of every panel in a table "
                                        "(each panel's caption shows its own numbers on hover)", checkable=True)
        self.stats_btn.setProperty("variant", "toggle")
        self.info = bar.add_widget(label("", "muted"))
        bar.add_stretch()
        self.window_btn = bar.add_button("external", "Open in window", "Show this capture in a separate window that "
                                         "can be enlarged (the dock is short)")
        self.npz_btn = bar.add_button("export", "Export NPZ…", "Save every stage's arrays (NumPy .npz)")
        root.addWidget(bar)
        self.bar = bar
        # a narrow dock: the icon buttons give up their text and the step line its place (the bar's full
        # width, 800 px, made a 340 px dock scroll sideways and hid the thumbnails)
        self._bar_texts = [(b, b.text()) for b in (self.capture_btn, self.window_btn, self.npz_btn)]
        bar.setMinimumWidth(1)
        self._window: TraceWindow | None = None
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.empty = EmptyState()
        self.empty.set_content("No step captured",
                               "Capture step records one solver step's stages as fields: the tentative velocity "
                               "after advection and diffusion, its divergence, the pressure correction and the "
                               "corrected velocity.")
        self.stack.addWidget(self.empty)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        # the enlarged panel (left) and the thumbnails (right); a tall, narrow viewer stacks them
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setHandleWidth(1)
        self.split.setChildrenCollapsible(False)
        detail = QWidget()
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(6, 4, 6, 4)
        dl.setSpacing(2)
        # the title rides on the view as its caption: a separate line cost the preview a fifth of a 200 px dock
        self.detail_title = label("", "secondary")
        self.detail_title.hide()
        dl.addWidget(self.detail_title)
        from .field_view import FieldView
        self.detail = FieldView()
        self.detail.setMinimumSize(self.DETAIL_MIN_W - 12, 60)
        self.detail.viewChanged.connect(lambda: self._link_from(self.detail))
        dl.addWidget(self.detail, 1)
        self.detail_stats = label("", "muted")
        dl.addWidget(self.detail_stats)
        detail.setMinimumWidth(self.DETAIL_MIN_W)
        self.detail_host = detail
        self.split.addWidget(detail)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setSpacing(4)
        sc = self.thumb_scroll = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setWidget(self.grid_host)
        self.split.addWidget(sc)
        self.split.setStretchFactor(0, 2)
        self.split.setStretchFactor(1, 3)
        self._split_user = False
        self._split_lock = False
        self.split.splitterMoved.connect(self._on_split_moved)
        bl.addWidget(self.split, 1)
        self.table = DataTable(["Stage", "Field", "Min", "Max", "RMS", "L2"])
        self.table.setMaximumHeight(170)
        self.table.hide()
        bl.addWidget(self.table)
        self.stack.addWidget(body)
        self._cells: list[QWidget] = []
        self._col_heads: list[QWidget] = []
        self._row_heads: list[QWidget] = []
        self._layout: tuple = ([], [], {})        # table_layout of the trace on screen
        self._grid_key: tuple | None = None
        self._selected = 0
        self._thumb_index: dict = {}                            # thumbnail widget -> panel index
        self._ranges: list[tuple[str, float, float]] = []      # (cmap, vmin, vmax) of each panel
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        from .plots import connect_theme
        connect_theme(self._mark_selected)
        self.capture_btn.clicked.connect(lambda *_: self.captureRequested.emit())
        self.shared_btn.toggled.connect(lambda *_: self._draw())
        self.stats_btn.toggled.connect(self.table.setVisible)
        self.npz_btn.clicked.connect(self.export_npz)
        self.npz_btn.setEnabled(False)
        self.window_btn.clicked.connect(lambda *_: self.open_window())
        self.window_btn.setEnabled(False)
        # Filters go on last: eventFilter reads detail, thumb_scroll and _thumb_index, and a filter installed
        # mid-construction received the first event before they existed (the Simulate page then failed to
        # build). _filters_ready keeps any event Qt delivers early from touching a half-built viewer.
        self._filters_ready = True
        self.detail.installEventFilter(self)
        sc.viewport().installEventFilter(self)

    # -- data ----------------------------------------------------------------
    def set_capture_enabled(self, enabled: bool, reason: str = "") -> None:
        self.capture_btn.setEnabled(bool(enabled))
        self.capture_btn.setToolTip(
            "Record the stages of the solver's next time step (advection–diffusion, divergence, pressure solve, "
            "correction)" if enabled else (reason or "No solver is running"))

    def set_waiting(self, waiting: bool) -> None:
        """Say that a capture was requested and the next frame will bring it."""
        if waiting:
            self.info.setText("Capturing the next step…")
        elif self._trace.stages:
            self._set_info()

    def set_export_dir(self, folder: str | Path) -> None:
        self._export_dir = Path(folder)

    def set_trace(self, trace, *, extent: Sequence[float] | None = None) -> TraceCapture:
        """Show a captured step (any form :func:`normalize_trace` accepts); returns the normalized capture."""
        self._trace = normalize_trace(trace, extent=extent)
        self._draw()
        if self._window is not None and self._window.isVisible():
            self._window.viewer.set_trace(self._trace)
        return self._trace

    def open_window(self) -> TraceWindow:
        """Show the capture in a floating, resizable window (one per viewer, raised when already open)."""
        if self._window is None:
            self._window = TraceWindow(self)
            self._window.destroyed.connect(self._window_gone)
        self._window.viewer.set_export_dir(self._export_dir)
        self._window.viewer.set_trace(self._trace)
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()
        return self._window

    def trace(self) -> TraceCapture:
        return self._trace

    def _set_info(self) -> None:
        tr = self._trace
        parts = []
        if tr.step is not None:
            parts.append(f"step {tr.step:,}")
        if tr.t is not None and math.isfinite(tr.t):
            from ...core.units import sig
            parts.append(f"t = {sig(tr.t, 4)} L/U")
        parts.append(f"{len(tr.stages)} panels")
        self.info.setText(" · ".join(parts))
        self.info.setToolTip("\n".join(tr.notes))

    # -- drawing -------------------------------------------------------------
    def _clear_grid(self) -> None:
        self._thumb_index = {}
        self._col_heads, self._row_heads = [], []
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._views = []
        self._cells = []
        self._grid_key = None

    def eventFilter(self, obj, event) -> bool:
        if not getattr(self, "_filters_ready", False):
            return False
        if obj is self.detail and event.type() == QEvent.Type.Resize:
            self.detail.show_scalebar = self.detail.height() >= self.DETAIL_SCALEBAR_MIN_H
        elif obj is self.thumb_scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._regrid()
        elif event.type() == QEvent.Type.MouseButtonPress and obj in self._thumb_index:
            self.select(self._thumb_index[obj])
            return True
        return super().eventFilter(obj, event)

    def _compact_bar(self) -> None:
        compact = self.width() < 760
        for b, text in self._bar_texts:
            want = "" if compact else text
            if b.text() != want:
                b.setText(want)
                b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly if compact
                                     else Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.info.setVisible(self.width() >= 560)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._compact_bar()
        # side by side when wide; the enlarged view above the thumbnails when tall and narrow
        tall = self.height() > 1.2 * self.width()
        want = Qt.Orientation.Vertical if tall else Qt.Orientation.Horizontal
        if self.split.orientation() != want:
            self.split.setOrientation(want)
            self._split_user = False
        if not self._split_user:
            QTimer.singleShot(0, self, self._apply_split)

    def _apply_split(self) -> None:
        """The enlarged view takes about 40 % (its FieldView's 640 px size hint left the thumbnails 52 px)."""
        if self._split_user:
            return
        horiz = self.split.orientation() == Qt.Orientation.Horizontal
        total = self.split.width() if horiz else self.split.height()
        if total <= 0:
            return
        if horiz:
            # as wide as the selected field needs at the pane's height, plus its colour bar: a fixed 40 %
            # of a short, wide dock framed a 100 px image in 640 px of nothing
            want = 0.4 * total
            stages = self._trace.stages
            if stages:
                x0, x1, y0, y1 = stages[min(self._selected, len(stages) - 1)].extent
                aspect = (x1 - x0) / (y1 - y0) if y1 > y0 and x1 > x0 else 1.0
                title_h = self.detail_title.sizeHint().height() if not self.detail_title.isHidden() else 0
                img_h = max(40.0, self.split.height() - title_h - self.detail_stats.sizeHint().height() - 16)
                want = img_h * aspect / 0.96 + self.detail.colorbar_reserve() + 28
            first = int(min(max(self.DETAIL_MIN_W, want), 0.45 * total))
        else:
            first = int(max(120, 0.45 * total))
        self._split_lock = True
        self.split.setSizes([first, max(1, total - first)])
        self._split_lock = False

    def _on_split_moved(self, *_args) -> None:
        if not self._split_lock:
            self._split_user = True          # the user's divider position stays

    def keyPressEvent(self, e) -> None:
        if self._trace.stages and e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Up):
            step = 1 if e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down) else -1
            self.select((self._selected + step) % len(self._trace.stages))
            return
        super().keyPressEvent(e)

    def grid_shape(self) -> tuple[int, int]:
        """(rows, columns) of the panel grid now (in the grouped table: stages x quantities)."""
        return (self._grid_key[0], self._grid_key[1]) if self._grid_key else (0, 0)

    def grouped(self) -> bool:
        """Whether the panels are laid out as the RK-stage x quantity table."""
        return bool(self._layout[0])

    def _caption_h(self) -> int:
        if self._layout[0]:
            return 0                                   # the table's headers carry the names
        if not self._cells:
            return 16
        lay = self._cells[0].layout()
        first = lay.itemAt(0).widget() if lay is not None and lay.count() else None
        return (first.sizeHint().height() if first is not None else 16) + (lay.spacing() if lay else 0)

    def _rows_for(self, height: int, n: int) -> tuple[int, int]:
        """(rows, thumbnail image height) for a thumbnail area ``height`` px tall."""
        m = self.grid.contentsMargins()
        avail = max(1, height - m.top() - m.bottom())
        row_h = self.THUMB_MIN_H + self._caption_h() + self.grid.verticalSpacing()
        rows = min(self.ROWS, n) if avail >= self.ROWS * row_h else 1
        # a tall area: more rows while two would make the thumbnails taller than THUMB_MAX_H
        while rows < n and avail / rows - self._caption_h() > self.THUMB_MAX_H:
            rows += 1
        img_h = int(max(self.THUMB_MIN_H, avail / rows - self._caption_h() - self.grid.verticalSpacing()))
        return rows, img_h

    def _size_panels(self, img_h: int) -> None:
        """Every thumbnail as tall as its row allows and as wide as its field's aspect ratio needs, so the
        whole field shows (a fixed minimum in a 110 px dock clipped every image to its top half)."""
        for v, s in zip(self._views, self._trace.stages, strict=False):
            x0, x1, y0, y1 = s.extent
            aspect = (x1 - x0) / (y1 - y0) if y1 > y0 and x1 > x0 else 1.0
            w = img_h * aspect / 0.98 + 2                       # the compact view pads its fit by 2 %
            v.setFixedSize(int(min(max(w, 40.0), 6.0 * img_h)), img_h)

    def _regrid(self) -> None:
        """The grouped table (one row per RK stage, one column per quantity), else two rows of thumbnails
        (one in a very short dock) in stage order; either scrolls when it does not fit."""
        n = len(self._cells)
        if not n:
            return
        h = max(self.thumb_scroll.viewport().height(), 1)
        rows_g, cols_g, cell = self._layout
        if rows_g:
            m = self.grid.contentsMargins()
            head = max((lab.sizeHint().height() for lab in self._col_heads), default=16) + self.grid.verticalSpacing()
            avail = h - m.top() - m.bottom() - head
            by_h = avail / len(rows_g) - self.grid.verticalSpacing() - 4
            # small enough that the whole table fits the width too where it can: it is an overview, the
            # enlarged view beside it is for reading (120 px thumbnails showed one column of fourteen)
            x0, x1, y0, y1 = self._trace.stages[0].extent
            aspect = (x1 - x0) / (y1 - y0) if y1 > y0 and x1 > x0 else 1.0
            head_w = max((lab.sizeHint().width() for lab in self._row_heads), default=40)
            vw = max(self.thumb_scroll.viewport().width(), 1) - m.left() - m.right() - head_w
            by_w = (vw / len(cols_g) - self.grid.horizontalSpacing() - 6) / aspect
            img_h = int(min(max(self.THUMB_MIN_H, min(by_h, by_w)), self.THUMB_MAX_H))
            # by_w is an estimate: it leaves out the thumbnails' 2 % fit pad and the column titles
            # ("u* adv.", "u corr.") that are wider than a small thumbnail, and the table ran ~40 px
            # past a 1600 px window, its last column cut. Shrink until the whole table fits.
            heads_w = [lab.sizeHint().width() for lab in self._col_heads]
            sp = self.grid.horizontalSpacing()
            # what a cell adds around its thumbnail (layout margins, the selection frame), measured
            self._size_panels(img_h)
            extra = 8
            if self._cells and self._views:
                extra = max(0, self._cells[0].sizeHint().width() - self._views[0].width())

            def table_w(hh: int) -> float:
                thumb = int(min(max(hh * aspect / 0.98 + 2, 40.0), 6.0 * hh)) + extra
                return m.left() + m.right() + head_w + sum(max(thumb, hw) + sp for hw in heads_w)

            fit_w = max(self.thumb_scroll.viewport().width(), 1)
            while img_h > self.THUMB_MIN_H and table_w(img_h) > fit_w:
                img_h -= 1
            self._size_panels(img_h)
            key = (len(rows_g), len(cols_g), n)
            if key == self._grid_key:
                return
            self._grid_key = key
            for c in self._cells + self._col_heads + self._row_heads:
                self.grid.removeWidget(c)
            for j, lab in enumerate(self._col_heads):
                self.grid.addWidget(lab, 0, j + 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
            for r, lab in enumerate(self._row_heads):
                self.grid.addWidget(lab, r + 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for i, c in enumerate(self._cells):
                r, col = cell[i]
                self.grid.addWidget(c, r + 1, col + 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            for c in range(len(cols_g) + 2):
                self.grid.setColumnStretch(c, 0)
            self.grid.setColumnStretch(len(cols_g) + 1, 1)
            self.grid.setRowStretch(len(rows_g) + 1, 1)
            return
        rows, img_h = self._rows_for(h, n)
        cols = math.ceil(n / rows)
        rows = math.ceil(n / cols)
        # sizes follow every height change (the sideways scroll bar appearing shortens the strip)
        self._size_panels(img_h)
        key = (rows, cols, n)
        if key == self._grid_key:
            return
        self._grid_key = key
        for c in self._cells:
            self.grid.removeWidget(c)
        for i, c in enumerate(self._cells):
            self.grid.addWidget(c, i // cols, i % cols, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for c in range(cols + 1):
            self.grid.setColumnStretch(c, 0)
        self.grid.setColumnStretch(cols, 1)
        self.grid.setRowStretch(rows, 1)
        self.thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def _draw(self) -> None:
        from .common import label
        from .field_view import FieldView
        from .kpi import fmt_sig
        tr = self._trace
        self._clear_grid()
        self.npz_btn.setEnabled(bool(tr.stages))
        if not tr.stages:
            self.stack.setCurrentIndex(0)
            self.info.setText("")
            self.table.set_data(["Stage", "Field", "Min", "Max", "RMS", "L2"], [])
            self.detail.clear()
            self._selected = 0
            return
        self.stack.setCurrentIndex(1)
        self.window_btn.setEnabled(True)
        self._set_info()
        self._layout = table_layout(tr.stages)
        rows_g, cols_g, _cell = self._layout
        for g in rows_g:
            lab = label(g, "secondary")
            lab.setToolTip({"End of step": "The fields at the end of the step"}.get(
                g, f"Runge–Kutta stage {g[3:]} of the step" if g.startswith("RK ") else g))
            self._row_heads.append(lab)
        for st_name, fld, _n in cols_g:
            lab = label(column_title(st_name, fld), "secondary")
            lab.setToolTip(f"{st_name}: {fld}" if fld else st_name)
            self._col_heads.append(lab)
        shared: dict[str, tuple[float, float]] = {}
        if self.shared_btn.isChecked():
            from .field_view import robust_range
            by_base: dict[str, list[np.ndarray]] = {}
            for s in tr.stages:
                by_base.setdefault(_base_name(s.field or s.stage), []).append(s.data)
            for base, arrs in by_base.items():
                signed = base in ("divergence", "phi", "p", "v", "w") or base.startswith("d")
                shared[base] = robust_range(np.concatenate([a.ravel() for a in arrs]), diverging=signed,
                                            lo=0.5, hi=99.5)
        rows = []
        self._ranges = []
        self._thumb_index = {}
        for i, s in enumerate(tr.stages):
            base = _base_name(s.field or s.stage)
            cmap, lo, hi = _cmap_and_range(base, s.data, shared.get(base))
            self._ranges.append((cmap, lo, hi))
            cell = QFrame()
            cell.setProperty("trace-cell", True)
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(1)
            title = self.panel_title(i)
            st = stage_stats(s.data)
            # the thumbnail's caption is its number and field ("3 θ*"); the stage words are in the
            # enlarged view's title and the tooltip (a full title on a 50 px thumbnail is "3 · Adv…").
            # In the grouped table the row and column headers name it, and the caption goes.
            cap = label(f"{i + 1} {s.field or s.stage}", "muted")
            if rows_g:
                cap.hide()
            nums = (f"min {fmt_sig(st['min'], 3)} · max {fmt_sig(st['max'], 3)} · RMS {fmt_sig(st['rms'], 3)} · "
                    f"L2 {fmt_sig(st['l2'], 3)}" + (f" {s.units}" if s.units else ""))
            tip = "\n".join(x for x in (title, nums, s.note, "Click to enlarge") if x)
            cap.setToolTip(tip)
            cl.addWidget(cap)
            v = FieldView(compact=True, interactive=False)
            v.show_scalebar = False
            v.show_colorbar = False
            v.setToolTip(tip)
            v.setCursor(Qt.CursorShape.PointingHandCursor)
            v.set_field(s.data, s.extent, field=base if base in ("temperature",) else "", cmap=cmap, vmin=lo,
                        vmax=hi, label=s.field or s.stage, units=s.units)
            v.viewChanged.connect(lambda src=v: self._link_from(src))
            v.installEventFilter(self)
            cap.installEventFilter(self)
            self._thumb_index[v] = i
            self._thumb_index[cap] = i
            cl.addWidget(v)
            self._cells.append(cell)
            self._views.append(v)
            rows.append([s.stage, s.field or "—", fmt_sig(st["min"], 3), fmt_sig(st["max"], 3),
                         fmt_sig(st["rms"], 3), fmt_sig(st["l2"], 3)])
        self._regrid()
        self.table.set_data(["Stage", "Field", "Min", "Max", "RMS", "L2"], rows, numeric=[2, 3, 4, 5])
        self.select(min(self._selected, len(tr.stages) - 1), keep_view=True)

    def panel_title(self, i: int) -> str:
        s = self._trace.stages[i]
        return (f"{i + 1} · {s.stage}" + (f": {s.field}" if s.field else "")
                + (f" · {s.group}" if s.group and s.group != s.stage else ""))

    def selected(self) -> int:
        """Index of the enlarged panel."""
        return self._selected

    def select(self, i: int, *, keep_view: bool = False) -> None:
        """Enlarge panel ``i`` beside the thumbnails (with its colour bar, scale bar and readout)."""
        from .kpi import fmt_sig
        tr = self._trace
        if not tr.stages:
            return
        i = max(0, min(int(i), len(tr.stages) - 1))
        self._selected = i
        s = tr.stages[i]
        cmap, lo, hi = self._ranges[i] if i < len(self._ranges) else _cmap_and_range(
            _base_name(s.field or s.stage), s.data, None)
        base = _base_name(s.field or s.stage)
        state = self.detail.view_state() if keep_view and self.detail.is_user_view() else None
        self.detail.set_field(s.data, s.extent, field=base if base in ("temperature",) else "", cmap=cmap, vmin=lo,
                              vmax=hi, label=s.field or s.stage, units=s.units)
        if state is not None:
            self.detail.set_view_state(state)
        self.detail_title.setText(self.panel_title(i))
        self.detail_title.setToolTip(s.note)
        self.detail.set_caption(self.panel_title(i))
        st = stage_stats(s.data)
        self.detail_stats.setText(f"min {fmt_sig(st['min'], 3)} · max {fmt_sig(st['max'], 3)} · "
                                  f"RMS {fmt_sig(st['rms'], 3)} · L2 {fmt_sig(st['l2'], 3)}"
                                  + (f" {s.units}" if s.units else ""))
        self._mark_selected()
        QTimer.singleShot(0, self, self._apply_split)       # the enlarged pane follows the field's shape
        if i < len(self._cells):
            self.thumb_scroll.ensureWidgetVisible(self._cells[i], 8, 8)

    def _mark_selected(self, *_args) -> None:
        """Outline the enlarged panel's thumbnail in the accent colour."""
        from .plots import tokens
        tk = tokens()
        for j, c in enumerate(self._cells):
            try:
                c.setStyleSheet(f"QFrame[trace-cell=\"true\"] {{ border: 2px solid {tk['accent']}; border-radius: 3px; }}"
                                if j == self._selected else
                                "QFrame[trace-cell=\"true\"] { border: 2px solid transparent; border-radius: 3px; }")
            except RuntimeError:             # a cell already deleted with the previous trace
                pass

    def _link_from(self, src) -> None:
        """Zoom and pan of one panel (the enlarged one, or a thumbnail zoomed in code) applied to every
        thumbnail -- same extent, the same cells -- at each one's own scale (they are smaller)."""
        if self._linking or not src.is_user_view():
            return                       # automatic fits stay per panel; only the user's zoom / pan is shared
        self._linking = True
        try:
            s, cx, cy = src.view_state()
            fit_src = self._fit_scale(src)
            for v in self._views + [self.detail]:
                if v is src:
                    continue
                fit_v = self._fit_scale(v)
                k = fit_v / fit_src if fit_src > 0 and fit_v > 0 else 1.0
                v.set_view_state((s * k, cx, cy))
        finally:
            self._linking = False

    @staticmethod
    def _fit_scale(v) -> float:
        """Pixels per metre the view would show its whole extent at (the zoom's reference)."""
        ext = v.extent()
        r = v.plot_rect()
        if ext is None or r.width() <= 0 or r.height() <= 0:
            return 0.0
        x0, x1, y0, y1 = ext
        return min(r.width() / max(x1 - x0, 1e-12), r.height() / max(y1 - y0, 1e-12))

    def views(self) -> list:
        return list(self._views)

    def _window_gone(self, *_args) -> None:
        self._window = None

    def closeEvent(self, e) -> None:
        if self._window is not None:
            self._window.close()
        super().closeEvent(e)

    # -- export ----------------------------------------------------------------
    def export_npz(self, path: str | None = None) -> str | None:
        """Save every panel's array (``"<i>_<stage>_<field>"``) with its extent, step and time."""
        tr = self._trace
        if not tr.stages:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export solver trace", str(self._export_dir / "trace.npz"),
                                                  "NumPy archive (*.npz)")
            if not path:
                return None
        arrays = {}
        for i, s in enumerate(tr.stages):
            key = re.sub(r"[^A-Za-z0-9]+", "_", f"{i:02d}_{s.stage}_{s.field}").strip("_")
            arrays[key] = s.data
            arrays[key + "__extent"] = np.asarray(s.extent, dtype=np.float64)
        arrays["step"] = np.asarray(-1 if tr.step is None else tr.step)
        arrays["t"] = np.asarray(math.nan if tr.t is None else tr.t)
        np.savez_compressed(path, **arrays)
        self.exported.emit(str(path))
        return str(path)


class TraceWindow(QWidget):
    """A floating, resizable window with a :class:`TraceViewer` of the same capture ("Open in window").

    The Diagnostics dock is a 200 px strip; a solver developer inspecting a
    stage wants the table and the enlarged field at full size. Capture step
    stays in the dock (the window shows what the dock captured, and follows
    new captures while open).
    """

    def __init__(self, source: TraceViewer) -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)   # a closed window is gone, not hidden
        self.setWindowTitle("Solver trace")
        self.resize(1400, 900)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.viewer = TraceViewer(self)
        self.viewer.capture_btn.hide()
        self.viewer.window_btn.hide()
        self.viewer.exported.connect(source.exported.emit)
        lay.addWidget(self.viewer)
