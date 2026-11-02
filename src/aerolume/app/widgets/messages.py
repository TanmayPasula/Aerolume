"""The Messages dock: planner notes, warnings, errors and the run log in one table.

Columns are time, a level glyph, source and text - the shape of an IDE's
problems / output pane, because that is what engineers already scan quickly.
Rows come from two places:

* ``AppState.messageLogged(level, source, text)`` - anything a page reports
  with ``state.log(text, level, source)`` (the Setup planner's notes and
  warnings, import problems, renames ...);
* the JobManager's lifecycle signals - started, finished (with the wall-clock
  duration), failed (with the reason), stopped - logged here so no page has
  to duplicate it.

Every row is tied to the simulation it is about (or to none, for general
lines such as "Opened project"), and the "Selected simulation" filter - on by
default - hides the rows of other simulations. Without it the dock showed the
airfoil planner's warning while the user looked at the sedan's results, and
its counters contradicted the page beside it. ``AppState.log`` does not carry
a simulation id, so :meth:`MessagesPanel.attribute` infers one: a quoted
simulation name in the text (“CFD · α 4°”, the style every page uses), else
the selected simulation for the per-simulation sources (Planner, Setup,
Solver), else none. A failed run is the one exception to the filter: the
status bar says "failed — see Messages", so that line always shows.

The filter buttons double as counters ("2 errors", "5 warnings") of what the
current scope contains, and ``latest`` repeats the newest line in scope, so
the dock can sit collapsed to its title row and a glance still tells whether
anything needs attention. The log is capped (oldest rows dropped) so a long
session cannot grow it without bound.
"""
from __future__ import annotations

import re
import time

from PySide6.QtCore import QSize, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QFontMetrics,
    QGuiApplication,
    QKeySequence,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .. import settings as app_settings
from .. import theme
from .common import ElidedLabel, IconButton

LEVELS = ("error", "warning", "info")
LEVEL_ROLE = Qt.ItemDataRole.UserRole + 1
SIM_ROLE = Qt.ItemDataRole.UserRole + 2         # id of the simulation a row is about ("" = general)
SIM_NAME_ROLE = Qt.ItemDataRole.UserRole + 3    # its name when logged (survives a later delete)
_ALIASES = {"warn": "warning", "err": "error", "critical": "error", "ok": "info", "success": "info",
            "note": "info", "debug": "info"}
_GLYPH = {"error": ("error", "bad"), "warning": ("warning", "warn"), "info": ("info", "text2")}
MAX_ROWS = 5000
# Sources whose lines are about the selected simulation even when they name none.
SIM_SOURCES = frozenset({"Planner", "Setup", "Solver", "Simulate"})
_QUOTED = re.compile(r"“([^”]+)”")


def normalise_level(level: str) -> str:
    """Map the many ways callers spell a level onto "error" / "warning" / "info"."""
    lv = (level or "info").lower()
    lv = _ALIASES.get(lv, lv)
    return lv if lv in LEVELS else "info"


def format_duration(seconds: float) -> str:
    """"42 s", "3 min 24 s", "1 h 05 min" - the style the guidelines use for status text."""
    s = max(0, round(seconds))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def in_scope(sim_id: str, level: str, source: str, current_only: bool, current: str) -> bool:
    """Whether a row belongs in the view under the "Selected simulation" filter.

    General rows (no simulation) always do, and so does everything while no
    simulation is selected - an empty log would hide project-level problems.
    A run failure shows from any simulation, because the status bar sends the
    user here to read it.
    """
    if not current_only or not current or not sim_id or sim_id == current:
        return True
    return level == "error" and source == "Run"


class _MessageFilter(QSortFilterProxyModel):
    """Level toggles + simulation scope; names the simulation of rows from another one."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.levels = set(LEVELS)
        self.current_only = True
        self.current = ""

    def row_in_scope(self, row: int, parent=None) -> bool:
        m = self.sourceModel()
        idx = m.index(row, 0) if parent is None else m.index(row, 0, parent)
        return in_scope(idx.data(SIM_ROLE) or "", idx.data(LEVEL_ROLE) or "info",
                        m.index(row, 2, idx.parent()).data() or "", self.current_only, self.current)

    def filterAcceptsRow(self, row: int, parent) -> bool:
        idx = self.sourceModel().index(row, 0, parent)
        return idx.data(LEVEL_ROLE) in self.levels and self.row_in_scope(row, parent)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        # A row about another simulation (visible with the filter off, or a failed run) says
        # which one in its Source cell, so a warning is never read as the selected one's.
        if role == Qt.ItemDataRole.DisplayRole and index.column() == 2:
            src = self.mapToSource(index)
            first = src.siblingAtColumn(0)
            sim = first.data(SIM_ROLE) or ""
            if sim and sim != self.current:
                name = first.data(SIM_NAME_ROLE) or ""
                text = src.data() or ""
                return f"{text} · {name}" if text and name else (text or name)
        return super().data(index, role)


class LatestLine(ElidedLabel):
    """The newest message on one elided line; a click asks the host to open the log.

    The press is accepted here (a QLabel would let it fall through to the dock
    title), otherwise the release never reaches this widget and the click is lost.
    """

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("", "secondary", mode=Qt.TextElideMode.ElideRight, parent=parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._level = "info"

    def set_message(self, level: str, source: str, text: str) -> None:
        self._level = level
        line = " ".join(str(text).split())
        self.setText(f"{source}: {line}" if source else line)
        theme.set_prop(self, "role", {"error": "status-bad", "warning": "status-warn"}.get(level, "secondary"))

    def reset(self) -> None:
        self._level = "info"
        self.setText("")
        theme.set_prop(self, "role", "secondary")

    def level(self) -> str:
        return self._level

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
            return
        super().mouseReleaseEvent(e)


class MessagesPanel(QWidget):
    """Time / level / source / message table with level and simulation filters, Clear and Copy.

    ``messageAdded`` fires only for rows the current scope shows, so a host
    that opens the dock on a new warning does not open it onto nothing.
    """

    countsChanged = Signal(int, int, int)       # errors, warnings, info - in the current scope
    messageAdded = Signal(str, str, str)        # normalised level, source, text (after the row is in)

    def __init__(self, state=None, jobs=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self._counts = dict.fromkeys(LEVELS, 0)
        self._job_started: dict[str, float] = {}

        self.model = QStandardItemModel(0, 4, self)
        self.model.setHorizontalHeaderLabels(["Time", "", "Source", "Message"])
        self.proxy = _MessageFilter(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.current_only = app_settings.get().messages_current_only
        self.proxy.current = self._current_sim()

        # Filter toggles + Clear live in ``self.bar`` so a host can move them into the dock's
        # title row (one row saved); standalone, the panel shows them above the table.
        self.bar = QWidget()
        bar = QHBoxLayout(self.bar)
        bar.setContentsMargins(4, 0, 4, 0)
        bar.setSpacing(2)
        self.filter_buttons: dict[str, QToolButton] = {}
        for lv in LEVELS:
            name, colour = _GLYPH[lv]
            # faint when the level is hidden, its status colour when shown: the state reads
            # from the glyph as well as the text colour, with no fill needed (flat buttons)
            b = self._toggle(theme.icon(name, "faint", active=colour), f"Show {lv} messages")
            b.setChecked(True)
            b.toggled.connect(self._apply_filter)
            bar.addWidget(b)
            self.filter_buttons[lv] = b
        bar.addSpacing(8)
        self.scope_button = self._toggle(theme.icon("filter", "faint", active="accent"),
                                         "Show only the selected simulation's messages and general ones "
                                         "(a failed run always shows)")
        self.scope_button.setText("Selected simulation")
        self.scope_button.setObjectName("messagesScope")
        self.scope_button.setChecked(self.proxy.current_only)
        self.scope_button.toggled.connect(self._scope_toggled)
        bar.addWidget(self.scope_button)
        bar.addSpacing(4)
        self.clear_button = IconButton("trash", "Clear messages", icon_size=14)
        self.clear_button.clicked.connect(self.clear)
        bar.addWidget(self.clear_button)
        # The newest line, for a host that collapses the panel to its title row. Not part of
        # ``bar`` because it should take the free width there, which a compact bar must not.
        self.latest = LatestLine()

        self.view = QTreeView()
        self.view.setModel(self.proxy)
        self.view.setProperty("pane", "true")
        self.view.setRootIsDecorated(False)
        self.view.setUniformRowHeights(True)
        self.view.setAlternatingRowColors(False)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.view.setIconSize(QSize(14, 14))
        self.view.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.view.setAccessibleName("Messages")
        h = self.view.header()
        h.setStretchLastSection(True)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self._fit_time_column()
        h.resizeSection(1, 26)
        h.resizeSection(2, 130)
        h.setMinimumSectionSize(22)

        copy = QAction("Copy", self.view)
        copy.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
        copy.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        copy.triggered.connect(self.copy_selection)
        self.view.addAction(copy)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        clear = QAction("Clear", self.view)
        clear.triggered.connect(self.clear)
        self.view.addAction(clear)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.bar)
        lay.addWidget(self.view, 1)
        self._update_counts()

        theme.changed.connect(self._recolour)    # bound method: auto-disconnects when deleted
        if state is not None:
            state.messageLogged.connect(self.add)
            for sig in ("currentSimulationChanged", "projectChanged"):
                s = getattr(state, sig, None)
                if s is not None:
                    s.connect(lambda *_a: self._rescope())
        if jobs is not None:
            self.connect_jobs(jobs)

    @staticmethod
    def _toggle(icon, tip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon)
        b.setIconSize(QSize(14, 14))
        b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        b.setFixedHeight(theme.MIN_TARGET)      # WCAG 2.2 SC 2.5.8: 24 px targets (was 20)
        b.setCheckable(True)
        b.setProperty("variant", "toggle")
        b.setToolTip(tip)
        b.setAccessibleName(tip)
        return b

    # -- attribution ------------------------------------------------------------
    def _current_sim(self) -> str:
        return str(getattr(self.state, "simulation_id", "") or "")

    def _sim_name(self, sim_id: str) -> str:
        proj = getattr(self.state, "project", None)
        sim = proj.simulations.get(sim_id) if proj is not None and sim_id else None
        return sim.name if sim is not None else ""

    def attribute(self, source: str, text: str) -> str:
        """The id of the simulation a message is about, or "" for a general one.

        A quoted name wins (Simulate logs runs that are not selected); a
        per-simulation source with no name means the selected simulation
        (the Setup planner logs the case on screen).
        """
        proj = getattr(self.state, "project", None)
        if proj is None:
            return ""
        names = _QUOTED.findall(str(text))
        if names:
            by_name = {s.name: sid for sid, s in proj.simulations.items()}
            for n in names:
                if n in by_name:
                    return by_name[n]
        if source in SIM_SOURCES:
            cur = self._current_sim()
            return cur if cur in proj.simulations else ""
        return ""

    # -- feeding ----------------------------------------------------------------
    def add(self, level: str, source: str, text: str, sim_id: str | None = None) -> None:
        """Append one message (levels are normalised; newest row at the bottom).

        ``sim_id`` ties the row to a simulation; ``None`` infers it
        (:meth:`attribute`) and ``""`` marks a general message.
        """
        lv = normalise_level(level)
        source = source or ""
        sim = self.attribute(source, text) if sim_id is None else str(sim_id)
        sim_name = self._sim_name(sim) or (sim if sim_id is not None else "")
        sb = self.view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 2
        t = QStandardItem(time.strftime("%H:%M:%S"))
        t.setFont(theme.mono_font())
        t.setForeground(theme.qcolor("muted"))
        t.setData(lv, LEVEL_ROLE)
        t.setData(sim, SIM_ROLE)
        t.setData(sim_name, SIM_NAME_ROLE)
        g = QStandardItem()
        name, colour = _GLYPH[lv]
        g.setIcon(theme.icon(name, colour))
        g.setToolTip(lv.capitalize())
        s = QStandardItem(source)
        s.setForeground(theme.qcolor("text2"))
        if sim_name:
            s.setToolTip(f"{source} · {sim_name}" if source else sim_name)
        m = QStandardItem(" ".join(str(text).split()))       # one line per message
        m.setToolTip(f"{sim_name}\n{text}" if sim_name else str(text))
        if lv == "error":
            m.setForeground(theme.qcolor("bad"))
        self.model.appendRow([t, g, s, m])
        shown = in_scope(sim, lv, source, self.proxy.current_only, self.proxy.current)
        if self.model.rowCount() > MAX_ROWS:
            self.model.removeRows(0, self.model.rowCount() - MAX_ROWS)
            self._recount()
        elif shown:
            self._counts[lv] += 1
            self._update_counts()
        if not shown:
            return
        self.latest.set_message(lv, source, str(text))
        if at_bottom:
            self.view.scrollToBottom()
        self.messageAdded.emit(lv, source, str(text))

    def connect_jobs(self, jobs) -> None:
        """Log job start / finish / failure / stop from a JobManager."""
        for sig, slot in (("jobStarted", self._started), ("jobFinished", self._finished),
                          ("jobFailed", self._failed), ("jobCancelled", self._cancelled)):
            s = getattr(jobs, sig, None)
            if s is not None:
                s.connect(slot)

    def _name(self, sim_id: str) -> str:
        return self._sim_name(sim_id) or sim_id

    def _elapsed(self, sim_id: str) -> str:
        t0 = self._job_started.pop(sim_id, None)
        return f" in {format_duration(time.monotonic() - t0)}" if t0 is not None else ""

    def _started(self, sim_id: str) -> None:
        self._job_started[sim_id] = time.monotonic()
        self.add("info", "Run", f"Started “{self._name(sim_id)}”", sim_id=sim_id)

    def _finished(self, sim_id: str, _result=None) -> None:
        self.add("info", "Run", f"“{self._name(sim_id)}” finished{self._elapsed(sim_id)}", sim_id=sim_id)

    def _failed(self, sim_id: str, error: str) -> None:
        self._job_started.pop(sim_id, None)
        self.add("error", "Run", f"“{self._name(sim_id)}” failed: {error}", sim_id=sim_id)

    def _cancelled(self, sim_id: str) -> None:
        self.add("warning", "Run", f"“{self._name(sim_id)}” stopped{self._elapsed(sim_id)}; "
                 "a partial result is kept when one was produced", sim_id=sim_id)

    # -- scope ------------------------------------------------------------------
    def current_only(self) -> bool:
        return self.proxy.current_only

    def set_current_only(self, on: bool) -> None:
        """Turn the "Selected simulation" filter on or off (remembered across sessions)."""
        self.scope_button.setChecked(bool(on))     # -> _scope_toggled

    def _scope_toggled(self, on: bool) -> None:
        app_settings.get().messages_current_only = bool(on)
        self._rescope()

    def _rescope(self) -> None:
        """Re-filter after the scope or the selected simulation changed."""
        self._refilter(current_only=self.scope_button.isChecked(), current=self._current_sim())
        self._recount()
        self._refresh_latest()

    def _refilter(self, **attrs) -> None:
        begin = getattr(self.proxy, "beginFilterChange", None)
        if callable(begin):                     # Qt >= 6.10: invalidateFilter() is deprecated
            begin()
            for k, v in attrs.items():
                setattr(self.proxy, k, v)
            self.proxy.endFilterChange()
        else:
            for k, v in attrs.items():
                setattr(self.proxy, k, v)
            self.proxy.invalidateFilter()
        # the Source cells of other simulations' rows depend on ``current`` too
        n = self.proxy.rowCount()
        if n:
            self.proxy.dataChanged.emit(self.proxy.index(0, 2), self.proxy.index(n - 1, 2))

    def _recount(self) -> None:
        counts = dict.fromkeys(LEVELS, 0)
        for r in range(self.model.rowCount()):
            if self.proxy.row_in_scope(r):
                lv = self.model.item(r, 0).data(LEVEL_ROLE)
                counts[lv] = counts.get(lv, 0) + 1
        self._counts = counts
        self._update_counts()

    def _refresh_latest(self) -> None:
        for r in range(self.model.rowCount() - 1, -1, -1):
            if self.proxy.row_in_scope(r):
                _t, lv, src, text = self.row(r)
                self.latest.set_message(lv, src, text)
                return
        self.latest.reset()

    # -- view -------------------------------------------------------------------
    def _fit_time_column(self) -> None:
        """Size the time column to "00:00:00" in the mono font, plus the item padding.

        A fixed 72 px cut the time to "11:47:…" with Atkinson Hyperlegible Mono,
        whose digits are wider than Consolas'; measuring keeps it right for
        either font and any display scale.
        """
        fm = QFontMetrics(theme.mono_font())
        self.view.header().resizeSection(0, fm.horizontalAdvance("00:00:00") + 4 * theme.SPACING)

    def _recolour(self, _name: str = "") -> None:
        """Re-apply token colours to existing rows after a theme switch."""
        self._fit_time_column()
        muted, text2, bad = theme.qcolor("muted"), theme.qcolor("text2"), theme.qcolor("bad")
        for r in range(self.model.rowCount()):
            self.model.item(r, 0).setForeground(muted)
            self.model.item(r, 2).setForeground(text2)
            if self.model.item(r, 0).data(LEVEL_ROLE) == "error":
                self.model.item(r, 3).setForeground(bad)

    def _apply_filter(self) -> None:
        self._refilter(levels={lv for lv, b in self.filter_buttons.items() if b.isChecked()})

    def _update_counts(self) -> None:
        words = {"error": ("error", "errors"), "warning": ("warning", "warnings"),
                 "info": ("message", "messages")}
        for lv, b in self.filter_buttons.items():
            n = self._counts[lv]
            b.setText(f"{n} {words[lv][0] if n == 1 else words[lv][1]}")
        self.countsChanged.emit(self._counts["error"], self._counts["warning"], self._counts["info"])

    def counts(self) -> dict[str, int]:
        """Messages per level in the current scope (what the counter buttons show)."""
        return dict(self._counts)

    def row_count(self) -> int:
        return self.model.rowCount()

    def row(self, r: int) -> tuple[str, str, str, str]:
        """(time, level, source, text) of a source-model row (tests, export)."""
        return (self.model.item(r, 0).text(), self.model.item(r, 0).data(LEVEL_ROLE),
                self.model.item(r, 2).text(), self.model.item(r, 3).text())

    def row_simulation(self, r: int) -> str:
        """Id of the simulation a source-model row is about ("" for a general message)."""
        return self.model.item(r, 0).data(SIM_ROLE) or ""

    def visible_rows(self) -> list[tuple[str, str, str, str]]:
        """The rows the view shows now, in order (tests, screenshots)."""
        return [self.row(self.proxy.mapToSource(self.proxy.index(r, 0)).row())
                for r in range(self.proxy.rowCount())]

    def clear(self) -> None:
        self.model.removeRows(0, self.model.rowCount())
        self._counts = dict.fromkeys(LEVELS, 0)
        self.latest.reset()
        self._update_counts()

    def copy_selection(self) -> str:
        rows = sorted({i.row() for i in self.view.selectionModel().selectedIndexes()})
        lines = []
        for r in rows:
            src = self.proxy.mapToSource(self.proxy.index(r, 0)).row()
            t, lv, s, m = self.row(src)
            sim = self.model.item(src, 0).data(SIM_NAME_ROLE) or ""
            lines.append(f"{t}\t{lv}\t{s}\t{m}" if not sim else f"{t}\t{lv}\t{s} · {sim}\t{m}")
        text = "\n".join(lines)
        if text:
            QGuiApplication.clipboard().setText(text)
        return text


__all__ = ["LEVELS", "MAX_ROWS", "SIM_SOURCES", "LatestLine", "MessagesPanel", "format_duration", "in_scope",
           "normalise_level"]
