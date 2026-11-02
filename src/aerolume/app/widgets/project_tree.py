"""The Project tree: designs with their simulations nested, in the left dock.

It replaces the old navigation rail and the design / simulation combo boxes
(docs/ui_guidelines.md section 3), so it is the one place the user picks what
the pages work on:

* **selection drives AppState both ways** - clicking a design selects it (and
  its first simulation if the current one belongs to another design); clicking
  a simulation selects it and its design. When a page changes the selection,
  the tree follows. A re-entrancy guard keeps the two directions from
  ping-ponging.
* **what each item is** - designs carry their kind glyph: the 2D outline
  glyphs (airfoil, car), a cube for 3D meshes, a planform for wings, and a
  muted "3D" / "Wing" tag on the right (2D designs, the common case, carry
  none). Simulation rows carry no tag: their names already say the analysis
  ("Polar · Re 1.0×10⁶", "CFD 3D · 30 m/s"), and in a narrow dock a tag
  plus a status left "CFD …" of the name. The analysis is in the tooltip.
  Names elide at the end (the start says what the run is); the tooltip has the whole name.
* **status per simulation** - a small glyph (not run, queued, done, failed,
  stopped; while running, a ring filled to the fraction done) plus a
  right-aligned suffix ("43 %", "queued", "failed"). Updated from AppState
  signals and directly from JobManager signals, so the percentage moves while
  a run is going without a project save per tick.
* **context menus** - design: new simulation (for wings: new wing analysis
  and new 3D CFD of the lofted wing; for meshes: new 3D CFD), open, rename,
  duplicate, delete; simulation: run / stop, open setup, show results,
  rename, duplicate, delete; empty area: import a 2D outline or a 3D mesh,
  new from template, new wing.
* **inline rename** (F2 or the menu) writes through ``AppState.update_*``.
* **experiments** - the assistant's experiment logs (``<project>/experiments/``,
  DESIGN.md section 9) are listed under an "Experiments" group below the
  designs, newest first, with their date and a status suffix while running,
  stopped or failed. Double-click (or "Open log") emits
  ``experimentActivated(path)``; the window shows the log. They are files, not
  project items, so the tree re-reads them on :meth:`ProjectTree.refresh_experiments`
  (the Assistant dock calls it) and on every rebuild, through a small cache
  keyed by the files' modification times.

Everything that has a page-level equivalent (new simulation, run, import) is
routed through ``AppState.navigate("page:action")`` so it behaves exactly like
the page's own button.
"""
from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QModelIndex, QPoint, QRectF, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QDesktopServices,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMenu,
    QMessageBox,
    QStackedLayout,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..state import analysis_description, analysis_label, design_description, design_icon, design_tag, is_wing
from .common import EmptyState

ID_ROLE = Qt.ItemDataRole.UserRole + 1          # design / simulation id
KIND_ROLE = Qt.ItemDataRole.UserRole + 2        # "design" | "sim"
SUFFIX_ROLE = Qt.ItemDataRole.UserRole + 3      # right-aligned status text ("43 %")
STATUS_ROLE = Qt.ItemDataRole.UserRole + 4      # simulation status string
TAG_ROLE = Qt.ItemDataRole.UserRole + 5         # muted right-most tag: "CFD 3D", "Wing", "3D"
_STATUS_ICON = {"new": ("circle", "muted"), "queued": ("clock", "text2"), "done": ("check-circle", "good"),
                "failed": ("error", "bad"), "cancelled": ("stop", "warn")}
_STATUS_TEXT = {"new": "Not run", "queued": "Queued", "running": "Running", "done": "Done",
                "failed": "Failed", "cancelled": "Stopped"}
#: id of the "Experiments" group item (experiment items use their folder path as id)
EXPERIMENTS_GROUP = "__experiments__"
_EXP_SUFFIX = {"running": "running", "stopped": "stopped", "failed": "failed", "unverified": "unverified"}


def _progress_icon(fraction: float) -> QIcon:
    """A 16 px ring with a filled sector for the running percentage (accent colour)."""
    dpr = 2.0
    pm = QPixmap(int(16 * dpr), int(16 * dpr))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(dpr, dpr)
    acc = theme.qcolor("accent")
    p.setPen(QPen(acc, 1.25))
    p.setBrush(Qt.BrushStyle.NoBrush)
    r = QRectF(2.5, 2.5, 11, 11)
    p.drawEllipse(r)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(acc)
    span = int(-360 * 16 * max(0.04, min(1.0, fraction)))
    p.drawPie(r.adjusted(2, 2, -2, -2), 90 * 16, span)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return QIcon(pm)


class _SuffixDelegate(QStyledItemDelegate):
    """Draws the item normally, then right-aligned: a status suffix and a muted tag.

    The tag (analysis or design dimension) is right-most so it lines up down
    the tree; the status suffix ("43 %") sits just before it in its status
    colour. The name is elided first, so both always stay readable.
    """

    GAP = 8

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        suffix = index.data(SUFFIX_ROLE) or ""
        tag = index.data(TAG_ROLE) or ""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if index.data(KIND_ROLE) == "experiment":
            # a goal is a sentence: keep its start (the view elides at the end too)
            opt.textElideMode = Qt.TextElideMode.ElideRight
        fm = opt.fontMetrics
        small = theme.ui_font(pt=theme.FONT_PT["small"])
        tag_w = QFontMetrics(small).horizontalAdvance(tag) if tag else 0
        suffix_w = fm.horizontalAdvance(suffix) if suffix else 0
        reserve = (suffix_w + self.GAP if suffix else 0) + (tag_w + self.GAP if tag else 0)
        reserve += 4 if reserve else 0
        if reserve:
            opt.rect = opt.rect.adjusted(0, 0, -reserve, 0)
            # paint the full-row background first so the suffix area is highlighted too
            full = QStyleOptionViewItem(option)
            full.text = ""
            full.icon = QIcon()
            style = opt.widget.style() if opt.widget is not None else None
            if style is not None:
                style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, full, painter, opt.widget)
        super().paint(painter, opt, index)
        right = option.rect.adjusted(0, 0, -6, 0)
        if tag:
            painter.save()
            painter.setPen(theme.qcolor("muted"))
            painter.setFont(small)
            painter.drawText(right, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, tag)
            painter.restore()
            right = right.adjusted(0, 0, -(tag_w + self.GAP), 0)
        if suffix:
            painter.save()
            status = index.data(STATUS_ROLE) or ""
            colour = {"running": "accent", "failed": "bad", "unverified": "warn"}.get(status, "muted")
            painter.setPen(theme.qcolor(colour))
            painter.setFont(opt.font)
            painter.drawText(right, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, suffix)
            painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        s = super().sizeHint(option, index)
        return QSize(s.width(), max(s.height(), theme.ROW_H))


class ProjectTree(QWidget):
    """Designs and simulations of the open project; selection is AppState's selection."""

    renameRequested = Signal(str)       # id of the item put into inline edit (tests / status)
    experimentActivated = Signal(str)   # folder of an experiment log the user opened

    def __init__(self, state, jobs=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state, self.jobs = state, jobs
        self._syncing = False
        # >0 while the tree itself writes items: itemChanged is then not a user rename. (Never block
        # the model's signals instead - the view would miss rowsRemoved and keep dangling items.)
        self._building = 0
        self._collapsed: set[str] = set()       # designs the user folded (new ones start open)
        self._progress: dict[str, float] = {}
        self._items: dict[str, QStandardItem] = {}
        self._exp_cache: dict[str, tuple[int, dict]] = {}

        self.model = QStandardItemModel(self)
        self.model.itemChanged.connect(self._item_changed)
        self.view = QTreeView()
        self.view.setModel(self.model)
        self.view.setProperty("pane", "true")
        self.view.setHeaderHidden(True)
        self.view.setUniformRowHeights(True)
        self.view.setIndentation(14)
        self.view.setIconSize(QSize(16, 16))
        # elide at the end: a name reads from its start ("CFD 3D · Re 100 · α 0°" -> "CFD 3D · Re 100…"),
        # which middle-eliding cut into "CFD 3D…α 0°"; the tooltip always has the whole name
        self.view.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.view.setAnimated(False)
        self.view.setExpandsOnDoubleClick(False)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)
        self.view.setItemDelegate(_SuffixDelegate(self.view))
        self.view.selectionModel().currentChanged.connect(self._current_changed)
        self.view.doubleClicked.connect(self._double_clicked)
        self.view.collapsed.connect(lambda idx: self._collapsed.add(idx.data(ID_ROLE)))
        self.view.expanded.connect(lambda idx: self._collapsed.discard(idx.data(ID_ROLE)))
        self.view.setAccessibleName("Project")

        self.empty = EmptyState(title="No project open", action_text=None)
        self._stack = QStackedLayout()
        holder = QWidget()
        holder.setLayout(self._stack)
        self._stack.addWidget(self.view)
        self._stack.addWidget(self.empty)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(holder)

        self.act_delete = QAction("Delete", self)
        self.act_delete.setShortcut(QKeySequence(QKeySequence.StandardKey.Delete))
        self.act_delete.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.act_delete.triggered.connect(self.delete_selected)
        self.view.addAction(self.act_delete)

        state.projectChanged.connect(self.rebuild)
        state.designsChanged.connect(self.rebuild)
        state.simulationsChanged.connect(self.rebuild)
        state.currentDesignChanged.connect(lambda _id: self.sync_selection())
        state.currentSimulationChanged.connect(lambda _id: self.sync_selection())
        theme.changed.connect(self._on_theme)    # bound method: auto-disconnects when deleted
        if jobs is not None:
            self.connect_jobs(jobs)
        self.rebuild()

    def _on_theme(self, _name: str) -> None:
        self.rebuild()

    # -- jobs -------------------------------------------------------------------
    def connect_jobs(self, jobs) -> None:
        self.jobs = jobs
        for sig, slot in (("jobQueued", self._job_queued), ("jobStarted", self._job_started),
                          ("jobProgress", self._job_progress), ("jobFinished", self._job_ended),
                          ("jobFailed", self._job_ended), ("jobCancelled", self._job_ended)):
            s = getattr(jobs, sig, None)
            if s is not None:
                s.connect(slot)

    def _job_queued(self, sim_id: str) -> None:
        self._set_sim_status(sim_id, "queued")

    def _job_started(self, sim_id: str) -> None:
        self._progress[sim_id] = 0.0
        self._set_sim_status(sim_id, "running")

    def _job_progress(self, sim_id: str, fraction: float, _message: str = "") -> None:
        old = self._progress.get(sim_id)
        self._progress[sim_id] = float(fraction)
        if old is None or int(old * 100) != int(fraction * 100):
            self._set_sim_status(sim_id, "running")

    def _job_ended(self, sim_id: str, *_args) -> None:
        self._progress.pop(sim_id, None)
        sim = self._sim(sim_id)
        if sim is not None:
            self._set_sim_status(sim_id, sim.status)

    # -- model ------------------------------------------------------------------
    def _sim(self, sim_id: str):
        p = self.state.project
        return p.simulations.get(sim_id) if p is not None else None

    def _design_of(self, sim):
        p = self.state.project
        return p.designs.get(sim.design_id) if p is not None and sim is not None else None

    def _effective_status(self, sim) -> str:
        jobs = self.jobs
        if jobs is not None:
            with contextlib.suppress(Exception):     # a fake / partial job manager
                if jobs.running_id() == sim.id:
                    return "running"
                if sim.id in jobs.queued_ids():
                    return "queued"
        if sim.id in self._progress:
            return "running"
        return sim.status if sim.status in _STATUS_TEXT else "new"

    def _decorate_sim(self, item: QStandardItem, sim, status: str | None = None) -> None:
        status = status or self._effective_status(sim)
        if status == "running":
            frac = self._progress.get(sim.id, 0.0)
            item.setIcon(_progress_icon(frac))
            suffix = f"{round(100 * frac)} %"
        else:
            name, colour = _STATUS_ICON.get(status, _STATUS_ICON["new"])
            item.setIcon(theme.icon(name, colour))
            suffix = {"queued": "queued", "failed": "failed", "cancelled": "stopped"}.get(status, "")
        item.setData(suffix, SUFFIX_ROLE)
        item.setData(status, STATUS_ROLE)
        design = self._design_of(sim)
        # No analysis tag on simulation rows: the design row above says 3D / Wing, the default names
        # carry the analysis ("Polar · Re 1.0×10⁶", "CFD 3D · 30 m/s"), and in a 185 px dock the tag
        # left "CFD ..." of the name. The analysis is in the tooltip.
        item.setData("", TAG_ROLE)
        kind = f"{analysis_label(sim, design)} · {analysis_description(sim, design)}"
        tip = f"{sim.name}\n{kind} · {_STATUS_TEXT.get(status, status)}"
        if status == "failed" and getattr(sim, "error", None):
            tip += f"\n{sim.error}"
        item.setToolTip(tip)

    def _set_sim_status(self, sim_id: str, status: str) -> None:
        item = self._items.get(sim_id)
        sim = self._sim(sim_id)
        if item is None or sim is None:
            return
        self._building += 1
        try:
            self._decorate_sim(item, sim, status)
        finally:
            self._building -= 1
        idx = item.index()
        self.view.update(idx)

    def rebuild(self) -> None:
        """Rebuild the model from the project, keeping collapsed designs and the selection."""
        proj = self.state.project
        self._building += 1
        try:
            self.model.clear()
            self._items = {}
            if proj is None or not proj.designs:
                if proj is None:
                    self.empty.set_text("No project open", "")
                else:
                    self.empty.set_text("No designs", "Import a 2D outline or a 3D mesh, "
                                                      "or start from a template or a wing.")
                self._stack.setCurrentWidget(self.empty)
                return
            self._stack.setCurrentWidget(self.view)
            root = self.model.invisibleRootItem()
            for did, d in proj.designs.items():
                it = QStandardItem(d.name)
                it.setData(did, ID_ROLE)
                it.setData("design", KIND_ROLE)
                it.setIcon(theme.icon(design_icon(d), "text2"))
                it.setEditable(True)
                tag = design_tag(d)
                it.setData("" if tag == "2D" else tag, TAG_ROLE)
                n = sum(1 for s in proj.simulations.values() if s.design_id == did)
                it.setToolTip(f"{d.name}\n{design_description(d)}\n{n} simulation{'s' if n != 1 else ''}")
                root.appendRow(it)
                self._items[did] = it
                for sid, s in proj.simulations.items():
                    if s.design_id != did:
                        continue
                    si = QStandardItem(s.name)
                    si.setData(sid, ID_ROLE)
                    si.setData("sim", KIND_ROLE)
                    si.setEditable(True)
                    self._decorate_sim(si, s)
                    it.appendRow(si)
                    self._items[sid] = si
            self._add_experiments(root, proj)
        finally:
            self._building -= 1
        collapsed = set(self._collapsed)
        for did, it in self._items.items():
            if it.data(KIND_ROLE) in ("design", "experiments") and did not in collapsed:
                self.view.setExpanded(it.index(), True)
        self._collapsed = collapsed & set(self._items)
        self.sync_selection()

    # -- experiments ----------------------------------------------------------------
    def experiments(self) -> list[dict]:
        """The experiment logs of the open project, newest first: path, goal, status, created, model."""
        proj = self.state.project
        folder = Path(getattr(proj, "path", "") or "") / "experiments" if proj is not None else None
        if folder is None or not folder.is_dir():
            return []
        out = []
        for d in sorted(folder.iterdir(), reverse=True):
            f = d / "experiment.json"
            try:
                stamp = f.stat().st_mtime_ns
            except OSError:
                continue
            cached = self._exp_cache.get(str(f))
            if cached is None or cached[0] != stamp:
                try:
                    raw = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(raw, dict):
                    continue
                info = {"path": str(d), "goal": str(raw.get("goal", "")).strip(), "status": str(raw.get("status", "")),
                        "created": str(raw.get("created", "")), "model": str(raw.get("model", "")),
                        "actions": len(raw.get("actions") or [])}
                cached = (stamp, info)
                self._exp_cache[str(f)] = cached
            out.append(cached[1])
        return out

    def _add_experiments(self, root: QStandardItem, proj) -> None:
        exps = self.experiments()
        if not exps:
            return
        group = QStandardItem("Experiments")
        group.setData(EXPERIMENTS_GROUP, ID_ROLE)
        group.setData("experiments", KIND_ROLE)
        group.setIcon(theme.icon("flask", "text2"))
        group.setEditable(False)
        group.setData(str(len(exps)), TAG_ROLE)
        group.setToolTip("Experiments run by the assistant (the project's experiments folder)")
        root.appendRow(group)
        self._items[EXPERIMENTS_GROUP] = group
        for e in exps:
            first = e["goal"].splitlines()[0] if e["goal"] else Path(e["path"]).name
            it = QStandardItem(first if len(first) <= 60 else first[:59].rstrip() + "…")
            it.setData(e["path"], ID_ROLE)
            it.setData("experiment", KIND_ROLE)
            it.setEditable(False)
            status = e["status"]
            # "unverified": the answer claimed an action no tool call performed and was withheld (ai/agent.py)
            glyph = {"failed": "error", "stopped": "stop", "unverified": "warning"}.get(status, "flask")
            it.setIcon(theme.icon(glyph, {"failed": "bad", "stopped": "warn", "unverified": "warn"}.get(status, "text2")))
            it.setData(_EXP_SUFFIX.get(status, ""), SUFFIX_ROLE)
            it.setData(status, STATUS_ROLE)
            # No date tag in the row: in a 180 px tree "running" plus "11 Oct 18:50" left no room for the
            # goal at all. The date is in the tooltip; the folder names sort by it.
            when = ""
            with contextlib.suppress(ValueError):
                when = datetime.fromisoformat(e["created"]).strftime("%d %b %Y %H:%M")
            it.setToolTip(f"{e['goal'][:400]}\n{when + ' · ' if when else ''}{e['model']} · {e['actions']} "
                          f"action{'s' if e['actions'] != 1 else ''} · {status or 'unknown'}\n"
                          "Double-click to open the log")
            group.appendRow(it)
            self._items[e["path"]] = it

    def refresh_experiments(self) -> None:
        """Re-read the experiment logs (the assistant created or updated one)."""
        self.rebuild()

    def open_experiment(self, path: str) -> None:
        self.experimentActivated.emit(str(path))

    def delete_experiment(self, path: str) -> bool:
        """Delete an experiment log folder after confirmation (its simulations stay in the project)."""
        p = Path(path)
        proj = self.state.project
        exp_root = (Path(proj.path) / "experiments").resolve() if proj is not None else None
        try:
            ok_place = exp_root is not None and p.resolve().parent == exp_root
        except OSError:
            ok_place = False
        if not ok_place or not p.is_dir():
            return False
        if not self._confirm("Delete experiment log",
                             f"Delete the log of “{p.name}”? Its designs and simulations stay in the project."):
            return False
        from ...project.fileio import remove_tree
        remove_tree(p)
        self._exp_cache.pop(str(p / "experiment.json"), None)
        self.rebuild()
        self.state.log(f"Deleted experiment log “{p.name}”", "info", "Project")
        return True

    # -- selection ----------------------------------------------------------------
    def sync_selection(self) -> None:
        """Select the item matching AppState (simulation first, else design)."""
        target = self._items.get(self.state.simulation_id) or self._items.get(self.state.design_id)
        sel = self.view.selectionModel()
        self._syncing = True
        try:
            if target is None:
                sel.clearSelection()
                sel.setCurrentIndex(QModelIndex(), sel.SelectionFlag.Clear)
            else:
                idx = target.index()
                if idx.parent().isValid():
                    self.view.setExpanded(idx.parent(), True)
                sel.setCurrentIndex(idx, sel.SelectionFlag.ClearAndSelect)
                self.view.scrollTo(idx)
        finally:
            self._syncing = False

    def _current_changed(self, cur: QModelIndex, _prev: QModelIndex) -> None:
        # While the model is rebuilt, Qt moves the current index to a neighbour of each row it
        # removes; acting on those moves made any design edit (a rename, a 3D transform) select
        # another design. Only the user's moves, and sync_selection's, count.
        if self._syncing or self._building or not cur.isValid():
            return
        kind, oid = cur.data(KIND_ROLE), cur.data(ID_ROLE)
        if kind in ("experiment", "experiments"):
            return                      # logs are files, not part of the AppState selection
        self._syncing = True
        try:
            if kind == "sim":
                self.state.select_simulation(oid)
            elif kind == "design":
                self.select_design(oid)
        finally:
            self._syncing = False

    def select_design(self, design_id: str) -> None:
        """Make a design current together with one of its own simulations (or none).

        ``AppState.select_design`` alone would keep another design's simulation
        current, and Run would then run that one.
        """
        self.state.select_design(design_id)
        sim = self.state.simulation
        if sim is None or sim.design_id != design_id:
            proj = self.state.project
            first = ""
            if proj is not None:
                first = next((s.id for s in proj.simulations.values() if s.design_id == design_id), "")
            self.state.select_simulation(first)

    def selected(self) -> tuple[str, str]:
        """(kind, id) of the current item, or ("", "")."""
        idx = self.view.currentIndex()
        if not idx.isValid():
            return "", ""
        return idx.data(KIND_ROLE) or "", idx.data(ID_ROLE) or ""

    def item(self, oid: str) -> QStandardItem | None:
        return self._items.get(oid)

    # -- actions --------------------------------------------------------------------
    def _double_clicked(self, idx: QModelIndex) -> None:
        kind, oid = idx.data(KIND_ROLE), idx.data(ID_ROLE)
        if kind == "experiment":
            self.open_experiment(oid)
        elif kind == "design":
            self.state.navigate("geometry")
        elif kind == "sim":
            sim = self._sim(oid)
            self.state.navigate("results" if sim is not None and sim.status in ("done", "cancelled")
                                else "setup")

    def _item_changed(self, item: QStandardItem) -> None:
        """Inline rename finished: write the new name through AppState (deferred).

        Writing through AppState rebuilds the model, which deletes ``item``; doing
        that inside the item's own change notification (still on the editor's
        commit path) would leave Qt holding dangling pointers, so the write is
        queued for the next event-loop turn.
        """
        if self.state.project is None or self._building:
            return
        oid, kind, name = item.data(ID_ROLE), item.data(KIND_ROLE), item.text().strip()
        QTimer.singleShot(0, self, lambda: self.apply_rename(oid, kind, name))

    def apply_rename(self, oid: str, kind: str, name: str) -> bool:
        """Rename a design or simulation; a blank or unchanged name restores the old one."""
        proj = self.state.project
        if proj is None:
            return False
        obj = proj.designs.get(oid) if kind == "design" else proj.simulations.get(oid)
        if obj is None:
            return False
        if not name or name == obj.name:
            it = self._items.get(oid)
            if it is not None and it.text() != obj.name:
                self._building += 1
                try:
                    it.setText(obj.name)
                finally:
                    self._building -= 1
            return False
        obj.name = name
        if kind == "design":
            self.state.update_design(obj)
        else:
            self.state.update_simulation(obj)
        self.state.log(f"Renamed to “{name}”", "info", "Project")
        return True

    def rename(self, oid: str) -> None:
        it = self._items.get(oid)
        if it is not None:
            self.view.setCurrentIndex(it.index())
            self.view.edit(it.index())
            self.renameRequested.emit(oid)

    def new_simulation(self, design_id: str, analysis: str | None = None) -> None:
        """A new simulation of a design, then Setup.

        Without ``analysis`` it is Setup's own "New simulation" (one code path
        with the page's button). With one ("wing", "cfd") the simulation is
        made here with that analysis: a wing offers both its lattice polar and
        a 3D CFD run of the lofted wing, which Setup's default cannot choose.
        """
        self.select_design(design_id)
        if analysis is None:
            self.state.navigate("setup:new")
            return
        try:
            sim = self.state.new_simulation(design_id, analysis)
        except ValueError as exc:
            self.state.log(str(exc), "error", "Project")
            return
        self.state.log(f"Added simulation “{sim.name}”", "info", "Project")
        self.state.navigate("setup")

    def run(self, sim_id: str) -> None:
        self.state.select_simulation(sim_id)
        self.state.navigate("simulate:run")

    def stop(self, sim_id: str) -> None:
        if self.jobs is not None:
            self.jobs.cancel(sim_id)

    def duplicate(self, oid: str) -> str:
        """Duplicate a design (shape only) or a simulation (setup, not result); returns the new id."""
        proj = self.state.project
        if proj is None:
            return ""
        if oid in proj.simulations:
            dup = proj.duplicate_simulation(oid)
            self.state.save()
            self.state.simulationsChanged.emit()
            self.state.select_simulation(dup.id)
            self.state.log(f"Duplicated simulation as “{dup.name}”", "info", "Project")
            return dup.id
        if oid in proj.designs:
            from ...model import Design, new_id, now_iso
            src = proj.designs[oid]
            d = Design.from_dict(src.to_dict())
            # to_dict leaves the mesh arrays out (they live in designs/<id>.npz): copy them over,
            # or the duplicate of a 3D body is a design with empty meshes
            d.attach_mesh_arrays({k: v.copy() for k, v in src.mesh_arrays().items()})
            d.id, d.created = new_id(), now_iso()
            d.name = f"{src.name} (copy)"
            self.state.add_design(d, select=True)
            self.state.select_simulation("")
            self.state.log(f"Duplicated design as “{d.name}”", "info", "Project")
            return d.id
        return ""

    def _busy(self, sim_id: str) -> bool:
        if self.jobs is None:
            return False
        try:
            return bool(self.jobs.running_id() == sim_id or sim_id in self.jobs.queued_ids())
        except Exception:  # noqa: BLE001
            return False

    def _confirm(self, title: str, text: str) -> bool:
        """Ask a yes/no question (tests monkeypatch this)."""
        ans = QMessageBox.question(self, title, text,
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        return ans == QMessageBox.StandardButton.Yes

    def delete(self, oid: str) -> bool:
        """Delete a design (with its simulations) or a simulation, after confirmation."""
        proj = self.state.project
        if proj is None:
            return False
        it = self._items.get(oid)
        if it is not None and it.data(KIND_ROLE) == "experiment":
            return self.delete_experiment(oid)
        if oid in proj.simulations:
            sim = proj.simulations[oid]
            if self._busy(oid):
                self.state.message("Stop the simulation before deleting it.", 5000)
                return False
            if not self._confirm("Delete simulation", f"Delete “{sim.name}” and its results?"):
                return False
            self.state.remove_simulation(oid)
            self.state.log(f"Deleted simulation “{sim.name}”", "info", "Project")
            return True
        if oid in proj.designs:
            d = proj.designs[oid]
            sims = [s for s in proj.simulations.values() if s.design_id == oid]
            if any(self._busy(s.id) for s in sims):
                self.state.message("Stop its running simulation before deleting this design.", 5000)
                return False
            extra = f" and its {len(sims)} simulation{'s' if len(sims) != 1 else ''}" if sims else ""
            if not self._confirm("Delete design", f"Delete “{d.name}”{extra}?"):
                return False
            self.state.remove_design(oid)
            self.state.log(f"Deleted design “{d.name}”{extra}", "info", "Project")
            return True
        return False

    def delete_selected(self) -> None:
        _kind, oid = self.selected()
        if oid:
            self.delete(oid)

    # -- context menu -----------------------------------------------------------------
    def build_menu(self, idx: QModelIndex) -> QMenu:
        """The context menu for an index (invalid index = empty area)."""
        m = QMenu(self)
        proj = self.state.project
        if proj is None:
            return m
        kind, oid = (idx.data(KIND_ROLE), idx.data(ID_ROLE)) if idx.isValid() else ("", "")
        if kind == "experiment":
            m.addAction(theme.icon("book"), "Open log", lambda: self.open_experiment(oid))
            m.addAction(theme.icon("folder"), "Show in folder",
                        lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(oid))))
            m.addSeparator()
            m.addAction(theme.icon("trash"), "Delete log…", lambda: self.delete_experiment(oid))
            return m
        if kind == "experiments":
            folder = Path(proj.path) / "experiments"
            m.addAction(theme.icon("folder"), "Show in folder",
                        lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))
            return m
        if kind == "design":
            d = proj.designs.get(oid)
            if is_wing(d):
                m.addAction(theme.icon("wing"), "New wing analysis", lambda: self.new_simulation(oid, "wing"))
                m.addAction(theme.icon("cube"), "New 3D CFD", lambda: self.new_simulation(oid, "cfd"))
            elif design_tag(d) == "3D":
                m.addAction(theme.icon("cube"), "New 3D CFD", lambda: self.new_simulation(oid))
            else:
                m.addAction(theme.icon("plus"), "New simulation", lambda: self.new_simulation(oid))
            m.addSeparator()
            m.addAction(theme.icon("geometry"), "Open in Geometry",
                        lambda: (self.select_design(oid), self.state.navigate("geometry")))
            a = m.addAction(theme.icon("edit"), "Rename", lambda: self.rename(oid))
            a.setShortcut(QKeySequence("F2"))
            m.addAction(theme.icon("copy"), "Duplicate", lambda: self.duplicate(oid))
            m.addSeparator()
            a = m.addAction(theme.icon("trash"), "Delete", lambda: self.delete(oid))
            a.setShortcut(QKeySequence(QKeySequence.StandardKey.Delete))
        elif kind == "sim":
            sim = proj.simulations.get(oid)
            busy = self._busy(oid)
            if busy:
                m.addAction(theme.icon("stop"), "Stop", lambda: self.stop(oid))
            else:
                a = m.addAction(theme.icon("run"), "Run", lambda: self.run(oid))
                a.setEnabled(self.jobs is not None)
            m.addSeparator()
            m.addAction(theme.icon("setup"), "Open setup",
                        lambda: (self.state.select_simulation(oid), self.state.navigate("setup")))
            a = m.addAction(theme.icon("results"), "Show results",
                            lambda: (self.state.select_simulation(oid), self.state.navigate("results")))
            a.setEnabled(sim is not None and sim.status in ("done", "cancelled"))
            m.addSeparator()
            a = m.addAction(theme.icon("edit"), "Rename", lambda: self.rename(oid))
            a.setShortcut(QKeySequence("F2"))
            m.addAction(theme.icon("copy"), "Duplicate", lambda: self.duplicate(oid))
            m.addSeparator()
            a = m.addAction(theme.icon("trash"), "Delete", lambda: self.delete(oid))
            a.setShortcut(QKeySequence(QKeySequence.StandardKey.Delete))
            a.setEnabled(not busy)
        else:
            m.addAction(theme.icon("import"), "Import 2D outline…",
                        lambda: self.state.navigate("geometry:import"))
            m.addAction(theme.icon("cube"), "Import 3D mesh…", lambda: self.state.request("import-mesh"))
            m.addSeparator()
            m.addAction(theme.icon("layers"), "New design from template…",
                        lambda: self.state.navigate("geometry:templates"))
            m.addAction(theme.icon("wing"), "New wing", lambda: self.state.request("new-wing"))
        return m

    def _context_menu(self, pos: QPoint) -> None:
        idx = self.view.indexAt(pos)
        if idx.isValid():
            self.view.setCurrentIndex(idx)
        m = self.build_menu(idx)
        if not m.isEmpty():
            m.exec(self.view.viewport().mapToGlobal(pos))


__all__ = ["EXPERIMENTS_GROUP", "ID_ROLE", "KIND_ROLE", "STATUS_ROLE", "SUFFIX_ROLE", "TAG_ROLE", "ProjectTree"]
