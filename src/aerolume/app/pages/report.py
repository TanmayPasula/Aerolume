"""Report page: choose simulations and sections, generate a PDF / HTML report, open it.

A report is the deliverable most users actually hand on, so the page is a plain
form (docs/ui_guidelines.md section 4) with sensible defaults - every simulation
that has a result, every section that applies, PDF - next to the list of the
files already generated. The form is a :class:`PropertyGrid` like every other
inspector in the app (Simulations, Sections, Document), so it reads the same
way as Setup.

Generation runs on a worker thread through :func:`aerolume.report.generate`
(rendering a dozen matplotlib figures takes seconds). The list is read from the
project's ``reports/`` folder rather than remembered, so reports made earlier,
from the command line or copied in by hand appear too.

Every kind of simulation can be reported: 2D and 3D Navier-Stokes runs,
airfoil polars and finite-wing sweeps. The list names the kind of each; a
section is offered only when a selected simulation has data for it (the flow
fields and convergence sections need a Navier-Stokes run, the polar section a
polar, a wing section a wing sweep), and the comparison only when two selected
results are of the same kind (2D, 3D CFD or wing), because results of
different kinds are never compared. The placed-physics sections (DESIGN.md 10:
heat transfer, flow visualisation, fans and jets) follow the simulations'
settings, and grid convergence needs three runs of one set-up at different
resolutions (:func:`grid_study_possible`); a section that does not apply is
disabled with the reason in its tooltip. The report module owns the list and
the headings (``aerolume.report.SECTIONS`` / ``SECTION_TITLES``), so a section
it adds appears here without a change to this page.
"""
from __future__ import annotations

import importlib
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..jobs import JobManager, run_in_background
from ..widgets.common import PropertyGrid, ToolStrip
from ..widgets.kpi import DataTable, EmptyState, StatusLabel, button, label
from ..widgets.plots import import_error, lazy_import
from .results import sim_kind

DEFAULT_SECTIONS = ("summary", "geometry", "setup", "results", "surface", "fields", "heat", "flowvis", "fans",
                    "convergence", "grid", "sensitivity", "polar", "comparison", "methodology")
SECTION_TEXT = {
    "summary": ("Summary", "Headline numbers and conclusions"),
    "geometry": ("Geometry", "Outlines and reference dimensions"),
    "setup": ("Setup", "Flow conditions, grid and solver settings"),
    "results": ("Results", "Forces and coefficients with uncertainty"),
    "surface": ("Surface pressure", "Cp and skin friction along the shape"),
    "fields": ("Flow fields", "Speed, pressure and vorticity maps"),
    "heat": ("Heat transfer", "Nusselt numbers with their uncertainty, the heat balance and the temperature field"),
    "flowvis": ("Flow visualisation", "The emitters, their smoke channels and particles"),
    "fans": ("Fans and jets", "What each fan delivered against momentum theory; jets and vents"),
    "convergence": ("Convergence", "Force histories and averaging window"),
    "grid": ("Grid convergence", "Observed order, Richardson extrapolation and GCI of a grid study"),
    "sensitivity": ("Sensitivity", "Where a shape change has the largest effect"),
    "polar": ("Polars", "Airfoil and wing polars: lift, drag and moment over angle"),
    "comparison": ("Comparison", "Variants against the baseline"),
    "methodology": ("Methodology", "Methods, validity and references"),
}
REPORT_HEADERS = ["File", "Format", "Modified", "Size"]
LIST_ROW_H = 22
#: How the list names each kind of simulation (results.sim_kind)
KIND_TEXT = {"polar": "airfoil polar", "cfd": "Navier–Stokes 2D", "cfd3d": "Navier–Stokes 3D", "wing": "wing sweep"}
#: Comparison groups: results of different groups are never compared (Compare page)
_GROUP = {"cfd": "2d", "polar": "2d", "cfd3d": "cfd3d", "wing": "wing"}


def grid_study_possible(sims) -> bool:
    """Whether three or more of ``sims`` can form a grid study (the report's own grouping,
    :func:`aerolume.analysis.gci.find_grid_studies`: one design, one set-up apart from the
    resolution settings, three or more resolutions), judged from the simulations alone: the
    report decides finally from the results (their cells per reference length)."""
    gci = lazy_import("aerolume.analysis.gci")
    key_of = getattr(gci, "_setup_key", None) if gci is not None else None
    res_keys = tuple(getattr(gci, "RESOLUTION_SETTINGS", ("quality", "cells_per_ref")))
    groups: dict = {}
    for s in sims:
        key = key_of(s) if callable(key_of) else None
        if key is None:
            continue
        cfd = getattr(s, "cfd", None)
        res = tuple(str(getattr(cfd, k, None)) for k in res_keys if k in ("quality", "cells_per_ref"))
        groups.setdefault((str(getattr(s, "design_id", "")), key), set()).add(res)
    return any(len(v) >= 3 for v in groups.values())


def slug(text: str) -> str:
    """File-name-safe version of a title (ASCII: reports travel by e-mail and zip)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    return s[:60] or "report"


def default_title(project_name: str) -> str:
    """The report title offered by default: the project's name.

    A project is usually already named for what it studies ("Wing and sedan
    study"), so "<name> — aerodynamic study" repeated words. Only a name that
    says nothing (empty or the stock "Untitled") gets a descriptive suffix.
    """
    name = (project_name or "").strip()
    if not name or name.lower().startswith("untitled"):
        return f"{name or 'Untitled'} — aerodynamic report"
    return name


def human_size(n: int) -> str:
    """'812 B', '1.4 MB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


class ReportPage(QWidget):
    """Report builder (DESIGN.md 5.8). ``handle_action("generate")`` starts a report with the current choices."""

    def __init__(self, state, jobs: JobManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.jobs = jobs
        self._call = None
        self._started = 0.0
        self._gl: dict | None = None            # viz3d.backend_info(), probed once a 3D case is selected
        self.last_path: Path | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)
        self._build()
        self._connect()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)
        self.empty = EmptyState()
        self.stack.addWidget(self.empty)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(1)
        self.stack.addWidget(split)

        # left: the form
        form_host = QFrame()
        form_host.setProperty("panel", "true")
        form_host.setMinimumWidth(380)
        form_host.setMaximumWidth(620)
        fl = QVBoxLayout(form_host)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(0)
        grid = self.grid = PropertyGrid(label_width=88)

        sims = grid.add_group("Simulations")
        self.sim_list = QListWidget()
        self.sim_list.setProperty("flat", "true")
        self.sim_list.setToolTip("Simulations with a result. Two or more add a comparison section.")
        sims.add_widget(self.sim_list)
        links = QWidget()
        lr = QHBoxLayout(links)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.setSpacing(12)
        self.all_btn = button("Select all", "link")
        self.none_btn = button("Clear", "link")
        lr.addWidget(self.all_btn)
        lr.addWidget(self.none_btn)
        lr.addStretch(1)
        sims.add_widget(links)

        sec = grid.add_group("Sections")
        self._boxes_host = QWidget()
        self._boxes_grid = QGridLayout(self._boxes_host)
        self._boxes_grid.setContentsMargins(0, 0, 0, 0)
        self._boxes_grid.setHorizontalSpacing(16)
        self._boxes_grid.setVerticalSpacing(4)
        self.section_boxes: dict[str, QCheckBox] = {}
        # What the user asked for per section. A section with no data is shown disabled AND
        # unchecked (a greyed tick still reads "this will be in the report"); the wish is kept
        # here so the tick comes back when a selected simulation has the data again.
        self._section_wish: dict[str, bool] = {}
        # The report module imports matplotlib (about a second): read it only if it is already
        # loaded, and load it off the GUI thread otherwise, so opening this page is instant.
        rep = sys.modules.get("aerolume.report")
        self._fill_sections(getattr(rep, "SECTIONS", None), getattr(rep, "SECTION_TITLES", None))
        if rep is None:
            run_in_background(importlib.import_module, "aerolume.report",
                              on_done=self._report_module_loaded, on_error=lambda _m: None)
        sec.add_widget(self._boxes_host)
        # 3D figures are off-screen VTK renders; a machine without OpenGL still gets the report
        # (each such figure becomes a sentence giving the reason), and the form says so beforehand
        self.gl_note = label("", "muted", wrap=True)
        self.gl_note.hide()
        sec.add_widget(self.gl_note)

        doc = grid.add_group("Document")
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Report title")
        self.title_edit.textEdited.connect(lambda *_: setattr(self, "_auto_title", ""))
        self._auto_title = ""                   # the title we filled in; replaced while unedited
        doc.add_row("Title", self.title_edit)
        self.author_edit = QLineEdit()
        self.author_edit.setPlaceholderText("Optional")
        doc.add_row("Author", self.author_edit)
        fmt = QWidget()
        fr = QHBoxLayout(fmt)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.setSpacing(16)
        self.pdf_radio = QRadioButton("PDF")
        self.html_radio = QRadioButton("HTML")
        self.pdf_radio.setChecked(True)
        self.fmt_group = QButtonGroup(self)
        self.fmt_group.addButton(self.pdf_radio)
        self.fmt_group.addButton(self.html_radio)
        fr.addWidget(self.pdf_radio)
        fr.addWidget(self.html_radio)
        fr.addStretch(1)
        doc.add_row("Format", fmt, tooltip="PDF for sending and printing; HTML opens in any browser")

        act = QWidget()
        al = QVBoxLayout(act)
        al.setContentsMargins(16, 8, 0, 0)
        al.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.generate_btn = button("Generate", "primary", icon="report",
                                   tooltip="Write the report into the project's reports folder")
        row.addWidget(self.generate_btn)
        self.state_badge = StatusLabel("", None)
        row.addWidget(self.state_badge)
        row.addStretch(1)
        al.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 0)
        self.progress.hide()
        al.addWidget(self.progress)
        self.status = label("", "muted", wrap=True)
        al.addWidget(self.status)
        grid.add_widget(act)
        grid.add_stretch()
        fl.addWidget(grid.scrollable())
        split.addWidget(form_host)

        # right: generated files
        files = QWidget()
        rl = QVBoxLayout(files)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        strip = ToolStrip()
        strip.add_widget(label("Generated reports", "h3"))
        strip.add_stretch()
        self.open_btn = strip.add_button("open", "Open", "Open the selected report")
        self.folder_btn = strip.add_button("folder", "Show folder", "Open the project's reports folder")
        rl.addWidget(strip)
        self.files_stack = QStackedWidget()
        self.no_reports = label("No reports in this project yet.", "muted")
        self.no_reports.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.no_reports.setContentsMargins(0, 24, 0, 0)
        self.files_stack.addWidget(self.no_reports)
        self.report_list = DataTable(REPORT_HEADERS)
        self.report_list.setProperty("flat", "true")
        self.report_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.files_stack.addWidget(self.report_list)
        rl.addWidget(self.files_stack, 1)
        split.addWidget(files)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([480, 900])

    def _connect(self) -> None:
        st = self.state
        st.projectChanged.connect(self.refresh)
        st.simulationsChanged.connect(self.refresh)
        st.resultChanged.connect(lambda *_: self.refresh())
        self.all_btn.clicked.connect(lambda: self._set_all(True))
        self.none_btn.clicked.connect(lambda: self._set_all(False))
        self.sim_list.itemChanged.connect(lambda *_: self._update_controls())
        self.generate_btn.clicked.connect(self.generate)
        self.open_btn.clicked.connect(self.open_selected)
        self.folder_btn.clicked.connect(self.show_in_folder)
        self.report_list.itemDoubleClicked.connect(lambda *_: self.open_selected())
        self.report_list.itemSelectionChanged.connect(self._update_controls)

    # ------------------------------------------------------------- page API
    def on_shown(self) -> None:
        self.refresh()

    def handle_action(self, action: str) -> None:
        if action == "generate":
            self.generate()

    # -------------------------------------------------------------- helpers
    def _fill_sections(self, sections, titles) -> None:
        """(Re)build the section check boxes, keeping the user's choices for known keys."""
        keys = tuple(sections or DEFAULT_SECTIONS)
        titles = dict(titles or {})                      # the report's own headings
        for cb in self.section_boxes.values():
            self._boxes_grid.removeWidget(cb)
            cb.deleteLater()
        self.section_boxes = {}
        for i, key in enumerate(keys):
            title, tip = SECTION_TEXT.get(key, (key.capitalize(), ""))
            cb = QCheckBox(titles.get(key, title))
            cb.setToolTip(tip)
            cb.setChecked(self._section_wish.get(key, True))
            # any change is a wish except _update_controls' own (made with signals blocked)
            cb.toggled.connect(lambda on, k=key: self._section_wish.__setitem__(k, bool(on)))
            self.section_boxes[key] = cb
            self._boxes_grid.addWidget(cb, i // 2, i % 2)

    def _report_module_loaded(self, rep) -> None:
        """The background import finished: adopt the report's own section list and headings."""
        sections = tuple(getattr(rep, "SECTIONS", None) or DEFAULT_SECTIONS)
        titles = dict(getattr(rep, "SECTION_TITLES", None) or {})
        current = {k: cb.text() for k, cb in self.section_boxes.items()}
        wanted = {k: titles.get(k, SECTION_TEXT.get(k, (k.capitalize(), ""))[0]) for k in sections}
        if current != wanted or tuple(current) != sections:
            self._fill_sections(sections, titles)
            self._update_controls()

    def _sections(self) -> tuple[str, ...]:
        return tuple(self.section_boxes)

    def _reports_dir(self) -> Path | None:
        p = self.state.project
        if p is None:
            return None
        d = getattr(p, "reports_dir", None)
        if d is not None:
            return Path(d)
        base = getattr(p, "path", None)
        return Path(base) / "reports" if base is not None else None

    def _available(self) -> list[str]:
        p = self.state.project
        if p is None:
            return []
        has = getattr(p, "has_result", None)
        return [sid for sid, sim in p.simulations.items()
                if (has(sid) if callable(has) else sim.status in ("done", "cancelled"))
                and sim.status not in ("running", "queued")]

    def selected_ids(self) -> list[str]:
        out = []
        for i in range(self.sim_list.count()):
            it = self.sim_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def _has_sensitivity(self, sim) -> bool:
        """Whether a CFD simulation's result carries a sensitivity map.

        Exact when the result is already loaded (the job manager's cache);
        otherwise the simulation's own "compute sensitivity" setting stands in,
        because loading every result (tens of MB each) to tick a box would make
        this page slow to open.
        """
        peek = getattr(self.jobs, "cached_result", None)
        cached = peek(sim.id) if callable(peek) else None
        if cached is not None:
            return any(getattr(sd, "sensitivity", None) is not None for sd in getattr(cached, "surface", []) or [])
        return bool(getattr(sim.cfd, "compute_sensitivity", False))

    def selected_sections(self) -> list[str]:
        return [k for k, cb in self.section_boxes.items() if cb.isChecked() and cb.isEnabled()]

    def report_paths(self) -> list[Path]:
        """Files in the generated-reports table, newest first."""
        out = []
        for i in range(self.report_list.rowCount()):
            it = self.report_list.item(i, 0)
            if it is not None:
                out.append(Path(it.data(Qt.ItemDataRole.UserRole)))
        return out

    # -------------------------------------------------------------- refresh
    def refresh(self) -> None:
        p = self.state.project
        if p is None:
            self.empty.set_content("No project open", "", "Open or create a project",
                                   lambda: self.state.navigate("start"))
            self.stack.setCurrentIndex(0)
            return
        avail = self._available()
        if not avail:
            self.empty.set_content("No results to report", "A report needs at least one simulation with a result.",
                                   "Open Simulate", lambda: self.state.navigate("simulate"))
            self.stack.setCurrentIndex(0)
            return
        self.stack.setCurrentIndex(1)
        # the project name is the title ("Wing and sedan study"); appending "aerodynamic study"
        # repeated words. It follows the project until the user types their own title.
        if not self.title_edit.text().strip() or self.title_edit.text() == self._auto_title:
            self._auto_title = default_title(p.name)
            self.title_edit.setText(self._auto_title)
        prev = set(self.selected_ids()) if self.sim_list.count() else None
        many_designs = len({p.simulations[s].design_id for s in avail}) > 1
        self.sim_list.blockSignals(True)
        self.sim_list.clear()
        for sid in avail:
            sim = p.simulations[sid]
            d = p.designs.get(sim.design_id)
            kind = KIND_TEXT.get(sim_kind(sim, d), sim.analysis)
            text = f"{sim.name} — {d.name}" if many_designs and d is not None else sim.name
            if sim.status == "cancelled":
                text += " (partial)"
            item = QListWidgetItem(text)
            item.setToolTip(f"{d.name if d else '?'} · {kind}")
            item.setData(Qt.ItemDataRole.UserRole, sid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            checked = (sid in prev) if prev is not None else (
                sid in self.state.compare_ids if self.state.compare_ids else True)
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            self.sim_list.addItem(item)
        self.sim_list.blockSignals(False)
        # show every simulation without an inner scroll bar up to ten rows; the form scrolls beyond
        rows = min(max(self.sim_list.count(), 2), 10)
        self.sim_list.setFixedHeight(rows * LIST_ROW_H + 2 * self.sim_list.frameWidth() + 4)
        self._refresh_reports()
        self._update_controls()

    def _refresh_reports(self) -> None:
        d = self._reports_dir()
        files = []
        if d is not None and d.is_dir():
            files = sorted([f for f in d.iterdir() if f.suffix.lower() in (".pdf", ".html")],
                           key=lambda f: f.stat().st_mtime, reverse=True)
        rows = []
        for f in files:
            st = f.stat()
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))
            rows.append([f.name, f.suffix[1:].upper(), when, human_size(st.st_size)])
        self.report_list.set_data(REPORT_HEADERS, rows, numeric=[3])
        for i, f in enumerate(files):
            it = self.report_list.item(i, 0)
            it.setData(Qt.ItemDataRole.UserRole, str(f))
            it.setToolTip(str(f))
        self.files_stack.setCurrentIndex(1 if files else 0)
        if files:
            self.report_list.selectRow(0)

    def _set_all(self, on: bool) -> None:
        self.sim_list.blockSignals(True)
        for i in range(self.sim_list.count()):
            self.sim_list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.sim_list.blockSignals(False)
        self._update_controls()

    def _update_controls(self) -> None:
        ids = self.selected_ids()
        p = self.state.project
        sims = [p.simulations[s] for s in ids if s in p.simulations] if p is not None else []
        kinds = [sim_kind(s, p.designs.get(s.design_id)) for s in sims] if p is not None else []
        has_polar = "polar" in kinds
        has_wing = "wing" in kinds
        cfd = [s for s in sims if s.analysis == "cfd"]
        groups = [_GROUP.get(k, k) for k in kinds]
        comparable = any(groups.count(g) >= 2 for g in set(groups))
        applies = {"comparison": comparable, "polar": has_polar or has_wing,
                   "fields": bool(cfd), "convergence": bool(cfd),
                   "sensitivity": any(self._has_sensitivity(s) for s in cfd),
                   # placed physics (DESIGN.md 10), from the simulations' own settings: loading every
                   # result to tick a box would make the page slow to open (see _has_sensitivity)
                   "heat": any(bool(getattr(getattr(s, "thermal", None), "enabled", False)) for s in cfd),
                   "flowvis": any(any(getattr(e, "enabled", True) for e in (getattr(s, "emitters", None) or []))
                                  for s in cfd),
                   "fans": any(bool(getattr(s, "fans", None) or getattr(s, "ports", None)) for s in cfd),
                   "grid": grid_study_possible(cfd)}
        for key in self.section_boxes:
            if "wing" in key:                     # the report's wing section(s), whatever their name
                applies[key] = has_wing
        reasons = {"sensitivity": "no selected simulation computed a sensitivity map (Setup › Solver)",
                   "comparison": "select two or more simulations of one kind (2D, 3D CFD or wing)",
                   "heat": "no selected simulation has heat transfer switched on (Setup › Physics)",
                   "flowvis": "no selected simulation has smoke or particle emitters",
                   "fans": "no selected simulation has fans, jets or vents",
                   "grid": ("select three or more runs of one design and set-up that differ only in grid "
                            "resolution (Setup › Solver › Cells per reference length)")}
        for key, cb in self.section_boxes.items():
            ok = applies.get(key, True)
            cb.setEnabled(ok)
            cb.blockSignals(True)
            cb.setChecked(ok and self._section_wish.get(key, True))
            cb.blockSignals(False)
            tip = SECTION_TEXT.get(key, ("", ""))[1]
            why = reasons.get(key, "does not apply to the selected simulations")
            cb.setToolTip(tip if ok else f"{tip} ({why})")
        self._update_gl_note(any(k in ("cfd3d", "wing") for k in kinds))
        busy = self._call is not None
        self.generate_btn.setEnabled(bool(ids) and not busy)
        self.open_btn.setEnabled(bool(self.report_list.selectedItems()))
        self.folder_btn.setEnabled(self._reports_dir() is not None)

    def _update_gl_note(self, three_d: bool) -> None:
        """The note under Sections when 3D figures cannot be rendered here (probed once, on demand:
        the probe loads VTK and opens a tiny context, which a 2D-only project never needs)."""
        if not three_d:
            self.gl_note.hide()
            return
        if self._gl is None:
            viz = lazy_import("aerolume.viz3d")
            try:
                self._gl = dict(viz.backend_info()) if viz is not None else \
                    {"ok": False, "error": import_error("aerolume.viz3d") or "the 3D viewer is not installed"}
            except Exception as exc:  # noqa: BLE001 - a failed probe is the answer: no 3D figures
                self._gl = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if self._gl.get("ok"):
            self.gl_note.hide()
            return
        why = str(self._gl.get("error") or "").strip().rstrip(".")
        self.gl_note.setText("3D figures need an OpenGL context, and none can be created on this machine"
                             + (f" ({why})" if why else "") + ". The report says so in place of each 3D "
                             "figure; tables and 2D figures are complete.")
        self.gl_note.show()

    # ------------------------------------------------------------- generate
    def generate(self) -> None:
        """Generate the report on a worker thread with the current choices."""
        p = self.state.project
        ids = self.selected_ids()
        if p is None or not ids or self._call is not None:
            return
        rep = lazy_import("aerolume.report")
        if rep is None or not hasattr(rep, "generate"):
            reason = import_error("aerolume.report") or "aerolume.report.generate is missing"
            self.state_badge.set_status("Unavailable", "warn")
            self.status.setText(f"Report generation is not available in this installation ({reason}).")
            return
        fmt = "pdf" if self.pdf_radio.isChecked() else "html"
        title = self.title_edit.text().strip() or default_title(p.name)
        author = self.author_edit.text().strip() or None
        folder = self._reports_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.state_badge.set_status("Failed", "bad")
            self.status.setText(f"The reports folder cannot be created: {exc}")
            return
        path = folder / f"{slug(title)}_{time.strftime('%Y%m%d-%H%M%S')}.{fmt}"
        sections = self.selected_sections()
        self._started = time.monotonic()
        self.progress.show()
        self._timer.start()
        self.state_badge.set_status("Generating", "accent")
        self.status.setText(f"Writing {path.name}")
        self.state.log(f"Writing {path.name} ({len(ids)} simulation{'s' if len(ids) != 1 else ''}, "
                       f"{len(sections)} sections)", "info", "Report")
        self._call = run_in_background(rep.generate, p, ids, path, fmt=fmt, sections=sections,
                                       title=title, author=author,
                                       on_done=self._done, on_error=self._failed)
        self._update_controls()

    def _tick(self) -> None:
        self.status.setText(f"Writing the report, {time.monotonic() - self._started:.0f} s")

    def _finish(self) -> None:
        self._call = None
        self._timer.stop()
        self.progress.hide()
        self._update_controls()

    def _done(self, path) -> None:
        self._finish()
        self.last_path = Path(path) if path else None
        self.state_badge.set_status("Ready", "good")
        secs = time.monotonic() - self._started
        name = self.last_path.name if self.last_path else "report"
        self.status.setText(f"Saved {name} in {secs:.1f} s")
        self.state.log(f"Saved {path}", "info", "Report")
        self.state.message(f"Report saved: {path}")
        self._refresh_reports()
        self._update_controls()
        changed = getattr(self.state, "reportsChanged", None)
        if changed is not None:              # the step bar ticks Report once a file exists
            changed.emit()

    def _failed(self, message: str) -> None:
        self._finish()
        self.state_badge.set_status("Failed", "bad")
        self.status.setText(f"The report could not be generated: {message}")
        self.state.log(f"Report failed: {message}", "error", "Report")

    # ----------------------------------------------------------- open files
    def open_selected(self) -> None:
        rows = {i.row() for i in self.report_list.selectedItems()}
        if rows:
            it = self.report_list.item(min(rows), 0)
            if it is not None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.ItemDataRole.UserRole)))

    def show_in_folder(self) -> None:
        d = self._reports_dir()
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))


__all__ = ["ReportPage", "human_size", "slug"]
