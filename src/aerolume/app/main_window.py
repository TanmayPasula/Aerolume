"""The main window: menus, toolbar, step bar, Project and Messages docks, status bar.

Layout (docs/ui_guidelines.md section 3)::

    Menu:     File  Edit  View  Simulation  Tools  Help
    Toolbar:  New Open Save | Import geometry ▾ | Run Stop | Report
    Steps:    Geometry  Setup  Simulate  Results  Compare  Report     3D | design › simulation
    +-----------+--------------------------------------------------------+
    | Project   |  current page (each page owns its right-hand inspector) |
    | (tree)    |                                                        |
    +-----------+--------------------------------------------------------+
    | Messages: time · level · source · text                             |
    +--------------------------------------------------------------------+
    Status:   message | Running NACA 2412 α 4° 43 % | CUDA · RTX 4060 Ti | SI

The Project tree is the selector (there are no design / simulation combos);
the step bar is the workflow. On the Start page (no project, or Help ▸
Welcome page) the docks and the step bar step aside so the welcome reads like
an IDE's, and come back as they were when a workflow page is shown.

Pages are imported and built lazily, the first time they are shown. A page
whose module is missing or raises is replaced by a placeholder that shows the
error and a Retry button, so one broken page can never stop the app from
starting (pages are written in parallel by different people).

Two tool windows sit beside the workflow, hidden until asked for (the
toolbar's Assistant and Script buttons, View menu, Alt+3 / Alt+4): the
**Assistant** dock on the right (``widgets/assistant_panel.py``, the AI
experiment assistant of DESIGN.md section 9: plans, runs and reports experiments
through the scripting API, logged under ``<project>/experiments/``) and the
**Script** dock at the bottom (``widgets/script_console.py``, Python against the
scripting API, section 8). Both are imported when the window is built; a build
whose module is missing or broken shows the reason in the dock instead. Script
opens at 38 % of the window height (as far as the page's minimum allows) and
folds Messages to its title row, since both share the bottom. Messages opens by
itself for errors only; warnings wait in its title row, counted.

Navigation is always ``state.navigate("page")`` or ``"page:action"``: the
window shows the page, calls ``page.on_shown()`` and then
``page.handle_action(action)`` when both exist. Shell actions that belong to
no page go through ``state.request(command)`` (``"import-mesh[:<path>]"``,
``"new-wing"``).

2D and 3D share the window. A project can hold 2D outlines, 3D meshes and
wings side by side, so the shell says which one is on screen instead of
switching modes: the Project tree's glyphs and tags, the step bar's "3D" /
"Wing" breadcrumb tag, "Import geometry" split into a 2D outline and a 3D mesh,
and Simulation menu entries that follow the selected design (a wing offers its
lattice polar and a 3D CFD run of the lofted wing).

**3D mesh import hook.** The units / orientation dialog for STL, OBJ and PLY
files is not part of the shell. It is looked up at call time as
:data:`MESH_IMPORT_HOOK`, ``aerolume.app.dialogs.mesh_import.import_mesh_dialog``::

    import_mesh_dialog(parent: QWidget, state: AppState, *, path: str | None = None)
        -> list[Design] | None

It shows its dialog (asking for the file when ``path`` is None) and returns the
3D designs to add, or None when the user cancels; it does not add them to the
project itself (designs it did add are not added twice). The same function in
``aerolume.app.dialogs.import_dialogs`` is used when ``mesh_import`` does not
exist (:data:`MESH_IMPORT_HOOK_ALTERNATIVES`). When the module is
missing, :meth:`MainWindow.import_mesh` imports the file with the automatic
unit guess as a generic body and says so in Messages, so the 3D path works
end to end in every build.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
import math
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDesktopServices,
    QImage,
    QKeySequence,
    QPixmap,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextFrameFormat,
    QTextTable,
    QTextTableCellFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid as shiboken_valid

from .. import __version__
from . import settings as app_settings
from . import theme
from .pages import start as start_mod
from .state import AppState, analysis_label, design_description, design_tag, is_3d, is_wing
from .widgets import docks as docks_mod
from .widgets.common import (
    CollapsibleSection,
    EmptyState,
    Notice,
    SegmentedControl,
    label,
    name_unnamed_views,
    tie_form_labels,
    tie_label,
    unfocus_scroll_areas,
)
from .widgets.docks import KEY_OF_OBJECT, DockHeader, DockManager
from .widgets.messages import MessagesPanel, format_duration
from .widgets.project_tree import ProjectTree
from .widgets.step_bar import STEP_KEYS, StepBar

_log = logging.getLogger(__name__)


def project_url() -> str | None:
    """The documentation / home page address the installed package declares, if any.

    Help falls back to it when the docs are not on disk (an installed wheel carries no
    docs/ folder). The address comes from the package metadata ([project.urls] in
    pyproject.toml) and is never guessed: a made-up URL would send users to a page
    that may not exist.
    """
    try:
        from importlib.metadata import PackageNotFoundError, metadata
        try:
            entries = metadata("aerolume").get_all("Project-URL") or []
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001 - metadata is optional information
        return None
    urls = {}
    for entry in entries:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in ("documentation", "homepage", "repository", "source"):
        if urls.get(key):
            return urls[key]
    return None


#: "module:function" of the 3D mesh import dialog (see the module docstring for the contract)
MESH_IMPORT_HOOK = "aerolume.app.dialogs.mesh_import:import_mesh_dialog"
#: where else the same function may live (the 2D import dialogs' module), tried in order
MESH_IMPORT_HOOK_ALTERNATIVES = ("aerolume.app.dialogs.import_dialogs:import_mesh_dialog",)


def mesh_import_hook():
    """The 3D mesh import dialog function, or None when this build has none.

    Only a hook module that is itself absent (or has no such function) means
    "none": an import error inside it (a missing dependency, a syntax error)
    is raised, so a broken dialog is reported instead of silently replaced by
    the fallback import.
    """
    for spec in (MESH_IMPORT_HOOK, *MESH_IMPORT_HOOK_ALTERNATIVES):
        mod_name, _, fn = spec.partition(":")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name and mod_name.startswith(exc.name):
                continue
            raise
        hook = getattr(mod, fn, None)
        if callable(hook):
            return hook
    return None


def _accepts_keyword(fn, name: str) -> bool:
    """Whether ``fn`` takes the keyword ``name`` (directly or through ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def format_cells(n: float) -> str:
    """Grid size for the status bar: "524 k cells", "1.25 M cells"."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.3g} M cells"
    if n >= 1e3:
        return f"{n / 1e3:.0f} k cells"
    return f"{n:.0f} cells"


class PageSpec:
    """Static description of one page: key, title, icon, step number (None = not a step)."""

    def __init__(self, key: str, title: str, icon: str, step: int | None, cls: str) -> None:
        self.key, self.title, self.icon, self.step, self.cls = key, title, icon, step, cls


PAGES: tuple[PageSpec, ...] = (
    PageSpec("start", "Welcome", "home", None, "StartPage"),
    PageSpec("geometry", "Geometry", "geometry", 1, "GeometryPage"),
    PageSpec("setup", "Setup", "setup", 2, "SetupPage"),
    PageSpec("simulate", "Simulate", "simulate", 3, "SimulatePage"),
    PageSpec("results", "Results", "results", 4, "ResultsPage"),
    PageSpec("compare", "Compare", "compare", 5, "ComparePage"),
    PageSpec("report", "Report", "report", 6, "ReportPage"),
)
PAGE_KEYS = tuple(p.key for p in PAGES)
_SPEC = {p.key: p for p in PAGES}


# ---------------------------------------------------------------------------
# status bar pieces
# ---------------------------------------------------------------------------

class _DeviceProbe(QObject):
    """Reads the compute device on a worker thread (importing torch takes seconds)."""

    done = Signal(str, str, str)       # kind, short label, tooltip

    def start(self, device: str) -> None:
        threading.Thread(target=self._run, args=(device,), name="aerolume-device-probe", daemon=True).start()

    def _run(self, device: str) -> None:
        result = self.probe(device)
        try:
            self.done.emit(*result)
        except RuntimeError:        # the window closed before torch finished importing
            pass

    @staticmethod
    def probe(device: str) -> tuple[str, str, str]:
        """(kind, status-bar text, tooltip) for a device preference."""
        try:
            from ..core import device as dev
            try:
                info = dev.info(device)
            except RuntimeError:
                info = dev.info("auto")
            return info.kind, device_label(info.kind, info.name), _device_tip(info)
        except Exception as exc:  # noqa: BLE001 - the label must never take the window down
            return "none", "No compute device", f"PyTorch could not be loaded:\n{exc}"


def device_label(kind: str, name: str) -> str:
    """Plain status-bar text: "CUDA · RTX 4060 Ti", "MPS · Apple M2" or "CPU"."""
    if kind == "cpu":
        return "CPU"
    short = name.split(" (")[0].replace("NVIDIA ", "").replace("GeForce ", "")
    return f"{kind.upper()} · {short}"


def _device_tip(info) -> str:
    tip = f"Compute device: {info.name}\nCPU threads: {info.threads}"
    if getattr(info, "memory_gb", None):
        tip += f"\nGPU memory: {info.memory_gb:.1f} GB"
    return tip + "\nChange it in Edit ▸ Preferences."


class RunStatus(QWidget):
    """Status-bar run indicator: "Running <name> 43 %" with a thin progress bar.

    While a CFD run streams frames it adds the grid size (from the plan the
    run uses) and the solver throughput: "1.25 M cells · 61 steps/s". For a 3D
    run those two numbers are what tells a user whether the next minutes are
    well spent, and they are read the same way in 2D. Wing and polar sweeps
    have neither and show the runner's own message.

    Hidden while idle. When a run ends it states the outcome in plain words
    ("Done in 3 min 24 s", "Failed — see Messages", "Stopped") for a while,
    so someone who looked away still sees what happened.
    """

    clicked = Signal()

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(8)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(90)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        self.text = QLabel("")
        self.text.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.text)
        self.setToolTip("Simulation queue - click to open Simulate")
        self._running: str | None = None
        self._fraction = 0.0
        self._message = ""
        self._started = 0.0
        self._queued: list[str] = []
        self._jobs = None
        self._cells: dict[str, int] = {}          # sim id -> cells of the plan it runs
        self._rate: dict[str, float] = {}         # sim id -> latest solver steps per second
        self._linger = QTimer(self)
        self._linger.setSingleShot(True)
        self._linger.timeout.connect(self._refresh)
        self.setVisible(False)

    def _name(self, sim_id: str) -> str:
        proj = self.state.project
        sim = proj.simulations.get(sim_id) if proj is not None else None
        return sim.name if sim is not None else "Simulation"

    def connect_jobs(self, jobs) -> None:
        self._jobs = jobs
        for sig, slot in (("jobQueued", self._on_queued), ("jobStarted", self._on_started),
                          ("jobProgress", self._on_progress), ("jobFinished", self._on_finished),
                          ("jobFailed", self._on_failed), ("jobCancelled", self._on_cancelled),
                          ("jobPlanned", self._on_planned), ("jobFrame", self._on_frame)):
            s = getattr(jobs, sig, None)
            if s is not None:
                s.connect(slot)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    # -- slots ----------------------------------------------------------------
    def _on_queued(self, sim_id: str) -> None:
        if sim_id not in self._queued and sim_id != self._running:
            self._queued.append(sim_id)
        self._refresh()

    def _on_started(self, sim_id: str) -> None:
        if sim_id != self._running:
            self._rate.pop(sim_id, None)          # a re-run starts with no throughput yet
        self._running, self._fraction, self._message = sim_id, 0.0, ""
        self._started = time.monotonic()
        if sim_id in self._queued:
            self._queued.remove(sim_id)
        self._refresh()

    def _on_planned(self, sim_id: str, plan) -> None:
        cells = getattr(plan, "cells", None)
        if isinstance(cells, (int, float)) and cells > 0:
            self._cells[sim_id] = int(cells)
            if sim_id == self._running:
                self._refresh()

    def _on_frame(self, sim_id: str, frame) -> None:
        rate = getattr(frame, "steps_per_s", None)
        if isinstance(rate, (int, float)) and rate > 0:
            old = self._rate.get(sim_id)
            self._rate[sim_id] = float(rate)
            # frames arrive at up to 30 Hz: repaint only when the shown number changes
            if sim_id == self._running and (old is None or round(old) != round(rate)):
                self._refresh()

    def metrics(self, sim_id: str) -> str:
        """"1.25 M cells · 61 steps/s" for a run (each part only once known)."""
        parts = []
        if sim_id in self._cells:
            parts.append(format_cells(self._cells[sim_id]))
        if sim_id in self._rate:
            r = self._rate[sim_id]
            parts.append(f"{r:.1f} steps/s" if r < 10 else f"{r:.0f} steps/s")
        return " · ".join(parts)

    def _on_progress(self, sim_id: str, fraction: float, message: str) -> None:
        if sim_id != self._running:
            self._on_started(sim_id)
        self._fraction = float(fraction)
        self._message = str(message or "")
        self._refresh()

    def _ended(self, sim_id: str, text: str, role: str) -> None:
        self._cells.pop(sim_id, None)
        self._rate.pop(sim_id, None)
        if sim_id in self._queued:
            self._queued.remove(sim_id)
        if sim_id == self._running:
            self._running = None
        if self._running is None and not self._queued:
            self.bar.setVisible(False)
            self.text.setText(text)
            theme.set_prop(self.text, "role", role)
            self.setVisible(True)
            self._linger.start(8000)
        else:
            self._refresh()

    def _on_finished(self, sim_id: str, _result) -> None:
        # the job manager's wall time (planning included) is the one duration the app quotes,
        # so the status bar and the Run panel never disagree; the local clock is the fallback
        seconds = None
        job_seconds = getattr(self._jobs, "job_seconds", None)
        if callable(job_seconds):
            with contextlib.suppress(Exception):
                seconds = job_seconds(sim_id)
        if not isinstance(seconds, (int, float)) and self._started:
            seconds = time.monotonic() - self._started
        took = format_duration(seconds) if isinstance(seconds, (int, float)) else ""
        self._ended(sim_id, f"{self._name(sim_id)}: done in {took}" if took else f"{self._name(sim_id)}: done",
                    "status-good")

    def _on_failed(self, sim_id: str, _err: str) -> None:
        self._ended(sim_id, f"{self._name(sim_id)}: failed — see Messages", "status-bad")

    def _on_cancelled(self, sim_id: str) -> None:
        self._ended(sim_id, f"{self._name(sim_id)}: stopped", "status-warn")

    def _refresh(self) -> None:
        if self._running is None and not self._queued:
            if not self._linger.isActive():
                self.setVisible(False)
            return
        self._linger.stop()
        theme.set_prop(self.text, "role", "")
        if self._running is not None:
            self.bar.setVisible(True)
            self.bar.setValue(round(1000 * max(0.0, min(1.0, self._fraction))))
            text = f"Running {self._name(self._running)}  {round(100 * self._fraction)} %"
            if self._message:
                msg = self._message if len(self._message) <= 40 else self._message[:39] + "…"
                text += f"  ·  {msg}"
            metrics = self.metrics(self._running)
            if metrics:
                text += f"  ·  {metrics}"
        else:
            self.bar.setVisible(False)
            text = "Waiting to start"
        if self._queued:
            text += f"  ·  {len(self._queued)} queued"
        self.text.setText(text)
        self.setVisible(True)

    def summary(self) -> str:
        return self.text.text() if not self.isHidden() else ""


#: the panel header of every dock (kept under its v2 name: tests and older code import it from here)
DockTitle = DockHeader


# ---------------------------------------------------------------------------
# placeholder for pages that fail to load
# ---------------------------------------------------------------------------

def breadcrumb_text(design, simulation) -> str:
    """The step bar's "design › simulation" text, with one name when both carry the same one.

    The physics templates name a case's simulation after its design, so the bar read
    "Heated cylinder · d 10 mm · ΔT 10 K  ›  Heated cylinder · d 10 mm · ΔT 10 K": about
    400 px of the bar repeating what the Project tree already shows twice (v3.3 audit).
    Names are compared without case or repeated spaces.
    """
    names = [str(x.name) for x in (design, simulation) if x is not None and str(x.name or "").strip()]
    if len(names) == 2 and " ".join(names[0].split()).casefold() == " ".join(names[1].split()).casefold():
        names = names[:1]
    return "  ›  ".join(names)


class PlaceholderPage(QWidget):
    """Stands in for a page whose module is missing or raised while loading.

    States the problem in one line, folds the traceback away underneath, and
    offers Retry (re-imports the module - handy while developing).
    """

    retryRequested = Signal(str)

    def __init__(self, spec: PageSpec, error: BaseException, tb: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec, self.error = spec, error
        missing = isinstance(error, ModuleNotFoundError) and f"pages.{spec.key}" in str(error)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 16)
        if missing:
            title = f"{spec.title} is not part of this build"
            text = f"Module aerolume.app.pages.{spec.key} is not installed. The other steps work."
        else:
            title = f"{spec.title} could not be loaded"
            text = f"{type(error).__name__}: {error}"
        self.empty = EmptyState("warning", title, text, "Retry", self._retry,
                                secondary_text="Welcome page", on_secondary=self._go_start)
        lay.addWidget(self.empty, 1)
        details = QPlainTextEdit(tb)
        details.setReadOnly(True)
        theme.use_mono(details)
        details.setMinimumHeight(160)
        self.details = CollapsibleSection("Technical details", details, expanded=False)
        lay.addWidget(self.details)
        self._state = None

    def _retry(self) -> None:
        self.retryRequested.emit(self.spec.key)

    def _go_start(self) -> None:
        if self._state is not None:
            self._state.navigate("start")


# ---------------------------------------------------------------------------
# dialogs
# ---------------------------------------------------------------------------

class CrashDialog(QDialog):
    """An unexpected error, as a report the user can copy into an issue (app/platform.CrashReport).

    Says what happened in one line, that the work is saved where it can be,
    and where the report file is; the full redacted report (version facts and
    traceback, the home folder shown as ``~``) is in a read-only box with
    Copy report. The app keeps running: Qt slot errors are recoverable.
    """

    def __init__(self, report, parent: QWidget | None = None, *, log_folder: Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aerolume ran into a problem")
        self.setMinimumSize(620, 420)
        self._reports = []
        self._log_folder = log_folder
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(10)
        self.summary = label("", "h2", wrap=True)
        lay.addWidget(self.summary)
        self.notice = Notice("", "warn")
        lay.addWidget(self.notice)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        theme.use_mono(self.details, small=True)
        self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.details.setAccessibleName("Error report")
        lay.addWidget(self.details, 1)
        box = QDialogButtonBox()
        self.copy_button = box.addButton("Copy report", QDialogButtonBox.ButtonRole.ActionRole)
        self.copy_button.clicked.connect(self.copy_report)
        self.folder_button = box.addButton("Open log folder", QDialogButtonBox.ButtonRole.ActionRole)
        self.folder_button.clicked.connect(self._open_folder)
        self.folder_button.setEnabled(log_folder is not None)
        close = box.addButton(QDialogButtonBox.StandardButton.Close)
        close.setProperty("variant", "primary")
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.add_report(report)

    def add_report(self, report) -> None:
        self._reports.append(report)
        n = len(self._reports)
        first = self._reports[0]
        summary = str(getattr(first, "summary", "") or first)
        self.summary.setText(summary if n == 1 else f"{n} unexpected errors; the first: {summary}")
        path = getattr(report, "path", None)
        where = (f"The report was saved to {path}." if path else "The report could not be saved to a file; "
                 "copy it from the box below.")
        self.notice.set_text(f"Aerolume keeps running. {where} Save the project (Ctrl+S) before you go on; "
                             "if the same action fails again, restart Aerolume.",
                             "Unexpected error" if n == 1 else f"{n} unexpected errors")
        self.details.setPlainText(self.report_text())

    def report_text(self) -> str:
        parts = [str(getattr(r, "text", "") or r) for r in self._reports]
        return ("\n" + "-" * 72 + "\n").join(parts)

    def copy_report(self) -> str:
        text = self.report_text()
        QApplication.clipboard().setText(text)
        self.copy_button.setText("Copied")
        QTimer.singleShot(2000, self, lambda: self.copy_button.setText("Copy report"))
        return text

    def _open_folder(self) -> None:
        if self._log_folder is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._log_folder)))


class _FitPage(QScrollArea):
    """A Preferences tab page that claims its height only while it is the current tab.

    QTabWidget sizes itself to the largest of its pages' size hints (whatever
    their size policy), so the eight-row General tab sat above about 420 px
    of nothing because the Assistant tab is long. Wrapping each page lets the
    dialog fit the tab on screen; a screen too short for a page scrolls it
    instead of clipping it.
    """

    def __init__(self, inner: QWidget) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidget(inner)
        self.viewport().setAutoFillBackground(False)        # the tab pane colour shows through
        inner.setAutoFillBackground(False)
        self.active = False

    def sizeHint(self) -> QSize:
        inner = self.widget()
        hint, low = inner.sizeHint(), inner.minimumSizeHint()
        return QSize(max(hint.width(), low.width()), max(hint.height(), low.height()) if self.active else 0)

    def minimumSizeHint(self) -> QSize:
        return QSize(self.widget().minimumSizeHint().width(), 120)


class _PageStack(QStackedWidget):
    """The page area: reports the *current* page's minimum size, not the largest of all pages.

    ``QStackedWidget`` takes its minimum from every page it holds, shown or not, so once the
    Results page (432 px minimum height) had been visited, the Script panel under the Setup
    page (334 px) or the Simulate page (274 px) could never take the height those pages give
    up: at 1280 x 800 the Scripting workspace showed 9 editor lines (measured, v3.3). Other
    pages are hidden, and a switch asks the window for a new layout, which hands height back
    to the page first (the docks give way, as for any page minimum).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(lambda _i: self.updateGeometry())

    def minimumSizeHint(self) -> QSize:
        w = self.currentWidget()
        if w is None:
            return super().minimumSizeHint()
        return w.minimumSizeHint().expandedTo(w.minimumSize())

    def sizeHint(self) -> QSize:
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()


class PreferencesDialog(QDialog):
    """Preferences, applied on OK.

    *General*: theme (Light, Dark or System), units, compute device and
    precision. *Assistant*: model provider and model, the Claude API key, the
    local model server, approvals
    (:class:`~aerolume.app.widgets.assistant_panel.AssistantSettingsPage`).
    *Account*: the optional Supabase project URL and anon key, a connection
    test and what leaves the computer
    (:class:`~aerolume.app.dialogs.account_dialogs.AccountSettingsPage`,
    DESIGN.md section 11). Both pages are imported when the dialog opens; a
    build without one says why in its tab. ``tab`` ("general" | "assistant" |
    "account") is the tab shown first.

    The dialog is sized to the tallest tab rather than a fixed height, so the
    General tab no longer floats in 400 px of empty space; a screen too short
    for that gets a scroll bar instead of a clipped dialog.
    """

    TABS = ("general", "assistant", "account")

    def __init__(self, parent: QWidget | None = None, *, tab: str = "general") -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(580)
        s = app_settings.get()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(12)
        self.tabs = QTabWidget()

        general = QWidget()
        gl = QVBoxLayout(general)
        gl.setContentsMargins(0, 12, 0, 0)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.theme = SegmentedControl([("light", "Light"), ("dark", "Dark"), ("system", "System")], s.theme)
        self.theme.setToolTip("System follows the light / dark setting of the operating system and switches "
                              "when it does.")
        _form_row(form, "Theme", self.theme)
        self.font = SegmentedControl([("bundled", "Atkinson Hyperlegible"), ("system", "System (Segoe UI)")],
                                     s.ui_font)
        self.font.setToolTip("Atkinson Hyperlegible keeps 0 / O, 1 / l / I and 5 / S apart in numbers; "
                             "System uses the platform font throughout.")
        _form_row(form, "Interface font", self.font)
        self.units = SegmentedControl([("SI", "SI (m, m/s, N)"), ("imperial", "Imperial (ft, mph, lbf)")],
                                      s.unit_system)
        self.units.setToolTip("Projects always store SI; this only changes what you read and type.")
        _form_row(form, "Units", self.units)
        self.device = QComboBox()
        self.device.setToolTip("Default for new simulations; each simulation can override it in Setup.")
        self._fill_devices(s.device)
        _form_row(form, "Compute device", self.device)
        self.precision = QComboBox()
        for key, text in (("auto", "Automatic (float32 on GPU; on CPU float64 in 2D, float32 in 3D)"),
                          ("float32", "float32"), ("float64", "float64")):
            self.precision.addItem(text, key)
        self.precision.setCurrentIndex(max(self.precision.findData(s.precision), 0))
        self.precision.setToolTip("Default for new simulations; each simulation can override it in Setup.")
        _form_row(form, "Precision", self.precision)
        self.update_url = QLineEdit(s.update_url)
        self.update_url.setPlaceholderText("Off: no release feed is published yet")
        self.update_url.setClearButtonEnabled(True)
        self.update_url.setToolTip("Address of a JSON release feed (a GitHub \"latest release\" API URL or "
                                   "{\"version\", \"url\"}). Help ▸ Check for updates downloads it and sends "
                                   "nothing. Empty turns the check off.")
        _form_row(form, "Release feed", self.update_url)
        gl.addLayout(form)
        gl.addStretch(1)
        self.tabs.addTab(_FitPage(general), "General")

        self.assistant = None
        try:
            from .widgets.assistant_panel import AssistantSettingsPage
            self.assistant = AssistantSettingsPage()
            page = self.assistant
            tie_form_labels(page)           # its addRow(QLabel, field) rows left nine fields unnamed
        except Exception as exc:  # noqa: BLE001 - the other preferences still work
            page = EmptyState("warning", "Assistant settings are not available", f"{type(exc).__name__}: {exc}")
        self.tabs.addTab(_FitPage(page), "Assistant")
        self.account = None
        try:
            from .dialogs.account_dialogs import AccountSettingsPage
            self.account = AccountSettingsPage()
            page = self.account
        except Exception as exc:  # noqa: BLE001 - accounts are optional; the app works offline without them
            page = EmptyState("info", "Account settings are not available",
                              f"Aerolume works fully offline without an account. ({type(exc).__name__}: {exc})")
        self.tabs.addTab(_FitPage(page), "Account")
        self.tabs.setCurrentIndex(self.TABS.index(tab) if tab in self.TABS else 0)
        lay.addWidget(self.tabs, 1)
        foot = label(f"Settings file: {s.path}", "small", wrap=True)
        foot.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(foot)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setProperty("variant", "primary")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        #: what the last accepted apply() changed (read by MainWindow.open_preferences)
        self.changed: dict[str, str] = {}
        self._fit_to_content()
        self.tabs.currentChanged.connect(self._fit_to_content)

    def page_widget(self, key: str) -> QWidget | None:
        """The content of a tab ("general" | "assistant" | "account"), without its scroll wrapper."""
        if key not in self.TABS:
            return None
        page = self.tabs.widget(self.TABS.index(key))
        return page.widget() if isinstance(page, _FitPage) else page

    def _fit_to_content(self, index: int | None = None) -> None:
        """Height = what the *current* tab needs (see :class:`_FitPage`), capped at 85 % of the screen.

        The width stays that of the widest page, so switching tabs never makes
        the dialog jump sideways; only the height follows the tab.
        """
        index = self.tabs.currentIndex() if index is None else index
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            if isinstance(page, _FitPage):
                page.active = i == index
                page.updateGeometry()
        self.tabs.updateGeometry()              # the dialog's layout caches the tab widget's hint
        screen = self.screen() or QApplication.primaryScreen()
        cap = int(screen.availableGeometry().height() * 0.85) if screen is not None else 900
        self.setMaximumHeight(max(cap, 360))
        self.layout().invalidate()
        self.layout().activate()
        width = max(self.width(), 580, self.sizeHint().width())
        self.resize(width, min(self.sizeHint().height(), cap))

    def accept(self) -> None:
        """OK: save everything; an invalid account entry keeps the dialog open on its tab."""
        self.changed = self.apply()
        if self.account is not None and getattr(self.account, "last_error", ""):
            self.tabs.setCurrentIndex(self.TABS.index("account"))
            return
        super().accept()

    def _fill_devices(self, current: str) -> None:
        names = {"auto": "Automatic (best available)", "cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (MPS)",
                 "cpu": "CPU"}
        # Only ask torch when it is already loaded (the status bar's device probe imports it in the
        # background); importing it here would freeze the dialog for seconds on first open.
        avail: set[str] | None = None
        if "torch" in sys.modules:
            try:
                from ..core import device as dev
                avail = set(dev.available())
            except Exception:  # noqa: BLE001 - torch broken: still let the user pick
                avail = {"cpu"}
        for key in ("auto", "cuda", "mps", "cpu"):
            if avail is None:
                self.device.addItem(names[key], key)
            elif key == "auto" or key in avail or key == current:
                text = names[key] + ("" if key == "auto" or key in avail else " (not available)")
                self.device.addItem(text, key)
        self.device.setCurrentIndex(max(self.device.findData(current), 0))

    def values(self) -> dict[str, str]:
        """The General tab's values."""
        return {"theme": self.theme.current(), "ui_font": self.font.current(), "unit_system": self.units.current(),
                "device": self.device.currentData(), "precision": self.precision.currentData(),
                "update_url": self.update_url.text().strip()}

    def apply(self) -> dict[str, str]:
        """Write the chosen values and switch the theme; returns what changed.

        The General keys that changed map to their new values; ``"assistant"``
        is present when the Assistant tab saved something (settings or key),
        ``"account"`` when the Supabase connection changed. An invalid account
        entry is not saved; ``self.account.last_error`` says why.
        """
        s = app_settings.get()
        changed = {}
        for key, v in self.values().items():
            if getattr(s, key) != v:
                setattr(s, key, v)
                changed[key] = v
        if "theme" in changed or "ui_font" in changed:
            theme.apply(None, s.theme, font=s.ui_font)
        if self.assistant is not None and self.assistant.apply():
            changed["assistant"] = "changed"
        if self.account is not None and self.account.apply():
            changed["account"] = "changed"
        return changed


def _form_row(form: QFormLayout, text: str, field: QWidget) -> QLabel:
    """A form row whose label names its field for screen readers (QFormLayout does not, for a QLabel)."""
    lab = label(text, "pg-label")
    form.addRow(lab, field)
    tie_label(lab, field)
    return lab


# -- update check ---------------------------------------------------------------------
#: schemes "Check for updates" accepts: a release page over HTTP(S), or a mirrored file
_FEED_SCHEMES = ("https", "http", "file")


def version_key(text: str) -> tuple[int, ...]:
    """A comparable key for "v0.1.0", "0.2", "0.2.0rc1" (pre-release letters end the number)."""
    out = []
    for part in str(text).strip().lstrip("vV").split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        out.append(int(digits) if digits else 0)
        if len(digits) < len(part):
            break
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out)


def parse_release_feed(data: object) -> tuple[str, str]:
    """(version, release page) from a feed: ``{"version", "url"}`` or a GitHub latest-release object.

    Raises ValueError for anything else, so a wrong address reads as "not a
    release feed" rather than as "you are up to date".
    """
    if isinstance(data, list) and data:          # a GitHub "releases" list: newest first
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("the address did not return a release feed (expected a JSON object)")  # noqa: TRY004
    version = data.get("version") or data.get("tag_name") or data.get("name")
    page = data.get("url") if "version" in data else data.get("html_url") or data.get("url")
    if not isinstance(version, str) or not version_key(version) or not any(version_key(version)):
        raise ValueError("the feed names no version")
    return version.strip(), str(page or "")


def fetch_release_feed(url: str, timeout: float = 8.0) -> tuple[str, str]:
    """Download and parse a release feed. Only a GET of ``url``: no user data, no identifiers."""
    import json
    import urllib.parse
    import urllib.request

    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _FEED_SCHEMES:
        raise ValueError(f"unsupported address {url!r} (use https://)")
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": f"Aerolume/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(1_000_000)
    return parse_release_feed(json.loads(raw.decode("utf-8")))


@contextlib.contextmanager
def _busy_cursor():
    """Wait cursor for a GUI-thread operation that takes noticeable time (always restored)."""
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


def _platform_name() -> str:
    """OS name for humans; ``platform.release()`` still says "10" on Windows 11."""
    system, release = platform.system(), platform.release()
    if system == "Windows" and hasattr(sys, "getwindowsversion") and sys.getwindowsversion().build >= 22000:
        release = "11"
    return f"{system} {release} ({platform.machine()})"


def _versions() -> list[tuple[str, str]]:
    """Component versions for the About dialog (never imports anything that may fail loudly)."""
    import PySide6
    from PySide6.QtCore import qVersion

    rows = [("Aerolume", __version__), ("Python", platform.python_version()),
            ("PySide6 / Qt", f"{PySide6.__version__} / {qVersion()}")]
    for mod in ("numpy", "scipy", "pyqtgraph", "matplotlib", "vtk"):
        try:
            rows.append((mod, importlib.import_module(mod).__version__))
        except Exception:  # noqa: BLE001
            rows.append((mod, "not installed"))
    try:
        import torch
        rows.append(("PyTorch", torch.__version__))
        rows.append(("CUDA", torch.version.cuda or "not available"))
        from ..core import device as dev
        rows.append(("Compute device", dev.info("auto").name))
    except Exception as exc:  # noqa: BLE001
        rows.append(("PyTorch", f"unavailable ({type(exc).__name__})"))
    rows.append(("Platform", _platform_name()))
    return rows


_APACHE_NOTICE = """Aerolume
Copyright the Aerolume contributors.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this software except in compliance with the License. You may obtain a copy of
the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

_THIRD_PARTY = """## mfoil

Aerolume's airfoil polar solver wraps **mfoil** by Krzysztof J. Fidkowski (University of Michigan),
an XFOIL-class coupled viscous-inviscid solver, used under the **MIT licence**. The full licence
text ships in `third_party/mfoil/LICENSE`.

Reference: K. J. Fidkowski, "A coupled inviscid-viscous airfoil analysis solver, revisited",
AIAA Journal, 2022.

## Libraries

| Component | Licence |
|---|---|
| Qt for Python (PySide6) | LGPL-3.0 (dynamically linked) |
| PyTorch | BSD-3-Clause |
| NumPy, SciPy | BSD-3-Clause |
| pyqtgraph | MIT |
| VTK (3D rendering) | BSD-3-Clause |
| Matplotlib | Matplotlib licence (PSF-based) |
| ezdxf | MIT |
| Pillow | MIT-CMU (HPND) |
| contourpy | BSD-3-Clause |
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class AboutDialog(QDialog):
    """Product, versions, Apache-2.0 licence and third-party notices (mfoil)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Aerolume")
        self.setMinimumSize(560, 480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(12)
        head = QHBoxLayout()
        head.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(theme.app_icon().pixmap(QSize(40, 40)))
        head.addWidget(ic)
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(label("Aerolume", "h1"))
        col.addWidget(label(f"Version {__version__} · 2D and 3D aerodynamics studio · Apache-2.0", "muted"))
        head.addLayout(col, 1)
        lay.addLayout(head)

        tabs = QTabWidget()
        info = QTreeWidget()
        info.setColumnCount(2)
        info.setHeaderLabels(["Component", "Version"])
        info.setRootIsDecorated(False)
        info.setProperty("flat", "true")
        self.version_rows = _versions()
        for k, v in self.version_rows:
            it = QTreeWidgetItem([k, v])
            it.setFont(1, theme.mono_font())
            info.addTopLevelItem(it)
        info.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        tabs.addTab(info, "Versions")

        lic = QPlainTextEdit(self._licence_text())
        lic.setReadOnly(True)
        theme.use_mono(lic)
        tabs.addTab(lic, "Licence")

        third = QTextBrowser()
        third.setOpenExternalLinks(True)
        third.setMarkdown(self._third_party_text())
        tabs.addTab(third, "Third-party notices")
        lay.addWidget(tabs, 1)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    @staticmethod
    def _licence_text() -> str:
        return _APACHE_NOTICE + "\nThe full licence text is in the LICENSE file of the distribution."

    @staticmethod
    def _third_party_text() -> str:
        text = _THIRD_PARTY
        f = _repo_root() / "THIRD_PARTY_NOTICES.md"
        if f.exists():
            try:
                text = f.read_text(encoding="utf-8") + "\n\n" + _THIRD_PARTY.split("## Libraries", 1)[1].join(
                    ["## Libraries", ""])
            except OSError:
                pass
        return text


#: heading sizes of in-app documents (docs/ui_guidelines.md: section titles 10-11 pt semibold, nothing larger)
DOC_HEADING_PT = {1: 11.0, 2: 10.0, 3: 9.0, 4: 9.0, 5: 9.0, 6: 9.0}
#: cell padding that gives a 9 pt table row the guidelines' 22 px height
DOC_CELL_PADDING = 3.0


def style_document(doc: QTextDocument) -> None:
    """Give a Markdown document the app's look: small semibold headings and token-bordered tables.

    Qt's Markdown import makes an H1 xx-large bold (about 18 pt) and draws
    tables with its default double grey border, so the experiment log read as
    a raw web page (one goal wrapped over three 18 pt lines above a browser
    table). Headings become :data:`DOC_HEADING_PT` semibold in the text colour;
    tables get a 1 px collapsed border in the ``border`` token, 22 px rows and
    a ``panel``-tinted header row. Called again after every reload, so a theme
    change restyles an open document.
    """
    header = theme.qcolor("panel")
    cursor = QTextCursor(doc)
    block = doc.begin()
    while block.isValid():
        level = block.blockFormat().headingLevel()
        if level > 0:
            cursor.setPosition(block.position())
            cursor.setPosition(block.position() + max(0, block.length() - 1), QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            # The Markdown reader sizes headings with a relative FontSizeAdjustment (+3 for H1), which the
            # layout applies on top of any point size: it must be cleared, not just overridden.
            fmt.setProperty(QTextFormat.Property.FontSizeAdjustment, 0)
            fmt.setFontPointSize(DOC_HEADING_PT.get(level, 9.0))
            fmt.setFontWeight(600)
            fmt.setForeground(theme.qcolor("text"))
            cursor.mergeCharFormat(fmt)
            bf = QTextBlockFormat()
            bf.setTopMargin(14 if level <= 2 else 10)
            bf.setBottomMargin(4)
            cursor.mergeBlockFormat(bf)
        block = block.next()

    def frames(frame):
        for child in frame.childFrames():
            yield child
            yield from frames(child)

    border = theme.qcolor("border")
    for frame in list(frames(doc.rootFrame())):
        if not isinstance(frame, QTextTable):
            continue
        tf = frame.format()
        tf.setBorder(1)
        tf.setBorderBrush(border)
        tf.setBorderStyle(QTextFrameFormat.BorderStyle.BorderStyle_Solid)
        tf.setBorderCollapse(True)
        tf.setCellSpacing(0)
        tf.setCellPadding(DOC_CELL_PADDING)
        tf.setTopMargin(4)
        tf.setBottomMargin(8)
        frame.setFormat(tf)
        for col in range(frame.columns()):
            cell = frame.cellAt(0, col)
            cf = QTextTableCellFormat()
            cf.setBackground(header)
            cf.setFontWeight(600)
            merged = cell.format()
            merged.merge(cf)
            cell.setFormat(merged)


class _FittingBrowser(QTextBrowser):
    """A QTextBrowser that shrinks images wider than its view to the view's width.

    The user guide's screenshots are ~1600 px wide; in the 880 px help dialog
    every figure was cropped and needed horizontal scrolling. Images are scaled
    as they are loaded (:meth:`loadResource`), never enlarged, and the document
    is re-laid out with freshly scaled images when the width changes noticeably.
    """

    REFIT_STEP_PX = 24

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._markdown: str | None = None
        self._fit_width = 0
        self._refit = QTimer(self)
        self._refit.setSingleShot(True)
        self._refit.setInterval(150)
        self._refit.timeout.connect(self._reload)

    def available_width(self) -> int:
        margin = int(self.document().documentMargin())
        return max(240, self.viewport().width() - 2 * margin - 6)

    def set_markdown_fitted(self, text: str) -> None:
        if self._markdown is None:
            theme.changed.connect(self._restyle)
        self._markdown = text
        self._fit_width = self.available_width()
        self.setMarkdown(text)
        style_document(self.document())

    def _restyle(self, *_args) -> None:
        """Re-render after a theme change: table borders and header tint are theme tokens."""
        if self._markdown is not None:
            self._reload()

    def loadResource(self, rtype, url):
        res = super().loadResource(rtype, url)
        if int(getattr(rtype, "value", rtype)) != int(QTextDocument.ResourceType.ImageResource.value) or res is None:
            return res
        img = res if isinstance(res, QImage) else QImage()
        if isinstance(res, QPixmap):
            img = res.toImage()
        elif not isinstance(res, QImage):
            try:
                img.loadFromData(bytes(res))
            except (TypeError, ValueError):
                return res
        width = self._fit_width or self.available_width()
        if img.isNull() or img.width() <= width:
            return res
        return img.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._markdown is not None and abs(self.available_width() - self._fit_width) >= self.REFIT_STEP_PX:
            self._refit.start()

    def _reload(self) -> None:
        bar = self.verticalScrollBar()
        frac = bar.value() / bar.maximum() if bar.maximum() > 0 else 0.0
        self._fit_width = self.available_width()
        self.setMarkdown(self._markdown or "")
        style_document(self.document())
        bar.setValue(round(frac * bar.maximum()))


class DocViewer(QDialog):
    """In-app Markdown viewer for the user guide and validation notes (works offline)."""

    def __init__(self, path: Path, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(880, 720)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        view = _FittingBrowser()
        view.setOpenExternalLinks(True)
        view.setSearchPaths([str(path.parent)])
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".md":
            view.set_markdown_fitted(text)
        else:
            view.setPlainText(text)
            theme.use_mono(view)
        lay.addWidget(view, 1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self.view = view


SHORTCUTS = (
    ("Ctrl+N", "New study"), ("Ctrl+O", "Open project"), ("Ctrl+S", "Save project"),
    ("Ctrl+I", "Import a 2D outline"), ("Ctrl+Shift+I", "Import a 3D mesh (STL, OBJ, PLY)"),
    ("Ctrl+1 … Ctrl+6", "Geometry … Report step"),
    ("F5", "Run the selected simulation"), ("Shift+F5", "Stop the running simulation"),
    ("F2", "Rename the selected design or simulation"), ("Delete", "Delete the selected design or simulation"),
    ("Ctrl+,", "Preferences"), ("Ctrl+Shift+T", "Switch light / dark theme"), ("F11", "Full screen"),
    ("Ctrl+Shift+A", "Experiment assistant"), ("Alt+3", "Show or hide the Assistant panel"),
    ("Alt+4", "Show or hide the Script panel"),
    ("Ctrl+Tab, Ctrl+Shift+Tab", "Leave the Script editor (Tab indents there); Esc then Tab also works"),
    ("F6 / Shift+F6", "Next / previous region: steps, page, then each panel's header"),
    ("Enter, Shift+F10, Delete", "On a panel header: collapse or expand, options, hide"),
    ("F1", "User guide"),
)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

def end_of_run_diagnostics(result):
    """A one-row :class:`~aerolume.app.widgets.diagnostics.DiagnosticsLog` of the end-of-run values a
    result stores (``result.diagnostics``: divergence, change per step, CFL, ... at the last step),
    or None when it stores none. The panel then lists them under Latest, as Results ▸ Diagnostics
    does in its table."""
    d = getattr(result, "diagnostics", None)
    if not isinstance(d, dict) or not d:
        return None
    values = {}
    for k, v in d.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            values[str(k)] = f
    step, t = values.pop("step", 0.0), values.pop("t", 0.0)
    if not values:
        return None
    from .widgets.diagnostics import DiagnosticsLog
    log_ = DiagnosticsLog()
    log_.add(step, t, values)
    log_.source = "End of the run (stored with the result)"
    return log_ if len(log_) else None


def diagnostics_log_for(jobs, sim_id: str):
    """The solver-diagnostics log to show for a simulation, or None when there is nothing to show.

    In order: the log the job manager is recording for it this session (a run
    in progress, or one that ran since the app started); else the diagnostic
    series stored in its saved result's history
    (:func:`aerolume.app.pages.results.history_diagnostics`); else the
    end-of-run values the result stores (:func:`end_of_run_diagnostics`). A run
    from an earlier session so shows what Results ▸ Diagnostics shows instead
    of "No diagnostics yet". Never raises: a result that cannot be read just
    means nothing stored. The Simulate page can use the same rule for its own
    Diagnostics tab.
    """
    if jobs is None or not sim_id:
        return None
    session = None
    fn = getattr(jobs, "diagnostics_for", None)
    if callable(fn):
        with contextlib.suppress(Exception):
            session = fn(sim_id)
        running = getattr(jobs, "running_id", lambda: None)()
        if session is not None and (len(session) or running == sim_id):
            return session
    result_fn = getattr(jobs, "result", None)
    if callable(result_fn):
        try:
            res = result_fn(sim_id)
            if res is not None:
                from .pages.results import history_diagnostics
                stored = history_diagnostics(res)
                if stored is not None and len(stored):
                    return stored
                end = end_of_run_diagnostics(res)
                if end is not None:
                    return end
        except Exception:  # noqa: BLE001 - a damaged result is reported by Results, not here
            _log.debug("no stored diagnostics for %s", sim_id, exc_info=True)
    return session


class MainWindow(QMainWindow):
    """The application window. Owns AppState, the JobManager and every page.

    ``state`` and ``jobs`` may be injected (tests); otherwise they are created
    here. ``jobs`` is optional at runtime: if ``app/jobs.py`` cannot be imported
    the window still runs and pages receive ``None``.
    """

    #: (version, page, error) of a finished update check, delivered on the GUI thread
    updateChecked = Signal(object)

    def __init__(self, state: AppState | None = None, jobs=None, *, load_jobs: bool = True,
                 probe_device: bool = True) -> None:
        super().__init__()
        self.state = state or AppState(self)
        self.jobs = jobs if jobs is not None else (self._make_jobs() if load_jobs else None)
        self._pages: dict[str, QWidget] = {}
        self._current = ""
        self._workspace_shown = True
        #: set while closing after the user (or a harness) already agreed to stop everything
        self._quit_confirmed = False
        self._step_bar_wanted = self._load_step_bar_pref()
        # Connected before the docks are built so it runs before the Project tree's own
        # currentDesignChanged slot: the tree then syncs to an already coherent selection.
        self.state.currentDesignChanged.connect(self._keep_selection_coherent)
        self.setWindowTitle("Aerolume")
        self.setWindowIcon(theme.app_icon())
        self.setMinimumSize(960, 600)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks)

        page_frame = QFrame()
        page_frame.setProperty("chrome", "page")
        pl = QVBoxLayout(page_frame)
        pl.setContentsMargins(1, 1, 1, 1)
        self.stack = _PageStack()
        pl.addWidget(self.stack)
        self.setCentralWidget(page_frame)
        # scroll areas leave the Tab chain once their page is on screen (common.unfocus_scroll_areas);
        # debounced, as pages and panels fill their content after show_page returns
        self._unfocus_timer = QTimer(self)
        self._unfocus_timer.setSingleShot(True)
        self._unfocus_timer.setInterval(0)
        # after a page switch: scroll areas out of the Tab chain, nameless tables and tab bars named
        self._unfocus_timer.timeout.connect(lambda: (unfocus_scroll_areas(self), name_unnamed_views(self)))

        self._build_actions()
        self._build_toolbars()
        self._build_docks()
        self._add_tool_window_buttons()
        self._build_menus()
        self._build_status_bar()

        st = self.state
        st.navigateRequested.connect(self._navigate)
        command = getattr(st, "commandRequested", None)
        if command is not None:                 # injected test states may predate it
            command.connect(self.run_command)
        st.statusMessage.connect(self._status)
        st.projectChanged.connect(self._project_changed)
        # a rename, a new analysis or a changed kind must reach the breadcrumb and the Simulation
        # menu too, not only the step checks (_selection_changed ends with _update_steps)
        st.designsChanged.connect(self._selection_changed)
        st.simulationsChanged.connect(self._selection_changed)
        st.resultChanged.connect(lambda _id: self._update_steps())
        reports = getattr(st, "reportsChanged", None)
        if reports is not None:                 # a saved report ticks the Report step
            reports.connect(self._update_steps)
        st.currentDesignChanged.connect(lambda _id: self._selection_changed())
        st.currentSimulationChanged.connect(lambda _id: self._selection_changed())
        app_settings.signals.changed.connect(self._settings_changed)
        theme.changed.connect(self._theme_changed)
        if self.jobs is not None:
            self.run_status.connect_jobs(self.jobs)
            self.project_tree.connect_jobs(self.jobs)
            self.messages.connect_jobs(self.jobs)
            for sig in ("jobQueued", "jobStarted", "jobFinished", "jobFailed", "jobCancelled"):
                s = getattr(self.jobs, sig, None)
                if s is not None:
                    s.connect(lambda *_a: (self._update_steps(), self._update_run_actions()))
            for sig in ("jobStarted", "jobFinished", "jobFailed", "jobCancelled"):
                s = getattr(self.jobs, sig, None)
                if s is not None:
                    s.connect(lambda *_a: self._clear_stopping())
            failed = getattr(self.jobs, "jobFailed", None)
            if failed is not None:
                failed.connect(lambda *_a: self._reveal_messages())

        #: the widget that last had focus in the step bar and in the page (F6 goes back to it)
        self._region_focus: dict[str, QWidget] = {}
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._remember_region_focus)
        self._restore_geometry()
        self.setAcceptDrops(True)                # a geometry file dropped anywhere is imported (dropEvent)
        self.installEventFilter(self)            # see eventFilter: tells window resizes from dock drags
        # the log starts as the workspace left it: Default folds it to its title row (counts + newest line)
        self.set_messages_expanded(not self.docks.is_collapsed("messages"))
        self._install_account_menu()
        self._project_changed()
        self.show_page("start")
        if probe_device:
            self.probe_device(app_settings.get().device)

    # -- construction -------------------------------------------------------------
    def _make_jobs(self):
        try:
            from .jobs import JobManager
            return JobManager(self.state)
        except Exception as exc:  # noqa: BLE001 - simulations unavailable, everything else works
            self._jobs_error = f"{type(exc).__name__}: {exc}"
            QTimer.singleShot(0, self, lambda: self.state.log(
                f"Simulation runner unavailable ({self._jobs_error})", "error", "Application"))
            return None

    def _act(self, text: str, slot, shortcut: str | QKeySequence | None = None, icon: str | None = None,
             tip: str | None = None) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if icon:
            a.setIcon(theme.icon(icon, "text2"))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(lambda _=False: slot())
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        nav = self.state.navigate
        self.act_new = self._act("&New study…", self.new_project_dialog, QKeySequence.StandardKey.New, "new",
                                 "New study (Ctrl+N)")
        self.act_open = self._act("&Open project…", self.open_project_dialog, QKeySequence.StandardKey.Open,
                                  "open", "Open project (Ctrl+O)")
        self.act_save = self._act("&Save", self.save_project, QKeySequence.StandardKey.Save, "save",
                                  "Save project (Ctrl+S)")
        self.act_close = self._act("&Close project", self.close_project, None, None)
        self.act_welcome = self._act("&Welcome page", lambda: nav("start"), None, "home")
        self.act_exit = self._act("E&xit", self.request_exit, QKeySequence("Ctrl+Q"))
        self.act_import = self._act("&2D outline…", lambda: nav("geometry:import"), QKeySequence("Ctrl+I"),
                                    "import",
                                    "Import a 2D outline: airfoil .dat, SVG, DXF, CSV or image (Ctrl+I)")
        self.act_import_mesh = self._act("&3D mesh (STL, OBJ, PLY)…", lambda: self.import_mesh(),
                                         QKeySequence("Ctrl+Shift+I"), "cube",
                                         "Import a 3D surface mesh: STL, OBJ or PLY (Ctrl+Shift+I)")
        self.act_new_wing = self._act("New &wing", lambda: self.new_wing(), None, "wing",
                                      "Add a tapered wing (planform, VLM / lifting-line analysis)")
        self.act_templates = self._act("New design from &template…", lambda: nav("geometry:templates"), None,
                                       "layers")
        self.act_naca = self._act("New &NACA airfoil…", lambda: nav("geometry:naca"), None, "airfoil")
        self.act_run = self._act("&Run", self.run_current, QKeySequence("F5"), "run",
                                 "Run the selected simulation (F5)")
        self.act_stop = self._act("S&top", self.stop_current, QKeySequence("Shift+F5"), "stop",
                                  "Stop the running simulation (Shift+F5)")
        self.act_new_sim = self._act("&New simulation", lambda: nav("setup:new"), None, "plus",
                                     "New simulation of the selected design")
        self.act_new_wing_sim = self._act("New &wing analysis",
                                          lambda: self.new_simulation_of_current("wing"), None, "wing",
                                          "Vortex-lattice / lifting-line polar of the selected wing")
        self.act_new_cfd3d = self._act("New 3D &CFD", lambda: self.new_simulation_of_current("cfd"), None,
                                       "cube", "Navier–Stokes run of the selected 3D design")
        self.act_explore = self._act("&Explore live", lambda: nav("simulate:live"), None, "simulate",
                                     "Real-time flow you can steer while it runs")
        self.act_save_results = self._act("Save unsaved &results", self._retry_result_saves, None, "save",
                                          "Try again to save results the project folder refused")
        self.act_save_results.setEnabled(False)
        self.act_report = self._act("&Report", lambda: nav("report"), None, "report", "Report step")
        self.act_generate = self._act("&Generate report…", lambda: nav("report:generate"), None, "export")
        self.act_rename = self._act("Re&name", self._rename_selected, QKeySequence("F2"), "edit")
        self.act_duplicate = self._act("D&uplicate", self._duplicate_selected, QKeySequence("Ctrl+D"), "copy")
        self.act_delete = self._act("&Delete", self._delete_selected, None, "trash")
        self.act_prefs = self._act("&Preferences…", self.open_preferences, QKeySequence("Ctrl+,"), "settings")
        self.act_full = self._act("&Full screen", self.toggle_full_screen, QKeySequence("F11"), "fullscreen")
        self.act_full.setCheckable(True)
        self.act_toggle_theme = self._act("Switch &theme", self.toggle_theme, QKeySequence("Ctrl+Shift+T"))
        self.act_guide = self._act("&User guide", lambda: self.show_help("guide"), QKeySequence("F1"), "book")
        self.act_validation = self._act("&Validation", lambda: self.show_help("validation"), None, "results")
        self.act_keys = self._act("&Keyboard shortcuts", self.show_shortcuts, None)
        # region cycling (Office, VS Code): the panel chrome is not in the Tab chain, F6 reaches it
        self.act_next_region = self._act("Next region", lambda: self.cycle_region(1), QKeySequence("F6"))
        self.act_prev_region = self._act("Previous region", lambda: self.cycle_region(-1),
                                         QKeySequence("Shift+F6"))
        self.act_about = self._act("&About Aerolume", self.show_about, None, "info")
        self.act_assistant = self._act("Experiment &assistant", lambda: self.show_assistant(),
                                       QKeySequence("Ctrl+Shift+A"), "assistant",
                                       "Plan, run and analyse an experiment with the AI assistant (Ctrl+Shift+A)")
        self.act_run_script = self._act("Run &script…", lambda: self.run_script_file(), None, "code",
                                        "Open a Python script in the Script panel and run it")
        self.act_assistant_guide = self._act("Experiment &assistant guide", lambda: self.show_help("assistant"),
                                             None, "book")
        self.act_scripting_guide = self._act("&Scripting guide", lambda: self.show_help("scripting"), None, "code")
        self.act_mcp_guide = self._act("Driving Aerolume from &Claude (MCP)", lambda: self.show_help("mcp"), None,
                                       "external", "Let Claude Desktop or Claude Code run experiments through the "
                                       "aerolume-mcp server (docs/mcp.md)")
        self.act_tour = self._act("Welcome &tour", lambda: self.start_tour(), None, "compass",
                                  "A short tour of what is on screen; skippable at any step")
        self.act_logs = self._act("Open &log folder", self.open_log_folder, None, "folder",
                                  "Crash reports and the application log")
        self.act_updates = self._act("Check for &updates…", lambda: self.check_for_updates(), None, "download",
                                     "Read the release feed set in Preferences (off until one is published)")
        self.page_actions: dict[str, QAction] = {"start": self.act_welcome}
        for i, key in enumerate(STEP_KEYS, start=1):
            spec = _SPEC[key]
            a = self._act(spec.title, lambda k=key: nav(k), QKeySequence(f"Ctrl+{i}"), spec.icon)
            self.page_actions[key] = a

    def _build_toolbars(self) -> None:
        tb = QToolBar("Main toolbar")
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        for a in (self.act_new, self.act_open, self.act_save):
            tb.addAction(a)
        tb.addSeparator()
        self.import_menu = QMenu("&Import geometry", self)
        self.import_menu.setIcon(theme.icon("import", "text2"))
        self.import_menu.addAction(self.act_import)
        self.import_menu.addAction(self.act_import_mesh)
        # the toolbar button opens the 2D / 3D choice: one click on "Import geometry" must not
        # guess which of two very different importers the user meant. It is a QAction the window
        # owns, not the menu's menuAction(): a cached Python wrapper of menuAction() went stale in
        # long scripted sessions ("Internal C++ object already deleted") and every later
        # _project_changed then aborted before the title and step bar were updated.
        self.act_import_geometry = QAction(theme.icon("import", "text2"), self.import_menu.title(), self)
        self.act_import_geometry.setMenu(self.import_menu)
        self.act_import_geometry.setIconText("Import geometry")
        self.act_import_geometry.setToolTip(
            "Import geometry: a 2D outline (Ctrl+I) or a 3D mesh (Ctrl+Shift+I)")
        tb.addAction(self.act_import_geometry)
        w = tb.widgetForAction(self.act_import_geometry)
        if isinstance(w, QToolButton):
            w.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            w.setProperty("dropdown", "true")
        tb.addSeparator()
        tb.addAction(self.act_run)
        tb.addAction(self.act_stop)
        tb.addSeparator()
        tb.addAction(self.act_report)
        labels = {self.act_import_geometry: "Import geometry", self.act_run: "Run", self.act_stop: "Stop",
                  self.act_report: "Report"}
        for a, text in labels.items():
            a.setIconText(text)
            w = tb.widgetForAction(a)
            if isinstance(w, QToolButton):
                w.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.main_toolbar = tb

        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        sb = QToolBar("Workflow steps")
        sb.setObjectName("stepToolbar")
        sb.setMovable(False)
        sb.setFloatable(False)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setStyleSheet("QToolBar { padding: 0; spacing: 0; }")
        self.step_bar = StepBar()
        self.step_bar.stepRequested.connect(self.state.navigate)
        sb.addWidget(self.step_bar)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, sb)
        self.step_toolbar = sb

    def _build_docks(self) -> None:
        """Every side panel is a dock of :class:`~aerolume.app.widgets.docks.DockManager`.

        The manager arranges them by the saved workspace; the window gives the
        shell's own panels their widgets (Project tree, Messages, Assistant,
        Script, Diagnostics) and keeps the attribute names older code and the
        tests use (``project_dock``, ``messages_dock``, ...).
        """
        self.docks = DockManager(self, app_settings.get())
        d = self.docks
        self._migrate_dock_prefs()
        self.project_tree = ProjectTree(self.state, None)
        d.set_content("project", self.project_tree)
        self.project_dock = d.panel("project")
        self.project_dock.setMinimumWidth(self.PROJECT_DOCK_MIN_KEPT)        # names stay readable beside the Assistant
        self.project_dock.visibilityChanged.connect(self._project_dock_visibility)
        self.messages = MessagesPanel(self.state, None)
        d.set_content("messages", self.messages)
        self.messages_dock = d.panel("messages")
        header = self.messages_dock.header
        header.add_widget(self.messages.bar)
        header.add_expanding(self.messages.latest)
        self.messages_toggle = header.collapse_button
        self.messages_dock.collapse_handler = lambda collapsed: self.set_messages_expanded(not collapsed, user=True)
        self.messages.latest.clicked.connect(lambda: self.set_messages_expanded(True, user=True))
        self.messages.messageAdded.connect(self._message_added)
        self._messages_expanded = not self.messages_dock.is_collapsed()
        self._messages_user_collapsed = False     # the user folded the log since the last project switch
        self._messages_heights = self._load_messages_heights()
        self._window_resizing = False
        self._applying_messages_height = False
        self.messages_dock.installEventFilter(self)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._build_tool_windows()
        self._build_diagnostics_panel()

    def _build_tool_windows(self) -> None:
        """The Assistant (right) and Script (bottom) panels, hidden until asked for in Default."""
        d = self.docks
        self.assistant = self._make_tool_window("assistant")
        d.set_content("assistant", self.assistant)
        self.assistant_dock = d.panel("assistant")
        self.script_console = self._make_tool_window("script")
        d.set_content("script", self.script_console)
        self.script_dock = d.panel("script")
        self.assistant_dock.visibilityChanged.connect(self._assistant_visibility)
        self._script_sized = False
        self.script_dock.visibilityChanged.connect(self._script_visibility)
        self._sync_script_share()
        for name, slot in (("experimentsChanged", self.project_tree.refresh_experiments),
                           ("preferencesRequested", lambda: self.open_preferences(tab="assistant")),
                           ("experimentOpened", self.open_experiment),
                           ("scriptRequested", self.show_script)):
            sig = getattr(self.assistant, name, None)
            if sig is not None:
                sig.connect(slot)
        self.project_tree.experimentActivated.connect(self.open_experiment)
        self.properties_dock = d.panel("properties")
        self.flowvis_dock = d.panel("flowvis")
        self.diagnostics_dock = d.panel("diagnostics")
        self.diagnostics_dock.visibilityChanged.connect(self._diagnostics_visibility)
        self.diagnostics_dock.collapsedChanged.connect(lambda c: c or self._diagnostics_visibility(True))

    def _build_diagnostics_panel(self) -> None:
        """The Diagnostics panel's own content: the solver diagnostics of the run on screen.

        ``widgets/diagnostics.DiagnosticsPanel`` is fed here from the job
        manager's frames (``DiagnosticsLog.add_frame``: a frame's history block
        when the solver sends one, else its latest values) and cleared when a
        run starts. A page that records its own log binds it with
        ``panel.content().set_log(log)``. The header switch is the panel's
        Record toggle (off keeps what was recorded and stops adding). Without
        the module the panel is simply unavailable.
        """
        self.diagnostics = None
        try:
            from .widgets.diagnostics import DiagnosticsPanel
            panel = DiagnosticsPanel()
        except Exception as exc:  # noqa: BLE001 - the panel is optional; everything else works
            self._diagnostics_error = f"{type(exc).__name__}: {exc}"
            return
        self.diagnostics = panel
        self.docks.set_content("diagnostics", panel)
        record = getattr(panel, "record_btn", None)
        if record is not None:
            sw = self.diagnostics_dock.enable_switch(
                record.isChecked(), "Record diagnostics (off keeps what was recorded and stops adding)")
            sw.toggled.connect(record.setChecked)
            record.toggled.connect(sw.setChecked)
        self._live_diagnostics = panel.log         # fed from job frames when the job manager keeps no logs
        self._diagnostics_dirty = True
        self._diag_timer = QTimer(self)             # coalesces selection / result / run changes into one bind
        self._diag_timer.setSingleShot(True)
        self._diag_timer.setInterval(0)
        self._diag_timer.timeout.connect(self._bind_diagnostics)
        jobs = self.jobs
        if jobs is not None:
            started = getattr(jobs, "jobStarted", None)
            frame = getattr(jobs, "jobFrame", None)
            if started is not None:
                started.connect(lambda _sid: (self._live_diagnostics.clear(), self._rebind_diagnostics()))
            if frame is not None:
                frame.connect(lambda _sid, f: self._diagnostics_frame(f))
            finished = getattr(jobs, "jobFinished", None)
            if finished is not None:
                finished.connect(lambda _sid, _r: self._rebind_diagnostics())
        self.state.currentSimulationChanged.connect(lambda _sid: self._rebind_diagnostics())
        self.state.resultChanged.connect(lambda _sid: self._rebind_diagnostics())
        self.state.projectChanged.connect(self._rebind_diagnostics)

    def _diagnostics_frame(self, frame) -> None:
        if callable(getattr(self.jobs, "diagnostics_for", None)):
            return                                  # the job manager logs per simulation; the panel shows its log
        with contextlib.suppress(Exception):         # a frame without diagnostics is not an error
            self._live_diagnostics.add_frame(frame)

    def _diagnostics_visibility(self, visible: bool) -> None:
        """Diagnostics opened: fold Messages if they share the bottom rail, and load the data shown.

        At 1280 x 800 an open log above an open Diagnostics panel left the page's
        field view 160-250 px tall; the log folds to its title row (its counters
        and newest line stay visible) exactly as it does for the Script panel.
        """
        if not visible:
            return
        if (self.docks.area_of("diagnostics") == "bottom" and self.docks.area_of("messages") == "bottom"
                and not self.diagnostics_dock.is_collapsed() and self._messages_expanded):
            self.set_messages_expanded(False)
        if getattr(self, "_diagnostics_dirty", False) and getattr(self, "diagnostics", None) is not None:
            self._diag_timer.start()

    def _rebind_diagnostics(self, *_args) -> None:
        """The selection, a run or a result changed: rebind now if the panel is on screen, else on show."""
        self._diagnostics_dirty = True
        if self.diagnostics is not None and self.diagnostics_dock.isVisible():
            self._diag_timer.start()

    def _bind_diagnostics(self) -> None:
        """Point the window's Diagnostics panel at the selected simulation's diagnostics.

        Pages that put their own widget in the panel (Simulate) bind that
        themselves; this is the panel's own content, shown on every other page.
        See :func:`diagnostics_log_for` for which log wins. Runs only while the
        panel is visible, because the stored-history fallback reads the result.
        """
        panel = self.diagnostics
        if panel is None or not self.diagnostics_dock.isVisible():
            return
        self._diagnostics_dirty = False
        sim = self.state.simulation
        log_ = diagnostics_log_for(self.jobs, sim.id) if sim is not None else None
        panel.set_log(log_ if log_ is not None else self._live_diagnostics)
        p = self.state.project
        if p is not None and getattr(p, "path", None) is not None and hasattr(panel, "set_export_dir"):
            panel.set_export_dir(p.path)

    def _make_tool_window(self, kind: str) -> QWidget:
        """The Assistant or Script panel content; a plain statement of the problem when it cannot load.

        Both modules are written alongside the shell, so the window must start
        without them (the same rule as for pages): the failure is logged and the
        panel says what is missing.
        """
        mod, cls, title = {"assistant": ("aerolume.app.widgets.assistant_panel", "AssistantPanel", "Assistant"),
                           "script": ("aerolume.app.widgets.script_console", "ScriptConsole", "Script console")}[kind]
        try:
            return getattr(importlib.import_module(mod), cls)(self.state, self.jobs)
        except Exception as exc:  # noqa: BLE001 - one broken tool window never stops the app
            traceback.print_exc()
            QTimer.singleShot(0, self, lambda e=exc: self.state.log(f"{title} could not be loaded: {type(e).__name__}: {e}",
                                                              "error", "Application"))
            w = EmptyState("warning", f"{title} is not available", f"{type(exc).__name__}: {exc}")
            w.load_error = exc
            return w

    def _add_tool_window_buttons(self) -> None:
        """Assistant and Script toggles, then the workspace switcher at the right end of the toolbar."""
        tb = self.main_toolbar
        tb.addSeparator()
        for key, icon, text, tip in (("assistant", "assistant", "Assistant",
                                      "Show or hide the experiment assistant (Alt+3)"),
                                     ("script", "code", "Script", "Show or hide the Script panel (Alt+4)")):
            a = self.docks.actions[key]
            a.setIcon(theme.icon(icon, "text2", active="accent"))
            a.setIconText(text)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            tb.addAction(a)
            w = tb.widgetForAction(a)
            if isinstance(w, QToolButton):
                w.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._toolbar_spacer = tb.addWidget(spacer)
        self.workspace_menu = QMenu("&Workspace", self)
        self.workspace_menu.aboutToShow.connect(lambda: self.docks.fill_workspace_menu(self.workspace_menu))
        self.docks.fill_workspace_menu(self.workspace_menu)
        # a window-owned QAction rather than a cached menuAction() wrapper (see act_import_geometry)
        self.act_workspace = QAction(theme.icon("layout", "text2"), self.workspace_menu.title(), self)
        self.act_workspace.setMenu(self.workspace_menu)
        tb.addAction(self.act_workspace)
        w = tb.widgetForAction(self.act_workspace)
        if isinstance(w, QToolButton):
            w.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            w.setProperty("dropdown", "true")
            w.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            w.setAccessibleName("Workspace")
        self.docks.workspaceChanged.connect(lambda _n: self._sync_workspace_button())
        self.docks.workspaceChanged.connect(lambda _n: self._sync_script_share())
        self._sync_workspace_button()

    def _sync_workspace_button(self) -> None:
        name = self.docks.workspace
        self.act_workspace.setIconText(name)
        hint = docks_mod.WORKSPACE_HINTS.get(name, "")
        tip = f"Workspace: {name}. {hint}" if hint else f"Workspace: {name}"
        self.act_workspace.setToolTip(tip + ".\nEach workspace keeps its own panels, rails and sizes.")
        self.act_workspace.setStatusTip(tip)

    def set_workspace(self, name: str) -> None:
        """Switch to a saved workspace (Default, Analysis, Live, Scripting)."""
        self.docks.set_workspace(name)

    def _sync_messages_from_dock(self) -> None:
        """After a workspace change the Messages panel's fold state is the workspace's."""
        want = not self.docks.is_collapsed("messages")
        if want != self._messages_expanded:
            self.set_messages_expanded(want)

    def _assistant_visibility(self, visible: bool) -> None:
        if visible:
            refresh = getattr(self.assistant, "ensure_checked", None)
            if callable(refresh):
                refresh()

    #: share of the window height the Script dock opens at (the first time in a session)
    SCRIPT_SHARE = 0.40

    def script_height(self) -> int:
        """The Script dock's opening height: 40 % of the window, at least 300 px, at most half.

        At its 150 px default the dock showed five editor lines and a sliver of
        each inline figure; 40 % (400 px at 1000 px) shows a short script, its
        output and a whole figure card. A workspace that sizes its bottom rail by
        share (Scripting: also 40 %, at least ``docks.BOTTOM_SHARE_MIN``) wins, so
        the two never pull the rail to different heights.
        """
        h = max(self.height(), 200)
        share = self.docks.bottom_share() if getattr(self, "docks", None) is not None else None
        if share:
            return int(max(docks_mod.BOTTOM_SHARE_MIN, share * h))
        return int(min(0.5 * h, max(300.0, self.SCRIPT_SHARE * h)))

    def _sync_script_share(self) -> None:
        """Tell the Script console which share of the window height to keep its dock at.

        The console keeps its dock's share across window resizes (a docked widget
        otherwise keeps its pixels and hands all new height to the page); it starts
        from :attr:`SCRIPT_SHARE`. In a workspace that sizes the bottom rail by share
        it must keep that share instead, or the two would pull the rail to different
        heights on every resize; elsewhere it goes back to its own default.
        """
        console = getattr(self, "script_console", None)
        if console is None:
            return
        share = self.docks.bottom_share()
        console.set_dock_share(share)

    def _script_visibility(self, visible: bool) -> None:
        """Script opened: fold Messages to its title row (both share the bottom) and size Script once."""
        if not visible or self.script_dock.isFloating():
            return
        if self._messages_expanded:
            self.set_messages_expanded(False)
        if not self._script_sized:
            self._script_sized = True
            # the dock is laid out after this signal: size it on the next turn of the event loop
            QTimer.singleShot(0, self, self._size_script_dock)

    def _size_script_dock(self) -> None:
        """Size the Script dock (see :meth:`script_height`), capped by what the page above can give up.

        The page keeps its minimum height (the Geometry page needs about 430 px),
        so at 1280x800 the dock gets what is left, about 27 % of the window.
        The layout is settled first, so the height Messages gave back by folding
        counts.
        """
        if self.script_dock.isVisible() and not self.script_dock.isFloating():
            self.layout().activate()
            self.resizeDocks([self.script_dock], [self.script_height()], Qt.Orientation.Vertical)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self.act_new)
        m.addAction(self.act_open)
        self.recent_menu = m.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.examples_menu = m.addMenu("Open &example")
        self.examples_menu.aboutToShow.connect(self._fill_examples_menu)
        m.aboutToShow.connect(self._sync_examples_menu)
        self._sync_examples_menu()
        m.addSeparator()
        m.addMenu(self.import_menu)
        m.addSeparator()
        m.addAction(self.act_save)
        m.addAction(self.act_close)
        m.addSeparator()
        m.addAction(self.act_welcome)
        m.addSeparator()
        m.addAction(self.act_exit)

        m = mb.addMenu("&Edit")
        m.addAction(self.act_rename)
        m.addAction(self.act_duplicate)
        m.addAction(self.act_delete)
        m.addSeparator()
        m.addAction(self.act_prefs)

        m = mb.addMenu("&View")
        for key in STEP_KEYS:
            m.addAction(self.page_actions[key])
        # one checkable entry per panel, at the top level of View under a "Panels" heading (Alt+1 ...)
        self.docks.fill_panels_menu(m)
        a = self.step_toolbar.toggleViewAction()
        a.setText("Step bar")
        a.triggered.connect(lambda checked: self._user_toggled(self.step_toolbar, checked))
        m.addAction(a)
        m.addSeparator()
        m.addMenu(self.workspace_menu)
        tm = m.addMenu("&Theme")
        group = QActionGroup(self)
        self.theme_actions: dict[str, QAction] = {}
        for name in theme.THEME_CHOICES:
            a = QAction("System (follow Windows)" if name == "system" and sys.platform == "win32"
                        else name.capitalize(), self, checkable=True)
            a.setChecked(theme.preference() == name)
            a.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            group.addAction(a)
            tm.addAction(a)
            self.theme_actions[name] = a
        tm.addSeparator()
        tm.addAction(self.act_toggle_theme)
        m.addAction(self.act_full)
        self.view_menu = m

        m = mb.addMenu("&Simulation")
        m.addAction(self.act_run)
        m.addAction(self.act_stop)
        m.addSeparator()
        m.addAction(self.act_new_sim)
        m.addAction(self.act_new_wing_sim)
        m.addAction(self.act_new_cfd3d)
        m.addAction(self.act_explore)
        m.addSeparator()
        m.addAction(self.act_save_results)
        m.aboutToShow.connect(self._sync_save_results_action)
        m.aboutToShow.connect(self._sync_simulation_actions)
        self.simulation_menu = m

        m = mb.addMenu("&Tools")
        m.addAction(self.act_import)
        m.addAction(self.act_import_mesh)
        m.addSeparator()
        m.addAction(self.act_templates)
        m.addAction(self.act_naca)
        m.addAction(self.act_new_wing)
        m.addSeparator()
        m.addAction(self.act_generate)
        m.addSeparator()
        m.addAction(self.act_assistant)
        m.addAction(self.act_run_script)

        m = mb.addMenu("&Help")
        m.addAction(self.act_guide)
        m.addAction(self.act_validation)
        m.addAction(self.act_assistant_guide)
        m.addAction(self.act_scripting_guide)
        m.addAction(self.act_mcp_guide)
        m.addAction(self.act_keys)
        m.addSeparator()
        m.addAction(self.act_tour)
        m.addAction(self.act_welcome)
        m.addSeparator()
        m.addAction(self.act_logs)
        m.addAction(self.act_updates)
        m.addAction(self.act_about)
        self.help_menu = m

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self._ready_label = QLabel("Ready")
        sb.addWidget(self._ready_label, 1)
        # QStatusBar hides its normal widgets for a temporary message only if they are
        # visible at that moment, so a message shown before the window was (a project
        # opened from the command line) painted over "Ready" ("OReadyd My wing").
        # Hide and show the label with the message, explicitly.
        sb.messageChanged.connect(lambda text: self._ready_label.setVisible(not text))
        self.run_status = RunStatus(self.state)
        self.run_status.clicked.connect(lambda: self.state.navigate("simulate"))
        sb.addPermanentWidget(self.run_status)
        self._device_label = QLabel("Detecting device…")
        self._device_label.setProperty("sep", "true")
        sb.addPermanentWidget(self._device_label)
        self._units_label = QLabel()
        self._units_label.setProperty("sep", "true")
        sb.addPermanentWidget(self._units_label)
        self._sync_units_label()
        self._probe = _DeviceProbe(self)
        self._probe.done.connect(self._device_ready)

    # -- device -----------------------------------------------------------------------
    def probe_device(self, device: str = "auto") -> None:
        self._device_label.setText("Detecting device…")
        self._probe.start(device)

    def _device_ready(self, kind: str, text: str, tip: str) -> None:
        self._device_label.setText(text)
        self._device_label.setToolTip(tip)
        theme.set_prop(self._device_label, "role", "status-bad" if kind == "none" else "")

    def device_text(self) -> str:
        return self._device_label.text()

    # -- recent / examples menus ----------------------------------------------------------
    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        items = start_mod.recent_projects(10)
        if not items:
            a = self.recent_menu.addAction("No recent projects")
            a.setEnabled(False)
            return
        for p in items:
            a = self.recent_menu.addAction(p.name.removesuffix(start_mod.PROJECT_SUFFIX))
            a.setToolTip(str(p))
            a.setStatusTip(str(p))
            a.triggered.connect(lambda _=False, path=str(p): self.open_project_path(path))
        self.recent_menu.addSeparator()
        clear = self.recent_menu.addAction("Clear recent")
        clear.triggered.connect(lambda: start_mod.clear_recent())

    def _sync_examples_menu(self) -> None:
        """Hide File ▸ Open example when no example projects are installed (no dead entry)."""
        self.examples_menu.menuAction().setVisible(bool(start_mod.example_projects()))

    def _fill_examples_menu(self) -> None:
        self.examples_menu.clear()
        items = start_mod.example_projects()
        if not items:
            a = self.examples_menu.addAction("No examples installed")
            a.setEnabled(False)
            return
        for p in items:
            a = self.examples_menu.addAction(start_mod.summarize(p).name)
            a.setStatusTip(str(p))
            a.triggered.connect(lambda _=False, path=str(p): self.open_example(path))

    # -- pages ----------------------------------------------------------------------
    def page(self, key: str) -> QWidget:
        """The page widget for a key, creating it on first use."""
        if key not in self._pages:
            w = self._create_page(key)
            self._pages[key] = w
            self.stack.addWidget(w)
        return self._pages[key]

    def _create_page(self, key: str) -> QWidget:
        spec = _SPEC[key]
        try:
            if key == "start":
                cls = start_mod.StartPage
            else:
                mod = importlib.import_module(f"aerolume.app.pages.{key}")
                cls = getattr(mod, spec.cls)
            w = cls(self.state, self.jobs)
        except Exception as exc:  # noqa: BLE001 - any page failure becomes a placeholder
            ph = PlaceholderPage(spec, exc, traceback.format_exc())
            ph._state = self.state
            ph.retryRequested.connect(self._retry_page)
            self.state.log(f"{spec.title} page could not be loaded: {type(exc).__name__}: {exc}", "error",
                           "Application")
            return ph
        self.docks.adopt_page_panels(key, w)
        if key == "start":
            w.newStudyRequested.connect(self.new_project_dialog)
            w.openRequested.connect(self.open_project_dialog)
            w.openPathRequested.connect(self.open_project_path)
            w.openExampleRequested.connect(self.open_example)
            help_sig = getattr(w, "helpRequested", None)
            if help_sig is not None:
                help_sig.connect(self.show_help)
        return w

    def _retry_page(self, key: str) -> None:
        old = self._pages.pop(key, None)
        for name in [m for m in sys.modules if m == f"aerolume.app.pages.{key}"]:
            del sys.modules[name]
        importlib.invalidate_caches()
        w = self.page(key)
        if old is not None:
            self.stack.removeWidget(old)
            old.deleteLater()
        self.stack.setCurrentWidget(w)
        self._call(w, "on_shown")

    def is_placeholder(self, key: str) -> bool:
        return isinstance(self._pages.get(key), PlaceholderPage)

    def current_page_key(self) -> str:
        return self._current

    def show_page(self, key: str) -> QWidget:
        """Make a page current (no project check) and call its ``on_shown``."""
        w = self.page(key)
        self.stack.setCurrentWidget(w)
        self._current = key
        self.step_bar.set_current(key if key in STEP_KEYS else "")
        for k, a in self.page_actions.items():
            a.setCheckable(k in STEP_KEYS)
            if k in STEP_KEYS:
                a.setChecked(k == key)
        self.docks.show_page(key)
        self._set_workspace_visible(key != "start")
        self._call(w, "on_shown")
        self._unfocus_timer.start()
        if key != "start" and self.state.project is not None:
            self._maybe_continue_tour()
        return w

    def _call(self, w: QWidget, method: str, *args) -> None:
        fn = getattr(w, method, None)
        if not callable(fn):
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - a page bug must not kill navigation
            traceback.print_exc()
            title = _SPEC.get(self._current, PAGES[0]).title
            self.state.log(f"{type(exc).__name__}: {exc}", "error", title)
            self._status(f"{title}: {type(exc).__name__}: {exc}", 8000)

    def _navigate(self, target: str) -> None:
        key, _, action = target.partition(":")
        key = key.strip().lower()
        if key not in _SPEC:
            self._status(f"Unknown page “{key}”", 4000)
            return
        if key != "start" and self.state.project is None:
            self.show_page("start")
            self._status("Create or open a project first.", 6000)
            return
        w = self.show_page(key)
        if action:
            self._call(w, "handle_action", action)

    # -- workspace (docks + step bar) visibility ------------------------------------------
    _STEP_BAR_KEY = "window/step_bar"

    def _load_step_bar_pref(self) -> bool:
        s = app_settings.get()
        v = s.value(self._STEP_BAR_KEY, None)
        if v is None:                           # v2 kept it in "window/docks" with the panels
            for part in str(s.value("window/docks", "") or "").split(","):
                k, _, val = part.partition("=")
                if k == "stepToolbar":
                    return val == "1"
            return True
        return str(v).strip().lower() not in ("false", "0", "no", "off")

    def _migrate_dock_prefs(self) -> None:
        """Carry a v2 "window/docks" visibility list into the Default workspace, once.

        v2 stored which docks were shown as ``projectDock=1,messagesDock=0,...``;
        v3 keeps whole workspaces. Without this a user who had hidden Messages
        would find it back after the update.
        """
        s = app_settings.get()
        raw = str(s.value("window/docks", "") or "")
        if not raw or s.value("workspaces/current", None) is not None:
            return
        for part in raw.split(","):
            k, _, v = part.partition("=")
            key = KEY_OF_OBJECT.get(k)
            if key is not None:
                self.docks._wanted[key] = v == "1"
        self.docks._apply_visibility()
        self.docks.save()

    def _user_toggled(self, w: QWidget, visible: bool) -> None:
        """Record a panel the user showed or hid (View menu, header close, context menu).

        Only explicit user actions change the preference; the welcome page
        hiding everything must not. Panels are the dock manager's; the step bar
        is a toolbar with its own setting.
        """
        name = w.objectName()
        if name == "stepToolbar":
            self._step_bar_wanted = bool(visible)
            app_settings.get().set_value(self._STEP_BAR_KEY, bool(visible))
        elif name in KEY_OF_OBJECT:
            self.docks.set_wanted(KEY_OF_OBJECT[name], visible)

    def panel_wanted(self, name: str) -> bool:
        """Whether the user wants a panel shown, by object name ("projectDock", "messagesDock",
        "assistantDock", "scriptDock", "propertiesDock", "flowvisDock", "diagnosticsDock") or
        "stepToolbar"."""
        if name == "stepToolbar":
            return self._step_bar_wanted
        key = KEY_OF_OBJECT.get(name)
        return self.docks.wanted(key) if key is not None else True

    def _set_workspace_visible(self, visible: bool) -> None:
        """Hide the panels and step bar on the welcome page; restore the user's choice elsewhere."""
        messages_was_hidden = self.messages_dock.isHidden()
        self.docks.set_workspace_visible(visible)
        self.step_toolbar.setVisible(visible and self._step_bar_wanted)
        self._workspace_shown = visible
        if messages_was_hidden and not self.messages_dock.isHidden() and self._messages_expanded:
            # resizeDocks on a hidden dock does nothing, so a log that expanded while hidden (a
            # warning while opening a project from the welcome page) gets its height now
            self._apply_messages_height()

    def _reveal_messages(self) -> None:
        """Show and expand the Messages panel after a failure (unless on the welcome page)."""
        if self._workspace_shown and self.messages_dock.isHidden():
            self.docks.show_panel("messages", raise_=False)
        self.set_messages_expanded(True)
        self.messages_dock.raise_()

    # -- Messages dock: collapsed to its title row until something needs attention -----------
    _MESSAGES_KEY = "window/messages_heights"

    def messages_expanded(self) -> bool:
        return self._messages_expanded

    def set_messages_expanded(self, expanded: bool, *, user: bool = False) -> None:
        """Fold the Messages panel to its header (counts + newest line) or open the log.

        Folding hides the table and caps the panel at the header height, so the
        page above gets the room back (at 1280x800 an always-open log took 19 %
        of the window for one "Opened project" line). Opening restores the
        height the user last chose at this window size. The fold state is part
        of the workspace, like every panel's.
        """
        expanded = bool(expanded)
        if user and not expanded:
            self._messages_user_collapsed = True
        dock = self.messages_dock
        dock.set_collapsed(not expanded, emit=False)
        self.messages.view.setVisible(expanded)
        self.messages.latest.setVisible(not expanded)
        tip = "Collapse the message log" if expanded else "Show the message log"
        self.messages_toggle.setToolTip(tip)
        self.messages_toggle.setAccessibleName(tip)
        was = self._messages_expanded
        self._messages_expanded = expanded
        if expanded and not was:
            self._apply_messages_height()
        if was != expanded:
            self.docks._collapsed_changed("messages", not expanded)

    def _messages_bucket(self) -> int:
        """Window heights are bucketed to 100 px, so a height chosen at 1000 px is reused at 1003."""
        return int(round(max(self.height(), 1) / 100.0) * 100)

    def _default_messages_height(self) -> int:
        h = self.height()
        return 150 if h >= 950 else max(100, int(0.16 * h))

    def messages_height(self) -> int:
        """The log height used when it opens at the current window size."""
        h = self._messages_heights.get(self._messages_bucket(), self._default_messages_height())
        return max(80, min(int(h), int(0.6 * max(self.height(), 200))))

    def _apply_messages_height(self) -> None:
        if not self._messages_expanded:
            return
        self._applying_messages_height = True
        try:
            self.resizeDocks([self.messages_dock], [self.messages_height()], Qt.Orientation.Vertical)
        finally:
            self._applying_messages_height = False

    def _load_messages_heights(self) -> dict[int, int]:
        raw = str(app_settings.get().value(self._MESSAGES_KEY, "") or "")
        out: dict[int, int] = {}
        for part in raw.split(","):
            k, _, v = part.partition(":")
            try:
                out[int(k)] = int(v)
            except ValueError:
                continue
        return out

    def _save_messages_heights(self) -> None:
        app_settings.get().set_value(self._MESSAGES_KEY,
                                     ",".join(f"{k}:{v}" for k, v in sorted(self._messages_heights.items())))

    #: sources whose warnings the pages already show where they apply (Setup's planner list, the Run
    #: panel's warning row, the Results header), so the log never opens for them
    _SHOWN_ON_PAGE = frozenset({"Planner"})

    def _message_added(self, level: str, source: str, _text: str) -> None:
        """Open the log for an error only.

        A warning stays in the folded title row, which already counts it and
        shows it in the warning colour. Opening the log for warnings took 140 to
        160 px from every 3D page for a planner note the page itself showed
        three times; at 1280x800 it pushed the Run panel's monitor plot out of
        view.
        """
        if self._messages_expanded or level != "error" or source in self._SHOWN_ON_PAGE:
            return
        self.set_messages_expanded(True)

    def eventFilter(self, obj, e) -> bool:
        """Remember the log height the user drags to, per window size.

        A dock's resize event comes from either a separator drag (the user's
        choice, worth keeping) or the window itself being resized (not a
        choice). Object event filters run before the layout reacts, so the
        window's own Resize marks the dock resizes that follow as "not the user".
        """
        et = e.type()
        if obj is self and et == QEvent.Type.Resize:
            self._window_resizing = True
            QTimer.singleShot(0, self, self._window_resize_done)
        elif (obj is getattr(self, "messages_dock", None) and et == QEvent.Type.Resize
              and self._messages_expanded and not self._window_resizing
              and not self._applying_messages_height and self.messages_dock.isVisible()
              and not self.messages_dock.isFloating()):
            h = self.messages_dock.height()
            if h > DockTitle.HEIGHT + 20:
                self._messages_heights[self._messages_bucket()] = h
        return super().eventFilter(obj, e)

    def _window_resize_done(self) -> None:
        self._window_resizing = False

    # -- run control ---------------------------------------------------------------------
    def run_target(self) -> tuple[str, str]:
        """What Run (F5) would do now: ``(kind, text)``.

        ``kind`` is "run" (the current simulation belongs to the current design),
        "setup" (the current design has no simulation yet: Run becomes "Set up
        simulation…"), or "none" with the reason in ``text``. Run never acts on
        a simulation of a design other than the one on screen: before this
        check, adding a Cylinder left NACA 2412's polar current, and F5 ran it.
        """
        st = self.state
        proj, d, sim = st.project, st.design, st.simulation
        if proj is None:
            return "none", "Open a project first."
        if d is None:
            return "none", "Add a design on the Geometry step first."
        if sim is not None and sim.design_id == d.id:
            if self.jobs is None:
                return "none", "The simulation runner is unavailable (see Messages)."
            running, queued = self._busy_jobs()
            if sim.id == running:
                return "none", f"“{sim.name}” is already running."
            if sim.id in queued:
                return "none", f"“{sim.name}” is queued."
            return "run", sim.name
        if not any(s.design_id == d.id for s in proj.simulations.values()):
            return "setup", d.name
        return "none", f"Select a simulation of “{d.name}” in the Project panel."

    def run_current(self) -> None:
        """Run the current design's selected simulation (F5) - routed through the Simulate page.

        A design without simulations is sent to Setup instead, so F5 always
        moves the user's own design forward.
        """
        kind, text = self.run_target()
        if kind == "run":
            self.state.navigate("simulate:run")
        elif kind == "setup":
            self.state.navigate("setup:new")
            self._status(f"Set up a simulation of “{text}”, then run it.", 6000)
        elif self.state.project is not None:
            self._status(text, 5000)

    _run_from_shortcut = run_current

    def stop_current(self) -> None:
        jobs = self.jobs
        if jobs is None:
            return
        try:
            rid = jobs.running_id()
        except Exception:  # noqa: BLE001
            rid = None
        if rid:
            jobs.cancel(rid)
            self._status(self.STOPPING, 10000)

    STOPPING = "Stopping…"

    def _sync_save_results_action(self) -> None:
        n = 0
        with contextlib.suppress(Exception):
            n = len(self.jobs.unsaved_ids()) if self.jobs is not None else 0
        self.act_save_results.setEnabled(n > 0)
        self.act_save_results.setText(f"Save unsaved &results ({n})" if n else "Save unsaved &results")

    def _retry_result_saves(self) -> None:
        retry = getattr(self.jobs, "retry_saves", None)
        if callable(retry):
            retry()
            self._status("Saving the results again…", 4000)

    def _clear_stopping(self) -> None:
        """Drop "Stopping…" once the job manager has moved on.

        The stopped run ends with jobCancelled and the next queued one starts
        right after; leaving the message up until its timeout made the status
        bar say "Stopping…" next to "Running <next one>". Only that message is
        cleared - anything posted since is left alone.
        """
        sb = self.statusBar()
        if sb.currentMessage() == self.STOPPING:
            sb.clearMessage()

    def _update_run_actions(self) -> None:
        kind, text = self.run_target()
        a = self.act_run
        if kind == "setup":
            a.setText("&Set up simulation…")
            a.setIconText("Set up simulation…")
            tip = f"“{text}” has no simulation yet: set one up (F5)"
        else:
            a.setText("&Run")
            a.setIconText("Run")
            tip = f"Run “{text}” (F5)" if kind == "run" else f"Run (F5): {text}"
        a.setToolTip(tip)
        a.setStatusTip(tip)
        a.setEnabled(kind in ("run", "setup"))
        rid = None
        if self.jobs is not None:
            try:
                rid = self.jobs.running_id()
            except Exception:  # noqa: BLE001
                rid = None
        self.act_stop.setEnabled(bool(rid))
        # the window's Stop acts on whatever runs, which need not be the selected simulation
        proj = self.state.project
        running_sim = proj.simulations.get(rid) if (rid and proj is not None) else None
        stop_tip = (f"Stop the running simulation, “{running_sim.name}” (Shift+F5)" if running_sim is not None
                    else "Stop the running simulation (Shift+F5)")
        self.act_stop.setToolTip(stop_tip)
        self.act_stop.setStatusTip(stop_tip)

    # -- selection-dependent actions -----------------------------------------------------
    def _selected_id(self) -> str:
        _kind, oid = self.project_tree.selected()
        return oid or self.state.simulation_id or self.state.design_id

    def _rename_selected(self) -> None:
        oid = self._selected_id()
        if oid:
            if self.project_dock.isHidden() and self._workspace_shown:
                self.docks.show_panel("project")
            elif self.project_dock.is_collapsed():
                self.docks.set_collapsed("project", False, user=True)
            self.project_tree.rename(oid)

    def _duplicate_selected(self) -> None:
        oid = self._selected_id()
        if oid:
            self.project_tree.duplicate(oid)

    def _delete_selected(self) -> None:
        oid = self._selected_id()
        if oid:
            self.project_tree.delete(oid)

    # -- state reactions ---------------------------------------------------------------
    def _project_changed(self) -> None:
        proj = self.state.project
        has = proj is not None
        self.step_bar.set_all_enabled(has)
        for key in STEP_KEYS:
            self.page_actions[key].setEnabled(has)
        for a in (self.act_save, self.act_close, self.act_import, self.act_import_mesh,
                  self.act_import_geometry, self.act_templates, self.act_naca, self.act_new_wing, self.act_report, self.act_generate,
                  self.act_new_sim, self.act_explore, self.act_rename, self.act_duplicate, self.act_delete):
            # one dead action must not abort the rest of the update (title, step bar, selection)
            if shiboken_valid(a):
                a.setEnabled(has)
        name = start_mod.project_name(proj) if has else ""
        self.setWindowTitle(f"{name} — Aerolume" if has else "Aerolume")
        self._update_steps()
        self._selection_changed()
        if not has and self._current not in ("", "start"):
            self.show_page("start")

    def _keep_selection_coherent(self, design_id: str) -> None:
        """Keep the current simulation one of the current design's (or none).

        ``AppState.select_design`` changes only the design, so selecting or
        adding a design (template, import, NACA, "Open in Geometry") used to
        leave the previous design's simulation current: the breadcrumb read
        "Cylinder › Polar · Re 1.0×10⁶" and Run ran NACA 2412. This picks the
        design's first simulation, or none when it has none.
        """
        st = self.state
        proj = st.project
        if proj is None:
            return
        sim = st.simulation
        if sim is not None and sim.design_id == design_id:
            return
        target = ""
        if design_id:
            target = next((s.id for s in proj.simulations.values() if s.design_id == design_id), "")
        if st.simulation_id != target:
            st.select_simulation(target)

    def _selection_changed(self) -> None:
        d, s = self.state.design, self.state.simulation
        tag = design_tag(d)
        self.step_bar.set_context(breadcrumb_text(d, s), "" if tag == "2D" else tag)
        self._sync_simulation_actions()
        self._update_steps()                    # "Results" is done only for a finished current run

    def _sync_simulation_actions(self) -> None:
        """Simulation-menu entries for what the selected design can have.

        2D designs keep "New simulation" (Setup decides CFD or polar). A wing
        gets "New wing analysis" and "New 3D CFD" (of the lofted wing); a 3D
        mesh gets "New 3D CFD". Explore live steers a CFD run, so it is off
        for a wing analysis and a polar, with the reason in the tooltip.
        """
        d, s = self.state.design, self.state.simulation
        has = self.state.project is not None and d is not None
        wing, mesh3d = is_wing(d), is_3d(d) and not is_wing(d)
        self.act_new_sim.setVisible(not wing and not mesh3d)
        self.act_new_sim.setEnabled(has)
        self.act_new_wing_sim.setVisible(wing)
        self.act_new_wing_sim.setEnabled(has and wing)
        self.act_new_cfd3d.setVisible(wing or mesh3d)
        self.act_new_cfd3d.setEnabled(has and (wing or mesh3d))
        self.act_new_cfd3d.setText("New 3D &CFD of the wing" if wing else "New 3D &CFD")
        cfd = s is not None and s.analysis == "cfd"
        self.act_explore.setEnabled(self.state.project is not None and (s is None or cfd))
        tip = ("Real-time flow you can steer while it runs" if s is None or cfd else
               f"{analysis_label(s, d)} analyses have no live flow: Explore live steers a CFD run")
        self.act_explore.setToolTip(tip)
        self.act_explore.setStatusTip(tip)

    def step_states(self) -> dict[str, bool]:
        """Which workflow steps have data (drives the step bar's check glyphs)."""
        proj = self.state.project
        out = dict.fromkeys(STEP_KEYS, False)
        if proj is None:
            return out
        sims = list(proj.simulations.values())
        done = [s for s in sims if s.status == "done"]
        cur = self.state.simulation
        out["geometry"] = bool(proj.designs)
        out["setup"] = bool(sims)
        out["simulate"] = bool(done)
        out["results"] = cur is not None and cur.status == "done"
        out["compare"] = len(done) >= 2 or len(self.state.compare_ids) >= 2
        root = start_mod.project_path(proj)
        reports = root / "reports" if root else None
        out["report"] = bool(reports and reports.is_dir() and any(reports.iterdir()))
        return out

    def _update_steps(self) -> None:
        self.step_bar.set_states(self.step_states())
        self._update_run_actions()

    def _status(self, text: str, timeout_ms: int = 4000) -> None:
        self.statusBar().showMessage(text, max(int(timeout_ms), 0))

    def _settings_changed(self, key: str) -> None:
        if key == "units/system":
            self._sync_units_label()
        elif key == "compute/device":
            self.probe_device(app_settings.get().device)

    def _sync_units_label(self) -> None:
        s = app_settings.get()
        self._units_label.setText(s.unit_system)
        self._units_label.setToolTip(f"Display units: {s.display_unit('speed')}, {s.display_unit('length')}, "
                                     f"{s.display_unit('force')} (Edit ▸ Preferences)")

    def _theme_changed(self, name: str) -> None:
        # "system" stays checked while the OS decides; otherwise the theme that was applied
        a = self.theme_actions.get(theme.preference())
        if a is not None:
            a.setChecked(True)
        self.update()

    # -- project actions ------------------------------------------------------------
    def _busy_jobs(self) -> tuple[str | None, list[str]]:
        """(running id, queued ids) of the job manager; (None, []) without one."""
        jobs = self.jobs
        if jobs is None:
            return None, []
        try:
            return jobs.running_id(), list(jobs.queued_ids())
        except Exception:  # noqa: BLE001 - a broken job manager has nothing to lose
            return None, []

    def _busy_text(self, running: str | None, queued: list[str]) -> str:
        """'A simulation is running (43 %)' / '2 simulations are queued' for the questions below."""
        if running:
            pct = ""
            with contextlib.suppress(Exception):
                pct = f" ({100 * float(self.jobs.progress_of(running)[0]):.0f} %)"
            sim = self.state.project.simulations.get(running) if self.state.project is not None else None
            name = f"“{sim.name}”" if sim is not None else "A simulation"
            more = f" and {len(queued)} more {'is' if len(queued) == 1 else 'are'} queued" if queued else ""
            return f"{name} is running{pct}{more}."
        return f"{len(queued)} simulation{'s are' if len(queued) != 1 else ' is'} queued."

    def _ask_switch(self, action: str = "switch project") -> bool:
        """Ask before abandoning running or queued simulations; True when it is fine to go on.

        Cancels nothing: callers ask first, do whatever can still fail (a file
        dialog, opening a project that may be damaged), and stop the jobs with
        :meth:`_stop_jobs` only right before the switch actually happens.
        """
        running, queued = self._busy_jobs()
        if running is None and not queued:
            return True
        ans = QMessageBox.question(self, "Simulation running",
                                   f"{self._busy_text(running, queued)} Stop it and {action}?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        return ans == QMessageBox.StandardButton.Yes

    def _stop_jobs(self) -> None:
        """Cancel the queued simulations, then the running one (best effort)."""
        running, queued = self._busy_jobs()
        for sid in [*queued, *([running] if running else [])]:
            try:
                self.jobs.cancel(sid)
            except Exception as exc:  # noqa: BLE001 - best effort; the job may have just ended
                self._status(f"Could not stop {sid}: {exc}", 6000)

    def _confirm_switch(self) -> bool:
        """Ask, then stop the jobs: for actions with nothing left that can fail (Close project)."""
        if not self._ask_switch():
            return False
        self._stop_jobs()
        return True

    def set_project(self, project, messages: list[str] | None = None, navigate_to: str | None = None) -> None:
        """Make ``project`` current (saving the old one) and optionally jump to a page."""
        if self.state.project is not None and self.state.project is not project:
            try:
                self.state.save()
            except Exception as exc:  # noqa: BLE001
                self.state.log(f"Could not save the previous project: {exc}", "error", "Project")
        self.state.set_project(project)
        for m in messages or []:
            self.state.log(m, "warning", "Project")
        if messages:
            self._status(" ".join(messages), 10000)
        elif project is not None:
            self._status(f"Opened {start_mod.project_name(project)}", 4000)
        if project is not None:
            where = start_mod.project_path(project)
            self.state.log(f"Opened {start_mod.project_name(project)}" + (f" ({where})" if where else ""),
                           "info", "Project")
        if navigate_to:
            self.state.navigate(navigate_to)

    def new_study(self, kind: str, name: str, folder: str | Path, *, stop_jobs: bool = False,
                  mesh_path: str | Path | None = None) -> object:
        """Create a study without dialogs (menus, start page and tests share this path).

        Creating the starter geometry and building the Geometry page takes
        about 1.7 s on the GUI thread the first time, so the wait cursor says
        the click registered. ``stop_jobs`` stops running and queued simulations
        once the study exists (the caller has asked the user), so a study that
        cannot be created never costs a run.

        A "mesh" study is created empty and opened, then the mesh import runs
        into it (``mesh_path``, or a file the import dialog asks for): the
        dialog sees the new project, and a cancelled import leaves a valid
        empty study rather than none.
        """
        with _busy_cursor():
            project, messages = start_mod.create_study(kind, name, folder)
            if stop_jobs:
                self._stop_jobs()
            self._leave_tool_workspace()
            self.set_project(project, messages, navigate_to="geometry")
        if not messages:
            self._status(f"Created {start_mod.project_name(project)}", 5000)
        if kind == "mesh":
            added = self.import_mesh(str(mesh_path) if mesh_path else None, starter=True)
            if not added:
                self.state.log("No mesh was imported. Use File ▸ Import geometry ▸ 3D mesh to add one.",
                               "info", "Project")
        return project

    def _leave_tool_workspace(self) -> None:
        """A new study opens in the user's last study workspace, not in Scripting.

        A "Heated room" study created from the Scripting workspace opened with an
        empty "Untitled" script taking about 520 of 1000 px and a Setup page of
        300 px under it (v3.4 audit). The Script panel's text is kept; View ▸
        Workspace ▸ Scripting brings it back.
        """
        docks = getattr(self, "docks", None)
        if docks is None:
            return
        target = docks.study_workspace()
        if target != docks.workspace:
            self.set_workspace(target)
            self.state.log(f"The new study opens in the {target} workspace. View ▸ Workspace ▸ Scripting "
                           "brings the Script panel back.", "info", "Project")

    # -- keyboard regions: Tab order and F6 --------------------------------------------
    def focus_regions(self) -> list[tuple[str, QWidget, QWidget | None]]:
        """The window's keyboard regions in visiting order: ``(name, root, header)``.

        The step bar, then the page, then each panel on screen rail by rail (left,
        right, bottom, in rail order); ``header`` is a panel's header (where F6
        lands), None for the step bar and the page.
        """
        out: list[tuple[str, QWidget, QWidget | None]] = []
        bar = getattr(self, "step_toolbar", None)
        if bar is not None and bar.isVisible():
            out.append(("steps", bar, None))
        out.append(("page", self.centralWidget(), None))
        docks = getattr(self, "docks", None)
        if docks is not None:
            for h in docks.headers_in_order():
                out.append((h.dock.objectName(), h.dock, h))
        return out

    @staticmethod
    def _is_tab_stop(w: QWidget, window: QWidget) -> bool:
        """Qt's own test for a Tab stop (QApplicationPrivate::focusNextPrevChild_helper)."""
        return (bool(w.focusPolicy() & Qt.FocusPolicy.TabFocus) and w.focusProxy() is None
                and w.isEnabled() and not w.isWindow() and w.window() is window and w.isVisibleTo(window))

    def _chain(self) -> tuple[list[tuple[tuple[int, int], QWidget]], dict[int, tuple[int, int]], list]:
        """Every widget of Qt's focus chain keyed by (region, place in the chain), and the Tab stops sorted.

        Qt's chain is creation order, and the panels are built before most of a
        page's widgets, so Tab from the step bar crossed every panel's chrome and
        content (29 stops in the Default workspace) before the Setup form's first
        field (v3.4 audit, WCAG 2.4.3). Sorting by region keeps Qt's order inside
        each region and puts the page first.
        """
        regions = self.focus_regions()

        def region_of(x: QWidget) -> int:
            for i, (_n, root, _h) in enumerate(regions):
                if root is x or root.isAncestorOf(x):
                    return i
            return len(regions)

        keys: dict[int, tuple[int, int]] = {}
        stops: list[tuple[tuple[int, int], QWidget]] = []
        w, i = self.nextInFocusChain(), 0
        while w is not None and w is not self and i < 50000:
            k = (region_of(w), i)
            keys[id(w)] = k
            if self._is_tab_stop(w, self):
                stops.append((k, w))
            w, i = w.nextInFocusChain(), i + 1
        stops.sort(key=lambda t: t[0])
        return stops, keys, regions

    def tab_chain(self) -> list[QWidget]:
        """The window's Tab stops in the order Tab visits them: step bar, page, then the panels."""
        return [w for _k, w in self._chain()[0]]

    def focusNextPrevChild(self, next: bool) -> bool:
        """Tab / Shift+Tab inside the window: the region order of :meth:`_chain`, wrapping round."""
        stops, keys, _regions = self._chain()
        if not stops:
            return super().focusNextPrevChild(next)
        cur = QApplication.focusWidget()
        here = keys.get(id(cur)) if cur is not None else None
        if here is None:
            target = stops[0][1] if next else stops[-1][1]
        elif next:
            next_stop = None
            for k, w in stops:
                if k > here:
                    next_stop = w
                    break
            target = next_stop if next_stop is not None else stops[0][1]
        else:
            prev_stop = None
            for k, w in stops:
                if k >= here:
                    break
                prev_stop = w
            target = prev_stop if prev_stop is not None else stops[-1][1]
        target.setFocus(Qt.FocusReason.TabFocusReason if next else Qt.FocusReason.BacktabFocusReason)
        return True

    def cycle_region(self, step: int = 1) -> str:
        """F6 / Shift+F6: focus the next (previous) region; returns its name ("" when there is none).

        The step bar and the page get back the widget that last had focus in
        them (else their first Tab stop); a panel gets its header, where Enter
        collapses it, Shift+F10 opens its options and Tab goes on into it. A
        region with nothing to focus is passed over.
        """
        stops, _keys, regions = self._chain()
        if not regions:
            return ""
        cur = QApplication.focusWidget()
        idx = -1
        if cur is not None:
            for i, (_n, root, _h) in enumerate(regions):
                if root is cur or root.isAncestorOf(cur):
                    idx = i
                    break
        n = len(regions)
        start = idx if idx >= 0 else (-1 if step > 0 else 0)
        for k in range(1, n + 1):
            j = (start + step * k) % n
            name, root, header = regions[j]
            if header is not None and callable(getattr(header, "focus_header", None)):
                header.focus_header()
                return name
            last = self._region_focus.get(name)
            if (last is not None and shiboken_valid(last) and self._is_tab_stop(last, self)
                    and root.isAncestorOf(last)):
                last.setFocus(Qt.FocusReason.ShortcutFocusReason)
                return name
            first = next((w for (r, _i), w in stops if r == j), None)
            if first is not None:
                first.setFocus(Qt.FocusReason.ShortcutFocusReason)
                return name
        return ""

    def _remember_region_focus(self, _old, new) -> None:
        """Remember which widget had focus in the step bar and the page (F6 returns to it)."""
        if not shiboken_valid(self) or new is None or not shiboken_valid(new) or new.window() is not self:
            return
        central = self.centralWidget()
        if central is not None and central.isAncestorOf(new):
            self._region_focus["page"] = new
        elif getattr(self, "step_toolbar", None) is not None and self.step_toolbar.isAncestorOf(new):
            self._region_focus["steps"] = new

    # -- shell commands (state.request) -----------------------------------------------
    def run_command(self, command: str) -> None:
        """Carry out an ``AppState.request`` command ("import-mesh[:<path>]", "new-wing")."""
        name, _, arg = str(command).partition(":")
        if self.state.project is None:
            self._status("Create or open a project first.", 6000)
            return
        if name == "import-mesh":
            self.import_mesh(arg or None)
        elif name == "new-wing":
            self.new_wing()
        else:
            self._status(f"Unknown command “{name}”", 4000)

    def import_mesh(self, path: str | None = None, *, starter: bool = False) -> list:
        """Import a 3D mesh (STL / OBJ / PLY) into the open project; returns the designs added.

        Goes through the mesh import dialog (:data:`MESH_IMPORT_HOOK`): file,
        units with the guessed scale, orientation. ``starter`` also gives each
        new design the starter CFD run of a new 3D study, so Run works at once.
        Every design added is logged; the first is selected and Geometry shown.
        """
        if self.state.project is None:
            self._status("Create or open a project first.", 6000)
            return []
        try:
            hook = mesh_import_hook()
        except Exception as exc:  # noqa: BLE001 - a broken dialog module must not take the window down
            self.state.log(f"The 3D mesh import dialog could not be loaded: {type(exc).__name__}: {exc}",
                           "error", "Geometry")
            return []
        try:
            if hook is None:
                designs = self._import_mesh_without_dialog(path)
            else:
                kwargs: dict = {"path": path} if path and _accepts_keyword(hook, "path") else {}
                if _accepts_keyword(hook, "add"):
                    kwargs["add"] = self._add_imported_design
                designs = hook(self, self.state, **kwargs)
        except Exception as exc:  # noqa: BLE001 - unreadable file, bad mesh: say so, keep going
            traceback.print_exc()
            self.state.log(f"3D import failed: {type(exc).__name__}: {exc}", "error", "Geometry")
            return []
        if not designs:
            return []
        proj = self.state.project
        added = []
        for d in designs:
            if d.id not in proj.designs:
                self.state.add_design(d, select=False)
            added.append(d)
            if starter:
                for sim in start_mod.starter_simulations("mesh", d):
                    self.state.add_simulation(sim, select=False)
            self.state.log(f"Imported “{d.name}”: {self._design_summary(d)}", "info", "Geometry")
        if added:
            self.project_tree.select_design(added[0].id)
            self.state.navigate("geometry")
        return added

    def _add_imported_design(self, design) -> None:
        """How the mesh dialog adds a design: a name not yet used in the project, not selected
        yet (:meth:`import_mesh` selects the first one once all are in)."""
        design.name = self.state.unique_design_name(design.name)
        self.state.add_design(design, select=False)

    @staticmethod
    def _design_summary(d) -> str:
        return design_description(d)

    def _ask_mesh_file(self) -> str:
        """File dialog for a mesh (tests patch this)."""
        pattern = " ".join(f"*{e}" for e in start_mod.MESH_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Import a 3D mesh", app_settings.get().last_dir,
                                              f"3D meshes ({pattern});;All files (*)")
        return path

    def _import_mesh_without_dialog(self, path: str | None) -> list:
        """Fallback when the build has no mesh import dialog: automatic units, generic body."""
        path = path or self._ask_mesh_file()
        if not path:
            return []
        from .. import geometry3d as g3
        notes: list[str] = []
        with _busy_cursor():
            meshes = g3.read_mesh(path, notes=notes)
            stem = Path(path).stem.replace("_", " ").strip() or "Mesh"
            design = g3.design_from_meshes(meshes, self.state.unique_design_name(stem[:1].upper() + stem[1:]),
                                           kind="generic", units="auto", notes=notes)
        app_settings.get().last_dir = Path(path).parent
        for n in design.source.get("import_notes", []) if isinstance(design.source, dict) else []:
            self.state.log(str(n), "info", "Geometry")
        self.state.log("The mesh import dialog is not part of this build: the file was read with the "
                       "automatic unit guess as a generic body. Check the size on the Geometry step.",
                       "warning", "Geometry")
        return [design]

    def new_wing(self) -> object | None:
        """Add the default tapered wing and show it.

        The Geometry page's own "New wing" does it when the page has one (its
        naming, logging and undo history apply); otherwise the wing of
        :func:`pages.start.default_wing_spec` is added here.
        """
        if self.state.project is None:
            return None
        page = self.page("geometry")
        if callable(getattr(page, "new_wing", None)) and not self.is_placeholder("geometry"):
            before = set(self.state.project.designs)
            self.state.navigate("geometry:wing")
            added = [d for i, d in self.state.project.designs.items() if i not in before]
            return added[0] if added else None
        from ..geometry3d import design_from_wing
        d = design_from_wing(start_mod.default_wing_spec(), self.state.unique_design_name("Tapered wing"))
        self.state.add_design(d, select=False)
        self.project_tree.select_design(d.id)
        self.state.log(f"Added “{d.name}”: {self._design_summary(d)}", "info", "Geometry")
        self.state.navigate("geometry")
        return d

    def new_simulation_of_current(self, analysis: str) -> object | None:
        """Simulation ▸ New wing analysis / New 3D CFD for the selected design, then Setup."""
        d = self.state.design
        if self.state.project is None or d is None:
            return None
        try:
            sim = self.state.new_simulation(d.id, analysis)
        except ValueError as exc:
            self._status(str(exc), 6000)
            return None
        self.state.log(f"Added simulation “{sim.name}”", "info", "Project")
        self.state.navigate("setup")
        return sim

    # The project-switching entry points ask about running simulations only once the
    # target is known, and stop them only right before the switch: a cancelled folder
    # dialog, a damaged project or re-opening the open project used to stop the run for
    # nothing, because the old single _confirm_switch cancelled the jobs when it asked.

    def new_project_dialog(self, kind: str = "airfoil") -> None:
        dlg = NewStudyDialog(kind if isinstance(kind, str) and kind else "airfoil", self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._ask_switch("create a new study"):
            return
        kind, name, folder = dlg.values()
        mesh = dlg.mesh_path() if hasattr(dlg, "mesh_path") else ""
        try:
            self.new_study(kind, name, folder, stop_jobs=True, mesh_path=mesh or None)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not create the study", f"{type(exc).__name__}: {exc}")

    def open_project_dialog(self) -> None:
        start = app_settings.get().last_dir
        d = QFileDialog.getExistingDirectory(self, "Open an Aerolume project folder (*.aerolume)", start)
        if d:
            self.open_project_path(d)

    def _is_open_project(self, path: str | Path) -> bool:
        """Whether ``path`` (a project folder or its project.json) is the project already open."""
        proj = self.state.project
        if proj is None:
            return False
        p = Path(path).expanduser()
        if p.is_file():
            p = p.parent
        if not p.exists() and p.with_name(p.name + start_mod.PROJECT_SUFFIX).exists():
            p = p.with_name(p.name + start_mod.PROJECT_SUFFIX)
        here = start_mod.project_path(proj)
        try:
            return here is not None and p.resolve() == Path(here).resolve()
        except OSError:
            return False

    def open_project_path(self, path: str | Path, confirmed: bool = False) -> object | None:
        """Open a project folder: load it, then ask about running jobs, then switch.

        ``confirmed`` skips the question (the caller has asked). Re-opening the
        project that is already open only shows it.
        """
        if self._is_open_project(path):
            self._status(f"{start_mod.project_name(self.state.project)} is already open.", 4000)
            if self._current in ("", "start"):
                self.state.navigate("geometry")
            return self.state.project
        try:
            with _busy_cursor():
                project = start_mod.open_project(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open project", f"{path}\n\n{type(exc).__name__}: {exc}")
            return None
        if not confirmed and not self._ask_switch():
            return None
        app_settings.get().last_dir = Path(path).parent
        self._stop_jobs()
        with _busy_cursor():
            self.set_project(project, getattr(project, "load_warnings", None) or None, navigate_to="geometry")
        return project

    def open_example(self, path: str | Path) -> object | None:
        try:
            project = start_mod.open_example(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not open example", f"{path}\n\n{type(exc).__name__}: {exc}")
            return None
        if not self._ask_switch("open the example"):
            return None
        self._stop_jobs()
        has_results = any(s.status == "done" for s in project.simulations.values())
        with _busy_cursor():
            self.set_project(project, navigate_to="results" if has_results else "geometry")
        return project

    def save_project(self) -> None:
        if self.state.project is None:
            return
        try:
            self.state.save()
            self._status("Project saved", 3000)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def close_project(self) -> None:
        if self.state.project is None or not self._confirm_switch():
            return
        try:
            self.state.save()
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not save: {exc}", 8000)
        self.state.set_project(None)
        self.show_page("start")
        self._status("Project closed", 3000)

    # -- view / help ----------------------------------------------------------------
    def set_theme(self, name: str) -> None:
        app_settings.get().theme = name
        theme.apply(None, name)

    def toggle_theme(self) -> None:
        self.set_theme("light" if theme.current() == "dark" else "dark")

    def toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self.act_full.setChecked(self.isFullScreen())

    def open_preferences(self, exec_: bool = True, tab: str = "general") -> PreferencesDialog:
        """Show the preferences dialog at ``tab`` (``exec_=False`` returns it unshown, for tests)."""
        dlg = PreferencesDialog(self, tab=tab)
        if exec_ and dlg.exec() == QDialog.DialogCode.Accepted:
            changed = dlg.changed           # accept() applied it (and stayed open on an invalid account)
            if changed:
                self._status("Preferences saved", 4000)
            if "assistant" in changed:
                refresh = getattr(self.assistant, "refresh_provider", None)
                if callable(refresh):
                    refresh()
        return dlg

    def help_path(self, topic: str) -> Path | None:
        """Local file for a help topic ("guide", "validation", "licence", "assistant", "scripting",
        "mcp"), or None. Looks in the source tree and, in an installed build, next to the executable."""
        roots = [_repo_root()]
        with contextlib.suppress(Exception):            # app/platform.py: the frozen layout
            from . import platform as plat
            roots.insert(0, plat.app_root())
        names = {
            "guide": ["docs/user_guide.md", "README.md"],
            "validation": ["docs/validation.md", "docs/validation_airfoil.md"],
            "licence": ["LICENSE"],
            "assistant": ["docs/ai_assistant.md"],
            "scripting": ["docs/scripting.md"],
            "mcp": ["docs/mcp.md"],
        }.get(topic, [])
        candidates = [root / n for n in names for root in roots]
        for p in candidates:
            if p.exists() and p.stat().st_size > 0:
                return p
        return None

    def user_guide_path(self) -> Path | None:
        return self.help_path("guide")

    def show_help(self, topic: str = "guide") -> None:
        if topic == "shortcuts":
            self.show_shortcuts()
            return
        if topic == "tour":
            self.start_tour()
            return
        p = self.help_path(topic)
        if p is None and topic == "mcp":
            QMessageBox.information(
                self, "MCP guide not installed",
                "docs/mcp.md is not part of this copy of Aerolume. It explains how to add the aerolume-mcp "
                "server to Claude Desktop or Claude Code, so Claude can run experiments through the "
                "scripting API without an API key in the app.")
            return
        if p is None:
            url = project_url()
            if url:
                QDesktopServices.openUrl(QUrl(url))
            else:
                QMessageBox.information(
                    self, "Documentation not installed",
                    "This copy of Aerolume was installed without its documentation. The user guide "
                    "is docs/user_guide.md in the source distribution and on the project website.")
            return
        titles = {"guide": "User guide", "validation": "Validation", "licence": "Licence",
                  "assistant": "Experiment assistant", "scripting": "Scripting",
                  "mcp": "Driving Aerolume from Claude (MCP)"}
        DocViewer(p, f"Aerolume — {titles.get(topic, topic)}", self).exec()

    def open_user_guide(self) -> None:
        self.show_help("guide")

    def shortcuts_dialog(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard shortcuts")
        dlg.resize(440, 400)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(12, 12, 12, 10)
        tree = QTreeWidget()
        tree.setColumnCount(2)
        tree.setHeaderLabels(["Keys", "Action"])
        tree.setRootIsDecorated(False)
        for keys, what in SHORTCUTS:
            it = QTreeWidgetItem([keys, what])
            it.setFont(0, theme.mono_font())
            tree.addTopLevelItem(it)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(tree, 1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(dlg.reject)
        lay.addWidget(box)
        return dlg

    def show_shortcuts(self) -> None:
        self.shortcuts_dialog().exec()

    def show_about(self, exec_: bool = True) -> AboutDialog:
        dlg = AboutDialog(self)
        if exec_:
            dlg.exec()
        return dlg

    # -- assistant and scripts ---------------------------------------------------------
    def show_assistant(self, goal: str | None = None) -> bool:
        """Show the Assistant dock, with ``goal`` in its goal box (False without a project)."""
        if self.state.project is None:
            self._status("Open or create a project first: experiments are recorded in the project.", 6000)
            return False
        self.docks.show_panel("assistant")
        if goal and callable(getattr(self.assistant, "set_goal", None)):
            self.assistant.set_goal(goal)
        elif callable(getattr(self.assistant, "focus_input", None)):
            self.assistant.focus_input()
        return True

    def show_script(self) -> None:
        """Show the Script panel and put the cursor in its editor."""
        self.docks.show_panel("script")
        editor = getattr(self.script_console, "editor", None)
        if editor is not None:
            editor.setFocus()

    def _ask_script_file(self) -> str:
        """File dialog for Tools > Run script (tests patch this)."""
        start = app_settings.get().last_dir
        root = start_mod.project_path(self.state.project) if self.state.project is not None else None
        if root is not None and (Path(root) / "scripts").is_dir():
            start = str(Path(root) / "scripts")
        path, _ = QFileDialog.getOpenFileName(self, "Run a Python script", start,
                                              "Python scripts (*.py);;All files (*)")
        return path

    def run_script_file(self, path: str | Path | None = None) -> bool:
        """Tools > Run script: open a ``.py`` file in the Script panel and run it."""
        console = self.script_console
        if not callable(getattr(console, "open_script", None)):
            self._status("The Script panel is not available in this build (see Messages).", 6000)
            return False
        path = path or self._ask_script_file()
        if not path:
            return False
        self.docks.show_panel("script")
        if not console.open_script(path):
            return False
        return bool(console.run())

    def open_experiment(self, path: str | Path, show: bool = True) -> QDialog | None:
        """Show an experiment log (its experiment.md, figures included) in a document window."""
        p = Path(path)
        md = p / "experiment.md" if p.is_dir() else p
        if not md.is_file():
            self._status(f"No experiment log at {p}", 6000)
            return None
        title = "Experiment"
        with contextlib.suppress(Exception):
            from ..ai.experiment import elide, load_record
            goal = elide(load_record(md.parent).goal)
            if goal:
                title = f"Experiment — {goal}"
        dlg = DocViewer(md, title, self)
        dlg.setModal(False)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        if show:
            dlg.show()
        return dlg

    # -- accounts, tour, crash reports, forwarded launches ------------------------------------
    #: "module:function" that adds the Account menu (the accounts agent's file; optional)
    ACCOUNT_MENU_HOOK = "aerolume.app.widgets.account_menu:install"

    def _install_account_menu(self) -> None:
        """Let the accounts module add its menu (sign in, cloud projects); nothing when it is absent.

        Accounts are optional and off until a Supabase project is configured
        (DESIGN.md section 11), so a build without the module simply has no
        Account menu. A module that is present but broken is reported in
        Messages, never raised.
        """
        mod_name, _, fn_name = self.ACCOUNT_MENU_HOOK.partition(":")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name and mod_name.startswith(exc.name):
                return
            self._account_menu_failed(exc)
            return
        except Exception as exc:  # noqa: BLE001 - accounts never stop the window
            self._account_menu_failed(exc)
            return
        install = getattr(mod, fn_name, None)
        if not callable(install):
            return
        try:
            self.account_menu = install(self, self.state)
        except Exception as exc:  # noqa: BLE001
            self._account_menu_failed(exc)

    def _account_menu_failed(self, exc: BaseException) -> None:
        traceback.print_exc()
        QTimer.singleShot(0, self, lambda: self.state.log(
            f"The Account menu could not be loaded: {type(exc).__name__}: {exc}", "error", "Application"))

    #: settings key: the welcome tour was offered once (its first part ran or was skipped)
    TOUR_KEY = "tour/seen"
    #: settings key: False while the project part is owed (the Start part finished, no project
    #: opened since); True once it ran or was skipped. Absent for users of the one-part tour.
    PROJECT_TOUR_KEY = "tour/project_seen"

    def start_tour(self, *, first_run: bool = False, part: str | None = None):
        """Show the welcome tour over the real window (Help ▸ Welcome tour); returns it.

        The tour explains only what is on screen (``widgets/tour.py``): on the
        Start page its *start* part (New study, Examples, Learn), with a
        project open its *project* part (step bar, Project panel, Run, panels,
        Script and Assistant). Finishing the start part leaves the project part
        owed; it runs the first time a project page shows
        (:meth:`_maybe_continue_tour`). Skipping either part ends the tour for
        good, so a user who said no is not stopped again on their first
        project. ``first_run`` only records how the tour started.
        """
        from .widgets.tour import WelcomeTour
        if part not in ("start", "project"):
            part = "start" if self._current in ("", "start") or self.state.project is None else "project"
        old = getattr(self, "tour", None)
        if old is not None:
            with contextlib.suppress(RuntimeError):
                old.finished.disconnect()
            with contextlib.suppress(RuntimeError):
                old.finish()
        self._tour_continue_pending = False
        self.tour = WelcomeTour(self, part=part)
        self.tour.first_run = bool(first_run)
        self.tour.finished.connect(lambda skipped, p=part: self._tour_finished(p, bool(skipped)))
        self.tour.start()
        return self.tour

    def _tour_finished(self, part: str, skipped: bool) -> None:
        """Record what ran: a finished start part owes the project part; anything else settles it."""
        s = app_settings.get()
        s.set_value(self.TOUR_KEY, True)
        owed = part == "start" and not skipped
        s.set_value(self.PROJECT_TOUR_KEY, not owed)

    def project_tour_owed(self) -> bool:
        """True when the start part finished and the project part has not run yet."""
        v = app_settings.get().value(self.PROJECT_TOUR_KEY, None)
        return v is not None and str(v).strip().lower() in ("false", "0", "no", "off")

    def _maybe_continue_tour(self) -> None:
        """Run the owed project part once a project page is on screen (``show_page`` calls this).

        Deferred a moment so the page and its panels have laid out (the tour
        measures them), and checked again when the timer fires: the user may
        have gone back to Start, closed the project or opened a dialog.
        """
        if getattr(self, "_tour_continue_pending", False) or not self.project_tour_owed():
            return
        tour = getattr(self, "tour", None)
        with contextlib.suppress(RuntimeError):
            if tour is not None and tour.isVisible():
                return
        self._tour_continue_pending = True

        def go() -> None:
            self._tour_continue_pending = False
            if (self.state.project is None or self._current in ("", "start") or not self.isVisible()
                    or QApplication.activeModalWidget() is not None or not self.project_tour_owed()):
                return
            self.start_tour(part="project")

        QTimer.singleShot(700, self, go)

    def maybe_show_first_run_tour(self) -> bool:
        """Start the tour on the very first start only (main.py calls this once the window shows)."""
        v = app_settings.get().value(self.TOUR_KEY, False)
        seen = str(v).strip().lower() in ("true", "1", "yes")
        if seen:
            return False
        QTimer.singleShot(600, self, lambda: self.start_tour(first_run=True))
        return True

    def log_folder(self) -> Path:
        """Where crash reports and the application log go (app/platform.py), else the settings folder."""
        with contextlib.suppress(Exception):
            from . import platform as plat
            return plat.logs_dir()
        return Path(app_settings.get().path).parent

    def open_log_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_folder())))

    # -- update check ------------------------------------------------------------------
    def check_for_updates(self, *, show: bool = True, fetch=None) -> threading.Thread | None:
        """Help ▸ Check for updates: read the configured release feed in the background.

        Off until a feed is configured (Preferences ▸ Release feed; there is no
        published one yet), in which case it says so and returns None. The
        check is one GET of that address - nothing about the user or the
        project is sent - on a worker thread, so a slow network never freezes
        the window; :attr:`updateChecked` brings the answer back. ``fetch``
        replaces :func:`fetch_release_feed` (tests).
        """
        url = app_settings.get().update_url
        if not url:
            self.last_update_check = None
            if show:
                self._update_box = self._update_message(
                    "Update checks are off",
                    "Aerolume has no published release feed yet, so there is nothing to check against. "
                    "When one exists, enter its address in Preferences ▸ General ▸ Release feed.",
                    QMessageBox.Icon.Information)
            return None
        if not getattr(self, "_update_hooked", False):
            self.updateChecked.connect(self._update_checked)
            self._update_hooked = True
        self._update_show = show
        self._status("Checking for updates…", 4000)
        get = fetch or fetch_release_feed

        def work() -> None:
            try:
                version, page = get(url)
                result = (version, page, "")
            except Exception as exc:  # noqa: BLE001 - any failure is reported, never raised
                result = ("", "", f"{type(exc).__name__}: {exc}")
            with contextlib.suppress(RuntimeError):     # the window closed meanwhile
                self.updateChecked.emit(result)

        t = threading.Thread(target=work, name="aerolume-update-check", daemon=True)
        t.start()
        return t

    def _update_checked(self, result) -> None:
        version, page, error = result
        self.last_update_check = result
        if not self._update_show:
            return
        if error:
            self._update_box = self._update_message(
                "Could not check for updates", f"The release feed could not be read.\n\n{error}",
                QMessageBox.Icon.Warning)
        elif version_key(version) > version_key(__version__):
            box = self._update_message(
                "Update available", f"Aerolume {version.lstrip('vV')} is available; this is {__version__}.",
                QMessageBox.Icon.Information, page)
            self._update_box = box
        else:
            self._update_box = self._update_message(
                "Aerolume is up to date", f"This is Aerolume {__version__}; the feed's newest release is "
                f"{version.lstrip('vV')}.", QMessageBox.Icon.Information)

    def _update_message(self, title: str, text: str, icon, page: str = "") -> QMessageBox:
        """A window-modal message (``open()``, so tests and the event loop are never blocked)."""
        box = QMessageBox(icon, title, text, QMessageBox.StandardButton.Close, self)
        if page and QUrl(page).scheme().lower() in ("https", "http"):    # never a file or app link
            b = box.addButton("Open release page", QMessageBox.ButtonRole.AcceptRole)
            b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(page)))
        box.open()
        return box

    def show_crash(self, report, *, exec_: bool = False) -> CrashDialog:
        """Show an uncaught error as a copyable report (one dialog collects every report of a burst)."""
        dlg = getattr(self, "_crash_dialog", None)
        if dlg is not None and dlg.isVisible():
            dlg.add_report(report)
            return dlg
        dlg = CrashDialog(report, self, log_folder=self.log_folder())
        self._crash_dialog = dlg
        with contextlib.suppress(Exception):
            self.state.log(f"Unexpected error: {getattr(report, 'summary', report)} (report saved)",
                           "error", "Application")
        if exec_:
            dlg.exec()
        else:
            dlg.setModal(False)
            dlg.show()
        return dlg

    def handle_forwarded_argv(self, argv: list) -> object | None:
        """A second launch handed over its arguments (app/platform.SingleInstance): open the project here.

        Called on the GUI thread of this, the first, instance; the window comes
        to the front either way, so the user sees where their double-click went.
        """
        project = None
        target = next((a for a in argv if a and not str(a).startswith("-")), None)
        if target:
            path = Path(target)
            with contextlib.suppress(Exception):
                from . import platform as plat
                path = plat.resolve_open_target(target) or path
            project = self.open_project_path(path)
        try:
            from . import platform as plat
            plat.activate_window(self)
        except Exception:  # noqa: BLE001 - bringing the window up is best effort
            self.showNormal()
            self.raise_()
            self.activateWindow()
        return project

    # -- window geometry / lifetime ----------------------------------------------------
    def _restore_geometry(self) -> None:
        """The window's size and place from the last session (the panels come from the workspace).

        v2 also restored ``QMainWindow.saveState()``; v3 does not, because the
        dock manager's workspace is the one description of the panels and an
        opaque Qt state restored over it would put back docks the workspace hid.
        """
        s = app_settings.get()
        g = s.window_geometry
        if g is None or not self.restoreGeometry(g):
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                w, h = min(1600, int(avail.width() * 0.9)), min(1000, int(avail.height() * 0.9))
                self.resize(w, h)
                self.move(avail.center().x() - w // 2, avail.center().y() - h // 2)
            else:
                self.resize(1600, 1000)
        self._dock_width_pending = True

    #: the Project panel's width on a first start, and the narrowest width a restored layout keeps
    PROJECT_DOCK_W = 230
    PROJECT_DOCK_MIN_KEPT = 200

    def showEvent(self, e) -> None:
        """Size the rails once the window is laid out (``resizeDocks`` before the first show is ignored).

        A first start left the tree at about 185 px, where "Wing polar · 25 m/s"
        and its copy both read "Wing polar · 2…". The workspace's rail widths
        apply, and the Project panel never stays narrower than
        :attr:`PROJECT_DOCK_MIN_KEPT`.
        """
        super().showEvent(e)
        QTimer.singleShot(0, self, self._apply_project_dock_width)

    def _project_dock_visibility(self, visible: bool) -> None:
        # the welcome page hides the panels, so the rails are sized when the tree first appears
        if visible:
            QTimer.singleShot(0, self, self._apply_project_dock_width)

    def _apply_project_dock_width(self) -> None:
        dock = self.project_dock
        if not getattr(self, "_dock_width_pending", False) or not self.isVisible() or not dock.isVisible():
            return
        self._dock_width_pending = False
        self.docks.apply_sizes_now()
        QTimer.singleShot(0, self, self._keep_project_dock_readable)

    def _keep_project_dock_readable(self) -> None:
        dock = self.project_dock
        if dock.isVisible() and not dock.isFloating() and dock.width() < self.PROJECT_DOCK_MIN_KEPT:
            self.resizeDocks([dock], [self.PROJECT_DOCK_W], Qt.Orientation.Horizontal)

    def save_window_state(self) -> None:
        """Window geometry, the Messages heights and the current workspace, written now."""
        s = app_settings.get()
        self.docks.save()
        self._save_messages_heights()
        s.window_geometry = self.saveGeometry()
        s.sync()

    # -- drag and drop --------------------------------------------------------------
    @staticmethod
    def _dropped_geometry(e) -> list[str]:
        """Local files in a drag that an importer reads: 2D outlines and 3D meshes (by extension)."""
        md = e.mimeData()
        if not md.hasUrls():
            return []
        exts = {x.lower() for x in start_mod.MESH_EXTENSIONS}
        with contextlib.suppress(Exception):     # without the 2D importer, meshes still import
            from ..geometry.importer import SUPPORTED_EXTENSIONS
            exts |= {x.lower() for x in SUPPORTED_EXTENSIONS}
        return [u.toLocalFile() for u in md.urls()
                if u.isLocalFile() and Path(u.toLocalFile()).suffix.lower() in exts]

    def dragEnterEvent(self, e) -> None:
        """Accept geometry files anywhere in the window (the Geometry view has its own drop target).

        The user guide promised "drop a file on the window", but only the
        Geometry viewport accepted drops: a .dat dropped on Setup, Results or
        the Project tree did nothing.
        """
        if self.state.project is not None and self._dropped_geometry(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e) -> None:
        if self.state.project is not None and self._dropped_geometry(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e) -> None:
        paths = self._dropped_geometry(e)
        if not paths or self.state.project is None:
            e.ignore()
            return
        e.acceptProposedAction()
        for p in paths:
            if Path(p).suffix.lower() in start_mod.MESH_EXTENSIONS:
                self.state.request(f"import-mesh:{p}")
            else:
                self.state.navigate(f"geometry:import:{p}")

    def request_exit(self) -> None:
        """File ▸ Exit (Ctrl+Q): the same questions as the window's close button, then close."""
        if self.confirm_quit():
            self._quit_confirmed = True
            try:
                self.close()
            finally:
                self._quit_confirmed = False

    def confirm_quit(self) -> bool:
        """Ask before quitting would stop simulations or lose an unsaved result; True to go on."""
        running, queued = self._busy_jobs()
        if running is not None or queued:
            ans = QMessageBox.question(self, "Simulation running",
                                       f"{self._busy_text(running, queued)} Stop it and quit?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return False
        assistant_busy = getattr(getattr(self, "assistant", None), "is_running", None)
        if running is None and not queued and callable(assistant_busy) and assistant_busy():
            ans = QMessageBox.question(self, "Experiment running",
                                       "The assistant is working on an experiment. Stop it and quit?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return False
        unsaved = []
        with contextlib.suppress(Exception):
            unsaved = list(self.jobs.unsaved_ids()) if self.jobs is not None else []
        if unsaved:
            n = len(unsaved)
            ans = QMessageBox.question(
                self, "Results not saved",
                f"{n} finished result{'s' if n != 1 else ''} could not be saved to the project folder yet. "
                "One more attempt is made when Aerolume quits; if it fails, the result is lost. Quit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return False
        return True

    def closeEvent(self, e: QCloseEvent) -> None:
        """Ask (user closes only), save, then stop pages and jobs, in an order where No changes nothing.

        The questions run whenever someone could answer them: for a close from
        the window system (title bar, Alt+F4) and for any close of a visible
        window; File ▸ Exit (:meth:`request_exit`) asks first and sets
        ``_quit_confirmed``. A hidden window closed from code (scripts, a test
        harness) closes without asking. The project is saved *before* anything
        is stopped: if that fails and the user keeps the window open, the running
        simulation and the queue are untouched (they used to be gone already).
        """
        if (not self._quit_confirmed and (e.spontaneous() or self.isVisible())
                and not self.confirm_quit()):
            e.ignore()
            return
        try:
            self.state.save()
        except Exception as exc:  # noqa: BLE001
            ans = QMessageBox.warning(self, "Save failed",
                                      f"The project could not be saved:\n{exc}\n\nClose anyway?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
        tool_windows = [getattr(self, "assistant", None), getattr(self, "script_console", None)]
        for w in [*self._pages.values(), *(t for t in tool_windows if t is not None)]:
            shutdown = getattr(w, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()          # e.g. Simulate stops its live explore worker
                except Exception:  # noqa: BLE001 - closing must always succeed
                    traceback.print_exc()
        if self.jobs is not None:
            shutdown = getattr(self.jobs, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001 - closing must always succeed
                    traceback.print_exc()
        with contextlib.suppress(Exception):    # the statuses the shutdown recorded; reported above if it fails
            self.state.save()
        self.save_window_state()
        super().closeEvent(e)


# imported late: the dialog lives with the start page that also opens it
from .pages.start import NewStudyDialog

__all__ = [
    "MESH_IMPORT_HOOK",
    "MESH_IMPORT_HOOK_ALTERNATIVES",
    "PAGES",
    "PAGE_KEYS",
    "SHORTCUTS",
    "STEP_KEYS",
    "AboutDialog",
    "DocViewer",
    "DockTitle",
    "MainWindow",
    "PageSpec",
    "PlaceholderPage",
    "PreferencesDialog",
    "RunStatus",
    "breadcrumb_text",
    "device_label",
    "diagnostics_log_for",
    "end_of_run_diagnostics",
    "format_cells",
    "mesh_import_hook",
]
