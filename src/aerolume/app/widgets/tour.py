"""The welcome tour: short steps over the real window, skippable, re-openable from Help.

A tour that shows screenshots of the program teaches less than one that
points at the program itself, so each step dims the window except for the
widget it is about (a cut-out in a ``scrim`` layer, outlined in the focus
colour) and puts a short box beside it: what the thing is, what it does, the
key that does it.

The tour comes in two parts because the window has two states. Before a
project is open only the Start page is on screen, so a tour that explained
the step bar, the Project panel and Run from there pointed at nothing: three
of its six boxes sat centred over a dimmed page, saying "appears once a
project is open". Now each part explains only what is on screen:

* :data:`START_TOUR_STEPS` - on the Start page: New study, Examples, Learn.
* :data:`PROJECT_TOUR_STEPS` - once a project is open: the step bar, the
  Project panel, Run and Stop, the panels, scripts and the assistant.

The window picks the part for the page that is showing, and after the Start
part it runs the project part the first time a project opens (see
``MainWindow.start_tour``). A step whose widget is still not on screen (a
panel the user hid) says how to show it and centres the box.

A box never covers its target's neighbours: when the target sits in a
toolbar the box goes *below the whole toolbar band*. Placed merely beside a
toolbar button it used to cover the next buttons - the step about Run and
Stop sat over Stop and Report. A step about two controls (Run and Stop,
Assistant and Script) outlines both.

Keys: Right / Enter next, Left back, Esc skips. Clicks outside the box do
nothing, so a stray click cannot act on the dimmed window behind it.
``finished(skipped)`` fires once at the end.

The copy follows docs/ui_guidelines.md section 6: nouns for titles, plain
verbs, the shortcut in the text, no cheerleading.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolBar, QVBoxLayout, QWidget

from .. import theme
from .common import SPACING, button, label


@dataclass(frozen=True)
class TourStep:
    """One stop: a key, a title, the text and how to find the widget it is about."""

    key: str
    title: str
    text: str
    #: the widget (or widgets: their union is outlined) the step is about
    target: Callable[[QWidget], QWidget | Sequence[QWidget | None] | None]
    #: said when the widget is not on screen (the box is then centred)
    absent: str = ""


def _toolbar_button(win: QWidget, action_name: str) -> QWidget | None:
    action = getattr(win, action_name, None)
    tb = getattr(win, "main_toolbar", None)
    if action is None or not isinstance(tb, QToolBar):
        return None
    return tb.widgetForAction(action)


def _panel_button(win: QWidget, key: str) -> QWidget | None:
    """The toolbar toggle of a dock panel ("assistant", "script": the dock manager's actions)."""
    docks = getattr(win, "docks", None)
    tb = getattr(win, "main_toolbar", None)
    a = docks.actions.get(key) if docks is not None else None
    return tb.widgetForAction(a) if a is not None and isinstance(tb, QToolBar) else None


def _assistant_button(win: QWidget) -> QWidget | None:
    """The toolbar's Assistant toggle (the dock manager's action for the Assistant panel)."""
    return _panel_button(win, "assistant")


def _start_page(win: QWidget) -> QWidget | None:
    return getattr(win, "_pages", {}).get("start")


def _start_widgets(win: QWidget, *names: str) -> list[QWidget | None]:
    """Named widgets of the Start page (missing names are skipped by the tour)."""
    page = _start_page(win)
    return [getattr(page, n, None) for n in names] if page is not None else []


def _start_actions(win: QWidget) -> QWidget | None:
    page = _start_page(win)
    return getattr(page, "actions_widget", None) if page is not None else None


def _start_examples(win: QWidget) -> list[QWidget | None]:
    """The Examples heading, list and note: outlined together as one group."""
    return _start_widgets(win, "examples_header", "examples", "examples_note")


def _start_learn(win: QWidget) -> list[QWidget | None]:
    """The Learn heading and its guide links."""
    page = _start_page(win)
    if page is None:
        return []
    links = list(getattr(page, "help_links", {}).values())
    return [getattr(page, "learn_header", None), *links]


#: the Start page part: only what is on screen before a project is open
START_TOUR_STEPS: tuple[TourStep, ...] = (
    TourStep("start", "Start a study",
             "Choose what to study: an airfoil, a vehicle side profile, a 3D mesh, a wing, a heated room or "
             "an electronics enclosure. Each starts with a design and a first simulation set up for it. "
             "Ctrl+N also offers an empty project.",
             _start_actions, "Start a study from File ▸ New study (Ctrl+N)."),
    TourStep("examples", "Open an example",
             "Each example is a finished study with its results, so every page has something to show. "
             "It opens as a copy in Documents/Aerolume/Examples; the original is never changed. Projects "
             "you open or create are listed above it.",
             _start_examples, "Open an example from File ▸ Open example."),
    TourStep("learn", "Guides",
             "The scripting guide, the experiment assistant, driving Aerolume from Claude and the keyboard "
             "shortcuts. The user guide is Documentation below, or F1 from anywhere. The rest of this tour "
             "(the steps, the panels and Run) follows when you open your first project.",
             _start_learn, "The guides are in the Help menu; F1 opens the user guide."),
)

#: the project part: the window once a project is open
PROJECT_TOUR_STEPS: tuple[TourStep, ...] = (
    TourStep("steps", "Work through the steps",
             "Geometry, Setup, Simulate, Results, Compare and Report. A check marks a step that has data; "
             "Ctrl+1 to Ctrl+6 jump between them.",
             lambda w: getattr(w, "step_bar", None), "The step bar shows on every page of an open project."),
    TourStep("project", "Pick what the pages show",
             "The Project panel lists each design with its simulations. Selecting one here changes every "
             "page. Right-click an entry to run, duplicate or rename it.",
             lambda w: getattr(w, "project_dock", None),
             "The Project panel is hidden in this workspace; View ▸ Panels ▸ Project shows it."),
    TourStep("run", "Run a simulation",
             "Run (F5) starts the selected simulation. Stop (Shift+F5) ends it and keeps the partial result, "
             "or the earlier complete one when it was a re-run. The status bar shows progress, and for a "
             "CFD run the grid size and steps per second.",
             lambda w: (_toolbar_button(w, "act_run"), _toolbar_button(w, "act_stop"))),
    TourStep("panels", "Arrange the panels",
             "Collapse a panel from its header, move it to another rail or hide it from its ⋯ menu, and bring "
             "it back from View ▸ Panels. Each workspace (Default, Analysis, Live, Scripting) keeps its "
             "own arrangement; switch between them here.",
             lambda w: _toolbar_button(w, "act_workspace")),
    TourStep("help", "Scripts, the assistant and help",
             "Script runs Python against the same API the pages use. The Assistant plans and runs "
             "experiments; runs above the approval threshold, code and deletions wait for your click. "
             "F1 opens the user guide; Help ▸ Welcome tour shows this tour again.",
             lambda w: (_panel_button(w, "assistant"), _panel_button(w, "script"))),
)

#: both parts in order (the full tour's content; the window runs one part at a time)
TOUR_STEPS: tuple[TourStep, ...] = START_TOUR_STEPS + PROJECT_TOUR_STEPS


class _Callout(QFrame):
    """The step box: counter, title, text, then Skip / Back / Next."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setProperty("callout", "true")
        self.setFixedWidth(360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4 * SPACING, 3 * SPACING, 4 * SPACING, 3 * SPACING)
        lay.setSpacing(2 * SPACING)
        self.counter = label("", "muted")
        lay.addWidget(self.counter)
        self.title = label("", "h2", wrap=True)
        lay.addWidget(self.title)
        self.text = label("", None, wrap=True)
        lay.addWidget(self.text)
        self.absent = label("", "muted", wrap=True)
        lay.addWidget(self.absent)
        row = QHBoxLayout()
        row.setSpacing(2 * SPACING)
        self.skip = button("Skip tour", "ghost")
        row.addWidget(self.skip)
        row.addStretch(1)
        self.back = button("Back")
        row.addWidget(self.back)
        self.next = button("Next", "primary")
        row.addWidget(self.next)
        lay.addSpacing(SPACING)
        lay.addLayout(row)


class WelcomeTour(QWidget):
    """The tour overlay, a child of the main window that covers it while it runs."""

    finished = Signal(bool)          # skipped
    stepChanged = Signal(int)

    MARGIN = 6                       # space between the highlighted widget and its outline

    #: the step lists of each part (``part`` picks one)
    PARTS: ClassVar[dict[str, tuple[TourStep, ...]]] = {"start": START_TOUR_STEPS, "project": PROJECT_TOUR_STEPS}

    def __init__(self, window: QWidget, steps: tuple[TourStep, ...] | None = None, *,
                 part: str = "project") -> None:
        super().__init__(window)
        self.window_ = window
        #: "start" (the Start page part) or "project"; explicit ``steps`` override the part's list
        self.part = part if part in self.PARTS else "project"
        self.steps = tuple(steps) if steps is not None else self.PARTS[self.part]
        self._index = -1
        self._done = False
        self._target_rect = QRect()
        self._band = QRect()                 # the toolbar the target sits in: the box keeps clear of it
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Welcome tour" if self.part == "start" else "Welcome tour: the project window")
        self.box = _Callout(self)
        self.box.skip.clicked.connect(self.skip)
        self.box.back.clicked.connect(self.back)
        self.box.next.clicked.connect(self.next)
        window.installEventFilter(self)
        theme.changed.connect(self._on_theme)     # bound method: auto-disconnects when deleted
        self.hide()

    # -- control --------------------------------------------------------------------
    def start(self) -> None:
        """Show the tour from the first step."""
        self._done = False
        self.setGeometry(self.window_.rect())
        self.show()
        self.raise_()
        self.go(0)

    def current_index(self) -> int:
        return self._index

    def current_step(self) -> TourStep | None:
        return self.steps[self._index] if 0 <= self._index < len(self.steps) else None

    def go(self, index: int) -> None:
        self._index = max(0, min(int(index), len(self.steps) - 1))
        self._layout_step()
        self.stepChanged.emit(self._index)
        self.box.next.setFocus()

    def next(self) -> None:
        if self._index >= len(self.steps) - 1:
            self.finish()
        else:
            self.go(self._index + 1)

    def back(self) -> None:
        if self._index > 0:
            self.go(self._index - 1)

    def skip(self) -> None:
        self._end(skipped=True)

    def finish(self) -> None:
        self._end(skipped=False)

    def _end(self, skipped: bool) -> None:
        if self._done:
            return
        self._done = True
        self.hide()
        self.window_.removeEventFilter(self)
        self.finished.emit(bool(skipped))
        self.deleteLater()

    # -- layout ---------------------------------------------------------------------
    def target_widgets(self) -> list[QWidget]:
        """The widgets the current step points at that are on screen (first = the main one)."""
        step = self.current_step()
        if step is None:
            return []
        try:
            got = step.target(self.window_)
        except Exception:  # noqa: BLE001 - a missing widget only means "no highlight"
            return []
        items = list(got) if isinstance(got, (list, tuple)) else [got]
        return [w for w in items if isinstance(w, QWidget) and w.isVisible() and w.width() > 0 and w.height() > 0]

    def target_widget(self) -> QWidget | None:
        """The (main) widget the current step points at, when it is on screen."""
        ws = self.target_widgets()
        return ws[0] if ws else None

    def _rect_in_tour(self, w: QWidget) -> QRect:
        return QRect(w.mapTo(self.window_, QPoint(0, 0)), w.size())

    def _band_of(self, w: QWidget) -> QRect:
        """The toolbar a target sits in (tour coordinates), or a null rect when it is not in one."""
        p = w.parentWidget()
        while p is not None and p is not self.window_:
            if isinstance(p, QToolBar):
                return self._rect_in_tour(p)
            p = p.parentWidget()
        return QRect()

    def target_rect(self) -> QRect:
        """The highlighted rectangle in tour coordinates (empty when the step has no widget on screen)."""
        return QRect(self._target_rect)

    def _layout_step(self) -> None:
        step = self.current_step()
        if step is None:
            return
        n = len(self.steps)
        b = self.box
        b.counter.setText(f"Step {self._index + 1} of {n}")
        b.title.setText(step.title)
        b.text.setText(step.text)
        b.back.setEnabled(self._index > 0)
        b.next.setText("Finish" if self._index == n - 1 else "Next")
        ws = self.target_widgets()
        self._band = QRect()
        if not ws:
            self._target_rect = QRect()
            b.absent.setText(step.absent)
            b.absent.setVisible(bool(step.absent))
        else:
            rect = QRect()
            for w in ws:
                rect = rect.united(self._rect_in_tour(w))
                self._band = self._band.united(self._band_of(w))
            m = self.MARGIN
            self._target_rect = rect.adjusted(-m, -m, m, m).intersected(self.rect())
            b.absent.setVisible(False)
        b.adjustSize()
        self._place_box()
        self.update()

    def box_rect(self) -> QRect:
        """Where the step box is (tour coordinates)."""
        return self.box.geometry()

    def _place_box(self) -> None:
        """Beside the target without covering it or its toolbar; else centred.

        A target in a toolbar: below the toolbar band, starting at the target's
        left edge (slid to stay inside the window). Anything else: right of it
        when there is room, else left, below, above.
        """
        b, r, area = self.box, self._target_rect, self.rect().adjusted(12, 12, -12, -12)
        bw, bh = b.width(), b.sizeHint().height()
        b.resize(bw, bh)
        if r.isNull():
            b.move(area.center().x() - bw // 2, area.center().y() - bh // 2)
            return
        gap = 12
        band = self._band
        avoid = r.united(band) if not band.isNull() else r
        if not band.isNull():
            candidates = (QPoint(r.left(), avoid.bottom() + gap),           # below the toolbar
                          QPoint(r.left(), avoid.top() - gap - bh))         # above it (a bottom toolbar)
        else:
            candidates = (
                QPoint(r.right() + gap, r.top()),                    # right
                QPoint(r.left() - gap - bw, r.top()),                # left
                QPoint(r.left(), r.bottom() + gap),                  # below
                QPoint(r.left(), r.top() - gap - bh),                # above
            )
        for pt in candidates:
            box = QRect(pt, b.size())
            # slide along the edge to stay inside the window
            dx = max(area.left() - box.left(), 0) - max(box.right() - area.right(), 0)
            dy = max(area.top() - box.top(), 0) - max(box.bottom() - area.bottom(), 0)
            moved = box.translated(dx, dy)
            if area.contains(moved) and not moved.intersects(avoid):
                b.move(moved.topLeft())
                return
        b.move(area.center().x() - bw // 2, area.center().y() - bh // 2)

    # -- events ---------------------------------------------------------------------
    def eventFilter(self, obj, e) -> bool:
        if obj is self.window_ and e.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest) and not self._done:
            self.setGeometry(self.window_.rect())
            QTimer.singleShot(0, self, self._layout_step)
        return super().eventFilter(obj, e)

    def _on_theme(self, _name: str = "") -> None:
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.skip()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_PageDown):
            self.next()
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_PageUp):
            self.back()
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not self.box.back.hasFocus() \
                and not self.box.skip.hasFocus():
            self.next()
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e) -> None:
        e.accept()                   # the dimmed window behind is not clickable during the tour

    def mouseReleaseEvent(self, e) -> None:
        e.accept()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scrim = QColor(theme.tokens()["scrim"])
        scrim.setAlphaF(theme.SCRIM_ALPHA)
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        r = self._target_rect
        if not r.isNull():
            hole = QPainterPath()
            hole.addRoundedRect(QRectF(r), 4, 4)
            path = path.subtracted(hole)
        p.fillPath(path, scrim)
        if not r.isNull():
            pen = QPen(theme.qcolor("focus"), 2.0)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(r).adjusted(1, 1, -1, -1), 4, 4)
        p.end()


__all__ = ["PROJECT_TOUR_STEPS", "START_TOUR_STEPS", "TOUR_STEPS", "TourStep", "WelcomeTour"]
