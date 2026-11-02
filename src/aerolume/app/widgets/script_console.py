"""The Script console: write, run and keep Python scripts against the open project (DESIGN.md 8).

An editor (Python highlighting, line numbers, auto-indent) on the left, the
output on the right, one tool strip above both -- the shape of Spyder's or Qt
Creator's script panes, sized for a bottom dock. Scripts run on a worker
thread through :class:`aerolume.app.scripting.ScriptEngine` with these names
bound: ``project`` (an :class:`aerolume.api.Project` on the open project, or
None), ``al`` (the ``aerolume`` package), ``show(obj)`` (figures appear inline
in the output; a click opens one at full size, to save or copy) and
``log(text, level="info")`` (a line in the output and in the Messages dock). Designs, simulations and results a script makes appear in the
Project tree as they are made; its simulations run through the app's job
queue, so the Simulate page shows them live. Variables persist between runs
until "Reset variables".

Saved scripts live in ``<project>/scripts/*.py`` and are listed in the script
box; a script with a file is saved before each run. The Examples menu holds
ready scripts (:data:`aerolume.app.scripting.SNIPPETS`): polars at several
Reynolds numbers, an angle sweep with a plot, two designs compared, a mesh
imported and run in 3D with streamlines and smoke, a wing taper study.

Keyboard (while the console has focus): Ctrl+Enter or F5 runs the script,
Ctrl+Shift+Enter runs the selection (or the current line), Shift+F5 stops,
Ctrl+S saves the script, Ctrl+/ comments lines, Tab / Shift+Tab indent.
Leaving the editor by keyboard (WCAG 2.1.2): Ctrl+Tab / Ctrl+Shift+Tab move
focus to the next / previous control, and Esc followed by Tab (or Shift+Tab)
does the same; More > "Tab key moves focus" makes plain Tab move focus too.
These win over the window's own F5 / Shift+F5 / Ctrl+S only while the
console has focus.

Embedding (the main window docks it)::

    console = ScriptConsole(state, jobs)          # parent optional
    dock = window._dock("Script", "scriptDock", console)

``ScriptConsole(state, jobs=None, parent=None)``: ``state`` the AppState,
``jobs`` the JobManager (None runs simulations on the script's own thread).
Signals: ``runStarted()``, ``runFinished(object)`` (a
:class:`~aerolume.app.scripting.ScriptOutcome`), ``runningChanged(bool)``,
``scriptSaved(str)`` (the path). Public methods: ``run()``,
``run_selection()``, ``stop()``, ``is_running()``, ``text()`` /
``set_text(text)``, ``new_script()``, ``open_script(path)``,
``save_script(name=None)``, ``insert_snippet(key)``, ``clear_output()``,
``reset_variables()``, ``shutdown()`` (call when the window closes),
``dock_share()``; attributes ``engine``, ``editor``, ``output``, ``current_path``.

While a script runs, the line beside the status shows the progress of the
simulation it started, prefixed "Simulation:", and is cleared when the script
ends. Docked in a main window, the console keeps the dock's share of the
window height across window resizes (see ``ScriptConsole._watch_window``).
"""
from __future__ import annotations

import keyword
import re
import textwrap
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QFont,
    QImage,
    QKeyEvent,
    QKeySequence,
    QPainter,
    QPixmap,
    QSyntaxHighlighter,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextImageFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import scripting, theme
from .common import SPACING, ElidedLabel, IconButton, ToolStrip, label

__all__ = ["CodeEditor", "FigureViewer", "OutputView", "PythonHighlighter", "ScriptConsole"]

UNTITLED = "Untitled"
#: what a new script starts with: the names a console script has, in one line
NEW_SCRIPT = "# project: the open project · al: the aerolume package · show(figure) · log(text)\n\n"
#: largest width an inline figure is drawn at in the output [px]
FIGURE_MAX_W = 640
#: an inline figure is shrunk to fit the pane's height only down to this width [px]: below it
#: axis labels and titles become unreadable, so a short dock scrolls instead
FIGURE_MIN_W = 400
#: vertical gap around each inline figure [px] (two figures in a row must not touch)
FIGURE_GAP = 8
MAX_SEGMENTS = 4000


def _use_mono(w: QWidget) -> None:
    """The code font for an editor or output pane.

    The global style sheet sets the UI family on every QWidget, which wins over
    ``setFont``; a widget-level rule with the font family alone (no colours, so
    the theme still owns every colour) is what makes the code monospace.
    """
    w.setStyleSheet(f'font-family: "{theme.mono_family()}"; font-size: {theme.FONT_PT["body"]:g}pt;')
    w.setFont(theme.mono_font())


# ---------------------------------------------------------------------------
# syntax highlighting
# ---------------------------------------------------------------------------

_CONSTANTS = ("True", "False", "None")
_KEYWORDS = frozenset(k for k in keyword.kwlist + keyword.softkwlist if k not in _CONSTANTS and k != "_")
_BUILTINS = frozenset(("abs", "all", "any", "dict", "enumerate", "filter", "float", "format", "getattr",
                       "hasattr", "int", "isinstance", "len", "list", "map", "max", "min", "next", "open",
                       "print", "range", "repr", "round", "set", "sorted", "str", "sum", "tuple", "type",
                       "zip", "Exception", "ValueError", "KeyError", "RuntimeError"))
_BOUND = ("al", "project", "show", "log", "args", "self", "cls")


class PythonHighlighter(QSyntaxHighlighter):
    """Python colouring from the theme tokens: keywords in the accent, strings and numbers in the
    two muted status hues, comments muted italic, names being defined in semibold.

    Triple-quoted strings continue across lines (block states 1 = ''' and 2 = \"\"\").
    The palette is restrained on purpose: an editor inside an engineering tool, not a theme.
    """

    _WORD = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    _NUMBER = re.compile(r"\b(?:0[xXoObB][0-9a-fA-F_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?j?|\.\d+(?:[eE][+-]?\d+)?)\b")
    _DEF = re.compile(r"\b(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")
    _DECO = re.compile(r"^\s*@[\w.]+")

    def __init__(self, document: QTextDocument) -> None:
        super().__init__(document)
        self._formats: dict[str, QTextCharFormat] = {}
        self._build()
        theme.changed.connect(self._retheme)

    def _fmt(self, token: str, *, bold: bool = False, italic: bool = False) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setForeground(theme.qcolor(token))
        if bold:
            f.setFontWeight(QFont.Weight.DemiBold)
        f.setFontItalic(italic)
        return f

    def _build(self) -> None:
        self._formats = {
            "keyword": self._fmt("accent"),
            "constant": self._fmt("accent"),
            "builtin": self._fmt("accent2"),
            "bound": self._fmt("text2", italic=True),
            "string": self._fmt("good"),
            "number": self._fmt("warn"),
            "comment": self._fmt("muted", italic=True),
            "define": self._fmt("text", bold=True),
            "decorator": self._fmt("text2"),
        }

    def _retheme(self, _name: str = "") -> None:
        self._build()
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        f = self._formats
        n = len(text)
        start = 0
        state = self.previousBlockState()
        self.setCurrentBlockState(0)
        if state in (1, 2):
            delim = "'''" if state == 1 else '"""'
            end = text.find(delim)
            if end < 0:
                self.setFormat(0, n, f["string"])
                self.setCurrentBlockState(state)
                return
            self.setFormat(0, end + 3, f["string"])
            start = end + 3
        for m in self._WORD.finditer(text, start):
            w = m.group(0)
            kind = ("keyword" if w in _KEYWORDS else "constant" if w in _CONSTANTS else
                    "builtin" if w in _BUILTINS else "bound" if w in _BOUND else None)
            if kind:
                self.setFormat(m.start(), len(w), f[kind])
        for m in self._NUMBER.finditer(text, start):
            self.setFormat(m.start(), m.end() - m.start(), f["number"])
        for m in self._DEF.finditer(text, start):
            self.setFormat(m.start(2), m.end(2) - m.start(2), f["define"])
        m = self._DECO.match(text)
        if m and m.start() >= start:
            self.setFormat(m.start(), m.end() - m.start(), f["decorator"])
        # strings and comments last: they win over everything found inside them
        i = start
        while i < n:
            c = text[i]
            if c == "#":
                self.setFormat(i, n - i, f["comment"])
                return
            if c in "'\"":
                if text.startswith(c * 3, i):
                    end = text.find(c * 3, i + 3)
                    if end < 0:
                        self.setFormat(i, n - i, f["string"])
                        self.setCurrentBlockState(1 if c == "'" else 2)
                        return
                    self.setFormat(i, end + 3 - i, f["string"])
                    i = end + 3
                    continue
                j = i + 1
                while j < n and text[j] != c:
                    j += 2 if text[j] == "\\" else 1
                self.setFormat(i, min(j + 1, n) - i, f["string"])
                i = j + 1
                continue
            i += 1


# ---------------------------------------------------------------------------
# the editor
# ---------------------------------------------------------------------------

class _Gutter(QWidget):
    """Line numbers, painted by the editor."""

    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self.editor.gutter_width(), 0)

    def paintEvent(self, e) -> None:
        self.editor.paint_gutter(e)


class CodeEditor(QPlainTextEdit):
    """A plain-text Python editor: line numbers, current-line band, error line, auto-indent.

    ``set_error_line(n)`` marks a line (1-based; None clears) in the error colour;
    ``goto_line(n)`` moves the cursor there. Tab inserts four spaces (indents a
    selection), Shift+Tab dedents, Enter keeps the indentation (one more level
    after a colon), Backspace in leading blanks removes one level, Ctrl+/ toggles
    comments.

    Because Tab indents, the editor would otherwise be a keyboard trap (WCAG
    2.1.2): a keyboard-only user could enter it and never reach the Run button
    or another panel. Three ways out, all stated in the tooltip:

    * Ctrl+Tab / Ctrl+Shift+Tab always move focus to the next / previous control
      (QPlainTextEdit would otherwise swallow them).
    * Esc arms a one-shot escape: the next Tab / Shift+Tab moves focus instead of
      indenting; any other key disarms it. This is the convention of VS Code's
      "Tab moves focus" escape and of the WAI-ARIA authoring practices for
      editors that need the Tab key.
    * ``set_tab_moves_focus(True)`` (More > "Tab key moves focus") makes plain
      Tab and Shift+Tab move focus for as long as it is on.
    """

    INDENT = 4
    #: The tooltip states the escape route so it is discoverable without docs.
    TOOLTIP = ("Python script. Ctrl+Enter runs it. Tab indents; to leave the editor press "
               "Ctrl+Tab (Ctrl+Shift+Tab goes back), or Esc then Tab.")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _use_mono(self)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(" ") * self.INDENT)
        self.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.setProperty("flat", "true")
        self._gutter = _Gutter(self)
        self._error_line: int | None = None
        self._escape_armed = False          # Esc pressed: the next Tab leaves the editor
        self._tab_moves_focus = False
        self.setToolTip(self.TOOLTIP)
        self.setAccessibleDescription(self.TOOLTIP)
        self.highlighter = PythonHighlighter(self.document())
        self.blockCountChanged.connect(self._update_margins)
        self.updateRequest.connect(self._update_gutter)
        self.cursorPositionChanged.connect(self._update_selections)
        theme.changed.connect(self._retheme)
        self._update_margins()
        self._update_selections()

    # -- gutter ----------------------------------------------------------------------
    def gutter_width(self) -> int:
        digits = max(3, len(str(max(1, self.blockCount()))))
        return 12 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_margins(self, _n: int = 0) -> None:
        self.setViewportMargins(self.gutter_width(), 0, 0, 0)

    def _update_gutter(self, rect: QRect, dy: int) -> None:
        if dy:
            self._gutter.scroll(0, dy)
        else:
            self._gutter.update(0, rect.y(), self._gutter.width(), rect.height())

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        cr = self.contentsRect()
        self._gutter.setGeometry(QRect(cr.left(), cr.top(), self.gutter_width(), cr.height()))

    def paint_gutter(self, e) -> None:
        p = QPainter(self._gutter)
        p.fillRect(e.rect(), theme.qcolor("panel"))
        p.setPen(theme.qcolor("border"))
        x = self._gutter.width() - 1
        p.drawLine(x, e.rect().top(), x, e.rect().bottom())
        block = self.firstVisibleBlock()
        number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())
        current = self.textCursor().blockNumber()
        p.setFont(self.font())
        h = self.fontMetrics().height()
        while block.isValid() and top <= e.rect().bottom():
            if block.isVisible() and bottom >= e.rect().top():
                line = number + 1
                colour = ("bad" if line == self._error_line else "text" if number == current else "muted")
                p.setPen(theme.qcolor(colour))
                p.drawText(0, top, self._gutter.width() - 6, h, Qt.AlignmentFlag.AlignRight, str(line))
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            number += 1
        p.end()

    # -- selections: current line, error line ------------------------------------------
    def set_error_line(self, line: int | None) -> None:
        self._error_line = line
        self._update_selections()
        self._gutter.update()

    def goto_line(self, line: int) -> None:
        block = self.document().findBlockByNumber(max(0, int(line) - 1))
        if block.isValid():
            cur = QTextCursor(block)
            self.setTextCursor(cur)
            self.centerCursor()
            self.setFocus()

    def _update_selections(self) -> None:
        sels = []
        cur = QTextEdit.ExtraSelection()
        cur.format.setBackground(theme.qcolor("hover"))
        cur.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
        cur.cursor = self.textCursor()
        cur.cursor.clearSelection()
        sels.append(cur)
        if self._error_line is not None:
            block = self.document().findBlockByNumber(self._error_line - 1)
            if block.isValid():
                err = QTextEdit.ExtraSelection()
                err.format.setBackground(theme.qcolor("bad", 0.16))
                err.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                err.cursor = QTextCursor(block)
                sels.append(err)
        self.setExtraSelections(sels)
        self._gutter.update()

    def _retheme(self, _name: str = "") -> None:
        _use_mono(self)
        self._update_selections()

    # -- leaving the editor by keyboard (WCAG 2.1.2) --------------------------------------
    def tab_moves_focus(self) -> bool:
        return self._tab_moves_focus

    def set_tab_moves_focus(self, on: bool) -> None:
        """Plain Tab / Shift+Tab move focus instead of indenting while ``on``."""
        self._tab_moves_focus = bool(on)

    def leave(self, forward: bool = True) -> bool:
        """Move keyboard focus to the next (``forward``) or previous control.

        QPlainTextEdit.focusNextPrevChild refuses while the text is editable (that
        is how it keeps Tab for itself), so the QWidget implementation is called
        directly; it walks the window's tab chain as a Tab press in any other
        control would.
        """
        self._escape_armed = False
        return QWidget.focusNextPrevChild(self, forward)

    def event(self, e: QEvent) -> bool:
        # Ctrl+Tab is handled here, before QPlainTextEdit (which keeps it) and before
        # any window shortcut: ShortcutOverride is accepted so it reaches KeyPress.
        t = e.type()
        if t in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and isinstance(e, QKeyEvent):
            ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if ctrl and e.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                e.accept()
                if t == QEvent.Type.KeyPress:
                    shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                    self.leave(not shift and e.key() == Qt.Key.Key_Tab)
                return True
        return super().event(e)

    def focusOutEvent(self, e) -> None:
        self._escape_armed = False
        super().focusOutEvent(e)

    # -- editing keys ------------------------------------------------------------------
    def keyPressEvent(self, e: QKeyEvent) -> None:
        key, mods = e.key(), e.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) and not ctrl and (self._escape_armed or self._tab_moves_focus):
            self.leave(key == Qt.Key.Key_Tab and not (mods & Qt.KeyboardModifier.ShiftModifier))
            return
        if key == Qt.Key.Key_Escape:
            self._escape_armed = True
            super().keyPressEvent(e)        # unhandled Esc still reaches the parent
            return
        if key not in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            self._escape_armed = False      # any real key cancels the one-shot escape
        if key == Qt.Key.Key_Tab and not ctrl:
            self._indent(+1) if self.textCursor().hasSelection() else self.insertPlainText(" " * self.INDENT)
            return
        if key == Qt.Key.Key_Backtab:
            self._indent(-1)
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not ctrl and not (mods & Qt.KeyboardModifier.ShiftModifier):
            line = self.textCursor().block().text()[: self.textCursor().positionInBlock()]
            indent = len(line) - len(line.lstrip(" "))
            if line.rstrip().endswith(":"):
                indent += self.INDENT
            super().keyPressEvent(e)
            self.insertPlainText(" " * indent)
            return
        if key == Qt.Key.Key_Backspace and not self.textCursor().hasSelection():
            cur = self.textCursor()
            before = cur.block().text()[: cur.positionInBlock()]
            if before and not before.strip(" "):
                n = (len(before) - 1) % self.INDENT + 1
                for _ in range(n):
                    cur.deletePreviousChar()
                return
        if ctrl and key == Qt.Key.Key_Slash:
            self.toggle_comment()
            return
        super().keyPressEvent(e)
        if self._error_line is not None and e.text():
            self.set_error_line(None)          # editing clears the old failure mark

    def _selected_blocks(self) -> tuple[int, int]:
        cur = self.textCursor()
        a = self.document().findBlock(cur.selectionStart()).blockNumber()
        b = self.document().findBlock(cur.selectionEnd()).blockNumber()
        if cur.hasSelection() and b > a and self.document().findBlock(cur.selectionEnd()).position() == cur.selectionEnd():
            b -= 1                              # a selection ending at a line start excludes that line
        return a, b

    def _edit_lines(self, fn) -> None:
        a, b = self._selected_blocks()
        cur = self.textCursor()
        cur.beginEditBlock()
        for i in range(a, b + 1):
            block = self.document().findBlockByNumber(i)
            new = fn(block.text())
            if new != block.text():
                c = QTextCursor(block)
                c.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                c.insertText(new)
        cur.endEditBlock()

    def _indent(self, direction: int) -> None:
        pad = " " * self.INDENT
        if direction > 0:
            self._edit_lines(lambda t: pad + t if t.strip() else t)
        else:
            self._edit_lines(lambda t: t[min(self.INDENT, len(t) - len(t.lstrip(" "))):])

    def toggle_comment(self) -> None:
        a, b = self._selected_blocks()
        lines = [self.document().findBlockByNumber(i).text() for i in range(a, b + 1)]
        body = [t for t in lines if t.strip()]
        commented = bool(body) and all(t.lstrip().startswith("#") for t in body)
        if commented:
            self._edit_lines(lambda t: re.sub(r"^(\s*)# ?", r"\1", t, count=1))
        else:
            ind = min((len(t) - len(t.lstrip(" ")) for t in body), default=0)
            self._edit_lines(lambda t: (t[:ind] + "# " + t[ind:]) if t.strip() else t)


# ---------------------------------------------------------------------------
# the output
# ---------------------------------------------------------------------------

_TB_LINE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')


class OutputView(QTextBrowser):
    """Script output: stdout, errors, log lines, the engine's notes and inline figures.

    Segments are kept so a theme switch re-renders them in the new colours;
    ``File "<script>", line N`` references to the current script become links
    (``lineRequested(int)``). Figures are drawn as framed cards as wide as the
    pane allows (:data:`FIGURE_MAX_W`), shrunk to the pane's height only while
    they stay readable (:data:`FIGURE_MIN_W`); clicking one opens it at full
    size in a :class:`FigureViewer` (``open_figure(n)``, 1-based), from which
    it can be saved or copied. ``save_dir`` is where that viewer's Save starts.

    Scrolling follows the newest text, as a terminal does, until a run shows
    its first figure: the view then scrolls to that figure's top edge and stays
    there while the run prints and plots more below it (``begin_run()`` marks
    where a run starts). Following the tail would leave only the bottom rows of
    the last figure and the "Finished" line in a bottom dock; the first figure
    whole, with the rest a scroll away, is what the user wants to see first.
    Scrolling to the end by hand resumes following.
    """

    lineRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        _use_mono(self)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.setPlaceholderText("Output")
        self.document().setMaximumBlockCount(20000)
        self.document().setDocumentMargin(6)
        self.anchorClicked.connect(self._anchor)
        self._segments: list[tuple[str, object]] = []
        self._figures = 0
        self._pngs: dict[int, bytes] = {}       # figure number (as shown in the anchors) -> PNG
        self._last_figure: int | None = None    # document position of the newest figure
        self._figure_pos: dict[int, int] = {}  # figure number -> document position (redone on re-render)
        #: number of the first figure of the current run (None until it shows one)
        self._run_figure: int | None = None
        #: the view is held at the run's first figure instead of following the newest output
        self._held = False
        #: the view follows the newest output (until the user scrolls up)
        self._follow = True
        self._scrolling = False           # set while the view scrolls itself (not the user)
        self._figure_box = (0.0, 0.0)     # the (width, height) figures were laid out for
        self.script_file = ""
        self.save_dir: Path | None = None
        #: the open full-size viewers (kept referenced; they close with the console)
        self.viewers: list[FigureViewer] = []
        self._relayout = QTimer(self)
        self._relayout.setSingleShot(True)
        self._relayout.setInterval(150)
        self._relayout.timeout.connect(self._fit_figures)
        self.verticalScrollBar().valueChanged.connect(self._scrolled)
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)
        theme.changed.connect(self._retheme)

    # -- scrolling -----------------------------------------------------------------------
    def _set_scroll(self, value: int) -> None:
        """Scroll the view itself (not counted as the user scrolling)."""
        prev, self._scrolling = self._scrolling, True
        try:
            self.verticalScrollBar().setValue(value)
        finally:
            self._scrolling = prev

    def _scrolled(self, value: int) -> None:
        """The user scrolled: that lets go of a held figure, and following resumes only at the end.

        Following is a state, not a test of the scroll position at the moment
        text arrives: a dock made shorter (a window resized, a theme re-render)
        grows the scroll range under an unchanged value, and the old test then
        stopped following for the rest of the run.
        """
        if self._scrolling:
            return
        self._held = False
        self._follow = value >= self.verticalScrollBar().maximum() - 4

    def _range_changed(self, _lo: int, hi: int) -> None:
        """New text, a new pane size or a lazy re-layout: keep the view where it belongs."""
        if self._held:
            self.reveal_run_figure()
        elif self._follow:
            self._set_scroll(hi)

    def _at_end(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def follow_tail(self) -> None:
        """Stop holding at a figure and scroll to the newest output (a run starts or fails)."""
        self._held = False
        self._follow = True
        self._set_scroll(self.verticalScrollBar().maximum())

    def is_holding_figure(self) -> bool:
        """True while the view is held at the current run's first figure."""
        return self._held

    def begin_run(self) -> None:
        """A new run starts: its first figure (not an older run's) is the one the view goes to."""
        self._run_figure = None
        self._held = False

    def reveal_figure(self, number: int) -> bool:
        """Scroll so figure ``number``'s top edge is at the top of the view; False if unknown."""
        pos = self._figure_pos.get(int(number))
        if pos is None:
            return False
        cur = self.textCursor()
        cur.setPosition(min(pos, self.document().characterCount() - 1))
        top = self.cursorRect(cur).top()
        bar = self.verticalScrollBar()
        self._set_scroll(max(0, min(bar.maximum(), bar.value() + top - 4)))
        return True

    def reveal_run_figure(self) -> None:
        """Back to the current run's first figure while the view is held there (a no-op once
        the user has scrolled to the end, or for a run without figures)."""
        if self._held and self._run_figure is not None:
            self.reveal_figure(self._run_figure)

    def _format(self, stream: str) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setFont(theme.mono_font())
        token = {"stdout": "text", "stderr": "bad", "log": "text2", "info": "muted", "head": "muted"}.get(stream, "text")
        f.setForeground(theme.qcolor(token))
        f.setFontItalic(stream in ("info", "head"))
        return f

    def _cursor_at_end(self) -> QTextCursor:
        cur = self.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        return cur

    def append_text(self, text: str, stream: str = "stdout", *, record: bool = True) -> None:
        if not text:
            return
        if record:
            self._remember(stream, text)
        bar = self.verticalScrollBar()
        follow = record and not self._held and self._follow
        cur = self._cursor_at_end()
        fmt = self._format(stream)
        if stream == "stderr" and self.script_file:
            pos = 0
            for m in _TB_LINE.finditer(text):
                if m.group("file") != self.script_file:
                    continue
                cur.insertText(text[pos:m.start("line") - len("line ")], fmt)
                link = QTextCharFormat(fmt)
                link.setAnchor(True)
                link.setAnchorHref(f"line:{m.group('line')}")
                link.setFontUnderline(True)
                cur.insertText(f"line {m.group('line')}", link)
                pos = m.end("line")
            cur.insertText(text[pos:], fmt)
        else:
            cur.insertText(text, fmt)
        if follow:
            self._set_scroll(bar.maximum())

    def figure_width(self, img_width: int, img_height: int) -> float:
        """The logical width an inline figure of that pixel size is drawn at.

        As wide as the pane allows (at most :data:`FIGURE_MAX_W`, never wider
        than the image), and narrower when that makes the whole figure fit the
        pane's height -- but not below :data:`FIGURE_MIN_W`: a figure squeezed
        into a short bottom dock had titles and tick labels too small to read,
        so a short dock scrolls instead (and the figure opens full size on a click).
        """
        aspect = img_width / max(img_height, 1)
        avail_w, avail_h = self._available_width(), self._available_height()
        return float(min(img_width, avail_w, max(min(FIGURE_MIN_W, avail_w), avail_h * aspect)))

    def append_figure(self, png: bytes, *, record: bool = True) -> None:
        img = QImage.fromData(png, "PNG")
        if img.isNull():
            return
        if record:
            self._remember("figure", png)
        following = record and not self._held and self._follow
        self._figures += 1
        self._pngs[self._figures] = png
        name = f"aerolume-figure://{self._figures}"
        # the logical width it is shown at, and a copy resampled smoothly to exactly that many
        # device pixels: the text document scales images by sampling, which turns the thin lines
        # and small labels of a plot ragged
        dpr = max(1.0, self.devicePixelRatioF())
        self._figure_box = (self._available_width(), self._available_height())
        aspect = img.width() / max(img.height(), 1)
        w = self.figure_width(img.width(), img.height())
        h = w / aspect
        target = round(w * dpr)
        if target != img.width():
            img = img.scaledToWidth(target, Qt.TransformationMode.SmoothTransformation)
        img = _framed(img, dpr)
        img.setDevicePixelRatio(dpr)
        self.document().addResource(QTextDocument.ResourceType.ImageResource, QUrl(name), img)
        cur = self._cursor_at_end()
        if cur.positionInBlock() > 0:
            cur.insertText("\n", self._format("stdout"))
        # the figure gets a block of its own with a gap above and below, so consecutive figures
        # read as separate cards rather than one patchwork image
        block = QTextBlockFormat()
        block.setTopMargin(FIGURE_GAP)
        block.setBottomMargin(FIGURE_GAP)
        cur.setBlockFormat(block)
        fmt = QTextImageFormat()
        fmt.setName(name)
        fmt.setWidth(w)
        fmt.setHeight(h)
        fmt.setAnchor(True)
        fmt.setAnchorHref(f"figure:{self._figures}")
        fmt.setToolTip("Click to open the figure at full size")
        self._last_figure = cur.position()
        self._figure_pos[self._figures] = cur.position()
        cur.insertImage(fmt)
        cur.insertBlock(QTextBlockFormat(), self._format("stdout"))
        if not record:
            return                      # re-rendering: _render restores the view afterwards
        if self._run_figure is None and following:
            # the run's first figure: bring its top edge into view and hold the view there
            self._run_figure = self._figures
            self._held = True
            self.reveal_run_figure()
            # the document lays the image out lazily: settle the scroll once the layout is final
            QTimer.singleShot(0, self.reveal_run_figure)
        elif following:
            self._set_scroll(self.verticalScrollBar().maximum())

    def figure_count(self) -> int:
        """Figures shown since the output was last cleared."""
        return len(self._pngs)

    def open_figure(self, number: int) -> FigureViewer | None:
        """Open figure ``number`` (1-based, in the order shown) at full size."""
        png = self._pngs.get(int(number))
        if png is None:
            return None
        viewer = FigureViewer(png, f"Figure {number}", save_dir=self.save_dir, parent=self.window())
        viewer.finished.connect(lambda _r, v=viewer: self.viewers.remove(v) if v in self.viewers else None)
        self.viewers.append(viewer)
        viewer.show()
        viewer.raise_()
        return viewer

    def reveal_last_figure(self) -> None:
        """Scroll so the newest figure's top edge is in view (a run that ends with a tall
        figure otherwise leaves only its bottom rows and the "Finished" line showing)."""
        if self._last_figure is None:
            return
        cur = self.textCursor()
        cur.setPosition(min(self._last_figure, self.document().characterCount() - 1))
        top = self.cursorRect(cur).top()
        bar = self.verticalScrollBar()
        if top < 0 or top > self.viewport().height():
            self._set_scroll(bar.value() + top - 4)

    def _remember(self, kind: str, payload: object) -> None:
        self._segments.append((kind, payload))
        if len(self._segments) > MAX_SEGMENTS:
            del self._segments[: len(self._segments) - MAX_SEGMENTS]

    def clear_output(self) -> None:
        self._segments.clear()
        self._last_figure = None
        self._run_figure = None
        self._held = False
        self._follow = True
        self._forget_figures()
        self.clear()

    def _forget_figures(self) -> None:
        self._figures = 0
        self._pngs.clear()
        self._figure_pos.clear()

    def plain_text(self) -> str:
        return self.toPlainText()

    def _available_width(self) -> float:
        return float(min(FIGURE_MAX_W, max(200, self.viewport().width() - 24)))

    def _available_height(self) -> float:
        return float(max(100, self.viewport().height() - 32))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._figures:
            self._relayout.start()

    def _fit_figures(self) -> None:
        """Lay the figures out again for a pane that became narrower or wider (dock resized)."""
        if not any(k == "figure" for k, _ in self._segments):
            return
        w0, h0 = self._figure_box
        if abs(self._available_width() - w0) > 24 or abs(self._available_height() - h0) > 24:
            self._render()

    def _retheme(self, _name: str = "") -> None:
        _use_mono(self)
        self._render()

    def _render(self) -> None:
        """Draw the kept segments again (new colours, new figure widths)."""
        bar = self.verticalScrollBar()
        was = bar.value()               # where a user who scrolled up was reading
        segs = list(self._segments)
        self._scrolling = True          # clearing jumps to the top: not the user scrolling
        try:
            self.clear()
            self._forget_figures()      # numbered again in the same order: the anchors stay right
            for kind, payload in segs:
                if kind == "figure":
                    self.append_figure(payload, record=False)       # type: ignore[arg-type]
                else:
                    self.append_text(payload, kind, record=False)   # type: ignore[arg-type]
        finally:
            self._scrolling = False
        if self._held:
            self.reveal_run_figure()       # the figure moved with the new widths: follow it
        elif self._follow:
            self._set_scroll(bar.maximum())
        else:
            self._set_scroll(min(was, bar.maximum()))

    def _anchor(self, url: QUrl) -> None:
        s = url.toString()
        if s.startswith("line:"):
            self.lineRequested.emit(int(s.split(":", 1)[1]))
        elif s.startswith("figure:"):
            self.open_figure(int(s.split(":", 1)[1]))

    def close_viewers(self) -> None:
        for v in list(self.viewers):
            v.close()
        self.viewers.clear()


def _framed(img: QImage, dpr: float) -> QImage:
    """``img`` with a one-pixel (logical) frame in the theme's border colour.

    Figures are white sheets (the print style, the same as saved PNGs and
    reports); the frame gives them an edge on the light theme's white output
    and keeps them a tidy card on the dark one.
    """
    out = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(out)
    pen_w = max(1, round(dpr))
    p.setPen(theme.qcolor("border"))
    for i in range(pen_w):
        p.drawRect(i, i, out.width() - 1 - 2 * i, out.height() - 1 - 2 * i)
    p.end()
    return out


class FigureViewer(QDialog):
    """A figure from the script output at full size: fit to the window, Save image… and Copy.

    ``FigureViewer(png, title, save_dir=None, parent=None)``. Modeless, so the
    script and the console stay usable; the image is resampled smoothly to the
    window (never enlarged past its own pixels). ``save(path)`` writes the PNG
    as the script produced it (no frame, full resolution).
    """

    def __init__(self, png: bytes, title: str, *, save_dir: Path | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setObjectName("FigureViewer")
        self.png = bytes(png)
        self.image = QImage.fromData(self.png, "PNG")
        self.save_dir = save_dir
        self._title = title
        #: the save-path prompt, swappable in tests
        self._ask_save_path = self._dialog_save_path
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = ToolStrip()
        self.save_button = bar.add_button("download", "Save image…", "Save the figure as a PNG file",
                                          on_click=lambda: self.save())
        self.copy_button = bar.add_button("copy", "Copy", "Copy the figure to the clipboard", on_click=self.copy)
        bar.add_stretch()
        size = label(f"{self.image.width()} × {self.image.height()} px", "muted")
        bar.add_widget(size)
        bar.row.addSpacing(SPACING)
        lay.addWidget(bar)
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumSize(200, 150)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        lay.addWidget(self.canvas, 1)
        dpr = max(1.0, self.devicePixelRatioF())
        avail = self.screen().availableGeometry().size() if self.screen() is not None else QSize(1600, 1000)
        natural = QSize(round(self.image.width() / dpr), round(self.image.height() / dpr) + bar.height())
        self.resize(min(natural.width() + 24, int(avail.width() * 0.8)),
                    min(natural.height() + 24, int(avail.height() * 0.8)))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._fit()

    def _fit(self) -> None:
        if self.image.isNull():
            return
        dpr = max(1.0, self.devicePixelRatioF())
        box = self.canvas.size() - QSize(24, 24)
        target = self.image.size().scaled(round(box.width() * dpr), round(box.height() * dpr),
                                          Qt.AspectRatioMode.KeepAspectRatio)
        if target.width() >= self.image.width():
            img = self.image
        else:
            img = self.image.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(dpr)
        self.canvas.setPixmap(pm)

    def copy(self) -> None:
        QApplication.clipboard().setImage(self.image)

    def save(self, path: str | Path | None = None) -> Path | None:
        """Write the PNG (asks for a path without one); returns the path written."""
        if path is None:
            path = self._ask_save_path()
            if not path:
                return None
        p = Path(path)
        if p.suffix.lower() != ".png":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.png)
        return p

    def _dialog_save_path(self) -> str | None:
        stem = re.sub(r"[^\w\-]+", "-", self._title.lower()).strip("-") or "figure"
        start = (self.save_dir / f"{stem}.png") if self.save_dir is not None else Path(f"{stem}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save image", str(start), "PNG image (*.png)")
        return path or None


# ---------------------------------------------------------------------------
# the console
# ---------------------------------------------------------------------------

def _seq(text: str) -> QKeySequence:
    return QKeySequence(text)


class ScriptConsole(QWidget):
    """The Script panel; see the module docstring for what it binds, its keys and how to dock it."""

    runStarted = Signal()
    runFinished = Signal(object)
    runningChanged = Signal(bool)
    scriptSaved = Signal(str)

    def __init__(self, state, jobs=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ScriptConsole")
        self.state = state
        self.jobs = jobs
        self.engine = scripting.ScriptEngine(state, jobs, self)
        self.current_path: Path | None = None
        self._buffers: dict[str, str] = {}
        self._runs = 0
        self._run_file = ""
        self._last_status = ""
        #: the main window and dock this console sits in (see _watch_window), and the share of
        #: the window height the dock is kept at when the window is resized
        self._host: QMainWindow | None = None
        self._dock: QDockWidget | None = None
        self._dock_share: float | None = None
        self._reshare = QTimer(self)
        self._reshare.setSingleShot(True)
        self._reshare.setInterval(0)
        self._reshare.timeout.connect(self._apply_share)
        #: dialogs, swappable in tests: name for a new script, a save path without a project, an open path
        self._ask_name = self._dialog_name
        self._ask_save_path = self._dialog_save_path
        self._ask_open_path = self._dialog_open_path
        self._build()
        self.engine.output.connect(self.output.append_text)
        self.engine.figure.connect(self.output.append_figure)
        self.engine.progress.connect(self._on_progress)
        self.engine.started.connect(self._on_started)
        self.engine.finished.connect(self._on_finished)
        self.engine.stateChanged.connect(self._on_running)
        self.output.lineRequested.connect(self.editor.goto_line)
        self.editor.document().modificationChanged.connect(self._refresh_combo_text)
        state.projectChanged.connect(self._on_project_changed)
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_running_status)
        for w in (self, self.editor, self.output, self.combo):
            w.installEventFilter(self)
        self._on_project_changed()
        self._on_running(False)

    # -- layout --------------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        bar = ToolStrip()
        self.toolbar = bar
        self.run_button = bar.add_button("run", "Run", "Run the script (Ctrl+Enter; F5 while the console has focus)",
                                         on_click=self.run)
        self.stop_button = bar.add_button("stop", "Stop", "Stop the script and its simulations (Shift+F5 while the "
                                          "console has focus); simulations keep their partial results",
                                          on_click=self.stop)
        bar.add_separator()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(170)
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo.setToolTip("Scripts saved in this project (the project's scripts folder)")
        self.combo.activated.connect(self._combo_activated)
        bar.add_widget(self.combo)
        self.new_button = IconButton("new", "New script")
        self.new_button.clicked.connect(lambda: self.new_script())
        bar.add_widget(self.new_button)
        self.open_button = IconButton("open", "Open a script file…")
        self.open_button.clicked.connect(lambda: self.open_script())
        bar.add_widget(self.open_button)
        self.save_button = IconButton("save", "Save the script in the project (Ctrl+S)")
        self.save_button.clicked.connect(lambda: self.save_script())
        bar.add_widget(self.save_button)
        bar.add_separator()
        self.examples_button = QToolButton()
        self.examples_button.setText("Examples")
        self.examples_button.setProperty("dropdown", "true")
        self.examples_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.examples_button.setToolTip("Insert a ready script")
        self.examples_button.setFixedHeight(theme.CONTROL_H)
        menu = QMenu(self.examples_button)
        for s in scripting.SNIPPETS:
            act = menu.addAction(s.title)
            act.setToolTip(s.description)
            act.setData(s.key)
            act.triggered.connect(lambda _=False, k=s.key: self.insert_snippet(k))
        menu.setToolTipsVisible(True)
        self.examples_button.setMenu(menu)
        bar.add_widget(self.examples_button)
        self.more_button = QToolButton()
        self.more_button.setIcon(theme.icon("more", "text2"))
        self.more_button.setProperty("variant", "icon")
        self.more_button.setToolTip("More")
        self.more_button.setFixedSize(theme.CONTROL_H, theme.CONTROL_H)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more = QMenu(self.more_button)
        self.act_run_selection = more.addAction("Run selection or line", self.run_selection)
        self.act_run_selection.setShortcut(_seq("Ctrl+Shift+Return"))
        self.act_save_as = more.addAction("Save as…", lambda: self.save_script(new_name=True))
        more.addSeparator()
        self.act_reset = more.addAction("Reset variables", self.reset_variables)
        self.act_reset.setToolTip("Forget the variables kept from earlier runs")
        self.act_clear = more.addAction("Clear output", self.clear_output)
        more.addSeparator()
        # A standing alternative to Ctrl+Tab / Esc-then-Tab for keyboard-only users.
        self.act_tab_focus = more.addAction("Tab key moves focus")
        self.act_tab_focus.setCheckable(True)
        self.act_tab_focus.setToolTip("Tab and Shift+Tab move between controls instead of indenting "
                                      "(Ctrl+Tab always does)")
        more.setToolTipsVisible(True)
        self.more_button.setMenu(more)
        bar.add_widget(self.more_button)
        bar.add_stretch()
        self.message = ElidedLabel("", "muted")
        self.message.setMaximumWidth(360)
        self.message.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        bar.add_widget(self.message)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(120)
        self.progress.setTextVisible(False)
        self.progress.hide()
        bar.add_widget(self.progress)
        bar.row.addSpacing(SPACING)
        self.status = label("Ready", "secondary")
        bar.add_widget(self.status)
        bar.row.addSpacing(SPACING)
        root.addWidget(bar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.editor = CodeEditor()
        self.editor.setPlaceholderText("Python · Ctrl+Enter runs the script · Examples has ready scripts")
        self.output = OutputView()
        self.act_tab_focus.toggled.connect(self.editor.set_tab_moves_focus)
        self.splitter.addWidget(self.editor)
        self.splitter.addWidget(self.output)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        root.addWidget(self.splitter, 1)
        self.setMinimumHeight(140)

    def sizeHint(self) -> QSize:
        return QSize(900, 300)

    # -- the dock's share of the window -------------------------------------------------------
    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._watch_window()

    def _watch_window(self) -> None:
        """Watch the main window this console is docked in (once), to keep the dock's share.

        The window opens the dock at about 40 % of its height, but a docked
        widget keeps its pixel height when the window changes size: shrinking
        the window squeezes the dock down to what the page above leaves (about
        215 px at 1280x800), and growing it again hands all the new height to
        the page, so after a restore or a maximise the dock showed only the top
        of one figure. The console therefore remembers its share of the window
        height -- taken before the first resize, and again whenever the user
        drags the dock's edge -- and asks for it back after every resize.
        """
        dock = self.parentWidget()
        while dock is not None and not isinstance(dock, QDockWidget):
            dock = dock.parentWidget()          # the window frames dock content in a QFrame
        host = dock.parentWidget() if dock is not None else None
        if dock is None or not isinstance(host, QMainWindow) or host is self._host:
            return
        if self._host is not None:
            self._host.removeEventFilter(self)
        self._host, self._dock = host, dock
        host.installEventFilter(self)

    def dock_share(self) -> float | None:
        """The share of the window height the Script dock is kept at (None before it is known)."""
        return self._dock_share

    def set_dock_share(self, share: float | None) -> None:
        """Keep the Script dock at ``share`` of the window height (the workspace's figure), or, with None,
        fall back to the host's own share on the next resize. The main window calls this when a
        workspace sets the bottom rail, so the console and the workspace never pull the rail to two
        different heights."""
        self._dock_share = float(share) if isinstance(share, (int, float)) and 0.0 < float(share) < 1.0 else None

    def _docked(self) -> bool:
        d = self._dock
        return d is not None and self._host is not None and d.isVisible() and not d.isFloating()

    def _host_event(self, e: QEvent) -> None:
        t = e.type()
        if t == QEvent.Type.Resize and self._docked():
            old = e.oldSize().height()
            if self._dock_share is None and old > 0:
                # the share the window opens the dock at, when it says (a small window caps the
                # dock below it, and that cap must not stick once the window grows); otherwise
                # the share it has now (the layout has not followed the new size yet)
                opening = getattr(self._host, "SCRIPT_SHARE", None)
                self._dock_share = (float(opening) if isinstance(opening, (int, float)) and 0 < opening < 1
                                    else self._dock.height() / old)
            if self._dock_share is not None:
                self._reshare.start()
        elif t == QEvent.Type.MouseButtonRelease and self._docked() and not self._reshare.isActive():
            # the window's own mouse events are its dock separators: the user set a new height
            QTimer.singleShot(0, self._remember_share)

    def _remember_share(self) -> None:
        if self._docked() and self._host.height() > 0:
            self._dock_share = self._dock.height() / self._host.height()

    def _apply_share(self) -> None:
        if self._docked() and self._dock_share is not None:
            want = round(self._dock_share * self._host.height())
            if abs(want - self._dock.height()) > 2:
                # the page above keeps its minimum height: resizeDocks gives what it can
                self._host.resizeDocks([self._dock], [want], Qt.Orientation.Vertical)

    # -- keys (they win over the window's actions while the console has focus) -------------
    _RUN_KEYS = ("Ctrl+Return", "Ctrl+Enter", "F5")
    _RUN_SELECTION_KEYS = ("Ctrl+Shift+Return", "Ctrl+Shift+Enter")
    _STOP_KEYS = ("Shift+F5",)
    _SAVE_KEYS = ("Ctrl+S",)

    @staticmethod
    def _matches(e: QKeyEvent, keys: tuple[str, ...]) -> bool:
        combo = QKeySequence(e.keyCombination().toCombined())
        return any(combo.matches(_seq(k)) == QKeySequence.SequenceMatch.ExactMatch for k in keys)

    def eventFilter(self, obj: QObject, e: QEvent) -> bool:
        if obj is self._host:
            self._host_event(e)     # the window's keys are its own: only resizes are watched there
            return False
        t = e.type()
        if t in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and isinstance(e, QKeyEvent):
            action = None
            if self._matches(e, self._RUN_SELECTION_KEYS):
                action = self.run_selection
            elif self._matches(e, self._RUN_KEYS):
                action = self.run
            elif self._matches(e, self._STOP_KEYS) and self.is_running():
                action = self.stop
            elif self._matches(e, self._SAVE_KEYS):
                action = self.save_script
            if action is not None:
                e.accept()
                if t == QEvent.Type.KeyPress:
                    action()
                return True
        return super().eventFilter(obj, e)

    # -- running ---------------------------------------------------------------------------
    def is_running(self) -> bool:
        return self.engine.is_running()

    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, text: str) -> None:
        """Replace the editor's text (undoable); for a script with a file, text that differs
        from the file counts as an unsaved edit."""
        self.editor.setPlainText(text)
        self.editor.set_error_line(None)
        path = self.current_path
        try:
            differs = path is not None and path.is_file() and text != path.read_text(encoding="utf-8")
        except OSError:
            differs = path is not None
        self.editor.document().setModified(differs)

    def _script_file(self) -> str:
        return str(self.current_path) if self.current_path is not None else "<untitled script>"

    def run(self) -> bool:
        """Run the whole script (a script with a file is saved first)."""
        if self.is_running():
            return False
        if self.current_path is not None and self.editor.document().isModified():
            self._write(self.current_path)
        return self._start(self.text(), self._script_file())

    def run_selection(self) -> bool:
        """Run the selected lines (or the current line), dedented, keeping the editor's line numbers."""
        if self.is_running():
            return False
        cur = self.editor.textCursor()
        if cur.hasSelection():
            a, b = self.editor._selected_blocks()
        else:
            a = b = cur.blockNumber()
        lines = [self.editor.document().findBlockByNumber(i).text() for i in range(a, b + 1)]
        code = textwrap.dedent("\n".join(lines))
        if not code.strip():
            return False
        # padded so a traceback's line numbers are the editor's
        return self._start("\n" * a + code + "\n", self._script_file(), selection=True)

    def _start(self, source: str, filename: str, *, selection: bool = False) -> bool:
        self.editor.set_error_line(None)
        self._runs += 1
        self._run_file = filename
        self.output.script_file = filename
        name = Path(filename).name if self.current_path is not None else UNTITLED
        what = "selection of " if selection else ""
        stamp = time.strftime("%H:%M:%S")
        sep = "\n" if self.output.plain_text() else ""
        self.output.begin_run()
        self.output.follow_tail()
        self.output.append_text(f"{sep}Run {self._runs} · {what}{name} · {stamp}\n", "head")
        return self.engine.run(source, filename=filename)

    def stop(self) -> None:
        self.engine.stop()
        if self.is_running():
            self.status.setText("Stopping…")

    def _on_started(self) -> None:
        self.progress.setValue(0)
        self.message.setText("")
        self.runStarted.emit()

    def _on_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.act_run_selection.setEnabled(not running)
        self.act_reset.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self._tick.start()
            self._update_running_status()
        else:
            self._tick.stop()
        self.runningChanged.emit(running)

    def _update_running_status(self) -> None:
        if self.is_running():
            self.status.setText("Stopping…" if self.engine._stops else f"Running · {self.engine.elapsed():.0f} s")
            theme.set_prop(self.status, "role", "secondary")

    #: what the run message beside the status is about (DESIGN.md 8: progress comes from the
    #: script's simulation calls, never from the script itself)
    MESSAGE_PREFIX = "Simulation: "

    def _on_progress(self, fraction: float, message: str) -> None:
        """A simulation the script started reports progress.

        Prefixed "Simulation:" because the line outlives the simulation: once it
        says "Done after 22 s" the script itself may still be plotting, and
        unprefixed it read as the script's verdict next to "Running · 23 s".
        A report that arrives after the script finished is dropped, so the
        cleared line stays clear.
        """
        if not self.is_running():
            return
        self.progress.setValue(round(1000 * max(0.0, min(1.0, fraction))))
        message = message.strip()
        self.message.setText(f"{self.MESSAGE_PREFIX}{message}" if message else "")

    def _on_finished(self, outcome: scripting.ScriptOutcome) -> None:
        secs = _duration(outcome.seconds)
        n = len(outcome.sim_ids)
        sims = f" · {n} simulation{'s' if n != 1 else ''} recorded" if n else ""
        reused = [s for s in getattr(outcome, "reused_ids", []) if s not in outcome.sim_ids]
        if reused:
            # "Finished in 1.3 s" alone read as a 1.3 s simulation: say that nothing ran and how to run it again
            k = len(reused)
            sims += (f" · {k} stored result{'s' if k != 1 else ''} of an identical run reused, not run again "
                     "(rerun=True runs it again)")
        if outcome.status == "finished":
            text, role = f"Finished in {secs}", "secondary"
            if reused:
                text += " · stored result reused"
            note = f"Finished in {secs}{sims}\n"
        elif outcome.status == "stopped":
            text, role = f"Stopped after {secs}", "status-warn"
            note = f"Stopped after {secs}{sims}\n"
        else:
            where = f" on line {outcome.line}" if outcome.line else ""
            text, role = f"Failed{where}", "status-bad"
            note = ""
            self.output.follow_tail()       # the error is what matters now, not the run's figure
            if outcome.traceback:
                self.output.append_text(outcome.traceback, "stderr")
            elif outcome.error:
                self.output.append_text(outcome.error + "\n", "stderr")
            if outcome.line and outcome.filename == self._script_file():
                self.editor.set_error_line(outcome.line)
        if note:
            self.output.append_text(note, "info")
        if outcome.figures and outcome.status == "finished":
            self.output.reveal_run_figure()
        self.status.setText(text)
        self.status.setToolTip(outcome.error or text)
        theme.set_prop(self.status, "role", role)
        self.message.setText("")
        self.progress.hide()
        self.runFinished.emit(outcome)

    def reset_variables(self) -> None:
        self.engine.reset()
        self.output.append_text("Variables reset.\n", "info")

    def clear_output(self) -> None:
        self.output.clear_output()

    def shutdown(self) -> None:
        """Stop a running script and stop waiting for the GUI (call when the window closes)."""
        self._tick.stop()
        self.output.close_viewers()
        self.engine.shutdown()

    # -- scripts ---------------------------------------------------------------------------
    def _key(self, path: Path | None) -> str:
        return str(path) if path is not None else UNTITLED

    def _scripts_dir(self) -> Path | None:
        return scripting.scripts_dir(self.state.project)

    def _refresh_combo(self) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        paths = scripting.list_scripts(self.state.project)
        if self.current_path is None:
            self.combo.addItem(UNTITLED, None)
        elif self.current_path not in paths:
            self.combo.addItem(self.current_path.name, str(self.current_path))
        for p in paths:
            self.combo.addItem(p.name, str(p))
        idx = self.combo.findData(None if self.current_path is None else str(self.current_path))
        self.combo.setCurrentIndex(max(idx, 0))
        self.combo.blockSignals(False)
        self._refresh_combo_text()
        has_project = self.state.project is not None
        self.save_button.setToolTip("Save the script in the project (Ctrl+S)" if has_project
                                    else "Save the script to a file (Ctrl+S)")

    def _refresh_combo_text(self, _modified: bool = False) -> None:
        i = self.combo.currentIndex()
        if i < 0:
            return
        base = UNTITLED if self.current_path is None else self.current_path.name
        mark = " *" if self.editor.document().isModified() and self.current_path is not None else ""
        self.combo.setItemText(i, base + mark)

    def _stash(self) -> None:
        self._buffers[self._key(self.current_path)] = self.text()

    def _load(self, path: Path | None, text: str | None = None) -> None:
        self._stash()
        self.current_path = path
        if text is None:
            key = self._key(path)
            if key in self._buffers:
                text = self._buffers[key]
            elif path is not None:
                text = path.read_text(encoding="utf-8")
            else:
                text = NEW_SCRIPT
        self.set_text(text)
        modified = path is not None and path.is_file() and text != path.read_text(encoding="utf-8")
        self.editor.document().setModified(modified)
        self._refresh_combo()

    def _combo_activated(self, index: int) -> None:
        data = self.combo.itemData(index)
        path = Path(data) if data else None
        if path != self.current_path:
            self._load(path)

    def new_script(self) -> None:
        """An empty untitled script (a modified untitled one is kept until replaced)."""
        self._buffers.pop(UNTITLED, None) if self.current_path is not None else None
        self._load(None, NEW_SCRIPT)
        self.editor.setFocus()

    def open_script(self, path: str | Path | None = None) -> bool:
        """Open a ``.py`` file into the editor (asks for one without ``path``)."""
        if path is None:
            path = self._ask_open_path()
            if not path:
                return False
        p = Path(path)
        if not p.is_file():
            self.state.message(f"No script at {p}", 6000)
            return False
        self._buffers.pop(str(p), None)
        self._load(p)
        return True

    def _write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(self.text(), encoding="utf-8", newline="\n")
        tmp.replace(path)
        self.editor.document().setModified(False)
        self._buffers.pop(str(path), None)
        self._refresh_combo_text()
        self.scriptSaved.emit(str(path))

    def save_script(self, name: str | None = None, *, new_name: bool = False) -> Path | None:
        """Save the script: to its file, or as ``<project>/scripts/<name>.py`` (asks for a
        name when needed; without a project, asks for a file)."""
        path = None if new_name else self.current_path
        if path is None:
            folder = self._scripts_dir()
            if folder is None:
                chosen = self._ask_save_path()
                if not chosen:
                    return None
                path = Path(chosen)
            else:
                name = name or self._ask_name(_suggest_name(self.text()))
                if not name:
                    return None
                stem = re.sub(r"[^\w\- .]+", "_", str(name).strip()).strip(" .") or "script"
                path = folder / (stem if stem.lower().endswith(".py") else stem + ".py")
        old = self.current_path
        self._write(path)
        if old != path:
            self._buffers.pop(self._key(old), None)
            self.current_path = path
        self._refresh_combo()
        self.state.message(f"Saved {path.name}", 3000)
        return path

    def insert_snippet(self, key: str) -> None:
        """Put an example script into the editor: replaces an empty or unchanged buffer, otherwise
        opens it as a new untitled script (the previous text stays in its buffer)."""
        code = scripting.snippet(key).code
        current = self.text().strip()
        pristine = not current or current == NEW_SCRIPT.strip() or (
            self.current_path is None and any(current == s.code.strip() for s in scripting.SNIPPETS))
        if pristine and self.current_path is None:
            self.set_text(code)
            self.editor.document().setModified(True)
            self._refresh_combo()
        else:
            self._load(None, code)
        self.editor.moveCursor(QTextCursor.MoveOperation.Start)
        self.editor.setFocus()

    # -- dialogs (swapped in tests) ----------------------------------------------------
    def _dialog_name(self, suggestion: str) -> str | None:
        text, ok = QInputDialog.getText(self, "Save script", "Script name:", text=suggestion)
        return text if ok and text.strip() else None

    def _dialog_save_path(self) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, "Save script", "script.py", "Python scripts (*.py)")
        return path or None

    def _dialog_open_path(self) -> str | None:
        start = str(self._scripts_dir() or "")
        path, _ = QFileDialog.getOpenFileName(self, "Open script", start, "Python scripts (*.py);;All files (*)")
        return path or None

    # -- project changes -----------------------------------------------------------------
    def _on_project_changed(self) -> None:
        project = self.state.project
        self.output.save_dir = Path(project.path) / "exports" if project is not None else None
        folder = self._scripts_dir()
        inside = (self.current_path is not None and folder is not None
                  and self.current_path.parent == folder)
        if self.current_path is not None and not inside and self.current_path.parent.name == "scripts":
            # a script of the previous project: keep unsaved text as an untitled script
            text = self.text() if self.editor.document().isModified() else None
            self._buffers.pop(str(self.current_path), None)
            self.current_path = None
            self._buffers.pop(UNTITLED, None)
            self._load(None, text if text is not None else NEW_SCRIPT)
            return
        if not self.text():
            self.set_text(NEW_SCRIPT)
            self.editor.document().setModified(False)
        self._refresh_combo()


def _duration(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 10:
        return f"{s:.1f} s"
    if s < 60:
        return f"{s:.0f} s"
    m, r = divmod(round(s), 60)
    return f"{m} min {r:02d} s" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def _suggest_name(text: str) -> str:
    """A file name from the script's docstring or first comment ("polar_sweep")."""
    for line in text.splitlines():
        s = line.strip().strip('"\'#').strip()
        if s and not s.startswith(("import ", "from ", "project:")):
            words = re.findall(r"[A-Za-z0-9]+", s.lower())[:4]
            if words:
                return "_".join(words)
    return "script"

