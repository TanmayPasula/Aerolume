"""The workflow step bar: Geometry - Setup - Simulate - Results - Compare - Report.

Flat tabs in one row under the main toolbar (docs/ui_guidelines.md section 3):
the active step has a 2 px ink underline, a completed step carries a small
check glyph, and nothing else decorates them - no numbers in circles, no
subtitles. The bar keeps the workflow guided without taking space: the
guidance is the order of the words.

It is painted by hand (one widget, no per-tab child widgets) so it is cheap to
repaint when a simulation's status changes. The right end shows a plain-text
context line (current design and simulation) because the project tree may be
hidden. For 3D work the line starts with a short tag ("3D", "Wing") in the
secondary text colour, so the breadcrumb says which kind of model the pages
are showing without a badge: the same pages serve 2D outlines, 3D meshes and
wings, and the tag is the one place the shell says which is on screen.

Conventions other code relies on:

* keys are the page keys ``"geometry"``, ``"setup"``, ... (``STEP_KEYS``);
* ``stepRequested(key)`` fires on click / Enter; the main window turns it into
  ``AppState.navigate(key)`` so pages never talk to the bar directly;
* ``set_states({key: done})`` drives the check glyphs, ``set_current(key)``
  the underline (``""`` = none, e.g. on the start page);
* ``set_context(text, tag)`` sets the breadcrumb and its leading tag
  (``"3D"``, ``"Wing"``; ``""`` for 2D, which needs no tag).
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QFont, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import theme

STEPS: tuple[tuple[str, str], ...] = (
    ("geometry", "Geometry"), ("setup", "Setup"), ("simulate", "Simulate"),
    ("results", "Results"), ("compare", "Compare"), ("report", "Report"),
)
STEP_KEYS = tuple(k for k, _ in STEPS)

_PAD = 12          # horizontal padding inside a tab
_CHECK = 10        # check glyph size
_HEIGHT = 30


class StepBar(QWidget):
    """Flat workflow tabs with an ink underline and completion checks."""

    stepRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setAccessibleName("Workflow steps")
        self._current = ""
        self._focus_key = STEP_KEYS[0]
        self._done: dict[str, bool] = dict.fromkeys(STEP_KEYS, False)
        self._enabled: dict[str, bool] = dict.fromkeys(STEP_KEYS, True)
        self._hover = ""
        self._context = ""
        self._tag = ""
        self._rects: dict[str, QRectF] = {}
        theme.changed.connect(self._on_theme)    # bound method: auto-disconnects when deleted

    def _on_theme(self, _name: str) -> None:
        self.update()

    # -- state ------------------------------------------------------------------
    def set_current(self, key: str) -> None:
        """Underline ``key`` ("" for none)."""
        if key != self._current:
            self._current = key if key in STEP_KEYS else ""
            if self._current:
                self._focus_key = self._current
            self.update()

    def current(self) -> str:
        return self._current

    def set_done(self, key: str, done: bool) -> None:
        if self._done.get(key) != done:
            self._done[key] = done
            self.update()

    def set_states(self, states: dict[str, bool]) -> None:
        """Completion per step (missing keys are left as they are)."""
        changed = False
        for k, v in states.items():
            if k in self._done and self._done[k] != bool(v):
                self._done[k] = bool(v)
                changed = True
        if changed:
            self.update()

    def is_done(self, key: str) -> bool:
        return self._done.get(key, False)

    def set_step_enabled(self, key: str, enabled: bool) -> None:
        if self._enabled.get(key) != enabled:
            self._enabled[key] = enabled
            self.update()

    def set_all_enabled(self, enabled: bool) -> None:
        for k in STEP_KEYS:
            self._enabled[k] = enabled
        self.update()

    def is_step_enabled(self, key: str) -> bool:
        return self._enabled.get(key, False)

    def set_context(self, text: str, tag: str = "") -> None:
        """Plain text at the right end (e.g. "NACA 2412  ›  CFD · α 4°"), after an optional
        short ``tag`` ("3D", "Wing") drawn in the secondary text colour."""
        if text != self._context or tag != self._tag:
            self._context, self._tag = text, tag
            self.setToolTip(self.full_context())
            self.update()

    def context(self) -> str:
        return self._context

    def tag(self) -> str:
        return self._tag

    def full_context(self) -> str:
        """The breadcrumb as read aloud / in the tooltip: "3D · Ahmed body  ›  CFD 3D · 30 m/s"."""
        return f"{self._tag} · {self._context}" if self._tag and self._context else self._context

    def step_rect(self, key: str) -> QRectF:
        """Geometry of one tab (for tests and tooltips)."""
        self._layout()
        return QRectF(self._rects.get(key, QRectF()))

    # -- geometry ---------------------------------------------------------------
    def _font(self, key: str) -> QFont:
        f = theme.ui_font()
        if key == self._current:
            f.setWeight(QFont.Weight.DemiBold)
        return f

    def _layout(self) -> None:
        # Width per tab is computed with the semibold font so the row does not
        # shift when the current step changes.
        bold = theme.ui_font()
        bold.setWeight(QFont.Weight.DemiBold)
        fm = QFontMetricsF(bold)
        x = 8.0
        self._rects = {}
        for key, title in STEPS:
            w = fm.horizontalAdvance(title) + 2 * _PAD + (_CHECK + 5 if self._done.get(key) else 0)
            self._rects[key] = QRectF(x, 0, w, _HEIGHT)
            x += w

    def sizeHint(self) -> QSize:
        self._layout()
        last = self._rects[STEP_KEYS[-1]]
        return QSize(int(last.right()) + 200, _HEIGHT)

    def minimumSizeHint(self) -> QSize:
        self._layout()
        return QSize(int(self._rects[STEP_KEYS[-1]].right()) + 8, _HEIGHT)

    def _key_at(self, pos: QPointF) -> str:
        for k, r in self._rects.items():
            if r.contains(pos):
                return k
        return ""

    # -- events -----------------------------------------------------------------
    def mouseMoveEvent(self, e) -> None:
        k = self._key_at(e.position())
        if k != self._hover:
            self._hover = k
            self.setCursor(Qt.CursorShape.PointingHandCursor if k and self._enabled.get(k)
                           else Qt.CursorShape.ArrowCursor)
            idx = STEP_KEYS.index(k) + 1 if k else 0
            self.setToolTip(f"{dict(STEPS)[k]} (Ctrl+{idx})" if k else self.full_context())
            self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:
        self._hover = ""
        self.update()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            k = self._key_at(e.position())
            if k and self._enabled.get(k):
                self._focus_key = k
                self.stepRequested.emit(k)
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e) -> None:
        keys = [k for k in STEP_KEYS if self._enabled.get(k)]
        if not keys:
            super().keyPressEvent(e)
            return
        i = keys.index(self._focus_key) if self._focus_key in keys else 0
        if e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Left):
            i = (i + (1 if e.key() == Qt.Key.Key_Right else -1)) % len(keys)
            self._focus_key = keys[i]
            self.update()
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.stepRequested.emit(keys[i])
        else:
            super().keyPressEvent(e)

    def resizeEvent(self, e) -> None:
        self._layout()
        super().resizeEvent(e)

    def paintEvent(self, _e) -> None:
        self._layout()
        t = theme.tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), theme.qcolor("bg"))
        # base line across the whole bar
        p.setPen(QPen(theme.qcolor("border"), 1))
        p.drawLine(QPointF(0, self.height() - 0.5), QPointF(self.width(), self.height() - 0.5))
        dpr = self.devicePixelRatioF()
        for key, title in STEPS:
            r = self._rects[key]
            enabled = self._enabled.get(key, False)
            current = key == self._current
            if enabled and key == self._hover and not current:
                p.fillRect(r.adjusted(0, 3, 0, -3), theme.qcolor("hover"))
            colour = t["text"] if current else (t["text2"] if enabled else t["faint"])
            p.setFont(self._font(key))
            p.setPen(theme.qcolor(colour))
            text_r = QRectF(r.left() + _PAD, r.top(), r.width() - 2 * _PAD, r.height())
            if self._done.get(key):
                pm = theme.render_icon("check", _CHECK, "good" if enabled else "faint", dpr)
                p.drawPixmap(QPointF(text_r.left(), r.center().y() - _CHECK / 2), pm)
                text_r.setLeft(text_r.left() + _CHECK + 5)
            p.drawText(text_r, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title)
            if current:
                # ink, as the website marks its current page (v4; the v3 underline was the blue accent)
                p.fillRect(QRectF(r.left() + 2, r.bottom() - 2, r.width() - 4, 2), theme.qcolor("fg"))
            if self.hasFocus() and key == self._focus_key:
                theme.paint_focus_ring(p, r.adjusted(2, 4, -2, -4), 2)
        if self._context:
            self._paint_context(p)
        p.end()

    def _paint_context(self, p: QPainter) -> None:
        """Right-aligned breadcrumb; a tag sits in front of it in text2 semibold, separated by a
        thin rule rather than boxed (a boxed tag would be the decorative pill the guidelines ban)."""
        last = self._rects[STEP_KEYS[-1]].right() + 24
        avail = self.width() - last - 12
        if avail <= 60:
            return
        font = theme.ui_font()
        p.setFont(font)
        fm = p.fontMetrics()
        tag_w = 0.0
        if self._tag:
            bold = theme.ui_font()
            bold.setWeight(QFont.Weight.DemiBold)
            tag_w = QFontMetricsF(bold).horizontalAdvance(self._tag) + 17
        text = fm.elidedText(self._context, Qt.TextElideMode.ElideMiddle, int(max(avail - tag_w, 20)))
        text_w = fm.horizontalAdvance(text)
        right = self.width() - 12
        p.setPen(theme.qcolor("muted"))
        p.drawText(QRectF(right - text_w - 1, 0, text_w + 1, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, text)
        if self._tag:
            x = right - text_w - 9
            p.setPen(QPen(theme.qcolor("border_strong"), 1))
            p.drawLine(QPointF(x + 0.5, 9), QPointF(x + 0.5, self.height() - 9))
            bold = theme.ui_font()
            bold.setWeight(QFont.Weight.DemiBold)
            p.setFont(bold)
            p.setPen(theme.qcolor("text2"))
            w = tag_w - 17
            p.drawText(QRectF(x - 8 - w, 0, w + 1, self.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, self._tag)


__all__ = ["STEPS", "STEP_KEYS", "StepBar"]
