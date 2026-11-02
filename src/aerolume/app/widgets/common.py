"""Shared widgets that give every page the same, quiet engineering-tool look.

They carry no aerodynamics. Each one is styled only through the theme's style
hooks (see the table in :mod:`aerolume.app.theme`) or, when it paints by hand,
through :func:`aerolume.app.theme.tokens`, so a theme switch restyles them for
free. Follow ``docs/ui_guidelines.md``: 4 px grid, 24 px controls, labels are
nouns, buttons are verbs, explanations go in tooltips.

Building blocks for page authors
================================

* :class:`PropertyGrid` - **the inspector**. Label column (text2) on the left,
  editor on the right, optional unit after the editor, groups with a thin
  header line and a collapse arrow. Use it for every right-hand "Properties"
  panel::

      grid = PropertyGrid()
      flow = grid.add_group("Flow")
      flow.add_row("Speed", UnitLineEdit("speed", "m/s", 30.0), tooltip="Free-stream speed")
      flow.add_row("Angle of attack", spin, unit="°")
      solver = grid.add_group("Solver", expanded=False)
      ...
      grid.add_stretch()

* :class:`KeyValueList` - read-only "Estimate" / facts table (key in text2,
  value right-aligned, optional unit).
* :class:`MessageLine` - one planner note / warning as a plain line with a
  small info / warn / error glyph.
* :class:`ToolStrip` - a page-level tool strip (viewport toolbar): icon
  buttons, checkable toggles, separators, a stretch, arbitrary widgets.
* :class:`EmptyState` - a one-line message plus one button, for empty views.
* :class:`CollapsibleSection` - header that folds its content away.
* :class:`Card` - a flat panel (1 px border, 2 px radius); use sparingly.
* :class:`SectionHeader` - title + optional muted subtitle + right actions.
* :class:`KpiTile` - one value with caption and delta (normal weight, 11 pt).
* :class:`Badge` - status as small plain text with an optional 8 px glyph.
* :class:`SegmentedControl` - exclusive choice among a few options.
* :class:`UnitLineEdit` - number entry that understands "120 km/h", "55 mph".
* :class:`Switch` - an on/off switch for a panel header (Lumen's A/B toggle).
* :class:`Notice` - an inline notice on its soft status background.
* :class:`IconButton`, :class:`Spinner`, :class:`StatusDot`,
  :class:`ElidedLabel`, :class:`Divider`, :func:`label`, :func:`button`.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import ClassVar

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import theme

SPACING = 4      # the 4 px grid every margin and gap is a multiple of
CONTROL_H = theme.CONTROL_H


def label(text: str = "", role: str | None = None, *, wrap: bool = False,
          parent: QWidget | None = None) -> QLabel:
    """A QLabel with a theme ``role`` ("h2", "secondary", "muted", "mono", ...) already set."""
    lab = QLabel(text, parent)
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    if wrap:
        lab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return lab


def button(text: str, variant: str | None = None, icon: str | None = None,
           on_click: Callable[[], None] | None = None, parent: QWidget | None = None) -> QPushButton:
    """A 24 px QPushButton with a theme ``variant`` ("primary", "ghost", "danger", "link")
    and an optional 16 px line icon tinted to match the text."""
    b = QPushButton(text, parent)
    if variant:
        b.setProperty("variant", variant)
    if icon:
        # links are ink like the website's text links, so their icon is too (v4; it was the blue accent)
        colour = {"primary": "on_accent", "danger": "bad", "link": "text"}.get(variant or "", "text2")
        b.setIcon(theme.icon(icon, colour))
        b.setIconSize(QSize(16, 16))
    if on_click is not None:
        b.clicked.connect(lambda _=False: on_click())
    return b


class Divider(QFrame):
    """A 1 px line in the border colour (horizontal by default)."""

    def __init__(self, vertical: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("divider", "vertical" if vertical else "true")


class Card(QFrame):
    """A flat panel (``QFrame[card="true"]``: panel fill, 1 px border) with a vertical layout.

    Prefer plain layouts, a :class:`PropertyGrid` or splitters; cards are for the
    rare block that really is a separate object. ``interactive=True`` adds a
    hover border and the ``clicked`` signal.
    """

    clicked = Signal()

    def __init__(self, parent: QWidget | None = None, *, interactive: bool = False,
                 padding: int = 3 * SPACING, spacing: int = 2 * SPACING) -> None:
        super().__init__(parent)
        self.setProperty("card", "true")
        if interactive:
            self.setProperty("interactive", "true")
            self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._interactive = interactive
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)

    def addWidget(self, w: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(w, stretch)
        return w

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._interactive and e.button() == Qt.MouseButton.LeftButton and self.rect().contains(
                e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e) -> None:
        if self._interactive and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(e)

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        if self._interactive and self.hasFocus():
            p = QPainter(self)
            theme.paint_focus_ring(p, QRectF(self.rect()), theme.RADIUS_LG)


class SectionHeader(QWidget):
    """Title (h2 / h3) with an optional muted subtitle and right-aligned action widgets."""

    def __init__(self, title: str, subtitle: str | None = None, *, level: str = "h2",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2 * SPACING)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.title = label(title, level)
        col.addWidget(self.title)
        self.subtitle = label(subtitle or "", "muted", wrap=True)
        self.subtitle.setVisible(bool(subtitle))
        col.addWidget(self.subtitle)
        row.addLayout(col, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(SPACING)
        row.addLayout(self.actions)

    def add_action(self, w: QWidget) -> QWidget:
        self.actions.addWidget(w)
        return w

    def set_subtitle(self, text: str) -> None:
        self.subtitle.setText(text)
        self.subtitle.setVisible(bool(text))


class KpiTile(QFrame):
    """One value with its caption, unit and delta - normal weight, 11 pt, no tile art.

    ``set_delta(text, good)`` colours the delta green when ``good`` is True,
    red when False and muted when None (no judgement, e.g. a different design).
    Results belong in tables; use this only for the one or two headline facts.
    """

    def __init__(self, label_text: str, value: str = "—", unit: str = "", *, hint: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(110)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.label = label(label_text, "kpi-label")
        if hint:
            self.label.setToolTip(hint)
            self.setToolTip(hint)
        lay.addWidget(self.label)
        row = QHBoxLayout()
        row.setSpacing(SPACING)
        self.value = label(value, "kpi-value")
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.unit = label(unit, "muted")
        row.addWidget(self.value, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addWidget(self.unit, 0, Qt.AlignmentFlag.AlignBaseline)
        row.addStretch(1)
        lay.addLayout(row)
        self.delta = label("", "small")
        self.delta.setVisible(False)
        lay.addWidget(self.delta)

    def set_value(self, value: str, unit: str | None = None) -> None:
        self.value.setText(value)
        if unit is not None:
            self.unit.setText(unit)

    def set_delta(self, text: str, good: bool | None = None) -> None:
        role = "small" if good is None else ("kpi-delta-good" if good else "kpi-delta-bad")
        theme.set_prop(self.delta, "role", role)
        self.delta.setText(text)
        self.delta.setVisible(bool(text))


class Badge(QLabel):
    """Status as small plain text in a status colour, with an optional 8 px glyph.

    ``kind``: "neutral", "ok", "warn", "bad" or "accent". With ``glyph=True`` a
    filled 8 px circle in the same colour precedes the text (the only status
    "chip" the guidelines allow - and only one per view).
    """

    _ROLES: ClassVar[dict[str, str]] = {"neutral": "badge", "ok": "badge-ok", "warn": "badge-warn",
                                        "bad": "badge-bad", "accent": "badge-accent"}
    _TOKENS: ClassVar[dict[str, str]] = {"neutral": "muted", "ok": "good", "warn": "warn", "bad": "bad",
                                         "accent": "accent"}

    def __init__(self, text: str = "", kind: str = "neutral", parent: QWidget | None = None, *,
                 glyph: bool = False) -> None:
        super().__init__(text, parent)
        self._kind, self._glyph = kind, glyph
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setProperty("role", self._ROLES.get(kind, "badge"))
        self._apply_indent()

    def _apply_indent(self) -> None:
        self.setIndent(14 if self._glyph else 0)
        self.setContentsMargins(0, 0, 0, 0)

    def set_kind(self, kind: str) -> None:
        self._kind = kind
        theme.set_prop(self, "role", self._ROLES.get(kind, "badge"))
        self.update()

    def set_glyph(self, on: bool) -> None:
        self._glyph = on
        self._apply_indent()
        self.update()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        if not self._glyph:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(self._TOKENS.get(self._kind, "muted")))
        p.drawEllipse(QRectF(1, self.height() / 2 - 4, 8, 8))


class SegmentedControl(QFrame):
    """Exclusive choice among a few options, drawn as joined flat buttons.

    ``options`` is a list of keys or of ``(key, label)`` pairs; ``changed(key)``
    fires on a user (or programmatic) change of the selection.
    """

    changed = Signal(str)

    def __init__(self, options: Sequence[str | tuple[str, str]], current: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("segmented", "true")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)      # the segments fill the frame; theme.paint_segmented_frame
        row.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for opt in options:
            key, text = (opt, opt) if isinstance(opt, str) else opt
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty("key", key)
            self._group.addButton(b)
            row.addWidget(b)
            self._buttons[key] = b
        self._group.buttonToggled.connect(self._toggled)
        first = current if current in self._buttons else next(iter(self._buttons), None)
        if first is not None:
            self._buttons[first].setChecked(True)

    def _toggled(self, b: QPushButton, checked: bool) -> None:
        if checked:
            self.changed.emit(b.property("key"))

    def paintEvent(self, _e) -> None:
        theme.paint_segmented_frame(self)

    def current(self) -> str:
        b = self._group.checkedButton()
        return b.property("key") if b is not None else ""

    def set_current(self, key: str) -> None:
        if key in self._buttons:
            self._buttons[key].setChecked(True)

    def keys(self) -> list[str]:
        return list(self._buttons)


class UnitLineEdit(QLineEdit):
    """Number entry with units, backed by :mod:`aerolume.core.units`.

    The user may type a bare number (read in the display ``unit``) or a number
    with any unit of the same kind ("120 km/h", "55 mph", "4.5 ft"); on Enter or
    focus-out the text is re-written in the display unit. Invalid or out-of-range
    input turns the border red and keeps the last good value, so a half-typed
    entry can never reach the model. ``valueChanged(float)`` emits SI values.
    """

    valueChanged = Signal(float)

    def __init__(self, kind: str, unit: str | None = None, value: float | None = None, *,
                 minimum: float | None = None, maximum: float | None = None, digits: int = 4,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from ...core import units as _units  # numpy-free, but keep the import local to the widget
        self._u = _units
        if kind not in _units.UNITS:
            raise ValueError(f"unknown quantity kind {kind!r}")
        self.kind = kind
        self._unit = unit or _units.DEFAULT_UNIT[kind]
        self._min, self._max, self._digits = minimum, maximum, digits
        self._value: float | None = None
        self.setPlaceholderText(self._unit)
        self._tip = f"Number in {self._unit}, or with a unit: " + ", ".join(_units.UNITS[kind])
        self.setToolTip(self._tip)
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.editingFinished.connect(self._commit)
        if value is not None:
            self.set_value(value)

    # -- public API -----------------------------------------------------------
    def value(self) -> float | None:
        """Current value in SI (None when empty and never set)."""
        return self._value

    def set_value(self, value_si: float | None, emit: bool = False) -> None:
        self._value = None if value_si is None else float(value_si)
        self._show()
        if emit and self._value is not None:
            self.valueChanged.emit(self._value)

    def unit(self) -> str:
        return self._unit

    def set_unit(self, unit: str) -> None:
        """Change the display unit (e.g. after a switch to imperial); the SI value is kept."""
        if unit not in self._u.UNITS[self.kind]:
            raise ValueError(f"unknown {self.kind} unit {unit!r}")
        self._unit = unit
        self.setPlaceholderText(unit)
        self._show()

    def is_valid(self) -> bool:
        return self.property("invalid") != "true"

    # -- internals ------------------------------------------------------------
    def _show(self) -> None:
        if self._value is None:
            self.setText("")
        else:
            v = self._u.convert(self._value, self.kind, self._unit)
            # no thousands separators: the text is parsed back on Enter / focus-out, and
            # "1,235 m" is ambiguous to units.parse (decimal comma or grouping?)
            self.setText(f"{self._u.sig(v, self._digits, grouping=False)} {self._unit}")
        self._set_invalid(None)

    def _set_invalid(self, message: str | None) -> None:
        theme.set_prop(self, "invalid", "true" if message else "false")
        self.setToolTip(message or self._tip)

    def _commit(self) -> None:
        text = self.text().strip()
        if not text:
            self._show()
            return
        try:
            v = self._u.parse(text, self.kind, self._unit)
        except (ValueError, KeyError) as exc:
            self._set_invalid(str(exc))
            return
        if self._min is not None and v < self._min:
            self._set_invalid(f"Must be at least {self._u.fmt(self._min, self.kind, self._unit)}")
            return
        if self._max is not None and v > self._max:
            self._set_invalid(f"Must be at most {self._u.fmt(self._max, self.kind, self._unit)}")
            return
        changed = self._value is None or not math.isclose(v, self._value, rel_tol=1e-12, abs_tol=0.0)
        self._value = v
        self._show()
        if changed:
            self.valueChanged.emit(v)


class EmptyState(QWidget):
    """A plain one-line message and one button, for views with nothing to show yet.

    ``title`` is the message ("No results"), ``text`` an optional short second
    line in muted type ("Run a simulation first."). Keep both factual. No
    illustration: the empty viewport is the illustration. ``actionTriggered``
    fires when the button is pressed. ``icon`` is accepted for compatibility
    and ignored.
    """

    actionTriggered = Signal()

    def __init__(self, icon: str = "info", title: str = "", text: str = "", action_text: str | None = None,
                 on_action: Callable[[], None] | None = None, *, secondary_text: str | None = None,
                 on_secondary: Callable[[], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        del icon
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6 * SPACING, 6 * SPACING, 6 * SPACING, 6 * SPACING)
        outer.addStretch(1)
        self.title = label(title, "secondary")
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(self.title)
        self.text = label(text, "muted", wrap=True)
        self.text.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.text.setVisible(bool(text))
        outer.addWidget(self.text)
        outer.addSpacing(2 * SPACING)
        row = QHBoxLayout()
        row.setSpacing(2 * SPACING)
        row.addStretch(1)
        self.button = button(action_text or "")
        self.button.setVisible(bool(action_text))
        self.button.clicked.connect(lambda _=False: self.actionTriggered.emit())
        if on_action:
            self.actionTriggered.connect(on_action)
        row.addWidget(self.button)
        self.secondary = button(secondary_text or "", "ghost")
        self.secondary.setVisible(bool(secondary_text))
        if on_secondary:
            self.secondary.clicked.connect(lambda _=False: on_secondary())
        row.addWidget(self.secondary)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)     # slightly above centre

    def set_text(self, title: str | None = None, text: str | None = None) -> None:
        if title is not None:
            self.title.setText(title)
        if text is not None:
            self.text.setText(text)
            self.text.setVisible(bool(text))

    def set_action(self, text: str | None, on_action: Callable[[], None] | None = None) -> None:
        """Change (or hide, with ``None``) the button."""
        self.button.setText(text or "")
        self.button.setVisible(bool(text))
        if on_action is not None:
            try:
                self.actionTriggered.disconnect()
            except (RuntimeError, TypeError):
                pass
            self.actionTriggered.connect(on_action)


class _GroupHeader(QToolButton):
    """Collapse arrow + semibold title + a thin rule to the right edge.

    The header is borderless text, so the style sheet's focus rule (a border that changes
    colour) has nothing to colour: the v3.3 keyboard audit found a focused "Heat transfer"
    header identical to an unfocused one (WCAG 2.4.7). It paints its own 2 px ring in the
    ``focus`` token around the arrow and the title, and the rule starts after the ring.
    """

    #: arrow (12 px) + gap (6 px) before the title; the ring's padding around arrow and title
    _TEXT_X = 12 + 6
    _RING_PAD = 4

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setText(title)
        self.setAccessibleDescription("Section heading: Space or Enter shows or hides its fields")
        self.setCheckable(True)
        self.setProperty("role", "section")
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setIconSize(QSize(12, 12))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(CONTROL_H)
        self.toggled.connect(self._sync_icon)
        self._sync_icon(False)

    def setText(self, text: str) -> None:
        """The title, and the same accessible name (pages retitle groups: "Queue (2)")."""
        super().setText(text)
        self.setAccessibleName(text)

    def _sync_icon(self, expanded: bool) -> None:
        self.setIcon(theme.icon("chevron-down" if expanded else "chevron-right", "text2"))

    def focus_rect(self) -> QRectF:
        """Where the focus ring goes: around the arrow and the title, the full height."""
        fm = QFontMetrics(self.font())
        right = self._TEXT_X + fm.horizontalAdvance(self.text()) + self._RING_PAD
        return QRectF(0, 0, min(float(right), float(self.width())), float(self.height()))

    def focusInEvent(self, e) -> None:
        super().focusInEvent(e)
        self.update()            # the ring is painted here, not by the style: repaint on the change

    def focusOutEvent(self, e) -> None:
        super().focusOutEvent(e)
        self.update()

    def paintEvent(self, e) -> None:
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        ring = self.focus_rect()
        if self.hasFocus():
            theme.paint_focus_ring(p, ring, width=2.0)
        x = ring.right() + 6
        if x < self.width():
            p.setPen(QPen(theme.qcolor("border"), 1))
            y = self.height() / 2 + 0.5
            p.drawLine(QPointF(x, y), QPointF(self.width(), y))


class CollapsibleSection(QWidget):
    """A header row (arrow + title + rule) that shows or hides its content.

    Used for "Advanced" groups so the default view stays short while every
    knob is one click away. ``toggled(bool)`` reports the new expanded state.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget | None = None, *, expanded: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)     # never spread the content out when stretched
        self.header = _GroupHeader(title)
        lay.addWidget(self.header)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, SPACING)
        self.content_layout.setSpacing(SPACING)
        lay.addWidget(self.content)
        if content is not None:
            self.content_layout.addWidget(content)
        self.header.toggled.connect(self._apply)
        self.header.setChecked(expanded)
        self._apply(expanded)

    def _apply(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggled.emit(expanded)

    def set_content(self, w: QWidget) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        self.content_layout.addWidget(w)

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)


# ---------------------------------------------------------------------------
# property grid (the inspector)
# ---------------------------------------------------------------------------

#: object-name prefix of the line edit Qt puts inside a spin box (part of the spin box, never named)
_SPIN_EDIT = "qt_spinbox_lineedit"
_AUTO_NAME = "aerolumeAutoName"


def _focusables(editor: QWidget) -> list[QWidget]:
    """``editor`` and its descendants that take keyboard focus, in creation order."""
    out = [editor] if editor.focusPolicy() != Qt.FocusPolicy.NoFocus else []
    for w in editor.findChildren(QWidget):
        if w.focusPolicy() != Qt.FocusPolicy.NoFocus and not w.objectName().startswith(_SPIN_EDIT):
            out.append(w)
    return out


def _rich(text: str) -> bool:
    """Qt's own rich-text guess (``Qt::mightBeRichText`` lives in QtGui's half of the namespace)."""
    from PySide6 import QtGui
    return bool(QtGui.Qt.mightBeRichText(text))


def _own_name(w: QWidget) -> str:
    """The name Qt's accessibility gives ``w`` by itself (a button's text; "" for an input)."""
    from PySide6.QtGui import QAccessible
    acc = QAccessible.queryAccessibleInterface(w)
    return (acc.text(QAccessible.Text.Name) or "").strip() if acc is not None else ""


def tie_label(lab: QLabel, editor: QWidget, unit: str | None = None) -> QWidget | None:
    """Make ``lab`` the visible label of ``editor`` for screen readers (WCAG 1.3.1 / 4.1.2).

    Qt names a field after a label only through ``QLabel.setBuddy`` - a label
    merely placed beside it (a grid row, or ``QFormLayout.addRow`` with a
    QLabel object) leaves the field nameless, measured with
    ``QAccessible.queryAccessibleInterface(field).text(Name)`` == "". The buddy
    is the first focusable widget of ``editor`` (a composite editor's spin
    box); the editor's other focusable parts (a unit selector) get
    "<label>: <their tooltip>" so they are not announced as bare "combo box".
    Names a page set itself are kept; names set here follow
    :meth:`PropertyRow.set_label`. Returns the buddy (None if nothing takes focus).
    """
    parts = _focusables(editor)
    if not parts:
        return None
    text = lab.text()
    if _rich(text):           # "C<sub>d</sub>" is read as "Cd", not as markup
        from PySide6.QtGui import QTextDocumentFragment
        text = QTextDocumentFragment.fromHtml(text).toPlainText()
    text = " ".join(text.split())
    # what each part is called by itself, read before the buddy below lends it the label's text
    own = [w.accessibleName() or _own_name(w) for w in parts]
    # a buddy turns "&" in a plain label into a mnemonic ("Lift & drag" would lose its "&"); such a
    # label keeps no buddy and its field is named directly below
    if _rich(lab.text()) or "&" not in lab.text():
        lab.setBuddy(parts[0])
    if editor not in parts and (not editor.accessibleName() or editor.property(_AUTO_NAME)):
        editor.setAccessibleName(text)                 # the group a segmented control / composite is
        editor.setProperty(_AUTO_NAME, True)
    for i, w in enumerate(parts):
        if w.property(_AUTO_NAME):
            pass                                      # ours: follows the label
        elif own[i]:
            continue                                  # a page's own name, or a button's text
        tip = w.toolTip().split("\n")[0].strip()
        name = text if i == 0 or not tip else f"{text}: {tip}"
        w.setAccessibleName(name)
        w.setProperty(_AUTO_NAME, True)
        if i == 0 and unit and not w.accessibleDescription():
            w.setAccessibleDescription(f"in {unit}")
    return parts[0]


def tie_form_labels(root: QWidget) -> int:
    """Name the fields of every ``QFormLayout`` under ``root`` after their row labels; returns the count.

    A page that builds its form with ``addRow(QLabel(...), field)`` leaves
    the fields nameless for a screen reader (see :func:`tie_label`): the
    automated pass over the shell measured nine unnamed controls on the
    Preferences ▸ Assistant tab alone. The window runs this over pages it
    hosts but does not own, so the fix does not wait on each page. A field
    that is a layout (an input with a button beside it) lends the label to
    its first focusable widget; rows whose label is not a :class:`QLabel`
    (string labels, which Qt already reads) and fields that already have a
    name of their own are left alone.
    """
    from PySide6.QtWidgets import QFormLayout

    def widgets_of(item) -> list[QWidget]:
        if item is None:
            return []
        if item.widget() is not None:
            return [item.widget()]
        lay = item.layout()
        out: list[QWidget] = []
        if lay is not None:
            for i in range(lay.count()):
                out.extend(widgets_of(lay.itemAt(i)))
        return out

    n = 0
    for form in root.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            lab_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            lab = lab_item.widget() if lab_item is not None else None
            if not isinstance(lab, QLabel) or not lab.text().strip():
                continue
            fields = [w for w in widgets_of(form.itemAt(row, QFormLayout.ItemRole.FieldRole)) if _focusables(w)]
            if not fields:
                continue
            first = fields[0]
            # _own_name: Qt already reads a name (a buddy label, as string-label rows get from Qt)
            unnamed = (not first.accessibleName() and not _own_name(first)) or first.property(_AUTO_NAME)
            if unnamed and tie_label(lab, first) is not None:
                n += 1
            text = " ".join(lab.text().replace("&", "").split())
            for w in fields[1:]:
                if not w.accessibleName() and not _own_name(w):
                    tip = w.toolTip().split("\n")[0].strip()
                    w.setAccessibleName(f"{text}: {tip}" if tip else text)
                    w.setProperty(_AUTO_NAME, True)
                    n += 1
    return n


class PropertyRow:
    """One label / editor / unit row of a :class:`PropertyGroup` (returned by ``add_row``)."""

    def __init__(self, label_widget: QLabel, editor: QWidget, unit: QLabel | None) -> None:
        self.label, self.editor, self.unit = label_widget, editor, unit

    def set_visible(self, visible: bool) -> None:
        for w in (self.label, self.editor, self.unit):
            if w is not None:
                w.setVisible(visible)

    def set_enabled(self, enabled: bool) -> None:
        for w in (self.label, self.editor, self.unit):
            if w is not None:
                w.setEnabled(enabled)

    def set_label(self, text: str) -> None:
        self.label.setText(text)
        tie_label(self.label, self.editor, self.unit.text() if self.unit is not None else None)


class PropertyGroup(QWidget):
    """A collapsible group of property rows inside a :class:`PropertyGrid`.

    Rows sit on a three-column grid: label (text2, fixed width shared by the
    whole grid), editor (stretches), unit (muted, optional). Full-width widgets
    (a plot, a note, a button row) go in with :meth:`add_widget`.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, *, expanded: bool = True, label_width: int = 128,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = _GroupHeader(title)
        lay.addWidget(self.header)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(16, SPACING, 0, 2 * SPACING)
        self.grid.setHorizontalSpacing(2 * SPACING)
        self.grid.setVerticalSpacing(SPACING)
        self.grid.setColumnMinimumWidth(0, label_width)
        self.grid.setColumnStretch(1, 1)
        lay.addWidget(self.body)
        self.rows: dict[str, PropertyRow] = {}
        self._next = 0
        self.header.toggled.connect(self._apply)
        self.header.setChecked(expanded)
        self._apply(expanded)

    @property
    def title(self) -> str:
        return self.header.text()

    def _apply(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.toggled.emit(expanded)

    def set_label_width(self, width: int) -> None:
        self.grid.setColumnMinimumWidth(0, width)

    def add_row(self, text: str, editor: QWidget, *, unit: str | None = None, tooltip: str | None = None,
                key: str | None = None) -> PropertyRow:
        """Add ``text`` | ``editor`` | ``unit``; the tooltip goes on label and editor."""
        lab = label(text, "pg-label")
        lab.setMinimumHeight(CONTROL_H)
        lab.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        unit_lab = None
        r = self._next
        self.grid.addWidget(lab, r, 0)
        if unit is not None:
            unit_lab = label(unit, "muted")
            self.grid.addWidget(editor, r, 1)
            self.grid.addWidget(unit_lab, r, 2)
        else:
            self.grid.addWidget(editor, r, 1, 1, 2)
        if tooltip:
            lab.setToolTip(tooltip)
            if not editor.toolTip():
                editor.setToolTip(tooltip)
        tie_label(lab, editor, unit)
        self._next += 1
        row = PropertyRow(lab, editor, unit_lab)
        self.rows[key or text] = row
        return row

    def add_widget(self, w: QWidget) -> QWidget:
        """A full-width widget spanning label, editor and unit columns."""
        self.grid.addWidget(w, self._next, 0, 1, 3)
        self._next += 1
        return w

    def row(self, key: str) -> PropertyRow:
        return self.rows[key]

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)


class PropertyGrid(QWidget):
    """The right-hand inspector: stacked :class:`PropertyGroup` s with a shared label column.

    Put it in a QScrollArea (``grid.scrollable()`` does that) on the panel
    colour. Group expansion is a user preference, so pages may persist
    ``expanded_state()`` and give it back with ``restore_expanded()``.
    """

    def __init__(self, *, label_width: int = 128, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label_width = label_width
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(3 * SPACING, 2 * SPACING, 3 * SPACING, 2 * SPACING)
        self.layout_.setSpacing(0)
        self.groups: dict[str, PropertyGroup] = {}

    def add_group(self, title: str, *, expanded: bool = True, key: str | None = None) -> PropertyGroup:
        g = PropertyGroup(title, expanded=expanded, label_width=self._label_width)
        self.layout_.addWidget(g)
        self.groups[key or title] = g
        return g

    def add_widget(self, w: QWidget) -> QWidget:
        self.layout_.addWidget(w)
        return w

    def add_stretch(self) -> None:
        self.layout_.addStretch(1)

    def group(self, key: str) -> PropertyGroup:
        return self.groups[key]

    def set_label_width(self, width: int) -> None:
        self._label_width = width
        for g in self.groups.values():
            g.set_label_width(width)

    def expanded_state(self) -> dict[str, bool]:
        return {k: g.is_expanded() for k, g in self.groups.items()}

    def restore_expanded(self, state: dict[str, bool]) -> None:
        for k, v in state.items():
            if k in self.groups:
                self.groups[k].set_expanded(bool(v))

    def scrollable(self) -> QWidget:
        """This grid inside a frameless vertical scroll area on the panel colour."""
        from PySide6.QtWidgets import QScrollArea

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc.setFocusPolicy(Qt.FocusPolicy.NoFocus)       # the fields inside take focus (see unfocus_scroll_areas)
        sc.setWidget(self)
        return sc


def unfocus_scroll_areas(root: QWidget) -> int:
    """Take plain scroll areas under ``root`` out of the Tab chain when their content takes focus.

    A ``QScrollArea`` is a focus stop of its own by default (Qt gives it StrongFocus for
    keyboard scrolling), with no accessible name: the v3.3 keyboard audit counted four such
    nameless stops on the Setup page, which a screen reader announces as nothing (WCAG
    4.1.2). When the content has focusable widgets, Tab reaches them anyway and the area
    scrolls each into view (``QScrollArea.focusNextPrevChild``), so the area itself is set
    to NoFocus; so is an area whose content fits (there is nothing to scroll). An area whose
    content takes no focus and overflows (read-only text) keeps its stop so the keyboard can
    still scroll it, named after its first heading when it has no name. Hidden areas are
    skipped (not laid out yet, they are judged when their page shows). Item views (trees,
    tables, text editors) are scroll areas too but not ``QScrollArea``, and are left alone.
    Returns how many areas were changed.
    """
    from PySide6.QtWidgets import QScrollArea

    changed = 0
    for sc in root.findChildren(QScrollArea):
        if sc.focusPolicy() == Qt.FocusPolicy.NoFocus or not sc.isVisible():
            continue
        inner = sc.widget()
        focusable = inner is not None and any(
            w.focusPolicy() & Qt.FocusPolicy.TabFocus for w in inner.findChildren(QWidget))
        fits = sc.verticalScrollBar().maximum() == 0 and sc.horizontalScrollBar().maximum() == 0
        if focusable or fits:
            sc.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            changed += 1
        elif sc.property("scrollstop") != "true":
            # it stays a stop: it needs a visible focus mark (a frameless area has no border to
            # colour, so the style sheet gives such stops a transparent one) and a name
            theme.set_prop(sc, "scrollstop", "true")
            if not sc.accessibleName():
                head = _heading_for(sc)
                sc.setAccessibleName(f"{head} (scrollable)" if head else "Scrollable content")
            changed += 1
    return changed


def name_unnamed_views(root: QWidget) -> int:
    """Give visible tables and tab bars under ``root`` that have no accessible name the heading above them.

    Results tables and the Results tab bar are Tab stops, and the v3.3 keyboard audit counted about
    forty nameless stops on the heated cylinder's Results page: a screen reader announced "table"
    with nothing to tell the Summary from the Heat balance (WCAG 4.1.2). A table under a heading
    gets that heading's text, a tab bar "<current tab> tabs". Returns how many were named."""
    from PySide6.QtWidgets import QTabBar, QTableView

    named = 0
    for view in root.findChildren(QTableView):
        if view.isVisible() and not view.accessibleName():
            head = _heading_for(view)
            if head:
                view.setAccessibleName(head)
                named += 1
    for bar in root.findChildren(QTabBar):
        if bar.isVisible() and not bar.accessibleName() and bar.count():
            bar.setAccessibleName("Tabs: " + ", ".join(bar.tabText(i) for i in range(min(bar.count(), 8))))
            named += 1
    return named


def _heading_for(area: QWidget) -> str:
    """The heading a scroll area sits under: its own first h1-h3 label, else the nearest one above it
    among its parent's labels (pages put the heading beside the area, e.g. Setup's "Estimate")."""
    roles = ("h1", "h2", "h3")
    inner = area.widget() if hasattr(area, "widget") else None
    for lab in (inner.findChildren(QLabel) if inner is not None else []):
        if lab.property("role") in roles and lab.text().strip():
            return lab.text().strip()
    parent = area.parentWidget()
    if parent is None:
        return ""
    top = area.mapTo(parent, area.rect().topLeft()).y()
    best, best_y = "", None
    for lab in parent.findChildren(QLabel):
        if lab.property("role") not in roles or not lab.text().strip() or area.isAncestorOf(lab):
            continue
        y = lab.mapTo(parent, lab.rect().topLeft()).y()
        if y <= top and (best_y is None or y > best_y):
            best, best_y = lab.text().strip(), y
    return best


class KeyValueList(QWidget):
    """Read-only facts as two columns: key (text2) and value (right-aligned), e.g. "Estimate".

    ``set_items([(key, value), ...])`` or ``[(key, value, tooltip), ...]``
    replaces the content. Values are selectable so they can be copied.
    """

    def __init__(self, items: Sequence[tuple] = (), *, mono_values: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mono = mono_values
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(3 * SPACING)
        self.grid.setVerticalSpacing(SPACING)
        self.grid.setColumnStretch(1, 1)
        self._rows: list[tuple[QLabel, QLabel]] = []
        self.set_items(items)

    def set_items(self, items: Sequence[tuple]) -> None:
        """Replace every row at once.

        The old labels are hidden and detached *before* ``deleteLater``: a
        deferred delete only runs when control returns to the event loop, and
        under a nested ``processEvents`` (tests, modal dialogs) the stale rows
        would otherwise keep painting underneath the new ones.
        """
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._rows = []
        for r, item in enumerate(items):
            key, value = item[0], item[1]
            tip = item[2] if len(item) > 2 else None
            k = label(str(key), "secondary")
            v = label(str(value), "mono" if self._mono else None)
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if tip:
                k.setToolTip(tip)
                v.setToolTip(tip)
            self.grid.addWidget(k, r, 0)
            self.grid.addWidget(v, r, 1)
            self._rows.append((k, v))

    def items(self) -> list[tuple[str, str]]:
        """The (key, value) texts currently shown, top to bottom."""
        return [(k.text(), v.text()) for k, v in self._rows]

    def value_text(self, key: str) -> str | None:
        # Reads the live rows, not the grid: QGridLayout.rowCount() never shrinks, so scanning it
        # would visit rows of earlier, longer item lists.
        for k, v in self._rows:
            if k.text() == key:
                return v.text()
        return None


_LEVEL_ICON = {"info": ("info", "text2"), "warning": ("warning", "warn"), "warn": ("warning", "warn"),
               "error": ("error", "bad"), "ok": ("check-circle", "good")}


class MessageLine(QWidget):
    """One planner note / warning / error as a plain line with a small leading glyph."""

    def __init__(self, text: str, level: str = "info", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2 * SPACING)
        self.glyph = QLabel()
        self.glyph.setFixedSize(16, 16)
        self.glyph.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.glyph, 0, Qt.AlignmentFlag.AlignTop)
        self.text = label(text, None, wrap=True)
        lay.addWidget(self.text, 1)
        self._level = level
        self.set_level(level)
        theme.changed.connect(self._retint)  # bound method: auto-disconnects when deleted

    def _retint(self, _name: str = "") -> None:
        self.set_level(self._level)

    def set_level(self, level: str) -> None:
        self._level = level
        name, colour = _LEVEL_ICON.get(level, _LEVEL_ICON["info"])
        self.glyph.setPixmap(theme.render_icon(name, 14, colour, self.devicePixelRatioF()))

    def set_text(self, text: str) -> None:
        self.text.setText(text)


class ToolStrip(QFrame):
    """A page-level tool strip (``QFrame[chrome="toolbar"]``) for viewport controls.

    ``add_button(icon, text, tooltip, checkable=False)`` returns a QToolButton
    (icon-only when ``text`` is empty); ``add_separator()``, ``add_stretch()``
    and ``add_widget(w)`` complete it. Height 30 px, buttons 24 px.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("chrome", "toolbar")
        self.setFixedHeight(CONTROL_H + 2 * SPACING + 1)
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        self.row.setSpacing(2)

    def add_button(self, icon: str | None, text: str = "", tooltip: str = "", *, checkable: bool = False,
                   on_click: Callable[[], None] | None = None) -> QToolButton:
        b = QToolButton()
        if icon:
            b.setIcon(theme.icon(icon, "text2", active="accent"))
            b.setIconSize(QSize(16, 16))
        if text:
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon if icon
                                 else Qt.ToolButtonStyle.ToolButtonTextOnly)
        b.setToolTip(tooltip or text)
        # an explicit name for assistive technology: an icon-only button has no text to fall back on,
        # and the v3.4 focus probe found seven nameless stops on Setup's view strip
        b.setAccessibleName(text or (tooltip.split(" (")[0] if tooltip else ""))
        b.setCheckable(checkable)
        b.setFixedHeight(CONTROL_H)
        if on_click is not None:
            b.clicked.connect(lambda _=False: on_click())
        self.row.addWidget(b)
        return b

    def add_separator(self) -> QFrame:
        d = Divider(vertical=True)
        d.setFixedHeight(16)
        self.row.addSpacing(SPACING)
        self.row.addWidget(d)
        self.row.addSpacing(SPACING)
        return d

    def add_widget(self, w: QWidget) -> QWidget:
        self.row.addWidget(w)
        return w

    def add_stretch(self) -> None:
        self.row.addStretch(1)


class IconButton(QToolButton):
    """Square 24 px icon-only button with a tooltip (always give one: it is the only label).

    ``size`` is never below :data:`theme.MIN_TARGET` (24 px, WCAG 2.2 SC 2.5.8): panel
    headers asked for 22 px and the Messages bar for 20, which the v3.3 audit counted among
    twelve undersized targets. A smaller ``icon_size`` keeps a compact glyph; the padding
    around it makes up the hit area.
    """

    def __init__(self, icon: str, tooltip: str, *, size: int = CONTROL_H, icon_size: int = 16,
                 checkable: bool = False, colour: str = "text2", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("variant", "icon")
        self.setIcon(theme.icon(icon, colour, active="accent" if checkable else None))
        self.setIconSize(QSize(icon_size, icon_size))
        size = max(int(size), theme.MIN_TARGET)
        self.setFixedSize(size, size)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setCheckable(checkable)
        self.setAutoRaise(True)

    def set_icon(self, icon: str, colour: str = "text2") -> None:
        self.setIcon(theme.icon(icon, colour, active="accent" if self.isCheckable() else None))


class Spinner(QWidget):
    """A small indeterminate progress arc (loading states). Animates only while visible."""

    def __init__(self, size: int = 14, colour: str = "accent", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour, self._angle = colour, 0
        self.setFixedSize(size, size)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def showEvent(self, e) -> None:
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self._timer.stop()
        super().hideEvent(e)

    def _tick(self) -> None:
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        p.setPen(QPen(theme.qcolor("border"), 1.5))
        p.drawEllipse(r)
        pen = QPen(theme.qcolor(self._colour), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen)
        p.drawArc(r, -self._angle * 16, 90 * 16)


class StatusDot(QWidget):
    """A small filled circle in a token colour (simulation status)."""

    def __init__(self, colour: str = "muted", size: int = 8, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour = colour
        self.setFixedSize(size, size)

    def set_colour(self, colour: str) -> None:
        self._colour = colour
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.qcolor(self._colour)))
        p.drawEllipse(self.rect())


class Switch(QAbstractButton):
    """An on/off switch (28 x 16 track): the Lumen panel-header switch for A/B toggling an effect.

    Use it where flipping a whole group on and off keeps its values (a
    panel's effect, recording diagnostics); a check box is still right for a
    setting that is part of a form. It is a checkable button, so ``toggled``,
    ``setChecked`` and keyboard Space work as usual and screen readers hear a
    toggle button with its accessible name and state. Off: an outlined track
    in the ``control`` colour (3:1 on every surface) with a ``control``
    knob; on: a ``primary`` track with a ``primary_fg`` knob at the right.
    The state never relies on colour alone: the knob changes sides.
    """

    TRACK_W, TRACK_H = 28, 16

    def __init__(self, checked: bool = True, tooltip: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if tooltip:
            self.setToolTip(tooltip)
            self.setAccessibleName(tooltip)
        theme.changed.connect(self._on_theme)     # bound method: auto-disconnects when deleted

    def _on_theme(self, _name: str = "") -> None:
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.TRACK_W + 4, max(self.TRACK_H + 4, 20))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:          # the whole widget is the target, not just the track
        return self.rect().contains(pos)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.TRACK_W, self.TRACK_H
        r = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        on, enabled = self.isChecked(), self.isEnabled()
        track = theme.qcolor("primary" if on else "surface")
        edge = theme.qcolor("primary" if on else ("control_hover" if self.underMouse() else "control"))
        if not enabled:
            track = theme.qcolor("line_strong" if on else "surface_2")
            edge = theme.qcolor("line_strong")
        p.setPen(QPen(edge, 1.0))
        p.setBrush(track)
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), h / 2, h / 2)
        d = h - 6
        x = r.right() - 3 - d if on else r.left() + 3
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor("primary_fg" if on and enabled else ("faint" if not enabled else "control")))
        p.drawEllipse(QRectF(x, r.top() + 3, d, d))
        if self.hasFocus():
            theme.paint_focus_ring(p, r.adjusted(-2, -2, 2, 2), h / 2 + 2)


class Notice(QFrame):
    """An inline notice on its soft status background: an optional title and one or two lines.

    ``kind`` "info" | "good" | "warn" | "bad" picks ``QFrame[notice=...]``:
    the ``*_soft`` background with a 3 px rule in the status colour on the
    left, body text in ``fg`` (>= 4.5:1 on every soft background, measured by
    tests/test_tokens.py). For facts the user should not miss beside a form
    (accounts not configured, a crash report saved); never for decoration.
    """

    def __init__(self, text: str = "", kind: str = "info", title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("notice", kind if kind in ("info", "good", "warn", "bad") else "info")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3 * SPACING, 2 * SPACING, 3 * SPACING, 2 * SPACING)
        lay.setSpacing(SPACING // 2)
        self.title = label(title, "notice-title")
        self.title.setVisible(bool(title))
        lay.addWidget(self.title)
        self.text = label(text, None, wrap=True)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.text)

    def set_kind(self, kind: str) -> None:
        theme.set_prop(self, "notice", kind)
        for w in (self.title, self.text):
            theme.repolish(w)

    def set_text(self, text: str, title: str | None = None) -> None:
        self.text.setText(text)
        if title is not None:
            self.title.setText(title)
            self.title.setVisible(bool(title))


class ElidedLabel(QLabel):
    """A label that shortens long text with an ellipsis instead of growing the layout
    (file paths, long project names). The full text is kept in the tooltip."""

    def __init__(self, text: str = "", role: str | None = None, *,
                 mode: Qt.TextElideMode = Qt.TextElideMode.ElideMiddle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full, self._mode = text, mode
        if role:
            self.setProperty("role", role)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def sizeHint(self) -> QSize:
        # ask for the full text width (capped by maximumWidth) so short text is never elided
        fm = QFontMetrics(self.font())
        base = super().sizeHint()
        return QSize(min(fm.horizontalAdvance(self._full) + 4, self.maximumWidth()), base.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(self.minimumWidth(), super().minimumSizeHint().height())

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._elide()

    def _elide(self) -> None:
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, self._mode, max(self.width(), 40)))


__all__ = [
    "CONTROL_H",
    "SPACING",
    "Badge",
    "Card",
    "CollapsibleSection",
    "Divider",
    "ElidedLabel",
    "EmptyState",
    "IconButton",
    "KeyValueList",
    "KpiTile",
    "MessageLine",
    "Notice",
    "PropertyGrid",
    "PropertyGroup",
    "PropertyRow",
    "SectionHeader",
    "SegmentedControl",
    "Spinner",
    "StatusDot",
    "Switch",
    "ToolStrip",
    "UnitLineEdit",
    "button",
    "label",
    "name_unnamed_views",
    "tie_form_labels",
    "tie_label",
    "unfocus_scroll_areas",
]
