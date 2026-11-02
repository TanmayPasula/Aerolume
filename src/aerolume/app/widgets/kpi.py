"""Number formatting and small read-only widgets for the runtime pages.

The runtime pages (Simulate, Results, Compare, Report) show numbers, not
decoration: results are tables with units and uncertainty, status is a word in
a status colour, and an empty view is one plain line plus one button
(docs/ui_guidelines.md sections 1, 2 and 6). This module holds the pieces they
share so the four pages format a Reynolds number, a standard error or a
duration identically:

* :func:`fmt_re`, :func:`fmt_err`, :func:`fmt_speed`, :func:`fmt_duration`,
  :func:`fmt_num`, :func:`fmt_fixed` - formatting with proper symbols
  (``8.22×10⁵``, ``± 0.00015``, ``3 min 24 s``, typographic minus).
* :func:`pretty`, :func:`headline`, :func:`subscripts` - solver text made fit
  for the screen: ``10^5`` -> ``10⁵``, a warning's first clause, C<sub>d</sub>.
* :class:`StatusLabel` - status as plain text in a status colour.
* :class:`WarningLine` - one warning line with the full text in its tooltip.
* :class:`DataTable` - the read-only results table: 22 px rows, quantity
  column in the UI font, numbers right-aligned in the mono font, optional
  per-cell tone (good / bad / muted) and tooltip; coefficient symbols in cells
  and headers are drawn with real subscripts.
* :class:`EmptyState` - the foundation's empty state plus the one-call
  ``set_content`` the pages use.

Everything is styled through the theme hooks (``role``, ``variant``) or
:func:`aerolume.app.theme.tokens`, never with literal colours.
"""
from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

from PySide6.QtCore import QPointF, QRect, QSize, Qt
from PySide6.QtGui import QAbstractTextDocumentLayout, QFont, QFontMetricsF, QPalette, QTextDocument
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionHeader,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

# the pure formatting lives Qt-free in core.format (the CLI, MCP server and API share it); re-exported here
from ...core.format import (
    _SUPERSCRIPT,
    MINUS,
    SCI_ABOVE,
    SCI_BELOW,
    STEADY_REL,
    STEADY_TEXT,
    STEADY_TIP,
    WITHIN_ERROR_TEXT,
    decimals_for,
    decimals_from_error,
    fmt_err,
    fmt_error,
    fmt_fixed,
    fmt_num,
    fmt_re,
    fmt_sci,
    fmt_sig,
    is_steady,
    needs_sci,
    round_error,
    stat_texts,
    value_error_texts,
    within_error,
)
from .. import theme
from . import common

#: Item-data role holding the rich-text (HTML) form of a cell or header label.
HTML_ROLE = int(Qt.ItemDataRole.UserRole) + 41
#: Item-data role: blank character cells to leave right of a number so its decimal point
#: lines up with the rest of its column (see :func:`decimal_tail`).
PAD_ROLE = int(Qt.ItemDataRole.UserRole) + 42


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def speed_units() -> tuple[str, str]:
    """(primary, secondary) speed units from the user's unit system: m/s + km/h, or mph + m/s."""
    try:
        from .. import settings
        system = settings.get().unit_system
    except Exception:  # noqa: BLE001 - preferences unavailable (tests, broken INI): SI
        system = "SI"
    return ("mph", "m/s") if system == "imperial" else ("m/s", "km/h")


def fmt_speed(v: float, *, both: bool = True, sep: str = " · ") -> str:
    """A speed in the preferred unit, optionally followed by a second one: '40.0 m/s · 144 km/h'."""
    from ...core.units import convert, sig
    a, b = speed_units()
    text = f"{sig(convert(v, 'speed', a), 3)} {a}"
    return f"{text}{sep}{sig(convert(v, 'speed', b), 3)} {b}" if both else text


def fmt_duration(seconds: float | None) -> str:
    """'45 s', '3 min 24 s', '1 h 05 min'; an em dash when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    s = round(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


_POW_RE = re.compile(r"10\^(-?\d+)")


def pretty(text: str) -> str:
    """Typographic clean-up for solver / planner text shown on screen.

    ``10^6`` -> ``10⁶``, `` -- `` -> `` — ``. The solver writes plain ASCII
    because the same strings go to logs, CSV and JSON files; the guidelines ask
    for proper symbols on screen (Re = 8.2×10⁵). The Messages dock and the Setup
    page apply the same conversion, so a warning reads identically everywhere.
    """
    return _POW_RE.sub(lambda m: "10" + m.group(1).translate(_SUPERSCRIPT), str(text)).replace(" -- ", " — ")


def headline(text: str, limit: int = 140) -> str:
    """The first clause of a warning, for a one-line summary (the full text goes in a tooltip).

    Cut at the first ':' / ';' / sentence end / dash outside parentheses that
    leaves at least 24 characters, then elided at ``limit`` characters.
    """
    t = " ".join(pretty(text).split())
    depth = 0
    for i, ch in enumerate(t):
        depth += (ch == "(") - (ch == ")")
        if depth > 0 or i < 24 or i + 1 >= len(t):
            continue
        if (ch in ":;." and t[i + 1] == " ") or (ch == "—" and t[i - 1] == " "):
            t = t[:i].rstrip(" ,")
            break
    if len(t) > limit:
        t = t[:limit - 1].rstrip(" ,;:") + "…"
    return t


#: Coefficient symbols that get a subscript. Section (2D) coefficients are lower case
#: (Cd, Cl, Cm, Cp, Cf, Cl,max, CdA); 3D and wing coefficients follow the aircraft
#: convention in capitals (CD, CL, CS, CDi, CDv, CD0, CL,max, CL,α, CDA) plus the
#: yawing moment Cn and the rolling moment, written ``C_roll`` because "Cl" is taken
#: by the section lift.
_COEF_RE = re.compile(r"\bC(?:([dlmpfn])((?:p|f)?(?:,(?:max|min|α))?|0)?|([DLS])(,(?:max|min|α)|0|i|v)?|_(roll))"
                      r"(A?)\b")


def subscripts(text: str) -> str:
    """HTML with coefficient subscripts: 'Cd' -> 'C<sub>d</sub>', 'Cl,max' -> 'C<sub>l,max</sub>'.

    Only whole tokens are touched ('Climb', 'Cruise', 'CSV' are not), the rest is
    escaped. The guidelines ask for C<sub>d</sub>, and Unicode has no subscript
    d, so tables, plot axes and labels render this HTML instead. 3D symbols:
    'CL,max' -> 'C<sub>L,max</sub>', 'CDi' -> 'C<sub>Di</sub>', 'C_roll' -> 'C<sub>roll</sub>'.
    """
    out, pos = [], 0
    s = str(text)
    for m in _COEF_RE.finditer(s):
        out.append(html.escape(s[pos:m.start()], quote=False))
        sub = "".join(g or "" for g in m.group(1, 2, 3, 4, 5))
        out.append(f"C<sub>{sub}</sub>{m.group(6) or ''}")
        pos = m.end()
    out.append(html.escape(s[pos:], quote=False))
    return "".join(out)


_PLAIN_NUMBER_RE = re.compile(r"^([+\-−]?\d[\d,]*)(\.\d*)?(\s?%)?$")


def decimal_tail(text: str) -> int | None:
    """Characters right of the units digit of a plain number ('0.0964' -> 5, '12' -> 0, '+3.1 %' -> 4).

    None for anything else (an em dash, '9.99×10⁵', 'yes'), which is simply
    right-aligned. A table pads every number of a column to the column's
    longest tail, so mixed decimal counts (0.0964 / 12.59 / 183.8) still line
    up on the decimal point in the monospace font, as in a printed table.
    """
    m = _PLAIN_NUMBER_RE.match(str(text).strip())
    return None if m is None else len(str(text).strip()) - len(m.group(1))


def has_symbols(text: str) -> bool:
    """Whether :func:`subscripts` would change ``text`` (i.e. it holds a coefficient symbol)."""
    return _COEF_RE.search(str(text)) is not None


# ---------------------------------------------------------------------------
# small widgets
# ---------------------------------------------------------------------------

def label(text: str = "", role: str | None = None, *, wrap: bool = False,
          parent: QWidget | None = None) -> QLabel:
    """A QLabel with a theme role ("h2", "secondary", "muted", "mono", ...)."""
    return common.label(text, role, wrap=wrap, parent=parent)


def button(text: str, variant: str | None = None, *, tooltip: str = "", icon: str | None = None,
           on_click: Callable[[], None] | None = None, parent: QWidget | None = None) -> QPushButton:
    """A 24 px QPushButton with a theme variant ("primary", "ghost", "danger", "link")."""
    b = common.button(text, variant, icon, on_click, parent)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def repolish(w: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property changed."""
    theme.repolish(w)


class StatusLabel(QLabel):
    """Status as plain text in a status colour: ``set_status("Running 43 %", "accent")``.

    ``kind`` is None (secondary text), "good", "warn", "bad", "accent" or "muted".
    The guidelines allow no pills or chips; the colour of the word is the status.
    """

    _ROLES: ClassVar[dict] = {None: "secondary", "": "secondary", "good": "status-good", "ok": "status-good",
              "warn": "status-warn", "bad": "status-bad", "accent": "badge-accent", "muted": "muted"}

    def __init__(self, text: str = "", kind: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._kind: str | None = "unset"
        self.set_status(text, kind)

    def set_status(self, text: str, kind: str | None = None, tooltip: str = "") -> None:
        self.setText(text)
        self.setToolTip(tooltip)
        if kind != self._kind:
            self._kind = kind
            theme.set_prop(self, "role", self._ROLES.get(kind, "secondary"))

    @property
    def kind(self) -> str | None:
        return self._kind


def reveal_messages(widget: QWidget) -> bool:
    """Show and raise the Messages dock of ``widget``'s main window; False if there is none.

    Duck-typed (``reveal_messages`` or the main window's ``_reveal_messages``)
    because the pages are also hosted standalone (tests, embedding) without one.
    """
    win = widget.window() if widget is not None else None
    for name in ("reveal_messages", "_reveal_messages"):
        fn = getattr(win, name, None)
        if callable(fn):
            fn()
            return True
    return False


class WarningLine(QWidget):
    """A single warning line: warn glyph, the first clause of the first warning, "Details".

    The guidelines allow one warning line under a header (section 4) and put
    explanations in tooltips (sections 1, 6): the planner's and the runner's
    warnings are 2-11 lines of prose, which pushed the controls below them off
    screen. The line shows :func:`headline` of the first warning (plus "+n
    more"), elided to the available width; the tooltip holds every warning in
    full; "Details" opens the Messages dock, where the run log has them too.
    """

    def __init__(self, parent: QWidget | None = None, *, details: bool = True) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.glyph = QLabel()
        self.glyph.setFixedSize(16, 16)
        lay.addWidget(self.glyph)
        self.text = common.ElidedLabel("", None, mode=Qt.TextElideMode.ElideRight)
        self.text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.text, 1)
        self.details = button("Details", "link", tooltip="Show every message in the Messages panel")
        self.details.clicked.connect(lambda *_: reveal_messages(self))
        self.details.setVisible(details)
        lay.addWidget(self.details)
        self._warnings: list[str] = []
        self._retint()
        theme.changed.connect(self._retint)  # bound method: auto-disconnects when deleted
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _retint(self, _name: str = "") -> None:
        self.glyph.setPixmap(theme.render_icon("warning", 14, "warn", self.devicePixelRatioF()))

    def set_warnings(self, warnings: Sequence[str]) -> None:
        """Show the first warning's headline (hidden when there are none)."""
        self._warnings = [pretty(w) for w in warnings if str(w).strip()]
        if not self._warnings:
            self.text.setText("")
            self.hide()
            return
        more = len(self._warnings) - 1
        line = headline(self._warnings[0]) + (f"  (+{more} more)" if more else "")
        self.text.setText(line)
        tip = "\n\n".join(self._warnings)
        self.text.setToolTip(tip)
        self.glyph.setToolTip(tip)
        self.setToolTip(tip)
        self.show()

    def summary(self) -> str:
        """The one line shown (for tests)."""
        return self.text.full_text()

    def warnings(self) -> list[str]:
        return list(self._warnings)


class EmptyState(common.EmptyState):
    """The foundation's empty state (one plain line, optional second line, one button).

    Adds :meth:`set_content`, which replaces message, button and the optional
    secondary button in one call, and the ``action`` alias for the button.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._cb: Callable[[], None] | None = None
        self._cb2: Callable[[], None] | None = None
        self.actionTriggered.connect(lambda: self._cb and self._cb())
        self.secondary.clicked.connect(lambda *_: self._cb2 and self._cb2())

    @property
    def action(self) -> QPushButton:
        return self.button

    def set_content(self, title: str, message: str = "", action_text: str = "",
                    callback: Callable[[], None] | None = None, *, secondary_text: str = "",
                    secondary_callback: Callable[[], None] | None = None) -> None:
        self.set_text(title, message)
        self.button.setText(action_text)
        self.button.setVisible(bool(action_text))
        self._cb = callback
        self.secondary.setText(secondary_text)
        self.secondary.setVisible(bool(secondary_text))
        self._cb2 = secondary_callback


# ---------------------------------------------------------------------------
# the read-only data table
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One table cell: text plus optional tone ("good" / "bad" / "warn" / "muted"), tooltip, bold."""
    text: str
    tone: str | None = None
    tooltip: str = ""
    bold: bool = False


def _html_doc(html_text: str, font: QFont, colour) -> QTextDocument:
    doc = QTextDocument()
    doc.setDocumentMargin(0)
    doc.setDefaultFont(font)
    doc.setHtml(f'<span style="color:{colour.name()}">{html_text}</span>')
    return doc


class RichTextDelegate(QStyledItemDelegate):
    """Draws cells that carry an :data:`HTML_ROLE` value as rich text (C<sub>d</sub>).

    The display role keeps the plain text ("Cd"), so copying, sorting, column
    sizing and tests see ordinary strings; only the painting differs. Cells
    without HTML go through the normal delegate untouched.
    """

    @staticmethod
    def _pad_px(option, index) -> float | None:
        """Decimal-alignment padding in px; None for cells that are not aligned numbers.

        Measured in the cell's own font (the monospace one of numeric columns),
        not ``option.fontMetrics``: before ``initStyleOption`` that is the
        view's UI font, whose digits are narrower, and the padding then fell
        short by a pixel or two per character - 12.59 sat 1.5 digits right of
        0.0965's decimal point instead of under it.
        """
        pad = index.data(PAD_ROLE)
        if pad is None:
            return None
        font = index.data(Qt.ItemDataRole.FontRole)
        fm = QFontMetricsF(font) if isinstance(font, QFont) else QFontMetricsF(option.font)
        return float(pad) * fm.horizontalAdvance("0")

    def sizeHint(self, option, index) -> QSize:
        size = super().sizeHint(option, index)
        pad = self._pad_px(option, index)
        return QSize(size.width() + math.ceil(pad), size.height()) if pad else size

    def paint(self, painter, option, index) -> None:
        html_text = index.data(HTML_ROLE)
        pad = self._pad_px(option, index)
        if pad is not None and not html_text:
            # every aligned number goes through the same path, padded or not, so the
            # text margin is identical down the column
            return self._paint_padded(painter, option, index, pad)
        if not html_text:
            return super().paint(painter, option, index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        fg = opt.palette.color(QPalette.ColorRole.HighlightedText if opt.state & QStyle.StateFlag.State_Selected
                               else QPalette.ColorRole.Text)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if brush is not None and not (opt.state & QStyle.StateFlag.State_Selected):
            fg = brush.color() if hasattr(brush, "color") else brush
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        doc = _html_doc(html_text, opt.font, fg)
        text_rect = (style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
                     if style is not None else option.rect)
        h = doc.size().height()
        w = doc.idealWidth()
        x = text_rect.left() + 2
        if opt.displayAlignment & Qt.AlignmentFlag.AlignRight:
            x = text_rect.right() - w - 2
        y = text_rect.top() + (text_rect.height() - h) / 2
        painter.save()
        painter.setClipRect(text_rect)
        painter.translate(QPointF(x, y))
        ctx = QAbstractTextDocumentLayout.PaintContext()
        ctx.palette.setColor(QPalette.ColorRole.Text, fg)
        doc.documentLayout().draw(painter, ctx)
        painter.restore()


    def _paint_padded(self, painter, option, index, pad: float) -> None:
        """A right-aligned number moved ``pad`` px left of the edge (decimal-point alignment).

        Padding with spaces would not work: Qt drops trailing white space when
        it right-aligns text. The cell background and selection still span the
        full cell; only the text rectangle is narrowed.
        """
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else None
        if style is not None:
            style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
            rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        else:
            rect = QRect(option.rect)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        fg = opt.palette.color(QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if brush is not None and not selected:
            fg = brush.color() if hasattr(brush, "color") else brush
        # the style's own text margin, as the plain path uses
        margin = (style.pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin, None, opt.widget) + 1
                  if style is not None else 3)
        rect = rect.adjusted(margin, 0, -margin - math.ceil(pad), 0)
        painter.save()
        painter.setFont(opt.font)
        painter.setPen(fg)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                         opt.fontMetrics.elidedText(text, Qt.TextElideMode.ElideRight, rect.width()))
        painter.restore()


class RichHeaderView(QHeaderView):
    """A header that renders labels carrying an :data:`HTML_ROLE` value as rich text."""

    def paintSection(self, painter, rect: QRect, logical: int) -> None:
        model = self.model()
        html_text = model.headerData(logical, self.orientation(), HTML_ROLE) if model is not None else None
        if not html_text or not rect.isValid():
            return super().paintSection(painter, rect, logical)
        opt = QStyleOptionHeader()
        self.initStyleOptionForIndex(opt, logical)
        opt.rect = rect
        opt.text = ""
        painter.save()
        self.style().drawControl(QStyle.ControlElement.CE_Header, opt, painter, self)
        painter.restore()
        doc = _html_doc(html_text, self.font(), theme.qcolor("text2"))
        w, h = doc.idealWidth(), doc.size().height()
        align = model.headerData(logical, self.orientation(), Qt.ItemDataRole.TextAlignmentRole)
        right = align is not None and (Qt.AlignmentFlag(int(align)) & Qt.AlignmentFlag.AlignRight)
        x = rect.right() - 6 - w if right else rect.left() + 6
        painter.save()
        painter.setClipRect(rect)
        painter.translate(QPointF(x, rect.top() + (rect.height() - h) / 2))
        ctx = QAbstractTextDocumentLayout.PaintContext()
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sectionSizeFromContents(self, logical: int) -> QSize:
        base = super().sectionSizeFromContents(logical)
        return QSize(base.width() + 2, base.height())


class DataTable(QTableWidget):
    """Read-only table for results: 22 px rows, numbers right-aligned in the mono font.

    ``set_data(headers, rows, numeric=...)`` replaces everything. A row is a
    sequence of ``str`` or :class:`Cell`. ``numeric`` lists the columns holding
    numbers (default: every column but the first and a trailing "Unit" column):
    those are right-aligned and drawn in the monospace font, as the guidelines
    ask of data tables, so digits line up between rows.
    """

    def __init__(self, headers: Sequence[str] = (), parent: QWidget | None = None, *,
                 stretch_last: bool = True) -> None:
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeader(RichHeaderView(Qt.Orientation.Horizontal, self))
        self.setItemDelegate(RichTextDelegate(self))
        if headers:
            self.setHorizontalHeaderLabels(list(headers))
            self._mark_rich_headers()
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(theme.ROW_H)
        self.verticalHeader().setMinimumSectionSize(theme.ROW_H)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)
        self.setWordWrap(False)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        hh = self.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(stretch_last)
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        hh.setMinimumSectionSize(48)
        self._numeric: set[int] = set()
        theme.changed.connect(self._retint)
        self._tones: dict[tuple[int, int], str] = {}

    def _mark_rich_headers(self) -> None:
        """Give header labels with coefficient symbols their rich-text form (C<sub>d</sub>)."""
        for j in range(self.columnCount()):
            item = self.horizontalHeaderItem(j)
            if item is not None and has_symbols(item.text()):
                item.setData(HTML_ROLE, subscripts(item.text()))

    def set_data(self, headers: Sequence[str], rows: Sequence[Sequence], *,
                 numeric: Sequence[int] | None = None) -> None:
        """Replace headers and rows (see the class docstring)."""
        n = len(headers)
        self.clear()
        self.setColumnCount(n)
        self.setHorizontalHeaderLabels(list(headers))
        self._mark_rich_headers()
        if numeric is None:
            last_unit = n > 1 and headers[-1].lower() == "unit"
            numeric = [j for j in range(1, n - (1 if last_unit else 0))]
        self._numeric = set(numeric)
        for j in range(n):
            item = self.horizontalHeaderItem(j)
            if item is not None:
                item.setTextAlignment((Qt.AlignmentFlag.AlignRight if j in self._numeric
                                       else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
        self.setRowCount(len(rows))
        self._tones = {}
        mono = theme.mono_font()
        mono_b = QFont(mono)
        mono_b.setWeight(QFont.Weight.DemiBold)
        ui_b = theme.ui_font(weight=QFont.Weight.DemiBold)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                c = value if isinstance(value, Cell) else Cell("" if value is None else str(value))
                it = QTableWidgetItem(c.text)
                if has_symbols(c.text):
                    it.setData(HTML_ROLE, subscripts(c.text))
                if j in self._numeric:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    it.setFont(mono_b if c.bold else mono)
                else:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    if c.bold:
                        it.setFont(ui_b)
                    elif j > 0:
                        it.setForeground(theme.qcolor("text2"))
                if c.tone:
                    self._tones[(i, j)] = c.tone
                    it.setForeground(theme.qcolor(self._tone_token(c.tone)))
                if c.tooltip:
                    it.setToolTip(c.tooltip)
                self.setItem(i, j, it)
        self._align_decimals()

    def _align_decimals(self) -> None:
        """Pad the numbers of each numeric column so their decimal points line up (:data:`PAD_ROLE`)."""
        for j in self._numeric:
            tails, dashes = {}, []
            for i in range(self.rowCount()):
                it = self.item(i, j)
                text = it.text().strip() if it is not None else ""
                tail = decimal_tail(text)
                if tail is not None:
                    tails[i] = tail
                elif text == "—":
                    dashes.append(i)
            if not tails or len(tails) + len(dashes) < 2:
                continue
            widest = max(tails.values())
            for i, tail in tails.items():
                self.item(i, j).setData(PAD_ROLE, widest - tail)
            for i in dashes:   # "no value" sits under the units digits, as in a printed table
                self.item(i, j).setData(PAD_ROLE, widest)

    def cell_pad(self, i: int, j: int) -> int:
        """Blank character cells right of cell (i, j)'s number (for tests)."""
        it = self.item(i, j)
        return int(it.data(PAD_ROLE) or 0) if it is not None else 0

    @staticmethod
    def _tone_token(tone: str) -> str:
        return {"good": "good", "bad": "bad", "warn": "warn", "muted": "muted", "accent": "accent"}.get(tone, "text")

    def _retint(self, _name: str = "") -> None:
        for (i, j), tone in self._tones.items():
            it = self.item(i, j)
            if it is not None:
                it.setForeground(theme.qcolor(self._tone_token(tone)))
        for i in range(self.rowCount()):
            for j in range(1, self.columnCount()):
                if (i, j) in self._tones or j in self._numeric:
                    continue
                it = self.item(i, j)
                if it is not None:
                    it.setForeground(theme.qcolor("text2"))

    def fit_height(self, max_rows: int | None = None) -> None:
        """Fix the height to show all rows (or ``max_rows``) without a scroll bar."""
        n = self.rowCount() if max_rows is None else min(self.rowCount(), max_rows)
        h = self.horizontalHeader().sizeHint().height() + n * theme.ROW_H + 2 * self.frameWidth() + 2
        self.setFixedHeight(h)

    def column_texts(self, j: int) -> list[str]:
        return [self.item(i, j).text() if self.item(i, j) else "" for i in range(self.rowCount())]

    def row_texts(self, i: int) -> list[str]:
        return [self.item(i, j).text() if self.item(i, j) else "" for j in range(self.columnCount())]

    def find_row(self, first_cell: str) -> int:
        """Index of the row whose first cell reads ``first_cell`` (-1 if none)."""
        for i in range(self.rowCount()):
            it = self.item(i, 0)
            if it is not None and it.text() == first_cell:
                return i
        return -1


__all__ = [
    "HTML_ROLE",
    "MINUS",
    "PAD_ROLE",
    "SCI_ABOVE",
    "SCI_BELOW",
    "STEADY_REL",
    "STEADY_TEXT",
    "STEADY_TIP",
    "WITHIN_ERROR_TEXT",
    "Cell",
    "DataTable",
    "EmptyState",
    "RichHeaderView",
    "RichTextDelegate",
    "StatusLabel",
    "WarningLine",
    "button",
    "decimal_tail",
    "decimals_for",
    "decimals_from_error",
    "fmt_duration",
    "fmt_err",
    "fmt_error",
    "fmt_fixed",
    "fmt_num",
    "fmt_re",
    "fmt_sci",
    "fmt_sig",
    "fmt_speed",
    "has_symbols",
    "headline",
    "is_steady",
    "label",
    "needs_sci",
    "pretty",
    "repolish",
    "reveal_messages",
    "round_error",
    "speed_units",
    "stat_texts",
    "subscripts",
    "value_error_texts",
    "within_error",
]
