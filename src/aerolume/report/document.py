"""Format-neutral report document model.

The report is built once as a list of semantic blocks (headings, paragraphs,
key-value tables, KPI tiles, figures ...) and then rendered by two independent
back ends: :mod:`.pdf` (A4 pages laid out with matplotlib) and :mod:`.html`
(one self-contained file). Keeping content and layout apart is what lets the two
formats say exactly the same thing, and lets tests check the *numbers* in a report
without parsing a PDF.

Inline markup in text is deliberately tiny: ``**bold**`` only. Everything else
(colour, size, spacing) is decided by the renderers so the document stays
consistent no matter which section wrote it.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Tones shared by badges, callouts, KPI tiles and table cells.
TONES = ("neutral", "ok", "warn", "bad", "info")


@dataclass
class Heading:
    """Section (level 1) or subsection (level 2) title. ``number`` like "3" or "3.2"."""
    text: str
    level: int = 1
    number: str = ""
    anchor: str = ""


@dataclass
class Paragraph:
    """Running text. ``tone``: "body" | "muted" | "small" | "lead"."""
    text: str
    tone: str = "body"


@dataclass
class Bullets:
    items: list[str]
    tone: str = "body"


@dataclass
class KeyValues:
    """Label/value pairs, laid out in ``columns`` pairs per row (1 or 2)."""
    rows: list[tuple[str, str]]
    columns: int = 2
    title: str = ""


@dataclass
class Cell:
    """A table cell with an optional tone (good/bad deltas) and weight."""
    text: str
    tone: str = "neutral"
    bold: bool = False


@dataclass
class Table:
    """A data table. ``align`` per column: "l" | "r" | "c". Long tables split across pages
    in the PDF with the header repeated. ``number`` is assigned by :class:`Document`."""
    header: list[str]
    rows: list[list[str | Cell]]
    align: list[str] | None = None
    caption: str = ""
    widths: list[float] | None = None      # relative column widths; None = from content
    number: int = 0
    font_scale: float = 1.0                 # < 1 for wide tables


@dataclass
class KPI:
    """One summary tile: a big number with its unit, a label and a small sub-line."""
    label: str
    value: str
    unit: str = ""
    sub: str = ""
    tone: str = "neutral"


@dataclass
class KPIRow:
    items: list[KPI]


@dataclass
class Badges:
    """Pill badges, e.g. validity flags: (text, tone)."""
    items: list[tuple[str, str]]


@dataclass
class Callout:
    """A boxed note. ``tone`` sets the colour bar: info / ok / warn / bad."""
    text: str
    tone: str = "info"
    title: str = ""


@dataclass
class FigureBlock:
    """A figure built lazily by ``build(style=..., figsize=...)`` -> matplotlib Figure.

    ``width`` is the fraction of the text width. ``size`` (inches) is passed as ``figsize``;
    ``None`` lets content-sized builders choose their own height at full width. Building is
    deferred to the renderer so a failure in one figure becomes a note, not a crashed report.
    """
    build: Callable[..., Any]
    caption: str = ""
    width: float = 1.0
    size: tuple[float, float] | None = None
    number: int = 0


@dataclass
class PageBreak:
    pass


Block = (Heading | Paragraph | Bullets | KeyValues | Table | KPIRow | Badges | Callout | FigureBlock
         | PageBreak)


@dataclass
class Section:
    key: str                  # one of report.SECTIONS, or "appendix"
    title: str
    number: str               # "1", "2", ... or "A"
    blocks: list[Any] = field(default_factory=list)

    @property
    def anchor(self) -> str:
        return f"sec-{self.key}"


@dataclass
class Document:
    """Everything a renderer needs. ``meta`` is shown on the title page in order."""
    title: str
    subtitle: str
    author: str
    date: str
    software: str
    meta: list[tuple[str, str]] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    cover_figure: FigureBlock | None = None
    disclaimer: str = ""

    def number_floats(self) -> None:
        """Assign running Figure / Table numbers in reading order."""
        nf = nt = 0
        for sec in self.sections:
            for b in sec.blocks:
                if isinstance(b, FigureBlock):
                    nf += 1
                    b.number = nf
                elif isinstance(b, Table) and b.caption:
                    nt += 1
                    b.number = nt

    def iter_blocks(self):
        for sec in self.sections:
            yield from sec.blocks

    def text(self) -> str:
        """All visible text, one block per line (search / tests / accessibility)."""
        out = [self.title, self.subtitle, *(f"{k}: {v}" for k, v in self.meta)]
        for sec in self.sections:
            out.append(f"{sec.number} {sec.title}")
            for b in sec.blocks:
                out.append(block_text(b))
        return "\n".join(x for x in out if x)


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def runs(text: str) -> list[tuple[str, bool]]:
    """Split ``**bold**`` markup into (text, is_bold) runs."""
    out: list[tuple[str, bool]] = []
    pos = 0
    for m in _BOLD.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(1), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def plain(text: str) -> str:
    """Text with markup removed."""
    return _BOLD.sub(r"\1", text)


def cell_text(c: str | Cell) -> str:
    return c.text if isinstance(c, Cell) else str(c)


def block_text(b: Any) -> str:
    if isinstance(b, Heading):
        return f"{b.number} {b.text}".strip()
    if isinstance(b, Paragraph):
        return plain(b.text)
    if isinstance(b, Bullets):
        return "\n".join("- " + plain(i) for i in b.items)
    if isinstance(b, KeyValues):
        return "\n".join(f"{k}: {v}" for k, v in b.rows)
    if isinstance(b, Table):
        lines = [b.caption, " | ".join(b.header)]
        lines += [" | ".join(cell_text(c) for c in r) for r in b.rows]
        return "\n".join(x for x in lines if x)
    if isinstance(b, KPIRow):
        return "\n".join(f"{k.label}: {k.value} {k.unit} {k.sub}".strip() for k in b.items)
    if isinstance(b, Badges):
        return " · ".join(t for t, _ in b.items)
    if isinstance(b, Callout):
        return f"{b.title}: {plain(b.text)}" if b.title else plain(b.text)
    if isinstance(b, FigureBlock):
        return f"Figure {b.number}. {plain(b.caption)}"
    return ""
