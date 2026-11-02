"""A4 PDF rendering of a report :class:`~.document.Document` with matplotlib only.

matplotlib has no document layout, so this module is a small one: blocks are measured
with the Agg text engine (same font metrics the PDF uses), broken into atomic
*pieces* (a line of text, a table row, a figure with its caption ...), flowed onto
pages with keep-with-next and table-header repetition, and finally drawn onto one
``Figure`` per A4 page whose single axes uses typographic points with y pointing
down. Page numbers ("Page 3 of 12") and the table of contents need the total
pagination, so drawing happens only after the whole document has been flowed.

Figures are rasterised at print resolution (default 220 dpi) and placed with
``interpolation="none"``: the PDF backend then embeds the pixels unresampled
(PNG-compressed) instead of resampling them to the 72-dpi page grid. Text on the
pages themselves is real, selectable TrueType text (``pdf.fonttype = 42``).
"""
from __future__ import annotations

import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextToPath

from ..viz import figures as F
from .document import (
    KPI,
    Badges,
    Bullets,
    Callout,
    Cell,
    Document,
    FigureBlock,
    Heading,
    KeyValues,
    KPIRow,
    PageBreak,
    Paragraph,
    Table,
    runs,
)

# --- page geometry (points) --------------------------------------------------
PAGE_W, PAGE_H = 595.276, 841.890            # A4
ML, MR = 56.0, 56.0
TOP, BOTTOM = 78.0, PAGE_H - 66.0            # content area
TW = PAGE_W - ML - MR                        # text width
IN = 72.0                                    # points per inch

# --- palette -----------------------------------------------------------------
INK = "#1d2733"
MUTED = "#5b6776"
FAINT = "#8a94a3"
RULE = "#d5dae1"
PANEL = "#f3f5f8"
ZEBRA = "#f8f9fb"
ACCENT = "#2463c4"
ACCENT_DARK = "#1a4f9c"
TONE_FG = {"ok": "#1e7a45", "warn": "#9a6212", "bad": "#b23434", "info": "#2b5f9e", "neutral": INK}
TONE_BG = {"ok": "#e5f3ea", "warn": "#fcf1dc", "bad": "#fbe6e6", "info": "#e6eef8", "neutral": "#eef1f5"}
TONE_BAR = {"ok": "#2e9e5b", "warn": "#d69a2d", "bad": "#cf4a4a", "info": "#3f7cc4", "neutral": ACCENT}

# --- type scale (size, leading) ----------------------------------------------
BODY = (9.4, 13.6)
LEAD = (10.4, 15.0)
SMALL = (8.0, 11.2)
CAPTION = (8.0, 11.0)
H1 = 15.5
H2 = 11.0
FIG_DPI = 220


# ---------------------------------------------------------------------------
# text measurement and wrapping
# ---------------------------------------------------------------------------

# Unhinted outline metrics at a large internal scale. Hinted Agg metrics at 72 dpi round every
# glyph advance to whole pixels, which under-measures a 90-character line by ~20 pt and makes a
# bold run overlap the text before it; the PDF itself is drawn from unhinted outlines.
_T2P = TextToPath()


@lru_cache(maxsize=4096)
def _prop(size: float, bold: bool) -> FontProperties:
    return FontProperties(family=F.FONT_STACK, size=size, weight="bold" if bold else "normal")


@lru_cache(maxsize=65536)
def _ink(text: str, size: float, bold: bool) -> float:
    w, _, _ = _T2P.get_text_width_height_descent(text, _prop(size, bold), ismath=False)
    return float(w)


@lru_cache(maxsize=65536)
def text_width(text: str, size: float, bold: bool = False) -> float:
    """Advance width of ``text`` in points (leading/trailing spaces included).

    FreeType reports the ink bounding box, which drops spaces at the ends of a run; bracketing
    the text with a glyph on both sides and subtracting recovers the true advance.
    """
    if not text:
        return 0.0
    return _ink("H" + text + "H", size, bold) - _ink("HH", size, bold)


def _space(size: float, bold: bool) -> float:
    return text_width("a a", size, bold) - text_width("aa", size, bold)


Line = list[tuple[str, bool]]        # runs of (text, bold) on one line


def wrap(text: str, width: float, size: float, *, bold: bool = False) -> list[Line]:
    """Greedy word wrap of ``**bold**``-marked text into lines of runs no wider than ``width``."""
    tokens: list[tuple[str, bool]] = []
    for chunk, b in runs(text):
        for i, word in enumerate(chunk.split(" ")):
            if i > 0:
                tokens.append((" ", b or bold))
            if word:
                tokens.append((word, b or bold))
    lines: list[Line] = []
    cur: Line = []
    w = 0.0
    pending_space = 0.0
    for tok, b in tokens:
        if tok == " ":
            if cur:
                pending_space = _space(size, b)
            continue
        tw = text_width(tok, size, b)
        if cur and w + pending_space + tw > width:
            lines.append(cur)
            cur, w, pending_space = [], 0.0, 0.0
        if not cur and tw > width:                      # a single over-long token: hard-break it
            piece = ""
            for ch in tok:
                if text_width(piece + ch, size, b) > width and piece:
                    lines.append([(piece, b)])
                    piece = ""
                piece += ch
            tok, tw = piece, text_width(piece, size, b)
        if cur and pending_space:
            last_t, last_b = cur[-1]
            if last_b == b:
                cur[-1] = (last_t + " " + tok, b)
            else:
                cur[-1] = (last_t + " ", last_b)
                cur.append((tok, b))
            w += pending_space + tw
        else:
            cur.append((tok, b))
            w += tw
        pending_space = 0.0
    if cur:
        lines.append(cur)
    return lines or [[("", bold)]]


def _line_width(line: Line, size: float) -> float:
    return sum(text_width(t, size, b) for t, b in line)


# ---------------------------------------------------------------------------
# drawing primitives on a page axes (points, y down)
# ---------------------------------------------------------------------------

def _text(ax: Any, x: float, y: float, s: str, size: float, *, color: str = INK, bold: bool = False,
          ha: str = "left", italic: bool = False) -> None:
    """Draw text with its baseline at y."""
    if not s:
        return
    ax.text(x, y, s, fontsize=size, color=color, fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal", ha=ha, va="baseline",
            fontfamily=F.FONT_STACK, parse_math=False, clip_on=False)


def _draw_line(ax: Any, x: float, y: float, line: Line, size: float, color: str = INK,
               bold_color: str | None = None, align: str = "left", width: float = 0.0) -> None:
    if align != "left":
        lw = _line_width(line, size)
        x = x + (width - lw) if align == "right" else x + (width - lw) / 2
    for t, b in line:
        _text(ax, x, y, t, size, color=(bold_color or color) if b else color, bold=b)
        x += text_width(t, size, b)


def _rule(ax: Any, x0: float, x1: float, y: float, color: str = RULE, lw: float = 0.5) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")


def _box(ax: Any, x: float, y: float, w: float, h: float, *, fc: str, ec: str = "none", lw: float = 0.0,
         r: float = 0.0) -> None:
    if r > 0:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    else:
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))


def _baseline(top: float, size: float, leading: float) -> float:
    """Baseline of a text line whose line box starts at ``top`` (cap height ~0.72 em)."""
    return top + (leading - size) / 2 + 0.80 * size


# ---------------------------------------------------------------------------
# pieces and flow
# ---------------------------------------------------------------------------

@dataclass
class Piece:
    """An unbreakable slice of content with a known height."""
    height: float
    draw: Callable[[Any, float, float], None]       # (ax, x, top_y)
    space_before: float = 0.0
    keep_with_next: bool = False
    min_space: float = 0.0          # start a new page unless this much room remains (section headings)
    cont_header: Piece | None = None  # table header to repeat when a page break lands before this row
    section: tuple[str, str, str] | None = None     # (key, number, title) for H1 pieces
    page_break: bool = False
    images: list = field(default_factory=list)


class _Ctx:
    """Rendering options shared by the block converters."""

    def __init__(self, fig_dpi: float, errors: list[str]) -> None:
        self.fig_dpi = fig_dpi
        self.errors = errors


def _text_pieces(text: str, size: float, leading: float, *, color: str = INK, width: float = TW,
                 indent: float = 0.0, space_before: float = 0.0, bullet: str | None = None,
                 bullet_color: str = ACCENT, bold: bool = False) -> list[Piece]:
    lines = wrap(text, width - indent, size, bold=bold)
    out = []
    for i, ln in enumerate(lines):
        def draw(ax: Any, x: float, y: float, ln: Line = ln, first: bool = i == 0) -> None:
            base = _baseline(y, size, leading)
            if bullet and first:
                _text(ax, x + indent - 10, base, bullet, size, color=bullet_color, bold=True)
            _draw_line(ax, x + indent, base, ln, size, color=color)
        out.append(Piece(leading, draw, space_before=space_before if i == 0 else 0.0,
                         keep_with_next=(i == 0 and len(lines) > 1)))
    return out


def _heading_pieces(h: Heading) -> list[Piece]:
    if h.level <= 1:
        size = H1
        height = size * 1.35 + 9

        def draw(ax: Any, x: float, y: float) -> None:
            base = y + size * 1.05
            nx = x
            if h.number:
                _text(ax, x, base, h.number, size, color=ACCENT, bold=True)
                nx = x + max(text_width(h.number, size, True) + 10, 24)
            _text(ax, nx, base, h.text, size, color=INK, bold=True)
            _rule(ax, x, x + TW, base + 7.5, color=RULE, lw=0.7)
            _rule(ax, x, x + 36, base + 7.5, color=ACCENT, lw=1.6)
        return [Piece(height, draw, space_before=22, keep_with_next=True, min_space=0.24 * (BOTTOM - TOP),
                      section=("", h.number, h.text))]
    size = H2
    height = size * 1.45

    def draw2(ax: Any, x: float, y: float) -> None:
        base = y + size * 1.05
        nx = x
        if h.number:
            _text(ax, x, base, h.number, size, color=ACCENT_DARK, bold=True)
            nx = x + text_width(h.number, size, True) + 8
        for i, ln in enumerate(wrap(h.text, TW - (nx - x), size, bold=True)[:1]):
            _draw_line(ax, nx, base + i * size * 1.3, ln, size, color=INK)
    return [Piece(height, draw2, space_before=13, keep_with_next=True, min_space=90)]


def _kv_pieces(kv: KeyValues) -> list[Piece]:
    out: list[Piece] = []
    if kv.title:
        out += _text_pieces(f"**{kv.title}**", 9.0, 14.0, space_before=8)
        out[-1].keep_with_next = True
    cols = max(1, min(kv.columns, 2))
    gap = 18.0
    cw = (TW - gap * (cols - 1)) / cols
    lw = cw * (0.44 if cols == 2 else 0.34)
    vw = cw - lw - 6
    size_l, size_v, lead = 8.2, 8.8, 11.6
    rows = [kv.rows[i:i + cols] for i in range(0, len(kv.rows), cols)]
    for ri, row in enumerate(rows):
        wrapped = [(wrap(k, lw, size_l), wrap(v, vw, size_v)) for k, v in row]
        nlines = max(max(len(a), len(b)) for a, b in wrapped)
        h = nlines * lead + 7

        def draw(ax: Any, x: float, y: float, wrapped: list = wrapped, h: float = h) -> None:
            for c, (kl, vl) in enumerate(wrapped):
                cx = x + c * (cw + gap)
                for i, ln in enumerate(kl):
                    _draw_line(ax, cx, _baseline(y + 3.5 + i * lead, size_l, lead), ln, size_l, color=MUTED)
                for i, ln in enumerate(vl):
                    _draw_line(ax, cx + lw + 6, _baseline(y + 3.5 + i * lead, size_v, lead), ln, size_v, color=INK)
                _rule(ax, cx, cx + cw, y + h, color=RULE, lw=0.45)
        out.append(Piece(h, draw, space_before=6 if ri == 0 and not kv.title else 0.0,
                         keep_with_next=len(rows) <= 6 and ri < len(rows) - 1))
    return out


def _cell(c: str | Cell) -> Cell:
    return c if isinstance(c, Cell) else Cell(str(c))


def _table_pieces(t: Table) -> list[Piece]:
    size = 8.3 * t.font_scale
    hsize = 7.9 * t.font_scale
    lead = size * 1.32
    pad_x, pad_y = 5.0, 3.6
    ncol = len(t.header)
    align = (t.align or ["l"] * ncol) + ["l"] * ncol
    cells = [[_cell(c) for c in r] + [Cell("")] * (ncol - len(r)) for r in t.rows]
    # column widths: given ratios, else natural widths scaled to the text width
    if t.widths and len(t.widths) == ncol:
        tot = sum(t.widths)
        widths = [TW * w / tot for w in t.widths]
    else:
        nat = []
        for j in range(ncol):
            w = text_width(t.header[j], hsize, True)
            for r in cells:
                w = max(w, text_width(r[j].text, size, r[j].bold))
            nat.append(w + 2 * pad_x)
        tot = sum(nat)
        if tot <= TW:
            extra = TW - tot
            widths = [w + extra * w / tot for w in nat]
        else:
            # shrink the widest columns first so short numeric columns stay on one line
            widths = list(nat)
            while sum(widths) > TW + 0.5:
                j = int(np.argmax(widths))
                widths[j] -= min(widths[j] - 30, sum(widths) - TW) if widths[j] > 30 else 0
                if widths[j] <= 30:
                    break
            if sum(widths) > TW:
                widths = [w * TW / sum(widths) for w in widths]
    xs = np.concatenate([[0.0], np.cumsum(widths)])

    def cell_lines(text: str, j: int, sz: float, bold: bool) -> list[Line]:
        return wrap(text, widths[j] - 2 * pad_x, sz, bold=bold)

    out: list[Piece] = []
    if t.caption:
        label = f"**Table {t.number}.** {t.caption}" if t.number else t.caption
        cap = _text_pieces(label, CAPTION[0], CAPTION[1], color=MUTED, space_before=10)
        for p in cap:
            p.keep_with_next = True
        out += cap
    hl = [cell_lines(h, j, hsize, True) for j, h in enumerate(t.header)]
    hh = max(len(x) for x in hl) * hsize * 1.3 + 2 * pad_y + 1

    def draw_header(ax: Any, x: float, y: float) -> None:
        _box(ax, x, y, TW, hh, fc=PANEL)
        for j, lines in enumerate(hl):
            for i, ln in enumerate(lines):
                _draw_line(ax, x + xs[j] + pad_x, _baseline(y + pad_y + i * hsize * 1.3, hsize, hsize * 1.3), ln,
                           hsize, color=INK, align={"r": "right", "c": "center"}.get(align[j], "left"),
                           width=widths[j] - 2 * pad_x)
        _rule(ax, x, x + TW, y + hh, color=INK, lw=0.7)
    header = Piece(hh, draw_header, space_before=0 if t.caption else 10, keep_with_next=True)
    out.append(header)
    for ri, row in enumerate(cells):
        rl = [cell_lines(c.text, j, size, c.bold) for j, c in enumerate(row)]
        rh = max(len(x) for x in rl) * lead + 2 * pad_y
        last = ri == len(cells) - 1

        def draw_row(ax: Any, x: float, y: float, row: list = row, rl: list = rl, rh: float = rh,
                     ri: int = ri, last: bool = last) -> None:
            if ri % 2 == 1:
                _box(ax, x, y, TW, rh, fc=ZEBRA)
            for j, (c, lines) in enumerate(zip(row, rl, strict=True)):
                col = TONE_FG.get(c.tone, INK)
                for i, ln in enumerate(lines):
                    _draw_line(ax, x + xs[j] + pad_x, _baseline(y + pad_y + i * lead, size, lead), ln, size,
                               color=col, bold_color=col,
                               align={"r": "right", "c": "center"}.get(align[j], "left"),
                               width=widths[j] - 2 * pad_x)
            _rule(ax, x, x + TW, y + rh, color=INK if last else RULE, lw=0.7 if last else 0.35)
        out.append(Piece(rh, draw_row, cont_header=header if ri > 0 else None,
                         keep_with_next=len(cells) <= 6 and not last))
    if not cells:
        out.append(Piece(lead + 2 * pad_y, lambda ax, x, y: _text(ax, x + pad_x, _baseline(y + pad_y, size, lead),
                                                                  "No rows", size, color=MUTED)))
    return out


def _kpi_pieces(row: KPIRow) -> list[Piece]:
    out: list[Piece] = []
    per = 4
    for start in range(0, len(row.items), per):
        items: list[KPI] = row.items[start:start + per]
        n = len(items)
        gap = 8.0
        w = (TW - gap * (n - 1)) / n
        inner = w - 18
        labels = [wrap(k.label, inner, 7.6) for k in items]
        subs = [wrap(k.sub, inner, 7.4) if k.sub else [] for k in items]
        nl = max(len(x) for x in labels)
        ns = max((len(x) for x in subs), default=0)
        h = 10 + nl * 9.6 + 25 + ns * 9.6 + 8

        def draw(ax: Any, x: float, y: float, items: list = items, labels: list = labels, subs: list = subs,
                 w: float = w, h: float = h, nl: int = nl, gap: float = gap) -> None:
            for i, k in enumerate(items):
                bx = x + i * (w + gap)
                # a flat cell under a 0.8 pt rule (docs/ui_guidelines.md: no tiles, no coloured bars)
                _rule(ax, bx, bx + w, y, color=INK, lw=0.8)
                tx = bx + 2
                for j, ln in enumerate(labels[i]):
                    _draw_line(ax, tx, y + 16 + j * 9.6, ln, 7.6, color=MUTED)
                vb = y + 10 + nl * 9.6 + 20
                vsize = 14.0 if text_width(k.value, 14.0, True) + text_width(k.unit, 8.5) + 4 < w - 6 else 12.0
                _text(ax, tx, vb, k.value, vsize, color=TONE_FG.get(k.tone, INK) if k.tone != "neutral" else INK,
                      bold=True)
                if k.unit:
                    _text(ax, tx + text_width(k.value, vsize, True) + 3, vb, k.unit, 8.5, color=MUTED)
                for j, ln in enumerate(subs[i]):
                    _draw_line(ax, tx, vb + 13 + j * 9.6, ln, 7.4, color=MUTED)
        out.append(Piece(h, draw, space_before=8 if start == 0 else 6))
    return out


def _badge_pieces(b: Badges) -> list[Piece]:
    """Validity flags as one line of plain status text, "a · b · c", coloured by tone.

    Status is text, not pills (docs/ui_guidelines.md section 1); only the words carry the
    good / warn / bad colour.
    """
    size = 7.8
    hb = 11.0
    sep = " · "
    sep_w = text_width(sep, size)
    rows: list[list[tuple[str, str, float]]] = [[]]
    x = 0.0
    for text, tone in b.items:
        w = text_width(text, size)
        if rows[-1] and x + sep_w + w > TW:
            rows.append([])
            x = 0.0
        rows[-1].append((text, tone, w))
        x += w + sep_w
    out = []
    for i, row in enumerate(rows):
        def draw(ax: Any, x0: float, y: float, row: list = row) -> None:
            x = x0
            for j, (text, tone, w) in enumerate(row):
                if j:
                    _text(ax, x, y + hb * 0.75, sep, size, color=FAINT)
                    x += sep_w
                colour = TONE_FG.get(tone, INK) if tone in ("ok", "warn", "bad") else MUTED
                _text(ax, x, y + hb * 0.75, text, size, color=colour)
                x += w
        out.append(Piece(hb + 2, draw, space_before=6 if i == 0 else 0.0))
    return out


def _callout_pieces(c: Callout) -> list[Piece]:
    size, lead = 8.6, 12.0
    inner = TW - 26
    lines = wrap((f"**{c.title}:** " if c.title else "") + c.text, inner, size)
    h = len(lines) * lead + 14

    def draw(ax: Any, x: float, y: float) -> None:
        _box(ax, x, y, TW, h, fc=TONE_BG.get(c.tone, TONE_BG["info"]), r=3)
        _box(ax, x, y, 3.0, h, fc=TONE_BAR.get(c.tone, ACCENT))
        for i, ln in enumerate(lines):
            _draw_line(ax, x + 14, _baseline(y + 7 + i * lead, size, lead), ln, size, color=INK,
                       bold_color=TONE_FG.get(c.tone, INK))
    return [Piece(h, draw, space_before=8)]


def figure_failure_text(fb: FigureBlock, exc: BaseException | None) -> str:
    """What the report says in place of a figure that could not be drawn.

    A missing OpenGL context (3D renders on a headless machine) is a property of the
    computer, not of the data, so it gets its own sentence with the reason.
    """
    from .render3d import RenderUnavailable
    if isinstance(exc, RenderUnavailable):
        return (f"Figure {fb.number} is a 3D rendering, which is not available on this computer ({exc}). "
                "Generate the report on a machine with OpenGL to include it.")
    return f"Figure {fb.number} could not be produced from the stored data ({fb.caption[:80]}…)."


def _render_figure(fb: FigureBlock, dpi: float, errors: list[str]) -> np.ndarray | None:
    try:
        fig = fb.build(style="print", figsize=fb.size)
        rgba = F.figure_to_rgba(fig, dpi=dpi)
        return rgba[..., :3].copy()
    except Exception as exc:  # noqa: BLE001 - one broken figure must not sink the report
        errors.append(f"Figure {fb.number}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
        fb.error = exc                      # read by figure_failure_text
        return None


def _figure_pieces(fb: FigureBlock, ctx: _Ctx) -> list[Piece]:
    img = _render_figure(fb, ctx.fig_dpi, ctx.errors)
    if img is None:
        return _callout_pieces(Callout(figure_failure_text(fb, getattr(fb, "error", None)), "warn"))
    hpx, wpx = img.shape[:2]
    w = TW * max(min(fb.width, 1.0), 0.2)
    h = w * hpx / wpx
    hmax = (BOTTOM - TOP) * 0.80
    if h > hmax:
        w, h = w * hmax / h, hmax
    cap = f"**Figure {fb.number}.** {fb.caption}" if fb.number else fb.caption
    cap_lines = wrap(cap, TW, CAPTION[0]) if fb.caption else []
    total = h + (6 + len(cap_lines) * CAPTION[1] if cap_lines else 0)
    holder: list = []

    def draw(ax: Any, x: float, y: float) -> None:
        x0 = x + (TW - w) / 2
        im = ax.imshow(img, extent=(x0, x0 + w, y + h, y), interpolation="none", zorder=2)
        holder.append(im)
        for i, ln in enumerate(cap_lines):
            _draw_line(ax, x, _baseline(y + h + 6 + i * CAPTION[1], CAPTION[0], CAPTION[1]), ln, CAPTION[0],
                       color=MUTED, bold_color=INK)
    return [Piece(total, draw, space_before=10, images=holder)]


def to_pieces(blocks: Sequence[Any], ctx: _Ctx) -> list[Piece]:
    out: list[Piece] = []
    for b in blocks:
        if isinstance(b, Heading):
            out += _heading_pieces(b)
        elif isinstance(b, Paragraph):
            size, lead = {"lead": LEAD, "small": SMALL, "muted": (8.6, 12.4)}.get(b.tone, BODY)
            color = MUTED if b.tone in ("muted", "small") else INK
            ps = _text_pieces(b.text, size, lead, color=color, space_before=7)
            if b.text.rstrip().endswith(":") and ps:
                ps[-1].keep_with_next = True        # an introduction stays with its list
            out += ps
        elif isinstance(b, Bullets):
            size, lead = SMALL if b.tone == "small" else ((8.6, 12.4) if b.tone == "muted" else BODY)
            color = MUTED if b.tone in ("muted", "small") else INK
            for i, item in enumerate(b.items):
                out += _text_pieces(item, size, lead, color=color, indent=12, bullet="•" if b.tone != "small" else "",
                                    space_before=6 if i == 0 else 2.5)
        elif isinstance(b, KeyValues):
            out += _kv_pieces(b)
        elif isinstance(b, Table):
            out += _table_pieces(b)
        elif isinstance(b, KPIRow):
            out += _kpi_pieces(b)
        elif isinstance(b, Badges):
            out += _badge_pieces(b)
        elif isinstance(b, Callout):
            out += _callout_pieces(b)
        elif isinstance(b, FigureBlock):
            out += _figure_pieces(b, ctx)
        elif isinstance(b, PageBreak):
            out.append(Piece(0.0, lambda ax, x, y: None, page_break=True))
    return out


@dataclass
class _Page:
    placed: list[tuple[Piece, float]] = field(default_factory=list)
    section: tuple[str, str] | None = None      # (number, title) shown in the running header


def flow(pieces: Sequence[Piece]) -> list[_Page]:
    """Place pieces top-to-bottom, breaking pages with keep-with-next and repeated table headers."""
    pages = [_Page()]
    y = TOP
    current: tuple[str, str] | None = None
    avail = BOTTOM - TOP

    def new_page() -> None:
        nonlocal y
        pages.append(_Page(section=current))
        y = TOP

    for i, pc in enumerate(pieces):
        if pc.page_break:
            if pages[-1].placed:
                new_page()
            continue
        at_top = y <= TOP + 0.01
        sb = 0.0 if at_top else pc.space_before
        need = sb + pc.height
        j, chain = i, need
        while pieces[j].keep_with_next and j + 1 < len(pieces) and chain < avail:
            nxt = pieces[j + 1]
            # the whole next piece when it can fit a page at all: a heading kept with a figure
            # taller than half a page would otherwise be left alone at the foot of a page
            chain += nxt.space_before + (nxt.height if nxt.height <= avail else avail * 0.5)
            j += 1
            if not nxt.keep_with_next:
                break
        room = BOTTOM - y
        if not at_top and (chain > room or (pc.min_space and room < pc.min_space)):
            new_page()
            sb = 0.0
            if pc.cont_header is not None:
                pages[-1].placed.append((pc.cont_header, y))
                y += pc.cont_header.height
        if pc.section is not None:
            current = (pc.section[1], pc.section[2])
            if pages[-1].section is None or not pages[-1].placed:
                pages[-1].section = current
        pages[-1].placed.append((pc, y + sb))
        y += sb + pc.height
        if pc.section is not None and pages[-1].section is None:
            pages[-1].section = current
    if not pages[-1].placed and len(pages) > 1:
        pages.pop()
    return pages


# ---------------------------------------------------------------------------
# page furniture
# ---------------------------------------------------------------------------

def _new_page_figure() -> tuple[Figure, Any]:
    fig = Figure(figsize=(PAGE_W / IN, PAGE_H / IN), dpi=IN, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(PAGE_H, 0)
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def _header_footer(ax: Any, doc: Document, page_no: int, n_pages: int, section: tuple[str, str] | None) -> None:
    hy = 44.0
    _box(ax, ML, hy - 7.2, 5.5, 5.5, fc=ACCENT)
    left = f"{doc.subtitle}  ·  {doc.title}"
    _text(ax, ML + 10, hy, left, 7.4, color=MUTED)
    if section:
        _text(ax, PAGE_W - MR, hy, f"{section[0]}   {section[1]}".strip(), 7.4, color=MUTED, ha="right")
    _rule(ax, ML, PAGE_W - MR, hy + 7, color=RULE, lw=0.5)
    fy = PAGE_H - 38.0
    _rule(ax, ML, PAGE_W - MR, fy - 12, color=RULE, lw=0.5)
    _text(ax, ML, fy, f"{doc.software} · open-source aerodynamics", 7.2, color=FAINT)
    _text(ax, PAGE_W - MR, fy, f"Page {page_no} of {n_pages}", 7.4, color=MUTED, ha="right")


def _draw_title_page(ax: Any, doc: Document, toc: list[tuple[str, str, int]], ctx: _Ctx) -> list:
    """Cover: brand band, title, metadata, geometry thumbnail, contents with page numbers, disclaimer."""
    band = 132.0
    _box(ax, 0, 0, PAGE_W, band, fc=ACCENT)
    _box(ax, 0, band, PAGE_W, 3.0, fc=ACCENT_DARK)
    _text(ax, ML, 54, "AEROLUME", 12.5, color="white", bold=True)
    _text(ax, ML + text_width("AEROLUME", 12.5, True) + 10, 54, "open-source aerodynamics", 9.0, color="#dbe6f7")
    _text(ax, ML, 100, doc.date, 9.0, color="#dbe6f7")
    y = band + 58
    for ln in wrap(doc.title, TW, 25.0, bold=True)[:3]:
        _draw_line(ax, ML, y, ln, 25.0, color=INK)
        y += 31
    for ln in wrap(doc.subtitle, TW, 14.0)[:2]:
        _draw_line(ax, ML, y + 2, ln, 14.0, color=MUTED)
        y += 20
    _rule(ax, ML, ML + 48, y + 6, color=ACCENT, lw=2.2)
    y += 26
    # metadata
    lw = 92.0
    for k, v in doc.meta:
        lines = wrap(v, TW - lw, 9.0)[:3]
        _text(ax, ML, _baseline(y, 8.4, 13), k, 8.4, color=MUTED)
        for i, ln in enumerate(lines):
            _draw_line(ax, ML + lw, _baseline(y + i * 13, 9.0, 13), ln, 9.0, color=INK)
        y += 13 * len(lines) + 4
    images: list = []
    # contents block height decides how much room the cover picture gets
    toc_h = 30 + 15.5 * len(toc)
    disc_lines = wrap(doc.disclaimer, TW - 24, 7.8) if doc.disclaimer else []
    disc_h = len(disc_lines) * 10.8 + 16 if disc_lines else 0
    bottom_block = PAGE_H - 50 - disc_h - 16 - toc_h
    if doc.cover_figure is not None:
        room = bottom_block - (y + 18) - 14
        if room > 70:
            img = _render_figure(doc.cover_figure, ctx.fig_dpi, ctx.errors)
            if img is not None:
                hpx, wpx = img.shape[:2]
                w = TW
                h = w * hpx / wpx
                if h > room:
                    w, h = w * room / h, room
                x0 = ML + (TW - w) / 2
                images.append(ax.imshow(img, extent=(x0, x0 + w, y + 18 + h, y + 18), interpolation="none"))
    # contents
    y = bottom_block
    _text(ax, ML, y + 12, "Contents", 11.0, color=INK, bold=True)
    _rule(ax, ML, PAGE_W - MR, y + 19, color=RULE, lw=0.6)
    y += 30
    for num, title, page in toc:
        base = y + 9
        _text(ax, ML, base, num, 9.0, color=ACCENT_DARK, bold=True)
        _text(ax, ML + 22, base, title, 9.0, color=INK)
        x_end = ML + 22 + text_width(title, 9.0) + 6
        pg = str(page)
        x_pg = PAGE_W - MR - text_width(pg, 9.0) - 6
        if x_pg > x_end:
            ax.plot([x_end, x_pg], [base - 1, base - 1], color=RULE, lw=0.8, ls=(0, (1, 2.5)))
        _text(ax, PAGE_W - MR, base, pg, 9.0, color=MUTED, ha="right")
        y += 15.5
    if disc_lines:
        dy = PAGE_H - 50 - disc_h
        _box(ax, ML, dy, TW, disc_h, fc=PANEL, r=3)
        for i, ln in enumerate(disc_lines):
            _draw_line(ax, ML + 12, _baseline(dy + 8 + i * 10.8, 7.8, 10.8), ln, 7.8, color=MUTED)
    return images


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def render_pdf(doc: Document, path: str | Path, *, fig_dpi: float = FIG_DPI, png_dir: str | Path | None = None,
               png_dpi: float = 110) -> dict:
    """Write ``doc`` as an A4 PDF. Returns ``{"pages": n, "errors": [...]}``.

    ``png_dir`` additionally saves every page as a PNG (layout review / debugging); those
    rasters resample the embedded figures, so they are previews, not print output.
    """
    path = Path(path)
    errors: list[str] = []
    ctx = _Ctx(fig_dpi, errors)
    st = F.get_style("print")
    # no_hinting: the PNG previews then lay text out with the same unhinted advances the PDF
    # uses (and that text_width measures), so previews show the real line breaks and spacing.
    rc = {**F._rc(st), "pdf.fonttype": 42, "pdf.compression": 9, "text.hinting": "no_hinting"}
    with mpl.rc_context(rc):
        pieces: list[Piece] = []
        sec_first_piece: list[tuple[int, Any]] = []
        for sec in doc.sections:
            sec_first_piece.append((len(pieces), sec))
            hp = _heading_pieces(Heading(sec.title, 1, sec.number))
            hp[0].section = (sec.key, sec.number, sec.title)
            pieces += hp
            pieces += to_pieces(sec.blocks, ctx)
        pages = flow(pieces)
        n_pages = len(pages) + 1
        # table of contents: page of each section heading
        where = {}
        for pi, pg in enumerate(pages):
            for pc, _ in pg.placed:
                if pc.section is not None and pc.section[0] not in where:
                    where[pc.section[0]] = pi + 2
        toc = [(s.number, s.title, where.get(s.key, 2)) for s in doc.sections]
        if png_dir is not None:
            Path(png_dir).mkdir(parents=True, exist_ok=True)
        meta = {"Title": doc.title, "Subject": doc.subtitle, "Author": doc.author or "",
                "Creator": doc.software, "Keywords": "aerodynamics, CFD, Aerolume"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(path, metadata=meta) as pdf:
            fig, ax = _new_page_figure()
            imgs = _draw_title_page(ax, doc, toc, ctx)
            _save_page(pdf, fig, imgs, png_dir, 1, png_dpi)
            for pi, pg in enumerate(pages):
                fig, ax = _new_page_figure()
                _header_footer(ax, doc, pi + 2, n_pages, pg.section)
                imgs = []
                for pc, y in pg.placed:
                    pc.draw(ax, ML, y)
                    for im in pc.images:
                        imgs.append(im)
                    pc.images.clear()
                _save_page(pdf, fig, imgs, png_dir, pi + 2, png_dpi)
    return {"pages": n_pages, "errors": errors}


def _save_page(pdf: PdfPages, fig: Figure, images: list, png_dir: str | Path | None, page_no: int,
               png_dpi: float) -> None:
    pdf.savefig(fig)
    if png_dir is not None:
        for im in images:
            im.set_interpolation("antialiased")       # nicer downsampling for the preview only
        fig.savefig(Path(png_dir) / f"page-{page_no:02d}.png", dpi=png_dpi, facecolor="white")
    fig.clear()


__all__ = ["PAGE_H", "PAGE_W", "TW", "flow", "render_pdf", "text_width", "to_pieces", "wrap"]

