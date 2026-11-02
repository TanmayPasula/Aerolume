"""Self-contained HTML rendering of a report :class:`~.document.Document`.

One file, no external requests: CSS is inline, figures are base64 PNGs, fonts are
the reader's system UI fonts. It is meant to be e-mailed, attached to a ticket or
opened offline years later, so nothing may depend on a CDN. A print stylesheet
lays it out on A4 with sensible page breaks for readers who prefer to print from
the browser.
"""
from __future__ import annotations

import base64
import html
import traceback
from pathlib import Path
from typing import Any

from ..viz import figures as F
from .document import (
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

FIG_DPI = 160

CSS = """
:root{--ink:#1d2733;--muted:#5b6776;--faint:#8a94a3;--rule:#d5dae1;--panel:#f3f5f8;--zebra:#f8f9fb;
--accent:#2463c4;--accent-dark:#1a4f9c;--ok:#1e7a45;--ok-bg:#e5f3ea;--ok-bar:#2e9e5b;--warn:#9a6212;
--warn-bg:#fcf1dc;--warn-bar:#d69a2d;--bad:#b23434;--bad-bg:#fbe6e6;--bad-bar:#cf4a4a;--info:#2b5f9e;
--info-bg:#e6eef8;--info-bar:#3f7cc4;--neutral-bg:#eef1f5}
*{box-sizing:border-box}
html{background:#e9ecf0}
body{margin:0;color:var(--ink);font:15px/1.55 "Segoe UI",system-ui,-apple-system,"Helvetica Neue",Arial,
"Liberation Sans",sans-serif;-webkit-font-smoothing:antialiased}
.page{max-width:900px;margin:24px auto;background:#fff;border:1px solid var(--rule);overflow:hidden}
.cover{background:var(--accent);color:#fff;padding:28px 48px 26px;border-bottom:3px solid var(--accent-dark)}
.brand{font-weight:700;letter-spacing:.08em;font-size:15px}.brand span{font-weight:400;letter-spacing:0;
color:#dbe6f7;margin-left:10px;font-size:13px}
.cover .date{color:#dbe6f7;font-size:13px;margin-top:22px}
.content{padding:8px 48px 40px}
h1.title{font-size:32px;line-height:1.2;margin:34px 0 4px}
.subtitle{font-size:19px;color:var(--muted);margin:0 0 10px}
.accent-rule{width:56px;height:3px;background:var(--accent);margin:14px 0 22px;border:0}
dl.meta{display:grid;grid-template-columns:120px 1fr;gap:4px 14px;margin:0 0 18px;font-size:14px}
dl.meta dt{color:var(--muted)}dl.meta dd{margin:0}
.toc{background:var(--panel);border-radius:6px;padding:14px 20px;margin:18px 0}
.toc h2{font-size:15px;margin:0 0 6px;border:0;padding:0}
.toc ol{list-style:none;margin:0;padding:0;columns:2;column-gap:28px}
.toc li{padding:2px 0}.toc a{color:var(--ink);text-decoration:none}.toc a:hover{color:var(--accent)}
.toc .n{display:inline-block;width:26px;color:var(--accent-dark);font-weight:700}
.disclaimer{font-size:12.5px;color:var(--muted);background:var(--panel);border-radius:6px;padding:10px 14px}
section{margin-top:34px}
h2.sec{font-size:23px;margin:0 0 12px;padding-bottom:8px;border-bottom:1px solid var(--rule);position:relative}
h2.sec:after{content:"";position:absolute;left:0;bottom:-1px;width:44px;height:2px;background:var(--accent)}
h2.sec .n{color:var(--accent);margin-right:12px}
h3{font-size:17px;margin:24px 0 8px}h3 .n{color:var(--accent-dark);margin-right:10px}
p{margin:8px 0}p.lead{font-size:16.5px}p.muted,ul.muted{color:var(--muted);font-size:14px}
p.small,ul.small{color:var(--muted);font-size:12.5px}
ul{margin:8px 0;padding-left:22px}li{margin:3px 0}li::marker{color:var(--accent)}
ul.small{list-style:none;padding-left:0}
.kv{display:grid;grid-template-columns:1fr 1fr;column-gap:26px;margin:10px 0}.kv.one{grid-template-columns:1fr}
.kv .row{display:grid;grid-template-columns:44% 56%;gap:6px;padding:5px 0;border-bottom:1px solid var(--rule);
font-size:14px}.kv .k{color:var(--muted)}
.kv-title{font-weight:700;margin:14px 0 2px;font-size:14.5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0 4px;font-variant-numeric:tabular-nums}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:13px;padding:10px 0 6px}
caption b{color:var(--ink)}
th{background:var(--panel);text-align:left;font-weight:700;padding:6px 8px;border-bottom:1.4px solid var(--ink)}
td{padding:5px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:nth-child(even) td{background:var(--zebra)}tbody tr:last-child td{border-bottom:1.4px solid var(--ink)}
.r{text-align:right}.c{text-align:center}
.t-ok{color:var(--ok);font-weight:600}.t-bad{color:var(--bad);font-weight:600}.t-warn{color:var(--warn);font-weight:600}
.t-info{color:var(--info)}.b{font-weight:700}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}
.kpi{border-top:1px solid var(--ink);padding:6px 2px 4px;position:relative}
.kpi .l{font-size:12px;color:var(--muted)}.kpi .v{font-size:19px;font-weight:600;line-height:1.3;
font-variant-numeric:tabular-nums}.kpi .u{font-size:13px;color:var(--muted);font-weight:400;margin-left:3px}
.kpi .s{font-size:12px;color:var(--muted)}
.badges{display:flex;flex-wrap:wrap;gap:2px 0;margin:8px 0;font-size:13px}
.badge{color:var(--muted)}.badge+.badge:before{content:"·";margin:0 8px;color:var(--faint)}
.badge.ok{color:var(--ok)}.badge.warn{color:var(--warn)}.badge.bad{color:var(--bad)}
.callout{border-radius:5px;padding:9px 14px;margin:10px 0;border-left:3px solid var(--info-bar);
background:var(--info-bg);font-size:14px}
.callout.ok{border-color:var(--ok-bar);background:var(--ok-bg)}.callout.warn{border-color:var(--warn-bar);
background:var(--warn-bg)}.callout.bad{border-color:var(--bad-bar);background:var(--bad-bg)}
figure{margin:16px 0 10px}figure img{width:100%;height:auto;display:block}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}figcaption b{color:var(--ink)}
footer{color:var(--faint);font-size:12px;border-top:1px solid var(--rule);margin-top:40px;padding-top:10px;
display:flex;justify-content:space-between}
@media (max-width:640px){.content,.cover{padding-left:16px;padding-right:16px}.kv{grid-template-columns:1fr}
.toc ol{columns:1}table{font-size:12px;display:block;overflow-x:auto}}
@page{size:A4;margin:16mm 15mm}
@media print{html{background:#fff}body{font-size:10.5pt}.page{box-shadow:none;margin:0;max-width:none;border-radius:0}
.content{padding:0 0 0}.cover{-webkit-print-color-adjust:exact;print-color-adjust:exact}
section.major{break-before:auto}h2.sec,h3{break-after:avoid}figure,.kpis,.callout,.kv .row,tr{break-inside:avoid}
.kpi,.badge,th,tbody tr:nth-child(even) td,.callout,.toc,.disclaimer{-webkit-print-color-adjust:exact;
print-color-adjust:exact}.toc a{color:var(--ink)}}
"""


def _inline(text: str) -> str:
    """``**bold**`` markup to escaped HTML."""
    return "".join(f"<b>{html.escape(t)}</b>" if b else html.escape(t) for t, b in runs(text))


def _cell(c: str | Cell, align: str) -> str:
    cls = []
    if align == "r":
        cls.append("r")
    elif align == "c":
        cls.append("c")
    if isinstance(c, Cell):
        if c.tone in ("ok", "bad", "warn", "info"):
            cls.append(f"t-{c.tone}")
        if c.bold:
            cls.append("b")
        text = c.text
    else:
        text = str(c)
    attr = f' class="{" ".join(cls)}"' if cls else ""
    return f"<td{attr}>{html.escape(text)}</td>"


def _figure_html(fb: FigureBlock, errors: list[str], dpi: float) -> str:
    try:
        fig = fb.build(style="print", figsize=fb.size)
        data = base64.b64encode(F.figure_to_png(fig, dpi=dpi)).decode("ascii")
    except Exception as exc:  # noqa: BLE001 - one broken figure must not sink the report
        errors.append(f"Figure {fb.number}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
        from .pdf import figure_failure_text
        return f'<div class="callout warn">{html.escape(figure_failure_text(fb, exc))}</div>'
    width = f' style="max-width:{fb.width * 100:.0f}%;margin:auto"' if fb.width < 1 else ""
    cap = (f"<figcaption><b>Figure {fb.number}.</b> {_inline(fb.caption)}</figcaption>"
           if fb.caption else "")
    alt = html.escape(fb.caption.split(".")[0] if fb.caption else "figure", quote=True)
    return f'<figure><img{width} alt="{alt}" src="data:image/png;base64,{data}">{cap}</figure>'


def _block_html(b: Any, errors: list[str], dpi: float) -> str:
    if isinstance(b, Heading):
        num = f'<span class="n">{html.escape(b.number)}</span>' if b.number else ""
        return f"<h3>{num}{html.escape(b.text)}</h3>"
    if isinstance(b, Paragraph):
        cls = f' class="{b.tone}"' if b.tone != "body" else ""
        return f"<p{cls}>{_inline(b.text)}</p>"
    if isinstance(b, Bullets):
        cls = f' class="{b.tone}"' if b.tone != "body" else ""
        return f"<ul{cls}>" + "".join(f"<li>{_inline(i)}</li>" for i in b.items) + "</ul>"
    if isinstance(b, KeyValues):
        rows = "".join(f'<div class="row"><div class="k">{html.escape(k)}</div><div>{html.escape(v)}</div></div>'
                       for k, v in b.rows)
        title = f'<div class="kv-title">{html.escape(b.title)}</div>' if b.title else ""
        return f'{title}<div class="kv{" one" if b.columns == 1 else ""}">{rows}</div>'
    if isinstance(b, Table):
        align = (b.align or []) + ["l"] * len(b.header)
        cap = ""
        if b.caption:
            label = f"<b>Table {b.number}.</b> " if b.number else ""
            cap = f"<caption>{label}{_inline(b.caption)}</caption>"
        head = "".join(f'<th class="{ {"r": "r", "c": "c"}.get(align[j], "") }">{html.escape(h)}</th>'
                       for j, h in enumerate(b.header))
        body = "".join("<tr>" + "".join(_cell(c, align[j]) for j, c in enumerate(r)) + "</tr>" for r in b.rows)
        return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    if isinstance(b, KPIRow):
        tiles = []
        for k in b.items:
            unit = f'<span class="u">{html.escape(k.unit)}</span>' if k.unit else ""
            sub = f'<div class="s">{html.escape(k.sub)}</div>' if k.sub else ""
            tiles.append(f'<div class="kpi {k.tone}"><div class="l">{html.escape(k.label)}</div>'
                         f'<div class="v">{html.escape(k.value)}{unit}</div>{sub}</div>')
        return f'<div class="kpis">{"".join(tiles)}</div>'
    if isinstance(b, Badges):
        return '<div class="badges">' + "".join(f'<span class="badge {t}">{html.escape(x)}</span>'
                                                for x, t in b.items) + "</div>"
    if isinstance(b, Callout):
        title = f"<b>{html.escape(b.title)}:</b> " if b.title else ""
        return f'<div class="callout {b.tone}">{title}{_inline(b.text)}</div>'
    if isinstance(b, FigureBlock):
        return _figure_html(b, errors, dpi)
    if isinstance(b, PageBreak):
        return '<div style="break-after:page"></div>'
    return ""


def render_html(doc: Document, path: str | Path, *, fig_dpi: float = FIG_DPI) -> dict:
    """Write ``doc`` as one self-contained HTML file. Returns ``{"errors": [...]}``."""
    path = Path(path)
    errors: list[str] = []
    meta = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in doc.meta)
    toc = "".join(f'<li><a href="#{s.anchor}"><span class="n">{html.escape(s.number)}</span>'
                  f"{html.escape(s.title)}</a></li>" for s in doc.sections)
    cover = _figure_html(doc.cover_figure, errors, fig_dpi) if doc.cover_figure is not None else ""
    parts = []
    for s in doc.sections:
        body = "".join(_block_html(b, errors, fig_dpi) for b in s.blocks)
        parts.append(f'<section id="{s.anchor}" class="major"><h2 class="sec"><span class="n">'
                     f"{html.escape(s.number)}</span>{html.escape(s.title)}</h2>{body}</section>")
    disclaimer = f'<p class="disclaimer">{html.escape(doc.disclaimer)}</p>' if doc.disclaimer else ""
    out = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="{html.escape(doc.software)}">
<meta name="author" content="{html.escape(doc.author)}">
<title>{html.escape(doc.title)} — {html.escape(doc.subtitle)}</title>
<style>{CSS}</style></head>
<body><div class="page">
<header class="cover"><div class="brand">AEROLUME<span>open-source aerodynamics</span></div>
<div class="date">{html.escape(doc.date)}</div></header>
<main class="content">
<h1 class="title">{html.escape(doc.title)}</h1>
<p class="subtitle">{html.escape(doc.subtitle)}</p><hr class="accent-rule">
<dl class="meta">{meta}</dl>
{cover}
<nav class="toc"><h2>Contents</h2><ol>{toc}</ol></nav>
{disclaimer}
{"".join(parts)}
<footer><span>{html.escape(doc.software)} · open-source aerodynamics</span>
<span>{html.escape(doc.date)}</span></footer>
</main></div></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    return {"errors": errors}
