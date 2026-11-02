"""PDF and HTML reports (DESIGN.md section 5.6b).

``generate(project, sim_ids, path, fmt="pdf"|"html", sections=None, title=None,
author=None) -> Path`` writes a professional, self-explaining report of one or more
simulations: summary with key numbers and validity badges, geometry, setup,
results, surface distributions, flow fields, convergence, sensitivity, polars,
comparison, methodology and an appendix with run details.

The content is assembled once (:func:`.content.build_document`) and rendered by
either back end, so the PDF and the HTML say exactly the same thing. Both run
headless: matplotlib's Agg/PDF backends only, no Qt, no pyplot, no torch.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .content import SECTION_TITLES, SECTIONS, build_document
from .document import Document

__all__ = ["SECTIONS", "SECTION_TITLES", "Document", "build_document", "generate"]

log = logging.getLogger(__name__)

_FORMATS = {"pdf": ".pdf", "html": ".html"}


def generate(project: Any, sim_ids: Sequence[str], path: str | Path, *, fmt: str = "pdf",
             sections: Iterable[str] | None = None, title: str | None = None,
             author: str | None = None) -> Path:
    """Write a report of ``sim_ids`` from ``project`` to ``path`` and return the written path.

    ``fmt`` is "pdf" (A4, print-ready) or "html" (one self-contained file). The suffix is
    added when ``path`` has none. ``sections`` selects from :data:`SECTIONS` (default: all
    that apply). Unknown simulation ids raise ``KeyError``; unknown sections or formats raise
    ``ValueError``. A figure that cannot be drawn from the stored data is replaced by a note
    in the report and logged, rather than failing the whole report.
    """
    fmt = str(fmt).lower()
    if fmt not in _FORMATS:
        raise ValueError(f"unknown report format {fmt!r}; use 'pdf' or 'html'")
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(_FORMATS[fmt])
    doc = build_document(project, list(sim_ids), sections=sections, title=title, author=author,
                         filename=path.name)
    if fmt == "pdf":
        from .pdf import render_pdf
        info = render_pdf(doc, path)
    else:
        from .html import render_html
        info = render_html(doc, path)
    for err in info.get("errors", []):
        log.warning("report figure failed: %s", err)
    return path
