"""Build the Aerolume website from the repository.

Usage::

    python website/build.py                       # writes website/index.html and website/docs/
    python website/build.py --dist _site          # also assembles a clean deploy folder
    python website/build.py --strict             # exit 1 if any internal link or anchor is broken
    python website/build.py --artifact DIR        # also writes a claude.ai Artifact copy into DIR
    python website/build.py --site-url URL        # absolute og:image / og:url (social previews)

The site is static, hand-built HTML and CSS. This script does five things:

1. Fills ``templates/index.html`` with facts that live elsewhere in the repository
   (version, repository URL, the flow-figure conditions in ``assets/figures.json``,
   the screenshots listed in ``assets/screens/MANIFEST.md``, the benchmark numbers and
   the 3D throughput from the documentation's tables, the README's list of limits, and
   the pass count and selected checks of the last full validation run from
   ``validation/records/validation-record.json``, the record the report's section 9 was
   written from), so the home page cannot drift from the code or from the docs. The hero's live flow is ``assets/hero/hero.js``, plain JavaScript the page loads
   as it is (no bundler, no external module).
2. Converts the repository's Markdown (README, docs/*.md, CONTRIBUTING, notices) into
   the ``docs/`` pages with a small converter written for the subset those files use.
   A general Markdown library would be a dependency for the Pages build and would
   still need the link rewriting, heading anchors, cross-references and equation
   typesetting done here.
3. Copies LICENSE as ``LICENSE.txt`` so the licence link never leaves the site.
4. Checks every relative link and ``#anchor`` in the output.
5. On request, writes the Artifact copy (``index.html`` reduced to its body content,
   because a claude.ai Artifact supplies the document skeleton itself, with the site's
   own CSS and scripts inlined so it renders and animates as one page). It is written
   outside ``website/`` so a stale second copy is never deployed.

The repository URL is not guessed. It comes from ``[project.urls]`` in pyproject.toml,
the ``AEROLUME_REPO_URL`` variable, GitHub Actions' own ``GITHUB_REPOSITORY`` or
``repository-code`` in CITATION.cff (the site's own address from ``url`` there); if none
is set the build warns, leaves out the links that need it and says in plain words that
the source is not public yet, instead of printing an address that may not exist.

Constraints the output keeps (tests/test_build.py checks the mechanical ones):
the only external request is a pinned CDN script (MathJax, only on pages with ``$...$``
maths; the fonts are self-hosted); colours are theme tokens defined for light, for
``prefers-color-scheme: dark`` and for ``[data-theme="dark"]``; every internal link is
relative; the layout works at 390 px without horizontal page scroll.

Only the standard library is used, so the GitHub Pages workflow needs no installs.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import os
import re
import shutil
import sys
import tomllib
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: One family for the whole site (the desktop application's own typeface, docs/ui_guidelines.md)
#: plus its monospace for numbers and code, self-hosted from assets/fonts/ (OFL); site.css holds
#: the @font-face rules and every page preloads the upright text face, the one above the fold.
FONT_PRELOAD = "assets/fonts/atkinson-hyperlegible-next-latin-wght-normal.woff2"
#: The only external request the site makes: MathJax, pinned, on pages with $...$ maths.
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"

SITE_DESCRIPTION = ("Open-source aerodynamics and heat-transfer studio: GPU Navier-Stokes solvers in 2D and "
                    "3D with smoke emitters, heated bodies, fans and rooms, an XFOIL-class airfoil solver and "
                    "finite wings, in one guided desktop workflow.")


@dataclass(frozen=True)
class DocSpec:
    """One documentation page: where its Markdown lives and how the site files it."""
    source: str          # repository-relative path of the Markdown file
    slug: str            # output file name without .html
    title: str           # name in the navigation and the <title>
    group: str           # navigation group
    summary: str         # one line for the docs index


#: Navigation order. Files that do not exist (or are empty) at build time are skipped
#: with a warning, so pages appear on the site as soon as their Markdown lands.
DOCS: tuple[DocSpec, ...] = (
    DocSpec("README.md", "overview", "Overview", "Guide",
            "What Aerolume is, how to install it and how to run a first analysis."),
    DocSpec("docs/install.md", "install", "Installing Aerolume", "Guide",
            "The Windows installer, the portable folder and a source install; building the installer."),
    DocSpec("docs/user_guide.md", "user-guide", "User guide", "Guide",
            "The pages of the application, 3D, placed physics, explore, scripting, the Assistant, accounts."),
    DocSpec("docs/user_guide_simulation.md", "simulation-guide", "Simulation guide", "Guide",
            "How a run is planned and executed, and how to read its results."),
    DocSpec("docs/scripting.md", "scripting", "Scripting API", "Guide",
            "import aerolume as al: designs, runs, sweeps, plots and exports from Python."),
    DocSpec("docs/ai_assistant.md", "assistant", "Experiment assistant", "Guide",
            "Planning, running and analysing experiments with Claude or a local model, and its rules."),
    DocSpec("docs/mcp.md", "mcp", "MCP server", "Guide",
            "Driving Aerolume from Claude Desktop or Claude Code through aerolume-mcp."),
    DocSpec("docs/accounts.md", "accounts", "Accounts (optional)", "Guide",
            "Connecting a Supabase project of your own: sign-in, cloud projects and share links."),
    DocSpec("docs/methods.md", "methods", "Methods", "Theory",
            "Every method in one place, with its references: the summary to cite."),
    DocSpec("docs/theory_ns2d.md", "theory-ns2d", "Navier–Stokes solver (2D)", "Theory",
            "Equations, discretisation, immersed boundaries, forces, heat transfer, fans, performance and limits."),
    DocSpec("docs/theory_ns3d.md", "theory-ns3d", "Navier–Stokes solver (3D)", "Theory",
            "The 3D solver: voxelised meshes, symmetry planes, moving roads, placed physics, verification."),
    DocSpec("docs/theory_airfoil.md", "theory-airfoil", "Airfoil analysis", "Theory",
            "The viscous–inviscid polar method, its inputs and where its results stop being valid."),
    DocSpec("docs/theory_wing.md", "theory-wing", "Finite wings", "Theory",
            "Vortex lattice and nonlinear lifting line on the section polars."),
    DocSpec("docs/theory_sensitivity.md", "theory-sensitivity", "Sensitivity maps", "Theory",
            "Shape sensitivities by reverse-mode differentiation through the solver, in 2D and 3D."),
    DocSpec("docs/validation.md", "validation", "Validation report", "Validation",
            "Exact answers, flow and heat-transfer benchmarks, fans and jets, polars, wings and performance."),
    DocSpec("docs/validation_airfoil.md", "validation-airfoil", "Airfoil validation", "Validation",
            "Polars compared with NACA 0012, 2412 and 4412 wind-tunnel data."),
    DocSpec("docs/validation_wing.md", "validation-wing", "Finite-wing validation", "Validation",
            "Lifting-line and lifting-surface theory, Weber & Brebner and NACA TN 1270 wings."),
    DocSpec("docs/ui_guidelines.md", "design-guidelines", "Design guidelines", "Project",
            "The visual and interaction rules of the desktop application."),
    DocSpec("CONTRIBUTING.md", "contributing", "Contributing", "Project",
            "Development setup and the rules every change follows."),
    DocSpec("THIRD_PARTY_NOTICES.md", "third-party-notices", "Third-party notices", "Project",
            "Software Aerolume includes and the licences it is used under."),
)
GROUP_ORDER = ("Guide", "Theory", "Validation", "Project")
#: How the docs index describes each group, so its intro names only what was built.
GROUP_PHRASES = {
    "Guide": "how to use the program",
    "Theory": "the theory behind every solver with its verification",
    "Validation": "the validation against published data",
    "Project": "the project&rsquo;s own rules",
}

#: Navier-Stokes throughput. The build reads the benchmark table of docs/theory_ns2d.md
#: (section 10, written by validation/bench_ns2d.py) so the home page always shows the
#: published numbers; this copy (2026-10-05, float32, cylinder case, best of 3) is used
#: only if that table cannot be found.
#: (device name, "GPU" | "CPU", grid nx, grid ny, configuration, steps per second)
PERF_FALLBACK: tuple[tuple[str, str, int, int, str, float], ...] = (
    ("RTX 4060 Ti", "GPU", 1024, 512, "default (upwind3, laminar, CUDA graphs)", 493.0),
    ("RTX 4060 Ti", "GPU", 1024, 512, "eager (no graphs)", 59.0),
    ("RTX 4060 Ti", "GPU", 1024, 512, "central advection", 619.0),
    ("RTX 4060 Ti", "GPU", 1024, 512, "upwind3 + WALE", 317.0),
    ("RTX 4060 Ti", "GPU", 1024, 512, "upwind3 + Smagorinsky", 373.0),
    ("RTX 4060 Ti", "GPU", 2048, 1024, "default", 75.0),
    ("i7-13700F", "CPU", 512, 256, "1 thread", 48.0),
    ("i7-13700F", "CPU", 512, 256, "4 threads", 61.0),
    ("i7-13700F", "CPU", 512, 256, "16 threads (torch default here)", 36.0),
)


def warn(msg: str) -> None:
    """Print a build warning (stderr), visible in CI logs."""
    print(f"warning: {msg}", file=sys.stderr)


# =============================================================================
# Markdown
# =============================================================================

_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*([^`\s]*)[^`]*$")
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_HR = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_BULLET = re.compile(r"^( {0,3})([-*+])( {1,4}(?=\S)| *$)(.*)$")
_ORDERED = re.compile(r"^( {0,3})(\d{1,9})([.)])( {1,4}(?=\S)| *$)(.*)$")
_QUOTE = re.compile(r"^ {0,3}> ?(.*)$")
_TABLE_DELIM = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")
_MATH_OPEN = re.compile(r"^ {0,3}\$\$")
_PLACEHOLDER = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_COMMENT_OPEN = re.compile(r"^ {0,3}<!--")
_LINK = re.compile(r'\[((?:[^\[\]]|\[[^\]]*\])*)\]\(\s*([^)\s]+)(?:\s+"([^"]*)")?\s*\)')
_IMAGE = re.compile(r'!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+"([^"]*)")?\s*\)')
_INLINE_HTML = re.compile(r"</?(sub|sup|kbd|br|b|i|em|strong|small|abbr)\s*/?>", re.IGNORECASE)

#: Characters that only occur in mathematics in these documents (Greek, operators, fractions,
#: sub/superscript digits). A code span or an unlabelled fence containing them is an
#: equation written in plain text, not code.
_MATH_CHARS = re.compile("[α-ωΑ-Ω∂∇√²³·≤≥−∮∑Σ½⅓⅔¾⅙←→∞≈±×∈‖′₀-₉ϑϕ]")
#: ...unless it also contains a programming identifier, a path or a keyword argument.
_CODE_SIGNS = re.compile(r"[A-Za-z]{2,}_[A-Za-z]{2,}|\.(?:py|md|json|toml)\b|\w/\w+/|\w\([^)]*\w=")
_BOX_DRAWING = re.compile("[─-╿]")
#: A bare mention of a repository Markdown file, optionally with a section number.
_DOC_MENTION = re.compile(r"(?<![\w/.-])((?:docs/)?[A-Za-z0-9_]+\.md)"
                          r"(?:(?:,?\s+section\s+|\s*&sect;\s*|\s*§\s*)(\d+(?:\.\d+)?))?(?![\w-])")

Xref = Callable[[str, str | None], tuple[str, str] | None]


@dataclass
class TocEntry:
    """A heading that appears in the page's table of contents."""
    level: int
    id: str
    text: str


@dataclass
class Rendered:
    """Result of converting one Markdown document."""
    html: str
    title: str
    toc: list[TocEntry] = field(default_factory=list)
    has_math: bool = False


def slugify(text: str) -> str:
    """GitHub-style heading id: lower case, punctuation dropped, spaces and dashes to '-'."""
    # NFKC folds superscripts and similar compatibility forms ("10⁶" -> "106"), so ids stay
    # typeable and match what readers paste into a URL.
    s = unicodedata.normalize("NFKC", html.unescape(re.sub(r"<[^>]+>", "", text))).lower()
    s = re.sub(r"[‐-―−]", "-", s)       # hyphens, en/em dashes, minus
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "section"


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def _attr(text: str) -> str:
    return html.escape(html.unescape(text), quote=True)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


# ----------------------------------------------------------------------------- plain-text maths
def is_math_span(text: str) -> bool:
    """True if a code span holds an equation (``ν_t = (C_s Δ)² √(2 S:S)``), not code.

    The documents write formulas in code spans; shown as code they read as program text
    and their ``^{...}`` / ``_{...}`` braces appear literally. Identifiers with
    underscores (``force_ref``), paths and keyword arguments keep a span as code.
    """
    if _CODE_SIGNS.search(text):
        return False
    if "^{" in text or "_{" in text or _MATH_CHARS.search(text):
        return True
    # a short subscripted symbol (u_s, X_k, C_w) and no longer identifiers: notation, not a name
    return bool(re.search(r"(?<![A-Za-z0-9_])[A-Za-z]{1,2}_[A-Za-z0-9]{1,3}(?![A-Za-z0-9_])", text))


def _match_brace(s: str, i: int) -> int:
    """Index of the ``}`` that closes the ``{`` at ``s[i]``, or -1."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _italicise(escaped: str) -> str:
    """Set single Latin letters (the variables) in italics, as typeset mathematics does.

    Multi-letter words (``Re``, ``lerp``, ``max``) stay upright, like operator names.
    The text is already HTML-escaped; entity names never stand alone between non-letters.
    """
    return re.sub(r"(?<![A-Za-z0-9&#])([A-Za-z])(?![A-Za-z0-9;])", r"<i>\1</i>", escaped)


def format_math(text: str) -> str:
    """Typeset a plain-text formula as HTML.

    ``x^{n+1}`` / ``x_{N/2}`` (braced, nested) and ``x^n`` / ``ν_t`` / ``Σ_faces`` (a run
    of letters and digits) become superscripts and subscripts; single Latin letters
    become italic variables. Everything else is escaped and kept as written.
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(_italicise(_esc("".join(buf))))
            buf.clear()

    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "^_" and i > 0 and i + 1 < n and not text[i - 1].isspace():
            tag = "sup" if c == "^" else "sub"
            if text[i + 1] == "{":
                j = _match_brace(text, i + 1)
                if j > 0:
                    flush()
                    out.append(f"<{tag}>{format_math(text[i + 2:j])}</{tag}>")
                    i = j + 1
                    continue
            else:
                m = re.match(r"[A-Za-z0-9α-ωΑ-Ω′]+", text[i + 1:])
                if m:
                    flush()
                    out.append(f"<{tag}>{format_math(m.group(0))}</{tag}>")
                    i += 1 + len(m.group(0))
                    continue
        buf.append(c)
        i += 1
    flush()
    return "".join(out)


_CODE_LINE = re.compile(r"\.(?:py|md|json|toml)\b|\w/\w+/|\w\([^)]*\w=|^\s*(?:def|class|import|from|for|if|"
                        r"return|pip|python|git|cd)\b|[;{}]\s*$")


def is_equation_block(lines: list[str]) -> bool:
    """An unlabelled fence whose every line is a formula (see :func:`equation_html`).

    Subscripted names such as ``Cd_cv`` are normal in formulas, so only statements,
    paths and keyword arguments mark a block as code here.
    """
    body = [ln for ln in lines if ln.strip()]
    return bool(body) and all(_MATH_CHARS.search(ln) for ln in body) \
        and not any(_CODE_LINE.search(ln) or _BOX_DRAWING.search(ln) for ln in body)


def equation_html(lines: list[str]) -> str:
    """Typeset a plain-text display equation as an aligned table.

    Each line splits at its first relation (``=``, ``←``) into left side, relation and
    right side, so a system of equations lines up on its relations the way a typeset
    ``align`` does; a line that starts with ``=`` continues the previous left side (the
    cases of a piecewise definition). Runs of three or more spaces separate trailing
    columns such as conditions. The plain text stays available to screen readers.
    """
    rows: list[list[str]] = []
    for ln in lines:
        if not ln.strip():
            continue
        body = ln.strip()
        m = re.match(r"^(\S.*?)\s+(=|←|≤|≥|≈)\s+(.*)$", body)
        if body[:1] in "=←":                                   # continuation: same left side
            lhs, rel, rest = "", body[0], body[1:].strip()
        elif m and not re.search(r"\s{3,}", m.group(1)):      # aligned "name   = ..." counts too
            lhs, rel, rest = m.groups()
        else:
            lhs, rel, rest = "", "", body
        segs = re.split(r"\s{3,}", rest)
        rows.append([lhs, rel, *segs])
    ncol = max(len(r) for r in rows)
    plain = " ; ".join(" ".join(x for x in ln.split()) for ln in lines if ln.strip())
    plain = re.sub(r"([\^_])\{([^{}]*)\}", r"\1(\2)", plain)    # u^{n+1} -> u^(n+1) for screen readers
    trs = []
    for r in rows:
        cells = []
        for k, c in enumerate(r + [""] * (ncol - len(r))):
            cls = ("l", "rel", "r")[k] if k < 3 else "cond"
            cells.append(f'<td class="{cls}">{format_math(c)}</td>')
        trs.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="eq" role="math" aria-label="{_attr(plain)}"><table role="presentation">'
            + "".join(trs) + "</table></div>")


def diagram_html(lines: list[str]) -> str:
    """A box-drawing diagram with every non-ASCII glyph held to one character cell.

    Box-drawing characters exist in the code font, but symbols such as ▶ ■ ▾ ● fall back
    to fonts with other advance widths and shift the rest of their line, so the frame no
    longer closes. Wrapping each such glyph in a 1ch inline block keeps the columns.
    """
    def cell(ch: str) -> str:
        if ord(ch) < 128 or _BOX_DRAWING.match(ch):
            return _esc(ch)
        return f'<span class="g">{_esc(ch)}</span>'
    body = "\n".join("".join(cell(ch) for ch in ln) for ln in lines)
    return f'<pre class="diagram"><code>{body}</code></pre>'


class MarkdownRenderer:
    """Converts the Markdown subset used by Aerolume's documents into HTML.

    Supported: ATX headings (with unique ids), paragraphs, ``*``/``_`` emphasis, inline
    code, fenced and indented code, pipe tables with alignment, nested ordered and
    unordered lists (tight and loose), block quotes, horizontal rules, links, autolinks,
    images (a lone image becomes a ``<figure>``), a few inline HTML tags (sub, sup,
    kbd, br, ...) and ``$...$`` / ``$$...$$`` maths, passed through to MathJax as
    ``\\(...\\)`` / ``\\[...\\]``. Everything else is escaped, never passed through.

    Plain-text maths, the way the current documents write it, is typeset too: a code
    span that holds a formula renders as ``<span class="math">`` with sub/superscripts
    and italic variables, an unlabelled fence of formulas as an aligned equation, and a
    box-drawing fence as a diagram whose glyphs keep their columns.

    ``link`` maps every link and image target (e.g. ``theory_airfoil.md#2-inputs``)
    to the URL used on the site, or to ``None`` to show the text without a link.
    ``xref`` turns a bare mention of a document (``theory_airfoil.md, section 5``) or a
    link whose text is a file name into a link labelled with the page title.
    ``figure`` may replace the HTML of a lone image (the web versions of the plots).
    A paragraph that is a bare ``UPPER_CASE`` token is a table placeholder a generator
    has not filled yet; it renders as a visible "pending" note instead of a stray word.
    """

    def __init__(self, link: Callable[[str], str | None] | None = None, *, xref: Xref | None = None,
                 figure: Callable[[str, str, str], str | None] | None = None,
                 size: Callable[[str], tuple[int, int] | None] | None = None) -> None:
        self.link = link or (lambda url: url)
        self.xref = xref
        self.figure = figure
        self.size = size
        self._ids: dict[str, int] = {}
        self._toc: list[TocEntry] = []
        self._title = ""
        self._math = False

    # ------------------------------------------------------------------ public
    def render(self, text: str) -> Rendered:
        """Convert a whole document."""
        self._ids, self._toc, self._title, self._math = {}, [], "", False
        lines = text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).split("\n")
        body = "\n".join(self._blocks(lines))
        return Rendered(body + "\n" if body else "", self._title, list(self._toc), self._math)

    # ------------------------------------------------------------------ blocks
    def _starts_block(self, line: str) -> bool:
        """True if ``line`` would interrupt a paragraph."""
        return bool(_FENCE.match(line) or _HEADING.match(line) or _HR.match(line) or _QUOTE.match(line)
                    or _MATH_OPEN.match(line) or self._marker(line) or _COMMENT_OPEN.match(line))

    @staticmethod
    def _marker(line: str) -> tuple[bool, int, int, str, int, str] | None:
        """List marker: (ordered, indent, content offset, bullet char / delimiter, start, text)."""
        m = _BULLET.match(line)
        if m and not _HR.match(line):
            indent, ch, gap, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            if not rest.strip() and not gap:
                gap = " "
            return False, len(indent), len(indent) + 1 + max(len(gap), 1), ch, 1, rest
        m = _ORDERED.match(line)
        if m:
            indent, num, delim, gap, rest = m.groups()
            return True, len(indent), len(indent) + len(num) + 1 + max(len(gap), 1), delim, int(num), rest
        return None

    def _blocks(self, lines: list[str], tight: bool = False) -> list[str]:
        out: list[str] = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            m = _FENCE.match(line)
            if m:
                i = self._fenced(lines, i, m, out)
                continue
            if _MATH_OPEN.match(line):
                i = self._math_block(lines, i, out)
                continue
            m = _HEADING.match(line)
            if m:
                self._heading(len(m.group(1)), m.group(2) or "", out)
                i += 1
                continue
            if _HR.match(line):
                out.append("<hr>")
                i += 1
                continue
            if _COMMENT_OPEN.match(line):
                while i < n and "-->" not in lines[i]:
                    i += 1
                i += 1
                continue
            if _QUOTE.match(line):
                i = self._quote(lines, i, out)
                continue
            if self._marker(line):
                i = self._list(lines, i, out)
                continue
            if _indent(line) >= 4:
                i = self._indented_code(lines, i, out)
                continue
            if "|" in line and i + 1 < n and _TABLE_DELIM.match(lines[i + 1]) and "-" in lines[i + 1]:
                i = self._table(lines, i, out)
                continue
            i = self._paragraph(lines, i, out, tight)
        return out

    def _fenced(self, lines: list[str], i: int, m: re.Match, out: list[str]) -> int:
        pad, fence, lang = len(m.group(1)), m.group(2), m.group(3)
        close = re.compile(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
        body: list[str] = []
        i += 1
        while i < len(lines) and not close.match(lines[i]):
            ln = lines[i]
            body.append(ln[min(pad, _indent(ln)):])
            i += 1
        if i >= len(lines):                         # unclosed fence: runs to the end, minus trailing blanks
            while body and not body[-1].strip():
                body.pop()
        if not lang and is_equation_block(body):
            out.append(equation_html(body))
        elif not lang and any(_BOX_DRAWING.search(ln) for ln in body):
            out.append(diagram_html(body))
        else:
            cls = f' class="language-{_attr(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{_esc(chr(10).join(body))}</code></pre>")
        return i + 1

    def _indented_code(self, lines: list[str], i: int, out: list[str]) -> int:
        body: list[str] = []
        while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) >= 4):
            body.append(lines[i][4:])
            i += 1
        while body and not body[-1].strip():
            body.pop()
        out.append(f"<pre><code>{_esc(chr(10).join(body))}</code></pre>")
        return i

    def _math_block(self, lines: list[str], i: int, out: list[str]) -> int:
        first = lines[i].strip()[2:]
        parts: list[str] = []
        if first.rstrip().endswith("$$"):                     # $$ ... $$ on one line
            parts.append(first.rstrip()[:-2])
            i += 1
        else:
            parts.append(first)
            i += 1
            while i < len(lines):
                s = lines[i].rstrip()
                if s.endswith("$$"):
                    parts.append(s[:-2])
                    i += 1
                    break
                parts.append(s)
                i += 1
        self._math = True
        tex = "\n".join(p for p in parts if p.strip())
        out.append(f'<div class="math-block">\\[{_esc(tex)}\\]</div>')
        return i

    def _heading(self, level: int, text: str, out: list[str]) -> None:
        inner = self.inline(text.strip())
        plain = html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()
        base = slugify(plain)
        count = self._ids.get(base, 0)
        self._ids[base] = count + 1
        hid = base if count == 0 else f"{base}-{count}"
        if level == 1:
            if not self._title:
                self._title = plain
            out.append(f'<h1 id="{hid}">{inner}</h1>')
            return
        if level in (2, 3):
            self._toc.append(TocEntry(level, hid, plain))
        anchor = f'<a class="anchor" href="#{hid}" aria-label="Link to this section">#</a>'
        out.append(f'<h{level} id="{hid}">{inner}{anchor}</h{level}>')

    def _quote(self, lines: list[str], i: int, out: list[str]) -> int:
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            m = _QUOTE.match(lines[i])
            if m:
                body.append(m.group(1))
            elif self._starts_block(lines[i]):
                break
            else:                                           # lazy continuation
                body.append(lines[i])
            i += 1
        out.append("<blockquote>\n" + "\n".join(self._blocks(body)) + "\n</blockquote>")
        return i

    def _list(self, lines: list[str], i: int, out: list[str]) -> int:
        n = len(lines)
        first = self._marker(lines[i])
        assert first is not None
        ordered, base_indent, _, kind, start, _ = first
        items: list[list[str]] = []
        loose = False
        while i < n:
            mk = self._marker(lines[i])
            if not mk or mk[0] != ordered or mk[3] != kind or _indent(lines[i]) > base_indent + 3:
                break
            offset = mk[2]
            item = [mk[5]]
            i += 1
            blank = False
            while i < n:
                ln = lines[i]
                if not ln.strip():
                    item.append("")
                    blank = True
                    i += 1
                    continue
                if _indent(ln) >= offset:
                    item.append(ln[offset:])
                    blank = False
                    i += 1
                    continue
                if blank or self._starts_block(ln):
                    break
                item.append(ln.strip())                     # lazy paragraph continuation
                i += 1
            while item and not item[-1].strip():
                item.pop()
            if any(not ln.strip() for ln in item[1:]) and not self._only_code_blanks(item):
                loose = True
            items.append(item)
            if blank and i < n:
                nxt = self._marker(lines[i])
                if nxt and nxt[0] == ordered and nxt[3] == kind:
                    loose = True
        tag = "ol" if ordered else "ul"
        start_attr = f' start="{start}"' if ordered and start != 1 else ""
        parts = []
        for item in items:
            inner = self._blocks(item, tight=not loose)
            if len(inner) == 1 and not inner[0].startswith("<"):
                parts.append(f"<li>{inner[0]}</li>")
            else:
                parts.append("<li>" + "\n".join(inner) + "</li>")
        out.append(f"<{tag}{start_attr}>\n" + "\n".join(parts) + f"\n</{tag}>")
        return i

    @staticmethod
    def _only_code_blanks(item: list[str]) -> bool:
        """True if every blank line of a list item sits inside a fenced code block."""
        inside = False
        for ln in item:
            if _FENCE.match(ln):
                inside = not inside
            elif not ln.strip() and not inside:
                return False
        return True

    @staticmethod
    def split_row(line: str) -> list[str]:
        """Split a table row on pipes, ignoring pipes in code spans or escaped as ``\\|``."""
        s = line.strip()
        s = s.removeprefix("|")
        if s.endswith("|") and not s.endswith("\\|"):
            s = s[:-1]
        cells, buf, i, tick = [], [], 0, 0
        while i < len(s):
            c = s[i]
            if c == "\\" and i + 1 < len(s) and s[i + 1] == "|":
                buf.append("|" if not tick else "\\|")
                i += 2
                continue
            if c == "`":
                j = i
                while j < len(s) and s[j] == "`":
                    j += 1
                run = j - i
                if tick == 0:
                    # open only if a matching run closes it later on this row
                    if re.search(r"(?<!`)" + "`" * run + r"(?!`)", s[j:]):
                        tick = run
                elif run == tick:
                    tick = 0
                buf.append(s[i:j])
                i = j
                continue
            if c == "|" and not tick:
                cells.append("".join(buf).strip())
                buf = []
            else:
                buf.append(c)
            i += 1
        cells.append("".join(buf).strip())
        return cells

    def _table(self, lines: list[str], i: int, out: list[str]) -> int:
        header = self.split_row(lines[i])
        aligns = []
        for cell in self.split_row(lines[i + 1]):
            c = cell.strip()
            aligns.append("c" if c.startswith(":") and c.endswith(":") else
                          "r" if c.endswith(":") else "l" if c.startswith(":") else "")
        ncol = len(header)
        aligns = (aligns + [""] * ncol)[:ncol]
        i += 2
        rows = []
        while i < len(lines) and lines[i].strip() and "|" in lines[i] and not self._starts_block(lines[i]):
            cells = self.split_row(lines[i])
            rows.append((cells + [""] * ncol)[:ncol])
            i += 1

        def cell(tag: str, text: str, a: str) -> str:
            cls = f' class="{a}"' if a in ("r", "c") else ""
            scope = ' scope="col"' if tag == "th" else ""
            return f"<{tag}{cls}{scope}>{self.inline(text)}</{tag}>"

        head = "".join(cell("th", h, a) for h, a in zip(header, aligns, strict=True))
        body = "\n".join("<tr>" + "".join(cell("td", c, a) for c, a in zip(r, aligns, strict=True)) + "</tr>"
                         for r in rows)
        out.append(f'<div class="table-scroll"><table>\n<thead><tr>{head}</tr></thead>\n'
                   f"<tbody>\n{body}\n</tbody>\n</table></div>")
        return i

    def _paragraph(self, lines: list[str], i: int, out: list[str], tight: bool) -> int:
        def keep_break(raw: str) -> str:
            # two trailing spaces are a hard line break; everything else is trimmed
            return raw.strip() + ("  " if raw.endswith("  ") else "")

        buf = [keep_break(lines[i])]
        i += 1
        while i < len(lines) and lines[i].strip() and not self._starts_block(lines[i]):
            if "|" in lines[i] and i + 1 < len(lines) and _TABLE_DELIM.match(lines[i + 1]) \
                    and "-" in lines[i + 1]:
                break
            buf.append(keep_break(lines[i]))
            i += 1
        text = "\n".join(buf).rstrip()
        if _PLACEHOLDER.match(text):
            out.append(f'<p class="placeholder">Pending: this table ({_esc(text)}) is generated by the '
                       "validation scripts and has not been filled in yet.</p>")
            return i
        img = re.fullmatch(r'!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"([^"]*)")?\s*\)', text)
        if img and not tight:
            alt = img.group(1)
            src = self.link(img.group(2)) or img.group(2)
            custom = self.figure(img.group(2), src, alt) if self.figure else None
            if custom:
                out.append(custom)
                return i
            cap = f"<figcaption>{self.inline(alt)}</figcaption>" if alt else ""
            out.append(f'<figure><img src="{_attr(src)}" alt="{_attr(alt)}"{self._dims(src)} loading="lazy">{cap}</figure>')
            return i
        inner = self.inline(text)
        out.append(inner if tight else f"<p>{inner}</p>")
        return i

    def _dims(self, src: str) -> str:
        """width/height attributes for an image whose size is known, so the browser reserves its
        box before the file arrives (CSS keeps height:auto, so it still scales): without them a
        docs figure grew from 52 to 412 px after load and pushed the page down (CLS 0.094)."""
        wh = self.size(src) if self.size else None
        return f' width="{wh[0]}" height="{wh[1]}"' if wh else ""

    # ------------------------------------------------------------------ inline
    def inline(self, text: str) -> str:
        """Convert inline Markdown (code, maths, links, emphasis) to HTML."""
        stash: list[str] = []

        def keep(fragment: str) -> str:
            stash.append(fragment)
            return f"\x00{len(stash) - 1}\x00"

        def code(m: re.Match) -> str:
            c = m.group(2).replace("\n", " ")
            if len(c) >= 2 and c[0] == " " and c[-1] == " " and c.strip():
                c = c[1:-1]
            if is_math_span(c):
                return keep(f'<span class="math">{format_math(c)}</span>')
            escaped = _esc(c)
            if len(c) > 24 and " " not in c:
                # A long path or identifier may break only after a separator on a phone, never
                # before its last letter (overflow-wrap: anywhere breaks at any character).
                escaped = re.sub(r"([/_])(?=\w)", r"\1<wbr>", escaped)
            return keep(f"<code>{escaped}</code>")

        def math(m: re.Match, display: bool) -> str:
            self._math = True
            tex = _esc(m.group(1))
            return keep(f"\\[{tex}\\]" if display else f"\\({tex}\\)")

        text = re.sub(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", code, text, flags=re.DOTALL)
        text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|$<>~])", lambda m: keep(_esc(m.group(1))), text)
        text = re.sub(r"\$\$(.+?)\$\$", lambda m: math(m, True), text, flags=re.DOTALL)
        text = re.sub(r"(?<![\w$])\$(?=[^\s$])(.+?)(?<=[^\s$])\$(?![\w$])", lambda m: math(m, False), text,
                      flags=re.DOTALL)
        text = re.sub(r"<((?:https?|mailto):[^>\s]+)>", self._autolink(keep), text)
        text = _INLINE_HTML.sub(lambda m: keep(m.group(0).lower().replace(" ", "")), text)
        text = re.sub(r"  \n", lambda m: keep("<br>\n"), text)
        text = _esc(text)

        def image(m: re.Match) -> str:
            raw = html.unescape(m.group(2))
            src = _attr(self.link(raw) or raw)
            title = f' title="{_attr(m.group(3))}"' if m.group(3) else ""
            return keep(f'<img src="{src}" alt="{_attr(m.group(1))}"{title}{self._dims(html.unescape(src))} loading="lazy">')

        def link(m: re.Match) -> str:
            target = html.unescape(m.group(2))
            label = self._emphasis(m.group(1))
            if self.xref and re.fullmatch(r"(?:docs/)?[\w.-]+\.md", html.unescape(m.group(1))):
                ref = self.xref(html.unescape(m.group(1)), None)
                if ref:
                    label = _esc(ref[1])
            href = self.link(target)
            if href is None:
                return keep(label)
            title = f' title="{_attr(m.group(3))}"' if m.group(3) else ""
            return keep(f'<a href="{_attr(href)}"{title}>{label}</a>')

        def mention(m: re.Match) -> str:
            ref = self.xref(m.group(1), m.group(2)) if self.xref else None
            if not ref:
                return m.group(0)
            return keep(f'<a href="{_attr(ref[0])}">{_esc(ref[1])}</a>')

        text = _IMAGE.sub(image, text)
        text = _LINK.sub(link, text)
        if self.xref:
            text = _DOC_MENTION.sub(mention, text)
        text = self._emphasis(text)
        while "\x00" in text:
            text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
        return text

    def _autolink(self, keep: Callable[[str], str]) -> Callable[[re.Match], str]:
        def sub(m: re.Match) -> str:
            href = self.link(m.group(1)) or m.group(1)
            return keep(f'<a href="{_attr(href)}">{_esc(m.group(1))}</a>')
        return sub

    @staticmethod
    def _emphasis(text: str) -> str:
        text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text, flags=re.DOTALL)
        text = re.sub(r"(?<![\w_])__(?=\S)(.+?)(?<=\S)__(?![\w_])", r"<strong>\1</strong>", text, flags=re.DOTALL)
        text = re.sub(r"(?<!\*)\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?!\*)", r"<em>\1</em>", text, flags=re.DOTALL)
        text = re.sub(r"(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])", r"<em>\1</em>", text, flags=re.DOTALL)
        return text


def markdown_to_html(text: str, link: Callable[[str], str | None] | None = None, **kw) -> Rendered:
    """Convenience wrapper: render ``text`` with a fresh :class:`MarkdownRenderer`."""
    return MarkdownRenderer(link, **kw).render(text)


# =============================================================================
# Repository facts
# =============================================================================

def read_version(repo: Path = REPO) -> str:
    """The package version, from src/aerolume/__init__.py (no import, no dependencies)."""
    text = (repo / "src" / "aerolume" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return m.group(1) if m else "0.0.0"


def pyproject_version(repo: Path = REPO) -> str | None:
    """``[project] version`` from pyproject.toml, or None. Only used to warn when it and the
    package's ``__version__`` disagree: the installer and pip read one, the site the other."""
    path = repo / "pyproject.toml"
    if not path.exists():
        return None
    return tomllib.loads(path.read_text(encoding="utf-8")).get("project", {}).get("version")


def released_versions(repo: Path = REPO) -> dict[str, str]:
    """Released versions in CHANGELOG.md with their dates, in the file's order (newest first):
    ``{"0.1.0": "2026-10-31"}``. A version counts as released when the changelog has a
    ``## [x.y.z] - YYYY-MM-DD`` heading for it; ``[Unreleased]`` does not.

    The home page uses this to say which release it describes ("the first release") and to tell
    a validation run of a released version from one of a development build. That matters
    because the tree carried an internal label (0.3.0) before the first public release was
    numbered 0.1.0, and a run recorded under that label must not read as a release that never
    happened."""
    path = repo / "CHANGELOG.md"
    if not path.exists():
        return {}
    heads = re.findall(r"^##\s*\[(\d+\.\d+\.\d+[^\]]*)\]\s*[-–]\s*(\d{4}-\d{2}-\d{2})",
                       path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return dict(heads)


def release_note(version: str, releases: dict[str, str]) -> str:
    """", the first release" when ``version`` is the changelog's only (oldest) release, else ""."""
    return ", the first release" if releases and list(releases)[-1] == version else ""


def run_facts(record: ValidationRecord, version: str, releases: dict[str, str]) -> str:
    """The ledger caption's run sentence: date, program and GPU of the validation run.

    The record names the version string the run was made with. It is quoted when it is the
    version this site describes or another released one; a string that never was a release (a
    development label) is described as what it is, a development build, with the release it
    preceded when the dates say so, so the page never names a version that was not published."""
    if not record.ok:
        return ""
    gpu = _esc(str(record.meta.get("gpu", "")).replace("NVIDIA GeForce ", ""))
    on = f" on an {gpu}" if gpu else ""
    ran = str(record.meta.get("aerolume", "") or "")
    when = record.run_date()
    if ran and (ran == version or ran in releases):
        return f"Run of {when}, Aerolume {_esc(ran)}{on}."
    released = releases.get(version, "")
    if released and str(record.meta.get("date", ""))[:10] <= released:
        return f"Run of {when}{on}, with a development build made before the {_esc(version)} release."
    return f"Run of {when}{on}, with a development build of Aerolume."


def figure_drawn(fig: dict, version: str, today: _dt.date) -> str:
    """The title block's "Drawn" cell: the date the opening figure was computed, plus the release that
    computed it when that is not the one the site describes.

    The cell used to read "Version <site version> · <figure date>", which paired the current version
    with the date of a figure drawn by an older release and read as that version's release date.
    Figures rendered before make_figures.py recorded the version carry only their date."""
    date = _esc(str(fig.get("date", today.isoformat())))
    drawn_by = str(fig.get("aerolume", "") or "")
    return f"{date}, with {_esc(drawn_by)}" if drawn_by and drawn_by != version else date


def count_templates(repo: Path = REPO) -> int:
    """Number of geometry templates registered in aerolume.geometry.templates."""
    path = repo / "src" / "aerolume" / "geometry" / "templates.py"
    if not path.exists():
        return 0
    return len(re.findall(r"^_register\(TemplateInfo\(", path.read_text(encoding="utf-8"), flags=re.MULTILINE))


def citation_field(repo: Path, key: str) -> str | None:
    """A top-level ``key: value`` of the repository's CITATION.cff (``repository-code``, ``url``),
    if it is an https address. CITATION.cff is where a published repository states its addresses,
    and it is not a build input of the app, so recording them there leaves the installer current."""
    path = repo / "CITATION.cff"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf'^{re.escape(key)}:\s*"?(https://[^"\s]+)"?\s*$', line)
        if m:
            return m.group(1)
    return None


def find_repo_url(repo: Path = REPO, env: dict[str, str] | None = None) -> str | None:
    """The public repository address, or None when nothing states it.

    In order: ``[project.urls]`` in pyproject.toml (Repository, Source, Source Code,
    Homepage), the ``AEROLUME_REPO_URL`` variable, and GitHub Actions'
    ``GITHUB_SERVER_URL``/``GITHUB_REPOSITORY`` (always set in the Pages workflow, and
    by definition the repository being built), and last ``repository-code`` in CITATION.cff.
    """
    env = os.environ if env is None else env
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        urls = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get("urls", {})
        by_key = {k.lower(): v for k, v in urls.items()}
        for key in ("repository", "source", "source code", "homepage"):
            url = by_key.get(key, "")
            if re.match(r"^https://\S+$", url):
                return url.rstrip("/").removesuffix(".git")
    url = env.get("AEROLUME_REPO_URL", "").strip()
    if url:
        return url.rstrip("/").removesuffix(".git")
    if env.get("GITHUB_REPOSITORY"):
        return f"{env.get('GITHUB_SERVER_URL', 'https://github.com').rstrip('/')}/{env['GITHUB_REPOSITORY']}"
    url = citation_field(repo, "repository-code")
    return url.rstrip("/").removesuffix(".git") if url else None


def build_date() -> _dt.date:
    """Today, or SOURCE_DATE_EPOCH for reproducible builds."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.isdigit():
        return _dt.datetime.fromtimestamp(int(epoch), tz=_dt.UTC).date()
    return _dt.datetime.now().astimezone().date()


def naca4_outline(code: str = "2412", n: int = 40) -> list[tuple[float, float]]:
    """NACA 4-digit outline (closed trailing edge), Selig order, unit chord.

    Used for the site's mark and favicon, so the logo is a real section shape.
    """
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    xs = [0.5 * (1 - math.cos(math.pi * k / n)) for k in range(n + 1)]
    upper, lower = [], []
    for x in xs:
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if m and x < p:
            yc, dy = m / p ** 2 * (2 * p * x - x * x), 2 * m / p ** 2 * (p - x)
        elif m:
            yc, dy = m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x * x), 2 * m / (1 - p) ** 2 * (p - x)
        else:
            yc, dy = 0.0, 0.0
        th = math.atan(dy)
        upper.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th), yc - yt * math.cos(th)))
    return upper[::-1] + lower[1:-1]


def airfoil_path(alpha_deg: float = 8.0, width: float = 28.0, height: float = 10.0) -> tuple[str, str]:
    """SVG path data and viewBox of a NACA 2412 at incidence, fitted to ``width`` x ``height``."""
    a = math.radians(-alpha_deg)
    pts = [((x - 0.25) * math.cos(a) - y * math.sin(a), (x - 0.25) * math.sin(a) + y * math.cos(a))
           for x, y in naca4_outline()]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    s = min((width - 2) / (max(xs) - min(xs)), (height - 2) / (max(ys) - min(ys)))
    ox = (width - s * (max(xs) - min(xs))) / 2 - s * min(xs)
    oy = (height - s * (max(ys) - min(ys))) / 2 + s * max(ys)
    d = "M" + " L".join(f"{ox + s * x:.2f} {oy - s * y:.2f}" for x, y in pts) + " Z"
    return d, f"0 0 {width:g} {height:g}"


def fmt_reynolds(re_: float) -> str:
    """5000 -> '5000'; 2e5 -> '2&times;10<sup>5</sup>'; 1e6 -> '10<sup>6</sup>'."""
    if re_ < 1e4:
        return f"{re_:.0f}"
    e = math.floor(math.log10(re_))
    mant = re_ / 10 ** e
    if abs(mant - 1) < 1e-9:
        return f"10<sup>{e}</sup>"
    return f"{mant:.3g}&times;10<sup>{e}</sup>"


def parse_manifest(text: str) -> list[dict[str, str]]:
    """Rows of the first pipe table in assets/screens/MANIFEST.md, keyed by header."""
    lines = text.splitlines()
    for k in range(len(lines) - 1):
        if lines[k].lstrip().startswith("|") and _TABLE_DELIM.match(lines[k + 1]):
            header = [h.strip().lower() for h in MarkdownRenderer.split_row(lines[k])]
            rows = []
            for ln in lines[k + 2:]:
                if not ln.lstrip().startswith("|"):
                    break
                cells = MarkdownRenderer.split_row(ln)
                rows.append(dict(zip(header, (cells + [""] * len(header))[:len(header)], strict=True)))
            return rows
    return []


@dataclass
class DocTable:
    """A pipe table of a Markdown document with the heading it sits under."""
    heading: str
    header: list[str]
    rows: list[list[str]]

    def col(self, name: str) -> int:
        """Index of the column headed ``name``, else the first whose header starts with it."""
        folded = [_fold(h) for h in self.header]
        if _fold(name) in folded:
            return folded.index(_fold(name))
        for k, h in enumerate(folded):
            if h.startswith(_fold(name)):
                return k
        raise KeyError(name)

    def row(self, first: str) -> list[str]:
        """The first row whose first cell equals ``first`` (whitespace and minus signs folded)."""
        key = _fold(first)
        for r in self.rows:
            if _fold(r[0]) == key:
                return r
        raise KeyError(first)


def _fold(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("−", "-").replace("*", "")).strip().lower()


def doc_tables(text: str) -> list[DocTable]:
    """Every pipe table in a Markdown text, outside code fences."""
    tables: list[DocTable] = []
    heading, fence = "", False
    lines = text.splitlines()
    k = 0
    while k < len(lines):
        ln = lines[k]
        if _FENCE.match(ln):
            fence = not fence
        elif not fence and (m := _HEADING.match(ln)):
            heading = (m.group(2) or "").strip()
        elif not fence and "|" in ln and k + 1 < len(lines) and _TABLE_DELIM.match(lines[k + 1]) \
                and "-" in lines[k + 1]:
            header = MarkdownRenderer.split_row(ln)
            rows = []
            k += 2
            while k < len(lines) and lines[k].strip() and "|" in lines[k]:
                rows.append(MarkdownRenderer.split_row(lines[k]))
                k += 1
            tables.append(DocTable(heading, header, rows))
            continue
        k += 1
    return tables


# =============================================================================
# Validation record (home page)
# =============================================================================

#: The JSON record that section 9 of docs/validation.md was written from
#: (``python validation/run_all.py --write-docs`` keeps it in the repository). The home page's
#: tally and checks are read from it, the same source as the report's tables, so rerunning
#: ``run_all.py --all`` reproduces what the page shows.
RECORD_PATH = Path("validation") / "records" / "validation-record.json"

#: Area keys of the record, in report order, with the headings of section 9's tables (the
#: home page links each area to its table by that heading).
AREA_TITLES: dict[str, str] = {
    "verification": "Verification (exact answers)",
    "flow2d": "2D flow",
    "flow3d": "3D flow",
    "thermal": "Heat transfer",
    "fans": "Fans and jets",
    "airfoil": "Airfoil polars",
    "wing": "Finite wings",
}

#: Checks whose tolerance band was revised after their result was known, with the report
#: section that says so (docs/validation.md section 3, "The square cylinder's band in section 9,
#: and how it changed"). The page marks them; tests/test_build.py checks the report still says it.
REVISED_BANDS: dict[tuple[str, str], str] = {
    ("square", "mean Cd"): "3.",
}

_GRID_SUFFIX = re.compile(r"\s*\((\d+)\s*cells/D[^)]*\)\s*$")


@dataclass(frozen=True)
class Check:
    """One check of the validation record: measured value, tolerance band and verdict."""
    area: str
    case_key: str
    case: str
    quantity: str
    value: float
    lo: float | None
    hi: float | None
    ref: float | None
    ref_text: str
    source: str
    passed: bool

    @property
    def band_use(self) -> float | None:
        """Where the value sits in its tolerance band: 0 at the centre, 1 on an edge, above 1
        outside it. A one-sided band (an exact zero with an upper limit) counts from the zero.
        None when the record gives no band."""
        lo, hi = self.lo, self.hi
        if lo is None or hi is None or hi <= lo:
            return None
        if self.ref is not None and self.ref == lo:
            return (self.value - lo) / (hi - lo)
        return abs(self.value - 0.5 * (lo + hi)) / (0.5 * (hi - lo))

    @property
    def one_sided(self) -> bool:
        return self.ref is not None and self.lo is not None and self.ref == self.lo == 0

    def published(self) -> tuple[float, float] | None:
        """The published value or range, read from the reference text without its remarks."""
        nums = re.findall(r"\d+(?:\.\d+)?", re.sub(r"\(.*?\)", "", self.ref_text))
        if not nums:
            return None
        return float(nums[0]), float(nums[-1])

    def published_text(self) -> str:
        """The reference as the report prints it, without the parenthesised remark."""
        return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", self.ref_text)).strip(" ,;")

    def difference(self) -> str:
        """Difference from the published value (or the nearest edge of the published range),
        in per cent; "in range" inside a range; an em dash against an exact zero, where a
        relative difference has no meaning (the band column carries the judgement)."""
        pub = self.published()
        if pub is None or pub[1] == 0:
            return "&mdash;"
        lo, hi = pub
        if lo < hi and lo <= self.value <= hi:
            return "in range"
        edge = hi if self.value > hi else lo
        return _pct(100 * (self.value / edge - 1))

    def band_text(self) -> str:
        if self.lo is None or self.hi is None:
            return "&mdash;"
        if self.one_sided:
            return f"&lt;&nbsp;{fmt_value(self.hi)}"
        return f"{fmt_value(self.lo)}&ndash;{fmt_value(self.hi)}"

    def revised_section(self) -> str | None:
        for (key, part), section in REVISED_BANDS.items():
            if self.case_key == key and part in self.quantity:
                return section
        return None

    def grid_base(self) -> str:
        """The quantity without its grid suffix, to pair one check measured on two grids."""
        return _GRID_SUFFIX.sub("", self.quantity)


def _minus(s: str) -> str:
    return s.replace("-", "&minus;").replace("−", "&minus;")


def _signed(x: float, nd: int) -> str:
    """'+0.027', '&minus;0.24'; a value that rounds to zero is shown unsigned ('0.000')."""
    if round(x, nd) == 0:
        return f"{0:.{nd}f}"
    return _minus(f"{x:+.{nd}f}")


def _pct(x: float) -> str:
    return _signed(x, 1) + "&nbsp;%"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt_value(x: float) -> str:
    """Four significant figures, as the report's tables print them; round-off-sized values in
    scientific notation (1.79 &times; 10⁻⁷)."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant}&nbsp;&times;&nbsp;10{str(int(exp)).translate(_SUPERSCRIPT)}"
    return _minus(f"{x:.4g}")


class ValidationRecord:
    """The last full validation run (validation/records/validation-record.json).

    ``ok`` is False when the record is missing or unreadable; the page then leaves the ledger
    out rather than showing numbers from anywhere else.
    """

    def __init__(self, repo: Path = REPO) -> None:
        self.repo = repo
        self.problems: list[str] = []
        self.meta: dict = {}
        self.cases: list[dict] = []
        self.checks: list[Check] = []
        path = repo / RECORD_PATH
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.meta, self.cases = data["meta"], data["results"]
            for c in self.cases:
                for k in c.get("checks", []):
                    self.checks.append(Check(
                        c.get("area", ""), c.get("key", ""), c.get("title", ""), str(k["quantity"]),
                        float(k["value"]), _opt(k.get("lo")), _opt(k.get("hi")), _opt(k.get("ref")),
                        str(k.get("ref_text", "")), str(k.get("source", "")), bool(k.get("passed"))))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.problems.append(f"validation record {RECORD_PATH.as_posix()} not readable ({exc!r}); "
                                 "the home page leaves the checks out")

    @property
    def ok(self) -> bool:
        return bool(self.checks)

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)

    def failing_cases(self) -> list[str]:
        """Titles of the cases with at least one failing check, in report order."""
        return [c.get("title", "") for c in self.cases if any(not k.get("passed") for k in c.get("checks", []))]

    def run_date(self) -> str:
        """'30 October 2026' from the record's '2026-10-30 23:05'."""
        try:
            d = _dt.date.fromisoformat(str(self.meta.get("date", ""))[:10])
        except ValueError:
            return ""
        return f"{d.day} {d.strftime('%B')} {d.year}"

    def area_checks(self, area: str) -> list[Check]:
        return [c for c in self.checks if c.area == area]

    def ledger(self) -> list[tuple[str, list[Check]]]:
        """For each solver area: the passing check that came closest to its band edge, then every
        failing check, each followed by the same quantity on the run's other grids.

        The rule is mechanical, so the selection cannot flatter: it shows the weakest pass of
        every area and every failure, whatever the numbers are."""
        out: list[tuple[str, list[Check]]] = []
        areas = list(AREA_TITLES) + sorted({c.area for c in self.checks} - set(AREA_TITLES))
        for area in areas:
            cs = self.area_checks(area)
            if not cs:
                continue
            picks: list[Check] = []
            banded = [c for c in cs if c.passed and c.band_use is not None]
            if banded:
                picks.append(max(banded, key=lambda c: c.band_use or 0.0))
            for f in (c for c in cs if not c.passed):
                picks.append(f)
                picks.extend(c for c in cs if c is not f and c.case_key == f.case_key
                             and c.grid_base() == f.grid_base())
            uniq: list[Check] = []
            for c in picks:
                if all(c is not u for u in uniq):
                    uniq.append(c)
            out.append((area, uniq))
        return out


def _opt(x: object) -> float | None:
    return None if x is None else float(x)  # type: ignore[arg-type]


def reference_labels(repo: Path = REPO) -> dict[str, str]:
    """Short citations ("Schäfer & Turek (1996)") of the record's source keys, read from the
    "References of the automated run" list in docs/validation.md, so the page and the report
    name each source the same way."""
    path = repo / "docs" / "validation.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    out: dict[str, str] = {}
    for key, body in re.findall(r'<a id="ref-([\w-]+)"></a>\*\*[\w-]+\*\*:\s*(.+)', text):
        # A surname may open with a lower-case particle ("de Vahl Davis", "van Dyke").
        m = re.search(r"((?:(?:de|van|von|der|den|di|du|la|le)\s)*[A-Z][^:;()]*?)\s\((\d{4}(?:/\d{4})?)\)", body)
        if m:
            names = re.split(r",\s*|\s+&\s+|\s+and\s+", m.group(1).strip())
            who = m.group(1).strip() if len(names) <= 2 else f"{names[0]} et al."
            out[key] = f"{who} ({m.group(2)})"
        else:
            out[key] = re.split(r"[,:(]", body, maxsplit=1)[0].strip()
    out.setdefault("exact", "Exact solution")
    if out.get("exact", "").lower().startswith("exact solution or"):
        out["exact"] = "Exact solution"
    return out


def doc_section(text: str, title: str) -> str:
    """The Markdown under the heading ``title`` (any level), up to the next heading of the same
    or a higher level; "" when the heading is missing."""
    lines = text.splitlines()
    out: list[str] = []
    level = 0
    fence = False
    for ln in lines:
        if _FENCE.match(ln):
            fence = not fence
        m = None if fence else _HEADING.match(ln)
        if m and level and len(m.group(1)) <= level:
            break
        if level:
            out.append(ln)
        elif m and _fold(m.group(2) or "") == _fold(title):
            level = len(m.group(1))
    return "\n".join(out).strip()


# =============================================================================
# Page parts
# =============================================================================

THEME_BOOT = ('<script>try{var t=localStorage.getItem("aerolume-theme");'
              'if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t)}catch(e){}</script>')

MATHJAX_BOOT = ('<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]},'
                'svg:{fontCache:"global"},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};'
                f'</script>\n<script defer src="{MATHJAX_URL}"></script>')


@dataclass(frozen=True)
class SiteInfo:
    """Facts every page needs: version, build date and the two optional addresses."""
    version: str
    date: _dt.date
    repo_url: str | None = None
    site_url: str | None = None


def head(title: str, root: str, site: SiteInfo, *, path: str = "", description: str = SITE_DESCRIPTION,
         math_: bool = False) -> str:
    """The shared <head> content (without the <head> tags).

    Open Graph needs absolute URLs (crawlers ignore relative ones), so og:url and
    og:image are only emitted when the site's own address is known.
    """
    parts = [
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        f"<title>{_esc(title)}</title>",
        f'<meta name="description" content="{_attr(description)}">',
        '<meta name="color-scheme" content="light dark">',
        f'<meta property="og:title" content="{_attr(title)}">',
        f'<meta property="og:description" content="{_attr(description)}">',
        '<meta property="og:type" content="website">',
    ]
    if site.site_url:
        base = site.site_url.rstrip("/") + "/"
        parts += [f'<meta property="og:url" content="{_attr(base + path)}">',
                  f'<meta property="og:image" content="{_attr(base)}assets/social-card.jpg">',
                  '<meta name="twitter:card" content="summary_large_image">']
    parts += [
        f'<link rel="icon" href="{root}assets/favicon.svg" type="image/svg+xml">',
        f'<link rel="preload" href="{root}{FONT_PRELOAD}" as="font" type="font/woff2" crossorigin>',
        f'<link rel="stylesheet" href="{root}assets/site.css">',
        THEME_BOOT,
    ]
    if math_:
        parts.append(MATHJAX_BOOT)
    return "\n".join(parts)


#: Icons of the header's round buttons (16 px grid, drawn in currentColor so they follow the theme).
ICON_THEME = ('<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.25" fill="none" '
              'stroke="currentColor" stroke-width="1.5"/><path d="M8 1.75a6.25 6.25 0 0 1 0 12.5z" '
              'fill="currentColor"/></svg>')
ICON_PAUSE = ('<svg class="i-pause" viewBox="0 0 16 16" aria-hidden="true"><path d="M5.5 3.5v9M10.5 3.5v9" '
              'stroke="currentColor" stroke-width="1.75" stroke-linecap="round"/></svg>'
              '<svg class="i-play" viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.25v9.5L12.5 8z" '
              'fill="currentColor"/></svg>')


def topbar(root: str, site: SiteInfo, current: str = "", *, home: bool = False) -> str:
    """Site header: the main links and the theme switch.

    On the home page the bar sits over the hero's live flow: there the big "Aerolume" of the
    hero is the wordmark, the bar carries the pause button of the animation, and
    ``data-hero-quiet`` asks the flow to stay calm behind it. Other pages get the wordmark and
    version on the left. Both round buttons start hidden: they need a script to work, and the
    script shows them.
    """
    index = f"{root}index.html"
    here = "" if home else index           # same-page anchors on the home page
    docs_attr = ' aria-current="page"' if current == "docs" else ""
    source = f'\n      <a href="{_attr(site.repo_url)}">Source</a>' if site.repo_url else ""
    mark = ("" if home else
            f'\n    <a class="wordmark" href="{index}">Aerolume<span class="ver">{_esc(site.version)}</span></a>')
    pause = ('\n      <button class="icon-btn hero-pause" type="button" data-hero-pause hidden '
             f'aria-label="Pause the background animation">{ICON_PAUSE}</button>') if home else ""
    quiet = " data-hero-quiet" if home else ""
    return f"""<header class="top">
  <div class="top-inner wrap"{quiet}>{mark}
    <nav aria-label="Site">
      <a href="{here}#validation">Validation</a>
      <a href="{here}#download">Get Aerolume</a>
      <a href="{root}docs/index.html"{docs_attr}>Docs</a>{source}
    </nav>
    <div class="top-tools">{pause}
      <button class="icon-btn theme-toggle" type="button" data-theme-toggle hidden aria-label="Switch theme">{ICON_THEME}</button>
    </div>
  </div>
</header>"""


def footer(root: str, site: SiteInfo) -> str:
    """Site footer with the build facts."""
    links = [f'<a href="{root}docs/index.html">Documentation</a>',
             f'<a href="{root}index.html#validation">Validation</a>',
             f'<a href="{root}index.html#download">Get Aerolume</a>']
    if site.repo_url:
        links += [f'<a href="{_attr(site.repo_url)}">Source</a>', f'<a href="{_attr(site.repo_url)}/issues">Issues</a>']
    links.append(f'<a href="{root}LICENSE.txt">Licence</a>')
    nav = "\n      ".join(links)
    return f"""<footer class="foot">
  <div class="wrap">
    <p>Aerolume {_esc(site.version)}, Apache License 2.0. Built from the repository by <code>website/build.py</code> on {site.date.isoformat()}. Every number on this site comes from the repository&rsquo;s validation and benchmark scripts, and the documentation states the conditions of each.</p>
    <nav aria-label="Footer">
      {nav}
    </nav>
  </div>
</footer>"""


def parse_perf_table(text: str) -> list[tuple[str, str, int, int, str, float]]:
    """Rows of the ``| device | grid | options | steps/s | ...`` table in a Markdown text.

    A device cell such as ``CUDA, RTX 4060 Ti`` names the hardware; a bare ``CUDA`` or
    ``CPU`` repeats the hardware of the row above. Returns [] if no such table exists.
    """
    lines = text.splitlines()
    for k in range(len(lines) - 1):
        hdr = [h.strip().lower() for h in MarkdownRenderer.split_row(lines[k])] if "|" in lines[k] else []
        if hdr[:4] != ["device", "grid", "options", "steps/s"] or not _TABLE_DELIM.match(lines[k + 1]):
            continue
        rows: list[tuple[str, str, int, int, str, float]] = []
        names = {"GPU": "GPU", "CPU": "CPU"}
        for ln in lines[k + 2:]:
            if not ln.lstrip().startswith("|"):
                break
            cells = [c.strip().strip("*").strip() for c in MarkdownRenderer.split_row(ln)]
            if len(cells) < 4:
                continue
            dev, grid, opts, sps = cells[:4]
            kind = "GPU" if dev.upper().startswith(("CUDA", "GPU", "MPS")) else "CPU"
            if "," in dev:
                names[kind] = dev.split(",", 1)[1].strip()
            m = re.match(r"(\d+)\s*[x×]\s*(\d+)", grid)
            try:
                value = float(sps.replace(",", ""))
            except ValueError:
                continue
            if m:
                rows.append((names[kind], kind, int(m.group(1)), int(m.group(2)), opts, value))
        return rows
    return []


def perf_rows(repo: Path = REPO) -> list[tuple[str, str, int, int, str, float]]:
    """Benchmark rows from docs/theory_ns2d.md, or the built-in copy."""
    doc = repo / "docs" / "theory_ns2d.md"
    rows = parse_perf_table(doc.read_text(encoding="utf-8")) if doc.exists() else []
    return rows or list(PERF_FALLBACK)


def perf_facts(rows: list[tuple[str, str, int, int, str, float]]) -> dict[str, str]:
    """Throughput quoted on the home page: the default GPU configuration and the fastest CPU
    configuration that was measured (no other CPU is claimed)."""
    gpu = [r for r in rows if r[1] == "GPU"]
    cpu = max((r for r in rows if r[1] == "CPU"), key=lambda r: r[5], default=None)
    facts = {"PERF_GPU_DEFAULT": f"{gpu[0][5]:.0f}" if gpu else "?",
             "PERF_GPU_NAME": _esc(gpu[0][0]) if gpu else "GPU"}
    if cpu:
        threads = re.match(r"(\d+)\s*threads?", cpu[4])
        facts["PERF_CPU"] = (f"<b>{cpu[5]:.0f}</b> steps/s on {cpu[2]}&nbsp;&times;&nbsp;{cpu[3]} cells in 2D on a "
                             f"desktop {_esc(cpu[0])}" + (f", {threads.group(1)} threads" if threads else ""))
    else:
        facts["PERF_CPU"] = "every analysis also runs on the CPU"
    return facts


#: Screens shown inside the feature sections of the home page, each cropped to the panel the
#: text talks about, as (x, y, width, height) in the 1600 x 1000 capture. A whole window shown
#: at 400-750 CSS px turns its tables into texture; the crop is drawn near 1:1 and links to the
#: full capture. Captions come from assets/screens/MANIFEST.md unless the crop leaves out part of
#: what the manifest caption describes; then CROP_CAPTIONS says what the crop shows.
FEATURE_CROPS: dict[str, tuple[int, int, int, int]] = {
    "results3d.png": (234, 96, 1076, 720),     # status line, warning and the 3D view
    "heat.png": (236, 150, 580, 446),          # Nusselt table, per-body table and heat balance
    "physics.png": (515, 150, 460, 354),       # the closed room: supply, vent, rake, fan, heater
    "explore.png": (222, 90, 1036, 630),       # drawing tools and the running flow with an obstacle
    "script.png": (0, 548, 960, 300),          # the script console
}
FEATURE_SHOTS = tuple(FEATURE_CROPS)
CROP_CAPTIONS: dict[str, str] = {
    "heat.png": "Results: the heated cylinder's Nusselt number and heat rate with their standard errors, "
                "per body, and the heat balance of the domain.",
    "explore.png": "Simulate: in explore mode, a wall drawn into the running flow of the room, with the "
                   "run's monitors.",
    "script.png": "Script: the scripting API reading the open project's results in the script console.",
}
SHOT_W, SHOT_H = 1600, 1000


def feature_shot(assets: Path, name: str, *, display_px: int = 760) -> str:
    """One manifest screenshot as a cropped, linked figure, or "" while it is missing (the page
    never shows a placeholder frame).

    The crop is CSS only (the image is scaled and offset inside a frame of the crop's shape), so
    the repository keeps one capture per page. When ``<name>-800.webp`` sits next to the PNG
    (website/tools/make_screens.py can write it), the image gets a srcset so phones do not
    download the 1600 px file; ``sizes`` accounts for the crop's magnification."""
    manifest = assets / "screens" / "MANIFEST.md"
    rows = parse_manifest(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    row = next((r for r in rows if r.get("file") == name), None)
    if row is None or not (assets / "screens" / name).exists():
        return ""
    page, caption = row.get("page", ""), CROP_CAPTIONS.get(name) or row.get("caption", "")
    text = caption
    if page and text.lower().startswith(page.lower() + ":"):
        text = text[len(page) + 1:].strip()
        text = text[:1].upper() + text[1:]
    x, y, w, h = FEATURE_CROPS.get(name, (0, 0, SHOT_W, SHOT_H))
    k = SHOT_W / w
    style = f"width:{100 * k:.3f}%;left:{-100 * x / w:.3f}%;top:{-100 * y / h:.3f}%"
    src = f"assets/screens/{_attr(name)}"
    webp = Path(name).stem + "-800.webp"
    srcset = ""
    if (assets / "screens" / webp).exists():
        srcset = (f' srcset="assets/screens/{_attr(webp)} 800w, {src} {SHOT_W}w"'
                  f' sizes="(max-width: 980px) {100 * k:.0f}vw, {display_px * k:.0f}px"')
    return (f'<figure class="shot shot-feature"><a class="frame crop" href="{src}" '
            f'style="--crop-ar: {w} / {h}" title="Open the full screenshot ({SHOT_W} &times; {SHOT_H})">'
            f'<img src="{src}"{srcset} alt="{_attr(caption)}" loading="lazy" decoding="async" '
            f'width="{SHOT_W}" height="{SHOT_H}" style="{style}"></a>'
            f"<figcaption><b>{_esc(page)}.</b> {_esc(text)}</figcaption></figure>")


def _count_word(n: int) -> str:
    return ("No", "One", "Two", "Three", "Four", "Five", "Six")[n] if n < 7 else str(n)


def validation_tally(record: ValidationRecord) -> str:
    """The trust section's sentence about the last full run, failing cases named.

    From the same record as the ledger and the report's section 9, so the claim next to
    "no black-box results" can never be rosier than the report. Empty without a record."""
    if not record.ok:
        return ""
    when = f" ({record.run_date()})" if record.run_date() else ""
    # The pass count is never animated (a count-in showed "0 of 125" and "89 of 125" on the way,
    # wrong numbers on a page whose point is that every number is measured). What moves is a thin
    # bar of the passing share, drawn once; the gap at its end is the failing checks.
    frac = record.passed / record.total
    bar = (f'<span class="tally-bar" aria-hidden="true" style="--frac:{frac:.4f}">'
           '<span class="tally-fill"></span></span>')
    head = (f"{bar}In the last full run{when}, <strong>{record.passed} of "
            f"{record.total} checks passed</strong>.")
    fails = [_esc(c) for c in record.failing_cases()]
    if not fails:
        return f'<p class="tally">{head}</p>'
    if len(fails) == 1:
        tail = f"One case has a failing check: {fails[0]}. It stays in the report as failing"
    else:
        tail = (f"{_count_word(len(fails))} cases have a failing check: {'; '.join(fails)}. "
                f"{'Both' if len(fails) == 2 else 'All'} stay in the report as failing")
    return f'<p class="tally">{head} {tail}, with the numbers and the reason.</p>'


def revised_note(record: ValidationRecord) -> str:
    """The trust lede's clause about bands revised after their result was known, from the record
    and REVISED_BANDS. One revised band checked at several grid levels shows on several ledger
    rows; saying "the one band ... is marked" over two marked rows read as a contradiction, so
    the clause names the case and the grid levels."""
    revised = [c for c in record.checks if c.revised_section()] if record.ok else []
    if not revised:
        return "No band was revised after its result was known."
    bands = {(c.case_key, c.grid_base()) for c in revised}
    if len(bands) == 1:
        c = revised[0]
        what = re.sub(r",?\s*Re\s*[\d.]+$", "", c.case).strip().lower()
        qty = "drag" if "Cd" in c.quantity else "result"
        grids = sorted({m.group(1) for m in (_GRID_SUFFIX.search(x.quantity) for x in revised) if m}, key=int)
        at = (f", checked at {_count_word(len(grids)).lower()} grid levels" if len(grids) > 1 else "")
        return (f"The one band revised after its result was known (the {_esc(what)}&rsquo;s {qty}{at}) "
                "is marked below.")
    return f"The {_count_word(len(bands)).lower()} bands revised after their results were known are marked below."


def band_gauge(c: Check) -> str:
    """A small drawing of where the value sits in its tolerance band: the band as a bracketed
    line, the value as a mark, outside the brackets when it fails (clamped a little beyond them,
    so a far miss still fits the cell). Decorative (aria-hidden): the band and the verdict are
    in the cells as text. "" when the record gives no band."""
    lo, hi = c.lo, c.hi
    if lo is None or hi is None or hi <= lo:
        return ""
    p = min(1.25, max(-0.25, (c.value - lo) / (hi - lo)))
    return f'<span class="gauge" style="--p:{p:.3f}" aria-hidden="true"><span class="band"></span><span class="mark"></span></span>'


_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def cavity_points(record: ValidationRecord) -> list[tuple[int, str, Check]]:
    """(Ra exponent, grid, check) of the square cavity's hot-wall mean Nu checks, by Ra."""
    out = []
    for c in record.checks:
        if c.case_key != "cavity2d" or not c.quantity.endswith("mean Nu, hot wall"):
            continue
        if c.lo is None or c.hi is None or c.ref is None or c.ref <= 0 or c.value <= 0:
            continue
        m = re.search(r"Ra 10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", c.quantity)
        g = re.search(r"\((\d+)²\)", c.quantity)
        if m:
            out.append((int(m.group(1).translate(_SUPERSCRIPT)), g.group(1) if g else "", c))
    return sorted(out, key=lambda t: t[0])


def cavity_plot(record: ValidationRecord, refs: dict[str, str], anchor: Callable[[str], str]) -> str:
    """One validation case drawn across its whole range: the differentially heated square cavity,
    mean Nusselt number of the hot wall against the Rayleigh number, Aerolume's values (dots and
    a line) over the published ones (rings), with a lower panel of the difference against each
    check's tolerance band. The ledger above shows only each area's closest call; this shows what
    a passing case looks like over four decades.

    Built from the validation record, like the ledger, so it cannot drift from the report; the
    numbers are also in the caption as text (the drawing is an image for assistive technology,
    with its own title and description). The line has pathLength 1 so site.css can draw it on
    scroll with stroke-dashoffset. "" without at least two points."""
    pts = cavity_points(record)
    if len(pts) < 2:
        return ""
    W, ml, mr = 440, 68, 12
    t0, t1 = 12, 188            # top panel: log Nu
    b0, b1 = 216, 296           # lower panel: difference in %
    H = 336
    e0, e1 = pts[0][0] - 0.4, pts[-1][0] + 0.4
    xs = {e: ml + (e - e0) / (e1 - e0) * (W - ml - mr) for e, _, _ in pts}
    nus = [c.value for _, _, c in pts] + [c.ref for _, _, c in pts]
    lo_nu = min(1.0, min(nus) / 1.15)
    hi_nu = max(nus) * 1.25
    ylo, yhi = math.log10(lo_nu), math.log10(hi_nu)

    def y_top(nu: float) -> float:
        return t0 + (yhi - math.log10(nu)) / (yhi - ylo) * (t1 - t0)

    band_pct = max(max(abs(c.lo / c.ref - 1), abs(c.hi / c.ref - 1)) for _, _, c in pts) * 100
    lim = math.ceil(band_pct + 0.5)

    def y_bot(pct: float) -> float:
        return b0 + (lim - pct) / (2 * lim) * (b1 - b0)

    f = lambda v: f"{v:.1f}"
    parts = []
    # gridlines and tick labels
    for nu in (1, 2, 5, 10, 20, 50):
        if lo_nu <= nu <= hi_nu:
            y = y_top(nu)
            parts.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{f(y)}" y2="{f(y)}"/>'
                         f'<text class="tick" x="{ml - 8}" y="{f(y + 4)}" text-anchor="end">{nu}</text>')
    for pct in (-lim, 0, lim):
        y = y_bot(pct)
        parts.append(f'<line class="{"zero" if pct == 0 else "grid"}" x1="{ml}" x2="{W - mr}" y1="{f(y)}" y2="{f(y)}"/>'
                     f'<text class="tick" x="{ml - 8}" y="{f(y + 4)}" text-anchor="end">'
                     f'{"0" if pct == 0 else _minus(f"{pct:+d}")}&nbsp;%</text>')
    for e, _, _ in pts:
        parts.append(f'<text class="tick" x="{f(xs[e])}" y="{b1 + 20}" text-anchor="middle">10<tspan dy="-5" '
                     f'font-size="9">{e}</tspan></text>')
    parts.append(f'<text class="axis" x="{(ml + W - mr) / 2:.0f}" y="{H - 2}" text-anchor="middle">Rayleigh number</text>'
                 f'<text class="axis" transform="translate(14 {(t0 + t1) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 'Mean Nu, hot wall</text>'
                 f'<text class="axis" transform="translate(14 {(b0 + b1) / 2:.0f}) rotate(-90)" text-anchor="middle">'
                 'Difference</text>')
    # published values (rings) and each check's band (a bracket in the lower panel)
    for e, _, c in pts:
        x = xs[e]
        ya, yb = y_bot((c.hi / c.ref - 1) * 100), y_bot((c.lo / c.ref - 1) * 100)
        parts.append(f'<path class="bandbar" d="M{f(x - 5)} {f(ya)}H{f(x + 5)}M{f(x)} {f(ya)}V{f(yb)}'
                     f'M{f(x - 5)} {f(yb)}H{f(x + 5)}"/>'
                     f'<circle class="ref" cx="{f(x)}" cy="{f(y_top(c.ref))}" r="5"/>')
    # Aerolume: the line (drawn on scroll), then the dots
    top = " ".join(f"{f(xs[e])},{f(y_top(c.value))}" for e, _, c in pts)
    bot = " ".join(f"{f(xs[e])},{f(y_bot((c.value / c.ref - 1) * 100))}" for e, _, c in pts)
    parts.append(f'<polyline class="curve" pathLength="1" points="{top}"/>'
                 f'<polyline class="curve" pathLength="1" points="{bot}"/>')
    for i, (e, _, c) in enumerate(pts):
        parts.append(f'<circle class="val" style="--i:{i}" cx="{f(xs[e])}" cy="{f(y_top(c.value))}" r="2.6"/>'
                     f'<circle class="val" style="--i:{i}" cx="{f(xs[e])}" cy="{f(y_bot((c.value / c.ref - 1) * 100))}" r="2.6"/>')

    src = refs.get(pts[0][2].source, pts[0][2].source)
    ras = ", ".join(f"10<sup>{e}</sup>" for e, _, _ in pts)
    grids = sorted({g for _, g, _ in pts if g}, key=int)
    grid_txt = (f" on {grids[0]}&sup2;&ndash;{grids[-1]}&sup2; grids" if len(grids) > 1
                else (f" on a {grids[0]}&sup2; grid" if grids else ""))
    ours = ", ".join(fmt_value(c.value) for _, _, c in pts)
    theirs = ", ".join(_esc(c.published_text()) for _, _, c in pts)
    diffs = [(c.value / c.ref - 1) * 100 for _, _, c in pts]
    span = f"{_signed(min(diffs), 1)} to {_signed(max(diffs), 1)}&nbsp;%"
    n_ok = sum(c.passed for _, _, c in pts)
    desc = (f"Aerolume {ours} against {html.unescape(theirs)} published, at Ra {', '.join(f'{10 ** e:,}' for e, _, _ in pts)}; "
            f"differences {html.unescape(span)}, inside every band.") if n_ok == len(pts) else (
            f"Aerolume {ours} against {html.unescape(theirs)} published.")
    src_html = (f'<a href="{_attr(anchor("ref-" + pts[0][2].source))}">{_esc(src)}</a>' if pts[0][2].source else "the benchmark")
    verdict = (f"All {_count_word(len(pts)).lower()} pass." if n_ok == len(pts)
               else f"{n_ok} of {len(pts)} pass.")
    svg = (f'<svg class="vplot-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="vplot-t vplot-d">'
           '<title id="vplot-t">Square cavity: mean Nusselt number against Rayleigh number</title>'
           f'<desc id="vplot-d">{_esc(desc)}</desc>{"".join(parts)}</svg>')
    return (f'<figure class="vplot" data-reveal="plot">{svg}'
            f'<figcaption><b>One case across four decades.</b> Natural convection in a differentially heated '
            f'square cavity, Ra = {ras}{grid_txt}. Mean Nusselt number of the hot wall: Aerolume {ours} '
            f'(dots) against {src_html}, {theirs} (rings), a difference of {span}. The lower panel shows each '
            f"check&rsquo;s tolerance band. {verdict}</figcaption></figure>")


def _rdp(pts: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker: drop the points of a polyline that lie within ``tol`` of the chord
    between their neighbours, so a streamline's long straight far field costs two points, not two
    hundred, while its bend round the body keeps every point it needs."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (ax, ay), (bx, by) = pts[a], pts[b]
        dx, dy = bx - ax, by - ay
        norm = math.hypot(dx, dy) or 1.0
        far, at = -1.0, -1
        for k in range(a + 1, b):
            d = abs(dy * (pts[k][0] - ax) - dx * (pts[k][1] - ay)) / norm
            if d > far:
                far, at = d, k
        if far > tol:
            keep[at] = True
            stack += [(a, at), (at, b)]
    return [p for p, k in zip(pts, keep, strict=True) if k]


def streakline_svg(width: int = 1200, height: int = 104) -> str:
    """The divider above the workflow section: streamlines of inviscid flow past a cambered
    Joukowski airfoil at 4 degrees with the Kutta condition, computed here rather than drawn by
    hand, so the one decorative drawing below the hero is also a correct flow (the gentle rise of
    the lines towards the airfoil is its upwash).

    The flow is the textbook one: in the circle plane, a uniform stream at angle alpha past a
    circle through zeta = 1 (centre mu, radius R) with circulation Gamma = 4 pi U R sin(alpha +
    beta), which puts the rear stagnation point on the cusp; z = zeta + 1/zeta maps the circle to
    the airfoil. Each streamline is traced in the circle plane (RK2 along the velocity direction,
    steps sized to about ``h`` in the airfoil plane) and mapped, then the picture is turned by
    -alpha so the stream runs level and the airfoil is nose-up, and centred vertically. Decorative
    (aria-hidden): it carries no number. site.js's reveal draws the lines left to right
    (pathLength 1, so a stroke-dashoffset of 1 to 0 is the whole line whatever its length)."""
    alpha = math.radians(4.0)
    mu = complex(-0.08, 0.05)
    radius = abs(1 - mu)
    beta = math.asin(mu.imag / radius)
    gamma = 4 * math.pi * radius * math.sin(alpha + beta)
    ea = complex(math.cos(alpha), -math.sin(alpha))
    rot = ea                                                    # turn the picture by -alpha

    def vel(zeta: complex) -> complex:
        # conj(dW/dzeta) is the velocity in the circle plane
        s = zeta - mu
        return (ea - radius * radius / (ea * s * s) + 1j * gamma / (2 * math.pi * s)).conjugate()

    def jac(zeta: complex) -> float:
        return abs(1 - 1 / (zeta * zeta)) or 1e-9

    # chord ~ 4 units = 120 px at 82 % of the width: a phone (the svg is cropped from the left,
    # xMaxYMid slice) still shows the whole airfoil down to a 288 px wide column
    scale, cx = 30.0, 0.82 * width
    x0, x1 = -cx / scale - 0.3, (width - cx) / scale + 0.3
    traced: list[tuple[int, float, list[complex]]] = []
    # Seeds at the left edge, unevenly spaced: a real rake is never a ruler.
    for k, y0 in enumerate((-1.3, -0.97, -0.69, -0.45, -0.19, 0.05, 0.3, 0.56, 0.86, 1.22)):
        zeta = complex(x0, y0) / rot                            # z ~ zeta this far out
        pts = [zeta + 1 / zeta]
        for _ in range(8000):
            z = pts[-1]
            if (z * rot).real > x1:
                break
            h = min(0.5, 0.025 + 0.05 * abs(z))
            v1 = vel(zeta)
            if abs(v1) < 1e-6:
                break
            mid = zeta + 0.5 * h * v1 / abs(v1) / jac(zeta)
            v2 = vel(mid)
            zeta = zeta + h * v2 / abs(v2) / jac(mid)
            if abs(zeta - mu) < radius:                         # never inside the body
                break
            pts.append(zeta + 1 / zeta)
        if (pts[-1] * rot).real > x1:                           # a line that stopped at the body is dropped
            traced.append((k, y0, [p * rot for p in pts]))
    ys = [p.imag for _, _, pts in traced for p in pts]
    mid_y = 0.5 * (min(ys) + max(ys))

    def to_svg(w: complex) -> tuple[float, float]:
        return cx + scale * w.real, 0.5 * height - scale * (w.imag - mid_y)

    lines = []
    # drawn from the body outwards: --k is a line's rank by its distance from the airfoil
    rank = {k: r for r, (k, _y) in enumerate(sorted(((k, y0) for k, y0, _ in traced), key=lambda t: abs(t[1] + 0.1)))}
    for k, y0, pts in traced:
        d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in _rdp([to_svg(p) for p in pts], 0.12))
        # brighter near the body, fainter towards the edges (the eye goes to the bend)
        op = 0.85 - 0.45 * min(1.0, abs(y0 + 0.1) / 1.5)
        lines.append(f'<path d="{d}" pathLength="1" style="--k:{rank[k]}" opacity="{op:.2f}"/>')
    ring = [mu + radius * complex(math.cos(t), math.sin(t)) for t in (2 * math.pi * i / 96 for i in range(96))]
    outline = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in (to_svg((r + 1 / r) * rot) for r in ring)) + "Z"
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMaxYMid slice" focusable="false">'
            f'<g class="sl" fill="none" stroke="currentColor" stroke-width="1">{"".join(lines)}</g>'
            f'<path class="foil" d="{outline}"/></svg>')


def png_size(path: Path) -> tuple[int, int] | None:
    """A PNG's pixel size from its IHDR chunk (the first chunk, bytes 16-24), or None for a file
    that is not a PNG. Standard library only: the build has no imaging dependency."""
    try:
        with path.open("rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w, h = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
    return (w, h) if w > 0 and h > 0 else None


def ledger_rows(record: ValidationRecord, refs: dict[str, str], anchor: Callable[[str], str]) -> str:
    """The ledger's <tbody> rows: one group row per solver area (its pass count, linked to its
    table in the report) and the checks :meth:`ValidationRecord.ledger` picks for it.

    ``anchor(title)`` returns the report URL for a section heading."""
    out = []
    for area, checks in record.ledger():
        allc = record.area_checks(area)
        n_ok = sum(c.passed for c in allc)
        title = AREA_TITLES.get(area, area)
        out.append(f'<tr class="group"><th colspan="6" scope="rowgroup"><a href="{_attr(anchor(title))}">'
                   f"{_esc(title)}</a><span>{n_ok} of {len(allc)} checks pass</span></th></tr>")
        for c in checks:
            src = refs.get(c.source, c.source)
            src_html = (f'<a href="{_attr(anchor("ref-" + c.source))}">{_esc(src)}</a>' if c.source else "")
            revised = c.revised_section()
            mark = (f'<span class="flag">Band revised after this result was known: '
                    f'<a href="{_attr(anchor("section " + revised))}">report, section {_esc(revised.rstrip("."))}</a>'
                    "</span>") if revised else ""
            result = "pass" if c.passed else "<strong>fail</strong>"
            out.append(
                f'<tr class="{"fail" if not c.passed else "pass"}"><th scope="row">'
                f'<span class="case">{_esc(c.quantity)}</span><span class="src">{src_html}</span>{mark}</th>'
                f'<td class="num val" data-label="Aerolume">{fmt_value(c.value)}</td>'
                f'<td class="num ref" data-label="Published">{_esc(c.published_text())}</td>'
                f'<td class="num diff" data-label="Difference">{c.difference()}</td>'
                f'<td class="num band" data-label="Band">{c.band_text()}{band_gauge(c)}</td>'
                f'<td class="res" data-label="Result">{result}</td></tr>')
    return "\n        ".join(out)


def limits_html(repo: Path = REPO) -> str:
    """The README's "Limits" list, rendered for the home page: the page and the overview are one
    text, so they cannot drift apart. "" when the README has no such section."""
    path = repo / "README.md"
    md_text = doc_section(path.read_text(encoding="utf-8"), "Limits") if path.exists() else ""
    if not md_text:
        return ""
    return markdown_to_html(md_text, link=lambda url: None).html


#: 3D throughput quoted on the home page, used only if docs/theory_ns3d.md's benchmark table
#: cannot be read (bench_ns3d.py perf, 256 x 128 x 128, default configuration, 2026-10-31).
NS3D_RATE_FALLBACK = "44.9&ndash;49.2"


def ns3d_rate(repo: Path = REPO) -> str | None:
    """Steps per second of the default 4.2 M-cell 3D case from docs/theory_ns3d.md, or None."""
    path = repo / "docs" / "theory_ns3d.md"
    if not path.exists():
        return None
    for t in doc_tables(path.read_text(encoding="utf-8")):
        if not t.header or _fold(t.header[0]) != "configuration":
            continue
        try:
            c = t.col("steps/s")
        except KeyError:
            continue
        for r in t.rows:
            if r[0].replace(" ", "").startswith(("256×128×128", "256x128x128")) and "default" in r[0] and len(r) > c:
                m = re.search(r"(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?", r[c].replace("*", ""))
                if m:
                    return m.group(1) + (f"&ndash;{m.group(2)}" if m.group(2) else "")
    return None


def ns3d_idle_rate(repo: Path = REPO) -> str | None:
    """The default 4.2 M-cell case's steps per second on an idle GPU, from the sentence in
    docs/theory_ns3d.md that explains why it differs from the benchmark table's range (that range
    was measured on a GPU shared with other jobs). None when the sentence is missing.

    The home page quotes both, so it never shows one throughput while the overview (from the
    README) shows another without saying why."""
    path = repo / "docs" / "theory_ns3d.md"
    if not path.exists():
        return None
    text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
    m = re.search(r"idle GPU.{0,120}?\s(\d+(?:\.\d+)?) steps/s", text)
    return m.group(1) if m else None


def installer_facts(repo: Path = REPO) -> str:
    """One sentence of installer sizes and timings, read from docs/install.md's measurement table
    (its last column is the newest build), so the home page cannot drift from the guide. Empty
    when the table is missing; the page then only describes the installer."""
    path = repo / "docs" / "install.md"
    if not path.exists():
        return ""
    tables = doc_tables(path.read_text(encoding="utf-8"))
    try:
        t = next(t for t in tables if t.header and _fold(t.header[0]) == "step")
    except StopIteration:
        return ""
    last = len(t.header) - 1
    rows = {_fold(r[0]): r[last] for r in t.rows if len(r) > last}
    folder = rows.get(_fold("app folder"), "")
    installer = rows.get(_fold("installer"), "")
    if not folder or not installer:
        return ""
    size = re.match(r"\s*([\d.]+\s*[GM]B)", installer)
    when = _esc(t.header[last])
    return (f"Measured for the newest build ({when}): the installer is "
            f"{_esc(size.group(1) if size else installer)}, and the installed program "
            f"{_esc(folder)}.")


def measured_build_version(repo: Path = REPO) -> str | None:
    """The version named by docs/install.md's generated "Measured release build: Aerolume X" line,
    or None. The home page quotes that block's sizes, so a build of another version is reported."""
    path = repo / "docs" / "install.md"
    if not path.exists():
        return None
    m = re.search(r"Measured release build: Aerolume ([0-9][\w.+-]*?)[,*\s]", path.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def command_block(lines: list[str]) -> str:
    """Lines of a shell block, each in a span so a phone can wrap it with a hanging indent.

    Each span ends with its newline, so copying the block yields the commands exactly
    and the desktop layout (plain ``pre``) is unchanged.
    """
    return "".join(f'<span class="ln">{ln}\n</span>' for ln in lines[:-1]) + f'<span class="ln">{lines[-1]}</span>'


# =============================================================================
# Build
# =============================================================================

@dataclass
class Page:
    """A documentation page ready to be written."""
    spec: DocSpec
    rendered: Rendered


class SiteBuilder:
    """Writes the site into ``out`` (normally the website/ folder itself)."""

    def __init__(self, out: Path = HERE, repo: Path = REPO, *, repo_url: str | None = None,
                 site_url: str | None = None, env: dict[str, str] | None = None) -> None:
        self.out = out
        self.repo = repo
        env = os.environ if env is None else env
        self.site = SiteInfo(read_version(repo), build_date(),
                             repo_url or find_repo_url(repo, env),
                             (site_url or env.get("AEROLUME_SITE_URL") or citation_field(repo, "url") or "").strip() or None)
        self.images: set[Path] = set()
        self.warnings: list[str] = []
        self.missing_docs: list[str] = []
        self.tocs: dict[str, list[TocEntry]] = {}

    # ---------------------------------------------------------------- docs
    def available_docs(self) -> list[DocSpec]:
        """Configured documents that exist and are not empty, plus any other docs/*.md."""
        specs = []
        self.missing_docs = []
        for s in DOCS:
            if self._has_text(self.repo / s.source):
                specs.append(s)
            else:
                self.missing_docs.append(s.source)
        known = {s.source for s in DOCS}
        for path in sorted((self.repo / "docs").glob("*.md")):
            rel = path.relative_to(self.repo).as_posix()
            if rel not in known and self._has_text(path):
                title = path.stem.replace("_", " ").capitalize()
                m = re.search(r"^#\s+(.+)$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
                if m:
                    title = m.group(1).strip()
                specs.append(DocSpec(rel, path.stem.replace("_", "-").lower(), title, "Guide", ""))
        order = {g: k for k, g in enumerate(GROUP_ORDER)}
        return sorted(specs, key=lambda s: order.get(s.group, len(order)))  # stable within groups

    @staticmethod
    def _has_text(path: Path) -> bool:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())

    def _resolver(self, source: Path, slugs: dict[str, str]) -> Callable[[str], str | None]:
        def resolve(url: str) -> str | None:
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith(("#", "//")):
                return url
            path, _, frag = url.partition("#")
            frag = f"#{frag}" if frag else ""
            if not path:
                return frag
            target = (source.parent / path).resolve()
            try:
                rel = target.relative_to(self.repo.resolve()).as_posix()
            except ValueError:
                return url
            if rel in slugs:
                return f"{slugs[rel]}.html{frag}"
            if rel.startswith("docs/img/") and target.is_file():
                self.images.add(target)
                return "img/" + rel[len("docs/img/"):]
            if rel == "LICENSE":
                return "../LICENSE.txt"
            if not self.site.repo_url:
                return None                                    # shown as text, not a dead link
            kind = "tree" if target.is_dir() else "blob"
            return f"{self.site.repo_url}/{kind}/main/{rel}{frag}"
        return resolve

    def _xref(self, source: Path, specs: list[DocSpec], tocs: dict[str, list[TocEntry]]) -> Xref:
        """Cross-references between documents, labelled with page titles and section numbers."""
        by_name: dict[str, DocSpec] = {}
        for s in specs:
            by_name[s.source] = s
            by_name.setdefault(Path(s.source).name, s)

        def xref(name: str, section: str | None) -> tuple[str, str] | None:
            rel = (source.parent / name).resolve()
            try:
                key = rel.relative_to(self.repo.resolve()).as_posix()
            except ValueError:
                key = name
            spec = by_name.get(key) or by_name.get(name) or by_name.get(Path(name).name)
            if spec is None:
                if Path(name).name == "DESIGN.md" and self.site.repo_url:
                    label = "design contracts" + (f", §{section}" if section else "")
                    return f"{self.site.repo_url}/blob/main/DESIGN.md", label
                return None
            href, label = f"{spec.slug}.html", spec.title
            if section:
                label += f", §{section}"
                for e in tocs.get(spec.slug, []):
                    if re.match(rf"^{re.escape(section)}[.\s]", e.text):
                        href += f"#{e.id}"
                        break
            return href, label
        return xref

    def _image_size(self, src: str) -> tuple[int, int] | None:
        """Size of a docs image by its site URL (img/... maps to docs/img/... in the repository)."""
        if not src.startswith("img/") or ".." in src:
            return None
        return png_size(self.repo / "docs" / "img" / src[len("img/"):].split("#")[0])

    def _figure_hook(self) -> Callable[[str, str, str], str | None]:
        """Lone images that have web panels (tools/make_doc_figures.py) render as a panel grid."""
        index_path = self.out / "assets" / "docfig" / "docfig.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

        def figure(raw: str, src: str, alt: str) -> str | None:
            entry = index.get(Path(raw).name)
            if not entry or not src.startswith("img/"):
                return None
            panels = []
            for p in entry["panels"]:
                imgs = "".join(
                    f'<img class="only-{theme}" src="../assets/docfig/{_attr(p[theme])}" width="{p["width"]}" '
                    f'height="{p["height"]}" alt="{_attr(p["alt"])}" loading="lazy">' for theme in ("light", "dark"))
                panels.append(f'<a class="panel" href="{_attr(src)}" title="Open the full-resolution figure">{imgs}</a>')
            n = len(panels)
            return (f'<figure class="doc-fig n{n}"><div class="panels">{"".join(panels)}</div>'
                    f'<figcaption>{_esc(alt)}. <a href="{_attr(src)}">Full-resolution figure</a> '
                    f"(PNG, as in the repository).</figcaption></figure>")
        return figure

    def render_docs(self) -> list[Page]:
        """Convert every available document (two passes: the second knows every page's sections)."""
        specs = self.available_docs()
        slugs = {s.source: s.slug for s in specs}
        first = {s.slug: MarkdownRenderer(self._resolver(self.repo / s.source, slugs))
                 .render((self.repo / s.source).read_text(encoding="utf-8")).toc for s in specs}
        self.tocs = first
        figure = self._figure_hook()
        pages = []
        for spec in specs:
            src = self.repo / spec.source
            renderer = MarkdownRenderer(self._resolver(src, slugs), xref=self._xref(src, specs, first),
                                        figure=figure, size=self._image_size)
            pages.append(Page(spec, renderer.render(src.read_text(encoding="utf-8"))))
        return pages

    def _nav(self, pages: list[Page], current: str, *, mobile: bool = False) -> str:
        groups: dict[str, list[Page]] = {}
        for p in pages:
            groups.setdefault(p.spec.group, []).append(p)
        html_parts = []
        for group, items in groups.items():
            links = "".join(
                f'<li><a href="{p.spec.slug}.html"'
                f'{" aria-current=" + chr(34) + "page" + chr(34) if p.spec.slug == current else ""}>'
                f"{_esc(p.spec.title)}</a></li>" for p in items)
            html_parts.append(f"<h2>{_esc(group)}</h2><ul>{links}</ul>" if not mobile else links)
        if mobile:
            return f'<ul><li><a href="index.html">All documentation</a></li>{"".join(html_parts)}</ul>'
        index_cur = ' aria-current="page"' if current == "index" else ""
        return (f'<ul><li><a href="index.html"{index_cur}>All documentation</a></li></ul>'
                + "".join(html_parts))

    @staticmethod
    def _toc(toc: list[TocEntry]) -> str:
        return "".join(f'<li class="l{e.level}"><a href="#{e.id}">{_esc(e.text)}</a></li>' for e in toc)

    def _doc_page(self, title: str, path: str, nav_html: str, mobile_nav: str, main_html: str,
                  toc: list[TocEntry], *, math_: bool, description: str) -> str:
        toc_html = self._toc(toc)
        aside = (f'<aside class="docs-toc" aria-label="On this page"><div class="sticky"><h2>On this page</h2>'
                 f"<ul>{toc_html}</ul></div></aside>") if toc_html else '<aside class="docs-toc"></aside>'
        mobile_toc = (f'<details class="m-toc"><summary>On this page</summary><ul>{toc_html}</ul></details>'
                      if toc_html else "")
        return f"""<!doctype html>
<html lang="en">
<head>
{head(title, "../", self.site, path=path, description=description, math_=math_)}
</head>
<body class="docs-page">
<a class="skip" href="#main">Skip to content</a>
{topbar("../", self.site, "docs")}
<div class="docs-layout">
<nav class="docs-nav" aria-label="Documentation"><div class="sticky">{nav_html}</div></nav>
<main id="main" class="doc">
<div class="docs-mobile"><details class="m-nav"><summary>Documentation</summary>{mobile_nav}</details>{mobile_toc}</div>
{main_html}
</main>
{aside}
</div>
{footer("../", self.site)}
<script src="../assets/site.js" defer></script>
</body>
</html>
"""

    def docs_intro(self, pages: list[Page]) -> str:
        """The docs index's first paragraph, naming only the groups that were built."""
        present = [g for g in GROUP_ORDER if any(p.spec.group == g for p in pages)]
        phrases = [GROUP_PHRASES[g] for g in present]
        if not phrases:
            listing = "The documentation"
        elif len(phrases) == 1:
            listing = phrases[0][:1].upper() + phrases[0][1:]
        else:
            listing = ", ".join(phrases[:-1]) + " and " + phrases[-1]
            listing = listing[:1].upper() + listing[1:]
        where = (f'the <a href="{_attr(self.site.repo_url)}">repository</a>' if self.site.repo_url
                 else "the repository")
        return f"<p>{listing}. Each page is generated from the Markdown in {where}.</p>"

    def write_docs(self, pages: list[Page]) -> None:
        """Write docs/index.html and one page per document, and copy referenced images."""
        docs_dir = self.out / "docs"
        if docs_dir.exists():
            for old in docs_dir.glob("*.html"):
                old.unlink()
        docs_dir.mkdir(parents=True, exist_ok=True)
        for k, page in enumerate(pages):
            spec = page.spec
            prev_p = pages[k - 1] if k > 0 else None
            next_p = pages[k + 1] if k + 1 < len(pages) else None
            pager = ['<nav class="pager" aria-label="Previous and next page">']
            if prev_p:
                pager.append(f'<a class="prev" href="{prev_p.spec.slug}.html"><span>Previous</span>'
                             f"<b>{_esc(prev_p.spec.title)}</b></a>")
            if next_p:
                pager.append(f'<a class="next" href="{next_p.spec.slug}.html"><span>Next</span>'
                             f"<b>{_esc(next_p.spec.title)}</b></a>")
            pager.append("</nav>")
            body = page.rendered.html
            if not page.rendered.title:
                body = f"<h1>{_esc(spec.title)}</h1>\n" + body
            source = (f'<p class="crumb src-link">Source: <a href="{_attr(self.site.repo_url)}/blob/main/'
                      f'{spec.source}">{_esc(spec.source)}</a></p>\n') if self.site.repo_url else ""
            main = (f'<p class="crumb"><a href="index.html">Documentation</a> / {_esc(spec.group)}</p>\n{body}'
                    + source + "".join(pager))
            page_html = self._doc_page(f"Aerolume docs · {spec.title}", f"docs/{spec.slug}.html",
                                       self._nav(pages, spec.slug), self._nav(pages, spec.slug, mobile=True), main,
                                       page.rendered.toc, math_=page.rendered.has_math,
                                       description=spec.summary or SITE_DESCRIPTION)
            (docs_dir / f"{spec.slug}.html").write_text(page_html, encoding="utf-8", newline="\n")

        groups: dict[str, list[Page]] = {}
        for p in pages:
            groups.setdefault(p.spec.group, []).append(p)
        listing = []
        for group, items in groups.items():
            rows = "".join(f'<li><a href="{p.spec.slug}.html">{_esc(p.spec.title)}</a>'
                           f"<p>{_esc(p.spec.summary or p.rendered.title)}</p></li>" for p in items)
            listing.append(f'<h2 class="doc-index-group" id="{slugify(group)}">{_esc(group)}</h2>'
                           f'<ul class="doc-index">{rows}</ul>')
        main = "<h1>Documentation</h1>\n" + self.docs_intro(pages) + "\n" + "\n".join(listing)
        description = "Aerolume documentation: " + ", ".join(p.spec.title for p in pages) + "."
        index_html = self._doc_page("Aerolume docs", "docs/index.html", self._nav(pages, "index"),
                                    self._nav(pages, "index", mobile=True), main, [], math_=False,
                                    description=description)
        (docs_dir / "index.html").write_text(index_html, encoding="utf-8", newline="\n")

        img_dir = docs_dir / "img"
        for src in sorted(self.images):
            rel = src.relative_to((self.repo / "docs" / "img").resolve())
            dst = img_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # ---------------------------------------------------------------- home page
    def figure_values(self) -> dict[str, str]:
        """Placeholders describing the two flow figures, from assets/figures.json."""
        path = self.out / "assets" / "figures.json"
        figs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        a, v = figs.get("airfoil", {}), figs.get("vehicle", {})
        apx, vpx = a.get("pixels", [2816, 1126]), v.get("pixels", [3072, 809])

        def mobile(fig: dict, name: str, default: str) -> str:
            return fig.get("mobile", {}).get(name, {}).get("file", default)

        def mobile_px(fig: dict, name: str, default: list[int]) -> list[int]:
            return fig.get("mobile", {}).get(name, {}).get("pixels", default)

        amp, vmp = mobile_px(a, "smoke", [1100, 826]), mobile_px(v, "vorticity", [1100, 453])

        return {
            "FIG_A_SMOKE": a.get("files", {}).get("smoke", "flow-airfoil-smoke.webp"),
            "FIG_A_VORT": a.get("files", {}).get("vorticity", "flow-airfoil-vorticity.webp"),
            "FIG_A_SMOKE_M": mobile(a, "smoke", "flow-airfoil-smoke-m.webp"),
            "FIG_A_VORT_M": mobile(a, "vorticity", "flow-airfoil-vorticity-m.webp"),
            "FIG_A_W": str(apx[0]), "FIG_A_H": str(apx[1]),
            "FIG_A_MW": str(amp[0]), "FIG_A_MH": str(amp[1]),
            "FIG_A_NACA": _esc(str(a.get("naca", "2412"))),
            "FIG_A_ALPHA": f"{a.get('alpha_deg', 8.0):g}",
            "FIG_A_RE": fmt_reynolds(float(a.get("re", 5000))),
            "FIG_A_T": f"{a.get('t', 12.0):.1f}",
            "FIG_A_NX": str(a.get("nx", "")), "FIG_A_NY": str(a.get("ny", "")),
            "FIG_A_CPR": str(a.get("cells_per_ref", 256)),
            "FIG_A_STEPS": f"{a.get('steps', 0):,}".replace(",", "&#8201;"),
            "FIG_A_GPU": _esc(str(a.get("gpu", "NVIDIA GPU")).replace("NVIDIA GeForce ", "")),
            "FIG_A_VLIM": f"{a.get('vorticity_limit', 0):.1f}",
            "FIG_A_DRAWN": figure_drawn(a, self.site.version, self.site.date),
            "FIG_V_VORT": v.get("files", {}).get("vorticity", "flow-vehicle-vorticity.webp"),
            "FIG_V_VORT_M": mobile(v, "vorticity", "flow-vehicle-vorticity-m.webp"),
            "FIG_V_W": str(vpx[0]), "FIG_V_H": str(vpx[1]),
            "FIG_V_MW": str(vmp[0]), "FIG_V_MH": str(vmp[1]),
            "FIG_V_LEN": f"{v.get('length_m', 4.5):g}",
            "FIG_V_RE": fmt_reynolds(float(v.get("re", 2e5))),
            "FIG_V_CPR": str(v.get("cells_per_ref", 512)),
            "FIG_V_T": f"{v.get('t', 4.0):.1f}",
        }

    def install_values(self) -> dict[str, str]:
        """The get-Aerolume block and every link that needs the repository address.

        Without a public repository there is nothing to clone, so the page says so in one plain
        sentence instead of printing a placeholder address, and the build block holds only the
        commands that run inside a copy of the repository."""
        url = self.site.repo_url
        build_cmds = ["pip install pyinstaller          # and Inno Setup 6",
                      "python packaging/build.py --variant cuda",
                      "python packaging/build.py --status    # does the build match the sources?"]
        if url:
            name = url.rstrip("/").rsplit("/", 1)[-1] or "aerolume"
            clone = [f"git clone {_esc(url)}.git", f"cd {_esc(name)}"]
            source_btn = f'\n      <a class="btn btn-secondary" href="{_attr(url)}">Source code</a>'
            repo_status = (f'Clone the <a href="{_attr(url)}">repository</a> and build the installer with the '
                           "commands beside this text.")
            build_label = "Clone and build the installer"
            oss_links = (f'Source code: <a href="{_attr(url)}">the repository</a>; questions and bug reports: '
                         f'<a href="{_attr(url)}/issues">the issue tracker</a>.')
        else:
            clone = []
            source_btn = ""
            repo_status = ("The source repository is not public yet either; once it is, this page will link to it "
                           "and show the command that clones it.")
            build_label = "Build the installer, inside a copy of the repository"
            oss_links = ("The repository is not public yet, so this site cannot link to the source code or an "
                         "issue tracker; it will once the repository is published.")
        return {
            "CMD_BUILD": command_block(clone + build_cmds),
            "BUILD_LABEL": build_label,
            "REPO_STATUS": repo_status,
            "INSTALLER_FACTS": (" " + installer_facts(self.repo)).rstrip(),
            "SOURCE_BUTTON": source_btn,
            "OSS_LINKS": oss_links,
        }

    def doc_anchor(self, slug: str, heading: str, *, after: str = "") -> str:
        """``docs/<slug>.html#id`` of the first section whose title is ``heading`` (or starts with
        it, for numbered titles such as "3."), looked up in the rendered table of contents; with
        ``after``, only sections following the one whose title starts with ``after``. Falls back
        to the page itself, so a renamed heading costs a jump, never a broken link."""
        toc = self.tocs.get(slug, [])
        start = 0
        if after:
            start = next((k for k, e in enumerate(toc) if e.text.startswith(after)), len(toc))
        for e in toc[start:]:
            if e.text == heading or (heading.endswith(".") and e.text.startswith(heading)):
                return f"docs/{slug}.html#{e.id}"
        self.warnings.append(f"home page: no section '{heading}' in docs/{slug}; the link goes to the page top")
        return f"docs/{slug}.html"

    def report_anchor(self, key: str) -> str:
        """Links of the ledger into the validation report: an area's table in section 9, a
        reference of the automated run, or a numbered section."""
        if key.startswith("ref-"):
            return f"docs/validation.html#{key}"
        if key.startswith("section "):
            return self.doc_anchor("validation", key[len("section "):])
        return self.doc_anchor("validation", key, after="9.")

    def version_problems(self, record: ValidationRecord, releases: dict[str, str]) -> list[str]:
        """Warnings for every place where a version the page quotes or relies on is not the one
        the site describes: pyproject.toml, the changelog, the validation run and the installer
        measurements in docs/install.md. The page itself stays truthful in each case (see
        run_facts); the warnings tell the release manager what still has to be re-run."""
        version, out = self.site.version, []
        pv = pyproject_version(self.repo)
        if pv and pv != version:
            out.append(f"version mismatch: src/aerolume/__init__.py says {version}, pyproject.toml {pv}")
        if version not in releases:
            out.append(f"CHANGELOG.md has no released entry '## [{version}] - <date>'; the page does not "
                       "call this version a release")
        ran = str(record.meta.get("aerolume", "") or "") if record.ok else ""
        if ran and ran != version and ran not in releases:
            out.append(f"the validation record was made by a development build ({ran}), not {version}; the "
                       "home page says so. Re-run validation/run_all.py --all to quote the release by version")
        built = measured_build_version(self.repo)
        if built and built != version:
            out.append(f"docs/install.md measures the installer of {built}, not {version}; the home page quotes "
                       "its sizes without a version. Rebuild and run packaging/build.py --write-measurements")
        return out

    def render_index(self) -> str:
        """Fill templates/index.html."""
        tpl = (HERE / "templates" / "index.html").read_text(encoding="utf-8")
        perf = perf_rows(self.repo)
        record = ValidationRecord(self.repo)
        self.warnings.extend(record.problems)
        rate3d = ns3d_rate(self.repo)
        idle3d = ns3d_idle_rate(self.repo) if rate3d else None
        if rate3d is None:
            self.warnings.append("3D throughput not found in docs/theory_ns3d.md; the home page quotes the "
                                 f"built-in copy ({html.unescape(NS3D_RATE_FALLBACK)} steps/s)")
        limits = limits_html(self.repo)
        if not limits:
            self.warnings.append("README.md has no 'Limits' section; the home page's limits block is empty")
        version = self.site.version
        refs = reference_labels(self.repo)
        releases = released_versions(self.repo)
        self.warnings.extend(self.version_problems(record, releases))
        values = {
            "HEAD": head("Aerolume", "", self.site, path=""),
            "TOPBAR": topbar("", self.site, home=True),
            "FOOTER": footer("", self.site),
            "VERSION": _esc(version),
            "RELEASE_NOTE": release_note(version, releases),
            **perf_facts(perf),
            "PERF_LINK": self.doc_anchor("theory-ns2d", "10.") if self.tocs else "docs/theory-ns2d.html",
            "NS3D_RATE": rate3d or NS3D_RATE_FALLBACK,
            # The table's range was measured on a shared GPU; the overview (README) quotes the idle
            # figure, so the page gives both and says which is which.
            "NS3D_NOTE": (f", same GPU shared with other jobs ({_esc(idle3d)} when idle)" if idle3d
                          else ", same GPU"),
            "TRUST_TALLY": validation_tally(record),
            "LEDGER_ROWS": ledger_rows(record, refs, self.report_anchor),
            "CAVITY_PLOT": cavity_plot(record, refs, self.report_anchor),
            "REVISED_NOTE": revised_note(record),
            "RUN_FACTS": run_facts(record, version, releases),
            "REPORT_RUN": self.doc_anchor("validation", "9."),
            "REPORT_NOT_VALIDATED": self.doc_anchor("validation", "10."),
            "LIMITS": limits,
            "STREAKLINES": streakline_svg(),
            "SHOT_RESULTS3D": feature_shot(self.out / "assets", "results3d.png"),
            "SHOT_HEAT": feature_shot(self.out / "assets", "heat.png", display_px=370),
            "SHOT_PHYSICS": feature_shot(self.out / "assets", "physics.png", display_px=370),
            "SHOT_EXPLORE": feature_shot(self.out / "assets", "explore.png"),
            "SHOT_SCRIPT": feature_shot(self.out / "assets", "script.png"),
            **self.install_values(),
            **self.figure_values(),
        }

        def sub(m: re.Match) -> str:
            key = m.group(1)
            if key not in values:
                raise KeyError(f"templates/index.html uses an unknown placeholder {{{{{key}}}}}")
            return values[key]

        return re.sub(r"\{\{([A-Z0-9_]+)\}\}", sub, tpl)

    def write_favicon(self) -> None:
        """assets/favicon.svg: the NACA 2412 mark on a square tile, in the site's ink and paper."""
        d, _ = airfoil_path(width=32, height=32)
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
               '<rect width="32" height="32" rx="6" fill="#0e0e0e"/>'
               f'<path d="{d}" fill="#f6f6f6" transform="translate(0 2)"/></svg>\n')
        (self.out / "assets" / "favicon.svg").write_text(svg, encoding="utf-8", newline="\n")

    def write_license(self) -> None:
        """LICENSE.txt next to index.html: the footer's licence link stays on the site."""
        src = self.repo / "LICENSE"
        if src.exists():
            shutil.copy2(src, self.out / "LICENSE.txt")
        else:
            self.warnings.append("LICENSE not found; the Licence link will be broken")

    # ---------------------------------------------------------------- artifact
    def write_artifact(self, dst: Path) -> Path:
        """Copy the built site to ``dst`` with index.html reduced to body content.

        The page's own stylesheet and scripts (site.css, site.js, hero/hero.js) are inlined,
        so the Artifact renders and animates as a single page; the images stay relative files
        next to it (publish them with the page)."""
        if dst.exists():
            shutil.rmtree(dst)
        copy_site(self.out, dst)
        index = dst / "index.html"
        page = inline_local_assets(index.read_text(encoding="utf-8"), self.out)
        index.write_text(strip_document(page), encoding="utf-8", newline="\n")
        return dst

    # ---------------------------------------------------------------- all
    def build(self) -> list[str]:
        """Build everything; returns the list of broken-link messages (empty when clean).

        Non-fatal problems (missing documents, an unknown repository address, values the
        validation table could not find) are collected in :attr:`warnings`.
        """
        (self.out / "assets").mkdir(parents=True, exist_ok=True)
        self.write_favicon()
        self.write_license()
        pages = self.render_docs()
        for src in self.missing_docs:
            self.warnings.append(f"{src} is listed in DOCS but missing or empty; its page is left out")
        if not self.site.repo_url:
            self.warnings.append("no repository URL (set [project.urls] Repository in pyproject.toml or "
                                 "AEROLUME_REPO_URL); Source/Issues links are left out and the home "
                                 "page says the source is not public yet")
        self.write_docs(pages)
        (self.out / "index.html").write_text(self.render_index(), encoding="utf-8", newline="\n")
        stale = self.out / "_artifact"
        if stale.is_dir():                     # older builds wrote the Artifact copy inside website/
            shutil.rmtree(stale)
        return check_links(self.out, exclude=("templates", "tests", "tools"))


SITE_ITEMS = ("index.html", "LICENSE.txt", "docs", "assets")


def copy_site(src: Path, dst: Path) -> None:
    """Copy the deployable files (pages and assets, not the build tooling or its indexes)."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in SITE_ITEMS:
        s = src / name
        if s.is_dir():
            shutil.copytree(s, dst / name, ignore=shutil.ignore_patterns("MANIFEST.md", "figures.json",
                                                                         "docfig.json"))
        elif s.is_file():
            shutil.copy2(s, dst / name)


_LOCAL_CSS = re.compile(r'<link rel="stylesheet" href="((?:\./)?assets/[^"]+\.css)">')
_LOCAL_JS = re.compile(r'<script src="((?:\./)?assets/[^"]+\.js)"(?: defer)?></script>')


def inline_local_assets(page: str, site_dir: Path) -> str:
    """Replace links to the site's own CSS and JS files with their contents.

    External resources (pinned CDN scripts) are left alone; ``url()`` references inside the
    stylesheet (the self-hosted fonts) are rewritten to stay relative to the page. An inline script
    cannot be deferred, so the scripts stay where they were (the end of the body), which is
    after every element they look up. A ``</`` in the inlined text is escaped so it cannot
    close the element early."""
    def css(m: re.Match) -> str:
        text = (site_dir / m.group(1)).read_text(encoding="utf-8")
        # url(...) in a stylesheet is relative to the stylesheet; inlined, it would be read
        # relative to the page, so local ones get the stylesheet's folder in front.
        base = Path(m.group(1)).parent.as_posix()

        def url(u: re.Match) -> str:
            ref = u.group(2)
            if re.match(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|/|#)", ref) or base in ("", "."):
                return u.group(0)
            return f"url({u.group(1)}{base}/{ref}{u.group(1)})"

        text = re.sub(r"""url\((["']?)([^"')]+)\1\)""", url, text)
        return "<style>\n" + text.replace("</", "<\\/") + "\n</style>"

    def js(m: re.Match) -> str:
        text = (site_dir / m.group(1)).read_text(encoding="utf-8")
        return "<script>\n" + text.replace("</", "<\\/") + "\n</script>"

    return _LOCAL_JS.sub(js, _LOCAL_CSS.sub(css, page))


def strip_document(page: str) -> str:
    """Reduce a full HTML document to Artifact page content.

    Drops the doctype and the html/head/body wrapper tags and the <meta> elements (the
    Artifact skeleton supplies charset and viewport); keeps <title>, <link>, <style> and
    <script> from the head, in order, ahead of the body content.
    """
    m_head = re.search(r"<head[^>]*>(.*?)</head>", page, flags=re.DOTALL | re.IGNORECASE)
    m_body = re.search(r"<body[^>]*>(.*)</body>", page, flags=re.DOTALL | re.IGNORECASE)
    head_part = m_head.group(1) if m_head else ""
    body_part = m_body.group(1) if m_body else page
    head_part = re.sub(r"<meta\b[^>]*>\s*", "", head_part, flags=re.IGNORECASE)
    body_part = re.sub(r"<!doctype[^>]*>|</?html[^>]*>", "", body_part, flags=re.IGNORECASE)
    return head_part.strip() + "\n" + body_part.strip() + "\n"


_REF = re.compile(r'\b(?:href|src|srcset)="([^"]+)"')
_ID = re.compile(r'\bid="([^"]+)"')


def check_links(root: Path, exclude: Iterable[str] = ()) -> list[str]:
    """Every relative href/src/srcset must exist, and every #fragment must name an id in its target."""
    excluded = {root / e for e in exclude}
    pages = [p for p in root.rglob("*.html") if not any(ex in p.parents for ex in excluded)]
    ids: dict[Path, set[str]] = {}

    def ids_of(p: Path) -> set[str]:
        if p not in ids:
            ids[p] = set(_ID.findall(p.read_text(encoding="utf-8"))) if p.exists() else set()
        return ids[p]

    problems = []
    for page in sorted(pages):
        text = page.read_text(encoding="utf-8")
        for ref in _REF.findall(text):
            ref = html.unescape(ref).split(" ")[0]           # srcset: "file.webp 1100w"
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:|^//", ref):
                continue
            path, _, frag = ref.partition("#")
            target = (page.parent / path).resolve() if path else page.resolve()
            where = page.relative_to(root).as_posix()
            if not target.exists():
                problems.append(f"{where}: missing {ref}")
            elif frag and target.suffix == ".html" and frag not in ids_of(target):
                problems.append(f"{where}: no anchor #{frag} in {path or where}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Aerolume website.")
    ap.add_argument("--dist", type=Path, help="also assemble a clean deploy folder here (for GitHub Pages)")
    ap.add_argument("--strict", action="store_true", help="exit with status 1 if a link or anchor is broken")
    ap.add_argument("--artifact", type=Path, help="also write the claude.ai Artifact copy into this folder")
    ap.add_argument("--site-url", help="the site's public address, for absolute og:image/og:url")
    args = ap.parse_args(argv)
    if args.artifact:
        art = args.artifact.resolve()
        if HERE in (art, *art.parents):
            print("error: --artifact must be outside website/ so it is never deployed", file=sys.stderr)
            return 2
    builder = SiteBuilder(site_url=args.site_url)
    problems = builder.build()
    for msg in builder.warnings:
        warn(msg)
    for msg in problems:
        warn(msg)
    if args.artifact:
        builder.write_artifact(args.artifact.resolve())
    if args.dist:
        if args.dist.exists():
            shutil.rmtree(args.dist)
        copy_site(HERE, args.dist)
        (args.dist / ".nojekyll").write_text("", encoding="utf-8")
    pages = len(list((HERE / "docs").glob("*.html")))
    print(f"built index.html and {pages} docs pages ({len(problems)} link problems, "
          f"{len(builder.warnings)} warnings)")
    return 1 if problems and args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
