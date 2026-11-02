"""The Assistant dock: state a goal, the assistant plans, runs and reports an experiment.

The engine is :mod:`aerolume.ai` (provider, tools, loop, experiment log); this
widget is its window. Layout, top to bottom::

    Local · qwen3:8b · Ollama 0.40.2                       [New experiment] [⚙]
    ----------------------------------------------------------------------------
    You
    Find the angle of attack with the best L/D of NACA 2412 at 20 m/s
    ----------------------------------------------------------------------------
    Assistant
    ✓ Read the project            1 design, 0 simulations                 0.0 s ▸
    ✓ Record plan                 Objective · Method · Variables …               ▸
    ◌ Polar analysis              NACA 2412 · 20 m/s · α −4…14°          43 %   ▸
    [approval card: title, detail, code, Approve / Decline]
    [figure thumbnail]
    The best L/D is 108.7 at α = 5° (Re 1.37×10⁶) …
    ✓ Done in 41 s · experiment log 20261011-143005-find-the-angle…   Open log
    ----------------------------------------------------------------------------
    ◌ Running run_polar · 12 s
    [ goal box …                                                         ] [Run]

Design rules (docs/ui_guidelines.md): messages are plain blocks with a "You" /
"Assistant" caption, not chat bubbles; tool calls are one-line rows with a
status glyph that fold out to their input and result; approvals are the one
framed block, because they need an answer; figures are inline thumbnails that
open full size; no emoji, no gradients, no avatars.

Threads. The agent runs on a worker thread and reports through one queued
signal (:class:`_Bridge`). Approvals block the worker on a ``threading.Event``
that the card's buttons set (Stop releases it too). Everything the tools change
goes through :class:`aerolume.app.scripting.AppHost`: project edits on the GUI
thread via ``AppState``, simulations through the window's ``JobManager`` (they
appear in the Project tree and on the Simulate page, and the window's Stop
stops them). Approved ``run_python`` code runs through the Script console's
engine (:class:`aerolume.app.scripting.ScriptEngine`).
"""
from __future__ import annotations

import contextlib
import html
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QPixmap, QShortcut, QTextDocument, QTextOption
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...ai.plaintext import plain_math
from .. import theme
from .common import (
    SPACING,
    CollapsibleSection,
    Divider,
    ElidedLabel,
    IconButton,
    KeyValueList,
    MessageLine,
    Spinner,
    button,
    label,
)

#: how each tool reads in the transcript
TOOL_TITLES = {
    "project_overview": "Read the project", "describe_design": "Describe design",
    "create_naca_airfoil": "Create NACA airfoil", "create_template": "Create design from template",
    "create_wing": "Create wing", "derive_design": "Derive design variant", "estimate_cost": "Estimate cost",
    "run_polar": "Polar analysis", "run_wing": "Wing analysis", "run_cfd": "CFD run", "run_sweep": "Parameter sweep",
    "get_results": "Read results", "compare_results": "Compare results", "make_plot": "Figure",
    "write_report": "Write report", "run_python": "Python script", "delete_item": "Delete",
    "record_plan": "Record plan", "record_conclusion": "Record conclusion",
    "describe_simulation": "Describe simulation", "run_simulation": "Run simulation",
    "create_case": "Create reference case", "suggest_settings": "Suggest settings",
    "run_grid_study": "Grid study", "compare_reference": "Compare with reference",
}

#: example goals for a project without designs (clicking one fills the goal box). They say "create" for
#: geometry the project does not hold yet. A project with designs gets examples drawn from its own designs and
#: physics instead (:func:`project_examples`); :func:`selection_example` adds one about the selected design.
EXAMPLES = (
    "Create a NACA 4412 airfoil and find its angle of best L/D at 20 m/s with the polar analysis.",
    "Create 2D Ahmed bodies with slant angles of 20°, 25° and 30° and compare their drag at 40 m/s (draft quality).",
    "How does the aspect ratio (6, 8, 10) of a 1.2 m² rectangular wing change its induced drag at 25 m/s?",
)


def _enabled(items: Any) -> list:
    return [x for x in (items or []) if getattr(x, "enabled", True)]


def _case_of(design: Any) -> str:
    """The reference case a design was made by ("cavity", "heated_cylinder", "fan_room"), or ""."""
    src = getattr(design, "source", None)
    return str(src.get("case", "")) if isinstance(src, dict) and src.get("type") == "case" else ""


def physics_questions(sim: Any, design: Any) -> list[tuple[str, str]]:
    """Every example goal a simulation's placed physics (heat, fans) supports, as (kind, goal); [] for aero.

    Worded as questions the tools answer: a reference comparison for the benchmark
    cases (compare_reference), Nu and heat rate for heat transfer, delivered thrust
    and volume flow for fans (fans_report), the grid error of Nu (run_grid_study).
    Nothing about the temperature at a point: no tool reports one. The first entry
    is the run's main question (:func:`physics_example`).
    """
    if sim is None or getattr(sim, "analysis", "") != "cfd":
        return []
    name = str(getattr(sim, "name", "") or "this simulation")
    case = _case_of(design)
    out: list[tuple[str, str]] = []
    if case == "cavity":
        out.append(("cavity", (f"Compare the mean Nusselt number of “{name}” with de Vahl Davis (1983): the "
                               "difference in percent and its uncertainty.")))
    elif case == "heated_cylinder":
        out.append(("heated_cylinder", (f"Compare the Nusselt number of “{name}” with the Lange et al. (1998) "
                                        "correlation, with its uncertainty.")))
    if _enabled(getattr(sim, "fans", None)):
        out.append(("fan", (f"What thrust and volume flow does the fan in “{name}” deliver, and how far can they "
                            "be trusted?")))
    th = getattr(sim, "thermal", None)
    if th is not None and getattr(th, "enabled", False):
        if not case:
            out.append(("heat", (f"What are the Nusselt number and heat rate of “{name}”, and how far can they be "
                                 "trusted?")))
        out.append(("grid", (f"How large is the grid error of the Nusselt number of “{name}”? Run a draft grid "
                             "study and quote the GCI; estimate the cost first.")))
    return out


def physics_example(sim: Any, design: Any) -> str:
    """The main example goal about a simulation's placed physics (heat, fans), or "" for a plain aero run."""
    q = physics_questions(sim, design)
    return q[0][1] if q else ""


def physics_examples(project: Any, limit: int = 3) -> list[str]:
    """Example goals from the open project's own physics, newest simulation first (at most ``limit``).

    A heat project offered only aerodynamic examples (drag of “Heated box”, NACA 4412,
    Ahmed bodies, wing aspect ratio; critic round 3.2). One example per kind of
    physics, so three fan rooms do not crowd out the cavity. Only each run's main
    question is offered here; :func:`project_examples` adds the others.
    """
    if project is None:
        return []
    designs = getattr(project, "designs", {}) or {}
    out: list[str] = []
    kinds: set[str] = set()
    for sim in reversed(list((getattr(project, "simulations", {}) or {}).values())):
        d = designs.get(getattr(sim, "design_id", ""))
        text = physics_example(sim, d)
        if not text:
            continue
        kind = _case_of(d) or ("fan" if _enabled(getattr(sim, "fans", None)) else "heat")
        if kind in kinds:
            continue
        kinds.add(kind)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _speed_text(speed: float) -> str:
    """A run's speed as a goal states it: three significant figures ("14.6", not the 14.6065 m/s a
    Reynolds-number study computes), never in exponent form."""
    return f"{float(f'{float(speed):.3g}'):g}"


def design_questions(design: Any, speed: float | None = None, sim: Any = None) -> list[str]:
    """Example goals about one design, worded for its kind and physics; the first is the main one.

    ``sim`` (the design's newest simulation) words them for that run's physics: a
    heated cavity gets Nusselt questions, not "compare the drag". Each kind gets
    two or three questions the tools answer (a polar's best L/D and stall, a
    wing's induced drag, a 3D body's drag and its trust), so a project with one
    design still fills the example slots with its own work.
    """
    if design is None:
        return []
    physics = [g for _k, g in physics_questions(sim, design)]
    if physics:
        return physics
    name = str(getattr(design, "name", "") or "this design")
    if any(getattr(b, "thermal", "adiabatic") != "adiabatic"
           for b in list(getattr(design, "bodies", []) or []) + list(getattr(design, "meshes", []) or [])):
        v = f"{_speed_text(speed)} m/s" if speed and speed > 0 else "1 m/s"
        return [(f"What are the Nusselt number and heat rate of the heated “{name}” at {v} (draft quality), and "
                 "how far can they be trusted? Estimate the cost first."),
                (f"How does the heat rate of the heated “{name}” change between 0.5 and 2 m/s (draft quality)? "
                 "Estimate the cost first.")]
    v = f"{_speed_text(speed)} m/s" if speed and speed > 0 else "30 m/s"
    kind = getattr(design, "kind", "")
    three = int(getattr(design, "dimension", 2) or 2) == 3
    if kind == "wing":
        return [(f"Find the angle of attack with the best L/D of the wing “{name}” at {v}, and say how confident "
                 "that answer is."),
                f"What are the induced drag and the span efficiency of the wing “{name}” at {v}?",
                f"How do the lift and drag of the wing “{name}” change between 20 and 40 m/s?"]
    trust = (f"What is the drag coefficient of “{name}” at {v}, and how much should I trust it? Estimate the "
             "cost first.")
    if three and kind == "vehicle":
        return [f"How does yaw 0–10° change the drag of “{name}” at {v}? Estimate the cost first.", trust]
    if three:
        return [trust, (f"How does the drag of “{name}” change between 10 and 30 m/s (draft quality)? Estimate the "
                        "cost first.")]
    if kind == "airfoil":
        return [f"Find the angle of attack with the best L/D of “{name}” at {v} with the polar analysis.",
                f"At what angle does “{name}” stall at {v}, and how far can that angle be trusted?",
                f"How do the lift and drag of “{name}” change between 10 and 40 m/s (polar analysis)?"]
    return [f"Compare the drag of “{name}” at 20, 30 and 40 m/s (draft quality).",
            f"What are the drag coefficient and the Strouhal number of “{name}” at {v} (draft quality)?"]


def selection_example(design: Any, speed: float | None = None, sim: Any = None) -> str:
    """An example goal about the selected design, worded for its kind (empty without a design).

    A first-time user with a 3D car selected was offered only goals about a
    NACA 2412 and a 2D Ahmed body that the project did not contain. The first of
    :func:`design_questions`.
    """
    q = design_questions(design, speed, sim)
    return q[0] if q else ""


def _design_runs(project: Any, design_id: str) -> tuple[float | None, Any]:
    """(speed of the newest run, newest run) of a design in ``project``."""
    sims = [s for s in (getattr(project, "simulations", {}) or {}).values()
            if getattr(s, "design_id", None) == design_id]
    if not sims:
        return None, None
    latest = sims[-1]
    return getattr(getattr(latest, "conditions", None), "speed", None), latest


def project_examples(project: Any, limit: int = 3, exclude: tuple[str, ...] = ()) -> list[str]:
    """Example goals drawn from the open project's own designs and physics (at most ``limit``).

    In a heated enclosure with fans three of the four examples were aerodynamic
    (NACA 4412, Ahmed slants, wing aspect ratio; critic round 3.4). Order: the
    main question of each kind of placed physics (:func:`physics_examples`), the
    other physics questions of the runs (heat beside a fan, the grid error of Nu),
    then each design's questions, newest design first and one per design per
    round. :data:`EXAMPLES` only for a project without designs; fewer than
    ``limit`` when the project offers fewer (the empty slots are hidden).
    """
    out: list[str] = []
    skip = set(exclude)

    def add(text: str) -> bool:
        if text and text not in skip and text not in out:
            out.append(text)
        return len(out) >= limit

    if project is None:
        return []
    for text in physics_examples(project, limit + len(skip)):
        if add(text):
            return out
    designs = getattr(project, "designs", {}) or {}
    kinds = set()
    for sim in reversed(list((getattr(project, "simulations", {}) or {}).values())):
        for kind, text in physics_questions(sim, designs.get(getattr(sim, "design_id", ""))):
            if kind not in kinds and text not in out:
                kinds.add(kind)
                if add(text):
                    return out
    per_design = []
    for did, d in reversed(list(designs.items())):
        speed, latest = _design_runs(project, did)
        per_design.append(design_questions(d, speed, latest))
    for rnd in range(max((len(q) for q in per_design), default=0)):
        for q in per_design:
            if rnd < len(q) and add(q[rnd]):
                return out
    if not designs:
        for text in EXAMPLES:
            if add(text):
                return out
    return out

_STATUS_GLYPH = {"ok": ("check-circle", "good"), "error": ("error", "bad"), "invalid": ("warning", "warn"),
                 "declined": ("minus", "muted"), "cancelled": ("stop", "warn")}
FIGURE_MAX_W = 560


def _fmt_secs(s: float) -> str:
    from ...ai.experiment import fmt_seconds
    return fmt_seconds(s)


def _deg(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    return (f"{x:g}" if x >= 0 else f"−{-x:g}") + "°"


def tool_subtitle(name: str, args: dict | None, project: Any = None) -> str:
    """One line saying what a call is about ("NACA 2412 · 20 m/s · α −4…14°")."""
    a = args if isinstance(args, dict) else {}
    designs = getattr(project, "designs", {}) or {}
    sims = getattr(project, "simulations", {}) or {}

    def dname(ref: Any) -> str:
        d = designs.get(str(ref)) if ref is not None else None
        return d.name if d is not None else str(ref or "")

    parts: list[str] = []
    if name in ("run_polar", "run_wing", "run_cfd", "run_sweep", "estimate_cost", "describe_design", "derive_design"):
        parts.append(dname(a.get("design_id", a.get("base_design_id"))))
        if a.get("speed_m_s") is not None:
            parts.append(f"{float(a['speed_m_s']):g} m/s")
        elif a.get("reynolds") is not None:
            parts.append(f"Re {float(a['reynolds']):.3g}")
        if a.get("alpha_start") is not None or a.get("alpha_end") is not None:
            parts.append(f"α {_deg(a.get('alpha_start', '…'))}…{_deg(a.get('alpha_end', '…'))}")
        if name == "run_cfd" and a.get("alpha_deg") is not None:
            parts.append(f"α {_deg(a['alpha_deg'])}")
        if name in ("run_cfd", "run_sweep") and a.get("quality"):
            parts.append(str(a["quality"]))
        if name == "run_sweep" and a.get("variable"):
            vals = ", ".join(f"{float(v):g}" for v in (a.get("values") or [])[:8])
            parts.append(f"{a['variable']} = {vals}")
        if name == "derive_design" and a.get("name"):
            parts.append(f"→ {a['name']}")
    elif name in ("create_naca_airfoil",):
        parts.append(f"NACA {a.get('code', '')}")
    elif name == "create_wing":
        parts.append(f"span {a.get('span_m', '')} m, chords {a.get('root_chord_m', '')}/"
                     f"{a.get('tip_chord_m', a.get('root_chord_m', ''))} m")
    elif name == "create_template":
        parts.append(str(a.get("template", "")))
    elif name in ("get_results", "compare_results", "make_plot", "write_report", "compare_reference"):
        ids = list(a.get("simulation_ids") or [])
        names = [sims[i].name if i in sims else str(i) for i in ids[:3]]
        if name == "make_plot" and a.get("kind"):
            parts.append(str(a["kind"]))
        parts.append(", ".join(names) + (f" (+{len(ids) - 3})" if len(ids) > 3 else ""))
    elif name == "run_python":
        parts.append(str(a.get("purpose", "")))
    elif name == "delete_item":
        parts.append(f"{a.get('kind', '')} {a.get('id', '')}")
    elif name in ("describe_simulation", "run_simulation", "suggest_settings", "run_grid_study"):
        sid = str(a.get("simulation_id", ""))
        parts.append(sims[sid].name if sid in sims else sid)
        if name == "suggest_settings":
            parts.append(", ".join(str(c.get("setting", "")) for c in (a.get("changes") or [])[:4]
                                   if isinstance(c, dict)))
    elif name == "create_case":
        case = str(a.get("case", ""))
        bits = {"cavity": f"heated cavity, Ra {a.get('rayleigh', 1e4):g}",
                "heated_cylinder": f"heated cylinder, Re {a.get('reynolds', 40):g}",
                "fan_room": f"fan in a room, {a.get('thrust_n', 1):g} N"}
        parts.append(bits.get(case, case))
    elif name == "record_plan":
        parts.append(str(a.get("objective", "")))
    elif name == "record_conclusion":
        parts.append(str(a.get("summary", "")))
    return " · ".join(p for p in parts if p)


class _Bridge(QObject):
    """Worker -> GUI: every agent event arrives here, queued onto the GUI thread."""
    event = Signal(str, object)


_TABLE_TAG = re.compile(r"<table\b[^>]*>")
_LIST_INDENT = re.compile(r"-qt-list-indent: (\d+);")


def markdown_html(text: str) -> str:
    """An answer's markdown as rich text whose tables and lists suit a narrow dock in either theme.

    Qt's own markdown rendering draws tables with a light default border that
    reads as a white grid on the dark theme, and indents every list level by
    40 px, a sixth of the dock. The text goes through :class:`QTextDocument`
    (the same CommonMark/GFM reader); table tags get 1 px collapsed borders in
    the ``border`` token with 4 px cell padding, like the app's data tables,
    and each list level is indented 20 px. Called again when the theme changes.
    """
    doc = QTextDocument()
    doc.setMarkdown(text, QTextDocument.MarkdownFeature.MarkdownDialectGitHub)
    body = doc.toHtml()
    border = theme.tokens()["border"]
    body = _TABLE_TAG.sub(f'<table border="1" cellspacing="0" cellpadding="4" style="border-collapse: collapse; '
                          f'border-color: {border}; border-style: solid; margin-top: 4px; margin-bottom: 4px;">',
                          body)
    # a later margin-left wins over the 0 px one Qt writes first; the level's 40 px indent then nets 20 px
    return _LIST_INDENT.sub(lambda m: f"{m.group(0)} margin-left: -{20 * int(m.group(1))}px;", body)


# ---------------------------------------------------------------------------
# transcript blocks
# ---------------------------------------------------------------------------

def _goal_link(text: str) -> str:
    """An example goal as a link whose units never break across lines.

    Qt breaks a line after a slash, which left "25 m/" at the end of one line and
    "s?" alone on the next. A number and its unit are joined by a no-break space and
    the slash of a unit by word joiners (U+2060, zero width, never drawn), which
    Qt's line breaker honours; a CSS ``white-space: nowrap`` span is ignored inside
    a paragraph. The goal text itself (``property("goal")``) is unchanged.
    """
    out = html.escape(text)
    out = re.sub(r"(\d+(?:[.,]\d+)?)\s(m/s|m²|N/m|°)", "\\1\u00a0\\2", out)
    out = out.replace("m/s", "m\u2060/\u2060s").replace("N/m", "N\u2060/\u2060m")
    return f'<a href="goal">{out}</a>'          # ink and underlined, as every link (theme v4)


class _WrapLabel(QLabel):
    """Plain wrapped text whose height always follows its width (the empty state's introduction).

    The introduction used to be a ``label(..., wrap=True)``: a Preferred-width
    label is sized from its size hint, measured at Qt's preferred wrap width, so
    in a dock narrower than that it got fewer lines' height than it needed and,
    vertically centred, lost its first line under the heading and its last line
    ("... logged in the project's experiments folder") at the bottom (critic
    round 3). An Ignored width makes the layout give it the column's width and
    ask ``heightForWidth`` for the height; top alignment keeps the first line in
    place whatever happens.
    """

    def __init__(self, text: str, role: str | None = None) -> None:
        super().__init__(text)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        sp = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        if role:
            self.setProperty("role", role)


class _LinkLabel(_WrapLabel):
    """A wrapped rich-text line with links (the example goals, the Script hint).

    These were plain QLabels with an Ignored width but no ``heightForWidth``:
    the transcript canvas is sized from ``heightForWidth`` (see
    :class:`_Transcript`), so each one got the height of its size hint, measured
    at Qt's wide preferred wrap width, and showed one full line plus the top
    half of the next (restyle critic round 2, Compare page at 1280 x 800).
    Deriving from :class:`_WrapLabel` gives them its height-for-width policy and
    top alignment; a Minimum vertical policy stops a short panel squeezing them
    (the empty state scrolls instead).
    """

    def __init__(self, text: str, role: str | None = None) -> None:
        super().__init__("", role)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(text)
        sp = self.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.setSizePolicy(sp)


class _TextBlock(QLabel):
    """Markdown text that wraps, selects and opens links (an answer, a user goal).

    Markdown is rendered through :func:`markdown_html` (themed tables);
    :meth:`restyle` re-renders it after a theme change.
    """

    def __init__(self, text: str = "", *, markdown: bool = True, role: str | None = None) -> None:
        super().__init__()
        self._markdown = markdown
        self.setTextFormat(Qt.TextFormat.RichText if markdown else Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.setOpenExternalLinks(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        if role:
            self.setProperty("role", role)
        self._text = ""
        self.set_text(text)

    def set_text(self, text: str) -> None:
        # answers arrive as plain Unicode already (the agent converts LaTeX math); converting here too keeps
        # anything else that reaches a block -- a replayed transcript, a test provider -- from showing "$\pm$"
        self._text = plain_math(text) if self._markdown else text
        self.setText(markdown_html(self._text) if self._markdown else self._text)

    def restyle(self) -> None:
        if self._markdown:
            self.setText(markdown_html(self._text))

    def append(self, delta: str) -> None:
        self.set_text(self._text + delta)

    def plain(self) -> str:
        return self._text

    def set_superseded(self) -> None:
        """Show an answer the loop sent back for correction in the muted colour (the next one replaces it)."""
        theme.set_prop(self, "role", "muted")
        self.setToolTip("Aerolume asked the model to correct this answer; the corrected answer follows.")


def run_row_text(name: str, summary: str) -> str:
    """A finished run's row text with the measured numbers first: "Nu 2.251 (steady), Q … · Heated cavity · Ra …".

    The run summary reads "<run>: <numbers>", and the dock (about 300 px at 1600
    px) elided it to "Heated cavity · Ra…", hiding the numbers the row exists for
    (critic round 3.3). The run is named by the row above it and in the tooltip;
    other tools keep their summary as written.
    """
    if not name.startswith("run_") or ": " not in summary:
        return summary
    what, numbers = summary.split(": ", 1)
    return f"{numbers} · {what}" if numbers.strip() else summary


class _ToolRow(QFrame):
    """One tool call: status glyph, title, what it is about, time / progress; folds out to details."""

    def __init__(self, call_id: str, name: str, args: Any, subtitle: str) -> None:
        super().__init__()
        self.call_id, self.name, self.args = call_id, name, args
        self.status = "running"
        self._t0 = time.monotonic()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2 * SPACING)
        self.spinner = Spinner(14)
        self.glyph = QLabel()
        self.glyph.setFixedSize(16, 16)
        self.glyph.hide()
        row.addWidget(self.spinner)
        row.addWidget(self.glyph)
        self.title = label(TOOL_TITLES.get(name, name), None)
        self.title.setToolTip(name)
        row.addWidget(self.title)
        self.subtitle = ElidedLabel(subtitle, "muted", mode=Qt.TextElideMode.ElideRight)
        self.subtitle.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.subtitle, 1)
        self.right = label("", "muted")
        self.right.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.right)
        self.toggle = IconButton("chevron-right", "Show the input and the result", size=20, icon_size=12)
        self.toggle.clicked.connect(self._toggle)
        row.addWidget(self.toggle)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(2)
        self.bar.hide()
        lay.addWidget(self.bar)
        self.extra = QWidget()                 # a plan or a result table shown under the row
        self.extra_lay = QVBoxLayout(self.extra)
        self.extra_lay.setContentsMargins(24, 2, 0, 2)
        self.extra.hide()
        lay.addWidget(self.extra)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setFont(theme.mono_font(pt=theme.FONT_PT["small"]))
        self.details.setFixedHeight(120)
        self.details.hide()
        self.details.setPlainText(self._args_text())
        lay.addWidget(self.details)
        self.message = ""

    def _args_text(self) -> str:
        try:
            body = json.dumps(self.args, indent=1, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            body = str(self.args)
        return f"{self.name}\n{body}"

    def _toggle(self) -> None:
        show = self.details.isHidden()
        self.details.setVisible(show)
        self.toggle.set_icon("chevron-down" if show else "chevron-right")

    def set_progress(self, fraction: float, message: str) -> None:
        if self.status != "running":
            return
        f = max(0.0, min(1.0, float(fraction)))
        self.bar.setValue(round(1000 * f))
        self.bar.show()
        self.right.setText(f"{round(100 * f)} %")
        if message:
            self.message = message
            self.subtitle.setToolTip(message)

    def tick(self) -> None:
        if self.status == "running" and self.bar.isHidden():
            secs = time.monotonic() - self._t0
            if secs >= 2:
                self.right.setText(_fmt_secs(secs))

    def finish(self, status: str, summary: str, seconds: float, args: Any = None) -> None:
        self.status = status
        self.spinner.hide()
        icon, colour = _STATUS_GLYPH.get(status, _STATUS_GLYPH["ok"])
        self.glyph.setPixmap(theme.render_icon(icon, 14, colour, self.devicePixelRatioF()))
        self.glyph.show()
        self.bar.hide()
        self.right.setText(_fmt_secs(seconds) if seconds >= 0.1 else "")
        if summary:
            self.subtitle.setText(run_row_text(self.name, summary) if status == "ok" else summary)
            self.subtitle.setToolTip(summary)
        if status in ("error", "invalid", "declined", "cancelled"):
            theme.set_prop(self.subtitle, "role", "status-bad" if status in ("error", "invalid") else "status-warn")
        if args is not None:
            self.args = args
        self.details.setPlainText(self._args_text() + (f"\n\n{status}: {summary}" if summary else ""))

    def add_extra(self, w: QWidget) -> None:
        self.extra_lay.addWidget(w)
        self.extra.show()


_ESTIMATE_SUFFIX = re.compile(r":\s*about\s+[^:]*$")


def card_title(title: str, items: list[tuple[str, str]]) -> str:
    """An approval's title for its card: without the ": about 25 s" when a row already gives the estimate.

    The title doubles as the log's one-line summary, so it carries the time; on
    the card the "Estimated time" row says it, and the repeated suffix wrapped a
    lone "s" onto a second line of the heading in a 330 px dock. "cost unknown"
    stays: it is why the card asks.
    """
    if any(k == "Estimated time" for k, _v in items):
        return _ESTIMATE_SUFFIX.sub("", title).rstrip() or title
    return title


#: characters that change how code reads without being visible: bidirectional overrides and isolates,
#: zero-width marks, the BOM, and other format / control characters (tab and newline excepted)
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff\u00ad\x00-\x08\x0b-\x1f\x7f]")
#: a run of spaces this long inside a line is shown as a marker (it can push code out of sight)
LONG_SPACES = 8
_SPACE_RUN = re.compile(f" {{{LONG_SPACES},}}")


def reveal_code(code: str) -> tuple[str, list[str]]:
    """The script as the approval card shows it, and what it had to reveal.

    Why: the card is the user's only look at code that will run with their rights. A line padded
    with 160 spaces put "; import shutil" past the right edge of a 430 px card, and a U+202E
    override can make text read backwards (critic round 5). Invisible characters become a visible
    "«U+202E»"; a run of spaces inside a line (after its indentation) becomes "«160 spaces»"; the
    card wraps lines, so nothing sits outside the box. The code that runs is unchanged."""
    notes: list[str] = []
    hidden = _INVISIBLE.findall(code)
    shown = _INVISIBLE.sub(lambda m: f"«U+{ord(m.group(0)):04X}»", code)
    padded = 0
    lines = []
    for line in shown.split("\n"):
        body = line.lstrip(" ")
        indent = line[:len(line) - len(body)]

        def mark(m: re.Match) -> str:
            nonlocal padded
            padded += 1
            return f" «{len(m.group(0))} spaces» "
        lines.append(indent + _SPACE_RUN.sub(mark, body))
    shown = "\n".join(lines)
    if hidden:
        notes.append(f"{len(hidden)} invisible character{'s' if len(hidden) != 1 else ''} (shown as «U+…»)")
    if padded:
        notes.append(f"{padded} long run{'s' if padded != 1 else ''} of spaces inside a line (shown as «n spaces»)")
    return shown, notes


class _ApprovalCard(QFrame):
    """The one framed block: a question the assistant cannot go on without."""

    decided = Signal(str, bool)

    def __init__(self, call_id: str, approval: Any) -> None:
        super().__init__()
        self.call_id = call_id
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3 * SPACING, 2 * SPACING, 3 * SPACING, 2 * SPACING)
        lay.setSpacing(SPACING)
        head = QHBoxLayout()
        head.setSpacing(2 * SPACING)
        icon = QLabel()
        kind = getattr(approval, "kind", "run")
        icon.setPixmap(theme.render_icon({"code": "code", "delete": "trash"}.get(kind, "clock"), 16,
                                         "bad" if kind == "delete" else "text2", self.devicePixelRatioF()))
        head.addWidget(icon)
        # exactly what will run, row by row (design, analysis, speed, angles, quality, grid, Reynolds numbers,
        # device, physics, estimated time): the card used to carry only "Wing analysis of Tapered wing: about
        # 25 s", and the user approved without seeing the speed or the angle range
        items = [(str(k), str(v)) for k, v in (getattr(approval, "items", None) or [])]
        self.title = label(card_title(str(getattr(approval, "title", "Approve?")), items), "h3", wrap=True)
        self.title.setToolTip(str(getattr(approval, "title", "")))
        head.addWidget(self.title, 1)
        lay.addLayout(head)
        self.items = None
        self.what = None
        if items:
            self.items = KeyValueList(items)
            for _k, v in self.items._rows:
                v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                v.setWordWrap(True)
            lay.addWidget(self.items)
        else:
            what = str(getattr(approval, "what", "") or "")
            if what:
                self.what = label(what, None, wrap=True)
                self.what.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                lay.addWidget(self.what)
        detail = str(getattr(approval, "detail", "") or "")
        if detail:
            lay.addWidget(label(detail, "secondary", wrap=True))
        code = getattr(approval, "code", None)
        self.code = None
        self.code_warning = None
        if code:
            shown, notes = reveal_code(str(code))
            if notes:
                self.code_warning = label("This script contains " + " and ".join(notes) + ". Read it to the end "
                                          "before running it.", "status-warn", wrap=True)
                lay.addWidget(self.code_warning)
            self.code = QPlainTextEdit(shown)
            self.code.setReadOnly(True)
            self.code.setFont(theme.mono_font())
            # wrapped: a long line must never run past the card's edge, where the user cannot see it
            self.code.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
            self.code.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            self.code.setAccessibleName("The script to run")
            lines = shown.count("\n") + 1 + sum(len(ln) // 48 for ln in shown.split("\n"))
            self.code.setFixedHeight(min(240, 20 + 17 * lines))
            lay.addWidget(self.code)
        row = QHBoxLayout()
        row.setSpacing(2 * SPACING)
        verb = {"code": "Run", "delete": "Delete"}.get(kind, "Run")
        self.approve = button(verb, "danger" if kind == "delete" else "primary")
        self.approve.clicked.connect(lambda: self._decide(True))
        self.decline = button("Decline", "ghost")
        self.decline.clicked.connect(lambda: self._decide(False))
        row.addWidget(self.approve)
        row.addWidget(self.decline)
        row.addStretch(1)
        self.outcome = label("", "secondary")
        self.outcome.hide()
        row.addWidget(self.outcome)
        lay.addLayout(row)

    def _decide(self, ok: bool) -> None:
        self.set_outcome(ok)
        self.decided.emit(self.call_id, ok)

    def set_outcome(self, ok: bool | None, text: str | None = None) -> None:
        self.approve.hide()
        self.decline.hide()
        self.outcome.setText(text or ("Approved" if ok else "Declined"))
        theme.set_prop(self.outcome, "role", "status-good" if ok else "status-warn")
        self.outcome.show()


def _fmt_value(v: Any, unit: str = "") -> str:
    """A setting's value for a suggestion row: numbers at 4 significant figures, on/off, "automatic"."""
    if v is None:
        return "automatic"
    if isinstance(v, bool):
        return "on" if v else "off"
    text = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
    if not unit:
        return text
    return text + unit if unit == "°" else f"{text} {unit}"


def _value_role(v: Any) -> str | None:
    """Mono for numbers only (docs/ui_guidelines.md: numbers in Atkinson Hyperlegible Mono).

    Words such as "draft", "standard" or "automatic" in the mono face read letter-spaced
    ("d r a f t") next to the proportional setting names.
    """
    return "mono" if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class _SuggestionCard(QFrame):
    """Proposed setting changes, old -> new, applied only on a click (Lumen's assistant rules).

    Every row was checked by the tool (unknown settings and invalid values are
    shown with the reason and are never applied). Apply checks the values again
    through :meth:`aerolume.api.Project.edit_simulation` (the API's own rules, so a
    value cannot slip past a limit that changed meanwhile) and records the change
    as one undo step: Undo takes every changed setting back at once. A
    simulation that already has a result is copied, not edited, so its result
    never sits under settings it was not computed with.
    """

    applied = Signal(str)            # the simulation id the change produced (the edited one or its copy)

    def __init__(self, data: dict, apply_fn, undo_fn) -> None:
        super().__init__()
        self.data = dict(data)
        self._apply_fn, self._undo_fn = apply_fn, undo_fn
        self.undo_token: Any = None
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3 * SPACING, 2 * SPACING, 3 * SPACING, 2 * SPACING)
        lay.setSpacing(SPACING)
        head = QHBoxLayout()
        head.setSpacing(2 * SPACING)
        icon = QLabel()
        icon.setPixmap(theme.render_icon("edit", 16, "text2", self.devicePixelRatioF()))
        head.addWidget(icon)
        name = str(self.data.get("simulation", ""))
        self.title = label(f"Suggested changes to “{name}”", "h3", wrap=True)
        head.addWidget(self.title, 1)
        lay.addLayout(head)
        reason = str(self.data.get("reason", "") or "")
        if reason:
            lay.addWidget(label(reason, "secondary", wrap=True))
        self.rows = [r for r in self.data.get("changes", []) if isinstance(r, dict)]
        grid = QGridLayout()
        grid.setHorizontalSpacing(3 * SPACING)
        grid.setVerticalSpacing(SPACING)
        grid.setColumnStretch(3, 1)
        i = 0
        for r in self.rows:
            unit = str(r.get("unit", "") or "")
            name_l = label(str(r.get("label", r.get("setting", ""))), "secondary")
            old_l = label(_fmt_value(r.get("old"), unit), _value_role(r.get("old")))
            arrow = label("→", "muted")
            new_l = label(_fmt_value(r.get("new"), unit), _value_role(r.get("new")))
            grid.addWidget(name_l, i, 0)
            grid.addWidget(old_l, i, 1)
            grid.addWidget(arrow, i, 2)
            grid.addWidget(new_l, i, 3)
            i += 1
            if not r.get("valid"):
                # struck out as rich text: a font set in code is replaced when the theme's stylesheet
                # polishes the label, so setStrikeOut() left refused rows looking like valid ones
                for w in (name_l, old_l, new_l):
                    theme.set_prop(w, "role", "muted")
                    w.setTextFormat(Qt.TextFormat.RichText)
                    w.setText(f"<s>{html.escape(w.text())}</s>")
                    w.setProperty("struck", True)
                    w.setToolTip("Not applied")
                # the reason on its own line under the row: a fifth column was one word wide in a 330 px dock
                why = label(f"Not applied: {r.get('problem', 'not valid')}", "small", wrap=True)
                theme.set_prop(why, "role", "status-warn")
                grid.addWidget(why, i, 0, 1, 4)
                i += 1
        lay.addLayout(grid)
        if self.data.get("has_result"):
            lay.addWidget(label("This simulation has a result: Apply makes an edited copy and keeps the original.",
                                "small", wrap=True))
        # the buttons and the outcome sit on separate rows: sharing one, a wrapped "Applied to “CFD · 20 m/s
        # · α 2° · Standard”" was squeezed into a narrow centred column between a stretch and Undo
        self.actions = QWidget()
        row = QHBoxLayout(self.actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2 * SPACING)
        n = sum(1 for r in self.rows if r.get("valid"))
        self.apply = button(f"Apply {n} change{'s' if n != 1 else ''}", "primary")
        self.apply.setEnabled(n > 0)
        self.apply.clicked.connect(self._on_apply)
        self.dismiss = button("Dismiss", "ghost")
        self.dismiss.clicked.connect(self._on_dismiss)
        row.addWidget(self.apply)
        row.addWidget(self.dismiss)
        row.addStretch(1)
        lay.addWidget(self.actions)
        self.result = QWidget()
        rr = QHBoxLayout(self.result)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.setSpacing(2 * SPACING)
        self.outcome = label("", "secondary", wrap=True)
        rr.addWidget(self.outcome, 1)
        self.undo = button("Undo", "link")
        self.undo.setToolTip("Take every change of this suggestion back (one undo step)")
        self.undo.clicked.connect(self._on_undo)
        self.undo.hide()
        rr.addWidget(self.undo, 0, Qt.AlignmentFlag.AlignTop)
        self.result.hide()
        lay.addWidget(self.result)

    def changes(self) -> dict[str, Any]:
        """The valid changes as API keywords ({"quality": "standard", ...})."""
        return {str(r["setting"]): r.get("new") for r in self.rows if r.get("valid")}

    def _set_outcome(self, text: str, role: str) -> None:
        self.outcome.setText(text)
        theme.set_prop(self.outcome, "role", role)
        self.result.show()

    def _on_apply(self) -> None:
        try:
            token, sim_name = self._apply_fn(str(self.data.get("simulation_id", "")), self.changes())
        except Exception as exc:  # noqa: BLE001 - a refused value is shown, never raised into Qt
            self._set_outcome(f"Not applied: {exc}", "status-bad")
            return
        self.undo_token = token
        self.actions.hide()
        self.undo.show()
        self._set_outcome(f"Applied to “{sim_name}”", "status-good")
        self.applied.emit(str(token.get("sim_id", "")) if isinstance(token, dict) else "")

    def _on_undo(self) -> None:
        if self.undo_token is None:
            return
        try:
            self._undo_fn(self.undo_token)
        except Exception as exc:  # noqa: BLE001
            self._set_outcome(f"Could not undo: {exc}", "status-bad")
            return
        self.undo_token = None
        self.undo.hide()
        self._set_outcome("Undone: the simulation is back as it was", "secondary")

    def _on_dismiss(self) -> None:
        self.actions.hide()
        self._set_outcome("Dismissed: nothing changed", "secondary")


_CLAIM_GLYPH = {"verified": ("check-circle", "good"), "derived": ("info", "text2"),
                "unsupported": ("warning", "warn"), "contradicted": ("error", "bad")}
_CLAIM_WORD = {"verified": "Verified", "derived": "Derived", "unsupported": "Not supported",
               "contradicted": "Contradicted"}
_MD_EMPHASIS = re.compile(r"\*\*|__|`")


def plain_statement(text: str) -> str:
    """A checked statement as plain text: the answer's markdown emphasis ("**26.45**") left out."""
    return " ".join(_MD_EMPHASIS.sub("", str(text)).split())


class _ClaimsBlock(QWidget):
    """Which statements of the final answer were checked, and how (the claims check, ai/claims.py).

    One row per statement with numbers or claimed actions: a glyph and the
    verdict, the statement, and the evidence (where each number came from, the
    formula of a derived one) or the problem (the value a misattributed number
    should have been). Folded away when everything is verified; open when
    something is not.
    """

    def __init__(self, claims: dict) -> None:
        super().__init__()
        self.claims = dict(claims or {})
        self.statements = [s for s in self.claims.get("statements", []) if isinstance(s, dict)
                           and s.get("status") != "plain"]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2 * SPACING)
        summary = str(self.claims.get("summary", "") or "")
        # the summary is a wrapped line of the body: in the (non-wrapping) header it set the transcript's
        # minimum width to 478 px in a 306 px dock, which cut every line off at the right edge
        self.summary = label(summary, "small", wrap=True) if summary else None
        if self.summary is not None:
            bl.addWidget(self.summary)
        self.rows: list[QWidget] = []
        for s in self.statements:
            status = str(s.get("status", ""))
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2 * SPACING)
            glyph = QLabel()
            glyph.setFixedSize(16, 16)
            icon, colour = _CLAIM_GLYPH.get(status, ("info", "muted"))
            glyph.setPixmap(theme.render_icon(icon, 14, colour, self.devicePixelRatioF()))
            glyph.setToolTip(_CLAIM_WORD.get(status, status))
            rl.addWidget(glyph, 0, Qt.AlignmentFlag.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(1)
            text = label(plain_statement(s.get("text", "")), None, wrap=True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            col.addWidget(text)
            detail = "; ".join(s.get("problems") or []) if status in ("unsupported", "contradicted") \
                else "; ".join(s.get("evidence") or [])
            note = label(f"{_CLAIM_WORD.get(status, status)}" + (f": {detail}" if detail else ""), "small",
                         wrap=True)
            note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            if status in ("unsupported", "contradicted"):
                theme.set_prop(note, "role", "status-warn" if status == "unsupported" else "status-bad")
            col.addWidget(note)
            rl.addLayout(col, 1)
            bl.addWidget(row)
            self.rows.append(row)
        problems = sum(1 for s in self.statements if s.get("status") in ("unsupported", "contradicted"))
        head = f"Checked statements · {problems} not verified" if problems else "Checked statements · all verified"
        self.section = CollapsibleSection(head, body, expanded=problems > 0)
        self.section.header.setToolTip("Every statement with numbers or actions, checked against the tool calls "
                                       "and their results")
        lay.addWidget(self.section)

    def has_problems(self) -> bool:
        return any(s.get("status") in ("unsupported", "contradicted") for s in self.statements)


class _ConsentCard(QFrame):
    """Permission to send the experiment's data to a model off this computer (asked before anything is sent)."""

    decided = Signal(bool, bool)          # allowed, remember

    def __init__(self, destination: str) -> None:
        super().__init__()
        from PySide6.QtWidgets import QCheckBox
        self.setProperty("card", "true")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3 * SPACING, 2 * SPACING, 3 * SPACING, 2 * SPACING)
        lay.setSpacing(SPACING)
        head = QHBoxLayout()
        head.setSpacing(2 * SPACING)
        icon = QLabel()
        icon.setPixmap(theme.render_icon("external", 16, "text2", self.devicePixelRatioF()))
        head.addWidget(icon)
        head.addWidget(label(f"Send this experiment to {destination}?", "h3", wrap=True), 1)
        lay.addLayout(head)
        self.facts = KeyValueList([
            ("Sent", ("the goal, the selected design's name, the project's design and simulation names, and the "
                      "tools' results (numbers, warnings, notes)")),
            ("Not sent", "meshes, flow fields, figures and the project's files"),
            ("Kept", "the experiment log stays in the project folder on this computer")])
        for _k, v in self.facts._rows:
            v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            v.setWordWrap(True)
        lay.addWidget(self.facts)
        self.remember = QCheckBox("Do not ask again for this destination")
        lay.addWidget(self.remember)
        row = QHBoxLayout()
        row.setSpacing(2 * SPACING)
        self.allow = button("Send", "primary")
        self.allow.clicked.connect(lambda: self._decide(True))
        self.cancel = button("Cancel", "ghost")
        self.cancel.clicked.connect(lambda: self._decide(False))
        row.addWidget(self.allow)
        row.addWidget(self.cancel)
        row.addStretch(1)
        self.outcome = label("", "secondary")
        self.outcome.hide()
        row.addWidget(self.outcome)
        lay.addLayout(row)

    def _decide(self, ok: bool) -> None:
        self.allow.hide()
        self.cancel.hide()
        self.remember.setEnabled(False)
        self.outcome.setText("Sent" if ok else "Not sent")
        theme.set_prop(self.outcome, "role", "status-good" if ok else "secondary")
        self.outcome.show()
        self.decided.emit(ok, bool(self.remember.isChecked()))


class _FigureBlock(QWidget):
    """An inline figure thumbnail; a click opens it full size."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.pixmap = QPixmap(path)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, SPACING, 0, SPACING)
        lay.setSpacing(2)
        self.image = QLabel()
        self.image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.image.setToolTip(f"Open the figure full size\n{path}")
        self.image.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.image.mousePressEvent = lambda _e: self.open_full()     # type: ignore[method-assign]
        lay.addWidget(self.image)
        cap = label(Path(path).name, "small")
        lay.addWidget(cap)
        self.fit(FIGURE_MAX_W)

    def fit(self, width: int) -> None:
        if self.pixmap.isNull():
            self.image.setText("Figure not readable")
            return
        w = max(120, min(int(width), FIGURE_MAX_W, self.pixmap.width()))
        dpr = self.devicePixelRatioF() or 1.0
        pm = self.pixmap.scaledToWidth(round(w * dpr), Qt.TransformationMode.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self.image.setPixmap(pm)
        self.image.setFixedSize(QSize(w, round(pm.height() / dpr)))

    def open_full(self) -> None:
        dlg = QDialog(self.window())
        dlg.setWindowTitle(Path(self.path).name)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(SPACING * 2, SPACING * 2, SPACING * 2, SPACING * 2)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        screen = self.screen().availableGeometry() if self.screen() is not None else None
        pm = self.pixmap
        if screen is not None and pm.width() > screen.width() * 0.85:
            pm = pm.scaledToWidth(int(screen.width() * 0.85), Qt.TransformationMode.SmoothTransformation)
        img.setPixmap(pm)
        sc.setWidget(img)
        lay.addWidget(sc, 1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        open_btn = box.addButton("Open file", QDialogButtonBox.ButtonRole.ActionRole)
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.path)))
        box.rejected.connect(dlg.reject)
        lay.addWidget(box)
        dlg.resize(min(pm.width() + 40, 1400), min(pm.height() + 90, 1000))
        dlg.exec()


class _Turn(QWidget):
    """One exchange: the user's request and everything the assistant did for it."""

    def __init__(self, user_text: str, model: str) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING)
        lay.addWidget(label("You", "h3"))
        self.user = _TextBlock(user_text, markdown=False)
        lay.addWidget(self.user)
        lay.addSpacing(SPACING)
        head = QHBoxLayout()
        head.setSpacing(2 * SPACING)
        head.addWidget(label("Assistant", "h3"))
        self.model = label(model, "small")
        head.addWidget(self.model)
        head.addStretch(1)
        lay.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(SPACING)
        lay.addLayout(self.body)
        self.text: _TextBlock | None = None
        self.answers: list[_TextBlock] = []          # every answer text block, in order
        self.reasoning: _TextBlock | None = None
        self.rows: dict[str, _ToolRow] = {}
        self.cards: dict[str, _ApprovalCard] = {}
        self.figures: list[_FigureBlock] = []
        self.suggestions: list[_SuggestionCard] = []
        self.claims: _ClaimsBlock | None = None

    def add(self, w: QWidget) -> QWidget:
        self.body.addWidget(w)
        return w

    def append_text(self, delta: str) -> None:
        if self.text is None:
            self.text = _TextBlock("")
            self.answers.append(self.text)
            self.add(self.text)
        self.text.append(delta)

    def drop_repeated_answers(self) -> int:
        """Hide earlier answers that the final one repeats word for word; returns how many.

        After a follow-up request ("record the conclusion, then give the final
        answer again") a model repeats its answer unchanged, and the transcript
        showed the same paragraphs twice around one Record conclusion row.
        """
        if len(self.answers) < 2:
            return 0

        def norm(b: _TextBlock) -> str:
            return " ".join(b.plain().split())
        last = norm(self.answers[-1])
        n = 0
        for b in self.answers[:-1]:
            if last and norm(b) == last and not b.property("dropped"):
                b.setProperty("dropped", True)
                b.hide()
                n += 1
        return n

    def append_reasoning(self, delta: str) -> None:
        if self.reasoning is None:
            self.reasoning = _TextBlock("", markdown=False, role="muted")
            sec = CollapsibleSection("Reasoning", self.reasoning, expanded=False)
            sec.header.setToolTip("The model's summary of its reasoning before this step")
            self.add(sec)
        self.reasoning.append(delta)

    def close_segment(self) -> None:
        """Text after a tool call starts a new block below it."""
        self.text = None
        self.reasoning = None


# ---------------------------------------------------------------------------
# the goal box
# ---------------------------------------------------------------------------

class _Transcript(QWidget):
    """The scrolling canvas of the transcript: as wide as the dock, as tall as its blocks at that width.

    A resizable QScrollArea grows its widget to at least the widget's minimum
    size hint, and the minimum height of wrapped text is measured at the
    minimum width, where every paragraph wraps into more lines. With a summary
    in a non-wrapping header (478 px) and an end-of-request row 2 px wider than
    the viewport, the canvas came out wider than the dock (every line cut off at
    the right) and 250 px taller than its content (a gap under the last block).
    Without a minimum, the scroll area sizes it from ``heightForWidth`` at the
    viewport's width instead.
    """

    def minimumSizeHint(self) -> QSize:
        return QSize(0, 0)


class _GoalEdit(QPlainTextEdit):
    """The goal box: Ctrl+Enter submits, Enter starts a new line."""

    submitted = Signal()

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(e)


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

class AssistantPanel(QWidget):
    """The Assistant dock content (see the module docstring).

    ``AssistantPanel(state, jobs=None, parent=None)``. Signals:
    ``experimentsChanged()`` (an experiment log was created or updated: the
    Project tree lists it), ``preferencesRequested()`` (the gear button: the
    window opens Preferences > Assistant), ``experimentOpened(str)`` (a log's
    folder to show), ``runningChanged(bool)``, ``scriptRequested()`` (the
    empty state's link to the Script panel). ``provider_factory(config)`` may
    be replaced (tests use fakes); it defaults to
    :func:`aerolume.ai.providers.make_provider`.
    """

    experimentsChanged = Signal()
    preferencesRequested = Signal()
    experimentOpened = Signal(str)
    runningChanged = Signal(bool)
    scriptRequested = Signal()

    def __init__(self, state, jobs=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AssistantPanel")
        self.state, self.jobs = state, jobs
        self.agent = None
        self.provider = None
        self.host = None
        self.provider_factory = None
        self._thread: threading.Thread | None = None
        self._turn: _Turn | None = None
        self._approval_lock = threading.Lock()
        self._approvals: dict[str, dict] = {}
        self._engine = None
        self._started = 0.0
        self._stick = True
        self._provider_ready: bool | None = None
        self._checked = False
        self._provider_gen = 0
        self._consented: set[str] = set()          # destinations allowed in this window session
        self._pending_goal = ""
        self._consent_card: _ConsentCard | None = None
        self._bridge = _Bridge()
        self._bridge.event.connect(self._on_event, Qt.ConnectionType.QueuedConnection)
        self._build()
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)
        state.projectChanged.connect(self._project_changed)
        for sig in ("currentDesignChanged", "designsChanged", "simulationsChanged"):
            s = getattr(state, sig, None)
            if s is not None:
                s.connect(lambda *_a: self._sync_selection_example())
        theme.changed.connect(self._on_theme)
        self._project_changed()

    # -- layout ---------------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        head = QFrame()
        head.setProperty("chrome", "toolbar")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(2 * SPACING, SPACING, SPACING, SPACING)
        hl.setSpacing(SPACING)
        self.model_label = ElidedLabel("", "secondary", mode=Qt.TextElideMode.ElideRight)
        self.model_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        hl.addWidget(self.model_label, 1)
        self.new_button = button("New experiment", "ghost", "new")
        self.new_button.setToolTip("Start a new experiment (a new conversation and a new log)")
        self.new_button.clicked.connect(self.new_experiment)
        hl.addWidget(self.new_button)
        self.prefs_button = IconButton("settings", "Assistant settings: model, key, approvals")
        self.prefs_button.clicked.connect(self.preferencesRequested.emit)
        hl.addWidget(self.prefs_button)
        root.addWidget(head)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.canvas = _Transcript()
        self.canvas.setObjectName("AssistantTranscript")
        self.flow = QVBoxLayout(self.canvas)
        self.flow.setSizeConstraint(QVBoxLayout.SizeConstraint.SetNoConstraint)      # see _Transcript
        self.flow.setContentsMargins(3 * SPACING, 3 * SPACING, 3 * SPACING, 3 * SPACING)
        self.flow.setSpacing(3 * SPACING)
        self.flow.addStretch(1)
        self.scroll.setWidget(self.canvas)
        bar = self.scroll.verticalScrollBar()
        bar.valueChanged.connect(lambda v: setattr(self, "_stick", v >= bar.maximum() - 8))
        bar.rangeChanged.connect(self._range_changed)
        root.addWidget(self.scroll, 1)

        self.empty = self._build_empty()
        self.flow.insertWidget(0, self.empty)

        root.addWidget(Divider())
        # The footer is a fixed block and the transcript is the only part that shrinks: sharing the
        # window's height with the Script panel used to cut the hint's second line and half the Run
        # button off at the bottom edge (restyle critic round 2). See minimumSizeHint.
        foot = QWidget()
        foot.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._head, self._foot = head, foot
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(3 * SPACING, 2 * SPACING, 3 * SPACING, 3 * SPACING)
        fl.setSpacing(2 * SPACING)
        srow = QHBoxLayout()
        srow.setSpacing(2 * SPACING)
        self.status_spinner = Spinner(12)
        srow.addWidget(self.status_spinner)
        self.status_text = ElidedLabel("", "muted", mode=Qt.TextElideMode.ElideRight)
        self.status_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        srow.addWidget(self.status_text, 1)
        self.elapsed = label("", "muted")
        srow.addWidget(self.elapsed)
        self.status_row = QWidget()
        self.status_row.setLayout(srow)
        srow.setContentsMargins(0, 0, 0, 0)
        self.status_row.hide()
        fl.addWidget(self.status_row)
        self.goal = _GoalEdit()
        self.goal.setPlaceholderText("Describe an experiment: what to find, for which design, at which conditions")
        self.goal.setFixedHeight(66)
        self.goal.setTabChangesFocus(True)
        self.goal.submitted.connect(self.submit)
        fl.addWidget(self.goal)
        brow = QHBoxLayout()
        brow.setSpacing(2 * SPACING)
        self.hint = _WrapLabel("Ctrl+Enter runs · runs above the approval threshold, code and deletions ask first",
                               "small")
        sp = self.hint.sizePolicy()
        sp.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.hint.setSizePolicy(sp)
        brow.addWidget(self.hint, 1)
        self.stop_button = button("Stop", "danger", "stop")
        self.stop_button.setToolTip("Stop the assistant and its running simulations (they keep partial results)")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.hide()
        brow.addWidget(self.stop_button)
        self.run_button = button("Run", "primary", "run")
        self.run_button.setToolTip("Send the goal to the assistant (Ctrl+Enter)")
        self.run_button.clicked.connect(self.submit)
        brow.addWidget(self.run_button)
        fl.addLayout(brow)
        root.addWidget(foot)
        self.scroll.setMinimumHeight(0)
        sc = QShortcut(QKeySequence("Ctrl+Return"), self.goal)
        sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc.activated.connect(self.submit)
        self.setMinimumWidth(300)

    def _build_empty(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(SPACING)
        lay.addWidget(label("Experiment assistant", "h2"))
        self.intro = _WrapLabel("Plans an experiment from a goal, runs the polar, wing and CFD solvers on this project "
                                "and reports the numbers with their uncertainty and validity limits. Each step is "
                                "logged in the project's experiments folder.", "secondary")
        lay.addWidget(self.intro)
        self.setup_line = MessageLine("", "warning")
        self.setup_line.hide()
        lay.addWidget(self.setup_line)
        self.setup_button = button("Assistant settings…", None, "settings")
        self.setup_button.clicked.connect(self.preferencesRequested.emit)
        self.setup_button.hide()
        lay.addWidget(self.setup_button, 0, Qt.AlignmentFlag.AlignLeft)
        lay.addSpacing(2 * SPACING)
        lay.addWidget(label("Examples", "h3"))
        # Wrapping link labels, not link buttons: a QPushButton cannot wrap, and a long goal was cut off at
        # the dock's edge. Clicking one puts the goal in the box (it does not run it).
        def link(text: str) -> QLabel:
            ex = _LinkLabel(_goal_link(text))
            ex.setToolTip("Put this goal in the box below")
            ex.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse
                                       | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
            return ex

        # first: a goal about the design selected in the window (kept current by _sync_selection_example)
        self.selection_example = link("")
        self.selection_example.linkActivated.connect(
            lambda _href: self.set_goal(self.selection_example.property("goal") or ""))
        self.selection_example.hide()
        lay.addWidget(self.selection_example)
        # three slots, filled from the project's own designs and physics (_sync_examples); hidden when it has fewer
        self.example_buttons: list[QLabel] = []
        for text in EXAMPLES:
            ex = link(text)
            ex.setProperty("goal", text)
            ex.linkActivated.connect(lambda _href, b=ex: self.set_goal(b.property("goal") or ""))
            lay.addWidget(ex)
            self.example_buttons.append(ex)
        lay.addSpacing(2 * SPACING)
        self.script_hint = _LinkLabel('The same studies can be written in Python: <a href="script">Script panel</a> '
                                      '(Alt+4), Help ▸ Scripting guide.', "muted")
        self.script_hint.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse
                                                 | Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        self.script_hint.linkActivated.connect(lambda _href: self.scriptRequested.emit())
        lay.addWidget(self.script_hint)
        return w

    def sizeHint(self) -> QSize:
        return QSize(430, 700)

    def hasHeightForWidth(self) -> bool:
        """No: the dock decides the panel's height, and the transcript absorbs the difference.

        The footer's wrapped hint has height-for-width, which made the whole panel
        report it; the dock's resizable scroll area then sized the panel from
        ``heightForWidth`` (header + the transcript's 700 px size hint + footer,
        540 px measured) inside a 307 px viewport, and the Run row ended up below
        the visible part. The panel's own layout still lays the hint out by
        height-for-width.
        """
        return False

    #: transcript height kept when the panel is at its minimum (about three lines of text)
    MIN_TRANSCRIPT = 72

    def minimumSizeHint(self) -> QSize:
        """Header + footer (at the current width) + a few transcript lines: the dock cannot squeeze the Run row.

        The dock's scroll area reports this as the panel's minimum, so the
        window's dock splitter stops here and the transcript scrolls instead of
        the footer being clipped. The footer's height is asked at the panel's
        width because its hint wraps (two lines in a narrow dock).
        """
        base = super().minimumSizeHint()
        width = max(self.width(), self.minimumWidth())
        foot = self._foot.heightForWidth(width) if self._foot.hasHeightForWidth() else -1
        if foot <= 0:
            foot = self._foot.sizeHint().height()
        height = self._head.sizeHint().height() + 1 + self.MIN_TRANSCRIPT + foot
        return QSize(base.width(), max(base.height(), height))

    # -- small helpers ----------------------------------------------------------------------
    def _range_changed(self, _lo: int, hi: int) -> None:
        # the transcript follows its newest block; the empty state is read from the top (it used to scroll to the
        # bottom when the selection example appeared, hiding the heading and the introduction's first lines)
        if self._stick and not self.empty.isVisibleTo(self.canvas):
            self.scroll.verticalScrollBar().setValue(hi)

    def _add_block(self, w: QWidget) -> QWidget:
        self.flow.insertWidget(self.flow.count() - 1, w)
        return w

    def _on_theme(self, _name: str) -> None:
        for row in self.findChildren(_ToolRow):
            if row.status != "running":
                icon, colour = _STATUS_GLYPH.get(row.status, _STATUS_GLYPH["ok"])
                row.glyph.setPixmap(theme.render_icon(icon, 14, colour, row.devicePixelRatioF()))
        for block in self.findChildren(_TextBlock):
            block.restyle()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if e.oldSize().width() != e.size().width():
            self.updateGeometry()       # the footer's wrapped height (minimumSizeHint) follows the width
        width = self.scroll.viewport().width() - 6 * SPACING
        for f in self.findChildren(_FigureBlock):
            f.fit(width)

    def set_goal(self, text: str) -> None:
        self.goal.setPlainText(text)
        self.goal.setFocus()
        cur = self.goal.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        self.goal.setTextCursor(cur)

    def focus_input(self) -> None:
        self.goal.setFocus()

    def is_running(self) -> bool:
        """Whether a request is in progress: from :meth:`submit` until :meth:`_done` has shown its outcome.

        Why not ``thread.is_alive()``: the worker's last act is to emit "done", which reaches
        this (GUI) thread through a queued signal, so the thread can be finished while its
        outcome still waits in the event queue. In that gap the panel still showed Stop and
        "Waiting for the model", yet reported itself idle, so :meth:`submit` accepted a new
        goal and the old run's late "done" then ended the new turn. ``_done`` clears
        ``_thread`` when it handles the event, and :meth:`_work` emits "done" on every exit,
        so a run can never stay busy for ever.
        """
        return self._thread is not None

    def transcript_text(self) -> str:
        """Everything in the transcript as plain text (tests, copy)."""
        out = []
        for w in self.canvas.findChildren(QLabel):
            if w.isVisibleTo(self.canvas) and w.text():
                if isinstance(w, _TextBlock):
                    out.append(w.plain())
                else:
                    out.append(w.full_text() if isinstance(w, ElidedLabel) else w.text())
        return "\n".join(out)

    # -- provider -------------------------------------------------------------------------------
    def _config(self):
        from ...ai.config import load_config
        return load_config()

    def _make_provider(self, cfg):
        if self.provider_factory is not None:
            return self.provider_factory(cfg)
        from ...ai.providers import make_provider
        return make_provider(cfg)

    def ensure_checked(self) -> None:
        """Check the model provider the first time the dock is shown (never at window start-up)."""
        if not self._checked:
            self.refresh_provider()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.ensure_checked()

    def refresh_provider(self) -> None:
        """Rebuild the provider from the preferences and check it, both on a worker thread.

        Building can import the ``anthropic`` SDK and look for credentials, and
        checking asks the Ollama server for its models: neither may stall the
        window, so the header says "checking" until the answer arrives.
        """
        self._checked = True
        self._provider_gen += 1
        gen = self._provider_gen
        cfg = self._config()
        self.model_label.setText("Checking the model…")
        bridge = self._bridge

        def work() -> None:
            provider = None
            try:
                provider = self._make_provider(cfg)
                ok, msg = provider.check()
            except Exception as exc:  # noqa: BLE001 - a broken provider is shown, never fatal
                ok, msg = False, f"The assistant could not start: {type(exc).__name__}: {exc}"
            with contextlib.suppress(RuntimeError):      # the panel was deleted meanwhile
                bridge.event.emit("_provider", {"ok": ok, "text": msg, "provider": provider, "gen": gen})

        threading.Thread(target=work, name="aerolume-assistant-check", daemon=True).start()

    def _show_provider(self, ok: bool, text: str) -> None:
        """The header names the model ("Local · qwen3:8b"); the check's full text is the tooltip.

        The check text carries the server version ("… · Ollama 0.40.2"), which
        made the line elide in a 430 px dock and says nothing a user acts on.
        """
        self._provider_ready = ok
        name = self.provider.describe() if self.provider is not None else ""
        self.model_label.setText(name or (text if ok else "Assistant unavailable"))
        self.model_label.setToolTip(text)
        theme.set_prop(self.model_label, "role", "secondary" if ok else "status-warn")
        self.setup_line.set_text(text)
        self.setup_line.setVisible(not ok)
        self.setup_button.setVisible(not ok)

    # -- project -------------------------------------------------------------------------------
    def _project_changed(self) -> None:
        if self.is_running():
            self.stop()
        has = self.state.project is not None
        self.goal.setEnabled(has)
        self.run_button.setEnabled(has)
        self.goal.setPlaceholderText(
            "Describe an experiment: what to find, for which design, at which conditions"
            if has else "Open or create a project first: experiments are recorded in the project.")
        self._reset_conversation()
        self._sync_selection_example()

    def _sync_selection_example(self) -> None:
        """Re-word the first example for the design now selected (hidden when none is)."""
        st = self.state
        d = getattr(st, "design", None) if getattr(st, "project", None) is not None else None
        speed = None
        if d is not None:
            sims = [s for s in st.project.simulations.values() if s.design_id == d.id]
            if sims:
                speed = getattr(sims[-1].conditions, "speed", None)
        latest = None
        if d is not None:
            sims = [s for s in st.project.simulations.values() if s.design_id == d.id]
            latest = sims[-1] if sims else None
        text = selection_example(d, speed, latest)
        self.selection_example.setProperty("goal", text)
        self.selection_example.setText(_goal_link(text))
        self.selection_example.setVisible(bool(text))
        self._sync_examples(text)

    def _sync_examples(self, selected: str = "") -> None:
        """Fill the example slots from the project's own designs and physics (:func:`project_examples`);
        a slot the project has nothing for is hidden rather than filled with someone else's study."""
        project = getattr(self.state, "project", None)
        texts = (project_examples(project, len(self.example_buttons), (selected,)) if project is not None
                 else list(EXAMPLES))
        for i, b in enumerate(self.example_buttons):
            t = texts[i] if i < len(texts) else ""
            b.setProperty("goal", t)
            b.setText(_goal_link(t) if t else "")
            b.setVisible(bool(t))

    def _reset_conversation(self) -> None:
        if self.host is not None:
            with contextlib.suppress(Exception):
                self.host.close()
        self.agent = None
        self.host = None
        self._turn = None
        while self.flow.count() > 1:
            item = self.flow.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.empty:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.flow.insertWidget(0, self.empty)
        self.empty.show()
        self.scroll.verticalScrollBar().setValue(0)          # the empty state reads from its heading
        self.new_button.setEnabled(False)

    def new_experiment(self) -> None:
        """Forget the conversation (the log stays in the project) and start over."""
        if self.is_running():
            return
        self._reset_conversation()
        self.goal.setFocus()

    # -- running ----------------------------------------------------------------------------------
    def _selection_context(self) -> str:
        d, s = self.state.design, self.state.simulation
        bits = []
        if d is not None:
            bits.append(f"design “{d.name}” (id {d.id})")
        if s is not None:
            bits.append(f"simulation “{s.name}” (id {s.id})")
        return ", ".join(bits)

    def _build_agent(self):
        from ... import api
        from ...ai.agent import Agent, consent_key
        from ...ai.workspace import Workspace
        from ..scripting import AppHost
        cfg = self._config()
        if self.provider is None:
            self.provider = self._make_provider(cfg)
        self.host = AppHost(self.state, self.jobs)
        project = api.Project(self.state.project, host=self.host)
        host = self.host
        ws = Workspace(project, experiments_changed=lambda: host.post(self.experimentsChanged.emit),
                       run_script=self._run_script)
        key = consent_key(self.provider)
        return Agent(self.provider, ws, config=cfg, approve=self._approve,
                     on_event=lambda k, d: self._bridge.event.emit(k, d), context=self._selection_context(),
                     consent=lambda _dest, k=key: k in self._consented)

    # -- consent before data leaves this computer (Lumen's rule) ---------------------------------
    def _needs_consent(self) -> str:
        """The destination to ask about before the first request of an experiment, or ""."""
        from ...ai.agent import consent_key, data_destination
        if self.agent is not None and self.agent.session is not None:
            return ""                                   # this conversation was already allowed
        if self.provider is None:
            self.provider = self._make_provider(self._config())
        dest = data_destination(self.provider)
        key = consent_key(self.provider)
        if not dest or key in self._consented or key in self._config().consent:
            return ""
        return dest

    def _ask_consent(self, dest: str, text: str) -> None:
        self.empty.hide()
        card = _ConsentCard(dest)
        self._consent_card = card
        self._pending_goal = text
        card.decided.connect(self._consent_decided)
        self._add_block(card)
        self._stick = True

    def _consent_decided(self, ok: bool, remember: bool) -> None:
        from ...ai.agent import consent_key
        from ...ai.config import save_config
        text, self._pending_goal = self._pending_goal, ""
        key = consent_key(self.provider)
        if not ok:
            self._add_block(MessageLine("Nothing was sent. Choose the local model in Preferences > Assistant to keep "
                                        "everything on this computer.", "info"))
            self.goal.setPlainText(text)
            return
        self._consented.add(key)
        if remember:
            cfg = self._config()
            if key not in cfg.consent:
                cfg.consent.append(key)
                save_config(cfg)
        self.goal.setPlainText(text)
        self.submit()

    def submit(self) -> bool:
        """Send the goal box to the assistant; False when nothing was sent.

        Before the first request of an experiment to a model off this computer
        (Claude, or an Ollama server on another host) the panel asks for consent
        with a card, and sends nothing until the user agrees (True is returned:
        the goal was taken and is waiting for that answer).
        """
        text = self.goal.toPlainText().strip()
        if not text or self.is_running() or self.state.project is None:
            return False
        try:
            dest = self._needs_consent()
            if dest:
                self.goal.clear()
                self._ask_consent(dest, text)
                return True
            if self.agent is None:
                self.agent = self._build_agent()
        except Exception as exc:  # noqa: BLE001 - say why instead of failing silently
            self._add_block(MessageLine(f"The assistant could not start: {type(exc).__name__}: {exc}", "error"))
            return False
        self.empty.hide()
        if self._turn is not None:
            self._add_block(Divider())
        desc = self.provider.describe() if hasattr(self.provider, "describe") else ""
        self._turn = _Turn(text, desc)
        self._add_block(self._turn)
        self._stick = True
        self.goal.clear()
        self._set_running(True)
        agent = self.agent
        self._thread = threading.Thread(target=self._work, args=(agent, text), name="aerolume-assistant",
                                        daemon=True)
        self._thread.start()
        return True

    def _work(self, agent, text: str) -> None:
        """Run the request on the worker thread; every way out ends with a "done" event.

        ``agent.run`` emits "done" on each of its own exits. When it raises instead, this
        emits a failed one, for a ``BaseException`` too (then re-raised), because
        :meth:`is_running` stays true until a "done" has been handled.
        """
        try:
            agent.run(text)
        except BaseException as exc:                               # reported in the transcript
            with contextlib.suppress(RuntimeError):                # the panel was deleted meanwhile
                self._bridge.event.emit("error", {"text": f"{type(exc).__name__}: {exc}", "kind": "internal"})
                self._bridge.event.emit("done", {"status": "failed", "text": "", "untraced": [], "path": ""})
            if not isinstance(exc, Exception):
                raise

    def stop(self) -> None:
        """Stop the assistant: the model, the running tool (and its simulation) and any approval wait."""
        if self.agent is not None:
            self.agent.stop()
        if self._engine is not None and self._engine.is_running():
            self._engine.stop()
        self.status_text.setText("Stopping…")

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop everything before the window closes (never waits for the GUI thread)."""
        if self.agent is not None:
            self.agent.stop()
        if self.host is not None:
            with contextlib.suppress(Exception):
                self.host.close()
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.shutdown(timeout)
        if self._thread is not None:
            self._thread.join(timeout)

    def _set_running(self, running: bool) -> None:
        self.run_button.setVisible(not running)
        self.stop_button.setVisible(running)
        self.new_button.setEnabled(not running and self._turn is not None)
        self.status_row.setVisible(running)
        if running:
            self._started = time.monotonic()
            self.status_text.setText("Starting")
            self.elapsed.setText("")
            self._tick.start()
        else:
            self._tick.stop()
        self.runningChanged.emit(running)

    def _on_tick(self) -> None:
        self.elapsed.setText(_fmt_secs(time.monotonic() - self._started))
        if self._turn is not None:
            for row in self._turn.rows.values():
                row.tick()

    # -- approvals (worker thread waits, GUI decides) ------------------------------------------------
    def _approval_entry(self, call_id: str) -> dict:
        with self._approval_lock:
            return self._approvals.setdefault(call_id, {"event": threading.Event(), "ok": False})

    def _approve(self, _approval, call_id: str) -> bool:
        entry = self._approval_entry(call_id)
        agent = self.agent
        while not entry["event"].wait(0.1):
            if agent is None or agent.cancel.is_set():
                return False
        return bool(entry["ok"])

    def decide(self, call_id: str, ok: bool) -> None:
        """Answer a pending approval (the card's buttons; tests call it directly)."""
        entry = self._approval_entry(call_id)
        entry["ok"] = bool(ok)
        entry["event"].set()
        if self._turn is not None and call_id in self._turn.cards:
            self._turn.cards[call_id].set_outcome(ok)

    # -- run_python through the Script console's engine -------------------------------------------
    def _run_script(self, code: str, cancel: threading.Event, progress) -> dict:
        """Run approved code with the Script engine (called on the agent's thread)."""
        from ..scripting import ScriptEngine
        host = self.host
        done = threading.Event()
        box: dict[str, Any] = {}

        def start() -> None:                       # GUI thread
            if self._engine is None:
                self._engine = ScriptEngine(self.state, self.jobs, self)
            eng = self._engine

            def finished(outcome) -> None:
                box["outcome"] = outcome
                with contextlib.suppress(RuntimeError, TypeError):
                    eng.finished.disconnect(finished)
                    eng.progress.disconnect(progress)
                done.set()

            eng.namespace.pop("result", None)
            eng.finished.connect(finished)
            eng.progress.connect(progress)
            if not eng.run(code, filename="<assistant script>") and "outcome" not in box:
                box["busy"] = True
                done.set()

        host.call(start)
        stopped = False
        while not done.wait(0.1):
            if cancel.is_set() and not stopped:
                stopped = True
                host.post(lambda: self._engine.stop())
        if box.get("busy"):
            return {"output": "", "error": "Another script is running in the Script console; try again when it "
                                           "has finished.", "seconds": 0.0}
        o = box["outcome"]
        figures = []
        for i, png in enumerate(o.figures or []):
            if self.agent is not None and self.agent.log is not None:
                p = self.agent.log.plots_dir
                p.mkdir(parents=True, exist_ok=True)
                f = p / f"script-{int(time.time())}-{i + 1}.png"
                f.write_bytes(png)
                self.agent.log.add_plot(f)
                figures.append(str(f))
        err = o.error if o.status == "failed" else ("Stopped by the user." if o.status == "stopped" else "")
        return {"output": o.output, "error": err, "traceback": o.traceback, "seconds": o.seconds,
                "sim_ids": list(o.sim_ids), "result": self._engine.namespace.get("result"), "figures": figures}

    # -- events from the agent (GUI thread) ------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        """Handle one worker event on the GUI thread; a "done" always ends the run.

        :meth:`_done` ends it while drawing the outcome. The ``finally`` covers the cases where
        it cannot: the conversation was reset while the request ran (another project opened,
        so there is no turn to draw into), or a widget was deleted half-way through drawing.
        Without it :meth:`is_running` would stay true and Stop would stay on screen.
        """
        try:
            self._handle(kind, data or {})
        except RuntimeError:
            pass          # a widget of a reset conversation was deleted while the event was queued
        finally:
            if kind == "done" and self._thread is not None:
                self._thread = None
                with contextlib.suppress(RuntimeError):          # the panel is being deleted
                    self._set_running(False)

    def _handle(self, kind: str, data: dict) -> None:
        if kind == "_provider":
            if data.get("gen") == self._provider_gen:
                new = data.get("provider")
                if new is not None and not self.is_running():
                    old = self.provider
                    self.provider = new
                    if self.agent is not None and (old is None or old.describe() != new.describe()):
                        # a session belongs to one model: the next request starts a new experiment
                        self.agent = None
                        self._add_block(MessageLine(f"Model changed to {new.describe()}: the next request starts "
                                                    "a new experiment.", "info"))
                self._show_provider(bool(data.get("ok")), str(data.get("text", "")))
            return
        turn = self._turn
        if turn is None:
            return
        if kind == "status":
            self.status_text.setText(str(data.get("text", "")))
        elif kind == "text":
            turn.append_text(str(data.get("delta", "")))
        elif kind == "thinking":
            turn.append_reasoning(str(data.get("delta", "")))
        elif kind == "turn":
            turn.close_segment()
        elif kind == "tool_start":
            turn.close_segment()
            cid, name = str(data.get("id", "")), str(data.get("name", ""))
            row = _ToolRow(cid, name, data.get("args"), tool_subtitle(name, data.get("args"), self.state.project))
            turn.rows[cid] = row
            turn.add(row)
        elif kind == "tool_progress":
            row = turn.rows.get(str(data.get("id", "")))
            if row is not None:
                row.set_progress(float(data.get("fraction", 0.0)), str(data.get("message", "")))
                if data.get("message"):
                    self.status_text.setText(str(data["message"]))
        elif kind == "approval":
            cid = str(data.get("id", ""))
            card = _ApprovalCard(cid, data.get("approval"))
            card.decided.connect(self.decide)
            turn.cards[cid] = card
            turn.add(card)
            self._stick = True
        elif kind == "tool_end":
            self._tool_end(turn, data)
        elif kind == "notice":
            # not isHidden(): a block added a moment ago is "hidden" until Qt's queued show reaches it, and a
            # correction notice queued right behind its text arrived first, so the answer it sent back kept the
            # normal colour about half the time (seen while replaying critic round 3's transcripts)
            if data.get("corrects") and turn.answers and not turn.answers[-1].property("dropped"):
                turn.answers[-1].set_superseded()
            level = "warning" if data.get("level") == "warning" else "info"
            line = MessageLine(str(data.get("text", "")), level)
            if level == "info":
                line.text.setProperty("role", "muted")
            turn.add(line)
        elif kind == "error":
            line = MessageLine(str(data.get("text", "")), "error")
            turn.add(line)
            if data.get("kind") in ("credentials", "auth", "connection", "not_found", "unavailable"):
                b = button("Assistant settings…", None, "settings")
                b.clicked.connect(self.preferencesRequested.emit)
                turn.add(b)
        elif kind == "done":
            self._done(turn, data)

    def _tool_end(self, turn: _Turn, data: dict) -> None:
        cid, name = str(data.get("id", "")), str(data.get("name", ""))
        row = turn.rows.get(cid)
        status = str(data.get("status", "ok"))
        if row is None:
            row = _ToolRow(cid, name, data.get("args"), "")
            turn.rows[cid] = row
            turn.add(row)
        row.finish(status, str(data.get("summary", "")), float(data.get("seconds", 0.0) or 0.0), data.get("args"))
        card = turn.cards.get(cid)
        if card is not None and card.approve.isVisible():
            card.set_outcome(status != "declined", "Declined" if status == "declined" else None)
        if name == "record_plan" and status == "ok" and isinstance(data.get("plan") or data.get("args"), dict):
            # the plan as recorded: its expected time is the planner's estimate, never the model's guess (a
            # "5 min 00 s" plan for an 11 s run, critic round 3.3); the call's own args only without one
            a = data["plan"] if isinstance(data.get("plan"), dict) else data["args"]
            items = []
            # "Controls" only next to variables: without them the entries are the run's settings (experiment.py)
            controls = "Controls" if a.get("variables") else "Settings"
            for key, lab in (("objective", "Objective"), ("method", "Method"), ("variables", "Variables"),
                             ("controls", controls), ("quality", "Quality"), ("expected_runs", "Runs"),
                             ("expected_seconds", "Expected time")):
                v = a.get(key)
                if v in (None, "", []):
                    continue
                if key == "expected_seconds":
                    with contextlib.suppress(TypeError, ValueError):
                        v = _fmt_secs(float(v))
                    if a.get("expected_time_basis"):
                        v = f"{v} ({a['expected_time_basis']})"
                if isinstance(v, list):
                    v = "; ".join(str(x) for x in v)
                items.append((lab, str(v)))
            if items:
                kv = KeyValueList(items)
                for k, v in kv._rows:
                    v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    v.setWordWrap(True)
                row.add_extra(kv)
        for f in data.get("files") or []:
            if str(f).lower().endswith(".png") and Path(f).is_file():
                fb = _FigureBlock(str(f))
                fb.fit(self.scroll.viewport().width() - 6 * SPACING)
                turn.figures.append(fb)
                turn.add(fb)
        if isinstance(data.get("suggestion"), dict) and status == "ok":
            card = _SuggestionCard(data["suggestion"], self._apply_suggestion, self._undo_suggestion)
            turn.suggestions.append(card)
            turn.add(card)
            self._stick = True

    # -- suggestions: applied on the user's click, one undo step -----------------------------------
    def _api_project(self):
        from ... import api
        if self.agent is not None:
            return self.agent.ws.project
        from ..scripting import AppHost
        if self.host is None:
            self.host = AppHost(self.state, self.jobs)
        return api.Project(self.state.project, host=self.host)

    def _apply_suggestion(self, sim_id: str, changes: dict) -> tuple[dict, str]:
        """Apply a suggestion's valid changes (checked again by the API); returns (undo token, name)."""
        import copy as _copy
        project = self._api_project()
        store = project.raw
        if sim_id not in store.simulations:
            raise LookupError("the simulation is no longer in the project")
        before = _copy.deepcopy(store.simulations[sim_id])
        new = project.edit_simulation(sim_id, **changes)
        with contextlib.suppress(Exception):
            self.state.select_simulation(new.id)
        return {"sim_id": new.id, "copied": new.id != sim_id, "before": before}, new.name

    def _undo_suggestion(self, token: dict) -> None:
        """Take a whole suggestion back: remove the copy it made, or restore the settings it changed."""
        project = self._api_project()
        store = project.raw
        if token.get("copied"):
            if token["sim_id"] in store.simulations:
                project.remove_simulation(token["sim_id"])
            return
        before = token["before"]
        cur = store.simulations.get(before.id)
        if cur is None:
            raise LookupError("the simulation is no longer in the project")
        if cur.status in ("running", "queued"):
            raise RuntimeError(f"“{cur.name}” is {cur.status}; undo when it has finished")
        project.host.update_simulation(store, before)

    def _done(self, turn: _Turn, data: dict) -> None:
        status = str(data.get("status", "done"))
        if status == "done":
            turn.drop_repeated_answers()
        self._set_running(False)
        self._thread = None
        path = str(data.get("path", "") or "")
        took = _fmt_secs(time.monotonic() - self._started)
        claims = data.get("claims") or {}
        block = _ClaimsBlock(claims) if claims.get("statements") else None
        n_bad = sum(1 for s in claims.get("statements", []) if isinstance(s, dict)
                    and s.get("status") in ("unsupported", "contradicted"))
        unverified = (f"{n_bad} statement{'s' if n_bad != 1 else ''} not verified · {took}" if n_bad
                      else f"Not verified · {took}")
        words = {"done": f"Done in {took}", "unverified": unverified,
                 "stopped": "Stopped", "failed": "Failed",
                 "refused": "The model declined the request"}.get(status, status)
        if status == "refused" and data.get("text"):
            turn.add(MessageLine(str(data["text"]), "warning"))
        if status == "unverified" and data.get("withheld"):
            self._withhold(turn, [str(t) for t in data.get("withheld") or []], str(data.get("text", "")))
        if block is not None and block.statements:
            turn.claims = block
            turn.add(block)
        row = QHBoxLayout()
        row.setSpacing(2 * SPACING)
        icon = QLabel()
        name, colour = {"done": ("check-circle", "good"), "stopped": ("stop", "warn"),
                        "unverified": ("warning", "warn")}.get(status, ("error", "bad"))
        icon.setPixmap(theme.render_icon(name, 14, colour, self.devicePixelRatioF()))
        row.addWidget(icon)
        said = ElidedLabel(words, "secondary", mode=Qt.TextElideMode.ElideRight)
        said.setToolTip(words)
        if status == "unverified":
            said.setToolTip(words + " — " + ("the answer claimed an action that no tool call performed, so it was "
                                             "withheld" if data.get("withheld") else
                                             "some statements are not supported by the tool results: see Checked "
                                             "statements"))
        row.addWidget(said, 1)
        if path:
            # the log's folder name is the button's tooltip: beside the outcome it squeezed the one thing the
            # row must say ("1 statement not verified · 3…" in a 306 px dock)
            open_btn = button("Open log", "link")
            open_btn.setToolTip(f"Experiment log {Path(path).name}\n{Path(path) / 'experiment.md'}")
            open_btn.clicked.connect(lambda: self.experimentOpened.emit(path))
            row.addWidget(open_btn)
        else:
            row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, SPACING, 0, 0)
        turn.add(holder)
        self.experimentsChanged.emit()
        self.goal.setFocus()

    @staticmethod
    def _withhold(turn: _Turn, withheld: list[str], replacement: str) -> None:
        """Hide answers the action check withheld; show why, with the last one folded away for reference.

        The answers claimed an action (a deletion, a run) that no tool call
        performed. They streamed in as ordinary answers, so they are taken out
        of the reading flow here rather than left looking like a result.
        """
        def norm(s: str) -> str:
            return " ".join(s.split())
        gone = {norm(t) for t in withheld}
        for b in turn.answers:
            if norm(b.plain()) in gone:
                b.hide()
        turn.add(MessageLine(replacement, "warning"))
        if withheld:
            block = _TextBlock(withheld[-1], role="muted")
            sec = CollapsibleSection("Withheld answer", block, expanded=False)
            sec.header.setToolTip("What the model answered; it is kept in the experiment log")
            turn.add(sec)


# ---------------------------------------------------------------------------
# Preferences > Assistant
# ---------------------------------------------------------------------------

class _ModelsBridge(QObject):
    listed = Signal(object)


class AssistantSettingsPage(QWidget):
    """The Assistant tab of Preferences: provider, models, endpoint, API key, approvals.

    :meth:`apply` writes ``assistant.json`` (and the key file) in the user
    config folder (:mod:`aerolume.ai.config`) and returns whether anything
    changed. The key field is a password field that starts empty: it never
    shows a saved key, only its masked form in the line beneath; typing a new
    one replaces it and "Remove saved key" deletes it.
    """

    CLAUDE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")

    def __init__(self, parent: QWidget | None = None, *, list_models: bool = True) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLineEdit, QSpinBox

        from ...ai import config as C
        self._C = C
        self.cfg = C.load_config()
        self._remove_key = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2 * SPACING, 0, 0)
        lay.setSpacing(2 * SPACING)

        def form() -> QFormLayout:
            f = QFormLayout()
            f.setHorizontalSpacing(3 * SPACING)
            f.setVerticalSpacing(2 * SPACING)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return f

        top = form()
        self.provider = QComboBox()
        for key, text in (("auto", "Automatic: Claude when credentials exist, else the local model"),
                          ("claude", "Claude (Anthropic API)"), ("ollama", "Local model (Ollama)")):
            self.provider.addItem(text, key)
        self.provider.setCurrentIndex(max(0, self.provider.findData(self.cfg.provider)))
        top.addRow(label("Model provider", "pg-label"), self.provider)
        lay.addLayout(top)

        lay.addWidget(label("Claude", "h3"))
        fc = form()
        self.claude_model = QComboBox()
        self.claude_model.setEditable(True)
        self.claude_model.addItems(list(self.CLAUDE_MODELS))
        self.claude_model.setCurrentText(self.cfg.claude_model)
        fc.addRow(label("Model", "pg-label"), self.claude_model)
        self.effort = QComboBox()
        for e in C.EFFORTS:
            self.effort.addItem(e, e)
        self.effort.setCurrentIndex(max(0, self.effort.findData(self.cfg.claude_effort)))
        self.effort.setToolTip("How much the model thinks before acting: higher is slower, costs more and plans "
                               "better")
        fc.addRow(label("Effort", "pg-label"), self.effort)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("Empty: use ANTHROPIC_API_KEY or an ant auth login profile")
        fc.addRow(label("API key", "pg-label"), self.key)
        krow = QHBoxLayout()
        krow.setSpacing(2 * SPACING)
        self.key_status = label("", "small", wrap=True)
        krow.addWidget(self.key_status, 1)
        self.remove_key = button("Remove saved key", "link")
        self.remove_key.clicked.connect(self._remove_saved_key)
        krow.addWidget(self.remove_key)
        fc.addRow(QWidget(), krow)
        lay.addLayout(fc)

        lay.addWidget(label("Local model", "h3"))
        fl = form()
        self.endpoint = QLineEdit(self.cfg.ollama_endpoint)
        self.endpoint.setToolTip("Address of the Ollama server (ollama.com); it runs the model on this computer")
        fl.addRow(label("Ollama address", "pg-label"), self.endpoint)
        mrow = QHBoxLayout()
        mrow.setSpacing(SPACING)
        self.local_model = QComboBox()
        self.local_model.setEditable(True)
        self.local_model.setCurrentText(self.cfg.ollama_model)
        self.local_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mrow.addWidget(self.local_model, 1)
        self.refresh = IconButton("refresh", "List the models installed in Ollama")
        self.refresh.clicked.connect(self.refresh_models)
        mrow.addWidget(self.refresh)
        fl.addRow(label("Model", "pg-label"), mrow)
        self.models_status = label("", "small", wrap=True)
        fl.addRow(QWidget(), self.models_status)
        self.think = QCheckBox("Reason before each answer (better plans, several times slower)")
        self.think.setChecked(self.cfg.ollama_think)
        fl.addRow(label("Reasoning", "pg-label"), self.think)
        self.num_ctx = QSpinBox()
        self.num_ctx.setRange(4096, 131072)
        self.num_ctx.setSingleStep(2048)
        self.num_ctx.setValue(self.cfg.ollama_num_ctx)
        self.num_ctx.setSuffix(" tokens")
        self.num_ctx.setToolTip("Context window of the local model. 16384 (default) holds an experiment with its "
                                "runs and corrections; 8192 keeps an 8-billion-parameter model entirely in 8 GB of "
                                "GPU memory but fills up after about one run")
        fl.addRow(label("Context", "pg-label"), self.num_ctx)
        lay.addLayout(fl)

        lay.addWidget(label("Approvals", "h3"))
        fa = form()
        self.threshold = QSpinBox()
        self.threshold.setRange(0, 86400)
        self.threshold.setSingleStep(30)
        self.threshold.setValue(round(self.cfg.approval_threshold_s))
        self.threshold.setSuffix(" s")
        self.threshold.setToolTip("Runs whose estimated time is longer ask first (0: every run asks). Python code "
                                  "and deletions always ask.")
        fa.addRow(label("Ask before runs longer than", "pg-label"), self.threshold)
        self.max_steps = QSpinBox()
        self.max_steps.setRange(4, 200)
        self.max_steps.setValue(self.cfg.max_steps)
        self.max_steps.setToolTip("Tool rounds per request before the assistant stops and reports")
        fa.addRow(label("Tool rounds per request", "pg-label"), self.max_steps)
        lay.addLayout(fa)
        note = label("Claude: your goals, the project's design and simulation names and the tools' results are sent "
                     "to Anthropic, after you agree on the first request of an experiment. Local model: nothing "
                     "leaves this computer. See docs/ai_assistant.md.", "small", wrap=True)
        lay.addWidget(note)
        crow = QHBoxLayout()
        crow.setSpacing(2 * SPACING)
        self._consent = list(self.cfg.consent)
        self.consent_status = label("", "small", wrap=True)
        crow.addWidget(self.consent_status, 1)
        self.forget_consent = button("Ask again", "link")
        self.forget_consent.setToolTip("Forget the destinations allowed with 'Do not ask again'")
        self.forget_consent.clicked.connect(self._forget_consent)
        crow.addWidget(self.forget_consent)
        lay.addLayout(crow)
        self._sync_consent()
        lay.addStretch(1)
        self._bridge = _ModelsBridge()
        self._bridge.listed.connect(self._fill_models, Qt.ConnectionType.QueuedConnection)
        self._sync_key_status()
        if list_models:
            self.refresh_models()

    # -- data consent ----------------------------------------------------------------------------
    def _sync_consent(self) -> None:
        names = {"claude": "Anthropic (Claude API)"}
        allowed = [names.get(k, k.replace("ollama:", "Ollama at ")) for k in self._consent]
        self.consent_status.setText("Sent without asking: " + ", ".join(allowed) if allowed else
                                    "Each experiment asks before anything is sent off this computer.")
        self.forget_consent.setVisible(bool(allowed))

    def _forget_consent(self) -> None:
        self._consent = []
        self._sync_consent()

    # -- key -----------------------------------------------------------------------------------
    def _sync_key_status(self) -> None:
        import os
        stored = "" if self._remove_key else self._C.stored_key()
        if stored:
            text = f"A key is saved in your user settings ({self._C.mask_key(stored)})."
        elif not self._C.anthropic_available():
            text = "Claude needs the optional anthropic package: pip install aerolume[ai]."
        elif os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            text = "No key saved here: the ANTHROPIC_API_KEY environment variable is used."
        else:
            text = "No key saved here: set ANTHROPIC_API_KEY, run ant auth login, or enter a key."
        self.key_status.setText(text)
        self.remove_key.setVisible(bool(stored))

    def _remove_saved_key(self) -> None:
        self._remove_key = True
        self.key.clear()
        self._sync_key_status()

    # -- local models ------------------------------------------------------------------------------
    def refresh_models(self) -> None:
        endpoint = self.endpoint.text().strip() or self._C.DEFAULT_OLLAMA_ENDPOINT
        self.models_status.setText("Looking for Ollama…")
        bridge = self._bridge

        def work() -> None:
            from ...ai.providers.ollama import list_models, server_version
            version = server_version(endpoint)
            models = list_models(endpoint) if version else []
            with contextlib.suppress(RuntimeError):
                bridge.listed.emit((version, models))

        threading.Thread(target=work, name="aerolume-ollama-models", daemon=True).start()

    def _fill_models(self, payload) -> None:
        version, models = payload
        current = self.local_model.currentText().strip()
        self.local_model.clear()
        for m in models:
            self.local_model.addItem(m["name"])
            i = self.local_model.count() - 1
            bits = [b for b in (m.get("parameters"), f"{m.get('size_gb', 0):g} GB") if b]
            self.local_model.setItemData(i, " · ".join(bits), Qt.ItemDataRole.ToolTipRole)
        self.local_model.setCurrentText(current or self._C.DEFAULT_OLLAMA_MODEL)
        if version is None:
            self.models_status.setText("Ollama is not running at this address. Install it from ollama.com, then "
                                       "run: ollama pull qwen3:8b")
        elif not models:
            self.models_status.setText(f"Ollama {version} · no model with tool calling is installed "
                                       "(ollama pull qwen3:8b).")
        else:
            n = len(models)
            self.models_status.setText(f"Ollama {version} · {n} model{'s' if n != 1 else ''} with tool calling")

    # -- values ------------------------------------------------------------------------------------
    def values(self):
        """The configuration the page shows (not yet saved)."""
        c = self._C.AssistantConfig(
            provider=self.provider.currentData(), claude_model=self.claude_model.currentText().strip(),
            claude_effort=self.effort.currentData(), claude_max_tokens=self.cfg.claude_max_tokens,
            ollama_model=self.local_model.currentText().strip(), ollama_endpoint=self.endpoint.text().strip(),
            ollama_think=self.think.isChecked(), ollama_num_ctx=self.num_ctx.value(),
            approval_threshold_s=float(self.threshold.value()), max_steps=self.max_steps.value(),
            consent=list(self._consent))
        return c.validated()

    def apply(self) -> bool:
        """Save what changed; True when anything did (settings or key)."""
        changed = False
        new = self.values()
        if new != self.cfg.validated():
            self._C.save_config(new)
            self.cfg = new
            changed = True
        typed = self.key.text().strip()
        if typed:
            self._C.store_key(typed)
            self.key.clear()
            changed = True
        elif self._remove_key and self._C.stored_key():
            self._C.store_key("")
            changed = True
        self._remove_key = False
        self._sync_key_status()
        return changed


__all__ = ["EXAMPLES", "TOOL_TITLES", "AssistantPanel", "AssistantSettingsPage", "design_questions",
           "physics_example", "physics_examples", "physics_questions", "project_examples", "selection_example",
           "tool_subtitle"]
