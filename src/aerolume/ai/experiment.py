"""Experiment logs: what the assistant was asked, planned, ran, found and concluded.

Every assistant conversation in a project is an *experiment*, stored next to the
designs and simulations so it travels with the project::

    <project>/experiments/<id>/experiment.json   machine-readable record
    <project>/experiments/<id>/experiment.md     the same as a readable report
    <project>/experiments/<id>/plots/*.png       figures the assistant made

The id is the start time plus a slug of the goal (``20261011-143005-best-l-d``),
so the folder list sorts by date. Both files are rewritten atomically after
every step, so a crash or a Stop never leaves a half-written log, and the
Project tree can list an experiment while it is still running.

**Number tracing.** The assistant must not invent numbers (DESIGN.md 9), and a
log is the place to check. Every number that appears in a tool result, a tool
input or the user's own words is remembered; numbers in the conclusions that
match none of them (at the precision they are written with) are listed under
"Number check". A derived quantity (a percentage difference the model computed)
shows up there too, which is the point: a reader sees which figures came
straight from a solver and which did not.

No Qt here: the CLI and the tests write logs as the GUI does.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .plaintext import brief

EXPERIMENTS_DIR = "experiments"
RECORD_FILE = "experiment.json"
REPORT_FILE = "experiment.md"
PLOTS_DIR = "plots"
#: "unverified": the model finished, but its answer claimed an action (a deletion, a run) that no tool call
#: in the experiment performed, even after one correction request; the answer was withheld (agent.py)
STATUSES = ("running", "done", "unverified", "stopped", "failed")
#: the longest goal shown as the log's title; the full goal is under "Goal"
TITLE_CHARS = 80


def _local_now() -> datetime:
    """The local time, timezone-aware (logs are read by people in their own time zone)."""
    return datetime.now().astimezone()


def _now() -> str:
    return _local_now().isoformat(timespec="seconds")


def slug(text: str, n: int = 40) -> str:
    """A short file-name-safe slug of a goal ("Find the best L/D" -> "find-the-best-l-d")."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-") or "experiment")


def fmt_seconds(s: float) -> str:
    """"42 s", "3 min 12 s", "1 h 05 min" (log and chat)."""
    if not (s >= 0 and math.isfinite(s)):
        return "—"
    if s < 60:
        return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"
    if s < 3600:
        m, sec = divmod(round(s), 60)
        return f"{m} min {sec:02d} s"
    h, rem = divmod(round(s), 3600)
    return f"{h} h {rem // 60:02d} min"


# ---------------------------------------------------------------------------
# number tracing
# ---------------------------------------------------------------------------

# A number not glued to a word on either side ("abc123", hex ids like "8e4012a" are not numbers).
_NUM = re.compile(r"(?<![\w.])[-−+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:\s*(?:[eE][-−+]?\d+|[×x]\s*10\s*(?:\^|\*\*)?\s*"
                  r"[-−⁻+]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?(?![\w⁰¹²³⁴⁵⁶⁷⁸⁹])")
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _parse_number(tok: str) -> tuple[float, float] | None:
    """(value, half a unit of its last written digit) of a number token, or None."""
    t = tok.replace("−", "-").replace(" ", "").translate(_SUPERSCRIPT)
    exp = 0
    m = re.match(r"^([-+]?[\d.,]+)(?:[eE]([-+]?\d+)|[×x]10(?:\^|\*\*)?([-+]?\d+))?$", t)
    if not m:
        return None
    mant = m.group(1)
    if mant.count(",") and not mant.count("."):
        # "1,250" (thousands) vs "0,5" (decimal comma): thousands when exactly 3 digits follow
        parts = mant.split(",")
        mant = mant.replace(",", "") if all(len(p) == 3 for p in parts[1:]) else mant.replace(",", ".")
    else:
        mant = mant.replace(",", "")
    if m.group(2):
        exp = int(m.group(2))
    elif m.group(3):
        exp = int(m.group(3))
    if abs(exp) > 300:
        return None
    try:
        v = float(mant) * 10.0 ** exp
    except (ValueError, OverflowError):
        return None
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    half = 0.5 * 10.0 ** (exp - decimals)
    return v, half


def extract_numbers(text: str) -> list[tuple[str, float, float]]:
    """(token, value, rounding half-width) for every number written in ``text``."""
    out = []
    for m in _NUM.finditer(text or ""):
        tok = m.group(0).strip()
        parsed = _parse_number(tok)
        if parsed is not None:
            out.append((tok, parsed[0], parsed[1]))
    return out


def _walk_numbers(obj: Any, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(v for _t, v, _h in extract_numbers(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, out)


def numbers_in(obj: Any) -> list[float]:
    """Every finite number in a JSON-like object, including numbers written inside strings."""
    out: list[float] = []
    _walk_numbers(obj, out)
    return out


#: keys whose strings are names someone typed (a design called "NACA 2412 (wind tunnel: Cd 0.0061)"):
#: their digits are never evidence for a claim. A project's names "verified" the numbers they carried
#: in an answer (critic round 5); measured values are JSON numbers, and tools' own sentences keep counting.
_NAME_KEY = re.compile(r"(?:^|_)(?:name|names|title|label|design|designs|simulation|file|path|folder|project|"
                       r"id|ids|code)$", re.IGNORECASE)


def labelled_numbers(obj: Any, *, origin: str, tool: str = "") -> list[Any]:
    """Every number of a JSON-like object with where it sits, for the claims checker.

    Each becomes a :class:`~aerolume.ai.claims.NumberSource` labelled by its key
    and the simulation it belongs to ("ld_max in “Wing polar · 25 m/s”
    (get_results)"): a dict holding a ``simulation_id`` (and a ``name``) starts
    that context for everything inside it. Numbers written inside text (validity
    sentences) are labelled as quoted text of the tool.
    """
    from .claims import NumberSource
    out: list[NumberSource] = []
    where = f" ({tool})" if tool else ""

    def walk(o: Any, key: str, sim_label: str, sim_id: str) -> None:
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            v = float(o)
            if math.isfinite(v):
                if origin == "goal":
                    label = "the goal"
                elif origin == "input":
                    label = f"{tool} input {key}".strip()
                elif origin == "plan":
                    label = f"the plan's {key or 'text'} (record_plan)"
                else:
                    label = f"{key} in {sim_label}{where}" if sim_label else f"{key or 'value'}{where}"
                out.append(NumberSource(v, origin, label, sim_id, key))
            return
        if isinstance(o, str):
            if origin in ("result", "input") and _NAME_KEY.search(key or ""):
                return
            for _t, v, _h in extract_numbers(o):
                label = "the goal" if origin == "goal" else f"text of {tool or 'a tool result'}"
                out.append(NumberSource(v, origin, label, sim_id, key))
            return
        if isinstance(o, dict):
            if isinstance(o.get("simulation_id"), str):
                sim_id = o["simulation_id"]
                sim_label = f"“{o['name']}”" if isinstance(o.get("name"), str) else f"simulation {sim_id}"
            for k, v in o.items():
                walk(v, str(k), sim_label, sim_id)
            return
        if isinstance(o, (list, tuple)):
            for v in o:
                walk(v, key, sim_label, sim_id)

    walk(obj, "", "", "")
    return out


def untraced_numbers(text: str, sources: list[float], *, rel_tol: float = 0.005) -> list[str]:
    """Numbers written in ``text`` that match no source value.

    A written number matches when a source value rounds to it at the precision
    it is written with (98.6 matches 98.5734), or lies within ``rel_tol`` of it
    (so "1.4×10⁶" matches 1.37e6 only if written as 1.37). Small integers
    (|x| <= 10: counts, step numbers, list markers) and four-digit years are
    not checked. A value also matches its negative (lift vs downforce wording)
    and percentages match fractions (12 % vs 0.12).
    """
    srcs = [abs(v) for v in sources]
    missing: list[str] = []
    for tok, v, half in extract_numbers(text):
        a = abs(v)
        if a <= 10 and float(v).is_integer():
            continue
        if 1900 <= a <= 2100 and float(v).is_integer():
            continue
        ok = False
        for s in srcs:
            if abs(s - a) <= max(half * 1.0001, rel_tol * max(a, s)) or (a and abs(s * 100 - a) <= max(half, rel_tol * a)):
                ok = True
                break
        if not ok and tok not in missing:
            missing.append(tok)
    return missing


# A NACA section named in prose; not one inside a list like "NACA 0012/2412/4412" (the validation record).
_NACA = re.compile(r"(?<![/\w])NACA[\s-]*(\d{4,5})(?![/\w])", re.IGNORECASE)


def design_mismatches(text: str, used_designs: list[str]) -> list[str]:
    """NACA sections an answer names that none of the experiment's runs used.

    ``used_designs`` are the names of the designs whose simulations the
    experiment ran or read. Empty when nothing ran (the number check covers
    that case) or every named section was run. Why it exists: asked for NACA
    4412, a small local model ran the NACA 2412 selected in the window and
    reported the result as NACA 4412; every number traced, the design did not.
    """
    named = list(dict.fromkeys(m.group(1) for m in _NACA.finditer(text or "")))
    if not named or not used_designs:
        return []
    used = {c for n in used_designs for c in re.findall(r"\d{4,5}", str(n))}
    return [f"NACA {c}" for c in named if c not in used]


# ---------------------------------------------------------------------------
# conditions a goal names, and which simulation a sentence is about
# ---------------------------------------------------------------------------

_NUMS_BEFORE_UNIT = r"((?:\d+(?:\.\d+)?\s*(?:,|and|or|to|vs\.?|versus|-|–|−)\s*)*\d+(?:\.\d+)?)"
_SPEEDS = re.compile(_NUMS_BEFORE_UNIT + r"\s*m/s\b", re.IGNORECASE)
_ALPHAS = re.compile(r"(?:\b(?:alpha|aoa|angle of attack)|α)\s*(?:=|of|at)?\s*(-?\d+(?:\.\d+)?)\s*(?:°|deg)?",
                     re.IGNORECASE)
_RE = re.compile(r"\bRe\s*(?:=|of)?\s*(\d+(?:\.\d+)?(?:\s*(?:[eE][-+]?\d+|[×x]\s*10\s*(?:\^|\*\*)?\s*[-+⁻]?"
                 r"[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)")


def _floats(text: str) -> list[float]:
    return [float(t) for t in re.findall(r"\d+(?:\.\d+)?", text)]


@dataclass
class GoalTarget:
    """A design a goal names together with the conditions named near it ("Tapered wing ... at 15 m/s")."""
    design: str
    speed: float | None = None
    alpha: float | None = None
    re: float | None = None

    def text(self) -> str:
        parts = [self.design]
        if self.speed is not None:
            parts.append(f"at {self.speed:g} m/s")
        if self.alpha is not None:
            parts.append(f"at α {self.alpha:g}°")
        if self.re is not None:
            parts.append(f"at Re {self.re:.3g}")
        return " ".join(parts)


def goal_speeds(goal: str) -> list[float]:
    """Every speed a goal names, in order ("between 15 and 25 m/s" -> [15, 25])."""
    out: list[float] = []
    for m in _SPEEDS.finditer(goal or ""):
        for v in _floats(m.group(1)):
            if v not in out:
                out.append(v)
    return out


def goal_targets(goal: str, design_names: list[str]) -> list[GoalTarget]:
    """The (design, condition) pairs a goal asks about.

    Each speed, angle or Reynolds number the goal names is paired with the
    design named closest before it (else the first design named after it), so
    "compare the Tapered wing at 15 m/s with the stored 25 m/s polar" asks for
    the Tapered wing at both speeds, and "NACA 2412 at 20 m/s and the sedan at
    30 m/s" pairs each design with its own speed. Design names are matched
    whole and case-insensitively; a longer name wins over a shorter one it
    contains. Goals that name no design or no condition give no targets: the
    coverage check (agent.py) then has nothing to hold the model to.
    """
    text = goal or ""
    low = text.lower()
    spans: list[tuple[int, int, str]] = []
    for name in sorted({n for n in design_names if n and n.strip()}, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", low):
            if not any(a < m.end() and m.start() < b for a, b, _n in spans):
                spans.append((m.start(), m.end(), name))
    if not spans:
        return []
    spans.sort()

    def owner(pos: int) -> str:
        before = [s for s in spans if s[1] <= pos]
        return before[-1][2] if before else spans[0][2]

    out: list[GoalTarget] = []

    def add(t: GoalTarget) -> None:
        if t not in out:
            out.append(t)

    for m in _SPEEDS.finditer(text):
        for v in _floats(m.group(1)):
            add(GoalTarget(owner(m.start()), speed=v))
    for m in _ALPHAS.finditer(text):
        add(GoalTarget(owner(m.start()), alpha=float(m.group(1))))
    for m in _RE.finditer(text):
        parsed = _parse_number(m.group(1))
        if parsed is not None and parsed[0] >= 1000:
            add(GoalTarget(owner(m.start()), re=parsed[0]))
    return out


@dataclass
class SimFacts:
    """What the attribution check knows about one simulation of the experiment.

    ``values`` are all numbers of its result (tables included): a number the
    answer gives for this simulation must be one of them. ``headline`` are only
    its summary metrics and coefficients: a number that misses the named
    simulation is called misattributed only when it is the headline figure of
    another one, which keeps validity prose ("within 2-2.5 %") out of it.
    """
    sim_id: str
    label: str
    design: str
    speed: float | None
    values: list[float] = field(default_factory=list)
    headline: list[float] = field(default_factory=list)


_CLAUSE_SPLIT = re.compile(r"[;\n]|(?<!\d)[.!?](?!\d)|,\s|\s(?:while|whereas|but|matching|compared (?:with|to)|"
                           r"versus|vs\.?|than|against)\s", re.IGNORECASE)


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 + 0.005 * max(abs(a), abs(b))


def attribution_problems(text: str, sims: list[SimFacts]) -> list[str]:
    """Numbers an answer gives for one simulation that belong to another.

    The text is cut into clauses (at sentence ends, semicolons, commas and
    words such as "while", "matching", "compared with"). A clause that names
    a speed ("25 m/s") or a design name is about the simulations with that
    speed and design. A number in it that matches none of those simulations'
    values but is a headline value of another simulation of the experiment is
    reported: qwen3:8b wrote "best L/D at 15 m/s is 26.47, matching the stored
    25 m/s polar's best L/D of 26.47" although the 25 m/s run's figure is
    29.52, and every number traced to *some* tool result. Numbers that match no
    simulation at all are the plain number check's job, not this one's.
    """
    if len(sims) < 2:
        return []
    out: list[str] = []
    for clause in _CLAUSE_SPLIT.split(text or ""):
        if not clause or not clause.strip():
            continue
        speeds = [v for m in _SPEEDS.finditer(clause) for v in _floats(m.group(1))]
        low = clause.lower()
        designs = {s.design for s in sims if s.design and s.design.lower() in low}
        if not speeds and not designs:
            continue
        named = [s for s in sims
                 if (not speeds or (s.speed is not None and any(_same(s.speed, v) for v in speeds)))
                 and (not designs or s.design in designs)]
        if not named or len(named) == len(sims):
            continue
        speed_tokens = {f"{v:g}" for v in speeds}
        for tok, v, half in extract_numbers(clause):
            a = abs(v)
            if (a <= 10 and float(v).is_integer()) or tok.lstrip("+-−") in speed_tokens:
                continue
            if not untraced_numbers(tok, [x for s in named for x in s.values]):
                continue
            others = [s for s in sims if s not in named and not untraced_numbers(tok, s.headline)]
            if others:
                where = " / ".join(s.label for s in named[:2])
                msg = (f"{tok} is given for {where}, but it is a value of {others[0].label}")
                if msg not in out:
                    out.append(msg)
    return out


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """One tool call the assistant made (or tried), as the log shows it."""
    time: str
    tool: str
    input: dict
    status: str                          # "ok" | "error" | "declined" | "cancelled" | "invalid"
    summary: str = ""
    seconds: float = 0.0
    simulation_ids: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    id: str
    goal: str
    created: str
    updated: str
    provider: str = ""
    model: str = ""
    status: str = "running"
    plan: dict | None = None
    actions: list[Action] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)       # key numbers: label, value, unit, uncertainty, source
    conclusions: str = ""
    findings: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)  # {"role": "user"|"assistant"|"aerolume", "text", "time"}
    untraced: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)          # other consistency findings (design check)
    #: the claims check of the final answer (:class:`aerolume.ai.claims.ClaimReport`.to_dict())
    claims: dict = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)           # paths relative to the experiment folder
    usage: dict = field(default_factory=dict)
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentRecord:
        names = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in names}
        kw["actions"] = [Action(**{k: v for k, v in a.items() if k in Action.__dataclass_fields__})
                         for a in d.get("actions", []) if isinstance(a, dict)]
        return cls(**kw)


class ExperimentLog:
    """The live log of one experiment (thread-safe; every change is saved at once)."""

    def __init__(self, project_root: str | Path, goal: str, *, provider: str = "", model: str = "",
                 experiment_id: str | None = None) -> None:
        self.root = Path(project_root) / EXPERIMENTS_DIR
        stamp = _local_now().strftime("%Y%m%d-%H%M%S")
        eid = experiment_id or f"{stamp}-{slug(goal)}"
        base, k = eid, 2
        while (self.root / eid).exists() and experiment_id is None:
            eid, k = f"{base}-{k}", k + 1
        self.path = self.root / eid
        self.record = ExperimentRecord(id=eid, goal=goal.strip(), created=_now(), updated=_now(),
                                       provider=provider, model=model)
        self._sources: list[float] = []
        self._labelled: list[Any] = []
        self._lock = threading.RLock()
        self._t0 = time.monotonic()
        #: what this experiment set up to run, for the plan's "Expected time" (not saved): "sim:<id>" -> None
        #: for a simulation created and not yet estimated, "cost:<call>" -> seconds for a planner estimate
        self.planned: dict[str, float | None] = {}
        self.record_message("user", goal)

    def note_planned(self, key: str, seconds: float | None = None) -> None:
        """Remember a run this experiment set up (``seconds`` None) or the planner's estimate of one.

        The plan's expected time is the planner's, never the model's: qwen3:8b
        wrote "Expected time 5 min 00 s" into a plan whose run took 11 s (critic
        round 3.3). :func:`aerolume.ai.tools.planner_expected` sums these.
        """
        with self._lock:
            if seconds is not None or key not in self.planned:
                self.planned[key] = None if seconds is None else float(seconds)

    # -- sources of trusted numbers -------------------------------------------------
    def add_source(self, obj: Any, *, origin: str = "result", tool: str = "") -> None:
        """Remember the numbers in ``obj`` (a tool result, a tool input, the user's words).

        ``origin`` ("result", "input", "goal" or "validity": the solvers' validity
        record from the system prompt) and ``tool`` label each number for the claims
        checker, which shows the user where a verified number came from.
        The plan's numbers are the model's own intentions, so they get the origin
        "plan": the claims checker accepts them only in a sentence about the plan
        (qwen3:8b's "trustworthy for differences above ±1.5 %" was "verified" by the
        ±1.5 % it had written into its own plan). The validity record's numbers count
        only in a statement about validity (claims check), so they stay out of the
        plain number check of the conclusions.
        """
        if tool == "record_plan" and origin == "input":
            origin = "plan"
        with self._lock:
            if origin != "validity":
                self._sources.extend(numbers_in(obj))
            self._labelled.extend(labelled_numbers(obj, origin=origin, tool=tool))

    def labelled_sources(self) -> list[Any]:
        """Every remembered number with its origin and label (:class:`~aerolume.ai.claims.NumberSource`)."""
        with self._lock:
            return list(self._labelled)

    def set_claims(self, report: dict) -> None:
        """Record the claims check of the final answer (shown under "Verification" in the log)."""
        with self._lock:
            self.record.claims = dict(report or {})
            self.save()

    def untraced_in(self, text: str) -> list[str]:
        """Numbers in ``text`` that match no recorded source (see :func:`untraced_numbers`)."""
        with self._lock:
            return untraced_numbers(text, list(self._sources))

    @property
    def plots_dir(self) -> Path:
        return self.path / PLOTS_DIR

    # -- recording --------------------------------------------------------------------
    def record_message(self, role: str, text: str) -> None:
        with self._lock:
            if role == "user":
                self.add_source(text, origin="goal")
            self.record.conversation.append({"role": role, "text": text, "time": _now()})
            self.save()

    def record_action(self, action: Action) -> None:
        with self._lock:
            self.record.actions.append(action)
            self.save()

    def set_plan(self, plan: dict) -> None:
        with self._lock:
            self.record.plan = dict(plan)
            self.add_source(plan, origin="input", tool="record_plan")
            self.save()

    def add_results(self, rows: list[dict]) -> None:
        """Key numbers from a tool result (label, value, unit, uncertainty, source simulation)."""
        with self._lock:
            self.record.results.extend(rows)
            self.save()

    def add_plot(self, path: Path) -> str:
        with self._lock:
            try:
                rel = Path(path).resolve().relative_to(self.path.resolve()).as_posix()
            except ValueError:
                rel = str(path)
            if rel not in self.record.plots:
                self.record.plots.append(rel)
            self.save()
            return rel

    def conclude(self, text: str, *, findings: list[dict] | None = None, limitations: list[str] | None = None,
                 status: str = "done") -> list[str]:
        """Record the conclusions; returns the numbers in them that no tool result supports."""
        with self._lock:
            if text:
                self.record.conclusions = text.strip()
            if findings is not None:
                self.record.findings = list(findings)
            if limitations is not None:
                self.record.limitations = list(limitations)
            body = self.record.conclusions + "\n" + "\n".join(
                f"{f.get('value', '')} {f.get('uncertainty', '')}" for f in self.record.findings)
            self.record.untraced = untraced_numbers(body, self._sources)
            self.set_status(status)
            return list(self.record.untraced)

    def add_check(self, text: str) -> None:
        """Record a consistency finding (shown with the number check)."""
        with self._lock:
            if text and text not in self.record.checks:
                self.record.checks.append(text)
            self.save()

    def set_status(self, status: str) -> None:
        with self._lock:
            if status in STATUSES:
                self.record.status = status
            self.record.wall_seconds = time.monotonic() - self._t0
            self.save()

    def set_usage(self, usage: dict) -> None:
        with self._lock:
            self.record.usage = dict(usage)
            self.save()

    # -- files --------------------------------------------------------------------------
    def save(self) -> None:
        from ..project.fileio import write_json, write_text
        with self._lock:
            self.record.updated = _now()
            self.path.mkdir(parents=True, exist_ok=True)
            write_json(self.path / RECORD_FILE, self.record.to_dict())
            write_text(self.path / REPORT_FILE, render_markdown(self.record))


# ---------------------------------------------------------------------------
# reading and listing
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSummary:
    id: str
    path: Path
    goal: str
    status: str
    created: str
    updated: str
    actions: int
    model: str

    @property
    def title(self) -> str:
        line = self.goal.strip().splitlines()[0] if self.goal.strip() else self.id
        return elide(line, 70)


def load_record(path: str | Path) -> ExperimentRecord:
    """The record of the experiment folder ``path`` (or its experiment.json)."""
    p = Path(path)
    if p.is_dir():
        p = p / RECORD_FILE
    from ..project.fileio import read_json
    d = read_json(p, "experiment log")
    if not isinstance(d, dict):
        raise TypeError(f"{p} is not an experiment log")
    return ExperimentRecord.from_dict(d)


def list_experiments(project_root: str | Path) -> list[ExperimentSummary]:
    """Experiments of a project, newest first; unreadable folders are skipped."""
    root = Path(project_root) / EXPERIMENTS_DIR
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        f = d / RECORD_FILE
        if not f.is_file():
            continue
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        out.append(ExperimentSummary(id=str(raw.get("id", d.name)), path=d, goal=str(raw.get("goal", "")),
                                     status=str(raw.get("status", "")), created=str(raw.get("created", "")),
                                     updated=str(raw.get("updated", "")),
                                     actions=len(raw.get("actions", []) or []), model=str(raw.get("model", ""))))
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

_STATUS_TEXT = {"running": "Running", "done": "Done", "unverified": "Finished, answer not verified",
                "stopped": "Stopped", "failed": "Failed"}


def elide(text: str, n: int = TITLE_CHARS) -> str:
    """The first line of ``text``, cut at a word boundary to about ``n`` characters (with an ellipsis).

    :func:`aerolume.ai.plaintext.brief` does the cut, so a number right after it
    is kept with its label ("Ra 1e4", not "Ra …").
    """
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return brief(line, n)


def plan_has_variables(plan: dict | None) -> bool:
    return bool(plan) and bool(plan.get("variables"))


def plan_controls_label(plan: dict | None) -> str:
    """"Controls (held fixed)" when the plan names what it varies, else "Settings".

    A plan without variables is a single run or a read of stored results: its
    "controls" are just the run's settings, and calling them "held fixed" next
    to a goal that compares two speeds read as a contradiction in the log.
    """
    return "Controls (held fixed)" if plan_has_variables(plan) else "Settings"


EMPTY_CELL = "—"


def _md_cell(v: Any) -> str:
    """One table cell: never empty, numbers at 4 significant figures, pipes escaped.

    An empty cell is written as an em dash because Markdown viewers (Qt's
    included) drop empty cells, which shifted every later value one column
    left: Best L/D appeared under "Uncertainty". Solver numbers arrive with 5
    or more significant figures (0.0076587); a log reads them at 4.
    """
    if isinstance(v, bool):
        text = "yes" if v else "no"
    elif isinstance(v, float):
        text = f"{v:.4g}" if math.isfinite(v) else ""
    elif v is None:
        text = ""
    else:
        text = str(v)
    text = text.replace("|", "\\|").replace("\n", " ").strip()
    return text or EMPTY_CELL


_CLAIM_WORDS = {"verified": "Verified", "derived": "Derived", "unsupported": "Not supported",
                "contradicted": "Contradicted"}


def _verification_md(claims: dict) -> list[str]:
    """The claims check of the final answer as a table: every checked statement, its verdict and evidence."""
    rows = [s for s in (claims or {}).get("statements", []) if isinstance(s, dict) and s.get("status") != "plain"]
    if not rows:
        return []
    out = ["### Verification", "", str(claims.get("summary", "")), "",
           "| Statement | Check | Evidence |", "|---|---|---|"]
    for s in rows:
        detail = "; ".join(s.get("problems") or []) or "; ".join(s.get("evidence") or [])
        out.append(f"| {_md_cell(elide(str(s.get('text', '')), 160))} | "
                   f"{_md_cell(_CLAIM_WORDS.get(str(s.get('status')), s.get('status')))} | "
                   f"{_md_cell(elide(detail, 220))} |")
    out.append("")
    return out


def render_markdown(rec: ExperimentRecord) -> str:
    """The experiment as a plain engineering note (goal, plan, actions, results, conclusions)."""
    title = elide(rec.goal) or rec.id
    lines = [f"# Experiment: {title}", ""]
    started = rec.created.replace("T", " ")[:16]
    status = _STATUS_TEXT.get(rec.status, rec.status)
    if rec.status != "running" and rec.wall_seconds:
        status += f" in {fmt_seconds(rec.wall_seconds)}"
    lines += ["| Item | Value |", "|---|---|", f"| Started | {started} |",
              f"| Assistant | {_md_cell(rec.model)} ({_md_cell(rec.provider)}) |", f"| Status | {status} |",
              f"| Actions | {len(rec.actions)} |", ""]
    lines += ["## Goal", "", rec.goal.strip(), ""]
    if rec.plan:
        lines += ["## Plan", ""]
        labels = (("objective", "Objective"), ("method", "Method"), ("variables", "Variables"),
                  ("controls", plan_controls_label(rec.plan)), ("quality", "Quality"),
                  ("expected_runs", "Expected runs"), ("expected_seconds", "Expected time"),
                  ("validity", "Validity limits"))
        for key, lab in labels:
            v = rec.plan.get(key)
            if v in (None, "", []):
                continue
            if key == "expected_seconds":
                try:
                    v = fmt_seconds(float(v))
                except (TypeError, ValueError):
                    pass
                if rec.plan.get("expected_time_basis"):
                    v = f"{v} ({rec.plan['expected_time_basis']})"
            if isinstance(v, list):
                v = "; ".join(str(x) for x in v)
            lines.append(f"- **{lab}:** {v}")
        lines.append("")
    if rec.actions:
        lines += ["## Actions", "", "| # | Time | Tool | Outcome | Duration |", "|---:|---|---|---|---:|"]
        for i, a in enumerate(rec.actions, 1):
            t = a.time[11:19] if len(a.time) >= 19 else a.time
            outcome = a.summary or a.status
            if a.status not in ("ok",):
                outcome = f"{a.status}: {outcome}" if a.summary else a.status
            lines.append(f"| {i} | {t} | `{a.tool}` | {_md_cell(outcome)} | "
                         f"{_md_cell(fmt_seconds(a.seconds) if a.seconds else '')} |")
        lines.append("")
    if rec.results:
        lines += ["## Results", "", "| Quantity | Value | Unit | Uncertainty | Source |", "|---|---:|---|---|---|"]
        for r in rec.results:
            lines.append(f"| {_md_cell(r.get('label'))} | {_md_cell(r.get('value'))} | {_md_cell(r.get('unit'))} | "
                         f"{_md_cell(r.get('uncertainty'))} | {_md_cell(r.get('source'))} |")
        lines.append("")
    if rec.conclusions or rec.findings:
        lines += ["## Conclusions", "", rec.conclusions.strip(), ""]
        if rec.findings:
            lines += ["| Finding | Value | Unit | Uncertainty | From |", "|---|---:|---|---|---|"]
            for f in rec.findings:
                lines.append(f"| {_md_cell(f.get('quantity'))} | {_md_cell(f.get('value'))} | "
                             f"{_md_cell(f.get('unit'))} | {_md_cell(f.get('uncertainty'))} | "
                             f"{_md_cell(f.get('simulation_id') or f.get('source'))} |")
            lines.append("")
    if rec.limitations:
        lines += ["### Validity and limitations", ""] + [f"- {x}" for x in rec.limitations] + [""]
    if rec.plots:
        lines += ["## Plots", ""] + [f"![{Path(p).stem}]({p})" for p in rec.plots] + [""]
    if rec.status != "running" and (rec.conclusions or rec.findings):
        lines += ["### Number check", ""]
        if rec.untraced:
            lines.append("These numbers in the conclusions do not appear in any tool result, tool input or "
                         "the goal (they may be derived, rounded differently or wrong): "
                         + ", ".join(f"`{n}`" for n in rec.untraced) + ".")
        else:
            lines.append("Every number in the conclusions appears in a tool result, a tool input or the goal.")
        lines.append("")
        for c in rec.checks:
            lines += [c, ""]
    lines += _verification_md(rec.claims)
    if len(rec.conversation) > 2:
        lines += ["## Conversation", ""]
        for m in rec.conversation:
            # "aerolume": a correction the loop sent on its own (no tool run yet, a design mismatch)
            who = {"user": "User", "assistant": "Assistant"}.get(str(m.get("role")), "Aerolume")
            text = str(m.get("text", "")).strip()
            if text:
                lines += [f"**{who}** ({str(m.get('time', ''))[11:19]})", "", text, ""]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["EXPERIMENTS_DIR", "PLOTS_DIR", "RECORD_FILE", "REPORT_FILE", "STATUSES", "TITLE_CHARS", "Action",
           "ExperimentLog", "ExperimentRecord", "ExperimentSummary", "GoalTarget", "SimFacts", "attribution_problems",
           "design_mismatches", "elide", "extract_numbers", "fmt_seconds", "goal_speeds", "goal_targets",
           "labelled_numbers",
           "list_experiments", "load_record", "numbers_in", "plan_controls_label", "plan_has_variables",
           "render_markdown", "slug", "untraced_numbers"]
