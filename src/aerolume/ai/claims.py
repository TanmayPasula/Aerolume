"""The claims checker: every statement of a final answer, checked against what the tools returned.

The assistant must not say it did something it did not do, and every number it
gives must come from a tool result (DESIGN.md 9). Two live failures with a local
8-billion-parameter model showed what a prompt alone does not prevent:

* asked to delete a simulation, it answered "The simulation ... has been deleted
  from the project" without calling any tool (it had no delete tool at all);
* it wrote "the stored 25 m/s polar's best L/D of 26.47" when that run's value
  was 29.52: 26.47 was a value of the *other* run, so a plain "does this number
  appear in some tool result" check passed it.

:func:`check_answer` cuts an answer into statements (sentences, list items,
table rows) and gives each a verdict:

``verified``
    every action it claims matches a successful tool call of that kind (on the
    design it names), and every number matches a value of the tool results, the
    tool inputs or the user's goal at the precision it is written with -- and,
    when the statement names a simulation (by its speed or design), a value of
    *that* simulation;
``derived``
    a number is not in any result but is a difference, a change in percent or a
    ratio of two result values; the formula is shown, so a reader sees how it
    was obtained;
``unsupported``
    a number matches nothing the experiment produced;
``contradicted``
    the evidence says otherwise: an action no tool call performed, a number that
    belongs to another simulation (with the value it should have been), a claimed
    run of a design that was not run;
``plain``
    nothing checkable (prose, a heading).

The agent sends one correction request listing the unsupported and contradicted
statements (:meth:`ClaimReport.correction_note`), checks the corrected answer
again, and hands the final report to the panel, which lists every checked
statement with its verdict and evidence, and to the experiment log. No Qt here.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import asdict, dataclass, field

from .experiment import _CLAUSE_SPLIT, _SPEEDS, _floats, extract_numbers
from .plaintext import brief

# ---------------------------------------------------------------------------
# claimed actions ("I deleted", "has been deleted", "I ran", "I created", "I changed")
# ---------------------------------------------------------------------------

# First person ("I deleted", "I have run", "I went ahead and removed") for every kind, and the passive for
# deletions ("has been deleted from the project", the qwen3:8b answer that deleted nothing). The passive of
# runs and creations is left out: "the 25 m/s polar has been run before" describes stored work.
_ADV = r"(?:\s+(?:just|now|already|successfully|also))?"
_AHEAD = r"(?:\s+(?:gone|went)\s+ahead\s+and)?"
_FIRST = r"\b(?:I|we)(?:['’]ve|\s+have|\s+had)?" + _ADV + _AHEAD + _ADV + r"\s+"
# A statement that starts with the verb, subject left out, as chat answers and check-list lines do
# ("Deleted “Polar · Re 3.0×10⁵” as requested.", "✅ Ran a polar at Re 10⁶"): at the start of a line or
# after a sentence end, behind an optional list mark, check mark or bold marker. Only verbs that are
# never adjectives there: "Added drag from the mirrors" is a quantity, so "Added" is not one of them.
_STATEMENT_START = r"(?:^|(?<=[.!?]\s)|(?<=[.!?]\s\s))[ \t]*(?:[✅✔☑✓]️?[ \t]*|[-*•][ \t]+|\d+[.)][ \t]+)?(?:\*\*)?"
# the things a deletion is of: "Removed the spoiler" is a geometry variant (a derived design), not a deletion
_ITEM = (r"(?:(?:the|both|all|its|this|that|these|those|\d+)\s+)?(?:(?:old|duplicate|stored|previous|unused|failed|"
         r"empty|cancelled|requested)\s+)?(?:simulations?|designs?|runs?|polars?|results?|cases?|sweeps?|items?|"
         r"experiments?|it\b|them\b|[“\"‘'])")
_RESULT_NOUN = r"(?:simulations?|designs?|runs?|polars?|results?|cases?|sweeps?)"
_CLAIM_PATTERNS = {
    "delete": re.compile(
        _FIRST + r"(?:deleted|removed)\b"
        + r"|\b(?:has|have)" + _ADV + r"\s+been" + _ADV + r"\s+(?:deleted|removed)\b"
        + r"|\b(?:was|were|is|are)" + _ADV + r"\s+(?:deleted|removed)\b"
        + r"|" + _STATEMENT_START + r"(?:deleted|removed)(?:\*\*)?\s+" + _ITEM
        # a headline: "Simulation “Polar · Re 3.0×10⁵” deleted.", "“Polar · Re 3.0×10⁵” removed."
        + r"|(?:\b" + _RESULT_NOUN + r"|[”\"’])\s+(?:deleted|removed)\b(?!\s+(?:by|when|if|once|after))"
        + r"|\band\s+(?:then\s+)?(?:also\s+)?(?:deleted|removed)\s+" + _ITEM,
        re.IGNORECASE | re.MULTILINE),
    "run": re.compile(
        _FIRST + r"(?:ran|run|re-?ran|executed|performed|simulated)\b(?!\s+(?:out|into)\b)"
        + r"|" + _FIRST + r"completed\s+(?:the\s+|a\s+|an\s+|all\s+|both\s+)?(?:[\w/·-]+\s+){0,3}?"
        + r"(?:simulations?|runs?|polars?|sweeps?|grid\s+stud(?:y|ies)|CFD)\b"
        + r"|" + _STATEMENT_START + r"(?:ran|re-?ran|executed)\b(?!\s+(?:out|into)\b)"
        # "Created a NACA 4412 airfoil and ran its polar": not "the solver converged and ran for 30 s"
        + r"|\band\s+(?:then\s+)?(?:also\s+)?(?:ran|re-?ran)\s+(?:it|its|the|a|an|them|both|all)\b",
        re.IGNORECASE | re.MULTILINE),
    "create": re.compile(
        _FIRST + r"(?:created|added)\b"
        + r"|" + _STATEMENT_START + r"created\b"
        + r"|\band\s+(?:then\s+)?(?:also\s+)?created\s+(?:a|an|the|it|two|three|\d+)\b",
        re.IGNORECASE | re.MULTILINE),
    # nothing the assistant does edits a simulation: a suggestion is applied by the user's click ("applied"
    # is left out: "I applied the polar analysis" names a method, not an edit). The passive counts when it
    # says the change was made ("has been changed to"), not when it describes a run's set-up ("was set to").
    "edit": re.compile(
        _FIRST + r"(?:changed|updated|modified|switched)\b|" + _FIRST
        + r"set\s+(?:the\s+|its\s+)?[\w\s/-]{1,40}?\s+to\b"
        + r"|\b(?:has|have)" + _ADV + r"\s+been" + _ADV + r"\s+(?:changed|updated|modified|switched)\b"
        + r"|\b(?:is|are)\s+now\s+set\s+to\b",
        re.IGNORECASE | re.MULTILINE),
}
#: tool kinds whose successful call makes a claim of that kind true
CLAIM_KINDS = {"delete": ("delete", "code"), "run": ("run", "code"), "create": ("create", "code"),
               "edit": ("edit", "code")}
_NEGATED = re.compile(r"\b(?:nothing|not|never|no|none|cannot|neither|nor|if|would|could|should|unless|"
                      r"whether|can|may|might|suggest\w*|propos\w*|recommend\w*)\b|n['’]t\b", re.IGNORECASE)
# a present-tense passive after "when / once / with ..." describes a configuration, not an action: "drag falls
# by 8 % when the spoiler is removed" (a geometry study). Past tense still counts: "after it was deleted".
_PRESENT_PASSIVE = re.compile(r"(?:is|are)\b", re.IGNORECASE)
_HYPOTHETICAL = re.compile(r"\b(?:when|whenever|once|after|before|with|without|where)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")          # not the point inside "3.0×10⁵"
_CLAUSE_START = re.compile(r"[.!?;:](?=\s|$)|\n|,\s")


def claimed_actions(answer: str) -> list[tuple[str, str]]:
    """(kind, sentence) for every action an answer says it performed: "delete", "run", "create" or "edit".

    A claim inside a negated, conditional or proposing clause ("nothing was
    deleted", "if I deleted it", "I suggest you set ...") is not a claim. The
    sentence is returned for the correction request and the log.
    """
    text = answer or ""
    out: list[tuple[str, str]] = []
    for kind, rx in _CLAIM_PATTERNS.items():
        for m in rx.finditer(text):
            starts = [x.end() for x in _CLAUSE_START.finditer(text, 0, m.start())]
            clause_start = starts[-1] if starts else 0
            before_claim = text[clause_start:m.start()]
            if _NEGATED.search(before_claim):
                continue
            after = _SENTENCE_END.search(text, m.end())
            if _PRESENT_PASSIVE.match(m.group(0)) and (
                    _HYPOTHETICAL.search(before_claim)
                    # a rule: "results are deleted when their design is deleted"
                    or _HYPOTHETICAL.search(text[m.end():after.start() if after else len(text)])):
                continue
            before = [x.end() for x in _SENTENCE_END.finditer(text, 0, m.start())]
            s0, s1 = (before[-1] if before else 0), (after.end() if after else len(text))
            sentence = " ".join(text[s0:s1].split()).strip(" *")
            if len(sentence) > 160:
                sentence = sentence[:159].rstrip() + "…"
            if (kind, sentence) not in out:
                out.append((kind, sentence))
    return out


# ---------------------------------------------------------------------------
# what the experiment produced
# ---------------------------------------------------------------------------

@dataclass
class NumberSource:
    """One number a tool returned (or the user or a tool input gave), and where it came from."""
    value: float
    origin: str                 # "result" | "input" | "goal" | "plan" (the model's own plan: about the plan only)
                                # | "validity" (the solvers' validity record of the system prompt)
    label: str                  # "ld_max in “Wing polar · 25 m/s” (get_results)"
    sim_id: str = ""
    key: str = ""


@dataclass
class SimRecord:
    """What the checker knows about one simulation the experiment ran or read.

    ``values`` are all numbers of its result (tables included); ``headline`` its
    summary metrics and coefficients by name. A number given for this simulation
    must be one of the values; a number that misses it is called misattributed
    only when it is a headline value of another simulation.
    """
    sim_id: str
    label: str                  # "“Wing polar · 25 m/s”"
    design: str
    speed: float | None
    values: list[float] = field(default_factory=list)
    headline: dict[str, float] = field(default_factory=dict)
    #: the physical models the run used (tools.physics_models: heat_transfer, buoyancy, turbulence, domain);
    #: empty when unknown (polars, wings, older callers): nothing is then checked against it
    physics: dict = field(default_factory=dict)


@dataclass
class ActionRecord:
    """One tool call of the experiment, as the action check sees it."""
    tool: str
    kind: str                   # the tool kind: read, create, run, code, delete, record, suggest
    status: str                 # "ok" | "error" | ...
    designs: list[str] = field(default_factory=list)     # names of the designs it ran, created or deleted
    reused: bool = False        # a run call that returned a stored result (nothing new ran)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

STATUSES = ("verified", "derived", "unsupported", "contradicted", "plain")
_RANK = {"plain": 0, "verified": 1, "derived": 2, "unsupported": 3, "contradicted": 4}
#: how each verdict reads in the panel and the log
STATUS_WORDS = {"verified": "Verified", "derived": "Derived", "unsupported": "Not supported",
                "contradicted": "Contradicted", "plain": "Not checkable"}


@dataclass
class NumberCheck:
    token: str
    value: float
    status: str                 # verified | derived | unsupported | contradicted
    evidence: str = ""          # where it came from, or the formula
    correction: str = ""        # the value it should have been (contradicted)


@dataclass
class Statement:
    text: str
    status: str
    numbers: list[NumberCheck] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def checked(self) -> bool:
        return self.status != "plain"


@dataclass
class ClaimReport:
    """The verdict on every statement of one answer (see the module docstring)."""
    statements: list[Statement] = field(default_factory=list)

    @property
    def checked(self) -> list[Statement]:
        return [s for s in self.statements if s.checked]

    @property
    def problems(self) -> list[Statement]:
        return [s for s in self.statements if s.status in ("unsupported", "contradicted")]

    @property
    def ok(self) -> bool:
        return not self.problems

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in STATUSES}
        for s in self.statements:
            out[s.status] += 1
        return out

    def summary_line(self) -> str:
        """"5 statements checked: 4 verified, 1 derived" / "...: 1 not supported, 1 contradicted"."""
        c = self.counts()
        n = len(self.checked)
        if not n:
            return "No checkable statements (no actions or numbers)."
        bits = [f"{c[k]} {STATUS_WORDS[k].lower()}" for k in ("verified", "derived", "unsupported", "contradicted")
                if c[k]]
        return f"{n} statement{'s' if n != 1 else ''} checked: " + ", ".join(bits)

    def correction_note(self) -> str:
        """The problems as a list for the model's one correction request."""
        parts = []
        for s in self.problems:
            why = "; ".join(s.problems) or STATUS_WORDS[s.status].lower()
            parts.append(f"“{_short(s.text, 140)}” ({why})")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {"statements": [asdict(s) for s in self.statements], "summary": self.summary_line(),
                "counts": self.counts()}

    @classmethod
    def from_dict(cls, d: dict | None) -> ClaimReport:
        out = cls()
        for s in (d or {}).get("statements", []) or []:
            if not isinstance(s, dict):
                continue
            nums = [NumberCheck(**{k: v for k, v in n.items() if k in NumberCheck.__dataclass_fields__})
                    for n in s.get("numbers", []) if isinstance(n, dict)]
            out.statements.append(Statement(text=str(s.get("text", "")), status=str(s.get("status", "plain")),
                                            numbers=nums, evidence=list(s.get("evidence", [])),
                                            problems=list(s.get("problems", []))))
        return out


def _short(text: str, n: int) -> str:
    return brief(text, n)


# ---------------------------------------------------------------------------
# statements
# ---------------------------------------------------------------------------

_TABLE_RULE = re.compile(r"^\|?\s*:?-{2,}")
_LIST_MARK = re.compile(r"^(?:[-*+•]|\d+[.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(“\"*_])")


def split_statements(text: str) -> list[str]:
    """An answer cut into statements: sentences of paragraphs, list items, table rows (headings dropped)."""
    out: list[str] = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or _TABLE_RULE.match(s):
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            row = " | ".join(c for c in cells if c)
            if row:
                out.append(row)
            continue
        s = _LIST_MARK.sub("", s)
        out.extend(p.strip() for p in _SENTENCE_SPLIT.split(s) if p.strip())
    return out


_UNIT_AFTER = re.compile(r"^\s*(?:°C?|deg\b|degrees?\b|m/s\b|%|K\b|N\b|W\b|Pa\b|mm\b|m\b|s\b|cells?\b|kg\b)")
# "step 3", "run 2", "figure 4", "NACA 2412": a label, not a value. "of" is deliberately not in the list:
# "the best L/D of 26.47" is the very claim the checker exists for (the live failure had bold markers
# around the number, which is the only reason a list containing "of" ever let it be checked)
_ORDINAL_BEFORE = re.compile(r"(?:\b(?:step|steps|run|runs|action|actions|grid|grids|level|levels|point|points|"
                             r"figure|section|table|finding|item|version|NACA)\s*#?\s*)$", re.IGNORECASE)
# the total of a count: "18 of 19 angles", "step 2 of the 3 runs"
_OF_COUNT_BEFORE = re.compile(r"\b\d+\s+of\s+(?:the\s+)?$", re.IGNORECASE)
_PERCENT_AFTER = re.compile(r"^\s*(?:%|per\s?cent\b)", re.IGNORECASE)
_PLUS_MINUS_BEFORE = re.compile(r"(?:±|\+/-|\+-)\s*$")
# what a "± x" may be: a standard error, a GCI band, an angle resolution or step, a stated uncertainty. In a live
# qwen3:8b answer "1.18% ± 0.05%" was "verified" by the speed_m_s 0.05 of an unrelated run
_UNCERTAINTY_SOURCE = re.compile(r"stderr|std|uncert|gci|error|resolution|step|band|confidence|sigma|toleran|"
                                 r"spread|±", re.IGNORECASE)
# a statement about the plan may quote the plan's numbers ("the plan expected 30 s"); nothing else may
_ABOUT_PLAN = re.compile(r"\b(?:plan\w*|expect\w*|estimat\w*|intend\w*|aim\w*)\b", re.IGNORECASE)
#: a statement about what a solver is validated for: only there may a number of the validity record count
#: ("validated against NACA 0012/2412/4412 data"), never as a result ("the best L/D is 4412")
_ABOUT_VALIDITY = re.compile(r"\bvalid\w*|\breliab\w*|\bwind[- ]tunnel\b|\breference solutions?\b",
                             re.IGNORECASE)


def _checkable(tok: str, value: float, text: str, end: int, start: int) -> bool:
    """Whether a written number is a claim: not a small count, a list number, a year or a NACA code."""
    a = abs(value)
    integer = float(value).is_integer()
    unit = bool(_UNIT_AFTER.match(text[end:end + 12]))
    if integer and 1900 <= a <= 2100 and not unit:
        return False
    before = text[max(0, start - 16):start]
    if _ORDINAL_BEFORE.search(before) or (integer and _OF_COUNT_BEFORE.search(before)):
        return False
    return not (integer and a <= 10 and not unit)


def _match(v: float, half: float, s: float, *, percent: bool = False, rel_tol: float = 1e-9) -> bool:
    """A written value matches a source value at the precision it is written with.

    "26.5" matches 26.47 (written to 0.05) but not 26.40; ``rel_tol`` only absorbs
    float noise. A 0.5 % tolerance used to let "31.2" pass for any value from 31.05
    to 31.36, and a per-angle table holds enough numbers for that to happen by
    chance. ``percent`` (a "%" follows the number) also reads a fraction as a
    percentage, "12 %" for 0.12 -- never otherwise: "25 m/s" matched a mean chord
    of 0.245 m that way.
    """
    a, b = abs(v), abs(s)
    if abs(b - a) <= max(half * 1.0001, rel_tol * max(a, b)):
        return True
    return percent and bool(a) and abs(b * 100.0 - a) <= half * 1.0001


# ---------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------

def _speeds_named(clause: str) -> list[float]:
    return [v for m in _SPEEDS.finditer(clause) for v in _floats(m.group(1))]


def _missing_speeds(clause: str, sims: list[SimRecord]) -> list[float]:
    """Speeds a clause names that none of the experiment's simulations ran at ("the 25 m/s run" with no 25 m/s run).

    Only when the experiment has simulations with speeds at all: a goal-only
    answer ("at 20 m/s you would expect...") has nothing to attribute to.
    """
    known = [s.speed for s in sims if s.speed is not None]
    if not known:
        return []
    return [v for v in _speeds_named(clause) if not any(abs(k - v) <= 1e-6 + 0.005 * v for k in known)]


_UNIT_KEYS = (
    (re.compile(r"^\s*(?:°(?!C)|deg\b|degrees?\b)"), ("alpha", "deg", "angle", "yaw")),
    (re.compile(r"^\s*m/s\b"), ("speed", "m_s", "velocity")),
    (re.compile(r"^\s*(?:%|per\s?cent\b)", re.IGNORECASE), ("pct", "percent", "%")),
)


def _evidence_rank(src: NumberSource, v: float, after: str) -> tuple:
    """Order matching sources: results first, then a label that fits the written unit, then the closest value."""
    origin = {"result": 0, "input": 1, "goal": 2, "plan": 3, "validity": 4}.get(src.origin, 5)
    unit_miss = 0
    for rx, words in _UNIT_KEYS:
        if rx.match(after):
            label = f"{src.key} {src.label}".lower()
            unit_miss = 0 if any(w in label for w in words) else 1
            break
    return (origin, unit_miss, abs(abs(src.value) - abs(v)) / max(abs(v), 1e-300))


def _sims_named(clause: str, sims: list[SimRecord]) -> list[SimRecord]:
    """The simulations a clause is about: those of the speed(s) and design(s) it names (none: no attribution)."""
    speeds = _speeds_named(clause)
    low = clause.lower()
    designs = {s.design for s in sims if s.design and s.design.lower() in low}
    if not speeds and not designs:
        return []
    named = [s for s in sims
             if (not speeds or (s.speed is not None and any(abs(s.speed - v) <= 1e-6 + 0.005 * v for v in speeds)))
             and (not designs or s.design in designs)]
    return [] if len(named) == len(sims) else named


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _derivation(v: float, half: float, sims: list[SimRecord], *, percent: bool = False) -> str:
    """A formula over two headline values that gives ``v``, or "".

    Only pairs that mean something are tried: the same quantity of two
    simulations (a difference, a change in percent, a ratio: "best L/D fell
    10 % from 25 to 15 m/s") and two quantities of one simulation (a ratio: Cl
    over Cd). A cross-quantity, cross-run difference would match almost any
    number by chance. The match is tight: the written precision, or 0.1 %.
    ``percent`` (a "%" was written) tries only the change in percent, and a
    number without one never matches a percentage: "±1.5 %" passed as the
    ratio 0.5101 / 0.3446 = 1.48 of two lift coefficients (live qwen3:8b run).
    """
    target = abs(v)
    tol = max(half * 1.0001, 0.001 * target)

    def num(x: object) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x != 0

    for sa, sb in itertools.permutations(sims, 2):
        for k, a in sa.headline.items():
            b = sb.headline.get(k)
            if not (num(a) and num(b)):
                continue
            a, b = float(a), float(b)
            pct = 100.0 * (a - b) / b
            if percent:
                if abs(abs(pct) - target) <= tol:
                    return (f"({_fmt(a)} − {_fmt(b)}) / {_fmt(b)} = {pct:+.3g} % ({k} of {sa.label} against "
                            f"{sb.label})")
                continue
            if abs(abs(a - b) - target) <= tol:
                return f"{_fmt(a)} − {_fmt(b)} = {_fmt(a - b)} ({k} of {sa.label} minus that of {sb.label})"
            if abs(a / b - target) <= tol:
                return f"{_fmt(a)} / {_fmt(b)} = {_fmt(a / b)} ({k} of {sa.label} over {sb.label})"
    if percent:
        return ""
    for s in sims:
        for (ka, a), (kb, b) in itertools.permutations(s.headline.items(), 2):
            if num(a) and num(b) and abs(abs(float(a) / float(b)) - target) <= tol:
                return f"{_fmt(float(a))} / {_fmt(float(b))} = {_fmt(float(a) / float(b))} ({ka} over {kb} of {s.label})"
    return ""


def _correction(v: float, half: float, named: list[SimRecord], owner: SimRecord) -> str:
    """"ld_max of “Wing polar · 25 m/s” is 29.52": the named run's value of the key the number came from."""
    for k, x in owner.headline.items():
        if isinstance(x, (int, float)) and (_match(v, half, float(x)) or _match(v, half, float(x), percent=True)):
            for s in named:
                y = s.headline.get(k)
                if isinstance(y, (int, float)) and math.isfinite(y):
                    return f"{k} of {s.label} is {_fmt(float(y))}"
    return ""


# a number followed by its own speed: "26.47 at 25 m/s", "26.47 for V = 25 m/s", "26.47 (25 m/s)"; at most two
# words in between and no other number. A range ("at 15–25 m/s", "at 15 to 25 m/s") never matches: the speed
# must be followed by its unit
_SPEED_AFTER = re.compile(r"\s*(?:\(\s*|(?:[^\W\d]+\s+){0,2}?(?:at|for|@)\s+(?:(?:V|U|u|v)\s*=\s*)?)"
                          r"(\d+(?:\.\d+)?)\s*m/s\b")
# a speed followed by its number: "at 25 m/s it is 26.47", "25 m/s: 26.47", "(25 m/s) 26.47"
_SPEED_BEFORE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*m/s\b\)?\s*[:=]?\s*(?:(?:[^\W\d]+|L/D\w*|C\w+)\s+){0,4}?"
                           r"(?:[:=]\s*)?$")


def _bound_speed(clause: str, start: int, end: int) -> float | None:
    """The speed a number in ``clause`` is given for when it is written right next to it, else None.

    The clause-level attribution names every run a clause mentions; a clause
    that mentions two ("29.52 at 15 m/s and 26.47 at 25 m/s") attributes
    nothing, so a swapped pair passed as verified. A speed is bound to the
    number only across a few words with no other number between them; a range
    ("between 15 and 25 m/s") binds nothing.
    """
    m = _SPEED_AFTER.match(clause, end)
    if m is not None and m.end() - end <= 40:
        return float(m.group(1))
    head = clause[max(0, start - 48):start]
    b = _SPEED_BEFORE.search(head)
    if b is not None and not _NUMBER_TOKENS.search(head[b.end(1):]):
        # a list of speeds ("15 and 25 m/s") is a range, not one run
        lead = head[:b.start(1)].rstrip()
        if not re.search(r"\d\s*(?:,|and|or|to|vs\.?|versus|-|–|−)$", lead):
            return float(b.group(1))
    return None


def _check_numbers(text: str, sources: list[NumberSource], sims: list[SimRecord]) -> list[NumberCheck]:
    out: list[NumberCheck] = []
    clauses = [c for c in _CLAUSE_SPLIT.split(text) if c and c.strip()] or [text]
    # a statement that names one run ("At 25 m/s, the best L/D is 29.52") is about that run in every
    # clause; one that names several leaves each clause to the run it names itself
    whole = _sims_named(text, sims)
    one_run = whole if len({s.sim_id for s in whole}) == 1 else []
    about_validity = bool(_ABOUT_VALIDITY.search(text))
    for clause in clauses:
        named = _sims_named(clause, sims) or one_run
        speed_tokens = {f"{v:g}" for m in _SPEEDS.finditer(clause) for v in _floats(m.group(1))}
        # "the 25 m/s run reached 31.2" when the experiment has no 25 m/s run: nothing of it can be verified
        absent = [] if named else (_missing_speeds(clause, sims) or _missing_speeds(text, sims))
        # a quoted name ("“Polar · Re 3.0×10⁵”") names a run; its digits are not a claim
        quoted = [(q.start(), q.end()) for q in _QUOTED.finditer(clause)]
        for m in _NUMBER_TOKENS.finditer(clause):
            if any(a <= m.start() < b for a, b in quoted):
                continue
            tok = m.group(0).strip()
            parsed = extract_numbers(tok)
            if not parsed:
                continue
            _t, v, half = parsed[0]
            if not _checkable(tok, v, clause, m.end(), m.start()):
                continue
            after = clause[m.end():m.end() + 12]
            pct = bool(_PERCENT_AFTER.match(after))
            # a speed that names the run ("at 25 m/s") is matched as the run's condition, below
            about_plan = bool(_ABOUT_PLAN.search(clause))
            hits = [s for s in sources if _match(v, half, s.value, percent=pct)
                    and (s.origin != "plan" or about_plan) and (s.origin != "validity" or about_validity)]
            # an uncertainty ("± 1.5 %") must come from a result (a standard error, a GCI, half the angle
            # step): arithmetic on two results is never one, and neither is any other number that happens to
            # have the same digits
            uncertainty = bool(_PLUS_MINUS_BEFORE.search(clause[max(0, m.start() - 6):m.start()]))
            if uncertainty:
                matched = hits
                hits = [s for s in hits if _UNCERTAINTY_SOURCE.search(f"{s.key} {s.label}")]
                if matched and not hits:
                    out.append(NumberCheck(tok, v, "unsupported", "an uncertainty that matches only "
                                           f"{min(matched, key=lambda s: _evidence_rank(s, v, after)).label}, "
                                           "not a standard error, grid uncertainty or resolution"))
                    continue
            is_speed = tok.lstrip("+-−") in speed_tokens
            # the run this number is about: the clause's, unless the number is bound to a speed of its own
            # ("29.52 at 15 m/s and 26.47 at 25 m/s" names both runs in one clause, each value for one)
            mine, gone = named, absent
            bound = None if is_speed else _bound_speed(clause, m.start(), m.end())
            if bound is not None:
                at_speed = [s for s in sims if s.speed is not None and abs(s.speed - bound) <= 1e-6 + 0.005 * bound]
                if at_speed and len(at_speed) < len(sims):
                    mine, gone = at_speed, []
                elif not at_speed and any(s.speed is not None for s in sims):
                    mine, gone = [], [bound]
            if mine and not is_speed and not uncertainty:
                own = [s for s in mine if any(_match(v, half, x, percent=pct) for x in s.values)]
                if own:
                    out.append(NumberCheck(tok, v, "verified", f"a value of {own[0].label}"))
                    continue
                others = [s for s in sims if s not in mine and any(_match(v, half, float(x), percent=pct)
                                                                   for x in s.headline.values()
                                                                   if isinstance(x, (int, float)))]
                if others:
                    fix = _correction(v, half, mine, others[0])
                    out.append(NumberCheck(tok, v, "contradicted", f"a value of {others[0].label}, not of "
                                           f"{' / '.join(s.label for s in mine[:2])}", fix))
                    continue
            if gone and not is_speed:
                given = [s for s in hits if s.origin in ("goal", "input")]
                if given:                   # a number the user (or a tool input) gave is quoted, not measured
                    best = min(given, key=lambda s: _evidence_rank(s, v, after))
                    out.append(NumberCheck(tok, v, "verified", best.label))
                    continue
                speeds = " and ".join(f"{x:g} m/s" for x in gone)
                out.append(NumberCheck(tok, v, "unsupported", f"given for a run at {speeds}, which this experiment "
                                                              "did not run or read"))
                continue
            if hits:
                best = min(hits, key=lambda s: _evidence_rank(s, v, after))
                out.append(NumberCheck(tok, v, "verified", best.label))
                continue
            formula = "" if uncertainty else _derivation(v, half, sims, percent=pct)
            if formula:
                out.append(NumberCheck(tok, v, "derived", formula))
                continue
            out.append(NumberCheck(tok, v, "unsupported", "in no tool result, tool input or the goal"))
    return out


_QUOTED = re.compile(r"“[^”]{1,120}”|\"[^\"\n]{1,120}\"|‘[^’]{1,120}’")
_NACA_NAMED = re.compile(r"(?<![/\w])NACA[\s-]*(\d{4,5})(?![/\w])", re.IGNORECASE)

# the number tokens of experiment.extract_numbers, with their positions
_NUMBER_TOKENS = re.compile(r"(?<![\w.])[-−+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:\s*(?:[eE][-−+]?\d+|[×x]\s*10\s*"
                            r"(?:\^|\*\*)?\s*[-−⁻+]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?(?![\w⁰¹²³⁴⁵⁶⁷⁸⁹])")


def _check_actions(text: str, actions: list[ActionRecord], designs: list[str]) -> tuple[list[str], list[str]]:
    """(evidence, problems) for the actions a statement claims."""
    evidence: list[str] = []
    problems: list[str] = []
    low = text.lower()
    named = [d for d in sorted(designs, key=len, reverse=True) if d and d.lower() in low]
    for kind, _sentence in claimed_actions(text):
        done = [a for a in actions if a.status == "ok" and a.kind in CLAIM_KINDS[kind]]
        if not done:
            what = {"delete": "deleted anything", "run": "ran an analysis", "create": "created anything",
                    "edit": "changed any setting (suggestions apply only when the user clicks Apply)"}[kind]
            problems.append(f"no tool call in this experiment {what}")
            continue
        if named and kind != "edit":
            hit = [a for a in done if a.kind == "code" or set(a.designs) & set(named)]
            if not hit:
                did = sorted({d for a in done for d in a.designs})
                problems.append(f"no {kind} call was of {', '.join(named)}"
                                + (f" (the {kind} calls were of {', '.join(did)})" if did else ""))
                continue
            done = hit
        codes = [m.group(1) for m in _NACA_NAMED.finditer(text)]
        if codes and kind in ("run", "create"):
            # "I ran NACA 4412" when only NACA 2412 ran: the section is named, the project may not hold it
            did = " ".join(d for a in done for d in a.designs)
            missing = [c for c in codes if c not in did and not any(a.kind == "code" for a in done)]
            if missing:
                problems.append(f"no {kind} call was of NACA {', NACA '.join(missing)}"
                                + (f" (they were of {', '.join(sorted({d for a in done for d in a.designs}))})"
                                   if did else ""))
                continue
        if kind == "run" and done and all(a.reused for a in done):
            evidence.append(f"{done[0].tool}: a stored result was reused (nothing new ran)")
        else:
            evidence.append(f"{done[-1].tool} succeeded" + (f" on {', '.join(done[-1].designs)}"
                                                           if done[-1].designs else ""))
    return evidence, problems


# ---------------------------------------------------------------------------
# "validated" claims
# ---------------------------------------------------------------------------

_VALIDATED_WORD = re.compile(r"\bvalidat(?:ed|es|ion)\b|\baccurate\s+(?:to|within)\b", re.IGNORECASE)
_VALIDATION_NEGATED = re.compile(r"\b(?:not|never|no|none|nothing|un|nor|neither|cannot|isn['’]t|aren['’]t|"
                                 r"without)\b|"
                                 r"n['’]t\b|\bunvalidated\b", re.IGNORECASE)
_CLAIM_CLAUSES = re.compile(r"[;:]|,\s+(?:but|while|whereas|though|although)\b|\s(?:but|while|whereas)\s",
                            re.IGNORECASE)
#: a clause that is only a heading: "Validated", "- **Not validated**", "Validated quantities", "Verified" (its
#: list follows the colon)
_LABEL_CLAUSE = re.compile(r"^[\W_]*(?:(?:not|un)\s*)?(?:validated|verified)"
                           r"(?:\s+(?:quantit(?:y|ies)|values?|results?|items?|for))?[\W_]*$", re.IGNORECASE)


def _claim_clauses(text: str) -> list[str]:
    """The clauses a "validated" claim is judged in: :data:`_CLAIM_CLAUSES` pieces, with a heading such as
    "Validated:" turned round into "<the list after its colon> Validated".

    The checks below read the quantities named before the word in its own clause ("L/D is validated"). The
    tools' own confidence note is written as "Validated: cl, cd, cm, L/D in attached flow. Not validated:
    Cl_max, ...", and models copy that form; split at the colon, the heading named nothing and the list
    held no claim word, so a correct "Validated: cl, cd" was called unsupported and a wrong "Validated:
    Cl_max" escaped the not_validated check (live qwen3:8b test, 1 Nov 2026).
    """
    parts = [p for p in _CLAIM_CLAUSES.split(text or "") if p]
    out: list[str] = []
    i = 0
    while i < len(parts):
        head = parts[i]
        if _LABEL_CLAUSE.match(head) and i + 1 < len(parts) and parts[i + 1].strip():
            word = re.sub(r"^[\W_]+|[\W_]+$", "", head)
            out.append(f"{parts[i + 1].strip().rstrip('.!')} {word}")
            i += 2
            continue
        out.append(head)
        i += 1
    return out


#: how a not-validated quantity is named in an answer (the result's own words are tried too)
_QUANTITY_WORDS = {
    "l/d": r"\bL\s*/\s*D\b|lift[- ]to[- ]drag|\bld_max\b|glide ratio",
    "cd": r"\bC[Dd]\b|drag coefficient",
    "absolute cd": r"\bC[Dd]\b|drag coefficient",
    "profile drag": r"profile drag",
    "cl_max": r"\bC[Ll][ _,.]?max\b|maximum lift",
    "cl": r"\bC[Ll]\b|lift coefficient",
    "oswald_e_parabolic_fit": r"oswald",
    "values near and past stall": r"\bstall",
    "nusselt number (absolute value)": r"nusselt|\bNu\b",
    "nusselt number": r"nusselt|\bNu\b",
    "heat rate": r"heat (?:transfer )?rate",
    "cl_alpha": r"lift[- ](?:curve )?slope|\bC[Ll][_ ,]?(?:α|alpha)\b",
    "e (span efficiency)": r"span efficiency",
    "cdi": r"induced drag|\bC[Dd]i\b",
    "l/d in attached flow": r"\bL\s*/\s*D\b|lift[- ]to[- ]drag",
}


def _names_quantity(clause_before: str, item: str) -> bool:
    words = _QUANTITY_WORDS.get(item.strip().lower())
    if words and re.search(words, clause_before, re.IGNORECASE):
        return True
    # the result's own words, as whole words: "CD" is not in "CDi" (which is validated)
    return bool(re.search(rf"(?<!\w){re.escape(item.strip())}(?!\w)", clause_before, re.IGNORECASE))


def _validation_problems(text: str, not_validated: list[str]) -> list[str]:
    """Quantities a statement calls validated that a result of the experiment lists as not validated.

    qwen3:8b ended a correct wing answer with "Best L/D is validated against
    reference solutions within 2-2.5%": every number in it traced to a result
    (2.5 is in the validity sentence, which says the opposite: the lift slope and
    span efficiency are within 2-2.5 %, L/D is not validated). A number check
    cannot see that; this does. Only the quantities named before the word in its
    own clause count ("L/D is validated"; not "validated lift slope, so the L/D
    trend"), and a negated clause ("L/D is not validated") is not a claim.
    """
    out: list[str] = []
    if not not_validated:
        return out
    for clause in _claim_clauses(text):
        if not clause:
            continue
        m = _VALIDATED_WORD.search(clause)
        if m is None or _VALIDATION_NEGATED.search(clause[:m.end()]):
            continue
        before = clause[:m.start()]
        for item in not_validated:
            if _names_quantity(before, item) and item not in out:
                out.append(item)
    return [f"it calls {q} validated, but the result lists {q} under not_validated" for q in out]


def _unvalidation_problems(text: str, validated: list[str]) -> list[str]:
    """Quantities a statement calls not validated that a result lists as validated, with its deviation.

    The converse of :func:`_validation_problems`, for the validated reference
    set-ups (``"Nusselt number: validated set-up ...; this run 2.2509 against
    2.238, +0.57 %"``): qwen3:8b wrote "The Nusselt number's absolute value and
    heat rate are not validated" right after matching de Vahl Davis to 0.575 %
    (critic round 3.3). ``validated`` holds those entries; the quantity is the
    text before the colon, named before the word in its own negated clause.
    """
    out: list[str] = []
    for clause in _claim_clauses(text):
        if not clause:
            continue
        m = _VALIDATED_WORD.search(clause)
        if m is None or not _VALIDATION_NEGATED.search(clause[:m.end()]):
            continue
        before = clause[:m.start()]
        for entry in validated:
            name, _sep, detail = entry.partition(":")
            msg = (f"it says {name.strip()} is not validated, but the result lists it as validated:"
                   f"{detail.split(';', 1)[-1] if ';' in detail else detail}")
            if name.strip() and _names_quantity(before, name.strip()) and msg not in out:
                out.append(msg)
    return out


# ---------------------------------------------------------------------------
# "validated" / "verified" without a reference check inside its band
# ---------------------------------------------------------------------------

#: a claim of validation or verification (the noun "validation" is left out: "needs further validation";
#: "accurate to ±x" belongs to the standard-error follow-up, agent.treats_stderr_as_accuracy)
_OVERCLAIM_WORD = re.compile(r"\b(?:validated|validates|verified|verifies)\b", re.IGNORECASE)
#: words that make the clause a condition or a to-do, not a claim ("must be verified by a grid study")
_OVERCLAIM_HEDGE = re.compile(r"\b(?:needs?|needed|would|should|could|can|must|might|may|until|before|if|unless|"
                              r"requires?|required|to\s+be|yet\s+to|still\s+to|once)\b", re.IGNORECASE)
#: "verified against the tool results": the answer describing the claims check, not the physics
_OVERCLAIM_SELF = re.compile(r"\bverifie[ds]\s+(?:by|against|with|from|in)\s+(?:the\s+)?tool", re.IGNORECASE)
#: how a reference check's quantity is named in an answer
_REF_QUANTITY_WORDS = {"nusselt": r"nusselt|\bNu\b", "fan thrust": r"thrust"}
#: quantities no reference case checks: a "validated" one of these is never covered by a case's check
_OTHER_QUANTITY_WORDS = {
    "drag": r"\bC[Dd]\b|\bdrag\b", "lift": r"\bC[Ll]\b|\blift\b", "L/D": r"\bL\s*/\s*D\b|lift[- ]to[- ]drag",
    "heat rate": r"heat (?:transfer )?rate", "volume flow": r"volume flow|flow rate",
    "temperature": r"temperature", "velocity": r"velocit|\bspeed\b",
}


def _band_text(r: dict) -> str:
    return f"± {float(r['band_pct']):g} %" if isinstance(r.get("band_pct"), (int, float)) else "no stated band"


def _overclaim_problems(text: str, references: list[dict], model_validated: list[str]) -> list[tuple[str, str]]:
    """(problem, status) for a statement that calls a result validated or verified without the evidence.

    qwen3:8b wrote "The result is validated against the reference setup" after a
    single draft run, and the claims check said "all verified": it checks numbers
    and actions, and this sentence has neither (critic round 3.4). A claim of
    validation passes only when the experiment made a reference check
    (:func:`aerolume.ai.tools.reference_summary`: a run's ``reference_check`` or
    compare_reference) of the quantity it names, the check is a validation (not a
    consistency check, which the set-up imposes) and the difference lies inside
    the tolerance band docs/validation.md states for that reference
    (``within_band``). A clause that names a quantity the result lists under
    ``validated_quantities`` from the documented validation (``model_validated``:
    the polar's cl, the lattice's CL_alpha) is that evidence and passes; a
    conditional or to-do ("must be verified by a grid study") and a negated
    clause are not claims.

    Status "contradicted" when the evidence says otherwise (outside the band, or
    only a consistency check), "unsupported" when nothing in the experiment
    bears on it, "verified" (the text is then the evidence, not a problem) when
    a check inside its band supports it.
    """
    out: list[tuple[str, str]] = []
    for clause in _claim_clauses(text):
        if not clause:
            continue
        m = _OVERCLAIM_WORD.search(clause)
        if m is None or _OVERCLAIM_SELF.search(clause):
            continue
        before = clause[:m.start()]
        if _VALIDATION_NEGATED.search(clause[:m.end()]) or _OVERCLAIM_HEDGE.search(before):
            continue
        if any(_names_quantity(clause, q) for q in model_validated):
            # documented validation of a quantity the result lists as validated ("the lift slope is
            # validated", "validated for CL_alpha")
            continue
        named_ref = {q for q, words in _REF_QUANTITY_WORDS.items() if re.search(words, clause, re.IGNORECASE)}
        named_other = [q for q, words in _OTHER_QUANTITY_WORDS.items() if re.search(words, before, re.IGNORECASE)]
        refs = [r for r in references if r.get("applies") and isinstance(r.get("difference_pct"), (int, float))]
        if named_ref:
            refs = [r for r in refs if str(r.get("quantity", "")) in named_ref]
        elif named_other:
            msg = (f"it calls the {named_other[0]} validated or verified, but no reference check in this "
                   f"experiment covers the {named_other[0]}")
            if (msg, "unsupported") not in out:
                out.append((msg, "unsupported"))
            continue
        word = m.group(0).lower()
        valid = [r for r in refs if r.get("kind", "validation") == "validation"]
        inside = [r for r in valid if r.get("within_band") is True]
        if inside:
            r = inside[0]
            ev = (f"reference check: {float(r['difference_pct']):+.2g} % against "
                  f"{r.get('source') or 'the published reference'}, inside its {_band_text(r)} band")
            if (ev, "verified") not in out:
                out.append((ev, "verified"))         # the evidence the panel lists under the statement
            continue
        if valid and any(isinstance(r.get("band_pct"), (int, float)) for r in valid):
            r = next(r for r in valid if isinstance(r.get("band_pct"), (int, float)))
            msg = (f"it says “{word}”, but the reference check differs by {float(r['difference_pct']):+.2g} %, "
                   f"outside its {_band_text(r)} band ({r.get('source') or 'published reference'})")
            status = "contradicted"
        elif valid:
            msg = (f"it says “{word}”, but no tolerance band is stated for the reference "
                   f"({valid[0].get('source') or 'published reference'}), so agreement within a stated band "
                   "cannot be shown; give the difference in percent instead")
            status = "unsupported"
        elif refs:
            msg = (f"it says “{word}”, but the only reference check is a consistency check: the set-up imposes "
                   "that value, so agreement validates nothing")
            status = "contradicted"
        else:
            msg = (f"it says “{word}”, but this experiment made no reference check (compare_reference against a "
                   "published value inside its tolerance band), so nothing supports it")
            status = "unsupported"
        if (msg, status) not in out:
            out.append((msg, status))
    return out


# ---------------------------------------------------------------------------
# a ± given for a steady quantity
# ---------------------------------------------------------------------------

#: how a steady quantity (a key of a result summary) is named in an answer
_STEADY_WORDS = {
    "nusselt": r"nusselt|\bNu\b",
    "cd": r"\bC[Dd]\b|drag coefficient",
    "cl": r"\bC[Ll]\b|lift coefficient",
    "cm": r"\bC[Mm]\b|moment coefficient",
    "cs": r"\bC[Ss]\b|side[- ]force coefficient",
    "c_roll": r"rolling moment",
    "cn": r"\bC[Nn]\b|yawing moment",
}
_PM_NUMBER = re.compile(r"(?:±|\+/-|\+-)\s*(\d+(?:[.,]\d+)?"
                        r"(?:\s*(?:[eE][-−+]?\d+|[×x]\s*10\s*[-−⁻+]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)")
# a ± that is a grid uncertainty is the right figure for a steady value
_GRID_ERROR = re.compile(r"\bGCI\b|grid (?:study|convergence|uncertainty|refinement)|discreti[sz]ation",
                         re.IGNORECASE)


def _steady_problems(text: str, steady: list[str]) -> list[str]:
    """A ± written for a quantity whose standard error is round-off (a steady run), or a ± of zero.

    qwen3:8b answered "Nu is 2.25 ± 0.00 (4.4e-16)" for the converged cavity
    and every number traced to a tool result (critic round 3.3). A steady value
    has no averaging uncertainty; its error is the discretisation error, which a
    grid study gives. A ± in a clause that names a GCI or grid study is that
    figure and passes.
    """
    out: list[str] = []
    for clause in _CLAIM_CLAUSES.split(text or ""):
        if not clause or _GRID_ERROR.search(clause):
            continue
        pms = list(_PM_NUMBER.finditer(clause))
        if not pms:
            continue
        zero = any(_zero(m.group(1)) for m in pms)
        named = [q for q in steady if re.search(_STEADY_WORDS.get(q.lower(), rf"(?<!\w){re.escape(q)}(?!\w)"),
                                                 clause, re.IGNORECASE)]
        if named:
            msg = (f"it gives a ± for {named[0]}, which is steady: its standard error is round-off, not an "
                   "uncertainty; its error is the discretisation error, which a grid study (GCI) gives")
        elif zero:
            msg = ("it gives an uncertainty of ± 0: a zero standard error means a steady value (round-off), not an "
                   "exact one; its error is the discretisation error, which a grid study (GCI) gives")
        else:
            continue
        if msg not in out:
            out.append(msg)
    return out


def _zero(token: str) -> bool:
    try:
        return float(token.replace(",", ".").replace("−", "-").split("×")[0].split("x")[0]) == 0.0
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# statements about the physical models a run used
# ---------------------------------------------------------------------------

# a model named in a statement: buoyancy (the Boussinesq approximation, natural convection) and a turbulence model
_BUOYANCY_WORDS = re.compile(r"\bBoussinesq\b|\bbuoyan\w*|\b(?:natural|free|mixed)\s+convection\b", re.IGNORECASE)
_LES_WORDS = re.compile(r"\bLES\b|\blarge[- ]eddy\b|\bSmagorinsky\b|\bsub-?grid(?:[- ]scale)?\b|\bturbulence model\b",
                        re.IGNORECASE)
_LAMINAR_WORDS = re.compile(r"\blaminar\s+(?:model|solver|simulation|run|setting|mode|computation|treatment|"
                            r"calculation)s?\b|\bno turbulence model\b|\bwithout (?:a |any )?turbulence model\b",
                            re.IGNORECASE)
# "buoyancy was off", "without buoyancy", "neglecting natural convection": a statement that the model was NOT used
_MODEL_OFF = re.compile(r"\b(?:off|no|not|without|neglect\w*|disabled|ignor\w*|exclud\w*|absent|omit\w*|none)\b|"
                        r"n['’]t\b", re.IGNORECASE)
# a hypothetical ("buoyancy would raise Nu", "if natural convection mattered"): not a claim about the run
_MODEL_HYPOTHETICAL = re.compile(r"\b(?:would|could|might|may|should|if|unless|whether|can|future|next|consider\w*|"
                                 r"recommend\w*|suggest\w*|propos\w*)\b", re.IGNORECASE)
_MODEL_CLAUSES = re.compile(r"[;:]|,\s+(?:but|while|whereas|though|although|and)\b|\s(?:but|while|whereas)\s|"
                            r"(?<!\d)\.(?!\d)", re.IGNORECASE)


def _physics_problems(text: str, sims: list[SimRecord]) -> list[str]:
    """Statements about a physical model that contradict the settings the runs used.

    qwen3:8b gave "the assumptions made in the Boussinesq energy equation" as a
    validity limit of a heated cylinder that ran with buoyancy off (forced
    convection), and every number in the answer traced, so the number check
    passed it. Here a statement that names buoyancy (Boussinesq, natural or free
    convection) or a turbulence model (LES, Smagorinsky, subgrid) is compared
    with :attr:`SimRecord.physics` of the runs it is about (those it names, else
    all with known physics): naming a model that was off in every one of them, or
    saying a model was off when every one of them used it, is contradicted.
    Hypothetical clauses ("buoyancy would raise Nu") are not claims.
    """
    known = [s for s in sims if s.physics]
    if not known:
        return []
    out: list[str] = []
    for clause in _MODEL_CLAUSES.split(text or ""):
        if not clause or not clause.strip() or _MODEL_HYPOTHETICAL.search(clause):
            continue
        about = [s for s in _sims_named(clause, known)] or [s for s in _sims_named(text, known)] or known
        off = bool(_MODEL_OFF.search(clause))
        names = ", ".join(s.label for s in about[:2]) + (" …" if len(about) > 2 else "")
        if _BUOYANCY_WORDS.search(clause) and all("buoyancy" in s.physics for s in about):
            used = [bool(s.physics["buoyancy"]) for s in about]
            if not off and not any(used):
                msg = (f"it refers to buoyancy (the Boussinesq model), but buoyancy was off in {names} "
                       "(forced convection: physics_models.buoyancy is false)")
                if msg not in out:
                    out.append(msg)
            elif off and all(used):
                msg = f"it says buoyancy was not modelled, but {names} ran with buoyancy on"
                if msg not in out:
                    out.append(msg)
        turbs = [str(s.physics.get("turbulence", "")).lower() for s in about]
        if not all(t in ("laminar", "les") for t in turbs):
            continue                                   # unknown or "auto": nothing to compare with
        les_named, laminar_named = bool(_LES_WORDS.search(clause)), bool(_LAMINAR_WORDS.search(clause))
        if les_named and not off and all(t == "laminar" for t in turbs):
            msg = f"it refers to a turbulence model (LES), but {names} ran laminar, with no turbulence model"
            if msg not in out:
                out.append(msg)
        elif (laminar_named or (les_named and off)) and all(t == "les" for t in turbs):
            msg = f"it says the run was laminar, but {names} used the LES turbulence model"
            if msg not in out:
                out.append(msg)
    return out


def check_answer(answer: str, *, sources: list[NumberSource], sims: list[SimRecord],
                 actions: list[ActionRecord], designs: list[str] | None = None,
                 not_validated: list[str] | None = None, steady: list[str] | None = None,
                 validated: list[str] | None = None, references: list[dict] | None = None,
                 model_validated: list[str] | None = None) -> ClaimReport:
    """Check every statement of ``answer`` (see the module docstring).

    ``sources`` are the numbers of the tool results, the tool inputs and the
    goal; ``sims`` the simulations the experiment ran or read; ``actions`` its
    tool calls; ``designs`` the project's design names (to see which design a
    claimed run is about); ``not_validated`` every quantity a result of the
    experiment marked not validated (a statement calling one of them validated
    is contradicted); ``steady`` every quantity a result marked steady (a ± given
    for one is round-off quoted as an uncertainty, and is contradicted too);
    ``validated`` the validated reference set-up entries of the results (a
    statement calling one of those quantities not validated is contradicted);
    ``references`` the reference checks the experiment made (``reference_check``
    of a result, the rows of compare_reference) and ``model_validated`` the
    ``validated_quantities`` of its results: a statement calling a result
    validated or verified needs one of them (:func:`_overclaim_problems`).
    """
    report = ClaimReport()
    for text in split_statements(answer):
        evidence, problems = _check_actions(text, actions, list(designs or []))
        contradicted_validation = _validation_problems(text, list(not_validated or []))
        problems.extend(contradicted_validation)
        problems.extend(_physics_problems(text, sims))
        problems.extend(_steady_problems(text, list(steady or [])))
        problems.extend(_unvalidation_problems(text, list(validated or [])))
        # a "validated" already contradicted by a not_validated entry needs no second verdict
        over = ([] if contradicted_validation else
                _overclaim_problems(text, list(references or []), list(model_validated or [])))
        hard = bool(problems) or any(st == "contradicted" for _m, st in over)
        problems.extend(msg for msg, st in over if st != "verified")
        evidence.extend(msg for msg, st in over if st == "verified")
        nums = _check_numbers(text, sources, sims)
        status = "plain"
        if evidence or problems or nums:
            ranks = [_RANK[n.status] for n in nums]
            if hard:
                ranks.append(_RANK["contradicted"])
            elif problems:
                ranks.append(_RANK["unsupported"])          # a claim nothing in the experiment bears on
            if evidence:
                ranks.append(_RANK["verified"])
            status = {v: k for k, v in _RANK.items()}[max(ranks)]
        for n in nums:
            if n.status == "contradicted":
                problems.append(f"{n.token} is {n.evidence}" + (f"; {n.correction}" if n.correction else ""))
            elif n.status == "unsupported":
                problems.append(f"{n.token} is {n.evidence}" if n.evidence.startswith("given for")
                                else f"{n.token} is in no tool result")
            elif n.status == "derived":
                evidence.append(f"{n.token} = {n.evidence}")
            else:
                evidence.append(f"{n.token}: {n.evidence}")
        report.statements.append(Statement(text, status, nums, evidence, problems))
    return report


__all__ = ["CLAIM_KINDS", "STATUSES", "STATUS_WORDS", "ActionRecord", "ClaimReport", "NumberCheck", "NumberSource",
           "SimRecord", "Statement", "check_answer", "claimed_actions", "split_statements"]
