"""Keyword routing from a goal to the reference-case tools, and the completeness checks that use it.

Two live failures with qwen3:8b (critic round 3) are what this module answers:

* asked to compare a heated cylinder's Nusselt number with the Lange
  correlation, it reported "Nu 3.3402 ± 0.001" with no comparison at all
  (Lange 3.28, +1.8 %), and the claims check passed it -- every number it did
  write traced to a result;
* asked to run a new differentially heated cavity at Ra 1e5 against de Vahl
  Davis, it answered after 7.5 s that "the tools cannot run a simulation for a
  differentially heated square cavity", and the loop accepted the refusal as
  the final answer although create_case sets that case up.

:func:`asks_reference` finds the reference a goal names, :func:`gives_comparison`
says whether an answer quotes it, :func:`says_cannot` spots a refusal, and
:func:`route` turns the goal's keywords into the concrete tool calls the loop
asks for once (cavity / Rayleigh -> ``create_case cavity``, cylinder / Nusselt /
Lange -> ``create_case heated_cylinder``, fan -> ``create_case fan_room``, which is
a consistency check only). Pure functions, no Qt, no model.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .experiment import extract_numbers

_I = re.IGNORECASE

#: (pattern, how the answer should name it, the case that reproduces it)
_REFERENCES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\bLange\b", _I), "the Lange et al. (1998) correlation", "heated_cylinder"),
    (re.compile(r"\bde\s*Vahl\s*Davis\b", _I), "de Vahl Davis (1983)", "cavity"),
    (re.compile(r"\bTric\b|\bFusegi\b", _I), "Tric et al. (2000)", "cavity"),
    (re.compile(r"\bmomentum theory\b|\bFroude\b|\bactuator[- ]disc theory\b", _I), "momentum theory", "fan_room"),
    (re.compile(r"\b(?:published|reference|benchmark|literature|experimental)\s+(?:values?|results?|data|"
                r"solutions?|numbers?|cases?)\b|\bbenchmark\b|\bcorrelation\b", _I), "the published reference", ""),
)

#: (pattern, case) of the keyword router; the first match wins
_ROUTES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcavity\b|\bRayleigh\b|\bRa\s*[=≈]?\s*\d|\bnatural convection\b|\bdifferentially heated\b|"
                r"\bde\s*Vahl\s*Davis\b|\bTric\b", _I), "cavity"),
    (re.compile(r"\bLange\b|\bcylinder\b[^.]*\b(?:Nusselt|heat\w*|convection)\b|"
                r"\b(?:Nusselt|heated|forced convection)\b[^.]*\bcylinder\b", _I), "heated_cylinder"),
    (re.compile(r"\bfans?\b|\bmomentum theory\b|\bactuator[- ]disc\b|\bthrust\b", _I), "fan_room"),
)

# a refusal to do the work: "the tools cannot run a simulation", "it is not possible to set up", "no tool can
# simulate" -- not any negation ("the difference isn't significant", "it did not converge")
_CANNOT = re.compile(
    r"\b(?:cannot|can\s+not|can['’]t|unable to|not able to|(?:is|are|was|were) not possible to|impossible to|"
    r"no (?:tool|way|function)s?(?: \w+)? (?:to|can|that|for))\s+(?:\w+\s+){0,3}?"
    r"(?:run\w*|simulat\w*|perform\w*|creat\w*|set\s+up|model\w*|comput\w*|calculat\w*|do|compar\w*|analy[sz]\w*|"
    r"reproduc\w*|carry out|execut\w*)\b"
    r"|\b(?:tools?|Aerolume|the solvers?)\s+(?:do|does|did)\s+not\s+(?:support|offer|provide|allow|include|have)\b",
    _I)

# a number after a keyword: "Ra 1e5", "Rayleigh number of 10^5", "Ra = 1.0×10⁵", "Re 40"
_NUMBER = (r"(?P<num>\d+(?:[.,]\d+)?(?:\s*(?:[eE][-+]?\d+|[×x*]\s*10\s*(?:\^|\*\*)?\s*[-+⁻]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+|"
           r"(?:\^|\*\*)\s*[-+]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?)")
_RA = re.compile(r"(?:\bRayleigh(?:\s+number)?|\bRa)\s*(?:=|≈|of|is|at)?\s*\$?\s*" + _NUMBER, _I)
_RE = re.compile(r"(?:\bReynolds(?:\s+number)?|\bRe)\s*(?:=|≈|of|is|at)?\s*\$?\s*" + _NUMBER, _I)
_THRUST = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*N\b(?!\w)")
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_PLAIN_NUMBER = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?(?![\w.]\d)")
_PERCENT = re.compile(r"\s*(?:%|per\s?cent\b)", _I)


def parse_value(token: str) -> float | None:
    """A number as people write it after "Ra" or "Re": 1e5, 1.0×10⁵, 10^5, 10**5, 10⁵, 100,000, 40."""
    t = token.strip().replace(" ", "").replace("−", "-")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\^|\*\*)([-+]?\d+)", t)
    if m:
        return float(m.group(1)) ** int(m.group(2))
    m = re.fullmatch(r"(\d+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", t)
    if m:
        return float(m.group(1)) ** int(m.group(2).translate(_SUPER))
    t = t.replace("*", "×")
    parsed = extract_numbers(t)
    return parsed[0][1] if parsed else None


def _number_after(rx: re.Pattern[str], text: str) -> float | None:
    for m in rx.finditer(text or ""):
        v = parse_value(m.group("num"))
        if v is not None and math.isfinite(v) and v > 0:
            return v
    return None


def asks_reference(goal: str) -> tuple[str, str]:
    """(how to name the reference, its case) for a goal that asks for a comparison with one, else ("", "")."""
    for rx, name, case in _REFERENCES:
        if rx.search(goal or ""):
            return name, case
    return "", ""


def says_cannot(answer: str) -> bool:
    """Whether an answer says the task cannot be done ("the tools cannot run ...", "not possible")."""
    return bool(_CANNOT.search(answer or ""))


@dataclass
class Route:
    """The tool calls that answer one kind of goal: ``case`` for create_case and its arguments."""
    case: str
    args: dict

    def text(self) -> str:
        """The calls as one instruction sentence for the model."""
        bits = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.args.items())
        call = f'call create_case with case="{self.case}"' + (f", {bits}" if bits else "")
        tail = (" then run_simulation with the simulation_id it returns, then compare_reference with that "
                "simulation_id.")
        if self.case == "fan_room":
            tail += (" The fan room is a consistency check only (thrust mode imposes the thrust) and momentum "
                     "theory does not apply in a closed room: say so.")
        return call[0].upper() + call[1:] + "," + tail


def route(goal: str, case: str = "") -> Route | None:
    """The reference-case call a goal's keywords point to (``case`` forces the kind), or None."""
    kind = case
    if not kind:
        for rx, k in _ROUTES:
            if rx.search(goal or ""):
                kind = k
                break
    if not kind:
        return None
    args: dict = {}
    if kind == "cavity":
        ra = _number_after(_RA, goal)
        if ra is not None:
            args["rayleigh"] = float(ra)
        if re.search(r"\bcub\w*|\b3D\b|\bthree[- ]dimensional\b", goal or "", _I):
            args["dimension"] = 3
    elif kind == "heated_cylinder":
        re_ = _number_after(_RE, goal)
        if re_ is not None:
            args["reynolds"] = float(re_)
    elif kind == "fan_room":
        m = _THRUST.search(goal or "")
        if m:
            args["thrust_n"] = float(m.group("num"))
    return Route(kind, args)


def gives_comparison(answer: str, checks: list[dict]) -> bool:
    """Whether an answer quotes a reference check: its reference value or its difference in %.

    A number of the answer counts when it equals the check's ``reference`` or
    the magnitude of its ``difference_pct`` at the precision it is written with
    ("3.28", "+1.8 %" for 1.835); with no checks at hand, a percentage sign next
    to the word reference / difference is accepted as a comparison.
    """
    text = answer or ""
    # (value, half a unit of the last digit, followed by a percent sign)
    nums: list[tuple[float, float, bool]] = []
    for m in _PLAIN_NUMBER.finditer(text):
        parsed = extract_numbers(m.group(0))
        if parsed:
            nums.append((parsed[0][1], parsed[0][2], bool(_PERCENT.match(text, m.end()))))
    for chk in checks:
        for key in ("reference", "difference_pct"):
            t = chk.get(key)
            if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(float(t)):
                continue
            for v, half, pct in nums:
                if key == "difference_pct" and not pct:
                    continue            # "2 runs" is no difference of 1.8 %: a difference is written with "%"
                if abs(abs(v) - abs(float(t))) <= max(half * 1.0001, 1e-9 * abs(float(t))):
                    return True
    if not checks:
        return bool(re.search(r"\b(?:differ\w*|deviat\w*|above|below|off)\b[^.]{0,60}%|%[^.]{0,60}\b(?:differ\w*|"
                              r"deviat\w*|reference|published)\b", text, _I))
    return False


__all__ = ["Route", "asks_reference", "gives_comparison", "parse_value", "route", "says_cannot"]
