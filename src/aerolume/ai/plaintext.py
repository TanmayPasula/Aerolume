"""Model answers as plain text: inline LaTeX math becomes Unicode before anyone reads it.

Language models write numbers as LaTeX out of habit -- qwen3:8b answered
"$1 \\times 10^5$", "$4.5624 \\pm 0.000019$" and "$1.18\\%$" in a de Vahl Davis
comparison (critic round 3.2) -- and the panel showed the dollar signs and
backslashes verbatim, the most machine-looking text in the app. The system
prompts ask for plain text, and :func:`plain_math` makes sure of it: it runs on
every model turn (so the transcript, the experiment log and the claims check all
read "1 × 10⁵", "4.5624 ± 0.000019", "1.18 %") and, through
:class:`MathStream`, on the streamed text as it arrives.

Rules (deliberately narrow: prose with a currency sign must survive):

* ``$...$`` is math only when the opening ``$`` is followed by a non-space, the
  closing one is preceded by a non-space and not followed by a digit, and the
  span holds no line break and at most :data:`MAX_INLINE` characters (pandoc's
  rule) -- "costs $5 and $10" is left alone. ``$$...$$``, ``\\(...\\)`` and
  ``\\[...\\]`` are math too.
* Inside math: known commands become their symbols (``\\times`` ×, ``\\pm`` ±,
  ``\\approx`` ≈, ``\\leq`` ≤, Greek letters, ``^\\circ`` °), ``^5`` / ``^{-3}``
  become superscripts when every character has one (else "^(…)"), ``_2`` a
  subscript digit (``C_d`` keeps its underscore: there is no subscript d),
  ``\\frac{a}{b}`` a/b, ``\\sqrt{x}`` √x, ``\\text{...}`` its text; braces go.
* Outside math only the unambiguous symbol commands are replaced (``\\times``,
  ``\\pm``, ``\\%``, Greek letters, ...): a model sometimes drops the dollars.

:func:`plain_math` is idempotent and leaves text without math untouched.
"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

#: longest inline span taken as math (a longer "$...$" is prose with two currency signs)
MAX_INLINE = 200
#: longest display span ($$...$$)
MAX_DISPLAY = 400

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_SYMBOLS = {
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓", "approx": "≈", "sim": "~", "simeq": "≃", "propto": "∝",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠", "ll": "≪", "gg": "≫",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "infty": "∞", "partial": "∂",
    "nabla": "∇", "degree": "°", "circ": "°", "prime": "′", "ldots": "…", "dots": "…", "cdots": "⋯",
    "%": "%", "$": "$", "#": "#", "&": "&", "_": "_", ",": " ", ";": " ", ":": " ", "!": "", " ": " ",
    "quad": " ", "qquad": "  ", "left": "", "right": "", "big": "", "Big": "", "displaystyle": "",
    **_GREEK,
}
#: commands replaced outside math as well: the symbols and Greek letters, never spacing, sizing or the
#: two-letter relations ("\to", "\le"), which a Windows path quoted in an answer could also contain
_PROSE_SAFE = ({k for k, v in _SYMBOLS.items() if k.isalpha() and len(k) > 2 and v}
               - {"quad", "qquad", "left", "right", "displaystyle", "circ", "sim", "dots"}) | {"%", "mu", "nu",
                                                                                                  "pi", "xi", "pm"}
_TEXT_CMDS = ("text", "mathrm", "textrm", "mathit", "textit", "mathbf", "textbf", "operatorname", "mbox", "rm")

_SUP = dict(zip("0123456789+-−=()n i", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿ ⁱ", strict=True))
_SUB = dict(zip("0123456789+-−=()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎", strict=True))

_SPAN = re.compile(
    rf"(?<!\\)\$\$(?P<d>[^$]{{1,{MAX_DISPLAY}}}?)(?<!\\)\$\$"                          # $$ display $$
    rf"|(?<![\\$])\$(?P<i>[^\s$](?:[^$\n]{{0,{MAX_INLINE - 2}}}?[^\s$\\])?)\$(?![\d$])"  # $inline$ (pandoc)
    rf"|\\\((?P<p>.{{1,{MAX_INLINE}}}?)\\\)"                                           # \( inline \)
    rf"|\\\[(?P<b>.{{1,{MAX_DISPLAY}}}?)\\\]",                                         # \[ display \]
    re.DOTALL)
_PROSE_CMD = re.compile(r"\\(%|[A-Za-z]+)(?![A-Za-z])")
_MATH_HINT = re.compile(r"[\\^_{}=]|\d")


def _script(arg: str, table: dict[str, str], mark: str) -> str:
    """``arg`` in super/subscript characters, or ``mark`` + arg ("^(...)" / "_x") when one has none."""
    arg = arg.strip()
    if arg and all(c in table for c in arg):
        return "".join(table[c] for c in arg)
    if mark == "^":
        return f"^({arg})" if len(arg) > 1 else f"^{arg}"
    return f"_{arg}"


def _group(s: str, i: int) -> tuple[str, int]:
    """The argument at ``s[i]``: a {braced} group (nested), a \\command, or one character; (text, next index)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _wrap(x: str) -> str:
    """Parenthesise a fraction part that is more than one term ("a+b" -> "(a+b)")."""
    return f"({x})" if re.search(r"[\s+\-−×·/]", x.strip()) else x.strip()


def _math(s: str) -> str:
    """One math span's content as plain Unicode text."""
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            if not m:
                i += 1
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in _TEXT_CMDS:
                arg, i = _group(s, i)
                out.append(arg)
            elif name in ("frac", "dfrac", "tfrac"):
                a, i = _group(s, i)
                b, i = _group(s, i)
                out.append(f"{_wrap(_math(a))}/{_wrap(_math(b))}")
            elif name == "sqrt":
                a, i = _group(s, i)
                a = _math(a)
                out.append("√" + (_wrap(a) if len(a.strip()) > 1 else a.strip()))
            elif name in _GREEK:
                # the space after a command only ends its name in LaTeX: "\Delta T" is ΔT
                out.append(_GREEK[name])
                while i < len(s) and s[i] == " " and i + 1 < len(s) and s[i + 1].isalpha():
                    i += 1
            elif name in ("pm", "mp") and not "".join(out).strip():
                # a leading ± is a sign, not an operator: "$\pm 0.5$" reads ±0.5
                out.append(_SYMBOLS[name])
                while i < len(s) and s[i] == " ":
                    i += 1
            elif name in _SYMBOLS:
                out.append(_SYMBOLS[name])
            else:
                out.append(name)              # an unknown command: its name reads better than a backslash
        elif c in "^_":
            arg, i = _group(s, i + 1)
            arg = _math(arg)
            if c == "^" and arg.strip() in ("°", "\\circ", "o"):
                out.append("°")
            else:
                out.append(_script(arg, _SUP if c == "^" else _SUB, c))
        elif c in "{}":
            i += 1
        elif c == "~":
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    text = "".join(out)
    text = re.sub(r"(\d)\s*°", r"\1°", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _prose(s: str) -> str:
    """Text outside math: only the unambiguous symbol commands (a model that dropped its dollars)."""
    if "\\" not in s:
        return s
    return _PROSE_CMD.sub(lambda m: _SYMBOLS[m.group(1)] if m.group(1) in _PROSE_SAFE else m.group(0), s)


def plain_math(text: str) -> str:
    """``text`` with its LaTeX math as plain Unicode (see the module docstring). Idempotent."""
    if not text or ("$" not in text and "\\" not in text):
        return text or ""
    out: list[str] = []
    pos = 0
    for m in _SPAN.finditer(text):
        body = next(g for g in (m.group("d"), m.group("i"), m.group("p"), m.group("b")) if g is not None)
        if m.group("i") is not None and " " in body and not _MATH_HINT.search(body):
            continue                          # "$two words$": prose between two dollar signs, left as written
        out.append(_prose(text[pos:m.start()]))
        out.append(_math(body).strip())
        pos = m.end()
    out.append(_prose(text[pos:]))
    return "".join(out)


class MathStream:
    """:func:`plain_math` for streamed text: holds back only what a later delta could still change.

    ``feed(delta)`` returns the converted text that is final; ``flush()`` the rest
    at the end of a turn. The concatenation of everything returned equals
    ``plain_math`` of the whole text, so the streamed answer and the final one
    (log, claims check, "done") never differ. Held back: an opening ``$`` /
    ``$$`` / ``\\(`` / ``\\[`` whose span could still close (no line break yet,
    within its length limit), a trailing ``\\command`` that may grow, and a
    trailing ``$`` (it may be the first of ``$$`` or close a span).
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buf += delta
        cut = self._safe_cut(self._buf)
        if cut <= 0:
            return ""
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return plain_math(ready)

    def flush(self) -> str:
        rest, self._buf = self._buf, ""
        return plain_math(rest)

    @staticmethod
    def _safe_cut(s: str) -> int:
        """Length of the prefix of ``s`` whose conversion no later text can change."""
        hold = len(s)
        end = 0
        last = None
        for last in _SPAN.finditer(s):
            end = last.end()
        if last is not None and end == len(s) and last.group("i") is not None:
            # an inline span closes on the last character: a digit after it would unmake it ("$x$5")
            hold = end = last.start()
        tail = s[end:]
        for opener, limit, newline_ends in (("$$", MAX_DISPLAY + 4, False), ("$", MAX_INLINE + 2, True),
                                           ("\\(", MAX_INLINE + 4, False), ("\\[", MAX_DISPLAY + 4, False)):
            k = tail.find(opener)
            while k >= 0:
                rest = tail[k:]
                still_open = len(rest) <= limit and not (newline_ends and "\n" in rest)
                if opener == "$" and len(rest) > 1 and rest[1].isspace():
                    still_open = False            # "$ 5": a "$" before a space opens nothing
                escaped = k > 0 and tail[k - 1] == "\\"
                if still_open and not escaped:
                    hold = min(hold, end + k)
                    break
                k = tail.find(opener, k + 1)
        m = re.search(r"\\[A-Za-z]*$", s)
        if m:
            hold = min(hold, m.start())
        return max(0, hold)


# ---------------------------------------------------------------------------
# a quoted uncertainty, rounded as the Results tables round it
# ---------------------------------------------------------------------------

def pdg_round(err) -> tuple[float, int, int] | None:
    """(rounded error, significant figures, decimals) of an uncertainty by the Particle Data Group rule.

    The Results tables' rule, :func:`aerolume.core.format.round_error` (Qt-free, so the assistant
    applies it headless too: the MCP server, a build without PySide6). The rule (Review of Particle
    Physics, "Rounding"): the error's three leading digits 100-354 are quoted to two significant
    figures, 355-949 to one, 950-999 round up to 1000 and are quoted to two (0.00095 -> 0.0010).
    None when the error is missing, zero or not finite.
    """
    from ..core.format import round_error
    return round_error(err)


_NUM = r"[0-9]+(?:\.[0-9]+)?"
#: "value ± error" (or "± error" alone) in plain decimals; a number in scientific notation is left alone
_PM_SPAN = re.compile(
    rf"(?:(?<![0-9.,])(?P<v>{_NUM})(?P<vp>\s*%)?(?P<gap>\s*))?"
    rf"(?P<pm>±|\+/-|\+-)(?P<sp>\s*)(?P<e>{_NUM})(?![0-9.])(?P<ep>\s*%)?"
    r"(?!\s*(?:[eE][-+−]?[0-9]|[×x·]\s*10))")


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


def _requantize(token: str, decimals: int) -> str:
    """``token`` (a decimal literal) rounded half-up to ``decimals`` places, from its own digits."""
    return str(Decimal(token).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def _round_pm_match(m: re.Match) -> str:
    err_tok, val_tok = m.group("e"), m.group("v")
    rule = pdg_round(float(err_tok))
    if rule is None:
        return m.group(0)
    rounded, _digits, decimals = rule
    if _decimals(err_tok) > decimals:
        # more figures than the rule quotes: round to them (a shorter error is never padded with zeros)
        err_tok = f"{rounded:.{decimals}f}"
    target = _decimals(err_tok)
    if val_tok is not None and _decimals(val_tok) > target:
        val_tok = _requantize(val_tok, target)        # the value to the error's decimal place
    head = "" if val_tok is None else f"{val_tok}{m.group('vp') or ''}{m.group('gap')}"
    return f"{head}{m.group('pm')}{m.group('sp')}{err_tok}{m.group('ep') or ''}"


def round_pm(text: str) -> str:
    """``text`` with every "value ± error" quoted as the Results tables quote it. Idempotent.

    qwen3:8b answered "3.3407 ± 0.00095" while the table beside the answer showed
    ± 0.0010 (critic round 3.4): one number printed two ways reads as
    carelessness. The error is rounded by :func:`pdg_round` when it carries more
    figures than the rule quotes (0.00095 -> 0.0010, 0.0046 -> 0.005) and the
    value to the error's decimal place (3.34079 ± 0.0010 -> 3.3408). Nothing is
    padded: "3.34 ± 0.01" stays as written. Numbers in scientific notation are
    left alone.
    """
    if not text or not ("±" in text or "+/-" in text or "+-" in text):
        return text or ""
    return _PM_SPAN.sub(_round_pm_match, text)


def plain_answer(text: str) -> str:
    """A model answer as the panel, the log and the claims check read it: :func:`plain_math`, then
    :func:`round_pm`. Idempotent."""
    return round_pm(plain_math(text))


_TAIL_NUM = r"[0-9][0-9.]*(?:\s*(?:[eE][-+−]?[0-9]*|[×x·]\s*(?:10?)?))?(?:\s*%)?\s*"
#: what a streamed answer may still be in the middle of: a number, a "±", a partial "+/-" or exponent
_TAIL_NUMBER = re.compile(rf"(?:{_TAIL_NUM})?(?:(?:±|\\pm|\+/?-?)\s*(?:{_TAIL_NUM})?)?$")


class AnswerStream(MathStream):
    """:class:`MathStream` with :func:`round_pm`: the streamed text is :func:`plain_answer` of the whole.

    Besides an open math span it holds back a trailing number, "±" or partial
    "+/-" (with its blanks): "3.3407 ± 0.000" may still become "0.00095", and a
    value's decimals depend on the error after it, so a "value ± error" is
    always released whole.
    """

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._buf += delta
        cut = self._safe_cut(self._buf)
        m = _TAIL_NUMBER.search(self._buf, 0, cut)
        if m is not None:
            cut = m.start()
        if cut <= 0:
            return ""
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return plain_answer(ready)

    def flush(self) -> str:
        rest, self._buf = self._buf, ""
        return plain_answer(rest)


# ---------------------------------------------------------------------------
# a short summary that keeps its numbers
# ---------------------------------------------------------------------------

#: a short unit that belongs to the number before it ("20 m/s", "1.2 m²")
_UNIT_WORD = re.compile(r"^(?:m/s|km/h|mph|kn|m|mm|cm|km|m²|m³|s|ms|K|°C|°|%|W|kW|W/m|W/m²|N|N/m|Pa|kPa|Hz|deg)"
                        r"[,.;:)]*$")
#: how far past ``limit`` :func:`brief` may run to keep a number with its label
BRIEF_SLACK = 16


def brief(text: str, limit: int = 90) -> str:
    """``text`` as plain text on one line, cut at a word boundary to about ``limit`` characters.

    A plan summary cut at a fixed character read "the cavity at Ra , about 28 s"
    (critic round 3.4): the slice ended between "Ra" and "1e4", so the one
    number the line exists for was lost. Here maths becomes plain text first
    (:func:`plain_math`), the cut falls between words, and a number (with a short
    unit) right after the cut is kept even when that runs up to
    :data:`BRIEF_SLACK` characters past ``limit``: a label is never left without
    its value. When the number is longer than that, the label is dropped too.
    """
    t = " ".join(plain_math(str(text or "")).split())
    if len(t) <= limit:
        return t
    words = t.split(" ")
    kept: list[str] = []
    size = 0
    for w in words:
        extra = len(w) + (1 if kept else 0)
        if size + extra > limit - 1:
            break
        kept.append(w)
        size += extra
    if not kept:                                   # one word longer than the limit
        return t[: limit - 1].rstrip() + "…"
    rest = words[len(kept):]
    if rest and re.search(r"[0-9]", rest[0]):
        tail = [rest[0]] + ([rest[1]] if len(rest) > 1 and _UNIT_WORD.match(rest[1]) else [])
        if len(" ".join(kept + tail)) <= limit + BRIEF_SLACK:
            kept += tail
        elif len(kept) > 1:
            kept.pop()                             # cut before the label too, not between it and its value
    out = " ".join(kept).rstrip(" ,;:(–—-")
    return out + ("…" if len(kept) < len(words) else "")


__all__ = ["BRIEF_SLACK", "MAX_DISPLAY", "MAX_INLINE", "AnswerStream", "MathStream", "brief", "pdg_round",
           "plain_answer", "plain_math", "round_pm"]
