"""LaTeX math in model answers becomes plain Unicode (aerolume.ai.plaintext), whole and streamed."""
from __future__ import annotations

import random

import pytest

from aerolume.ai.plaintext import MathStream, plain_math

# the answer qwen3:8b gave in critic round 3.2 (de Vahl Davis cavity), as the panel showed it
CRITIC_ANSWER = ("The simulation of the heated cavity at Rayleigh number $1 \\times 10^5$ was completed, and the mean "
                 "Nusselt number was calculated as $4.5624 \\pm 0.000019$ (standard error). The difference from the "
                 "de Vahl Davis (1983) reference value of $4.509$ is $1.18\\%$.")


def test_the_critic_answer_reads_as_plain_text():
    out = plain_math(CRITIC_ANSWER)
    assert "$" not in out and "\\" not in out
    assert "Rayleigh number 1 × 10⁵ was completed" in out
    assert "4.5624 ± 0.000019 (standard error)" in out
    assert "value of 4.509 is 1.18%." in out


@pytest.mark.parametrize("src, want", [
    ("$Re = 3.2 \\times 10^{5}$", "Re = 3.2 × 10⁵"),
    ("$10^{-3}$ m", "10⁻³ m"),
    ("$\\alpha = 4^\\circ$", "α = 4°"),
    ("$\\Delta T = 10\\,\\text{K}$", "ΔT = 10\u2009K"),
    ("$C_d = 0.31$", "C_d = 0.31"),
    ("$T_2 \\approx 300$ K", "T₂ ≈ 300 K"),
    ("$\\frac{q L}{k \\Delta T}$", "(q L)/(k ΔT)"),
    ("$\\sqrt{g \\beta \\Delta T H}$", "√(g β ΔT H)"),
    ("$$Nu = 4.52$$", "Nu = 4.52"),
    ("\\(Ra \\le 10^6\\)", "Ra ≤ 10⁶"),
    ("$U$ is the reference speed", "U is the reference speed"),
    ("1.2 \\times 10^5 without dollars, 3 \\pm 0.1, 5\\%", "1.2 × 10^5 without dollars, 3 ± 0.1, 5%"),
])
def test_math_spans_and_commands(src, want):
    assert plain_math(src) == want


@pytest.mark.parametrize("prose", [
    "It costs $5 and $10 per run.",                     # pandoc's rule: a closer before a digit is not one
    "Between $ 5 and 6 $ nothing is math.",
    "The log is in C:\\Users\\me\\to\\experiments.",     # a path: "\\to" is not an arrow outside math
    "Cd 0.31 ± 0.004 at Re 1.0×10⁵ (already plain).",
    "",
])
def test_prose_is_left_alone(prose):
    assert plain_math(prose) == prose


def test_idempotent():
    once = plain_math(CRITIC_ANSWER)
    assert plain_math(once) == once


@pytest.mark.parametrize("seed", range(40))
def test_streaming_equals_the_whole(seed):
    """Whatever the chunking, the streamed text is exactly plain_math of the whole answer."""
    rng = random.Random(seed)
    texts = [CRITIC_ANSWER, "A $$\\frac{a}{b}$$ and \\(x^2\\) then $5 and $10.\nNext $x$5 line $y$.",
             "Ends with a command \\times", "Ends open $1 \\times 10^5", "Trailing $", "Re $10^5$"]
    text = texts[seed % len(texts)]
    ms = MathStream()
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 7)
        out.append(ms.feed(text[i:i + n]))
        i += n
    out.append(ms.flush())
    assert "".join(out) == plain_math(text)


def test_streaming_releases_text_promptly():
    """Plain text is not held back: only an open span or a trailing command waits."""
    ms = MathStream()
    assert ms.feed("The mean Nusselt number is ") == "The mean Nusselt number is "
    assert ms.feed("$4.56 \\p") == ""
    assert ms.feed("m 0.01$") == ""                  # the closer is last: a digit could still follow
    assert ms.feed(" (standard error).") == "4.56 ± 0.01 (standard error)."
    assert ms.flush() == ""
