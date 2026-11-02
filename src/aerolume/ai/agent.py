"""The experiment loop: model turn -> validated tool calls (with approvals) -> results -> next turn.

One :class:`Agent` is one experiment (one conversation, one log). It runs on a
worker thread; everything the user should see goes out through ``on_event(kind,
data)`` (the GUI turns it into a queued Qt signal, the CLI prints it):

==============  ==========================================================================
kind            data
==============  ==========================================================================
"status"        text - what is happening now ("Waiting for the model", "Running ...")
"text"          delta - streamed answer text
"thinking"      delta - streamed reasoning summary (Claude) or thinking (local model)
"turn"          text, thinking, usage, served_by - a finished model turn
"tool_start"    id, name, args, title
"tool_progress" id, fraction, message
"approval"      id, approval (:class:`~aerolume.ai.tools.Approval`) - the loop now waits
"tool_end"      id, name, status ("ok" | "error" | "invalid" | "declined" | "cancelled"),
                summary, files (PNG paths), seconds, args
"notice"        text, level ("info" | "warning"), corrects (the answer before it is being
                sent back to the model: the panel shows it as superseded)
"done"          status ("done" | "unverified" | "stopped" | "failed" | "refused"), text,
                untraced, path, withheld (answers replaced by the action check)
"error"         text, kind
==============  ==========================================================================

Rules the loop enforces whatever the model does:

* every tool input is validated against its schema before anything runs; an
  invalid input goes back as an error result (``{"INVALID_JSON": ...}`` when it
  was not even an object) so the model can retry;
* all tool calls of one turn are answered together in one message (both APIs
  require it), in the order they were made;
* long runs (estimated time above the threshold), every deletion and all code
  wait for :attr:`approve`; a declined call is answered "declined by the user";
* :meth:`stop` (the Stop button) sets the cancel event that the model stream,
  the running solver and any approval wait all watch, and closes a local
  model's connection; the experiment is logged as stopped;
* at most ``max_steps`` tool rounds per request;
* a model that answers with numbers without having run a single tool is
  reminded once that numbers must come from tool results;
* a model whose answer names a NACA section that none of the experiment's
  runs used is asked once to run it or correct the answer, and a mismatch that
  remains is flagged in the panel and the log (:func:`~aerolume.ai.experiment.
  design_mismatches`);
* an answer that claims an action ("has been deleted", "I ran") that no
  successful tool call of that kind performed is sent back once; if the
  corrected answer repeats the claim it is replaced by a fixed sentence and
  the experiment ends "unverified" (:func:`claimed_actions`);
* an answer that leaves a (design, speed / α / Re) pair of the goal without a
  matching run or read is asked once to run it (:func:`~aerolume.ai.experiment.
  goal_targets`), and one that gives a number for one simulation that belongs
  to another is asked once to correct it and flagged in the log
  (:func:`~aerolume.ai.experiment.attribution_problems`);
* a read call that repeats an earlier one exactly is answered with a note
  instead of the same payload (nothing changed in between);
* an empty reply (no text, no tool call) is asked about once; a second one ends
  the request "failed" with a plain message, never "done" with nothing to read;
* an answer that says the task "cannot" be done when the goal's keywords route
  to a reference case (:mod:`aerolume.ai.router`: cavity / Rayleigh, heated
  cylinder / Nusselt / Lange, fan) and the model never tried that route is sent
  back once with the concrete calls (qwen3:8b refused a de Vahl Davis cavity that
  create_case sets up);
* a goal that asks for a comparison with a reference (Lange, de Vahl Davis,
  momentum theory, "the published value") needs an answer that quotes it: when
  it does not, the loop calls compare_reference itself on the experiment's runs
  and sends the numbers back once (or, when no run has a reference, the router's
  calls); an answer that still leaves the comparison out ends "unverified";
* **every final answer goes through the claims checker** (:mod:`aerolume.ai.claims`):
  each statement's claimed actions must match successful tool calls and its
  numbers must trace to tool results (of the simulation the statement names);
  unsupported or contradicted statements are sent back once with the values they
  should have been, and the final report -- every checked statement with its
  verdict and evidence -- goes to the panel and the log. An answer that still has
  unsupported statements ends "unverified" (warn glyph, never the green check);
* an approval never fails open: an estimate that raises asks the user ("cost
  unknown"), a call whose own input is wrong is answered with the error and runs
  nothing, and any unexpected failure while preparing the question is treated as
  "cost unknown" too;
* nothing is sent to a model off this computer (Claude, or an Ollama server on
  another host) until the user has agreed (:func:`data_destination`, ``consent``).
"""
from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .claims import CLAIM_KINDS as _CLAIM_KINDS
from .claims import ActionRecord, ClaimReport, SimRecord, check_answer, claimed_actions
from .config import AssistantConfig, load_config
from .experiment import (
    ExperimentLog,
    GoalTarget,
    SimFacts,
    attribution_problems,
    design_mismatches,
    fmt_seconds,
    goal_targets,
    numbers_in,
)
from .plaintext import AnswerStream, plain_answer
from .prompts import (
    CANNOT_NUDGE,
    CLAIM_HINT_DELETE,
    CLAIM_HINT_DELETE_TOOL,
    CLAIM_HINT_EDIT,
    CLAIM_NUDGE,
    CLAIM_WITHHELD,
    CLAIMS_NUDGE,
    CONCLUSION_NUDGE,
    CONFIDENCE_NUDGE,
    COVERAGE_NUDGE,
    DESIGN_NUDGE,
    EMPTY_NUDGE,
    FOLLOW_UP,
    NO_ANSWER,
    NO_TOOLS_NUDGE,
    PLAN_FIRST,
    REFERENCE_NUDGE,
    REFERENCE_SETUP_NUDGE,
    STDERR_NUDGE,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_LOCAL,
    VALIDITY,
    goal_message,
)
from .providers.base import Cancelled, ProviderError, ToolCall, ToolResult, Turn
from .router import asks_reference, gives_comparison, route, says_cannot
from .tools import Approval, InputError, Toolbox, ToolContext, ToolOutcome, result_json, run_items
from .workspace import Workspace

EventFn = Callable[[str, dict], None]
ApproveFn = Callable[[Approval, str], bool]
ConsentFn = Callable[[str], bool]


@dataclass
class Outcome:
    """How a request ended (also sent as the "done" event)."""
    status: str            # "done" | "unverified" | "stopped" | "failed" | "refused"
    text: str
    untraced: list[str]
    path: str
    #: the claims check of the final answer (:class:`~aerolume.ai.claims.ClaimReport`), when there was one
    claims: dict = field(default_factory=dict)


#: the words of the local provider's context warning (providers/ollama.py), shown once per experiment
_CONTEXT_FULL = "context is nearly full"

_ASKS_NEW_RUN = re.compile(
    r"\b(?:new|fresh)\s+(?:[\w-]+\s+){0,3}?(?:simulations?|runs?|cfd|computations?|calculations?)\b"
    r"|\b(?:re-?run|run\s+(?:it\s+|them\s+)?again|from\s+scratch)\b", re.IGNORECASE)


_MENTIONS_REUSE = re.compile(r"\breus\w*|\bstored result|\bexisting result|\bnot (?:been )?(?:run|re-?run) "
                             r"again|\balready (?:had|computed|stored|in the project)", re.IGNORECASE)


def asks_new_run(goal: str) -> bool:
    """Whether a goal asks for a new run ("Run a new draft simulation", "re-run it"), not a stored result.

    The API returns the stored result of an identical, complete simulation instead of
    running it again; for "run a new draft simulation of the cavity" that answered with
    a result computed earlier, said only in the run card (critic round 3.2).
    """
    return bool(_ASKS_NEW_RUN.search(goal or ""))


def _deny_all(_approval: Approval, _call_id: str) -> bool:
    return False


_LOOPBACK = ("localhost", "127.0.0.1", "::1", "0:0:0:0:0:0:0:1")


def data_destination(provider: Any) -> str:
    """Where the conversation goes when it leaves this computer, or "" when it stays local.

    Claude: "Anthropic (Claude API)" (or the custom gateway in ANTHROPIC_BASE_URL).
    Ollama: "" on this computer (localhost / 127.x), else "the Ollama server at
    <endpoint>". Test and scripted providers keep everything local. Lumen's rule,
    adopted here: data is sent to a model only with the user's consent.
    """
    name = getattr(provider, "name", "")
    if name == "claude":
        import os
        base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
        custom = bool(base) and "api.anthropic.com" not in base       # a gateway, not the Claude API itself
        return f"Anthropic (Claude API) through {base}" if custom else "Anthropic (Claude API)"
    if name == "ollama":
        from urllib.parse import urlparse
        endpoint = str(getattr(provider, "endpoint", "") or "")
        host = (urlparse(endpoint).hostname or "").lower()
        if not host or host in _LOOPBACK or host.startswith("127."):
            return ""
        return f"the Ollama server at {endpoint}"
    return ""


def consent_key(provider: Any) -> str:
    """The identity a remembered consent is stored under ("claude", "ollama:http://host:11434")."""
    name = getattr(provider, "name", "")
    if name == "ollama":
        return f"ollama:{getattr(provider, 'endpoint', '')}"
    return str(name)


_ASKS_CONFIDENCE = re.compile(r"\b(confiden\w*|trust\w*|uncertaint\w*|reliab\w*|accura\w*|error bars?|"
                              r"how (sure|good|close)|margin of error)\b", re.IGNORECASE)
_STATES_UNCERTAINTY = re.compile(r"±|\+/-|\+-\s*\d|\bstandard error\b|\bresolution\b|\bhalf (?:of )?(?:the|a|an) "
                                 r"(?:\d+(?:\.\d+)?\s*°?\s*)?(?:angle )?step\b", re.IGNORECASE)
# a steady value has no ± to give: "steady", with the grid study (GCI) that bounds its discretisation error,
# is its uncertainty statement (critic round 3.3: the run's round-off was quoted as "± 0.00")
_STEADY_WORD = re.compile(r"\bsteady\b", re.IGNORECASE)
_GRID_WORDS = re.compile(r"\bGCI\b|grid (?:study|convergence|refinement)|discreti[sz]ation", re.IGNORECASE)


_STDERR_AS_ACCURACY = re.compile(
    r"\b(?:confiden\w*|accura\w*|precis\w*|reliab\w*|certain\w*|correct)\s+(?:to\s+|within\s+|of\s+|at\s+)*"
    r"(?:±|\+/-|\+-)"
    r"|within\s*(?:±|\+/-|\+-)\s*[\d.]+(?:\s+\S+){0,4}?\s+(?:due to|from|because of|given by|set by)\s+"
    r"(?:the\s+|its\s+)?(?:statistical\s+)?standard error", re.IGNORECASE)


def treats_stderr_as_accuracy(answer: str) -> bool:
    """Whether an answer presents a standard error as the accuracy ("I am confident within ±0.00455")."""
    return bool(_STDERR_AS_ACCURACY.search(answer or ""))


def asks_confidence(goal: str) -> bool:
    """Whether a goal asks how far the answer can be trusted ("how confident", "should I trust", "accuracy")."""
    return bool(_ASKS_CONFIDENCE.search(goal or ""))


def states_uncertainty(answer: str) -> bool:
    """Whether an answer gives an uncertainty figure: a ±, a standard error, or the angle resolution.

    A steady value has none to give: an answer that calls it steady and names the
    grid study (GCI, discretisation error) that bounds it states its uncertainty,
    wherever in the answer the two are said.
    """
    text = answer or ""
    return bool(_STATES_UNCERTAINTY.search(text)) or bool(_STEADY_WORD.search(text) and _GRID_WORDS.search(text))


class Agent:
    """One experiment with one model (see the module docstring)."""

    def __init__(self, provider: Any, workspace: Workspace, *, config: AssistantConfig | None = None,
                 toolbox: Toolbox | None = None, approve: ApproveFn | None = None,
                 on_event: EventFn | None = None, context: str = "", consent: ConsentFn | None = None) -> None:
        self.provider = provider
        self.ws = workspace
        self.config = (config or load_config()).validated()
        self.toolbox = toolbox or Toolbox()
        self.approve: ApproveFn = approve or _deny_all
        #: asked once, before anything is sent to a model off this computer (``data_destination``); None
        #: allows only destinations the preferences already have consent for
        self.consent = consent
        self.on_event = on_event
        self.context = context          # the window's selection, prepended to the first goal
        self.cancel = threading.Event()
        self.session = None
        self.log: ExperimentLog | None = None
        self.compact = getattr(provider, "name", "") == "ollama"
        self.tools_run = 0
        self._lock = threading.Lock()
        self._busy = False
        #: read calls answered so far: (name, canonical input) -> the action number that answered it
        self._reads: dict[tuple[str, str], int] = {}
        #: every quantity a result of this experiment marked not validated (for the standard-error follow-up)
        self._not_validated: list[str] = []
        #: quantities a tool result marked steady (their standard error is round-off, not an uncertainty)
        self._steady: list[str] = []
        #: validated reference set-up entries of the results ("Nusselt number: validated set-up ...")
        self._validated: list[str] = []
        #: every validated_quantities entry of the results (documented validation, e.g. a polar's cl)
        self._model_validated: list[str] = []
        #: the reference checks the results carried (reference_check, compare_reference rows)
        self._references: list[dict] = []
        #: the reference a goal asked to compare with that the final answer still leaves out ("" = complete)
        self._incomplete = ""
        self._loop_calls = 0
        #: streamed answer text on its way to plain Unicode (inline LaTeX math, see plaintext.py)
        self._math = AnswerStream()
        #: the local model's "context is nearly full" warning was shown (once per experiment is enough)
        self._context_warned = False
        #: the goal asked for a new run: simulations re-run so far (None: identical stored results are reused)
        self._fresh: set[str] | None = None
        #: names of the stored results this experiment reused instead of running (said once at the end)
        self._reused: list[str] = []

    # -- events -----------------------------------------------------------------------
    def _emit(self, event: str, /, **data: Any) -> None:
        """Send one event to on_event (positional-only, so data may carry a "kind" key)."""
        if self.on_event is None:
            return
        try:
            self.on_event(event, data)
        except Exception:  # a UI failure must not end the experiment
            import logging
            logging.getLogger(__name__).exception("assistant event %s failed", event)

    def _stream(self, kind: str, data: dict) -> None:
        if kind == "text":
            out = self._math.feed(str(data.get("delta", "")))
            if out:
                self._emit("text", delta=out)
        elif kind == "thinking":
            self._emit(kind, delta=data.get("delta", ""))
        elif kind == "notice":
            self._emit("notice", text=data.get("text", ""), level="info")
        elif kind == "tool_start":
            self._emit("status", text=f"Preparing {data.get('name', 'a tool call')}")

    # -- control -------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def experiment_path(self) -> str:
        return str(self.log.path) if self.log is not None else ""

    def stop(self) -> None:
        """Stop the model, the running tool and any wait for approval (thread-safe)."""
        self.cancel.set()
        abort = getattr(self.session, "abort", None)
        if callable(abort):
            abort()

    # -- the loop -------------------------------------------------------------------------
    def run(self, text: str) -> Outcome:
        """Send ``text`` (the goal, or a follow-up) and work until the model has answered.

        Blocking; call it on a worker thread. Never raises for model or tool
        problems: they end the request with an "error"/"done" event and a
        failed or stopped log status.
        """
        with self._lock:
            if self._busy:
                raise RuntimeError("the assistant is already working on a request")
            self._busy = True
        self.cancel.clear()
        t0 = time.monotonic()
        try:
            return self._run(text, t0)
        finally:
            self._busy = False

    def _consented(self) -> bool:
        """Whether the conversation may go to the provider: it stays local, or the user agreed."""
        dest = data_destination(self.provider)
        if not dest or consent_key(self.provider) in self.config.consent:
            return True
        if self.consent is None:
            return False
        try:
            return bool(self.consent(dest))
        except Exception:  # noqa: BLE001 - a broken consent prompt is a "no"
            return False

    def _run(self, text: str, t0: float) -> Outcome:
        first = self.session is None
        if first and not self._consented():
            dest = data_destination(self.provider)
            msg = (f"Nothing was sent: the experiment needs your permission to send the goal, the project's design "
                   f"and simulation names and the tools' results to {dest}. Allow it, or choose the local model "
                   "in Preferences > Assistant.")
            self._emit("notice", text=msg, level="warning")
            self._emit("status", text="Not sent")
            out = Outcome("stopped", msg, [], "")
            self._emit("done", status="stopped", text=msg, untraced=[], path="", withheld=[], claims={})
            return out
        try:
            if first:
                self._start(text)
                overview = self._overview_for_local_model()
                turn = self._send(user_text=goal_message(text, selected=self.context, overview=overview))
            else:
                self.log.record_message("user", text)
                turn = self._send(user_text=text)
            nudged = design_nudged = plan_nudged = claim_nudged = coverage_nudged = claims_nudged = False
            # per request: "run a new simulation" re-runs each simulation once instead of returning its stored
            # result; a follow-up that does not ask for that reads stored results again
            self._fresh = set() if asks_new_run(text) else None
            self._reused = []
            empty_nudged = cannot_nudged = reference_nudged = False
            self._incomplete = ""
            claimed: list[str] = []            # answers that made a false claim (withheld if it is repeated)
            followed_up: set[str] = set()
            rounds = 0
            while True:
                if turn.stop_reason == "refusal":
                    return self._finish("refused", turn.refusal, t0)
                if not turn.tool_calls:
                    cut = turn.stop_reason in ("max_tokens", "context")
                    if cut:
                        why = ("the model's reply hit its token limit" if turn.stop_reason == "max_tokens"
                               else "the conversation filled the model's context window")
                        self._emit("notice", text=f"The answer is incomplete: {why}. Start a new experiment or ask "
                                                  "for a shorter answer.", level="warning")
                    if not cut and not (turn.text or "").strip():
                        # an empty reply used to end "Done" with the green check and nothing to read (qwen3:8b,
                        # live, after its record_conclusion was refused): ask once, then end failed, saying so
                        if empty_nudged:
                            self._emit("notice", text=NO_ANSWER, level="warning")
                            return self._finish("failed", NO_ANSWER, t0, role="aerolume")
                        empty_nudged = True
                        self._emit("notice", text="The model ended without an answer; asking it to answer the goal.",
                                   level="warning")
                        self.log.record_message("aerolume", EMPTY_NUDGE)
                        turn = self._send(user_text=EMPTY_NUDGE)
                        continue
                    if not cut and not cannot_nudged and says_cannot(turn.text):
                        hint = self._route_hint(text)
                        if hint:
                            # a refusal of something a tool does (qwen3:8b: "the tools cannot run a simulation for
                            # a differentially heated square cavity", 7.5 s, accepted as the answer): once
                            cannot_nudged = True
                            self._emit("notice", text="The model says this cannot be done, but create_case sets it "
                                                      "up; asking it to use the tools.", level="warning",
                                       corrects=True)
                            turn = self._correct(turn.text, CANNOT_NUDGE.format(calls=hint))
                            continue
                    if (not nudged and self.tools_run == 0 and turn.text and self.log is not None
                            and self.log.untraced_in(turn.text)):
                        nudged = True
                        self._emit("notice", text="The model answered with numbers without running anything; "
                                                  "asking it to use the tools.", level="warning", corrects=True)
                        turn = self._correct(turn.text, NO_TOOLS_NUDGE)
                        continue
                    false = self._false_claims(turn.text)
                    if false:
                        claimed.append(turn.text)
                        if claim_nudged:
                            return self._withhold(false, claimed, t0)
                        claim_nudged = True
                        quoted = "; ".join(f"“{s}”" for _k, s in false)
                        # the panel line reads as one sentence: the quoted claim without its own full stop
                        # (“… as requested.”, but no tool call ...)
                        shown = "; ".join(f"“{s.rstrip(' .!')}”" for _k, s in false)
                        self._emit("notice", text=f"The answer says {shown}, but no tool call in this experiment "
                                                  "did that; asking the model to correct it.", level="warning",
                                   corrects=True)
                        hint = ""
                        if any(k == "delete" for k, _s in false):
                            can_delete = not self.compact and self.toolbox.get("delete_item") is not None
                            hint = CLAIM_HINT_DELETE_TOOL if can_delete else CLAIM_HINT_DELETE
                        if any(k == "edit" for k, _s in false):
                            hint += CLAIM_HINT_EDIT
                        turn = self._correct(turn.text, CLAIM_NUDGE.format(claims=quoted, hint=hint))
                        continue
                    used = self._used_designs()
                    wrong = design_mismatches(turn.text, used)
                    if wrong and not design_nudged:
                        design_nudged = True
                        self._emit("notice", text=f"The answer names {', '.join(wrong)}, but this experiment ran "
                                                  f"{', '.join(used)}; asking the model to correct it.",
                                   level="warning", corrects=True)
                        turn = self._correct(turn.text, DESIGN_NUDGE.format(named=", ".join(wrong),
                                                                            used=", ".join(used)))
                        continue
                    missing = [] if cut or coverage_nudged else self._uncovered_targets(text)
                    if missing:
                        coverage_nudged = True
                        what = ", ".join(t.text() for t in missing)
                        calls = [self._call_for(t) for t in missing]
                        verb = "read" if all(c.startswith("Read ") for c in calls) else "run"
                        self._emit("notice", text=f"The goal asks about {what}, but no run or result of this "
                                                  f"experiment matches; asking the model to {verb} "
                                                  f"{'it' if len(missing) == 1 else 'them'}.", level="warning",
                                   corrects=True)
                        turn = self._correct(turn.text, COVERAGE_NUDGE.format(
                            targets=what, it="it" if len(missing) == 1 else "them", calls="; ".join(calls)))
                        continue
                    self._incomplete = ""            # judged on this answer, not an earlier one
                    named, ref_case = asks_reference(text)
                    if named and not cut:
                        checks = self._reference_checks()
                        if not gives_comparison(turn.text, checks):
                            if not reference_nudged:
                                reference_nudged = True
                                turn = self._ask_comparison(turn.text, named, ref_case, text, checks)
                                continue
                            self._incomplete = named
                    report = self.check_claims(turn.text)
                    if report.problems and not claims_nudged and not cut:
                        # one correction request listing every unsupported or contradicted statement, with the
                        # value a misattributed number should have had (the 26.47-for-29.52 case)
                        claims_nudged = True
                        self._emit("notice", text=f"Claims check: {report.summary_line()}. Asking the model to "
                                                  "correct the statements that are not supported.",
                                   level="warning", corrects=True)
                        turn = self._correct(turn.text, CLAIMS_NUDGE.format(problems=report.correction_note()))
                        continue
                    items = ([] if cut else self._missing_steps(text, turn.text, followed_up))
                    if items:
                        followed_up.update(k for k, _v in items)
                        note = FOLLOW_UP.format(items="; and ".join(v for _k, v in items))
                        self._emit("notice", text=self._follow_up_notice([k for k, _v in items]), level="info",
                                   corrects=True)
                        self.log.record_message("assistant", turn.text)
                        self.log.record_message("aerolume", note)
                        turn = self._send(user_text=note)
                        continue
                    return self._finish("done", turn.text, t0, report=report)
                rounds += 1
                if rounds > self.config.max_steps:
                    self._emit("notice", text=f"Stopped after {self.config.max_steps} tool rounds (the limit in "
                                              "Preferences > Assistant).", level="warning")
                    return self._finish("stopped", turn.text, t0)
                if turn.text.strip():
                    self.log.record_message("assistant", turn.text)
                if not plan_nudged and self._needs_plan_first(turn.tool_calls):
                    plan_nudged = True
                    self._emit("notice", text="Asking the model to record its plan before it runs anything.",
                               level="info")
                    self.log.record_message("aerolume", PLAN_FIRST)
                    results = [self._defer(call) if self._kind(call.name) == "run" else self._execute(call)
                               for call in turn.tool_calls]
                else:
                    results = [self._execute(call) for call in turn.tool_calls]
                if self.cancel.is_set():
                    raise Cancelled()
                turn = self._send(tool_results=results)
        except Cancelled:
            return self._finish("stopped", "Stopped.", t0)
        except ProviderError as e:
            self._emit("error", text=str(e), kind=e.kind)
            return self._finish("failed", str(e), t0)

    def _start(self, goal: str) -> None:
        cfg_model = getattr(self.provider, "model", "")
        self.log = ExperimentLog(self.ws.root, goal, provider=getattr(self.provider, "name", ""),
                                 model=self.provider.describe() if hasattr(self.provider, "describe") else cfg_model)
        # The validity record both system prompts carry is the app's own documented evidence, and the prompt asks
        # the model to quote it. A local model's compact results replace the validity sentences with short codes,
        # so its quote "validated against NACA 0012/2412/4412 data" was "4412 is in no tool result" and a correct
        # answer ended unverified (live qwen3:8b test, 1 Nov 2026). Its numbers rank below every tool result.
        self.log.add_source(VALIDITY, origin="validity", tool="the solvers' validity record")
        self.ws.experiments_changed()
        system = SYSTEM_PROMPT_LOCAL if self.compact else SYSTEM_PROMPT
        self.session = self.provider.start(system, self.toolbox.specs(compact=self.compact))

    def _send(self, *, user_text: str | None = None, tool_results: list[ToolResult] | None = None) -> Turn:
        if self.cancel.is_set():
            raise Cancelled()
        self._emit("status", text="Waiting for the model")
        self._math = AnswerStream()
        try:
            turn = self.session.send(user_text=user_text, tool_results=tool_results, on_event=self._stream,
                                     cancel=self.cancel)
        finally:
            rest = self._math.flush()           # what a span still open at the end of the reply held back
            if rest:
                self._emit("text", delta=rest)
        # LaTeX math becomes Unicode for everything downstream: the claims check, the log, the panel's
        # final answer ("$1 \times 10^5$" read as code), and a "value ± error" is quoted as the Results
        # tables quote it (PDG rule, plaintext.round_pm); the same plain_answer made the streamed text
        turn.text = plain_answer(turn.text or "")
        for n in turn.notices:
            if _CONTEXT_FULL in n:
                # Ollama reports the prompt size on every turn, and the warning was printed four times in one
                # 34 s experiment: it says what to do, so once per experiment is enough
                if self._context_warned:
                    continue
                self._context_warned = True
            self._emit("notice", text=n, level="warning" if "context" in n or "cut off" in n else "info")
        usage = dict(self.log.record.usage) if self.log is not None else {}
        for k, v in (turn.usage or {}).items():
            usage[k] = usage.get(k, 0) + int(v)
        if self.log is not None:
            self.log.set_usage(usage)
        self._emit("turn", text=turn.text, thinking=turn.thinking, usage=turn.usage, served_by=turn.served_by)
        return turn

    # -- steps a small model skips -----------------------------------------------------
    def _kind(self, name: str) -> str:
        tool = self.toolbox.get(name)
        return tool.kind if tool is not None else ""

    def _overview_for_local_model(self) -> str:
        """Run project_overview for a local model and return its result for the goal message ("" for Claude).

        The prompt's "call project_overview first" is not enough for an
        8-billion-parameter model: qwen3:8b went straight to run_wing and
        recomputed a polar the project already held. Ollama has no forced tool
        choice, so the loop makes the call itself; the user sees it as the
        first row of the transcript, and the log records it like any call.
        """
        if not self.compact or self.toolbox.get("project_overview") is None:
            return ""
        result = self._execute(ToolCall("aerolume-overview", "project_overview", {}), counted=False)
        return "" if result.is_error else result.content

    def _needs_plan_first(self, calls: list[ToolCall]) -> bool:
        """A local model is starting a run without having recorded a plan (DESIGN.md 9: plan, then run)."""
        if not self.compact or self.log is None or self.log.record.plan is not None:
            return False
        names = [c.name for c in calls]
        return "record_plan" not in names and any(self._kind(n) == "run" for n in names)

    def _defer(self, call: ToolCall) -> ToolResult:
        """The answer to a run call held back until the plan is recorded (nothing runs, no row is shown)."""
        return ToolResult(call.id, call.name, json.dumps({"error": PLAN_FIRST}), is_error=True)

    def _missing_steps(self, request: str, answer: str, done: set[str]) -> list[tuple[str, str]]:
        """What a final answer still lacks, as (key, request) pairs; each key is asked for once per request.

        * ``conclusion`` - a local model that read or produced results but never
          called record_conclusion (the log would have no findings table);
        * ``confidence`` - any model whose answer to "how confident / how much
          to trust" carries no ± figure, resolution or standard error: qwen3:8b
          answered "the result is confident because it is validated" for a
          best-L/D angle that the 1° step fixes only to ±0.5°.
        """
        if self.log is None:
            return []
        actions = list(self.log.record.actions)
        has_results = any(a.status == "ok" and a.simulation_ids for a in actions)
        out: list[tuple[str, str]] = []
        concluded = any(a.tool == "record_conclusion" and a.status == "ok" for a in actions)
        if self.compact and has_results and not concluded and "conclusion" not in done:
            out.append(("conclusion", CONCLUSION_NUDGE))
        if has_results and "confidence" not in done and asks_confidence(request) and not states_uncertainty(answer):
            out.append(("confidence", CONFIDENCE_NUDGE))
        if "stderr" not in done and self._not_validated and treats_stderr_as_accuracy(answer):
            out.append(("stderr", STDERR_NUDGE.format(not_validated=", ".join(self._not_validated[:3]))))
        return out

    @staticmethod
    def _follow_up_notice(keys: list[str]) -> str:
        what = {"conclusion": "record its conclusion", "confidence": "state how confident the answer is",
                "stderr": "stop presenting a standard error as the accuracy"}
        return "Asking the model to " + " and to ".join(what[k] for k in keys) + "."

    # -- answer checks ----------------------------------------------------------------------
    def _correct(self, answer: str, note: str) -> Turn:
        """Log an answer and the loop's correction request, and send the request."""
        self.log.record_message("assistant", answer)
        self.log.record_message("aerolume", note)
        return self._send(user_text=note)

    def _false_claims(self, answer: str) -> list[tuple[str, str]]:
        """Actions the answer claims (:func:`claimed_actions`) that no successful tool call of that kind did."""
        if self.log is None:
            return []
        done_kinds = {self._kind(a.tool) for a in list(self.log.record.actions) if a.status == "ok"}
        return [(k, s) for k, s in claimed_actions(answer) if not done_kinds & set(_CLAIM_KINDS[k])]

    def _withhold(self, claims: list[tuple[str, str]], answers: list[str], t0: float) -> Outcome:
        """End with the answer replaced: it repeated a claimed action after one correction request."""
        facts = []
        kinds = {k for k, _s in claims}
        if "delete" in kinds:
            facts.append("Nothing was deleted. To delete a design or simulation, select it in the Project tree and "
                         "press Delete.")
        if "run" in kinds:
            facts.append("No analysis ran in this experiment.")
        if "create" in kinds:
            facts.append("Nothing was created in this experiment.")
        if "edit" in kinds:
            facts.append("No setting was changed: a suggestion changes a simulation only when you click Apply on "
                         "its card.")
        quoted = "; ".join(f"“{s}”" for _k, s in claims)
        said = {"delete": "something was deleted", "run": "an analysis was run", "create": "something was created",
                "edit": "settings were changed"}
        text = CLAIM_WITHHELD.format(claims=" and ".join(said[k] for k in sorted(kinds)), facts=" ".join(facts))
        if self.log is not None:
            self.log.record_message("assistant", answers[-1])
            self.log.add_check(f"Action check: the answer said {quoted}, but no tool call did that; it was "
                               "withheld after one correction request.")
        return self._finish("unverified", text, t0, withheld=answers)

    def _uncovered_targets(self, goal: str) -> list[GoalTarget]:
        """(design, condition) pairs the goal names that no run or read of this experiment matched."""
        if self.log is None:
            return []
        store = self.ws.store
        designs = {d.name: d for d in list(store.designs.values())}
        targets = goal_targets(goal, list(designs))
        if not targets:
            return []
        seen = [store.simulations.get(sid) for a in list(self.log.record.actions) if a.status == "ok"
                for sid in a.simulation_ids]
        sims = [s for s in seen if s is not None]
        return [t for t in targets if not any(self._matches(s, t, designs[t.design]) for s in sims)]

    @staticmethod
    def _matches(sim: Any, t: GoalTarget, design: Any) -> bool:
        if sim.design_id != design.id:
            return False
        c = sim.conditions
        if t.speed is not None and abs(float(c.speed) - t.speed) > 1e-6 + 0.005 * t.speed:
            return False
        if t.alpha is not None:
            if sim.analysis in ("polar", "wing"):
                st = sim.airfoil if sim.analysis == "polar" else sim.wing
                if not (st.alpha_start - 1e-6 <= t.alpha <= st.alpha_end + 1e-6):
                    return False
            elif abs(float(c.alpha_deg) - t.alpha) > 1e-6:
                return False
        if t.re is not None:
            try:
                re_sim = float(c.reynolds(float(design.length_ref)))
            except (TypeError, ValueError, AttributeError):
                return False
            if abs(re_sim - t.re) > 0.02 * t.re:
                return False
        return True

    def _call_for(self, t: GoalTarget) -> str:
        """The tool call that covers a goal target: read a stored match, else run it.

        "Read simulation 570fea75ccc5 (“Wing polar · 25 m/s”) with get_results"
        when the project already holds a finished run that matches, else
        "Call run_wing on Tapered wing at 15 m/s".
        """
        store = self.ws.store
        d = next((x for x in list(store.designs.values()) if x.name == t.design), None)
        if d is not None:
            for s in reversed(list(store.simulations.values())):
                if s.status == "done" and self._matches(s, t, d):
                    return f"Read simulation {s.id} (“{s.name}”) with get_results"
        kind = getattr(d, "kind", "")
        tool = "run_wing" if kind == "wing" else "run_polar" if kind == "airfoil" else "run_cfd"
        return f"Call {tool} on {t.text()}"

    def _sim_facts(self) -> list[SimFacts]:
        """What the attribution check knows about every simulation this experiment ran or read."""
        from .tools import summarize_result
        if self.log is None:
            return []
        out: list[SimFacts] = []
        ids = list(dict.fromkeys(sid for a in list(self.log.record.actions) if a.status == "ok"
                                 for sid in a.simulation_ids))
        for sid in ids:
            with contextlib.suppress(Exception):     # a result that cannot be read is simply not checked
                r = self.ws.project.result(sid)
                if r is None:
                    continue
                design = r.design.raw if r.design is not None else None
                s = summarize_result(r.raw, r.simulation, design)
                head = list(numbers_in(s.get("metrics") or {}))
                for st in (s.get("coefficients") or {}).values():
                    head += numbers_in(st)
                if s.get("l_over_d") is not None:
                    head.append(float(s["l_over_d"]))
                out.append(SimFacts(sid, f"“{r.simulation.name}”", getattr(design, "name", ""),
                                    float(r.simulation.conditions.speed), numbers_in(s), head))
        return out

    def _attribution(self, text: str) -> list[str]:
        return attribution_problems(text, self._sim_facts())

    # -- reference comparisons and the keyword router ---------------------------------------
    def _used_ids(self) -> list[str]:
        if self.log is None:
            return []
        return list(dict.fromkeys(sid for a in list(self.log.record.actions) if a.status == "ok"
                                  for sid in a.simulation_ids))

    def _reference_checks(self) -> list[dict]:
        """The applicable reference checks of the simulations this experiment ran or read (tools.reference_summary)."""
        from .tools import reference_summary
        out: list[dict] = []
        for sid in self._used_ids():
            with contextlib.suppress(Exception):         # a result that cannot be read is simply not checked
                r = self.ws.project.result(sid)
                if r is None:
                    continue
                chk = reference_summary(r.raw, r.simulation, r.design.raw if r.design is not None else None)
                if chk and chk.get("applies"):
                    out.append({"simulation_id": sid, "name": r.simulation.name, **chk})
        return out

    def _tried(self, *tools: str) -> bool:
        """Whether this experiment called any of ``tools`` (successfully or not)."""
        return self.log is not None and any(a.tool in tools for a in list(self.log.record.actions))

    def _route_hint(self, goal: str, case: str = "") -> str:
        """The router's calls for a goal, or "" when it has none or the model already tried that route."""
        r = route(goal, case)
        if r is None or self.toolbox.get("create_case") is None:
            return ""
        if self._tried("create_case", "run_simulation", "compare_reference"):
            return ""                   # it tried: a refusal after a failed run is the honest answer
        return r.text()

    def _ask_comparison(self, answer: str, named: str, case: str, goal: str, checks: list[dict]) -> Turn:
        """Send the answer back once: with the compare_reference numbers, or with the calls that produce them."""
        if checks:
            # the loop makes the call itself, so the numbers are a logged tool result (traceable by the claims
            # check) and the user sees the row; the model gets them in the note
            self._loop_calls += 1
            self._execute(ToolCall(f"aerolume-reference-{self._loop_calls}", "compare_reference",
                                   {"simulation_ids": [c["simulation_id"] for c in checks]}), counted=False)
            parts = []
            for c in checks:
                se = c.get("stderr")
                bit = (f"“{c['name']}” ({c['simulation_id']}): {c['quantity']} {c['measured']}"
                       + (" (steady: no averaging error; the discretisation error needs a grid study)"
                          if c.get("steady") else f" ± {se} (standard error)" if isinstance(se, (int, float)) else "")
                       + f", reference {c['reference']} ({c['source']}), difference {c['difference_pct']:+.3g} %")
                if c.get("kind") == "consistency":
                    bit += " -- a consistency check (the value is imposed), not a validation"
                parts.append(bit)
            note = REFERENCE_NUDGE.format(named=named, checks="; ".join(parts))
            msg = f"The goal asks for a comparison with {named}; the answer gives none. Sending the reference check."
        else:
            r = route(goal, case)
            hint = r.text() if r is not None and self.toolbox.get("create_case") is not None else ""
            note = REFERENCE_SETUP_NUDGE.format(named=named, calls=hint or "Say which run would answer it and why "
                                                                          "none of this experiment's runs does.")
            msg = (f"The goal asks for a comparison with {named}, but no run of this experiment has a published "
                   "reference; asking the model to set one up.")
        self._emit("notice", text=msg, level="warning", corrects=True)
        return self._correct(answer, note)

    # -- the claims check -----------------------------------------------------------------
    def _sim_records(self) -> list[SimRecord]:
        """Every simulation this experiment ran or read, with all its values and its headline ones by name."""
        from .tools import physics_models, summarize_result
        if self.log is None:
            return []
        out: list[SimRecord] = []
        ids = list(dict.fromkeys(sid for a in list(self.log.record.actions) if a.status == "ok"
                                 for sid in a.simulation_ids))
        for sid in ids:
            with contextlib.suppress(Exception):     # a result that cannot be read is simply not checked
                r = self.ws.project.result(sid)
                if r is None:
                    continue
                design = r.design.raw if r.design is not None else None
                s = summarize_result(r.raw, r.simulation, design)
                head: dict[str, float] = {}
                for k, v in (s.get("metrics") or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        head[k] = float(v)
                for k, st in (s.get("coefficients") or {}).items():
                    if isinstance(st, dict) and isinstance(st.get("mean"), (int, float)):
                        head[k] = float(st["mean"])
                if isinstance(s.get("nusselt"), dict) and isinstance(s["nusselt"].get("mean"), (int, float)):
                    head["nusselt"] = float(s["nusselt"]["mean"])
                if s.get("l_over_d") is not None:
                    head["l_over_d"] = float(s["l_over_d"])
                out.append(SimRecord(sid, f"“{r.simulation.name}”", getattr(design, "name", ""),
                                     float(r.simulation.conditions.speed), numbers_in(s), head,
                                     physics=physics_models(r.simulation, r.raw)))
        return out

    def _action_records(self) -> list[ActionRecord]:
        """The experiment's tool calls with the designs they concern (for claimed runs, creations, deletions)."""
        if self.log is None:
            return []
        store = self.ws.store
        names = sorted((d.name for d in list(store.designs.values())), key=len, reverse=True)
        out = []
        for a in list(self.log.record.actions):
            designs: list[str] = []
            for sid in a.simulation_ids:
                sim = store.simulations.get(sid)
                d = store.designs.get(sim.design_id) if sim is not None else None
                if d is not None and d.name not in designs:
                    designs.append(d.name)
            text = f"{a.summary} {json.dumps(a.input, default=str, ensure_ascii=False)}".lower()
            for n in names:
                if n and n.lower() in text and n not in designs:
                    designs.append(n)
            out.append(ActionRecord(a.tool, self._kind(a.tool), a.status, designs,
                                    reused="stored result reused" in (a.summary or "")))
        return out

    def check_claims(self, answer: str) -> ClaimReport:
        """The claims check of an answer against this experiment's tool calls and results (see claims.py)."""
        if self.log is None:
            return ClaimReport()
        store = self.ws.store
        return check_answer(answer, sources=self.log.labelled_sources(), sims=self._sim_records(),
                            actions=self._action_records(), designs=[d.name for d in list(store.designs.values())],
                            not_validated=list(self._not_validated), steady=list(self._steady),
                            validated=list(self._validated), references=list(self._references),
                            model_validated=list(self._model_validated))

    def _execute(self, call: ToolCall, *, counted: bool = True) -> ToolResult:
        """Validate, ask when needed, run one tool call; always returns its result for the model.

        ``counted=False`` (the overview the loop makes itself) leaves
        :attr:`tools_run` alone, so "answered without running anything" still
        means the model ran nothing.
        """
        tool = self.toolbox.get(call.name)
        args, problems = self.toolbox.prepare(call)
        title = call.name
        self._emit("tool_start", id=call.id, name=call.name, args=args if args is not None else call.input,
                   title=title)
        if problems:
            from .experiment import Action, _now
            if not isinstance(call.input, dict) and call.input is not None:
                content = json.dumps({"INVALID_JSON": call.raw if call.raw is not None else json.dumps(call.input,
                                                                                                    default=str)})
            else:
                content = json.dumps({"error": "Invalid input: " + "; ".join(problems)})
            if self.log is not None:
                self.log.record_action(Action(_now(), call.name, call.input if isinstance(call.input, dict) else
                                              {"raw": str(call.raw or call.input)[:500]}, "invalid",
                                              "; ".join(problems)[:300]))
            self._emit("tool_end", id=call.id, name=call.name, status="invalid", summary="; ".join(problems),
                       files=[], seconds=0.0, args=call.input)
            return ToolResult(call.id, call.name, content, is_error=True)

        key = (call.name, json.dumps(args, sort_keys=True, default=str))
        if tool is not None and tool.kind == "read" and key in self._reads:
            return self._repeated(call, args, self._reads[key], counted)
        ctx = ToolContext(self.ws, self.log, self.cancel,
                          lambda f, m: self._emit("tool_progress", id=call.id, fraction=f, message=m),
                          getattr(self.provider, "release_gpu", None), compact=self.compact, fresh=self._fresh)
        try:
            approval = self.toolbox.approval(call.name, args, ctx, self.config.approval_threshold_s)
        except InputError as exc:
            # the call's own input is wrong: answer with that and run nothing (fail closed, no question)
            return self._refused(call, args, str(exc))
        except Exception as exc:  # noqa: BLE001 - preparing the question failed: ask anyway, never run unasked
            if tool is None or tool.kind not in ("run", "code", "delete"):
                approval = None
            else:
                approval = Approval(tool.kind, f"{call.name}: cost unknown",
                                    f"Aerolume could not prepare this approval ({type(exc).__name__}: {exc}); "
                                    "approve to run it anyway.", None,
                                    items=run_items(call.name, args, ctx, None, None) if tool.kind == "run" else [])
        if approval is not None:
            self._emit("approval", id=call.id, approval=approval)
            self._emit("status", text="Waiting for your approval")
            ok = False if self.cancel.is_set() else bool(self.approve(approval, call.id))
            if self.cancel.is_set():
                raise Cancelled()
            if not ok:
                from .experiment import Action, _now
                if self.log is not None:
                    self.log.record_action(Action(_now(), call.name, dict(args), "declined", approval.title))
                self._emit("tool_end", id=call.id, name=call.name, status="declined",
                           summary="Declined by the user", files=[], seconds=0.0, args=args)
                return ToolResult(call.id, call.name, json.dumps(
                    {"error": "Declined by the user. Do not ask for the same action again; choose a cheaper "
                              "alternative or explain what you would need."}), is_error=True)
        self._emit("status", text=f"Running {call.name}")
        t0 = time.monotonic()
        outcome: ToolOutcome = self.toolbox.run(call.name, args, ctx)
        if counted:
            self.tools_run += 1
        status = "ok"
        if outcome.is_error:
            status = "cancelled" if self.cancel.is_set() else "error"
        else:
            kind = tool.kind if tool is not None else ""
            pending = any("analysis" not in r for r in outcome.content.get("results", []) if isinstance(r, dict))
            if kind == "read" and not pending:      # "no result yet" may change while the experiment runs
                self._reads[key] = len(self.log.record.actions) if self.log is not None else 0
            elif kind in ("run", "create", "delete", "code"):
                self._reads.clear()          # the project changed: a repeated read may now answer differently
            if kind == "run" and outcome.content.get("reused_stored_result"):
                name = str(outcome.content.get("name") or outcome.summary.split(":")[0]).strip()
                if name and name not in self._reused:
                    self._reused.append(name)
            self.observe(outcome.content)
        extra: dict[str, Any] = {}
        if call.name == "record_plan" and not outcome.is_error and self.log is not None and self.log.record.plan:
            # the plan as recorded (the planner's expected time), not as the model wrote it: the card shows this
            extra["plan"] = dict(self.log.record.plan)
        if tool is not None and tool.kind == "suggest" and not outcome.is_error:
            extra["suggestion"] = dict(outcome.content)      # the panel shows it as a card: old -> new, Apply
        self._emit("tool_end", id=call.id, name=call.name, status=status, summary=outcome.summary,
                   files=[str(p) for p in outcome.files], seconds=time.monotonic() - t0, args=args,
                   kind=tool.kind if tool is not None else "", **extra)
        return ToolResult(call.id, call.name, result_json(outcome), is_error=outcome.is_error)

    def _refused(self, call: ToolCall, args: dict, why: str) -> ToolResult:
        """Answer a call whose input the approval check found wrong, without running anything."""
        from .experiment import Action, _now
        if self.log is not None:
            self.log.record_action(Action(_now(), call.name, dict(args), "error", why[:300]))
        self._emit("tool_end", id=call.id, name=call.name, status="error", summary=why, files=[], seconds=0.0,
                   args=args)
        return ToolResult(call.id, call.name, json.dumps({"error": why}), is_error=True)

    def _repeated(self, call: ToolCall, args: dict, number: int, counted: bool) -> ToolResult:
        """Answer a read call that repeats an earlier one exactly, without sending its payload again.

        qwen3:8b called get_results twice with the same ids and filled its 8k
        context (8,025 of 8,192 tokens) with the second copy. Nothing changed
        in between (a run, a new design or a deletion clears this cache), so
        the earlier result still stands and is still in the conversation.
        """
        from .experiment import Action, _now
        summary = f"Same call as action {number}: answered from the earlier result"
        if self.log is not None:
            self.log.record_action(Action(_now(), call.name, dict(args), "ok", summary))
        if counted:
            self.tools_run += 1
        self._emit("tool_end", id=call.id, name=call.name, status="ok", summary=summary, files=[], seconds=0.0,
                   args=args, kind="read")
        where = ("the project overview in the first message"
                 if self.compact and number <= 1 and call.name == "project_overview"
                 else f"the result of the identical call earlier in this conversation (action {number})")
        return ToolResult(call.id, call.name, json.dumps(
            {"data": {"repeated_call": True,
                      "note": f"Nothing changed since: use {where}. Do the next step instead of repeating it."}}))

    def observe(self, content: Any) -> None:
        """Remember what a successful tool result says about validity, for the claims check.

        The steady quantities, the not-validated and validated quantities and the
        reference checks (with their bands). The MCP server calls this for every
        result of its session, so its check_claims judges an answer exactly as
        the in-app assistant does.
        """
        self._collect_not_validated(content)
        self._collect_steady(content)
        self._collect_references(content)

    def _collect_references(self, obj: Any) -> None:
        """Remember every reference check of a tool result (any depth): a dict that ``applies`` with a
        ``difference_pct`` (reference_summary's shape), once per simulation and quantity."""
        if isinstance(obj, dict):
            if obj.get("applies") is True and isinstance(obj.get("difference_pct"), (int, float)):
                key = (obj.get("simulation_id"), obj.get("quantity"), obj.get("measured"))
                if all((r.get("simulation_id"), r.get("quantity"), r.get("measured")) != key
                       for r in self._references):
                    self._references.append(dict(obj))
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    self._collect_references(v)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_references(v)

    def _collect_steady(self, obj: Any, key: str = "") -> None:
        """Remember every quantity a tool result marks ``steady`` (any depth): its ± would be round-off.

        The name is the key the statistic sits under ("nusselt", "cd"), or the
        ``quantity`` of a reference check; the claims check flags a ± given for one.
        """
        if isinstance(obj, dict):
            if obj.get("steady") is True:
                name = str(obj.get("quantity") or key)
                if name and name not in self._steady:
                    self._steady.append(name)
            for k, v in obj.items():
                if k == "steady_quantities" and isinstance(v, list):
                    for q in v:
                        if str(q) not in self._steady:
                            self._steady.append(str(q))
                elif isinstance(v, (dict, list)):
                    self._collect_steady(v, str(k))
        elif isinstance(obj, list):
            for v in obj:
                self._collect_steady(v, key)

    def _collect_not_validated(self, obj: Any) -> None:
        """Remember every "not_validated" quantity a tool result lists (any depth)."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "not_validated" and isinstance(v, list):
                    for q in v:
                        if str(q) not in self._not_validated:
                            self._not_validated.append(str(q))
                elif k == "validated_quantities" and isinstance(v, list):
                    for q in v:
                        if "validated set-up" in str(q):
                            if str(q) not in self._validated:
                                self._validated.append(str(q))
                        elif str(q) not in self._model_validated:
                            self._model_validated.append(str(q))
                else:
                    self._collect_not_validated(v)
        elif isinstance(obj, list):
            for v in obj:
                self._collect_not_validated(v)

    def _used_designs(self) -> list[str]:
        """Names of the designs whose simulations this experiment ran or read (in order of first use)."""
        if self.log is None:
            return []
        store = self.ws.store
        names: list[str] = []
        for a in list(self.log.record.actions):
            if a.status != "ok":
                continue
            for sid in a.simulation_ids:
                sim = store.simulations.get(sid)
                d = store.designs.get(sim.design_id) if sim is not None else None
                if d is not None and d.name not in names:
                    names.append(d.name)
        return names

    def _say_reused(self, answer: str) -> None:
        """Say under the answer which results were reused, when the answer does not.

        A stored result of an identical run answers the question just as well, but a
        reader of "the simulation was completed" assumes it ran now. The run card's
        "(stored result reused)" was the only trace (critic round 3.2). The log needs
        no extra line: its action already reads "(stored result reused)".
        """
        if not self._reused or _MENTIONS_REUSE.search(answer or ""):
            return
        names = ", ".join(f"“{n}”" for n in self._reused[:4]) + (" …" if len(self._reused) > 4 else "")
        many = len(self._reused) > 1
        note = (f"Not run again: {names} {'are' if many else 'is'} the stored result{'s' if many else ''} of "
                f"{'identical runs' if many else 'an identical run'} computed earlier. Ask for a new run to "
                "compute it again.")
        self._emit("notice", text=note, level="info")

    def _finish(self, status: str, text: str, t0: float, *, withheld: list[str] | None = None,
                report: ClaimReport | None = None, role: str = "assistant") -> Outcome:
        """End the request: log the outcome, emit "status" and "done", return the :class:`Outcome`.

        ``role`` is who said ``text`` in the log's conversation when the request did
        not end "done" ("aerolume" for the loop's own explanation, such as the
        no-answer message, so the log never puts words in the model's mouth).
        """
        untraced: list[str] = []
        claims: dict = {}
        if self.log is not None:
            if status == "done":
                if report is None:
                    report = self.check_claims(text)
                if self._incomplete:
                    # the goal asked for a comparison the answer still leaves out after one request with the numbers
                    status = "unverified"
                    check = (f"Completeness check: the goal asks for a comparison with {self._incomplete}, and the "
                             "answer gives no reference value or difference (compare_reference has them).")
                    self.log.add_check(check)
                    self._emit("notice", text=check, level="warning")
                if report.problems:
                    # numbers or actions the tools do not support remain after the correction request: the
                    # answer is shown with them flagged, and the experiment does not get the green check
                    status = "unverified"
                    self.log.add_check(f"Claims check: {report.summary_line()}. Not supported: "
                                       f"{report.correction_note()}")
                claims = report.to_dict()
                self.log.set_claims(claims)
                self.log.record_message("assistant", text)
                self._say_reused(text)
                untraced = self.log.conclude(text, status=status)
                used = self._used_designs()
                wrong = design_mismatches(text, used)
                if wrong:
                    check = (f"Design check: the answer names {', '.join(wrong)}, but this experiment's runs used "
                             f"{', '.join(used)}.")
                    self.log.add_check(check)
                    self._emit("notice", text=check, level="warning")
                misattributed = self._attribution(text)
                if misattributed:
                    check = "Attribution check: " + "; ".join(misattributed) + "."
                    self.log.add_check(check)
                    self._emit("notice", text=check, level="warning")
            elif status == "unverified":
                self.log.record_message("aerolume", text)
                self.log.conclude(text, status="unverified")
            else:
                if text and status != "stopped":
                    self.log.record_message(role, text)
                self.log.set_status("failed" if status in ("failed", "refused") else status)
            self.ws.experiments_changed()
        if status == "done" and untraced:
            self._emit("notice", text=("Number check: " + ", ".join(untraced[:8]) + (" …" if len(untraced) > 8 else "")
                                       + " do not appear in any tool result (they may be derived; see the log)."),
                       level="warning")
        took = fmt_seconds(time.monotonic() - t0)
        self._emit("status", text={"done": f"Done in {took}", "unverified": f"Answer not verified · {took}",
                                   "stopped": "Stopped", "failed": "Failed",
                                   "refused": "Declined by the model"}.get(status, status))
        out = Outcome(status, text, untraced, self.experiment_path, claims)
        self._emit("done", status=status, text=text, untraced=untraced, path=out.path, withheld=list(withheld or []),
                   claims=claims)
        return out


__all__ = ["Agent", "ApproveFn", "ConsentFn", "EventFn", "Outcome", "asks_confidence", "claimed_actions",
           "consent_key", "data_destination", "states_uncertainty", "treats_stderr_as_accuracy"]
