"""Claude through the official ``anthropic`` SDK (optional dependency: ``aerolume[ai]``).

Request shape (claude-api skill, Python; every choice below is from it):

* ``client.beta.messages.stream(...)`` - streaming, read to the end with
  ``get_final_message()``; the beta namespace is needed for the server-side
  refusal fallback: ``fallbacks="default"`` with beta header
  ``server-side-fallback-2026-07-01`` (the scalar form and its own header;
  the array form uses a different header and the two must not be mixed). On a
  policy decline the API re-runs the request on Anthropic's recommended model
  for that category inside the same call.
* model ``claude-opus-5-5``; ``thinking={"type": "adaptive", "display":
  "summarized"}`` (thinking is always on for this model; the summaries are what
  the chat shows as "Reasoning"); ``output_config={"effort": "high"}``.
* Prompt caching: the system prompt is one text block with an explicit
  ``cache_control`` breakpoint (tools render before it, so tools + system are
  cached together), and top-level automatic caching covers the growing
  conversation. The system prompt and the tool list never change within a
  session and tools are sorted by name, so the prefix stays byte-identical.
* Client tools carry ``eager_input_streaming: True`` (streamed requests); the
  API then does not validate the input and the SDK's tolerant parser can return
  a truncated object, so the agent validates every input against its schema
  before running anything (:mod:`aerolume.ai.schema`). JSON the SDK cannot parse
  at all raises ``ValueError`` from the stream iterator: the turn is re-issued
  (at most twice), since there is no tool-use id to answer.
* Stop reasons: ``refusal`` -> no tool runs, a clear message; ``max_tokens``
  with a tool call -> the truncated input is never run, the turn is retried
  once with twice the budget (nothing was appended, so history stays
  append-only); ``pause_turn`` -> the paused turn is appended and re-sent.
* History is append-only and assistant content is passed back unmodified, which
  Claude Opus 5.5's preserved thinking requires (an edited prefix makes replayed
  thinking blocks a 400). After a server-side fallback, blocks before the last
  ``fallback`` block other than text are not echoed (the documented rule), and
  only tool calls after it are executed.
* Errors are the SDK's typed exceptions, caught most specific first, and become
  :class:`~aerolume.ai.providers.base.ProviderError` with a plain sentence.

Credentials: an explicit key from Preferences (``api_key=``) when the user
typed one, otherwise the SDK's own chain (``ANTHROPIC_API_KEY`` /
``ANTHROPIC_AUTH_TOKEN`` / ``ant auth login`` profile). The key is never logged.
"""
from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

from .base import Cancelled, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, emit

FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MODEL = "claude-opus-5-5"
MAX_TOKENS_CAP = 128_000
JSON_RETRIES = 2
PAUSE_LIMIT = 5

#: block types that are never echoed when they precede the last fallback block
_NOT_ECHOED_BEFORE_FALLBACK = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}


def _custom_base_url() -> bool:
    """True when ``ANTHROPIC_BASE_URL`` points somewhere other than the Claude API.

    A proxy or gateway may reject fields the real API accepts
    (``eager_input_streaming``, ``fallbacks``), so those are left off there.
    """
    url = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    return bool(url) and "api.anthropic.com" not in url


def _block_type(block: Any) -> str:
    return block.get("type", "") if isinstance(block, dict) else str(getattr(block, "type", ""))


def _block_dict(block: Any) -> dict:
    """A JSON-safe dict of a content block (SDK model or dict) for the experiment log."""
    if isinstance(block, dict):
        d = dict(block)
    else:
        try:
            d = block.model_dump(mode="json", exclude_none=True, by_alias=True)
        except Exception:  # noqa: BLE001 - logging must never fail the run
            d = {"type": _block_type(block)}
    d.pop("signature", None)              # opaque, long, useless in a log
    if d.get("type") == "redacted_thinking":
        d.pop("data", None)
    return d


class ClaudeSession:
    """One conversation with Claude (see the module docstring for the request rules)."""

    def __init__(self, client: Any, model: str, system: str, tools: list[ToolSpec], *, effort: str,
                 max_tokens: int, fallbacks: bool, eager: bool, strict_tools: bool = False) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.fallbacks = fallbacks
        self._system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self._tools: list[dict] = []
        for spec in sorted(tools, key=lambda t: t.name):
            t: dict[str, Any] = {"name": spec.name, "description": spec.description,
                                 "input_schema": spec.input_schema}
            if eager:
                t["eager_input_streaming"] = True
            if strict_tools:
                t["strict"] = True
            self._tools.append(t)
        self._messages: list[dict] = []
        self.usage_total: dict[str, int] = {}
        self._stream: Any = None

    # -- requests -------------------------------------------------------------------
    def _request(self, max_tokens: int) -> dict:
        kw: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens, "system": self._system, "tools": self._tools,
            "messages": self._messages, "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.effort}, "cache_control": {"type": "ephemeral"},
        }
        if self.fallbacks:
            kw["betas"] = [FALLBACK_BETA]
            kw["fallbacks"] = "default"
        return kw

    def abort(self) -> None:
        """Close the open stream (from another thread): Stop while the model is silent.

        The cancel event is checked between stream events; before the first one
        (the model is reading a long prompt) nothing arrives for a while, so
        Stop also closes the response, which ends the blocked read at once.
        """
        stream = self._stream
        if stream is not None:
            with contextlib.suppress(Exception):     # already closed or finished
                stream.close()

    def _stream_once(self, max_tokens: int, on_event, cancel: threading.Event | None):
        """One streamed request; returns the final message. Raises ValueError for unparseable JSON."""
        tool_names: dict[int, str] = {}
        with self._client.beta.messages.stream(**self._request(max_tokens)) as stream:
            self._stream = stream
            try:
                for event in stream:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    et = getattr(event, "type", "")
                    if et == "content_block_start":
                        block = event.content_block
                        bt = _block_type(block)
                        if bt == "tool_use":
                            tool_names[event.index] = block.name
                            emit(on_event, "tool_start", id=block.id, name=block.name)
                        elif bt == "fallback":
                            frm = getattr(getattr(block, "from_", None), "model", "")
                            to = getattr(getattr(block, "to", None), "model", "")
                            emit(on_event, "notice", text=f"{frm or 'The model'} declined; "
                                                          f"{to or 'a fallback model'} continued the answer.")
                    elif et == "content_block_delta":
                        d = event.delta
                        dt = getattr(d, "type", "")
                        if dt == "text_delta":
                            emit(on_event, "text", delta=d.text)
                        elif dt == "thinking_delta":
                            emit(on_event, "thinking", delta=d.thinking)
                        elif dt == "input_json_delta":
                            emit(on_event, "tool_input", name=tool_names.get(event.index, ""), delta=d.partial_json)
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                return stream.get_final_message()
            except Exception:
                # abort() closed the response under the reader: whatever the read raised, it was a Stop
                if cancel is not None and cancel.is_set():
                    raise Cancelled() from None
                raise
            finally:
                self._stream = None

    def _call(self, max_tokens: int, on_event, cancel) -> Any:
        """``_stream_once`` with the SDK's typed errors turned into ProviderError (most specific first)."""
        try:
            import anthropic
        except ImportError as exc:      # pragma: no cover - checked by ClaudeProvider.check
            raise ProviderError("The anthropic package is not installed (pip install aerolume[ai]).",
                                "unavailable") from exc
        try:
            return self._stream_once(max_tokens, on_event, cancel)
        except (Cancelled, ValueError):
            raise
        except anthropic.AuthenticationError as e:
            raise ProviderError("Claude rejected the credentials (401). Check the API key in "
                                "Preferences > Assistant or ANTHROPIC_API_KEY.", "auth",
                                request_id=getattr(e, "request_id", None)) from None
        except anthropic.PermissionDeniedError as e:
            raise ProviderError(f"This API key may not use {self.model} (403): {e.message}", "permission",
                                request_id=getattr(e, "request_id", None)) from None
        except anthropic.NotFoundError as e:
            raise ProviderError(f"Model {self.model!r} was not found or is not available to this "
                                "organisation (404). Pick another model in Preferences > Assistant.", "not_found",
                                request_id=getattr(e, "request_id", None)) from None
        except anthropic.RateLimitError as e:
            after = ""
            with contextlib.suppress(Exception):     # a response without headers still is a 429
                after = e.response.headers.get("retry-after", "")
            raise ProviderError("Claude's rate limit was reached (429)" + (f"; retry in {after} s." if after else
                                                                           "; try again shortly."),
                                "rate_limit", retryable=True, request_id=getattr(e, "request_id", None)) from None
        except anthropic.RequestTooLargeError as e:
            raise ProviderError("The conversation is too large for one request (413). Start a new "
                                "experiment.", "too_large", request_id=getattr(e, "request_id", None)) from None
        except anthropic.BadRequestError as e:
            raise ProviderError(f"Claude rejected the request (400): {e.message}", "bad_request",
                                request_id=getattr(e, "request_id", None)) from None
        except anthropic.OverloadedError as e:
            raise ProviderError("Claude is overloaded (529); try again in a minute.", "overloaded",
                                retryable=True, request_id=getattr(e, "request_id", None)) from None
        except anthropic.InternalServerError as e:
            raise ProviderError(f"Claude had a server error ({e.status_code}); try again.", "server",
                                retryable=True, request_id=getattr(e, "request_id", None)) from None
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Claude API error {e.status_code}: {e.message}", "server",
                                retryable=e.status_code >= 500, request_id=getattr(e, "request_id", None)) from None
        except anthropic.APIConnectionError:          # includes APITimeoutError
            raise ProviderError("Could not reach the Claude API (network error or timeout).", "connection",
                                retryable=True) from None
        except anthropic.CredentialsError as e:
            raise ProviderError(f"No usable Claude credentials: {e}", "credentials") from None
        except TypeError as e:
            if "authentication" in str(e).lower():
                raise ProviderError("No Claude credentials found. Set ANTHROPIC_API_KEY, run `ant auth login`, "
                                    "or enter a key in Preferences > Assistant.", "credentials") from None
            raise

    # -- the conversation -------------------------------------------------------------
    def send(self, *, user_text: str | None = None, tool_results: list[ToolResult] | None = None,
             on_event=None, cancel: threading.Event | None = None) -> Turn:
        if user_text is not None:
            self._messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        elif tool_results:
            self._messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": bool(r.is_error)}
                for r in tool_results]})
        else:
            raise ValueError("send() needs user_text or tool_results")

        max_tokens = self.max_tokens
        json_failures = pauses = 0
        notices: list[str] = []
        while True:
            try:
                message = self._call(max_tokens, on_event, cancel)
            except ValueError:
                # tool input the SDK could not parse at all: no tool_use id to answer, re-issue the turn
                json_failures += 1
                if json_failures > JSON_RETRIES:
                    raise ProviderError("Claude's tool input could not be parsed three times in a row.",
                                        "protocol") from None
                emit(on_event, "notice", text="The tool input could not be parsed; asking again.")
                continue
            stop = getattr(message, "stop_reason", None) or "end_turn"
            content = list(getattr(message, "content", None) or [])
            has_tool = any(_block_type(b) == "tool_use" for b in content)
            if stop == "pause_turn" and pauses < PAUSE_LIMIT:
                pauses += 1
                self._messages.append({"role": "assistant", "content": self._echo(content)})
                continue
            if stop == "max_tokens" and has_tool and max_tokens < MAX_TOKENS_CAP:
                # a truncated tool input parses as a valid partial object: never run it; ask again
                # with more room (nothing was appended, so the history is unchanged)
                max_tokens = min(2 * max_tokens, MAX_TOKENS_CAP)
                notices.append(f"The reply was cut off at the token limit; retried with {max_tokens} tokens.")
                emit(on_event, "notice", text=notices[-1])
                continue
            break
        return self._finish(message, content, stop, notices)

    def _echo(self, content: list) -> list:
        """Assistant content to pass back: unchanged, except the documented post-fallback omissions."""
        last_fb = max((i for i, b in enumerate(content) if _block_type(b) == "fallback"), default=-1)
        if last_fb < 0:
            return content
        out = []
        for i, b in enumerate(content):
            bt = _block_type(b)
            if bt == "fallback":
                continue                    # an audit marker; may be dropped
            if i < last_fb and bt in _NOT_ECHOED_BEFORE_FALLBACK:
                continue
            out.append(b)
        return out

    def _finish(self, message: Any, content: list, stop: str, notices: list[str]) -> Turn:
        usage = getattr(message, "usage", None)
        turn_usage: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            v = getattr(usage, key, None) if usage is not None else None
            if isinstance(v, int):
                turn_usage[key] = v
                self.usage_total[key] = self.usage_total.get(key, 0) + v
        iterations = getattr(usage, "iterations", None) or []
        fell_back = any(getattr(it, "type", "") == "fallback_message" for it in iterations)
        served = str(getattr(message, "model", "") or self.model)
        if fell_back and stop != "refusal":
            notices.append(f"Answered by the fallback model {served}.")
        turn = Turn(stop_reason="end_turn", served_by=served, usage=turn_usage, notices=notices)
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            explanation = getattr(details, "explanation", None) if details is not None else None
            turn.stop_reason = "refusal"
            turn.refusal = ("Claude declined this request" + (f" (category: {category})" if category else "")
                            + (f": {explanation}" if explanation else ".")
                            + " Nothing it started was run. Rephrase the goal or switch to another model.")
            return turn                     # the refused content is not appended: no tool of it ever runs
        echoed = self._echo(content)
        self._messages.append({"role": "assistant", "content": echoed})
        last_fb = max((i for i, b in enumerate(content) if _block_type(b) == "fallback"), default=-1)
        texts, thoughts = [], []
        for i, b in enumerate(content):
            bt = _block_type(b)
            if bt == "text":
                texts.append(b.text if not isinstance(b, dict) else b.get("text", ""))
            elif bt == "thinking":
                t = b.thinking if not isinstance(b, dict) else b.get("thinking", "")
                if t:
                    thoughts.append(t)
            elif bt == "tool_use" and i > last_fb:
                bid = b.id if not isinstance(b, dict) else b.get("id", "")
                name = b.name if not isinstance(b, dict) else b.get("name", "")
                inp = b.input if not isinstance(b, dict) else b.get("input")
                turn.tool_calls.append(ToolCall(bid, name, inp))
        turn.text = "".join(texts)
        turn.thinking = "\n\n".join(thoughts)
        if stop == "max_tokens":
            turn.stop_reason = "max_tokens"
            turn.tool_calls = []            # a truncated input is never run
        elif stop == "model_context_window_exceeded":
            turn.stop_reason = "context"
            turn.tool_calls = []
        elif turn.tool_calls:
            turn.stop_reason = "tool_use"
        elif stop not in ("end_turn", "stop_sequence", "tool_use"):
            turn.stop_reason = "other"
        return turn

    def transcript(self) -> list[dict]:
        out = []
        for m in self._messages:
            content = m["content"]
            blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            out.append({"role": m["role"], "content": [_block_dict(b) for b in blocks]})
        return out


class ClaudeProvider:
    """Claude models through the Anthropic API (see the module docstring)."""

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, *, effort: str = "high", max_tokens: int = 64000,
                 api_key: str | None = None, client: Any = None, fallbacks: bool | None = None,
                 eager_streaming: bool | None = None, strict_tools: bool = False) -> None:
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self._api_key = api_key or None
        self._client = client
        custom = _custom_base_url()
        self.fallbacks = (not custom) if fallbacks is None else bool(fallbacks)
        self.eager_streaming = (not custom) if eager_streaming is None else bool(eager_streaming)
        self.strict_tools = strict_tools

    def describe(self) -> str:
        return f"Claude · {self.model}"

    def client(self) -> Any:
        """The SDK client (built on first use; the SDK's credential chain unless a key was given)."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError("Claude needs the optional anthropic package: pip install aerolume[ai]",
                                    "unavailable") from exc
            try:
                self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
            except anthropic.CredentialsError as e:
                raise ProviderError(f"No usable Claude credentials: {e}", "credentials") from None
        return self._client

    def check(self) -> tuple[bool, str]:
        if self._client is not None:
            return True, f"{self.describe()} · ready"
        from ..config import anthropic_available, claude_credential_source
        if not anthropic_available():
            return False, "Claude needs the optional anthropic package (pip install aerolume[ai])."
        source = "preferences" if self._api_key else claude_credential_source()
        if not source:
            return False, ("No Claude credentials: set ANTHROPIC_API_KEY, run `ant auth login`, "
                           "or enter a key in Preferences > Assistant.")
        where = {"preferences": "key from Preferences", "environment": "key from the environment",
                 "profile": "ant auth profile"}.get(source, source)
        return True, f"{self.describe()} · {where}"

    def start(self, system: str, tools: list[ToolSpec]) -> ClaudeSession:
        return ClaudeSession(self.client(), self.model, system, tools, effort=self.effort,
                             max_tokens=self.max_tokens, fallbacks=self.fallbacks, eager=self.eager_streaming,
                             strict_tools=self.strict_tools)

    def release_gpu(self) -> None:
        """Nothing to release: Claude runs remotely."""


__all__ = ["DEFAULT_MODEL", "FALLBACK_BETA", "ClaudeProvider", "ClaudeSession"]
