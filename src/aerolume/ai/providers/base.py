"""The provider-neutral conversation interface the agent loop talks to.

Claude (Anthropic Messages API) and local models (Ollama) store a conversation
in different shapes: content blocks with signed thinking blocks that must be
passed back byte for byte, versus plain role/content messages. The agent loop
must not care, so each provider hands out a :class:`ChatSession` that owns its
own transcript and exchanges only these neutral objects with the loop:

* the loop sends a user goal (:meth:`ChatSession.send` with ``user_text``) or
  the results of the tools the model asked for (``tool_results``, all of one
  turn together, as both APIs require);
* the session streams the reply through ``on_event(kind, data)`` callbacks
  ("text", "thinking", "tool_start", "tool_input", "notice") and returns a
  :class:`Turn`: the text, the tool calls and why the model stopped.

Errors cross the boundary as :class:`ProviderError` with a ``kind`` the GUI
turns into one plain sentence ("No Claude API key", "Ollama is not running").
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

EventFn = Callable[[str, dict], None]


class Cancelled(Exception):
    """The user pressed Stop while the model was answering."""


class ProviderError(Exception):
    """A model call that failed in a way the user can act on.

    ``kind``: "credentials", "auth", "permission", "not_found", "rate_limit",
    "overloaded", "connection", "bad_request", "server", "unavailable",
    "protocol" or "too_large". ``retryable`` is True when trying again later
    can work (rate limits, overload, network).
    """

    def __init__(self, message: str, kind: str = "server", *, retryable: bool = False,
                 request_id: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.request_id = request_id


@dataclass
class ToolSpec:
    """A tool as the model sees it: name, when to use it, and the JSON schema of its input."""
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    """One tool call from the model. ``input`` is whatever was parsed (validate it!);
    ``raw`` is the unparsed argument text when the provider had one."""
    id: str
    name: str
    input: Any
    raw: str | None = None


@dataclass
class ToolResult:
    """The answer to one :class:`ToolCall`: JSON text for the model, ``is_error`` for failures."""
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    """One model reply.

    ``stop_reason`` is normalised: "end_turn", "tool_use", "max_tokens",
    "refusal", "context" (the context window is full) or "other".
    ``refusal`` carries the provider's explanation when the model declined;
    ``served_by`` names the model that actually answered (a server-side
    fallback can hand a declined request to another model).
    """
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    refusal: str = ""
    served_by: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)


class ChatSession(Protocol):
    """A conversation with one model (see the module docstring)."""

    def send(self, *, user_text: str | None = None, tool_results: list[ToolResult] | None = None,
             on_event: EventFn | None = None, cancel: threading.Event | None = None) -> Turn: ...

    def transcript(self) -> list[dict]:
        """The conversation so far as JSON-safe dicts (for the experiment log)."""
        ...


class Provider(Protocol):
    """A model provider: describes itself, checks it can run, and opens sessions."""

    name: str
    model: str

    def describe(self) -> str: ...

    def check(self) -> tuple[bool, str]:
        """(ready, one-line status) without spending tokens."""
        ...

    def start(self, system: str, tools: list[ToolSpec]) -> ChatSession: ...


def emit(on_event: EventFn | None, kind: str, **data: Any) -> None:
    """Call ``on_event`` if given; a failing UI callback must never break the model call."""
    if on_event is None:
        return
    try:
        on_event(kind, data)
    except Exception:  # the transcript matters more than a repaint
        import logging
        logging.getLogger(__name__).exception("assistant event callback failed")


__all__ = ["Cancelled", "ChatSession", "EventFn", "Provider", "ProviderError", "ToolCall", "ToolResult",
           "ToolSpec", "Turn", "emit"]
