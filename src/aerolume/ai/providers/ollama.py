"""Local models through Ollama's HTTP API (``/api/chat`` with tools), standard library only.

Fully offline and open source: the model runs on this computer (qwen3:8b is the
default; any installed model whose capabilities include "tools" works) and no
data leaves it. No new dependency: ``urllib`` speaks the API.

Protocol notes (measured against Ollama 0.40):

* ``POST /api/chat`` with ``stream: true`` returns one JSON object per line;
  ``message.content`` / ``message.thinking`` carry text deltas, and a
  ``message.tool_calls`` list arrives complete in one line, each call
  ``{"id", "function": {"name", "arguments": {...}}}``. The last line has
  ``done: true``, ``done_reason`` and token counts.
* Tool results go back as ``{"role": "tool", "content", "tool_name"}`` messages,
  one per call, in the order of the calls.
* ``think`` switches a reasoning model's thinking on or off; off is several
  times faster for qwen3:8b and still calls tools well.
* ``options.num_ctx`` must be set: Ollama's default context is far smaller than
  the system prompt plus tool schemas. 8192 tokens keeps an 8-billion-parameter
  Q4 model (6.2 GB with its KV cache) entirely on an 8 GB GPU, but one run with
  its plan and two corrections filled it; the default is 16384 (7.8 GB, 80 % on
  an 8 GB GPU; see ``ai.config.DEFAULT_OLLAMA_NUM_CTX`` for the measurement).
* A model loaded for chat holds that GPU memory until ``keep_alive`` expires,
  which would starve a GPU CFD run: :meth:`OllamaProvider.release_gpu` unloads
  it (``keep_alive: 0``) before the agent starts one; the next turn reloads it
  in a few seconds.

Cancellation is checked between streamed lines; a model that is still loading
or reading a long prompt sends nothing for a while, so Stop takes effect at
the next token.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any

from .base import Cancelled, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, emit

DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
#: seconds to wait for the next streamed line (a cold model load on a laptop can take a minute)
READ_TIMEOUT_S = 300.0
CONTEXT_WARN_FRACTION = 0.85
#: retries when the model server crashes before answering (a shared GPU), and the pause between
SERVER_RETRIES = 1
RETRY_DELAY_S = 2.0


class _ServerCrash(Exception):
    """Ollama answered 5xx before streaming anything (its model runner died)."""


_RESOLVED: dict[str, str] = {}
_RESOLVED_LOCK = threading.Lock()


def _fast_endpoint(endpoint: str) -> str:
    """``endpoint`` with "localhost" replaced by 127.0.0.1 when Ollama answers there (cached per endpoint).

    Windows resolves "localhost" to ::1 first; Ollama listens on IPv4 only by
    default, so every request waited about 2 s for the IPv6 attempt to fail
    before falling back (measured: 2.06 s vs 0.001 s per /api/version). That
    was 2 s added to every model turn, and the Preferences page still said
    "Looking for Ollama…" seconds after opening. A server that really listens
    on ::1 only keeps the name.
    """
    with _RESOLVED_LOCK:
        hit = _RESOLVED.get(endpoint)
    if hit is not None:
        return hit
    out = endpoint
    parts = urllib.parse.urlsplit(endpoint)
    if (parts.hostname or "").lower() == "localhost":
        import socket
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                pass
            netloc = "127.0.0.1" + (f":{parts.port}" if parts.port else "")
            out = urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
        except OSError:
            return endpoint             # nothing on IPv4 (yet): ask again next time
    with _RESOLVED_LOCK:
        _RESOLVED[endpoint] = out
    return out


def _url(endpoint: str, path: str) -> str:
    return _fast_endpoint(endpoint.rstrip("/")) + path


def _get_json(endpoint: str, path: str, timeout: float = 3.0) -> Any:
    req = urllib.request.Request(_url(endpoint, path), headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:      # the user-configured local endpoint
        return json.loads(r.read().decode("utf-8"))


def _post(endpoint: str, path: str, body: dict, timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(_url(endpoint, path), data=data, method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"})
    return urllib.request.urlopen(req, timeout=timeout)          # the user-configured local endpoint


def server_version(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 2.0) -> str | None:
    """The Ollama version string, or None when no server answers at ``endpoint``."""
    try:
        d = _get_json(endpoint, "/api/version", timeout)
    except (OSError, ValueError):
        return None
    return str(d.get("version", "")) if isinstance(d, dict) else None


def list_models(endpoint: str = DEFAULT_ENDPOINT, *, tools_only: bool = True, timeout: float = 3.0) -> list[dict]:
    """Installed models as ``{"name", "size_gb", "parameters", "tools", "thinking"}``, tool-capable first.

    ``tools_only`` hides models that declare capabilities without "tools" (code
    completion models cannot run experiments). Empty when the server is down.
    """
    try:
        d = _get_json(endpoint, "/api/tags", timeout)
    except (OSError, ValueError):
        return []
    out = []
    for m in d.get("models", []) if isinstance(d, dict) else []:
        caps = m.get("capabilities")
        tools = caps is None or "tools" in caps          # older servers do not report capabilities
        if tools_only and not tools:
            continue
        details = m.get("details") or {}
        out.append({"name": str(m.get("name", "")), "size_gb": round(float(m.get("size", 0) or 0) / 1e9, 1),
                    "parameters": str(details.get("parameter_size", "")), "tools": tools,
                    "thinking": bool(caps and "thinking" in caps)})
    out.sort(key=lambda m: (not m["tools"], m["name"]))
    return out


def unload(endpoint: str, model: str, timeout: float = 10.0) -> bool:
    """Ask Ollama to free the model's memory now (``keep_alive: 0``); True when it answered."""
    try:
        with _post(endpoint, "/api/generate", {"model": model, "keep_alive": 0}, timeout) as r:
            r.read()
        return True
    except (OSError, ValueError):
        return False


def _arguments(raw: Any) -> tuple[Any, str | None]:
    """Tool arguments as parsed JSON (dict normally) and the raw text when it was a string."""
    if isinstance(raw, str):
        try:
            return json.loads(raw), raw
        except ValueError:
            return raw, raw
    return raw, None


class OllamaSession:
    """One conversation with a local model (see the module docstring)."""

    def __init__(self, endpoint: str, model: str, system: str, tools: list[ToolSpec], *, think: bool,
                 num_ctx: int, keep_alive: str = "10m") -> None:
        self.endpoint, self.model = endpoint, model
        self.think, self.num_ctx, self.keep_alive = think, int(num_ctx), keep_alive
        self._tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                         "parameters": t.input_schema}}
                       for t in sorted(tools, key=lambda t: t.name)]
        self._messages: list[dict] = [{"role": "system", "content": system}]
        self.prompt_tokens = 0
        self._resp = None

    def abort(self) -> None:
        """Close the open response (from another thread): Stop during a long prompt read."""
        resp = self._resp
        if resp is not None:
            with contextlib.suppress(Exception):     # already closed
                resp.close()

    def _request(self, body: dict, on_event, cancel: threading.Event | None):
        """One streamed /api/chat request: (text parts, thinking parts, calls, raw calls, final event)."""
        text: list[str] = []
        thinking: list[str] = []
        calls: list[ToolCall] = []
        raw_calls: list[dict] = []
        done: dict = {}
        try:
            with _post(self.endpoint, "/api/chat", body, READ_TIMEOUT_S) as resp:
                self._resp = resp
                for line in resp:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line.decode("utf-8"))
                    if ev.get("error"):
                        raise ProviderError(f"Ollama: {ev['error']}", "server")
                    msg = ev.get("message") or {}
                    if msg.get("thinking"):
                        thinking.append(msg["thinking"])
                        emit(on_event, "thinking", delta=msg["thinking"])
                    if msg.get("content"):
                        text.append(msg["content"])
                        emit(on_event, "text", delta=msg["content"])
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        args, raw = _arguments(fn.get("arguments"))
                        cid = str(tc.get("id") or f"call_{uuid.uuid4().hex[:10]}")
                        name = str(fn.get("name", ""))
                        calls.append(ToolCall(cid, name, args, raw))
                        raw_calls.append({"id": cid, "type": "function",
                                          "function": {"name": name, "arguments": args}})
                        emit(on_event, "tool_start", id=cid, name=name)
                    if ev.get("done"):
                        done = ev
                        break
        except (Cancelled, ProviderError):
            raise
        except urllib.error.HTTPError as e:
            detail = ""
            with contextlib.suppress(Exception):     # the error body is optional
                detail = json.loads(e.read().decode("utf-8")).get("error", "")
            if e.code == 404:
                raise ProviderError(f"Ollama has no model {self.model!r}: run `ollama pull {self.model}` or pick "
                                    "an installed model in Preferences > Assistant.", "not_found") from None
            if e.code >= 500 and not text and not calls:
                raise _ServerCrash(detail or str(e.reason)) from None
            raise ProviderError(f"Ollama error {e.code}: {detail or e.reason}", "server") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, AttributeError) as e:
            if cancel is not None and cancel.is_set():
                raise Cancelled() from None     # abort() closed the connection under the reader
            reason = getattr(e, "reason", e)
            if isinstance(e, TimeoutError) or isinstance(reason, TimeoutError) or "timed out" in str(reason):
                # The server answered the connection but not the request: measured with qwen3:8b (7.8 GB)
                # on an 8 GB GPU shared with a game, the model ran partly on the CPU and its first token
                # took over 300 s. "Not reachable" sent people to restart a server that was running.
                raise ProviderError(
                    f"Ollama at {self.endpoint} did not answer within {READ_TIMEOUT_S:.0f} s. The model "
                    f"{self.model!r} may still be loading, or may not fit in the GPU's free memory and be "
                    "running on the CPU: close other GPU programs, pick a smaller model in Preferences > "
                    "Assistant, or try again.", "connection", retryable=True) from None
            raise ProviderError(f"Ollama is not reachable at {self.endpoint} ({reason}). "
                                "Start Ollama, or switch to Claude in Preferences > Assistant.", "connection",
                                retryable=True) from None
        except ValueError as e:
            if cancel is not None and cancel.is_set():
                raise Cancelled() from None
            raise ProviderError(f"Ollama sent a reply that is not JSON: {e}", "protocol") from None
        finally:
            self._resp = None
        return text, thinking, calls, raw_calls, done

    def send(self, *, user_text: str | None = None, tool_results: list[ToolResult] | None = None,
             on_event=None, cancel: threading.Event | None = None) -> Turn:
        if user_text is not None:
            self._messages.append({"role": "user", "content": user_text})
        elif tool_results:
            for r in tool_results:
                self._messages.append({"role": "tool", "content": r.content, "tool_name": r.name,
                                       "tool_call_id": r.call_id})
        else:
            raise ValueError("send() needs user_text or tool_results")
        body = {"model": self.model, "messages": self._messages, "tools": self._tools, "stream": True,
                "think": bool(self.think), "keep_alive": self.keep_alive,
                "options": {"num_ctx": self.num_ctx, "temperature": 0.2}}
        for attempt in range(SERVER_RETRIES + 1):
            try:
                text, thinking, calls, raw_calls, done = self._request(body, on_event, cancel)
                break
            except _ServerCrash as crash:
                # the model server died before it answered (seen when the GPU is shared with a CFD
                # run that grabbed memory): nothing was streamed, so asking again is safe
                if attempt == SERVER_RETRIES:
                    raise ProviderError(f"Ollama could not run {self.model}: {crash}", "server") from None
                emit(on_event, "notice", text="The local model server stopped; trying again.")
                time.sleep(RETRY_DELAY_S)

        assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
        if raw_calls:
            assistant["tool_calls"] = raw_calls
        self._messages.append(assistant)
        usage = {"input_tokens": int(done.get("prompt_eval_count", 0) or 0),
                 "output_tokens": int(done.get("eval_count", 0) or 0)}
        self.prompt_tokens = usage["input_tokens"]
        turn = Turn(text="".join(text), thinking="".join(thinking), tool_calls=calls, served_by=self.model,
                    usage=usage)
        if calls:
            turn.stop_reason = "tool_use"
        elif done.get("done_reason") == "length":
            turn.stop_reason = "max_tokens"
        if self.prompt_tokens > CONTEXT_WARN_FRACTION * self.num_ctx:
            turn.notices.append(f"The local model's context is nearly full ({self.prompt_tokens:,} of "
                                f"{self.num_ctx:,} tokens): older messages may be dropped. Start a new "
                                "experiment, or raise the context size in Preferences > Assistant.")
        return turn

    def transcript(self) -> list[dict]:
        return [dict(m) for m in self._messages]


class OllamaProvider:
    """A local model served by Ollama (see the module docstring)."""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, endpoint: str = DEFAULT_ENDPOINT, *, think: bool = False,
                 num_ctx: int = 16384) -> None:
        self.model = model or DEFAULT_MODEL
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.think = think
        self.num_ctx = int(num_ctx)

    def describe(self) -> str:
        return f"Local · {self.model}"

    def check(self) -> tuple[bool, str]:
        version = server_version(self.endpoint)
        if version is None:
            return False, (f"Ollama is not running at {self.endpoint}. Install it from ollama.com and start it, "
                           "or switch to Claude in Preferences > Assistant.")
        names = [m["name"] for m in list_models(self.endpoint, tools_only=False)]
        if self.model not in names and f"{self.model}:latest" not in names:
            return False, f"Model {self.model} is not installed: run `ollama pull {self.model}`."
        return True, f"{self.describe()} · Ollama {version}"

    def start(self, system: str, tools: list[ToolSpec]) -> OllamaSession:
        return OllamaSession(self.endpoint, self.model, system, tools, think=self.think, num_ctx=self.num_ctx)

    def release_gpu(self) -> None:
        """Unload the model so a GPU simulation gets the memory (see the module docstring)."""
        unload(self.endpoint, self.model)


__all__ = ["DEFAULT_ENDPOINT", "DEFAULT_MODEL", "OllamaProvider", "OllamaSession", "list_models",
           "server_version", "unload"]
