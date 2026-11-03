"""Test doubles for the assistant: a scripted provider and a fake Ollama HTTP server.

Imported by the other ``test_ai_*`` modules (``import test_ai_fakes as fakes``);
the few tests at the bottom check the doubles themselves, so a broken fake
fails here rather than as a confusing failure elsewhere.

* :class:`ScriptedProvider` plays a fixed conversation: each step is a
  :class:`~aerolume.ai.providers.base.Turn` or a function of what the agent sent
  (tool results), so a test can react to real tool output.
* :class:`FakeOllama` is an ``http.server`` on a free localhost port speaking
  the subset of Ollama's API the provider uses (``/api/version``, ``/api/tags``,
  ``/api/chat`` streaming NDJSON, ``/api/generate`` for unloading). It records
  every request body.
* :class:`FakeClaudeAPI` is the Messages API in-process: a real
  ``anthropic.Anthropic`` client whose HTTP transport (``httpx2.MockTransport``)
  answers with scripted server-sent events or error responses. The SDK's own
  SSE parser, message accumulator, tolerant tool-JSON parser and typed
  exceptions all run, so the provider is tested against what the SDK really
  does rather than against a mock of it. No network, no key.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self

from aerolume.ai.providers.base import ToolCall, ToolResult, Turn, emit

# ---------------------------------------------------------------------------
# scripted provider (no network)
# ---------------------------------------------------------------------------

Step = Turn | Callable[["ScriptedSession", "str | None", "list[ToolResult] | None"], Turn]


class ScriptedSession:
    def __init__(self, provider: ScriptedProvider, system: str, tools: list) -> None:
        self.provider = provider
        self.system, self.tools = system, tools
        self.i = 0
        self.sent: list[dict] = []

    def send(self, *, user_text=None, tool_results=None, on_event=None, cancel=None) -> Turn:
        self.sent.append({"user_text": user_text, "tool_results": tool_results})
        if self.i >= len(self.provider.steps):
            return Turn(text="(no more scripted turns)", stop_reason="end_turn")
        step = self.provider.steps[self.i]
        self.i += 1
        turn = step(self, user_text, tool_results) if callable(step) else step
        if turn.text:
            emit(on_event, "text", delta=turn.text)
        if turn.thinking:
            emit(on_event, "thinking", delta=turn.thinking)
        return turn

    def transcript(self) -> list[dict]:
        return list(self.sent)

    def last_result(self, name: str | None = None) -> dict:
        """The JSON of the newest tool result (of ``name``) the agent sent back."""
        for sent in reversed(self.sent):
            for r in reversed(sent["tool_results"] or []):
                if name is None or r.name == name:
                    return json.loads(r.content)
        raise AssertionError(f"no tool result {name!r} was sent")


class ScriptedProvider:
    """A provider that plays ``steps`` (see the module docstring)."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, steps: list[Step], *, ready: bool = True) -> None:
        self.steps = list(steps)
        self.ready = ready
        self.sessions: list[ScriptedSession] = []
        self.released = 0

    def describe(self) -> str:
        return "Scripted · test"

    def check(self) -> tuple[bool, str]:
        return self.ready, "Scripted · test · ready" if self.ready else "Scripted provider unavailable"

    def start(self, system: str, tools: list) -> ScriptedSession:
        s = ScriptedSession(self, system, tools)
        self.sessions.append(s)
        return s

    def release_gpu(self) -> None:
        self.released += 1


def call(cid: str, name: str, args: Any = None) -> ToolCall:
    return ToolCall(cid, name, {} if args is None else args)


def tools_turn(*calls: ToolCall, text: str = "") -> Turn:
    return Turn(text=text, tool_calls=list(calls), stop_reason="tool_use")


def final(text: str) -> Turn:
    return Turn(text=text, stop_reason="end_turn")


def make_project(tmp_path, name: str = "AI test", *, naca: str | None = "2412"):
    """A headless :class:`aerolume.api.Project` (with one NACA airfoil unless ``naca`` is None)."""
    from aerolume import api
    p = api.new_project(tmp_path / f"{name}.aerolume", name)
    if naca:
        p.add_naca(naca)
    return p


# ---------------------------------------------------------------------------
# fake Ollama server
# ---------------------------------------------------------------------------

class FakeOllama:
    """``with FakeOllama(script) as srv: OllamaProvider(endpoint=srv.url)``.

    ``script`` is a list of chat replies, one per /api/chat request; each reply
    is a list of NDJSON events (dicts) to stream, or an int HTTP status to fail
    with. ``models`` is the /api/tags list; ``delay`` pauses between streamed
    events (cancellation tests).
    """

    def __init__(self, script: list | None = None, *, models: list[dict] | None = None, version: str = "0.40.2",
                 delay: float = 0.0) -> None:
        self.script = list(script or [])
        self.models = models if models is not None else [
            {"name": "qwen3:8b", "size": 5_225_388_164, "capabilities": ["completion", "tools", "thinking"],
             "details": {"parameter_size": "8.2B"}},
            {"name": "deepseek-coder-v2:16b", "size": 8_905_126_121, "capabilities": ["completion", "insert"],
             "details": {"parameter_size": "15.7B"}},
        ]
        self.version = version
        self.delay = delay
        self.requests: list[tuple[str, dict]] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def chat_bodies(self) -> list[dict]:
        return [b for p, b in self.requests if p == "/api/chat"]

    def __enter__(self) -> Self:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a) -> None:     # keep pytest output clean
                pass

            def _json(self, code: int, obj: Any) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/api/version":
                    self._json(200, {"version": fake.version})
                elif self.path == "/api/tags":
                    self._json(200, {"models": fake.models})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append((self.path, body))
                if self.path == "/api/generate":
                    self._json(200, {"model": body.get("model"), "done": True, "done_reason": "unload"})
                    return
                if self.path != "/api/chat":
                    self._json(404, {"error": "not found"})
                    return
                if not fake.script:
                    self._json(500, {"error": "no scripted reply left"})
                    return
                reply = fake.script.pop(0)
                if isinstance(reply, int):
                    self._json(reply, {"error": f"scripted failure {reply}"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    for ev in reply:
                        self.wfile.write((json.dumps(ev) + "\n").encode())
                        self.wfile.flush()
                        if fake.delay:
                            time.sleep(fake.delay)
                except (ConnectionError, OSError):
                    pass                          # the client went away (cancel)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def ollama_text(text: str, *, chunks: int = 3, prompt_tokens: int = 900) -> list[dict]:
    """NDJSON events of a plain text answer."""
    step = max(1, len(text) // chunks)
    evs = [{"message": {"role": "assistant", "content": text[i:i + step]}, "done": False}
           for i in range(0, len(text), step)]
    evs.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
                "prompt_eval_count": prompt_tokens, "eval_count": 40})
    return evs


def ollama_tools(*calls: tuple[str, dict], prompt_tokens: int = 800) -> list[dict]:
    """NDJSON events of a turn that calls tools (Ollama sends the calls complete in one event)."""
    tcs = [{"id": f"call_{i}", "function": {"index": i, "name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]
    return [{"message": {"role": "assistant", "content": "", "tool_calls": tcs}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
             "prompt_eval_count": prompt_tokens, "eval_count": 30}]


# ---------------------------------------------------------------------------
# fake Claude Messages API (the real SDK over a mock transport)
# ---------------------------------------------------------------------------

def claude_message(*blocks: tuple, stop: str = "end_turn", model: str = "claude-opus-5-5",
                   usage: dict | None = None, stop_details: dict | None = None,
                   iterations: list[dict] | None = None) -> list[dict]:
    """The stream events of one assistant message, built from simple block specs.

    Blocks: ``("thinking", text, signature)``, ``("text", text)``,
    ``("tool", id, name, partial_json_text)`` and ``("fallback", from_model,
    to_model)``. The events follow the documented stream: ``message_start``,
    per block ``content_block_start`` / ``*_delta`` / ``content_block_stop``,
    then ``message_delta`` (stop reason, stop details, usage) and ``message_stop``.
    """
    u = {"input_tokens": 1200, "output_tokens": 1, "cache_read_input_tokens": 0,
         "cache_creation_input_tokens": 0, **(usage or {})}
    evs: list[dict] = [{"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": u}}]
    for i, b in enumerate(blocks):
        kind = b[0]
        if kind == "thinking":
            evs.append({"type": "content_block_start", "index": i,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            evs.append({"type": "content_block_delta", "index": i,
                        "delta": {"type": "thinking_delta", "thinking": b[1]}})
            evs.append({"type": "content_block_delta", "index": i,
                        "delta": {"type": "signature_delta", "signature": b[2]}})
        elif kind == "text":
            evs.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            for k in range(0, len(b[1]), 12):
                evs.append({"type": "content_block_delta", "index": i,
                            "delta": {"type": "text_delta", "text": b[1][k:k + 12]}})
        elif kind == "tool":
            evs.append({"type": "content_block_start", "index": i, "content_block": {
                "type": "tool_use", "id": b[1], "name": b[2], "input": {}}})
            text = b[3]
            for k in range(0, max(len(text), 1), 9):
                evs.append({"type": "content_block_delta", "index": i,
                            "delta": {"type": "input_json_delta", "partial_json": text[k:k + 9]}})
        elif kind == "fallback":
            evs.append({"type": "content_block_start", "index": i, "content_block": {
                "type": "fallback", "from": {"model": b[1]}, "to": {"model": b[2]}}})
        else:
            raise ValueError(kind)
        evs.append({"type": "content_block_stop", "index": i})
    delta: dict[str, Any] = {"stop_reason": stop, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    end_usage: dict[str, Any] = {"output_tokens": 240}
    if iterations is not None:
        end_usage["iterations"] = iterations
    evs.append({"type": "message_delta", "delta": delta, "usage": end_usage})
    evs.append({"type": "message_stop"})
    return evs


class FakeClaudeAPI:
    """``FakeClaudeAPI(replies).client()`` is an ``anthropic.Anthropic`` that talks to a script.

    Each reply answers one request: a list of stream events (see
    :func:`claude_message`) or ``(status, error_type, message)`` for an API
    error (with ``retry-after`` on a 429). ``requests`` keeps every request
    body (parsed JSON) and ``headers`` its headers, so tests can check what the
    provider actually sent. ``on_event(i)`` is called before streaming event
    ``i`` (cancellation tests).
    """

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.on_event: Callable[[int], None] | None = None

    def _handle(self, request):
        import httpx2
        self.requests.append(json.loads(request.content.decode("utf-8")))
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        if not self.replies:
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error",
                                                                          "message": "no scripted reply left"}})
        reply = self.replies.pop(0)
        if isinstance(reply, tuple):
            status, etype, message = reply
            headers = {"request-id": "req_test_1"}
            if status == 429:
                headers["retry-after"] = "17"
            return httpx2.Response(status, headers=headers,
                                   json={"type": "error", "error": {"type": etype, "message": message}})
        fake = self

        def body():
            for i, ev in enumerate(reply):
                if fake.on_event is not None:
                    fake.on_event(i)
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()
        return httpx2.Response(200, headers={"content-type": "text/event-stream", "request-id": "req_test_ok"},
                               content=body())

    def client(self):
        import anthropic
        import httpx2
        return anthropic.Anthropic(api_key="sk-ant-test-0000000000000000",
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self._handle)),
                                   max_retries=0)


# ---------------------------------------------------------------------------
# the doubles themselves
# ---------------------------------------------------------------------------

def test_fake_ollama_serves_the_api_subset():
    import urllib.request
    with FakeOllama([ollama_text("hello")]) as srv:
        with urllib.request.urlopen(srv.url + "/api/version", timeout=5) as r:
            assert json.loads(r.read())["version"] == "0.40.2"
        req = urllib.request.Request(srv.url + "/api/chat", data=b'{"model": "m"}', method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            lines = [json.loads(x) for x in r.read().splitlines() if x.strip()]
        assert "".join(e["message"]["content"] for e in lines) == "hello" and lines[-1]["done"]
        assert srv.chat_bodies() == [{"model": "m"}]


def test_fake_claude_api_streams_through_the_real_sdk():
    import pytest
    pytest.importorskip("anthropic")        # the optional [ai] extra; CI installs only [dev]
    api = FakeClaudeAPI([claude_message(("text", "Hello there"), ("tool", "toolu_1", "x", '{"a": 1}'),
                                        stop="tool_use")])
    with api.client().beta.messages.stream(model="claude-opus-5-5", max_tokens=64,
                                            messages=[{"role": "user", "content": "hi"}]) as s:
        kinds = [ev.type for ev in s]
        msg = s.get_final_message()
    assert "content_block_delta" in kinds and msg.stop_reason == "tool_use"
    assert msg.content[0].text == "Hello there" and msg.content[1].input == {"a": 1}
    assert api.requests[0]["model"] == "claude-opus-5-5"


def test_scripted_provider_plays_its_steps():
    prov = ScriptedProvider([tools_turn(call("a", "project_overview")),
                             lambda s, _u, results: final(f"got {len(results)} result")])
    s = prov.start("system", [])
    t1 = s.send(user_text="goal")
    assert t1.stop_reason == "tool_use" and t1.tool_calls[0].name == "project_overview"
    t2 = s.send(tool_results=[ToolResult("a", "project_overview", '{"data": {}}')])
    assert t2.text == "got 1 result"
    assert s.last_result("project_overview") == {"data": {}}
