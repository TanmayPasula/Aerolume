"""The model providers on their wire protocols, without a network or a key.

Claude runs through the real ``anthropic`` SDK over a scripted transport
(:class:`test_ai_fakes.FakeClaudeAPI`): the tests check the request the
provider builds (claude-api skill: adaptive thinking with summaries, effort,
server-side fallbacks with their beta header, prompt caching, eager input
streaming) and how it handles what the API can send back (refusals, truncated
tool input, unparseable JSON, pause_turn, fallback switch points, typed
errors). Ollama runs against :class:`test_ai_fakes.FakeOllama`, a real HTTP
server on localhost.
"""
from __future__ import annotations

import json
import threading

import pytest
import test_ai_fakes as fakes

from aerolume.ai.providers.base import Cancelled, ProviderError, ToolResult, ToolSpec

anthropic = pytest.importorskip("anthropic")

from aerolume.ai.providers.claude import FALLBACK_BETA, ClaudeProvider
from aerolume.ai.providers.ollama import OllamaProvider, list_models, server_version

TOOLS = [ToolSpec("run_polar", "Run a polar.", {"type": "object", "properties": {
             "design_id": {"type": "string"}}, "required": ["design_id"], "additionalProperties": False}),
         ToolSpec("project_overview", "List the project.", {"type": "object", "properties": {}, "required": [],
                                                             "additionalProperties": False})]


def _claude(replies, **kw):
    api = fakes.FakeClaudeAPI(replies)
    prov = ClaudeProvider(client=api.client(), **kw)
    return api, prov, prov.start("SYSTEM PROMPT", TOOLS)


class Events:
    """Collects on_event callbacks."""

    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.items.append((kind, data))

    def kinds(self) -> list[str]:
        return [k for k, _ in self.items]

    def text(self, kind: str) -> str:
        return "".join(d.get("delta", "") for k, d in self.items if k == kind)


# ---------------------------------------------------------------------------
# Claude: the request
# ---------------------------------------------------------------------------

def test_claude_request_follows_the_skill_defaults():
    api, _prov, s = _claude([fakes.claude_message(("text", "Ready."))])
    s.send(user_text="goal")
    body, headers = api.requests[0], api.headers[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["stream"] is True
    # server-side refusal fallback: the scalar "default" form with its own beta header
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA == "server-side-fallback-2026-07-01"
    assert FALLBACK_BETA in headers.get("anthropic-beta", "")
    assert "server-side-fallback-2026-06-01" not in headers.get("anthropic-beta", "")
    # prompt caching: an explicit breakpoint on the stable system prompt plus automatic caching
    assert body["system"] == [{"type": "text", "text": "SYSTEM PROMPT", "cache_control": {"type": "ephemeral"}}]
    assert body["cache_control"] == {"type": "ephemeral"}
    # tools: sorted (a stable cached prefix), eagerly streamed, never forced
    assert [t["name"] for t in body["tools"]] == ["project_overview", "run_polar"]
    assert all(t["eager_input_streaming"] is True for t in body["tools"])
    assert "tool_choice" not in body and "temperature" not in body and "budget_tokens" not in json.dumps(body)
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "goal"}]}]
    assert "sk-ant-test" not in json.dumps(body)            # the key travels in a header only


def test_claude_custom_base_url_drops_fields_a_gateway_may_reject(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-gateway.example.internal")
    api, prov, s = _claude([fakes.claude_message(("text", "ok"))])
    assert not prov.fallbacks and not prov.eager_streaming
    s.send(user_text="goal")
    assert "fallbacks" not in api.requests[0]
    assert all("eager_input_streaming" not in t for t in api.requests[0]["tools"])


# ---------------------------------------------------------------------------
# Claude: the conversation
# ---------------------------------------------------------------------------

def test_claude_streams_and_returns_a_tool_turn_then_echoes_it_unchanged():
    api, _prov, s = _claude([
        fakes.claude_message(("thinking", "Read the project first.", "sig-AAA"), ("text", "Reading the project."),
                             ("tool", "toolu_1", "project_overview", "{}"),
                             ("tool", "toolu_2", "run_polar", '{"design_id": "d1"}'), stop="tool_use",
                             usage={"cache_read_input_tokens": 1000}),
        fakes.claude_message(("text", "Best L/D 108.7 at 5°."))])
    ev = Events()
    turn = s.send(user_text="goal", on_event=ev)
    assert turn.stop_reason == "tool_use"
    assert [(c.id, c.name, c.input) for c in turn.tool_calls] == [("toolu_1", "project_overview", {}),
                                                                 ("toolu_2", "run_polar", {"design_id": "d1"})]
    assert turn.text == "Reading the project." and turn.thinking == "Read the project first."
    assert ev.text("text") == "Reading the project." and ev.text("thinking") == "Read the project first."
    assert [d["name"] for k, d in ev.items if k == "tool_start"] == ["project_overview", "run_polar"]
    assert "tool_input" in ev.kinds()
    assert turn.usage["cache_read_input_tokens"] == 1000
    # all results of the turn go back together, in one user message
    s.send(tool_results=[ToolResult("toolu_1", "project_overview", '{"data": {}}'),
                         ToolResult("toolu_2", "run_polar", '{"error": "x"}', is_error=True)])
    msgs = api.requests[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    # preserved thinking: the assistant turn is passed back as received, signature included
    assert msgs[1]["content"][0] == {"type": "thinking", "thinking": "Read the project first.", "signature": "sig-AAA"}
    assert [b["type"] for b in msgs[1]["content"]] == ["thinking", "text", "tool_use", "tool_use"]
    assert msgs[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"data": {}}', "is_error": False},
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": '{"error": "x"}', "is_error": True}]
    # the request prefix is byte-identical between turns (cache hits, preserved-thinking binding)
    assert api.requests[1]["system"] == api.requests[0]["system"]
    assert api.requests[1]["tools"] == api.requests[0]["tools"]
    log = s.transcript()
    assert "signature" not in json.dumps(log) and log[1]["content"][1]["text"] == "Reading the project."


def test_claude_refusal_runs_nothing_and_is_not_appended():
    api, _prov, s = _claude([
        fakes.claude_message(("text", "Partial answer"), ("tool", "toolu_9", "run_polar", '{"design_id": "d'),
                             stop="refusal", stop_details={"type": "refusal", "category": "cyber",
                                                           "explanation": "Declined by a classifier."}),
        fakes.claude_message(("text", "ok"))])
    turn = s.send(user_text="goal")
    assert turn.stop_reason == "refusal" and turn.tool_calls == []
    assert "category: cyber" in turn.refusal and "Declined by a classifier." in turn.refusal
    s.send(user_text="rephrased")
    # the refused partial is discarded: the next request holds two user turns, no assistant turn
    assert [m["role"] for m in api.requests[1]["messages"]] == ["user", "user"]


def test_claude_truncated_tool_input_is_never_run_and_is_retried_with_more_room():
    api, _prov, s = _claude([
        fakes.claude_message(("tool", "toolu_1", "run_polar", '{"design_id": "d'), stop="max_tokens"),
        fakes.claude_message(("tool", "toolu_1", "run_polar", '{"design_id": "d1"}'), stop="tool_use")],
        max_tokens=8000)
    ev = Events()
    turn = s.send(user_text="goal", on_event=ev)
    assert [r["max_tokens"] for r in api.requests] == [8000, 16000]
    assert api.requests[1]["messages"] == api.requests[0]["messages"]     # nothing appended for the cut turn
    assert turn.tool_calls[0].input == {"design_id": "d1"}
    assert any("token limit" in d.get("text", "") for k, d in ev.items if k == "notice")


def test_claude_unparseable_tool_json_reissues_the_turn_then_gives_up():
    bad = fakes.claude_message(("tool", "toolu_1", "run_polar", "not json"), stop="tool_use")
    api, _prov, s = _claude([bad, fakes.claude_message(("text", "fine"))])
    turn = s.send(user_text="goal")
    assert turn.text == "fine" and len(api.requests) == 2
    _api, _prov, s2 = _claude([bad, bad, bad, bad])
    with pytest.raises(ProviderError) as e:
        s2.send(user_text="goal")
    assert e.value.kind == "protocol"


def test_claude_silently_truncated_json_reaches_the_caller_for_validation():
    # the SDK's tolerant parser turns '{"design_id": "d"x"}' into {"design_id": "d"}: the provider hands
    # it on as parsed, and the agent's schema validation is what must catch such input
    _api, _prov, s = _claude([fakes.claude_message(("tool", "t1", "run_polar", '{"design_id": "d"x", "n": 1}'),
                                                   stop="tool_use")])
    turn = s.send(user_text="goal")
    assert turn.tool_calls[0].input == {"design_id": "d"}


def test_claude_pause_turn_is_resumed():
    api, _prov, s = _claude([fakes.claude_message(("text", "Working"), stop="pause_turn"),
                             fakes.claude_message(("text", " done."))])
    turn = s.send(user_text="goal")
    assert turn.text == " done." and [m["role"] for m in api.requests[1]["messages"]] == ["user", "assistant"]


def test_claude_fallback_switch_point_is_echoed_by_the_documented_rule():
    api, _prov, s = _claude([
        fakes.claude_message(("thinking", "t", "sig-1"), ("tool", "toolu_0", "run_polar", '{"design_id": "x"}'),
                             ("text", "Partial "), ("fallback", "claude-opus-5-5", "claude-opus-5"),
                             ("text", "continued"), ("tool", "toolu_1", "project_overview", "{}"),
                             stop="tool_use", model="claude-opus-5",
                             iterations=[{"type": "message", "model": "claude-opus-5-5", "input_tokens": 10,
                                          "output_tokens": 5},
                                         {"type": "fallback_message", "model": "claude-opus-5", "input_tokens": 10,
                                          "output_tokens": 5}]),
        fakes.claude_message(("text", "ok"))])
    ev = Events()
    turn = s.send(user_text="goal", on_event=ev)
    # only calls after the last switch point run
    assert [c.id for c in turn.tool_calls] == ["toolu_1"]
    assert turn.served_by == "claude-opus-5"
    assert any("fallback" in n for n in turn.notices)
    assert any("declined" in d.get("text", "") for k, d in ev.items if k == "notice")
    s.send(tool_results=[ToolResult("toolu_1", "project_overview", "{}")])
    echoed = api.requests[1]["messages"][1]["content"]
    # thinking and tool_use before the switch point are omitted; text and everything after it stay
    assert [b["type"] for b in echoed] == ["text", "text", "tool_use"]
    assert echoed[2]["id"] == "toolu_1"


@pytest.mark.parametrize("status,etype,kind,retryable", [
    (401, "authentication_error", "auth", False),
    (403, "permission_error", "permission", False),
    (404, "not_found_error", "not_found", False),
    (400, "invalid_request_error", "bad_request", False),
    (413, "request_too_large", "too_large", False),
    (429, "rate_limit_error", "rate_limit", True),
    (529, "overloaded_error", "overloaded", True),
    (500, "api_error", "server", True),
])
def test_claude_typed_errors_become_plain_provider_errors(status, etype, kind, retryable):
    _api, _prov, s = _claude([(status, etype, "scripted failure")])
    with pytest.raises(ProviderError) as e:
        s.send(user_text="goal")
    assert e.value.kind == kind and e.value.retryable == retryable
    assert "sk-ant" not in str(e.value)
    if status == 429:
        assert "17 s" in str(e.value)


def test_claude_connection_error():
    import httpx2

    def fail(_request):
        raise httpx2.ConnectError("no route")
    client = anthropic.Anthropic(api_key="sk-ant-test-x", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(fail)))
    s = ClaudeProvider(client=client).start("S", TOOLS)
    with pytest.raises(ProviderError) as e:
        s.send(user_text="goal")
    assert e.value.kind == "connection" and e.value.retryable


def test_claude_stop_cancels_the_stream():
    api, _prov, s = _claude([fakes.claude_message(("text", "x" * 400))])
    cancel = threading.Event()
    api.on_event = lambda i: cancel.set() if i == 5 else None
    with pytest.raises(Cancelled):
        s.send(user_text="goal", cancel=cancel)


def test_claude_stop_closes_a_silent_stream():
    # the model is silent (reading a long prompt): Stop from another thread must not wait for the next event
    gate = threading.Event()
    api, _prov, s = _claude([fakes.claude_message(("text", "late answer"))])
    api.on_event = lambda i: gate.wait(10) if i == 1 else None
    cancel = threading.Event()
    box: dict = {}

    def work():
        try:
            s.send(user_text="goal", cancel=cancel)
            box["out"] = "returned"
        except Cancelled:
            box["out"] = "cancelled"
        except Exception as exc:  # noqa: BLE001 - reported by the assertion below
            box["out"] = repr(exc)
    t = threading.Thread(target=work)
    t.start()
    import time
    time.sleep(0.3)
    cancel.set()
    s.abort()
    gate.set()
    t.join(10)
    assert box.get("out") == "cancelled"
    assert s._messages[-1]["role"] == "user"                 # nothing of the stopped turn was appended


def test_claude_provider_check_and_credentials(monkeypatch, tmp_path):
    from aerolume.ai import config as C
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(C, "claude_credential_source", lambda: "")
    ok, msg = ClaudeProvider().check()
    assert not ok and "ANTHROPIC_API_KEY" in msg and "Preferences" in msg
    ok, msg = ClaudeProvider(api_key="sk-ant-api03-secret-value-1234").check()
    assert ok and "key from Preferences" in msg and "secret" not in msg
    assert ClaudeProvider().describe() == "Claude · claude-opus-5-5"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def test_ollama_discovery():
    with fakes.FakeOllama() as srv:
        assert server_version(srv.url) == "0.40.2"
        names = [m["name"] for m in list_models(srv.url)]
        assert names == ["qwen3:8b"]                     # the code model without "tools" is hidden
        assert {m["name"] for m in list_models(srv.url, tools_only=False)} == {"qwen3:8b", "deepseek-coder-v2:16b"}
        ok, msg = OllamaProvider("qwen3:8b", srv.url).check()
        assert ok and msg == "Local · qwen3:8b · Ollama 0.40.2"
        ok, msg = OllamaProvider("llama9:70b", srv.url).check()
        assert not ok and "ollama pull llama9:70b" in msg
    ok, msg = OllamaProvider("qwen3:8b", "http://127.0.0.1:9").check()
    assert not ok and "not running" in msg and server_version("http://127.0.0.1:9") is None


def test_ollama_conversation_on_the_wire():
    script = [fakes.ollama_tools(("project_overview", {}), ("run_polar", {"design_id": "d1", "speed_m_s": 20})),
              fakes.ollama_text("Best L/D 108.7 at 5°.")]
    with fakes.FakeOllama(script) as srv:
        s = OllamaProvider("qwen3:8b", srv.url, num_ctx=8192).start("SYSTEM", TOOLS)
        ev = Events()
        t1 = s.send(user_text="goal", on_event=ev)
        assert t1.stop_reason == "tool_use"
        assert [(c.id, c.name, c.input) for c in t1.tool_calls] == [
            ("call_0", "project_overview", {}), ("call_1", "run_polar", {"design_id": "d1", "speed_m_s": 20})]
        t2 = s.send(tool_results=[ToolResult("call_0", "project_overview", '{"data": {}}'),
                                  ToolResult("call_1", "run_polar", '{"data": {"ld": 108.7}}')], on_event=ev)
        assert t2.stop_reason == "end_turn" and t2.text == "Best L/D 108.7 at 5°." and ev.text("text") == t2.text
        b1, b2 = srv.chat_bodies()
    assert b1["model"] == "qwen3:8b" and b1["stream"] is True and b1["think"] is False
    assert b1["options"]["num_ctx"] == 8192
    assert b1["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert [t["function"]["name"] for t in b1["tools"]] == ["project_overview", "run_polar"]
    assert b1["tools"][1]["function"]["parameters"]["required"] == ["design_id"]
    # the assistant turn with its calls, then one tool message per call, in call order
    assert b2["messages"][2]["role"] == "assistant" and len(b2["messages"][2]["tool_calls"]) == 2
    assert [(m["role"], m["tool_name"]) for m in b2["messages"][3:]] == [("tool", "project_overview"),
                                                                       ("tool", "run_polar")]


def test_ollama_string_arguments_and_errors():
    raw = [{"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "run_polar", "arguments": '{"design_id": "d1"}'}},
        {"function": {"name": "run_polar", "arguments": "{broken"}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}]
    with fakes.FakeOllama([raw, 404]) as srv:
        s = OllamaProvider("qwen3:8b", srv.url).start("S", TOOLS)
        t = s.send(user_text="goal")
        assert t.tool_calls[0].input == {"design_id": "d1"} and t.tool_calls[0].raw == '{"design_id": "d1"}'
        assert t.tool_calls[1].input == "{broken"                  # left for the agent to reject as INVALID_JSON
        assert t.tool_calls[0].id.startswith("call_")              # ids are made up when the server has none
        with pytest.raises(ProviderError) as e:
            s.send(user_text="again")
        assert e.value.kind == "not_found" and "ollama pull" in str(e.value)


def test_ollama_server_crash_is_retried_once(monkeypatch):
    import aerolume.ai.providers.ollama as O
    monkeypatch.setattr(O, "RETRY_DELAY_S", 0.0)
    with fakes.FakeOllama([500, fakes.ollama_text("recovered")]) as srv:
        ev = Events()
        t = OllamaProvider("qwen3:8b", srv.url).start("S", TOOLS).send(user_text="goal", on_event=ev)
        assert t.text == "recovered" and any("trying again" in d.get("text", "") for k, d in ev.items)
    with fakes.FakeOllama([500, 500]) as srv:
        with pytest.raises(ProviderError) as e:
            OllamaProvider("qwen3:8b", srv.url).start("S", TOOLS).send(user_text="goal")
        assert e.value.kind == "server"


def test_ollama_unreachable_and_context_warning():
    with pytest.raises(ProviderError) as e:
        OllamaProvider("qwen3:8b", "http://127.0.0.1:9").start("S", TOOLS).send(user_text="goal")
    assert e.value.kind == "connection" and "Start Ollama" in str(e.value)
    with fakes.FakeOllama([fakes.ollama_text("ok", prompt_tokens=7800)]) as srv:
        t = OllamaProvider("qwen3:8b", srv.url, num_ctx=8192).start("S", TOOLS).send(user_text="goal")
        assert any("nearly full" in n for n in t.notices)


def test_ollama_stop_mid_stream_and_gpu_release():
    with fakes.FakeOllama([fakes.ollama_text("x" * 600, chunks=60)], delay=0.05) as srv:
        prov = OllamaProvider("qwen3:8b", srv.url)
        s = prov.start("S", TOOLS)
        cancel = threading.Event()
        ev = Events()

        def on_event(kind, data):
            ev(kind, data)
            if len(ev.items) == 3:
                cancel.set()
        with pytest.raises(Cancelled):
            s.send(user_text="goal", on_event=on_event, cancel=cancel)
        assert len(ev.items) < 10
        prov.release_gpu()
        gen = [b for p, b in srv.requests if p == "/api/generate"]
        assert gen == [{"model": "qwen3:8b", "keep_alive": 0}]


def test_ollama_timeout_is_not_reported_as_unreachable(monkeypatch):
    """A server that accepts the request but sends nothing (a model loading, or spilling onto the CPU)
    times out: the message says it did not answer in time and why that happens, not "not reachable"."""
    import socket
    import threading

    import aerolume.ai.providers.ollama as O
    monkeypatch.setattr(O, "READ_TIMEOUT_S", 0.5)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    held = []
    threading.Thread(target=lambda: held.append(srv.accept()), daemon=True).start()   # accept, never answer
    try:
        with pytest.raises(ProviderError) as e:
            OllamaProvider("qwen3:8b", f"http://127.0.0.1:{srv.getsockname()[1]}").start("S", TOOLS).send(
                user_text="goal")
        text = str(e.value)
        assert e.value.kind == "connection" and e.value.retryable
        assert "did not answer within" in text and "not reachable" not in text and "smaller model" in text
    finally:
        for conn, _addr in held:
            conn.close()
        srv.close()
