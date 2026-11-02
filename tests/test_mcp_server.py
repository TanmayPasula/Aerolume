"""``aerolume-mcp``: the MCP stdio server, driven by a real subprocess client.

The client speaks the published protocol (JSON-RPC 2.0, one message per line):
``initialize`` / ``notifications/initialized``, ``ping``, ``tools/list``,
``tools/call`` with the confirm rules for long runs and deletions, and the error
codes for bad JSON, unknown methods and unknown tools. A second test runs the
server in-process to check progress notifications and cancellation of a running
call. Runs are inviscid wing sweeps (well under a second) on a temporary project.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

from aerolume.ai.experiment import list_experiments


@pytest.fixture(autouse=True)
def _private_config(tmp_path, monkeypatch):
    """Every test here gets its own config folder and preferences file.

    The server adds each project it opens to the recent-projects list. The
    in-process test used to write its "Slow" project into the shared
    AEROLUME_CONFIG_DIR, and the app's Start page then listed a pytest temp
    folder under Recent projects (critic round 3). The subprocess tests pass
    their own folder in the environment as well.
    """
    from aerolume.app import settings
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    yield
    settings.use_file(tmp_path / "settings-after.ini")


class Client:
    """A minimal MCP client over a subprocess's stdin / stdout."""

    def __init__(self, args: list[str], env: dict) -> None:
        self.proc = subprocess.Popen([sys.executable, "-m", "aerolume.mcp_server", *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.messages: queue.Queue = queue.Queue()
        self.stderr: list[str] = []
        self.next_id = 1
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()

    def _read(self) -> None:
        for line in iter(self.proc.stdout.readline, b""):
            self.messages.put(json.loads(line.decode("utf-8")))       # every stdout line must be JSON

    def _read_err(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            self.stderr.append(line.decode("utf-8", "replace"))

    def send(self, msg: dict | str) -> None:
        data = msg if isinstance(msg, str) else json.dumps(msg)
        self.proc.stdin.write(data.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 120.0) -> dict:
        mid = self.next_id
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        return self.wait_for(mid, timeout)

    def wait_for(self, mid, timeout: float = 120.0) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                m = self.messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if m.get("id") == mid and ("result" in m or "error" in m):
                return m
        raise AssertionError(f"no response to {mid}; stderr: {''.join(self.stderr)[-2000:]}")

    def call(self, name: str, arguments: dict | None = None, timeout: float = 180.0) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout)["result"]

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait(timeout=60)


def _data(result: dict) -> dict:
    payload = json.loads(result["content"][0]["text"])
    assert result.get("structuredContent") == payload               # 2025-06-18: structured and text agree
    return payload


@pytest.fixture()
def client(tmp_path):
    env = dict(os.environ, AEROLUME_CONFIG_DIR=str(tmp_path / "config"), PYTHONIOENCODING="utf-8")
    c = Client(["--project", str(tmp_path / "MCP test.aerolume"), "--create", "--confirm-above", "0"], env)
    yield c, tmp_path
    if c.proc.poll() is None:
        c.proc.kill()


def test_handshake_tools_and_confirmed_calls(client):
    c, tmp = client
    init = c.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "pytest-client", "version": "1.0"}})["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "aerolume"
    assert init["capabilities"]["tools"] == {"listChanged": False} and "confirm=true" in init["instructions"]
    c.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert c.request("ping")["result"] == {}

    tools = {t["name"]: t for t in c.request("tools/list")["result"]["tools"]}
    for name in ("project_overview", "describe_design", "estimate_cost", "run_polar", "run_wing", "run_cfd",
                 "get_results", "compare_results", "write_report", "open_project", "list_projects", "check_claims",
                 "create_case", "compare_reference"):
        assert name in tools, name
    assert tools["compare_reference"]["title"] == "Compare with the published reference"
    assert tools["compare_reference"]["annotations"]["readOnlyHint"] is True
    assert "run_python" not in tools and "suggest_settings" not in tools
    assert "confirm" in tools["run_wing"]["inputSchema"]["properties"]
    assert "confirm" in tools["delete_item"]["inputSchema"]["properties"]
    assert "confirm" not in tools["get_results"]["inputSchema"]["properties"]
    assert tools["delete_item"]["annotations"]["destructiveHint"] is True
    assert tools["project_overview"]["annotations"]["readOnlyHint"] is True

    wing = _data(c.call("create_wing", {"span_m": 2.0, "root_chord_m": 0.3, "tip_chord_m": 0.18}))["data"]
    args = {"design_id": wing["design_id"], "speed_m_s": 20, "alpha_start": 0, "alpha_end": 4, "alpha_step": 2,
            "viscous": False}
    held = c.call("run_wing", args)
    payload = _data(held)
    assert held["isError"] is True and payload["needs_confirmation"] is True
    would = {r["item"]: r["value"] for r in payload["would_do"]}
    assert would["Speed"] == "20 m/s" and would["Angles"] == "α 0° to 4°, step 2° (3 angles)"
    assert would["Method"] == "vortex lattice, inviscid"
    assert "aerolume-mcp --confirm-above" in payload["reason"] and "Preferences" not in payload["reason"]

    ran = c.call("run_wing", {**args, "confirm": True})
    out = _data(ran)["data"]
    assert ran["isError"] is False and out["analysis"] == "wing" and out["converged_points"] == 3
    slope = out["metrics"]["cl_alpha_per_deg"]
    sid = out["simulation_id"]
    read = _data(c.call("get_results", {"simulation_ids": [sid]}))["data"]["results"][0]
    assert read["metrics"]["cl_alpha_per_deg"] == slope and "not_validated" in read

    checked = _data(c.call("check_claims", {"text": f"The lift slope is {slope} per degree. Cd is 0.031."}))["data"]
    assert [s["status"] for s in checked["statements"]] == ["verified", "unsupported"]
    assert checked["ok"] is False and "0.031" in checked["correct_before_answering"]

    gone = c.call("delete_item", {"kind": "simulation", "id": sid})
    assert gone["isError"] is True and _data(gone)["needs_confirmation"] is True
    deleted = _data(c.call("delete_item", {"kind": "simulation", "id": sid, "confirm": True}))["data"]
    assert deleted["deleted"] == sid

    # protocol errors
    assert c.request("tools/call", {"name": "no_such_tool", "arguments": {}})["error"]["code"] == -32602
    assert c.request("resources/list")["error"]["code"] == -32601
    c.send("{this is not json")
    err = c.wait_for(None, 30)
    assert err["error"]["code"] == -32700
    bad = c.call("run_wing", {"design_id": wing["design_id"], "speed_m_s": "fast"})
    assert bad["isError"] is True and "Invalid input" in _data(bad)["error"]

    assert c.close() == 0
    logs = list_experiments(tmp / "MCP test.aerolume")
    assert logs and logs[0].goal == "MCP session from pytest-client" and logs[0].actions >= 6


def test_a_new_client_version_falls_back_and_no_project_is_explained(tmp_path):
    env = dict(os.environ, AEROLUME_CONFIG_DIR=str(tmp_path / "config"), PYTHONIOENCODING="utf-8")
    c = Client([], env)
    try:
        init = c.request("initialize", {"protocolVersion": "2099-01-01", "capabilities": {},
                                        "clientInfo": {"name": "future"}})["result"]
        assert init["protocolVersion"] == "2025-11-25"
        res = c.call("project_overview")
        assert res["isError"] is True and "open_project" in res["content"][0]["text"]
        opened = c.call("open_project", {"path": str(tmp_path / "Later.aerolume"), "create": True})
        assert opened["isError"] is False
        overview = json.loads(c.call("project_overview")["content"][0]["text"])["data"]
        assert overview["project"] == "Later" and overview["designs"] == []
    finally:
        assert c.close() == 0


def test_stdin_reader_splits_lines_and_never_leaves_a_read_pending():
    """The server's stdin reader: whole lines across partial writes, EOF, and no pending read on Windows.

    A pending synchronous read on a pipe makes ``GetFileInformationByHandle`` (and the
    old C runtime's ``fstat``) on that pipe wait; scipy's mingw-built BLAS calls it while
    it loads, which hung the first wing run of every MCP session. While the reader
    waits for data, that query must return at once.
    """
    from aerolume.mcp_server import _StdinReader

    r, w = os.pipe()
    reader = _StdinReader(r)
    got: list[bytes] = []
    t = threading.Thread(target=lambda: got.extend([reader.readline(), reader.readline(), reader.readline()]),
                         daemon=True)
    t.start()
    time.sleep(0.2)
    if os.name == "nt":
        import ctypes
        import msvcrt

        class Info(ctypes.Structure):
            _fields_ = [("raw", ctypes.c_ulong * 13)]

        done = threading.Event()
        info = Info()

        def query() -> None:
            ctypes.windll.kernel32.GetFileInformationByHandle(msvcrt.get_osfhandle(r), ctypes.byref(info))
            done.set()

        threading.Thread(target=query, daemon=True).start()
        assert done.wait(2.0), "a read was left pending on the pipe: handle queries of other threads wait"
    os.write(w, b'{"a": 1}\n{"b"')
    time.sleep(0.05)
    os.write(w, b': 2}\n{"c": 3}')
    os.close(w)
    t.join(5)
    assert got == [b'{"a": 1}\n', b'{"b": 2}\n', b'{"c": 3}']
    assert reader.readline() == b""
    os.close(r)


def test_progress_and_cancellation_in_process(tmp_path):
    from aerolume import api
    from aerolume.ai.tools import Tool, ToolOutcome
    from aerolume.mcp_server import McpServer
    from aerolume.project.recent import config_dir, recent_projects
    api.new_project(tmp_path / "Slow.aerolume")
    server = McpServer(tmp_path / "Slow.aerolume", confirm_above_s=1e9)
    # the project went on this test's own recent list, never the shared one
    assert config_dir() == tmp_path / "config"
    assert [p.name for p in recent_projects()] == ["Slow.aerolume"]
    seen = {"stopped": None}

    def slow(ctx, _args):
        ctx.report(0.5, "half way")
        time.sleep(0.6)                 # past the 0.5 s rate limit, so only the falling value is what drops it
        ctx.report(0.2, "estimate grew")     # a falling fraction: MCP progress must still increase
        end = time.monotonic() + 20
        while not ctx.cancel.is_set() and time.monotonic() < end:
            time.sleep(0.02)
        seen["stopped"] = ctx.cancel.is_set()
        return ToolOutcome({"stopped": seen["stopped"]}, "slow done")
    server.toolbox.tools["slow"] = Tool("slow", "A slow test tool that reports progress and honours cancel.",
                                        {"type": "object", "properties": {}, "required": [],
                                         "additionalProperties": False}, slow, "read")

    r_in, w_in = os.pipe()
    sent: list[dict] = []
    got = threading.Event()

    class Sink:
        def write(self, data: bytes) -> None:
            for line in data.decode("utf-8").splitlines():
                sent.append(json.loads(line))
            got.set()

        def flush(self) -> None:
            pass

    stdin = os.fdopen(r_in, "rb")
    t = threading.Thread(target=server.serve, args=(stdin, Sink()), daemon=True)
    t.start()
    out = os.fdopen(w_in, "wb", buffering=0)

    def send(msg: dict) -> None:
        out.write(json.dumps(msg).encode() + b"\n")

    def wait(cond, timeout=10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.02)
        return cond()

    send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
    assert wait(lambda: any(m.get("id") == 1 for m in sent))
    send({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
          "params": {"name": "slow", "arguments": {}, "_meta": {"progressToken": "tok"}}})
    assert wait(lambda: any(m.get("method") == "notifications/progress" for m in sent))
    prog = next(m for m in sent if m.get("method") == "notifications/progress")["params"]
    assert prog["progressToken"] == "tok" and prog["progress"] == 0.5 and prog["message"] == "half way"
    send({"jsonrpc": "2.0", "id": 8, "method": "ping"})               # answered while the call runs
    assert wait(lambda: any(m.get("id") == 8 for m in sent))
    send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 7, "reason": "test"}})
    assert wait(lambda: seen["stopped"] is not None)
    assert seen["stopped"] is True
    time.sleep(0.3)
    assert not any(m.get("id") == 7 for m in sent)                     # no response to a cancelled request
    values = [m["params"]["progress"] for m in sent if m.get("method") == "notifications/progress"]
    assert values == [0.5]                                             # the 0.2 was not sent (MCP: MUST increase)
    out.close()
    t.join(10)
    assert not t.is_alive()
    first = next(m for m in sent if m.get("id") == 1)["result"]
    assert first["protocolVersion"] == "2025-03-26"
