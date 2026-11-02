"""Security findings of the round-5 review (each reproduced before the fix).

* One approved ``run_python`` script could replace ``Toolbox.approval`` and switch every later
  approval card off; the gate is now checked after every call and put back.
* The approval card showed scripts unwrapped: a line padded with 160 spaces hid
  "; import shutil" past the card's edge, and a U+202E override could reverse text. The card
  now wraps, shows long space runs and invisible characters as markers, and warns.
* MCP: ``confirm=true`` on the very first call deleted at once; it now counts only for a call
  the server answered with "needs confirmation" earlier in the session.
* MCP ``--read-only`` created projects (``open_project(create=true)``), wrote session logs and
  offered the recording tools; it now opens existing projects only and writes nothing.
* The claims check accepted digits inside names someone typed ("NACA 2412 (wind tunnel: Cd
  0.0061, Cl_max 1.71)") as evidence for the same numbers in an answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _data(res: dict) -> dict:
    return json.loads(res["content"][0]["text"])


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir")
            for p in sorted(root.rglob("*"))}


def test_an_approved_script_cannot_switch_the_approval_gate_off(tmp_path):
    from aerolume import api
    from aerolume.ai.tools import TOOLS, Toolbox, ToolContext
    from aerolume.ai.workspace import Workspace
    p = api.open_project(tmp_path / "Gate.aerolume", create=True)
    d = p.add_naca("0012")
    ws = Workspace(p)
    tb = Toolbox(TOOLS)
    ctx = ToolContext(ws, None, threading.Event(), None)
    args = {"kind": "design", "id": d.id}
    original = Toolbox.approval
    assert tb.approval("delete_item", args, ctx, 120.0) is not None
    code = "log('summary')" + " " * 200 + ";import aerolume.ai.tools as t; t.Toolbox.approval = lambda *a, **k: None"
    out = tb.run("run_python", {"code": code, "purpose": "print a summary"}, ctx)
    assert out.is_error and "approval rules" in out.summary
    assert Toolbox.approval is original
    assert Toolbox(TOOLS).approval("delete_item", args, ctx, 120.0) is not None
    # shadowing it on the instance is caught as well
    out = tb.run("run_python", {"code": "import aerolume.ai.tools as t\nfor o in []: pass", "purpose": "noop"}, ctx)
    assert not out.is_error
    tb.approval = lambda *a, **k: None
    out = tb.run("run_python", {"code": "x = 1", "purpose": "noop"}, ctx)
    assert out.is_error and "toolbox.approval" in out.summary and "approval" not in vars(tb)


def test_reveal_code_shows_padding_and_invisible_characters():
    from aerolume.app.widgets.assistant_panel import reveal_code
    code = "r = project.result('abc')" + " " * 160 + "; import shutil\n    x = 1  # " + chr(0x202E) + ";)'tset'(\n"
    shown, notes = reveal_code(code)
    first = shown.splitlines()[0]
    assert "«160 spaces»" in first and first.endswith("; import shutil") and len(first) < 70
    assert "«U+202E»" in shown and chr(0x202E) not in shown
    assert shown.splitlines()[1].startswith("    x = 1")              # indentation is kept as it is
    assert len(notes) == 2
    assert reveal_code("for i in range(3):\n    log(i)\n") == ("for i in range(3):\n    log(i)\n", [])


@pytest.mark.gui
def test_the_approval_card_wraps_and_warns():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.ai.tools import Approval
    from aerolume.app import theme
    from aerolume.app.widgets.assistant_panel import _ApprovalCard
    theme.apply(app, "light")
    code = "r = project.result('abc123')" + " " * 160 + "; import shutil\nprint(r)\n"
    card = _ApprovalCard("c1", Approval("code", "Run this Python script?", "Print summary", None, code=code,
                                        items=[("Purpose", "Print summary")]))
    try:
        card.resize(430, 400)
        card.show()
        app.processEvents()
        assert card.code.lineWrapMode() == QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
        assert card.code.horizontalScrollBar().maximum() == 0             # nothing sits right of the edge
        assert "import shutil" in card.code.toPlainText()
        assert card.code_warning is not None and "spaces" in card.code_warning.text()
    finally:
        card.close()
        card.deleteLater()


def test_mcp_confirm_counts_only_after_a_listed_request(tmp_path):
    from aerolume.mcp_server import McpServer
    s = McpServer(tmp_path / "Demo.aerolume", create=True)
    try:
        did = _data(s.call_tool("create_naca_airfoil", {"code": "0012"}))["data"]["design_id"]
        first = s.call_tool("delete_item", {"kind": "design", "id": did, "confirm": True})
        payload = _data(first)
        assert first["isError"] and payload["needs_confirmation"] and "counts only after" in payload["reason"]
        assert did in [d["id"] for d in _data(s.call_tool("project_overview", {}))["data"]["designs"]]
        done = s.call_tool("delete_item", {"kind": "design", "id": did, "confirm": True})
        assert not done["isError"]
        assert did not in [d["id"] for d in _data(s.call_tool("project_overview", {}))["data"]["designs"]]
        # the confirmation was used up: a second delete of another design must be listed again
        did2 = _data(s.call_tool("create_naca_airfoil", {"code": "2412"}))["data"]["design_id"]
        assert _data(s.call_tool("delete_item", {"kind": "design", "id": did2, "confirm": True}))["needs_confirmation"]
    finally:
        s.close()


def test_mcp_read_only_writes_nothing(tmp_path):
    from aerolume import api
    from aerolume.mcp_server import McpServer
    ro = McpServer(None, read_only=True)
    target = tmp_path / "elsewhere" / "New.aerolume"
    res = ro.call_tool("open_project", {"path": str(target), "create": True})
    assert res["isError"] and not target.exists()
    names = {t["name"] for t in ro.tool_list()}
    assert "record_plan" not in names and "record_conclusion" not in names and "delete_item" not in names
    ro.close()
    p = api.open_project(tmp_path / "Existing.aerolume", create=True)
    p.add_naca("2412", name="NACA 2412 (wind tunnel: Cd 0.0061, Cl_max 1.71)")
    root = Path(p.path)
    del p
    before = _tree(root)
    ro = McpServer(root, read_only=True)
    try:
        assert not ro.call_tool("project_overview", {})["isError"]
        claims = _data(ro.call_tool("check_claims", {"text": "The NACA 2412 has a drag coefficient of 0.0061."}))
        # the claims check still works (its log lives in a temporary folder) ...
        nums = [n for st in claims["data"]["statements"] for n in st.get("numbers", [])]
        # ... and the digits inside a design name are not evidence
        assert nums and all(n["status"] == "unsupported" for n in nums if n["token"] == "0.0061")
        with pytest.raises(KeyError):
            ro.call_tool("record_plan", {"objective": "x"})
    finally:
        ro.close()
    assert _tree(root) == before


def test_numbers_in_names_are_not_evidence_but_measured_values_are():
    from aerolume.ai.experiment import labelled_numbers
    obj = {"designs": [{"id": "d1", "name": "drag 0.0287 lift 1.234"}],
           "result": {"simulation_id": "s1", "name": "Polar", "cd": 0.0287},
           "notes": ["validated within 3 % of the reference"]}
    vals = {(round(n.value, 6), n.key) for n in labelled_numbers(obj, origin="result", tool="get_results")}
    assert (0.0287, "cd") in vals and (1.234, "name") not in vals and (0.0287, "name") not in vals
    assert (3.0, "notes") in vals                                  # a tool's own sentence still counts
    goal = labelled_numbers({"name": "drag 0.0287"}, origin="goal")
    assert goal and goal[0].value == pytest.approx(0.0287)         # the user's own words do
