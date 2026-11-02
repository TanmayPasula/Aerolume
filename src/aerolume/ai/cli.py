"""Run an assistant experiment from a terminal or a script: ``python -m aerolume.ai``.

Examples::

    python -m aerolume.ai --project "My study.aerolume" "find the angle of attack with the best L/D \
of NACA 2412 at 20 m/s using the polar analysis"
    python -m aerolume.ai --project P --provider claude --approve runs "compare the drag of ..."

Approvals in a terminal: runs above the threshold, code and deletions are asked
interactively (y/N). ``--approve runs`` approves long runs without asking (code
and deletions still ask); ``--approve none`` declines everything that would ask.
When standard input is not a terminal, anything that needs approval is declined
unless ``--approve`` allows it. The experiment is logged in the project exactly
as from the GUI.

Data consent: with Claude (or an Ollama server on another computer) the goal,
the project's names and the tools' results leave this computer. The terminal
asks once (y/N); ``--allow-data`` answers yes without asking, and a destination
allowed in Preferences > Assistant is not asked about. The final answer is
followed by the claims check: every statement with numbers or actions, verified,
derived, not supported or contradicted.
"""
from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .agent import Agent, Outcome
from .config import AssistantConfig, load_config
from .tools import Approval


def _printer(out=sys.stdout) -> Callable[[str, dict], None]:
    state = {"in_text": False}
    lock = threading.Lock()

    def on_event(kind: str, data: dict) -> None:
        with lock:
            if kind == "text":
                out.write(data.get("delta", ""))
                state["in_text"] = True
                out.flush()
                return
            if state["in_text"]:
                out.write("\n")
                state["in_text"] = False
            if kind == "tool_start":
                out.write(f"  > {data.get('name')}\n")
            elif kind == "tool_end":
                secs = data.get("seconds") or 0.0
                out.write(f"    {data.get('status')}: {data.get('summary', '')}"
                          + (f" ({secs:.1f} s)" if secs >= 0.1 else "") + "\n")
                for f in data.get("files") or []:
                    out.write(f"    figure: {f}\n")
            elif kind == "notice":
                out.write(f"  note: {data.get('text', '')}\n")
            elif kind == "error":
                out.write(f"  error: {data.get('text', '')}\n")
            elif kind == "done":
                if data.get("status") == "unverified" and data.get("withheld"):
                    # the streamed answer above claimed an action no tool call performed
                    out.write(f"  not verified: {data.get('text', '')}\n")
                _print_claims(out, data.get("claims") or {})
            out.flush()

    return on_event


def make_approver(policy: str, *, interactive: bool | None = None,
                  ask: Callable[[str], str] = input) -> Callable[[Approval, str], bool]:
    """The approval callback of the terminal: "ask" (default), "runs" or "none"."""
    tty = sys.stdin.isatty() if interactive is None else interactive

    def approve(a: Approval, _call_id: str) -> bool:
        if policy == "runs" and a.kind == "run":
            return True
        if policy == "none" or not tty:
            return False
        print(f"\n  Approval needed: {a.title}")
        items = list(getattr(a, "items", []) or [])
        if items:
            w = max(len(k) for k, _v in items)
            for k, v in items:
                print(f"    {k.ljust(w)}  {v}")
        elif getattr(a, "what", ""):
            print(f"  {a.what}")
        print(f"  {a.detail}")
        if a.code:
            print("  --- code ---\n" + "\n".join("  " + line for line in a.code.splitlines()) + "\n  ------------")
        try:
            return ask("  Approve? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    return approve


def run_experiment(project: str | Path | Any, goal: str, *, config: AssistantConfig | None = None,
                   provider: Any = None, approve: Callable[[Approval, str], bool] | None = None,
                   on_event: Callable[[str, dict], None] | None = None,
                   consent: bool | Callable[[str], bool] | None = None) -> Outcome:
    """Run one experiment headless and return how it ended (the log is in the project).

    ``project`` is a project folder, an open :class:`aerolume.project.Project` store
    or an :class:`aerolume.api.Project`; ``provider`` defaults to the one the
    preferences select. ``consent``: True allows sending the experiment's data to a
    model off this computer, a function is asked (with the destination), None
    allows only destinations the preferences already allow.
    """
    from .providers import make_provider
    from .workspace import Workspace
    cfg = (config or load_config()).validated()
    prov = provider or make_provider(cfg)
    ask = (lambda _dest: True) if consent is True else (consent if callable(consent) else None)
    agent = Agent(prov, Workspace.open(project), config=cfg, approve=approve, on_event=on_event, consent=ask)
    return agent.run(goal)


def _print_claims(out, claims: dict) -> None:
    """The claims check after the answer: one line per checked statement."""
    rows = [s for s in claims.get("statements", []) if isinstance(s, dict) and s.get("status") != "plain"]
    if not rows:
        return
    out.write(f"  claims check: {claims.get('summary', '')}\n")
    mark = {"verified": "ok ", "derived": "der", "unsupported": "?? ", "contradicted": "!! "}
    for s in rows:
        text = " ".join(str(s.get("text", "")).split())
        text = text if len(text) <= 100 else text[:99] + "…"
        detail = "; ".join(s.get("problems") or []) or "; ".join(s.get("evidence") or [])
        out.write(f"    [{mark.get(s.get('status'), '   ')}] {text}\n")
        if detail and s.get("status") != "verified":
            out.write(f"          {detail[:200]}\n")


def make_consent(allow: bool, *, interactive: bool | None = None,
                 ask: Callable[[str], str] = input) -> Callable[[str], bool]:
    """The terminal's consent prompt: ``allow`` answers yes; otherwise ask once (no terminal: no)."""
    tty = sys.stdin.isatty() if interactive is None else interactive

    def consent(dest: str) -> bool:
        if allow:
            return True
        if not tty:
            print(f"  Not sent: sending the experiment to {dest} needs consent (--allow-data).")
            return False
        print(f"\n  The goal, the project's design and simulation names and the tools' results will be sent to "
              f"{dest}.")
        try:
            return ask("  Send them? [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False

    return consent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aerolume.ai", description="Run an Aerolume assistant experiment.")
    ap.add_argument("goal", help="what to find out, in plain language")
    ap.add_argument("--project", required=True, help="project folder (*.aerolume)")
    ap.add_argument("--provider", choices=("auto", "claude", "ollama"), help="model provider (default: preferences)")
    ap.add_argument("--model", help="model name (default: preferences)")
    ap.add_argument("--approve", choices=("ask", "runs", "none"), default="ask",
                    help="what to do when a run, code or a deletion needs approval")
    ap.add_argument("--threshold", type=float, help="ask before runs estimated longer than this many seconds")
    ap.add_argument("--allow-data", action="store_true",
                    help="send the experiment's data to a model off this computer (Claude) without asking")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.provider:
        cfg.provider = args.provider
    if args.model:
        if cfg.validated().effective_provider() == "claude":
            cfg.claude_model = args.model
        else:
            cfg.ollama_model = args.model
    if args.threshold is not None:
        cfg.approval_threshold_s = args.threshold
    from .providers import make_provider
    prov = make_provider(cfg)
    ok, status = prov.check()
    print(f"Assistant: {status}")
    if not ok:
        return 2
    try:
        out = run_experiment(args.project, args.goal, config=cfg, provider=prov,
                             approve=make_approver(args.approve), on_event=_printer(),
                             consent=make_consent(args.allow_data))
    except KeyboardInterrupt:
        return 1
    print(f"\n[{out.status}] log: {out.path}")
    return 0 if out.status == "done" else 1


__all__ = ["main", "make_approver", "make_consent", "run_experiment"]
