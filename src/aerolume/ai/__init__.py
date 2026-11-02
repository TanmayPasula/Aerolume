"""The AI experiment assistant (DESIGN.md section 9).

State a goal in plain language ("find the angle of attack with the best L/D of
NACA 2412 at 20 m/s") and the assistant plans an experiment, runs it with the
same solvers as the GUI and the scripting API, analyses the results with their
uncertainty and validity limits, and logs everything in the project
(``experiments/<id>/``).

Two model providers: **Claude** through the official ``anthropic`` SDK (optional
dependency ``aerolume[ai]``) and **local models through Ollama** (offline, open
source, standard library only). No Qt in this package: the GUI's Assistant dock
(``app/widgets/assistant_panel.py``), the command line (``python -m aerolume.ai``)
and scripts drive the same :class:`Agent`.

Modules: :mod:`.config` (preferences, API key), :mod:`.providers` (Claude,
Ollama), :mod:`.tools` (the JSON-schema tools), :mod:`.agent` (the loop),
:mod:`.experiment` (logs), :mod:`.claims` (the check of every final answer),
:mod:`.workspace` (where tools act), :mod:`.schema`
(input validation), :mod:`.prompts` (system prompts), :mod:`.plaintext` (model
answers as plain Unicode: LaTeX math converted as it streams). Imports are lazy, so
``import aerolume.ai`` costs nothing until something is used.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "Agent": ("agent", "Agent"),
    "Outcome": ("agent", "Outcome"),
    "AssistantConfig": ("config", "AssistantConfig"),
    "load_config": ("config", "load_config"),
    "save_config": ("config", "save_config"),
    "make_provider": ("providers", "make_provider"),
    "ProviderError": ("providers.base", "ProviderError"),
    "Toolbox": ("tools", "Toolbox"),
    "TOOLS": ("tools", "TOOLS"),
    "Approval": ("tools", "Approval"),
    "ExperimentLog": ("experiment", "ExperimentLog"),
    "list_experiments": ("experiment", "list_experiments"),
    "load_record": ("experiment", "load_record"),
    "Workspace": ("workspace", "Workspace"),
    "run_experiment": ("cli", "run_experiment"),
    "ClaimReport": ("claims", "ClaimReport"),
    "check_answer": ("claims", "check_answer"),
    "plain_math": ("plaintext", "plain_math"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(f"{__name__}.{mod}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TOOLS", "Agent", "Approval", "AssistantConfig", "ClaimReport", "ExperimentLog", "Outcome",
           "ProviderError", "Toolbox", "Workspace", "check_answer", "list_experiments", "load_config", "load_record",
           "make_provider", "plain_math", "run_experiment", "save_config"]
