"""The recent-projects list shown on the Start page, kept in the user config folder.

Location: ``%APPDATA%\\Aerolume`` on Windows, ``$XDG_CONFIG_HOME/aerolume`` (default
``~/.config/aerolume``) elsewhere. ``AEROLUME_CONFIG_DIR`` overrides both, which is
how tests and portable installs keep their own list.

The list is advisory: a corrupt or unwritable file must never stop the app from
starting, so reading falls back to an empty list and writing failures are
swallowed (the worst outcome is a shorter list on the Start page).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from pathlib import Path

from .fileio import write_json

MAX_RECENT = 12
_FILE = "recent_projects.json"
_lock = threading.Lock()


def config_dir() -> Path:
    """The per-user Aerolume configuration folder (not created here)."""
    env = os.environ.get("AEROLUME_CONFIG_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Aerolume"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "aerolume"


def _norm(path: str | os.PathLike) -> str:
    return str(Path(path).expanduser().resolve())


def _read() -> list[str]:
    try:
        data = json.loads((config_dir() / _FILE).read_text(encoding="utf-8"))
        items = data.get("recent", []) if isinstance(data, dict) else []
        return [str(p) for p in items if isinstance(p, str)]
    except (OSError, ValueError):
        return []


def _write(items: list[str]) -> None:
    with contextlib.suppress(OSError):
        write_json(config_dir() / _FILE, {"recent": items[:MAX_RECENT]})


def recent_projects(existing_only: bool = True) -> list[Path]:
    """Recently used project folders, most recent first.

    ``existing_only`` hides projects that were moved or deleted (they stay in the
    file, so a re-attached USB drive brings them back).
    """
    with _lock:
        items = _read()
    paths = [Path(p) for p in items]
    if existing_only:
        paths = [p for p in paths if (p / "project.json").is_file()]
    return paths


def add_recent(path: str | os.PathLike) -> None:
    """Move ``path`` to the top of the list (call after opening or creating a project)."""
    p = _norm(path)
    with _lock:
        items = [q for q in _read() if os.path.normcase(q) != os.path.normcase(p)]
        _write([p, *items])


def remove_recent(path: str | os.PathLike) -> None:
    """Drop ``path`` from the list (the Start page's "remove from list")."""
    p = _norm(path)
    with _lock:
        _write([q for q in _read() if os.path.normcase(q) != os.path.normcase(p)])


def clear_recent() -> None:
    """Empty the list."""
    with _lock:
        _write([])
