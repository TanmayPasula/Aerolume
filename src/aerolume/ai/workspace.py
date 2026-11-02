"""Where the assistant's tools act: an :class:`aerolume.api.Project` and a few window hooks.

The tools drive the scripting API (DESIGN.md section 8), the same facade people
use in scripts and the Script console, so everything the assistant does is
recorded in the project exactly as a run started by hand: designs and
simulations appear in the Project tree, results open on the Results page, and
identical runs are reused instead of recomputed.

Who edits the project is the API's :class:`~aerolume.api.Host`:

* headless (CLI, scripts, tests): the default host edits the store and runs
  solvers on the calling thread - :meth:`Workspace.open`;
* in the app: :class:`aerolume.app.scripting.AppHost`, which makes every change
  on the GUI thread through ``AppState`` and runs simulations through the
  window's ``JobManager`` (the user sees them in the tree, the status bar and on
  the Simulate page, and Stop stops them) - built by the Assistant dock.

:attr:`Workspace.run_script` lets the app run approved ``run_python`` code
through the Script console's engine (stdout routed per thread, Stop that also
interrupts plain Python); without it the tool runs the code itself.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ScriptRunner = Callable[[str, threading.Event, Callable[[float, str], None]], dict]


def inside(path: str | Path, root: str | Path) -> bool:
    """Whether ``path`` resolves inside ``root`` (after ``..``, symlinks, absolute paths)."""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False


def _nothing() -> None:
    return None


@dataclass
class Workspace:
    """The open project (an :class:`aerolume.api.Project`) plus the window's hooks."""

    project: Any
    experiments_changed: Callable[[], None] = _nothing
    run_script: ScriptRunner | None = None

    @property
    def store(self) -> Any:
        """The :class:`aerolume.project.Project` store behind the API project."""
        return self.project.raw

    @property
    def root(self) -> Path:
        return Path(self.project.path)

    @property
    def headless(self) -> bool:
        """True when runs are not shown live in a window (estimates leave out the live view)."""
        return not bool(getattr(self.project.host, "live_view", False))

    def busy(self, sim_id: str) -> bool:
        """Whether a simulation is running or queued in the window's job manager."""
        jobs = getattr(self.project.host, "jobs", None)
        if jobs is None:
            return False
        try:
            return bool(self.project.host.call(lambda: jobs.is_busy(sim_id)))
        except Exception:  # noqa: BLE001 - a job manager that cannot tell has nothing running for us
            return False

    @classmethod
    def open(cls, path_or_store: Any) -> Workspace:
        """A headless workspace on a project folder or an open project store."""
        from .. import api
        if isinstance(path_or_store, api.Project):
            return cls(path_or_store)
        if hasattr(path_or_store, "designs") and hasattr(path_or_store, "simulations"):
            return cls(api.Project(path_or_store))
        return cls(api.open_project(path_or_store))


__all__ = ["ScriptRunner", "Workspace", "inside"]
