"""Project persistence: the ``<Name>.aerolume/`` folder format (DESIGN.md 5.7),
crash-safe writes and the recent-projects list."""
from .fileio import ProjectCorruptError, ProjectError, ProjectExistsError, ProjectNotFoundError
from .recent import add_recent, clear_recent, config_dir, recent_projects, remove_recent
from .store import PROJECT_FORMAT, SUFFIX, Project

__all__ = [
    "PROJECT_FORMAT", "SUFFIX", "Project", "ProjectCorruptError", "ProjectError", "ProjectExistsError",
    "ProjectNotFoundError", "add_recent", "clear_recent", "config_dir", "recent_projects", "remove_recent",
]
