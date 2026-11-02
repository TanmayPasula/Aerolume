"""User preferences that outlive a session (QSettings wrapper).

Everything the user sets once and expects to stick lives here: theme and
interface font, unit system, preferred compute device and precision, window
geometry, the dock workspaces (``workspaces/<name>``, JSON written by
:mod:`aerolume.app.widgets.docks`), whether the welcome tour ran, the last
folder used in a file dialog and a recent-projects list. Values are stored in
an INI file in the per-user config folder (``%APPDATA%/Aerolume/Aerolume.ini``
on Windows), which is easy to inspect or delete when something goes wrong.

Use the module-level :func:`get` singleton. Tests point it at a throw-away file
with :func:`use_file` so they never touch the real user's preferences.

``signals.changed(key)`` fires after a value is written through a property, so
pages can react to a unit-system or device change without polling.
"""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, QSettings, QStandardPaths, Signal

ORG = "Aerolume"
APP = "Aerolume"

THEMES = ("light", "dark", "system")      # "system" follows the OS light / dark setting
UI_FONTS = ("bundled", "system")
UNIT_SYSTEMS = ("SI", "imperial")
DEVICES = ("auto", "cuda", "cpu", "mps")
PRECISIONS = ("auto", "float32", "float64")
MAX_RECENT = 12

# Display unit per quantity kind (keys of aerolume.core.units.UNITS). Model data is
# always SI; this only decides what the user reads and types.
DISPLAY_UNITS = {
    "SI": {"speed": "m/s", "length": "m", "pressure": "Pa", "force": "N",
           "force_per_span": "N/m", "power": "W", "angle": "deg", "temperature": "C"},
    "imperial": {"speed": "mph", "length": "ft", "pressure": "psi", "force": "lbf",
                 "force_per_span": "N/m", "power": "hp", "angle": "deg", "temperature": "C"},
}


class _Signals(QObject):
    changed = Signal(str)       # settings key that was written


signals = _Signals()


class Settings:
    """Typed accessors over one QSettings store.

    Reads are validated against the allowed values, so a hand-edited or stale
    INI file falls back to defaults instead of crashing the app.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        # AEROLUME_CONFIG_DIR redirects the INI exactly as it redirects the project
        # store's recent list (aerolume.project.recent), so a test run or a scripted
        # session can never write into the real user's preferences.
        env_dir = os.environ.get("AEROLUME_CONFIG_DIR")
        if path is None and env_dir:
            path = Path(env_dir) / f"{APP}.ini"
        if path is None:
            self._qs = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, ORG, APP)
        else:
            self._qs = QSettings(str(path), QSettings.Format.IniFormat)

    # -- generic -----------------------------------------------------------
    @property
    def path(self) -> str:
        """The backing file (useful in the About dialog and bug reports)."""
        return self._qs.fileName()

    def value(self, key: str, default=None):
        return self._qs.value(key, default)

    def set_value(self, key: str, value) -> None:
        self._qs.setValue(key, value)
        self._qs.sync()
        signals.changed.emit(key)

    def _choice(self, key: str, allowed: tuple[str, ...], default: str) -> str:
        v = self._qs.value(key, default)
        return v if v in allowed else default

    # -- appearance ----------------------------------------------------------
    @property
    def theme(self) -> str:
        """"light" (default, the norm for engineering tools), "dark" (near-black) or "system"
        (whatever the operating system is set to; theme.apply resolves and follows it)."""
        return self._choice("appearance/theme", THEMES, "light")

    @theme.setter
    def theme(self, v: str) -> None:
        if v not in THEMES:
            raise ValueError(f"unknown theme {v!r}")
        self.set_value("appearance/theme", v)

    @property
    def ui_font(self) -> str:
        """"bundled" (Atkinson Hyperlegible, the default) or "system" (the platform font)."""
        return self._choice("appearance/font", UI_FONTS, "bundled")

    @ui_font.setter
    def ui_font(self, v: str) -> None:
        if v not in UI_FONTS:
            raise ValueError(f"unknown interface font {v!r}")
        self.set_value("appearance/font", v)

    def flag(self, key: str, default: bool = False) -> bool:
        """A boolean setting; QSettings hands INI booleans back as the strings "true" / "false"."""
        v = self._qs.value(key, default)
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(v)

    # -- units -------------------------------------------------------------
    @property
    def unit_system(self) -> str:
        """"SI" (default) or "imperial". Only affects display and entry, never storage."""
        return self._choice("units/system", UNIT_SYSTEMS, "SI")

    @unit_system.setter
    def unit_system(self, v: str) -> None:
        if v not in UNIT_SYSTEMS:
            raise ValueError(f"unknown unit system {v!r}")
        self.set_value("units/system", v)

    def display_unit(self, kind: str) -> str:
        """Unit to show for a quantity kind ("speed", "length", ...) in the active system."""
        return DISPLAY_UNITS[self.unit_system].get(kind, DISPLAY_UNITS["SI"].get(kind, ""))

    # -- compute -----------------------------------------------------------
    @property
    def device(self) -> str:
        """Preferred compute device for new simulations: "auto", "cuda", "cpu" or "mps"."""
        return self._choice("compute/device", DEVICES, "auto")

    @device.setter
    def device(self, v: str) -> None:
        if v not in DEVICES:
            raise ValueError(f"unknown device {v!r}")
        self.set_value("compute/device", v)

    @property
    def precision(self) -> str:
        """Preferred floating-point precision for new simulations."""
        return self._choice("compute/precision", PRECISIONS, "auto")

    @precision.setter
    def precision(self, v: str) -> None:
        if v not in PRECISIONS:
            raise ValueError(f"unknown precision {v!r}")
        self.set_value("compute/precision", v)

    # -- window ------------------------------------------------------------
    @property
    def window_geometry(self) -> QByteArray | None:
        v = self._qs.value("window/geometry")
        return v if isinstance(v, QByteArray) and not v.isEmpty() else None

    @window_geometry.setter
    def window_geometry(self, v: QByteArray) -> None:
        self._qs.setValue("window/geometry", v)

    @property
    def window_state(self) -> QByteArray | None:
        v = self._qs.value("window/state")
        return v if isinstance(v, QByteArray) and not v.isEmpty() else None

    @window_state.setter
    def window_state(self, v: QByteArray) -> None:
        self._qs.setValue("window/state", v)

    @property
    def messages_current_only(self) -> bool:
        """Whether the Messages dock shows only the selected simulation's lines (default on).

        QSettings hands booleans back from an INI file as the strings "true" /
        "false", so the value is parsed rather than passed to ``bool()``.
        """
        v = self._qs.value("window/messages_current_only", True)
        if isinstance(v, str):
            return v.strip().lower() not in ("false", "0", "no", "off")
        return bool(v)

    @messages_current_only.setter
    def messages_current_only(self, v: bool) -> None:
        self.set_value("window/messages_current_only", bool(v))

    # -- updates -------------------------------------------------------------
    @property
    def update_url(self) -> str:
        """The release feed Help ▸ Check for updates reads; "" (the default) turns the check off.

        Off by default because Aerolume has no published release feed yet: a
        built-in address would point at nothing. ``AEROLUME_UPDATE_URL`` sets it
        for a build that ships with one (an installer, a lab mirror).
        """
        v = str(self._qs.value("updates/url", "") or "").strip()
        return v or os.environ.get("AEROLUME_UPDATE_URL", "").strip()

    @update_url.setter
    def update_url(self, v: str) -> None:
        self.set_value("updates/url", str(v or "").strip())

    # -- folders -----------------------------------------------------------
    @property
    def last_dir(self) -> str:
        """Folder of the last file dialog; defaults to the projects folder."""
        v = self._qs.value("paths/last_dir", "")
        return v if v and Path(v).is_dir() else str(default_projects_dir())

    @last_dir.setter
    def last_dir(self, v: str | Path) -> None:
        p = Path(v)
        self.set_value("paths/last_dir", str(p if p.is_dir() else p.parent))

    # -- recent projects ----------------------------------------------------
    def recent_projects(self) -> list[str]:
        """Recently opened project folders that still exist, newest first."""
        raw = self._qs.value("paths/recent", [])
        if isinstance(raw, str):
            raw = [raw] if raw else []
        out: list[str] = []
        for p in raw or []:
            if p and p not in out and Path(p).exists():
                out.append(p)
        return out[:MAX_RECENT]

    def add_recent(self, path: str | Path) -> None:
        """Move a project folder to the top of the recent list."""
        p = str(Path(path))
        items = [x for x in self.recent_projects() if Path(x) != Path(p)]
        self._qs.setValue("paths/recent", [p, *items][:MAX_RECENT])
        self._qs.sync()
        signals.changed.emit("paths/recent")

    def remove_recent(self, path: str | Path) -> None:
        items = [x for x in self.recent_projects() if Path(x) != Path(path)]
        self._qs.setValue("paths/recent", items)
        self._qs.sync()
        signals.changed.emit("paths/recent")

    def clear_recent(self) -> None:
        self._qs.setValue("paths/recent", [])
        self._qs.sync()
        signals.changed.emit("paths/recent")

    def sync(self) -> None:
        self._qs.sync()


def default_projects_dir() -> Path:
    """``~/Documents/Aerolume`` (created lazily by whoever saves into it)."""
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    base = Path(docs) if docs else Path.home() / "Documents"
    return base / "Aerolume"


_instance: Settings | None = None


def get() -> Settings:
    """The process-wide settings object."""
    global _instance
    if _instance is None:
        _instance = Settings()
    return _instance


def use_file(path: str | Path | None) -> Settings:
    """Redirect the singleton to an INI file (tests, portable installs); None = default store."""
    global _instance
    _instance = Settings(path)
    return _instance
