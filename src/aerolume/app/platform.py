"""Desktop-integration helpers: where files are, taskbar identity, one instance, crash reports.

Aerolume runs in two layouts and every file lookup has to work in both:

* **from source** (editable install): the package lives in ``<repo>/src/aerolume`` and the
  user-facing files (``docs/``, ``validation/``, ``LICENSE``) sit at the repository root;
* **frozen** (the PyInstaller build in ``packaging/``, DESIGN.md section 12): the package is
  unpacked under ``<install>/_internal`` (``sys._MEIPASS``) and the user-facing files are
  copied next to ``Aerolume.exe`` so a user can find the licence and the docs in Explorer.

:func:`app_root` hides that difference. Package data (icons, fonts, example projects) is
always found relative to the package itself (:func:`package_dir`, :func:`resource_path`), which
PyInstaller preserves because ``packaging/aerolume.spec`` collects the data at the same
relative paths.

The rest of the module is what makes a Python program behave like a Windows application:

* :func:`set_app_user_model_id` - without an explicit AppUserModelID Windows groups the window
  under ``python.exe`` with the Python icon, and taskbar pins / jump lists attach to Python.
  The installer's shortcuts carry the same ID (``packaging/aerolume.iss``), so a pinned
  shortcut and the running window are one taskbar button.
* :class:`SingleInstance` - double-clicking a project while Aerolume is open must open it in
  the running window, not start a second 2 GB torch/VTK process that fights the first for the
  GPU. A later launch forwards its arguments over a per-user local socket and exits.
* :func:`hold_app_mutex` - a named mutex the installer checks (Inno Setup ``AppMutex``) so
  setup and the uninstaller refuse to replace files of a running Aerolume.
* :func:`install_crash_handler` / :func:`write_crash_report` - a windowed executable has no
  console, so an uncaught exception would vanish. Every one is written to a log under the
  user config folder and handed to the UI as a copyable, privacy-redacted report.
* :func:`prepare_runtime` - in a windowed build ``sys.stdout``/``sys.stderr`` are ``None``;
  this points them at a log file and enables :mod:`faulthandler`, so a native crash (an
  OpenGL driver fault inside VTK, a CUDA error) still leaves a trace that the next start can
  report (:func:`previous_native_crash`).
* :func:`self_test` - the frozen build's ``--selftest`` mode (``packaging/entry.py``): imports
  torch and VTK, runs a few GPU and solver steps, renders one VTK frame, finds the resources
  and optionally builds the main window, then reports every check with timings.

Typical wiring in ``main.py`` (the order matters, see each function)::

    from . import platform as plat
    plat.prepare_runtime()                       # before anything can print
    plat.set_app_user_model_id()                 # before the first window exists
    app = create_app(argv)
    plat.install_crash_handler(on_crash)         # on_crash(report) runs on the GUI thread
    guard = plat.SingleInstance()
    if not guard.start(argv[1:]):                # forwarded to the running instance
        return 0
    guard.message_received.connect(win.handle_forwarded_argv)
    plat.hold_app_mutex()
    project = plat.resolve_open_target(args.project)   # project.aerolume marker -> folder
    plat.note_project_opened(project_folder)     # after every open/create/save-as: writes the
                                                 # marker the association needs + jump list

Qt is imported at module level (QtCore only; QtWidgets lazily) because :class:`SingleInstance`
is a QObject with a signal; nothing here imports torch or VTK except :func:`self_test`.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import faulthandler
import hashlib
import json
import os
import platform as _pyplatform
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray, QLockFile, QObject, Signal

from .. import __version__

#: Windows AppUserModelID. Must match ``packaging/aerolume.iss`` (shortcut ``AppUserModelID``).
APP_USER_MODEL_ID = "Aerolume.Aerolume.Desktop"
#: Named mutex held while the app runs. Must match ``AppMutex`` in ``packaging/aerolume.iss``.
APP_MUTEX_NAME = "Aerolume.Aerolume.Desktop.Running"
#: Marker file a project folder may contain; the installer associates ``.aerolume`` files with
#: the app, so double-clicking ``<name>.aerolume/project.aerolume`` opens the folder.
PROJECT_MARKER = "project.aerolume"
#: Index file every project folder has (``aerolume.project.store``).
PROJECT_INDEX = "project.json"

#: Crash logs kept in ``logs/``; older ones are removed so a crash loop cannot fill the disk.
MAX_CRASH_LOGS = 20
#: The application log (stdout/stderr of a windowed build) is rotated beyond this size.
MAX_APP_LOG_BYTES = 2 * 1024 * 1024
#: Largest message a later launch may forward (argv + cwd); anything bigger is dropped.
MAX_FORWARD_BYTES = 64 * 1024

APP_LOG = "aerolume.log"
NATIVE_CRASH_LOG = "native-crash.log"


# ---------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------

def is_frozen() -> bool:
    """True inside a PyInstaller build (``sys.frozen`` is set by the bootloader)."""
    return bool(getattr(sys, "frozen", False))


def package_dir() -> Path:
    """The ``aerolume`` package folder (``src/aerolume`` or ``_internal/aerolume``)."""
    return Path(__file__).resolve().parents[1]


def bundle_dir() -> Path:
    """Where the frozen build unpacked its files (``sys._MEIPASS``); the ``src`` folder otherwise."""
    meipass = getattr(sys, "_MEIPASS", None)
    if is_frozen() and meipass:
        return Path(meipass)
    return package_dir().parent


def app_root() -> Path:
    """The folder holding ``docs/``, ``validation/``, ``LICENSE`` and ``THIRD_PARTY_NOTICES.md``.

    Frozen: the folder of ``Aerolume.exe`` (``packaging/build.py`` copies the files there so
    they are visible in the install folder). From source: the repository root, i.e. two levels
    above the package (``<repo>/src/aerolume``). Use this instead of ``Path(__file__).parents[n]``,
    which happens to agree for some modules and not others.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return package_dir().parents[1]


def resource_path(*parts: str) -> Path:
    """A file under ``aerolume/app/resources`` (icons, fonts, the app icon); may not exist."""
    return package_dir().joinpath("app", "resources", *parts)


def app_file(*parts: str) -> Path | None:
    """``app_root()/parts`` if it exists, else ``None`` (docs, licences, validation scripts)."""
    p = app_root().joinpath(*parts)
    return p if p.exists() else None


def docs_dir() -> Path:
    """The documentation folder (``docs/`` in the repo or next to the executable)."""
    return app_root() / "docs"


def examples_dir() -> Path:
    """Bundled example projects (package data ``aerolume/examples/projects``)."""
    return package_dir() / "examples" / "projects"


def config_dir() -> Path:
    """The per-user configuration folder; delegates to :func:`aerolume.project.recent.config_dir`."""
    from ..project.recent import config_dir as _config_dir
    return _config_dir()


def logs_dir(create: bool = True) -> Path:
    """``<config>/logs``: crash reports, the application log and the native-crash trace.

    Falls back to the temporary folder when the config folder cannot be created (a locked-down
    profile), because losing a crash report is worse than writing it somewhere unusual.
    """
    d = config_dir() / "logs"
    if create:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile
            d = Path(tempfile.gettempdir()) / "Aerolume" / "logs"
            d.mkdir(parents=True, exist_ok=True)
    return d


_BUILD_INFO: dict | None = None


def build_info() -> dict:
    """``build-info.json`` next to the frozen executable, or ``{}`` from source / when missing.

    ``packaging/build.py`` writes it with the version, variant, build time, library versions
    and the SHA-256 fingerprint of the sources the build was made from. It is what lets a
    crash report or an About box say *which* build is running, beyond a version number that
    can be reused by many builds. Read once and cached; never raises.
    """
    global _BUILD_INFO
    if _BUILD_INFO is None:
        data: dict = {}
        if is_frozen():
            with contextlib.suppress(OSError, ValueError):
                loaded = json.loads((app_root() / "build-info.json").read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
        _BUILD_INFO = data
    return dict(_BUILD_INFO)


def version_label(*, detailed: bool = False) -> str:
    """The version string for people: ``0.1.0``; with ``detailed``, which build it is.

    Detailed examples: ``0.1.0 (from source)`` and ``0.1.0 (CUDA build 2026-10-31, source
    3f2a9c1b7d0e)``. The short fingerprint is what a bug report needs to identify the exact
    sources, because builds made on different days can share a version number.
    """
    if not detailed:
        return __version__
    if not is_frozen():
        return f"{__version__} (from source)"
    info = build_info()
    bits = []
    variant = str(info.get("variant") or "").upper()
    built = str(info.get("built") or "")[:10]
    if variant or built:
        bits.append(" ".join(b for b in (variant, "build" if variant else "built", built) if b))
    src = str(info.get("source_sha256") or "")[:12]
    if src:
        bits.append(f"source {src}")
    return f"{__version__} ({', '.join(bits)})" if bits else f"{__version__} (installed build)"


def resolve_open_target(arg: str | os.PathLike | None, cwd: str | os.PathLike | None = None) -> Path | None:
    """Map a command-line argument to the project folder it means.

    Accepts a project folder, its ``project.json`` index, the ``project.aerolume`` marker (what
    a double-click on an associated file passes) or any other ``*.aerolume`` file inside a
    project folder. Relative paths resolve against ``cwd`` (default: the current folder); this
    matters for arguments forwarded from another process whose working folder differs.
    Returns ``None`` for an empty argument. A path that does not exist is returned unchanged
    (made absolute) so the caller can report it with the user's own spelling.
    """
    if arg is None or str(arg).strip() == "":
        return None
    p = Path(arg).expanduser()
    if not p.is_absolute():
        p = Path(cwd or os.getcwd()) / p
    named_like_project_file = p.name.lower() == PROJECT_INDEX or p.suffix.lower() == ".aerolume"
    if p.is_file() and named_like_project_file and (
            (p.parent / PROJECT_INDEX).is_file() or p.name.lower() == PROJECT_MARKER):
        return p.parent
    return p


# ---------------------------------------------------------------------------
# Windows identity
# ---------------------------------------------------------------------------

def set_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Give the process its own taskbar identity; call before the first window is shown.

    Returns True when Windows accepted the ID (False on other platforms or on failure, which
    only costs the taskbar grouping, never the start-up).
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        hr = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id))
        return hr == 0
    except (AttributeError, OSError):
        return False


def current_app_user_model_id() -> str | None:
    """The process's explicit AppUserModelID (Windows), for tests and the self-test."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        out = ctypes.c_wchar_p()
        fn = ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID
        fn.argtypes = [ctypes.POINTER(ctypes.c_wchar_p)]
        if fn(ctypes.byref(out)) != 0 or not out.value:
            return None
        value = out.value
        ctypes.windll.ole32.CoTaskMemFree(ctypes.cast(out, wintypes.LPVOID))
        return value
    except (AttributeError, OSError):
        return None


_mutex_handle: list[int] = []


def hold_app_mutex(name: str = APP_MUTEX_NAME) -> bool:
    """Create the named mutex the installer looks for (Inno Setup ``AppMutex``); idempotent.

    The handle stays open for the life of the process and Windows releases it when the process
    ends, crash or not, so there is nothing to clean up. Returns True when held.
    """
    if _mutex_handle:
        return True
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        h = kernel32.CreateMutexW(None, False, ctypes.c_wchar_p(name))
    except (AttributeError, OSError):
        return False
    if not h:
        return False
    _mutex_handle.append(h)
    return True


def mutex_exists(name: str = APP_MUTEX_NAME) -> bool:
    """True if some process holds the named mutex (what setup checks before installing)."""
    if sys.platform != "win32":
        return False
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    synchronize = 0x00100000
    h = kernel32.OpenMutexW(synchronize, False, ctypes.c_wchar_p(name))
    if not h:
        return False
    kernel32.CloseHandle(ctypes.c_void_p(h))
    return True


# ---------------------------------------------------------------------------
# project marker and the taskbar jump list
# ---------------------------------------------------------------------------

#: Content of a new marker file. Read by nobody (the folder's ``project.json`` is the project);
#: the text only tells a user who opens the file in Notepad what it is for.
_MARKER_TEXT = (
    "Aerolume project marker.\r\n"
    "Double-click this file to open the project folder it is in. The project itself is\r\n"
    "project.json and the subfolders; this file holds no data and may be deleted.\r\n"
)


def ensure_project_marker(folder: str | os.PathLike) -> Path | None:
    """Create ``<folder>/project.aerolume`` if the folder is a project and has none yet.

    Windows associates file types, not folders, so the installer's ``.aerolume`` association
    can only reach a project through a file inside it. The marker carries no data: it is never
    overwritten, and a read-only or vanished folder is not an error (the project still opens
    from the app; only the double-click shortcut is missing). Returns the marker path, or
    ``None`` when ``folder`` is not a project folder or the file could not be written.
    """
    p = Path(folder)
    if not (p / PROJECT_INDEX).is_file():
        return None
    marker = p / PROJECT_MARKER
    if marker.is_file():
        return marker
    try:
        with open(marker, "x", encoding="utf-8", newline="") as fh:
            fh.write(_MARKER_TEXT)
    except FileExistsError:
        return marker
    except OSError:
        return None
    return marker


def add_recent_document(path: str | os.PathLike, *, force: bool = False) -> bool:
    """Put ``path`` into the Windows recent list, which feeds the taskbar jump list.

    ``SHAddToRecentDocs`` files the entry under the calling process's AppUserModelID, and the
    shell shows it in that app's jump list ("Recent") only when the file type is registered to
    the app, which the installer does for ``.aerolume`` (``packaging/aerolume.iss``). That is
    why this is a no-op from source (``force=False``): a developer run would only add entries to
    the user's Explorer "Recent" list under python.exe. Pass the ``project.aerolume`` marker
    (see :func:`note_project_opened`), not the folder: a folder entry would open Explorer.
    Returns True when the call was made.
    """
    if sys.platform != "win32" or not (force or is_frozen()):
        return False
    p = Path(path)
    if not p.is_file():
        return False
    try:
        import ctypes
        shard_pathw = 0x00000003
        fn = ctypes.windll.shell32.SHAddToRecentDocs
        fn.argtypes = [ctypes.c_uint, ctypes.c_wchar_p]
        fn.restype = None
        fn(shard_pathw, str(p.resolve()))
    except (AttributeError, OSError):
        return False
    return True


def note_project_opened(folder: str | os.PathLike) -> Path | None:
    """Everything the desktop needs when a project is opened or saved: marker + jump list.

    Call after a successful open, create or "save as" (``main_window`` and the start page).
    Cheap (two stat calls once the marker exists) and never raises.
    """
    try:
        marker = ensure_project_marker(folder)
        if marker is not None:
            add_recent_document(marker)
        return marker
    except Exception:  # noqa: BLE001 - desktop sugar must never break opening a project
        return None


def _allow_set_foreground(pid: int) -> None:
    """Let ``pid`` (the running instance) take the foreground when it opens a forwarded project.

    Windows only lets the process that received the user's click bring a window to the front;
    without this hand-over the running instance would only flash in the taskbar.
    """
    if sys.platform != "win32" or pid <= 0:
        return
    with contextlib.suppress(AttributeError, OSError):
        import ctypes
        ctypes.windll.user32.AllowSetForegroundWindow(int(pid))


def activate_window(win) -> None:
    """Bring ``win`` (a QWidget) to the front: un-minimise, raise, focus.

    Call from the slot connected to :attr:`SingleInstance.message_received`, after opening
    the forwarded project, so the user sees where their double-click went.
    """
    from PySide6.QtCore import Qt
    state = win.windowState()
    if state & Qt.WindowState.WindowMinimized:
        win.setWindowState(state & ~Qt.WindowState.WindowMinimized)
    win.show()
    win.raise_()
    win.activateWindow()
    if sys.platform == "win32":
        with contextlib.suppress(AttributeError, OSError, TypeError, ValueError):
            import ctypes
            ctypes.windll.user32.SetForegroundWindow(int(win.winId()))


# ---------------------------------------------------------------------------
# single instance
# ---------------------------------------------------------------------------

def _instance_key() -> str:
    """Per user and per config folder: two users, or a test using AEROLUME_CONFIG_DIR, never collide."""
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    with contextlib.suppress(Exception):
        import getpass
        user = getpass.getuser()
    seed = f"{APP_USER_MODEL_ID}|{user}|{os.path.normcase(str(config_dir()))}"
    return "aerolume-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _absolutise(argv: Sequence[str], cwd: str) -> list[str]:
    """Make path-like arguments absolute in the sender, whose working folder the receiver lacks."""
    out: list[str] = []
    for a in argv:
        s = str(a)
        if s and not s.startswith("-"):
            p = Path(s).expanduser()
            if not p.is_absolute() and (Path(cwd) / p).exists():
                s = str((Path(cwd) / p).resolve())
        out.append(s)
    return out


class SingleInstance(QObject):
    """One running Aerolume per user; later launches forward their arguments to it.

    The first process takes a lock file (``<config>/aerolume-<key>.lock``; a lock whose owner
    PID is gone counts as stale, so a crash never blocks the next start) and listens on a local socket
    (a named pipe on Windows) restricted to the current user. A later process finds the lock
    taken, connects, receives the running instance's PID (to hand it the foreground right),
    sends one JSON line ``{"argv": [...], "cwd": "..."}`` and waits for ``{"ok": true}``.

    Failure is never fatal: if the running instance does not answer within the timeout (it is
    still starting, or it hung), :meth:`start` returns True and the new process simply runs
    as a second instance - the user gets a window either way.

    Signals:
        message_received(list): the forwarded argv (paths already absolute), emitted on the
            GUI thread of the first instance.
    """

    message_received = Signal(list)

    def __init__(self, key: str | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._key = key or _instance_key()
        self._lock: QLockFile | None = None
        self._server = None
        self._buffers: dict[Any, bytearray] = {}
        self.is_primary = False
        self.forwarded = False

    @property
    def server_name(self) -> str:
        """The local-socket name (``\\\\.\\pipe\\<name>`` on Windows)."""
        return self._key

    def _lock_path(self) -> Path:
        try:
            d = config_dir()
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile
            d = Path(tempfile.gettempdir())
        return d / f"{self._key}.lock"

    def try_primary(self) -> bool:
        """Take the instance lock and start listening; False if another instance holds it."""
        if self.is_primary:
            return True
        lock = QLockFile(str(self._lock_path()))
        lock.setStaleLockTime(0)                 # stale only when the owning PID is gone
        # a short retry, not a single attempt: right after a crash the lock file can be held for a
        # moment by the exiting process or a virus scanner, and a single tryLock(0) then took the
        # stale lock for a live one (seen once in the full test suite). A second launch while the
        # first runs pays these 250 ms before it forwards its arguments.
        if not lock.tryLock(250):
            return False
        self._lock = lock
        self._listen()
        self.is_primary = True
        return True

    def _listen(self) -> None:
        from PySide6.QtNetwork import QLocalServer
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        QLocalServer.removeServer(self._key)     # a stale Unix socket file from a crashed run
        if not server.listen(self._key):
            # Another process owns the name without holding the lock (should not happen); run
            # without forwarding rather than fail.
            server.deleteLater()
            return
        server.newConnection.connect(self._on_connection)
        self._server = server

    def _on_connection(self) -> None:
        while self._server is not None and self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            self._buffers[sock] = bytearray()
            sock.readyRead.connect(lambda s=sock: self._on_ready_read(s))
            sock.disconnected.connect(lambda s=sock: self._drop(s))
            sock.write(QByteArray((json.dumps({"pid": os.getpid()}) + "\n").encode("utf-8")))
            sock.flush()

    def _drop(self, sock) -> None:
        self._buffers.pop(sock, None)
        sock.deleteLater()

    def _on_ready_read(self, sock) -> None:
        buf = self._buffers.get(sock)
        if buf is None:
            return
        buf.extend(bytes(sock.readAll().data()))
        if len(buf) > MAX_FORWARD_BYTES:
            self._buffers.pop(sock, None)
            sock.abort()
            return
        if b"\n" not in buf:
            return
        line = bytes(buf.split(b"\n", 1)[0])
        self._buffers[sock] = bytearray()
        try:
            msg = json.loads(line.decode("utf-8"))
            argv = [str(a) for a in msg.get("argv", [])]
            cwd = str(msg.get("cwd") or "")
        except (ValueError, AttributeError, TypeError):
            sock.write(QByteArray(b'{"ok": false}\n'))
            sock.flush()
            sock.disconnectFromServer()
            return
        sock.write(QByteArray(b'{"ok": true}\n'))
        sock.flush()
        sock.disconnectFromServer()
        self.message_received.emit(_absolutise(argv, cwd) if cwd else argv)

    def forward(self, argv: Sequence[str], timeout_ms: int = 4000) -> bool:
        """Send ``argv`` to the running instance (blocking, no event loop needed).

        Retries the connection until ``timeout_ms`` because the first instance may hold the
        lock while it is still importing torch and has not started listening yet. Returns True
        only when the running instance acknowledged the message.
        """
        from PySide6.QtNetwork import QLocalSocket
        cwd = os.getcwd()
        payload = (json.dumps({"argv": _absolutise(list(argv), cwd), "cwd": cwd}) + "\n").encode("utf-8")
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            sock = QLocalSocket()
            sock.connectToServer(self._key)
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            if sock.waitForConnected(min(remaining, 1000)):
                try:
                    return self._exchange(sock, payload, deadline)
                finally:
                    sock.abort()
            sock.abort()
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    @staticmethod
    def _read_line(sock, deadline: float) -> bytes | None:
        while not sock.canReadLine():
            remaining = int((deadline - time.monotonic()) * 1000)
            waited = remaining > 0 and sock.waitForReadyRead(min(remaining, 500))
            if not waited and time.monotonic() >= deadline:
                return None
        return bytes(sock.readLine().data())

    def _exchange(self, sock, payload: bytes, deadline: float) -> bool:
        hello = self._read_line(sock, deadline)
        if hello is None:
            return False
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            _allow_set_foreground(int(json.loads(hello.decode("utf-8")).get("pid", 0)))
        sock.write(QByteArray(payload))
        if not sock.waitForBytesWritten(max(1, int((deadline - time.monotonic()) * 1000))):
            return False
        ack = self._read_line(sock, deadline)
        if ack is None:
            return False
        try:
            return bool(json.loads(ack.decode("utf-8")).get("ok"))
        except (ValueError, AttributeError):
            return False

    def start(self, argv: Sequence[str], timeout_ms: int = 4000) -> bool:
        """Become the primary instance, or forward ``argv`` to it.

        Returns True if this process should carry on starting (it is the first instance, or
        the first instance did not answer), False if the arguments were delivered and this
        process should exit with code 0.
        """
        if self.try_primary():
            return True
        if self.forward(argv, timeout_ms):
            self.forwarded = True
            return False
        return True

    def close(self) -> None:
        """Stop listening and release the lock (also happens automatically at process exit)."""
        if self._server is not None:
            self._server.close()
            self._server.deleteLater()
            self._server = None
        if self._lock is not None:
            self._lock.unlock()
            self._lock = None
        self.is_primary = False


# ---------------------------------------------------------------------------
# crash reports
# ---------------------------------------------------------------------------

@dataclass
class CrashReport:
    """An uncaught exception, formatted for a dialog and saved to ``path``.

    ``text`` is redacted (the home folder becomes ``~``) so a user can paste it into a public
    issue; ``path`` is None when the log could not be written. ``summary`` is the one-line
    ``Type: message`` for a dialog title or status line.
    """

    text: str
    path: Path | None
    summary: str
    thread: str = "MainThread"
    when: str = field(default_factory=lambda: _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC"))


def redact(text: str) -> str:
    """Replace the user's home folder (and user name in paths) with ``~`` / ``<user>``."""
    home = str(Path.home())
    variants = {home, home.replace("\\", "/"), os.path.normcase(home)}
    for v in sorted(variants, key=len, reverse=True):
        if v and len(v) > 3:
            text = re.sub(re.escape(v), "~", text, flags=re.IGNORECASE)
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if user and len(user) > 2:
        text = re.sub(r"([\\/])" + re.escape(user) + r"(?=[\\/])", r"\1<user>", text, flags=re.IGNORECASE)
    return text


def environment_lines() -> list[str]:
    """Version facts for a report, from modules already loaded (never imports torch or VTK).

    A crash handler that imports torch to describe the GPU could itself crash or take seconds;
    so only what is in ``sys.modules`` is described, and CUDA is queried only if initialised.
    """
    lines = [f"Aerolume {version_label(detailed=True)}",
             f"Python {sys.version.split()[0]} {_pyplatform.architecture()[0]}",
             f"OS {_pyplatform.platform()}"]
    mods = sys.modules
    if "PySide6.QtCore" in mods:
        with contextlib.suppress(Exception):
            from PySide6 import __version__ as pyside_version
            from PySide6.QtCore import qVersion
            lines.append(f"Qt {qVersion()} / PySide6 {pyside_version}")
    if "numpy" in mods:
        lines.append(f"NumPy {getattr(mods['numpy'], '__version__', '?')}")
    torch = mods.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            cuda = getattr(torch.version, "cuda", None)
            line = f"PyTorch {torch.__version__} (CUDA build {cuda or 'none'})"
            if cuda and torch.cuda.is_initialized():
                line += f", device {torch.cuda.get_device_name(torch.cuda.current_device())}"
            lines.append(line)
    vtk = mods.get("vtkmodules.vtkCommonCore")
    if vtk is not None:
        with contextlib.suppress(Exception):
            lines.append(f"VTK {vtk.vtkVersion.GetVTKVersion()}")
    return lines


def format_crash_report(exc_type, exc, tb, *, thread: str | None = None, context: str = "") -> str:
    """The full report text (redacted): summary, environment, traceback, optional context."""
    when = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts = [f"{exc_type.__name__}: {exc}", "", f"Time: {when}", f"Thread: {thread or threading.current_thread().name}"]
    parts += environment_lines()
    if context:
        parts += ["", context.rstrip()]
    parts += ["", "Traceback:", "".join(traceback.format_exception(exc_type, exc, tb)).rstrip()]
    return redact("\n".join(parts)) + "\n"


def _local_now() -> _dt.datetime:
    """Local wall-clock time (file names and log headers are read by the user, in their zone)."""
    return _dt.datetime.now(_dt.UTC).astimezone()


def _prune_logs(folder: Path, pattern: str, keep: int) -> None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        with contextlib.suppress(OSError):
            old.unlink()


def write_crash_report(exc_type, exc, tb, *, thread: str | None = None, context: str = "") -> CrashReport:
    """Format the exception, save it as ``logs/crash-<time>.log`` and return the report.

    Never raises: a crash handler that crashes hides the original error.
    """
    text = format_crash_report(exc_type, exc, tb, thread=thread, context=context)
    path: Path | None = None
    try:
        d = logs_dir()
        stamp = _local_now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        # Exclusive create, with a counter when the name is taken: two reports in the same
        # millisecond (a burst of worker-thread errors) used to share a name, and the second
        # silently overwrote the first (seen as 19 of 20 logs in tests/test_packaging.py).
        n = 1
        while True:
            path = d / (f"crash-{stamp}.log" if n == 1 else f"crash-{stamp}-{n}.log")
            try:
                with open(path, "x", encoding="utf-8") as fh:
                    fh.write(text)
                break
            except FileExistsError:
                n += 1
        _prune_logs(d, "crash-*.log", MAX_CRASH_LOGS)
    except OSError:
        path = None
    if path is not None:
        text += f"\nThis report was saved to {redact(str(path))}\n"
    return CrashReport(text=text, path=path, summary=redact(f"{exc_type.__name__}: {exc}")[:300],
                       thread=thread or threading.current_thread().name)


_crash_state: dict[str, Any] = {"callback": None, "relay": None, "count": 0, "installed": False}
#: Reports written per process; past this, exceptions are still logged to stderr but not saved,
#: so a paint-event exception repeating at 60 Hz cannot write thousands of files.
MAX_REPORTS_PER_SESSION = 25


class _CrashRelay(QObject):
    """Carries a report from any thread to the GUI thread (the callback touches widgets)."""

    report = Signal(object)


def _deliver(report: CrashReport) -> None:
    cb = _crash_state.get("callback")
    if cb is None:
        return
    relay = _crash_state.get("relay")
    if relay is not None and threading.current_thread() is not threading.main_thread():
        relay.report.emit(report)                # queued to the relay's (GUI) thread
        return
    with contextlib.suppress(Exception):
        cb(report)


def _handle(exc_type, exc, tb, thread: str | None) -> None:
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        return
    if _crash_state["count"] >= MAX_REPORTS_PER_SESSION:
        return
    _crash_state["count"] += 1
    report = write_crash_report(exc_type, exc, tb, thread=thread)
    _deliver(report)


def install_crash_handler(callback: Callable[[CrashReport], None] | None = None) -> None:
    """Save every uncaught exception (main thread, Qt slots, worker threads) as a crash report.

    ``callback(report)`` is called on the GUI thread (when a QApplication exists at install
    time) so it may show a dialog; pass None to only log. Calling again replaces the callback
    without stacking hooks. The previous ``sys.excepthook`` still runs first, so the traceback
    keeps reaching the console and any hook installed earlier (main.py's Messages routing).
    """
    _crash_state["callback"] = callback
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance()
    if app is not None and _crash_state["relay"] is None:
        relay = _CrashRelay()
        # objects live on the thread that made them, so only a call from a worker needs a move
        # (QObject.thread() wrappers can tie the main QThread's lifetime to a Python wrapper,
        # see scripting.AppHost)
        if threading.current_thread() is not threading.main_thread():
            relay.moveToThread(app.thread())
        relay.report.connect(callback_or_none)
        _crash_state["relay"] = relay
    if _crash_state["installed"]:
        return
    _crash_state["installed"] = True
    previous = sys.excepthook
    previous_thread = threading.excepthook

    def excepthook(exc_type, exc, tb) -> None:
        with contextlib.suppress(Exception):
            previous(exc_type, exc, tb)
        _handle(exc_type, exc, tb, threading.current_thread().name)

    def thread_hook(args) -> None:
        with contextlib.suppress(Exception):
            previous_thread(args)
        if args.exc_type is not None:
            _handle(args.exc_type, args.exc_value, args.exc_traceback,
                    args.thread.name if args.thread is not None else "worker")

    excepthook._aerolume_crash = True             # type: ignore[attr-defined]
    sys.excepthook = excepthook
    threading.excepthook = thread_hook


def callback_or_none(report: CrashReport) -> None:
    """Run the current crash callback (used by the cross-thread relay)."""
    cb = _crash_state.get("callback")
    if cb is not None:
        with contextlib.suppress(Exception):
            cb(report)


def show_fatal_error(title: str, text: str) -> None:
    """Tell the user about a failure that stops start-up, with or without a QApplication.

    Uses a Qt message box when an application object exists, else the Win32 ``MessageBoxW``
    (the windowed build has no console to print to), else stderr.
    """
    with contextlib.suppress(Exception):
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, title, text.splitlines()[0] if text else title)
            box.setDetailedText(text)
            box.exec()
            return
    if sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        with contextlib.suppress(Exception):
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text[:4000], title, 0x10)
            return
    with contextlib.suppress(Exception):
        sys.stderr.write(f"{title}\n{text}\n")


# ---------------------------------------------------------------------------
# runtime preparation (windowed builds)
# ---------------------------------------------------------------------------

_runtime: dict[str, Any] = {"prepared": False, "previous_native": None, "fault_file": None}


def _rotate(path: Path, limit: int) -> None:
    with contextlib.suppress(OSError):
        if path.exists() and path.stat().st_size > limit:
            backup = path.with_name(path.name + ".1")
            if backup.exists():
                backup.unlink()
            path.rename(backup)


def _close_fault_file() -> None:
    """At a clean exit the native-crash trace is discarded: only a crash leaves it behind."""
    f = _runtime.get("fault_file")
    if f is None:
        return
    with contextlib.suppress(Exception):
        faulthandler.disable()
    with contextlib.suppress(Exception):
        f.close()
    with contextlib.suppress(OSError):
        Path(f.name).unlink()
    _runtime["fault_file"] = None


def prepare_runtime() -> None:
    """Make a windowed build observable; idempotent and safe to call from source.

    * ``sys.stdout``/``sys.stderr`` are ``None`` in a PyInstaller windowed executable, so a
      library's ``print`` or warning is lost and code writing to ``sys.stderr`` raises. They
      are pointed at ``logs/aerolume.log`` (rotated at 2 MB).
    * :mod:`faulthandler` writes Python tracebacks of a native crash to
      ``logs/native-crash.log``. The file is deleted at a clean exit, so if it exists at the
      next start the previous session died in native code; its text is kept for
      :func:`previous_native_crash`.
    """
    if _runtime["prepared"]:
        return
    _runtime["prepared"] = True
    try:
        d = logs_dir()
    except OSError:
        return
    if sys.stdout is None or sys.stderr is None:
        log = d / APP_LOG
        _rotate(log, MAX_APP_LOG_BYTES)
        with contextlib.suppress(OSError):
            stream = open(log, "a", encoding="utf-8", errors="backslashreplace", buffering=1)  # noqa: SIM115
            stream.write(f"\n--- Aerolume {version_label(detailed=True)} started "
                         f"{_local_now().strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} ---\n")
            if sys.stdout is None:
                sys.stdout = stream
            if sys.stderr is None:
                sys.stderr = stream
    native = d / NATIVE_CRASH_LOG
    with contextlib.suppress(OSError, UnicodeDecodeError):
        if native.exists():
            text = native.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                _runtime["previous_native"] = redact(text)
                mtime = _dt.datetime.fromtimestamp(native.stat().st_mtime, tz=_dt.UTC).astimezone()
                stamp = mtime.strftime("%Y%m%d-%H%M%S")
                native.replace(d / f"native-crash-{stamp}.log")
                _prune_logs(d, "native-crash-*.log", MAX_CRASH_LOGS)
    if faulthandler.is_enabled() and _runtime["fault_file"] is None and not is_frozen():
        return                                   # pytest / a developer already enabled it
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        f = open(native, "w", encoding="utf-8")  # noqa: SIM115 - must stay open for the process
        faulthandler.enable(file=f, all_threads=True)
        _runtime["fault_file"] = f
        import atexit
        atexit.register(_close_fault_file)


def previous_native_crash() -> str | None:
    """The native-crash trace left by the previous session, if it ended in a native crash.

    Only meaningful after :func:`prepare_runtime`. Note that Windows' vectored exception
    handler also records first-chance C++ exceptions that a library caught, so the text may
    show ``Windows fatal exception: code 0xe06d7363`` lines; the session still did not exit
    cleanly, which is the fact this reports.
    """
    return _runtime.get("previous_native")


# ---------------------------------------------------------------------------
# self-test (the frozen build's --selftest)
# ---------------------------------------------------------------------------

#: Standard-library modules a user script in the Script console commonly imports. The frozen
#: build only contains modules the analysis found, so packaging/aerolume.spec adds the whole
#: standard library; this list proves it.
SCRIPT_STDLIB = ("csv", "json", "sqlite3", "statistics", "decimal", "fractions", "itertools", "functools",
                 "collections", "dataclasses", "xml.etree.ElementTree", "urllib.request", "http.client",
                 "zipfile", "tarfile", "gzip", "lzma", "bz2", "uuid", "hashlib", "ssl", "ctypes",
                 "concurrent.futures", "asyncio", "unittest", "pickle", "shelve", "tomllib", "argparse",
                 "logging.handlers", "email.message", "html.parser", "difflib", "pprint", "timeit")
#: Third-party modules scripts are told they can use (docs/scripting.md).
SCRIPT_THIRD_PARTY = ("numpy", "scipy.optimize", "scipy.interpolate", "scipy.signal", "scipy.integrate",
                      "scipy.stats", "scipy.spatial", "matplotlib.figure", "PIL.Image")


@dataclass
class _Check:
    name: str
    ok: bool
    detail: str
    seconds: float


def _run_check(name: str, fn: Callable[[], str], out: list[_Check]) -> None:
    t0 = time.perf_counter()
    try:
        detail = fn()
        ok = True
    except Exception as exc:  # noqa: BLE001 - every failure is reported, none stops the test
        detail = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=6)
        detail += "\n" + tb[-1500:]
        ok = False
    out.append(_Check(name, ok, redact(detail), round(time.perf_counter() - t0, 3)))


def _check_resources() -> str:
    svg = resource_path("aerolume.svg")
    if not svg.is_file():
        raise FileNotFoundError(f"app icon missing: {svg}")
    icons = sorted(resource_path("icons").glob("*.svg"))
    if len(icons) < 10:
        raise FileNotFoundError(f"only {len(icons)} icons in {resource_path('icons')}")
    fonts = sorted(p.name for p in resource_path().rglob("*") if p.suffix.lower() in (".ttf", ".otf", ".woff", ".woff2"))
    projects = [p for p in examples_dir().glob("*") if (p / PROJECT_INDEX).is_file()] if examples_dir().is_dir() else []
    if not projects:
        raise FileNotFoundError(f"no example projects in {examples_dir()}")
    return f"{len(icons)} icons, {len(fonts)} font files, {len(projects)} example projects"


def _check_app_files() -> str:
    missing = [n for n in ("LICENSE", "THIRD_PARTY_NOTICES.md", "docs/user_guide.md", "validation/ns2d_cases.py")
               if app_file(*n.split("/")) is None]
    if missing:
        raise FileNotFoundError(f"missing next to the app ({app_root()}): {', '.join(missing)}")
    n_docs = len(list(docs_dir().glob("*.md")))
    return f"licence, notices, {n_docs} docs and the validation scripts found in {app_root()}"


def _check_build_info() -> str:
    """The frozen folder's ``build-info.json`` must exist and name the version it bundles.

    A mismatch means the folder was assembled from two builds (a copied-over executable, a
    half-finished update); a missing fingerprint means nothing ties it to its sources.
    """
    info = build_info()
    if not info:
        raise FileNotFoundError(f"no readable build-info.json in {app_root()}")
    if info.get("version") != __version__:
        raise RuntimeError(f"build-info.json says version {info.get('version')!r}, the bundled package is {__version__}")
    if not info.get("source_sha256"):
        raise RuntimeError("build-info.json records no source fingerprint")
    note = "; sources changed during the build" if info.get("source_changed_during_build") else ""
    return f"{version_label(detailed=True)}{note}"


def _check_imports(names: Sequence[str]) -> str:
    import importlib
    missing = []
    for n in names:
        try:
            importlib.import_module(n)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{n} ({type(exc).__name__}: {exc})")
    if missing:
        raise ImportError("; ".join(missing))
    return f"{len(names)} modules import"


def _check_metadata() -> str:
    """Installed-package metadata (``importlib.metadata``) must survive freezing.

    ``aerolume-cli doctor``, the About dialog and some libraries ask for versions this way; a
    build without the ``.dist-info`` folders reports VTK as "not installed" although it works.
    """
    from importlib import metadata
    out, missing = [], []
    for dist in ("torch", "vtk", "numpy", "scipy", "PySide6_Essentials", "pyqtgraph", "matplotlib"):
        try:
            out.append(f"{dist} {metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            missing.append(dist)
    if missing:
        raise RuntimeError("no package metadata for " + ", ".join(missing))
    # Aerolume's own metadata is generated from pyproject.toml at build time; in a frozen app it
    # must name the same version as the code, or "Help > Project page" and report provenance
    # describe a different release. An editable install may lag (it is written at install
    # time), so outside a build a mismatch is only reported.
    from aerolume import __version__
    try:
        own = metadata.version("aerolume")
    except metadata.PackageNotFoundError:
        own = None
    if own != __version__:
        if is_frozen():
            raise RuntimeError(f"bundled aerolume metadata says {own or 'nothing'}, the code is {__version__}")
        out.append(f"aerolume metadata {own or 'missing'} (code {__version__}; reinstall to refresh)")
    else:
        out.insert(0, f"aerolume {own}")
    return ", ".join(out)


def _check_torch() -> str:
    import torch
    detail = f"torch {torch.__version__}, CUDA build {torch.version.cuda or 'none'}"
    x = torch.linspace(0, 1, 4096, dtype=torch.float64)
    if abs(float(torch.fft.rfft(x).real[0]) - float(x.sum())) > 1e-9:
        raise RuntimeError("CPU FFT check failed")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        a = torch.randn(256, 256, device=dev, generator=torch.Generator(dev).manual_seed(0))
        b = a @ a.T                                   # cuBLAS
        f = torch.fft.rfft2(a)                        # cuFFT
        e = torch.special.erf(a).sum()                # a jiterator / NVRTC-backed kernel family
        ref = a.double().cpu() @ a.double().cpu().T
        err = float((b.double().cpu() - ref).abs().max() / ref.abs().max())
        torch.cuda.synchronize()
        if not (err < 1e-3 and torch.isfinite(f.abs().sum()) and torch.isfinite(e)):
            raise RuntimeError(f"CUDA arithmetic check failed (relative matmul error {err:.2e})")
        detail += f"; CUDA device {torch.cuda.get_device_name(0)}, matmul rel. error {err:.1e}"
    else:
        detail += "; no CUDA device (CPU only)"
    return detail


def _check_ns2d() -> str:
    import numpy as np
    import torch

    from ..solvers.ns2d import FlowSolver, ImmersedBody, SolverConfig
    th = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    circle = np.stack([0.5 * np.cos(th), 0.5 * np.sin(th)], axis=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SolverConfig(nx=96, ny=48, h=1 / 12, x0=-2.0, y0=-2.0, re=100.0, device=device, dtype="float32",
                       bodies=[ImmersedBody(circle, name="cyl")], moment_ref=(0.0, 0.0))
    s = FlowSolver(cfg)
    s.advance(20)
    div = float(s.field("divergence").abs().max()) * s.ops.h
    speed = s.field("speed")
    if not bool(torch.isfinite(speed).all()) or div > 1e-4:
        raise RuntimeError(f"2D solver check failed (max |div| h = {div:.2e})")
    cd = float(s.coefficients()[0])
    return f"20 steps on {device}, max |div|·h = {div:.1e}, Cd(t) = {cd:.3f}"


def _octahedron(center: tuple[float, float, float], r: float):
    """A closed 8-face mesh: the smallest watertight body the 3D immersed boundary accepts."""
    import numpy as np
    v = np.array([(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)], float) + np.asarray(center)
    f = np.array([(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)],
                 np.int64)
    return v, f


def _check_ns3d() -> str:
    """Ten steps of the 3D solver with every placed-physics path the GPU build compiles.

    A heated body (energy equation + Boussinesq buoyancy), an actuator-disc fan, a smoke
    emitter and a particle emitter. On CUDA the solver compiles its fused kernels with NVRTC
    (``torch.cuda.jiterator``) and captures CUDA graphs; when that fails it silently falls back
    to the slower reference arithmetic and records why (``fused_error``, ``theta_error``,
    ``dye_error``, ``graph_error``). From source that fallback is a performance note; in a
    frozen build it means a CUDA runtime file (the NVRTC builtins, typically) was not bundled,
    so here it is a failure rather than something a user discovers as a 3x slower run.
    """
    import numpy as np
    import torch

    from ..solvers.ns3d import FlowSolver3D, ImmersedMesh, SolverConfig3D
    from ..solvers.ns3d.config import Emitter3D, Fan3D, ThermalConfig3D
    v, f = _octahedron((0.5, 0.0, 0.0), 0.35)
    body = ImmersedMesh(v, f, name="heated", thermal="temperature", thermal_value=1.0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SolverConfig3D(
        nx=48, ny=24, nz=24, h=1 / 8, x0=-1.5, y0=-1.5, z0=-1.5, re=100.0, device=device, dtype="float32",
        bodies=[body], perturb=False, dye=True, diagnostics=True,
        thermal=ThermalConfig3D(prandtl=0.71, richardson=0.5),
        fans=[Fan3D(center=(-0.8, 0.0, 0.0), diameter=1.0, value=0.1, name="fan")],
        emitters=[Emitter3D(species="dye", kind="point", position=(-1.2, 0.0, 0.2), name="smoke"),
                  Emitter3D(species="particles", kind="point", position=(-1.2, 0.0, -0.2), rate=200.0,
                            name="tracers")])
    s = FlowSolver3D(cfg)
    s.advance(10)
    if device == "cuda":
        torch.cuda.synchronize()
    speed = s.field("speed")
    theta = s.field("temperature")
    div = float(s.field("divergence").abs().max()) * cfg.h
    if not (bool(torch.isfinite(speed).all()) and bool(torch.isfinite(theta).all())) or div > 1e-3:
        raise RuntimeError(f"3D solver check failed (max |div| h = {div:.2e})")
    q_body = float(s.heat_balance().get("heat_in_bodies", float("nan")))
    if not q_body > 0.0:                       # a body held at theta = 1 must heat the fluid
        raise RuntimeError(f"heated body delivers {q_body} (expected > 0)")
    thrust = float(s.fan_report()[0]["thrust"])
    if not (np.isfinite(thrust) and abs(thrust - 0.1) < 1e-3):
        raise RuntimeError(f"fan delivered thrust {thrust} instead of the set 0.1")
    if s.particle_count() <= 0:
        raise RuntimeError("the particle emitter released no tracers")
    fallbacks = {k: getattr(s, k) for k in ("fused_error", "theta_error", "dye_error", "graph_error")
                 if getattr(s, k, None)}
    path = "reference kernels"
    if device == "cuda":
        path = ("fused NVRTC kernels" if s.uses_fused_kernels else "reference kernels") + (
            " + CUDA graphs" if s.uses_cuda_graphs and not s.graph_error else "")
        if fallbacks and is_frozen():
            raise RuntimeError("3D GPU fast path unavailable in this build: "
                               + "; ".join(f"{k}: {str(e)[:200]}" for k, e in fallbacks.items()))
    note = f"; fallbacks: {', '.join(fallbacks)}" if fallbacks else ""
    return (f"10 steps on {device} ({path}), heated body + fan + smoke + particles: max |div|·h = {div:.1e}, "
            f"fan thrust {thrust:.4f} (set 0.1), body heat in {q_body:.3g}, "
            f"{s.particle_count()} tracers{note}")


def _check_airfoil() -> str:
    from ..geometry.airfoils import naca4
    from ..solvers.airfoil import analyze
    p = analyze(naca4("0012"), 1e6, 2.0, panels=120)
    if not (p.converged and 0.1 < p.cl < 0.4):
        raise RuntimeError(f"NACA 0012 at alpha 2 deg, Re 1e6 did not converge sensibly (cl={p.cl})")
    return f"NACA 0012, Re 1e6, alpha 2 deg: Cl {p.cl:.4f}, Cd {p.cd:.5f}"


def _check_vtk() -> str:
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkVersion
    from vtkmodules.vtkFiltersSources import vtkSphereSource
    from vtkmodules.vtkIOImage import vtkPNGWriter  # noqa: F401 - report renders need it
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )
    from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow  # noqa: F401 - registers OpenGL
    src = vtkSphereSource()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    ren = vtkRenderer()
    ren.AddActor(actor)
    ren.SetBackground(0.0, 0.0, 0.0)
    win = vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)
    win.SetSize(64, 64)
    win.Render()
    grab = vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.Update()
    px = vtk_to_numpy(grab.GetOutput().GetPointData().GetScalars())
    lit = int((px.reshape(len(px), -1)[:, :3].sum(axis=1) > 0).sum())
    info = ""
    with contextlib.suppress(Exception):
        report = win.ReportCapabilities() or ""
        m = re.search(r"OpenGL renderer string:\s*(.+)", report)
        info = f", {m.group(1).strip()}" if m else ""
    win.Finalize()
    if lit < 100:
        raise RuntimeError(f"VTK off-screen render produced {lit} lit pixels of 4096")
    return f"VTK {vtkVersion.GetVTKVersion()}, off-screen render {lit}/4096 pixels lit{info}"


def _check_qt() -> str:
    from PySide6.QtGui import QIcon, QImageReader
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([sys.argv[0], "-platform", os.environ.get("QT_QPA_PLATFORM", "offscreen")])
    fmts = {bytes(f.data()).decode() for f in QImageReader.supportedImageFormats()}
    need = {"svg", "png", "jpg", "ico"}
    if not need <= fmts:
        raise RuntimeError(f"Qt image plugins missing: {sorted(need - fmts)}")
    icon = QIcon(str(resource_path("aerolume.svg")))
    if icon.pixmap(32, 32).isNull():
        raise RuntimeError("the SVG app icon does not render (iconengines plugin missing?)")
    return f"Qt platform {app.platformName()}, image formats ok ({len(fmts)})"


def _check_window() -> str:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    from PySide6.QtWidgets import QApplication

    from . import main as app_main
    t0 = time.perf_counter()
    _app, win = app_main.build(["aerolume"])
    win.resize(1280, 800)
    win.show()
    for _ in range(20):
        QApplication.processEvents()
    t_build = time.perf_counter() - t0
    title = win.windowTitle()
    grab = win.grab()
    if grab.isNull() or grab.width() < 100:
        raise RuntimeError("the main window did not paint")
    fonts = _font_status(QApplication.instance())
    win.close()
    QApplication.processEvents()
    return (f"main window built and painted in {t_build:.1f} s ({title!r}, {grab.width()}x{grab.height()}); "
            f"{fonts}")


def _font_status(app) -> str:
    """Check that the interface font really is the bundled Atkinson face, not just that its files shipped.

    ``_check_resources`` only counts the font files. A build that ships them but
    fails to register them with QFontDatabase (a missing Qt font plugin, a
    truncated file) would otherwise pass the self-test and quietly draw the
    whole interface in Segoe UI. ``QFontInfo`` reports the family Qt actually
    resolved for the application font, so a substitution shows up here. With
    ``font="system"`` chosen in the preferences, the platform face is correct.
    """
    from PySide6.QtGui import QFontInfo

    from . import theme
    registered = theme.bundled_fonts_available()
    resolved = QFontInfo(app.font()).family()
    choice = theme.font_choice()
    detail = f"font {resolved!r} (bundled faces registered: {registered}, choice: {choice})"
    if choice == "bundled" and resolved != theme.BUNDLED_UI_FAMILY:
        raise RuntimeError(f"the interface font is not {theme.BUNDLED_UI_FAMILY!r}: {detail}")
    return detail


def self_test(*, window: bool = True, gpu: bool = True) -> dict:
    """Check that this installation works; returns a JSON-able dict with ``ok`` and ``checks``.

    The checks cover what a broken build typically loses: data files (icons, examples, docs,
    licences), standard-library and SciPy modules for user scripts, torch with its CUDA
    libraries (a real cuBLAS / cuFFT / NVRTC call), the 2D solver and the airfoil solver, an
    OpenGL VTK frame, the Qt image plugins and, with ``window=True``, the full main window.
    Nothing is written except what the app itself writes to its config folder.
    """
    t0 = time.perf_counter()
    checks: list[_Check] = []
    _run_check("resources", _check_resources, checks)
    _run_check("app files", _check_app_files, checks)
    if is_frozen():
        _run_check("build info", _check_build_info, checks)
    _run_check("stdlib for scripts", lambda: _check_imports(SCRIPT_STDLIB), checks)
    _run_check("scientific stack for scripts", lambda: _check_imports(SCRIPT_THIRD_PARTY), checks)
    _run_check("aerolume API", lambda: _check_imports(("aerolume.api", "aerolume.cli", "aerolume.report",
                                                        "aerolume.export", "aerolume.ai",
                                                        "aerolume.accounts")), checks)
    _run_check("package metadata", _check_metadata, checks)
    _run_check("qt", _check_qt, checks)
    if gpu:
        _run_check("torch", _check_torch, checks)
        _run_check("2D solver", _check_ns2d, checks)
        _run_check("3D solver", _check_ns3d, checks)
    _run_check("airfoil solver", _check_airfoil, checks)
    _run_check("vtk", _check_vtk, checks)
    if window:
        _run_check("main window", _check_window, checks)
    return {"ok": all(c.ok for c in checks), "version": __version__, "frozen": is_frozen(),
            "executable": redact(sys.executable), "app_root": redact(str(app_root())),
            "seconds": round(time.perf_counter() - t0, 2),
            "checks": [c.__dict__ for c in checks]}


def run_self_test_cli(argv: Sequence[str]) -> int:
    """``--selftest [--selftest-out FILE] [--no-window] [--no-gpu]``: run, write JSON, exit code 0/1.

    The windowed build has no console, so the JSON goes to ``FILE`` (default
    ``logs/selftest.json``) as well as to stdout when one exists.
    """
    args = list(argv)
    out_path: Path | None = None
    if "--selftest-out" in args:
        i = args.index("--selftest-out")
        if i + 1 < len(args):
            out_path = Path(args[i + 1])
    result = self_test(window="--no-window" not in args, gpu="--no-gpu" not in args)
    text = json.dumps(result, indent=2)
    if out_path is None:
        out_path = logs_dir() / "selftest.json"
    with contextlib.suppress(OSError):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        with contextlib.suppress(Exception):
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
    return 0 if result["ok"] else 1


__all__ = [
    "APP_MUTEX_NAME",
    "APP_USER_MODEL_ID",
    "MAX_CRASH_LOGS",
    "PROJECT_INDEX",
    "PROJECT_MARKER",
    "CrashReport",
    "SingleInstance",
    "activate_window",
    "app_file",
    "app_root",
    "build_info",
    "bundle_dir",
    "config_dir",
    "current_app_user_model_id",
    "docs_dir",
    "environment_lines",
    "examples_dir",
    "format_crash_report",
    "hold_app_mutex",
    "install_crash_handler",
    "is_frozen",
    "logs_dir",
    "mutex_exists",
    "package_dir",
    "prepare_runtime",
    "previous_native_crash",
    "redact",
    "resolve_open_target",
    "resource_path",
    "run_self_test_cli",
    "self_test",
    "set_app_user_model_id",
    "show_fatal_error",
    "version_label",
    "write_crash_report",
]
