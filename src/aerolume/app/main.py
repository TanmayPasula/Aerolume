"""Desktop entry point: ``aerolume [PROJECT] [--theme dark|light|system] [--no-splash]``.

Order matters here, and each step says why:

1. :func:`aerolume.app.platform.prepare_runtime` before anything can print (a
   windowed build has no console; output goes to the log file instead).
2. The taskbar identity and the high-DPI policy before the QApplication
   exists (Windows otherwise groups the window under python.exe).
3. **One instance**: a second launch hands its arguments (the project a user
   double-clicked) to the running window and exits with 0, instead of a
   second 2 GB torch/VTK process competing for the GPU
   (``platform.SingleInstance``; a fake can be injected for tests).
4. The theme before the first widget is built, so nothing flashes in the
   default palette.
5. The splash while PyTorch and VTK load (``widgets/splash.py``), then the
   window; uncaught errors from then on open a copyable report
   (``platform.install_crash_handler`` -> ``MainWindow.show_crash``).
6. The project argument once the window exists, so errors can be shown in
   it; the first-run tour once it is on screen.

:func:`build` makes (app, window) without the event loop, single instance or
splash - what the tests use.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from .. import __version__

APP_USER_MODEL_ID = "Aerolume.Aerolume.Desktop"


def _platform():
    """``aerolume.app.platform`` (desktop integration), or None when this build lacks it."""
    try:
        from . import platform as plat
        return plat
    except Exception:  # noqa: BLE001 - every feature it adds is optional
        return None


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="aerolume",
                                 description="Aerolume - open-source 2D and 3D aerodynamics studio")
    ap.add_argument("project", nargs="?", help="project folder (*.aerolume) to open")
    ap.add_argument("--theme", choices=("dark", "light", "system"), help="override the saved theme for this session")
    ap.add_argument("--no-splash", action="store_true", help="start without the splash screen")
    ap.add_argument("--new-instance", action="store_true",
                    help="start a second window instead of handing the project to the running one")
    ap.add_argument("--version", action="version", version=f"Aerolume {__version__}")
    # Qt consumes its own options (-platform, -style, ...); ignore anything else we do not know
    args, _unknown = ap.parse_known_args(list(argv))
    return args


def _windows_taskbar_identity() -> None:
    """Give the process its own taskbar identity so Windows shows our icon, not Python's."""
    plat = _platform()
    if plat is not None:
        plat.set_app_user_model_id(APP_USER_MODEL_ID)
        return
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def create_app(argv: Sequence[str] | None = None) -> QApplication:
    """The QApplication with high-DPI policy, names and icon set (reuses an existing one)."""
    app = QApplication.instance()
    if app is not None:
        return app  # type: ignore[return-value]
    # Qt 6 always scales for high DPI; PassThrough keeps 125 % / 150 % displays at their true
    # factor instead of rounding to 100 % / 200 % (blurry or oversized UI).
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    _windows_taskbar_identity()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("Aerolume")
    app.setApplicationDisplayName("Aerolume")
    app.setOrganizationName("Aerolume")
    app.setApplicationVersion(__version__)
    app.setDesktopFileName("aerolume")
    from . import theme
    app.setWindowIcon(theme.app_icon())
    return app


_hook_window: list = []          # the window that receives uncaught exceptions (latest build)


def _route_uncaught_exceptions(win) -> None:
    """Show exceptions raised in Qt slots in the Messages panel instead of losing them.

    PySide prints an exception escaping a slot to stderr and carries on, which a
    user launching from a shortcut never sees. The original hook still runs, so
    the traceback also reaches the console. Installing twice (tests build
    several windows) only retargets the window; hooks never stack. The crash
    report (:func:`_install_crash_reports`) chains after this hook.
    """
    _hook_window[:] = [win]
    if getattr(sys.excepthook, "_aerolume", False) or getattr(sys.excepthook, "_aerolume_crash", False):
        return
    original = sys.excepthook

    def hook(exc_type, exc, tb) -> None:
        original(exc_type, exc, tb)
        if issubclass(exc_type, KeyboardInterrupt) or not _hook_window:
            return
        with contextlib.suppress(Exception):     # the window may already be gone
            _hook_window[0].state.log(f"{exc_type.__name__}: {exc}", "error", "Application")

    hook._aerolume = True
    sys.excepthook = hook


def _install_crash_reports(win) -> bool:
    """Save every uncaught error as a report and show it in the window (app/platform.py)."""
    plat = _platform()
    if plat is None or not hasattr(plat, "install_crash_handler"):
        return False

    def show(report) -> None:
        target = _hook_window[0] if _hook_window else win
        target.show_crash(report)

    plat.install_crash_handler(show)
    return True


def _settings_and_theme(app: QApplication, args: argparse.Namespace) -> None:
    from . import settings, theme
    s = settings.get()
    theme.apply(app, args.theme or s.theme, font=getattr(s, "ui_font", "bundled"))


def build(argv: Sequence[str] | None = None):
    """Create (app, window) without entering the event loop - used by :func:`main` and tests."""
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")   # never let pyqtgraph pick another binding
    argv = list(sys.argv if argv is None else argv)
    args = _parse_args(argv[1:])
    app = create_app(argv)
    _settings_and_theme(app, args)
    from .main_window import MainWindow
    win = MainWindow()
    _route_uncaught_exceptions(win)
    if args.project:
        win.open_project_path(_open_target(args.project), confirmed=True)
    return app, win


def _open_target(arg: str) -> Path:
    """The project folder a command-line argument means (a ``project.aerolume`` marker, a folder...)."""
    plat = _platform()
    if plat is not None and hasattr(plat, "resolve_open_target"):
        with contextlib.suppress(Exception):
            p = plat.resolve_open_target(arg)
            if p is not None:
                return Path(p)
    return Path(arg).expanduser()


def main(argv: Sequence[str] | None = None, *, instance=None) -> int:
    """Run the desktop application; returns the exit code.

    ``instance`` replaces ``platform.SingleInstance()`` (tests pass a fake with
    ``start(argv) -> bool`` and a ``message_received`` signal).
    """
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    argv = list(sys.argv if argv is None else argv)
    plat = _platform()
    if plat is not None:
        with contextlib.suppress(Exception):
            plat.prepare_runtime()
    args = _parse_args(argv[1:])
    app = create_app(argv)

    guard = instance
    if guard is None and plat is not None and not args.new_instance:
        guard = plat.SingleInstance()
    if guard is not None and not guard.start(argv[1:]):
        return 0                            # handed to the running window
    if plat is not None:
        with contextlib.suppress(Exception):
            plat.hold_app_mutex()

    _settings_and_theme(app, args)
    _use_calibration()
    splash = None
    if not args.no_splash and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        from .widgets.splash import Splash, preload
        splash = Splash()
        splash.centre_on_screen()
        splash.show()
        app.processEvents()
        preload(splash)
    from .main_window import MainWindow
    win = MainWindow()
    _route_uncaught_exceptions(win)
    _install_crash_reports(win)
    if guard is not None and hasattr(guard, "message_received"):
        guard.message_received.connect(win.handle_forwarded_argv)
    win.show()
    if splash is not None:
        splash.finish(win)
    if args.project:
        win.open_project_path(_open_target(args.project), confirmed=True)
    _report_previous_native_crash(win)
    win.maybe_show_first_run_tour()
    return app.exec()


def _use_calibration() -> None:
    """Load and keep the run-time calibration in the user config folder for this session.

    Each run's measured speed refines the planner's next estimate; without the file every app
    session started from the reference table, and only the CLI kept what it learned. The CLI
    reads and writes the same file (``cli._calibration``), so both share their measurements.
    ``AEROLUME_CALIBRATION=off`` disables it, as for the CLI. Advisory only: a failure is ignored.
    """
    if os.environ.get("AEROLUME_CALIBRATION", "").lower() in ("0", "off", "no"):
        return
    with contextlib.suppress(Exception):
        from ..project import config_dir
        from ..simulation.plan import CALIBRATION_FILE, use_calibration_file
        use_calibration_file(config_dir() / CALIBRATION_FILE)


def _report_previous_native_crash(win) -> None:
    """If the last session died in native code (a driver fault), say so once in Messages."""
    plat = _platform()
    text = None
    with contextlib.suppress(Exception):
        text = plat.previous_native_crash() if plat is not None else None
    if text:
        first = text.strip().splitlines()[0][:160]
        win.state.log(f"The previous session ended unexpectedly in native code ({first}). The trace is in "
                      "Help ▸ Open log folder.", "warning", "Application")


if __name__ == "__main__":
    sys.exit(main())
