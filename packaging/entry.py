"""Entry script of the frozen build: one program, two executables, a self-test.

PyInstaller turns this file into both ``Aerolume.exe`` (windowed) and ``aerolume-cli.exe``
(console). The two share one analysed module set and one ``_internal`` folder; this script
picks the role from the executable's own name, so the 2.9 GB of torch/CUDA libraries ship once.

* ``aerolume-cli.exe ...`` -> :func:`aerolume.cli.main` (headless runs, validation, scripts).
* ``Aerolume.exe --selftest [--selftest-out FILE] [--no-window] [--no-gpu]`` ->
  :func:`aerolume.app.platform.run_self_test_cli`; used by ``packaging/smoke_test.py`` and by
  support ("run the self-test and send me the JSON").
* ``Aerolume.exe [PROJECT]`` -> :func:`aerolume.app.main.main`.

:func:`aerolume.app.platform.prepare_runtime` runs first in the windowed roles: a windowed
executable has no stdout/stderr, and without it a library warning or a native crash during
the torch/VTK import would leave no trace at all.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _role() -> str:
    name = Path(sys.executable if getattr(sys, "frozen", False) else sys.argv[0]).stem.lower()
    if name.startswith("aerolume-cli"):
        return "cli"
    if "--selftest" in sys.argv[1:]:
        return "selftest"
    return "gui"


def _run() -> int:
    if getattr(sys, "frozen", False):
        # The install folder is read-only for all-users installs and owned by the installer for
        # per-user ones: user scripts and the validation scripts must not drop __pycache__ there
        # (the uninstaller would leave it behind).
        sys.dont_write_bytecode = True
    role = _role()
    if role == "cli":
        from aerolume.cli import main as cli_main
        return int(cli_main() or 0)

    from aerolume.app import platform as plat
    plat.prepare_runtime()
    if role == "selftest":
        os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
        return plat.run_self_test_cli(sys.argv[1:])
    try:
        from aerolume.app.main import main as gui_main
    except Exception:  # noqa: BLE001 - an import failure in a windowed build must be shown
        report = plat.write_crash_report(*sys.exc_info(), context="While starting Aerolume (import).")
        plat.show_fatal_error("Aerolume could not start", report.text)
        return 1
    try:
        return int(gui_main() or 0)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - shown in a dialog, the windowed build has no console
        report = plat.write_crash_report(*sys.exc_info(), context="While starting Aerolume.")
        plat.show_fatal_error("Aerolume could not start", report.text)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()     # harmless; keeps any future worker process from re-launching the GUI
    sys.exit(_run())
