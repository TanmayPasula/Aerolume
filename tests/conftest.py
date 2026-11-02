"""Suite-wide test hygiene.

GUI tests build pages and windows and mostly just ``close()`` them, which only
hides a widget. Without the fixture below they accumulate for the whole
session (the Qt application is shared), and every later ``theme.apply()`` has
to restyle all of them: in the full suite a single theme switch in
``test_app_shell.py`` took minutes behind the pages left over by
``test_app_runtime.py`` and ``test_app_geometry_setup.py``. The application
itself does not have this problem (it has one window), so the fix belongs in
the test harness.
"""
from __future__ import annotations

import gc
import sys

import pytest


def _delete_top_level_widgets() -> None:
    """Close and delete the application's own top-level widgets, then flush.

    Only parentless widgets whose class comes from ``aerolume`` (pages, the main
    window, dialogs) are deleted. Libraries also keep parentless widgets that a
    Python object owns - pyqtgraph's plot context menus, for instance - and
    deleting those from C++ before their owner makes Qt abort when the owner is
    destroyed later; they go away with their owner instead (``gc.collect``).
    Qt is only touched when a test already imported it, so non-GUI tests never
    pay for (or depend on) PySide6.
    """
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    owned = []
    for w in app.topLevelWidgets():
        try:
            if w.parent() is None and type(w).__module__.startswith("aerolume"):
                owned.append(w)
        except RuntimeError:        # already deleted on the C++ side
            pass
    for w in owned:
        try:
            if hasattr(w, "_quit_confirmed"):
                w._quit_confirmed = True         # the main window would ask about running jobs
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    if owned:
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    gc.collect()
    app.processEvents()


def pytest_configure(config) -> None:
    """Point every config store at a throw-away folder for the whole run.

    The preferences INI (app/settings.py) and the recent-projects list
    (project/recent.py) both honour ``AEROLUME_CONFIG_DIR``. Without this, a
    test that opens a project through the real window adds it to the user's
    recent list, and a theme test leaves the user's app in the other theme.
    A value already set by the caller (CI, a developer) is respected.
    """
    import os
    import tempfile

    if not os.environ.get("AEROLUME_CONFIG_DIR"):
        os.environ["AEROLUME_CONFIG_DIR"] = tempfile.mkdtemp(prefix="aerolume-test-config-")
    _probe_vtk_out_of_process()


def _probe_vtk_out_of_process() -> None:
    """Decide once, in a child process, whether VTK can render off-screen here.

    ``viz3d.scene.backend_info()`` opens a real OpenGL context. On machines without
    a usable driver (the GitHub Actions Windows and Linux runners) that call does not
    fail politely: the driver crashes the process with an access violation or an
    abort, and the whole test run dies at collection. The child takes that risk
    instead. If it crashes, hangs or reports no context, the parent's probe cache is
    filled with "not ok", so every 3D test skips with its usual reason and nothing in
    this process touches OpenGL.
    """
    import subprocess

    code = ("from aerolume.viz3d.scene import backend_info; "
            "print('VTKOK' if backend_info().get('ok') else 'VTKNO')")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                             timeout=120).stdout
        ok, why = "VTKOK" in out, "no off-screen OpenGL context (checked in a child process)"
    except subprocess.TimeoutExpired:
        ok, why = False, "the off-screen OpenGL probe hung (checked in a child process)"
    if ok:
        return
    try:
        from aerolume.viz3d import scene
    except Exception:  # noqa: BLE001 - no VTK at all: the 3D tests skip on import already
        return
    scene._BACKEND = {"ok": False, "window": "", "renderer": "", "version": "", "error": why}


@pytest.fixture(autouse=True)
def _release_qt_widgets():
    """Delete what a test left on screen once it (and its own fixtures) finished."""
    yield
    _delete_top_level_widgets()


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Make a modal dialog that a test did not expect fail the test instead of hanging the suite.

    On the offscreen platform nobody can answer ``QMessageBox.question`` or a
    dialog's ``exec()``: the whole run used to stop there until killed. Tests
    that expect a dialog patch it themselves (their monkeypatch runs after this
    fixture and wins).
    """
    if "PySide6.QtWidgets" not in sys.modules:
        yield
        return
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def refuse(name: str):
        def modal(*_a, **_k):
            raise AssertionError(f"a test opened a modal dialog ({name}) without patching it")
        return modal

    for name in ("question", "warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(refuse(f"QMessageBox.{name}")))
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(refuse(f"QFileDialog.{name}")))
    monkeypatch.setattr(QDialog, "exec", refuse("QDialog.exec"))
    monkeypatch.setattr(QMessageBox, "exec", refuse("QMessageBox.exec"))
    yield
