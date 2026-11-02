"""Theme-switch cost of the full window (UI critique round 3: dark mode froze the app for seconds).

``theme.apply()`` used to let Qt replace the application style sheet in place,
which re-polishes each widget once per ancestor: 3.5 s with six results open.
It now swaps the sheet in two linear passes. These tests pin both the shape of
the fix (StyleChange events per widget, independent of the machine) and the
budget the critique set (< 0.8 s offscreen with every page built).
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


@pytest.fixture()
def isolated(tmp_path, monkeypatch, qapp):
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    projects = tmp_path / "projects"
    projects.mkdir()
    yield projects
    theme.apply(qapp, "light")
    settings.use_file(tmp_path / "settings-after.ini")


class _StyleEvents(QObject):
    """Counts StyleChange events delivered to widgets (one per re-polish)."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def eventFilter(self, _obj, e) -> bool:
        if e.type() == QEvent.Type.StyleChange:
            self.n += 1
        return False


def _pump(app, n: int = 5) -> None:
    for _ in range(n):
        app.processEvents()


def test_theme_switch_is_linear_and_quick_with_every_page_built(isolated, qapp):
    from PySide6.QtWidgets import QApplication

    from aerolume.app import theme
    from aerolume.app.main_window import STEP_KEYS, MainWindow
    w = MainWindow(probe_device=False, load_jobs=False)
    try:
        w.resize(1600, 1000)
        w.show()
        w.new_study("vehicle", "Theme cost", isolated)
        _pump(qapp, 4)
        for key in STEP_KEYS:                    # build every page, as a user who visited them all
            w.show_page(key)
            _pump(qapp, 3)
        widgets = len(QApplication.allWidgets())
        counter = _StyleEvents()
        qapp.installEventFilter(counter)
        try:
            t0 = time.perf_counter()
            theme.apply(qapp, "dark")
            qapp.processEvents()
            dark_s = time.perf_counter() - t0
        finally:
            qapp.removeEventFilter(counter)
        t0 = time.perf_counter()
        theme.apply(qapp, "light")
        qapp.processEvents()
        light_s = time.perf_counter() - t0
        # Qt's in-place replacement sent ~11 StyleChange events per widget (one per ancestor);
        # the two-pass swap sends about two (one per pass) - this does not depend on the machine
        assert counter.n <= 3 * widgets, (counter.n, widgets)
        assert max(dark_s, light_s) < 0.8, (dark_s, light_s, widgets)
        assert theme.current() == "light" and qapp.styleSheet() == theme.stylesheet("light")
    finally:
        w._quit_confirmed = True
        w.close()
        w.deleteLater()
        _pump(qapp, 2)


def test_reapplying_the_same_theme_does_not_restyle(isolated, qapp):
    from aerolume.app import theme
    theme.apply(qapp, "light")
    counter = _StyleEvents()
    qapp.installEventFilter(counter)
    try:
        theme.apply(qapp, "light")              # same sheet: font and palette only, no re-polish
    finally:
        qapp.removeEventFilter(counter)
    assert counter.n == 0
    assert qapp.property("aerolumeFusionBase") is True
    assert qapp.style().objectName().lower() in ("fusion", "") and qapp.styleSheet()


def test_system_theme_follows_the_operating_system(isolated, qapp, monkeypatch):
    from aerolume.app import settings, theme
    monkeypatch.setattr(theme, "system_theme", lambda app=None: "dark")
    theme.apply(qapp, "system")
    assert theme.current() == "dark" and theme.preference() == "system"
    assert theme.resolve("system") == "dark" and theme.resolve("light") == "light"
    monkeypatch.setattr(theme, "system_theme", lambda app=None: "light")
    theme._on_system_scheme()                    # what the OS colour-scheme signal calls
    assert theme.current() == "light" and theme.preference() == "system"
    theme.apply(qapp, "dark")                    # an explicit choice stops following
    assert theme.preference() == "dark"
    theme._on_system_scheme()
    assert theme.current() == "dark"
    s = settings.get()
    s.theme = "system"
    assert s.theme == "system"
    with pytest.raises(ValueError):
        s.theme = "sepia"
