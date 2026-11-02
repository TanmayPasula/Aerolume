"""The start-up splash: what Aerolume is loading, while PyTorch and VTK take their seconds.

Importing PyTorch (with CUDA) and VTK takes 3-8 s on a cold start, during
which a desktop program that shows nothing looks broken and gets launched
twice. The splash appears within a few hundred milliseconds of the
QApplication, states what is being loaded right now ("Loading PyTorch and
CUDA…") with a thin determinate bar, and closes when the window is up.

It is a plain panel in the theme's surface colour with the app mark, the
name and version and one status line - no artwork, no animation beyond the
bar. :func:`preload` runs the loading steps on the GUI thread (the only
thread allowed to create the OpenGL-backed VTK classes) and processes events
between steps so the text repaints; a step that fails is recorded and
skipped, never fatal - the window reports the problem where it matters
(status bar device label, the 3D views).
"""
from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from ... import __version__
from .. import theme
from .common import SPACING, label


@dataclass
class PreloadStep:
    """One thing the splash loads: the line it shows, and the work."""

    message: str
    run: Callable[[], object]
    #: what is recorded instead of failing (filled by :func:`preload`)
    error: str = ""
    seconds: float = 0.0


def _import(*names: str) -> Callable[[], None]:
    def go() -> None:
        for n in names:
            importlib.import_module(n)
    return go


#: the default start-up work: the two slow imports, in the order the window needs them
DEFAULT_STEPS: tuple[tuple[str, Callable[[], object]], ...] = (
    ("Loading NumPy and SciPy…", _import("numpy", "scipy.sparse")),
    ("Loading PyTorch…", _import("torch")),
    ("Loading VTK for the 3D views…", _import("vtkmodules.vtkRenderingCore", "vtkmodules.vtkRenderingOpenGL2",
                                             "vtkmodules.vtkFiltersCore")),
    ("Loading the plotting library…", _import("pyqtgraph")),
)


class Splash(QWidget):
    """A frameless 520 x 300 panel: mark, name, version, status line and a thin progress bar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setFixedSize(520, 300)
        self.setAccessibleName("Aerolume is starting")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setProperty("splash", "true")
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(8 * SPACING, 8 * SPACING, 8 * SPACING, 6 * SPACING)
        lay.setSpacing(0)
        head = QHBoxLayout()
        head.setSpacing(4 * SPACING)
        self.mark = QLabel()
        self.mark.setPixmap(theme.app_icon_pixmap(48, self.devicePixelRatioF()))
        self.mark.setFixedSize(48, 48)
        head.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        names = QVBoxLayout()
        names.setSpacing(SPACING)
        self.title = label("Aerolume", "title")
        names.addWidget(self.title)
        self.subtitle = label("Aerodynamics and heat transfer in 2D and 3D", "secondary")
        names.addWidget(self.subtitle)
        head.addLayout(names, 1)
        lay.addLayout(head)
        lay.addStretch(1)
        self.message = label("Starting…", None)
        self.message.setAccessibleName("Start-up progress")
        lay.addWidget(self.message)
        lay.addSpacing(2 * SPACING)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        lay.addWidget(self.bar)
        lay.addSpacing(4 * SPACING)
        self.footer = label(f"Version {__version__}. Open source under the Apache License 2.0.", "small")
        lay.addWidget(self.footer)
        self.timings: list[tuple[str, float, str]] = []

    def show_message(self, text: str, fraction: float | None = None) -> None:
        """Say what is loading now (and how far through the steps), and repaint at once."""
        self.message.setText(text)
        if fraction is not None:
            self.bar.setValue(round(1000 * max(0.0, min(1.0, fraction))))
        if self.isVisible():
            self.repaint()
            QCoreApplication.processEvents()

    def finish(self, window: QWidget | None = None) -> None:
        """Close once ``window`` is up (it may be None: close now)."""
        self.bar.setValue(1000)
        self.close()

    def centre_on_screen(self) -> None:
        screen = self.screen()
        if screen is not None:
            g = screen.availableGeometry()
            self.move(g.center().x() - self.width() // 2, g.center().y() - self.height() // 2)


def preload(splash: Splash | None, steps: Sequence[tuple[str, Callable[[], object]]] = DEFAULT_STEPS,
            *, clock: Callable[[], float] = time.perf_counter) -> list[PreloadStep]:
    """Run the start-up steps, telling the splash what each one is; returns what happened.

    Each step is timed and a failure is recorded on its :class:`PreloadStep`
    (``error``), so a missing VTK still starts the program - the 3D views
    then say "3D view unavailable" with the reason.
    """
    done: list[PreloadStep] = []
    n = max(len(steps), 1)
    for i, (message, run) in enumerate(steps):
        step = PreloadStep(message, run)
        if splash is not None:
            splash.show_message(message, i / n)
        t0 = clock()
        try:
            run()
        except Exception as exc:  # noqa: BLE001 - the window starts without the module
            step.error = f"{type(exc).__name__}: {exc}"
        step.seconds = clock() - t0
        done.append(step)
        if splash is not None:
            splash.timings.append((message, step.seconds, step.error))
    if splash is not None:
        splash.show_message("Opening the window…", 1.0)
    return done


__all__ = ["DEFAULT_STEPS", "PreloadStep", "Splash", "preload"]
