"""Fixes from the first critique of the v4 restyle ("a little more like the website").

* rich-text links take the new theme's colour after a switch (the Assistant's
  "Script panel" link stayed near-black on the dark panel, about 1.1:1);
* a checked tool button or segment shows its state at >= 3:1 (WCAG 1.4.11), not
  only by a soft fill measured at 1.07-1.46:1;
* the Start page's study tiles paint their focus ring only for keyboard focus
  (the website's ``:focus-visible``), not for the window's initial focus;
* the 3D scene's floor, grid and body are neutral greys, and no widget keeps a
  v3 blue-grey colour as a fallback;
* the Setup canvas caption ends in an ellipsis instead of being cut mid-word.

Offscreen, private settings, as the other shell tests.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.gui

SRC = Path(__file__).resolve().parents[1] / "src" / "aerolume"


@pytest.fixture(scope="session")
def qapp():
    from aerolume.app.main import create_app
    return create_app(["aerolume-tests"])


_WINDOWS: list = []


@pytest.fixture()
def isolated(tmp_path, monkeypatch, qapp):
    from aerolume.app import settings, theme
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    theme.apply(qapp, "light")
    yield tmp_path
    while _WINDOWS:
        w = _WINDOWS.pop()
        try:
            w._quit_confirmed = True
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    from PySide6.QtCore import QEvent
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    theme.apply(qapp, "light")
    settings.use_file(tmp_path / "settings-after.ini")


def _pump(app, n: int = 5) -> None:
    for _ in range(n):
        app.processEvents()


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lum(rgb) -> float:
    r, g, b = (_lin(v / 255) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a, b) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _pixels(widget):
    """(h, w, 3) RGB array of ``widget`` as drawn in its window.

    The window is grabbed and cropped, because ``widget.grab()`` of a child without its own
    background comes back on black, not on the panel it is drawn over."""
    import numpy as np
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QImage
    top = widget.window()
    rect = QRect(widget.mapTo(top, QPoint(0, 0)), widget.size())
    img = top.grab(rect).toImage().convertToFormat(QImage.Format.Format_RGB32)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
    px = a[:, :img.width() * 4].reshape(img.height(), img.width(), 4)
    return px[..., [2, 1, 0]].copy()                 # B, G, R, A -> R, G, B


# ---------------------------------------------------------------------------
# rich-text links follow a theme switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("first,second", [("light", "dark"), ("dark", "light")])
def test_rich_text_links_take_the_new_theme_colour(isolated, qapp, first, second):
    """A link laid out in one theme must not keep that theme's ink after a switch.

    The probe is a link-only label on a panel (every pixel that is not background is link
    text), then the Assistant's own "Script panel" hint, whose link drew #0e0e0e on #111111.
    """
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    from aerolume.app import theme
    from aerolume.app.state import AppState
    from aerolume.app.widgets.assistant_panel import AssistantPanel

    theme.apply(qapp, first)
    host = QWidget()
    host.setAutoFillBackground(True)
    lay = QVBoxLayout(host)
    probe = QLabel('<a href="x">Script panel</a>')
    probe.setTextFormat(Qt.TextFormat.RichText)
    lay.addWidget(probe)
    panel = AssistantPanel(AppState())
    lay.addWidget(panel)
    host.resize(430, 700)
    host.show()
    _pump(qapp)
    theme.apply(qapp, second)
    _pump(qapp)
    bg = theme.qcolor("bg").getRgb()[:3]
    px = _pixels(probe).reshape(-1, 3)
    back = np.bincount((px[:, 0].astype(int) << 16) | (px[:, 1].astype(int) << 8) | px[:, 2]).argmax()
    back_rgb = ((back >> 16) & 255, (back >> 8) & 255, back & 255)
    ink = max((tuple(int(v) for v in p) for p in px), key=lambda p: _ratio(p, back_rgb))
    assert _ratio(ink, back_rgb) >= 4.5, (first, second, ink, back_rgb, bg)
    # the hint's link: on the dark panel nothing may be darker than the panel, on the light one
    # nothing lighter (the stale link was the old theme's ink, on the far side of the background)
    hint = _pixels(panel.script_hint).reshape(-1, 3).astype(int)
    lum = np.array([_lum(p) for p in hint[:: max(1, len(hint) // 4000)]])
    hint_bg = _lum(np.median(hint, axis=0))
    if second == "dark":
        assert lum.min() >= hint_bg - 0.01, (lum.min(), hint_bg)
    else:
        assert lum.max() <= hint_bg + 0.01, (lum.max(), hint_bg)
    host.close()
    host.deleteLater()


# ---------------------------------------------------------------------------
# checked state at 3:1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["light", "dark"])
def test_checked_tool_buttons_and_segments_show_their_state_at_3_to_1(isolated, qapp, name):
    """The checked indicator (a 1 px ink border) against the panel around it, measured on pixels.

    Before: the soft fill alone, 1.07:1 on ``bg_subtle`` to 1.46:1 (dark)."""
    from aerolume.app import theme
    from aerolume.app.widgets.common import SegmentedControl, ToolStrip

    theme.apply(qapp, name)
    sheet = theme.stylesheet(name)
    t = theme.tokens()
    assert f"QToolButton:checked {{ background: {t['primary_soft']}; border-color: {t['fg']}; }}" in sheet
    assert ("fg", "primary_soft", 3.0, "checked indicator (ink border)") in theme.CONTRAST_RULES
    strip = ToolStrip()
    on = strip.add_button("grid", "Grid", "", checkable=True)
    off = strip.add_button("layers", "Physics", "", checkable=True)
    on.setChecked(True)
    seg = SegmentedControl([("a", "CFD"), ("b", "Polar")], "a")
    strip.add_widget(seg)
    strip.resize(400, strip.height())
    strip.show()
    _pump(qapp)
    for checked, unchecked in ((on, off), (seg._buttons["a"], seg._buttons["b"])):
        pc, pu = _pixels(checked), _pixels(unchecked)
        mid = pc.shape[0] // 2
        edge = tuple(int(v) for v in pc[mid, 0])
        # inside the unchecked twin's corner: the panel (or the segment frame) the border sits on
        around = tuple(int(v) for v in pu[2, 1])
        assert _ratio(edge, around) >= 3.0, (name, checked.text(), edge, around)
    strip.close()
    strip.deleteLater()


# ---------------------------------------------------------------------------
# focus ring only for keyboard focus
# ---------------------------------------------------------------------------

def test_study_tile_ring_shows_only_for_keyboard_focus(isolated, qapp):
    from PySide6.QtCore import Qt

    from aerolume.app.main_window import MainWindow
    w = MainWindow(probe_device=False, load_jobs=False)
    _WINDOWS.append(w)
    w.resize(1600, 1000)
    w.show()
    _pump(qapp, 10)
    tile = next(iter(w.page("start").tiles.values()))
    tile.setFocus(Qt.FocusReason.ActiveWindowFocusReason)      # what the window's first focus is
    _pump(qapp)
    assert tile.hasFocus() and not tile.focus_ring_visible()
    tile.clearFocus()
    tile.setFocus(Qt.FocusReason.TabFocusReason)
    _pump(qapp)
    assert tile.focus_ring_visible()
    tile.clearFocus()
    tile.setFocus(Qt.FocusReason.MouseFocusReason)
    _pump(qapp)
    assert tile.hasFocus() and not tile.focus_ring_visible()


# ---------------------------------------------------------------------------
# neutral 3D greys and no v3 fallbacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["light", "dark"])
def test_scene_neutral_greys(name):
    from aerolume.viz3d import colors
    st = colors.scene_theme(name)
    for key in ("ground", "grid", "grid_major", "body", "edge", "solid", "smoke", "background", "background2"):
        r, g, b = getattr(st, key)
        assert abs(r - g) < 1e-9 and abs(g - b) < 1e-9, (name, key, (r, g, b))


#: the v3 blue-grey palette (theme <= 0.3 rc): none of it may survive as a literal colour
V3_HEX = ("#b5bac3", "#dde1e6", "#1d2129", "#454c59", "#6b7280", "#a86a12", "#2f6fdb",
          "#5c6470", "#c9cdd3", "#575e68", "#5b6270", "#e6e8eb")


@pytest.mark.parametrize("rel", ["app/widgets/account_menu.py", "app/widgets/param_form.py",
                                 "app/widgets/draw_tools.py", "app/widgets/view3d.py",
                                 "app/widgets/view3d_panel.py", "viz3d/colors.py"])
def test_no_v3_blue_grey_fallbacks(rel):
    text = (SRC / rel).read_text(encoding="utf-8").lower()
    found = [h for h in V3_HEX if re.search(re.escape(h) + r"\b", text)]
    assert not found, (rel, found)


# ---------------------------------------------------------------------------
# the Setup caption elides
# ---------------------------------------------------------------------------

def test_setup_frame_caption_elides_instead_of_cutting(isolated, qapp):
    from PySide6.QtGui import QFontMetrics
    from PySide6.QtWidgets import QLabel

    from aerolume.app.pages.setup import FrameCaption, frame_text
    full = frame_text(False, True)
    cap = FrameCaption(full)
    cap.show()
    cap.resize(60, 20)
    _pump(qapp)
    shown = QLabel.text(cap)
    assert shown != full and shown.endswith("…")
    assert QFontMetrics(cap.font()).horizontalAdvance(shown) <= cap.width()
    assert cap.text() == full and cap.toolTip() == full
    cap.resize(400, 20)
    _pump(qapp)
    assert QLabel.text(cap) == full
    cap.close()
    cap.deleteLater()


def test_segments_fit_inside_a_24_px_frame(isolated, qapp):
    """The checked segment's ink border must not be clipped: with a QSS border the frame's
    contents were inset 1 px, and a frame given 24 px (a tool strip, an inspector row) cut
    the bottom of its 24 px segments, ink border included."""
    from PySide6.QtWidgets import QAbstractButton

    from aerolume.app.widgets.common import SegmentedControl, ToolStrip
    from aerolume.app.widgets.param_form import SegmentedButtons
    strip = ToolStrip()
    seg = SegmentedControl([("3d", "3D"), ("s", "Slice"), ("p", "Split")], "3d")
    strip.add_widget(seg)
    buttons = SegmentedButtons([("cfd", "CFD"), ("polar", "Polar")])
    buttons.buttons["polar"].setChecked(True)
    strip.add_widget(buttons)
    strip.resize(500, strip.height())
    strip.show()
    _pump(qapp)
    for frame in (seg, buttons):
        for b in frame.findChildren(QAbstractButton):
            assert b.height() >= 24, b.text()                               # WCAG 2.5.8 target
            assert b.geometry().top() >= 0 and b.geometry().bottom() < frame.height(), (
                b.text(), b.geometry().getRect(), frame.height())
    strip.close()
    strip.deleteLater()
