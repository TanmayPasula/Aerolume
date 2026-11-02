"""View3D in offscreen Qt: same pixels as Scene.render, mouse camera control, toolbar, theme."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("vtkmodules")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from test_viz3d_fixtures import vehicle_meshes, vehicle_surfaces

from aerolume.viz3d import scene as scene_mod
from aerolume.viz3d.scene import backend_info

_INFO = backend_info()
pytestmark = [pytest.mark.gui,
              pytest.mark.skipif(not _INFO["ok"], reason=f"no off-screen OpenGL: {_INFO['error']}")]

SAMPLES = os.environ.get("AEROLUME_VIZ3D_SAMPLES", "")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    before = theme.current()
    theme.apply(app, "light")
    yield app
    theme.apply(app, before)


@pytest.fixture
def view(qapp):
    from aerolume.app.widgets.view3d import View3D
    v = View3D()
    v.resize(640, 400)
    yield v
    v.close()
    v.scene.close()
    v.deleteLater()
    qapp.processEvents()


def grab_rgb(widget) -> np.ndarray:
    img = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
    return buf.reshape(h, img.bytesPerLine())[:, : 3 * w].reshape(h, w, 3).copy()


def send_mouse(widget, kind, pos, button, buttons, mods=Qt.KeyboardModifier.NoModifier):
    p = QPointF(*pos)
    ev = QMouseEvent(kind, p, widget.mapToGlobal(p), button, buttons, mods)
    QtWidgets.QApplication.sendEvent(widget, ev)


def drag(widget, a, b, button=Qt.MouseButton.LeftButton, mods=Qt.KeyboardModifier.NoModifier):
    send_mouse(widget, QEvent.Type.MouseButtonPress, a, button, button, mods)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    send_mouse(widget, QEvent.Type.MouseMove, mid, Qt.MouseButton.NoButton, button, mods)
    send_mouse(widget, QEvent.Type.MouseMove, b, Qt.MouseButton.NoButton, button, mods)
    send_mouse(widget, QEvent.Type.MouseButtonRelease, b, button, Qt.MouseButton.NoButton, mods)


def test_grab_is_pixel_identical_to_scene_render(view):
    view.scene.add_meshes(vehicle_surfaces(), "cp")
    view.scene.add_ground(0.0)
    view.scene.add_colorbar()
    view.set_view("iso")
    got = grab_rgb(view).astype(int)
    want = view.scene.render(640, 400).astype(int)
    again = view.scene.render(640, 400).astype(int)
    assert got.shape == want.shape == (400, 640, 3)
    # Two GPU renders of one scene differ by at most 1 level in a few dozen anti-aliased
    # edge pixels; the widget's paint adds nothing beyond that (no scaling, no colour change).
    jitter = int((np.abs(again - want).max(axis=2) > 0).sum())
    diff = np.abs(got - want).max(axis=2)
    assert diff.max() <= 1
    assert (diff > 0).sum() <= max(200, 4 * jitter)
    if SAMPLES:
        from PIL import Image
        Image.fromarray(got.astype(np.uint8)).save(os.path.join(SAMPLES, "view3d_grab.png"))


def test_renders_only_when_something_changed(view, qapp):
    view.scene.add_meshes(vehicle_meshes())
    count = []
    view.rendered.connect(count.append)
    grab_rgb(view)
    grab_rgb(view)
    assert len(count) == 1                      # the second paint reuses the cached image
    view.scene.set_edges(True)
    grab_rgb(view)
    assert len(count) == 2
    view.resize(500, 300)
    grab_rgb(view)
    assert len(count) == 3
    assert view.last_render_ms < 100


def test_left_drag_orbits_right_drag_pans_wheel_zooms(view):
    view.scene.add_meshes(vehicle_meshes())
    view.set_view("side")
    grab_rgb(view)
    moved = []
    view.cameraChanged.connect(lambda: moved.append(1))
    c0 = view.scene.camera
    drag(view, (300, 200), (380, 200))                              # orbit right
    c1 = view.scene.camera
    assert c1.distance == pytest.approx(c0.distance)
    assert c1.position[0] < c0.position[0]                          # camera swung towards the nose
    assert moved
    drag(view, (300, 200), (340, 170), Qt.MouseButton.RightButton)   # pan
    c2 = view.scene.camera
    assert c2.distance == pytest.approx(c1.distance)
    assert not np.allclose(c2.focal_point, c1.focal_point)
    fp = c2.focal_point
    p0 = view.scene.project([fp])[0]
    drag(view, (300, 200), (310, 220), Qt.MouseButton.MiddleButton)
    p1 = view.scene.project([fp])[0]
    assert np.allclose(p1 - p0, [10, 20], atol=1e-6)                # the scene follows the cursor
    drag(view, (300, 200), (290, 200), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    assert view.scene.camera.distance == pytest.approx(c2.distance)  # shift + left pans too
    d0 = view.scene.camera.distance
    pos = QPointF(320, 200)
    ev = QWheelEvent(pos, view.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 240), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(view, ev)
    assert view.scene.camera.distance == pytest.approx(d0 / 1.18 ** 2)


def test_double_click_fits_the_model_again(view):
    view.scene.add_meshes(vehicle_meshes())
    view.set_view("iso")
    grab_rgb(view)
    view.scene.zoom(6.0, at=(30, 30))
    grab_rgb(view)
    p = QPointF(200, 200)
    ev = QMouseEvent(QEvent.Type.MouseButtonDblClick, p, view.mapToGlobal(p), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(view, ev)
    grab_rgb(view)
    pts = view.scene.project(view.scene.focus_points())
    assert pts[:, 0].min() > 0 and pts[:, 0].max() < 640 and pts[:, 1].min() > 0 and pts[:, 1].max() < 400


def test_toolbar_presets_edges_and_overlay(view, qapp):
    view.scene.add_meshes(vehicle_meshes())
    bar = view.toolbar()
    assert list(bar.preset_buttons) == ["iso", "front", "side", "top", "back"]
    bar.preset_buttons["top"].click()
    _, _, f = view.scene.camera.frame()
    assert np.allclose(f, [0, 0, -1], atol=1e-9)
    bar.edges_button.click()
    assert view.edges() and view.scene.edges
    view.set_edges(False)
    assert not bar.edges_button.isChecked()             # every toolbar of the view follows
    plain = grab_rgb(view)
    view.set_overlay_toolbar(True)
    assert view.overlay_toolbar() is not None
    with_bar = grab_rgb(view)
    assert np.abs(with_bar[:40, :200].astype(int) - plain[:40, :200]).mean() > 3
    view.overlay_toolbar().edges_button.click()
    assert view.edges()
    view.set_overlay_toolbar(False)
    bar.deleteLater()


def test_theme_switch_repaints_background(view, qapp):
    """The viewport is a well: it paints the theme's ``surface`` token, and repaints it on a switch.

    Read from the token tables rather than pinned as literals, so a palette change (v4 took the
    website's near-black, #1a1a1a, over v3's #2d2f34 graphite) cannot fail this for the wrong reason.
    """
    from aerolume.app import theme

    def rgb(name: str) -> list[int]:
        h = theme.semantic_tokens(name)["surface"].lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

    view.scene.add_meshes(vehicle_meshes())
    light = grab_rgb(view)
    assert np.array_equal(light[2, 320], rgb("light")) and rgb("light") == [255, 255, 255]
    theme.apply(qapp, "dark")
    try:
        dark = grab_rgb(view)
        assert np.array_equal(dark[2, 320], rgb("dark"))
        assert sum(rgb("dark")) < 3 * 64          # still a dark well, not the light one
    finally:
        theme.apply(qapp, "light")


def test_device_pixel_ratio_renders_at_device_resolution(view):
    view.scene.add_meshes(vehicle_meshes())
    view._dpr = lambda: 2.0
    img = view._ensure_image()
    assert (img.width(), img.height()) == (1280, 800)
    assert img.devicePixelRatio() == 2.0


def test_placeholder_and_unavailable_backend(qapp, monkeypatch):
    from aerolume.app.widgets.view3d import View3D
    v = View3D()
    v.resize(400, 300)
    v.set_placeholder("No geometry", "Import an STL, OBJ or PLY file")
    img = grab_rgb(v)
    assert (np.abs(img.astype(int) - 255).max(axis=2) > 30).any()       # text drawn on white
    v.scene.close()
    v.close()
    monkeypatch.setattr(scene_mod, "_BACKEND", {"ok": False, "window": "", "renderer": "", "version": "",
                                                "error": "no OpenGL 3.2 context"})
    u = View3D()
    u.resize(400, 300)
    u.scene.add_meshes(vehicle_meshes())
    img = grab_rgb(u)                                                     # must not raise
    assert img.shape == (300, 400, 3)
    assert (np.abs(img.astype(int) - 255).max(axis=2) > 30).any()       # the message is painted
    u.close()


def test_overlay_toolbar_is_borderless_icon_buttons(view, qapp):
    from PySide6.QtWidgets import QFrame
    view.scene.add_meshes(vehicle_meshes())
    view.set_overlay_toolbar(True)
    bar = view.overlay_toolbar()
    buttons = [*bar.preset_buttons.values(), bar.fit_button, bar.edges_button]
    assert len(buttons) == 7
    for b in buttons:
        assert b.text() == "" or b.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly
        assert not b.icon().isNull() and b.toolTip()                # icon + tooltip, no label
        assert b.width() == b.height() == 24
        pm = b.icon().pixmap(16, 16).toImage()
        assert any(pm.pixelColor(x, y).alpha() > 0 for x in range(16) for y in range(16))   # it draws
    assert bar.frameShape() == QFrame.Shape.NoFrame and not bar.autoFillBackground()
    assert bar.height() == 24 and bar.width() < 7 * 24 + 16
    bar.preset_buttons["side"].click()
    _, _, f = view.scene.camera.frame()
    assert abs(abs(f[1]) - 1.0) < 1e-9                              # side view: looking along y
    view.set_overlay_toolbar(False)


def _split(qapp, view, host: bool):
    from PySide6.QtWidgets import QFrame, QSplitter, QVBoxLayout

    from aerolume.app.widgets.field_view import FieldView
    sp = QSplitter()
    if host:                                   # Simulate: the view sits in a host frame
        frame = QFrame()
        QVBoxLayout(frame).addWidget(view)
        sp.addWidget(frame)
    else:                                      # Results: directly in the splitter
        sp.addWidget(view)
    fv = FieldView()
    sp.addWidget(fv)
    sp.resize(1100, 420)
    sp.show()
    qapp.processEvents()
    return sp, fv


@pytest.mark.parametrize("host", [False, True])
def test_split_view_shows_one_legend_and_one_time_label(view, qapp, host):
    from test_viz3d_fixtures import sphere_surface
    h = view.scene.add_mesh(sphere_surface(1.0), "cp")
    view.scene.add_colorbar(layer=h)
    view.scene.set_caption("t = 3 L/U · step 1,200")
    spec = view.scene.colorbar_spec()
    sp, fv = _split(qapp, view, host)
    try:
        assert view.annotations_shown() == {"colorbar": True, "caption": True}   # partner has no image yet
        vals = np.linspace(spec.vmin, spec.vmax, 40 * 60, dtype=np.float32).reshape(40, 60)
        fv.set_field(vals, (-1.0, 2.0, -1.0, 1.0), field="cp", vmin=spec.vmin, vmax=spec.vmax)
        fv.set_caption("t = 3 L/U · step 1,200")
        qapp.processEvents()
        assert view.legend_partner() is fv
        assert view.annotations_shown() == {"colorbar": False, "caption": False}
        grab_rgb(view)
        assert view.scene.view_size()[0] == round(view.width() * view.devicePixelRatioF())   # full width
        fv.set_caption("t = 3.1 L/U · step 1,262")                    # the slice is a frame ahead
        assert not view.annotations_shown()["caption"]                # still the same label: shown once
        fv.set_caption("t = 3 L/U · y = 0 m")                         # different text: keep ours
        fv.set_field(vals, (-1.0, 2.0, -1.0, 1.0), field="cp", vmin=spec.vmin, vmax=spec.vmax + 5.0)
        assert view.annotations_shown() == {"colorbar": True, "caption": True}   # another legend: keep ours
        fv.set_field(vals, (-1.0, 2.0, -1.0, 1.0), field="cp", vmin=spec.vmin, vmax=spec.vmax)
        view.set_legend(True)                                         # forced on
        assert view.annotations_shown()["colorbar"]
        view.set_legend(None)
        assert not view.annotations_shown()["colorbar"]
        fv.hide()                                                     # View: 3D again
        qapp.processEvents()
        assert view.annotations_shown()["colorbar"]
        assert view.legend_partner() is None
    finally:
        view.setParent(None)
        sp.close()
        sp.deleteLater()
        qapp.processEvents()
