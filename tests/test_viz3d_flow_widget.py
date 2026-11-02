"""View3D with flow in the volume (DESIGN.md 7.13): smoke, streamlines and the draggable rake.

* smoke and streamlines are one layer each, updated in place (no layer churn, so no
  flashing during a run);
* ``show_live_frame`` re-uploads a volume only when its serial changed;
* the rake handle is dragged with the mouse in the plane facing the camera, the
  grabbed point stays under the cursor, ``rakeMoving`` / ``rakeChanged`` report it;
* what the widget shows is exactly the scene's off-screen render.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vtkmodules")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from test_viz3d_fixtures import sphere_flow_volume, uv_sphere

from aerolume.viz3d.flow import resolve_rake, trace_streamlines
from aerolume.viz3d.scene import backend_info

_INFO = backend_info()
pytestmark = [pytest.mark.gui,
              pytest.mark.skipif(not _INFO["ok"], reason=f"no off-screen OpenGL: {_INFO['error']}")]
BODY = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
DOMAIN = (-1.5, 2.5, -1.0, 1.0, -1.0, 1.0)


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
    v.scene.add_mesh(uv_sphere(1.0, 24, 48))
    yield v
    v.close()
    v.scene.close()
    v.deleteLater()
    qapp.processEvents()


def _smoke_volume():
    vol = sphere_flow_volume(n=(64, 40, 40))
    z = (np.arange(40) + 0.5) / 40 * 2 - 1
    dye = np.exp(-((z[:, None, None] - 0.2) ** 2) / 0.01) * np.ones((40, 40, 64))
    dye[vol.mask > 0.5] = 0
    vol.fields["dye"] = dye.astype(np.float16)
    return vol


def grab_rgb(widget) -> np.ndarray:
    img = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(img.constBits(), np.uint8, count=img.bytesPerLine() * h)
    return buf.reshape(h, img.bytesPerLine())[:, : 3 * w].reshape(h, w, 3).copy()


def send_mouse(widget, kind, pos, button, buttons):
    p = QPointF(*pos)
    ev = QMouseEvent(kind, p, widget.mapToGlobal(p), button, buttons, Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, ev)


def test_smoke_and_streamlines_are_single_layers_updated_in_place(view):
    vol = _smoke_volume()
    view.set_smoke(vol)
    view.set_streamlines(vol, seeds={"kind": "rake", "n": 10})
    layers = dict(view.scene.layers())
    view.set_smoke(vol)
    view.set_streamlines(vol, seeds={"kind": "rake", "n": 6})
    assert view.scene.layers() == layers                          # no new layers: in-place updates
    assert sorted(layers.values()) == ["mesh", "smoke", "streamlines"]
    view.set_smoke(None)
    view.set_streamlines(None)
    assert sorted(view.scene.layers().values()) == ["mesh"]


def test_live_frames_upload_a_volume_only_when_it_is_new(view, monkeypatch):
    vol = _smoke_volume()
    lines = trace_streamlines(vol, resolve_rake({"n": 8}, BODY, DOMAIN))
    spec = resolve_rake(None, BODY, DOMAIN).spec()
    calls = []
    orig = view.scene.update_smoke
    monkeypatch.setattr(view.scene, "update_smoke", lambda *a, **k: (calls.append(1), orig(*a, **k)))
    fr = SimpleNamespace(volume=vol, volume_serial=1, streamlines3d=lines, seed_rake=spec)
    view.show_live_frame(fr)
    assert sorted(view.scene.layers().values()) == ["mesh", "rake", "smoke", "streamlines"]
    view.show_live_frame(fr)                                      # the same volume again: nothing uploaded
    view.show_live_frame(SimpleNamespace(volume=vol, volume_serial=2, streamlines3d=lines, seed_rake=spec))
    assert len(calls) == 1
    view.show_live_frame(SimpleNamespace(volume=None, volume_serial=0, streamlines3d=None, seed_rake=spec))
    assert "smoke" in view.scene.layers().values()                 # a frame without a volume keeps the smoke
    view.show_live_frame(fr, smoke=False, streamlines=False, rake=False)
    assert sorted(view.scene.layers().values()) == ["mesh"]


def test_widget_shows_exactly_the_scene_render(view, qapp):
    view.set_smoke(_smoke_volume())
    view.set_seed_rake({"kind": "rake"}, domain_bounds=DOMAIN)
    view.set_view("iso")
    qapp.processEvents()
    a = grab_rgb(view).astype(int)
    b = view.scene.render(view.width(), view.height()).astype(int)
    assert a.shape == b.shape and np.abs(a - b).max() <= 2


@pytest.mark.parametrize("preset,fixed", [("front", 0), ("side", 1), ("top", 2)])
def test_dragging_the_rake_handle_moves_it_under_the_cursor(view, qapp, preset, fixed):
    view.set_seed_rake({"kind": "rake"}, body_bounds=BODY, domain_bounds=DOMAIN)
    view.set_view(preset)
    qapp.processEvents()
    view.scene.render(view.width(), view.height())
    sr = view.seed_rake()
    start = view.scene.project(sr.centre[None, :])[0]
    moving, changed = [], []
    view.rakeMoving.connect(moving.append)
    view.rakeChanged.connect(changed.append)
    a = (float(start[0]), float(start[1]))
    b = (a[0] + 40.0, a[1] - 25.0)
    send_mouse(view, QEvent.Type.MouseButtonPress, a, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    send_mouse(view, QEvent.Type.MouseMove, b, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    send_mouse(view, QEvent.Type.MouseButtonRelease, b, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert moving and changed
    moved = view.seed_rake()
    delta = moved.centre - sr.centre
    assert abs(delta[fixed]) < 1e-9                              # moved in the plane facing the camera
    assert np.linalg.norm(delta) > 0.05
    end = view.scene.project(moved.centre[None, :])[0]
    assert np.hypot(end[0] - b[0], end[1] - b[1]) < 1.0           # the handle followed the cursor
    spec = changed[-1]
    assert resolve_rake(spec, BODY, (-9, 9, -9, 9, -9, 9)).centre == pytest.approx(moved.centre)


def test_a_click_away_from_the_handle_still_orbits(view, qapp):
    view.set_seed_rake({"kind": "rake"}, body_bounds=BODY, domain_bounds=DOMAIN)
    view.set_view("iso")
    qapp.processEvents()
    view.scene.render(view.width(), view.height())
    cam0 = view.scene.camera
    changed = []
    view.rakeChanged.connect(changed.append)
    a = (20.0, 20.0)
    send_mouse(view, QEvent.Type.MouseButtonPress, a, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    send_mouse(view, QEvent.Type.MouseMove, (60.0, 30.0), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    send_mouse(view, QEvent.Type.MouseButtonRelease, (60.0, 30.0), Qt.MouseButton.LeftButton,
               Qt.MouseButton.NoButton)
    assert not changed and view.scene.camera.position != cam0.position


def test_clear_flow_and_read_only_rake(view):
    view.set_seed_rake({"kind": "grid"}, editable=False, body_bounds=BODY, domain_bounds=DOMAIN)
    assert not view.hasMouseTracking()
    view.set_smoke(_smoke_volume())
    view.clear_flow()
    assert sorted(view.scene.layers().values()) == ["mesh"] and view.seed_rake() is None
