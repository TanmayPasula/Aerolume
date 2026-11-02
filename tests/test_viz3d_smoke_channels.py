"""Several smoke emitters in one 3D picture (critique round 3): a white rake, an amber rake and a blue
point emitter on a car. On the light theme the white smoke vanished against the floor and the point
emitter drew nothing at all; the side panel did not say which colour was which emitter.

* :func:`aerolume.viz3d.scene.smoke_gain` lifts a weak, sparse channel to a visible density (measured
  on the stored car run: the point emitter's 90th percentile was 0.15, where the opacity curve is
  nearly clear);
* the smoke colour darkens towards its cores on the light theme (brightens on the dark one);
* :class:`~aerolume.app.widgets.view3d_panel.SmokeLegend` names each channel with its drawn colour and
  switches it on and off.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerolume.model import Emitter, Simulation, VolumeData


def _thin_stream(n=(48, 40, 96), conc: float = 0.15) -> np.ndarray:
    """A point emitter's stream: two cells wide, weak (0.1-0.2), along x."""
    nz, ny, nx = n
    a = np.zeros(n, np.float32)
    a[nz // 2 - 1: nz // 2 + 1, ny // 2 - 1: ny // 2 + 1, 10:] = conc * np.linspace(1.3, 0.7, nx - 10)
    return a


def _sheet(n=(48, 40, 96)) -> np.ndarray:
    """A rake's sheet: wide, dense near the rake."""
    _nz, ny, nx = n
    a = np.zeros(n, np.float32)
    a[10:38, ny // 2 - 2: ny // 2 + 2, 10:] = np.linspace(0.9, 0.3, nx - 10)
    return a


def test_smoke_gain_lifts_a_weak_sparse_channel_and_leaves_a_dense_one():
    from aerolume.viz3d.scene import SMOKE_GAIN_MAX, SMOKE_GAIN_TARGET, smoke_gain
    g_sheet, b_sheet = smoke_gain(_sheet())
    assert g_sheet == pytest.approx(1.0) and b_sheet >= 1.0
    g, b = smoke_gain(_thin_stream())
    smoky = _thin_stream()[_thin_stream() > 0.02]
    assert g == pytest.approx(SMOKE_GAIN_TARGET / np.percentile(smoky, 90), rel=1e-6)
    assert 1.0 < g <= SMOKE_GAIN_MAX and b > 1.0
    # never lowers a channel, never more than the cap, and an empty channel is left alone
    assert smoke_gain(np.full((4, 4, 4), 0.03, np.float32))[0] == SMOKE_GAIN_MAX
    assert smoke_gain(np.zeros((4, 4, 4), np.float32)) == (1.0, 1.0)
    # uint8 input reads as 0..1
    assert smoke_gain((_thin_stream() * 255).astype(np.uint8))[0] == pytest.approx(g, rel=0.05)


def test_smoke_core_colour_keeps_a_colour_saturated_and_shades_a_grey():
    """Critic round 4: shaded towards black / white, the car's orange and blue channels became one grey-brown
    haze. A coloured channel now keeps its hue at full saturation (a little darker on the light theme); a
    neutral one (white smoke, the theme grey) still darkens towards its cores on light and brightens on dark."""
    import colorsys

    from aerolume.viz3d.scene import (
        SMOKE_CHROMA_DARKEN,
        SMOKE_CORE_SHADE,
        smoke_core_colour,
        smoke_display_rgb,
    )
    amber = (0.94, 0.63, 0.25)
    shown = smoke_display_rgb(amber)
    h0, _s0, v0 = colorsys.rgb_to_hsv(*amber)
    h1, s1, v1 = colorsys.rgb_to_hsv(*shown)
    assert s1 == pytest.approx(1.0) and h1 == pytest.approx(h0) and v1 == pytest.approx(v0)
    for dark in (False, True):
        h, s, v = colorsys.rgb_to_hsv(*smoke_core_colour(amber, dark=dark))
        assert h == pytest.approx(h0, abs=1e-6) and s == pytest.approx(1.0)
        assert v == pytest.approx(v0 * (1 - SMOKE_CHROMA_DARKEN) if not dark else 0.95)
    blue = (0.34, 0.71, 0.91)                       # "#56b4e9": sky blue, drawn as a saturated blue
    assert colorsys.rgb_to_hsv(*smoke_display_rgb(blue))[1] == pytest.approx(1.0)
    grey = (0.6, 0.6, 0.62)
    assert smoke_display_rgb(grey) == pytest.approx(grey)
    light, dark = smoke_core_colour(grey, dark=False), smoke_core_colour(grey, dark=True)
    assert all(c2 < c1 for c1, c2 in zip(grey, light, strict=True))
    assert all(c2 > c1 for c1, c2 in zip(grey, dark, strict=True))
    assert light[0] == pytest.approx(0.6 * (1 - SMOKE_CORE_SHADE))


def test_smoke_next_to_a_wall_is_cleared():
    """Smoke within one cell of a solid is cleared (the boundary layer's dye coated the car body)."""
    from aerolume.viz3d.scene import SMOKE_WALL_CLEAR_CELLS, Scene
    n = (20, 20, 20)
    mask = np.zeros(n, np.float32)
    mask[8:12, 8:12, 8:12] = 1.0
    dye = np.full(n, 0.8, np.float32)
    vol = VolumeData(origin=(0.0, 0.0, 0.0), spacing=(0.1, 0.1, 0.1), fields={"dye": dye}, mask=mask,
                     label="instant")
    _o, _s, q = Scene.__new__(Scene)._smoke_array(vol, "dye")
    assert SMOKE_WALL_CLEAR_CELLS == 1
    assert q[10, 10, 10] == 0 and q[10, 10, 7] == 0 and q[7, 7, 7] == 0      # solid, face and corner neighbours
    assert q[10, 10, 6] > 0 and q[0, 0, 0] > 0                                # two cells away: kept


def test_colour_words_and_channel_labels_name_unnamed_emitters():
    from aerolume.app.widgets.view3d_panel import colour_word, smoke_channel_labels
    assert colour_word("#f5f5f5") == "white" and colour_word("#40a0f0") == "blue"
    assert colour_word("#f0a040") == "orange" and colour_word("#808080") == "grey"
    sim = Simulation(name="Car smoke", design_id="d", emitters=[
        Emitter(kind="rake", position=(0, 0, 0), end=(0, 0, 1), color="#f5f5f5"),
        Emitter(kind="rake", position=(0, 0, 0), end=(0, 1, 0), color="#f0a040"),
        Emitter(kind="point", position=(0, 0, 0), color="#40a0f0"),
        Emitter(name="Mirror wake", kind="point", position=(0, 0, 0), color="#f0a040")])
    labels = smoke_channel_labels(sim)
    assert list(labels) == ["dye", "dye1", "dye2"]
    assert labels["dye"] == ["Rake, white"] and labels["dye1"] == ["Rake, orange", "Mirror wake"]
    assert labels["dye2"] == ["Point, blue"]


vtk = pytest.importorskip("vtkmodules")


def _backend_ok() -> bool:
    from aerolume.viz3d.scene import backend_info
    return bool(backend_info()["ok"])


needs_gl = pytest.mark.skipif(not _backend_ok(), reason="no off-screen OpenGL")


def _volume(fields: dict) -> VolumeData:
    _nz, ny, _nx = next(iter(fields.values())).shape
    return VolumeData(origin=(0.0, -0.5 * ny * 0.02, 0.0), spacing=(0.02, 0.02, 0.02), fields=fields, mask=None,
                      label="instant")


@needs_gl
def test_gain_reaches_the_uploaded_volume_and_makes_a_point_stream_visible():
    """Rendered on the light theme: without the gain the weak two-cell stream barely shifts the background;
    with :func:`smoke_gain` it is a clearly visible line (measured as the largest colour change)."""
    from aerolume.viz3d.scene import Scene, smoke_gain
    vol = _volume({"dye": _thin_stream()})
    diffs = {}
    for label, gain_boost in (("plain", (1.0, 1.0)), ("gained", smoke_gain(vol.fields["dye"]))):
        s = Scene("light")
        try:
            h = s.add_smoke(vol, "dye", color="#40a0f0", opacity=gain_boost[1], gain=gain_boost[0])
            assert s.layer_info(h)["gain"] == pytest.approx(gain_boost[0])
            s.set_camera("top")
            img = s.render(320, 200).astype(np.int16)
            s.remove(h)
            bg = s.render(320, 200).astype(np.int16)
        finally:
            s.close()
        diffs[label] = int(np.abs(img - bg).sum(axis=2).max())
    assert diffs["gained"] >= 1.5 * diffs["plain"] and diffs["gained"] >= 60, diffs


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


@needs_gl
@pytest.mark.gui
def test_smoke_legend_names_channels_and_switches_them(qapp):
    from aerolume.app.widgets.view3d import View3D
    from aerolume.app.widgets.view3d_panel import FlowScene, SmokeLegend, dye_colours_of, smoke_channel_labels
    sim = Simulation(name="Car smoke", design_id="d", emitters=[
        Emitter(kind="rake", position=(0, 0, 0), end=(0, 0, 1), color="#f5f5f5"),
        Emitter(kind="rake", position=(0, 0, 0), end=(0, 1, 0), color="#f0a040"),
        Emitter(kind="point", position=(0, 0, 0), color="#40a0f0")])
    view = View3D()
    try:
        view.resize(480, 320)
        flow = FlowScene(view, shown={"smoke": True})
        flow.set_dye_colors(dye_colours_of(sim), rebuild=False)
        flow.set_channel_names(smoke_channel_labels(sim))
        flow.set_smoke(_volume({"dye": _sheet(), "dye1": _sheet()[:, ::-1], "dye2": _thin_stream()}))
        legend = SmokeLegend()
        legend.bind(flow)
        items = flow.smoke_legend()
        assert [i["field"] for i in items] == ["dye", "dye1", "dye2"] and legend.rows() == ["dye", "dye1", "dye2"]
        assert items[0]["color"] != "#f5f5f5"          # white smoke is drawn (and shown) in the theme grey
        assert items[2]["color"] == "#0083f0" and items[2]["names"] == ["Point, blue"]   # drawn (and shown) at full saturation
        assert len(flow.handles.get("smoke", [])) == 3
        assert flow.smoke_gains["dye2"][0] > 1.0 and flow.smoke_gains["dye"][0] == pytest.approx(1.0)
        box = legend._rows["dye"][0]
        box.setChecked(False)                          # switch the white rake off
        assert not flow.channel_visible("dye") and len(flow.handles.get("smoke", [])) == 2
        box.setChecked(True)
        assert len(flow.handles.get("smoke", [])) == 3
    finally:
        view.close()
