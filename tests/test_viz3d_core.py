"""viz3d without OpenGL: colours, camera maths, data conversion and overlay layout."""
from __future__ import annotations

import math

import numpy as np
import pytest
from test_viz3d_fixtures import sphere_flow_volume, uv_sphere, vortex_pair_volume

from aerolume.model import Mesh3D, SliceData, SurfaceData3D
from aerolume.viz import colormaps as cm2d
from aerolume.viz3d import camera as cam
from aerolume.viz3d import colors, data, overlay
from aerolume.viz3d.scene import rake_seeds

# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------

def test_viz3d_package_import_does_not_load_vtk():
    import subprocess
    import sys
    code = ("import sys, aerolume.viz3d, aerolume.viz3d.colors, aerolume.viz3d.camera, aerolume.viz3d.scene;"
            "print(any(m.startswith('vtkmodules') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_colorize_uses_the_2d_luts_exactly():
    vals = np.linspace(-1.0, 1.0, 256)
    rgb = colors.colorize(vals, "coolwarm", -1.0, 1.0)
    assert np.array_equal(rgb, cm2d.COLORMAPS["coolwarm"])
    rgb = colors.colorize(np.array([0.0, 0.5, 1.0]), "viridis", 0.0, 1.0, mask=np.array([0, 1, 0]))
    assert tuple(rgb[1]) == (143, 148, 156)        # solid grey, as in the 2D views
    assert tuple(rgb[0]) == tuple(cm2d.COLORMAPS["viridis"][0])


def test_lookup_table_matches_2d_lut():
    pytest.importorskip("vtkmodules")
    lut = colors.lookup_table("RdBu_r", -2.0, 2.0)
    assert lut.GetNumberOfTableValues() == 256
    assert lut.GetRange() == (-2.0, 2.0)
    ref = cm2d.COLORMAPS["RdBu_r"]
    for i in (0, 77, 128, 255):
        r, g, b, _ = lut.GetTableValue(i)
        assert np.allclose(np.array([r, g, b]) * 255, ref[i], atol=0.51)


def test_display_range_centred_for_diverging_and_zero_based_for_magnitudes():
    rng = np.random.default_rng(0)
    cp = np.concatenate([rng.uniform(-2.5, 1.0, 5000), [40.0]])        # one wild vertex
    lo, hi = colors.display_range(cp, field="cp")
    assert lo == -hi and 2.3 < hi < 2.6                                # centred, outlier ignored
    lo, hi = colors.display_range(rng.uniform(4.0, 12.0, 1000), field="speed")
    assert lo == 0.0 and 11.5 < hi <= 12.0
    lo, hi = colors.display_range(rng.uniform(4.0, 12.0, 1000), field="u")
    assert lo > 3.9                                                     # u is not a magnitude
    lo, hi = colors.display_range(np.full(10, 3.0), field="u")
    assert lo < 3.0 < hi                                                # constant -> usable range
    assert colors.display_range(np.array([np.nan]), field="cp") == (-1.0, 1.0)
    lo, hi = colors.display_range(rng.normal(0, 1, 100), field="sensitivity")
    assert lo == -hi


def test_field_info_3d_units_and_maps():
    assert colors.field_info("q").unit == "1/s²"
    assert colors.field_info("vorticity").diverging is False           # |omega| in 3D
    assert colors.default_cmap("cp") == "coolwarm"
    assert colors.default_cmap("unknown_field") == "viridis"
    for info in colors.FIELD_INFO_3D.values():
        assert info.cmap in cm2d.COLORMAPS


def test_scene_theme_from_tokens_and_names():
    light, dark = colors.scene_theme("light"), colors.scene_theme("dark")
    assert light.background == (1.0, 1.0, 1.0) and not light.dark
    assert dark.dark and dark.background == colors.hex_to_rgb("#1a1a1a")
    custom = colors.scene_theme({"surface": "#202020", "text": "#eeeeee"})
    assert custom.dark and custom.text == colors.hex_to_rgb("#eeeeee")
    assert colors.scene_theme("light", gradient=True).gradient
    with pytest.raises(ValueError):
        colors.scene_theme("navy")
    # restrained: floor and grid only a few percent away from the background
    assert max(abs(a - b) for a, b in zip(light.ground, light.background)) < 0.06


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------

BOX = (-2.2, 2.2, -0.9, 0.9, 0.0, 1.6)


@pytest.mark.parametrize("preset", list(cam.PRESETS))
@pytest.mark.parametrize("aspect", [0.6, 1.0, 16 / 9, 3.0])
def test_fit_puts_every_corner_inside_with_margin(preset, aspect):
    d, up = cam.preset(preset)
    st = cam.fit(cam.CameraState(), BOX, aspect, direction=d, view_up=up)
    w, h = 1000.0 * aspect, 1000.0
    px = cam.project(st, cam._corners(BOX), w, h)
    assert np.all(np.isfinite(px))
    assert px[:, 0].min() >= 0 and px[:, 0].max() <= w
    assert px[:, 1].min() >= 0 and px[:, 1].max() <= h
    # tight in at least one direction (the margin is 12 % of the half size)
    span = max((px[:, 0].max() - px[:, 0].min()) / w, (px[:, 1].max() - px[:, 1].min()) / h)
    assert span > 0.8
    # centred
    assert abs(0.5 * (px[:, 0].max() + px[:, 0].min()) - w / 2) < 0.02 * w
    assert abs(0.5 * (px[:, 1].max() + px[:, 1].min()) - h / 2) < 0.02 * h
    # looks from the preset direction
    _, _, f = st.frame()
    assert np.dot(-f, d) > 0.999


def test_fit_to_points_is_tighter_than_the_box():
    v, _ = uv_sphere(1.0, 24, 48)
    box = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    d, up = cam.preset("iso")
    a = cam.fit(cam.CameraState(), box, 1.5, direction=d, view_up=up)
    b = cam.fit(cam.CameraState(), box, 1.5, direction=d, view_up=up, points=v)
    assert b.distance < 0.85 * a.distance


def test_presets_follow_the_frame_conventions():
    # side: flow (+x) to the right of the screen, z up; top: +x right, +y up the screen
    for name, right, up in (("side", (1, 0, 0), (0, 0, 1)), ("top", (1, 0, 0), (0, 1, 0)),
                            ("front", (0, -1, 0), (0, 0, 1))):
        d, vu = cam.preset(name)
        st = cam.fit(cam.CameraState(), BOX, 1.5, direction=d, view_up=vu)
        r, u, _ = st.frame()
        assert np.allclose(r, right, atol=1e-9), name
        assert np.allclose(u, up, atol=1e-9), name
    with pytest.raises(ValueError):
        cam.preset("diagonal")


def test_orbit_turntable_keeps_distance_horizon_and_clamps_elevation():
    d, up = cam.preset("side")
    st = cam.fit(cam.CameraState(), BOX, 1.5, direction=d, view_up=up)
    o = cam.orbit(st, 30.0, 0.0)
    assert math.isclose(o.distance, st.distance, rel_tol=1e-9)
    assert np.allclose(o.focal_point, st.focal_point)
    r, _, _ = o.frame()
    assert abs(r[2]) < 1e-9                         # no roll: the horizon stays level
    # dragging right moves the camera towards the nose (-x), turning the model with the mouse
    assert o.position[0] < st.position[0]
    hi = cam.orbit(st, 0.0, 200.0)
    off = np.subtract(hi.position, hi.focal_point)
    assert math.isclose(off[2] / np.linalg.norm(off), 1.0, abs_tol=1e-9)    # clamped at +90
    back = cam.orbit(hi, 0.0, -90.0)
    assert np.allclose(back.position, st.position, atol=1e-9 * st.distance)


def test_pan_follows_the_cursor_and_zoom_keeps_the_cursor_point():
    d, up = cam.preset("iso")
    w, h = 1200.0, 800.0
    st = cam.fit(cam.CameraState(), BOX, w / h, direction=d, view_up=up)
    p0 = cam.project(st, [st.focal_point], w, h)[0]
    moved = cam.pan(st, 40.0, -25.0, h)
    p1 = cam.project(moved, [st.focal_point], w, h)[0]
    assert np.allclose(p1 - p0, [40.0, -25.0], atol=1e-6)
    # zoom at a cursor: the world point on the focal plane under the cursor stays there
    at = (900.0, 250.0)
    r, u, _ = st.frame()
    hh = st.half_height()
    world = (np.asarray(st.focal_point) + (2 * at[0] / w - 1) * hh * (w / h) * r + (1 - 2 * at[1] / h) * hh * u)
    z = cam.zoom(st, 1.7, at=at, size=(w, h))
    assert math.isclose(z.distance, st.distance / 1.7, rel_tol=1e-9)
    assert np.allclose(cam.project(z, [world], w, h)[0], at, atol=1e-6)
    assert cam.zoom(st, 0.0) == st                  # non-positive factor is ignored


def test_parallel_projection_fit_and_project():
    d, up = cam.preset("side")
    st = cam.fit(cam.CameraState(parallel=True), BOX, 2.0, direction=d, view_up=up)
    px = cam.project(st, cam._corners(BOX), 2000.0, 1000.0)
    assert px[:, 0].min() >= 0 and px[:, 0].max() <= 2000.0
    assert px[:, 1].min() >= 0 and px[:, 1].max() <= 1000.0
    z = cam.zoom(st, 2.0)
    assert math.isclose(z.parallel_scale, st.parallel_scale / 2)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def test_mesh_arrays_accepts_model_types_and_drops_bad_faces():
    v, f = uv_sphere(1.0, 8, 16)
    m = data.mesh_arrays(Mesh3D("s", v, np.vstack([f, [[0, 1, 10_000]]])))
    assert len(m.faces) == len(f) and m.name == "s" and m.fields == {}
    tau = np.tile([0.0, 3.0, 4.0], (len(v), 1))
    s = data.mesh_arrays(SurfaceData3D("b", v, f, cp=np.zeros(len(v)), tau=tau))
    assert set(s.fields) == {"cp", "tau"} and np.allclose(s.fields["tau"], 5.0)
    t = data.mesh_arrays((v, f))
    assert t.vertices.shape == v.shape


def test_mirrored_volume_reflects_and_flips_v():
    half = sphere_flow_volume(n=(32, 20, 20), mirrored=True)
    full = sphere_flow_volume(n=(32, 20, 20))
    o, sp, f, m = data.full_volume(half)
    o2, sp2, f2, m2 = data.full_volume(full)
    assert np.allclose(o, o2) and np.allclose(sp, sp2)
    for k in ("u", "v", "w", "speed"):
        assert np.allclose(f[k], f2[k], atol=1e-5), k
    assert np.array_equal(m > 0.5, m2 > 0.5)


def test_slice_from_volume_orientation_and_extent():
    vol = vortex_pair_volume(n=(40, 24, 20), lo=(-1.0, -1.2, -1.0), hi=(3.0, 1.2, 1.0))
    sl = data.slice_from_volume(vol, "x", 1.0, "q")
    assert sl.fields["q"].shape == (20, 24)                    # rows z, cols y
    assert sl.extent == pytest.approx((-1.2, 1.2, -1.0, 1.0))
    # the two cores (Q max) sit at y = +-0.45, z = 0 -> columns either side of the middle, middle rows
    q = sl.fields["q"]
    r, c = np.unravel_index(np.argmax(q), q.shape)
    ycol = -1.2 + (c + 0.5) * 0.1
    assert abs(abs(ycol) - 0.45) < 0.11 and abs(-1.0 + (r + 0.5) * 0.1) < 0.11
    sy = data.slice_from_volume(vol, "y", None, "speed")
    assert sy.fields["speed"].shape == (20, 40) and sy.position == pytest.approx(0.05, abs=0.051)
    sz = data.slice_from_volume(vol, "z", 99.0, "u")          # clamped into the block
    assert sz.fields["u"].shape == (24, 40) and sz.position == pytest.approx(0.95)
    with pytest.raises(ValueError):
        data.slice_from_volume(vol, "r", 0.0, "u")
    with pytest.raises(KeyError):
        data.slice_from_volume(vol, "x", 0.0, "pressure")


def test_slice_from_volume_interpolates_linearly_between_layers():
    vol = vortex_pair_volume(n=(10, 6, 6))
    vol.fields["u"] = np.broadcast_to(np.arange(10, dtype=np.float32), (6, 6, 10)).copy()
    x0, dx = vol.origin[0], vol.spacing[0]
    sl = data.slice_from_volume(vol, "x", x0 + 3.75 * dx, "u")     # between centres 3 and 4
    assert np.allclose(sl.fields["u"], 3.25)


def test_slice_corners_follow_slicedata_rules():
    c = data.slice_corners("y", 0.3, (-1.0, 2.0, 0.0, 1.5))
    assert np.allclose(c, [[-1, 0.3, 0], [2, 0.3, 0], [2, 0.3, 1.5], [-1, 0.3, 1.5]])
    c = data.slice_corners("x", 4.0, (-1.0, 1.0, 0.0, 2.0))
    assert np.allclose(c[2], [4.0, 1.0, 2.0])
    c = data.slice_corners("z", 0.5, (0.0, 3.0, -1.0, 1.0))
    assert np.allclose(c[1], [3.0, -1.0, 0.5])


def test_rake_seeds_sit_upstream_inside_the_volume():
    fb = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    vb = (-1.5, 2.5, -1.0, 1.0, -1.0, 1.0)
    for kind in ("rake", "plane", "vertical"):
        p = rake_seeds(fb, vb, kind)
        assert np.all(p[:, 0] < fb[0]) and np.all(p[:, 0] > vb[0])
        assert np.all((p[:, 1] > vb[2]) & (p[:, 1] < vb[3]) & (p[:, 2] > vb[4]) & (p[:, 2] < vb[5]))
    assert len(rake_seeds(fb, vb, "rake", 12)) == 12
    assert 100 <= len(rake_seeds(fb, vb, "plane", 120)) <= 140
    with pytest.raises(ValueError):
        rake_seeds(fb, vb, "cloud")


# ---------------------------------------------------------------------------
# overlay layout
# ---------------------------------------------------------------------------

def test_tick_labels_round_values_with_true_minus():
    t = overlay.tick_labels(-1.25, 1.25)
    vals = [v for v, _ in t]
    assert vals[0] >= -1.25 and vals[-1] <= 1.25 and 0.0 in vals
    assert all("−" in s for v, s in t if v < 0)
    assert len({len(s.lstrip("−")) for v, s in t}) == 1           # common decimals


def test_colorbar_layout_inside_the_image_and_scales_with_dpr():
    spec = overlay.ColorbarSpec("coolwarm", -1.0, 1.0, "Pressure coefficient Cp", "", (False, True))
    lay = overlay.colorbar_layout(spec, 1600, 900, 1.0)
    x0, y0, x1, y1 = lay["bar"]
    assert 1600 - lay["strip"] <= x0 < x1 < 1600 and 0 < y0 < y1 < 900
    assert lay["extend"] and lay["extend"][0][0] == "above"
    assert all(y0 - 0.5 <= y <= y1 + 0.5 for y, _ in lay["ticks"])
    lay2 = overlay.colorbar_layout(spec, 3200, 1800, 2.0)
    assert lay2["strip"] == pytest.approx(2 * lay["strip"])
    assert overlay.colorbar_strip_width(None) == 0.0


def test_triad_segments_project_axes_and_sort_far_first():
    d, up = cam.preset("side")
    st = cam.fit(cam.CameraState(), BOX, 1.5, direction=d, view_up=up)
    segs = overlay.triad_segments(*st.frame(), origin=(50, 50), length=30)
    by = {name: (a, b) for name, a, b, _ in segs}
    assert np.allclose(by["x"][1] - by["x"][0], [30, 0])       # x to the right
    assert np.allclose(by["z"][1] - by["z"][0], [0, 30])       # z up (display y up)
    assert np.allclose(by["y"][1] - by["y"][0], [0, 0])        # y points into the screen
    assert segs[0][0] == "y"                                   # away from the viewer: drawn first


def test_slice_data_is_the_model_type():
    sd = data.slice_from_volume(vortex_pair_volume(n=(8, 6, 6)), "z", 0.0, "u")
    assert isinstance(sd, SliceData)
