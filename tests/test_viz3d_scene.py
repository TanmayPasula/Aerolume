"""viz3d.Scene off-screen rendering: pixels in the right places, cameras, layers, timings.

Every check is made on rendered pixels or on the geometry VTK generated, using
the analytic fields of ``test_viz3d_fixtures``. The whole module is skipped
when no off-screen OpenGL context can be created (``backend_info()``), which
is exactly the case where the app shows "3D view unavailable" instead.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from test_viz3d_fixtures import (
    gradient_slice,
    sphere_flow_volume,
    sphere_surface,
    uv_sphere,
    vehicle_meshes,
    vortex_pair_volume,
)

from aerolume.model import Mesh3D
from aerolume.viz import colormaps as cm2d
from aerolume.viz3d import overlay
from aerolume.viz3d import scene as scene_mod
from aerolume.viz3d.scene import Scene, backend_info

pytest.importorskip("vtkmodules")
_INFO = backend_info()
pytestmark = pytest.mark.skipif(not _INFO["ok"], reason=f"no off-screen OpenGL: {_INFO['error']}")

WHITE = np.array([255, 255, 255])


@pytest.fixture
def make_scene():
    made = []

    def make(*a, **kw):
        s = Scene(*a, **kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def non_background(img: np.ndarray, bg=WHITE, tol: int = 6) -> np.ndarray:
    return np.abs(img.astype(int) - np.asarray(bg)).max(axis=2) > tol


def pixel(img, xy):
    x, y = (round(c) for c in xy)
    return img[y, x].astype(int)


def test_backend_is_hardware_opengl():
    assert _INFO["ok"] and _INFO["window"]
    assert _INFO["renderer"], "the capability report names the GL renderer"


def test_sphere_render_is_a_shaded_neutral_body_in_the_middle(make_scene):
    s = make_scene("light")
    v, f = uv_sphere(1.0, 32, 64)
    s.add_mesh(Mesh3D("ball", v, f))
    s.set_camera("iso")
    img = s.render(640, 480)
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    body = non_background(img)
    body[380:, :120] = False                          # the axes triad
    frac = body.mean()
    assert 0.15 < frac < 0.6
    ys, xs = np.nonzero(body)
    assert abs(xs.mean() - 320) < 12 and abs(ys.mean() - 240) < 12
    c = pixel(img, s.project([[0.0, 0.0, 0.0]])[0])
    assert c.max() - c.min() < 14                      # neutral grey, no colour cast from the lights
    assert 120 < c.mean() < 245
    assert np.array_equal(img[5, 320], WHITE)          # light theme background = surface white
    # shading: the lit upper-left is brighter than the lower-right limb
    lum = img.astype(float).mean(axis=2)
    assert lum[body].std() > 8


def test_cp_colours_where_potential_flow_puts_them(make_scene):
    s = make_scene("light")
    s.add_mesh(sphere_surface(1.0, 64, 128), "cp")
    s.set_camera("front")                              # look at the nose (stagnation point)
    img = s.render(600, 600)
    nose = pixel(img, s.project([[-0.5, 0.0, 0.0]])[0])
    assert nose[0] > nose[2] + 60                      # Cp = +1: red end of coolwarm
    s.set_camera("side")
    img = s.render(600, 600)
    top = pixel(img, s.project([[0.0, 0.0, 0.48]])[0])   # equator, Cp = -1.25: blue end
    assert top[2] > top[0] + 40
    info = s.scalar_info(next(iter(s.layers())))
    assert info["vmin"] == -info["vmax"] and info["cmap"] == "coolwarm"


def test_slice_pixels_are_the_exact_lut_colours(make_scene):
    s = make_scene("light", axes=False)
    sd = gradient_slice("y", 0.0, shape=(40, 80), extent=(-2.0, 2.0, 0.0, 2.0))
    sd.mask = np.zeros((40, 80))
    sd.mask[:, 36:44] = 1.0                            # a solid band in the middle
    s.add_slice(sd, field="u", cmap="viridis", range=(-2.0, 2.0))
    s.set_camera("side")
    img = s.render(800, 500)
    lut = cm2d.COLORMAPS["viridis"]
    for x in (-1.5, -0.8, 0.9, 1.6):
        got = pixel(img, s.project([[x, 0.0, 1.0]])[0])
        want = lut[round((x + 2.0) / 4.0 * 255)]
        assert np.abs(got - want).max() <= 4, (x, got, want)     # unlit: exact colours
    solid = pixel(img, s.project([[0.0, 0.0, 1.0]])[0])
    assert np.abs(solid - [143, 143, 143]).max() <= 3     # SceneTheme.solid, the v4 neutral #8f8f8f


def test_colorbar_shows_the_lut_and_reserves_the_right_strip(make_scene):
    s = make_scene("light")
    s.add_mesh(sphere_surface(1.0, 32, 64), "cp")
    s.add_colorbar()
    W, H = 1200, 600
    img = s.render(W, H)
    spec = s.colorbar_spec()
    assert spec.title == "Pressure coefficient Cp" and spec.units == ""
    lay = overlay.colorbar_layout(spec, W, H)
    x0, y0, x1, y1 = lay["bar"]
    xc = int((x0 + x1) / 2)
    lut = cm2d.COLORMAPS["coolwarm"]
    for f in (0.1, 0.5, 0.9):
        row = round(H - (y0 + f * (y1 - y0)))
        assert np.abs(img[row, xc].astype(int) - lut[int(f * 255)]).max() <= 6
    # the model is centred in the area left of the bar
    strip = s._strip_px(W)
    cx, cy = s.project([[0.0, 0.0, 0.0]])[0]
    assert abs(cx - (W - strip) / 2) < 1.0 and abs(cy - H / 2) < 1.0
    body = non_background(img)
    body[:, W - strip:] = False
    body[H - 110:, :110] = False
    _, xs = np.nonzero(body)
    assert abs(xs.mean() - (W - strip) / 2) < 10


def test_camera_presets_give_different_pictures(make_scene):
    s = make_scene("light")
    s.add_meshes(vehicle_meshes())
    s.add_ground(0.0)
    imgs = {}
    for p in ("iso", "front", "side", "top", "back"):
        s.set_camera(p)
        imgs[p] = s.render(400, 300).astype(int)
    names = list(imgs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if {names[i], names[j]} == {"front", "back"}:
                continue        # this test body is fore-aft symmetric: those two views coincide
            d = np.abs(imgs[names[i]] - imgs[names[j]]).mean()
            assert d > 3.0, (names[i], names[j], d)
    # front and back see the same silhouette (mirrored left-right)
    fr = non_background(imgs["front"].astype(np.uint8), tol=40)
    bk = non_background(imgs["back"].astype(np.uint8), tol=40)
    assert abs(fr.mean() - bk.mean()) < 0.02


def test_orbit_pan_zoom_and_fit(make_scene):
    s = make_scene("light")
    s.add_meshes(vehicle_meshes())
    s.set_camera("iso")
    base = s.render(500, 350)
    rev = s.revision
    s.orbit(25, 10)
    assert s.revision > rev
    orbited = s.render(500, 350)
    assert np.abs(orbited.astype(int) - base).mean() > 2
    fp = s.camera.focal_point                  # a point on the focal plane follows the cursor exactly
    p_before = s.project([fp])[0]
    s.pan(30, -20)
    p_after = s.project([fp])[0]
    assert np.allclose(p_after - p_before, [30, -20], atol=1e-6)
    d0 = s.camera.distance
    s.zoom(2.0, at=(100, 100))
    assert s.camera.distance == pytest.approx(d0 / 2)
    s.fit()
    s.render(500, 350)
    pts = s.project(s.focus_points())
    assert pts[:, 0].min() > 0 and pts[:, 0].max() < 500 and pts[:, 1].min() > 0 and pts[:, 1].max() < 350


def test_auto_fit_follows_the_aspect_until_the_user_moves(make_scene):
    s = make_scene("light")
    s.add_meshes(vehicle_meshes())
    s.set_camera("side")
    s.render(400, 400)
    narrow = s.camera.distance
    s.render(1600, 400)
    assert s.camera.distance < narrow                     # re-fitted for the wide view
    s.orbit(5, 0)
    moved = s.camera
    s.render(400, 400)
    assert s.camera == moved                              # user camera is kept


def test_streamlines_spiral_around_the_vortex_core(make_scene):
    vol = vortex_pair_volume()
    s = make_scene("light")
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    seeds = np.stack([np.full(8, -0.9), 0.45 + 0.2 * np.cos(ang), 0.2 * np.sin(ang)], 1)
    h = s.add_streamlines(vol, seeds, direction="forward", color_by="speed")
    lines = s.layer_lines(h)
    assert len(lines) == 8
    turns = []
    for ln in lines:
        assert ln[:, 0].max() > 2.8                          # carried through the whole block
        a = np.unwrap(np.arctan2(ln[:, 2], ln[:, 1] - 0.45))
        turns.append(a[-1] - a[0])
        r = np.hypot(ln[:, 1] - 0.45, ln[:, 2])
        assert 0.12 < r.min() and r.max() < 0.4              # bound to its own core (the partner distorts it)
    # counter-clockwise about +x as the core's sign says; ~6.7 m/s swirl at r = 0.2 over 0.39 s is
    # ~13 rad, less on the side facing the partner vortex, whose field distorts the orbits
    assert min(turns) > 4.0 and np.mean(turns) > 2 * np.pi
    s.add_colorbar()
    img = s.render(800, 500)
    assert non_background(img).mean() > 0.01
    assert s.colorbar_spec().title == "Speed |V|"


def test_rake_streamlines_go_around_the_sphere_not_through_it(make_scene):
    vol = sphere_flow_volume(n=(80, 50, 50))
    s = make_scene("light")
    s.add_mesh(sphere_surface(1.0, 32, 64))
    h = s.add_streamlines(vol, "plane", n=60)
    pts = s.layer_points(h)
    assert len(pts) > 1000
    r = np.linalg.norm(pts, axis=1)
    assert np.mean(r < 0.45) < 0.01                          # (almost) nothing inside the body


def test_isosurface_of_q_wraps_both_cores(make_scene):
    vol = vortex_pair_volume()
    s = make_scene("light")
    h = s.add_isosurface(vol, "q")
    pts = s.layer_points(h)
    assert len(pts) > 200
    d = np.minimum(np.hypot(pts[:, 1] - 0.45, pts[:, 2]), np.hypot(pts[:, 1] + 0.45, pts[:, 2]))
    assert d.max() < 0.24                                    # within 2 core radii
    assert (pts[:, 1] > 0).any() and (pts[:, 1] < 0).any()
    s.set_camera("iso")
    img = s.render(800, 500)
    bluish = (img[..., 2].astype(int) - img[..., 0] > 15) & non_background(img)
    assert bluish.mean() > 0.01                              # the uniform muted-blue surface is visible


def test_isosurface_coloured_by_another_field_drives_the_colorbar(make_scene):
    vol = vortex_pair_volume()
    s = make_scene("dark")
    h = s.add_isosurface(vol, "q", color_by="speed")
    s.add_slice(vol, "x", 2.0, "vorticity")
    s.add_colorbar(layer=h)
    spec = s.colorbar_spec()
    assert spec.title == "Speed |V|" and spec.units == "m/s" and spec.vmin == 0.0
    s.add_colorbar()                                         # default: the newest mapped layer
    assert s.colorbar_spec().title == "Vorticity |ω|"


def test_a_second_colour_map_gets_its_own_bar_left_of_the_first(make_scene):
    """Two colour maps on screen (an iso-surface coloured by speed, a vorticity slice) draw two bars,
    each with its own gradient; the second sits left of the first and the view shrinks by both."""
    vol = vortex_pair_volume()
    s = make_scene("light")
    h_iso = s.add_isosurface(vol, "q", color_by="speed")
    h_sl = s.add_slice(vol, "x", 2.0, "vorticity")
    s.add_colorbar(layer=h_sl)
    s.set_camera("iso")
    W, H = 900, 520
    s.render(W, H)
    one_w = s.view_size()[0]
    s.add_colorbar(layer=h_iso, second=True)
    specs = s.colorbar_specs()
    assert [sp.title for sp in specs] == ["Vorticity |ω|", "Speed |V|"]
    img = s.render(W, H)
    first = overlay.colorbar_layout(specs[0], W, H)
    second = overlay.colorbar_layout(specs[1], W, H, right=first["strip"])
    assert second["bar"][2] < first["bar"][0] - 8                 # left of the first, plates apart
    for spec, lay in ((specs[0], first), (specs[1], second)):
        x0, y0, x1, y1 = lay["bar"]
        xc = 0.5 * (x0 + x1)
        lut = cm2d.get_lut(spec.cmap).astype(int)
        for frac in (0.15, 0.85):                                    # VTK y is up, image rows down
            got = pixel(img, (xc, H - (y0 + frac * (y1 - y0))))
            want = lut[round(frac * (len(lut) - 1))]
            assert np.abs(got - want).max() <= 12, (spec.title, frac, got, want)
    assert s.view_size()[0] == pytest.approx(one_w - second["strip"], abs=2)   # the model moves left
    # the second bar goes with its layer, and is never a copy of the first
    s.set_visible(h_iso, False)
    assert [sp.title for sp in s.colorbar_specs()] == ["Vorticity |ω|"]
    s.set_visible(h_iso, True)
    s.add_colorbar(layer=h_sl, second=True)
    assert len(s.colorbar_specs()) == 1
    s.add_colorbar(layer=h_iso, second=True)
    s.remove(h_iso)
    assert len(s.colorbar_specs()) == 1
    s.remove_colorbar()
    assert s.colorbar_specs() == []


def test_mirrored_volume_draws_like_the_full_one(make_scene):
    half = sphere_flow_volume(n=(48, 32, 32), mirrored=True)
    full = sphere_flow_volume(n=(48, 32, 32))
    imgs = []
    for vol in (half, full):
        s = make_scene("light")
        s.add_slice(vol, "x", 0.6, "v", range=(-3, 3))
        s.set_camera("front")
        imgs.append(s.render(500, 400).astype(int))
    assert np.abs(imgs[0] - imgs[1]).mean() < 0.5


def test_theme_switch_and_edges(make_scene):
    s = make_scene("light")
    v, f = uv_sphere(1.0, 12, 24)
    s.add_mesh(Mesh3D("ball", v, f))
    s.add_ground(-0.5)
    s.set_camera("iso")
    light = s.render(400, 300)
    s.set_edges(True)
    edged = s.render(400, 300)
    assert np.abs(edged.astype(int) - light).mean() > 0.5
    s.set_theme("dark")
    dark = s.render(400, 300)
    assert np.array_equal(dark[3, 200], [26, 26, 26])          # tokens()["surface"] of the dark theme (v4)
    assert dark.mean() < light.mean() - 40


def test_hidpi_render_matches_the_1x_render_downsampled(make_scene):
    s = make_scene("light")
    s.add_mesh(sphere_surface(1.0, 32, 64), "cp")
    s.add_colorbar()
    s.set_camera("iso")
    one = s.render(400, 300, scale=1.0).astype(float)
    two = s.render(800, 600, scale=2.0).astype(float)
    down = two.reshape(300, 2, 400, 2, 3).mean(axis=(1, 3))
    assert np.abs(down - one).mean() < 6.0


def test_layer_management(make_scene):
    s = make_scene("light", axes=False)
    v, f = uv_sphere(1.0, 16, 32)
    h = s.add_mesh(Mesh3D("ball", v, f), np.linspace(0, 1, len(v)), cmap="viridis", range=(0, 1))
    g = s.add_ground(-0.5)
    assert s.layers() == {h: "mesh", g: "ground"}
    s.set_camera("side")
    a = s.render(300, 200).astype(int)
    s.set_cmap(h, "magma")
    s.set_scalar_range(h, 0.4, 0.6)
    assert s.scalar_info(h) == {"field": None, "cmap": "magma", "vmin": 0.4, "vmax": 0.6}
    b = s.render(300, 200).astype(int)
    assert np.abs(a - b).mean() > 1
    s.set_visible(h, False)
    s.set_visible(g, False)
    blank = s.render(300, 200)
    assert not non_background(blank).any()
    s.remove(h)
    s.clear()
    assert s.layers() == {}
    with pytest.raises(KeyError):
        s.add_mesh(sphere_surface(1.0, 8, 16), "sensitivity")
    with pytest.raises(ValueError):
        s.add_mesh(Mesh3D("ball", v, f), np.zeros(3))


def test_open_mesh_with_mixed_winding_renders(make_scene):
    v, f = uv_sphere(1.0, 24, 48)
    keep = v[f].mean(axis=1)[:, 2] < 0.3                     # cut the cap off: not watertight
    f = f[keep].copy()
    f[::2] = f[::2, ::-1]                                    # flip every other triangle
    s = make_scene("light")
    s.add_mesh(Mesh3D("open", v, f))
    s.set_camera("iso")
    img = s.render(400, 300)
    body = non_background(img)
    body[200:, :90] = False
    assert body.mean() > 0.1
    # two-sided lighting: the inside seen through the hole is lit, not black
    assert img[body].astype(int).mean() > 80


def test_render_time_200k_triangles_1600x900(make_scene):
    if any(k in _INFO["renderer"].lower() for k in ("llvmpipe", "softpipe", "basic render")):
        pytest.skip("software OpenGL")
    s = make_scene("light")
    surf = sphere_surface(1.0, 200, 500)
    assert len(surf.faces) >= 199_000
    s.add_mesh(surf, "cp")
    s.add_colorbar()
    s.set_camera("iso")
    s.render(1600, 900)
    times = []
    for _ in range(12):
        s.orbit(2.0, 0.0)
        t0 = time.perf_counter()
        s.render(1600, 900)
        times.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(times))
    print(f"render 200k tris 1600x900: median {med:.1f} ms, max {max(times):.1f} ms ({_INFO['renderer']})")
    assert med < 30.0


def test_unavailable_backend_degrades_to_a_blank_image(monkeypatch):
    monkeypatch.setattr(scene_mod, "_BACKEND", {"ok": False, "window": "", "renderer": "", "version": "",
                                                "error": "no OpenGL 3.2 context"})
    s = Scene("dark")
    assert not s.available and "OpenGL" in s.error
    v, f = uv_sphere(1.0, 8, 16)
    h = s.add_mesh(Mesh3D("ball", v, f))
    s.add_colorbar()
    s.orbit(10, 5)
    img = s.render(120, 80)
    assert img.shape == (80, 120, 3) and np.all(img == [26, 26, 26])
    assert s.layer_points(h) is None
    s.close()


def test_translucent_mesh_shows_what_is_inside(make_scene):
    s = make_scene("light", axes=False)
    v, f = uv_sphere(0.5, 24, 48)
    s.add_mesh(Mesh3D("core", v, f), color="#c0392b")
    v2, f2 = uv_sphere(1.2, 32, 64)
    s.add_mesh(Mesh3D("shell", v2, f2), opacity=0.35)
    s.set_camera("front")
    img = s.render(400, 400)
    centre = pixel(img, s.project([[-0.25, 0.0, 0.0]])[0])
    assert centre[0] > centre[2] + 40                  # the red core is seen through the grey shell
    rim = pixel(img, s.project([[-0.2, 0.0, 0.52]])[0])
    assert abs(rim[0] - rim[2]) < 25 and rim.mean() < 250   # outside the core: only the faint shell


# ----------------------------------------------------------------------------
# Q iso-surface: the near-wall skin is cut along a smooth distance, not voxels
# ----------------------------------------------------------------------------

def _skin_and_tube_volume(n=(80, 50, 50)):
    """A sphere (d = 1) whose Q has a wall skin (r < 0.62) and a vortex tube that touches the body.

    The skin is what the diffuse immersed wall produces on every surface; the
    tube (radius 0.15 along x at z = 0.45, from x = 0 on) is a real vortex
    starting on the wall, so its cut end shows the trim's rim.
    """
    from test_viz3d_fixtures import grid

    from aerolume.model import VolumeData
    xx, yy, zz, h = grid(n)
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    skin = np.exp(-((r - 0.5) / 0.06) ** 2) * (r > 0.5)
    tube = np.exp(-(yy ** 2 + (zz - 0.45) ** 2) / 0.15 ** 2) * (xx > 0.0) * (xx < 2.0)
    q = (100.0 * np.maximum(skin, tube)).astype(np.float32)
    solid = r < 0.5
    q[solid] = 0.0
    sp = np.full_like(q, 10.0)
    return VolumeData(origin=(-1.5, -1.0, -1.0), spacing=h, fields={"q": q, "speed": sp},
                      mask=solid.astype(np.float32)), h


def _boundary_points(pd) -> np.ndarray:
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkFeatureEdges
    fe = vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    out = fe.GetOutput()
    return vtk_to_numpy(out.GetPoints().GetData()).astype(float) if out.GetNumberOfPoints() else np.zeros((0, 3))


def test_q_isosurface_is_trimmed_smoothly_near_the_wall(make_scene):
    vol, h = _skin_and_tube_volume()
    cell = min(h)
    s = make_scene("light")
    s.add_mesh(sphere_surface(1.0, 48, 96))
    hq = s.add_isosurface(vol, "q", 30.0)
    lay = s._layers[hq]
    assert lay.extra["wall_distance"] == "mesh"
    assert lay.extra["wall_clearance"] == scene_mod.WALL_CLEARANCE_CELLS
    pts = s.layer_points(hq)
    dist = np.linalg.norm(pts, axis=1) - 0.5
    clearance = scene_mod.WALL_CLEARANCE_CELLS * cell
    assert len(pts) > 500                                         # the tube is still there
    assert dist.min() > clearance - 0.05 * cell                   # no skin, nothing closer than the cut
    rim = _boundary_points(lay.extra["polydata"])
    assert len(rim) > 12                                          # the tube is open where it was cut
    rd = np.linalg.norm(rim, axis=1) - 0.5
    # A smooth cut: every rim point lies on the offset surface. Masking whole voxels left a
    # saw-tooth rim whose points scatter over two cells of distance.
    assert np.abs(rd - clearance).max() < 0.05 * cell


def test_q_trim_falls_back_to_the_mask_and_can_be_switched_off(make_scene):
    vol, h = _skin_and_tube_volume()
    cell = min(h)
    s = make_scene("light")
    keep = s.add_isosurface(vol, "q", 30.0, wall_clearance=0.0)        # explicit: no trim
    near = np.linalg.norm(s.layer_points(keep), axis=1) - 0.5
    assert near.min() < 0.1                                            # the wall skin is there
    hq = s.add_isosurface(vol, "q", 30.0)                              # no mesh in the scene: the mask
    assert s._layers[hq].extra["wall_distance"] == "mask"
    d = np.linalg.norm(s.layer_points(hq), axis=1) - 0.5
    assert d.min() > scene_mod.WALL_CLEARANCE_CELLS * cell - 0.6 * cell
    sp = s.add_isosurface(vol, "speed", 5.0)                           # not a vortex field: untouched
    assert "wall_clearance" not in s._layers[sp].extra


def test_annotations_can_leave_out_the_colorbar_and_caption(make_scene):
    s = make_scene("light")
    h = s.add_mesh(sphere_surface(1.0), "cp")
    s.add_colorbar(layer=h)
    s.set_caption("t = 3 L/U")
    s.render(800, 500)
    assert s.view_size()[0] < 800                                      # the bar's strip is reserved
    rev = s.revision
    s.set_annotations(colorbar=False, caption=False)
    assert s.revision > rev
    rev = s.revision
    s.set_annotations(colorbar=False)                                  # no change, no new render
    assert s.revision == rev
    img = s.render(800, 500)
    assert s.view_size() == (800, 500)                                 # the model gets the full width
    assert s.colorbar_spec() is not None and s.caption() == "t = 3 L/U"   # still defined
    assert not non_background(img[:, 740:]).any() or non_background(img[:, 740:]).mean() < 0.01
    assert s.annotations() == {"colorbar": False, "caption": False}
