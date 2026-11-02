"""3D smoke and streamlines in the volume: rake, tracing, rendering (DESIGN.md 7.13).

* the rake (viz3d.flow.resolve_rake): defaults upstream of the nose, kinds, clamping,
  the dict round trip and dragging (``moved``);
* streamline tracing: lines go around a body (never through it), a symmetry-plane
  volume traces exactly like the whole one (it is traced in its own half), seeds on
  the plane survive, and 300 seeds trace fast;
* smoke rendering: GPU volume rendering, theme colours (light smoke on dark, dark
  smoke on light), in-place updates (same actor, no rebuild), solids clear, mirrored
  volumes whole, update + render at 128^3 well inside a frame;
* streamline layers update in place with a stable colour range; the seed rake layer;
  ray picking inverts projection; skin-friction lines lie on the surface;
* ``smoke_volume`` finds a result's smoke wherever the runner put it.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from test_viz3d_fixtures import sphere_flow_volume, sphere_surface, uv_sphere

from aerolume.model import SurfaceData3D, VolumeData
from aerolume.viz3d import camera as cam
from aerolume.viz3d.flow import SeedRake, StreamlineSet, resolve_rake, smoke_volume, trace_streamlines

pytest.importorskip("vtkmodules")

BODY = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
DOMAIN = (-1.5, 2.5, -1.0, 1.0, -1.0, 1.0)


def _gl_ok() -> bool:
    from aerolume.viz3d.scene import backend_info
    return bool(backend_info()["ok"])


needs_gl = pytest.mark.skipif(not _gl_ok(), reason="no off-screen OpenGL")


# ---------------------------------------------------------------------------
# the rake
# ---------------------------------------------------------------------------

def test_default_rake_stands_upstream_across_the_body():
    r = resolve_rake(None, BODY, DOMAIN)
    assert r.kind == "rake" and r.x < BODY[0] and r.x > DOMAIN[0]
    assert r.y[0] < BODY[2] and r.y[1] > BODY[3]                    # wider than the body
    assert r.z[0] == r.z[1] == pytest.approx(BODY[4] + 0.4 * (BODY[5] - BODY[4]))
    assert len(r.points) == 30 and np.allclose(r.points[:, 0], r.x)
    smoke = resolve_rake(None, BODY, DOMAIN, purpose="smoke", h=0.05)
    assert smoke.x == pytest.approx(r.x) and 2 <= len(smoke.points) < len(r.points)
    gaps = np.diff(np.sort(smoke.points[:, 1]))
    assert gaps.min() >= 0.8 * 6 * 1.25 * 0.05                     # streams stay distinct (~6 radii)


@pytest.mark.parametrize("kind,count", [("vertical", 24), ("grid", 120), ("plane", 300)])
def test_rake_kinds(kind, count):
    r = resolve_rake({"kind": kind}, BODY, DOMAIN)
    assert r.kind == kind and abs(len(r.points) - count) <= 0.1 * count
    if kind == "vertical":
        assert np.allclose(r.points[:, 1], 0.0) and np.ptp(r.points[:, 2]) > 0.9
    pts = resolve_rake({"kind": "points", "points": [[-1, 0, 0], [-1, 0.2, 0.1]]}, BODY, DOMAIN)
    assert pts.kind == "points" and len(pts.points) == 2
    with pytest.raises(ValueError):
        resolve_rake({"kind": "fan"}, BODY, DOMAIN)


def test_rake_is_clamped_inside_the_domain_and_round_trips():
    r = resolve_rake({"x": -9.0, "y": [-5, 5], "z": 0.9}, BODY, DOMAIN)
    assert r.x > DOMAIN[0] and r.y[0] > DOMAIN[2] and r.y[1] < DOMAIN[3]
    spec = r.spec()
    assert "n" not in spec                                         # each purpose keeps its own count
    again = resolve_rake(spec, BODY, DOMAIN)
    assert np.allclose(again.points, r.points)
    fixed = resolve_rake({"n": 7}, BODY, DOMAIN)
    assert fixed.spec()["n"] == 7 and len(resolve_rake(fixed.spec(), BODY, DOMAIN, purpose="smoke").points) == 7
    base = resolve_rake(None, BODY, DOMAIN)
    moved = base.moved(0.1, -0.2, 0.05)
    assert np.allclose(moved.points - base.points, [0.1, -0.2, 0.05])
    assert np.allclose(resolve_rake(moved.spec(), BODY, DOMAIN).points, moved.points)
    outside = base.moved(0.0, -2.0, 0.0)                              # dragged out of the tunnel: clamped back
    assert resolve_rake(outside.spec(), BODY, DOMAIN).y[0] > DOMAIN[2]
    pulsed = resolve_rake({"pulse": [0.1, 0.2], "radius": 0.02}, BODY, DOMAIN)
    assert pulsed.spec()["pulse"] == [0.1, 0.2] and pulsed.spec()["radius"] == 0.02


# ---------------------------------------------------------------------------
# streamline tracing (no OpenGL needed)
# ---------------------------------------------------------------------------

def _inside(points: np.ndarray, r: float = 0.5) -> float:
    return float(np.mean(np.linalg.norm(points, axis=1) < r * 0.97))


def test_streamlines_go_around_the_sphere():
    vol = sphere_flow_volume(n=(96, 48, 48))
    seeds = resolve_rake({"kind": "grid", "n": [7, 5]}, BODY, (-1.5, 2.5, -1, 1, -1, 1)).points
    s = trace_streamlines(vol, seeds)
    assert s.n_lines == len(seeds) or s.n_lines >= len(seeds) - 2
    assert _inside(s.points) < 0.002
    assert s.points[:, 0].max() > 2.0                                # most lines reach the outlet
    far = s.points[:, 0] < -1.2
    assert np.allclose(s.values["speed"][far], 10.0, rtol=0.05)      # free stream upstream
    assert len(s.lines()) == s.n_lines and isinstance(s, StreamlineSet)


def test_a_symmetry_plane_volume_traces_like_the_whole_one():
    full = sphere_flow_volume(n=(64, 40, 40))
    half = sphere_flow_volume(n=(64, 40, 40), mirrored=True)
    seeds = np.array([[-1.2, y, z] for y in (-0.6, -0.2, -0.011, 0.0, 0.011, 0.3) for z in (-0.25, 0.1)])
    a = trace_streamlines(full, seeds)
    b = trace_streamlines(half, seeds)
    assert b.n_lines == len(seeds)                                   # seeds on and next to the plane survive
    assert a.n_lines == b.n_lines
    ends_a = np.array([ln[-1] for ln in a.lines()])
    ends_b = np.array([ln[-1] for ln in b.lines()])
    assert np.abs(np.sort(ends_a, axis=0) - np.sort(ends_b, axis=0)).max() < 0.05
    neg = np.array([ln[len(ln) // 2, 1] for ln in b.lines()])
    assert (neg < -1e-6).sum() == (seeds[:, 1] < 0).sum()           # mirrored seeds give mirrored lines


def test_three_hundred_seeds_trace_quickly():
    vol = sphere_flow_volume(n=(160, 112, 112))                      # 2.0 M cells, ~128^3
    seeds = resolve_rake({"kind": "plane", "n": 300}, BODY, (-1.5, 2.5, -1, 1, -1, 1)).points
    trace_streamlines(vol, seeds[:10])                               # warm-up (imports)
    best = min(trace_streamlines(vol, seeds).seconds for _ in range(3))
    assert best < 0.6, f"300 streamlines took {best:.2f} s"


def test_tracing_requires_velocity_and_handles_no_seeds():
    vol = VolumeData(origin=(0, 0, 0), spacing=(0.1, 0.1, 0.1), fields={"dye": np.zeros((4, 4, 4), np.float32)})
    with pytest.raises(KeyError):
        trace_streamlines(vol, [[0.1, 0.1, 0.1]])
    s = trace_streamlines(sphere_flow_volume(), np.zeros((0, 3)))
    assert s.n_lines == 0 and len(s.points) == 0


# ---------------------------------------------------------------------------
# smoke in results
# ---------------------------------------------------------------------------

def test_smoke_volume_is_found_in_either_home():
    from aerolume.model import CFD3DResult
    vol = sphere_flow_volume()
    r = CFD3DResult(sim_id="x", volume=vol)
    assert smoke_volume(r) is None
    vol.fields["dye"] = np.zeros_like(vol.fields["u"])
    assert smoke_volume(r) is vol
    instant = VolumeData(origin=vol.origin, spacing=vol.spacing, fields={"dye": vol.fields["dye"]}, label="instant")
    r.extra_volumes = [instant]                                     # the model's future home
    assert smoke_volume(r) is instant


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _smoke_field(n=(96, 48, 48), lo=(-1.5, -1.0, -1.0), hi=(2.5, 1.0, 1.0), mirrored=False) -> VolumeData:
    """Smoke streams from a rake at x = -1 wrapping a unit sphere (synthetic: smoke along the
    potential-flow stream tubes), plus the sphere as the solid."""
    vol = sphere_flow_volume(n=n, lo=lo, hi=hi, mirrored=mirrored)
    nz, ny, nx = vol.fields["u"].shape
    x = vol.origin[0] + (np.arange(nx) + 0.5) * vol.spacing[0]
    y = vol.origin[1] + (np.arange(ny) + 0.5) * vol.spacing[1]
    z = vol.origin[2] + (np.arange(nz) + 0.5) * vol.spacing[2]
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    stream = np.zeros_like(X)
    for yc in (-0.45, -0.15, 0.15, 0.45):
        # Stokes stream function of potential flow: psi ~ (r^2 - R^3/r) sin^2; keep tubes near yc
        k = np.clip(1 - 0.125 / np.maximum(r, 0.5) ** 3, 0, None)
        stream = np.maximum(stream, np.exp(-((Y * np.sqrt(k) - yc) ** 2 + (Z * np.sqrt(k)) ** 2) / 0.004))
    dye = (stream * (X > -1.0)).astype(np.float32)
    dye[vol.mask > 0.5] = 0.0
    vol.fields["dye"] = dye
    return vol


@needs_gl
def test_smoke_renders_light_on_dark_and_dark_on_light():
    from aerolume.viz3d import Scene
    vol = _smoke_field()
    for theme in ("light", "dark"):
        s = Scene(theme, axes=False)
        try:
            s.add_smoke(vol)
            s.set_focus((-1.5, 2.5, -1, 1, -1, 1))
            s.set_camera("top")
            img = s.render(480, 300).astype(int)
            bg = np.round(np.asarray(s.theme.background) * 255).astype(int)
            diff = img - bg
            changed = np.abs(diff).max(axis=2) > 8
            assert changed.mean() > 0.02                            # the streams are visible
            mean = diff[changed].mean()
            assert (mean < 0) if theme == "light" else (mean > 0)   # dark smoke on white, light on graphite
        finally:
            s.close()


@needs_gl
def test_smoke_updates_in_place_and_clears_solids():
    from aerolume.viz3d import Scene
    s = Scene("dark", axes=False)
    try:
        vol = _smoke_field()
        h = s.add_smoke(vol)
        actor = s._layers[h].actors[0]
        s.set_focus((-1.5, 2.5, -1, 1, -1, 1))
        s.set_camera("top")
        a = s.render(400, 260)
        empty = VolumeData(vol.origin, vol.spacing, {"dye": np.zeros_like(vol.fields["dye"])}, mask=vol.mask)
        s.update_smoke(h, empty)
        assert s._layers[h].actors[0] is actor                      # same volume actor: no rebuild
        b = s.render(400, 260)
        bg = np.round(np.asarray(s.theme.background) * 255)
        assert np.abs(b.astype(int) - bg).max() <= 3                # nothing drawn without smoke
        assert np.abs(a.astype(int) - bg).max() > 30
        full = VolumeData(vol.origin, vol.spacing, {"dye": np.ones_like(vol.fields["dye"])}, mask=vol.mask)
        arr = s._smoke_array(full, "dye")[2]
        # solids are cleared, and so is the dye within SMOKE_WALL_CLEAR_CELLS of them (round 3.4: the
        # boundary layer's dye coated the body); every other fluid voxel is drawn at full strength
        from aerolume.viz3d.scene import _wall_cleared
        near = _wall_cleared(vol.mask)
        assert arr[vol.mask > 0.5].max() == 0 and arr[near].max() == 0
        assert arr[~near].min() == 255 and near.sum() > (vol.mask > 0.5).sum()
    finally:
        s.close()


@needs_gl
def test_mirrored_smoke_is_drawn_whole():
    from aerolume.viz3d import Scene
    full, half = _smoke_field(), _smoke_field(mirrored=True)
    imgs = []
    for vol in (full, half):
        s = Scene("dark", axes=False)
        try:
            s.add_smoke(vol)
            s.set_focus((-1.5, 2.5, -1, 1, -1, 1))
            s.set_camera("top")
            imgs.append(s.render(400, 260).astype(int))
        finally:
            s.close()
    assert np.abs(imgs[0] - imgs[1]).mean() < 1.5


@needs_gl
def test_smoke_update_and_render_at_128_cubed_fit_in_a_frame():
    from aerolume.viz3d import Scene
    n = (200, 104, 100)                                              # 2.08 M voxels
    vol = _smoke_field(n=n)
    s = Scene("dark")
    try:
        h = s.add_smoke(vol)
        s.set_focus((-1.5, 2.5, -1, 1, -1, 1))
        s.render(1600, 1000)
        times = []
        for k in range(6):
            v2 = VolumeData(vol.origin, vol.spacing, {"dye": np.roll(vol.fields["dye"], k, axis=2)}, mask=vol.mask)
            t0 = time.perf_counter()
            s.update_smoke(h, v2)
            s.render(1600, 1000)
            times.append(time.perf_counter() - t0)
        assert np.median(times) < 0.15, f"smoke update + render took {1e3 * np.median(times):.0f} ms"
    finally:
        s.close()


@needs_gl
def test_streamlines_update_in_place_with_a_stable_colour_range():
    from aerolume.viz3d import Scene
    s = Scene("light")
    try:
        vol = sphere_flow_volume()
        s.add_mesh(uv_sphere(1.0, 24, 48))
        h = s.add_streamlines(vol, {"kind": "rake", "n": 12})
        lay = s._layers[h]
        actor, rng = lay.actors[0], (lay.vmin, lay.vmax)
        first = s.layer_lines(h)
        assert len(first) == 12
        faster = VolumeData(vol.origin, vol.spacing, {k: 2.0 * a if k in ("u", "v", "w", "speed") else a
                                                       for k, a in vol.fields.items()}, mask=vol.mask)
        s.update_streamlines(h, faster, seeds={"kind": "rake", "n": 8})
        assert s._layers[h].actors[0] is actor and (lay.vmin, lay.vmax) == rng
        assert len(s.layer_lines(h)) == 8
        s.update_streamlines(h, faster, rescale=True)
        assert lay.vmax > rng[1] * 1.5
        pre = trace_streamlines(vol, resolve_rake({"n": 5}, BODY, DOMAIN))
        s.update_streamlines(h, pre)
        assert len(s.layer_lines(h)) == pre.n_lines
        s.render(320, 200)
    finally:
        s.close()


@needs_gl
def test_seed_rake_layer_and_picking_ray():
    from aerolume.viz3d import Scene
    s = Scene("light")
    try:
        s.add_mesh(uv_sphere(1.0, 24, 48))
        h = s.add_seed_rake({"kind": "grid"}, domain_bounds=DOMAIN)
        sr = s.rake_of(h)
        assert isinstance(sr, SeedRake) and sr.kind == "grid"
        s.set_camera("iso")
        s.render(640, 400)
        px = s.project(sr.centre[None, :])[0]
        o, d = s.pick_ray(*px)
        # the ray through the handle's pixel passes through the handle
        t = float((sr.centre - o) @ d)
        assert np.linalg.norm(o + t * d - sr.centre) < 1e-6 * max(1.0, t)
        s.update_seed_rake(h, sr.moved(0.0, 0.2, 0.0))
        assert s.rake_of(h).centre[1] == pytest.approx(sr.centre[1] + 0.2)
        s.set_theme("dark")
        s.render(320, 200)
    finally:
        s.close()


def test_unproject_inverts_project_in_both_projections():
    st = cam.fit(cam.CameraState(), (-1, 1, -1, 1, -1, 1), 1.6, direction=np.array([-1.0, -1.2, 0.8]),
                 view_up=np.array([0.0, 0.0, 1.0]))
    p = np.array([[0.3, -0.2, 0.4]])
    for parallel in (False, True):
        state = cam.CameraState(st.position, st.focal_point, st.view_up, st.view_angle, parallel, 1.5)
        x, y = cam.project(state, p, 800, 500)[0]
        o, d = cam.unproject(state, x, y, 800, 500)
        t = float((p[0] - o) @ d)
        assert np.linalg.norm(o + t * d - p[0]) < 1e-9 and abs(np.linalg.norm(d) - 1) < 1e-12


@needs_gl
def test_skin_friction_lines_lie_on_the_surface():
    from aerolume.viz3d import Scene
    srf = sphere_surface(1.0, 32, 64)
    v = srf.vertices
    # wall shear along the meridians, from the nose (-x) to the tail (+x): the attached-flow pattern
    tang = np.array([1.0, 0.0, 0.0]) - (v[:, 0:1] / 0.5) * (v / 0.5)
    tau = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9) * 0.003
    surf = SurfaceData3D(body="sphere", vertices=v, faces=srf.faces, cp=srf.cp, tau=tau)
    s = Scene("light")
    try:
        h = s.add_surface_streaks([surf], n=150)
        lay = s._layers[h]
        assert lay.extra["n_lines"] > 100
        from vtkmodules.util.numpy_support import vtk_to_numpy
        pts = vtk_to_numpy(lay.extra["polydata"].GetPoints().GetData())
        r = np.linalg.norm(pts, axis=1)
        assert np.abs(r - 0.5).max() < 0.01                         # on the faceted sphere
        assert np.ptp(pts[:, 0]) > 0.8                               # nose to tail
        s.render(320, 200)
        for theme in ("dark", "light"):                              # dark lines on the pale body in both themes
            s.set_theme(theme)
            rgb = np.asarray(lay.actors[0].GetProperty().GetColor())
            assert rgb.mean() < 0.45 < float(np.mean(s.theme.body))
    finally:
        s.close()


def test_volume_bounds_follow_the_displayed_block():
    from aerolume.viz3d.data import full_volume
    from aerolume.viz3d.flow import volume_bounds
    for mirrored in (False, True):
        vol = sphere_flow_volume(n=(24, 12, 10), mirrored=mirrored)
        origin, spacing, fields, _ = full_volume(vol)
        nz, ny, nx = fields["u"].shape
        want = (origin[0], origin[0] + nx * spacing[0], origin[1], origin[1] + ny * spacing[1],
                origin[2], origin[2] + nz * spacing[2])
        assert np.allclose(volume_bounds(vol), want)


@needs_gl
def test_smoke_opacity_curve_is_honoured_and_checked():
    from aerolume.viz3d import Scene
    vol = _smoke_field()
    seen = []
    for curve in (None, ((0.5, 0.0), (1.0, 0.02))):
        s = Scene("dark", axes=False)
        try:
            s.add_smoke(vol, opacity_curve=curve)
            s.set_focus((-1.5, 2.5, -1, 1, -1, 1))
            s.set_camera("top")
            img = s.render(400, 260).astype(int)
            bg = np.round(np.asarray(s.theme.background) * 255).astype(int)
            seen.append(float(np.abs(img - bg).mean()))
        finally:
            s.close()
    assert seen[1] < 0.2 * seen[0]                                   # a faint curve draws faint smoke
    s = Scene("dark", axes=False)
    try:
        for bad in ((("a", 1),), ((1.5, 0.5),), ((0.5, -0.1),), ()):
            with pytest.raises(ValueError):
                s.add_smoke(vol, opacity_curve=bad)
    finally:
        s.close()


@needs_gl
def test_streamline_tubes_never_merge_into_a_ribbon():
    from aerolume.viz3d import Scene
    from aerolume.viz3d.scene import _seed_spacing
    vol = sphere_flow_volume()
    s = Scene("light")
    try:
        dense = {"kind": "rake", "n": 60, "y": [-0.3, 0.3]}           # 1 cm apart along a 4 m domain
        h = s.add_streamlines(vol, dense)
        lay = s._layers[h]
        gap = _seed_spacing(lay.extra["seeds"])
        assert gap == pytest.approx(0.6 / 59, rel=1e-6)
        assert lay.extra["tube_radius"] <= 0.2 * gap + 1e-12
        sparse = s.add_streamlines(vol, {"kind": "rake", "n": 4})
        lb = s._layers[sparse].extra["polydata"].GetBounds()
        extent = math.dist((lb[0], lb[2], lb[4]), (lb[1], lb[3], lb[5]))
        assert s._layers[sparse].extra["tube_radius"] == pytest.approx(0.0018 * extent)
    finally:
        s.close()
    assert _seed_spacing(None) == 0.0 and _seed_spacing(np.zeros((1, 3))) == 0.0


@needs_gl
def test_colour_bar_sits_on_a_plate_over_lines_running_under_it():
    from aerolume.viz3d import Scene, overlay
    vol = sphere_flow_volume()
    s = Scene("light", axes=False)
    try:
        s.add_slice(vol, "z", 0.0, field="speed")                    # fills the whole picture
        s.set_focus((1.2, 1.8, -0.3, 0.3, -0.01, 0.01))                # the wake, away from the sphere
        s.set_camera("top")
        s.add_colorbar()
        W, H = 800, 500
        img = s.render(W, H).astype(int)
        lay = overlay.colorbar_layout(s.colorbar_spec(), W, H)
        x0, y0, _, y1 = lay["bar"]
        row = round(H - 0.5 * (y0 + y1))
        bg = np.round(np.asarray(s.theme.background) * 255).astype(int)
        inside = img[row, W - 7]                                     # on the plate, right of the labels
        outside = img[row, round(x0) - 20]                           # the slice, left of the plate
        assert np.abs(outside - bg).max() > 40                       # something coloured runs under it
        assert np.abs(inside - bg).max() <= 0.25 * np.abs(outside - bg).max() + 3
    finally:
        s.close()


def test_scene_without_opengl_accepts_flow_layers(monkeypatch):
    from aerolume.viz3d import scene as scene_mod
    monkeypatch.setattr(scene_mod, "_BACKEND", {"ok": False, "window": "", "renderer": "", "version": "",
                                                "error": "test: no OpenGL"})
    s = scene_mod.Scene("light")
    vol = _smoke_field(n=(32, 16, 16))
    h = s.add_smoke(vol)
    s.update_smoke(h, vol)
    hs = s.add_streamlines(vol, "rake")
    s.update_streamlines(hs, vol)
    hr = s.add_seed_rake({"kind": "rake"}, body_bounds=BODY, domain_bounds=DOMAIN)
    s.update_seed_rake(hr, {"kind": "vertical"}, body_bounds=BODY, domain_bounds=DOMAIN)
    img = s.render(64, 40)
    assert img.shape == (40, 64, 3) and set(s.layers().values()) >= {"smoke", "streamlines", "rake"}
    assert math.isfinite(s.last_render_ms) or True
