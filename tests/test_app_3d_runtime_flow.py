"""Flow visualisation in the volume around 3D bodies (DESIGN.md 7.13): smoke, streamlines, slice planes, rake.

Covers the GUI half: the rake model and its dict (shared with the viewer's and
the solver's rake), the framing boxes, :class:`FlowScene` (layers, one colour
bar, recolouring and in-place updates instead of rebuilds), the fallback smoke
of a viewer without a volume renderer, the Simulate page's live 3D view (smoke
and streamlines from ``LiveFrame.volume``, the settings it hands the solver,
the cost of a stream of frames), the job manager's volume hand-over, the
Results page's stored smoke snapshot and 3D-view export, and the Compare tiles.
Synthetic data: potential flow past a sphere with a dye plume from a rake.
"""
# The shared fixtures (qapp, project3d, state3d) are imported from test_app_3d_runtime, so pytest
# finds them here; their names as test arguments are those fixtures, not redefinitions.
# ruff: noqa: F811
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from aerolume.model import CFD3DResult, VolumeData

sys.path.insert(0, os.path.dirname(__file__))
from test_app_3d_runtime import (
    needs_vtk,
    project3d,  # noqa: F401  (a fixture used through state3d)
    pump,
    pump_until,
    qapp,  # noqa: F401  (a fixture)
    slice_frame,
    state3d,  # noqa: F401  (a fixture)
)
from test_viz3d_fixtures import sphere_flow_volume, sphere_surface

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# synthetic smoke
# ---------------------------------------------------------------------------

def dye_volume(vol: VolumeData, *, rake_x: float = -1.2, z: float = 0.1, ys=(-0.45, -0.15, 0.15, 0.45),
               r: float = 0.08, shift: float = 0.0) -> VolumeData:
    """Smoke streams released at x = rake_x: Gaussian tubes along x (bent by nothing; enough to test with),
    zero upstream of the rake and in the solid, plus the volume's own velocity (a LiveFrame.volume)."""
    nz, ny, nx = next(iter(vol.fields.values())).shape
    ox, oy, oz = vol.origin
    dx, dy, dz = vol.spacing
    z_, y_, x_ = np.meshgrid(oz + (np.arange(nz) + 0.5) * dz, oy + (np.arange(ny) + 0.5) * dy,
                             ox + (np.arange(nx) + 0.5) * dx, indexing="ij")
    dye = np.zeros_like(x_)
    for yc in ys:
        dye = np.maximum(dye, np.exp(-((y_ - yc - shift) ** 2 + (z_ - z) ** 2) / r ** 2))
    dye[x_ < rake_x] = 0.0
    if vol.mask is not None:
        dye[np.asarray(vol.mask) > 0.5] = 0.0
    fields = {"dye": dye.astype(np.float32)}
    for k in ("u", "v", "w"):
        fields[k] = np.asarray(vol.fields[k], np.float32) * (1.0 + 0.02 * shift)
    return VolumeData(origin=vol.origin, spacing=vol.spacing, fields=fields, mask=vol.mask, label="instant",
                      mirrored=vol.mirrored)


def volume_frame(axis: str = "y", *, volume=None, step: int = 900):
    """A live 3D slice frame (test_app_3d_runtime.slice_frame) carrying a flow volume."""
    f = slice_frame(axis)
    f.volume = volume
    f.step = step
    return f


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_rake_spec_defaults_seeds_and_view_dict():
    from aerolume.app.widgets.view3d_panel import AUTO_SEEDS, RakeSpec, rake_mesh
    flow = pytest.importorskip("aerolume.viz3d.flow")
    bounds = (0.0, 4.0, -0.9, 0.9, 0.0, 1.4)                      # a car-sized box
    domain = (-10.0, 20.0, -5.0, 5.0, 0.0, 6.0)
    r = RakeSpec.around(bounds, domain=domain, n=10)
    # the default is the solver's own (a run never given a rake releases its smoke where this is drawn)
    sr0 = flow.resolve_rake({"kind": "rake"}, bounds, domain, purpose="streamlines")
    assert (r.x, r.z) == pytest.approx((sr0.x, sr0.z[0])) and r.x < 0.0 < r.z < 1.4
    assert (r.y_min, r.y_max) == pytest.approx((-0.9 - 0.36, 0.9 + 0.36))   # 20 % of the width each side
    seeds = r.seeds()
    assert seeds.shape == (10, 3) and np.allclose(seeds[:, 0], r.x) and np.allclose(seeds[:, 2], r.z)
    d = r.to_view()
    assert json.loads(json.dumps(d)) == d and d["kind"] == "rake" and d["n"] == 10
    assert RakeSpec.from_view(d) == r and RakeSpec.from_view({"kind": "grid"}) is None
    # the dict means the same rake to the viewer and the solver (viz3d.flow.resolve_rake)
    sr = flow.resolve_rake(d, bounds, domain)
    np.testing.assert_allclose(sr.points, seeds, atol=1e-9)
    # automatic counts stay out of the dict: the solver spaces its smoke nozzles itself
    auto = RakeSpec.around(bounds, domain=domain)
    assert auto.auto_n and "n" not in auto.to_view() and len(auto.seeds()) == AUTO_SEEDS["rake"]
    c = r.clamped((-0.5, 10.0, -1.0, 1.0, 0.0, 3.0))
    assert -0.5 < c.x and -1.0 < c.y_min and c.y_max < 1.0
    assert r.with_span(1.0).span == pytest.approx(1.0) and r.with_span(1.0).y_centre == pytest.approx(r.y_centre)
    v, f = rake_mesh(r, 4.0)
    assert f.max() < len(v) and len(f) > 10 * 8


@pytest.mark.parametrize("spec", [
    {"kind": "vertical", "z": [0.2, 1.2], "n": 9},
    {"kind": "grid", "z": [0.2, 1.2], "n": [5, 3]},
    {"kind": "plane", "z": [0.3, 1.0]},
    {"kind": "rake", "z": 0.5, "radius": 0.02, "pulse": [0.1, 0.2]},
])
def test_every_setup_emitter_round_trips_and_seeds_like_the_solver(spec):
    """Setup's emitters (rake, vertical rake, grid, plane band) keep their kind, extent, counts and extras
    through the GUI's rake, and its seeds are the solver's seed points."""
    from aerolume.app.widgets.view3d_panel import AUTO_SEEDS, RakeSpec, rake_mesh
    flow = pytest.importorskip("aerolume.viz3d.flow")
    bounds = (0.0, 4.0, -0.9, 0.9, 0.0, 1.4)
    domain = (-10.0, 20.0, -5.0, 5.0, 0.0, 6.0)
    r = RakeSpec.resolve(spec, bounds, domain=domain)
    assert r.kind == spec["kind"] and r.auto_n == ("n" not in spec)
    d = r.to_view()
    assert json.loads(json.dumps(d)) == d and RakeSpec.from_view(d) == r
    if r.ranged:
        assert (r.z, r.z_max) == pytest.approx(tuple(spec["z"])) and d["z"] == pytest.approx(spec["z"])
    if not r.auto_n:
        np.testing.assert_allclose(r.seeds(), flow.resolve_rake(d, bounds, domain).points, atol=1e-9)
    else:
        assert len(r.seeds()) <= AUTO_SEEDS[r.kind] + 4               # a plane is traced here: capped
    if "radius" in spec:
        assert d["radius"] == pytest.approx(0.02) and d["pulse"] == pytest.approx([0.1, 0.2])
    moved = r.moved_to(RakeSpec.from_view({**d, "x": d["x"] - 0.3}))
    assert moved.kind == r.kind and moved.x == pytest.approx(r.x - 0.3) and moved.counts() == r.counts()
    v, f = rake_mesh(r, 4.0)
    assert len(v) and f.max() < len(v)


def test_boxes_follow_each_dimension():
    from aerolume.app.widgets.view3d_panel import bounds_of, flow_focus_box, near_field_box, plane_window
    wing = (0.0, 0.3, -1.2, 1.2, -0.02, 0.03)
    car = (0.0, 4.4, -0.9, 0.9, 0.0, 1.4)
    fw = flow_focus_box(wing)
    assert fw[1] - fw[0] < fw[3] - fw[2] and fw[5] - fw[4] < 0.5 * (fw[3] - fw[2])   # wide, not tall
    assert fw[1] >= 0.3 + 0.3 * 2.4 - 1e-9                      # the trailing vortices are in the frame
    fc = flow_focus_box(car, ground=0.0)
    assert fc[4] == 0.0 and fc[1] >= 4.4 + 4.4 - 1e-9          # down to the road, one length of wake
    nb = near_field_box(car, ground=0.0)
    assert all(nb[i] <= fc[i] for i in (0, 2, 4)) and all(nb[i] >= fc[i] for i in (1, 3, 5))
    assert plane_window(nb, "y") == (nb[0], nb[1], nb[4], nb[5])
    assert plane_window(nb, "x") == (nb[2], nb[3], nb[4], nb[5])
    assert flow_focus_box(None) is None and bounds_of([]) is None
    s = sphere_surface(1.0, 12, 24)
    assert bounds_of([s]) == pytest.approx((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), abs=0.02)


def test_smoke_volume_of_reads_every_documented_home(tmp_path):
    from aerolume.app.widgets.view3d_panel import smoke_volume_of
    vel = {k: np.zeros((2, 3, 4), np.float32) for k in ("u", "v", "w")}
    smoke = VolumeData(origin=(0, 0, 0), spacing=(1, 1, 1), fields={"dye": np.ones((2, 3, 4), np.float32)},
                       label="instant")
    r = CFD3DResult(sim_id="x")
    assert smoke_volume_of(r) is None
    r.extra_volumes = [VolumeData(origin=(0, 0, 0), spacing=(1, 1, 1), fields=dict(vel)), smoke]
    assert smoke_volume_of(r) is smoke
    r2 = CFD3DResult(sim_id="y", volume=VolumeData(origin=(0, 0, 0), spacing=(1, 1, 1),
                                                   fields={**vel, "dye": np.ones((2, 3, 4), np.float32)}))
    assert smoke_volume_of(r2) is r2.volume
    # through the project store: a saved snapshot comes back as smoke
    from test_app_3d_runtime import sphere_design

    from aerolume.model import Simulation
    from aerolume.project import Project
    pr = Project.create(tmp_path / "Smoke.aerolume", "Smoke")
    d = sphere_design()
    pr.add_design(d)
    pr.add_simulation(Simulation(name="s", design_id=d.id, id="s1"))
    pr.save_result("s1", r)
    back = pr.load_result("s1")
    got = smoke_volume_of(back)
    assert got is not None and got.label == "instant" and np.allclose(np.asarray(got.fields["dye"], float), 1.0)


def test_make_view_settings_has_the_3d_names():
    from aerolume.app.jobs import VIEW3D_DEFAULTS, FallbackViewSettings, make_view_settings
    v = make_view_settings()
    for k in VIEW3D_DEFAULTS:
        assert hasattr(v, k), k
    fb = FallbackViewSettings()
    assert fb.smoke is True and fb.view3d is False and fb.seed_rake is None and fb.slice_axis == "y"


def test_jobs_carry_the_newest_volume_and_never_drop_volume_frames(qapp, state3d):
    from aerolume.app.jobs import _Mailbox, carry_volume
    mb = _Mailbox(merge=carry_volume)
    a, b = SimpleNamespace(volume="V1", step=1), SimpleNamespace(volume=None, step=2)
    assert mb.put(a) and not mb.put(b)
    got = mb.take()
    assert got.step == 2 and got.volume == "V1"                  # the newer slice keeps the volume
    mb.put(SimpleNamespace(volume="V2", step=3))
    mb.put(SimpleNamespace(volume="V3", step=4))
    assert mb.take().volume == "V3"
    assert carry_volume(None, b) is b and carry_volume(a, None) is None
    # the volume's serial goes with it, and solver-traced streamlines are carried on their own
    c = carry_volume(SimpleNamespace(volume="V4", volume_serial=7, streamlines3d="L1"),
                     SimpleNamespace(volume=None, volume_serial=0, streamlines3d=None))
    assert (c.volume, c.volume_serial, c.streamlines3d) == ("V4", 7, "L1")
    c = carry_volume(SimpleNamespace(volume=None, streamlines3d="L2"), SimpleNamespace(volume="V5", streamlines3d=None))
    assert (c.volume, c.streamlines3d) == ("V5", "L2")
    # through the job manager: frames sent faster than the display rate, every third with a volume
    _state, jobs = state3d
    seen = []
    jobs.jobFrame.connect(lambda sid, f: seen.append((f.step, getattr(f, "volume", None))))

    def run(design, sim, *, progress, frame, cancel, view, **_kw):
        for i in range(30):
            frame(SimpleNamespace(step=i, phase="averaging", volume=f"vol{i}" if i % 3 == 0 else None))
        frame(SimpleNamespace(step=30, phase="done", volume=None))
        time.sleep(0.05)
        from test_app_3d_runtime import synthetic_cfd3d
        return synthetic_cfd3d(sim.id)

    jobs.runner = run
    jobs.submit("s3n")
    assert jobs.wait(20)
    pump(qapp, 0.1)
    assert seen and seen[-1][0] == 30
    vols = [v for _s, v in seen if v is not None]
    assert vols and vols[-1] == "vol27"                           # the newest volume always arrived


# ---------------------------------------------------------------------------
# FlowScene
# ---------------------------------------------------------------------------

def _flow(qapp, shown):
    from aerolume.app.widgets.view3d import View3D
    from aerolume.app.widgets.view3d_panel import FlowScene
    view = View3D()
    view.resize(800, 600)
    return view, FlowScene(view, shown=shown)


@needs_vtk
def test_flow_scene_layers_one_colour_bar_and_cheap_updates(qapp):
    from aerolume.app.widgets.view3d_panel import RakeSpec, bounds_of, flow_focus_box
    from aerolume.viz3d.data import slice_from_volume
    view, flow = _flow(qapp, {"smoke": True, "streamlines": True, "slice": True, "q": False, "surface": False})
    surf = sphere_surface(1.0, 24, 48)
    vol = sphere_flow_volume(n=(48, 32, 32))
    smoke = dye_volume(vol)
    bounds = bounds_of([surf])
    flow.set_data(bodies=[surf], body_key="sphere", focus=flow_focus_box(bounds), volume=vol, smoke=smoke,
                  rake=RakeSpec.around(bounds, n=16))
    sd = slice_from_volume(vol, "y", 0.0, "speed")
    flow.set_data(slice=sd, field="speed", cmap=None, rng=(0.0, 15.0))
    h = flow.layer_handles()
    assert {"body", "smoke", "streamlines", "slice", "rake"} <= set(h), (h, flow.errors)
    sc = view.scene
    assert sc.layers()[h["smoke"][0]] == ("smoke" if hasattr(sc, "add_smoke") else "isosurface")
    assert len(sc.layer_lines(h["streamlines"][0])) >= 10
    spec = sc.colorbar_spec()
    assert spec is not None and spec.title.startswith("Speed") and (spec.vmin, spec.vmax) == (0.0, 15.0)
    info = sc.scalar_info(h["streamlines"][0])
    assert info["field"] == "speed" and info["vmax"] == pytest.approx(15.0)
    # a new colour range recolours the lines in place: no new trace, same handle
    flow.set_data(slice=sd, field="speed", rng=(0.0, 12.0))
    assert flow.layer_handles()["streamlines"] == h["streamlines"]
    assert sc.scalar_info(h["streamlines"][0])["vmax"] == pytest.approx(12.0)
    # the same volume again rebuilds nothing; a new live volume updates smoke and lines in place
    assert flow.set_data(volume=vol, smoke=smoke) == []
    vol2 = dye_volume(vol, shift=0.05)
    flow.set_data(volume=vol2, smoke=vol2)
    if hasattr(sc, "update_smoke"):
        assert flow.layer_handles()["smoke"] == h["smoke"]
    if hasattr(sc, "update_streamlines"):
        assert flow.layer_handles()["streamlines"] == h["streamlines"]
    # the colour bar follows what is shown: slice > streamlines > surface Cp
    flow.show("slice", False)
    assert sc.colorbar_spec().title.startswith("Speed")
    flow.show("surface", True)
    flow.show("streamlines", False)
    spec = sc.colorbar_spec()
    assert spec is not None and "Cp" in spec.title and spec.vmin == pytest.approx(-spec.vmax)
    assert "smoke" in flow.describe()
    view.scene.close()


@needs_vtk
def test_a_failing_layer_never_takes_the_others_down(qapp, monkeypatch):
    from aerolume.app.widgets.view3d_panel import RakeSpec, bounds_of
    view, flow = _flow(qapp, {"smoke": False, "streamlines": True, "slice": False, "q": True, "surface": False})
    surf = sphere_surface(1.0, 12, 24)
    vol = sphere_flow_volume(n=(32, 24, 24))
    vol.fields["q"] = np.asarray(vol.fields["speed"], np.float32) - 9.0

    def boom(*_a, **_k):
        raise RuntimeError("no iso-surface today")
    monkeypatch.setattr(view.scene, "add_isosurface", boom)
    flow.set_data(bodies=[surf], body_key="s", volume=vol, rake=RakeSpec.around(bounds_of([surf])))
    flow.set_q(0.5)
    assert "q" in flow.errors and "no iso-surface" in flow.errors["q"]
    assert {"body", "streamlines"} <= set(flow.layer_handles())
    view.scene.close()


@needs_vtk
def test_fallback_smoke_without_a_volume_renderer(qapp, monkeypatch):
    from aerolume.app.widgets.view3d_panel import SMOKE_SHELLS, add_smoke_layers
    from aerolume.viz3d import Scene
    monkeypatch.setattr(Scene, "add_smoke", None, raising=False)
    sc = Scene("light")
    vol = sphere_flow_volume(n=(32, 24, 24))
    hs = add_smoke_layers(sc, dye_volume(vol))
    assert len(hs) == len(SMOKE_SHELLS) and all(sc.layers()[h] == "isosurface" for h in hs)
    empty = dye_volume(vol)
    empty.fields["dye"][:] = 0.0
    assert add_smoke_layers(sc, empty) == []
    sc.close()


# ---------------------------------------------------------------------------
# Simulate page: the live 3D view
# ---------------------------------------------------------------------------

@needs_vtk
def test_simulate_page_live_smoke_streamlines_and_rake(qapp, state3d, monkeypatch):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3n")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.3)
    toggles = page.flow_toggles
    # before the run: the body and the rake the smoke will leave from (the slice plane, a grey plane
    # where the solver will cut, only on request); smoke waits for the run
    assert page._mode == "field3d" and page.view3d.isVisible()
    h = page.flow.layer_handles()
    assert {"body", "rake"} <= set(h) and "slice" not in h and page.flow.slice_neutral
    toggles.buttons["slice"].setChecked(True)
    assert "slice" in page.flow.layer_handles()
    toggles.buttons["slice"].setChecked(False)
    assert not toggles.buttons["smoke"].isEnabled() and "once the run starts" in toggles.buttons["smoke"].toolTip()
    assert toggles.wanted("smoke") and not toggles.wanted("streamlines")
    v = jobs.view
    assert v.view3d is True and v.smoke is True and v.streamlines3d is False
    # the rake belongs to the simulation: a page showing a case never sets the rake of whatever runs
    assert v.seed_rake is None and page.rake_editor.is_default() and page.flow.rake is not None
    # the run streams frames; every few carries a flow volume (smoke + velocity)
    monkeypatch.setattr(jobs, "running_id", lambda: "s3n")
    page._shown_sim_id = "s3n"
    vol = sphere_flow_volume(n=(48, 32, 32))
    t0 = time.perf_counter()
    n_frames = 24
    for i in range(n_frames):
        jobs.jobFrame.emit("s3n", volume_frame("y", volume=dye_volume(vol, shift=0.01 * i) if i % 4 == 0 else None,
                                               step=900 + i))
        qapp.processEvents()
    per_frame_ms = 1000.0 * (time.perf_counter() - t0) / n_frames
    pump(qapp, 0.6)
    h = page.flow.layer_handles()
    assert "smoke" in h and "rake" in h and not page.flow.slice_neutral, (h, page.flow.errors)
    assert toggles.buttons["smoke"].isEnabled() and toggles.buttons["streamlines"].isEnabled()
    # GUI time per frame (slice in 2D and 3D, smoke every fourth frame) stays far below a frame period
    # at the frame rates runs send (<= 20 per second); measured ~10-30 ms here
    assert per_frame_ms < 120.0, per_frame_ms
    smoke_h = h["smoke"]
    jobs.jobFrame.emit("s3n", volume_frame("y", volume=None, step=999))
    pump(qapp, 0.2)
    assert page.flow.layer_handles()["smoke"] == smoke_h       # a frame without a volume keeps the smoke
    # streamlines: one toggle for the 3D lines and the slice's in-plane lines; the solver is told
    toggles.buttons["streamlines"].setChecked(True)
    assert pump_until(qapp, lambda: "streamlines" in page.flow.layer_handles(), 5), page.flow.errors
    assert v.streamlines3d is True and v.streamlines is True
    lines = page.view3d.scene.layer_lines(page.flow.layer_handles()["streamlines"][0])
    assert len(lines) >= 8
    # moving the rake moves the seeds and tells the solver where the smoke nozzles go
    z_new = page.rake_editor.z.value() + 0.1
    page.rake_editor.z.setValue(z_new)
    pump(qapp, 0.2)
    assert v.seed_rake["z"] == pytest.approx(z_new, abs=1e-3)
    assert page.flow.rake.z == pytest.approx(z_new, abs=1e-3)
    # the rake dragged by its handle in the view: the rows follow while it moves, the drop is an edit
    if hasattr(page.view3d, "rakeChanged"):
        moved = dict(v.seed_rake)
        moved["x"] = v.seed_rake["x"] - 0.05
        page.view3d.rakeMoving.emit(moved)
        assert page.rake_editor.x.value() == pytest.approx(moved["x"], abs=1e-3)
        page.view3d.rakeChanged.emit(moved)
        pump(qapp, 0.2)
        assert v.seed_rake["x"] == pytest.approx(moved["x"], abs=1e-3)
        assert page.flow.rake.x == pytest.approx(moved["x"], abs=1e-3)
    # streamlines the run traced on its own thread are uploaded, not traced again here
    flow_mod = pytest.importorskip("aerolume.viz3d.flow")
    if hasattr(flow_mod, "trace_streamlines"):
        traced = flow_mod.trace_streamlines(vol, page.flow.rake.seeds())
        f = volume_frame("y", volume=None, step=1500)
        f.streamlines3d = traced
        jobs.jobFrame.emit("s3n", f)
        assert pump_until(qapp, lambda: page.flow.lines is traced, 5)
        assert "streamlines" in page.flow.layer_handles()
    # View: Slice hides the 3D view (no 3D work), and the solver may stop sending volumes
    page.view_mode.set_current("slice")
    pump(qapp, 0.1)
    assert not page.view3d_host.isVisible() and page.view.isVisible() and v.view3d is False
    assert not toggles.buttons["smoke"].isVisible()
    page.view_mode.set_current("split")
    pump(qapp, 0.1)
    assert page.view3d_host.isVisible() and page.view.isVisible() and v.view3d is True
    # the run ends: the 3D view shows the stored result (surface Cp becomes available)
    from test_app_3d_runtime import synthetic_cfd3d
    res = synthetic_cfd3d("s3n")
    state.project.save_result("s3n", res)
    state.project.simulations["s3n"].status = "done"
    jobs.jobFinished.emit("s3n", res)
    pump(qapp, 0.3)
    assert toggles.buttons["surface"].isEnabled()
    assert "time-averaged flow" in page.flow.caption       # streamlines are on: they show the mean flow
    page.shutdown()
    page.close()


@needs_vtk
def test_setup_emitter_reaches_the_run_and_the_results(qapp, state3d, monkeypatch):
    """Setup ▸ Flow visualisation picks a grid emitter for one case: Simulate draws and positions that
    grid, a run of the case releases its smoke there (another case keeps its own rake), a move on the
    Simulate page goes with the case, Reset hands it back to Setup, and Results starts its streamlines
    on it."""
    setup = pytest.importorskip("aerolume.app.pages.setup")
    from test_app_3d_runtime import synthetic_cfd3d

    from aerolume.app.pages.results import ResultsPage
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    sim = state.project.simulations["s3n"]
    setup.set_flow_viz(sim, dict(setup.FLOW_VIZ_DEFAULTS, kind="grid", z_min=-0.3, z_max=0.3, n=[5, 3]))
    grid_dict = {"kind": "grid", "z": [-0.3, 0.3], "n": [5, 3]}
    assert jobs.seed_rake_for(sim) == grid_dict
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3n")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.3)
    ed = page.rake_editor
    assert ed.kind() == "grid" and ed.kind_value.text() == "Grid" and not ed._row_z_top.label.isHidden()
    r = page.flow.rake
    assert r.kind == "grid" and (r.z, r.z_max) == pytest.approx((-0.3, 0.3)) and r.counts() == (5, 3)
    assert page.view3d.seed_rake() is not None                       # drawn in the view

    seen = {}

    def run(design, sim, *, progress, frame, cancel, view, **_kw):
        seen[sim.id] = view.seed_rake
        return synthetic_cfd3d(sim.id)

    jobs.runner = run
    jobs.submit("s3n")
    assert jobs.wait(20)
    assert seen["s3n"] == grid_dict                                   # Setup's emitter, resolved by the solver
    # the user moves the rake for this case: the next run of it starts there; another case does not
    x_new = ed.x.value() - 0.1
    ed.x.setValue(x_new)
    assert pump_until(qapp, lambda: abs(page.flow.rake.x - x_new) < 1e-6, 3)
    assert not ed.is_default()
    jobs.submit("s3n")
    assert jobs.wait(20)
    pump(qapp, 0.2)
    assert seen["s3n"]["kind"] == "grid" and seen["s3n"]["x"] == pytest.approx(x_new)
    assert seen["s3n"]["n"] == [5, 3]
    jobs.submit("s3d")
    assert jobs.wait(20)
    assert seen["s3d"] == {"kind": "rake"}                           # its own (default) rake
    # Results of the case: the streamlines start on the rake its run used
    rp = ResultsPage(state, jobs)
    rp.resize(1600, 1000)
    state.select_simulation("s3n")
    rp.show()
    pump(qapp)
    rp.handle_action("fields")
    pump(qapp, 0.4)
    fr = rp.view3d.fields_panel.flow.rake
    assert fr.kind == "grid" and fr.x == pytest.approx(x_new, abs=1e-6)
    rp.close()
    # Reset: back to Setup's emitter, which the case's runs follow again (the viewport still shows the
    # last run, s3d: the user picks the case again in the Project tree)
    state.select_simulation("s3d")
    state.select_simulation("s3n")
    pump(qapp, 0.2)
    assert page._rake_sim == "s3n" and page.flow.rake.x == pytest.approx(x_new, abs=1e-6)
    page.rake_editor.reset()
    pump(qapp, 0.2)
    assert page.rake_editor.is_default() and jobs.seed_rake_for(sim) == grid_dict
    page.shutdown()
    page.close()


# ---------------------------------------------------------------------------
# Results page: stored smoke, rake, exports
# ---------------------------------------------------------------------------

@needs_vtk
def test_results_fields_show_the_stored_smoke_and_export_the_3d_view(qapp, state3d, tmp_path, monkeypatch):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QFileDialog

    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    r = state.project.load_result("s3d")
    r.extra_volumes = [dye_volume(r.volume)]
    state.project.save_result("s3d", r)
    state.select_simulation("s3d")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    v = page.view3d
    loaded = jobs.result("s3d")
    if not getattr(loaded, "extra_volumes", None):
        pytest.skip("the project store does not keep extra volumes yet")
    page.handle_action("fields")
    pump(qapp, 0.4)
    fp = v.fields_panel
    # a result with a smoke snapshot opens on the smoke with thin streamlines over it (DESIGN 7.13)
    assert fp.smoke_on.isEnabled() and {"smoke", "rake", "stream"} <= set(fp.handles())
    assert fp.smoke_on.isChecked() and fp.stream_on.isChecked()
    assert "instantaneous dye" in fp.note.text()
    # smoke alone: the seed rake (it seeds streamlines, not this stored smoke) is not drawn
    fp.stream_on.setChecked(False)
    assert pump_until(qapp, lambda: "stream" not in fp.handles() and "rake" not in fp.handles(), 5)
    # streamlines with the smoke: thin translucent lines, not shaded tubes; alone again: tubes
    fp.stream_on.setChecked(True)
    assert pump_until(qapp, lambda: "stream" in fp.handles() and "rake" in fp.handles(), 5), fp.flow.errors
    lay = fp.view.scene._layers[fp.handles()["stream"]]
    assert lay.extra["tubes"] is False
    assert all(a.GetProperty().GetOpacity() == pytest.approx(0.6) for a in lay.actors)
    fp.smoke_on.setChecked(False)
    assert pump_until(qapp, lambda: "stream" in fp.handles()
                      and fp.view.scene._layers[fp.handles()["stream"]].extra["tubes"] is True, 5)
    fp.smoke_on.setChecked(True)
    pump(qapp, 0.2)
    # the rake edits re-seed the streamlines
    x_new = fp.rake_editor.x.value() + 0.1
    fp.rake_editor.x.setValue(x_new)
    assert pump_until(qapp, lambda: fp.flow.rake is not None and abs(fp.flow.rake.x - x_new) < 1e-3, 3)
    lines = fp.view.scene.layer_lines(fp.handles()["stream"])
    assert lines and abs(float(lines[0][0][0]) - x_new) < 0.05
    fp.rake_editor.reset()
    pump(qapp, 0.3)
    assert fp.flow.rake.x != pytest.approx(x_new)
    # exports: the smoke volume and a PNG of the 3D flow view
    for key in ("smoke_vti", "view_png"):
        path = tmp_path / f"out_{key}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, p=str(path): (p, "")))
        v.data_tab.export(key)
        assert pump_until(qapp, lambda p=path: p.exists() and v.data_tab.status.text().startswith("Saved"), 15), \
            v.data_tab.status.text()
    img = QImage(str(tmp_path / "out_view_png"))
    assert (img.width(), img.height()) == (1600, 1000)
    page.close()


@needs_vtk
def test_compare_tiles_show_streamlines_from_one_rake(qapp, state3d):
    from aerolume.app.pages.compare import ComparePage
    state, jobs = state3d
    state.compare_ids[:] = ["s3d", "s3y"]
    page = ComparePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    page.handle_action("fields")
    pump(qapp, 0.3)
    grid = page.surface_grid
    assert grid is not None and len(grid.tiles) == 2
    assert grid.toggles.buttons["streamlines"].isEnabled()
    assert not grid.toggles.buttons["smoke"].isEnabled()           # no run stored a smoke snapshot
    grid.toggles.buttons["streamlines"].setChecked(True)
    pump(qapp, 0.5)
    assert len(grid.tiles) == 2
    seeds = []
    for tile in grid.tiles:
        kinds = tile.scene.layers()
        hs = [h for h, k in kinds.items() if k == "streamlines"]
        assert hs and len(tile.scene.layer_lines(hs[0])) >= 8
        seeds.append(tile.scene._layers[hs[0]].extra["seeds"])
    np.testing.assert_allclose(seeds[0], seeds[1])                 # one rake for every run
    page.close()
