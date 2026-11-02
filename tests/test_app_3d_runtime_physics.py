"""3D placed physics in the viewer and on the Results / Simulate pages (DESIGN.md 10).

* device glyphs (viz3d.placed): the attitude rotation is the solver's, sizes are true, a vent points in;
* the plume colour range spans the warm tail around the ambient, not the percentiles of all cells;
* Scene: colour-mapped scalar volumes, tracer particles with trails, glyphs outside the framing;
* FlowScene: smoke in one colour per channel, plume / iso-surface, particles in place, devices;
* Results 3D: Heat transfer (Nu with errors, per body, balance, temperature slice), Fans, Diagnostics,
  the Thermal plume toggle (hidden for isothermal runs);
* Simulate 3D: a live frame with temperature and particles reaches the 3D layers.
"""
# The shared fixtures (qapp, ...) come from test_app_3d_runtime; their names as test arguments are those
# fixtures, not redefinitions.
# ruff: noqa: F811
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

sys.path.insert(0, os.path.dirname(__file__))
from test_app_3d_runtime import (
    needs_vtk,
    pump,
    pump_until,
    qapp,  # noqa: F401  (a fixture)
    slice_frame,
    sphere_design,
    synthetic_cfd3d,
)
from test_viz3d_fixtures import grid, sphere_flow_volume

from aerolume.model import (
    Emitter,
    Fan,
    FlowConditions,
    HeatSource,
    Port,
    Simulation,
    Stat,
    ThermalSettings,
    VolumeData,
)

pytestmark = pytest.mark.gui

T_AMB, T_BODY = 20.0, 60.0


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def plume_temperature(xx, yy, zz, d: float = 1.0, width0: float = 0.35, growth: float = 0.15,
                      shell: float = 0.15) -> np.ndarray:
    """A buoyant plume of a heated sphere in a cross-flow: hot at the wall, rising along +z, drifting downstream."""
    R = d / 2
    r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    s = np.clip(xx, 0.0, None)                       # downstream distance
    zc = 0.25 * s                                    # the plume rises as it is carried
    width = width0 + growth * s
    upstream = np.exp(-np.clip(-xx, 0.0, None) / 0.1)   # nothing is warm upstream of the sphere
    T = T_AMB + (T_BODY - T_AMB) * np.exp(-((zz - zc) ** 2 + yy ** 2) / (2 * width ** 2)) * np.exp(-s / 1.5) * upstream
    layer = np.exp(-np.clip(r - R, 0, None) / shell)  # the thermal boundary layer around the sphere
    return np.maximum(T, T_AMB + (T_BODY - T_AMB) * layer).astype(np.float32)


def thermal_cfd3d(sim_id: str) -> object:
    """synthetic_cfd3d with a mean temperature field, two smoke channels in its snapshot, Nu, Q, a fan and
    diagnostics: what a 3D heat-transfer run with emitters and a fan stores."""
    r = synthetic_cfd3d(sim_id, symmetric=False)
    vol = r.volume
    nz, ny, nx = vol.fields["u"].shape
    xx, yy, zz, _h = grid((nx, ny, nz), vol.origin, tuple(vol.origin[i] + (nx, ny, nz)[i] * vol.spacing[i]
                                                         for i in range(3)))
    T = plume_temperature(xx, yy, zz)
    T[np.asarray(vol.mask) > 0.5] = T_BODY
    vol.fields["temperature"] = T
    dye = np.exp(-((yy - 0.25) ** 2 + (zz - 0.1) ** 2) / 0.01) * (xx > -1.2)
    dye1 = np.exp(-((yy + 0.25) ** 2 + (zz - 0.1) ** 2) / 0.01) * (xx > -1.2)
    snap = VolumeData(origin=vol.origin, spacing=vol.spacing, mask=vol.mask, label="instant",
                      fields={"dye": dye.astype(np.float32), "dye1": dye1.astype(np.float32),
                              "u": vol.fields["u"], "v": vol.fields["v"], "w": vol.fields["w"]})
    r.extra_volumes = [snap]
    r.nusselt = Stat(mean=12.4, rms=0.2, min=11.9, max=12.9, stderr=0.06)
    r.heat_rate = 151.0
    r.per_body = {"sphere": {"cd": 0.45, "cl": 0.02, "nu": 12.4, "heat_rate": 151.0, "nu_stderr": 0.06}}
    r.fans_report = [{"name": "Fan", "mode": "thrust", "target": 2.0, "thrust": 1.98, "thrust_stderr": 0.02,
                      "volume_flow": 0.21, "induced_velocity": 1.6, "ideal_power": 3.1, "pressure_jump": 28.0}]
    r.diagnostics = {"divergence_rms": 3e-7, "pressure_residual": 2e-14, "cfl_max": 0.41,
                     "heat_in_bodies": 151.0, "heat_out_outlet": 150.2, "heat_storage_rate": 0.4}
    t = r.history["t"]
    r.history.update(nu=12.4 + 0.1 * np.sin(t), heat_rate=151 + np.sin(t), div_rms=3e-7 + 0 * t,
                     du_rel=1e-3 / (1 + t), cfl=0.41 + 0 * t)
    return r


def thermal_sim(design_id: str, sim_id: str = "h3d") -> Simulation:
    return Simulation(
        name="Heated sphere", design_id=design_id, conditions=FlowConditions(speed=10.0), status="done", id=sim_id,
        thermal=ThermalSettings(enabled=True, t_ambient=T_AMB),
        emitters=[Emitter(name="Left", kind="point", position=(-1.2, 0.25, 0.1)),
                  Emitter(name="Right", kind="point", position=(-1.2, -0.25, 0.1), color="#e69f00"),
                  Emitter(name="Tracer", species="particles", kind="rake", position=(-1.2, -0.3, -0.2),
                          end=(-1.2, 0.3, -0.2), count=5, color="#3d8be8")],
        fans=[Fan(name="Fan", position=(-1.0, 0.0, 0.6), direction=(1.0, 0.0, 0.0), diameter=0.4, value=2.0)],
        heat_sources=[HeatSource(name="Heater", kind="box", position=(1.5, 0.0, -0.5), size=(0.3, 0.3, 0.1),
                                 power=50.0)])


@pytest.fixture()
def project_heat(tmp_path):
    from aerolume.project import Project
    pr = Project.create(tmp_path / "Heat3D.aerolume", "Heat 3D")
    d = sphere_design("Heated sphere")
    d.meshes[0].thermal = "temperature"
    d.meshes[0].thermal_value = T_BODY
    pr.add_design(d)
    pr.add_simulation(thermal_sim(d.id))
    pr.add_simulation(Simulation(name="Cold sphere", design_id=d.id, conditions=FlowConditions(speed=10.0),
                                 status="done", id="c3d"))
    pr.add_simulation(Simulation(name="Heated sphere new", design_id=d.id, conditions=FlowConditions(speed=10.0),
                                 id="h3n", thermal=ThermalSettings(enabled=True, t_ambient=T_AMB),
                                 emitters=[Emitter(name="Tracer", species="particles", position=(-1.2, 0, 0))]))
    pr.save()
    pr.save_result("h3d", thermal_cfd3d("h3d"))
    pr.save_result("c3d", synthetic_cfd3d("c3d", symmetric=False))
    return pr


@pytest.fixture()
def state_heat(qapp, project_heat):
    from aerolume.app.jobs import JobManager
    from aerolume.app.state import AppState
    state = AppState()
    state.set_project(project_heat)
    jobs = JobManager(state)
    yield state, jobs
    jobs.shutdown(timeout=5)


# ---------------------------------------------------------------------------
# glyphs and ranges (no Qt, no VTK)
# ---------------------------------------------------------------------------

def test_attitude_matrix_is_the_solvers_and_body_frame_devices_turn_with_the_body():
    torch = pytest.importorskip("torch")  # noqa: F841  (the solver module needs it)
    from aerolume.solvers.ns3d.geometry import rotation_matrix
    from aerolume.viz3d.placed import attitude_matrix, placed_glyphs
    for a, y in ((0.0, 0.0), (8.0, 0.0), (0.0, 12.0), (-5.0, 7.5)):
        assert np.allclose(attitude_matrix(a, y), rotation_matrix(a, y), atol=1e-14)
    d = sphere_design()
    d.moment_ref3 = (0.0, 0.0, 0.0)
    sim = Simulation(name="s", design_id=d.id, conditions=FlowConditions(alpha_deg=10.0),
                     heat_sources=[HeatSource(kind="sphere", position=(1.0, 0.0, 0.0), size=(0.1,), frame="body")],
                     fans=[Fan(position=(0.0, 0.0, 1.0), direction=(1, 0, 0), diameter=0.5, frame="wind")])
    g = {x.kind: x for x in placed_glyphs(sim, d)}
    # +10 deg pitch turns a point at +x down (nose up: DESIGN 7.2): R @ (1, 0, 0) = (cos a, 0, -sin a)
    c = np.asarray(g["heater"].centre)
    assert np.allclose(c, (math.cos(math.radians(10)), 0.0, -math.sin(math.radians(10))), atol=1e-12)
    assert np.allclose(g["fan"].centre, (0.0, 0.0, 1.0))           # a wind-frame fan stays where it is


def test_fan_glyph_has_its_true_diameter_and_a_vent_arrow_points_into_the_vent():
    from aerolume.viz3d.placed import placed_glyphs
    sim = Simulation(name="s", design_id="d",
                     fans=[Fan(position=(0.0, 0.0, 0.0), direction=(0, 0, 1), diameter=0.6)],
                     ports=[Port(kind="vent", position=(2.0, 0.0, 0.0), direction=(1, 0, 0), size=0.2)])
    fan, vent = placed_glyphs(sim, scale=1.0)
    assert fan.kind == "fan" and vent.kind == "vent"
    disc_v = fan.parts[0][0]
    assert np.isclose(np.hypot(disc_v[:, 0], disc_v[:, 1]).max(), 0.3, rtol=1e-9)  # radius 0.3 m in the disc plane
    arrow_v = fan.parts[-1][0]
    assert arrow_v[:, 2].max() == pytest.approx(0.8 * 0.6, rel=1e-6)                 # along +z, 0.8 diameters
    arrow = vent.parts[-1][0]
    # the cone tip (the point furthest along the arrow) sits on the vent: the flow goes in
    tip = arrow[np.argmin(arrow[:, 0])]
    assert tip[0] == pytest.approx(2.0, abs=1e-6)
    assert arrow[:, 0].max() > 2.2


def test_tiny_devices_are_floored_and_disabled_ones_are_kept():
    from aerolume.viz3d.placed import MIN_FRACTION, placed_glyphs
    sim = Simulation(name="s", design_id="d",
                     emitters=[Emitter(position=(0, 0, 0), radius=1e-5, enabled=False)])
    (g,) = placed_glyphs(sim, scale=4.0)
    b = g.bounds()
    assert b[1] - b[0] >= 2 * MIN_FRACTION * 4.0 and not g.enabled
    assert placed_glyphs(sim, scale=4.0, include_disabled=False) == []


def test_heat_range_spans_the_plume_not_the_ambient_cells():
    from aerolume.app.widgets.view3d_panel import heat_display_range
    # a thin plume in a tunnel-sized domain: 1.7 % of the cells are more than 0.5 K above ambient
    xx, yy, zz, _h = grid((120, 60, 60), (-3.0, -3.0, -3.0), (9.0, 3.0, 3.0))
    T = plume_temperature(xx, yy, zz, width0=0.12, growth=0.06, shell=0.04)
    warm_share = float((T > T_AMB + 0.5).mean())
    lo, hi, mode = heat_display_range(T)
    # the field's coolest fluid (this plume's tails never quite reach ambient: 0.04 K above it)
    assert lo == pytest.approx(T_AMB, abs=0.0025 * (T_BODY - T_AMB)) and mode == "high"
    assert hi > T_AMB + 0.8 * (T_BODY - T_AMB)            # the hot core, not a hair above ambient
    # the generic rule (the 99.5th percentile of every cell) ends at about 23 °C there: the whole plume
    # would be one saturated colour
    p995 = float(np.percentile(T, 99.5))
    assert warm_share < 0.03 and p995 < T_AMB + 0.1 * (T_BODY - T_AMB) and hi == pytest.approx(T_BODY, abs=0.5)
    both = np.concatenate([T.ravel(), 2 * T_AMB - T.ravel()])      # a heater and a cooler
    lo2, hi2, mode2 = heat_display_range(both)
    assert mode2 == "both" and lo2 < T_AMB < hi2
    assert heat_display_range(np.full(100, 20.0))[2] == "high"    # isothermal: a small range, no crash


# ---------------------------------------------------------------------------
# the scene (VTK, off screen)
# ---------------------------------------------------------------------------

@needs_vtk
def test_scene_scalar_volume_particles_and_glyphs():
    from aerolume.viz3d import Scene
    from aerolume.viz3d.placed import sphere_mesh
    vol = sphere_flow_volume(n=(48, 32, 32))
    nz, ny, nx = vol.fields["u"].shape
    xx, yy, zz, _h = grid((nx, ny, nz), (-1.5, -1.0, -1.0), (2.5, 1.0, 1.0))
    vol.fields["temperature"] = plume_temperature(xx, yy, zz)
    s = Scene("light")
    try:
        body = s.add_mesh(sphere_mesh((0, 0, 0), 0.5))
        h = s.add_scalar_volume(vol, "temperature", range=(T_AMB, T_BODY), floor=0.1)
        info = s.scalar_info(h)
        assert info["field"] == "temperature" and info["cmap"] == "thermal"
        assert (info["vmin"], info["vmax"]) == (T_AMB, T_BODY)
        s.add_colorbar(layer=h)
        assert s.colorbar_spec().title.startswith("Temperature") and s.colorbar_spec().units == "°C"
        s.set_cmap(h, "magma")
        assert s.scalar_info(h)["cmap"] == "magma"
        rev = s.revision
        s.update_scalar_volume(h, vol)                      # in place: same handle, range kept
        assert s.revision > rev and s.scalar_info(h)["vmax"] == T_BODY
        with pytest.raises(ValueError):
            s.add_scalar_volume(vol, "temperature", mode="sideways")
        pts = np.column_stack([np.linspace(-1, 2, 50), np.zeros(50), np.zeros(50)])
        trails = pts[:, None, :] - np.linspace(0.2, 0.0, 4)[None, :, None] * np.array([1.0, 0.0, 0.0])
        trails[0, :2] = np.nan                              # released two samples ago
        cols = np.tile(np.array([[230, 159, 0]], np.uint8), (50, 1))
        hp = s.add_particles(pts, colors=cols, trails=trails, mirror_y=True)
        assert s.layer_info(hp)["n"] == 100                  # mirrored for a half domain
        s.update_particles(hp, pts[:10], colors=cols[:10])
        assert s.layer_info(hp)["n"] == 20
        with pytest.raises(ValueError):
            s.update_particles(hp, pts, colors=cols[:3])
        before = s.focus_bounds()
        hg = s.add_glyph(sphere_mesh((5.0, 0, 0), 0.2), color="#2f6fdb", opacity=0.4)
        assert s.layer_info(hg)["kind"] == "glyph"
        assert s.focus_bounds() == before                   # a device never moves the camera's framing
        img = s.render(400, 300)
        assert img.shape == (300, 400, 3) and img.std() > 1.0
        s.remove(body)
    finally:
        s.close()


# ---------------------------------------------------------------------------
# FlowScene
# ---------------------------------------------------------------------------

@needs_vtk
def test_flow_scene_keys_the_plume_and_the_streamlines_with_one_bar_each(qapp):
    """The heated box's Fields view showed the thermal plume and streamlines coloured by speed under one
    bar, "Temperature T": the streamlines' greens and yellows were read off a temperature scale. Two
    colour maps on screen get two bars; one map, one bar."""
    from aerolume.app.widgets.view3d import View3D
    from aerolume.app.widgets.view3d_panel import FlowScene, RakeSpec
    r = thermal_cfd3d("x")
    view = View3D()
    view.resize(900, 600)
    try:
        flow = FlowScene(view, shown={"heat": True, "streamlines": True})
        flow.set_bodies(r.surface, key="b")
        flow.set_volume(r.volume, rebuild=False)
        flow.set_rake(RakeSpec.around((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), n=8), rebuild=False)
        flow.set_heat(r.volume)
        flow.rebuild()
        assert flow.handles.get("streamlines") and flow.handles.get("heat")
        titles = [s.title for s in view.scene.colorbar_specs()]
        assert len(titles) == 2 and titles[0].startswith("Temperature") and titles[1].startswith("Speed")
        assert [k for k, _h in flow.colorbar_layers()] == ["heat", "streamlines"]
        img = view.scene.render(900, 600)
        assert img.shape == (600, 900, 3)
        flow.show("streamlines", False)
        assert [s.title[:11] for s in view.scene.colorbar_specs()] == ["Temperature"]
        flow.show("streamlines", True)
        flow.show("heat", False)
        assert [s.title[:5] for s in view.scene.colorbar_specs()] == ["Speed"]
    finally:
        view.close()


@needs_vtk
def test_flow_scene_smoke_channels_plume_particles_and_devices(qapp):
    from aerolume.app.widgets.view3d import View3D
    from aerolume.app.widgets.view3d_panel import FlowScene, dye_colours_of, smoke_colour
    from aerolume.viz3d.placed import placed_glyphs
    r = thermal_cfd3d("x")
    sim = thermal_sim("d")
    assert smoke_colour("#f5f5f5") == "auto" and smoke_colour("#E69F00") == "#e69f00"
    assert dye_colours_of(sim) == {"dye": "auto", "dye1": "#e69f00"}
    view = View3D()
    view.resize(800, 600)
    try:
        flow = FlowScene(view, shown={"smoke": True, "heat": True, "particles": True, "devices": True})
        flow.set_bodies(r.surface, key="b")
        flow.set_dye_colors(dye_colours_of(sim), rebuild=False)
        flow.set_smoke(r.extra_volumes[0])
        sm = flow.handles["smoke"]
        assert len(sm) == 2                                  # one layer per channel with smoke in it
        cols = [view.scene.layer_info(h).get("field") for h in sm]
        assert cols == ["dye", "dye1"]
        assert "smoke in 2 colours" in flow.describe()
        snap = r.extra_volumes[0]
        nxt = VolumeData(origin=snap.origin, spacing=snap.spacing, mask=snap.mask, label="instant",
                         fields={k: 0.9 * a for k, a in snap.fields.items()})
        assert flow.set_data(smoke=nxt) == []                # the next frame of the same channels: in place
        assert flow.handles["smoke"] == sm and flow.smoke is nxt
        flow.set_heat(r.volume)
        (hh,) = flow.handles["heat"]
        assert view.scene.layers()[hh] == "volume"
        lo, hi = flow.heat_colour_range()
        assert lo == pytest.approx(T_AMB, abs=0.05) and hi > T_AMB + 0.8 * (T_BODY - T_AMB)
        assert "thermal plume T ≥" in flow.describe() and "°C" in flow.describe()
        assert view.scene.colorbar_spec() is not None and view.scene.colorbar_spec().title.startswith("Temperature")
        flow.set_heat(r.volume, style="surface", level=40.0)
        (hs,) = flow.handles["heat"]
        assert view.scene.layers()[hs] == "isosurface" and "iso-surface T = 40" in flow.describe()
        with pytest.raises(ValueError):
            flow.set_heat(r.volume, style="cloud")
        pts = np.column_stack([np.linspace(-1, 2, 30), np.zeros(30), np.zeros(30)])
        flow.set_particles(pts, colors=np.tile([[61, 139, 232]], (30, 1)))
        (hp,) = flow.handles["particles"]
        flow.set_particles(pts[:5], colors=np.tile([[61, 139, 232]], (5, 1)))
        assert flow.handles["particles"] == [hp]              # moved in place, not rebuilt
        assert "5 particles" in flow.describe()
        flow.set_devices(placed_glyphs(sim, scale=1.0))
        kinds = {view.scene.layers()[h] for h in flow.handles["devices"]}
        assert kinds == {"glyph"} and len(flow.handles["devices"]) >= 4 * 1
        flow.show("devices", False)
        assert not flow.handles.get("devices")
        flow.set_heat(None)
        flow.set_particles(None)
        assert flow.available("heat")[0] is False and "heat transfer is off" in flow.available("heat")[1]
        assert not flow.handles.get("heat") and not flow.handles.get("particles")
        assert not flow.errors, flow.errors
    finally:
        view.close()


# ---------------------------------------------------------------------------
# Results 3D
# ---------------------------------------------------------------------------

@needs_vtk
def test_results_3d_heat_fans_diagnostics_and_the_plume_toggle(qapp, state_heat, tmp_path):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = state_heat
    state.select_simulation("h3d")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    try:
        assert pump_until(qapp, lambda: page.view3d is not None and page.view3d.result is not None, 10)
        v = page.view3d
        tabs = v.tabs
        for key in ("heat", "fans", "diagnostics"):
            assert tabs.isTabVisible(v.TABS.index(key)), key
        # the Summary carries Nu with its standard error
        texts = [v.summary_table.item(i, 0).text() for i in range(v.summary_table.rowCount())]
        assert any("Nusselt" in t for t in texts)
        assert v.select_tab("heat")
        pump(qapp, 0.3)
        hp = v.heat_panel
        assert hp is not None and hp.table.rowCount() == 2
        assert hp.body_table.rowCount() == 1 and hp.body_table.item(0, 1).text() == "60 °C"
        assert hp.balance_table.rowCount() >= 3 and hp.view.has_image()
        assert "Ambient 20 °C" in hp.note.text()
        hp.axis_seg.set_current("z")
        pump(qapp, 0.1)
        assert hp.view.has_image() and hp.current_slice().axis == "z"
        assert v.select_tab("fans")
        pump(qapp, 0.2)
        assert v.fan_table.rowCount() == 1 and v.fan_table.item(0, 0).text() == "Fan"
        assert v.select_tab("diagnostics")
        pump(qapp, 0.2)
        assert v.diag_table.rowCount() >= 3 and len(v.diag_panel.log) > 0
        # the plume button opens Fields with the thermal plume on and devices drawn
        assert v.select_tab("heat")
        hp.plume_btn.click()
        pump(qapp, 0.5)
        assert v.current_tab() == "fields"
        fp = v.fields_panel
        b = fp.toggles.buttons["heat"]
        assert not b.isHidden() and b.isChecked() and b.isEnabled()
        # at 1600 px the View control and the seven layer toggles share one row; narrower they take two
        assert pump_until(qapp, lambda: fp.bar_rows() == 1, 3), fp.bar_host.width()
        page.resize(1100, 800)
        assert pump_until(qapp, lambda: fp.bar_rows() == 2, 3)
        for btn in fp.toggles.buttons.values():
            if not btn.isHidden():
                assert btn.width() >= btn.sizeHint().width() - 2, (btn.text(), btn.width())
        page.resize(1600, 1000)
        pump(qapp, 0.3)
        assert fp.heat_group.isVisibleTo(fp) and fp.flow.handles.get("heat")
        assert fp.flow.handles.get("devices") and len(fp.flow.handles.get("smoke", [])) == 2
        assert "temperature" in [fp.field_combo.itemData(i) for i in range(fp.field_combo.count())]
        fp.heat_style.set_current("surface")
        fp.heat_slider.setValue(50)
        assert pump_until(qapp, lambda: fp.flow.heat_style == "surface"
                          and fp.view.scene.layers().get((fp.flow.handles.get("heat") or [""])[0]) == "isosurface", 5)
        lo, hi = fp._heat_range                              # 20 °C to the hottest fluid (99.5th percentile)
        assert lo == pytest.approx(T_AMB, abs=0.1) and 55.0 < hi <= T_BODY
        from aerolume.core.units import sig
        assert fp.heat_value.text() == f"{sig(lo + 0.5 * (hi - lo), 3)} °C"
        # the particle replay: the run's particle emitter releases tracers into the stored mean flow
        pb = fp.toggles.buttons["particles"]
        assert pb.isEnabled() and not pb.isChecked()
        pb.setChecked(True)
        assert pump_until(qapp, lambda: fp.flow.particles is not None and len(fp.flow.particles["positions"]) > 20,
                          30), "no replay particles"
        pts = fp.flow.particles["positions"]
        assert pts[:, 0].min() >= -1.5 - 1e-9 and pts[:, 0].max() <= 2.5 + 1e-9      # inside the stored volume
        assert fp.flow.particles["colors"][0].tolist() == [61, 139, 232]            # the emitter's own colour
        assert fp.flow.handles.get("particles")
        # an isothermal run: no plume toggle at all, no heat tabs
        state.select_simulation("c3d")
        assert pump_until(qapp, lambda: page.view3d.result is not None and page.view3d.result.sim_id == "c3d", 10)
        for key in ("heat", "fans", "diagnostics"):
            assert not tabs.isTabVisible(v.TABS.index(key)), key
        v.select_tab("fields")
        pump(qapp, 0.3)
        assert fp.toggles.buttons["heat"].isHidden() and not fp.heat_group.isVisibleTo(fp)
        assert not fp.flow.handles.get("heat")
        assert fp._replay is None and not fp.toggles.buttons["particles"].isChecked()    # a new result stops it
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Simulate 3D live
# ---------------------------------------------------------------------------

@needs_vtk
def test_simulate_live_3d_frame_feeds_plume_particles_and_devices(qapp, state_heat, monkeypatch):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state_heat
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("h3n")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.3)
    try:
        tg = page.flow_toggles
        assert not tg.buttons["heat"].isHidden() and not tg.buttons["particles"].isHidden()
        monkeypatch.setattr(jobs, "running_id", lambda: "h3n")
        page._shown_sim_id = "h3n"
        r = thermal_cfd3d("h3n")
        f = slice_frame("y")
        vol = r.volume
        f.volume = VolumeData(origin=vol.origin, spacing=vol.spacing, mask=vol.mask, label="instant",
                              fields={"temperature": vol.fields["temperature"], **r.extra_volumes[0].fields})
        f.volume_serial = 1
        f.particles = np.column_stack([np.linspace(-1, 2, 40), np.zeros(40), np.full(40, -0.2)])
        f.particle_ids = np.zeros(40, dtype=int)
        f.particle_palette = ["#3d8be8"]
        jobs.jobFrame.emit("h3n", f)
        assert pump_until(qapp, lambda: page.flow is not None and page.flow.handles.get("heat")
                          and page.flow.handles.get("particles"), 5)
        flow = page.flow
        # the live plume's colours are the case's own temperatures, fixed for the run: the 20 °C ambient
        # and the sphere's 60 °C wall (not the data's percentiles, which drift as the plume grows)
        assert flow.heat_colour_range() == (T_AMB, T_BODY)
        assert flow.particles["colors"][0].tolist() == [61, 139, 232]
        assert flow.handles.get("devices")                      # the particle emitter's glyph
        assert tg.buttons["heat"].isEnabled() and tg.buttons["particles"].isEnabled()
        assert getattr(jobs.view, "temperature3d", None) is True
        # a simulation without heat or particles: those toggles go
        state.select_simulation("c3d")
        page._sync_physics3d(state.simulation, page._design_of(state.simulation))
        assert tg.buttons["heat"].isHidden() and tg.buttons["particles"].isHidden()
    finally:
        page.shutdown()
        page.close()
