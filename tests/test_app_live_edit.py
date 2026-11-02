"""Live editing and diagnostics (DESIGN.md 10.4): obstacle sketches, placements, the diagnostics log,
solver traces and the explore worker's hand-over to (and fallbacks for) a live session.

The sketch and the logs are plain data and are checked against exact geometry; the worker is driven
with a small fake session that records what it is posted and what its solver is asked to do.
"""
from __future__ import annotations

import itertools
import math
import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from aerolume.model import Design, Emitter, Fan, HeatSource, Port, Simulation

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


def _wait(app, cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return cond()


# ---------------------------------------------------------------- obstacle sketch
EXT = (0.0, 2.0, 0.0, 1.0)
SHAPE = (100, 200)          # h = 0.01 m


def test_brush_stroke_rasterizes_to_a_capsule_with_one_cell_ramp():
    from aerolume.app.widgets.draw_tools import ObstacleSketch
    s = ObstacleSketch()
    s.add_stroke([(0.5, 0.5), (1.5, 0.5)], radius=0.1)
    m = s.rasterize(EXT, SHAPE)
    assert m.shape == SHAPE and m.dtype == np.float32
    # area of a capsule: 2 r L + pi r^2 = 0.2 + 0.0314 m^2, cells of 1e-4 m^2
    assert m.sum() * 1e-4 == pytest.approx(0.2 + math.pi * 0.01, rel=0.01)
    assert m[50, 100] == 1.0 and m[50, 20] == 0.0                   # inside / outside
    edge = m[60:61, 100][0]                                           # the cell centred 0.105 m off the axis
    assert 0.0 <= edge <= 1.0


def test_erase_polygon_undo_redo_and_clear():
    from aerolume.app.widgets.draw_tools import ObstacleSketch
    s = ObstacleSketch()
    op = s.add_polygon([(0.5, 0.25), (1.5, 0.25), (1.5, 0.75), (0.5, 0.75)])
    assert op is not None and s.add_polygon([(0, 0), (1, 1)]) is None   # degenerate: refused
    full = s.rasterize(EXT, SHAPE).sum() * 1e-4
    assert full == pytest.approx(0.5, rel=0.01)
    s.add_stroke([(1.0, 0.0), (1.0, 1.0)], radius=0.05, erase=True)       # cut it in two
    cut = s.rasterize(EXT, SHAPE)
    assert cut.sum() * 1e-4 == pytest.approx(0.5 - 0.1 * 0.5, rel=0.03)
    polys = s.polygons(EXT, SHAPE)
    assert len(polys) == 2 and all(p.shape[1] == 2 for p in polys)
    rev = s.revision
    assert s.undo() and s.revision > rev
    assert s.rasterize(EXT, SHAPE).sum() * 1e-4 == pytest.approx(full, rel=1e-6)
    assert s.redo() and len(s.ops) == 2
    s.clear()
    assert s.is_empty() and s.can_redo()
    assert s.undo() is False
    assert s.redo() and not s.is_empty()                                 # clear is undoable


def test_sketch_round_trip_and_payload():
    from aerolume.app.widgets.draw_tools import ObstacleSketch
    s = ObstacleSketch()
    s.add_stroke([(0.3, 0.3), (0.6, 0.6)], radius=0.04)
    t = ObstacleSketch.from_dict(s.to_dict())
    assert np.array_equal(t.rasterize(EXT, SHAPE), s.rasterize(EXT, SHAPE))
    p = s.payload(EXT, (50, 100))
    assert p["mask"].shape == (50, 100) and p["polygons"] and p["revision"] == s.revision
    from aerolume.geometry import polygon as P
    assert all(P.signed_area(q) > 0 for q in p["polygons"])                # CCW, as Body outlines are


# ---------------------------------------------------------------- placements
def _sim_with_placements():
    sim = Simulation("s", "d")
    sim.conditions.alpha_deg = 10.0
    sim.emitters = [Emitter(name="Rake", kind="rake", position=(-0.5, -0.2), end=(-0.5, 0.2), count=5,
                            color="#e69f00"),
                    Emitter(name="Dot", species="particles", kind="point", position=(1.0, 0.0), frame="body")]
    sim.fans = [Fan(name="Fan", position=(1.0, 0.1, 0.0), direction=(1.0, 0.0, 0.0), diameter=0.2, frame="body")]
    sim.heat_sources = [HeatSource(name="Heater", kind="rect", position=(0.5, 0.0), size=(0.1, 0.05))]
    sim.ports = [Port(name="Vent", kind="vent", position=(2.0, 0.0), direction=(0.0, 1.0), size=0.05,
                      frame="wind")]
    design = Design(id="d", name="d", kind="generic", bodies=[], length_ref=1.0, force_ref=1.0, moment_ref=(0.25, 0.0))
    return sim, design


def test_placements_turn_with_the_body_and_moves_go_back_to_the_model():
    from aerolume.app.widgets.draw_tools import apply_move, placements_of
    sim, design = _sim_with_placements()
    pls = {p.key: p for p in placements_of(sim, design)}
    assert set(pls) == {"emitters:0", "emitters:1", "fans:0", "heat_sources:0", "ports:0"}
    # a wind-frame item stays put; a body-frame one turns by -alpha about the moment reference
    assert (pls["emitters:0"].x, pls["emitters:0"].y) == (-0.5, -0.2)
    a = math.radians(-10.0)
    x, y = 0.25 + math.cos(a) * 0.75, math.sin(a) * 0.75
    assert pls["emitters:1"].x == pytest.approx(x) and pls["emitters:1"].y == pytest.approx(y)
    assert pls["fans:0"].direction[1] == pytest.approx(math.sin(a))
    assert pls["ports:0"].kind == "vent" and pls["emitters:0"].count == 5
    # moving the body-frame emitter to a wind-frame point stores the body-frame position
    assert apply_move(sim, "emitters:1", 0.25, -0.75, design)
    p = sim.emitters[1].position
    back = placements_of(sim, design)[1]
    assert (back.x, back.y) == pytest.approx((0.25, -0.75))
    assert len(p) == 2
    # a rake keeps its length; a 3-component position keeps its z
    assert apply_move(sim, "emitters:0", -1.0, 0.0, design)
    e = sim.emitters[0]
    assert e.position == pytest.approx((-1.0, 0.0)) and e.end == pytest.approx((-1.0, 0.4))
    assert apply_move(sim, "fans:0", 1.2, 0.2, design) and len(sim.fans[0].position) == 3
    assert not apply_move(sim, "fans:7", 0, 0, design) and not apply_move(sim, "nonsense", 0, 0, design)


def test_field_view_paints_glyphs_tools_and_coloured_particles(qapp):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from aerolume.app.widgets.draw_tools import MoveTool, ObstacleSketch, SketchTool, placements_of
    from aerolume.app.widgets.field_view import FieldView
    sim, design = _sim_with_placements()
    v = FieldView()
    v.resize(800, 400)
    vals = np.linspace(0, 1, 200 * 100, dtype=np.float32).reshape(100, 200)
    v.set_field(vals, (-1.0, 3.0, -1.0, 1.0), field="speed")
    v.set_bodies([np.array([[0, -0.05], [1, -0.05], [1, 0.05], [0, 0.05]], float)])
    v.set_placements(placements_of(sim, design))
    rng = np.random.default_rng(0)
    pts = rng.uniform([-1, -1], [3, 1], (300, 2))
    trails = np.stack([pts - [0.02 * k, 0] for k in range(5, 0, -1)], axis=1)
    trails[:10, :2] = np.nan
    v.set_particles(pts, colors=np.tile([[230, 159, 0]], (300, 1)).astype(np.uint8), trails=trails)
    v.set_legend([("#f5f5f5", "Rake"), ("#e69f00", "Jet")])
    v.show()
    qapp.processEvents()
    img = v.grab().toImage()
    assert img.width() >= 800
    # drag the heater with the move tool: the callback receives the new wind-frame position
    moved = []
    tool = MoveTool(placements_of(sim, design), on_move=lambda k, x, y, final: moved.append((k, x, y, final)))
    v.set_tool(tool)
    xf = v.world_to_screen()
    start = xf.map(QPointF(0.5, 0.0))
    end = xf.map(QPointF(0.7, 0.3))

    def ev(kind, pos, buttons):
        return QMouseEvent(kind, pos, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)

    v.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton))
    v.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton))
    v.mouseReleaseEvent(ev(QMouseEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton))
    assert moved and moved[-1][0] == "heat_sources:0" and moved[-1][3] is True
    # the heater is body-fixed (turned by -10°): its handle keeps its offset from where it was grabbed
    h = next(p for p in placements_of(sim, design) if p.key == "heat_sources:0")
    assert moved[-1][1] == pytest.approx(0.7 + h.x - 0.5, abs=0.01)
    assert moved[-1][2] == pytest.approx(0.3 + h.y - 0.0, abs=0.01)
    # draw a stroke with the brush: it is committed on release, the view did not pan
    sketch = ObstacleSketch()
    changes = []
    brush = SketchTool(sketch, mode="brush", radius=0.05, on_change=changes.append)
    v.set_tool(brush)
    before = v.view_state()
    a, b = xf.map(QPointF(1.5, 0.5)), xf.map(QPointF(2.2, 0.5))
    v.mousePressEvent(ev(QMouseEvent.Type.MouseButtonPress, a, Qt.MouseButton.LeftButton))
    for k in range(1, 11):
        p = a + (b - a) * (k / 10)
        v.mouseMoveEvent(ev(QMouseEvent.Type.MouseMove, p, Qt.MouseButton.LeftButton))
    v.mouseReleaseEvent(ev(QMouseEvent.Type.MouseButtonRelease, b, Qt.MouseButton.NoButton))
    assert v.view_state() == before and len(sketch.ops) == 1 and changes[-1] is True and False in changes
    stroke = sketch.ops[0].points
    assert stroke[0] == pytest.approx((1.5, 0.5), abs=0.01) and stroke[-1][0] == pytest.approx(2.2, abs=0.02)
    v.set_sketch(sketch.polygons((-1, 3, -1, 1), (100, 200)))
    v.grab()
    v.close()


def test_default_framing_takes_in_the_placed_devices_and_a_drag_does_not_move_the_view(qapp):
    """A rake two body lengths upstream and a heater far in the wake lie outside the body's near-field
    window; the default framing grows to show them, and re-placing a glyph without ``refit`` keeps the view."""
    from PySide6.QtCore import QPointF

    from aerolume.app.widgets.draw_tools import placements_of
    from aerolume.app.widgets.field_view import FieldView, placements_box
    sim = Simulation("s", "d")
    sim.emitters = [Emitter(name="Rake", kind="rake", position=(-0.25, -0.05), end=(-0.25, 0.05), count=5)]
    sim.heat_sources = [HeatSource(name="H", kind="rect", position=(0.6, -0.12), size=(0.05, 0.02))]
    design = Design(id="d", name="d", kind="generic", bodies=[], length_ref=0.1, force_ref=1.0, moment_ref=(0.0, 0.0))
    v = FieldView()
    v.resize(900, 700)
    v.show()
    v.set_field(np.zeros((240, 600), np.float32), (-1.0, 2.0, -0.6, 0.6), field="speed", vmin=0.0, vmax=1.0)
    v.set_bodies([np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])])
    qapp.processEvents()
    pls = placements_of(sim, design)
    box = placements_box(pls)
    assert box[0] == pytest.approx(-0.25, abs=1e-9) and box[1] == pytest.approx(0.625, abs=1e-9)
    xf = v.world_to_screen()
    assert not v.plot_rect().contains(xf.map(QPointF(0.6, -0.12)))      # the near field alone misses it
    v.set_placements(pls, refit=True)
    xf, plot = v.world_to_screen(), v.plot_rect()
    for x, y in ((-0.25, -0.05), (-0.25, 0.05), (0.6, -0.12), (0.0, 0.0)):
        assert plot.contains(xf.map(QPointF(x, y))), (x, y)
    state = v.view_state()
    sim.heat_sources[0].position = (1.5, 0.4)
    v.set_placements(placements_of(sim, design))                       # a drag: the view stays
    assert v.view_state() == state
    assert placements_box([]) is None
    v.close()


def test_a_3d_frame_with_tracer_particles_shows_the_ones_on_its_slice(qapp):
    """A 3D frame carries (N, 3) particles; its slice view once crashed on every frame reshaping them to
    two columns. The slice draws those within a few cells of its plane, in its own (column, row) axes."""
    from types import SimpleNamespace

    from aerolume.app.widgets.field_view import SLICE_PARTICLE_BAND_PX, FieldView, frame_particles2d
    rgba = np.zeros((50, 100, 4), np.uint8)
    rgba[..., 3] = 255
    pts = np.array([[0.5, 0.0, 0.2], [0.6, 0.009, -0.1], [0.7, 0.5, 0.3], [0.8, -0.011, 0.0]])
    f = SimpleNamespace(rgba=rgba, extent=(-1.0, 1.0, -0.5, 0.5), field="speed", cmap=None, vmin=0.0, vmax=1.0,
                        units=None, slice_axis="y", slice_pos=0.0, bodies=[], streamlines=None, particles=pts,
                        particle_ids=np.array([0, 0, 1, 1]), particle_palette=["#3d8be8", "#e69f00"])
    cell = 2.0 / 100                                       # one image pixel: 0.02 m
    assert SLICE_PARTICLE_BAND_PX * cell > np.abs(pts[[0, 1, 3], 1]).max()   # the near points are inside the band
    xy, cols, trails = frame_particles2d(f)
    assert xy.tolist() == [[0.5, 0.2], [0.6, -0.1], [0.8, 0.0]]          # (x, z) of a y-slice; y = 0.5 is not
    assert cols.tolist() == [[61, 139, 232], [61, 139, 232], [230, 159, 0]] and trails is None
    f.slice_axis, f.slice_pos = "z", 0.3                    # a plan view: (x, y) of the points near z = 0.3
    assert frame_particles2d(f)[0].tolist() == [[0.7, 0.5]]
    v = FieldView()
    v.resize(400, 240)
    v.set_frame(f)                                          # no crash on (N, 3)
    v.set_particles(pts)                                    # nor here: the first two axes are drawn
    v.grab()
    v.close()
    assert frame_particles2d(SimpleNamespace(particles=np.zeros((0, 3))))[0] is None


def test_a_live_temperature_bar_does_not_claim_fluid_below_the_ambient():
    """A live temperature frame carries colours, not values; most of its fluid sits exactly at the range's
    floor (the ambient), which read as "clipped below" and drew an extend triangle under every live bar.
    Overheated fluid above the top still extends."""
    from aerolume.app.widgets.field_view import colorize, colormap_lut, frame_clipped_ends
    lut = colormap_lut("thermal", "temperature")
    T = np.full((60, 120), 20.0)
    T[20:40, 30:60] = np.linspace(20.0, 75.0, 30)[None, :]      # a plume, hotter than the 60 °C top
    rgba = colorize(T, lut, 20.0, 60.0)
    assert frame_clipped_ends(rgba, lut, field="temperature") == (False, True)
    assert frame_clipped_ends(rgba, lut, field="vorticity")[0] is True    # a signed field still extends


def test_a_heater_beside_the_window_does_not_hatch_the_colour_bar(qapp):
    """Glyphs are clipped to the plot area: the heater's hatching once replaced the view's clip with the
    heater's own outline, so a heater just right of the window hatched the colour-bar strip."""
    from aerolume.app.widgets.draw_tools import placements_of
    from aerolume.app.widgets.field_view import FieldView
    sim = Simulation("s", "d")
    design = Design(id="d", name="d", kind="generic", bodies=[], length_ref=1.0, force_ref=1.0, moment_ref=(0.0, 0.0))
    v = FieldView()
    v.resize(800, 400)
    v.set_field(np.zeros((100, 400), np.float32), (-1.0, 3.0, -1.0, 1.0), field="speed", vmin=0.0, vmax=1.0)
    v.show()
    qapp.processEvents()
    v.set_view_state((200.0, 0.0, 0.0))                # 1 m = 200 px, centred on the origin
    strip_left = v.plot_rect().right()
    x_world = (strip_left + 30 - v.plot_rect().center().x()) / 200.0     # 30 px into the colour-bar strip
    clean = v.grab().toImage()
    sim.heat_sources = [HeatSource(name="H", kind="rect", position=(x_world, -0.6), size=(0.4, 0.2))]
    v.set_placements(placements_of(sim, design))
    hatched = v.grab().toImage()
    y0 = int(v.plot_rect().center().y() + 0.6 * 200 - 25)
    changed = [(x, y) for x in range(int(strip_left) + 2, v.width(), 2) for y in range(y0, y0 + 50, 2)
               if clean.pixel(x, y) != hatched.pixel(x, y)]
    assert not changed, changed[:5]
    v.close()


# ---------------------------------------------------------------- diagnostics
class _Frame:
    def __init__(self, step, t, diag, sps=500.0):
        self.step, self.t, self.diagnostics, self.steps_per_s = step, t, diag, sps


def test_diagnostics_log_records_frames_restarts_and_writes_csv(tmp_path):
    from aerolume.app.widgets.diagnostics import DiagnosticsLog, describe
    log = DiagnosticsLog(max_rows=100)
    for k in range(120):
        log.add_frame(_Frame(10 * k, 0.01 * k, {"div_rms": 10.0 ** (-k / 20), "cfl": 0.5}))
    snap = log.snapshot()
    assert len(snap["step"]) == 100 and snap["step"][0] == 200          # bounded: the oldest rows dropped
    assert set(log.keys()) == {"div_rms", "cfl", "steps_per_s"} and log.keys()[0] == "div_rms"
    log.add_frame(_Frame(1190, 11.9, {"du_rel": 1e-3}))                  # a key first seen late
    assert np.isnan(log.snapshot()["du_rel"][0]) and log.latest()["du_rel"] == 1e-3
    log.add_frame(_Frame(0, 0.0, {"div_rms": 1.0}))                      # a reset starts a new table
    assert len(log) == 1
    log.recording = False
    assert not log.add(5, 0.1, {"cfl": 0.3}) and len(log) == 1
    log.recording = True
    log.add(10, 0.2, {"cfl": 0.31})
    path = log.to_csv(tmp_path / "d.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("step,t,div_rms") and len(lines) == 3
    assert describe("dtheta_rel").group == "Change per step" and describe("foo_residual").log
    assert describe("du_l2").label == "L2 change of u" and describe("mystery").group == "Other"
    assert describe("dtheta_l2").label == "L2 change of θ" and describe("dw_max").unit == "U"


def test_diagnostics_rows_that_repeat_their_own_step_keep_every_column_aligned(qapp):
    """``FlowSolver.latest_diagnostics()`` carries the row's "step" and "t" (and "perf_steps"): they are
    the row's index, not quantities. Taken as values they grew the step column by two per row and the
    panel failed every redraw on mismatched lengths (found with the real ns2d solver)."""
    from aerolume.app.widgets.diagnostics import DiagnosticsLog, DiagnosticsPanel
    log = DiagnosticsLog()
    for k in range(50):
        row = {"step": 10.0 * k, "t": 0.02 * k, "dt": 0.002, "cfl": 0.45, "div_rms": 1e-6, "perf_steps": 10.0,
               "steps_per_s": 480.0}
        log.add_frame(_Frame(10 * k, 0.02 * k, row))
    snap = log.snapshot()
    assert {len(v) for v in snap.values()} == {50}
    quantities = log.keys()                     # the log's recorded quantities (a list), not dict keys
    assert "step" not in quantities and "perf_steps" not in quantities and "dt" in quantities
    assert np.array_equal(snap["step"], 10.0 * np.arange(50))
    panel = DiagnosticsPanel(log=log)
    panel.resize(900, 300)
    panel.show()
    panel.refresh(force=True)                    # no broadcast error
    panel.close()


def test_diagnostics_plots_flow_into_columns_and_plot_relative_changes(qapp):
    """A wide, short bottom dock gets its plot groups side by side (each at least MIN_PLOT_H tall), a
    tall side dock stacks them; "Change per step" draws the relative changes only."""
    from aerolume.app.widgets.diagnostics import MIN_PLOT_H, DiagnosticsLog, DiagnosticsPanel
    assert DiagnosticsPanel.plotted_keys(["du_rel", "du_l2", "dv_max", "dv_rel", "div_rms"]) == \
        ["du_rel", "dv_rel", "div_rms"]
    assert DiagnosticsPanel.plotted_keys(["du_l2", "div_rms"]) == ["du_l2", "div_rms"]   # nothing relative: all
    log = DiagnosticsLog()
    for k in range(30):
        log.add(10 * k, 0.02 * k, {"div_rms": 1e-6, "div_max": 3e-6, "du_rel": 1e-3, "du_l2": 1e-2, "dv_rel": 2e-3,
                                   "p_residual": 1e-9, "cfl": 0.4, "dt": 0.002, "steps_per_s": 400.0})
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1500, 260)
        panel.show()
        panel.refresh(force=True)
        _wait(qapp, lambda: False, 0.2)
        rows, cols = panel.grid_shape()
        assert len(panel._panels) == 6 and cols >= 3 and rows * cols >= 6
        assert set(panel._curves) == {"div_rms", "div_max", "du_rel", "dv_rel", "p_residual", "cfl", "dt",
                                      "steps_per_s"}
        assert all(pi.height() >= MIN_PLOT_H - 30 for pi in panel._panels.values())   # less the title line
        assert "rms" in panel._panels["Divergence"].titleLabel.text
        panel.resize(420, 900)
        assert _wait(qapp, lambda: panel.grid_shape()[1] == 1, 2.0)                                    # a tall dock stacks them
    finally:
        panel.close()


def test_diagnostics_of_the_3d_solver_fold_aliases_and_update_in_place(qapp):
    """The 3D solver's rows (``solvers.ns3d.solver.DIAG_KEYS``) carry the pressure-solve residual twice
    (``poisson_residual`` and ``p_residual``), plus ``umax`` and ``heat_storage``: one residual curve, the two
    extras in their own groups, and the Latest list keeps its labels across refreshes (rebuilt five times
    a second it flickered, and a repaint in between drew it empty)."""
    from aerolume.app.widgets.diagnostics import DiagnosticsLog, DiagnosticsPanel, canonical, describe
    assert canonical({"poisson_residual": 1e-5, "p_residual": 1e-5, "cfl": 0.4}) == {"p_residual": 1e-5, "cfl": 0.4}
    assert canonical({"poisson_residual": 2e-5}) == {"p_residual": 2e-5}
    assert describe("umax").group == "Velocity" and describe("heat_storage").group == "Heat"
    log = DiagnosticsLog()
    for k in range(1, 40):
        log.add_frame(_Frame(10 * k, 0.01 * k, {"step": 10.0 * k, "t": 0.01 * k, "dt": 0.01, "cfl": 0.45, "umax": 1.6,
                                                 "div_rms": 1e-6, "div_max": 5e-6, "poisson_residual": 1e-5,
                                                 "p_residual": 1e-5, "du_rel": 1e-3, "dw_rel": 2e-3,
                                                 "heat_storage": 0.3}))
    recorded = log.keys()                         # a list of quantities, not a dict view
    assert "poisson_residual" not in recorded and "p_residual" in recorded
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1500, 260)
        panel.show()
        panel.refresh(force=True)
        qapp.processEvents()
        assert sum(describe(k).group == "Residual" for k in panel._curves) == 1
        assert {"Velocity", "Heat"} <= set(panel._panels) and "Other" not in panel._panels
        before = [id(v) for _k, v in panel.values._rows]
        log.add_frame(_Frame(400, 0.4, {"dt": 0.011, "cfl": 0.46, "umax": 1.7, "div_rms": 2e-6, "div_max": 6e-6,
                                        "p_residual": 1e-5, "du_rel": 1e-3, "dw_rel": 2e-3, "heat_storage": 0.31}))
        panel.refresh()
        assert [id(v) for _k, v in panel.values._rows] == before          # same labels, new texts
        assert panel.values.value_text("Time step Δt") == "0.0110 L/U"
        assert all(v.isVisible() for _k, v in panel.values._rows)
        # every value ends inside the visible list (a long key once pushed them past its right edge)
        qapp.processEvents()
        vp = panel.side_scroll.viewport()
        for _k, v in panel.values._rows:
            right = v.mapTo(vp, v.rect().topRight()).x()
            assert right <= vp.width(), (v.text(), right, vp.width())
            assert v.width() >= v.sizeHint().width()
    finally:
        panel.close()


def test_diagnostics_axes_widen_for_long_tick_labels_and_stay_aligned(qapp):
    """A time step wandering between 0.0064 and 0.0070 has six-character tick labels: the left axes widen
    to the rotated name plus the widest label (they ran into "Δt (L/U)" at a fixed 54 px), all alike."""
    from aerolume.app.widgets.diagnostics import DiagnosticsLog, DiagnosticsPanel
    log = DiagnosticsLog()
    for k in range(60):
        log.add(10 * k, 0.02 * k, {"div_rms": 1e-6, "cfl": 0.40 + 0.004 * math.sin(k / 5),
                                   "dt": 0.0067 + 0.0003 * math.sin(k / 4), "steps_per_s": 100.0 + k})
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1300, 240)
        panel.show()
        panel.refresh(force=True)
        _wait(qapp, lambda: False, 0.4)                  # paint (the labels get measured), then the re-fit
        axes = [pi.getAxis("left") for pi in panel._panels.values()]
        dt_axis = panel._panels["Time step"].getAxis("left")
        assert dt_axis.textWidth > 30                    # "0.0070" was measured
        need = panel.needed_axis_width()
        assert need > DiagnosticsPanel.AXIS_MIN_W
        assert {ax.fixedWidth for ax in axes} == {need}                  # one width: frames aligned
        lab = dt_axis.label.boundingRect().height() * 0.8
        assert dt_axis.fixedWidth >= dt_axis.textWidth + lab + max(0, dt_axis.style["tickLength"])
    finally:
        panel.close()


def test_diagnostics_plots_wrap_into_rows_at_their_minimum_width(qapp):
    """At 1280 x 800 the bottom dock is about 975 px wide: eight groups in one row were 20 px wide. Each
    plot keeps MIN_PLOT_W, the groups wrap into further rows, and the rows scroll when the dock is short."""
    from aerolume.app.widgets.diagnostics import MIN_PLOT_H, MIN_PLOT_W, DiagnosticsLog, DiagnosticsPanel
    log = DiagnosticsLog()
    for k in range(1, 60):
        log.add(10 * k, 0.01 * k, {"div_rms": 1e-6, "div_max": 5e-6, "du_rel": 1e-3, "p_residual": 1e-5,
                                   "cfl": 0.5, "dt": 0.004 + 1e-5 * math.sin(k), "umax": 3.2 + 0.1 * math.sin(k / 3),
                                   "steps_per_s": 400.0 + k})
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1280, 200)
        panel.show()
        panel.refresh(force=True)
        _wait(qapp, lambda: False, 0.3)
        rows, cols = panel.grid_shape()
        assert rows * cols >= len(panel._panels) and rows >= 2
        for pi in panel._panels.values():
            assert pi.width() >= MIN_PLOT_W - 12, pi.width()
        assert panel.plots.minimumHeight() >= rows * MIN_PLOT_H - 1      # taller than the dock: it scrolls
        assert panel._plot_scroll.verticalScrollBar().maximum() > 0
    finally:
        panel.close()


def test_diagnostics_axes_have_three_si_ticks_and_constants_read_as_constants(qapp):
    """Linear axes carry at most three ticks with SI labels ("0", "2k", "4k"), and a series constant to
    float noise (the CFL held at 0.5 read 0.49999998 .. 0.50000006) is drawn as one tick, titled
    "constant 0.500"."""
    from aerolume.app.widgets.diagnostics import (
        DiagnosticsLog,
        DiagnosticsPanel,
        compact_ticks,
        constant_value,
        si_ticks,
    )
    assert si_ticks([0, 2000, 4000], 2000) == ["0", "2k", "4k"]
    assert si_ticks([0.004, 0.0045, 0.005], 0.0005) == ["0.0040", "0.0045", "0.0050"]
    assert si_ticks([1.5e6, 2e6], 5e5) == ["1.5M", "2.0M"]
    assert si_ticks([-1000, 0, 1000], 1000) == ["−1k", "0", "1k"]
    assert 1 < len(compact_ticks(0, 5660)) <= 3 and len(compact_ticks(0.4, 0.48)) <= 3
    assert constant_value([np.full(10, 0.5) + np.float32(3e-8) * np.arange(10)]) == pytest.approx(0.5)
    assert constant_value([np.linspace(0.4, 0.5, 10)]) is None
    log = DiagnosticsLog()
    for k in range(1, 80):
        log.add(70 * k, 0.01 * k, {"div_rms": 1e-6, "cfl": 0.5 + 3e-8 * math.sin(k), "dt": 0.004 + 1e-4 * math.sin(k),
                                   "steps_per_s": 400.0})
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1280, 300)
        panel.show()
        panel.refresh(force=True)
        _wait(qapp, lambda: False, 0.3)
        cfl = panel._panels["CFL"]
        assert "constant 0.500" in cfl.titleLabel.text
        ticks = cfl.getAxis("left")._tickLevels
        assert ticks is not None and len(ticks[0]) == 1
        for g, pi in panel._panels.items():
            ax = pi.getAxis("bottom")
            lo, hi = ax.range
            levels = ax.tickValues(lo, hi, 200)
            assert sum(len(v) for _s, v in levels) <= 3, (g, levels)
        dt_ticks = panel._panels["Time step"].getAxis("left")
        lo, hi = dt_ticks.range
        assert sum(len(v) for _s, v in dt_ticks.tickValues(lo, hi, 80)) <= 3
    finally:
        panel.close()


def test_diagnostics_leave_out_empty_and_thermal_columns_without_heat_transfer(qapp):
    """The 3D solver keeps θ and heat-storage columns (NaN without heat transfer): a car's diagnostics
    listed θ and plotted an empty "heat rate" axis. Empty columns are never shown; θ / heat columns are left
    out for a run without heat transfer even when they hold numbers."""
    from aerolume.app.widgets.diagnostics import DiagnosticsLog, DiagnosticsPanel
    log = DiagnosticsLog()
    for k in range(1, 30):
        log.add(10 * k, 0.01 * k, {"div_rms": 1e-6, "du_rel": 1e-3, "dtheta_rel": math.nan, "heat_storage": math.nan,
                                   "cfl": 0.45})
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1280, 300)
        panel.show()
        panel.refresh(force=True)
        assert "Heat" not in panel._panels and "dtheta_rel" not in panel._curves
        assert "Relative change of θ" not in dict(panel.values.items())
        log.add(400, 0.4, {"div_rms": 1e-6, "du_rel": 1e-3, "dtheta_rel": 1e-4, "heat_storage": 0.2, "cfl": 0.45})
        panel.refresh(force=True)
        assert "Heat" in panel._panels and "dtheta_rel" in panel._curves      # numbers, thermal unknown: shown
        panel.set_thermal(False)
        assert "Heat" not in panel._panels and "dtheta_rel" not in panel._curves
        panel.set_thermal(True)
        assert "Heat" in panel._panels
    finally:
        panel.close()


def test_diagnostics_panel_shows_latest_values_and_exports(qapp, tmp_path):
    from aerolume.app.widgets.diagnostics import DiagnosticsLog, DiagnosticsPanel
    log = DiagnosticsLog()
    panel = DiagnosticsPanel(log=log)
    panel.resize(900, 260)
    panel.show()
    panel.refresh(force=True)
    assert panel.stack.currentIndex() == 0                              # empty state first
    for k in range(50):
        log.add(k * 20, k * 0.02, {"div_rms": 1e-6 * (1 + k), "du_rel": 1e-2 / (1 + k), "dv_rel": 2e-2 / (1 + k),
                                   "cfl": 0.45, "dt": 0.002, "residual": 1e-7})
    panel.refresh(force=True)
    assert panel.stack.currentIndex() == 1
    vals = dict(panel.values.items())
    assert vals["RMS divergence"] == "5.00×10⁻⁵ U/L" and vals["CFL number"] == "0.450"
    assert set(panel._panels) == {"Divergence", "Change per step", "Residual", "CFL", "Time step"}
    out = panel.export_csv(str(tmp_path / "x.csv"))
    assert out and (tmp_path / "x.csv").stat().st_size > 100
    panel.record_btn.setChecked(False)
    assert not log.recording
    panel.grab()
    panel.close()


# ---------------------------------------------------------------- solver trace
def test_trace_forms_normalize_and_stats():
    from aerolume.app.widgets.trace_viewer import normalize_trace, stage_stats
    a = np.ones((8, 16), np.float32)
    t1 = normalize_trace({"extent": (0, 2, 0, 1), "step": 7, "stages": [
        {"name": "Advection–diffusion", "fields": {"u*": a, "v*": a * 0}},
        ("Divergence", a * 1e-3), {"name": "Pressure solve", "field": "phi", "data": -a}]})
    assert [(s.stage, s.field) for s in t1.stages] == [("Advection–diffusion", "u*"), ("Advection–diffusion", "v*"),
                                                       ("Divergence", ""), ("Pressure solve", "phi")]
    assert t1.step == 7 and t1.stages[0].extent == (0.0, 2.0, 0.0, 1.0)
    t2 = normalize_trace({"u*": a, "div": a})
    assert len(t2.stages) == 2 and t2.stages[0].extent == (0.0, 16.0, 0.0, 8.0)    # cell indices without extent
    st = stage_stats(np.array([[3.0, -4.0], [np.nan, 0.0]]))
    assert st["min"] == -4 and st["max"] == 3 and st["rms"] == pytest.approx(math.sqrt(25 / 3))
    assert st["l2"] == pytest.approx(5.0) and st["n"] == 3
    assert normalize_trace(None).stages == []


def test_trace_viewer_panels_link_and_export(qapp, tmp_path):
    from aerolume.app.widgets.trace_viewer import TraceViewer
    tv = TraceViewer()
    tv.resize(1000, 600)
    tv.show()
    y, x = np.mgrid[0:32, 0:64]
    stages = [("Advection–diffusion", {"u*": (1 + 0.1 * np.sin(x / 5)).astype(np.float32)}),
              ("Divergence before", {"div": (1e-2 * np.cos(x / 4) * np.sin(y / 3)).astype(np.float32)}),
              ("Pressure solve", {"phi": (np.sin(x / 9) * np.cos(y / 7)).astype(np.float32)}),
              ("Divergence after", {"div": (1e-7 * np.cos(x / 4)).astype(np.float32)})]
    cap = tv.set_trace({"extent": (0, 2, 0, 1), "step": 1200, "t": 3.5, "stages": stages})
    qapp.processEvents()
    assert len(cap.stages) == 4 and len(tv.views()) == 4 and tv.stack.currentIndex() == 1
    assert tv.table.rowCount() == 4 and "step 1,200" in tv.info.text()
    v0, v1 = tv.views()[0], tv.views()[1]
    v0.zoom_by(2.0, animate=False)
    assert v1.view_state() == pytest.approx(v0.view_state())             # linked zoom
    tv.shared_btn.setChecked(True)                                       # divergence before / after on one scale
    assert tv.views()[1].colour_range() == pytest.approx(tv.views()[3].colour_range())
    out = tv.export_npz(str(tmp_path / "t.npz"))
    with np.load(out) as z:
        assert int(z["step"]) == 1200 and any(k.startswith("03_Divergence_after") for k in z.files)
    tv.grab()
    tv.close()


def test_trace_viewer_names_solver_stages_and_fits_a_short_dock(qapp):
    """``FlowSolver.trace_step`` keys read as stage words ("stage3/explicit_u" -> Advection–diffusion: u*); the
    thumbnails sit in two rows that scroll sideways, each showing its whole field, and the selected panel is
    enlarged beside them (40 panels as 170 x 90 px images with their own colour bars were unreadable)."""
    from aerolume.app.widgets.trace_viewer import TraceViewer, normalize_trace, stage_label
    assert stage_label("stage3/explicit_u") == ("Advection–diffusion", "u*", "Runge–Kutta stage 3")
    assert stage_label("stage1/divergence")[:2] == ("Divergence of u*", "∇·u*")
    assert stage_label("final_theta")[:2] == ("End of step", "θ")
    assert stage_label("my_custom_stage") == ("my_custom_stage", "", "")
    a = np.random.default_rng(0).normal(size=(24, 48)).astype(np.float32)
    keys = ["stage3/explicit_u", "stage3/explicit_v", "stage3/divergence", "stage3/pressure_potential",
            "stage3/correction_u", "stage3/projected_u", "stage3/theta", "final_p"]
    cap = normalize_trace({"extent": (0, 2, 0, 1), "stages": [(k, a) for k in keys]})
    assert [s.stage for s in cap.stages][:3] == ["Advection–diffusion", "Advection–diffusion", "Divergence of u*"]
    assert "solver key stage3/explicit_u" in cap.stages[0].note
    tv = TraceViewer()
    try:
        tv.resize(1500, 240)
        tv.show()
        tv.set_trace(cap)
        # the grouped table: one row per RK stage (stage 3) and the end of the step, one column per quantity
        assert _wait(qapp, lambda: tv.grid_shape()[0] == 2, 2.0), tv.grid_shape()
        assert tv.grouped() and tv.grid_shape() == (2, 8) and tv.table.isHidden()
        _wait(qapp, lambda: False, 0.2)                     # let the sideways scroll bar settle the heights
        from PySide6.QtCore import QRectF
        vp_h = tv.thumb_scroll.viewport().height()
        for v in tv.views():                                # every thumbnail shows its whole field, none is cut off
            assert not v.show_colorbar                      # the scale is the enlarged view's
            assert 2 * v.height() <= vp_h, (v.height(), vp_h)
            img = v.world_to_screen().mapRect(QRectF(0, 0, 2, 1))
            plot = v.plot_rect()
            assert img.top() >= plot.top() - 1.5 and img.bottom() <= plot.bottom() + 1.5, (img, plot)
            assert img.left() >= plot.left() - 1.5 and img.right() <= plot.right() + 1.5, (img, plot)
            assert img.height() >= tv.THUMB_MIN_H - 4
        # click to enlarge: the detail view takes the panel, with its colour bar and title
        assert tv.selected() == 0 and tv.detail.show_colorbar
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        QTest.mouseClick(tv.views()[3], Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(5, 5))
        assert tv.selected() == 3 and tv.detail_title.text().startswith("4 · Pressure solve")
        assert tv.detail.colour_range() == pytest.approx(tv.views()[3].colour_range())
        assert "min" in tv.detail_stats.text() and "RMS" in tv.detail_stats.text()
        tv.setFocus()
        QTest.keyClick(tv, Qt.Key.Key_Right)
        assert tv.selected() == 4
        tv.stats_btn.setChecked(True)
        assert not tv.table.isHidden() and tv.table.rowCount() == 8
        tv.stats_btn.setChecked(False)
        # a trace without RK stages keeps the two-row strip: one row of whole fields in a very short dock
        tv.set_trace({"extent": (0, 2, 0, 1), "stages": [(f"panel {k}", a) for k in range(8)]})
        assert not tv.grouped()
        assert tv._rows_for(90, 8)[0] == 1
        assert tv._rows_for(150, 8)[0] == 2 and tv._rows_for(1000, 8)[0] > 2
        tv.resize(700, 1000)                               # tall and narrow: the enlarged view above the grid
        assert _wait(qapp, lambda: tv.split.orientation() == Qt.Orientation.Vertical, 2.0)
        assert _wait(qapp, lambda: tv.grid_shape()[0] >= 2, 2.0)
    finally:
        tv.close()


def test_trace_table_fits_its_width_and_the_colour_bar_keeps_whole_labels(qapp):
    """The 2D solver's real layout (3 RK stages x 13 quantities and the end of the step) in a 1600 px
    window: the table ran ~40 px past the right edge, its last column cut, because the thumbnail size
    ignored the column titles and the views' fit pad. And the enlarged view's colour bar was 2 px narrower
    than the painter draws in, so "−1" lost its last glyph."""
    from PySide6.QtGui import QFontMetricsF

    from aerolume.app.widgets.field_view import _ui_font, colorbar_width, tick_labels
    from aerolume.app.widgets.trace_viewer import TraceWindow
    rng = np.random.default_rng(2)
    qty = ("explicit_u", "explicit_v", "explicit_theta", "penalised_u", "penalised_v", "divergence",
           "pressure_potential", "correction_u", "correction_v", "projected_u", "projected_v", "theta")
    stages = [(f"stage{k}/{q}", rng.normal(size=(48, 64)).astype(np.float32)) for k in (1, 2, 3) for q in qty]
    stages += [(f"final_{q}", rng.normal(size=(48, 64)).astype(np.float32)) for q in ("u", "v", "theta", "p")]
    from aerolume.app.widgets.trace_viewer import TraceViewer
    host = TraceViewer()
    win = TraceWindow(host)
    try:
        win.resize(1600, 1000)
        win.show()
        tv = win.viewer
        tv.set_trace({"extent": (0, 4, 0, 3), "stages": stages})
        assert _wait(qapp, lambda: tv.grouped() and tv.grid_shape()[0] == 4, 3.0), tv.grid_shape()
        _wait(qapp, lambda: False, 0.3)
        right = max(c.geometry().right() for c in tv._cells + tv._col_heads)
        assert right <= tv.thumb_scroll.viewport().width(), (right, tv.thumb_scroll.viewport().width())
        assert not tv.thumb_scroll.horizontalScrollBar().isVisible()
    finally:
        win.close()
        host.close()
    fm = QFontMetricsF(_ui_font(8.0))
    for lo, hi in ((-1.32, 3.07), (-0.04, 0.04), (-1234.5, 99.0), (0.0, 1.0)):
        need = max(fm.horizontalAdvance(t) for _v, t in tick_labels(lo, hi)) + 42      # the painter's text box
        assert colorbar_width("u*", "", lo, hi) >= need, (lo, hi)


def test_diagnostics_sample_count_keeps_its_width_while_the_run_goes(qapp):
    """The "940 samples · steps 10–9,400" line was resized at every refresh (proportional digits) and ran
    under the Clear button for a frame; its width now changes only with the number of digits."""
    from aerolume.app.widgets.diagnostics import DiagnosticsPanel
    p = DiagnosticsPanel()
    try:
        p.resize(1200, 260)
        p.show()
        widths = []
        for text in ("940 samples · steps 10–9,390", "941 samples · steps 10–9,400", "942 samples · steps 10–9,410"):
            p._set_count_text(text)
            widths.append(p.count_label.minimumWidth())
            assert p.count_label.fontMetrics().horizontalAdvance(text) <= widths[-1]
        assert len(set(widths)) == 1
        p._set_count_text("1,002 samples · steps 10–10,020")
        assert p.count_label.minimumWidth() > widths[0]
    finally:
        p.close()


def test_trace_viewer_with_forty_panels_keeps_labels_apart(qapp):
    """The 2D solver's trace has 40 panels: two rows of thumbnails without colour bars, and the enlarged
    view's colour bar drawing only as many tick labels as its height holds (they overlapped at 90 px)."""
    from aerolume.app.widgets.trace_viewer import TraceViewer
    rng = np.random.default_rng(1)
    stages = [(f"stage{k % 3 + 1}/explicit_u", (rng.normal(size=(48, 64)) * 0.04).astype(np.float32))
              for k in range(40)]
    tv = TraceViewer()
    try:
        tv.resize(1600, 170)
        tv.show()
        tv.set_trace({"extent": (0, 4, 0, 3), "stages": stages})
        # grouped by RK stage (rows), a quantity's repeats in further columns
        assert _wait(qapp, lambda: tv.grid_shape() == (3, 14), 2.0), tv.grid_shape()
        assert len(tv.views()) == 40 and all(not v.show_colorbar for v in tv.views())
        _wait(qapp, lambda: False, 0.2)
        img = tv.grab().toImage()
        assert img.width() == 1600
        # colour-bar tick labels never overlap, however short the bar (n chosen from its height)
        from aerolume.app.widgets.field_view import colorbar_ticks
        for bar_h in (40.0, 60.0, 90.0, 300.0):
            ticks = colorbar_ticks(-0.04, 0.04, bar_h, 13.0)
            assert len(ticks) >= 2
            offs = [o for _v, _t, o in ticks]
            assert all(b - a >= 13.0 for a, b in itertools.pairwise(offs)), (bar_h, ticks)
        assert len(colorbar_ticks(-0.04, 0.04, 40.0, 13.0)) <= 3
    finally:
        tv.close()


def test_trace_table_groups_rk_stages_shares_colours_and_opens_a_window(qapp):
    """The 2D solver's real keys: rows RK 1..3 and the end of the step, columns per quantity (the same
    quantity down a column), one colour range per quantity by default, and a floating window of the same
    capture (the 200 px dock showed 45 x 35 px thumbnails with "1 u*" .. "6 ∇·u*" repeating per stage)."""
    from aerolume.app.widgets.trace_viewer import TraceViewer, column_title, normalize_trace, table_layout
    rng = np.random.default_rng(3)
    keys = []
    for k in (1, 2, 3):
        keys += [f"stage{k}/explicit_u", f"stage{k}/explicit_v", f"stage{k}/divergence",
                 f"stage{k}/pressure_potential", f"stage{k}/projected_u", f"stage{k}/projected_v"]
    keys += ["final_u", "final_v", "final_p"]
    cap = normalize_trace({"extent": (0, 2, 0, 1),
                           "stages": [(key, (rng.normal(size=(16, 32)) * (1 + i)).astype(np.float32))
                                      for i, key in enumerate(keys)]})
    rows, cols, cell = table_layout(cap.stages)
    assert rows == ["RK 1", "RK 2", "RK 3", "End of step"]
    assert len(cols) == 7 and cell[0] == (0, 0) and cell[6] == (1, 0) and cell[12] == (2, 0)   # u* down column 0
    assert cell[18] == (3, 4) and cell[20] == (3, 6)       # end-of-step u under the projected u; p on its own
    assert column_title("Advection–diffusion", "u*") == "u* adv." and column_title("Projection", "u") == "u proj."
    assert table_layout(normalize_trace({"a": np.ones((4, 4)), "b": np.ones((4, 4))}).stages) == ([], [], {})
    tv = TraceViewer()
    try:
        tv.resize(1600, 230)
        tv.show()
        tv.set_trace(cap)
        assert _wait(qapp, lambda: tv.grid_shape() == (4, 7), 2.0), tv.grid_shape()
        assert tv.shared_btn.isChecked()
        views = tv.views()
        # u* after advection in RK 1 and RK 3 share one colour map and range (one quantity, one scale)
        assert views[0].colour_range() == pytest.approx(views[12].colour_range())
        tv.shared_btn.setChecked(False)
        assert tv.views()[0].colour_range() != pytest.approx(tv.views()[12].colour_range())
        tv.select(2)
        assert tv.detail_title.text().startswith("3 · Divergence of u*") and "RK 1" in tv.detail_title.text()
        win = tv.open_window()
        _wait(qapp, lambda: False, 0.2)
        assert win.isVisible() and len(win.viewer.trace().stages) == len(cap.stages)
        assert win.viewer.capture_btn.isHidden() and win.viewer.grid_shape() == (4, 7)
        assert tv.open_window() is win                     # one window, raised again
        win.close()
        _wait(qapp, lambda: tv._window is None, 2.0)
        assert tv._window is None                          # closed and deleted: the next opens a new one
    finally:
        tv.close()


# ---------------------------------------------------------------- explore worker hand-over
class _Body:
    def __init__(self, points, name="", role="body"):
        self.points, self.name, self.role = np.asarray(points), name, role


class _FakeSolver:
    def __init__(self):
        self.advanced = []
        self.device = type("D", (), {"type": "cpu"})()
        self.cfg = type("C", (), {})()
        self.cfg.bodies = [_Body([[0, 0], [1, 0], [0.5, 0.1]], "Airfoil")]     # the design's own body

    @property
    def bodies(self):
        return self.cfg.bodies

    def advance(self, n):
        self.advanced.append(n)

    def set_bodies(self, bodies):
        self.cfg.bodies = list(bodies)


class _FakeSession:
    """Accepts the classic keys only (like an explore session that predates live editing)."""
    POST_KEYS = frozenset({"alpha_deg", "speed", "paused", "field", "fps"})

    def __init__(self, design, sim, *, device=None, view=None):
        from types import SimpleNamespace
        self.design, self.sim = design, sim
        self.plan = SimpleNamespace(length_ref=1.0, extent=(-4.0, 6.0, -3.0, 3.0), ground=False, dimension=2)
        self.solver = _FakeSolver()
        self.posts = []
        self.alpha_deg = float(sim.conditions.alpha_deg)
        self.speed = float(sim.conditions.speed)
        self._bodies = [np.array([[0, 0], [1, 0], [0.5, 0.1]], float)]
        self.renderer = type("R", (), {"invalidate_geometry": lambda self: None})()
        self.ticks = 0

    def rebuild_bodies(self):
        """What the real session does on a pitch change: its own bodies only."""
        self.solver.set_bodies([_Body([[0, 0], [1, 0], [0.5, 0.1]], "Airfoil")])

    def post(self, **changes):
        unknown = set(changes) - self.POST_KEYS
        if unknown:
            raise TypeError(f"unknown live setting(s): {', '.join(sorted(unknown))}")
        self.posts.append(dict(changes))
        if "speed" in changes:
            self.speed = float(changes["speed"])

    def tick(self, max_ms=33.0):
        self.ticks += 1
        return type("F", (), {"step": self.ticks * 10, "t": self.ticks * 0.01, "steps_per_s": 1000.0,
                              "diagnostics": {"div_rms": 1e-6, "cfl": 0.4}})()

    def close(self):
        pass


def test_live_worker_splits_unknown_keys_and_runs_its_fallbacks(qapp):
    from aerolume.app.jobs import LiveWorker, accepted_post_keys
    from aerolume.app.widgets.draw_tools import ObstacleSketch
    from aerolume.geometry import design_from_airfoil, naca
    design = design_from_airfoil(naca("0012", n=80), "NACA 0012", 1.0)
    sim = Simulation("Explore", design.id)
    sim.conditions.speed = 1.0
    w = LiveWorker()
    sessions = []

    def factory(d, s, *, device=None, view=None):
        sessions.append(_FakeSession(d, s, device=device, view=view))
        return sessions[-1]

    w.session_factory = factory
    w.start(design, sim)
    w.post(alpha_deg=2.0, steps_per_frame=5, paused=True)          # buffered before the session exists
    assert _wait(qapp, lambda: sessions and sessions[0].ticks > 2)
    s = sessions[0]
    assert accepted_post_keys(s) == _FakeSession.POST_KEYS
    assert {"alpha_deg": 2.0, "paused": True} in s.posts                # the known keys still arrived
    assert w.supports("steps_per_frame") is False and w.supports("step") is True and w.supports("speed")
    w.step(7)
    assert _wait(qapp, lambda: s.solver.advanced == [7])              # stepped by the worker while paused
    w.post(nu=2e-5)
    assert _wait(qapp, lambda: math.isclose(s.sim.conditions.fluid.viscosity, 2e-5 * s.sim.conditions.fluid.density))
    assert any("speed" in p for p in s.posts)                          # the session recomputed its Reynolds number
    sk = ObstacleSketch()
    sk.add_polygon([(1.5, -0.2), (1.8, -0.2), (1.8, 0.2), (1.5, 0.2)])
    w.post(obstacles=sk.payload(s.plan.extent, (64, 128)))
    assert _wait(qapp, lambda: len(s.solver.bodies) == 2)
    assert s.solver.bodies[0].name == "Airfoil" and s.solver.bodies[-1].name == "Drawn 1"
    assert len(s._bodies) == 2 and w.obstacle_error == ""
    s.alpha_deg = 4.0                                                  # the session rebuilt its bodies itself
    s.rebuild_bodies()
    assert _wait(qapp, lambda: len(s.solver.bodies) == 2 and s.solver.bodies[-1].name == "Drawn 1")
    w.post(obstacles=None)                                             # everything erased
    assert _wait(qapp, lambda: len(s.solver.bodies) == 1)
    assert len(w.diagnostics) > 0 and w.diagnostics.latest()["div_rms"] == 1e-6
    w.stop()
    assert _wait(qapp, lambda: not w.is_running())
    w.shutdown()


def test_job_manager_feeds_diagnostics_and_delivers_traces(qapp, tmp_path):
    from aerolume.app.jobs import JobManager
    from aerolume.app.state import AppState
    from aerolume.geometry import design_from_airfoil, naca
    from aerolume.model import CFDResult
    from aerolume.project import Project
    st = AppState()
    p = Project.create(tmp_path / "P.aerolume", "P")
    d = design_from_airfoil(naca("0012", n=60), "A", 1.0)
    p.add_design(d)
    sim = Simulation("run", d.id)
    p.add_simulation(sim)
    st.set_project(p) if hasattr(st, "set_project") else setattr(st, "project", p)
    jm = JobManager(st)
    traces = []
    jm.jobTrace.connect(lambda sid, tr: traces.append((sid, tr)))

    def fake(design, sim_, *, progress=None, frame=None, cancel=None, view=None):
        for k in range(30):
            tr = {"stages": [("Divergence", np.ones((4, 4)))]} if getattr(view, "capture_trace", False) else None
            if tr is not None:
                view.capture_trace = False
            frame(type("F", (), {"step": k, "t": 0.1 * k, "phase": "transient", "steps_per_s": 100.0,
                                 "diagnostics": {"div_rms": 1e-5 / (k + 1)}, "trace": tr, "volume": None,
                                 "streamlines3d": None})())
            if k == 3:
                jm.view.capture_trace = True         # what request_trace does from the GUI
            time.sleep(0.003)
        return CFDResult(sim_id=sim_.id)

    jm.runner = fake
    log = jm.diagnostics_for(sim.id)
    jm.submit(sim.id)
    assert jm.request_trace(sim.id) in (True, False)
    assert jm.wait(10)
    assert _wait(qapp, lambda: traces)
    assert len(log) == 30 and log.latest()["div_rms"] == pytest.approx(1e-5 / 30)
    assert traces[0][0] == sim.id and jm.view.capture_trace is False
    assert jm.request_trace(sim.id) is False                                # nothing runs now
    jm.shutdown()
    _ = threading.active_count()
