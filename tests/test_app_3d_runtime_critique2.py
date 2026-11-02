"""Regressions of the 3D run and result pages (critique round 2, "run" area).

* Simulate: the flow toggles keep their natural width, left-aligned, when the strip wraps; the slice
  controls show only while a slice is on screen; Field / Colormap are disabled (and say why) when
  only smoke is shown; the 3D caption names the slice only while one is drawn; the monitor plot
  fits under the Run rows at 1280 x 800;
* Split views put a flat slice under the 3D view; a ``keep_body`` slice view zooms towards the body
  instead of showing a thin band; the scale bar never sits on the body;
* Compare bars span the plot; the Surface tab's colour-map combo shows its longest item whole;
* Fields opens on smoke with thin streamlines, and the seed rake is drawn only with streamlines;
* tiny values are written as mantissa × 10ⁿ, not as zeros; the Results fact line elides.
"""
# The shared fixtures (qapp, project3d, state3d) are imported from test_app_3d_runtime, so pytest
# finds them here; their names as test arguments are those fixtures, not redefinitions.
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
    project3d,  # noqa: F401  (a fixture used through state3d)
    pump,
    pump_until,
    qapp,  # noqa: F401  (a fixture)
    slice_frame,
    sphere_design,
    state3d,  # noqa: F401  (a fixture)
    synthetic_cfd3d,
)
from test_app_3d_runtime_flow import dye_volume

pytestmark = pytest.mark.gui


@pytest.fixture()
def light(qapp):
    from aerolume.app import theme
    theme.apply(qapp, "light")
    yield
    theme.apply(qapp, "light")


def _live_page(qapp, state3d, monkeypatch, size=(1280, 800)):
    """A Simulate page showing a live 3D frame (smoke volume) of the sphere run "s3n"."""
    from test_viz3d_fixtures import sphere_flow_volume

    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3n")
    page = SimulatePage(state, jobs)
    page.resize(*size)
    page.show()
    pump(qapp, 0.3)
    monkeypatch.setattr(jobs, "running_id", lambda: "s3n")
    page._shown_sim_id = "s3n"
    f = slice_frame("y")
    f.volume = dye_volume(sphere_flow_volume(n=(48, 32, 32)))
    jobs.jobFrame.emit("s3n", f)
    pump(qapp, 0.5)
    return page


# ---------------------------------------------------------------------------
# Simulate: the 3D strip
# ---------------------------------------------------------------------------

@needs_vtk
def test_flow_toggles_keep_their_width_and_the_slice_row_follows_the_slice(qapp, state3d, monkeypatch, light):
    from PySide6.QtWidgets import QBoxLayout
    page = _live_page(qapp, state3d, monkeypatch, size=(900, 800))
    try:
        tg = page.flow_toggles
        assert tg.wanted("smoke") and not tg.wanted("slice") and page.view_mode.current() == "3d"
        # smoke alone: no slice controls, nothing coloured, no "y = ..." in the 3D caption
        assert not page._slice_row.isVisible()
        assert not page.field_combo.isEnabled() and not page.cmap_combo.isEnabled()
        assert "switch one of them on" in page.field_combo.toolTip()
        assert " = " not in page.flow.caption.split("step", 1)[-1], page.flow.caption
        # the toggles keep their natural width, left-aligned, whatever room the strip has
        buttons = [b for b in tg.buttons.values() if b.isVisible()]
        for b in buttons:
            assert b.width() <= b.sizeHint().width() + 2, (b.text(), b.width(), b.sizeHint().width())
        right = max(b.mapTo(page.slice_strip, b.rect().topRight()).x() for b in buttons)
        assert right < 0.75 * page.slice_strip.width()
        # streamlines colour by the field: Field and Colormap come back
        tg.buttons["streamlines"].setChecked(True)
        pump(qapp, 0.2)
        assert page.field_combo.isEnabled() and page.cmap_combo.isEnabled()
        # the slice plane brings the slice controls (wrapped in two rows at this width) and its position
        tg.buttons["slice"].setChecked(True)
        pump(qapp, 0.2)
        assert page._slice_row.isVisible() and page.slice_axis.isEnabled()
        assert page._slice_box.direction() == QBoxLayout.Direction.TopToBottom
        jobs = state3d[1]
        jobs.jobFrame.emit("s3n", slice_frame("y"))
        assert pump_until(qapp, lambda: "y = " in page.flow.caption, 5), page.flow.caption
        for b in buttons:                                   # still natural width in the two-row layout
            assert b.width() <= b.sizeHint().width() + 2
        tg.buttons["slice"].setChecked(False)
        tg.buttons["streamlines"].setChecked(False)
        pump(qapp, 0.2)
        assert not page._slice_row.isVisible()
        assert "y = " not in page.flow.caption
        # Slice and Split views always show a slice: the controls are there
        page.view_mode.set_current("split")
        pump(qapp, 0.3)
        assert page._slice_row.isVisible() and page.field_combo.isEnabled()
        page.view_mode.set_current("3d")
        pump(qapp, 0.2)
        assert not page._slice_row.isVisible()
    finally:
        page.shutdown()
        page.close()


@needs_vtk
def test_stored_result_caption_names_only_what_is_shown(qapp, state3d, monkeypatch, light):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3d")                      # a finished run without a smoke snapshot
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    try:
        assert pump_until(qapp, lambda: page.flow is not None and page.flow.caption.startswith("Last run"), 5),             getattr(page.flow, "caption", None)
        page.flow_toggles.buttons["streamlines"].setChecked(True)
        pump(qapp, 0.3)
        assert page.flow.caption == "Last run · time-averaged flow"
        page.flow_toggles.buttons["streamlines"].setChecked(False)
        pump(qapp, 0.3)
        assert page.flow.caption == "Last run"
    finally:
        page.shutdown()
        page.close()


@needs_vtk
def test_monitor_plot_fits_under_the_run_rows_at_1280x800(qapp, state3d, monkeypatch, light):
    page = _live_page(qapp, state3d, monkeypatch, size=(1280, 800 - 120))    # the page in an 800 px window
    try:
        pump(qapp, 0.3)
        sc = page._side_scroll
        h = page.history
        assert h.isVisible()
        bottom = h.mapTo(sc.viewport(), h.rect().bottomLeft()).y()
        room = sc.viewport().height()
        assert page.MONITOR_H_MIN <= h.height() <= page.MONITOR_H_MAX
        if h.height() > page.MONITOR_H_MIN:                  # it shrank to fit: the whole plot is in view
            assert bottom <= room, (bottom, room)
        small = h.height()
        page.resize(1600, 1000)
        pump(qapp, 0.4)
        # a taller window gives the plot its full height back, never more
        assert small <= h.height() <= page.MONITOR_H_MAX
        assert h.height() == page.MONITOR_H_MAX or h.mapTo(sc.viewport(), h.rect().bottomLeft()).y() <=             sc.viewport().height()
    finally:
        page.shutdown()
        page.close()


# ---------------------------------------------------------------------------
# Split views, keep_body framing and the scale bar
# ---------------------------------------------------------------------------

def test_split_orientation_follows_the_slice_shape():
    from PySide6.QtCore import Qt

    from aerolume.app.widgets.field_view import split_orientation
    # a car's side slice (about 3:1) in a 1060 x 760 viewport goes under the 3D view
    assert split_orientation(1060, 760, 14.8 / 5.0) == Qt.Orientation.Vertical
    # a cross-flow slice (about 1:1.2) stays beside it
    assert split_orientation(1060, 760, 3.0 / 3.6) == Qt.Orientation.Horizontal
    assert split_orientation(1060, 760, None) == Qt.Orientation.Horizontal


def _car_slice_view(w: int, h: int):
    """A FieldView with a flat 14.8 x 5 m side slice of a 4.8 m car on the road, ``keep_body`` on."""
    from aerolume.app.widgets.field_view import FieldView
    fv = FieldView()
    fv.keep_body = True
    fv.resize(w, h)
    nx, ny = 296, 100
    vals = np.tile(np.linspace(0.0, 30.0, nx, dtype=np.float32), (ny, 1))
    mask = np.zeros((ny, nx), np.float32)
    x = np.linspace(-4.0, 10.8, nx)
    z = np.linspace(0.0, 5.0, ny)
    car = (x[None, :] > 0.0) & (x[None, :] < 4.8) & (z[:, None] > 0.15) & (z[:, None] < 1.45)
    mask[car] = 1.0
    car_loop = np.array([[0.0, 0.15], [4.8, 0.15], [4.8, 1.45], [0.0, 1.45]])
    fv.set_ground(0.0)
    fv.set_sections([np.vstack([car_loop, car_loop[:1]])])
    fv.set_field(vals, (-4.0, 10.8, 0.0, 5.0), field="speed", mask=mask)
    fv.show()
    return fv


def test_keep_body_zooms_to_the_body_instead_of_a_thin_band(qapp, light):
    fv = _car_slice_view(380, 700)                          # a portrait Split pane
    pump(qapp, 0.1)
    fv.fit()
    img_h = 5.0 * fv._scale
    # the essential window alone (14.4 m across the plot area) gave a band this tall; now the plot
    # area spans 1.5 body lengths (7.2 m), so the slice is twice as tall and the body twice as large
    band = 5.0 * fv.plot_rect().width() / 14.4
    assert img_h >= 1.8 * band, (img_h, band)
    # the body is whole and in view
    xf = fv.world_to_screen()
    from PySide6.QtCore import QPointF
    a, b = xf.map(QPointF(0.0, 1.45)), xf.map(QPointF(4.8, 0.15))
    r = fv.plot_rect()
    assert r.left() <= a.x() and b.x() <= r.right() and r.top() <= a.y() and b.y() <= r.bottom()
    fv.close()


def test_scale_bar_avoids_the_body(qapp, light):
    from PySide6.QtCore import QPointF, QRectF
    fv = _car_slice_view(380, 160)                          # a short pane: the body reaches the bottom-left
    pump(qapp, 0.1)
    fv.fit()
    xf = fv.world_to_screen()
    body = QRectF(xf.map(QPointF(0.0, 1.45)), xf.map(QPointF(4.8, 0.15))).normalized()
    sb = fv.scalebar_rect()
    assert sb is not None and fv.plot_rect().contains(sb)
    assert not sb.intersects(body), (sb, body)
    img = fv.grab().toImage()
    assert not img.isNull()
    fv.close()


@needs_vtk
def test_results_split_stacks_a_flat_slice_under_the_3d_view(qapp, state3d, light):
    from PySide6.QtCore import Qt

    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("s3d")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.2)
    page.handle_action("fields")
    pump(qapp, 0.5)
    fp = page.view3d.fields_panel
    fp.mode.set_current("split")
    pump(qapp, 0.4)
    aspect = fp.slice_view.framing_aspect()
    assert aspect is not None
    want = Qt.Orientation.Vertical if aspect > 1.6 else Qt.Orientation.Horizontal
    assert fp.split.orientation() == want, (aspect, fp.split.orientation())
    page.close()


# ---------------------------------------------------------------------------
# Compare bars, Surface colour-map combo
# ---------------------------------------------------------------------------

def test_compare_bars_span_the_plot(qapp, light):
    from aerolume.app.widgets.plots import BarChart
    chart = BarChart()
    chart.resize(1000, 300)
    chart.show()
    for n in (2, 3, 6):
        chart.set_bars([f"run {i}" for i in range(n)], np.linspace(0.5, 1.2, n), errors=np.full(n, 0.01))
        pump(qapp, 0.2)
        spans = chart.bar_spans_px()
        left, right = chart.plot_span_px()
        width = right - left
        assert len(spans) == n
        assert spans[0][0] - left < 0.2 * width, (n, spans[0], left, width)        # first bar near the left
        assert right - spans[-1][1] < 0.2 * width, (n, spans[-1], right, width)    # last bar near the right
        assert spans[0][1] - spans[0][0] < 0.25 * width                            # two bars not half a plot
        ticks = chart.getPlotItem().getAxis("bottom")._tickLevels[0]
        assert [t[0] for t in ticks] == list(range(n))
    chart.hide()            # closed by the suite's teardown (a second PlotWidget.close() raises)


@needs_vtk
def test_surface_colormap_combo_shows_its_longest_item(qapp, state3d, light):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("s3d")
    page = ResultsPage(state, jobs)
    for size in ((1280, 800), (1600, 1000)):
        page.resize(*size)
        page.show()
        pump(qapp, 0.2)
        page.handle_action("surface")
        pump(qapp, 0.4)
        panel = page.view3d.surface_panel
        combo = panel.cmap_combo
        fm = combo.fontMetrics()
        longest = max(fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count()))
        assert combo.width() >= longest + 45, (size, combo.width(), longest)     # text, padding and arrow
    page.close()


# ---------------------------------------------------------------------------
# Fields: default layers and the rake
# ---------------------------------------------------------------------------

@needs_vtk
def test_fields_opens_on_smoke_and_thin_streamlines_and_hides_the_seed_rake_without_them(qapp, state3d, light):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    r = state.project.load_result("s3d")
    r.extra_volumes = [dye_volume(r.volume)]
    state.project.save_result("s3d", r)
    state.select_simulation("s3d")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.2)
    if not getattr(jobs.result("s3d"), "extra_volumes", None):
        page.close()
        pytest.skip("the project store does not keep extra volumes yet")
    page.handle_action("fields")
    pump(qapp, 0.5)
    fp = page.view3d.fields_panel
    assert fp.smoke_on.isChecked() and fp.stream_on.isChecked()
    h = fp.handles()
    assert {"smoke", "stream", "rake"} <= set(h)
    lay = fp.view.scene._layers[h["stream"]]
    assert lay.extra["tubes"] is False                    # thin lines over the smoke
    fp.stream_on.setChecked(False)
    assert pump_until(qapp, lambda: "rake" not in fp.handles(), 5)
    assert "smoke" in fp.handles()
    page.close()


# ---------------------------------------------------------------------------
# Number formatting and the Results fact line
# ---------------------------------------------------------------------------

def test_tiny_values_are_written_with_powers_of_ten_not_zeros():
    from aerolume.app.widgets.kpi import MINUS, fmt_num, fmt_sci, fmt_sig, needs_sci, stat_texts
    assert fmt_sci(1.1834e-6) == "1.183×10⁻⁶"
    assert fmt_sci(-3.183e-13) == f"{MINUS}3.183×10⁻¹³"
    assert fmt_sig(1.1834e-6) == "1.183×10⁻⁶"
    assert fmt_sig(0.03621) == "0.03621" and fmt_sig(640.2) == "640.2" and fmt_sig(0) == "0"
    assert fmt_sig(2.5e8) == "2.500×10⁸"
    assert fmt_num(3.267e-05) == "3.267×10⁻⁵"              # the Conditions list's dynamic pressure
    assert needs_sci([1e-6, -2e-6]) and not needs_sci([1e-6, 0.4]) and not needs_sci([0.0, float("nan")])
    mean, se, rms, lo, hi = stat_texts(1.1834e-6, 1.5e-11, 2.1e-11, 1.1830e-6, 1.1838e-6)
    assert mean == "1.183×10⁻⁶" and se == "1.5×10⁻¹¹" and lo.endswith("×10⁻⁶")
    assert "0.000000" not in (mean, se, rms, lo, hi)
    # an ordinary row keeps the metrology rule (rounded to its standard error, decimals aligned)
    assert stat_texts(1.155809, 1.5e-5, 1.8e-5, 1.155787, 1.155854)[:2] == ("1.155809", "0.000015")
    # a spread too small for the row's decimals is not printed as zeros
    # (the error by the Particle Data Group rule: 490 in its leading digits is quoted to one figure)
    assert stat_texts(0.036208, 4.9e-7, 1.0e-6, 0.036208, 0.036210)[1:3] == ("5×10⁻⁷", "1.0×10⁻⁶")


def test_3d_summary_of_a_creeping_flow_run_has_no_zero_rows():
    from aerolume.app.widgets.kpi import MINUS
    from aerolume.app.widgets.results3d import _q_ref, summary_rows3d
    from aerolume.model import FlowConditions, Simulation, Stat
    r = synthetic_cfd3d("tiny", cd=1.155, cl=3.1e-7)
    r.cl = Stat(mean=3.1e-7, rms=4.0e-7, min=-2.0e-7, max=9.0e-7, stderr=2.0e-8)
    d = sphere_design()
    sim = Simulation(name="slow", design_id=d.id, conditions=FlowConditions(speed=0.0073))
    qa = 0.5 * 1.225 * 0.0073 ** 2 * d.ref_area             # a 7 mm/s run: q A = 2.6e-5 N
    r.drag, r.lift = r.cd.mean * qa, r.cl.mean * qa
    assert _q_ref(r, d, sim) == pytest.approx(qa)
    r.power = r.drag * 0.0073
    rows = summary_rows3d(r, d, sim)
    names = [row[0].text if hasattr(row[0], "text") else row[0] for row in rows]
    for name in ("Drag", "Drag power", "Lift coefficient CL"):
        row = rows[names.index(name)]
        texts = [c.text if hasattr(c, "text") else str(c) for c in row[1:6]]
        assert all(t.replace(MINUS, "").strip("0.") != "" for t in texts if t != "—"), (name, texts)
        assert "×10" in texts[0], (name, texts)
    drag = rows[names.index("Drag")][1].text
    mant = float(drag.split("×")[0])
    assert math.isclose(mant * 10 ** math.floor(math.log10(r.drag)), r.drag, rel_tol=1e-3)



def test_results_fact_line_elides_with_the_full_text_in_its_tooltip(qapp, state3d, light):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("s3d")
    page = ResultsPage(state, jobs)
    page.resize(700, 600)
    page.show()
    pump(qapp, 0.3)
    full = page.facts.full_text()
    assert "half domain" in full
    shown = page.facts.text()
    assert page.facts.toolTip() == full
    if shown != full:                                       # too narrow for the line: cut with an ellipsis
        assert shown.endswith("…") and page.facts.fontMetrics().horizontalAdvance(shown) <= page.facts.width()
    page.resize(1800, 600)
    pump(qapp, 0.2)
    assert page.facts.text() == full
    page.close()
