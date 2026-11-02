"""Layout and run-state regressions of the 3D result and run pages (critique round 1, "run" area).

* the Summary tab scrolls instead of painting its note over the table (1280 x 800, Assistant dock open);
* the Fields side panel never clips its rows, whatever the resize / theme history;
* a 3D run the Simulate page did not see start (built mid-run, as when a script or the assistant
  starts it) and a run whose mode was resolved without its design switch to the 3D live UI;
* results open on smoke with thin streamlines (streamlines alone without smoke), layer toggles styled like
  the View control, Strouhal is quoted only with a clear spectral peak, an early stop at
  convergence is named, and one duration is quoted per run;
* Compare: table first, legible bar labels in the dark theme, middle-elided simulation names.
"""
# The shared fixtures (qapp, project3d, state3d) are imported from test_app_3d_runtime, so pytest
# finds them here; their names as test arguments are those fixtures, not redefinitions.
# ruff: noqa: F811
from __future__ import annotations

import os
import sys
import threading
import time

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
    state3d,  # noqa: F401  (a fixture)
    synthetic_cfd3d,
)
from test_app_3d_runtime_flow import dye_volume

pytestmark = pytest.mark.gui


def _rect_in(widget, ancestor):
    """``widget``'s geometry in ``ancestor``'s coordinates."""
    from PySide6.QtCore import QRect
    top_left = widget.mapTo(ancestor, widget.rect().topLeft())
    return QRect(top_left, widget.size())


@pytest.fixture()
def light(qapp):
    from aerolume.app import theme
    theme.apply(qapp, "light")
    yield
    theme.apply(qapp, "light")


# ---------------------------------------------------------------------------
# Results ▸ Summary: never overlapping
# ---------------------------------------------------------------------------

def test_summary_tab_scrolls_with_the_assistant_dock_open_at_1280x800(qapp, project3d, tmp_path, monkeypatch,
                                                                       light):
    """The real window, 1280 x 800, Assistant dock open: the note and the per-part table sit below the
    summary table (no overlap) and the page scrolls; the per-part numbers sit side by side."""
    from aerolume.app import settings
    from aerolume.app.main_window import MainWindow
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    settings.use_file(tmp_path / "settings.ini")
    r = project3d.load_result("s3d")
    r.per_body = {"body": {"cd": 0.4123, "cl": 0.0151}, "wheel_fl": {"cd": -0.0024, "cl": -0.0089},
                  "wheel_rr": {"cd": -0.0121, "cl": -0.0191}}
    project3d.save_result("s3d", r)
    w = MainWindow(probe_device=False, load_jobs=True)
    try:
        w.resize(1280, 800)
        w.show()
        w.open_project_path(str(project3d.path), confirmed=True)
        pump(qapp, 0.3)
        w.state.select_simulation("s3d")
        w.state.navigate("results")
        pump(qapp, 0.3)
        assert w.show_assistant()
        pump(qapp, 0.5)
        view = w.page("results").view3d
        assert view is not None and view.isVisible()
        view.select_tab("summary")
        pump(qapp, 0.3)
        table, note, parts, cap = view.summary_table, view.summary_note, view.part_table, view.part_caption
        assert parts.isVisible() and cap.isVisible()
        host = table.parentWidget()
        rt, rn, rc, rp = (_rect_in(x, host) for x in (table, note, cap, parts))
        # every row of the table is drawn (fixed to its rows) and nothing is laid over it
        assert table.height() >= table.rowCount() * 22
        assert not rn.intersects(rt), (rt, rn)
        assert not rc.intersects(rt) and not rc.intersects(rn) and not rp.intersects(rc), (rn, rc, rp)
        assert rn.top() >= rt.bottom() and rc.top() >= rn.bottom() and rp.top() >= rc.bottom()
        # the table is whole: it never scrolls inside itself (a bar over its last row, Max out of sight);
        # too narrow for Conditions beside it, the page puts them below
        assert not table.horizontalScrollBar().isVisible() and not table.verticalScrollBar().isVisible()
        assert table.width() >= table.minimumWidth() > 0
        cond = view.conditions
        rc_cond = _rect_in(cond, host)
        assert not rc_cond.intersects(rt)
        # the page is shorter than its content here: it scrolls
        from PySide6.QtWidgets import QScrollArea
        sc = next(a for a in view.tabs.currentWidget().findChildren(QScrollArea) if a.isAncestorOf(table))
        assert sc.verticalScrollBar().maximum() > 0
        # per part: C_D and C_L next to each other, not 900 px apart
        assert parts.columnWidth(1) == parts.columnWidth(2) == 90
        assert parts.width() <= 2 * 90 + parts.columnWidth(0) + 8
        w.grab().save(str(tmp_path / "summary_1280_assistant.png"))
    finally:
        w._quit_confirmed = True
        w.close()
        w.deleteLater()
        pump(qapp, 0.1)


def test_2d_summary_is_scrollable_and_per_body_compact(qapp, state3d, light):
    from PySide6.QtWidgets import QScrollArea

    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("f2d")
    page = ResultsPage(state, jobs)
    page.resize(900, 420)
    page.show()
    pump(qapp, 0.3)
    table, note = page.summary_table, page.summary_note
    sc = next(a for a in page.findChildren(QScrollArea) if a.isAncestorOf(table))
    host = table.parentWidget()
    assert not _rect_in(note, host).intersects(_rect_in(table, host))
    assert sc.verticalScrollBar().maximum() > 0
    page.close()


# ---------------------------------------------------------------------------
# Results ▸ Fields: the side panel never clips; default layers
# ---------------------------------------------------------------------------

@needs_vtk
def test_fields_side_panel_fits_after_resizes_and_theme_changes(qapp, state3d, light):
    from aerolume.app import theme
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
    vp = fp.side_scroll.viewport()

    def overflow() -> list[str]:
        out = []
        for child in fp.grid.findChildren(QtWidgets.QWidget):
            if not child.isVisibleTo(fp.grid) or child.width() <= 0:
                continue
            right = child.mapTo(vp, child.rect().topRight()).x()
            if right > vp.width():
                out.append(f"{type(child).__name__} '{getattr(child, 'text', lambda: '')()}' ends at {right}")
        return out

    for size, th in (((1280, 800), "dark"), ((1600, 1000), "dark"), ((1280, 800), "light"),
                     ((1600, 1000), "light"), ((1280, 800), "light")):
        theme.apply(qapp, th)
        page.resize(*size)
        pump(qapp, 0.3)
        assert overflow() == [], (size, th, overflow())
        assert fp.grid.width() <= vp.width()
    # the readout is a plain distance (the normal is the control above it)
    assert fp.pos_value.text().endswith(" m") and "=" not in fp.pos_value.text()
    page.close()


@needs_vtk
def test_results_without_smoke_open_on_streamlines(qapp, state3d, light):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("s3d")                    # this run stored no smoke snapshot
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.2)
    page.handle_action("fields")
    pump(qapp, 0.5)
    fp = page.view3d.fields_panel
    assert not fp.smoke_on.isEnabled() and fp.stream_on.isChecked()
    assert "stream" in fp.handles()
    lay = fp.view.scene._layers[fp.handles()["stream"]]
    assert lay.extra["tubes"] is True                 # alone: shaded tubes
    page.close()


def test_layer_toggles_look_like_the_view_control_in_both_themes(qapp, light):
    from aerolume.app import theme
    from aerolume.app.widgets.view3d_panel import FlowToggles, layer_toggle_qss
    t = FlowToggles(("smoke", "streamlines"), shown={"smoke": True, "streamlines": False})
    b_on, b_off = t.buttons["smoke"], t.buttons["streamlines"]
    for name in ("light", "dark"):
        theme.apply(qapp, name)
        pump(qapp, 0.05)
        tk = theme.tokens()
        qss = b_on.styleSheet()
        assert qss == layer_toggle_qss(tk)
        checked = qss.split("QToolButton:checked {", 1)[1].split("}", 1)[0]
        assert tk["accent_soft"] in checked and tk["text"] in checked       # filled, 1 px ink border (v4)
        base = qss.split("QToolButton {", 1)[1].split("}", 1)[0]
        assert tk["text2"] in base and tk["muted"] not in base               # off = text2, not "disabled" grey
    assert b_on.isChecked() and not b_off.isChecked()
    # a checked button renders a different (filled) background from an unchecked one
    for b in (b_on, b_off):
        b.resize(90, 24)
    img_on, img_off = b_on.grab().toImage(), b_off.grab().toImage()
    assert img_on.pixelColor(4, 12) != img_off.pixelColor(4, 12)


# ---------------------------------------------------------------------------
# Results ▸ Summary: Strouhal only with a clear peak
# ---------------------------------------------------------------------------

def test_strouhal_is_quoted_only_with_a_clear_spectral_peak():
    from aerolume.app.widgets.results3d import (
        STROUHAL_PEAK_RATIO,
        STROUHAL_PEAK_SHARE,
        force_spectrum_peak,
        strouhal_cell,
    )
    periodic = synthetic_cfd3d("p")                       # C_L = 0.02 + 0.02 sin(1.7 t): one clean line
    ratio, share, f, key = force_spectrum_peak(periodic)
    assert key == "cl" and ratio > 100 * STROUHAL_PEAK_RATIO and share > 0.9
    assert f == pytest.approx(1.7 / (2 * np.pi), abs=0.05)
    text, tip = strouhal_cell(periodic)
    assert text == "0.190" and "spectrum" in tip
    broadband = synthetic_cfd3d("b")
    t = broadband.history["t"]
    for seed in range(8):                                 # a broadband wake: no line, whatever the draw
        rng = np.random.default_rng(seed)
        broadband.history["cl"] = 0.17 + 0.05 * rng.standard_normal(t.size)
        ratio, share, _f, _k = force_spectrum_peak(broadband)
        assert share < STROUHAL_PEAK_SHARE, (seed, ratio, share)
        text, tip = strouhal_cell(broadband)
        assert text == "—" and "No clear shedding frequency" in tip
    broadband.strouhal = None
    assert strouhal_cell(broadband)[0] == "—"


# ---------------------------------------------------------------------------
# Simulate: 3D runs the page did not see start
# ---------------------------------------------------------------------------

def _blocking_runner(release: threading.Event, frames: int = 3):
    def run(design, sim, *, progress, frame, cancel, view, **_kw):
        for i in range(frames):
            progress(0.2 + 0.1 * i, f"Developing the flow (t = {2 + i} of 20)")
            f = slice_frame("y", phase="transient", progress=0.2 + 0.1 * i)
            f.step = 100 + i
            frame(f)
            time.sleep(0.08)
        release.wait(20)
        return synthetic_cfd3d(sim.id)
    return run


@needs_vtk
def test_3d_run_added_in_one_tick_shows_the_3d_live_ui(qapp, state3d, monkeypatch, light):
    """A design and its 3D simulation added in the same event-loop tick and submitted at once (what the
    Script console and the assistant do through AppHost): the page shows Smoke after the first frame."""
    from test_app_3d_runtime import sphere_design

    from aerolume.app.pages.simulate import SimulatePage
    from aerolume.model import FlowConditions, Simulation
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("f2d")                     # the page starts on a 2D case
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.2)
    assert page._mode == "field"
    release = threading.Event()
    jobs.runner = _blocking_runner(release)
    d = sphere_design("Script sphere")
    sim = Simulation(name="Script run", design_id=d.id, conditions=FlowConditions(speed=10.0), id="scr1")
    state.add_design(d, select=False)
    state.add_simulation(sim, select=False)
    jobs.submit("scr1")
    state.select_simulation("scr1")
    try:
        assert pump_until(qapp, lambda: page._last_frame3d is not None, 10)
        assert page._mode == "field3d"
        assert page.flow_toggles.buttons["smoke"].isVisible()
        assert page.streamlines_btn is None or not page.streamlines_btn.isVisible()   # no 2D toggle
        assert page._row_cd.label.text() == "C<sub>D</sub>"
    finally:
        release.set()
        assert jobs.wait(20)
    page.shutdown()
    page.close()


@needs_vtk
def test_simulate_page_built_mid_run_adopts_the_3d_run(qapp, state3d, monkeypatch, light):
    """The critic's case: the Simulate page is first opened while a 3D run (started by a script) is
    in progress; it must show the run in 3D at once, not a full-size 2D slice until the end."""
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    release = threading.Event()
    jobs.runner = _blocking_runner(release)
    jobs.submit("s3n")
    try:
        assert pump_until(qapp, lambda: jobs.running_id() == "s3n", 5)
        pump(qapp, 0.3)
        state.select_simulation("s3n")
        page = SimulatePage(state, jobs)                   # built now, mid-run
        page.resize(1280, 800)
        page.show()
        pump(qapp, 0.2)
        assert page._mode == "field3d" and page._shown_sim_id == "s3n"
        assert page.flow_toggles.buttons["smoke"].isVisible() and page.view3d.isVisible()
        assert page.status_text.text().startswith("Running")
        # and a page whose mode was somehow resolved as 2D switches on the next 3D frame
        page._set_mode("field", None, state.simulation)
        assert not page.flow_toggles.buttons["smoke"].isVisible()
        jobs.jobFrame.emit("s3n", slice_frame("y", phase="transient", progress=0.5))
        pump(qapp, 0.2)
        assert page._mode == "field3d" and page.flow_toggles.buttons["smoke"].isVisible()
    finally:
        release.set()
        assert jobs.wait(20)
    pump(qapp, 0.2)
    page.shutdown()
    page.close()


# ---------------------------------------------------------------------------
# Simulate: early stop and one duration
# ---------------------------------------------------------------------------

def test_run_panel_names_an_early_stop_and_quotes_the_job_time(qapp, state3d, monkeypatch, light):
    from aerolume.app.pages.simulate import SimulatePage
    from aerolume.app.widgets.kpi import fmt_duration
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)

    def run(design, sim, *, progress, frame, cancel, view, **_kw):
        progress(0.5, "Averaging forces (t = 10 of 20)")
        time.sleep(0.3)
        res = synthetic_cfd3d(sim.id)
        res.t_end, res.converged, res.wall_seconds = 16.3, True, 0.05      # the solver's own (shorter) time
        return res

    jobs.runner = run
    state.select_simulation("s3n")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.2)
    page.handle_action("run")
    assert jobs.wait(20)
    pump(qapp, 0.3)
    assert page.phase_value.text() == "Converged (stopped early)"
    assert page.time_value.text() == "16.3 of 20.0 L/U"
    assert "Converged at 16.3 of 20.0 L/U" in page.phase_value.toolTip()
    took = jobs.job_seconds("s3n")
    assert took is not None and took >= 0.3
    assert page.status_text.text() == f"Done in {fmt_duration(took)}"    # the job's time, not 0.05 s
    # a run that went to its planned end just says Done
    sim = state.project.simulations["s3n"]
    res = synthetic_cfd3d("s3n")
    res.t_end = 20.0
    page._done_rows(sim, res)
    assert page.phase_value.text() == "Done" and page.time_value.text() == "20.0 L/U"
    page.shutdown()
    page.close()


# ---------------------------------------------------------------------------
# Compare: table first, legible bars, elided names
# ---------------------------------------------------------------------------

def test_compare_table_first_bars_legible_names_elided(qapp, state3d, tmp_path, light):
    from PySide6.QtCore import Qt

    from aerolume.app import settings, theme
    from aerolume.app.pages.compare import TABLE_MIN_FRACTION, ComparePage
    settings.use_file(tmp_path / "settings.ini")
    state, jobs = state3d
    state.compare_ids[:] = ["s3d", "s3y"]
    page = ComparePage(state, jobs)
    page.resize(1280, 640)
    page.show()
    pump(qapp, 0.3)
    body = page.body_split
    table_h, total = body.sizes()[0], sum(body.sizes())
    rows = page.table_rows_height()
    # every row when they fit next to a usable plot, else at least 45 % of the height
    assert table_h >= min(rows, round(TABLE_MIN_FRACTION * total)) - 1, (table_h, rows, total)
    page.resize(1600, 1000)
    pump(qapp, 0.2)
    assert body.sizes()[0] >= min(page.table_rows_height(), round(TABLE_MIN_FRACTION * sum(body.sizes()))) - 1
    # the user's split is remembered (as a fraction) and survives a resize
    body.setSizes([200, sum(body.sizes()) - 200])
    page._on_split_moved()
    page.resize(1500, 900)
    pump(qapp, 0.2)
    frac = body.sizes()[0] / sum(body.sizes())
    assert abs(frac - page._table_fraction) < 0.02
    assert settings.get().value("compare/table_fraction") is not None
    # bar labels in the primary text colour, recoloured with the theme
    page.tabs.setCurrentIndex(0)
    page._fill_bars()
    theme.apply(qapp, "dark")
    pump(qapp, 0.1)
    cols = page.bars.value_label_colors()
    assert cols and all(c == theme.tokens()["text"] for c in cols)
    theme.apply(qapp, "light")
    pump(qapp, 0.1)
    assert all(c == theme.tokens()["text"] for c in page.bars.value_label_colors())
    # long names lose their middle and the list never scrolls sideways
    lst = page.sim_list
    assert lst.textElideMode() == Qt.TextElideMode.ElideMiddle
    assert lst.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert lst.width() >= 220
    page.close()


def test_job_seconds_is_none_for_runs_of_other_sessions(qapp, state3d):
    _state, jobs = state3d
    assert jobs.job_seconds("s3d") is None


@needs_vtk
def test_thin_streamlines_follow_the_smoke_in_the_live_view(qapp, state3d, monkeypatch, light):
    """Live: streamlines switched on while smoke shows are thin translucent lines; without smoke, tubes."""
    from test_viz3d_fixtures import sphere_flow_volume

    from aerolume.app.pages.simulate import SimulatePage
    from aerolume.app.widgets.view3d_panel import STREAMLINE_OPACITY_WITH_SMOKE
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3n")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.3)
    monkeypatch.setattr(jobs, "running_id", lambda: "s3n")
    page._shown_sim_id = "s3n"
    vol = sphere_flow_volume(n=(48, 32, 32))
    f = slice_frame("y")
    f.volume = dye_volume(vol)
    jobs.jobFrame.emit("s3n", f)
    pump(qapp, 0.5)
    page.flow_toggles.buttons["streamlines"].setChecked(True)
    assert pump_until(qapp, lambda: "streamlines" in page.flow.layer_handles(), 5), page.flow.errors
    h = page.flow.layer_handles()["streamlines"][0]
    lay = page.view3d.scene._layers[h]
    assert lay.extra["tubes"] is False
    assert all(a.GetProperty().GetOpacity() == pytest.approx(STREAMLINE_OPACITY_WITH_SMOKE) for a in lay.actors)
    page.flow_toggles.buttons["smoke"].setChecked(False)
    assert pump_until(qapp, lambda: page.view3d.scene._layers[
        page.flow.layer_handles()["streamlines"][0]].extra["tubes"] is True, 5)
    page.shutdown()
    page.close()


def test_early_stop_is_named_once_the_plan_of_a_finished_run_arrives(qapp, state3d, monkeypatch, light):
    """A finished run is shown before its plan is known (plans are made off the GUI thread); when the
    plan arrives, the Run rows say the run stopped early at convergence."""
    from types import SimpleNamespace

    from aerolume.app.pages.simulate import SimulatePage, plan_key
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3d")                      # converged at t = 30 L/U
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.3)
    assert page.phase_value.text() == "Done" and page.time_value.text() == "30.0 L/U"
    sim = state.project.simulations["s3d"]
    plan = SimpleNamespace(t_end=40.0, est_seconds=100.0, re_sim=6.8e5, re_real=6.8e5, warnings=[], cells=0)
    page._plans[plan_key(sim, page._design_of(sim))] = plan
    page._apply_plan(sim, plan)
    assert page.phase_value.text() == "Converged (stopped early)"
    assert page.time_value.text() == "30.0 of 40.0 L/U"
    page.shutdown()
    page.close()


@needs_vtk
def test_simulate_3d_strips_reflow_instead_of_eliding(qapp, state3d, monkeypatch, light):
    """A narrow viewport stacks the layer buttons and the slice controls in two rows; no button is
    squeezed below its text (they read "S…e" and "iide · y" at 1280 px with a dock open)."""
    from PySide6.QtWidgets import QAbstractButton, QBoxLayout

    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3n")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.3)
    page.flow_toggles.buttons["slice"].setChecked(True)      # the slice controls show with a slice
    pump(qapp, 0.1)
    assert page._slice_box.direction() == QBoxLayout.Direction.LeftToRight
    page.resize(760, 800)
    pump(qapp, 0.3)
    assert page._slice_box.direction() == QBoxLayout.Direction.TopToBottom
    whole = [b for b in page.flow_toggles.buttons.values() if b.isVisible()]
    whole += [b for b in page.slice_axis.findChildren(QAbstractButton)]
    whole += [b for b in page.view_mode.findChildren(QAbstractButton) if b.isVisible()]
    for b in whole:
        assert b.width() >= b.sizeHint().width(), (b.text(), b.width(), b.sizeHint().width())
    page.resize(1600, 1000)
    pump(qapp, 0.3)
    assert page._slice_box.direction() == QBoxLayout.Direction.LeftToRight
    page.shutdown()
    page.close()
