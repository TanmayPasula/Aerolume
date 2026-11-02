"""Runtime pages of the desktop app (Simulate, Results, Compare, Report) and the job manager.

Everything runs offscreen. The solver is never called: JobManager tests swap in
fake run functions (that is what ``JobManager.runner`` is for), pages are fed
synthetic LiveFrames and realistic synthetic results (tests/report_fixtures.py)
saved through the real project store, so the save -> load -> display path is
the one users exercise.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from aerolume.model import (
    Body,
    CFDResult,
    Design,
    FieldData,
    FlowConditions,
    PolarPoint,
    PolarResult,
    Simulation,
    Stat,
    SurfaceData,
)

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# fixtures and synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        from aerolume.app import theme
        theme.apply(app, "dark")
    except ImportError:  # the pages must also work unthemed
        theme = None
    return app


def _project_mod():
    return pytest.importorskip("aerolume.project")


def pump(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def ellipse(cx: float, cy: float, a: float, b: float, n: int = 120) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], axis=1)


def simple_design(name: str = "Ellipse") -> Design:
    return Design(name=name, kind="airfoil", bodies=[Body("body", ellipse(0.5, 0.0, 0.5, 0.06))],
                  length_ref=1.0, force_ref=1.0, moment_ref=(0.25, 0.0))


def synthetic_fields(extent=(-1.0, 3.0, -1.0, 1.0), nx: int = 160, ny: int = 80) -> FieldData:
    x0, x1, y0, y1 = extent
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(x, y)       # row 0 = y min (solver order)
    u = 30.0 * (1 - 0.5 * np.exp(-((Y / 0.15) ** 2)) * (X > 0.5))
    v = 3.0 * np.sin(X) * np.exp(-(Y**2))
    mask = (((X - 0.5) / 0.5) ** 2 + (Y / 0.06) ** 2 <= 1).astype(np.float32)
    speed = np.hypot(u, v)
    cp = 1 - (speed / 30.0) ** 2
    return FieldData(extent=extent, fields={"u": u, "v": v, "speed": speed, "cp": cp,
                                            "vorticity": np.gradient(v, axis=1) - np.gradient(u, axis=0)},
                     mask=mask, label="mean")


def synthetic_cfd(sim_id: str, cd: float = 0.03, cl: float = 0.5, *, sensitivity: bool = True) -> CFDResult:
    t = np.linspace(0, 40, 2000)
    hist = {"t": t, "cd": cd + 0.002 * np.sin(t * 3), "cl": cl + 0.05 * np.sin(t * 1.5),
            "cm": -0.05 + 0.0 * t}
    pts = ellipse(0.5, 0.0, 0.5, 0.06, 200)
    nrm = (pts - [0.5, 0.0]) / np.array([0.5**2, 0.06**2])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    cpv = 1 - 2 * np.abs(nrm[:, 0])
    surf = SurfaceData("body", s, pts[:, 0], pts[:, 1], nrm[:, 0], nrm[:, 1], cpv, 0.004 * np.ones(len(s)),
                       sensitivity=np.sin(np.linspace(0, 6, len(s))) if sensitivity else None)
    return CFDResult(sim_id=sim_id, cd=Stat(cd, 0.002, cd - 0.004, cd + 0.004, 1.5e-4),
                     cl=Stat(cl, 0.05, cl - 0.1, cl + 0.1, 0.002), cm=Stat(-0.05, 0.001, -0.06, -0.04, 1e-4),
                     drag=0.5 * 1.225 * 900 * cd, lift=0.5 * 1.225 * 900 * cl, moment=-1.0, drag_area=cd,
                     power=0.5 * 1.225 * 900 * cd * 30, strouhal=0.2, re_real=2e6, re_sim=2e4, grid=(1024, 512),
                     cells_per_ref=128, t_end=40, t_avg_start=15, converged=True, wall_seconds=95,
                     device="cuda", precision="float32", turbulence="les-wale",
                     warnings=["Re capped for this run."], history=hist, surface=[surf],
                     mean_fields=synthetic_fields(), bodies_wind=[Body("body", pts)])


def synthetic_polar(sim_id: str) -> PolarResult:
    pts = []
    x = np.concatenate([np.linspace(1, 0, 40), np.linspace(0, 1, 40)[1:]])
    for a in np.arange(-4, 13, 1.0):
        cl = 0.11 * (a + 2)
        cp = 1 - (1 + 0.3 * np.sign(np.arange(len(x)) - 39) * cl) * np.sqrt(np.clip(x, 1e-3, 1))
        pts.append(PolarPoint(alpha=float(a), cl=cl, cd=0.006 + 0.004 * cl**2, cdp=0.002, cdf=0.004,
                              cm=-0.05, converged=bool(a < 12), x=x, y=0.0 * x, cp=cp, cp_inviscid=cp * 1.05,
                              cf=0.003 + 0 * x))
    return PolarResult(sim_id=sim_id, re=1e6, points=pts, wall_seconds=3.0)


@pytest.fixture()
def project(tmp_path):
    """A real project with one CFD result, one polar result and one simulation never run."""
    pm = _project_mod()
    pr = pm.Project.create(tmp_path / "Test.aerolume", "Test study")
    d = simple_design()
    pr.add_design(d)
    sims = {
        "cfd": Simulation(name="Cruise", design_id=d.id, conditions=FlowConditions(speed=30.0, alpha_deg=2.0),
                          status="done", id="cfd1"),
        "cfd2": Simulation(name="Cruise B", design_id=d.id, conditions=FlowConditions(speed=30.0, alpha_deg=4.0),
                           status="done", id="cfd2"),
        "polar": Simulation(name="Polar", design_id=d.id, analysis="polar", status="done", id="pol1"),
        "new": Simulation(name="Not run", design_id=d.id, id="new1"),
    }
    for s in sims.values():
        pr.add_simulation(s)
    pr.save()
    pr.save_result("cfd1", synthetic_cfd("cfd1"))
    pr.save_result("cfd2", synthetic_cfd("cfd2", cd=0.027, cl=0.6))
    pr.save_result("pol1", synthetic_polar("pol1"))
    return pr


@pytest.fixture()
def app_state(qapp, project):
    from aerolume.app.jobs import JobManager
    from aerolume.app.state import AppState
    state = AppState()
    state.set_project(project)
    jobs = JobManager(state)
    yield state, jobs
    jobs.shutdown(timeout=5)


def synthetic_frame(phase: str = "averaging", progress: float = 0.5):
    from aerolume.app.widgets.field_view import colorize, colormap_lut
    fd = synthetic_fields()
    rgba = colorize(fd.fields["speed"], colormap_lut(None, "speed"), 0.0, 35.0, mask=fd.mask)
    t = np.linspace(0, 10, 300)
    return SimpleNamespace(rgba=rgba, extent=fd.extent, field="speed", cmap=None, vmin=0.0, vmax=35.0,
                           units="m/s", t=10.0, step=1234, steps_per_s=400.0,
                           coeffs={"cd": 0.031, "cl": 0.52, "cm": -0.05},
                           history={"t": t, "cd": 0.03 + 0 * t, "cl": 0.5 + 0 * t, "cm": 0 * t},
                           progress=progress, eta_s=42.0, phase=phase, bodies=[ellipse(0.5, 0, 0.5, 0.06)],
                           particles=np.array([[0.0, 0.1], [1.0, 0.2]]),
                           streamlines=[np.array([[-1.0, 0.5], [3.0, 0.5]])])


class Recorder:
    """Collects every JobManager signal as (name, args)."""

    def __init__(self, jobs) -> None:
        self.events: list[tuple[str, tuple]] = []
        for name in ("jobQueued", "jobStarted", "jobProgress", "jobFrame", "jobFinished", "jobFailed",
                     "jobCancelled"):
            getattr(jobs, name).connect(lambda *a, n=name: self.events.append((n, a)))

    def names(self) -> list[str]:
        return [n for n, _ in self.events]


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_colorize_flips_to_image_order_and_paints_solids():
    from aerolume.app.widgets.field_view import colorize, colormap_lut
    lut = colormap_lut("viridis")
    vals = np.array([[0.0, 1.0], [2.0, np.nan]])          # row 0 = y min
    mask = np.array([[0, 0], [1, 0]])
    rgba = colorize(vals, lut, 0.0, 2.0, mask=mask, solid=(1, 2, 3, 255))
    assert rgba.shape == (2, 2, 4) and rgba.dtype == np.uint8
    # flipped: image row 0 is the solver's top row (y max)
    assert tuple(rgba[0, 0]) == (1, 2, 3, 255)               # masked cell
    assert tuple(rgba[0, 1]) == (1, 2, 3, 255)               # NaN cell
    assert tuple(rgba[1, 0, :3]) == tuple(lut[0])
    assert tuple(rgba[1, 1, :3]) == tuple(lut[128])


def test_ranges_lengths_and_ticks():
    from aerolume.app.widgets.field_view import format_length, nice_length, robust_range
    from aerolume.app.widgets.plots import nice_ticks
    a, b = robust_range(np.linspace(-3, 1, 1000), diverging=True)
    assert a == pytest.approx(-b) and b > 2.5
    a, b = robust_range(np.full((4, 4), 5.0))
    assert b > a
    assert nice_length(130.0) == 100.0 and nice_length(0.37) == 0.2
    assert format_length(0.05) == "5 cm" and format_length(0.2) == "200 mm" and format_length(2) == "2 m"
    ticks = nice_ticks(0.0, 1.0, 5)
    assert ticks[0] == 0.0 and ticks[-1] == pytest.approx(1.0) and len(ticks) in (5, 6)


def test_display_range_samples_near_the_body_and_starts_speed_at_zero():
    """A free stream of 14.6 m/s with a slow wake and a fast suction peak near the body.

    Over the whole domain the 1-99 % percentiles collapse to about U ± 20 %; the
    rule (speed from 0, 99.5th percentile inside bbox ± 1 length) must span the
    wake and the peak instead.
    """
    from aerolume.app.widgets.field_view import body_window, clipped_ends, display_range, robust_range
    extent = (-10.0, 30.0, -10.0, 10.0)
    ny, nx = 200, 400
    X, Y = np.meshgrid(np.linspace(*extent[:2], nx), np.linspace(*extent[2:], ny))
    speed = np.full((ny, nx), 14.6)
    speed[(np.abs(Y) < 0.3) & (X > 1) & (X < 3)] = 2.0          # wake just behind the body
    speed[(np.abs(Y - 0.3) < 0.2) & (X > 0) & (X < 1)] = 22.0  # suction peak over it
    body = [ellipse(0.5, 0.0, 0.5, 0.06)]
    whole = robust_range(speed)
    assert whole[0] > 12 and whole[1] < 16                     # the old behaviour: free stream only
    win = body_window(body, extent)
    assert win == pytest.approx((-1.0, 2.0, -1.06, 1.06))     # bbox ± 1 chord
    lo, hi = display_range("speed", speed, extent=extent, window=win)
    assert lo == 0.0 and hi == pytest.approx(22.0)
    assert clipped_ends(speed, lo, hi) == (False, False)
    assert clipped_ends(speed, 0.0, 20.0) == (False, True)
    a, b = display_range("cp", 1 - (speed / 14.6) ** 2, extent=extent, window=win)
    assert a == -b and 1.0 <= b <= 3.0                         # Cp: centred, clamped
    assert display_range("dye", speed) == (0.0, 1.0)


def test_live_range_tracker_converges_on_the_near_body_percentile():
    """Frames coloured by the solver's whole-domain range saturate; the tracker steers it to 0..p99.5."""
    from aerolume.app.widgets.field_view import LiveRangeTracker, colorize, colormap_lut, frame_clipped_ends
    fd = synthetic_fields()
    speed = fd.fields["speed"]                                  # 15..30 m/s, a slow wake
    lut = colormap_lut("viridis")
    body = [ellipse(0.5, 0, 0.5, 0.06)]
    tracker = LiveRangeTracker()
    vmin, vmax = 12.0, 16.0                                     # saturated: the peak is beyond the range
    for _ in range(30):
        rgba = colorize(speed, lut, vmin, vmax, mask=fd.mask)
        frame = SimpleNamespace(rgba=rgba, extent=fd.extent, field="speed", cmap="viridis", vmin=vmin,
                                vmax=vmax, bodies=body)
        vmin, vmax = tracker.update(frame)
    from aerolume.app.widgets.field_view import body_window, window_slices
    rows, cols = window_slices(speed.shape, fd.extent, body_window(body, fd.extent))
    near = speed[rows, cols][fd.mask[rows, cols] < 0.5]
    assert vmin == 0.0
    # grown past the peak, then eased back to within the 6 % dead band above it
    p995 = float(np.percentile(near, 99.5))
    assert p995 * 0.99 <= vmax <= p995 / (1 - LiveRangeTracker.DEAD_BAND) * 1.01
    below, above = frame_clipped_ends(colorize(speed, lut, 0.0, 20.0, mask=fd.mask), lut, field="speed")
    assert above and not below                                  # magnitudes never extend below 0
    frame.field = "vorticity"
    assert tracker.update(frame) is None                         # other fields keep the solver's range


def test_number_formatting():
    from aerolume.app.widgets.kpi import (
        decimals_for,
        fmt_err,
        fmt_fixed,
        fmt_re,
        headline,
        pretty,
        subscripts,
    )
    from aerolume.app.widgets.plots import decade_scale
    assert fmt_re(8.22e5) == "8.22×10⁵"
    assert fmt_re(1500) == "1,500"
    assert fmt_re(float("nan")) == "—"
    assert fmt_err(6.1e-5) == "± 0.000061"
    assert fmt_err(0.0024) == "± 0.0024"
    assert fmt_err(float("nan")) == ""
    assert fmt_fixed(0, 2) == "0.00" and fmt_fixed(-4, 2) == "−4.00" and fmt_fixed(-1e-9, 3) == "0.000"
    assert decimals_for([0.0300, 0.5]) == 4 and decimals_for([16.5]) == 2 and decimals_for([1234.0]) == 0
    assert pretty("Re = 9.99×10^5 -- capped") == "Re = 9.99×10⁵ — capped"
    w = ("This is a 2D simulation at Re = 9.99×10^5 with boundary layers thinner than the grid: flow "
         "patterns are reliable; absolute drag much less so.")
    assert headline(w) == "This is a 2D simulation at Re = 9.99×10⁵ with boundary layers thinner than the grid"
    assert headline("Not converged by t = 12 L/U (Cd, target ±5 %: noisy). The averages are usable.") == \
        "Not converged by t = 12 L/U (Cd, target ±5 %: noisy)"
    assert subscripts("Drag coefficient Cd") == "Drag coefficient C<sub>d</sub>"
    assert subscripts("Cl,max at α") == "C<sub>l,max</sub> at α"
    assert subscripts("Climb α 8° & CdA") == "Climb α 8° &amp; C<sub>d</sub>A"
    assert decade_scale([0.0, 8e-5]) == (pytest.approx(1e5), " ×10⁻⁵")
    assert decade_scale([0.2, 1.5]) == (1.0, "")


def test_data_table_aligns_decimal_points(qapp):
    """Mixed decimal counts in one column are padded so the decimal points line up."""
    from aerolume.app.widgets.kpi import DataTable, decimal_tail
    assert decimal_tail("0.0964") == 5 and decimal_tail("12") == 0 and decimal_tail("−4.72") == 3
    assert decimal_tail("+78.4 %") == 4 and decimal_tail("+100 %") == 2
    assert decimal_tail("—") is None and decimal_tail("9.99×10⁵") is None and decimal_tail("yes") is None
    t = DataTable(["Quantity", "Mean", "Unit"])
    t.set_data(["Quantity", "Mean", "Unit"],
               [["Cd", "0.0964", ""], ["Drag", "12.59", "N/m"], ["Power", "183", "W/m"], ["Re", "9.99×10⁵", ""],
                ["St", "—", ""]])
    assert [t.cell_pad(i, 1) for i in range(5)] == [0, 2, 5, 0, 5]
    assert t.column_texts(1) == ["0.0964", "12.59", "183", "9.99×10⁵", "—"]   # the text itself is unchanged
    t.resize(400, 200)
    t.show()
    qapp.processEvents()
    assert not t.grab().isNull()                                        # the padded paint path runs


def test_relabelled_axis_never_rescales_ticks(qapp):
    """pyqtgraph 0.14 stores a x1000 SI scale when a labelled 0..1 axis is restyled; ours must not."""
    from aerolume.app.widgets.plots import BarChart
    bars = BarChart()
    bars.resize(400, 300)
    bars.set_bars(["a", "b"], [0.0964, 0.172])
    for _ in range(2):                                          # restyle while the range spans 0..0.2
        bars.set_labels(ylabel="Drag coefficient C<sub>d</sub>")
        bars.set_bars(["a", "b"], [0.0964, 0.172])
    ax = bars.getPlotItem().getAxis("left")
    assert ax.autoSIPrefixScale == 1.0 and "(x" not in ax.labelString()
    assert ax.tickStrings([-0.5, 0.0, 0.5], 1.0, 0.5) == ["−0.5", "0", "0.5"]   # real minus sign


def test_history_range_ignores_startup_spike():
    from aerolume.app.widgets.plots import robust_history_range
    t = np.linspace(0, 20, 2001)
    cd = 0.11 + 0.002 * np.sin(3 * t)
    cd[:5] = 25.0                                        # impulsive start
    lo, hi = robust_history_range(t, cd)
    assert 0.10 < lo < 0.108 and 0.112 < hi < 0.12      # the spike does not set the scale
    assert robust_history_range(t[:1], cd[:1]) is None
    lo, hi = robust_history_range(t, np.full_like(t, 0.5))
    assert lo < 0.5 < hi                                 # a flat trace still gets a finite band


def test_result_text_helpers():
    from aerolume.app.pages.compare import fmt_pct
    from aerolume.app.pages.results import cp_range, short_device
    assert short_device("GPU · NVIDIA GeForce RTX 4060 Ti") == "RTX 4060 Ti"
    assert short_device("NVIDIA GeForce RTX 4060 Ti (8.0 GB)") == "RTX 4060 Ti"
    assert short_device("cuda") == "CUDA"
    assert cp_range(-0.4) == (-1.0, 1.0) and cp_range(-2.0) == (-2.0, 2.0) and cp_range(-9.0) == (-3.0, 3.0)
    assert fmt_pct(-10.0) == "−10.0 %" and fmt_pct(4.25) == "+4.2 %" and fmt_pct(None) == "—"


def test_split_loop_and_surface_curves():
    from aerolume.app.pages.results import split_loop, surface_curves
    p = ellipse(0.5, 0.0, 0.5, 0.1, 100)                     # CCW
    up, lo = split_loop(p[:, 0], p[:, 1])
    assert np.all(p[up, 1] >= -1e-12) and np.all(p[lo, 1] <= 1e-12)
    up2, _ = split_loop(p[::-1, 0], p[::-1, 1])            # CW input: same geometry split
    assert np.all(p[::-1][up2, 1] >= -1e-12)
    res = synthetic_cfd("x")
    cur = surface_curves(res.surface[0], alpha_deg=0.0, about=(0.25, 0.0), length=1.0)
    xu = cur["upper"][0]
    assert xu.min() == pytest.approx(0.0, abs=1e-9) and xu.max() == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(xu) >= 0)


# ---------------------------------------------------------------------------
# FieldView
# ---------------------------------------------------------------------------

def test_field_view_frame_hover_and_zoom(qapp):
    from aerolume.app.widgets.field_view import FieldView
    v = FieldView()
    v.resize(800, 400)
    v.show()
    v.set_frame(synthetic_frame())
    pump(qapp)
    assert v.has_image()
    # equal aspect: 4 m x 2 m extent in 800 x 400 px -> about 188-200 px per metre
    xf = v.world_to_screen()
    assert xf.m11() == pytest.approx(-xf.m22())
    # value under a free-stream point recovered from the colours (no raw values for live frames)
    val = v.value_at(-0.8, 0.8)
    assert val is not None and val == pytest.approx(30.0, abs=0.5)
    assert v.value_at(0.5, 0.0) is None                       # inside the solid
    s0 = v.world_to_screen().m11()
    v.zoom_by(2.0, animate=False)
    assert v.world_to_screen().m11() == pytest.approx(2 * s0)
    v.fit()
    assert v.world_to_screen().m11() == pytest.approx(s0)
    img = v.grab()
    assert img.width() == 800
    v.close()


def test_field_view_raw_values_and_focus(qapp):
    from aerolume.app.widgets.field_view import FieldView
    fd = synthetic_fields()
    v = FieldView()
    v.resize(600, 300)
    vmin, vmax = v.set_field(fd.fields["u"], fd.extent, field="u", mask=fd.mask)
    assert vmax > vmin
    assert v.value_at(-0.9, 0.9) == pytest.approx(30.0, rel=1e-3)
    v.set_focus_rect((0.0, 1.0, -0.25, 0.25))
    assert v.world_to_screen().m11() > 400                    # zoomed onto the body region


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

def test_jobmanager_runs_saves_and_signals(qapp, app_state):
    state, jobs = app_state
    rec = Recorder(jobs)
    changed: list[str] = []
    state.resultChanged.connect(changed.append)
    worker_threads = []

    def fake_run(design, sim, *, progress, frame, cancel, view):
        worker_threads.append(threading.current_thread())
        for i in range(5):
            progress((i + 1) / 5, f"step {i}")
            time.sleep(0.02)
        frame(synthetic_frame("done", 1.0))
        assert view is jobs.view
        return synthetic_cfd(sim.id, cd=0.0411)

    jobs.runner = fake_run
    finished_result = []
    jobs.jobFinished.connect(lambda sid, res: finished_result.append(res))
    jobs.submit("new1")
    assert jobs.running_id() == "new1"
    assert jobs.wait(10)
    names = rec.names()
    assert names[0] == "jobQueued" and "jobStarted" in names and "jobFinished" in names
    assert "jobProgress" in names and "jobFrame" in names
    assert worker_threads and worker_threads[0] is not threading.main_thread()
    sim = state.project.simulations["new1"]
    assert sim.status == "done" and sim.finished
    assert changed == ["new1"]
    loaded = state.project.load_result("new1")
    assert loaded.cd.mean == pytest.approx(0.0411)
    assert jobs.result("new1").cd.mean == pytest.approx(0.0411)
    assert jobs.result("new1") is finished_result[0]            # served from memory, not re-read
    finished = next(a for n, a in rec.events if n == "jobFinished")
    assert finished[0] == "new1" and isinstance(finished[1], CFDResult)


def test_jobmanager_cancel_keeps_partial_result(qapp, app_state):
    state, jobs = app_state
    rec = Recorder(jobs)
    started = threading.Event()

    def slow_run(design, sim, *, progress, frame, cancel, view):
        started.set()
        while not cancel.is_set():
            time.sleep(0.01)
        r = synthetic_cfd(sim.id)
        r.warnings.append("Stopped early by the user.")
        return r

    jobs.runner = slow_run
    jobs.submit("new1")
    assert started.wait(5)
    jobs.cancel("new1")
    assert jobs.wait(10)
    assert "jobCancelled" in rec.names() and "jobFinished" not in rec.names()
    assert state.project.simulations["new1"].status == "cancelled"
    assert state.project.has_result("new1")


def test_jobmanager_shutdown_records_cancelled_before_the_event_loop_ends(qapp, app_state):
    """Window close: the worker's queued 'done' never arrives, the status must still say cancelled."""
    state, jobs = app_state
    rec = Recorder(jobs)
    started = threading.Event()

    def slow_run(design, sim, *, progress, frame, cancel, view):
        started.set()
        while not cancel.is_set():
            time.sleep(0.01)
        return synthetic_cfd(sim.id)

    jobs.runner = slow_run
    jobs.submit("new1")
    assert started.wait(5)
    jobs.shutdown(timeout=5)                         # no processEvents() in between, as at exit
    assert state.project.simulations["new1"].status == "cancelled"
    assert state.project.has_result("new1") and "jobCancelled" in rec.names()
    assert jobs.running_id() is None
    pump(qapp)                                       # the late queued signal is ignored
    assert rec.names().count("jobCancelled") == 1
    assert state.project.simulations["new1"].status == "cancelled"


def test_jobmanager_failure_is_readable(qapp, app_state):
    state, jobs = app_state
    rec = Recorder(jobs)

    def bad_run(design, sim, **kw):
        raise ValueError("grid too coarse for this shape")

    jobs.runner = bad_run
    jobs.submit("new1")
    assert jobs.wait(10)
    failed = [a for n, a in rec.events if n == "jobFailed"]
    assert failed and failed[0][0] == "new1" and "grid too coarse" in failed[0][1]
    sim = state.project.simulations["new1"]
    assert sim.status == "failed" and "grid too coarse" in sim.error
    assert not state.project.has_result("new1")


def test_jobmanager_queue_is_fifo_and_queued_jobs_cancel(qapp, app_state):
    state, jobs = app_state
    order: list[str] = []
    gate = threading.Event()

    def run(design, sim, *, progress, frame, cancel, view):
        order.append(sim.id)
        gate.wait(5)
        return synthetic_cfd(sim.id)

    jobs.runner = run
    jobs.submit("new1")
    jobs.submit("cfd1")
    jobs.submit("cfd2")
    assert jobs.running_id() == "new1" and jobs.queued_ids() == ["cfd1", "cfd2"]
    assert state.project.simulations["cfd2"].status == "queued"
    jobs.cancel("cfd1")
    assert jobs.queued_ids() == ["cfd2"]
    # it never ran: back to the status it had before it was queued (done, with its result)
    assert state.project.simulations["cfd1"].status == "done"
    gate.set()
    assert jobs.wait(10)
    assert order == ["new1", "cfd2"]


def test_jobmanager_project_switch_cancels_and_resets(qapp, app_state, tmp_path):
    state, jobs = app_state
    old = state.project
    gate = threading.Event()

    def run(design, sim, *, progress, frame, cancel, view):
        while not cancel.is_set() and not gate.is_set():
            time.sleep(0.01)
        return synthetic_cfd(sim.id)

    jobs.runner = run
    jobs.submit("new1")
    jobs.submit("cfd2")
    assert old.simulations["cfd2"].status == "queued"
    pm = _project_mod()
    state.set_project(pm.Project.create(tmp_path / "Other.aerolume", "Other"))
    assert jobs.queued_ids() == []
    assert old.simulations["cfd2"].status == "done"            # it has a result from before
    assert jobs.wait(10)
    assert old.simulations["new1"].status == "cancelled"       # the running one was stopped
    assert old.has_result("new1")                              # ... and kept its partial result


def test_jobmanager_missing_solver_fails_gracefully(qapp, app_state, monkeypatch):
    """With no runner override and no simulation package the job fails with a message, not a crash."""
    import aerolume.app.jobs as jobs_mod
    _, jobs = app_state

    def missing(*a, **k):
        raise ModuleNotFoundError("No module named 'aerolume.simulation.runner'", name="aerolume.simulation.runner")

    monkeypatch.setattr(jobs_mod, "execute_simulation", missing)
    rec = Recorder(jobs)
    jobs.submit("new1")
    assert jobs.wait(10)
    msg = next(a for n, a in rec.events if n == "jobFailed")[1]
    assert "missing" in msg.lower()


# ---------------------------------------------------------------------------
# LiveWorker
# ---------------------------------------------------------------------------

class FakeSession:
    def __init__(self, design, sim, *, device=None, view=None) -> None:
        self.posts: list[dict] = []
        self.closed = False
        self.ticks = 0
        # the explore plan the Run panel describes while the session runs (smaller than a run's)
        self.plan = SimpleNamespace(solver_config=SimpleNamespace(nx=320, ny=160), cells=51_200, re_sim=1000.0)

    def post(self, **changes) -> None:
        self.posts.append(changes)

    def tick(self, max_ms: float = 33.0):
        self.ticks += 1
        time.sleep(0.005)
        f = synthetic_frame("live")
        f.step = self.ticks
        return f

    def close(self) -> None:
        self.closed = True


def test_live_worker_ticks_posts_and_stops(qapp, project):
    from aerolume.app.jobs import LiveWorker
    sessions: list[FakeSession] = []

    def factory(design, sim, **kw):
        s = FakeSession(design, sim, **kw)
        sessions.append(s)
        return s

    lw = LiveWorker()
    lw.session_factory = factory
    frames = []
    lw.frameReady.connect(frames.append)
    d = next(iter(project.designs.values()))
    sim = project.simulations["cfd1"]
    lw.post(alpha_deg=1.0)                          # before start: buffered
    lw.start(d, sim)
    lw.post(alpha_deg=5.0)
    deadline = time.monotonic() + 5
    while len(frames) < 3 and time.monotonic() < deadline:
        pump(qapp, 0.02)
    assert len(frames) >= 3 and frames[-1].phase == "live"
    lw.post(speed=12.0)
    lw.stop()
    pump(qapp, 0.05)
    s = sessions[0]
    assert s.closed and not lw.is_running()
    assert {"speed": 12.0} in s.posts and {"alpha_deg": 5.0} in s.posts
    # frames are coalesced: never more frames delivered than ticks run
    assert len(frames) <= s.ticks


def test_live_worker_reports_construction_failure(qapp, project):
    from aerolume.app.jobs import LiveWorker

    def broken(*a, **k):
        raise RuntimeError("CUDA error: no kernel image")

    lw = LiveWorker()
    lw.session_factory = broken
    errors = []
    lw.failed.connect(errors.append)
    lw.start(next(iter(project.designs.values())), project.simulations["cfd1"])
    deadline = time.monotonic() + 5
    while not errors and time.monotonic() < deadline:
        pump(qapp, 0.02)
    assert errors and "GPU" in errors[0]


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

PAGES = (("simulate", "SimulatePage"), ("results", "ResultsPage"), ("compare", "ComparePage"),
         ("report", "ReportPage"))


@pytest.mark.parametrize("module,cls", PAGES)
def test_pages_show_empty_state_without_project(qapp, module, cls):
    _project_mod()
    import importlib

    from aerolume.app.jobs import JobManager
    from aerolume.app.state import AppState
    state = AppState()
    jobs = JobManager(state)
    page = getattr(importlib.import_module(f"aerolume.app.pages.{module}"), cls)(state, jobs)
    page.resize(1200, 800)
    page.on_shown()
    assert page.stack.currentIndex() == 0
    assert page.empty.title.text()
    assert page.empty.action.text()                            # a call to action is always offered
    page.close()


def test_simulate_page_shows_frames_and_progress(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    assert page.stack.currentIndex() == 1
    jobs.jobFrame.emit("cfd1", synthetic_frame())
    pump(qapp)
    assert page.view.has_image()
    assert page.coef_values["cd"].text() == "0.03100"
    assert page.coef_values["ld"].text() == "16.8"
    assert page.phase_value.text() == "Averaging"
    assert page.eta_value.text() == "42 s"
    assert page.rate_value.text() == "400 steps/s"
    assert page.view.caption().startswith("t = 10")
    # the monitor plot keeps a usable height inside the scrolling inspector
    assert page.history.height() >= 200
    # speed frames steer the solver's colour range to 0 .. near-body percentile
    assert jobs.view.range_mode == "fixed" and jobs.view.vmin == 0.0 and 25 < jobs.view.vmax < 35
    # the colour-map list names the maps; the field's recommended one is marked
    assert page.cmap_combo.currentText() == "Viridis (default)" and page.cmap_combo.currentData() is None
    # view toggles are bound to the shared ViewSettings
    page.field_combo.setCurrentIndex(page.field_combo.findData("vorticity"))
    page.particles_btn.setChecked(True)
    assert jobs.view.field == "vorticity" and jobs.view.particles
    assert jobs.view.range_mode == "auto"            # a speed range must not colour vorticity
    assert page.cmap_combo.currentText().endswith("(default)") and "Viridis" not in page.cmap_combo.currentText()
    page.close()


def test_simulate_page_time_and_remaining_follow_the_runner(qapp, app_state, monkeypatch):
    """One source for t_end (the runner's message, extended past the plan); planner ETA early on."""
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("cfd1")
    page = SimulatePage(state, jobs)
    page._job_plans["cfd1"] = SimpleNamespace(t_end=80.0, est_seconds=308.0)   # what jobPlanned delivers
    monkeypatch.setattr(jobs, "running_id", lambda: "cfd1")
    elapsed = {"s": 3.0}
    monkeypatch.setattr(jobs, "progress_of", lambda sid: (0.01, "", elapsed["s"]))
    page.view.clear()
    jobs.jobProgress.emit("cfd1", 0.01, "Developing the flow (t = 0.8 of 80)")
    assert page.eta_value.text() == "5 min 05 s"                 # the plan's 308 s minus 3 s elapsed
    assert page.time_value.text() == "0.800 of 80.0 L/U"
    jobs.jobProgress.emit("cfd1", 0.74, "Averaging forces (t = 64.8 of 92)")
    assert page.time_value.text() == "64.8 of 92.0 L/U"           # the runner extended the run
    f = synthetic_frame("averaging", 0.74)
    f.t, f.eta_s = 64.8, 60.0
    jobs.jobFrame.emit("cfd1", f)
    assert page.time_value.text() == "64.8 of 92.0 L/U" and page.status_text.text() == "Running 74 %"
    remaining = page._eta
    assert remaining is not None and 60.0 < remaining < 305.0    # smoothed towards the measurement
    page.close()


def test_simulate_page_runs_via_jobmanager(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state

    def run(design, sim, *, progress, frame, cancel, view):
        progress(0.5, "half way")
        frame(synthetic_frame("done", 1.0))
        return synthetic_cfd(sim.id)

    jobs.runner = run
    logs: list[tuple[str, str, str]] = []
    state.messageLogged.connect(lambda lvl, src, txt: logs.append((lvl, src, txt)))
    state.select_simulation("new1")
    page = SimulatePage(state, jobs)
    page.handle_action("run")
    assert jobs.wait(10)
    pump(qapp)
    assert state.project.simulations["new1"].status == "done"
    # the synthetic result is converged before its planned end: the phase says it stopped early
    assert page.results_btn.isVisibleTo(page) and page.phase_value.text() in ("Done", "Converged (stopped early)")
    assert page.status_text.text().startswith("Done")
    assert page.coef_values["cd"].text() == "0.03000  ± 0.00015"    # rounded to its error, as in Results
    logged = [(lvl, src, txt) for lvl, src, txt in logs if src == "Solver"]
    assert any("Cd 0.03000" in txt for _, _, txt in logged)          # run milestones go to Messages
    assert any(lvl == "warning" and "Re capped" in txt for lvl, _, txt in logged)


def test_simulate_page_explore_mode(qapp, app_state):
    import aerolume.app.pages.simulate as sim_mod
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = sim_mod.SimulatePage(state, jobs)
    sessions: list[FakeSession] = []
    page.live.session_factory = lambda d, s, **kw: sessions.append(FakeSession(d, s)) or sessions[-1]
    if sim_mod.lazy_import("aerolume.simulation.live") is None:
        # the real live module is optional here; fake its presence for the page's guard
        sys.modules.setdefault("aerolume.simulation.live", SimpleNamespace(LiveSession=FakeSession))
        added = True
    else:
        added = False
    try:
        page.handle_action("live")
        deadline = time.monotonic() + 5
        while not page.view.has_image() and time.monotonic() < deadline:
            pump(qapp, 0.02)
        assert page.view.has_image() and page.live.is_running()
        pump(qapp, 0.05)
        assert page.grid_value.text().startswith("320 × 160") and "explore" in page.grid_value.text()
        page.alpha_slider.setValue(12)                          # +6 deg
        pump(qapp, 0.05)
        assert any(p.get("alpha_deg") == 6.0 for p in sessions[0].posts)
        jobs.runner = lambda design, sim, **kw: synthetic_cfd(sim.id)
        page.promote_live_settings()                            # cfd1 has a result: a new sim is made
        assert not page.live.is_running()
        new = [s for s in state.project.simulations.values() if s.conditions.alpha_deg == 6.0]
        assert new and new[0].id != "cfd1"
        assert jobs.is_busy(new[0].id)
        assert jobs.wait(10)
        assert new[0].status == "done" and state.project.simulations["cfd1"].conditions.alpha_deg == 2.0
    finally:
        page.live.stop()
        if added:
            sys.modules.pop("aerolume.simulation.live", None)


def test_explore_start_blanks_the_last_runs_rows_until_its_first_frame(qapp, app_state):
    """A 3D explore takes seconds to build its grid; meanwhile the Run panel showed the last run's
    "Time 11.1 of 20.0 L/U" and its mean C_D under "Live". Starting explore blanks them."""
    import aerolume.app.pages.simulate as sim_mod
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = sim_mod.SimulatePage(state, jobs)
    page.live.session_factory = lambda d, s, **kw: FakeSession(d, s)
    added = sim_mod.lazy_import("aerolume.simulation.live") is None
    if added:
        sys.modules.setdefault("aerolume.simulation.live", SimpleNamespace(LiveSession=FakeSession))
    try:
        page.time_value.setText("11.1 of 20.0 L/U")
        page.rate_value.setText("386 steps/s")
        page.coef_values["cd"].setText("0.877 ± 0.013")
        page.monitor_note.setText("Mean of the last run")
        page.toggle_live()                    # no event processed yet: no frame can have arrived
        assert page.time_value.text() == "—" and page.rate_value.text() == "—"
        assert page.coef_values["cd"].text() == "—" and page.monitor_note.text() == "Latest time step"
        deadline = time.monotonic() + 5
        while not page.view.has_image() and time.monotonic() < deadline:
            pump(qapp, 0.02)
        assert page.view.has_image() and page.time_value.text() not in ("—", "11.1 of 20.0 L/U")
    finally:
        page.live.stop()
        if added:
            sys.modules.pop("aerolume.simulation.live", None)
        page.close()


def test_explore_notices_reach_the_message_log(qapp, app_state):
    """``LiveFrame.notices`` (edits the session could not carry out) are logged as warnings,
    once each, so a control that had no effect says why."""
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    logs: list[tuple[str, str]] = []
    state.messageLogged.connect(lambda lvl, src, txt: logs.append((lvl, txt)))
    state.select_simulation("cfd1")
    page = SimulatePage(state, jobs)
    frame = synthetic_frame("live", 0.0)
    frame.notices = ["A closed box has no free stream: the speed slider has no effect.", "  "]
    page._on_live_frame(frame)
    frame.notices = None
    page._on_live_frame(frame)
    hits = [lvl for lvl, txt in logs if "closed box has no free stream" in txt]
    assert hits == ["warning"]
    page.close()


def test_simulate_slider_speed_roundtrip(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = SimulatePage(state, jobs)
    assert page.slider_speed() == pytest.approx(30.0)
    assert page.slider_alpha() == pytest.approx(2.0)


def test_results_page_cfd(qapp, app_state):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    assert page.stack.currentIndex() == 1
    t = page.summary_table
    assert [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())] == \
        ["Quantity", "Mean", "± Std. error", "RMS", "Min", "Max", "Unit"]
    assert t.item(0, 0).text() == "Drag coefficient Cd"
    # every row is rounded to its standard error: the error to two significant figures,
    # mean / RMS / min / max to the same decimal place (Cd ± 0.00015, Cl ± 0.0020, Cm ± 0.00010)
    assert t.item(0, 1).text() == "0.03000" and t.item(0, 2).text() == "0.00015"
    assert t.item(0, 4).text() == "0.02600" and t.item(0, 5).text() == "0.03400"
    assert t.item(1, 1).text() == "0.5000" and t.item(1, 2).text() == "0.0020"
    assert t.item(2, 1).text() == "−0.05000"
    # ... and the table pads them so the decimal points line up (5 decimals vs 4: one cell)
    assert t.cell_pad(1, 1) == t.cell_pad(0, 1) + 1
    assert t.find_row("Lift-to-drag ratio L/D") > 0 and t.find_row("Lift") > 0     # an airfoil: L/D shown
    assert "chord 1.000 m" in page.summary_note.text()
    drag = t.find_row("Drag")
    assert t.item(drag, 6).text() == "N/m"
    assert page.status_text.text() == "Converged" and page.status_text.kind == "good"
    facts = page.facts.full_text()
    assert "simulated" in facts and "1024×512 cells, 128 per chord" in facts   # Re capped: both shown
    # one warning line, the full text in its tooltip
    assert page.warning_line.isVisibleTo(page) and "Re capped" in page.warning_line.summary()
    assert "Re capped" in page.warning_line.toolTip()
    from aerolume.app.pages.results import TAB_KEYS
    for key in ("summary", "forces", "surface", "fields", "sensitivity", "data"):
        assert page.tabs.isTabVisible(TAB_KEYS.index(key)), key
    assert not page.tabs.isTabVisible(TAB_KEYS.index("polar"))  # no polar tab for CFD
    page.handle_action("fields")
    pump(qapp)
    assert page.field_view.has_image()
    assert page.fld_combo.count() >= 4
    page.surf_view.set_current("arrows")
    assert page.surf_stack.currentIndex() == 1
    page.surf_view.set_current("plots")
    assert page.surf_stack.currentIndex() == 0
    assert "×10" in page.cf_plot.getPlotItem().getAxis("left").labelText    # |Cf| ~ 4e-3: scaled axis
    assert page.sens_stack.currentIndex() == 0                    # sensitivity present
    assert page.data_table.rowCount() > 20


def test_results_page_polar_and_empty_states(qapp, app_state):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = app_state
    state.select_simulation("pol1")
    page = ResultsPage(state, jobs)
    from aerolume.app.pages.results import TAB_KEYS
    assert page.tabs.isTabVisible(TAB_KEYS.index("polar")) and not page.tabs.isTabVisible(TAB_KEYS.index("forces"))
    assert page.status_text.text() == "16 of 17 points converged" and page.status_text.kind == "warn"
    assert page.summary_table.find_row("Maximum lift Cl,max") == 0
    assert page.polar_table.rowCount() == 17
    assert page.surf_alpha.count() == 16
    state.select_simulation("new1")
    assert page.stack.currentIndex() == 0
    assert "not been run" in page.empty.title.text()


def test_results_page_exports(qapp, app_state, tmp_path, monkeypatch):
    from aerolume.app.pages import results as results_mod
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = results_mod.ResultsPage(state, jobs)
    calls = []
    fake = SimpleNamespace(history_csv=lambda r, p: calls.append((r.sim_id, p)) or Path(p).write_text("t,cd\n"))
    monkeypatch.setitem(sys.modules, "aerolume.export", fake)
    target = tmp_path / "h.csv"
    monkeypatch.setattr(results_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    page.export("history_csv")
    deadline = time.monotonic() + 5
    while not calls and time.monotonic() < deadline:
        pump(qapp, 0.02)
    pump(qapp, 0.05)
    assert calls and calls[0][0] == "cfd1" and target.exists()
    assert "Saved" in page.export_status.text()


def test_compare_page(qapp, app_state):
    from aerolume.app.pages.compare import ComparePage, judge
    state, jobs = app_state
    state.compare_ids[:] = ["cfd1", "cfd2"]
    page = ComparePage(state, jobs)
    page.resize(1600, 1000)
    assert page.stack.currentIndex() == 1 and page.inner.currentIndex() == 1
    keys = [r["key"] for r in page._rows]
    assert "cd" in keys and "ld" in keys
    # Quantity | baseline | variant | change | Unit
    assert page.table.columnCount() == 5
    assert page.table.horizontalHeaderItem(1).text() == "Cruise (baseline)"
    cd_row = next(r for r in page._rows if r["key"] == "cd")
    cell = next(c for c in cd_row["cells"] if c["sim_id"] == "cfd2")
    assert judge(cd_row, cell, None) is True                       # lower drag is better
    col = page.delta_column("cfd2")
    assert col == 3 and page.table.item(keys.index("cd"), col).text() == "−10.0 %"
    assert page.table.item(keys.index("cd"), 2).text() == "0.02700"
    cl_row = next(r for r in page._rows if r["key"] == "cl")
    cl_cell = next(c for c in cl_row["cells"] if c["sim_id"] == "cfd2")
    assert judge(cl_row, cl_cell, None) is None and judge(cl_row, cl_cell, "lower") is False
    page.handle_action("fields")
    assert page.tabs.currentIndex() == 4
    assert len(page._thumb_views) == 2
    page.diff_check.setChecked(True)                               # variant minus baseline
    assert len(page._thumb_views) == 2 and page._thumb_views[1].has_image()
    assert "baseline" in page._thumb_views[1]._title
    # baseline switch reorders
    page.baseline_combo.setCurrentIndex(1)
    page._update_views()
    assert next(iter(page._results)) == "cfd2"


def test_compare_default_selection_is_like_with_like_and_names_are_unique(qapp, app_state):
    """With nothing picked, Compare opens on the CFD runs, not the polar (a polar has no
    single Cd, so a polar baseline gave a table of dashes); runs that share a name are
    labelled with their design."""
    from aerolume.app.pages.compare import ComparePage
    state, jobs = app_state
    state.compare_ids[:] = []
    page = ComparePage(state, jobs)
    assert sorted(state.compare_ids) == ["cfd1", "cfd2"]
    assert all(state.project.simulations[s].analysis == "cfd" for s in state.compare_ids)
    sims = state.project.simulations
    sims["cfd2"].name = sims["cfd1"].name                   # e.g. two designs' "CFD · 30 m/s"
    d1 = state.project.designs[sims["cfd1"].design_id].name
    assert page._name("cfd1") == f"{d1} · {sims['cfd1'].name}"
    page.close()


def test_compare_page_flags_unconverged_runs_and_labels_the_chart(qapp, app_state):
    from aerolume.app.pages.compare import ComparePage
    state, jobs = app_state
    jobs.result("cfd2").converged = False              # the cached object is what the page reads
    state.compare_ids[:] = ["cfd1", "cfd2"]
    page = ComparePage(state, jobs)
    assert page.unconverged_ids() == {"cfd2"}
    col = 2                                             # Quantity | Cruise (baseline) | Cruise B | Δ | Unit
    head = page.table.horizontalHeaderItem(col)
    assert head.text() == "Cruise B" and not head.icon().isNull() and "not statistically converged" in \
        head.toolTip()
    assert page.table.horizontalHeaderItem(1).icon().isNull()
    page.bar_metric.setCurrentIndex(page.bar_metric.findData("cd"))
    assert "C<sub>d</sub>" in page.bars.getPlotItem().getAxis("left").labelText
    page.close()


def test_compare_resample_is_exact_on_the_same_grid():
    from aerolume.app.pages.compare import _resample
    fd = synthetic_fields()
    ny, nx = fd.fields["u"].shape
    x0, x1, y0, y1 = fd.extent
    xc = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    yc = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    np.testing.assert_allclose(_resample(fd.fields["u"], fd.extent, xc, yc), fd.fields["u"], atol=1e-9)
    shifted = _resample(fd.fields["u"], fd.extent, xc + 10.0, yc)    # entirely outside: NaN
    assert np.isnan(shifted).all()


def test_compare_page_pairs_single_pick_and_needs_two(qapp, app_state):
    from aerolume.app.pages.compare import ComparePage
    state, jobs = app_state
    state.compare_ids[:] = ["cfd1"]
    page = ComparePage(state, jobs)
    assert state.compare_ids == ["cfd1", "cfd2"]          # one pick is paired with a similar result
    assert page.inner.currentIndex() == 1
    page._set_all(False)
    assert page.inner.currentIndex() == 0


def test_report_page_generates_in_background(qapp, app_state, monkeypatch):
    from aerolume.app.pages.report import ReportPage
    state, jobs = app_state
    seen = {}

    def fake_generate(project, sim_ids, path, *, fmt="pdf", sections=None, title=None, author=None):
        seen.update(thread=threading.current_thread(), ids=sim_ids, fmt=fmt, sections=sections, title=title)
        Path(path).write_bytes(b"%PDF-1.4\n")
        return Path(path)

    fake = SimpleNamespace(SECTIONS=("summary", "results", "comparison", "polar", "methodology"),
                           generate=fake_generate)
    monkeypatch.setitem(sys.modules, "aerolume.report", fake)
    page = ReportPage(state, jobs)
    assert page.stack.currentIndex() == 1
    assert set(page.selected_ids()) == {"cfd1", "cfd2", "pol1"}
    assert list(page.section_boxes) == list(fake.SECTIONS)
    page.html_radio.setChecked(True)
    page.handle_action("generate")
    deadline = time.monotonic() + 5
    while page._call is not None and time.monotonic() < deadline:
        pump(qapp, 0.02)
    assert seen["thread"] is not threading.main_thread()
    assert seen["fmt"] == "html" and "comparison" in seen["sections"]
    assert page.last_path is not None and page.last_path.exists() and page.last_path.suffix == ".html"
    assert page.report_list.rowCount() == 1 and page.state_badge.text() == "Ready"
    assert page.report_paths() == [page.last_path]
    assert page.report_list.item(0, 1).text() == "HTML"


def test_report_page_title_and_sections_without_data(qapp, app_state, monkeypatch):
    from aerolume.app.pages.report import ReportPage, default_title
    state, jobs = app_state
    fake = SimpleNamespace(SECTIONS=("summary", "results", "sensitivity", "comparison"), generate=None)
    monkeypatch.setitem(sys.modules, "aerolume.report", fake)
    assert default_title("Wing and sedan study") == "Wing and sedan study"     # no "study — aerodynamic study"
    assert default_title("") == "Untitled — aerodynamic report"
    page = ReportPage(state, jobs)
    assert page.title_edit.text() == "Test study"
    # no simulation asked for a sensitivity map and no loaded result has one: the box is off
    assert not page.section_boxes["sensitivity"].isEnabled()
    assert not page.section_boxes["sensitivity"].isChecked()   # a greyed tick would read "included"
    assert "sensitivity" not in page.selected_sections()
    jobs.result("cfd1")                                    # loaded result with a map: exact answer
    page.refresh()
    assert page.section_boxes["sensitivity"].isEnabled()
    assert page.section_boxes["sensitivity"].isChecked()       # the default wish comes back
    assert "sensitivity" in page.selected_sections()
    # a section the user turned off stays off when it stops and starts applying again
    page.section_boxes["comparison"].setChecked(False)
    page._set_all(False)
    page._set_all(True)
    assert not page.section_boxes["comparison"].isChecked()
    page.section_boxes["comparison"].setChecked(True)
    # the title follows the project until the user edits it
    state.project.name = "Renamed"
    page.refresh()
    assert page.title_edit.text() == "Renamed"
    page.title_edit.setText("My own title")
    page.title_edit.textEdited.emit("My own title")
    state.project.name = "Again"
    page.refresh()
    assert page.title_edit.text() == "My own title"


def test_report_page_without_report_module(qapp, app_state, monkeypatch):
    from aerolume.app.pages.report import ReportPage
    state, jobs = app_state
    monkeypatch.setitem(sys.modules, "aerolume.report", SimpleNamespace(SECTIONS=("summary",)))
    page = ReportPage(state, jobs)
    page.generate()
    assert page.state_badge.text() == "Unavailable"
    assert page._call is None


# ---------------------------------------------------------------------------
# round-2 UI fixes: stale plans, polar runs, framing, road, vehicle tables
# ---------------------------------------------------------------------------

def _fake_plan_module(calls: list):
    """A stand-in for aerolume.simulation.plan whose plan reflects the simulation's speed."""
    def plan_simulation(design, sim, **_kw):
        calls.append(sim.conditions.speed)
        re_ = sim.conditions.reynolds(design.length_ref)
        return SimpleNamespace(re_real=re_, re_sim=re_, cells=1000, est_seconds=12.0 * sim.conditions.speed,
                               solver_config=SimpleNamespace(nx=50, ny=20), warnings=[], t_end=20.0,
                               t_avg_start=8.0, turbulence="laminar")
    return SimpleNamespace(plan_simulation=plan_simulation)


def _pump_until(qapp, cond, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        pump(qapp, 0.02)
    return cond()


def test_plan_key_follows_settings_not_identity():
    from aerolume.app.pages.simulate import plan_key
    d = simple_design()
    s = Simulation(name="A", design_id=d.id, conditions=FlowConditions(speed=30.0))
    k = plan_key(s, d)
    s.name, s.status = "Renamed", "done"
    assert plan_key(s, d) == k                      # bookkeeping does not change the plan
    s.conditions.speed = 5.0
    assert plan_key(s, d) != k                      # an edit in Setup does
    k5 = plan_key(s, d)
    d.length_ref = 2.0
    assert plan_key(s, d) != k5                     # so does an edit of the design


def test_simulate_plan_is_never_stale(qapp, app_state, monkeypatch):
    """Editing the speed re-plans; during a run the panel shows the runner's own plan."""
    from aerolume.app.pages.simulate import SimulatePage
    from aerolume.app.widgets.kpi import fmt_re
    state, jobs = app_state
    calls: list[float] = []
    monkeypatch.setitem(sys.modules, "aerolume.simulation.plan", _fake_plan_module(calls))
    logs: list[str] = []
    state.messageLogged.connect(lambda lvl, src, txt: logs.append(txt))
    state.select_simulation("new1")
    page = SimulatePage(state, jobs)
    sim = state.project.simulations["new1"]
    d = state.project.designs[sim.design_id]
    assert _pump_until(qapp, lambda: page.re_value.text() == fmt_re(sim.conditions.reynolds(d.length_ref)))
    sim.conditions.speed = 5.0
    state.update_simulation(sim)                     # what Setup does
    want = fmt_re(sim.conditions.reynolds(d.length_ref))
    assert _pump_until(qapp, lambda: page.re_value.text() == want), page.re_value.text()
    assert page.estimate_value.text() == "1 min 00 s"                # 12 s x 5 m/s: the 5 m/s plan
    # a run reports the plan it executes; the panel and the run log follow that plan
    monkeypatch.setattr(jobs, "running_id", lambda: "new1")
    job_plan = _fake_plan_module([]).plan_simulation(d, Simulation(name="x", design_id=d.id,
                                                                  conditions=FlowConditions(speed=7.0)))
    jobs.jobPlanned.emit("new1", job_plan)
    assert page.re_value.text() == fmt_re(job_plan.re_real)
    assert any("50 × 20 cells" in t and fmt_re(job_plan.re_sim) in t for t in logs)
    page.close()


def test_jobmanager_reports_the_runners_plan(qapp, app_state):
    _, jobs = app_state
    planned_seen = []
    jobs.jobPlanned.connect(lambda sid, plan: planned_seen.append((sid, plan, jobs.plan_of(sid))))
    marker = SimpleNamespace(t_end=10.0)

    def run(design, sim, *, progress, frame, cancel, view, planned):
        planned(marker)
        time.sleep(0.05)
        return synthetic_cfd(sim.id)

    jobs.runner = run
    jobs.submit("new1")
    assert jobs.wait(10)
    assert planned_seen and planned_seen[0][0] == "new1" and planned_seen[0][1] is marker
    assert planned_seen[0][2] is marker                       # plan_of() answers while the job runs
    assert jobs.plan_of("new1") is None                       # ... and forgets it afterwards


def test_parse_polar_point():
    from aerolume.app.pages.simulate import parse_polar_point
    assert parse_polar_point("Polar: alpha = 4 deg: Cl = 0.718, Cd = 0.00697") == (4.0, 0.718, 0.00697, True)
    a, cl, _cd, ok = parse_polar_point("Polar: alpha = -3.5 deg: not converged")
    assert a == -3.5 and not ok and np.isnan(cl)
    assert parse_polar_point("Polar sweep: 19 angles at Re = 1e6") is None


def test_simulate_polar_run_shows_polar_plots_not_a_stuck_viewport(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    pol = state.project.simulations["pol1"]
    pol.status = "new"
    state.update_simulation(pol)
    alphas = [float(a) for a in pol.airfoil.alphas()]

    def run(design, sim, *, progress, frame, cancel, view, **_kw):
        for i, a in enumerate(alphas[:6]):
            progress(0.99 * (i + 1) / len(alphas), f"Polar: alpha = {a:g} deg: Cl = {0.11 * (a + 2):.3f}, "
                                                    f"Cd = {0.006 + 0.0004 * a * a:.5f}")
            time.sleep(0.08)                     # the JobManager passes at most 15 progress updates / s
        return synthetic_polar(sim.id)

    jobs.runner = run
    state.select_simulation("pol1")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    assert page.vp_stack.currentIndex() == 1 and not page.toolbar.isVisible()    # no field controls
    seen = []
    jobs.jobProgress.connect(lambda *a: seen.append(len(page.polar_view.cl_alpha.series_names())))
    page.handle_action("run")
    assert jobs.wait(10)
    pump(qapp)
    assert any(n == 1 for n in seen)                                  # the lift curve filled in during the run
    assert page.polar_caption.text().startswith("Done: 16 of 17 angles converged")
    # one duration everywhere: the job's wall time (the status bar's), not the result's own 3 s
    from aerolume.app.widgets.kpi import fmt_duration
    took = jobs.job_seconds("pol1")
    assert took is not None and 0.3 < took < 10
    assert page.status_text.text() == f"Done in {fmt_duration(took)}" and page.results_btn.isVisibleTo(page)
    assert "Starting" not in page.view._placeholder[0]
    assert page.polar_monitor.isVisibleTo(page) and not page.history.isVisibleTo(page)
    assert page.polar_monitor.series_names() == ["Cd", "Cl"]
    assert page.monitor_note.text().startswith("At best L/D")
    page.close()


def test_simulate_monitor_axis_and_range_at_run_start(qapp, app_state, monkeypatch):
    """The first frames: a fixed 0..t_end axis and a range that ignores the impulsive start."""
    from aerolume.app.pages.simulate import SimulatePage
    from aerolume.app.widgets.plots import robust_history_range
    state, jobs = app_state
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("new1")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    monkeypatch.setattr(jobs, "running_id", lambda: "new1")
    jobs.jobStarted.emit("new1")
    jobs.jobPlanned.emit("new1", SimpleNamespace(t_end=40.0, est_seconds=60.0))
    assert page.history.x_limits() == (0.0, 40.0)
    f = synthetic_frame("transient", 0.01)
    t = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    f.history = {"t": t, "cd": np.array([95.0, 4.0, 1.5, 1.2, 1.1]), "cl": np.array([30.0, 1, 0.5, 0.4, 0.3]),
                 "cm": 0 * t}
    jobs.jobFrame.emit("new1", f)
    pump(qapp)
    _, hi = page.history.y_range("cd")
    assert hi < 10                                              # not 0..100 around the spike
    vb = page.history.panels["cd"].getViewBox()
    assert vb.viewRange()[0] == pytest.approx([0.0, 40.0], abs=1e-6)
    tt = np.linspace(0, 5, 501)
    cd = 0.2 + 0.01 * np.sin(tt)
    cd[:3] = 80.0
    assert robust_history_range(tt, cd, t_min=0.5)[1] < 0.3
    page.close()


def test_simulate_shows_saved_field_when_no_frame_arrived(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    jobs.runner = lambda design, sim, **kw: synthetic_cfd(sim.id)       # finishes without a frame
    state.select_simulation("new1")
    page = SimulatePage(state, jobs)
    page.handle_action("run")
    assert jobs.wait(10)
    pump(qapp)
    assert page.view.has_image() and page.view.caption().startswith("Time-averaged")


def test_field_view_frames_the_body_and_fills_the_view(qapp):
    from aerolume.app.widgets.field_view import FieldView, near_field_window
    extent = (-8.0, 23.0, -10.0, 10.0)                  # a typical airfoil domain: 31 x 20 m
    nx, ny = 310, 200
    vals = np.full((ny, nx), 30.0)
    v = FieldView()
    v.resize(1000, 700)
    v.show()
    body = ellipse(0.5, 0.0, 0.5, 0.06)
    v.set_bodies([body])
    v.set_field(vals, extent, field="speed")
    pump(qapp)
    r = v.plot_rect()
    s = v.world_to_screen().m11()
    assert s * 1.0 >= 250                               # the chord is hundreds of pixels, not 80
    x0, y1 = v.screen_to_world(r.topLeft())
    x1, y0 = v.screen_to_world(r.bottomRight())
    # no letterboxing: the field covers the plot area ...
    assert x0 >= extent[0] - 1e-6 and x1 <= extent[1] + 1e-6 and y0 >= extent[2] - 1e-6 and y1 <= extent[3] + 1e-6
    # ... and the body with some flow ahead of it and a chord of wake is in view
    assert x0 <= -0.25 and x1 >= 2.0 and y0 <= -0.3 and y1 >= 0.3
    assert near_field_window([body]) == pytest.approx((-1.0, 4.0, -1.06, 1.06))
    changes = []
    v.wholeDomainChanged.connect(changes.append)
    v.set_whole_domain(True)                            # the Fit toggle: the whole domain
    x0, y1 = v.screen_to_world(r.topLeft())
    x1, y0 = v.screen_to_world(r.bottomRight())
    assert x0 <= extent[0] and x1 >= extent[1] and y0 <= extent[2] and y1 >= extent[3]
    v.zoom_by(1.5, animate=False)                       # zooming by hand leaves whole-domain mode
    assert changes == [True, False] and not v.whole_domain()
    v.fit()
    assert v.world_to_screen().m11() == pytest.approx(s)
    v.close()


def test_field_view_draws_the_road_for_ground_runs(qapp):
    from aerolume.app.widgets.field_view import ROAD_BAND_PX, FieldView
    from aerolume.app.widgets.plots import qcolor, tokens
    extent = (-5.0, 25.0, 0.0, 12.0)                     # a vehicle domain ends at the road
    v = FieldView()
    v.resize(1000, 600)
    v.show()
    car = np.array([[0.0, 0.2], [4.6, 0.2], [4.6, 1.2], [3.0, 1.45], [1.0, 1.4], [0.0, 0.8]])
    v.set_bodies([car])
    v.set_ground(0.0)
    v.set_field(np.full((120, 300), 30.0), extent, field="speed")
    pump(qapp)
    sy = v.world_to_screen().map(0.0, 0.0)[1]
    gap = v.plot_rect().bottom() - sy
    assert ROAD_BAND_PX <= gap <= ROAD_BAND_PX + 8       # the road sits on the bottom, its band visible
    img = v.grab().toImage()
    surface = qcolor(tokens()["surface"]).rgb()
    band_row = int(sy) + int(ROAD_BAND_PX // 2)
    cols = [img.pixel(x, band_row) for x in range(200, 800)]
    assert any(c != surface for c in cols)              # hatching in the band
    assert sum(c == surface for c in cols) > len(cols) // 2   # ... over the surface colour, not the field
    v.close()


def _vehicle_project(project):
    d = Design(name="Sedan", kind="vehicle", bodies=[Body("body", ellipse(2.3, 0.8, 2.3, 0.6))],
               length_ref=4.6, force_ref=1.3, moment_ref=(2.3, 0.0))
    project.add_design(d)
    sim = Simulation(name="Highway", design_id=d.id, conditions=FlowConditions(speed=30.0), status="done",
                     id="car1")
    sim.ground.enabled = True
    project.add_simulation(sim)
    project.save()
    project.save_result("car1", synthetic_cfd("car1", cd=0.33, cl=-0.2))
    return d, sim


def test_results_vehicle_hides_ld_and_names_the_reference(qapp, app_state):
    from aerolume.app.pages.results import ResultsPage, conditions_items
    state, jobs = app_state
    _vehicle_project(state.project)
    state.select_simulation("car1")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    t = page.summary_table
    assert t.find_row("Lift-to-drag ratio L/D") == -1
    row = t.find_row("Lift (downforce if negative)")
    assert row > 0 and t.item(row, 1).text().startswith("−")        # the sign is kept: downforce
    assert "frontal height 1.30 m" in page.summary_note.text()
    assert "frontal height 1.30 m" in t.item(t.find_row("Lift coefficient Cl"), 0).toolTip()
    assert all(r[0] != "L/D" for r in [page.data_table.row_texts(i) for i in range(page.data_table.rowCount())])
    page.handle_action("fields")
    pump(qapp)
    assert page.field_view.ground() == 0.0
    # a polar result lists the swept range, not the simulation's single angle
    pol = state.project.simulations["pol1"]
    items = dict(conditions_items(jobs.result("pol1"), state.project.designs[pol.design_id], pol))
    assert "Angle of attack" not in items
    assert items["Angle range"] == "−4° to 12°, step 1°" and items["Ncrit"] == "9"
    page.close()


def test_compare_drops_ld_for_vehicles(qapp, app_state):
    from aerolume.app.pages.compare import ComparePage, adapt_rows
    state, jobs = app_state
    d, _ = _vehicle_project(state.project)
    s2 = Simulation(name="Highway B", design_id=d.id, status="done", id="car2")
    s2.ground.enabled = True
    state.project.add_simulation(s2)
    state.project.save_result("car2", synthetic_cfd("car2", cd=0.31, cl=-0.25))
    state.compare_ids[:] = ["car1", "car2"]
    page = ComparePage(state, jobs)
    keys = [r["key"] for r in page._rows]
    assert "ld" not in keys and "cd" in keys
    assert next(r for r in page._rows if r["key"] == "lift")["label"] == "Lift (downforce if negative)"
    assert all(v.ground() == 0.0 for v in page._thumb_views)
    rows = [{"key": "ld", "label": "L/D", "cells": [{"sim_id": "a", "value": 10.0}, {"sim_id": "b", "value": 2.0}]}]
    mixed = adapt_rows(rows, {"a": True, "b": False})
    assert mixed[0]["cells"][0]["value"] == 10.0 and mixed[0]["cells"][1]["value"] is None
    page.close()


# ---------------------------------------------------------------------------
# integration with the real export / report modules (skipped while they are absent)
# ---------------------------------------------------------------------------

def test_results_page_real_exports(qapp, app_state, tmp_path, monkeypatch):
    pytest.importorskip("aerolume.export")
    from aerolume.app.pages import results as results_mod
    state, jobs = app_state
    state.select_simulation("cfd1")
    page = results_mod.ResultsPage(state, jobs)
    for key, name in (("result_csv", "r.csv"), ("history_csv", "h.csv"), ("surface_csv", "s.csv"),
                      ("fields_vtk", "f.vti"), ("fields_npz", "f.npz")):
        target = tmp_path / name
        monkeypatch.setattr(results_mod.QFileDialog, "getSaveFileName", lambda *a, t=target, **k: (str(t), ""))
        page.export(key)
        deadline = time.monotonic() + 30
        while "Exporting" in page.export_status.text() and time.monotonic() < deadline:
            pump(qapp, 0.02)
        assert page.export_status.text() == f"Saved {name}", page.export_status.text()
        assert target.stat().st_size > 100


def _real_report_module():
    """The real ``aerolume.report``, never a stand-in an earlier test left behind.

    Report pages built by earlier tests import the module on a worker thread;
    one that finishes while a test has put a fake into ``sys.modules`` hands
    that fake to the page (and the package attribute), so a plain
    ``importorskip`` here found a module without ``generate`` depending on test
    order. Drop anything that is not the module file and import it again.
    """
    import importlib
    mod = sys.modules.get("aerolume.report")
    if mod is not None and not getattr(mod, "__file__", None):
        del sys.modules["aerolume.report"]
        import aerolume
        if getattr(aerolume, "report", None) is mod:
            delattr(aerolume, "report")
    try:
        return importlib.import_module("aerolume.report")
    except ImportError as exc:
        pytest.skip(f"aerolume.report is not importable: {exc}")


def test_report_page_real_html_report(qapp, app_state):
    rep = _real_report_module()
    if not hasattr(rep, "generate"):
        pytest.skip("aerolume.report.generate not available")
    from aerolume.app.pages.report import ReportPage
    state, jobs = app_state
    page = ReportPage(state, jobs)
    page.html_radio.setChecked(True)
    page.generate()
    deadline = time.monotonic() + 180
    while page._call is not None and time.monotonic() < deadline:
        pump(qapp, 0.05)
    assert page.state_badge.text() == "Ready", page.status.text()
    html = page.last_path.read_text(encoding="utf-8")
    assert "Cruise" in html and page.report_list.rowCount() == 1


# ---------------------------------------------------------------------------
# review round: robustness of the job manager, the explore worker and the pages
# ---------------------------------------------------------------------------

def _slow_until_cancel(started: threading.Event, cd: float = 0.9):
    def run(design, sim, *, progress, frame, cancel, view):
        started.set()
        while not cancel.is_set():
            time.sleep(0.01)
        r = synthetic_cfd(sim.id, cd=cd)
        r.warnings.append("Stopped by the user at t = 1 of 20 L/U: rough indication only.")
        return r
    return run


def test_jobmanager_stopping_a_rerun_keeps_the_previous_complete_result(qapp, app_state):
    """'Run again' + Stop used to overwrite a converged result with the partial one."""
    state, jobs = app_state
    rec = Recorder(jobs)
    logs = []
    state.messageLogged.connect(lambda lvl, src, txt: logs.append(txt))
    before = state.project.load_result("cfd1")
    started = threading.Event()
    jobs.runner = _slow_until_cancel(started)
    jobs.submit("cfd1")
    assert started.wait(5)
    jobs.cancel("cfd1")
    assert jobs.wait(10)
    sim = state.project.simulations["cfd1"]
    assert sim.status == "done" and sim.error is None
    after = state.project.load_result("cfd1")
    assert after.cd.mean == pytest.approx(before.cd.mean) and after.t_end == pytest.approx(before.t_end)
    assert jobs.result("cfd1").cd.mean == pytest.approx(before.cd.mean)     # pages see the kept one
    assert "jobCancelled" in rec.names() and "jobFinished" not in rec.names()
    assert any("previous result was kept" in t for t in logs)
    # a first run that is stopped still keeps its partial result (nothing to protect)
    started.clear()
    jobs.submit("new1")
    assert started.wait(5)
    jobs.cancel("new1")
    assert jobs.wait(10)
    assert state.project.simulations["new1"].status == "cancelled" and state.project.has_result("new1")


def test_jobmanager_window_close_during_a_rerun_keeps_the_previous_result(qapp, app_state):
    state, jobs = app_state
    before = state.project.load_result("cfd2").cd.mean
    started = threading.Event()
    jobs.runner = _slow_until_cancel(started)
    jobs.submit("cfd2")
    assert started.wait(5)
    jobs.shutdown(timeout=5)                        # no event loop in between, as at exit
    assert state.project.simulations["cfd2"].status == "done"
    assert state.project.load_result("cfd2").cd.mean == pytest.approx(before)


def test_jobmanager_a_status_that_cannot_be_saved_never_stalls_the_queue(qapp, app_state, monkeypatch):
    """r10: simulation.json held by OneDrive / read-only while A ran: jobFinished never came,
    B stayed queued forever and wait() never returned."""
    state, jobs = app_state
    rec = Recorder(jobs)
    logs = []
    state.messageLogged.connect(lambda lvl, src, txt: logs.append((lvl, txt)))
    gate = threading.Event()

    def run(design, sim, *, progress, frame, cancel, view):
        gate.wait(5)
        return synthetic_cfd(sim.id)

    jobs.runner = run
    jobs.submit("new1")
    jobs.submit("cfd2")
    real_save = state.project.save

    def refuse() -> None:
        raise PermissionError(13, "Access is denied", "simulation.json")

    monkeypatch.setattr(state.project, "save", refuse)
    gate.set()
    assert jobs.wait(10)                                        # the queue drained
    assert rec.names().count("jobFinished") == 2
    assert state.project.simulations["new1"].status == "done"  # in memory, written with the next save
    assert state.project.simulations["cfd2"].status == "done"
    assert any(lvl == "error" and "could not be saved" in txt for lvl, txt in logs)
    monkeypatch.setattr(state.project, "save", real_save)
    state.project.save()
    assert _project_mod().Project.open(state.project.path).simulations["new1"].status == "done"


def test_jobmanager_a_result_that_cannot_be_saved_is_delivered_and_saved_later(qapp, app_state, monkeypatch):
    """r11: a blocked arrays.npz threw the computed result away and marked the run failed."""
    import aerolume.app.jobs as jobs_mod
    state, jobs = app_state
    monkeypatch.setattr(jobs_mod, "SAVE_RETRY_DELAYS_S", (0.05, 0.05, 0.05))
    rec = Recorder(jobs)
    proj = state.project
    real = proj.save_result
    fails = {"n": 2}

    def flaky(sim_id, result):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise PermissionError(13, "Access is denied", "arrays.npz")
        return real(sim_id, result)

    monkeypatch.setattr(proj, "save_result", flaky)
    jobs.runner = lambda design, sim, **kw: synthetic_cfd(sim.id, cd=0.0777)
    jobs.submit("new1")
    assert jobs.wait(10)
    assert "jobFinished" in rec.names() and "jobFailed" not in rec.names()
    sim = proj.simulations["new1"]
    assert sim.status == "done" and "not saved" in (sim.error or "")
    assert jobs.unsaved_ids() == ["new1"]
    assert jobs.result("new1").cd.mean == pytest.approx(0.0777)        # the pages have it at once
    deadline = time.monotonic() + 5
    while jobs.unsaved_ids() and time.monotonic() < deadline:
        pump(qapp, 0.02)
    assert jobs.unsaved_ids() == [] and sim.error is None
    assert proj.load_result("new1").cd.mean == pytest.approx(0.0777)


def test_jobmanager_shutdown_gives_queued_simulations_their_status_back(qapp, app_state):
    state, jobs = app_state
    started = threading.Event()
    jobs.runner = _slow_until_cancel(started)
    jobs.submit("new1")
    jobs.submit("cfd2")                     # done before: back to done
    assert started.wait(5)
    assert state.project.simulations["cfd2"].status == "queued"
    jobs.shutdown(timeout=5)
    assert jobs.queued_ids() == [] and state.project.simulations["cfd2"].status == "done"


def test_live_worker_a_session_built_after_a_restart_never_takes_the_new_ones_slot(qapp, project):
    """r2: stop() timed out during a slow first build; the old loop then installed its session,
    took the pending slider changes and finally emptied the slot for good."""
    from aerolume.app.jobs import LiveWorker
    sessions: list[FakeSession] = []
    slow = {"first": True}

    def factory(design, sim, **kw):
        if slow["first"]:
            slow["first"] = False
            time.sleep(0.6)                 # a cold CUDA initialisation
        s = FakeSession(design, sim, **kw)
        sessions.append(s)
        return s

    lw = LiveWorker()
    lw.session_factory = factory
    d = next(iter(project.designs.values()))
    sim = project.simulations["cfd1"]
    lw.start(d, sim)
    t0 = time.monotonic()
    lw.stop(timeout=0.05)                   # returns at once: the GUI never waits for a build
    assert time.monotonic() - t0 < 0.4
    lw.start(d, sim)                        # restart while the first is still building
    deadline = time.monotonic() + 5
    while len(sessions) < 2 and time.monotonic() < deadline:
        pump(qapp, 0.02)
    pump(qapp, 0.1)
    assert len(sessions) == 2
    new, old = sessions[0], sessions[1]     # the fast second build finished first
    lw.post(alpha_deg=9.0)
    pump(qapp, 0.05)
    assert lw._session is new and {"alpha_deg": 9.0} in new.posts
    assert old.closed and {"alpha_deg": 9.0} not in old.posts
    lw.shutdown(timeout=5)
    assert new.closed and not lw.is_running()


def test_simulate_stop_acts_on_the_simulation_the_panel_names(qapp, app_state):
    """Stop under 'CFD α 4° — Queued' used to cancel the running polar instead."""
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    started = threading.Event()
    jobs.runner = _slow_until_cancel(started)
    page = SimulatePage(state, jobs)
    jobs.submit("new1")
    assert started.wait(5)
    jobs.jobFrame.emit("new1", synthetic_frame())
    jobs.submit("cfd2")
    state.select_simulation("cfd2")
    pump(qapp)
    assert page.stop_btn.text() == "Remove from queue" and page.stop_btn.isEnabled()
    assert page.other_note.isVisibleTo(page) and "Not run" in page.other_note.text()
    assert "angles" not in page.time_value.text() and "queue" in page.time_value.text()
    page.stop_btn.click()
    assert jobs.running_id() == "new1" and "cfd2" not in jobs.queued_ids()
    assert state.project.simulations["cfd2"].status == "done"
    state.select_simulation("cfd1")         # neither running nor queued: nothing to stop here
    pump(qapp)
    assert not page.stop_btn.isEnabled() and "Running" in page.stop_btn.toolTip()
    state.select_simulation("new1")
    pump(qapp)
    assert page.stop_btn.text() == "Stop" and page.stop_btn.isEnabled()
    assert not page.other_note.isVisibleTo(page)
    page.stop_btn.click()
    assert jobs.wait(10)
    page.close()


def test_simulate_polar_sweep_shows_a_busy_state_until_the_first_angle(qapp, app_state):
    """'0 of 19 angles' over empty plots for the ~9 s of the first angle read as 'it does not run'."""
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    gate = threading.Event()

    def run(design, sim, *, progress, frame, cancel, view):
        progress(0.0, "Polar sweep: 17 angles at Re = 1.00×10^6")
        gate.wait(5)
        progress(0.06, "Polar: alpha = 0 deg: Cl = 0.22, Cd = 0.0056")
        return synthetic_polar(sim.id)

    jobs.runner = run
    state.select_simulation("pol1")
    page = SimulatePage(state, jobs)
    page.handle_action("run")
    pump(qapp, 0.2)
    assert page.progress.maximum() == 0                         # busy, not an empty bar
    assert "solving α = 0°" in page.polar_caption.text() and "first angle" in page.polar_caption.text()
    assert page.status_text.text().startswith("Running · solving")
    gate.set()
    assert jobs.wait(10)
    pump(qapp)
    assert page.progress.maximum() == 1000


def test_compare_baseline_is_never_a_polar_against_cfd_runs(qapp, app_state):
    """A new airfoil study's [polar, CFD] set took the polar as baseline: every Δ was '–', and
    the Fields tab called a different run the baseline than the table."""
    from aerolume.app.pages.compare import ComparePage
    from aerolume.app.pages.results import ResultsPage
    state, jobs = app_state
    state.compare_ids[:] = ["pol1", "cfd1"]
    page = ComparePage(state, jobs)
    assert next(iter(page._results)) == "cfd1" and page.baseline_combo.currentData() == "cfd1"
    page.handle_action("fields")
    assert page._thumb_views and page._thumb_views[0]._title.endswith("(baseline)")
    # "Add to comparison" keeps the set like with like: the polar leaves, the CFD runs stay
    results = ResultsPage(state, jobs)
    state.select_simulation("cfd2")
    results._add_to_compare()
    assert state.compare_ids == ["cfd1", "cfd2"]
    page.refresh()
    rows = {r["key"]: r for r in page._rows}
    assert rows["cd"]["cells"][1]["delta"] is not None          # a real comparison now
    page.close()


def test_compare_page_warns_when_runs_are_not_comparable(qapp, app_state):
    from aerolume.app.pages.compare import ComparePage
    state, jobs = app_state
    state.compare_ids[:] = ["cfd1", "cfd2"]                     # alpha 2 vs 4 deg
    page = ComparePage(state, jobs)
    assert any("different angles" in w for w in page.condition_note.warnings())
    state.project.simulations["cfd2"].conditions.alpha_deg = 2.0
    jobs.result("cfd2").re_sim = 6.4e3                          # a run with a grid-chosen cap
    page.refresh()
    assert any("different Reynolds numbers" in w for w in page.condition_note.warnings())
    page.close()


def test_results_polar_cl_max_still_rising_is_shown_as_a_lower_bound(qapp, app_state):
    from aerolume.app.pages.results import polar_summary_rows
    rows = polar_summary_rows(synthetic_polar("p"))             # lift still rising at the last angle
    value = rows[0][1].text if hasattr(rows[0][1], "text") else str(rows[0][1])
    assert value.startswith("≥") and "still rising" in rows[0][2]


def test_run_summary_message_rounds_like_the_results_table():
    """The Messages line of a finished run uses the Summary table's rule: a lift inside its own error
    bar reads "≈0", the error has one or two significant figures, a settled value says "steady"."""
    from aerolume.app.pages.simulate import mean_error_text
    from aerolume.model import Stat
    assert mean_error_text(Stat(mean=1.653083965, stderr=0.00110994)) == "1.6531 ± 0.0011"
    assert mean_error_text(Stat(mean=-0.0015337, stderr=0.00288978)).startswith("≈0 ± 0.0029")
    assert mean_error_text(Stat(mean=4.562398518, stderr=2e-17)) == "4.562 (steady)"
    assert mean_error_text(Stat(mean=0.25, stderr=float("nan"))) == "0.2500"


def test_run_summary_message_of_a_closed_box_leads_with_what_it_measured():
    """A cavity or a room has no forces (NaN): its Messages line names Nu, the heat rate and the fan
    thrust, with units per metre in 2D, never "Cd —, Cl —"."""
    from aerolume.app.pages.simulate import physics_summary
    from aerolume.model import CFDResult, Stat
    r = CFDResult(sim_id="s", nusselt=Stat(mean=4.562398518, stderr=2e-17), heat_rate=0.10213,
                  fans_report=[{"name": "Fan", "thrust": 1.0}])
    text = physics_summary(r)
    assert text.startswith("Nu 4.562 (steady), Q 0.1021 W/m") and "“Fan” thrust 1.000 N/m" in text
    assert physics_summary(CFDResult(sim_id="s")) == ""


def test_simulate_page_frames_a_closed_box_whole(qapp, app_state, monkeypatch):
    """A wide enclosure (0.30 m x 0.12 m with a circuit board) is shown whole on the Simulate page, as on
    Results: the default near-field framing filled the view with the board and cropped the box to 0.14 m,
    leaving the fan, the intake and the exhaust outside the picture in Explore."""
    from aerolume.app.dialogs.physics_templates import add_template
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    _design, sims = add_template(state, "enclosure", device="cpu", precision="float32")
    state.select_simulation(sims[0].id)
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp, 0.2)
    box = tuple(float(v) for v in sims[0].cfd.domain_box)[:4]
    assert page.view.default_window() == pytest.approx(box)
    x0, y0 = page.view.screen_to_world(page.view.plot_rect().topLeft())
    x1, y1 = page.view.screen_to_world(page.view.plot_rect().bottomRight())
    assert x0 <= box[0] + 1e-9 and x1 >= box[1] - 1e-9 and y1 <= box[2] + 1e-9 and y0 >= box[3] - 1e-9
    page.close()
