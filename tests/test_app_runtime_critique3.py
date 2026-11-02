"""Fixes of critique round 3 on the run pages (Simulate, Results).

* switching from a 10 mm cylinder to a 4 m room re-frames the field view (the room was shown at
  "2 mm" as one flat rectangle, so the run looked empty);
* a closed cavity's Summary leaves out force rows that are blank in every column, prints a steady
  Nusselt number to four significant figures (not "2.250865 ± 4.4×10⁻¹⁶"), names Ra, Pr, Gr, ΔT and
  the wall temperatures, and quotes its published reference;
* a closed domain's live monitors are its fans, temperatures and kinetic energy, not Cd = Cl = 0;
  a closed 3D domain is framed by its whole box.
"""
from __future__ import annotations

import os
import time
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
    Simulation,
    Stat,
)

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


def pump(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def circle(cx, cy, r, n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def _fields(extent, nx=120, ny=90, *, body_r: float | None = None) -> FieldData:
    x0, x1, y0, y1 = extent
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    L = x1 - x0
    u = 1.0 + 0.3 * np.sin(6 * X / L) * np.cos(4 * Y / L)
    v = 0.2 * np.cos(5 * X / L)
    mask = None if body_r is None else (np.hypot(X, Y) <= body_r).astype(np.float32)
    T = 20 + 5 * np.exp(-((X - x0 - 0.3 * L) ** 2 + (Y - y0 - 0.3 * L) ** 2) / (0.02 * L * L))
    return FieldData(extent=extent, fields={"u": u, "v": v, "speed": np.hypot(u, v), "temperature": T},
                     mask=mask, label="mean")


# ---------------------------------------------------------------- framing
def test_field_view_refits_when_a_new_study_has_another_extent(qapp):
    """A cylinder of 10 mm in a 60 mm domain, then a 4 m room without bodies: the scale bar follows the new
    domain (within 10x of it), even after the user zoomed into the cylinder."""
    from aerolume.app.widgets.field_view import FieldView, boxes_overlap, extent_jumped
    assert extent_jumped((0, 0.06, -0.02, 0.02), (0, 4, 0, 3)) and not extent_jumped((0, 4, 0, 3), (0, 4.04, 0, 3))
    assert boxes_overlap((0, 1, 0, 1), (1, 2, 0, 1)) and not boxes_overlap((0, 1, 0, 1), (1.1, 2, 0, 1))
    fv = FieldView()
    fv.resize(1000, 600)
    fv.show()
    pump(qapp)
    cyl = (-0.02, 0.06, -0.025, 0.025)
    fd = _fields(cyl, body_r=0.005)
    fv.set_bodies([circle(0, 0, 0.005)])
    fv.set_field(fd.fields["speed"], cyl, field="speed", mask=fd.mask)
    fv.fit()
    small = fv.scalebar_length()
    assert small is not None and small < 0.02
    fv.zoom_by(3.0, animate=False)                       # the user zoomed into the cylinder
    room = (0.0, 4.0, 0.0, 3.0)
    for _ in range(2):                                   # Results' order: field, then (no) bodies, then fit
        fr = _fields(room)
        fv.set_field(fr.fields["speed"], room, field="speed")
        fv.set_bodies([])
        length = fv.scalebar_length()
        assert length is not None and 4.0 / 10 <= length <= 4.0, length
        assert not fv.is_user_view()
        # and back: the cylinder is framed around its body again
        fv.set_bodies([circle(0, 0, 0.005)])
        fv.set_field(fd.fields["speed"], cyl, field="speed", mask=fd.mask)
        assert fv.scalebar_length() < 0.02
    fv.close()


# ---------------------------------------------------------------- project with a cylinder, a cavity and a room
@pytest.fixture()
def project(tmp_path):
    from aerolume import api
    from aerolume.project import Project
    pr = Project.create(tmp_path / "Closed.aerolume", "Closed")
    # a 10 mm cylinder in a tunnel
    cyl = Design(name="Cylinder 10 mm", kind="generic", bodies=[Body("Cylinder", circle(0, 0, 0.005))],
                 length_ref=0.01, force_ref=0.01, moment_ref=(0.0, 0.0))
    pr.add_design(cyl)
    s_cyl = Simulation(name="Cylinder", design_id=cyl.id, conditions=FlowConditions(speed=0.1), status="done",
                       id="cyl")
    pr.add_simulation(s_cyl)
    # the de Vahl Davis cavity at Ra 1e4 (closed, natural convection), from the API's reference case
    case = api.cavity_case(ra=1e4)
    cav = case.design
    pr.add_design(cav)
    st = case.settings
    s_cav = Simulation(name="Cavity", design_id=cav.id, status="done", id="cav",
                       conditions=FlowConditions(fluid=st["fluid"], speed=0.0173), thermal=st["thermal"])
    s_cav.cfd.domain_mode = "closed"
    s_cav.cfd.domain_box = tuple(st["domain_box"])
    pr.add_simulation(s_cav)
    # a 4 m room without bodies
    room = Design(name="Room", kind="generic", bodies=[], length_ref=0.5, force_ref=0.5, moment_ref=(2.0, 1.5))
    pr.add_design(room)
    s_room = Simulation(name="Room", design_id=room.id, conditions=FlowConditions(speed=0.9), status="done", id="room")
    s_room.cfd.domain_mode = "closed"
    s_room.cfd.domain_box = (0.0, 4.0, 0.0, 3.0)
    pr.add_simulation(s_room)
    pr.save()
    t = np.linspace(0, 40, 400)
    pr.save_result("cyl", CFDResult(
        sim_id="cyl", cd=Stat(1.52, 0.01, 1.50, 1.54, 0.002), cl=Stat(0.0, 0.01, -0.01, 0.01, 0.001),
        cm=Stat(0.0, 0.0, 0.0, 0.0, 1e-6), drag=1e-5, lift=0.0, moment=0.0, drag_area=0.015, power=1e-6,
        re_real=66.0, re_sim=66.0, grid=(300, 200), cells_per_ref=30, t_end=40, t_avg_start=10, converged=True,
        history={"t": t, "cd": 1.52 + 0 * t, "cl": 0 * t, "cm": 0 * t},
        mean_fields=_fields((-0.02, 0.06, -0.025, 0.025), body_r=0.005),
        bodies_wind=[Body("Cylinder", circle(0, 0, 0.005))]))
    nan = float("nan")
    pr.save_result("cav", CFDResult(
        sim_id="cav", re_real=119.0, re_sim=119.0, grid=(64, 64), cells_per_ref=64, t_end=42.1, t_avg_start=32.0,
        converged=True, nusselt=Stat(2.250865, 4.4e-16, 2.250865, 2.250865, 4.4e-16), heat_rate=0.005039,
        history={"t": t, "nu": 2.250865 + 0 * t, "heat_rate": 0.005039 + 0 * t},
        mean_fields=_fields((0.0, 0.1, 0.0, 0.1)), cd=Stat(nan, nan, nan, nan, nan)))
    pr.save_result("room", CFDResult(
        sim_id="room", re_real=3.09e4, re_sim=3.09e4, grid=(96, 72), cells_per_ref=12, t_end=240, t_avg_start=120,
        converged=False, history={"t": t, "fan0_velocity": 0.9 + 0.01 * np.sin(t), "fan0_thrust": 1.0 + 0 * t},
        mean_fields=_fields((0.0, 4.0, 0.0, 3.0))))
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


def test_results_fields_reframe_from_a_cylinder_to_a_room(qapp, app_state):
    from aerolume.app.pages.results import ResultsPage
    state, jobs = app_state
    page = ResultsPage(state, jobs)
    page.resize(1280, 800)
    page.show()
    try:
        # the cylinder is framed by its near field (its 0.01 m size), the room and the cavity by their domains
        for sid, size in (("cyl", 0.01), ("room", 4.0), ("cav", 0.1), ("cyl", 0.01)):
            state.select_simulation(sid)
            pump(qapp, 0.2)
            page.handle_action("fields")
            pump(qapp, 0.2)
            length = page.field_view.scalebar_length()
            assert length is not None and size / 10 <= length <= size, (sid, length)
    finally:
        page.close()


def test_closed_cavity_summary_conditions_and_reference(project):
    from aerolume.app.pages import results as R
    from aerolume.app.widgets.kpi import STEADY_TEXT
    sim, r = project.simulations["cav"], project.load_result("cav")
    design = project.designs[sim.design_id]
    rows = R.summary_rows(r, design, sim)
    names = [R._cell_text(row[0]) for row in rows]
    assert not any(n.startswith(("Drag", "Lift", "Moment", "Pitching", "Strouhal")) for n in names), names
    assert not any(n.startswith("Reynolds") for n in names)               # natural convection: Ra, not Re
    nu = rows[names.index("Mean Nusselt number Nu")]
    assert R._cell_text(nu[1]) == "2.251" and R._cell_text(nu[2]) == STEADY_TEXT and nu[2].tone is None
    ref = next(row for row in rows if R._cell_text(row[0]).startswith("Reference Nu"))
    assert "de Vahl Davis" in R._cell_text(ref[0]) and R._cell_text(ref[1]) == "2.238"
    assert R._cell_text(ref[6]) == f"measured +{100 * (2.250865 / 2.238 - 1):.1f} %"     # +0.6 %
    cond = dict(R.conditions_items(r, design, sim))
    for key in ("Rayleigh number Ra", "Prandtl number Pr", "Grashof number Gr", "Temperature difference ΔT",
                "Left wall", "Right wall"):
        assert key in cond, (key, cond)
    assert cond["Rayleigh number Ra"] == "1.0×10⁴" and cond["Prandtl number Pr"] == "0.710"
    assert "Angle of attack" not in cond and "Dynamic pressure" not in cond and "Speed" not in cond
    _status, _kind, facts = R.header_facts(r, sim, design)
    assert facts.startswith("Ra 1.0×10⁴")
    assert "Coefficients" not in R.summary_note(design, sim, r)
    # a tunnel run keeps its forces and Re
    cyl = project.simulations["cyl"]
    names_cyl = [R._cell_text(row[0]) for row in R.summary_rows(project.load_result("cyl"), project.designs[
        cyl.design_id], cyl)]
    assert "Drag coefficient Cd" in names_cyl and "Reynolds number (real)" in names_cyl
    assert "Strouhal number St" not in names_cyl                           # no shedding: a blank row, left out


def test_steady_stat_texts():
    from aerolume.app.widgets.kpi import STEADY_TEXT, is_steady, stat_texts
    assert is_steady(2.25, 4.4e-16) and not is_steady(2.25, 1e-3) and not is_steady(0.0, 0.0)
    assert not is_steady(float("nan"), 0.0)
    assert stat_texts(2.250865, 4.4e-16, 4.4e-16, 2.250865, 2.250865) == ("2.251", STEADY_TEXT, "—", "2.251",
                                                                          "2.251")
    assert stat_texts(1.4, 0.01, 0.02, 1.38, 1.42)[1] == "0.010"


# ---------------------------------------------------------------- simulate monitors
def test_closed_monitor_helpers():
    from aerolume.app.pages.simulate import closed_box3, closed_history_keys, closed_monitor_items
    phys = {"fan0_velocity": 1.234, "fan1_velocity": 0.5, "fan0_thrust": 1.0, "t_max": 31.5, "t_min": 19.9,
            "ke": 0.12, "nu": float("nan")}
    items = closed_monitor_items(phys)
    labels = [lab for _k, lab, _t in items]
    assert labels == ["Fan 1 disc velocity", "Fan 2 disc velocity", "Max temperature", "Min temperature",
                      "Kinetic energy"]
    assert items[0][2] == "1.23 m/s" and items[2][2].endswith("°C")
    keys, hl = closed_history_keys([k for k, _l, _t in items])
    assert keys == ["fan0_velocity", "t_max", "ke"] and "fan" in hl["fan0_velocity"]
    room = Simulation(name="Room", design_id="d")
    room.cfd.domain_mode = "closed"
    room.cfd.domain_box = (0, 4, 0, 3, 0, 2.5)
    assert closed_box3(room) == (0, 4, 0, 3, 0, 2.5)
    assert closed_box3(room, (0, 4.1, 0, 3, 0, 2.5)) == (0, 4.1, 0, 3, 0, 2.5)
    tunnel = Simulation(name="Car", design_id="d")
    assert closed_box3(tunnel, (0, 1, 0, 1, 0, 1)) is None


def test_simulate_monitors_a_closed_domain_by_its_fans_and_temperatures(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    page = SimulatePage(state, jobs)
    page.resize(1400, 900)
    page.show()
    try:
        state.select_simulation("room")
        pump(qapp, 0.3)
        assert page._closed_mon and not page._row_cd.label.isVisible()
        assert page.closed_values.isVisible()
        # the stored run: the fan series it kept
        assert "fan0_velocity" in page.history.keys
        rgba = np.zeros((30, 40, 4), np.uint8)
        for k in range(5):
            frame = SimpleNamespace(rgba=rgba, extent=(0.0, 4.0, 0.0, 3.0), field="speed", cmap="viridis", vmin=0.0,
                                    vmax=1.0, units="m/s", t=1.0 + k, step=100 * k, steps_per_s=500.0, coeffs={},
                                    history={}, bodies=[], physics={"fan0_velocity": 0.9 + 0.01 * k,
                                                                    "t_max": 30.0 - k, "t_min": 20.0, "ke": 0.1})
            page._show_frame(frame)
        texts = dict(page.closed_values.items())
        assert texts["Fan 1 disc velocity"] == "0.940 m/s" and "Max temperature" in texts
        assert page.history.keys == ("fan0_velocity", "t_max", "ke")
        assert page.history.series_names() == ["fan0_velocity", "t_max", "ke"]
        # a tunnel run is back on its forces
        state.select_simulation("cyl")
        pump(qapp, 0.3)
        assert not page._closed_mon and not page.closed_values.isVisible()
        assert page.history.keys[:2] == ("cd", "cl")
    finally:
        page.close()
