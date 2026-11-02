"""Results, Compare and the trace / 3D strips for closed domains and the shared rounding rule (critique 3.2).

* a closed room's Summary leads with temperatures, heater powers ("Heater power (set)") and heat paths,
  never with the coefficients of the furniture inside it, and never with one signed mean Nusselt number
  over a heated and a cooled surface;
* every Mean / ± cell is rounded by one rule (the Particle Data Group's), a value inside its own error bar
  reads "≈0 (within error)", and the steady sentence is printed only when a row is steady;
* Compare pre-selects runs of one design only, and colours only differences beyond two combined errors;
* the Results field menu offers the smoke of the last snapshot, opens on it when the run released smoke,
  frames a room's whole box, and names the RMS fields.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from aerolume.model import (
    Body,
    CFDResult,
    Design,
    Emitter,
    FieldData,
    FlowConditions,
    HeatSource,
    Port,
    Simulation,
    Stat,
    ThermalSettings,
    WallThermal,
)

pytestmark = pytest.mark.gui
NAN = float("nan")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


def room_fields(label="mean", smoke=False) -> FieldData:
    """A 4 m x 3 m room: 20 °C, a 30 °C patch over the heater at (1, 0.5) (0.4 m x 0.2 m)."""
    nx, ny = 80, 60
    x = (np.arange(nx) + 0.5) * 4.0 / nx
    y = (np.arange(ny) + 0.5) * 3.0 / ny
    X, Y = np.meshgrid(x, y)
    T = 20.0 + 10.0 * ((np.abs(X - 1.0) <= 0.2) & (np.abs(Y - 0.5) <= 0.1))
    f = {"u": 0.1 + 0 * X, "v": 0 * X, "speed": 0.1 + 0 * X, "temperature": T, "u_rms": 0.01 + 0 * X}
    if smoke:
        f["dye"] = np.clip(1 - np.hypot(X - 1, Y - 2), 0, 1)
        f["dye1"] = np.clip(1 - np.hypot(X - 3, Y - 1), 0, 1)
    if label != "mean":
        f.pop("u_rms")
    return FieldData(extent=(0.0, 4.0, 0.0, 3.0), fields=f, mask=np.zeros((ny, nx), np.float32), label=label)


def room_case(*, body_heat=None):
    d = Design(name="Room", kind="generic", bodies=[Body("Cabinet", np.array([[2.0, 0.0], [2.4, 0.0], [2.4, 1.0],
                                                                          [2.0, 1.0]]))],
               length_ref=0.5, force_ref=0.5, moment_ref=(0.0, 0.0))
    sim = Simulation(name="Room", design_id=d.id, conditions=FlowConditions(speed=1.0), status="done",
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0,
                                             walls=[WallThermal(side="left", mode="temperature", value=12.0)]),
                     heat_sources=[HeatSource(name="Radiator", kind="rect", position=(1.0, 0.5), size=(0.4, 0.2),
                                              power=200.0)],
                     ports=[Port(name="Vent", kind="vent", position=(3.9, 2.5), direction=(1.0, 0.0))],
                     emitters=[Emitter(name="Smoke", position=(1.0, 2.0)),
                               Emitter(name="Smoke 2", position=(3.0, 1.0), color="#e69f00")])
    sim.cfd.domain_mode = "closed"
    sim.cfd.domain_box = (0.0, 4.0, 0.0, 3.0)
    pb = {"Cabinet": {"cd": -0.48, "cl": 0.18, "cm": 0.05}}
    if body_heat is not None:
        pb = {"Heater": {"cd": 0.1, "cl": 0.0, "cm": 0.0, "heat_rate": body_heat[0], "nu": 12.0},
              "Window": {"cd": 0.0, "cl": 0.0, "cm": 0.0, "heat_rate": body_heat[1], "nu": -8.0}}
    r = CFDResult(sim_id=sim.id, cd=Stat(-0.48, 0.3, -1.1, 0.0, 0.2), cl=Stat(0.18, 0.1, -0.1, 0.4, 0.04),
                  cm=Stat(0.05, 0.1, -0.1, 0.2, 0.07), drag=-0.1, lift=0.04, moment=0.0, drag_area=-0.2, power=-0.1,
                  re_real=3e4, re_sim=3e4, grid=(80, 60), cells_per_ref=10, t_end=100, t_avg_start=50, converged=False,
                  wall_seconds=10, device="cuda", precision="float32", turbulence="les-wale",
                  nusselt=Stat(-53.6, 2.0, -57.0, -50.0, 1.7), heat_rate=-98.5, per_body=pb,
                  diagnostics={"heat_in_bodies": 0.0, "heat_in_walls": -92.9, "heat_in_sources": 200.0,
                               "heat_in_ports": -7.5, "heat_out_ports": 30.0, "heat_storage_rate": 85.0,
                               "heat_balance_residual": 0.1, "t_fluid_mean": 21.0, "t_fluid_max": 30.0},
                  history={"t": np.linspace(0, 100, 50), "cd": np.zeros(50), "cl": np.zeros(50), "cm": np.zeros(50)},
                  surface=[], mean_fields=room_fields("mean"), instant_fields=room_fields("instant", smoke=True))
    return d, sim, r


def texts(rows):
    from aerolume.app.pages.results import _cell_text
    return [[_cell_text(c) for c in row] for row in rows]


# ---------------------------------------------------------------- rounding
def test_errors_are_rounded_by_the_particle_data_group_rule():
    from aerolume.app.widgets.kpi import (
        WITHIN_ERROR_TEXT,
        fmt_error,
        round_error,
        stat_texts,
        value_error_texts,
    )
    assert fmt_error(0.00095) == "0.0010" and fmt_error(0.00076) == "0.0008" and fmt_error(0.000019) == "0.000019"
    assert fmt_error(0.354) == "0.35" and fmt_error(0.355) == "0.4" and fmt_error(31.0) == "31"
    assert round_error(0.0) is None and round_error(float("nan")) is None
    # Nu 3.34079 ± 0.00095 and Q 2.69880 ± 0.00076 (critique 3.2) under the rule
    assert value_error_texts(3.3407889, 0.00095) == ("3.3408", "0.0010")
    assert value_error_texts(2.6988, 0.00076) == ("2.6988", "0.0008")
    # a mean inside its own error bar is not given digits it does not have
    assert stat_texts(-1.237e-12, 1.7e-11, 5.6e-11, -1e-10, 9e-11)[0] == WITHIN_ERROR_TEXT
    assert stat_texts(-0.0016, 0.0028, 0.009, -0.0157, 0.0141)[:2] == (WITHIN_ERROR_TEXT, "0.0028")
    # a steady cavity Nusselt number (error 4×10⁻⁶ of the mean): four figures, "steady"
    assert stat_texts(4.5624, 1.93e-5, 3.7e-5, 4.56225, 4.562435)[:2] == ("4.562", "steady")
    # a spread that would print as "0.8" stays a decimal, not 7.7×10⁻¹
    assert stat_texts(0.58, 0.45, 0.77, -0.9, 2.7)[2] == "0.8"


def test_steady_sentence_and_colour_legend_only_when_a_row_has_them():
    from aerolume.app.pages import results as R
    from aerolume.app.widgets.kpi import STEADY_TEXT
    steady_row = [R.Cell("Nu"), "2.251", R.Cell(STEADY_TEXT), "—", "—", "—", ""]
    rough_row = R.stat_cells("Cd", Stat(1.2, 0.05, 1.1, 1.3, 0.03, stderr_reliable=False))
    plain_row = R.stat_cells("Cd", Stat(1.2, 0.05, 1.1, 1.3, 0.03))
    assert "steady" in R.rounding_note([steady_row]) and "steady" not in R.rounding_note([plain_row])
    assert rough_row[2].tone == "warn" and rough_row[2].tooltip == R.ROUGH_TIP
    assert "warning colour" in R.rounding_note([rough_row]) and "warning colour" not in R.rounding_note([plain_row])


# ---------------------------------------------------------------- closed rooms
def test_closed_room_summary_leads_with_temperatures_and_heat_paths():
    from aerolume.app.pages import results as R
    d, sim, r = room_case()
    rows = texts(R.summary_rows(r, d, sim))
    names = [row[0] for row in rows]
    assert not any(n.startswith(("Drag", "Lift", "Moment", "Pitching", "Strouhal", "Mean Nusselt", "Heat transfer"))
                   for n in names), names
    assert names[:2] == ["Maximum fluid temperature", "Mean fluid temperature"]
    t = {row[0]: row for row in rows}
    assert t["Maximum fluid temperature"][1] == "30.0" and t["Maximum fluid temperature"][6] == "°C"
    heater = t["Heater power (set): Radiator"]
    assert heater[1] == "200" and heater[2] == R.SET_TEXT and heater[6] == "W/m"
    assert t["Peak temperature at Radiator"][1] == "30.0"          # read in the heater's footprint
    assert t["Heat lost through the walls"][1] == "92.90"
    assert t["Heat removed by the vents (net)"][1] == "37.50"         # 30 out + 7.5 of cooler jet air in
    assert "Still being stored (warming)" in t
    note = R.summary_note(d, sim, r)
    assert "Coefficients" not in note and "steady" not in note and "0 (set)" in note
    # the coefficients stay in Forces; the closed conditions name no free stream
    cond = {item[0]: item[1] for item in R.conditions_items(r, d, sim)}
    assert "Dynamic pressure" not in cond and "Angle of attack" not in cond


def test_heated_and_cooled_bodies_are_never_averaged_into_one_nusselt_number():
    from aerolume.app.pages import results as R
    d, sim, r = room_case(body_heat=(150.0, -60.0))
    assert not R.mean_nu_meaningful(r, sim)
    names = [row[0] for row in texts(R.summary_rows(r, d, sim))]
    assert "Heat rate: Heater (heated)" in names and "Heat rate: Window (cooled)" in names
    assert "Nusselt number: Heater" in names and "Mean Nusselt number Nu" not in names
    # the same rule in a tunnel: two bodies of opposite sign, no mean Nu and no summed Q
    sim.cfd.domain_mode = "tunnel"
    r.nusselt = Stat(2.0, 0.1, 1.9, 2.1, 0.01)
    assert [row[0] for row in texts(R.thermal_summary_rows(r, "", sim))] == []


def test_a_fan_in_a_closed_room_has_no_momentum_theory_number(qapp):
    """Froude's actuator disc needs an unbounded stream. In a room the planner's speed is a reference scale,
    and the column read 0.56 m/s of "theory" beside 0.22 m/s measured; it is now empty, and says why. In a
    tunnel the column stays the theory at the run's speed."""
    import math

    from aerolume.app.pages import results as R
    from aerolume.model import Fan
    _d, sim, r = room_case()
    sim.fans = [Fan(name="Fan", position=(1.3, 1.5), direction=(1.0, 0.0), diameter=0.5, mode="thrust", value=1.0)]
    sim.conditions.speed = 0.9035
    r.fans_report = [{"name": "Fan", "mode": "thrust", "thrust": 1.0, "pressure_jump": 2.0, "volume_flow": 0.1123,
                      "induced_velocity": 0.2247, "ideal_power": 0.2247, "target": 1.0, "diameter": 0.5}]
    row = R.fan_rows(r, sim)[0]
    assert R._cell_text(row[5]) == "0.2247"                                  # measured stays
    assert R._cell_text(row[6]) == "—" and row[6].tooltip == R.CLOSED_FAN_TIP
    assert "not shown in a closed domain" in R.fan_note_text(sim)
    sim.cfd.domain_mode = "tunnel"
    U, rho, A = 0.9035, sim.conditions.fluid.density, 0.5
    w = 0.5 * (-U + math.sqrt(U * U + 2.0 / (rho * A)))
    assert float(R._cell_text(R.fan_rows(r, sim)[0][6])) == pytest.approx(w, rel=1e-3)
    assert "Froude actuator disc of the fan's width" in R.fan_note_text(sim)


def test_one_heated_body_reads_the_summarys_nusselt_number():
    """With one heat-exchanging body the per-body Nu *is* the mean Nu: the Heat transfer tab read
    "Mean Nu 183 ± 12" above "Box 181.8 —" (the per-body value is built from time-averaged temperatures).
    Both rows now show the Summary's statistic, with its rough error in the warning colour. Two bodies keep
    their own values."""
    from aerolume.app.pages import results as R
    d, sim, r = room_case()
    sim.cfd.domain_mode = "tunnel"
    d.bodies[0].name = "Box"
    d.bodies[0].thermal, d.bodies[0].thermal_value = "heat_flux", 200.0
    r.per_body = {"Box": {"cd": 1.2, "cl": 0.0, "cm": 0.0, "nu": 181.76, "heat_rate": 156.0, "t_wall": 41.4}}
    r.nusselt = Stat(182.86, 13.9, 165.6, 212.3, 12.04, stderr_reliable=False)
    summ = next(row for row in R.thermal_summary_rows(r, "", sim, d) if R._cell_text(row[0]).startswith("Mean Nu"))
    body = R.heat_body_rows(r, d)[0]
    assert R._cell_text(body[2]) == R._cell_text(summ[1]) == "183"
    assert R._cell_text(body[3]) == "12" and body[3].tone == "warn"
    d.bodies.append(Body("Fin", np.array([[3.0, 0.0], [3.1, 0.0], [3.1, 0.5], [3.0, 0.5]]), thermal="temperature",
                         thermal_value=40.0))
    r.per_body["Fin"] = {"cd": 0.1, "cl": 0.0, "cm": 0.0, "nu": 9.87, "heat_rate": 12.0}
    rows = {row[0]: row for row in R.heat_body_rows(r, d)}
    assert R._cell_text(rows["Box"][2]) == "181.8" and R._cell_text(rows["Fin"][2]) == "9.870"


def test_room_fields_are_framed_on_the_box_and_open_on_the_smoke(qapp):
    from aerolume.app.pages import results as R
    d, sim, r = room_case()
    assert R.closed_rect(sim) == (0.0, 4.0, 0.0, 3.0)
    focus = R.heat_focus_rect(r, d, sim)
    assert focus[0] < 1.0 - 0.2 < 1.0 + 0.2 < focus[1] and focus[2] <= 0.5 - 0.1 and focus[1] - focus[0] < 4.0
    # smoke is kept as the last snapshot: the averaged set offers it from there; the run opens on it
    menu = R.field_menu(r.mean_fields, r)
    assert menu[0] == (R.SMOKE_COMPOSITE, "Smoke (all channels, last snapshot)")
    assert ("u_rms", "u RMS fluctuation") in menu
    assert R.default_field(r, sim) == R.SMOKE_COMPOSITE
    inst = dict(R.field_menu(r.instant_fields, r))
    assert inst["dye"] == "Smoke, channel 1" and inst["dye1"] == "Smoke, channel 2"
    sim.emitters = []
    assert R.default_field(r, sim) == "speed"


def test_results_page_frames_a_room_and_draws_its_devices(qapp, tmp_path):
    from aerolume.app.jobs import JobManager
    from aerolume.app.pages.results import ResultsPage
    from aerolume.app.state import AppState
    from aerolume.project import Project
    d, sim, r = room_case()
    pr = Project.create(tmp_path / "Room.aerolume", "Room")
    pr.add_design(d)
    pr.add_simulation(sim)
    pr.save()
    pr.save_result(sim.id, r)
    state = AppState()
    state.set_project(pr)
    jobs = JobManager(state)
    page = ResultsPage(state, jobs)
    try:
        page.resize(1400, 800)
        page.show()
        state.select_simulation(sim.id)
        page.refresh()
        qapp.processEvents()
        page.tabs.setCurrentIndex(page.TABS.index("fields"))
        for _ in range(20):
            qapp.processEvents()
        assert page.fld_combo.currentData() == "smoke_all" and page.fld_slot.current() == "instant"
        assert page.field_view.placements()                        # heater, vent and emitters drawn
        s, cx, cy = page.field_view.view_state()
        assert (cx, cy) == pytest.approx((2.0, 1.5)) and 4.0 * s <= page.field_view.plot_rect().width() + 1
        page.fld_devices.setChecked(False)
        assert not page.field_view.placements()
        page.fld_slot.set_current("mean")                          # the averages, not a jump back to smoke
        qapp.processEvents()
        assert page.fld_slot.current() == "mean" and page.fld_combo.currentData() == "speed"
    finally:
        page.close()
        jobs.shutdown(timeout=5)


# ---------------------------------------------------------------- compare
def test_compare_preselects_one_design_and_blanks_meaningless_percentages():
    from aerolume.app.pages.compare import comparable_designs, default_selection, delta_state
    kind = {"a": "cfd3d", "b": "cfd3d", "c": "cfd3d", "d": "cfd3d"}
    design = {"a": "car", "b": "sphere", "c": "box", "d": "car"}
    assert default_selection(list(kind), kind, design, "b") == ["a", "d"]       # no other sphere run: the car pair
    assert default_selection(["a", "b", "c"], kind, design, "a",
                             comparable=lambda x, y: design[x] == design[y]) == []   # nothing related: none ticked
    # no design with two runs, but variants of one object (two airfoils): those
    assert default_selection(["a", "b", "c"], kind, design, "a", comparable=lambda x, y: {x, y} <= {"a", "c"})         == ["a", "c"]
    car = Design(name="Car", kind="vehicle", bodies=[], length_ref=4.0, force_ref=1.3)
    car2 = Design(name="Car B", kind="vehicle", bodies=[], length_ref=4.2, force_ref=1.3)
    sphere = Design(name="Sphere", kind="generic", bodies=[], length_ref=0.1, force_ref=0.1)
    assert comparable_designs(car, car2) and not comparable_designs(car, sphere)
    row = {"key": "cd"}
    base = {"value": 0.30, "stderr": 0.01, "is_baseline": True}
    assert delta_state(row, {"delta": 0.02, "stderr": 0.01}, base)[0] == "noise"          # 2 % < 2 sigma (0.028)
    assert delta_state(row, {"delta": 0.05, "stderr": 0.01}, base)[0] == "significant"
    assert delta_state(row, {"delta": 0.05, "stderr": 0.01}, base, comparable=False)[0] == "none"
    zero = {"value": 1e-12, "stderr": 1e-11, "is_baseline": True}                          # zero within its error
    assert delta_state({"key": "cs"}, {"delta": 0.01, "stderr": 0.001}, zero)[0] == "none"
    assert delta_state(row, {"delta": 0.05}, {"value": 0.3, "is_baseline": True})[0] == "pct"   # no errors (polar)


# ---------------------------------------------------------------- 3D strip
def test_flow_toggles_overflow_into_a_menu_when_the_row_is_too_narrow(qapp):
    from aerolume.app.widgets.common import ToolStrip
    from aerolume.app.widgets.view3d_panel import FlowToggles
    strip = ToolStrip()
    tg = FlowToggles(("smoke", "streamlines", "slice", "heat", "surface"))
    tg.add_to(strip)
    strip.resize(1000, 30)
    strip.show()
    tg.overflow_to(1000)
    assert tg.overflowed() == [] and tg.more_btn.isHidden()
    tg.overflow_to(260)
    over = tg.overflowed()
    assert over and over[-1] == "surface" and not tg.more_btn.isHidden()
    assert all(tg.buttons[k].isHidden() for k in over)
    tg.set_shown_keys(["smoke", "streamlines", "slice", "surface"])            # no heat: never in the menu either
    assert "heat" not in tg.overflowed() and tg.buttons["heat"].isHidden()
    tg._fill_menu()
    acts = tg.more_btn.menu().actions()
    assert [a.text() for a in acts] == [tg.buttons[k].text() for k in tg.overflowed()]
    seen = []
    tg.changed.connect(lambda k, on: seen.append((k, on)))
    acts[-1].setChecked(not acts[-1].isChecked())
    assert seen and seen[-1][0] == tg.overflowed()[-1]                         # the menu entry is the layer's toggle
    strip.close()
