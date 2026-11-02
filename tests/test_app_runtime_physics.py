"""Results, Compare and Simulate with placed physics (DESIGN.md 10): heat transfer, fans, smoke channels,
solver diagnostics. Synthetic results go through the real project store, as in test_app_runtime.py.
"""
from __future__ import annotations

import math
import os
import time

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
    Fan,
    FieldData,
    FlowConditions,
    Simulation,
    Stat,
    SurfaceData,
    ThermalSettings,
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


def thermal_fields(extent=(-0.2, 0.6, -0.2, 0.2), nx=160, ny=80) -> FieldData:
    x0, x1, y0, y1 = extent
    X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    r = np.hypot(X, Y)
    mask = (r <= 0.05).astype(np.float32)
    u = 0.5 * (1 - np.exp(-(Y / 0.06) ** 2) * (X > 0))
    v = 0.05 * np.sin(8 * X) * np.exp(-(Y / 0.1) ** 2)
    T = 20 + 40 * np.exp(-((X - 0.05) ** 2 / 0.02 + Y ** 2 / 0.002)) * (X > -0.05)
    dye = np.clip(np.exp(-(Y - 0.02) ** 2 / 2e-4) * (X > -0.15), 0, 1)
    dye1 = np.clip(np.exp(-(Y + 0.03) ** 2 / 2e-4) * (X > -0.15), 0, 1)
    return FieldData(extent=extent, fields={"u": u, "v": v, "speed": np.hypot(u, v), "temperature": T,
                                            "dye": dye, "dye1": dye1, "cp": 1 - (np.hypot(u, v) / 0.5) ** 2},
                     mask=mask, label="mean")


def thermal_result(sim_id: str, nu=7.31, q=48.0) -> CFDResult:
    t = np.linspace(0, 30, 1500)
    hist = {"t": t, "cd": 1.4 + 0.01 * np.sin(3 * t), "cl": 0.02 * np.sin(1.5 * t), "cm": 0 * t,
            "nu": nu + 0.05 * np.sin(2 * t), "heat_rate": q + 0.3 * np.sin(2 * t),
            "div_rms": 1e-7 * (1 + 0 * t), "du_rel": 1e-3 / (1 + t), "cfl": 0.45 + 0 * t}
    pts = circle(0, 0, 0.05, 200)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    n = pts / 0.05
    surf = SurfaceData("Cylinder", s, pts[:, 0], pts[:, 1], n[:, 0], n[:, 1], 1 - 2 * np.abs(n[:, 0]), 0.01 + 0 * s)
    surf.heat_flux = 900 + 400 * n[:, 0]          # what a solver may attach: q'' along the wall
    return CFDResult(
        sim_id=sim_id, cd=Stat(1.4, 0.01, 1.38, 1.42, 0.002), cl=Stat(0.0, 0.02, -0.02, 0.02, 0.001),
        cm=Stat(0.0, 0.0, 0.0, 0.0, 1e-6), drag=0.1, lift=0.0, moment=0.0, drag_area=0.14, power=0.05,
        re_real=40.0, re_sim=40.0, grid=(400, 200), cells_per_ref=40, t_end=30, t_avg_start=10, converged=True,
        wall_seconds=12, device="cuda", precision="float32", turbulence="laminar",
        nusselt=Stat(nu, 0.05, nu - 0.07, nu + 0.07, 0.004), heat_rate=q,
        per_body={"Cylinder": {"cd": 1.4, "cl": 0.0, "cm": 0.0, "nu": nu, "heat_rate": q, "nu_stderr": 0.004}},
        fans_report=[{"name": "Fan", "mode": "thrust", "target": 2.0, "thrust": 1.97, "thrust_stderr": 0.01,
                      "volume_flow": 0.03, "induced_velocity": 0.6, "ideal_power": 1.4, "pressure_jump": 9.8}],
        diagnostics={"divergence_rms": 2e-7, "pressure_residual": 3e-15, "cfl_max": 0.46,
                     "heat_in_bodies": q, "heat_in_sources": 2.0, "heat_out_outlet": q + 1.99,
                     "heat_storage_rate": 0.0},
        history=hist, surface=[surf], mean_fields=thermal_fields(), bodies_wind=[Body("Cylinder", pts)])


@pytest.fixture()
def project(tmp_path):
    from aerolume.project import Project
    pr = Project.create(tmp_path / "Heat.aerolume", "Heat")
    d = Design(name="Cylinder", kind="generic", bodies=[Body("Cylinder", circle(0, 0, 0.05), thermal="temperature",
                                                             thermal_value=60.0)],
               length_ref=0.1, force_ref=0.1, moment_ref=(0.0, 0.0))
    pr.add_design(d)
    sim = Simulation(name="Heated cylinder", design_id=d.id, conditions=FlowConditions(speed=0.5), status="done",
                     id="th1", thermal=ThermalSettings(enabled=True, t_ambient=20.0),
                     emitters=[Emitter(name="Upper", position=(-0.15, 0.02)),
                               Emitter(name="Lower", position=(-0.15, -0.03), color="#e69f00")],
                     fans=[Fan(name="Fan", position=(0.3, 0.1), diameter=0.05, value=2.0)])
    pr.add_simulation(sim)
    sim2 = Simulation(name="Heated cylinder B", design_id=d.id, conditions=FlowConditions(speed=0.5), status="done",
                      id="th2", thermal=ThermalSettings(enabled=True, t_ambient=20.0))
    pr.add_simulation(sim2)
    pr.save()
    pr.save_result("th1", thermal_result("th1"))
    pr.save_result("th2", thermal_result("th2", nu=6.90, q=45.1))
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


def test_physics_helpers_read_through_analysis_physics(project):
    from aerolume.app.pages import results as R
    r = project.load_result("th1")
    assert R.has_heat(r) and R.has_fan_report(r)
    rows = R.thermal_summary_rows(r)
    # rounded to its error by the Particle Data Group rule: ± 0.004 has one figure, so 7.310
    assert rows[0][0].text == "Mean Nusselt number Nu" and rows[0][1].text == "7.310"
    assert rows[1][0].text == "Heat transfer rate Q" and rows[1][-1] == "W/m"
    body = R.heat_body_rows(r, next(iter(project.designs.values())))
    assert body[0][:3] == ["Cylinder", "60 °C", "7.310"]
    brows, note, tone = R.heat_balance_rows(r)
    assert any(str(c[0]).startswith("In: Bodies") for c in brows) and tone is None     # 48 + 2 - 49.99 = 0.01
    assert "computed here" in note
    fans = R.fan_rows(r, project.simulations["th1"])
    assert fans[0][0] == "Fan" and R._cell_text(fans[0][3]).startswith("1.970 ± 0.010")
    # momentum theory at U = 0.5 m/s, rho = 1.225, width 0.05 m: w = (-U + sqrt(U^2 + 2 T/(rho A))) / 2
    w = 0.5 * (-0.5 + math.sqrt(0.25 + 2 * 1.97 / (1.225 * 0.05)))
    assert float(fans[0][6]) == pytest.approx(w, rel=1e-3)
    assert R.temperature_field(r.mean_fields)[0] == "temperature"
    sh = R.surface_heat(r.surface[0])
    assert sh is None or sh[1].startswith("Wall heat flux")       # the store may drop unknown surface arrays
    log = R.history_diagnostics(r)
    assert log is not None and set(log.keys()) >= {"div_rms", "du_rel", "cfl"}


def test_heat_curves_split_a_body_into_branches_like_the_surface_tab():
    """A wall distribution on a closed body reads as an upper and a lower branch against x / L from the
    leading edge (in outline order against x it was one loop folded back over itself)."""
    from aerolume.app.pages.results import heat_curves
    pts = circle(0, 0, 0.05, 200)
    n = pts / 0.05
    s = np.zeros(len(pts))
    sd = SurfaceData("Cylinder", s, pts[:, 0], pts[:, 1], n[:, 0], n[:, 1], 0 * s, 0 * s)
    nu_local = 2.0 - n[:, 0] + 0.5 * n[:, 1]                    # front (x < 0) higher; upper side 0.5 more
    c = heat_curves(sd, nu_local, length=0.1)
    for branch in ("upper", "lower"):
        x, v = c[branch]
        assert np.all(np.diff(x) >= 0) and x[0] == pytest.approx(0.0, abs=1e-9)
        assert x[-1] == pytest.approx(1.0, abs=1e-3)            # leading to trailing edge over one diameter
        assert v[0] == pytest.approx(3.0, abs=0.02) and v[-1] == pytest.approx(1.0, abs=0.02)
    mid_u = c["upper"][1][np.argmin(np.abs(c["upper"][0] - 0.5))]
    mid_l = c["lower"][1][np.argmin(np.abs(c["lower"][0] - 0.5))]
    assert mid_u == pytest.approx(2.5, abs=0.05) and mid_l == pytest.approx(1.5, abs=0.05)
    # a body pitched by alpha is read in its own frame: rotating the points and alpha together changes nothing
    from aerolume.app.pages.results import rotate
    xr, yr = rotate(pts[:, 0], pts[:, 1], -20.0)
    sd2 = SurfaceData("Cylinder", s, xr, yr, n[:, 0], n[:, 1], 0 * s, 0 * s)
    c2 = heat_curves(sd2, nu_local, alpha_deg=20.0, length=0.1)
    assert np.allclose(np.sort(c2["upper"][1]), np.sort(c["upper"][1]), atol=1e-9)


def test_result_diagnostics_name_change_norms_like_the_live_dock():
    """Change norms ("du_l2", "dtheta_max") read as the live dock and the report name them, with units that
    are right: the L2 norm is a domain integral (unit × L in 2D, × L^1.5 in 3D), so it carries none, and
    θ changes are in ΔT. Keys nobody names keep the report's own reading."""
    from types import SimpleNamespace

    from aerolume.app.pages.results import result_diagnostics_rows
    from aerolume.app.widgets.diagnostics import describe
    rows = result_diagnostics_rows(SimpleNamespace(diagnostics={"div_rms": 1e-6, "du_l2": 6.6e-4,
                                                                "dtheta_max": 1.3e-4, "mystery_key": 2.0}))
    by = {r[0]: r for r in rows}
    assert "RMS velocity divergence" in by
    l2 = next(r for r in rows if r[0].startswith("L2 change of u"))
    th = next(r for r in rows if r[0].startswith("Max change of θ"))
    assert l2[2] == "" == describe("du_l2").unit and th[2] == "ΔT" == describe("dtheta_max").unit
    assert "mystery key" in by


def test_a_legend_restyled_for_another_theme_recolours_its_entries(qapp):
    """Legend entries made in the light theme stayed dark grey on the dark plot after a theme switch."""
    from aerolume.app.widgets.plots import LinePlot, style_legend, tokens
    p = LinePlot(xlabel="x / L", ylabel="Nu")
    p.set_series("upper", [0, 1], [3, 1])
    p.set_series("lower", [0, 1], [3, 1.1])
    tk = dict(tokens())
    tk["text"] = "#fa0a0a"
    style_legend(p.legend, tk)
    htmls = [lab.item.toHtml().lower() for _s, lab in p.legend.items]
    assert len(htmls) == 2 and all("#fa0a0a" in h for h in htmls), htmls
    p.close()


def test_compact_columns_never_elide_a_bold_value(qapp):
    """The heat balance's bold residual "−2.40×10⁻⁴" was elided at the fixed numeric width (found on a real
    Re 40 run); a numeric column is at least as wide as its widest cell."""
    from aerolume.app.widgets.kpi import Cell, DataTable
    from aerolume.app.widgets.results3d import PART_COL_W, compact_columns
    t = DataTable(["Term", "Rate", "Unit"], stretch_last=False)
    t.set_data(["Term", "Rate", "Unit"], [["In: Bodies", "10.69", "W/m"],
                                          [Cell("Residual (in − out − stored)", bold=True),
                                           Cell("−2.40×10⁻⁴", bold=True), "W/m"]], numeric=[1])
    compact_columns(t, (1,))
    assert t.columnWidth(1) >= max(PART_COL_W, t.sizeHintForColumn(1))
    assert t.maximumWidth() >= sum(t.columnWidth(j) for j in range(3))


def test_results_page_shows_heat_fans_diagnostics_and_smoke_channels(qapp, app_state, tmp_path):
    from aerolume.app.pages.results import SMOKE_COMPOSITE, TAB_KEYS, ResultsPage
    state, jobs = app_state
    page = ResultsPage(state, jobs)
    page.resize(1400, 900)
    page.show()
    state.select_simulation("th1")
    pump(qapp, 0.2)
    for key in ("heat", "fans", "diagnostics", "fields"):
        assert page.tabs.isTabVisible(TAB_KEYS.index(key)), key
    page.handle_action("heat")
    pump(qapp, 0.2)
    assert page.heat_table.rowCount() == 2 and page.heat_view.has_image()
    assert page.balance_table.rowCount() >= 4
    page.handle_action("fans")
    assert page.fan_table.rowCount() == 1 and "Thrust (N/m)" in [page.fan_table.horizontalHeaderItem(j).text()
                                                                  for j in range(page.fan_table.columnCount())]
    page.handle_action("diagnostics")
    pump(qapp, 0.3)
    assert page.diag_table.rowCount() >= 3 and len(page.diag_history.log) > 0
    # the summary carries the Nusselt number too
    page.handle_action("summary")
    texts = [page.summary_table.item(i, 0).text() for i in range(page.summary_table.rowCount())]
    assert "Mean Nusselt number Nu" in texts
    # fields: temperature with the thermal map, and the composite of the two smoke channels with a legend
    page.handle_action("fields")
    keys = [page.fld_combo.itemData(i) for i in range(page.fld_combo.count())]
    assert "temperature" in keys and SMOKE_COMPOSITE in keys
    page.fld_combo.setCurrentIndex(keys.index(SMOKE_COMPOSITE))
    pump(qapp, 0.1)
    assert [t for _c, t in page.field_view._legend] == ["Upper", "Lower"]
    page.fld_combo.setCurrentIndex(keys.index("temperature"))
    pump(qapp, 0.1)
    assert page.field_view._label.startswith("Temperature") and page.field_view._units == "°C"
    assert not page.field_view._legend
    page.grab()
    page.close()


def test_particle_replay_runs_in_the_stored_field(qapp, app_state):
    pytest.importorskip("torch")
    from aerolume.app.pages.results import ResultsPage
    state, jobs = app_state
    page = ResultsPage(state, jobs)
    page.resize(1200, 800)
    page.show()
    state.select_simulation("th1")
    page.handle_action("fields")
    pump(qapp, 0.2)
    page.fld_replay.setChecked(True)
    end = time.monotonic() + 15
    while time.monotonic() < end and (page._replay is None or page._replay.count() < 20):
        pump(qapp, 0.05)
    assert page._replay is not None and page._replay.count() >= 20
    assert len(page.field_view._particles) > 0 and "frozen field" in page.field_view.caption()
    page.fld_replay.setChecked(False)
    pump(qapp, 0.05)
    assert page._replay is None and len(page.field_view._particles) == 0
    page.close()


def test_simulate_offers_temperature_and_shows_placements(qapp, app_state):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = app_state
    page = SimulatePage(state, jobs)
    page.resize(1400, 900)
    page.show()
    state.select_simulation("th1")
    pump(qapp, 0.3)
    keys = [page.field_combo.itemData(i) for i in range(page.field_combo.count())]
    assert "temperature" in keys
    i = keys.index("temperature")
    page.field_combo.setCurrentIndex(i)
    assert page.cmap_combo.currentText() == "Thermal (default)"
    pls = page.view.placements()
    assert {p.key for p in pls} == {"emitters:0", "emitters:1", "fans:0"}
    assert not page.draw_strip.isVisible() and not page.move_tool.enabled       # editing needs explore
    page.close()


def test_compare_page_rows_carry_heat_and_fans(qapp, app_state):
    from aerolume.app.pages.compare import ComparePage
    state, jobs = app_state
    state.compare_ids[:] = ["th1", "th2"]
    page = ComparePage(state, jobs)
    page.resize(1400, 900)
    page.show()
    page.on_shown()
    pump(qapp, 0.5)
    texts = [page.table.item(i, 0).text() for i in range(page.table.rowCount())]
    assert any("Nusselt" in t for t in texts), texts
    # the heat and fan quantities are bar-chart choices; the Fields tab compares the temperature
    bars = [page.bar_metric.itemData(i) for i in range(page.bar_metric.count())]
    assert {"nusselt", "heat_rate"} <= set(bars), bars
    i = page.bar_metric.findData("nusselt")
    page.bar_metric.setCurrentIndex(i)
    pump(qapp, 0.1)
    page._fill_fields()
    keys = [page.field_combo.itemData(i) for i in range(page.field_combo.count())]
    assert keys[-1] == "temperature" and keys[:4] == ["speed", "cp", "vorticity", "p"]
    page.field_combo.setCurrentIndex(keys.index("temperature"))
    pump(qapp, 0.2)
    assert page._thumb_views and all(v.has_image() for v in page._thumb_views)
    page.close()


def _toggle_item(page, sid: str, on: bool) -> None:
    from PySide6.QtCore import Qt
    for i in range(page.sim_list.count()):
        it = page.sim_list.item(i)
        if it.data(Qt.ItemDataRole.UserRole) == sid:
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)


def test_report_page_offers_physics_sections_by_what_the_runs_have(qapp, app_state):
    from aerolume.app.pages.report import ReportPage, grid_study_possible
    state, jobs = app_state
    page = ReportPage(state, jobs)
    pump(qapp, 0.2)
    boxes = page.section_boxes
    for key in ("heat", "flowvis", "fans", "grid"):
        assert key in boxes, list(boxes)
    # th1 has heat, two smoke emitters and a fan; th2 only heat
    assert boxes["heat"].isEnabled() and boxes["flowvis"].isEnabled() and boxes["fans"].isEnabled()
    assert not boxes["grid"].isEnabled() and "grid resolution" in boxes["grid"].toolTip()
    _toggle_item(page, "th1", False)
    pump(qapp, 0.1)
    page._update_controls()
    assert boxes["heat"].isEnabled()
    assert not boxes["flowvis"].isEnabled() and "emitters" in boxes["flowvis"].toolTip()
    assert not boxes["fans"].isEnabled() and not boxes["fans"].isChecked()
    assert "fans" not in page.selected_sections() and "heat" in page.selected_sections()
    # three runs of one set-up at three resolutions form a grid study; two do not
    sims = [state.project.simulations["th2"]]
    import copy
    for cpr in (24, 48):
        s = copy.deepcopy(sims[0])
        s.id, s.cfd.cells_per_ref = f"g{cpr}", cpr
        sims.append(s)
    assert not grid_study_possible(sims[:2]) and grid_study_possible(sims)
    page.close()


def test_report_page_real_html_report_with_heat_sections(qapp, app_state):
    rep = pytest.importorskip("aerolume.report")
    if not hasattr(rep, "generate") or "heat" not in getattr(rep, "SECTIONS", ()):
        pytest.skip("the report module has no heat-transfer section")
    from aerolume.app.pages.report import ReportPage
    state, jobs = app_state
    page = ReportPage(state, jobs)
    pump(qapp, 0.3)
    page._report_module_loaded(rep)
    _toggle_item(page, "th2", False)
    page._update_controls()
    page.html_radio.setChecked(True)
    page.generate()
    deadline = time.monotonic() + 240
    while page._call is not None and time.monotonic() < deadline:
        pump(qapp, 0.05)
    assert page.state_badge.text() == "Ready", page.status.text()
    html = page.last_path.read_text(encoding="utf-8")
    for heading in (rep.SECTION_TITLES["heat"], rep.SECTION_TITLES["fans"], rep.SECTION_TITLES["flowvis"]):
        assert heading in html, heading
    assert "Nusselt" in html
    page.close()


def _hot_sphere():
    """The critic's hot sphere (as in test_simulation_peak_speed): D 0.1 m at Re 100, 60 °C in 20 °C air."""
    from aerolume.core import fluid as F
    from aerolume.geometry3d import design_from_meshes, sphere
    from aerolume.model import CFDSettings
    des = design_from_meshes([sphere(0.1, 32)], "Hot sphere", kind="generic")
    for m in des.meshes:
        m.thermal, m.thermal_value = "temperature", 60.0
    air = F.air(20.0)
    sim = Simulation("s", des.id, conditions=FlowConditions(air, 100.0 * air.kinematic_viscosity / des.length_ref,
                                                            0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", reynolds_mode="real"),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True))
    return des, sim


def test_buoyancy_dominated_summary_leads_with_the_buoyant_scale_coefficients():
    """Ri 627: the free-stream CD 6.14 / CL 119 are buoyant force over a tiny q. The Summary carries CD and CL
    on ½ρu_b² directly under the coefficients, with the runner's own numbers, and a caption naming u_b."""
    from aerolume.app.pages import results as R
    from aerolume.app.widgets import results3d as R3
    from aerolume.model import CFD3DResult
    from aerolume.simulation import physics as PH
    from aerolume.simulation import plan3d as P3
    des, sim = _hot_sphere()
    r = CFD3DResult(sim_id="s", cd=Stat(6.14, 0.05, 6.0, 6.3, 0.01), cl=Stat(119.08, 0.4, 118.0, 120.0, 0.08),
                    re_real=100.0, re_sim=100.0, t_end=30.0, t_avg_start=10.0)
    plan = P3.plan_simulation3d(des, sim)
    bs = R.buoyant_scale(des, sim)
    assert bs is not None and bs.ri == pytest.approx(plan.physics.ri_flow, rel=1e-9) and bs.ri > PH.RI_BUOYANT
    assert bs.factor == pytest.approx(1.0 / plan.speed_ratio ** 2, rel=1e-9)        # u_b / U = sqrt(Ri)
    rows = R3.summary_rows3d(r, des, sim)
    names = [R._cell_text(row[0]) for row in rows]
    n_coef = len(R3.COEFFS_3D)
    assert names[n_coef:n_coef + 2] == ["CD on the buoyant scale", "CL on the buoyant scale"], names
    cd_b, cl_b = (float(R._cell_text(rows[n_coef + i][1]).replace(R.MINUS, "-")) for i in (0, 1))
    assert cd_b == pytest.approx(6.14 * bs.factor, rel=0.02) and cl_b == pytest.approx(119.08 * bs.factor, rel=0.02)
    # the same numbers the runner writes into the result's notes
    note = PH.buoyant_coefficients_note(r, plan)
    assert f"Cd {PH.units.sig(6.14 * bs.factor, 3)}" in note and f"Cl {PH.units.sig(119.08 * bs.factor, 3)}" in note
    assert "u_b" in R._cell_text(rows[n_coef][0]) or "u_b =" in rows[n_coef][0].tooltip
    foot = R3.summary_note3d(r, des, sim, rows)
    assert "u_b = " in foot and "buoyant scale" in foot
    # buoyancy off (a forced run on the free stream): no buoyant rows, no caption
    sim.thermal.buoyancy = False
    assert R.buoyant_scale(des, sim) is None
    assert not any("buoyant scale" in R._cell_text(row[0]) for row in R3.summary_rows3d(r, des, sim))
    assert "u_b" not in R3.summary_note3d(r, des, sim)


def test_buoyancy_dominated_2d_summary_has_the_buoyant_rows_under_the_coefficients():
    from aerolume.app.pages import results as R
    from aerolume.core import fluid as F
    from aerolume.geometry import templates
    from aerolume.model import CFDSettings
    des = templates.make("cylinder", diameter=0.1)
    des.bodies[0].thermal, des.bodies[0].thermal_value = "temperature", 60.0
    air = F.air(20.0)
    sim = Simulation("t", des.id, conditions=FlowConditions(air, 100 * air.kinematic_viscosity / 0.1, 0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", reynolds_mode="real"),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True))
    r = thermal_result("t")
    bs = R.buoyant_scale(des, sim)
    assert bs is not None and 0 < bs.factor < 0.01
    names = [R._cell_text(row[0]) for row in R.summary_rows(r, des, sim)]
    i = names.index("Moment coefficient Cm")
    assert names[i + 1:i + 3] == ["Cd on the buoyant scale", "Cl on the buoyant scale"], names
    assert "u_b = " in R.summary_note(des, sim, r)
