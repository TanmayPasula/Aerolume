"""Fixes of critique round 4 on the run pages (Results, Compare, the Diagnostics panel, the job manager).

* the heat balance judges its residual against the heat *through* the domain and lists the hot wall's
  heat in and the cold wall's heat out (the de Vahl Davis cavity's round-off residual read in red);
* smoke that filled a closed room is contrast-stretched per channel, and the page says so;
* the Fans tab rounds the volume flow, the disc velocity and the ideal power to their errors;
* the Conditions panel labels a heater room's ΔT, Ra and Gr as planned estimates beside the measured ones;
* the Summary quotes the published range of the heated cylinder beside the correlation;
* Compare opens its chart on a measured quantity and shows "—" in Δ for settings;
* a finished run stores about 200 rows of its diagnostics, and the panel's plots keep a readable height.
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
    CFDResult,
    CFDSettings,
    Simulation,
    Stat,
    ThermalSettings,
    WallThermal,
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


def _text(c) -> str:
    return c.text if hasattr(c, "text") else str(c)


def _cavity(residual: float = -5.28e-9, walls: float = -4.66e-9, storage: float = 6.24e-10):
    """The de Vahl Davis cavity as the critic's run stored it: 0.102 W/m in through the hot wall, out through
    the cold one, so the walls net to ~0; only the net term in the diagnostics."""
    sim = Simulation(name="Heated cavity", design_id="d",
                     cfd=CFDSettings(domain_mode="closed"),
                     thermal=ThermalSettings(enabled=True, t_ambient=15.0, walls=[
                         WallThermal("left", "temperature", 15.5), WallThermal("right", "temperature", 14.5)]))
    diag = {"heat_in_bodies": 0.0, "heat_in_walls": walls, "heat_in_sources": 0.0, "heat_in_ports": 0.0,
            "heat_in_inlet": 0.0, "heat_out_outlet": 0.0, "heat_out_ports": 0.0, "heat_storage_rate": storage,
            "heat_balance_residual": residual}
    r = CFDResult(sim_id="c", heat_rate=0.10213, nusselt=Stat(mean=4.5624, stderr=1.9e-5), diagnostics=diag)
    return r, sim


def test_cavity_balance_lists_the_hot_and_cold_walls_and_reads_round_off_as_round_off():
    from aerolume.app.pages import results as R
    r, sim = _cavity()
    rows, note, tone = R.heat_balance_rows(r, sim, None)
    names = [_text(row[0]) for row in rows]
    assert tone is None, (rows, note)
    assert "In: Hot wall (left)" in names and "Out: Cold wall (right)" in names
    hot = rows[names.index("In: Hot wall (left)")]
    assert _text(hot[1]).startswith("0.1021")
    # the zero terms of a closed box are named in the note, not listed as rows of zeros
    assert not any(n.startswith(("In: Jets", "In: Inlet", "Out: Outlet", "Out: Vents", "In: Bodies")) for n in names)
    assert "jets" in note and "round-off" in note and "0.1021" in note
    resid = rows[-1]
    assert _text(resid[0]).startswith("Residual") and resid[1].tone is None
    # without the simulation the walls stay one net term, still judged against the measured heat rate
    rows2, _note2, tone2 = R.heat_balance_rows(r)
    assert tone2 is None and any(_text(x[0]).startswith("In: Domain walls") for x in rows2)


@pytest.mark.parametrize(("share", "want"), [(0.02, "bad"), (0.005, "warn"), (0.0005, None)])
def test_balance_residual_colour_follows_its_share_of_the_throughput(share, want):
    from aerolume.app.pages import results as R
    r, sim = _cavity(residual=share * 0.10213)
    _rows, _note, tone = R.heat_balance_rows(r, sim, None)
    assert tone == want


def test_filled_smoke_is_stretched_and_sparse_smoke_is_not():
    from aerolume.viz import colormaps as CM
    rng = np.random.default_rng(1)
    filled = np.clip(0.85 + 0.1 * rng.standard_normal((72, 96)), 0, 1)
    filled[30:40, 40:60] = 0.4                     # a thinner patch: the structure the stretch brings out
    lo_hi = CM.dye_stretch(filled)
    assert lo_hi is not None and 0.4 <= lo_hi[0] < lo_hi[1] <= 1.0
    sparse = np.zeros((72, 96))
    sparse[30:34, :] = 0.9
    assert CM.dye_stretch(sparse) is None
    plain = CM.composite_dye([filled], ["#f0a040"])
    stretched = CM.composite_dye([filled], ["#f0a040"], ranges=[lo_hi])
    # stretched, the variation across the room (the flow's structure) spans far more of the colour range
    def spread(img):
        return float(img[..., :3].astype(float).mean(axis=2).std())
    assert spread(stretched) > 2 * spread(plain)
    from aerolume.app.pages.results import smoke_stretch_note
    note = smoke_stretch_note([lo_hi, None], closed=True)
    assert "filled the room" in note and "not its absolute concentration" in note


def test_fans_tab_rounds_flow_velocity_and_power_to_their_errors():
    from aerolume.app.pages import results as R
    sim = Simulation(name="Room", design_id="d", cfd=CFDSettings(domain_mode="closed"))
    t = np.linspace(0, 240, 4801)
    rng = np.random.default_rng(3)
    vel = 0.158 + 0.5 * rng.standard_normal(t.size)
    r = CFDResult(sim_id="r", t_avg_start=120.0, t_end=240.0,
                  fans_report=[{"name": "Fan", "mode": "thrust", "thrust": 1.0, "thrust_stderr": 0.0,
                                "volume_flow": 0.0791, "disc_velocity": 0.1583, "induced_velocity": 0.1583,
                                "ideal_power": 0.1583, "pressure_jump": 2.0, "target": 1.0, "diameter": 0.5}],
                  history={"t": t, "fan0_thrust": np.ones_like(t), "fan0_velocity": vel, "fan0_flow": 0.5 * vel})
    row = R.fan_rows(r, sim)[0]
    texts = [_text(c) for c in row]
    assert texts[3].endswith("(set)")
    for k in (4, 5, 7):                          # volume flow, induced velocity, ideal power
        assert " ± " in texts[k], texts
    # one rounding rule: the flow's text is the Summary's for the same statistic
    from aerolume.analysis import physics as P
    from aerolume.app.widgets.kpi import value_error_texts
    st = P.series_stat(r, "fan0_flow")
    v_t, se_t = value_error_texts(st.mean, st.stderr)
    assert texts[4] == f"{v_t} ± {se_t}"


def test_heater_room_conditions_label_the_plan_and_add_the_measurement():
    from aerolume.app.dialogs.physics_templates import build_template
    from aerolume.app.pages import results as R
    design, sims = build_template("enclosure")
    sim = sims[0]
    sc = R.run_scales(design, sim)
    if sc is None or not getattr(sc, "thermal", False):
        pytest.skip("the enclosure template has no heat-transfer scales here")
    from aerolume.simulation.physics import delta_t_is_estimate
    if not delta_t_is_estimate(sc):
        pytest.skip("the enclosure's ΔT is set, not estimated")
    t0 = float(sim.thermal.t_ambient)
    r = CFDResult(sim_id=sim.id, t_avg_start=1.0, t_end=2.0,
                  diagnostics={"t_fluid_max": t0 + 3.31, "t_fluid_min": t0, "t_fluid_mean": t0 + 1.0})
    items = {str(it[0]): str(it[1]) for it in R.closed_conditions(r, design, sim)}
    assert "ΔT (planned estimate)" in items and "ΔT measured" in items
    assert items["ΔT measured"] == "3.31 K"
    assert "Ra measured" in items and "Ra (planned estimate)" in items
    # the run's own record of its plan wins over today's planner (Ra scales with ΔT)
    r.notes = ["Planned estimate: Temperature scale ΔT = 44.1 K (estimated rise at the heater)"]
    assert R.planned_delta_t(r) == pytest.approx(44.1)
    items2 = {str(it[0]): str(it[1]) for it in R.closed_conditions(r, design, sim)}
    assert items2["ΔT (planned estimate)"] == "44.1 K" and items2["ΔT measured"] == "3.31 K"


def test_cylinder_summary_quotes_the_published_range(monkeypatch):
    from aerolume.app.pages import results as R
    monkeypatch.setattr(R, "reference_value",
                        lambda r, d, s: (3.2805, "nusselt", "Lange, Durst & Breuer (1998) correlation, Pr 0.7"))
    r = CFDResult(sim_id="c", re_sim=40.0, nusselt=Stat(mean=3.339, stderr=0.002))
    rows = R.reference_rows(r, SimpleNamespace(), SimpleNamespace())
    assert len(rows) == 2
    assert _text(rows[0][0]) == "Reference Nu: Lange" and "+1.8 %" in _text(rows[0][-1])
    assert _text(rows[1][0]) == "Published range Nu" and _text(rows[1][1]) == "3.200–3.255"
    assert _text(rows[1][-1]) == "measured +2.6 % above" and rows[1][-1].tone is None
    inside = CFDResult(sim_id="c", re_sim=40.0, nusselt=Stat(mean=3.23, stderr=0.002))
    assert R.reference_rows(inside, SimpleNamespace(), SimpleNamespace())[1][-1].tone == "good"
    other = CFDResult(sim_id="c", re_sim=100.0, nusselt=Stat(mean=5.0, stderr=0.01))
    assert len(R.reference_rows(other, SimpleNamespace(), SimpleNamespace())) == 1     # no range at Re 100


def test_compare_settings_have_no_delta_and_the_chart_opens_on_a_measurement():
    from aerolume.app.pages import compare as C
    from aerolume.model import Fan
    fan_sim = Simulation(name="Room", design_id="d", fans=[Fan(mode="thrust", value=1.0)])
    row = {"key": "fan_thrust", "label": "Fan thrust (all fans)", "input": False}
    assert C.is_set_value(row, fan_sim)
    assert C.is_set_value({"key": "heater_power", "input": True}, None)
    assert not C.is_set_value({"key": "fan_flow", "input": False}, fan_sim)
    state, why = C.delta_state({"key": "heater_power", "input": True}, {"value": 300.0, "delta": 0.0},
                               {"value": 300.0, "is_baseline": True})
    assert state == "none" and "setting" in why
    # rounded to the error where one is known (the fan room's flow read 0.07915 against 0.079 ± 0.028)
    assert C.fmt_value({"key": "fan_flow"}, 0.07918, stderr=0.0276) == "0.079"
    assert C.fmt_value({"key": "fan_flow"}, 0.07918) == "0.07918"
    assert "vent_heat_out" in C.BAR_METRICS


def test_finished_run_stores_a_decimated_diagnostics_table():
    from aerolume.app.jobs import attach_diagnostics_history
    from aerolume.app.pages.results import history_diagnostics
    from aerolume.app.widgets.diagnostics import STORED_ROWS, DiagnosticsLog
    log = DiagnosticsLog()
    for k in range(1, 7041):
        log.add(k, 0.01 * k, {"div_rms": 1e-6, "du_rel": 1.0 / k, "cfl": 0.4})
    r = CFDResult(sim_id="x", diagnostics={"div_rms": 1e-6})
    assert attach_diagnostics_history(r, log)
    dh = r.diagnostics_history
    assert len(dh["step"]) == STORED_ROWS and dh["step"][0] == 1 and dh["step"][-1] == 7040
    assert set(dh) >= {"step", "t", "div_rms", "du_rel", "cfl"}
    assert not attach_diagnostics_history(r, log)              # a table already there is kept
    one = DiagnosticsLog()
    one.add(5, 0.1, {"cfl": 0.4})
    assert not attach_diagnostics_history(CFDResult(sim_id="y"), one)   # one row is no history
    stored = history_diagnostics(r)
    assert stored is not None and len(stored) == STORED_ROWS and stored.source == "Stored with the result"


def test_short_dock_plots_keep_their_height_and_lead_with_divergence(qapp):
    from aerolume.app.widgets.diagnostics import BOTTOM_AXIS_H, MIN_PLOT_H, DiagnosticsLog, DiagnosticsPanel
    log = DiagnosticsLog()
    for k in range(1, 80):
        log.add(10 * k, 0.01 * k, {"div_rms": 1e-6, "du_rel": 1e-3, "p_residual": 1e-5, "cfl": 0.5,
                                   "dt": 0.004 + 1e-5 * np.sin(k), "umax": 3.2 + 0.1 * np.sin(k / 3),
                                   "steps_per_s": 400.0 + k})
    panel = DiagnosticsPanel(log=log)
    try:
        panel.resize(1280, 200)
        panel.show()
        panel.refresh(force=True)
        pump(qapp, 0.4)
        rows, _cols = panel.grid_shape()
        assert rows >= 2
        for pi in panel._panels.values():
            assert pi.height() >= MIN_PLOT_H + BOTTOM_AXIS_H - 6, pi.height()
        # row-major: the first row starts with the divergence and the change per step
        first_row = [g for g in panel._panels if panel._glw.ci.getItem(0, list(panel._panels).index(g)) is not None]
        assert list(panel._panels)[:2] == ["Divergence", "Change per step"]
        assert panel._glw.ci.getItem(0, 0) is panel._panels["Divergence"]
        assert panel._glw.ci.getItem(0, 1) is panel._panels["Change per step"]
        assert first_row
        # every plot carries its x axis while the rows scroll
        assert panel._bottom == set(panel._panels)
    finally:
        panel.close()
