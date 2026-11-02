"""Report sections for placed physics and grid studies (DESIGN.md section 10; analysis.gci).

The results are synthetic but shaped like the solvers' (a heated cylinder with a
Nusselt history, a fan with a regulated thrust history, two smoke channels, a
three-grid study), so the sections are exercised end to end: content, numbers,
figures and both renderers.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import report_fixtures as rf

from aerolume import report
from aerolume.core import units
from aerolume.model import (
    Body,
    CFDResult,
    CFDSettings,
    Design,
    Emitter,
    Fan,
    FieldData,
    FlowConditions,
    HeatSource,
    Port,
    Simulation,
    Stat,
    ThermalSettings,
    WallThermal,
)
from aerolume.report import content_physics as CP
from aerolume.report.document import FigureBlock, Table


def _hist(seed: int, n: int = 3000, **series: tuple[float, float]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 30.0, n)
    out = {"t": t}
    for k, (mean, noise) in series.items():
        out[k] = mean * (1 + 0.5 * np.exp(-t / 3)) + rng.normal(0, noise, n)
    return out


def heated_cylinder() -> tuple[Design, Simulation, CFDResult]:
    pts = rf.circle(0.0, 0.0, 0.05, 96)
    d = Design(name="Heated cylinder", kind="generic", bodies=[Body("cylinder", pts, thermal="temperature",
                                                                    thermal_value=60.0)],
               length_ref=0.1, force_ref=0.1, id="d-cyl")
    sim = Simulation(name="Forced convection · Re 40", design_id=d.id, id="s-heat", status="done",
                     conditions=FlowConditions(rf.AIR, 0.006, 0.0),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=False,
                                             walls=[WallThermal("bottom", "temperature", 20.0)]),
                     heat_sources=[HeatSource("chip", "rect", (0.3, 0.0), (0.02, 0.02), 5.0)],
                     emitters=[Emitter("rake", "dye", "rake", (-0.2, -0.05), end=(-0.2, 0.05), count=5),
                               Emitter("tracer", "dye", "point", (-0.2, 0.08), color="#e98a2e")])
    h = _hist(1, cd=(1.6, 0.002), nu=(3.3, 0.01), heat_rate=(4.4, 0.02))
    nx, ny = 240, 120
    ext = (-0.3, 0.9, -0.3, 0.3)
    xx = np.linspace(ext[0], ext[1], nx)[None, :]
    yy = np.linspace(ext[2], ext[3], ny)[:, None]
    temp = 20.0 + 40.0 * np.exp(-((yy / 0.05) ** 2)) * np.exp(-np.clip(xx, 0, None) / 0.4) * (xx > -0.05)
    mask = ((xx ** 2 + yy ** 2) < 0.05 ** 2).astype(np.float32)
    dye0 = np.clip(np.exp(-((yy / 0.06) ** 2)) * (xx > -0.2), 0, 1)
    dye1 = np.clip(np.exp(-(((yy - 0.08) / 0.02) ** 2)) * (xx > -0.2), 0, 1)
    u = 0.006 * np.ones((ny, nx))
    inst = FieldData(ext, {"u": u, "v": 0 * u, "temperature": temp, "dye": dye0, "dye1": dye1}, mask, "instant")
    mean = FieldData(ext, {"u": u, "v": 0 * u, "temperature": temp}, mask, "mean")
    r = CFDResult(sim_id=sim.id, cd=Stat(1.6, 0.01, 1.58, 1.62, 0.002), cl=Stat(0.0, 0.001, -0.01, 0.01, 1e-4),
                  cm=Stat(0.0, 0.0, 0.0, 0.0, 1e-5), re_real=40.0, re_sim=40.0, grid=(960, 480),
                  cells_per_ref=80.0, t_end=30.0, t_avg_start=10.0, converged=True,
                  nusselt=Stat(3.30, 0.01, 3.2, 3.4, 0.004), heat_rate=4.40,
                  diagnostics={"divergence_rms": 2.1e-7, "cfl_max": 0.48, "heat_in_bodies": 4.40,
                               "heat_in_sources": 5.0, "heat_out_outlet": 9.35, "heat_storage_rate": 0.02},
                  history=h, mean_fields=mean, instant_fields=inst, bodies_wind=list(d.bodies))
    td = np.linspace(0, 30, 300)
    r.diagnostics_history = {"t": td, "div_rms": 1e-7 + 0 * td, "div_max": 5e-7 + 0 * td,
                             "du_rel": 1e-2 * np.exp(-td / 5), "dtheta_rel": 2e-2 * np.exp(-td / 6),
                             "cfl": 0.45 + 0 * td, "dt": 0.004 + 0 * td}
    return d, sim, r


def fan_case() -> tuple[Design, Simulation, CFDResult]:
    d = rf.vehicle_design()
    sim = Simulation(name="Cooling fan", design_id=d.id, id="s-fan", status="done",
                     conditions=FlowConditions(rf.AIR, 5.0, 0.0),
                     fans=[Fan("Radiator fan", (1.0, 0.5), (1.0, 0.0), diameter=0.3, mode="thrust", value=2.0)],
                     ports=[Port("Exhaust", "jet", (4.0, 0.3), (1.0, 0.0), 0.05, 12.0, temperature=80.0)])
    h = _hist(2, cd=(0.6, 0.01), fan0_thrust=(2.0, 0.01))
    r = CFDResult(sim_id=sim.id, cd=Stat(0.6, 0.05, 0.5, 0.7, 0.01), cl=Stat(0.1, 0.02, 0, 0.2, 0.005),
                  cm=Stat(0.0, 0.0, 0.0, 0.0, 0.001), re_real=1.3e6, re_sim=2e4, grid=(720, 450),
                  cells_per_ref=150.0, t_end=30.0, t_avg_start=10.0, converged=True,
                  fans_report=[{"name": "Radiator fan", "mode": "thrust", "thrust": 1.98, "flow": 1.71,
                                "induced_velocity": 5.7, "power": 11.2, "dp": 6.6}],
                  history=h)
    return d, sim, r


def grid_case(pr: rf.FakeProject) -> list[str]:
    d = rf.airfoil_design("0012", name="NACA 0012 grid study")
    ids = []
    for cpr, cd, se in ((128.0, 1.5115, 0.0004), (90.5, 1.5231, 0.0004), (64.0, 1.5462, 0.0005)):
        sim = rf.cfd_sim(d, f"Grid {cpr:g}", 40.0, 0.0, f"s-grid-{int(cpr)}")
        sim.cfd = CFDSettings(quality="custom", cells_per_ref=int(cpr), reynolds_mode="capped", reynolds_cap=2e4)
        n = round(cpr * 30), round(cpr * 20)
        r = CFDResult(sim_id=sim.id, cd=Stat(cd, 0.01, cd - 0.02, cd + 0.02, se), cl=Stat(0.0, 0.0, 0, 0, 1e-6),
                      cm=Stat(0.0, 0.0, 0, 0, 1e-6), re_real=8e5, re_sim=2e4, grid=n, cells_per_ref=cpr,
                      t_end=40, t_avg_start=10, converged=True)
        pr.add(d, sim, r)
        ids.append(sim.id)
    return ids


@pytest.fixture(scope="module")
def project() -> rf.FakeProject:
    pr = rf.FakeProject("Placed physics study")
    pr.add(*heated_cylinder())
    pr.add(*fan_case())
    grid_case(pr)
    return pr


def test_sections_appear_only_with_their_physics(project):
    doc = report.build_document(project, ["s-heat"])
    keys = [s.key for s in doc.sections]
    assert "heat" in keys and "flowvis" in keys and "fans" not in keys and "grid" not in keys
    doc = report.build_document(project, ["s-fan"])
    assert "fans" in [s.key for s in doc.sections] and "heat" not in [s.key for s in doc.sections]
    plain = report.build_document(rf.make_project(field_res=(80, 40)), ["s-2412-cfd"])
    assert not {"heat", "flowvis", "fans", "grid"} & {s.key for s in plain.sections}


def test_heat_section_numbers_and_balance(project):
    doc = report.build_document(project, ["s-heat"], sections=["heat", "convergence", "methodology"])
    text = doc.text()
    heat = next(s for s in doc.sections if s.key == "heat")
    nu_row = next(b for b in heat.blocks if isinstance(b, Table) and b.header[:3] == ["Quantity", "Mean",
                                                                                     "Std. error"]).rows[0]
    assert [c if isinstance(c, str) else c.text for c in nu_row[:3]] == ["Mean Nusselt number Nu", "3.300", "0.004"]
    assert "60.00 °C" in text and "Domain wall: bottom" in text and "chip" in text
    tables = [b for b in heat.blocks if isinstance(b, Table)]
    balance = next(t for t in tables if t.header[0] == "Heat balance term")
    resid = balance.rows[-1][1].text
    # in 4.40 + 5.0, out 9.35, stored 0.02: residual 0.03 W/m of 9.40 (0.32 %)
    assert resid.startswith("0.03") and "0.32 %" in resid
    assert any(isinstance(b, FigureBlock) for b in heat.blocks)
    assert "RMS velocity divergence" in text and "Heat transfer model" in text and "de Vahl Davis" in text
    conv = next(s for s in doc.sections if s.key == "convergence")
    assert any(isinstance(b, FigureBlock) and "solver diagnostics over the run" in b.caption for b in conv.blocks)


def test_emitters_map_to_smoke_channels_by_colour():
    ems = [Emitter("a", color="#FFFFFF"), Emitter("b", color="#e98a2e"), Emitter("c", color="#ffffff"),
           Emitter("p", species="particles"), Emitter("off", color="#123456", enabled=False)]
    chans = CP.dye_channels(ems)
    assert [(c, n) for c, _, n in chans] == [("dye", ["a", "c"]), ("dye1", ["b"])]
    five = [Emitter(f"e{i}", color=f"#0000{i:02x}") for i in range(6)]
    assert len(CP.dye_channels(five)) == 4               # at most four channels


def test_fan_section_against_momentum_theory(project):
    doc = report.build_document(project, ["s-fan"], sections=["fans"])
    fans = doc.sections[0]
    t = next(b for b in fans.blocks if isinstance(b, Table) and b.header[0] == "Fan")
    row = [c if isinstance(c, str) else c.text for c in t.rows[0]]
    w = 0.5 * (-5.0 + math.sqrt(25.0 + 2 * 1.98 / (1.225 * 0.3)))     # 2D disc: A = width
    assert row[5] == units.sig(w, 3) == "0.491"
    assert any(isinstance(b, Table) and b.header[0] == "Port" for b in fans.blocks)
    assert any(isinstance(b, FigureBlock) for b in fans.blocks)        # the thrust history


def test_grid_section_reports_gci(project):
    ids = [s for s in project.simulations if s.startswith("s-grid")]
    doc = report.build_document(project, ids, sections=["grid"])
    text = doc.text()
    assert "Grid convergence" == doc.sections[0].title
    assert "monotone" in text and "Celik" in text
    st = CP.studies_in(report.content.load_entries(project, ids))
    assert len(st) == 1 and st[0].sim_ids == ["s-grid-128", "s-grid-90", "s-grid-64"]
    q = st[0].quantities["cd"]
    assert q.convergence == "monotone" and 1.5 < q.p < 2.5 and q.phi_ext < 1.5115
    # the symmetric airfoil's Cl and Cm are ≈ 0 on every grid: named once as having nothing to
    # estimate, not given "noise-limited" rows and warnings that read as a verdict on the study
    gci_table = next(b for b in doc.sections[0].blocks if isinstance(b, Table) and b.header[0] == "Quantity")
    assert [r[0].text for r in gci_table.rows] == ["Cd"]
    assert "Cl, Cm: ≈ 0 on every grid (a symmetric flow)" in text
    assert "noise" not in text.split("Procedure:")[0].lower()
    assert "e-0" not in text                                                # no "2.0e-06" in narrow columns


def test_key_findings_leave_out_missing_dimensional_values(project):
    """A result without dimensional forces gets no "drag — N ..., needing — W/m" sentence."""
    doc = report.build_document(project, ["s-heat"], sections=["summary"])
    text = doc.text()
    assert "Cd = 1.6000 ± 0.0020" in text and "drag —" not in text and "needing —" not in text


def test_reports_render_with_physics_figures(project, tmp_path):
    ids = ["s-heat", "s-fan"] + [s for s in project.simulations if s.startswith("s-grid")]
    for fmt_ in ("html", "pdf"):
        info: dict = {}
        doc = report.build_document(project, ids)
        if fmt_ == "html":
            from aerolume.report.html import render_html
            info = render_html(doc, tmp_path / "r.html")
        else:
            from aerolume.report.pdf import render_pdf
            info = render_pdf(doc, tmp_path / "r.pdf")
        assert not info.get("errors"), info.get("errors")
        assert (tmp_path / f"r.{fmt_}").stat().st_size > 20_000


def test_appendix_lists_provenance(project):
    _, sim, r = heated_cylinder()
    r.provenance = {"config_hash": "ab" * 32, "seed": None, "software": {"aerolume": "0.3.0",
                                                                         "packages": {"torch": "2.14.1"}},
                    "plan": {"solver_hash": "cd" * 32}}
    pr = rf.FakeProject("p")
    pr.add(Design(name="c", kind="generic", bodies=[Body("c", rf.circle(0, 0, 0.05))], length_ref=0.1, force_ref=0.1,
                  id=sim.design_id), sim, r)
    doc = report.build_document(pr, [sim.id])
    app = next(s for s in doc.sections if s.key == "appendix")
    prov = next(b for b in app.blocks if isinstance(b, Table) and b.header[1] == "Configuration hash")
    assert prov.rows[0][1] == "ab" * 6 and prov.rows[0][2] == "cd" * 6 and prov.rows[0][3] == "none"
    assert "torch 2.14.1" in prov.rows[0][4]


def room_case(sid: str, *, fan: bool, speed: float, t_max: float) -> tuple[Design, Simulation, CFDResult]:
    """A closed room heated by a 300 W/m heater (optionally with a fan and a vent), as the runner reports it:
    no forces, no Nusselt number, and a heat rate that is the heater power set."""
    d = Design(name=f"Room {sid}", kind="generic", bodies=[], length_ref=1.0, force_ref=1.0, id=f"d-{sid}")
    sim = Simulation(name=f"Room {sid}", design_id=d.id, id=sid, status="done",
                     conditions=FlowConditions(rf.AIR, speed, 0.0),
                     cfd=CFDSettings(domain_mode="closed", domain_box=(0.0, 6.0, 0.0, 3.0)),
                     thermal=ThermalSettings(enabled=True, t_ambient=20.0, buoyancy=True),
                     heat_sources=[HeatSource("Heater", "rect", (3.5, 0.3), (0.3, 0.2), 300.0)],
                     fans=[Fan("Fan", (1.0, 1.5), (1.0, 0.0), diameter=0.3, mode="thrust", value=1.0)] if fan else [],
                     ports=[Port("Vent", "vent", (5.8, 2.8), (1.0, 0.0), 0.2, 0.5)])
    diag = {"heat_sources_set": 300.0, "heat_in_sources": 300.00003, "heat_out_ports": 120.0, "heat_in_ports": 0.0,
            "heat_in_walls": -150.0, "heat_storage_rate": 30.0, "t_fluid_max": t_max, "t_fluid_mean": 24.5,
            "t_fluid_min": 20.0}
    r = CFDResult(sim_id=sid, re_real=3.1e4, re_sim=3.1e4, grid=(240, 120), cells_per_ref=40.0, t_end=60.0,
                  t_avg_start=20.0, converged=False, heat_rate=300.00003, diagnostics=diag,
                  fans_report=[{"name": "Fan", "mode": "thrust", "thrust": 1.0, "thrust_stderr": 1e-16,
                                "flow": 0.1352, "induced_velocity": 0.270, "power": 0.2705, "dp": 2.0}] if fan else [])
    return d, sim, r


def test_closed_room_fans_and_heater_are_not_checked_against_free_stream_theory():
    """Report Table 10 used to set 'w, momentum theory 0.904' (the planner's reference speed) beside the
    room's induced 0.270 and call it a check; the room's 'heat rate' was the 300 W/m heater power set."""
    pr = rf.FakeProject("Rooms")
    pr.add(*room_case("s-room", fan=True, speed=0.904, t_max=41.2))
    doc = report.build_document(pr, ["s-room"], sections=["summary", "heat", "fans"])
    fans = next(s for s in doc.sections if s.key == "fans")
    t = next(b for b in fans.blocks if isinstance(b, Table) and b.header[0] == "Fan")
    row = [c if isinstance(c, str) else c.text for c in t.rows[0]]
    assert row[5] == "—" and "0.904" not in row and row[2] == "1.000 (set)"
    assert "closed domain" in t.caption and "next to Froude" not in t.caption
    text = doc.text()
    assert "not shown in a closed domain" in text
    assert "Heater power (set) [W/m]" in text and "set, not measured" in text
    assert "Heat put into the fluid [W/m]" not in text                       # not presented as a measurement
    kpis = [k for s in doc.sections for b in s.blocks if hasattr(b, "items") for k in getattr(b, "items", [])
            if hasattr(k, "label")]
    assert any(k.label == "Heater power" and "set" in k.sub for k in kpis)
    assert "hottest fluid 41.20 °C" in text and "heat rate 300" not in text


def test_two_rooms_compare_their_inputs_not_their_speeds():
    pr = rf.FakeProject("Rooms")
    pr.add(*room_case("s-room", fan=True, speed=0.904, t_max=41.2))
    pr.add(*room_case("s-room2", fan=False, speed=0.31, t_max=52.7))
    doc = report.build_document(pr, ["s-room", "s-room2"], sections=["comparison"])
    text = doc.text()
    assert "different speeds" not in text and "different Reynolds" not in text
    assert "driven by different inputs: fans" in text
    tables = [b for s in doc.sections for b in s.blocks if isinstance(b, Table) and b.header[0] == "Simulation"
              and "Heat transfer compared" in b.caption]
    assert len(tables) == 2 and all(len(t.header) <= 1 + 2 * CP.PHYS_COLUMNS_PER_TABLE for t in tables)
    assert ", 1 of 2" in tables[0].caption                      # four quantities: split, not one overflowing table
    assert not any("Heat rate [W/m]" in t.header for t in tables)    # the set heater power is not a measured rate
    t = next(t for t in tables if "Heater power (set) [W/m]" in t.header)
    i = t.header.index("Heater power (set) [W/m]")
    assert [r[i] for r in t.rows] == ["300.0", "300.0"] and t.rows[1][i + 1].text == "—"   # no Δ on a setting
    t = next(t for t in tables if "Hottest fluid [°C]" in t.header)
    j = t.header.index("Hottest fluid [°C]")
    assert t.rows[1][j + 1].text == "+11.5 K"                   # a temperature difference, not a % of °C
