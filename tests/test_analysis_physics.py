"""Heat-transfer and fan outputs (analysis.physics) and their rows in compare_rows.

The readers must give one answer for the report, the Compare page and the CSV
exports, read every spelling a solver may use for a fan quantity, measure standard
errors from recorded histories with the batch-means rules of the forces, and never
invent a number that the result does not hold.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aerolume.analysis import compare_rows, comparison_warnings
from aerolume.analysis import physics as P
from aerolume.model import CFD3DResult, CFDResult, FlowConditions, Stat


def _cfd(sid: str = "a", **kw) -> CFDResult:
    base = {"cd": Stat(1.0, 0.01, 0.9, 1.1, 0.002), "cl": Stat(0.1, 0.01, 0.0, 0.2, 0.002),
            "cm": Stat(0.0, 0.0, 0.0, 0.0, 1e-4), "re_real": 40.0, "re_sim": 40.0, "t_avg_start": 10.0}
    base.update(kw)
    return CFDResult(sim_id=sid, **base)


def _history(n: int = 4000, seed: int = 0, **series: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40, n)
    out = {"t": t}
    for k, mean in series.items():
        out[k] = mean + rng.normal(0, 0.01 * abs(mean) + 1e-3, n)
    return out


def test_nusselt_and_heat_rate_statistics():
    plain = _cfd()
    assert not P.has_thermal(plain) and P.nusselt_stat(plain) is None
    assert math.isnan(P.heat_rate_stat(plain)[0])
    own = _cfd(nusselt=Stat(3.3, 0.02, 3.2, 3.4, 0.004), heat_rate=4.4)
    assert P.has_thermal(own) and P.nusselt_stat(own).stderr == 0.004
    q, se, ok = P.heat_rate_stat(own)
    assert q == 4.4 and se == pytest.approx(4.4 * 0.004 / 3.3) and ok        # scaled from Nu's error
    # no Stat on the result: measured from the "nu" history over the averaging window
    hist = _cfd(history=_history(nu=3.3, heat_rate=4.4))
    s = P.nusselt_stat(hist)
    assert s.mean == pytest.approx(3.3, abs=0.01) and 0 < s.stderr < 0.01
    q, se, _ = P.heat_rate_stat(hist)
    assert q == pytest.approx(4.4, abs=0.01) and 0 < se < 0.01
    assert P.series_stat(hist, "missing") is None
    # a Stat without an error bar takes the measured one
    nan_se = _cfd(nusselt=Stat(3.3, 0.02, 3.2, 3.4, float("nan")), history=_history(nu=3.3))
    assert math.isfinite(P.nusselt_stat(nan_se).stderr)


def test_fan_entries_read_every_spelling_and_measure_errors():
    r = _cfd(fans_report=[{"name": "A", "mode": "thrust", "thrust": 2.0, "flow": 1.5, "u_disc": 5.0,
                           "power": 11.0, "dp": 6.6},
                          {"thrust_n": 1.0, "volume_flow": 0.7, "induced_velocity": 2.0, "ideal_power": 3.0},
                          "not a dict"],
             history=_history(fan0_thrust=2.0))
    fans = P.fan_entries(r)
    assert len(fans) == 2
    a, b = fans
    assert (a["name"], a["thrust"], a["volume_flow"], a["induced_velocity"], a["ideal_power"],
            a["pressure_jump"]) == ("A", 2.0, 1.5, 5.0, 11.0, 6.6)
    assert 0 < a["thrust_stderr"] < 0.01                     # measured from fan0_thrust
    assert b["name"] == "Fan 2" and b["thrust"] == 1.0 and math.isnan(b["thrust_stderr"])
    assert math.isnan(b["pressure_jump"])
    assert P.has_fans(r) and not P.has_fans(_cfd())


def test_momentum_theory():
    rho, area, u = 1.2, 0.5, 10.0
    for ct in (0.1, 0.5, 2.0):
        thrust = ct * 0.5 * rho * u * u * area
        m = P.momentum_theory(thrust, area, rho, u)
        w = m["induced_velocity"]
        assert 2 * rho * area * w * (u + w) == pytest.approx(thrust)          # T = 2 rho A w (U + w)
        assert m["induction"] == pytest.approx(0.5 * (-1 + math.sqrt(1 + ct)))  # C_T = 4 a (1 + a)
        assert m["ideal_power"] == pytest.approx(thrust * (u + w))
        assert m["wake_increment"] == pytest.approx(2 * w)
    static = P.momentum_theory(10.0, 0.5, 1.2, 0.0)
    assert static["induced_velocity"] == pytest.approx(math.sqrt(10.0 / (2 * 1.2 * 0.5)))
    assert math.isnan(P.momentum_theory(-1.0, 0.5, 1.2, 10.0)["induced_velocity"])


def test_heat_balance_terms_and_residual():
    assert P.heat_balance(_cfd()) is None
    r = _cfd(heat_rate=4.0, diagnostics={"heat_in_sources": 6.0, "heat_out_outlet": 9.9, "heat_storage_rate": 0.0})
    bal = P.heat_balance(r)
    assert bal["in"] == {"Bodies": 4.0, "Heat sources": 6.0}             # bodies from result.heat_rate
    assert bal["residual"] == pytest.approx(0.1) and bal["relative"] == pytest.approx(0.01)
    assert bal["reported"] is False and bal["missing"] == []
    own = _cfd(diagnostics={"heat_in_bodies": 1.0, "heat_balance_residual": 1e-4})
    b2 = P.heat_balance(own)
    assert b2["reported"] is True and b2["residual"] == 1e-4
    assert "heat carried out by the flow" in b2["missing"] and "storage in the domain" in b2["missing"]


def test_diagnostics_rows_known_first():
    r = _cfd(diagnostics={"zzz_custom": 3.0, "cfl_max": 0.5, "divergence_rms": 1e-7, "heat_in_bodies": 2.0,
                          "label": "text"})
    keys = [k for k, *_ in P.diagnostics_rows(r)]
    assert keys == ["divergence_rms", "cfl_max", "zzz_custom"]                # heat terms and text excluded


def test_compare_rows_include_thermal_and_fan_metrics():
    base = _cfd("a", nusselt=Stat(3.30, 0.02, 3.2, 3.4, 0.004), heat_rate=4.40,
                fans_report=[{"thrust": 2.0, "thrust_stderr": 0.01, "volume_flow": 1.5, "ideal_power": 11.0}])
    var = _cfd("b", nusselt=Stat(3.60, 0.02, 3.5, 3.7, 0.004), heat_rate=4.80,
               fans_report=[{"thrust": 2.01, "thrust_stderr": 0.01, "volume_flow": 1.6, "ideal_power": 11.5},
                            {"thrust": 0.5, "thrust_stderr": 0.02, "volume_flow": 0.2, "ideal_power": 1.0}])
    rows = {r["key"]: r for r in compare_rows([base, var])}
    nu = rows["nusselt"]["cells"][1]
    assert nu["delta"] == pytest.approx(0.30) and nu["significant"] is True and nu["improved"] is None
    assert rows["heat_rate"]["unit"] == "W/m"
    th = rows["fan_thrust"]["cells"][1]
    assert th["value"] == pytest.approx(2.51) and th["stderr"] == pytest.approx(math.hypot(0.01, 0.02))
    assert rows["fan_power"]["better"] == "lower"
    plain = {r["key"] for r in compare_rows([_cfd("x"), _cfd("y")])}
    assert not {"nusselt", "heat_rate", "fan_thrust", "fan_flow", "fan_power"} & plain      # no empty rows
    r3 = {r["key"]: r for r in compare_rows([CFD3DResult(sim_id="p", heat_rate=10.0),
                                              CFD3DResult(sim_id="q", heat_rate=12.0)])}
    assert r3["heat_rate"]["unit"] == "W" and r3["heat_rate"]["cells"][1]["pct"] == pytest.approx(20.0)


def test_reynolds_warning_only_for_capped_differences():
    same = {"a": FlowConditions(speed=30.0), "b": FlowConditions(speed=30.0)}
    # two cars at their real Reynolds numbers: the lengths differ, which is the design
    a = _cfd("a", re_real=8.0e6, re_sim=8.0e6)
    b = _cfd("b", re_real=8.35e6, re_sim=8.35e6)
    assert not any("Reynolds" in w for w in comparison_warnings([a, b], same))
    # one of them capped: the simulated flows differ for a modelling reason
    c = _cfd("b", re_real=8.35e6, re_sim=2.0e4)
    assert any("different Reynolds numbers" in w for w in comparison_warnings([a, c], same))


def test_readers_understand_the_real_solvers_reports():
    """The fan reports and the heat balance as the 2D and 3D solvers actually write them
    (tiny CPU float64 runs) read back complete: no quantity silently becomes nan because a
    solver spells a key differently (2D "flow_rate", 3D "flow")."""
    pytest.importorskip("torch")
    from aerolume.solvers.ns2d import FanSpec, FlowSolver, SolverConfig
    from aerolume.solvers.ns3d import FlowSolver3D, ImmersedMesh, SolverConfig3D
    from aerolume.solvers.ns3d.config import Fan3D, ThermalConfig3D

    s2 = FlowSolver(SolverConfig(nx=48, ny=24, h=1 / 8, x0=-2.0, y0=-1.5, re=50.0, device="cpu", dtype="float64",
                                 dye=False, perturb=False, fans=[FanSpec((0.0, 0.0), width=1.0, value=0.25,
                                                                        name="strip")]))
    s2.advance(20)
    r2 = _cfd(fans_report=s2.fans_report())
    f2 = P.fan_entries(r2)[0]
    for key in ("thrust", "volume_flow", "induced_velocity", "ideal_power", "pressure_jump"):
        assert math.isfinite(f2[key]), ("2D", key)
    assert f2["name"] == "strip" and f2["mode"] == "thrust"
    assert f2["thrust"] == pytest.approx(0.25, rel=1e-6)          # thrust mode delivers its set value
    theta = np.linspace(0, np.pi, 7)
    ring = np.stack([np.cos(theta), np.sin(theta)], 1)
    verts = [(0.0, 0.0, -0.3), (0.0, 0.0, 0.3)] + [(0.3 * x, 0.3 * y, 0.0) for x, y in ring] + \
            [(0.3 * x, -0.3 * y, 0.0) for x, y in ring]
    v = np.asarray(verts, float)
    n = len(ring)
    faces = [(0, 2 + i, 3 + i) for i in range(n - 1)] + [(1, 3 + i, 2 + i) for i in range(n - 1)] + \
            [(0, 3 + n + i, 2 + n + i) for i in range(n - 1)] + [(1, 2 + n + i, 3 + n + i) for i in range(n - 1)]
    body = ImmersedMesh(v + np.array([1.2, 0.0, 0.0]), np.asarray(faces, np.int64), name="hot",
                        thermal="temperature", thermal_value=1.0)
    cfg3 = SolverConfig3D(nx=32, ny=12, nz=12, h=1 / 4, x0=-2.0, y0=-1.5, z0=-1.5, re=40.0, device="cpu",
                          dtype="float64", dye=False, perturb=False, bodies=[body], diagnostics=True,
                          thermal=ThermalConfig3D(prandtl=0.71, richardson=0.0),
                          fans=[Fan3D(center=(0.0, 0.0, 0.0), diameter=1.0, value=0.1, name="disc")])
    s3 = FlowSolver3D(cfg3)
    s3.advance(12)
    r3 = CFD3DResult(sim_id="t", fans_report=s3.fan_report(), diagnostics=dict(s3.heat_balance()))
    f3 = P.fan_entries(r3)[0]
    for key in ("thrust", "volume_flow", "induced_velocity", "ideal_power", "pressure_jump"):
        assert math.isfinite(f3[key]), ("3D", key)
    bal = P.heat_balance(r3)
    assert bal is not None and bal["reported"] is True
    assert set(bal["in"]) >= {"Bodies", "Domain walls (net)", "Heat sources", "Inlet (enthalpy in)"}
    assert set(bal["out"]) >= {"Outlet (enthalpy out)"}
    assert bal["in"]["Bodies"] > 0                                 # the hot body heats the fluid


def _room(sid: str, *, q_set: float = 300.0, t_max: float = 41.2, vent: float | None = 120.0) -> CFDResult:
    """A heater-only room as the runner reports it: no forces, no Nusselt number, heat rate = heater power."""
    diag = {"heat_sources_set": q_set, "heat_in_sources": q_set * (1 + 1e-7), "t_fluid_max": t_max,
            "t_fluid_mean": 24.5}
    if vent is not None:
        diag.update(heat_out_ports=vent, heat_in_ports=0.0)
    return CFDResult(sim_id=sid, re_real=3e4, re_sim=3e4, heat_rate=q_set * (1 + 1e-7), diagnostics=diag)


def _room_sim(sid: str, *, closed: bool = True, speed: float = 0.9, power: float = 300.0, fan: bool = True):
    from aerolume.model import CFDSettings, Fan, HeatSource, Simulation, ThermalSettings
    return Simulation(name=sid, design_id="d", id=sid, conditions=FlowConditions(speed=speed),
                      cfd=CFDSettings(domain_mode="closed" if closed else "tunnel"),
                      thermal=ThermalSettings(enabled=True, buoyancy=True),
                      heat_sources=[HeatSource("Heater", "rect", (1.0, 0.3), (0.3, 0.2), power)],
                      fans=[Fan("Fan", (1.0, 1.5), (1.0, 0.0), diameter=0.3, mode="thrust", value=1.0)] if fan else [])


def test_heater_power_is_an_input_not_a_measured_heat_rate():
    room = _room("r")
    assert P.heat_rate_is_input(room) and P.heater_power_set(room) == 300.0
    assert P.vent_heat_out(room) == pytest.approx(120.0)
    assert math.isnan(P.vent_heat_out(_room("n", vent=None)))
    # a heated body has a Nusselt number: its heat rate is measured even with a heater in the run
    body = _cfd("b", nusselt=Stat(3.3, 0.01, 3.2, 3.4, 0.004), heat_rate=300.0,
                diagnostics={"heat_sources_set": 300.0})
    assert not P.heat_rate_is_input(body)
    # walls that also add heat: the heat put in is no longer the setting alone
    assert not P.heat_rate_is_input(CFDResult(sim_id="w", heat_rate=450.0, diagnostics={"heat_sources_set": 300.0}))
    assert not P.heat_rate_is_input(_cfd("plain"))


def test_compare_rows_for_rooms_lead_with_temperatures_and_mark_the_setting():
    rows = {r["key"]: r for r in compare_rows([_room("a"), _room("b", t_max=52.7)])}
    assert "heat_rate" not in rows                                       # no "300.0 vs 300.0, −0.0 %"
    hp = rows["heater_power"]
    assert hp["input"] is True and hp["label"] == "Heater power (set)" and hp["unit"] == "W/m"
    cell = hp["cells"][1]
    assert cell["pct"] == 0.0 and cell["significant"] is None and cell["improved"] is None
    t = rows["t_fluid_max"]["cells"][1]
    assert t["delta"] == pytest.approx(11.5) and t["pct"] is None and rows["t_fluid_max"]["unit"] == "°C"
    assert rows["vent_heat_out"]["cells"][0]["value"] == pytest.approx(120.0)
    keys = [r["key"] for r in compare_rows([_room("a"), _room("b")])]
    assert keys.index("t_fluid_max") < keys.index("heater_power") < keys.index("re_real")
    # a heated body against a room: the body's measured rate and the room's setting never share a row
    body = _cfd("c", nusselt=Stat(3.3, 0.01, 3.2, 3.4, 0.004), heat_rate=2.7)
    mixed = {r["key"]: r for r in compare_rows([body, _room("a")])}
    assert mixed["heat_rate"]["cells"][1]["value"] is None and mixed["heat_rate"]["cells"][1]["pct"] is None
    assert mixed["heater_power"]["cells"][0]["value"] is None


def test_closed_rooms_are_compared_by_their_inputs_not_their_speeds():
    a, b = _room("a"), _room("b")
    a.re_sim, b.re_sim = 3.1e4, 7.9e4                     # Reynolds numbers of two reference speeds
    # FlowConditions only (older callers): the speed rule as before
    conds = {"a": FlowConditions(speed=0.9), "b": FlowConditions(speed=0.3)}
    assert any("different speeds" in w for w in comparison_warnings([a, b], conds))
    # the simulations: rooms have no free stream, so no speed or Reynolds warning ...
    same = {"a": _room_sim("a", speed=0.9), "b": _room_sim("b", speed=0.3)}
    assert comparison_warnings([a, b], same) == []
    # ... but different driving inputs are named
    other = {"a": _room_sim("a", speed=0.9), "b": _room_sim("b", speed=0.3, power=150.0, fan=False)}
    (msg,) = comparison_warnings([a, b], other)
    assert "driven by different inputs: heater power (150 and 300 W/m) and fans" in msg
    # a room next to a wind-tunnel run gets its own sentence, not a speed warning
    mix = {"a": _room_sim("a"), "b": _room_sim("b", closed=False, speed=10.0)}
    (msg,) = comparison_warnings([a, b], mix)
    assert "1 closed-domain run" in msg and "1 wind-tunnel run" in msg and "different speeds" not in msg
