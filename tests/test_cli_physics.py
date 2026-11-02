"""aerolume-cli and placed physics (DESIGN.md 10): --heated, --fan, --closed, and the cavity command.

Plans are printed with ``--plan-only`` (no solver); the runs use tiny CPU grids, a few seconds
each. The checks are on what the user reads: the physics rows of the plan, the Nusselt number
and its reference, the delivered fan thrust, and clean errors (exit 2) for malformed options.
"""
from __future__ import annotations

import argparse
import math
import re

import pytest

from aerolume import cli
from aerolume.core import fluid as F

AIR = F.air_isa(0.0)


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    from aerolume.simulation import plan as planmod
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AEROLUME_CALIBRATION", raising=False)
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def test_parse_fan_reads_position_size_value_mode_and_direction():
    f = cli.parse_fan("1.5,0.4:0.3:10")
    assert f == {"position": (1.5, 0.4), "diameter": 0.3, "value": 10.0, "mode": "thrust", "direction": (1.0, 0.0)}
    g = cli.parse_fan("0.4,0,0.3:0.2:3:velocity:0,0,1")
    assert g["mode"] == "velocity" and g["direction"] == (0.0, 0.0, 1.0) and len(g["position"]) == 3
    assert cli.parse_fan("0,0:0.5:20:pressure")["mode"] == "pressure_jump"
    for bad in ("1,2", "1,2:0:5", "1,2:0.3:x", "1,2:0.3:5:spin", "1,2:0.3:5:thrust:0,0", "1:0.3:5",
                "1,2:0.3:5:thrust:1,0,0"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_fan(bad)


def test_parse_box_needs_min_below_max_on_every_axis():
    assert cli.parse_box("0,4,0,3") == (0.0, 4.0, 0.0, 3.0)
    assert cli.parse_box("-1,1,-2,2,0,3") == (-1.0, 1.0, -2.0, 2.0, 0.0, 3.0)
    for bad in ("0,4,0", "4,0,0,3", "0,4,0,3,1,1", "a,b,c,d"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.parse_box(bad)


def test_quick_plan_with_a_heated_body_and_a_fan(capsys):
    rc = cli.main(["quick", "--template", "cylinder", "--param", "diameter=0.05", "--speed", "0.5", "--heated", "40",
                   "--fan", "0.15,0:0.03:0.02", "--device", "cpu", "--plan-only"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bodies at 40 °C in 15 °C air, buoyancy on" in out and "Fan 1 0.0200 N/m" in out
    for row in ("Temperature scale", "25.0 K", "Prandtl number", "Péclet number", "Convection", "Fans"):
        assert row in out


def test_quick_rejects_a_fan_of_the_wrong_dimension_and_bad_values(capsys):
    rc = cli.main(["quick", "--template", "cylinder", "--fan", "0.5,0,0:0.1:1", "--device", "cpu", "--plan-only"])
    assert rc == 1 and "needs 2 coordinates" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main(["quick", "--template", "cylinder", "--fan", "0.5,0:-1:1", "--plan-only"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["quick", "--template", "cylinder", "--heated", "-300", "--plan-only"])
    assert exc.value.code == 2


def test_quick_in_a_closed_room(capsys):
    rc = cli.main(["quick", "--template", "cylinder", "--param", "diameter=0.2", "--closed", "-1,3,-1,1",
                   "--fan", "-0.5,0:0.4:0.5", "--device", "cpu", "--plan-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "closed box 4.00 × 2.00 m" in out and "Closed box (walls on every side)" in out
    assert "momentum theory" in out and "Blockage" not in out


def test_quick3d_plan_with_a_heated_sphere(capsys):
    rc = cli.main(["quick3d", "--sphere", "0.02", "--speed", "0.04", "--heated", "25", "--no-buoyancy",
                   "--device", "cpu", "--plan-only"])
    out = capsys.readouterr().out
    assert rc == 0 and "bodies at 25 °C in 15 °C air" in out and "buoyancy" not in out.split("\n")[1]
    assert "Temperature scale" in out and "Forced (no buoyancy)" in out


def _reference(ra: float, dim: int = 2) -> float:
    """The case's reference Nusselt number (the API owns the table: de Vahl Davis 1983 in 2D, whose
    hot-wall Nu0 and mean Nu differ by under 0.5 %; Tric et al. 2000 in 3D)."""
    from aerolume.api import cavity_case
    return float(cavity_case(ra=ra, dimension=dim).reference["value"])


def test_cavity_plan_names_its_reference(capsys):
    ref = _reference(1e5)
    assert ref == pytest.approx(4.519, rel=5e-3)                       # de Vahl Davis (1983), Ra 10^5
    assert cli.main(["cavity", "--ra", "1e5", "--device", "cpu", "--plan-only"]) == 0
    out = capsys.readouterr().out
    assert f"Reference: Nu = {ref:g} (de Vahl Davis (1983)" in out and "Rayleigh number" in out
    assert cli.main(["cavity", "--ra", "1e5", "--3d", "--device", "cpu", "--plan-only"]) == 0
    assert f"Nu = {_reference(1e5, 3):g} (Tric, Labrosse & Betrouni (2000)" in capsys.readouterr().out
    assert cli.main(["cavity", "--ra", "5e3", "--device", "cpu", "--plan-only"]) == 0
    assert "none published at these parameters" in capsys.readouterr().out


def test_cavity_runs_and_compares_with_de_vahl_davis(capsys):
    rc = cli.main(["cavity", "--ra", "1e3", "--cells-per-ref", "16", "--duration", "6", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0
    nu = float(re.search(r"Nusselt number \(hot wall\)\s+([0-9.]+)", out).group(1))
    ref = _reference(1e3)
    assert ref == pytest.approx(1.118, rel=5e-3)
    assert 0.9 < nu < 1.4 and f"Reference (de Vahl Davis (1983))  {ref:g}" in out
    diff = float(re.search(r"Difference\s+([+-][0-9.]+) %", out).group(1))
    assert diff == pytest.approx(100 * (nu - ref) / ref, abs=0.2)
    assert "Heat balance residual" in out and "Divergence (rms / max)" in out


def test_quick_with_a_heated_cylinder_prints_nusselt_and_heat_rate(capsys):
    speed = 40 * AIR.kinematic_viscosity / 0.01
    rc = cli.main(["quick", "--template", "cylinder", "--param", "diameter=0.01", "--speed", f"{speed:.6f}",
                   "--heated", "25", "--no-buoyancy", "--cells-per-ref", "8", "--duration", "2", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0
    nu = float(re.search(r"Nusselt number\s+([0-9.]+)", out).group(1))
    q = float(re.search(r"Heat rate\s+([0-9.]+) W per metre of span", out).group(1))
    k = 0.0241 * (288.15 / 273.15) ** 1.5 * 467.15 / 482.15
    assert math.isfinite(nu) and q == pytest.approx(nu * k * 10.0 * math.pi, rel=5e-2)    # Q = Nu k dT pi


def test_quick_with_a_fan_in_a_room_prints_the_delivered_thrust(capsys):
    rc = cli.main(["quick", "--template", "cylinder", "--param", "diameter=0.2", "--closed", "-1,3,-1,1",
                   "--fan", "-0.5,0:0.4:0.5", "--cells-per-ref", "6", "--duration", "4", "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0
    row = next(ln for ln in out.splitlines() if ln.startswith("Fan 1"))
    assert float(row.split()[2]) == pytest.approx(0.5, rel=1e-3)          # thrust N/m, as set
    assert "fan  " in out and "flow m²/s" in out


def test_heat_balance_share_is_of_the_heat_through_the_domain():
    """A wall-heated cavity: 0.102 W/m in through the hot wall and out through the cold one, so the walls
    net to ~0. Against the largest single term the 4e-9 round-off residual read "118 %" (release gate 3,
    2 November 2026); it is a share of the heat *through* the domain, as on the Results page."""
    from aerolume.analysis.physics import heat_balance
    diag = {"heat_in_walls": 3.5e-9, "heat_in_bodies": 0.0, "heat_in_sources": 0.0, "heat_out_ports": 0.0,
            "heat_storage_rate": -6e-10, "heat_balance_residual": 4.1e-9}
    res = argparse.Namespace(diagnostics=diag, heat_rate=0.1022)
    text = dict(cli._diag_rows(diag, res, dim=2))["Heat balance residual"]
    assert "%" not in text and "through the domain" in text and "0.1022 W/m" in text
    rel = float(re.search(r"\(([-0-9.e+]+) of the", text).group(1))
    assert rel == pytest.approx(heat_balance(res)["relative"], rel=0.05)
    assert rel == pytest.approx(4.1e-9 / 0.1022, rel=0.05)
    # a balance that does not close reads in per cent of the same throughput
    bad = dict(diag, heat_balance_residual=0.0051)
    text = dict(cli._diag_rows(bad, argparse.Namespace(diagnostics=bad, heat_rate=0.1022)))["Heat balance residual"]
    assert "(5.0 % of the 0.1022 W/m through the domain)" in text, text
    # without a result the share still comes from the reported terms, never from the largest single one
    text = dict(cli._diag_rows({"heat_in_bodies": 2.0, "heat_out_outlet": 2.0, "heat_balance_residual": 1e-3}))
    assert "(0.050 % of the 2.000 W/m through the domain)" in text["Heat balance residual"]
