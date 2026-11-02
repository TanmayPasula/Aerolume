"""aerolume-cli: doctor, bench, polar, quick and run on a temporary project.

The CFD commands run tiny CPU grids (``--cells-per-ref 12``, a few convective
times) so the whole module stays within a few seconds.
"""
from __future__ import annotations

import csv
import signal
import subprocess
import sys

import pytest

from aerolume import cli
from aerolume.core import fluid as F
from aerolume.geometry import design_from_airfoil, naca
from aerolume.model import AirfoilSettings, CFDSettings, FlowConditions, Simulation
from aerolume.project import Project

AIR = F.air_isa(0.0)


def speed_for(re: float, length: float = 1.0) -> float:
    return re * AIR.viscosity / (AIR.density * length)


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    """Solver commands persist the run-time calibration in the user config folder:
    point that at a temporary folder and start every test from the reference model."""
    from aerolume.simulation import plan as planmod
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AEROLUME_CALIBRATION", raising=False)
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)
    return tmp_path / "config"


def test_help_does_not_import_torch():
    code = "import sys, aerolume.cli; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0 and out.stdout.strip() == "False"
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_parse_alpha_range():
    assert cli.parse_alpha_range("-4:14:1") == (-4.0, 14.0, 1.0)
    assert cli.parse_alpha_range("4") == (4.0, 4.0, 1.0)
    assert cli.parse_alpha_range("0:10") == (0.0, 10.0, 1.0)
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_alpha_range("0:10:-1")
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_alpha_range("a:b")


def test_negative_option_values_are_accepted_as_written_in_the_docs():
    """`--alpha -4:14:1` used to fail with "expected one argument" (argparse took the
    value for an option); the documented sweep must parse as written."""
    assert cli.attach_negative_values(["polar", "--naca", "2412", "--alpha", "-4:14:1", "--re", "1e6"]) == \
        ["polar", "--naca", "2412", "--alpha=-4:14:1", "--re", "1e6"]
    assert cli.attach_negative_values(["quick", "--alpha", "-2.5", "--plan-only"]) == \
        ["quick", "--alpha=-2.5", "--plan-only"]
    assert cli.attach_negative_values(["quick", "--plan-only", "-v"]) == ["quick", "--plan-only", "-v"]
    args = cli.build_parser().parse_args(cli.attach_negative_values(
        ["polar", "--naca", "2412", "--re", "1e6", "--alpha", "-4:14:2"]))
    assert args.alpha == (-4.0, 14.0, 2.0)


def test_table_aligns_columns():
    text = cli.table(["name", "value"], [("a", "1.5"), ("long name", "10")])
    lines = text.splitlines()
    assert len({len(line) for line in lines[:2]}) == 1 and lines[3].endswith("10")


def test_doctor_without_the_solver_check(capsys):
    assert cli.main(["doctor", "--no-solver"]) == 0
    out = capsys.readouterr().out
    for fragment in ("Packages", "torch", "Devices", "cpu", "All checks passed."):
        assert fragment in out


def test_bench_quick_on_the_cpu(capsys):
    assert cli.main(["bench", "--device", "cpu", "--quick"]) == 0
    out = capsys.readouterr().out
    assert "steps/s" in out and "256 × 128" in out


def test_polar_prints_a_table_and_writes_csv(tmp_path, capsys):
    path = tmp_path / "polar.csv"
    assert cli.main(["polar", "--naca", "0012", "--re", "1e6", "--alpha", "0:4:2", "--csv", str(path)]) == 0
    out = capsys.readouterr().out
    assert "NACA 0012: Re = 1.00×10^6" in out and "Best L/D" in out and "Wrote" in out
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert sum(1 for r in rows if r and r[0].strip().lstrip("-").replace(".", "").isdigit()) == 3


def test_quick_plan_only_shows_the_plan(capsys):
    assert cli.main(["quick", "--template", "sedan", "--plan-only", "--device", "cpu", "-v"]) == 0
    out = capsys.readouterr().out
    for fragment in ("Sedan at 30", "Blockage", "Grid", "moving belt", "· Domain:"):
        assert fragment in out


def test_quick_caps_airfoil_reynolds_numbers_unless_told_otherwise(capsys):
    """--re-mode auto (the default) makes the same choice a new simulation gets in the app."""
    base = ["quick", "--naca", "2412", "--speed", "15", "--alpha", "4", "--plan-only", "--device", "cpu"]
    assert cli.main(base) == 0
    assert "(capped)" in capsys.readouterr().out
    assert cli.main(base + ["--re-mode", "real"]) == 0
    out = capsys.readouterr().out
    assert "(capped)" not in out and "Lift and drag are not reliable" in out
    assert cli.main(["quick", "--template", "sedan", "--plan-only", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "(capped)" not in out and "Underbody gap under-resolved" in out


def test_quick_rejects_unknown_templates_and_bad_grids(capsys):
    assert cli.main(["quick", "--template", "no_such_shape", "--plan-only"]) == 2
    assert cli.main(["quick", "--naca", "0012", "--cells-per-ref", "2", "--plan-only"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["quick", "--naca", "0012", "--speed", "-3"])


def test_quick_runs_and_saves_a_project(tmp_path, capsys):
    target = tmp_path / "quick"
    rc = cli.main(["quick", "--naca", "0012", "--speed", f"{speed_for(200.0):.6g}", "--alpha", "4",
                   "--device", "cpu", "--cells-per-ref", "12", "--duration", "2", "--save", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Cd" in out and "Converged" in out and "Saved project" in out
    proj = Project.open(target.with_name(target.name + ".aerolume"))
    (sim,) = proj.simulations.values()
    assert sim.status == "done" and sim.finished and proj.has_result(sim.id)
    res = proj.load_result(sim.id)
    assert res.grid[0] > 0 and res.history["t"].size > 10


def test_bench_feeds_the_persisted_run_time_calibration(private_config, monkeypatch, capsys):
    from aerolume.simulation import plan as planmod
    assert cli.main(["bench", "--device", "cpu", "--quick"]) == 0
    assert (private_config / planmod.CALIBRATION_FILE).is_file()
    assert planmod._CALIBRATION_FILE is None                    # only while the command ran
    (private_config / planmod.CALIBRATION_FILE).unlink()
    monkeypatch.setenv("AEROLUME_CALIBRATION", "off")
    assert cli.main(["bench", "--device", "cpu", "--quick"]) == 0
    assert not (private_config / planmod.CALIBRATION_FILE).exists()


def test_the_cli_plan_shows_the_headless_estimate(capsys):
    """The plan's own estimate includes the app's live view; the CLI renders no frames."""
    from types import SimpleNamespace

    from aerolume.simulation.plan import fmt_duration
    plan = SimpleNamespace(summary=lambda: {"Grid": "10 × 10", "Estimated run time": fmt_duration(300.0)},
                           est_seconds_headless=240.0, notes=[], warnings=[])
    cli._print_plan(plan, verbose=False)
    out = capsys.readouterr().out
    assert fmt_duration(240.0) in out and fmt_duration(300.0) not in out


def _project(tmp_path) -> Project:
    proj = Project.create(tmp_path / "Bench", "Bench")
    d = proj.add_design(design_from_airfoil(naca("0012", 160, 1.0), "NACA 0012", 1.0))
    proj.add_simulation(Simulation("tiny cfd", d.id, conditions=FlowConditions(AIR, speed_for(200.0), 4.0),
                                   cfd=CFDSettings(quality="draft", cells_per_ref=12, duration=2.0, device="cpu")))
    proj.add_simulation(Simulation("polar", d.id, analysis="polar",
                                   conditions=FlowConditions(AIR, speed_for(1e6), 0.0),
                                   airfoil=AirfoilSettings(alpha_start=0.0, alpha_end=2.0, alpha_step=2.0)))
    proj.save()
    return proj


def test_run_stores_results_in_the_project(tmp_path, capsys):
    proj = _project(tmp_path)
    assert cli.main(["run", str(proj.path), "tiny cfd"]) == 0
    assert cli.main(["run", str(proj.path), "polar"]) == 0
    out = capsys.readouterr().out
    assert "Result saved in" in out and "alpha°" in out
    again = Project.open(proj.path)
    for sim in again.simulations.values():
        assert sim.status == "done" and again.has_result(sim.id) and sim.finished


def test_run_reports_unknown_simulations(tmp_path, capsys):
    proj = _project(tmp_path)
    assert cli.main(["run", str(proj.path), "nope"]) == 1
    err = capsys.readouterr().err
    assert "no simulation 'nope'" in err and "tiny cfd" in err and "polar" in err


def test_run_records_failures_on_the_simulation(tmp_path, capsys):
    proj = _project(tmp_path)
    sim = next(s for s in proj.simulations.values() if s.name == "tiny cfd")
    sim.conditions.speed = 0.0
    proj.update_simulation(sim)
    proj.save()
    assert cli.main(["run", str(proj.path), "tiny cfd"]) == 1
    assert "Failed" in capsys.readouterr().err
    again = Project.open(proj.path)
    s = again.simulations[sim.id]
    assert s.status == "failed" and "speed" in (s.error or "")


def test_ctrl_c_stops_softly_then_aborts():
    old = signal.getsignal(signal.SIGINT)
    stop = cli.StopOnInterrupt()
    with stop as event:
        assert signal.getsignal(signal.SIGINT) == stop._handler
        stop._handler(signal.SIGINT, None)
        assert event.is_set()
        with pytest.raises(KeyboardInterrupt):
            stop._handler(signal.SIGINT, None)
    assert signal.getsignal(signal.SIGINT) == old


def test_validate_accepts_quick_and_full_but_not_both():
    """``validate --quick`` names the default subset (validation/run_all.py spells it that way); it was
    refused as an unknown argument, so the documented command failed before running anything."""
    p = cli.build_parser()
    assert not p.parse_args(["validate", "--quick"]).full
    assert p.parse_args(["validate", "--full"]).full
    assert not p.parse_args(["validate"]).full
    with pytest.raises(SystemExit):
        p.parse_args(["validate", "--quick", "--full"])
