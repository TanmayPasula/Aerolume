"""aerolume-cli 3D and wing commands: quick3d, wing, run on 3D / wing simulations, doctor's VTK report.

The CFD runs are tiny CPU grids (a few convective times at 6 cells per
reference length), the wing runs inviscid, so the module takes seconds.
"""
from __future__ import annotations

import csv

import pytest

from aerolume import cli
from aerolume.core import fluid as F
from aerolume.geometry3d import box, cylinder, write_obj
from aerolume.model import CFD3DResult, WingResult
from aerolume.project import Project

AIR = F.air_isa(0.0)


def speed_for(re: float, length: float = 1.0) -> float:
    return re * AIR.viscosity / (AIR.density * length)


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    """Solver commands persist the run-time calibration: keep it in a temporary folder."""
    from aerolume.simulation import plan as planmod
    from aerolume.simulation import wing as wingmod
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("AEROLUME_CALIBRATION", raising=False)
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE", "_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)
    wingmod.clear_polar_cache()
    return tmp_path / "config"


@pytest.fixture
def car_stl(tmp_path):
    """A box car with four wheels as OBJ groups (binary STL has no parts), in millimetres (the
    importer guesses the units)."""
    body = box(4500.0, 1800.0, 1200.0, center=(2250.0, 0.0, 800.0), name="body")
    wheels = [cylinder(640.0, 250.0, axis="y", center=(x, y, 320.0), name=f"wheel {i}")
              for i, (x, y) in enumerate([(900, 750), (900, -750), (3600, 750), (3600, -750)])]
    path = tmp_path / "boxcar.obj"
    write_obj([body] + wheels, path)
    return path


def test_help_lists_the_3d_commands(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "quick3d" in out and "wing" in out and "3D" in out


def test_quick3d_plan_only_for_a_sphere(capsys):
    assert cli.main(["quick3d", "--sphere", "1", "--speed", f"{speed_for(100.0):.6g}", "--device", "cpu",
                     "--plan-only"]) == 0
    out = capsys.readouterr().out
    for fragment in ("Sphere 1 m", "Symmetry plane", "y = 0 (half domain simulated)", "Cells (3D)", "Laminar",
                     "Estimated run time"):
        assert fragment in out, fragment


def test_quick3d_plan_only_for_an_imported_car(car_stl, capsys):
    assert cli.main(["quick3d", "--stl", str(car_stl), "--speed", "30", "--device", "cpu", "--plan-only",
                     "-v"]) == 0
    out = capsys.readouterr().out
    assert "moving belt" in out and "LES (WALE" in out and "5 parts" in out
    assert "length 4.500 m" in out                                 # millimetres recognised
    assert "Wheels roll with the belt" in out
    assert cli.main(["quick3d", "--stl", str(car_stl), "--speed", "30", "--yaw", "5", "--device", "cpu",
                     "--plan-only"]) == 0
    assert "none (full domain)" in capsys.readouterr().out


def test_quick3d_rejects_bad_inputs(tmp_path, capsys):
    assert cli.main(["quick3d", "--stl", str(tmp_path / "missing.stl"), "--plan-only"]) == 1
    assert "Cannot read the geometry" in capsys.readouterr().err
    assert cli.main(["quick3d", "--sphere", "1", "--cells-per-ref", "2", "--plan-only"]) == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["quick3d", "--plan-only"])                        # neither --stl nor --sphere
    assert exc.value.code == 2


def test_quick3d_runs_saves_and_run_reruns_it(tmp_path, capsys):
    target = tmp_path / "Ball.aerolume"
    args = ["quick3d", "--sphere", "1", "--speed", f"{speed_for(50.0):.6g}", "--device", "cpu", "--cells-per-ref",
            "6", "--duration", "1", "--save", str(target)]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "CD (drag)" in out and "half domain, symmetry plane" in out and "Saved project" in out
    proj = Project.open(target)
    (sim,) = proj.simulations.values()
    assert sim.status == "done"
    res = proj.load_result(sim.id)
    assert isinstance(res, CFD3DResult) and res.surface and res.slices
    assert cli.main(["run", str(target), sim.id, "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "Cells (3D)" in out and "CD (drag)" in out and "Result saved" in out


def test_wing_command_prints_the_polar_and_writes_csv(tmp_path, capsys):
    path = tmp_path / "wing.csv"
    assert cli.main(["wing", "--naca", "2412", "--span", "8", "--chord", "1", "--taper", "0.5", "--sweep", "5",
                     "--alpha", "-2:4:2", "--inviscid", "--csv", str(path)]) == 0
    out = capsys.readouterr().out
    for fragment in ("CL", "CDi", "Planform", "AR", "Lift slope", "vortex lattice, inviscid", "Wrote"):
        assert fragment in out, fragment
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [float(r["alpha"]) for r in rows] == [-2.0, 0.0, 2.0, 4.0]
    assert float(rows[-1]["cl"]) > float(rows[0]["cl"])


def test_wing_command_rejects_bad_sections(capsys):
    assert cli.main(["wing", "--naca", "abcd", "--span", "8", "--chord", "1", "--inviscid"]) == 2
    assert "NACA" in capsys.readouterr().err
    assert cli.main(["wing", "--naca", "2412", "--span", "8", "--chord", "1", "--taper", "0", "--inviscid"]) == 2


def test_wing_saved_project_runs_with_run(tmp_path, capsys):
    target = tmp_path / "Wing.aerolume"
    assert cli.main(["wing", "--naca", "0012", "--span", "6", "--chord", "1", "--alpha", "0:4:2", "--method", "llt",
                     "--inviscid", "--save", str(target)]) == 0
    capsys.readouterr()
    proj = Project.open(target)
    (sim,) = proj.simulations.values()
    assert sim.analysis == "wing" and isinstance(proj.load_result(sim.id), WingResult)
    assert cli.main(["run", str(target), sim.id]) == 0
    out = capsys.readouterr().out
    assert "(wing)" in out and "CDi" in out and "nonlinear lifting line" in out


def test_doctor_reports_vtk(capsys):
    assert cli.main(["doctor", "--no-solver"]) == 0
    out = capsys.readouterr().out
    assert "VTK (3D viewer)" in out and "3D viewer" in out and "All checks passed." in out
