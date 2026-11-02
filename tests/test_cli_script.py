"""``aerolume-cli script FILE [--project P [--create]] [-- args]`` (DESIGN.md sections 5.9 and 8)."""
from __future__ import annotations

from pathlib import Path

import pytest

from aerolume import cli
from aerolume.project import Project as Store

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "scripts"


@pytest.fixture(autouse=True)
def private_config(tmp_path, monkeypatch):
    """The script command feeds the run-time calibration like `run`: keep it in a temporary folder."""
    from aerolume.simulation import plan as planmod
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "config"))
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


def _script(tmp_path: Path, text: str, name: str = "study.py") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_split_script_args():
    assert cli.split_script_args(["script", "a.py", "--project", "P", "--", "x", "--project"]) == \
        (["script", "a.py", "--project", "P"], ["x", "--project"])
    assert cli.split_script_args(["script", "a.py"]) == (["script", "a.py"], [])


def test_script_runs_with_project_args_log_and_show(tmp_path, capsys):
    script = _script(tmp_path, """
import sys
print("args", args, sys.argv[1:])
d = project.add_naca("2412")
w = project.add_wing(span=2, root_chord=0.3, tip_chord=0.18)
r = project.run_wing(w, speed=20, alpha=(0, 4, 2), viscous=False)
print(r)
log("note to the log")
log("careful", "warning")
show(r.plot("span"))
print(al.__name__, __name__)
""")
    rc = cli.main(["script", str(script), "--project", str(tmp_path / "CLI study"), "--create",
                   "--figures", str(tmp_path / "figs"), "--", "0.3", "--fast"])
    out, err = capsys.readouterr()
    assert rc == 0, err
    assert "args ['0.3', '--fast'] ['0.3', '--fast']" in out
    assert "Wing polar · 20 m/s · inviscid" in out and "note to the log" in out and "Warning: careful" in err
    assert "aerolume __main__" in out and (tmp_path / "figs" / "study-figure-1.png").is_file()
    store = Store.open(tmp_path / "CLI study.aerolume")
    assert [d.name for d in store.designs.values()] == ["NACA 2412", "Wing · span 2 m, taper 0.6"]
    (sim,) = store.simulations.values()
    assert sim.status == "done" and store.has_result(sim.id)


def test_script_errors_show_the_script_line_not_the_internals(tmp_path, capsys):
    script = _script(tmp_path, "x = 1\nproject.run_cfd('nope')\n")
    rc = cli.main(["script", str(script), "--project", str(tmp_path / "P"), "--create"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "line 2, in <module>" in err and "NotFoundError: No design 'nope' in this project" in err
    assert "inside Aerolume or other libraries not shown" in err and "api.py" not in err


def test_script_exit_codes(tmp_path, capsys):
    assert cli.main(["script", str(tmp_path / "missing.py")]) == 2
    assert "No script at" in capsys.readouterr().err
    assert cli.main(["script", str(_script(tmp_path, "raise SystemExit(3)\n"))]) == 3
    assert cli.main(["script", str(_script(tmp_path, "import sys\nsys.exit()\n"))]) == 0
    assert cli.main(["script", str(_script(tmp_path, "def broken(:\n"))]) == 1
    assert "SyntaxError" in capsys.readouterr().err
    assert cli.main(["script", str(_script(tmp_path, "print(project)\n")), "--project",
                     str(tmp_path / "absent")]) == 1
    assert "--create" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["doctor", "--", "x"])               # arguments after -- belong to scripts only


def test_script_without_a_project_and_sibling_imports(tmp_path, capsys):
    (tmp_path / "helpers.py").write_text("VALUE = 42\n", encoding="utf-8")
    script = _script(tmp_path, "from helpers import VALUE\nprint(project, VALUE)\n")
    assert cli.main(["script", str(script)]) == 0
    assert "None 42" in capsys.readouterr().out


def test_examples_run_from_the_command_line(tmp_path, capsys, monkeypatch):
    """The wing study example end to end (vortex lattice, seconds): the examples really run."""
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["script", str(EXAMPLES / "wing_taper.py"), "--project", str(tmp_path / "Examples"), "--create"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "oswald" in out and "Highest span efficiency" in out and (tmp_path / "wing_taper-figure-1.png").is_file()
    assert len(Store.open(tmp_path / "Examples.aerolume").simulations) == 5
