"""Provenance records, reproduction scripts and precise temperature storage (project/).

A result must say what produced it (software, settings as run, hashes, device, seed)
and a reviewer must be able to recompute it from one script; both are checked here
against a real store and, for the script, a real (fast, inviscid) wing analysis
run end to end through the scripting API.
"""
from __future__ import annotations

import json
import runpy
import sys

import numpy as np
import pytest

from aerolume import model as M
from aerolume.project import Project
from aerolume.project import provenance as prov
from aerolume.project import reproduce as rep
from aerolume.project.results import pack_result, unpack_result


def _design(name: str = "Plate") -> M.Design:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]])
    return M.Design(name=name, kind="generic", bodies=[M.Body("plate", pts)], length_ref=1.0, force_ref=0.1)


def _result(sim_id: str) -> M.CFDResult:
    return M.CFDResult(sim_id=sim_id, cd=M.Stat(1.2, 0.1, 1.0, 1.4, 0.01), cl=M.Stat(0.0, 0.0, 0, 0, 1e-4),
                       cm=M.Stat(0.0, 0.0, 0, 0, 1e-4), device="CPU (4 threads)", precision="float32",
                       wall_seconds=12.5, grid=(256, 128), cells_per_ref=64.0, re_sim=100.0, re_real=100.0,
                       history={"t": np.linspace(0, 1, 5), "cd": np.ones(5)})


@pytest.fixture
def store(tmp_path):
    p = Project.create(tmp_path / "Prov", "Prov")
    d = p.add_design(_design())
    s = p.add_simulation(M.Simulation(name="CFD", design_id=d.id))
    p.save()
    return p, d, s


def test_hashes_follow_the_computation_not_the_names():
    d1, d2 = _design("A"), _design("B")
    s1 = M.Simulation(name="one", design_id=d1.id)
    s2 = M.Simulation(name="two", design_id=d2.id, status="done", notes="renamed")
    assert prov.design_digest(d1) == prov.design_digest(d2)               # names and ids excluded
    assert prov.settings_digest(s1) == prov.settings_digest(s2)
    assert prov.config_hash(d1, s1) == prov.config_hash(d2, s2)
    s2.conditions.speed = 31.0
    assert prov.config_hash(d1, s1) != prov.config_hash(d2, s2)
    d2.bodies[0].points[0, 0] += 1e-12                                    # exact bytes, not printed digits
    assert prov.design_digest(d1) != prov.design_digest(d2)
    assert len(prov.config_hash(d1, s1)) == 64
    # 3D meshes are hashed by their exact arrays (to_dict leaves them out)
    v = np.random.default_rng(0).random((10, 3))
    f = np.array([[0, 1, 2], [2, 3, 4]])
    m1 = M.Design(name="m", kind="generic", bodies=[], length_ref=1, force_ref=1, dimension=3,
                  meshes=[M.Mesh3D("a", v, f)])
    m2 = M.Design(name="m", kind="generic", bodies=[], length_ref=1, force_ref=1, dimension=3,
                  meshes=[M.Mesh3D("a", v + 1e-9, f)])
    assert prov.design_digest(m1) != prov.design_digest(m2)


def test_saved_results_carry_provenance(store):
    p, d, s = store
    p.save_result(s.id, _result(s.id))
    rec = p.provenance(s.id)
    assert rec["format"] == prov.PROVENANCE_FORMAT and rec["schema"] == prov.PROVENANCE_SCHEMA
    assert rec["config_hash"] == prov.config_hash(d, s)
    assert rec["simulation"]["id"] == s.id and rec["design"]["name"] == "Plate"
    assert rec["software"]["aerolume"] and rec["software"]["python"]
    assert rec["software"]["packages"]["numpy"] == np.__version__
    assert rec["run"]["device"] == "CPU (4 threads)" and rec["run"]["precision"] == "float32"
    assert rec["seed"] is None and rec["deterministic_inputs"] is True
    assert set(rec["source"].values()) == {"store"} and "saved" in rec
    loaded = p.load_result(s.id)
    assert loaded.provenance["config_hash"] == rec["config_hash"]
    assert p.load_result(s.id, arrays=False).provenance == rec
    reopened = Project.open(p.path)
    assert reopened.provenance(s.id) == rec
    doc = json.loads((p.sim_dir(s.id) / "result.json").read_text(encoding="utf-8"))
    assert "provenance" not in doc["result"]                             # beside the scalars, not inside


def test_runner_record_wins_and_is_merged(store):
    p, d, s = store

    class FakePlan:
        def __init__(self):
            self.cells, self.dt, self.quality = 32768, 0.004, "draft"
            self.solver_config = M.Body("x", np.zeros((3, 2)))           # any dataclass stands in

        def summary(self):
            return {"Grid": "256 × 128", "Rayleigh number": "1.0 × 10⁴"}

    r = _result(s.id)
    prov.attach(r, d, s, plan=FakePlan(), seed=7)
    s.conditions.speed = 99.0                                            # edited while the run was going
    p.save_result(s.id, r, provenance={"command": "test"})
    rec = p.provenance(s.id)
    assert rec["simulation"]["conditions"]["speed"] == 30.0              # as run, not as edited
    assert rec["plan"]["summary"]["Grid"] == "256 × 128" and rec["plan"]["solver_hash"]
    assert rec["seed"] == 7 and rec["deterministic_inputs"] is False
    assert rec["command"] == "test"
    assert rec["source"]["plan"] == "runner" and rec["source"]["run"] == "store"
    assert rec["run"]["wall_seconds"] == 12.5                            # filled in at save time


def test_provenance_never_costs_a_save(store, monkeypatch):
    p, _d, s = store

    def boom(*a, **k):
        raise RuntimeError("no")
    monkeypatch.setattr(prov, "capture", boom)
    p.save_result(s.id, _result(s.id))
    rec = p.provenance(s.id)
    assert p.has_result(s.id) and "provenance could not be recorded" in rec["errors"][0]


def test_old_results_have_no_provenance(store):
    p, _d, s = store
    p.save_result(s.id, _result(s.id))
    f = p.sim_dir(s.id) / "result.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc.pop("provenance")
    f.write_text(json.dumps(doc), encoding="utf-8")
    assert p.provenance(s.id) is None
    assert getattr(p.load_result(s.id), "provenance", None) in (None, {})
    assert prov.summary_rows(None) == []


def test_temperature_fields_keep_their_resolution():
    t = 20.0 + 0.05 * np.sin(np.linspace(0, 6, 4000)).reshape(40, 100).astype(np.float32)
    u = 30.0 + np.cos(np.linspace(0, 6, 4000)).reshape(40, 100).astype(np.float32)
    fd = M.FieldData((0, 1, 0, 0.4), {"temperature": t, "u": u}, label="mean")
    r = M.CFDResult(sim_id="x", mean_fields=fd)
    doc, arrays = pack_result(r)
    lay = doc["arrays"]["mean_fields"]["fields"]
    assert "offset" in lay["temperature"] and "offset" not in lay["u"]   # only the affine-scale field
    back = unpack_result(doc, arrays)
    err = np.abs(back.mean_fields.fields["temperature"] - t).max()
    assert err < 3e-4 * 0.1                                              # 0.03 % of the 0.1 K spread
    naive = np.abs((t / 20.05).astype(np.float16).astype(np.float32) * 20.05 - t).max()
    assert naive > 50 * err                                              # what the plain scaling kept
    # 3D planar blocks use the same rule
    vol = M.VolumeData((0, 0, 0), (0.1, 0.1, 0.1), {"temperature": t.reshape(4, 10, 100)})
    r3 = M.CFD3DResult(sim_id="y", volume=vol)
    doc3, arr3 = pack_result(r3)
    back3 = unpack_result(doc3, arr3)
    assert np.abs(back3.volume.fields["temperature"] - t.reshape(4, 10, 100)).max() < 3e-5


def test_diagnostics_table_round_trips():
    r = _result("x")
    r.diagnostics_history = {"t": np.linspace(0, 1, 7), "div_rms": np.full(7, 2e-7)}
    doc, arrays = pack_result(r)
    assert doc["arrays"]["diagnostics_history"] == ["t", "div_rms"]
    assert "diagnostics_history" not in doc["result"]
    back = unpack_result(doc, arrays)
    assert np.array_equal(back.diagnostics_history["div_rms"], r.diagnostics_history["div_rms"])
    assert not getattr(unpack_result(*pack_result(_result("y"))), "diagnostics_history", None)
    r3 = M.CFD3DResult(sim_id="z")
    r3.diagnostics_history = {"t": np.arange(3.0)}
    assert list(unpack_result(*pack_result(r3)).diagnostics_history) == ["t"]


def test_expected_values_and_check_tolerances():
    r = _result("x")
    ex = rep.expected_values(r)
    assert ex["kind"] == "cfd" and ex["values"]["cd"] == [1.2, 0.01]
    close = _result("y")
    close.cd = M.Stat(1.2 + 0.02, 0.1, 1, 1.4, 0.01)                     # 1.4 combined s.e.
    rows, ok = rep.check_reproduction(close, ex)
    assert ok and rows[0]["tolerance"] == pytest.approx(3 * np.hypot(0.01, 0.01))
    far = _result("z")
    far.cd = M.Stat(1.3, 0.1, 1, 1.4, 0.01)
    rows, ok = rep.check_reproduction(far, ex)
    assert not ok and "DIFFERENT" in rows[0]["text"]
    pol = M.PolarResult(sim_id="p", re=1e6, points=[M.PolarPoint(2.0, 0.4, 0.008, cm=-0.05, converged=True),
                                                     M.PolarPoint(4.0, 0.6, 0.009, converged=False)])
    ex = rep.expected_values(pol)
    assert set(ex["values"]) == {"cl@2", "cd@2", "cm@2"}
    pol.points[0].cl += 1e-6                                             # beyond 1e-5 relative? no: 2.5e-6
    assert rep.check_reproduction(pol, ex)[1]
    pol.points[0].cl += 1e-4
    assert not rep.check_reproduction(pol, ex)[1]


def test_reproduction_script_reruns_a_wing_and_agrees(tmp_path, monkeypatch, capsys):
    import aerolume as al
    p = al.new_project(tmp_path / "Study")
    w = p.add_wing(span=6.0, root_chord=1.0, tip_chord=0.5, sweep_deg=5.0, airfoil="2412")
    r = p.run_wing(w, speed=20.0, alpha=(0.0, 4.0, 2.0), method="vlm", viscous=False)
    path = p.raw.export_reproduction_script(r.sim_id)
    assert path.parent == p.raw.scripts_dir and path.suffix == ".py"
    text = path.read_text(encoding="utf-8")
    assert "Configuration hash" in text and r.sim_id in text and "check_reproduction" in text
    compile(text, str(path), "exec")
    monkeypatch.setattr(sys, "argv", [str(path)])
    with pytest.raises(SystemExit) as ex:
        runpy.run_path(str(path), run_name="__main__")
    out = capsys.readouterr().out
    assert ex.value.code == 0, out
    assert "Reproduced: every value agrees." in out
    store = Project.open(path.with_name(path.stem + ".aerolume"))
    assert len(store.simulations) == 1 and len(store.designs) == 1
    sim = next(iter(store.simulations.values()))
    assert prov.settings_digest(sim) == prov.settings_digest(p.raw.simulations[r.sim_id])
    rec = store.provenance(sim.id)
    assert rec["design_hash"] == p.raw.provenance(r.sim_id)["design_hash"]  # the same wing


def test_reproduction_script_of_a_3d_design_ships_its_mesh(tmp_path):
    p = Project.create(tmp_path / "P3", "P3")
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    d = p.add_design(M.Design(name="Tet", kind="generic", bodies=[], length_ref=1, force_ref=1, dimension=3,
                              meshes=[M.Mesh3D("tet", v, f)], ref_area=0.5))
    s = p.add_simulation(M.Simulation(name="CFD 3D", design_id=d.id))
    p.save()
    out = p.export_reproduction_script(s.id, tmp_path / "out" / "repro")
    assert out.name == "repro.py"
    mesh = out.with_name("repro_mesh.npz")
    with np.load(mesh) as z:
        assert np.array_equal(z["mesh0_vertices"], v) and np.array_equal(z["mesh0_faces"], f)
    text = out.read_text(encoding="utf-8")
    assert "MESH_FILE = 'repro_mesh.npz'" in text and "no stored result" in text
    with pytest.raises(KeyError):
        p.export_reproduction_script("nope")


def test_rebuild_maps_wing_section_airfoils(tmp_path):
    import aerolume as al
    src = al.new_project(tmp_path / "Src")
    af = src.add_naca("4412")
    w = src.add_wing(span=4.0, root_chord=1.0, airfoil=af)
    raw_w = src.raw.designs[w.id]
    assert raw_w.wing.sections[0].airfoil == f"design:{af.id}"
    dst = al.new_project(tmp_path / "Dst")
    sim = M.Simulation(name="Wing", design_id=raw_w.id, analysis="wing")
    out = rep.rebuild(dst, [raw_w.to_dict(), src.raw.designs[af.id].to_dict()], sim.to_dict())
    new_w = dst.raw.designs[out.design_id]
    ref = new_w.wing.sections[0].airfoil
    assert ref.startswith("design:") and ref[7:] in dst.raw.designs and ref[7:] != af.id
    again = rep.rebuild(dst, [raw_w.to_dict(), src.raw.designs[af.id].to_dict()], sim.to_dict())
    assert again.id == out.id and len(dst.raw.simulations) == 1          # reused, not duplicated
