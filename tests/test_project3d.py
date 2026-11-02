"""Project store, 3D: mesh arrays of 3D designs, cfd3d / wing results, forward compatibility, v0.1 projects."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from aerolume.model import (
    CFD3DResult,
    CFDResult,
    Design,
    Mesh3D,
    PolarResult,
    Simulation,
    SliceData,
    Stat,
    SurfaceData3D,
    VolumeData,
    WingPoint,
    WingResult,
    WingSection,
    WingSpec,
)
from aerolume.project import Project, ProjectCorruptError
from aerolume.project.meshes import mesh_digest

EXAMPLES = Path(__file__).resolve().parents[1] / "src" / "aerolume" / "examples" / "projects"


def uv_sphere(n_theta: int = 40, n_phi: int = 60, r: float = 0.5, centre=(0.0, 0.0, 0.6)) -> tuple[np.ndarray, np.ndarray]:
    """A closed UV sphere (vertices from float32 like an imported STL), as vertex / face arrays."""
    th = np.linspace(0.0, np.pi, n_theta)
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3) * r
    v = (v + np.asarray(centre)).astype(np.float32).astype(np.float64)
    i = np.arange(n_theta - 1)[:, None] * n_phi
    j = np.arange(n_phi)[None, :]
    a, b = (i + j).ravel(), (i + (j + 1) % n_phi).ravel()
    c, d = a + n_phi, b + n_phi
    return v, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)]).astype(np.int64)


def make_design3d(name: str = "Car") -> Design:
    v, f = uv_sphere()
    wv, wf = uv_sphere(12, 16, r=0.2, centre=(0.4, 0.5, 0.2))
    return Design(name=name, kind="vehicle", bodies=[], length_ref=4.2, force_ref=0.0, dimension=3,
                  meshes=[Mesh3D("body", v, f), Mesh3D("wheel_fl", wv, wf, role="wheel", omega=150.0,
                                                      center=(0.4, 0.5, 0.2), axis=(0.0, 1.0, 0.0))],
                  ref_area=2.1, span_ref=1.8, moment_ref3=(1.4, 0.0, 0.0), symmetric=True)


def make_wing_design() -> Design:
    spec = WingSpec(sections=[WingSection(0.0, 0.3, airfoil="2412"),
                              WingSection(1.2, 0.18, x_le=0.08, z_le=0.05, twist_deg=-2.0, airfoil="design:abc")])
    return Design(name="Glider wing", kind="wing", bodies=[], length_ref=0.25, force_ref=0.0, dimension=3,
                  wing=spec, ref_area=0.576, span_ref=2.4)


def make_sim(design: Design, analysis: str = "cfd") -> Simulation:
    return Simulation(name="Run", design_id=design.id, analysis=analysis)


def st(mean: float, se: float = 0.003) -> Stat:
    return Stat(mean=mean, rms=0.02, min=mean - 0.05, max=mean + 0.05, stderr=se, stderr_reliable=se < 0.01)


def make_cfd3d(sim_id: str, rng: np.random.Generator, *, vol_shape=(24, 20, 40), surf_n: tuple = (30, 40),
               sens: bool = True) -> CFD3DResult:
    t = np.linspace(0, 20, 801)
    hist = {"t": t, **{k: rng.standard_normal(t.size) * 0.01 + m for k, m in
                       (("cd", 0.3), ("cs", 0.0), ("cl", -0.1), ("c_roll", 0.0), ("cm", 0.02), ("cn", 0.0))}}
    v, f = uv_sphere(*surf_n)
    n = len(v)
    surf = [SurfaceData3D("body", v, f, cp=rng.standard_normal(n), cf=np.abs(rng.standard_normal(n)) * 1e-3,
                          tau=rng.standard_normal((n, 3)) * 1e-3,
                          sensitivity=rng.standard_normal(n) * 1e-4 if sens else None),
            SurfaceData3D("wheel", v[:50] * 0.3, f[(f < 50).all(axis=1)], cp=rng.standard_normal(50))]
    vol_fields = {k: (rng.standard_normal(vol_shape) * s + m).astype(np.float32)
                  for k, s, m in (("u", 3.0, 30.0), ("v", 1.0, 0.0), ("w", 1.0, 0.0), ("p", 300.0, -50.0),
                                  ("q", 1e4, 0.0))}
    vol_fields["cp"] = vol_fields["p"] / 551.25
    mask = np.zeros(vol_shape, np.float32)
    mask[8:14, :6, 10:20] = 1.0
    mask[8:14, 6, 10:20] = 0.4
    slices = [SliceData("y", 0.0, (-2.0, 8.0, 0.0, 3.0), {"u": rng.standard_normal((30, 100)) * 5 + 30,
                                                          "cp": rng.standard_normal((30, 100))},
                        mask=(rng.random((30, 100)) > 0.9).astype(float)),
              SliceData("x", 4.5, (-1.5, 1.5, 0.0, 3.0), {"u": rng.standard_normal((30, 30))}, label="instant")]
    return CFD3DResult(
        sim_id=sim_id, cd=st(0.31), cs=st(0.0, 0.02), cl=st(-0.12), c_roll=st(0.0), cm=st(0.021), cn=st(0.0),
        drag=350.0, side=0.0, lift=-135.0, moments=(0.0, 95.0, 0.0), drag_area=0.651, power=10500.0,
        strouhal=None, strouhal_length=1.4, re_real=8.6e6, re_sim=2.0e5, grid=(320, 96, 128), cells=320 * 96 * 128,
        cells_per_ref=76.0, symmetry_plane=True, t_end=20.0, t_avg_start=6.0, converged=True, wall_seconds=212.0,
        device="cuda", precision="float32", turbulence="wale", notes=["half domain"], warnings=["capped Re"],
        per_body={"body": {"cd": 0.29, "cl": -0.1}, "wheel_fl": {"cd": 0.02, "cl": -0.02}},
        history=hist, surface=surf, slices=slices,
        volume=VolumeData(origin=(-2.0, 0.0, 0.0), spacing=(0.25, 0.1, 0.125), fields=vol_fields, mask=mask,
                          mirrored=True))


def make_wing_result(sim_id: str) -> WingResult:
    pts = []
    for a in np.arange(-4.0, 11.0, 1.0):
        y = np.linspace(-1.2, 1.2, 24)
        cl = 0.09 * (a + 2)
        pts.append(WingPoint(alpha=float(a), cl=cl, cd=0.01 + cl * cl / 26.0, cdi=cl * cl / 28.0,
                             cdv=0.01 + cl * cl / 364.0, cm=-0.04, e=0.98, converged=bool(a != 7.0),
                             y=y, cl_local=cl * np.sqrt(1 - (y / 1.3) ** 2), c_cl=np.cos(y),
                             alpha_i=np.full(24, -0.8), cd_local=None if a < 0 else np.full(24, 0.01)))
    return WingResult(sim_id=sim_id, method="vlm", viscous=True, area=0.576, span=2.4, aspect_ratio=10.0,
                      mac=0.245, re_mac=4.1e5, points=pts, wall_seconds=1.7, notes=["n"], warnings=["w"])


@pytest.fixture
def proj(tmp_path):
    return Project.create(tmp_path / "Three D")


# --- 3D designs ---------------------------------------------------------------------

def test_3d_design_round_trip_keeps_mesh_arrays_exact(proj):
    d = proj.add_design(make_design3d())
    w = proj.add_design(make_wing_design())
    proj.save()
    j = proj.path / "designs" / f"{d.id}.json"
    npz = proj.path / "designs" / f"{d.id}.npz"
    assert npz.is_file() and not (proj.path / "designs" / f"{w.id}.npz").exists()
    doc = json.loads(j.read_text(encoding="utf-8"))
    assert "vertices" not in doc["meshes"][0] and doc["meshes"][0]["n_vertices"] == 2400
    assert doc["mesh_file"]["digest"] == mesh_digest(d)
    assert j.stat().st_size < 4000                              # arrays are not in the JSON

    q = Project.open(proj.path)
    assert q.load_warnings == []
    e = q.designs[d.id]
    assert e.to_dict() == d.to_dict()
    assert e.dimension == 3 and e.symmetric and e.ref_area == 2.1 and e.moment_ref3 == (1.4, 0.0, 0.0)
    for a, b in zip(e.meshes, d.meshes, strict=True):
        assert a.vertices.dtype == np.float64 and a.faces.dtype == np.int64
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.faces, b.faces)
    assert e.meshes[1].role == "wheel" and e.meshes[1].axis == (0.0, 1.0, 0.0) and e.meshes[1].omega == 150.0
    ww = q.designs[w.id]
    assert ww.kind == "wing" and ww.meshes == [] and ww.wing.sections[1].airfoil == "design:abc"
    assert ww.wing.to_dict() == w.wing.to_dict()


def test_mesh_file_is_rewritten_only_when_the_arrays_change(proj):
    d = proj.add_design(make_design3d())
    proj.save()
    npz = proj.path / "designs" / f"{d.id}.npz"
    j = proj.path / "designs" / f"{d.id}.json"
    t_npz, t_json = npz.stat().st_mtime_ns, j.stat().st_mtime_ns
    time.sleep(0.02)
    proj.save()
    d.notes = "edited"                                              # JSON-only change
    proj.save()
    assert npz.stat().st_mtime_ns == t_npz and j.stat().st_mtime_ns != t_json
    d.meshes[0].vertices[5, 2] += 0.01                              # in-place geometry edit
    proj.save()
    assert npz.stat().st_mtime_ns != t_npz
    q = Project.open(proj.path)
    assert q.designs[d.id].meshes[0].vertices[5, 2] == d.meshes[0].vertices[5, 2]
    # opening and saving again rewrites neither file
    t_npz, t_json = npz.stat().st_mtime_ns, j.stat().st_mtime_ns
    q.save()
    assert (npz.stat().st_mtime_ns, j.stat().st_mtime_ns) == (t_npz, t_json)
    # dropping the meshes removes the npz; removing the design removes both files
    q.designs[d.id].meshes = []
    q.save()
    assert not npz.exists() and "mesh_file" not in json.loads(j.read_text(encoding="utf-8"))
    d2 = q.add_design(make_design3d("Other"))
    q.save()
    q.remove_design(d2.id)
    assert not (q.path / "designs" / f"{d2.id}.npz").exists()
    assert not (q.path / "designs" / f"{d2.id}.json").exists()


def test_mesh_file_is_compact(proj):
    v, f = uv_sphere(400, 500, r=1.0)                                # 200k vertices, ~400k faces
    d = proj.add_design(Design(name="Scan", kind="generic", bodies=[], length_ref=2.0, force_ref=0.0,
                               dimension=3, meshes=[Mesh3D("scan", v, f)], ref_area=3.14))
    proj.save()
    size = (proj.path / "designs" / f"{d.id}.npz").stat().st_size
    assert size < 0.25 * (v.nbytes + f.nbytes)                       # 14.4 MB in memory
    t0 = time.perf_counter()
    q = Project.open(proj.path)
    dt = time.perf_counter() - t0
    np.testing.assert_array_equal(q.designs[d.id].meshes[0].vertices, v)
    assert dt < 5.0


def _damage_setup(proj):
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    proj.save()
    return d, s, proj.path / "designs" / f"{d.id}.npz", proj.path / "designs" / f"{d.id}.json"


def test_missing_or_damaged_mesh_file_skips_the_design_and_keeps_its_files(proj):
    d, s, npz, j = _damage_setup(proj)
    keep = npz.read_bytes()
    npz.unlink()
    q = Project.open(proj.path)
    assert d.id not in q.designs and s.id in q.simulations
    assert any("mesh file" in w and "missing" in w for w in q.load_warnings)
    q.save()
    assert j.is_file()                                               # never deleted by a save
    npz.write_bytes(keep[: len(keep) // 2])                          # truncated
    q = Project.open(proj.path)
    assert d.id not in q.designs and any("could not be read" in w for w in q.load_warnings)
    npz.write_bytes(keep)
    q = Project.open(proj.path)        # the save above dropped it from the index: recovered from disk
    assert d.id in q.designs and any("Recovered design" in w for w in q.load_warnings)
    np.testing.assert_array_equal(q.designs[d.id].meshes[0].faces, d.meshes[0].faces)


def test_mesh_file_of_another_save_is_detected(proj):
    d, _, _, j = _damage_setup(proj)
    # same shapes, different geometry: e.g. a crash between the npz and the JSON write
    old_json = j.read_text(encoding="utf-8")
    d.meshes[0].vertices = d.meshes[0].vertices * 1.01
    proj.save()
    j.write_text(old_json, encoding="utf-8")
    q = Project.open(proj.path)
    assert any("different save" in w for w in q.load_warnings)
    np.testing.assert_array_equal(q.designs[d.id].meshes[0].vertices, d.meshes[0].vertices)
    q.save()                                                         # pairs them up again
    assert not Project.open(proj.path).load_warnings
    # different shapes: the design is skipped
    d.meshes[0].vertices = d.meshes[0].vertices[:100]
    d.meshes[0].faces = d.meshes[0].faces[:10]
    proj.save()
    j.write_text(old_json, encoding="utf-8")
    q = Project.open(proj.path)
    assert d.id not in q.designs and any("does not fit" in w for w in q.load_warnings)


def test_plain_npz_of_mesh_arrays_is_accepted(proj):
    d = make_design3d()
    proj.save()
    (proj.path / "designs" / f"{d.id}.json").write_text(json.dumps(d.to_dict()), encoding="utf-8")
    np.savez(proj.path / "designs" / f"{d.id}.npz", **{k: (a.astype(np.float32) if "vert" in k else a)
                                                        for k, a in d.mesh_arrays().items()})
    q = Project.open(proj.path)
    e = q.designs[d.id]
    np.testing.assert_array_equal(e.meshes[0].vertices, d.meshes[0].vertices)  # float32-exact vertices
    assert e.meshes[0].vertices.dtype == np.float64
    q.save()                                                       # rewritten in the store's own format
    with np.load(proj.path / "designs" / f"{d.id}.npz") as z:
        assert "__digest__" in z.files


# --- 3D results -----------------------------------------------------------------------

def test_cfd3d_result_round_trip(proj):
    rng = np.random.default_rng(1)
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    proj.save()
    r = make_cfd3d(s.id, rng)
    proj.save_result(s.id, r)
    assert proj.has_result(s.id)
    back = Project.open(proj.path).load_result(s.id)
    assert isinstance(back, CFD3DResult)
    for k in ("cd", "cs", "cl", "c_roll", "cm", "cn"):
        assert getattr(back, k) == getattr(r, k)
    assert back.moments == r.moments and back.grid == r.grid and back.cells == r.cells
    assert back.per_body == r.per_body and back.symmetry_plane is True and back.strouhal is None
    assert back.notes == r.notes and back.warnings == r.warnings and back.turbulence == "wale"
    # histories exact float64
    assert list(back.history) == list(r.history)
    for k, v in r.history.items():
        assert back.history[k].dtype == np.float64
        np.testing.assert_array_equal(back.history[k], v)
    # surfaces: float32 precision, faces exact, optional arrays preserved / absent
    assert [sd.body for sd in back.surface] == ["body", "wheel"]
    for a, b in zip(back.surface, r.surface, strict=True):
        np.testing.assert_array_equal(a.faces, b.faces)
        assert a.faces.dtype == np.int64
        np.testing.assert_allclose(a.vertices, b.vertices, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(a.cp, b.cp, rtol=1e-6)
    np.testing.assert_allclose(back.surface[0].tau, r.surface[0].tau, rtol=1e-6)
    np.testing.assert_allclose(back.surface[0].sensitivity, r.surface[0].sensitivity, rtol=1e-6)
    assert back.surface[1].cf is None and back.surface[1].tau is None and back.surface[1].sensitivity is None
    # slices and volume: float16 with a per-field scale -> 1e-3 of each field's peak
    assert [(sl.axis, sl.position, sl.label) for sl in back.slices] == [("y", 0.0, "mean"), ("x", 4.5, "instant")]
    assert back.slices[0].extent == r.slices[0].extent and back.slices[1].mask is None
    for a, b in zip(back.slices, r.slices, strict=True):
        for k, v in b.fields.items():
            assert a.fields[k].dtype == np.float32 and a.fields[k].shape == v.shape
            assert np.abs(a.fields[k] - v).max() <= 1e-3 * np.abs(v).max()
    np.testing.assert_allclose(back.slices[0].mask, r.slices[0].mask, atol=1e-3)
    vol, rv = back.volume, r.volume
    assert vol.origin == rv.origin and vol.spacing == rv.spacing and vol.mirrored is True and vol.label == "mean"
    assert set(vol.fields) == set(rv.fields)
    for k, v in rv.fields.items():
        assert vol.fields[k].shape == v.shape and vol.fields[k].dtype == np.float32
        assert np.abs(vol.fields[k] - v).max() <= 1e-3 * np.abs(v).max()
    np.testing.assert_allclose(vol.mask, rv.mask, atol=1e-3)
    # the 3D arrays really are stored shuffled (uint8 blocks), the history as float64
    with np.load(proj.sim_dir(s.id) / "arrays.npz") as z:
        assert z["volume/u"].dtype == np.uint8 and z["volume/u"].shape == (2, rv.fields["u"].size)
        assert z["history/t"].dtype == np.float64
    summ = proj.result_summary(s.id)
    assert summ["type"] == "cfd3d" and summ["cd"]["mean"] == 0.31 and "volume" not in summ
    light = proj.load_result(s.id, arrays=False)
    assert isinstance(light, CFD3DResult) and light.volume is None and light.surface == [] and light.cd == r.cd
    proj.delete_result(s.id)
    assert not proj.has_result(s.id) and proj.load_result(s.id) is None and proj.result_summary(s.id) is None


def test_cfd3d_result_without_optional_parts(proj):
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    r = CFD3DResult(sim_id=s.id, cd=st(0.4))
    proj.save_result(s.id, r)
    back = proj.load_result(s.id)
    assert back.volume is None and back.slices == [] and back.surface == [] and back.history == {}
    assert back.extra_volumes == [] and back.volumes() == [] and back.find_volume("dye") is None
    assert back.cd == r.cd and math.isnan(back.drag) and back.grid == (0, 0, 0)
    doc = json.loads((proj.sim_dir(s.id) / "result.json").read_text(encoding="utf-8"))
    assert "extra_volumes" not in doc["arrays"]          # nothing new in files that do not need it


def test_smoke_snapshot_volume_round_trip(proj):
    """DESIGN.md 7.13: the instantaneous smoke ("dye") block survives the store next to the mean
    volume, and find_volume tells the viewer which block to use for smoke and for streamlines."""
    rng = np.random.default_rng(7)
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    r = make_cfd3d(s.id, rng)
    shape = r.volume.fields["u"].shape
    dye = np.clip(rng.random(shape), 0.0, 1.0).astype(np.float32)
    dye[0] = 0.0                                                   # clean air below the smoke
    snap = VolumeData(origin=r.volume.origin, spacing=r.volume.spacing, label="instant", mirrored=True,
                      fields={"dye": dye, "u": rng.normal(30.0, 3.0, shape).astype(np.float32)},
                      mask=r.volume.mask)
    r.extra_volumes = [snap]
    proj.save_result(s.id, r)
    back = Project.open(proj.path).load_result(s.id)
    assert [v.label for v in back.volumes()] == ["mean", "instant"]
    smoke = back.find_volume("dye")
    assert smoke is back.extra_volumes[0] and smoke.mirrored is True
    assert smoke.origin == snap.origin and smoke.spacing == snap.spacing
    assert np.abs(smoke.fields["dye"] - dye).max() <= 1e-3 and float(smoke.fields["dye"][0].max()) == 0.0
    np.testing.assert_allclose(smoke.mask, snap.mask, atol=1e-3)
    assert back.find_volume("u", "mean") is back.volume and back.find_volume("u", "instant") is smoke
    assert back.find_volume("q", "instant") is None
    # the light (scalars-only) load leaves the blocks out, like the mean volume
    assert proj.load_result(s.id, arrays=False).extra_volumes == []


def test_project_folders_for_scripts_and_experiments(proj):
    """DESIGN.md 8 / 9: scripts and experiment logs live in the project folder, are created on
    demand, and saving the project never touches them."""
    scripts, experiments = proj.scripts_dir, proj.experiments_dir
    assert scripts == proj.path / "scripts" and experiments == proj.path / "experiments"
    assert scripts.is_dir() and experiments.is_dir()
    (scripts / "sweep.py").write_text("print('hi')\n", encoding="utf-8")
    (experiments / "e1").mkdir()
    (experiments / "e1" / "experiment.json").write_text("{}", encoding="utf-8")
    proj.add_design(make_design3d())
    proj.save()
    again = Project.open(proj.path)
    assert (again.scripts_dir / "sweep.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert (again.experiments_dir / "e1" / "experiment.json").exists() and not again.load_warnings


def test_wing_result_round_trip(proj):
    d = proj.add_design(make_wing_design())
    s = proj.add_simulation(make_sim(d, "wing"))
    w = make_wing_result(s.id)
    proj.save_result(s.id, w)
    back = Project.open(proj.path).load_result(s.id)
    assert isinstance(back, WingResult)
    assert (back.method, back.viscous, back.area, back.aspect_ratio, back.re_mac) == ("vlm", True, 0.576, 10.0, 4.1e5)
    assert back.notes == ["n"] and back.warnings == ["w"] and len(back.points) == len(w.points)
    for a, b in zip(back.points, w.points, strict=True):
        assert (a.alpha, a.cl, a.cd, a.cdi, a.cdv, a.cm, a.e, a.converged) == \
            (b.alpha, b.cl, b.cd, b.cdi, b.cdv, b.cm, b.e, b.converged)
        for k in WingPoint._arrays:
            va, vb = getattr(a, k), getattr(b, k)
            if vb is None:
                assert va is None
            else:
                np.testing.assert_array_equal(va, vb)
    summ = proj.result_summary(s.id)
    assert summ["type"] == "wing" and summ["points"][0]["y"] is None and summ["points"][0]["cl"] == w.points[0].cl
    np.testing.assert_array_equal(back.table()["cl"], w.table()["cl"])


def test_result_types_can_replace_each_other(proj):
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    proj.save_result(s.id, make_cfd3d(s.id, np.random.default_rng(0)))
    assert proj.result_summary(s.id)["type"] == "cfd3d"
    proj.save_result(s.id, make_wing_result(s.id))
    assert proj.result_summary(s.id)["type"] == "wing"
    assert isinstance(proj.load_result(s.id), WingResult)


def test_forward_compatible_3d_results(proj):
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    proj.save()
    proj.save_result(s.id, make_cfd3d(s.id, np.random.default_rng(0)))
    rj = proj.sim_dir(s.id) / "result.json"
    doc = json.loads(rj.read_text(encoding="utf-8"))
    doc["result"]["future_metric"] = 3.0
    doc["arrays"]["future_block"] = {"x": 1}
    doc["arrays"]["volume"]["future_key"] = True
    rj.write_text(json.dumps(doc), encoding="utf-8")
    dj = proj.path / "designs" / f"{d.id}.json"
    ddoc = json.loads(dj.read_text(encoding="utf-8"))
    ddoc["hologram"] = 1
    ddoc["meshes"][0]["texture"] = "chrome"
    dj.write_text(json.dumps(ddoc), encoding="utf-8")
    q = Project.open(proj.path)
    assert q.load_warnings == [] and isinstance(q.load_result(s.id), CFD3DResult)
    # a result type from a newer version: a clear error, the summary still readable
    doc["type"] = "cfd4d"
    rj.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ProjectCorruptError, match="newer"):
        q.load_result(s.id)
    assert q.result_summary(s.id)["type"] == "cfd4d"
    # an array encoding this version does not know is a clear error, not a crash
    doc["type"] = "cfd3d"
    doc["arrays"]["volume"]["fields"]["u"]["enc"] = "zstd"
    rj.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ProjectCorruptError, match="encoding"):
        q.load_result(s.id)


def test_mismatched_3d_arrays_are_reported(proj):
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    proj.save_result(s.id, make_cfd3d(s.id, np.random.default_rng(0)))
    keep = (proj.sim_dir(s.id) / "arrays.npz").read_bytes()
    proj.save_result(s.id, make_cfd3d(s.id, np.random.default_rng(1)))
    (proj.sim_dir(s.id) / "arrays.npz").write_bytes(keep)
    with pytest.raises(ProjectCorruptError, match="interrupted"):
        proj.load_result(s.id)


# --- v0.1 projects open unchanged ------------------------------------------------------

def _example_dimensions(path: Path) -> set[int]:
    """Dimensions of an example project's designs, read from their JSON (a wing is 3D)."""
    return {int(json.loads(f.read_text(encoding="utf-8")).get("dimension", 2) or 2)
            for f in (path / "designs").glob("*.json")}


@pytest.mark.parametrize("name", sorted(p.name for p in EXAMPLES.glob("*.aerolume")) if EXAMPLES.is_dir() else [])
def test_example_projects_open_and_save_unchanged(tmp_path, name):
    """Every bundled example opens without warnings, loads its results, and saving it
    rewrites nothing but the index's time stamp (the app opens a copy, and a copy
    that rewrote its arrays on first save would hide a format drift)."""
    src = EXAMPLES / name
    dst = tmp_path / name
    shutil.copytree(src, dst)
    before = {p.relative_to(dst).as_posix(): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    q = Project.open(dst)
    assert q.load_warnings == []
    dims = {d.dimension for d in q.designs.values()}
    assert dims == _example_dimensions(src) and dims
    for sid, sim in q.simulations.items():
        if q.has_result(sid):
            res = q.load_result(sid)
            design = q.designs[sim.design_id]
            if design.dimension == 2:
                assert isinstance(res, (CFDResult, PolarResult))
            else:
                assert isinstance(res, CFD3DResult if sim.analysis == "cfd" else WingResult)
    q.save()
    after = {p.relative_to(dst).as_posix(): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    assert set(after) == set(before)                              # no npz or anything else added
    changed = [k for k in before if before[k] != after[k]]
    assert changed in ([], ["project.json"])                       # only the index's time stamp may move
    if changed:
        a, b = json.loads(before["project.json"]), json.loads(after["project.json"])
        a.pop("modified")
        b.pop("modified")
        a.pop("app_version", None)
        b.pop("app_version", None)
        assert a == b


def test_a_3d_example_ships_with_smoke_streamline_volume_and_a_wing_polar():
    """The Start page lists 3D examples first so a new user can open a 3D result
    without a GPU run: one car with surface cp, a mean volume (streamlines) and a smoke
    snapshot, and one wing polar -- each example small enough to ship as package data."""
    if not EXAMPLES.is_dir():
        pytest.skip("no bundled examples in this checkout")
    three_d = [p for p in EXAMPLES.glob("*.aerolume") if 3 in _example_dimensions(p)]
    assert three_d, "no 3D example project is bundled"
    found_car = found_wing = False
    for path in three_d:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        assert size <= 15e6, f"{path.name} is {size / 1e6:.1f} MB"
        q = Project.open(path)
        for sid, sim in q.simulations.items():
            assert sim.status == "done" and q.has_result(sid)
            res = q.load_result(sid)
            if isinstance(res, CFD3DResult):
                mean, smoke = res.find_volume("u", "mean"), res.find_volume("dye")
                assert mean is not None and smoke is not None and smoke.label == "instant"
                assert {"u", "v", "w"} <= set(smoke.fields)            # instantaneous streamlines too
                assert float(np.nanmax(smoke.fields["dye"])) > 0.05     # the smoke actually left the rake
                assert res.surface and np.isfinite(res.surface[0].cp).all()
                assert res.cd.mean > 0 and math.isfinite(res.cl.mean)
                assert "m/s" in sim.name and "0.0" not in sim.name     # named by its speed, readable
                found_car = True
            elif isinstance(res, WingResult):
                assert sum(p.converged for p in res.points) >= 10
                found_wing = True
    assert found_car and found_wing


# --- schema policy: Aerolume 0.1 must never rewrite a 3D design as a 2D one ----------------

def _v01_design_from_dict(doc: dict) -> Design:
    """What Aerolume 0.1 did with a design file: the dataclass built from the fields it knew
    (unknown keys ignored), its kinds only, and ``bodies`` a required argument."""
    v01 = {"name", "kind", "bodies", "length_ref", "force_ref", "moment_ref", "id", "source", "created", "notes"}
    if doc.get("kind") not in ("airfoil", "vehicle", "generic"):
        raise ValueError(f"unknown design kind {doc.get('kind')!r}")
    return Design(**{k: v for k, v in doc.items() if k in v01})    # TypeError without "bodies"


def test_schema_is_1_for_2d_projects_and_2_with_3d_designs(proj):
    from aerolume.geometry import design_from_airfoil, naca
    from aerolume.model import SCHEMA_VERSION, SCHEMA_VERSION_2D

    assert (SCHEMA_VERSION, SCHEMA_VERSION_2D) == (2, 1)
    proj.add_design(design_from_airfoil(naca("2412"), name="NACA 2412"))
    proj.save()
    assert json.loads((proj.path / "project.json").read_text())["schema"] == 1 and proj.schema == 1
    d3 = proj.add_design(make_design3d())
    proj.save()
    assert json.loads((proj.path / "project.json").read_text())["schema"] == 2 and proj.schema == 2
    assert Project.open(proj.path).schema == 2
    proj.remove_design(d3.id)
    proj.save()
    assert json.loads((proj.path / "project.json").read_text())["schema"] == 1


def test_3d_design_files_cannot_be_read_as_2d_by_v01(proj):
    """Aerolume 0.1 requires ``bodies``: a 3D design file leaves it out, so 0.1 skips the file
    (and never writes it back) instead of loading an empty 2D design; this version reads it,
    also when 0.1 has since dropped it from project.json's list."""
    d = proj.add_design(make_design3d())
    w = proj.add_design(make_wing_design())
    proj.save()
    for did in (d.id, w.id):
        doc = json.loads((proj.path / "designs" / f"{did}.json").read_text())
        assert "bodies" not in doc and doc["dimension"] == 3
        with pytest.raises((KeyError, TypeError, ValueError)):
            _v01_design_from_dict(doc)
    # 0.1 saved the project without the designs it skipped, and with its own schema
    meta = json.loads((proj.path / "project.json").read_text())
    meta["designs"], meta["schema"] = [], 1
    (proj.path / "project.json").write_text(json.dumps(meta))
    q = Project.open(proj.path)
    assert set(q.designs) == {d.id, w.id}
    np.testing.assert_array_equal(q.designs[d.id].meshes[0].vertices, d.meshes[0].vertices)
    assert q.designs[w.id].wing.sections[1].airfoil == "design:abc"
    assert any("Recovered design" in m for m in q.load_warnings)
    # a design file from before this rule (with "bodies": []) still opens, and is rewritten once
    doc = json.loads((proj.path / "designs" / f"{d.id}.json").read_text())
    doc["bodies"] = []
    (proj.path / "designs" / f"{d.id}.json").write_text(json.dumps(doc))
    r = Project.open(proj.path)
    assert r.designs[d.id].meshes[0].faces.shape == d.meshes[0].faces.shape
    r.save()
    assert "bodies" not in json.loads((proj.path / "designs" / f"{d.id}.json").read_text())


def test_a_newer_schema_opens_with_a_warning(proj):
    proj.save()
    meta = json.loads((proj.path / "project.json").read_text())
    meta["schema"] = 3
    (proj.path / "project.json").write_text(json.dumps(meta))
    q = Project.open(proj.path)
    assert q.schema == 3 and any("newer Aerolume (schema 3" in m for m in q.load_warnings)


def test_numpy_bools_and_stall_onset_survive_the_store(proj):
    """WingPoint.converged from a numpy comparison is np.bool_, which json refuses; the model
    converts it. The stall-onset estimate is stored as fields (NaN when there is none)."""
    d = proj.add_design(make_wing_design())
    s = proj.add_simulation(make_sim(d, "wing"))
    w = make_wing_result(s.id)
    for p in w.points:
        p.converged = np.float64(p.cl) < np.float64(5.0)            # np.bool_
    assert isinstance(w.points[0].converged, np.bool_)
    w.stall_alpha, w.stall_cl = 8.4, np.float64(0.97)
    proj.save_result(s.id, w)
    back = proj.load_result(s.id)
    assert back.points[0].converged is True and (back.stall_alpha, back.stall_cl) == (8.4, 0.97)
    assert math.isnan(WingResult.from_dict({"sim_id": "x"}).stall_alpha)        # older results


# --- size of a realistic 3D result --------------------------------------------------------

def test_realistic_3d_result_size(proj):
    """2 M-cell volume (u, v, w, p, cp, q + mask) and a 200k-vertex surface: the file stays compact."""
    rng = np.random.default_rng(7)
    nz, ny, nx = 96, 112, 192
    z, y, x = np.meshgrid(np.linspace(0, 2, nz, dtype=np.float32), np.linspace(0, 1.5, ny, dtype=np.float32),
                          np.linspace(-2, 8, nx, dtype=np.float32), indexing="ij")
    wake = np.exp(-(y ** 2 + (z - 0.7) ** 2) / 0.3) * (x > 1.5)
    noise = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    fields = {"u": 30 * (1 - 0.6 * wake) + 1.5 * wake * noise, "v": 3 * wake * np.roll(noise, 1, 0),
              "w": 3 * wake * np.roll(noise, 1, 1), "p": -200 * wake + 20 * wake * np.roll(noise, 1, 2)}
    fields["cp"] = fields["p"] / 551.25
    fields["q"] = 1e3 * wake * noise ** 2
    mask = (((x - 1) / 2.2) ** 2 + (y / 0.9) ** 2 + ((z - 0.6) / 0.6) ** 2 < 1).astype(np.float32)
    v, f = uv_sphere(400, 500, r=1.0)
    n = len(v)
    r = make_cfd3d("x", rng)
    r.volume = VolumeData(origin=(-2, 0, 0), spacing=(10 / nx, 1.5 / ny, 2 / nz), fields=fields, mask=mask,
                          mirrored=True)
    r.surface = [SurfaceData3D("car", v, f, cp=np.cos(v[:, 0]) + 0.01 * rng.standard_normal(n),
                               cf=np.abs(np.sin(v[:, 1])) * 3e-3, tau=np.cos(v) * 1e-3,
                               sensitivity=np.sin(3 * v[:, 2]) * 1e-4)]
    d = proj.add_design(make_design3d())
    s = proj.add_simulation(make_sim(d))
    r.sim_id = s.id
    t0 = time.perf_counter()
    proj.save_result(s.id, r)
    t_save = time.perf_counter() - t0
    size = (proj.sim_dir(s.id) / "arrays.npz").stat().st_size
    t0 = time.perf_counter()
    back = proj.load_result(s.id)
    t_load = time.perf_counter() - t0
    print(f"\n3D result: {size / 1e6:.1f} MB on disk, save {t_save:.2f} s, load {t_load:.2f} s "
          f"({x.size / 1e6:.2f} M cells, {n} surface vertices)")
    raw = sum(a.size for a in fields.values()) * 4 + mask.size * 4 + n * (3 * 4 + 4 + 4 + 12 + 4) + f.size * 8
    assert size < 0.35 * raw and size < 30e6
    assert back.volume.fields["u"].shape == (nz, ny, nx) and len(back.surface[0].cp) == n
