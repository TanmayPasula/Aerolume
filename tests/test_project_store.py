"""Project store: round trips of every model type, robustness, concurrency, recent list."""
from __future__ import annotations

import json
import math
import threading

import numpy as np
import pytest

from aerolume.model import (
    AirfoilSettings,
    Body,
    CFDResult,
    CFDSettings,
    Design,
    FieldData,
    FlowConditions,
    Fluid,
    GroundConfig,
    PolarPoint,
    PolarResult,
    Simulation,
    Stat,
    SurfaceData,
)
from aerolume.project import (
    Project,
    ProjectCorruptError,
    ProjectExistsError,
    ProjectNotFoundError,
    add_recent,
    clear_recent,
    recent_projects,
    remove_recent,
)

# --- builders -------------------------------------------------------------------

def make_design(name: str = "Wing", n: int = 60) -> Design:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.c_[0.5 + 0.5 * np.cos(th), 0.06 * np.sin(th)]
    wheel = np.c_[0.2 + 0.05 * np.cos(th), -0.3 + 0.05 * np.sin(th)]
    return Design(name=name, kind="vehicle", bodies=[Body("main", pts), Body("wheel", wheel, role="wheel",
                                                                            omega=-60.0, center=(0.2, -0.3))],
                  length_ref=1.0, force_ref=0.12, moment_ref=(0.25, 0.0),
                  source={"type": "test", "params": {"n": n}}, notes="ünïcödé ✓")


def make_sim(design: Design, name: str = "Run", analysis: str = "cfd") -> Simulation:
    return Simulation(name=name, design_id=design.id, analysis=analysis,
                      conditions=FlowConditions(Fluid("Water", 998.0, 1e-3), speed=4.5, alpha_deg=3.0),
                      ground=GroundConfig(enabled=True, moving=False),
                      cfd=CFDSettings(quality="fine", duration=40.0, turbulence="les", reynolds_cap=5e4),
                      airfoil=AirfoilSettings(alpha_start=-2, alpha_end=8, ncrit=7.0), notes="n")


def make_cfd_result(sim_id: str, rng: np.random.Generator, ny: int = 40, nx: int = 96) -> CFDResult:
    t = np.linspace(0.0, 30.0, 3001)
    hist = {"t": t, "cd": 1.3 + 0.01 * rng.standard_normal(t.size), "cl": np.sin(1.3 * t) * 0.4 + 1e-12,
            "cm": -0.1 + 0.001 * rng.standard_normal(t.size)}
    s = np.linspace(0, 2.1, 200)
    surf = [SurfaceData("main", s, np.cos(s), np.sin(s), -np.sin(s), np.cos(s), 1 - 4 * np.sin(s) ** 2,
                        0.003 * np.cos(s)),
            SurfaceData("wheel", s[:50], s[:50], s[:50], s[:50], s[:50], s[:50], s[:50], sensitivity=s[:50] * 7)]
    fields = {"u": rng.normal(4.5, 1.0, (ny, nx)), "v": rng.normal(0, 0.5, (ny, nx)),
              "p": rng.normal(0, 3e5, (ny, nx)),           # beyond float16 range on purpose
              "vorticity": rng.normal(0, 1e-3, (ny, nx))}  # tiny values on purpose
    mask = np.zeros((ny, nx))
    mask[ny // 3: ny // 2, nx // 3: nx // 2] = 1.0
    mask[ny // 3, nx // 3] = 0.37
    return CFDResult(sim_id=sim_id, cd=Stat(1.3, 0.01, 1.27, 1.33, 0.0004), cl=Stat(0.0, 0.28, -0.4, 0.4, 0.002),
                     cm=Stat(-0.1, 0.001, -0.11, -0.09, float("nan")), drag=4.1, lift=-0.2, moment=0.03,
                     drag_area=0.156, power=18.4, strouhal=0.207, re_real=4.5e6, re_sim=5e4, grid=(nx, ny),
                     cells_per_ref=96.0, t_end=30.0, t_avg_start=10.0, converged=True, wall_seconds=12.5,
                     device="cuda", precision="float32", turbulence="les-wale", notes=["note"],
                     warnings=["capped Re"], per_body={"main": {"cd": 1.2, "cl": 0.0, "cm": -0.1}},
                     bodies_wind=[Body("main", np.array([[0, 0], [1, 0], [0.5, 0.2]]))], history=hist,
                     surface=surf,
                     mean_fields=FieldData((-1.0, 3.0, -1.0, 0.6667), fields, mask, "mean"),
                     instant_fields=FieldData((-1.0, 3.0, -1.0, 0.6667), {"dye": rng.random((ny, nx))}, None,
                                              "instant"))


def make_polar(sim_id: str) -> PolarResult:
    xs = np.linspace(0, 1, 30)
    pts = [PolarPoint(alpha=a, cl=0.1 * a, cd=0.01, cdp=0.004, cdf=0.006, cm=-0.05, xtr_upper=0.4,
                      xtr_lower=0.9, converged=a != 3, x=xs, y=xs * 0.1, cp=xs * a, cp_inviscid=xs, cf=xs * 0.01)
           for a in range(-2, 6)]
    pts[0].cp = None
    return PolarResult(sim_id=sim_id, re=1e6, mach=0.1, ncrit=7.0, points=pts, wall_seconds=3.0,
                       warnings=["point 3 failed"])


@pytest.fixture
def proj(tmp_path):
    return Project.create(tmp_path / "Demo", "Demo project")


# --- round trips ---------------------------------------------------------------

def test_create_layout_and_suffix(tmp_path):
    p = Project.create(tmp_path / "My Car")
    assert p.path.name == "My Car.aerolume" and p.name == "My Car"
    for sub in ("project.json", "designs", "sims", "reports"):
        assert (p.path / sub).exists()
    meta = json.loads((p.path / "project.json").read_text(encoding="utf-8"))
    assert meta["format"] == "aerolume-project" and meta["schema"] == 1
    assert Project.open(tmp_path / "My Car").path == p.path           # suffix optional on open
    assert Project.open(p.path / "project.json").name == "My Car"


def test_round_trip_designs_and_simulations(proj):
    d1, d2 = make_design("A"), make_design("B", 40)
    proj.add_design(d1)
    proj.add_design(d2)
    s1, s2, s3 = make_sim(d2, "z-first"), make_sim(d1, "a-second", "polar"), make_sim(d2, "m-third")
    for s in (s1, s2, s3):
        proj.add_simulation(s)
    proj.comparisons = [{"name": "cmp", "sim_ids": [s1.id, s2.id], "baseline": s1.id}]
    proj.settings = {"units": {"speed": "km/h"}}
    proj.save()

    q = Project.open(proj.path)
    assert q.name == "Demo project" and q.load_warnings == []
    assert list(q.designs) == [d1.id, d2.id]                          # insertion order kept
    assert list(q.simulations) == [s1.id, s2.id, s3.id]
    for d in (d1, d2):
        e = q.designs[d.id]
        assert e.to_dict() == d.to_dict()
        assert e.bodies[1].center == (0.2, -0.3) and e.moment_ref == (0.25, 0.0)
        np.testing.assert_array_equal(e.bodies[0].points, d.bodies[0].points)
    for s in (s1, s2, s3):
        assert q.simulations[s.id].to_dict() == s.to_dict()
    assert q.simulations[s1.id].conditions.fluid.density == 998.0
    assert q.comparisons == proj.comparisons and q.settings == proj.settings


def test_round_trip_cfd_result_with_arrays(proj):
    rng = np.random.default_rng(0)
    d = proj.add_design(make_design())
    s = proj.add_simulation(make_sim(d))
    proj.save()
    assert not proj.has_result(s.id) and proj.load_result(s.id) is None and proj.result_summary(s.id) is None
    r = make_cfd_result(s.id, rng)
    proj.save_result(s.id, r)
    assert proj.has_result(s.id)

    back = Project.open(proj.path).load_result(s.id)
    assert isinstance(back, CFDResult)
    assert back.cd == r.cd and back.cl == r.cl
    assert math.isnan(back.cm.stderr) and back.cm.mean == r.cm.mean
    assert back.grid == r.grid and back.strouhal == r.strouhal and back.per_body == r.per_body
    assert back.warnings == ["capped Re"] and back.converged is True
    np.testing.assert_array_equal(back.bodies_wind[0].points, r.bodies_wind[0].points)
    # histories are float64 and exact
    assert set(back.history) == {"t", "cd", "cl", "cm"}
    for k, v in r.history.items():
        assert back.history[k].dtype == np.float64
        np.testing.assert_array_equal(back.history[k], v)
    # surfaces exact, optional sensitivity preserved / absent
    assert [sd.body for sd in back.surface] == ["main", "wheel"]
    np.testing.assert_array_equal(back.surface[0].cp, r.surface[0].cp)
    assert back.surface[0].sensitivity is None
    np.testing.assert_array_equal(back.surface[1].sensitivity, r.surface[1].sensitivity)
    # fields: float16 storage with per-field scale -> 1e-3 of each field's peak
    mf = back.mean_fields
    assert mf.label == "mean" and mf.extent == r.mean_fields.extent
    for k, v in r.mean_fields.fields.items():
        got = mf.fields[k]
        assert got.shape == v.shape and got.dtype == np.float32
        assert np.all(np.isfinite(got))
        assert np.abs(got - v).max() <= 1e-3 * np.abs(v).max()
    np.testing.assert_allclose(mf.mask, r.mean_fields.mask, atol=1e-3)
    assert back.instant_fields.label == "instant" and back.instant_fields.mask is None
    assert set(back.instant_fields.fields) == {"dye"}
    # fields really are stored as float16
    with np.load(proj.sim_dir(s.id) / "arrays.npz") as z:
        assert z["mean_fields/p"].dtype == np.float16 and z["history/t"].dtype == np.float64
    # cheap summary and scalar-only load
    summ = proj.result_summary(s.id)
    assert summ["type"] == "cfd" and summ["cd"]["mean"] == 1.3 and "history" not in summ
    light = proj.load_result(s.id, arrays=False)
    assert light.history == {} and light.mean_fields is None and light.cd == r.cd


def test_round_trip_polar_result(proj):
    d = proj.add_design(make_design())
    s = proj.add_simulation(make_sim(d, analysis="polar"))
    pr = make_polar(s.id)
    proj.save_result(s.id, pr)
    back = Project.open(proj.path).load_result(s.id)
    assert isinstance(back, PolarResult)
    assert (back.re, back.mach, back.ncrit, back.warnings) == (1e6, 0.1, 7.0, ["point 3 failed"])
    assert len(back.points) == len(pr.points)
    for a, b in zip(back.points, pr.points, strict=True):
        assert (a.alpha, a.cl, a.cd, a.cm, a.converged) == (b.alpha, b.cl, b.cd, b.cm, b.converged)
        for k in ("x", "y", "cp", "cp_inviscid", "cf"):
            va, vb = getattr(a, k), getattr(b, k)
            if vb is None:
                assert va is None
            else:
                np.testing.assert_array_equal(va, vb)
    assert proj.result_summary(s.id)["points"][1]["cp"] is None      # arrays not in the summary
    np.testing.assert_array_equal(back.table()["alpha"], pr.table()["alpha"])


def test_result_overwrite_and_summary_refresh(proj):
    d = proj.add_design(make_design())
    s = proj.add_simulation(make_sim(d, analysis="polar"))
    proj.save_result(s.id, make_polar(s.id))
    assert proj.result_summary(s.id)["re"] == 1e6
    pr = make_polar(s.id)
    pr.re = 2e6
    proj.save_result(s.id, pr)
    assert proj.result_summary(s.id)["re"] == 2e6
    assert proj.load_result(s.id).re == 2e6
    proj.delete_result(s.id)
    assert not proj.has_result(s.id) and proj.result_summary(s.id) is None


# --- editing ----------------------------------------------------------------------

def test_remove_and_duplicate(proj):
    rng = np.random.default_rng(1)
    d1, d2 = proj.add_design(make_design("A")), proj.add_design(make_design("B"))
    s1 = proj.add_simulation(make_sim(d1, "one"))
    s2 = proj.add_simulation(make_sim(d2, "two"))
    s3 = proj.add_simulation(make_sim(d2, "three"))
    proj.save()
    proj.save_result(s1.id, make_cfd_result(s1.id, rng, 8, 16))
    s1.status = "done"

    dup = proj.duplicate_simulation(s1.id)
    assert dup.id != s1.id and dup.name == "one (copy)" and dup.status == "new"
    assert list(proj.simulations) == [s1.id, dup.id, s2.id, s3.id]
    assert dup.cfd.to_dict() == s1.cfd.to_dict() and not proj.has_result(dup.id)
    dup.cfd.duration = 99.0                                          # deep copy
    assert s1.cfd.duration == 40.0
    proj.save()

    folder = proj.sim_dir(s1.id)
    assert folder.is_dir()
    proj.remove_simulation(s1.id)
    assert not folder.exists() and s1.id not in proj.simulations
    assert s1.id not in Project.open(proj.path).simulations          # index rewritten at once

    removed = proj.remove_design(d2.id)
    assert set(removed) == {s2.id, s3.id}
    assert not (proj.path / "designs" / f"{d2.id}.json").exists()
    assert not proj.sim_dir(s2.id).exists()
    q = Project.open(proj.path)
    assert list(q.designs) == [d1.id] and list(q.simulations) == [dup.id] and q.load_warnings == []

    with pytest.raises(KeyError):
        proj.add_simulation(make_sim(d2))                            # design gone
    with pytest.raises(KeyError):
        proj.save_result(s1.id, make_polar(s1.id))                   # simulation gone
    with pytest.raises(ValueError):
        proj.add_design(d1)                                          # duplicate id


def test_save_only_rewrites_changed_files(proj):
    d = proj.add_design(make_design())
    proj.save()
    f = proj.path / "designs" / f"{d.id}.json"
    before = f.stat().st_mtime_ns
    idx_before = (proj.path / "project.json").read_text(encoding="utf-8")
    proj.save()
    assert f.stat().st_mtime_ns == before
    assert (proj.path / "project.json").read_text(encoding="utf-8") == idx_before
    d.notes = "edited in place"                                      # no update_design call
    proj.save()
    assert json.loads(f.read_text(encoding="utf-8"))["notes"] == "edited in place"


# --- robustness ---------------------------------------------------------------------

def test_create_refuses_non_empty_folder(tmp_path):
    target = tmp_path / "x.aerolume"
    target.mkdir()
    Project.create(target)                                           # empty folder is fine
    with pytest.raises(ProjectExistsError):
        Project.create(target)
    other = tmp_path / "docs.aerolume"
    other.mkdir()
    (other / "thesis.docx").write_bytes(b"precious")
    with pytest.raises(ProjectExistsError):
        Project.create(other)
    assert (other / "thesis.docx").read_bytes() == b"precious"
    (tmp_path / "file.aerolume").write_text("x")
    with pytest.raises(ProjectExistsError):
        Project.create(tmp_path / "file.aerolume")


def test_missing_and_corrupt_files(tmp_path, proj):
    with pytest.raises(ProjectNotFoundError):
        Project.open(tmp_path / "nowhere")
    empty = tmp_path / "empty.aerolume"
    empty.mkdir()
    with pytest.raises(ProjectNotFoundError, match="project.json"):
        Project.open(empty)
    idx = proj.path / "project.json"
    good = idx.read_text(encoding="utf-8")
    idx.write_text(good[: len(good) // 2], encoding="utf-8")
    with pytest.raises(ProjectCorruptError, match="not valid JSON"):
        Project.open(proj.path)
    idx.write_text(json.dumps({"hello": 1}), encoding="utf-8")
    with pytest.raises(ProjectCorruptError, match="not an Aerolume project"):
        Project.open(proj.path)
    idx.write_text(good, encoding="utf-8")

    rng = np.random.default_rng(2)
    d = proj.add_design(make_design())
    s = proj.add_simulation(make_sim(d))
    proj.save()
    proj.save_result(s.id, make_cfd_result(s.id, rng, 8, 16))
    npz = proj.sim_dir(s.id) / "arrays.npz"
    data = npz.read_bytes()
    npz.write_bytes(data[: len(data) // 3])
    with pytest.raises(ProjectCorruptError, match="cannot be read"):
        proj.load_result(s.id)
    assert proj.load_result(s.id, arrays=False).cd.mean == 1.3      # scalars still readable
    npz.unlink()
    with pytest.raises(ProjectCorruptError, match="missing"):
        proj.load_result(s.id)
    # arrays from a different save (crash between the two writes) are detected
    proj.save_result(s.id, make_cfd_result(s.id, rng, 8, 16))
    old = npz.read_bytes()
    proj.save_result(s.id, make_cfd_result(s.id, rng, 8, 16))
    npz.write_bytes(old)
    with pytest.raises(ProjectCorruptError, match="do not belong"):
        proj.load_result(s.id)
    (proj.sim_dir(s.id) / "result.json").write_text("{", encoding="utf-8")
    with pytest.raises(ProjectCorruptError):
        proj.load_result(s.id)
    with pytest.raises(ProjectCorruptError):
        proj.result_summary(s.id)


def test_forward_compatible_loading(proj):
    d = proj.add_design(make_design())
    s = proj.add_simulation(make_sim(d))
    proj.save()
    proj.save_result(s.id, make_polar(s.id))
    idx = proj.path / "project.json"
    meta = json.loads(idx.read_text(encoding="utf-8"))
    meta.update(schema=99, future_feature={"x": 1})
    idx.write_text(json.dumps(meta), encoding="utf-8")
    for f, extra in ((proj.path / "designs" / f"{d.id}.json", {"hologram": True}),
                     (proj.sim_dir(s.id) / "simulation.json", {"quantum": 3})):
        obj = json.loads(f.read_text(encoding="utf-8"))
        obj.update(extra)
        f.write_text(json.dumps(obj), encoding="utf-8")
    rj = proj.sim_dir(s.id) / "result.json"
    obj = json.loads(rj.read_text(encoding="utf-8"))
    obj["result"]["new_metric"] = 1.0
    obj["result"]["points"][0]["new_point_field"] = 2.0
    rj.write_text(json.dumps(obj), encoding="utf-8")

    q = Project.open(proj.path)
    assert any("newer Aerolume" in w for w in q.load_warnings)
    assert q.designs[d.id].name == d.name and q.simulations[s.id].name == s.name
    assert isinstance(q.load_result(s.id), PolarResult)
    q.name = "Renamed"
    q.save()
    meta2 = json.loads(idx.read_text(encoding="utf-8"))
    assert meta2["future_feature"] == {"x": 1} and meta2["name"] == "Renamed"


def test_recovery_of_unindexed_items_and_stale_status(proj):
    d = proj.add_design(make_design())
    s = proj.add_simulation(make_sim(d))
    s.status = "running"
    proj.save()
    idx = proj.path / "project.json"
    meta = json.loads(idx.read_text(encoding="utf-8"))
    meta["designs"], meta["simulations"] = [], ["ghost"]
    idx.write_text(json.dumps(meta), encoding="utf-8")
    q = Project.open(proj.path)
    assert d.id in q.designs and s.id in q.simulations
    assert q.simulations[s.id].status == "new"                       # no live process, no result
    text = " ".join(q.load_warnings)
    assert "Recovered design" in text and "Recovered simulation" in text and "ghost" in text
    q.save()
    assert json.loads((q.sim_dir(s.id) / "simulation.json").read_text(encoding="utf-8"))["status"] == "new"
    (proj.path / "designs" / f"{d.id}.json").write_text("garbage", encoding="utf-8")
    r = Project.open(proj.path)
    assert d.id not in r.designs and any("could not be read" in w for w in r.load_warnings)


# --- concurrency ------------------------------------------------------------------------

def test_concurrent_result_saves_while_editing(proj):
    """A worker thread saves results while the GUI thread adds and saves designs."""
    rng = np.random.default_rng(3)
    d0 = proj.add_design(make_design("base"))
    sims = [proj.add_simulation(make_sim(d0, f"s{i}")) for i in range(4)]
    proj.save()
    results = {s.id: make_cfd_result(s.id, rng, 16, 32) for s in sims}
    errors: list[BaseException] = []

    def worker():
        try:
            for _ in range(5):
                for sid, r in results.items():
                    proj.save_result(sid, r)
        except BaseException as exc:   # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    th = threading.Thread(target=worker)
    th.start()
    added = []
    for i in range(25):
        d = proj.add_design(make_design(f"gui{i}", 30))
        added.append(d.id)
        proj.add_simulation(make_sim(d, f"g{i}"))
        proj.save()
    th.join(timeout=120)
    assert not th.is_alive() and not errors
    q = Project.open(proj.path)
    assert q.load_warnings == []
    assert list(q.designs)[1:] == added
    for sid, r in results.items():
        back = q.load_result(sid)
        np.testing.assert_array_equal(back.history["cd"], r.history["cd"])
    leftovers = [p for p in proj.path.rglob("*") if p.name.endswith(".tmp") or ".trash-" in p.name]
    assert leftovers == []


# --- recent projects -------------------------------------------------------------------

def test_recent_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "cfg"))
    assert recent_projects() == []
    a = Project.create(tmp_path / "A")
    b = Project.create(tmp_path / "B")
    add_recent(a.path)
    add_recent(b.path)
    add_recent(a.path)                                               # moves to the top, no duplicate
    assert recent_projects() == [a.path, b.path]
    add_recent(tmp_path / "gone.aerolume")
    assert recent_projects() == [a.path, b.path]                     # missing ones hidden
    assert len(recent_projects(existing_only=False)) == 3
    remove_recent(b.path)
    assert recent_projects() == [a.path]
    (tmp_path / "cfg" / "recent_projects.json").write_text("{broken", encoding="utf-8")
    assert recent_projects() == []                                   # never crashes the Start page
    add_recent(a.path)
    clear_recent()
    assert recent_projects() == []


def test_description_round_trips_and_reaches_the_index(tmp_path):
    """The Start page lists examples with this one-line description (project.json "description")."""
    p = Project.create(tmp_path / "Described", "Described")
    assert p.description == ""
    p.description = "Two side profiles at 30 m/s."
    p.save()
    meta = json.loads((p.path / "project.json").read_text(encoding="utf-8"))
    assert meta["description"] == "Two side profiles at 30 m/s."
    assert Project.open(p.path).description == "Two side profiles at 30 m/s."


# --- review round: damaged entries cost only themselves; caps are pinned --------------------

def _two_of_each(proj: Project) -> tuple[list[Design], list[Simulation]]:
    ds = [proj.add_design(make_design(f"D{i}")) for i in range(2)]
    ss = [proj.add_simulation(make_sim(ds[i], f"S{i}")) for i in range(2)]
    proj.save()
    return ds, ss


@pytest.mark.parametrize("damage", ["design_list", "sim_string", "conditions_list"])
def test_a_damaged_item_file_is_skipped_with_a_warning(proj, damage):
    """A JSON list where a design belongs, a string for a simulation, a list for its
    conditions: each raised AttributeError out of Project.open and locked the user out."""
    ds, ss = _two_of_each(proj)
    if damage == "design_list":
        target = proj.path / "designs" / f"{ds[0].id}.json"
        target.write_text("[1, 2]", encoding="utf-8")
    elif damage == "sim_string":
        target = proj.sim_dir(ss[0].id) / "simulation.json"
        target.write_text('"hello"', encoding="utf-8")
    else:
        target = proj.sim_dir(ss[0].id) / "simulation.json"
        doc = json.loads(target.read_text(encoding="utf-8"))
        doc["conditions"] = [1, 2]
        target.write_text(json.dumps(doc), encoding="utf-8")
    p = Project.open(proj.path)
    assert p.load_warnings and any(target.name in w for w in p.load_warnings)
    if damage == "design_list":
        assert ds[1].id in p.designs and ds[0].id not in p.designs
        assert ss[1].id in p.simulations
    else:
        assert ss[1].id in p.simulations and ss[0].id not in p.simulations
        assert len(p.designs) == 2


def test_damaged_project_settings_and_comparisons_are_reset_with_a_warning(proj):
    _two_of_each(proj)
    meta = json.loads((proj.path / "project.json").read_text(encoding="utf-8"))
    meta["settings"] = [1, 2]
    meta["comparisons"] = "abc"
    (proj.path / "project.json").write_text(json.dumps(meta), encoding="utf-8")
    p = Project.open(proj.path)
    assert p.settings == {} and p.comparisons == []          # 'abc' used to become ['a', 'b', 'c']
    assert any("settings" in w for w in p.load_warnings)
    assert any("comparisons" in w for w in p.load_warnings)
    assert len(p.designs) == 2 and len(p.simulations) == 2
    meta["simulations"] = "abc"                               # an index that is not a list
    (proj.path / "project.json").write_text(json.dumps(meta), encoding="utf-8")
    assert len(Project.open(proj.path).simulations) == 2      # recovered from disk


def test_a_capped_simulation_without_a_cap_is_pinned_to_its_results_reynolds_number(proj):
    """Projects from before the cap was stored: the planner chose it from the grid, so a run at
    another quality would simulate another flow. Opening pins it to the result's re_sim."""
    d = proj.add_design(make_design())
    sim = proj.add_simulation(make_sim(d))
    sim.cfd.reynolds_mode, sim.cfd.reynolds_cap = "capped", None
    other = proj.add_simulation(make_sim(d, "no result"))
    other.cfd.reynolds_mode, other.cfd.reynolds_cap = "capped", None
    proj.save()
    res = make_cfd_result(sim.id, np.random.default_rng(1))
    res.re_sim = 6400.0
    proj.save_result(sim.id, res)
    p = Project.open(proj.path)
    assert p.simulations[sim.id].cfd.reynolds_cap == 6400.0
    assert p.simulations[other.id].cfd.reynolds_cap is None   # nothing to pin it to
    p.save()
    assert Project.open(proj.path).simulations[sim.id].cfd.reynolds_cap == 6400.0


def test_results_saved_before_new_fields_load_with_defaults(proj):
    """Stat.stderr_reliable and CFDResult.strouhal_length are additions: old files load."""
    d = proj.add_design(make_design())
    sim = proj.add_simulation(make_sim(d))
    proj.save()
    proj.save_result(sim.id, make_cfd_result(sim.id, np.random.default_rng(2)))
    f = proj.sim_dir(sim.id) / "result.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    for k in ("cd", "cl", "cm"):
        doc["result"][k].pop("stderr_reliable", None)
    doc["result"].pop("strouhal_length", None)
    f.write_text(json.dumps(doc), encoding="utf-8")
    r = Project.open(proj.path).load_result(sim.id, arrays=False)
    assert r.cd.stderr_reliable is True and math.isnan(r.strouhal_length)
