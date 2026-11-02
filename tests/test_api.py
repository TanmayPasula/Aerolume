"""The public scripting API (``import aerolume as al``, DESIGN.md section 8), end to end.

The CFD runs are tiny CPU grids (Re 200, 12 cells per chord, two convective
times: a few seconds each) -- the point is the API's bookkeeping (what is
recorded in the project, reuse, cancel, errors), which the solver tests do not
cover. One 3D run on the GPU is marked slow.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import aerolume as al
from aerolume import api
from aerolume.model import PolarPoint, PolarResult
from aerolume.project import Project as Store

EXAMPLES3D = Path(__file__).resolve().parents[1] / "examples" / "geometry3d"
TINY = {"re": 200, "device": "cpu", "cells_per_ref": 12, "duration": 2}


@pytest.fixture(autouse=True)
def private_calibration(tmp_path, monkeypatch):
    """Solver runs record their throughput in the planner's in-memory calibration: start each
    test from the reference model so estimates (and other test modules) are not affected."""
    from aerolume.simulation import plan as planmod
    for name in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)


@pytest.fixture
def proj(tmp_path):
    return al.new_project(tmp_path / "API test")


@pytest.fixture(scope="module")
def tiny_study(tmp_path_factory):
    """One project with a NACA 0012 and a tiny two-angle CFD sweep, shared by the read-only tests."""
    from aerolume.simulation import plan as planmod
    saved = {n: getattr(planmod, n) for n in ("_RATE_CACHE", "_SPEED_CACHE", "_LOOP_CACHE")}
    p = al.new_project(tmp_path_factory.mktemp("tiny") / "Tiny")
    d = p.add_naca("0012")
    sweep = p.sweep(d, alpha=[0, 4], **TINY)
    for n, v in saved.items():
        setattr(planmod, n, v)
    return p, d, sweep


def test_import_is_fast_and_torch_free():
    code = ("import sys, aerolume as al; al.new_project; al.Project; al.AerolumeError; "
            "print('torch' in sys.modules, 'PySide6' in sys.modules, 'vtk' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "False"]
    assert "new_project" in dir(al) and al.version() == al.__version__


def test_designs_are_recorded_named_and_reused(proj, tmp_path):
    d = proj.add_naca("2412", chord=0.5)
    assert d.name == "NACA 2412" and d.kind == "airfoil" and d.dimension == 2
    assert d.references["length_ref"] == pytest.approx(0.5)
    assert proj.add_naca("NACA 2412", chord=0.5).id == d.id             # same shape and name: reused
    other = proj.add_naca("2412", chord=1.0)                          # same name, other shape: new design
    assert other.id != d.id and other.name == "NACA 2412 2"
    car = proj.add_template("ahmed", slant_angle=30)
    assert car.name == "Ahmed body (2D) · slant angle 30°" and car.kind == "vehicle"
    sphere = proj.import_mesh(EXAMPLES3D / "sphere_d100mm_binary.stl", kind="generic")
    assert sphere.is_3d and sphere.summary()["triangles"] > 100
    assert sphere.references["ref_area"] == pytest.approx(np.pi * 0.05 ** 2, rel=0.05)
    wing = proj.add_wing(span=2, root_chord=0.3, tip_chord=0.18, sweep_deg=5)
    s = wing.summary()
    assert s["aspect_ratio"] == pytest.approx(2 ** 2 / 0.48) and s["sweep_c4_deg"] == pytest.approx(5)
    # lookups by name (any case), id and id prefix
    assert proj.design("naca 2412 2").id == other.id
    assert proj.design(car.id[:6]).id == car.id
    with pytest.raises(al.NotFoundError, match="Designs: 'NACA 2412'"):
        proj.design("nope")
    # recorded in the store exactly like the GUI's designs: a reopened project has them all
    again = Store.open(proj.path)
    assert not again.load_warnings
    assert [x.name for x in again.designs.values()] == [x.name for x in proj.designs]
    assert len(again.designs[sphere.id].meshes[0].faces) == len(sphere.raw.meshes[0].faces)
    # edits through the wrappers are recorded too
    d.rename("Root section")
    assert Store.open(proj.path).designs[d.id].name == "Root section"
    copy = sphere.copy()
    assert copy.id != sphere.id and len(copy.raw.meshes[0].vertices) == len(sphere.raw.meshes[0].vertices)
    sphere.transform(scale=2.0)
    assert sphere.raw.bounds3()[1] - sphere.raw.bounds3()[0] == pytest.approx(0.2, rel=1e-6)
    assert d.export(tmp_path / "root.dat").is_file() and sphere.export(tmp_path / "s.stl").stat().st_size > 1000


def test_cfd_run_is_recorded_like_the_gui_and_reused(proj):
    d = proj.add_naca("0012")
    reports: list[tuple[float, str]] = []
    r = proj.run_cfd(d, alpha=4, progress=lambda f, m: reports.append((f, m)), **TINY)
    assert r.kind == "cfd2d" and not r.reused and r.status == "done"
    assert r.name == "CFD · Re 200 · α 4°"        # set by re=: named by it (simulation_name)
    assert np.isfinite(r.cd) and np.isfinite(r.cl) and r.cd > 0
    assert r.stats["cd"].mean == r.cd and np.isfinite(r.stderr("cd"))
    assert r.ld == pytest.approx(r.cl / r.cd)
    assert r.re_sim == pytest.approx(200) and r.forces["drag"] > 0 and r.units["drag"] == "N/m"
    assert {row["key"] for row in r.table()} >= {"cd", "cl", "cm", "ld", "drag", "power"}
    assert r.summary()["converged"] in (True, False) and r.summary()["grid"]
    assert reports and reports[-1][0] == pytest.approx(1.0) and all(0 <= f <= 1 for f, _ in reports)
    # the store holds what the GUI would have: the simulation, its status and the result
    store = Store.open(proj.path)
    (sim,) = store.simulations.values()
    assert sim.status == "done" and sim.finished and sim.design_id == d.id and sim.cfd.device == "cpu"
    assert store.load_result(sim.id).cd.mean == pytest.approx(r.cd)
    # the same call again returns the stored result; rerun=True runs the same simulation again
    again = proj.run_cfd(d, alpha=4, **TINY)
    assert again.reused and again.sim_id == r.sim_id and len(proj.simulations) == 1
    fresh = proj.run_cfd(d, alpha=4, rerun=True, **TINY)
    assert not fresh.reused and fresh.sim_id == r.sim_id and len(proj.simulations) == 1
    assert proj.result(r.name).cd == pytest.approx(fresh.cd)
    assert proj.run(r.name).reused
    assert repr(proj) == "<Project 'API test': 1 design, 1 simulation>"      # counts in the right number
    assert "L/D" in str(r) and "(symmetry)" not in str(r)                     # an airfoil lifts; 2D has no CS


def _fake_3d(proj, design, sid: str, cd: float = 0.5, *, symmetric: bool = True):
    """A stored-looking 3D result of ``design`` (report fixtures: no solver run)."""
    import report_fixtures3d as rf3
    sim = rf3.cfd3d_sim(design.raw, f"CFD 3D · {sid}", sid)
    raw = rf3.vehicle_result(sim, design.raw, cd=cd)       # CS, C_roll, Cn, side, roll, yaw exactly 0
    raw.symmetry_plane = symmetric
    return api.Result(proj, sim, raw)


def test_3d_text_says_symmetry_and_drops_ld_of_bluff_bodies(proj):
    sphere = proj.add_sphere(0.2)
    r = _fake_3d(proj, sphere, "a")
    assert r.kind == "cfd3d" and not r.lifting
    rows = {row["key"]: row for row in r.table()}
    assert "ld" not in rows and "ld" not in r.summary() and np.isfinite(r.ld)     # still there on request
    assert {k for k, row in rows.items() if row["by_symmetry"]} == {"cs", "c_roll", "cn", "side", "roll", "yaw"}
    assert r.summary()["symmetry_plane"] is True
    text = str(r)
    assert "L/D" not in text and text.count("0 (symmetry)") == 6
    cs_line = next(line for line in text.splitlines() if line.startswith("CS "))
    assert cs_line.split() == ["CS", "0", "(symmetry)"]                         # no "0  0" read as measured
    # a full-domain run measured its zeros: they print as numbers
    full = _fake_3d(proj, sphere, "b", symmetric=False)
    assert "(symmetry)" not in str(full) and not any(row["by_symmetry"] for row in full.table())
    # a comparison of bluff bodies has no L/D row; their lift row says how to read its sign
    comp = api.Comparison([r, _fake_3d(proj, sphere, "c", cd=0.45)])
    keys = [row["key"] for row in comp.rows]
    assert "ld" not in keys and "L/D" not in str(comp)
    assert repr(comp).startswith("<Comparison of 2 results")


def test_3d_text_of_a_wing_keeps_ld(proj):
    wing = proj.add_wing(span=1.0, root_chord=0.2)
    r = _fake_3d(proj, wing, "w")
    assert r.lifting and "ld" in r.summary() and "L/D" in str(r)


def test_sweep_compare_plot_and_export(tiny_study, tmp_path):
    p, d, sweep = tiny_study
    assert len(sweep) == 2 and sweep.params == ["alpha"] and [v["alpha"] for v in sweep.values] == [0, 4]
    rows = sweep.table()
    assert {"alpha", "cd", "cd_stderr", "cl", "ld", "converged", "re_sim"} <= set(rows[0])
    assert sweep.best("cd").cd == min(r.cd for r in sweep)
    assert sweep.best("cl").alpha == 4
    assert "alpha" in str(sweep) and "converged" in str(sweep)
    png = sweep.plot("alpha", ["cl", "cd"], path=tmp_path / "sweep.png")
    assert png.is_file() and png.stat().st_size > 5000
    csv_path = sweep.export(tmp_path / "sweep.csv")
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("alpha,")
    cmp = p.compare(sweep)
    keys = [row["key"] for row in cmp.rows]
    assert keys[:2] == ["cd", "cl"] and "converged" in keys
    assert any("different angles" in w for w in cmp.warnings)
    table = cmp.table()
    assert table[0]["metric"].startswith("Drag coefficient") and any(k.endswith("Δ%") for k in table[0])
    assert "baseline" in str(cmp)
    assert cmp.plot("cd", tmp_path / "bars.png").is_file()
    # the sweep's points are ordinary simulations of the project, reused by a second sweep
    assert len(p.simulations) == 2
    again = p.sweep(d, alpha=[0, 4], **TINY)
    assert all(r.reused for r in again) and len(p.simulations) == 2


def test_results_plot_and_export_by_extension(tiny_study, tmp_path):
    _p, _d, sweep = tiny_study
    r = sweep[1]
    assert r.plot_kinds()[0] == "history"
    for kind in ("history", "cp", "field", "geometry"):
        assert r.plot(kind, tmp_path / f"{kind}.png").is_file()
    assert r.plot("field", field="vorticity").__class__.__name__ == "Figure"
    with pytest.raises(al.InputError, match="no 'polar' plot"):
        r.plot("polar")
    with pytest.raises(al.InputError, match="sensitivity"):
        r.plot("sensitivity")
    for name in ("r.csv", "r.vti", "r.npz", "r.png"):
        assert r.export(tmp_path / name).stat().st_size > 100
    assert r.export(tmp_path / "h.csv", "history").read_text(encoding="utf-8").startswith("t,")
    data = json.loads((r.export(tmp_path / "r.json")).read_text(encoding="utf-8"))
    assert data["sim_id"] == r.sim_id and data["kind"] == "cfd2d"
    # strict JSON for tools: no NaN, no numpy types
    for obj in (r.summary(), r.table(), sweep.table()):
        json.dumps(al.plain(obj), allow_nan=False)
    assert al.plain({"a": float("nan"), "b": np.float32(2.5), "c": (1, np.int64(2))}) == \
        {"a": None, "b": 2.5, "c": [1, 2]}
    with pytest.raises(al.InputError, match="Unknown export format"):
        r.export(tmp_path / "r.xyz")


def test_report_of_results(tiny_study, tmp_path):
    p, _d, sweep = tiny_study
    path = p.report(sweep, tmp_path / "study.html", title="API report")
    text = path.read_text(encoding="utf-8")
    assert path.suffix == ".html" and "API report" in text and sweep[0].name in text


def test_polar_wing_and_a_geometry_sweep(proj, tmp_path):
    d = proj.add_naca("2412")
    pol = proj.run_polar(d, re=1e6, alpha=(0, 4, 2))
    assert pol.kind == "polar" and pol.name == "Polar · Re 1.0×10⁶" and pol.converged
    assert list(pol.alpha) == [0, 2, 4] and pol.cl[2] > pol.cl[0] and np.all(pol.cd > 0)
    assert pol.re_sim == pytest.approx(1e6, rel=1e-9)
    best = pol.best()
    assert best["alpha"] == 4 and best["ld"] == pytest.approx(best["cl"] / best["cd"])
    assert pol.at(3)["cl"] == pytest.approx(0.5 * (pol.cl[1] + pol.cl[2]))
    assert pol.metrics["n_converged"] == 3 and pol.summary()["ld_max"] > 50
    assert pol.plot(path=tmp_path / "polar.png").is_file() and pol.export(tmp_path / "p.csv", "polar").is_file()
    with pytest.raises(al.InputError, match="outside"):
        pol.at(20)
    # finite wings: inviscid lattice in about a second, a taper study through a design factory
    w = proj.add_wing(span=2, root_chord=0.3, tip_chord=0.18)
    rw = proj.run_wing(w, speed=20, alpha=(0, 8, 2), viscous=False)
    assert rw.kind == "wing" and rw.name == "Wing polar · 20 m/s · inviscid" and rw.n_converged == 5
    assert 0.9 < rw.metrics["oswald"] <= 1.0 and rw.metrics["cl_alpha_per_deg"] > 0.07
    study = proj.sweep(lambda taper: proj.add_wing(span=2, root_chord=0.3, tip_chord=0.3 * taper),
                       taper=[0.5, 1.0], speed=20, alpha=(0, 8, 2), viscous=False)
    assert [v["taper"] for v in study.values] == [0.5, 1.0]
    assert all(len(r.raw.points) == 5 for r in study)          # alpha stayed the wing's own range
    assert study.best("oswald").design.name.endswith("taper 0.5")
    assert study.plot("alpha", "cl", path=tmp_path / "curves.png").is_file()       # overlaid curves
    assert study.plot("taper", "oswald", path=tmp_path / "e.png").is_file()


def test_errors_are_readable_api_errors(proj, tmp_path):
    d = proj.add_naca("0012")
    s3 = proj.add_sphere(0.1)
    cases = [
        (lambda: proj.add_naca("12"), al.InputError, "not a NACA"),
        (lambda: proj.run_cfd(d, speed=-1), al.InputError, "speed must be positive"),
        (lambda: proj.run_polar(s3), al.InputError, "cannot have a 'polar' analysis"),
        (lambda: proj.run_cfd(d, sped=3), al.InputError, "Did you mean 'speed'"),
        (lambda: proj.run_cfd(d, alpha=[0, 4]), al.InputError, "use sweep"),
        (lambda: proj.run_cfd(d, yaw=5), al.InputError, "needs a 3D design"),
        (lambda: proj.sweep(d, alpha=4), al.InputError, "Nothing to sweep"),
        (lambda: proj.add_template("spaceship"), al.NotFoundError, "unknown template"),
        (lambda: proj.import_mesh(tmp_path / "missing.stl"), al.NotFoundError, "No mesh file"),
        (lambda: al.open_project(tmp_path / "nothing here"), al.NotFoundError, "No project"),
        (lambda: al.new_project(proj.path), al.ProjectFileError, "already exists"),
        (lambda: proj.run_polar(d, alpha=[0, 1, 5]), al.InputError, "evenly spaced"),
    ]
    for fn, cls, text in cases:
        with pytest.raises(cls, match=text) as info:
            fn()
        assert isinstance(info.value, al.AerolumeError)
    # the standard exception families still work for callers that catch those
    with pytest.raises(ValueError):
        proj.add_naca("12")
    with pytest.raises(LookupError):
        proj.simulation("nope")
    assert not proj.simulations                                  # nothing half-recorded


def test_cancel_before_and_during_a_run(proj):
    d = proj.add_naca("0012")
    stop = threading.Event()
    stop.set()
    with pytest.raises(al.Cancelled, match="Stopped before"):
        proj.run_cfd(d, alpha=2, cancel=stop, **TINY)
    assert not proj.simulations
    stop = threading.Event()

    def progress(fraction: float, _message: str) -> None:
        if fraction > 0.3:
            stop.set()
    with pytest.raises(al.Cancelled) as info:
        proj.run_cfd(d, alpha=2, cancel=stop, progress=progress, **dict(TINY, duration=6))
    partial = info.value.result
    assert partial is not None and partial.cancelled and partial.status == "cancelled"
    store = Store.open(proj.path)
    (sim,) = store.simulations.values()
    assert sim.status == "cancelled" and store.has_result(sim.id)
    # a cancelled simulation is run again (not reused) by the next identical call
    r = proj.run_cfd(d, alpha=2, **dict(TINY, duration=6))
    assert not r.reused and r.sim_id == sim.id and r.status == "done"


def test_context_supplies_cancel_and_progress(proj):
    d = proj.add_naca("0012")
    seen: list[str] = []
    stop = threading.Event()
    with al.context(progress=lambda f, m: seen.append(m)):
        proj.run_cfd(d, alpha=1, **TINY)
    assert any("Done" in m or "finished" in m.lower() for m in seen)
    stop.set()
    with al.context(cancel=stop), pytest.raises(al.Cancelled):
        proj.run_cfd(d, alpha=3, **TINY)


def test_plan_estimates_without_recording(proj):
    d = proj.add_naca("2412")
    pl = proj.plan(d, speed=30, alpha=4)
    assert pl.analysis == "cfd" and pl.cells > 0 and len(pl.grid) == 2 and pl.estimate
    assert pl.re_sim == pytest.approx(2500) and pl.re_real > 1e6     # the GUI's capped start for airfoils
    assert "Simulated Reynolds number" in pl.summary() and "cells" in repr(pl)
    s3 = proj.add_sphere(0.2)
    pl3 = proj.plan(s3, speed=10, device="cpu")
    assert len(pl3.grid) == 3 and pl3.cells > 1000 and "Symmetry plane" in pl3.summary()
    polar = al.plan(d.raw, "polar", re=1e6)
    assert polar.analysis == "polar" and polar.est_seconds > 0
    assert not proj.simulations


class _RecordingHost(api.Host):
    """A host that records every call and runs nothing: the contract AppHost implements."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def add_design(self, store, design):
        self.calls.append("add_design")
        super().add_design(store, design)

    def add_simulation(self, store, sim):
        self.calls.append("add_simulation")
        super().add_simulation(store, sim)

    def update_simulation(self, store, sim):
        self.calls.append(f"update_simulation:{sim.status}")
        super().update_simulation(store, sim)

    def run_simulation(self, store, sim_id, *, progress, cancel):
        self.calls.append("run_simulation")
        progress(0.5, "half way")
        res = PolarResult(sim_id=sim_id, re=1e6, points=[
            PolarPoint(alpha=a, cl=0.1 * a, cd=0.01, cm=-0.05, converged=True) for a in (0.0, 2.0, 4.0)])
        store.save_result(sim_id, res)
        sim = store.simulations[sim_id]
        sim.status = "done"
        self.update_simulation(store, sim)
        return res, "done"


def test_a_host_carries_out_every_change_and_run(tmp_path):
    host = _RecordingHost()
    p = al.Project(Store.create(tmp_path / "Hosted", "Hosted"), host=host)
    d = p.add_naca("2412")
    r = p.run_polar(d, alpha=(0, 4, 2))
    assert host.calls[:3] == ["add_design", "add_simulation", "run_simulation"]
    assert r.cl[-1] == pytest.approx(0.4) and r.status == "done"
    assert p.run_polar(d, alpha=(0, 4, 2)).reused and host.calls.count("run_simulation") == 1
    # two designs at the same conditions: the Compare page's labels (design first), Re as the GUI writes it
    r2 = p.run_polar(p.add_naca("0012"), alpha=(0, 4, 2))
    cmp = p.compare([r, r2])
    assert cmp.names == ["NACA 2412 · Polar · Re 1.0×10⁶", "NACA 0012 · Polar · Re 1.0×10⁶"]
    assert "NACA 0012 · Polar" in str(cmp) and "1.0×10⁶" in str(cmp) and "1000000" not in str(cmp)


def test_open_project_returns_the_bound_project(proj):
    with al.context(project=proj):
        assert al.open_project(proj.path) is proj
    assert al.open_project(proj.path) is not proj


def test_temporary_projects_and_saving(tmp_path):
    p = al.new_project()
    assert p.path.is_dir() and p.path.suffix == ".aerolume"
    with al.new_project(tmp_path / "Kept") as kept:
        kept.add_naca("0012")
    assert al.open_project(tmp_path / "Kept").designs[0].name == "NACA 0012"
    assert al.new_project(tmp_path / "Kept", exist_ok=True).designs[0].name == "NACA 0012"


@pytest.mark.slow
@pytest.mark.gpu
def test_tiny_3d_run_records_surface_volume_and_pictures(proj, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    s = proj.import_mesh(EXAMPLES3D / "sphere_d100mm_binary.stl", kind="generic")
    r = proj.run_cfd(s, speed=0.3, cells_per_ref=10, duration=6, device="cuda")
    assert r.kind == "cfd3d" and r.name.startswith("CFD 3D · 0.3 m/s") and np.isfinite(r.cd) and r.cd > 0
    assert {"drag", "side", "lift", "roll", "pitch", "yaw"} <= set(r.forces) and r.raw.surface
    assert r.raw.symmetry_plane and "0 (symmetry)" in str(r) and "L/D" not in str(r)   # half domain, bluff
    assert r.export(tmp_path / "surface.vtp").stat().st_size > 1000
    assert r.export(tmp_path / "volume.vti").stat().st_size > 1000
    assert r.plot("history", tmp_path / "history.png").is_file()
    # flow in the volume around the body, not only on its surface (DESIGN.md 7.13): the run keeps
    # an instantaneous smoke snapshot next to the mean flow, and both export and draw
    smoke = r.raw.find_volume("dye")
    assert smoke is not None and smoke.label == "instant" and float(np.max(smoke.fields["dye"])) > 0.05
    assert r.raw.find_volume("u", "mean") is not None
    assert r.export(tmp_path / "smoke.vti", "smoke").stat().st_size > 1000
    from aerolume.viz3d import backend_info
    if backend_info().get("ok"):
        for kind in ("surface", "streamlines", "smoke"):
            assert r.plot(kind, tmp_path / f"{kind}.png").stat().st_size > 10_000
    stored = Store.open(proj.path).load_result(r.sim_id)
    assert stored.cd.mean == pytest.approx(r.cd) and stored.find_volume("dye") is not None


def test_result_headline_rounds_to_the_standard_error():
    """The first line of a printed result uses its table's rule: "CL 0.06323 ± 0.019" printed digits the
    run did not measure; it reads "0.063 ± 0.019", a lift inside its error bar "≈0", a settled Nu "steady"."""
    from aerolume.api import _mean_error
    assert _mean_error(0.7510, 0.0094) == "0.751 ± 0.009"
    assert _mean_error(0.06323, 0.019) == "0.063 ± 0.019"
    assert _mean_error(-0.0015, 0.0029).startswith("≈0 ± 0.0029")
    assert _mean_error(4.5624, 2e-17) == "4.562 (steady)"
    assert _mean_error(1.5, float("nan")) == "1.500"
