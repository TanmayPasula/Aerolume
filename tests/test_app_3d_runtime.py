"""3D CFD and finite-wing results in the runtime pages (Simulate, Results, Compare, Report) and the job manager.

Everything runs offscreen. Results are realistic synthetic ones (analytic sphere
flow, an elliptic-ish wing sweep) saved and re-loaded through the real project
store, so the save -> load -> display path is the one users exercise. The 3D
views render through VTK off-screen, as in the application. Two tests run the
real solvers: an inviscid vortex-lattice sweep (fast) and, marked slow, a short
Draft 3D run of a sphere and a viscous wing sweep through the job manager.
"""
from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from aerolume.model import (
    CFD3DResult,
    Design,
    FlowConditions,
    GroundConfig,
    Mesh3D,
    Simulation,
    Stat,
    SurfaceData3D,
    WingPoint,
    WingResult,
    WingSection,
    WingSettings,
    WingSpec,
)

sys.path.insert(0, os.path.dirname(__file__))
from test_viz3d_fixtures import sphere_cp, sphere_flow_volume, uv_sphere

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# fixtures and synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from aerolume.app import theme
    theme.apply(app, "light")
    return app


def pump(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()


def pump_until(app, cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        pump(app, 0.02)
    return bool(cond())


def sphere_design(name: str = "Sphere") -> Design:
    v, f = uv_sphere(1.0, 24, 48)
    return Design(name=name, kind="generic", bodies=[], length_ref=1.0, force_ref=1.0, dimension=3,
                  meshes=[Mesh3D("sphere", v, f)], ref_area=np.pi / 4, span_ref=1.0, symmetric=True)


def wing_design(name: str = "Wing") -> Design:
    w = WingSpec(sections=[WingSection(y=0.0, chord=0.3, airfoil="2412"),
                           WingSection(y=1.2, chord=0.15, x_le=0.04, airfoil="2412")])
    return Design(name=name, kind="wing", bodies=[], length_ref=0.233, force_ref=0.233, dimension=3, wing=w,
                  ref_area=0.54, span_ref=2.4, moment_ref3=(0.08, 0.0, 0.0), symmetric=True)


def st(mean: float, se: float = 0.003, rms: float = 0.01) -> Stat:
    return Stat(mean=mean, rms=rms, min=mean - 3 * rms, max=mean + 3 * rms, stderr=se)


def synthetic_cfd3d(sim_id: str, *, cd: float = 0.45, cl: float = 0.02, symmetric: bool = True,
                    yawed: bool = False) -> CFD3DResult:
    """Potential flow past a sphere (d = 1 m, 10 m/s) dressed up as a 3D run."""
    from aerolume.viz3d.data import slice_from_volume
    v, f = uv_sphere(1.0, 24, 48)
    cp = sphere_cp(v)
    surf = SurfaceData3D(body="sphere", vertices=v, faces=f, cp=cp, cf=0.004 + 0.002 * np.abs(v[:, 2]),
                         sensitivity=np.sin(3 * v[:, 0]) * np.cos(2 * v[:, 2]))
    vol = sphere_flow_volume(n=(48, 32, 32), lo=(-1.5, -1.0, -1.0), hi=(2.5, 1.0, 1.0), mirrored=symmetric)
    sp = vol.fields["speed"]
    vol.fields["q"] = (np.exp(-((sp - 10.0) / 2.0) ** 2) - 0.3).astype(np.float32)
    vol.fields["vorticity"] = np.abs(np.gradient(vol.fields["u"], axis=0)).astype(np.float32)
    slices = [slice_from_volume(vol, a, p, "speed") for a, p in (("y", 0.05), ("z", 0.0), ("x", 1.0))]
    for s, (a, p) in zip(slices, (("y", 0.05), ("z", 0.0), ("x", 1.0))):
        s.fields["cp"] = slice_from_volume(vol, a, p, "cp").fields["cp"]
    t = np.linspace(0, 30, 1500)
    side = 0.08 if yawed else 0.0
    hist = {"t": t, "cd": cd + 0.01 * np.sin(t), "cs": side + 0.004 * np.sin(1.3 * t) * (1 if yawed else 0),
            "cl": cl + 0.02 * np.sin(1.7 * t), "c_roll": 0.0 * t + (0.01 if yawed else 0.0), "cm": -0.01 + 0 * t,
            "cn": 0.0 * t + (0.02 if yawed else 0.0)}
    q_area = 0.5 * 1.225 * 100.0 * np.pi / 4
    return CFD3DResult(sim_id=sim_id, cd=st(cd), cs=st(side, 0.002 if yawed else 0.0, 0.004 if yawed else 0.0),
                       cl=st(cl), c_roll=st(0.01 if yawed else 0.0, 0.0, 0.0), cm=st(-0.01),
                       cn=st(0.02 if yawed else 0.0, 0.0, 0.0), drag=cd * q_area, side=side * q_area,
                       lift=cl * q_area, moments=(0.0, -0.01 * q_area, 0.0), drag_area=cd * np.pi / 4,
                       power=cd * q_area * 10.0, strouhal=0.19, re_real=6.8e5, re_sim=6.8e5, grid=(256, 64, 128),
                       cells=256 * 64 * 128, cells_per_ref=32, symmetry_plane=symmetric, t_end=30.0, t_avg_start=10.0,
                       converged=True, wall_seconds=125.0, device="GPU · NVIDIA GeForce RTX 4060 Ti (8.0 GB)",
                       precision="float32", turbulence="les-wale", warnings=["Draft quality: indicative only."],
                       history=hist, surface=[surf], slices=slices, volume=vol)


def synthetic_wing(sim_id: str, *, scale: float = 1.0, viscous: bool = True) -> WingResult:
    pts = []
    y = np.linspace(-1.2, 1.2, 40)
    for a in np.arange(-4.0, 13.0, 1.0):
        cl = scale * 0.085 * (a + 1.3)
        cdi = cl * cl / (np.pi * 10.0 * 0.98)
        cdv = (0.0075 + 0.004 * cl * cl) if viscous else 0.0
        shape = np.sqrt(np.clip(1 - (y / 1.2) ** 2, 0, 1))
        pts.append(WingPoint(alpha=float(a), cl=cl, cd=cdi + cdv, cdi=cdi, cdv=cdv, cm=-0.05 - 0.002 * a,
                             e=0.98, converged=True, y=y, cl_local=cl * 1.1 * shape, c_cl=4 * cl / np.pi * shape,
                             alpha_i=-np.degrees(cl / (np.pi * 10)) * np.ones_like(y),
                             cd_local=np.full_like(y, cdv) if viscous else None))
    return WingResult(sim_id=sim_id, method="vlm", viscous=viscous, area=0.54, span=2.4, aspect_ratio=10.67, mac=0.233,
                      re_mac=3.99e5, points=pts, wall_seconds=21.0,
                      notes=[("Stall onset (critical-section estimate): the first strip, at y = +0.9 m, reaches its "
                              "section cl_max at alpha ≈ 11.2 deg, CL ≈ 1.050. Treat this as the wing's CL_max "
                              "estimate; the curves above it are not a stall prediction.")])


@pytest.fixture()
def project3d(tmp_path):
    """A real project: a sphere with two 3D runs (one yawed) and a wing with two sweeps, plus one 2D design."""
    from aerolume.model import Body
    from aerolume.project import Project
    pr = Project.create(tmp_path / "Study3D.aerolume", "3D study")
    sph, wing = sphere_design(), wing_design()
    t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    foil = Design(name="Ellipse", kind="airfoil", bodies=[Body("b", np.stack([0.5 + 0.5 * np.cos(t),
                                                                             0.06 * np.sin(t)], 1))],
                  length_ref=1.0, force_ref=1.0)
    for d in (sph, wing, foil):
        pr.add_design(d)
    sims = [Simulation(name="Sphere 3D", design_id=sph.id, conditions=FlowConditions(speed=10.0), status="done",
                       id="s3d"),
            Simulation(name="Sphere 3D yawed", design_id=sph.id,
                       conditions=FlowConditions(speed=10.0, yaw_deg=5.0), status="done", id="s3y"),
            Simulation(name="Sphere new", design_id=sph.id, conditions=FlowConditions(speed=10.0), id="s3n"),
            Simulation(name="Wing VLM", design_id=wing.id, analysis="wing", status="done", id="w1",
                       wing=WingSettings(method="vlm")),
            Simulation(name="Wing B", design_id=wing.id, analysis="wing", status="done", id="w2"),
            Simulation(name="Wing new", design_id=wing.id, analysis="wing", id="wn",
                       wing=WingSettings(viscous=False)),
            Simulation(name="Foil CFD", design_id=foil.id, status="done", id="f2d")]
    for s in sims:
        pr.add_simulation(s)
    pr.save()
    pr.save_result("s3d", synthetic_cfd3d("s3d"))
    pr.save_result("s3y", synthetic_cfd3d("s3y", cd=0.47, cl=0.03, symmetric=False, yawed=True))
    pr.save_result("w1", synthetic_wing("w1"))
    pr.save_result("w2", synthetic_wing("w2", scale=1.05))
    sys.path.insert(0, os.path.dirname(__file__))
    from test_app_runtime import synthetic_cfd
    pr.save_result("f2d", synthetic_cfd("f2d"))
    return pr


@pytest.fixture()
def state3d(qapp, project3d):
    from aerolume.app.jobs import JobManager
    from aerolume.app.state import AppState
    state = AppState()
    state.set_project(project3d)
    jobs = JobManager(state)
    yield state, jobs
    jobs.shutdown(timeout=5)


def slice_frame(axis: str = "y", *, phase: str = "averaging", progress: float = 0.5, yawed: bool = False):
    """A live 3D frame: a speed slice of the sphere flow with the sphere's section as body."""
    from aerolume.app.widgets.field_view import colorize, colormap_lut
    from aerolume.viz3d.data import slice_from_volume
    vol = sphere_flow_volume(n=(48, 32, 32))
    pos = {"y": 0.0, "z": 0.0, "x": 1.0}[axis]
    sd = slice_from_volume(vol, axis, pos, "speed")
    rgba = colorize(sd.fields["speed"], colormap_lut("viridis"), 0.0, 15.0, mask=sd.mask)
    th = np.linspace(0, 2 * np.pi, 60)
    loop = np.stack([0.5 * np.cos(th), 0.5 * np.sin(th)], 1)            # first point repeated: closed
    sheet = np.array([[1.5, 0.4], [2.0, 0.5], [2.4, 0.45]])               # an open chain
    t = np.linspace(0, 8, 200)
    hist = {"t": t, "cd": 0.5 + 0 * t, "cs": 0.05 + 0 * t, "cl": 0.01 + 0 * t, "c_roll": 0 * t, "cm": 0 * t,
            "cn": 0 * t}
    return SimpleNamespace(rgba=rgba, extent=sd.extent, field="speed", cmap=None, vmin=0.0, vmax=15.0, units="m/s",
                           t=8.0, step=900, steps_per_s=45.0,
                           coeffs={"cd": 0.51, "cs": 0.052 if yawed else 0.0, "cl": 0.012, "c_roll": 0.0,
                                   "cm": 0.0, "cn": 0.0},
                           history=hist, progress=progress, eta_s=30.0, phase=phase,
                           bodies=[loop, sheet] if axis == "y" else [loop], particles=None, streamlines=None,
                           focus=None, slice_axis=axis, slice_pos=pos)


def _vtk_ok() -> bool:
    try:
        from aerolume.viz3d.scene import backend_info
        return bool(backend_info()["ok"])
    except Exception:  # noqa: BLE001 - no VTK, no 3D views
        return False


needs_vtk = pytest.mark.skipif(not _vtk_ok(), reason="no off-screen OpenGL context for VTK")


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_parse_wing_message():
    from aerolume.app.pages.simulate import parse_wing_message
    sec = parse_wing_message("Wing: Section polar 2412, Re 3.6e+05: alpha = 4 deg: Cl = 0.718, Cd = 0.00697")
    assert sec == ("section", "2412", 3.6e5, 4.0, 0.718, 0.00697, True)
    bad = parse_wing_message("Wing: Section polar design:ab12, Re 120000: alpha = 15 deg: not converged")
    assert bad[0] == "section" and bad[1] == "design:ab12" and bad[5] != bad[5] and bad[6] is False
    assert parse_wing_message("Wing: alpha = -2 deg") == ("angle", -2.0)
    assert parse_wing_message("Wing: planning section polars") is None
    assert parse_wing_message("Wing polar finished: 17 of 17 angles converged") is None


def test_plan_key_sees_mesh_edits():
    from aerolume.app.pages.simulate import plan_key
    d = sphere_design()
    s = Simulation(name="A", design_id=d.id)
    k = plan_key(s, d)
    d.meshes[0].vertices = d.meshes[0].vertices * 1.5           # what a scale in Geometry does
    assert plan_key(s, d) != k


def test_clone_design_keeps_mesh_arrays_and_snapshots_airfoils():
    """The job copies are real copies: a to_dict round trip of a 3D design has no triangles."""
    from aerolume.app.jobs import airfoil_designs, clone_design
    d = sphere_design()
    c = clone_design(d)
    assert len(c.meshes[0].faces) == len(d.meshes[0].faces) > 0
    assert c.meshes[0].vertices is not d.meshes[0].vertices
    assert len(Design.from_dict(d.to_dict()).meshes[0].faces) == 0      # the trap clone_design avoids
    foil = Design(name="Foil", kind="airfoil", bodies=[], length_ref=1.0, force_ref=1.0)
    w = wing_design()
    w.wing.sections[1].airfoil = f"design:{foil.id}"
    snap = airfoil_designs(SimpleNamespace(designs={foil.id: foil, w.id: w}), w)
    assert list(snap) == [foil.id] and snap[foil.id] is not foil
    assert airfoil_designs(SimpleNamespace(designs={}), d) == {}


def test_mesh_sections_closed_loops_and_open_chains():
    from aerolume.app.widgets.field_view import section_paths
    from aerolume.app.widgets.results3d import mesh_sections
    v, f = uv_sphere(1.0, 24, 48)
    secs = mesh_sections([(v, f)], "y", 0.0)
    assert len(secs) == 1 and np.allclose(secs[0][0], secs[0][-1])
    assert np.allclose(np.linalg.norm(secs[0], axis=1), 0.5, atol=0.01)
    top = v[f].mean(axis=1)[:, 2] > 0                      # half a sphere: an open sheet
    secs = mesh_sections([(v, f[top])], "y", 0.0)
    closed, open_ = section_paths(secs)
    assert not closed and len(open_) == 1 and open_[0][:, 1].min() > -1e-6
    assert mesh_sections([(v, f)], "x", 2.0) == []          # the plane misses the body


def test_subscripts_for_3d_symbols():
    from aerolume.app.widgets.kpi import has_symbols, subscripts
    assert subscripts("Drag coefficient CD") == "Drag coefficient C<sub>D</sub>"
    assert subscripts("CL,max") == "C<sub>L,max</sub>" and subscripts("CDi") == "C<sub>Di</sub>"
    assert subscripts("C_roll") == "C<sub>roll</sub>" and subscripts("CDA") == "C<sub>D</sub>A"
    assert not has_symbols("CSV file") and not has_symbols("CPU")


def test_elliptic_loading_averages_to_cl():
    from aerolume.app.widgets.wing_plots import elliptic_loading
    y = np.linspace(-1.0, 1.0, 20001)
    e = elliptic_loading(y, 2.0, 0.6)
    assert np.trapezoid(e, y) / 2.0 == pytest.approx(0.6, rel=1e-3)


def test_frame_to_slice_inverts_the_colour_map():
    from aerolume.app.widgets.results3d import frame_to_slice
    fr = slice_frame("y")
    sd = frame_to_slice(fr, max_cols=1000)
    vals = sd.fields["speed"]
    assert sd.axis == "y" and vals.shape == fr.rgba.shape[:2]
    ok = sd.mask < 0.5
    assert 0.0 <= vals[ok].min() and vals[ok].max() <= 15.0
    assert sd.mask.sum() > 0                               # the solid sphere is not a map colour
    assert frame_to_slice(SimpleNamespace(slice_axis=None)) is None


def test_summary_rows3d_scale_coefficients_into_forces():
    from aerolume.app.widgets.results3d import history_keys3d, summary_rows3d
    r = synthetic_cfd3d("x")
    d, sim = sphere_design(), Simulation(name="s", design_id="d", conditions=FlowConditions(speed=10.0))
    rows = summary_rows3d(r, d, sim)
    names = [row[0].text if hasattr(row[0], "text") else row[0] for row in rows]
    assert names[:6] == ["Drag coefficient CD", "Side-force coefficient CS", "Lift coefficient CL",
                         "Rolling-moment coefficient C_roll", "Pitching-moment coefficient Cm",
                         "Yawing-moment coefficient Cn"]
    drag = rows[names.index("Drag")]
    assert float(drag[1].text) == pytest.approx(r.drag, rel=1e-3)
    side = rows[names.index("Side force")]                 # zero by symmetry: greyed, with the reason
    assert side[1].text == "0" and side[1].tone == "muted" and "symmetry" in side[0].tooltip
    assert history_keys3d(r) == ("cd", "cl", "cm")
    ry = synthetic_cfd3d("y", symmetric=False, yawed=True)
    assert history_keys3d(ry) == ("cd", "cs", "cl", "c_roll", "cm", "cn")
    rows = summary_rows3d(ry, d, sim)
    names = [row[0].text if hasattr(row[0], "text") else row[0] for row in rows]
    assert rows[names.index("Side force")][1].tone is None


def test_wing_summary_rows_read_the_metrics():
    from aerolume.app.widgets.results3d import stall_onset, wing_header_facts, wing_summary_rows
    w = synthetic_wing("w")
    rows = wing_summary_rows(w)
    names = [r[0].text if hasattr(r[0], "text") else r[0] for r in rows]
    assert "Stall onset (estimate)" in names and stall_onset(w) == (11.2, 1.05)
    clmax = rows[names.index("Maximum lift CL,max")]
    assert str(clmax[1].text).startswith("≥")                # still rising at 12°: a lower bound
    slope = rows[[n.startswith("Lift-curve") for n in names].index(True)]
    assert float(slope[1]) == pytest.approx(0.085 * 180 / np.pi, rel=0.01)
    status, kind, facts = wing_header_facts(w)
    assert status == "17 of 17 angles converged" and kind == "good" and "Vortex lattice" in facts
    inv = wing_summary_rows(synthetic_wing("w", viscous=False))
    best = next(r for r in inv if (r[0].text if hasattr(r[0], "text") else r[0]).startswith("Best"))
    assert best[1].text == "—"                                 # no profile drag: not applicable


# ---------------------------------------------------------------------------
# job manager
# ---------------------------------------------------------------------------

def test_execute_simulation_goes_through_the_shared_dispatcher(monkeypatch):
    """The job manager runs what the CLI and the scripting API run: runner.run_simulation, with the
    airfoil snapshot (never the live project) as its project."""
    import aerolume.app.jobs as jobs_mod
    calls = []

    def run_simulation(design, sim, **kw):
        calls.append((sim.analysis, kw))
        return f"{sim.analysis}-result"

    monkeypatch.setitem(sys.modules, "aerolume.simulation.runner", SimpleNamespace(run_simulation=run_simulation))
    w = wing_design()
    foil = Design(name="Foil", kind="airfoil", bodies=[], length_ref=1.0, force_ref=1.0)
    snap = {foil.id: foil}
    seen, cancel, view = [], object(), object()
    out = jobs_mod.execute_simulation(w, Simulation(name="c", design_id=w.id), planned=seen.append, airfoils=snap,
                                      cancel=cancel, view=view)
    assert out == "cfd-result"
    kw = calls[-1][1]
    assert kw["project"] is snap and kw["cancel"] is cancel and kw["view"] is view and kw["planned"] == seen.append
    jobs_mod.execute_simulation(w, Simulation(name="w", design_id=w.id, analysis="wing"))
    assert calls[-1][0] == "wing" and calls[-1][1]["project"] is None      # no project airfoils: none passed


def test_execute_simulation_dispatches_wings_and_3d(monkeypatch):
    """A runner module without run_simulation (an older build) is dispatched by the job manager itself."""
    import aerolume.app.jobs as jobs_mod
    calls = []
    runner = SimpleNamespace(
        run_polar=lambda *a, **k: calls.append("polar"),
        run_wing=lambda design, sim, **k: calls.append(("wing", k.get("project"))) or "wing-result",
        run_cfd=lambda plan, **k: calls.append(("cfd", plan)) or "cfd-result")
    plan_mod = SimpleNamespace(plan_simulation=lambda design, sim, **kw: calls.append(("plan", sorted(kw))) or "PLAN")
    monkeypatch.setitem(sys.modules, "aerolume.simulation.runner", runner)
    monkeypatch.setitem(sys.modules, "aerolume.simulation.plan", plan_mod)
    w = wing_design()
    foil = Design(name="Foil", kind="airfoil", bodies=[], length_ref=1.0, force_ref=1.0)
    snap = {foil.id: foil}
    sim_w = Simulation(name="w", design_id=w.id, analysis="wing")
    assert jobs_mod.execute_simulation(w, sim_w, airfoils=snap) == "wing-result"
    assert calls[-1] == ("wing", snap)
    sim_c = Simulation(name="c", design_id=w.id)
    seen = []
    assert jobs_mod.execute_simulation(w, sim_c, planned=seen.append, airfoils=snap) == "cfd-result"
    assert ("plan", ["airfoil_coords"]) in calls and seen == ["PLAN"]


def test_jobmanager_runs_a_3d_job_on_real_copies(qapp, state3d):
    """The worker gets the mesh arrays (the old to_dict copy had none) and the 3D result is saved."""
    state, jobs = state3d
    seen = {}

    def run(design, sim, *, progress, frame, cancel, view, planned):
        seen["faces"] = len(design.meshes[0].faces)
        planned(SimpleNamespace(t_end=30.0, dimension=3))
        frame(slice_frame("y", phase="done", progress=1.0))
        return synthetic_cfd3d(sim.id)

    jobs.runner = run
    jobs.submit("s3n")
    assert jobs.wait(20)
    assert seen["faces"] > 0
    assert state.project.simulations["s3n"].status == "done"
    r = state.project.load_result("s3n")
    assert isinstance(r, CFD3DResult) and r.volume is not None and r.symmetry_plane


def test_live_worker_hands_airfoils_to_factories_that_take_them(qapp):
    from aerolume.app.jobs import LiveWorker
    got = {}

    class Session:
        plan = None

        def __init__(self, design, sim, *, device=None, view=None, airfoils=None):
            got["airfoils"] = airfoils
            got["faces"] = len(design.meshes[0].faces) if design.meshes else 0

        def post(self, **_k):
            pass

        def tick(self, max_ms=33.0):
            time.sleep(0.005)

        def close(self):
            pass

    lw = LiveWorker()
    lw.session_factory = Session
    d = sphere_design()
    lw.start(d, Simulation(name="s", design_id=d.id), airfoils={"x": d})
    assert pump_until(qapp, lambda: "airfoils" in got, 5)
    lw.stop()
    assert got["airfoils"] == {"x": d} or set(got["airfoils"]) == {"x"}
    assert got["faces"] > 0


# ---------------------------------------------------------------------------
# Results page
# ---------------------------------------------------------------------------

@needs_vtk
def test_results_page_3d_summary_surface_fields_and_exports(qapp, state3d, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("s3d")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    assert page.body_stack.currentWidget() is page.view3d
    assert page.status_text.text() == "Converged"
    facts = page.facts.full_text()
    assert "256×64×128 grid" in facts and "half domain" in facts and "RTX 4060 Ti" in facts
    v = page.view3d
    assert v.summary_table.find_row("Drag coefficient CD") == 0
    assert v.summary_table.find_row("Drag power") > 0 and "q × 0.7854 m²" in v.summary_table.item(0, 0).toolTip()
    # Forces: CD, CL, Cm (CS, C_roll, Cn are zero by symmetry)
    page.handle_action("forces")
    assert v.history_plot.series_names() == ["Cd", "Cl", "Cm"]
    # Surface: the body coloured by Cp, centred range of at least +-1
    page.handle_action("surface")
    pump(qapp, 0.1)
    sp = v.surface_panel
    assert sp is not None and sp.field() == "cp"
    info = [sp.view.scene.scalar_info(h) for h in sp.view.scene.layers()]
    cp_info = next(i for i in info if i is not None)
    assert cp_info["field"] == "cp" and cp_info["vmin"] == pytest.approx(-cp_info["vmax"]) and cp_info["vmax"] >= 1.0
    # Fields: opens on the 3D flow -- mean-flow streamlines from the drawn rake; the slice plane and
    # the Q iso-surface on request; the same slice as a 2D field view
    page.handle_action("fields")
    pump(qapp, 0.3)
    fp = v.fields_panel
    assert {"stream", "rake", "body"} <= set(fp.handles()) and "slice" not in fp.handles()
    assert fp.mode.current() == "3d" and fp.view.isVisible() and not fp.slice_view.isVisible()
    assert fp.slice_view.has_image() and fp.slice_view.axis_names() == ("x", "z")
    assert not fp.smoke_on.isEnabled() and "no smoke" in fp.smoke_on.toolTip()     # no dye stored
    fp.slice_3d.setChecked(True)
    assert pump_until(qapp, lambda: "slice" in fp.handles(), 3)
    fp.axis_seg.set_current("x")
    fp.q_on.setChecked(True)
    assert pump_until(qapp, lambda: "iso" in fp.handles(), 3)
    assert fp.slice_view.axis_names() == ("y", "z")
    fp.mode.set_current("split")
    assert fp.view.isVisible() and fp.slice_view.isVisible()
    lines = fp.view.scene.layer_lines(fp.handles()["stream"])
    assert len(lines) >= 10
    # Sensitivity: relative map on +-1
    page.handle_action("sensitivity")
    pump(qapp, 0.1)
    assert v.sens_stack.currentIndex() == 0 and "pattern, not the size" in v.sens_panel.caption.text()
    # Data: exports go through aerolume.export on a worker thread
    out = {}
    for key in ("surface_vtp", "volume_vti", "history_csv", "slice_csv"):
        path = tmp_path / f"out_{key}"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, p=str(path): (p, "")))
        v.data_tab.export(key)
        assert pump_until(qapp, lambda p=path: p.exists() and v.data_tab.status.text().startswith("Saved"), 10), \
            v.data_tab.status.text()
        out[key] = path.stat().st_size
    assert all(n > 100 for n in out.values())
    page.close()


@needs_vtk
def test_results_page_wing(qapp, state3d, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    state.select_simulation("w1")
    page = ResultsPage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    v = page.wing_view
    assert page.body_stack.currentWidget() is v
    assert page.status_text.text() == "17 of 17 angles converged"
    assert v.summary_table.find_row("Maximum lift CL,max") >= 0
    assert set(v.polars.drag_polar.series_names()) == {"CD (total)", "CDi (induced)", "CDv (profile)"}
    page.handle_action("span")
    assert pump_until(qapp, lambda: v._mesh is not None, 15)        # the lofted wing, built off the GUI thread
    pt = v.current_point()
    assert pt is not None and v.alpha_combo.count() == 17
    assert "Elliptic (same CL)" in v.span_plots.loading.series_names()
    layers = v.wing_view.scene.layers()
    assert any(v.wing_view.scene.scalar_info(h) for h in layers)          # coloured by local cl
    assert v.points_table.rowCount() == 17
    path = tmp_path / "span.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a: (str(path), "")))
    v.data_tab.export("wing_span_csv")
    assert pump_until(qapp, lambda: path.exists(), 10)
    assert path.read_text().splitlines()[0].startswith("alpha,y,")
    page.close()


# ---------------------------------------------------------------------------
# Simulate page
# ---------------------------------------------------------------------------

@needs_vtk
def test_simulate_page_3d_slice_controls_and_frames(qapp, state3d, monkeypatch):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d
    monkeypatch.setattr(SimulatePage, "_request_plan", lambda self, sim, design: None)
    state.select_simulation("s3y")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    assert page._mode == "field3d" and page.slice_strip.isVisible() and page.view3d is not None
    assert page.view3d.isVisible() and not page.view.isVisible()          # the 3D view is the viewport
    assert page._row_cs.label.isVisible()                      # a yawed run: side force monitored
    assert page.history.keys == ("cd", "cs", "cl")
    assert page.field_combo.findData("q") >= 0 and page.field_combo.findData("dye") >= 0   # dye: smoke in the slice
    assert not page.particles_btn.isVisibleTo(page)
    # the stored result is shown: a slice through the mean volume, in 2D (and as a plane in 3D on request:
    # by default the 3D view shows the body and the flow around it, not one more coloured surface)
    assert page.view.has_image() and page.view.axis_names() == ("x", "z")
    assert "slice" not in page.flow.layer_handles() and "body" in page.flow.layer_handles()
    page.flow_toggles.buttons["slice"].setChecked(True)
    assert "slice" in page.flow.layer_handles()
    # the slice strip drives the shared view settings the runner reads
    page.slice_axis.set_current("z")
    assert jobs.view.slice_axis == "z" and jobs.view.slice_pos is None
    assert page.view.axis_names() == ("x", "y")
    page.slice_slider.setValue(700)
    assert jobs.view.slice_pos == pytest.approx(page.slice_position())
    # live frames of a run: slice image, sections (filled loop + stroked sheet), CS monitor
    monkeypatch.setattr(jobs, "running_id", lambda: "s3y")
    page._shown_sim_id = "s3y"
    jobs.jobFrame.emit("s3y", slice_frame("y", yawed=True))
    pump(qapp)
    assert page.view.has_image() and page.view.axis_names() == ("x", "z")
    assert len(page.view._open_lines) == 1 and not page.view._bodies_path.isEmpty()
    assert page.coef_values["cs"].text() == "0.05200"
    assert page.slice_axis.current() == "y"                    # the strip follows the frame
    assert "y = 0 m" in page.view.caption()
    page.close()


def test_simulate_page_wing_run_progress_and_result(qapp, state3d):
    from aerolume.app.pages.simulate import SimulatePage
    state, jobs = state3d

    def run(design, sim, *, progress, frame, cancel, view):
        progress(0.0, "Wing polar (vortex lattice): 17 angles at 30 m/s")
        for a, cl in ((0, 0.25), (1, 0.36), (2, 0.47)):
            progress(0.1 * (a + 1), f"Wing: Section polar 2412, Re 3.6e+05: alpha = {a} deg: Cl = {cl}, Cd = 0.0071")
            time.sleep(0.08)                                   # the job manager throttles progress at 15 Hz
        progress(0.95, "Wing: alpha = 4 deg")
        time.sleep(0.08)
        return synthetic_wing(sim.id)

    jobs.runner = run
    state.select_simulation("wn")
    page = SimulatePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    assert page._mode == "wing" and page.vp_stack.currentWidget().findChild(type(page.wing_view)) is page.wing_view
    assert page._row_grid.label.text() == "Angles" and page.estimate_value.text().startswith("Vortex lattice")
    assert not page.grid.group("Explore").isVisible()
    seen_nodes = []
    jobs.jobProgress.connect(lambda *_: seen_nodes.append(len(page.wing_view.nodes())))
    page.run_current()
    assert jobs.wait(20)
    pump(qapp)
    assert max(seen_nodes) == 1                                 # the section polar listed while it ran
    assert page.wing_view.table_title.text() == "Wing polar" and page.wing_view.table.rowCount() == 17
    assert page.monitor_note.text().startswith("At best L/D")
    assert page._row_cd.label.text() == "C<sub>D</sub>"
    page.close()


# ---------------------------------------------------------------------------
# Compare page
# ---------------------------------------------------------------------------

@needs_vtk
def test_compare_page_3d_table_forces_and_linked_surfaces(qapp, state3d):
    from aerolume.app.pages.compare import TAB_KEYS, ComparePage
    state, jobs = state3d
    state.compare_ids[:] = ["s3d", "s3y"]
    page = ComparePage(state, jobs)
    page.resize(1600, 1000)
    page.show()
    pump(qapp)
    assert page._group == "cfd3d" and page.inner.currentIndex() == 1
    assert page.table.find_row("Side-force coefficient CS") >= 0 and page.table.find_row("Drag") >= 0
    visible = [k for i, k in enumerate(TAB_KEYS) if page.tabs.isTabVisible(i)]
    assert visible == ["chart", "forces", "surface"]
    assert page.forces3d.run_names() == ["Sphere 3D", "Sphere 3D yawed"]
    assert "cs" in page.forces3d.keys                           # the yawed run has a side force
    page.handle_action("fields")                                # 3D: the surfaces
    pump(qapp, 0.2)
    grid = page.surface_grid
    assert grid is not None and len(grid.tiles) == 2
    a, b = grid.tiles
    a.scene.orbit(30.0, 10.0)
    a._camera_moved()
    assert b.scene.camera.position == pytest.approx(a.scene.camera.position)
    page.close()


def test_compare_page_refuses_to_mix_kinds(qapp, state3d):
    from aerolume.app.pages.compare import ComparePage
    state, jobs = state3d
    state.compare_ids[:] = ["s3d", "s3y", "w1"]
    page = ComparePage(state, jobs)
    pump(qapp)
    assert page.inner.currentIndex() == 0 and page._group == "mixed"
    assert page.inner_empty.title.text() == "These results cannot be compared"
    assert "reference area" in page.inner_empty.text.text() and "operating point" in page.inner_empty.text.text()
    assert page.inner_empty.button.text() == "Compare the 3D CFD results (2)"
    page.inner_empty.button.click()
    pump(qapp)
    assert state.compare_ids == ["s3d", "s3y"] and page.inner.currentIndex() == 1
    page.close()


def test_compare_page_wings(qapp, state3d):
    from aerolume.app.pages.compare import TAB_KEYS, ComparePage
    state, jobs = state3d
    state.select_simulation("w1")
    state.compare_ids[:] = []
    page = ComparePage(state, jobs)
    pump(qapp)
    assert sorted(state.compare_ids) == ["w1", "w2"] and page._group == "wing"
    visible = [k for i, k in enumerate(TAB_KEYS) if page.tabs.isTabVisible(i)]
    assert visible == ["chart", "wing_polars", "span"]
    assert page.table.find_row("Maximum lift CL,max") >= 0
    assert len(page.wing_polars.cl_alpha.series_names()) == 2
    assert page.span_alpha.currentData() == 4.0 and len(page.span_plots.local_cl.series_names()) == 2
    page.handle_action("polars")
    assert TAB_KEYS[page.tabs.currentIndex()] == "wing_polars"
    page.close()


def test_report_page_names_kinds_and_compares_like_with_like(qapp, state3d):
    from aerolume.app.pages.report import ReportPage
    state, jobs = state3d
    page = ReportPage(state, jobs)
    pump(qapp)
    from PySide6.QtCore import Qt
    role = Qt.ItemDataRole.UserRole                     # the simulation id of a list row
    tips = {page.sim_list.item(i).data(role): page.sim_list.item(i).toolTip() for i in range(page.sim_list.count())}
    assert "Navier–Stokes 3D" in tips["s3d"] and "wing sweep" in tips["w1"] and "Navier–Stokes 2D" in tips["f2d"]

    def tick(ids):
        for i in range(page.sim_list.count()):
            it = page.sim_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(role) in ids else Qt.CheckState.Unchecked)
        page._update_controls()

    tick({"s3d", "w1"})
    assert not page.section_boxes["comparison"].isEnabled()      # one 3D run and one wing: not comparable
    assert page.section_boxes["polar"].isEnabled()               # the wing sweep has polars
    tick({"s3d", "s3y"})
    assert page.section_boxes["comparison"].isEnabled() and page.section_boxes["fields"].isEnabled()
    # without an OpenGL context the form says beforehand that 3D figures become a sentence
    page.show()
    page._gl = {"ok": False, "error": "no OpenGL 3.2 context could be created for off-screen rendering"}
    tick({"s3d"})
    assert page.gl_note.isVisible() and "OpenGL" in page.gl_note.text() and "3.2 context" in page.gl_note.text()
    tick({"f2d"})
    assert not page.gl_note.isVisible()                          # a 2D report has no 3D figures
    page._gl = {"ok": True}
    tick({"w1"})
    assert not page.gl_note.isVisible()
    page.close()


# ---------------------------------------------------------------------------
# real solvers
# ---------------------------------------------------------------------------

def test_real_inviscid_wing_sweep_through_the_job_manager(qapp, state3d):
    """The real vortex lattice (inviscid: no section polars, a second or two) saved and shown."""
    from aerolume.app.pages.results import ResultsPage
    state, jobs = state3d
    sim = state.project.simulations["wn"]
    sim.wing.alpha_start, sim.wing.alpha_end, sim.wing.alpha_step = -2.0, 8.0, 2.0
    state.update_simulation(sim)
    jobs.submit("wn")
    assert jobs.wait(60)
    r = state.project.load_result("wn")
    assert isinstance(r, WingResult) and len(r.points) == 6 and all(p.converged for p in r.points)
    assert 0.05 < r.points[1].cl < 0.3 and all(p.cdv == 0 for p in r.points)
    state.select_simulation("wn")
    page = ResultsPage(state, jobs)
    pump(qapp)
    assert page.status_text.text() == "6 of 6 angles converged"
    assert page.wing_view.polars.ld_alpha.series_names() == []     # inviscid: L/D is not plotted
    page.close()


@pytest.mark.slow
@pytest.mark.gpu
def test_real_draft_3d_run_and_viscous_wing(qapp, tmp_path):
    """A real Draft 3D run of a sphere and a real viscous wing polar, start to Results page."""
    from aerolume.app.jobs import JobManager
    from aerolume.app.pages.results import ResultsPage
    from aerolume.app.pages.simulate import SimulatePage
    from aerolume.app.state import AppState
    from aerolume.geometry3d import design_from_meshes
    from aerolume.geometry3d.primitives import sphere
    from aerolume.project import Project
    from aerolume.simulation.defaults import default_simulation
    pr = Project.create(tmp_path / "Real.aerolume", "Real")
    sph = design_from_meshes([sphere(0.1, 24, name="Sphere")], "Sphere", kind="generic", on_road=False)
    pr.add_design(sph)
    s = default_simulation(sph, "Sphere", FlowConditions(speed=0.5))
    s.cfd.duration = 12.0
    pr.add_simulation(s)
    wd = wing_design()
    pr.add_design(wd)
    sw = default_simulation(wd, "Wing", FlowConditions(speed=25.0))
    sw.wing.alpha_start, sw.wing.alpha_end, sw.wing.alpha_step = 0.0, 6.0, 2.0
    pr.add_simulation(sw)
    pr.save()
    state = AppState()
    state.set_project(pr)
    jobs = JobManager(state)
    state.select_simulation(s.id)
    page = SimulatePage(state, jobs)
    page.resize(1280, 800)
    page.show()
    frames = []          # (slice axis, carries smoke): frames themselves would keep every volume alive
    jobs.jobFrame.connect(lambda sid, f: frames.append(
        (f.slice_axis, getattr(f, "volume", None) is not None and "dye" in f.volume.fields)))
    page.run_current()
    assert jobs.wait(600)
    jobs.submit(sw.id)
    assert jobs.wait(300)
    pump(qapp)
    r = pr.load_result(s.id)
    assert isinstance(r, CFD3DResult) and r.surface and r.volume is not None and np.isfinite(r.cd.mean)
    assert frames and frames[-1][0] in ("x", "y", "z")
    # the smoke in the volume: live during the run (the page's 3D view asked for it), stored with the result
    from aerolume.app.widgets.view3d_panel import smoke_volume_of
    assert any(smoky for _axis, smoky in frames)
    smoke = smoke_volume_of(r)
    assert smoke is not None and float(np.nanmax(np.asarray(smoke.fields["dye"], float))) > 0.1
    w = pr.load_result(sw.id)
    assert isinstance(w, WingResult) and w.viscous and sum(p.converged for p in w.points) >= 3
    res = ResultsPage(state, jobs)
    state.select_simulation(s.id)
    pump(qapp)
    assert res.view3d is not None and res.view3d.summary_table.find_row("Drag coefficient CD") == 0
    jobs.shutdown(timeout=10)
    page.close()
    res.close()


def test_ground_config_road_shows_in_3d_conditions():
    from aerolume.app.widgets.results3d import conditions_items3d
    sim = Simulation(name="s", design_id="d", ground=GroundConfig(enabled=True, moving=True))
    items = dict(conditions_items3d(synthetic_cfd3d("x"), sphere_design(), sim))
    assert items["Road"] == "moving belt" and items["Domain"].startswith("half")
