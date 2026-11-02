"""Take the screenshots listed in website/assets/screens/MANIFEST.md from the real application.

Usage (from the repository root; a CUDA GPU makes it take about five minutes)::

    python website/tools/make_screens.py --work DIR      # DIR: a scratch folder outside the repository

What it does, so that every caption in the manifest stays true:

1. builds a project in ``DIR`` with a NACA 2412 section (chord 1 m, a CFD simulation at
   8 degrees, not run yet) and three fastback variants (fastback slope 12, 17 and 22 degrees,
   30 m/s, moving road), and runs the three cars at Draft quality with the same functions the
   Run button uses (``examples/make_example_projects.run_and_store``);
2. opens the real ``MainWindow`` offscreen at 1600 x 1000 in the light theme (after
   ``theme.apply``, without which offscreen text renders as boxes), runs the NACA 2412
   simulation through the window's own job manager and grabs the Simulate page halfway through
   the run, then the other pages;
3. writes ``website/assets/screens/<page>.png`` and ``docs/img/ui_<page>.png`` (plus the start
   page for the user guide);
4. takes the placed-physics feature screens (:data:`EXTRA`) from copies of the bundled example projects,
   whose results are real Draft runs: the 3D car's smoke and streamlines, the heated cylinder's
   heat-transfer tab, the room's placed-physics editor, an explore session of the room with a
   wall drawn into the running flow beside the Diagnostics panel, and the Script console
   (``--only extra`` takes just these; ``--only pages`` just the workflow pages).

Screens are grabbed with ``QWidget.grab()``: nothing is edited or composed afterwards. Run
``python website/build.py`` afterwards.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCREENS = REPO / "website" / "assets" / "screens"
DOC_IMG = REPO / "docs" / "img"
PAGES = ("geometry", "setup", "simulate", "results", "compare", "report")


def _example_helpers():
    spec = importlib.util.spec_from_file_location("make_example_projects",
                                                  REPO / "examples" / "make_example_projects.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_project(work: Path) -> Path:
    """The demo project: NACA 2412 (alpha 8, not run) and three fastbacks (run here)."""
    from aerolume.geometry import design_from_airfoil, naca, templates
    from aerolume.model import FlowConditions, GroundConfig, Simulation
    from aerolume.project import Project
    from aerolume.simulation.defaults import default_cfd_settings

    path = work / "Aerolume demo.aerolume"
    if path.exists():
        shutil.rmtree(path)
    work.mkdir(parents=True, exist_ok=True)
    p = Project.create(path, "Aerolume demo")
    d = design_from_airfoil(naca("2412"), "NACA 2412", 1.0, source={"type": "naca", "code": "2412"})
    p.add_design(d)
    cond = FlowConditions()
    cond.speed = 1.0e6 * cond.fluid.viscosity / (cond.fluid.density * d.length_ref)      # Re = 1e6
    c8 = FlowConditions(fluid=cond.fluid, speed=cond.speed, alpha_deg=8.0)
    p.add_simulation(Simulation(name="CFD · α 8°", design_id=d.id, analysis="cfd", conditions=c8,
                                cfd=default_cfd_settings(d, c8)))
    cars = []
    for slope in (12.0, 17.0, 22.0):
        vd = templates.make("fastback", rear_angle=slope)
        vd.name = f"Fastback {slope:g}°"
        p.add_design(vd)
        vc = FlowConditions(speed=30.0, alpha_deg=0.0)
        sim = Simulation(name="CFD · 30 m/s", design_id=vd.id, analysis="cfd", conditions=vc,
                         ground=GroundConfig(enabled=True, moving=True, rotate_wheels=True),
                         cfd=default_cfd_settings(vd, vc))
        p.add_simulation(sim)
        cars.append(sim.id)
    p.save()
    helpers = _example_helpers()
    for sid in cars:
        helpers.run_and_store(p, sid)
    return path


def take(project_path: Path, work: Path, out: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["AEROLUME_CONFIG_DIR"] = str(work / "config")
    from PySide6.QtCore import QEventLoop, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    from aerolume.app import settings, theme
    settings.use_file(work / "settings.ini")
    theme.apply(app, "light")
    from aerolume.app.main_window import MainWindow
    from aerolume.project import Project

    def pump(sec: float = 0.4) -> None:
        t = time.time()
        while time.time() - t < sec:
            app.processEvents()
            time.sleep(0.01)

    def shot(name: str) -> None:
        pump(0.6)
        w.grab().save(str(out / name))
        print("saved", name, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    pump(0.5)
    st, jobs = w.state, w.jobs
    shot("start.png")
    w.set_project(Project.open(project_path))
    pump(0.5)
    # "Opened <project> (<absolute path>)" would publish the local folder (and user name)
    w.messages.clear()
    p = st.project
    naca_design = next(d for d in p.designs.values() if d.name.startswith("NACA"))
    a8 = next(s.id for s in p.simulations.values() if s.design_id == naca_design.id)
    cars = sorted((s.id for s in p.simulations.values() if s.design_id != naca_design.id),
                  key=lambda sid: p.designs[p.simulations[sid].design_id].name)

    st.select_design(naca_design.id)
    st.navigate("geometry")
    shot("geometry.png")
    st.select_simulation(cars[1])                      # the 17 degree fastback
    st.navigate("setup")
    pump(1.5)                                          # the planner runs on a worker thread
    shot("setup.png")

    st.select_simulation(a8)
    st.navigate("simulate")
    pump(0.5)
    sp = w.page("simulate")
    sp.field_combo.setCurrentIndex(sp.field_combo.findData("vorticity"))
    progress = {"frac": 0.0, "done": False, "shot": False}
    jobs.jobProgress.connect(lambda _sid, frac, _msg: progress.__setitem__("frac", frac))
    for sig in (jobs.jobFinished, jobs.jobFailed, jobs.jobCancelled):
        sig.connect(lambda *_a: progress.__setitem__("done", True))
    w.run_current()
    loop, t0 = QEventLoop(), time.time()

    def check() -> None:
        if not progress["shot"] and progress["frac"] >= 0.45:
            progress["shot"] = True
            w.grab().save(str(out / "simulate.png"))
            print("saved simulate.png", flush=True)
        if progress["done"] or time.time() - t0 > 600:
            loop.quit()
    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(50)
    loop.exec()
    timer.stop()

    st.navigate("results")
    shot("results.png")
    st.compare_ids[:] = list(cars)
    st.navigate("compare")
    shot("compare.png")
    st.select_simulation(a8)
    st.navigate("report")
    rp = w.page("report")
    pump(0.3)
    for i in range(rp.sim_list.count()):
        it = rp.sim_list.item(i)
        it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) == a8
                         else Qt.CheckState.Unchecked)
    rp.generate()
    t1 = time.time()
    while rp._call is not None and time.time() - t1 < 300:
        pump(0.1)
    w.messages.clear()                  # "Saved <absolute path>" lines: local folders again
    w.statusBar().clearMessage()
    shot("report.png")
    jobs.shutdown()
    w.close()


#: Screens of the placed-physics features, taken from the bundled example projects (their results are
#: real Draft runs, see examples/make_example_projects.py), so no extra solver run is needed
#: except the explore session.
EXTRA = ("results3d", "heat", "physics", "explore", "script")


def take_extra(work: Path, out: Path) -> None:
    """3D smoke and streamlines, a heat-transfer result, the placed-physics editor, explore with
    a drawn obstacle and the Diagnostics panel, and the Script console, from copies of the
    bundled examples (``Sedan in 3D``, ``Heated cylinder``, ``Room with a fan and a heater``)."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["AEROLUME_CONFIG_DIR"] = str(work / "config")
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    from aerolume.app import settings, theme
    settings.use_file(work / "settings.ini")
    theme.apply(app, "light")
    from aerolume.app.main_window import MainWindow
    from aerolume.app.pages.start import example_projects
    from aerolume.project import Project

    def pump(sec: float = 0.4) -> None:
        t = time.time()
        while time.time() - t < sec:
            app.processEvents()
            time.sleep(0.01)

    def shot(name: str, settle: float = 0.8) -> None:
        pump(settle)
        w.grab().save(str(out / f"{name}.png"))
        print("saved", name, flush=True)

    examples = {p.name.removesuffix(".aerolume"): p for p in example_projects()}
    copies = work / "examples"
    shutil.rmtree(copies, ignore_errors=True)
    copies.mkdir(parents=True)

    def open_copy(name: str):
        dst = copies / examples[name].name
        shutil.copytree(examples[name], dst)
        w.set_project(Project.open(dst))
        pump(0.6)
        w.messages.clear()                      # "Opened <project> (<path>)": a local folder
        p = w.state.project
        sim = next(iter(p.simulations.values()))
        w.state.select_design(sim.design_id)
        w.state.select_simulation(sim.id)
        pump(0.3)
        return sim

    out.mkdir(parents=True, exist_ok=True)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    pump(0.5)

    open_copy("Sedan in 3D")
    rp = w.show_page("results")
    pump(2.0)
    if rp.view3d is not None:
        rp.view3d.select_tab("fields")
    shot("results3d", settle=6.0)               # the off-screen VTK views build on first show

    open_copy("Heated cylinder")
    rp = w.show_page("results")
    pump(1.5)
    rp.tabs.setCurrentWidget(rp.heat_tab)
    shot("heat", settle=2.0)

    open_copy("Room with a fan and a heater")
    w.show_page("setup")
    shot("physics", settle=2.5)                 # the planner runs on a worker thread
    w.set_workspace("Live")
    sp = w.show_page("simulate")
    pump(1.0)
    sp.toggle_live()
    t0 = time.time()
    while not sp.live.is_running() and time.time() - t0 < 60:
        pump(0.1)
    pump(4.0)
    sp.draw_strip.buttons["brush"].click()
    pump(0.3)
    xf = sp.view.world_to_screen()
    pts = [xf.map(QPointF(x, y)).toPoint() for x, y in ((1.6, 1.0), (1.9, 1.2), (2.2, 1.4))]
    QTest.mousePress(sp.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pts[0])
    for q in pts[1:]:
        QTest.mouseMove(sp.view, q, 20)
        pump(0.05)
    QTest.mouseRelease(sp.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pts[-1])
    pump(8.0)                                   # the wake of the drawn wall develops
    w.messages.clear()
    shot("explore")
    sp.toggle_live()
    pump(1.0)
    w.set_workspace("Scripting")
    w.show_page("results")
    pump(0.5)
    con = w.script_console
    con.editor.setPlainText(
        "# the room's fan and its temperature field, read through the scripting API\n"
        "r = project.results()[0]\n"
        "print(r.name, '-', 'converged' if r.converged else 'not converged')\n"
        "for f in r.fans:\n"
        "    print({k: round(v, 4) for k, v in f.items() if isinstance(v, float)})\n"
        "show(r.plot('field', field='temperature'))\n")
    con.run()
    t0 = time.time()
    while con.is_running() and time.time() - t0 < 120:
        pump(0.1)
    w.messages.clear()
    shot("script", settle=1.5)
    w.set_workspace("Default")
    w.jobs.shutdown()
    w.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, required=True, help="scratch folder (outside the repository)")
    ap.add_argument("--reuse", action="store_true", help="reuse the project from an earlier run")
    ap.add_argument("--only", choices=("pages", "extra"), default=None,
                    help="only the six workflow pages, or only the placed-physics feature screens")
    a = ap.parse_args(argv)
    work = a.work.resolve()
    if REPO in work.parents or work == REPO:
        ap.error("--work must be outside the repository")
    shots = work / "png"
    if a.only in (None, "pages"):
        project = work / "Aerolume demo.aerolume"
        if not (a.reuse and project.exists()):
            project = build_project(work)
        run_copy = work / "run"
        shutil.rmtree(run_copy, ignore_errors=True)
        shutil.copytree(project, run_copy / project.name)
        take(run_copy / project.name, run_copy, shots)
        for page in PAGES:
            shutil.copyfile(shots / f"{page}.png", SCREENS / f"{page}.png")
            shutil.copyfile(shots / f"{page}.png", DOC_IMG / f"ui_{page}.png")
        shutil.copyfile(shots / "start.png", DOC_IMG / "ui_start.png")
        print(f"copied {len(PAGES)} screens to {SCREENS} and {DOC_IMG}")
    if a.only in (None, "extra"):
        take_extra(work / "extra", shots)
        for name in EXTRA:
            shutil.copyfile(shots / f"{name}.png", SCREENS / f"{name}.png")
            shutil.copyfile(shots / f"{name}.png", DOC_IMG / f"ui_{name}.png")
        print(f"copied {len(EXTRA)} feature screens to {SCREENS} and {DOC_IMG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
