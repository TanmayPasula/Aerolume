"""Report sections for 3D Navier–Stokes results and finite-wing sweeps (DESIGN.md 5.6b, 7).

Content tests check what the report *says* (numbers with their uncertainty, symmetry
zeros, 2D and 3D never compared, the validity numbers of the methodology). Rendering
tests draw real PDF/HTML through VTK off-screen and skip where no OpenGL context exists.
One test runs the real solvers on tiny cases (a coarse CPU sphere, an inviscid wing) and
reports them, so the report is checked against true solver output, not only fixtures.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import report_fixtures as rf
import report_fixtures3d as rf3

from aerolume import report
from aerolume.report import content3d as C3
from aerolume.report import fmt
from aerolume.report import render3d as R
from aerolume.report.common import Entry
from aerolume.report.document import Callout, FigureBlock, KPIRow, Table
from aerolume.report.html import render_html
from aerolume.report.pdf import figure_failure_text, render_pdf
from aerolume.viz import figures as F


def _vtk_ok() -> bool:
    try:
        from aerolume.viz3d import backend_info
        return bool(backend_info()["ok"])
    except Exception:  # noqa: BLE001
        return False


needs_gl = pytest.mark.skipif(not _vtk_ok(), reason="no OpenGL context for VTK off-screen rendering")


@pytest.fixture(scope="module")
def project3d():
    return rf3.make_project3d()


def _pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


# ---------------------------------------------------------------------------
# entries and formatting
# ---------------------------------------------------------------------------

def test_entry_types_keep_2d_and_3d_apart(project3d):
    sph = Entry("sph-a", project3d.simulations["sph-a"], None, project3d.load_result("sph-a"))
    wing = Entry("w-tap", project3d.simulations["w-tap"], None, project3d.load_result("w-tap"))
    assert sph.is_cfd3d and not sph.is_cfd and sph.dimension == 3 and sph.type_label == "CFD 3D"
    assert wing.is_wing and not wing.is_polar and wing.dimension == 3 and wing.type_label == "Wing"


def test_fmt_quotes_round_off_as_zero_and_caps_digits():
    assert fmt.pm(-6.1e-8, 1.7e-8) == "≈ 0" and fmt.mean_only(2e-9, 1e-9) == "≈ 0"
    assert fmt.pm(1.1886079, 2.0e-6) == "1.1886 ± 2.0×10⁻⁶"
    assert fmt.pm(0.0254843, 0.0000613) == "0.02548 ± 0.00006"          # 2D behaviour unchanged
    assert fmt.pct(0.00017, 2, signed=False) == "< 0.01 %"
    assert fmt.eq("CL", "≈ 0") == "CL ≈ 0" and fmt.eq("Cm", "0.131") == "Cm = 0.131"
    assert fmt.solver_text("0.2 cells -- too thin, 10^6") == "0.2 cells – too thin, 10⁶"


# ---------------------------------------------------------------------------
# document content
# ---------------------------------------------------------------------------

def test_3d_document_sections_and_numbers(project3d):
    doc = report.build_document(project3d, list(project3d.simulations), author="QA")
    keys = [s.key for s in doc.sections]
    assert keys == ["summary", "geometry", "setup", "results", "surface", "fields", "convergence", "sensitivity",
                    "polar", "comparison", "methodology", "appendix"]
    titles = {s.key: s.title for s in doc.sections}
    assert titles["polar"] == "Wing polars"
    text = doc.text()
    r = project3d.load_result("sph-a")
    assert fmt.pm(r.cd.mean, r.cd.stderr) in text                         # CD with its standard error
    assert "0 (symmetry)" in text                                          # half-domain van: CS, roll, yaw
    assert "Grid 240×96×96" in text and "forces on the whole body" in text
    assert "per metre of span" not in doc.sections[0].blocks[0].text       # a 3D-only report says newtons
    assert "three-dimensional" in doc.disclaimer
    assert "Absolute forces indicative" in text                            # LES of a vehicle: validity flag
    # six coefficients in the results table, side force row present
    tab = next(b for b in doc.iter_blocks() if isinstance(b, Table) and "whole body" in b.caption)
    assert [str(row[0].text) for row in tab.rows] == list(C3.NAMES3.values())
    # wings: figures of merit and the points table
    assert "Oswald efficiency e (parabolic fit)" in text and "Span efficiency at cruise" in text
    assert any(isinstance(b, KPIRow) for b in doc.sections[0].blocks)
    figs = [b.number for b in doc.iter_blocks() if isinstance(b, FigureBlock)]
    assert figs == list(range(1, len(figs) + 1))


def test_methodology_states_measured_limits(project3d):
    doc = report.build_document(project3d, list(project3d.simulations), sections=["methodology"])
    text = doc.text()
    assert "2.1–2.2" in text and "Ahmed" in text and "0.29" in text        # vehicle drag over-prediction
    assert "Re 20–300" in text and "Johnson & Patel" in text               # what the 3D solver is validated on
    assert "NACA TN 1270" in text and "13–22 %" in text and "not** validated" not in text
    assert "CL,max is not validated" in text.replace("**", "")
    assert "Phillips" in text and "Katz" in text                           # references for VLM and LLT
    heads = [b.text for b in doc.sections[0].blocks if type(b).__name__ == "Heading"]
    assert heads.index("3D Navier–Stokes solver") < heads.index("Validity and limits (3D)")


def test_comparison_never_mixes_2d_and_3d():
    pr = rf3.make_project3d()
    d2 = rf.airfoil_design("2412")
    for sid, alpha, seed in (("a2", 4.0, 1), ("b2", 6.0, 2)):
        s = rf.cfd_sim(d2, f"2D {sid}", alpha=alpha, speed=40.0, sid=sid)
        pr.add(d2, s, rf.airfoil_cfd(d2, s, seed=seed, field_res=(60, 30)))
    doc = report.build_document(pr, list(pr.simulations), sections=["comparison"])
    sec = doc.sections[0]
    heads = [b.text for b in sec.blocks if type(b).__name__ == "Heading"]
    assert heads == ["2D CFD results", "3D CFD results", "Wing polars"]
    tables = [b for b in sec.blocks if isinstance(b, Table)]
    names_3d = {pr.simulations[s].name for s in ("sph-a", "sph-b", "van")}
    t3 = next(t for t in tables if t.caption.startswith("3D CFD"))
    assert {r[0].text for r in t3.rows} == names_3d
    t2 = next(t for t in tables if t.caption.startswith("CFD comparison"))
    assert not ({r[0].text for r in t2.rows} & names_3d)
    assert "compared only within their own group" in doc.text()


def test_zero_baseline_gives_no_percentage(project3d):
    """A symmetric body's lift is round-off: a Δ% against it would be a seven-digit number."""
    pr = rf3.make_project3d()
    pr.results["sph-a"].cl.mean = 3e-9
    doc = report.build_document(pr, ["sph-a", "sph-b"], sections=["comparison"])
    t = next(b for b in doc.iter_blocks() if isinstance(b, Table))
    assert t.rows[1][6].text == fmt.DASH and t.rows[0][5] == "≈ 0"


def test_hotspots_name_places_once_on_a_symmetric_body(project3d):
    r = project3d.load_result("sph-a")
    hs = C3.hotspots3d(r.surface, "cd")
    assert 1 <= len(hs) <= 3 and all("CD" in h for h in hs)
    assert len({h.split(":")[0] for h in hs}) == len(hs)


# ---------------------------------------------------------------------------
# geometry helpers and figures
# ---------------------------------------------------------------------------

def test_plane_section_of_a_box_and_a_triangle_soup():
    from aerolume.geometry3d import box
    b = box(2.0, 1.0, 0.5, center=(0.0, 0.0, 0.25))
    secs = R.plane_section(b.vertices, b.faces, "y", 0.1, values=b.vertices[:, 0])
    assert len(secs) == 1 and secs[0].closed
    assert abs(abs(R.polygon_area(secs[0].points)) - 1.0) < 1e-9          # 2 m x 0.5 m in the (x, z) plane
    assert np.allclose(secs[0].values, secs[0].points[:, 0])               # values interpolated linearly
    soup_v = b.vertices[b.faces].reshape(-1, 3)                            # no shared vertices (STL soup)
    soup_f = np.arange(len(soup_v)).reshape(-1, 3)
    secs = R.plane_section(soup_v, soup_f, "x", 0.3)
    assert len(secs) == 1 and secs[0].closed and abs(abs(R.polygon_area(secs[0].points)) - 0.5) < 1e-9
    assert R.plane_section(b.vertices, b.faces, "z", 5.0) == []


def test_wing_and_history_figures_build():
    d = rf3.wing_design(True)
    s = rf3.wing_sim(d, "w", "w")
    w = rf3.wing_result(s, d)
    for fig in (F.fig_wing_polar(w), F.fig_wing_polar([w, rf3.wing_result(s, d, e_span=0.9)], labels=["a", "b"]),
                F.fig_span_loading(w), F.fig_wing_planform(d.wing, metrics={"mac": 0.8, "mac_le_x": 0.06,
                                                                              "mac_y": 2.2})):
        assert F.figure_to_rgba(fig, dpi=40).shape[2] == 4
    inviscid = rf3.wing_result(s, d, viscous=False)
    assert F.figure_to_rgba(F.fig_wing_polar(inviscid), dpi=40).size
    h = rf3.history3d(3, 0.6, 0.1, lateral=False)
    fig = F.fig_history3d(h, 5.0, None, symmetry_plane=True)
    assert len(fig.axes) == 3                                              # zero lateral loads not drawn
    assert len(F.fig_history3d(rf3.history3d(3, 0.6, 0.1), 5.0, None).axes) == 6


def test_views_figure_draws_an_engineering_drawing():
    img = np.full((60, 120, 3), 255, np.uint8)
    img[20:40, 10:110] = 120
    sq = np.full((60, 60, 3), 255, np.uint8)
    sq[20:40, 20:40] = 120
    names = ["iso", "side", "front", "top", "back"]
    ext = [None, (-0.1, 2.1, -0.1, 0.6), (0.6, -0.6, -0.1, 0.6), (-0.1, 2.1, -0.6, 0.6), (-0.6, 0.6, -0.1, 0.6)]
    color = R.ColorSpec("coolwarm", -1.0, 1.0, "Pressure coefficient Cp", extend="both")
    fig = F.fig_views3d([img, img, sq, img, sq], names, names, ext, color=color, ground=0.0)
    side, front, top = fig.axes[1], fig.axes[2], fig.axes[3]
    # one scale: x of the side view equals x of the top view, z of the side view equals z of the front view
    w_side, w_top = side.get_position().width, top.get_position().width
    assert abs(w_side - w_top) < 1e-9
    assert abs(side.get_position().height - front.get_position().height) < 1e-9
    assert len(fig.axes) == 6                                              # five views + the colour bar


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

@needs_gl
def test_render_views_crops_orthographic_views_to_world_extent():
    from aerolume.geometry3d import box
    b = box(2.0, 1.0, 0.5, center=(0.0, 0.0, 0.25))
    v = R.render_views(lambda s: s.add_meshes([b]) and None, ("iso", "side", "front", "top"), size=500, ground=0.0)
    assert v.axes[1] == ("x", "z") and v.axes[3] == ("x", "y") and v.axes[0] is None
    left, right, bottom, top = v.extents[1]
    m = 0.05 * 2.0
    assert abs(left - (-1.0 - m)) < 0.02 and abs(right - (1.0 + m)) < 0.02
    assert abs(bottom - (0.0 - m)) < 0.02 and abs(top - (0.5 + m)) < 0.02
    lf, rf_, _, _ = v.extents[2]
    assert lf > rf_                                                        # front view: +y to the left
    for img in v.images:
        assert img.ndim == 3 and (img < 250).any()


@needs_gl
def test_3d_report_pdf_and_html_render_without_errors(project3d, tmp_path):
    doc = report.build_document(project3d, ["sph-a", "van", "w-tap"], author="QA")
    info = render_pdf(doc, tmp_path / "r3.pdf", fig_dpi=60)
    assert info["errors"] == [] and info["pages"] >= 12
    assert _pages(tmp_path / "r3.pdf") == info["pages"]
    info = render_html(doc, tmp_path / "r3.html", fig_dpi=50)
    assert info["errors"] == []
    html = (tmp_path / "r3.html").read_text(encoding="utf-8")
    n_figs = sum(1 for b in doc.iter_blocks() if isinstance(b, FigureBlock))
    assert html.count("data:image/png;base64,") == n_figs + 1              # + the cover render


def test_missing_opengl_becomes_a_note(project3d, tmp_path, monkeypatch):
    def unavailable(*a, **k):
        raise R.RenderUnavailable("no OpenGL context")
    monkeypatch.setattr(R, "render_views", unavailable)
    doc = report.build_document(project3d, ["van"], sections=["geometry", "surface"])
    info = render_pdf(doc, tmp_path / "n.pdf", fig_dpi=40)
    assert any("RenderUnavailable" in e for e in info["errors"])
    fb = next(b for b in doc.iter_blocks() if isinstance(b, FigureBlock))
    assert "not available on this computer (no OpenGL context)" in figure_failure_text(fb, R.RenderUnavailable(
        "no OpenGL context"))
    html_info = render_html(doc, tmp_path / "n.html", fig_dpi=40)
    assert "not available on this computer" in (tmp_path / "n.html").read_text(encoding="utf-8")
    assert html_info["errors"]


# ---------------------------------------------------------------------------
# real solver output
# ---------------------------------------------------------------------------

@needs_gl
def test_report_of_real_tiny_runs(tmp_path, monkeypatch):
    """A coarse CPU sphere run and an inviscid wing sweep, through the real solvers, then the report."""
    from aerolume.core import fluid
    from aerolume.geometry3d import design_from_meshes, sphere
    from aerolume.model import CFDSettings, FlowConditions, Simulation, WingSettings
    from aerolume.simulation import plan as planmod
    from aerolume.simulation import run_cfd
    from aerolume.simulation.plan import plan_simulation
    from aerolume.simulation.wing import run_wing
    for name in ("_RATE_CACHE3D", "_SPEED_CACHE3D", "_LOOP_CACHE3D"):
        monkeypatch.setattr(planmod, name, {})
    monkeypatch.setattr(planmod, "_CALIBRATION_FILE", None)
    air = fluid.air_isa(0.0)
    pr = rf.FakeProject("Real tiny runs")
    d = design_from_meshes([sphere(1.0, 24)], "Sphere", kind="generic")
    sim = Simulation("Sphere · Re 50", d.id, status="done",
                     conditions=FlowConditions(air, 50 * air.viscosity / air.density, 0.0),
                     cfd=CFDSettings(quality="draft", device="cpu", cells_per_ref=6, duration=2.5, average_from=1.0))
    pr.add(d, sim, run_cfd(plan_simulation(d, sim)))
    dw = rf3.wing_design(True)
    ws = Simulation("Wing · inviscid", dw.id, analysis="wing", status="done", conditions=FlowConditions(air, 30.0, 0.0),
                    wing=WingSettings(alpha_start=0, alpha_end=8, alpha_step=4, method="vlm", viscous=False))
    pr.add(dw, ws, run_wing(dw, ws))
    doc = report.build_document(pr, [sim.id, ws.id])
    text = doc.text()
    assert "3D incompressible Navier–Stokes" in text and "Vortex lattice" in text
    info = render_pdf(doc, tmp_path / "real.pdf", fig_dpi=50)
    assert info["errors"] == []
    assert not any(isinstance(b, Callout) and "could not be produced" in b.text for b in doc.iter_blocks())
