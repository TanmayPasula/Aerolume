"""Tests for aerolume.report: content, layout engine, PDF and HTML output."""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import report_fixtures as rf

from aerolume import report
from aerolume.report import fmt
from aerolume.report.document import Callout, FigureBlock, KPIRow, Table
from aerolume.report.html import render_html
from aerolume.report.pdf import TW, _table_pieces, flow, render_pdf, text_width, wrap

#: Where the slow test leaves its sample report for review. Outside the repository by default
#: (the repository may live in a synced folder); AEROLUME_SCRATCH points it elsewhere.
SCRATCH = Path(os.environ.get("AEROLUME_SCRATCH") or tempfile.gettempdir()) / "aerolume-report-sample"


def _pdf_pages(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


@pytest.fixture(scope="module")
def project():
    return rf.make_project(field_res=(120, 60))


@pytest.fixture(scope="module")
def small():
    """One airfoil design: two CFD runs (for comparison) and one polar, low-resolution fields."""
    pr = rf.FakeProject("Small study")
    d = rf.airfoil_design("2412")
    s1 = rf.cfd_sim(d, "A · α 4°", 40.0, 4.0, "a")
    s2 = rf.cfd_sim(d, "B · α 6°", 40.0, 6.0, "b")
    p = rf.polar_sim(d, "Polar", "p")
    pr.add(d, s1, rf.airfoil_cfd(d, s1, seed=1, with_sensitivity=True, field_res=(80, 40)))
    pr.add(d, s2, rf.airfoil_cfd(d, s2, seed=2, field_res=(80, 40)))
    pr.add(d, p, rf.airfoil_polar(d, p))
    return pr


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def test_fmt_uncertainty_rounding():
    assert fmt.pm(0.0254843, 0.0000613) == "0.02548 ± 0.00006"
    assert fmt.pm(0.58617, 0.00243) == "0.5862 ± 0.0024"
    assert fmt.pm(12.3456, 0.9) == "12.3 ± 0.9"
    assert fmt.pm(1.5, float("nan")) == "1.500"
    assert fmt.pm(float("nan"), 0.1) == fmt.DASH
    assert fmt.mean_only(0.0254843, 0.0000613) == "0.02548"


def test_fmt_misc():
    assert fmt.reynolds(8.216e5) == "8.22 × 10⁵"
    assert fmt.reynolds(950) == "950"
    assert fmt.reynolds(float("nan")) == fmt.DASH
    assert fmt.pct(13.84) == "+13.8 %"
    assert fmt.seconds(42) == "42 s"
    assert fmt.seconds(184) == "3 min 04 s"
    assert fmt.seconds(4000) == "1 h 06 min"


# ---------------------------------------------------------------------------
# layout engine
# ---------------------------------------------------------------------------

def test_wrap_respects_width_and_bold_runs():
    text = "The **quick brown fox** jumps over the lazy dog. " * 8
    lines = wrap(text, 200.0, 9.4)
    assert len(lines) > 3
    for ln in lines:
        assert sum(text_width(t, 9.4, b) for t, b in ln) <= 200.0 + 1e-6
    assert any(b for ln in lines for _, b in ln)
    joined = " ".join("".join(t for t, _ in ln) for ln in lines).split()
    assert joined == text.replace("**", "").split()


def test_wrap_breaks_overlong_token():
    lines = wrap("x" * 400, 100.0, 9.0)
    assert len(lines) > 2
    assert all(text_width(ln[0][0], 9.0) <= 100.0 for ln in lines)


def test_text_width_counts_spaces():
    assert text_width("ab ", 10.0) > text_width("ab", 10.0) + 1.0
    assert text_width("abc" * 20, 10.0) == pytest.approx(20 * text_width("abc", 10.0), rel=0.02)


def test_long_table_splits_with_repeated_header():
    rows = [[str(i), f"{i * 0.1:.3f}"] for i in range(120)]
    pieces = _table_pieces(Table(["α", "Cl"], rows, caption="long", number=1))
    pages = flow(pieces)
    assert len(pages) >= 3
    header = pieces[1]
    for pg in pages:                       # every page starts its table part with the header
        assert any(pc is header for pc, _ in pg.placed)
    placed_rows = sum(1 for pg in pages for pc, _ in pg.placed if pc is not header and pc not in pieces[:1])
    assert placed_rows == 120


def test_table_columns_fit_text_width():
    t = Table(["Simulation", "Value"], [["a very long simulation name " * 8, "1.0"]])
    pieces = _table_pieces(t)
    assert pieces[-1].height > pieces[0].height   # the long cell wrapped instead of overflowing
    assert TW > 400


# ---------------------------------------------------------------------------
# document content
# ---------------------------------------------------------------------------

def test_sections_constant_matches_design():
    assert report.SECTIONS == ("summary", "geometry", "setup", "results", "surface", "fields", "heat", "flowvis",
                               "fans", "convergence", "grid", "sensitivity", "polar", "comparison", "methodology")
    assert set(report.SECTIONS) <= set(report.SECTION_TITLES)


#: sections that need placed physics (DESIGN.md 10) or a grid study: absent from the plain fixtures
PHYSICS_SECTIONS = ("heat", "flowvis", "fans", "grid")


def test_document_content(project):
    ids = list(project.simulations)
    doc = report.build_document(project, ids, author="Tester", title="My report")
    keys = [s.key for s in doc.sections]
    assert keys == [*(k for k in report.SECTIONS if k not in PHYSICS_SECTIONS), "appendix"]
    assert [s.number for s in doc.sections] == [str(i) for i in range(1, 12)] + ["A"]
    text = doc.text()
    r = project.load_result("s-2412-cfd")
    assert fmt.pm(r.cd.mean, r.cd.stderr) in text           # key number with its uncertainty
    assert "8.22 × 10⁵" in text and "2.00 × 10⁴" in text     # Re real and simulated side by side
    assert "Beyond benchmark range" in text
    assert "has no result yet" in text                       # the queued simulation
    assert "Tester" in text and doc.title == "My report"
    assert "Section 11" in doc.disclaimer
    # figures and tables are numbered in reading order
    figs = [b.number for b in doc.iter_blocks() if isinstance(b, FigureBlock)]
    assert figs == list(range(1, len(figs) + 1))
    tabs = [b.number for b in doc.iter_blocks() if isinstance(b, Table) and b.caption]
    assert tabs == list(range(1, len(tabs) + 1))


def test_document_inapplicable_sections_are_omitted(project):
    doc = report.build_document(project, ["s-2412-polar"])
    keys = [s.key for s in doc.sections]
    assert keys == ["summary", "geometry", "setup", "polar", "methodology", "appendix"]
    assert any(isinstance(b, KPIRow) for b in doc.sections[0].blocks)


def test_document_explicit_sections_keep_order_and_explain(project):
    doc = report.build_document(project, ["s-2412-polar"], sections=["comparison", "summary", "results"])
    assert [s.key for s in doc.sections] == ["summary", "results", "comparison"]
    assert "None of the selected simulations has a CFD result" in doc.text()
    assert "at least two" in doc.text()


def test_document_validation(project):
    with pytest.raises(ValueError):
        report.build_document(project, [])
    with pytest.raises(ValueError):
        report.build_document(project, ["s-2412-cfd"], sections=["summary", "bogus"])
    with pytest.raises(KeyError):
        report.build_document(project, ["nope"])


def test_comparison_marks_significance(project):
    doc = report.build_document(project, ["s-2412-cfd", "s-4412-cfd"], sections=["comparison"])
    tables = [b for b in doc.iter_blocks() if isinstance(b, Table)]
    cmp = tables[0]
    assert cmp.rows[0][4].text == "baseline"
    delta = cmp.rows[1][4]
    assert delta.text.startswith("+13.8 %") and delta.tone == "bad" and "✱" in delta.text


def test_unreadable_result_is_reported_not_fatal(project):
    class Broken(rf.FakeProject):
        def load_result(self, sim_id):
            raise OSError("arrays.npz is damaged")
    pr = Broken("Broken")
    pr.designs, pr.simulations = project.designs, project.simulations
    doc = report.build_document(pr, ["s-2412-cfd"])
    assert "could not be read" in doc.text() and "damaged" in doc.text()


def test_vehicle_report_mentions_2d_limits(project):
    doc = report.build_document(project, ["s-saloon"])
    text = doc.text()
    assert "side profile" in text and "downforce" in text
    assert "Contribution of each body" in text


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_pdf_small(small, tmp_path):
    doc = report.build_document(small, list(small.simulations), author="QA")
    out = tmp_path / "r.pdf"
    info = render_pdf(doc, out, fig_dpi=60)
    data = out.read_bytes()
    assert data[:5] == b"%PDF-"
    assert info["errors"] == []
    assert _pdf_pages(out) == info["pages"] >= 8
    assert b"/FontFile2" in data            # TrueType fonts embedded (pdf.fonttype 42)


def test_generate_pdf_and_html(small, tmp_path):
    p = report.generate(small, ["a"], tmp_path / "one", fmt="pdf", sections=["summary", "results", "methodology"],
                        title="Quick look", author="QA")
    assert p.suffix == ".pdf" and p.exists()
    assert b"Quick look" in p.read_bytes() or b"Quick" in p.read_bytes()   # metadata title
    h = report.generate(small, ["a", "b"], tmp_path / "two.html", fmt="html", sections=["summary", "comparison"])
    html = h.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "<script" not in html
    assert not re.search(r'(src|href)="https?://', html)    # self-contained
    assert html.count("data:image/png;base64,") >= 3        # cover + bars + Cp overlay
    assert 'id="sec-comparison"' in html and "baseline" in html
    with pytest.raises(ValueError):
        report.generate(small, ["a"], tmp_path / "x", fmt="docx")


def test_broken_figure_becomes_note(small, tmp_path):
    doc = report.build_document(small, ["a"], sections=["summary"])

    def boom(**_):
        raise RuntimeError("no data")
    doc.sections[0].blocks.append(FigureBlock(boom, caption="Broken figure", number=99))
    info = render_pdf(doc, tmp_path / "b.pdf", fig_dpi=50)
    assert len(info["errors"]) == 1 and "no data" in info["errors"][0]
    info = render_html(doc, tmp_path / "b.html", fig_dpi=50)
    assert "could not be produced" in (tmp_path / "b.html").read_text(encoding="utf-8")


def test_callout_and_kpi_blocks_render(tmp_path, small):
    doc = report.build_document(small, ["p"], sections=["summary"])
    doc.sections[0].blocks.append(Callout("A **very** important note. " * 20, "bad", title="Heads up"))
    info = render_pdf(doc, tmp_path / "c.pdf", fig_dpi=50, png_dir=tmp_path / "png")
    assert info["errors"] == []
    pngs = sorted((tmp_path / "png").glob("page-*.png"))
    assert len(pngs) == info["pages"]


@pytest.mark.slow
def test_full_sample_report(project):
    """The full fixture project in both formats; also leaves a sample (SCRATCH) for review."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    doc = report.build_document(project, list(project.simulations), author="A. Designer",
                                filename="sample.pdf")
    info = render_pdf(doc, SCRATCH / "sample.pdf", png_dir=SCRATCH / "pages")
    assert info["errors"] == [] and info["pages"] >= 20
    info = render_html(doc, SCRATCH / "sample.html")
    assert info["errors"] == []
    html = (SCRATCH / "sample.html").read_text(encoding="utf-8")
    n_figs = sum(1 for b in doc.iter_blocks() if isinstance(b, FigureBlock))
    assert html.count("data:image/png;base64,") == n_figs + 1
    assert np.isfinite(_pdf_pages(SCRATCH / "sample.pdf"))


def test_duplicate_simulation_names_are_labelled_with_their_design():
    """Two car designs each with a "CFD · 30 m/s" must not read the same in every table."""
    from types import SimpleNamespace

    from aerolume.report.content import Entry, _disambiguate
    def entry(sid, name, design):
        return Entry(sid, SimpleNamespace(name=name), SimpleNamespace(name=design), None)
    es = [entry("a", "CFD · 30 m/s", "Sedan"), entry("b", "CFD · 30 m/s", "Fastback"), entry("c", "Polar", "Wing")]
    _disambiguate(es)
    assert [e.name for e in es] == ["Sedan · CFD · 30 m/s", "Fastback · CFD · 30 m/s", "Polar"]
    same = [entry("a", "Run", "Sedan"), entry("b", "Run", "Sedan")]
    _disambiguate(same)
    assert [e.name for e in same] == ["Sedan · Run (1)", "Sedan · Run (2)"]


def test_solver_messages_get_typeset_powers_of_ten():
    assert fmt.solver_text("Re = 1.00×10^6, dt 3×10^-4") == "Re = 1.00×10⁶, dt 3×10⁻⁴"
