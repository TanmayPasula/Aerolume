"""Tests for the website build: the Markdown converter (including the plain-text maths and
cross-references), the repository-fact parsers, the validation table read from the docs,
and a full build into a temporary folder (links, anchors, placeholders, the Artifact copy).

Run: python -m pytest website/tests/test_build.py -q -p no:cacheprovider
"""
from __future__ import annotations

import importlib
import itertools
import re
import shutil
import sys
from pathlib import Path

import pytest

WEBSITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBSITE))

build = importlib.import_module("build")          # website/build.py, not an installed package
MarkdownRenderer, markdown_to_html, slugify = build.MarkdownRenderer, build.markdown_to_html, build.slugify
NO_REPO_ENV: dict[str, str] = {}                  # an environment without any repository address


def md(text: str, **kw) -> str:
    return markdown_to_html(text, **kw).html


# --------------------------------------------------------------------------- headings
def test_slugify_matches_github_style():
    assert slugify("1. Equations and scaling") == "1-equations-and-scaling"
    assert slugify("12. Limitations \u2014 what to trust") == "12-limitations-what-to-trust"
    assert slugify("NACA 0012, Re 6\u00d710\u2076, tripped") == "naca-0012-re-6106-tripped"
    assert slugify("<code>x</code> & y") == "x-y"
    assert slugify("!!!") == "section"


def test_headings_get_unique_ids_title_and_toc():
    r = markdown_to_html("# Title\n\n## Part\n\n### Detail\n\n## Part\n\n#### Deep\n")
    assert r.title == "Title"
    assert '<h1 id="title">Title</h1>' in r.html
    assert '<h2 id="part">' in r.html and '<h2 id="part-1">' in r.html
    assert [(e.level, e.id) for e in r.toc] == [(2, "part"), (3, "detail"), (2, "part-1")]
    assert 'href="#detail"' in r.html               # self-link anchor
    assert "<h4" in r.html and all(e.level < 4 for e in r.toc)


def test_heading_with_inline_markup():
    r = markdown_to_html("## The `step` *function*\n")
    assert "<code>step</code>" in r.html and "<em>function</em>" in r.html
    assert r.toc[0].text == "The step function" and r.toc[0].id == "the-step-function"


# --------------------------------------------------------------------------- inline
def test_emphasis_and_code_spans():
    out = md("**bold** and *it* and _also_ but snake_case_name and `a*b*c` stay")
    assert "<strong>bold</strong>" in out
    assert "<em>it</em>" in out and "<em>also</em>" in out
    assert "snake_case_name" in out
    assert "<code>a*b*c</code>" in out


def test_html_is_escaped_except_whitelisted_tags():
    out = md("C<sub>d</sub> <script>alert(1)</script> a < b & c")
    assert "C<sub>d</sub>" in out
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "a &lt; b &amp; c" in out


def test_backslash_escapes():
    assert "*not em*" in md(r"\*not em\*")


def test_links_images_and_autolinks_go_through_the_resolver():
    seen = []

    def link(url: str) -> str:
        seen.append(url)
        return "X/" + url

    out = md('See [the *theory*](theory.md#2-inputs "Theory") and <https://example.org/a_b>.\n'
             "Inline ![alt](img/p.png) image.", link=link)
    assert '<a href="X/theory.md#2-inputs" title="Theory">the <em>theory</em></a>' in out
    assert '<a href="X/https://example.org/a_b">https://example.org/a_b</a>' in out
    assert '<img src="X/img/p.png" alt="alt" loading="lazy">' in out
    assert sorted(seen) == ["https://example.org/a_b", "img/p.png", "theory.md#2-inputs"]


def test_resolver_returning_none_keeps_the_text_without_a_link():
    out = md("See [the source](../src/x.py) here.", link=lambda url: None)
    assert "<a" not in out and "See the source here." in out


def test_lone_image_becomes_a_figure():
    out = md("![NACA 0012 lift and drag](img/airfoil_0012.png)")
    assert out.startswith("<figure>") and "<figcaption>NACA 0012 lift and drag</figcaption>" in out


def test_figure_hook_replaces_a_lone_image():
    out = md("![Lift](img/a.png)", figure=lambda raw, src, alt: f'<figure class="x">{raw}|{src}|{alt}</figure>')
    assert out.strip() == '<figure class="x">img/a.png|img/a.png|Lift</figure>'


def test_underscores_inside_link_urls_are_not_emphasis():
    out = md("[a](docs/user_guide_simulation.md) and [b](x_y_z.md)")
    assert 'href="docs/user_guide_simulation.md"' in out and "<em>" not in out


# --------------------------------------------------------------------------- cross-references
def _xref(name: str, section: str | None):
    if name.endswith("theory_airfoil.md"):
        return "theory-airfoil.html" + (f"#{section}-x" if section else ""), \
            "Airfoil analysis" + (f", §{section}" if section else "")
    return None


def test_bare_doc_mentions_become_titled_links():
    out = md("the method in theory_airfoil.md (theory_airfoil.md, section 5), see docs/theory_airfoil.md "
             "section 4; DESIGN.md §3 and `theory_airfoil.md` stay", xref=_xref)
    assert '<a href="theory-airfoil.html">Airfoil analysis</a> (' in out
    assert '<a href="theory-airfoil.html#5-x">Airfoil analysis, §5</a>' in out
    assert '<a href="theory-airfoil.html#4-x">Airfoil analysis, §4</a>' in out
    assert "DESIGN.md §3" in out and "<code>theory_airfoil.md</code>" in out


def test_link_text_that_is_a_file_name_gets_the_page_title():
    out = md("[theory_airfoil.md](theory_airfoil.md) and [the theory](theory_airfoil.md)", xref=_xref)
    assert ">Airfoil analysis</a>" in out and ">the theory</a>" in out and ".md</a>" not in out


# --------------------------------------------------------------------------- plain-text maths
@pytest.mark.parametrize("span, is_math", [
    ("ν_t = (C_s Δ)² √(2 S:S)", True),
    ("u^{n+1}", True),
    ("u_s", True),
    ("C_w = 0.5", True),
    ("Im Z_{N/2} = −Re Z_{N/2}", True),
    ("set_inflow(re=…)", False),
    ("w = mask_width · h", False),
    ("tests/test_ns2d_solver.py", False),
    ("Cd_surf", False),
    ("$x$", False),
    ("upwind3", False),
])
def test_math_span_classifier(span, is_math):
    assert build.is_math_span(span) is is_math


def test_format_math_sub_superscripts_and_italic_variables():
    assert build.format_math("u^{n+1}") == "<i>u</i><sup><i>n</i>+1</sup>"
    assert build.format_math("Z_{N/2}") == "<i>Z</i><sub><i>N</i>/2</sub>"
    assert build.format_math("ν_t S") == "ν<sub><i>t</i></sub> <i>S</i>"
    assert build.format_math("(S^d:S^d)^{3/2}") == "(<i>S</i><sup><i>d</i></sup>:<i>S</i><sup><i>d</i></sup>)<sup>3/2</sup>"
    assert build.format_math("Re < x") == "Re &lt; <i>x</i>"      # words upright, HTML escaped
    assert "{" not in build.format_math("e^{−iπk/2N} V_k")


def test_code_span_with_a_formula_renders_as_maths():
    out = md("WALE: `ν_t = (C_w Δ)² (S^d:S^d)^{3/2}` and code `set_inflow(re=…)`.")
    assert '<span class="math">ν<sub><i>t</i></sub>' in out and "^{" not in out
    assert "<code>set_inflow(re=…)</code>" in out


def test_equation_fence_is_typeset_and_aligned_on_relations():
    text = "```\n∂u/∂t + ∇·(u u) = −∇p\n∇·u = 0\n```\n"
    out = md(text)
    assert out.startswith('<div class="eq" role="math" aria-label="∂u/∂t + ∇·(u u) = −∇p ; ∇·u = 0">')
    assert out.count('<td class="rel">=</td>') == 2 and "<pre>" not in out


def test_piecewise_equation_continues_the_left_side():
    text = ("```\nχ(d) = 1                    d ≤ −w\n     = ½ (1 − sin(d))     |d| < w\n"
            "flux   = V · c\n```\n")
    out = md(text)
    rows = re.findall(r"<tr>(.*?)</tr>", out)
    assert len(rows) == 3
    assert rows[0].startswith('<td class="l">χ(<i>d</i>)</td><td class="rel">=</td><td class="r">1</td>')
    assert rows[1].startswith('<td class="l"></td><td class="rel">=</td>')
    assert rows[2].startswith('<td class="l">flux</td><td class="rel">=</td>')


def test_code_fences_stay_code():
    assert md("```\npython validation/airfoil_validation.py\n```\n").startswith("<pre><code>")
    assert md("```\nphi = [7(b + c) - (a + d)]/12\n```\n").startswith("<pre><code>")   # no maths glyphs


def test_box_drawing_diagram_holds_glyphs_to_one_cell():
    out = md("```\n┌──┐\n│▶ │\n└──┘\n```\n")
    assert out.startswith('<pre class="diagram"><code>')
    assert '<span class="g">▶</span>' in out and '<span class="g">│' not in out


# --------------------------------------------------------------------------- $ maths (MathJax)
def test_math_is_passed_to_mathjax():
    r = markdown_to_html("Energy $E = \\tfrac{1}{2} u^2$ and\n\n$$\n\\nabla \\cdot u = 0\n$$\n")
    assert r.has_math
    assert "\\(E = \\tfrac{1}{2} u^2\\)" in r.html
    assert '<div class="math-block">\\[\\nabla \\cdot u = 0\\]</div>' in r.html


def test_dollar_amounts_and_code_are_not_math():
    r = markdown_to_html("It costs $5 and $10, and `$x$` is code.")
    assert not r.has_math
    assert "$5 and $10" in r.html and "<code>$x$</code>" in r.html


def test_math_content_is_html_escaped():
    r = markdown_to_html("$a < b$")
    assert "\\(a &lt; b\\)" in r.html


# --------------------------------------------------------------------------- blocks
def test_fenced_code_keeps_content_and_language():
    out = md("```python\nif a < b:\n    print('*x*')\n```\n")
    assert out.startswith('<pre><code class="language-python">')
    assert "if a &lt; b:\n    print('*x*')" in out and "<em>" not in out


def test_tilde_fence_and_unclosed_fence():
    assert "<pre><code>a\n```\nb</code></pre>" in md("~~~\na\n```\nb\n~~~\n")
    assert "<pre><code>tail</code></pre>" in md("```\ntail\n")


def test_indented_code_block():
    out = md("Text\n\n    ue = ue_inviscid + D\n    second\n\nAfter")
    assert "<pre><code>ue = ue_inviscid + D\nsecond</code></pre>" in out
    assert out.endswith("<p>After</p>\n")


def test_table_alignment_and_pipes_in_code():
    text = ("| check | result | bar |\n|-------|-------:|:---:|\n"
            "| divergence `max|∇·u| h/U`, float64 | 4.4e-16 | < 1e-10 |\n| a \\| b | 2 |\n")
    out = md(text)
    assert '<div class="table-scroll"><table>' in out
    assert '<th class="r" scope="col">result</th>' in out and '<th class="c" scope="col">bar</th>' in out
    assert '<span class="math">max|∇·<i>u</i>| <i>h</i>/<i>U</i></span>' in out    # pipes kept inside the span
    assert '<td class="r">4.4e-16</td>' in out
    assert "<td>a | b</td>" in out
    assert out.count("<tr>") == 3                   # short row padded, not dropped


def test_split_row_handles_unmatched_backtick():
    assert MarkdownRenderer.split_row("| a ` b | c |") == ["a ` b", "c"]


def test_nested_tight_lists():
    out = md("* one\n* two\n  * two-a\n  * two-b\n* three\n")
    assert out.count("<ul>") == 2
    assert "<li>one</li>" in out and "<p>" not in out
    assert re.search(r"<li>two\s*<ul>\s*<li>two-a</li>", out)


def test_ordered_list_start_and_lazy_continuation():
    out = md("3. third\ncontinued lazily\n4. fourth\n")
    assert out.startswith('<ol start="3">')
    assert "<li>third\ncontinued lazily</li>" in out


def test_loose_list_item_with_fenced_code_like_the_theory_doc():
    text = ("* `central`: 2nd order.\n"
            "* `upwind3` (default): reconstruction\n  for flow from the left. Written branch-free:\n\n"
            "  ```\n  phi = [7(b + c) - (a + d)]/12\n  ```\n\n  i.e. a 4th-order central flux.\n"
            "\n**Diffusion**: next paragraph.\n")
    out = md(text)
    assert "<li><p><code>upwind3</code> (default): reconstruction\nfor flow from the left." in out
    assert "<pre><code>phi = [7(b + c) - (a + d)]/12</code></pre>" in out
    assert "<p>i.e. a 4th-order central flux.</p></li>" in out
    assert out.rstrip().endswith("<p><strong>Diffusion</strong>: next paragraph.</p>")


def test_blockquote_and_rule():
    out = md("> quoted *text*\n> more\n\n---\n\nafter")
    assert "<blockquote>\n<p>quoted <em>text</em>\nmore</p>\n</blockquote>" in out
    assert "<hr>" in out


def test_hard_break_and_comment_skipped():
    out = md("line one  \nline two\n\n<!-- hidden\nstill hidden -->\n\nshown")
    assert "line one<br>\nline two" in out
    assert "hidden" not in out and "<p>shown</p>" in out


def test_unfilled_generator_placeholder_renders_as_pending():
    out = md("Measured throughput:\n\nPERF_TABLE\n\nCPU notes.")
    assert '<p class="placeholder">Pending' in out and "PERF_TABLE" in out


def test_real_documents_convert_without_leftover_markdown():
    for name in ("docs/theory_ns2d.md", "docs/theory_airfoil.md", "docs/validation_airfoil.md",
                 "docs/ui_guidelines.md", "CONTRIBUTING.md"):
        path = build.REPO / name
        if not path.exists():
            continue
        r = markdown_to_html(path.read_text(encoding="utf-8"))
        text = re.sub(r"<pre.*?</pre>|<code>.*?</code>", "", r.html, flags=re.DOTALL)
        assert r.title, name
        assert "```" not in text, name
        assert not re.search(r"^\s*\|[-:| ]+\|\s*$", text, flags=re.MULTILINE), f"unconverted table in {name}"
        assert not re.search(r"(?<![\w*])\*\*\w", text), f"unconverted bold in {name}"
        assert not re.search(r"[\^_]\{", re.sub(r"<[^>]+>", "", text)), f"LaTeX braces left as text in {name}"
        assert "\x00" not in r.html


# --------------------------------------------------------------------------- repository facts
def test_fmt_reynolds():
    assert build.fmt_reynolds(5000) == "5000"
    assert build.fmt_reynolds(2e5) == "2&times;10<sup>5</sup>"
    assert build.fmt_reynolds(1e6) == "10<sup>6</sup>"
    assert build.fmt_reynolds(8.2e5) == "8.2&times;10<sup>5</sup>"


def test_find_repo_url_sources_in_order(tmp_path):
    assert build.find_repo_url(tmp_path, {}) is None
    assert build.find_repo_url(tmp_path, {"GITHUB_REPOSITORY": "o/r"}) == "https://github.com/o/r"
    assert build.find_repo_url(tmp_path, {"AEROLUME_REPO_URL": "https://git.example/x.git/",
                                          "GITHUB_REPOSITORY": "o/r"}) == "https://git.example/x"
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n[project.urls]\n'
                                             'Repository = "https://github.com/a/b"\n', encoding="utf-8")
    assert build.find_repo_url(tmp_path, {"AEROLUME_REPO_URL": "https://other"}) == "https://github.com/a/b"


def test_find_repo_url_falls_back_to_citation(tmp_path):
    (tmp_path / "CITATION.cff").write_text('cff-version: 1.2.0\nrepository-code: "https://github.com/c/d.git"\n'
                                           'url: "https://c.github.io/d/"\n', encoding="utf-8")
    assert build.find_repo_url(tmp_path, {}) == "https://github.com/c/d"
    assert build.find_repo_url(tmp_path, {"GITHUB_REPOSITORY": "o/r"}) == "https://github.com/o/r"
    assert build.citation_field(tmp_path, "url") == "https://c.github.io/d/"
    assert build.citation_field(tmp_path, "license") is None


def test_parse_perf_table_inherits_device_names():
    text = ("| device | grid | options | steps/s | target |\n|---|---|---|--:|--:|\n"
            "| CUDA, RTX 4060 Ti | 1024×512 | default | **493** | ≥ 300 |\n"
            "| CUDA | 2048x1024 | default | 75 | |\n| CPU, i7-13700F | 512×256 | 4 threads | **61** | ≥ 15 |\n")
    rows = build.parse_perf_table(text)
    assert rows == [("RTX 4060 Ti", "GPU", 1024, 512, "default", 493.0),
                    ("RTX 4060 Ti", "GPU", 2048, 1024, "default", 75.0),
                    ("i7-13700F", "CPU", 512, 256, "4 threads", 61.0)]
    assert build.parse_perf_table("no table here") == []


def test_perf_facts_quote_only_measured_hardware():
    rows = [("RTX 4060 Ti", "GPU", 1024, 512, "default", 493.0), ("i7-13700F", "CPU", 512, 256, "4 threads", 61.0),
            ("i7-13700F", "CPU", 512, 256, "1 thread", 48.0)]
    f = build.perf_facts(rows)
    assert f["PERF_GPU_DEFAULT"] == "493" and f["PERF_GPU_NAME"] == "RTX 4060 Ti"
    assert "<b>61</b> steps/s" in f["PERF_CPU"] and "desktop i7-13700F" in f["PERF_CPU"] and "4 threads" in f["PERF_CPU"]
    assert "laptop" not in f["PERF_CPU"]                      # no laptop was measured
    assert build.perf_facts([])["PERF_CPU"] == "every analysis also runs on the CPU"


def test_doc_tables_keep_their_heading():
    tables = build.doc_tables("## A\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\n```\n| no | table |\n|--|--|\n```\n")
    assert len(tables) == 1 and tables[0].heading == "A" and tables[0].row("1") == ["1", "2"]
    assert tables[0].col("y") == 1


def _record(tmp_path, cases, date="2026-10-30 23:05"):
    """A validation record in a fake repository: cases are (key, title, area, [checks])."""
    import json
    d = tmp_path / "validation" / "records"
    d.mkdir(parents=True, exist_ok=True)
    results = [{"key": k, "title": t, "area": a, "checks": cs} for k, t, a, cs in cases]
    (d / "validation-record.json").write_text(json.dumps({"meta": {"date": date, "aerolume": "9.9", "gpu": "G"},
                                                          "results": results}), encoding="utf-8")
    return build.ValidationRecord(tmp_path)


def _chk(q, v, lo, hi, ok, ref=None, ref_text="1.0"):
    return {"quantity": q, "value": v, "lo": lo, "hi": hi, "ref": ref, "ref_text": ref_text, "source": "exact", "passed": ok}


def test_ledger_shows_each_areas_weakest_pass_and_every_failure(tmp_path):
    """The selection is mechanical, so it cannot flatter: per area the passing check nearest its
    band edge, then each failing check followed by the same quantity on the case's other grids."""
    rec = _record(tmp_path, [
        ("a", "Case A", "flow2d", [_chk("Cd", 1.0, 0.9, 1.1, True), _chk("St", 0.208, 0.2, 0.21, True)]),
        ("square", "Square", "flow2d", [_chk("mean Cd (32 cells/D)", 1.7, 1.4, 1.6, False),
                                        _chk("mean Cd (48 cells/D)", 1.55, 1.4, 1.6, True)]),
        ("t", "Plate", "thermal", [_chk("Nu", 5.0, 4.0, 6.0, True)]),
    ])
    assert rec.ok and rec.passed == 4 and rec.total == 5 and rec.failing_cases() == ["Square"]
    led = dict(rec.ledger())
    assert [c.quantity for c in led["flow2d"]] == ["St", "mean Cd (32 cells/D)", "mean Cd (48 cells/D)"]
    assert [c.quantity for c in led["thermal"]] == ["Nu"]
    assert rec.run_date() == "30 October 2026"
    rows = build.ledger_rows(rec, {"exact": "Exact solution"}, lambda key: "docs/validation.html#" + build.slugify(key))
    assert rows.count('<tr class="group">') == 2 and rows.count('<tr class="fail">') == 1
    assert "Band revised after this result was known" in rows        # the square cylinder's band (REVISED_BANDS)
    assert "1 of 2 checks pass" not in rows and "3 of 4 checks pass" in rows


def test_missing_record_leaves_the_checks_out(tmp_path):
    rec = build.ValidationRecord(tmp_path)
    assert not rec.ok and rec.problems and build.validation_tally(rec) == ""
    assert build.ledger_rows(rec, {}, lambda k: k) == ""


def test_band_gauge_places_the_value_in_its_band():
    c = build.Check("flow2d", "a", "A", "Cd", 1.05, 1.0, 1.1, 1.05, "1.05", "exact", True)
    g = build.band_gauge(c)
    assert 'style="--p:0.500"' in g and 'aria-hidden="true"' in g
    far = build.Check("flow2d", "a", "A", "Cd", 9.0, 1.0, 1.1, 1.05, "1.05", "exact", False)
    assert 'style="--p:1.250"' in build.band_gauge(far)              # a far miss is clamped beyond the bracket
    assert build.band_gauge(build.Check("x", "a", "A", "q", 1.0, None, None, None, "", "", True)) == ""


def test_the_real_record_feeds_the_home_page():
    rec = build.ValidationRecord(build.REPO)
    assert rec.ok, rec.problems
    assert rec.passed <= rec.total and all(rows for _, rows in rec.ledger())
    # the report still says that the square cylinder's band was revised (the page marks that row)
    report = (build.REPO / "docs" / "validation.md").read_text(encoding="utf-8")
    assert "band" in report.lower() and "square cylinder" in report.lower()


def test_parse_manifest_reads_every_row():
    rows = build.parse_manifest((WEBSITE / "assets" / "screens" / "MANIFEST.md").read_text(encoding="utf-8"))
    assert len(rows) >= 6
    assert {"file", "page", "state", "caption"} <= set(rows[0])
    assert all(r["file"].endswith(".png") for r in rows)


def test_feature_shot_is_cropped_linked_and_left_out_when_missing(tmp_path):
    (tmp_path / "screens").mkdir()
    shutil.copy(WEBSITE / "assets" / "screens" / "MANIFEST.md", tmp_path / "screens" / "MANIFEST.md")
    assert build.feature_shot(tmp_path, "heat.png") == ""            # no placeholder frame
    (tmp_path / "screens" / "heat.png").write_bytes(b"png")
    out = build.feature_shot(tmp_path, "heat.png", display_px=370)
    assert 'href="assets/screens/heat.png"' in out and "--crop-ar: 580 / 446" in out
    assert "srcset" not in out                                         # no 800 px variant next to it
    (tmp_path / "screens" / "heat-800.webp").write_bytes(b"webp")
    assert 'srcset="assets/screens/heat-800.webp 800w' in build.feature_shot(tmp_path, "heat.png")


def test_command_block_copies_exactly():
    out = build.command_block(["a b", "c"])
    assert re.sub(r"<[^>]+>", "", out) == "a b\nc"
    assert out.count('<span class="ln">') == 2


def test_naca_outline_is_closed_and_thick_enough():
    pts = build.naca4_outline("0012", n=60)
    ys = [y for _, y in pts]
    assert max(ys) == pytest.approx(0.06, abs=2e-3) and min(ys) == pytest.approx(-0.06, abs=2e-3)
    assert pts[0] == pytest.approx((1.0, 0.0), abs=1e-9)


def test_strip_document_keeps_head_resources_but_no_wrappers():
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>T</title>\n"
            "<link rel=\"stylesheet\" href=\"a.css\">\n<script>x()</script>\n</head>\n<body>\n<main>hi</main>\n"
            "</body>\n</html>\n")
    out = build.strip_document(page)
    assert out.startswith("<title>T</title>")
    assert '<link rel="stylesheet" href="a.css">' in out and "<script>x()</script>" in out
    assert "<main>hi</main>" in out
    for tag in ("<!doctype", "<html", "<head", "<body", "</body", "</html", "<meta"):
        assert tag not in out.lower()


def test_check_links_reports_missing_files_and_anchors(tmp_path):
    (tmp_path / "a.html").write_text('<a href="b.html#yes">ok</a><a href="b.html#no">x</a>'
                                     '<img src="missing.png"><a href="https://x.org/q">ext</a>'
                                     '<source srcset="gone.webp"><a href="#self">s</a><h2 id="self">S</h2>',
                                     encoding="utf-8")
    (tmp_path / "b.html").write_text('<h2 id="yes">Y</h2>', encoding="utf-8")
    problems = build.check_links(tmp_path)
    assert problems == ["a.html: no anchor #no in b.html", "a.html: missing missing.png",
                        "a.html: missing gone.webp"]


# --------------------------------------------------------------------------- theme tokens
def _luminance(hex_: str) -> float:
    c = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _theme_tokens() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    css = (WEBSITE / "assets" / "site.css").read_text(encoding="utf-8")
    light = re.search(r":root \{(.*?)\n\}", css, flags=re.DOTALL).group(1)
    media = re.search(r':root:not\(\[data-theme="light"\]\) \{(.*?)\n  \}', css, flags=re.DOTALL).group(1)
    forced = re.search(r':root\[data-theme="dark"\] \{(.*?)\n\}', css, flags=re.DOTALL).group(1)

    def tokens(block: str) -> dict[str, str]:
        return dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", block))

    return tokens(light), tokens(media), tokens(forced)


def test_theme_tokens_exist_in_both_dark_blocks():
    lt, mt, ft = _theme_tokens()
    assert {"--bg", "--fg", "--muted", "--faint", "--code-bg", "--field-bg", "--field-ink"} <= set(lt)
    assert set(lt) <= set(mt) and set(lt) <= set(ft)
    assert mt == ft                                  # the two dark blocks agree


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_and_control_tokens_pass_wcag_aa(theme):
    """Text at least 4.5:1 on every ground it is set on; control borders at least 3:1 (WCAG 1.4.11)."""
    lt, mt, _ = _theme_tokens()
    t = lt if theme == "light" else mt
    for text in ("--fg", "--muted", "--faint"):
        for ground in ("--bg", "--code-bg", "--surface"):
            assert _contrast(t[text], t[ground]) >= 4.5, (theme, text, ground, _contrast(t[text], t[ground]))
    assert _contrast(t["--btn-fg"], t["--btn-bg"]) >= 4.5
    assert _contrast(t["--line-strong"], t["--bg"]) >= 3.0
    assert _contrast(t["--focus"], t["--bg"]) >= 3.0


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_hero_field_follows_the_page_colours(theme):
    """The hero's flow mixes between --field-bg and --field-ink: the ground is the page's own, so the
    canvas fading in never changes the colour behind the text, and the ink is the page's ink side."""
    lt, mt, _ = _theme_tokens()
    t = lt if theme == "light" else mt
    assert t["--field-bg"] == t["--bg"]
    assert _contrast(t["--field-ink"], t["--field-bg"]) >= 7.0


# --------------------------------------------------------------------------- full build
@pytest.fixture(scope="module")
def built_site(tmp_path_factory):
    """A build that knows no public address: no environment and, for this build only, no
    CITATION.cff addresses either, so the not-yet-public wording stays tested."""
    out = tmp_path_factory.mktemp("site")
    shutil.copytree(WEBSITE / "assets", out / "assets")
    real = build.citation_field
    build.citation_field = lambda repo, key: None
    try:
        builder = build.SiteBuilder(out=out, env=NO_REPO_ENV)
        problems = builder.build()
    finally:
        build.citation_field = real
    return out, problems, builder


@pytest.fixture(scope="module")
def built_with_urls(tmp_path_factory):
    out = tmp_path_factory.mktemp("site_urls")
    shutil.copytree(WEBSITE / "assets", out / "assets")
    builder = build.SiteBuilder(out=out, repo_url="https://github.com/example/aerolume",
                                site_url="https://example.github.io/aerolume/", env=NO_REPO_ENV)
    return out, builder.build(), builder


def test_full_build_has_no_broken_links(built_site):
    _, problems, _ = built_site
    assert problems == []


def test_index_is_filled_in(built_site):
    out, _, _ = built_site
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "{{" not in page and "}}" not in page
    assert "<title>Aerolume</title>" in page
    assert "Screenshot pending" not in page
    assert page.count('<tr class="group">') >= 5     # the ledger: one group per solver area
    assert '<span class="gauge"' in page
    # no external requests at all on the home page: fonts are self-hosted, MathJax is docs-only
    assert not re.findall(r'(?:href|src)="(https?://[^"]+)"', page.split("<main", 1)[0])
    assert 'rel="preload" href="assets/fonts/atkinson-hyperlegible-next-latin-wght-normal.woff2"' in page
    for p in (out / "docs").glob("*.html"):
        for url in re.findall(r'<(?:script|link)[^>]+(?:href|src)="(https?://[^"]+)"', p.read_text(encoding="utf-8")):
            assert url.startswith("https://cdn.jsdelivr.net/npm/mathjax@3."), (p.name, url)


def test_without_a_repository_url_nothing_points_at_a_guessed_address(built_site):
    out, _, builder = built_site
    assert any("no repository URL" in w for w in builder.warnings)
    for page in [out / "index.html", *(out / "docs").glob("*.html")]:
        text = page.read_text(encoding="utf-8")
        assert "github.com/aerolume" not in text, page.name          # the old guessed address
        assert "/blob/main/" not in text and "/issues" not in text, page.name
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "repository URL" not in index and "git clone" not in index    # never a placeholder address
    assert "The source repository is not public yet" in index
    assert 'og:image' not in index                 # relative og:image is useless to crawlers
    assert (out / "LICENSE.txt").exists() and 'href="LICENSE.txt"' in index


def test_with_urls_links_and_social_tags_are_absolute(built_with_urls):
    out, problems, _ = built_with_urls
    assert problems == []
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "git clone https://github.com/example/aerolume.git" in index and "cd aerolume" in index
    assert '<meta property="og:image" content="https://example.github.io/aerolume/assets/social-card.jpg">' in index
    assert 'href="https://github.com/example/aerolume/issues"' in index
    doc = (out / "docs" / "theory-ns2d.html").read_text(encoding="utf-8")
    assert 'content="https://example.github.io/aerolume/docs/theory-ns2d.html"' in doc


def test_missing_documents_are_reported_and_the_index_only_promises_what_exists(built_site):
    out, _, builder = built_site
    index = (out / "docs" / "index.html").read_text(encoding="utf-8")
    missing = [s for s in build.DOCS if not (build.REPO / s.source).exists()
               or not (build.REPO / s.source).read_text(encoding="utf-8").strip()]
    for spec in missing:
        assert any(spec.source in w for w in builder.warnings)
        assert f'href="{spec.slug}.html"' not in index
    if not any(s.group == "Guide" for s in build.DOCS if s not in missing):
        assert "how to use the program" not in index and "user guide" not in index.lower()


def test_docs_pages_have_titles_navigation_and_pager(built_site):
    out, _, _ = built_site
    pages = sorted((out / "docs").glob("*.html"))
    assert any(p.name == "theory-ns2d.html" for p in pages)
    for p in pages:
        text = p.read_text(encoding="utf-8")
        title = re.search(r"<title>(.*?)</title>", text).group(1)
        assert title.startswith("Aerolume docs"), p.name
        assert 'class="docs-nav"' in text and 'class="docs-page"' in text
        if p.name != "index.html":
            assert 'class="pager"' in text and 'aria-current="page"' in text


def test_docs_have_no_raw_file_names_or_latex(built_site):
    out, _, _ = built_site
    for p in (out / "docs").glob("*.html"):
        text = p.read_text(encoding="utf-8")
        prose = re.sub(r"<pre.*?</pre>|<code>.*?</code>|<script.*?</script>|<head>.*?</head>", "", text,
                       flags=re.DOTALL)
        visible = re.sub(r"<[^>]+>", "", prose)
        assert not re.search(r"\b(?:theory_ns2d|theory_airfoil|validation_airfoil)\.md\b", visible), p.name
        assert not re.search(r"[\^_]\{", visible), p.name


def test_doc_figures_use_theme_panels_when_available(built_site):
    out, _, _ = built_site
    if not (WEBSITE / "assets" / "docfig" / "docfig.json").exists():
        pytest.skip("web figure panels not generated")
    page = (out / "docs" / "validation-airfoil.html").read_text(encoding="utf-8")
    assert page.count('class="doc-fig') == 3
    assert 'class="only-light"' in page and 'class="only-dark"' in page
    assert 'href="img/airfoil_0012.png"' in page   # each panel links to the full-resolution PNG


def test_mathjax_only_on_pages_with_math(built_site):
    out, _, _ = built_site
    for p in (out / "docs").glob("*.html"):
        text = p.read_text(encoding="utf-8")
        prose = re.sub(r"<pre.*?</pre>|<script.*?</script>", "", text, flags=re.DOTALL)
        has_math = "\\(" in prose or 'class="math-block"' in prose
        assert ("mathjax@3" in text) == has_math, p.name


def test_artifact_copy(built_site, tmp_path):
    out, _, builder = built_site
    art = builder.write_artifact(tmp_path / "artifact")
    index = (art / "index.html").read_text(encoding="utf-8")
    assert index.lstrip().startswith("<title>Aerolume</title>")
    for tag in ("!doctype", "html", "head", "body", "meta"):
        assert not re.search(rf"</?{tag}[\s>]", index, flags=re.IGNORECASE), tag
    # a single page: the site's own stylesheet and both scripts are inlined; no external links at all
    assert '<link rel="stylesheet" href="assets/' not in index and '<script src="assets/' not in index
    assert "<style>" in index and "--field-ink" in index
    assert "data-hero-field" in index and "webgl2" in index and "data-theme-toggle" in index
    assert not re.findall(r'<link[^>]+href="(https?://[^"]+)"', index)
    assert 'url("assets/fonts/atkinson-hyperlegible-next-latin-wght-normal.woff2")' in index
    assert (art / "assets" / "fonts" / "atkinson-hyperlegible-next-latin-wght-normal.woff2").exists()
    assert (art / "assets" / "site.css").exists() and (art / "docs" / "theory-ns2d.html").exists()
    other = (art / "docs" / "theory-ns2d.html").read_text(encoding="utf-8")
    assert other.startswith("<!doctype html>") and "<meta charset" in other
    assert not (art / "assets" / "screens" / "MANIFEST.md").exists()
    assert not (art / "assets" / "docfig" / "docfig.json").exists()
    assert build.check_links(art) == []
    assert not (out / "_artifact").exists()        # never written inside the site folder


def test_artifact_inside_website_is_refused(tmp_path):
    assert build.main(["--artifact", str(WEBSITE / "_artifact_test")]) == 2
    assert not (WEBSITE / "_artifact_test").exists()


# --------------------------------------------------------------------------- home page (v0.3 design)
def _visible(page: str) -> str:
    """The page's visible text: no head, scripts, styles or tags."""
    page = re.sub(r"<head>.*?</head>|<script.*?</script>|<style.*?</style>", "", page, flags=re.DOTALL)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))


def test_hero_markup_follows_the_hero_contract(built_site):
    """hero.js finds its parts by these hooks; the canvas is decorative and the pause control is a
    real, named button (WCAG 2.2.2) that starts hidden until the script can honour it."""
    out, _, _ = built_site
    page = (out / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<section class="hero" data-hero aria-labelledby="hero-title">', page)
    assert re.search(r'<canvas class="field" data-hero-field aria-hidden="true"></canvas>', page)
    assert "data-hero-copy" in page and 'data-hero-quiet' in page
    assert re.search(r'<button class="icon-btn hero-pause" type="button" data-hero-pause hidden '
                     r'aria-label="Pause the background animation">', page)
    assert '<script src="assets/hero/hero.js" defer></script>' in page
    assert page.index('<script src="assets/hero/hero.js"') > page.index("</main>")   # after every hook
    assert page.count("<h1") == 1


def test_no_caption_about_the_background(built_site):
    """The background speaks for itself: the page carries no readout line about it."""
    out, _, _ = built_site
    text = _visible((out / "index.html").read_text(encoding="utf-8"))
    assert "Background:" not in text
    assert "schlieren" not in text.lower() and "tracers lit" not in text


def test_positioning_is_on_the_home_page(built_site):
    out, _, _ = built_site
    text = _visible((out / "index.html").read_text(encoding="utf-8"))
    assert "built for independent student researchers who need numbers they can trust" in text
    assert "No black-box results." in text
    assert "comes from a script in validation/" in text
    # the revised band shows on two ledger rows (one band, two grid levels): the lede says so
    assert "The one band revised after its result was known (the square cylinder&rsquo;s drag, checked at two grid levels)" in text
    assert "fixed before the run" not in text                         # one band was revised afterwards
    assert "laptop" not in text                                       # no laptop CPU was measured
    # the download section stays honest about what exists
    assert "There is no public download yet." in text
    assert "Before you quote a number" in text                        # the limits block


def test_trust_ledger_quotes_the_record(built_site):
    out, _, _ = built_site
    page = (out / "index.html").read_text(encoding="utf-8")
    ledger = re.search(r'<table class="ledger">(.*?)</table>', page, flags=re.DOTALL).group(1)
    rec = build.ValidationRecord(build.REPO)
    picks = [c for _, cs in rec.ledger() for c in cs]
    assert ledger.count('<th scope="row">') == len(picks)
    for c in picks:
        assert f'data-label="Aerolume">{build.fmt_value(c.value)}</td>' in ledger, c.quantity
    assert ledger.count('<tr class="fail">') == sum(not c.passed for c in picks)
    assert "pending" not in ledger


def test_validation_tally_names_failures(tmp_path):
    rec = _record(tmp_path, [("sq", "Square cylinder", "flow2d", [_chk("Cd", 2.0, 1.0, 1.5, False), _chk("St", 1.2, 1.0, 1.5, True)]),
                             ("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])
    out = build.validation_tally(rec)
    # the pass count is static text (a count-in showed wrong numbers on the way); the bar is the motion
    assert "<strong>2 of 3 checks passed</strong>" in out and "data-count" not in out
    assert '<span class="tally-bar" aria-hidden="true" style="--frac:0.6667">' in out
    assert "Square cylinder" in out and "stays in the report as failing" in out and "(30 October 2026)" in out
    ok = _record(tmp_path / "b", [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])
    assert build.validation_tally(ok) == ('<p class="tally"><span class="tally-bar" aria-hidden="true" style="--frac:1.0000">'
                                          '<span class="tally-fill"></span></span>In the last full run (30 October 2026), '
                                          '<strong>1 of 1 checks passed</strong>.</p>')


CHANGELOG = """# Changelog

## [Unreleased]

### Planned for 0.2
- more

## [0.1.0] - 2026-10-31

First public release.
"""


def test_released_versions_and_the_release_note(tmp_path):
    """Only dated headings are releases; the oldest one is "the first release"."""
    assert build.released_versions(tmp_path) == {}
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    rel = build.released_versions(tmp_path)
    assert rel == {"0.1.0": "2026-10-31"}
    assert build.release_note("0.1.0", rel) == ", the first release"
    two = {"0.2.0": "2027-03-01", "0.1.0": "2026-10-31"}
    assert build.release_note("0.2.0", two) == "" and build.release_note("0.1.0", {}) == ""


def test_run_facts_never_names_a_version_that_was_not_released(tmp_path):
    """The validation run of the first release was recorded under the tree's internal label
    (0.3.0); the page must call that a development build, not quote a release that never was."""
    rec = _record(tmp_path, [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])   # meta aerolume 9.9
    rel = {"0.1.0": "2026-10-31"}
    dev = build.run_facts(rec, "0.1.0", rel)
    assert dev == "Run of 30 October 2026 on an G, with a development build made before the 0.1.0 release."
    assert "9.9" not in dev
    assert build.run_facts(rec, "9.9", {}) == "Run of 30 October 2026, Aerolume 9.9 on an G."
    assert build.run_facts(rec, "1.0.0", {"9.9": "2026-01-01"}) == "Run of 30 October 2026, Aerolume 9.9 on an G."
    late = _record(tmp_path / "late", [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])], date="2026-11-02 10:00")
    assert build.run_facts(late, "0.1.0", rel) == "Run of 2 November 2026 on an G, with a development build of Aerolume."
    assert build.run_facts(build.ValidationRecord(tmp_path / "none"), "0.1.0", rel) == ""


def test_version_problems_name_what_still_has_to_be_rerun(tmp_path):
    (tmp_path / "src" / "aerolume").mkdir(parents=True)
    (tmp_path / "src" / "aerolume" / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "aerolume"\nversion = "0.3.0"\n', encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "install.md").write_text(
        "**Measured release build: Aerolume 0.3.0, CUDA variant.** Every number ...\n", encoding="utf-8")
    assert build.measured_build_version(tmp_path) == "0.3.0"
    rec = _record(tmp_path, [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])
    b = build.SiteBuilder(out=tmp_path / "out", repo=tmp_path, env=NO_REPO_ENV)
    probs = " | ".join(b.version_problems(rec, {}))
    assert "pyproject.toml 0.3.0" in probs and "no released entry" in probs
    assert "development build (9.9)" in probs and "installer of 0.3.0" in probs
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "aerolume"\nversion = "0.1.0"\n', encoding="utf-8")
    (tmp_path / "docs" / "install.md").write_text("**Measured release build: Aerolume 0.1.0, CUDA variant.**\n",
                                                  encoding="utf-8")
    good = _record(tmp_path / "g", [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])
    good.meta["aerolume"] = "0.1.0"
    assert b.version_problems(good, {"0.1.0": "2026-10-31"}) == []


def test_home_page_shows_the_package_version_everywhere(built_site):
    """Every version on the site is the package's own, and no internal development label (the
    tree was 0.3.0 before the first public release was numbered 0.1.0) appears beside it."""
    out, _, _ = built_site
    version = build.read_version()
    page = (out / "index.html").read_text(encoding="utf-8")
    text = _visible(page)
    assert f"Version {version}" in text and f"Aerolume {version}, Apache License 2.0" in text
    assert f"Aerolume-{version}-cuda-setup.exe" in text
    rel = build.released_versions()
    others = {v for v in re.findall(r"\b\d+\.\d+\.\d+\b", text) if v != version and v not in rel}
    assert not (others & {"0.2.0", "0.3.0"}), others
    if build.release_note(version, rel):
        assert f'Version <span class="ver">{version}</span>, the first release, for Windows' in page


def test_ns3d_rate_is_read_from_the_theory_doc(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "theory_ns3d.md").write_text(
        "| configuration | steps/s | peak memory |\n|---|---:|---:|\n"
        "| 256×128×128 (4.2 M cells), default: upwind3 | **44.9–49.2** (target ≥ 40) | 0.56 GB |\n"
        "| same, eager | 42.1 | 0.49 GB |\n", encoding="utf-8")
    assert build.ns3d_rate(tmp_path) == "44.9&ndash;49.2"
    (tmp_path / "docs" / "theory_ns3d.md").write_text("no table", encoding="utf-8")
    assert build.ns3d_rate(tmp_path) is None
    assert build.ns3d_rate(build.REPO) is not None        # the real doc still has the table


def test_home_quotes_the_3d_rate_from_the_docs_not_a_stale_copy(built_site):
    out, _, _ = built_site
    page = (out / "index.html").read_text(encoding="utf-8")
    assert f"{build.ns3d_rate(build.REPO)}</b> steps/s on 4.2&nbsp;M cells" in page
    # the range was measured on a shared GPU and the overview quotes the idle figure: both, labelled
    idle = build.ns3d_idle_rate(build.REPO)
    assert idle and f"in 3D, same GPU shared with other jobs ({idle} when idle)</li>" in page


def test_ns3d_idle_rate_is_read_from_the_theory_doc(tmp_path):
    (tmp_path / "docs").mkdir()
    doc = tmp_path / "docs" / "theory_ns3d.md"
    doc.write_text("On the idle GPU of the table below the default 4.2 M-cell case runs 54.7 steps/s with\n"
                   "17.5 ms of kernels per step.\n", encoding="utf-8")
    assert build.ns3d_idle_rate(tmp_path) == "54.7"
    doc.write_text("no such sentence", encoding="utf-8")
    assert build.ns3d_idle_rate(tmp_path) is None


def test_reference_labels_keep_lower_case_surname_particles(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "validation.md").write_text(
        '<a id="ref-devahl"></a>**devahl**: de Vahl Davis (1983), Natural convection of air in a square cavity.\n'
        '<a id="ref-st"></a>**st**: Schäfer & Turek (1996), Benchmark computations.\n', encoding="utf-8")
    labels = build.reference_labels(tmp_path)
    assert labels["devahl"] == "de Vahl Davis (1983)" and labels["st"] == "Schäfer & Turek (1996)"


def test_revised_note_names_the_case_and_its_grid_levels(tmp_path):
    rec = _record(tmp_path, [("square", "Square cylinder, Re 100", "flow2d",
                              [_chk("Square cylinder Re 100, mean Cd (32 cells/D)", 1.5, 1.4, 1.57, True),
                               _chk("Square cylinder Re 100, mean Cd (48 cells/D)", 1.5, 1.4, 1.57, True)])])
    assert build.revised_note(rec) == ("The one band revised after its result was known (the square cylinder&rsquo;s "
                                       "drag, checked at two grid levels) is marked below.")
    none = _record(tmp_path / "n", [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])
    assert build.revised_note(none) == "No band was revised after its result was known."


def test_cavity_plot_is_drawn_from_the_record(built_site):
    """The one validation curve on the page: every number in it comes from the record, the
    numbers are in the caption as text, and the line can be drawn on scroll (pathLength 1)."""
    out, _, _ = built_site
    page = (out / "index.html").read_text(encoding="utf-8")
    fig = re.search(r'<figure class="vplot" data-reveal="plot">.*?</figure>', page, flags=re.DOTALL).group(0)
    pts = build.cavity_points(build.ValidationRecord(build.REPO))
    assert [e for e, _, _ in pts] == [3, 4, 5, 6]
    cap = re.search(r"<figcaption>(.*?)</figcaption>", fig, flags=re.DOTALL).group(1)
    for _, _, c in pts:
        assert build.fmt_value(c.value) in cap and build.fmt_value(c.ref) in cap
    assert "de Vahl Davis (1983)" in cap and "All four pass." in cap
    assert fig.count('class="ref"') == 4 and fig.count('class="val"') == 8 and fig.count('class="bandbar"') == 4
    assert fig.count('class="curve" pathLength="1"') == 2
    assert 'role="img" aria-labelledby="vplot-t vplot-d"' in fig and '<desc id="vplot-d">' in fig


def test_cavity_plot_is_left_out_without_the_case(tmp_path):
    rec = _record(tmp_path, [("d", "Disc", "flow2d", [_chk("Cd", 1.2, 1.0, 1.5, True)])])
    assert build.cavity_plot(rec, {}, lambda t: "#") == ""


def test_topbar_variants():
    site = build.SiteInfo("9.9.9", build.build_date())
    home = build.topbar("", site, home=True)
    docs = build.topbar("../", site, "docs")
    assert "wordmark" not in home and "data-hero-pause" in home and "data-hero-quiet" in home
    assert 'href="#validation"' in home                          # same-page anchors on the home page
    assert 'class="wordmark" href="../index.html"' in docs and "9.9.9" in docs
    assert 'href="../docs/index.html" aria-current="page"' in docs and "data-hero-pause" not in docs
    assert 'href="../index.html#download"' in docs


def test_inline_local_assets_inlines_only_the_sites_own_files(tmp_path):
    (tmp_path / "assets" / "hero").mkdir(parents=True)
    (tmp_path / "assets" / "site.css").write_text("a{color:red}", encoding="utf-8")
    (tmp_path / "assets" / "fonts.css").write_text(
        '@font-face{src:url("fonts/a.woff2"),url(b.woff2),url(data:x),url(https://e.org/c.woff2),local("Arial")}',
        encoding="utf-8")
    (tmp_path / "assets" / "hero" / "hero.js").write_text("var s='</script>';", encoding="utf-8")
    page = ('<link rel="stylesheet" href="https://fonts.googleapis.com/x">\n<link rel="stylesheet" href="assets/site.css">\n'
            '<script src="assets/hero/hero.js" defer></script>')
    out = build.inline_local_assets(page, tmp_path)
    assert 'href="https://fonts.googleapis.com/x"' in out
    assert "<style>\na{color:red}\n</style>" in out
    assert r"var s='<\/script>';" in out and out.count("</script>") == 1
    css = build.inline_local_assets('<link rel="stylesheet" href="assets/fonts.css">', tmp_path)
    assert 'url("assets/fonts/a.woff2")' in css and "url(assets/b.woff2)" in css
    assert "url(data:x)" in css and "url(https://e.org/c.woff2)" in css and 'local("Arial")' in css


def test_hero_script_keeps_its_public_page_guarantees():
    """Static checks on assets/hero/hero.js for the behaviour the brief requires; the capture
    scripts measure the rest (frame time, contrast from pixels)."""
    js = (WEBSITE / "assets" / "hero" / "hero.js").read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in js                     # still frame
    assert "visibilitychange" in js and "IntersectionObserver" in js  # pauses when hidden / off-screen
    assert "Math.min(window.devicePixelRatio || 1, 2" in js           # DPR cap
    assert "2.6e6" in js                                               # canvas pixel budget
    assert "webglcontextlost" in js and "startCPU" in js               # never a blank hero
    assert "--field-bg" in js and "--field-ink" in js                  # follows the theme tokens
    assert js.count("minFrameMs:") == 2                                # both engines paced (60 / 30 fps)
    assert "minFrameMs: 1000 / 60" in js and "minFrameMs: 1000 / 30" in js
    assert "KHR_parallel_shader_compile" in js                         # compile off the main thread
    assert 'window.addEventListener("load", later' in js             # starts after the page has loaded
    # the pause button keeps one accessible name; aria-pressed alone carries the state (WCAG 4.1.2)
    assert 'setAttribute("aria-label"' not in js and 'setAttribute("aria-pressed"' in js
    assert "for (const r of [calm, calm2])" in js                      # fallback's hard ink bound behind text
    assert not re.search(r"\bimport\s*\(|^\s*import\s|https?://", js, flags=re.MULTILINE)   # self-contained
    # storage can be blocked (private windows, previews): every access sits inside a try on its line
    for line in js.splitlines():
        if "localStorage." in line:
            assert "try {" in line, line


# --------------------------------------------------------------------------- scroll motion, fonts
def test_scroll_reveals_are_opt_in_and_respect_reduced_motion(built_site):
    """Blocks are hidden only once site.js has put "motion" on the root element, which it does only with
    IntersectionObserver and without reduced motion; CSS hides nothing on its own."""
    out, _, _ = built_site
    page = (out / "index.html").read_text(encoding="utf-8")
    # few, large units (website round 4): section heads with their text, the ledger as one table,
    # and every picture uncovered in the flow direction; prose never animates on its own
    assert 8 <= page.count("data-reveal") <= 16
    assert page.count('data-reveal="wipe"') == 4 and "data-stagger" not in page
    assert '<div class="table-scroll ledger-wrap" data-reveal>' in page
    assert not re.search(r'class="(?:feature-text|prose|codeblock|more)"[^>]*data-reveal', page)
    hero = re.search(r'<section class="hero".*?</section>', page, flags=re.DOTALL).group(0)
    assert "data-reveal" not in hero                                  # the first screen never waits
    css = (WEBSITE / "assets" / "site.css").read_text(encoding="utf-8")
    hidden = re.findall(r"([^{}]+)\{[^{}]*opacity:\s*0[;\s]", css)
    assert hidden and all(".motion" in sel or "::after" in sel for sel in hidden if "reveal" in sel or "gauge" in sel)
    for sel in re.findall(r"([^{}]*\[data-reveal[^\]]*\][^{}]*)\{", css):
        assert ".motion" in sel, sel
    js = (WEBSITE / "assets" / "site.js").read_text(encoding="utf-8")
    assert 'root.classList.add("motion")' in js and "prefers-reduced-motion: reduce" in js
    assert "IntersectionObserver" in js and "beforeprint" in js
    motion = css.split("scroll motion", 1)[1].split("@media (prefers-reduced-motion: reduce)", 1)[0]
    # only transform and opacity move (plus the divider's paint-only stroke-dashoffset), and every
    # sequence settles within about 600 ms: duration + fixed delay, plus at most 180 ms of stagger
    for t in re.findall(r"transition:([^;]+);", motion):
        props = re.findall(r"(?:^|,)\s*([a-z-]+)\s+\d", t)
        assert set(props) <= {"opacity", "transform", "stroke-dashoffset"}, t
        for part in t.split(","):
            times = [float(x) for x in re.findall(r"(?<![\w.])(\d*\.?\d+)s\b", part)]
            fixed = sum(float(x) for x in re.findall(r"\+\s*(\d+)ms", part)) / 1000
            if times:
                assert times[0] + fixed <= 0.6, part
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "Math.min(h * 60, 180)" in js                              # stagger cap
    assert "countIn" not in js and "data-count" not in page           # no number is ever animated


def test_hero_copy_never_fades_or_moves_while_it_is_read():
    """Website round 4: a fading, lifting copy over a lagging field dropped the lede to 2.7:1 on
    phones. The copy scrolls with the page; only the field lags, and hero.js moves the calm zone
    with the copy (measured against the canvas's own transformed box, re-measured on scroll)."""
    css = (WEBSITE / "assets" / "site.css").read_text(encoding="utf-8")
    assert "hero-copy-out" not in css and not re.search(r"\.hero-copy[^{]*\{[^}]*animation", css)
    js = (WEBSITE / "assets" / "hero" / "hero.js").read_text(encoding="utf-8")
    assert "const hr = canvas.getBoundingClientRect();" in js
    assert "E.frame(dt); track(); E.present();" in js and 'addEventListener("scroll"' in js


def test_streakline_divider_is_a_computed_flow():
    """The divider's lines are streamlines of inviscid flow past a Joukowski airfoil: they start at
    the left edge, run past the right edge, never cross the airfoil, and are drawn as pathLength 1
    lines so the reveal's dashoffset covers each whole line."""
    svg = build.streakline_svg()
    paths = re.findall(r'<path d="M([^"]+)" pathLength="1" style="--k:(\d+)" opacity="([\d.]+)"/>', svg)
    assert len(paths) == 10 and sorted(int(k) for _, k, _ in paths) == list(range(10))
    foil = re.search(r'<path class="foil" d="M([^"]+)Z"/>', svg).group(1)
    fx = [float(v) for v in re.findall(r"(-?\d+\.\d) -?\d+\.\d", foil)]
    assert 900 < min(fx) < 950 and 1020 < max(fx) < 1070              # chord ~120 px at 82 % of 1200
    for d, _, op in paths:
        pts = [tuple(map(float, xy.split())) for xy in d.split(" L")]
        assert pts[0][0] < 0 and pts[-1][0] > 1200 and 0.3 < float(op) <= 0.85
        assert all(-20 < y < 124 for _, y in pts)
        assert all(b[0] > a[0] for a, b in itertools.pairwise(pts))       # flows left to right
        assert len(pts) < 120                                          # simplified, not a point per step
    assert "nan" not in svg.lower() and 'preserveAspectRatio="xMaxYMid slice"' in svg


def test_png_size_reads_the_ihdr(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + (640).to_bytes(4, "big")
                    + (480).to_bytes(4, "big") + b"\x08\x02\x00\x00\x00")
    assert build.png_size(png) == (640, 480)
    (tmp_path / "b.png").write_bytes(b"GIF89a" + bytes(30))
    assert build.png_size(tmp_path / "b.png") is None and build.png_size(tmp_path / "missing.png") is None


def test_docs_images_reserve_their_box(built_site):
    """Every image on a docs page carries width and height, so it cannot push the page down when
    it loads (it grew from 52 to 412 px on docs/overview before, CLS 0.094)."""
    out, _, _ = built_site
    n = 0
    for page in (out / "docs").glob("*.html"):
        for tag in re.findall(r"<img\b[^>]*>", page.read_text(encoding="utf-8")):
            assert re.search(r'\bwidth="\d+" height="\d+"', tag), (page.name, tag[:120])
            n += 1
    assert n >= 20


def test_fallback_faces_match_the_web_fonts():
    """The stand-ins are metric-matched (measured in Chrome) and name ArialMT first: Chrome on Windows
    matched local("Arial") to a 17 % wider face, which rewrapped every paragraph on swap."""
    css = (WEBSITE / "assets" / "site.css").read_text(encoding="utf-8")
    fb = re.search(r'font-family: "Atkinson Fallback";([^}]+)\}', css).group(1)
    assert fb.split("src:", 1)[1].strip().startswith('local("ArialMT")') and "size-adjust: 98.4%" in fb
    mono = re.search(r'font-family: "Atkinson Mono Fallback";([^}]+)\}', css).group(1)
    assert 'local("Consolas")' in mono and "size-adjust: 114.95%" in mono
    assert '"Atkinson Mono Fallback", "Atkinson Mono Fallback 2", ui-monospace' in css
    upright = re.search(r'font-family: "Atkinson Hyperlegible Next";\s*font-style: normal;([^}]+)\}', css).group(1)
    assert "font-display: optional" in upright                       # preloaded: no swap, no rewrap


def test_fonts_are_self_hosted_with_their_licences():
    fonts = WEBSITE / "assets" / "fonts"
    css = (WEBSITE / "assets" / "site.css").read_text(encoding="utf-8")
    for ref in re.findall(r'url\("(fonts/[^"]+)"\)', css):
        assert (WEBSITE / "assets" / ref).exists(), ref
    assert css.count("@font-face") >= 4                               # Next, Next italic, Mono, the fallback
    for fam in ("next", "mono"):
        lic = (fonts / f"OFL-atkinson-hyperlegible-{fam}.txt").read_text(encoding="utf-8")
        assert "SIL Open Font License" in lic
    assert (WEBSITE / "assets" / build.FONT_PRELOAD.removeprefix("assets/")).exists()
