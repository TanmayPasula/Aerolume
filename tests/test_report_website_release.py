"""The published face of a release: the committed website and the user guide's screenshots.

A pre-release critique found the committed ``website/index.html`` still announcing an older version (built on
2026-10-05) although its template was current, the site's screenshot folder missing every 0.3
screen its manifest lists, and seven user-guide screenshots showing the 0.1 program. Nothing failed,
because the Pages workflow rebuilds the site on deploy and no test compared the committed copy with
the code. These tests do, from the package side (the website's own tests run without the package):

* the committed home page and every committed docs page state ``aerolume.__version__``;
* the committed home page is what ``website/build.py`` builds from the repository today (apart from
  the build date), so a template, docs or version change without a rebuild fails here;
* every screenshot in ``website/assets/screens/MANIFEST.md`` exists, and each one that the user guide
  also uses (``docs/img/ui_<name>.png``) is the same capture, so the two cannot drift apart again
  (``website/tools/make_screens.py`` writes both).

When one fails: ``python website/tools/make_screens.py --work <scratch>`` (screens), then
``python website/build.py --strict`` (pages).
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pytest

import aerolume

REPO = Path(__file__).resolve().parents[1]
WEBSITE = REPO / "website"
SCREENS = WEBSITE / "assets" / "screens"
DOC_IMG = REPO / "docs" / "img"
REBUILD = "rebuild the site: python website/build.py --strict"

pytestmark = pytest.mark.skipif(not (WEBSITE / "build.py").is_file(), reason="no website/ in this checkout")


def _build_module():
    """website/build.py, loaded by path (it is a script, not part of the installed package)."""
    spec = importlib.util.spec_from_file_location("aerolume_website_build", WEBSITE / "build.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _manifest_files() -> list[str]:
    rows = _build_module().parse_manifest((SCREENS / "MANIFEST.md").read_text(encoding="utf-8"))
    return [r["file"] for r in rows]


# ---------------------------------------------------------------------------------- version
def test_home_page_states_the_package_version():
    page = (WEBSITE / "index.html").read_text(encoding="utf-8")
    shown = re.findall(r'class="ver">([^<]+)<', page)
    assert shown == [aerolume.__version__], f"header shows {shown}; {REBUILD}"
    footer = re.search(r"<p>Aerolume ([^,<]+), Apache License", page)
    assert footer and footer.group(1) == aerolume.__version__, REBUILD


def test_every_docs_page_states_the_package_version():
    pages = sorted((WEBSITE / "docs").glob("*.html"))
    assert pages, "no docs pages: " + REBUILD
    stale = {}
    for p in pages:
        shown = set(re.findall(r'class="ver">([^<]+)<', p.read_text(encoding="utf-8")))
        if shown != {aerolume.__version__}:
            stale[p.name] = sorted(shown)
    assert stale == {}, REBUILD


# ---------------------------------------------------------------------------------- staleness
_BUILD_DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")


def _normalise(page: str) -> str:
    """The page without its build date (the footer's "Built ... on <date>"), the only part of a
    rebuild that changes by itself. Figure dates live in the template's data, so they stay."""
    return re.sub(r"(by <code>website/build\.py</code> on )" + _BUILD_DATE.pattern, r"\1<date>", page)


def test_committed_home_page_matches_a_fresh_build(tmp_path):
    """The committed index.html must be what build.py writes from today's repository: the same
    builder, the same assets, no repository address (none is configured in pyproject.toml, so a local
    build has none either)."""
    build = _build_module()
    out = tmp_path / "site"
    out.mkdir()
    shutil.copytree(WEBSITE / "assets", out / "assets")
    build.SiteBuilder(out=out, env={}).build()
    fresh = _normalise((out / "index.html").read_text(encoding="utf-8"))
    committed = _normalise((WEBSITE / "index.html").read_text(encoding="utf-8"))
    if fresh != committed:
        a, b = committed.splitlines(), fresh.splitlines()
        first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        pytest.fail(f"website/index.html is stale from line {first + 1} "
                    f"(committed {a[first][:120] if first < len(a) else '<end>'!r}, "
                    f"fresh {b[first][:120] if first < len(b) else '<end>'!r}); {REBUILD}")


# ---------------------------------------------------------------------------------- screenshots
def test_every_manifest_screenshot_exists():
    files = _manifest_files()
    assert len(files) >= 6, "MANIFEST.md lists no screens: the table format changed"
    missing = [f for f in files if not (SCREENS / f).is_file()]
    assert missing == [], "retake them: python website/tools/make_screens.py --work <scratch>"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_user_guide_screenshots_are_the_site_captures():
    """make_screens.py saves each capture to both places; a user-guide image that differs from the
    site's copy was left behind by a partial retake (how seven 0.1 images survived into 0.3)."""
    differ, compared = [], 0
    for f in _manifest_files():
        site, doc = SCREENS / f, DOC_IMG / f"ui_{f}"
        if site.is_file() and doc.is_file():
            compared += 1
            if _digest(site) != _digest(doc):
                differ.append(f)
    assert compared >= 6, "no screenshot is shared between the site and docs/img"
    assert differ == [], "retake them: python website/tools/make_screens.py --work <scratch>"


def test_figure_title_block_does_not_pair_the_version_with_an_old_date():
    """The opening figure's "Drawn" cell names the release that drew it when that is an older one."""
    import datetime as dt
    build = _build_module()
    today = dt.date(2026, 10, 31)
    assert build.figure_drawn({"date": "2026-10-31", "aerolume": "0.3.0"}, "0.3.0", today) == "2026-10-31"
    assert build.figure_drawn({"date": "2026-10-05", "aerolume": "0.1.0"}, "0.3.0", today) == "2026-10-05, with 0.1.0"
    assert build.figure_drawn({"date": "2026-10-05"}, "0.3.0", today) == "2026-10-05"
    assert build.figure_drawn({}, "0.3.0", today) == "2026-10-31"
