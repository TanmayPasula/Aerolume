"""The research documentation stays consistent with the code: CITATION.cff, docs/methods.md,
validation/README.md and the generated section of docs/validation.md.

These are cheap text checks, so a version bump, a renamed file or a hand edit inside the
generated section is caught by the normal test run instead of by a reader.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

import aerolume

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ra():
    spec = importlib.util.spec_from_file_location("aerolume_run_all_docs_test", ROOT / "validation" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cff() -> dict[str, str]:
    """Top-level scalar keys of CITATION.cff (a YAML subset: ``key: value`` lines at column 0)."""
    out = {}
    for line in (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-z-]+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def test_citation_file_matches_the_package():
    cff = _cff()
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software" and cff["license"] == "Apache-2.0"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert cff["version"] == aerolume.__version__ == project["version"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", cff["date-released"])
    for key in ("title", "message", "authors", "abstract", "references"):
        assert key in cff, key
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "\t" not in text                                     # YAML forbids tab indentation
    # the public addresses (the website build reads them from here)
    assert cff["repository-code"] == "https://github.com/TanmayPasula/Aerolume"
    assert cff["url"].startswith("https://")


def test_methods_document_links_and_references():
    text = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    for link in re.findall(r"\]\(([^)#]+\.md)", text):
        assert (ROOT / "docs" / link).resolve().is_file(), link
    refs = text.split("## References", 1)[1]
    dois = re.findall(r"doi:(\S+)", refs)
    assert len(dois) >= 25 and len(dois) == len(set(dois))
    assert all(d.startswith("10.") for d in dois)
    # every author-year citation in the body is in the reference list
    body = text.split("## References", 1)[0]
    for name in ("Angot", "Shu", "Celik", "Roache", "Eça", "White", "Flegal", "Sokal", "Garrett", "van Leer",
                 "Morinishi", "Makhoul", "Drela", "Fidkowski", "Phillips", "Katz", "Nicoud", "Vreman",
                 "Smagorinsky", "Harlow", "Gray", "Oberkampf", "Rankine", "Froude"):
        assert name in body and name in refs, name
    assert "Lattice-Boltzmann" in text                          # the deferred features are stated


def test_validation_readme_matches_the_runner(ra):
    text = (ROOT / "validation" / "README.md").read_text(encoding="utf-8")
    for name in re.findall(r"^\| `([a-z0-9_]+\.py)`", text, flags=re.MULTILINE):
        assert (ROOT / "validation" / name).is_file(), name
    for flag in ("--quick", "--all", "--stretch", "--list", "--cases", "--area", "--write-docs", "--from-json",
                 "--strict", "--device"):
        assert flag in text, flag
    assert ra.BEGIN in text and ra.END in text


def test_generated_section_is_intact(ra):
    """docs/validation.md holds exactly one generated section, starting with the runner's title,
    and the figures it names exist."""
    text = (ROOT / "docs" / "validation.md").read_text(encoding="utf-8")
    assert text.count(ra.BEGIN) == 1 and text.count(ra.END) == 1
    block = text.split(ra.BEGIN, 1)[1].split(ra.END, 1)[0]
    assert block.lstrip().startswith(ra.SECTION_TITLE)
    for img in re.findall(r"\]\((img/[^)]+)\)", block):
        assert (ROOT / "docs" / img).is_file(), img
    # the hand-written limits come after the generated section
    assert text.index("What is not validated") > text.index(ra.END)


def test_the_release_date_is_the_same_everywhere():
    """CHANGELOG, CITATION.cff and the README give one release date, and the validation run the release
    quotes was made on or before it (the release gate of 1 November 2026 found the release dated
    31 October while quoting the run of 1 November)."""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(rf"^## \[{re.escape(aerolume.__version__)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", changelog, re.MULTILINE)
    assert m, "the changelog has no dated heading for the package version"
    released = datetime.date.fromisoformat(m.group(1))
    assert _cff()["date-released"] == m.group(1)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"first public release ({released.day} {released:%B} {released.year})" in readme
    record = json.loads((ROOT / "validation" / "records" / "validation-record.json").read_text(encoding="utf-8"))
    assert str(record["meta"]["date"])[:10] <= m.group(1)


#: what a local folder looks like in a committed file: a drive letter, a home folder, a scratch area
LOCAL_PATH = re.compile(r"\b[A-Za-z]:[\\/]|/home/|/Users/|\\Users\\|aerolume-scratch")


def test_committed_records_name_no_local_paths():
    """The records are public (the website quotes them): no user name or folder of the measuring machine."""
    records = sorted((ROOT / "validation" / "records").glob("*.json"))
    assert records
    for rec in records:
        hit = LOCAL_PATH.search(rec.read_text(encoding="utf-8"))
        assert hit is None, (rec.name, hit.group(0) if hit else "")


def test_the_run_records_paths_relative_to_the_repository_or_not_at_all(ra, tmp_path):
    outside = str(tmp_path / "valout")
    assert ra.public_argv(["--all", "--write-docs", "--out", outside], cwd=ROOT) == [
        "--all", "--write-docs", "--out", ra.OUTSIDE_REPO]
    assert ra.public_argv([f"--out={outside}"], cwd=ROOT) == [f"--out={ra.OUTSIDE_REPO}"]
    assert ra.public_argv(["--out", "validation/out", "--cases", "cyl40,tg2d", "--device", "cuda"], cwd=ROOT) == [
        "--out", "validation/out", "--cases", "cyl40,tg2d", "--device", "cuda"]
    rec = ROOT / "validation" / "records" / "validation-record.json"
    assert ra.public_argv(["--from-json", str(rec), outside], cwd=ROOT) == [
        "--from-json", "validation/records/validation-record.json", ra.OUTSIDE_REPO]
    # already public arguments stay as they are when a --from-json run re-writes the record
    assert ra.public_argv(ra.public_argv(["--out", outside], cwd=ROOT), cwd=ROOT) == ["--out", ra.OUTSIDE_REPO]
