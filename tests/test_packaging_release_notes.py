"""Release measurements come from the build being released, and say which run they are from.

The failure these guard against happened before the first release: docs/install.md and the CHANGELOG quoted a
29.3 s self-test, a 49 s smoke test, a 169 s installer compile and a 12.2 min build - all real,
but from the full build, while the installer that shipped had been recompiled twice since by
``--refresh-docs`` and recorded 14.7 s, 34.1 s, 162.7 s and 4.2 min. So: every run appends a
record to the report's history, the docs block is generated from that history with one column
per run, and the exact installer gets its own release-notes file (fast; no PyInstaller run).
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"


def _load(name: str, key: str | None = None):
    key = key or name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, PACKAGING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


buildlib = _load("buildlib")
release_notes = _load("release_notes")
sys.path.insert(0, str(PACKAGING))      # build.py imports buildlib and release_notes by name
build = _load("build", "aerolume_packaging_build")   # "build" is also a PyPI package

FULL_SHA = "2f0417fc0db2" + "0" * 52
REFRESH_SHA = "686774816f68" + "1" * 52
MIDDLE_SHA = "f5fd8655bde8" + "2" * 52


def _smoke(first: float, second: float, ok: bool = True) -> dict:
    inner = [{"name": "resources", "ok": True, "seconds": 0.002, "detail": "73 icons"},
             {"name": "torch", "ok": True, "seconds": 5.47,
              "detail": "torch 2.14.1+cu130, CUDA build 13.0; CUDA device NVIDIA GeForce RTX 4060 Ti, matmul ok"}]
    return {"ok": ok, "checks": [
        {"name": release_notes.FIRST_LAUNCH, "ok": ok, "wall_seconds": first, "report": {"checks": inner}},
        {"name": release_notes.SECOND_LAUNCH, "ok": True, "wall_seconds": second, "report": {"checks": inner}},
        {"name": "cli --version", "ok": True, "wall_seconds": 0.3}]}


def _report(sha: str, *, pyinstaller: float | None, smoke_s: float, first: float, second: float,
            iscc: float, total: float, inst_sha: str, refresh: bool = False) -> dict:
    rep = {"version": "0.3.0", "variant": "cuda", "source": {"sha256": sha, "short": sha[:12]},
           "smoke": _smoke(first, second), "smoke_seconds": smoke_s, "installer_seconds": iscc,
           "installer_compression": "lzma2/max",
           "installer": {"path": "C:/b/cuda/installer/Aerolume-0.3.0-cuda-setup.exe", "bytes": 1942806099,
                         "sha256": inst_sha},
           "installer_manifest": {"ok": True, "files": 4048},
           "dist_bytes": 3538587213, "dist_files": 4048, "total_seconds": total,
           "interpreter": {"machine": {"cpu": "Test CPU", "logical_cpus": 24, "ram_bytes": 16 * 2**30}}}
    if pyinstaller is not None:
        rep["pyinstaller_seconds"] = pyinstaller
    if refresh:
        rep["docs_refresh"] = {"changed": ["docs/install.md"], "added": [], "removed": []}
    return rep


def _history() -> dict:
    """The real 0.3.0 sequence: a full build, then a docs refresh carrying its PyInstaller time."""
    full = _report(FULL_SHA, pyinstaller=457.6, smoke_s=49.3, first=29.28, second=8.79, iscc=168.5,
                   total=729.7, inst_sha="64e9a0ad5937" + "a" * 52)
    ref = _report(REFRESH_SHA, pyinstaller=457.6, smoke_s=34.1, first=14.69, second=8.03, iscc=162.7,
                  total=251.8, inst_sha="81fed9b44efa" + "b" * 52, refresh=True)
    ref["runs"] = [release_notes.run_record(full, "full", finished="2026-10-30 23:38:40"),
                   release_notes.run_record(ref, "refresh-docs", finished="2026-10-30 23:48:20", cpu_busy=41.26)]
    ref["build_info"] = {"source_sha256": REFRESH_SHA, "docs_refreshed": [
        {"from_sha256": FULL_SHA, "at": "2026-10-30 23:39:47"},
        {"from_sha256": MIDDLE_SHA, "at": "2026-10-30 23:44:27"}]}
    return ref


# ---------------------------------------------------------------------------
# run records
# ---------------------------------------------------------------------------

def test_run_record_keeps_only_what_the_run_measured():
    rep = _history()
    full, ref = rep["runs"]
    assert full["pyinstaller_seconds"] == 457.6 and full["smoke_seconds"] == 49.3
    assert full["selftest_first_seconds"] == 29.28 and full["installer_seconds"] == 168.5
    # The refresh's report still carries the full build's PyInstaller time; the run did not measure it.
    assert ref["pyinstaller_seconds"] is None
    assert (ref["selftest_first_seconds"], ref["selftest_second_seconds"]) == (14.69, 8.03)
    assert ref["installer_sha256"].startswith("81fed9b44efa") and ref["manifest_ok"] is True
    assert ref["gpu"] == "NVIDIA GeForce RTX 4060 Ti" and ref["selftest_checks"] == 2
    assert ref["cpu_busy_percent"] == 41.26


def test_reused_smoke_result_is_not_this_runs_measurement():
    rep = _report(REFRESH_SHA, pyinstaller=None, smoke_s=34.1, first=14.69, second=8.03, iscc=1.0,
                  total=5.0, inst_sha="c" * 64)
    rep["smoke_reused"] = True
    run = release_notes.run_record(rep, "reuse-dist")
    assert run["smoke_seconds"] is None and run["selftest_first_seconds"] is None and run["smoke_ok"] is None


def test_report_without_history_yields_its_own_run_only():
    rep = _report(REFRESH_SHA, pyinstaller=457.6, smoke_s=34.1, first=14.69, second=8.03, iscc=162.7,
                  total=251.8, inst_sha="d" * 64, refresh=True)
    runs = release_notes.runs_of(rep)
    assert len(runs) == 1 and runs[0]["kind"] == "refresh-docs" and runs[0]["pyinstaller_seconds"] is None
    assert release_notes.runs_of({}) == []


def test_cpu_busy_share():
    assert release_notes.cpu_busy_percent((10, 100), (60, 200)) == 50.0
    assert release_notes.cpu_busy_percent(None, (1, 2)) is None
    assert release_notes.cpu_busy_percent((5, 100), (5, 100)) is None
    sample = release_notes.system_cpu_times()
    if sys.platform == "win32":
        assert sample is not None and 0 <= sample[0] <= sample[1]


def test_machine_info_describes_this_machine():
    m = release_notes.machine_info()
    assert set(m) == {"cpu", "logical_cpus", "ram_bytes", "os"}
    assert m["logical_cpus"] and m["logical_cpus"] >= 1
    if sys.platform == "win32":
        assert m["ram_bytes"] and m["ram_bytes"] > 2**30


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_measurement_table_names_the_run_of_every_column():
    text = release_notes.render_measurements(_history())
    header = next(line for line in text.splitlines() if line.startswith("| step |"))
    assert header == "| step | full build `2f0417fc0db2` | docs refresh `686774816f68` |"
    rows = {line.split(" | ")[0].strip("| "): line for line in text.splitlines() if line.startswith("| ")}
    assert rows["PyInstaller"].endswith("| 458 s | not run |")
    assert "| 49.3 s | 34.1 s |" in rows["smoke test (self-test twice, console program, `doctor`, single instance)"]
    assert "| 29.3 s / 8.8 s | 14.7 s / 8.0 s |" in text
    assert rows["installer compile (lzma2/max)"].endswith("| 168 s | 163 s |")
    assert rows["whole run"].endswith("| 12.2 min | 4.2 min |")
    assert "`64e9a0ad5937`" in text and "`81fed9b44efa`" in text
    assert "81fed9b44efa" + "b" * 52 in text          # the full hash of the last installer
    assert "docs refresh `686774816f68` 41 %" in text  # measured load, per run
    # The refresh in between produced an installer that was replaced; it is named, not dropped.
    assert release_notes.unmeasured_refreshes(_history(), _history()["runs"]) == [(MIDDLE_SHA, "2026-10-30 23:39:47")]
    assert "`f5fd8655bde8` (30 Oct 2026, 23:39)" in text
    assert "cannot describe the installer that contains it" in text
    assert "Aerolume-0.3.0-cuda-release-notes.md" in text


def test_summary_quotes_the_last_run_and_says_which_it_is():
    text = release_notes.render_summary(_history())
    assert "docs refresh `686774816f68`" in text and "14.7 s on the first launch" in text
    assert "full build `2f0417fc0db2` took 12.2 min" in text
    assert "3.54 GB in 4,048 files" in text and "1.94 GB" in text
    assert "29.3" not in text                     # the full build's self-test is not passed off as the last


def test_release_notes_describe_the_exact_installer():
    text = release_notes.render_release_notes(_history())
    assert "`81fed9b44efa" + "b" * 52 + "`" in text and "1,942,806,099 bytes" in text
    assert "| made by | docs refresh, finished 30 Oct 2026, 23:48 |" in text
    assert "full build of `" + FULL_SHA + "`" in text
    assert "| smoke test | 34.1 s |" in text and "| PyInstaller | not run |" in text
    assert "Get-FileHash" in text and "| torch | pass |" in text


def test_nothing_recorded_says_so():
    assert "No build of Aerolume" in release_notes.render_measurements({"version": "0.3.0"})
    assert release_notes.render_summary({}) == "- Release build: none recorded yet."


# ---------------------------------------------------------------------------
# generated blocks
# ---------------------------------------------------------------------------

def test_replace_block_keeps_markers_and_line_endings():
    doc = "intro\r\n<!-- x:begin -->\r\nold 12.2 min\r\n<!-- x:end -->\r\noutro\r\n"
    once = release_notes.replace_block(doc, "x", "new\nlines")
    assert once == "intro\r\n<!-- x:begin -->\r\nnew\r\nlines\r\n<!-- x:end -->\r\noutro\r\n"
    assert release_notes.replace_block(once, "x", "new\nlines") == once
    with pytest.raises(release_notes.MarkerError):
        release_notes.replace_block("no markers", "x", "body")
    with pytest.raises(release_notes.MarkerError):
        release_notes.replace_block(doc + doc, "x", "body")
    with pytest.raises(release_notes.MarkerError):
        release_notes.replace_block("<!-- x:end -->\n<!-- x:begin -->", "x", "body")


def test_write_blocks_only_edits_documents_with_markers(tmp_path):
    install = tmp_path / "install.md"
    install.write_text("# Install\n<!-- build-measurements:begin -->\n<!-- build-measurements:end -->\n",
                       encoding="utf-8")
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n29.3 s\n", encoding="utf-8")
    targets = {install: release_notes.MEASUREMENTS_BLOCK, changelog: release_notes.SUMMARY_BLOCK,
               tmp_path / "missing.md": release_notes.SUMMARY_BLOCK}
    res = release_notes.write_blocks(_history(), targets)
    assert res["changed"] == [str(install)]
    assert sorted(res["no_markers"]) == sorted([str(changelog), str(tmp_path / "missing.md")])
    assert changelog.read_text(encoding="utf-8") == "# Changelog\n29.3 s\n"
    assert "| step | full build `2f0417fc0db2`" in install.read_text(encoding="utf-8")
    assert release_notes.write_blocks(_history(), {install: release_notes.MEASUREMENTS_BLOCK})["unchanged"]


def test_install_md_quotes_the_build_only_through_the_generated_block():
    text = (REPO / "docs" / "install.md").read_text(encoding="utf-8")
    begin, end = "<!-- build-measurements:begin -->", "<!-- build-measurements:end -->"
    assert text.count(begin) == 1 and text.count(end) == 1
    block = text.split(begin, 1)[1].split(end, 1)[0]
    assert block.strip().startswith("**Measured release build: Aerolume")
    outside = text.replace(block, "")
    # No timing of a particular build is typed outside the block (the 0.3.0 notes did that).
    assert not re.search(r"\b\d+(\.\d+)? (s|min|minutes)\b", outside.split("## Troubleshooting")[0]), \
        re.findall(r".{40}\b\d+(?:\.\d+)? (?:s|min|minutes)\b.{0,20}", outside)


# ---------------------------------------------------------------------------
# build.py: the history grows with each run, the installer gets its notes
# ---------------------------------------------------------------------------

def _fake_dist(vd: Path, sha: str, version: str) -> dict:
    dist = vd / "dist" / buildlib.APP_NAME
    dist.mkdir(parents=True, exist_ok=True)
    (dist / buildlib.GUI_EXE).write_bytes(b"MZ")
    info = {"version": version, "variant": "cuda", "built": "2026-10-30 23:34:26", "source_sha256": sha}
    (dist / "build-info.json").write_text(json.dumps(info), encoding="utf-8")
    return info


def test_reuse_dist_appends_to_the_history_and_writes_release_notes(tmp_path, monkeypatch):
    version = buildlib.read_version()
    fp = buildlib.source_fingerprint()
    vd = tmp_path / "builds" / "cuda"
    info = _fake_dist(vd, fp["sha256"], version)
    earlier = _report(fp["sha256"], pyinstaller=400.0, smoke_s=40.0, first=20.0, second=8.0, iscc=150.0,
                      total=600.0, inst_sha="e" * 64)
    earlier.update(version=version, build_info=info)
    earlier["runs"] = [release_notes.run_record(earlier, "full", finished="2026-10-30 23:00:00")]
    (vd / "build-report.json").write_text(json.dumps(earlier), encoding="utf-8")

    icon = tmp_path / "aerolume.ico"
    icon.write_bytes(b"\0")
    monkeypatch.setattr(build, "preflight", lambda *_a, **_k: {
        "python": "3.11", "torch": "x", "torch_cuda": "13.0", "vtk": "x", "PySide6-Essentials": "x",
        "pyinstaller": "x", "iscc": str(tmp_path / "ISCC.exe"), "machine": {"cpu": "Fake CPU"}})
    monkeypatch.setattr(buildlib, "ensure_generated", lambda *_a, **_k: {"icon": icon})
    monkeypatch.setattr(buildlib, "read_ico_sizes", lambda _p: [16])
    monkeypatch.setattr(buildlib, "legacy_build_roots", list)
    smoke = SimpleNamespace(smoke=lambda *_a, **_k: _smoke(12.5, 7.5), _print=lambda _r: None)
    monkeypatch.setitem(sys.modules, "smoke_test", smoke)

    def fake_iscc(_iscc, _dist, out_dir, *_a, **_k):
        out_dir.mkdir(parents=True, exist_ok=True)
        setup = out_dir / f"{buildlib.APP_NAME}-{version}-cuda-setup.exe"
        setup.write_bytes(b"setup" * 10)
        return setup, 3.25

    monkeypatch.setattr(build, "compile_installer", fake_iscc)
    monkeypatch.setattr(buildlib, "inspect_installer",
                        lambda p: {"path": str(p), "bytes": p.stat().st_size, "sha256": "f" * 64,
                                   "inno_signature": "Inno Setup Setup Data (6.7.0)"})
    monkeypatch.setattr(buildlib, "verify_installer_manifest", lambda *_a, **_k: {"ok": True, "files": 1})

    code = build.main(["--build-root", str(tmp_path / "builds"), "--reuse-dist", "--compression", "lzma2/fast"])
    assert code in (0, 3)                     # 3 if someone saved a file in the tree meanwhile
    report = json.loads((vd / "build-report.json").read_text(encoding="utf-8"))
    runs = report["runs"]
    assert [r["kind"] for r in runs] == ["full", "reuse-dist"]
    assert runs[0]["pyinstaller_seconds"] == 400.0 and runs[1]["pyinstaller_seconds"] is None
    assert runs[1]["selftest_first_seconds"] == 12.5 and runs[1]["installer_seconds"] == 3.2
    assert runs[1]["installer_compression"] == "lzma2/fast" and runs[1]["installer_sha256"] == "f" * 64
    assert runs[1]["machine"] == {"cpu": "Fake CPU"}
    notes = vd / "installer" / f"{buildlib.APP_NAME}-{version}-cuda-release-notes.md"
    assert report["release_notes"] == str(notes)
    text = notes.read_text(encoding="utf-8")
    assert "`" + "f" * 64 + "`" in text and "| made by | re-package" in text
    # The notes go with their installer when it is discarded.
    assert notes.name in build.discard_installers(vd)


def test_failed_run_is_recorded_in_the_history(tmp_path, monkeypatch):
    version = buildlib.read_version()
    vd = tmp_path / "builds" / "cuda"
    info = _fake_dist(vd, buildlib.source_fingerprint()["sha256"], version)
    (vd / "build-report.json").write_text(json.dumps({"version": version, "build_info": info}), encoding="utf-8")
    icon = tmp_path / "aerolume.ico"
    icon.write_bytes(b"\0")
    monkeypatch.setattr(build, "preflight", lambda *_a, **_k: {
        "python": "3.11", "torch": "x", "torch_cuda": None, "vtk": "x", "PySide6-Essentials": "x",
        "pyinstaller": "x", "iscc": "x"})
    monkeypatch.setattr(buildlib, "ensure_generated", lambda *_a, **_k: {"icon": icon})
    monkeypatch.setattr(buildlib, "read_ico_sizes", lambda _p: [16])
    monkeypatch.setitem(sys.modules, "smoke_test",
                        SimpleNamespace(smoke=lambda *_a, **_k: _smoke(9.0, 4.0, ok=False), _print=lambda _r: None))
    with pytest.raises(SystemExit, match="smoke test failed"):
        build.main(["--build-root", str(tmp_path / "builds"), "--reuse-dist"])
    report = json.loads((vd / "build-report.json").read_text(encoding="utf-8"))
    assert report["runs"][-1]["kind"] == "reuse-dist" and report["runs"][-1]["smoke_ok"] is False


# ---------------------------------------------------------------------------
# build.py --write-measurements: only a tested build of the tree's program
# ---------------------------------------------------------------------------

def _repo_with_build(tmp_path) -> tuple[Path, Path, SimpleNamespace]:
    repo = tmp_path / "repo"
    for rel, text in {"src/aerolume/__init__.py": '__version__ = "0.3.0"\n',
                      "pyproject.toml": "[project]\nname='aerolume'\n",
                      "packaging/buildlib.py": "# helpers\n",
                      "docs/install.md": "# Install\n<!-- build-measurements:begin -->\n"
                                         "<!-- build-measurements:end -->\n",
                      "CHANGELOG.md": "# Changelog\n- other\n<!-- build-summary:begin -->\n<!-- build-summary:end -->\n"
                      }.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    vd = tmp_path / "out" / "cuda"
    fp = buildlib.snapshot_sources(vd / buildlib.SNAPSHOT_DIR, repo, settle_seconds=0)
    info = _fake_dist(vd, fp["sha256"], "0.3.0")
    rep = _history()
    rep["build_info"] = info
    (vd / "build-report.json").write_text(json.dumps(rep), encoding="utf-8")
    args = SimpleNamespace(build_root=tmp_path / "out", variant="cuda", allow_stale=False)
    return repo, vd, args


def test_write_measurements_fills_both_documents(tmp_path):
    repo, _vd, args = _repo_with_build(tmp_path)
    (repo / "docs/install.md").write_text((repo / "docs/install.md").read_text(encoding="utf-8") + "edited\n",
                                          encoding="utf-8")     # docs-only changes are allowed
    assert build.run_write_measurements(args, repo=repo) == 0
    install = (repo / "docs/install.md").read_text(encoding="utf-8")
    assert "| step | full build `2f0417fc0db2` | docs refresh `686774816f68` |" in install
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "\n<!-- build-summary:begin -->\n- Release build, measured by" in changelog
    assert "docs refresh `686774816f68`, the last recorded run, the bundled self-test passed all 2" in changelog


def test_write_measurements_refuses_another_program_or_an_untested_build(tmp_path):
    repo, vd, args = _repo_with_build(tmp_path)
    (repo / "src/aerolume/__init__.py").write_text('__version__ = "0.3.0"\nX = 1\n', encoding="utf-8")
    before = (repo / "docs/install.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match=r"no longer holds the measured program \(src/aerolume/__init__\.py"):
        build.run_write_measurements(args, repo=repo)
    assert (repo / "docs/install.md").read_text(encoding="utf-8") == before
    args.allow_stale = True                   # quoting the recorded build anyway is explicit
    assert build.run_write_measurements(args, repo=repo) == 0

    report = json.loads((vd / "build-report.json").read_text(encoding="utf-8"))
    report["smoke"] = {"ok": None, "skipped": True}
    (vd / "build-report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke test is skipped"):
        build.run_write_measurements(args, repo=repo)
