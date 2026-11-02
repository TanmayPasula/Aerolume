"""Build freshness and build location (fast; no PyInstaller run).

The failure these guard against happened: a 1.9 GB installer built at 16:01 was still the only
installer after a day of changes, reported nothing about its sources, and sat in a folder that
Windows had silently redirected into a packaged app's private store. So: every build records a
source fingerprint, ``build.py --status`` / ``--reuse-dist`` refuse to call anything else
current, and the default build root is a folder no process sees redirected.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from aerolume import __version__
from aerolume.app import platform as plat

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"


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
sys.path.insert(0, str(PACKAGING))      # build.py imports buildlib (and smoke_test) by name
build = _load("build", "aerolume_packaging_build")   # "build" is also a PyPI package


# ---------------------------------------------------------------------------
# a miniature repository
# ---------------------------------------------------------------------------

def _fake_repo(root: Path, version: str = "9.8.7") -> Path:
    files = {
        "src/aerolume/__init__.py": f'__version__ = "{version}"\n',
        "src/aerolume/app/resources/icon.svg": "<svg/>",
        "src/aerolume/__pycache__/x.cpython-311.pyc": "bytecode",
        "pyproject.toml": "[project]\nname='aerolume'\n",
        "packaging/buildlib.py": "# build helpers\n",
        "packaging/smoke_test.py": "# reads the build only\n",
        "docs/user_guide.md": "# Guide\n",
        "validation/case.py": "print(1)\n",
        "README.md": "readme\n",
        "notes/scratch.txt": "not shipped\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def _age_all(repo: Path, seconds_ago: float) -> None:
    t = time.time() - seconds_ago
    for p in repo.rglob("*"):
        if p.is_file():
            os.utime(p, (t, t))


def _fake_build(variant_dir: Path, *, version: str, source_sha256: str | None, built: str | None = None,
                changed: bool = False, installer: bool = True, installer_hash: str | None = "same") -> None:
    dist = variant_dir / "dist" / buildlib.APP_NAME
    dist.mkdir(parents=True, exist_ok=True)
    (dist / buildlib.GUI_EXE).write_bytes(b"MZ")
    info = {"version": version, "variant": "cuda", "built": built or time.strftime("%Y-%m-%d %H:%M:%S")}
    if source_sha256:
        info["source_sha256"] = source_sha256
    if changed:
        info["source_changed_during_build"] = True
    (dist / "build-info.json").write_text(json.dumps(info), encoding="utf-8")
    report: dict = {"version": version}
    if installer:
        setup = variant_dir / "installer" / f"{buildlib.APP_NAME}-{version}-cuda-setup.exe"
        setup.parent.mkdir(parents=True, exist_ok=True)
        setup.write_bytes(b"setup" * 10)
        report["installer"] = {"path": str(setup), "bytes": setup.stat().st_size,
                               "source_sha256": source_sha256 if installer_hash == "same" else installer_hash}
    (variant_dir / "build-report.json").write_text(json.dumps(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_covers_shipped_inputs_only(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    rels = [rel for rel, _p in buildlib.source_files(repo)]
    assert "src/aerolume/__init__.py" in rels and "src/aerolume/app/resources/icon.svg" in rels
    assert "pyproject.toml" in rels and "packaging/buildlib.py" in rels
    assert "docs/user_guide.md" in rels and "validation/case.py" in rels and "README.md" in rels
    assert not any("__pycache__" in r or r.endswith(".pyc") for r in rels)
    assert "packaging/smoke_test.py" not in rels and "notes/scratch.txt" not in rels

    fp = buildlib.source_fingerprint(repo)
    assert fp == buildlib.source_fingerprint(repo)                      # deterministic
    assert len(fp["sha256"]) == 64 and fp["short"] == fp["sha256"][:12] and fp["files"] == len(rels)

    (repo / "src/aerolume/__pycache__/x.cpython-311.pyc").write_text("other bytecode", encoding="utf-8")
    (repo / "notes/scratch.txt").write_text("edited", encoding="utf-8")
    assert buildlib.source_fingerprint(repo)["sha256"] == fp["sha256"]  # non-inputs do not count

    (repo / "docs/user_guide.md").write_text("# Guide.\n", encoding="utf-8")
    fp2 = buildlib.source_fingerprint(repo)
    assert fp2["sha256"] != fp["sha256"]
    assert fp2["newest_file"] == "docs/user_guide.md"

    # A rename with identical bytes is a different tree too (paths are hashed).
    (repo / "docs/user_guide.md").rename(repo / "docs/guide.md")
    assert buildlib.source_fingerprint(repo)["sha256"] != fp2["sha256"]


def test_fingerprint_of_the_real_tree_is_quick():
    t0 = time.perf_counter()
    fp = buildlib.source_fingerprint()
    dt = time.perf_counter() - t0
    assert fp["files"] > 100 and any(r == "src/aerolume/__init__.py" for r, _ in buildlib.source_files())
    assert dt < 10.0, f"fingerprinting took {dt:.1f} s"


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_missing(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    st = buildlib.build_status(tmp_path / "out" / "cuda", repo=repo)
    assert st["state"] == "missing" and not st["current"] and not st["installer_current"]


def test_status_current_only_for_identical_sources(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    fp = buildlib.source_fingerprint(repo)
    vd = tmp_path / "out" / "cuda"
    _fake_build(vd, version="9.8.7", source_sha256=fp["sha256"])
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "current" and st["current"] and st["installer_current"], st
    assert st["installers"][0]["current"]

    (repo / "src/aerolume/app/resources/icon.svg").write_text("<svg width='1'/>", encoding="utf-8")
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "stale" and not st["current"] and not st["installer_current"]
    assert any("fingerprint" in r for r in st["reasons"])
    assert not st["installers"][0]["current"]


def test_status_stale_on_version_or_mid_build_change(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    fp = buildlib.source_fingerprint(repo)
    vd = tmp_path / "a" / "cuda"
    _fake_build(vd, version="9.8.6", source_sha256=fp["sha256"])
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "stale" and any("version 9.8.6" in r for r in st["reasons"])
    assert any("another version" in r for r in st["installers"][0]["reasons"])

    vd = tmp_path / "b" / "cuda"
    _fake_build(vd, version="9.8.7", source_sha256=fp["sha256"], changed=True)
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "stale" and any("changed while" in r for r in st["reasons"])


def test_status_of_builds_without_a_fingerprint(tmp_path):
    """Pre-fingerprint builds are never current; proven stale once any input is newer."""
    repo = _fake_repo(tmp_path / "repo")
    _age_all(repo, 3600)
    built = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 600))
    vd = tmp_path / "out" / "cuda"
    _fake_build(vd, version="9.8.7", source_sha256=None, built=built)
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "unverifiable" and not st["current"]
    assert any("no source fingerprint" in r for r in st["installers"][0]["reasons"])

    (repo / "src/aerolume/__init__.py").write_text('__version__ = "9.8.7"\n# edited\n', encoding="utf-8")
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "stale"
    assert any("modified after it was built" in r for r in st["reasons"])


def test_installer_from_other_sources_is_not_current(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    fp = buildlib.source_fingerprint(repo)
    vd = tmp_path / "out" / "cuda"
    _fake_build(vd, version="9.8.7", source_sha256=fp["sha256"], installer_hash="0" * 64)
    st = buildlib.build_status(vd, repo=repo)
    assert st["current"] and not st["installer_current"]
    assert "compiled from other sources" in st["installers"][0]["reasons"]


# ---------------------------------------------------------------------------
# build.py entry points (no PyInstaller, no ISCC)
# ---------------------------------------------------------------------------

def test_status_command_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(buildlib, "legacy_build_roots", list)
    root = tmp_path / "builds"
    assert build.main(["--status", "--build-root", str(root), "--variant", "cuda"]) == 1
    assert "MISSING" in capsys.readouterr().out

    fp = buildlib.source_fingerprint()
    _fake_build(root / "cuda", version=buildlib.read_version(), source_sha256=fp["sha256"])
    assert build.main(["--status", "--build-root", str(root), "--variant", "cuda"]) == 0
    out = capsys.readouterr().out
    assert "app folder: CURRENT" in out and "installer: CURRENT" in out

    _fake_build(root / "cuda", version=buildlib.read_version(), source_sha256="f" * 64)
    assert build.main(["--status", "--build-root", str(root), "--variant", "cuda"]) == 1
    assert "STALE" in capsys.readouterr().out


def test_status_command_reports_legacy_root(tmp_path, monkeypatch, capsys):
    legacy = tmp_path / "AppData" / "Local" / "aerolume-build"
    _fake_build(legacy / "cuda", version="0.0.1", source_sha256=None, built="2020-01-01 00:00:00")
    monkeypatch.setattr(buildlib, "legacy_build_roots", lambda: [legacy])
    assert build.main(["--status", "--build-root", str(tmp_path / "new")]) == 1
    out = capsys.readouterr().out
    assert "legacy build root" in out and "Aerolume-0.0.1-cuda-setup.exe" in out and "NOT CURRENT" in out


def test_reuse_dist_refuses_a_stale_folder(tmp_path, monkeypatch):
    def no_preflight(*_a, **_k):
        raise AssertionError("the stale check must come before the slow preflight")
    monkeypatch.setattr(build, "preflight", no_preflight)
    root = tmp_path / "builds"
    _fake_build(root / "cuda", version=buildlib.read_version(), source_sha256="e" * 64)
    with pytest.raises(SystemExit) as exc:
        build.main(["--build-root", str(root), "--reuse-dist", "--skip-smoke", "--skip-installer"])
    assert "stale" in str(exc.value) and "--allow-stale" in str(exc.value)

    _fake_build(root / "cuda", version=buildlib.read_version(), source_sha256=None, built="2020-01-01 00:00:00")
    with pytest.raises(SystemExit) as exc:
        build.main(["--build-root", str(root), "--reuse-dist", "--skip-smoke", "--skip-installer"])
    assert "stale" in str(exc.value)


# ---------------------------------------------------------------------------
# where builds go
# ---------------------------------------------------------------------------

def test_default_build_root_is_outside_appdata_and_onedrive(monkeypatch, tmp_path):
    monkeypatch.delenv(buildlib.BUILD_ROOT_ENV, raising=False)
    root = buildlib.default_build_root()
    assert root.name == "aerolume-build"
    assert "appdata" not in [p.lower() for p in root.parts], root   # AppData is what MSIX redirects
    assert not buildlib.is_under_onedrive(root)
    assert buildlib.redirected(root) is None
    assert not buildlib.is_msix_private(root)
    monkeypatch.setenv(buildlib.BUILD_ROOT_ENV, str(tmp_path / "elsewhere"))
    assert buildlib.default_build_root() == tmp_path / "elsewhere"
    assert all(r.name == "aerolume-build" and "appdata" in [p.lower() for p in r.parts]
               for r in buildlib.legacy_build_roots())


def test_msix_private_store_is_recognised(tmp_path):
    private = Path("C:/Users/u/AppData/Local/Packages/Some.App_abc123/LocalCache/Local/aerolume-build")
    assert buildlib.is_msix_private(private)
    assert not buildlib.is_msix_private(Path("C:/Users/u/aerolume-build"))
    assert not buildlib.is_msix_private(Path("C:/Users/u/Packages/LocalCache-notes"))
    assert buildlib.redirected(tmp_path) is None


# ---------------------------------------------------------------------------
# the frozen app knows which build it is
# ---------------------------------------------------------------------------

@pytest.fixture
def frozen_app(tmp_path, monkeypatch):
    exe = tmp_path / "Aerolume" / "Aerolume.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(plat, "_BUILD_INFO", None)
    yield exe.parent
    plat._BUILD_INFO = None


def test_version_label_from_source():
    assert plat.version_label() == __version__
    if not plat.is_frozen():
        assert plat.version_label(detailed=True) == f"{__version__} (from source)"
        assert plat.build_info() == {}


def test_version_label_and_build_check_in_a_frozen_folder(frozen_app):
    info = {"version": __version__, "variant": "cuda", "built": "2026-10-19 18:00:00",
            "source_sha256": "3f2a9c1b7d0e" + "0" * 52}
    (frozen_app / "build-info.json").write_text(json.dumps(info), encoding="utf-8")
    assert plat.version_label(detailed=True) == f"{__version__} (CUDA build 2026-10-19, source 3f2a9c1b7d0e)"
    assert plat.environment_lines()[0] == f"Aerolume {__version__} (CUDA build 2026-10-19, source 3f2a9c1b7d0e)"
    assert "source 3f2a9c1b7d0e" in plat._check_build_info()


def test_build_check_rejects_a_mismatched_folder(frozen_app):
    (frozen_app / "build-info.json").write_text(json.dumps({"version": "0.0.0", "source_sha256": "a" * 64}),
                                                encoding="utf-8")
    with pytest.raises(RuntimeError, match="0.0.0"):
        plat._check_build_info()
    plat._BUILD_INFO = None
    (frozen_app / "build-info.json").write_text(json.dumps({"version": __version__}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        plat._check_build_info()
    plat._BUILD_INFO = None
    (frozen_app / "build-info.json").unlink()
    assert plat.version_label(detailed=True) == f"{__version__} (installed build)"
    with pytest.raises(FileNotFoundError):
        plat._check_build_info()


# ---------------------------------------------------------------------------
# the build runs from a snapshot of the tree
# ---------------------------------------------------------------------------

def test_snapshot_is_a_byte_identical_copy_of_the_inputs(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    snap = tmp_path / "out" / "cuda" / buildlib.SNAPSHOT_DIR
    fp = buildlib.snapshot_sources(snap, repo, settle_seconds=0)
    assert fp["sha256"] == buildlib.source_fingerprint(repo)["sha256"] and fp["attempts"] == 1
    assert fp["taken"] and (snap / "src/aerolume/__init__.py").is_file()
    assert not (snap / "notes").exists() and not (snap / "src/aerolume/__pycache__").exists()
    # modification times survive, so "newest input" means the same thing in both
    assert buildlib.source_fingerprint(snap)["newest_file"] == buildlib.source_fingerprint(repo)["newest_file"]
    # the snapshot's own version and icon are what the build reads
    assert buildlib.read_version(snap) == "9.8.7"


def test_snapshot_waits_for_a_half_written_file(tmp_path):
    """A file that does not compile is an editor mid-save: wait and copy again."""
    repo = _fake_repo(tmp_path / "repo")
    mod = repo / "src/aerolume/solver.py"
    mod.write_text("def step(:\n", encoding="utf-8")
    seen: list[str] = []

    def fix_on_first_rejection(msg: str) -> None:
        seen.append(msg)
        mod.write_text("def step():\n    return 1\n", encoding="utf-8")

    fp = buildlib.snapshot_sources(tmp_path / "snap", repo, settle_seconds=0, log=fix_on_first_rejection)
    assert fp["attempts"] == 2 and len(seen) == 1 and "solver.py" in seen[0]
    assert (tmp_path / "snap/src/aerolume/solver.py").read_text(encoding="utf-8").startswith("def step():")


def test_snapshot_gives_up_on_a_tree_that_never_settles(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    (repo / "src/aerolume/broken.py").write_text("x = (\n", encoding="utf-8")
    with pytest.raises(buildlib.SnapshotError, match="broken.py"):
        buildlib.snapshot_sources(tmp_path / "snap", repo, attempts=2, settle_seconds=0)


def test_snapshot_rejects_a_tree_edited_while_copying(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path / "repo")
    real_copy = buildlib.shutil.copy2
    edits = {"n": 0}

    def copy_then_edit(src, dst, *a, **k):
        out = real_copy(src, dst, *a, **k)
        if edits["n"] == 0 and Path(src).name == "README.md":
            edits["n"] += 1
            (repo / "docs/user_guide.md").write_text("# Guide, edited mid-copy\n", encoding="utf-8")
        return out

    monkeypatch.setattr(buildlib.shutil, "copy2", copy_then_edit)
    msgs: list[str] = []
    fp = buildlib.snapshot_sources(tmp_path / "snap", repo, settle_seconds=0, log=msgs.append)
    assert fp["attempts"] == 2 and "changed while it was copied" in msgs[0]
    assert fp["sha256"] == buildlib.source_fingerprint(repo)["sha256"]
    assert "edited mid-copy" in (tmp_path / "snap/docs/user_guide.md").read_text(encoding="utf-8")


def test_status_names_the_files_changed_since_the_snapshot(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    vd = tmp_path / "out" / "cuda"
    fp = buildlib.snapshot_sources(vd / buildlib.SNAPSHOT_DIR, repo, settle_seconds=0)
    _fake_build(vd, version="9.8.7", source_sha256=fp["sha256"])
    assert buildlib.build_status(vd, repo=repo)["current"]

    (repo / "docs/user_guide.md").write_text("# Guide v2\n", encoding="utf-8")
    (repo / "docs/new.md").write_text("new\n", encoding="utf-8")
    (repo / "validation/case.py").unlink()
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "stale"
    line = next(r for r in st["reasons"] if r.startswith("since its snapshot"))
    assert "changed 1, added 1 and removed 1" in line
    assert "docs/user_guide.md" in line and "docs/new.md" in line and "validation/case.py" in line


def test_diff_sources():
    a = {"x": "1", "y": "2", "z": "3"}
    b = {"x": "1", "y": "9", "w": "4"}
    assert buildlib.diff_sources(a, b) == {"changed": ["y"], "added": ["w"], "removed": ["z"]}


def test_bundled_package_must_be_the_snapshots(tmp_path):
    snap = _fake_repo(tmp_path / "snap")
    (snap / "src/aerolume/tests").mkdir()
    (snap / "src/aerolume/tests/test_x.py").write_text("pass\n", encoding="utf-8")
    # a developer script in a data folder (no __init__.py on the way) is not shipped, rightly
    (snap / "src/aerolume/app/resources/fonts").mkdir(parents=True)
    (snap / "src/aerolume/app/resources/fonts/build_fonts.py").write_text("pass\n", encoding="utf-8")
    bundled = tmp_path / "dist" / "_internal" / "aerolume"
    for rel in ("__init__.py", "app/resources/icon.svg"):
        (bundled / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundled / rel).write_bytes((snap / "src/aerolume" / rel).read_bytes())
    res = buildlib.verify_bundled_package(tmp_path / "dist", snap)
    assert res["ok"] and res["identical"] == 2 and res["skipped_tests"] == 1, res
    assert res["skipped_scripts"] == ["app/resources/fonts/build_fonts.py"]

    (bundled / "__init__.py").write_text('__version__ = "live-tree"\n', encoding="utf-8")
    (bundled / "app/resources/icon.svg").unlink()
    res = buildlib.verify_bundled_package(tmp_path / "dist", snap)
    assert not res["ok"] and res["differ"] == ["__init__.py"] and res["missing"] == ["app/resources/icon.svg"]


def test_generated_metadata_follows_pyproject(tmp_path):
    """The app ships metadata made from the sources, not the editable install's stale copy."""
    from importlib import metadata

    repo = _fake_repo(tmp_path / "repo", version="4.5.6")
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'aerolume'\ndescription = 'Aero studio'\nlicense = 'Apache-2.0'\n"
        "requires-python = '>=3.11'\ndependencies = ['numpy>=1.26']\n"
        "[project.optional-dependencies]\nai = ['anthropic>=1.0']\n"
        "[project.urls]\nHomepage = 'https://example.org/aerolume'\n", encoding="utf-8")
    out = tmp_path / "gen" / "metadata"
    (out / "aerolume-0.0.1.dist-info").mkdir(parents=True)          # an older one is replaced
    folder = buildlib.write_dist_info(out, repo)
    assert folder.name == "aerolume-4.5.6.dist-info" and [p.name for p in out.iterdir()] == [folder.name]
    dist = metadata.PathDistribution(folder)
    assert dist.version == "4.5.6" and dist.metadata["Summary"] == "Aero studio"
    assert dist.metadata.get_all("Project-URL") == ["Homepage, https://example.org/aerolume"]
    assert dist.requires == ["numpy>=1.26", 'anthropic>=1.0; extra == "ai"']
    assert not (folder / "direct_url.json").exists()


# ---------------------------------------------------------------------------
# a failed smoke test ends the build: non-zero exit, no installer, never "current"
# ---------------------------------------------------------------------------

def _report_smoke(variant_dir: Path, smoke: dict) -> None:
    """Record ``smoke`` in the fake build's report, tied to its dist the way build.py ties it."""
    info = json.loads((variant_dir / "dist" / buildlib.APP_NAME / "build-info.json").read_text(encoding="utf-8"))
    report = json.loads((variant_dir / "build-report.json").read_text(encoding="utf-8"))
    report.update(build_info=info, smoke=smoke)
    (variant_dir / "build-report.json").write_text(json.dumps(report), encoding="utf-8")


_FAILED_SMOKE = {"ok": False, "checks": [
    {"name": "selftest (first launch)", "ok": False, "exit_code": 1,
     "report": {"ok": False, "checks": [{"name": "package metadata", "ok": False}, {"name": "qt", "ok": True}]}},
    {"name": "cli --version", "ok": True}]}


def test_a_folder_that_failed_its_smoke_test_is_never_current(tmp_path):
    """The 24 Oct build matched its sources byte for byte but said it was 0.1.0."""
    repo = _fake_repo(tmp_path / "repo")
    fp = buildlib.source_fingerprint(repo)
    vd = tmp_path / "out" / "cuda"
    _fake_build(vd, version="9.8.7", source_sha256=fp["sha256"])
    _report_smoke(vd, {"ok": True, "checks": []})
    assert buildlib.build_status(vd, repo=repo)["current"]

    _report_smoke(vd, _FAILED_SMOKE)
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "failed" and not st["current"] and not st["installer_current"]
    assert any("selftest (first launch) (package metadata)" in r for r in st["reasons"]), st["reasons"]
    assert "failed its smoke test" in " ".join(st["installers"][0]["reasons"])

    # A report about an older folder neither condemns nor vouches for a rebuilt one.
    info_path = vd / "dist" / buildlib.APP_NAME / "build-info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["built"] = "2099-01-01 00:00:00"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    assert buildlib.build_status(vd, repo=repo)["state"] == "current"


def test_skipped_smoke_test_is_reported_as_untested(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    vd = tmp_path / "out" / "cuda"
    _fake_build(vd, version="9.8.7", source_sha256=buildlib.source_fingerprint(repo)["sha256"])
    _report_smoke(vd, {"ok": None, "skipped": True})
    st = buildlib.build_status(vd, repo=repo)
    assert st["state"] == "untested" and not st["current"]
    assert any("never smoke-tested" in r for r in st["reasons"])


class _FakeSmoke:
    """Stands in for packaging/smoke_test.py (which launches the real executable)."""

    def __init__(self, result: dict):
        self.result, self.calls = result, 0

    def smoke(self, dist, window=True, work=None):
        self.calls += 1
        return self.result

    @staticmethod
    def _print(_result):
        pass


def _no_slow_steps(monkeypatch, tmp_path, smoke_result: dict) -> _FakeSmoke:
    """Run build.main without an interpreter probe, icon rendering or a real executable."""
    icon = tmp_path / "aerolume.ico"
    icon.write_bytes(b"\0")
    monkeypatch.setattr(build, "preflight", lambda *_a, **_k: {
        "python": "3.11", "torch": "x", "torch_cuda": None, "vtk": "x", "PySide6-Essentials": "x",
        "pyinstaller": "x", "iscc": str(tmp_path / "ISCC.exe")})
    monkeypatch.setattr(buildlib, "ensure_generated", lambda *_a, **_k: {"icon": icon})
    monkeypatch.setattr(buildlib, "read_ico_sizes", lambda _p: [16])

    def no_iscc(*_a, **_k):
        raise AssertionError("ISCC must not run after a failed smoke test")

    monkeypatch.setattr(build, "compile_installer", no_iscc)
    fake = _FakeSmoke(smoke_result)
    monkeypatch.setitem(sys.modules, "smoke_test", fake)
    monkeypatch.setattr(buildlib, "legacy_build_roots", list)
    return fake


def test_failed_smoke_test_exits_non_zero_and_leaves_no_installer(tmp_path, monkeypatch, capsys):
    """smoke.ok == False: build.py stops before ISCC, removes the old installer, and --status fails."""
    fake = _no_slow_steps(monkeypatch, tmp_path, _FAILED_SMOKE)
    root = tmp_path / "builds"
    vd = root / "cuda"
    version = buildlib.read_version()
    _fake_build(vd, version=version, source_sha256=buildlib.source_fingerprint()["sha256"])
    (vd / "installer" / f"{buildlib.APP_NAME}-{version}-cuda-manifest.txt").write_text("x")
    assert list((vd / "installer").glob("*-setup.exe"))          # an installer of this very folder

    with pytest.raises(SystemExit) as exc:
        build.main(["--build-root", str(root), "--reuse-dist"])
    assert fake.calls == 1
    code = exc.value.code
    assert code not in (0, None) and "smoke test failed" in str(code) and "package metadata" in str(code)
    assert list((vd / "installer").iterdir()) == []
    report = json.loads((vd / "build-report.json").read_text(encoding="utf-8"))
    assert report["smoke"]["ok"] is False and "installer" not in report
    assert sorted(report["installers_discarded"]) == sorted(
        [f"{buildlib.APP_NAME}-{version}-cuda-setup.exe", f"{buildlib.APP_NAME}-{version}-cuda-manifest.txt"])

    capsys.readouterr()
    assert build.main(["--status", "--build-root", str(root), "--variant", "cuda", "--skip-installer"]) == 1
    out = capsys.readouterr().out
    assert "app folder: FAILED" in out and "NOT current" in out


def test_failed_installer_compile_saves_the_runs_record(tmp_path, monkeypatch):
    """ISCC fails after a passing smoke test: the report is saved with the error, so the folder's
    smoke result and the run's history survive and --reuse-dist can compile the installer next.

    A real 0.1.0 release build lost its LZMA worker ("terminated unexpectedly") after a good
    PyInstaller run and smoke test; nothing was saved, so the report still described the build
    before it, and only a second full build could give a coherent measurement history."""
    _no_slow_steps(monkeypatch, tmp_path, {"ok": True, "checks": []})

    def iscc_fails(*_a, **_k):
        raise SystemExit("ISCC failed with exit code 2; see iscc.log")

    monkeypatch.setattr(build, "compile_installer", iscc_fails)
    root = tmp_path / "builds"
    vd = root / "cuda"
    _fake_build(vd, version=buildlib.read_version(), source_sha256=buildlib.source_fingerprint()["sha256"],
                installer=False)

    with pytest.raises(SystemExit) as exc:
        build.main(["--build-root", str(root), "--reuse-dist"])
    assert "ISCC failed" in str(exc.value.code) and "--reuse-dist" in str(exc.value.code)
    report = json.loads((vd / "build-report.json").read_text(encoding="utf-8"))
    assert report["installer_error"].startswith("ISCC failed") and "installer" not in report
    assert report["smoke"]["ok"] is True and report["total_seconds"] >= 0
    assert report["runs"][-1]["kind"] == "reuse-dist" and report["runs"][-1]["smoke_ok"] is True
    assert report["runs"][-1]["installer_name"] is None


@pytest.mark.skipif(sys.platform != "win32", reason="the build and its process check are Windows-only")
def test_processes_running_from_finds_a_program_started_from_the_folder(tmp_path):
    """The check build.py makes before it deletes an app folder: a program started from inside
    the folder is found (by pid and path), one started elsewhere is not."""
    import subprocess

    dist = tmp_path / "dist" / "Aerolume"
    dist.mkdir(parents=True)
    exe = dist / "probe.exe"
    shutil.copyfile(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "PING.EXE", exe)
    assert buildlib.processes_running_from(dist) == []
    proc = subprocess.Popen([str(exe), "-n", "30", "127.0.0.1"], stdout=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        found = buildlib.processes_running_from(dist)
        assert [pid for pid, _exe in found] == [proc.pid]
        assert Path(found[0][1]).resolve() == exe.resolve()
        assert buildlib.processes_running_from(tmp_path / "elsewhere") == []
    finally:
        proc.kill()
        proc.wait()
    assert buildlib.processes_running_from(dist) == []


def test_discard_installers_only_touches_installer_products(tmp_path):
    vd = tmp_path / "cuda"
    (vd / "installer").mkdir(parents=True)
    for name in ("Aerolume-0.3.0-cuda-setup.exe", "Aerolume-0.3.0-cuda-manifest.txt", "notes.txt"):
        (vd / "installer" / name).write_text("x")
    assert build.discard_installers(vd) == ["Aerolume-0.3.0-cuda-setup.exe", "Aerolume-0.3.0-cuda-manifest.txt"]
    assert [p.name for p in (vd / "installer").iterdir()] == ["notes.txt"]
    assert build.discard_installers(tmp_path / "nothing") == []


# ---------------------------------------------------------------------------
# --refresh-docs: re-copy files PyInstaller never reads, refuse anything else
# ---------------------------------------------------------------------------

def _snapshotted_build(tmp_path) -> tuple[Path, Path]:
    repo = _fake_repo(tmp_path / "repo")
    vd = tmp_path / "out" / "cuda"
    fp = buildlib.snapshot_sources(vd / buildlib.SNAPSHOT_DIR, repo, settle_seconds=0)
    _fake_build(vd, version="9.8.7", source_sha256=fp["sha256"])
    buildlib.copy_app_files(vd / "dist" / buildlib.APP_NAME, vd / buildlib.SNAPSHOT_DIR)
    _report_smoke(vd, {"ok": True, "checks": []})
    return repo, vd


def test_copied_file_paths_match_what_copy_app_files_copies(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    (tmp_path / "dist").mkdir()
    copied = buildlib.copy_app_files(tmp_path / "dist", repo)
    assert buildlib.copied_file_paths(repo) == set(copied)
    assert "src/aerolume/__init__.py" not in copied and "docs/user_guide.md" in copied


def test_refresh_docs_recopies_only_copied_files(tmp_path):
    repo, vd = _snapshotted_build(tmp_path)
    dist = vd / "dist" / buildlib.APP_NAME
    (repo / "docs/user_guide.md").write_text("# Guide with measured numbers\n", encoding="utf-8")
    (repo / "docs/new.md").write_text("new\n", encoding="utf-8")
    (repo / "validation/case.py").unlink()
    assert buildlib.build_status(vd, repo=repo)["state"] == "stale"

    res = buildlib.refresh_copied_files(vd, repo, settle_seconds=0)
    assert res["changed"] == ["docs/user_guide.md"] and res["added"] == ["docs/new.md"]
    assert res["removed"] == ["validation/case.py"]
    assert (dist / "docs/user_guide.md").read_text(encoding="utf-8") == "# Guide with measured numbers\n"
    assert (dist / "docs/new.md").is_file() and not (dist / "validation/case.py").exists()
    info = json.loads((dist / "build-info.json").read_text(encoding="utf-8"))
    assert info["source_sha256"] == buildlib.source_fingerprint(repo)["sha256"]
    assert info["docs_refreshed"][0]["changed"] == ["docs/user_guide.md"]
    assert buildlib.source_fingerprint(vd / buildlib.SNAPSHOT_DIR)["sha256"] == info["source_sha256"]
    assert not (vd / (buildlib.SNAPSHOT_DIR + ".new")).exists()
    st = buildlib.build_status(vd, repo=repo)
    # current again; the old smoke result no longer applies to the changed folder
    assert st["state"] == "current", st["reasons"]
    report = json.loads((vd / "build-report.json").read_text(encoding="utf-8"))
    assert buildlib.smoke_result_for(report, info) is None

    # nothing more to do: an empty refresh
    assert buildlib.refresh_copied_files(vd, repo, settle_seconds=0)["source"] is None


def test_refresh_docs_refuses_when_bundled_inputs_changed(tmp_path):
    repo, vd = _snapshotted_build(tmp_path)
    before = buildlib.source_fingerprint(vd / buildlib.SNAPSHOT_DIR)["sha256"]
    (repo / "docs/user_guide.md").write_text("# Guide v2\n", encoding="utf-8")
    (repo / "src/aerolume/__init__.py").write_text('__version__ = "9.8.7"\nX = 1\n', encoding="utf-8")
    with pytest.raises(buildlib.RefreshRefused, match=r"src/aerolume/__init__\.py.*full build"):
        buildlib.refresh_copied_files(vd, repo, settle_seconds=0)
    assert buildlib.source_fingerprint(vd / buildlib.SNAPSHOT_DIR)["sha256"] == before
    assert not (vd / (buildlib.SNAPSHOT_DIR + ".new")).exists()
    dist = vd / "dist" / buildlib.APP_NAME
    assert (dist / "docs/user_guide.md").read_text(encoding="utf-8") == "# Guide\n"


def test_refresh_docs_refuses_a_failed_or_foreign_folder(tmp_path):
    repo, vd = _snapshotted_build(tmp_path)
    _report_smoke(vd, _FAILED_SMOKE)
    with pytest.raises(buildlib.RefreshRefused, match="failed its smoke test"):
        buildlib.refresh_copied_files(vd, repo, settle_seconds=0)
    (vd / buildlib.SNAPSHOT_DIR / "README.md").write_text("touched\n", encoding="utf-8")
    with pytest.raises(buildlib.RefreshRefused, match="not the snapshot"):
        buildlib.refresh_copied_files(vd, repo, settle_seconds=0)
