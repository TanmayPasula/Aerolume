r"""The built app folder and installer (slow; skipped until ``packaging/build.py`` has run).

Looks for the folder in ``AEROLUME_BUILD_DIST``, then ``%USERPROFILE%\aerolume-build\<variant>\
dist\Aerolume`` (``buildlib.default_build_root``), then the legacy ``%LOCALAPPDATA%`` root.
Launches the frozen executables with the Qt ``offscreen`` platform and a throw-away config
folder; reads the installer as a file and never runs it.

A build made from other sources than the working tree is not silently tested as if it were
current: the checks that compare it with the tree skip and say why (``build.py --status``
gives the same verdict), and ``AEROLUME_REQUIRE_CURRENT_BUILD=1`` turns that skip into a
failure for release checks.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
pytestmark = [pytest.mark.slow, pytest.mark.skipif(sys.platform != "win32", reason="Windows build")]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


buildlib = _load("buildlib")
smoke_test = _load("smoke_test")


def _dist() -> Path | None:
    env = os.environ.get("AEROLUME_BUILD_DIST")
    candidates = [Path(env)] if env else []
    for root in [buildlib.default_build_root(), *buildlib.legacy_build_roots()]:
        candidates += [root / v / "dist" / "Aerolume" for v in buildlib.VARIANTS]
    for c in candidates:
        if (c / buildlib.GUI_EXE).is_file():
            return c
    return None


DIST = _dist()
needs_build = pytest.mark.skipif(DIST is None, reason="no built Aerolume folder (run packaging/build.py)")
STATUS = buildlib.build_status(DIST.parents[1]) if DIST is not None else None


def _require_current() -> None:
    """Skip (or fail, for release checks) when the build does not match the working tree."""
    if STATUS is None or STATUS["current"]:
        return
    why = f"build in {STATUS['real_variant_dir']} is {STATUS['state']}: " + "; ".join(STATUS["reasons"])
    if os.environ.get("AEROLUME_REQUIRE_CURRENT_BUILD") == "1":
        pytest.fail(why)
    pytest.skip(why)


@needs_build
def test_build_matches_the_sources():
    _require_current()
    assert STATUS["current"]


@needs_build
def test_build_passed_its_own_smoke_test():
    """The folder on disk is the one build.py smoke-tested, and that test passed.

    Holds whatever the tree did since: a build that failed its smoke test (24 Oct: bundled
    metadata said 0.1.0) must never be the folder these tests, or a user, pick up as the build.
    """
    report = buildlib.read_json(DIST.parents[1] / "build-report.json") or {}
    info = buildlib.read_json(DIST / "build-info.json") or {}
    verdict = buildlib.smoke_result_for(report, info)
    if verdict is None:
        pytest.skip("build-report.json holds no smoke result for this folder (older build)")
    assert verdict == "passed", buildlib.smoke_failure_summary(report.get("smoke") or {})
    assert STATUS["state"] not in ("failed", "untested")
    if not os.environ.get("AEROLUME_BUILD_DIST"):
        # every installer beside it is the one this report compiled from it
        setups = list((DIST.parents[1] / "installer").glob("Aerolume-*-setup.exe"))
        assert all(s.name == os.path.basename((report.get("installer") or {}).get("path", "")) for s in setups)


@needs_build
def test_build_is_exactly_its_snapshot():
    """Holds even after the tree moves on: the folder contains the sources it says it does."""
    snap = DIST.parents[1] / buildlib.SNAPSHOT_DIR
    info = buildlib.read_json(DIST / "build-info.json") or {}
    if not snap.is_dir() or not info.get("source_sha256"):
        pytest.skip("build made before snapshots (no <variant>/source folder)")
    assert buildlib.source_fingerprint(snap)["sha256"] == info["source_sha256"]
    assert not info.get("source_changed_during_build")
    res = buildlib.verify_bundled_package(DIST, snap)
    assert res["ok"] and res["identical"] > 100, {k: res[k] for k in ("identical", "differ", "missing")}
    assert info.get("version") == buildlib.read_version(snap)


@needs_build
def test_layout_of_the_app_folder():
    assert (DIST / buildlib.CLI_EXE).is_file()
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "build-info.json", "docs/user_guide.md",
                 "validation/ns2d_cases.py", "licenses/README.txt"):
        assert (DIST / name).exists(), name
    internal = DIST / "_internal"
    assert (internal / "aerolume" / "app" / "resources" / "aerolume.svg").is_file()
    assert list((internal / "torch" / "lib").glob("torch_cpu.dll"))
    assert not list(DIST.rglob("__pycache__"))
    own_meta = [p.name for p in internal.glob("aerolume-*.dist-info")]
    assert len(own_meta) == 1, own_meta                       # no stale editable-install metadata
    version = buildlib.pe_version_strings(DIST / buildlib.GUI_EXE)
    assert version.get("ProductName") == "Aerolume"
    _require_current()
    assert version.get("FileVersion") == buildlib.read_version()
    info = buildlib.read_json(DIST / "build-info.json") or {}
    assert info.get("source_sha256") == buildlib.source_fingerprint()["sha256"]


@needs_build
def test_every_bundled_component_has_its_licence_or_notice():
    """The licences folder covers Qt (LGPL-3.0 and GPL-3.0 texts), CPython (PSF and the libraries in its
    LICENSE.txt and license.html), every NVIDIA DLL in torch/lib and Intel's OpenMP runtime, and keeps the
    third-party licences PyTorch ships in folders (release gate, 1 November 2026)."""
    _require_current()
    lic = DIST / "licenses"
    for dist in ("PySide6_Essentials", "shiboken6"):
        for text in ("LGPL-3.0.txt", "GPL-3.0.txt", "README.txt"):
            assert (lic / dist / text).is_file(), (dist, text)
    assert (lic / "python" / "LICENSE.txt").is_file() and (lic / "python" / "license.html").is_file()
    torch_lib = DIST / "_internal" / "torch" / "lib"
    nvidia = sorted(p.name for p in torch_lib.glob("*.dll") if buildlib.NVIDIA_DLL_RE.match(p.name))
    if nvidia:                                                  # the CUDA variant
        notice = (lic / "nvidia-cuda" / "NOTICE.txt").read_text(encoding="utf-8")
        missing = [n for n in nvidia if f"_internal/torch/lib/{n}" not in notice]
        assert not missing, missing
    if list(torch_lib.glob("libiomp5md.dll")):
        assert "libiomp5md.dll" in (lic / "intel-openmp" / "NOTICE.txt").read_text(encoding="utf-8")
    assert len(list((lic / "torch" / "third_party").rglob("LICENSE*"))) > 50
    index = (lic / "README.txt").read_text(encoding="utf-8")
    assert "python/" in index and "LGPL-3.0" in index
    # VTK's vendored libraries and the third-party code in Qt (release gate 3, 2 November 2026)
    vtk_dlls = sorted(m.group(1) for p in DIST.rglob("vtk*.dll") if (m := buildlib.VTK_THIRD_PARTY_RE.match(p.name)))
    if vtk_dlls:
        vtk_index = (lic / "vtk" / "third_party" / "README.txt").read_text(encoding="utf-8")
        missing = [d for d in vtk_dlls if f"{d}-" not in vtk_index]
        assert not missing, missing
    qt_index = (lic / "PySide6_Essentials" / "third_party" / "README.txt").read_text(encoding="utf-8")
    for needle in ("HarfBuzz", "Freetype", "LibPNG", "MD4C", "Mesa"):
        assert needle in qt_index, needle


@needs_build
def test_frozen_self_test_passes(tmp_path):
    result = smoke_test.run_selftest(DIST, tmp_path, window=True)
    bad = [c for c in result["report"].get("checks", []) if not c["ok"]]
    assert result["ok"], (result["exit_code"], bad)


@needs_build
def test_console_executable(tmp_path):
    _require_current()
    r = smoke_test.run_cli(DIST, tmp_path, ["--version"], expect=buildlib.read_version())
    assert r["ok"], r


@needs_build
def test_installer_is_complete():
    setups = sorted((DIST.parents[1] / "installer").glob("Aerolume-*-setup.exe"))
    if not setups:
        pytest.skip("installer not compiled")
    _require_current()
    info = buildlib.inspect_installer(setups[-1])
    assert info["inno_signature"] and info["inno_signature"].startswith("Inno Setup Setup Data (")
    assert info["version"].get("ProductVersion", "").startswith(buildlib.read_version())
    manifest = setups[-1].with_name(setups[-1].name.replace("-setup.exe", "-manifest.txt"))
    if manifest.is_file():  # installers compiled before aerolume.iss asked for a manifest have none
        check = buildlib.verify_installer_manifest(manifest, DIST, check_hashes=False)
        assert check["ok"], check
