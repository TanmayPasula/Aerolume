r"""Build the Aerolume Windows app: PyInstaller folder, smoke test, Inno Setup installer.

    python packaging/build.py --variant cuda            # dist folder + installer
    python packaging/build.py --variant cpu --python C:/venvs/aerolume-cpu/Scripts/python.exe
    python packaging/build.py --skip-installer          # portable folder only
    python packaging/build.py --status                  # is the existing build current? (exit 0/1)
    python packaging/build.py --refresh-docs            # only docs changed: re-copy, re-test, re-pack
    python packaging/build.py --write-measurements      # write the measured numbers into the docs

Everything is written under ``%USERPROFILE%\aerolume-build\<variant>`` (``--build-root`` or the
``AEROLUME_BUILD_ROOT`` variable change it), never inside the repository: the repository lives
in OneDrive on the development machine, and OneDrive opens freshly written files to sync them,
which makes PyInstaller fail with "permission denied" half-way through copying 2.5 GB of CUDA
libraries. The build refuses an output folder inside OneDrive unless ``--allow-onedrive`` is
given. Earlier versions built under ``%LOCALAPPDATA%``, which a packaged (MSIX) host app
redirects to its private ``Packages\<app>\LocalCache`` folder where Explorer does not look;
``--status`` still reports builds found there, and every path printed names the real folder.

Snapshot: the build never reads the working tree directly. It first copies every input
(``buildlib.source_files``: the ``aerolume`` package, ``pyproject.toml``, the packaging scripts,
docs, licences) to ``<variant>/source/`` - only once the tree holds still and every ``.py`` file
compiles - and runs PyInstaller, the post-processing and the installer from that copy. People
editing the tree during the ten-minute build therefore cannot produce a folder that mixes two
versions; afterwards the bundled ``aerolume`` files are compared with the snapshot byte for byte.

Freshness: the snapshot's SHA-256 fingerprint is recorded in ``dist/Aerolume/build-info.json``
and in ``build-report.json``. ``--status`` and ``--reuse-dist`` compare it with the working tree
and never call a build "current" when the version, a single byte of input, or the fingerprint
itself is missing or different; while the snapshot is kept they also name the files that
changed since. A folder whose smoke test failed is reported as FAILED (and one built with
``--skip-smoke`` as UNTESTED), never current; a failed smoke test also removes any installer in
``<variant>/installer``, so no installer outlives the folder it would install. Exit codes: 0
built and still matching the tree; 3 built correctly from the snapshot, but the tree has moved
on since (or the smoke test was skipped); anything else failed.

``--refresh-docs``: the files ``copy_app_files`` puts next to the executable (README, CHANGELOG,
docs, validation scripts) are never compiled into the bundle. When those are all that changed
since the snapshot - typically because the release build's measurements were written into
``docs/install.md`` - this re-copies them, re-runs the smoke test and recompiles the installer
in a few minutes instead of a full rebuild (``build-report.json`` records each run's time). It refuses if any bundled input changed.

Measurements: every run appends a record to ``runs`` in ``build-report.json`` (a full build
starts a new history, a refresh or re-package extends it) and writes
``installer/Aerolume-<version>-<variant>-release-notes.md`` beside the installer it compiled,
with that installer's SHA-256 and timings. ``--write-measurements`` copies the history into the
marked blocks of ``docs/install.md`` (and of ``CHANGELOG.md`` when it carries the markers), so
no release number is typed by hand and every one names the run it came from (see
``release_notes.py``). Release order: full build, ``--write-measurements``, ``--refresh-docs``.

Layout of the build root::

    <variant>/source/     the snapshot of the inputs the build was made from
    <variant>/gen/       generated icon and version resources
    <variant>/work/       PyInstaller work folder (deleted after a successful build)
    <variant>/dist/Aerolume/   the portable app folder (what the installer installs)
    <variant>/installer/  Aerolume-<version>-<variant>-setup.exe, -manifest.txt, -release-notes.md
    <variant>/build-report.json   versions, sizes, timings, fingerprint and smoke-test results

Variants: ``cuda`` bundles the CUDA build of PyTorch the interpreter has; ``cpu`` needs an
interpreter whose torch is the CPU-only wheel (``pip install torch --index-url
https://download.pytorch.org/whl/cpu`` in a separate virtual environment) and gives a
~0.5 GB app instead of ~4.5 GB. The script checks that the interpreter matches the variant.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import buildlib
import release_notes

#: Free space needed before starting (dist + installer + work, measured on the CUDA build).
REQUIRED_FREE_BYTES = {"cuda": 12 * 10**9, "cpu": 3 * 10**9}


def _log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe_interpreter(python: Path) -> dict:
    """Versions of the build interpreter's key packages (run in that interpreter)."""
    code = (
        "import json, sys, importlib.metadata as m\n"
        "out = {'python': sys.version.split()[0], 'prefix': sys.prefix}\n"
        "for d in ('torch', 'vtk', 'PySide6-Essentials', 'pyinstaller', 'numpy', 'scipy', 'anthropic'):\n"
        "    try: out[d] = m.version(d)\n"
        "    except m.PackageNotFoundError: out[d] = None\n"
        "import torch\n"
        "out['torch_cuda'] = torch.version.cuda\n"
        "print(json.dumps(out))\n"
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([str(python), "-c", code], capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"cannot inspect {python}:\n{proc.stderr[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def preflight(args, python: Path) -> dict:
    """Check interpreter, variant, output location and disk space before a long build."""
    info = probe_interpreter(python)
    if not info.get("pyinstaller"):
        raise SystemExit(f"PyInstaller is not installed in {python}")
    cuda = info.get("torch_cuda")
    if args.variant == "cuda" and not cuda:
        raise SystemExit("variant 'cuda' needs a CUDA build of torch; this interpreter has the CPU-only wheel")
    if args.variant == "cpu" and cuda:
        raise SystemExit(f"variant 'cpu' needs the CPU-only torch wheel, but this interpreter has CUDA {cuda}.\n"
                         "Create a separate venv with: pip install torch --index-url "
                         "https://download.pytorch.org/whl/cpu  and pass --python <that venv>\\Scripts\\python.exe")
    root = args.build_root
    if buildlib.is_under_onedrive(root) and not args.allow_onedrive:
        raise SystemExit(f"{root} is inside OneDrive; OneDrive locks files while PyInstaller writes them. "
                         "Use the default build root or pass --allow-onedrive.")
    root.mkdir(parents=True, exist_ok=True)
    real = buildlib.real_location(root)
    moved = buildlib.redirected(root)
    if moved is not None:
        # A process inside an MSIX-packaged app (some IDEs and assistants) has AppData
        # redirected to the package's private folder: the build would land where Explorer and
        # the user's own shell do not look. Say so loudly and print both spellings.
        _log(f"WARNING: Windows redirects {root} for this process to {moved}")
        if buildlib.is_msix_private(root):
            _log("         that is a packaged app's private store; other programs will not see the build "
                 f"at {root}. Use the default build root ({buildlib.default_build_root()}) instead.")
    info["build_root_real"] = str(real)
    info["build_root_redirected"] = moved is not None
    free = shutil.disk_usage(root).free
    need = REQUIRED_FREE_BYTES[args.variant]
    if free < need:
        raise SystemExit(f"only {buildlib.human(free)} free on {root.anchor}; the {args.variant} build needs "
                         f"about {buildlib.human(need)}")
    info["free_bytes_before"] = free
    info["machine"] = release_notes.machine_info()
    if not args.skip_installer:
        iscc = buildlib.find_iscc()
        if iscc is None:
            raise SystemExit("Inno Setup 6 (ISCC.exe) not found; install it or pass --skip-installer "
                             "(set AEROLUME_ISCC to its path if it is somewhere unusual)")
        info["iscc"] = str(iscc)
    return info


def run_pyinstaller(python: Path, variant_dir: Path, variant: str, gen: Path, log_path: Path,
                    source_root: Path = buildlib.REPO) -> float:
    """Run the spec found in ``source_root``; returns the wall time (full log in ``log_path``).

    ``source_root`` is the snapshot. Its own ``packaging/aerolume.spec`` and ``buildlib`` run,
    so every path they derive points into the snapshot, and ``PYTHONPATH`` puts the snapshot's
    ``src`` ahead of the editable install's ``.pth`` entry (which points at the live tree) - in
    the analysis process and in the child processes PyInstaller spawns to run hooks.
    """
    spec = source_root / "packaging" / "aerolume.spec"
    cmd = [str(python), "-m", "PyInstaller", str(spec), "--noconfirm", "--clean",
           "--distpath", str(variant_dir / "dist"), "--workpath", str(variant_dir / "work"),
           "--log-level", "INFO"]
    env = dict(os.environ)
    env.update(AEROLUME_VARIANT=variant, AEROLUME_GEN_DIR=str(gen), PYTHONIOENCODING="utf-8",
               PYTHONUTF8="1", QT_QPA_PLATFORM="offscreen")
    env["PYTHONPATH"] = str(source_root / "src")
    _log("PyInstaller: " + " ".join(cmd))
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                                encoding="utf-8", errors="replace", cwd=str(variant_dir))
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if any(k in line for k in ("ERROR", "Building EXE", "Building COLLECT", "completed successfully",
                                       "Analyzing ")):
                print("    " + line.rstrip()[:200], flush=True)
        code = proc.wait()
    seconds = time.perf_counter() - t0
    if code != 0:
        raise SystemExit(f"PyInstaller failed with exit code {code}; see {log_path}")
    return seconds


def postprocess(dist: Path, work: Path, variant: str, version: str, info: dict, source: dict,
                changed_during_build: bool = False, source_root: Path = buildlib.REPO) -> dict:
    """Copy user-facing files and licences, write build-info.json, measure the folder.

    ``build-info.json`` ships inside the app (the frozen app shows it in crash reports and the
    self-test) and carries the source fingerprint, so any copy of the folder - or the installed
    app - can be traced back to the exact sources and compared with a tree by ``--status``.
    """
    copied = buildlib.copy_app_files(dist, repo=source_root)
    tops_file = work / "aerolume" / "bundled_modules.json"
    if not tops_file.is_file():
        tops_file = next(work.rglob("bundled_modules.json"), tops_file)
    tops = json.loads(tops_file.read_text(encoding="utf-8")) if tops_file.is_file() else []
    licences = buildlib.collect_licenses(dist, tops, repo=source_root)
    pyc = buildlib.remove_pycache(dist)
    # the folder aerolume.spec generated (AEROLUME_GEN_DIR = <variant>/gen, see run_pyinstaller)
    generated = dist.parent.parent / "gen" / "metadata" / f"aerolume-{version}.dist-info"
    stale_meta = buildlib.prune_own_metadata(dist, version, generated=generated)
    if stale_meta:
        _log("removed stale package metadata collected from the editable install: " + ", ".join(stale_meta))
    build_info = {"version": version, "variant": variant, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "python": info.get("python"), "torch": info.get("torch"), "cuda": info.get("torch_cuda"),
                  "vtk": info.get("vtk"), "pyside6": info.get("PySide6-Essentials"),
                  "pyinstaller": info.get("pyinstaller"),
                  "source_sha256": source["sha256"], "source_files": source["files"],
                  "source_newest": source["newest"], "source_newest_file": source["newest_file"],
                  "source_snapshot_taken": source.get("taken"),
                  "source_changed_during_build": bool(changed_during_build)}
    (dist / "build-info.json").write_text(json.dumps(build_info, indent=2), encoding="utf-8")
    stats = buildlib.dir_stats(dist)
    return {"copied_app_files": len(copied), "licence_distributions": len(licences), "pycache_removed": pyc,
            "stale_metadata_removed": stale_meta,
            "dist_bytes": stats["bytes"], "dist_files": stats["files"],
            "largest_files": buildlib.largest_files(dist, 15), "build_info": build_info}


def compile_installer(iscc: Path, dist: Path, out_dir: Path, gen: Path, variant: str, version: str,
                      compression: str, threads: int, log_path: Path,
                      source_root: Path = buildlib.REPO) -> tuple[Path, float]:
    """Compile ``aerolume.iss`` with ISCC; returns (setup.exe, wall time)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    quad = ".".join(str(n) for n in buildlib.version_quad(version))
    cmd = [str(iscc), "/Qp", f"/DAppVersion={version}", f"/DVersionQuad={quad}", f"/DVariant={variant}",
           f"/DSourceDir={dist}", f"/DOutputDir={out_dir}", f"/DIconFile={gen / 'aerolume.ico'}",
           f"/DCompression={compression}", f"/DLzmaThreads={threads}",
           str(source_root / "packaging" / "aerolume.iss")]
    _log("ISCC: " + " ".join(cmd))
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    seconds = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(f"ISCC failed with exit code {proc.returncode}; see {log_path}")
    setup = out_dir / f"Aerolume-{version}-{variant}-setup.exe"
    if not setup.is_file():
        raise SystemExit(f"ISCC succeeded but {setup} is missing; see {log_path}")
    return setup, seconds


def installer_manifest(setup: Path) -> Path:
    """The ``OutputManifestFile`` ISCC writes next to ``setup`` (named in ``aerolume.iss``)."""
    return setup.with_name(setup.name.replace("-setup.exe", "-manifest.txt"))


def discard_installers(variant_dir: Path) -> list[str]:
    """Delete the compiled installers (and their manifests) in ``<variant>/installer``.

    Called when the app folder they were compiled from is replaced or fails its smoke test: an
    installer is only meaningful next to the report and folder it came from, and the one that
    survived a failed 24 Oct rebuild was the kind of file people pick up as "the release". The
    build root is this script's own output, so removing its stale products is part of the build.
    """
    out = variant_dir / "installer"
    removed = []
    for pattern in (f"{buildlib.APP_NAME}-*-setup.exe", f"{buildlib.APP_NAME}-*-manifest.txt",
                    f"{buildlib.APP_NAME}-*-release-notes.md"):
        for p in sorted(out.glob(pattern)):
            p.unlink()
            removed.append(p.name)
    if removed:
        _log("removed installer files that no longer match the app folder: " + ", ".join(removed))
    return removed


def _write_report(variant_dir: Path, report: dict) -> Path:
    path = variant_dir / "build-report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def print_status(status: dict, *, label: str = "") -> None:
    """One build's freshness in plain words; both path spellings when Windows redirects."""
    print(f"{label}{status['variant_dir']}")
    if status["real_variant_dir"].lower() != status["variant_dir"].lower():
        print(f"    really at {status['real_variant_dir']} (Windows redirects the path above for this "
              "process only; Explorer and other programs see this one)")
    built = f"version {status['built_version']}, built {status['built']}" if status["built_version"] else "no build"
    print(f"    app folder: {status['state'].upper()} ({built}; source {status['source']}, "
          f"build {status['built_source'] or 'unrecorded'})")
    for r in status["reasons"]:
        print(f"      - {r}")
    for inst in status["installers"]:
        verdict = "CURRENT" if inst["current"] else "NOT CURRENT"
        print(f"    installer: {verdict}  {Path(inst['path']).name} ({buildlib.human(inst['bytes'])}, "
              f"{inst['modified']})")
        if inst["real_path"].lower() != inst["path"].lower():
            print(f"      really at {inst['real_path']}")
        for r in inst["reasons"]:
            print(f"      - {r}")
    if not status["installers"] and status["state"] != "missing":
        print("    installer: none compiled")


def run_status(args) -> int:
    """``--status``: report the build root (and any legacy root); exit 0 only if current.

    "Current" means the app folder - and, unless ``--skip-installer``, an installer compiled
    from it - match the working tree's version and source fingerprint byte for byte.
    """
    fp = buildlib.source_fingerprint()
    version = buildlib.read_version()
    print(f"Source tree: version {version}, fingerprint {fp['short']} over {fp['files']} files, "
          f"newest {fp['newest_file']} ({fp['newest']})")
    variants = [args.variant] if args.variant_given else list(buildlib.VARIANTS)
    ok = False
    for variant in variants:
        st = buildlib.build_status(args.build_root / variant, version=version, fingerprint=fp)
        if variant == args.variant:
            ok = st["current"] and (args.skip_installer or st["installer_current"])
        if st["state"] == "missing" and not st["installers"] and variant != args.variant:
            continue
        print_status(st)
    seen = {os.path.normcase(str(buildlib.real_location(args.build_root)))}
    for legacy in buildlib.legacy_build_roots():
        if os.path.normcase(str(buildlib.real_location(legacy))) in seen or not legacy.is_dir():
            continue
        for variant in buildlib.VARIANTS:
            st = buildlib.build_status(legacy / variant, version=version, fingerprint=fp)
            if st["state"] != "missing" or st["installers"]:
                print_status(st, label="legacy build root (no longer used): ")
    what = "app folder" if args.skip_installer else "app folder and installer"
    print(f"\n{args.variant}: {what} {'CURRENT' if ok else 'NOT current - rebuild with packaging/build.py'}")
    return 0 if ok else 1


def bundled_changes_since_build(variant_dir: Path, repo: Path = buildlib.REPO) -> list[str] | None:
    """Inputs that differ between the build's snapshot and ``repo`` and are not copied docs.

    ``None`` when there is no snapshot to compare with. Empty means the working tree still holds
    the program that was measured - only files that ``--refresh-docs`` re-copies may differ.
    """
    snap = variant_dir / buildlib.SNAPSHOT_DIR
    if not snap.is_dir():
        return None
    diff = buildlib.diff_sources(buildlib.file_hashes(snap), buildlib.file_hashes(repo))
    copied = buildlib.copied_file_paths(snap) | buildlib.copied_file_paths(repo)
    return sorted(n for n in diff["changed"] + diff["added"] + diff["removed"] if n not in copied)


def measurement_targets(repo: Path = buildlib.REPO) -> dict[Path, str]:
    """The documents ``--write-measurements`` fills, and the generated block each carries."""
    return {repo / "docs" / "install.md": release_notes.MEASUREMENTS_BLOCK,
            repo / "CHANGELOG.md": release_notes.SUMMARY_BLOCK}


def run_write_measurements(args, repo: Path = buildlib.REPO) -> int:
    """``--write-measurements``: the docs quote the recorded runs of the build being released.

    Refused when the numbers would describe something other than the program in the tree: no
    report, a folder that did not pass its smoke test, no installer, or a bundled input (code,
    ``pyproject.toml``, a packaging script) changed since the build's snapshot. Docs-only
    differences are fine - they are what ``--refresh-docs`` ships next.
    """
    variant_dir = args.build_root / args.variant
    report = buildlib.read_json(variant_dir / "build-report.json")
    if not report:
        raise SystemExit(f"--write-measurements: no build-report.json in {variant_dir}; build first")
    info = buildlib.read_json(variant_dir / "dist" / buildlib.APP_NAME / "build-info.json") or {}
    verdict = buildlib.smoke_result_for(report, info)
    if verdict != "passed":
        raise SystemExit(f"--write-measurements: the app folder's smoke test is {verdict or 'not recorded'}; "
                         "only a tested build's numbers go into the docs")
    if not (report.get("installer") or {}).get("sha256"):
        raise SystemExit("--write-measurements: the report records no compiled installer")
    if report.get("version") != buildlib.read_version(repo):
        raise SystemExit(f"--write-measurements: the build is version {report.get('version')}, "
                         f"the tree is {buildlib.read_version(repo)}")
    bundled = bundled_changes_since_build(variant_dir, repo)
    if bundled is None:
        raise SystemExit(f"--write-measurements: {variant_dir / buildlib.SNAPSHOT_DIR} is missing, so nothing "
                         "proves which program the numbers describe")
    if bundled and not args.allow_stale:
        more = f" and {len(bundled) - 6} more" if len(bundled) > 6 else ""
        raise SystemExit("--write-measurements: the tree no longer holds the measured program ("
                         + ", ".join(bundled[:6]) + more + " changed since the build's snapshot); "
                         "rebuild first, or pass --allow-stale to quote the recorded build anyway "
                         "(the block names its fingerprint either way)")
    result = release_notes.write_blocks(report, measurement_targets(repo))
    for path in result["changed"]:
        _log(f"wrote the measured runs into {path}")
    for path in result["unchanged"]:
        _log(f"{path} already quotes these runs")
    for path in result["no_markers"]:
        _log(f"skipped {path}: it has no generated block (add the markers to let the build write it)")
    runs = release_notes.runs_of(report)
    _log(f"{len(runs)} run(s) quoted, last: {runs[-1]['kind']} {str(runs[-1].get('source_sha256'))[:12]}")
    if bundled:
        _log("the tree's program differs from this build: a full build, then --write-measurements again, "
             "is needed before a release")
    elif result["changed"]:
        _log("the docs changed: run  python packaging/build.py --refresh-docs  to ship them")
    return 0


def run_release_notes(args) -> int:
    """``--release-notes``: (re)write the file beside the installer from the current report."""
    variant_dir = args.build_root / args.variant
    report = buildlib.read_json(variant_dir / "build-report.json")
    inst = (report or {}).get("installer") or {}
    setup = Path(inst.get("path", ""))
    if not report or not inst.get("sha256") or not setup.is_file():
        raise SystemExit(f"--release-notes: no compiled installer recorded in {variant_dir / 'build-report.json'}")
    if setup.stat().st_size != inst.get("bytes"):
        raise SystemExit(f"--release-notes: {setup} is not the installer the report describes (size differs)")
    notes = release_notes.write_release_notes(report, setup)
    _log(f"wrote {notes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=buildlib.VARIANTS, default=None, help="cuda (default) or cpu")
    ap.add_argument("--python", type=Path, default=Path(sys.executable),
                    help="interpreter whose packages are bundled (default: this one)")
    ap.add_argument("--build-root", type=Path, default=buildlib.default_build_root(),
                    help=f"default: %%USERPROFILE%%\\aerolume-build, or ${buildlib.BUILD_ROOT_ENV} when set")
    ap.add_argument("--allow-onedrive", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="only report whether the existing build matches the source tree (exit 0 if current)")
    ap.add_argument("--skip-installer", action="store_true", help="build the portable folder only")
    ap.add_argument("--skip-smoke", action="store_true", help="do not launch the built executable")
    ap.add_argument("--reuse-dist", action="store_true",
                    help="skip PyInstaller and reuse the existing dist folder (installer / smoke only); "
                         "refused unless that folder is current")
    ap.add_argument("--refresh-docs", action="store_true",
                    help="skip PyInstaller: when only copied files (docs, README, CHANGELOG, validation "
                         "scripts) changed since the build's snapshot, re-copy them, re-run the smoke "
                         "test and recompile the installer; refused if anything bundled changed")
    ap.add_argument("--write-measurements", action="store_true",
                    help="copy the measured run history of build-report.json into the marked blocks of "
                         "docs/install.md and CHANGELOG.md (then run --refresh-docs to ship them)")
    ap.add_argument("--release-notes", action="store_true",
                    help="rewrite the release-notes file beside the installer from build-report.json")
    ap.add_argument("--allow-stale", action="store_true",
                    help="with --reuse-dist: package a folder that does not match the sources "
                         "(the report and --status say so); with --write-measurements: quote a build "
                         "whose program the tree has moved on from (the block names its fingerprint)")
    ap.add_argument("--keep-work", action="store_true", help="keep PyInstaller's work folder")
    ap.add_argument("--compression", default="lzma2/max",
                    help="Inno Setup compression (lzma2/max is ~2x smaller than zip; lzma2/fast is quicker)")
    ap.add_argument("--lzma-threads", type=int, default=4,
                    help="LZMA block threads (each needs ~700 MB of RAM with lzma2/max)")
    args = ap.parse_args(argv)
    args.variant_given = args.variant is not None
    args.variant = args.variant or "cuda"
    if args.status:
        return run_status(args)
    if args.write_measurements:
        return run_write_measurements(args)
    if args.release_notes:
        return run_release_notes(args)

    t_all = time.perf_counter()
    cpu_start = release_notes.system_cpu_times()
    variant_dir = args.build_root / args.variant
    gen = variant_dir / "gen"
    dist = variant_dir / "dist" / "Aerolume"
    work = variant_dir / "work"
    info: dict | None = None
    previous = buildlib.read_json(variant_dir / "build-report.json") or {}
    refreshed: dict | None = None
    if args.reuse_dist and args.refresh_docs:
        raise SystemExit("--reuse-dist and --refresh-docs are alternatives; pass one")
    # A full build replaces the program, so its history starts again; a refresh or re-package
    # tests and packs the same program, so the full build that compiled it stays on record.
    kind = "reuse-dist" if args.reuse_dist else "refresh-docs" if args.refresh_docs else "full"
    prior_runs = [] if kind == "full" else release_notes.runs_of(previous)

    def _save(report: dict) -> Path:
        """Write the report with this run appended to its history (also on failure)."""
        busy = release_notes.cpu_busy_percent(cpu_start, release_notes.system_cpu_times())
        report["runs"] = prior_runs + [release_notes.run_record(report, kind, cpu_busy=busy)]
        return _write_report(variant_dir, report)

    if args.reuse_dist:
        # Nothing is read from the sources except to prove the reused folder matches them.
        source_root = buildlib.REPO
        source = buildlib.source_fingerprint()
    elif args.refresh_docs:
        info = preflight(args, args.python)
        _log("refreshing the copied files of the existing build (no PyInstaller run)")
        try:
            refreshed = buildlib.refresh_copied_files(variant_dir, log=_log)
        except (buildlib.RefreshRefused, buildlib.SnapshotError) as exc:
            raise SystemExit(f"--refresh-docs: {exc}") from None
        source_root = variant_dir / buildlib.SNAPSHOT_DIR
        source = refreshed["source"] or previous.get("source") or buildlib.source_fingerprint(source_root)
        if not refreshed["source"]:
            _log("the snapshot already matches the tree; re-testing and re-packaging the folder as it is")
    else:
        info = preflight(args, args.python)     # refuses OneDrive, low disk, a wrong torch first
        snapshot = variant_dir / buildlib.SNAPSHOT_DIR
        _log(f"copying the inputs to {snapshot} once the tree holds still")
        try:
            source = buildlib.snapshot_sources(snapshot, log=_log)
        except buildlib.SnapshotError as exc:
            raise SystemExit(str(exc)) from None
        source_root = snapshot
        _log(f"snapshot taken on attempt {source['attempts']}; the build reads nothing else from the tree")
    version = buildlib.read_version(source_root)
    report: dict = {"version": version, "variant": variant_dir.name, "build_root": str(args.build_root),
                    "source": source, "source_root": str(source_root)}
    if refreshed is not None:
        # Keep what the PyInstaller run measured (it is still the folder's history); drop every
        # result that described the folder before its files were re-copied.
        dropped = ("smoke", "smoke_seconds", "installer", "installer_seconds", "installer_manifest",
                   "installers_discarded", "total_seconds")
        report = {**{k: v for k, v in previous.items() if k not in dropped}, **report}
        report["docs_refresh"] = {k: refreshed[k] for k in ("changed", "added", "removed")}
        report["build_info"] = buildlib.read_json(dist / "build-info.json") or {}

    _log(f"Aerolume {version}, variant {args.variant}, output {variant_dir}")
    _log(f"source fingerprint {source['short']} over {source['files']} files "
         f"(newest {source['newest_file']}, {source['newest']})")
    if args.reuse_dist:
        # Checked before the (slow) preflight: packaging an old folder under today's version
        # number is exactly how a stale installer gets called current.
        st = buildlib.build_status(variant_dir, version=version, fingerprint=source)
        if st["state"] == "missing":
            raise SystemExit(f"--reuse-dist: {dist} has no {buildlib.GUI_EXE}")
        if not st["current"]:
            msg = f"--reuse-dist: {dist} is {st['state']}: " + "; ".join(st["reasons"])
            if not args.allow_stale:
                raise SystemExit(msg + "\nRebuild without --reuse-dist (or pass --allow-stale to package it anyway).")
            _log("WARNING " + msg)
            report["stale_dist_packaged"] = st["reasons"]
        report["build_info"] = buildlib.read_json(dist / "build-info.json") or {}

    if info is None:
        info = preflight(args, args.python)
    report["interpreter"] = info
    _log(f"interpreter {args.python}: Python {info['python']}, torch {info['torch']} (CUDA {info['torch_cuda']}), "
         f"VTK {info['vtk']}, PySide6 {info['PySide6-Essentials']}, PyInstaller {info['pyinstaller']}")
    variant_dir.mkdir(parents=True, exist_ok=True)

    files = buildlib.ensure_generated(gen, args.variant, version, repo=source_root)
    report["icon_sizes"] = buildlib.read_ico_sizes(files["icon"])

    if args.reuse_dist:
        _log(f"reusing {dist}")
    elif refreshed is not None:
        report["installers_discarded"] = discard_installers(variant_dir)
        bundled = buildlib.verify_bundled_package(dist, source_root)
        report["bundled_package"] = bundled
        if not bundled["ok"]:
            _save(report)
            raise SystemExit("the app folder does not hold the snapshot's aerolume package "
                             f"(differ: {bundled['differ'][:5]}, missing: {bundled['missing'][:5]})")
        _log(f"bundled aerolume package: {bundled['identical']} files still identical to the snapshot")
        stats = buildlib.dir_stats(dist)
        report.update(dist_bytes=stats["bytes"], dist_files=stats["files"])
    else:
        # Checked before anything is removed: Windows cannot delete a running program, and a
        # rebuild that found out half-way through deleting the folder left it broken.
        running = buildlib.processes_running_from(dist)
        if running:
            raise SystemExit(f"{dist} is in use by " + ", ".join(f"pid {pid} ({Path(exe).name})"
                                                                 for pid, exe in running)
                             + ": close every Aerolume started from this folder, then build again "
                               "(nothing was removed)")
        # The old installer was compiled from the folder about to be deleted: it goes with it,
        # so neither a failed nor an interrupted rebuild can leave it looking like the result.
        report["installers_discarded"] = discard_installers(variant_dir)
        if dist.exists():
            shutil.rmtree(dist)
        log_path = variant_dir / "pyinstaller.log"
        try:
            secs = run_pyinstaller(args.python, variant_dir, args.variant, gen, log_path, source_root)
        except SystemExit:
            # PyInstaller imports every hooked package in throw-away child processes; when the
            # machine is short of commit memory (OneDrive's sync client can hold 20 GB) one of
            # them dies with exit code 101 and the whole analysis aborts. That is transient, so
            # retry once - any other failure is reported as it is.
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            if "SubprocessDiedError" not in text:
                raise
            _log("an analysis child process died (low memory?); retrying PyInstaller once")
            report["pyinstaller_retried"] = True
            shutil.copy2(log_path, log_path.with_suffix(".failed.log"))
            secs = run_pyinstaller(args.python, variant_dir, args.variant, gen, log_path, source_root)
        report["pyinstaller_seconds"] = round(secs, 1)
        _log(f"PyInstaller finished in {report['pyinstaller_seconds']:.0f} s")
        # The snapshot cannot change under PyInstaller, but prove it: the copy must still carry
        # the fingerprint it was taken with, and the package inside the app folder must be the
        # snapshot's byte for byte - not the live tree's, which the editable install also puts
        # on sys.path.
        after = buildlib.source_fingerprint(source_root)
        changed = after["sha256"] != source["sha256"]
        if changed:
            report["source_after_pyinstaller"] = after
            _log(f"WARNING: the snapshot changed during the build ({source['short']} -> {after['short']})")
        bundled = buildlib.verify_bundled_package(dist, source_root)
        report["bundled_package"] = bundled
        _log(f"bundled aerolume package: {bundled['identical']} files identical to the snapshot"
             + ("" if bundled["ok"] else f"; {len(bundled['differ'])} differ, {len(bundled['missing'])} missing"))
        if not bundled["ok"]:
            _save(report)
            raise SystemExit("the app folder does not hold the snapshot's aerolume package "
                             f"(differ: {bundled['differ'][:5]}, missing: {bundled['missing'][:5]})")
        report.update(postprocess(dist, work, args.variant, version, info, source, changed, source_root))
        _log(f"app folder: {buildlib.human(report['dist_bytes'])} in {report['dist_files']} files")
        if not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)

    if args.skip_smoke:
        # Record the skip, so --status calls the folder "untested" rather than current - unless
        # this is --reuse-dist of a folder whose own smoke result is in the previous report.
        if args.reuse_dist and buildlib.smoke_result_for(previous, report.get("build_info") or {}) == "passed":
            report["smoke"] = previous["smoke"]
            report["smoke_seconds"] = previous.get("smoke_seconds")
            report["smoke_reused"] = True      # not measured by this run (run_record skips it)
        else:
            report["smoke"] = {"ok": None, "skipped": True}
    else:
        import smoke_test
        _log("smoke test (launching the built executable headless)")
        t0 = time.perf_counter()
        smoke = smoke_test.smoke(dist, window=True, work=variant_dir / "smoke")
        report["smoke"] = smoke
        report["smoke_seconds"] = round(time.perf_counter() - t0, 1)
        smoke_test._print(smoke)
        if not smoke["ok"]:
            # No installer may outlive a failed folder: not this run's (none was compiled) and
            # not an earlier one compiled from the same folder (--reuse-dist).
            report["installers_discarded"] = (report.get("installers_discarded") or []) + \
                discard_installers(variant_dir)
            _save(report)
            raise SystemExit("smoke test failed: " + buildlib.smoke_failure_summary(smoke)
                             + " - no installer was built (see build-report.json)")

    setup: Path | None = None
    if not args.skip_installer:
        iscc = Path(info["iscc"])
        try:
            setup, secs = compile_installer(iscc, dist, variant_dir / "installer", gen, args.variant, version,
                                            args.compression, args.lzma_threads, variant_dir / "iscc.log",
                                            source_root)
        except SystemExit as exc:
            # The folder is built and smoke-tested; only the installer failed (an LZMA worker that
            # runs short of memory dies with "terminated unexpectedly"). Save what this run measured,
            # so --status still knows the folder and --reuse-dist compiles the installer in minutes
            # and extends this record. Without the save, the report kept describing the previous
            # build, and only a second full build gave a coherent measurement history.
            report["installer_error"] = str(exc.code)
            report["total_seconds"] = round(time.perf_counter() - t_all, 1)
            _save(report)
            raise SystemExit(f"{exc.code}\nThe app folder is built and tested; compile its installer with "
                             "packaging/build.py --reuse-dist (with --lzma-threads 2 if LZMA ran out of "
                             "memory).") from None
        report["installer_seconds"] = round(secs, 1)
        report["installer_compression"] = args.compression
        report["installer"] = buildlib.inspect_installer(setup)
        # The installer packs the app folder, so it carries that folder's fingerprint.
        report["installer"]["source_sha256"] = (report.get("build_info") or {}).get("source_sha256")
        _log(f"installer {setup.name}: {buildlib.human(report['installer']['bytes'])} in {secs:.0f} s, "
             f"{report['installer']['inno_signature']}")
        check = buildlib.verify_installer_manifest(installer_manifest(setup), dist)
        report["installer_manifest"] = check
        _log(f"installer contents: {check['files']} files, "
             + ("identical to the app folder (sizes and SHA-256)" if check["ok"] else "DIFFER from the app folder"))
        if not check["ok"]:
            _save(report)
            raise SystemExit("the installer does not contain exactly the app folder (see build-report.json)")

    report["total_seconds"] = round(time.perf_counter() - t_all, 1)
    if setup is not None:
        report["release_notes"] = str(release_notes.release_notes_path(setup))
    report_path = _save(report)
    if setup is not None:
        notes = release_notes.write_release_notes(report, setup)
        _log(f"release notes for this installer (SHA-256 and timings): {notes}")
    _log(f"done in {report['total_seconds'] / 60:.1f} min; report: {report_path}")
    status = buildlib.build_status(variant_dir, version=buildlib.read_version())
    print_status(status, label="result: ")
    if status["state"] == "untested":
        _log("built, but not smoke-tested (--skip-smoke): --status will not call it current")
        return 3
    if not (status["current"] and (args.skip_installer or status["installer_current"])):
        _log(f"built correctly from snapshot {source['short']} ({source.get('taken') or 'the tree'}), but the "
             "working tree has changed since (listed above); run the build again for a current one")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
