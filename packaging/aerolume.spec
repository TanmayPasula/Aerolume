"""PyInstaller spec for Aerolume: one onedir bundle with a windowed and a console executable.

Run it through ``packaging/build.py`` (which builds outside OneDrive, generates the icon and
version resources, post-processes the folder and compiles the installer). Running
``pyinstaller packaging/aerolume.spec`` directly also works and generates the resources itself.

What goes in, and why:

* ``aerolume`` - every sub-module (some are imported by name at run time) as bytecode *and*
  source (``pyz+py``), so tracebacks in the Script console and crash reports show code lines;
  all package data (icons, fonts, example projects, geometry data) at the same relative paths,
  which is what ``aerolume.app.platform.resource_path`` relies on.
* torch - the contrib hook collects ``torch/lib`` (CUDA runtime, cuBLAS, cuFFT, cuDNN, NVRTC)
  as-is. Nothing is trimmed: kernels load several of these libraries lazily (NVRTC for fused
  element-wise kernels, cuDNN for convolutions), so a "slim" build fails only on some inputs.
* the whole standard library, NumPy and SciPy - the Script console runs user code that can
  import anything a normal Python offers; the self-test (``--selftest``) proves it.
* VTK's OpenGL object factories and the Qt plugins (platforms, image formats, SVG icon engine)
  that are loaded by name and would otherwise be missed by the import analysis.

Excluded: other Qt bindings, Tk, notebooks, test frameworks and test suites. UPX is off - it
corrupts CUDA fat binaries and slows start-up.

macOS: the same spec makes ``dist/Aerolume.app`` when PyInstaller runs on a Mac (it cannot
cross-build, so this has to happen on a Mac; no ``.app`` has been built or tested by the project
yet). There the variant defaults to ``cpu`` (PyTorch for macOS has no CUDA; Apple-GPU support is
not used by the solvers), the Windows version resources are skipped, and a ``BUNDLE`` target wraps
the folder into an app bundle whose icon, ``aerolume.icns``, is generated from the same SVG as the
Windows ``.ico`` (``buildlib.make_icns``), with the bundle identifier, version and ``.aerolume``
document type in its Info.plist. ``packaging/build.py`` (snapshot, smoke test, Inno Setup) is
Windows-only; on a Mac run ``pyinstaller packaging/aerolume.spec`` directly (docs/install.md,
"Building the macOS app"). Signing and notarisation are not covered.
"""
import json
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_DIR = Path(SPECPATH)                       # noqa: F821 - injected by PyInstaller
sys.path.insert(0, str(SPEC_DIR))
import buildlib

REPO = buildlib.REPO
IS_MAC = sys.platform == "darwin"
VARIANT = os.environ.get("AEROLUME_VARIANT", "cpu" if IS_MAC else "cuda")
if VARIANT not in buildlib.VARIANTS:
    raise SystemExit(f"AEROLUME_VARIANT must be one of {buildlib.VARIANTS}, not {VARIANT!r}")
VERSION = buildlib.read_version()
GEN = Path(os.environ.get("AEROLUME_GEN_DIR") or (Path(workpath) / "gen"))   # noqa: F821
generated = buildlib.ensure_generated(GEN, VARIANT, VERSION)

# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------
hiddenimports = []
for pkg in buildlib.COLLECT_ALL_SUBMODULES:
    try:
        hiddenimports += collect_submodules(pkg, filter=buildlib.not_a_test_module)
    except Exception as exc:  # noqa: BLE001 - an optional package (anthropic) may be absent
        print(f"aerolume.spec: skipping {pkg}: {exc}")
hiddenimports += buildlib.stdlib_hiddenimports()
for pkg in buildlib.STDLIB_PACKAGES:
    hiddenimports += collect_submodules(pkg, filter=buildlib.not_a_test_module)
hiddenimports += list(buildlib.EXTRA_HIDDENIMPORTS)
hiddenimports = sorted({h for h in hiddenimports if h.split(".")[0] not in buildlib.STDLIB_EXCLUDES})

excludes = sorted(set(buildlib.STDLIB_EXCLUDES) | set(buildlib.THIRD_PARTY_EXCLUDES))

datas = collect_data_files("aerolume", excludes=["**/__pycache__/**", "**/*.pyc"])
# Distribution metadata, so importlib.metadata.version() works in the frozen app (doctor,
# About, and libraries that check their own or their dependencies' versions).
# Aerolume's own metadata is generated from this tree's pyproject.toml rather than copied from
# the editable install, whose version, requirements and direct_url.json are those of the day it
# was installed (see buildlib.write_dist_info).
_own_meta = buildlib.write_dist_info(GEN / "metadata", REPO)
datas.append((str(_own_meta), _own_meta.name))
for dist_name in buildlib.dependency_closure():
    if dist_name.lower() == "aerolume":
        continue
    try:
        datas += copy_metadata(dist_name)
    except Exception as exc:  # noqa: BLE001 - a dist without metadata files only loses its version
        print(f"aerolume.spec: no metadata for {dist_name}: {exc}")

a = Analysis(                                   # noqa: F821
    [str(SPEC_DIR / "entry.py")],
    pathex=[str(REPO / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,                                 # keep docstrings and asserts: the API help() needs them
    module_collection_mode={"aerolume": "pyz+py"},
)

# Record the bundled top-level modules so build.py can ship the matching licence files.
_tops = sorted({name.split(".")[0] for name, _path, _kind in a.pure}
               | {Path(dest).parts[0] for dest, _src, _kind in a.binaries if len(Path(dest).parts) > 1}
               | {Path(dest).parts[0] for dest, _src, _kind in a.datas if len(Path(dest).parts) > 1})
Path(workpath, "bundled_modules.json").write_text(json.dumps(_tops, indent=0), encoding="utf-8")   # noqa: F821

pyz = PYZ(a.pure)                               # noqa: F821

exe_gui = EXE(                                  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Aerolume",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(generated["icns" if IS_MAC else "icon"]),
    version=None if IS_MAC else str(generated["version_gui"]),
    contents_directory="_internal",
)

exe_cli = EXE(                                  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aerolume-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(generated["icns" if IS_MAC else "icon"]),
    version=None if IS_MAC else str(generated["version_cli"]),
    contents_directory="_internal",
)

coll = COLLECT(                                 # noqa: F821
    exe_gui,
    exe_cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Aerolume",
)

if IS_MAC:
    # dist/Aerolume.app: Finder, the Dock and Launch Services need a bundle, not a folder. The
    # Info.plist gives it the same identity the Windows build has (name, version, the .aerolume
    # project folders as its document type) and lets it use the Retina resolution Qt supports.
    app = BUNDLE(                               # noqa: F821
        coll,
        name="Aerolume.app",
        icon=str(generated["icns"]),
        bundle_identifier="org.aerolume.Aerolume",
        version=VERSION,
        info_plist={
            "CFBundleName": "Aerolume",
            "CFBundleDisplayName": "Aerolume",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Apache License 2.0, Aerolume contributors",
            "LSMinimumSystemVersion": "11.0",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Aerolume project",
                "CFBundleTypeRole": "Editor",
                "LSTypeIsPackage": True,
                "CFBundleTypeExtensions": ["aerolume"],
                "CFBundleTypeIconFile": "aerolume.icns",
            }],
        },
    )
