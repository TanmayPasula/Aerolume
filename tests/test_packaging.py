"""Desktop integration (app/platform.py) and the packaging helpers (packaging/*), without a build.

The single-instance, mutex and crash-trace tests run the "other" process as a real subprocess
(a second launch is always a second process), so they exercise the same named pipe, lock file
and kernel objects the installed app uses. The full build is covered by
``tests/test_packaging_build.py`` (slow, needs a built folder).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from aerolume import __version__
from aerolume.app import platform as plat

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"
PY = sys.executable
on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only integration")


def _load(name: str):
    # Reuse a module another test file already loaded: build.py binds the buildlib it imported
    # first, so replacing sys.modules["buildlib"] here made the release-notes test's monkeypatches
    # miss build.py whenever the freshness tests had run before this file (order-dependent).
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


buildlib = _load("buildlib")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    d = tmp_path / "config"
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(d))
    return d


def _child_env(config_dir: Path) -> dict[str, str]:
    env = dict(os.environ, AEROLUME_CONFIG_DIR=str(config_dir), QT_QPA_PLATFORM="offscreen",
               PYTHONIOENCODING="utf-8")
    return env


def _run_child(code: str, config_dir: Path, cwd: Path | None = None, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-c", textwrap.dedent(code)], env=_child_env(config_dir), cwd=cwd,
                          capture_output=True, text=True, timeout=timeout, check=False)


# ---------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------

def test_source_layout_paths():
    assert plat.package_dir() == REPO / "src" / "aerolume"
    assert plat.app_root() == REPO
    assert plat.resource_path("aerolume.svg").is_file()
    assert (plat.docs_dir() / "user_guide.md").is_file()
    assert plat.app_file("LICENSE") == REPO / "LICENSE"
    assert plat.app_file("no-such-file.txt") is None
    assert any((p / "project.json").is_file() for p in plat.examples_dir().iterdir())
    assert not plat.is_frozen()


def test_frozen_layout_paths(tmp_path, monkeypatch):
    install = tmp_path / "Aerolume"
    (install / "_internal").mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(install / "_internal"), raising=False)
    monkeypatch.setattr(sys, "executable", str(install / "Aerolume.exe"))
    assert plat.is_frozen()
    assert plat.app_root() == install.resolve()
    assert plat.bundle_dir() == install / "_internal"
    assert plat.docs_dir() == install.resolve() / "docs"


def test_resolve_open_target(tmp_path):
    proj = tmp_path / "Wing study.aerolume"
    proj.mkdir()
    (proj / "project.json").write_text("{}", encoding="utf-8")
    (proj / plat.PROJECT_MARKER).write_text("marker", encoding="utf-8")
    assert plat.resolve_open_target(proj) == proj
    assert plat.resolve_open_target(proj / plat.PROJECT_MARKER) == proj
    assert plat.resolve_open_target(proj / "project.json") == proj
    assert plat.resolve_open_target("Wing study.aerolume", cwd=tmp_path) == proj
    assert plat.resolve_open_target(None) is None and plat.resolve_open_target("  ") is None
    # a lone marker whose folder lost its index still means "that folder" (the store reports it)
    lone = tmp_path / "broken.aerolume"
    lone.mkdir()
    (lone / plat.PROJECT_MARKER).write_text("", encoding="utf-8")
    assert plat.resolve_open_target(lone / plat.PROJECT_MARKER) == lone
    # an unrelated .aerolume file outside a project is returned unchanged
    stray = tmp_path / "notes.aerolume"
    stray.write_text("x", encoding="utf-8")
    assert plat.resolve_open_target(stray) == stray
    missing = plat.resolve_open_target("missing.aerolume", cwd=tmp_path)
    assert missing == tmp_path / "missing.aerolume"


def test_project_marker_is_created_once_and_only_in_projects(tmp_path):
    proj = tmp_path / "Duct.aerolume"
    proj.mkdir()
    assert plat.ensure_project_marker(proj) is None  # no project.json: not a project
    (proj / "project.json").write_text("{}", encoding="utf-8")
    marker = plat.ensure_project_marker(proj)
    assert marker == proj / plat.PROJECT_MARKER and marker.is_file()
    assert "Aerolume project marker" in marker.read_text(encoding="utf-8")
    marker.write_text("user edited", encoding="utf-8")
    assert plat.ensure_project_marker(proj) == marker
    assert marker.read_text(encoding="utf-8") == "user edited"  # never overwritten
    # the marker round-trips through the double-click path
    assert plat.resolve_open_target(marker) == proj
    assert plat.ensure_project_marker(tmp_path / "gone.aerolume") is None


def test_recent_document_is_only_registered_by_the_frozen_app(tmp_path, monkeypatch):
    """From source the call must not touch the user's Recent list; frozen it goes to the shell."""
    proj = tmp_path / "Fan.aerolume"
    proj.mkdir()
    (proj / "project.json").write_text("{}", encoding="utf-8")
    calls: list[tuple] = []

    class _Fn:
        argtypes = restype = None

        def __call__(self, *args):
            calls.append(args)

    import ctypes

    class _Shell:
        SHAddToRecentDocs = _Fn()

    class _Windll:
        shell32 = _Shell()

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert plat.note_project_opened(proj) == proj / plat.PROJECT_MARKER
    assert calls == []  # not frozen: no shell call
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert plat.add_recent_document(proj / plat.PROJECT_MARKER)
    assert calls == [(3, str((proj / plat.PROJECT_MARKER).resolve()))]
    assert not plat.add_recent_document(proj / "missing.aerolume")
    assert plat.note_project_opened(tmp_path / "not-a-project") is None


def test_constants_agree_with_installer_and_build_helpers():
    iss = (PACKAGING / "aerolume.iss").read_text(encoding="utf-8")
    assert plat.APP_USER_MODEL_ID == buildlib.APP_USER_MODEL_ID
    assert plat.APP_MUTEX_NAME == buildlib.APP_MUTEX_NAME
    assert f'#define AppUserModelId "{plat.APP_USER_MODEL_ID}"' in iss
    assert f'#define AppMutexName "{plat.APP_MUTEX_NAME}"' in iss
    assert f'#define AppId "{buildlib.INNO_APP_ID}"' in iss
    from aerolume.app import main as app_main
    assert app_main.APP_USER_MODEL_ID == plat.APP_USER_MODEL_ID


# ---------------------------------------------------------------------------
# Windows identity
# ---------------------------------------------------------------------------

@on_windows
def test_app_user_model_id_is_set(tmp_path):
    r = _run_child("""
        from aerolume.app import platform as p
        print(p.set_app_user_model_id(), p.current_app_user_model_id())
    """, tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["True", plat.APP_USER_MODEL_ID]


@on_windows
def test_app_mutex_lives_exactly_as_long_as_the_process(tmp_path):
    name = f"Aerolume.Test.{os.getpid()}.{time.time_ns()}"
    child = subprocess.Popen([PY, "-c", textwrap.dedent(f"""
        import sys
        from aerolume.app import platform as p
        print(p.hold_app_mutex({name!r}), flush=True)
        sys.stdin.readline()
    """)], env=_child_env(tmp_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "True"
        assert plat.mutex_exists(name)
    finally:
        child.communicate("\n", timeout=30)
    assert not plat.mutex_exists(name)


# ---------------------------------------------------------------------------
# single instance
# ---------------------------------------------------------------------------

_SECOND_LAUNCH = """
    import sys
    from PySide6.QtCore import QCoreApplication
    from aerolume.app.platform import SingleInstance
    app = QCoreApplication([])
    guard = SingleInstance()
    carry_on = guard.start(sys.argv[1:], timeout_ms=8000)
    print("continue" if carry_on else "forwarded", guard.is_primary, flush=True)
"""


def _wait(app, proc: subprocess.Popen, timeout: float = 60.0) -> tuple[str, str]:
    t0 = time.monotonic()
    while proc.poll() is None and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    if proc.poll() is None:
        proc.kill()
    out, err = proc.communicate(timeout=10)
    for _ in range(20):
        app.processEvents()
    return out, err


def test_second_launch_forwards_arguments(qapp, config, tmp_path):
    guard = plat.SingleInstance()
    received: list[list[str]] = []
    guard.message_received.connect(received.append)
    try:
        assert guard.start(["unused"]) is True and guard.is_primary
        work = tmp_path / "elsewhere"
        work.mkdir()
        (work / "study.aerolume").mkdir()
        proc = subprocess.Popen([PY, "-c", textwrap.dedent(_SECOND_LAUNCH), "study.aerolume", "--theme", "dark"],
                                env=_child_env(config), cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        out, err = _wait(qapp, proc)
        assert proc.returncode == 0, err
        assert out.split() == ["forwarded", "False"]
        assert received == [[str((work / "study.aerolume").resolve()), "--theme", "dark"]]
    finally:
        guard.close()


def test_lock_held_but_silent_primary_lets_the_new_process_run(qapp, config):
    from PySide6.QtCore import QLockFile
    guard = plat.SingleInstance()
    lock = QLockFile(str(guard._lock_path()))
    lock.setStaleLockTime(0)
    assert lock.tryLock(0)                       # an instance that holds the lock but never listens
    try:
        other = plat.SingleInstance()
        t0 = time.monotonic()
        assert other.start(["x"], timeout_ms=600) is True
        assert not other.is_primary and not other.forwarded
        assert time.monotonic() - t0 < 5
    finally:
        lock.unlock()


def test_crashed_primary_does_not_block_the_next_start(qapp, config):
    holder = subprocess.Popen([PY, "-c", textwrap.dedent("""
        import sys
        from PySide6.QtCore import QCoreApplication
        from aerolume.app.platform import SingleInstance
        app = QCoreApplication([])
        g = SingleInstance()
        print(g.try_primary(), flush=True)
        sys.stdin.readline()
    """)], env=_child_env(config), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "True"
    assert plat.SingleInstance().try_primary() is False     # held by a live process
    holder.kill()                                            # a crash: no clean unlock
    holder.wait(timeout=30)
    guard = plat.SingleInstance()
    try:
        assert guard.try_primary() is True                   # stale lock (dead PID) is taken over
    finally:
        guard.close()


def test_instance_key_depends_on_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "a"))
    a = plat._instance_key()
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(tmp_path / "b"))
    assert plat._instance_key() != a and a.startswith("aerolume-")


# ---------------------------------------------------------------------------
# crash reports
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_hooks():
    saved = (sys.excepthook, threading.excepthook, dict(plat._crash_state))
    yield
    sys.excepthook, threading.excepthook = saved[0], saved[1]
    plat._crash_state.clear()
    plat._crash_state.update(saved[2])


def _raise_in(path_text: str):
    try:
        raise ValueError(f"cannot read {path_text}")
    except ValueError:
        return sys.exc_info()


def test_crash_report_is_complete_and_redacted(config):
    home = str(Path.home())
    exc_info = _raise_in(os.path.join(home, "Documents", "wing.aerolume"))
    report = plat.write_crash_report(*exc_info, context="While opening a project.")
    assert report.path is not None and report.path.parent == config / "logs"
    saved = report.path.read_text(encoding="utf-8")
    for text in (report.text, saved):
        assert "ValueError: cannot read" in text
        assert f"Aerolume {__version__}" in text and "Python " in text and "Traceback:" in text
        assert "While opening a project." in text
        assert home.lower() not in text.lower()
        assert "~" in text
    assert report.summary.startswith("ValueError: cannot read ~")
    assert "This report was saved to" in report.text


def test_crash_logs_are_pruned(config):
    for _ in range(plat.MAX_CRASH_LOGS + 4):
        plat.write_crash_report(*_raise_in("x"))
    assert len(list((config / "logs").glob("crash-*.log"))) == plat.MAX_CRASH_LOGS


def test_crash_reports_in_the_same_millisecond_keep_separate_files(config, monkeypatch):
    """The log name has millisecond resolution; reports that share a stamp get a counter instead
    of overwriting each other (the pruning test above once kept 19 of 20 on a busy machine)."""
    import datetime as dt
    fixed = dt.datetime(2026, 11, 1, 12, 0, 0, 123456, tzinfo=dt.UTC)
    monkeypatch.setattr(plat, "_local_now", lambda: fixed)
    paths = [plat.write_crash_report(*_raise_in(f"file {i}")).path for i in range(3)]
    assert [p.name for p in paths] == ["crash-20261101-120000-123.log", "crash-20261101-120000-123-2.log",
                                       "crash-20261101-120000-123-3.log"]
    assert [f"file {i}" in p.read_text(encoding="utf-8") for i, p in enumerate(paths)] == [True] * 3


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_crash_handler_routes_main_and_worker_thread_exceptions(qapp, config, restore_hooks):
    seen: list[tuple[str, str]] = []
    plat._crash_state.update(callback=None, relay=None, count=0, installed=False)
    plat.install_crash_handler(lambda r: seen.append((r.summary, threading.current_thread().name)))
    hook = sys.excepthook
    plat.install_crash_handler(lambda r: seen.append((r.summary, threading.current_thread().name)))
    assert sys.excepthook is hook                               # re-installing does not stack hooks
    sys.excepthook(*_raise_in("main"))
    t = threading.Thread(target=lambda: 1 / 0, name="solver-worker")
    t.start()
    t.join()
    for _ in range(50):
        qapp.processEvents()
        if len(seen) >= 2:
            break
        time.sleep(0.01)
    assert seen[0] == ("ValueError: cannot read main", "MainThread")
    assert seen[1] == ("ZeroDivisionError: division by zero", "MainThread")   # delivered on the GUI thread
    logs = list((config / "logs").glob("crash-*.log"))
    assert len(logs) == 2
    assert any("Thread: solver-worker" in p.read_text(encoding="utf-8") for p in logs)


def test_environment_lines_never_import_heavy_modules():
    code = ("import sys; from aerolume.app import platform as p; p.environment_lines(); "
            "print('torch' in sys.modules, 'vtkmodules' in sys.modules)")
    r = subprocess.run([PY, "-c", code],
                       capture_output=True, text=True, timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False"]


def test_windowed_streams_go_to_the_log_and_native_crash_is_reported(tmp_path):
    cfg = tmp_path / "cfg"
    crash = _run_child("""
        import sys
        sys.stdout = None
        sys.stderr = None
        from aerolume.app import platform as p
        p.prepare_runtime()
        print("hello from a windowed build")
        sys.stdout.flush()
        import faulthandler
        faulthandler._sigsegv()
    """, cfg)
    assert crash.returncode != 0
    logs = cfg / "logs"
    assert "hello from a windowed build" in (logs / plat.APP_LOG).read_text(encoding="utf-8")
    assert "Fatal Python error" in (logs / plat.NATIVE_CRASH_LOG).read_text(encoding="utf-8")
    after = _run_child("""
        from aerolume.app import platform as p
        p.prepare_runtime()
        text = p.previous_native_crash()
        print("found" if text and "Segmentation fault" in text else repr(text))
    """, cfg)
    assert after.returncode == 0, after.stderr
    assert after.stdout.strip() == "found"
    assert list(logs.glob("native-crash-*.log"))                # archived, not lost
    assert not (logs / plat.NATIVE_CRASH_LOG).exists()          # the clean exit removed the live trace
    clean = _run_child("from aerolume.app import platform as p; p.prepare_runtime(); "
                       "print(p.previous_native_crash())", cfg)
    assert clean.stdout.strip() == "None"


# ---------------------------------------------------------------------------
# self-test (source mode; the frozen run is in test_packaging_build.py)
# ---------------------------------------------------------------------------

@pytest.mark.gui
def test_self_test_from_source_without_gpu_or_window(qapp):
    result = plat.self_test(window=False, gpu=False)
    bad = [c for c in result["checks"] if not c["ok"]]
    assert result["ok"], bad
    names = {c["name"] for c in result["checks"]}
    assert {"resources", "app files", "stdlib for scripts", "qt", "airfoil solver", "vtk"} <= names
    json.dumps(result)                                          # the CLI writes it as JSON


@pytest.mark.slow
def test_self_test_3d_check_exercises_placed_physics():
    """The 3D check runs a heated body, a fan and both emitter species; on the device the
    frozen build will use. Fused-kernel fallbacks are only fatal when frozen."""
    pytest.importorskip("torch")
    detail = plat._check_ns3d()
    assert "heated body + fan + smoke + particles" in detail
    assert "fan thrust 0.1000" in detail
    assert " 0 tracers" not in detail


def test_self_test_3d_check_fails_a_frozen_build_without_the_gpu_fast_path(monkeypatch):
    """A frozen CUDA build whose fused kernels fell back must fail the self-test: that is
    what a missing NVRTC builtins DLL looks like, and from outside it is only a slow run."""
    pytest.importorskip("torch")
    import torch

    from aerolume.solvers import ns3d

    class FakeSolver:
        fused_error = "nvrtc: cannot open nvrtc-builtins64_130.dll"
        theta_error = dye_error = graph_error = None
        uses_fused_kernels = False
        uses_cuda_graphs = True

        def __init__(self, cfg):
            self.cfg = cfg

        def advance(self, n):
            pass

        def field(self, name):
            return torch.zeros(2, 2, 2)

        def heat_balance(self):
            return {"heat_in_bodies": 1.0}

        def fan_report(self):
            return [{"thrust": 0.1}]

        def particle_count(self):
            return 5

    monkeypatch.setattr(ns3d, "FlowSolver3D", FakeSolver)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="nvrtc-builtins"):
        plat._check_ns3d()
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    assert "fallbacks: fused_error" in plat._check_ns3d()


def test_entry_script_picks_its_role(monkeypatch):
    entry = _load("entry")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\Aerolume\aerolume-cli.exe")
    assert entry._role() == "cli"
    monkeypatch.setattr(sys, "executable", r"C:\Apps\Aerolume\Aerolume.exe")
    monkeypatch.setattr(sys, "argv", ["Aerolume.exe", "--selftest"])
    assert entry._role() == "selftest"
    monkeypatch.setattr(sys, "argv", ["Aerolume.exe", r"D:\p\x.aerolume\project.aerolume"])
    assert entry._role() == "gui"


# ---------------------------------------------------------------------------
# build helpers
# ---------------------------------------------------------------------------

def test_version_comes_from_the_package():
    assert buildlib.read_version() == __version__
    assert buildlib.version_quad("0.2.1") == (0, 2, 1, 0)
    assert buildlib.version_quad("1.0rc2") == (1, 0, 2, 0)
    assert buildlib.version_quad("3") == (3, 0, 0, 0)


def test_version_resource_is_valid_for_pyinstaller(tmp_path):
    versioninfo = pytest.importorskip("PyInstaller.utils.win32.versioninfo")
    f = tmp_path / "v.txt"
    f.write_text(buildlib.version_resource("0.2.0", "cuda", exe_name="Aerolume.exe", description="Aerolume"),
                 encoding="utf-8")
    vi = versioninfo.load_version_info_from_text_file(str(f))
    text = str(vi)
    assert "filevers=(0, 2, 0, 0)" in text and "'ProductVersion', '0.2.0 (CUDA)'" in text
    assert "'OriginalFilename', 'Aerolume.exe'" in text


def test_icon_has_every_size_and_loads(qapp, tmp_path):
    from PySide6.QtGui import QImageReader
    ico = buildlib.make_ico(plat.resource_path("aerolume.svg"), tmp_path / "a.ico")
    assert buildlib.read_ico_sizes(ico) == list(buildlib.ICON_SIZES)
    reader = QImageReader(str(ico))
    assert reader.imageCount() == len(buildlib.ICON_SIZES)
    for i in range(reader.imageCount()):
        reader.jumpToImage(i)
        img = reader.read()
        assert not img.isNull()
        corner, centre = img.pixelColor(0, 0), img.pixelColor(img.width() // 2, img.height() // 2)
        assert corner.alpha() == 0 and centre.alpha() == 255   # transparent rounded corner, opaque mark


def test_generated_files_are_cached_by_content(qapp, tmp_path):
    a = buildlib.ensure_generated(tmp_path, "cuda", "0.2.0")
    mtime = a["icon"].stat().st_mtime_ns
    b = buildlib.ensure_generated(tmp_path, "cuda", "0.2.0")
    assert b["icon"].stat().st_mtime_ns == mtime
    buildlib.ensure_generated(tmp_path, "cpu", "0.2.0")
    assert "0.2.0 (CPU)" in a["version_gui"].read_text(encoding="utf-8")


def test_icns_for_the_mac_bundle_is_a_png_icon_family(qapp, tmp_path):
    """The .icns the spec's macOS BUNDLE uses: every iconutil size, PNG data, rendered from the SVG."""
    from PySide6.QtGui import QImage
    icns = buildlib.make_icns(plat.resource_path("aerolume.svg"), tmp_path / "a.icns")
    assert buildlib.read_icns_entries(icns) == [(t, s) for t, s in buildlib.ICNS_ENTRIES]
    data = icns.read_bytes()
    first_png = data[16:16 + int.from_bytes(data[12:16], "big") - 8]
    img = QImage.fromData(first_png, "PNG")
    assert img.width() == 16 and img.pixelColor(8, 8).alpha() == 255
    assert buildlib.ensure_generated(tmp_path / "gen", "cpu", "0.2.0")["icns"].is_file()


def test_spec_builds_a_mac_app_bundle_only_on_macos():
    """The spec describes the .app (BUNDLE, .icns, Info.plist) behind a darwin guard, so the
    Windows build is unchanged; nothing claims a macOS build was made (docs/install.md)."""
    spec = (PACKAGING / "aerolume.spec").read_text(encoding="utf-8")
    assert 'IS_MAC = sys.platform == "darwin"' in spec
    assert "if IS_MAC:" in spec and "BUNDLE(" in spec and 'name="Aerolume.app"' in spec
    assert 'icon=str(generated["icns"])' in spec and "bundle_identifier=" in spec
    assert '"cpu" if IS_MAC else "cuda"' in spec
    install = (REPO / "docs" / "install.md").read_text(encoding="utf-8")
    assert "Building the macOS app" in install and "pyinstaller packaging/aerolume.spec" in install
    assert "no `.app` has been built" in install


def test_spec_and_scripts_parse():
    for name in ("aerolume.spec", "build.py", "smoke_test.py", "entry.py", "buildlib.py"):
        ast.parse((PACKAGING / name).read_text(encoding="utf-8"), filename=name)


def test_installer_script_directives():
    iss = (PACKAGING / "aerolume.iss").read_text(encoding="utf-8")
    for directive in ("PrivilegesRequired=lowest", "ChangesAssociations=yes", "AppMutex={#AppMutexName}",
                      "ArchitecturesInstallIn64BitMode=x64compatible", "UninstallDisplayIcon={app}\\{#AppExe}",
                      "LicenseFile={#SourceDir}\\LICENSE", "VersionInfoVersion={#VersionQuad}",
                      'Type: filesandordirs; Name: "{app}\\_internal"',
                      'Software\\Classes\\.aerolume', 'Name: "desktopicon"', "Flags: unchecked",
                      "OutputManifestFile=Aerolume-{#AppVersion}-{#Variant}-manifest.txt"):
        assert directive in iss, directive
    # user data is never removed by the uninstaller
    uninstall = iss.split("[UninstallDelete]", 1)[1].split("[", 1)[0]
    assert "{userappdata}" not in uninstall and "{localappdata}" not in uninstall


def test_stdlib_bundle_list():
    names = set(buildlib.stdlib_hiddenimports())
    assert {"csv", "sqlite3", "statistics", "json", "ctypes", "asyncio"} <= names
    assert not names & {"tkinter", "idlelib", "test", "turtle", "lib2to3"}
    assert all(m.split(".")[0] in sys.stdlib_module_names for m in plat.SCRIPT_STDLIB)
    assert buildlib.not_a_test_module("scipy.optimize") and not buildlib.not_a_test_module("scipy.optimize.tests")


def test_copy_app_files(tmp_path):
    copied = buildlib.copy_app_files(tmp_path)
    assert (tmp_path / "LICENSE").is_file() and (tmp_path / "THIRD_PARTY_NOTICES.md").is_file()
    assert (tmp_path / "docs" / "user_guide.md").is_file()
    assert (tmp_path / "validation" / "ns2d_cases.py").is_file()
    assert "docs/install.md" in copied
    assert not list(tmp_path.rglob("__pycache__"))


#: SHA-256 of the Free Software Foundation's gpl-3.0.txt and lgpl-3.0.txt (packaging/licenses/README.md)
FSF_TEXTS = {"GPL-3.0.txt": "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
             "LGPL-3.0.txt": "da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768"}


def test_vendored_licence_texts_are_the_fsf_originals_and_build_inputs():
    import hashlib
    for name, digest in FSF_TEXTS.items():
        assert hashlib.sha256((buildlib.VENDORED_LICENSES / name).read_bytes()).hexdigest() == digest, name
    inputs = {rel for rel, _ in buildlib.source_files()}
    for name in (*FSF_TEXTS, "qt-for-python.txt", "python-runtime.txt", "nvidia-cuda.txt", "intel-openmp.txt"):
        assert f"packaging/licenses/{name}" in inputs, name     # editing one makes the build stale


#: SHA-256 of packaging/licenses/UPSTREAM.sha256, the list of the vendored VTK/Qt/Mesa licence files
#: with their own SHA-256 values: changing a vendored text needs both lists changed on purpose.
UPSTREAM_SUMS_SHA256 = "e94da874f7caae886186384d8573d90d425b56a271e1b810d2b4f4ae1416125d"


def test_vendored_vtk_and_qt_licences_are_pinned_and_complete():
    """Every upstream licence text of VTK's and Qt's third-party code is byte-for-byte what the pinned
    list says, every file of those folders is listed, and every file a component names exists."""
    import hashlib
    import json
    sums = buildlib.UPSTREAM_LICENSE_SUMS
    assert hashlib.sha256(sums.read_bytes()).hexdigest() == UPSTREAM_SUMS_SHA256
    pinned = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        pinned[rel] = digest
    vend = buildlib.VENDORED_LICENSES
    on_disk = {p.relative_to(vend).as_posix() for sub in ("vtk", "qt") for p in (vend / sub).rglob("*") if p.is_file()}
    assert on_disk == set(pinned)
    for rel, digest in pinned.items():
        assert hashlib.sha256((vend / rel).read_bytes()).hexdigest() == digest, rel
    inputs = {rel for rel, _ in buildlib.source_files()}
    assert {f"packaging/licenses/{rel}" for rel in pinned} <= inputs           # editing one makes the build stale
    for sub in ("vtk", "qt"):
        manifest = json.loads((vend / sub / "components.json").read_text(encoding="utf-8"))
        for comp in manifest["components"]:
            assert comp["files"] and comp["license"], comp["name"]
            for f in comp["files"]:
                assert (vend / sub / f).is_file(), (sub, f)
    ids = {c["id"] for c in json.loads((vend / "qt" / "components.json").read_text(encoding="utf-8"))["components"]}
    assert ids >= {"harfbuzz-ng", "freetype", "libpng", "libjpeg", "md4c", "pcre2", "zlib", "libtiff", "libwebp",
                   "mesa-llvmpipe"}


def test_vtk_third_party_dll_names():
    """VTK's wheel holds only VTK's own licence; the libraries it vendors as DLLs are told apart from its
    CamelCase modules by name (the build refuses one without an entry, see the test below)."""
    assert buildlib.VTK_THIRD_PARTY_RE.match("vtkzlib-9.7.1.dll").group(1) == "vtkzlib"
    assert buildlib.VTK_THIRD_PARTY_RE.match("vtkDICOMParser-9.7.1.dll")
    assert not buildlib.VTK_THIRD_PARTY_RE.match("vtkCommonCore-9.7.1.dll")
    assert not buildlib.VTK_THIRD_PARTY_RE.match("vtkWrappingPythonCore3.11-9.7.1.dll")


def test_licences_cover_vtk_and_qt_third_party_code(tmp_path):
    app = tmp_path / "Aerolume"
    for rel in ("_internal/vtk.libs/vtkzlib-9.7.1.dll", "_internal/vtk.libs/vtkfreetype-9.7.1.dll",
                "_internal/vtk.libs/vtkCommonCore-9.7.1.dll", "_internal/PySide6/opengl32sw.dll"):
        (app / rel).parent.mkdir(parents=True, exist_ok=True)
        (app / rel).write_bytes(b"")
    rows = buildlib.collect_licenses(app, ["vtkmodules", "PySide6", "shiboken6"])
    lic = app / "licenses"
    vend = buildlib.VENDORED_LICENSES
    vtk_index = (lic / "vtk" / "third_party" / "README.txt").read_text(encoding="utf-8")
    assert "_internal/vtk.libs/vtkzlib-9.7.1.dll" in vtk_index and "The FreeType Project" in vtk_index
    assert "vtkCommonCore" not in vtk_index and "lz4" not in vtk_index.lower()      # only what is bundled
    assert (lic / "vtk" / "third_party" / "zlib" / "LICENSE").read_bytes() == (vend / "vtk" / "zlib" / "LICENSE").read_bytes()
    qt_index = (lic / "PySide6_Essentials" / "third_party" / "README.txt").read_text(encoding="utf-8")
    for needle in ("HarfBuzz", "MD4C", "LibPNG", "Mesa", "Copyright"):
        assert needle in qt_index, needle
    assert (lic / "PySide6_Essentials" / "third_party" / "qtbase" / "src" / "3rdparty" / "harfbuzz-ng" / "COPYING").is_file()
    assert (lic / "PySide6_Essentials" / "third_party" / "mesa-llvmpipe" / "llvm-LICENSE.TXT").is_file()
    index = (lic / "README.txt").read_text(encoding="utf-8")
    assert "vtk/third_party/" in index and "PySide6_Essentials/third_party/" in index
    names = {r.name for r in rows}
    assert "VTK third-party libraries (2)" in names and any(n.startswith("Qt third-party code") for n in names)
    (app / "_internal" / "vtk.libs" / "vtknewlib-9.7.1.dll").write_bytes(b"")
    with pytest.raises(RuntimeError, match="vtknewlib"):
        buildlib.collect_vtk_third_party(app)


def test_nested_licence_files_keep_their_folders():
    """PyTorch ships 95 licence files as .dist-info/licenses/third_party/<project>/LICENSE: flattened to
    their names they overwrote each other and only two survived (release gate, 1 November 2026)."""
    sub = buildlib._licence_subpath
    assert sub("torch-2.14.1.dist-info/licenses/third_party/cutlass/LICENSE.txt") == "third_party/cutlass/LICENSE.txt"
    assert sub("torch-2.14.1.dist-info/licenses/LICENSE") == "LICENSE"
    assert sub("six-1.17.0.dist-info/LICENSE") == "LICENSE"
    assert sub("numpy/LICENSE.txt") == "LICENSE.txt"
    assert buildlib.NVIDIA_DLL_RE.match("cudnn_engines_precompiled64_9.dll")
    assert buildlib.NVIDIA_DLL_RE.match("nvrtc-builtins64_130.dll") and buildlib.NVIDIA_DLL_RE.match("nvToolsExt64_1.dll")
    assert not buildlib.NVIDIA_DLL_RE.match("caffe2_nvrtc.dll") and not buildlib.NVIDIA_DLL_RE.match("torch_cuda.dll")
    assert buildlib.INTEL_OPENMP_RE.match("libiomp5md.dll") and buildlib.INTEL_OPENMP_RE.match("libiompstubs5md.dll")


def _fake_app(root: Path, *, gpu: bool) -> Path:
    """A miniature app folder holding real binaries of this environment where the notices read them."""
    import importlib.util as iu
    torch_lib = Path(iu.find_spec("torch").origin).parent / "lib"
    qt_dir = Path(iu.find_spec("PySide6").origin).parent
    lib = root / "_internal" / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "torch_cpu.dll").write_bytes(b"")    # PyTorch's own (250 MB): only its name matters here
    names = ["libiomp5md.dll"] + (["cudart64_13.dll", "caffe2_nvrtc.dll", "nvToolsExt64_1.dll"] if gpu else [])
    for name in names:
        src = torch_lib / name
        if src.is_file():
            shutil.copy2(src, lib / name)
        else:                                   # another torch build: an empty stand-in keeps the name
            (lib / name).write_bytes(b"")
    (root / "_internal" / "PySide6").mkdir(parents=True)
    if (qt_dir / "Qt6Core.dll").is_file():
        shutil.copy2(qt_dir / "Qt6Core.dll", root / "_internal" / "PySide6" / "Qt6Core.dll")
    return root


def test_licences_cover_qt_python_nvidia_and_intel(tmp_path):
    """The frozen app ships the licence or notice of everything it contains, not only of the PyPI
    distributions: the LGPL/GPL texts for Qt, CPython's licences, and notices for NVIDIA's and Intel's
    proprietary runtime libraries (release gate, 1 November 2026: all three were missing)."""
    pytest.importorskip("torch")
    from importlib import metadata
    app = _fake_app(tmp_path / "Aerolume", gpu=True)
    rows = buildlib.collect_licenses(app, ["torch", "PySide6", "shiboken6", "numpy"])
    lic = app / "licenses"
    vend = buildlib.VENDORED_LICENSES
    for dist in ("PySide6_Essentials", "shiboken6"):
        for text in FSF_TEXTS:
            assert (lic / dist / text).read_bytes() == (vend / text).read_bytes(), (dist, text)
        readme = (lic / dist / "README.txt").read_text(encoding="utf-8")
        assert metadata.version("PySide6_Essentials") in readme and "download.qt.io" in readme
        assert "{" not in readme                                    # every placeholder filled
    base_licence = Path(sys.base_prefix) / "LICENSE.txt"
    if base_licence.is_file():
        assert (lic / "python" / "LICENSE.txt").read_bytes() == base_licence.read_bytes()
        assert "OpenSSL" in (lic / "python" / "README.txt").read_text(encoding="utf-8")
    nvidia = (lic / "nvidia-cuda" / "NOTICE.txt").read_text(encoding="utf-8")
    assert "_internal/torch/lib/cudart64_13.dll" in nvidia and "_internal/torch/lib/nvToolsExt64_1.dll" in nvidia
    assert "caffe2_nvrtc.dll" not in nvidia and "torch_cpu.dll" not in nvidia
    assert "https://docs.nvidia.com/cuda/eula/index.html" in nvidia and "cuDNN Software License Agreement" in nvidia
    intel = (lic / "intel-openmp" / "NOTICE.txt").read_text(encoding="utf-8")
    assert "_internal/torch/lib/libiomp5md.dll" in intel and "Intel" in intel
    if sys.platform == "win32" and (app / "_internal" / "torch" / "lib" / "cudart64_13.dll").stat().st_size:
        assert "NVIDIA CUDA Runtime" in nvidia                      # read from the file's version resource
    shipped = {f.as_posix() for f in metadata.distribution("torch").files or []}
    if any(s.endswith("licenses/third_party/cutlass/LICENSE.txt") for s in shipped):
        assert (lic / "torch" / "third_party" / "cutlass" / "LICENSE.txt").is_file()
    index = (lic / "README.txt").read_text(encoding="utf-8")
    for needle in ("python/", "nvidia-cuda/", "intel-openmp/", "LGPL-3.0", "PySide6_Essentials", "torch"):
        assert needle in index, needle
    assert {r.name.split(" (")[0] for r in rows} >= {"Python", "NVIDIA CUDA/cuDNN", "Intel OpenMP runtime", "torch"}


def test_a_cpu_build_lists_no_nvidia_notice(tmp_path):
    pytest.importorskip("torch")
    app = _fake_app(tmp_path / "Aerolume", gpu=False)
    buildlib.collect_licenses(app, ["PySide6", "shiboken6"])
    assert not (app / "licenses" / "nvidia-cuda").exists()
    assert "nvidia-cuda/" not in (app / "licenses" / "README.txt").read_text(encoding="utf-8")


def test_onedrive_detection(monkeypatch, tmp_path):
    root = tmp_path / "OneDrive-root"
    (root / "Desktop").mkdir(parents=True)
    monkeypatch.setenv("OneDrive", str(root))
    assert buildlib.is_under_onedrive(root / "Desktop")
    assert not buildlib.is_under_onedrive(tmp_path)
    assert buildlib.is_under_onedrive(tmp_path / "OneDrive - Contoso" / "x")
    assert not buildlib.is_under_onedrive(buildlib.default_build_root()) or "OneDrive" in str(buildlib.default_build_root())


def test_human_sizes():
    assert buildlib.human(999) == "999 B"
    assert buildlib.human(1_500_000) == "1.50 MB"
    assert buildlib.human(4.2e9) == "4.20 GB"


def test_bundled_metadata_follows_pyproject_not_the_stale_install():
    reqs = " ".join(buildlib.project_requirements())
    assert "vtk" in reqs and "anthropic" in reqs and "torch" in reqs
    closure = {n.lower().replace("-", "_") for n in buildlib.dependency_closure()}
    assert {"aerolume", "torch", "vtk", "numpy", "scipy", "pyside6_essentials", "shiboken6"} <= closure
    assert "pytest" not in closure and "ruff" not in closure          # dev extra not bundled


def test_stale_own_metadata_is_pruned_from_the_app(tmp_path):
    """PyInstaller copies the editable install's old dist-info next to the generated one;
    the build keeps only the generated one (a development build otherwise reported the version
    the editable install was made with)."""
    internal = tmp_path / "_internal"
    for name in ("aerolume-0.0.9.dist-info", "aerolume-0.1.0.dist-info", "aerolume_extra-1.0.dist-info",
                 "torch-2.14.1.dist-info"):
        (internal / name).mkdir(parents=True)
    assert buildlib.prune_own_metadata(tmp_path, "0.1.0") == ["aerolume-0.0.9.dist-info"]
    assert sorted(p.name for p in internal.iterdir()) == ["aerolume-0.1.0.dist-info",
                                                          "aerolume_extra-1.0.dist-info", "torch-2.14.1.dist-info"]
    with pytest.raises(FileNotFoundError, match="aerolume-9.9.9"):
        buildlib.prune_own_metadata(tmp_path, "9.9.9")             # never leave the app with none


def test_same_version_editable_metadata_is_not_merged_into_the_app(tmp_path):
    """0.1.0 is both the release and the version the development venv was installed with, so
    PyInstaller merges the editable install's dist-info into the generated folder of the same
    name. The app must ship exactly the generated files: no RECORD and no direct_url.json, which
    names the developer's own source folder."""
    gen = buildlib.write_dist_info(tmp_path / "gen", REPO)
    kept = tmp_path / "dist" / "_internal" / gen.name
    shutil.copytree(gen, kept)
    (kept / "RECORD").write_text("aerolume/__init__.py,,\n", encoding="utf-8")
    (kept / "direct_url.json").write_text('{"url": "file:///C:/Users/someone/OneDrive/src"}', encoding="utf-8")
    (kept / "METADATA").write_text("Metadata-Version: 2.1\nName: aerolume\nVersion: old\n", encoding="utf-8")
    version = gen.name.removeprefix("aerolume-").removesuffix(".dist-info")
    removed = buildlib.prune_own_metadata(tmp_path / "dist", version, generated=gen)
    assert sorted(removed) == sorted([f"{gen.name}/RECORD", f"{gen.name}/direct_url.json"])
    assert sorted(p.name for p in kept.iterdir()) == sorted(p.name for p in gen.iterdir())
    assert (kept / "METADATA").read_bytes() == (gen / "METADATA").read_bytes()


@on_windows
def test_installer_script_compiles(tmp_path):
    """ISCC accepts the script for both variants (a tiny fake app, zip compression: ~2 s)."""
    iscc = buildlib.find_iscc()
    if iscc is None:
        pytest.skip("Inno Setup 6 not installed")
    src = tmp_path / "src"
    (src / "_internal").mkdir(parents=True)
    (src / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    (src / "Aerolume.exe").write_bytes(b"MZ")
    (src / "_internal" / "x.txt").write_text("x", encoding="utf-8")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    ico = buildlib.make_ico(plat.resource_path("aerolume.svg"), tmp_path / "a.ico")
    for variant in buildlib.VARIANTS:
        r = subprocess.run([str(iscc), "/Q", "/DAppVersion=9.8.7", "/DVersionQuad=9.8.7.0", f"/DVariant={variant}",
                            f"/DSourceDir={src}", f"/DOutputDir={tmp_path / 'out'}", f"/DIconFile={ico}",
                            "/DCompression=zip", str(PACKAGING / "aerolume.iss")],
                           capture_output=True, text=True, timeout=120, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        setup = tmp_path / "out" / f"Aerolume-9.8.7-{variant}-setup.exe"
        info = buildlib.inspect_installer(setup)
        assert info["inno_signature"].startswith("Inno Setup Setup Data (")
        assert info["version"]["ProductVersion"].startswith("9.8.7 (")
        assert info["version"]["FileVersion"] == "9.8.7.0"
        # the compiler's manifest proves the setup holds exactly the app folder
        manifest = tmp_path / "out" / f"Aerolume-9.8.7-{variant}-manifest.txt"
        check = buildlib.verify_installer_manifest(manifest, src)
        assert check["ok"] and check["files"] == 3, check
    # and a folder that changed after compiling is caught (size, then content, then a new file)
    (src / "_internal" / "x.txt").write_text("xy", encoding="utf-8")
    bad = buildlib.verify_installer_manifest(manifest, src)
    assert not bad["ok"] and bad["size_mismatch"] == ["_internal/x.txt"]
    (src / "_internal" / "x.txt").write_text("y", encoding="utf-8")
    assert buildlib.verify_installer_manifest(manifest, src)["hash_mismatch"] == ["_internal/x.txt"]
    (src / "_internal" / "x.txt").write_text("x", encoding="utf-8")
    (src / "new.dll").write_bytes(b"\0")
    bad = buildlib.verify_installer_manifest(manifest, src)
    assert bad["missing"] == ["new.dll"] and not bad["ok"]


def test_installer_manifest_parsing_and_path_spelling(tmp_path):
    """Entries are matched relative to their common folder, whatever spelling ISCC saw."""
    import hashlib
    dist = tmp_path / "dist"
    (dist / "_internal").mkdir(parents=True)
    (dist / "Aerolume.exe").write_bytes(b"MZ")
    (dist / "_internal" / "a.dll").write_bytes(b"abc")
    other = "C:\\Users\\X\\AppData\\Local\\Packages\\P\\LocalCache\\Local\\b\\dist"
    rows = [("0", other + "\\Aerolume.exe", hashlib.sha256(b"MZ").hexdigest(), "2"),
            ("1", other + "\\_internal\\a.dll", hashlib.sha256(b"abc").hexdigest(), "3")]
    header = "Index\tSourceFilename\tTimeStamp\tVersion\tSHA256Sum\tOriginalSize"
    lines = [header] + [f"{i}\t{s}\t2026-01-01\t\t{h}\t{n}" for i, s, h, n in rows]
    manifest = tmp_path / "m.txt"
    manifest.write_text("\ufeff" + "\r\n".join(lines) + "\r\n", encoding="utf-8")
    assert [r["OriginalSize"] for r in buildlib.read_installer_manifest(manifest)] == ["2", "3"]
    check = buildlib.verify_installer_manifest(manifest, dist)
    assert check["ok"] and check["bytes"] == 5, check
    manifest.write_text("", encoding="utf-8")
    assert not buildlib.verify_installer_manifest(manifest, dist)["ok"]
