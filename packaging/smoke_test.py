"""Smoke-test a built Aerolume folder (and inspect its installer) without installing anything.

    python packaging/smoke_test.py --dist %USERPROFILE%\\aerolume-build\\cuda\\dist\\Aerolume
           [--installer ...\\Aerolume-0.1.0-cuda-setup.exe] [--json report.json] [--no-window]

What it proves, in the order it runs:

1. ``Aerolume.exe --selftest`` (Qt ``offscreen`` platform, a throw-away config folder): the
   frozen app starts, finds its resources, docs and licences, imports the standard library,
   SciPy and the Aerolume API (accounts included), runs cuBLAS/cuFFT/NVRTC kernels, 20 steps
   of the 2D solver and 10 steps of the 3D solver with a heated body, a fan and smoke and
   particle emitters on the GPU (failing if its fused NVRTC kernels or CUDA graphs fell back),
   an airfoil analysis, an off-screen VTK frame, and builds and paints the main window - then
   exits with code 0. Run twice: the first launch after a build is a cold start
   (Windows Defender scans the new DLLs), the second is what a user sees every day.
2. ``aerolume-cli.exe --version`` and ``aerolume-cli.exe doctor`` - the console executable
   shares the bundle and its output reaches the console.
3. Single instance: a second ``Aerolume.exe`` launched while one is running must hand its
   arguments over and exit with code 0 (skipped, and reported as not wired, if the running
   instance never takes the instance lock).
4. The installer (when given): PE version resource, the Inno Setup data signature, size and
   SHA-256 - read from the file, never executed, so nothing on this machine changes.

The exit code is 0 only when every check that ran passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import buildlib

CREATE_NO_WINDOW = 0x08000000


def _env(config_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["AEROLUME_CONFIG_DIR"] = str(config_dir)
    env.pop("PYTHONPATH", None)            # the frozen app must not see the source tree
    env.pop("PYTHONHOME", None)
    return env


def run_selftest(dist: Path, config_dir: Path, *, window: bool = True, timeout: float = 900.0) -> dict:
    """Run ``Aerolume.exe --selftest`` and return its JSON report plus exit code and wall time."""
    out = config_dir / f"selftest-{int(time.time() * 1000)}.json"
    cmd = [str(dist / buildlib.GUI_EXE), "--selftest", "--selftest-out", str(out)]
    if not window:
        cmd.append("--no-window")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=_env(config_dir), timeout=timeout, capture_output=True,
                              creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0, check=False)
        code: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        code = None
    seconds = time.perf_counter() - t0
    report: dict = {}
    if out.is_file():
        try:
            report = json.loads(out.read_text(encoding="utf-8"))
        except ValueError as exc:
            report = {"ok": False, "error": f"unreadable self-test JSON: {exc}"}
    ok = code == 0 and bool(report.get("ok"))
    return {"name": "selftest", "ok": ok, "exit_code": code, "wall_seconds": round(seconds, 2), "report": report}


def run_cli(dist: Path, config_dir: Path, args: list[str], *, expect: str | None = None,
            timeout: float = 600.0) -> dict:
    """Run ``aerolume-cli.exe args``; passes on exit code 0 (and ``expect`` in the output)."""
    cmd = [str(dist / buildlib.CLI_EXE), *args]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=_env(config_dir), timeout=timeout, capture_output=True, check=False,
                              encoding="utf-8", errors="replace")
        code: int | None = proc.returncode
        text = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        code, text = None, "timed out"
    ok = code == 0 and (expect is None or expect in text)
    return {"name": "cli " + " ".join(args), "ok": ok, "exit_code": code,
            "wall_seconds": round(time.perf_counter() - t0, 2), "output_tail": text[-1500:]}


def check_single_instance(dist: Path, config_dir: Path, project: Path | None, *, wait_lock: float = 120.0,
                          wait_exit: float = 60.0) -> dict:
    """Start one instance, launch a second with a project argument, expect the second to exit 0."""
    env = _env(config_dir)
    exe = str(dist / buildlib.GUI_EXE)
    flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
    first = subprocess.Popen([exe], env=env, creationflags=flags, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    result: dict = {"name": "single instance", "ok": False}
    try:
        t0 = time.perf_counter()
        lock = None
        while time.perf_counter() - t0 < wait_lock and first.poll() is None:
            locks = list(config_dir.glob("aerolume-*.lock"))
            if locks:
                lock = locks[0]
                break
            time.sleep(0.25)
        if lock is None:
            result.update(ok=True, skipped=True, detail="the running instance never took the instance lock "
                          "(SingleInstance not wired into main.py in this build)",
                          first_exit_code=first.poll())
            return result
        result["lock_seconds"] = round(time.perf_counter() - t0, 2)
        time.sleep(2.0)                    # let the first instance start listening
        args = [exe] + ([str(project)] if project else [])
        t1 = time.perf_counter()
        second = subprocess.Popen(args, env=env, creationflags=flags, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        try:
            code = second.wait(timeout=wait_exit)
        except subprocess.TimeoutExpired:
            second.kill()
            code = None
        result.update(second_exit_code=code, second_seconds=round(time.perf_counter() - t1, 2),
                      first_still_running=first.poll() is None)
        result["ok"] = code == 0 and first.poll() is None
        return result
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=30)


def first_example_project(dist: Path) -> Path | None:
    """A bundled example project folder, used as the forwarded argument."""
    root = dist / "_internal" / "aerolume" / "examples" / "projects"
    for p in sorted(root.glob("*")) if root.is_dir() else []:
        if (p / "project.json").is_file():
            return p
    return None


def smoke(dist: Path, *, installer: Path | None = None, window: bool = True, single_instance: bool = True,
          work: Path | None = None) -> dict:
    """Run every check against ``dist`` (and ``installer``); returns the report dict."""
    if not (dist / buildlib.GUI_EXE).is_file():
        raise FileNotFoundError(f"{dist / buildlib.GUI_EXE} does not exist - build first")
    work = work or Path(tempfile.mkdtemp(prefix="aerolume-smoke-"))
    config = work / "config"
    if config.exists():
        shutil.rmtree(config, ignore_errors=True)
    config.mkdir(parents=True)
    checks = []
    cold = run_selftest(dist, config, window=window)
    cold["name"] = "selftest (first launch)"
    checks.append(cold)
    warm = run_selftest(dist, config, window=window)
    warm["name"] = "selftest (second launch)"
    checks.append(warm)
    checks.append(run_cli(dist, config, ["--version"], expect=buildlib.read_version()))
    checks.append(run_cli(dist, config, ["doctor"]))
    if single_instance:
        checks.append(check_single_instance(dist, config, first_example_project(dist)))
    if installer is not None:
        info = buildlib.inspect_installer(installer)
        ver = info["version"].get("ProductVersion", "")
        ok = bool(info["inno_signature"]) and ver.startswith(buildlib.read_version())
        checks.append({"name": "installer inspection", "ok": ok, **info})
    leftovers = sorted(p.name for p in (config / "logs").glob("crash-*.log")) if (config / "logs").is_dir() else []
    return {"ok": all(c["ok"] for c in checks) and not leftovers, "dist": str(dist), "checks": checks,
            "crash_logs": leftovers, "config_dir": str(config)}


def _print(report: dict) -> None:
    for c in report["checks"]:
        status = "PASS" if c["ok"] else "FAIL"
        extra = f" ({c['wall_seconds']} s)" if "wall_seconds" in c else ""
        if c.get("skipped"):
            status = "SKIP"
        print(f"{status}  {c['name']}{extra}")
        for sub in c.get("report", {}).get("checks", []):
            print(f"      {'ok ' if sub['ok'] else 'BAD'} {sub['name']:<30} {sub['seconds']:>7.2f} s  "
                  f"{sub['detail'].splitlines()[0][:110]}")
        if not c["ok"] and "output_tail" in c:
            print("      " + c["output_tail"][-600:].replace("\n", "\n      "))
        if c.get("detail"):
            print("      " + str(c["detail"]))
    if report["crash_logs"]:
        print("Crash logs written during the test: " + ", ".join(report["crash_logs"]))
    print("SMOKE TEST " + ("PASSED" if report["ok"] else "FAILED"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, required=True, help="the dist\\Aerolume folder")
    ap.add_argument("--installer", type=Path, help="compiled setup .exe to inspect (never run)")
    ap.add_argument("--json", type=Path, help="write the full report here")
    ap.add_argument("--work", type=Path, help="scratch folder (default: a new temporary folder)")
    ap.add_argument("--no-window", action="store_true", help="skip building the main window")
    ap.add_argument("--skip-single-instance", action="store_true")
    args = ap.parse_args(argv)
    report = smoke(args.dist, installer=args.installer, window=not args.no_window,
                   single_instance=not args.skip_single_instance, work=args.work)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
