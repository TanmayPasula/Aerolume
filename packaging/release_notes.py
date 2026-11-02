r"""Release measurements written from ``build-report.json``, never typed by hand.

Why this module exists: draft release notes during development quoted a self-test time, a smoke-test time, an
installer compile time and a total build time that belonged to an earlier run than the
installer that shipped. The numbers were real, but they were copied into ``docs/install.md``
and the CHANGELOG by hand from the full build, and the shipped installer had since been
recompiled twice by ``build.py --refresh-docs`` with different timings. Nothing said which run
the numbers came from.

Three rules follow:

* Every run of ``build.py`` appends a record to ``runs`` in ``build-report.json``
  (:func:`run_record`): what kind of run it was, the fingerprint it was built from, the smoke
  test, the self-test, the installer compile, the installer's SHA-256 and the machine.
  A full build starts a new history; ``--refresh-docs`` and ``--reuse-dist`` extend it, so the
  report of the shipped installer still holds the full build that produced its program.
* Text that quotes these numbers is generated (:func:`render_measurements`,
  :func:`render_summary`) between ``<!-- name:begin -->`` / ``<!-- name:end -->`` markers by
  ``build.py --write-measurements``, and every table names the run each column comes from.
* A document inside an installer cannot describe that installer: writing it changes the
  installer's contents (and the installer's SHA-256 is not known until it is compiled). The
  docs therefore describe the runs recorded *before* they were written, and say so; the exact
  installer is described by ``Aerolume-<version>-<variant>-release-notes.md``, which
  :func:`render_release_notes` writes beside it after every compile (it ships on the release
  page, not inside the installer).

Everything here is plain standard library and has no side effects except :func:`write_blocks`
and :func:`write_release_notes`, so it is unit-tested without a build.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import sys
import time
from pathlib import Path

#: Marker names of the generated blocks. ``build-measurements`` is the full table
#: (docs/install.md); ``build-summary`` is one paragraph (the CHANGELOG's release-build bullet).
MEASUREMENTS_BLOCK = "build-measurements"
SUMMARY_BLOCK = "build-summary"

#: Smoke-test check names (``smoke_test.smoke``) that carry the self-test launch times.
FIRST_LAUNCH = "selftest (first launch)"
SECOND_LAUNCH = "selftest (second launch)"

#: Run kinds as recorded and as shown to readers.
KIND_LABELS = {"full": "full build", "refresh-docs": "docs refresh", "reuse-dist": "re-package",
               "unknown": "build"}


# ---------------------------------------------------------------------------
# measuring the machine
# ---------------------------------------------------------------------------

def system_cpu_times() -> tuple[int, int] | None:
    """(busy, total) CPU time of the whole machine in 100 ns units, or ``None`` off Windows.

    Sampled at the start and end of a run, the difference gives the average share of the
    machine's CPU that was busy - with the build and with everything else that ran. The release
    builds so far were made while other jobs shared the machine; this turns "other jobs were
    running" into a number. ``GetSystemTimes`` counts idle time inside kernel time, so busy is
    ``kernel + user - idle``.
    """
    if sys.platform != "win32":
        return None

    class FILETIME(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

    idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
    try:
        ok = ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    except (AttributeError, OSError):
        return None
    if not ok:
        return None

    def val(ft: FILETIME) -> int:
        return (ft.hi << 32) | ft.lo

    total = val(kernel) + val(user)
    return total - val(idle), total


def cpu_busy_percent(start: tuple[int, int] | None, end: tuple[int, int] | None) -> float | None:
    """Average busy share of all CPUs between two :func:`system_cpu_times` samples, in percent."""
    if not start or not end:
        return None
    busy, total = end[0] - start[0], end[1] - start[1]
    if total <= 0:
        return None
    return round(100.0 * busy / total, 1)


def _cpu_name() -> str | None:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return " ".join(str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).split())
        except OSError:
            pass
    return platform.processor() or None


def _ram_bytes() -> int | None:
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys)
        except (AttributeError, OSError):
            return None
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def machine_info() -> dict:
    """The build machine as the docs describe it: CPU, logical CPUs, physical RAM, OS."""
    return {"cpu": _cpu_name(), "logical_cpus": os.cpu_count(), "ram_bytes": _ram_bytes(),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})"}


def gpu_from_smoke(smoke: dict) -> str | None:
    """The CUDA device the bundled self-test ran on, from its ``torch`` check's detail line."""
    for check in (smoke or {}).get("checks") or []:
        for inner in ((check.get("report") or {}).get("checks") or []):
            if inner.get("name") == "torch":
                m = re.search(r"CUDA device (.+?)(?:,|$)", str(inner.get("detail") or ""))
                if m:
                    return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# the run history
# ---------------------------------------------------------------------------

def _check(smoke: dict, name: str) -> dict:
    for check in (smoke or {}).get("checks") or []:
        if check.get("name") == name:
            return check
    return {}


def run_record(report: dict, kind: str, *, finished: str | None = None,
               cpu_busy: float | None = None) -> dict:
    """One run of ``build.py``, summarised from the report it is about to write.

    Only values this run measured go in: a docs refresh carries the PyInstaller time of the
    full build in its report (the folder's history), so ``pyinstaller_seconds`` is recorded for
    full builds only. ``None`` means "not measured by this run", never zero.
    """
    smoke = report.get("smoke") or {}
    inst = report.get("installer") or {}
    manifest = report.get("installer_manifest") or {}
    first, second = _check(smoke, FIRST_LAUNCH), _check(smoke, SECOND_LAUNCH)
    source = report.get("source") or {}
    if report.get("smoke_reused"):
        # --reuse-dist --skip-smoke carries the folder's earlier result; this run measured nothing.
        smoke, first, second = {"skipped": True}, {}, {}
    if smoke.get("skipped"):
        smoke_ok = None
    else:
        smoke_ok = smoke.get("ok") if "ok" in smoke else None
    return {
        "kind": kind,
        "finished": finished or time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_sha256": source.get("sha256"),
        "pyinstaller_seconds": report.get("pyinstaller_seconds") if kind == "full" else None,
        "smoke_ok": smoke_ok,
        "smoke_seconds": report.get("smoke_seconds") if "smoke" in report and not smoke.get("skipped") else None,
        "selftest_checks": len((first.get("report") or {}).get("checks") or []) or None,
        "selftest_first_seconds": first.get("wall_seconds"),
        "selftest_second_seconds": second.get("wall_seconds"),
        "installer_seconds": report.get("installer_seconds") if inst else None,
        "installer_compression": report.get("installer_compression") if inst else None,
        "installer_name": Path(inst["path"]).name if inst.get("path") else None,
        "installer_bytes": inst.get("bytes"),
        "installer_sha256": inst.get("sha256"),
        "manifest_ok": manifest.get("ok") if manifest else None,
        "manifest_files": manifest.get("files") if manifest else None,
        "dist_bytes": report.get("dist_bytes"),
        "dist_files": report.get("dist_files"),
        "total_seconds": report.get("total_seconds"),
        "cpu_busy_percent": cpu_busy,
        "gpu": gpu_from_smoke(smoke),
        "machine": (report.get("interpreter") or {}).get("machine"),
    }


def runs_of(report: dict) -> list[dict]:
    """The run history of a report; a report written before histories existed yields one run.

    For such a report the kind is inferred (``docs_refresh`` marks a refresh) and its values
    are only those the report itself measured, so no number is ever attributed to a run that
    did not measure it.
    """
    runs = report.get("runs")
    if isinstance(runs, list) and runs:
        return [r for r in runs if isinstance(r, dict)]
    if not report.get("source"):
        return []
    kind = "refresh-docs" if report.get("docs_refresh") else ("full" if report.get("pyinstaller_seconds") else "unknown")
    return [run_record(report, kind, finished=(report.get("build_info") or {}).get("built"))]


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _gb(n: int | None) -> str:
    return "not recorded" if n is None else f"{n / 1e9:.2f} GB"


def _secs(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s:.0f} s" if s >= 100 else f"{s:.1f} s"


def _mins(s: float | None) -> str:
    return "-" if s is None else f"{s / 60:.1f} min"


def _short(h: str | None, n: int = 12) -> str:
    return (h or "")[:n] or "?"


def _date(stamp: str | None) -> str:
    """``2026-10-30 23:38:40`` -> ``30 Oct 2026, 23:38`` (stable across locales)."""
    try:
        t = time.strptime(stamp or "", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stamp or "an unrecorded time"
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{t.tm_mday} {months[t.tm_mon - 1]} {t.tm_year}, {t.tm_hour:02d}:{t.tm_min:02d}"


def _machine_line(run: dict) -> str:
    m = run.get("machine") or {}
    parts = [p for p in (m.get("cpu"),
                         f"{m['logical_cpus']} logical CPUs" if m.get("logical_cpus") else None,
                         f"{m['ram_bytes'] / 2**30:.0f} GB RAM" if m.get("ram_bytes") else None,
                         run.get("gpu")) if p]
    return ", ".join(parts) if parts else "not recorded"


def _kind(run: dict) -> str:
    return KIND_LABELS.get(run.get("kind") or "unknown", run.get("kind") or "build")


def _column(run: dict) -> str:
    return f"{_kind(run)} `{_short(run.get('source_sha256'))}`"


def unmeasured_refreshes(report: dict, runs: list[dict]) -> list[tuple[str, str]]:
    """``(fingerprint, time)`` of docs refreshes in ``build-info.json`` missing from ``runs``.

    ``docs_refreshed`` records the fingerprint each refresh started *from*; the one it produced
    is the next record's ``from_sha256``, and the last one produced the folder's current
    fingerprint. Listing the ones the history lacks keeps a replaced intermediate installer
    from disappearing silently.
    """
    info = report.get("build_info") or {}
    records = [r for r in info.get("docs_refreshed") or [] if isinstance(r, dict)]
    measured = {r.get("source_sha256") for r in runs}
    out = []
    for i, rec in enumerate(records):
        produced = records[i + 1].get("from_sha256") if i + 1 < len(records) else info.get("source_sha256")
        if produced and produced not in measured:
            out.append((produced, rec.get("at") or ""))
    return out


def _program_run(runs: list[dict]) -> dict | None:
    """The last full build: the run that compiled the program every later run re-packaged."""
    for run in reversed(runs):
        if run.get("kind") == "full":
            return run
    return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_measurements(report: dict, *, generated: str | None = None) -> str:
    """The measurement table for ``docs/install.md``: one column per recorded run.

    The text states which run produced the program, which run compiled the installer the
    numbers end with (with its SHA-256), the machine and how busy it was, and why this table
    can never include the installer that ships it.
    """
    runs = runs_of(report)
    version, variant = report.get("version", "?"), report.get("variant", "?")
    if not runs:
        return (f"No build of Aerolume {version} has been recorded yet: run `python packaging/build.py`, "
                "then `python packaging/build.py --write-measurements`.")
    program = _program_run(runs)
    last = runs[-1]
    lines = [f"**Measured release build: Aerolume {version}, {variant.upper()} variant.** "
             "Every number in this block is copied from the `build-report.json` of the runs named in "
             "its column headings by `python packaging/build.py --write-measurements`"
             + (f" ({generated})" if generated else "") + "; none is typed by hand.", ""]
    if program:
        lines.append(f"* The program was compiled by the full build of snapshot "
                     f"`{_short(program.get('source_sha256'))}`, finished {_date(program.get('finished'))}.")
    else:
        lines.append("* The full build that compiled the program is not in this report's history.")
    later = [r for r in runs if r is not program]
    if later:
        lines.append("* Later runs re-tested and re-packaged that same program without recompiling it: "
                     + "; ".join(f"{_kind(r)} to `{_short(r.get('source_sha256'))}` ({_date(r.get('finished'))})"
                                 for r in later) + ".")
    if last.get("installer_sha256"):
        lines.append(f"* The last run recorded here compiled `{last.get('installer_name')}`, SHA-256 "
                     f"`{last['installer_sha256']}`.")
    busy = [f"{_column(r)} {r['cpu_busy_percent']:.0f} %" for r in runs if r.get("cpu_busy_percent") is not None]
    lines.append(f"* Machine: {_machine_line(last)}. "
                 + ("Average CPU use of the whole machine during each run, the build included: "
                    + ", ".join(busy) + ". " if busy else "The machine's load was not recorded for these runs. ")
                 + "Times depend on load and on what is already in the file cache; "
                 "sizes, file counts and hashes do not.")
    lines.append("")
    cols = [_column(r) for r in runs]
    lines.append("| step | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(runs))

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("PyInstaller", [_secs(r.get("pyinstaller_seconds")) if r.get("kind") == "full" else "not run"
                        for r in runs])
    row("smoke test (self-test twice, console program, `doctor`, single instance)",
        [("**failed** " if r.get("smoke_ok") is False else "") + _secs(r.get("smoke_seconds")) for r in runs])
    checks = next((r.get("selftest_checks") for r in reversed(runs) if r.get("selftest_checks")), None)
    row(f"self-test of the built app{f', {checks} checks' if checks else ''}: first / second launch",
        [f"{_secs(r.get('selftest_first_seconds'))} / {_secs(r.get('selftest_second_seconds'))}" for r in runs])
    methods = {r.get("installer_compression") for r in runs if r.get("installer_seconds") is not None}
    if len(methods) == 1 and None not in methods:
        row(f"installer compile ({methods.pop()})", [_secs(r.get("installer_seconds")) for r in runs])
    else:
        row("installer compile", [_secs(r.get("installer_seconds"))
                                  + (f" ({r['installer_compression']})" if r.get("installer_compression") else "")
                                  for r in runs])
    row("installer SHA-256", [f"`{_short(r.get('installer_sha256'))}`" if r.get("installer_sha256") else "-"
                              for r in runs])
    row("whole run", [_mins(r.get("total_seconds")) for r in runs])
    lines.append("")
    size = last
    files_line = (f"App folder {_gb(size.get('dist_bytes'))} in {size.get('dist_files'):,} files; installer "
                  f"{_gb(size.get('installer_bytes'))}." if size.get("dist_files") else "")
    if size.get("manifest_ok"):
        files_line += (f" The installer's manifest lists {size.get('manifest_files'):,} files, identical in size "
                       "and SHA-256 to the tested app folder.")
    if files_line:
        lines += [files_line, ""]
    unmeasured = unmeasured_refreshes(report, runs)
    if unmeasured:
        lines += ["Docs refreshes recorded in the app's `build-info.json` but not measured in this history "
                  "(their installers were replaced by later runs): "
                  + ", ".join(f"`{_short(sha)}` ({_date(at)})" for sha, at in unmeasured) + ".", ""]
    rebuilt = [r for r in runs if r.get("recorded_from")]
    if rebuilt:
        lines += ["Columns converted from reports written before `build.py` kept a run history: "
                  + "; ".join(f"{_column(r)} from {r['recorded_from']}" for r in rebuilt) + ".", ""]
    lines.append("A *first launch* in a docs refresh or re-package starts with the app's libraries already "
                 "in the file cache, so it is faster than the first start after installing, when Windows "
                 "also reads 3.5 GB from disk and antivirus software scans it. A `-` means the run did not "
                 "measure that step.")
    lines.append("")
    lines.append("This page ships inside the installer, so it cannot describe the installer that contains "
                 "it: writing these numbers changes the docs, and the release installer is then recompiled "
                 "from the same program with `--refresh-docs`. That run's own figures and the SHA-256 of "
                 f"the installer you download are in `Aerolume-{version}-{variant}-release-notes.md` beside "
                 "the installer on the release page.")
    return "\n".join(lines)


def render_summary(report: dict) -> str:
    """The CHANGELOG's release-build bullet (a ``- `` list item), naming the run it quotes.

    It is a whole list item so that its markers can sit on their own lines, outside the text:
    a comment inside a Markdown paragraph would end it.
    """
    runs = runs_of(report)
    if not runs:
        return "- Release build: none recorded yet."
    last, program = runs[-1], _program_run(runs)
    parts = [("- Release build, measured by `packaging/build.py` (docs/install.md has the table of every run): "
              f"the app folder is {_gb(last.get('dist_bytes'))} in {last.get('dist_files') or 0:,} files")]
    if last.get("installer_bytes"):
        parts[0] += f" and `{last.get('installer_name')}` is {_gb(last.get('installer_bytes'))}"
    parts[0] += "."
    if last.get("selftest_first_seconds") is not None:
        checks = last.get("selftest_checks")
        verdict = "failed" if last.get("smoke_ok") is False else (
            f"passed all {checks} checks" if checks else "passed")
        parts.append(f"In the {_kind(last)} `{_short(last.get('source_sha256'))}`, the last recorded run, the "
                     f"bundled self-test {verdict} in {_secs(last.get('selftest_first_seconds'))} on the first "
                     f"launch and {_secs(last.get('selftest_second_seconds'))} on the second.")
    if program and program.get("total_seconds") is not None:
        parts.append(f"The full build `{_short(program.get('source_sha256'))}` took "
                     f"{_mins(program.get('total_seconds'))}. Times depend on the machine's load.")
    return " ".join(parts)


def render_release_notes(report: dict) -> str:
    """The file published beside the installer: this exact installer's run, hash and checks."""
    runs = runs_of(report)
    version, variant = report.get("version", "?"), report.get("variant", "?")
    last = runs[-1] if runs else {}
    program = _program_run(runs)
    smoke = report.get("smoke") or {}
    out = [f"# Aerolume {version} ({variant.upper()}) - build measurements", "",
           ("Written by `packaging/build.py` from `build-report.json` right after the installer was compiled "
           "and checked. It describes this installer; `docs/install.md` inside it describes the runs before it."),
           ""]
    if last.get("installer_sha256"):
        out += ["| | |", "|---|---|",
                (f"| installer | `{last.get('installer_name')}`, {last.get('installer_bytes'):,} bytes "
                f"({_gb(last.get('installer_bytes'))}) |"),
                f"| SHA-256 | `{last['installer_sha256']}` |",
                f"| made by | {_kind(last)}, finished {_date(last.get('finished'))} |",
                f"| sources | snapshot `{last.get('source_sha256')}` |"]
        if program and program is not last:
            out.append(f"| program compiled by | full build of `{program.get('source_sha256')}`, finished "
                       f"{_date(program.get('finished'))} |")
        if last.get("dist_files"):
            out.append(f"| contents | {last['dist_files']:,} files, {last.get('dist_bytes'):,} bytes; the manifest "
                       + ("matches the tested app folder file for file (size and SHA-256)"
                          if last.get("manifest_ok") else "was NOT verified against the app folder") + " |")
        out.append(f"| machine | {_machine_line(last)}"
                   + (f"; whole-machine CPU use during the run {last['cpu_busy_percent']:.0f} %"
                      if last.get("cpu_busy_percent") is not None else "") + " |")
        out += ["", ("Check a download in PowerShell with "
                f"`Get-FileHash .\\{last.get('installer_name')} -Algorithm SHA256`."), ""]
    else:
        out += ["No installer was compiled by the last run.", ""]
    out += ["## Timings of this run", "", "| step | time |", "|---|---|",
            f"| PyInstaller | {_secs(last.get('pyinstaller_seconds')) if last.get('kind') == 'full' else 'not run'} |",
            f"| smoke test | {_secs(last.get('smoke_seconds'))} |",
            (f"| self-test, first / second launch | {_secs(last.get('selftest_first_seconds'))} / "
            f"{_secs(last.get('selftest_second_seconds'))} |"),
            f"| installer compile | {_secs(last.get('installer_seconds'))} |",
            f"| whole run | {_mins(last.get('total_seconds'))} |", ""]
    first = _check(smoke, FIRST_LAUNCH)
    inner = (first.get("report") or {}).get("checks") or []
    if smoke.get("checks"):
        out += ["## Smoke test", "", "| check | result | time |", "|---|---|---|"]
        for c in smoke["checks"]:
            out.append(f"| {c.get('name')} | {'pass' if c.get('ok') else '**FAIL**'} | {_secs(c.get('wall_seconds'))} |")
        out.append("")
    if inner:
        out += ["## Self-test checks (first launch)", "", "| check | result | time | detail |", "|---|---|---|---|"]
        for c in inner:
            detail = str(c.get("detail") or "").replace("|", "/").replace("\n", " ")
            out.append(f"| {c.get('name')} | {'pass' if c.get('ok') else '**FAIL**'} | "
                       f"{_secs(c.get('seconds'))} | {detail} |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

class MarkerError(ValueError):
    """A document has no, or more than one, generated block of the requested name."""


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:begin -->", f"<!-- {name}:end -->"


def has_block(text: str, name: str) -> bool:
    begin, end = _markers(name)
    return begin in text and end in text


def replace_block(text: str, name: str, body: str) -> str:
    """``text`` with the lines between the ``name`` markers replaced by ``body``.

    The markers stay, so the block can be regenerated after every build. Exactly one pair is
    required: a second pair would leave one copy of the numbers stale.
    """
    begin, end = _markers(name)
    if text.count(begin) != 1 or text.count(end) != 1:
        raise MarkerError(f"expected exactly one '{begin}' ... '{end}' pair, found "
                          f"{text.count(begin)} and {text.count(end)}")
    if text.index(end) < text.index(begin):
        raise MarkerError(f"'{end}' comes before '{begin}'")
    head, rest = text.split(begin, 1)
    _old, tail = rest.split(end, 1)
    newline = "\r\n" if "\r\n" in text else "\n"
    body = body.strip("\n").replace("\r\n", "\n").replace("\n", newline)
    return f"{head}{begin}{newline}{body}{newline}{end}{tail}"


def write_blocks(report: dict, paths: dict[Path, str], *, generated: str | None = None) -> dict[str, list[str]]:
    """Regenerate the named block in each file that has its markers.

    ``paths`` maps a document to the block it carries (``MEASUREMENTS_BLOCK`` or
    ``SUMMARY_BLOCK``). Returns ``{"changed", "unchanged", "no_markers"}`` lists of paths; a
    file without markers is reported, never edited, so the tool cannot write numbers where a
    document's owner did not ask for them.
    """
    out: dict[str, list[str]] = {"changed": [], "unchanged": [], "no_markers": []}
    for path, name in paths.items():
        if not path.is_file():
            out["no_markers"].append(str(path))
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if not has_block(text, name):
            out["no_markers"].append(str(path))
            continue
        body = render_measurements(report, generated=generated) if name == MEASUREMENTS_BLOCK else render_summary(report)
        new = replace_block(text, name, body)
        if new == text:
            out["unchanged"].append(str(path))
            continue
        path.write_bytes(new.encode("utf-8"))
        out["changed"].append(str(path))
    return out


def release_notes_path(setup: Path) -> Path:
    """``Aerolume-0.1.0-cuda-setup.exe`` -> ``Aerolume-0.1.0-cuda-release-notes.md`` beside it."""
    return setup.with_name(setup.name.replace("-setup.exe", "-release-notes.md"))


def write_release_notes(report: dict, setup: Path) -> Path:
    """Write :func:`render_release_notes` beside ``setup``; returns the path."""
    path = release_notes_path(setup)
    path.write_text(render_release_notes(report), encoding="utf-8")
    return path
