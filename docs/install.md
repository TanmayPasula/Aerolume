# Installing Aerolume

There are three ways to run Aerolume on Windows. They run the same code and give the same
numbers; they differ in what you need to have installed and how you update.

| | Installer | Portable folder | From source |
|---|---|---|---|
| What you get | `Aerolume-<version>-cuda-setup.exe` | the `Aerolume` folder the installer would install | a Python environment with the `aerolume` package |
| Needs Python | no | no | Python 3.11+ |
| Administrator rights | no (installs for the current user) | no | no |
| Start-menu entry, `.aerolume` association, uninstaller | yes | no | no |
| Disk space | 3.54 GB installed (download 1.94 GB) | 3.54 GB | about 4 GB with the CUDA build of PyTorch (4.1 GB measured) |
| Best for | using Aerolume | USB sticks, locked-down PCs, trying a build | development, scripting against your own Python packages |

The installer and the portable folder include everything: Python, PyTorch with the NVIDIA
CUDA libraries, VTK, Qt, the documentation and the validation scripts. They also include a
licence text or notice for every bundled component: the licence files of each Python package,
the LGPL-3.0 and GPL-3.0 texts for Qt, CPython's licences, the licence texts of the libraries
VTK builds into `_internal\vtk.libs` (`licenses\vtk\third_party`), Qt's attribution of the
third-party code built into it and of Mesa's llvmpipe in `opengl32sw.dll`
(`licenses\PySide6_Essentials\third_party`), and notices for the proprietary
NVIDIA CUDA/cuDNN and Intel OpenMP libraries that come with PyTorch. The notices name each
agreement and give its address (see `licenses\README.txt` and THIRD_PARTY_NOTICES.md). A **CPU-only** build can be made with `packaging/build.py --variant cpu`;
it runs every feature, only slower. It has not been built and measured yet; leaving out the
2.5 GB of CUDA libraries should make the folder about 1 GB.

## System requirements

* Windows 10 version 1809 or later, or Windows 11, 64-bit.
* For GPU simulations: an NVIDIA GPU with a current driver (the bundled CUDA 13 runtime needs
  driver 580 or newer). Without a GPU or driver everything runs on the CPU; the setup warns if
  it finds no NVIDIA driver.
* OpenGL 3.2 for the 3D views (any GPU driver from the last ten years; see Troubleshooting for
  remote desktops and virtual machines).
* 8 GB of RAM; 16 GB for fine 3D grids.

## Installer

1. Run `Aerolume-<version>-cuda-setup.exe`. Windows SmartScreen may say the publisher is
   unknown: the installer is not code-signed (open-source builds have no signing certificate
   yet). Choose **More info ▸ Run anyway** only if you got the file from the project's own
   release page.
2. The installer asks for the licence (Apache 2.0), the install folder (default
   `%LOCALAPPDATA%\Programs\Aerolume`), and two options: a desktop shortcut (off by default)
   and opening `.aerolume` project files with Aerolume (on).
3. Aerolume appears in the Start menu. **Settings ▸ Apps ▸ Installed apps ▸ Aerolume ▸
   Uninstall** removes it.

What the installer changes, all inside your user profile:

* the program folder, a Start-menu shortcut (and the desktop shortcut if chosen);
* `HKEY_CURRENT_USER\Software\Classes\.aerolume` and `Aerolume.Project` (the file
  association), `Applications\Aerolume.exe` ("Open with") and an `App Paths` entry so
  **Win+R ▸ aerolume** starts the program;
* nothing else: no services, no drivers, no PATH changes, no telemetry.

Installing a newer version over an older one upgrades in place; the old program files are
removed first, so no stale library survives an upgrade. Close Aerolume before installing or
uninstalling - setup detects a running Aerolume and asks you to close it.

**Your data is never removed by the uninstaller**: projects stay where you saved them, and
settings, recent projects, AI-assistant settings and logs stay in `%APPDATA%\Aerolume`
(delete that folder by hand for a completely clean removal).

For all users of a PC (administrator rights needed), run the installer with `/ALLUSERS`; it then
installs to `C:\Program Files\Aerolume` and registers the association machine-wide. Silent
installation: `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART` (add `/TASKS="associate"` or
`/TASKS="associate,desktopicon"`; `/LOG=setup.log` writes a log).

### Opening projects

A project is a folder named `<Name>.aerolume`. Windows cannot associate a folder with a
program, so Aerolume puts a small file called `project.aerolume` into every project folder it
opens or saves; double-clicking that file opens the project. The file holds no data (the
project is `project.json` and the subfolders), and deleting it only removes the shortcut. If
Aerolume is already running, the project opens in the running window instead of starting a
second copy. Projects you open in the installed app also appear under **Recent** when you
right-click Aerolume in the taskbar or the Start menu.

## Portable folder

The portable build is the `Aerolume` folder produced by `packaging/build.py` (the same files
the installer installs). Copy it anywhere you can write to and run `Aerolume.exe`. Nothing is
registered; settings still go to `%APPDATA%\Aerolume` unless you set the environment variable
`AEROLUME_CONFIG_DIR` to another folder (for example on the stick itself) before starting.
If you pass the folder on to someone else, pass it on whole: `LICENSE`,
`THIRD_PARTY_NOTICES.md` and `licenses\` must go with the libraries. Qt's LGPL, for example,
requires its text to accompany the Qt DLLs.

Next to `Aerolume.exe` you find:

| | |
|---|---|
| `aerolume-cli.exe` | the command line (`aerolume-cli doctor`, `quick`, `polar`, `run`, `script`, `validate`, ...) |
| `docs\` | this documentation, also reachable from the Help menu |
| `validation\` | the benchmark cases behind `aerolume-cli validate` and docs/validation.md |
| `LICENSE`, `THIRD_PARTY_NOTICES.md`, `licenses\` | Aerolume's licence, and the licence texts or notices of every bundled component (`licenses\README.txt` lists them) |
| `build-info.json` | version, variant, build time, the versions of Python, PyTorch, CUDA, VTK and Qt, and the SHA-256 fingerprint of the sources the build was made from |
| `_internal\` | Python, the libraries and the app's resources - do not edit |

### Checking an installation

`Aerolume.exe --selftest` runs a self-test and exits: it checks the bundled resources, docs
and licences, that `build-info.json` names the version actually bundled and records its source
fingerprint, imports the standard library and SciPy for scripts and the Aerolume API
(accounts included), runs cuBLAS, cuFFT and NVRTC kernels, 20 time steps of the 2D solver and
10 of the 3D solver with a heated body, a fan, a smoke emitter and a particle emitter on the
GPU, an airfoil analysis, an off-screen VTK frame, and builds the main window. The 3D check
also fails an installed build whose fused GPU kernels or CUDA graphs could not start: the
solver would otherwise fall back to its slower reference arithmetic without telling you,
which is what a CUDA runtime file missing from the bundle looks like. The result (every check with its time) is
written to `%APPDATA%\Aerolume\logs\selftest.json`; add `--selftest-out <file>` to choose the
file, `--no-window` or `--no-gpu` to skip parts. `aerolume-cli doctor` prints versions, the
devices and a one-second solver check in the console. Crash reports start with a line of the
form `Aerolume 0.1.0 (CUDA build 2026-11-01, source <first 12 hex digits of the source
fingerprint>)`, which identifies the exact build even when several builds share a version
number.

## From source

Use this to develop Aerolume or to script it from your own Python environment. Put the virtual
environment **outside** OneDrive, Dropbox or other synced folders: the CUDA build of PyTorch is
several gigabytes of small and large files that the sync client would upload and lock.

```bash
git clone <repository URL> aerolume
cd aerolume
python -m venv %LOCALAPPDATA%\venvs\aerolume
%LOCALAPPDATA%\venvs\aerolume\Scripts\activate
```

PyTorch, with CUDA (NVIDIA GPU; pick the build that matches your driver on
[pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

or CPU only (every feature works; CFD runs use coarser grids and take several times longer):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

then Aerolume and the optional AI assistant and development tools:

```bash
pip install -e ".[dev,ai]"
aerolume-cli doctor
aerolume
```

### Building the installer

Requirements: the source install above plus `pip install pyinstaller` and
[Inno Setup 6](https://jrsoftware.org/isinfo.php) (6.6 or newer; the per-user install is fine).

```bash
python packaging/build.py --variant cuda          # portable folder + smoke test + installer
python packaging/build.py --variant cuda --skip-installer
python packaging/build.py --variant cpu --python <cpu venv>\Scripts\python.exe
python packaging/build.py --status                 # is the existing build current?
```

The build writes to `%USERPROFILE%\aerolume-build\<variant>\` (set `AEROLUME_BUILD_ROOT` or
pass `--build-root` to change it) - never into the repository, because OneDrive and other sync
clients lock files while PyInstaller writes them. The finished installer is
`%USERPROFILE%\aerolume-build\cuda\installer\Aerolume-<version>-cuda-setup.exe`. It checks
that the interpreter's PyTorch matches the variant, generates the icon and version resources
from `src/aerolume/app/resources/aerolume.svg` and `aerolume.__version__`, runs PyInstaller
(`packaging/aerolume.spec`), copies the docs, validation scripts and licence files, launches
the built program headless (`packaging/smoke_test.py`: the self-test twice, the console
program, single-instance forwarding) and only then compiles `packaging/aerolume.iss`. The
compiler also writes `Aerolume-<version>-<variant>-manifest.txt`, a list of every file inside
the installer with its size and SHA-256; the build checks it against the app folder, so the
installer is verified to contain exactly the tested files without being run. A summary with
versions, sizes, timings and every smoke-test result is written to `build-report.json`.

**The build reads a snapshot, not your working tree.** PyInstaller reads the package for about
ten minutes; if you (or anyone sharing the tree) save a file meanwhile, the folder would mix two
versions. So the build first copies every input (the whole `aerolume` package with its data
files, `pyproject.toml`, the packaging scripts, the docs, validation scripts and licence files)
to `<build root>\<variant>\source\`, accepting the copy only when the tree did not change while
it was copied and every `.py` file compiles (a half-saved file does not); otherwise it waits five
seconds and tries again, six times at most. PyInstaller, the post-processing and the installer
then work from that copy alone, and afterwards every file of the bundled `aerolume` package is
compared byte for byte with the snapshot - the build stops if one differs. You can keep editing
while it runs. Aerolume's own package metadata (version, project links, requirements) is
generated from the snapshot's `pyproject.toml` too, rather than copied from the editable
install, which keeps whatever was true on the day `pip install -e` ran.

**Is this build current?** Every build records the snapshot's SHA-256 fingerprint in
`build-info.json` inside the app folder and in `build-report.json`.
`python packaging/build.py --status` compares it with your working tree and prints, for the
app folder and every installer it finds, `CURRENT` or the reasons it is not: another version, a
different fingerprint (naming the most recently changed input and, while the snapshot is still
there, exactly which files changed, were added or were removed since), or no fingerprint at all
(builds made before this check existed - shown as stale once any input is newer). A folder whose
smoke test failed is shown as `FAILED` with the checks that failed, and one built with
`--skip-smoke` as `UNTESTED`; neither is ever called current, even when its sources match. It
exits with code 0 only when the folder and an installer compiled from it both match. `--reuse-dist`
(compile an installer from an existing folder) refuses a folder that is not current unless you
add `--allow-stale`. When only the installer compile fails (an LZMA worker short of memory stops
with "terminated unexpectedly"), the build still saves its report with the error, so
`--reuse-dist` (with `--lzma-threads 2` if memory was the cause) compiles the installer in a few
minutes and adds to the same record instead of needing a second full build. A build exits with 0 when it still matches the tree at the end and with 3
when it was built correctly from its snapshot but the tree has moved on since; either way the
folder and installer are internally consistent. So a version number alone never passes for
"this installer contains today's code".

**A failed smoke test leaves no installer.** The build stops before the installer compiler
runs, exits with a non-zero code naming the failed checks, and deletes any installer in
`<variant>\installer\`. An older installer there was compiled from the folder this build
replaced, so keeping it would leave a file that looks like this build's result. A full build
also removes the old installer when it deletes the old app folder at the start, so an
interrupted build cannot leave one either. The build root holds only the build's own output;
nothing outside it is touched.

**Only the docs changed.** README, CHANGELOG, `docs\` and `validation\` are copied next to the
executable after PyInstaller and nothing is compiled from them. When they are the only inputs
that changed since the build's snapshot, `python packaging/build.py --refresh-docs` takes a new
snapshot and re-copies them. It then re-runs the smoke test and recompiles the installer, which
takes about a third of a full build (both are in the measurements below). It refuses, and
changes nothing, if any bundled input changed (a module, a data file in the package, `pyproject.toml` or a packaging script),
and it refuses a folder that failed its smoke test. `build-info.json` records each refresh under
`docs_refreshed` with the files it re-copied. This is how the release build's own measurements
below get into this file, which the app ships, without a second PyInstaller run.

**Why not `%LOCALAPPDATA%`.** Earlier builds went to `%LOCALAPPDATA%\aerolume-build`. When the
build runs inside a packaged (MSIX) Windows app - the Claude desktop app, some terminals and
IDEs - Windows silently redirects new folders under `AppData` to that app's private
`%LOCALAPPDATA%\Packages\<app>\LocalCache\Local\` store, so the installer existed for the
build but not in Explorer at the path the log printed. Only `AppData` is redirected, so the
profile folder is the same for every program. `--status` still lists builds left in the old
location (marked "legacy build root"), and whenever a path is redirected the build log and
`--status` print both spellings.

**Release measurements are generated, not typed.** Every run of `build.py` appends a record to
`runs` in `build-report.json`: the kind of run, the fingerprint it built, the smoke-test and
self-test times, the installer compile time and the installer's SHA-256, the machine, and the
average CPU use of the whole machine during the run. A full build starts a new history; a
docs refresh extends it, so the report of the final installer still holds the full build that
compiled its program. After every compile the build also writes
`Aerolume-<version>-<variant>-release-notes.md` beside the installer, with that installer's
SHA-256, its timings and every smoke-test and self-test result; that file goes on the release
page. `python packaging/build.py --write-measurements` copies the history into the block below
(and into the CHANGELOG's release-build paragraph when it carries the markers). It refuses a
build that failed or skipped its smoke test, has no installer, or whose program the working
tree no longer holds. A release is therefore: full build, `--write-measurements`,
`--refresh-docs`.

<!-- build-measurements:begin -->
**Measured release build: Aerolume 0.1.0, CUDA variant.** Every number in this block is copied from the `build-report.json` of the runs named in its column headings by `python packaging/build.py --write-measurements`; none is typed by hand.

* The program was compiled by the full build of snapshot `d7b30ff02f9f`, finished 2 Nov 2026, 03:59.
* The last run recorded here compiled `Aerolume-0.1.0-cuda-setup.exe`, SHA-256 `866bc41540343927a004880cebee2349109db6aa8e93731e6a7f703a3f10f0ed`.
* Machine: 13th Gen Intel(R) Core(TM) i7-13700F, 24 logical CPUs, 16 GB RAM, NVIDIA GeForce RTX 4060 Ti. Average CPU use of the whole machine during each run, the build included: full build `d7b30ff02f9f` 18 %. Times depend on load and on what is already in the file cache; sizes, file counts and hashes do not.

| step | full build `d7b30ff02f9f` |
|---|---|
| PyInstaller | 302 s |
| smoke test (self-test twice, console program, `doctor`, single instance) | 35.8 s |
| self-test of the built app, 14 checks: first / second launch | 20.1 s / 6.4 s |
| installer compile (lzma2/max) | 125 s |
| installer SHA-256 | `866bc4154034` |
| whole run | 8.4 min |

App folder 3.54 GB in 4,241 files; installer 1.94 GB. The installer's manifest lists 4,241 files, identical in size and SHA-256 to the tested app folder.

A *first launch* in a docs refresh or re-package starts with the app's libraries already in the file cache, so it is faster than the first start after installing, when Windows also reads 3.5 GB from disk and antivirus software scans it. A `-` means the run did not measure that step.

This page ships inside the installer, so it cannot describe the installer that contains it: writing these numbers changes the docs, and the release installer is then recompiled from the same program with `--refresh-docs`. That run's own figures and the SHA-256 of the installer you download are in `Aerolume-0.1.0-cuda-release-notes.md` beside the installer on the release page.
<!-- build-measurements:end -->

A development build on 24 October failed its self-test (its bundled metadata named a different
version than its code), which led to the
`FAILED` state and the no-installer rule described above. The development builds of 30 October,
made before the first release was numbered 0.1.0, a first 0.1.0 build on 1 November
(snapshot `533e0bc49fbc`, rebuilt after a fix to the Assistant's claims check) and a second one
the same day (snapshot `e7c21614f43d`, built before the interface took the website's look)
passed every check but were replaced by the build above; their reports are not in this history.
A third build that day (snapshot `a9a2c464abc1`) passed its smoke test, but its installer compile
stopped when an LZMA worker died ("terminated unexpectedly"); that run saved no report, which is
why a failed compile now saves one (see `--reuse-dist` above). A fourth (snapshot `b54186fba8f1`)
passed every check and was replaced after a fix to the names of crash logs (two in the same
millisecond overwrote each other). The rebuild after it was started while the app was open from
the build folder: Windows refused to delete the running program half-way through removing the
folder, so the build now lists the processes running from the folder and refuses before it
removes anything. The last of these, the full build of snapshot `7858260c37c5` on the evening of
1 November, passed every check too. It was replaced because the release check found that its
`licenses` folder lacked the LGPL and GPL texts for Qt, CPython's licences and notices for the
NVIDIA and Intel libraries in `torch\lib`. The build above adds them (THIRD_PARTY_NOTICES.md).
The release check of 2 November then found the licences of VTK's vendored libraries in
`vtk.libs` and of the third-party code inside Qt missing; the build now adds those too, and
refuses to finish when VTK brings a library whose licence it does not have.

### Building the macOS app

Aerolume has not been tested on macOS, and no `.app` has been built by the project: PyInstaller
cannot cross-build, and the release is made on Windows. The spec is ready for it, though. On a
Mac (Apple silicon or Intel, macOS 11 or later) with Python 3.11 or newer:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ai]" pyinstaller           # PyTorch for macOS: CPU (no CUDA)
pyinstaller packaging/aerolume.spec          # writes dist/Aerolume.app
```

What the spec does on a Mac (`packaging/aerolume.spec`, under `if IS_MAC`): the variant defaults
to `cpu`; the Windows version resources are skipped; a `BUNDLE` target wraps the folder into
`dist/Aerolume.app` with the bundle identifier `org.aerolume.Aerolume`, the version, the
`.aerolume` project folders as its document type, and the icon `aerolume.icns`, which
`buildlib.make_icns` renders from the same `aerolume.svg` as the Windows `.ico` (no `iconutil`
needed). `packaging/build.py` is Windows-only (it compiles the Inno Setup installer), so on a Mac
run PyInstaller directly as above, then check the bundle with
`dist/Aerolume.app/Contents/MacOS/Aerolume --selftest`. The app is not signed or notarised:
Gatekeeper will ask before the first start (Control-click, Open). Please report what you find.

## Troubleshooting

Logs are in `%APPDATA%\Aerolume\logs`: `crash-*.log` (one per unexpected error, with the
report the crash dialog shows), `aerolume.log` (output of the installed program), and
`native-crash-*.log` when the previous session ended inside native code (a graphics driver,
CUDA). Paths in these reports have your home folder replaced by `~`, so they can be attached
to an issue as they are.

**The repository or venv is in OneDrive.** Symptoms: `PermissionError` or "file in use" while
installing packages or building, random import errors, very high OneDrive memory use, and
CUDA reporting "out of memory" with an empty GPU (the system has run out of commit memory).
Move the venv and the build folder out of OneDrive (the build script refuses to build inside
it), or pause syncing while you work.

**"CUDA not available" / simulations run on the CPU.** Check the device in the status bar or run
`aerolume-cli doctor`. Install the current NVIDIA driver (GeForce or Studio driver; the CUDA
toolkit is *not* needed, the libraries are bundled). Laptops with two GPUs: set Aerolume to
the high-performance NVIDIA GPU in **Settings ▸ System ▸ Display ▸ Graphics**. A driver older
than the bundled CUDA runtime makes PyTorch fall back to the CPU; the self-test's `torch` line
says which device it used.

**The 3D view says "3D view unavailable" or is black.** VTK needs OpenGL 3.2. Remote Desktop
sessions, some virtual machines and the Windows "Basic Display Adapter" provide only OpenGL
1.1. Update the graphics driver; over Remote Desktop, start Aerolume before connecting or use
a remote-access tool that renders on the host GPU instead of Remote Desktop. As a last
resort set the environment variable `AEROLUME_VTK_BACKEND=osmesa` to render in software, if an
`osmesa.dll` is available. 2D work is unaffected.

**SmartScreen blocks the installer.** The installer is not code-signed; see step 1 above.
Antivirus software may also scan the 3.5 GB of libraries on the first start, which is why the
first launch after installing takes noticeably longer than later ones.

**A second window does not open when I double-click a project.** By design: the project opens
in the running Aerolume. If that window is hung, close it from the Task Manager; the next start
recovers (the instance lock is released when the process ends).

**Scripts fail with `ModuleNotFoundError` in the installed app.** The installed app bundles the
whole standard library, NumPy, SciPy, Matplotlib and Pillow, but not packages from your own
Python installation. Use a source install for scripts that need other packages.
