"""Script execution inside the desktop app: the engine behind the Script console (DESIGN.md 8).

Three pieces, none of them a widget (the console is
:mod:`aerolume.app.widgets.script_console`; the AI assistant reuses the same
engine for its approved ``run_python`` tool):

* :class:`AppHost` -- the :class:`aerolume.api.Host` of the app. A script runs
  on a worker thread, but the project it edits is the one every page shows,
  so each change (add a design, record a status, a new result) is carried out
  **on the GUI thread through** ``AppState`` -- the Project tree and the pages
  refresh at once, exactly as for a change made by hand -- and every
  simulation runs **through the JobManager**, so a script never competes with
  a GUI run for the GPU, the Simulate page shows its live view, and the
  JobManager's save, retry and cancel rules apply unchanged.
* :class:`ScriptEngine` -- runs Python source on a worker thread with ``al``
  (the ``aerolume`` package), ``project`` (an :class:`aerolume.api.Project`
  bound to the open project through an :class:`AppHost`, or None), ``log`` and
  ``show`` bound; captures stdout / stderr; reports progress; stops on request
  and never blocks the GUI. The namespace persists between runs of one engine
  (as in an IPython console) until :meth:`ScriptEngine.reset`.
* Snippets (:data:`SNIPPETS`) -- ready, runnable examples the console offers
  in its menu; ``examples/scripts/*.py`` holds the same text (a test keeps them
  identical) -- and the per-project script folder helpers
  (:func:`scripts_dir`, :func:`list_scripts`).

How Stop works
--------------

Stop first sets the run's cancel event, which every API call of the script
uses by default (``aerolume.api.context``): a running simulation stops at its
next step block and keeps its partial result, marked "cancelled", and the API
call raises :class:`aerolume.api.Cancelled`, which ends the script. Plain
Python that never calls the API (a long loop) would not notice, so after
:data:`STOP_GRACE_S` -- or at once on a second Stop -- a ``KeyboardInterrupt``
is raised inside the script's thread (``PyThreadState_SetAsyncExc``), and
again every :data:`STOP_RETRY_S` while it keeps running. It is only ever
raised while the script's own code runs, never in the engine's clean-up. A
thread stuck in one long native call (a single huge numpy operation) notices
when that call returns.

Output routing
--------------

``sys.stdout`` / ``sys.stderr`` are process-wide, so while scripts run they are
replaced by a router that sends writes made *on a script's thread* to that
script and everything else to the original stream (the terminal, pytest's
capture). Output of threads a script starts itself goes to the terminal.
"""
from __future__ import annotations

import builtins
import contextlib
import ctypes
import io
import linecache
import queue
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from .. import api

__all__ = ["MAX_OUTPUT_CHARS", "SNIPPETS", "STOP_GRACE_S", "STOP_RETRY_S", "AppHost", "ScriptEngine",
           "ScriptOutcome", "Snippet", "list_scripts", "scripts_dir", "snippet"]

#: Seconds between the first Stop (cancel event) and the first KeyboardInterrupt.
STOP_GRACE_S = 1.5
#: Seconds between further KeyboardInterrupts while the script keeps running.
STOP_RETRY_S = 2.0
#: Interrupt attempts before the engine reports that the script does not respond.
STOP_MAX_TRIES = 6
#: Captured output kept for :attr:`ScriptOutcome.output` (the console shows everything).
MAX_OUTPUT_CHARS = 1_000_000
#: Messages-dock source of script lines.
LOG_SOURCE = "Script"


# ===========================================================================
# running callables on the GUI thread
# ===========================================================================

class _Call:
    """One callable to run on the GUI thread, with its outcome."""

    __slots__ = ("done", "error", "fn", "value")

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.fn = fn
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None

    def execute(self) -> None:
        try:
            self.value = self.fn()
        except BaseException as exc:  # noqa: BLE001 - handed back to the waiting thread
            self.error = exc
        finally:
            self.done.set()


class _Invoker(QObject):
    """Lives on the GUI thread; runs posted callables there (queued signal delivery)."""

    request = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.request.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, call: _Call) -> None:
        call.execute()


# ===========================================================================
# the app host
# ===========================================================================

class AppHost(api.Host):
    """An :class:`aerolume.api.Host` that edits the app's open project through ``AppState``.

    ``AppHost(state, jobs=None)`` must be made on the GUI thread. Every project
    change runs on the GUI thread (blocking the calling worker until it is
    done, so the API stays synchronous); ``run_simulation`` submits the
    simulation to ``jobs`` (a :class:`~aerolume.app.jobs.JobManager`) and waits
    for it, forwarding progress to the calling thread and cancelling the job
    when the cancel event is set or the waiting thread is interrupted. Without
    ``jobs`` the API runs simulations on the calling thread and the host only
    tells the pages about the new result.

    ``touched`` lists the simulations the script added or ran (the console
    reports them; the assistant links them). :meth:`close` makes every pending
    and later call raise :class:`aerolume.api.Cancelled` (window closing).
    """

    live_view = True

    def __init__(self, state: Any, jobs: Any = None) -> None:
        self.state = state
        self.jobs = jobs
        self._invoker = _Invoker()
        # The GUI thread, recorded the Python way. Asking Qt instead (``QThread.currentThread()
        # is self._invoker.thread()`` from the script's worker thread) made shiboken tie the
        # main thread's QThread wrapper to the invoker's wrapper; when an old host was later
        # garbage-collected, the application's main QThread was deleted with it, and the next
        # style-sheet icon (a spin box arrow) crashed inside Qt's plugin loader. Measured: the
        # fast suite died with an access violation after the assistant tests; gc.collect() in a
        # test teardown fired the main QThread's ``destroyed`` signal.
        self._gui_thread = threading.get_ident()
        self._closed = False
        self.touched: list[str] = []
        #: simulations whose stored result a run returned instead of running (list.append is thread-safe)
        self.reused: list[str] = []

    # -- threads ------------------------------------------------------------------
    def _on_gui_thread(self) -> bool:
        return threading.get_ident() == self._gui_thread

    def call(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the GUI thread and return its value (re-raising its exception)."""
        if self._on_gui_thread():
            return fn()
        if self._closed:
            raise api.Cancelled("The script console was closed.")
        job = _Call(fn)
        self._invoker.request.emit(job)
        while not job.done.wait(0.05):
            if self._closed:
                raise api.Cancelled("The script console was closed.")
        if job.error is not None:
            raise job.error
        return job.value

    def post(self, fn: Callable[[], Any]) -> None:
        """Run ``fn`` on the GUI thread without waiting (fire and forget)."""
        if self._on_gui_thread():
            fn()
        elif not self._closed:
            self._invoker.request.emit(_Call(fn))

    def close(self) -> None:
        self._closed = True

    # -- the Host protocol -----------------------------------------------------------
    def defaults(self) -> dict[str, str]:
        def read() -> dict[str, str]:
            from . import settings as app_settings
            s = app_settings.get()
            return {"device": s.device, "precision": s.precision}
        try:
            return self.call(read)
        except api.AerolumeError:
            raise
        except Exception:  # noqa: BLE001 - preferences unreadable: the planner's own choice
            return {"device": "auto", "precision": "auto"}

    def check(self, store: Any) -> None:
        if self._closed:
            raise api.Cancelled("The script console was closed.")
        if self.state.project is not store:
            raise api.AerolumeError("The project was closed, or another one was opened, while the script ran; "
                                    "the script's project is no longer the open one.")

    def _edit(self, store: Any, fn: Callable[[], Any]) -> Any:
        def run() -> Any:
            self.check(store)
            return fn()
        return self.call(run)

    def add_design(self, store: Any, design: Any) -> None:
        self._edit(store, lambda: self.state.add_design(design, select=False))

    def update_design(self, store: Any, design: Any) -> None:
        self._edit(store, lambda: self.state.update_design(design))

    def remove_design(self, store: Any, design_id: str) -> None:
        self._edit(store, lambda: self.state.remove_design(design_id))

    def add_simulation(self, store: Any, sim: Any) -> None:
        self.touched.append(sim.id)
        self._edit(store, lambda: self.state.add_simulation(sim, select=False))

    def update_simulation(self, store: Any, sim: Any) -> None:
        self._edit(store, lambda: self.state.update_simulation(sim))

    def remove_simulation(self, store: Any, sim_id: str) -> None:
        self._edit(store, lambda: self.state.remove_simulation(sim_id))

    def result_saved(self, store: Any, sim_id: str) -> None:
        self._edit(store, lambda: self.state.notify_result(sim_id))

    def report_written(self, store: Any, path: Path) -> None:
        self._edit(store, lambda: self.state.reportsChanged.emit())

    def result_reused(self, store: Any, sim_id: str) -> None:
        """Remember a reused result for the console's end line (no GUI work: nothing changed)."""
        self.reused.append(sim_id)

    def show(self, store: Any, sim_id: str) -> None:
        def go() -> None:
            self.state.select_simulation(sim_id)
            self.state.navigate("results")
        self._edit(store, go)

    def run_simulation(self, store: Any, sim_id: str, *, progress: api.ProgressFn,
                       cancel: threading.Event | None) -> tuple[Any, str] | None:
        """Run through the JobManager (see the class docstring); None without one."""
        jobs = self.jobs
        if jobs is None:
            return None
        if sim_id not in self.touched:
            self.touched.append(sim_id)
        done = threading.Event()
        box: dict[str, Any] = {}
        reports: queue.SimpleQueue = queue.SimpleQueue()

        # these run on the GUI thread (the JobManager emits there)
        def on_progress(sid: str, fraction: float, message: str) -> None:
            if sid == sim_id:
                reports.put((fraction, message))

        def on_finished(sid: str, result: Any) -> None:
            if sid == sim_id:
                box.update(result=result, status="done")
                done.set()

        def on_failed(sid: str, message: str) -> None:
            if sid == sim_id:
                box["error"] = message
                done.set()

        def on_cancelled(sid: str) -> None:
            if sid == sim_id:
                box.update(result=jobs.cached_result(sid), status="cancelled")
                done.set()

        links = ((jobs.jobProgress, on_progress), (jobs.jobFinished, on_finished),
                 (jobs.jobFailed, on_failed), (jobs.jobCancelled, on_cancelled))

        def start() -> bool:
            self.check(store)
            for sig, fn in links:
                sig.connect(fn)
            jobs.submit(sim_id)
            return sim_id in jobs.queued_ids()

        def unlink() -> None:
            for sig, fn in links:
                with contextlib.suppress(RuntimeError, TypeError):
                    sig.disconnect(fn)

        def drain() -> None:
            while True:
                try:
                    f, m = reports.get_nowait()
                except queue.Empty:
                    return
                progress(f, m)

        try:
            if self.call(start):
                running = jobs.running_id()
                progress(0.0, "Waiting in the queue" + (" behind the running simulation" if running else ""))
            asked = False
            while not done.wait(0.05):
                drain()
                if not asked and ((cancel is not None and cancel.is_set()) or self._closed):
                    asked = True
                    self.post(lambda: jobs.cancel(sim_id))
            drain()
        except BaseException:
            # Stop (KeyboardInterrupt) or a closed console: never leave a run going for a script
            # that is no longer waiting for it
            if not done.is_set():
                self.post(lambda: jobs.cancel(sim_id))
            raise
        finally:
            self.post(unlink)
        if "error" in box:
            sim = store.simulations.get(sim_id)
            raise api.RunError(f"“{getattr(sim, 'name', sim_id)}” failed: {box['error']}")
        return box.get("result"), box.get("status", "done")


# ===========================================================================
# stdout / stderr routing
# ===========================================================================

class _Router(io.TextIOBase):
    """Stands in for sys.stdout / sys.stderr: writes from a script thread go to its sink."""

    def __init__(self, fallback: Any, name: str) -> None:
        super().__init__()
        self.fallback = fallback
        self.name = name

    def write(self, text: str) -> int:
        sink = _SINKS.get(threading.get_ident())
        if sink is not None:
            sink(self.name, str(text))
            return len(text)
        if self.fallback is not None:
            return self.fallback.write(text)
        return len(text)

    def flush(self) -> None:
        if self.fallback is not None:
            with contextlib.suppress(Exception):
                self.fallback.flush()

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:      # type: ignore[override]
        return "utf-8"

    def writable(self) -> bool:
        return True


_SINKS: dict[int, Callable[[str, str], None]] = {}
_ROUTE_LOCK = threading.Lock()
_ROUTE_COUNT = 0
_SAVED: dict[str, Any] = {}


def _route_on(tid: int, sink: Callable[[str, str], None]) -> None:
    global _ROUTE_COUNT
    with _ROUTE_LOCK:
        _SINKS[tid] = sink
        if _ROUTE_COUNT == 0:
            _SAVED["stdout"], _SAVED["stderr"] = sys.stdout, sys.stderr
            sys.stdout = _Router(sys.stdout, "stdout")
            sys.stderr = _Router(sys.stderr, "stderr")
        _ROUTE_COUNT += 1


def _route_off(tid: int) -> None:
    global _ROUTE_COUNT
    with _ROUTE_LOCK:
        _SINKS.pop(tid, None)
        _ROUTE_COUNT = max(0, _ROUTE_COUNT - 1)
        if _ROUTE_COUNT == 0:
            # restore only what is still ours: someone (pytest) may have swapped them since
            if isinstance(sys.stdout, _Router):
                sys.stdout = _SAVED.get("stdout", sys.__stdout__)
            if isinstance(sys.stderr, _Router):
                sys.stderr = _SAVED.get("stderr", sys.__stderr__)


def _pyplot_off_screen() -> None:
    """Make ``matplotlib.pyplot`` in a script draw off-screen (Agg) instead of opening windows.

    A pyplot window is a Qt widget; created from a script's worker thread it would crash the
    application. The app itself never uses pyplot (DESIGN.md 2: Figure objects only), so
    choosing Agg before pyplot is first imported changes nothing else; scripts' pyplot figures
    are shown inline when they end (:meth:`ScriptEngine._show_pyplot_figures`).
    """
    if "matplotlib.pyplot" in sys.modules:
        return
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")


def _async_raise(tid: int, exc: type[BaseException] | None) -> bool:
    """Raise ``exc`` in thread ``tid`` at its next bytecode (None clears a pending one)."""
    obj = ctypes.py_object(exc) if exc is not None else None
    n = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), obj)
    if n > 1:                                   # cannot happen for a valid id; undo to be safe
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    return n == 1


# ===========================================================================
# the engine
# ===========================================================================

@dataclass
class ScriptOutcome:
    """How a script run ended.

    ``status`` is "finished", "failed" or "stopped"; ``error`` one line for a
    status bar ("InputError: speed must be positive, got -1."), ``line`` the
    script line it failed on, ``traceback`` the traceback through the script's
    own lines (library frames left out and counted), ``output`` everything it
    printed (capped at :data:`MAX_OUTPUT_CHARS`), ``figures`` the PNGs it
    showed, ``sim_ids`` the simulations it added or ran, ``reused_ids`` those whose
    stored result a run returned without running (an identical run was already
    complete), so the console never reports a reused result as a fresh run.
    """
    status: str
    seconds: float
    error: str = ""
    line: int | None = None
    traceback: str = ""
    output: str = ""
    figures: list[bytes] = field(default_factory=list)
    sim_ids: list[str] = field(default_factory=list)
    filename: str = ""
    reused_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "finished"


class _Mailbox:
    """Latest-value slot shared by a worker and the GUI (progress reports)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Any = None
        self._full = False

    def put(self, item: Any) -> bool:
        with self._lock:
            was_empty = not self._full
            self._item, self._full = item, True
            return was_empty

    def take(self) -> Any:
        with self._lock:
            item, self._item, self._full = self._item, None, False
            return item


class _Bridge(QObject):
    """Worker -> GUI signals (queued)."""
    output = Signal()
    progress = Signal()
    figure = Signal(object)
    log = Signal(str, str)
    done = Signal(object)


class ScriptEngine(QObject):
    """Runs scripts on a worker thread against the app's open project.

    ``ScriptEngine(state, jobs=None, parent=None)``; ``state`` is the
    :class:`~aerolume.app.state.AppState`, ``jobs`` the JobManager scripts' runs
    go through (None: runs execute on the script's thread). Signals (all
    delivered on the GUI thread):

    * ``started()``
    * ``output(text: str, stream: str)`` -- stream "stdout", "stderr", "log" or
      "info" (the engine's own lines); consecutive writes are batched
    * ``progress(fraction: float, message: str)`` -- from the API calls of the script
    * ``figure(png: bytes)`` -- a figure the script passed to ``show()``
    * ``finished(outcome: ScriptOutcome)``
    * ``stateChanged(running: bool)``

    :meth:`run` returns False when a script is already running or the source
    does not compile (the outcome is still reported through ``finished``).
    """

    started = Signal()
    output = Signal(str, str)
    progress = Signal(float, str)
    figure = Signal(object)
    finished = Signal(object)
    stateChanged = Signal(bool)

    def __init__(self, state: Any, jobs: Any = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.jobs = jobs
        self.namespace: dict[str, Any] = {}
        self._bridge = _Bridge()
        q = Qt.ConnectionType.QueuedConnection
        self._bridge.output.connect(self._flush_output, q)
        self._bridge.progress.connect(self._flush_progress, q)
        self._bridge.figure.connect(self._on_figure, q)
        self._bridge.log.connect(self._on_log, q)
        self._bridge.done.connect(self._on_done, q)
        self._out_lock = threading.Lock()
        self._out: list[tuple[str, str]] = []
        self._out_pending = False
        self._captured: list[str] = []
        self._captured_len = 0
        self._figures: list[bytes] = []
        self._progress = _Mailbox()
        self._thread: threading.Thread | None = None
        self._tid: int | None = None
        self._cancel = threading.Event()
        self._host: AppHost | None = None
        self._user_lock = threading.Lock()
        self._in_user = False
        self._stops = 0
        self._tries = 0
        self._started_at = 0.0
        self._filename = ""
        self._stop_timer = QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._interrupt)

    # -- state -------------------------------------------------------------------------
    def is_running(self) -> bool:
        return self._thread is not None

    def elapsed(self) -> float:
        return time.monotonic() - self._started_at if self.is_running() else 0.0

    def reset(self) -> None:
        """Forget the variables kept between runs (not while a script runs)."""
        if not self.is_running():
            self.namespace.clear()

    # -- running -------------------------------------------------------------------------
    def run(self, source: str, *, filename: str | None = None, argv: Sequence[str] | None = None) -> bool:
        """Start ``source`` on a worker thread; see the class docstring."""
        if self.is_running():
            return False
        self._filename = filename or "<script>"
        text = str(source)
        # registered so tracebacks show the script's lines even for an unsaved buffer
        linecache.cache[self._filename] = (len(text), None, text.splitlines(True), self._filename)
        self._started_at = time.monotonic()
        try:
            code = compile(text, self._filename, "exec")
        except SyntaxError as exc:
            msg = f"SyntaxError: {exc.msg}"
            tb = "".join(traceback.format_exception_only(type(exc), exc))
            self.started.emit()
            self.output.emit(tb, "stderr")
            self.finished.emit(ScriptOutcome("failed", 0.0, msg, exc.lineno, tb, tb, filename=self._filename))
            return False
        self._cancel = threading.Event()
        self._stops = self._tries = 0
        self._captured, self._captured_len, self._figures = [], 0, []
        _pyplot_off_screen()
        project = self.state.project
        self._host = AppHost(self.state, self.jobs) if project is not None else None
        bound = api.Project(project, host=self._host) if project is not None else None
        import aerolume
        self.namespace.update({"__name__": "__main__", "__file__": self._filename, "__builtins__": builtins,
                               "al": aerolume, "project": bound, "log": self._log, "show": self._show,
                               "args": list(argv or [])})
        self._thread = threading.Thread(target=self._work, args=(code, bound), name="aerolume-script", daemon=True)
        self.started.emit()
        self.stateChanged.emit(True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the running script: cancel its runs, then interrupt it (module docstring)."""
        if not self.is_running():
            return
        self._stops += 1
        self._cancel.set()
        if self._stops == 1:
            self._emit_line("Stopping: running simulations keep their partial results…", "info")
            self._stop_timer.start(int(1000 * STOP_GRACE_S))
        else:
            self._interrupt()

    def _interrupt(self) -> None:
        if not self.is_running() or self._tid is None:
            return
        with self._user_lock:
            if self._in_user:
                _async_raise(self._tid, KeyboardInterrupt)
        self._tries += 1
        if self._tries == STOP_MAX_TRIES:
            self._emit_line("The script does not respond to Stop: it is inside a long call that cannot be "
                            "interrupted. It stops when that call returns.", "stderr")
        self._stop_timer.start(int(1000 * STOP_RETRY_S))

    def wait(self, timeout: float = 30.0) -> bool:
        """Process events until the running script has finished (tests, shutdown)."""
        from PySide6.QtCore import QCoreApplication
        end = time.monotonic() + timeout
        while self.is_running() and time.monotonic() < end:
            QCoreApplication.processEvents()
            time.sleep(0.005)
        QCoreApplication.processEvents()
        return not self.is_running()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop a running script and stop waiting for the GUI (window closing)."""
        if not self.is_running():
            return
        self._cancel.set()
        if self._host is not None:
            self._host.close()
        self._interrupt()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- the worker thread --------------------------------------------------------------
    def _work(self, code: Any, bound: Any) -> None:
        tid = threading.get_ident()
        self._tid = tid
        _route_on(tid, self._write)
        status, error, line, tb_text = "finished", "", None, ""
        stop_exc: BaseException | None = None
        try:
            with api.context(cancel=self._cancel, progress=self._report, show=self._show, project=bound):
                try:
                    with self._user_lock:
                        self._in_user = True
                    exec(code, self.namespace)       # noqa: S102 - running the user's script is the point
                    self._show_pyplot_figures()
                except SystemExit as exc:
                    if exc.code not in (None, 0):
                        status, error = "failed", (str(exc.code) if not isinstance(exc.code, int)
                                                   else f"The script exited with code {exc.code}.")
                except (KeyboardInterrupt, api.Cancelled) as exc:
                    status, stop_exc = "stopped", exc
                except BaseException as exc:  # noqa: BLE001 - every failure is the user's to read
                    status = "failed"
                    error, line, tb_text = self._describe(exc)
                finally:
                    with self._user_lock:
                        self._in_user = False
                    _async_raise(tid, None)      # an interrupt that arrived too late is dropped
        except KeyboardInterrupt:                # landed in the clean-up above: the run was stopped
            status = "stopped"
        finally:
            _route_off(tid)
        if status == "stopped" and not self._cancel.is_set():
            # not our Stop: the script's own cancel event or its own KeyboardInterrupt
            status = "failed"
            error, line, tb_text = self._describe(stop_exc) if stop_exc is not None else ("KeyboardInterrupt", None, "")
        touched = list(dict.fromkeys(self._host.touched)) if self._host is not None else []
        reused = list(dict.fromkeys(self._host.reused)) if self._host is not None else []
        outcome = ScriptOutcome(status, time.monotonic() - self._started_at, error, line, tb_text,
                                "".join(self._captured), list(self._figures), touched, self._filename,
                                reused_ids=reused)
        self._bridge.done.emit(outcome)

    def _show_pyplot_figures(self) -> None:
        """Figures a script drew with ``matplotlib.pyplot`` appear inline when it ends (as in a
        notebook) and are closed; pyplot runs on the non-interactive Agg backend here
        (:func:`_pyplot_off_screen`), so ``plt.show()`` itself opens nothing."""
        plt = sys.modules.get("matplotlib.pyplot")
        if plt is None:
            return
        for num in list(plt.get_fignums()):
            self._show(plt.figure(num))
        plt.close("all")

    def _describe(self, exc: BaseException) -> tuple[str, int | None, str]:
        """(one-line error, script line, traceback through the script's own frames)."""
        return api.script_traceback(exc, self._filename)

    def _write(self, stream: str, text: str) -> None:
        """Sink of the stdout / stderr router (worker thread)."""
        if not text:
            return
        if self._captured_len < MAX_OUTPUT_CHARS:
            self._captured.append(text)
            self._captured_len += len(text)
        with self._out_lock:
            self._out.append((stream, text))
            wake = not self._out_pending
            self._out_pending = True
        if wake:
            self._bridge.output.emit()

    def _emit_line(self, text: str, stream: str) -> None:
        self._write(stream, text if text.endswith("\n") else text + "\n")

    def _report(self, fraction: float, message: str) -> None:
        if self._progress.put((float(fraction), str(message))):
            self._bridge.progress.emit()

    def _log(self, text: Any, level: str = "info") -> None:
        """``log(text, level="info")`` in a script: a line in the output and in the Messages dock."""
        text = str(text)
        lv = level if level in ("info", "warning", "error") else "info"
        self._emit_line(text if lv == "info" else f"{lv.capitalize()}: {text}", "log")
        self._bridge.log.emit(text, lv)

    def _show(self, obj: Any) -> Any:
        """``show(obj)`` in a script: figures inline in the console, anything else printed."""
        png = None
        if api._is_figure(obj):
            from ..viz.figures import figure_to_png
            png = figure_to_png(obj, dpi=160)      # sharp at up to ~1.7x device pixel ratio
        elif isinstance(obj, (str, Path)) and str(obj).lower().endswith(".png") and Path(obj).is_file():
            png = Path(obj).read_bytes()
        if png is not None:
            self._figures.append(png)
            self._bridge.figure.emit(png)
        else:
            print(obj)
        return obj

    # -- GUI side -------------------------------------------------------------------------
    def _flush_output(self) -> None:
        with self._out_lock:
            items, self._out, self._out_pending = self._out, [], False
        merged: list[tuple[str, str]] = []
        for stream, text in items:
            if merged and merged[-1][0] == stream:
                merged[-1] = (stream, merged[-1][1] + text)
            else:
                merged.append((stream, text))
        for stream, text in merged:
            self.output.emit(text, stream)

    def _flush_progress(self) -> None:
        item = self._progress.take()
        if item is not None:
            self.progress.emit(*item)

    def _on_figure(self, png: bytes) -> None:
        self._flush_output()            # the figure follows the text printed before it
        self.figure.emit(png)

    def _on_log(self, text: str, level: str) -> None:
        with contextlib.suppress(Exception):
            self.state.log(text, level, LOG_SOURCE)

    def _on_done(self, outcome: ScriptOutcome) -> None:
        self._stop_timer.stop()
        thread, self._thread, self._tid = self._thread, None, None
        if thread is not None:
            thread.join(1.0)
        self._flush_output()
        self._flush_progress()
        # changes went through AppState already; this catches edits a script made to the store
        # directly (project.raw) so no page keeps showing a stale tree
        if self.state.project is not None and (outcome.sim_ids or outcome.status != "finished"):
            with contextlib.suppress(Exception):
                self.state.designsChanged.emit()
                self.state.simulationsChanged.emit()
        if outcome.status == "failed" and outcome.error:
            where = f" (line {outcome.line})" if outcome.line else ""
            with contextlib.suppress(Exception):
                self.state.log(f"Script failed{where}: {outcome.error}", "error", LOG_SOURCE)
        self.stateChanged.emit(False)
        self.finished.emit(outcome)


# ===========================================================================
# saved scripts
# ===========================================================================

def scripts_dir(project: Any, *, create: bool = False) -> Path | None:
    """``<project>/scripts`` (None without a project); ``create`` makes the folder.

    The store's own :attr:`aerolume.project.Project.scripts_dir` creates the
    folder whenever it is read, so it is only used with ``create``: listing the
    scripts of a project must not leave an empty folder in every project the
    user merely opened.
    """
    if project is None:
        return None
    if create:
        return Path(project.scripts_dir)
    return Path(project.path) / "scripts"


def list_scripts(project: Any) -> list[Path]:
    """The project's saved scripts (``scripts/*.py``), by name."""
    d = scripts_dir(project)
    if d is None or not d.is_dir():
        return []
    return sorted((p for p in d.glob("*.py") if p.is_file()), key=lambda p: p.name.lower())


# ===========================================================================
# snippets (examples/scripts/<key>.py holds the same text)
# ===========================================================================

@dataclass(frozen=True)
class Snippet:
    """A ready example: ``key`` (also the file name in examples/scripts), menu ``title``,
    one-line ``description`` (tooltip) and the ``code``."""
    key: str
    title: str
    description: str
    code: str


_PREAMBLE = '''import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()
'''

SNIPPETS: tuple[Snippet, ...] = (
    Snippet("polar_sweep", "Airfoil polars at three Reynolds numbers",
            "Viscous-inviscid polars of a NACA 2412 at Re 2×10⁵, 5×10⁵ and 10⁶ (about a minute).",
            '"""Viscous-inviscid polars of a NACA 2412 at three Reynolds numbers (about a minute)."""\n'
            + _PREAMBLE + '''
airfoil = project.add_naca("2412", chord=1.0)
polars = project.sweep(airfoil, analysis="polar", re=[2e5, 5e5, 1e6], alpha=(-4, 14, 1))
print(polars)                                  # Cl,max and best L/D per Reynolds number

best = polars.best("ld_max")
print(f"Best L/D {best.metrics['ld_max']:.1f} at Re {best.re_sim:.3g}, alpha {best.metrics['alpha_ld_max']:g} deg")
al.show(polars.plot("alpha", "cl"))            # lift curves, one per Reynolds number
al.show(polars.plot("cd", "cl"))               # drag polars
'''),
    Snippet("alpha_sweep", "Angle-of-attack sweep (CFD) with a plot",
            "2D CFD of a NACA 0012 at four angles, Draft quality; lift and drag with error bars.",
            '"""2D CFD of a NACA 0012 at four angles (Draft quality): lift and drag with error bars."""\n'
            + _PREAMBLE + '''
airfoil = project.add_naca("0012")
# Airfoils start, as in the app, in the capped-Reynolds mode: no affordable 2D grid resolves the
# boundary layer at Re 2e6, so the run is made at a low Reynolds number and the trends are what to
# read (re_sim against re_real below). project.run_polar(airfoil, re=2e6) gives real-Re numbers.
sweep = project.sweep(airfoil, alpha=[0, 2, 4, 6], speed=30, quality="draft")
print(sweep)

al.show(sweep.plot("alpha", ["cl", "cd"]))     # error bars: +-1 standard error of the mean
best = sweep.best("ld")
print(f"Best L/D {best.ld:.2f} at alpha {best.alpha:g} deg "
      f"(Re {best.re_sim:.3g} simulated, {best.re_real:.3g} real)")
'''),
    Snippet("compare_designs", "Compare two car shapes",
            "Sedan against fastback at 30 m/s on a moving road (2D CFD, Draft): the Compare table and bars.",
            '"""Sedan against fastback at 30 m/s on a moving road (2D CFD, Draft quality)."""\n'
            + _PREAMBLE + '''
sedan = project.add_template("sedan")
fastback = project.add_template("fastback")
runs = [project.run_cfd(design, speed=30) for design in (sedan, fastback)]

comparison = project.compare(runs)             # the Compare page's table, with significance
print(comparison)
al.show(comparison.plot(["cd", "cl"]))
'''),
    Snippet("mesh_3d", "Import a mesh and run 3D CFD",
            "Import an STL / OBJ / PLY file (a sphere at Re 100 when none is given), plan, run a Draft 3D "
            "simulation and show the surface pressure, 3D streamlines and the smoke around the body.",
            '"""Import a mesh, run a Draft 3D simulation, and look at the flow around the body.\n\n'
            'Set MESH to your STL, OBJ or PLY file. Without one, a 0.2 m sphere runs at Re 100,\n'
            'a validated case with a known answer, so the first run gives a number to check.\n"""\n'
            + _PREAMBLE + '''from pathlib import Path

MESH = Path("my_car.stl")                      # <- your mesh file
SPEED = 30.0                                   # m/s, for your mesh

if MESH.is_file():
    body = project.import_mesh(MESH, units="auto", kind="vehicle")
    flow = {"speed": SPEED}
else:
    # Re 100 is inside the solver's validated range (Re 20 to 300): the standard drag
    # curve gives CD 1.09, and the Draft grid (20 cells across the sphere) reads about 1.16.
    print(f"{MESH} not found: using a 0.2 m sphere at Re 100 instead.")
    body = project.add_sphere(0.2)
    flow = {"re": 100}                         # the speed follows from the diameter

print(project.plan(body, **flow))              # cells, estimated run time, planner warnings
result = project.run_cfd(body, quality="draft", **flow)
print(result)

al.show(result.plot("surface", field="cp"))    # pressure on the surface
al.show(result.plot("streamlines"))            # 3D streamlines through the flow around it
try:
    al.show(result.plot("smoke"))              # the smoke cloud (when the run recorded it)
except al.AerolumeError as exc:
    print(exc)
'''),
    Snippet("wing_taper", "Wing taper study",
            "Wings of equal span and area with taper 0.3 to 1.0 (vortex lattice): span efficiency and lift slope.",
            '"""Wings of equal span and area, taper 0.3 to 1.0 (vortex lattice, seconds)."""\n'
            + _PREAMBLE + '''
SPAN, AREA = 2.0, 0.48                         # m, m2 (aspect ratio 8.3)


def wing(taper):
    root = 2 * AREA / (SPAN * (1 + taper))
    return project.add_wing(span=SPAN, root_chord=root, tip_chord=root * taper, airfoil="2412")


study = project.sweep(wing, taper=[0.3, 0.45, 0.6, 0.8, 1.0], speed=20, alpha=(-2, 10, 1), viscous=False)
print(study)

al.show(study.plot("taper", ["oswald", "cl_alpha_per_deg"]))
best = study.best("oswald")
print(f"Highest span efficiency: {best.design.name} (e = {best.metrics['oswald']:.3f})")
'''),
    Snippet("flow_pictures", "3D streamlines, smoke and vortices of the latest result",
            "Pictures of the newest 3D CFD result in the project: streamlines, smoke, vortex cores, pressure.",
            '"""Pictures of the newest 3D CFD result in this project: streamlines, smoke, vortices."""\n'
            + _PREAMBLE + '''
results = [r for r in project.results() if r.kind == "cfd3d"]
if not results:
    print("No 3D CFD result in this project yet: run the snippet 'Import a mesh and run 3D CFD' first.")
else:
    result = results[-1]
    print(result)
    for kind in ("streamlines", "smoke", "vortices"):
        try:
            al.show(result.plot(kind))
        except al.AerolumeError as exc:
            print(f"{kind}: {exc}")
    result.export(project.path / "exports" / "surface.vtp")    # for ParaView
'''),
    Snippet("heated_cavity", "Natural convection in a heated cavity",
            "Air in a square box with a hot and a cold wall at Ra 10⁴ (2D CFD, Draft): the mean Nusselt number "
            "against de Vahl Davis (1983).",
            '"""Natural convection in a heated square cavity at Ra 1e4, checked against de Vahl Davis (1983)."""\n'
            + _PREAMBLE + '''
case = al.cavity_case(ra=1e4)                  # air (Pr 0.71) in a 10 cm box: hot left wall, cold right wall
print(case)
print(*case.notes, sep="\\n")                   # the wall temperatures that give Ra 1e4, the free-fall speed
print(project.plan(case))                      # grid, time step, Ra, Pr, Gr and the estimated time
result = project.run_case(case, quality="draft")
print(result)

check = case.check(result)                     # mean Nusselt number of the hot wall against the paper
print(f"Nu {check['measured']:.4g} +- {check['stderr']:.2g} (standard error), de Vahl Davis "
      f"{check['reference']:g}: {check['difference_pct']:+.1f} %")
# The standard error is the averaging uncertainty only. The grid error needs three grids:
# project.grid_study(case, quantity="nusselt") gives the observed order and the GCI.
try:
    al.show(result.plot("field", field="temperature"))
except al.AerolumeError as exc:
    print(exc)
'''),
    Snippet("heated_cylinder", "Heated cylinder and a grid study",
            "Forced convection from a cylinder 10 K above the stream at Re 40 (2D CFD, Draft), against the "
            "Lange et al. (1998) correlation; optionally on three grids with the GCI.",
            '"""Forced convection from a heated cylinder at Re 40, against the Lange et al. (1998) correlation."""\n'
            + _PREAMBLE + '''
case = al.heated_cylinder_case(re=40)          # a 10 mm cylinder held 10 K above the stream, buoyancy off
print(case)
result = project.run_case(case, quality="draft")
print(result)

check = case.check(result)
print(f"Nu {check['measured']:.4g} +- {check['stderr']:.2g}, Lange et al. {check['reference']:.4g}: "
      f"{check['difference_pct']:+.1f} %")
try:
    al.show(result.plot("field", field="temperature"))
except al.AerolumeError as exc:
    print(exc)

GRID_STUDY = False                             # True: three grids 1.5x apart, Richardson extrapolation, GCI
if GRID_STUDY:
    study = project.grid_study(case, quality="draft", quantity="nusselt")
    print(study)                               # observed order, extrapolated Nu and the GCI band
    al.show(study.plot("nusselt"))
'''),
    Snippet("fan_room", "A fan in a closed room",
            "A 0.5 m fan blowing in a closed 4 m × 3 m room (2D CFD, Draft) with smoke: delivered thrust "
            "against the set thrust, and the circulation it drives.",
            '"""A fan blowing in a closed 4 m x 3 m room (2D): its delivered thrust, and the circulation it drives."""\n'
            + _PREAMBLE + '''
case = al.fan_room_case(thrust=1.0)            # one 0.5 m fan a third of the way along, 1 N per metre of span
print(case)
smoke = [{"kind": "rake", "position": (0.4, 0.4), "end": (0.4, 2.6), "count": 8}]    # to see the circulation
result = project.run_case(case, quality="draft", emitters=smoke)
print(result)
# In 2D a closed room's speeds are qualitative and grid-dependent (no vortex stretching; theory_ns2d.md
# section 18.4): the thrust is exact and the flow pattern meaningful, but use dimension=3 for numbers.
print("Note:", result.warnings[0] if result.warnings else case.notes[0])

for fan in result.fans:                        # thrust, volume flow, mean induced velocity, ideal power
    print(", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in fan.items()))
# A consistency check, not a validation: thrust mode imposes the thrust, so this shows the fan delivers what
# was set. Momentum theory does not apply in a closed room (the fan recirculates its own wake).
check = case.check(result)
print(f"Delivered {check['measured']:.4g} N/m against {check['reference']:g} N/m set: "
      f"{check['difference_pct']:+.1f} % ({check['kind']})")
try:
    al.show(result.plot("field", field="speed"))    # with streamlines: the recirculation in the room
except al.AerolumeError as exc:
    print(exc)
'''),
    Snippet("smoke_3d", "3D smoke around a sphere",
            "Two coloured smoke rakes upstream of a sphere at Re 300, where the wake sheds (3D CFD, Draft): smoke, "
            "streamlines and vortex cores.",
            '"""3D smoke around a sphere at Re 300, where the wake sheds vortices: two coloured smoke rakes."""\n'
            + _PREAMBLE + '''
sphere = project.add_sphere(0.2)               # centred at the origin; the flow is along +x
# two rakes 1.5 diameters upstream: one across the height (white), one across the width (amber).
# Each colour is its own smoke channel (up to four).
rakes = [
    {"kind": "rake", "position": (-0.3, 0.0, -0.08), "end": (-0.3, 0.0, 0.08), "count": 7, "color": "#f5f5f5"},
    {"kind": "rake", "position": (-0.3, -0.08, 0.0), "end": (-0.3, 0.08, 0.0), "count": 7, "color": "#f0a040"},
]
if "emitters" not in al.supported_physics(3):
    print("This build's 3D solver has no placed emitters yet: the run uses its default smoke rake.")
    rakes = None
result = project.run_cfd(sphere, re=300, quality="draft", emitters=rakes)
# Draft is a grid for pictures: it read CD 0.75 here, against 0.656 computed by Johnson & Patel
# (1999) at Re 300; quality="standard" and a longer duration are for numbers.
print(result)

for kind in ("smoke", "streamlines", "vortices"):
    try:
        al.show(result.plot(kind))
    except al.AerolumeError as exc:
        print(f"{kind}: {exc}")
try:
    print(result.export(project.path / "exports" / "sphere_smoke.vti", what="smoke"))   # for ParaView
except al.AerolumeError as exc:
    print(exc)
'''),
    Snippet("report", "Report of the latest results",
            "A PDF report of the three newest results, written to the project's reports folder.",
            '"""A PDF report of the three newest results in this project."""\n'
            + _PREAMBLE + '''
results = project.results()[-3:]
if not results:
    print("No results yet: run a simulation first.")
else:
    path = project.report(results, title="Script report")
    print(f"Report written to {path}")
'''),
)


def snippet(key: str) -> Snippet:
    """The snippet with this key; KeyError lists the keys."""
    for s in SNIPPETS:
        if s.key == key:
            return s
    raise KeyError(f"no snippet {key!r}; available: {', '.join(s.key for s in SNIPPETS)}")
