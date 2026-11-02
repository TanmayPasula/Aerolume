"""Crash-safe file writing and clear errors for the project store and exporters.

Every file Aerolume writes goes through :func:`atomic_writer`: the content is
written to a temporary file in the same folder, flushed to disk, and then moved
over the target with ``os.replace``, which is atomic on NTFS and POSIX file
systems. A crash, a full disk or a killed process therefore leaves either the old
file or the new one, never half of each — which matters because a CFD result can
take an hour to compute.

On Windows ``os.replace`` fails with ``PermissionError`` while another process
(OneDrive sync, an antivirus scanner, a viewer) holds the target open for a
moment, so the move is retried for about a second before giving up.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any


class ProjectError(Exception):
    """Base class of every error the project store raises on purpose."""


class ProjectNotFoundError(ProjectError, FileNotFoundError):
    """The project folder or one of its required files does not exist."""


class ProjectCorruptError(ProjectError, ValueError):
    """A project file exists but cannot be read (bad JSON, wrong format, truncated npz)."""


class ProjectExistsError(ProjectError, FileExistsError):
    """Creating a project where a non-empty folder or a file already exists."""


_REPLACE_ATTEMPTS = 20
_REPLACE_DELAY_S = 0.05


def _replace(src: str, dst: str) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY_S)


@contextlib.contextmanager
def atomic_writer(path: str | os.PathLike, mode: str = "wb", encoding: str | None = None) -> Iterator[IO[Any]]:
    """Context manager yielding a file handle whose content replaces ``path`` on success.

    The parent folder is created if needed. If the ``with`` body raises, the target is
    untouched and the temporary file is removed. Text mode writes ``\\n`` line endings
    on every platform (``newline=""``), so CSV and JSON files are byte-identical
    across operating systems.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=f".{threading.get_ident()}.tmp",
                               dir=path.parent)
    try:
        if "b" in mode:
            fh = os.fdopen(fd, mode)
        else:
            fh = os.fdopen(fd, mode, encoding=encoding or "utf-8", newline="")
        with fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_text(path: str | os.PathLike, text: str) -> Path:
    """Atomically write UTF-8 text."""
    with atomic_writer(path, "w") as fh:
        fh.write(text)
    return Path(path)


def write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    """Atomically write bytes."""
    with atomic_writer(path, "wb") as fh:
        fh.write(data)
    return Path(path)


def dumps_json(obj: Any) -> str:
    """The project's JSON style: indented, UTF-8, key order kept.

    Non-finite floats are written as ``NaN`` / ``Infinity`` (Python's JSON dialect);
    result statistics are legitimately ``nan`` when they could not be estimated, and
    turning them into ``null`` would lose the float type on reload.
    """
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def write_json(path: str | os.PathLike, obj: Any) -> Path:
    """Atomically write ``obj`` as JSON."""
    return write_text(path, dumps_json(obj))


def read_json(path: str | os.PathLike, what: str = "file") -> Any:
    """Read JSON with errors that say which project file is broken and how."""
    path = Path(path)
    try:
        # utf-8-sig: a file saved by an editor that writes a byte-order mark (Notepad)
        # is still the same JSON, not a corrupt project
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ProjectNotFoundError(f"{what} is missing: {path}") from None
    except OSError as exc:
        raise ProjectError(f"cannot read {what} {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProjectCorruptError(f"{what} is not valid JSON ({exc.msg} at line {exc.lineno}): {path}") from None


def npz_bytes(arrays: dict[str, Any], *, level: int | None = 1) -> bytes:
    """The bytes of an ``.npz`` archive of ``arrays``, deflated at zlib ``level``.

    ``np.savez_compressed`` always deflates at zlib's default level 6, which on a
    3D result costs about three times the time of level 1 for 2-5 % less size
    (measured on a 2 M-cell float16 volume: 0.93 s against 0.24 s). Large 3D
    payloads are written with this instead; ``np.load`` reads both identically.
    ``level=None`` stores the members uncompressed. Built in memory so the caller
    can write it atomically (``atomic_writer``) without holding a lock while
    compressing.
    """
    import io
    import zipfile

    import numpy as np

    buf = io.BytesIO()
    comp = zipfile.ZIP_STORED if level is None else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buf, "w", compression=comp, compresslevel=level, allowZip64=True) as zf:
        for name, a in arrays.items():
            with zf.open(f"{name}.npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(a), allow_pickle=False)
    return buf.getvalue()


def remove_tree(path: str | os.PathLike) -> None:
    """Delete a folder robustly: rename it aside first, then remove it.

    The rename makes the folder disappear from the project at once even if a file in
    it is briefly locked (Windows); a leftover ``.trash-*`` folder is removed by
    :func:`sweep_trash` the next time the project opens.
    """
    path = Path(path)
    if not path.exists():
        return
    trash = path.with_name(f".trash-{path.name}-{os.getpid()}-{time.monotonic_ns()}")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            path.rename(trash)
            break
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY_S)
    shutil.rmtree(trash, ignore_errors=True)


def sweep_trash(folder: str | os.PathLike) -> None:
    """Remove ``.trash-*`` leftovers and stale ``*.tmp`` files from crashed writes."""
    folder = Path(folder)
    if not folder.is_dir():
        return
    for p in folder.iterdir():
        if p.name.startswith(".trash-") and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.name.startswith(".") and p.name.endswith(".tmp") and p.is_file():
            # only files old enough that no live writer can own them
            with contextlib.suppress(OSError):
                if time.time() - p.stat().st_mtime > 3600:
                    p.unlink()
