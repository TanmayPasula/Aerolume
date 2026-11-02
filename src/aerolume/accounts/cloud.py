"""Project folders as cloud objects: packing, content hashes, sync state, conflicts.

A project folder (``Name.aerolume/``) is uploaded as one zip to the Storage
bucket ``projects`` at ``<user_id>/<project_id>/<content_hash>.zip`` with a
metadata row in the ``projects`` table; a zip larger than the server's
per-file limit is stored as numbered parts (see :func:`part_paths`), so project
size is bounded by the storage quota, not by the 50 MB file limit of
Supabase's free plan. Three ideas keep this safe:

**Content hash, not timestamps.** :func:`content_hash` is SHA-256 over the
sorted relative paths and the SHA-256 of each file's bytes, so it is the same
on every machine for the same content regardless of zip timestamps, file
modification times or extraction order. The one exception is ``project.json``,
hashed without its ``modified`` / ``app_version`` keys: they record when and by
which version it was last written, and identical edits on two computers must
not look like a conflict. Comparing hashes is how sync status is decided;
clocks are never trusted.

**Three-way comparison.** For a linked folder we remember the hash it had at
the last successful upload or download (the *base*). With the current local
hash and the hash in the cloud row, :func:`classify` gives:

====================  =================  =======================================
local vs base         remote vs base     status
====================  =================  =======================================
same                  same               ``synced``
changed               same               ``local_changes``  (upload)
same                  changed            ``remote_changes`` (download)
changed               changed            ``conflict`` (unless local == remote)
====================  =================  =======================================

**Content-addressed objects + optimistic locking.** A new version is uploaded
under its own hash first, then the row is updated *only if its hash is still
the one we compared against* (``PATCH ... ?content_hash=eq.<seen>``). If another
computer uploaded in between, zero rows change, our object is removed and the
caller gets a :class:`~aerolume.accounts.errors.SyncConflictError`; the other
computer's version is never overwritten. The previous object is deleted only
after the row points at the new one, so a crash mid-upload never leaves a row
pointing at a missing file.

Sync links (folder -> cloud project id + base hash) live in
``cloud_sync.json`` in the user config folder, per Supabase URL, not inside the
project: writing into the folder would change its own hash.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..project.recent import config_dir
from .errors import AccountError

SYNC_FILE = "cloud_sync.json"
HASH_PREFIX = b"aerolume-project-v1\n"

# status keys (stable strings: the GUI maps them to labels, tests compare them)
LOCAL_ONLY = "local_only"          # folder never uploaded (or link forgotten)
SYNCED = "synced"
LOCAL_CHANGES = "local_changes"
REMOTE_CHANGES = "remote_changes"
CONFLICT = "conflict"
REMOTE_ONLY = "remote_only"        # in the cloud, not on this computer
REMOTE_MISSING = "remote_missing"  # linked, but the cloud copy was deleted

STATUS_LABELS = {
    LOCAL_ONLY: "Not uploaded",
    SYNCED: "Up to date",
    LOCAL_CHANGES: "Local changes",
    REMOTE_CHANGES: "Newer in cloud",
    CONFLICT: "Conflict",
    REMOTE_ONLY: "Cloud only",
    REMOTE_MISSING: "Deleted from cloud",
}

# Files that are never part of a project's content: crash leftovers of the
# project store's atomic writes and trash renames (project/fileio.py), Python
# caches, and the litter operating systems and sync clients drop into folders.
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
_SKIP_FILES = {"Thumbs.db", "desktop.ini", ".DS_Store"}


def _skipped(rel: PurePosixPath) -> bool:
    for part in rel.parts[:-1]:
        if part in _SKIP_DIRS or part.startswith(".trash-"):
            return True
    name = rel.name
    return (name in _SKIP_FILES or name.endswith(".pyc") or name.startswith("~$")
            or (name.startswith(".") and name.endswith(".tmp")))


def project_files(root: str | os.PathLike) -> list[tuple[str, Path]]:
    """``(relative posix path, absolute path)`` of every content file, sorted by path."""
    root = Path(root)
    if not (root / "project.json").is_file():
        raise AccountError(f"{root} is not an Aerolume project folder (no project.json).", code="not_project")
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".trash-"))
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue        # never follow links out of the project
            rel = PurePosixPath(p.relative_to(root).as_posix())
            if not _skipped(rel):
                out.append((str(rel), p))
    out.sort(key=lambda t: t[0])
    return out


#: Keys of ``project.json`` that record *when* or *with what* it was saved, not what the
#: project contains. Two computers that make the same edit write different timestamps;
#: hashing them would turn identical work into a conflict.
_INDEX_VOLATILE = ("modified", "app_version")


def _file_digest(rel: str, data: bytes | None, path: Path | None = None) -> str:
    """SHA-256 of one file's content; ``project.json`` without its volatile keys."""
    if rel == "project.json":
        raw = data if data is not None else path.read_bytes()  # type: ignore[union-attr]
        try:
            doc = json.loads(raw.decode("utf-8-sig"))
            if isinstance(doc, dict):
                for k in _INDEX_VOLATILE:
                    doc.pop(k, None)
                raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except (UnicodeDecodeError, ValueError):
            pass
        return hashlib.sha256(raw).hexdigest()
    if data is not None:
        return hashlib.sha256(data).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as fh:  # type: ignore[arg-type]
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(root: str | os.PathLike) -> str:
    """SHA-256 identifying a project folder's content (see the module docstring)."""
    h = hashlib.sha256(HASH_PREFIX)
    for rel, p in project_files(root):
        h.update(rel.encode("utf-8") + b"\0" + _file_digest(rel, None, p).encode("ascii") + b"\n")
    return h.hexdigest()


_STORED_EXT = (".npz", ".png", ".jpg", ".pdf", ".zip")
_ZIP_ENTRY_OVERHEAD = 30 + 46      # local file header + central directory record, without the name
_ZIP_END_RECORD = 22
_size_cache: dict[tuple[str, int, int], int] = {}
_size_lock = threading.Lock()


@dataclass(frozen=True)
class Packed:
    data: bytes
    content_hash: str
    file_count: int
    size_bytes: int         # uncompressed bytes of the content


@dataclass(frozen=True)
class PackInfo:
    """What :func:`pack_project_to` wrote: the hash, counts and each entry's packed size."""

    content_hash: str
    file_count: int
    size_bytes: int                     # uncompressed bytes of the content
    packed_bytes: int                   # bytes of the zip itself
    entry_bytes: dict[str, int]         # relative path -> bytes it takes in the zip (compressed)


def pack_project_to(root: str | os.PathLike, fh) -> PackInfo:
    """Zip a project folder into the binary file object ``fh`` and hash its content.

    Writing to a file (the upload uses a temporary file) rather than memory keeps
    a large project - hundreds of MB of 3D fields - from needing that much RAM
    on top of the parts being sent. Entries get a fixed 1980-01-01 timestamp so
    packing the same content twice gives the same bytes; the hash is computed
    from the files, not the zip.
    """
    files = project_files(root)
    h = hashlib.sha256(HASH_PREFIX)
    total = 0
    start = fh.tell()
    with zipfile.ZipFile(fh, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, p in files:
            data = p.read_bytes()
            total += len(data)
            h.update(rel.encode("utf-8") + b"\0" + _file_digest(rel, data).encode("ascii") + b"\n")
            info = zipfile.ZipInfo(rel, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            # .npz / images are already compressed; storing them saves CPU for nothing lost
            ctype = zipfile.ZIP_STORED if rel.endswith(_STORED_EXT) else zipfile.ZIP_DEFLATED
            zf.writestr(info, data, compress_type=ctype)
        entries = {i.filename: i.compress_size for i in zf.infolist()}
    fh.flush()
    return PackInfo(h.hexdigest(), len(files), total, fh.tell() - start, entries)


def _deflated_size(p: Path) -> int:
    """Bytes ``p`` takes deflated exactly as :func:`pack_project_to` does it (zlib level 6, raw).

    Cached by (path, size, mtime): a 50 MB VTK export costs about a second to
    deflate, and the Cloud projects list asks again on every refresh.
    """
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    with _size_lock:
        hit = _size_cache.get(key)
    if hit is not None:
        return hit
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    n = 0
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += len(co.compress(chunk))
    n += len(co.flush())
    with _size_lock:
        if len(_size_cache) > 20_000:
            _size_cache.clear()
        _size_cache[key] = n
    return n


def packed_sizes(root: str | os.PathLike) -> tuple[dict[str, int], int]:
    """Per-file packed sizes and the archive's total size, computed without writing an archive.

    The archive layout is fixed (no data descriptors, no extra fields, stored or
    raw-deflated level 6), so the total is exact: ``sum(entries) + 76 + 2 * len(name)
    per entry + 22``. The tests check it against :func:`pack_project_to`. This is
    what the Cloud projects dialog shows *before* an upload.
    """
    entries: dict[str, int] = {}
    total = _ZIP_END_RECORD
    for rel, p in project_files(root):
        size = p.stat().st_size
        n = size if rel.endswith(_STORED_EXT) else _deflated_size(p)
        entries[rel] = n
        total += n + _ZIP_ENTRY_OVERHEAD + 2 * len(rel.encode("utf-8"))
    return entries, total


def pack_project(root: str | os.PathLike) -> Packed:
    """Zip a project folder in memory (see :func:`pack_project_to`)."""
    buf = io.BytesIO()
    info = pack_project_to(root, buf)
    return Packed(buf.getvalue(), info.content_hash, info.file_count, info.size_bytes)


# ---------------------------------------------------------------------------
# parts: archives larger than the server's per-file limit
# ---------------------------------------------------------------------------
#
# Supabase's free plan refuses any single file over 50 MB, and one 3D run can
# take 15 MB, so a normal project with a handful of 3D results passes that.
# Rather than asking the user to throw results away, an archive larger than one
# part is stored as several objects, ``<hash>.zip.part001`` ... ``.partNNN``,
# each under the limit; the row's ``part_count`` says how many there are. The
# parts are plain byte ranges of the zip (concatenated, they are the zip), so the
# content hash still covers the whole project and download checks it as before.
# One-part archives keep the old single ``<hash>.zip`` name.

def part_paths(object_path: str, part_count: int) -> list[str]:
    """Storage object names of an archive stored in ``part_count`` parts."""
    n = max(1, int(part_count or 1))
    if n == 1:
        return [object_path]
    return [f"{object_path}.part{i:03d}" for i in range(1, n + 1)]


def part_count_for(packed_bytes: int, part_bytes: int) -> int:
    """How many parts of at most ``part_bytes`` an archive of ``packed_bytes`` needs."""
    return max(1, -(-int(packed_bytes) // max(1, int(part_bytes))))


# ---------------------------------------------------------------------------
# what takes the space: per-run size breakdown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SizeGroup:
    """One line of "what will be uploaded": a run, or a folder of the project."""

    label: str              # "Car smoke (3D run)", "Exports", ...
    kind: str               # "run" | "designs" | "exports" | "reports" | "experiments" | "other"
    prefix: str             # relative folder, e.g. "sims/d2e9b26ba784" ("" for loose files)
    size_bytes: int         # bytes on disk, or packed bytes when entry sizes were given
    file_count: int


_FOLDER_LABELS = {"designs": "Designs", "exports": "Exported files", "reports": "Reports",
                  "experiments": "Assistant experiments"}


def _run_label(folder: Path) -> str:
    """"<name> (2D run)" / "(3D run)" from a run folder's own files; the folder name if unreadable."""
    name, dim = folder.name, ""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        doc = json.loads((folder / "simulation.json").read_text(encoding="utf-8-sig"))
        name = str(doc.get("name") or name)
    with contextlib.suppress(OSError, ValueError, AttributeError):
        kind = str(json.loads((folder / "result.json").read_text(encoding="utf-8-sig")).get("type") or "")
        dim = "3D" if kind.endswith("3d") else "2D" if kind else ""
    return f"{name} ({dim} run)" if dim else f"{name} (run)"


def size_breakdown(root: str | os.PathLike, entry_bytes: dict[str, int] | None = None) -> list[SizeGroup]:
    """Where a project's bytes are, largest first: one line per run, one per other folder.

    With ``entry_bytes`` (from :class:`PackInfo`) the sizes are what each file
    takes in the archive; without, they are sizes on disk (exact, and close to
    packed: results are stored uncompressed in the zip because ``.npz`` is
    already compressed).
    """
    root = Path(root)
    acc: dict[tuple[str, str], list[int]] = {}
    for rel, p in project_files(root):
        parts = rel.split("/")
        if parts[0] == "sims" and len(parts) >= 3:
            key = ("run", f"sims/{parts[1]}")
        elif len(parts) >= 2:
            key = (parts[0] if parts[0] in _FOLDER_LABELS else "other", parts[0])
        else:
            key = ("other", "")
        if entry_bytes is not None and rel in entry_bytes:
            size = int(entry_bytes[rel])
        else:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
        slot = acc.setdefault(key, [0, 0])
        slot[0] += size
        slot[1] += 1
    # Loose files and unknown folders are one "Project settings and other files" line.
    merged: dict[tuple[str, str], list[int]] = {}
    for (kind, prefix), (size, count) in acc.items():
        k = ("other", "") if kind == "other" else (kind, prefix)
        slot = merged.setdefault(k, [0, 0])
        slot[0] += size
        slot[1] += count
    out = []
    for (kind, prefix), (size, count) in merged.items():
        if kind == "run":
            lab = _run_label(root / prefix)
        elif kind == "other":
            lab = "Project settings and other files"
        else:
            lab = _FOLDER_LABELS[kind]
        out.append(SizeGroup(lab, kind, prefix, size, count))
    out.sort(key=lambda g: (-g.size_bytes, g.label))
    return out


_WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$", re.IGNORECASE)


def _safe_member(name: str) -> PurePosixPath:
    """Reject zip-slip and Windows-hostile member names before anything is written."""
    p = PurePosixPath(name.replace("\\", "/"))
    if (p.is_absolute() or not p.parts or any(part in ("..", "") for part in p.parts)
            or ":" in name or any(_WIN_RESERVED.match(part) for part in p.parts)):
        raise AccountError(f"The cloud copy contains an unsafe path ({name!r}); nothing was written.",
                           code="unsafe_archive")
    return p


def unpack_project(data, dest: str | os.PathLike, *, expected_hash: str = "",
                   max_bytes: int = 20 * 1024**3) -> Path:
    """Extract a project zip into the new folder ``dest`` (which must not exist).

    ``data`` is the archive as bytes or as a seekable binary file (a multi-part
    download is reassembled into a temporary file rather than memory).
    Extraction goes to a temporary sibling first and is renamed into place only
    after every member passed the safety checks and the content hash matches
    ``expected_hash`` (when given), so a damaged or tampered download never
    leaves a half-written project behind.
    """
    dest = Path(dest)
    if dest.exists():
        raise AccountError(f"{dest} already exists.", code="exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".trash-download-", dir=str(dest.parent)))
    src = io.BytesIO(data) if isinstance(data, bytes | bytearray | memoryview) else data
    try:
        try:
            zf_ctx = zipfile.ZipFile(src)
        except zipfile.BadZipFile as exc:
            raise AccountError("The cloud copy is damaged (not a readable archive); nothing was written. "
                               "Try again.", code="hash_mismatch") from exc
        with zf_ctx as zf:
            infos = zf.infolist()
            if sum(i.file_size for i in infos) > max_bytes:
                raise AccountError("The cloud copy is larger than expected; nothing was written.",
                                   code="unsafe_archive")
            for info in infos:
                if info.is_dir():
                    continue
                rel = _safe_member(info.filename)
                target = tmp.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
        if not (tmp / "project.json").is_file():
            raise AccountError("The cloud copy is not an Aerolume project (no project.json).", code="not_project")
        if expected_hash:
            got = content_hash(tmp)
            if got != expected_hash:
                raise AccountError("The downloaded project does not match its recorded checksum; "
                                   "nothing was written. Try again.", code="hash_mismatch")
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def classify(local_hash: str | None, base_hash: str | None, remote_hash: str | None) -> str:
    """Sync status from the three hashes (None = that side does not exist / is unknown)."""
    if local_hash is None:
        return REMOTE_ONLY if remote_hash else LOCAL_ONLY
    if not base_hash:
        if remote_hash is None:
            return LOCAL_ONLY
        return SYNCED if remote_hash == local_hash else CONFLICT
    if remote_hash is None:
        return REMOTE_MISSING
    if local_hash == remote_hash:
        return SYNCED
    local_changed, remote_changed = local_hash != base_hash, remote_hash != base_hash
    if local_changed and remote_changed:
        return CONFLICT
    if local_changed:
        return LOCAL_CHANGES
    return REMOTE_CHANGES


def safe_folder_name(name: str) -> str:
    """A project name made safe as a Windows / POSIX folder name (without the suffix)."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(".")
    if not s or _WIN_RESERVED.match(s):
        s = f"Project {s}".strip()
    return s[:120]


# ---------------------------------------------------------------------------
# sync links
# ---------------------------------------------------------------------------

def _norm(path: str | os.PathLike) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


@dataclass
class SyncLink:
    project_id: str
    base_hash: str
    path: str
    synced_at: float


class SyncStore:
    """``cloud_sync.json``: {supabase url: {normalised folder path: link}}."""

    _lock = threading.Lock()

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self._path = Path(path) if path is not None else None

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else config_dir() / SYNC_FILE

    def _read(self) -> dict:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, doc: dict) -> None:
        from ..project.fileio import write_text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_text(self.path, json.dumps(doc, indent=1, sort_keys=True))

    def get(self, url: str, folder: str | os.PathLike) -> SyncLink | None:
        d = self._read().get(url, {}).get(_norm(folder))
        if not isinstance(d, dict):
            return None
        try:
            return SyncLink(str(d["project_id"]), str(d["base_hash"]), str(d.get("path") or folder),
                            float(d.get("synced_at") or 0))
        except (KeyError, TypeError, ValueError):
            return None

    def links(self, url: str) -> list[SyncLink]:
        out = []
        for key, d in self._read().get(url, {}).items():
            with contextlib.suppress(KeyError, TypeError, ValueError):
                out.append(SyncLink(str(d["project_id"]), str(d["base_hash"]), str(d.get("path") or key),
                                    float(d.get("synced_at") or 0)))
        return out

    def set(self, url: str, folder: str | os.PathLike, project_id: str, base_hash: str) -> None:
        with self._lock:
            doc = self._read()
            doc.setdefault(url, {})[_norm(folder)] = {
                "project_id": project_id, "base_hash": base_hash, "path": str(Path(folder)),
                "synced_at": time.time()}
            self._write(doc)

    def forget(self, url: str, *, folder: str | os.PathLike | None = None, project_id: str = "") -> None:
        with self._lock:
            doc = self._read()
            links = doc.get(url, {})
            for key in list(links):
                if (folder is not None and key == _norm(folder)) or \
                        (project_id and links[key].get("project_id") == project_id):
                    del links[key]
            self._write(doc)
