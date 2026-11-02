"""The signed-in session and where it is kept between launches.

A :class:`Session` is the access token (a short-lived JWT, normally one hour),
the refresh token that renews it, and the user record the auth server returned.

Storage (:class:`SessionStore`): one file, ``account_session`` in the user config
folder (``AEROLUME_CONFIG_DIR`` aware), bound to the project URL it came from so
pointing the app at another project never sends one project's token to another.

* **Windows**: the JSON is encrypted with DPAPI (``CryptProtectData``, user
  scope) through :mod:`ctypes`, so only the same Windows account on the same
  machine can read it - a copied file or a backup is useless elsewhere. The
  profile folder's ACL already keeps other users out; DPAPI also protects
  against the file leaking (sync clients, support bundles).
* **POSIX**: created with mode 0600 through ``os.open`` (no window where the
  file is world-readable), as ``ssh`` and ``gh`` do.

If DPAPI is unavailable the file falls back to the POSIX scheme (plain JSON in
a user-only folder); the header byte records which, so a later read never
guesses. Tokens never appear in ``repr()``, log lines or exception messages.
"""
from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..project.recent import config_dir

SESSION_FILE = "account_session"
_MAGIC_DPAPI = b"AL1D"     # DPAPI-protected JSON follows
_MAGIC_PLAIN = b"AL1P"     # plain JSON follows (POSIX, 0600)
_ENTROPY = b"aerolume-account-session-v1"


def jwt_claims(token: str) -> dict:
    """The (unverified) payload of a JWT, or {} - used only to read ``sub`` / ``exp``.

    The client never trusts these claims for security decisions; the server
    verifies the signature on every request. Reading them locally just avoids a
    round trip to learn the user id.
    """
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        return data if isinstance(data, dict) else {}
    except (IndexError, ValueError, UnicodeDecodeError):
        return {}


@dataclass
class Session:
    access_token: str
    refresh_token: str
    expires_at: float               # local clock, seconds since the epoch
    user: dict[str, Any] = field(default_factory=dict)
    token_type: str = "bearer"

    # -- construction -------------------------------------------------------
    @classmethod
    def from_auth_response(cls, payload: dict, now: float | None = None) -> Session:
        """Build from a GoTrue token response.

        ``expires_at`` is computed from ``expires_in`` against the *local* clock
        at receipt, not taken from the server's absolute ``expires_at``: a PC
        whose clock is a few minutes off would otherwise refresh too late (401s)
        or in a loop.
        """
        now = time.time() if now is None else now
        access = str(payload.get("access_token") or "")
        refresh = str(payload.get("refresh_token") or "")
        if "expires_in" in payload:
            expires = now + float(payload["expires_in"])
        else:
            exp = jwt_claims(access).get("exp")
            expires = float(exp) if exp else now + 3600.0
        user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
        return cls(access, refresh, expires, dict(user), str(payload.get("token_type") or "bearer"))

    # -- facts --------------------------------------------------------------
    @property
    def user_id(self) -> str:
        return str(self.user.get("id") or jwt_claims(self.access_token).get("sub") or "")

    @property
    def email(self) -> str:
        return str(self.user.get("email") or jwt_claims(self.access_token).get("email") or "")

    @property
    def provider(self) -> str:
        meta = self.user.get("app_metadata") or {}
        return str(meta.get("provider") or "email")

    def expires_within(self, seconds: float, now: float | None = None) -> bool:
        return (time.time() if now is None else now) >= self.expires_at - seconds

    def to_dict(self) -> dict:
        return {"access_token": self.access_token, "refresh_token": self.refresh_token,
                "expires_at": self.expires_at, "user": self.user, "token_type": self.token_type}

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        return cls(str(d["access_token"]), str(d["refresh_token"]), float(d["expires_at"]),
                   dict(d.get("user") or {}), str(d.get("token_type") or "bearer"))

    def __repr__(self) -> str:
        return f"Session(user_id={self.user_id!r}, email={self.email!r}, expires_at={self.expires_at:.0f})"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# DPAPI (Windows) through ctypes
# ---------------------------------------------------------------------------

def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def blob(b: bytes) -> Blob:
        buf = ctypes.create_string_buffer(b, len(b))
        return Blob(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    src, ent, out = blob(data), blob(_ENTROPY), Blob()
    flags = 0x1   # CRYPTPROTECT_UI_FORBIDDEN: never pop a dialog
    # Same argument list for both calls: (in, description, entropy, reserved, prompt, flags, out).
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not fn(ctypes.byref(src), None, ctypes.byref(ent), None, None, flags, ctypes.byref(out)):
        raise OSError(f"DPAPI call failed (error {ctypes.get_last_error()})")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def protect(data: bytes) -> bytes:
    """Header + payload: DPAPI-encrypted on Windows when possible, else plain."""
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            return _MAGIC_DPAPI + _dpapi(data, True)
    return _MAGIC_PLAIN + data


def unprotect(blob: bytes) -> bytes:
    if blob.startswith(_MAGIC_DPAPI):
        if sys.platform != "win32":
            raise OSError("session file was encrypted on Windows")
        return _dpapi(blob[len(_MAGIC_DPAPI):], False)
    if blob.startswith(_MAGIC_PLAIN):
        return blob[len(_MAGIC_PLAIN):]
    raise OSError("unknown session file format")


def write_private(path: Path, data: bytes) -> None:
    """Atomically write ``data`` readable by the owner only (0600 on POSIX).

    The temporary file is created with ``mkstemp`` (0600 on POSIX), in the same
    folder so the final ``os.replace`` is atomic and the file is never visible
    half-written or with wider permissions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class SessionStore:
    """Load / save / clear the one stored session (bound to a project URL)."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self._path = Path(path) if path is not None else None

    @property
    def path(self) -> Path:
        # Resolved on every use, so a test (or the user) changing AEROLUME_CONFIG_DIR
        # after import is honoured, as the recent list and assistant config do.
        return self._path if self._path is not None else config_dir() / SESSION_FILE

    def load(self, url: str) -> Session | None:
        """The stored session for ``url``, or None (missing, unreadable, other project)."""
        try:
            doc = json.loads(unprotect(self.path.read_bytes()).decode("utf-8"))
            if doc.get("url") != url:
                return None
            return Session.from_dict(doc["session"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, url: str, session: Session) -> None:
        payload = json.dumps({"url": url, "session": session.to_dict(), "saved": time.time()})
        write_private(self.path, protect(payload.encode("utf-8")))

    def clear(self) -> None:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            self.path.unlink()
