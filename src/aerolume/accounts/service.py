"""``AccountService``: the one object the app, the CLI and scripts use for accounts.

It owns the connection settings, the stored session (and its refresh), and the
higher-level operations - profile, cloud projects, share links - on top of the
endpoint wrappers in :mod:`aerolume.accounts.supabase`. Every method is a
plain blocking call; the GUI runs them on a worker thread
(:mod:`aerolume.app.widgets.account_menu`).

Offline by default: with nothing configured, :attr:`configured` is False,
:attr:`signed_in` is False and every network method raises
:class:`~aerolume.accounts.errors.NotConfiguredError` before touching a socket.

Session refresh is serialised by a lock. Supabase rotates refresh tokens on
every use; two threads refreshing with the same token at once would make the
second one fail and (worse) look like a revoked session. Inside the lock the
expiry is re-checked, so the second caller just picks up the new token.

A request that comes back 401 "JWT expired" despite a fresh-looking token (the
server revoked it, or the PC clock is far off) is retried exactly once after a
forced refresh.

Listeners (:meth:`subscribe`) receive one of ``"signed_in"``, ``"signed_out"``,
``"refreshed"``, ``"profile"``, ``"config"``; they are called on the thread that
caused the change, so GUI code must hop to its own thread (the account menu's
controller does).
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import tempfile
import threading
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import cloud
from .config import AccountConfig, load_config
from .errors import (
    AccountError,
    AuthError,
    NotConfiguredError,
    NotSignedInError,
    RemoteError,
    SyncConflictError,
    TooLargeError,
)
from .http import HttpClient
from .loopback import LoopbackReceiver, challenge_for, new_verifier
from .session import Session, SessionStore
from .supabase import OAUTH_PROVIDERS, SupabaseBackend

log = logging.getLogger(__name__)

REFRESH_LEEWAY_S = 60.0
PROJECTS_BUCKET = "projects"
AVATARS_BUCKET = "avatars"
SHARES_BUCKET = "shares"
MAX_AVATAR_BYTES = 1024 * 1024
MAX_SHARE_DAYS = 365
# Bare MIME types: Storage matches uploads against the bucket's allowed_mime_types, and a
# "; charset=" parameter is not part of that list (the report HTML declares UTF-8 itself).
SHARE_TYPES = {".html": "text/html", ".htm": "text/html",
               ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MIN_PASSWORD = 8
SCHEMA_VERSION = 2          # what supabase/migrations provide (public.aerolume_schema_version())
#: The migration that added multi-part project archives (schema version 2); named in errors
#: from a database that has not applied it yet.
PARTS_MIGRATION = "20261030120000_aerolume_project_parts.sql"
ZIP_TYPE = "application/zip"
PART_TYPE = "application/octet-stream"   # one part alone is a byte range, not a zip

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


# ---------------------------------------------------------------------------
# value types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountUser:
    id: str
    email: str
    display_name: str = ""
    provider: str = "email"

    @property
    def label(self) -> str:
        """What the UI calls the user: the display name, else the email."""
        return self.display_name or self.email

    @property
    def initials(self) -> str:
        """Up to two letters for the avatar button ("Ada Lovelace" -> "AL", "ada@x" -> "A")."""
        src = self.display_name.strip() or self.email.split("@")[0]
        words = [w for w in re.split(r"[\s._\-]+", src) if w and w[0].isalnum()]
        if not words:
            return "?"
        if len(words) == 1:
            return words[0][0].upper()
        return (words[0][0] + words[-1][0]).upper()


@dataclass
class Profile:
    id: str
    display_name: str = ""
    affiliation: str = ""
    avatar_path: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def from_row(cls, row: dict) -> Profile:
        prefs = row.get("preferences")
        return cls(str(row.get("id") or ""), str(row.get("display_name") or ""),
                   str(row.get("affiliation") or ""), str(row.get("avatar_path") or ""),
                   dict(prefs) if isinstance(prefs, dict) else {}, str(row.get("updated_at") or ""))


@dataclass(frozen=True)
class CloudProject:
    id: str
    name: str
    content_hash: str
    size_bytes: int
    file_count: int
    object_path: str
    updated_at: str = ""
    created_at: str = ""
    app_version: str = ""
    part_count: int = 1

    @classmethod
    def from_row(cls, row: dict) -> CloudProject:
        return cls(str(row["id"]), str(row.get("name") or ""), str(row.get("content_hash") or ""),
                   int(row.get("size_bytes") or 0), int(row.get("file_count") or 0),
                   str(row.get("object_path") or ""), str(row.get("updated_at") or ""),
                   str(row.get("created_at") or ""), str(row.get("app_version") or ""),
                   max(1, int(row.get("part_count") or 1)))

    @property
    def object_paths(self) -> list[str]:
        """Every Storage object of this version (one, or its numbered parts)."""
        return cloud.part_paths(self.object_path, self.part_count) if self.object_path else []


@dataclass(frozen=True)
class ProjectStatus:
    """One line of the Cloud projects list: a local folder, a cloud project, or both."""

    status: str                         # cloud.SYNCED, cloud.CONFLICT, ...
    name: str
    local_path: str = ""
    cloud: CloudProject | None = None
    local_hash: str = ""
    local_bytes: int = 0                # bytes on disk of the folder's content (0: no local folder)

    @property
    def label(self) -> str:
        return cloud.STATUS_LABELS.get(self.status, self.status)


@dataclass(frozen=True)
class UploadPreview:
    """What uploading a folder would send, computed before anything is uploaded.

    The Cloud projects dialog shows it for the selected folder, so the size and
    where it comes from are known before the upload starts, not after it fails.
    ``size_bytes`` is the exact packed size (:func:`aerolume.accounts.cloud.packed_sizes`),
    which can be far below the size on disk: a VTK export deflates to a fraction.
    """

    name: str
    size_bytes: int                      # bytes of the archive that would be uploaded
    disk_bytes: int                      # bytes on disk of the files it would contain
    file_count: int
    groups: tuple[cloud.SizeGroup, ...]  # packed bytes, largest first: runs, designs, exports, ...
    part_bytes: int                      # size of one uploaded part
    max_project_bytes: int               # the configured project limit

    @property
    def parts(self) -> int:
        """Parts the upload will be split into."""
        return cloud.part_count_for(self.size_bytes, self.part_bytes)

    @property
    def runs(self) -> int:
        return sum(1 for g in self.groups if g.kind == "run")

    @property
    def over_limit(self) -> bool:
        return self.size_bytes > self.max_project_bytes


@dataclass(frozen=True)
class ShareLink:
    token: str
    title: str
    files: tuple[str, ...]
    links: dict[str, str]             # file name -> signed URL
    created_at: str = ""
    expires_at: str = ""
    project_id: str = ""

    @property
    def url(self) -> str:
        """The link to hand out: the report page."""
        return self.links.get("index.html", "")


def _share_from_row(r: dict) -> ShareLink:
    links = r.get("links") if isinstance(r.get("links"), dict) else {}
    return ShareLink(str(r["token"]), str(r.get("title") or ""), tuple(r.get("files") or ()),
                     {str(k): str(v) for k, v in links.items()}, str(r.get("created_at") or ""),
                     str(r.get("expires_at") or ""), str(r.get("project_id") or ""))


@dataclass(frozen=True)
class SignUpResult:
    user: AccountUser | None          # signed in at once (project auto-confirms emails)
    needs_confirmation: bool          # else: an email with a link + code was sent


@dataclass(frozen=True)
class ConnectionReport:
    """What "Test connection" found, one fact per field (the dialog prints them)."""

    reachable: bool
    key_accepted: bool
    schema_version: int | None
    providers: tuple[str, ...]
    signup_enabled: bool
    message: str


class PendingSignIn:
    """A browser or email sign-in waiting for its loopback callback (or a typed code).

    Created by :meth:`AccountService.begin_oauth`, :meth:`~AccountService.begin_email_link`
    or :meth:`~AccountService.begin_password_reset`. :meth:`wait` blocks until the
    browser hits the loopback URL and returns the signed-in user;
    :meth:`complete_with_code` finishes with the 6-digit code from the email
    instead; :meth:`cancel` (any thread) makes a waiting :meth:`wait` raise.
    """

    def __init__(self, service: AccountService, kind: str, receiver: LoopbackReceiver, verifier: str,
                 email: str = "", url: str = "") -> None:
        self.service, self.kind, self.email, self.url = service, kind, email, url
        self._receiver, self._verifier = receiver, verifier
        self._closed = False

    @property
    def redirect_uri(self) -> str:
        return self._receiver.redirect_uri

    def wait(self, timeout: float = 600.0) -> AccountUser:
        try:
            params = self._receiver.wait(timeout)
        finally:
            self.close()
        if "code" not in params:
            desc = params.get("error_description") or params.get("error") or "unknown error"
            raise AuthError(f"Sign-in did not complete: {desc}.", code=params.get("error") or "oauth_error")
        payload = self.service._backend().token_pkce(params["code"], self._verifier)
        return self.service._adopt(payload, recovery=self.kind == "recovery")

    def complete_with_code(self, code: str) -> AccountUser:
        """Finish with the code from the email ("email" link / "signup" / "recovery")."""
        code = re.sub(r"\s+", "", code or "")
        if not code:
            raise AuthError("Enter the code from the email.", code="missing_code")
        kind = {"email": "email", "signup": "signup", "recovery": "recovery"}.get(self.kind, "email")
        payload = self.service._backend().verify_otp(self.email, code, kind)
        self.cancel()
        return self.service._adopt(payload, recovery=self.kind == "recovery")

    def cancel(self) -> None:
        self._receiver.cancel()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._receiver.close()


# ---------------------------------------------------------------------------
# the service
# ---------------------------------------------------------------------------

class AccountService:
    """Accounts, profile, cloud projects and shares for one configured project."""

    def __init__(self, config: AccountConfig | None = None, *, http: HttpClient | None = None,
                 store: SessionStore | None = None, sync_store: cloud.SyncStore | None = None,
                 open_browser: Callable[[str], Any] | None = None) -> None:
        self._config = config if config is not None else load_config()
        self._http = http or HttpClient()
        self._store = store or SessionStore()
        self._sync = sync_store or cloud.SyncStore()
        self._open_browser = open_browser or webbrowser.open
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._session: Session | None = None
        self._loaded = False
        self._listeners: list[Callable[[str], None]] = []
        self._profile: Profile | None = None
        self.recovery_pending = False   # signed in from a reset link: ask for a new password

    # -- configuration --------------------------------------------------------
    @property
    def config(self) -> AccountConfig:
        return self._config

    @property
    def configured(self) -> bool:
        return self._config.configured

    def set_config(self, config: AccountConfig) -> None:
        """Switch projects (Account settings). The session of another URL is not reused."""
        with self._lock:
            changed = (config.url, config.anon_key) != (self._config.url, self._config.anon_key)
            self._config = config
            if changed:
                self._session, self._loaded, self._profile = None, False, None
        if changed:
            self._emit("config")

    def reload_config(self) -> None:
        self.set_config(load_config())

    def _backend(self) -> SupabaseBackend:
        if not self._config.configured:
            raise NotConfiguredError()
        return SupabaseBackend(self._config, self._http)

    # -- listeners ------------------------------------------------------------
    def subscribe(self, fn: Callable[[str], None]) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    def _emit(self, event: str) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:
                log.exception("account listener failed for %s", event)

    # -- session --------------------------------------------------------------
    @property
    def session(self) -> Session | None:
        with self._lock:
            if not self._loaded:
                self._loaded = True
                if self._config.configured:
                    self._session = self._store.load(self._config.url)
            return self._session

    @property
    def signed_in(self) -> bool:
        return self.session is not None

    @property
    def user(self) -> AccountUser | None:
        s = self.session
        if s is None:
            return None
        meta = s.user.get("user_metadata") or {}
        name = (self._profile.display_name if self._profile and self._profile.id == s.user_id else "") \
            or str(meta.get("display_name") or meta.get("full_name") or meta.get("name") or "")
        return AccountUser(s.user_id, s.email, name, s.provider)

    @property
    def profile_cached(self) -> Profile | None:
        return self._profile

    def _adopt(self, payload: dict, *, recovery: bool = False) -> AccountUser:
        """Store the session from a token response and announce the sign-in."""
        if not payload or not payload.get("access_token"):
            raise AuthError("The server did not return a session.", code="no_session")
        session = Session.from_auth_response(payload)
        with self._lock:
            self._session, self._loaded, self._profile = session, True, None
            self.recovery_pending = recovery
            self._store.save(self._config.url, session)
        log.info("signed in (provider %s)", session.provider)
        self._emit("signed_in")
        return self.user  # type: ignore[return-value]

    def _drop_session(self) -> None:
        with self._lock:
            had = self._session is not None
            self._session, self._loaded, self._profile = None, True, None
            self.recovery_pending = False
            self._store.clear()
        if had:
            self._emit("signed_out")

    def refresh(self, *, force: bool = True) -> Session:
        """Renew the access token (rotating the refresh token). Signs out if the server refuses."""
        with self._refresh_lock:
            s = self.session
            if s is None:
                raise NotSignedInError()
            if not force and not s.expires_within(REFRESH_LEEWAY_S):
                return s
            try:
                payload = self._backend().token_refresh(s.refresh_token)
            except AuthError as exc:
                if exc.code == "invalid_api_key":
                    raise
                log.info("session refresh refused (%s); signing out", exc.code)
                self._drop_session()
                raise NotSignedInError("Your session has ended. Sign in again.") from None
            new = Session.from_auth_response(payload)
            if not new.user:
                new.user = s.user
            with self._lock:
                self._session = new
                self._store.save(self._config.url, new)
        self._emit("refreshed")
        return new

    def access_token(self) -> str:
        """A valid access token, refreshed first when it expires within a minute."""
        s = self.session
        if s is None:
            raise NotSignedInError()
        if s.expires_within(REFRESH_LEEWAY_S):
            s = self.refresh(force=False)
        return s.access_token

    def _authed(self, fn: Callable[[str], Any]) -> Any:
        """Run ``fn(token)``; on a 401/403 JWT error refresh once and retry."""
        try:
            return fn(self.access_token())
        except AuthError as exc:
            if exc.code != "bad_jwt":
                raise
        return fn(self.refresh(force=True).access_token)

    def _uid(self) -> str:
        s = self.session
        if s is None or not s.user_id:
            raise NotSignedInError()
        return s.user_id

    # -- connection check ----------------------------------------------------
    def test_connection(self) -> ConnectionReport:
        """Reach the project, check the key, the auth settings and the database schema."""
        b = self._backend()
        try:
            settings = b.auth_settings()
        except AuthError as exc:
            return ConnectionReport(True, exc.code != "invalid_api_key", None, (), False, str(exc))
        external = settings.get("external") or {}
        providers = tuple(p for p in OAUTH_PROVIDERS if external.get(p))
        signup = not settings.get("disable_signup", False)
        try:
            version = b.rpc("aerolume_schema_version", None, what="Checking the database")
            version = int(version) if version is not None else None
        except (RemoteError, AuthError, TypeError, ValueError) as exc:
            msg = str(exc) if isinstance(exc, AccountError) else "Unexpected answer from the database."
            return ConnectionReport(True, True, None, providers, signup, msg)
        if version is None or version < SCHEMA_VERSION:
            msg = (f"Connected, but the database schema is version {version} and this Aerolume needs "
                   f"{SCHEMA_VERSION}. Apply the newer files in supabase/migrations.")
        else:
            msg = "Connected. Sign-in, cloud projects and share links are available."
        return ConnectionReport(True, True, version, providers, signup, msg)

    def auth_providers(self) -> tuple[str, ...]:
        """OAuth providers the project has enabled (GitHub, Google), from its public settings."""
        external = self._backend().auth_settings().get("external") or {}
        return tuple(p for p in OAUTH_PROVIDERS if external.get(p))

    # -- email + password ----------------------------------------------------
    @staticmethod
    def _check_email(email: str) -> str:
        email = (email or "").strip()
        if not valid_email(email):
            raise AuthError("Enter a valid email address.", code="invalid_email")
        return email

    def sign_in(self, email: str, password: str) -> AccountUser:
        email = self._check_email(email)
        if not password:
            raise AuthError("Enter your password.", code="missing_password")
        return self._adopt(self._backend().token_password(email, password))

    def sign_up(self, email: str, password: str, display_name: str = "") -> tuple[SignUpResult, PendingSignIn | None]:
        """Create an account. Returns the result and, when confirmation is needed, a pending
        sign-in that completes when the confirmation link reaches this computer (or with the code).
        """
        email = self._check_email(email)
        if len(password or "") < MIN_PASSWORD:
            raise AuthError(f"Use at least {MIN_PASSWORD} characters for the password.", code="weak_password")
        data = {"display_name": display_name.strip()} if display_name.strip() else {}
        verifier = new_verifier()
        receiver = LoopbackReceiver(self._config.redirect_ports)
        try:
            payload = self._backend().sign_up(email, password, data=data, redirect_to=receiver.redirect_uri,
                                              code_challenge=challenge_for(verifier))
        except BaseException:
            receiver.close()
            raise
        if payload.get("access_token"):
            receiver.close()
            return SignUpResult(self._adopt(payload), False), None
        # Supabase answers an existing, confirmed address with a fake user (no identities) so
        # the endpoint cannot be used to probe which emails have accounts; say so neutrally.
        return SignUpResult(None, True), PendingSignIn(self, "signup", receiver, verifier, email)

    def begin_email_link(self, email: str, *, create_user: bool = True) -> PendingSignIn:
        """Email a sign-in link (+ code). Returns a pending sign-in to :meth:`~PendingSignIn.wait` on."""
        email = self._check_email(email)
        verifier = new_verifier()
        receiver = LoopbackReceiver(self._config.redirect_ports)
        try:
            self._backend().send_otp(email, create_user=create_user, redirect_to=receiver.redirect_uri,
                                     code_challenge=challenge_for(verifier))
        except BaseException:
            receiver.close()
            raise
        return PendingSignIn(self, "email", receiver, verifier, email)

    def begin_password_reset(self, email: str) -> PendingSignIn:
        """Email a reset link (+ code). Completing it signs in with ``recovery_pending`` set,
        and the caller then asks for a new password (:meth:`set_password`)."""
        email = self._check_email(email)
        verifier = new_verifier()
        receiver = LoopbackReceiver(self._config.redirect_ports)
        try:
            self._backend().recover(email, redirect_to=receiver.redirect_uri,
                                    code_challenge=challenge_for(verifier))
        except BaseException:
            receiver.close()
            raise
        return PendingSignIn(self, "recovery", receiver, verifier, email)

    def set_password(self, new_password: str) -> None:
        if len(new_password or "") < MIN_PASSWORD:
            raise AuthError(f"Use at least {MIN_PASSWORD} characters for the password.", code="weak_password")
        self._authed(lambda tok: self._backend().update_user(tok, {"password": new_password}))
        self.recovery_pending = False

    # -- OAuth ------------------------------------------------------------------
    def begin_oauth(self, provider: str, *, open_browser: bool = True) -> PendingSignIn:
        """Open the system browser at the provider's sign-in page (PKCE + loopback redirect)."""
        if provider not in OAUTH_PROVIDERS:
            raise AuthError(f"Unknown sign-in provider {provider!r}.", code="provider")
        verifier = new_verifier()
        receiver = LoopbackReceiver(self._config.redirect_ports)
        url = self._backend().authorize_url(provider, redirect_to=receiver.redirect_uri,
                                            code_challenge=challenge_for(verifier))
        pending = PendingSignIn(self, "oauth", receiver, verifier, url=url)
        if open_browser:
            try:
                self._open_browser(url)
            except Exception:
                pending.close()
                raise
        return pending

    def sign_in_with_provider(self, provider: str, timeout: float = 300.0) -> AccountUser:
        return self.begin_oauth(provider).wait(timeout)

    # -- sign out ---------------------------------------------------------------
    def sign_out(self, *, everywhere: bool = False) -> None:
        """Forget the session here; revoke it on the server too when reachable.

        The local session is removed even when the server cannot be reached:
        the user asked to be signed out of this computer, and that must not
        depend on the network.
        """
        s = self.session
        if s is None:
            return
        try:
            self._backend().logout(s.access_token, "global" if everywhere else "local")
        except AccountError as exc:
            log.info("server-side sign-out skipped (%s)", exc.code)
        self._drop_session()

    # -- profile ----------------------------------------------------------------
    def get_profile(self) -> Profile:
        uid = self._uid()
        rows = self._authed(lambda tok: self._backend().select("profiles", tok, {"id": uid},
                                                               what="Reading the profile"))
        if rows:
            prof = Profile.from_row(rows[0])
        else:   # the sign-up trigger did not run (older schema): create the row now
            meta = (self.session.user.get("user_metadata") or {}) if self.session else {}
            row = {"id": uid, "display_name": str(meta.get("display_name") or meta.get("full_name") or "")}
            out = self._authed(lambda tok: self._backend().insert("profiles", tok, row, upsert=True,
                                                                  on_conflict="id", what="Creating the profile"))
            prof = Profile.from_row(out[0] if out else row)
        self._profile = prof
        self._emit("profile")
        return prof

    def update_profile(self, *, display_name: str | None = None, affiliation: str | None = None,
                       preferences: dict | None = None) -> Profile:
        uid = self._uid()
        values: dict[str, Any] = {}
        if display_name is not None:
            if len(display_name.strip()) > 80:
                raise AccountError("Use at most 80 characters for the display name.", code="too_long")
            values["display_name"] = display_name.strip()
        if affiliation is not None:
            if len(affiliation.strip()) > 120:
                raise AccountError("Use at most 120 characters for the affiliation.", code="too_long")
            values["affiliation"] = affiliation.strip()
        if preferences is not None:
            values["preferences"] = dict(preferences)
        if not values:
            return self._profile or self.get_profile()
        rows = self._authed(lambda tok: self._backend().update("profiles", tok, values, {"id": uid},
                                                               what="Saving the profile"))
        if not rows:
            rows = self._authed(lambda tok: self._backend().insert(
                "profiles", tok, {"id": uid, **values}, upsert=True, on_conflict="id", what="Saving the profile"))
        self._profile = Profile.from_row(rows[0]) if rows else Profile(uid, **values)
        self._emit("profile")
        return self._profile

    @staticmethod
    def _image_type(data: bytes) -> tuple[str, str]:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png", "png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg", "jpg"
        raise AccountError("Use a PNG or JPEG picture.", code="image_type")

    def set_avatar(self, data: bytes) -> Profile:
        uid = self._uid()
        if len(data) > MAX_AVATAR_BYTES:
            raise TooLargeError("Use a picture of at most 1 MB.", code="too_large")
        ctype, ext = self._image_type(data)
        path = f"{uid}/avatar.{ext}"
        self._authed(lambda tok: self._backend().upload(AVATARS_BUCKET, path, data, tok, content_type=ctype,
                                                        upsert=True, what="Uploading the picture"))
        old = (self._profile or self.get_profile()).avatar_path
        rows = self._authed(lambda tok: self._backend().update("profiles", tok, {"avatar_path": path},
                                                               {"id": uid}, what="Saving the profile"))
        if old and old != path:
            self._quiet_remove(AVATARS_BUCKET, [old])
        self._profile = Profile.from_row(rows[0]) if rows else self.get_profile()
        self._emit("profile")
        return self._profile

    def avatar_bytes(self) -> bytes | None:
        prof = self._profile or self.get_profile()
        if not prof.avatar_path:
            return None
        try:
            return self._authed(lambda tok: self._backend().download(AVATARS_BUCKET, prof.avatar_path, tok))
        except RemoteError as exc:
            if exc.code == "not_found":
                return None
            raise

    def remove_avatar(self) -> Profile:
        uid = self._uid()
        prof = self._profile or self.get_profile()
        if prof.avatar_path:
            self._quiet_remove(AVATARS_BUCKET, [prof.avatar_path])
        rows = self._authed(lambda tok: self._backend().update("profiles", tok, {"avatar_path": None},
                                                               {"id": uid}, what="Saving the profile"))
        self._profile = Profile.from_row(rows[0]) if rows else Profile(uid)
        self._emit("profile")
        return self._profile

    def _quiet_remove(self, bucket: str, paths: list[str]) -> None:
        """Best-effort cleanup of objects nothing references any more (never fails the operation)."""
        try:
            self._authed(lambda tok: self._backend().remove(bucket, paths, tok))
        except AccountError as exc:
            log.info("could not remove %d old object(s) from %s (%s)", len(paths), bucket, exc.code)

    # -- cloud projects -------------------------------------------------------
    def list_cloud_projects(self) -> list[CloudProject]:
        uid = self._uid()
        rows = self._authed(lambda tok: self._backend().select(
            "projects", tok, {"owner": uid}, order="updated_at.desc", what="Listing cloud projects"))
        return [CloudProject.from_row(r) for r in rows]

    def _cloud_row(self, project_id: str) -> CloudProject | None:
        rows = self._authed(lambda tok: self._backend().select("projects", tok, {"id": project_id},
                                                               what="Reading the cloud project"))
        return CloudProject.from_row(rows[0]) if rows else None

    def project_status(self, folder: str | Path, remote: list[CloudProject] | None = None) -> ProjectStatus:
        """Sync status of one local project folder."""
        folder = Path(folder)
        local = cloud.content_hash(folder)
        size = _disk_bytes(folder)
        name = _project_name(folder)
        link = self._sync.get(self._config.url, folder)
        if link is None:
            return ProjectStatus(cloud.LOCAL_ONLY, name, str(folder), None, local, size)
        if remote is None:
            row = self._cloud_row(link.project_id)
        else:
            row = next((p for p in remote if p.id == link.project_id), None)
        status = cloud.classify(local, link.base_hash, row.content_hash if row else None)
        return ProjectStatus(status, row.name if row else name, str(folder), row, local, size)

    def sync_overview(self, extra_folders: list[str | Path] = ()) -> list[ProjectStatus]:
        """Every cloud project with its local status, plus linked or given folders not in the cloud."""
        remote = self.list_cloud_projects()
        out: list[ProjectStatus] = []
        seen_folders: set[str] = set()
        linked: dict[str, list[str]] = {}
        for link in self._sync.links(self._config.url):
            if Path(link.path, "project.json").is_file():
                linked.setdefault(link.project_id, []).append(link.path)
        for p in remote:
            folders = linked.get(p.id, [])
            if not folders:
                out.append(ProjectStatus(cloud.REMOTE_ONLY, p.name, "", p))
            for f in folders:
                seen_folders.add(_key(f))
                out.append(self.project_status(f, remote))
        for f in [*[x for v in linked.values() for x in v], *extra_folders]:
            if _key(f) in seen_folders or not Path(f, "project.json").is_file():
                continue
            seen_folders.add(_key(f))
            out.append(self.project_status(f, remote))
        return out

    def upload_preview(self, folder: str | Path) -> UploadPreview:
        """Packed size, parts and per-run breakdown of a folder (local only: no network, no archive)."""
        folder = Path(folder)
        entries, total = cloud.packed_sizes(folder)
        groups = tuple(cloud.size_breakdown(folder, entries))
        return UploadPreview(_project_name(folder), total, _disk_bytes(folder), len(entries), groups,
                             self._config.part_bytes, self._config.max_project_mb * 1024 * 1024)

    def upload_project(self, folder: str | Path, *, force: bool = False,
                       progress: Callable[[str], None] | None = None) -> ProjectStatus:
        """Upload a project folder (new cloud project or a new version of its linked one).

        Refuses with :class:`SyncConflictError` when the cloud copy changed since
        this folder last synced (unless ``force``: "keep mine"). A folder whose
        content equals the cloud copy uploads nothing.

        The archive is packed into a temporary file and sent in parts of at most
        :attr:`AccountConfig.part_bytes` (45 MiB with the free plan's 50 MB file
        limit), so a project's size is bounded by ``max_project_mb``, not by the
        per-file limit: measured before this, one more 3D run pushed a typical
        project (seven Draft runs, 54 MB packed) over 50 MB and the only way out
        offered was deleting results. Every part is uploaded before the row
        changes, and the previous version's objects are deleted only after the
        row points at the new ones, so an interrupted upload leaves the old cloud
        copy intact.
        """
        folder = Path(folder)
        uid = self._uid()
        say = progress or (lambda _m: None)
        say("Packing the project")
        with tempfile.TemporaryFile(prefix="aerolume-upload-") as fh:
            info = cloud.pack_project_to(folder, fh)
            if info.packed_bytes > self._config.max_project_mb * 1024 * 1024:
                raise TooLargeError(self._too_large_text(folder, info), code="too_large")
            link = self._sync.get(self._config.url, folder)
            row = self._cloud_row(link.project_id) if link else None
            if row is not None and not force:
                status = cloud.classify(info.content_hash, link.base_hash, row.content_hash)
                if status == cloud.SYNCED:
                    self._sync.set(self._config.url, folder, row.id, row.content_hash)
                    return ProjectStatus(cloud.SYNCED, row.name, str(folder), row, info.content_hash)
                if status in (cloud.CONFLICT, cloud.REMOTE_CHANGES):
                    raise SyncConflictError(
                        "The cloud copy changed since this folder was last synced"
                        + (" and so did this folder." if status == cloud.CONFLICT else ".")
                        + " Nothing was uploaded.", local_hash=info.content_hash,
                        remote_hash=row.content_hash, project_id=row.id)
            pid = row.id if row else str(uuid.uuid4())
            name = _project_name(folder)
            obj = f"{uid}/{pid}/{info.content_hash}.zip"
            n = cloud.part_count_for(info.packed_bytes, self._config.part_bytes)
            paths = cloud.part_paths(obj, n)
            old_paths = set(row.object_paths) if row is not None else set()
            self._upload_parts(fh, paths, info.packed_bytes, say, keep=old_paths)
        meta = {"name": name, "content_hash": info.content_hash, "size_bytes": info.packed_bytes,
                "file_count": info.file_count, "object_path": obj, "part_count": n,
                "app_version": _app_version()}
        ours = [p for p in paths if p not in old_paths]       # objects only this upload created
        say("Recording the new version")
        try:
            if row is None:
                rows = self._authed(lambda tok: self._backend().insert(
                    "projects", tok, {"id": pid, "owner": uid, **meta}, what="Recording the cloud project"))
            else:
                seen = row.content_hash
                rows = self._authed(lambda tok: self._backend().update(
                    "projects", tok, meta, {"id": pid, "content_hash": seen}, what="Recording the new version"))
        except RemoteError as exc:
            self._quiet_remove(PROJECTS_BUCKET, ours)
            raise self._schema_hint(exc) from exc
        if row is not None and not rows:        # someone else uploaded between our read and this write
            self._quiet_remove(PROJECTS_BUCKET, ours)
            fresh = self._cloud_row(pid)
            raise SyncConflictError("Another computer uploaded this project a moment ago. "
                                    "Nothing was overwritten.", local_hash=info.content_hash,
                                    remote_hash=fresh.content_hash if fresh else "", project_id=pid)
        stale = sorted(old_paths - set(paths))
        if stale:
            self._quiet_remove(PROJECTS_BUCKET, stale)
        new = CloudProject.from_row(rows[0]) if rows else None
        self._sync.set(self._config.url, folder, pid, info.content_hash)
        log.info("uploaded project %s (%d files, %d bytes, %d part(s))", pid, info.file_count,
                 info.packed_bytes, n)
        return ProjectStatus(cloud.SYNCED, name, str(folder), new, info.content_hash)

    def _upload_parts(self, fh, paths: list[str], total: int, say: Callable[[str], None], *,
                      keep: set[str] = frozenset()) -> None:
        """Send the packed archive in ``fh`` as ``paths`` (one object, or numbered parts).

        A failure part-way removes the parts this call already sent (best effort,
        never one in ``keep``: those belong to the current cloud version) and
        re-raises; nothing in the database refers to them yet.
        """
        n, part = len(paths), self._config.part_bytes
        sent: list[str] = []
        try:
            for i, path in enumerate(paths):
                fh.seek(i * part)
                data = fh.read(part if n > 1 else total)
                if n == 1:
                    say(f"Uploading {_mb(total)}")
                else:
                    say(f"Uploading part {i + 1} of {n} ("
                        + (f"{_mb(i * part)} of {_mb(total)} sent)" if i else f"{_mb(total)} in all)"))
                ctype = ZIP_TYPE if n == 1 else PART_TYPE
                self._authed(lambda tok, p=path, d=data, c=ctype: self._backend().upload(
                    PROJECTS_BUCKET, p, d, tok, content_type=c, upsert=True, what="Uploading the project"))
                if path not in keep:
                    sent.append(path)
        except TooLargeError as exc:
            self._quiet_remove(PROJECTS_BUCKET, sent)
            raise TooLargeError(
                f"The server refused a {_mb(min(part, total))} file: its size limit is lower than the "
                f"{self._config.max_upload_mb} MB this app assumes. Set \"max_upload_mb\" in account.json "
                "to the bucket's limit (supabase/README.md); projects then go up in smaller parts.",
                code="too_large", status=exc.status) from exc
        except RemoteError as exc:
            self._quiet_remove(PROJECTS_BUCKET, sent)
            raise self._schema_hint(exc, parts=n > 1) from exc
        except BaseException:
            self._quiet_remove(PROJECTS_BUCKET, sent)
            raise

    def _too_large_text(self, folder: Path, info: cloud.PackInfo) -> str:
        """Why a project is over ``max_project_mb``, naming what takes the space (largest first)."""
        groups = cloud.size_breakdown(folder, info.entry_bytes)
        top = "; ".join(f"{g.label} {_mb(g.size_bytes)}" for g in groups[:3])
        exports = next((g for g in groups if g.kind == "exports"), None)
        extra = (" Exported files can be written again from their results at any time."
                 if exports is not None and exports.size_bytes >= info.packed_bytes * 0.1 else "")
        return (f"The packed project is {_mb(info.packed_bytes)}, over this server's project limit of "
                f"{self._config.max_project_mb} MB. The largest items: {top}.{extra} On a paid plan, raise "
                "\"max_project_mb\" in account.json (docs/accounts.md, Upload size).")

    @staticmethod
    def _schema_hint(exc: RemoteError, *, parts: bool = False) -> AccountError:
        """Name the migration to apply when the failure is a server without schema version 2.

        Such a server rejects the ``part_count`` column when the row is written
        (PostgREST: "could not find the column", mapped to ``schema_missing``;
        the table itself was just read, so the column is what is missing) and
        refuses the part MIME type on a multi-part upload (``mime``).
        """
        if exc.code == "schema_missing" or (parts and exc.code == "mime"):
            return RemoteError("The server's database is older than this version of Aerolume. Apply the "
                               f"files in supabase/migrations that it lacks (the newest is {PARTS_MIGRATION}; "
                               "see supabase/README.md), then try again.",
                               code="schema_outdated", status=exc.status)
        return exc

    def download_project(self, project_id: str, dest_parent: str | Path, *, replace: str | Path | None = None,
                         progress: Callable[[str], None] | None = None) -> Path:
        """Download a cloud project into ``dest_parent`` (a new folder), or over ``replace``.

        With ``replace`` (a local folder linked to this project), the folder is
        swapped for the cloud version. If that folder had changes that were never
        uploaded, it is kept beside the new one as ``<name> (local copy <date>)``
        rather than deleted.
        """
        say = progress or (lambda _m: None)
        row = self._cloud_row(project_id)
        if row is None:
            raise RemoteError("This project is no longer in the cloud.", code="not_found")
        # Parts are reassembled into a temporary file, not memory: a project may be
        # several hundred MB, and the unpack reads the zip from the file directly.
        with tempfile.TemporaryFile(prefix="aerolume-download-") as fh:
            paths, got = row.object_paths, 0
            for i, path in enumerate(paths):
                if len(paths) == 1:
                    say(f"Downloading {_mb(row.size_bytes)}")
                else:
                    say(f"Downloading part {i + 1} of {len(paths)} ("
                        + (f"{_mb(got)} of {_mb(row.size_bytes)} received)" if i else f"{_mb(row.size_bytes)} in all)"))
                data = self._authed(lambda tok, p=path: self._backend().download(
                    PROJECTS_BUCKET, p, tok, what="Downloading the project"))
                fh.write(data)
                got += len(data)
                del data
            fh.flush()
            fh.seek(0)
            say("Unpacking")
            if replace is None:
                dest = _free_folder(Path(dest_parent), cloud.safe_folder_name(row.name))
                cloud.unpack_project(fh, dest, expected_hash=row.content_hash)
            else:
                dest = Path(replace)
                staged = dest.with_name(f".trash-incoming-{secrets.token_hex(4)}-{dest.name}")
                cloud.unpack_project(fh, staged, expected_hash=row.content_hash)
        if replace is not None:
            link = self._sync.get(self._config.url, dest)
            keep = dest.exists() and (link is None or cloud.content_hash(dest) != link.base_hash)
            if dest.exists():
                if keep:
                    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H%M")
                    backup = _free_folder(dest.parent, f"{dest.name.removesuffix('.aerolume')} "
                                                       f"(local copy {stamp})")
                    dest.rename(backup)
                else:
                    from ..project.fileio import remove_tree
                    remove_tree(dest)
            staged.rename(dest)
        self._sync.set(self._config.url, dest, row.id, row.content_hash)
        return dest

    def delete_cloud_project(self, project_id: str) -> None:
        """Delete the cloud copy (row and file). Local folders are not touched; their link is forgotten."""
        row = self._cloud_row(project_id)
        if row is None:
            self._sync.forget(self._config.url, project_id=project_id)
            return
        self._authed(lambda tok: self._backend().delete("projects", tok, {"id": project_id},
                                                        what="Deleting the cloud project"))
        self._quiet_remove(PROJECTS_BUCKET, row.object_paths)
        self._sync.forget(self._config.url, project_id=project_id)

    def link_status(self, folder: str | Path) -> cloud.SyncLink | None:
        return self._sync.get(self._config.url, folder)

    def unlink_folder(self, folder: str | Path) -> None:
        self._sync.forget(self._config.url, folder=folder)

    # -- shares -----------------------------------------------------------------
    def create_share(self, files: list[str | Path], *, title: str, project_id: str = "", sim_id: str = "",
                     expires_days: int = 30) -> ShareLink:
        """Publish files (the first must be the report .html) as read-only links.

        The files go to the *private* ``shares`` bucket under a random token
        folder (32 bytes, base64url) and are handed out as **signed URLs** that
        Storage itself stops honouring after ``expires_days``; revoking deletes
        the files, which kills the links at once. A public bucket would have
        made "expires" a promise nothing enforces. The ``shares`` row (owner-only)
        keeps the links so they can be copied again later.
        """
        uid = self._uid()
        if not 1 <= int(expires_days) <= MAX_SHARE_DAYS:
            raise AccountError(f"Choose an expiry between 1 and {MAX_SHARE_DAYS} days.", code="share_expiry")
        paths = [Path(f) for f in files]
        if not paths or paths[0].suffix.lower() not in (".html", ".htm"):
            raise AccountError("Choose a report in HTML format to share.", code="share_html")
        total = 0
        for p in paths:
            if p.suffix.lower() not in SHARE_TYPES:
                raise AccountError(f"{p.name}: only HTML, PDF, PNG and JPEG files can be shared.",
                                   code="share_type")
            total += p.stat().st_size
        if total > self._config.max_upload_mb * 1024 * 1024:
            raise TooLargeError(f"The files add up to {_mb(total)}; the upload limit is "
                                f"{self._config.max_upload_mb} MB.", code="too_large")
        token = secrets.token_urlsafe(32)
        names = ["index.html", *[_share_name(p, i) for i, p in enumerate(paths[1:], 1)]]
        seconds = int(expires_days) * 86400
        expires = (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(timespec="seconds")
        row = {"token": token, "owner": uid, "title": title.strip()[:200] or "Aerolume result",
               "files": names, "links": {}, "sim_id": sim_id or None, "project_id": project_id or None,
               "expires_at": expires}
        self._authed(lambda tok: self._backend().insert("shares", tok, row, what="Creating the share"))
        links: dict[str, str] = {}
        try:
            for p, name in zip(paths, names, strict=True):
                data, obj = p.read_bytes(), f"{token}/{name}"
                ctype = SHARE_TYPES[p.suffix.lower()]
                self._authed(lambda tok, d=data, o=obj, c=ctype: self._backend().upload(
                    SHARES_BUCKET, o, d, tok, content_type=c, what="Uploading the shared files"))
                links[name] = self._authed(lambda tok, o=obj: self._backend().sign_url(
                    SHARES_BUCKET, o, tok, seconds))
            rows = self._authed(lambda tok: self._backend().update(
                "shares", tok, {"links": links}, {"token": token}, what="Saving the share links"))
        except BaseException:
            self._revoke(token, names)
            raise
        log.info("created a share with %d file(s), expiring in %d days", len(names), expires_days)
        return _share_from_row(rows[0] if rows else {**row, "links": links})

    def list_shares(self) -> list[ShareLink]:
        uid = self._uid()
        rows = self._authed(lambda tok: self._backend().select("shares", tok, {"owner": uid},
                                                               order="created_at.desc", what="Listing shares"))
        return [_share_from_row(r) for r in rows]

    def revoke_share(self, token: str) -> None:
        """Delete the shared files (the links stop working immediately) and the share row."""
        rows = self._authed(lambda tok: self._backend().select("shares", tok, {"token": token},
                                                               what="Reading the share"))
        names = list(rows[0].get("files") or []) if rows else []
        self._revoke(token, names or ["index.html"])

    def _revoke(self, token: str, names: list[str]) -> None:
        # Files first: the storage policy checks ownership through the share row.
        self._quiet_remove(SHARES_BUCKET, [f"{token}/{n}" for n in names])
        self._authed(lambda tok: self._backend().delete("shares", tok, {"token": token},
                                                        what="Revoking the share"))

    def resolve_share(self, token: str) -> dict | None:
        """Public facts of a live share (title, files, dates) by token, without signing in.

        Goes through the ``public.get_share`` function (security definer): the
        table itself is owner-only, so tokens cannot be listed, but whoever
        holds a token can look its share up - the "readable by token" rule.
        """
        rows = self._backend().rpc("get_share", None, {"share_token": token}, what="Opening the share")
        if isinstance(rows, list):
            return rows[0] if rows else None
        return rows or None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _project_name(folder: Path) -> str:
    try:
        name = json.loads((folder / "project.json").read_text(encoding="utf-8-sig")).get("name")
        if name:
            return str(name)
    except (OSError, ValueError, AttributeError):
        pass
    return folder.name.removesuffix(".aerolume")


def _disk_bytes(folder: Path) -> int:
    """Bytes on disk of the files an upload would pack (same file list as the content hash)."""
    total = 0
    for _rel, p in cloud.project_files(folder):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def _free_folder(parent: Path, stem: str) -> Path:
    cand = parent / f"{stem}.aerolume"
    n = 2
    while cand.exists():
        cand = parent / f"{stem} ({n}).aerolume"
        n += 1
    return cand


def _share_name(p: Path, i: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", p.stem).strip("-")[:60] or "file"
    return f"{i:02d}-{stem}{p.suffix.lower()}"


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 100 * 1024 else f"{n / 1024:.0f} kB"


def _app_version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return ""


_default: AccountService | None = None
_default_lock = threading.Lock()


def get_service() -> AccountService:
    """The process-wide service (configured from Preferences / the environment)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = AccountService()
        return _default


def set_service(service: AccountService | None) -> None:
    """Replace the process-wide service (tests; a GUI pointed at the mock server)."""
    global _default
    with _default_lock:
        _default = service
