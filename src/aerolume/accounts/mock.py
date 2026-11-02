"""An in-process stand-in for a Supabase project (tests, demos, offline development).

``MockSupabase`` runs :mod:`http.server` on 127.0.0.1 and implements the subset
of Auth (GoTrue), PostgREST and Storage that :mod:`aerolume.accounts` uses,
with the same URL layout, JSON shapes, status codes and error codes as the real
services (including their quirks: Storage reports a missing object as HTTP 400
with ``"statusCode": "404"``; sign-up of an existing confirmed address returns a
user with no identities instead of an error). It also enforces the access
rules of ``supabase/migrations`` - owner-only rows, owner-only storage folders,
share files writable only by the share's owner - so a client bug that sends the
wrong token or path fails here as it would in production.

Nothing leaves the machine: emails are appended to :attr:`MockSupabase.outbox`
(and printed when run as a program), OAuth "providers" sign in a fixed identity
at once, and JWTs are HS256-signed with a random per-instance secret.

Run it to try the account UI without any cloud project::

    python -m aerolume.accounts.mock            # prints the URL and the anon key
    python -m aerolume.accounts.mock --autoconfirm --port 54399

then paste both into Account settings. State lives in memory only and is
gone when the process stops.

It is *not* a security boundary or a full emulation: one schema, no SQL, no
rate limits, no email delivery, no resumable (TUS) uploads. Large project
archives do not need those: the client sends them as several ordinary objects.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Self
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{32,}$")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="seconds")


@dataclass
class Email:
    to: str
    kind: str            # "signup" | "magiclink" | "recovery"
    code: str            # the 6-digit code (Supabase's {{ .Token }})
    link: str            # the verify link (Supabase's {{ .ConfirmationURL }})


@dataclass
class _User:
    id: str
    email: str
    password: str | None
    confirmed: bool
    provider: str = "email"
    meta: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"id": self.id, "aud": "authenticated", "role": "authenticated", "email": self.email,
                "email_confirmed_at": _iso(self.created) if self.confirmed else None,
                "app_metadata": {"provider": self.provider, "providers": [self.provider]},
                "user_metadata": dict(self.meta), "identities": [{"provider": self.provider}],
                "created_at": _iso(self.created)}


class MockSupabase:
    """Start with ``with MockSupabase() as mock:``; ``mock.url`` and ``mock.anon_key`` connect to it."""

    TABLES: ClassVar[dict[str, str]] = {"profiles": "id", "projects": "id", "shares": "token"}
    OWNER: ClassVar[dict[str, str]] = {"profiles": "id", "projects": "owner", "shares": "owner"}
    BUCKETS: ClassVar[dict[str, tuple[str, ...]]] = {
        "projects": ("application/zip", "application/octet-stream"),
        "avatars": ("image/png", "image/jpeg"),
        "shares": ("text/html", "application/pdf", "image/png", "image/jpeg")}
    #: What schema version 2 (supabase/migrations/20261030120000_aerolume_project_parts.sql)
    #: added; with ``schema_version = 1`` the mock behaves like a database without it.
    V2_COLUMNS: ClassVar[dict[str, tuple[str, ...]]] = {"projects": ("part_count",)}
    V2_TYPES: ClassVar[dict[str, tuple[str, ...]]] = {"projects": ("application/octet-stream",)}

    def __init__(self, *, port: int = 0, autoconfirm: bool = False, access_ttl: float = 3600.0,
                 providers: tuple[str, ...] = ("github", "google"), echo: bool = False) -> None:
        self.anon_key = "mock-anon-" + secrets.token_urlsafe(24)
        self.autoconfirm = autoconfirm
        self.access_ttl = access_ttl
        self.providers = set(providers)
        self.disable_signup = False
        self.schema_version: int | None = 2    # None: the version function is missing
        self.file_size_limit = 50 * 1024 * 1024
        self.oauth_identity = {"github": "octo@example.org", "google": "gee@example.org"}
        self.echo = echo
        self.outbox: list[Email] = []
        self.requests: list[tuple[str, str, dict[str, str]]] = []   # (method, path, headers) for assertions
        self._secret = secrets.token_bytes(32)
        self._lock = threading.RLock()
        self._users: dict[str, _User] = {}
        self._refresh: dict[str, dict] = {}       # refresh token -> {user, session, used}
        self._codes: dict[str, dict] = {}         # email code -> {...}
        self._links: dict[str, dict] = {}         # link token -> {...}
        self._flows: dict[str, dict] = {}         # pkce auth code -> {user, challenge}
        self.rows: dict[str, dict[str, dict]] = {t: {} for t in self.TABLES}
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._server.daemon_threads = True
        self._server.mock = self  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> MockSupabase:
        self._thread = threading.Thread(target=self._server.serve_forever, name="mock-supabase", daemon=True,
                                        kwargs={"poll_interval": 0.1})
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- test helpers ------------------------------------------------------------
    def last_email(self, to: str | None = None) -> Email:
        with self._lock:
            for e in reversed(self.outbox):
                if to is None or e.to == to:
                    return e
        raise LookupError(f"no email to {to}")

    def add_user(self, email: str, password: str, *, confirmed: bool = True, display_name: str = "") -> str:
        with self._lock:
            u = _User(str(uuid.uuid4()), email.lower(), password, confirmed,
                      meta={"display_name": display_name} if display_name else {})
            self._users[u.id] = u
            self._profile_trigger(u)
            return u.id

    def invalidate_access_tokens(self) -> None:
        """Rotate the JWT secret: every issued access token now fails as "invalid JWT"."""
        with self._lock:
            self._secret = secrets.token_bytes(32)

    def user_id(self, email: str) -> str:
        u = self._user_by_email(email)
        assert u is not None
        return u.id

    # -- jwt ---------------------------------------------------------------------
    def _jwt(self, claims: dict) -> str:
        head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body = _b64(json.dumps(claims, separators=(",", ":")).encode())
        sig = _b64(hmac.new(self._secret, f"{head}.{body}".encode(), hashlib.sha256).digest())
        return f"{head}.{body}.{sig}"

    def _verify_jwt(self, token: str) -> dict | None:
        try:
            head, body, sig = token.split(".")
        except ValueError:
            return None
        good = _b64(hmac.new(self._secret, f"{head}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(good, sig):
            return None
        try:
            claims = json.loads(_unb64(body))
        except ValueError:
            return None
        if float(claims.get("exp", 0)) < time.time():
            return None
        return claims

    def _session(self, user: _User) -> dict:
        now = time.time()
        sid = str(uuid.uuid4())
        access = self._jwt({"sub": user.id, "email": user.email, "role": "authenticated", "aud": "authenticated",
                            "session_id": sid, "iat": int(now), "exp": int(now + self.access_ttl)})
        refresh = secrets.token_urlsafe(18)
        self._refresh[refresh] = {"user": user.id, "session": sid, "used": False}
        return {"access_token": access, "token_type": "bearer", "expires_in": self.access_ttl,
                "expires_at": int(now + self.access_ttl), "refresh_token": refresh, "user": user.public()}

    # -- auth helpers ------------------------------------------------------------
    def _user_by_email(self, email: str) -> _User | None:
        email = (email or "").lower()
        return next((u for u in self._users.values() if u.email == email), None)

    def _profile_trigger(self, u: _User) -> None:
        """What the ``on_auth_user_created`` trigger of the migrations does."""
        now = _iso(time.time())
        self.rows["profiles"].setdefault(u.id, {
            "id": u.id, "display_name": str(u.meta.get("display_name") or u.meta.get("full_name") or ""),
            "affiliation": "", "avatar_path": None, "preferences": {}, "created_at": now, "updated_at": now})

    def _send(self, user: _User, kind: str, redirect_to: str, challenge: str) -> None:
        code = f"{secrets.randbelow(10**6):06d}"
        link_token = secrets.token_urlsafe(16)
        info = {"user": user.id, "kind": kind, "redirect_to": redirect_to, "challenge": challenge,
                "expires": time.time() + 3600, "code": code, "link": link_token}
        self._codes[f"{user.email}|{code}"] = info
        self._links[link_token] = info
        q = urlencode({"token": link_token, "type": kind, "redirect_to": redirect_to})
        mail = Email(user.email, kind, code, f"{self.url}/auth/v1/verify?{q}")
        self.outbox.append(mail)
        if self.echo:
            print(f"[mock email] to {mail.to} ({kind}): code {mail.code}\n  link {mail.link}", flush=True)

    def _consume(self, info: dict) -> None:
        self._codes.pop(f"{self._users[info['user']].email}|{info['code']}", None)
        self._links.pop(info["link"], None)

    # -- request dispatch --------------------------------------------------------
    def handle(self, h: _Handler, method: str) -> None:
        parts = urlsplit(h.path)
        path, query = unquote(parts.path), {k: v[-1] for k, v in parse_qs(parts.query).items()}
        hdr = {k.lower(): v for k, v in h.headers.items()}
        self.requests.append((method, path, hdr))
        length = int(hdr.get("content-length") or 0)
        raw = h.rfile.read(length) if length else b""
        browser = (method == "GET" and (path in ("/auth/v1/verify", "/auth/v1/authorize")
                                        or path.startswith("/storage/v1/object/sign/")))
        if not browser and hdr.get("apikey") != self.anon_key:
            return h.reply(401, {"message": "Invalid API key", "hint": "Double check your Supabase `anon` "
                                 "or `service_role` API key."})
        with self._lock:
            try:
                if path.startswith("/auth/v1/"):
                    return self._auth(h, method, path[len("/auth/v1"):], query, hdr, raw)
                if path.startswith("/rest/v1/"):
                    return self._rest(h, method, path[len("/rest/v1/"):], query, hdr, raw)
                if path.startswith("/storage/v1/"):
                    return self._storage(h, method, path[len("/storage/v1/"):], query, hdr, raw)
            except _Reply as r:
                return h.reply(r.status, r.body, r.headers)
        h.reply(404, {"message": "no route"})

    @staticmethod
    def _json(raw: bytes) -> Any:
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            raise _Reply(400, {"message": "invalid JSON"}) from None

    def _claims(self, hdr: dict, *, required: bool = True) -> dict | None:
        auth = hdr.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            if required:
                raise _Reply(401, {"code": 401, "error_code": "no_authorization",
                                   "msg": "This endpoint requires a Bearer token"})
            return None
        claims = self._verify_jwt(auth[7:].strip())
        if claims is None:
            raise _Reply(403, {"code": 403, "error_code": "bad_jwt",
                               "msg": "invalid JWT: unable to parse or verify signature, token is expired"})
        return claims

    # -- auth --------------------------------------------------------------------
    def _auth(self, h: _Handler, method: str, path: str, q: dict, hdr: dict, raw: bytes) -> None:
        def err(status: int, code: str, msg: str) -> _Reply:
            return _Reply(status, {"code": status, "error_code": code, "msg": msg})

        if method == "GET" and path == "/settings":
            ext = {"email": True, **{p: p in self.providers for p in ("github", "google", "apple")}}
            return h.reply(200, {"external": ext, "disable_signup": self.disable_signup,
                                 "mailer_autoconfirm": self.autoconfirm})
        if method == "POST" and path == "/signup":
            b = self._json(raw)
            email, pw = str(b.get("email") or "").lower(), str(b.get("password") or "")
            if self.disable_signup:
                raise err(422, "signup_disabled", "Signups not allowed for this instance")
            if len(pw) < 6:
                raise err(422, "weak_password", "Password should be at least 6 characters.")
            u = self._user_by_email(email)
            if u is not None and u.confirmed:
                # real GoTrue: an obfuscated user, no error, so emails cannot be probed
                return h.reply(200, {"id": str(uuid.uuid4()), "email": email, "identities": [],
                                     "aud": "authenticated", "role": "authenticated"})
            if u is None:
                u = _User(str(uuid.uuid4()), email, pw, self.autoconfirm, meta=dict(b.get("data") or {}))
                self._users[u.id] = u
                self._profile_trigger(u)
            if self.autoconfirm:
                return h.reply(200, self._session(u))
            self._send(u, "signup", q.get("redirect_to", ""), str(b.get("code_challenge") or ""))
            return h.reply(200, u.public())
        if method == "POST" and path == "/token":
            b = self._json(raw)
            grant = q.get("grant_type")
            if grant == "password":
                u = self._user_by_email(str(b.get("email") or ""))
                if u is None or u.password is None or not hmac.compare_digest(u.password, str(b.get("password"))):
                    raise err(400, "invalid_credentials", "Invalid login credentials")
                if not u.confirmed:
                    raise err(400, "email_not_confirmed", "Email not confirmed")
                return h.reply(200, self._session(u))
            if grant == "refresh_token":
                rt = self._refresh.get(str(b.get("refresh_token") or ""))
                if rt is None:
                    raise err(400, "refresh_token_not_found", "Invalid Refresh Token: Refresh Token Not Found")
                if rt["used"]:
                    raise err(400, "refresh_token_already_used", "Invalid Refresh Token: Already Used")
                rt["used"] = True
                return h.reply(200, self._session(self._users[rt["user"]]))
            if grant == "pkce":
                flow = self._flows.pop(str(b.get("auth_code") or ""), None)
                if flow is None:
                    raise err(404, "flow_state_not_found", "invalid flow state, no valid flow state found")
                verifier = str(b.get("code_verifier") or "")
                want = _b64(hashlib.sha256(verifier.encode()).digest())
                if not hmac.compare_digest(want, flow["challenge"]):
                    raise err(403, "bad_code_verifier", "code challenge does not match previously saved code verifier")
                return h.reply(200, self._session(self._users[flow["user"]]))
            raise err(400, "unsupported_grant_type", "unsupported_grant_type")
        if method == "POST" and path == "/otp":
            b = self._json(raw)
            email = str(b.get("email") or "").lower()
            u = self._user_by_email(email)
            if u is None:
                if not b.get("create_user", True) or self.disable_signup:
                    raise err(422, "otp_disabled", "Signups not allowed for otp")
                u = _User(str(uuid.uuid4()), email, None, False)
                self._users[u.id] = u
                self._profile_trigger(u)
            self._send(u, "magiclink", q.get("redirect_to", ""), str(b.get("code_challenge") or ""))
            return h.reply(200, {})
        if method == "POST" and path == "/recover":
            b = self._json(raw)
            u = self._user_by_email(str(b.get("email") or ""))
            if u is not None:          # silent for unknown emails, like the real service
                self._send(u, "recovery", q.get("redirect_to", ""), str(b.get("code_challenge") or ""))
            return h.reply(200, {})
        if method == "POST" and path == "/verify":
            b = self._json(raw)
            info = self._codes.get(f"{str(b.get('email') or '').lower()}|{b.get('token')}")
            kind = str(b.get("type") or "")
            ok_kinds = {"email": ("magiclink", "signup"), "magiclink": ("magiclink",), "signup": ("signup",),
                        "recovery": ("recovery",)}.get(kind, ())
            if info is None or info["kind"] not in ok_kinds or info["expires"] < time.time():
                raise err(403, "otp_expired", "Token has expired or is invalid")
            self._consume(info)
            u = self._users[info["user"]]
            u.confirmed = True
            return h.reply(200, self._session(u))
        if method == "GET" and path == "/verify":       # the link in the email, opened in a browser
            info = self._links.get(q.get("token", ""))
            redirect = q.get("redirect_to") or (info or {}).get("redirect_to") or ""
            if info is None or info["expires"] < time.time():
                return h.redirect(f"{redirect}?{urlencode({'error': 'access_denied', 'error_code': 'otp_expired', 'error_description': 'Email link is invalid or has expired'})}")
            self._consume(info)
            u = self._users[info["user"]]
            u.confirmed = True
            code = str(uuid.uuid4())
            self._flows[code] = {"user": u.id, "challenge": info["challenge"]}
            return h.redirect(f"{redirect}?{urlencode({'code': code})}")
        if method == "GET" and path == "/authorize":
            provider, redirect = q.get("provider", ""), q.get("redirect_to", "")
            if provider not in self.providers:
                return h.reply(400, {"code": 400, "error_code": "validation_failed",
                                     "msg": "Unsupported provider: provider is not enabled"})
            if q.get("code_challenge_method", "").lower() != "s256" or not q.get("code_challenge"):
                return h.reply(400, {"code": 400, "error_code": "validation_failed", "msg": "PKCE required"})
            email = self.oauth_identity.get(provider, f"{provider}@example.org")
            u = self._user_by_email(email)
            if u is None:
                u = _User(str(uuid.uuid4()), email, None, True, provider,
                          {"full_name": f"{provider.title()} User"})
                self._users[u.id] = u
                self._profile_trigger(u)
            code = str(uuid.uuid4())
            self._flows[code] = {"user": u.id, "challenge": q["code_challenge"]}
            return h.redirect(f"{redirect}?{urlencode({'code': code})}")
        if path == "/user":
            claims = self._claims(hdr)
            u = self._users.get(claims["sub"])
            if u is None:
                raise err(404, "user_not_found", "User not found")
            if method == "PUT":
                b = self._json(raw)
                if "password" in b:
                    if u.password is not None and hmac.compare_digest(u.password, str(b["password"])):
                        raise err(422, "same_password", "New password should be different from the old password.")
                    u.password = str(b["password"])
                if isinstance(b.get("data"), dict):
                    u.meta.update(b["data"])
            return h.reply(200, u.public())
        if method == "POST" and path == "/logout":
            claims = self._claims(hdr)
            scope = q.get("scope", "global")
            for info in self._refresh.values():
                if info["user"] == claims["sub"] and (scope == "global" or info["session"] == claims.get("session_id")):
                    info["used"] = True
            return h.reply(204, None)
        h.reply(404, {"code": 404, "error_code": "not_found", "msg": "Not found"})

    # -- PostgREST -----------------------------------------------------------------
    def _rest(self, h: _Handler, method: str, path: str, q: dict, hdr: dict, raw: bytes) -> None:
        if path.startswith("rpc/"):
            return self._rpc(h, path[4:], hdr, raw)
        table = path
        if table not in self.TABLES:
            raise _Reply(404, {"code": "PGRST205", "message": f"Could not find the table 'public.{table}' in the "
                                                             "schema cache"})
        claims = self._claims(hdr, required=False)
        if claims is None:   # migrations revoke table access from anon
            raise _Reply(401, {"code": "42501", "message": f"permission denied for table {table}"})
        uid, pk, owner = claims["sub"], self.TABLES[table], self.OWNER[table]
        rows = self.rows[table]
        filters = {k: v[3:] for k, v in q.items() if v.startswith("eq.")}
        prefer = hdr.get("prefer", "")

        def visible() -> list[dict]:
            return [r for r in rows.values() if str(r.get(owner)) == uid
                    and all(str(r.get(k)) == v for k, v in filters.items())]

        now = _iso(time.time())
        if method == "GET":
            out = visible()
            if "order" in q:
                col, _, direction = q["order"].partition(".")
                out.sort(key=lambda r: str(r.get(col) or ""), reverse=direction == "desc")
            if "limit" in q:
                out = out[: int(q["limit"])]
            return h.reply(200, copy.deepcopy(out))
        if method == "POST":
            body = self._json(raw)
            items = body if isinstance(body, list) else [body]
            merge = "resolution=merge-duplicates" in prefer
            out = []
            for item in items:
                row = dict(item)
                row.setdefault(owner, uid)
                if str(row[owner]) != uid:
                    raise _Reply(403, {"code": "42501", "message": f'new row violates row-level security policy '
                                                                   f'for table "{table}"'})
                self._columns(table, row)
                self._check(table, row)
                key = str(row.get(pk) or "")
                if table == "projects" and not key:
                    key = row[pk] = str(uuid.uuid4())
                if key in rows:
                    if not merge:
                        raise _Reply(409, {"code": "23505", "message": "duplicate key value violates unique "
                                                                       f'constraint "{table}_pkey"'})
                    if str(rows[key].get(owner)) != uid:
                        raise _Reply(403, {"code": "42501", "message": "new row violates row-level security"})
                    rows[key].update(row, updated_at=now)
                else:
                    rows[key] = {**self._defaults(table, now), **row}
                out.append(copy.deepcopy(rows[key]))
            return h.reply(201, out if "return=representation" in prefer else None)
        if method == "PATCH":
            values = self._json(raw)
            self._columns(table, values)
            out = []
            for r in visible():
                cand = {**r, **values}
                if str(cand.get(owner)) != uid:
                    raise _Reply(403, {"code": "42501", "message": "new row violates row-level security"})
                self._check(table, cand)
                r.update(values, updated_at=now)
                out.append(copy.deepcopy(r))
            return h.reply(200, out if "return=representation" in prefer else None)
        if method == "DELETE":
            out = visible()
            for r in out:
                del rows[str(r[pk])]
            return h.reply(200, copy.deepcopy(out) if "return=representation" in prefer else None)
        raise _Reply(405, {"message": "method not allowed"})

    def _columns(self, table: str, row: dict) -> None:
        """PostgREST's answer to a column the schema does not have (PGRST204)."""
        if self.schema_version == 1:
            for col in self.V2_COLUMNS.get(table, ()):
                if col in row:
                    raise _Reply(400, {"code": "PGRST204", "details": None, "hint": None,
                                       "message": f"Could not find the '{col}' column of '{table}' in the "
                                                  "schema cache"})

    @staticmethod
    def _defaults(table: str, now: str) -> dict:
        base = {"created_at": now, "updated_at": now}
        if table == "projects":
            base.update(part_count=1)
        if table == "profiles":
            base.update(display_name="", affiliation="", avatar_path=None, preferences={})
        elif table == "shares":
            base.update(files=[], links={}, expires_at=None, project_id=None, sim_id=None)
        return base

    @staticmethod
    def _check(table: str, row: dict) -> None:
        """The CHECK constraints of the migrations."""
        def bad(name: str) -> _Reply:
            return _Reply(400, {"code": "23514", "message": f'new row for relation "{table}" violates check '
                                                            f'constraint "{name}"'})
        if table == "profiles":
            if len(str(row.get("display_name") or "")) > 80:
                raise bad("profiles_display_name_check")
            if len(str(row.get("affiliation") or "")) > 120:
                raise bad("profiles_affiliation_check")
            if "preferences" in row and not isinstance(row["preferences"], dict):
                raise bad("profiles_preferences_check")
            if row.get("avatar_path") and not str(row["avatar_path"]).startswith(f"{row.get('id')}/"):
                raise bad("profiles_avatar_path_check")
        elif table == "projects":
            if not _HEX64.match(str(row.get("content_hash") or "")):
                raise bad("projects_content_hash_check")
            if not 1 <= len(str(row.get("name") or "")) <= 200:
                raise bad("projects_name_check")
            if not str(row.get("object_path") or "").startswith(f"{row.get('owner')}/{row.get('id')}/"):
                raise bad("projects_object_path_check")
            parts = row.get("part_count", 1)
            if not isinstance(parts, int) or isinstance(parts, bool) or not 1 <= parts <= 1000:
                raise bad("projects_part_count_check")
        elif table == "shares":
            if not _TOKEN.match(str(row.get("token") or "")):
                raise bad("shares_token_check")

    def _rpc(self, h: _Handler, fn: str, hdr: dict, raw: bytes) -> None:
        args = self._json(raw)
        if fn == "aerolume_schema_version" and self.schema_version is not None:
            return h.reply(200, self.schema_version)
        if fn == "get_share":
            r = self.rows["shares"].get(str(args.get("share_token") or ""))
            now = _iso(time.time())
            if r is None or (r.get("expires_at") and str(r["expires_at"]) <= now):
                return h.reply(200, [])
            return h.reply(200, [{k: r.get(k) for k in ("token", "title", "files", "created_at", "expires_at")}])
        raise _Reply(404, {"code": "PGRST202", "message": f"Could not find the function public.{fn} in the "
                                                         "schema cache"})

    # -- Storage ---------------------------------------------------------------
    def _may(self, uid: str, bucket: str, path: str) -> bool:
        """The storage.objects policies of the migrations."""
        first = path.split("/", 1)[0]
        if bucket in ("projects", "avatars"):
            return first == uid
        if bucket == "shares":
            r = self.rows["shares"].get(first)
            return r is not None and str(r.get("owner")) == uid
        return False

    @staticmethod
    def _not_found() -> _Reply:
        return _Reply(400, {"statusCode": "404", "error": "not_found", "message": "Object not found"})

    def _storage(self, h: _Handler, method: str, path: str, q: dict, hdr: dict, raw: bytes) -> None:
        if method == "GET" and path.startswith("object/sign/"):
            rest = path[len("object/sign/"):]
            claims = self._verify_jwt(q.get("token", ""))
            if claims is None or claims.get("url") != rest:
                return h.reply(400, {"statusCode": "400", "error": "InvalidJWT", "message": "invalid signature"})
            obj = self.objects.get(tuple(rest.split("/", 1)))  # type: ignore[arg-type]
            if obj is None:
                raise self._not_found()
            return h.reply_bytes(200, obj[0], obj[1])
        claims = self._claims(hdr, required=False)
        if claims is None:
            raise _Reply(400, {"statusCode": "403", "error": "Unauthorized", "message": "Missing authorization"})
        uid = claims["sub"]
        if method == "POST" and path.startswith("object/sign/"):
            bucket, _, obj_path = path[len("object/sign/"):].partition("/")
            if not self._may(uid, bucket, obj_path) or (bucket, obj_path) not in self.objects:
                raise self._not_found()
            exp = int(time.time() + int(self._json(raw).get("expiresIn") or 60))
            tok = self._jwt({"url": f"{bucket}/{obj_path}", "iat": int(time.time()), "exp": exp})
            return h.reply(200, {"signedURL": f"/object/sign/{quote(bucket)}/{quote(obj_path)}?token={tok}"})
        if method == "POST" and path.startswith("object/list/"):
            bucket = path[len("object/list/"):]
            prefix = str(self._json(raw).get("prefix") or "")
            names = [p for (b, p) in self.objects if b == bucket and p.startswith(prefix) and self._may(uid, b, p)]
            return h.reply(200, [{"name": n[len(prefix):].lstrip("/"), "metadata": {"size": len(self.objects[(bucket, n)][0])}}
                                 for n in sorted(names)])
        if method == "DELETE" and path.startswith("object/") and path.count("/") == 1:
            bucket = path[len("object/"):]
            gone = []
            for p in self._json(raw).get("prefixes") or []:
                if (bucket, p) in self.objects and self._may(uid, bucket, p):
                    del self.objects[(bucket, p)]
                    gone.append({"name": p, "bucket_id": bucket})
            return h.reply(200, gone)
        if path.startswith("object/"):
            rest = path[len("object/"):]
            rest = rest.removeprefix("authenticated/")
            bucket, _, obj_path = rest.partition("/")
            if bucket not in self.BUCKETS:
                raise _Reply(400, {"statusCode": "404", "error": "Bucket not found", "message": "Bucket not found"})
            if method == "GET":
                if not self._may(uid, bucket, obj_path) or (bucket, obj_path) not in self.objects:
                    raise self._not_found()
                data, ctype = self.objects[(bucket, obj_path)]
                return h.reply_bytes(200, data, ctype)
            if method in ("POST", "PUT"):
                if not self._may(uid, bucket, obj_path):
                    raise _Reply(400, {"statusCode": "403", "error": "Unauthorized",
                                       "message": "new row violates row-level security policy"})
                ctype = hdr.get("content-type", "application/octet-stream").split(";")[0].strip()
                allowed = self.BUCKETS[bucket]
                if self.schema_version == 1:
                    allowed = tuple(t for t in allowed if t not in self.V2_TYPES.get(bucket, ()))
                if ctype not in allowed:
                    raise _Reply(400, {"statusCode": "415", "error": "invalid_mime_type",
                                       "message": f"mime type {ctype} is not supported"})
                if len(raw) > self.file_size_limit:
                    raise _Reply(413, {"statusCode": "413", "error": "Payload too large",
                                       "message": "The object exceeded the maximum allowed size"})
                upsert = hdr.get("x-upsert", "false").lower() == "true"
                if (bucket, obj_path) in self.objects and not upsert:
                    raise _Reply(400, {"statusCode": "409", "error": "Duplicate",
                                       "message": "The resource already exists"})
                self.objects[(bucket, obj_path)] = (raw, hdr.get("content-type", ctype))
                return h.reply(200, {"Key": f"{bucket}/{obj_path}", "Id": str(uuid.uuid4())})
        raise _Reply(404, {"statusCode": "404", "error": "not_found", "message": "route not found"})


class _Reply(Exception):
    def __init__(self, status: int, body: Any, headers: dict | None = None) -> None:
        super().__init__(status)
        self.status, self.body, self.headers = status, body, headers or {}


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockSupabase/1"
    protocol_version = "HTTP/1.1"

    def _go(self, method: str) -> None:
        self.server.mock.handle(self, method)  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self._go("GET")

    def do_POST(self) -> None:
        self._go("POST")

    def do_PUT(self) -> None:
        self._go("PUT")

    def do_PATCH(self) -> None:
        self._go("PATCH")

    def do_DELETE(self) -> None:
        self._go("DELETE")

    def reply(self, status: int, body: Any, headers: dict | None = None) -> None:
        data = b"" if body is None else json.dumps(body).encode("utf-8")
        self.send_response(status)
        if data:
            self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def reply_bytes(self, status: int, data: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        """Quiet: request lines carry codes and tokens."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aerolume.accounts.mock",
                                 description="Local stand-in for a Supabase project (memory only).")
    ap.add_argument("--port", type=int, default=54399)
    ap.add_argument("--autoconfirm", action="store_true", help="sign-up signs in at once (no email step)")
    ap.add_argument("--user", metavar="EMAIL:PASSWORD", help="create a confirmed user at start")
    args = ap.parse_args(argv)
    mock = MockSupabase(port=args.port, autoconfirm=args.autoconfirm, echo=True)
    if args.user and ":" in args.user:
        email, pw = args.user.split(":", 1)
        mock.add_user(email, pw)
    mock.start()
    print("Mock Supabase running (data is kept in memory only).", flush=True)
    print(f"  Project URL : {mock.url}", flush=True)
    print(f"  Anon key    : {mock.anon_key}", flush=True)
    print("Paste both into Aerolume's Account settings (Account button, top right). "
          "Emails are printed here. Ctrl+C stops.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        mock.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
