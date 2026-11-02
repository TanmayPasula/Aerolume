"""Supabase's three REST APIs, one thin method per endpoint Aerolume uses.

* **Auth** (GoTrue, ``/auth/v1``): sign-up, password / refresh / PKCE token
  grants, email OTP + magic link, password recovery, the OAuth authorize URL,
  user read / update, logout, public settings (enabled providers).
* **Database** (PostgREST, ``/rest/v1``): select / insert / upsert / update /
  delete on the ``profiles``, ``projects`` and ``shares`` tables and RPC calls.
  Row-level security in ``supabase/migrations`` decides what each request may
  touch; the client only ever sends the user's own JWT.
* **Storage** (``/storage/v1``): upload / download / remove / list objects and
  public URLs.

Headers: ``apikey`` (the anon or publishable key) on every request, plus
``Authorization: Bearer <user access token>`` when signed in. The key itself is
*not* sent as a bearer token: Supabase's newer publishable keys are not JWTs,
and an anonymous request does not need one.

This layer is stateless (no session, no refresh logic - that is
:class:`~aerolume.accounts.service.AccountService`). It turns every non-2xx
response into an exception with a human message; error *codes* from the three
servers (they use three different JSON shapes) are normalised so callers can
branch on ``exc.code``.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote, urlencode

from .config import AccountConfig
from .errors import AuthError, NotConfiguredError, RemoteError, TooLargeError
from .http import HttpClient, Response

OAUTH_PROVIDERS = ("github", "google")

# GoTrue error codes -> what the user should read. Unknown codes fall back to the
# server's own message, which GoTrue writes for humans.
_AUTH_MESSAGES = {
    "invalid_credentials": "Email or password is incorrect.",
    "invalid_grant": "Email or password is incorrect.",
    "email_not_confirmed": "Confirm your email address first: open the link in the email we sent.",
    "user_already_exists": "An account with this email already exists. Sign in instead.",
    "email_exists": "An account with this email already exists. Sign in instead.",
    "signup_disabled": "This project does not accept new accounts.",
    "email_provider_disabled": "Email sign-in is turned off for this project.",
    "provider_disabled": "This sign-in method is turned off for this project.",
    "over_email_send_rate_limit": "Too many emails were sent. Wait a minute, then try again.",
    "over_request_rate_limit": "Too many attempts. Wait a minute, then try again.",
    "otp_expired": "The code or link has expired or was already used. Request a new one.",
    "flow_state_expired": "The sign-in link has expired. Start again.",
    "flow_state_not_found": "The sign-in link was already used or has expired. Start again.",
    "bad_code_verifier": "The sign-in link belongs to another sign-in attempt. Start again.",
    "refresh_token_not_found": "Your session has ended. Sign in again.",
    "refresh_token_already_used": "Your session has ended. Sign in again.",
    "session_not_found": "Your session has ended. Sign in again.",
    "bad_jwt": "Your session has ended. Sign in again.",
    "same_password": "Choose a password different from the current one.",
    "user_not_found": "No account with this email.",
    "otp_disabled": "No account uses this email. Choose Create account to make one.",
}


def _error_fields(resp: Response) -> tuple[str, str]:
    """(code, message) from any of GoTrue's, PostgREST's or Storage's error bodies."""
    body = resp.json()
    if not isinstance(body, dict):
        return "", ""
    code = (body.get("error_code") or body.get("code") or body.get("error") or "")
    msg = (body.get("msg") or body.get("message") or body.get("error_description") or "")
    return str(code), str(msg)


class SupabaseBackend:
    """Endpoint wrappers bound to one project (URL + key)."""

    def __init__(self, config: AccountConfig, http: HttpClient | None = None) -> None:
        if not config.configured:
            raise NotConfiguredError()
        self.config = config
        self.http = http or HttpClient()
        self.base = config.url

    # -- plumbing -------------------------------------------------------------
    def _headers(self, token: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"apikey": self.config.anon_key}
        if token:
            h["Authorization"] = f"Bearer {token}"
        if extra:
            h.update(extra)
        return h

    def _auth_fail(self, resp: Response, what: str) -> AuthError:
        code, msg = _error_fields(resp)
        if resp.status == 401 and not code and "api key" in msg.lower():
            code = "invalid_api_key"
        if code == "invalid_api_key" or (resp.status == 401 and "apikey" in msg.lower()):
            return AuthError("The project rejected the anon key. Check it in Account settings.",
                             code="invalid_api_key", status=resp.status)
        text = _AUTH_MESSAGES.get(code) or msg or f"{what} failed (HTTP {resp.status})."
        if code == "weak_password" and msg:
            text = msg if msg.endswith(".") else msg + "."
        return AuthError(text, code=code or f"http_{resp.status}", status=resp.status)

    def _remote_fail(self, resp: Response, what: str) -> Exception:
        code, msg = _error_fields(resp)
        low = msg.lower()
        body = resp.json()
        # Storage answers most failures with HTTP 400 and the real status in "statusCode".
        status = resp.status
        if isinstance(body, dict) and str(body.get("statusCode") or "").isdigit():
            status = int(body["statusCode"])
            resp = Response(status, resp.headers, resp.body)
        if status == 409:
            return RemoteError(f"{what}: the file already exists.", code="exists", status=status)
        if status == 415 or code == "invalid_mime_type":
            return RemoteError(f"{what}: the project does not accept this file type.", code="mime", status=status)
        if resp.status == 413 or "payload too large" in low or "exceeded the maximum allowed size" in low:
            return TooLargeError(f"{what}: the file is larger than the project's upload limit.",
                                 code="too_large", status=resp.status)
        if resp.status in (401, 403) and ("jwt" in low or "claim timestamp" in low
                                           or code in ("PGRST301", "PGRST302", "PGRST303", "bad_jwt", "InvalidJWT")):
            return AuthError("Your session has ended. Sign in again.", code="bad_jwt", status=resp.status)
        if code in ("PGRST205", "42P01", "PGRST202") or "could not find the" in low:
            return RemoteError(f"{what}: the database is missing Aerolume's tables or functions. Apply "
                               "supabase/migrations to the project (see supabase/README.md).",
                               code="schema_missing", status=resp.status)
        if resp.status == 404:
            return RemoteError(f"{what}: not found.", code="not_found", status=resp.status)
        if resp.status in (401, 403) or code == "42501":
            return RemoteError(f"{what}: permission denied by the project's access rules.",
                               code="forbidden", status=resp.status)
        return RemoteError(f"{what} failed (HTTP {resp.status}{': ' + msg if msg else ''}).",
                           code=code or f"http_{resp.status}", status=resp.status)

    def _auth(self, method: str, path: str, *, token: str | None = None, body: Any = None,
              params: dict[str, str] | None = None, what: str = "Sign-in") -> Any:
        resp = self.http.request(method, f"{self.base}/auth/v1{path}", headers=self._headers(token),
                                 params=params, json_body=body)
        if not resp.ok:
            raise self._auth_fail(resp, what)
        return resp.json()

    # -- auth -------------------------------------------------------------------
    def auth_settings(self) -> dict:
        """Public auth settings: ``external`` (enabled providers), ``disable_signup``, ..."""
        return self._auth("GET", "/settings", what="Reading the project's sign-in settings") or {}

    def sign_up(self, email: str, password: str, *, data: dict | None = None, redirect_to: str = "",
                code_challenge: str = "") -> dict:
        body: dict[str, Any] = {"email": email, "password": password, "data": data or {}}
        if code_challenge:
            body.update(code_challenge=code_challenge, code_challenge_method="s256")
        params = {"redirect_to": redirect_to} if redirect_to else None
        return self._auth("POST", "/signup", body=body, params=params, what="Creating the account") or {}

    def token_password(self, email: str, password: str) -> dict:
        return self._auth("POST", "/token", params={"grant_type": "password"},
                          body={"email": email, "password": password})

    def token_refresh(self, refresh_token: str) -> dict:
        return self._auth("POST", "/token", params={"grant_type": "refresh_token"},
                          body={"refresh_token": refresh_token}, what="Renewing the session")

    def token_pkce(self, auth_code: str, code_verifier: str) -> dict:
        return self._auth("POST", "/token", params={"grant_type": "pkce"},
                          body={"auth_code": auth_code, "code_verifier": code_verifier})

    def send_otp(self, email: str, *, create_user: bool, redirect_to: str = "",
                 code_challenge: str = "") -> None:
        """Email a magic link (and the 6-digit code, if the email template includes it)."""
        body: dict[str, Any] = {"email": email, "create_user": create_user}
        if code_challenge:
            body.update(code_challenge=code_challenge, code_challenge_method="s256")
        params = {"redirect_to": redirect_to} if redirect_to else None
        self._auth("POST", "/otp", body=body, params=params, what="Sending the sign-in email")

    def verify_otp(self, email: str, token: str, kind: str = "email") -> dict:
        """Exchange the code from an email (``kind`` "email", "signup", "recovery") for a session."""
        return self._auth("POST", "/verify", body={"type": kind, "email": email, "token": token},
                          what="Checking the code")

    def recover(self, email: str, *, redirect_to: str = "", code_challenge: str = "") -> None:
        body: dict[str, Any] = {"email": email}
        if code_challenge:
            body.update(code_challenge=code_challenge, code_challenge_method="s256")
        params = {"redirect_to": redirect_to} if redirect_to else None
        self._auth("POST", "/recover", body=body, params=params, what="Sending the reset email")

    def authorize_url(self, provider: str, *, redirect_to: str, code_challenge: str,
                      scopes: str = "") -> str:
        """The URL to open in the system browser for an OAuth sign-in (PKCE, S256)."""
        if provider not in OAUTH_PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        q = {"provider": provider, "redirect_to": redirect_to, "code_challenge": code_challenge,
             "code_challenge_method": "s256"}
        if scopes:
            q["scopes"] = scopes
        return f"{self.base}/auth/v1/authorize?{urlencode(q)}"

    def get_user(self, token: str) -> dict:
        return self._auth("GET", "/user", token=token, what="Reading the account")

    def update_user(self, token: str, attrs: dict) -> dict:
        return self._auth("PUT", "/user", token=token, body=attrs, what="Updating the account")

    def logout(self, token: str, scope: str = "local") -> None:
        """Revoke the refresh token(s): ``local`` = this device, ``global`` = every device."""
        self._auth("POST", "/logout", token=token, params={"scope": scope}, what="Signing out")

    # -- database (PostgREST) -------------------------------------------------
    @staticmethod
    def _filters(filters: dict[str, str] | None) -> dict[str, str]:
        """``{"id": "abc"}`` -> ``{"id": "eq.abc"}``.

        Only equality is offered on purpose: every filter Aerolume needs is "this
        row" or "this owner", and a value is never interpreted as an operator, so
        a project name or id can never change what a query matches.
        """
        return {k: f"eq.{v}" for k, v in (filters or {}).items()}

    def _rest(self, method: str, table: str, token: str, *, params: dict | None = None, body: Any = None,
              prefer: str = "", what: str = "Database request") -> Any:
        extra = {"Prefer": prefer} if prefer else None
        resp = self.http.request(method, f"{self.base}/rest/v1/{quote(table)}", params=params,
                                 headers=self._headers(token, extra), json_body=body)
        if not resp.ok:
            raise self._remote_fail(resp, what)
        return resp.json()

    def select(self, table: str, token: str, filters: dict[str, str] | None = None, *, columns: str = "*",
               order: str = "", limit: int | None = None, what: str = "Reading") -> list[dict]:
        params = {"select": columns, **self._filters(filters)}
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)
        rows = self._rest("GET", table, token, params=params, what=what)
        return rows if isinstance(rows, list) else []

    def insert(self, table: str, token: str, rows: dict | list[dict], *, upsert: bool = False,
               on_conflict: str = "", what: str = "Saving") -> list[dict]:
        prefer = "return=representation"
        if upsert:
            prefer += ",resolution=merge-duplicates"
        params = {"on_conflict": on_conflict} if on_conflict else None
        out = self._rest("POST", table, token, params=params, body=rows, prefer=prefer, what=what)
        return out if isinstance(out, list) else []

    def update(self, table: str, token: str, values: dict, filters: dict[str, str], *,
               what: str = "Saving") -> list[dict]:
        """PATCH matching rows; returns the rows actually changed (empty = no match / no right)."""
        out = self._rest("PATCH", table, token, params=self._filters(filters), body=values,
                         prefer="return=representation", what=what)
        return out if isinstance(out, list) else []

    def delete(self, table: str, token: str, filters: dict[str, str], *, what: str = "Deleting") -> list[dict]:
        if not filters:
            raise ValueError("refusing an unfiltered delete")
        out = self._rest("DELETE", table, token, params=self._filters(filters),
                         prefer="return=representation", what=what)
        return out if isinstance(out, list) else []

    def rpc(self, fn: str, token: str | None, args: dict | None = None, *, what: str = "Database call") -> Any:
        resp = self.http.request("POST", f"{self.base}/rest/v1/rpc/{quote(fn)}",
                                 headers=self._headers(token), json_body=args or {})
        if not resp.ok:
            raise self._remote_fail(resp, what)
        return resp.json()

    # -- storage --------------------------------------------------------------
    def _object_url(self, bucket: str, path: str, kind: str = "") -> str:
        seg = f"/{kind}" if kind else ""
        return f"{self.base}/storage/v1/object{seg}/{quote(bucket)}/{quote(path)}"

    def upload(self, bucket: str, path: str, data: bytes, token: str, *, content_type: str,
               upsert: bool = False, what: str = "Uploading") -> None:
        headers = self._headers(token, {"Content-Type": content_type, "x-upsert": "true" if upsert else "false",
                                        "Cache-Control": "max-age=60"})
        resp = self.http.request("POST", self._object_url(bucket, path), headers=headers, data=data,
                                 timeout=max(60.0, len(data) / 250_000))   # >= 250 kB/s before giving up
        if not resp.ok:
            raise self._remote_fail(resp, what)

    def download(self, bucket: str, path: str, token: str, *, what: str = "Downloading") -> bytes:
        resp = self.http.request("GET", self._object_url(bucket, path, "authenticated"),
                                 headers=self._headers(token, {"Accept": "*/*"}), timeout=300.0)
        if not resp.ok:
            raise self._remote_fail(resp, what)
        return resp.body

    def remove(self, bucket: str, paths: list[str], token: str, *, what: str = "Deleting files") -> None:
        if not paths:
            return
        resp = self.http.request("DELETE", f"{self.base}/storage/v1/object/{quote(bucket)}",
                                 headers=self._headers(token), json_body={"prefixes": list(paths)})
        if not resp.ok:
            raise self._remote_fail(resp, what)

    def list_objects(self, bucket: str, prefix: str, token: str, *, limit: int = 1000) -> list[dict]:
        resp = self.http.request("POST", f"{self.base}/storage/v1/object/list/{quote(bucket)}",
                                 headers=self._headers(token),
                                 json_body={"prefix": prefix, "limit": limit, "offset": 0})
        if not resp.ok:
            raise self._remote_fail(resp, "Listing files")
        out = resp.json()
        return out if isinstance(out, list) else []

    def sign_url(self, bucket: str, path: str, token: str, expires_in: int) -> str:
        """A time-limited read URL for a private object (Storage enforces the expiry)."""
        resp = self.http.request("POST", f"{self.base}/storage/v1/object/sign/{quote(bucket)}/{quote(path)}",
                                 headers=self._headers(token), json_body={"expiresIn": int(expires_in)})
        if not resp.ok:
            raise self._remote_fail(resp, "Creating the link")
        body = resp.json() or {}
        signed = str(body.get("signedURL") or body.get("signedUrl") or "")
        if not signed:
            raise RemoteError("Creating the link: the server returned no URL.", code="no_url")
        if signed.startswith("http"):
            return signed
        return f"{self.base}/storage/v1{signed if signed.startswith('/') else '/' + signed}"

    def public_url(self, bucket: str, path: str) -> str:
        return self._object_url(bucket, path, "public")
