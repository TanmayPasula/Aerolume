"""A deliberately small HTTP client on :mod:`urllib.request` (no new dependency).

Why not ``requests`` / ``httpx``: Aerolume ships as a frozen desktop app and
the account feature is optional, so it must not add a package the solver
users never need. Supabase's APIs are plain JSON over HTTPS, which the
standard library handles fine.

Two choices matter for safety:

* **Redirects are not followed.** urllib would happily re-send a request to
  wherever a 3xx points; with an ``Authorization`` header that is a token leak
  waiting to happen. Supabase's REST endpoints never redirect, so a 3xx comes
  back to the caller as a normal :class:`Response`.
* **Error messages never contain headers or bodies we sent.** A
  :class:`~aerolume.accounts.errors.NetworkError` names the host and the
  socket-level reason only, so a traceback pasted into a bug report cannot
  carry a session token.
"""
from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlsplit

from .errors import NetworkError

DEFAULT_TIMEOUT = 30.0


def _user_agent() -> str:
    try:
        from .. import __version__
    except Exception:  # noqa: BLE001 - a version string is cosmetic
        __version__ = "0"
    return f"Aerolume/{__version__} (accounts; urllib)"


@dataclass
class Response:
    """Status, lower-cased headers and the raw body of one HTTP exchange."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        """The body parsed as JSON (``None`` for an empty body; ``{}`` for a non-JSON one)."""
        if not self.body:
            return None
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpClient:
    """``request()`` returns a :class:`Response` for every HTTP status; raises only on transport failure."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout
        self._ua = _user_agent()
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method: str, url: str, *, headers: dict[str, str] | None = None,
                params: dict[str, str] | None = None, json_body: Any = None, data: bytes | None = None,
                timeout: float | None = None) -> Response:
        if params:
            url = f"{url}{'&' if '?' in url else '?'}{urlencode(params)}"
        hdrs = {"User-Agent": self._ua, "Accept": "application/json"}
        hdrs.update(headers or {})
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=hdrs)
        host = urlsplit(url).hostname or "the server"
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                return Response(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read())
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read()
            except OSError:
                body = b""
            hdr = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
            return Response(exc.code, hdr, body)
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, ssl.SSLError):
                raise NetworkError(f"Secure connection to {host} failed ({reason.reason or 'TLS error'}).") from None
            if isinstance(reason, socket.gaierror):
                raise NetworkError(f"Cannot find {host}. Check the project URL and your internet "
                                   "connection.") from None
            if isinstance(reason, ConnectionRefusedError):
                raise NetworkError(f"{host} refused the connection. Is the server running?") from None
            if isinstance(reason, TimeoutError | socket.timeout):
                raise NetworkError(f"{host} did not answer in time.") from None
            raise NetworkError(f"Cannot reach {host} ({type(reason).__name__}).") from None
        except TimeoutError:
            raise NetworkError(f"{host} did not answer in time.") from None
        except (ConnectionError, OSError) as exc:
            raise NetworkError(f"Connection to {host} failed ({type(exc).__name__}).") from None
