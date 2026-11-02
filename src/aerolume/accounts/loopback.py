"""PKCE and the 127.0.0.1 redirect receiver for browser sign-in (RFC 7636, RFC 8252).

A desktop app cannot keep an OAuth client secret, so it signs in the way RFC
8252 ("OAuth 2.0 for Native Apps") prescribes: open the system browser at the
authorize URL with a PKCE *code challenge*, receive the one-time code on a
loopback redirect (``http://127.0.0.1:<port>/callback/<nonce>``), and exchange it
together with the *code verifier* that never left this process. An attacker
who intercepts the code cannot use it without the verifier, which is why no
``state`` round trip is needed for safety (Supabase's PKCE flow binds the code
to the challenge server-side).

The same receiver serves email links (magic link, sign-up confirmation,
password reset): Supabase redirects the clicked link to the loopback URL with a
``code`` when the request carried a challenge. The link only completes
automatically when it is opened on this computer while Aerolume waits; the
6-digit code in the same email is the fallback (see the service).

The receiver listens on 127.0.0.1 only (never 0.0.0.0), answers exactly one
callback, and serves a plain page telling the user to return to the app. It
shuts down after the first callback, a timeout, or a cancel.

**Why a nonce in the path.** The redirect ports are fixed (a Supabase project
must allow-list them) and so predictable. Without a per-attempt secret, any web
page open in the user's browser during a sign-in could load
``<img src="http://127.0.0.1:47821/callback?error=x">`` and end the wait with
its own error, or with a bogus code that then fails the exchange: PKCE stops an
account takeover, but sign-in could be reliably blocked (critic round 5). Each
receiver therefore waits on ``/callback/<nonce>``, 16 random bytes that only the
authorize / email request carries; a request to any other path gets a 404 and
is never delivered. The success page only says to return to the app: the code
has not been exchanged yet, so it cannot say "signed in".
"""
from __future__ import annotations

import base64
import hashlib
import html
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self
from urllib.parse import parse_qs, urlsplit

from .errors import CancelledError, SignInTimeoutError

CALLBACK_PATH = "/callback"


def new_verifier() -> str:
    """A 64-character code verifier (RFC 7636 allows 43-128 unreserved characters)."""
    return secrets.token_urlsafe(48)[:64]


def challenge_for(verifier: str) -> str:
    """The S256 code challenge: base64url(SHA-256(verifier)) without padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Aerolume</title>
<style>body{{font:16px/1.5 "Segoe UI",system-ui,sans-serif;margin:15vh auto;max-width:34em;
padding:0 16px;color:#1d2129;background:#fff}}h1{{font-size:20px;font-weight:600;margin:0 0 8px}}
p{{margin:0;color:#454c59}}@media (prefers-color-scheme:dark){{body{{background:#1e1f22;color:#dfe1e5}}
p{{color:#b9bdc5}}}}</style></head><body><h1>{title}</h1><p>{text}</p></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "AerolumeSignIn/1"

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        receiver: LoopbackReceiver = self.server.receiver  # type: ignore[attr-defined]
        # bytes: compare_digest refuses non-ASCII str, and a request line may carry any byte
        if not secrets.compare_digest(parts.path.encode("utf-8", "surrogateescape"),
                                      receiver.callback_path.encode("ascii")):
            self.send_error(404)            # not this attempt's address: never delivered
            return
        params = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        if "error" in params or "error_description" in params:
            title = "Sign-in did not complete"
            text = html.escape(params.get("error_description") or params.get("error") or "Unknown error")
            text += ". Return to Aerolume to try again."
        elif "code" in params:
            title, text = "Return to Aerolume", "Return to Aerolume to finish signing in. You can close this tab."
        else:
            title = "Nothing to do here"
            text = "This address only receives sign-in replies for Aerolume."
        body = _PAGE.format(title=title, text=text).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)
        if "code" in params or "error" in params or "error_description" in params:
            receiver._deliver(params)

    def log_message(self, format: str, *args) -> None:
        """Silent: the request line carries the one-time code, which must not reach a log."""


class _Server(ThreadingHTTPServer):
    """No address reuse: on Windows ``SO_REUSEADDR`` lets a second socket bind a port that
    is already in use, so another program's listener could silently share the redirect
    port (and "port taken" would never be detected). ``SO_EXCLUSIVEADDRUSE`` forbids it."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class LoopbackReceiver:
    """Listen on the first free port of ``ports`` (0 = any) and wait for one callback.

    Use as a context manager; :attr:`redirect_uri` is what goes into the
    authorize / email request. :meth:`wait` returns the callback's query
    parameters (``code`` or ``error`` / ``error_description``).
    """

    def __init__(self, ports: tuple[int, ...] = (0,)) -> None:
        self._nonce = secrets.token_urlsafe(16)      # [A-Za-z0-9_-]: matches a "callback/*" allow-list glob
        self._result: dict[str, str] | None = None
        self._event = threading.Event()
        self._cancelled = False
        self._server: ThreadingHTTPServer | None = None
        last: OSError | None = None
        for port in ports or (0,):
            try:
                self._server = _Server(("127.0.0.1", port), _Handler)
                break
            except OSError as exc:      # port taken: try the next one
                last = exc
        if self._server is None:
            raise OSError(f"no free loopback port among {ports}") from last
        self._server.receiver = self  # type: ignore[attr-defined]
        self._port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="aerolume-signin",
                                        daemon=True, kwargs={"poll_interval": 0.1})
        self._thread.start()

    @property
    def port(self) -> int:
        """The bound port (still known after :meth:`close`, for messages and tests)."""
        return self._port

    @property
    def callback_path(self) -> str:
        """This attempt's callback path, ``/callback/<nonce>`` (see the module docstring)."""
        return f"{CALLBACK_PATH}/{self._nonce}"

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.callback_path}"

    def _deliver(self, params: dict[str, str]) -> None:
        if self._result is None:
            self._result = params
            self._event.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._event.set()

    def wait(self, timeout: float) -> dict[str, str]:
        """Block until the callback arrives; raises on timeout or :meth:`cancel`."""
        self._event.wait(timeout)
        if self._result is not None:
            return self._result
        if self._cancelled:
            raise CancelledError()
        raise SignInTimeoutError("No reply from the browser in time. Start the sign-in again.")

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
