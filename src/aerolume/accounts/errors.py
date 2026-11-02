"""Exceptions raised by :mod:`aerolume.accounts`.

Every message is written for the person at the keyboard: it says what
happened and what to do next, in one or two plain sentences, and it never
contains a token, a password or an API key (the HTTP layer builds messages from
status codes and server error codes only). The GUI shows ``str(exc)`` as is.
"""
from __future__ import annotations


class AccountError(Exception):
    """Base class: something an account operation could not do.

    ``code`` is a short machine-readable reason ("invalid_credentials",
    "network", "conflict", ...) so callers and tests can branch without parsing
    the human message.
    """

    code = "account_error"

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code
        self.status = status


class NotConfiguredError(AccountError):
    """No Supabase project is connected (the default: the app is offline-only)."""

    code = "not_configured"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Accounts are not set up. Connect a Supabase project in "
                                    "Account settings (Account button, top right; see docs/accounts.md).")


class ConfigError(AccountError):
    """The connection settings are invalid (bad URL, plain http to a remote host, ...)."""

    code = "config"


class NetworkError(AccountError):
    """The server could not be reached (offline, DNS, TLS, timeout)."""

    code = "network"


class AuthError(AccountError):
    """The auth server refused the request (wrong password, unconfirmed email, ...)."""

    code = "auth"


class NotSignedInError(AuthError):
    """The operation needs a signed-in user and there is none (or the session expired)."""

    code = "not_signed_in"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Sign in first.")


class RemoteError(AccountError):
    """The database or storage API returned an error that is not an auth problem."""

    code = "remote"


class SyncConflictError(AccountError):
    """Both the local folder and the cloud copy changed since the last sync.

    Nothing was overwritten. ``local_hash`` / ``remote_hash`` identify the two
    versions so the caller can offer "upload mine" or "download theirs".
    """

    code = "conflict"

    def __init__(self, message: str, *, local_hash: str = "", remote_hash: str = "",
                 project_id: str = "") -> None:
        super().__init__(message)
        self.local_hash = local_hash
        self.remote_hash = remote_hash
        self.project_id = project_id


class TooLargeError(AccountError):
    """A file is over the upload size limit of the connected project."""

    code = "too_large"


class CancelledError(AccountError):
    """The user cancelled a browser or email sign-in that was waiting."""

    code = "cancelled"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Sign-in cancelled.")


class SignInTimeoutError(AccountError):
    """A browser or email sign-in did not come back in time."""

    code = "timeout"
