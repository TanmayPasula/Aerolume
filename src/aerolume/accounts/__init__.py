"""Optional online accounts for Aerolume (DESIGN.md section 11).

Aerolume works fully offline with no account; that is the default and nothing
here runs unless a Supabase project is connected (Account settings, or
``AEROLUME_SUPABASE_URL`` + ``AEROLUME_SUPABASE_ANON_KEY``). When one is:

* sign up / sign in with email + password, an email link or code, or GitHub /
  Google through the system browser (PKCE, 127.0.0.1 loopback redirect);
* a profile (display name, affiliation, avatar, preferences);
* cloud copies of project folders with content-hash sync status and conflict
  detection;
* read-only, expiring share links for a result's report.

Only the standard library is used (``urllib``, ``http.server``, ``ctypes`` for
Windows DPAPI). ``supabase/migrations`` holds the tables, row-level security and
storage policies; ``supabase/README.md`` says how to create and link a project;
``python -m aerolume.accounts.mock`` runs a local stand-in server for trying the
feature without any cloud project. The user guide is ``docs/accounts.md``.

Typical use::

    from aerolume.accounts import get_service
    svc = get_service()
    if svc.configured and not svc.signed_in:
        svc.sign_in("ada@example.org", "correct horse battery staple")
    svc.upload_project(project.path)
"""
from __future__ import annotations

from .cloud import STATUS_LABELS, classify, content_hash
from .config import AccountConfig, load_config, save_config
from .errors import (
    AccountError,
    AuthError,
    CancelledError,
    ConfigError,
    NetworkError,
    NotConfiguredError,
    NotSignedInError,
    RemoteError,
    SignInTimeoutError,
    SyncConflictError,
    TooLargeError,
)
from .service import (
    AccountService,
    AccountUser,
    CloudProject,
    ConnectionReport,
    PendingSignIn,
    Profile,
    ProjectStatus,
    ShareLink,
    SignUpResult,
    UploadPreview,
    get_service,
    set_service,
)
from .session import Session, SessionStore

__all__ = [
    "STATUS_LABELS", "AccountConfig", "AccountError", "AccountService", "AccountUser", "AuthError",
    "CancelledError", "CloudProject", "ConfigError", "ConnectionReport", "NetworkError", "NotConfiguredError",
    "NotSignedInError", "PendingSignIn", "Profile", "ProjectStatus", "RemoteError", "Session",
    "SessionStore", "ShareLink", "SignInTimeoutError", "SignUpResult", "SyncConflictError", "TooLargeError",
    "UploadPreview", "classify", "content_hash", "get_service", "load_config", "save_config", "set_service",
]
