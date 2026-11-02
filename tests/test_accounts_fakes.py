"""Shared helpers for the account tests: the mock server, a service wired to it, a fake browser.

Every test talks to :class:`aerolume.accounts.mock.MockSupabase` on 127.0.0.1;
nothing here can reach a real Supabase project. The redirect receiver binds an
OS-chosen port (``redirect_ports=(0,)``) so tests never fight over 47821.
"""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

from aerolume.accounts import AccountService, SessionStore
from aerolume.accounts.cloud import SyncStore
from aerolume.accounts.config import AccountConfig
from aerolume.accounts.mock import MockSupabase

PASSWORD = "correct horse battery"


class FakeBrowser:
    """Opens URLs like a browser would: GET, following redirects, on a background thread."""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self.threads: list[threading.Thread] = []

    def __call__(self, url: str) -> bool:
        self.opened.append(url)
        t = threading.Thread(target=self._get, args=(url,), daemon=True)
        t.start()
        self.threads.append(t)
        return True

    @staticmethod
    def _get(url: str) -> None:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:   # follows the 303 to the loopback
                r.read()
        except Exception:  # noqa: BLE001, S110 - an error page is still "the browser showed something"
            pass

    def join(self) -> None:
        for t in self.threads:
            t.join(10)


def click(link: str) -> None:
    """The user opens an email link in their browser on this computer."""
    FakeBrowser._get(link)


def make_service(mock: MockSupabase, folder: Path, *, browser: FakeBrowser | None = None,
                 max_upload_mb: int = 50, max_project_mb: int = 1024, key: str | None = None) -> AccountService:
    cfg = AccountConfig(mock.url, key if key is not None else mock.anon_key, "file",
                        max_upload_mb=max_upload_mb, redirect_ports=(0,), max_project_mb=max_project_mb)
    folder.mkdir(parents=True, exist_ok=True)
    return AccountService(cfg, store=SessionStore(folder / "session"), sync_store=SyncStore(folder / "sync.json"),
                          open_browser=browser or FakeBrowser())


def signed_in(mock: MockSupabase, folder: Path, email: str = "ada@example.org", **kw) -> AccountService:
    """A service already signed in as a confirmed user (created on the mock if needed)."""
    try:
        mock.user_id(email)
    except AssertionError:
        mock.add_user(email, PASSWORD, display_name=email.split("@")[0].title())
    svc = make_service(mock, folder, **kw)
    svc.sign_in(email, PASSWORD)
    return svc
