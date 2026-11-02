"""Accounts: configuration, every sign-in flow, refresh, sign-out, session storage, redaction.

All traffic goes to the in-process mock Supabase (aerolume.accounts.mock) on
127.0.0.1; the mock enforces the same rules as the real services (PKCE
verifier check, refresh-token rotation, JWT expiry, apikey) so these tests
fail for the reasons production would.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import socket
import sys
import threading
import urllib.parse
import urllib.request

import pytest
from test_accounts_fakes import PASSWORD, FakeBrowser, click, make_service, signed_in

from aerolume.accounts import (
    AccountService,
    AuthError,
    CancelledError,
    ConfigError,
    NetworkError,
    NotConfiguredError,
    NotSignedInError,
    SessionStore,
    SignInTimeoutError,
    service,
)
from aerolume.accounts import config as C
from aerolume.accounts import session as S
from aerolume.accounts.config import AccountConfig
from aerolume.accounts.loopback import LoopbackReceiver, challenge_for, new_verifier
from aerolume.accounts.mock import MockSupabase


@pytest.fixture()
def mock():
    with MockSupabase() as m:
        yield m


@pytest.fixture()
def cfgdir(tmp_path, monkeypatch):
    d = tmp_path / "config"
    monkeypatch.setenv("AEROLUME_CONFIG_DIR", str(d))
    monkeypatch.delenv(C.ENV_URL, raising=False)
    monkeypatch.delenv(C.ENV_KEY, raising=False)
    return d


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_unconfigured_by_default_and_never_opens_a_socket(cfgdir, monkeypatch):
    cfg = C.load_config()
    assert not cfg.configured and cfg.source == "none"

    def no_network(*a, **k):
        raise AssertionError("an unconfigured service must not touch the network")
    monkeypatch.setattr("aerolume.accounts.http.HttpClient.request", no_network)
    svc = AccountService(cfg)
    assert not svc.configured and not svc.signed_in and svc.user is None
    for call in (lambda: svc.sign_in("a@b.org", "x" * 8), lambda: svc.test_connection(),
                 lambda: svc.begin_oauth("github", open_browser=False), lambda: svc.auth_providers()):
        with pytest.raises(NotConfiguredError) as ei:
            call()
        assert "Account settings" in str(ei.value)
    svc.sign_out()      # signed out already: a no-op, not an error


def test_server_label_says_what_kind_of_server():
    """Menus and footers name the server, never a bare host that reads like a project name."""
    hosted = C.AccountConfig("https://abcd1234.supabase.co", "k", "file")
    assert (hosted.project_ref, hosted.server_label, hosted.server_name) == (
        "abcd1234", "Supabase project abcd1234", "Supabase project abcd1234")
    local = C.AccountConfig("http://127.0.0.1:54321", "k", "env")
    assert local.project_ref == "" and local.server_label == "Server: 127.0.0.1 (local test server)"
    assert local.server_name == "the local test server (127.0.0.1)"
    assert C.AccountConfig("http://localhost:54321", "k").server_label == "Server: localhost (local test server)"
    own = C.AccountConfig("https://auth.example.org", "k")
    assert (own.server_label, own.server_name) == ("Server: auth.example.org", "auth.example.org")
    assert C.AccountConfig().server_label == "" and C.AccountConfig().server_name == ""


def test_config_sources_validation_and_masking(cfgdir, monkeypatch):
    saved = C.save_config("https://abcd.supabase.co/", "eyJhbGciOiJIUzI1NiJ9.payload.signature")
    assert saved.configured and saved.source == "file" and saved.url == "https://abcd.supabase.co"
    assert saved.host == "abcd.supabase.co"
    assert "payload" not in repr(saved) and "eyJhbG" in repr(saved)     # masked, recognisable
    assert C.mask_key("short") == "•" * 5
    # the environment wins over the file
    monkeypatch.setenv(C.ENV_URL, "http://127.0.0.1:54321")
    monkeypatch.setenv(C.ENV_KEY, "local-key")
    cfg = C.load_config()
    assert (cfg.url, cfg.source, cfg.is_loopback) == ("http://127.0.0.1:54321", "env", True)
    assert C.env_override()
    monkeypatch.setenv(C.ENV_KEY, "")
    assert "Set both" in C.config_problem()
    monkeypatch.delenv(C.ENV_URL)
    monkeypatch.delenv(C.ENV_KEY)
    # plain http to another machine would send tokens in clear text: refused
    for bad in ("http://abcd.supabase.co", "ftp://x.org", "abcd", "https://x.org/?a=1"):
        with pytest.raises(ConfigError):
            C.normalise_url(bad)
    with pytest.raises(ConfigError):
        C.save_config("https://abcd.supabase.co", "")
    assert C.load_config().url == "https://abcd.supabase.co"           # a failed save changes nothing
    # a hand-edited bad file leaves the app offline, with a reason for Preferences
    C.config_path().write_text('{"url": "http://evil.example", "anon_key": "k"}', encoding="utf-8")
    assert not C.load_config().configured
    assert "https" in C.config_problem()
    C.config_path().write_text('{"url": "https://a.supabase.co", "anon_key": "k", "max_upload_mb": 500, '
                               '"redirect_ports": [50000, 50001], "max_project_mb": 4096}', encoding="utf-8")
    cfg = C.load_config()
    assert cfg.max_upload_mb == 500 and cfg.redirect_ports == (50000, 50001) and cfg.max_project_mb == 4096
    assert cfg.part_bytes == int(500 * 1024 * 1024 * 0.9)            # parts stay under the file limit
    assert C.AccountConfig().max_project_mb == 1024 and C.AccountConfig().part_bytes == (45 * 1024 * 1024)
    C.save_config("", "")                                               # disconnect keeps the options
    assert not C.load_config().configured and C.load_config().max_upload_mb == 500


def test_test_connection_reports_key_schema_and_providers(mock, tmp_path):
    svc = make_service(mock, tmp_path)
    rep = svc.test_connection()
    assert rep.reachable and rep.key_accepted and rep.schema_version == service.SCHEMA_VERSION == 2
    assert rep.providers == ("github", "google") and rep.signup_enabled
    mock.providers = {"github"}
    assert svc.auth_providers() == ("github",)
    bad = make_service(mock, tmp_path / "bad", key="wrong-key")
    rep = bad.test_connection()
    assert not rep.key_accepted and "anon key" in rep.message
    with pytest.raises(AuthError) as ei:
        bad.sign_in("a@example.org", PASSWORD)
    assert ei.value.code == "invalid_api_key"
    mock.schema_version = 1                         # a project set up for an earlier version
    rep = svc.test_connection()
    assert rep.key_accepted and rep.schema_version == 1 and "supabase/migrations" in rep.message
    mock.schema_version = None
    rep = svc.test_connection()
    assert rep.key_accepted and rep.schema_version is None and "supabase/migrations" in rep.message


def test_unreachable_server_is_a_network_error(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    svc = AccountService(AccountConfig(f"http://127.0.0.1:{port}", "k", "file"),
                         store=SessionStore(tmp_path / "s"))
    with pytest.raises(NetworkError) as ei:
        svc.sign_in("a@example.org", PASSWORD)
    assert "127.0.0.1" in str(ei.value)


# ---------------------------------------------------------------------------
# email + password, email links and codes
# ---------------------------------------------------------------------------

def test_sign_up_confirmed_through_the_email_link(mock, tmp_path):
    svc = make_service(mock, tmp_path)
    events = []
    svc.subscribe(events.append)
    result, pending = svc.sign_up("Ada@Example.org", PASSWORD, "Ada Lovelace")
    assert result.needs_confirmation and result.user is None and not svc.signed_in
    mail = mock.last_email("ada@example.org")
    assert mail.kind == "signup"
    # the link carries the loopback redirect; opening it on this computer completes sign-in
    assert urllib.parse.quote(pending.redirect_uri, safe="") in mail.link
    click(mail.link)
    user = pending.wait(10)
    assert svc.signed_in and user.email == "ada@example.org" and user.initials == "AL"
    assert events == ["signed_in"]
    assert svc.get_profile().display_name == "Ada Lovelace"     # created by the sign-up trigger


def test_sign_up_with_the_code_and_wrong_code(mock, tmp_path):
    svc = make_service(mock, tmp_path)
    _, pending = svc.sign_up("grace@example.org", PASSWORD)
    with pytest.raises(AuthError) as ei:
        pending.complete_with_code("000000" if mock.last_email().code != "000000" else "111111")
    assert ei.value.code == "otp_expired" and "Request a new one" in str(ei.value)
    user = pending.complete_with_code(" " + mock.last_email().code[:3] + " " + mock.last_email().code[3:])
    assert user.email == "grace@example.org" and svc.signed_in


def test_sign_up_autoconfirm_existing_and_weak(mock, tmp_path):
    mock.autoconfirm = True
    svc = make_service(mock, tmp_path)
    result, pending = svc.sign_up("lin@example.org", PASSWORD, "Lin")
    assert result.user is not None and not result.needs_confirmation and pending is None and svc.signed_in
    svc.sign_out()
    # an existing confirmed address looks like "check your email" (no account probing), no session
    result, pending = svc.sign_up("lin@example.org", PASSWORD)
    assert result.needs_confirmation and not svc.signed_in
    pending.close()
    with pytest.raises(AuthError, match="at least 8"):
        svc.sign_up("x@example.org", "short")
    with pytest.raises(AuthError, match="valid email"):
        svc.sign_up("not-an-email", PASSWORD)


def test_password_sign_in_errors_are_plain_language(mock, tmp_path):
    mock.add_user("ada@example.org", PASSWORD)
    mock.add_user("new@example.org", PASSWORD, confirmed=False)
    svc = make_service(mock, tmp_path)
    with pytest.raises(AuthError) as ei:
        svc.sign_in("ada@example.org", "wrong password")
    assert str(ei.value) == "Email or password is incorrect." and ei.value.code == "invalid_credentials"
    with pytest.raises(AuthError) as ei:
        svc.sign_in("new@example.org", PASSWORD)
    assert ei.value.code == "email_not_confirmed" and "Confirm your email" in str(ei.value)
    user = svc.sign_in(" ada@example.org ", PASSWORD)
    assert user.email == "ada@example.org" and user.provider == "email"


def test_magic_link_and_code(mock, tmp_path):
    svc = make_service(mock, tmp_path)
    pending = svc.begin_email_link("kat@example.org")
    mail = mock.last_email("kat@example.org")
    assert mail.kind == "magiclink" and len(mail.code) == 6
    click(mail.link)
    assert pending.wait(10).email == "kat@example.org"
    svc.sign_out()
    pending = svc.begin_email_link("kat@example.org")
    assert pending.complete_with_code(mock.last_email().code).email == "kat@example.org"
    # a link used once is dead: the browser lands on the error page, the app reports it
    svc.sign_out()
    pending = svc.begin_email_link("kat@example.org", create_user=False)
    link = mock.last_email().link
    click(link)
    pending.wait(10)
    svc.sign_out()
    pending2 = svc.begin_email_link("kat@example.org")
    used = link.replace(urllib.parse.quote(pending.redirect_uri, safe=""),
                        urllib.parse.quote(pending2.redirect_uri, safe=""))
    click(used)
    with pytest.raises(AuthError, match="expired"):
        pending2.wait(10)


def test_password_reset_by_code_then_new_password(mock, tmp_path):
    mock.add_user("ada@example.org", PASSWORD)
    svc = make_service(mock, tmp_path)
    pending = svc.begin_password_reset("ada@example.org")
    assert mock.last_email("ada@example.org").kind == "recovery"
    pending.complete_with_code(mock.last_email().code)
    assert svc.signed_in and svc.recovery_pending
    with pytest.raises(AuthError, match="at least 8"):
        svc.set_password("short")
    svc.set_password("a brand new password")
    assert not svc.recovery_pending
    svc.sign_out()
    with pytest.raises(AuthError):
        svc.sign_in("ada@example.org", PASSWORD)
    assert svc.sign_in("ada@example.org", "a brand new password").email == "ada@example.org"
    # unknown addresses get the same silent success (no account probing)
    svc.sign_out()
    svc.begin_password_reset("nobody@example.org").close()


def test_reset_link_opened_in_browser(mock, tmp_path):
    mock.add_user("ada@example.org", PASSWORD)
    svc = make_service(mock, tmp_path)
    pending = svc.begin_password_reset("ada@example.org")
    click(mock.last_email().link)
    pending.wait(10)
    assert svc.recovery_pending


# ---------------------------------------------------------------------------
# OAuth (system browser, PKCE, loopback)
# ---------------------------------------------------------------------------

def test_oauth_github_with_pkce(mock, tmp_path):
    browser = FakeBrowser()
    svc = make_service(mock, tmp_path, browser=browser)
    pending = svc.begin_oauth("github")
    user = pending.wait(10)
    assert user.provider == "github" and user.email == "octo@example.org" and user.display_name == "Github User"
    url = urllib.parse.urlsplit(browser.opened[0])
    q = dict(urllib.parse.parse_qsl(url.query))
    assert url.path == "/auth/v1/authorize" and q["provider"] == "github"
    assert q["code_challenge_method"] == "s256" and len(q["code_challenge"]) == 43
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/callback/[A-Za-z0-9_-]{22}", q["redirect_to"])
    # The verifier never travelled: the only request carrying it is the token exchange
    sent = [r for r in mock.requests if r[1] == "/auth/v1/token"]
    assert len(sent) == 1
    browser.join()


def test_pkce_rejects_a_code_used_with_another_verifier(mock, tmp_path):
    svc = make_service(mock, tmp_path)
    verifier = new_verifier()
    assert 43 <= len(verifier) <= 128
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge_for(verifier) == expected
    url = svc._backend().authorize_url("google", redirect_to="http://127.0.0.1:9/callback",
                                       code_challenge=challenge_for(verifier))
    opener = urllib.request.build_opener(_StopRedirect())
    with pytest.raises(urllib.error.HTTPError) as ei:
        opener.open(url)
    code = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(ei.value.headers["Location"]).query))["code"]
    with pytest.raises(AuthError) as ae:
        svc._backend().token_pkce(code, new_verifier())         # an attacker's verifier
    assert ae.value.code == "bad_code_verifier"


class _StopRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def test_oauth_disabled_provider_and_unknown_provider(mock, tmp_path):
    mock.providers = set()
    svc = make_service(mock, tmp_path, browser=FakeBrowser())
    with pytest.raises(AuthError):
        svc.begin_oauth("facebook")
    pending = svc.begin_oauth("github")      # the browser shows the provider error; the app times out
    with pytest.raises(SignInTimeoutError):
        pending.wait(0.5)


def test_loopback_receiver_errors_cancel_timeout_and_ports():
    with LoopbackReceiver() as rx:
        assert rx.redirect_uri.startswith("http://127.0.0.1:")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(rx.redirect_uri.replace("/callback", "/other"))
        assert ei.value.code == 404
        q = urllib.parse.urlencode({"error": "access_denied", "error_description": "User <b>denied</b>"})
        page = urllib.request.urlopen(f"{rx.redirect_uri}?{q}").read().decode()
        assert "&lt;b&gt;denied&lt;/b&gt;" in page and "<b>" not in page     # escaped
        assert rx.wait(1)["error"] == "access_denied"
    rx = LoopbackReceiver()
    threading.Timer(0.1, rx.cancel).start()
    with pytest.raises(CancelledError):
        rx.wait(5)
    rx.close()
    with LoopbackReceiver() as rx, pytest.raises(SignInTimeoutError):
        rx.wait(0.2)


def test_loopback_ignores_callbacks_without_this_attempts_nonce():
    """Another web page can reach the predictable port (an <img> to 127.0.0.1:47821/callback?error=x):
    without this attempt's nonce the request is a 404 and never ends the wait (critic round 5)."""
    with LoopbackReceiver() as rx:
        base = f"http://127.0.0.1:{rx.port}/callback"
        assert rx.redirect_uri.startswith(base + "/") and rx.redirect_uri != base
        for url in (f"{base}?error=access_denied&error_description=blocked", f"{base}?code=bogus",
                    f"{base}/not-the-nonce?error=x", f"{rx.redirect_uri}x?code=bogus"):
            with pytest.raises(urllib.error.HTTPError) as ei:
                urllib.request.urlopen(url)
            assert ei.value.code == 404
        with pytest.raises(SignInTimeoutError):
            rx.wait(0.2)                                 # nothing was delivered
        page = urllib.request.urlopen(f"{rx.redirect_uri}?code=real").read().decode()
        assert "finish signing in" in page and "Signed in" not in page   # the code is not exchanged yet
        assert rx.wait(1) == {"code": "real"}
    with LoopbackReceiver() as a, LoopbackReceiver() as b:
        assert a.callback_path != b.callback_path       # a fresh nonce per attempt
    # a taken port is skipped, and the listener is on loopback only
    with LoopbackReceiver() as first, LoopbackReceiver((first.port, 0)) as second:
        assert second.port != first.port
        assert second._server.server_address[0] == "127.0.0.1"


def test_oauth_error_reply_raises(mock, tmp_path):
    def deny(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        threading.Thread(target=click, args=(q["redirect_to"] + "?error=access_denied&error_description="
                                             "The+user+denied+access",), daemon=True).start()
    svc = make_service(mock, tmp_path, browser=deny)
    with pytest.raises(AuthError, match="denied"):
        svc.begin_oauth("github").wait(10)
    assert not svc.signed_in


# ---------------------------------------------------------------------------
# refresh, retries, sign-out
# ---------------------------------------------------------------------------

def test_refresh_rotates_and_persists(mock, tmp_path):
    mock.access_ttl = 30            # inside the 60 s leeway: every access_token() call refreshes
    svc = signed_in(mock, tmp_path)
    s0 = svc.session
    tok = svc.access_token()
    assert tok != s0.access_token and svc.session.refresh_token != s0.refresh_token
    # rotation: the old refresh token is dead on the server
    with pytest.raises(AuthError) as ei:
        svc._backend().token_refresh(s0.refresh_token)
    assert ei.value.code == "refresh_token_already_used"
    # the rotated token was saved: a new process picks it up
    again = make_service(mock, tmp_path)
    assert again.session.refresh_token == svc.session.refresh_token


def test_concurrent_callers_refresh_once(mock, tmp_path):
    mock.access_ttl = 3600
    svc = signed_in(mock, tmp_path)
    svc.session.expires_at = 0          # looks expired to every thread at once
    before = sum(1 for r in mock.requests if r[1] == "/auth/v1/token")
    tokens, errors = [], []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            tokens.append(svc.access_token())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    after = sum(1 for r in mock.requests if r[1] == "/auth/v1/token")
    assert not errors and len(set(tokens)) == 1 and after - before == 1 and svc.signed_in


def test_revoked_access_token_is_retried_after_refresh(mock, tmp_path):
    svc = signed_in(mock, tmp_path)
    mock.invalidate_access_tokens()             # the server no longer accepts the current JWT
    assert svc.get_profile().id == svc.user.id  # 403 bad_jwt -> refresh -> retry succeeds


def test_refresh_refused_signs_out(mock, tmp_path):
    svc = signed_in(mock, tmp_path)
    other = make_service(mock, tmp_path / "other")
    other.sign_in("ada@example.org", PASSWORD)
    other.sign_out(everywhere=True)             # revokes every session of this user
    events = []
    svc.subscribe(events.append)
    with pytest.raises(NotSignedInError, match="Sign in again"):
        svc.refresh()
    assert not svc.signed_in and events == ["signed_out"]
    assert not (tmp_path / "session").exists()


def test_sign_out_revokes_and_works_offline(mock, tmp_path):
    svc = signed_in(mock, tmp_path)
    rt = svc.session.refresh_token
    svc.sign_out()
    assert not svc.signed_in and not (tmp_path / "session").exists()
    with pytest.raises(AuthError):
        svc._backend().token_refresh(rt)
    gone = MockSupabase().start()
    svc = signed_in(gone, tmp_path / "b")
    gone.stop()                                 # server gone: local sign-out still happens
    svc.sign_out()
    assert not svc.signed_in and not (tmp_path / "b" / "session").exists()


# ---------------------------------------------------------------------------
# session storage and redaction
# ---------------------------------------------------------------------------

def test_session_store_is_bound_to_the_url_and_protected(mock, tmp_path):
    svc = signed_in(mock, tmp_path)
    raw = (tmp_path / "session").read_bytes()
    s = svc.session
    if sys.platform == "win32":
        assert raw.startswith(b"AL1D")                    # DPAPI
        assert s.refresh_token.encode() not in raw and s.access_token.encode() not in raw
    else:
        assert raw.startswith(b"AL1P")
        assert oct(os.stat(tmp_path / "session").st_mode & 0o777) == "0o600"
    store = SessionStore(tmp_path / "session")
    assert store.load(mock.url).refresh_token == s.refresh_token
    assert store.load("https://other.supabase.co") is None       # never reused for another project
    (tmp_path / "session").write_bytes(b"garbage")
    assert store.load(mock.url) is None                          # unreadable = signed out, no crash


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI is Windows-only")
def test_dpapi_round_trip():
    blob = S.protect(b"secret payload \x00\xff")
    assert blob.startswith(b"AL1D") and b"secret payload" not in blob
    assert S.unprotect(blob) == b"secret payload \x00\xff"


def test_no_secret_reaches_logs_reprs_or_errors(mock, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    mock.access_ttl = 30
    svc = make_service(mock, tmp_path, browser=FakeBrowser())
    _, pending = svc.sign_up("ada@example.org", PASSWORD, "Ada")
    code = mock.last_email().code
    pending.complete_with_code(code)
    svc.access_token()
    svc.begin_oauth("github").wait(10)
    s = svc.session
    secrets_ = [PASSWORD, code, s.access_token, s.refresh_token, mock.anon_key]
    text = "\n".join(r.getMessage() for r in caplog.records) + repr(s) + str(s) + repr(svc.config)
    try:
        svc.sign_in("ada@example.org", "wrong " + PASSWORD)
    except AuthError as exc:
        text += str(exc) + repr(exc)
    for secret in secrets_:
        assert secret not in text
