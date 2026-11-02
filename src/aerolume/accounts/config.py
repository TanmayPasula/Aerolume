"""Which Supabase project (if any) the app talks to.

Nothing is configured by default: with no URL the whole app works offline and
:class:`~aerolume.accounts.service.AccountService` refuses every network call
with :class:`~aerolume.accounts.errors.NotConfiguredError`. A project is
connected in one of two ways, the first one found wins:

1. the environment variables ``AEROLUME_SUPABASE_URL`` and
   ``AEROLUME_SUPABASE_ANON_KEY`` (CI, portable installs, a lab-wide setup);
2. ``account.json`` in the user config folder
   (:func:`aerolume.project.recent.config_dir`, honours ``AEROLUME_CONFIG_DIR``),
   written by Account settings (also the Account tab of Preferences) through
   :func:`save_config`.

The anon (or "publishable") key is a *public* client key by Supabase's design:
it identifies the project, and row-level security on the server decides what a
request may touch. It is still masked in the UI and never logged, because a
leaked key invites abuse of the project's rate limits.

URLs must be ``https://`` - the session tokens travel in headers - except for
loopback hosts (``127.0.0.1``, ``localhost``, ``::1``), which is what the local
Supabase CLI stack and the in-process mock server use.

No Qt and no network here: the CLI, scripts and tests read the same settings.
"""
from __future__ import annotations

import contextlib
import ipaddress
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ..project.recent import config_dir
from .errors import ConfigError

ENV_URL = "AEROLUME_SUPABASE_URL"
ENV_KEY = "AEROLUME_SUPABASE_ANON_KEY"
CONFIG_FILE = "account.json"

#: Supabase's default per-file upload limit on the free plan (50 MB). A project
#: on a paid plan can raise it in the dashboard and here (``max_upload_mb``).
#: It caps each *object*, not a project: larger project archives go up in parts
#: of :attr:`AccountConfig.part_bytes` (aerolume.accounts.cloud.part_paths).
DEFAULT_MAX_UPLOAD_MB = 50

#: Largest project archive the app uploads, in MB (``max_project_mb``). The free
#: plan stores 1 GB in total, and an upload briefly holds the old and the new
#: version, so 1024 MB is the most a free project can take; a paid plan raises it.
DEFAULT_MAX_PROJECT_MB = 1024

#: Each part is this fraction of the per-file limit: the slack keeps a part under
#: the limit even where a server counts MB as 10^6 bytes instead of 2^20.
PART_FRACTION = 0.9

#: Loopback ports tried, in order, for the OAuth / email-link redirect. Fixed
#: ports (rather than "any free port") let a project allow-list exact redirect
#: URLs; supabase/README.md lists them, plus the glob alternative.
DEFAULT_REDIRECT_PORTS = (47821, 47822, 47823, 47824)

#: hosted Supabase projects live at <project-ref>.supabase.co
SUPABASE_SUFFIX = ".supabase.co"


@dataclass(frozen=True)
class AccountConfig:
    """Connection settings. ``source`` is "env", "file" or "none"."""

    url: str = ""
    anon_key: str = ""
    source: str = "none"
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    redirect_ports: tuple[int, ...] = field(default=DEFAULT_REDIRECT_PORTS)
    max_project_mb: int = DEFAULT_MAX_PROJECT_MB

    @property
    def configured(self) -> bool:
        return bool(self.url and self.anon_key)

    @property
    def part_bytes(self) -> int:
        """Size of one uploaded part: 90 % of the per-file limit (45 MiB on the free plan)."""
        return max(64 * 1024, int(self.max_upload_mb * 1024 * 1024 * PART_FRACTION))

    @property
    def host(self) -> str:
        """The server's host name, for status lines ("Connected to abc.supabase.co")."""
        return (urlsplit(self.url).hostname or "") if self.url else ""

    @property
    def is_loopback(self) -> bool:
        return bool(self.url) and is_loopback_host(urlsplit(self.url).hostname or "")

    @property
    def project_ref(self) -> str:
        """The Supabase project reference ("abcd" in abcd.supabase.co); "" for any other host."""
        host = self.host.lower()
        return host[: -len(SUPABASE_SUFFIX)] if host.endswith(SUPABASE_SUFFIX) else ""

    @property
    def server_label(self) -> str:
        """One line saying which server accounts use, for menus and dialog footers.

        A bare host name ("Project: 127.0.0.1") reads as a project called
        127.0.0.1, so say what kind of server it is: "Supabase project abcd",
        "Server: 127.0.0.1 (local test server)" or "Server: auth.example.org".
        """
        if not self.url:
            return ""
        if self.project_ref:
            return f"Supabase project {self.project_ref}"
        if self.is_loopback:
            return f"Server: {self.host} (local test server)"
        return f"Server: {self.host}"

    @property
    def server_name(self) -> str:
        """The same server as a noun phrase for sentences ("copies on <server_name>")."""
        if not self.url:
            return ""
        if self.project_ref:
            return f"Supabase project {self.project_ref}"
        if self.is_loopback:
            return f"the local test server ({self.host})"
        return self.host

    def __repr__(self) -> str:      # the key never appears in a traceback or a log line
        return (f"AccountConfig(url={self.url!r}, anon_key={mask_key(self.anon_key)!r}, "
                f"source={self.source!r})")


def is_loopback_host(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalise_url(url: str) -> str:
    """Validate a project URL and return it without a trailing slash.

    Raises :class:`ConfigError` for anything that would send tokens in clear
    text to another machine, or that is not an http(s) URL at all.
    """
    url = (url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ConfigError("The project URL must look like https://<project-ref>.supabase.co")
    if parts.scheme == "http" and not is_loopback_host(parts.hostname):
        raise ConfigError("Use an https:// URL. Plain http is only allowed for a local server "
                          "(127.0.0.1 or localhost), because sign-in tokens travel with every request.")
    if parts.query or parts.fragment:
        raise ConfigError("The project URL must not contain ? or # parts.")
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def mask_key(key: str) -> str:
    """``eyJhbG…x1Yz`` style masking for display; never the whole key."""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 12:
        return "•" * len(key)
    return f"{key[:6]}…{key[-4:]}"


def config_path() -> Path:
    return config_dir() / CONFIG_FILE


def _read_file() -> dict:
    try:
        raw = json.loads(config_path().read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _options(raw: dict) -> dict:
    """The optional knobs of ``account.json``, validated (a bad value falls back to the default)."""
    out: dict = {}
    mb = raw.get("max_upload_mb")
    if isinstance(mb, int | float) and 1 <= mb <= 50_000:
        out["max_upload_mb"] = int(mb)
    pmb = raw.get("max_project_mb")
    if isinstance(pmb, int | float) and 1 <= pmb <= 5_000_000:
        out["max_project_mb"] = int(pmb)
    ports = raw.get("redirect_ports")
    if isinstance(ports, list) and ports and all(isinstance(p, int) and 1024 <= p <= 65535 for p in ports):
        out["redirect_ports"] = tuple(ports)
    return out


def load_config() -> AccountConfig:
    """The active configuration (environment first, then ``account.json``, else none).

    An invalid URL in either place yields an unconfigured result rather than an
    exception, so a typo can never stop the app from starting; Preferences >
    Account shows the problem via :func:`config_problem`.
    """
    raw = _read_file()
    opts = _options(raw)
    env_url, env_key = os.environ.get(ENV_URL, "").strip(), os.environ.get(ENV_KEY, "").strip()
    if env_url or env_key:
        try:
            return AccountConfig(normalise_url(env_url), env_key, "env", **opts)
        except ConfigError:
            return AccountConfig(source="none", **opts)
    url, key = str(raw.get("url") or ""), str(raw.get("anon_key") or "")
    if url and key:
        try:
            return AccountConfig(normalise_url(url), key.strip(), "file", **opts)
        except ConfigError:
            pass
    return AccountConfig(source="none", **opts)


def config_problem() -> str:
    """Why the saved / environment settings are not usable, or "" when they are (or are absent)."""
    env_url, env_key = os.environ.get(ENV_URL, "").strip(), os.environ.get(ENV_KEY, "").strip()
    if env_url or env_key:
        if not (env_url and env_key):
            return f"Set both {ENV_URL} and {ENV_KEY}, or neither."
        try:
            normalise_url(env_url)
        except ConfigError as exc:
            return f"{ENV_URL}: {exc}"
        return ""
    raw = _read_file()
    url, key = str(raw.get("url") or ""), str(raw.get("anon_key") or "")
    if url or key:
        if not (url and key):
            return "Both the project URL and the anon key are needed."
        try:
            normalise_url(url)
        except ConfigError as exc:
            return str(exc)
    return ""


def env_override() -> bool:
    """True when the environment variables decide the configuration (Preferences is then read-only)."""
    return bool(os.environ.get(ENV_URL, "").strip() or os.environ.get(ENV_KEY, "").strip())


def save_config(url: str, anon_key: str) -> AccountConfig:
    """Write ``account.json`` (both empty = disconnect: the file's connection keys are removed).

    Validates first, so a bad URL raises :class:`ConfigError` and the old file
    stays as it was. Other keys in the file (``max_upload_mb``, ...) survive.
    """
    url = normalise_url(url)
    anon_key = (anon_key or "").strip()
    if bool(url) != bool(anon_key):
        raise ConfigError("Enter both the project URL and the anon key, or clear both.")
    raw = _read_file()
    raw.pop("url", None)
    raw.pop("anon_key", None)
    if url:
        raw["url"], raw["anon_key"] = url, anon_key
    path = config_path()
    if raw:
        from ..project.fileio import write_text
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, json.dumps(raw, indent=2) + "\n")
        if os.name != "nt":
            with contextlib.suppress(OSError):
                os.chmod(path, 0o600)
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    return load_config()
