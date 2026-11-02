"""Assistant preferences and the Claude API key, kept in the per-user config folder.

Two files in :func:`aerolume.project.recent.config_dir` (``%APPDATA%\\Aerolume`` on
Windows, ``~/.config/aerolume`` elsewhere, or ``AEROLUME_CONFIG_DIR``):

* ``assistant.json`` - provider, model, endpoint, approval threshold and the
  other non-secret choices of Preferences > Assistant. Plain JSON so the CLI,
  scripts and the GUI read the same settings without Qt.
* ``assistant_key`` - the Anthropic API key, **only** when the user typed one into
  Preferences. It is a separate file so the settings INI and ``assistant.json``,
  which people attach to bug reports, never contain it; on POSIX it is created
  with mode 0600. The key is never logged, never written into a project or an
  experiment log, and the GUI shows it masked (:func:`mask_key`).

When no key is stored, Claude uses the SDK's normal credential chain
(``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login``
profile), which is the recommended setup (docs/ai_assistant.md).

Nothing here imports Qt or the ``anthropic`` package.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..project.recent import config_dir

PROVIDERS = ("auto", "claude", "ollama")
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
SETTINGS_FILE = "assistant.json"
#: the local model's default context window [tokens]. 8192 filled up in a one-run experiment (7,309 of 8,192
#: tokens after the plan, critic round 3.2: tool schemas, project overview, a run and two corrections); qwen3
#: takes 32k. On the RTX 4060 Ti 8 GB, 16384 puts qwen3:8b Q4_K_M at 7.8 GB, 80 % of it on the GPU (Ollama
#: 0.40, measured; 8192: 6.2 GB, all on the GPU). Generation is slower once the window fills (measured 14.7
#: tokens/s after a 12k-token prompt at 16384, against 42.7 after a 4k one at 8192), but the experiment keeps
#: its early messages -- the plan and the results the answer is checked against -- instead of dropping them.
DEFAULT_OLLAMA_NUM_CTX = 16384
#: the default an older assistant.json saved as an explicit value (raised in defaults version 2)
_OLD_NUM_CTX = 8192
#: version of the defaults written into assistant.json: lets a later default replace an old *default* that was
#: saved as a value, without overriding what a user chose
DEFAULTS_VERSION = 2
KEY_FILE = "assistant_key"


@dataclass
class AssistantConfig:
    """What Preferences > Assistant sets (all fields have safe defaults).

    ``provider``: "auto" (Claude when credentials exist, otherwise the local
    model), "claude" or "ollama".
    ``approval_threshold_s``: runs whose *estimated* time exceeds this many
    seconds ask the user first (deleting and running code always ask).
    ``ollama_think``: let a local reasoning model (qwen3) think before it
    answers - better plans, but several times slower on a laptop GPU.
    ``ollama_num_ctx``: the local model's context window in tokens (default
    :data:`DEFAULT_OLLAMA_NUM_CTX`, 16384: an experiment with one run filled 8192).
    ``defaults_version``: :data:`DEFAULTS_VERSION` of the file; older files had the
    then-default 8192 written out, which :func:`load_config` raises to the new default.
    ``consent``: where the user allowed project data to be sent without asking each time (Lumen's
    rule: nothing goes to a model off this computer without consent).
    ``claude_max_tokens``: the output budget of one Claude turn. Thinking is
    always on for Claude Opus 5.5 and counts toward it, so it is sized for the
    thinking plus the reply (64k, the claude-api skill's starting point for long
    agentic turns); it is only an upper bound, not what a turn costs.
    """
    provider: str = "auto"
    claude_model: str = DEFAULT_CLAUDE_MODEL
    claude_effort: str = "high"
    claude_max_tokens: int = 64000
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    ollama_think: bool = False
    ollama_num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    approval_threshold_s: float = 120.0
    max_steps: int = 40
    #: destinations the user allowed to receive project data without asking again ("claude",
    #: "ollama:http://host:11434"); a model on this computer never needs it (agent.data_destination)
    consent: list[str] = field(default_factory=list)
    defaults_version: int = DEFAULTS_VERSION

    def effective_provider(self) -> str:
        """"claude" or "ollama": "auto" picks Claude when credentials exist, else the local model.

        A fresh install with an ``ANTHROPIC_API_KEY`` (or an ``ant auth login``
        profile) gets Claude; one without gets the fully local, open-source path.
        """
        if self.provider in ("claude", "ollama"):
            return self.provider
        return "claude" if claude_credential_source() else "ollama"

    @property
    def model(self) -> str:
        """The model of the effective provider."""
        return self.claude_model if self.effective_provider() == "claude" else self.ollama_model

    def validated(self) -> AssistantConfig:
        """A copy with every field forced into its allowed range (hand-edited files)."""
        c = AssistantConfig(**asdict(self))
        if c.provider not in PROVIDERS:
            c.provider = "auto"
        if c.claude_effort not in EFFORTS:
            c.claude_effort = "high"
        c.claude_model = str(c.claude_model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
        c.ollama_model = str(c.ollama_model or DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
        c.ollama_endpoint = (str(c.ollama_endpoint or "").strip().rstrip("/") or DEFAULT_OLLAMA_ENDPOINT)
        try:
            c.claude_max_tokens = int(min(max(int(c.claude_max_tokens), 4096), 128000))
        except (TypeError, ValueError):
            c.claude_max_tokens = 64000
        try:
            c.ollama_num_ctx = int(min(max(int(c.ollama_num_ctx), 4096), 131072))
        except (TypeError, ValueError):
            c.ollama_num_ctx = DEFAULT_OLLAMA_NUM_CTX
        try:
            c.approval_threshold_s = float(min(max(float(c.approval_threshold_s), 0.0), 86400.0))
        except (TypeError, ValueError):
            c.approval_threshold_s = 120.0
        try:
            c.max_steps = int(min(max(int(c.max_steps), 4), 200))
        except (TypeError, ValueError):
            c.max_steps = 40
        c.ollama_think = bool(c.ollama_think)
        raw = c.consent if isinstance(c.consent, (list, tuple)) else []
        c.consent = list(dict.fromkeys(str(x) for x in raw if isinstance(x, str) and x.strip()))
        return c


def settings_path() -> Path:
    return config_dir() / SETTINGS_FILE


def key_path() -> Path:
    return config_dir() / KEY_FILE


def load_config() -> AssistantConfig:
    """The saved assistant preferences, or the defaults (a damaged file is ignored)."""
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AssistantConfig()
    if not isinstance(raw, dict):
        return AssistantConfig()
    names = {f.name for f in fields(AssistantConfig)}
    vals = {k: v for k, v in raw.items() if k in names}
    try:
        version = int(raw.get("defaults_version", 1))
    except (TypeError, ValueError):
        version = 1
    if version < 2 and vals.get("ollama_num_ctx") == _OLD_NUM_CTX:
        # every save wrote all fields, so 8192 here is the old default, not a choice: take the new one
        vals.pop("ollama_num_ctx")
    vals["defaults_version"] = DEFAULTS_VERSION
    try:
        return AssistantConfig(**vals).validated()
    except TypeError:
        return AssistantConfig()


def save_config(cfg: AssistantConfig) -> Path:
    """Write the preferences (atomically) and return the file."""
    from ..project.fileio import write_json
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json(path, asdict(cfg.validated()))


# ---------------------------------------------------------------------------
# the Claude API key typed into Preferences
# ---------------------------------------------------------------------------

def stored_key() -> str:
    """The API key saved from Preferences, or "" (the SDK's own chain is then used)."""
    try:
        return key_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def store_key(key: str) -> None:
    """Save (or, with an empty string, delete) the key typed into Preferences.

    The file is written with owner-only permissions where the platform has
    them; on Windows the per-user profile folder already restricts access.
    """
    key = (key or "").strip()
    path = key_path()
    if not key:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)
        return
    path.write_text(key, encoding="utf-8")


def mask_key(key: str) -> str:
    """"sk-ant-…a1b2" style masking for display (never the whole key)."""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 12:
        return "•" * len(key)
    return f"{key[:7]}…{key[-4:]}"


def claude_credential_source() -> str:
    """Where Claude credentials would come from: "preferences", "environment", "profile" or "".

    Never reads or returns the secret itself beyond checking that it exists.
    "profile" is reported when the SDK can build a client from an ``ant auth
    login`` profile; that check imports ``anthropic`` (optional dependency).
    """
    if stored_key():
        return "preferences"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "environment"
    try:
        import anthropic
    except ImportError:
        return ""
    try:
        client = anthropic.Anthropic()
    except Exception:  # noqa: BLE001 - a broken profile is "no credentials", not a crash
        return ""
    if getattr(client, "api_key", None) or getattr(client, "auth_token", None):
        return "environment"
    if getattr(client, "_token_cache", None) is not None:
        return "profile"
    return ""


def anthropic_available() -> bool:
    """Whether the optional ``anthropic`` package (``pip install aerolume[ai]``) is installed."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def as_dict(cfg: AssistantConfig) -> dict[str, Any]:
    return asdict(cfg)


__all__ = [
    "DEFAULTS_VERSION", "DEFAULT_CLAUDE_MODEL", "DEFAULT_OLLAMA_ENDPOINT", "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_NUM_CTX", "EFFORTS", "PROVIDERS",
    "AssistantConfig", "anthropic_available", "as_dict", "claude_credential_source", "key_path",
    "load_config", "mask_key", "save_config", "settings_path", "store_key", "stored_key",
]
