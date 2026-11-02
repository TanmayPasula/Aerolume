"""Model providers for the assistant: Claude (Anthropic API) and local models (Ollama).

:func:`make_provider` builds the provider the preferences select. Neither module
imports anything heavy at import time: ``anthropic`` is imported when a Claude
client is first needed, Ollama is spoken with ``urllib``.
"""
from __future__ import annotations

from .base import Cancelled, ChatSession, Provider, ProviderError, ToolCall, ToolResult, ToolSpec, Turn


def make_provider(cfg=None):
    """The provider of an :class:`~aerolume.ai.config.AssistantConfig` (default: saved preferences)."""
    from ..config import load_config, stored_key
    cfg = (cfg or load_config()).validated()
    if cfg.effective_provider() == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(cfg.claude_model, effort=cfg.claude_effort, max_tokens=cfg.claude_max_tokens,
                              api_key=stored_key() or None)
    from .ollama import OllamaProvider
    return OllamaProvider(cfg.ollama_model, cfg.ollama_endpoint, think=cfg.ollama_think, num_ctx=cfg.ollama_num_ctx)


__all__ = ["Cancelled", "ChatSession", "Provider", "ProviderError", "ToolCall", "ToolResult", "ToolSpec", "Turn",
           "make_provider"]
