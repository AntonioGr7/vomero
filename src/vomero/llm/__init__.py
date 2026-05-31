"""Provider-agnostic LLM layer.

`base.py` defines the wire-neutral types and the `LLMClient` protocol. The
engine talks only to those. Each concrete provider (OpenAI today; Anthropic /
Gemini later) translates to and from its own schema behind that protocol.
"""

from __future__ import annotations

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolSpec, Usage

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_client",
]


def build_client(settings) -> LLMClient:
    """Factory: map `settings.provider` to a concrete client."""
    provider = settings.provider.lower()
    if provider == "openai":
        from .openai_client import OpenAIClient

        return OpenAIClient(
            model=settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
    raise ValueError(
        f"Unknown provider {provider!r}. Implemented: 'openai'. "
        "Add a client in vomero/llm/ that satisfies the LLMClient protocol."
    )
