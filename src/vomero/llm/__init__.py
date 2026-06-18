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
    "build_embedder",
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
    if provider == "gemini":
        from .gemini_client import DEFAULT_MODEL, GeminiClient

        # If the model still looks like an OpenAI one (e.g. the default), fall
        # back to a Gemini model so the request doesn't bounce.
        model = settings.model
        if not model or model.lower().startswith(("gpt", "o1", "o3", "o4")):
            model = DEFAULT_MODEL
        return GeminiClient(
            model=model,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
    raise ValueError(
        f"Unknown provider {provider!r}. Implemented: 'openai', 'gemini'. "
        "Add a client in vomero/llm/ that satisfies the LLMClient protocol."
    )


def build_embedder(settings):
    """Factory: an `Embedder` for dense search when `embedding_model` is set,
    else None (search stays pure-BM25). Reuses the main credentials unless the
    embedding-specific overrides are given."""
    if not getattr(settings, "embedding_model", ""):
        return None
    from .embeddings import OpenAIEmbedder

    return OpenAIEmbedder(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url or settings.base_url,
        api_key=settings.embedding_api_key or settings.api_key,
    )
