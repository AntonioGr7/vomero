"""Google Gemini, via its OpenAI-compatible endpoint.

Google exposes an OpenAI-compatible API at `…/v1beta/openai/`, so we reuse the
OpenAI wire translation wholesale (messages, the single `python` tool, and token
usage all map straight through) — no extra dependency, no bespoke schema code.

Set a Gemini API key (`VOMERO_API_KEY`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY`)
and a Gemini model (`VOMERO_MODEL=gemini-2.5-flash`). Point `VOMERO_BASE_URL` at
a different endpoint only if you need to override the default.

If you later need Gemini-specific features (thinking budgets, safety settings,
inline files) that the compatibility layer doesn't expose, drop in a native
`google-genai` client that satisfies the same `LLMClient` protocol — the engine
won't change.
"""

from __future__ import annotations

from .openai_client import OpenAIClient

# Gemini's OpenAI-compatibility base URL.
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Used when the configured model isn't a Gemini model (e.g. the openai default).
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient(OpenAIClient):
    """Gemini behind the OpenAI SDK. Same protocol as `OpenAIClient`."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        super().__init__(
            model=model,
            base_url=base_url or GEMINI_OPENAI_BASE,
            api_key=api_key,
        )
