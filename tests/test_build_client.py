"""Provider factory wiring (no network — just construction)."""

from dataclasses import dataclass

from vomero.llm import build_client
from vomero.llm.gemini_client import DEFAULT_MODEL, GEMINI_OPENAI_BASE, GeminiClient
from vomero.llm.openai_client import OpenAIClient


@dataclass
class S:
    provider: str
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = "test-key"


def test_openai_provider():
    c = build_client(S(provider="openai"))
    assert isinstance(c, OpenAIClient)


def test_gemini_provider_uses_compat_base():
    c = build_client(S(provider="gemini", model="gemini-2.5-flash"))
    assert isinstance(c, GeminiClient)
    # Points at Gemini's OpenAI-compatible endpoint.
    assert str(c._client.base_url).rstrip("/") == GEMINI_OPENAI_BASE.rstrip("/")
    assert c.model == "gemini-2.5-flash"


def test_gemini_falls_back_when_model_looks_like_openai():
    # Default config model is an OpenAI one; gemini must substitute a real model.
    c = build_client(S(provider="gemini", model="gpt-4o-mini"))
    assert c.model == DEFAULT_MODEL


def test_unknown_provider_errors():
    try:
        build_client(S(provider="bananas"))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "gemini" in str(e)  # error lists implemented providers
