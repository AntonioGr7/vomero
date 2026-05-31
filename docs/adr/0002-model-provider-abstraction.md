# ADR 0002 — Provider abstraction: OpenAI-compatible now, Anthropic/Gemini later

Status: accepted (v0)
Date: 2026-05-31

## Context

We want to start on an OpenAI-compatible backend (works with OpenAI, vLLM,
LM Studio, OpenRouter, local servers) but keep the door open for Anthropic and
Gemini without rewriting the engine.

## Decision

Define a tiny, wire-neutral surface in `vomero/llm/base.py`:

- Types: `Message`, `ToolCall`, `ToolSpec`, `LLMResponse`.
- Protocol: `LLMClient.complete(messages, *, tools, model, temperature)`.

The engine constructs and reads only those types. Each provider implements the
protocol and translates to/from its own schema. `build_client(settings)` in
`vomero/llm/__init__.py` maps `settings.provider` to a concrete client.

v0 implements `OpenAIClient` only.

## Adding a provider later

1. Add `vomero/llm/anthropic_client.py` with a class satisfying `LLMClient`.
2. Translate `Message`/`ToolSpec` <-> that provider's schema. Notes:
   - Anthropic: system prompt is a top-level `system` param (not a message);
     tool calls are `tool_use` blocks; tool results are `tool_result` blocks in
     a `user` turn. Map `tool_call_id` -> `tool_use_id`.
   - Gemini: roles are `user`/`model`; tools are `functionDeclarations`; tool
     results are `functionResponse` parts.
3. Register it in `build_client`.

No engine changes required.

## Consequences

- Our types are a least-common-denominator (system/user/assistant/tool +
  function-style tools). Provider-specific features (extended thinking, citation
  blocks, structured outputs) aren't modeled yet; add them as optional fields on
  the neutral types only when a real need appears, keeping the engine agnostic.
