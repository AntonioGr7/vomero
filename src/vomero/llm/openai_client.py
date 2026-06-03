"""OpenAI-compatible client.

Works against api.openai.com or any OpenAI-compatible server (vLLM, LM Studio,
OpenRouter, Together, ...) by setting `base_url`. Translates our wire-neutral
Message/ToolSpec types to the chat-completions schema and back.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .base import LLMResponse, Message, ToolCall, ToolSpec, Usage


class OpenAIClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        force_single_tool: bool = False,
    ):
        self.model = model
        # `api_key` may be None for some local servers; the SDK tolerates a dummy.
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        # When exactly one tool is offered, force the model to call it instead of
        # `tool_choice="auto"`. Gemini's OpenAI-compat layer otherwise frequently
        # returns finish_reason=MALFORMED_FUNCTION_CALL (an empty reply with no
        # tool call) or invents a function name. Forcing the single tool fixes
        # both. Safe for the RLM loop because its real exit is `answer(...)` run
        # inside the REPL, not a plain-text reply. See GeminiClient.
        self.force_single_tool = force_single_tool

    # -- translation: ours -> OpenAI ------------------------------------
    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""}
                )
            elif m.role == "assistant" and m.tool_calls:
                out.append(
                    {
                        "role": "assistant",
                        "content": m.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                                # Echo provider passthrough (e.g. Gemini 3's
                                # required thought_signature) back verbatim.
                                **tc.extra,
                            }
                            for tc in m.tool_calls
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content or ""})
        return out

    @staticmethod
    def _to_openai_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    # -- the protocol method --------------------------------------------
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._to_openai_messages(messages),
        }
        oai_tools = self._to_openai_tools(tools)
        if oai_tools:
            kwargs["tools"] = oai_tools
            if self.force_single_tool and len(oai_tools) == 1:
                # Force this exact function rather than leaving it to the model.
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": oai_tools[0]["function"]["name"]},
                }
            else:
                kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            # Carry any non-standard fields (Gemini 3 puts its required
            # `thought_signature` under `extra_content`) so we can replay them.
            extra = {}
            extra_content = getattr(tc, "extra_content", None) or (
                tc.model_extra or {}
            ).get("extra_content")
            if extra_content:
                extra["extra_content"] = extra_content
            tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args, extra=extra)
            )

        usage = None
        if resp.usage is not None:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
            )

        return LLMResponse(content=msg.content, tool_calls=tool_calls, usage=usage)
