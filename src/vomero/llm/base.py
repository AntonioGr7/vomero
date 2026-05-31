"""Wire-neutral message/tool types and the LLMClient protocol.

These types are deliberately a tiny subset of what every major provider
supports (system/user/assistant/tool roles + function-style tool calls). The
engine only ever constructs and reads these; providers translate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One conversation turn.

    role:
      - "system" / "user": plain `content`.
      - "assistant": `content` and/or `tool_calls`.
      - "tool": result of a tool call; set `content` and `tool_call_id`.
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class ToolSpec:
    """A tool advertised to the model (JSON-Schema parameters)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Usage:
    """Token counts a provider reports for one `complete` call.

    `prompt_tokens` is the size of the context we sent; `completion_tokens` is
    what the model generated. Providers that don't report usage (some local
    servers, the test fake) leave this off and the meter estimates instead.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """A single assistant reply."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


@runtime_checkable
class LLMClient(Protocol):
    """The only surface the engine depends on. Implement this per provider."""

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        ...
