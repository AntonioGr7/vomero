"""Token accounting for the RLM loop.

Two numbers matter, and they are NOT the same:

* **context size** — how many tokens the *current* message stack renders to.
  This is the live gauge of how "blown" the context is. It climbs as the loop
  appends tool output and *drops* when history is compacted. The authoritative
  value is the prompt-token count the provider reports for the most recent
  call (exactly the size of what we just sent); we fall back to a char-based
  estimate when a provider omits usage.

* **cumulative tokens** — every prompt + completion token spent since the run
  started, summed across the root loop and every recursive ``llm()`` / ``rlm()``
  sub-call. Only ever increases; this is the cost / throughput figure.

A single `UsageMeter` is threaded through an entire `run` tree (the recursion
shares it), so the cumulative figure spans root and sub-calls. The per-loop
context size is computed fresh at each step from that loop's own messages.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm.base import Message, Usage


def estimate_tokens(text: str | None) -> int:
    """Rough, dependency-free token estimate (~4 chars per token).

    Used only when a provider does not report token usage. Good enough to keep
    the gauge meaningful; the OpenAI path uses exact provider counts instead.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(messages: list[Message]) -> int:
    """Estimate the rendered size of a message stack.

    Counts content plus serialized tool-call arguments, with a small per-message
    overhead standing in for role/structural tokens.
    """
    total = 0
    for m in messages:
        total += 4  # per-message structural overhead (role, delimiters)
        total += estimate_tokens(m.content)
        for tc in m.tool_calls:
            total += estimate_tokens(tc.name)
            for key, value in tc.arguments.items():
                total += estimate_tokens(str(key)) + estimate_tokens(str(value))
    return total


@dataclass
class UsageSnapshot:
    """A point-in-time reading handed to `on_event` observers.

    `context_estimated` and `cumulative_estimated` are deliberately separate:
    a single call can report exact usage (so the live context gauge is exact)
    even though an *earlier* call did not (so the cumulative total is partly
    estimated). Conflating them mislabels exact readings as approximate.
    """

    context_tokens: int  # live size of THIS loop's context
    cumulative_tokens: int  # total spent since start (whole run tree)
    context_estimated: bool  # was THIS context reading estimated?
    cumulative_estimated: bool  # was any call in the run estimated?


@dataclass
class UsageMeter:
    """Accumulates token usage across an entire `run` tree (shared via recursion)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(
        self,
        usage: Usage | None,
        *,
        sent_messages: list[Message],
        response_text: str | None = None,
    ) -> tuple[int, bool]:
        """Fold one model call into the running totals.

        Returns ``(context_tokens, estimated)`` for THIS call: the prompt-token
        count (how big the context we just sent was) and whether that figure was
        estimated rather than reported by the provider.
        """
        self.calls += 1
        if usage is not None and (usage.prompt_tokens or usage.completion_tokens):
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            return usage.prompt_tokens, False

        # No provider usage — estimate both sides and flag the run as approximate.
        self.estimated = True
        context = estimate_message_tokens(sent_messages)
        self.prompt_tokens += context
        self.completion_tokens += estimate_tokens(response_text)
        return context, True

    def snapshot(self, context_tokens: int, context_estimated: bool) -> UsageSnapshot:
        return UsageSnapshot(
            context_tokens=context_tokens,
            cumulative_tokens=self.total_tokens,
            context_estimated=context_estimated,
            cumulative_estimated=self.estimated,
        )
