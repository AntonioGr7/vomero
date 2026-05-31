"""History compaction for the RLM loop.

When the live context size crosses a fraction of the model's window, we replace
the bulk of the transcript with a single dense summary so the loop can keep
going without losing task-critical state. The approach mirrors what production
agent stacks do:

* **Keep the preamble verbatim** — the system prompt and the original question
  are never summarized away (they frame everything).
* **Keep the most recent turns verbatim** — recent tool output is the most
  load-bearing for the next decision, and recency is exactly what a summary
  blurs. Only the *middle* is compressed.
* **Never orphan a tool result** — in the chat schema an assistant message with
  tool calls must be immediately followed by its `tool` results. The recent
  tail is snapped to a clean boundary so it never *starts* with a `tool`
  message (an instant API error). The middle is rendered to a flat transcript,
  so it carries no pairing constraint.
* **Faithful, structured summary** — a dedicated model call distills the middle
  into Task / Progress / Key findings / Open threads, told to preserve exact
  paths, identifiers, and values, and to MERGE any earlier summary rather than
  nest one. Idempotent across repeated compactions.
* **REPL state is preserved** — because the execution namespace survives
  compaction, the summary appends the still-defined variables so the model
  reuses them instead of recomputing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import Message
from ..usage import UsageMeter, estimate_message_tokens

_SUMMARY_SYSTEM = (
    "You are a compaction engine for an autonomous agent. You compress an "
    "in-progress transcript into a dense, faithful state summary that lets the "
    "agent continue with NO loss of task-critical information. Preserve every "
    "specific the agent will need: exact file paths, names, numbers, "
    "identifiers, decisions, and partial conclusions. Never invent facts. Be "
    "terse but complete — this summary REPLACES the transcript."
)

_SUMMARY_INSTRUCTION = """\
Summarize the transcript above into exactly these sections. Keep every concrete \
detail needed to finish the task; drop only redundant exploration noise.

## Task
The original question / goal, restated precisely.

## Progress so far
What has been done: files read, searches run, sub-questions delegated, code executed.

## Key findings
Facts established, each with its source (exact file path, and line/identifier \
where relevant) and exact values. This is the section that must not lose detail.

## Open threads / next steps
What remains: hypotheses to check, partial answers not yet verified, the planned next move.

If the transcript already contains a "## Task" block from an earlier \
compaction, MERGE it into one consolidated summary — do not summarize the \
summary or nest sections."""

_COMPACTION_HEADER = (
    "[Conversation compacted to reclaim context. Earlier exploration has been "
    "summarized below; the most recent steps follow verbatim. Continue from "
    "this state — do not restart.]\n\n"
)


@dataclass
class CompactionEvent:
    """Emitted to `on_event` when a compaction happens (for observability)."""

    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    summarized_messages: int


def _preamble_end(messages: list[Message]) -> int:
    """Index just past the preamble (leading system msgs + first user message)."""
    i = 0
    n = len(messages)
    while i < n and messages[i].role == "system":
        i += 1
    if i < n and messages[i].role == "user":
        i += 1
    return i


def _render_transcript(messages: list[Message]) -> str:
    """Flatten messages to a readable transcript for the summarizer."""
    parts: list[str] = []
    for m in messages:
        if m.role == "user":
            parts.append(f"USER: {m.content or ''}")
        elif m.role == "assistant":
            for tc in m.tool_calls:
                code = tc.arguments.get("code", "")
                parts.append(f"ASSISTANT ran python:\n{code}")
            if m.content:
                parts.append(f"ASSISTANT: {m.content}")
        elif m.role == "tool":
            parts.append(f"RESULT:\n{m.content or ''}")
    return "\n\n".join(parts)


class Compactor:
    """Decides when to compact and produces the compacted message list."""

    def __init__(
        self,
        *,
        context_window: int,
        ratio: float = 0.8,
        keep_recent_messages: int = 6,
        min_summarize_messages: int = 4,
        min_reclaim_tokens: int = 2048,
    ):
        self.context_window = context_window
        self.ratio = ratio
        self.trigger_tokens = int(context_window * ratio)
        self.keep_recent_messages = keep_recent_messages
        self.min_summarize_messages = min_summarize_messages
        # Don't spend a summarization call unless the summarizable *middle* is at
        # least this big (estimated). Guards against the case where the context
        # is bloated by the protected preamble/recent-tail rather than the
        # middle — there, compaction can't reclaim much and would waste a call
        # (and can even grow the context). Set 0 to disable the floor.
        self.min_reclaim_tokens = min_reclaim_tokens

    def should_compact(self, context_tokens: int) -> bool:
        return self.trigger_tokens > 0 and context_tokens >= self.trigger_tokens

    def compact(
        self,
        messages: list[Message],
        *,
        client,
        model: str | None,
        meter: UsageMeter,
        state_description: str = "",
    ) -> tuple[list[Message], int]:
        """Return (new_messages, summarized_count).

        `summarized_count == 0` means it was a no-op (nothing worth summarizing,
        no model call made) and `messages` is returned unchanged.
        """
        preamble_end = _preamble_end(messages)
        n = len(messages)

        # Tail = last `keep_recent_messages`, snapped forward off any leading
        # `tool` message so the replayed tail is never an orphaned result.
        split = max(preamble_end, n - self.keep_recent_messages)
        while split < n and messages[split].role == "tool":
            split += 1

        middle = messages[preamble_end:split]
        if len(middle) < self.min_summarize_messages:
            return messages, 0  # not enough turns to be worth a summarization call
        if estimate_message_tokens(middle) < self.min_reclaim_tokens:
            # The bloat isn't in the summarizable middle — compacting now would
            # spend a call to reclaim ~nothing. Skip; the threshold will fire
            # again once enough reclaimable history accumulates in the middle.
            return messages, 0

        summary = self._summarize(
            _render_transcript(middle), client=client, model=model, meter=meter
        )

        body = _COMPACTION_HEADER + summary
        if state_description.strip():
            body += (
                "\n\n## Live REPL state\nThese variables are STILL DEFINED in your "
                "Python REPL — reuse them, do not recompute:\n" + state_description
            )

        new_messages = (
            list(messages[:preamble_end])
            + [Message("user", content=body)]
            + list(messages[split:])
        )
        return new_messages, len(middle)

    def _summarize(self, transcript: str, *, client, model, meter: UsageMeter) -> str:
        msgs = [
            Message("system", _SUMMARY_SYSTEM),
            Message("user", transcript + "\n\n" + _SUMMARY_INSTRUCTION),
        ]
        resp = client.complete(msgs, model=model)
        # The compaction call costs tokens too — fold it into the running total.
        meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
        return (resp.content or "").strip()
