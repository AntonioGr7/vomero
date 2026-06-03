"""Runners: the two systems an eval compares on the same questions.

* `RLMRunner` — the real Vomero engine over a `Source` (folder or in-memory
  context). The system under test.
* `StuffBaselineRunner` — the control: paste the ENTIRE context into one prompt
  and ask the model directly. This is the long-context baseline RLM has to beat;
  on data larger than the window it simply can't run, which is itself the point.

Both return an `Outcome` carrying the answer plus the cost signals an eval needs
(tokens, model calls, wall-clock), read from a per-item `UsageMeter`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..channel import NullChannel
from ..context import Context, Corpus
from ..llm.base import Message
from ..usage import UsageMeter


@dataclass
class Outcome:
    answer: str
    tokens: int
    calls: int
    seconds: float
    truncated: bool = False  # baseline only: context didn't fit and was cut


class RLMRunner:
    """Runs the Vomero RLM engine over a per-item source. `engine` is reused
    across items (it holds no per-run state); each item gets its own meter."""

    name = "rlm"

    def __init__(self, engine, *, max_total_tokens: int = 0, max_total_calls: int = 0):
        self.engine = engine
        self._budget = (max_total_tokens, max_total_calls)

    def answer(self, question: str, source) -> Outcome:
        meter = UsageMeter(
            max_total_tokens=self._budget[0], max_total_calls=self._budget[1]
        )
        t0 = time.monotonic()
        ans = self.engine.run(question, source, channel=NullChannel(), meter=meter)
        return Outcome(
            answer=ans, tokens=meter.total_tokens, calls=meter.calls,
            seconds=time.monotonic() - t0,
        )


class StuffBaselineRunner:
    """The long-context control: render the whole source into one prompt and ask
    the model in a single call. Truncates to `max_chars` (≈ the window) when the
    data doesn't fit — flagged on the Outcome, since that is exactly the failure
    mode RLM exists to avoid."""

    name = "baseline"

    _SYSTEM = (
        "Answer the question using ONLY the provided context. Be concise and "
        "answer directly."
    )

    def __init__(self, client, *, model=None, max_chars: int = 400_000):
        self.client = client
        self.model = model
        self.max_chars = max_chars

    def _render(self, source) -> tuple[str, bool]:
        if isinstance(source, Context):
            text = source.text
        elif isinstance(source, Corpus):
            text = "\n\n".join(
                f"### {p}\n{source.read(p)}" for p in source.files()
            )
        else:  # best effort for any other Source
            text = str(source.overview())
        truncated = len(text) > self.max_chars
        return (text[: self.max_chars] if truncated else text), truncated

    def answer(self, question: str, source) -> Outcome:
        context, truncated = self._render(source)
        meter = UsageMeter()
        msgs = [
            Message("system", self._SYSTEM),
            Message("user", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        t0 = time.monotonic()
        resp = self.client.complete(msgs, model=self.model)
        meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
        return Outcome(
            answer=resp.content or "", tokens=meter.total_tokens, calls=meter.calls,
            seconds=time.monotonic() - t0, truncated=truncated,
        )
