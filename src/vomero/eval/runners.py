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


# A terse-answer instruction shared by both runners so EM/F1 are a FAIR
# comparison: gold answers in these benchmarks are short spans, so a verbose
# (but correct) answer is unfairly punished by string metrics. Applied to the
# RLM engine via `extra_instructions` and to the baseline's system prompt.
TERSE_ANSWER = (
    "Answer with ONLY the shortest exact answer span — a name, entity, number, "
    "date, or yes/no. No explanation, no full sentence, no surrounding text. "
    "If — after searching thoroughly — the data does not contain enough to "
    "answer the question, reply with exactly: Insufficient information. Do not "
    "guess: a wrong guess scores no better than abstaining, and abstaining when "
    "the answer IS present is itself an error."
)


@dataclass
class Outcome:
    answer: str
    tokens: int
    calls: int
    seconds: float
    truncated: bool = False  # baseline only: context didn't fit and was cut
    # Relative paths the RLM run actually retrieved (read/peek/grep), from the
    # source access log. The retrieval-recall metric scores this against the
    # question's gold evidence docs. None when provenance wasn't captured
    # (baseline/closed-book don't navigate a source).
    retrieved_docs: set[str] | None = None


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
        # Isolate this item's provenance: the source is shared across items, so
        # clear the log before the run (return_trajectory re-enables it). The set
        # of docs the run touched is the input to the retrieval-recall metric.
        if hasattr(source, "reset_access_log"):
            source.reset_access_log()
        t0 = time.monotonic()
        result = self.engine.run(question, source, channel=NullChannel(),
                                 meter=meter, return_trajectory=True)
        retrieved = {str(e.doc) for e in result.provenance}
        return Outcome(
            answer=result.answer, tokens=meter.total_tokens, calls=meter.calls,
            seconds=time.monotonic() - t0, retrieved_docs=retrieved,
        )


class StuffBaselineRunner:
    """The long-context control: render the whole source into one prompt and ask
    the model in a single call. Truncates to `max_chars` (≈ the window) when the
    data doesn't fit — flagged on the Outcome, since that is exactly the failure
    mode RLM exists to avoid."""

    name = "baseline"

    _SYSTEM = "Answer the question using ONLY the provided context."

    def __init__(self, client, *, model=None, max_chars: int = 400_000, terse: bool = False):
        self.client = client
        self.model = model
        self.max_chars = max_chars
        self.system = self._SYSTEM + ("\n" + TERSE_ANSWER if terse else "")
        # Rendering a folder reads every file; cache per source so a shared
        # corpus isn't re-read once per question.
        self._cache: dict[int, str] = {}

    def _full_text(self, source) -> str:
        key = id(source)
        if key not in self._cache:
            if isinstance(source, Context):
                text = source.text
            elif isinstance(source, Corpus):
                text = "\n\n".join(f"### {p}\n{source.read(p)}" for p in source.files())
            else:  # best effort for any other Source
                text = str(source.overview())
            self._cache[key] = text
        return self._cache[key]

    def _render(self, source) -> tuple[str, bool]:
        text = self._full_text(source)
        truncated = len(text) > self.max_chars
        return (text[: self.max_chars] if truncated else text), truncated

    def answer(self, question: str, source) -> Outcome:
        context, truncated = self._render(source)
        meter = UsageMeter()
        msgs = [
            Message("system", self.system),
            Message("user", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        t0 = time.monotonic()
        resp = self.client.complete(msgs, model=self.model)
        meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
        return Outcome(
            answer=resp.content or "", tokens=meter.total_tokens, calls=meter.calls,
            seconds=time.monotonic() - t0, truncated=truncated,
        )


class ClosedBookRunner:
    """The contamination control: answer with NO context at all, purely from the
    model's parametric memory. Its score is the leakage floor — on a benchmark
    built from public text the model was trained on, the model may already
    "know" the answers. The real context contribution of any context-using
    system is *its* score minus this one. If closed-book ≈ baseline, the task
    isn't testing context use, and RLM-vs-baseline is meaningless."""

    name = "closed_book"

    _SYSTEM = (
        "Answer the question from your own knowledge alone. No documents are "
        "provided. If unsure, give your single best guess."
    )

    def __init__(self, client, *, model=None, terse: bool = False):
        self.client = client
        self.model = model
        self.system = self._SYSTEM + ("\n" + TERSE_ANSWER if terse else "")

    def answer(self, question: str, source) -> Outcome:  # source ignored, by design
        meter = UsageMeter()
        msgs = [Message("system", self.system), Message("user", f"Question: {question}")]
        t0 = time.monotonic()
        resp = self.client.complete(msgs, model=self.model)
        meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
        return Outcome(
            answer=resp.content or "", tokens=meter.total_tokens, calls=meter.calls,
            seconds=time.monotonic() - t0,
        )
