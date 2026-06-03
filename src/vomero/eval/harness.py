"""The eval loop: score a runner on a set of items, and compare runners.

This is the measurement piece — without it "beats SOTA" is unverifiable. An
`EvalItem` carries a question, the gold answer, and the data to answer over
(a `Source`). `evaluate()` runs a runner across the items and aggregates
correctness (exact-match, token-F1, contains, optional LLM-judge) alongside the
cost signals (tokens, calls, latency). `compare()` runs several runners over the
SAME items so RLM and the stuff-the-context baseline sit side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import metrics


@dataclass
class EvalItem:
    question: str
    answer: str
    source: object  # a Source (Corpus/Context) to answer over
    meta: dict = field(default_factory=dict)


@dataclass
class ItemResult:
    question: str
    gold: str
    pred: str
    exact: float
    f1: float
    contains: float
    judge: float | None
    tokens: int
    calls: int
    seconds: float
    truncated: bool


@dataclass
class Report:
    runner: str
    results: list[ItemResult]

    @property
    def n(self) -> int:
        return len(self.results)

    def _mean(self, attr: str) -> float:
        return (sum(getattr(r, attr) for r in self.results) / self.n) if self.n else 0.0

    @property
    def exact(self) -> float:
        return self._mean("exact")

    @property
    def f1(self) -> float:
        return self._mean("f1")

    @property
    def contains(self) -> float:
        return self._mean("contains")

    @property
    def judge(self) -> float | None:
        graded = [r.judge for r in self.results if r.judge is not None]
        return sum(graded) / len(graded) if graded else None

    @property
    def mean_tokens(self) -> float:
        return self._mean("tokens")

    @property
    def mean_seconds(self) -> float:
        return self._mean("seconds")

    @property
    def truncated_frac(self) -> float:
        return self._mean("truncated")

    def summary(self) -> str:
        j = f"  judge {self.judge:.3f}" if self.judge is not None else ""
        return (
            f"[{self.runner}] n={self.n}  EM {self.exact:.3f}  F1 {self.f1:.3f}  "
            f"contains {self.contains:.3f}{j}  | {self.mean_tokens:,.0f} tok/q  "
            f"{self.mean_seconds:.1f}s/q  truncated {self.truncated_frac:.0%}"
        )


def evaluate(
    items: list[EvalItem],
    runner,
    *,
    judge_client=None,
    judge_model=None,
    on_item: Callable[[ItemResult], None] | None = None,
) -> Report:
    """Run `runner` over `items` and aggregate. If `judge_client` is given, each
    item is also graded by a model (for free-form answers string metrics miss)."""
    results: list[ItemResult] = []
    for it in items:
        outcome = runner.answer(it.question, it.source)
        judge = (
            metrics.llm_judge(judge_client, it.question, outcome.answer, it.answer,
                              model=judge_model)
            if judge_client is not None else None
        )
        r = ItemResult(
            question=it.question, gold=it.answer, pred=outcome.answer,
            exact=metrics.exact_match(outcome.answer, it.answer),
            f1=metrics.token_f1(outcome.answer, it.answer),
            contains=metrics.contains_gold(outcome.answer, it.answer),
            judge=judge,
            tokens=outcome.tokens, calls=outcome.calls, seconds=outcome.seconds,
            truncated=outcome.truncated,
        )
        results.append(r)
        if on_item is not None:
            on_item(r)
    return Report(runner=runner.name, results=results)


def compare(items: list[EvalItem], runners: list, **kw) -> list[Report]:
    """Evaluate several runners over the SAME items (RLM vs. baseline)."""
    return [evaluate(items, r, **kw) for r in runners]
