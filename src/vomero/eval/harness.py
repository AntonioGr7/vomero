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


def is_abstention(text: str) -> bool:
    """Whether an answer declines to answer ('Insufficient information.')."""
    return "insufficient information" in metrics.normalize(text)


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
    # Retrieval diagnostics (None when the question has no mapped gold evidence,
    # i.e. null queries, or the runner captured no provenance — baseline/closed):
    #   doc_recall    fraction of gold evidence docs the run retrieved
    #   all_docs_found 1.0 iff EVERY gold evidence doc was retrieved (the
    #                  multi-hop completeness signal — one missed hop sinks it)
    doc_recall: float | None = None
    all_docs_found: float | None = None
    # Whether the prediction abstained, and whether this item is a null query
    # (gold = "Insufficient information."). Together these give the abstention
    # confusion matrix: correct abstention vs. hallucinated answer vs. giving up
    # on an answerable question.
    abstained: bool = False
    is_null: bool = False
    # The item's meta (question_type, n_hops, ...), kept for sliced breakdowns.
    meta: dict = field(default_factory=dict)


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

    @staticmethod
    def _mean_opt(rows: list[ItemResult], attr: str) -> float | None:
        """Mean over rows where `attr` is not None (recall is undefined for
        null queries / non-navigating runners)."""
        vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def doc_recall(self) -> float | None:
        return self._mean_opt(self.results, "doc_recall")

    @property
    def all_docs_found(self) -> float | None:
        return self._mean_opt(self.results, "all_docs_found")

    def recall_summary(self) -> str | None:
        """Retrieval recall, overall and sliced by hop count (n evidence docs).
        None when no item carried provenance + gold docs (e.g. baseline only)."""
        scored = [r for r in self.results if r.doc_recall is not None]
        if not scored:
            return None
        lines = [
            f"[{self.runner}] retrieval recall (n={len(scored)}): "
            f"doc-recall {self.doc_recall:.3f}  all-hops-found {self.all_docs_found:.3f}"
        ]
        for hops in sorted({r.meta.get("n_hops", 0) for r in scored}):
            rows = [r for r in scored if r.meta.get("n_hops", 0) == hops]
            lines.append(
                f"    {hops}-hop (n={len(rows)}): "
                f"doc-recall {self._mean_opt(rows, 'doc_recall'):.3f}  "
                f"all-found {self._mean_opt(rows, 'all_docs_found'):.3f}"
            )
        return "\n".join(lines)

    def abstention_summary(self) -> str:
        """The abstention confusion matrix: on null queries, how often we
        correctly said 'Insufficient information'; on answerable ones, how often
        we wrongly gave up (false abstention)."""
        nulls = [r for r in self.results if r.is_null]
        answerable = [r for r in self.results if not r.is_null]
        correct_abstain = sum(r.abstained for r in nulls)
        false_abstain = sum(r.abstained for r in answerable)
        parts = [f"[{self.runner}] abstention:"]
        if nulls:
            parts.append(
                f"  null queries n={len(nulls)}: correct-abstention "
                f"{correct_abstain / len(nulls):.0%} ({correct_abstain}/{len(nulls)})")
        if answerable:
            parts.append(
                f"  answerable n={len(answerable)}: false-abstention "
                f"{false_abstain / len(answerable):.0%} ({false_abstain}/{len(answerable)})")
        return "\n".join(parts)

    def by_type(self) -> str:
        """Accuracy (judge if available, else EM) sliced by question_type."""
        types = sorted({r.meta.get("type", "") for r in self.results})
        if types == [""]:
            return ""
        lines = [f"[{self.runner}] by question_type:"]
        for t in types:
            rows = [r for r in self.results if r.meta.get("type", "") == t]
            acc = self._mean_opt(rows, "judge")
            label, val = ("judge", acc) if acc is not None else \
                ("EM", sum(r.exact for r in rows) / len(rows))
            lines.append(f"    {t or '(none)'} (n={len(rows)}): {label} {val:.3f}")
        return "\n".join(lines)

    def summary(self) -> str:
        j = f"  judge {self.judge:.3f}" if self.judge is not None else ""
        rc = f"  recall {self.doc_recall:.3f}" if self.doc_recall is not None else ""
        return (
            f"[{self.runner}] n={self.n}  EM {self.exact:.3f}  F1 {self.f1:.3f}  "
            f"contains {self.contains:.3f}{j}{rc}  | {self.mean_tokens:,.0f} tok/q  "
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
        # Retrieval recall: did the run actually touch the gold evidence docs?
        # Only defined when the item has mapped gold docs and provenance was
        # captured (RLM runner). One missed doc => all_docs_found is 0.
        gold_docs = set(it.meta.get("evidence_docs") or [])
        recall = all_found = None
        if gold_docs and outcome.retrieved_docs is not None:
            hit = gold_docs & outcome.retrieved_docs
            recall = len(hit) / len(gold_docs)
            all_found = float(hit == gold_docs)
        r = ItemResult(
            question=it.question, gold=it.answer, pred=outcome.answer,
            exact=metrics.exact_match(outcome.answer, it.answer),
            f1=metrics.token_f1(outcome.answer, it.answer),
            contains=metrics.contains_gold(outcome.answer, it.answer),
            judge=judge,
            tokens=outcome.tokens, calls=outcome.calls, seconds=outcome.seconds,
            truncated=outcome.truncated,
            doc_recall=recall, all_docs_found=all_found,
            abstained=is_abstention(outcome.answer),
            is_null=bool(it.meta.get("is_null")),
            meta=it.meta,
        )
        results.append(r)
        if on_item is not None:
            on_item(r)
    return Report(runner=runner.name, results=results)


def compare(items: list[EvalItem], runners: list, **kw) -> list[Report]:
    """Evaluate several runners over the SAME items (RLM vs. baseline)."""
    return [evaluate(items, r, **kw) for r in runners]
