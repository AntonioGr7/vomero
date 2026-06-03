"""A native prompt optimizer — the DSPy-style "tune the prompt to a metric" loop,
built directly on the eval harness instead of a separate framework.

What gets tuned is the engine's `extra_instructions` block (appended to the
system prompt). `optimize()` scores each candidate instruction block on a train
set with the same metrics the harness reports, and returns the best — so a
better prompt is *measured*, not hand-guessed. `propose_instructions()` is the
optional bootstrap step: ask the model to generate candidate blocks to search
over.

This is deliberately simple (selection over candidates, no gradient/DSP magic):
it's transparent, dependency-free, and reuses everything in eval/. Swap in a
fancier search later behind the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..llm.base import Message
from .harness import EvalItem, Report, evaluate
from .runners import RLMRunner


@dataclass
class OptimizeResult:
    best_instructions: str | None
    best_score: float
    # (candidate, score, full report) for every candidate, best first.
    scored: list[tuple[str | None, float, Report]]

    def summary(self) -> str:
        lines = [f"best score={self.best_score:.3f} (metric on train set)"]
        for cand, score, _ in self.scored:
            tag = "<baseline>" if not cand else (cand[:60].replace("\n", " ") + "…")
            lines.append(f"  {score:.3f}  {tag}")
        return "\n".join(lines)


def optimize(
    engine,
    train_items: list[EvalItem],
    candidates: list[str | None],
    *,
    metric: str = "f1",
    judge_client=None,
    judge_model=None,
    on_candidate: Callable[[str | None, float], None] | None = None,
    keep_best: bool = True,
) -> OptimizeResult:
    """Score each candidate `extra_instructions` block on `train_items` and pick
    the best by `metric` (one of: exact, f1, contains, judge).

    The engine's `extra_instructions` is set per candidate (sequentially — the
    engine holds no per-run state otherwise). With `keep_best=True` the engine is
    left configured with the winning block so it's ready to use.
    """
    original = engine.extra_instructions
    scored: list[tuple[str | None, float, Report]] = []
    try:
        for cand in candidates:
            engine.extra_instructions = cand
            report = evaluate(
                train_items, RLMRunner(engine),
                judge_client=judge_client, judge_model=judge_model,
            )
            value = getattr(report, metric)
            score = float(value) if value is not None else 0.0
            scored.append((cand, score, report))
            if on_candidate is not None:
                on_candidate(cand, score)
    finally:
        # Restore unless we're keeping the winner (set below).
        if not scored or not keep_best:
            engine.extra_instructions = original

    scored.sort(key=lambda t: t[1], reverse=True)
    best_cand, best_score, _ = scored[0]
    if keep_best:
        engine.extra_instructions = best_cand
    return OptimizeResult(best_instructions=best_cand, best_score=best_score, scored=scored)


_PROPOSE_SYSTEM = (
    "You improve the system prompt of an agent that answers questions by writing "
    "Python in a REPL to explore data (grep/slice/chunk), delegating heavy reading "
    "to sub-model calls, and recursing on sub-questions. Propose additional "
    "instruction blocks that would make it more accurate and efficient."
)


def propose_instructions(client, n: int = 4, *, base: str = "", model=None) -> list[str]:
    """Bootstrap candidate instruction blocks from the model. Returns up to `n`
    distinct blocks to feed to `optimize(...)` as candidates. Pure helper — no
    scoring here. Always pair with `None` (the baseline) in the candidate list."""
    ask = (
        f"Propose {n} DISTINCT, concise instruction blocks (2-5 lines each) that "
        "could improve such an agent's accuracy and token-efficiency. Separate "
        "each block with a line containing only '---'. No numbering, no preamble."
    )
    if base.strip():
        ask += f"\n\nCurrent extra instructions to improve on:\n{base.strip()}"
    resp = client.complete(
        [Message("system", _PROPOSE_SYSTEM), Message("user", ask)], model=model
    )
    blocks = [b.strip() for b in (resp.content or "").split("---")]
    return [b for b in blocks if b][:n]
