"""Evaluation harness: measure Vomero against a stuff-the-context baseline.

The piece that turns "should be better" into numbers. Build items with a
dataset adapter, then `compare()` the `RLMRunner` against the
`StuffBaselineRunner` over the same questions.
"""

from __future__ import annotations

from .datasets import load_jsonl, load_multihoprag
from .harness import EvalItem, ItemResult, Report, compare, evaluate
from .optimize import OptimizeResult, optimize, propose_instructions
from .runners import Outcome, RLMRunner, StuffBaselineRunner

__all__ = [
    "EvalItem", "ItemResult", "Report", "evaluate", "compare",
    "Outcome", "RLMRunner", "StuffBaselineRunner",
    "load_jsonl", "load_multihoprag",
    "optimize", "OptimizeResult", "propose_instructions",
]
