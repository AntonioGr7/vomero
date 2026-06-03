"""Answer-scoring metrics (SQuAD-style), dependency-free.

`exact_match` and `token_f1` are the standard short-answer QA metrics: they
normalize (lowercase, strip articles/punctuation/extra whitespace) before
comparing, so "The P-ATLAS team." scores against "p atlas team". For long or
free-form answers these undercount correctness — pair them with `llm_judge`
(a model graded yes/no), which the harness can use when a client is supplied.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize(text: str) -> str:
    """Lowercase; drop punctuation, articles, and redundant whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    """1.0 iff the normalized strings are identical."""
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold (SQuAD-style)."""
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)  # both empty => 1.0, else 0.0
    common = Counter(p_toks) & Counter(g_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def contains_gold(pred: str, gold: str) -> float:
    """1.0 if the normalized gold answer appears as a substring of the
    prediction — a lenient check for verbose answers that bury the right span."""
    return float(normalize(gold) in normalize(pred))


_JUDGE_SYSTEM = (
    "You grade a candidate answer against a reference answer for the same "
    "question. Reply with exactly one token: YES if the candidate conveys the "
    "same answer as the reference (wording may differ), otherwise NO."
)


def llm_judge(client, question: str, pred: str, gold: str, *, model=None) -> float:
    """Model-graded correctness (1.0/0.0). For free-form answers where string
    metrics undercount. `client` is any LLMClient; one cheap call per item."""
    from ..llm.base import Message

    msg = (
        f"Question: {question}\nReference answer: {gold}\n"
        f"Candidate answer: {pred}\n\nSame answer? Reply YES or NO."
    )
    resp = client.complete(
        [Message("system", _JUDGE_SYSTEM), Message("user", msg)], model=model
    )
    return float((resp.content or "").strip().upper().startswith("YES"))
