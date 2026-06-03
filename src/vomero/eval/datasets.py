"""Dataset adapters → `list[EvalItem]`.

* `load_jsonl` — the generic, offline path. One JSON object per line with
  `question`/`answer` (aliases accepted) and EITHER an inline `context`
  (string or list of strings → a `Context`) or a shared `source` passed in.
* `load_multihoprag` — reads the MultiHopRAG QA set and points every item at the
  shared `Corpus(data/multihoprag)` materialized by data/download_corpus.py.

Adapters stay thin: they only build `EvalItem`s; scoring lives in the harness.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..context import Context, Corpus
from .harness import EvalItem

# Field-name aliases seen across QA datasets.
_Q_KEYS = ("question", "query", "q", "prompt")
_A_KEYS = ("answer", "gold", "label", "a", "output")


def _pick(d: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None


def load_jsonl(path: str | Path, *, source=None, limit: int | None = None) -> list[EvalItem]:
    """Load items from a JSONL file. Each line needs a question and an answer;
    its data is an inline `context` (string/list) or the shared `source`."""
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        q, a = _pick(row, _Q_KEYS), _pick(row, _A_KEYS)
        if q is None or a is None:
            raise ValueError(f"row missing question/answer: {row!r}")
        src = source
        if "context" in row and row["context"] is not None:
            src = Context(row["context"])
        if src is None:
            raise ValueError("row has no inline 'context' and no shared source given")
        items.append(EvalItem(question=q, answer=a, source=src,
                              meta={k: row[k] for k in row if k not in (*_Q_KEYS, *_A_KEYS)}))
        if limit is not None and len(items) >= limit:
            break
    return items


# The MultiHopRAG QA set (queries + gold answers). corpus.json is laid out by
# data/download_corpus.py; this is its companion question set.
MULTIHOPRAG_QA_URL = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"


def load_multihoprag(data_dir: str | Path = "data/multihoprag", *,
                     qa_file: str | Path | None = None,
                     limit: int | None = None,
                     mode: str = "corpus") -> list[EvalItem]:
    """Build EvalItems for MultiHopRAG over the materialized corpus.

    `mode="corpus"` points each item at the folder `Corpus` (RLM navigates
    files); `mode="context"` loads every article into one in-memory `Context`
    (the long-prompt / context-as-a-variable case). `qa_file` defaults to
    `<data_dir>/MultiHopRAG.json`; if absent, raises with the download URL.
    """
    data_dir = Path(data_dir)
    qa_path = Path(qa_file) if qa_file else data_dir / "MultiHopRAG.json"
    if not qa_path.exists():
        raise FileNotFoundError(
            f"MultiHopRAG QA set not found at {qa_path}. Download it:\n"
            f"  curl -L -o {qa_path} {MULTIHOPRAG_QA_URL}\n"
            "and materialize the corpus with: uv run python data/download_corpus.py"
        )
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    if mode == "context":
        corpus = Corpus(data_dir)
        docs = [corpus.read(p) for p in corpus.files()]
        shared = Context(docs)
    else:
        shared = Corpus(data_dir)

    items: list[EvalItem] = []
    for row in qa:
        q, a = _pick(row, _Q_KEYS), _pick(row, _A_KEYS)
        if q is None or a is None:
            continue
        items.append(EvalItem(question=q, answer=a, source=shared,
                              meta={"type": row.get("question_type", "")}))
        if limit is not None and len(items) >= limit:
            break
    return items


# --- synthetic needle-in-a-haystack (leakage-proof) --------------------------
#
# The contamination-free benchmark. Each "needle" is an INVENTED fact (a random
# vault code), so no model can answer it from training memory — closed-book must
# score ~0. Many needles are injected at known depths into one large shared
# haystack big enough to overflow the baseline's window: the stuff-it baseline
# then misses any needle past its truncation point, while an RLM that greps finds
# them at any depth. This is the setup that actually isolates retrieval.

_FILLER_SENTENCES = [
    "The quarterly logistics review recorded no significant anomalies in regional throughput.",
    "Maintenance crews completed the scheduled inspection well ahead of the projected timeline.",
    "Analysts noted that seasonal demand remained broadly consistent with prior-year patterns.",
    "The committee deferred the procurement decision pending a further compliance review.",
    "Field reports indicated stable network performance across all monitored districts.",
    "A routine audit confirmed that inventory counts reconciled with the central ledger.",
    "The working group circulated revised guidelines for the upcoming reporting cycle.",
    "Operational metrics stayed within expected tolerances throughout the measurement window.",
]


def _rand_token(rng: random.Random) -> str:
    """A distinctive invented identifier like 'KQF-8312' — unmemorizable."""
    alpha = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # drop I/O to avoid digit ambiguity
    return ("".join(rng.choice(alpha) for _ in range(3)) + "-"
            + "".join(rng.choice("0123456789") for _ in range(4)))


def _filler_text(filler, target: int, rng: random.Random) -> str:
    """Build ≈`target` chars of haystack: from a Corpus, a string, or synthesized
    filler sentences. Real-text filler makes a more realistic haystack."""
    if isinstance(filler, Corpus):
        parts, total = [], 0
        for p in filler.files():
            t = filler.read(p)
            parts.append(t)
            total += len(t)
            if total >= target:
                break
        text = "\n\n".join(parts)
    elif isinstance(filler, str):
        text = filler
    else:
        buf, total, i = [], 0, 0
        while total < target:
            s = _FILLER_SENTENCES[i % len(_FILLER_SENTENCES)]
            buf.append(s)
            total += len(s) + 1
            i += 1
        text = " ".join(buf)
    if text and len(text) < target:  # pad by repetition to reach the target size
        text = (text + "\n\n") * ((target // len(text)) + 1)
    return text[:target]


def make_needle_items(
    n: int = 20,
    *,
    total_chars: int = 2_000_000,
    filler=None,
    seed: int = 0,
) -> list[EvalItem]:
    """Generate `n` needle questions over ONE shared haystack `Context`.

    `total_chars` sizes the haystack (default ~2M ≈ 500k tokens, enough to
    overflow a 128k-token window). `filler` is a `Corpus`/string to use as
    haystack text, else synthetic filler. `seed` makes it reproducible. Needles
    are spread across depths 0..1 so you can see accuracy-vs-depth: a truncating
    baseline fails the deep ones; a grepping RLM should not. All items share one
    `Context` object (memory-efficient)."""
    rng = random.Random(seed)
    base = _filler_text(filler, total_chars, rng)

    # Compute insertion points on the ORIGINAL base, then insert deepest-first so
    # earlier insertions don't shift the positions of later ones.
    plan = []
    for i in range(n):
        depth = (i + 0.5) / n
        key, val = _rand_token(rng), _rand_token(rng)
        pos = int(len(base) * depth)
        nl = base.find("\n", pos)
        if nl != -1:
            pos = nl + 1
        plan.append((pos, key, val, round(depth, 3)))

    text = base
    for pos, key, val, _ in sorted(plan, key=lambda x: x[0], reverse=True):
        sentence = f"\nNOTE: The access code for vault {key} is {val}.\n"
        text = text[:pos] + sentence + text[pos:]
    shared = Context(text)

    return [
        EvalItem(
            question=f"What is the access code for vault {key}?",
            answer=val,
            source=shared,
            meta={"depth": depth},
        )
        for _, key, val, depth in sorted(plan, key=lambda x: x[3])
    ]
