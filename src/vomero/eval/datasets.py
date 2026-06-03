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
