"""Persistent, read-only search index — built once, loaded, never rebuilt.

The in-memory `HybridIndex` (search.py) re-reads and (for dense) re-embeds the
whole corpus on first search of every process. That is fine to tens of thousands
of documents but insane at millions: you pay the embed cost and a full corpus
read on every run, and dies-with-the-process. `PersistentIndex` separates the
two phases the way real retrieval systems do:

* BUILD (offline, once — `vomero index`): read each document once, write a
  persistent lexical index (SQLite FTS5 — on-disk, queried without loading the
  corpus) and, if an embedder is configured, the document vectors (embedded
  ONCE, never again). Keyed by content hash, so a rebuild re-embeds only changed
  docs.

* OPEN + SEARCH (serving): open the on-disk index READ-ONLY and query it. No
  corpus read, no re-embedding, process start is ~instant. Because it is opened
  read-only and is identical for every user, it is shared infrastructure (like
  the corpus folder itself), not per-run state — so it fits Vomero's stateless,
  reusable-pod model: the index belongs to the corpus, not to the run.

Dependency-light: FTS5 ships with stdlib `sqlite3`; dense vectors are stored as
raw float32 and scanned in pure Python. That linear dense scan is the one piece
that doesn't reach true millions — an ANN backend (faiss/hnswlib) slots in behind
the same `search()` method when needed; lexical (FTS5) is already sublinear.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from .search import Hit, tokenize

_MANIFEST = "manifest.json"
_LEXICAL = "lexical.sqlite"
_VECTORS = "dense.f32"
_VERSION = 1


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _fts_query(query: str) -> str:
    """Turn a natural-language query into a safe FTS5 MATCH expression: the
    distinct content terms OR-ed together, each quoted so punctuation can't be
    read as FTS operators. Empty when the query has no indexable terms."""
    terms = sorted({t for t in tokenize(query)})
    return " OR ".join(f'"{t}"' for t in terms)


def _snippet(text: str, query: str, width: int = 240) -> str:
    low = text.lower()
    pos = -1
    for t in tokenize(query):
        p = low.find(t)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos <= 0:
        return text[:width].strip()
    start = max(0, pos - width // 3)
    return ("…" if start > 0 else "") + text[start:start + width].strip()


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class PersistentIndex:
    """A built-on-disk lexical (+ optional dense) index over a document set."""

    _RRF_C = 60

    def __init__(self, index_dir: str | Path, *, embedder=None):
        self.dir = Path(index_dir)
        self._embedder = embedder
        self._manifest: dict | None = None
        self._conn: sqlite3.Connection | None = None
        self._doc_ids: list[str] | None = None   # row order of the vector file
        self._vecs: list[array.array] | None = None

    # -- build ----------------------------------------------------------------
    @classmethod
    def build(cls, docs: list[tuple[str, str]], index_dir: str | Path, *,
              embedder=None, embedding_model: str = "", embed_chars: int = 8000) -> "PersistentIndex":
        """Build the index from `(doc_id, text)` pairs and write it to
        `index_dir`. `doc_id` is the locator the source uses to fetch a doc
        (a relative path for a Corpus). Overwrites any existing index there."""
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        for f in (_LEXICAL, _VECTORS):
            (d / f).unlink(missing_ok=True)

        # Lexical: FTS5 with the doc id stored UNINDEXED alongside the body.
        conn = sqlite3.connect(str(d / _LEXICAL))
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, body)")
        conn.executemany("INSERT INTO docs(doc_id, body) VALUES (?, ?)", docs)
        conn.commit()
        conn.close()

        manifest = {
            "version": _VERSION,
            "embedding_model": embedding_model if embedder else "",
            "dim": 0,
            "docs": {doc_id: _sha1(text) for doc_id, text in docs},
        }

        # Dense: embed each doc ONCE, write float32 rows aligned to doc_ids.
        if embedder is not None and docs:
            vecs = embedder.embed([t[:embed_chars] for _, t in docs])
            dim = len(vecs[0]) if vecs else 0
            manifest["dim"] = dim
            with open(d / _VECTORS, "wb") as fh:
                for v in vecs:
                    fh.write(array.array("f", v).tobytes())
            manifest["doc_order"] = [doc_id for doc_id, _ in docs]

        (d / _MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        return cls(d, embedder=embedder)

    # -- open + introspect ----------------------------------------------------
    @classmethod
    def exists(cls, index_dir: str | Path) -> bool:
        d = Path(index_dir)
        return (d / _MANIFEST).exists() and (d / _LEXICAL).exists()

    def _load(self) -> None:
        if self._manifest is not None:
            return
        self._manifest = json.loads((self.dir / _MANIFEST).read_text(encoding="utf-8"))
        # Read-only connection: a serving pod must never mutate shared index data.
        uri = f"file:{self.dir / _LEXICAL}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

    def _load_vectors(self) -> None:
        if self._vecs is not None:
            return
        self._load()
        dim = self._manifest.get("dim", 0)
        order = self._manifest.get("doc_order", [])
        vpath = self.dir / _VECTORS
        if not dim or not order or not vpath.exists():
            self._doc_ids, self._vecs = [], []
            return
        raw = vpath.read_bytes()
        stride = dim * 4  # float32
        self._doc_ids = order
        self._vecs = [
            array.array("f", raw[i * stride:(i + 1) * stride]) for i in range(len(order))
        ]

    def warmup(self) -> None:
        """Open the on-disk index and load the dense vectors now, so a server
        pays it at startup instead of on the first dense query."""
        self._load()
        if self._embedder is not None:
            self._load_vectors()

    def is_stale(self, docs: list[tuple[str, str]]) -> bool:
        """True if the current documents differ from what was indexed (added,
        removed, or content-changed) — so the caller knows to rebuild."""
        self._load()
        indexed = self._manifest.get("docs", {})
        current = {doc_id: _sha1(text) for doc_id, text in docs}
        return indexed != current

    def _doc_text(self, doc_id: str) -> str:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT body FROM docs WHERE doc_id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row[0] if row else ""

    # -- search ---------------------------------------------------------------
    def _lexical_rank(self, query: str, limit: int) -> list[tuple[str, float]]:
        self._load()
        assert self._conn is not None
        match = _fts_query(query)
        if not match:
            return []
        # bm25() returns a cost (lower = better); negate so higher = better.
        rows = self._conn.execute(
            "SELECT doc_id, -bm25(docs) AS score FROM docs "
            "WHERE docs MATCH ? ORDER BY score DESC LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(doc_id, float(score)) for doc_id, score in rows]

    def _dense_rank(self, query: str, limit: int) -> list[tuple[str, float]]:
        if self._embedder is None:
            return []
        self._load_vectors()
        if not self._vecs:
            return []
        qv = self._embedder.embed([query])[0]
        scored = [(self._doc_ids[i], _cosine(qv, v)) for i, v in enumerate(self._vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        # Over-fetch per ranker so fusion has something to combine.
        depth = max(k, 20)
        lexical = self._lexical_rank(query, depth)
        if mode == "lexical" or (mode == "hybrid" and self._embedder is None):
            ranked = lexical
        elif mode == "dense":
            ranked = self._dense_rank(query, depth)
        else:
            ranked = self._rrf(lexical, self._dense_rank(query, depth))
        hits: list[Hit] = []
        for doc_id, score in ranked[:k]:
            hits.append(Hit(doc=doc_id, score=score,
                            snippet=_snippet(self._doc_text(doc_id), query)))
        return hits

    def _rrf(self, *rank_lists: list[tuple[str, float]]) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for rl in rank_lists:
            for rank, (doc_id, _score) in enumerate(rl):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self._RRF_C + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)
