"""Ranked retrieval behind the `Source.search(...)` primitive.

`grep` is exact-substring/regex: it finds lines that literally match, and the
RLM strategy prompt itself warns that a literal keyword match "often lands on a
distractor" — the bridge entity in a multi-hop question is rarely phrased the
way the question is. `search` closes that recall gap with *ranked relevance*:

* `BM25Index` — pure-Python lexical ranking (term-frequency / inverse-document-
  frequency, Okapi BM25). No dependencies, always available. A large step up
  from grep: it ranks documents by how well they match ALL query terms, weighted
  by rarity, instead of returning the first lines containing a substring.

* Optional dense retrieval — when an `Embedder` is supplied, documents and the
  query are embedded and ranked by cosine similarity, catching paraphrase
  ("spouse" ~ "married to") that lexical matching misses.

* `HybridIndex` fuses the two with Reciprocal Rank Fusion (RRF) — the standard,
  score-scale-free way to combine a lexical and a dense ranker. With no embedder
  it degrades to pure BM25.

The index is built lazily over whatever documents the source exposes and cached
on the source; this in-memory store is deliberately behind the `search()` API so
a persistent/ANN backend can replace it later without changing the model-facing
primitive.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Tokenization: lowercase alphanumeric runs. A small stopword set keeps very
# common words from dominating BM25's term statistics. Deliberately tiny — over-
# aggressive stopwording hurts recall on short queries.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on at for and or but is are was were be been being this "
    "that these those it its as by with from into out about over under".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    """One ranked retrieval result. `doc` is the locator the source uses to
    fetch the full text (a file path for a Corpus, a document index for a
    Context). `span`, when set, is a char range within that document (a Context
    chunk). `snippet` is a short preview so the model can triage without reading
    the whole document."""

    doc: str | int
    score: float
    snippet: str
    span: tuple[int, int] | None = None

    def __repr__(self) -> str:  # readable when printed in the REPL
        where = f"{self.doc}" + (f"[{self.span[0]}:{self.span[1]}]" if self.span else "")
        return f"{where} (score {self.score:.3f}): {self.snippet.strip()[:200]}"


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors for dense retrieval. Implemented over any
    embeddings endpoint (see llm/embeddings.py). Optional: with no embedder,
    search is pure BM25."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def _snippet(text: str, query_tokens: set[str], width: int = 240) -> str:
    """A preview centered on the first query-term hit, so the snippet shows WHY
    the document matched rather than just its opening line."""
    low = text.lower()
    pos = -1
    for tok in query_tokens:
        p = low.find(tok)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos <= 0:
        return text[:width].strip()
    start = max(0, pos - width // 3)
    return ("…" if start > 0 else "") + text[start:start + width].strip()


class BM25Index:
    """Okapi BM25 over a fixed set of documents. Build once, query many times."""

    def __init__(self, docs: list[tuple[object, str]], *, k1: float = 1.5, b: float = 0.75):
        # docs: list of (locator, text). Locator is opaque (path / index / span).
        self.locators = [d[0] for d in docs]
        self.texts = [d[1] for d in docs]
        self.k1, self.b = k1, b
        self._tokens = [Counter(tokenize(t)) for t in self.texts]
        self._len = [sum(c.values()) for c in self._tokens]
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        n = len(self.texts)
        df: Counter = Counter()
        for c in self._tokens:
            df.update(c.keys())
        # BM25 idf with the standard +0.5 smoothing; floored at 0 so a term in
        # nearly every document can't drag a score negative.
        self._idf = {
            t: max(0.0, math.log((n - d + 0.5) / (d + 0.5) + 1.0))
            for t, d in df.items()
        }

    def rank(self, query: str) -> list[tuple[int, float]]:
        """Ranked (doc position, score), best first, scores > 0 only."""
        q = [t for t in tokenize(query) if t in self._idf]
        if not q or not self._avgdl:
            return []
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tokens):
            dl = self._len[i] or 1
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s += self._idf[t] * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class HybridIndex:
    """Lexical (BM25) + optional dense retrieval, fused with Reciprocal Rank
    Fusion. Built lazily by a Source and cached on it."""

    # RRF constant. 60 is the value from the original RRF paper; it damps the
    # influence of very high ranks so neither ranker dominates.
    _RRF_C = 60

    def __init__(self, docs: list[tuple[object, str]], *,
                 embedder: Embedder | None = None, embed_chars: int = 8000):
        self._docs = docs
        self._bm25 = BM25Index(docs)
        self._embedder = embedder
        self._embed_chars = embed_chars
        self._doc_vecs: list[list[float]] | None = None  # built on first dense use

    def _ensure_doc_vecs(self) -> None:
        if self._doc_vecs is None and self._embedder is not None:
            # Embed a bounded prefix of each doc to cap token cost; one batch call.
            self._doc_vecs = self._embedder.embed(
                [t[:self._embed_chars] for _, t in self._docs]
            )

    def _dense_rank(self, query: str) -> list[tuple[int, float]]:
        if self._embedder is None:
            return []
        self._ensure_doc_vecs()
        if not self._doc_vecs:
            return []
        qv = self._embedder.embed([query])[0]
        scored = [(i, _cosine(qv, dv)) for i, dv in enumerate(self._doc_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def warmup(self) -> None:
        """Force the lazy dense vectors to embed/load now (BM25 is already built
        in __init__). Lets a server pay this at startup, not on the first query."""
        if self._embedder is not None:
            self._ensure_doc_vecs()

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        """Top-`k` ranked hits. `mode`: 'lexical' (BM25 only), 'dense' (embeddings
        only — needs an embedder), or 'hybrid' (RRF fusion of both; falls back to
        lexical when no embedder is configured)."""
        lexical = self._bm25.rank(query)
        if mode == "lexical" or (mode == "hybrid" and self._embedder is None):
            ranked = lexical
        elif mode == "dense":
            ranked = self._dense_rank(query)
        else:  # hybrid: fuse the two rank lists with RRF
            ranked = self._rrf(lexical, self._dense_rank(query))
        qtok = set(tokenize(query))
        hits: list[Hit] = []
        for pos, score in ranked[:k]:
            loc, text = self._docs[pos]
            hits.append(Hit(doc=loc, score=score, snippet=_snippet(text, qtok)))
        return hits

    def _rrf(self, *rank_lists: list[tuple[int, float]]) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for rl in rank_lists:
            for rank, (pos, _score) in enumerate(rl):
                fused[pos] = fused.get(pos, 0.0) + 1.0 / (self._RRF_C + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)
