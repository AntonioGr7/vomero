"""The retrieval seam: `RetrievalBackend` — where search() actually runs.

`Source.search()` doesn't implement ranking itself; it delegates to a backend
satisfying this one-method protocol. That keeps the model-facing primitive fixed
while the *where* and *how* of retrieval varies by deployment:

* `HybridIndex` (search.py) — in-memory, built per process. Zero setup; fine to
  tens of thousands of docs, single-tenant.
* `PersistentIndex` (index.py) — built once, opened read-only. Both of the above
  already satisfy `RetrievalBackend`, so they ARE backends, no wrapper needed.
* `RemoteBackend` (here) — delegates to an EXTERNAL retrieval service over HTTP.
  The vectors, the ANN index, and the query-embedding all live in the service;
  the Vomero process holds nothing but this client. This is the answer to
  "millions of docs, many tenants, different embedding models": none of that
  state sits in the serving pod, so its memory stays flat and it remains
  stateless — the same principle that keeps the engine stateless. The embedding
  model is a property of the service's index (built offline), not of the pod.

Under the gVisor sandbox, search() is RPC-delegated to the host (the sandbox has
no network); the host then calls whichever backend is configured. With a
`RemoteBackend`, the host is a thin proxy: sandbox -> host -> retrieval service.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol, runtime_checkable

from .search import Hit


@runtime_checkable
class RetrievalBackend(Protocol):
    """Anything that can rank documents for a query. `HybridIndex` and
    `PersistentIndex` conform as-is; `RemoteBackend` calls out to a service."""

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        ...


class RemoteBackend:
    """Delegates search to an external HTTP retrieval service.

    Store-agnostic by design: it speaks a tiny JSON contract you can put in front
    of any vector DB / search service (Qdrant, Weaviate, Elasticsearch, pgvector,
    a managed API) with a thin adapter, or subclass `_request`/`_parse` to match
    an existing API directly.

      POST <url>   {"query": str, "k": int, "mode": str, "collection": str|null}
      -> 200       {"hits": [{"doc": str|int, "score": float,
                               "snippet": str, "span": [int,int]|null}, ...]}

    The service owns the index AND the query-embedding (it knows which model
    built the index), so nothing model- or vector-shaped lives in this process.
    """

    def __init__(self, url: str, *, collection: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0):
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        payload = self._request_body(query, k, mode)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return self._parse(body)

    # -- override these two to adapt to a service with a different shape -------
    def _request_body(self, query: str, k: int, mode: str) -> dict:
        return {"query": query, "k": k, "mode": mode, "collection": self.collection}

    def _parse(self, body: dict) -> list[Hit]:
        hits: list[Hit] = []
        for h in body.get("hits", []):
            span = h.get("span")
            hits.append(Hit(doc=h["doc"], score=float(h.get("score", 0.0)),
                            snippet=h.get("snippet", ""),
                            span=tuple(span) if span else None))
        return hits


def build_retrieval_backend(settings) -> RetrievalBackend | None:
    """Factory: a `RemoteBackend` when `retrieval_url` is configured, else None
    (the Source builds a local index from `index_dir`/in-memory). Duck-typed on
    settings so it stays decoupled from the config module."""
    url = getattr(settings, "retrieval_url", "") or ""
    if not url:
        return None
    return RemoteBackend(
        url,
        collection=getattr(settings, "retrieval_collection", None) or None,
        api_key=getattr(settings, "retrieval_api_key", None) or None,
    )
