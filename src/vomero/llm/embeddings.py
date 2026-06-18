"""Dense-retrieval embedder over an OpenAI-compatible embeddings endpoint.

Satisfies the `context.search.Embedder` protocol (`embed(texts) -> vectors`).
Provider-agnostic via `base_url`, exactly like the chat client — point it at
OpenAI, a local server (vLLM/LM Studio), or any compatible API. Used only when
`Settings.embedding_model` is set; otherwise search() stays pure-BM25 and no
embedding calls are made.
"""

from __future__ import annotations

from openai import OpenAI


class OpenAIEmbedder:
    """Embeds text with an OpenAI-compatible `/embeddings` endpoint, batched."""

    def __init__(self, model: str, *, base_url: str | None = None,
                 api_key: str | None = None, batch_size: int = 256):
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # The API rejects empty strings; send a single space placeholder so
            # indices stay aligned with the input list.
            resp = self._client.embeddings.create(
                model=self.model, input=[t or " " for t in batch]
            )
            out.extend(d.embedding for d in resp.data)
        return out
