"""Context: an in-memory blob, mounted as a navigable REPL variable.

This is the canonical Recursive-Language-Model surface (Zhang/Kraska/Khattab,
2025): the *context itself* lives as a Python variable the model explores
programmatically — grep/slice/chunk — instead of being pasted into the model's
own token window. It is the in-memory sibling of `Corpus` (which mounts a
folder); both satisfy the same `Source` seam the engine drives, so everything
built on the engine (recursion, compaction, planning, sandbox) works unchanged.

A `Context` wraps either a single string (one long document — the OOLONG /
long-prompt case) or a list of strings ("documents" — the multi-doc / retrieval
case). The model sees only metadata + a short preview via `overview()`, then
pulls in only what it needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .source import AccessLogged

# Separator used when the documents are joined into one addressable text (for
# slice/chunk). Kept distinctive so it doesn't collide with document content.
_JOIN = "\n\n"


@dataclass
class Match:
    """A grep hit. `doc` is the document index (always 0 for a single string)."""

    doc: int
    lineno: int
    line: str

    def __repr__(self) -> str:  # readable when printed in the REPL
        return f"[doc {self.doc}] line {self.lineno}: {self.line.strip()[:200]}"


class Context(AccessLogged):
    """A read-only, lazy handle on an in-memory blob.

    Parameters
    ----------
    data:
        A single string, or a list of strings (documents).
    name:
        The REPL variable name (default ``"context"``); also what `subset`
        carries forward so recursive sub-calls describe themselves consistently.
    """

    repl_name: str

    def __init__(self, data: str | list[str], *, name: str = "context"):
        if isinstance(data, str):
            self._docs: list[str] = [data]
            self._single = True
        else:
            self._docs = [str(d) for d in data]
            self._single = False
        self.repl_name = name

    # -- sizing ----------------------------------------------------------
    @property
    def n_docs(self) -> int:
        """Number of documents (1 for a single string)."""
        return len(self._docs)

    @property
    def chars(self) -> int:
        """Total character count across all documents."""
        return sum(len(d) for d in self._docs)

    def __len__(self) -> int:
        """Total characters — the 'how big is this' figure. Use `n_docs` for the
        document count."""
        return self.chars

    @property
    def text(self) -> str:
        """All documents joined into one addressable string (for slice/chunk).
        Materializes the whole blob — prefer grep/peek on large data."""
        return self._docs[0] if self._single else _JOIN.join(self._docs)

    # -- reading ---------------------------------------------------------
    def read(self, doc: int | None = None) -> str:
        """Full text of document `doc`. For a single-string context, `doc` may
        be omitted. For a multi-doc context an index is required (reading them
        all at once would defeat the point — chunk/grep instead)."""
        if doc is None:
            if self._single:
                self._record("read", 0)
                return self._docs[0]
            raise ValueError(
                f"This context has {self.n_docs} documents; pass an index, "
                "e.g. context.read(0). To scan them all, use grep/chunk."
            )
        self._record("read", doc)
        return self._docs[doc]

    def peek(self, doc: int = 0, lines: int = 40) -> str:
        """First `lines` lines of document `doc`."""
        self._record("peek", doc)
        return "\n".join(self._docs[doc].splitlines()[:lines])

    def slice(self, start: int, end: int) -> str:
        """A character slice ``[start:end]`` of the joined text."""
        return self.text[start:end]

    # -- search ----------------------------------------------------------
    def grep(
        self,
        pattern: str,
        ignore_case: bool = True,
        max_results: int = 200,
    ) -> list[Match]:
        """Regex search across all documents. Returns Match(doc, lineno, line)."""
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern, flags)
        out: list[Match] = []
        for di, text in enumerate(self._docs):
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    out.append(Match(di, i, line))
                    self._record("grep", di, i, line)
                    if len(out) >= max_results:
                        return out
        return out

    def docs_matching(self, pattern: str, ignore_case: bool = True) -> list[int]:
        """Indices of documents whose text matches `pattern` — pair with
        `subset(...)` to scope a recursive call to just the relevant documents."""
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern, flags)
        return [i for i, d in enumerate(self._docs) if rx.search(d)]

    # -- chunking (for partition + map over sub-LM calls) ----------------
    def chunk(self, size: int, overlap: int = 0) -> list[str]:
        """Split the joined text into ~`size`-character chunks (optionally
        overlapping). The standard partition step before mapping `llm()` over
        the pieces."""
        if size <= 0:
            raise ValueError("chunk size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be in [0, size)")
        text = self.text
        step = size - overlap
        return [text[i : i + size] for i in range(0, len(text), step)] or [""]

    # -- scoping ---------------------------------------------------------
    def subset(self, selector: list[int] | tuple[int, int]) -> "Context":
        """A new Context scoped for recursion. `selector` is either a list of
        document indices, or a ``(start, end)`` character range over the joined
        text."""
        if isinstance(selector, tuple):
            start, end = selector
            sub = Context(self.text[start:end], name=self.repl_name)
        else:
            sub = Context([self._docs[i] for i in selector], name=self.repl_name)
        # Share the access log so a scoped recursive sub-call records into the
        # same provenance record as the root's.
        sub._access = self._access
        return sub

    # -- Source seam -----------------------------------------------------
    def overview(self, preview_chars: int = 800) -> str:
        kind = "one document" if self._single else f"{self.n_docs} documents"
        preview = self.text[:preview_chars]
        ell = " …" if self.chars > preview_chars else ""
        return (
            f"Context held in memory — {kind}, {self.chars:,} characters total.\n"
            f"Preview (first {len(preview):,} chars):\n{preview}{ell}"
        )

    def guide(self) -> str:
        n = self.repl_name
        return (
            f"  {n}        A read-only handle on the data, held in memory (NOT in your\n"
            f"                context window). It is a string or list of documents. Key methods:\n"
            f"                  {n}.overview()            -> size, doc count + a preview (start here)\n"
            f"                  len({n})                  -> total characters\n"
            f"                  {n}.n_docs                -> number of documents\n"
            f"                  {n}.peek(doc=0, lines=40) -> first lines of a document\n"
            f"                  {n}.read(doc)             -> full text of one document\n"
            f"                  {n}.slice(start, end)     -> a character slice of the data\n"
            f"                  {n}.grep(pattern, ...)    -> regex search -> [Match(doc, lineno, line)]\n"
            f"                  {n}.docs_matching(pat)    -> indices of documents that match (to scope rlm)\n"
            f"                  {n}.chunk(size, overlap=0)-> split into char chunks (for map/reduce over llm())\n"
            f"                  {n}.subset(sel)           -> scope to doc indices [i, j] or a (start, end) char range\n"
        )

    def start_hint(self) -> str:
        return f"{self.repl_name}.overview()"

    def __repr__(self) -> str:
        kind = "1 doc" if self._single else f"{self.n_docs} docs"
        return f"<Context {kind}, {self.chars:,} chars>"
