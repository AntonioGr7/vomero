"""The `Source` seam: a navigable data handle the model drives from the REPL.

Both kinds of data the engine can mount satisfy this protocol:

* `Corpus` — a folder of files on disk (agentic-RAG over a directory).
* `Context` — an in-memory blob (a string or list of strings) held as a REPL
  variable. This is the canonical Recursive-Language-Model surface: the prompt/
  context *itself* lives as a variable the model explores programmatically,
  never pasted into its own token window.

The engine depends only on this protocol — it injects the source into the REPL
under `repl_name`, builds the system prompt from `guide()`/`start_hint()`, and
scopes recursive `rlm()` sub-calls via `subset()`. New source kinds (a SQL
table, a parquet file, an HTTP API) slot in by implementing these four members.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Source(Protocol):
    """A read-only, lazily-navigated handle on a body of data."""

    # The variable name the source is injected under in the REPL ("corpus"/"context").
    repl_name: str

    def overview(self) -> str:
        """A short, self-describing summary (size + a preview), shown up front.
        Must NOT dump the full data — that defeats the whole approach."""
        ...

    def guide(self) -> str:
        """The methods block for this source, spliced into the system prompt so
        the model knows the navigation API available on `repl_name`."""
        ...

    def start_hint(self) -> str:
        """The first call the model should make, e.g. ``corpus.overview()``."""
        ...

    def subset(self, selector: Any) -> "Source":
        """A new source scoped by `selector`, for scoped recursive `rlm()` calls.
        The selector's meaning is source-specific (file paths, doc indices, a
        char range)."""
        ...
