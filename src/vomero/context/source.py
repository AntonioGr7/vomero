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

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class AccessEvent:
    """One retrieval the model made against the source — structured provenance.

    The model navigates the source from the REPL; the engine only ever sees the
    printed stdout of that navigation. Downstream consumers (e.g. grounding a
    cited answer back to its source spans) would otherwise have to reverse-
    engineer which doc/line a grep hit came from out of that stdout. An access
    log records it directly instead.

    `doc` identifies the document (a relative path for a `Corpus`, an index for
    a `Context`). `lineno`/`text` carry the located line for a `grep` hit; both
    are None for a whole-document `read`/`peek`, where the consumer already has
    the document and only needs to know it was touched.
    """

    op: str  # "grep" | "read" | "peek"
    doc: str | int
    lineno: int | None = None
    text: str | None = None


class AccessLogged:
    """Mixin giving a `Source` an opt-in access log.

    Off by default: standalone use records nothing and behaves exactly as before
    (no overhead, retrieval methods return identical results). The engine calls
    `enable_access_log()` at the start of a run that wants provenance, and
    `subset()` shares the same list so recursively-scoped sub-views record into
    the one log — the whole run's retrieval lands in a single place.
    """

    _access: list[AccessEvent] | None = None

    def enable_access_log(self) -> list[AccessEvent]:
        """Start recording (idempotent); returns the live log list."""
        if self._access is None:
            self._access = []
        return self._access

    def reset_access_log(self) -> None:
        """Clear the log in place (keeps it enabled). A `Source` is often shared
        across many runs (e.g. one Corpus reused for every eval item); resetting
        between runs isolates each run's provenance instead of accumulating it.
        Clears in place so `subset()` views sharing the list see the reset too."""
        if self._access is not None:
            self._access.clear()

    def _record(self, op: str, doc: str | int,
                lineno: int | None = None, text: str | None = None) -> None:
        if self._access is not None:
            self._access.append(AccessEvent(op, doc, lineno, text))

    @property
    def access_log(self) -> list[AccessEvent]:
        """A snapshot of what the model has retrieved so far (empty if logging
        was never enabled)."""
        return list(self._access) if self._access is not None else []


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
