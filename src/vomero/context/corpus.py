"""Corpus: a lazy, read-only view over a data folder.

This is the object the model navigates from the REPL. Crucially it is *lazy* —
nothing is read into memory until the model asks. That is the whole point of
the RLM approach: the model decides what to look at, programmatically, instead
of having retrieved text shoved into its context.

Designed to be self-describing: the methods double as the model's API, and
`overview()` is what we show it up front.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .source import AccessLogged

if TYPE_CHECKING:  # annotations only; never imported at module load
    from .search import Embedder, Hit

# Extensions we treat as text-readable by default.
_TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".html", ".xml", ".sql", ".sh", ".log", ".env", ".tex",
}


@dataclass
class Match:
    path: str
    lineno: int
    line: str

    def __repr__(self) -> str:  # readable when printed in the REPL
        return f"{self.path}:{self.lineno}: {self.line.strip()[:200]}"


class Corpus(AccessLogged):
    """A read-only, lazy handle on a folder of data/documents.

    Satisfies the `Source` seam (see context/source.py), so the engine drives it
    interchangeably with an in-memory `Context`.

    Parameters
    ----------
    root:
        Folder to mount.
    allow:
        Optional explicit list of relative paths this view is restricted to
        (used by `subset` so recursive sub-calls can be scoped).
    """

    # The REPL variable name the engine injects this under (the `Source` seam).
    repl_name = "corpus"

    def __init__(self, root: str | Path, allow: list[str] | None = None,
                 *, embedder: "Embedder | None" = None,
                 index_dir: str | Path | None = None,
                 backend=None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Corpus root does not exist: {self.root}")
        self._allow = set(allow) if allow is not None else None
        # Optional dense-retrieval embedder for the LOCAL search backends; None
        # => BM25-only. Ignored when an external `backend` is supplied.
        self._embedder = embedder
        # A prebuilt persistent index dir (see `vomero index`). When set and
        # valid, search() opens it READ-ONLY — built once, loaded, never rebuilt
        # in the request path. None => the lazy in-memory index below.
        self._index_dir = Path(index_dir).expanduser().resolve() if index_dir else None
        # An explicit RetrievalBackend (context/retrieval.py). When given it wins
        # over the local index paths — e.g. a RemoteBackend so the vectors/index
        # live in an external service and this process holds no retrieval state.
        self._backend = backend
        # Lazily-resolved search backend, cached per view (a subset rebuilds).
        self._index = None

    # -- discovery -------------------------------------------------------
    def files(self, glob: str = "**/*", text_only: bool = True) -> list[str]:
        """Relative paths of files matching `glob` (sorted)."""
        out = []
        for p in sorted(self.root.glob(glob)):
            if not p.is_file():
                continue
            rel = p.relative_to(self.root).as_posix()
            if self._allow is not None and rel not in self._allow:
                continue
            if text_only and p.suffix.lower() not in _TEXT_EXT:
                continue
            out.append(rel)
        return out

    def tree(self, max_entries: int = 500) -> str:
        """An indented file tree, for getting your bearings."""
        lines = [self.root.name + "/"]
        paths = self.files(text_only=False)[:max_entries]
        for rel in paths:
            depth = rel.count("/")
            lines.append("  " * (depth + 1) + Path(rel).name)
        if len(self.files(text_only=False)) > max_entries:
            lines.append(f"  ... (truncated at {max_entries} entries)")
        return "\n".join(lines)

    # -- reading ---------------------------------------------------------
    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"Path escapes corpus root: {path}")
        if self._allow is not None and Path(path).as_posix() not in self._allow:
            raise ValueError(f"Path not in this corpus view: {path}")
        return p

    def read(self, path: str, encoding: str = "utf-8") -> str:
        """Full text of one file. Use sparingly on big files — prefer peek/grep
        or delegate to `llm()`/`rlm()`."""
        text = self._resolve(path).read_text(encoding=encoding, errors="replace")
        self._record("read", path)
        return text

    def peek(self, path: str, lines: int = 40) -> str:
        """First `lines` lines of a file."""
        # Read directly (not via self.read) so this logs a single "peek", not
        # an extra "read".
        text = self._resolve(path).read_text(encoding="utf-8", errors="replace")
        head = text.splitlines()[:lines]
        self._record("peek", path)
        return "\n".join(head)

    def size(self, path: str) -> int:
        """File size in bytes."""
        return self._resolve(path).stat().st_size

    # -- search ----------------------------------------------------------
    def grep(
        self,
        pattern: str,
        glob: str = "**/*",
        ignore_case: bool = True,
        max_results: int = 200,
    ) -> list[Match]:
        """Regex search across files. Returns Match(path, lineno, line)."""
        flags = re.IGNORECASE if ignore_case else 0
        rx = re.compile(pattern, flags)
        results: list[Match] = []
        for rel in self.files(glob=glob):
            try:
                # Read raw (not self.read) — scanning a file during grep is not
                # a model "read"; only the matching lines are logged, below.
                text = self._resolve(rel).read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    results.append(Match(rel, i, line))
                    self._record("grep", rel, i, line)
                    if len(results) >= max_results:
                        return results
        return results

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        """Ranked relevance search — the recall-friendly alternative to grep.

        Returns the top-`k` files ranked by how well they match the WHOLE query
        (BM25, plus dense embeddings when configured), not just lines containing
        a literal substring. Use it when the phrasing is uncertain or you want
        the most relevant documents rather than every exact hit; follow up with
        read()/peek() on the hits. `mode` is 'lexical', 'dense', or 'hybrid'."""
        if self._index is None:
            self._index = self._open_index()
        hits = self._index.search(query, k=k, mode=mode)
        for h in hits:
            self._record("search", h.doc, text=h.snippet)
        return hits

    def _open_index(self):
        """Resolve the search backend, in precedence order: an explicitly
        injected `backend` (e.g. a RemoteBackend → external service); else a
        prebuilt persistent index opened read-only when `index_dir` is present;
        else a lazy in-memory index built from this view's files. Imported lazily
        so the module loads even where search.py isn't importable (the sandbox
        loads corpus.py standalone; there search() is delegated to the host)."""
        if self._backend is not None:
            return self._backend

        from .search import HybridIndex

        if self._index_dir is not None:
            from .index import PersistentIndex

            if PersistentIndex.exists(self._index_dir):
                return PersistentIndex(self._index_dir, embedder=self._embedder)
        # Read raw (not self.read) — building the index is not a model "read";
        # only the returned hits are logged as retrieval, by the caller.
        docs = []
        for rel in self.files():
            try:
                docs.append((rel, self._resolve(rel).read_text(
                    encoding="utf-8", errors="replace")))
            except Exception:
                continue
        return HybridIndex(docs, embedder=self._embedder)

    # -- scoping ---------------------------------------------------------
    def subset(self, paths: list[str]) -> "Corpus":
        """A new Corpus view restricted to `paths` (for scoped recursion)."""
        # A subset only sees `paths`, so it builds its own in-memory index over
        # them rather than the whole-corpus persistent index (which would return
        # out-of-scope docs); index_dir is intentionally not propagated.
        sub = Corpus(self.root, allow=list(paths), embedder=self._embedder)
        # Share the access log so a scoped recursive sub-call's retrieval lands
        # in the same provenance record as the root's.
        sub._access = self._access
        return sub

    # -- self-description ------------------------------------------------
    def overview(self, max_files: int = 40) -> str:
        files = self.files()
        n = len(files)
        shown = files[:max_files]
        body = "\n".join(f"  - {f}" for f in shown)
        more = f"\n  ... and {n - max_files} more" if n > max_files else ""
        return f"Corpus at {self.root} — {n} text file(s):\n{body}{more}"

    # -- Source seam (system-prompt surface) -----------------------------
    def guide(self) -> str:
        """The methods block the engine splices into the system prompt."""
        return (
            "  corpus        A read-only handle on the data folder. Key methods:\n"
            "                  corpus.overview()           -> summary + file list (start here)\n"
            "                  corpus.tree()               -> file tree\n"
            "                  corpus.files(glob=\"**/*\")   -> list of relative paths\n"
            "                  corpus.grep(pattern, ...)   -> regex search -> [Match(path, lineno, line)]\n"
            "                  corpus.search(query, k=10)  -> ranked-relevance search -> [Hit(doc, score, snippet)]\n"
            "                  corpus.peek(path, lines=40) -> first lines of a file\n"
            "                  corpus.read(path)           -> full text of a file\n"
            "                  corpus.size(path)           -> bytes\n"
            "                  corpus.subset([paths])      -> a corpus scoped to those files\n"
        )

    def start_hint(self) -> str:
        return "corpus.overview()"

    def __repr__(self) -> str:
        scope = "" if self._allow is None else f", scoped to {len(self._allow)} files"
        return f"<Corpus {self.root}{scope}>"
