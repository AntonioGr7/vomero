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


class Corpus:
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

    def __init__(self, root: str | Path, allow: list[str] | None = None):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Corpus root does not exist: {self.root}")
        self._allow = set(allow) if allow is not None else None

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
        return self._resolve(path).read_text(encoding=encoding, errors="replace")

    def peek(self, path: str, lines: int = 40) -> str:
        """First `lines` lines of a file."""
        text = self.read(path)
        head = text.splitlines()[:lines]
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
                text = self.read(rel)
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    results.append(Match(rel, i, line))
                    if len(results) >= max_results:
                        return results
        return results

    # -- scoping ---------------------------------------------------------
    def subset(self, paths: list[str]) -> "Corpus":
        """A new Corpus view restricted to `paths` (for scoped recursion)."""
        return Corpus(self.root, allow=list(paths))

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
