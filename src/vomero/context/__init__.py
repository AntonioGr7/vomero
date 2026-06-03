"""The data the model navigates, mounted as a variable (not injected as text).

Two interchangeable sources behind one `Source` seam:

* `Corpus`  — a folder of files on disk.
* `Context` — an in-memory blob (string or list of strings) held as a REPL
  variable; the canonical Recursive-Language-Model surface.
"""

from __future__ import annotations

from .corpus import Corpus
from .source import Source
from .variable import Context

__all__ = ["Corpus", "Context", "Source"]
