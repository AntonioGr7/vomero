"""Vomero — a Recursive Language Model (RLM) assistant over a data folder.

The big idea: instead of retrieving chunks and injecting them into the model's
context (RAG), the data lives as a `corpus` variable inside a Python REPL. The
model writes code to explore, grep and slice it, and delegates heavy reading to
recursive sub-model calls. Raw content never enters the root model's context.
"""

from .engine.rlm import RLMEngine
from .context.corpus import Corpus
from .config import Settings

__all__ = ["RLMEngine", "Corpus", "Settings"]
__version__ = "0.0.1"
