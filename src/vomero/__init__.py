"""Vomero — a Recursive Language Model (RLM) assistant over your data.

The big idea: instead of retrieving chunks and injecting them into the model's
context (RAG), the data lives as a variable inside a Python REPL. The model
writes code to explore, grep and slice it, and delegates heavy reading to
recursive sub-model calls. Raw content never enters the root model's context.

Use it from Python in three lines:

    from vomero import build_engine, Context
    engine = build_engine(model="gpt-4o-mini")
    print(engine.run("What are the key risks?", Context(open("contract.txt").read())))

See docs/library.md for the full programmatic guide.
"""

from __future__ import annotations

from .config import Settings
from .context import Context, Corpus
from .engine.rlm import RLMEngine

__all__ = ["RLMEngine", "Corpus", "Context", "Settings", "build_engine"]
__version__ = "0.0.1"


def build_engine(settings: Settings | None = None, **overrides):
    """Build a ready-to-use `RLMEngine` from `Settings` in one call.

    The convenience constructor the CLI/server assemble by hand: it wires the
    provider client, the execution backend (in-process or gVisor sandbox), and
    history compaction from config, so callers don't have to. `settings` defaults
    to `Settings.from_env()`; pass keyword `overrides` to tweak any field without
    mutating the original, e.g. ``build_engine(model="gpt-4o", max_depth=2)``.

    The engine holds no per-run state — build one and reuse it across threads.
    """
    from dataclasses import replace

    from .engine import Compactor, RLMEngine
    from .execution import build_env_factory
    from .llm import build_client

    settings = settings or Settings.from_env()
    if overrides:
        settings = replace(settings, **overrides)

    compactor = (
        Compactor(
            context_window=settings.context_window,
            ratio=settings.compact_ratio,
            keep_recent_messages=settings.compact_keep_recent,
            min_reclaim_tokens=settings.compact_min_reclaim,
        )
        if settings.compact_ratio > 0
        else None
    )
    return RLMEngine(
        build_client(settings),
        env_factory=build_env_factory(settings),
        model=settings.model,
        max_steps=settings.max_steps,
        max_depth=settings.max_depth,
        max_output_chars=settings.max_output_chars,
        max_parallel_calls=settings.max_parallel_calls,
        compactor=compactor,
        enable_planning=settings.enable_planning,
        planning_root_only=settings.planning_root_only,
        enable_interaction=settings.enable_interaction,
        interaction_root_only=settings.interaction_root_only,
    )
