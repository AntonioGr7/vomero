#!/usr/bin/env python3
"""Scratch script to run the real RLM engine from the shell.

Edit the QUESTION (and FOLLOW_UPS, and the knobs) below, then:

    uv run python test_loop.py

It builds the engine from your environment (model, sandbox, planning, etc. —
same settings the CLI/server read) and prints a trace of every step. Use a real
API key in the environment, just like `vomero ask`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from vomero.config import Settings
from vomero.context.corpus import Corpus
from vomero.engine import Compactor, RLMEngine
from vomero.execution import build_env_factory, build_session_pool
from vomero.llm import build_client
from vomero.llm.base import Message
from vomero.usage import UsageMeter

# ---------------------------------------------------------------------------
# EDIT HERE
# ---------------------------------------------------------------------------

# The question to ask.
QUESTION = """
what are the % of ebita margin yoy for northwind
""".strip()

# Optional follow-up questions. Each one is asked AFTER the previous, carrying
# the prior conversation as context (the multi-turn history seam). Leave empty
# for a single-shot run.
FOLLOW_UPS: list[str] = [
    # "Why does that block it?",
    # "Who owns the blocker?",
]

# Corpus the engine reasons over.
CORPUS_PATH = Path(__file__).resolve().parent / "data" / "finance" / "triarch-2024-2026"

# Per-run feature toggles (override the environment defaults; None = use env).
ENABLE_PLANNING = True       # True/False to force the TODO surface on/off
ENABLE_INTERACTION = True    # True/False to allow ask_user (answered on stdin)

# Persist the SAME execution environment across QUESTION + FOLLOW_UPS, so the
# model's REPL variables (and, with a workspace, its files) carry over — a
# follow-up can reuse a variable an earlier turn defined. Set False to give each
# turn a fresh env (variables reset every turn).
PERSIST_SESSION = True
# Durable workspace dir for the session (sandbox backend only). None => none.
# Set to a host dir to persist files the model writes (its cwd is /workspace in
# the sandbox); they land under <WORKSPACE_ROOT>/<session-key>/. Here: a
# gitignored folder so outputs show up in the IDE without being committed.
WORKSPACE_ROOT: str | None = str(Path(__file__).resolve().parent / "data" / "workspaces")


# ---------------------------------------------------------------------------
# Trace printer — shows each Step the engine emits.
# ---------------------------------------------------------------------------

class TraceChannel:
    # Marks for the three TODO states, shared by the live and per-turn renders.
    _MARKS = {"pending": "○", "in_progress": "◐", "completed": "●"}

    def __init__(self) -> None:
        # Latest plan snapshot, per depth, so we can reprint it at the start of
        # every turn (Claude-style) instead of only when the model mutates it.
        # The engine emits a `todo` event only on mutation, so without this the
        # checklist vanishes on any turn the model doesn't touch the surface.
        self._todo: dict[int, list] = {}
        # Last rendered (text, status) per depth, to skip identical reprints.
        self._last_rendered: dict[int, tuple] = {}

    def _render_todo(self, depth: int, dedup: bool = False) -> None:
        # dedup=False (turn start): always print, so the plan recurs every turn.
        # dedup=True (final): print only if it changed since the last render, so
        # we surface last-second ticks without duplicating an unchanged list.
        items = self._todo.get(depth)
        if not items:
            return
        snapshot = tuple((t.text, t.status) for t in items)
        if dedup and self._last_rendered.get(depth) == snapshot:
            return
        self._last_rendered[depth] = snapshot
        d = "  " * depth
        print(f"{d}      📋 " + "  ".join(f"{self._MARKS.get(t.status, '?')} {t.text}"
                                         for t in items))

    def emit(self, step) -> None:
        d = "  " * step.depth
        tag = f"{d}[d{step.depth}.s{step.index}]"
        if step.usage is not None:
            u = step.usage
            print(f"{tag} 📊 context={u.context_tokens} · cumulative={u.cumulative_tokens} tok")
            # Reprint the current plan at the top of each turn so it's always on
            # screen with up-to-date marks, even when the model didn't touch it.
            self._render_todo(step.depth)
        if step.message:
            print(f"{tag} 💬 {step.message}")
        if step.todo is not None:
            # Cache for per-turn reprints, and show the live change as it happens.
            self._todo[step.depth] = step.todo
            self._last_rendered[step.depth] = tuple(
                (t.text, t.status) for t in step.todo
            )
            print(f"{tag} 📋 " + "  ".join(f"{self._MARKS.get(t.status, '?')} {t.text}"
                                          for t in step.todo))
        if step.code is not None:
            print(f"{tag} 🐍")
            for line in step.code.splitlines():
                print(f"{d}      | {line}")
        if step.output is not None:
            print(f"{tag} ↳")
            for line in step.output.splitlines():
                print(f"{d}      {line}")
        if step.llm_call is not None:
            c = step.llm_call
            print(f"{tag} 🔹 llm(…) -> {c.response[:60]!r} (+{c.tokens} tok)")
        if step.interaction is not None:
            it = step.interaction
            print(f"{tag} 🙋 ask_{it.kind}: {it.question!r} -> {it.answer!r}")
        if step.compaction is not None:
            c = step.compaction
            print(f"{tag} 🗜  {c.messages_before}→{c.messages_after} msgs, "
                  f"~{c.tokens_before}→{c.tokens_after} tok")
        if step.final is not None:
            self._render_todo(step.depth, dedup=True)  # surface last-second ticks
            print(f"{tag} ✅ FINAL: {step.final}")

    def ask_user(self, question: str) -> str:
        if sys.stdin.isatty():
            return input(f"      ask_user> {question}\n      > ")
        return "No user available; proceed with your best judgment."


# ---------------------------------------------------------------------------

def build_engine(settings: Settings) -> RLMEngine:
    compactor = None
    if settings.compact_ratio > 0:
        compactor = Compactor(
            context_window=settings.context_window,
            ratio=settings.compact_ratio,
            keep_recent_messages=settings.compact_keep_recent,
            min_reclaim_tokens=settings.compact_min_reclaim,
        )
    return RLMEngine(
        build_client(settings),
        env_factory=build_env_factory(settings),
        model=settings.model,
        max_steps=settings.max_steps,
        max_depth=settings.max_depth,
        compactor=compactor,
        enable_planning=settings.enable_planning,
        enable_interaction=settings.enable_interaction,
    )


def main() -> int:
    settings = Settings.from_env()
    print(f"[model] {settings.model}  ·  [backend] {settings.exec_backend}  ·  "
          f"[corpus] {CORPUS_PATH}")
    engine = build_engine(settings)
    corpus = Corpus(CORPUS_PATH)
    channel = TraceChannel()
    meter = UsageMeter()

    run_kwargs = {}
    if ENABLE_PLANNING is not None:
        run_kwargs["enable_planning"] = ENABLE_PLANNING
    if ENABLE_INTERACTION is not None:
        engine.enable_interaction = ENABLE_INTERACTION

    # When persisting, reuse one env (the model's variables/workspace survive
    # across turns); otherwise each turn gets a fresh env from the engine.
    pool = None
    key = ("local", "scratch")
    if PERSIST_SESSION:
        pool = build_session_pool(settings, ttl_seconds=settings.session_ttl,
                                  workspace_root=WORKSPACE_ROOT)
        print(f"[session] persisting variables across turns "
              f"(ttl={settings.session_ttl:.0f}s, workspace={WORKSPACE_ROOT})")

    history: list[Message] = []
    for question in [QUESTION, *FOLLOW_UPS]:
        print(f"\n──── QUESTION: {question!r} ────")
        transcript: list[Message] = []
        if pool is not None:
            with pool.session(key) as env:
                answer = engine.run(
                    question, corpus, channel=channel, meter=meter,
                    history=history, transcript_sink=transcript, env=env, **run_kwargs,
                )
        else:
            answer = engine.run(
                question, corpus, channel=channel, meter=meter,
                history=history, transcript_sink=transcript, **run_kwargs,
            )
        print(f"──── ANSWER: {answer!r} ────")
        history = transcript  # carry context into the next follow-up
        

    if pool is not None:
        pool.close_all()
    print(f"\n[usage] {meter.calls} call(s) · {meter.total_tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
