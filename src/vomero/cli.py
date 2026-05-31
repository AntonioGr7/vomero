"""Vomero command-line entry point.

Usage:
  vomero ask "your question" --data ./path/to/folder [-v]
  vomero ask "..." --data ./folder --model gpt-4o

Reads model/provider/credentials from the environment (.env supported):
  VOMERO_PROVIDER (default: openai)
  VOMERO_MODEL    (default: gpt-4o-mini)
  VOMERO_BASE_URL / OPENAI_BASE_URL   (for OpenAI-compatible servers)
  VOMERO_API_KEY  / OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .context.corpus import Corpus
from .engine import Compactor, RLMEngine
from .engine.rlm import Step
from .llm import build_client


def _verbose_printer():
    def emit(step: Step) -> None:
        pad = "  " * step.depth
        tag = f"{pad}[d{step.depth}.{step.index}]"
        if step.compaction is not None:
            c = step.compaction
            print(
                f"{tag} ⟳ compacted {c.summarized_messages} msg(s): "
                f"~{c.tokens_before:,} → ~{c.tokens_after:,} tok "
                f"({c.messages_before} → {c.messages_after} msgs)",
                file=sys.stderr,
            )
        elif step.usage is not None:
            u = step.usage
            ctx_approx = "~" if u.context_estimated else ""
            tot_approx = "~" if u.cumulative_estimated else ""
            print(
                f"{tag} ctx {ctx_approx}{u.context_tokens:,} tok | total {tot_approx}{u.cumulative_tokens:,} tok",
                file=sys.stderr,
            )
        elif step.code is not None:
            print(f"\n{tag} python:\n{pad}  " + step.code.replace("\n", "\n" + pad + "  "),
                  file=sys.stderr)
        elif step.output is not None:
            snippet = step.output if len(step.output) < 1500 else step.output[:1500] + " …[truncated]"
            print(f"{tag} -> " + snippet.replace("\n", "\n" + pad + "     "), file=sys.stderr)
        elif step.final is not None:
            print(f"{tag} FINAL (depth {step.depth})", file=sys.stderr)

    return emit


def cmd_ask(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    if args.max_depth is not None:
        settings.max_depth = args.max_depth
    if args.max_steps is not None:
        settings.max_steps = args.max_steps
    if args.context_window is not None:
        settings.context_window = args.context_window
    if args.compact_ratio is not None:
        settings.compact_ratio = args.compact_ratio
    if args.no_compact:
        settings.compact_ratio = 0.0

    try:
        corpus = Corpus(args.data)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    compactor = None
    if settings.compact_ratio > 0:
        compactor = Compactor(
            context_window=settings.context_window,
            ratio=settings.compact_ratio,
            keep_recent_messages=settings.compact_keep_recent,
            min_reclaim_tokens=settings.compact_min_reclaim,
        )

    client = build_client(settings)
    engine = RLMEngine(
        client,
        model=settings.model,
        max_steps=settings.max_steps,
        max_depth=settings.max_depth,
        compactor=compactor,
    )

    on_event = _verbose_printer() if args.verbose else None
    answer = engine.run(args.question, corpus, on_event=on_event)
    if args.verbose:
        print("\n" + "=" * 60, file=sys.stderr)
    print(answer)

    # Always report token usage on stderr (keeps stdout to just the answer).
    u = engine.last_usage
    if u is not None:
        approx = "~" if u.estimated else ""
        print(
            f"[usage] {u.calls} model call(s) · {approx}{u.total_tokens:,} tokens total "
            f"({approx}{u.prompt_tokens:,} in / {approx}{u.completion_tokens:,} out)"
            + ("  (estimated — provider did not report usage)" if u.estimated else ""),
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vomero", description="Recursive LM assistant over a data folder.")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question about a data folder.")
    ask.add_argument("question", help="The question to answer.")
    ask.add_argument("--data", required=True, help="Path to the data folder (the corpus).")
    ask.add_argument("--model", default=None, help="Override the model name.")
    ask.add_argument("--max-depth", type=int, default=None, help="Max recursion depth.")
    ask.add_argument("--max-steps", type=int, default=None, help="Max REPL steps per level.")
    ask.add_argument("--context-window", type=int, default=None,
                     help="Model context window in tokens (compaction threshold = ratio * this).")
    ask.add_argument("--compact-ratio", type=float, default=None,
                     help="Compact when context reaches this fraction of the window (default 0.8).")
    ask.add_argument("--no-compact", action="store_true", help="Disable history compaction.")
    ask.add_argument("-v", "--verbose", action="store_true", help="Stream the model's code/output to stderr.")
    ask.set_defaults(func=cmd_ask)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
