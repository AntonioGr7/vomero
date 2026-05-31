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


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and truncate, for one-line previews."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …"


def _verbose_printer():
    # Bind the real stderr now. `llm()`/`rlm()` events fire from inside
    # `env.execute()`, which redirects sys.stderr to capture the model's own
    # output — so a late `sys.stderr` lookup would land trace lines in that
    # buffer instead of the terminal. Capturing it here keeps them separate.
    stream = sys.stderr

    def emit(step: Step) -> None:
        pad = "  " * step.depth
        tag = f"{pad}[d{step.depth}.{step.index}]"
        if step.compaction is not None:
            c = step.compaction
            print(
                f"{tag} ⟳ compacted {c.summarized_messages} msg(s): "
                f"~{c.tokens_before:,} → ~{c.tokens_after:,} tok "
                f"({c.messages_before} → {c.messages_after} msgs)",
                file=stream,
            )
        elif step.usage is not None:
            u = step.usage
            ctx_approx = "~" if u.context_estimated else ""
            tot_approx = "~" if u.cumulative_estimated else ""
            print(
                f"{tag} ctx {ctx_approx}{u.context_tokens:,} tok | total {tot_approx}{u.cumulative_tokens:,} tok",
                file=stream,
            )
        elif step.message is not None:
            print(f"{tag} 💬 " + step.message.replace("\n", "\n" + pad + "   "), file=stream)
        elif step.code is not None:
            print(f"\n{tag} python:\n{pad}  " + step.code.replace("\n", "\n" + pad + "  "),
                  file=stream)
        elif step.llm_call is not None:
            c = step.llm_call
            print(f"{tag} llm() distilled (+{c.tokens:,} tok)\n"
                  f"{pad}  in : {_clip(c.prompt, 160)}\n"
                  f"{pad}  out: {_clip(c.response, 300)}", file=stream)
        elif step.output is not None:
            snippet = step.output if len(step.output) < 1500 else step.output[:1500] + " …[truncated]"
            print(f"{tag} -> " + snippet.replace("\n", "\n" + pad + "     "), file=stream)
        elif step.interaction is not None:
            it = step.interaction
            print(f"{tag} ❓ asked: {_clip(it.question, 200)}\n"
                  f"{pad}   ↳ user: {_clip(it.answer, 200)}", file=stream)
        elif step.final is not None:
            print(f"{tag} FINAL (depth {step.depth}):\n{pad}  "
                  + step.final.replace("\n", "\n" + pad + "  "), file=stream)

    return emit


def _terminal_ask_handler():
    """Prompt the real user on the terminal.

    Binds the real stdout/stdin now, because `ask_user` is called from inside
    `env.execute()`, which redirects sys.stdout/stderr to capture the model's
    own output — a late lookup would send the prompt into that buffer."""
    out = sys.stderr
    src = sys.stdin

    def ask(question: str) -> str:
        print(f"\n\033[1m❓ The assistant needs your input:\033[0m\n   {question}", file=out)
        print("   > ", end="", file=out, flush=True)
        line = src.readline()
        return line.rstrip("\n") if line else "(no answer provided)"

    return ask


_TODO_GLYPH = {"completed": "✔", "in_progress": "▶", "pending": "☐"}


def _plan_printer():
    """Renders the live plan checklist on each TODO mutation (the `--plan` view)."""
    stream = sys.stderr  # bind real stderr (see _verbose_printer for why)

    def emit(step: Step) -> None:
        if step.todo is None:
            return
        pad = "  " * step.depth
        done = sum(1 for it in step.todo if it.status == "completed")
        lines = [f"{pad}Plan ({done}/{len(step.todo)} done):"]
        for it in step.todo:
            lines.append(f"{pad}  {_TODO_GLYPH.get(it.status, '?')} {it.text}")
        print("\n".join(lines), file=stream)

    return emit


def _compose(*emitters):
    """Fan one event out to several emitters; None if there are none."""
    active = [e for e in emitters if e is not None]
    if not active:
        return None
    return lambda step: [e(step) for e in active]


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
    if args.plan:
        settings.enable_planning = True
    if args.plan_root_only:
        settings.enable_planning = True
        settings.planning_root_only = True
    if args.no_interactive:
        settings.enable_interaction = False
    # Interaction needs a real terminal to prompt on; auto-disable when piped.
    interactive = settings.enable_interaction and sys.stdin.isatty()

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
        enable_planning=settings.enable_planning,
        planning_root_only=settings.planning_root_only,
        enable_interaction=interactive,
    )

    on_event = _compose(
        _verbose_printer() if args.verbose else None,
        _plan_printer() if settings.enable_planning else None,
    )
    ask_handler = _terminal_ask_handler() if interactive else None
    answer = engine.run(args.question, corpus, on_event=on_event, ask_handler=ask_handler)
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
    ask.add_argument("--plan", action="store_true",
                     help="Let the model maintain a live TODO plan, shown as a checklist.")
    ask.add_argument("--plan-root-only", action="store_true",
                     help="Enable planning, but for the root agent only (sub-agents don't plan).")
    ask.add_argument("--no-interactive", action="store_true",
                     help="Don't let the model ask the user for help (also auto-off when piped).")
    ask.add_argument("-v", "--verbose", action="store_true", help="Stream the model's code/output to stderr.")
    ask.set_defaults(func=cmd_ask)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
