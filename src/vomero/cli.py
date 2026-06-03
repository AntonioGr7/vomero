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

from pathlib import Path

from .channel import CallbackChannel
from .config import Settings
from .context import Context, Corpus
from .engine import Compactor, RLMEngine
from .engine.rlm import Step
from .execution import build_env_factory
from .llm import build_client
from .usage import UsageMeter


def _clip(text: str, limit: int) -> str:
    """Collapse whitespace and truncate, for one-line previews."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " …"


def _read_text(path: str) -> str:
    """Read a file as the in-memory context blob (for `--text PATH`)."""
    return Path(path).expanduser().read_text(encoding="utf-8", errors="replace")


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
            glyph, who = ("↑", "parent") if it.kind == "parent" else ("❓", "user")
            print(f"{tag} {glyph} asked {who}: {_clip(it.question, 200)}\n"
                  f"{pad}   ↳ {who}: {_clip(it.answer, 200)}", file=stream)
        elif step.note is not None:
            print(f"{tag} ⚠ {step.note}", file=stream)
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
    if args.max_output_chars is not None:
        settings.max_output_chars = args.max_output_chars
    if args.max_total_tokens is not None:
        settings.max_total_tokens = args.max_total_tokens
    if args.max_total_calls is not None:
        settings.max_total_calls = args.max_total_calls
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
    if args.ask_root_only:
        settings.interaction_root_only = True
    if args.sandbox:
        settings.exec_backend = "sandbox"
    if args.sandbox_memory is not None:
        settings.sandbox_memory = args.sandbox_memory
    if args.sandbox_cpus is not None:
        settings.sandbox_cpus = args.sandbox_cpus
    if args.sandbox_image is not None:
        settings.sandbox_image = args.sandbox_image
    if args.sandbox_runtime is not None:
        settings.sandbox_runtime = args.sandbox_runtime
    # The capability stays on even when piped, so sub-agents can still consult
    # their parent (model-to-model, no human). Only reaching the *human* needs a
    # real terminal — without one, `ask_user` degrades gracefully.
    human_reachable = settings.enable_interaction and sys.stdin.isatty()

    # Mount the data as a `Source`: a folder (`--data`) or an in-memory blob
    # (`--text PATH`, or `--text -` to read the context from stdin). Exactly one.
    if bool(args.data) == bool(args.text):
        print("error: pass exactly one of --data <folder> or --text <file|->", file=sys.stderr)
        return 2
    try:
        if args.data:
            source = Corpus(args.data)
        else:
            blob = sys.stdin.read() if args.text == "-" else _read_text(args.text)
            source = Context(blob)
    except (FileNotFoundError, OSError) as e:
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
    if settings.exec_backend == "sandbox":
        print(
            f"[sandbox] gVisor backend: image={settings.sandbox_image} "
            f"runtime={settings.sandbox_runtime} "
            f"mem={settings.sandbox_memory} cpus={settings.sandbox_cpus} "
            f"net={settings.sandbox_network}",
            file=sys.stderr,
        )

    # The shell frontend is a Channel: printers receive events, the terminal
    # handler answers ask_user. A browser frontend would swap in its own Channel.
    on_event = _compose(
        _verbose_printer() if args.verbose else None,
        _plan_printer() if settings.enable_planning else None,
    )
    ask_handler = _terminal_ask_handler() if human_reachable else None
    channel = CallbackChannel(on_event=on_event, ask_handler=ask_handler)

    # Caller owns the meter; the engine keeps no per-run state. The budget rides
    # on the meter, so it spans the root loop and every recursive sub-call.
    meter = UsageMeter(
        max_total_tokens=settings.max_total_tokens,
        max_total_calls=settings.max_total_calls,
    )
    answer = engine.run(args.question, source, channel=channel, meter=meter)
    if args.verbose:
        print("\n" + "=" * 60, file=sys.stderr)
    print(answer)

    # Always report token usage on stderr (keeps stdout to just the answer).
    approx = "~" if meter.estimated else ""
    print(
        f"[usage] {meter.calls} model call(s) · {approx}{meter.total_tokens:,} tokens total "
        f"({approx}{meter.prompt_tokens:,} in / {approx}{meter.completion_tokens:,} out)"
        + ("  (estimated — provider did not report usage)" if meter.estimated else ""),
        file=sys.stderr,
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    try:
        serve(args.data, host=args.host, port=args.port, settings=settings)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Measure RLM vs. a stuff-the-context baseline on a benchmark."""
    from .eval import (RLMRunner, StuffBaselineRunner, compare,
                       load_jsonl, load_multihoprag)

    settings = Settings.from_env()
    if args.model:
        settings.model = args.model

    # Load items (each carries the question, gold answer, and its data source).
    try:
        if args.jsonl:
            source = Corpus(args.data) if args.data else None
            items = load_jsonl(args.jsonl, source=source, limit=args.limit)
        elif args.benchmark == "multihoprag":
            items = load_multihoprag(args.data or "data/multihoprag",
                                     limit=args.limit, mode=args.source_mode)
        else:
            print(f"error: unknown benchmark {args.benchmark!r}", file=sys.stderr)
            return 2
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    client = build_client(settings)
    compactor = None
    if settings.compact_ratio > 0:
        compactor = Compactor(
            context_window=settings.context_window, ratio=settings.compact_ratio,
            keep_recent_messages=settings.compact_keep_recent,
            min_reclaim_tokens=settings.compact_min_reclaim,
        )
    engine = RLMEngine(
        client, env_factory=build_env_factory(settings), model=settings.model,
        max_steps=settings.max_steps, max_depth=settings.max_depth,
        max_output_chars=settings.max_output_chars,
        max_parallel_calls=settings.max_parallel_calls, compactor=compactor,
    )

    runners = []
    if args.mode in ("rlm", "both"):
        runners.append(RLMRunner(engine, max_total_tokens=settings.max_total_tokens,
                                 max_total_calls=settings.max_total_calls))
    if args.mode in ("baseline", "both"):
        runners.append(StuffBaselineRunner(client, model=settings.model))

    judge_client = client if args.judge else None
    print(f"Evaluating {len(items)} item(s) with: {', '.join(r.name for r in runners)}"
          + ("  (LLM-judged)" if args.judge else ""), file=sys.stderr)

    def progress(r):
        mark = "✓" if (r.judge or r.contains or r.exact) else "·"
        print(f"  {mark} {_clip(r.question, 70)}  (EM {r.exact:.0f} F1 {r.f1:.2f} "
              f"{r.tokens:,} tok)", file=sys.stderr)

    reports = compare(items, runners, judge_client=judge_client,
                      judge_model=settings.model, on_item=progress)
    print("\n" + "=" * 70, file=sys.stderr)
    for rep in reports:
        print(rep.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vomero", description="Recursive LM assistant over a data folder.")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question about a data folder.")
    ask.add_argument("question", help="The question to answer.")
    ask.add_argument("--data", default=None, help="Path to the data folder (the corpus).")
    ask.add_argument("--text", default=None,
                     help="Mount an in-memory context instead of a folder: a file path, "
                          "or '-' to read the context from stdin. (RLM context-as-a-variable.)")
    ask.add_argument("--model", default=None, help="Override the model name.")
    ask.add_argument("--max-depth", type=int, default=None, help="Max recursion depth.")
    ask.add_argument("--max-steps", type=int, default=None, help="Max REPL steps per level.")
    ask.add_argument("--max-output-chars", type=int, default=None,
                     help="Cap a single tool result's size before it enters context (0 = no cap).")
    ask.add_argument("--max-total-tokens", type=int, default=None,
                     help="Global token budget across the whole run tree (0 = unlimited).")
    ask.add_argument("--max-total-calls", type=int, default=None,
                     help="Global model-call budget across the whole run tree (0 = unlimited).")
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
    ask.add_argument("--ask-root-only", action="store_true",
                     help="Only the root agent may ask the user; sub-agents consult their parent instead.")
    ask.add_argument("--sandbox", action="store_true",
                     help="Run the model's code in a gVisor sandbox (isolated, capped) instead of in-process.")
    ask.add_argument("--sandbox-memory", default=None,
                     help="Max memory per sandbox container (e.g. 512m, 2g). Default 512m.")
    ask.add_argument("--sandbox-cpus", type=float, default=None,
                     help="Max vCPUs per sandbox container (fractional, e.g. 1.5). Default 1.0.")
    ask.add_argument("--sandbox-image", default=None,
                     help="Container image for the sandbox (default python:3.11-slim).")
    ask.add_argument("--sandbox-runtime", default=None,
                     help="OCI runtime for the sandbox (default runsc / gVisor).")
    ask.add_argument("-v", "--verbose", action="store_true", help="Stream the model's code/output to stderr.")
    ask.set_defaults(func=cmd_ask)

    srv = sub.add_parser("serve", help="Serve the corpus over HTTP/SSE for a browser or external client.")
    srv.add_argument("--data", required=True, help="Path to the data folder (the corpus).")
    srv.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    srv.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")
    srv.add_argument("--model", default=None, help="Override the model name.")
    srv.set_defaults(func=cmd_serve)

    ev = sub.add_parser("eval", help="Measure RLM vs. a stuff-the-context baseline on a benchmark.")
    ev.add_argument("--benchmark", default="multihoprag",
                    help="Built-in benchmark to run (default: multihoprag).")
    ev.add_argument("--jsonl", default=None,
                    help="Instead of a built-in: a JSONL file of {question, answer, context?} rows.")
    ev.add_argument("--data", default=None,
                    help="Corpus folder (benchmark default, or the shared source for --jsonl rows).")
    ev.add_argument("--mode", choices=["rlm", "baseline", "both"], default="both",
                    help="Which system(s) to score (default: both).")
    ev.add_argument("--source-mode", choices=["corpus", "context"], default="corpus",
                    help="Mount the benchmark data as a folder or one in-memory context.")
    ev.add_argument("--limit", type=int, default=20, help="Max items to evaluate (default 20).")
    ev.add_argument("--judge", action="store_true",
                    help="Also grade each answer with an LLM judge (for free-form answers).")
    ev.add_argument("--model", default=None, help="Override the model name.")
    ev.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
