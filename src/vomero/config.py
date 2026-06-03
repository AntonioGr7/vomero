"""Runtime configuration, read from the environment (and a local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional, but convenient for local dev
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


@dataclass
class Settings:
    """All knobs Vomero reads from the environment.

    `provider` selects the LLM backend. Today only "openai" (and any
    OpenAI-compatible server via `base_url`) is implemented, but the engine is
    written against an abstract client so "anthropic"/"gemini" can be added
    without touching engine code.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None

    # RLM loop limits
    max_steps: int = 24
    max_depth: int = 3
    # Hard cap (chars) on a single tool result before it enters the transcript,
    # so one oversized print can't permanently bloat the protected recent tail.
    # 0 disables truncation.
    max_output_chars: int = 10_000
    # Fan-out width for llm_batched(...) — max concurrent flat sub-calls.
    max_parallel_calls: int = 8
    # Global budget across the WHOLE run tree (root + every recursive sub-call),
    # enforced on the shared UsageMeter. The run stops spawning model calls once
    # a limit is met and returns its best effort. Both 0 = unlimited.
    max_total_tokens: int = 0
    max_total_calls: int = 0

    # Execution backend for the model's code. "inprocess" (default) runs it
    # in-process with `exec` — fast and full-power, but NOT sandboxed; fine for
    # local dev/testing on trusted data. "sandbox" runs each step inside a
    # gVisor container with hard memory/CPU caps and no network.
    exec_backend: str = "inprocess"
    sandbox_image: str = "python:3.11-slim"
    sandbox_runtime: str = "runsc"      # gVisor; registered with the Docker daemon
    sandbox_memory: str = "512m"        # hard per-container memory cap
    sandbox_cpus: float = 1.0           # fractional vCPUs per container
    sandbox_network: str = "none"       # no network by default
    sandbox_pids_limit: int = 256       # fork-bomb guard
    sandbox_startup_timeout: float = 60.0

    # Per-session persistence (server). When a request carries {user_id,
    # session_id}, its execution environment is kept alive so a follow-up
    # resumes the model's REPL variables; idle sessions are reclaimed after
    # `session_ttl` seconds. `workspace_root`, if set, gives each session a
    # durable directory (mounted read-write in the sandbox) whose files survive
    # even after the variables are reclaimed. None => no warm reuse / workspace.
    workspace_root: str | None = None
    session_ttl: float = 900.0          # 15 min idle before variables are dropped

    # Heavy-load guardrails (server). Both default to 0 = unlimited (prior
    # behavior); set them in any multi-user deployment so the node fails
    # gracefully instead of OOMing under load.
    #  * `max_concurrent_runs` caps in-flight runs per replica; excess POST /runs
    #    get HTTP 429. Size it to node_mem / per-container-mem — NOT CPU, since
    #    runs spend most of their wall-clock blocked on the model.
    #  * `max_sessions` caps the warm/idle session envs the pool keeps alive
    #    (LRU-evicted past the cap), bounding memory held by idle containers.
    max_concurrent_runs: int = 0
    max_sessions: int = 0

    # Context / compaction. When the live context size crosses
    # `compact_ratio * context_window`, the middle of the transcript is
    # summarized. `compact_ratio <= 0` disables compaction.
    context_window: int = 128_000
    compact_ratio: float = 0.8
    compact_keep_recent: int = 6
    compact_min_reclaim: int = 2048

    # Show a live plan/TODO checklist driven by the model.
    enable_planning: bool = False
    # Give the plan surface to the root agent only (default: every depth plans).
    planning_root_only: bool = False
    # Let the model ask the user for help when stuck (auto-disabled off a TTY).
    enable_interaction: bool = True
    # Let only the root agent reach the human (default: any depth may ask).
    interaction_root_only: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("VOMERO_PROVIDER", "openai"),
            model=os.getenv("VOMERO_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("VOMERO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            api_key=(
                os.getenv("VOMERO_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
            ),
            max_steps=int(os.getenv("VOMERO_MAX_STEPS", "24")),
            max_depth=int(os.getenv("VOMERO_MAX_DEPTH", "3")),
            max_output_chars=int(os.getenv("VOMERO_MAX_OUTPUT_CHARS", "10000")),
            max_parallel_calls=int(os.getenv("VOMERO_MAX_PARALLEL_CALLS", "8")),
            max_total_tokens=int(os.getenv("VOMERO_MAX_TOTAL_TOKENS", "0")),
            max_total_calls=int(os.getenv("VOMERO_MAX_TOTAL_CALLS", "0")),
            # VOMERO_SANDBOX=1 is a friendly shortcut for VOMERO_EXEC_BACKEND=sandbox.
            exec_backend=(
                "sandbox"
                if os.getenv("VOMERO_SANDBOX", "").lower() in ("1", "true", "yes")
                else os.getenv("VOMERO_EXEC_BACKEND", "inprocess")
            ),
            sandbox_image=os.getenv("VOMERO_SANDBOX_IMAGE", "python:3.11-slim"),
            sandbox_runtime=os.getenv("VOMERO_SANDBOX_RUNTIME", "runsc"),
            sandbox_memory=os.getenv("VOMERO_SANDBOX_MEMORY", "512m"),
            sandbox_cpus=float(os.getenv("VOMERO_SANDBOX_CPUS", "1.0")),
            sandbox_network=os.getenv("VOMERO_SANDBOX_NETWORK", "none"),
            sandbox_pids_limit=int(os.getenv("VOMERO_SANDBOX_PIDS", "256")),
            sandbox_startup_timeout=float(
                os.getenv("VOMERO_SANDBOX_STARTUP_TIMEOUT", "60")
            ),
            workspace_root=os.getenv("VOMERO_WORKSPACE_ROOT") or None,
            session_ttl=float(os.getenv("VOMERO_SESSION_TTL", "900")),
            max_concurrent_runs=int(os.getenv("VOMERO_MAX_CONCURRENT_RUNS", "0")),
            max_sessions=int(os.getenv("VOMERO_MAX_SESSIONS", "0")),
            context_window=int(os.getenv("VOMERO_CONTEXT_WINDOW", "128000")),
            compact_ratio=float(os.getenv("VOMERO_COMPACT_RATIO", "0.8")),
            compact_keep_recent=int(os.getenv("VOMERO_COMPACT_KEEP_RECENT", "6")),
            compact_min_reclaim=int(os.getenv("VOMERO_COMPACT_MIN_RECLAIM", "2048")),
            enable_planning=os.getenv("VOMERO_PLAN", "").lower() in ("1", "true", "yes"),
            planning_root_only=os.getenv("VOMERO_PLAN_ROOT_ONLY", "").lower() in ("1", "true", "yes"),
            enable_interaction=os.getenv("VOMERO_INTERACTIVE", "true").lower() in ("1", "true", "yes"),
            interaction_root_only=os.getenv("VOMERO_ASK_ROOT_ONLY", "").lower() in ("1", "true", "yes"),
        )
