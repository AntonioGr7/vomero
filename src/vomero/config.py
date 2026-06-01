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
    without touching engine code. See docs/adr/0002.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None

    # RLM loop limits
    max_steps: int = 24
    max_depth: int = 3

    # Execution backend for the model's code. "inprocess" (default) runs it
    # in-process with `exec` — fast and full-power, but NOT sandboxed; fine for
    # local dev/testing on trusted data. "sandbox" runs each step inside a
    # gVisor container with hard memory/CPU caps and no network (docs/adr/0004).
    exec_backend: str = "inprocess"
    sandbox_image: str = "python:3.11-slim"
    sandbox_runtime: str = "runsc"      # gVisor; registered with the Docker daemon
    sandbox_memory: str = "512m"        # hard per-container memory cap
    sandbox_cpus: float = 1.0           # fractional vCPUs per container
    sandbox_network: str = "none"       # no network by default
    sandbox_pids_limit: int = 256       # fork-bomb guard
    sandbox_startup_timeout: float = 60.0

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
            context_window=int(os.getenv("VOMERO_CONTEXT_WINDOW", "128000")),
            compact_ratio=float(os.getenv("VOMERO_COMPACT_RATIO", "0.8")),
            compact_keep_recent=int(os.getenv("VOMERO_COMPACT_KEEP_RECENT", "6")),
            compact_min_reclaim=int(os.getenv("VOMERO_COMPACT_MIN_RECLAIM", "2048")),
            enable_planning=os.getenv("VOMERO_PLAN", "").lower() in ("1", "true", "yes"),
            planning_root_only=os.getenv("VOMERO_PLAN_ROOT_ONLY", "").lower() in ("1", "true", "yes"),
            enable_interaction=os.getenv("VOMERO_INTERACTIVE", "true").lower() in ("1", "true", "yes"),
            interaction_root_only=os.getenv("VOMERO_ASK_ROOT_ONLY", "").lower() in ("1", "true", "yes"),
        )
