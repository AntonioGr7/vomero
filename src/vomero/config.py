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

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("VOMERO_PROVIDER", "openai"),
            model=os.getenv("VOMERO_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("VOMERO_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("VOMERO_API_KEY") or os.getenv("OPENAI_API_KEY"),
            max_steps=int(os.getenv("VOMERO_MAX_STEPS", "24")),
            max_depth=int(os.getenv("VOMERO_MAX_DEPTH", "3")),
            context_window=int(os.getenv("VOMERO_CONTEXT_WINDOW", "128000")),
            compact_ratio=float(os.getenv("VOMERO_COMPACT_RATIO", "0.8")),
            compact_keep_recent=int(os.getenv("VOMERO_COMPACT_KEEP_RECENT", "6")),
            compact_min_reclaim=int(os.getenv("VOMERO_COMPACT_MIN_RECLAIM", "2048")),
            enable_planning=os.getenv("VOMERO_PLAN", "").lower() in ("1", "true", "yes"),
            planning_root_only=os.getenv("VOMERO_PLAN_ROOT_ONLY", "").lower() in ("1", "true", "yes"),
            enable_interaction=os.getenv("VOMERO_INTERACTIVE", "true").lower() in ("1", "true", "yes"),
        )
