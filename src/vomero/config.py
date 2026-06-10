"""Runtime configuration, read from the environment (and a local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional, but convenient for local dev
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# How the model's REPL code is isolated, and the aliases we accept (so legacy
# values and friendly synonyms all resolve to one canonical name). NB: running
# the WHOLE engine inside a vomero_sandbox pod is a deployment topology (see
# sandbox_runner.py), NOT an exec backend — the engine there just runs inprocess.
EXEC_BACKENDS = ("inprocess", "gvisor")
_EXEC_BACKEND_ALIASES = {
    "": "inprocess",
    "none": "inprocess",
    "inprocess": "inprocess",
    "sandbox": "gvisor",      # legacy: "sandbox" used to mean the gVisor backend
    "gvisor": "gvisor",
    "runsc": "gvisor",
}


def normalize_exec_backend(value: str) -> str:
    """Map a backend name/alias to its canonical form, or raise on unknown input."""
    key = (value or "").strip().lower()
    if key not in _EXEC_BACKEND_ALIASES:
        raise ValueError(
            f"unknown exec backend {value!r}; choose one of: "
            f"{', '.join(EXEC_BACKENDS)}"
        )
    return _EXEC_BACKEND_ALIASES[key]


def _resolve_exec_backend() -> str:
    """Resolve the execution-isolation choice from the environment.

    `VOMERO_EXEC_BACKEND` is authoritative when set (`inprocess` | `gvisor`);
    otherwise the legacy `VOMERO_SANDBOX=1` shortcut maps onto `gvisor`."""
    raw = os.getenv("VOMERO_EXEC_BACKEND", "").strip().lower()
    if not raw and os.getenv("VOMERO_SANDBOX", "").strip().lower() in ("1", "true", "yes"):
        raw = "gvisor"
    return normalize_exec_backend(raw)


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

    # Where the model's REPL code runs — one of two isolation strategies:
    #   "inprocess"  run on the machine with `exec`: fast, full-power, NOT
    #                isolated. Fine for local dev/testing on trusted data.
    #   "gvisor"     each REPL step runs inside a gVisor container (Docker + runsc)
    #                with hard memory/CPU caps and no network.
    # Resolved in from_env(); the legacy VOMERO_SANDBOX=1 shortcut maps to gvisor.
    # (To run the WHOLE engine inside a hardened Kubernetes pod, that's a
    # deployment topology — see sandbox_runner.py — and the engine there simply
    # runs with exec_backend="inprocess". It is not an exec backend value.)
    exec_backend: str = "inprocess"
    sandbox_image: str = "python:3.11-slim"
    sandbox_runtime: str = "runsc"      # gVisor; registered with the Docker daemon
    sandbox_memory: str = "512m"        # hard per-container memory cap
    sandbox_cpus: float = 1.0           # fractional vCPUs per container
    sandbox_network: str = "none"       # no network by default
    sandbox_pids_limit: int = 256       # fork-bomb guard
    sandbox_startup_timeout: float = 60.0

    # Knobs for the sandboxed RUNNER (sandbox_runner.py): a warm pool of
    # vomero_sandbox worker pods, each running the whole engine. Map onto
    # vomero_sandbox.SandboxConfig.
    k8s_sandbox_namespace: str = "sandbox"
    k8s_sandbox_image: str = "vomero-worker:latest"  # image with vomero baked in
    k8s_sandbox_pool_size: int = 3          # warm pods == concurrency ceiling
    k8s_sandbox_runtime_class: str | None = None   # e.g. "gvisor" (needs cluster setup)
    k8s_sandbox_timeout: float = 120.0      # per-request wall-clock limit (a full run)
    k8s_sandbox_egress_proxy: str | None = None    # allowlisting proxy URL; else deny-all
    k8s_sandbox_kube_context: str | None = None    # None = default kubeconfig context
    # Set False to skip the default-deny egress NetworkPolicy — needed for the
    # worker to reach the LLM endpoint when you have no egress proxy yet (dev/test).
    k8s_sandbox_manage_network_policy: bool = True

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
            exec_backend=_resolve_exec_backend(),
            sandbox_image=os.getenv("VOMERO_SANDBOX_IMAGE", "python:3.11-slim"),
            sandbox_runtime=os.getenv("VOMERO_SANDBOX_RUNTIME", "runsc"),
            sandbox_memory=os.getenv("VOMERO_SANDBOX_MEMORY", "512m"),
            sandbox_cpus=float(os.getenv("VOMERO_SANDBOX_CPUS", "1.0")),
            sandbox_network=os.getenv("VOMERO_SANDBOX_NETWORK", "none"),
            sandbox_pids_limit=int(os.getenv("VOMERO_SANDBOX_PIDS", "256")),
            sandbox_startup_timeout=float(
                os.getenv("VOMERO_SANDBOX_STARTUP_TIMEOUT", "60")
            ),
            k8s_sandbox_namespace=os.getenv("VOMERO_K8S_SANDBOX_NAMESPACE", "sandbox"),
            k8s_sandbox_image=os.getenv("VOMERO_K8S_SANDBOX_IMAGE", "vomero-worker:latest"),
            k8s_sandbox_pool_size=int(os.getenv("VOMERO_K8S_SANDBOX_POOL_SIZE", "3")),
            k8s_sandbox_runtime_class=os.getenv("VOMERO_K8S_SANDBOX_RUNTIME_CLASS") or None,
            k8s_sandbox_timeout=float(os.getenv("VOMERO_K8S_SANDBOX_TIMEOUT", "120")),
            k8s_sandbox_egress_proxy=os.getenv("VOMERO_K8S_SANDBOX_EGRESS_PROXY") or None,
            k8s_sandbox_kube_context=os.getenv("VOMERO_K8S_SANDBOX_KUBE_CONTEXT") or None,
            k8s_sandbox_manage_network_policy=os.getenv(
                "VOMERO_K8S_SANDBOX_MANAGE_NETPOL", "true"
            ).strip().lower() in ("1", "true", "yes"),
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
