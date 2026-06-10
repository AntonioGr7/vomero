"""Run the whole vomero engine inside a hardened vomero_sandbox pod.

This is the "vomero-in-the-pod" topology. The host is a thin **dispatcher**: each
request runs the *entire* RLM engine inside a Kubernetes worker pod (drawn from a
warm pool). The pod holds the corpus, runs corpus navigation + `llm`/`rlm`/
`answer` + the model's generated code, and reaches the LLM through the cluster's
egress (an allowlisting proxy in production; open network in dev/test). The host
never runs the loop, so it can't become the CPU/GIL bottleneck under load —
throughput scales with pods/nodes, not with one host process.

Inside the pod the engine runs with `VOMERO_EXEC_BACKEND=inprocess`: it's already
sandboxed, so there's no nested isolation. The worker image must have `vomero`
installed and on PATH (see `deploy/worker.Dockerfile`).

Security note: because the engine runs in the pod, the LLM key (passed per-run via
`env=`) shares the pod with model-generated code. The sandbox protects the host,
the cluster, and other tenants; lock egress to the LLM endpoint and use per-tenant
pods (`max_uses=1`) / scoped keys so the key can't be exfiltrated. (A host-side LLM
broker keeps the key out of the pod entirely — a later option.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Corpus files are uploaded under this subdir of the pod's working directory, so
# the in-pod `vomero ask --data <CORPUS_DIR>` reads them as an ordinary folder.
_CORPUS_DIR = "corpus"


def build_sandbox_pool(settings: Any) -> Any:
    """Build the warm `vomero_sandbox.SandboxPool` of worker pods from `settings`.

    Raises a clear error if `vomero[k8s]` isn't installed. The pool is
    process-global and thread-safe: build it once, reuse across requests."""
    try:
        from vomero_sandbox import SandboxConfig, SandboxPool
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise RuntimeError(
            "the sandboxed runner needs the vomero_sandbox package. "
            "Install it with: pip install 'vomero[k8s]'"
        ) from exc

    config = SandboxConfig(
        namespace=settings.k8s_sandbox_namespace,
        image=settings.k8s_sandbox_image,
        pool_size=settings.k8s_sandbox_pool_size,
        runtime_class=settings.k8s_sandbox_runtime_class,
        default_timeout_s=settings.k8s_sandbox_timeout,
        egress_proxy=settings.k8s_sandbox_egress_proxy,
        kube_context=settings.k8s_sandbox_kube_context,
        manage_network_policy=settings.k8s_sandbox_manage_network_policy,
    )
    return SandboxPool(config)


def _read_corpus_tree(corpus_dir: str | Path, max_bytes: int) -> dict[str, bytes]:
    """Read a host corpus folder into an `input_files` map under `corpus/`."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    files: dict[str, bytes] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        total += len(data)
        if total > max_bytes:
            raise ValueError(
                f"corpus exceeds the {max_bytes // (1024 * 1024)} MiB upload cap; "
                "mount a volume or read from object storage for large corpora"
            )
        rel = path.relative_to(root).as_posix()
        files[f"{_CORPUS_DIR}/{rel}"] = data
    return files


class SandboxedRunner:
    """Dispatches full vomero runs into worker pods. Build once, serve many."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.pool = build_sandbox_pool(settings)

    def start(self) -> None:
        """Warm the worker pods up front so the first request isn't cold."""
        return self.pool.start()

    def _worker_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Environment handed to the in-pod engine. The LLM key rides here,
        per-run (scoped to one process; it doesn't persist on the worker)."""
        s = self.settings
        env = {
            "VOMERO_EXEC_BACKEND": "inprocess",  # already in a sandbox; no nesting
            "VOMERO_PROVIDER": s.provider,
            "VOMERO_MODEL": s.model,
            "VOMERO_MAX_STEPS": str(s.max_steps),
            "VOMERO_MAX_DEPTH": str(s.max_depth),
        }
        if s.api_key:
            env["VOMERO_API_KEY"] = s.api_key
        if s.base_url:
            env["VOMERO_BASE_URL"] = s.base_url
        if extra:
            env.update(extra)
        return env

    def ask(
        self,
        question: str,
        corpus_dir: str | Path,
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Run one full request in a worker pod and return the answer.

        The corpus folder is uploaded into the pod; the in-pod `vomero ask` reads
        it via `--data`. Infrastructure failures raise; a non-zero engine exit (a
        bad run) raises with the worker's stderr so you can see what happened."""
        max_upload = getattr(self.pool.config, "max_upload_bytes", 32 * 1024 * 1024)
        files = _read_corpus_tree(corpus_dir, max_upload)
        argv = ["vomero", "ask", question, "--data", _CORPUS_DIR, "--no-interactive"]
        result = self.pool.exec(
            argv,
            input_files=files,
            env=self._worker_env(env),
            timeout_s=timeout_s or self.settings.k8s_sandbox_timeout,
        )
        if result.timed_out:
            raise TimeoutError(
                f"worker run timed out after {self.settings.k8s_sandbox_timeout}s"
            )
        if not result.ok:
            raise RuntimeError(
                f"worker run failed (exit {result.exit_code}):\n{result.stderr.strip()}"
            )
        # `vomero ask` prints the answer to stdout and usage to stderr.
        return result.stdout.strip()

    def close(self) -> None:
        """Delete the worker pods. Idempotent."""
        self.pool.close()
