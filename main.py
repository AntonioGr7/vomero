#!/usr/bin/env python3
"""main.py — vomero inside a service, running the whole engine in a sandbox pod.

The "vomero-in-the-pod" topology: this process is a thin **dispatcher**. Each
request runs the *entire* RLM engine inside a hardened Kubernetes worker pod (a
warm pool managed by `vomero_sandbox`). The pod holds the corpus, runs the loop +
the model's code, and calls the LLM out through the cluster's egress. The host
never runs the loop, so throughput scales with pods/nodes, not one host process.

Three moves:
  * build the `SandboxedRunner` once at startup (reads your `.env` / environment),
  * warm the worker pods,
  * dispatch each request into a pod, tear the pool down at shutdown.

Prerequisites:
  * a worker image with vomero baked in, loaded into your cluster:
        docker build -f deploy/worker.Dockerfile -t vomero-worker:latest .
        kind load docker-image vomero-worker:latest --name <cluster>
  * a reachable cluster (kubeconfig / in-cluster) and `pip install 'vomero[k8s]'`,
  * the worker needs network egress to the LLM. For dev/test with no egress proxy,
    set VOMERO_K8S_SANDBOX_MANAGE_NETPOL=0 to skip the default-deny policy.

Run it:

    uv run python main.py
"""

from __future__ import annotations

from pathlib import Path

from vomero import Settings
from vomero.sandbox_runner import SandboxedRunner

# The corpus folder the service reasons over (uploaded into the pod per request).
CORPUS_PATH = Path(__file__).resolve().parent / "data" /"multihoprag"

# A couple of "requests" to fire at the running service.
REQUESTS = [
  "Considering the financial performance overview from a Forbes article and the strategic partnership developments mentioned in a Wall Street Journal article on Advance Auto Parts, which single letter symbol represents the company's stock ticker on the New York Stock Exchange?"
]


def main() -> int:
    settings = Settings.from_env()
    print(
        f"[model] {settings.model}  ·  "
        f"[worker image] {settings.k8s_sandbox_image}  ·  "
        f"[namespace] {settings.k8s_sandbox_namespace}  ·  "
        f"[pool] {settings.k8s_sandbox_pool_size}"
    )

    runner = SandboxedRunner(settings)
    report = runner.start()
    print(f"[sandbox] warmed {report.placed}/{report.requested} worker pod(s)")
    try:
        for question in REQUESTS:
            print(f"\n──── REQUEST: {question!r}")
            print("──── live trace (streamed from the pod) ────")
            # ask_stream forwards the in-pod engine's step trace to stderr live,
            # so you watch each step (code, output, sub-calls) as it happens.
            answer = runner.ask_stream(question, CORPUS_PATH)
            print(f"\n──── ANSWER:  {answer}")
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
