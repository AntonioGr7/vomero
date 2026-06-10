# Worker image for the SANDBOXED RUNNER (src/vomero/sandbox_runner.py): a pod
# that runs the WHOLE vomero engine inside a vomero_sandbox worker. The host
# dispatcher execs `vomero ask` into this image, passing the corpus + LLM key.
#
# Build from the REPO ROOT, then load into your cluster:
#   docker build -f deploy/worker.Dockerfile -t vomero-worker:latest .
#   kind load docker-image vomero-worker:latest --name <cluster>     # local kind
#   # production: push to your registry and set VOMERO_K8S_SANDBOX_IMAGE to it
FROM python:3.13-slim

# Non-root, matching the sandbox's hardened defaults (runAsUser=1000).
RUN useradd --uid 1000 --create-home app

WORKDIR /opt/vomero

# Copy only what the package build needs first, so the dep layer caches.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Base vomero only — the worker runs the engine in-process and does NOT need the
# vomero_sandbox (k8s) client; that lives on the dispatcher side.
RUN pip install --no-cache-dir .

# The engine is already sandboxed by the pod, so it runs in-process inside it.
# Provider/model/key are injected per-run by the dispatcher via env=.
ENV VOMERO_EXEC_BACKEND=inprocess

USER 1000

# No CMD/ENTRYPOINT needed: the vomero_sandbox pool controls the pod's lifecycle
# (its own watchdog) and execs `vomero ask ...` into the container per request.
# We only need `vomero` (and python) to be on PATH, which `pip install` provides.
