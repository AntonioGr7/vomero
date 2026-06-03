# Sandbox image for the gVisor exec backend (VOMERO_SANDBOX_IMAGE).
#
# This is NOT the service image (that's deploy/Dockerfile) — it's the throwaway
# container the model's REPL code runs *inside*. Vomero bind-mounts its own
# agent.py + corpus.py at runtime, so this image only needs Python plus whatever
# libraries the model's code imports. Add deps here; you can't pip-install at
# runtime because the sandbox runs --network none and --read-only.
#
# Build (no context needed — nothing is COPYed in):
#   docker build -t vomero-sandbox:latest - < deploy/sandbox.Dockerfile
# Use:
#   VOMERO_SANDBOX=1 VOMERO_SANDBOX_IMAGE=vomero-sandbox:latest uv run python test_loop.py
#
# Deps land in /usr/local/lib/.../site-packages (world-readable), so they import
# fine under the container's non-root --user and the read-only rootfs.
FROM python:3.11-slim

RUN pip install --no-cache-dir pandas openpyxl
