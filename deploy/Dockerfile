# ⚠ For the FALLBACK in-cluster Strategy B only. The chosen VM setup (deploy/vm/)
# runs Vomero directly on the host so it can drive Docker — it does NOT use this
# image. Build from the REPO ROOT: docker build -f deploy/Dockerfile -t vomero-service .
FROM python:3.11-slim

# Non-root user (the manifest also enforces runAsNonRoot).
RUN useradd --uid 10001 --create-home app

WORKDIR /app

# Install Vomero + the web server. Copy only what the package needs first so
# Docker can cache the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . "fastapi>=0.110" "uvicorn[standard]>=0.29"

COPY deploy/app.py ./app.py

USER 10001
EXPOSE 8000

# in-process backend (Strategy B): isolation comes from the gVisor-sandboxed pod.
# DATA_ROOT is where the UI's downloads land at runtime (a writable volume).
ENV VOMERO_EXEC_BACKEND=inprocess \
    VOMERO_DATA_ROOT=/data

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
