# Deploying Vomero

A practical guide for someone who **did not write this library** and wants to
run it as a service — embedded in FastAPI, behind a custom socket server, or
scheduled as a microservice on Kubernetes — **safely**.

If you only want to try it on the command line, read the [README](../README.md)
first; this document picks up where that leaves off. For the exact HTTP/SSE wire
protocol of the bundled server, see [serving.md](serving.md).

> **The one thing to internalise before deploying:** Vomero answers questions by
> letting the model **write and run Python code**. That code is as trusted as
> whoever can influence the question *and* the corpus. Section
> [4. Security & isolation](#4-security--isolation) is not optional reading — it
> determines which deployment shape is safe for you.

**Contents**

1. [Getting started](#1-getting-started)
2. [The embedding API](#2-the-embedding-api) — `RLMEngine`, `Corpus`, `Channel`, sessions
3. [Environment & configuration](#3-environment--configuration)
4. [Security & isolation](#4-security--isolation) — the threat model and the two sandbox strategies
5. [Behind FastAPI](#5-behind-fastapi)
6. [Behind your own socket server](#6-behind-your-own-socket-server)
7. [On Kubernetes](#7-on-kubernetes)
8. [Operating it](#8-operating-it) — concurrency, sessions, scaling, shutdown

---

## 1. Getting started

### Install

Vomero is a normal Python package (Python ≥ 3.11). Its only runtime
dependencies are `openai` and `python-dotenv`.

```bash
# in your own project
pip install -e /path/to/vomero        # or add it to your dependencies
# or, for local hacking on the repo itself:
uv venv && uv pip install -e ".[dev]"
```

### Point it at a model

Vomero talks to any OpenAI-compatible endpoint, or to Google Gemini. Configure
via environment variables (a local `.env` is auto-loaded — see
[`.env.example`](../.env.example)):

```bash
# OpenAI
export VOMERO_PROVIDER=openai
export VOMERO_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-...

# …or any OpenAI-compatible server (vLLM, LM Studio, OpenRouter, Together, …)
export VOMERO_BASE_URL=http://localhost:1234/v1
export VOMERO_API_KEY=not-needed-for-local

# …or Gemini (uses its OpenAI-compatible endpoint; base_url is automatic)
export VOMERO_PROVIDER=gemini
export VOMERO_MODEL=gemini-2.5-flash
export GEMINI_API_KEY=...
```

### Smoke test

```bash
vomero ask "What blocks P-BEACON, and which team owns the fix?" \
  --data examples/sample_corpus -v
```

If that prints an answer, you're wired up. Everything below is about turning
this into a service.

---

## 2. The embedding API

You don't need the CLI or the bundled server to use Vomero — they're both thin
shells over three objects. This is all the public surface a service needs.

### The corpus

```python
from vomero.context.corpus import Corpus

corpus = Corpus("/data/my-folder")          # a read-only, lazy view of a folder
corpus = corpus.subset(["a.md", "docs/b.md"])  # scope to specific files
```

`Corpus` never loads the folder into memory; the model navigates it with
`grep` / `peek` / `read` / `files` from inside the REPL. The root is resolved
and path-escapes are rejected, so a corpus can't read outside its folder.

### The engine

```python
from vomero.config import Settings
from vomero.engine import RLMEngine, Compactor
from vomero.execution import build_env_factory
from vomero.llm import build_client
from vomero.usage import UsageMeter
from vomero.channel import NullChannel

settings = Settings.from_env()             # reads the env vars from §3

engine = RLMEngine(
    build_client(settings),                # the LLM client (provider-agnostic)
    env_factory=build_env_factory(settings),  # where model code runs (in-process or gVisor)
    model=settings.model,
    max_steps=settings.max_steps,          # cost ceiling per loop
    max_depth=settings.max_depth,          # recursion ceiling
    compactor=Compactor(context_window=settings.context_window,
                        ratio=settings.compact_ratio) if settings.compact_ratio > 0 else None,
    enable_planning=settings.enable_planning,
    enable_interaction=settings.enable_interaction,
)

meter = UsageMeter()                       # caller-owned; the engine keeps NO per-run state
answer = engine.run("your question", corpus, channel=NullChannel(), meter=meter)
print(answer, meter.calls, meter.total_tokens)
```

Key property: **the engine holds no per-run state.** Usage comes back through
the `meter` you pass in, so a single `RLMEngine` instance is safe to call from
many threads at once. Build it once at startup; reuse it for every request.

### The Channel (your frontend seam)

The engine emits progress and asks the human only through a `Channel`
([vomero/channel.py](../src/vomero/channel.py)):

```python
class Channel(Protocol):
    def emit(self, step: Step) -> None: ...        # one progress/usage/plan/final event
    def ask_user(self, question: str) -> str: ...  # reach the human (may block the run)
```

To build any frontend (HTTP, WebSocket, gRPC, a socket protocol), you implement
those two methods. `emit` receives a `Step` dataclass; serialize the fields you
care about (there's a ready-made `vomero.server.step_to_payload(step)` that
turns a `Step` into a JSON-able dict). `ask_user` should block until the human
answers — or return a sensible default when there's no human (see
`vomero.channel.NO_USER_REPLY`).

Built-ins: `NullChannel` (drops events, no human — safe default for headless),
`CallbackChannel` (adapts plain callbacks).

### `run()` parameters worth knowing

```python
engine.run(
    question, corpus,
    channel=...,            # your Channel
    meter=...,              # your UsageMeter
    history=prev_messages,  # seed with a prior conversation (multi-turn follow-ups)
    transcript_sink=out,    # a list the engine fills with the resumable transcript
    env=pooled_env,         # reuse a persistent env to keep variables/workspace
    enable_planning=True,   # per-request override of the engine default
)
```

- **`history` / `transcript_sink`** give you **multi-turn conversations**: pass
  an empty list as `transcript_sink`, store it under your session key when the
  run finishes, and feed it back as `history` on the next question.
- **`env`** gives you **stateful sessions**: reuse the same execution environment
  across turns so the model's REPL variables (and, with the sandbox, its
  workspace files) survive. The [`SessionEnvPool`](#sessions-variables-and-a-durable-workspace)
  manages that for you.

### Sessions: variables and a durable workspace

```python
from vomero.execution import build_session_pool

pool = build_session_pool(settings)        # keyed by (user_id, session_id)

with pool.session((user_id, session_id)) as env:
    engine.run(question, corpus, channel=ch, meter=m,
               history=history, transcript_sink=transcript, env=env)
```

The pool keeps one environment alive per session so a follow-up resumes the
model's variables; idle sessions are reclaimed after `session_ttl` seconds. If
`workspace_root` is set, each session also gets a durable directory (mounted
read-write into the sandbox at `/workspace`) whose files survive even after the
variables are reclaimed. Call `pool.start_sweeper()` to reclaim idle sessions on
a timer, and `pool.close_all()` on shutdown.

---

## 3. Environment & configuration

Everything is driven by environment variables, read once by `Settings.from_env()`.
A `.env` file in the working directory is loaded automatically.

### Model / provider

| Variable | Default | Notes |
|---|---|---|
| `VOMERO_PROVIDER` | `openai` | `openai` or `gemini` |
| `VOMERO_MODEL` | `gpt-4o-mini` | model name as the provider expects it |
| `VOMERO_BASE_URL` / `OPENAI_BASE_URL` | — | for OpenAI-compatible servers |
| `VOMERO_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | credentials |

### Loop limits (cost & blast-radius ceilings)

| Variable | Default | What it bounds |
|---|---|---|
| `VOMERO_MAX_STEPS` | `24` | REPL steps per agent loop |
| `VOMERO_MAX_DEPTH` | `3` | recursion depth (`rlm()` sub-agents) |
| `VOMERO_CONTEXT_WINDOW` | `128000` | model context window (compaction threshold = ratio × this) |
| `VOMERO_COMPACT_RATIO` | `0.8` | compact when context hits this fraction; `0` disables |
| `VOMERO_COMPACT_KEEP_RECENT` | `6` | recent messages kept verbatim |
| `VOMERO_COMPACT_MIN_RECLAIM` | `2048` | veto compaction below this reclaim |

### Execution backend (the security-critical knob)

| Variable | Default | What it does |
|---|---|---|
| `VOMERO_SANDBOX=1` | off | shortcut for `VOMERO_EXEC_BACKEND=sandbox` |
| `VOMERO_EXEC_BACKEND` | `inprocess` | `inprocess` (fast, **not isolated**) or `sandbox` (gVisor) |
| `VOMERO_SANDBOX_IMAGE` | `python:3.11-slim` | container image (point at one with your deps) |
| `VOMERO_SANDBOX_RUNTIME` | `runsc` | OCI runtime (gVisor) |
| `VOMERO_SANDBOX_MEMORY` | `512m` | hard memory cap (e.g. `1g`) |
| `VOMERO_SANDBOX_CPUS` | `1.0` | fractional vCPUs |
| `VOMERO_SANDBOX_NETWORK` | `none` | docker `--network`; keep `none` |
| `VOMERO_SANDBOX_PIDS` | `256` | max processes (fork-bomb guard) |
| `VOMERO_SANDBOX_STARTUP_TIMEOUT` | `60` | seconds to wait for the container (first run pulls the image) |

### Sessions / workspace (server, multi-turn)

| Variable | Default | What it does |
|---|---|---|
| `VOMERO_WORKSPACE_ROOT` | — | host dir for per-session writable workspaces; unset = no durable files |
| `VOMERO_SESSION_TTL` | `900` | seconds of idle before a session's variables are dropped (lower to 120–300 under load) |
| `VOMERO_MAX_CONCURRENT_RUNS` | `0` (unlimited) | in-flight runs per replica; excess `POST /runs` get **HTTP 429**. Size to `node_mem / per-container-mem`, not CPU |
| `VOMERO_MAX_SESSIONS` | `0` (unlimited) | warm/idle session envs the pool keeps; LRU-evicted past the cap (bounds memory held by idle containers) |

### Behaviour toggles

| Variable | Default | What it does |
|---|---|---|
| `VOMERO_PLAN` | off | model maintains a live TODO plan (emitted as `todo` events) |
| `VOMERO_PLAN_ROOT_ONLY` | off | only the root agent plans |
| `VOMERO_INTERACTIVE` | `true` | model may call `ask_user`; degrades gracefully when headless |
| `VOMERO_ASK_ROOT_ONLY` | off | only the root agent may ask the human |

---

## 4. Security & isolation

### Threat model

Model-authored Python runs on **every** question. Treat it as untrusted as soon
as **either** the question **or** the corpus can be influenced by someone you
don't fully trust. The risks of that code, in order of how often they bite:

1. **Secret exfiltration** — reading credentials and sending them out.
2. **Network egress** — calling arbitrary hosts.
3. **Filesystem access** — reading/altering anything the process can.
4. **Resource exhaustion** — fork bombs, memory blowups, CPU spin.
5. **Node/host escape** — breaking out of the container to the kernel.

The default `inprocess` backend defends against **none** of these — model code
runs in your service process with `exec`, so it can read `os.environ`
(your API key!), open sockets, and touch the filesystem. **It is only for
trusted corpora and trusted callers.**

### Two isolation strategies — pick by trust level

#### Strategy A — per-step container sandbox (Vomero's gVisor backend)

Set `VOMERO_SANDBOX=1`. Each REPL step runs in a fresh gVisor (`runsc`)
container that:

- has **no network** (`--network none`),
- carries **no secrets** — your API key stays in the *host* Vomero process;
  the model's `llm()`/`rlm()` calls round-trip back over an RPC socket, so the
  key never enters the sandbox,
- sees the corpus **read-only** and nothing else of the host filesystem,
- runs **non-root**, capability-dropped, with a read-only rootfs and hard
  **memory / CPU / PID** caps,
- is **torn down** at the end of the run.

This is the strongest option and the right default for **untrusted input**,
because it protects *both the node and your credentials* from model code. Its
cost: it needs a Docker daemon with the `runsc` runtime reachable from the
service (see [README → Sandboxed execution](../README.md#sandboxed-execution-gvisor)
for installing gVisor). That requirement shapes the Kubernetes story in §7.

#### Strategy B — pod/VM-level sandbox + in-process backend

Run Vomero with `inprocess` **inside** an environment that is itself a gVisor
sandbox (a gVisor `RuntimeClass` pod on Kubernetes, or a locked-down VM). The
sandbox now protects the **node** from model code, but the model code still
shares your service's network and can read its environment/secrets.

Use this only when input is **semi-trusted** and your main worry is node escape,
**and** combine it with: keep the LLM key out of reach where you can, lock down
egress with a `NetworkPolicy`, and use a budget-capped / scoped API key so a
leaked key is low-value.

> Rule of thumb: **untrusted input → Strategy A.** Internal/trusted tool where
> you mainly want kernel-level node protection → Strategy B.

### Hardening that applies either way

- **Bind a corpus at startup, never accept paths from clients.** The bundled
  server does this; do the same in your frontend.
- **Egress allow-list.** The model talks to the LLM through Vomero; the only
  egress your service needs is to the LLM endpoint. Deny everything else.
- **Secrets as references, not env where avoidable.** With Strategy A the key is
  safe in the host process; with Strategy B remember model code can read env.
- **Resource limits at two layers:** Vomero's loop limits (`MAX_STEPS`,
  `MAX_DEPTH`) for cost, and the container/pod limits for memory/CPU.
- **Timeouts.** Wrap `engine.run` in a wall-clock timeout in your frontend; a
  model can otherwise loop up to `MAX_STEPS`.

---

## 5. Behind FastAPI

A complete, production-shaped example: one shared engine, a per-request
`Channel` that streams Server-Sent Events, multi-turn history, and per-session
stateful environments. `engine.run` is blocking, so it runs in a worker thread
while the request streams from a queue.

```python
# app.py
import asyncio, json, queue, threading, uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from vomero.config import Settings
from vomero.context.corpus import Corpus
from vomero.engine import RLMEngine, Compactor
from vomero.execution import build_env_factory, build_session_pool
from vomero.llm import build_client
from vomero.llm.base import Message
from vomero.usage import UsageMeter
from vomero.server import step_to_payload   # reuse the Step -> JSON serializer

settings = Settings.from_env()
CORPUS = Corpus("/data/my-folder")          # bound once; clients never send paths

engine = RLMEngine(
    build_client(settings),
    env_factory=build_env_factory(settings),
    model=settings.model,
    max_steps=settings.max_steps,
    max_depth=settings.max_depth,
    compactor=(Compactor(context_window=settings.context_window,
                         ratio=settings.compact_ratio)
               if settings.compact_ratio > 0 else None),
    enable_interaction=True,
)
pool = build_session_pool(settings)         # variables + workspace per session
HISTORY: dict[tuple[str, str], list[Message]] = {}   # swap for Redis in prod

app = FastAPI()


class SSEChannel:
    """Streams events to one client; ask_user blocks on a reply queue."""
    def __init__(self):
        self.events: queue.Queue = queue.Queue()
        self.replies: queue.Queue = queue.Queue()
    def emit(self, step):
        self.events.put(step_to_payload(step))
    def ask_user(self, question: str) -> str:
        self.events.put({"type": "ask_user", "question": str(question)})
        return self.replies.get()           # blocks the run thread until /reply


SESSIONS: dict[str, SSEChannel] = {}
_END = object()


@app.post("/runs")
async def start_run(req: Request):
    body = await req.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "missing question"}, status_code=400)
    user_id = body.get("user_id", "anonymous")
    session_id = body.get("session_id") or uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    ch = SSEChannel()
    SESSIONS[run_id] = ch

    def worker():
        key = (user_id, session_id)
        meter = UsageMeter()
        transcript: list[Message] = []
        try:
            with pool.session(key) as env:          # stateful env for this session
                engine.run(question, CORPUS, channel=ch, meter=meter,
                           history=HISTORY.get(key, []), transcript_sink=transcript,
                           env=env)
            HISTORY[key] = transcript               # remember the conversation
            ch.events.put({"type": "done", "session_id": session_id,
                           "usage": {"calls": meter.calls,
                                     "total_tokens": meter.total_tokens}})
        except Exception as exc:
            ch.events.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            ch.events.put(_END)

    threading.Thread(target=worker, daemon=True).start()
    return {"session_id": session_id, "run_id": run_id,
            "events": f"/runs/{run_id}/events", "reply": f"/runs/{run_id}/reply"}


@app.get("/runs/{run_id}/events")
async def events(run_id: str):
    ch = SESSIONS.get(run_id)
    if ch is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)

    async def gen():
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await loop.run_in_executor(None, ch.events.get)
                if item is _END:
                    yield "event: end\ndata: {}\n\n"
                    break
                yield f"event: {item.get('type','event')}\ndata: {json.dumps(item)}\n\n"
        finally:
            SESSIONS.pop(run_id, None)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/runs/{run_id}/reply")
async def reply(run_id: str, req: Request):
    ch = SESSIONS.get(run_id)
    if ch is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    body = await req.json()
    ch.replies.put(str(body.get("answer", "")))
    return {"ok": True}


@app.get("/healthz")
async def healthz():
    return {"ok": True}                     # liveness/readiness probe target


@app.on_event("shutdown")
def _cleanup():
    pool.close_all()                        # tear down warm sandboxes
```

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Notes:

- **Why threads?** `engine.run` is synchronous and CPU/IO-blocking; running it in
  a thread keeps the event loop responsive. The engine is thread-safe (no
  per-run state), so one `engine` serves all requests.
- **History and sessions are in-memory here.** For more than one replica, move
  `HISTORY` and the session affinity to an external store (see §8).
- This is the same shape as the bundled `vomero.server`; that one is stdlib-only
  and is a fine reference if you don't want FastAPI.

---

## 6. Behind your own socket server

If you're not exposing HTTP — e.g. an internal worker that takes jobs over a
Unix or TCP socket — you only need a `Channel` that writes events to the
connection. Here's a minimal newline-delimited-JSON server over a TCP socket:

```python
# socket_server.py
import json, socketserver, threading
from vomero.config import Settings
from vomero.context.corpus import Corpus
from vomero.engine import RLMEngine
from vomero.execution import build_env_factory
from vomero.llm import build_client
from vomero.usage import UsageMeter
from vomero.server import step_to_payload

settings = Settings.from_env()
CORPUS = Corpus("/data/my-folder")
ENGINE = RLMEngine(build_client(settings), env_factory=build_env_factory(settings),
                   model=settings.model, max_steps=settings.max_steps,
                   max_depth=settings.max_depth)


class JSONLChannel:
    """Writes one JSON object per line to the socket. ask_user reads a line back."""
    def __init__(self, wfile, rfile):
        self._w, self._r = wfile, rfile
        self._lock = threading.Lock()
    def _send(self, obj):
        with self._lock:
            self._w.write((json.dumps(obj) + "\n").encode()); self._w.flush()
    def emit(self, step):
        self._send(step_to_payload(step))
    def ask_user(self, question: str) -> str:
        self._send({"type": "ask_user", "question": str(question)})
        line = self._r.readline()
        return json.loads(line)["answer"] if line else "(no answer)"


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        req = json.loads(self.rfile.readline())     # {"question": "..."}
        ch = JSONLChannel(self.wfile, self.rfile)
        meter = UsageMeter()
        try:
            answer = ENGINE.run(req["question"], CORPUS, channel=ch, meter=meter)
            ch._send({"type": "final", "final": answer,
                      "usage": {"calls": meter.calls, "total_tokens": meter.total_tokens}})
        except Exception as exc:
            ch._send({"type": "error", "error": f"{type(exc).__name__}: {exc}"})


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    with Server(("0.0.0.0", 9999), Handler) as s:
        s.serve_forever()
```

Test it:

```bash
printf '{"question": "What blocks P-BEACON?"}\n' | nc localhost 9999
```

For a **Unix domain socket**, swap `ThreadingTCPServer` →
`socketserver.ThreadingUnixStreamServer` and bind a path instead of a host/port.
The `Channel` is the only Vomero-specific part; the transport is entirely yours.

---

## 7. On Kubernetes

The deciding question is the same as §4: **what isolates the model's code?** That
choice drives the whole manifest, because Vomero's own sandbox (Strategy A)
shells out to `docker run --runtime=runsc`, which is awkward inside a vanilla
pod.

### Recommended: Strategy B — sandbox the *pod*, run `inprocess` inside it

Make the pod itself a gVisor sandbox with a `RuntimeClass` (GKE Sandbox, or
gVisor installed on your nodes), and run Vomero with the default in-process
backend. No Docker-in-Docker, no privileged pod. The pod sandbox protects the
**node**; you protect **secrets and egress** with the rest of the manifest.

```yaml
# runtimeclass.yaml  (cluster-scoped; provided by your gVisor/GKE-Sandbox setup)
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: vomero }
spec:
  replicas: 2
  selector: { matchLabels: { app: vomero } }
  template:
    metadata: { labels: { app: vomero } }
    spec:
      runtimeClassName: gvisor            # <-- the pod runs under gVisor
      automountServiceAccountToken: false
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
        seccompProfile: { type: RuntimeDefault }
      containers:
        - name: vomero
          image: your-registry/vomero:latest    # your image with vomero + uvicorn
          args: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
          ports: [{ containerPort: 8000 }]
          env:
            - { name: VOMERO_PROVIDER, value: "openai" }
            - { name: VOMERO_MODEL, value: "gpt-4o-mini" }
            - { name: VOMERO_MAX_STEPS, value: "16" }   # cost ceiling
            - { name: VOMERO_MAX_DEPTH, value: "2" }
          envFrom:
            - secretRef: { name: vomero-llm }           # OPENAI_API_KEY lives here
          resources:
            requests: { cpu: "500m", memory: "512Mi" }
            limits:   { cpu: "2",    memory: "2Gi" }     # hard caps
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }
          volumeMounts:
            - { name: corpus, mountPath: /data/my-folder, readOnly: true }
            - { name: tmp, mountPath: /tmp }             # writable scratch
          readinessProbe:
            httpGet: { path: /healthz, port: 8000 }
      volumes:
        - name: corpus
          persistentVolumeClaim: { claimName: my-corpus, readOnly: true }
        - name: tmp
          emptyDir: {}
```

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata: { name: vomero-llm }
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-..."
---
# networkpolicy.yaml — deny all egress except DNS + the LLM endpoint
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: { name: vomero-egress }
spec:
  podSelector: { matchLabels: { app: vomero } }
  policyTypes: ["Egress"]
  egress:
    - to: []                       # DNS
      ports: [{ protocol: UDP, port: 53 }, { protocol: TCP, port: 53 }]
    - to:                          # your LLM provider (egress gateway / CIDR)
        - ipBlock: { cidr: 0.0.0.0/0 }   # TIGHTEN: restrict to the provider's range
      ports: [{ protocol: TCP, port: 443 }]
```

> With in-process execution the model's code can read this pod's env (your LLM
> key) and use its network. That's why the `NetworkPolicy` egress allow-list and
> a **scoped, budget-capped** API key matter — they limit the damage if the key
> leaks. If your input is genuinely untrusted, prefer Strategy A below.

### Strongest: Strategy A — Vomero's per-step gVisor sandbox

`VOMERO_SANDBOX=1` keeps your **credentials out of the model's reach** (they stay
in the host process; the sandbox has no network and no secrets). The catch is
that it needs a Docker daemon with `runsc` reachable from the service, which is a
privilege you don't want loose in a shared cluster. Practical ways to get it:

- **Dedicated node pool with Docker + gVisor**, with the Docker socket mounted
  *only* into the Vomero pod (and that pod isolated by node taints/affinity).
  This is the closest fit but treat the pod as privileged-adjacent.
- **Run Vomero on a VM** (where installing Docker + gVisor is straightforward,
  per the README) and call it from the cluster over the network. Often the
  simplest way to get strong isolation without fighting Kubernetes.

Either way, keep the rest of the hardening from the manifest above (non-root,
read-only rootfs, dropped caps, `NetworkPolicy`, resource limits, secrets).

### Scheduling / scaling notes specific to Vomero

- **Cap load per replica, or a spike OOMs the node.** Each run holds a
  container (and recursion spawns more), and warm session envs linger for
  `VOMERO_SESSION_TTL`. Set **`VOMERO_MAX_CONCURRENT_RUNS`** (in-flight runs →
  HTTP 429 over the cap) and **`VOMERO_MAX_SESSIONS`** (warm envs → LRU-evicted
  over the cap). Size `MAX_CONCURRENT_RUNS` to `node_mem / per-container-mem`,
  **not** CPU — runs spend most of their wall-clock blocked on the model. A 429
  is the signal for the HPA/load balancer to add or pick another replica.
- **Sessions are per-replica and in-memory.** Conversation history
  (`transcript_sink`/`history`) and the `SessionEnvPool` (variables, warm
  sandboxes) live in the replica that handled the turn. For multi-replica
  deployments, either enable **session affinity** (sticky routing on
  `session_id`) so a conversation returns to the same pod, or externalise
  history to Redis/SQLite and accept that REPL variables reset on reschedule.
- **`workspace_root` needs durable storage** if you want files to survive a pod
  restart — back it with a PVC, not an `emptyDir`.
- **HPA** on CPU works, but a run can spend minutes blocked on the model; prefer
  scaling on a custom "in-flight runs" metric or a queue depth.
- **Graceful shutdown:** call `pool.close_all()` on `SIGTERM` so warm sandboxes
  are torn down; set `terminationGracePeriodSeconds` long enough for in-flight
  runs (or drain them).

---

## 8. Operating it

**Concurrency.** One `RLMEngine` is thread-safe and is meant to be shared. Each
run wants its own `UsageMeter` and `Channel`. The bundled server and the FastAPI
example both run each request in its own thread — copy that shape.

**Cost control.** Two independent ceilings: `VOMERO_MAX_STEPS` (REPL steps per
loop) and `VOMERO_MAX_DEPTH` (recursion). Lower them in production. Read
`meter.calls` / `meter.total_tokens` after each run for per-request accounting;
the meter also distinguishes provider-reported vs estimated tokens
(`meter.estimated`).

**Timeouts.** Always wrap `engine.run` in a wall-clock timeout in your frontend.
A run won't exceed `MAX_STEPS` model calls, but each call's latency is the
provider's; a slow model can still tie up a worker.

**Observability.** Every `Step` flows through your `Channel`; log the `usage`,
`compaction`, and `final` events for metrics. `step_to_payload` gives you a
JSON-able dict per event.

**Statefulness recap.** Three things can persist across turns, all caller-owned:
the **conversation** (`history`/`transcript_sink`), the **REPL variables** and
**workspace files** (`SessionEnvPool` + `env=`). The engine itself stores
nothing — if you don't persist these, every question starts fresh.

**Shutdown.** If you use a `SessionEnvPool`, call `pool.stop_sweeper()` and
`pool.close_all()` on exit so warm containers are removed.
