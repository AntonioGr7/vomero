# Vomero

A personal coding/data assistant built on the **Recursive Language Model (RLM)**
idea: instead of retrieving chunks and stuffing them into the prompt (RAG), the
data lives as a `corpus` **variable inside a Python REPL**. The model writes code
to explore, grep and slice it, and delegates heavy reading to recursive
sub-model calls. Raw content never enters the root model's context.

This targets the cases RAG handles poorly: multi-hop questions, full-corpus
aggregation, and tasks that need exact (not fuzzy) lookups.

## How it works

```
question ─▶ RLMEngine ─▶ root model ──(python tool)──▶ REPL
                                                       ├─ corpus.grep/peek/read/...
                                                       ├─ llm(chunk)   flat distillation
                                                       ├─ rlm(subq)    recursive sub-call
                                                       └─ answer(text) finish
```

The model only ever acts through one tool: **run Python**. Inside the REPL it
has `corpus` (lazy, read-only handle on the folder), `llm()` (a fresh, memoryless
sub-call to distill a chunk), `rlm()` (a recursive Vomero call on a scoped
sub-corpus), and `answer()`.

## Layout

```
src/vomero/
  llm/        provider-agnostic client (base protocol + OpenAI impl)   [ADR 0002]
  execution/  swappable execution backend: in-process + gVisor sandbox [ADR 0001/0004]
  context/    Corpus — lazy navigable view over the data folder
  engine/     RLMEngine — the recursive REPL loop  + system prompt
  config.py   env-driven settings
  cli.py      `vomero ask`
docs/adr/     architecture decisions (provider swap, sandbox swap)
examples/sample_corpus/   tiny interlinked demo data for multi-hop
tests/        runs without an API key (scripted fake client)
```

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env   # then set OPENAI_API_KEY (or a local base_url)
```

## Use

```bash
# Ask against the bundled demo corpus, streaming the model's reasoning:
vomero ask "What blocks P-BEACON, and which team owns the fix?" \
  --data examples/sample_corpus -v

# Or point it at your own folder:
vomero ask "..." --data ./data
```

## Run tests

```bash
uv run pytest        # no API key required
```

## Status & decisions (v0)

- **Backend:** OpenAI-compatible (works with OpenAI and any compatible server —
  vLLM, LM Studio, OpenRouter, local) and **Gemini** (`VOMERO_PROVIDER=gemini`,
  via its OpenAI-compatible endpoint). Anthropic slots in behind the same
  `LLMClient` protocol — see [ADR 0002](docs/adr/0002-model-provider-abstraction.md).
- **Execution:** two backends behind one `ExecutionEnvironment` seam.
  - *in-process* `exec` (default) — fast, full-power, **not sandboxed**; for
    trusted local/dev use. See [ADR 0001](docs/adr/0001-execution-environment.md).
  - *gVisor sandbox* (opt-in) — runs each step in a `runsc` container with hard
    memory/CPU caps, no network, read-only corpus, non-root, no host filesystem
    access. The host-stateful helpers (`llm`/`rlm`/`answer`/…) stay available
    via an RPC surface. See [ADR 0004](docs/adr/0004-gvisor-sandbox.md).

### Sandboxed execution (gVisor)

By default the model's code runs in-process with `exec` — fast, but it can touch
the host's filesystem and network. For untrusted input (or just to be safe), the
**gVisor sandbox** runs each step inside an isolated `runsc` container with hard
memory/CPU caps, no network, a read-only corpus, and no host filesystem access.
It's **off by default** and opt-in per run.

#### 1. Prerequisites

You need Docker, plus the [`runsc` (gVisor) runtime](https://gvisor.dev/docs/user_guide/install/)
registered with the Docker daemon:

```bash
# Install gVisor (see the link above for your platform), then register it:
sudo runsc install          # writes the "runsc" runtime into /etc/docker/daemon.json
sudo systemctl restart docker

# Verify Docker can see it:
docker run --rm --runtime=runsc hello-world
```

The default image is `python:3.11-slim`; it's pulled automatically on first use
(so the first run may take longer — see `VOMERO_SANDBOX_STARTUP_TIMEOUT`).

#### 2. Run it

```bash
# Turn it on per-run and size the container:
vomero ask "What blocks P-BEACON?" --data ./data \
  --sandbox --sandbox-memory 1g --sandbox-cpus 2

# Or via the environment — also applies to `vomero serve`:
VOMERO_SANDBOX=1 VOMERO_SANDBOX_MEMORY=1g VOMERO_SANDBOX_CPUS=2 \
  vomero ask "..." --data ./data

# Serve a corpus with isolated execution (the warning banner flips to 🔒):
VOMERO_SANDBOX=1 vomero serve --data ./data --port 8000
```

CLI flags: `--sandbox`, `--sandbox-memory`, `--sandbox-cpus`, `--sandbox-image`,
`--sandbox-runtime`. They override the environment for that run.

#### 3. Configuration

| Env var | CLI flag | Default | What it does |
|---|---|---|---|
| `VOMERO_SANDBOX=1` | `--sandbox` | off | Shortcut for `VOMERO_EXEC_BACKEND=sandbox` |
| `VOMERO_EXEC_BACKEND` | — | `inprocess` | `inprocess` or `sandbox` |
| `VOMERO_SANDBOX_MEMORY` | `--sandbox-memory` | `512m` | Hard memory cap (e.g. `1g`, `2g`) |
| `VOMERO_SANDBOX_CPUS` | `--sandbox-cpus` | `1.0` | Fractional vCPUs (e.g. `0.5`, `2`) |
| `VOMERO_SANDBOX_IMAGE` | `--sandbox-image` | `python:3.11-slim` | Container image |
| `VOMERO_SANDBOX_RUNTIME` | `--sandbox-runtime` | `runsc` | OCI runtime (gVisor) |
| `VOMERO_SANDBOX_NETWORK` | — | `none` | Docker `--network` |
| `VOMERO_SANDBOX_PIDS` | — | `256` | Max processes (fork-bomb guard) |
| `VOMERO_SANDBOX_STARTUP_TIMEOUT` | — | `60` | Seconds to wait for the container (first run pulls the image) |

#### Code that needs third-party libraries

The default image only has Python's stdlib. If the model's code needs `pandas`,
`numpy`, etc., point the sandbox at an image that has them — Vomero's own source
is bind-mounted in at runtime, so the image needs **only** Python plus your deps:

```dockerfile
# Dockerfile.sandbox
FROM python:3.11-slim
RUN pip install --no-cache-dir pandas numpy
```

```bash
docker build -t vomero-sandbox -f Dockerfile.sandbox .
VOMERO_SANDBOX=1 VOMERO_SANDBOX_IMAGE=vomero-sandbox \
  vomero ask "..." --data ./data
```

#### How it works (you don't launch the agent — Vomero does)

There's no separate process to start: the container and the in-sandbox **agent**
are launched automatically, lazily, on the first line of code the model runs.
Per run:

1. Vomero opens a Unix control socket on a temp dir and runs
   `docker run --runtime=runsc … python …/agent.py <socket> <corpus.py>`,
   bind-mounting your corpus **read-only** at `/corpus`, the agent + `corpus.py`,
   and the socket dir. It runs as your host `uid:gid` (non-root in the container).
2. The agent connects back, rebuilds `corpus` locally over the mount, and turns
   `llm`/`rlm`/`answer`/`ask_user`/`todo` into **RPC stubs**.
3. Each REPL step ships code over the socket; the agent `exec`s it and streams
   stdout/errors back. When the code calls a helper, it round-trips to the host
   (which holds the real engine/LLM/recursion) and returns the result. The
   corpus is local, so `grep`/`read` stay fast.
4. The container is **reused for every step of the run** (gVisor startup paid
   once) and torn down when the run ends. An `rlm()` sub-call gets its own
   isolated container.

> Exceeding `--sandbox-memory` OOM-kills the container mid-run (a hard cap, by
> design); the step returns a clear error. Full design in
> [ADR 0004](docs/adr/0004-gvisor-sandbox.md).

## Token accounting

Every model call is metered, so you can see how "blown" the context is and what
the run has cost. Two distinct figures:

- **context size** — tokens in the *current* message stack (the live gauge; it
  drops when history is compacted). Sourced from the provider's reported
  prompt-token count for the latest call.
- **cumulative tokens** — every prompt + completion token spent since the run
  started, summed across the root loop and all recursive `llm()`/`rlm()` calls.

With `-v`, each step prints `ctx … tok | total … tok`; a `[usage]` summary line
is always written to stderr. Providers that don't report usage fall back to a
char-based estimate, flagged with `~`. Read totals programmatically off
`engine.last_usage` after `run()` returns.

## Compaction

When projected context crosses `--compact-ratio` of `--context-window` (default
0.8 of 128k), the loop summarizes the *middle* of the transcript and continues —
the context gauge above visibly drops. The design (see
[ADR 0003](docs/adr/0003-context-compaction.md)):

- **preamble + recent tail kept verbatim**; only the middle is summarized, and
  the tail boundary is snapped so it never orphans a tool result;
- the summary is a faithful, structured distillation (Task / Progress / Key
  findings / Open threads) that **merges** prior summaries (idempotent);
- because the REPL namespace survives compaction, the summary lists the
  **still-defined variables** so the model reuses them instead of recomputing;
- compaction is **vetoed** when the reclaimable middle is too small
  (`--compact-ratio 0`, or env knobs, to disable/tune).

```bash
vomero ask "…" --data ./data --context-window 32000 --compact-ratio 0.75 -v
vomero ask "…" --data ./data --no-compact     # disable
```

## Planning (live TODO checklist)

With `--plan`, the model maintains a plan you watch tick off in real time — the
"here's what I'm doing" view. It drives a `todo` surface from the REPL
(`todo.plan([...])`, `todo.start(n)`, `todo.complete(n)`, `todo.add(...)`), and
each change reprints the checklist:

```
Plan (1/3 done):
  ✔ Locate P-BEACON file
  ▶ Find its blocker
  ☐ Identify owning team
```

It's opt-in (off by default; `--plan` or `VOMERO_PLAN=1`) and pairs with `-v`.
The checklist is host-side observability — it's kept out of the model's own
context, so it costs no tokens in the loop. By default each recursive `rlm()`
sub-agent keeps its own plan (rendered indented by depth); pass
`--plan-root-only` (or `VOMERO_PLAN_ROOT_ONLY=1`) to give the plan surface to the
root agent alone.

```bash
vomero ask "What blocks P-BEACON, and who owns the fix?" \
  --data examples/sample_corpus --plan
```

## Interactivity (ask the user for help)

The model can ask you for help when it's genuinely stuck — an ambiguous request,
missing information only you have, or a consequential decision the data can't
resolve. It calls `ask_user(question)` from the REPL; the loop pauses, prompts
you on the terminal, and feeds your reply back as the function's return value,
which the model incorporates into its work:

```
❓ The assistant needs your input:
   P-BEACON is blocked by P-ATLAS — recommend waiting, or propose a workaround?
   > ship after the auth library lands
```

On by default when running on a terminal; auto-disabled when stdin is piped, and
`--no-interactive` (or `VOMERO_INTERACTIVE=0`) turns it off. Headless (no
prompter) is safe: `ask_user` returns a "no user available, proceed with best
judgment" reply instead of hanging. The model is told to ask *sparingly* —
explore the corpus first, ask only when proceeding would mean guessing.

By default any depth may ask the user; `--ask-root-only` (or
`VOMERO_ASK_ROOT_ONLY=1`) restricts the human prompt to the root agent.

**Sub-agents consult their parent first.** A recursive `rlm()` sub-agent is
isolated, so it may lack context the parent had. Before escalating to the human
it can call `ask_parent(question)` — the engine answers with a one-shot
completion over the *parent's* live context (which reflects compaction), so the
delegating agent clarifies intent/scope without involving you. The model is told
to prefer `ask_parent` for anything about the task's intent, and `ask_user` only
for things the parent wouldn't know either. With `--ask-root-only`, sub-agents
lose `ask_user` but keep `ask_parent`.

Since it's model-to-model (no human), `ask_parent` works even headless/piped.
The exchange is recorded in the **sub-agent's** history (so it remembers the
clarification on later steps) but never in the **parent's** — the parent answers
from a copy of its context, preserving the isolation that keeps the root small.

## Frontends (the Channel seam)

The engine doesn't know whether it's talking to a shell, a test, or a browser —
it depends only on a **`Channel`** ([vomero/channel.py](src/vomero/channel.py)):

```python
class Channel(Protocol):
    def emit(self, step: Step) -> None: ...      # progress / usage / plan events
    def ask_user(self, question: str) -> str: ... # reach the human (may block)
```

Built-ins: `NullChannel` (drops events, no human — the safe default) and
`CallbackChannel` (adapts the CLI's printers + terminal prompt). To put the RLM
behind a browser, implement a `Channel` that serializes each `Step` to JSON and
pushes it over a WebSocket, and whose `ask_user` blocks on a queue the socket
fills when the user replies — then run `engine.run(...)` in a worker thread per
session. No engine changes required.

Token usage is read from a **caller-owned `UsageMeter`** (`engine.run(...,
meter=m)`), not from engine state — so one engine instance serves concurrent
runs safely.

A working HTTP/SSE frontend ships in the box — `vomero serve` streams events to
a browser or any HTTP client and accepts human replies for `ask_user`:

```bash
vomero serve --data examples/sample_corpus --port 8000
# then open examples/browser_client.html, or curl the API
```

See **[docs/serving.md](docs/serving.md)** for the protocol, event types, and
browser / curl / Python client examples.

## Deploying it as a service

Embedding Vomero in your own backend (FastAPI, a custom socket server), the full
environment/configuration reference, multi-turn sessions, and — importantly —
how to run it **safely** as a microservice (including on Kubernetes): see
**[docs/deployment.md](docs/deployment.md)**.

> Before exposing this to a browser on untrusted input, turn on the gVisor
> sandbox (`VOMERO_SANDBOX=1`, [ADR 0004](docs/adr/0004-gvisor-sandbox.md)): by
> default the model's code runs in-process with `exec`.

## Roadmap (next)

- Interactive `vomero chat` (multi-turn, persistent REPL across questions).
- Output truncation at the `execute` boundary (a single oversized tool result
  can't be reclaimed by compaction — it stays in the protected tail).
- Binary/large-file handling (PDF, parquet) via lazy adapters on `corpus`.
- Sandbox: a warm container pool to cut per-run startup latency (ADR 0004).
- Caching of `llm()`/`rlm()` sub-answers to cut cost on repeated sub-questions.
