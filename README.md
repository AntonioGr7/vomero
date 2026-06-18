# Vomero

A coding/data assistant built on the **Recursive Language Model (RLM)** idea:
instead of retrieving chunks and stuffing them into the prompt (RAG), the data
lives as a **variable inside a Python REPL**. The model writes code to explore,
grep and slice it, distills chunks with fresh sub-model calls, and recurses on
self-contained sub-questions. **Raw content never enters the root model's
context** — only what the model deliberately pulls in.

This targets the cases RAG and plain long-context handle poorly: multi-hop
questions, full-corpus aggregation, exact (not fuzzy) lookups, and inputs far
larger than the model's context window.

```
question ─▶ RLMEngine ─▶ root model ──(python tool)──▶ REPL
                                                       ├─ corpus / context   navigate the data
                                                       ├─ llm(chunk)         flat distillation
                                                       ├─ llm_batched([...]) parallel distillation
                                                       ├─ rlm(subq)          recursive sub-call
                                                       └─ answer(value)      finish
```

The model acts through exactly **one tool: run Python**. Everything else —
navigation, distillation, recursion, finishing — is a function in the REPL.

## The two data modes

The model navigates a **`Source`** ([src/vomero/context/source.py](src/vomero/context/source.py)).
There are two, behind one seam, so recursion / compaction / planning / sandbox
all work identically over either:

| Mode | What it is | Mount it with |
|---|---|---|
| **`Corpus`** | a folder of files on disk | `--data ./folder` |
| **`Context`** | an in-memory blob (a string, or a list of documents) held as a REPL **variable** — the canonical RLM surface | `--text PATH` (or `--text -` for stdin) |

```bash
# Folder corpus:
vomero ask "What blocks P-BEACON, and which team owns the fix?" \
  --data examples/sample_corpus -v

# In-memory context (the context-as-a-variable case):
vomero ask "Summarize the key risks." --text ./long_contract.txt
cat huge_transcript.txt | vomero ask "Who decided to ship after auth landed?" --text -
```

## The REPL surface

The model's REPL has the data handle plus a few host-backed functions:

```
corpus / context   The data, navigated lazily (never dumped into context).
    Corpus:   overview() · tree() · files(glob) · grep(pat) · search(query,k) · peek(path) · read(path) · size(path) · subset([paths])
    Context:  overview() · len() · n_docs · peek(doc) · read(doc) · slice(a,b) · grep(pat) · search(query,k) · docs_matching(pat) · chunk(size,overlap) · subset(sel)

llm(text, system=None) -> str
    A fresh, memoryless sub-call. Distill one chunk into a variable. The chunk
    you pass is the ONLY thing it sees.

llm_batched(texts, system=None) -> list[str]
    Like llm(), but runs many distillations CONCURRENTLY, results in order — the
    partition+map workhorse (chunk the data, distill all chunks in parallel).

rlm(question, scope=None) -> str
    A recursive Vomero call on the (optionally scoped) source — a sub-question
    that itself needs exploration gets the same full power. Depth is capped.

answer(value)
    Record the FINAL answer and finish. `value` may be a string OR any REPL
    variable; passing a variable means the answer is NOT limited by the model's
    output size (assemble a long result programmatically, then answer(it)).
```

`todo` (with `--plan`) and `ask_user`/`ask_parent` (interactivity) are added when
those features are enabled — see below.

## Search & retrieval

Alongside `grep` (exact substring/regex), the data handle has `search(query, k)`
— *ranked relevance* retrieval, which is what closes the recall gap on multi-hop
questions where the bridge entity isn't worded like the query. The same
`search()` method runs on top of one of three backends, chosen by how you
configure it — but the method the model calls never changes, so you can move from
the zero-setup default to a planet-scale service without touching anything the
model sees:

1. **In-memory (default, zero setup).** The first `search()` call builds a
   pure-Python **BM25** index over the data and caches it for the process. No
   dependencies, no build step, nothing to configure — it just works. Fine up to
   tens of thousands of documents. If you also set an embedding model (below),
   this same in-memory index adds dense vectors and fuses the two (hybrid).
2. **Persistent index (built once, loaded read-only).** Run `vomero index --data
   <folder> --index-dir <dir>` once: it reads every document a single time,
   writes an on-disk lexical index (SQLite FTS5) and — if an embedding model is
   set — the document vectors, embedded once and never again. Then point a run at
   it (`--index-dir <dir>`, or `VOMERO_INDEX_DIR`); it opens read-only, so a fresh
   process pays no re-read and no re-embed. This is what a serving pod mounts as a
   read-only volume. Good for large, single-deployment corpora.
3. **External retrieval service (the scalable, multi-tenant path).** Set
   `VOMERO_RETRIEVAL_URL` and `search()` delegates to your own retrieval service
   (any vector DB / search engine behind a thin HTTP/JSON endpoint). The
   documents, the index, AND the query-embedding all live in that service, so the
   Vomero process holds **no vectors and no embedding model** — its memory stays
   flat no matter how many corpora, how many documents, or how many different
   embedding models are in play (each index pins its own model on the service
   side). This is the answer to "millions of documents, many tenants": retrieval
   is external infrastructure, and Vomero stays stateless. Precedence is
   service → persistent index → in-memory, so setting the URL overrides the rest.

Under the gVisor sandbox, `search()` is the one data method delegated back to the
host over RPC (the sandbox is network-less and shouldn't hold the index);
`read`/`grep`/`peek` stay local on the read-only mount. So with a service
configured, the path is: sandbox → host (thin proxy) → your retrieval service.

**You can ignore all of this.** `search()` is purely additive. Vomero works
exactly as before without it: the model can navigate with `overview`/`grep`/
`peek`/`read`/`slice` alone, and `llm`/`rlm`/`answer` are unchanged. You don't
have to build an index, set an embedding model, or run a service — none of the
three backends above is required for Vomero to answer questions. Ranked `search`
is there when you want better recall; the original grep-and-read workflow is
untouched.

### Using BM25 alone (no embeddings, no service)

BM25-only is the **default** — it's what you get when no embedding model is
configured, so you don't need to do anything special:

```python
from vomero.context import Corpus
corpus = Corpus("mydata")            # no embedder, no index_dir, no backend
hits = corpus.search("who acquired the cryptocurrency exchange", k=5)
#   -> [Hit(doc, score, snippet), ...]   ranked by BM25, pure-Python, no setup
```

To be explicit (or to force lexical even when an embedding model *is* configured),
pass `mode="lexical"`:

```python
hits = corpus.search("...", k=5, mode="lexical")   # BM25 only, ignores embeddings
```

The modes are `"lexical"` (BM25 only), `"dense"` (embeddings only — needs an
embedder/service), and `"hybrid"` (both, fused; the default, and it falls back to
pure BM25 when no embedder is configured). So: leave `VOMERO_EMBEDDING_MODEL`
unset (and don't pass `--index-dir`/`VOMERO_RETRIEVAL_URL`) and every `search()`
is plain BM25 over the in-memory index.

## Setup

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env   # then set OPENAI_API_KEY (or a local base_url)
```

Provider backends (behind one `LLMClient` protocol):

- **OpenAI-compatible** — OpenAI, or any compatible server (vLLM, LM Studio,
  OpenRouter, Together, local) via `VOMERO_BASE_URL`.
- **Gemini** — `VOMERO_PROVIDER=gemini`, via its OpenAI-compatible endpoint.
- Anthropic slots in behind the same protocol.

```bash
uv run pytest        # full suite, no API key required (scripted fake client)
```

## From Python

The CLI is a thin shell over the library — drive the engine directly in three
lines (the `dspy.RLM(...)` equivalent):

```python
from vomero import build_engine, Context

engine = build_engine(model="gpt-4o-mini")          # wires client + backend + compaction
print(engine.run("What are the key risks?", Context(open("contract.txt").read())))
```

`run(..., return_trajectory=True)` returns a `RunResult` (answer + per-step
trajectory + cost) instead of a string. Full programmatic guide — sources,
streaming progress, multi-turn, eval/optimize — in **[docs/library.md](docs/library.md)**.

## Token accounting & budgets

Every model call is metered. Two distinct figures:

- **context size** — tokens in the *current* message stack (the live gauge; it
  drops when history is compacted), from the provider's reported prompt-token
  count for the latest call.
- **cumulative tokens** — every prompt + completion token spent since the run
  started, summed across the root loop and all recursive `llm()`/`rlm()` calls.

With `-v`, each step prints `ctx … tok | total … tok`; a `[usage]` summary line
is always written to stderr. Providers that don't report usage fall back to a
char-based estimate, flagged with `~`. Usage is read from a **caller-owned
`UsageMeter`** (`engine.run(..., meter=m)`), not from engine state — so one
engine instance serves concurrent runs safely.

Guardrails:

- **Global budget across the whole run tree** — `--max-total-tokens` /
  `--max-total-calls` (or `VOMERO_MAX_TOTAL_TOKENS` / `…_CALLS`) cap total spend
  across the root loop *and* every recursive sub-call (the meter is shared down
  the tree). On hitting a limit the run stops spawning work, skips further
  sub-calls, and returns its best effort. Both default to 0 = unlimited.
- **Per-result output cap** — `--max-output-chars` (default 10k) head/tail-
  truncates any single tool result before it enters the transcript, so one
  oversized `print` can't permanently bloat the protected recent tail.

## Compaction

When projected context crosses `--compact-ratio` of `--context-window` (default
0.8 of 128k), the loop summarizes the *middle* of the transcript and continues —
the context gauge visibly drops. The design:

- **preamble + recent tail kept verbatim**; only the middle is summarized, and
  the tail boundary is snapped so it never orphans a tool result;
- the summary is a faithful, structured distillation (Task / Progress / Key
  findings / Open threads) that **merges** prior summaries (idempotent);
- because the REPL namespace survives compaction, the summary lists the
  **still-defined variables** so the model reuses them instead of recomputing;
- compaction is **vetoed** when the reclaimable middle is too small.

```bash
vomero ask "…" --data ./data --context-window 32000 --compact-ratio 0.75 -v
vomero ask "…" --data ./data --no-compact     # disable
```

## Planning (live TODO checklist)

With `--plan`, the model maintains a plan you watch tick off in real time. It
drives a `todo` surface from the REPL (`todo.plan([...])`, `todo.start(n)`,
`todo.complete(n)`, `todo.add(...)`), and each change reprints the checklist:

```
Plan (1/3 done):
  ✔ Locate P-BEACON file
  ▶ Find its blocker
  ☐ Identify owning team
```

Opt-in (`--plan` or `VOMERO_PLAN=1`), pairs with `-v`. The checklist is host-side
observability — kept out of the model's context, so it costs no tokens. Each
recursive sub-agent keeps its own plan (indented by depth); `--plan-root-only`
gives the surface to the root alone.

## Interactivity (ask the user, or the parent)

The model can ask for help when genuinely stuck. It calls `ask_user(question)`
from the REPL; the loop pauses, prompts you on the terminal, and feeds your reply
back as the return value:

```
❓ The assistant needs your input:
   P-BEACON is blocked by P-ATLAS — recommend waiting, or propose a workaround?
   > ship after the auth library lands
```

On by default on a terminal; auto-disabled when stdin is piped; `--no-interactive`
(or `VOMERO_INTERACTIVE=0`) turns it off. Headless is safe — `ask_user` returns a
"no user available, proceed with best judgment" reply instead of hanging. The
model is told to ask *sparingly*. `--ask-root-only` restricts the human prompt to
the root agent.

**Sub-agents consult their parent first.** A recursive `rlm()` sub-agent is
isolated, so before escalating to a human it can call `ask_parent(question)` —
answered by a one-shot completion over the *parent's* live context. The exchange
is recorded in the **sub-agent's** history but never the **parent's** (the parent
answers from a copy), preserving the isolation that keeps the root small. Being
model-to-model, it works even headless/piped.

## Evaluating (RLM vs. baseline vs. a leakage control)

`vomero eval` measures correctness (exact-match / token-F1 / contains / optional
LLM-judge) and cost (tokens, latency) across up to three arms over the **same**
questions:

| `--mode` arm | What it does | Why |
|---|---|---|
| `rlm` | the Vomero engine | the system under test |
| `baseline` | paste the whole source into one prompt, ask once | the long-context control RLM must beat; on data bigger than its window it truncates (flagged) |
| `closed_book` | answer with **no context at all** | the **contamination control** — its score is the parametric-memory floor |

```bash
# Synthetic needle-in-a-haystack — leakage-proof (invented facts at known depths):
vomero eval --benchmark needle --limit 20 --mode all --judge

# MultiHopRAG (needs data/download_corpus.py + MultiHopRAG.json):
vomero eval --benchmark multihoprag --limit 50 --mode all --judge

# Any JSONL of {question, answer, context} rows:
vomero eval --jsonl my_qa.jsonl --mode both
```

Key things the harness does for honest numbers:

- **Closed-book control.** On benchmarks built from public text, a frontier model
  may already *know* the answers. `closed_book` quantifies that; the real context
  contribution of any arm is `that arm − closed_book`. If `closed_book ≈ baseline`,
  the benchmark isn't testing retrieval at all.
- **Leakage-proof benchmark.** `--benchmark needle` injects invented facts
  (random vault codes) at known depths into a haystack sized to overflow the
  window. Closed-book *must* score ~0; a truncating baseline fails the deep
  needles; an RLM that greps should find them at any depth.
- **Regime line.** Up front it prints whether the data fits or **OVERFLOWS** the
  baseline's window — if it fits, the baseline can win without RLM ever helping.
- **Fair string metrics.** Both arms are told to answer with a short span by
  default (gold answers are short), so EM/F1 aren't punishing verbosity; disable
  with `--no-terse`.

Eval runs **in-process by default** (fast, and the only backend that can hold an
in-memory needle context); add `--sandbox` for folder-corpus evals. See
[src/vomero/eval/](src/vomero/eval/) — adapters, runners, metrics, optimizer.

## Tuning the prompt to a metric

`engine.run(..., return_trajectory=True)` returns a `RunResult` (answer +
per-step trajectory + cost) for inspection. On top of that, a native prompt
optimizer ([src/vomero/eval/optimize.py](src/vomero/eval/optimize.py)) searches
candidate instruction blocks (the engine's `extra_instructions`) and keeps the
one that maximizes an eval metric on a train set — the DSPy idea (a *measured*
prompt), built on the eval harness with no extra dependency:

```python
from vomero.eval import optimize, propose_instructions
cands = [None] + propose_instructions(client, n=4)      # baseline + model-proposed
result = optimize(engine, train_items, cands, metric="f1", judge_client=client)
print(result.summary())   # engine is left configured with the winning block
```

## Sandboxed execution (gVisor)

By default the model's code runs **in-process** with `exec` — fast and
full-power, but it can touch the host's filesystem and network (fine for trusted
local/dev use). For untrusted input, the **gVisor sandbox** runs each step inside
an isolated `runsc` container with hard memory/CPU caps, no network, a read-only
corpus, non-root, and no host filesystem access. Off by default, opt-in per run.

> The sandbox mounts a **folder corpus** only; in-memory `--text` context runs use
> the in-process backend (see Roadmap).

### Prerequisites

Docker plus the [`runsc` (gVisor) runtime](https://gvisor.dev/docs/user_guide/install/)
registered with the daemon:

```bash
sudo runsc install          # writes the "runsc" runtime into /etc/docker/daemon.json
sudo systemctl restart docker
docker run --rm --runtime=runsc hello-world   # verify
```

The default image `python:3.11-slim` is pulled on first use.

### Run it

```bash
vomero ask "What blocks P-BEACON?" --data ./data \
  --sandbox --sandbox-memory 1g --sandbox-cpus 2

VOMERO_SANDBOX=1 VOMERO_SANDBOX_MEMORY=1g \
  vomero ask "..." --data ./data           # via env (also applies to `serve`)
```

| Env var | CLI flag | Default | What it does |
|---|---|---|---|
| `VOMERO_SANDBOX=1` | `--sandbox` | off | Shortcut for `VOMERO_EXEC_BACKEND=sandbox` |
| `VOMERO_EXEC_BACKEND` | — | `inprocess` | `inprocess` or `sandbox` |
| `VOMERO_SANDBOX_MEMORY` | `--sandbox-memory` | `512m` | Hard memory cap |
| `VOMERO_SANDBOX_CPUS` | `--sandbox-cpus` | `1.0` | Fractional vCPUs |
| `VOMERO_SANDBOX_IMAGE` | `--sandbox-image` | `python:3.11-slim` | Container image |
| `VOMERO_SANDBOX_RUNTIME` | `--sandbox-runtime` | `runsc` | OCI runtime (gVisor) |
| `VOMERO_SANDBOX_NETWORK` | — | `none` | Docker `--network` |
| `VOMERO_SANDBOX_PIDS` | — | `256` | Max processes (fork-bomb guard) |
| `VOMERO_SANDBOX_STARTUP_TIMEOUT` | — | `60` | Seconds to wait for the container |

**Third-party libraries:** the default image is stdlib-only. Point the sandbox at
an image with your deps (Vomero's own source is bind-mounted at runtime):

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir pandas numpy
```
```bash
docker build -t vomero-sandbox -f Dockerfile.sandbox .
VOMERO_SANDBOX=1 VOMERO_SANDBOX_IMAGE=vomero-sandbox vomero ask "..." --data ./data
```

**How it works (Vomero launches the agent, not you).** On the first line of code:
Vomero opens a Unix control socket and runs `docker run --runtime=runsc … agent.py`,
bind-mounting your corpus **read-only** at `/corpus`. The in-container agent
connects back, rebuilds `corpus` over the mount, and turns
`llm`/`rlm`/`answer`/`ask_user`/`todo` into **RPC stubs** back to the host (which
holds the real engine/LLM/recursion). The corpus is local, so `grep`/`read` stay
fast. The container is reused for every step (gVisor startup paid once) and torn
down at the end; each `rlm()` sub-call gets its own container. Exceeding the
memory cap OOM-kills the container mid-step, which returns a clear error.

## Frontends (the Channel seam) & serving

The engine doesn't know whether it's talking to a shell, a test, or a browser —
it depends only on a **`Channel`** ([src/vomero/channel.py](src/vomero/channel.py)):

```python
class Channel(Protocol):
    def emit(self, step: Step) -> None: ...        # progress / usage / plan events
    def ask_user(self, question: str) -> str: ...  # reach the human (may block)
```

Built-ins: `NullChannel` (drops events; the safe default) and `CallbackChannel`
(CLI printers + terminal prompt). A working HTTP/SSE frontend ships in the box:

```bash
vomero serve --data examples/sample_corpus --port 8000
# then open examples/browser_client.html, or curl the API
```

To put the RLM behind any other frontend, implement a `Channel` that serializes
each `Step` and pushes it over your transport, and whose `ask_user` blocks on a
reply queue — then run `engine.run(...)` in a worker thread per session. No engine
changes required. See **[docs/serving.md](docs/serving.md)** for the protocol and
client examples.

## Deploying it as a service

Embedding Vomero in your own backend, the full environment/configuration
reference, multi-turn sessions, and how to run it **safely** as a microservice
(including on Kubernetes): see **[docs/deployment.md](docs/deployment.md)**.

> Before exposing this to untrusted input, turn on the gVisor sandbox
> (`VOMERO_SANDBOX=1`) — by default the model's code runs in-process with `exec`.

## Layout

```
src/vomero/
  llm/        provider-agnostic client (base protocol + OpenAI + Gemini)
  execution/  swappable backend behind one seam: in-process + gVisor sandbox
  context/    Source seam — Corpus (folder) and Context (in-memory variable)
  engine/     RLMEngine — the recursive REPL loop, compaction, system prompt
  eval/       eval harness: runners, metrics, datasets/benchmarks, optimizer
  config.py   env-driven settings
  cli.py      `vomero ask` / `serve` / `eval`
examples/sample_corpus/   tiny interlinked demo data for multi-hop
tests/        runs without an API key (scripted fake client)
```

## Roadmap (next)

- In-memory `Context` under the gVisor sandbox (today the sandbox mounts a folder
  corpus only; context-as-a-variable runs use the in-process backend).
- Binary/large-file handling (PDF, parquet) via lazy adapters on the source.
- Caching of `llm()`/`rlm()` sub-answers to cut cost on repeated sub-questions.
- Prompt/prefix caching at the provider client to cut per-step cost.
- Per-depth accuracy breakdown in the needle eval (the classic depth curve).
