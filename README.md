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
  env/        swappable execution backend (in-process now)             [ADR 0001]
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
  vLLM, LM Studio, OpenRouter, local). Anthropic/Gemini slot in behind the same
  `LLMClient` protocol — see [ADR 0002](docs/adr/0002-model-provider-abstraction.md).
- **Execution:** in-process `exec` (fast, full-power, **not sandboxed** — trusted
  personal use only). The `ExecutionEnvironment` interface is the seam where a
  sandbox lands later — see [ADR 0001](docs/adr/0001-execution-environment.md).

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

## Roadmap (next)

- Interactive `vomero chat` (multi-turn, persistent REPL across questions).
- Output truncation at the `execute` boundary (a single oversized tool result
  can't be reclaimed by compaction — it stays in the protected tail).
- Binary/large-file handling (PDF, parquet) via lazy adapters on `corpus`.
- Sandboxed execution backend (ADR 0001) once running on untrusted data.
- Caching of `llm()`/`rlm()` sub-answers to cut cost on repeated sub-questions.
