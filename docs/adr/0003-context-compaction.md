# ADR 0003 — Context metering and history compaction

Status: accepted (v0)
Date: 2026-05-31

## Context

The RLM loop appends an assistant message + a tool result every step. Over a
long exploration the message stack grows until it approaches the model's context
window — at which point calls get slow, expensive, and eventually fail. We need
to (a) see how full the context is and (b) shed history without losing
task-critical state. The RLM premise *helps* here (raw file text is supposed to
stay out of the root context via `llm()`/`rlm()`), but nothing enforced it, and
tool outputs still accumulate.

## Decision

Two cooperating pieces, both provider-agnostic (`vomero/usage.py`,
`vomero/engine/compaction.py`).

### Metering

Track two distinct figures, never conflated:

- **context size** — the prompt-token count of the most recent call (exactly the
  size of what we just sent). This is the live "how full" gauge; it *drops*
  after compaction. Authoritative from the provider's reported usage; estimated
  (~4 chars/token) only when a provider omits usage.
- **cumulative tokens** — every prompt + completion token since the run started,
  summed across the root loop and all recursive `llm()`/`rlm()` calls. Includes
  the compaction summarizer calls. Only ever rises.

`estimated` is tracked per-reading for context and per-run for cumulative, so an
exact context reading is never mislabeled `~` just because an earlier call was
estimated.

### Compaction

When projected context (last authoritative size + this step's freshly appended
output) crosses `ratio * context_window` (default 0.8):

1. **Keep the preamble verbatim** — system prompt + original question.
2. **Keep the recent tail verbatim** — the last `keep_recent_messages`, snapped
   forward off any leading `tool` message so the replayed tail never starts with
   an orphaned tool result (an instant chat-API 400).
3. **Summarize only the middle** — rendered to a flat transcript (no pairing
   constraints) and distilled by a dedicated model call into Task / Progress /
   Key findings / Open threads, instructed to preserve exact paths/values and to
   *merge* any earlier summary (idempotent across repeated compactions).
4. **Preserve REPL state** — because the execution namespace survives
   compaction, the summary appends the still-defined variables
   (`env.describe_state()`) so the model reuses them instead of recomputing.

### Don't compact when it won't help

Triggering is based on *total* size, but the preamble and recent tail are
protected. If the bloat lives there (e.g. one large recent tool output), the
summarizable middle is small and compaction would spend a model call to reclaim
almost nothing — and can even grow the context. So compaction is vetoed unless
the estimated middle exceeds `min_reclaim_tokens` (default 2048). The threshold
fires again once enough reclaimable history accumulates in the middle.

## Consequences

- Compaction quality depends on the summarizer model; the prompt is tuned for
  faithfulness (preserve specifics, don't invent) but a weak model can still
  drop detail. The verbatim recent tail bounds the blast radius.
- A single tool output larger than the window cannot be compacted away (it stays
  in the protected tail). Output truncation at the `execute` boundary is the
  natural follow-up and is noted as future work.
- Estimates are clearly flagged (`~`); the OpenAI path is exact.
- `compact_ratio <= 0` disables compaction entirely (small corpora, tests).
