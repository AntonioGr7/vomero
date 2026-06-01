# ADR 0001 — Execution environment: in-process now, sandbox later

Status: accepted (v0) — sandbox now implemented, see ADR 0004
Date: 2026-05-31
Update: 2026-06-01 — the sandbox foreseen below shipped as an optional gVisor
backend (`vomero/execution/sandbox/`). The "one real wrinkle" was resolved as
predicted (corpus bind-mounted read-only; helpers exposed as an RPC surface).
In-process stays the default; the sandbox is opt-in. Details in ADR 0004.

## Context

The RLM design runs model-authored Python in a persistent REPL where the data
lives as a variable. That code is powerful and, in principle, untrusted (it is
written by an LLM). We need an execution backend.

## Decision

Ship an **in-process** backend (`exec` in a persistent dict namespace,
`vomero/execution/inprocess.py`) for v0, behind the `ExecutionEnvironment` ABC
(`vomero/execution/base.py`). The engine depends only on that ABC, so the backend is
swappable.

This is acceptable because v0 is a **personal, trusted, single-user tool**.

## The swap plan (when we sandbox)

A sandboxed backend (subprocess, container, microVM, or Pyodide/WASM) must
implement the same `inject(**names)` + `execute(code) -> ExecResult` contract.

The one real wrinkle, recorded here so it isn't a surprise:

- In-process, we inject **live Python callables** (`llm`, `rlm`, `answer`) and a
  live `corpus` object directly into the namespace. Across a process/sandbox
  boundary that no longer works — closures and objects don't cross.
- Resolution: expose `llm` / `rlm` / `answer` / `corpus.*` to sandboxed code as
  an **RPC surface** (a stub module inside the sandbox that calls back to the
  host over a pipe/socket), while file *reads* are mediated by the host so the
  sandbox only sees the mounted corpus. Keep the stub's API identical to the
  in-process names so the system prompt and model behavior don't change.

So: design helpers as if they may become RPC (small, serializable args/returns;
strings in, strings out). They already are.

## Consequences

- v0 is fast to build and fully capable.
- Anyone running Vomero on untrusted data must understand code executes with the
  host's privileges until the sandbox lands. Documented in the README.
