# ADR 0004 — Optional gVisor sandbox for the execution environment

Status: accepted
Date: 2026-06-01

## Context

ADR 0001 shipped an in-process `exec` backend and named the wrinkle a real
sandbox would hit: the model's REPL is injected with **live Python callables**
(`llm`, `rlm`, `answer`, `ask_user`, `ask_parent`, `todo`) and a live `corpus`,
none of which cross a process boundary. We now want a sandbox so Vomero can run
untrusted, model-authored code (e.g. behind the server) without giving it the
host's filesystem, network and privileges — while staying fast enough to be
usable, and without forcing it on local/dev users.

## Decision

Add an **optional** `SandboxEnvironment` (in `vomero/execution/sandbox/`) that
satisfies the same `inject()` + `execute()` contract as `InProcessEnvironment`.
The default backend is unchanged; the sandbox is opt-in via
`VOMERO_SANDBOX=1` / `VOMERO_EXEC_BACKEND=sandbox` / `--sandbox`.

**Isolation: gVisor via Docker.** Each run gets a container launched with
`docker run --runtime=runsc`. gVisor intercepts syscalls in userspace, giving
VM-grade isolation at near-container speed — the SOTA point on the
security/latency curve for running untrusted code. We go through Docker (rather
than driving `runsc` + OCI bundles directly) because it gives us mounts, cgroup
limits and lifecycle for free, and is how gVisor is normally deployed.

**Resource limits (the headline knob).** Per-container caps map straight onto
Docker flags, configurable via settings/CLI:
`--memory` (hard cap, swap disabled), `--cpus` (fractional vCPUs),
`--pids-limit` (fork-bomb guard), `--network=none`, a small writable `/tmp`
tmpfs over a `--read-only` rootfs, `--cap-drop=ALL`, `--security-opt
no-new-privileges`, and `--user <host uid:gid>` (code runs non-root; the
read-only corpus mount stays readable with the host user's own permissions).

**The wrinkle, resolved (RPC + a read-only mount).**

- `corpus` is **bind-mounted read-only** into the container and a real `Corpus`
  is reconstructed over the mount inside the sandbox. So `grep`/`read`/`peek`
  run locally at full speed — no per-call round trip. This is safe: the sandbox
  protects the *host* from the model's code; the corpus is data the model is
  allowed to read anyway. (Scoping via `subset`'s allow-list is preserved.)
- `llm` / `rlm` / `answer` / `ask_user` / `ask_parent` and the `todo` surface
  become **RPC stubs**. Calling one sends a request over a Unix-domain control
  socket (a shared bind-mounted dir) back to the host, which runs the real
  callable (it closes over the engine, meter, channel, recursion) and returns
  the result. The stub API is identical to the in-process names, so the system
  prompt and the model's behavior don't change.

**Speed.** The container is started once per run and reused across every REPL
step (gVisor startup is paid once, then amortized over the loop). The control
channel is length-prefixed JSON; both ends are strictly synchronous
request/response, so framing is trivial and there's no multiplexing overhead.

**Module isolation.** Everything lives in `vomero/execution/sandbox/`. Only the host
side (`environment.py`) is imported by the rest of Vomero, lazily, from
`build_env_factory`. `agent.py` runs *inside* the container and is standalone
(stdlib + the bind-mounted `protocol.py` and `corpus.py` loaded by path), so it
runs on a stock `python:3.11-slim` with nothing installed.

## Consequences

- Running the sandbox needs Docker with the `runsc` runtime registered. Without
  it, the default in-process backend is unaffected; selecting the sandbox
  surfaces a clear startup error.
- One container per run/recursion: an `rlm()` sub-call spins up its own
  container (its own isolation + reused across the sub-loop's steps). Correct
  and isolated; a warm pool can come later if startup latency matters.
- An OOM-kill (exceeding `--memory`) tears the container down mid-run; the step
  returns a clear error and subsequent steps in that run will fail. That's the
  intended hard limit, not a soft throttle.
- The image needs only Python. Code needing third-party libs points
  `VOMERO_SANDBOX_IMAGE` at an image that has them.
- Tests: the host<->agent protocol, RPC, persistence, corpus access and
  teardown are covered with a `runner="local"` seam (agent as a plain
  subprocess, no Docker). The full gVisor path is a Docker-gated test
  (`VOMERO_TEST_SANDBOX_DOCKER=1`).
```
