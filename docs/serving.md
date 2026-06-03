# Serving Vomero to a browser / external client

`vomero serve` exposes the RLM over HTTP, streaming the agent's work with
Server-Sent Events (SSE) and accepting human replies over a POST. It's the
`Channel` seam ([vomero/channel.py](../src/vomero/channel.py)) projected onto the
wire — any client that speaks HTTP can drive the agent and watch it think.

## Start the server

```bash
vomero serve --data examples/sample_corpus --port 8000
# optional: --host 0.0.0.0  --model gpt-4o
```

The server is bound to **one corpus** (chosen at startup) so clients never send
filesystem paths. It reads model/provider/keys from the environment like
`vomero ask` (see `.env.example`).

> ⚠ **Not sandboxed.** Model-authored code runs in-process with `exec`.
> Serve only trusted corpora and trusted clients until a sandboxed
> `ExecutionEnvironment` lands. Don't expose this to the open internet as-is.

## Protocol

Three endpoints. A "run" is one question; it gets a `session_id` and an event
stream.

| Method & path | Body | Returns |
|---|---|---|
| `POST /runs` | `{"question": "...", "plan"?: bool, "plan_root_only"?: bool}` | `{"session_id", "events", "reply"}` |
| `GET /runs/<id>/events` | — | `text/event-stream` (see events below) |
| `POST /runs/<id>/reply` | `{"answer": "..."}` | `{"ok": true}` |

`plan` is **per request**: send `{"plan": true}` to have that run maintain a live
TODO checklist (streamed as `todo` events); omit it to use the server's default.
`plan_root_only` (also per request) restricts the plan to the root agent.

### Events (SSE `event:` name → `data:` JSON)

Every event carries `depth` and `index`. The `type` matches the SSE event name.

| event | payload | meaning |
|---|---|---|
| `usage` | `usage: {context_tokens, cumulative_tokens, …}` | live context size + running total |
| `message` | `message` | the model's natural-language narration |
| `code` | `code` | Python the model is about to run |
| `output` | `output` | that code's (truncated) stdout/traceback |
| `llm_call` | `llm_call: {prompt, response, tokens}` | a flat `llm()` distillation sub-call |
| `todo` | `todo: [{text, status}]` | the live plan (if planning is on) |
| `interaction` | `interaction: {question, answer, kind}` | a resolved ask (`kind`: `user`/`parent`) |
| `compaction` | `compaction: {tokens_before, tokens_after, …}` | history was compacted |
| `final` | `final` | **the answer** |
| `ask_user` | `question` | agent needs you — reply via `POST .../reply` |
| `done` | `usage: {calls, total_tokens, …}` | run finished; final usage summary |
| `end` | `{}` | stream closed |
| `error` | `error` | the run failed |

Sub-agents (`rlm()`) appear with `depth > 0`; group by `depth` to render the
recursion tree.

## Browser

A complete, dependency-free client is in
[examples/browser_client.html](../examples/browser_client.html): `POST /runs`,
open an `EventSource` on the `events` URL, dispatch by event name, and on
`ask_user` `window.prompt(...)` then POST the answer. Open it after starting the
server. (For real apps, render `ask_user` as an inline form rather than a modal
prompt.)

## curl

```bash
# start a run
SID=$(curl -s localhost:8000/runs -d '{"question":"What blocks P-BEACON?"}' | jq -r .session_id)

# stream events (in another shell)
curl -N localhost:8000/runs/$SID/events

# if you see an ask_user event, answer it:
curl -s localhost:8000/runs/$SID/reply -d '{"answer":"focus on the auth library"}'
```

## Python client

```python
import json, threading, urllib.request

BASE = "http://127.0.0.1:8000"

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

run = post("/runs", {"question": "What blocks P-BEACON, and who owns the fix?"})

with urllib.request.urlopen(BASE + run["events"]) as stream:
    etype, data = None, []
    for raw in stream:
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"): etype = line[6:].strip()
        elif line.startswith("data:"): data.append(line[5:].strip())
        elif line == "" and etype:
            payload = json.loads("\n".join(data) or "{}")
            if etype == "ask_user":
                post(run["reply"], {"answer": input(payload["question"] + " ")})
            elif etype == "final":
                print("ANSWER:", payload["final"])
            elif etype == "end":
                break
            etype, data = None, []
```

## How it maps to the engine

- One shared `RLMEngine` serves all sessions — the engine holds **no per-run
  state**, so concurrent runs are safe. Each run gets its own `Channel`,
  `UsageMeter`, and (via `env_factory`) its own REPL.
- Each run executes in a **daemon thread**. `ask_user` blocks that thread on a
  reply queue; the `POST .../reply` fulfills it. SSE is one-directional, which is
  exactly why the reply comes back on its own request.
- The server is a thin `SSEChannel`: `emit(step)` → serialize → SSE frame;
  `ask_user(q)` → emit an `ask_user` event, then block for the reply.

## Going to production

This is a reference/dev server (stdlib only). For real deployments, keep the
`Channel`/threading shape and add:

- **A sandbox** — the blocker. Implement a sandboxed `ExecutionEnvironment`
  before any untrusted exposure.
- **Auth & limits** — the endpoints are open and unbounded.
- **Cancellation/timeouts** — a disconnected client leaves its run finishing in
  the background; `ask_user` blocks indefinitely until a reply.
- **A real ASGI framework** (FastAPI/Starlette) if you want WebSockets, backpressure,
  or many concurrent streams — the `SSEChannel` translates directly.
