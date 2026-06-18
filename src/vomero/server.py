"""A reference HTTP/SSE server that exposes the RLM to a browser or any client.

It implements the `Channel` seam over the wire:

* server -> client: **Server-Sent Events** (SSE) stream every `Step` the engine
  emits (progress, code, output, usage, plan, compaction, the final answer) as
  JSON, plus a synthetic `ask_user` event when the agent needs the human.
* client -> server: a plain `POST .../reply` fulfills a pending `ask_user`
  (SSE is one-directional, so the answer comes back on its own request).

Endpoints (the server is bound to ONE corpus, chosen at startup):

  POST /runs                 {"question", "user_id"?, "session_id"?}
                                                  -> {"session_id", "events", "reply"}
  GET  /runs/<id>/events     text/event-stream    -> the live event stream
  POST /runs/<id>/reply      {"answer": "..."}    -> fulfills the current ask_user

To ask a follow-up that builds on a previous run, send the SAME {user_id,
session_id} again: the server replays that conversation's transcript as context.
Omit session_id to begin a new conversation (the id is returned in the response
and in the `done` event).

Concurrency: the engine holds no per-run state, so one engine instance serves
all sessions. Each run executes in its own daemon thread with its own Channel,
UsageMeter and REPL; `ask_user` blocks that thread until a reply POST arrives.

Stdlib only — this is a reference/dev server. For production put a real ASGI
framework in front (the Channel/threading shape is identical), add auth, request
limits, cancellation, and — critically — a sandboxed ExecutionEnvironment: this
runs model-authored code in-process with `exec`.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .channel import Channel
from .config import Settings
from .context.corpus import Corpus
from .engine import Compactor, RLMEngine
from .engine.rlm import Step
from .llm.base import Message
from .execution import SessionEnvPool, build_env_factory, build_session_pool
from .llm import build_client
from .usage import UsageMeter

# Sentinel pushed onto a session's event queue to close the SSE stream.
_END = object()


def _event_type(step: Step) -> str:
    """A discriminator the client can switch on (the SSE `event:` name)."""
    for name in ("compaction", "usage", "message", "code", "llm_call",
                 "output", "interaction", "todo", "note", "final"):
        if getattr(step, name) is not None:
            return name
    return "event"


def step_to_payload(step: Step) -> dict:
    """Serialize a Step to a JSON-able dict (drops unset fields, adds a type)."""
    payload = {k: v for k, v in asdict(step).items() if v is not None}
    payload["type"] = _event_type(step)
    return payload


class SSEChannel:
    """A Channel that streams events to one SSE client and blocks ask_user on a
    reply queue the client fills via POST."""

    def __init__(self, events: "queue.Queue", replies: "queue.Queue"):
        self._events = events
        self._replies = replies

    def emit(self, step: Step) -> None:
        self._events.put(step_to_payload(step))

    def ask_user(self, question: str) -> str:
        self._events.put({"type": "ask_user", "question": str(question)})
        return self._replies.get()  # blocks the run thread until a reply POST


class Session:
    def __init__(self, sid: str):
        self.id = sid
        self.events: queue.Queue = queue.Queue()
        self.replies: queue.Queue = queue.Queue()


class RunManager:
    """Owns the shared engine + corpus and tracks live sessions.

    Three stores, deliberately separate and keyed differently:
      * `_sessions` — live runs, keyed by run id, for SSE/reply routing. A
        session is dropped when its event stream ends.
      * `_history` — conversation transcripts, keyed by (user_id, session_id),
        so a follow-up replays the prior turns. OUTLIVES the live session.
      * `_pool` — persistent execution environments, keyed by (user_id,
        session_id), so a follow-up resumes the model's REPL variables and
        workspace files. Idle envs are reclaimed after the configured TTL.

    Together: `_history` restores the conversation, `_pool` restores the live
    state. Both are in-process here; back them with SQLite/Redis + a durable
    workspace volume for restart survival.

    Admission control: `max_concurrent_runs` caps in-flight runs per replica via
    a semaphore. Over the cap, `start()` returns None and the handler replies
    HTTP 429 — graceful backpressure instead of unbounded threads/containers and
    an eventual OOM. 0 = unlimited (the default).
    """

    def __init__(self, corpus: Corpus, engine: RLMEngine,
                 pool: SessionEnvPool | None = None,
                 *, max_concurrent_runs: int = 0,
                 max_total_tokens: int = 0, max_total_calls: int = 0):
        self.corpus = corpus
        self.engine = engine
        self._pool = pool
        # Per-run global budget (rides on each run's UsageMeter).
        self._max_total_tokens = max_total_tokens
        self._max_total_calls = max_total_calls
        self._sessions: dict[str, Session] = {}
        self._history: dict[tuple[str, str], list[Message]] = {}
        self._lock = threading.Lock()
        # Bounded so a balanced acquire/release can't over-release; None when
        # unlimited. A permit is held for a run's whole lifetime (incl. time
        # blocked in ask_user), since that's exactly when it holds a container.
        self._runs = (
            threading.BoundedSemaphore(max_concurrent_runs)
            if max_concurrent_runs > 0 else None
        )

    def start(self, question: str, *, user_id: str = "anonymous",
              session_id: str | None = None,
              enable_planning: bool | None = None,
              planning_root_only: bool | None = None) -> Session | None:
        # Admission control: take a permit up front (non-blocking). None means
        # we're at capacity — the handler turns that into HTTP 429. The permit
        # is released in `_run`'s finally (or here if the thread never launches).
        if self._runs is not None and not self._runs.acquire(blocking=False):
            return None
        try:
            # A new conversation gets a fresh session_id; passing back a known
            # (user_id, session_id) continues that conversation.
            session_id = session_id or uuid.uuid4().hex
            key = (user_id, session_id)
            with self._lock:
                # Seed with a copy of the stored transcript (None on first turn).
                history = list(self._history.get(key, []))
                session = Session(session_id)
                self._sessions[session_id] = session
            threading.Thread(
                target=self._run, args=(session, question, key, history),
                kwargs={"enable_planning": enable_planning,
                        "planning_root_only": planning_root_only},
                daemon=True,
            ).start()
            return session
        except BaseException:
            # The run thread never took ownership of the permit; release it so a
            # failed launch doesn't leak capacity.
            if self._runs is not None:
                self._runs.release()
            raise

    def get(self, sid: str) -> Session | None:
        with self._lock:
            return self._sessions.get(sid)

    def _drop(self, sid: str) -> None:
        # Only the live-run handle is dropped; the conversation transcript in
        # `_history` is kept so the next turn can resume it.
        with self._lock:
            self._sessions.pop(sid, None)

    def _run(self, session: Session, question: str,
             key: tuple[str, str], history: list[Message], *,
             enable_planning: bool | None = None,
             planning_root_only: bool | None = None) -> None:
        channel = SSEChannel(session.events, session.replies)
        meter = UsageMeter(
            max_total_tokens=self._max_total_tokens,
            max_total_calls=self._max_total_calls,
        )
        transcript: list[Message] = []
        try:
            # A pooled, persistent env (when configured) keeps this session's
            # variables/workspace across turns; its lock serializes same-session
            # runs. Without a pool, the engine builds a fresh env per run.
            if self._pool is not None:
                with self._pool.session(key) as env:
                    self.engine.run(
                        question, self.corpus, channel=channel, meter=meter,
                        enable_planning=enable_planning,
                        planning_root_only=planning_root_only,
                        history=history, transcript_sink=transcript, env=env,
                    )
            else:
                self.engine.run(
                    question, self.corpus, channel=channel, meter=meter,
                    enable_planning=enable_planning,
                    planning_root_only=planning_root_only,
                    history=history, transcript_sink=transcript,
                )
            # Persist the resumable transcript only on success (the sink is
            # filled at the engine's normal exit points, empty on error, so a
            # failed run leaves the prior history intact).
            with self._lock:
                self._history[key] = transcript
            session.events.put({
                "type": "done",
                "session_id": key[1],
                "usage": {
                    "calls": meter.calls,
                    "total_tokens": meter.total_tokens,
                    "prompt_tokens": meter.prompt_tokens,
                    "completion_tokens": meter.completion_tokens,
                    "estimated": meter.estimated,
                },
            })
        except Exception as exc:  # surface to the client instead of dying silently
            session.events.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            # Free the admission permit FIRST so capacity recovers immediately,
            # then close the event stream.
            if self._runs is not None:
                self._runs.release()
            session.events.put(_END)


class _Handler(BaseHTTPRequestHandler):
    server_version = "vomero-rlm/0.1"

    # -- helpers --------------------------------------------------------
    @property
    def manager(self) -> RunManager:
        return self.server.manager  # type: ignore[attr-defined]

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, body: dict,
              *, extra_headers: dict[str, str] | None = None) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- routes ---------------------------------------------------------
    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        parts = self.path.strip("/").split("/")
        if parts == ["runs"]:
            body = self._read_json()
            question = (body.get("question") or "").strip()
            if not question:
                return self._json(400, {"error": "missing 'question'"})
            # Optional per-request planning (else the server/engine default).
            plan = bool(body["plan"]) if "plan" in body else None
            plan_root = bool(body["plan_root_only"]) if "plan_root_only" in body else None
            # Conversation continuity: pass back a prior {user_id, session_id}
            # to ask a follow-up that builds on that conversation. Omit
            # session_id to start a fresh one (returned below).
            user_id = (body.get("user_id") or "anonymous").strip() or "anonymous"
            session_id = (body.get("session_id") or "").strip() or None
            session = self.manager.start(
                question, user_id=user_id, session_id=session_id,
                enable_planning=plan, planning_root_only=plan_root)
            if session is None:  # at capacity — backpressure, not a crash
                return self._json(
                    429, {"error": "server at capacity; retry shortly"},
                    extra_headers={"Retry-After": "5"})
            return self._json(200, {
                "session_id": session.id,
                "events": f"/runs/{session.id}/events",
                "reply": f"/runs/{session.id}/reply",
            })
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "reply":
            session = self.manager.get(parts[1])
            if session is None:
                return self._json(404, {"error": "unknown session"})
            session.replies.put(str(self._read_json().get("answer", "")))
            return self._json(200, {"ok": True})
        self._json(404, {"error": "not found"})

    def do_GET(self) -> None:
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "runs" and parts[2] == "events":
            session = self.manager.get(parts[1])
            if session is None:
                return self._json(404, {"error": "unknown session"})
            return self._stream(session)
        self._json(404, {"error": "not found"})

    def _stream(self, session: Session) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()
        self.wfile.write(b": connected\n\n")
        self.wfile.flush()
        try:
            while True:
                item = session.events.get()
                if item is _END:
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                    break
                frame = f"event: {item.get('type', 'event')}\ndata: {json.dumps(item)}\n\n"
                self.wfile.write(frame.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected; the run keeps going in the background.
            pass
        finally:
            self.manager._drop(session.id)

    def log_message(self, fmt: str, *args) -> None:  # concise, to stderr
        import sys
        print("[server] " + (fmt % args), file=sys.stderr)


def build_manager(settings: Settings, data: str) -> RunManager:
    """Wire a RunManager from settings + a corpus path (interaction forced on).

    The corpus's search() uses, in precedence: an external retrieval service
    (VOMERO_RETRIEVAL_URL → RemoteBackend, so the pod holds no vectors — the
    multi-tenant path); else a prebuilt persistent index (VOMERO_INDEX_DIR,
    opened read-only); else a lazy in-memory index. Embedder is for the local
    dense paths only."""
    from .context.retrieval import build_retrieval_backend
    from .llm import build_embedder

    corpus = Corpus(
        data,
        embedder=build_embedder(settings),
        index_dir=settings.index_dir or None,
        backend=build_retrieval_backend(settings),
    )
    compactor = None
    if settings.compact_ratio > 0:
        compactor = Compactor(
            context_window=settings.context_window,
            ratio=settings.compact_ratio,
            keep_recent_messages=settings.compact_keep_recent,
            min_reclaim_tokens=settings.compact_min_reclaim,
        )
    engine = RLMEngine(
        build_client(settings),
        env_factory=build_env_factory(settings),
        model=settings.model,
        max_steps=settings.max_steps,
        max_depth=settings.max_depth,
        max_output_chars=settings.max_output_chars,
        max_parallel_calls=settings.max_parallel_calls,
        compactor=compactor,
        enable_planning=settings.enable_planning,
        planning_root_only=settings.planning_root_only,
        enable_interaction=True, 
        interaction_root_only=settings.interaction_root_only,
    )
    # Persistent per-session envs so follow-ups resume variables + workspace.
    pool = build_session_pool(settings)
    return RunManager(corpus, engine, pool,
                      max_concurrent_runs=settings.max_concurrent_runs,
                      max_total_tokens=settings.max_total_tokens,
                      max_total_calls=settings.max_total_calls)


def serve(data: str, host: str = "127.0.0.1", port: int = 8000,
          settings: Settings | None = None) -> None:
    import sys

    settings = settings or Settings.from_env()
    manager = build_manager(settings, data)

    # Build/open the search index at STARTUP (not lazily on the first question),
    # with visible progress — so the first user isn't slow and the operator can
    # see the corpus being prepared. Skipped with VOMERO_WARMUP=0 (grep-only).
    if settings.warmup_search:
        import time
        print(f"preparing search index for {manager.corpus.root} … "
              "(this can take a while on a large corpus)", file=sys.stderr, flush=True)
        t0 = time.monotonic()
        status = manager.corpus.warmup()
        print(f"  ✓ search ready: {status}  ({time.monotonic() - t0:.1f}s)",
              file=sys.stderr, flush=True)

    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.manager = manager  # type: ignore[attr-defined]

    # Reclaim idle session envs on time even without new traffic.
    if manager._pool is not None:
        manager._pool.start_sweeper()

    print(f"vomero serving corpus {manager.corpus.root} on http://{host}:{port}",
          file=sys.stderr)
    print("  POST /runs  ·  GET /runs/<id>/events (SSE)  ·  POST /runs/<id>/reply",
          file=sys.stderr)
    if settings.workspace_root:
        print(f"  ↻ sessions persist: variables for {settings.session_ttl:.0f}s idle, "
              f"workspace under {settings.workspace_root}", file=sys.stderr)
    caps = []
    if settings.max_concurrent_runs > 0:
        caps.append(f"max {settings.max_concurrent_runs} concurrent runs (429 over)")
    if settings.max_sessions > 0:
        caps.append(f"max {settings.max_sessions} warm sessions (LRU)")
    if caps:
        print("  🚦 limits: " + " · ".join(caps), file=sys.stderr)
    elif settings.exec_backend == "sandbox":
        print("  ⚠  no admission limits set (VOMERO_MAX_CONCURRENT_RUNS / "
              "VOMERO_MAX_SESSIONS) — a load spike can exhaust the node.",
              file=sys.stderr)
    if settings.exec_backend == "sandbox":
        print(f"\n  🔒 sandbox: gVisor ({settings.sandbox_runtime}), "
              f"mem={settings.sandbox_memory} cpus={settings.sandbox_cpus} "
              f"net={settings.sandbox_network} — model code is isolated.\n",
              file=sys.stderr)
    else:
        print("\n  ⚠  NOT SANDBOXED: model-authored code runs in-process with exec. "
              "Serve only trusted corpora/users, or set VOMERO_SANDBOX=1 for the "
              "gVisor backend.\n",
              file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        httpd.shutdown()
    finally:
        if manager._pool is not None:
            manager._pool.stop_sweeper()
            manager._pool.close_all()  # tear down any warm containers
