"""A reference HTTP/SSE server that exposes the RLM to a browser or any client.

It implements the `Channel` seam over the wire:

* server -> client: **Server-Sent Events** (SSE) stream every `Step` the engine
  emits (progress, code, output, usage, plan, compaction, the final answer) as
  JSON, plus a synthetic `ask_user` event when the agent needs the human.
* client -> server: a plain `POST .../reply` fulfills a pending `ask_user`
  (SSE is one-directional, so the answer comes back on its own request).

Endpoints (the server is bound to ONE corpus, chosen at startup):

  POST /runs                 {"question": "..."}  -> {"session_id", "events", "reply"}
  GET  /runs/<id>/events     text/event-stream    -> the live event stream
  POST /runs/<id>/reply      {"answer": "..."}    -> fulfills the current ask_user

Concurrency: the engine holds no per-run state, so one engine instance serves
all sessions. Each run executes in its own daemon thread with its own Channel,
UsageMeter and REPL; `ask_user` blocks that thread until a reply POST arrives.

Stdlib only — this is a reference/dev server. For production put a real ASGI
framework in front (the Channel/threading shape is identical), add auth, request
limits, cancellation, and — critically — a sandboxed ExecutionEnvironment: this
runs model-authored code in-process with `exec` (see ADR 0001).
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
from .execution import build_env_factory
from .llm import build_client
from .usage import UsageMeter

# Sentinel pushed onto a session's event queue to close the SSE stream.
_END = object()


def _event_type(step: Step) -> str:
    """A discriminator the client can switch on (the SSE `event:` name)."""
    for name in ("compaction", "usage", "message", "code", "llm_call",
                 "output", "interaction", "todo", "final"):
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
    """Owns the shared engine + corpus and tracks live sessions."""

    def __init__(self, corpus: Corpus, engine: RLMEngine):
        self.corpus = corpus
        self.engine = engine
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def start(self, question: str, *, enable_planning: bool | None = None,
              planning_root_only: bool | None = None) -> Session:
        session = Session(uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        threading.Thread(
            target=self._run, args=(session, question),
            kwargs={"enable_planning": enable_planning,
                    "planning_root_only": planning_root_only},
            daemon=True,
        ).start()
        return session

    def get(self, sid: str) -> Session | None:
        with self._lock:
            return self._sessions.get(sid)

    def _drop(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def _run(self, session: Session, question: str, *,
             enable_planning: bool | None = None,
             planning_root_only: bool | None = None) -> None:
        channel = SSEChannel(session.events, session.replies)
        meter = UsageMeter()
        try:
            # The engine emits the final answer as a `final` Step via the
            # channel; we add a `done` event carrying the usage summary.
            self.engine.run(
                question, self.corpus, channel=channel, meter=meter,
                enable_planning=enable_planning,
                planning_root_only=planning_root_only,
            )
            session.events.put({
                "type": "done",
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

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
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
            session = self.manager.start(
                question, enable_planning=plan, planning_root_only=plan_root)
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
    """Wire a RunManager from settings + a corpus path (interaction forced on)."""
    corpus = Corpus(data)
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
        compactor=compactor,
        enable_planning=settings.enable_planning,
        planning_root_only=settings.planning_root_only,
        enable_interaction=True,  # the whole point of a remote client
        interaction_root_only=settings.interaction_root_only,
    )
    return RunManager(corpus, engine)


def serve(data: str, host: str = "127.0.0.1", port: int = 8000,
          settings: Settings | None = None) -> None:
    import sys

    settings = settings or Settings.from_env()
    manager = build_manager(settings, data)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.manager = manager  # type: ignore[attr-defined]

    print(f"vomero serving corpus {manager.corpus.root} on http://{host}:{port}",
          file=sys.stderr)
    print("  POST /runs  ·  GET /runs/<id>/events (SSE)  ·  POST /runs/<id>/reply",
          file=sys.stderr)
    if settings.exec_backend == "sandbox":
        print(f"\n  🔒 sandbox: gVisor ({settings.sandbox_runtime}), "
              f"mem={settings.sandbox_memory} cpus={settings.sandbox_cpus} "
              f"net={settings.sandbox_network} — model code is isolated.\n",
              file=sys.stderr)
    else:
        print("\n  ⚠  NOT SANDBOXED: model-authored code runs in-process with exec. "
              "Serve only trusted corpora/users, or set VOMERO_SANDBOX=1 for the "
              "gVisor backend (ADR 0001/0004).\n",
              file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
        httpd.shutdown()
