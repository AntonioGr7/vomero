"""Minimal FastAPI service wrapping Vomero — the app the Kubernetes manifests run.

Same shape as the example in docs/deployment.md, with a /healthz probe and the
corpus path taken from VOMERO_CORPUS_PATH (so the container mount is configurable).
History and sessions are in-memory: fine for one replica; for several, add
sticky routing on session_id or an external store (see docs/deployment.md §8).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vomero.config import Settings
from vomero.context.corpus import Corpus
from vomero.engine import Compactor, RLMEngine
from vomero.execution import build_env_factory, build_session_pool
from vomero.llm import build_client
from vomero.llm.base import Message
from vomero.server import step_to_payload  # Step -> JSON-able dict
from vomero.usage import UsageMeter

settings = Settings.from_env()

# The corpus is chosen at RUNTIME: the user picks data in the UI, it's downloaded
# into a subfolder of DATA_ROOT, and each request names that folder via `dataset`.
# DATA_ROOT must be a WRITABLE volume (the download target) — see deployment.yaml.
DATA_ROOT = Path(os.getenv("VOMERO_DATA_ROOT", "/data")).resolve()
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def resolve_corpus(dataset: str) -> Corpus:
    """Map a client-supplied dataset id to its on-disk folder, safely.

    Sanitizes the id to a single path segment and confirms the resolved folder
    stays under DATA_ROOT, so a request can never reach arbitrary host paths.
    Raises FileNotFoundError if the data hasn't been downloaded yet."""
    name = _SAFE_NAME.sub("_", dataset).strip("._")
    if not name:
        raise ValueError("invalid dataset id")
    folder = (DATA_ROOT / name).resolve()
    if folder != DATA_ROOT and DATA_ROOT not in folder.parents:
        raise ValueError("dataset id escapes the data root")
    return Corpus(folder)  # Corpus raises FileNotFoundError if the folder is absent

engine = RLMEngine(
    build_client(settings),
    env_factory=build_env_factory(settings),
    model=settings.model,
    max_steps=settings.max_steps,
    max_depth=settings.max_depth,
    compactor=(
        Compactor(context_window=settings.context_window, ratio=settings.compact_ratio)
        if settings.compact_ratio > 0
        else None
    ),
    enable_interaction=True,
)
pool = build_session_pool(settings)
HISTORY: dict[tuple[str, str], list[Message]] = {}

app = FastAPI()
SESSIONS: dict[str, "SSEChannel"] = {}
_END = object()


class SSEChannel:
    """Streams events to one client; ask_user blocks on a reply queue."""

    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self.replies: queue.Queue = queue.Queue()

    def emit(self, step) -> None:
        self.events.put(step_to_payload(step))

    def ask_user(self, question: str) -> str:
        self.events.put({"type": "ask_user", "question": str(question)})
        return self.replies.get()


@app.post("/runs")
async def start_run(req: Request):
    body = await req.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "missing question"}, status_code=400)
    # Which data the user selected (downloaded into DATA_ROOT/<dataset>).
    dataset = body.get("dataset") or ""
    try:
        corpus = resolve_corpus(dataset)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except FileNotFoundError:
        return JSONResponse({"error": "dataset not found — download it first"},
                            status_code=404)
    user_id = body.get("user_id", "anonymous")
    session_id = body.get("session_id") or uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    ch = SSEChannel()
    SESSIONS[run_id] = ch

    def worker() -> None:
        # Key by dataset too: a warm sandbox pins the corpus it first mounted,
        # so a session that switches datasets must get a distinct env (and its
        # own history/workspace). A follow-up just re-sends the same dataset.
        key = (user_id, f"{session_id}:{dataset}")
        meter = UsageMeter()
        transcript: list[Message] = []
        try:
            with pool.session(key) as env:
                engine.run(
                    question, corpus, channel=ch, meter=meter,
                    history=HISTORY.get(key, []), transcript_sink=transcript, env=env,
                )
            HISTORY[key] = transcript
            ch.events.put({"type": "done", "session_id": session_id,
                           "usage": {"calls": meter.calls,
                                     "total_tokens": meter.total_tokens}})
        except Exception as exc:  # noqa: BLE001 - surface to the client
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
                yield f"event: {item.get('type', 'event')}\ndata: {json.dumps(item)}\n\n"
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
    return {"ok": True}


@app.on_event("shutdown")
def _cleanup():
    pool.stop_sweeper()
    pool.close_all()
