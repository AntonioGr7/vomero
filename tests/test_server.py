"""End-to-end test of the HTTP/SSE server against a scripted fake client.

Spins up the real ThreadingHTTPServer on an ephemeral port, drives it over
localhost, and checks that: events stream as SSE, an `ask_user` round-trips via
the reply endpoint, and the final answer + usage arrive.
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from vomero.context.corpus import Corpus
from vomero.engine import RLMEngine
from vomero.llm.base import LLMResponse, ToolCall, Usage
from vomero.server import RunManager, _Handler

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def _py(call_id, code):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name="python", arguments={"code": code})],
        usage=Usage(prompt_tokens=100, completion_tokens=8),
    )


def _post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _read_sse(url, on_event, stop_type="end"):
    """Read an SSE stream, calling on_event(type, data) per frame until stop_type."""
    with urllib.request.urlopen(url) as r:
        etype, lines = None, []
        for raw in r:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                etype = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].strip())
            elif line == "":  # frame boundary
                if etype:
                    on_event(etype, json.loads("\n".join(lines) or "{}"))
                    if etype == stop_type:
                        return
                etype, lines = None, []


def _run_server(engine):
    """Start the server on an ephemeral port; return (base_url, shutdown)."""
    manager = RunManager(Corpus(CORPUS), engine)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.manager = manager
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd.shutdown


def test_server_streams_todo_events_only_when_plan_requested():
    def make_engine():
        # Engine default planning OFF (as the server is by default).
        return RLMEngine(FakeClient([
            _py("1", "todo.plan(['look', 'answer']); todo.start(1)"),
            _py("2", "answer('done')"),
        ]))

    # plan: true -> todo events stream to the client.
    base, shutdown = _run_server(make_engine())
    try:
        started = _post(f"{base}/runs", {"question": "go", "plan": True})
        types = []
        _read_sse(f"{base}{started['events']}", lambda t, d: types.append(t))
        assert "todo" in types
    finally:
        shutdown()

    # No plan -> referencing `todo` errors; no todo events.
    base, shutdown = _run_server(make_engine())
    try:
        started = _post(f"{base}/runs", {"question": "go"})
        types = []
        _read_sse(f"{base}{started['events']}", lambda t, d: types.append(t))
        assert "todo" not in types
    finally:
        shutdown()


def test_server_resumes_conversation_for_follow_up():
    """Reusing {user_id, session_id} on a second POST /runs replays the first
    run's transcript into the engine, so the follow-up has prior context."""
    captured: dict = {}

    class RecordingClient(FakeClient):
        def complete(self, messages, *, tools=None, model=None, temperature=None):
            # Record the context handed to the FIRST call of the second run.
            if self.calls == 1:
                captured["context"] = list(messages)
            return super().complete(messages, tools=tools, model=model)

    # Turn 1 answers; turn 2 (same client/engine, since calls continue) answers
    # the follow-up. Four scripted responses span both runs.
    engine = RLMEngine(RecordingClient([
        _py("1", "answer('P-BEACON is blocked by P-ATLAS')"),  # turn 1
        LLMResponse(content="P-ATLAS owns it.", tool_calls=[]),  # turn 2, 1st call
    ]))
    base, shutdown = _run_server(engine)
    try:
        # Turn 1: new conversation (no session_id supplied).
        started = _post(f"{base}/runs", {"question": "What blocks P-BEACON?",
                                         "user_id": "u1"})
        sid = started["session_id"]
        finals = []
        _read_sse(f"{base}{started['events']}",
                  lambda t, d: finals.append((t, d)))

        # Turn 2: SAME user_id + session_id => follow-up with history.
        started2 = _post(f"{base}/runs", {"question": "Who owns it?",
                                          "user_id": "u1", "session_id": sid})
        assert started2["session_id"] == sid          # same conversation
        types2 = []
        _read_sse(f"{base}{started2['events']}",
                  lambda t, d: types2.append((t, d)))

        ctx = captured["context"]
        contents = [m.content for m in ctx]
        blob = "\n".join(
            (m.content or "") + "".join(tc.arguments.get("code", "") for tc in m.tool_calls)
            for m in ctx
        )
        assert ctx[0].role == "system"
        assert "What blocks P-BEACON?" in contents    # prior question replayed
        assert "P-ATLAS" in blob                      # prior answer replayed
        assert contents[-1] == "Who owns it?"         # new question last
        final2 = next(d["final"] for t, d in types2 if t == "final")
        assert final2 == "P-ATLAS owns it."
    finally:
        shutdown()


def test_server_streams_events_and_round_trips_ask_user():
    # Step 2 asks the user; step 3 uses the reply in the final answer.
    script = [
        _py("1", "print(corpus.files())"),
        _py("2", "pref = ask_user('which project?'); print(pref)"),
        _py("3", "answer('focusing on ' + pref)"),
    ]
    engine = RLMEngine(FakeClient(script), enable_interaction=True)
    manager = RunManager(Corpus(CORPUS), engine)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.manager = manager
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    try:
        started = _post(f"{base}/runs", {"question": "what should I focus on?"})
        sid = started["session_id"]

        events = []
        replied = threading.Event()

        def on_event(etype, data):
            events.append((etype, data))
            if etype == "ask_user" and not replied.is_set():
                replied.set()
                _post(f"{base}{started['reply']}", {"answer": "P-BEACON"})

        _read_sse(f"{base}{started['events']}", on_event)

        types = [t for t, _ in events]
        assert "ask_user" in types          # the agent asked
        assert "final" in types             # and produced a final answer
        assert "done" in types              # usage summary at the end

        ask = next(d for t, d in events if t == "ask_user")
        assert ask["question"] == "which project?"

        final = next(d for t, d in events if t == "final")
        assert final["final"] == "focusing on P-BEACON"  # reply incorporated

        done = next(d for t, d in events if t == "done")
        assert done["usage"]["calls"] >= 3
    finally:
        httpd.shutdown()
