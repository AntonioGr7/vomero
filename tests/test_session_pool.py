"""Per-session persistence: reused envs keep variables, the pool evicts on TTL.

The sandbox bits use the "local" runner (agent.py as a plain subprocess, no
Docker), same as test_sandbox.py — enough to exercise warm-reuse / rebind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vomero.context.corpus import Corpus
from vomero.engine.rlm import RLMEngine
from vomero.execution import InProcessEnvironment, SessionEnvPool
from vomero.execution.base import ExecResult, ExecutionEnvironment
from vomero.execution.sandbox import SandboxConfig, SandboxEnvironment
from vomero.llm.base import LLMResponse, ToolCall, Usage

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"


# --- scripted client (mirrors the other engine tests) ----------------------

class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


_ID = [0]


def py(code):
    _ID[0] += 1
    return LLMResponse(content=None, usage=Usage(prompt_tokens=10, completion_tokens=2),
                       tool_calls=[ToolCall(id=f"c{_ID[0]}", name="python",
                                            arguments={"code": code})])


def say(text):
    return LLMResponse(content=text, tool_calls=[],
                       usage=Usage(prompt_tokens=10, completion_tokens=2))


# --- engine: a reused env keeps the model's variables across runs -----------

def test_reused_env_persists_variables_across_runs():
    env = InProcessEnvironment()
    # turn 1 defines `kept`; turn 2 (same env) answers with it.
    client = FakeClient([py("kept = 7 * 6"), say("turn1 done"), py("answer(str(kept))")])
    engine = RLMEngine(client)

    out1 = engine.run("set it", Corpus(CORPUS), env=env)
    assert out1 == "turn1 done"
    out2 = engine.run("use it", Corpus(CORPUS), env=env)
    assert out2 == "42"  # `kept` survived because the SAME env was reused


def test_fresh_env_loses_variables_between_runs():
    # No env= => each run builds its own env; the variable does NOT carry over.
    client = FakeClient([py("kept = 42"), say("done"),
                         py("answer(str(kept))"), say("recovered")])
    engine = RLMEngine(client)

    engine.run("set it", Corpus(CORPUS))
    out = engine.run("use it", Corpus(CORPUS))
    # `kept` is undefined in the fresh env, so the answer code raises and the
    # model falls through to a plain-text reply — proving no persistence.
    assert out == "recovered"


# --- sandbox: re-inject on a warm container rebinds without losing state ----

@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text("hello\n", encoding="utf-8")
    return Corpus(tmp_path)


def test_sandbox_reinject_rebinds_callables_and_keeps_variables(corpus):
    calls = []
    with SandboxEnvironment(SandboxConfig(runner="local")) as env:
        env.inject(corpus=corpus, tag=lambda: (calls.append("A"), "A")[1])
        env.execute("x = 99")          # variable lives in the warm namespace
        env.execute("tag()")           # resolves to the first closure -> "A"

        # Re-inject AFTER start: same name, new closure. Must not restart the
        # container (so `x` survives) but must rebind `tag` to the new function.
        env.inject(corpus=corpus, tag=lambda: (calls.append("B"), "B")[1])
        r = env.execute("print(x); tag()")

    assert r.ok
    assert r.stdout.strip() == "99"    # the variable survived the re-inject
    assert calls == ["A", "B"]         # the second run used the rebound closure


# --- pool: reuse within TTL, evict (and close) after -----------------------

class _FakeEnv(ExecutionEnvironment):
    def __init__(self, workspace):
        self.workspace = workspace
        self.closed = False

    def inject(self, **names):  # noqa: D401 - test stub
        pass

    def execute(self, code):
        return ExecResult(stdout="")

    def close(self):
        self.closed = True


def test_pool_reuses_within_ttl_then_evicts_after():
    clock = {"t": 0.0}
    created: list[_FakeEnv] = []

    def make_env(key, workspace):
        env = _FakeEnv(workspace)
        created.append(env)
        return env

    pool = SessionEnvPool(make_env, ttl_seconds=100.0, clock=lambda: clock["t"])
    key = ("u1", "s1")

    with pool.session(key) as e1:
        pass
    with pool.session(key) as e2:
        pass
    assert e1 is e2 and len(created) == 1       # reused while warm

    clock["t"] = 250.0                          # idle past the TTL
    with pool.session(key) as e3:
        pass
    assert e3 is not e1 and len(created) == 2   # old one evicted, new one built
    assert created[0].closed is True            # the evicted env was closed


def test_pool_workspace_dir_persists_until_discarded(tmp_path):
    pool = SessionEnvPool(lambda key, ws: _FakeEnv(ws), workspace_root=str(tmp_path))
    key = ("u1", "s1")

    with pool.session(key) as env:
        ws = env.workspace
    assert ws is not None and Path(ws).is_dir()  # a durable dir was created

    (Path(ws) / "report.md").write_text("kept", encoding="utf-8")
    pool.discard(key)                            # drop variables, keep files
    assert (Path(ws) / "report.md").exists()

    pool.discard(key, remove_workspace=True)     # now drop the files too
    assert not Path(ws).exists()
