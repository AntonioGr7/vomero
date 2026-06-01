"""Sandbox backend tests.

The expensive part — gVisor isolation itself — needs Docker + the `runsc`
runtime, so it's opt-in (set VOMERO_TEST_SANDBOX_DOCKER=1). Everything else (the
host<->agent wire protocol, RPC dispatch, namespace persistence, corpus access,
error capture, describe_state, the config factory) is exercised with the "local"
runner, which runs the same `agent.py` as a plain host subprocess — no Docker
required. That covers all the host/agent logic; only the container wrapper is
left for the gated test.
"""

from __future__ import annotations

import os
import shutil
import socket

import pytest

from vomero.context.corpus import Corpus
from vomero.execution import InProcessEnvironment, build_env_factory
from vomero.execution.sandbox import SandboxConfig, SandboxEnvironment
from vomero.execution.sandbox import protocol


# --- wire protocol ----------------------------------------------------------

def test_protocol_roundtrip():
    a, b = socket.socketpair()
    try:
        protocol.send_msg(a, {"op": "hello", "n": 3, "xs": [1, 2, 3]})
        assert protocol.recv_msg(b) == {"op": "hello", "n": 3, "xs": [1, 2, 3]}
    finally:
        a.close()
        b.close()


def test_protocol_eof_returns_none():
    a, b = socket.socketpair()
    a.close()
    assert protocol.recv_msg(b) is None
    b.close()


# --- factory ----------------------------------------------------------------

class _Settings:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_factory_defaults_to_inprocess():
    assert build_env_factory(_Settings()) is InProcessEnvironment
    assert build_env_factory(_Settings(exec_backend="inprocess")) is InProcessEnvironment


def test_factory_builds_sandbox_with_limits():
    factory = build_env_factory(_Settings(
        exec_backend="sandbox", sandbox_memory="2g", sandbox_cpus=1.5,
    ))
    env = factory()
    assert isinstance(env, SandboxEnvironment)
    assert env.config.memory == "2g"
    assert env.config.cpus == 1.5


# --- end-to-end via the local (non-Docker) runner ---------------------------

@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.md").write_text("hello world\nNEEDLE here\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("nothing to see\n", encoding="utf-8")
    return Corpus(tmp_path)


def _local_env() -> SandboxEnvironment:
    return SandboxEnvironment(SandboxConfig(runner="local"))


def test_persistent_namespace_and_stdout(corpus):
    with _local_env() as env:
        env.inject(corpus=corpus)
        r1 = env.execute("x = 41")
        assert r1.ok and r1.stdout == ""
        r2 = env.execute("print(x + 1)")
        assert r2.ok
        assert r2.stdout.strip() == "42"  # state persists across execute() calls


def test_corpus_is_real_and_local(corpus):
    with _local_env() as env:
        env.inject(corpus=corpus)
        r = env.execute("print([m.path for m in corpus.grep('NEEDLE')])")
        assert r.ok
        assert "a.md" in r.stdout


def test_error_is_captured_as_traceback(corpus):
    with _local_env() as env:
        env.inject(corpus=corpus)
        r = env.execute("1 / 0")
        assert not r.ok
        assert "ZeroDivisionError" in (r.error or "")


def test_rpc_calls_host_helpers(corpus):
    seen = {}

    def llm(text, system=None):
        seen["llm"] = text
        return f"distilled:{text}"

    def answer(text):
        seen["answer"] = text

    with _local_env() as env:
        env.inject(corpus=corpus, llm=llm, answer=answer)
        r = env.execute(
            "out = llm('summarize this')\n"
            "print(out)\n"
            "answer('final')\n"
        )
    assert r.ok
    assert r.stdout.strip() == "distilled:summarize this"
    assert seen["llm"] == "summarize this"
    assert seen["answer"] == "final"  # the host callable ran, in-process state mutated


def test_rpc_error_surfaces_inside_sandbox(corpus):
    def llm(text, system=None):
        raise ValueError("boom")

    with _local_env() as env:
        env.inject(corpus=corpus, llm=llm)
        r = env.execute("llm('x')")
    assert not r.ok
    assert "boom" in (r.error or "")


def test_object_proxy_methods(corpus):
    class FakeTodo:
        def __init__(self):
            self.calls = []

        def plan(self, items):
            self.calls.append(("plan", items))

        def start(self, n):
            self.calls.append(("start", n))

    todo = FakeTodo()
    with _local_env() as env:
        env.inject(corpus=corpus, todo=todo)
        r = env.execute("todo.plan(['a', 'b'])\ntodo.start(1)")
    assert r.ok
    assert todo.calls == [("plan", ["a", "b"]), ("start", 1)]


def test_describe_state_lists_user_variables(corpus):
    with _local_env() as env:
        env.inject(corpus=corpus, llm=lambda t, system=None: "x")
        env.execute("data = [1, 2, 3]\nname = 'vomero'")
        desc = env.describe_state()
    assert "data" in desc and "name" in desc
    assert "corpus" not in desc  # injected names are not the model's own vars
    assert "llm" not in desc


def test_execute_before_corpus_inject_errors():
    with _local_env() as env:
        with pytest.raises(RuntimeError):
            env.execute("x = 1")


# --- gated full gVisor integration ------------------------------------------

@pytest.mark.skipif(
    os.getenv("VOMERO_TEST_SANDBOX_DOCKER") != "1" or shutil.which("docker") is None,
    reason="set VOMERO_TEST_SANDBOX_DOCKER=1 with Docker + runsc available",
)
def test_docker_gvisor_end_to_end(corpus):
    env = SandboxEnvironment(SandboxConfig())  # real docker + runsc
    try:
        env.inject(corpus=corpus, answer=lambda t: None)
        r = env.execute("print([m.path for m in corpus.grep('NEEDLE')])")
        assert r.ok and "a.md" in r.stdout
        # No network: a socket connect should fail inside the sandbox.
        r2 = env.execute(
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
            "    print('REACHED')\n"
            "except Exception as e:\n"
            "    print('BLOCKED')\n"
        )
        assert "BLOCKED" in r2.stdout
    finally:
        env.close()
