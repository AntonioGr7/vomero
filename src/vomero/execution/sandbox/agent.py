"""In-sandbox agent — the process that actually runs the model's code.

This file runs INSIDE the sandbox (a gVisor container). It is deliberately
standalone — stdlib only, plus `protocol.py` sitting next to it and the
`context/` package dir (corpus.py + its siblings) bind-mounted in by path — so
it runs on a stock `python:3.11-slim` image with nothing installed.

It mirrors `InProcessEnvironment`: a persistent namespace the model's code runs
in across many `execute` calls, with stdout/stderr captured. The difference is
the host-stateful helpers. `corpus` is a REAL local `Corpus` over the read-only
bind mount (so grep/read/peek stay fast, no per-call round trip), but `llm`,
`rlm`, `answer`, `ask_user`, `ask_parent`, the `todo` surface AND `corpus.search`
are RPC stubs: calling one sends a request back to the host over the control
socket and blocks for the reply. `corpus.search` is delegated because it needs
the host's loaded index and (for dense) network the sandbox doesn't have; the
host runs the real callable (closing over the engine, meter, channel, index, ...)
and sends the result back.

Invocation:  python agent.py <socket-path> <context-pkg-dir>
Everything else (corpus root, which helpers exist, scoping) arrives in the
`init` message over the socket.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import socket
import sys
import traceback
from typing import Any

import protocol  # bind-mounted next to this file; see SandboxEnvironment


def main() -> int:
    socket_path = sys.argv[1]
    corpus_src = sys.argv[2]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    init = protocol.recv_msg(sock)
    if init is None or init.get("op") != "init":
        return 1

    ns, injected = _build_namespace(sock, init, corpus_src)
    protocol.send_msg(sock, {"op": "ready"})

    # Serve requests until the host closes the socket (or asks us to stop).
    while True:
        msg = protocol.recv_msg(sock)
        if msg is None or msg.get("op") == "shutdown":
            return 0
        op = msg.get("op")
        if op == "execute":
            _execute(sock, ns, msg.get("code", ""))
        elif op == "describe_state":
            text = _describe_state(ns, injected, msg.get("max_vars", 40))
            protocol.send_msg(sock, {"op": "value", "text": text})
        # Unknown ops are ignored — forward-compatible.


# --- namespace construction -------------------------------------------------

def _build_namespace(
    sock: socket.socket, init: dict, corpus_src: str
) -> tuple[dict[str, Any], set[str]]:
    """Recreate the REPL surface the host described in the `init` message."""
    ns: dict[str, Any] = {"__name__": "__vomero_repl__"}
    injected: set[str] = set()

    corpus_spec = init.get("corpus")
    if corpus_spec:
        Corpus = _load_corpus(corpus_src)
        ns["corpus"] = Corpus(corpus_spec["root"], allow=corpus_spec.get("allow"))
        injected.add("corpus")

    # Bare helper functions (llm/rlm/answer/ask_user/ask_parent) -> RPC stubs.
    # A dotted key like "corpus.search" is bound as a METHOD on the real local
    # corpus: read/grep/peek stay fast and local, but search() is delegated to
    # the host (which holds the loaded index and the network for dense queries),
    # keeping the sandbox pure-compute and network-free. The host returns plain
    # dicts; we wrap them in a Hit-like object so the model sees readable hits.
    for name in init.get("functions", []):
        if "." in name:
            obj_name, method = name.split(".", 1)
            target = ns.get(obj_name)
            if target is not None:
                setattr(target, method, _make_hit_stub(sock, name))
            injected.add(name)
            continue
        ns[name] = _make_stub(sock, name)
        injected.add(name)

    # Objects with methods (e.g. `todo`) -> a proxy whose methods RPC by
    # "<object>.<method>" key.
    for name, methods in init.get("objects", {}).items():
        ns[name] = _make_proxy(sock, name, methods)
        injected.add(name)

    # Plain JSON-serializable data the host wanted available by name.
    for key, value in init.get("data", {}).items():
        ns[key] = value
        injected.add(key)

    return ns, injected


def _load_corpus(pkg_dir: str):
    """Load the `Corpus` class from the bind-mounted `context/` package dir.

    `corpus.py` imports its siblings relatively (`from .source import ...`), so
    it can't be exec'd as a lone file — it must load as a submodule of a package
    whose `__path__` points at the bind-mounted dir, so those relative imports
    resolve to the sibling files. We register a synthetic parent package, then
    load `corpus` under it. No `import vomero...`, nothing installed: every file
    in `context/` is stdlib-only, so this works on a stock python image."""
    import importlib.machinery
    import os

    pkg_name = "_vomero_ctx"
    pkg = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(pkg_name, None, is_package=True)
    )
    pkg.__path__ = [pkg_dir]
    sys.modules[pkg_name] = pkg
    spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.corpus", os.path.join(pkg_dir, "corpus.py")
    )
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which fails if the module isn't there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.Corpus


# --- RPC stubs ---------------------------------------------------------------

def _rpc(sock: socket.socket, key: str, args: tuple, kwargs: dict) -> Any:
    """Call a host-side helper and block for its result."""
    protocol.send_msg(sock, {"op": "rpc", "key": key,
                             "args": list(args), "kwargs": kwargs})
    reply = protocol.recv_msg(sock)
    if reply is None:
        raise RuntimeError(f"host closed the connection during {key}()")
    if reply.get("op") == "rpc_error":
        raise RuntimeError(reply.get("error", f"{key}() failed on the host"))
    return reply.get("value")


def _make_stub(sock: socket.socket, name: str):
    def stub(*args, **kwargs):
        return _rpc(sock, name, args, kwargs)

    stub.__name__ = name
    return stub


class _Hit:
    """Local stand-in for context.search.Hit — reconstructed from the dicts the
    host returns, so a delegated search() prints the same way it does in-process
    and exposes .doc/.score/.snippet/.span."""

    __slots__ = ("doc", "score", "snippet", "span")

    def __init__(self, doc, score, snippet, span=None):
        self.doc, self.score, self.snippet, self.span = doc, score, snippet, span

    def __repr__(self):
        where = f"{self.doc}" + (f"[{self.span[0]}:{self.span[1]}]" if self.span else "")
        return f"{where} (score {self.score:.3f}): {str(self.snippet).strip()[:200]}"


def _make_hit_stub(sock: socket.socket, name: str):
    """An RPC stub whose dict results are rewrapped as _Hit objects."""
    def stub(*args, **kwargs):
        rows = _rpc(sock, name, args, kwargs) or []
        return [_Hit(**r) if isinstance(r, dict) else r for r in rows]

    stub.__name__ = name
    return stub


class _Proxy:
    """A namespace object whose attributes are RPC method stubs."""


def _make_proxy(sock: socket.socket, name: str, methods: list[str]) -> _Proxy:
    proxy = _Proxy()
    for method in methods:
        setattr(proxy, method, _make_stub(sock, f"{name}.{method}"))
    return proxy


# --- execution ---------------------------------------------------------------

def _execute(sock: socket.socket, ns: dict, code: str) -> None:
    """Run one code block in the persistent namespace, capturing output.

    Mirrors InProcessEnvironment.execute(). The control socket is a separate
    fd, so redirecting stdout/stderr here never disturbs the RPC channel even if
    the model's code writes to fd 1/2 directly."""
    buf = io.StringIO()
    error: str | None = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(code, "<vomero>", "exec"), ns)
    except Exception:
        error = traceback.format_exc()
    protocol.send_msg(sock, {"op": "result", "stdout": buf.getvalue(), "error": error})


def _describe_state(ns: dict, injected: set[str], max_vars: int) -> str:
    """The model's own variables, one per line (for compaction). Mirrors
    InProcessEnvironment.describe_state()."""
    lines: list[str] = []
    for name, value in ns.items():
        if name.startswith("__") or name in injected:
            continue
        lines.append(f"  {name} : {_summarize_value(value)}")
        if len(lines) >= max_vars:
            lines.append("  … (more variables omitted)")
            break
    return "\n".join(lines)


def _summarize_value(value: Any) -> str:
    type_name = type(value).__name__
    if callable(value) and not isinstance(value, (str, bytes)):
        return f"{type_name} (callable)"
    try:
        return f"{type_name} (len={len(value)})"
    except TypeError:
        pass
    try:
        text = repr(value)
    except Exception:
        return type_name
    if len(text) > 60:
        text = text[:57] + "…"
    return f"{type_name} = {text}"


if __name__ == "__main__":
    raise SystemExit(main())
