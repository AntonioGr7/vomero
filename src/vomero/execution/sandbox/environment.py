"""Host side of the gVisor sandbox: `SandboxEnvironment`.

Implements the same `inject(**names)` + `execute(code) -> ExecResult` contract
as `InProcessEnvironment`, but runs the model's code inside a gVisor container
instead of this process.

Shape of one run:

  1. The engine calls `inject(corpus=..., llm=..., rlm=..., answer=..., ...)`.
     We sort those into: the corpus (bind-mounted read-only), bare helper
     functions, and method-bearing objects (`todo`). Nothing starts yet.
  2. On the first `execute()`, we open a Unix socket on a host temp dir, launch
     `docker run --runtime=runsc ...` with that dir, the corpus, and Vomero's
     `agent.py` + the `context/` package dir bind-mounted in, and hand off
     control to the agent.
  3. Each `execute()` ships the code over the socket. While the agent runs it,
     any `llm()/rlm()/answer()/...` call comes back as an `rpc` message; we
     invoke the real host callable and return its result. When the agent sends
     `result`, the step is done.
  4. The container is reused for every step of the run (gVisor's startup cost is
     paid once). It is torn down when the environment is garbage-collected:
     closing the socket is an EOF the agent exits on, and `docker run --rm`
     removes the container.

The corpus is mounted read-only rather than proxied because it is read-only data
the model is allowed to read; the sandbox exists to protect the *host* from the
model's code (writes, network, fork bombs), not to gate corpus reads — and
grepping a folder over RPC would be painfully slow. The one exception is
`corpus.search`: it IS delegated to the host (registered as a host function),
because ranked search needs the host's loaded/persistent index and, for dense
retrieval, the network the sandbox is denied.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import weakref
from pathlib import Path
from typing import Any, Callable

from ...context import corpus as _corpus_module
from ..base import ExecResult, ExecutionEnvironment
from . import protocol
from .config import SandboxConfig

# Where things live inside the container (fixed; the host maps onto these).
_CONTAINER_CORPUS = "/corpus"
_CONTAINER_AGENT_DIR = "/opt/vomero-agent"
# Kept OUTSIDE the agent-dir mount: a file mountpoint can't be created inside
# another (read-only) bind mount.
_CONTAINER_CORPUS_SRC = "/opt/vomero_context"  # the context/ package dir, mounted ro
_CONTAINER_SOCK_DIR = "/sock"
_SOCK_NAME = "vomero.sock"
_CONTAINER_WORKSPACE = "/workspace"


class SandboxEnvironment(ExecutionEnvironment):
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

        # Filled by inject(), consumed at startup.
        self._corpus: Any | None = None
        self._functions: dict[str, Callable[..., Any]] = {}
        self._objects: dict[str, dict[str, Callable[..., Any]]] = {}
        self._data: dict[str, Any] = {}

        # Live connection state.
        self._proc: subprocess.Popen | None = None
        self._conn: socket.socket | None = None
        self._listener: socket.socket | None = None
        self._tmpdir: str | None = None
        self._started = False
        self._finalizer: weakref.finalize | None = None

    # -- ExecutionEnvironment contract ----------------------------------

    def inject(self, **names: Any) -> None:
        """Register the host surface. Callables become RPC stubs in the sandbox;
        the corpus is bind-mounted; objects with methods (e.g. `todo`) are
        method-proxied; anything JSON-serializable is sent as plain data.

        Reuse-safe: calling inject() again on an already-started container (the
        warm-session path) does NOT restart it. The corpus mount, any plain data
        and the model's REPL variables were fixed when it started and stay put;
        only the host-side callables are rebound. That works because the agent's
        RPC stubs resolve by NAME at call time, so the warm container transparently
        picks up this run's fresh closures (new channel/meter/answer-holder).
        The set of injected names must stay stable across a session (it does for
        a fixed engine); a brand-new name has no stub in the warm namespace."""
        functions: dict[str, Callable[..., Any]] = {}
        objects: dict[str, dict[str, Callable[..., Any]]] = {}
        for name, value in names.items():
            if name == "corpus":
                if not self._started:
                    self._corpus = value
            elif callable(value):
                functions[name] = value
            elif _is_jsonable(value):
                if not self._started:
                    self._data[name] = value
            else:
                objects[name] = {
                    attr: getattr(value, attr)
                    for attr in dir(value)
                    if not attr.startswith("_") and callable(getattr(value, attr))
                }
        # Callables/objects are always (re)bound; corpus + plain data only matter
        # before start (they're baked into the container at the init handshake).
        self._functions.update(functions)
        self._objects.update(objects)
        # Delegate corpus.search() to the host: the sandbox keeps read/grep/peek
        # local over the read-only mount, but search() needs the loaded index and
        # (for dense) network the network-less sandbox can't reach. Registering it
        # as a host function means the agent binds it as a method RPC stub. Listed
        # in `functions`, so it ships in the init handshake.
        if self._corpus is not None and "corpus.search" not in self._functions:
            self._functions["corpus.search"] = self._corpus_search

    def _corpus_search(self, query, k: int = 10, mode: str = "hybrid"):
        """Run the real search on the host corpus and return JSON-able hit dicts
        (the agent rewraps them as Hit-like objects inside the sandbox)."""
        hits = self._corpus.search(query, k=k, mode=mode)
        return [{"doc": h.doc, "score": h.score, "snippet": h.snippet,
                 "span": list(h.span) if h.span else None} for h in hits]

    def execute(self, code: str) -> ExecResult:
        self._ensure_started()
        assert self._conn is not None
        protocol.send_msg(self._conn, {"op": "execute", "code": code})
        msg = self._pump()
        if msg is None:
            return ExecResult(
                stdout="",
                error=self._terminated_message(),
            )
        return ExecResult(stdout=msg.get("stdout", ""), error=msg.get("error"))

    def describe_state(self, max_vars: int = 40) -> str:
        if not self._started or self._conn is None:
            return ""
        try:
            protocol.send_msg(self._conn, {"op": "describe_state", "max_vars": max_vars})
            msg = self._pump()
        except OSError:
            return ""
        return (msg or {}).get("text", "")

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Tear down the container and clean up. Idempotent."""
        self._teardown()

    def __enter__(self) -> "SandboxEnvironment":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- RPC pump -------------------------------------------------------

    def _pump(self) -> dict | None:
        """Read messages, servicing `rpc` callbacks, until a terminal message
        (`result`/`value`) arrives. None means the agent/container went away."""
        assert self._conn is not None
        while True:
            msg = protocol.recv_msg(self._conn)
            if msg is None:
                return None
            if msg.get("op") == "rpc":
                self._handle_rpc(msg)
                continue
            return msg

    def _handle_rpc(self, msg: dict) -> None:
        assert self._conn is not None
        key = msg.get("key", "")
        args = msg.get("args", [])
        kwargs = msg.get("kwargs", {})
        try:
            target = self._resolve_target(key)
            value = target(*args, **kwargs)
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = str(value)  # helpers return str/None, but be defensive
            protocol.send_msg(self._conn, {"op": "rpc_result", "value": value})
        except Exception as exc:  # surface as an exception inside the sandbox
            protocol.send_msg(
                self._conn,
                {"op": "rpc_error", "error": f"{type(exc).__name__}: {exc}"},
            )

    def _resolve_target(self, key: str) -> Callable[..., Any]:
        # Exact-match a registered function first, so dotted delegates like
        # "corpus.search" resolve as host functions rather than object proxies.
        if key in self._functions:
            return self._functions[key]
        if "." in key:
            obj, method = key.split(".", 1)
            return self._objects[obj][method]
        return self._functions[key]

    # -- startup --------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        if self._corpus is None:
            raise RuntimeError(
                "The gVisor sandbox backend supports a folder-mounted `corpus` only; "
                "it cannot mount an in-memory `context`. Use the in-process backend "
                "(VOMERO_EXEC_BACKEND=inprocess) for context-as-a-variable runs."
            )

        self._tmpdir = tempfile.mkdtemp(prefix="vomero-sock-")
        sock_path = os.path.join(self._tmpdir, _SOCK_NAME)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(sock_path)
        listener.listen(1)
        listener.settimeout(self.config.startup_timeout)
        # The container drops ALL capabilities (incl. CAP_DAC_OVERRIDE), so even
        # its root can't traverse the default 0700 temp dir — open up just this
        # socket dir/file so the agent can connect. It holds nothing but the
        # socket, and the gVisor sandbox is what actually contains the code.
        os.chmod(self._tmpdir, 0o777)
        os.chmod(sock_path, 0o666)
        self._listener = listener

        cmd, corpus_root_in_sandbox, corpus_src_in_sandbox = self._build_launch(sock_path)
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        except FileNotFoundError as exc:
            self._teardown()
            raise RuntimeError(
                f"could not launch the sandbox ({cmd[0]!r} not found). Install it, "
                "or set VOMERO_EXEC_BACKEND=inprocess to run without a sandbox."
            ) from exc

        try:
            conn, _ = listener.accept()
            self._conn = conn
            allow = getattr(self._corpus, "_allow", None)
            protocol.send_msg(conn, {
                "op": "init",
                "corpus": {
                    "root": corpus_root_in_sandbox,
                    "allow": list(allow) if allow is not None else None,
                },
                "functions": list(self._functions),
                "objects": {name: list(m) for name, m in self._objects.items()},
                "data": self._data,
            })
            ready = protocol.recv_msg(conn)
            if not ready or ready.get("op") != "ready":
                raise RuntimeError(self._startup_failure_message())
        except socket.timeout:
            msg = self._startup_failure_message()
            self._teardown()
            raise RuntimeError(msg) from None
        except Exception:
            self._teardown()
            raise

        # Fully up: register teardown to run when this env is GC'd (the engine
        # never explicitly closes it). Closing the socket EOFs the agent, which
        # exits and lets `docker run --rm` remove the container.
        self._finalizer = weakref.finalize(
            self, _teardown_resources,
            self._proc, self._listener, self._conn, self._tmpdir,
        )
        self._started = True

    def _build_launch(self, sock_path: str) -> tuple[list[str], str, str]:
        """Return (argv, corpus-root-as-seen-by-agent, corpus-src-as-seen).

        Two runners: "docker" (a real gVisor container, with bind mounts) and
        "local" (the agent as a plain host subprocess — test only, no
        isolation, no mounts, paths are the host's real paths)."""
        agent_dir = Path(__file__).resolve().parent
        agent_py = agent_dir / "agent.py"
        # The whole `context/` package dir (not just corpus.py): corpus.py uses
        # relative imports (`.source`, `.search`), so it must load as a package
        # submodule. Every file in it is stdlib-only, so it runs on a bare image.
        corpus_src = Path(_corpus_module.__file__).resolve().parent
        corpus_root = str(Path(self._corpus.root))

        if self.config.runner == "local":
            import sys
            cmd = [sys.executable, str(agent_py), sock_path, str(corpus_src)]
            return cmd, corpus_root, str(corpus_src)

        cfg = self.config
        # Run as the host user by default: the corpus is bind-mounted read-only
        # and owned by the host user, and a capability-less container root can't
        # traverse it (no CAP_DAC_OVERRIDE). Matching the uid also avoids running
        # the model's code as root inside the sandbox.
        user = cfg.user
        if user is None and hasattr(os, "getuid"):
            user = f"{os.getuid()}:{os.getgid()}"
        # gVisor refuses host Unix-socket connections by default; enable just
        # "open" (connect to existing) so the agent can reach our bind-mounted
        # control socket. Per-container annotation -> no daemon.json change.
        uds_annotation = (
            ["--annotation", f"dev.gvisor.flag.host-uds={cfg.host_uds}"]
            if cfg.runtime == "runsc" and cfg.host_uds
            else []
        )
        # A durable workspace (if configured) is mounted read-write and becomes
        # the working dir, so the model's files persist past teardown; otherwise
        # the cwd is the ephemeral /tmp tmpfs.
        workspace_mount = (
            ["-v", f"{cfg.workspace}:{_CONTAINER_WORKSPACE}:rw"]
            if cfg.workspace else []
        )
        workdir = _CONTAINER_WORKSPACE if cfg.workspace else "/tmp"
        cmd = [
            cfg.docker_path, "run", "--rm", "-i",
            "--runtime", cfg.runtime,
            *uds_annotation,
            "--network", cfg.network,
            *(["--user", user] if user else []),
            "--memory", cfg.memory,
            "--memory-swap", cfg.memory,        # == memory -> swap disabled, hard cap
            "--cpus", str(cfg.cpus),
            "--pids-limit", str(cfg.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",                       # immutable rootfs...
            "--tmpfs", f"/tmp:rw,size={cfg.tmpfs_size}",  # ...except a small /tmp
            *workspace_mount,
            "-w", workdir,
            "-v", f"{corpus_root}:{_CONTAINER_CORPUS}:ro",
            "-v", f"{agent_dir}:{_CONTAINER_AGENT_DIR}:ro",
            "-v", f"{corpus_src}:{_CONTAINER_CORPUS_SRC}:ro",
            "-v", f"{self._tmpdir}:{_CONTAINER_SOCK_DIR}",
            *cfg.extra_run_args,
            cfg.image,
            "python", f"{_CONTAINER_AGENT_DIR}/agent.py",
            f"{_CONTAINER_SOCK_DIR}/{_SOCK_NAME}", _CONTAINER_CORPUS_SRC,
        ]
        return cmd, _CONTAINER_CORPUS, _CONTAINER_CORPUS_SRC

    # -- diagnostics ----------------------------------------------------

    def _terminated_message(self) -> str:
        """Explain a mid-step disconnect (the common one is an OOM kill)."""
        detail = self._drain_logs()
        base = ("sandbox terminated unexpectedly mid-execution — the code may "
                f"have exceeded the {self.config.memory} memory cap (OOM-killed) "
                "or crashed the interpreter")
        return f"{base}\n{detail}" if detail else base

    def _startup_failure_message(self) -> str:
        detail = self._drain_logs()
        base = (
            f"sandbox container did not start within {self.config.startup_timeout}s. "
            f"Check that Docker is running and the {self.config.runtime!r} runtime "
            "is registered (gVisor), and that the image is available."
        )
        return f"{base}\nContainer output:\n{detail}" if detail else base

    def _drain_logs(self) -> str:
        """Best-effort read of whatever the container printed (errors, tracebacks)."""
        if self._proc is None or self._proc.stdout is None:
            return ""
        try:
            self._proc.poll()
            data = self._proc.stdout.read() if self._proc.poll() is not None else b""
            return data.decode("utf-8", "replace").strip()
        except Exception:
            return ""

    def _teardown(self) -> None:
        """Tear down whatever has been set up so far (failure paths + close())."""
        if self._finalizer is not None:
            self._finalizer()  # the fully-started path: runs exactly once
            return
        _teardown_resources(self._proc, self._listener, self._conn, self._tmpdir)


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _teardown_resources(
    proc: subprocess.Popen | None,
    listener: socket.socket | None,
    conn: socket.socket | None,
    tmpdir: str | None,
) -> None:
    """Close everything and reap the container. Module-level so weakref.finalize
    can hold it without capturing `self`. Closing the socket EOFs the agent,
    which exits and lets `docker run --rm` remove the container."""
    for sock in (conn, listener):
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)
