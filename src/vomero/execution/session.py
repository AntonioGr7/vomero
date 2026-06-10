"""Per-session environment reuse — durable variables + workspace across turns.

By default every `engine.run()` builds a fresh `ExecutionEnvironment` and throws
it away, so nothing survives between interactions. `SessionEnvPool`
keeps ONE environment alive per `(user_id, session_id)` so a follow-up resumes
the model's live REPL variables and its workspace files — the engine's `env=`
override hands the pooled env back into the loop.

Lifetime — two clocks, deliberately different:

  * Variables (the live env: an in-process namespace, or a warm gVisor
    container) are evicted after `ttl_seconds` of idle time. After that the env
    is closed and the next turn starts with a COLD namespace.
  * Workspace files live in a real host directory per session and are NOT
    deleted on eviction — so a turn after the TTL still sees the files the model
    wrote, just without the variables. Call `discard(key, remove_workspace=True)`
    to drop them too.

Concurrency: one env per session is a single namespace/container, so turns of
the same session must not run concurrently. `session()` hands out a per-session
lock; eviction skips a session whose lock is held, so a long run is never torn
down from under itself.

Heavy-load bound: `max_sessions` caps how many warm envs the pool keeps. When a
new session would exceed it, the least-recently-used IDLE env is closed to make
room (LRU). In-flight sessions are never evicted, so a burst of active runs can
briefly exceed the cap — the *server's* concurrency semaphore bounds active
load; this cap bounds the memory held by warm/idle containers between turns.
0 = unlimited (the default).
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .base import ExecutionEnvironment

# (user_id, session_id)
SessionKey = tuple[str, str]

_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_component(s: str) -> str:
    """Sanitize a key part into a single safe path segment (no traversal)."""
    return _SAFE.sub("_", s) or "_"


def _close(env: ExecutionEnvironment) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


@dataclass
class _Entry:
    env: ExecutionEnvironment
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = 0.0


class SessionEnvPool:
    """Reuses an `ExecutionEnvironment` per session, with idle-TTL eviction."""

    def __init__(
        self,
        make_env: Callable[[SessionKey, str | None], ExecutionEnvironment],
        *,
        ttl_seconds: float = 900.0,
        workspace_root: str | None = None,
        max_sessions: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # make_env(key, workspace_path|None) -> a fresh environment.
        self._make_env = make_env
        self._ttl = ttl_seconds
        self._workspace_root = workspace_root
        # Max warm envs kept alive; 0 = unlimited. LRU-evicted past this cap.
        self._max_sessions = max_sessions
        self._clock = clock
        self._entries: dict[SessionKey, _Entry] = {}
        self._lock = threading.Lock()
        self._sweeper_stop: threading.Event | None = None

    # -- workspace ------------------------------------------------------

    def workspace_for(self, key: SessionKey) -> str | None:
        """The session's durable workspace dir (created on demand), or None when
        no workspace_root is configured."""
        if not self._workspace_root:
            return None
        path = os.path.join(self._workspace_root, *map(_safe_component, key))
        os.makedirs(path, exist_ok=True)
        return path

    # -- acquire / use --------------------------------------------------

    @contextmanager
    def session(self, key: SessionKey) -> Iterator[ExecutionEnvironment]:
        """Yield the session's persistent env, holding its per-session lock for
        the duration so turns of one session serialize and can't be evicted
        mid-run. Reuses an existing env or builds one (resuming the workspace)."""
        entry = self._acquire(key)
        with entry.lock:
            entry.last_used = self._clock()
            try:
                yield entry.env
            finally:
                entry.last_used = self._clock()

    def _acquire(self, key: SessionKey) -> _Entry:
        now = self._clock()
        with self._lock:
            self._evict_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                if self._max_sessions:
                    self._evict_lru_over_cap()
                env = self._make_env(key, self.workspace_for(key))
                entry = _Entry(env=env, last_used=now)
                self._entries[key] = entry
            return entry

    def _evict_lru_over_cap(self) -> None:
        """Make room for one new session by closing the least-recently-used IDLE
        env while at/over `max_sessions`. Caller holds `self._lock`. Sessions
        with a run in flight (lock held) are never evicted, so the cap may be
        transiently exceeded under a burst of active runs."""
        while len(self._entries) >= self._max_sessions:
            idle = [(e.last_used, k) for k, e in self._entries.items()
                    if not e.lock.locked()]
            if not idle:
                return  # everything is in use; let the new env push us over
            _, key = min(idle)
            entry = self._entries.pop(key)
            _close(entry.env)

    # -- eviction / lifecycle -------------------------------------------

    def _evict_expired(self, now: float) -> None:
        """Close envs idle past the TTL. Caller holds `self._lock`. A session
        whose lock is held (a run in flight) is left alone."""
        for k, entry in list(self._entries.items()):
            if now - entry.last_used <= self._ttl:
                continue
            if entry.lock.locked():  # a turn is using it; skip this round
                continue
            self._entries.pop(k, None)
            _close(entry.env)

    def sweep(self) -> None:
        """Evict idle sessions now (used by the background sweeper)."""
        with self._lock:
            self._evict_expired(self._clock())

    def discard(self, key: SessionKey, *, remove_workspace: bool = False) -> None:
        """Forget a session: close its env and optionally delete its workspace."""
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is not None:
            _close(entry.env)
        if remove_workspace and self._workspace_root:
            path = os.path.join(self._workspace_root, *map(_safe_component, key))
            shutil.rmtree(path, ignore_errors=True)

    def close_all(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            _close(entry.env)

    def start_sweeper(self, interval: float = 60.0) -> Callable[[], None]:
        """Run a daemon thread that evicts idle sessions every `interval`s, so
        variables are reclaimed on time even with no new traffic. Returns a
        function that stops it. Idempotent: a second call replaces the first."""
        self.stop_sweeper()
        stop = threading.Event()
        self._sweeper_stop = stop

        def loop() -> None:
            while not stop.wait(interval):
                self.sweep()

        threading.Thread(target=loop, name="vomero-session-sweeper", daemon=True).start()
        return self.stop_sweeper

    def stop_sweeper(self) -> None:
        if self._sweeper_stop is not None:
            self._sweeper_stop.set()
            self._sweeper_stop = None


def build_session_pool(
    settings: Any,
    *,
    ttl_seconds: float | None = None,
    workspace_root: str | None = None,
) -> SessionEnvPool:
    """Wire a pool from `settings`, picking the backend the same way the engine
    does. For the sandbox backend the per-session workspace dir is mounted into
    the container read-write; in-process keeps variables by reusing the same
    namespace object (its file writes go to the host cwd as before)."""
    ttl = ttl_seconds if ttl_seconds is not None else getattr(settings, "session_ttl", 900.0)
    root = workspace_root if workspace_root is not None else getattr(settings, "workspace_root", None)

    if getattr(settings, "exec_backend", "inprocess") != "gvisor":
        from .inprocess import InProcessEnvironment

        def make_env(key: SessionKey, workspace: str | None) -> ExecutionEnvironment:
            return InProcessEnvironment()
    else:
        from .sandbox import SandboxConfig, SandboxEnvironment

        def make_env(key: SessionKey, workspace: str | None) -> ExecutionEnvironment:
            config = SandboxConfig(
                image=getattr(settings, "sandbox_image", SandboxConfig.image),
                runtime=getattr(settings, "sandbox_runtime", SandboxConfig.runtime),
                memory=getattr(settings, "sandbox_memory", SandboxConfig.memory),
                cpus=getattr(settings, "sandbox_cpus", SandboxConfig.cpus),
                network=getattr(settings, "sandbox_network", SandboxConfig.network),
                pids_limit=getattr(settings, "sandbox_pids_limit", SandboxConfig.pids_limit),
                startup_timeout=getattr(
                    settings, "sandbox_startup_timeout", SandboxConfig.startup_timeout
                ),
                workspace=workspace,
            )
            return SandboxEnvironment(config)

    return SessionEnvPool(
        make_env, ttl_seconds=ttl, workspace_root=root,
        max_sessions=getattr(settings, "max_sessions", 0),
    )
