"""Execution environments — where the model's code actually runs.

The engine depends only on the `ExecutionEnvironment` ABC. The default backend
is in-process `exec` (fast, full-power, NOT sandboxed). An optional gVisor
sandbox (`sandbox/`) implements the same interface with real isolation.

`build_env_factory(settings)` is the single place that picks a backend from
config, so the CLI and server stay identical.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import ExecResult, ExecutionEnvironment
from .inprocess import InProcessEnvironment
from .session import SessionEnvPool, build_session_pool

__all__ = [
    "ExecResult",
    "ExecutionEnvironment",
    "InProcessEnvironment",
    "SessionEnvPool",
    "build_env_factory",
    "build_session_pool",
]


def build_env_factory(settings: Any) -> Callable[[], ExecutionEnvironment]:
    """Map `settings` to an `env_factory` for `RLMEngine`.

    Returns `InProcessEnvironment` unless `settings.exec_backend == "sandbox"`,
    in which case it returns a thunk that builds a fresh `SandboxEnvironment`
    (one container per run/recursion) from the `sandbox_*` settings. The sandbox
    module is imported lazily so nothing changes for the default path."""
    if getattr(settings, "exec_backend", "inprocess") != "sandbox":
        return InProcessEnvironment

    from .sandbox import SandboxConfig, SandboxEnvironment

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
    )
    return lambda: SandboxEnvironment(config)
