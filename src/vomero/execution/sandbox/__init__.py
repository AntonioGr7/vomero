"""gVisor-isolated execution backend.

A drop-in `ExecutionEnvironment` that runs the model's code inside a gVisor
(`runsc`) container with hard memory/CPU caps and no network, instead of
in-process. Optional — the default backend stays `InProcessEnvironment`.

The package is self-contained: only `environment.py` (the host side) is imported
into the rest of Vomero; `agent.py` runs standalone inside the container.
"""

from __future__ import annotations

from .config import SandboxConfig
from .environment import SandboxEnvironment

__all__ = ["SandboxConfig", "SandboxEnvironment"]
