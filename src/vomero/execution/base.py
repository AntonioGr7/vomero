"""The execution-environment contract the engine codes against."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecResult:
    """Outcome of running one code snippet."""

    stdout: str
    error: str | None = None  # formatted traceback, or None on success

    @property
    def ok(self) -> bool:
        return self.error is None


class ExecutionEnvironment(ABC):
    """A persistent namespace the model can run code in across many steps.

    Implementations must keep state between `execute` calls (so variables and
    imports persist), and must let the host inject names (the `corpus`, the
    `llm`/`rlm` helpers, `answer`, ...) via `inject`.
    """

    @abstractmethod
    def inject(self, **names: Any) -> None:
        """Add/overwrite names in the persistent namespace."""

    @abstractmethod
    def execute(self, code: str) -> ExecResult:
        """Run `code` in the persistent namespace, capturing stdout/stderr."""

    def describe_state(self, max_vars: int = 40) -> str:
        """Describe user-defined names live in the namespace, one per line.

        Used by compaction: because the REPL namespace SURVIVES compaction, the
        summary must tell the model which variables are still defined so it does
        not recompute them. Default is empty (backends may opt in)."""
        return ""
