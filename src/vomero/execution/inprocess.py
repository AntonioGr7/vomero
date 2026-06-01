"""In-process execution environment.

Runs the model's code with `exec` in a persistent dict namespace, capturing
stdout+stderr. NOT sandboxed: the code can touch the filesystem and import
anything. That is acceptable for a personal, trusted tool and is the explicit
v0 trade-off (docs/adr/0001). The `ExecutionEnvironment` interface is the seam
where a real sandbox slots in later.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any

from .base import ExecResult, ExecutionEnvironment


class InProcessEnvironment(ExecutionEnvironment):
    def __init__(self) -> None:
        # A fresh module-like namespace. Builtins are available by default.
        self._ns: dict[str, Any] = {"__name__": "__vomero_repl__"}
        # Names the host injected (corpus/llm/rlm/answer). Tracked so
        # `describe_state` can report only the *model's own* variables.
        self._injected: set[str] = set()

    def inject(self, **names: Any) -> None:
        self._ns.update(names)
        self._injected.update(names)

    def execute(self, code: str) -> ExecResult:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<vomero>", "exec"), self._ns)
            return ExecResult(stdout=buf.getvalue())
        except Exception:
            # Surface the traceback to the model so it can self-correct.
            return ExecResult(stdout=buf.getvalue(), error=traceback.format_exc())

    def describe_state(self, max_vars: int = 40) -> str:
        lines: list[str] = []
        for name, value in self._ns.items():
            if name.startswith("__") or name in self._injected:
                continue
            lines.append(f"  {name} : {self._summarize_value(value)}")
            if len(lines) >= max_vars:
                lines.append("  … (more variables omitted)")
                break
        return "\n".join(lines)

    @staticmethod
    def _summarize_value(value: Any) -> str:
        """A compact, safe one-line description of a REPL value."""
        type_name = type(value).__name__
        if callable(value) and not isinstance(value, (str, bytes)):
            return f"{type_name} (callable)"
        try:
            return f"{type_name} (len={len(value)})"  # sized containers/strings
        except TypeError:
            pass
        try:
            text = repr(value)
        except Exception:
            return type_name
        if len(text) > 60:
            text = text[:57] + "…"
        return f"{type_name} = {text}"
