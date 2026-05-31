"""A lightweight plan/TODO surface the model drives from the REPL.

The model externalizes its plan by calling `todo.plan([...])` and then marking
items as it works (`todo.start(n)` / `todo.complete(n)`). Every mutation fires a
callback so the host can render the evolving checklist in the shell — the same
"here's my plan, watch me tick it off" view Claude Code shows.

Design notes:
- Mutations DON'T return the rendered list and the model is told it needn't
  print it: the checklist is host-side observability and must stay OUT of the
  model's context (printing it would spend tokens on every update).
- Indices are 1-based to match what the user sees.
- This is opt-in (see `RLMEngine(enable_planning=...)`); when off, `todo` is not
  injected at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"


@dataclass
class TodoItem:
    text: str
    status: str = PENDING


class TodoList:
    def __init__(self, on_change: Callable[[list[TodoItem]], None]):
        self._items: list[TodoItem] = []
        self._on_change = on_change

    # -- mutation (each fires a render) ---------------------------------
    def plan(self, items: list[str]) -> None:
        """Set the plan to `items` (replaces any existing plan)."""
        self._items = [TodoItem(str(t)) for t in items]
        self._changed()

    def add(self, text: str) -> None:
        """Append a newly-discovered step."""
        self._items.append(TodoItem(str(text)))
        self._changed()

    def start(self, index: int) -> None:
        """Mark item `index` (1-based) in progress."""
        self._at(index).status = IN_PROGRESS
        self._changed()

    def complete(self, index: int) -> None:
        """Mark item `index` (1-based) completed."""
        self._at(index).status = COMPLETED
        self._changed()

    # -- internals ------------------------------------------------------
    def _at(self, index: int) -> TodoItem:
        if not isinstance(index, int) or not (1 <= index <= len(self._items)):
            raise ValueError(
                f"todo index {index!r} out of range (valid: 1..{len(self._items)})"
            )
        return self._items[index - 1]

    def _changed(self) -> None:
        # Hand the callback a copy so a later mutation can't alter an event
        # snapshot that was already emitted.
        self._on_change([TodoItem(it.text, it.status) for it in self._items])

    def __repr__(self) -> str:
        done = sum(1 for it in self._items if it.status == COMPLETED)
        return f"<TodoList {done}/{len(self._items)} done>"
