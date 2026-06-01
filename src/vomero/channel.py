"""Transport seam between the engine and whatever it's talking to.

The engine doesn't know if it's driving a shell, a test harness, or a browser.
It depends only on a `Channel`: somewhere to emit progress events, and a way to
ask the human user a question. Swap the implementation to change the frontend.

* `NullChannel` — drops events; no human (ask_user degrades). Safe default.
* `CallbackChannel` — adapts the loose `on_event` / `ask_handler` callbacks the
  CLI already uses to the interface.
* A browser/server frontend implements `emit` (serialize + push the Step) and
  `ask_user` (round-trip to the UI — e.g. block on a queue a websocket fills).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .engine.rlm import Step

# Returned by ask_user when no human is reachable, so headless runs never hang.
NO_USER_REPLY = (
    "No user is available to answer (running non-interactively). Proceed with "
    "your best judgment and state any assumptions in your answer."
)


@runtime_checkable
class Channel(Protocol):
    """What the engine needs from its frontend."""

    def emit(self, step: "Step") -> None:
        """Receive one observability event (progress, usage, plan, ...)."""

    def ask_user(self, question: str) -> str:
        """Ask the human a question and return their reply (may block)."""


class NullChannel:
    """No observers and no human. Safe default for library/headless use."""

    def emit(self, step: "Step") -> None:
        pass

    def ask_user(self, question: str) -> str:
        return NO_USER_REPLY


class CallbackChannel:
    """Adapts plain callbacks to a Channel (how the CLI wires its printers)."""

    def __init__(
        self,
        on_event: Callable[["Step"], None] | None = None,
        ask_handler: Callable[[str], str] | None = None,
    ):
        self._on_event = on_event
        self._ask_handler = ask_handler

    def emit(self, step: "Step") -> None:
        if self._on_event is not None:
            self._on_event(step)

    def ask_user(self, question: str) -> str:
        if self._ask_handler is None:
            return NO_USER_REPLY
        return str(self._ask_handler(question))
