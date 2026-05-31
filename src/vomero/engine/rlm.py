"""RLMEngine — the recursive REPL loop.

Flow (one `run`):
  1. Spin up an ExecutionEnvironment and inject the navigation surface:
     `corpus`, `llm(...)`, `rlm(...)`, `answer(...)`.
  2. Give the root model a system prompt describing that surface, plus the
     user's question. It can ONLY act via the `python` tool.
  3. Each step: model -> python(code) -> we exec -> feed stdout/traceback back.
  4. Stop when the model calls `answer(...)` from the REPL, or replies with
     plain text (no tool call). Either becomes the final answer.

The recursion: `llm()` is a flat sub-call (cheap distillation of a chunk);
`rlm()` re-enters this engine on a (optionally scoped) corpus at depth+1, so a
sub-question gets the same full power. Depth is capped to keep it finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..context.corpus import Corpus
from ..env import ExecutionEnvironment, InProcessEnvironment
from ..llm.base import LLMClient, Message, ToolSpec
from ..usage import UsageMeter, UsageSnapshot, estimate_message_tokens
from .compaction import Compactor, CompactionEvent
from .todo import TodoItem, TodoList

# The single tool the root model gets. Keeping it to one tool (run Python)
# maps cleanly onto every provider's function-calling and matches the RLM idea:
# the model's lever on the world is code.
PYTHON_TOOL = ToolSpec(
    name="python",
    description=(
        "Run Python in a persistent REPL to explore the data and build your "
        "answer. State (variables, imports) persists across calls. Use print() "
        "to see anything. Call answer(text) when you are done."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute."}
        },
        "required": ["code"],
    },
)

SYSTEM_PROMPT = """You are Vomero, an assistant that answers questions about a \
collection of files WITHOUT loading them into your own context. The data is too \
large and too important to paste into this conversation. Instead you operate a \
Python REPL via the `python` tool and reason *programmatically* over the data.

Your REPL already has these names available:

  corpus        A read-only handle on the data folder. Key methods:
                  corpus.overview()           -> summary + file list (start here)
                  corpus.tree()               -> file tree
                  corpus.files(glob="**/*")   -> list of relative paths
                  corpus.grep(pattern, ...)   -> regex search -> [Match(path, lineno, line)]
                  corpus.peek(path, lines=40) -> first lines of a file
                  corpus.read(path)           -> full text of a file
                  corpus.size(path)           -> bytes
                  corpus.subset([paths])      -> a corpus scoped to those files

  llm(text, system=None) -> str
                A single, fresh model call with NO memory and NO tools. Use it to
                distill a chunk you have already read into a variable, e.g.
                summarize, extract, or answer a narrow question about that text.
                The chunk you pass is the ONLY thing that sub-call sees.

  rlm(question, paths=None) -> str
                A recursive Vomero call on the (optionally scoped) corpus. Use it
                to delegate a self-contained sub-question that itself needs
                exploration. Returns the sub-answer as a string.

  answer(text)  Record your FINAL answer and finish.

Strategy:
  - Start by calling corpus.overview() (print it) to see what you have.
  - Locate relevant files with grep/files; peek before reading whole files.
  - NEVER print the full text of a large file into the transcript. Read it into
    a variable and pass it to llm()/rlm() to distill — keep raw text out of your
    own context.
  - For multi-hop questions, chain: find A, then use what you learned to find B,
    cross-check, aggregate.
  - Verify before answering: re-grep to confirm claims, count, cite file paths.
  - When confident, call answer("..."), citing the file paths you relied on.

Keep each code block small and purposeful. Print only what you need to see."""


# Appended to the system prompt only when planning is enabled.
PLANNING_PROMPT = """

You also have a TODO surface to externalize your plan — the user watches it live:

  todo.plan([...])   Set your step-by-step plan. Call this FIRST, before exploring.
  todo.start(n)      Mark item n (1-based) in progress, right before you work on it.
  todo.complete(n)   Mark item n done, right after you finish it.
  todo.add("...")    Append a step you discover mid-task.

Keep the plan to a handful of concrete, verifiable steps. Update it as you go —
exactly one item in progress at a time. You do NOT need to print the list; it is
shown to the user automatically, so keep it out of your printed output."""


# Appended to the system prompt only when interaction is enabled.
INTERACTION_PROMPT = """

You can ask the user for help when you are genuinely stuck:

  ask_user(question) -> str   Pause and ask the user a question; returns their
                              reply as a string.

Use it SPARINGLY and only when proceeding would mean guessing on something that
matters: the request is ambiguous, you are missing information only the user has,
or you face a consequential decision the data cannot resolve. Explore the corpus
first — never ask what you could find yourself. Ask one specific, answerable
question at a time, then capture the reply (e.g. `reply = ask_user(...)`), act on
it, and reflect it in your final answer."""


@dataclass
class LLMCall:
    """A flat `llm()` distillation sub-call — surfaced so the trace explains
    every token spent (these calls move the cumulative total but run inside the
    model's code, so they'd otherwise be invisible)."""

    prompt: str
    response: str
    tokens: int  # cumulative-total delta this call cost


@dataclass
class Interaction:
    """A round-trip where the model asked the user something and got a reply."""

    question: str
    answer: str


@dataclass
class Step:
    """One turn in the loop — handed to `on_event` for observability."""

    depth: int
    index: int
    code: str | None = None
    output: str | None = None
    final: str | None = None
    # Assistant natural-language content emitted alongside a tool call.
    message: str | None = None
    # The current plan, emitted whenever the model mutates the TODO surface.
    todo: list[TodoItem] | None = None
    # Set when the model asked the user a question and received a reply.
    interaction: Interaction | None = None
    # Token-accounting snapshot, emitted right after each model call.
    usage: UsageSnapshot | None = None
    # Set on the step where history was compacted.
    compaction: CompactionEvent | None = None
    # A flat `llm()` sub-call made from within the model's code this step.
    llm_call: LLMCall | None = None


class RLMEngine:
    def __init__(
        self,
        client: LLMClient,
        *,
        env_factory: Callable[[], ExecutionEnvironment] = InProcessEnvironment,
        model: str | None = None,
        max_steps: int = 24,
        max_depth: int = 3,
        compactor: Compactor | None = None,
        enable_planning: bool = False,
        planning_root_only: bool = False,
        enable_interaction: bool = False,
    ):
        self.client = client
        self.env_factory = env_factory
        self.model = model
        self.max_steps = max_steps
        self.max_depth = max_depth
        # When True, inject a `todo` plan surface and tell the model to use it.
        self.enable_planning = enable_planning
        # By default every depth keeps its own plan (recursive sub-agents plan
        # too). Set True to give the plan surface to the root agent only.
        self.planning_root_only = planning_root_only
        # When True, inject `ask_user` and tell the model it may ask for help.
        self.enable_interaction = enable_interaction
        # Optional history compaction. None => never compact (small corpora,
        # tests). Shared across the recursion; each loop compacts its own stack.
        self.compactor = compactor
        # Token accounting for the most recent top-level run. The reference is
        # stored as soon as the run starts and stays live, so callers can read
        # final totals off `engine.last_usage` after `run` returns.
        self.last_usage: UsageMeter | None = None

    def run(
        self,
        question: str,
        corpus: Corpus,
        *,
        depth: int = 0,
        on_event: Callable[[Step], None] | None = None,
        meter: UsageMeter | None = None,
        ask_handler: Callable[[str], str] | None = None,
    ) -> str:
        # Top-level run owns a fresh meter; recursive sub-calls share it so the
        # cumulative figure spans the whole tree.
        meter = meter or UsageMeter()
        if depth == 0:
            self.last_usage = meter

        env = self.env_factory()
        holder: dict[str, str] = {}
        # Live step index, so sub-calls made from inside the model's code (which
        # runs mid-step) can tag their events with the step they belong to.
        current: dict[str, int] = {"index": 0}

        def answer(text) -> None:
            holder["value"] = str(text)

        def ask_user(question: str) -> str:
            q = str(question)
            if ask_handler is None:
                # Headless: don't hang. Tell the model to proceed autonomously.
                reply = ("No user is available to answer (running "
                         "non-interactively). Proceed with your best judgment "
                         "and state any assumptions in your answer.")
            else:
                reply = str(ask_handler(q))
            if on_event:
                on_event(Step(depth=depth, index=current["index"],
                              interaction=Interaction(question=q, answer=reply)))
            return reply

        def llm(text: str, system: str | None = None) -> str:
            msgs = []
            if system:
                msgs.append(Message("system", system))
            msgs.append(Message("user", text))
            before = meter.total_tokens
            resp = self.client.complete(msgs, model=self.model)
            # A flat sub-call still spends tokens; fold it into the shared total.
            meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
            out = resp.content or ""
            if on_event:
                on_event(Step(
                    depth=depth, index=current["index"],
                    llm_call=LLMCall(prompt=text, response=out,
                                     tokens=meter.total_tokens - before),
                ))
            return out

        def rlm(sub_question: str, paths: list[str] | None = None) -> str:
            if depth + 1 > self.max_depth:
                # Out of recursion budget: degrade to a flat call so we still
                # make progress instead of failing.
                return llm(sub_question)
            sub_corpus = corpus.subset(paths) if paths else corpus
            return self.run(
                sub_question,
                sub_corpus,
                depth=depth + 1,
                on_event=on_event,
                meter=meter,
                ask_handler=ask_handler,
            )

        names = dict(corpus=corpus, llm=llm, rlm=rlm, answer=answer)
        system_prompt = SYSTEM_PROMPT
        planning_here = self.enable_planning and (not self.planning_root_only or depth == 0)
        if planning_here:
            system_prompt += PLANNING_PROMPT

            def on_todo_change(items: list[TodoItem]) -> None:
                if on_event:
                    on_event(Step(depth=depth, index=current["index"], todo=items))

            names["todo"] = TodoList(on_todo_change)
        if self.enable_interaction:
            system_prompt += INTERACTION_PROMPT
            names["ask_user"] = ask_user
        env.inject(**names)

        messages: list[Message] = [
            Message("system", system_prompt),
            Message("user", question),
        ]

        for i in range(self.max_steps):
            current["index"] = i
            resp = self.client.complete(messages, tools=[PYTHON_TOOL], model=self.model)
            # Record before appending the reply: `messages` is exactly the
            # context we just sent, so its size is this step's context gauge.
            context_tokens, context_estimated = meter.record(
                resp.usage, sent_messages=messages, response_text=resp.content
            )
            if on_event:
                on_event(Step(
                    depth=depth, index=i,
                    usage=meter.snapshot(context_tokens, context_estimated),
                ))
                # Natural-language the model emitted alongside its tool call.
                if resp.content and resp.tool_calls:
                    on_event(Step(depth=depth, index=i, message=resp.content))
            step_start = len(messages)  # for projecting this step's growth
            messages.append(
                Message("assistant", content=resp.content, tool_calls=resp.tool_calls)
            )

            # No tool call => the model is answering directly.
            if not resp.tool_calls:
                final = resp.content or holder.get("value", "")
                if on_event:
                    on_event(Step(depth=depth, index=i, final=final))
                return final

            for tc in resp.tool_calls:
                code = tc.arguments.get("code", "")
                if on_event:
                    on_event(Step(depth=depth, index=i, code=code))
                result = env.execute(code)
                output = result.stdout
                if result.error:
                    output = (output + "\n" if output else "") + result.error
                output = output.strip() or "(no output)"
                if on_event:
                    on_event(Step(depth=depth, index=i, output=output))
                messages.append(
                    Message("tool", content=output, tool_call_id=tc.id)
                )

            if "value" in holder:
                if on_event:
                    on_event(Step(depth=depth, index=i, final=holder["value"]))
                return holder["value"]

            # Compaction check. `context_tokens` is the authoritative size of
            # what we just sent; project this step's appended output on top so a
            # single large tool result triggers a compaction before the next
            # (now-oversized) call rather than after it.
            if self.compactor is not None:
                added = estimate_message_tokens(messages[step_start:])
                projected = context_tokens + added
                if self.compactor.should_compact(projected):
                    before_n = len(messages)
                    # Calibrate the rough char-estimator to the authoritative
                    # size we actually measured, so the event's before/after are
                    # on the same scale as the live token gauge.
                    est_before = estimate_message_tokens(messages) or 1
                    scale = projected / est_before
                    messages, summarized = self.compactor.compact(
                        messages,
                        client=self.client,
                        model=self.model,
                        meter=meter,
                        state_description=env.describe_state(),
                    )
                    if summarized and on_event:
                        on_event(Step(
                            depth=depth,
                            index=i,
                            compaction=CompactionEvent(
                                tokens_before=round(projected),
                                tokens_after=round(estimate_message_tokens(messages) * scale),
                                messages_before=before_n,
                                messages_after=len(messages),
                                summarized_messages=summarized,
                            ),
                        ))

        # Ran out of steps — return whatever we have.
        return holder.get("value") or "Stopped: reached max steps without a final answer."
