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
from ..execution import ExecutionEnvironment, InProcessEnvironment
from ..llm.base import LLMClient, Message, ToolSpec
from ..channel import Channel, CallbackChannel
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


# Appended to the system prompt when the model may ask the human user.
ASK_USER_PROMPT = """

You can ask the user for help when you are genuinely stuck:

  ask_user(question) -> str   Pause and ask the user a question; returns their
                              reply as a string.

Use it SPARINGLY and only when proceeding would mean guessing on something that
matters: the request is ambiguous, you are missing information only the user has,
or you face a consequential decision the data cannot resolve. Explore the corpus
first — never ask what you could find yourself. Ask one specific, answerable
question at a time, then capture the reply (e.g. `reply = ask_user(...)`), act on
it, and reflect it in your final answer."""

# Appended for sub-agents (a delegated task that has a parent to consult).
ASK_PARENT_PROMPT = """

You are a sub-task delegated by a parent agent that holds the broader goal and
the original request. When you lack context to proceed — the task is
underspecified, or you need information the parent likely has — ask it first:

  ask_parent(question) -> str   Ask the delegating agent for clarification;
                                returns its reply as a string.

Prefer this over asking the user for anything about the task's intent or scope:
the parent set up this sub-task and can usually answer without involving a human.
Ask the user only for things the parent also wouldn't know. The exchange is added
to your context automatically — you don't need to print it."""


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
    """A round-trip where the model asked for help and got a reply.

    `kind` is "user" (asked the human) or "parent" (a sub-agent asked the agent
    that delegated it)."""

    question: str
    answer: str
    kind: str = "user"


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
        interaction_root_only: bool = False,
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
        # By default any depth may ask the user. Set True to let only the root
        # agent reach the human (sub-agents still consult their parent).
        self.interaction_root_only = interaction_root_only
        # Optional history compaction. None => never compact (small corpora,
        # tests). Shared across the recursion; each loop compacts its own stack.
        self.compactor = compactor
        # NOTE: the engine holds no per-run state. Usage is returned via the
        # caller-owned `meter`, so one engine can serve concurrent runs safely.

    def run(
        self,
        question: str,
        corpus: Corpus,
        *,
        depth: int = 0,
        channel: Channel | None = None,
        meter: UsageMeter | None = None,
        clarify_handler: Callable[[str], str] | None = None,
        # Per-run planning overrides (None => use the engine defaults). Lets a
        # shared server engine turn the plan surface on/off per request.
        enable_planning: bool | None = None,
        planning_root_only: bool | None = None,
        # Conversation continuity (root-level; not used by recursion). `history`
        # seeds this run's transcript with prior turns so a follow-up question
        # builds on them. `transcript_sink`, if given, is cleared and filled with
        # the resumable transcript (every message except the rebuilt system
        # prompt) at the end of the run — pass last run's sink as the next run's
        # `history` to chain turns. The engine still keeps no state of its own;
        # the caller owns the store. See ADR 0004 / server.py.
        history: list[Message] | None = None,
        transcript_sink: list[Message] | None = None,
        # Persistent execution environment (root-level; recursion always gets a
        # fresh one via env_factory). Pass a reused env to keep the model's REPL
        # variables AND its workspace files across turns of the same session —
        # `inject()` rebinds this run's helper closures onto the warm namespace.
        # None => a fresh env from env_factory, the stateless default. The caller
        # owns the env's lifetime (see SessionEnvPool).
        env: ExecutionEnvironment | None = None,
        # Back-compat conveniences; folded into a CallbackChannel when no
        # `channel` is given. New frontends should pass a `channel` instead.
        on_event: Callable[[Step], None] | None = None,
        ask_handler: Callable[[str], str] | None = None,
    ) -> str:
        # The frontend is a single Channel. Pass your own UsageMeter to read
        # token usage after the run — the engine keeps no per-run state, so the
        # same engine instance is safe to use from concurrent threads.
        if channel is None:
            channel = CallbackChannel(on_event=on_event, ask_handler=ask_handler)
        meter = meter or UsageMeter()
        # Resolve per-run planning against the engine defaults (so recursion and
        # the server can override without mutating the shared engine).
        if enable_planning is None:
            enable_planning = self.enable_planning
        if planning_root_only is None:
            planning_root_only = self.planning_root_only

        # A caller-supplied `env` (root only) is reused across turns to keep the
        # model's variables/workspace; otherwise build a fresh, stateless one.
        env = env if env is not None else self.env_factory()
        holder: dict[str, str] = {}
        # Live step index, so sub-calls made from inside the model's code (which
        # runs mid-step) can tag their events with the step they belong to.
        current: dict[str, int] = {"index": 0}
        # Messages to splice into THIS agent's history after the current step's
        # tool results (appending mid-step would split a tool-call/result pair).
        pending_context: list[Message] = []

        def answer(text) -> None:
            holder["value"] = str(text)

        def ask_user(question: str) -> str:
            q = str(question)
            # The channel reaches the human (or degrades gracefully headless).
            reply = channel.ask_user(q)
            channel.emit(Step(depth=depth, index=current["index"],
                              interaction=Interaction(question=q, answer=reply)))
            return reply

        def ask_parent(question: str) -> str:
            # Only injected for sub-agents (clarify_handler is then non-None).
            # The parent answers from a COPY of its context, so this exchange
            # never enters the parent's history — only ours, recorded below.
            q = str(question)
            reply = str(clarify_handler(q))
            channel.emit(Step(depth=depth, index=current["index"],
                              interaction=Interaction(question=q, answer=reply,
                                                      kind="parent")))
            pending_context.append(Message(
                "user",
                f"[Clarification from the parent agent]\nYour question: {q}\n"
                f"Parent's answer: {reply}",
            ))
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
            channel.emit(Step(
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

            # The sub-agent can call ask_parent(...) to consult us. We're
            # suspended inside execute() while it runs, so we answer with a
            # one-shot completion over OUR live context (`messages`, which
            # reflects compaction) plus the sub-agent's question.
            def clarify(child_question: str) -> str:
                ctx = list(messages) + [Message(
                    "user",
                    "A sub-task you delegated needs clarification before it can "
                    f'proceed:\n\n  "{child_question}"\n\nAnswer concisely from '
                    "what you already know about the overall goal and context. "
                    "If you don't have the answer either, say so briefly so the "
                    "sub-task knows to find it or ask the user.",
                )]
                resp = self.client.complete(ctx, model=self.model)
                meter.record(resp.usage, sent_messages=ctx, response_text=resp.content)
                return resp.content or ""

            return self.run(
                sub_question,
                sub_corpus,
                depth=depth + 1,
                channel=channel,
                meter=meter,
                clarify_handler=clarify,
                enable_planning=enable_planning,
                planning_root_only=planning_root_only,
            )

        names = dict(corpus=corpus, llm=llm, rlm=rlm, answer=answer)
        system_prompt = SYSTEM_PROMPT
        planning_here = enable_planning and (not planning_root_only or depth == 0)
        if planning_here:
            system_prompt += PLANNING_PROMPT

            def on_todo_change(items: list[TodoItem]) -> None:
                channel.emit(Step(depth=depth, index=current["index"], todo=items))

            names["todo"] = TodoList(on_todo_change)
        if self.enable_interaction:
            # ask_user reaches the human (root-only optionally); ask_parent lets
            # a sub-agent consult the agent that delegated it (depth > 0 only).
            if not self.interaction_root_only or depth == 0:
                system_prompt += ASK_USER_PROMPT
                names["ask_user"] = ask_user
            if clarify_handler is not None:
                system_prompt += ASK_PARENT_PROMPT
                names["ask_parent"] = ask_parent
        env.inject(**names)

        # The system prompt is rebuilt every run (it depends on the planning /
        # interaction flags), so prior turns are spliced in AFTER it, never the
        # old system message. Each persisted turn keeps its tool-call/result
        # pairing intact, so the seeded transcript is a valid continuation.
        messages: list[Message] = [Message("system", system_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message("user", question))

        def _save_transcript() -> None:
            # Resumable transcript = everything except the rebuilt system prompt.
            # `messages` may have been rebound by compaction; read it live.
            if transcript_sink is not None:
                transcript_sink.clear()
                transcript_sink.extend(messages[1:])

        for i in range(self.max_steps):
            current["index"] = i
            resp = self.client.complete(messages, tools=[PYTHON_TOOL], model=self.model)
            # Record before appending the reply: `messages` is exactly the
            # context we just sent, so its size is this step's context gauge.
            context_tokens, context_estimated = meter.record(
                resp.usage, sent_messages=messages, response_text=resp.content
            )
            channel.emit(Step(
                depth=depth, index=i,
                usage=meter.snapshot(context_tokens, context_estimated),
            ))
            # Natural-language the model emitted alongside its tool call.
            if resp.content and resp.tool_calls:
                channel.emit(Step(depth=depth, index=i, message=resp.content))
            step_start = len(messages)  # for projecting this step's growth
            messages.append(
                Message("assistant", content=resp.content, tool_calls=resp.tool_calls)
            )

            # No tool call => the model is answering directly.
            if not resp.tool_calls:
                final = resp.content or holder.get("value", "")
                channel.emit(Step(depth=depth, index=i, final=final))
                _save_transcript()
                return final

            for tc in resp.tool_calls:
                code = tc.arguments.get("code", "")
                channel.emit(Step(depth=depth, index=i, code=code))
                result = env.execute(code)
                output = result.stdout
                if result.error:
                    output = (output + "\n" if output else "") + result.error
                output = output.strip() or "(no output)"
                channel.emit(Step(depth=depth, index=i, output=output))
                messages.append(
                    Message("tool", content=output, tool_call_id=tc.id)
                )

            # Splice in any parent-clarification exchanges now that all tool
            # results for this step are in place (keeps the pairing intact).
            if pending_context:
                messages.extend(pending_context)
                pending_context.clear()

            if "value" in holder:
                channel.emit(Step(depth=depth, index=i, final=holder["value"]))
                _save_transcript()
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
                    if summarized:
                        channel.emit(Step(
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
        _save_transcript()
        return holder.get("value") or "Stopped: reached max steps without a final answer."
