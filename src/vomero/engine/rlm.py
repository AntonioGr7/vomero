"""RLMEngine — the recursive REPL loop.

Flow (one `run`):
  1. Spin up an ExecutionEnvironment and inject the navigation surface: the data
     handle (a `Source` — `corpus` or `context`), `llm(...)`, `rlm(...)`,
     `answer(...)`.
  2. Give the root model a system prompt describing that surface, plus the
     user's question. It can ONLY act via the `python` tool.
  3. Each step: model -> python(code) -> we exec -> feed stdout/traceback back.
  4. Stop when the model calls `answer(...)` from the REPL, or replies with
     plain text (no tool call). Either becomes the final answer.

The data is a `Source` (context/source.py): either a `Corpus` (a folder) or a
`Context` (an in-memory blob held as a REPL variable — the canonical RLM case).
The engine is agnostic between them.

The recursion: `llm()` is a flat sub-call (cheap distillation of a chunk);
`rlm()` re-enters this engine on a (optionally scoped) source at depth+1, so a
sub-question gets the same full power. Depth is capped to keep it finite.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from ..context.source import AccessEvent, Source
from ..execution import ExecutionEnvironment, InProcessEnvironment
from ..llm.base import LLMClient, Message, ToolSpec
from ..channel import Channel, CallbackChannel
from ..usage import UsageMeter, UsageSnapshot, estimate_message_tokens
from .compaction import Compactor, CompactionEvent
from .todo import TodoItem, TodoList

# The single tool the root model gets. Keeping it to one tool (run Python)
# maps cleanly onto every provider's function-calling and matches the RLM idea:
# the model's lever on the world is code.
# Returned from llm()/rlm() when the global budget is spent, so the model's code
# keeps running (and can call answer()) instead of erroring mid-step.
_BUDGET_NOTICE = (
    "[budget exhausted: this sub-call was skipped to keep the run within its "
    "token/call limit. Answer now with what you already have.]"
)

# Emitted as a final user turn when the run hits its step limit before
# answering: ask the model to organize the partial findings rather than
# leaving the user with a bare "stopped" notice.
_PARTIAL_SYNTHESIS_PROMPT = (
    "You have reached the step limit and cannot do any more work or tool "
    "calls. Using ONLY what you have already gathered above, write a brief, "
    "honest reply that:\n"
    "1. States up front that the step limit was reached, so this answer is "
    "partial and may be incomplete.\n"
    "2. Organizes and presents the relevant findings you did gather so far, "
    "if any are useful.\n"
    "3. Notes briefly what is still unresolved.\n"
    "If nothing useful was found, say that plainly instead of guessing. Be "
    "concise and do not fabricate an answer."
)

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

# The system prompt is assembled per run from three parts: a generic head, the
# source-specific surface (`source.guide()` — corpus vs. in-memory context), and
# the shared helpers/strategy. Splitting it this way keeps the prompt accurate
# whichever `Source` is mounted, with no corpus/context assumptions baked in.
SYSTEM_PROMPT_HEAD = """You are Vomero, an assistant that answers questions about \
a body of data WITHOUT loading it into your own context. The data is too large \
and too important to paste into this conversation. Instead you operate a Python \
REPL via the `python` tool and reason *programmatically* over the data.

Your REPL already has these names available:

"""

# `{name}` is the data handle's REPL name; `{start}` its first-call hint.
HELPERS_AND_STRATEGY = """
  llm(text, system=None) -> str
                A single, fresh model call with NO memory and NO tools. Use it to
                distill a chunk you have already read into a variable, e.g.
                summarize, extract, or answer a narrow question about that text.
                The chunk you pass is the ONLY thing that sub-call sees.

  llm_batched(texts, system=None) -> list[str]
                Like llm(), but runs MANY distillations concurrently and returns
                the results in input order. This is the partition+map workhorse:
                split the data into chunks (e.g. {name}.chunk(...)), then distill
                them all in parallel — far faster than calling llm() in a loop.

  rlm(question, scope=None) -> str
                Spawn a fresh Vomero SUB-AGENT on the (optionally scoped) data.
                It gets the same full power you have — its own REPL, its own
                exploration — and returns just its sub-answer, so the raw work
                stays out of YOUR context. This is how you decompose: delegate a
                self-contained sub-question that itself needs searching/reading,
                rather than doing every hop inline and bloating your own
                transcript. `scope` narrows what the sub-call sees, using the
                same selector `{name}.subset(...)` takes. Returns the sub-answer.

  answer(value) Record your FINAL answer and finish. `value` may be a string OR
                any REPL variable — pass a variable you built up (e.g.
                answer(report)) and its FULL contents become the answer. Because
                you reference it by name, the answer can be arbitrarily long: it
                is NOT limited by your own output size. Assemble long outputs
                programmatically in a variable, then answer(that_variable).

Strategy:
  - Start by calling {start} (print it) to see what you have.
  - Locate what's relevant by searching/slicing before reading wholesale.
  - NEVER print a large amount of raw text into the transcript. Hold it in a
    variable and pass it to llm()/rlm() to distill — keep raw text out of your
    own context.
  - DECOMPOSE before diving in. For a multi-hop question, name the hops first
    (e.g. "who is the Green performer?" -> "who is their spouse?"), then resolve
    them in order: find A, use A to find B, aggregate. Match each hop by MEANING,
    not surface words — a "spouse" hop is also satisfied by "partner", "married
    to", "wife"; the bridge entity is rarely phrased the way the question is, so
    a literal keyword match often lands on a distractor.
  - DISAMBIGUATE the anchor. When the linking term is ambiguous (e.g. many
    "Green"s — a person, an album, a party, a place), enumerate the candidates
    first and keep only the one that satisfies EVERY constraint in the question
    (here: is a *performer* AND tied to "Green"). Don't latch onto the first
    lexical hit; the data is full of near-misses placed to mislead.
  - Delegate a hop with rlm(hop, scope=...) ONLY when it needs its own heavy
    searching/reading — that keeps its raw work out of your context and lets it
    recurse further. When the relevant data is small enough to hold and reason
    over directly, do the hops inline: spawning a sub-agent over a handful of
    short documents costs more than it saves and can sever the cross-document
    link you need.
  - Verify the WHOLE chain before answering: re-read the supporting text for
    each hop, confirm the bridge entity is the same one throughout, and check
    the final answer actually satisfies the original question. Cite the sources
    you relied on.
  - When confident, call answer(...) — a string, or a variable holding your
    full result — citing what you relied on.

Keep each code block small and purposeful. Print only what you need to see."""


def build_system_prompt(source, extra: str | None = None) -> str:
    """Assemble the root system prompt for a given `Source` (corpus/context).

    `extra` is an optional instruction block appended verbatim — the tunable
    surface the prompt optimizer (eval/optimize.py) searches over."""
    prompt = (
        SYSTEM_PROMPT_HEAD
        + source.guide()
        + HELPERS_AND_STRATEGY.format(name=source.repl_name, start=source.start_hint())
    )
    if extra and extra.strip():
        prompt += "\n\nAdditional instructions:\n" + extra.strip()
    return prompt


def truncate_output(text: str, limit: int) -> str:
    """Cap a tool result before it enters the transcript.

    A single oversized `print` would otherwise land in the protected recent
    tail — which compaction never summarizes — and permanently bloat context.
    We keep the head and tail (the most load-bearing parts of a traceback or a
    listing) and elide the middle with a marker that nudges the model to slice/
    grep or distill instead of printing wholesale. `limit <= 0` disables it.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    dropped = len(text) - head - tail
    marker = (
        f"\n\n…[output truncated: {dropped:,} of {len(text):,} chars elided. "
        "Don't print large values wholesale — slice/grep them, or hold them in a "
        "variable and pass to llm()/rlm() to distill.]…\n\n"
    )
    return text[:head] + marker + text[-tail:]


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
    # An engine-level notice (e.g. the global budget was exhausted).
    note: str | None = None


@dataclass
class RunResult:
    """Structured outcome of a root `run(..., return_trajectory=True)`.

    The default `run` returns just the answer string (back-compat). Opting in
    returns this instead: the answer plus the per-step `trajectory` (the root
    agent's Steps — code, output, messages, sub-calls) and the run's cost. The
    trajectory is what the prompt optimizer and downstream tooling inspect."""

    answer: str
    trajectory: list[Step]
    tokens: int
    calls: int
    # What the model retrieved from the source over the whole run (root + any
    # recursive sub-calls): grep hits carry their doc+line+text, reads/peeks
    # just the doc that was touched. Structured provenance for grounding the
    # answer back to its source, so consumers don't parse it out of stdout.
    provenance: list[AccessEvent] = field(default_factory=list)


class _TrajectoryRecorder:
    """A Channel wrapper that captures the root agent's Steps while delegating
    every event to the real channel. Only depth-0 steps are kept (sub-agent
    steps belong to their own recursive runs)."""

    def __init__(self, inner: Channel):
        self.inner = inner
        self.steps: list[Step] = []

    def emit(self, step: Step) -> None:
        if step.depth == 0:
            self.steps.append(step)
        self.inner.emit(step)

    def ask_user(self, question: str) -> str:
        return self.inner.ask_user(question)


class RLMEngine:
    def __init__(
        self,
        client: LLMClient,
        *,
        env_factory: Callable[[], ExecutionEnvironment] = InProcessEnvironment,
        model: str | None = None,
        max_steps: int = 24,
        max_depth: int = 3,
        max_output_chars: int = 10_000,
        max_parallel_calls: int = 8,
        extra_instructions: str | None = None,
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
        # Hard cap on a single tool result's size (chars) before it enters the
        # transcript. Guards the protected recent-tail against one giant print.
        self.max_output_chars = max_output_chars
        # Max concurrent flat sub-calls in one llm_batched(...) — the fan-out
        # width for the partition+map pattern.
        self.max_parallel_calls = max_parallel_calls
        # Optional instruction block appended to the system prompt — the surface
        # the prompt optimizer tunes. Plain config (not per-run state).
        self.extra_instructions = extra_instructions
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
        source: Source,
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
        # the caller owns the store. See server.py.
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
        # Opt-in: return a RunResult (answer + per-step trajectory + cost) instead
        # of the bare answer string. Root-level only; recursion always returns a
        # string (rlm() consumes sub-answers as strings). Default off = back-compat.
        return_trajectory: bool = False,
    ) -> str | RunResult:
        # The frontend is a single Channel. Pass your own UsageMeter to read
        # token usage after the run — the engine keeps no per-run state, so the
        # same engine instance is safe to use from concurrent threads.
        if channel is None:
            channel = CallbackChannel(on_event=on_event, ask_handler=ask_handler)
        # Capture the root agent's steps when a trajectory was requested.
        recorder = _TrajectoryRecorder(channel) if return_trajectory else None
        if recorder is not None:
            channel = recorder
            # Capture structured retrieval provenance alongside the trajectory.
            # subset() shares this log, so recursive sub-calls record into it too.
            if hasattr(source, "enable_access_log"):
                source.enable_access_log()
        meter = meter or UsageMeter()

        def _finish(ans: str):
            """Wrap the answer as a RunResult when a trajectory was requested."""
            if recorder is None:
                return ans
            provenance = source.access_log if hasattr(source, "access_log") else []
            return RunResult(answer=ans, trajectory=recorder.steps,
                            tokens=meter.total_tokens, calls=meter.calls,
                            provenance=provenance)
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

        def _distill_messages(text: str, system: str | None) -> list[Message]:
            msgs = []
            if system:
                msgs.append(Message("system", system))
            msgs.append(Message("user", text))
            return msgs

        def _record_distillation(text: str, msgs: list[Message], resp) -> str:
            # Meter + channel are NOT thread-safe, so recording/emitting always
            # happens on the main thread (after any concurrent calls return).
            before = meter.total_tokens
            meter.record(resp.usage, sent_messages=msgs, response_text=resp.content)
            out = resp.content or ""
            channel.emit(Step(
                depth=depth, index=current["index"],
                llm_call=LLMCall(prompt=text, response=out,
                                 tokens=meter.total_tokens - before),
            ))
            return out

        def llm(text: str, system: str | None = None) -> str:
            if meter.exhausted:
                return _BUDGET_NOTICE
            msgs = _distill_messages(text, system)
            resp = self.client.complete(msgs, model=self.model)
            # A flat sub-call still spends tokens; fold it into the shared total.
            return _record_distillation(text, msgs, resp)

        def llm_batched(texts, system: str | None = None) -> list[str]:
            # Partition + map: run many flat distillations CONCURRENTLY and return
            # their results in input order. The dominant RLM pattern (chunk the
            # data, distill every chunk) is embarrassingly parallel — doing it
            # serially is the latency cost the reference RLM work calls out.
            texts = list(texts)
            if not texts:
                return []
            if meter.exhausted:
                return [_BUDGET_NOTICE] * len(texts)

            def _call(t: str):
                try:
                    msgs = _distill_messages(t, system)
                    return msgs, self.client.complete(msgs, model=self.model), None
                except Exception as e:  # one failure must not sink the batch
                    return _distill_messages(t, system), None, e

            workers = max(1, min(len(texts), self.max_parallel_calls))
            slots: list = [None] * len(texts)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_call, t): i for i, t in enumerate(texts)}
                for fut in as_completed(futs):
                    slots[futs[fut]] = fut.result()

            # Record + emit sequentially on the main thread, in input order.
            outs: list[str] = []
            for t, (msgs, resp, err) in zip(texts, slots):
                if err is not None:
                    out = f"[llm error: {err}]"
                    channel.emit(Step(
                        depth=depth, index=current["index"],
                        llm_call=LLMCall(prompt=t, response=out, tokens=0),
                    ))
                else:
                    out = _record_distillation(t, msgs, resp)
                outs.append(out)
            return outs

        def rlm(sub_question: str, scope=None, paths=None) -> str:
            if meter.exhausted:
                return _BUDGET_NOTICE
            if depth + 1 > self.max_depth:
                # Out of recursion budget: degrade to a flat call so we still
                # make progress instead of failing.
                return llm(sub_question)
            # `paths` is the legacy corpus-only alias for `scope`; the selector's
            # meaning is the source's (file paths, doc indices, a char range).
            sel = scope if scope is not None else paths
            sub_source = source.subset(sel) if sel is not None else source

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
                sub_source,
                depth=depth + 1,
                channel=channel,
                meter=meter,
                clarify_handler=clarify,
                enable_planning=enable_planning,
                planning_root_only=planning_root_only,
            )

        names = {source.repl_name: source, "llm": llm, "llm_batched": llm_batched,
                 "rlm": rlm, "answer": answer}
        system_prompt = build_system_prompt(source, extra=self.extra_instructions)
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
            # Global budget guard (shared meter => spans the whole run tree).
            # Stop before another model call rather than running away on cost.
            if meter.exhausted:
                channel.emit(Step(depth=depth, index=i,
                                  note="budget exhausted — stopping before further model calls"))
                break
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
                return _finish(final)

            for tc in resp.tool_calls:
                code = tc.arguments.get("code", "")
                channel.emit(Step(depth=depth, index=i, code=code))
                result = env.execute(code)
                output = result.stdout
                if result.error:
                    output = (output + "\n" if output else "") + result.error
                output = output.strip() or "(no output)"
                # Cap before it enters context (and before we emit it, so the
                # trace shows exactly what the model saw).
                output = truncate_output(output, self.max_output_chars)
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
                return _finish(holder["value"])

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

        # Ran out of steps (or budget) — return whatever we have.
        _save_transcript()
        if "value" in holder:
            return _finish(holder["value"])
        reason = "budget exhausted" if meter.exhausted else "reached max steps"

        # No final answer was produced. Rather than returning a bare stop
        # notice, make one best-effort synthesis pass: organize whatever
        # partial findings the run gathered and hand them back with a clear
        # caveat that the run was cut short. Skipped when the budget is
        # exhausted — another model call would run past the cost guard we
        # just stopped at, so there we fall back to the plain notice.
        if not meter.exhausted:
            synth = list(messages) + [Message("user", _PARTIAL_SYNTHESIS_PROMPT)]
            try:
                resp = self.client.complete(synth, model=self.model)
            except Exception:
                resp = None
            if resp is not None:
                meter.record(resp.usage, sent_messages=synth,
                             response_text=resp.content)
                partial = (resp.content or "").strip()
                if partial:
                    channel.emit(Step(depth=depth, index=current["index"],
                                      final=partial))
                    return _finish(partial)
        return _finish(f"Stopped: {reason} without a final answer.")
