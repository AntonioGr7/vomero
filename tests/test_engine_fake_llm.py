"""Exercise the full RLM loop with a scripted fake client — no API key needed.

The fake plays the role of the root model: it emits a sequence of `python` tool
calls (explore -> grep -> answer), proving the engine wires corpus/answer into
the REPL and terminates correctly.
"""

from pathlib import Path

from vomero.context.corpus import Corpus
from vomero.engine import Compactor
from vomero.engine.rlm import RLMEngine, Step
from vomero.llm.base import LLMResponse, Message, ToolCall, Usage
from vomero.usage import UsageMeter

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"


class FakeClient:
    """Returns a pre-scripted sequence of responses, one per `complete` call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


def _py(call_id, code, usage=None):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name="python", arguments={"code": code})],
        usage=usage,
    )


def test_loop_explores_then_answers():
    script = [
        _py("1", "print(corpus.files())"),
        _py("2", "hits = corpus.grep(r'Blocked by'); print(hits)"),
        _py("3", "answer('P-BEACON is blocked by P-ATLAS')"),
    ]
    engine = RLMEngine(FakeClient(script))
    out = engine.run("What blocks P-BEACON?", Corpus(CORPUS))
    assert "P-ATLAS" in out


def test_plain_text_reply_is_final():
    client = FakeClient([LLMResponse(content="42", tool_calls=[])])
    engine = RLMEngine(client)
    out = engine.run("trivial", Corpus(CORPUS))
    assert out == "42"


def test_usage_sums_provider_reported_tokens():
    script = [
        _py("1", "print(corpus.files())", usage=Usage(prompt_tokens=100, completion_tokens=10)),
        _py("2", "answer('done')", usage=Usage(prompt_tokens=180, completion_tokens=5)),
    ]
    engine = RLMEngine(FakeClient(script))
    engine.run("q", Corpus(CORPUS))

    u = engine.last_usage
    assert u is not None
    assert u.calls == 2
    assert u.estimated is False
    assert u.prompt_tokens == 280
    assert u.completion_tokens == 15
    assert u.total_tokens == 295


def test_usage_estimates_when_provider_omits_it():
    script = [_py("1", "print(1)"), _py("2", "answer('done')")]
    engine = RLMEngine(FakeClient(script))
    engine.run("q", Corpus(CORPUS))

    u = engine.last_usage
    assert u is not None
    assert u.calls == 2
    assert u.estimated is True
    assert u.total_tokens > 0


def test_context_snapshot_emitted_per_step():
    script = [
        _py("1", "print(1)", usage=Usage(prompt_tokens=120, completion_tokens=4)),
        _py("2", "answer('done')", usage=Usage(prompt_tokens=260, completion_tokens=6)),
    ]
    snapshots = []

    def on_event(step: Step):
        if step.usage is not None:
            snapshots.append(step.usage)

    engine = RLMEngine(FakeClient(script))
    engine.run("q", Corpus(CORPUS), on_event=on_event)

    # One snapshot per model call: context size is that call's prompt size,
    # cumulative climbs monotonically.
    assert [s.context_tokens for s in snapshots] == [120, 260]
    assert [s.cumulative_tokens for s in snapshots] == [124, 390]


# --- compaction --------------------------------------------------------------

class SummaryClient:
    """A client that returns a fixed summary for the (no-tool) summarizer call."""

    def __init__(self, summary="## Task\nrecap\n## Key findings\nfoo"):
        self.summary = summary
        self.summarize_calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        self.summarize_calls += 1
        return LLMResponse(content=self.summary, tool_calls=[])


def _tool_turn(call_id, code):
    return Message("assistant", tool_calls=[ToolCall(id=call_id, name="python", arguments={"code": code})])


def test_compactor_keeps_preamble_and_recent_tail_verbatim():
    msgs = [
        Message("system", "SYS"),
        Message("user", "the original question"),
        _tool_turn("a", "x = 1"),
        Message("tool", content="ran a", tool_call_id="a"),
        _tool_turn("b", "y = 2"),
        Message("tool", content="ran b", tool_call_id="b"),
        _tool_turn("c", "z = 3"),
        Message("tool", content="ran c", tool_call_id="c"),
    ]
    client = SummaryClient()
    new, n = Compactor(
        context_window=1000, keep_recent_messages=2,
        min_summarize_messages=2, min_reclaim_tokens=0,
    ).compact(
        msgs, client=client, model=None, meter=UsageMeter(),
        state_description="  hits : list (len=3)",
    )

    assert client.summarize_calls == 1
    assert n == 4  # middle = [A1, T1, A2, T2]
    # Preamble verbatim.
    assert new[0].role == "system" and new[1].content == "the original question"
    # One synthetic summary message carrying the header, summary, and REPL state.
    assert new[2].role == "user"
    assert "compacted" in new[2].content.lower()
    assert "## Task" in new[2].content
    assert "hits : list" in new[2].content  # live REPL variables preserved
    # Recent tail kept verbatim and NOT orphaned (starts at an assistant turn).
    assert new[3].role == "assistant"
    assert new[-1].role == "tool" and new[-1].content == "ran c"


def test_compactor_snaps_tail_off_orphan_tool_message():
    # keep_recent would land the tail boundary on a `tool` message; the snap
    # must push it forward to the owning assistant so nothing is orphaned.
    msgs = [
        Message("system", "SYS"),
        Message("user", "Q"),
        _tool_turn("a", "x = 1"),
        Message("tool", content="ran a", tool_call_id="a"),
        _tool_turn("b", "y = 2"),
        Message("tool", content="ran b", tool_call_id="b"),
    ]
    new, n = Compactor(
        context_window=1000, keep_recent_messages=3,
        min_summarize_messages=1, min_reclaim_tokens=0,
    ).compact(msgs, client=SummaryClient(), model=None, meter=UsageMeter())

    assert new[3].role == "assistant"  # tail starts clean, not on a tool result


def test_compactor_skips_when_reclaimable_middle_is_too_small():
    # Plenty of messages (passes the count gate) but the summarizable middle is
    # tiny — the token floor must veto the wasteful summarization call.
    msgs = [
        Message("system", "SYS"),
        Message("user", "Q"),
        _tool_turn("a", "x=1"),
        Message("tool", content="ok", tool_call_id="a"),
        _tool_turn("b", "y=2"),
        Message("tool", content="ok", tool_call_id="b"),
        _tool_turn("c", "z=3"),
        Message("tool", content="ok", tool_call_id="c"),
    ]
    client = SummaryClient()
    new, n = Compactor(
        context_window=1000, keep_recent_messages=2,
        min_summarize_messages=2, min_reclaim_tokens=2048,
    ).compact(msgs, client=client, model=None, meter=UsageMeter())

    assert n == 0 and new is msgs
    assert client.summarize_calls == 0  # no call spent on a no-win compaction


def test_compactor_noop_when_nothing_to_summarize():
    msgs = [Message("system", "SYS"), Message("user", "Q"), _tool_turn("a", "x=1"),
            Message("tool", content="r", tool_call_id="a")]
    client = SummaryClient()
    new, n = Compactor(
        context_window=1000, keep_recent_messages=6, min_summarize_messages=2
    ).compact(msgs, client=client, model=None, meter=UsageMeter())

    assert n == 0 and new is msgs  # untouched, and no summarization call spent
    assert client.summarize_calls == 0


class CompactionLoopClient:
    """Serves scripted python turns for tool calls, a summary for no-tool calls."""

    def __init__(self, tool_responses, summary="## Task\nrecap"):
        self._tool = list(tool_responses)
        self.i = 0
        self.summary = summary
        self.summarize_calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        if not tools:  # the summarizer call passes no tools
            self.summarize_calls += 1
            return LLMResponse(content=self.summary, tool_calls=[])
        resp = self._tool[self.i]
        self.i += 1
        return resp


def test_loop_compacts_when_context_crosses_threshold():
    script = [
        _py("1", "a = corpus.files()", usage=Usage(prompt_tokens=400, completion_tokens=10)),
        _py("2", "b = corpus.grep('x')", usage=Usage(prompt_tokens=900, completion_tokens=10)),
        _py("3", "answer('done')", usage=Usage(prompt_tokens=120, completion_tokens=5)),
    ]
    client = CompactionLoopClient(script)
    compactor = Compactor(
        context_window=1000, ratio=0.8, keep_recent_messages=2,
        min_summarize_messages=2, min_reclaim_tokens=0,
    )
    engine = RLMEngine(client, compactor=compactor)

    events = []
    out = engine.run(
        "q", Corpus(CORPUS),
        on_event=lambda s: events.append(s.compaction) if s.compaction else None,
    )

    assert out == "done"
    assert client.summarize_calls == 1  # exactly one compaction
    assert len(events) == 1 and events[0].summarized_messages == 2
    # Cumulative accounting includes the 3 tool turns + the summarization call.
    assert engine.last_usage.calls == 4


def test_trace_surfaces_llm_subcalls_and_final_text():
    class Client:
        def __init__(self):
            self.tool_calls = 0

        def complete(self, messages, *, tools=None, model=None, temperature=None):
            if not tools:  # the flat llm() distillation sub-call
                return LLMResponse(content="distilled summary", tool_calls=[],
                                   usage=Usage(prompt_tokens=50, completion_tokens=5))
            self.tool_calls += 1
            if self.tool_calls == 1:
                return _py("1", "out = llm('summarize X'); print(out)",
                           usage=Usage(prompt_tokens=100, completion_tokens=8))
            return _py("2", "answer('the final answer')",
                       usage=Usage(prompt_tokens=120, completion_tokens=8))

    steps = []
    out = RLMEngine(Client()).run("q", Corpus(CORPUS), on_event=steps.append)

    assert out == "the final answer"

    llm_calls = [s for s in steps if s.llm_call is not None]
    assert len(llm_calls) == 1
    assert llm_calls[0].llm_call.response == "distilled summary"
    assert llm_calls[0].llm_call.tokens == 55  # 50 prompt + 5 completion, metered

    finals = [s for s in steps if s.final is not None]
    assert finals and finals[0].final == "the final answer"  # text carried, not a marker
