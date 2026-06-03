"""The in-memory `Context` source (RLM context-as-a-variable) and the engine
running over it — the canonical RLM surface, exercised with the scripted fake
client (no API key needed)."""

from vomero.context import Context, Corpus, Source
from vomero.engine.rlm import RLMEngine, build_system_prompt
from vomero.llm.base import LLMResponse, ToolCall


def _py(call_id, code):
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name="python", arguments={"code": code})],
    )


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        resp = self._responses[self.calls]
        self.calls += 1
        return resp


# --- Context unit behavior ---------------------------------------------------

def test_both_sources_satisfy_the_seam():
    import tempfile
    assert isinstance(Context("hi"), Source)
    with tempfile.TemporaryDirectory() as d:
        assert isinstance(Corpus(d), Source)


def test_single_string_context_sizing_and_read():
    ctx = Context("line one\nline two\nline three")
    assert ctx.n_docs == 1
    assert len(ctx) == ctx.chars == len("line one\nline two\nline three")
    assert ctx.read() == "line one\nline two\nline three"
    assert ctx.peek(lines=1) == "line one"


def test_multi_doc_context_requires_index_to_read_all():
    ctx = Context(["alpha doc", "beta doc", "gamma doc"])
    assert ctx.n_docs == 3
    assert ctx.read(1) == "beta doc"
    # Reading every doc at once defeats the point — must pick one.
    try:
        ctx.read()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "documents" in str(e)


def test_grep_reports_doc_and_line():
    ctx = Context(["nothing here", "the secret is 42\nmore text"])
    hits = ctx.grep(r"secret is (\d+)")
    assert len(hits) == 1
    assert hits[0].doc == 1 and hits[0].lineno == 1
    assert "42" in hits[0].line


def test_docs_matching_then_subset_scopes_recursion_target():
    ctx = Context(["apples", "bananas", "apple pie"])
    idx = ctx.docs_matching(r"apple")
    assert idx == [0, 2]
    sub = ctx.subset(idx)
    assert isinstance(sub, Context) and sub.n_docs == 2
    assert sub.read(0) == "apples" and sub.read(1) == "apple pie"


def test_subset_char_range():
    ctx = Context("0123456789")
    sub = ctx.subset((2, 5))
    assert sub.read() == "234"


def test_chunk_with_overlap():
    ctx = Context("abcdefghij")  # 10 chars
    assert ctx.chunk(4) == ["abcd", "efgh", "ij"]
    assert ctx.chunk(4, overlap=1) == ["abcd", "defg", "ghij", "j"]


def test_overview_previews_not_dumps():
    ctx = Context("X" * 5000)
    ov = ctx.overview(preview_chars=100)
    assert "5,000 characters" in ov
    assert ov.count("X") == 100  # only the preview, never the whole blob


# --- engine over a Context ---------------------------------------------------

def test_system_prompt_adapts_to_the_source():
    p_ctx = build_system_prompt(Context("hi"))
    assert "context.overview()" in p_ctx
    assert "context.chunk(" in p_ctx
    assert "corpus" not in p_ctx  # no folder assumptions leak in

    # And the folder source still produces its own surface.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p_corp = build_system_prompt(Corpus(d))
    assert "corpus.overview()" in p_corp
    assert "corpus.tree()" in p_corp


def test_engine_explores_an_in_memory_context_then_answers():
    ctx = Context("The capital of Atlantis is Marisol.\nOther filler text.")
    script = [
        _py("1", "print(context.overview())"),
        _py("2", "hits = context.grep(r'capital of Atlantis'); print(hits)"),
        _py("3", "answer('Marisol')"),
    ]
    client = FakeClient(script)
    out = RLMEngine(client).run("What is the capital of Atlantis?", ctx)
    assert out == "Marisol"


def test_recursion_scopes_the_context_via_subset():
    # Root delegates a sub-question scoped to doc 1; the sub-agent must see ONLY
    # that one document (n_docs == 1, and its doc 0 is the originally-doc-1 text).
    ctx = Context(["irrelevant", "the answer token is ZEBRA"])
    script = [
        _py("r1", "sub = rlm('find the answer token', scope=[1]); print(sub)"),
        _py("s1", "answer(f'{context.n_docs}|{context.read(0)}')"),
        _py("r2", "answer(sub)"),  # bubble the sub-agent's answer up
    ]
    out = RLMEngine(FakeClient(script)).run("q", ctx)
    assert out == "1|the answer token is ZEBRA"


# --- output truncation (#4) --------------------------------------------------

def test_truncate_output_keeps_head_and_tail_with_marker():
    from vomero.engine.rlm import truncate_output
    text = "H" * 100 + "M" * 500 + "SENTINEL" + "M" * 500 + "T" * 100
    out = truncate_output(text, limit=200)
    assert len(out) < len(text)
    assert out.startswith("H")          # head kept
    assert out.rstrip().endswith("T")   # tail kept
    assert "truncated" in out           # marker explains the elision
    assert "SENTINEL" not in out        # the dead-center is elided
    # Disabled / under-limit passes through untouched.
    assert truncate_output(text, limit=0) == text
    assert truncate_output("short", limit=200) == "short"


def test_big_print_is_capped_before_entering_the_transcript():
    # A single huge print must NOT land verbatim in history (it would bloat the
    # protected tail forever). The engine caps it at max_output_chars.
    script = [
        _py("1", "print('Z' * 50000)"),
        _py("2", "answer('done')"),
    ]
    captured = {}

    class Recorder(FakeClient):
        def complete(self, messages, *, tools=None, model=None, temperature=None):
            captured["msgs"] = list(messages)  # last-call snapshot
            return super().complete(messages, tools=tools, model=model)

    out = RLMEngine(Recorder(script), max_output_chars=2000).run("q", Context("data"))
    assert out == "done"
    tool_msgs = [m for m in captured["msgs"] if m.role == "tool"]
    assert tool_msgs and len(tool_msgs[0].content) <= 2200  # capped (+ marker slack)
    assert "truncated" in tool_msgs[0].content


# --- global budget across the recursion tree (#5) ----------------------------

def test_budget_stops_the_loop_and_returns_best_effort():
    from vomero.usage import UsageMeter
    from vomero.llm.base import Usage

    # Each call reports 110 tokens; a 100-token budget allows exactly one call,
    # then the top-of-loop guard stops before the second.
    script = [
        _py("1", "x = 1"),
        _py("2", "answer('should not get here')"),
    ]

    def with_usage(resp):
        resp.usage = Usage(prompt_tokens=100, completion_tokens=10)
        return resp

    client = FakeClient([with_usage(r) for r in script])
    meter = UsageMeter(max_total_tokens=100)
    notes = []
    out = RLMEngine(client).run(
        "q", Context("data"), meter=meter,
        on_event=lambda s: notes.append(s.note) if s.note else None,
    )
    # One call happened (110 tok), the budget guard then stopped the loop.
    assert client.calls == 1
    assert any("budget exhausted" in n for n in notes)
    assert "Stopped: budget exhausted" in out


def test_rlm_subcall_skipped_when_budget_exhausted():
    from vomero.usage import UsageMeter
    from vomero.llm.base import Usage

    # Root's first call already blows a tiny budget; the rlm() it then issues
    # must return the budget notice instead of recursing into another run.
    script = [
        _py("r1", "sub = rlm('explore deeply'); print('SUB=' + sub); answer('root: ' + sub)"),
    ]
    r = script[0]
    r.usage = Usage(prompt_tokens=100, completion_tokens=10)
    meter = UsageMeter(max_total_tokens=50)  # already over after call 1
    out = RLMEngine(FakeClient(script)).run("q", Context("data"), meter=meter)
    assert "budget exhausted" in out  # the skipped sub-call's notice bubbled up


# --- unbounded output via answer(variable) (#2) ------------------------------

def test_answer_returns_full_variable_contents_unbounded():
    # The model assembles a large result in a variable and passes it to
    # answer(); the FULL value is returned (not limited by output size, and not
    # subject to the tool-output truncation, which only caps prints).
    big = "PARAGRAPH " * 5000  # ~50k chars, far beyond any output cap
    script = [
        _py("1", "report = 'PARAGRAPH ' * 5000; answer(report)"),
    ]
    out = RLMEngine(FakeClient(script), max_output_chars=2000).run("q", Context("data"))
    assert out == big
    assert len(out) > 40000  # unbounded — the cap on prints did not apply


# --- batched / parallel sub-calls (#3) ---------------------------------------

def test_llm_batched_runs_concurrently_and_preserves_order():
    import threading
    import time
    from vomero.usage import UsageMeter
    from vomero.llm.base import Usage

    # A client whose distillation calls (no tools) sleep, so a serial run would
    # take N*delay but a concurrent one ~1*delay. We assert both ordering and
    # that real overlap happened (peak concurrency > 1).
    class SlowClient:
        def __init__(self):
            self.tool_calls = 0
            self._live = 0
            self._peak = 0
            self._lock = threading.Lock()

        def complete(self, messages, *, tools=None, model=None, temperature=None):
            if tools:  # the root loop's python-tool call
                self.tool_calls += 1
                if self.tool_calls == 1:
                    return _py("1", "outs = llm_batched(['a','b','c','d']); print(outs)")
                return _py("2", "answer('|'.join(outs))")
            # a flat distillation: track concurrency, echo the prompt upper-cased
            with self._lock:
                self._live += 1
                self._peak = max(self._peak, self._live)
            time.sleep(0.05)
            with self._lock:
                self._live -= 1
            prompt = messages[-1].content
            return LLMResponse(content=prompt.upper(),
                               tool_calls=[], usage=Usage(prompt_tokens=5, completion_tokens=1))

    client = SlowClient()
    meter = UsageMeter()
    out = RLMEngine(client, max_parallel_calls=4).run("q", Context("data"), meter=meter)
    assert out == "A|B|C|D"            # order preserved despite concurrency
    assert client._peak > 1            # genuinely ran in parallel
    # Each of the 4 distillations was metered into the shared total.
    assert meter.calls == 2 + 4        # 2 tool-loop calls + 4 distillations


def test_llm_batched_isolates_a_failing_call():
    from vomero.llm.base import Usage

    class FlakyClient:
        def __init__(self):
            self.tool_calls = 0

        def complete(self, messages, *, tools=None, model=None, temperature=None):
            if tools:
                self.tool_calls += 1
                if self.tool_calls == 1:
                    return _py("1", "outs = llm_batched(['ok1','BOOM','ok2']); print(outs)")
                return _py("2", "answer(repr(outs))")
            prompt = messages[-1].content
            if "BOOM" in prompt:
                raise RuntimeError("kaboom")
            return LLMResponse(content="ok", tool_calls=[], usage=Usage(prompt_tokens=5, completion_tokens=1))

    out = RLMEngine(FlakyClient()).run("q", Context("data"))
    # The batch survived: good results returned, the failure became an error string.
    assert "ok" in out and "llm error" in out
