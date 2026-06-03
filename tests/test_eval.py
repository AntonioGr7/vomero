"""The eval harness — metrics, both runners, and aggregation — exercised
offline with scripted fake clients (no API key, no network)."""

import json

from vomero.context import Context
from vomero.engine.rlm import RLMEngine
from vomero.eval import (EvalItem, RLMRunner, StuffBaselineRunner, compare,
                         evaluate, load_jsonl)
from vomero.eval import metrics
from vomero.llm.base import LLMResponse, ToolCall, Usage


def _py(call_id, code):
    return LLMResponse(content=None, usage=Usage(prompt_tokens=50, completion_tokens=5),
                       tool_calls=[ToolCall(id=call_id, name="python", arguments={"code": code})])


# --- metrics -----------------------------------------------------------------

def test_metric_normalization_and_scores():
    assert metrics.exact_match("The Marisol team.", "marisol team") == 1.0
    assert metrics.exact_match("nope", "marisol team") == 0.0
    assert metrics.token_f1("the red fox", "a red fox") == 1.0  # articles dropped
    assert 0.0 < metrics.token_f1("red fox runs", "red fox") < 1.0
    assert metrics.contains_gold("I believe the answer is Marisol, the capital.", "Marisol") == 1.0


def test_llm_judge_reads_yes_no():
    class JudgeClient:
        def __init__(self, verdict):
            self.verdict = verdict

        def complete(self, messages, *, tools=None, model=None, temperature=None):
            return LLMResponse(content=self.verdict, tool_calls=[])

    assert metrics.llm_judge(JudgeClient("YES"), "q", "pred", "gold") == 1.0
    assert metrics.llm_judge(JudgeClient("no, different"), "q", "pred", "gold") == 0.0


# --- runners -----------------------------------------------------------------

class _ScriptClient:
    """Tool calls for the RLM loop; a plain reply for the baseline's one-shot."""

    def __init__(self, script, oneshot="Marisol"):
        self.script = list(script)
        self.i = 0
        self.oneshot = oneshot
        self.last_user = None

    def complete(self, messages, *, tools=None, model=None, temperature=None):
        if not tools:  # baseline / judge: single completion
            self.last_user = messages[-1].content
            return LLMResponse(content=self.oneshot,
                               usage=Usage(prompt_tokens=999, completion_tokens=3), tool_calls=[])
        resp = self.script[self.i]
        self.i += 1
        return resp


def test_rlm_runner_reports_answer_and_cost():
    client = _ScriptClient([_py("1", "answer('Marisol')")])
    runner = RLMRunner(RLMEngine(client))
    out = runner.answer("capital?", Context("The capital is Marisol."))
    assert out.answer == "Marisol"
    assert out.calls == 1 and out.tokens == 55  # metered through the engine
    assert out.seconds >= 0.0


def test_baseline_runner_stuffs_whole_context_and_flags_truncation():
    client = _ScriptClient([], oneshot="Marisol")
    runner = StuffBaselineRunner(client, max_chars=20)
    ctx = Context("The capital is Marisol. " * 10)  # > 20 chars => truncated
    out = runner.answer("capital?", ctx)
    assert out.answer == "Marisol"
    assert out.truncated is True               # didn't fit — the baseline's failure mode
    assert "The capital" in client.last_user   # the context was pasted into the prompt
    assert "Question: capital?" in client.last_user


# --- harness aggregation -----------------------------------------------------

def test_evaluate_aggregates_correctness_and_cost():
    items = [
        EvalItem("capital?", "Marisol", Context("The capital is Marisol.")),
        EvalItem("capital?", "Atlantis", Context("The capital is Marisol.")),  # wrong gold
    ]
    runner = StuffBaselineRunner(_ScriptClient([], oneshot="Marisol"))
    rep = evaluate(items, runner)
    assert rep.n == 2
    assert rep.exact == 0.5                 # 1 of 2 exact-matches "Marisol"
    assert rep.mean_tokens == 1002          # 999 + 3 per item
    assert "EM 0.500" in rep.summary()


def test_compare_runs_both_runners_over_same_items():
    items = [EvalItem("capital?", "Marisol", Context("The capital is Marisol."))]
    rlm = RLMRunner(RLMEngine(_ScriptClient([_py("1", "answer('Marisol')")])))
    base = StuffBaselineRunner(_ScriptClient([], oneshot="Marisol"))
    reports = compare(items, [rlm, base])
    assert [r.runner for r in reports] == ["rlm", "baseline"]
    assert all(r.exact == 1.0 for r in reports)


def test_load_jsonl_with_inline_context(tmp_path):
    p = tmp_path / "qa.jsonl"
    p.write_text(
        json.dumps({"question": "capital?", "answer": "Marisol",
                    "context": "The capital is Marisol."}) + "\n"
        + json.dumps({"query": "who?", "gold": "Bob", "context": ["Bob did it."]}) + "\n",
        encoding="utf-8",
    )
    items = load_jsonl(p)
    assert len(items) == 2
    assert items[0].question == "capital?" and items[0].answer == "Marisol"
    assert isinstance(items[0].source, Context)
    assert items[1].question == "who?" and items[1].answer == "Bob"  # alias keys


# --- structured trajectory return (#6) ---------------------------------------

def test_return_trajectory_gives_runresult_with_steps():
    from vomero.engine.rlm import RunResult
    script = [
        _py("1", "print(context.overview())"),
        _py("2", "answer('Marisol')"),
    ]
    res = RLMEngine(_ScriptClient(script)).run(
        "capital?", Context("The capital is Marisol."), return_trajectory=True
    )
    assert isinstance(res, RunResult)
    assert res.answer == "Marisol"
    assert res.calls == 2 and res.tokens > 0
    # The trajectory captured the root agent's code, output, and final.
    codes = [s.code for s in res.trajectory if s.code]
    assert any("overview" in c for c in codes)
    assert any(s.final == "Marisol" for s in res.trajectory)


def test_default_run_still_returns_plain_string():
    out = RLMEngine(_ScriptClient([_py("1", "answer('x')")])).run("q", Context("d"))
    assert out == "x" and isinstance(out, str)  # back-compat preserved


# --- prompt optimizer (#6) ---------------------------------------------------

def test_optimizer_selects_the_candidate_that_improves_the_metric():
    from vomero.eval import optimize

    # A client whose answer depends on whether the system prompt carries a magic
    # token — so the candidate instruction block that includes it scores higher.
    class PromptSensitiveClient:
        def complete(self, messages, *, tools=None, model=None, temperature=None):
            sys_prompt = messages[0].content
            if "SAY_MARISOL" in sys_prompt:
                return _py("1", "answer('Marisol')")
            return _py("1", "answer('Atlantis')")

    items = [EvalItem("capital?", "Marisol", Context("The capital is Marisol."))]
    engine = RLMEngine(PromptSensitiveClient())
    candidates = [None, "Always answer the capital. SAY_MARISOL when relevant."]

    result = optimize(engine, items, candidates, metric="exact")
    assert result.best_score == 1.0
    assert "SAY_MARISOL" in result.best_instructions      # the winning block
    # And the engine is left configured with the winner (keep_best default).
    assert engine.extra_instructions == result.best_instructions
    # Baseline scored 0 (answered "Atlantis"); ranking puts the winner first.
    assert result.scored[0][1] == 1.0 and result.scored[-1][1] == 0.0


def test_propose_instructions_splits_on_separators():
    from vomero.eval import propose_instructions

    class Proposer:
        def complete(self, messages, *, tools=None, model=None, temperature=None):
            return LLMResponse(content="Block one.\n---\nBlock two.\n---\nBlock three.",
                               tool_calls=[])

    blocks = propose_instructions(Proposer(), n=2)
    assert blocks == ["Block one.", "Block two."]  # capped at n, separator-split
