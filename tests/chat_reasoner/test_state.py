"""
test_state.py — ALLOWED_TOOLS, reducer, all Pydantic schemas, ChatReasonerState shape.
"""


def test_allowed_tools_exact():
    from chat_reasoner.state import ALLOWED_TOOLS
    assert ALLOWED_TOOLS == {"case_doc_rag", "civil_law_rag", "fetch_summary_report"}


def test_allowed_tools_is_frozenset():
    from chat_reasoner.state import ALLOWED_TOOLS
    assert isinstance(ALLOWED_TOOLS, frozenset)


def test_merge_step_failures_max_per_key():
    from chat_reasoner.state import _merge_step_failures
    result = _merge_step_failures({"a": 1, "b": 3}, {"a": 2, "c": 1})
    assert result == {"a": 2, "b": 3, "c": 1}


def test_merge_step_failures_no_overlap():
    from chat_reasoner.state import _merge_step_failures
    assert _merge_step_failures({"x": 1}, {"y": 2}) == {"x": 1, "y": 2}


def test_merge_step_failures_empty():
    from chat_reasoner.state import _merge_step_failures
    assert _merge_step_failures({}, {"a": 1}) == {"a": 1}
    assert _merge_step_failures({"a": 1}, {}) == {"a": 1}


def test_merge_step_failures_keeps_higher():
    from chat_reasoner.state import _merge_step_failures
    assert _merge_step_failures({"s1": 3}, {"s1": 1}) == {"s1": 3}


def test_plan_step_required_fields():
    from chat_reasoner.state import PlanStep
    s = PlanStep(step_id="s1", tool="civil_law_rag", query="سؤال")
    assert s.step_id == "s1"
    assert s.tool == "civil_law_rag"
    assert s.query == "سؤال"
    assert s.depends_on == []


def test_plan_step_with_depends_on():
    from chat_reasoner.state import PlanStep
    s = PlanStep(step_id="s2", tool="case_doc_rag", query="سؤال", depends_on=["s1"])
    assert s.depends_on == ["s1"]


def test_plan_schema():
    from chat_reasoner.state import Plan, PlanStep
    p = Plan(
        steps=[PlanStep(step_id="s1", tool="civil_law_rag", query="q")],
        parallel_groups_note="s1 يعمل بشكل مستقل",
    )
    assert len(p.steps) == 1
    assert p.parallel_groups_note


def test_plan_validation_result_valid():
    from chat_reasoner.state import PlanValidationResult
    r = PlanValidationResult(valid=True, failed_checks=[], feedback="")
    assert r.valid is True
    assert r.failed_checks == []


def test_plan_validation_result_invalid():
    from chat_reasoner.state import PlanValidationResult
    r = PlanValidationResult(valid=False, failed_checks=["valid_tool_names"], feedback="خطأ")
    assert r.valid is False
    assert "valid_tool_names" in r.failed_checks


def test_step_result_required():
    from chat_reasoner.state import StepResult
    r = StepResult(
        step_id="s1", tool="civil_law_rag", query="q",
        status="success", response="الجواب",
    )
    assert r.status == "success"
    assert r.sources == []
    assert r.error is None
    assert r.raw_output == {}


def test_step_result_failure():
    from chat_reasoner.state import StepResult
    r = StepResult(
        step_id="s1", tool="case_doc_rag", query="q",
        status="failure", response="", error="boom",
    )
    assert r.error == "boom"


def test_synthesizer_decision_sufficient():
    from chat_reasoner.state import SynthesizerDecision
    d = SynthesizerDecision(answer="الجواب", sufficient=True)
    assert d.sufficient is True
    assert d.insufficiency_reason is None


def test_synthesizer_decision_insufficient():
    from chat_reasoner.state import SynthesizerDecision
    d = SynthesizerDecision(
        answer="غير كافٍ", sufficient=False,
        insufficiency_reason="لا توجد وثائق كافية",
    )
    assert d.sufficient is False
    assert d.insufficiency_reason


def test_chat_reasoner_state_shape():
    """Minimal ChatReasonerState dict includes all required keys."""
    from chat_reasoner.state import ChatReasonerState
    state: ChatReasonerState = {
        "original_query": "سؤال",
        "case_id": "1234",
        "conversation_history": [],
        "escalation_reason": "test",
        "plan": [],
        "plan_validation_status": "pending",
        "plan_validation_feedback": "",
        "validator_retry_count": 0,
        "step_results": [],
        "step_failures": {},
        "replan_count": 0,
        "replan_trigger_step_id": None,
        "replan_trigger_error": None,
        "run_count": 0,
        "synthesis_attempts": 0,
        "final_answer": "",
        "final_sources": [],
        "status": "running",
        "error_message": None,
        "synth_sufficient": True,
        "session_id": "1234::2024-01-01T00:00:00+00:00",
        "started_at": "2024-01-01T00:00:00+00:00",
        "tool_calls_log": [],
        "replan_events": [],
    }
    # All expected keys are present
    expected_keys = {
        "original_query", "case_id", "conversation_history", "escalation_reason",
        "plan", "plan_validation_status", "plan_validation_feedback", "validator_retry_count",
        "step_results", "step_failures", "replan_count", "replan_trigger_step_id",
        "replan_trigger_error", "run_count", "synthesis_attempts", "final_answer",
        "final_sources", "status", "error_message", "synth_sufficient",
        "session_id", "started_at", "tool_calls_log", "replan_events",
    }
    assert expected_keys.issubset(set(state.keys()))
