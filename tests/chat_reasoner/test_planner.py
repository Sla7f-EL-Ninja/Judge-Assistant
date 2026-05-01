"""
test_planner.py — planner_node with real LLM.
"""

from chat_reasoner.state import ALLOWED_TOOLS


def _base_state(**overrides):
    state = {
        "original_query": "قارن وقائع هذه القضية بأحكام القانون المدني واكتب ملخصاً",
        "case_id": "1234",
        "conversation_history": [],
        "escalation_reason": "طلب مقارنة قانونية",
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
        "session_id": "1234::test",
        "started_at": "2024-01-01T00:00:00+00:00",
        "tool_calls_log": [],
        "replan_events": [],
    }
    state.update(overrides)
    return state


def test_planner_produces_valid_pending_plan(_preflight):
    from chat_reasoner.nodes.planner import planner_node
    state = _base_state()
    updates = planner_node(state)

    assert updates.get("plan_validation_status") == "pending"
    plan = updates.get("plan", [])
    assert len(plan) > 0, "Planner must produce at least one step"

    for step in plan:
        assert step.get("tool") in ALLOWED_TOOLS, (
            f"Step {step.get('step_id')} uses unknown tool {step.get('tool')!r}"
        )
        assert str(step.get("query", "")).strip(), (
            f"Step {step.get('step_id')} has empty query"
        )


def test_planner_has_parallel_root_step(_preflight):
    from chat_reasoner.nodes.planner import planner_node
    state = _base_state()
    updates = planner_node(state)
    plan = updates.get("plan", [])

    root_steps = [s for s in plan if not s.get("depends_on")]
    assert len(root_steps) >= 1, "Plan must have at least one root (parallel) step"


def test_planner_with_validator_feedback(_preflight):
    from chat_reasoner.nodes.planner import planner_node
    state = _base_state(
        validator_retry_count=1,
        plan_validation_feedback="لم يتم استخدام fetch_summary_report — يجب إضافتها",
    )
    updates = planner_node(state)

    plan = updates.get("plan", [])
    assert len(plan) > 0
    # Planner should incorporate the feedback and include fetch_summary_report
    tools_used = {s.get("tool") for s in plan}
    assert "fetch_summary_report" in tools_used, (
        "Planner with feedback requesting fetch_summary_report must include it"
    )
