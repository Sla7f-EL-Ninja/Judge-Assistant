"""
test_replanner.py — replanner_node with real LLM, plus hard-guard test.
"""

import json
from datetime import datetime, timezone


def _make_step_dict(step_id, tool="civil_law_rag", query="سؤال قانوني", depends_on=None):
    return {
        "step_id": step_id,
        "tool": tool,
        "query": query,
        "depends_on": depends_on or [],
    }


def _base_state(**overrides):
    state = {
        "original_query": "ما هي وقائع القضية وما الأحكام القانونية المنطبقة؟",
        "case_id": "1234",
        "conversation_history": [],
        "escalation_reason": "تحليل قانوني متعدد الخطوات",
        "plan": [
            _make_step_dict("s1", tool="case_doc_rag", query="وقائع القضية"),
            _make_step_dict("s2", tool="civil_law_rag", query="الأحكام المنطبقة", depends_on=["s1"]),
        ],
        "plan_validation_status": "valid",
        "plan_validation_feedback": "",
        "validator_retry_count": 0,
        "step_results": [
            {
                "step_id": "s1", "tool": "case_doc_rag",
                "query": "وقائع القضية", "status": "failure",
                "response": "", "sources": [],
                "error": "boom", "raw_output": {},
            }
        ],
        "step_failures": {"s1": 3},
        "replan_count": 0,
        "replan_trigger_step_id": "s1",
        "replan_trigger_error": "boom",
        "run_count": 0,
        "synthesis_attempts": 0,
        "final_answer": "",
        "final_sources": [],
        "status": "running",
        "error_message": None,
        "synth_sufficient": True,
        "session_id": "1234::test",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tool_calls_log": [],
        "replan_events": [],
    }
    state.update(overrides)
    return state


def test_replanner_increments_replan_count(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state()
    updates = replanner_node(state)
    assert updates.get("replan_count") == 1


def test_replanner_resets_validator_state(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state(validator_retry_count=2)
    updates = replanner_node(state)
    assert updates.get("validator_retry_count") == 0
    assert updates.get("plan_validation_status") == "pending"
    assert updates.get("plan_validation_feedback") == ""


def test_replanner_clears_trigger(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state()
    updates = replanner_node(state)
    assert updates.get("replan_trigger_step_id") is None
    assert updates.get("replan_trigger_error") is None


def test_replanner_new_plan_present(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state()
    updates = replanner_node(state)
    new_plan = updates.get("plan", [])
    assert len(new_plan) > 0, "Replanner must produce a non-empty plan"


def test_replanner_emits_replan_event(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state()
    updates = replanner_node(state)
    events = updates.get("replan_events", [])
    assert len(events) == 1
    evt = events[0]
    assert evt["trigger_step_id"] == "s1"
    assert evt["replan_index"] == 1


def test_replanner_prunes_failures_and_results(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state()
    updates = replanner_node(state)
    new_step_ids = {s["step_id"] for s in updates.get("plan", [])}
    pruned_failures = updates.get("step_failures", {})
    for sid in pruned_failures:
        assert sid in new_step_ids, f"Orphaned step_id {sid!r} in pruned failures"
    for r in updates.get("step_results", []):
        r_dict = r if isinstance(r, dict) else json.loads(r)
        assert r_dict.get("step_id") in new_step_ids


def test_replanner_failing_step_not_reproduced_verbatim(_preflight):
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state()
    old_step = ("case_doc_rag", "وقائع القضية")
    updates = replanner_node(state)
    new_plan = updates.get("plan", [])
    verbatim_matches = [
        s for s in new_plan
        if (s.get("tool"), s.get("query")) == old_step
    ]
    assert len(verbatim_matches) == 0, (
        "Replanner must not reproduce the exact (tool, query) of the failing step"
    )


def test_replanner_hard_guard_at_cap():
    from chat_reasoner.nodes.replanner import replanner_node
    state = _base_state(replan_count=2)
    updates = replanner_node(state)
    assert updates.get("status") == "failed"
    assert "plan" not in updates or updates.get("plan") is None or updates.get("plan") == []
