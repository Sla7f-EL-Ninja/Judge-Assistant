"""
test_plan_validator.py — all 6 checks + router branches, no LLM.
"""

import pytest


def _make_step(step_id, tool="civil_law_rag", query="سؤال قانوني", depends_on=None):
    return {
        "step_id": step_id,
        "tool": tool,
        "query": query,
        "depends_on": depends_on or [],
    }


def _make_state(plan, validator_retry_count=0, validation_status="invalid"):
    return {
        "plan": plan,
        "plan_validation_status": validation_status,
        "plan_validation_feedback": "",
        "validator_retry_count": validator_retry_count,
    }


# ---------------------------------------------------------------------------
# _validate_plan checks
# ---------------------------------------------------------------------------

def test_valid_plan():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    plan = [
        _make_step("s1"),
        _make_step("s2", tool="case_doc_rag", depends_on=["s1"]),
    ]
    result = _validate_plan(plan)
    assert result.valid is True
    assert result.failed_checks == []


def test_empty_plan_fails_at_least_one_step():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    result = _validate_plan([])
    assert result.valid is False
    assert "at_least_one_step" in result.failed_checks


def test_duplicate_step_ids():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    plan = [_make_step("s1"), _make_step("s1", tool="case_doc_rag")]
    result = _validate_plan(plan)
    assert "unique_step_ids" in result.failed_checks


def test_invalid_tool_name():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    plan = [_make_step("s1", tool="nonexistent_tool")]
    result = _validate_plan(plan)
    assert "valid_tool_names" in result.failed_checks


def test_empty_query():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    plan = [_make_step("s1", query="   ")]
    result = _validate_plan(plan)
    assert "non_empty_queries" in result.failed_checks


def test_unresolvable_depends_on():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    plan = [_make_step("s1", depends_on=["ghost"])]
    result = _validate_plan(plan)
    assert "depends_on_resolvable" in result.failed_checks


def test_circular_dependency():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    # A→B, B→A — only checked when no other failure exists
    plan = [
        _make_step("A", depends_on=["B"]),
        _make_step("B", depends_on=["A"]),
    ]
    result = _validate_plan(plan)
    assert "acyclic" in result.failed_checks


def test_multiple_simultaneous_failures():
    from chat_reasoner.nodes.plan_validator import _validate_plan
    plan = [
        _make_step("s1", tool="bad_tool", query="   "),
        _make_step("s1", tool="another_bad", query="ok"),  # dup + bad tool
    ]
    result = _validate_plan(plan)
    assert result.valid is False
    # dup IDs, bad tools, empty query all present
    assert "unique_step_ids" in result.failed_checks
    assert "valid_tool_names" in result.failed_checks
    assert "non_empty_queries" in result.failed_checks


# ---------------------------------------------------------------------------
# plan_validator_node
# ---------------------------------------------------------------------------

def test_validator_node_valid_plan():
    from chat_reasoner.nodes.plan_validator import plan_validator_node
    plan = [_make_step("s1")]
    state = _make_state(plan)
    result = plan_validator_node(state)
    assert result["plan_validation_status"] == "valid"
    assert result["plan_validation_feedback"] == ""


def test_validator_node_invalid_increments_retry():
    from chat_reasoner.nodes.plan_validator import plan_validator_node
    state = _make_state([], validator_retry_count=0)
    result = plan_validator_node(state)
    assert result["plan_validation_status"] == "invalid"
    assert result["validator_retry_count"] == 1


# ---------------------------------------------------------------------------
# validator_router
# ---------------------------------------------------------------------------

def test_router_valid_goes_to_executor_fanout():
    from chat_reasoner.nodes.plan_validator import validator_router
    state = _make_state([], validation_status="valid")
    assert validator_router(state) == "executor_fanout"


def test_router_invalid_retry_lt_3_goes_to_planner():
    from chat_reasoner.nodes.plan_validator import validator_router
    state = _make_state([], validation_status="invalid", validator_retry_count=2)
    assert validator_router(state) == "planner"


def test_router_invalid_retry_eq_3_goes_to_replanner():
    from chat_reasoner.nodes.plan_validator import validator_router
    state = _make_state([], validation_status="invalid", validator_retry_count=3)
    assert validator_router(state) == "replanner"
