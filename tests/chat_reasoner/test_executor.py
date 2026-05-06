"""
test_executor.py — isolated unit + integration tests for every function in
chat_reasoner/nodes/executor.py. No full graph invocation.

Only step_worker_node tests touch real external systems (seeded_case).
All other tests operate on hand-crafted state dicts.
"""

import pytest


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _step(step_id, tool="civil_law_rag", query="سؤال قانوني", depends_on=None):
    return {"step_id": step_id, "tool": tool, "query": query, "depends_on": depends_on or []}


def _result(step_id, status, error=""):
    return {"step_id": step_id, "tool": "civil_law_rag", "query": "q",
            "status": status, "response": "r", "sources": [], "error": error or None,
            "raw_output": {}}


def _state(plan=None, results=None, failures=None, **overrides):
    s = {
        "plan": plan or [],
        "step_results": results or [],
        "step_failures": failures or {},
        "case_id": "1234",
        "conversation_history": [],
        "status": "running",
        "replan_count": 0,
        "replan_trigger_step_id": None,
        "replan_trigger_error": None,
    }
    s.update(overrides)
    return s


# ---------------------------------------------------------------------------
# _find_ready_steps
# ---------------------------------------------------------------------------

def test_find_ready_steps_empty_plan():
    from chat_reasoner.nodes.executor import _find_ready_steps
    assert _find_ready_steps(_state()) == []


def test_find_ready_steps_all_roots_ready_with_no_results():
    from chat_reasoner.nodes.executor import _find_ready_steps
    plan = [_step("s1"), _step("s2")]
    ready = _find_ready_steps(_state(plan=plan))
    assert len(ready) == 2


def test_find_ready_steps_dep_ready_after_completion():
    from chat_reasoner.nodes.executor import _find_ready_steps
    plan = [_step("s1"), _step("s2", depends_on=["s1"])]
    results = [_result("s1", "success")]
    ready = _find_ready_steps(_state(plan=plan, results=results))
    ready_ids = {s["step_id"] for s in ready}
    assert ready_ids == {"s2"}


def test_find_ready_steps_dep_not_ready_until_completed():
    from chat_reasoner.nodes.executor import _find_ready_steps
    plan = [_step("s1"), _step("s2", depends_on=["s1"])]
    ready = _find_ready_steps(_state(plan=plan))
    ready_ids = {s["step_id"] for s in ready}
    assert "s2" not in ready_ids
    assert "s1" in ready_ids


def test_find_ready_steps_capped_step_excluded():
    from chat_reasoner.nodes.executor import _find_ready_steps
    plan = [_step("s1"), _step("s2")]
    failures = {"s1": 3}
    ready = _find_ready_steps(_state(plan=plan, failures=failures))
    ready_ids = {s["step_id"] for s in ready}
    assert "s1" not in ready_ids
    assert "s2" in ready_ids


def test_find_ready_steps_completed_not_redispatched():
    from chat_reasoner.nodes.executor import _find_ready_steps
    plan = [_step("s1"), _step("s2")]
    results = [_result("s1", "success"), _result("s2", "skipped")]
    ready = _find_ready_steps(_state(plan=plan, results=results))
    assert ready == []


# ---------------------------------------------------------------------------
# _all_steps_terminal
# ---------------------------------------------------------------------------

def test_all_steps_terminal_all_success():
    from chat_reasoner.nodes.executor import _all_steps_terminal
    plan = [_step("s1"), _step("s2")]
    results = [_result("s1", "success"), _result("s2", "success")]
    assert _all_steps_terminal(_state(plan=plan, results=results)) is True


def test_all_steps_terminal_mix_success_and_capped():
    from chat_reasoner.nodes.executor import _all_steps_terminal
    plan = [_step("s1"), _step("s2")]
    results = [_result("s1", "success")]
    failures = {"s2": 3}
    assert _all_steps_terminal(_state(plan=plan, results=results, failures=failures)) is True


def test_all_steps_terminal_one_pending():
    from chat_reasoner.nodes.executor import _all_steps_terminal
    plan = [_step("s1"), _step("s2")]
    results = [_result("s1", "success")]
    assert _all_steps_terminal(_state(plan=plan, results=results)) is False


def test_all_steps_terminal_empty_plan():
    from chat_reasoner.nodes.executor import _all_steps_terminal
    assert _all_steps_terminal(_state()) is True


# ---------------------------------------------------------------------------
# _first_capped_failure
# ---------------------------------------------------------------------------

def test_first_capped_failure_none():
    from chat_reasoner.nodes.executor import _first_capped_failure
    sid, err = _first_capped_failure(_state())
    assert sid is None
    assert err is None


def test_first_capped_failure_with_error_record():
    from chat_reasoner.nodes.executor import _first_capped_failure
    plan = [_step("s1")]
    results = [_result("s1", "failure", error="connection refused")]
    failures = {"s1": 3}
    sid, err = _first_capped_failure(_state(plan=plan, results=results, failures=failures))
    assert sid == "s1"
    assert err == "connection refused"


def test_first_capped_failure_no_failure_record():
    from chat_reasoner.nodes.executor import _first_capped_failure
    plan = [_step("s1")]
    failures = {"s1": 3}
    sid, err = _first_capped_failure(_state(plan=plan, failures=failures))
    assert sid == "s1"
    assert err == ""


# ---------------------------------------------------------------------------
# executor_fanout_node
# ---------------------------------------------------------------------------

def test_executor_fanout_returns_empty_dict():
    from chat_reasoner.nodes.executor import executor_fanout_node
    assert executor_fanout_node(_state()) == {}


# ---------------------------------------------------------------------------
# executor_dispatch_router
# ---------------------------------------------------------------------------

def test_executor_dispatch_router_no_ready_returns_collector():
    from chat_reasoner.nodes.executor import executor_dispatch_router
    # Plan with one step already succeeded → no ready steps
    plan = [_step("s1")]
    results = [_result("s1", "success")]
    result = executor_dispatch_router(_state(plan=plan, results=results))
    assert result == "collector"


def test_executor_dispatch_router_ready_steps_returns_list_of_sends():
    from langgraph.types import Send
    from chat_reasoner.nodes.executor import executor_dispatch_router
    plan = [_step("s1"), _step("s2")]
    result = executor_dispatch_router(_state(plan=plan))
    assert isinstance(result, list)
    assert len(result) == 2
    for send_obj in result:
        assert isinstance(send_obj, Send)
        assert send_obj.node == "step_worker"
        payload = send_obj.arg
        assert "step" in payload
        assert "case_id" in payload
        assert "conversation_history" in payload


def test_executor_dispatch_router_payload_uses_state_case_id():
    from langgraph.types import Send
    from chat_reasoner.nodes.executor import executor_dispatch_router
    plan = [_step("s1")]
    state = _state(plan=plan, case_id="XYZ", conversation_history=[{"role": "user", "content": "q"}])
    result = executor_dispatch_router(state)
    assert isinstance(result, list)
    payload = result[0].arg
    assert payload["case_id"] == "XYZ"
    assert payload["conversation_history"] == [{"role": "user", "content": "q"}]


# ---------------------------------------------------------------------------
# collector_node
# ---------------------------------------------------------------------------

def test_collector_node_no_capped_returns_empty():
    from chat_reasoner.nodes.executor import collector_node
    plan = [_step("s1")]
    results = [_result("s1", "success")]
    assert collector_node(_state(plan=plan, results=results)) == {}


def test_collector_node_capped_below_replan_cap_sets_trigger():
    from chat_reasoner.nodes.executor import collector_node
    plan = [_step("s1")]
    results = [_result("s1", "failure", error="boom")]
    failures = {"s1": 3}
    updates = collector_node(_state(plan=plan, results=results, failures=failures, replan_count=0))
    assert updates.get("replan_trigger_step_id") == "s1"
    assert updates.get("replan_trigger_error") is not None


def test_collector_node_capped_at_replan_cap_marks_failed():
    from chat_reasoner.nodes.executor import collector_node
    plan = [_step("s1")]
    results = [_result("s1", "failure", error="boom")]
    failures = {"s1": 3}
    updates = collector_node(_state(plan=plan, results=results, failures=failures, replan_count=2))
    assert updates.get("status") == "failed"
    assert updates.get("error_message")


# ---------------------------------------------------------------------------
# collector_router
# ---------------------------------------------------------------------------

def test_collector_router_failed_goes_to_trace_writer():
    from chat_reasoner.nodes.executor import collector_router
    state = _state(status="failed")
    assert collector_router(state) == "trace_writer"


def test_collector_router_capped_goes_to_replanner():
    from chat_reasoner.nodes.executor import collector_router
    plan = [_step("s1")]
    failures = {"s1": 3}
    state = _state(plan=plan, failures=failures)
    assert collector_router(state) == "replanner"


def test_collector_router_not_terminal_goes_to_executor_fanout():
    from chat_reasoner.nodes.executor import collector_router
    plan = [_step("s1"), _step("s2")]
    results = [_result("s1", "success")]
    state = _state(plan=plan, results=results)
    assert collector_router(state) == "executor_fanout"


def test_collector_router_all_terminal_goes_to_synthesizer():
    from chat_reasoner.nodes.executor import collector_router
    plan = [_step("s1"), _step("s2")]
    results = [_result("s1", "success"), _result("s2", "success")]
    state = _state(plan=plan, results=results)
    assert collector_router(state) == "synthesizer"


# ---------------------------------------------------------------------------
# step_worker_node (real systems, requires seeded_case)
# ---------------------------------------------------------------------------

def test_step_worker_success_path(seeded_case, _preflight):
    from chat_reasoner.nodes.executor import step_worker_node
    step = _step("sw1", tool="civil_law_rag", query="ما هي أركان المسؤولية التقصيرية في القانون المدني المصري؟")
    payload = {"step": step, "case_id": seeded_case["case_id"], "conversation_history": []}

    updates = step_worker_node(payload)

    step_results = updates.get("step_results", [])
    assert len(step_results) == 1
    r = step_results[0]
    assert r["status"] == "success", f"Expected success, got {r['status']}: {r.get('error')}"
    assert updates.get("step_failures") == {}

    log = updates.get("tool_calls_log", [])
    assert len(log) == 1
    entry = log[0]
    for key in ("step_id", "tool", "query", "status", "response_preview", "sources", "error", "timestamp"):
        assert key in entry, f"Missing key {key!r} in tool_calls_log entry"
    # ISO timestamp format
    assert "T" in entry["timestamp"]


def test_step_worker_failure_path(seeded_case, _preflight):
    from chat_reasoner.nodes.executor import step_worker_node
    from chat_reasoner import tools as _tools
    from chat_reasoner.state import StepResult

    original = _tools.TOOL_REGISTRY["civil_law_rag"]
    def always_fail(step, case_id, conversation_history):
        return StepResult(
            step_id=step["step_id"], tool="civil_law_rag", query=step["query"],
            status="failure", response="", error="simulated failure",
        )
    try:
        _tools.TOOL_REGISTRY["civil_law_rag"] = always_fail
        step = _step("sw_fail", tool="civil_law_rag", query="q")
        payload = {"step": step, "case_id": seeded_case["case_id"], "conversation_history": []}
        updates = step_worker_node(payload)

        assert updates["step_failures"] == {"sw_fail": 1}
        assert updates["step_results"][0]["status"] == "failure"
        assert updates["tool_calls_log"][0]["status"] == "failure"
    finally:
        _tools.TOOL_REGISTRY["civil_law_rag"] = original


def test_step_worker_response_preview_truncated(seeded_case, _preflight):
    from chat_reasoner.nodes.executor import step_worker_node
    from chat_reasoner import tools as _tools
    from chat_reasoner.state import StepResult

    long_response = "أ" * 500
    original = _tools.TOOL_REGISTRY["civil_law_rag"]
    def long_response_tool(step, case_id, conversation_history):
        return StepResult(
            step_id=step["step_id"], tool="civil_law_rag", query=step["query"],
            status="success", response=long_response,
        )
    try:
        _tools.TOOL_REGISTRY["civil_law_rag"] = long_response_tool
        step = _step("sw_long", tool="civil_law_rag", query="q")
        payload = {"step": step, "case_id": seeded_case["case_id"], "conversation_history": []}
        updates = step_worker_node(payload)

        preview = updates["tool_calls_log"][0]["response_preview"]
        assert len(preview) <= 200, f"response_preview not truncated: {len(preview)} chars"
    finally:
        _tools.TOOL_REGISTRY["civil_law_rag"] = original
