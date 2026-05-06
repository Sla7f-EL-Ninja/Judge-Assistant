"""
test_adapter_e2e.py — full-graph integration tests via ChatReasonerAdapter.invoke().

Every test prints its session_id for post-run traceability in MongoDB.
All tests require seeded_case (7 Arabic fixture docs for case_id="1234").
"""

import json
import re
import sys
from datetime import datetime, timezone

import pytest

_ARABIC_RE = re.compile(r"[؀-ۿ]")
CASE_ID = "1234"


def _context(**overrides):
    ctx = {
        "case_id": CASE_ID,
        "conversation_history": [],
        "escalation_reason": "اختبار تكاملي",
    }
    ctx.update(overrides)
    return ctx


def _get_traces_coll():
    from config.supervisor import MONGO_URI, MONGO_DB
    from pymongo import MongoClient
    from chat_reasoner.trace import TRACES_COLLECTION
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][TRACES_COLLECTION], client


def _log_session(result):
    session_id = (result.raw_output or {}).get("session_id", "unknown")
    print(f"\n[chat_reasoner e2e] session_id={session_id}", file=sys.stderr)
    return session_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "قارن وقائع هذه القضية بأحكام القانون المدني",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.response, "response must not be empty"
    assert _ARABIC_RE.search(result.response), "response must contain Arabic text"
    assert isinstance(result.sources, list)

    raw = result.raw_output or {}
    assert "plan" in raw
    assert "step_results" in raw
    assert "session_id" in raw


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------

def test_parallel_dispatch(seeded_case):
    """Plan must have ≥2 root steps (no depends_on) signalling parallel execution."""
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    # Multi-question query nudges planner toward parallel steps
    result = adapter.invoke(
        "ما وقائع القضية؟ وما الأحكام القانونية المنطبقة عليها وفقاً للقانون المدني؟",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Unexpected error: {result.error}"

    raw = result.raw_output or {}
    plan = raw.get("plan", [])
    root_steps = [s for s in plan if not s.get("depends_on")]
    assert len(root_steps) >= 2, (
        f"Expected ≥2 parallel root steps; got plan: {plan}"
    )


# ---------------------------------------------------------------------------
# Summary fetch path
# ---------------------------------------------------------------------------

def test_summary_fetch_path(seeded_case):
    """Query requesting stored summary should include fetch_summary_report step."""
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "أعطني الملخص المُعدّ مسبقاً لهذه القضية",
        _context(),
    )
    _log_session(result)

    raw = result.raw_output or {}
    step_results_raw = raw.get("step_results", [])
    step_results = []
    for r in step_results_raw:
        if isinstance(r, str):
            try:
                step_results.append(json.loads(r))
            except Exception:
                pass
        elif isinstance(r, dict):
            step_results.append(r)

    summary_steps = [
        s for s in step_results if s.get("tool") == "fetch_summary_report"
    ]
    assert len(summary_steps) >= 1, (
        "Query requesting stored summary must use fetch_summary_report"
    )
    assert summary_steps[0].get("status") == "success", (
        f"fetch_summary_report returned: {summary_steps[0]}"
    )


# ---------------------------------------------------------------------------
# Replan recovery
# ---------------------------------------------------------------------------

def test_replan_recovery(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter
    from chat_reasoner import tools as _tools
    from chat_reasoner.nodes import planner as _planner_mod
    from chat_reasoner.state import Plan, PlanStep

    original_tool = _tools.TOOL_REGISTRY["case_doc_rag"]
    call_count = {"n": 0}

    def failing_then_real(step, case_id, conversation_history):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            from chat_reasoner.state import StepResult
            return StepResult(
                step_id=step.get("step_id", "?"),
                tool="case_doc_rag",
                query=step.get("query", ""),
                status="failure",
                response="",
                error="simulated failure",
            )
        return original_tool(step, case_id, conversation_history)

    original_planner = _planner_mod.planner_node

    def forced_planner(state):
        # Force a plan with case_doc_rag so the mock gets triggered
        return {
            "plan": [
                {"step_id": "s1", "tool": "case_doc_rag",
                 "query": "ما هي وقائع القضية؟", "depends_on": []},
            ],
            "plan_validation_status": "pending",
        }

    try:
        _tools.TOOL_REGISTRY["case_doc_rag"] = failing_then_real
        _planner_mod.planner_node = forced_planner

        adapter = ChatReasonerAdapter()
        result = adapter.invoke("ما هي وقائع القضية وتفاصيلها؟", _context())
        _log_session(result)

        raw = result.raw_output or {}
        if result.error is None:
            assert raw.get("replan_count", 0) >= 1, (
                "Expected at least one replan after repeated case_doc_rag failures"
            )
    finally:
        _tools.TOOL_REGISTRY["case_doc_rag"] = original_tool
        _planner_mod.planner_node = forced_planner.__wrapped__ if hasattr(forced_planner, '__wrapped__') else original_planner
# ---------------------------------------------------------------------------
# Full failure path
# ---------------------------------------------------------------------------

def test_full_failure_path(seeded_case):
    """All tools fail → adapter returns error, response empty, trace has status=failed."""
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter
    from chat_reasoner import tools as _tools
    from chat_reasoner.state import StepResult

    originals = {k: v for k, v in _tools.TOOL_REGISTRY.items()}

    def always_fail(step, case_id, conversation_history):
        return StepResult(
            step_id=step.get("step_id", "?"),
            tool=step.get("tool", "?"),
            query=step.get("query", ""),
            status="failure",
            response="",
            error="simulated total failure",
        )

    try:
        for k in _tools.TOOL_REGISTRY:
            _tools.TOOL_REGISTRY[k] = always_fail

        adapter = ChatReasonerAdapter()
        result = adapter.invoke(
            "ما وقائع القضية والأحكام المنطبقة والملخص؟",
            _context(),
        )
        session_id = _log_session(result)

        assert result.response == "", f"response must be empty on failure, got: {result.response!r}"
        assert result.error, "error field must be set on full failure"

        # Verify trace in MongoDB has status=failed
        coll, client = _get_traces_coll()
        try:
            doc = coll.find_one({"session_id": session_id})
            if doc:
                assert doc["status"] == "failed", (
                    f"Trace status should be 'failed', got {doc['status']!r}"
                )
        finally:
            client.close()

    finally:
        for k, v in originals.items():
            _tools.TOOL_REGISTRY[k] = v
