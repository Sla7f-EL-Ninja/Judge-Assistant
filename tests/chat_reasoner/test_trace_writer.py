"""
test_trace_writer.py — trace_writer_node writes to Mongo, indexes exist, failures isolated.
"""

from datetime import datetime, timezone

import pytest


REQUIRED_TRACE_FIELDS = {
    "session_id", "case_id", "timestamp", "original_query",
    "escalation_reason", "plan", "tool_calls", "replan_events",
    "final_answer", "run_count", "replan_count", "status",
}


def _get_traces_collection():
    from config.supervisor import MONGO_URI, MONGO_DB
    from pymongo import MongoClient
    from chat_reasoner.trace import TRACES_COLLECTION
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB][TRACES_COLLECTION], client


def _base_state(session_id, **overrides):
    state = {
        "original_query": "ما الأحكام المنطبقة؟",
        "case_id": "1234",
        "escalation_reason": "test trace",
        "plan": [{"step_id": "s1", "tool": "civil_law_rag", "query": "q"}],
        "tool_calls_log": [],
        "replan_events": [],
        "final_answer": "إجابة اختبارية",
        "run_count": 1,
        "replan_count": 0,
        "status": "succeeded",
        "error_message": None,
        "synth_sufficient": True,
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "conversation_history": [],
        "step_results": [],
        "step_failures": {},
        "replan_trigger_step_id": None,
        "replan_trigger_error": None,
        "synthesis_attempts": 1,
        "final_sources": [],
        "plan_validation_status": "valid",
        "plan_validation_feedback": "",
        "validator_retry_count": 0,
    }
    state.update(overrides)
    return state


def test_trace_writer_inserts_document(_preflight):
    from chat_reasoner.nodes.trace_writer import trace_writer_node
    session_id = f"1234::trace_test::{datetime.now(timezone.utc).isoformat()}"
    state = _base_state(session_id)

    coll, client = _get_traces_collection()
    try:
        result = trace_writer_node(state)
        assert "status" in result

        doc = coll.find_one({"session_id": session_id})
        assert doc is not None, f"Trace not found for session_id={session_id!r}"

        for field in REQUIRED_TRACE_FIELDS:
            assert field in doc, f"Missing field {field!r} in trace doc"
    finally:
        coll.delete_many({"session_id": session_id})
        client.close()


def test_trace_writer_fields_correct(_preflight):
    from chat_reasoner.nodes.trace_writer import trace_writer_node
    session_id = f"1234::trace_fields::{datetime.now(timezone.utc).isoformat()}"
    state = _base_state(session_id, final_answer="الجواب النهائي", status="succeeded")

    coll, client = _get_traces_collection()
    try:
        trace_writer_node(state)
        doc = coll.find_one({"session_id": session_id})
        assert doc["case_id"] == "1234"
        assert doc["original_query"] == "ما الأحكام المنطبقة؟"
        assert doc["final_answer"] == "الجواب النهائي"
        assert doc["status"] in ("succeeded", "failed")
        assert doc["replan_count"] == 0
        assert doc["run_count"] == 1
    finally:
        coll.delete_many({"session_id": session_id})
        client.close()


def test_trace_indexes_exist(_preflight):
    from chat_reasoner.nodes.trace_writer import trace_writer_node
    session_id = f"1234::trace_idx::{datetime.now(timezone.utc).isoformat()}"
    state = _base_state(session_id)

    coll, client = _get_traces_collection()
    try:
        trace_writer_node(state)
        indexes = {idx["name"] for idx in coll.list_indexes()}
        assert "session_id_1" in indexes, f"Index session_id_1 missing. Found: {indexes}"
        assert "case_id_1_timestamp_1" in indexes, (
            f"Compound index case_id_1_timestamp_1 missing. Found: {indexes}"
        )
        assert "timestamp_1" in indexes, f"Index timestamp_1 missing. Found: {indexes}"
    finally:
        coll.delete_many({"session_id": session_id})
        client.close()


def test_trace_writer_failure_isolation(_preflight):
    """trace_writer_node must not raise even on broken state."""
    from chat_reasoner.nodes.trace_writer import trace_writer_node

    broken_state = {
        "session_id": "1234::broken",
        "case_id": "1234",
        # started_at missing — forces KeyError / AttributeError path
        "original_query": "q",
        "escalation_reason": "x",
        "plan": [],
        "tool_calls_log": [],
        "replan_events": [],
        "final_answer": "",
        "run_count": 0,
        "replan_count": 0,
        "status": "failed",
    }

    # Must not raise
    result = trace_writer_node(broken_state)
    assert isinstance(result, dict)
    assert "status" in result
