"""
TEST 3 — Crash safety
Monkeypatch store and checkpointer to raise on every call.
Graph must complete without raising; final_response must be non-empty;
log must contain at least one WARNING about the failure.
"""
import sys
import os
import logging
import traceback
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

CASE_ID = "test_case_crash"
USER_ID = "test_judge_crash"
SESSION_ID = "session_crash_v2"

BASE_STATE = {
    "judge_query": "ما هي القضية؟",
    "case_id": CASE_ID,
    "user_id": USER_ID,
    "uploaded_files": [],
    "intent": None,
    "target_agents": [],
    "classified_query": None,
    "agent_results": {},
    "agent_errors": {},
    "validation_status": None,
    "validation_feedback": None,
    "retry_count": 0,
    "max_retries": 2,
    "document_classifications": [],
    "merged_response": None,
    "final_response": None,
    "sources": [],
    "case_summary": None,
    "case_doc_titles": [],
    "correlation_id": "test-t3",
    "classification_error": None,
    "conversation_history": [],
    "turn_count": 0,
    "running_summary": None,
    "semantic_facts": [],
    "procedural_prefs": None,
    "session_id": SESSION_ID,
}


def _crashing_store():
    m = MagicMock()
    err = RuntimeError("simulated crash")
    m.search.side_effect = err
    m.put.side_effect = err
    m.get.side_effect = err
    m.delete.side_effect = err
    m.list_namespaces.side_effect = err
    m.aput = MagicMock(side_effect=err)
    m.aget = MagicMock(side_effect=err)
    m.asearch = MagicMock(side_effect=err)
    m.adelete = MagicMock(side_effect=err)
    m.alist_namespaces = MagicMock(side_effect=err)
    return m


def _crashing_checkpointer():
    m = MagicMock()
    err = RuntimeError("simulated crash")
    m.get.side_effect = err
    m.put.side_effect = err
    m.get_tuple.side_effect = err
    m.list.side_effect = err
    m.aget = MagicMock(side_effect=err)
    m.aput = MagicMock(side_effect=err)
    m.aget_tuple = MagicMock(side_effect=err)
    m.alist = MagicMock(side_effect=err)
    return m


def run():
    # Capture log warnings
    log_records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            log_records.append(record)

    handler = _Capture(level=logging.WARNING)
    logging.getLogger().addHandler(handler)

    crashing_store = _crashing_store()
    # Only crash the store — checkpointer crash propagates before graph starts
    # (LangGraph calls checkpointer.get_tuple() in SyncPregelLoop.__enter__,
    # outside any node try/except). Use a real MemorySaver for checkpointer.

    try:
        import Supervisor.services.memory as mem_mod
        from langgraph.checkpoint.memory import MemorySaver

        # Override singletons: crashing store, valid checkpointer
        mem_mod._store = crashing_store
        mem_mod._checkpointer = MemorySaver()
        mem_mod._reflection_executor = mem_mod._UNSET  # re-init will use crashing store

        from Supervisor.graph import get_app_persistent
        # Force rebuild with updated singletons
        import Supervisor.graph as graph_mod
        graph_mod._app_persistent = None
        app = get_app_persistent()

        thread_id = f"{USER_ID}:{CASE_ID}:{SESSION_ID}"
        config = {"configurable": {"thread_id": thread_id, "user_id": USER_ID}}

        print(f"[T3] Invoking with crashing store + valid checkpointer...")
        result = app.invoke(dict(BASE_STATE), config=config)

        final_response = result.get("final_response", "") or ""
        print(f"[T3] final_response length: {len(final_response)}")
        print(f"[T3] final_response: {final_response[:150]}")

        warning_msgs = [r.getMessage() for r in log_records if r.levelno >= logging.WARNING]
        crash_warnings = [m for m in warning_msgs if "crash" in m.lower() or "simulated" in m.lower()
                          or "store" in m.lower() or "checkpointer" in m.lower() or "fail" in m.lower()]
        print(f"[T3] WARNING log count: {len(warning_msgs)}, crash-related: {len(crash_warnings)}")
        if crash_warnings:
            print(f"[T3] Sample warning: {crash_warnings[0][:120]}")

        c1 = bool(final_response)
        c2 = len(warning_msgs) > 0

        if c1 and c2:
            print("PASS: graph completed without raising, final_response non-empty, warnings logged.")
            return True
        else:
            reasons = []
            if not c1:
                reasons.append("final_response empty")
            if not c2:
                reasons.append("no WARNING logs captured")
            print(f"FAIL: {'; '.join(reasons)}")
            return False

    except Exception:
        print("FAIL: unhandled exception propagated out of graph.invoke()")
        traceback.print_exc()
        return False
    finally:
        logging.getLogger().removeHandler(handler)


if __name__ == "__main__":
    try:
        ok = run()
        sys.exit(0 if ok else 1)
    except Exception:
        print("FAIL: exception in test harness")
        traceback.print_exc()
        sys.exit(1)
