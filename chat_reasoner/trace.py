"""
trace.py

Developer-only trace writer for chat_reasoner_traces MongoDB collection.
Always best-effort: any exception is logged and swallowed so a Mongo outage
never fails the reasoning run.
"""

import logging

logger = logging.getLogger(__name__)

TRACES_COLLECTION = "chat_reasoner_traces"


def write_trace(state: dict) -> None:
    try:
        from pymongo import MongoClient, ASCENDING
        from config.supervisor import MONGO_URI, MONGO_DB

        client = MongoClient(MONGO_URI)
        try:
            coll = client[MONGO_DB][TRACES_COLLECTION]

            # Idempotent index creation (cheap after first call)
            coll.create_index([("session_id", ASCENDING)])
            coll.create_index([("case_id", ASCENDING), ("timestamp", ASCENDING)])
            coll.create_index([("timestamp", ASCENDING)])

            coll.insert_one({
                "session_id":        state.get("session_id", ""),
                "case_id":           state.get("case_id", ""),
                "timestamp":         state.get("started_at", ""),
                "original_query":    state.get("original_query", ""),
                "escalation_reason": state.get("escalation_reason", ""),
                "plan":              state.get("plan", []),
                "tool_calls":        state.get("tool_calls_log", []),
                "replan_events":     state.get("replan_events", []),
                "final_answer":      state.get("final_answer", ""),
                "run_count":         state.get("run_count", 0),
                "replan_count":      state.get("replan_count", 0),
                "status":            state.get("status", "unknown"),
            })
        finally:
            client.close()

    except Exception as exc:
        logger.exception("Trace write failed (non-fatal): %s", exc)
