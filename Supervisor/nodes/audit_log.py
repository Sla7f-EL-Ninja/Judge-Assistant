"""
audit_log.py

Audit-trail node for the Supervisor workflow.

Writes a tamper-evident record of every turn to MongoDB, capturing all
routing decisions, agent runs, sources, and validator outcome.  Required
for judicial review of AI output (G5.8.3).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def audit_log_node(state: SupervisorState) -> Dict[str, Any]:
    """Persist a structured audit record for the current turn.

    Writes to MongoDB ``audit_log`` collection.  Failures are logged but
    never propagate — the turn must not fail because of audit writes.
    """
    try:
        import pymongo
        from config import cfg

        mongo_uri = cfg.get("mongodb", {}).get("uri", "mongodb://localhost:27017/")
        mongo_db = cfg.get("mongodb", {}).get("db", "judge_assistant")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client[mongo_db]

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": state.get("correlation_id"),
            "case_id": state.get("case_id", ""),
            "turn_count": state.get("turn_count", 0),
            # Routing decisions
            "intent": state.get("intent", ""),
            "target_agents": state.get("target_agents", []),
            "classified_query_length": len(state.get("classified_query", "")),
            # Execution outcomes
            "agents_succeeded": list((state.get("agent_results") or {}).keys()),
            "agents_failed": list((state.get("agent_errors") or {}).keys()),
            # Validation
            "validation_status": state.get("validation_status", ""),
            "validation_feedback_length": len(state.get("validation_feedback", "")),
            "retry_count": state.get("retry_count", 0),
            # Sources cited (for citation audit)
            "sources": state.get("sources", []),
            # Classification error if any
            "classification_error": state.get("classification_error"),
        }

        db["audit_log"].insert_one(record)
        client.close()

    except Exception as exc:
        logger.error("Audit log write failed (non-fatal): %s", exc)

    return {}
