"""
enrich_context.py

Context-enrichment node for the Supervisor workflow.

Pre-fetches case-level data from MongoDB once per turn so that individual
agents do not each query the database independently.  Writes enriched fields
into state before dispatch.  Failures are non-fatal — agents fall back to
their own retrieval paths.
"""

import logging
from typing import Any, Dict

from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)


def enrich_context_node(state: SupervisorState) -> Dict[str, Any]:
    """Fetch case summary and document metadata and merge into state.

    Adds ``case_summary`` and ``case_doc_titles`` to state if not already
    present.  Agents can read these without hitting MongoDB themselves.
    """
    case_id = state.get("case_id", "")
    if not case_id:
        return {}

    # Only enrich for intents that need case data
    intent = state.get("intent", "")
    if intent in ("civil_law_rag", "off_topic"):
        return {}

    enriched: Dict[str, Any] = {}

    try:
        import pymongo
        from config import cfg

        mongo_uri = cfg.get("mongodb", {}).get("uri", "mongodb://localhost:27017/")
        mongo_db_name = cfg.get("mongodb", {}).get("db", "judge_assistant")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db = client[mongo_db_name]

        # Fetch latest case summary
        summary_doc = db["summaries"].find_one(
            {"case_id": case_id},
            {"summary": 1, "_id": 0},
            sort=[("generated_at", -1)],
        )
        if summary_doc:
            enriched["case_summary"] = summary_doc.get("summary", "")

        # Fetch document titles for this case (used by case_doc_rag for context)
        docs_cursor = db["documents"].find(
            {"case_id": case_id},
            {"title": 1, "doc_type": 1, "_id": 0},
        ).limit(20)
        doc_titles = [
            f"{d.get('title', '?')} ({d.get('doc_type', '?')})"
            for d in docs_cursor
        ]
        if doc_titles:
            enriched["case_doc_titles"] = doc_titles

        client.close()

    except Exception as exc:
        logger.warning("Context enrichment failed (non-fatal): %s", exc)

    return enriched
