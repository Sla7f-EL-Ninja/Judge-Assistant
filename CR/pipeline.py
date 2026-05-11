"""
Public programmatic interface for the Case Reasoner pipeline.

    from CR.pipeline import run_case_reasoning

    result = run_case_reasoning(
        case_id="abc123",
        case_brief={...},
        rendered_brief="..."
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import cfg, get_llm
from CR import build_case_reasoner_graph

logger = logging.getLogger("hakim.cr.pipeline")


@dataclass
class CaseReasoningResult:
    """Structured result returned by run_case_reasoning."""

    final_report: str
    issue_analyses: List[Dict[str, Any]] = field(default_factory=list)
    cross_issue_relationships: List[Dict[str, Any]] = field(default_factory=list)
    consistency_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    reconciliation_paragraphs: List[str] = field(default_factory=list)
    per_issue_confidence: List[Dict[str, Any]] = field(default_factory=list)
    case_level_confidence: Dict[str, Any] = field(default_factory=dict)

    saved_to_db: bool = False
    case_id: Optional[str] = None


def _stringify_keys(obj: Any) -> Any:
    """
    Recursively convert all dict keys to strings.

    MongoDB requires every key in a document to be a string.
    The compute_confidence node emits per_issue_confidence as
    [{1: 0.73}, {2: 0.86}, ...] with integer keys, which causes:
        "documents must have only string keys, key was 1"
    """
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(item) for item in obj]
    return obj


def _upsert_case_reasoning(
    case_id: str,
    result: CaseReasoningResult,
) -> bool:
    """Upsert result into MongoDB under case_id."""
    try:
        from pymongo import MongoClient
    except ImportError:
        logger.error(
            "pymongo is not installed — cannot save to MongoDB. "
            "Run: pip install pymongo"
        )
        return False

    mongo_uri  = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
    db_name    = cfg.mongodb.get("database", "TESTING")
    collection = "case_reasonings"  # Matches your Motor collection name
    timeout_ms = cfg.mongodb.get("server_selection_timeout_ms", 5000)

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        doc = {
            "case_id": case_id,
            "final_report": result.final_report,
            "issue_analyses": _stringify_keys(result.issue_analyses),
            "cross_issue_relationships": _stringify_keys(result.cross_issue_relationships),
            "consistency_conflicts": _stringify_keys(result.consistency_conflicts),
            "reconciliation_paragraphs": result.reconciliation_paragraphs,
            "per_issue_confidence": _stringify_keys(result.per_issue_confidence),
            "case_level_confidence": _stringify_keys(result.case_level_confidence),
            "generated_at": datetime.now(timezone.utc),
        }
        client[db_name][collection].update_one(
            {"case_id": case_id},
            {"$set": doc},
            upsert=True,
        )
        logger.info(
            "Case reasoning upserted  db='%s'  collection='%s'  case_id='%s'",
            db_name, collection, case_id,
        )
        return True
    except Exception as exc:
        logger.error("MongoDB upsert failed: %s", exc)
        return False
    finally:
        client.close()


def run_case_reasoning(
    *,
    case_brief: dict,
    rendered_brief: str,
    judge_query: str = "",
    case_id: Optional[str] = None,
    save_to_db: bool = True,
    llm=None,
) -> CaseReasoningResult:
    """Run the Case Reasoner graph and return a structured result."""
    if llm is None:
        llm = get_llm("high")

    app = build_case_reasoner_graph()

    initial_state = {
        "case_id": case_id or "",
        "judge_query": judge_query,
        "case_brief": case_brief,
        "rendered_brief": rendered_brief,
        "identified_issues": [],
        "issue_analyses": [],
        "cross_issue_relationships": [],
        "consistency_conflicts": [],
        "reconciliation_paragraphs": [],
        "per_issue_confidence": [],
        "case_level_confidence": {},
        "final_report": "",
        "intermediate_steps": [],
        "error_log": [],
    }

    logger.info("run_case_reasoning: starting  case_id=%s", case_id)

    try:
        final_state = app.invoke(initial_state)
    except Exception as exc:
        raise RuntimeError(f"Case Reasoner pipeline failed: {exc}") from exc

    result = CaseReasoningResult(
        final_report              = final_state.get("final_report", ""),
        issue_analyses            = final_state.get("issue_analyses", []),
        cross_issue_relationships = final_state.get("cross_issue_relationships", []),
        consistency_conflicts     = final_state.get("consistency_conflicts", []),
        reconciliation_paragraphs = final_state.get("reconciliation_paragraphs", []),
        per_issue_confidence      = final_state.get("per_issue_confidence", []),
        case_level_confidence     = final_state.get("case_level_confidence", {}),
        case_id                   = case_id,
    )

    logger.info(
        "run_case_reasoning: done  issues_found=%d  report_len=%d",
        len(result.issue_analyses), len(result.final_report)
    )

    if case_id and save_to_db:
        result.saved_to_db = _upsert_case_reasoning(case_id, result)

    return result