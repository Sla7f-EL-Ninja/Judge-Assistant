"""state_factory.py — builds SupervisorState dicts for tests."""

import uuid
from typing import Any, Dict, List, Optional


def make_state(
    judge_query: str = "ما نص المادة 163 من القانون المدني المصري؟",
    case_id: str = "",
    uploaded_files: Optional[List[str]] = None,
    conversation_history: Optional[List[dict]] = None,
    turn_count: int = 0,
    intent: str = "",
    target_agents: Optional[List[str]] = None,
    classified_query: str = "",
    agent_results: Optional[Dict[str, Any]] = None,
    agent_errors: Optional[Dict[str, str]] = None,
    validation_status: str = "",
    validation_feedback: str = "",
    retry_count: int = 0,
    max_retries: int = 3,
    document_classifications: Optional[List[Dict[str, Any]]] = None,
    merged_response: str = "",
    final_response: str = "",
    sources: Optional[List[str]] = None,
    case_summary: Optional[str] = None,
    case_doc_titles: Optional[List[str]] = None,
    correlation_id: Optional[str] = None,
    classification_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a fully populated SupervisorState-compatible dict."""
    return {
        "judge_query": judge_query,
        "case_id": case_id or f"test-case-{uuid.uuid4()}",
        "uploaded_files": uploaded_files or [],
        "conversation_history": conversation_history or [],
        "turn_count": turn_count,
        "intent": intent,
        "target_agents": target_agents or [],
        "classified_query": classified_query or judge_query,
        "agent_results": agent_results or {},
        "agent_errors": agent_errors or {},
        "validation_status": validation_status,
        "validation_feedback": validation_feedback,
        "retry_count": retry_count,
        "max_retries": max_retries,
        "document_classifications": document_classifications or [],
        "merged_response": merged_response,
        "final_response": final_response,
        "sources": sources or [],
        "case_summary": case_summary,
        "case_doc_titles": case_doc_titles or [],
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "classification_error": classification_error,
    }


def make_agent_result(
    response: str = "إجابة تجريبية",
    sources: Optional[List[str]] = None,
    raw_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "response": response,
        "sources": sources or [],
        "raw_output": raw_output or {},
    }
