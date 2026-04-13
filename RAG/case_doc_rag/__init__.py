"""case_doc_rag -- Case Document RAG package for Egyptian civil-case files.

Public API:
    run(query, case_id, conversation_history, request_id) -> Dict[str, Any]
    set_vectorstore(vectorstore) -> None
"""

import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

from RAG.case_doc_rag.graph import build_graph
from RAG.case_doc_rag.infrastructure import set_vectorstore

__all__ = ["run", "set_vectorstore"]

logger = logging.getLogger("case_doc_rag")

# ---------------------------------------------------------------------------
# Lazy graph singleton
# ---------------------------------------------------------------------------

_app = None
_app_lock = threading.Lock()


def _get_app():
    """Return the compiled graph (lazy, thread-safe singleton)."""
    global _app
    with _app_lock:
        if _app is None:
            logger.info("Compiling Case Doc RAG graph...")
            _app = build_graph()
            logger.info("Case Doc RAG graph compiled successfully")
        return _app


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    query: str,
    case_id: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the Case Doc RAG pipeline.

    Parameters
    ----------
    query:
        The judge's question as a plain Python string.
    case_id:
        Identifies which case to scope retrieval to.
    conversation_history:
        Prior conversation turns as list of {"role": ..., "content": ...} dicts.
    request_id:
        UUID for log tracing. Generated if not provided.

    Returns
    -------
    Dict with keys: sub_answers, final_answer, error (and other state fields).
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    logger.info(
        "[%s] run: case_id=%s query=%s",
        request_id, case_id, (query[:100] if query else ""),
    )

    initial_state = {
        "query": query,
        "case_id": case_id,
        "conversation_history": conversation_history or [],
        "request_id": request_id,
        "sub_questions": [],
        "on_topic": False,
        "doc_selection_mode": "no_doc_specified",
        "selected_doc_id": None,
        "doc_titles": [],
        "sub_answers": [],
        "final_answer": "",
        "error": None,
    }

    try:
        result = _get_app().invoke(initial_state)
    except Exception:
        logger.critical(
            "[%s] Unhandled exception in pipeline", request_id, exc_info=True
        )
        return {
            "sub_answers": [],
            "final_answer": "",
            "error": "Internal pipeline failure",
        }

    logger.info(
        "[%s] run complete: %d sub_answers, error=%s",
        request_id,
        len(result.get("sub_answers", [])),
        result.get("error"),
    )
    return result
