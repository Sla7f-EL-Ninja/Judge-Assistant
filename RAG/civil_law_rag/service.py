"""
service.py
----------
Public service interface for the Civil Law RAG pipeline.

This is the SINGLE entry point for all callers — including the Supervisor
adapter.  It handles:

1. Input validation (Arabic ratio, length limits)
2. Versioned semantic cache lookup
3. Graph invocation with a fresh, deep-copied state
4. Structured source extraction
5. Typed error propagation

The Supervisor adapter must call ask_question() — never app.invoke()
directly — so that validation and caching always apply.
"""

from __future__ import annotations

import logging
import re
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

from config import cfg

# Part 3.3d: read at call time so the cache key reflects the live config,
# not a value frozen at import time (which becomes stale after env-var reload).
def _current_llm_model() -> str:
    return cfg.llm.get("high", {}).get("model", "gemini-2.5-flash")
MAX_QUERY_LENGTH: int = 2000
MIN_QUERY_LENGTH: int = 3
MIN_ARABIC_RATIO: float = 0.3
from RAG.civil_law_rag.errors import QueryValidationError
from RAG.civil_law_rag.indexing.normalizer import normalize
from RAG.civil_law_rag.state import make_initial_state
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

# Module-level singleton cache
from RAG.civil_law_rag.cache import SemanticCache
_cache = SemanticCache()

# Module-level singleton graph (avoid rebuilding per call)
_app = None
_app_lock = threading.Lock()


def _get_app():
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                from RAG.civil_law_rag.graph import build_graph
                _app = build_graph()
    return _app


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CivilLawResult:
    answer: str
    sources: List[dict] = field(default_factory=list)
    classification: Optional[str] = None
    retrieval_confidence: Optional[float] = None
    citation_integrity: Optional[str] = None
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_query(query: str) -> str:
    """Validate and normalize a query.  Returns stripped query on success."""
    if not query or not isinstance(query, str):
        raise QueryValidationError("الاستعلام فارغ أو غير صالح.")

    query = query.strip()

    if len(query) < MIN_QUERY_LENGTH:
        raise QueryValidationError(
            f"الاستعلام قصير جدًا (الحد الأدنى {MIN_QUERY_LENGTH} أحرف)."
        )
    if len(query) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"الاستعلام طويل جدًا (الحد الأقصى {MAX_QUERY_LENGTH} حرف)."
        )

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", query))
    total_chars  = len(query.replace(" ", ""))
    if total_chars > 0 and (arabic_chars / total_chars) < MIN_ARABIC_RATIO:
        raise QueryValidationError(
            "الاستعلام لا يحتوي على نسبة كافية من النص العربي."
        )

    return query


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------

def _extract_sources(result_state: dict) -> List[dict]:
    """Build structured citation list from last_results."""
    sources = []
    for doc in result_state.get("last_results", []):
        meta = getattr(doc, "metadata", {})
        idx  = meta.get("index")
        if idx is None:
            continue
        sources.append({
            "article": idx,
            "title":   meta.get("title", f"المادة {idx}"),
            "book":    meta.get("book"),
            "part":    meta.get("part"),
            "chapter": meta.get("chapter"),
        })
    return sources


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_question(query: str) -> CivilLawResult:
    """Process a query through the Civil Law RAG pipeline.

    Args:
        query: The user's legal question in Arabic.

    Returns:
        CivilLawResult with answer, sources, and metadata.

    Raises:
        QueryValidationError: for invalid input (callers may choose to catch).
    """
    # 1. Validate
    query = validate_query(query)

    # 2. Cache lookup
    llm_model = _current_llm_model()
    cached = _cache.get(query, llm_model=llm_model)
    if cached is not None:
        cached_answer, cached_sources = cached
        return CivilLawResult(answer=cached_answer, sources=cached_sources, from_cache=True)

    # 3. Graph invocation
    try:
        app = _get_app()

        state = make_initial_state()
        state["last_query"] = query

        result_state = app.invoke(state)

        answer = result_state.get("final_answer") or "تعذر الحصول على إجابة."
        sources = _extract_sources(result_state)

        # 4. Cache successful answers
        if answer:
            _cache.set(query, answer, sources=sources, llm_model=llm_model)
        return CivilLawResult(
            answer=answer,
            sources=sources,
            classification=result_state.get("classification"),
            retrieval_confidence=result_state.get("retrieval_confidence"),
            citation_integrity=result_state.get("citation_integrity"),
        )

    except Exception:
        log_event(
            logger, "ask_question_error",
            query=query[:200],
            traceback=traceback.format_exc(),
            level=logging.ERROR,
        )
        return CivilLawResult(
            answer="حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى لاحقًا."
        )


def clear_cache() -> None:
    _cache.clear()