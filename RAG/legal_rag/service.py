"""
service.py
----------
Public service interface for the legal_rag engine.

This is the SINGLE entry point for all callers — including the Supervisor
adapter and each corpus's thin __init__.py wrapper.  It handles:

1. Input validation (Arabic ratio, length limits)
2. Per-corpus versioned semantic cache lookup
3. Graph invocation with a fresh, deep-copied state
4. Structured source extraction
5. Typed error propagation

Callers must always call ask_question(query, corpus_config) — never
app.invoke() directly — so that validation and caching always apply.
"""

from __future__ import annotations

import logging
import re
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from RAG.legal_rag.cache import SemanticCache
from RAG.legal_rag.config import LLM_MODEL, MAX_QUERY_LENGTH, MIN_ARABIC_RATIO, MIN_QUERY_LENGTH
from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.errors import QueryValidationError
from RAG.legal_rag.indexing.normalizer import normalize
from RAG.legal_rag.state import make_initial_state
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

# One SemanticCache instance per corpus  {corpus_name: SemanticCache}
_caches: Dict[str, SemanticCache] = {}


def _get_cache(corpus_config: CorpusConfig) -> SemanticCache:
    if corpus_config.name not in _caches:
        _caches[corpus_config.name] = SemanticCache()
    return _caches[corpus_config.name]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class LegalRAGResult:
    answer: str
    sources: List[dict]            = field(default_factory=list)
    classification: Optional[str]  = None
    retrieval_confidence: Optional[float] = None
    citation_integrity: Optional[str]     = None
    from_cache: bool               = False
    corpus: Optional[str]          = None   # corpus_config.name for traceability


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_query(query: str) -> str:
    """Validate and normalize a query. Returns stripped query on success."""
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

def ask_question(query: str, corpus_config: CorpusConfig) -> LegalRAGResult:
    """Process a query through the legal_rag pipeline for *corpus_config*.

    Args:
        query:         The user's legal question in Arabic.
        corpus_config: Which legal corpus to query.

    Returns:
        LegalRAGResult with answer, sources, and metadata.

    Raises:
        QueryValidationError: for invalid input (callers may choose to catch).
    """
    # 1. Validate
    query = validate_query(query)

    # 2. Cache lookup
    cache  = _get_cache(corpus_config)
    cached = cache.get(query, corpus_config=corpus_config, llm_model=LLM_MODEL)
    if cached is not None:
        return LegalRAGResult(
            answer=cached, from_cache=True, corpus=corpus_config.name
        )

    # 3. Graph invocation
    try:
        from RAG.legal_rag.graph import build_graph
        app = build_graph(corpus_config)

        state               = make_initial_state(corpus_config)
        state["last_query"] = query

        result_state = app.invoke(state)

        answer  = result_state.get("final_answer") or "تعذر الحصول على إجابة."
        sources = _extract_sources(result_state)

        # 4. Cache successful answers
        if answer:
            cache.set(query, answer, corpus_config=corpus_config, llm_model=LLM_MODEL)

        return LegalRAGResult(
            answer=answer,
            sources=sources,
            classification=result_state.get("classification"),
            retrieval_confidence=result_state.get("retrieval_confidence"),
            citation_integrity=result_state.get("citation_integrity"),
            corpus=corpus_config.name,
        )

    except Exception:
        log_event(logger, "ask_question_error",
                  corpus=corpus_config.name,
                  query=query[:200],
                  traceback=traceback.format_exc(),
                  level=logging.ERROR)
        return LegalRAGResult(
            answer="حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى لاحقًا.",
            corpus=corpus_config.name,
        )


def clear_cache(corpus_config: CorpusConfig) -> None:
    """Clear the semantic cache for one corpus."""
    _get_cache(corpus_config).clear()


def clear_all_caches() -> None:
    """Clear all corpus caches."""
    for cache in _caches.values():
        cache.clear()
