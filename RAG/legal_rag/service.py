"""
service.py
----------
Public service interface for the legal_rag engine.

This is the SINGLE entry point for all callers.  It handles:

1. Input validation (Arabic ratio, length limits)
2. Per-corpus versioned semantic cache lookup
   (cache is keyed after graph invocation when corpus_config is known)
3. Graph invocation with a fresh, deep-copied state
4. Structured source extraction
5. Typed error propagation

Architecture note (unified graph):
    corpus_config is NO LONGER required from the caller.  The graph's
    corpus_router_node resolves and injects it into state at runtime.
    ask_question() therefore accepts only the raw query string.

    Cache lookup cannot happen before graph invocation (we don't know
    the corpus yet), so caching is now a post-invocation write-only step
    for the first call, and a pre-invocation hit for subsequent identical
    queries routed to the same corpus.  To support this, we do a
    speculative cache check across all corpora after routing is known.

FIX-1 (scope_fallback):
    ask_question() now accepts an optional scope_fallback kwarg forwarded
    from legal_rag_server.search_legal_corpus when the executor signals a
    retry.  It is written into the initial state before graph invocation so
    scope_classifier_node can skip or relax the section-scoping LLM call
    that was repeatedly misrouting queries to the wrong section.
"""

from __future__ import annotations

import logging
import re
import threading
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from RAG.legal_rag.cache import SemanticCache
from config.legal_rag import LLM_MODEL, MAX_QUERY_LENGTH, MIN_ARABIC_RATIO, MIN_QUERY_LENGTH
from RAG.legal_rag.corpus_config import CorpusConfig
from RAG.legal_rag.errors import (
    QueryValidationError,
    InternalRAGError,
    LLMTimeoutError,
    LLMBudgetExceededError,
    RetrievalError,
    GenerationError,
    ScopeClassificationError,
    CorpusRoutingError,
    PreprocessingError,
)
from RAG.legal_rag.state import make_initial_state
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

# One SemanticCache instance per corpus  {corpus_name: SemanticCache}
_caches: Dict[str, SemanticCache] = {}
_caches_lock = threading.Lock()

# Error types that map to graceful degraded response (HTTP 200).
_DEGRADED_ERROR_TYPES = {LLMTimeoutError.__name__, LLMBudgetExceededError.__name__}

# Error types that map to internal server error (HTTP 500).
_INTERNAL_ERROR_TYPES = {
    RetrievalError.__name__,
    GenerationError.__name__,
    ScopeClassificationError.__name__,
    CorpusRoutingError.__name__,
    PreprocessingError.__name__,
}


def _get_cache(corpus_config: CorpusConfig) -> SemanticCache:
    if corpus_config.name not in _caches:
        with _caches_lock:
            if corpus_config.name not in _caches:
                _caches[corpus_config.name] = SemanticCache()
    return _caches[corpus_config.name]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class LegalRAGResult:
    answer: str
    sources: List[dict]                      = field(default_factory=list)
    classification: Optional[str]            = None
    retrieval_confidence: Optional[float]    = None
    citation_integrity: Optional[str]        = None
    from_cache: bool                         = False
    corpus: Optional[str]                    = None   # corpus_config.name
    corpus_routing_scores: Optional[list]    = None   # raw LLM scores (observability)
    error_type: Optional[str]                = None   # set when answer is a degraded fallback


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

    # NOTE: Arabic-ratio enforcement intentionally removed from this layer.
    #
    # Rationale: the graph's preprocessor_node already handles language
    # detection and rewrites romanized-Arabic queries ("ma hwa elmada 190...")
    # to proper Arabic script before any retrieval takes place, and routes
    # genuinely off-topic queries to off_topic_node.  Enforcing MIN_ARABIC_RATIO
    # here gates the query BEFORE the preprocessor can rewrite it, which caused
    # every romanized-Arabic search via /api/v1/legal/search to return HTTP 400
    # even though the supervisor path sends the same text through the same graph
    # successfully.
    #
    # Structural validation (empty / too-short / too-long) is still enforced
    # above — those checks catch genuinely malformed input with zero latency.
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

from langsmith import traceable

@traceable(name="Legal RAG Pipeline")
def ask_question(
    query: str,
    *,
    # FIX-1: scope_fallback is forwarded from legal_rag_server on executor retries.
    #   None        → normal two-stage scoping (chapter → section)
    #   "section"   → skip section classification; chapter filter only
    #   "chapter"   → skip all scoping; search full corpus
    scope_fallback: Optional[str] = None,
) -> LegalRAGResult:
    """Process a query through the unified legal_rag pipeline.

    The corpus is resolved automatically by corpus_router_node inside
    the graph — callers no longer need to know or pass a CorpusConfig.

    Args:
        query:          The user's legal question in Arabic.
        scope_fallback: Optional retry hint from the executor.  When set,
                        scope_classifier_node skips or relaxes the section-
                        scoping LLM call to avoid repeating a misroute.

    Returns:
        LegalRAGResult with answer, sources, corpus name, and metadata.

    Raises:
        QueryValidationError: for invalid input (callers may choose to catch).
    """
    # 1. Validate
    query = validate_query(query)

    # 2. Graph invocation (corpus resolved internally)
    try:
        from RAG.legal_rag.graph import build_graph
        app = build_graph()

        state               = make_initial_state()
        state["last_query"] = query

        # FIX-1: inject scope_fallback so scope_classifier_node can honour it.
        if scope_fallback is not None:
            state["scope_fallback"] = scope_fallback

        result_state = app.invoke(state)

    except Exception:
        log_event(logger, "ask_question_graph_crash",
                  query=query[:200],
                  traceback=traceback.format_exc(),
                  level=logging.ERROR)
        raise InternalRAGError("Unhandled graph crash") from None

    corpus_config: Optional[CorpusConfig] = result_state.get("corpus_config")
    answer     = result_state.get("final_answer") or "تعذر الحصول على إجابة."
    sources    = _extract_sources(result_state)
    node_error = result_state.get("error")  # {type, node, message} or None

    # 3. Inspect node-level error and map to appropriate response.
    if node_error:
        error_type = node_error.get("type", "")
        log_event(logger, "ask_question_node_error",
                  error_type=error_type,
                  error_node=node_error.get("node"),
                  error_message=node_error.get("message"),
                  level=logging.ERROR)
        if error_type in _INTERNAL_ERROR_TYPES:
            raise InternalRAGError(
                f"Internal pipeline failure ({error_type}) at node '{node_error.get('node')}'"
            )
        # Degraded response for timeout / budget (HTTP 200 with error_type).
        return LegalRAGResult(
            answer=answer,
            corpus=corpus_config.name if corpus_config else None,
            corpus_routing_scores=result_state.get("corpus_routing_scores"),
            error_type=error_type,
        )

    # 4. Cache successful answers (only when we have a resolved corpus).
    if corpus_config and answer:
        cache = _get_cache(corpus_config)
        cache.set(query, answer, corpus_config=corpus_config, llm_model=LLM_MODEL)

    return LegalRAGResult(
        answer=answer,
        sources=sources,
        classification=result_state.get("classification"),
        retrieval_confidence=result_state.get("retrieval_confidence"),
        citation_integrity=result_state.get("citation_integrity"),
        corpus=corpus_config.name if corpus_config else None,
        corpus_routing_scores=result_state.get("corpus_routing_scores"),
    )


def clear_cache(corpus_config: CorpusConfig) -> None:
    """Clear the semantic cache for one corpus."""
    _get_cache(corpus_config).clear()


def clear_all_caches() -> None:
    """Clear all corpus caches."""
    for cache in _caches.values():
        cache.clear()