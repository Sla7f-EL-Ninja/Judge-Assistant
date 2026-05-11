"""
state.py
--------
LangGraph state type and factory for the legal_rag engine.

Critical: ALWAYS use make_initial_state() — never copy the template dict.
Shallow copies share the same underlying lists across concurrent requests,
causing state bleed under production load.

Architecture note (unified graph):
    corpus_config is now optional at state creation time.  It is injected
    by corpus_router_node during graph execution and must NOT be pre-set
    by the caller (service.py).

FIX-1 (scope_fallback):
    scope_fallback is an optional string written by service.ask_question()
    before graph invocation.  scope_classifier_node reads it to skip or
    relax the two-stage scoping LLM calls on retry attempts:
        None        → normal two-stage scoping (chapter → section)
        "section"   → skip section classification; chapter filter only
        "chapter"   → skip all scoping; return empty filter (full corpus)
"""

from __future__ import annotations

import copy
from typing import List, Optional, TypedDict

from langchain_core.documents import Document

from RAG.legal_rag.corpus_config import CorpusConfig


class State(TypedDict):
    # ── Corpus identity ───────────────────────────────────────────────────
    corpus_config: Optional[CorpusConfig]   # set by corpus_router_node at runtime
    corpus_routing_scores: list             # raw LLM scores from corpus_router_node

    # ── Core pipeline state ───────────────────────────────────────────────
    last_query: Optional[str]
    last_results: List[Document]
    last_answer: Optional[str]
    current_book: Optional[str]
    current_part: Optional[str]
    current_chapter: Optional[str]
    current_article: Optional[int]
    filter_type: str
    k: int
    books_in_scope: List[str]
    query_history: List[str]
    retrieval_history: List[List[Document]]
    answer_history: List[str]
    db_initialized: bool
    split_config: dict
    rewritten_question: Optional[str]
    classification: Optional[str]           # analytical | textual | off_topic
    retrieval_confidence: Optional[float]
    retry_count: int
    max_retries: int
    refined_query: Optional[str]
    grade: Optional[str]                    # pass | refine | fail
    llm_pass: Optional[bool]
    failure_reason: Optional[str]
    proceedToGenerate: Optional[bool]
    retrieval_attempts: int
    llm_call_count: int                     # tracks budget; see MAX_LLM_CALLS
    final_answer: Optional[str]
    current_section: Optional[str]
    scope_confidence: Optional[float]       # 0–1, from scope_classifier_node
    scope_filter: dict                      # metadata filter applied during retrieval
    citation_integrity: Optional[str]       # "full" | "partial" | "none"

    # ── Scope override (FIX-1) ────────────────────────────────────────────
    # Injected by service.ask_question() on retry attempts.
    # scope_classifier_node reads this to skip or relax LLM scoping calls
    # when a previous attempt at a narrower scope returned zero documents.
    #   None        → normal two-stage scoping (chapter → section)
    #   "section"   → skip section classification; filter by chapter only
    #   "chapter"   → skip all scoping; return empty filter (full corpus)
    scope_fallback: Optional[str]

    # ── Error tracking ────────────────────────────────────────────────────
    error: Optional[dict]                   # {type, node, message} — set on hard failures


# ---------------------------------------------------------------------------
# Default values
# corpus_config intentionally starts as None — corpus_router_node sets it.
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "corpus_config":          None,
    "corpus_routing_scores":  [],
    "last_query":             None,
    "last_results":           [],
    "last_answer":            None,
    "current_book":           None,
    "current_part":           None,
    "current_chapter":        None,
    "current_article":        None,
    "filter_type":            "",
    "k":                      8,
    "books_in_scope":         [],
    "query_history":          [],
    "retrieval_history":      [],
    "answer_history":         [],
    "db_initialized":         True,
    "split_config":           {},
    "rewritten_question":     None,
    "classification":         None,
    "retrieval_confidence":   None,
    "retry_count":            0,
    "max_retries":            3,
    "refined_query":          None,
    "grade":                  None,
    "llm_pass":               None,
    "failure_reason":         None,
    "proceedToGenerate":      None,
    "retrieval_attempts":     0,
    "llm_call_count":         0,
    "final_answer":           None,
    "current_section":        None,
    "scope_confidence":       None,
    "scope_filter":           {},
    "citation_integrity":     None,
    "scope_fallback":         None,   # FIX-1
    "error":                  None,
}


def make_initial_state() -> dict:
    """Return a fresh, deep-copied state dict safe for concurrent use.

    corpus_config is intentionally absent from the arguments — it is
    resolved at runtime by corpus_router_node and must not be pre-injected
    by the caller.

    scope_fallback defaults to None (normal scoping).  service.ask_question()
    sets it when the executor has signalled a scope-broadening retry.

    Every list/dict inside is a new object — no shared mutable state
    between concurrent graph invocations.

    Returns:
        A fresh state dict ready to pass to app.invoke().
    """
    return copy.deepcopy(_DEFAULTS)