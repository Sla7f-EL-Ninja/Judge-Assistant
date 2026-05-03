# """
# state.py
# --------
# LangGraph state type and factory for the legal_rag engine.

# Critical: ALWAYS use make_initial_state(corpus_config) — never copy the
# template dict.  Shallow copies share the same underlying lists across
# concurrent requests, causing state bleed under production load.
# """

# from __future__ import annotations

# import copy
# from typing import List, Optional, TypedDict

# from langchain_core.documents import Document

# from RAG.legal_rag.corpus_config import CorpusConfig


# class State(TypedDict):
#     # ── Corpus identity ──────────────────────────────────────────────────────
#     corpus_config: Optional[CorpusConfig]  # injected by ask_question(); never mutated

#     # ── Core pipeline state ───────────────────────────────────────────────────
#     last_query: Optional[str]
#     last_results: List[Document]
#     last_answer: Optional[str]
#     current_book: Optional[str]
#     current_part: Optional[str]
#     current_chapter: Optional[str]
#     current_article: Optional[int]
#     filter_type: str
#     k: int
#     books_in_scope: List[str]
#     query_history: List[str]
#     retrieval_history: List[List[Document]]
#     answer_history: List[str]
#     db_initialized: bool
#     split_config: dict
#     rewritten_question: Optional[str]
#     classification: Optional[str]        # analytical | textual | off_topic
#     retrieval_confidence: Optional[float]
#     retry_count: int
#     max_retries: int
#     refined_query: Optional[str]
#     grade: Optional[str]                 # pass | refine | fail
#     llm_pass: Optional[bool]
#     failure_reason: Optional[str]
#     proceedToGenerate: Optional[bool]
#     retrieval_attempts: int
#     llm_call_count: int                  # tracks budget; see MAX_LLM_CALLS
#     final_answer: Optional[str]
#     current_section: Optional[str]
#     scope_confidence: Optional[float]    # 0–1, from scope_classifier_node
#     scope_filter: dict                   # metadata filter applied during retrieval
#     citation_integrity: Optional[str]    # "full" | "partial" | "none"


# # ---------------------------------------------------------------------------
# # Default values — plain dict, not a TypedDict instance.
# # Note: corpus_config has no default; it MUST be set by make_initial_state().
# # ---------------------------------------------------------------------------
# _DEFAULTS: dict = {
#     "corpus_config": None,
#     "last_query": None,
#     "last_results": [],
#     "last_answer": None,
#     "current_book": None,
#     "current_part": None,
#     "current_chapter": None,
#     "current_article": None,
#     "filter_type": "",
#     "k": 8,
#     "books_in_scope": [],
#     "query_history": [],
#     "retrieval_history": [],
#     "answer_history": [],
#     "db_initialized": True,
#     "split_config": {},
#     "rewritten_question": None,
#     "classification": None,
#     "retrieval_confidence": None,
#     "retry_count": 0,
#     "max_retries": 3,
#     "refined_query": None,
#     "grade": None,
#     "llm_pass": None,
#     "failure_reason": None,
#     "proceedToGenerate": None,
#     "retrieval_attempts": 0,
#     "llm_call_count": 0,
#     "final_answer": None,
#     "current_section": None,
#     "scope_confidence": None,
#     "scope_filter": {},
#     "citation_integrity": None,
# }


# def make_initial_state(corpus_config: CorpusConfig) -> dict:
#     """Return a fresh, deep-copied state dict safe for concurrent use.

#     corpus_config is the only required argument — it identifies which
#     legal corpus this invocation targets.  Every list/dict inside is a
#     new object — no shared mutable state between concurrent graph
#     invocations.

#     Args:
#         corpus_config: The CorpusConfig for the target law (civil, evidence, procedures…).

#     Returns:
#         A fresh state dict ready to pass to app.invoke().
#     """
#     state = copy.deepcopy(_DEFAULTS)
#     # CorpusConfig is frozen/immutable — no need to deep-copy it.
#     state["corpus_config"] = corpus_config
#     return state


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
}


def make_initial_state() -> dict:
    """Return a fresh, deep-copied state dict safe for concurrent use.

    corpus_config is intentionally absent from the arguments — it is
    resolved at runtime by corpus_router_node and must not be pre-injected
    by the caller.

    Every list/dict inside is a new object — no shared mutable state
    between concurrent graph invocations.

    Returns:
        A fresh state dict ready to pass to app.invoke().
    """
    return copy.deepcopy(_DEFAULTS)