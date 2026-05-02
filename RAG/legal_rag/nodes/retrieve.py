"""
retrieve.py
-----------
Retrieval node: dense search with cross-encoder reranking.

Pipeline:
    1. Normalize query
    2. Build Qdrant filter — always type=article + source from corpus_config,
       plus optional chapter/section from state['scope_filter']
    3. Single dense search (or HyDE multi-query when hyde_enabled=true)
    4. Deduplicate candidates by article index (keep max-score per article)
    5. Cross-encoder reranker (union → top-20)
    6. Store results + confidence in state
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from qdrant_client.models import FieldCondition, Filter, MatchValue
from langsmith import traceable

from config.legal_rag import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT, cfg
from RAG.legal_rag.indexing.normalizer import normalize
from RAG.legal_rag.prompts import HYDE_EXPANSION_PROMPT
from RAG.legal_rag.retrieval.vectorstore import load_vectorstore, source_filter
from RAG.legal_rag.retrieval.reranker import rerank
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_RETRIEVE_K  = 40
_RERANK_TOP  = 20
_MAX_WORKERS = 4

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("medium")
    return _llm


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _expand_queries(query: str, law_name: str, state: dict) -> List[str]:
    """Return [original, hyde_doc, para1, para2]. Falls back to [original] on failure."""
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        return [query]
    try:
        prompt = HYDE_EXPANSION_PROMPT.format(law_name=law_name, query=query)
        response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
        state["llm_call_count"] = state.get("llm_call_count", 0) + 1
        data    = json.loads(_strip_code_fences(response.content.strip()))
        queries = [query]
        hyde    = data.get("hypothetical_article", "").strip()
        if hyde:
            queries.append(normalize(hyde))
        for p in data.get("paraphrases", []):
            if p and p.strip():
                queries.append(normalize(p.strip()))
        log_event(logger, "expand_queries", original=query, expansion_count=len(queries))
        return queries
    except Exception as exc:
        log_event(logger, "expand_error", error=str(exc), fallback="single_query",
                  level=logging.WARNING)
        return [query]


def _parallel_search(db, queries: List[str], article_filter) -> List[Tuple[Document, float]]:
    all_pairs: List[Tuple[Document, float]] = []

    def _search(q: str):
        return db.similarity_search_with_relevance_scores(q, k=_RETRIEVE_K, filter=article_filter)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_search, q): q for q in queries}
        for future in as_completed(futures):
            try:
                all_pairs.extend(future.result())
            except Exception as exc:
                log_event(logger, "search_error", query=futures[future],
                          error=str(exc), level=logging.WARNING)
    return all_pairs


def _dedupe_by_index(pairs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
    best: dict    = {}
    no_index: list = []
    for doc, score in pairs:
        idx = doc.metadata.get("index")
        if idx is None:
            no_index.append((doc, score))
            continue
        existing = best.get(idx)
        if existing is None or score > existing[1]:
            best[idx] = (doc, score)
    combined = list(best.values()) + no_index
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


def _hyde_enabled() -> bool:
    return bool(cfg.get("rag", {}).get("legal", {}).get("hyde_enabled", False))


def _build_filter(source_value: str, scope_filter: dict) -> Filter:
    """Build Qdrant filter: mandatory article + source + optional scope."""
    conditions = [
        FieldCondition(key="metadata.type", match=MatchValue(value="article"))
    ]
    chapter = scope_filter.get("chapter")
    section = scope_filter.get("section")
    if chapter:
        conditions.append(
            FieldCondition(key="metadata.chapter", match=MatchValue(value=chapter))
        )
    if section:
        conditions.append(
            FieldCondition(key="metadata.section", match=MatchValue(value=section))
        )
    return source_filter(source_value, conditions)


@traceable(name="Retrieve Node")
def retrieve_node(state: dict) -> dict:
    """Dense retrieval + reranking into state['last_results']."""
    corpus_config  = state.get("corpus_config")
    collection     = corpus_config.collection_name     if corpus_config else "civil_law_docs"
    source_val     = corpus_config.source_filter_value if corpus_config else "civil_law"
    law_name       = corpus_config.law_display_name    if corpus_config else "القانون"

    db = load_vectorstore(collection)

    raw_query = (
        state.get("refined_query")
        or state.get("rewritten_question")
        or state.get("last_query", "")
    )
    query         = normalize(raw_query)
    scope_filt    = state.get("scope_filter") or {}
    article_filter = _build_filter(source_val, scope_filt)

    if _hyde_enabled():
        queries         = _expand_queries(query, law_name, state)
        all_pairs       = _parallel_search(db, queries, article_filter)
        expansion_count = len(queries)
    else:
        all_pairs       = db.similarity_search_with_relevance_scores(
            query, k=_RETRIEVE_K, filter=article_filter
        )
        expansion_count = 1

    if not all_pairs:
        state["last_results"]         = []
        state["retrieval_confidence"] = 0.0
        log_event(logger, "retrieve", query=query, docs=0, confidence=0.0,
                  scope=scope_filt, corpus=source_val)
        return state

    unique_pairs  = _dedupe_by_index(all_pairs)
    unique_docs   = [doc for doc, _ in unique_pairs]
    unique_scores = {doc.metadata.get("index"): score for doc, score in unique_pairs}

    reranked_docs   = rerank(query, unique_docs, top_k=_RERANK_TOP)
    reranked_scores = [
        unique_scores.get(d.metadata.get("index"), 0.0) for d in reranked_docs
    ]
    confidence = sum(reranked_scores) / len(reranked_scores) if reranked_scores else 0.0

    state["last_results"]         = reranked_docs
    state["retrieval_confidence"] = round(confidence, 3)

    log_event(logger, "retrieve",
              query=query,
              corpus=source_val,
              scope=scope_filt,
              expansion_count=expansion_count,
              raw_candidates=len(all_pairs),
              unique_candidates=len(unique_docs),
              reranked_docs=len(reranked_docs),
              confidence=round(confidence, 3),
              top_indices=[d.metadata.get("index") for d in reranked_docs])
    return state
