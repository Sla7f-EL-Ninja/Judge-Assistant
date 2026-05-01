"""
textual.py
----------
Textual node: handles queries that ask for the literal text of articles.

Uses direct Qdrant scroll (no embedding call) for exact article lookup,
which eliminates all embedding overhead for textual queries.
"""

from __future__ import annotations

import re
import logging
from typing import List

from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)
from langsmith import traceable

from RAG.civil_law_rag.retrieval.vectorstore import get_qdrant_client, COLLECTION_NAME
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)


def _scroll_by_index(article_numbers: List[int]) -> List[Document]:
    """Fetch articles by exact index using Qdrant scroll (no embedding).

    Returns Documents in ascending article-number order.
    """
    client = get_qdrant_client()
    filt = Filter(
        must=[
            FieldCondition(key="metadata.source", match=MatchValue(value="civil_law")),
            FieldCondition(key="metadata.type", match=MatchValue(value="article")),
            FieldCondition(
                key="metadata.index",
                match=MatchAny(any=article_numbers),
            ),
        ]
    )
    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filt,
        limit=len(article_numbers) + 5,  # small buffer
        with_payload=True,
        with_vectors=False,
    )

    # Build Documents and sort by article index
    docs: List[Document] = []
    for point in results:
        payload = point.payload or {}
        meta = payload.get("metadata", payload)
        content = payload.get("page_content", "")
        docs.append(Document(page_content=content, metadata=meta))

    docs.sort(key=lambda d: d.metadata.get("index", 0))
    return docs


@traceable(name="Textual Node")
def textual_node(state: dict) -> dict:
    """Return the literal text of requested article(s)."""
    query = state.get("rewritten_question") or state.get("last_query", "")

    # ── Range: "بين X و Y" ────────────────────────────────────────────────
    range_match = re.search(r"(?:بين\s*(\d+)\s*و\s*(\d+)|من\s*(?:المادة\s*)?(\d+)\s*(?:الي|إلى|إلي|الى)\s*(?:المادة\s*)?(\d+))", query)
    if range_match:
        if range_match.group(1) is not None: # This means the first pattern "بين X و Y" matched
            start = int(range_match.group(1))
            end = int(range_match.group(2))
        else: # This means the second pattern "من X الي Y" matched
            start = int(range_match.group(3))
            end = int(range_match.group(4))
        numbers = list(range(start, end + 1))
        docs = _scroll_by_index(numbers)
        state["current_article"] = f"{start}-{end}"
        state["last_results"] = docs
        state["final_answer"] = (
            "\n\n".join(d.page_content for d in docs)
            if docs
            else "عذرًا، لم أتمكن من العثور على نص المواد المطلوبة."
        )
        log_event(logger, "textual_range",
                  start=start, end=end, found=len(docs))
        return state

    # ── Exact article number: "المادة X" ─────────────────────────────────
    article_match = re.search(r"المادة\s*(\d+)", query)
    if article_match:
        article_num = int(article_match.group(1))
        docs = _scroll_by_index([article_num])
        state["current_article"] = article_num
        state["last_results"] = docs
        state["final_answer"] = (
            "\n\n".join(d.page_content for d in docs)
            if docs
            else "عذرًا، لم أتمكن من العثور على نص المادة المطلوبة."
        )
        log_event(logger, "textual_exact",
                  article=article_num, found=len(docs))
        return state

    # ── Fallback: semantic search on article chunks ──────────────────────
    from RAG.civil_law_rag.retrieval.vectorstore import load_vectorstore, civil_law_filter
    from qdrant_client.models import FieldCondition, MatchValue

    db = load_vectorstore()
    article_filter = civil_law_filter([
        FieldCondition(key="metadata.type", match=MatchValue(value="article"))
    ])
    docs = db.similarity_search(query, k=3, filter=article_filter)
    state["last_results"] = docs
    state["final_answer"] = (
        "\n\n".join(d.page_content for d in docs)
        if docs
        else "عذرًا، لم أتمكن من العثور على نص المادة المطلوبة."
    )
    log_event(logger, "textual_semantic_fallback", found=len(docs))
    return state
