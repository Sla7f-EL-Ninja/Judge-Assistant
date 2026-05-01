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

from RAG.legal_rag.retrieval.vectorstore import get_qdrant_client, load_vectorstore, source_filter
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)


def _scroll_by_index(
    article_numbers: List[int],
    collection_name: str,
    source_value: str,
) -> List[Document]:
    """Fetch articles by exact index using Qdrant scroll (no embedding)."""
    client = get_qdrant_client()
    filt = Filter(
        must=[
            FieldCondition(key="metadata.source", match=MatchValue(value=source_value)),
            FieldCondition(key="metadata.type",   match=MatchValue(value="article")),
            FieldCondition(
                key="metadata.index",
                match=MatchAny(any=article_numbers),
            ),
        ]
    )
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=filt,
        limit=len(article_numbers) + 5,
        with_payload=True,
        with_vectors=False,
    )

    docs: List[Document] = []
    for point in results:
        payload = point.payload or {}
        meta    = payload.get("metadata", payload)
        content = payload.get("page_content", "")
        docs.append(Document(page_content=content, metadata=meta))

    docs.sort(key=lambda d: d.metadata.get("index", 0))
    return docs


@traceable(name="Textual Node")
def textual_node(state: dict) -> dict:
    """Return the literal text of requested article(s)."""
    query         = state.get("rewritten_question") or state.get("last_query", "")
    corpus_config = state.get("corpus_config")
    collection    = corpus_config.collection_name     if corpus_config else "civil_law_docs"
    source_val    = corpus_config.source_filter_value if corpus_config else "civil_law"

    # ── Range: "بين X و Y" or "من X إلى Y" ──────────────────────────────
    range_match = re.search(
        r"(?:بين\s*(\d+)\s*و\s*(\d+)"
        r"|من\s*(?:المادة\s*)?(\d+)\s*(?:الي|إلى|إلي|الى)\s*(?:المادة\s*)?(\d+))",
        query,
    )
    if range_match:
        if range_match.group(1) is not None:
            start, end = int(range_match.group(1)), int(range_match.group(2))
        else:
            start, end = int(range_match.group(3)), int(range_match.group(4))
        numbers = list(range(start, end + 1))
        docs    = _scroll_by_index(numbers, collection, source_val)
        state["current_article"] = f"{start}-{end}"
        state["last_results"]    = docs
        state["final_answer"]    = (
            "\n\n".join(d.page_content for d in docs)
            if docs
            else "عذرًا، لم أتمكن من العثور على نص المواد المطلوبة."
        )
        log_event(logger, "textual_range",
                  corpus=source_val, start=start, end=end, found=len(docs))
        return state

    # ── Exact article number: "المادة X" ──────────────────────────────────
    article_match = re.search(r"المادة\s*(\d+)", query)
    if article_match:
        article_num = int(article_match.group(1))
        docs        = _scroll_by_index([article_num], collection, source_val)
        state["current_article"] = article_num
        state["last_results"]    = docs
        state["final_answer"]    = (
            "\n\n".join(d.page_content for d in docs)
            if docs
            else "عذرًا، لم أتمكن من العثور على نص المادة المطلوبة."
        )
        log_event(logger, "textual_exact",
                  corpus=source_val, article=article_num, found=len(docs))
        return state

    # ── Fallback: semantic search on article chunks ───────────────────────
    db             = load_vectorstore(collection)
    article_filter = source_filter(
        source_val,
        [FieldCondition(key="metadata.type", match=MatchValue(value="article"))],
    )
    docs = db.similarity_search(query, k=3, filter=article_filter)
    state["last_results"] = docs
    state["final_answer"] = (
        "\n\n".join(d.page_content for d in docs)
        if docs
        else "عذرًا، لم أتمكن من العثور على نص المادة المطلوبة."
    )
    log_event(logger, "textual_semantic_fallback", corpus=source_val, found=len(docs))
    return state
