"""
reranker.py
-----------
HTTP client for the remote TEI reranker service (bge-reranker-v2-m3).

The reranker takes a query and a list of candidate texts, then returns
a relevance score for each pair.  This is a cross-encoder: unlike
bi-encoders (BAAI/bge-m3), it processes query + document together and
is significantly more accurate for legal retrieval.

Pipeline position:
    dense/hybrid retrieval (top-13) → reranker → top-k (default 5) → graders

Priority:
    1. TEI HTTP service (production — GPU-accelerated, shared across workers)
    2. In-process CrossEncoder (dev/CI — slower but functionally identical)
    3. Original retrieval order (last resort if model fails to load)
"""

from __future__ import annotations

import logging
import threading
from typing import List

import httpx
from langchain_core.documents import Document

from RAG.legal_rag.config import TEI_RERANKER_URL
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_reranker_lock = threading.Lock()
_reranker_available: bool | None = None  # None = not yet probed

# In-process CrossEncoder is intentionally disabled.
# Loading ~1 GB weights in-process blocks the event loop and exhausts
# virtual memory on Windows. When TEI is unavailable the pipeline falls
# back to original retrieval order via _rerank_inprocess().


# ---------------------------------------------------------------------------
# TEI probe
# ---------------------------------------------------------------------------

def _probe_reranker() -> bool:
    """Return True if the TEI reranker endpoint is reachable."""
    global _reranker_available
    if _reranker_available is not None:
        return _reranker_available
    with _reranker_lock:
        if _reranker_available is not None:
            return _reranker_available
        try:
            resp = httpx.post(
                f"{TEI_RERANKER_URL}/rerank",
                json={"query": "probe", "texts": ["probe"]},
                timeout=5,
            )
            resp.raise_for_status()
            _reranker_available = True
            log_event(logger, "reranker_init", backend="tei", url=TEI_RERANKER_URL)
        except Exception as exc:
            _reranker_available = False
            log_event(
                logger, "reranker_init",
                backend="unavailable", reason=str(exc),
                level=logging.WARNING,
            )
    return _reranker_available


# ---------------------------------------------------------------------------
# In-process CrossEncoder fallback
# ---------------------------------------------------------------------------


def _rerank_inprocess(
    query: str,
    docs: List[Document],
    top_k: int,
) -> List[Document]:
    """Rerank using in-process CrossEncoder.

    Skipped entirely when TEI is unavailable — loading a ~1 GB model
    in-process blocks the event loop, exhausts virtual memory on Windows,
    and is not acceptable in a dev environment without a GPU.
    Falls back to original retrieval order instead.
    """
    log_event(
        logger, "rerank_skipped",
        reason="tei_unavailable_inprocess_disabled",
        level=logging.WARNING,
    )
    return docs[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    docs: List[Document],
    top_k: int = 20,
    timeout: int = 30,
) -> List[Document]:
    """Rerank *docs* by relevance to *query*, return top *top_k*.

    Tries TEI first, falls back to in-process CrossEncoder, then
    falls back to original retrieval order if both are unavailable.

    Args:
        query:   The search query.
        docs:    Candidate documents from hybrid retrieval.
        top_k:   How many to keep after reranking.
        timeout: HTTP timeout in seconds.

    Returns:
        Reranked and truncated list of Documents.
    """
    if not docs:
        return docs

    # ── TEI path ────────────────────────────────────────────────────────────
    if _probe_reranker():
        texts = [d.page_content for d in docs]
        try:
            resp = httpx.post(
                f"{TEI_RERANKER_URL}/rerank",
                json={"query": query, "texts": texts},
                timeout=timeout,
            )
            resp.raise_for_status()
            scores: List[dict] = resp.json()

            ranked = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
            result = [docs[item["index"]] for item in ranked]

            log_event(
                logger, "rerank",
                input_docs=len(docs),
                output_docs=len(result),
                top_score=ranked[0]["score"] if ranked else None,
            )
            return result

        except Exception as exc:
            log_event(
                logger, "rerank_tei_error",
                error=str(exc),
                fallback="inprocess",
                level=logging.WARNING,
            )

    # ── In-process CrossEncoder path ────────────────────────────────────────
    log_event(logger, "rerank_using_inprocess")
    return _rerank_inprocess(query, docs, top_k)