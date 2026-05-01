# """
# reranker.py
# -----------
# HTTP client for the remote TEI reranker service (bge-reranker-v2-m3).

# The reranker takes a query and a list of candidate texts, then returns
# a relevance score for each pair.  This is a cross-encoder: unlike
# bi-encoders (BAAI/bge-m3), it processes query + document together and
# is significantly more accurate for legal retrieval.

# Pipeline position:
#     dense/hybrid retrieval (top-30) → reranker → top-k (default 5) → graders

# Falls back gracefully: if TEI reranker is unreachable, the original
# retrieval order is preserved and a warning is logged.
# """

# from __future__ import annotations
# import os
# import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# import logging
# import threading
# from typing import List

# import httpx
# from langchain_core.documents import Document

# from RAG.civil_law_rag.config import TEI_RERANKER_URL
# from RAG.civil_law_rag.telemetry import get_logger, log_event

# logger = get_logger(__name__)

# _reranker_lock = threading.Lock()
# _reranker_available: bool | None = None  # None = not yet probed


# def _probe_reranker() -> bool:
#     """Return True if the TEI reranker endpoint is reachable."""
#     global _reranker_available
#     if _reranker_available is not None:
#         return _reranker_available
#     with _reranker_lock:
#         if _reranker_available is not None:
#             return _reranker_available
#         try:
#             resp = httpx.post(
#                 f"{TEI_RERANKER_URL}/rerank",
#                 json={"query": "probe", "texts": ["probe"]},
#                 timeout=5,
#             )
#             resp.raise_for_status()
#             _reranker_available = True
#             log_event(logger, "reranker_init", backend="tei", url=TEI_RERANKER_URL)
#         except Exception as exc:
#             _reranker_available = False
#             log_event(
#                 logger, "reranker_init",
#                 backend="unavailable", reason=str(exc),
#                 level=logging.WARNING,
#             )
#     return _reranker_available


# def rerank(
#     query: str,
#     docs: List[Document],
#     top_k: int = 5,
#     timeout: int = 30,
# ) -> List[Document]:
#     """Rerank *docs* by relevance to *query*, return top *top_k*.

#     If the TEI reranker is unavailable, returns the first *top_k* docs
#     from the original retrieval order without modification.

#     Args:
#         query:   The search query.
#         docs:    Candidate documents from hybrid retrieval.
#         top_k:   How many to keep after reranking.
#         timeout: HTTP timeout in seconds.

#     Returns:
#         Reranked and truncated list of Documents.
#     """
#     if not docs:
#         return docs

#     if not _probe_reranker():
#         log_event(logger, "rerank_skipped", reason="reranker_unavailable", top_k=top_k)
#         return docs[:top_k]

#     texts = [d.page_content for d in docs]

#     try:
#         resp = httpx.post(
#             f"{TEI_RERANKER_URL}/rerank",
#             json={"query": query, "texts": texts},
#             timeout=timeout,
#         )
#         resp.raise_for_status()
#         scores: List[dict] = resp.json()  # [{"index": int, "score": float}, ...]

#         # Sort by descending score, keep top_k
#         ranked = sorted(scores, key=lambda x: x["score"], reverse=True)[:top_k]
#         result = [docs[item["index"]] for item in ranked]

#         log_event(
#             logger, "rerank",
#             input_docs=len(docs),
#             output_docs=len(result),
#             top_score=ranked[0]["score"] if ranked else None,
#         )
#         return result

#     except Exception as exc:
#         log_event(
#             logger, "rerank_error",
#             error=str(exc),
#             fallback="original_order",
#             level=logging.WARNING,
#         )
#         return docs[:top_k]


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

from RAG.civil_law_rag.config import TEI_RERANKER_URL
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_reranker_lock = threading.Lock()
_reranker_available: bool | None = None  # None = not yet probed

# In-process CrossEncoder fallback
_cross_encoder = None
_cross_encoder_lock = threading.Lock()
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


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

def _get_cross_encoder():
    """Return the shared CrossEncoder instance, loading it once."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    with _cross_encoder_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        try:
            from sentence_transformers import CrossEncoder
            log_event(logger, "reranker_inprocess_loading", model=_RERANKER_MODEL)
            _cross_encoder = CrossEncoder(_RERANKER_MODEL)
            log_event(logger, "reranker_inprocess_ready", model=_RERANKER_MODEL)
        except Exception as exc:
            log_event(
                logger, "reranker_inprocess_failed",
                error=str(exc),
                level=logging.ERROR,
            )
            _cross_encoder = None

    return _cross_encoder


def _rerank_inprocess(
    query: str,
    docs: List[Document],
    top_k: int,
) -> List[Document]:
    """Rerank using in-process CrossEncoder."""
    encoder = _get_cross_encoder()
    if encoder is None:
        log_event(logger, "rerank_skipped", reason="cross_encoder_unavailable")
        return docs[:top_k]

    pairs = [(query, d.page_content) for d in docs]
    scores = encoder.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in ranked[:top_k]]

    log_event(
        logger, "rerank_inprocess",
        input_docs=len(docs),
        output_docs=len(result),
        top_score=float(ranked[0][0]) if ranked else None,
    )
    return result


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