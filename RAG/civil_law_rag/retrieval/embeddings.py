"""
embeddings.py
-------------
HTTP client for the remote TEI (Text Embeddings Inference) embedding service.

Why remote instead of in-process?
- In-process HuggingFaceEmbeddings loads ~2 GB of weights inside the
  FastAPI worker, blocking the event loop on CPU inference and
  multiplying memory usage per worker.
- TEI runs as a dedicated container (with optional GPU), serves
  requests concurrently over HTTP, and is shared by all workers.

Falls back to in-process HuggingFaceEmbeddings when the TEI service
is unreachable (dev / CI mode), so tests don't require docker.
"""

from __future__ import annotations

import logging
import threading
from typing import List

import httpx

import os

from config import cfg
from RAG.civil_law_rag.telemetry import get_logger, log_event

EMBEDDING_MODEL: str = cfg.embedding.get("model", "BAAI/bge-m3")
TEI_EMBEDDING_URL: str = os.environ.get(
    "JA_TEI_EMBEDDING_URL",
    cfg.tei.get("embedding_url", "http://localhost:8080"),
)

logger = get_logger(__name__)

_client_lock = threading.Lock()
_client_instance = None  # TEIEmbeddings singleton


# ---------------------------------------------------------------------------
# TEI HTTP client
# ---------------------------------------------------------------------------

class TEIEmbeddings:
    """Lightweight client for TEI's /embed endpoint.

    Compatible with langchain_core.embeddings.Embeddings interface so it
    can be passed directly to QdrantVectorStore as the ``embedding=`` arg.
    """

    def __init__(self, url: str, timeout: int = 30, max_retries: int = 3) -> None:
        self._url = url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._url,
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=max_retries),
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self._http.post("/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    # langchain compatibility
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# In-process fallback
# ---------------------------------------------------------------------------

def _make_inprocess_embeddings():
    """Load HuggingFaceEmbeddings in-process (dev / CI fallback)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.warning(
        "TEI service unavailable at %s — falling back to in-process embeddings. "
        "This is only acceptable in dev/CI environments.",
        TEI_EMBEDDING_URL,
    )
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_client():
    """Return the shared embedding client (TEI or in-process fallback).

    Thread-safe double-checked locking ensures single initialization.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    with _client_lock:
        if _client_instance is not None:
            return _client_instance

        # Probe TEI with a tiny request
        try:
            probe = httpx.post(
                f"{TEI_EMBEDDING_URL}/embed",
                json={"inputs": ["probe"]},
                timeout=5,
            )
            probe.raise_for_status()
            _client_instance = TEIEmbeddings(TEI_EMBEDDING_URL)
            log_event(logger, "embeddings_init", backend="tei", url=TEI_EMBEDDING_URL)
        except Exception as exc:
            log_event(
                logger, "embeddings_init",
                backend="inprocess", reason=str(exc),
                level=logging.WARNING,
            )
            _client_instance = _make_inprocess_embeddings()

    return _client_instance