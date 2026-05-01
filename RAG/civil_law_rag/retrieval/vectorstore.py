"""
vectorstore.py
--------------
Qdrant vector store initialization for the Civil Law RAG pipeline.

Design decisions:
- Uses a dedicated Qdrant collection ``civil_law_docs`` (separate from the
  shared ``judicial_docs`` collection) to avoid filter collisions with the
  Case Doc RAG.  All civil law chunks have ``source="civil_law"`` metadata.
- Creates payload indexes on retrieval-time filter fields so Qdrant can
  short-circuit full-scan with an indexed lookup.
- Supports sparse BM25 vectors alongside dense vectors for hybrid search.
- Singleton pattern with double-checked locking (thread-safe).
"""

from __future__ import annotations

import threading
from typing import Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from RAG.civil_law_rag.config import cfg
from RAG.civil_law_rag.retrieval.embeddings import get_client as get_embeddings
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

COLLECTION_NAME = "civil_law_docs"

_vectorstore_instance: Optional[QdrantVectorStore] = None
_qdrant_client_instance: Optional[QdrantClient] = None
_vs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _QdrantVSNoValidation(QdrantVectorStore):
    """Skip the per-init embedding validation (causes crash on Windows with
    background threads alive alongside PyTorch)."""

    def _validate_collection_config(self, *args, **kwargs):
        pass


def _make_qdrant_client() -> QdrantClient:
    global _qdrant_client_instance
    if _qdrant_client_instance is not None:
        return _qdrant_client_instance
    qcfg = cfg.qdrant
    _qdrant_client_instance = QdrantClient(
        host=qcfg.get("host", "localhost"),
        port=qcfg.get("port", 6333),
        grpc_port=qcfg.get("grpc_port", 6334),
        prefer_grpc=qcfg.get("prefer_grpc", True),
        check_compatibility=False,
    )
    return _qdrant_client_instance


def get_qdrant_client() -> QdrantClient:
    """Public accessor for the shared Qdrant client."""
    return _make_qdrant_client()


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """Create the civil_law_docs collection if it doesn't exist, then
    create payload indexes for fast filtered queries."""
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        log_event(logger, "collection_created", collection=COLLECTION_NAME)

    # Payload indexes — idempotent, safe to call on existing collection.
    # These turn O(N) full-scan filters into O(log N) indexed lookups.
    _create_payload_index_if_missing(client, COLLECTION_NAME, "metadata.type",    "keyword")
    _create_payload_index_if_missing(client, COLLECTION_NAME, "metadata.index",   "integer")
    _create_payload_index_if_missing(client, COLLECTION_NAME, "metadata.source",  "keyword")
    _create_payload_index_if_missing(client, COLLECTION_NAME, "metadata.chapter", "keyword")
    _create_payload_index_if_missing(client, COLLECTION_NAME, "metadata.section", "keyword")


def _create_payload_index_if_missing(
    client: QdrantClient,
    collection: str,
    field: str,
    schema_type: str,
) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema_type,
        )
        log_event(logger, "payload_index_created", collection=collection, field=field)
    except Exception as exc:
        # Index already exists → Qdrant raises; ignore silently.
        if "already exists" not in str(exc).lower():
            log_event(
                logger, "payload_index_warning",
                field=field, error=str(exc),
                level=20,  # WARNING
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vectorstore() -> QdrantVectorStore:
    """Return the shared QdrantVectorStore instance, creating it once."""
    global _vectorstore_instance
    if _vectorstore_instance is not None:
        return _vectorstore_instance

    with _vs_lock:
        if _vectorstore_instance is not None:
            return _vectorstore_instance

        embeddings = get_embeddings()
        client = _make_qdrant_client()
        vector_size = cfg.qdrant.get("vector_size", 1024)

        _ensure_collection(client, vector_size)

        _vectorstore_instance = _QdrantVSNoValidation(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        log_event(logger, "vectorstore_ready", collection=COLLECTION_NAME)

    return _vectorstore_instance


def civil_law_filter(extra_conditions: list | None = None) -> Filter:
    """Return a Qdrant filter that restricts to civil law articles.

    Always includes ``source=civil_law`` so civil law queries never
    accidentally match case-doc or OCR chunks in the same collection.
    """
    must = [
        FieldCondition(
            key="metadata.source",
            match=MatchValue(value="civil_law"),
        )
    ]
    if extra_conditions:
        must.extend(extra_conditions)
    return Filter(must=must)
