"""
vectorstore.py
--------------
Qdrant vector store initialization for the legal_rag engine.

Design decisions:
- One Qdrant collection per legal corpus (collection_name comes from
  CorpusConfig).  Collections are independent so filters, payload
  indexes, and point counts never collide across corpora.
- load_vectorstore(collection_name) is keyed by collection_name with a
  per-key lock so two corpora can initialize concurrently without races.
- source_filter(source_value, extra_conditions) replaces the old
  civil_law_filter() helper — it is corpus-agnostic.
- Payload indexes are created idempotently on every cold start.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from RAG.legal_rag.config import cfg
from RAG.legal_rag.retrieval.embeddings import get_client as get_embeddings
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton stores
# ---------------------------------------------------------------------------
_qdrant_client_instance: Optional[QdrantClient] = None
_qdrant_client_lock = threading.Lock()

# Per-collection vectorstore singletons  {collection_name: QdrantVectorStore}
_vectorstore_instances: Dict[str, QdrantVectorStore] = {}
_vectorstore_locks: Dict[str, threading.Lock] = {}
_vs_registry_lock = threading.Lock()  # guards the dicts above


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _QdrantVSNoValidation(QdrantVectorStore):
    """Skip the per-init embedding validation (crashes on Windows with
    background threads alive alongside PyTorch)."""

    def _validate_collection_config(self, *args, **kwargs):
        pass


def _make_qdrant_client() -> QdrantClient:
    global _qdrant_client_instance
    if _qdrant_client_instance is not None:
        return _qdrant_client_instance
    with _qdrant_client_lock:
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


def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create collection if absent, then create payload indexes idempotently."""
    existing = {c.name for c in client.get_collections().collections}

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        log_event(logger, "collection_created", collection=collection_name)

    # Payload indexes — idempotent.
    for field, schema in [
        ("metadata.type",    "keyword"),
        ("metadata.index",   "integer"),
        ("metadata.source",  "keyword"),
        ("metadata.chapter", "keyword"),
        ("metadata.section", "keyword"),
    ]:
        _create_payload_index_if_missing(client, collection_name, field, schema)


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
        if "already exists" not in str(exc).lower():
            log_event(logger, "payload_index_warning",
                      field=field, error=str(exc), level=20)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vectorstore(collection_name: str) -> QdrantVectorStore:
    """Return the shared QdrantVectorStore for *collection_name*, creating it once.

    Thread-safe: two callers with the same collection_name will not
    double-initialize; two callers with different names may initialize
    concurrently.
    """
    if collection_name in _vectorstore_instances:
        return _vectorstore_instances[collection_name]

    # Get or create a per-collection lock
    with _vs_registry_lock:
        if collection_name not in _vectorstore_locks:
            _vectorstore_locks[collection_name] = threading.Lock()

    with _vectorstore_locks[collection_name]:
        if collection_name in _vectorstore_instances:
            return _vectorstore_instances[collection_name]

        embeddings  = get_embeddings()
        client      = _make_qdrant_client()
        vector_size = cfg.qdrant.get("vector_size", 1024)

        _ensure_collection(client, collection_name, vector_size)

        vs = _QdrantVSNoValidation(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        _vectorstore_instances[collection_name] = vs
        log_event(logger, "vectorstore_ready", collection=collection_name)

    return _vectorstore_instances[collection_name]


def source_filter(
    source_value: str,
    extra_conditions: Optional[List] = None,
) -> Filter:
    """Return a Qdrant Filter that restricts to one corpus source.

    Replaces the old civil_law_filter().  Always pins metadata.source
    so queries on collection X never accidentally match documents from a
    different corpus that was bulk-loaded into the same collection.

    Args:
        source_value:      Value of metadata.source, e.g. "civil_law" or
                           "evidence_law".  Comes from CorpusConfig.
        extra_conditions:  Additional FieldCondition objects (type, chapter…).
    """
    must = [
        FieldCondition(
            key="metadata.source",
            match=MatchValue(value=source_value),
        )
    ]
    if extra_conditions:
        must.extend(extra_conditions)
    return Filter(must=must)


def collection_point_count(collection_name: str) -> int:
    """Return the number of points in *collection_name*, 0 if absent."""
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception as exc:
        is_not_found = (
            getattr(exc, "status_code", None) == 404
            or "not found" in str(exc).lower()
        )
        if is_not_found:
            return 0
        raise
