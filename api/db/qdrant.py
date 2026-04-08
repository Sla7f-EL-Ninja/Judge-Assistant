"""
qdrant.py

Qdrant vector store client management.

Provides async-compatible connection lifecycle for the Qdrant vector
database, used for civil law article embeddings and case document
embeddings (replaces ChromaDB).

``connect_qdrant`` / ``close_qdrant`` are called from the FastAPI lifespan.
``get_qdrant_client`` returns the active client for dependency injection.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.api import Settings

logger = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None


def connect_qdrant(settings: Settings) -> None:
    """Open the Qdrant client connection and ensure collections exist."""
    global _client
    _client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )

    # Ensure the main collection exists
    _ensure_collection(
        client=_client,
        collection_name=settings.qdrant_collection,
        vector_size=settings.qdrant_vector_size,
    )

    logger.info(
        "Qdrant connected at %s:%s (collection=%s)",
        settings.qdrant_host,
        settings.qdrant_port,
        settings.qdrant_collection,
    )


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Create the Qdrant collection if it does not exist."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d)", collection_name, vector_size)

        # Create payload indexes for filtered search
        client.create_payload_index(
            collection_name=collection_name,
            field_name="case_id",
            field_schema="keyword",
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_type",
            field_schema="keyword",
        )
        logger.info("Created payload indexes on 'case_id' and 'doc_type'")
    else:
        logger.info("Qdrant collection '%s' already exists", collection_name)


def close_qdrant() -> None:
    """Close the Qdrant client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def get_qdrant_client() -> QdrantClient:
    """Return the active Qdrant client. Raises if not connected."""
    if _client is None:
        raise RuntimeError("Qdrant is not connected. Call connect_qdrant first.")
    return _client
