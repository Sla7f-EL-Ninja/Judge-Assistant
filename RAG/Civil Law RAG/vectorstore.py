"""
vectorstore.py

Vector database initialization and embedding configuration module.

Purpose:
---------
Provides:
- Embedding model initialization
- Vectorstore (Qdrant) loading logic

This module abstracts away:
- Embedding model selection
- Vector database connection management

Why this exists:
----------------
To prevent vectorstore logic from being mixed with indexing or
runtime AI logic.

All components requiring a retriever should call `load_vectorstore()`
instead of manually initializing Qdrant.

Design Principle:
-----------------
Centralized infrastructure layer.
Embedding configuration must exist in a single location to prevent
inconsistency across the system.

Migration Note:
---------------
Replaced ChromaDB (embedded SQLite) with Qdrant (client-server) for
production readiness: replication, metadata filtering indexes, access
control, and horizontal scaling.
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config.rag import EMBEDDING_MODEL
from config import cfg


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def _get_qdrant_client():
    """Create a Qdrant client from centralized config."""
    qdrant_cfg = cfg.qdrant
    return QdrantClient(
        host=qdrant_cfg.get("host", "localhost"),
        port=qdrant_cfg.get("port", 6333),
        grpc_port=qdrant_cfg.get("grpc_port", 6334),
        prefer_grpc=qdrant_cfg.get("prefer_grpc", True),
    )


def load_vectorstore():
    embeddings = get_embeddings()
    client = _get_qdrant_client()
    collection_name = cfg.qdrant.get("collection", "judicial_docs")
    vector_size = cfg.qdrant.get("vector_size", 1024)  # BAAI/bge-m3 outputs 1024

    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        from qdrant_client.models import Distance, VectorParams
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
