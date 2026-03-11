"""
vectorstore.py

Vector database initialization and embedding configuration module.

Purpose:
---------
Provides:
- Embedding model initialization
- Vectorstore (Chroma) loading logic

This module abstracts away:
- Embedding model selection
- Persistent database directory management

Why this exists:
----------------
To prevent vectorstore logic from being mixed with indexing or
runtime AI logic.

All components requiring a retriever should call `load_vectorstore()`
instead of manually initializing Chroma.

Design Principle:
-----------------
Centralized infrastructure layer.
Embedding configuration must exist in a single location to prevent
inconsistency across the system.
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, DB_DIR


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def load_vectorstore():
    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
