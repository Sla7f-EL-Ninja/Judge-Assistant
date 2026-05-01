"""
civil_law_rag
=============
Thin wrapper around the legal_rag engine for the Egyptian Civil Law corpus.

Public API is identical to the old civil_law_rag package so all existing
callers (Supervisor adapter, tests, api/app.py) need zero changes.

    ask_question(query: str) -> LegalRAGResult
    build_graph()            -> CompiledGraph
    ensure_indexed()         -> None
    CIVIL_LAW_CORPUS         -> CorpusConfig
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS  # noqa: F401

from RAG.legal_rag.service  import ask_question  as _ask
from RAG.legal_rag.graph    import build_graph   as _build
from RAG.legal_rag.indexing.indexer import ensure_indexed as _ensure


def ask_question(query: str):
    """Ask a question against the Civil Law corpus."""
    return _ask(query, CIVIL_LAW_CORPUS)


def build_graph():
    """Build/return the compiled Civil Law RAG graph."""
    return _build(CIVIL_LAW_CORPUS)


def ensure_indexed():
    """Ensure the Civil Law corpus is indexed in Qdrant."""
    return _ensure(CIVIL_LAW_CORPUS)


__all__ = ["ask_question", "build_graph", "ensure_indexed", "CIVIL_LAW_CORPUS"]
