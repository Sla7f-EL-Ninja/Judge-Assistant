"""
procedures_rag
============
Thin wrapper around the legal_rag engine for the Egyptian Law of Procedures
in Civil and Commercial Matters corpus.

Public API:

    ask_question(query: str) -> LegalRAGResult
    build_graph()            -> CompiledGraph
    ensure_indexed()         -> None
    PROCEDURES_CORPUS        -> CorpusConfig
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from RAG.legal_rag.procedures_rag.corpus import PROCEDURES_CORPUS  # noqa: F401

from RAG.legal_rag.service  import ask_question  as _ask
from RAG.legal_rag.graph    import build_graph   as _build
from RAG.legal_rag.indexing.indexer import ensure_indexed as _ensure


def ask_question(query: str):
    """Ask a question against the Procedures Law corpus."""
    return _ask(query, PROCEDURES_CORPUS)


def build_graph():
    """Build/return the compiled Procedures Law RAG graph."""
    return _build(PROCEDURES_CORPUS)


def ensure_indexed():
    """Ensure the Procedures Law corpus is indexed in Qdrant."""
    return _ensure(PROCEDURES_CORPUS)


__all__ = ["ask_question", "build_graph", "ensure_indexed", "PROCEDURES_CORPUS"]
