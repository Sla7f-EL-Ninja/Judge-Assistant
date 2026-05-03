"""
civil_law_rag
=============
Thin wrapper around the legal_rag engine for the Egyptian Civil Law corpus.
 
    ask_question(query: str) -> LegalRAGResult
    ensure_indexed()         -> None
    CIVIL_LAW_CORPUS         -> CorpusConfig
 
Note: build_graph() is no longer exposed here — there is one unified graph
shared across all corpora.  Import it directly if needed:
    from RAG.legal_rag.graph import build_graph
"""
 
import os
import sys
 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
 
from RAG.legal_rag.civil_law_rag.corpus import CIVIL_LAW_CORPUS  # noqa: F401
from RAG.legal_rag.service import ask_question                    # noqa: F401
from RAG.legal_rag.indexing.indexer import ensure_indexed as _ensure
 
 
def ensure_indexed():
    """Ensure the Civil Law corpus is indexed in Qdrant."""
    return _ensure(CIVIL_LAW_CORPUS)
 
 
__all__ = ["ask_question", "ensure_indexed", "CIVIL_LAW_CORPUS"]