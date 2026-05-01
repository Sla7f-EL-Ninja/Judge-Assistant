"""
civil_law_rag
=============
Egyptian Civil Law RAG agent — production package.

Public API:
    ask_question(query: str) -> str
    build_graph() -> CompiledGraph
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from RAG.civil_law_rag.service import ask_question  # noqa: F401
from RAG.civil_law_rag.graph import build_graph  # noqa: F401

__all__ = ["ask_question", "build_graph"]
