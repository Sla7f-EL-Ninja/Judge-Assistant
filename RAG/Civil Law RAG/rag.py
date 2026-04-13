# """
# config.rag
# ----------
# Civil Law RAG constants and default state template.

# Moved from ``RAG/Civil Law RAG/config.py`` during config consolidation.
# All values are sourced from the centralized ``config`` module.
# """

# import os

# from config import cfg

# # -----------------------------
# # Paths -- resolve relative to project root
# # -----------------------------
# _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DOCS_PATH = os.path.join(_PROJECT_ROOT, "RAG", "Civil Law RAG", "docs", "civil_law_clean.txt")

# # -----------------------------
# # Initializations -- from central config
# # -----------------------------
# EMBEDDING_MODEL = cfg.embedding.get("model", "BAAI/bge-m3")
# BATCH_SIZE = 50
# LLM_MODEL = cfg.llm.get("high", {}).get("model", "gemini-2.5-flash")

# # -----------------------------
# # P1-5: Input validation constants
# # -----------------------------
# MAX_QUERY_LENGTH = 2000
# MIN_QUERY_LENGTH = 3
# MIN_ARABIC_RATIO = 0.3

# # -----------------------------
# # P3-7: Semantic cache settings
# # -----------------------------
# CACHE_SIMILARITY_THRESHOLD = 0.97
# MAX_CACHE_SIZE = 500

# # -----------------------------
# # Default State Template
# # -----------------------------
# # P1-5: kept as a plain dict for LangGraph TypedDict compatibility.
# # Use ``get_default_state()`` which validates via the Pydantic RAGState model.
# default_state_template = {
#     "last_query": None,
#     "last_results": [],
#     "last_answer": None,
#     "current_book": None,
#     "current_part": None,
#     "current_chapter": None,
#     "current_article": None,
#     "filter_type": "",
#     "k": 8,
#     "books_in_scope": [],
#     "query_history": [],
#     "retrieval_history": [],
#     "retry_count": 0,
#     "max_retries": 2,
#     "answer_history": [],
#     "db_initialized": True,
#     "db": None,
#     "split_config": {},
#     "rewritten_question": None,
#     "classification": None,
#     "retrieval_confidence": None,
#     "refined_query": None,
#     "grade": None,
#     "llm_pass": None,
#     "failure_reason": None,
#     "proceedToGenerate": None,
#     "retrieval_attempts": 0,
#     "final_answer": None,
# }


# def get_default_state() -> dict:
#     """Return a validated copy of the default state template.

#     P1-5: Uses the Pydantic RAGState model for validation, then
#     returns a plain dict for LangGraph compatibility.
#     """
#     # Import here to avoid circular dependency (nodes.py imports from config.rag)
#     # The validation is a development-time safety net, not a hot path.
#     try:
#         import sys, os
#         _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         _rag_dir = os.path.join(_project_root, "RAG", "Civil Law RAG")
#         if _rag_dir not in sys.path:
#             sys.path.insert(0, _rag_dir)
#         from nodes import RAGState
#         validated = RAGState(**default_state_template)
#         return validated.model_dump()
#     except Exception:
#         # Fallback: return a plain copy if validation module is not available
#         return default_state_template.copy()


# # -----------------------------
# # Graph Constants
# # -----------------------------
# START = "__start__"
# END = "__end__"


"""
rag.py
------
Civil Law RAG constants, path bootstrap, and default state template.

This module is the **single source of truth** for all RAG-level
configuration.  It also re-exports ``cfg`` and ``get_llm`` from the
project-level ``config`` package so that every sibling module (indexer,
vectorstore, nodes, …) can do:

    from rag import cfg, get_llm, DOCS_PATH, ...

and never need to touch the local ``config.py`` shim (which has been
deleted because it caused a circular-import crash).

Path bootstrap
--------------
This file lives at:
    <project_root>/RAG/Civil Law RAG/rag.py

The project-level ``config`` package lives at:
    <project_root>/config/

We resolve <project_root> at import time using ``pathlib`` (works on
every OS and drive letter) and insert it at the *front* of ``sys.path``
so that ``import config`` always finds the real package, not this file's
own directory.

The insertion is idempotent -- we check before inserting.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Robust, OS-agnostic path bootstrap
#    __file__ → .../RAG/Civil Law RAG/rag.py
#    .parent   → .../RAG/Civil Law RAG/
#    .parent   → .../RAG/
#    .parent   → <project_root>/
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_CIVIL_LAW_RAG_DIR = _THIS_FILE.parent          # RAG/Civil Law RAG/
_PROJECT_ROOT = _CIVIL_LAW_RAG_DIR.parent.parent  # <project_root>/

# Ensure project root is on sys.path so `import config` finds the real package
_project_root_str = str(_PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# ---------------------------------------------------------------------------
# 2. Import from the real project-level config package
#    This is now guaranteed to work because _PROJECT_ROOT is on sys.path
#    and there is no longer a local config.py shim shadowing it.
# ---------------------------------------------------------------------------
from config import cfg, get_llm  # noqa: E402  (must come after sys.path fix)

# Re-export so sibling modules can do `from rag import cfg, get_llm`
__all__ = [
    "cfg",
    "get_llm",
    "DOCS_PATH",
    "EMBEDDING_MODEL",
    "BATCH_SIZE",
    "LLM_MODEL",
    "MAX_QUERY_LENGTH",
    "MIN_QUERY_LENGTH",
    "MIN_ARABIC_RATIO",
    "CACHE_SIMILARITY_THRESHOLD",
    "MAX_CACHE_SIZE",
    "default_state_template",
    "get_default_state",
    "START",
    "END",
]

# ---------------------------------------------------------------------------
# 3. Paths -- always resolved relative to project root, never hard-coded
# ---------------------------------------------------------------------------
DOCS_PATH: str = str(
    _PROJECT_ROOT / "RAG" / "Civil Law RAG" / "docs" / "civil_law_clean.txt"
)

# ---------------------------------------------------------------------------
# 4. RAG-level constants sourced from centralized config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = cfg.embedding.get("model", "BAAI/bge-m3")
BATCH_SIZE: int = 50
LLM_MODEL: str = cfg.llm.get("high", {}).get("model", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# 5. Input validation constants (P1-5)
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH: int = 2000
MIN_QUERY_LENGTH: int = 3
MIN_ARABIC_RATIO: float = 0.3

# ---------------------------------------------------------------------------
# 6. Semantic cache settings (P3-7)
# ---------------------------------------------------------------------------
CACHE_SIMILARITY_THRESHOLD: float = 0.97
MAX_CACHE_SIZE: int = 500

# ---------------------------------------------------------------------------
# 7. Default state template
#    Kept as a plain dict for LangGraph TypedDict compatibility.
#    Use get_default_state() which validates via the Pydantic RAGState model.
# ---------------------------------------------------------------------------
default_state_template: dict = {
    "last_query": None,
    "last_results": [],
    "last_answer": None,
    "current_book": None,
    "current_part": None,
    "current_chapter": None,
    "current_article": None,
    "filter_type": "",
    "k": 8,
    "books_in_scope": [],
    "query_history": [],
    "retrieval_history": [],
    "retry_count": 0,
    "max_retries": 2,
    "answer_history": [],
    "db_initialized": True,
    "db": None,
    "split_config": {},
    "rewritten_question": None,
    "classification": None,
    "retrieval_confidence": None,
    "refined_query": None,
    "grade": None,
    "llm_pass": None,
    "failure_reason": None,
    "proceedToGenerate": None,
    "retrieval_attempts": 0,
    "final_answer": None,
}


def get_default_state() -> dict:
    """Return a validated copy of the default state template.

    P1-5: Validates via the Pydantic RAGState model, then returns a plain
    dict for LangGraph compatibility.  Falls back to a raw copy if the
    nodes module is not yet importable (e.g. during early bootstrap).
    """
    try:
        # Late import to avoid a circular dependency:
        # nodes.py imports from rag, so we cannot import nodes at module level.
        _civil_law_rag_str = str(_CIVIL_LAW_RAG_DIR)
        if _civil_law_rag_str not in sys.path:
            sys.path.insert(0, _civil_law_rag_str)

        from nodes import RAGState  # noqa: PLC0415

        validated = RAGState(**default_state_template)
        return validated.model_dump()
    except Exception:
        # Safe fallback: return an unvalidated copy
        return default_state_template.copy()


# ---------------------------------------------------------------------------
# 8. Graph sentinel constants
# ---------------------------------------------------------------------------
START: str = "__start__"
END: str = "__end__"