"""
config.py
---------
Single source of truth for all Civil Law RAG constants.

Imports from the project-level ``config`` package via the normal
package path (RAG/civil_law_rag is a proper sub-package of the project,
no sys.path manipulation needed).
"""

from __future__ import annotations

import os
from pathlib import Path

from config import cfg, get_llm  # project-level config package

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
    "PROMPTS_VERSION",
    "CORPUS_VERSION",
    "MAX_LLM_CALLS",
    "LLM_TIMEOUT",
    "TEI_EMBEDDING_URL",
    "TEI_RERANKER_URL",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parent            # RAG/civil_law_rag/
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent  # <project_root>/

DOCS_PATH: str = str(
    _PROJECT_ROOT / "RAG" / "civil_law_rag" / "docs" / "civil_law.txt"
)

# ---------------------------------------------------------------------------
# Embedding / LLM constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = cfg.embedding.get("model", "BAAI/bge-m3")
BATCH_SIZE: int = 50
LLM_MODEL: str = cfg.llm.get("high", {}).get("model", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH: int = 2000
MIN_QUERY_LENGTH: int = 3
MIN_ARABIC_RATIO: float = 0.3

# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------
CACHE_SIMILARITY_THRESHOLD: float = 0.97
MAX_CACHE_SIZE: int = 500

# ---------------------------------------------------------------------------
# Versioning (bump when prompts or corpus changes so cache auto-invalidates)
# ---------------------------------------------------------------------------
PROMPTS_VERSION: str = "1.2.0"
CORPUS_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Production safety
# ---------------------------------------------------------------------------
MAX_LLM_CALLS: int = 5          # max LLM invocations per single query
LLM_TIMEOUT: int = 30           # seconds before LangChain call times out

# ---------------------------------------------------------------------------
# Remote TEI service URLs (override via JA_TEI_EMBEDDING_URL / JA_TEI_RERANKER_URL)
# ---------------------------------------------------------------------------
TEI_EMBEDDING_URL: str = os.environ.get(
    "JA_TEI_EMBEDDING_URL",
    cfg.tei.get("embedding_url", "http://localhost:8080"),
)
TEI_RERANKER_URL: str = os.environ.get(
    "JA_TEI_RERANKER_URL",
    cfg.tei.get("reranker_url", "http://localhost:8081"),
)
