"""
config.py
---------
Shared engine constants for the legal_rag engine.

Corpus-specific values (docs_path, collection_name, source_filter_value,
law_display_name, corpus_version) have moved to CorpusConfig in each
corpus's own corpus.py.  What remains here are tuning knobs that apply
to ALL legal corpora served by this engine.
"""

from __future__ import annotations

import os

from config import cfg, get_llm  # project-level config package

__all__ = [
    "cfg",
    "get_llm",
    "EMBEDDING_MODEL",
    "BATCH_SIZE",
    "LLM_MODEL",
    "MAX_QUERY_LENGTH",
    "MIN_QUERY_LENGTH",
    "MIN_ARABIC_RATIO",
    "CACHE_SIMILARITY_THRESHOLD",
    "MAX_CACHE_SIZE",
    "MAX_LLM_CALLS",
    "LLM_TIMEOUT",
    "TEI_EMBEDDING_URL",
    "TEI_RERANKER_URL",
]

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
# Production safety
# ---------------------------------------------------------------------------
MAX_LLM_CALLS: int = 5          # max LLM invocations per single query
LLM_TIMEOUT: int = 30           # seconds before LangChain call times out

# ---------------------------------------------------------------------------
# Remote TEI service URLs (override via env vars)
# ---------------------------------------------------------------------------
TEI_EMBEDDING_URL: str = os.environ.get(
    "JA_TEI_EMBEDDING_URL",
    cfg.tei.get("embedding_url", "http://localhost:8080"),
)
TEI_RERANKER_URL: str = os.environ.get(
    "JA_TEI_RERANKER_URL",
    cfg.tei.get("reranker_url", "http://localhost:8081"),
)
