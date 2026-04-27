"""
cache.py
--------
Versioned semantic response cache for the Civil Law RAG pipeline.

Cache key = sha256(normalized_query + prompts_version + corpus_version + llm_model)
so that a change to prompts, the corpus, or the LLM automatically
invalidates all cached answers.

The cache is in-memory (lost on restart) — intentional for a legal
system where staleness carries risk.  TTL-based eviction is not
implemented; entries are evicted FIFO when max_size is reached.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import List, Optional

import numpy as np

from RAG.civil_law_rag.prompts import PROMPTS_VERSION

CACHE_SIMILARITY_THRESHOLD: float = 0.97
MAX_CACHE_SIZE: int = 500
CORPUS_VERSION: str = "1.0.0"
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)


def _cache_key_prefix(query: str, llm_model: str = "") -> str:
    """Return a deterministic prefix from versioned context."""
    raw = f"{query}|{PROMPTS_VERSION}|{CORPUS_VERSION}|{llm_model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class SemanticCache:
    """In-memory semantic similarity cache with versioned keys."""

    def __init__(
        self,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_size: int = MAX_CACHE_SIZE,
    ) -> None:
        self._threshold  = similarity_threshold
        self._max_size   = max_size
        # list of (key_prefix: str, embedding: np.ndarray, answer: str)
        self._entries: List[tuple] = []
        self._lock       = threading.Lock()
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            from RAG.civil_law_rag.retrieval.embeddings import get_client
            self._embeddings = get_client()
        return self._embeddings

    def _embed(self, text: str) -> np.ndarray:
        model = self._get_embeddings()
        vec   = model.embed_query(text)
        arr   = np.array(vec, dtype=np.float32)
        norm  = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def get(self, query: str, llm_model: str = "") -> Optional[tuple]:
        """Return a cached (answer, sources) tuple or None."""
        if not self._entries:
            return None

        key_prefix = _cache_key_prefix(query, llm_model)
        query_vec  = self._embed(query)

        with self._lock:
            best_score   = -1.0
            best_answer  = None
            best_sources = []
            for entry_key, cached_vec, cached_answer, cached_sources in self._entries:
                # Only compare entries with matching version prefix
                if entry_key != key_prefix:
                    continue
                score = float(np.dot(query_vec, cached_vec))
                if score > best_score:
                    best_score   = score
                    best_answer  = cached_answer
                    best_sources = cached_sources

            if best_score >= self._threshold:
                log_event(logger, "cache_hit",
                        score=round(best_score, 4),
                        threshold=self._threshold)
                return best_answer, best_sources

        log_event(logger, "cache_miss",
                best_score=round(best_score, 4),
                level=logging.DEBUG)
        return None

    def set(self, query: str, answer: str, sources: list = None, llm_model: str = "") -> None:
        """Store a (query, answer, sources) entry."""
        key_prefix = _cache_key_prefix(query, llm_model)
        query_vec  = self._embed(query)

        with self._lock:
            if len(self._entries) >= self._max_size:
                self._entries.pop(0)  # FIFO eviction
            self._entries.append((key_prefix, query_vec, answer, sources or []))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        log_event(logger, "cache_cleared")