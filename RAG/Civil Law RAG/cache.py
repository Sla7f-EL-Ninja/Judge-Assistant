"""
cache.py

Semantic response cache for the Civil Law RAG pipeline.

P3-7: Caches previous (query, answer) pairs and returns a cached answer
when a new query is semantically close enough (cosine similarity above
threshold).  This avoids redundant LLM calls for near-duplicate questions.

The cache is:
- In-memory (lost on restart -- intentional for a legal system where
  freshness of index matters)
- Lazy -- only loads the embedding model on first ``get()`` call
- Thread-safe
"""

import logging
import threading

import numpy as np

from config.rag import CACHE_SIMILARITY_THRESHOLD, MAX_CACHE_SIZE

logger = logging.getLogger("civil_law_rag.cache")


class SemanticCache:
    """In-memory semantic similarity cache for RAG answers."""

    def __init__(
        self,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
        max_size: int = MAX_CACHE_SIZE,
    ):
        self._threshold = similarity_threshold
        self._max_size = max_size
        self._entries: list[tuple[np.ndarray, str]] = []  # (embedding, answer)
        self._lock = threading.Lock()
        self._embeddings = None  # lazy-loaded

    def _get_embeddings(self):
        """Lazy-load the embedding model on first use."""
        if self._embeddings is None:
            from vectorstore import get_embeddings
            self._embeddings = get_embeddings()
        return self._embeddings

    def _embed(self, text: str) -> np.ndarray:
        model = self._get_embeddings()
        vec = model.embed_query(text)
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def get(self, query: str) -> str | None:
        """Return a cached answer if the query is semantically similar enough."""
        if not self._entries:
            return None

        query_vec = self._embed(query)

        with self._lock:
            best_score = -1.0
            best_answer = None
            for cached_vec, cached_answer in self._entries:
                score = float(np.dot(query_vec, cached_vec))
                if score > best_score:
                    best_score = score
                    best_answer = cached_answer

            if best_score >= self._threshold:
                logger.info(
                    "Cache HIT (score=%.4f, threshold=%.4f)",
                    best_score,
                    self._threshold,
                )
                return best_answer

        logger.debug("Cache MISS (best_score=%.4f)", best_score)
        return None

    def set(self, query: str, answer: str) -> None:
        """Store a (query, answer) pair in the cache."""
        query_vec = self._embed(query)

        with self._lock:
            if len(self._entries) >= self._max_size:
                # Evict oldest entry (FIFO)
                self._entries.pop(0)
            self._entries.append((query_vec, answer))

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._entries.clear()
        logger.info("Semantic cache cleared")
