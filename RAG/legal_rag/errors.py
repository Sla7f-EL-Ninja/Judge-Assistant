"""
errors.py
---------
Typed exceptions for the legal_rag pipeline.
"""


class LegalRAGError(Exception):
    """Base class for all legal_rag errors."""


class QueryValidationError(LegalRAGError):
    """Query failed input validation (length, Arabic ratio, etc.)."""


class RetrievalError(LegalRAGError):
    """Qdrant or embedding service unavailable / returned no results."""


class GenerationError(LegalRAGError):
    """LLM call failed or returned unparseable output."""


class LLMBudgetExceededError(LegalRAGError):
    """Query hit the per-request LLM call budget (MAX_LLM_CALLS)."""


class LLMTimeoutError(LegalRAGError):
    """An individual LLM call timed out."""
