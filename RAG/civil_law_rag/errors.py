"""
errors.py
---------
Typed exceptions for the Civil Law RAG pipeline.

Using typed exceptions (rather than bare strings) lets the Supervisor
classify failures and respond appropriately — e.g. distinguish a
transient retrieval timeout from a permanent validation error.
"""


class CivilLawRAGError(Exception):
    """Base class for all Civil Law RAG errors."""


class QueryValidationError(CivilLawRAGError):
    """Query failed input validation (length, Arabic ratio, etc.)."""


class RetrievalError(CivilLawRAGError):
    """Qdrant or embedding service unavailable / returned no results."""


class GenerationError(CivilLawRAGError):
    """LLM call failed or returned unparseable output."""


class LLMBudgetExceededError(CivilLawRAGError):
    """Query hit the per-request LLM call budget (MAX_LLM_CALLS)."""


class LLMTimeoutError(CivilLawRAGError):
    """An individual LLM call timed out."""
