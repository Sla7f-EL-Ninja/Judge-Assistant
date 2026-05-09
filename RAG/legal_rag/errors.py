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


class PreprocessingError(LegalRAGError):
    """preprocessor_node failed to rewrite or classify the query."""


class CorpusRoutingError(LegalRAGError):
    """corpus_classifier_node failed to determine the target corpus."""


class ScopeClassificationError(LegalRAGError):
    """scope_classifier_node failed to identify chapter/section scope."""


class InternalRAGError(LegalRAGError):
    """Unrecoverable internal error — maps to HTTP 500 at the service layer."""
