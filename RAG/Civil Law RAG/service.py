"""
service.py

Public service interface for the Civil Law RAG pipeline.

P3-4: Extracted from ``main.py`` to provide a clean, production-ready
entry point that:
- Validates input (P3-1)
- Checks the semantic cache (P3-7)
- Invokes the LangGraph workflow
- Handles errors gracefully (P3-6)
"""

import logging
import re
import traceback

from config.rag import (
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    MIN_ARABIC_RATIO,
    get_default_state,
)
from cache import SemanticCache

logger = logging.getLogger("civil_law_rag.service")

# Module-level semantic cache instance
_cache = SemanticCache()


class QueryValidationError(ValueError):
    """Raised when a query fails input validation."""
    pass


def validate_query(query: str) -> str:
    """Validate and sanitize a user query before pipeline execution.

    P3-1: Enforces length limits and minimum Arabic content ratio.

    Returns the stripped query on success.
    Raises ``QueryValidationError`` on failure.
    """
    if not query or not isinstance(query, str):
        raise QueryValidationError("الاستعلام فارغ أو غير صالح.")

    query = query.strip()

    if len(query) < MIN_QUERY_LENGTH:
        raise QueryValidationError(
            f"الاستعلام قصير جدًا (الحد الأدنى {MIN_QUERY_LENGTH} أحرف)."
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise QueryValidationError(
            f"الاستعلام طويل جدًا (الحد الأقصى {MAX_QUERY_LENGTH} حرف)."
        )

    # Check minimum Arabic character ratio
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", query))
    total_chars = len(query.replace(" ", ""))
    if total_chars > 0:
        ratio = arabic_chars / total_chars
        if ratio < MIN_ARABIC_RATIO:
            raise QueryValidationError(
                "الاستعلام لا يحتوي على نسبة كافية من النص العربي."
            )

    return query


def ask_question(query: str) -> str:
    """Process a user query through the Civil Law RAG pipeline.

    P3-4: Single entry point that handles validation, caching, graph
    invocation, and error handling.

    Args:
        query: The user's question in Arabic.

    Returns:
        The final answer string. On error, returns a fixed Arabic
        error message rather than raising.
    """
    # P3-1: Validate input
    try:
        query = validate_query(query)
    except QueryValidationError as exc:
        logger.warning("Query validation failed: %s", exc)
        return str(exc)

    # P3-7: Check semantic cache
    cached = _cache.get(query)
    if cached is not None:
        return cached

    # P3-6: Wrap graph invocation in try/except
    try:
        # Lazy import to avoid circular dependency at module load time
        from graph import app

        state = get_default_state()
        state["last_query"] = query

        result_state = app.invoke(state)
        answer = result_state.get("final_answer", "تعذر الحصول على إجابة.")

        # P3-7: Cache the result
        if answer:
            _cache.set(query, answer)

        return answer

    except Exception:
        logger.error(
            "Graph invocation failed for query=%r\n%s",
            query[:200],
            traceback.format_exc(),
        )
        return (
            "حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى لاحقًا."
        )


def clear_cache() -> None:
    """Clear the semantic response cache."""
    _cache.clear()
