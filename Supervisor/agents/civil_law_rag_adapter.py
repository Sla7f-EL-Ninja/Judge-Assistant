"""
civil_law_rag_adapter.py
------------------------
Supervisor adapter for the Civil Law RAG agent.

Routes all calls through RAG.civil_law_rag.service.ask_question so that
input validation, semantic caching, and typed error handling always apply.

No sys.path manipulation, no sys.modules eviction, no per-call imports.
The service module is imported at class instantiation via a lazy import
to avoid circular dependencies at startup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


class CivilLawRAGAdapter(AgentAdapter):
    """Thin adapter that delegates to the Civil Law RAG service layer."""

    # Part 3.3a: service.py wraps all graph errors in a generic except and
    # returns them as an Arabic error string inside a valid CivilLawResult.
    # These prefixes mark answers that are actually service error messages.
    _SERVICE_ERROR_PREFIXES = (
        "حدث خطأ",      # "An error occurred"
        "تعذّر",         # "Failed to"
        "تعذر",          # variant without shadda
        "لم يتمكن",      # "Could not"
        "خطأ في",        # "Error in"
    )

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            from RAG.civil_law_rag.service import ask_question, CivilLawResult

            result: CivilLawResult = ask_question(query)

            # Part 3.3a: detect service-swallowed errors returned as Arabic answer strings.
            # from_cache=False guards against legitimate cached error-adjacent answers.
            if not result.from_cache and any(
                result.answer.startswith(p) for p in self._SERVICE_ERROR_PREFIXES
            ):
                logger.warning(
                    "Civil Law RAG service returned error string as answer (swallowed exception): %s",
                    result.answer[:200],
                )
                return AgentResult(
                    response="",
                    error=f"Civil Law RAG service error: {result.answer[:200]}",
                )

            # Format sources as strings for AgentResult (list[str] contract)
            sources = [
                f"المادة {s['article']}"
                + (f" — {s['title']}" if s.get("title") else "")
                for s in result.sources
            ]

            return AgentResult(
                response=result.answer,
                sources=sources,
                raw_output={
                    "classification":        result.classification,
                    "retrieval_confidence":  result.retrieval_confidence,
                    "citation_integrity":    result.citation_integrity,
                    "from_cache":            result.from_cache,
                },
            )

        except Exception as exc:
            error_msg = f"Civil Law RAG adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
