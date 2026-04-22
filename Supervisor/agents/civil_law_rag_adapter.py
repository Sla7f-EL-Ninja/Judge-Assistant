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

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            from RAG.civil_law_rag.service import ask_question, CivilLawResult

            result: CivilLawResult = ask_question(query)

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
