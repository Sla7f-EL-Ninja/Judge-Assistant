"""
case_doc_rag_adapter.py

Adapter for the Case Document RAG agent (RAG/case_doc_rag/).
"""

import logging
import uuid
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


class CaseDocRAGAdapter(AgentAdapter):
    """Thin wrapper around the Case Doc RAG LangGraph workflow."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Query the Case Doc RAG about specific case documents.

        Parameters
        ----------
        query:
            The (rewritten) judge query about case documents.
        context:
            Should contain ``case_id`` and ``conversation_history``.
        """
        try:
            from RAG.case_doc_rag.graph import build_graph

            case_id = context.get("case_id", "")
            conversation_history = context.get("conversation_history", [])

            initial_state = {
                "query": query,
                "case_id": case_id,
                "conversation_history": conversation_history,
                "request_id": str(uuid.uuid4()),
                # processing fields — use feature defaults
                "sub_questions": [],
                "on_topic": True,
                "doc_selection_mode": "no_doc_specified",
                "selected_doc_id": None,
                "doc_titles": [],
                "sub_answers": [],
                "final_answer": "",
                "error": None,
            }

            app = build_graph()
            result = app.invoke(initial_state)

            error = result.get("error")
            if error:
                return AgentResult(response="", error=str(error))

            final_answer = result.get("final_answer", "")

            # P1.4.3: empty answer with no explicit error is a silent failure —
            # surface it so the supervisor can retry or fall back.
            if not final_answer:
                on_topic = result.get("on_topic", True)
                if not on_topic:
                    return AgentResult(
                        response="",
                        error="Case Doc RAG: query classified as off-topic for case documents",
                    )
                return AgentResult(
                    response="",
                    error="Case Doc RAG: no answer produced (empty result)",
                )

            # Sources live inside each sub_answer entry
            sources = []
            for sub in result.get("sub_answers", []):
                for src in sub.get("sources", []):
                    if src:
                        s = str(src)
                        if s not in sources:
                            sources.append(s)

            return AgentResult(
                response=final_answer,
                sources=sources,
                raw_output={
                    "final_answer": final_answer,
                    "sub_answers": result.get("sub_answers", []),
                    "doc_selection_mode": result.get("doc_selection_mode"),
                    "selected_doc_id": result.get("selected_doc_id"),
                },
            )

        except Exception as exc:
            error_msg = f"Case Doc RAG adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
