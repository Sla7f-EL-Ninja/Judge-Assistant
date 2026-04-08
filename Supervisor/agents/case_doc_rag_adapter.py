"""
case_doc_rag_adapter.py

Adapter for the Case Document RAG agent (RAG/case_doc_rag/).

Wraps the refactored ``run()`` function and returns an AgentResult with the
answer extracted from case documents.
"""

import logging
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


def _get_shared_vectorstore():
    """Return the shared Qdrant vector store used by the FileIngestor.

    This ensures the Case Doc RAG reads from the *same* store that
    documents were indexed into, avoiding the empty-store problem that
    occurs when a separate Qdrant client instance is created.
    """
    from Supervisor.nodes.classify_and_store_document import _get_ingestor
    ingestor = _get_ingestor()
    return ingestor.vectorstore


class CaseDocRAGAdapter(AgentAdapter):
    """Thin wrapper around the Case Doc RAG LangGraph workflow."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Query the Case Doc RAG about specific case documents.

        Parameters
        ----------
        query:
            The (rewritten) judge query about case documents.
        context:
            Should contain ``case_id``.  May also contain
            ``conversation_history`` for multi-turn context.
        """
        try:
            from RAG.case_doc_rag import run, set_vectorstore

            # Inject the shared vector store so the pipeline queries the
            # same Qdrant instance that documents were indexed into.
            try:
                shared_vs = _get_shared_vectorstore()
                set_vectorstore(shared_vs)
                logger.info("Injected shared vectorstore into case_doc_rag")
            except Exception as exc:
                logger.warning(
                    "Could not inject shared vectorstore: %s. "
                    "case_doc_rag will use its own Qdrant instance.",
                    exc,
                )

            case_id = context.get("case_id", "")
            conversation_history = context.get("conversation_history", [])
            request_id = context.get("request_id")

            result = run(
                query=query,
                case_id=case_id,
                conversation_history=conversation_history,
                request_id=request_id,
            )

            # Check for errors
            error = result.get("error")
            if error:
                logger.warning("Case Doc RAG returned error: %s", error)

            # Extract the answer
            answer = result.get("final_answer", "")

            # For multi-question responses, compose from sub_answers
            if not answer and result.get("sub_answers"):
                parts = []
                for sa in result["sub_answers"]:
                    if sa.get("found") and sa.get("answer"):
                        parts.append(sa["answer"])
                answer = "\n\n".join(parts) if parts else ""

            # Extract sources from sub_answers
            sources = []
            for sa in result.get("sub_answers", []):
                for src in sa.get("sources", []):
                    if src and str(src) not in sources:
                        sources.append(str(src))

            return AgentResult(
                response=answer,
                sources=sources,
                raw_output={
                    "answer": answer,
                    "sub_answers": result.get("sub_answers", []),
                    "doc_selection_mode": result.get("doc_selection_mode"),
                    "selected_doc_id": result.get("selected_doc_id"),
                    "error": error,
                },
            )

        except Exception as exc:
            error_msg = f"Case Doc RAG adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
