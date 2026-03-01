"""
summarize_adapter.py

Adapter for the Summarization pipeline (Summerize/graph.py).

Wraps ``create_pipeline(llm)`` and returns an AgentResult with the
rendered Arabic case brief.
"""

import logging
import os
import sys
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)


class SummarizeAdapter(AgentAdapter):
    """Thin wrapper around the Summarization LangGraph pipeline."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """Run the summarisation pipeline on the provided documents.

        Parameters
        ----------
        query:
            The judge query (used contextually but the pipeline works
            on full documents).
        context:
            Should contain ``documents`` -- a list of dicts with keys
            ``raw_text`` and ``doc_id``.  If not present, the adapter
            will attempt to build a minimal document list from
            ``agent_results.ocr`` if OCR ran earlier in the same turn.
        """
        try:
            # Add Summerize directory to path
            summerize_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "Summerize"
            )
            summerize_dir = os.path.normpath(summerize_dir)
            if summerize_dir not in sys.path:
                sys.path.insert(0, summerize_dir)

            from dotenv import load_dotenv
            load_dotenv()

            from langchain_groq import ChatGroq
            from graph import create_pipeline

            # Build the documents list
            documents = context.get("documents")
            if not documents:
                # Fall back to OCR output if available from an earlier agent
                ocr_result = (context.get("agent_results") or {}).get("ocr")
                if ocr_result and isinstance(ocr_result, dict):
                    raw_texts = ocr_result.get("raw_texts", [])
                    documents = [
                        {"raw_text": t, "doc_id": f"doc_{i}"}
                        for i, t in enumerate(raw_texts)
                    ]

            if not documents:
                return AgentResult(
                    response="",
                    error="No documents provided for summarisation.",
                )

            llm = ChatGroq(model_name="llama-3.3-70b-versatile")
            pipeline = create_pipeline(llm)
            result = pipeline.invoke({"documents": documents})

            rendered_brief = result.get("rendered_brief", "")
            all_sources = result.get("all_sources", [])

            return AgentResult(
                response=rendered_brief,
                sources=all_sources,
                raw_output={
                    "rendered_brief": rendered_brief,
                    "case_brief": result.get("case_brief", {}),
                },
            )

        except Exception as exc:
            error_msg = f"Summarize adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)
