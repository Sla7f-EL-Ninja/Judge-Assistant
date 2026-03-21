"""
civil_law_rag_adapter.py

Adapter for the Civil Law RAG agent (RAG/Civil Law RAG/graph.py).
"""

import logging
import os
import sys
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult

logger = logging.getLogger(__name__)

# Modules that belong to the RAG package and share names with other
# packages already on sys.path (e.g. the API's config.py).
# These must be evicted from sys.modules before each import so Python
# re-resolves them from the RAG directory instead of the cache.
_RAG_MODULES = ("config", "graph", "nodes", "state", "edges", "utils")


class CivilLawRAGAdapter(AgentAdapter):
    """Thin wrapper around the Civil Law RAG LangGraph workflow."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            # 1. Resolve RAG directory
            rag_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "RAG", "Civil Law RAG",
            ))

            # 2. Put RAG dir at the FRONT of sys.path
            if rag_dir in sys.path:
                sys.path.remove(rag_dir)
            sys.path.insert(0, rag_dir)

            # 3. Evict any cached versions of RAG-owned modules so Python
            #    re-imports them from rag_dir, not from the API package.
            for mod in list(sys.modules.keys()):
                if mod in _RAG_MODULES or any(
                    mod.startswith(f"{m}.") for m in _RAG_MODULES
                ):
                    del sys.modules[mod]

            # 4. Now import RAG modules — they will resolve from rag_dir
            from dotenv import load_dotenv
            load_dotenv()

            from graph import app
            from config import default_state_template

            # Build initial state
            state = dict(default_state_template)
            state["last_query"] = query

            result = app.invoke(state)

            final_answer = result.get("final_answer", "")
            last_results = result.get("last_results", [])

            sources = []
            for doc in last_results:
                if hasattr(doc, "metadata"):
                    meta = doc.metadata
                    ref = meta.get("article", meta.get("source", ""))
                    if ref:
                        sources.append(str(ref))

            return AgentResult(
                response=final_answer or "",
                sources=sources,
                raw_output={
                    "final_answer": final_answer,
                    "classification": result.get("classification"),
                    "retrieval_confidence": result.get("retrieval_confidence"),
                },
            )

        except Exception as exc:
            error_msg = f"Civil Law RAG adapter error: {exc}"
            logger.exception(error_msg)
            return AgentResult(response="", error=error_msg)