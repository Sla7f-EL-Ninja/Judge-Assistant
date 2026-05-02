"""
civil_law_rag_adapter.py
------------------------
Supervisor adapter for the Civil Law RAG agent.

Delegates to the legal_rag MCP server via get_client("legal_rag").
No direct RAG imports, no sys.path manipulation, no Arabic error-prefix scanning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult
from mcp_servers.errors import ErrorCode, MCPUnavailable, ToolError

logger = logging.getLogger(__name__)


class CivilLawRAGAdapter(AgentAdapter):
    """Thin adapter that delegates to the legal_rag MCP server."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            from mcp_servers.lifecycle import get_client

            resp = get_client("legal_rag").call(
                "search_legal_corpus",
                query=query,
                corpus="civil_law",
            )

            # Format sources as strings (list[str] AgentResult contract)
            sources = [
                f"المادة {s['article']}"
                + (f" — {s['title']}" if s.get("title") else "")
                for s in resp.get("sources", [])
                if s.get("article") is not None
            ]

            return AgentResult(
                response=resp["answer"],
                sources=sources,
                raw_output={k: resp.get(k) for k in (
                    "classification", "retrieval_confidence",
                    "citation_integrity", "from_cache", "corpus",
                )},
            )

        except ToolError as e:
            return AgentResult(response="", error=f"Civil Law RAG: {e.code} — {e.message}")
        except MCPUnavailable:
            return AgentResult(response="", error="MCP_UNAVAILABLE: legal_rag server unreachable")
        except Exception as exc:
            logger.exception("CivilLawRAGAdapter unexpected error")
            return AgentResult(response="", error=f"Civil Law RAG adapter error: {exc}")
