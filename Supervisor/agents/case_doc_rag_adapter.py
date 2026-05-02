"""
case_doc_rag_adapter.py
-----------------------
Supervisor adapter for the Case Document RAG agent.

Delegates to the case_doc_rag MCP server via get_client("case_doc_rag").
No direct RAG imports, no build_graph() per-call, no state dict construction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult
from mcp_servers.errors import ErrorCode, MCPUnavailable, ToolError

logger = logging.getLogger(__name__)


class CaseDocRAGAdapter(AgentAdapter):
    """Thin adapter that delegates to the case_doc_rag MCP server."""

    def invoke(self, query: str, context: Dict[str, Any]) -> AgentResult:
        try:
            from mcp_servers.lifecycle import get_client

            resp = get_client("case_doc_rag").call(
                "search_case_docs",
                query=query,
                case_id=context.get("case_id", ""),
                conversation_history=context.get("conversation_history", []),
                request_id=context.get("correlation_id"),
            )

            return AgentResult(
                response=resp["answer"],
                sources=resp.get("sources", []),
                raw_output={
                    "final_answer":      resp["answer"],
                    "sub_answers":       resp.get("sub_answers", []),
                    "doc_selection_mode": resp.get("doc_selection_mode"),
                    "selected_doc_id":   resp.get("selected_doc_id"),
                },
            )

        except ToolError as e:
            if e.code == ErrorCode.OFF_TOPIC:
                # Preserve exact error string downstream supervisor routing expects
                return AgentResult(
                    response="",
                    error="Case Doc RAG: query classified as off-topic for case documents",
                )
            return AgentResult(response="", error=f"Case Doc RAG: {e.code} — {e.message}")
        except MCPUnavailable:
            return AgentResult(response="", error="MCP_UNAVAILABLE: case_doc_rag server unreachable")
        except Exception as exc:
            logger.exception("CaseDocRAGAdapter unexpected error")
            return AgentResult(response="", error=f"Case Doc RAG adapter error: {exc}")
