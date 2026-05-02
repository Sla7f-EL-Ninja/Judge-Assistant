"""
case_doc_server.py
------------------
FastMCP stdio server: wraps RAG.case_doc_rag.

Vector-store injection and graph compilation happen once at module import
(eager warmup) so the cold-start cost is paid at child-process spawn time,
not inside the first request — mirrors legal_rag_server pattern.

Spawn: python -m mcp_servers.case_doc_server
"""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # prevent LangSmith from hanging stdio pipes

import asyncio
import json
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional

import anyio
from mcp.server.fastmcp import FastMCP

from mcp_servers.errors import ErrorCode, raise_tool_error
from dotenv import load_dotenv
load_dotenv(override=True)

logger = logging.getLogger(__name__)

mcp = FastMCP("case-doc-rag-server")

# ---------------------------------------------------------------------------
# Singleton graph with double-checked locking
# ---------------------------------------------------------------------------

_app = None
_app_lock = threading.Lock()


def _get_app():
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                from RAG.case_doc_rag.graph import build_graph
                from RAG.case_doc_rag.infrastructure import (
                    get_embedding_function,
                    get_qdrant_client,
                    set_vectorstore,
                )
                from langchain_qdrant import QdrantVectorStore

                qdrant_client = get_qdrant_client()
                embedding_fn = get_embedding_function()
                vs = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name="case_docs",
                    embedding=embedding_fn,
                )
                set_vectorstore(vs)
                _app = build_graph()
                logger.info("Case doc RAG graph initialised with vector store")
    return _app


# ---------------------------------------------------------------------------
# Module-level warmup — pays cold-start cost at child-process spawn time.
# ---------------------------------------------------------------------------

logger.info("Warming up case_doc_rag...")
_get_app()
logger.info("case_doc_rag ready")

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_case_docs(
    query: str,
    case_id: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
) -> str:
    """Search case documents and return a structured JSON answer."""
    if not case_id:
        raise_tool_error(ErrorCode.INVALID_ARG, "case_id must be non-empty")

    initial_state = {
        "query": query,
        "case_id": case_id,
        "conversation_history": conversation_history or [],
        "request_id": request_id or str(uuid.uuid4()),
        "sub_questions": [],
        "on_topic": True,
        "doc_selection_mode": "no_doc_specified",
        "selected_doc_id": None,
        "doc_titles": [],
        "sub_answers": [],
        "final_answer": "",
        "error": None,
    }

    try:
        def _run_graph():
            return asyncio.run(_get_app().ainvoke(initial_state))

        result = await anyio.to_thread.run_sync(_run_graph)
    except Exception as e:
        logger.error("search_case_docs graph failed: %s", e)
        raise_tool_error(ErrorCode.INTERNAL, f"Graph invocation failed: {e}")

    if result.get("on_topic") is False:
        raise_tool_error(
            ErrorCode.OFF_TOPIC,
            "query classified as off-topic for case documents",
            case_id=case_id,
        )

    if result.get("error"):
        raise_tool_error(ErrorCode.INTERNAL, str(result["error"]))

    final_answer = result.get("final_answer", "")
    if not final_answer:
        raise_tool_error(ErrorCode.INTERNAL, "empty final_answer")

    seen: set = set()
    sources: List[str] = []
    for sub in result.get("sub_answers", []):
        for src in sub.get("sources", []):
            s = str(src)
            if s and s not in seen:
                seen.add(s)
                sources.append(s)

    return json.dumps({
        "answer": final_answer,
        "sources": sources,
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "selected_doc_id": result.get("selected_doc_id"),
    }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")