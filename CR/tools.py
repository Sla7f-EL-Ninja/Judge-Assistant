from typing import Any, Dict

from mcp_servers.errors import ErrorCode, MCPUnavailable, ToolError
from mcp_servers.lifecycle import get_client


def civil_law_rag_tool(query: str) -> Dict[str, Any]:
    try:
        resp = get_client("legal_rag").call(
            "search_legal_corpus",
            query=query,
            corpus="civil_law",
        )
        return {
            "answer":               resp.get("answer", ""),
            "sources":              resp.get("sources", []),   # [{article, title, ...}]
            "classification":       resp.get("classification"),
            "retrieval_confidence": resp.get("retrieval_confidence"),
            "citation_integrity":   resp.get("citation_integrity"),
            "from_cache":           resp.get("from_cache", False),
            "error":                None,
        }
    except ToolError as e:
        return {
            "answer": "", "sources": [], "error": f"{e.code}: {e.message}",
            "classification": None, "retrieval_confidence": None,
            "citation_integrity": None, "from_cache": False,
        }
    except MCPUnavailable:
        return {
            "answer": "", "sources": [], "error": "MCP_UNAVAILABLE",
            "classification": None, "retrieval_confidence": None,
            "citation_integrity": None, "from_cache": False,
        }


def case_documents_rag_tool(query: str, case_id: str) -> Dict[str, Any]:
    try:
        resp = get_client("case_doc_rag").call(
            "search_case_docs",
            query=query,
            case_id=case_id,
        )
        return {
            "final_answer": resp.get("answer", ""),
            "sources":      resp.get("sources", []),
            "sub_answers":  resp.get("sub_answers", []),
            "error":        None,
        }
    except ToolError as e:
        if e.code == ErrorCode.OFF_TOPIC:
            return {"final_answer": "", "sources": [], "sub_answers": [], "error": "off_topic"}
        return {
            "final_answer": "", "sources": [], "sub_answers": [],
            "error": f"{e.code}: {e.message}",
        }
    except MCPUnavailable:
        return {"final_answer": "", "sources": [], "sub_answers": [], "error": "MCP_UNAVAILABLE"}
