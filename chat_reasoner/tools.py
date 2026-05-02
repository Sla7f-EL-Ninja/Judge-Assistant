"""
tools.py

Tool registry for Chat Reasoner. All RAG calls route through the MCP server
layer — no direct RAG imports, no sys.path manipulation, no module eviction.
"""

import inspect as _inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from chat_reasoner.state import StepResult
from mcp_servers.errors import ErrorCode, MCPUnavailable, ToolError
from mcp_servers.lifecycle import get_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities (kept as-is — used by other modules)
# ---------------------------------------------------------------------------


def _extract_last_ai_message(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content or ""
    return ""


def _extract_doc_sources(retrieved_docs: list) -> List[str]:
    sources = []
    for doc in retrieved_docs:
        if isinstance(doc, dict):
            title = doc.get("title") or doc.get("doc_id") or ""
            if title:
                sources.append(str(title))
    return sources


def _ref_from_metadata(doc) -> str:
    if hasattr(doc, "metadata"):
        meta = doc.metadata
        return str(meta.get("article") or meta.get("source") or "")
    return ""


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


def _run_case_doc_rag(
    step: dict,
    case_id: str,
    conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    try:
        resp = get_client("case_doc_rag").call(
            "search_case_docs",
            query=step["query"],
            case_id=case_id,
            conversation_history=[
                {"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in (conversation_history or [])
            ],
            request_id=step.get("request_id"),
        )
        answer = resp.get("answer", "")
        if not answer:
            return StepResult(
                step_id=step["step_id"],
                tool="case_doc_rag",
                query=step["query"],
                status="failure",
                response="",
                sources=[],
                error="case_doc_rag returned empty answer",
                raw_output={},
            )
        return StepResult(
            step_id=step["step_id"],
            tool="case_doc_rag",
            query=step["query"],
            status="success",
            response=answer,
            sources=resp.get("sources", []),
            raw_output={"sub_answers_count": len(resp.get("sub_answers", []))},
        )

    except ToolError as e:
        if e.code == ErrorCode.OFF_TOPIC:
            return StepResult(
                step_id=step["step_id"],
                tool="case_doc_rag",
                query=step["query"],
                status="failure",
                response="",
                sources=[],
                error="off_topic",
                raw_output={},
            )
        return StepResult(
            step_id=step["step_id"],
            tool="case_doc_rag",
            query=step["query"],
            status="failure",
            response="",
            error=f"{e.code}: {e.message}",
        )
    except MCPUnavailable:
        return StepResult(
            step_id=step["step_id"],
            tool="case_doc_rag",
            query=step["query"],
            status="failure",
            response="",
            error="MCP_UNAVAILABLE",
        )
    except Exception as exc:
        logger.exception("case_doc_rag tool error (step %s): %s", step.get("step_id"), exc)
        return StepResult(
            step_id=step["step_id"],
            tool="case_doc_rag",
            query=step["query"],
            status="failure",
            response="",
            error=str(exc),
        )


def _rewrite_civil_query_with_context(original_query: str, prior_results: List[dict]) -> str:
    """Rewrite original_query to embed concrete case facts from prior step results."""
    try:
        from config import get_llm
        context_block = "\n\n".join(
            r["response"][:1500] for r in prior_results if r.get("response")
        )
        if not context_block.strip():
            return original_query

        prompt = (
            "أعد صياغة السؤال التالي ليستهدف مواد القانون المدني المصري المنطبقة على الوقائع المذكورة. "
            "ادمج الوقائع الجوهرية فقط (أسماء الأطراف والتواريخ غير ضرورية). "
            "اجعل السؤال ≤ 400 حرف. لا تضف مقدمات.\n\n"
            f"السؤال الأصلي:\n{original_query}\n\n"
            f"وقائع القضية من الخطوات السابقة:\n{context_block}"
        )
        response = get_llm("medium").invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        if rewritten:
            logger.info(
                "civil_law_rag query rewritten: [%s] → [%s]",
                original_query[:80],
                rewritten[:80],
            )
            return rewritten
    except Exception as exc:
        logger.warning("civil_law_rag query rewrite failed, using original: %s", exc)
    return original_query


def _run_civil_law_rag(
    step: dict,
    case_id: str,
    conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    try:
        query = step["query"]
        if prior_results:
            query = _rewrite_civil_query_with_context(query, prior_results)

        resp = get_client("legal_rag").call(
            "search_legal_corpus",
            query=query,
            corpus="civil_law",
        )

        final = resp.get("answer", "")
        if not final:
            return StepResult(
                step_id=step["step_id"],
                tool="civil_law_rag",
                query=step["query"],
                status="failure",
                response="",
                sources=[],
                error="civil_law_rag returned empty final_answer",
                raw_output={},
            )

        # Sources from server are dicts [{article, title, ...}]; format as strings
        sources = [
            f"المادة {s['article']}" + (f" — {s['title']}" if s.get("title") else "")
            for s in resp.get("sources", [])
            if s.get("article") is not None
        ]

        return StepResult(
            step_id=step["step_id"],
            tool="civil_law_rag",
            query=step["query"],
            status="success",
            response=final,
            sources=sources,
            raw_output={
                "classification":       resp.get("classification"),
                "retrieval_confidence": resp.get("retrieval_confidence"),
            },
        )

    except ToolError as e:
        return StepResult(
            step_id=step["step_id"],
            tool="civil_law_rag",
            query=step["query"],
            status="failure",
            response="",
            error=f"{e.code}: {e.message}",
        )
    except MCPUnavailable:
        return StepResult(
            step_id=step["step_id"],
            tool="civil_law_rag",
            query=step["query"],
            status="failure",
            response="",
            error="MCP_UNAVAILABLE",
        )
    except Exception as exc:
        logger.exception("civil_law_rag tool error (step %s): %s", step.get("step_id"), exc)
        return StepResult(
            step_id=step["step_id"],
            tool="civil_law_rag",
            query=step["query"],
            status="failure",
            response="",
            error=str(exc),
        )


def _run_fetch_summary_report(
    step: dict,
    case_id: str,
    conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    """Sync pymongo read from the summaries collection."""
    try:
        from pymongo import MongoClient
        from config.supervisor import MONGO_URI, MONGO_DB
        from api.db.collections import SUMMARIES

        client = MongoClient(MONGO_URI)
        try:
            doc = client[MONGO_DB][SUMMARIES].find_one({"case_id": case_id})
        finally:
            client.close()

        if not doc:
            return StepResult(
                step_id=step["step_id"],
                tool="fetch_summary_report",
                query=step["query"],
                status="skipped",
                response="",
                sources=[],
                raw_output={"reason": "no summary found for this case_id"},
            )

        body = doc.get("summary", "")
        sources = list(doc.get("sources", []) or [])
        return StepResult(
            step_id=step["step_id"],
            tool="fetch_summary_report",
            query=step["query"],
            status="success",
            response=body,
            sources=sources,
            raw_output={"generated_at": str(doc.get("generated_at", ""))},
        )

    except Exception as exc:
        logger.exception(
            "fetch_summary_report error (step %s): %s", step.get("step_id"), exc
        )
        return StepResult(
            step_id=step["step_id"],
            tool="fetch_summary_report",
            query=step["query"],
            status="failure",
            response="",
            error=f"mongo error: {exc}",
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, Callable] = {
    "case_doc_rag":       _run_case_doc_rag,
    "civil_law_rag":      _run_civil_law_rag,
    "fetch_summary_report": _run_fetch_summary_report,
}


def dispatch_tool(
    step: dict,
    case_id: str,
    conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    """Route a step to its tool function."""
    fn = TOOL_REGISTRY.get(step.get("tool", ""))
    if fn is None:
        return StepResult(
            step_id=step.get("step_id", "unknown"),
            tool=step.get("tool", "unknown"),
            query=step.get("query", ""),
            status="failure",
            response="",
            error=f"unknown tool: {step.get('tool')}",
        )
    try:
        sig = _inspect.signature(fn)
        nparams = sum(
            1 for p in sig.parameters.values()
            if p.kind not in (
                _inspect.Parameter.VAR_POSITIONAL,
                _inspect.Parameter.VAR_KEYWORD,
            )
        )
        accepts_prior = nparams >= 4 or any(
            p.kind in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD)
            for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        accepts_prior = True

    if accepts_prior:
        return fn(step, case_id, conversation_history, prior_results or [])
    return fn(step, case_id, conversation_history)
