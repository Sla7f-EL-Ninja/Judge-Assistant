"""
tools.py

Tool registry and self-contained RAG loaders for Chat Reasoner.

Loaders replicate the sys.path manipulation from the Supervisor adapters
(Supervisor/agents/civil_law_rag_adapter.py, case_doc_rag_adapter.py) so
the chat_reasoner package has no Supervisor/ imports.

RAG graph instances are cached at module level (singletons) to avoid
repeated import overhead on each invocation.
"""

import copy
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from chat_reasoner.state import StepResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_CIVIL_LAW_RAG_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "RAG", "civil_law_rag"))
_CASE_DOC_RAG_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "RAG", "Case Doc RAG"))

# ---------------------------------------------------------------------------
# Civil Law RAG — cached singleton
# ---------------------------------------------------------------------------

_civil_law_app = None
_civil_law_state_template = None


def _load_civil_law_rag():
    """Import the Civil Law RAG compiled graph, handling sys.path isolation.

    Verbatim logic from civil_law_rag_adapter._load_rag_app — put the RAG
    dir at the front of sys.path so its config/graph/nodes shadow any
    same-named API-level packages, then evict only the stale non-RAG copies.
    """
    rag_dir = _CIVIL_LAW_RAG_DIR

    if rag_dir in sys.path:
        sys.path.remove(rag_dir)
    sys.path.insert(0, rag_dir)

    _RAG_MODULES = ("config", "graph", "nodes", "state", "edges", "utils")
    for mod_name in list(sys.modules.keys()):
        if mod_name not in _RAG_MODULES and not any(
            mod_name.startswith(f"{m}.") for m in _RAG_MODULES
        ):
            continue
        mod = sys.modules[mod_name]
        mod_file = getattr(mod, "__file__", None) or ""
        if not mod_file.startswith(rag_dir):
            del sys.modules[mod_name]

    from dotenv import load_dotenv
    load_dotenv()

    from graph import build_graph  # noqa: E402 — resolved from rag_dir
    from config.rag import get_default_state  # noqa: E402

    return build_graph(), get_default_state()


def _get_civil_law_app():
    global _civil_law_app, _civil_law_state_template
    if _civil_law_app is None:
        logger.info("Loading Civil Law RAG graph (first call)...")
        _civil_law_app, _civil_law_state_template = _load_civil_law_rag()
        logger.info("Civil Law RAG graph loaded and cached.")
    return _civil_law_app, _civil_law_state_template


# ---------------------------------------------------------------------------
# Case Doc RAG — imported lazily (no persistent cache needed; rag_docs is
# a module-level singleton after first import)
# ---------------------------------------------------------------------------

_case_doc_app = None


def _get_case_doc_rag_app():
    """Import the Case Doc RAG compiled graph from rag_docs module.

    Unlike the Supervisor adapter we do NOT inject a shared vector store
    (that would require importing Supervisor/nodes/). Chat Reasoner therefore
    queries the persistent Qdrant collection directly. Documents indexed in
    the same request turn but not yet flushed will not be visible, which is
    acceptable for a fallback reasoning agent operating on stored case material.
    """
    global _case_doc_app
    if _case_doc_app is not None:
        return _case_doc_app

    from RAG.case_doc_rag.graph import build_graph
    _case_doc_app = build_graph()
    logger.info("Case Doc RAG graph loaded and cached.")
    return _case_doc_app


# ---------------------------------------------------------------------------
# Helper utilities
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
    step: dict, case_id: str, conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    try:
        app = _get_case_doc_rag_app()

        import uuid
        initial_state = {
            "query": step["query"],
            "case_id": case_id,
            "conversation_history": [
                {"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in (conversation_history or [])
            ],
            "request_id": step.get("request_id") or str(uuid.uuid4()),
            # required reducer fields
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
            from RAG.case_doc_rag.infrastructure import (
                set_vectorstore as _set_case_vs,
                get_qdrant_client as _get_case_client,
                get_embedding_function as _get_case_emb,
            )
            from langchain_qdrant import QdrantVectorStore
            _set_case_vs(QdrantVectorStore(
                client=_get_case_client(),
                collection_name="case_docs",
                embedding=_get_case_emb(),
            ))
        except Exception as _vs_exc:
            logger.warning("Could not inject case vectorstore for case_id=%s: %s", case_id, _vs_exc)
        result = app.invoke(initial_state)
        answer = result.get("final_answer", "")
        sources = list({
            src
            for sa in result.get("sub_answers", [])
            for src in sa.get("sources", [])
        })

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
            sources=sources,
            raw_output={
                "sub_answers_count": len(result.get("sub_answers", [])),
            },
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
    """Rewrite original_query to embed concrete case facts from prior step results.

    Uses medium-tier LLM to produce a focused civil law retrieval query.
    Falls back to original_query on any error.
    """
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
    step: dict, case_id: str, conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    try:
        app, template = _get_civil_law_app()
        state = copy.deepcopy(template)
        query = step["query"]
        if prior_results:
            query = _rewrite_civil_query_with_context(query, prior_results)
        state["last_query"] = query
        result = app.invoke(state)

        final = result.get("final_answer", "")
        sources = [
            r for r in (
                _ref_from_metadata(doc) for doc in result.get("last_results", [])
            )
            if r
        ]

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

        return StepResult(
            step_id=step["step_id"],
            tool="civil_law_rag",
            query=step["query"],
            status="success",
            response=final,
            sources=sources,
            raw_output={
                "classification": result.get("classification"),
                "retrieval_confidence": result.get("retrieval_confidence"),
            },
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
    step: dict, case_id: str, conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    """Sync pymongo read from the summaries collection.

    Does NOT call api.services.summary_service.get_summary — that is async
    Motor and cannot be awaited from a sync LangGraph node running inside the
    FastAPI event loop. Mirrors the pattern in summarize_adapter.py.
    """
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
    "case_doc_rag": _run_case_doc_rag,
    "civil_law_rag": _run_civil_law_rag,
    "fetch_summary_report": _run_fetch_summary_report,
}


# ---------------------------------------------------------------------------
# Pre-warm Civil Law RAG at import time so the SentenceTransformer model
# is loaded before any request arrives, preventing gRPC DEADLINE_EXCEEDED
# on the first call.
# ---------------------------------------------------------------------------
try:
    _get_civil_law_app()
    logger.info("Civil Law RAG pre-warmed successfully.")
except Exception as _prewarm_exc:
    logger.warning(
        "Civil Law RAG pre-warm failed (will retry on first call): %s", _prewarm_exc
    )


import inspect as _inspect


def dispatch_tool(
    step: dict,
    case_id: str,
    conversation_history: List[dict],
    prior_results: Optional[List[dict]] = None,
) -> StepResult:
    """Route a step to its tool function. Returns a failure StepResult if the
    tool name is not in TOOL_REGISTRY (should never happen after validation)."""
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
    # Call with prior_results only if the registered function accepts a 4th argument
    # (test mocks may only declare 3 params)
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