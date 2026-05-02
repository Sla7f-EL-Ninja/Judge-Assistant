"""
refine.py
---------
Refine node: rewrites a failed/borderline query to improve retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable

from config.legal_rag import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT
from RAG.legal_rag.prompts import UNIFIED_REFINE_PROMPT
from RAG.legal_rag.indexing.normalizer import normalize
from RAG.legal_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("medium")
    return _llm


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


@traceable(name="Refine Node")
def refine_node(state: dict) -> dict:
    """Rewrite the query to improve next retrieval attempt."""
    state["retry_count"] = state.get("retry_count", 0) + 1

    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        log_event(logger, "refine_skipped", reason="llm_budget_exhausted",
                  level=logging.WARNING)
        return state

    corpus_config = state.get("corpus_config")
    law_name      = corpus_config.law_display_name if corpus_config else "القانون"

    query  = (
        state.get("refined_query")
        or state.get("rewritten_question")
        or state.get("last_query", "")
    )
    reason = state.get("failure_reason", "")
    reason_block = f"سبب فشل البحث السابق:\n{reason}" if reason else ""

    prompt = (
        UNIFIED_REFINE_PROMPT
        .replace("{law_name}", law_name)
        .replace("{query}", query)
        .replace("{reason_block}", reason_block)
    )

    response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
    state["llm_call_count"] = state.get("llm_call_count", 0) + 1

    try:
        data    = json.loads(_strip_code_fences(response.content))
        refined = data.get("refined_query")
        if refined:
            state["refined_query"] = normalize(refined)
        log_event(logger, "refine",
                  retry=state["retry_count"],
                  original=query,
                  refined=state.get("refined_query"))
    except Exception as exc:
        state["refined_query"] = query
        log_event(logger, "refine_json_error",
                  error=str(exc), raw=response.content[:300], level=logging.WARNING)

    return state
