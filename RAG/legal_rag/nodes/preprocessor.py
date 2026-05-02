"""
preprocessor.py
---------------
Preprocessor node: rewrites and classifies the user query.
"""

from __future__ import annotations

import json
import logging
import re
from dotenv import load_dotenv

load_dotenv()

from langsmith import traceable

from config.legal_rag import get_llm, MAX_LLM_CALLS, LLM_TIMEOUT
from RAG.legal_rag.prompts import PREPROCESSOR_PROMPT
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


def _fast_filter(query: str) -> str | None:
    """Rule-based off-topic check before spending an LLM call."""
    q = query.strip()
    if len(q) < 5:
        return "off_topic"
    if not re.search(r"[\u0600-\u06FF]", q):
        return "off_topic"
    return None


_CLASSIFICATION_MAP = {
    "تحليلي":       "analytical",
    "نصّي":         "textual",
    "خارج السياق":  "off_topic",
}


@traceable(name="Preprocessor Node")
def preprocessor_node(state: dict) -> dict:
    """Rewrite + classify the user query."""
    query         = state.get("last_query", "")
    corpus_config = state.get("corpus_config")
    law_name      = corpus_config.law_display_name if corpus_config else "القانون"

    if not query:
        state["classification"]    = "off_topic"
        state["rewritten_question"] = None
        return state

    # Budget guard
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        state["classification"]    = "analytical"
        state["rewritten_question"] = normalize(query)
        return state

    # Rule-based fast path
    if _fast_filter(query) == "off_topic":
        state["classification"]    = "off_topic"
        state["rewritten_question"] = None
        return state

    normalized_query = normalize(query)

    prompt = PREPROCESSOR_PROMPT.format(
        law_name=law_name,
        question=normalized_query,
    )
    response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
    state["llm_call_count"] = state.get("llm_call_count", 0) + 1
    content = _strip_code_fences(response.content.strip())

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log_event(logger, "preprocessor_json_error",
                  error=str(exc), raw=content[:300], level=logging.WARNING)
        state["rewritten_question"] = normalized_query
        state["classification"]    = "analytical"
        state["query_history"].append(query)
        return state

    state["rewritten_question"] = normalize(
        data.get("rewritten_question", normalized_query)
    )
    state["classification"] = _CLASSIFICATION_MAP.get(
        data.get("classification"), "analytical"
    )
    state["query_history"].append(query)

    log_event(logger, "preprocessor",
              corpus=corpus_config.name if corpus_config else "unknown",
              classification=state["classification"],
              rewritten=state["rewritten_question"])
    return state
