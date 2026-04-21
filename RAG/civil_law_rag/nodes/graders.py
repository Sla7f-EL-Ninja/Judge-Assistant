"""
graders.py
----------
Rule-based and LLM-based document graders.

rule_grader_node: fast, cheap quality gate — no LLM call.
llm_grader_node:  slow, expensive — only runs when rule grader is
                  borderline (grade == "fail" with at least 1 doc).

Routing logic (see routers.py):
    grade == "pass"    → generate_answer_node
    grade == "refine"  → refine_node
    grade == "fail"    → cannot_answer_node (empty docs — skip LLM)
    grade == "llm"     → llm_grader_node    (some docs but ambiguous)
"""

from __future__ import annotations

import json
import logging
import re
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langsmith import traceable

from config import get_llm

MAX_LLM_CALLS: int = 5
LLM_TIMEOUT: int = 30
from RAG.civil_law_rag.prompts import LLM_GRADER_PROMPT
from RAG.civil_law_rag.telemetry import get_logger, log_event

logger = get_logger(__name__)

_llm = None
_MIN_DOCS = 1
_MIN_CONFIDENCE = 0.35


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm("medium")
    return _llm


def _strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


@traceable(name="Rule Grader Node")
def rule_grader_node(state: dict) -> dict:
    """Fast rule-based grading — no LLM call."""
    docs       = state.get("last_results", [])
    confidence = state.get("retrieval_confidence", 0.0)
    retry      = state.get("retry_count", 0)
    max_r      = state.get("max_retries", 3)

    # Retry budget exhausted
    if retry >= max_r:
        state["grade"] = "fail"
        state["failure_reason"] = "تم تجاوز الحد الأقصى لمحاولات تحسين الاستعلام."
        log_event(logger, "rule_grader", grade="fail", reason="max_retries")
        return state

    # No documents at all → hard fail (LLM grader won't help)
    if not docs:
        state["grade"] = "fail"
        state["failure_reason"] = "لم يتم العثور على مواد قانونية."
        log_event(logger, "rule_grader", grade="fail", reason="no_docs")
        return state

    # Too few or low confidence → refine
    if len(docs) < _MIN_DOCS or confidence < _MIN_CONFIDENCE:
        state["grade"] = "refine"
        log_event(logger, "rule_grader", grade="refine",
                  docs=len(docs), confidence=confidence)
        return state

    # Borderline confidence (pass rule-based but worth LLM verification)
    if confidence < 0.55:
        state["grade"] = "llm"
        log_event(logger, "rule_grader", grade="llm",
                  docs=len(docs), confidence=confidence)
        return state

    # Clear pass
    state["grade"] = "pass"
    log_event(logger, "rule_grader", grade="pass",
              docs=len(docs), confidence=confidence, retry=retry)
    return state


@traceable(name="LLM Grader Node")
def llm_grader_node(state: dict) -> dict:
    """LLM-based grading for borderline retrieval quality."""
    # Budget guard
    if state.get("llm_call_count", 0) >= MAX_LLM_CALLS:
        state["llm_pass"] = False
        state["failure_reason"] = "تجاوز ميزانية استدعاءات النموذج."
        return state

    query = (
        state.get("refined_query")
        or state.get("rewritten_question")
        or state.get("last_query", "")
    )
    docs = state.get("last_results", [])

    docs_text = "\n\n".join(
        f"المادة {d.metadata.get('index', '?')}:\n{d.page_content}"
        for d in docs
    )

    prompt = LLM_GRADER_PROMPT.format(query=query, docs=docs_text)
    response = _get_llm().invoke(prompt, config={"timeout": LLM_TIMEOUT})
    state["llm_call_count"] = state.get("llm_call_count", 0) + 1

    try:
        result = json.loads(_strip_code_fences(response.content))
        state["llm_pass"]       = result["pass"]
        state["failure_reason"] = result.get("reason", "")
        log_event(logger, "llm_grader",
                  llm_pass=state["llm_pass"],
                  reason=state["failure_reason"])
    except Exception as exc:
        state["llm_pass"] = False
        state["failure_reason"] = "فشل تحليل المستندات بسبب خطأ في تنسيق الرد."
        log_event(logger, "llm_grader_error",
                  error=str(exc), raw=response.content[:300],
                  level=logging.WARNING)

    return state