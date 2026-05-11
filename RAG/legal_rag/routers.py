"""
routers.py
----------
LangGraph routing functions for the legal_rag workflow.

Graph flow:
    START → preprocessor_node
              ↓ post_preprocessor_router
         off_topic_node | corpus_classifier_node
                               ↓ corpus_classifier_router
                    off_topic_node | textual_node | scope_classifier_node
                                              ↓ ...
                                         rule_grader_router / llm_grader_router

Grade semantics (set by rule_grader_node):
    "pass"   → confidence high enough, go to generate
    "refine" → confidence too low, rewrite query and retry
    "llm"    → borderline, send to LLM grader
    "fail"   → no docs at all or retries exhausted, terminal failure
"""

from __future__ import annotations


def post_preprocessor_router(state: dict) -> str:
    """Route after preprocessor_node.

    Off-topic queries (non-Arabic, too short, or classified off_topic by the
    LLM) short-circuit to off_topic_node.  All other classifications proceed
    to corpus_classifier_node so we know which corpus to search.
    """
    if state.get("error"):
        return "cannot_answer_node"
    if state.get("classification") == "off_topic":
        return "off_topic_node"
    return "corpus_classifier_node"


def corpus_classifier_router(state: dict) -> str:
    """Route after corpus_classifier_node.

    Possible outcomes:
    - Hard error set by corpus_classifier_node → cannot_answer_node
    - classification overwritten to "off_topic" (unsupported domain) → off_topic_node
    - corpus_config set + "textual"    → textual_node
    - corpus_config set + "analytical" → scope_classifier_node
    - corpus_config absent (defensive) → off_topic_node
    """
    if state.get("error"):
        return "cannot_answer_node"
    if state.get("classification") == "off_topic":
        return "off_topic_node"
    if state.get("corpus_config") is None:
        return "off_topic_node"

    classification = state.get("classification")
    if classification == "textual":
        return "textual_node"
    if classification == "analytical":
        return "scope_classifier_node"
    # Defensive fallback for any unexpected classification value.
    return "cannot_answer_node"


def rule_grader_router(state: dict) -> str:
    if state.get("retry_count", 0) >= state.get("max_retries", 3):
        return "cannot_answer_node"

    grade = state.get("grade")
    if grade == "pass":
        return "generate_answer_node"
    if grade == "refine":
        return "refine_node"
    if grade == "llm":
        return "llm_grader_node"
    return "cannot_answer_node"


def llm_grader_router(state: dict) -> str:
    if state.get("retry_count", 0) >= state.get("max_retries", 3):
        return "cannot_answer_node"

    if state.get("llm_pass", False):
        return "generate_answer_node"
    return "refine_node"
