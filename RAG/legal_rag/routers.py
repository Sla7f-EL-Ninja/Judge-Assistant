# """
# routers.py
# ----------
# LangGraph routing functions for the legal_rag workflow.

# Grade semantics (set by rule_grader_node):
#     "pass"   → confidence high enough, go to generate
#     "refine" → confidence too low, rewrite query and retry
#     "llm"    → borderline, send to LLM grader
#     "fail"   → no docs at all or retries exhausted, terminal failure
# """

# from __future__ import annotations


# def top_level_router(state: dict) -> str:
#     classification = state.get("classification")
#     if classification == "off_topic":
#         return "off_topic_node"
#     if classification == "textual":
#         return "textual_node"
#     if classification == "analytical":
#         return "scope_classifier_node"
#     return "cannot_answer_node"


# def rule_grader_router(state: dict) -> str:
#     if state.get("retry_count", 0) >= state.get("max_retries", 3):
#         return "cannot_answer_node"

#     grade = state.get("grade")
#     if grade == "pass":
#         return "generate_answer_node"
#     if grade == "refine":
#         return "refine_node"
#     if grade == "llm":
#         return "llm_grader_node"
#     return "cannot_answer_node"


# def llm_grader_router(state: dict) -> str:
#     if state.get("retry_count", 0) >= state.get("max_retries", 3):
#         return "cannot_answer_node"

#     if state.get("llm_pass", False):
#         return "generate_answer_node"
#     return "refine_node"


"""
routers.py
----------
LangGraph routing functions for the legal_rag workflow.

Grade semantics (set by rule_grader_node):
    "pass"   → confidence high enough, go to generate
    "refine" → confidence too low, rewrite query and retry
    "llm"    → borderline, send to LLM grader
    "fail"   → no docs at all or retries exhausted, terminal failure
"""

from __future__ import annotations


def corpus_router_router(state: dict) -> str:
    """Route after corpus_router_node.

    If corpus_config was set, proceed to the preprocessor.
    If classification was set to off_topic (no corpus matched or fast filter
    triggered), short-circuit directly to off_topic_node.
    """
    if state.get("classification") == "off_topic":
        return "off_topic_node"
    if state.get("corpus_config") is not None:
        return "preprocessor_node"
    # Defensive: should not happen, but avoid a silent hang.
    return "off_topic_node"


def top_level_router(state: dict) -> str:
    classification = state.get("classification")
    if classification == "off_topic":
        return "off_topic_node"
    if classification == "textual":
        return "textual_node"
    if classification == "analytical":
        return "scope_classifier_node"
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