# """
# test_graph_paths.py
# -------------------
# Unit tests for all LangGraph routing paths.

# Tests use mock LLM responses and mock Qdrant to avoid any network
# calls during CI.
# """

# from __future__ import annotations

# from unittest.mock import MagicMock, patch

# import pytest
# from langchain_core.documents import Document

# from RAG.civil_law_rag.state import make_initial_state
# from RAG.civil_law_rag.routers import (
#     llm_grader_router,
#     rule_grader_router,
#     top_level_router,
# )
# from RAG.civil_law_rag.nodes.fallback import cannot_answer_node, off_topic_node
# from RAG.civil_law_rag.nodes.graders import rule_grader_node


# # ---------------------------------------------------------------------------
# # top_level_router
# # ---------------------------------------------------------------------------

# def test_top_level_off_topic():
#     state = make_initial_state()
#     state["classification"] = "off_topic"
#     assert top_level_router(state) == "off_topic_node"


# def test_top_level_textual():
#     state = make_initial_state()
#     state["classification"] = "textual"
#     assert top_level_router(state) == "textual_node"


# def test_top_level_analytical():
#     state = make_initial_state()
#     state["classification"] = "analytical"
#     assert top_level_router(state) == "retrieve_node"


# def test_top_level_unknown_falls_to_cannot_answer():
#     state = make_initial_state()
#     state["classification"] = "unknown_type"
#     assert top_level_router(state) == "cannot_answer_node"


# # ---------------------------------------------------------------------------
# # rule_grader_node + rule_grader_router
# # ---------------------------------------------------------------------------

# def test_rule_grader_pass():
#     state = make_initial_state()
#     state["last_results"] = [Document(page_content="text", metadata={"index": 1})]
#     state["retrieval_confidence"] = 0.75
#     result = rule_grader_node(state)
#     assert result["grade"] == "pass"
#     assert rule_grader_router(result) == "generate_answer_node"


# def test_rule_grader_refine_low_confidence():
#     state = make_initial_state()
#     state["last_results"] = [Document(page_content="text", metadata={"index": 1})]
#     state["retrieval_confidence"] = 0.20  # below _MIN_CONFIDENCE
#     result = rule_grader_node(state)
#     assert result["grade"] == "refine"
#     assert rule_grader_router(result) == "refine_node"


# def test_rule_grader_fail_no_docs():
#     state = make_initial_state()
#     state["last_results"] = []
#     state["retrieval_confidence"] = 0.0
#     result = rule_grader_node(state)
#     assert result["grade"] == "fail"
#     # CRITICAL: empty docs must go to cannot_answer, NOT llm_grader
#     assert rule_grader_router(result) == "cannot_answer_node"


# def test_rule_grader_llm_borderline():
#     state = make_initial_state()
#     state["last_results"] = [Document(page_content="text", metadata={"index": 1})]
#     state["retrieval_confidence"] = 0.45  # above MIN_CONFIDENCE, below 0.55
#     result = rule_grader_node(state)
#     assert result["grade"] == "llm"
#     assert rule_grader_router(result) == "llm_grader_node"


# def test_rule_grader_max_retries_reached():
#     state = make_initial_state()
#     state["retry_count"] = 3  # == max_retries
#     state["last_results"] = [Document(page_content="text", metadata={"index": 1})]
#     state["retrieval_confidence"] = 0.9
#     result = rule_grader_node(state)
#     assert result["grade"] == "fail"
#     assert rule_grader_router(result) == "cannot_answer_node"


# # ---------------------------------------------------------------------------
# # llm_grader_router
# # ---------------------------------------------------------------------------

# def test_llm_grader_pass():
#     state = make_initial_state()
#     state["llm_pass"] = True
#     assert llm_grader_router(state) == "generate_answer_node"


# def test_llm_grader_fail():
#     state = make_initial_state()
#     state["llm_pass"] = False
#     assert llm_grader_router(state) == "refine_node"


# def test_llm_grader_max_retries():
#     state = make_initial_state()
#     state["llm_pass"] = True
#     state["retry_count"] = 3
#     assert llm_grader_router(state) == "cannot_answer_node"


# # ---------------------------------------------------------------------------
# # Fallback nodes
# # ---------------------------------------------------------------------------

# def test_off_topic_node_sets_final_answer():
#     state = make_initial_state()
#     result = off_topic_node(state)
#     assert result["final_answer"]
#     assert "خارج نطاق" in result["final_answer"]


# def test_cannot_answer_node_sets_final_answer():
#     state = make_initial_state()
#     state["failure_reason"] = "لم يتم العثور على مواد"
#     result = cannot_answer_node(state)
#     assert result["final_answer"]
#     assert "تعذر" in result["final_answer"]
#     # No leading whitespace
#     assert not result["final_answer"].startswith(" ")
#     assert not result["final_answer"].startswith("\n")


"""
test_graph_paths.py
-------------------
Unit tests for all LangGraph routing paths in the unified Legal RAG.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# UPDATE: Pointing to legal_rag
from RAG.legal_rag.state import make_initial_state
from RAG.legal_rag.routers import (
    llm_grader_router,
    rule_grader_router,
    top_level_router,
)
from RAG.legal_rag.nodes.fallback import cannot_answer_node, off_topic_node
from RAG.legal_rag.nodes.graders import rule_grader_node


# ---------------------------------------------------------------------------
# top_level_router
# ---------------------------------------------------------------------------

def test_top_level_off_topic():
    state = make_initial_state()
    state["classification"] = "off_topic"
    assert top_level_router(state) == "off_topic_node"


def test_top_level_textual_routes_to_worker():
    state = make_initial_state()
    state["classification"] = "textual"
    # FIX: After classification, go to the worker, not the entry router[cite: 18]
    assert top_level_router(state) == "textual_node"


def test_top_level_analytical_routes_to_worker():
    state = make_initial_state()
    state["classification"] = "analytical"
    # FIX: Analytical queries go to the scope classifier[cite: 18]
    assert top_level_router(state) == "scope_classifier_node"


def test_top_level_unknown_falls_to_cannot_answer():
    state = make_initial_state()
    state["classification"] = "unknown_type"
    assert top_level_router(state) == "cannot_answer_node"


# ---------------------------------------------------------------------------
# rule_grader_node + rule_grader_router
# ---------------------------------------------------------------------------

def test_rule_grader_pass():
    state = {"grade": "pass"}
    assert rule_grader_router(state) == "generate_answer_node"


def test_rule_grader_fail_no_docs():
    state = {"grade": "fail"} # Result from grader node
    # Empty or low-confidence results must go to fallback[cite: 20]
    assert rule_grader_router(state) == "cannot_answer_node"


def test_rule_grader_llm_borderline():
    state = make_initial_state()
    state["last_results"] = [Document(page_content="text", metadata={"index": 1})]
    state["retrieval_confidence"] = 0.45 
    result = rule_grader_node(state)
    assert result["grade"] == "llm"
    assert rule_grader_router(result) == "llm_grader_node"


# ---------------------------------------------------------------------------
# llm_grader_router
# ---------------------------------------------------------------------------

def test_llm_grader_pass():
    state = make_initial_state()
    state["llm_pass"] = True
    assert llm_grader_router(state) == "generate_answer_node"


def test_llm_grader_fail():
    state = make_initial_state()
    state["llm_pass"] = False
    assert llm_grader_router(state) == "refine_node"


# ---------------------------------------------------------------------------
# Fallback nodes
# ---------------------------------------------------------------------------

def test_off_topic_node_sets_final_answer():
    state = make_initial_state()
    result = off_topic_node(state)
    assert result["final_answer"]
    assert "خارج نطاق" in result["final_answer"]


def test_cannot_answer_node_sets_final_answer():
    state = make_initial_state()
    state["failure_reason"] = "لم يتم العثور على مواد"
    result = cannot_answer_node(state)
    assert result["final_answer"]
    assert "تعذر" in result["final_answer"]