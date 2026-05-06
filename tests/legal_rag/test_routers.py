"""
test_routers.py
---------------
Unit tests for all four routing functions in routers.py.
No mocks needed — pure state-dict logic.
"""

from __future__ import annotations

import pytest
from RAG.legal_rag.routers import (
    corpus_router_router,
    llm_grader_router,
    rule_grader_router,
    top_level_router,
)


# ===========================================================================
# corpus_router_router
# ===========================================================================
class TestCorpusRouterRouter:

    def test_corpus_config_set_routes_to_preprocessor(self, civil_corpus):
        state = {"corpus_config": civil_corpus, "classification": None}
        assert corpus_router_router(state) == "preprocessor_node"

    def test_off_topic_classification_routes_to_off_topic(self, civil_corpus):
        state = {"corpus_config": civil_corpus, "classification": "off_topic"}
        assert corpus_router_router(state) == "off_topic_node"

    def test_no_corpus_config_routes_to_off_topic(self):
        state = {"corpus_config": None, "classification": None}
        assert corpus_router_router(state) == "off_topic_node"

    def test_off_topic_takes_priority_over_corpus_config(self, civil_corpus):
        # Even if corpus_config is set, off_topic wins
        state = {"corpus_config": civil_corpus, "classification": "off_topic"}
        assert corpus_router_router(state) == "off_topic_node"

    def test_empty_state_routes_to_off_topic(self):
        assert corpus_router_router({}) == "off_topic_node"


# ===========================================================================
# top_level_router
# ===========================================================================
class TestTopLevelRouter:

    def test_off_topic(self):
        assert top_level_router({"classification": "off_topic"}) == "off_topic_node"

    def test_textual(self):
        assert top_level_router({"classification": "textual"}) == "textual_node"

    def test_analytical(self):
        assert top_level_router({"classification": "analytical"}) == "scope_classifier_node"

    def test_unknown_classification_routes_to_cannot_answer(self):
        assert top_level_router({"classification": "unknown"}) == "cannot_answer_node"

    def test_none_classification_routes_to_cannot_answer(self):
        assert top_level_router({"classification": None}) == "cannot_answer_node"

    def test_missing_classification_routes_to_cannot_answer(self):
        assert top_level_router({}) == "cannot_answer_node"


# ===========================================================================
# rule_grader_router
# ===========================================================================
class TestRuleGraderRouter:

    def test_grade_pass_routes_to_generate(self):
        state = {"grade": "pass", "retry_count": 0, "max_retries": 3}
        assert rule_grader_router(state) == "generate_answer_node"

    def test_grade_refine_routes_to_refine(self):
        state = {"grade": "refine", "retry_count": 0, "max_retries": 3}
        assert rule_grader_router(state) == "refine_node"

    def test_grade_llm_routes_to_llm_grader(self):
        state = {"grade": "llm", "retry_count": 0, "max_retries": 3}
        assert rule_grader_router(state) == "llm_grader_node"

    def test_grade_fail_routes_to_cannot_answer(self):
        state = {"grade": "fail", "retry_count": 0, "max_retries": 3}
        assert rule_grader_router(state) == "cannot_answer_node"

    def test_max_retries_reached_overrides_grade(self):
        # Even if grade == "pass", retry exhaustion must win
        state = {"grade": "pass", "retry_count": 3, "max_retries": 3}
        assert rule_grader_router(state) == "cannot_answer_node"

    def test_retry_exceeds_max_overrides_grade(self):
        state = {"grade": "refine", "retry_count": 5, "max_retries": 3}
        assert rule_grader_router(state) == "cannot_answer_node"

    def test_retry_below_max_allows_routing(self):
        state = {"grade": "refine", "retry_count": 2, "max_retries": 3}
        assert rule_grader_router(state) == "refine_node"

    def test_unknown_grade_routes_to_cannot_answer(self):
        state = {"grade": "unknown", "retry_count": 0, "max_retries": 3}
        assert rule_grader_router(state) == "cannot_answer_node"

    def test_none_grade_routes_to_cannot_answer(self):
        state = {"grade": None, "retry_count": 0, "max_retries": 3}
        assert rule_grader_router(state) == "cannot_answer_node"

    def test_default_max_retries_when_missing(self):
        # retry_count defaults to 0, max_retries defaults to 3 via state.get
        state = {"grade": "pass"}
        assert rule_grader_router(state) == "generate_answer_node"


# ===========================================================================
# llm_grader_router
# ===========================================================================
class TestLLMGraderRouter:

    def test_llm_pass_true_routes_to_generate(self):
        state = {"llm_pass": True, "retry_count": 0, "max_retries": 3}
        assert llm_grader_router(state) == "generate_answer_node"

    def test_llm_pass_false_routes_to_refine(self):
        state = {"llm_pass": False, "retry_count": 0, "max_retries": 3}
        assert llm_grader_router(state) == "refine_node"

    def test_llm_pass_none_routes_to_refine(self):
        state = {"llm_pass": None, "retry_count": 0, "max_retries": 3}
        assert llm_grader_router(state) == "refine_node"

    def test_max_retries_reached_routes_to_cannot_answer(self):
        state = {"llm_pass": True, "retry_count": 3, "max_retries": 3}
        assert llm_grader_router(state) == "cannot_answer_node"

    def test_retry_exceeds_max_routes_to_cannot_answer(self):
        state = {"llm_pass": False, "retry_count": 10, "max_retries": 3}
        assert llm_grader_router(state) == "cannot_answer_node"

    def test_retry_below_max_allows_llm_pass(self):
        state = {"llm_pass": True, "retry_count": 2, "max_retries": 3}
        assert llm_grader_router(state) == "generate_answer_node"
