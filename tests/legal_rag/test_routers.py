"""
test_routers.py
---------------
Unit tests for all routing functions in routers.py.
No mocks needed — pure state-dict logic.
"""

from __future__ import annotations

import pytest
from RAG.legal_rag.routers import (
    post_preprocessor_router,
    corpus_classifier_router,
    llm_grader_router,
    rule_grader_router,
)


# ===========================================================================
# post_preprocessor_router
# ===========================================================================
class TestPostPreprocessorRouter:

    def test_off_topic_routes_to_off_topic(self):
        state = {"classification": "off_topic", "error": None}
        assert post_preprocessor_router(state) == "off_topic_node"

    def test_analytical_routes_to_corpus_classifier(self):
        state = {"classification": "analytical", "error": None}
        assert post_preprocessor_router(state) == "corpus_classifier_node"

    def test_textual_routes_to_corpus_classifier(self):
        state = {"classification": "textual", "error": None}
        assert post_preprocessor_router(state) == "corpus_classifier_node"

    def test_error_set_routes_to_cannot_answer(self):
        state = {"classification": "analytical", "error": {"type": "PreprocessingError"}}
        assert post_preprocessor_router(state) == "cannot_answer_node"

    def test_none_classification_routes_to_corpus_classifier(self):
        state = {"classification": None, "error": None}
        assert post_preprocessor_router(state) == "corpus_classifier_node"


# ===========================================================================
# corpus_classifier_router
# ===========================================================================
class TestCorpusClassifierRouter:

    def test_textual_with_corpus_routes_to_textual(self, civil_corpus):
        state = {
            "classification": "textual",
            "corpus_config":  civil_corpus,
            "error":          None,
        }
        assert corpus_classifier_router(state) == "textual_node"

    def test_analytical_with_corpus_routes_to_scope(self, civil_corpus):
        state = {
            "classification": "analytical",
            "corpus_config":  civil_corpus,
            "error":          None,
        }
        assert corpus_classifier_router(state) == "scope_classifier_node"

    def test_off_topic_routes_to_off_topic(self, civil_corpus):
        state = {
            "classification": "off_topic",
            "corpus_config":  civil_corpus,
            "error":          None,
        }
        assert corpus_classifier_router(state) == "off_topic_node"

    def test_no_corpus_routes_to_off_topic(self):
        state = {"classification": "analytical", "corpus_config": None, "error": None}
        assert corpus_classifier_router(state) == "off_topic_node"

    def test_error_routes_to_cannot_answer(self, civil_corpus):
        state = {
            "classification": "analytical",
            "corpus_config":  civil_corpus,
            "error":          {"type": "CorpusRoutingError"},
        }
        assert corpus_classifier_router(state) == "cannot_answer_node"

    def test_unknown_classification_routes_to_cannot_answer(self, civil_corpus):
        state = {
            "classification": "unknown",
            "corpus_config":  civil_corpus,
            "error":          None,
        }
        assert corpus_classifier_router(state) == "cannot_answer_node"


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
