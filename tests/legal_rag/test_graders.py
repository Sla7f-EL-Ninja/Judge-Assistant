"""
test_graders.py
---------------
Unit tests for rule_grader_node (no LLM) and llm_grader_node (mock LLM).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from RAG.legal_rag.state import make_initial_state
from tests.legal_rag.conftest import make_mock_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _grader_json(passed: bool, reason: str = "") -> str:
    return json.dumps({"pass": passed, "reason": reason})


def _state(
    docs=None,
    confidence: float = 0.7,
    retry_count: int = 0,
    max_retries: int = 3,
    llm_call_count: int = 0,
    corpus_fixture=None,
) -> dict:
    s = make_initial_state()
    s["last_results"] = docs or []
    s["retrieval_confidence"] = confidence
    s["retry_count"] = retry_count
    s["max_retries"] = max_retries
    s["llm_call_count"] = llm_call_count
    s["last_query"] = "ما هي شروط صحة العقد؟"
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


def _make_docs(n: int = 3) -> list:
    return [
        Document(
            page_content=f"نص المادة {i}",
            metadata={"index": i, "source": "civil_law", "type": "article"},
        )
        for i in range(1, n + 1)
    ]


# ===========================================================================
# rule_grader_node
# ===========================================================================
class TestRuleGraderNode:

    def test_max_retries_reached_sets_fail(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        state = _state(docs=_make_docs(), confidence=0.9, retry_count=3, max_retries=3)
        result = rule_grader_node(state)
        assert result["grade"] == "fail"
        assert result["failure_reason"]

    def test_no_docs_sets_fail(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        state = _state(docs=[], confidence=0.0)
        result = rule_grader_node(state)
        assert result["grade"] == "fail"
        assert result["failure_reason"]

    def test_low_confidence_sets_refine(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        # confidence < 0.35 → refine
        state = _state(docs=_make_docs(1), confidence=0.20)
        result = rule_grader_node(state)
        assert result["grade"] == "refine"

    def test_borderline_confidence_sets_llm(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        # 0.35 <= confidence < 0.55 → llm
        state = _state(docs=_make_docs(3), confidence=0.45)
        result = rule_grader_node(state)
        assert result["grade"] == "llm"

    def test_high_confidence_sets_pass(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        # confidence >= 0.55 → pass
        state = _state(docs=_make_docs(5), confidence=0.75)
        result = rule_grader_node(state)
        assert result["grade"] == "pass"

    def test_exactly_min_confidence_threshold_sets_refine(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        # Exactly at _MIN_CONFIDENCE (0.35) — should pass the min check
        # but still be below 0.55, so → llm
        state = _state(docs=_make_docs(2), confidence=0.35)
        result = rule_grader_node(state)
        assert result["grade"] == "llm"

    def test_exactly_pass_threshold_sets_pass(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        state = _state(docs=_make_docs(3), confidence=0.55)
        result = rule_grader_node(state)
        assert result["grade"] == "pass"

    def test_retry_below_max_does_not_fail(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        state = _state(docs=_make_docs(3), confidence=0.8, retry_count=2, max_retries=3)
        result = rule_grader_node(state)
        assert result["grade"] == "pass"

    def test_failure_reason_set_on_no_docs(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        state = _state(docs=[], confidence=0.0)
        result = rule_grader_node(state)
        assert "failure_reason" in result
        assert result["failure_reason"]

    def test_failure_reason_set_on_max_retries(self):
        from RAG.legal_rag.nodes.graders import rule_grader_node
        state = _state(docs=_make_docs(), confidence=0.8, retry_count=3, max_retries=3)
        result = rule_grader_node(state)
        assert result["failure_reason"]


# ===========================================================================
# llm_grader_node
# ===========================================================================
class TestLLMGraderNode:

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_llm_pass_true(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(_grader_json(True, "المستندات مناسبة"))
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), corpus_fixture=civil_corpus)
        result = llm_grader_node(state)
        assert result["llm_pass"] is True

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_llm_pass_false_with_reason(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _grader_json(False, "المستندات لا تتعلق بالسؤال")
        )
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), corpus_fixture=civil_corpus)
        result = llm_grader_node(state)
        assert result["llm_pass"] is False
        assert result["failure_reason"]

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_malformed_json_defaults_to_fail(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm("invalid json {{{")
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), corpus_fixture=civil_corpus)
        result = llm_grader_node(state)
        assert result["llm_pass"] is False
        assert result["failure_reason"]

    def test_budget_exhausted_sets_fail_without_llm_call(self, civil_corpus):
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), llm_call_count=5, corpus_fixture=civil_corpus)
        with patch("RAG.legal_rag.nodes.graders._get_llm") as mock_llm:
            result = llm_grader_node(state)
            mock_llm.assert_not_called()
        assert result["llm_pass"] is False
        assert result["failure_reason"]

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_llm_call_count_incremented(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(_grader_json(True))
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), llm_call_count=2, corpus_fixture=civil_corpus)
        result = llm_grader_node(state)
        assert result["llm_call_count"] == 3

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_law_name_from_corpus_config(self, mock_llm, evidence_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = _grader_json(True)
            return resp

        mock_llm.return_value.invoke = capture_invoke
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), corpus_fixture=evidence_corpus)
        llm_grader_node(state)
        assert "قانون الإثبات" in captured.get("prompt", "")

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_refined_query_used_over_last_query(self, mock_llm, civil_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = _grader_json(True)
            return resp

        mock_llm.return_value.invoke = capture_invoke
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), corpus_fixture=civil_corpus)
        state["refined_query"] = "استفسار محسّن عن العقد"
        llm_grader_node(state)
        assert "استفسار محسّن" in captured.get("prompt", "")

    @patch("RAG.legal_rag.nodes.graders._get_llm")
    def test_llm_exception_sets_fail(self, mock_llm, civil_corpus):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("timeout")
        mock_llm.return_value = llm
        from RAG.legal_rag.nodes.graders import llm_grader_node
        state = _state(docs=_make_docs(), corpus_fixture=civil_corpus)
        result = llm_grader_node(state)
        assert result["llm_pass"] is False
