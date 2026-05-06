"""
test_refine.py
--------------
Unit tests for refine_node.
Mocks: _get_llm()
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from RAG.legal_rag.state import make_initial_state
from tests.legal_rag.conftest import make_mock_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _refine_json(refined_query: str) -> str:
    return json.dumps({"refined_query": refined_query})


def _state(
    query: str = "ما هي شروط صحة العقد؟",
    retry_count: int = 0,
    llm_call_count: int = 0,
    corpus_fixture=None,
    refined_query: str | None = None,
    failure_reason: str | None = None,
) -> dict:
    s = make_initial_state()
    s["last_query"] = query
    s["retry_count"] = retry_count
    s["llm_call_count"] = llm_call_count
    s["refined_query"] = refined_query
    s["failure_reason"] = failure_reason
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRefineNode:

    # ── Core behaviour ───────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_refined_query_stored_in_state(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(_refine_json("ما هي شروط انعقاد العقد الصحيح؟"))
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(corpus_fixture=civil_corpus)
        result = refine_node(state)
        assert result["refined_query"] == "ما هي شروط انعقاد العقد الصحيح؟"

    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_retry_count_always_incremented(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(_refine_json("سؤال محسّن"))
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(retry_count=1, corpus_fixture=civil_corpus)
        result = refine_node(state)
        assert result["retry_count"] == 2

    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_retry_count_incremented_even_on_budget_exhaust(self, mock_llm, civil_corpus):
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(retry_count=0, llm_call_count=5, corpus_fixture=civil_corpus)
        result = refine_node(state)
        mock_llm.assert_not_called()
        assert result["retry_count"] == 1

    # ── Budget guard ──────────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_budget_exhausted_skips_llm(self, mock_llm, civil_corpus):
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(llm_call_count=5, corpus_fixture=civil_corpus)
        refine_node(state)
        mock_llm.assert_not_called()

    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_budget_exhausted_leaves_refined_query_unchanged(self, mock_llm, civil_corpus):
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(
            llm_call_count=5,
            refined_query="استفسار سابق",
            corpus_fixture=civil_corpus,
        )
        result = refine_node(state)
        # refined_query should remain as it was
        assert result["refined_query"] == "استفسار سابق"

    # ── Query priority (refined > rewritten > last) ──────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_refined_query_takes_priority_over_last_query(self, mock_llm, civil_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = _refine_json("محسّن جديد")
            return resp
        mock_llm.return_value.invoke = capture_invoke

        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(
            query="سؤال أصلي",
            refined_query="استفسار محسّن سابق",
            corpus_fixture=civil_corpus,
        )
        refine_node(state)
        assert "استفسار محسّن سابق" in captured.get("prompt", "")

    # ── Failure reason injected into prompt ──────────────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_failure_reason_included_in_prompt(self, mock_llm, civil_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = _refine_json("سؤال محسّن")
            return resp
        mock_llm.return_value.invoke = capture_invoke

        reason = "لم يتم العثور على مواد ذات صلة"
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(failure_reason=reason, corpus_fixture=civil_corpus)
        refine_node(state)
        assert reason in captured.get("prompt", "")

    # ── Malformed JSON ────────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_malformed_json_falls_back_to_original_query(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm("not json at all {{{")
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(query="سؤال أصلي", corpus_fixture=civil_corpus)
        result = refine_node(state)
        # Should not raise; falls back to original
        assert result["refined_query"] == "سؤال أصلي"

    # ── LLM call count ────────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_llm_call_count_incremented(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(_refine_json("سؤال محسّن"))
        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(llm_call_count=2, corpus_fixture=civil_corpus)
        result = refine_node(state)
        assert result["llm_call_count"] == 3

    # ── Law name from corpus_config ───────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.refine._get_llm")
    def test_law_name_used_from_corpus_config(self, mock_llm, evidence_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = _refine_json("سؤال محسّن")
            return resp
        mock_llm.return_value.invoke = capture_invoke

        from RAG.legal_rag.nodes.refine import refine_node
        state = _state(corpus_fixture=evidence_corpus)
        refine_node(state)
        assert "قانون الإثبات" in captured.get("prompt", "")
