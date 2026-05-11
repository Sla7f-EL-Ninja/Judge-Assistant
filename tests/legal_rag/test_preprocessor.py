"""
test_preprocessor.py
--------------------
Unit tests for preprocessor_node.
Mocks: _get_llm(), normalize() (optional — keep real for integration purity)
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
def _preproc_json(classification: str, rewritten: str) -> str:
    return json.dumps({
        "classification": classification,
        "rewritten_question": rewritten,
    })


def _state(
    query: str = "ما هي شروط صحة العقد؟",
    llm_call_count: int = 0,
    corpus_fixture=None,
) -> dict:
    s = make_initial_state()
    s["last_query"] = query
    s["llm_call_count"] = llm_call_count
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPreprocessorNode:

    # ── Fast off-topic gate (no LLM) ─────────────────────────────────────────
    def test_empty_query_sets_off_topic(self, civil_corpus):
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state(query="", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "off_topic"

    def test_short_non_arabic_query_is_off_topic_via_fast_filter(self, civil_corpus):
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state(query="hi", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "off_topic"

    def test_non_arabic_query_is_off_topic_via_fast_filter(self, civil_corpus):
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state(query="What is the contract law?", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "off_topic"

    # ── LLM budget guard ─────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_budget_exhausted_defaults_to_analytical(self, mock_llm, civil_corpus):
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state(query="ما هي شروط صحة العقد؟", llm_call_count=5, corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        # Should not call LLM
        mock_llm.assert_not_called()
        assert result["classification"] == "analytical"
        assert result["rewritten_question"] is not None

    # ── Analytical classification ─────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_analytical_classification_parsed(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _preproc_json("تحليلي", "ما هي شروط انعقاد العقد الصحيح؟")
        )
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("ما هي شروط صحة العقد؟", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "analytical"
        assert result["rewritten_question"]

    # ── Textual classification ────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_textual_classification_parsed(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _preproc_json("نصّي", "ما نص المادة 89؟")
        )
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("ما نص المادة 89 من القانون المدني؟", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "textual"

    # ── Off-topic classification from LLM ────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_off_topic_classification_from_llm(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _preproc_json("خارج السياق", "")
        )
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("ما هو أفضل مطعم في القاهرة؟", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "off_topic"

    # ── Unknown classification defaults to analytical ─────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_unknown_classification_defaults_to_analytical(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _preproc_json("غير معروف", "سؤال ما؟")
        )
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("سؤال ما؟", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["classification"] == "analytical"

    # ── Malformed JSON ────────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_malformed_json_defaults_to_analytical(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm("not json {{")
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("ما هي شروط صحة العقد؟", corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        # Should not raise; defaults gracefully
        assert result["classification"] == "analytical"
        assert result["rewritten_question"]

    # ── LLM call count incremented ────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_llm_call_count_incremented(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _preproc_json("تحليلي", "سؤال محسّن")
        )
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("ما هي شروط صحة العقد؟", llm_call_count=2, corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert result["llm_call_count"] == 3

    # ── Query history updated ─────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_query_added_to_history(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm(
            _preproc_json("تحليلي", "سؤال محسّن")
        )
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        query = "ما هي شروط صحة العقد؟"
        state = _state(query, corpus_fixture=civil_corpus)
        result = preprocessor_node(state)
        assert query in result["query_history"]

    # ── law_name used from corpus_config ─────────────────────────────────────
    @patch("RAG.legal_rag.nodes.preprocessor._get_llm")
    def test_law_name_from_corpus_config_injected_into_prompt(self, mock_llm, evidence_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = _preproc_json("تحليلي", "سؤال")
            return resp

        mock_llm.return_value.invoke = capture_invoke
        from RAG.legal_rag.nodes.preprocessor import preprocessor_node
        state = _state("ما هي طرق الإثبات؟", corpus_fixture=evidence_corpus)
        preprocessor_node(state)
        assert "قانون الإثبات" in captured.get("prompt", "")
