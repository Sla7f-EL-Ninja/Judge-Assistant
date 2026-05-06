"""
test_scope_classifier.py
------------------------
Unit tests for scope_classifier_node (two-stage chapter → section narrowing).
Mocks: _get_llm(), load_toc()
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from RAG.legal_rag.state import make_initial_state
from tests.legal_rag.conftest import SAMPLE_TOC, make_mock_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ch_json(chapter_id: str, confidence: float) -> str:
    return json.dumps({"chapter_id": chapter_id, "confidence": confidence})


def _sec_json(section_id: str, confidence: float) -> str:
    return json.dumps({"section_id": section_id, "confidence": confidence})


def _state(query: str = "ما هي شروط صحة العقد؟", llm_call_count: int = 0, corpus_fixture=None) -> dict:
    s = make_initial_state()
    s["last_query"] = query
    s["llm_call_count"] = llm_call_count
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestScopeClassifierNode:

    # ── No chapter match (below threshold) → empty scope_filter ─────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    def test_low_chapter_confidence_sets_empty_scope_filter(
        self, mock_threshold, mock_llm, mock_toc, civil_corpus
    ):
        # Chapter confidence below threshold → no filter
        mock_llm.return_value = make_mock_llm(_ch_json("1", 0.2))
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert result["scope_filter"] == {}

    # ── Chapter matched, section skipped → chapter-only filter ───────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    @patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5)
    def test_chapter_matched_section_below_threshold_gives_chapter_only_filter(
        self, mock_sec_threshold, mock_ch_threshold, mock_llm, mock_toc, civil_corpus
    ):
        # chapter high, section low
        mock_llm.return_value = make_mock_llm(
            _ch_json("1", 0.8),   # first call → chapter
            _sec_json("1", 0.2),  # second call → section (below threshold)
        )
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert "chapter" in result["scope_filter"]
        assert "section" not in result["scope_filter"]
        assert result["current_chapter"] == SAMPLE_TOC[0]["title"]

    # ── Both chapter and section matched → chapter + section filter ──────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    @patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5)
    def test_chapter_and_section_matched_gives_full_filter(
        self, mock_sec_threshold, mock_ch_threshold, mock_llm, mock_toc, civil_corpus
    ):
        mock_llm.return_value = make_mock_llm(
            _ch_json("1", 0.9),   # chapter high
            _sec_json("2", 0.85), # section high
        )
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert result["scope_filter"].get("chapter")
        assert result["scope_filter"].get("section")
        assert result["current_chapter"]
        assert result["current_section"]

    # ── LLM budget guard ─────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    def test_budget_exhausted_skips_llm_and_sets_empty_filter(self, mock_toc, civil_corpus):
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(llm_call_count=5, corpus_fixture=civil_corpus)
        with patch("RAG.legal_rag.nodes.scope_classifier._get_llm") as mock_llm:
            result = scope_classifier_node(state)
            mock_llm.assert_not_called()
        assert result["scope_filter"] == {}

    # ── LLM exception on chapter → empty filter ──────────────────────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    def test_llm_exception_on_chapter_sets_empty_filter(self, mock_llm, civil_corpus):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("timeout")
        mock_llm.return_value = llm
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert result["scope_filter"] == {}

    # ── Unknown chapter_id from LLM → empty filter ───────────────────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    def test_unknown_chapter_id_sets_empty_filter(
        self, mock_threshold, mock_llm, mock_toc, civil_corpus
    ):
        mock_llm.return_value = make_mock_llm(_ch_json("999", 0.99))
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert result["scope_filter"] == {}

    # ── Scope confidence stored ───────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    @patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5)
    def test_scope_confidence_stored(
        self, mock_sec_threshold, mock_ch_threshold, mock_llm, mock_toc, civil_corpus
    ):
        mock_llm.return_value = make_mock_llm(
            _ch_json("1", 0.75),
            _sec_json("1", 0.2),  # below section threshold
        )
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert result["scope_confidence"] == pytest.approx(0.75)

    # ── LLM call count incremented per stage ─────────────────────────────────
    @patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC)
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    @patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5)
    def test_llm_call_count_incremented_for_both_stages(
        self, mock_sec_threshold, mock_ch_threshold, mock_llm, mock_toc, civil_corpus
    ):
        mock_llm.return_value = make_mock_llm(
            _ch_json("1", 0.9),
            _sec_json("1", 0.8),
        )
        from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
        state = _state(corpus_fixture=civil_corpus)
        result = scope_classifier_node(state)
        assert result["llm_call_count"] == 2  # one per stage

    # ── Chapter with no sections → no section stage ───────────────────────────
    @patch("RAG.legal_rag.nodes.scope_classifier._get_llm")
    @patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5)
    @patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5)
    def test_chapter_with_no_sections_skips_section_stage(
        self, mock_sec_threshold, mock_ch_threshold, mock_llm, civil_corpus
    ):
        toc_no_sections = [
            {"id": "1", "title": "الفصل الأول", "book": "", "part": "", "sections": []}
        ]
        with patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=toc_no_sections):
            mock_llm.return_value = make_mock_llm(_ch_json("1", 0.9))
            from RAG.legal_rag.nodes.scope_classifier import scope_classifier_node
            state = _state(corpus_fixture=civil_corpus)
            result = scope_classifier_node(state)
            # Chapter found, no section → chapter-only filter
            assert result["scope_filter"].get("chapter")
            assert "section" not in result["scope_filter"]
            assert result["current_section"] is None
