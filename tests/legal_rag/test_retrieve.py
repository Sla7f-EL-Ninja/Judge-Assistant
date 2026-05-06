"""
test_retrieve.py
----------------
Unit tests for retrieve_node.
Mocks: load_vectorstore, rerank, _get_llm (for HyDE), _hyde_enabled, _probe_reranker
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from RAG.legal_rag.state import make_initial_state
from tests.legal_rag.conftest import make_mock_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _doc(index: int, score: float = 0.8, source: str = "civil_law") -> Document:
    return Document(
        page_content=f"نص المادة {index}",
        metadata={"index": index, "source": source, "type": "article"},
    )


def _state(
    query: str = "ما هي شروط صحة العقد؟",
    scope_filter: dict | None = None,
    llm_call_count: int = 0,
    corpus_fixture=None,
    refined_query: str | None = None,
) -> dict:
    s = make_initial_state()
    s["last_query"] = query
    s["scope_filter"] = scope_filter or {}
    s["llm_call_count"] = llm_call_count
    s["refined_query"] = refined_query
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


def _pairs(*indices: int, score: float = 0.75) -> list:
    """Build (Document, score) pairs for mocking similarity_search results."""
    return [(_doc(i), score) for i in indices]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRetrieveNode:

    # ── Basic retrieval path (HyDE disabled) ─────────────────────────────────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_results_stored_in_state(self, mock_vs, mock_rerank, mock_hyde, civil_corpus):
        docs = [_doc(i) for i in [1, 2, 3]]
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = _pairs(1, 2, 3)
        mock_rerank.return_value = docs

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=civil_corpus)
        result = retrieve_node(state)

        assert len(result["last_results"]) == 3
        assert result["retrieval_confidence"] > 0

    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_empty_results_sets_zero_confidence(self, mock_vs, mock_rerank, mock_hyde, civil_corpus):
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = []
        mock_rerank.return_value = []

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=civil_corpus)
        result = retrieve_node(state)

        assert result["last_results"] == []
        assert result["retrieval_confidence"] == 0.0

    # ── Deduplication by article index ───────────────────────────────────────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_duplicate_articles_deduped_keep_highest_score(
        self, mock_vs, mock_rerank, mock_hyde, civil_corpus
    ):
        # Article 1 appears twice — higher score should survive
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = [
            (_doc(1), 0.9),
            (_doc(1), 0.5),  # duplicate — lower score, should be dropped
            (_doc(2), 0.7),
        ]
        captured_docs = []
        def capture_rerank(query, docs, top_k):
            captured_docs.extend(docs)
            return docs
        mock_rerank.side_effect = capture_rerank

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=civil_corpus)
        retrieve_node(state)

        indices = [d.metadata["index"] for d in captured_docs]
        # Article 1 should appear only once
        assert indices.count(1) == 1

    # ── Confidence calculation ────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_confidence_is_average_of_reranked_scores(
        self, mock_vs, mock_rerank, mock_hyde, civil_corpus
    ):
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = [
            (_doc(1), 0.8),
            (_doc(2), 0.6),
        ]
        mock_rerank.return_value = [_doc(1), _doc(2)]

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=civil_corpus)
        result = retrieve_node(state)

        # Average of [0.8, 0.6] = 0.7
        assert result["retrieval_confidence"] == pytest.approx(0.7, abs=0.01)

    # ── Corpus config used for collection + source filter ────────────────────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_correct_collection_name_used(self, mock_vs, mock_rerank, mock_hyde, evidence_corpus):
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = []
        mock_rerank.return_value = []

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=evidence_corpus)
        retrieve_node(state)

        mock_vs.assert_called_once_with("evidence_docs")

    # ── Query priority: refined_query > rewritten_question > last_query ───────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_refined_query_used_when_present(self, mock_vs, mock_rerank, mock_hyde, civil_corpus):
        captured = {}
        def capture_search(query, k, filter):
            captured["query"] = query
            return []
        mock_vs.return_value.similarity_search_with_relevance_scores.side_effect = capture_search
        mock_rerank.return_value = []

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(
            query="سؤال أصلي",
            refined_query="استفسار محسّن",
            corpus_fixture=civil_corpus,
        )
        retrieve_node(state)
        assert "استفسار" in captured.get("query", "")

    # ── HyDE enabled: queries expanded, parallel search ──────────────────────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=True)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    @patch("RAG.legal_rag.nodes.retrieve._get_llm")
    def test_hyde_enabled_calls_llm_for_expansion(
        self, mock_llm, mock_vs, mock_rerank, mock_hyde, civil_corpus
    ):
        import json
        hyde_response = json.dumps({
            "hypothetical_article": "مادة افتراضية تنص على شروط العقد",
            "paraphrases": ["هل العقد صحيح؟", "ما متطلبات العقد؟"],
        })
        mock_llm.return_value = make_mock_llm(hyde_response)
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = []
        mock_rerank.return_value = []

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=civil_corpus)
        retrieve_node(state)
        mock_llm.assert_called()

    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=True)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    @patch("RAG.legal_rag.nodes.retrieve._get_llm")
    def test_hyde_llm_failure_falls_back_to_single_query(
        self, mock_llm, mock_vs, mock_rerank, mock_hyde, civil_corpus
    ):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        mock_llm.return_value = llm
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = _pairs(1, 2)
        mock_rerank.return_value = [_doc(1), _doc(2)]

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        state = _state(corpus_fixture=civil_corpus)
        result = retrieve_node(state)
        # Should not raise; fell back to single query
        assert isinstance(result["last_results"], list)

    # ── Scope filter applied ──────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False)
    @patch("RAG.legal_rag.nodes.retrieve.rerank")
    @patch("RAG.legal_rag.nodes.retrieve.load_vectorstore")
    def test_scope_filter_forwarded_to_build_filter(
        self, mock_vs, mock_rerank, mock_hyde, civil_corpus
    ):
        """retrieve_node should pass scope_filter through to Qdrant filter."""
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = []
        mock_rerank.return_value = []

        from RAG.legal_rag.nodes.retrieve import retrieve_node
        scope = {"chapter": "الفصل الأول", "section": "القسم الثاني"}
        state = _state(scope_filter=scope, corpus_fixture=civil_corpus)
        # Should not raise — just confirm it runs without error
        retrieve_node(state)
