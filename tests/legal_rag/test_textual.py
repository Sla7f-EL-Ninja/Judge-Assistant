"""
test_textual.py
---------------
Unit tests for textual_node.
Mocks: get_qdrant_client, load_vectorstore
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from RAG.legal_rag.state import make_initial_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _state(query: str, corpus_fixture=None) -> dict:
    s = make_initial_state()
    s["last_query"] = query
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


def _qdrant_point(index: int, content: str, source: str = "civil_law") -> MagicMock:
    """Simulate a Qdrant ScoredPoint with proper payload structure."""
    point = MagicMock()
    point.payload = {
        "page_content": content,
        "metadata": {
            "index": index,
            "source": source,
            "type": "article",
        },
    }
    return point


def _mock_client(*points) -> MagicMock:
    client = MagicMock()
    client.scroll.return_value = (list(points), None)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTextualNode:

    # ── Exact single article lookup ──────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_exact_article_returns_content(self, mock_client, civil_corpus):
        content = "المادة 89: يجوز..."
        mock_client.return_value = _mock_client(_qdrant_point(89, content))
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المادة 89 من القانون المدني؟", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert content in result["final_answer"]
        assert result["current_article"] == 89

    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_exact_article_not_found_returns_error_message(self, mock_client, civil_corpus):
        client = MagicMock()
        client.scroll.return_value = ([], None)
        mock_client.return_value = client
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المادة 9999 من القانون المدني؟", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert result["final_answer"]
        assert "عذرًا" in result["final_answer"] or "لم" in result["final_answer"]

    # ── Article range ────────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_range_query_fetches_all_articles(self, mock_client, civil_corpus):
        points = [_qdrant_point(i, f"المادة {i}") for i in range(89, 93)]
        mock_client.return_value = _mock_client(*points)
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المواد من 89 إلى 92؟", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert len(result["last_results"]) == 4
        assert result["current_article"] == "89-92"

    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_range_query_arabic_bein_syntax(self, mock_client, civil_corpus):
        """Test 'بين X و Y' syntax."""
        points = [_qdrant_point(i, f"المادة {i}") for i in range(10, 13)]
        mock_client.return_value = _mock_client(*points)
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المواد بين 10 و 12؟", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert result["current_article"] == "10-12"

    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_range_not_found_returns_error_message(self, mock_client, civil_corpus):
        mock_client.return_value = _mock_client()
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المواد من 5000 إلى 5002؟", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert "عذرًا" in result["final_answer"] or "لم" in result["final_answer"]

    # ── Fallback: semantic search ────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.textual.load_vectorstore")
    def test_no_article_number_falls_back_to_semantic(self, mock_vs, civil_corpus):
        docs = [
            Document(
                page_content="نص المادة 1",
                metadata={"index": 1, "source": "civil_law", "type": "article"},
            )
        ]
        mock_vs.return_value.similarity_search.return_value = docs
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("أعطني نص مادة عن العقود", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert result["last_results"]
        mock_vs.return_value.similarity_search.assert_called_once()

    @patch("RAG.legal_rag.nodes.textual.load_vectorstore")
    def test_semantic_fallback_no_results_returns_error(self, mock_vs, civil_corpus):
        mock_vs.return_value.similarity_search.return_value = []
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("أعطني نص مادة غير موجودة", corpus_fixture=civil_corpus)
        result = textual_node(state)
        assert "عذرًا" in result["final_answer"] or "لم" in result["final_answer"]

    # ── Corpus config used correctly ─────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_evidence_corpus_uses_correct_collection(self, mock_client, evidence_corpus):
        mock_client.return_value = _mock_client(_qdrant_point(1, "نص", source="evidence_law"))
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المادة 1؟", corpus_fixture=evidence_corpus)
        textual_node(state)
        call_kwargs = mock_client.return_value.scroll.call_args
        assert call_kwargs[1]["collection_name"] == "evidence_docs"

    # ── Docs sorted by article index ─────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.textual.get_qdrant_client")
    def test_results_sorted_by_article_index(self, mock_client, civil_corpus):
        # Return articles out of order
        points = [_qdrant_point(i, f"المادة {i}") for i in [92, 89, 91, 90]]
        mock_client.return_value = _mock_client(*points)
        from RAG.legal_rag.nodes.textual import textual_node
        state = _state("ما نص المواد من 89 إلى 92؟", corpus_fixture=civil_corpus)
        result = textual_node(state)
        indices = [d.metadata["index"] for d in result["last_results"]]
        assert indices == sorted(indices)
