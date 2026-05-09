"""
test_generate.py
----------------
Unit tests for generate_answer_node and the internal citation-integrity helpers.
Mocks: _get_llm()
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
def _state(
    docs=None,
    query: str = "ما هي شروط صحة العقد؟",
    llm_call_count: int = 0,
    corpus_fixture=None,
    refined_query: str | None = None,
) -> dict:
    s = make_initial_state()
    s["last_results"] = docs or []
    s["last_query"] = query
    s["llm_call_count"] = llm_call_count
    s["refined_query"] = refined_query
    if corpus_fixture:
        s["corpus_config"] = corpus_fixture
    return s


def _doc(index: int, content: str | None = None) -> Document:
    return Document(
        page_content=content or f"نص المادة {index}",
        metadata={"index": index, "source": "civil_law", "type": "article"},
    )


# ---------------------------------------------------------------------------
# Citation integrity helpers (tested directly)
# ---------------------------------------------------------------------------
class TestVerifyCitations:

    def test_no_article_mentions_returns_none_integrity(self):
        from RAG.legal_rag.nodes.generate import _verify_citations
        answer = "هذه إجابة بدون ذكر مواد."
        cleaned, integrity = _verify_citations(answer, {89, 90})
        assert integrity == "none"
        assert cleaned == answer

    def test_all_cited_articles_valid_returns_full(self):
        from RAG.legal_rag.nodes.generate import _verify_citations
        answer = "وفقًا للمادة 89 والمادة 90 من القانون."
        cleaned, integrity = _verify_citations(answer, {89, 90})
        assert integrity == "full"
        assert "89" in cleaned
        assert "90" in cleaned

    def test_invented_citation_stripped_returns_partial(self):
        from RAG.legal_rag.nodes.generate import _verify_citations
        answer = "وفقًا للمادة 89 والمادة 999 من القانون."
        cleaned, integrity = _verify_citations(answer, {89})
        assert integrity == "partial"
        assert "89" in cleaned
        assert "999" not in cleaned

    def test_all_citations_invalid_returns_partial(self):
        from RAG.legal_rag.nodes.generate import _verify_citations
        answer = "وفقًا لأحكام المادة 999 من القانون."
        cleaned, integrity = _verify_citations(answer, {89})
        assert integrity == "partial"
        assert "999" not in cleaned

    def test_multiple_invalid_citations_all_stripped(self):
        from RAG.legal_rag.nodes.generate import _verify_citations
        answer = "المادة 1 والمادة 2 والمادة 3 تنص على..."
        cleaned, integrity = _verify_citations(answer, {2})
        assert "المادة 1" not in cleaned
        assert "المادة 3" not in cleaned
        # Article 2 is valid — should remain
        assert "المادة 2" in cleaned
        assert integrity == "partial"

    def test_empty_retrieved_indices_strips_all(self):
        from RAG.legal_rag.nodes.generate import _verify_citations
        answer = "المادة 89 تنص على ذلك."
        cleaned, integrity = _verify_citations(answer, set())
        assert "89" not in cleaned
        assert integrity == "partial"


# ---------------------------------------------------------------------------
# _retrieved_article_indices helper
# ---------------------------------------------------------------------------
class TestRetrievedArticleIndices:

    def test_extracts_integer_indices(self):
        from RAG.legal_rag.nodes.generate import _retrieved_article_indices
        docs = [_doc(i) for i in [10, 20, 30]]
        assert _retrieved_article_indices(docs) == {10, 20, 30}

    def test_skips_docs_without_index(self):
        from RAG.legal_rag.nodes.generate import _retrieved_article_indices
        doc = Document(page_content="text", metadata={"source": "civil_law"})
        assert _retrieved_article_indices([doc]) == set()

    def test_skips_non_integer_index(self):
        from RAG.legal_rag.nodes.generate import _retrieved_article_indices
        doc = Document(page_content="text", metadata={"index": "not_int"})
        assert _retrieved_article_indices([doc]) == set()


# ---------------------------------------------------------------------------
# generate_answer_node
# ---------------------------------------------------------------------------
class TestGenerateAnswerNode:

    @patch("RAG.legal_rag.nodes.generate._get_llm")
    def test_answer_set_in_state(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm("إجابة قانونية شاملة حول شروط صحة العقد.")
        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[_doc(89), _doc(90)], corpus_fixture=civil_corpus)
        result = generate_answer_node(state)
        assert result["final_answer"]
        assert "إجابة قانونية" in result["final_answer"]

    @patch("RAG.legal_rag.nodes.generate._get_llm")
    def test_citation_integrity_stored(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm("وفقًا للمادة 89 من القانون.")
        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[_doc(89)], corpus_fixture=civil_corpus)
        result = generate_answer_node(state)
        assert result["citation_integrity"] in ("full", "partial", "none")

    def test_no_docs_returns_no_results_message(self, civil_corpus):
        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[], corpus_fixture=civil_corpus)
        with patch("RAG.legal_rag.nodes.generate._get_llm") as mock_llm:
            result = generate_answer_node(state)
            mock_llm.assert_not_called()
        assert result["final_answer"]
        assert "لم يتم" in result["final_answer"]

    def test_budget_exhausted_returns_budget_message(self, civil_corpus):
        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[_doc(89)], llm_call_count=5, corpus_fixture=civil_corpus)
        with patch("RAG.legal_rag.nodes.generate._get_llm") as mock_llm:
            result = generate_answer_node(state)
            mock_llm.assert_not_called()
        assert result["final_answer"]
        assert "ميزانية" in result["final_answer"]

    @patch("RAG.legal_rag.nodes.generate._get_llm")
    def test_llm_call_count_incremented(self, mock_llm, civil_corpus):
        mock_llm.return_value = make_mock_llm("إجابة.")
        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[_doc(1)], llm_call_count=2, corpus_fixture=civil_corpus)
        result = generate_answer_node(state)
        assert result["llm_call_count"] == 3

    @patch("RAG.legal_rag.nodes.generate._get_llm")
    def test_refined_query_preferred_over_last_query(self, mock_llm, civil_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = "إجابة."
            return resp
        mock_llm.return_value.invoke = capture_invoke

        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(
            docs=[_doc(1)],
            query="سؤال أصلي",
            refined_query="استفسار محسّن",
            corpus_fixture=civil_corpus,
        )
        generate_answer_node(state)
        assert "استفسار محسّن" in captured.get("prompt", "")

    @patch("RAG.legal_rag.nodes.generate._get_llm")
    def test_law_name_from_corpus_config_in_prompt(self, mock_llm, evidence_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = "إجابة."
            return resp
        mock_llm.return_value.invoke = capture_invoke

        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[_doc(1)], corpus_fixture=evidence_corpus)
        generate_answer_node(state)
        assert "قانون الإثبات" in captured.get("prompt", "")

    @patch("RAG.legal_rag.nodes.generate._get_llm")
    def test_context_contains_article_indices(self, mock_llm, civil_corpus):
        captured = {}
        def capture_invoke(prompt, **kwargs):
            captured["prompt"] = prompt
            resp = MagicMock()
            resp.content = "إجابة."
            return resp
        mock_llm.return_value.invoke = capture_invoke

        from RAG.legal_rag.nodes.generate import generate_answer_node
        state = _state(docs=[_doc(42), _doc(99)], corpus_fixture=civil_corpus)
        generate_answer_node(state)
        assert "42" in captured.get("prompt", "")
        assert "99" in captured.get("prompt", "")
