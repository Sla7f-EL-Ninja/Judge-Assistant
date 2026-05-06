"""
test_service.py
---------------
Unit tests for service.py:
  - validate_query   (pure logic, no mocks)
  - _extract_sources (pure logic, no mocks)
  - ask_question     (mocks build_graph and SemanticCache)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from RAG.legal_rag.errors import QueryValidationError
from RAG.legal_rag.service import LegalRAGResult, _extract_sources, validate_query


# ===========================================================================
# validate_query — pure logic, zero mocks
# ===========================================================================
class TestValidateQuery:

    # ── valid inputs ─────────────────────────────────────────────────────────
    def test_valid_arabic_query_returned_stripped(self):
        q = "  ما هي شروط صحة العقد؟  "
        assert validate_query(q) == q.strip()

    def test_minimum_length_arabic_passes(self):
        result = validate_query("عقد")  # exactly 3 chars
        assert result == "عقد"

    def test_mixed_arabic_and_digits_passes(self):
        result = validate_query("المادة 89 من القانون المدني")
        assert result

    # ── empty / None ──────────────────────────────────────────────────────────
    def test_empty_string_raises(self):
        with pytest.raises(QueryValidationError):
            validate_query("")

    def test_whitespace_only_raises(self):
        with pytest.raises(QueryValidationError):
            validate_query("   ")

    def test_none_raises(self):
        with pytest.raises(QueryValidationError):
            validate_query(None)

    def test_non_string_raises(self):
        with pytest.raises(QueryValidationError):
            validate_query(123)

    # ── length boundaries ─────────────────────────────────────────────────────
    def test_too_short_raises(self):
        with pytest.raises(QueryValidationError, match="قصير"):
            validate_query("عق")  # 2 chars < MIN_QUERY_LENGTH (3)

    def test_too_long_raises(self):
        with pytest.raises(QueryValidationError, match="طويل"):
            validate_query("ا" * 2001)  # > MAX_QUERY_LENGTH (2000)

    def test_exactly_max_length_passes(self):
        query = "ا" * 2000
        result = validate_query(query)
        assert result == query

    # ── Arabic ratio ──────────────────────────────────────────────────────────
    def test_pure_latin_raises(self):
        with pytest.raises(QueryValidationError, match="عربي"):
            validate_query("What is the contract law?")

    def test_low_arabic_ratio_raises(self):
        # Only 1 Arabic char in a long Latin string → ratio < 0.3
        with pytest.raises(QueryValidationError, match="عربي"):
            validate_query("hello world this is a contract ا")

    def test_sufficient_arabic_ratio_passes(self):
        # 70% Arabic is fine
        result = validate_query("عقد contract law قانون عقد")
        assert result

    def test_all_spaces_counted_correctly(self):
        # Spaces are excluded from total_chars; pure Arabic with spaces passes
        result = validate_query("ما هي شروط الصحة")
        assert result


# ===========================================================================
# _extract_sources — pure logic, no mocks
# ===========================================================================
class TestExtractSources:

    def test_extracts_article_index(self, make_doc):
        doc = make_doc(index=89, title="المادة 89", chapter="الفصل الأول")
        result_state = {"last_results": [doc]}
        sources = _extract_sources(result_state)
        assert len(sources) == 1
        assert sources[0]["article"] == 89

    def test_doc_without_index_skipped(self):
        doc = Document(page_content="text", metadata={"source": "civil_law"})
        result_state = {"last_results": [doc]}
        sources = _extract_sources(result_state)
        assert sources == []

    def test_multiple_docs_extracted_in_order(self, make_doc):
        docs = [make_doc(index=i) for i in [10, 20, 30]]
        result_state = {"last_results": docs}
        sources = _extract_sources(result_state)
        assert len(sources) == 3
        assert [s["article"] for s in sources] == [10, 20, 30]

    def test_optional_fields_included_when_present(self, make_doc):
        doc = make_doc(index=5, book="الكتاب الأول", part="الجزء الأول", chapter="الفصل الأول")
        doc.metadata["book"] = "الكتاب الأول"
        doc.metadata["part"] = "الجزء الأول"
        sources = _extract_sources({"last_results": [doc]})
        assert sources[0]["book"] == "الكتاب الأول"
        assert sources[0]["part"] == "الجزء الأول"

    def test_empty_last_results_returns_empty_list(self):
        assert _extract_sources({"last_results": []}) == []

    def test_missing_last_results_key_returns_empty_list(self):
        assert _extract_sources({}) == []

    def test_title_defaults_to_article_number_when_missing(self):
        doc = Document(page_content="text", metadata={"index": 42, "source": "civil_law"})
        sources = _extract_sources({"last_results": [doc]})
        assert "42" in sources[0]["title"]


# ===========================================================================
# ask_question — mocks graph + cache
# ===========================================================================
class TestAskQuestion:

    def _make_result_state(self, civil_corpus, final_answer="إجابة اختبار", **overrides):
        state = {
            "final_answer": final_answer,
            "corpus_config": civil_corpus,
            "classification": "analytical",
            "retrieval_confidence": 0.75,
            "citation_integrity": "full",
            "corpus_routing_scores": [],
            "last_results": [],
        }
        state.update(overrides)
        return state

    @patch("RAG.legal_rag.graph.build_graph")
    @patch("RAG.legal_rag.service._get_cache")
    def test_returns_legal_rag_result(self, mock_get_cache, mock_build_graph, civil_corpus):
        mock_get_cache.return_value.set = MagicMock()
        mock_build_graph.return_value.invoke.return_value = self._make_result_state(civil_corpus)

        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")

        assert isinstance(result, LegalRAGResult)
        assert result.answer == "إجابة اختبار"
        assert result.corpus == "civil"

    @patch("RAG.legal_rag.graph.build_graph")
    @patch("RAG.legal_rag.service._get_cache")
    def test_answer_cached_after_successful_invocation(
        self, mock_get_cache, mock_build_graph, civil_corpus
    ):
        cache_mock = MagicMock()
        mock_get_cache.return_value = cache_mock
        mock_build_graph.return_value.invoke.return_value = self._make_result_state(civil_corpus)

        from RAG.legal_rag.service import ask_question
        ask_question("ما هي شروط صحة العقد؟")

        cache_mock.set.assert_called_once()

    @patch("RAG.legal_rag.graph.build_graph")
    @patch("RAG.legal_rag.service._get_cache")
    def test_no_corpus_config_in_result_skips_caching(
        self, mock_get_cache, mock_build_graph
    ):
        cache_mock = MagicMock()
        mock_get_cache.return_value = cache_mock
        mock_build_graph.return_value.invoke.return_value = {
            "final_answer": "إجابة",
            "corpus_config": None,
            "last_results": [],
        }

        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")

        cache_mock.set.assert_not_called()
        assert result.corpus is None

    @patch("RAG.legal_rag.graph.build_graph")
    @patch("RAG.legal_rag.service._get_cache")
    def test_missing_final_answer_returns_fallback_string(
        self, mock_get_cache, mock_build_graph, civil_corpus
    ):
        mock_get_cache.return_value.set = MagicMock()
        mock_build_graph.return_value.invoke.return_value = self._make_result_state(
            civil_corpus, final_answer=None
        )

        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")

        assert result.answer  # should be the fallback string, not empty/None

    @patch("RAG.legal_rag.graph.build_graph")
    def test_graph_exception_returns_error_result_not_raises(self, mock_build_graph):
        mock_build_graph.return_value.invoke.side_effect = RuntimeError("boom")

        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")

        assert result.answer  # graceful error message
        assert "خطأ" in result.answer

    def test_invalid_query_raises_validation_error(self):
        from RAG.legal_rag.service import ask_question
        with pytest.raises(QueryValidationError):
            ask_question("")

    def test_non_arabic_query_raises_validation_error(self):
        from RAG.legal_rag.service import ask_question
        with pytest.raises(QueryValidationError):
            ask_question("What is contract law?")

    @patch("RAG.legal_rag.graph.build_graph")
    @patch("RAG.legal_rag.service._get_cache")
    def test_sources_extracted_from_last_results(
        self, mock_get_cache, mock_build_graph, civil_corpus, make_doc
    ):
        docs = [make_doc(index=i) for i in [10, 20]]
        mock_get_cache.return_value.set = MagicMock()
        mock_build_graph.return_value.invoke.return_value = self._make_result_state(
            civil_corpus, last_results=docs
        )

        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")

        assert len(result.sources) == 2
        assert result.sources[0]["article"] == 10

    @patch("RAG.legal_rag.graph.build_graph")
    @patch("RAG.legal_rag.service._get_cache")
    def test_metadata_fields_populated(
        self, mock_get_cache, mock_build_graph, civil_corpus
    ):
        mock_get_cache.return_value.set = MagicMock()
        mock_build_graph.return_value.invoke.return_value = self._make_result_state(
            civil_corpus,
            classification="analytical",
            retrieval_confidence=0.82,
            citation_integrity="full",
            corpus_routing_scores=[{"corpus_name": "civil", "confidence": 0.9}],
        )

        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")

        assert result.classification == "analytical"
        assert result.retrieval_confidence == 0.82
        assert result.citation_integrity == "full"
        assert result.corpus_routing_scores
