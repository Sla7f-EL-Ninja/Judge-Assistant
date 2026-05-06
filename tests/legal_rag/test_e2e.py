"""
test_e2e.py
-----------
End-to-end tests that invoke the full legal_rag pipeline against real services
(LLM API, Qdrant, TEI embedding, TEI reranker).

How to run:
    # Run ONLY e2e tests:
    pytest RAG/legal_rag/tests/test_e2e.py -m e2e -v

    # Run everything including e2e:
    pytest RAG/legal_rag/tests/ -m "e2e or not e2e" -v

    # Skip e2e (default):
    pytest RAG/legal_rag/tests/ -v       # e2e tests will be skipped automatically

Requirements before running:
    - LANGCHAIN_API_KEY (or LANGCHAIN_TRACING_V2=false) env var set
    - Qdrant running and all three corpora indexed
    - TEI embedding and reranker services reachable (or acceptable fallback)
    - LLM API key in environment (as used by get_llm())
"""

from __future__ import annotations

import os
import time

import pytest

# ---------------------------------------------------------------------------
# Skip all tests in this module unless the e2e marker is explicitly requested
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.e2e


def _services_available() -> bool:
    """Best-effort check that required env vars / service URLs are set."""
    llm_key_present = bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    return llm_key_present


SKIP_REASON = "Live LLM/Qdrant services not configured. Set API keys and ensure Qdrant is running."


# ===========================================================================
# Service-availability gate
# ===========================================================================
@pytest.fixture(autouse=True)
def require_live_services():
    if not _services_available():
        pytest.skip(SKIP_REASON)


# ===========================================================================
# Helpers
# ===========================================================================
def _ask(query: str):
    from RAG.legal_rag.service import ask_question
    return ask_question(query)


def _ensure_indexed():
    """Ensure all corpora are indexed before running e2e tests."""
    from RAG.legal_rag.civil_law_rag import ensure_indexed as civil_idx
    from RAG.legal_rag.evidence_rag import ensure_indexed as evidence_idx
    from RAG.legal_rag.procedures_rag import ensure_indexed as procedures_idx
    civil_idx()
    evidence_idx()
    procedures_idx()


# ===========================================================================
# Graph build
# ===========================================================================
@pytest.mark.e2e
def test_graph_builds_successfully():
    from RAG.legal_rag.graph import build_graph
    graph = build_graph()
    assert graph is not None


# ===========================================================================
# Input validation (no live services needed — but kept here for completeness)
# ===========================================================================
@pytest.mark.e2e
def test_empty_query_raises():
    from RAG.legal_rag.errors import QueryValidationError
    from RAG.legal_rag.service import ask_question
    with pytest.raises(QueryValidationError):
        ask_question("")


@pytest.mark.e2e
def test_non_arabic_query_raises():
    from RAG.legal_rag.errors import QueryValidationError
    from RAG.legal_rag.service import ask_question
    with pytest.raises(QueryValidationError):
        ask_question("What is the contract law?")


@pytest.mark.e2e
def test_too_short_query_raises():
    from RAG.legal_rag.errors import QueryValidationError
    from RAG.legal_rag.service import ask_question
    with pytest.raises(QueryValidationError):
        ask_question("عق")


# ===========================================================================
# Off-topic queries
# ===========================================================================
@pytest.mark.e2e
def test_off_topic_query_returns_graceful_response():
    result = _ask("ما هو أفضل مطعم في القاهرة؟")
    assert result.answer
    # Should be off_topic or unresolved corpus
    assert result.classification == "off_topic" or result.corpus is None


# ===========================================================================
# Civil law corpus
# ===========================================================================
@pytest.mark.e2e
@pytest.mark.slow
def test_civil_law_analytical_query():
    _ensure_indexed()
    result = _ask("ما هي شروط صحة العقد؟")
    assert result.answer
    assert result.corpus == "civil_law"
    assert result.classification in ("analytical", "textual")
    assert result.retrieval_confidence is not None


@pytest.mark.e2e
@pytest.mark.slow
def test_civil_law_textual_query():
    _ensure_indexed()
    result = _ask("ما نص المادة 89 من القانون المدني؟")
    assert result.answer
    assert result.corpus == "civil_law"


@pytest.mark.e2e
@pytest.mark.slow
def test_civil_law_article_range_query():
    _ensure_indexed()
    result = _ask("ما نص المواد من 89 إلى 91؟")
    assert result.answer
    assert result.corpus == "civil_law"


# ===========================================================================
# Evidence corpus
# ===========================================================================
@pytest.mark.e2e
@pytest.mark.slow
def test_evidence_analytical_query():
    _ensure_indexed()
    result = _ask("ما هي طرق الإثبات في المواد المدنية؟")
    assert result.answer
    assert result.corpus == "evidence_law"


@pytest.mark.e2e
@pytest.mark.slow
def test_evidence_textual_query():
    _ensure_indexed()
    result = _ask("ما نص المادة 1 من قانون الإثبات؟")
    assert result.answer
    assert result.corpus == "evidence_law"


# ===========================================================================
# Procedures corpus
# ===========================================================================
@pytest.mark.e2e
@pytest.mark.slow
def test_procedures_analytical_query():
    _ensure_indexed()
    result = _ask("ما هي شروط رفع الدعوى؟")
    assert result.answer
    assert result.corpus == "procedures_law"


@pytest.mark.e2e
@pytest.mark.slow
def test_procedures_textual_query():
    _ensure_indexed()
    result = _ask("ما نص المادة 10 من قانون المرافعات؟")
    assert result.answer
    assert result.corpus == "procedures_law"


# ===========================================================================
# Cross-corpus disambiguation
# ===========================================================================
@pytest.mark.e2e
@pytest.mark.slow
def test_cross_corpus_query_resolved_to_one_corpus():
    _ensure_indexed()
    result = _ask("هل يجوز الإثبات بالشهادة في عقد تجاوز قيمته عشرة آلاف جنيه؟")
    # Should resolve to exactly one corpus (evidence or civil)
    assert result.corpus in ("civil_law", "evidence_law", "procedures_law")
    assert result.answer


# ===========================================================================
# Result structure integrity
# ===========================================================================
@pytest.mark.e2e
@pytest.mark.slow
def test_result_has_expected_fields():
    _ensure_indexed()
    result = _ask("ما هي أحكام التعويض عن الضرر؟")
    assert hasattr(result, "answer")
    assert hasattr(result, "sources")
    assert hasattr(result, "classification")
    assert hasattr(result, "retrieval_confidence")
    assert hasattr(result, "citation_integrity")
    assert hasattr(result, "corpus")
    assert hasattr(result, "from_cache")
    assert hasattr(result, "corpus_routing_scores")
    assert isinstance(result.sources, list)


@pytest.mark.e2e
@pytest.mark.slow
def test_sources_have_expected_structure():
    _ensure_indexed()
    result = _ask("ما هي شروط صحة العقد؟")
    for source in result.sources:
        assert "article" in source
        assert "title" in source


# ===========================================================================
# Cache behaviour
# ===========================================================================
@pytest.mark.e2e
@pytest.mark.slow
def test_second_call_is_faster_due_to_cache():
    """The second identical call should hit the semantic cache and be noticeably faster."""
    _ensure_indexed()
    query = "ما هي شروط صحة العقد وفقًا للقانون المدني المصري؟"

    t0 = time.perf_counter()
    result1 = _ask(query)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    result2 = _ask(query)
    t2 = time.perf_counter() - t0

    assert result2.from_cache is True
    # Cache hit should be at least 5× faster
    assert t2 < t1 / 5, f"Cache hit ({t2:.2f}s) not faster than first call ({t1:.2f}s)"


# ===========================================================================
# Error resilience
# ===========================================================================
@pytest.mark.e2e
def test_exception_in_graph_returns_error_message_not_raises():
    """
    Simulate a graph that raises unexpectedly; service must return graceful answer.
    This patches build_graph at the service level only.
    """
    from unittest.mock import patch
    with patch("RAG.legal_rag.graph.build_graph") as mock_bg:
        mock_bg.return_value.invoke.side_effect = RuntimeError("unexpected error")
        from RAG.legal_rag.service import ask_question
        result = ask_question("ما هي شروط صحة العقد؟")
    assert result.answer
    assert "خطأ" in result.answer
