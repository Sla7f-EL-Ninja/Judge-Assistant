"""
test_corpus_router.py
---------------------
Unit tests for corpus_router_node.
Mocks: _get_llm(), _get_registry(), _corpus_threshold()
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from RAG.legal_rag.state import make_initial_state
from tests.legal_rag.conftest import make_mock_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scores_json(*entries) -> str:
    """Build the JSON string the LLM returns for a given list of score dicts."""
    return json.dumps({"scores": list(entries)})


def _score(corpus_name: str, confidence: float, reason: str = "") -> dict:
    return {"corpus_name": corpus_name, "confidence": confidence, "reason": reason}


def _state(query: str = "ما هي شروط صحة العقد؟", llm_call_count: int = 0) -> dict:
    s = make_initial_state()
    s["last_query"] = query
    s["llm_call_count"] = llm_call_count
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCorpusRouterNode:

    # ── Fast off-topic gate ──────────────────────────────────────────────────
    def test_empty_query_is_fast_off_topic(self):
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state(query="")
        result = corpus_router_node(state)
        assert result["classification"] == "off_topic"
        assert result["corpus_config"] is None

    def test_very_short_query_is_fast_off_topic(self):
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state(query="عقد")  # 3 chars — fast gate fires at < 5
        result = corpus_router_node(state)
        assert result["classification"] == "off_topic"

    def test_non_arabic_query_is_fast_off_topic(self):
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state(query="What is contract law?")
        result = corpus_router_node(state)
        assert result["classification"] == "off_topic"

    # ── LLM budget guard ─────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    def test_budget_exhausted_falls_back_to_civil(self, mock_registry, mock_registry_fixture):
        mock_registry.return_value = mock_registry_fixture
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state(query="ما هي شروط صحة العقد؟", llm_call_count=5)
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "civil"
        assert result["corpus_routing_scores"] == []

    # ── Successful routing ────────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_civil_query_routed_to_civil(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(
                _score("civil", 0.9, "سؤال مدني"),
                _score("evidence", 0.1),
                _score("procedures", 0.1),
            )
        )
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط صحة العقد؟")
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "civil"
        assert result.get("classification") != "off_topic"

    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_evidence_query_routed_to_evidence(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(
                _score("civil", 0.1),
                _score("evidence", 0.85),
                _score("procedures", 0.05),
            )
        )
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي طرق الإثبات في المواد المدنية؟")
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "evidence"

    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_procedures_query_routed_to_procedures(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(
                _score("civil", 0.1),
                _score("evidence", 0.1),
                _score("procedures", 0.88),
            )
        )
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط رفع الدعوى؟")
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "procedures"

    # ── Winner selection (highest confidence) ────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_highest_confidence_wins(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(
                _score("civil", 0.6),
                _score("evidence", 0.8),  # highest
                _score("procedures", 0.7),
            )
        )
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي طرق الإثبات؟")
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "evidence"

    # ── Below threshold → off_topic ───────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_all_scores_below_threshold_marks_off_topic(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(
                _score("civil", 0.1),
                _score("evidence", 0.2),
                _score("procedures", 0.1),
            )
        )
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هو أفضل مطعم في القاهرة؟")
        result = corpus_router_node(state)
        assert result["classification"] == "off_topic"
        assert result["corpus_config"] is None

    # ── Unknown corpus name ───────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_unknown_corpus_name_from_llm_marks_off_topic(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(_score("unknown_corpus", 0.99))
        )
        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط صحة العقد؟")
        result = corpus_router_node(state)
        assert result["classification"] == "off_topic"

    # ── LLM exception fallback ────────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    def test_llm_exception_falls_back_to_civil(
        self, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM is down")
        mock_llm.return_value = llm

        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط صحة العقد؟")
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "civil"

    # ── Malformed JSON from LLM ───────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    def test_malformed_json_falls_back_to_civil(
        self, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm("not valid json at all")

        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط صحة العقد؟")
        result = corpus_router_node(state)
        assert result["corpus_config"].name == "civil"

    # ── Routing scores stored in state ───────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_routing_scores_stored_in_state(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        scores = [
            _score("civil", 0.9),
            _score("evidence", 0.1),
            _score("procedures", 0.05),
        ]
        mock_llm.return_value = make_mock_llm(_scores_json(*scores))

        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط صحة العقد؟")
        result = corpus_router_node(state)
        assert len(result["corpus_routing_scores"]) == 3

    # ── LLM call count incremented ────────────────────────────────────────────
    @patch("RAG.legal_rag.nodes.corpus_router._get_registry")
    @patch("RAG.legal_rag.nodes.corpus_router._get_llm")
    @patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4)
    def test_llm_call_count_incremented(
        self, mock_threshold, mock_llm, mock_registry, mock_registry_fixture
    ):
        mock_registry.return_value = mock_registry_fixture
        mock_llm.return_value = make_mock_llm(
            _scores_json(_score("civil", 0.9))
        )

        from RAG.legal_rag.nodes.corpus_router import corpus_router_node
        state = _state("ما هي شروط صحة العقد؟")
        state["llm_call_count"] = 1
        result = corpus_router_node(state)
        assert result["llm_call_count"] == 2


# ---------------------------------------------------------------------------
# Fixture alias (registry lives in conftest but test needs it locally too)
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_registry_fixture(mock_registry):
    return mock_registry
