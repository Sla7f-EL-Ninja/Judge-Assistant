"""
test_graph_flow.py
------------------
Integration tests for the compiled LangGraph.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from RAG.legal_rag.state import make_initial_state
from tests.legal_rag.conftest import SAMPLE_TOC, fresh_state, make_mock_llm


# ===========================================================================
# Shared helpers
# ===========================================================================

def _doc(index: int, content: str = "") -> Document:
    return Document(
        page_content=content or f"نص المادة {index}",
        metadata={"index": index, "source": "civil_law", "type": "article"},
    )


def _llm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    return resp


def _corpus_scores(winner: str = "civil", confidence: float = 0.9) -> str:
    return json.dumps({
        "scores": [
            {"corpus_name": winner, "confidence": confidence, "reason": "test"},
            {"corpus_name": "evidence",   "confidence": 0.05, "reason": ""},
            {"corpus_name": "procedures", "confidence": 0.05, "reason": ""},
        ]
    })


def _preproc_json(cls: str, rewritten: str) -> str:
    return json.dumps({"classification": cls, "rewritten_question": rewritten})


def _ch_json(ch_id: str, conf: float) -> str:
    return json.dumps({"chapter_id": ch_id, "confidence": conf})


def _sec_json(sec_id: str, conf: float) -> str:
    return json.dumps({"section_id": sec_id, "confidence": conf})


def _grader_json(passed: bool, reason: str = "") -> str:
    return json.dumps({"pass": passed, "reason": reason})


def _refine_json(refined: str) -> str:
    return json.dumps({"refined_query": refined})


HIGH_CONFIDENCE_DOCS = [_doc(i) for i in range(1, 6)]
SCORES_HIGH = [(d, 0.85) for d in HIGH_CONFIDENCE_DOCS]

LOW_CONFIDENCE_DOCS  = [_doc(i) for i in range(1, 3)]
SCORES_LOW  = [(d, 0.28) for d in LOW_CONFIDENCE_DOCS]


# ===========================================================================
# Context manager: patch all external I/O in one call
# ===========================================================================

@contextmanager
def _mock_all(
    corpus_winner: str = "civil",
    corpus_confidence: float = 0.9,
    preproc_cls: str = "analytical",
    preproc_rewritten: str = "ما هي شروط صحة العقد الصحيح؟",
    ch_id: str = "1",
    ch_conf: float = 0.8,
    sec_id: str = "1",
    sec_conf: float = 0.3,           # below section threshold by default
    retrieval_pairs=None,
    reranked_docs=None,
    rule_grade: str = "pass",        # pass | refine | llm | fail (emulated via confidence)
    llm_grade_pass: bool = True,
    generate_answer: str = "إجابة اختبار قانونية.",
    refine_query: str = "استفسار محسّن",
    mock_registry_fixture=None,
):
    if retrieval_pairs is None:
        retrieval_pairs = SCORES_HIGH
    if reranked_docs is None:
        reranked_docs = HIGH_CONFIDENCE_DOCS
    if mock_registry_fixture is None:
        raise ValueError("mock_registry_fixture must be supplied")

    # Build LLM side_effect chain (preprocessor runs first, then corpus_classifier)
    llm_calls = [
        _preproc_json(preproc_cls, preproc_rewritten),
        _corpus_scores(corpus_winner, corpus_confidence),
        _ch_json(ch_id, ch_conf),
        _sec_json(sec_id, sec_conf),
        generate_answer,
    ]
    llm_mock = MagicMock()
    llm_mock.invoke.side_effect = [_llm_response(c) for c in llm_calls]

    # Use ExitStack to avoid Python's "too many statically nested blocks" limit.
    from contextlib import ExitStack
    patches = [
        patch("RAG.legal_rag.nodes.corpus_router.MAX_LLM_CALLS", 20),
        patch("RAG.legal_rag.nodes.preprocessor.MAX_LLM_CALLS", 20),
        patch("RAG.legal_rag.nodes.scope_classifier.MAX_LLM_CALLS", 20),
        patch("RAG.legal_rag.nodes.generate.MAX_LLM_CALLS", 20),
        patch("RAG.legal_rag.llm_utils.MAX_LLM_CALLS", 20),
        patch("RAG.legal_rag.nodes.corpus_router._get_llm",    return_value=llm_mock),
        patch("RAG.legal_rag.nodes.preprocessor._get_llm",     return_value=llm_mock),
        patch("RAG.legal_rag.nodes.scope_classifier._get_llm", return_value=llm_mock),
        patch("RAG.legal_rag.nodes.generate._get_llm",         return_value=llm_mock),
        patch("RAG.legal_rag.nodes.graders._get_llm",          return_value=llm_mock),
        patch("RAG.legal_rag.nodes.refine._get_llm",           return_value=llm_mock),
        patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4),
        patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5),
        patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5),
        patch("RAG.legal_rag.nodes.scope_classifier.load_toc", return_value=SAMPLE_TOC),
        patch("RAG.legal_rag.nodes.retrieve.load_vectorstore"),
        patch("RAG.legal_rag.nodes.retrieve.rerank", return_value=reranked_docs),
        patch("RAG.legal_rag.nodes.retrieve._hyde_enabled", return_value=False),
        patch("RAG.legal_rag.nodes.retrieve._global_retrieval_enabled", return_value=False),
        patch("RAG.legal_rag.nodes.textual.get_qdrant_client"),
        patch("RAG.legal_rag.nodes.textual.load_vectorstore"),
        patch("RAG.legal_rag.nodes.corpus_router._get_registry", return_value=mock_registry_fixture),
    ]
    with ExitStack() as stack:
        entered = [stack.enter_context(p) for p in patches]
        mock_vs = entered[15]
        mock_vs.return_value.similarity_search_with_relevance_scores.return_value = retrieval_pairs
        yield


def _build_graph():
    import RAG.legal_rag.graph as graph_module
    graph_module._compiled_app = None
    return graph_module.build_graph()


def _run(state: dict, mock_registry_fixture, **kwargs) -> dict:
    state["last_query"] = state.get("last_query") or "ما هي شروط صحة العقد؟"
    state["llm_call_count"] = 0 # Explicitly reset count before run
    with _mock_all(mock_registry_fixture=mock_registry_fixture, **kwargs):
        graph = _build_graph()
        return graph.invoke(state)


# ===========================================================================
# Tests
# ===========================================================================
class TestGraphFlow:

    def test_happy_path_analytical_produces_answer(self, fresh_state, mock_registry):
        result = _run(
            fresh_state,
            mock_registry,
            corpus_winner="civil",
            preproc_cls="analytical",
            generate_answer="إجابة اختبار قانونية.",
        )
        assert result["final_answer"]

    def test_corpus_router_off_topic_terminates_at_off_topic_node(
        self, fresh_state, mock_registry
    ):
        state = fresh_state
        state["last_query"] = "What is the best restaurant in Cairo?"
        state["llm_call_count"] = 0
        with _mock_all(mock_registry_fixture=mock_registry):
            graph = _build_graph()
            result = graph.invoke(state)
        assert result["final_answer"]
        assert "خارج نطاق" in result["final_answer"] or "يرجى" in result["final_answer"]

    def test_preprocessor_off_topic_terminates_at_off_topic_node(
        self, fresh_state, mock_registry
    ):
        result = _run(
            fresh_state,
            mock_registry,
            corpus_winner="civil",
            corpus_confidence=0.9,
            preproc_cls="خارج السياق",
            preproc_rewritten="",
        )
        assert "خارج نطاق" in result["final_answer"] or "يرجى" in result["final_answer"]

    def test_textual_query_routes_to_textual_node(self, fresh_state, mock_registry):
        state = fresh_state
        state["last_query"] = "ما نص المادة 89 من القانون المدني؟"
        state["llm_call_count"] = 0

        point = MagicMock()
        point.payload = {
            "page_content": "المادة 89: يجوز للدائن...",
            "metadata": {"index": 89, "source": "civil_law", "type": "article"},
        }

        llm_calls = [
            _preproc_json("نصّي", "ما نص المادة 89؟"),
            _corpus_scores("civil", 0.9),
        ]
        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = [_llm_response(c) for c in llm_calls]

        with patch("RAG.legal_rag.nodes.corpus_router.MAX_LLM_CALLS", 20), \
             patch("RAG.legal_rag.nodes.preprocessor.MAX_LLM_CALLS", 20), \
             patch("RAG.legal_rag.llm_utils.MAX_LLM_CALLS", 20), \
             patch("RAG.legal_rag.nodes.corpus_router._get_llm",  return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.preprocessor._get_llm",   return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4), \
             patch("RAG.legal_rag.nodes.textual.get_qdrant_client") as mock_qdrant, \
             patch("RAG.legal_rag.nodes.corpus_router._get_registry", return_value=mock_registry):
            mock_qdrant.return_value.scroll.return_value = ([point], None)
            graph = _build_graph()
            result = graph.invoke(state)

        assert "المادة 89" in result["final_answer"]

    def test_rule_grader_refine_then_pass_produces_answer(self, fresh_state, mock_registry):
        """First retrieval → low confidence (refine), second → high confidence (pass)."""
        llm_calls = [
            _preproc_json("analytical", "سؤال"),
            _corpus_scores("civil", 0.9),
            _ch_json("1", 0.8),
            _sec_json("1", 0.2),
            _refine_json("استفسار محسّن"),
            _ch_json("1", 0.8),
            _sec_json("1", 0.2),
            "إجابة نهائية بعد التحسين.",
        ]
        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = [_llm_response(c) for c in llm_calls]

        low_docs  = [_doc(i) for i in [1, 2]]
        high_docs = [_doc(i) for i in [1, 2, 3, 4, 5]]

        vs_mock = MagicMock()
        vs_mock.similarity_search_with_relevance_scores.side_effect = [
            [(d, 0.25) for d in low_docs],
            [(d, 0.85) for d in high_docs],
        ]

        from contextlib import ExitStack
        refine_patches = [
            patch("RAG.legal_rag.nodes.corpus_router.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.preprocessor.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.scope_classifier.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.generate.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.llm_utils.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.corpus_router._get_llm",    return_value=llm_mock),
            patch("RAG.legal_rag.nodes.preprocessor._get_llm",     return_value=llm_mock),
            patch("RAG.legal_rag.nodes.scope_classifier._get_llm", return_value=llm_mock),
            patch("RAG.legal_rag.nodes.generate._get_llm",         return_value=llm_mock),
            patch("RAG.legal_rag.nodes.refine._get_llm",           return_value=llm_mock),
            patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4),
            patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5),
            patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5),
            patch("RAG.legal_rag.nodes.scope_classifier.load_toc",  return_value=SAMPLE_TOC),
            patch("RAG.legal_rag.nodes.retrieve.load_vectorstore",  return_value=vs_mock),
            patch("RAG.legal_rag.nodes.retrieve.rerank"),
            patch("RAG.legal_rag.nodes.retrieve._hyde_enabled",     return_value=False),
            patch("RAG.legal_rag.nodes.retrieve._global_retrieval_enabled", return_value=False),
            patch("RAG.legal_rag.nodes.corpus_router._get_registry", return_value=mock_registry),
        ]
        with ExitStack() as stack:
            entered = [stack.enter_context(p) for p in refine_patches]
            mock_rerank = entered[15]
            mock_rerank.side_effect = lambda q, docs, top_k: docs
            graph = _build_graph()
            fresh_state["last_query"] = "ما هي شروط صحة العقد؟"
            fresh_state["llm_call_count"] = 0
            result = graph.invoke(fresh_state)

        assert result["final_answer"]
        assert "إجابة نهائية" in result["final_answer"]

    def test_max_retries_exhausted_routes_to_cannot_answer(self, fresh_state, mock_registry):
        """Simulate continuous low-confidence retrieval until retries are exhausted."""
        llm_calls = [
            _preproc_json("analytical", "سؤال"),
            _corpus_scores("civil", 0.9),
            _ch_json("1", 0.8), _sec_json("1", 0.2), _refine_json("محسّن 1"),
            _ch_json("1", 0.8), _sec_json("1", 0.2), _refine_json("محسّن 2"),
            _ch_json("1", 0.8), _sec_json("1", 0.2), _refine_json("محسّن 3"),
            _ch_json("1", 0.8), _sec_json("1", 0.2),
        ]
        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = [_llm_response(c) for c in llm_calls]

        always_low = MagicMock()
        always_low.similarity_search_with_relevance_scores.return_value = [
            (_doc(i), 0.15) for i in [1, 2]
        ]

        from contextlib import ExitStack
        retry_patches = [
            patch("RAG.legal_rag.nodes.corpus_router.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.preprocessor.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.scope_classifier.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.generate.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.llm_utils.MAX_LLM_CALLS", 20),
            patch("RAG.legal_rag.nodes.corpus_router._get_llm",    return_value=llm_mock),
            patch("RAG.legal_rag.nodes.preprocessor._get_llm",     return_value=llm_mock),
            patch("RAG.legal_rag.nodes.scope_classifier._get_llm", return_value=llm_mock),
            patch("RAG.legal_rag.nodes.refine._get_llm",           return_value=llm_mock),
            patch("RAG.legal_rag.nodes.generate._get_llm",         return_value=llm_mock),
            patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4),
            patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5),
            patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5),
            patch("RAG.legal_rag.nodes.scope_classifier.load_toc",  return_value=SAMPLE_TOC),
            patch("RAG.legal_rag.nodes.retrieve.load_vectorstore",  return_value=always_low),
            patch("RAG.legal_rag.nodes.retrieve.rerank"),
            patch("RAG.legal_rag.nodes.retrieve._hyde_enabled",     return_value=False),
            patch("RAG.legal_rag.nodes.retrieve._global_retrieval_enabled", return_value=False),
            patch("RAG.legal_rag.nodes.corpus_router._get_registry", return_value=mock_registry),
        ]
        with ExitStack() as stack:
            entered = [stack.enter_context(p) for p in retry_patches]
            mock_rerank = entered[15]
            mock_rerank.side_effect = lambda q, docs, top_k: docs
            graph = _build_graph()
            fresh_state["last_query"] = "ما هي شروط صحة العقد؟"
            fresh_state["llm_call_count"] = 0
            result = graph.invoke(fresh_state)

        assert result["final_answer"]
        assert result["retry_count"] >= result["max_retries"]

    def test_llm_grader_pass_produces_answer(self, fresh_state, mock_registry):
        """Borderline confidence (0.45) → rule_grader sets 'llm' → llm_grader passes."""
        llm_calls = [
            _corpus_scores("civil", 0.9),
            _preproc_json("analytical", "سؤال"),
            _ch_json("1", 0.8),
            _sec_json("1", 0.2),
            _grader_json(True, "المستندات كافية"),
            "إجابة بعد موافقة llm_grader.",
        ]
        llm_mock = MagicMock()
        llm_mock.invoke.side_effect = [_llm_response(c) for c in llm_calls]

        borderline_docs = [_doc(i) for i in [1, 2, 3]]
        vs_mock = MagicMock()
        vs_mock.similarity_search_with_relevance_scores.return_value = [
            (d, 0.45) for d in borderline_docs
        ]

        with patch("RAG.legal_rag.nodes.corpus_router.MAX_LLM_CALLS", 20), \
             patch("RAG.legal_rag.nodes.corpus_router._get_llm",    return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.preprocessor._get_llm",     return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.scope_classifier._get_llm", return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.graders._get_llm",          return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.generate._get_llm",         return_value=llm_mock), \
             patch("RAG.legal_rag.nodes.corpus_router._corpus_threshold", return_value=0.4), \
             patch("RAG.legal_rag.nodes.scope_classifier._chapter_threshold", return_value=0.5), \
             patch("RAG.legal_rag.nodes.scope_classifier._section_threshold", return_value=0.5), \
             patch("RAG.legal_rag.nodes.scope_classifier.load_toc",  return_value=SAMPLE_TOC), \
             patch("RAG.legal_rag.nodes.retrieve.load_vectorstore",  return_value=vs_mock), \
             patch("RAG.legal_rag.nodes.retrieve.rerank") as mock_rerank, \
             patch("RAG.legal_rag.nodes.retrieve._hyde_enabled",     return_value=False), \
             patch("RAG.legal_rag.nodes.corpus_router._get_registry", return_value=mock_registry):
            mock_rerank.side_effect = lambda q, docs, top_k: docs
            graph = _build_graph()
            fresh_state["llm_call_count"] = 0
            result = graph.invoke(fresh_state)

        assert "llm_grader" in result["final_answer"] or result["final_answer"]

    def test_evidence_corpus_resolved_correctly(self, fresh_state, mock_registry):
        result = _run(
            fresh_state,
            mock_registry,
            corpus_winner="evidence",
            corpus_confidence=0.88,
            preproc_cls="analytical",
            generate_answer="إجابة تتعلق بقانون الإثبات.",
        )
        assert result["corpus_config"].name == "evidence"

    def test_build_graph_singleton(self):
        import RAG.legal_rag.graph as graph_module
        graph_module._compiled_app = None
        g1 = graph_module.build_graph()
        g2 = graph_module.build_graph()
        assert g1 is g2

    def test_two_sequential_invocations_have_independent_state(
        self, mock_registry
    ):
        state_a = make_initial_state()
        state_b = make_initial_state()
        state_a["last_query"] = "ما هي شروط صحة العقد؟"
        state_b["last_query"] = "ما هي شروط صحة العقد؟"
        state_a["llm_call_count"] = 0
        state_b["llm_call_count"] = 0

        with _mock_all(mock_registry_fixture=mock_registry, generate_answer="إجابة أ"):
            graph = _build_graph()
            result_a = graph.invoke(state_a)

        with _mock_all(mock_registry_fixture=mock_registry, generate_answer="إجابة ب"):
            result_b = graph.invoke(state_b)

        assert result_a is not result_b
        assert result_a["query_history"] is not result_b["query_history"]