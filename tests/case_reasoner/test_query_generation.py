"""
test_query_generation.py — Unit tests for generate_retrieval_queries_node.

Patch target: nodes.query_generation.get_llm

The query generation node sits between decompose_issue and retrieve_law/retrieve_facts.
It produces per-element law_queries and fact_queries lists.
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.query_generation import generate_retrieval_queries_node
from schemas import ElementQuery, RetrievalQueries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_element(eid="E1", desc="وجود عقد صحيح ملزم", etype="legal"):
    return {"element_id": eid, "description": desc, "element_type": etype}


def _make_element_query(eid, law_q, fact_q):
    return ElementQuery(element_id=eid, law_query=law_q, fact_query=fact_q)


def _make_retrieval_queries(*element_queries):
    return RetrievalQueries(queries=list(element_queries))


def _make_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "legal_domain": "العقود المدنية",
        "source_text": "نص مصدري من الملخص",
        "required_elements": [
            _make_element("E1", "وجود عقد صحيح", "legal"),
            _make_element("E2", "وقوع الإخلال", "factual"),
        ],
    }
    base.update(kwargs)
    return base


def _make_mock_result(queries):
    """Build a RetrievalQueries mock result from a list of (eid, law_q, fact_q) tuples."""
    return _make_retrieval_queries(*[
        _make_element_query(eid, lq, fq) for eid, lq, fq in queries
    ])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryGenerationHappy:
    """T-QUERY-GEN-01: generates per-element law and fact queries."""

    def test_returns_law_queries_and_fact_queries(self):
        mock_result = _make_mock_result([
            ("E1", "ما هي أحكام العقد في القانون المدني؟", "ما الوقائع المتعلقة بالعقد؟"),
            ("E2", "ما هي أحكام الإخلال بالعقد؟", "ما الوقائع المتعلقة بالإخلال؟"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        assert "law_queries" in output
        assert "fact_queries" in output

    def test_query_count_matches_element_count(self):
        mock_result = _make_mock_result([
            ("E1", "سؤال قانوني 1", "سؤال وقائعي 1"),
            ("E2", "سؤال قانوني 2", "سؤال وقائعي 2"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        assert len(output["law_queries"]) == 2
        assert len(output["fact_queries"]) == 2

    def test_each_query_dict_has_element_id_and_query(self):
        mock_result = _make_mock_result([
            ("E1", "سؤال قانوني", "سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1")]
            ))

        lq = output["law_queries"][0]
        fq = output["fact_queries"][0]
        assert "element_id" in lq and "query" in lq
        assert "element_id" in fq and "query" in fq

    def test_element_ids_preserved_in_output(self):
        mock_result = _make_mock_result([
            ("E1", "سؤال قانوني 1", "سؤال وقائعي 1"),
            ("E2", "سؤال قانوني 2", "سؤال وقائعي 2"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        law_ids = {q["element_id"] for q in output["law_queries"]}
        fact_ids = {q["element_id"] for q in output["fact_queries"]}
        assert law_ids == {"E1", "E2"}
        assert fact_ids == {"E1", "E2"}

    def test_law_query_and_fact_query_are_different(self):
        """Law query must be doctrinal; fact query must be factual — they must differ."""
        mock_result = _make_mock_result([
            ("E1", "ما هي أحكام العقد في القانون المدني المصري؟",
                   "ما الوقائع المتعلقة بتوقيع العقد في مستندات القضية؟"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1")]
            ))

        assert output["law_queries"][0]["query"] != output["fact_queries"][0]["query"]

    def test_prompt_includes_issue_title(self):
        """Prompt must contain issue_title for context."""
        captured = []
        mock_result = _make_mock_result([("E1", "سؤال", "سؤال")])
        parser = MagicMock()

        def capture(prompt):
            captured.append(prompt)
            return mock_result

        parser.invoke.side_effect = capture
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.query_generation.get_llm", return_value=llm):
            generate_retrieval_queries_node(_make_state(
                issue_title="مسألة التعويض الخاصة",
                required_elements=[_make_element("E1")],
            ))

        assert "مسألة التعويض الخاصة" in captured[0]

    def test_prompt_includes_all_element_descriptions(self):
        """All element descriptions appear in prompt."""
        captured = []
        mock_result = _make_mock_result([
            ("E1", "سؤال 1", "سؤال 1"),
            ("E2", "سؤال 2", "سؤال 2"),
        ])
        parser = MagicMock()

        def capture(prompt):
            captured.append(prompt)
            return mock_result

        parser.invoke.side_effect = capture
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.query_generation.get_llm", return_value=llm):
            generate_retrieval_queries_node(_make_state())

        assert "وجود عقد صحيح" in captured[0]
        assert "وقوع الإخلال" in captured[0]

    def test_node_never_calls_rag_tools(self):
        """Query generation node must NOT call civil_law_rag_tool or case_documents_rag_tool."""
        mock_result = _make_mock_result([("E1", "سؤال", "سؤال")])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm), \
             patch("tools.civil_law_rag_tool") as mock_law, \
             patch("tools.case_documents_rag_tool") as mock_fact:
            generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1")]
            ))

        mock_law.assert_not_called()
        mock_fact.assert_not_called()

    def test_intermediate_steps_logged(self):
        mock_result = _make_mock_result([("E1", "سؤال", "سؤال")])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1")]
            ))

        assert "intermediate_steps" in output
        assert len(output["intermediate_steps"]) > 0

    def test_uses_get_prompt_query_generation(self):
        """Node must use get_prompt('query_generation') — not inline strings."""
        mock_result = _make_mock_result([("E1", "سؤال", "سؤال")])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm), \
             patch("nodes.query_generation.get_prompt", return_value=("sys", "user: {issue_title} {legal_domain} {source_text} {elements_text}")) as mock_get_prompt:
            generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1")]
            ))

        mock_get_prompt.assert_called_once_with("query_generation")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryGenerationEdge:
    """T-QUERY-GEN-02: Edge cases."""

    def test_empty_required_elements_returns_empty_lists(self):
        """No elements → no LLM call, empty lists returned."""
        llm = MagicMock()

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(required_elements=[]))

        assert output["law_queries"] == []
        assert output["fact_queries"] == []
        llm.with_structured_output.return_value.invoke.assert_not_called()

    def test_single_element_produces_one_query_pair(self):
        mock_result = _make_mock_result([
            ("E1", "سؤال قانوني", "سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1")]
            ))

        assert len(output["law_queries"]) == 1
        assert len(output["fact_queries"]) == 1

    def test_five_elements_produce_five_query_pairs(self):
        queries = [(f"E{i}", f"سؤال قانوني {i}", f"سؤال وقائعي {i}") for i in range(1, 6)]
        mock_result = _make_mock_result(queries)
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        elements = [_make_element(f"E{i}", f"عنصر {i}") for i in range(1, 6)]

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(required_elements=elements))

        assert len(output["law_queries"]) == 5
        assert len(output["fact_queries"]) == 5


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryGenerationFallback:
    """T-QUERY-GEN-03: Fallback to generic queries on LLM exception."""

    def test_fallback_returns_queries_on_llm_exception(self):
        """LLM raises → fallback generic queries generated, no crash."""
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        assert "law_queries" in output
        assert "fact_queries" in output
        assert len(output["law_queries"]) == 2
        assert len(output["fact_queries"]) == 2

    def test_fallback_queries_contain_element_descriptions(self):
        """Fallback law queries use element description + issue title."""
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state(
                required_elements=[_make_element("E1", "وجود عقد صحيح ملزم")]
            ))

        # Fallback query should reference the element description somehow
        lq = output["law_queries"][0]["query"]
        assert "وجود عقد صحيح ملزم" in lq or "التعويض" in lq

    def test_fallback_preserves_element_ids(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        law_ids = {q["element_id"] for q in output["law_queries"]}
        assert "E1" in law_ids
        assert "E2" in law_ids

    def test_error_log_populated_on_fallback(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_intermediate_steps_logged_on_fallback(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.query_generation.get_llm", return_value=llm):
            output = generate_retrieval_queries_node(_make_state())

        assert "intermediate_steps" in output
