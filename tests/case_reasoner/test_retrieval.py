"""
test_retrieval.py — Unit tests for retrieve_law_node and retrieve_facts_node.

Patch targets: tools.civil_law_rag_tool, tools.case_documents_rag_tool
(Tools imported lazily inside node functions via `from tools import X`)
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.retrieval import retrieve_law_node, retrieve_facts_node


def _make_law_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "source_text": "نص مصدري",
        "case_id": "test-001",
    }
    base.update(kwargs)
    return base


def _make_fact_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "source_text": "نص مصدري",
        "case_id": "test-001",
    }
    base.update(kwargs)
    return base


def _make_civil_law_result(answer="نص قانوني مسترد", sources=None):
    return {
        "answer": answer,
        "sources": sources or [{"article": 148, "content": "نص المادة 148", "title": "عقد"}],
        "classification": "",
        "retrieval_confidence": 0.9,
        "citation_integrity": 0.9,
        "from_cache": False,
        "error": None,
    }


def _make_fact_result(final_answer="وقائع القضية المسترداة"):
    return {
        "final_answer": final_answer,
        "sub_answers": [],
        "error": None,
    }


# ---------------------------------------------------------------------------
# retrieve_law_node
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveLawNode:
    """T-RETRIEVAL-01: retrieve_law_node calls only civil_law_rag_tool."""

    def test_returns_law_retrieval_result_and_articles(self):
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
            output = retrieve_law_node(_make_law_state())

        assert "law_retrieval_result" in output
        assert "retrieved_articles" in output
        assert len(output["retrieved_articles"]) == 1
        assert output["retrieved_articles"][0]["article_number"] == 148

    def test_parse_articles_normalizes_arabic_digits(self):
        result = _make_civil_law_result(sources=[
            {"article": "٢٤٨", "content": "نص المادة", "title": "عقد"},
        ])
        with patch("tools.civil_law_rag_tool", return_value=result):
            output = retrieve_law_node(_make_law_state())

        assert output["retrieved_articles"][0]["article_number"] == 248

    def test_query_uses_issue_title_and_source_text(self):
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_tool:
            retrieve_law_node(_make_law_state(
                issue_title="مسألة التعويض",
                source_text="نص الخلاف",
            ))

        call_args = mock_tool.call_args[0][0]  # first positional arg
        assert "مسألة التعويض" in call_args
        assert "نص الخلاف" in call_args

    def test_query_truncated_to_500_chars(self):
        long_title = "ع" * 300
        long_source = "ن" * 300
        captured = []

        def capture_call(query):
            captured.append(query)
            return _make_civil_law_result()

        with patch("tools.civil_law_rag_tool", side_effect=capture_call):
            retrieve_law_node(_make_law_state(issue_title=long_title, source_text=long_source))

        assert len(captured[0]) <= 500

    def test_empty_result_on_tool_failure(self):
        with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_law_node(_make_law_state())

        assert output["law_retrieval_result"]["answer"] == ""
        assert output["retrieved_articles"] == []

    def test_error_log_on_failure(self):
        with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_law_node(_make_law_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_intermediate_steps_logged(self):
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()):
            output = retrieve_law_node(_make_law_state())

        assert "intermediate_steps" in output

    def test_law_tool_only_civil_law_rag_called(self):
        """Constraint #3: law retrieval node NEVER calls case_documents_rag_tool."""
        with patch("tools.civil_law_rag_tool", return_value=_make_civil_law_result()) as mock_law, \
             patch("tools.case_documents_rag_tool") as mock_fact:
            retrieve_law_node(_make_law_state())

        mock_law.assert_called_once()
        mock_fact.assert_not_called()

    def test_multiple_articles_parsed(self):
        result = _make_civil_law_result(sources=[
            {"article": 148, "content": "نص 148"},
            {"article": 176, "content": "نص 176"},
        ])
        with patch("tools.civil_law_rag_tool", return_value=result):
            output = retrieve_law_node(_make_law_state())

        nums = {a["article_number"] for a in output["retrieved_articles"]}
        assert nums == {148, 176}


# ---------------------------------------------------------------------------
# retrieve_facts_node
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRetrieveFactsNode:
    """T-RETRIEVAL-02: retrieve_facts_node calls only case_documents_rag_tool."""

    def test_returns_retrieved_facts(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result("وقائع مهمة")):
            output = retrieve_facts_node(_make_fact_state())

        assert "retrieved_facts" in output
        assert output["retrieved_facts"] == "وقائع مهمة"

    def test_passes_case_id_to_tool(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_tool:
            retrieve_facts_node(_make_fact_state(case_id="CASE-XYZ-007"))

        call_kwargs = mock_tool.call_args
        case_id_passed = call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1].get("case_id")
        assert case_id_passed == "CASE-XYZ-007"

    def test_partial_error_logged_as_warning(self, caplog):
        import logging
        result = _make_fact_result()
        result["error"] = "نتيجة جزئية"
        with patch("tools.case_documents_rag_tool", return_value=result):
            with caplog.at_level(logging.WARNING):
                retrieve_facts_node(_make_fact_state())
        # Warning should be logged but node should not fail

    def test_empty_result_on_tool_failure(self):
        with patch("tools.case_documents_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_facts_node(_make_fact_state())

        assert output["retrieved_facts"] == ""
        assert output["fact_retrieval_result"]["final_answer"] == ""

    def test_error_log_on_failure(self):
        with patch("tools.case_documents_rag_tool", side_effect=RuntimeError("tool error")):
            output = retrieve_facts_node(_make_fact_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_intermediate_steps_logged(self):
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()):
            output = retrieve_facts_node(_make_fact_state())

        assert "intermediate_steps" in output

    def test_fact_node_only_calls_case_doc_tool(self):
        """Constraint #3: fact retrieval node NEVER calls civil_law_rag_tool."""
        with patch("tools.case_documents_rag_tool", return_value=_make_fact_result()) as mock_fact, \
             patch("tools.civil_law_rag_tool") as mock_law:
            retrieve_facts_node(_make_fact_state())

        mock_fact.assert_called_once()
        mock_law.assert_not_called()

    def test_returns_fact_retrieval_result(self):
        fact_result = {"final_answer": "وقائع", "sub_answers": ["جزء أول"], "error": None}
        with patch("tools.case_documents_rag_tool", return_value=fact_result):
            output = retrieve_facts_node(_make_fact_state())

        assert "fact_retrieval_result" in output
        assert output["fact_retrieval_result"]["final_answer"] == "وقائع"
