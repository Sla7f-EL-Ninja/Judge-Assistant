"""
test_evidence.py — Unit tests for classify_evidence_node.

Patch target: nodes.evidence.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.evidence import classify_evidence_node
from schemas import ElementClassification, EvidenceSufficiencyResult


def _make_classification_result(*pairs):
    """Create EvidenceSufficiencyResult from (element_id, status) pairs."""
    return EvidenceSufficiencyResult(classifications=[
        ElementClassification(element_id=eid, status=status, evidence_summary="ملخص الدليل")
        for eid, status in pairs
    ])


def _make_state(**kwargs):
    base = {
        "required_elements": [
            {"element_id": "E1", "description": "وجود عقد صحيح", "element_type": "legal"},
            {"element_id": "E2", "description": "وقوع الإخلال", "element_type": "factual"},
        ],
        "retrieved_facts": "وقائع القضية المسترداة من مستندات الدعوى",
        "law_retrieval_result": {"answer": "نص المادة القانونية المسترد"},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyEvidenceHappy:
    """Happy path: evidence classification succeeds."""

    def test_classifies_all_elements(self):
        mock_result = _make_classification_result(("E1", "established"), ("E2", "disputed"))
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state())

        assert "element_classifications" in output
        assert len(output["element_classifications"]) == 2

    def test_classification_dict_has_required_keys(self):
        mock_result = _make_classification_result(("E1", "established"))
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state(required_elements=[
                {"element_id": "E1", "description": "عنصر", "element_type": "legal"}
            ]))

        c = output["element_classifications"][0]
        assert "element_id" in c
        assert "status" in c
        assert "evidence_summary" in c
        assert "notes" in c

    def test_all_valid_statuses_pass_through(self):
        for status in ["established", "not_established", "disputed", "insufficient_evidence"]:
            mock_result = _make_classification_result(("E1", status))
            llm = MagicMock()
            llm.with_structured_output.return_value.invoke.return_value = mock_result

            with patch("nodes.evidence.get_llm", return_value=llm):
                output = classify_evidence_node(_make_state(required_elements=[
                    {"element_id": "E1", "description": "عنصر", "element_type": "legal"}
                ]))

            assert output["element_classifications"][0]["status"] == status

    def test_intermediate_steps_logged(self):
        mock_result = _make_classification_result(("E1", "established"))
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state(required_elements=[
                {"element_id": "E1", "description": "عنصر", "element_type": "legal"}
            ]))

        assert "intermediate_steps" in output

    def test_prompt_uses_rag_results_not_brief(self):
        """Constraint #2: prompt must use retrieved_facts and law answer, not brief fields."""
        captured = []
        mock_result = _make_classification_result(("E1", "disputed"))
        parser = MagicMock()

        def capture_invoke(prompt):
            captured.append(prompt)
            return mock_result

        parser.invoke.side_effect = capture_invoke
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.evidence.get_llm", return_value=llm):
            classify_evidence_node(_make_state(
                retrieved_facts="وقائع مسترداة خاصة",
                law_retrieval_result={"answer": "نص قانوني خاص"},
            ))

        assert len(captured) == 1
        prompt = captured[0]
        assert "وقائع مسترداة خاصة" in prompt
        assert "نص قانوني خاص" in prompt


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyEvidenceFallback:
    """Fallback: all elements default to 'disputed' on exception."""

    def test_all_disputed_on_exception(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state())

        assert len(output["element_classifications"]) == 2
        for c in output["element_classifications"]:
            assert c["status"] == "disputed"

    def test_error_log_populated_on_fallback(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_fallback_preserves_element_ids(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state())

        ids = {c["element_id"] for c in output["element_classifications"]}
        assert ids == {"E1", "E2"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyEvidenceEdge:
    """Edge cases: empty elements, missing retrieval fields."""

    def test_empty_required_elements_returns_empty(self):
        llm = MagicMock()

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state(required_elements=[]))

        assert output["element_classifications"] == []
        llm.with_structured_output.return_value.invoke.assert_not_called()

    def test_missing_retrieved_facts_uses_fallback_text(self):
        """Missing retrieved_facts → node still runs (uses placeholder in prompt)."""
        mock_result = _make_classification_result(("E1", "disputed"))
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state(retrieved_facts=""))

        assert "element_classifications" in output

    def test_missing_law_answer_uses_fallback_text(self):
        """Missing law_retrieval_result answer → node still runs."""
        mock_result = _make_classification_result(("E1", "disputed"))
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.evidence.get_llm", return_value=llm):
            output = classify_evidence_node(_make_state(law_retrieval_result={}))

        assert "element_classifications" in output
