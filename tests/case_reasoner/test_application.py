"""
test_application.py — Unit tests for apply_law_node.

Patch target: nodes.application.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.application import apply_law_node
from schemas import AppliedElement, LawApplicationResult


def _make_law_result(*elements, synthesis="تحليل إجمالي محايد"):
    return LawApplicationResult(
        elements=list(elements),
        synthesis=synthesis,
    )


def _make_state(**kwargs):
    base = {
        "required_elements": [
            {"element_id": "E1", "description": "وجود عقد صحيح", "element_type": "legal"},
            {"element_id": "E2", "description": "وقوع الإخلال", "element_type": "factual"},
        ],
        "element_classifications": [
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "established"},
        ],
        "retrieved_facts": "وقائع القضية المسترداة",
        "law_retrieval_result": {"answer": "نص المادة القانونية المسترد"},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApplyLawHappy:
    """Happy path: law application returns applied elements."""

    def test_returns_applied_elements(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل E1", cited_articles=[148]),
            AppliedElement(element_id="E2", reasoning="تحليل E2", cited_articles=[176]),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        assert "applied_elements" in output
        assert len(output["applied_elements"]) == 2

    def test_applied_element_dict_has_required_keys(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        el = output["applied_elements"][0]
        assert "element_id" in el
        assert "reasoning" in el
        assert "cited_articles" in el

    def test_law_application_synthesis_preserved(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
            synthesis="التحليل الإجمالي المحايد للمسألة",
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        assert output["law_application"] == "التحليل الإجمالي المحايد للمسألة"

    def test_intermediate_steps_logged(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        assert "intermediate_steps" in output

    def test_skipped_elements_empty_when_none_insufficient(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        assert output["skipped_elements"] == []


# ---------------------------------------------------------------------------
# Insufficient evidence skipping (Constraint #5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInsufficientEvidenceSkipping:
    """Constraint #5: insufficient_evidence elements are skipped, not classified False."""

    def test_skips_insufficient_evidence_element(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        state = _make_state(element_classifications=[
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "insufficient_evidence"},
        ])

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(state)

        applied_ids = {el["element_id"] for el in output["applied_elements"]}
        assert "E2" not in applied_ids
        assert "E2" in output["skipped_elements"]

    def test_skipped_elements_listed_in_output(self):
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        state = _make_state(element_classifications=[
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "insufficient_evidence"},
        ])

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(state)

        assert "E2" in output["skipped_elements"]
        assert "skipped_elements" in output

    def test_all_elements_insufficient_skips_llm_call(self):
        """Constraint #5: when all elements are insufficient, LLM never called."""
        llm = MagicMock()

        state = _make_state(element_classifications=[
            {"element_id": "E1", "status": "insufficient_evidence"},
            {"element_id": "E2", "status": "insufficient_evidence"},
        ])

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(state)

        llm.with_structured_output.return_value.invoke.assert_not_called()
        assert output["applied_elements"] == []
        assert set(output["skipped_elements"]) == {"E1", "E2"}

    def test_mixed_statuses_active_elements_only_in_prompt(self):
        """2 active + 1 insufficient → only 2 processed."""
        captured = []
        mock_result = _make_law_result(
            AppliedElement(element_id="E1", reasoning="تحليل", cited_articles=[148]),
            AppliedElement(element_id="E2", reasoning="تحليل", cited_articles=[176]),
        )

        parser = MagicMock()

        def capture(prompt):
            captured.append(prompt)
            return mock_result

        parser.invoke.side_effect = capture
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        state = _make_state(element_classifications=[
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "disputed"},
            {"element_id": "E3", "status": "insufficient_evidence"},
        ], required_elements=[
            {"element_id": "E1", "description": "عنصر 1", "element_type": "legal"},
            {"element_id": "E2", "description": "عنصر 2", "element_type": "factual"},
            {"element_id": "E3", "description": "عنصر 3", "element_type": "factual"},
        ])

        with patch("nodes.application.get_llm", return_value=llm):
            apply_law_node(state)

        # E3 (insufficient_evidence) must not appear in prompt
        assert "E3" not in captured[0]


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApplyLawFallback:
    """Fallback behavior on LLM exception."""

    def test_error_message_returned_on_exception(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        assert "law_application" in output
        assert len(output["law_application"]) > 0  # error message
        assert output["applied_elements"] == []

    def test_skipped_elements_preserved_on_failure(self):
        """Even when LLM fails, insufficient elements still appear in skipped_elements."""
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        state = _make_state(element_classifications=[
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "insufficient_evidence"},
        ])

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(state)

        assert "E2" in output["skipped_elements"]

    def test_error_log_populated_on_failure(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.application.get_llm", return_value=llm):
            output = apply_law_node(_make_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0
