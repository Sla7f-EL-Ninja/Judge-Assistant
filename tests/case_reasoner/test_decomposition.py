"""
test_decomposition.py — Unit tests for decompose_issue_node.

Patch target: nodes.decomposition.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.decomposition import decompose_issue_node
from schemas import DecomposedIssue, RequiredElement


def _make_decomposed(*elements):
    return DecomposedIssue(elements=list(elements))


def _make_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "legal_domain": "العقود المدنية",
        "source_text": "نص مصدري من الملخص",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDecomposeIssueHappy:
    """Happy path: decomposition returns structured elements."""

    def test_returns_required_elements(self):
        mock_result = _make_decomposed(
            RequiredElement(element_id="E1", description="وجود عقد صحيح", element_type="legal"),
            RequiredElement(element_id="E2", description="وقوع الإخلال", element_type="factual"),
            RequiredElement(element_id="E3", description="وجود الضرر", element_type="factual"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        assert "required_elements" in output
        assert len(output["required_elements"]) == 3

    def test_element_dict_has_required_keys(self):
        mock_result = _make_decomposed(
            RequiredElement(element_id="E1", description="وجود عقد", element_type="legal"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        el = output["required_elements"][0]
        assert "element_id" in el
        assert "description" in el
        assert "element_type" in el

    def test_element_ids_preserved(self):
        mock_result = _make_decomposed(
            RequiredElement(element_id="E42", description="عنصر", element_type="legal"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        assert output["required_elements"][0]["element_id"] == "E42"

    def test_intermediate_steps_logged(self):
        mock_result = _make_decomposed(
            RequiredElement(element_id="E1", description="عنصر", element_type="legal"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        assert "intermediate_steps" in output
        assert len(output["intermediate_steps"]) > 0


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDecomposeIssueFallback:
    """Fallback: single E0 element when LLM raises."""

    def test_fallback_single_element_on_exception(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM error")

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state(issue_title="العنوان المعطى"))

        elements = output["required_elements"]
        assert len(elements) == 1
        assert elements[0]["element_id"] == "E0"
        assert "العنوان المعطى" in elements[0]["description"]
        assert elements[0]["element_type"] == "legal"

    def test_error_log_populated_on_fallback(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM error")

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_intermediate_steps_logged_on_fallback(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM error")

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        assert "intermediate_steps" in output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDecomposeIssueEdge:
    """Edge cases: empty input fields."""

    def test_empty_issue_title(self):
        mock_result = _make_decomposed(
            RequiredElement(element_id="E1", description="عنصر", element_type="legal"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state(issue_title=""))

        assert "required_elements" in output

    def test_zero_elements_from_llm(self):
        """LLM returns empty elements list."""
        mock_result = DecomposedIssue(elements=[])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            output = decompose_issue_node(_make_state())

        assert output["required_elements"] == []
