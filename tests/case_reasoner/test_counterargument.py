"""
test_counterargument.py — Unit tests for counterargument_node.

Patch target: nodes.counterargument.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.counterargument import counterargument_node
from schemas import Counterarguments


def _make_counterarguments(plaintiff=None, defendant=None, analysis="مقارنة محايدة"):
    return Counterarguments(
        plaintiff_arguments=plaintiff or ["حجة المدعي"],
        defendant_arguments=defendant or ["حجة المدعى عليه"],
        analysis=analysis,
    )


def _make_state(**kwargs):
    base = {
        "law_application": "تحليل القانون على الوقائع",
        "element_classifications": [
            {"element_id": "E1", "status": "established", "evidence_summary": "ثابت"},
        ],
        "retrieved_facts": "وقائع مسترداة من مستندات القضية",
        "issue_title": "التعويض عن الإخلال بالعقد",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterargumentHappy:
    """Happy path: counterargument node returns both parties' arguments."""

    def test_returns_counterarguments_dict(self):
        mock_result = _make_counterarguments()
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert "counterarguments" in output
        assert "plaintiff_arguments" in output["counterarguments"]
        assert "defendant_arguments" in output["counterarguments"]
        assert "analysis" in output["counterarguments"]

    def test_both_party_arguments_present(self):
        mock_result = _make_counterarguments(
            plaintiff=["حجة أولى للمدعي", "حجة ثانية للمدعي"],
            defendant=["دفع أول للمدعى عليه"],
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert len(output["counterarguments"]["plaintiff_arguments"]) == 2
        assert len(output["counterarguments"]["defendant_arguments"]) == 1

    def test_analysis_field_preserved(self):
        mock_result = _make_counterarguments(analysis="تحليل محايد دقيق")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert output["counterarguments"]["analysis"] == "تحليل محايد دقيق"

    def test_intermediate_steps_logged(self):
        mock_result = _make_counterarguments()
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert "intermediate_steps" in output

    def test_prompt_uses_law_application_not_brief(self):
        """Constraint #2: prompt uses law_application and retrieved_facts, not brief fields."""
        captured = []
        mock_result = _make_counterarguments()
        parser = MagicMock()

        def capture(prompt):
            captured.append(prompt)
            return mock_result

        parser.invoke.side_effect = capture
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.counterargument.get_llm", return_value=llm):
            counterargument_node(_make_state(
                law_application="تطبيق القانون الخاص بالمسألة",
                retrieved_facts="وقائع خاصة من ملف القضية",
            ))

        assert "تطبيق القانون الخاص بالمسألة" in captured[0]
        assert "وقائع خاصة من ملف القضية" in captured[0]


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterargumentFallback:
    """Fallback: empty counterarguments on exception."""

    def test_empty_counterarguments_on_exception(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert output["counterarguments"]["plaintiff_arguments"] == []
        assert output["counterarguments"]["defendant_arguments"] == []

    def test_default_analysis_message_on_exception(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert len(output["counterarguments"]["analysis"]) > 0

    def test_error_log_populated_on_failure(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterargumentEdge:
    """Edge cases: missing input fields."""

    def test_missing_law_application_uses_placeholder(self):
        """Empty law_application → node still runs with placeholder in prompt."""
        mock_result = _make_counterarguments()
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state(law_application=""))

        assert "counterarguments" in output

    def test_missing_retrieved_facts_uses_placeholder(self):
        mock_result = _make_counterarguments()
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.counterargument.get_llm", return_value=llm):
            output = counterargument_node(_make_state(retrieved_facts=""))

        assert "counterarguments" in output
