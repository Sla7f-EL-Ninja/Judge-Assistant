"""
test_report.py — Unit tests for generate_report_node and generate_empty_report_node.

Patch target: nodes.report.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.report import generate_report_node, generate_empty_report_node, _build_fallback_report


def _make_full_state(**kwargs):
    base = {
        "identified_issues": [
            {"issue_id": 1, "issue_title": "التعويض", "legal_domain": "عقود"},
        ],
        "issue_analyses": [
            {
                "issue_id": 1,
                "issue_title": "التعويض",
                "applied_elements": [
                    {"element_id": "E1", "reasoning": "تحليل", "cited_articles": [148]},
                ],
                "counterarguments": {
                    "plaintiff_arguments": ["حجة المدعي"],
                    "defendant_arguments": ["حجة المدعى عليه"],
                },
                "law_application": "تحليل قانوني محايد",
            }
        ],
        "per_issue_confidence": [
            {
                "issue_id": 1,
                "issue_title": "التعويض",
                "level": "high",
                "raw_score": 0.85,
                "justification": "مستوى ثقة مرتفع لثبوت الأدلة",
            }
        ],
        "case_level_confidence": {
            "level": "high",
            "raw_score": 0.85,
            "justification": "مستوى ثقة إجمالي مرتفع",
        },
        "consistency_conflicts": [],
        "reconciliation_paragraphs": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# generate_report_node
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateReportNode:
    """T-REPORT-01: generate_report_node."""

    def test_returns_final_report(self):
        response = MagicMock()
        response.content = "# تقرير التحليل القانوني\n## القسم الأول..."
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.report.get_llm", return_value=llm):
            output = generate_report_node(_make_full_state())

        assert "final_report" in output
        assert len(output["final_report"]) > 0

    def test_llm_content_used_as_report(self):
        report_text = "التقرير القانوني الكامل"
        response = MagicMock()
        response.content = report_text
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.report.get_llm", return_value=llm):
            output = generate_report_node(_make_full_state())

        assert output["final_report"] == report_text

    def test_fallback_report_on_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.report.get_llm", return_value=llm):
            output = generate_report_node(_make_full_state())

        assert "final_report" in output
        assert len(output["final_report"]) > 0  # fallback report produced

    def test_error_log_on_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.report.get_llm", return_value=llm):
            output = generate_report_node(_make_full_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_intermediate_steps_logged(self):
        response = MagicMock()
        response.content = "تقرير"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.report.get_llm", return_value=llm):
            output = generate_report_node(_make_full_state())

        assert "intermediate_steps" in output


# ---------------------------------------------------------------------------
# generate_empty_report_node
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateEmptyReportNode:
    """T-REPORT-02: generate_empty_report_node produces static message."""

    def test_returns_final_report(self):
        output = generate_empty_report_node({})
        assert "final_report" in output
        assert len(output["final_report"]) > 0

    def test_no_llm_call(self):
        """Empty report never calls LLM."""
        with patch("nodes.report.get_llm") as mock_get_llm:
            generate_empty_report_node({})
        mock_get_llm.assert_not_called()

    def test_intermediate_steps_logged(self):
        output = generate_empty_report_node({})
        assert "intermediate_steps" in output

    def test_report_is_arabic(self):
        """Static message must be in Arabic."""
        output = generate_empty_report_node({})
        # At minimum should contain Arabic characters
        arabic_chars = set("أبتثجحخدذرزسشصضطظعغفقكلمنهوي")
        assert any(c in output["final_report"] for c in arabic_chars)


# ---------------------------------------------------------------------------
# _build_fallback_report
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildFallbackReport:
    """T-REPORT-03: _build_fallback_report assembles minimal report without LLM."""

    def test_contains_identified_issues(self):
        state = _make_full_state()
        report = _build_fallback_report(state)
        assert "التعويض" in report  # issue title present

    def test_contains_section_headers(self):
        state = _make_full_state()
        report = _build_fallback_report(state)
        assert "القسم الأول" in report

    def test_contains_confidence_level(self):
        state = _make_full_state()
        report = _build_fallback_report(state)
        assert "مرتفع" in report or "0.85" in report

    def test_handles_empty_state(self):
        """Empty state → fallback report still produces a string."""
        report = _build_fallback_report({})
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_article_citations_from_applied_elements(self):
        state = _make_full_state()
        report = _build_fallback_report(state)
        assert "148" in report  # cited article appears
