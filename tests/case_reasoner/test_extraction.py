"""
test_extraction.py — Unit tests for extract_issues_node.

Patch target: nodes.extraction.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.extraction import extract_issues_node
from schemas import ExtractedIssues, LegalIssue


def _make_extracted_issues(*titles):
    return ExtractedIssues(issues=[
        LegalIssue(issue_id=i + 1, issue_title=t, legal_domain="عقود", source_text="نص")
        for i, t in enumerate(titles)
    ])


def _make_state(legal_questions="أسئلة قانونية", key_disputes="نقاط خلاف"):
    return {
        "case_brief": {
            "legal_questions": legal_questions,
            "key_disputes": key_disputes,
        }
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractIssuesHappy:
    """Happy path: extraction succeeds and returns correctly structured output."""

    def test_returns_identified_issues(self):
        mock_result = _make_extracted_issues("التعويض عن الإخلال", "صحة العقد")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert "identified_issues" in output
        assert len(output["identified_issues"]) == 2

    def test_consumes_both_legal_questions_and_key_disputes(self):
        """Constraint #1: extraction prompt must include BOTH fields."""
        mock_result = _make_extracted_issues("مسألة واحدة")
        llm = MagicMock()
        parser = llm.with_structured_output.return_value
        parser.invoke.return_value = mock_result
        captured_prompts = []
        original_invoke = parser.invoke

        def capture_invoke(prompt):
            captured_prompts.append(prompt)
            return mock_result

        parser.invoke.side_effect = capture_invoke

        with patch("nodes.extraction.get_llm", return_value=llm):
            extract_issues_node(_make_state(
                legal_questions="هل العقد صحيح؟",
                key_disputes="المبلغ المطالب به",
            ))

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "هل العقد صحيح؟" in prompt
        assert "المبلغ المطالب به" in prompt

    def test_issue_dict_has_required_keys(self):
        mock_result = _make_extracted_issues("التعويض")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        for issue in output["identified_issues"]:
            assert "issue_id" in issue
            assert "issue_title" in issue
            assert "legal_domain" in issue
            assert "source_text" in issue

    def test_intermediate_steps_logged(self):
        mock_result = _make_extracted_issues("مسألة أولى", "مسألة ثانية")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert "intermediate_steps" in output
        assert len(output["intermediate_steps"]) > 0

    def test_issue_ids_preserved(self):
        mock_result = ExtractedIssues(issues=[
            LegalIssue(issue_id=42, issue_title="مسألة", legal_domain="عقود", source_text="نص"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert output["identified_issues"][0]["issue_id"] == 42


# ---------------------------------------------------------------------------
# Exception / fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractIssuesFallback:
    """Fallback behavior: retry on failure, empty list after 2 failures."""

    def test_retry_on_first_failure(self):
        """First invoke raises, second succeeds → issues returned."""
        mock_result = _make_extracted_issues("مسألة أولى")
        parser = MagicMock()
        parser.invoke.side_effect = [RuntimeError("first failure"), mock_result]
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert len(output["identified_issues"]) == 1
        assert parser.invoke.call_count == 2

    def test_empty_list_after_two_failures(self):
        """Both invokes raise → identified_issues is empty list."""
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("always fails")
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert output["identified_issues"] == []

    def test_error_log_populated_on_failure(self):
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("always fails")
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert "error_log" in output
        assert len(output["error_log"]) > 0

    def test_exactly_two_attempts_made(self):
        parser = MagicMock()
        parser.invoke.side_effect = RuntimeError("always fails")
        llm = MagicMock()
        llm.with_structured_output.return_value = parser

        with patch("nodes.extraction.get_llm", return_value=llm):
            extract_issues_node(_make_state())

        assert parser.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractIssuesEdge:
    """Edge cases: empty fields, missing keys."""

    def test_empty_brief_fields(self):
        """Empty legal_questions and key_disputes → node runs without error."""
        mock_result = ExtractedIssues(issues=[])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state(legal_questions="", key_disputes=""))

        assert "identified_issues" in output

    def test_missing_case_brief_key(self):
        """State with no case_brief key → node runs without crash."""
        mock_result = ExtractedIssues(issues=[])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node({})

        assert "identified_issues" in output

    def test_zero_issues_extracted(self):
        """LLM returns 0 issues → empty list, no error."""
        mock_result = ExtractedIssues(issues=[])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            output = extract_issues_node(_make_state())

        assert output["identified_issues"] == []
        assert "error_log" not in output or output.get("error_log") == []
