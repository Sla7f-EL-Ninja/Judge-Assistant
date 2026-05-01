"""
test_consistency.py — Unit tests for check_global_consistency_node.

Patch target: nodes.consistency.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.consistency import check_global_consistency_node
from schemas import ConsistencyCheckResult, ConsistencyConflict


def _make_conflict(issue_ids=None, conflict_type="contradictory_article_application"):
    return ConsistencyConflict(
        issue_ids=issue_ids or [1, 2],
        conflict_type=conflict_type,
        description="تطبيق متناقض للمادة 148",
    )


def _make_consistency_result(conflicts=None):
    conflicts = conflicts or []
    return ConsistencyCheckResult(
        conflicts=conflicts,
        has_conflicts=len(conflicts) > 0,
    )


def _make_analysis(issue_id, law_app="تحليل قانوني"):
    return {
        "issue_id": issue_id,
        "issue_title": f"مسألة {issue_id}",
        "law_application": law_app,
        "applied_elements": [{"cited_articles": [148]}],
    }


def _make_state(analyses=None, relationships=None):
    return {
        "issue_analyses": analyses or [],
        "cross_issue_relationships": relationships or [],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGlobalConsistencyHappy:
    """T-CONSISTENCY-01: Conflict detection and reconciliation."""

    def test_detects_conflicts(self):
        conflict_result = _make_consistency_result([_make_conflict()])
        reconciliation_content = "فقرة التوفيق بين المسألتين"
        response = MagicMock()
        response.content = reconciliation_content

        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = conflict_result
        llm.invoke.return_value = response

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert len(output["consistency_conflicts"]) == 1
        assert output["consistency_conflicts"][0]["conflict_type"] == "contradictory_article_application"

    def test_generates_reconciliation_paragraph_per_conflict(self):
        """Constraint #8: one reconciliation paragraph per conflict."""
        conflict_result = _make_consistency_result([_make_conflict()])
        response = MagicMock()
        response.content = "فقرة التوفيق"
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = conflict_result
        llm.invoke.return_value = response

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert len(output["reconciliation_paragraphs"]) == 1

    def test_multiple_conflicts_multiple_paragraphs(self):
        """Two conflicts → two reconciliation paragraphs."""
        conflict_result = _make_consistency_result([
            _make_conflict([1, 2], "contradictory_article_application"),
            _make_conflict([2, 3], "contradictory_fact_evaluation"),
        ])
        response = MagicMock()
        response.content = "فقرة توفيق"
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = conflict_result
        llm.invoke.return_value = response

        analyses = [_make_analysis(1), _make_analysis(2), _make_analysis(3)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert len(output["reconciliation_paragraphs"]) == 2

    def test_no_conflicts_empty_reconciliation(self):
        """Constraint #8: no conflicts → reconciliation_paragraphs is empty."""
        no_conflict_result = _make_consistency_result([])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = no_conflict_result

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert output["reconciliation_paragraphs"] == []
        assert output["consistency_conflicts"] == []

    def test_reconciliation_only_on_conflict(self):
        """Constraint #8: llm.invoke() for reconciliation called IFF conflicts exist."""
        no_conflict_result = _make_consistency_result([])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = no_conflict_result

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            check_global_consistency_node(_make_state(analyses))

        # .invoke() for reconciliation text should NOT be called when no conflicts
        llm.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGlobalConsistencyEdge:
    """T-CONSISTENCY-02: Edge cases."""

    def test_single_issue_skips_check(self):
        """< 2 issues → empty conflicts, no LLM call."""
        llm = MagicMock()

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state([_make_analysis(1)]))

        assert output["consistency_conflicts"] == []
        assert output["reconciliation_paragraphs"] == []
        llm.with_structured_output.assert_not_called()

    def test_empty_analyses_skips_check(self):
        llm = MagicMock()

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state([]))

        assert output["consistency_conflicts"] == []

    def test_conflict_detection_failure_assumes_none(self):
        """Exception in conflict detection → empty conflicts, no crash."""
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert output["consistency_conflicts"] == []
        assert output["reconciliation_paragraphs"] == []

    def test_reconciliation_exception_produces_placeholder(self):
        """Reconciliation LLM call raises → placeholder text, no crash."""
        conflict_result = _make_consistency_result([_make_conflict()])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = conflict_result
        llm.invoke.side_effect = RuntimeError("reconciliation LLM failed")

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert len(output["reconciliation_paragraphs"]) == 1
        assert len(output["reconciliation_paragraphs"][0]) > 0

    def test_intermediate_steps_logged(self):
        no_conflict_result = _make_consistency_result([])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = no_conflict_result

        analyses = [_make_analysis(1), _make_analysis(2)]

        with patch("nodes.consistency.get_llm", return_value=llm):
            output = check_global_consistency_node(_make_state(analyses))

        assert "intermediate_steps" in output
