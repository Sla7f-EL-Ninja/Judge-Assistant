"""
test_validation.py — Unit tests for validate_analysis_node and its 3 sub-steps.

Patch targets:
  - nodes.validation.get_llm (for logical consistency sub-step)
  - tools.civil_law_rag_tool (for citation retry in _citation_check)
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.validation import (
    validate_analysis_node,
    _citation_check,
    _logical_consistency_check,
    _completeness_check,
)
from schemas import LogicalConsistencyResult


def _make_state(**kwargs):
    base = {
        "issue_title": "التعويض عن الإخلال بالعقد",
        "law_application": "وفقاً للمادة 148 من القانون المدني، يتبين أن الإخلال ثابت.",
        "applied_elements": [
            {"element_id": "E1", "reasoning": "تحليل", "cited_articles": [148]},
        ],
        "retrieved_articles": [
            {"article_number": 148, "article_text": "نص المادة 148"},
        ],
        "element_classifications": [
            {"element_id": "E1", "status": "established", "evidence_summary": "ثابت"},
        ],
        "counterarguments": {
            "plaintiff_arguments": ["حجة المدعي"],
            "defendant_arguments": ["حجة المدعى عليه"],
        },
        "required_elements": [
            {"element_id": "E1", "description": "وجود عقد", "element_type": "civil_law"},
        ],
        "skipped_elements": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# _citation_check sub-step
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCitationCheck:
    """T-VALIDATION-01: Citation check decision tree."""

    def test_all_citations_verified(self):
        """Cited article 148 is in retrieved_articles → passed=True."""
        state = _make_state()
        with patch("tools.civil_law_rag_tool") as mock_tool:
            result = _citation_check(state)
        assert result["passed"] is True
        mock_tool.assert_not_called()  # No retry needed

    def test_missing_citation_triggers_retry(self):
        """Article 176 cited but not in retrieved → civil_law_rag_tool called."""
        state = _make_state(
            law_application="المادة 148 والمادة 176 تُطبَّقان على هذه الواقعة.",
            retrieved_articles=[{"article_number": 148, "article_text": "نص 148"}],
        )
        retry_result = {"answer": "نص المادة 176", "sources": []}
        with patch("tools.civil_law_rag_tool", return_value=retry_result) as mock_tool:
            result = _citation_check(state)
        mock_tool.assert_called_once()
        assert "176" in mock_tool.call_args[0][0]

    def test_retry_success_marks_found(self):
        """Retry for article 176 returns valid answer → missing resolved → passed=True."""
        state = _make_state(
            law_application="وفقاً للمادة 176 من القانون",
            retrieved_articles=[],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [176]}],
        )
        with patch("tools.civil_law_rag_tool", return_value={"answer": "نص المادة 176", "sources": []}):
            result = _citation_check(state)
        assert result["passed"] is True

    def test_retry_failure_marks_unsupported(self):
        """Retry returns empty answer → article still missing → passed=False."""
        state = _make_state(
            law_application="وفقاً للمادة 176",
            retrieved_articles=[],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [176]}],
        )
        with patch("tools.civil_law_rag_tool", return_value={"answer": "", "sources": []}):
            result = _citation_check(state)
        assert result["passed"] is False

    def test_retry_exception_marks_unsupported(self):
        """Retry raises → article still missing → passed=False."""
        state = _make_state(
            law_application="وفقاً للمادة 176",
            retrieved_articles=[],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [176]}],
        )
        with patch("tools.civil_law_rag_tool", side_effect=RuntimeError("service unavailable")):
            result = _citation_check(state)
        assert result["passed"] is False

    def test_retry_targets_only_law_tool(self):
        """Constraint #4: citation retry NEVER calls case_documents_rag_tool."""
        state = _make_state(
            law_application="وفقاً للمادة 176",
            retrieved_articles=[],
        )
        with patch("tools.civil_law_rag_tool", return_value={"answer": "", "sources": []}) as mock_law, \
             patch("tools.case_documents_rag_tool") as mock_fact:
            _citation_check(state)
        mock_fact.assert_not_called()

    def test_retry_called_once_per_missing_article(self):
        """Constraint #4: retry called exactly once per missing article, not more."""
        state = _make_state(
            law_application="المادة 176 والمادة 220 تُطبَّقان.",
            retrieved_articles=[],
        )
        with patch("tools.civil_law_rag_tool", return_value={"answer": "", "sources": []}) as mock_tool:
            _citation_check(state)
        assert mock_tool.call_count == 2  # Once for 176, once for 220

    def test_uncited_element_fails_check(self):
        """applied_elements with empty cited_articles → unsupported_conclusions → passed=False."""
        state = _make_state(
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": []}],
        )
        with patch("tools.civil_law_rag_tool"):
            result = _citation_check(state)
        assert result["passed"] is False
        assert "E1" in result["unsupported_conclusions"]

    def test_all_verified_no_uncited_passes(self):
        state = _make_state(
            law_application="وفقاً للمادة 148",
            retrieved_articles=[{"article_number": 148, "article_text": "نص"}],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [148]}],
        )
        with patch("tools.civil_law_rag_tool") as mock_tool:
            result = _citation_check(state)
        assert result["passed"] is True
        mock_tool.assert_not_called()

    def test_arabic_digit_citation_in_text(self):
        """المادة ٢٤٨ in law_application → article 248 extracted and checked."""
        state = _make_state(
            law_application="تطبيقاً للمادة ٢٤٨ من القانون المدني",
            retrieved_articles=[{"article_number": 248, "article_text": "نص 248"}],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [248]}],
        )
        with patch("tools.civil_law_rag_tool") as mock_tool:
            result = _citation_check(state)
        assert result["passed"] is True
        mock_tool.assert_not_called()


# ---------------------------------------------------------------------------
# _logical_consistency_check sub-step
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLogicalConsistencyCheck:
    """T-VALIDATION-02: Logical consistency check decision tree."""

    def test_passed_when_no_issues(self):
        mock_result = LogicalConsistencyResult(passed=True, issues_found=[], severity="none")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.validation.get_llm", return_value=llm):
            result = _logical_consistency_check(_make_state())

        assert result["passed"] is True
        assert result["severity"] == "none"

    def test_failed_on_major_issues(self):
        mock_result = LogicalConsistencyResult(
            passed=False,
            issues_found=["تناقض في تقييم E1"],
            severity="major",
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.validation.get_llm", return_value=llm):
            result = _logical_consistency_check(_make_state())

        assert result["passed"] is False
        assert result["severity"] == "major"

    def test_failed_on_minor_issues(self):
        mock_result = LogicalConsistencyResult(
            passed=False,
            issues_found=["مشكلة بسيطة"],
            severity="minor",
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.validation.get_llm", return_value=llm):
            result = _logical_consistency_check(_make_state())

        assert result["severity"] == "minor"

    def test_defaults_to_passed_on_exception(self):
        """Exception → passed=True, severity='none' (graceful default)."""
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM failed")

        with patch("nodes.validation.get_llm", return_value=llm):
            result = _logical_consistency_check(_make_state())

        assert result["passed"] is True
        assert result["severity"] == "none"


# ---------------------------------------------------------------------------
# _completeness_check sub-step
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCompletenessCheck:
    """T-VALIDATION-03: Completeness check decision tree."""

    def test_all_elements_covered(self):
        """All required elements appear in applied or skipped → passed=True, ratio=1.0."""
        state = _make_state(
            required_elements=[{"element_id": "E1"}, {"element_id": "E2"}],
            applied_elements=[{"element_id": "E1", "cited_articles": [148]}],
            skipped_elements=["E2"],
        )
        result = _completeness_check(state)
        assert result["passed"] is True
        assert result["coverage_ratio"] == pytest.approx(1.0)

    def test_missing_element_detected(self):
        state = _make_state(
            required_elements=[{"element_id": "E1"}, {"element_id": "E2"}],
            applied_elements=[{"element_id": "E1", "cited_articles": [148]}],
            skipped_elements=[],
        )
        result = _completeness_check(state)
        assert result["passed"] is False
        assert "E2" in result["missing_elements"]

    def test_coverage_ratio_calculated(self):
        """2 out of 3 covered → ratio=0.667."""
        state = _make_state(
            required_elements=[{"element_id": "E1"}, {"element_id": "E2"}, {"element_id": "E3"}],
            applied_elements=[{"element_id": "E1", "cited_articles": [148]}],
            skipped_elements=["E2"],
        )
        result = _completeness_check(state)
        assert result["coverage_ratio"] == pytest.approx(0.667, abs=0.001)
        assert "E3" in result["missing_elements"]

    def test_empty_required_elements(self):
        """No required elements → trivially passed."""
        state = _make_state(required_elements=[], applied_elements=[], skipped_elements=[])
        result = _completeness_check(state)
        assert result["passed"] is True

    def test_all_elements_in_skipped(self):
        """All insufficient_evidence → all skipped, still counts as covered."""
        state = _make_state(
            required_elements=[{"element_id": "E1"}, {"element_id": "E2"}],
            applied_elements=[],
            skipped_elements=["E1", "E2"],
        )
        result = _completeness_check(state)
        assert result["passed"] is True
        assert result["coverage_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# validate_analysis_node (main orchestrator)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateAnalysisNode:
    """T-VALIDATION-04: Main validation orchestrator decision matrix."""

    def _run_node(self, state, consistency_mock=None):
        """Helper to run validate_analysis_node with standard mocks."""
        if consistency_mock is None:
            consistency_mock = LogicalConsistencyResult(passed=True, issues_found=[], severity="none")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = consistency_mock
        with patch("nodes.validation.get_llm", return_value=llm), \
             patch("tools.civil_law_rag_tool", return_value={"answer": "", "sources": []}):
            return validate_analysis_node(state)

    def test_all_pass_means_validation_passed(self):
        state = _make_state(
            law_application="وفقاً للمادة 148",
            retrieved_articles=[{"article_number": 148, "article_text": "نص"}],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [148]}],
            required_elements=[{"element_id": "E1"}],
            skipped_elements=[],
        )
        output = self._run_node(state)
        assert output["validation_passed"] is True

    def test_citation_fail_means_validation_failed(self):
        """Uncited element → citation_check fails → validation_passed=False."""
        state = _make_state(
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": []}],
        )
        output = self._run_node(state)
        assert output["validation_passed"] is False

    def test_consistency_fail_means_validation_failed(self):
        consistency_fail = LogicalConsistencyResult(
            passed=False, issues_found=["تناقض"], severity="major"
        )
        state = _make_state(
            law_application="وفقاً للمادة 148",
            retrieved_articles=[{"article_number": 148}],
            applied_elements=[{"element_id": "E1", "reasoning": "...", "cited_articles": [148]}],
            required_elements=[{"element_id": "E1"}],
        )
        output = self._run_node(state, consistency_mock=consistency_fail)
        assert output["validation_passed"] is False

    def test_completeness_fail_means_validation_failed(self):
        """Missing required element → completeness fails → validation_passed=False."""
        state = _make_state(
            law_application="وفقاً للمادة 148",
            retrieved_articles=[{"article_number": 148}],
            applied_elements=[],
            required_elements=[{"element_id": "E1"}, {"element_id": "E2"}],
            skipped_elements=[],
        )
        output = self._run_node(state)
        assert output["validation_passed"] is False

    def test_output_contains_all_three_checks(self):
        state = _make_state()
        output = self._run_node(state)
        assert "citation_check" in output
        assert "logical_consistency_check" in output
        assert "completeness_check" in output

    def test_intermediate_steps_logged(self):
        state = _make_state()
        output = self._run_node(state)
        assert "intermediate_steps" in output
        assert len(output["intermediate_steps"]) > 0
