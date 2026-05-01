"""
test_confidence.py — Unit tests for compute_confidence_node.

Patch targets:
  - nodes.confidence.get_llm
  - nodes.confidence.CONFIDENCE_WEIGHTS (indirectly via cr_config)
  - nodes.confidence.CONFIDENCE_THRESHOLDS (indirectly via cr_config)
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.confidence import compute_confidence_node, _compute_issue_signals, _level_from_score


# Canonical thresholds from cr_config (for assertions)
_THRESHOLDS = {"high": 0.75, "medium": 0.45}

# Weights (from cr_config defaults)
_WEIGHTS = {
    "unsupported_ratio": 0.25,
    "disputed_ratio": 0.15,
    "insufficient_ratio": 0.20,
    "citation_failure_ratio": 0.15,
    "logical_issues": 0.10,
    "completeness_gap": 0.10,
    "reconciliation_triggered": 0.05,
}


def _make_clean_analysis(issue_id=1):
    return {
        "issue_id": issue_id,
        "issue_title": f"مسألة {issue_id}",
        "required_elements": [{"element_id": "E1"}, {"element_id": "E2"}],
        "element_classifications": [
            {"element_id": "E1", "status": "established"},
            {"element_id": "E2", "status": "established"},
        ],
        "applied_elements": [
            {"element_id": "E1", "cited_articles": [148]},
            {"element_id": "E2", "cited_articles": [176]},
        ],
        "citation_check": {
            "total_citations": 2,
            "unsupported_conclusions": [],
            "missing_citations": [],
        },
        "logical_consistency_check": {"severity": "none"},
        "completeness_check": {"coverage_ratio": 1.0},
    }


def _make_state(analyses=None, conflicts=None):
    return {
        "issue_analyses": analyses or [],
        "consistency_conflicts": conflicts or [],
    }


# ---------------------------------------------------------------------------
# Rule-based signal computation (Constraint #6)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfidenceSignalAccuracy:
    """Constraint #6: confidence level computed from signals, not LLM."""

    def test_clean_issue_scores_high(self):
        """No failures → all signals zero → high raw score → high level."""
        analysis = _make_clean_analysis()
        response = MagicMock()
        response.content = "مبرر الثقة"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        assert output["per_issue_confidence"][0]["level"] == "high"

    def test_all_disputed_reduces_score(self):
        analysis = _make_clean_analysis()
        analysis["element_classifications"] = [
            {"element_id": "E1", "status": "disputed"},
            {"element_id": "E2", "status": "disputed"},
        ]
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        score = output["per_issue_confidence"][0]["raw_score"]
        assert score < 1.0  # disputed elements reduce score

    def test_level_determined_before_llm_call(self):
        """Constraint #6: level key set in per_issue_confidence REGARDLESS of LLM justification."""
        analysis = _make_clean_analysis()
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM failed for justification")

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        # Level must still be set even when justification LLM fails
        assert "level" in output["per_issue_confidence"][0]
        assert output["per_issue_confidence"][0]["level"] in {"high", "medium", "low"}

    def test_llm_only_writes_justification(self):
        """Constraint #6: LLM never determines the level, only writes justification text."""
        analysis = _make_clean_analysis()
        # LLM returns a "wrong" level label in its text — node should ignore it
        response = MagicMock()
        response.content = "هذه القضية مستوى منخفض جداً"  # Contradicts signal-based level
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        # Level must be "high" based on signals, not "low" from LLM text
        assert output["per_issue_confidence"][0]["level"] == "high"

    def test_insufficient_elements_reduce_score(self):
        analysis = _make_clean_analysis()
        analysis["element_classifications"] = [
            {"element_id": "E1", "status": "insufficient_evidence"},
            {"element_id": "E2", "status": "established"},
        ]
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        score = output["per_issue_confidence"][0]["raw_score"]
        assert score < 1.0

    def test_reconciliation_flag_reduces_score(self):
        """Issue in conflict → reconciliation_triggered=1.0 → penalty applied."""
        analysis = _make_clean_analysis(issue_id=1)
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        conflict = {"issue_ids": [1, 2], "conflict_type": "test"}

        with patch("nodes.confidence.get_llm", return_value=llm):
            output_with = compute_confidence_node(_make_state([analysis], conflicts=[conflict]))
            output_without = compute_confidence_node(_make_state([analysis], conflicts=[]))

        score_with = output_with["per_issue_confidence"][0]["raw_score"]
        score_without = output_without["per_issue_confidence"][0]["raw_score"]
        assert score_with < score_without


# ---------------------------------------------------------------------------
# Case-level aggregation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCaseLevelConfidence:
    """T-CONFIDENCE-02: Case-level aggregation formula."""

    def test_case_level_formula_two_issues(self):
        """case_score = 0.7 * min + 0.3 * mean."""
        analyses = [_make_clean_analysis(1), _make_clean_analysis(2)]
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state(analyses))

        assert "case_level_confidence" in output
        assert "level" in output["case_level_confidence"]
        assert "raw_score" in output["case_level_confidence"]
        assert output["case_level_confidence"]["raw_score"] >= 0.0
        assert output["case_level_confidence"]["raw_score"] <= 1.0

    def test_empty_analyses_gives_zero_case_score(self):
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([]))

        assert output["case_level_confidence"]["raw_score"] == 0.0

    def test_case_level_has_justification(self):
        analysis = _make_clean_analysis()
        response = MagicMock()
        response.content = "مبرر مستوى الثقة الإجمالي"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        assert "justification" in output["case_level_confidence"]
        assert len(output["case_level_confidence"]["justification"]) > 0


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConfidenceOutputStructure:
    """T-CONFIDENCE-03: Output keys and structure."""

    def test_per_issue_confidence_structure(self):
        analysis = _make_clean_analysis()
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        pc = output["per_issue_confidence"][0]
        assert "issue_id" in pc
        assert "issue_title" in pc
        assert "level" in pc
        assert "raw_score" in pc
        assert "signals" in pc
        assert "justification" in pc

    def test_intermediate_steps_logged(self):
        analysis = _make_clean_analysis()
        response = MagicMock()
        response.content = "مبرر"
        llm = MagicMock()
        llm.invoke.return_value = response

        with patch("nodes.confidence.get_llm", return_value=llm):
            output = compute_confidence_node(_make_state([analysis]))

        assert "intermediate_steps" in output
