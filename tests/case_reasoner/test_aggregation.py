"""
test_aggregation.py — Unit tests for aggregate_issues_node.

Patch target: nodes.aggregation.get_llm
"""

import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.aggregation import aggregate_issues_node
from schemas import IssueDependencies, IssueDependency


def _make_analysis(issue_id, applied_articles=None, facts=""):
    return {
        "issue_id": issue_id,
        "issue_title": f"مسألة {issue_id}",
        "applied_elements": [
            {"element_id": "E1", "cited_articles": applied_articles or []}
        ],
        "retrieved_facts": facts,
    }


def _make_state(analyses=None, identified_issues=None):
    if analyses is None:
        analyses = []
    if identified_issues is None:
        identified_issues = [
            {"issue_id": a["issue_id"], "issue_title": a["issue_title"], "legal_domain": "عقود"}
            for a in analyses
        ]
    return {
        "issue_analyses": analyses,
        "identified_issues": identified_issues,
    }


def _make_empty_dependencies():
    return IssueDependencies(dependencies=[])


def _make_dependency(upstream, downstream, dep_type="شرطية"):
    return IssueDependencies(dependencies=[
        IssueDependency(
            upstream_issue_id=upstream,
            downstream_issue_id=downstream,
            dependency_type=dep_type,
            explanation="يجب إثبات الأولى قبل الثانية",
        )
    ])


# ---------------------------------------------------------------------------
# Shared articles detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSharedArticleDetection:
    """T-AGGREGATION-01: Rule-based shared article detection."""

    def test_detects_shared_article_between_two_issues(self):
        analyses = [
            _make_analysis(1, applied_articles=[148]),
            _make_analysis(2, applied_articles=[148]),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_article"]
        assert len(shared) == 1
        assert shared[0]["article_number"] == 148
        assert 1 in shared[0]["issue_ids"]
        assert 2 in shared[0]["issue_ids"]

    def test_no_shared_articles_when_different(self):
        analyses = [
            _make_analysis(1, applied_articles=[148]),
            _make_analysis(2, applied_articles=[176]),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_article"]
        assert len(shared) == 0

    def test_three_issues_sharing_article(self):
        analyses = [
            _make_analysis(1, applied_articles=[148]),
            _make_analysis(2, applied_articles=[148]),
            _make_analysis(3, applied_articles=[148]),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_article"]
        assert len(shared) == 1
        assert len(shared[0]["issue_ids"]) == 3

    def test_partial_sharing_only_shared_pair_detected(self):
        """Issues 1 and 3 share article 220; issue 2 has different article."""
        analyses = [
            _make_analysis(1, applied_articles=[220]),
            _make_analysis(2, applied_articles=[176]),
            _make_analysis(3, applied_articles=[220]),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_article"]
        assert len(shared) == 1
        ids = shared[0]["issue_ids"]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids


# ---------------------------------------------------------------------------
# Shared facts detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSharedFactDetection:
    """T-AGGREGATION-02: Rule-based shared entity detection."""

    def test_detects_shared_fact_with_date_and_amount(self):
        shared_text = "بتاريخ 25/03/2024 بمبلغ 100,000 جنيه"
        analyses = [
            _make_analysis(1, facts=shared_text),
            _make_analysis(2, facts=shared_text),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_fact"]
        assert len(shared) >= 1

    def test_single_shared_entity_not_enough(self):
        """Only 1 shared entity → threshold is >= 2 → no shared_fact relationship."""
        analyses = [
            _make_analysis(1, facts="بتاريخ 25/03/2024"),
            _make_analysis(2, facts="بتاريخ 25/03/2024"),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_fact"]
        # 1 entity is not enough → no shared_fact
        assert len(shared) == 0


# ---------------------------------------------------------------------------
# Issue dependency detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIssueDependencyDetection:
    """T-AGGREGATION-03: LLM-assisted issue dependency detection."""

    def test_detects_dependency_from_llm(self):
        analyses = [_make_analysis(1), _make_analysis(2)]
        dep_result = _make_dependency(upstream=1, downstream=2)
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = dep_result

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        deps = [r for r in output["cross_issue_relationships"] if r["type"] == "dependency"]
        assert len(deps) == 1
        assert deps[0]["upstream"] == 1
        assert deps[0]["downstream"] == 2

    def test_dependency_detection_failure_graceful(self):
        """LLM raises → only rule-based relationships returned, no crash."""
        analyses = [
            _make_analysis(1, applied_articles=[148]),
            _make_analysis(2, applied_articles=[148]),
        ]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM error")

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        assert "cross_issue_relationships" in output
        # Rule-based shared article should still be detected
        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_article"]
        assert len(shared) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAggregationEdge:
    """T-AGGREGATION-04: Edge cases."""

    def test_empty_issue_analyses(self):
        llm = MagicMock()
        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state([]))

        assert output["cross_issue_relationships"] == []
        llm.with_structured_output.assert_not_called()

    def test_single_issue_no_cross_analysis(self):
        """Single issue → no shared articles possible."""
        analyses = [_make_analysis(1, applied_articles=[148])]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        shared = [r for r in output["cross_issue_relationships"] if r["type"] == "shared_article"]
        assert len(shared) == 0

    def test_intermediate_steps_logged(self):
        analyses = [_make_analysis(1, applied_articles=[148])]
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = _make_empty_dependencies()

        with patch("nodes.aggregation.get_llm", return_value=llm):
            output = aggregate_issues_node(_make_state(analyses))

        assert "intermediate_steps" in output
