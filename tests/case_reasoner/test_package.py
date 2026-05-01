"""
test_package.py — Unit tests for package_result_node.

No mocking needed — this is a pure function that copies state fields into a dict.
"""

import pathlib
import sys
import copy

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from nodes.package import package_result_node


def _make_branch_state(**kwargs):
    base = {
        "case_id": "test-001",
        "issue_id": 1,
        "issue_title": "التعويض عن الإخلال بالعقد",
        "legal_domain": "العقود المدنية",
        "source_text": "نص مصدري",
        "required_elements": [{"element_id": "E1", "description": "عنصر", "element_type": "legal"}],
        "law_retrieval_result": {"answer": "نص قانوني"},
        "retrieved_articles": [{"article_number": 148}],
        "retrieved_facts": "وقائع مسترداة",
        "element_classifications": [{"element_id": "E1", "status": "established"}],
        "law_application": "تحليل قانوني",
        "applied_elements": [{"element_id": "E1", "reasoning": "تحليل", "cited_articles": [148]}],
        "skipped_elements": [],
        "counterarguments": {"plaintiff_arguments": [], "defendant_arguments": [], "analysis": ""},
        "citation_check": {"passed": True},
        "logical_consistency_check": {"passed": True},
        "completeness_check": {"passed": True},
        "validation_passed": True,
        "issue_analyses": [],
        "intermediate_steps": [],
        "error_log": [],
    }
    base.update(kwargs)
    return base


_EXPECTED_PACKAGE_KEYS = {
    "issue_id", "issue_title", "legal_domain", "source_text",
    "required_elements", "law_retrieval_result", "retrieved_articles",
    "retrieved_facts", "element_classifications", "law_application",
    "applied_elements", "skipped_elements", "counterarguments",
    "citation_check", "logical_consistency_check", "completeness_check",
    "validation_passed",
}


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPackageResultNode:
    """T-PACKAGE-01: package_result_node wraps all branch state correctly."""

    def test_returns_issue_analyses_key(self):
        output = package_result_node(_make_branch_state())
        assert "issue_analyses" in output

    def test_returns_single_element_list(self):
        output = package_result_node(_make_branch_state())
        assert isinstance(output["issue_analyses"], list)
        assert len(output["issue_analyses"]) == 1

    def test_wrapped_dict_has_all_required_keys(self):
        output = package_result_node(_make_branch_state())
        wrapped = output["issue_analyses"][0]
        missing = _EXPECTED_PACKAGE_KEYS - set(wrapped.keys())
        assert missing == set(), f"Missing keys in package: {missing}"

    def test_issue_id_preserved(self):
        output = package_result_node(_make_branch_state(issue_id=42))
        assert output["issue_analyses"][0]["issue_id"] == 42

    def test_validation_passed_preserved(self):
        output = package_result_node(_make_branch_state(validation_passed=True))
        assert output["issue_analyses"][0]["validation_passed"] is True

    def test_validation_failed_preserved(self):
        output = package_result_node(_make_branch_state(validation_passed=False))
        assert output["issue_analyses"][0]["validation_passed"] is False

    def test_applied_elements_preserved(self):
        applied = [{"element_id": "E1", "reasoning": "تحليل", "cited_articles": [148]}]
        output = package_result_node(_make_branch_state(applied_elements=applied))
        assert output["issue_analyses"][0]["applied_elements"] == applied

    def test_skipped_elements_preserved(self):
        skipped = ["E2", "E3"]
        output = package_result_node(_make_branch_state(skipped_elements=skipped))
        assert output["issue_analyses"][0]["skipped_elements"] == skipped


# ---------------------------------------------------------------------------
# Missing / empty state
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPackageResultEmpty:
    """Missing state keys default to empty values."""

    def test_handles_missing_state_keys(self):
        minimal_state = {
            "issue_id": 1,
            "issue_title": "مسألة",
        }
        output = package_result_node(minimal_state)
        wrapped = output["issue_analyses"][0]
        assert wrapped["required_elements"] == []
        assert wrapped["law_retrieval_result"] == {}
        assert wrapped["retrieved_articles"] == []
        assert wrapped["retrieved_facts"] == ""
        assert wrapped["applied_elements"] == []
        assert wrapped["skipped_elements"] == []
        assert wrapped["validation_passed"] is False

    def test_no_mutation_of_input_state(self):
        state = _make_branch_state()
        original = copy.deepcopy(state)
        package_result_node(state)
        assert state == original
