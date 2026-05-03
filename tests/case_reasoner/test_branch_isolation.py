"""
test_branch_isolation.py — Parallel Send() branch independence tests.

Tests verify:
1. Router Send payloads are fully independent (no shared references)
2. Package node doesn't mutate input state
3. Reducer merging via operator.add works correctly
4. Branch results identify only their own issue_id
"""

import copy
import operator
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from routers import issue_dispatch_router
from nodes.package import package_result_node
from nodes.extraction import extract_issues_node
from nodes.decomposition import decompose_issue_node
from schemas import ExtractedIssues, LegalIssue, DecomposedIssue, RequiredElement


def _make_issues(n):
    return [
        {"issue_id": i, "issue_title": f"مسألة {i}", "legal_domain": "عقود", "source_text": f"نص {i}"}
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Send payload isolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSendPayloadIsolation:
    """T-ISOLATION-01: Router produces fully isolated Send payloads."""

    def test_each_send_is_independent_dict(self):
        """Modifying one payload does not affect another."""
        state = {"identified_issues": _make_issues(3), "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        original_title = sends[1].arg["issue_title"]
        sends[0].arg["issue_title"] = "معدّل"

        assert sends[1].arg["issue_title"] == original_title

    def test_no_shared_references_between_payloads(self):
        """Each payload is a unique dict object."""
        state = {"identified_issues": _make_issues(3), "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        ids = [id(s.arg) for s in sends]
        assert len(set(ids)) == len(ids)

    def test_send_payloads_have_separate_reducer_lists(self):
        """issue_analyses, intermediate_steps, error_log are separate list instances."""
        state = {"identified_issues": _make_issues(2), "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        p0 = sends[0].arg
        p1 = sends[1].arg

        p0["issue_analyses"].append("test")
        assert len(p1["issue_analyses"]) == 0

        p0["intermediate_steps"].append("step")
        assert len(p1["intermediate_steps"]) == 0

        p0["error_log"].append("error")
        assert len(p1["error_log"]) == 0

    def test_send_payload_lists_are_empty_at_start(self):
        state = {"identified_issues": _make_issues(2), "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        for send in sends:
            payload = send.arg
            assert payload["issue_analyses"] == []
            assert payload["intermediate_steps"] == []
            assert payload["error_log"] == []

    def test_each_send_targets_correct_issue(self):
        """Each Send payload carries the correct issue identity."""
        issues = _make_issues(3)
        state = {"identified_issues": issues, "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        dispatched_ids = {s.arg["issue_id"] for s in sends}
        expected_ids = {1, 2, 3}
        assert dispatched_ids == expected_ids

    def test_send_payload_query_fields_initialized_empty(self):
        """law_queries and fact_queries are initialized as empty lists in each payload."""
        state = {"identified_issues": _make_issues(2), "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        for send in sends:
            payload = send.arg
            assert payload["law_queries"] == []
            assert payload["fact_queries"] == []

    def test_query_fields_are_separate_list_instances_per_payload(self):
        """law_queries and fact_queries lists are not shared between payloads."""
        state = {"identified_issues": _make_issues(2), "case_id": "test-001"}
        sends = issue_dispatch_router(state)

        p0 = sends[0].arg
        p1 = sends[1].arg

        p0["law_queries"].append({"element_id": "E1", "query": "سؤال"})
        assert len(p1["law_queries"]) == 0

        p0["fact_queries"].append({"element_id": "E1", "query": "سؤال"})
        assert len(p1["fact_queries"]) == 0


# ---------------------------------------------------------------------------
# No state mutation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNoStateMutation:
    """T-ISOLATION-02: Nodes do not mutate their input state dict."""

    def test_package_node_does_not_mutate_input(self):
        state = {
            "issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود",
            "source_text": "نص",
            "required_elements": [{"element_id": "E1"}],
            "law_retrieval_result": {"answer": "نص قانوني"},
            "retrieved_articles": [],
            "retrieved_facts": "وقائع",
            "element_classifications": [{"element_id": "E1", "status": "established"}],
            "law_application": "تحليل",
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
        original = copy.deepcopy(state)
        package_result_node(state)
        assert state == original

    def test_extraction_node_does_not_mutate_input(self):
        state = {
            "case_brief": {
                "legal_questions": "أسئلة قانونية",
                "key_disputes": "نقاط خلاف",
            }
        }
        original = copy.deepcopy(state)
        mock_result = ExtractedIssues(issues=[
            LegalIssue(issue_id=1, issue_title="مسألة", legal_domain="عقود", source_text="نص"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.extraction.get_llm", return_value=llm):
            extract_issues_node(state)

        assert state == original

    def test_decomposition_node_does_not_mutate_input(self):
        state = {
            "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
        }
        original = copy.deepcopy(state)
        mock_result = DecomposedIssue(elements=[
            RequiredElement(element_id="E1", description="عنصر", element_type="legal"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.decomposition.get_llm", return_value=llm):
            decompose_issue_node(state)

        assert state == original

    def test_query_generation_node_does_not_mutate_input(self):
        """T-ISOLATION-02: generate_retrieval_queries_node does not mutate input state."""
        from nodes.query_generation import generate_retrieval_queries_node
        from schemas import ElementQuery, RetrievalQueries

        state = {
            "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص",
            "required_elements": [
                {"element_id": "E1", "description": "عنصر", "element_type": "legal"},
            ],
        }
        original = copy.deepcopy(state)

        mock_result = RetrievalQueries(queries=[
            ElementQuery(element_id="E1", law_query="سؤال قانوني", fact_query="سؤال وقائعي"),
        ])
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("nodes.query_generation.get_llm", return_value=llm):
            generate_retrieval_queries_node(state)

        assert state == original


# ---------------------------------------------------------------------------
# Reducer merge behavior
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReducerMerge:
    """T-ISOLATION-03: operator.add reducer correctly merges branch results."""

    def test_issue_analyses_merge_via_add(self):
        """Three branches produce one entry each; merged result has 3 entries."""
        results = []
        for i in range(1, 4):
            payload = {
                "issue_id": i, "issue_title": f"مسألة {i}", "legal_domain": "عقود",
                "source_text": "نص", "required_elements": [], "law_retrieval_result": {},
                "retrieved_articles": [], "retrieved_facts": "",
                "element_classifications": [], "law_application": "",
                "applied_elements": [], "skipped_elements": [],
                "counterarguments": {},
                "citation_check": {}, "logical_consistency_check": {},
                "completeness_check": {}, "validation_passed": True,
                "issue_analyses": [],
                "intermediate_steps": [],
                "error_log": [],
            }
            branch_output = package_result_node(payload)
            results.append(branch_output["issue_analyses"])

        merged = []
        for r in results:
            merged = operator.add(merged, r)

        assert len(merged) == 3
        ids = {entry["issue_id"] for entry in merged}
        assert ids == {1, 2, 3}

    def test_intermediate_steps_accumulate(self):
        steps_per_branch = ["خطوة من الفرع 1", "خطوة من الفرع 2", "خطوة من الفرع 3"]

        merged = []
        for step in steps_per_branch:
            merged = operator.add(merged, [step])

        assert len(merged) == 3
        assert "خطوة من الفرع 1" in merged
        assert "خطوة من الفرع 3" in merged

    def test_error_log_accumulates_from_all_branches(self):
        errors_branch1 = ["خطأ في الفرع 1"]
        errors_branch2 = []
        errors_branch3 = ["خطأ في الفرع 3"]

        merged = operator.add(
            operator.add(errors_branch1, errors_branch2),
            errors_branch3,
        )

        assert "خطأ في الفرع 1" in merged
        assert "خطأ في الفرع 3" in merged
        assert len(merged) == 2

    def test_each_branch_result_identifies_own_issue(self):
        payloads = []
        for i in range(1, 4):
            state = {
                "issue_id": i, "issue_title": f"مسألة {i}", "legal_domain": "عقود",
                "source_text": "نص", "required_elements": [], "law_retrieval_result": {},
                "retrieved_articles": [], "retrieved_facts": "",
                "element_classifications": [], "law_application": "",
                "applied_elements": [], "skipped_elements": [],
                "counterarguments": {}, "citation_check": {},
                "logical_consistency_check": {}, "completeness_check": {},
                "validation_passed": True,
                "issue_analyses": [], "intermediate_steps": [], "error_log": [],
            }
            payloads.append(state)

        merged = []
        for p in payloads:
            result = package_result_node(p)
            merged = operator.add(merged, result["issue_analyses"])

        for i, entry in enumerate(sorted(merged, key=lambda x: x["issue_id"])):
            assert entry["issue_id"] == i + 1

    def test_branch_isolation_modifying_one_does_not_affect_others(self):
        states = []
        for i in range(1, 3):
            state = {
                "issue_id": i, "issue_title": f"مسألة {i}", "legal_domain": "عقود",
                "source_text": "نص", "required_elements": [], "law_retrieval_result": {},
                "retrieved_articles": [], "retrieved_facts": "",
                "element_classifications": [], "law_application": "",
                "applied_elements": [], "skipped_elements": [],
                "counterarguments": {}, "citation_check": {},
                "logical_consistency_check": {}, "completeness_check": {},
                "validation_passed": True,
                "issue_analyses": [], "intermediate_steps": [], "error_log": [],
            }
            states.append(state)

        result1 = package_result_node(states[0])["issue_analyses"][0]
        result2 = package_result_node(states[1])["issue_analyses"][0]

        result1["issue_title"] = "معدّل"
        assert result2["issue_title"] == "مسألة 2"