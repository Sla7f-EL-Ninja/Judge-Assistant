"""
test_state_and_graph.py — State TypedDict key coverage, graph construction, and router logic.

Tests:
    T-STATE-01: CaseReasonerState key coverage
    T-STATE-02: IssueAnalysisState key coverage
    T-GRAPH-01: Graph construction (main + branch)
    T-ROUTER-01: issue_dispatch_router logic
"""

import pathlib
import sys

import pytest

_CR_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CR"
if str(_CR_DIR) not in sys.path:
    sys.path.insert(0, str(_CR_DIR))

from state import CaseReasonerState, IssueAnalysisState
from routers import issue_dispatch_router


# ---------------------------------------------------------------------------
# T-STATE-01: CaseReasonerState
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCaseReasonerStateKeys:
    """T-STATE-01: CaseReasonerState has exactly 14 expected keys."""

    _EXPECTED_KEYS = {
        # Input
        "case_id", "judge_query", "case_brief", "rendered_brief",
        # Extraction output
        "identified_issues",
        # Branch merge
        "issue_analyses",
        # Aggregation
        "cross_issue_relationships",
        # Consistency
        "consistency_conflicts", "reconciliation_paragraphs",
        # Confidence
        "per_issue_confidence", "case_level_confidence",
        # Final output
        "final_report",
        # Telemetry
        "intermediate_steps", "error_log",
    }

    def test_has_all_expected_keys(self):
        annotations = CaseReasonerState.__annotations__
        missing = self._EXPECTED_KEYS - set(annotations.keys())
        assert missing == set(), f"Missing keys: {missing}"

    def test_total_key_count(self):
        annotations = CaseReasonerState.__annotations__
        assert len(annotations) == 14

    def test_has_input_keys(self):
        keys = set(CaseReasonerState.__annotations__.keys())
        assert {"case_id", "judge_query", "case_brief", "rendered_brief"}.issubset(keys)

    def test_has_extraction_key(self):
        assert "identified_issues" in CaseReasonerState.__annotations__

    def test_has_branch_merge_key(self):
        assert "issue_analyses" in CaseReasonerState.__annotations__

    def test_has_aggregation_key(self):
        assert "cross_issue_relationships" in CaseReasonerState.__annotations__

    def test_has_consistency_keys(self):
        keys = set(CaseReasonerState.__annotations__.keys())
        assert {"consistency_conflicts", "reconciliation_paragraphs"}.issubset(keys)

    def test_has_confidence_keys(self):
        keys = set(CaseReasonerState.__annotations__.keys())
        assert {"per_issue_confidence", "case_level_confidence"}.issubset(keys)

    def test_has_output_key(self):
        assert "final_report" in CaseReasonerState.__annotations__

    def test_has_telemetry_keys(self):
        keys = set(CaseReasonerState.__annotations__.keys())
        assert {"intermediate_steps", "error_log"}.issubset(keys)


# ---------------------------------------------------------------------------
# T-STATE-02: IssueAnalysisState
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIssueAnalysisStateKeys:
    """T-STATE-02: IssueAnalysisState has exactly 24 expected keys."""

    _EXPECTED_KEYS = {
        # Identity
        "case_id", "issue_id", "issue_title", "legal_domain", "source_text",
        # Decomposition
        "required_elements",
        # Query Generation  ← NEW
        "law_queries", "fact_queries",
        # Law retrieval
        "law_retrieval_result", "retrieved_articles",
        # Fact retrieval
        "fact_retrieval_result", "retrieved_facts",
        # Evidence
        "element_classifications",
        # Application
        "law_application", "applied_elements", "skipped_elements",
        # Counterarguments
        "counterarguments",
        # Validation
        "citation_check", "logical_consistency_check", "completeness_check", "validation_passed",
        # Merge channel
        "issue_analyses",
        # Telemetry
        "intermediate_steps", "error_log",
    }

    def test_has_all_expected_keys(self):
        annotations = IssueAnalysisState.__annotations__
        missing = self._EXPECTED_KEYS - set(annotations.keys())
        assert missing == set(), f"Missing keys: {missing}"

    def test_total_key_count(self):
        # Was 22, now 24 after adding law_queries and fact_queries
        assert len(IssueAnalysisState.__annotations__) == 24

    def test_has_identity_keys(self):
        keys = set(IssueAnalysisState.__annotations__.keys())
        assert {"case_id", "issue_id", "issue_title", "legal_domain", "source_text"}.issubset(keys)

    def test_has_query_generation_keys(self):
        """T-STATE-02: law_queries and fact_queries present after query generation node added."""
        keys = set(IssueAnalysisState.__annotations__.keys())
        assert {"law_queries", "fact_queries"}.issubset(keys)

    def test_has_retrieval_keys(self):
        keys = set(IssueAnalysisState.__annotations__.keys())
        assert {"law_retrieval_result", "retrieved_articles", "fact_retrieval_result", "retrieved_facts"}.issubset(keys)

    def test_has_evidence_key(self):
        assert "element_classifications" in IssueAnalysisState.__annotations__

    def test_has_application_keys(self):
        keys = set(IssueAnalysisState.__annotations__.keys())
        assert {"law_application", "applied_elements", "skipped_elements"}.issubset(keys)

    def test_has_validation_keys(self):
        keys = set(IssueAnalysisState.__annotations__.keys())
        assert {"citation_check", "logical_consistency_check", "completeness_check", "validation_passed"}.issubset(keys)

    def test_has_merge_channel(self):
        assert "issue_analyses" in IssueAnalysisState.__annotations__


# ---------------------------------------------------------------------------
# T-GRAPH-01: Graph construction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGraphConstruction:
    """T-GRAPH-01: build_issue_branch and build_case_reasoner_graph compile correctly."""

    def test_build_issue_branch_compiles(self):
        from graph import build_issue_branch
        branch = build_issue_branch()
        assert branch is not None

    def test_issue_branch_has_expected_nodes(self):
        from graph import build_issue_branch
        branch = build_issue_branch()
        node_names = set(branch.nodes.keys())
        expected = {
            "decompose_issue",
            "generate_retrieval_queries",  # NEW
            "retrieve_law", "retrieve_facts",
            "classify_evidence", "apply_law", "generate_counterarguments",
            "validate_analysis", "package_result",
        }
        assert expected.issubset(node_names)

    def test_generate_retrieval_queries_node_present(self):
        """T-GRAPH-01: new query generation node sits between decompose and retrieve."""
        from graph import build_issue_branch
        branch = build_issue_branch()
        assert "generate_retrieval_queries" in branch.nodes

    def test_build_case_reasoner_graph_compiles(self):
        from graph import build_case_reasoner_graph
        graph = build_case_reasoner_graph()
        assert graph is not None

    def test_main_graph_has_expected_nodes(self):
        from graph import build_case_reasoner_graph
        graph = build_case_reasoner_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "extract_issues", "issue_analysis_branch",
            "aggregate_issues", "check_global_consistency",
            "compute_confidence", "generate_report", "generate_empty_report",
        }
        assert expected.issubset(node_names)


# ---------------------------------------------------------------------------
# T-ROUTER-01: issue_dispatch_router
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIssueDispatchRouter:
    """T-ROUTER-01: issue_dispatch_router fan-out and fallback logic."""

    def _make_state(self, issues):
        return {
            "identified_issues": issues,
            "case_id": "test-case-001",
        }

    def test_empty_issues_returns_string(self):
        state = self._make_state([])
        result = issue_dispatch_router(state)
        assert result == "generate_empty_report"

    def test_none_issues_returns_string(self):
        state = {"case_id": "test-case-001"}
        result = issue_dispatch_router(state)
        assert result == "generate_empty_report"

    def test_none_value_returns_string(self):
        state = {"identified_issues": None, "case_id": "test-case-001"}
        result = issue_dispatch_router(state)
        assert result == "generate_empty_report"

    def test_single_issue_returns_one_send(self):
        from langgraph.types import Send
        issues = [{"issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص"}]
        result = issue_dispatch_router(self._make_state(issues))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Send)

    def test_three_issues_returns_three_sends(self):
        issues = [
            {"issue_id": i, "issue_title": f"مسألة {i}", "legal_domain": "عقود", "source_text": "نص"}
            for i in range(1, 4)
        ]
        result = issue_dispatch_router(self._make_state(issues))
        assert len(result) == 3

    def test_send_target_is_issue_analysis_branch(self):
        from langgraph.types import Send
        issues = [{"issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص"}]
        result = issue_dispatch_router(self._make_state(issues))
        assert result[0].node == "issue_analysis_branch"

    def test_send_payload_has_identity_fields(self):
        issues = [{"issue_id": 1, "issue_title": "التعويض", "legal_domain": "عقود", "source_text": "نص مصدري"}]
        result = issue_dispatch_router(self._make_state(issues))
        payload = result[0].arg
        assert payload["issue_id"] == 1
        assert payload["issue_title"] == "التعويض"
        assert payload["legal_domain"] == "عقود"
        assert payload["source_text"] == "نص مصدري"

    def test_send_payload_case_id_propagated(self):
        issues = [{"issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص"}]
        state = {"identified_issues": issues, "case_id": "my-case-999"}
        result = issue_dispatch_router(state)
        assert result[0].arg["case_id"] == "my-case-999"

    def test_send_payload_output_fields_initialized_empty(self):
        issues = [{"issue_id": 1, "issue_title": "مسألة", "legal_domain": "عقود", "source_text": "نص"}]
        result = issue_dispatch_router(self._make_state(issues))
        payload = result[0].arg
        assert payload["required_elements"] == []
        assert payload["law_retrieval_result"] == {}
        assert payload["retrieved_articles"] == []
        assert payload["element_classifications"] == []
        assert payload["applied_elements"] == []
        assert payload["skipped_elements"] == []
        assert payload["validation_passed"] is False
        # New query generation fields
        assert payload["law_queries"] == []
        assert payload["fact_queries"] == []

    def test_each_send_payload_is_independent_dict(self):
        issues = [
            {"issue_id": 1, "issue_title": "مسألة 1", "legal_domain": "عقود", "source_text": "نص 1"},
            {"issue_id": 2, "issue_title": "مسألة 2", "legal_domain": "مسؤولية", "source_text": "نص 2"},
        ]
        result = issue_dispatch_router(self._make_state(issues))
        result[0].arg["issue_title"] = "معدّل"
        assert result[1].arg["issue_title"] == "مسألة 2"

    def test_warning_logged_on_many_issues(self, caplog):
        import logging
        issues = [
            {"issue_id": i, "issue_title": f"مسألة {i}", "legal_domain": "عقود", "source_text": "نص"}
            for i in range(1, 10)
        ]
        with caplog.at_level(logging.WARNING, logger="routers"):
            issue_dispatch_router(self._make_state(issues))
        assert any("9" in record.message or "impact latency" in record.message
                   for record in caplog.records)