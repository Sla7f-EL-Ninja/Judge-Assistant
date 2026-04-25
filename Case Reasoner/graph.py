"""Graph builders for the Case Reasoner pipeline."""
from langgraph.graph import StateGraph, START, END

from state import CaseReasonerState, IssueAnalysisState
from nodes.extraction import extract_issues_node
from nodes.decomposition import decompose_issue_node
from nodes.retrieval import retrieve_law_node, retrieve_facts_node
from nodes.evidence import classify_evidence_node
from nodes.application import apply_law_node
from nodes.counterargument import counterargument_node
from nodes.validation import validate_analysis_node
from nodes.package import package_result_node
from nodes.aggregation import aggregate_issues_node
from nodes.consistency import check_global_consistency_node
from nodes.confidence import compute_confidence_node
from nodes.report import generate_report_node, generate_empty_report_node
from routers import issue_dispatch_router


def build_issue_branch():
    """Compile the per-issue analysis sub-graph."""
    branch = StateGraph(IssueAnalysisState)

    branch.add_node("decompose_issue", decompose_issue_node)
    branch.add_node("retrieve_law", retrieve_law_node)
    branch.add_node("retrieve_facts", retrieve_facts_node)
    branch.add_node("classify_evidence", classify_evidence_node)
    branch.add_node("apply_law", apply_law_node)
    branch.add_node("generate_counterarguments", counterargument_node)
    branch.add_node("validate_analysis", validate_analysis_node)
    branch.add_node("package_result", package_result_node)

    branch.add_edge(START, "decompose_issue")
    branch.add_edge("decompose_issue", "retrieve_law")
    branch.add_edge("retrieve_law", "retrieve_facts")
    branch.add_edge("retrieve_facts", "classify_evidence")
    branch.add_edge("classify_evidence", "apply_law")
    branch.add_edge("apply_law", "generate_counterarguments")
    branch.add_edge("generate_counterarguments", "validate_analysis")
    branch.add_edge("validate_analysis", "package_result")
    branch.add_edge("package_result", END)

    return branch.compile()


def build_case_reasoner_graph():
    """Compile the main Case Reasoner graph with parallel issue branches."""
    issue_branch = build_issue_branch()

    workflow = StateGraph(CaseReasonerState)

    workflow.add_node("extract_issues", extract_issues_node)
    workflow.add_node("issue_analysis_branch", issue_branch)
    workflow.add_node("aggregate_issues", aggregate_issues_node)
    workflow.add_node("check_global_consistency", check_global_consistency_node)
    workflow.add_node("compute_confidence", compute_confidence_node)
    workflow.add_node("generate_report", generate_report_node)
    workflow.add_node("generate_empty_report", generate_empty_report_node)

    workflow.add_edge(START, "extract_issues")

    workflow.add_conditional_edges(
        "extract_issues",
        issue_dispatch_router,
        {"generate_empty_report": "generate_empty_report"},
    )

    workflow.add_edge("issue_analysis_branch", "aggregate_issues")
    workflow.add_edge("aggregate_issues", "check_global_consistency")
    workflow.add_edge("check_global_consistency", "compute_confidence")
    workflow.add_edge("compute_confidence", "generate_report")
    workflow.add_edge("generate_report", END)
    workflow.add_edge("generate_empty_report", END)

    return workflow.compile()
