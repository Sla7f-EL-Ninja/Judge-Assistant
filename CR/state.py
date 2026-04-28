import operator
from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict


def _last_value(existing: Any, new: Any) -> Any:
    return new


# ---------------------------------------------------------------------------
# Reducer helpers
# ---------------------------------------------------------------------------


_STEP_RESULTS_RESET = {"__reset__": True}


def _add_or_reset_step_results(a: List[dict], b: List[dict]) -> List[dict]:
    """Append reducer with reset support.

    An empty b list signals a full reset (used by replanner to wipe old results
    before a new execution wave). A non-empty b is appended to a as usual.
    Legacy sentinel check retained for backwards compatibility.
    """
    if not b:
        return []
    if b[0] == _STEP_RESULTS_RESET:
        return list(b[1:])
    return a + b


def _merge_step_failures(
    a: Dict[str, int], b: Dict[str, int]
) -> Dict[str, int]:
    """Merge two step-failure dicts, keeping the higher count per step_id.

    Used as a state reducer so parallel Send-dispatched step_workers can each
    contribute their failure increments without clobbering sibling counters.
    """
    out = dict(a)
    for k, v in b.items():
        out[k] = max(out.get(k, 0), v)
    return out


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class CaseReasonerState(TypedDict):
    # Input — set by adapter, never modified
    case_id: Annotated[str, _last_value]
    judge_query: str
    case_brief: Dict[str, str]          # CaseBrief serialized: 7 Arabic prose fields
    rendered_brief: str

    # Issue Extraction output
    identified_issues: List[Dict[str, Any]]  # [{issue_id, issue_title, legal_domain, source_text}]

    # Per-issue branch results — merged via operator.add
    issue_analyses: Annotated[List[Dict[str, Any]], operator.add]

    # Aggregation output
    cross_issue_relationships: List[Dict[str, Any]]

    # Global Consistency output
    consistency_conflicts: List[Dict[str, Any]]
    reconciliation_paragraphs: List[str]

    # Confidence output
    per_issue_confidence: List[Dict[str, Any]]
    case_level_confidence: Dict[str, Any]

    # Final output
    final_report: str

    # Telemetry — accumulated across all nodes
    intermediate_steps: Annotated[List[str], operator.add]
    error_log: Annotated[List[str], operator.add]


class IssueAnalysisState(TypedDict):
    # Propagated from main state (read-only in branch)
    case_id: Annotated[str, _last_value]

    # Issue identity
    issue_id: int
    issue_title: str
    legal_domain: str
    source_text: str    # Exact Arabic excerpt from brief that raised this issue

    # Decomposition
    required_elements: List[Dict[str, Any]]    # [{element_id, description, element_type}]

    # Law Retrieval
    law_retrieval_result: Dict[str, Any]
    retrieved_articles: List[Dict[str, Any]]   # [{article_number, article_text, title, book, ...}]

    # Fact Retrieval
    fact_retrieval_result: Dict[str, Any]
    retrieved_facts: str                        # final_answer from case_doc_rag

    # Evidence Sufficiency
    element_classifications: List[Dict[str, Any]]  # [{element_id, status, evidence_summary, notes}]
    # status values (English): established | not_established | disputed | insufficient_evidence

    # Law Application
    law_application: str                         # Arabic reasoning prose
    applied_elements: List[Dict[str, Any]]       # [{element_id, reasoning, cited_articles: List[int]}]
    skipped_elements: List[str]                  # element_ids excluded (insufficient_evidence)

    # Counterarguments
    counterarguments: Dict[str, Any]             # {plaintiff_arguments, defendant_arguments, analysis}

    # Validation
    citation_check: Dict[str, Any]
    logical_consistency_check: Dict[str, Any]
    completeness_check: Dict[str, Any]
    validation_passed: bool

    # Merge channel back to main state
    issue_analyses: Annotated[List[Dict[str, Any]], operator.add]

    # Branch telemetry
    intermediate_steps: Annotated[List[str], operator.add]
    error_log: Annotated[List[str], operator.add]
