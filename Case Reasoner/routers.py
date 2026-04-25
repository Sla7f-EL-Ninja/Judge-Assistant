"""Routing functions for the Case Reasoner graph."""
import logging
from typing import Any, Dict, List, Union

from langgraph.types import Send

logger = logging.getLogger(__name__)

_MAX_ISSUES_WARNING = 8  # warn if extraction produces more than this


def issue_dispatch_router(state: Dict[str, Any]) -> Union[str, List[Send]]:
    """Fan out one branch per identified issue via Send().

    Returns "generate_empty_report" when no issues were extracted.
    Otherwise returns a list of Send objects — one per issue — targeting
    the compiled branch sub-graph node "issue_analysis_branch".
    """
    issues = state.get("identified_issues") or []

    if not issues:
        logger.warning("issue_dispatch_router: no issues extracted — routing to empty report")
        return "generate_empty_report"

    if len(issues) > _MAX_ISSUES_WARNING:
        logger.warning(
            "issue_dispatch_router: %d issues extracted (>%d) — may impact latency",
            len(issues), _MAX_ISSUES_WARNING,
        )

    sends: List[Send] = []
    for issue in issues:
        sends.append(
            Send(
                "issue_analysis_branch",
                {
                    # Identity
                    "case_id": state.get("case_id", ""),
                    "issue_id": issue["issue_id"],
                    "issue_title": issue["issue_title"],
                    "legal_domain": issue["legal_domain"],
                    "source_text": issue["source_text"],
                    # Output fields — initialized empty
                    "required_elements": [],
                    "law_retrieval_result": {},
                    "retrieved_articles": [],
                    "fact_retrieval_result": {},
                    "retrieved_facts": "",
                    "element_classifications": [],
                    "law_application": "",
                    "applied_elements": [],
                    "skipped_elements": [],
                    "counterarguments": {},
                    "citation_check": {},
                    "logical_consistency_check": {},
                    "completeness_check": {},
                    "validation_passed": False,
                    # Reducer fields
                    "issue_analyses": [],
                    "intermediate_steps": [],
                    "error_log": [],
                },
            )
        )

    logger.info("issue_dispatch_router: dispatching %d issue branches", len(sends))
    return sends
