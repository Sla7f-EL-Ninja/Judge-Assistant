"""Package Result Node — wraps branch state into a single dict for merge via operator.add."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from typing import Any, Dict


def package_result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "issue_analyses": [
            {
                "issue_id": state.get("issue_id"),
                "issue_title": state.get("issue_title"),
                "legal_domain": state.get("legal_domain"),
                "source_text": state.get("source_text"),
                "required_elements": state.get("required_elements") or [],
                "law_retrieval_result": state.get("law_retrieval_result") or {},
                "retrieved_articles": state.get("retrieved_articles") or [],
                "retrieved_facts": state.get("retrieved_facts") or "",
                "element_classifications": state.get("element_classifications") or [],
                "law_application": state.get("law_application") or "",
                "applied_elements": state.get("applied_elements") or [],
                "skipped_elements": state.get("skipped_elements") or [],
                "counterarguments": state.get("counterarguments") or {},
                "citation_check": state.get("citation_check") or {},
                "logical_consistency_check": state.get("logical_consistency_check") or {},
                "completeness_check": state.get("completeness_check") or {},
                "validation_passed": state.get("validation_passed", False),
            }
        ],
    }
