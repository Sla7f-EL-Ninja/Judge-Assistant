"""Issue Decomposition Node — breaks each legal issue into required elements."""
import logging
from typing import Any, Dict

from config import get_llm
from ..prompts import get_prompt

logger = logging.getLogger(__name__)


def decompose_issue_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from ..schemas import DecomposedIssue
    _DECOMPOSITION_SYSTEM, _DECOMPOSITION_USER = get_prompt("decomposition")

    issue_title = state.get("issue_title", "")
    legal_domain = state.get("legal_domain", "")
    source_text = state.get("source_text", "")

    llm = get_llm("high")
    structured_llm = llm.with_structured_output(DecomposedIssue)
    prompt = f"{_DECOMPOSITION_SYSTEM}\n\n{_DECOMPOSITION_USER.format(issue_title=issue_title, legal_domain=legal_domain, source_text=source_text)}"

    try:
        result: DecomposedIssue = structured_llm.invoke(prompt)
        elements = [
            {"element_id": el.element_id, "description": el.description, "element_type": el.element_type}
            for el in result.elements
        ]
        logger.info("Decomposed issue '%s' into %d elements", issue_title, len(elements))
        return {
            "required_elements": elements,
            "intermediate_steps": [f"تحليل مسألة '{issue_title}': {len(elements)} عناصر"],
        }
    except Exception as exc:
        logger.warning("decompose_issue_node failed for '%s': %s — using fallback", issue_title, exc)
        fallback = [{"element_id": "E0", "description": issue_title, "element_type": "legal"}]
        return {
            "required_elements": fallback,
            "error_log": [f"decompose_issue_node '{issue_title}': {exc} — تم استخدام عنصر احتياطي"],
            "intermediate_steps": [f"تحليل مسألة '{issue_title}': فشل، عنصر احتياطي"],
        }