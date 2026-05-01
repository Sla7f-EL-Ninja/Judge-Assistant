"""Counterargument Node — surfaces strongest arguments for each party."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from typing import Any, Dict

from config import get_llm

logger = logging.getLogger(__name__)

from prompts import get_prompt


def _format_classifications(classifications: list) -> str:
    status_labels = {
        "established": "ثابت",
        "not_established": "غير ثابت",
        "disputed": "متنازع عليه",
        "insufficient_evidence": "غير كافي الأدلة",
    }
    return "\n".join(
        f"- [{c['element_id']}] {status_labels.get(c['status'], c['status'])}: {c['evidence_summary']}"
        for c in classifications
    )


def counterargument_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from schemas import Counterarguments
    _COUNTERARGUMENT_SYSTEM, _COUNTERARGUMENT_USER = get_prompt("counterargument")


    law_application: str = state.get("law_application") or ""
    classifications: list = state.get("element_classifications") or []
    retrieved_facts: str = state.get("retrieved_facts") or ""
    issue_title: str = state.get("issue_title", "")

    classifications_text = _format_classifications(classifications)
    prompt = f"{_COUNTERARGUMENT_SYSTEM}\n\n{_COUNTERARGUMENT_USER.format(law_application=law_application or 'غير متاح', classifications_text=classifications_text or 'غير متاح', retrieved_facts=retrieved_facts or 'غير متاح')}"

    llm = get_llm("high")
    structured_llm = llm.with_structured_output(Counterarguments)

    try:
        result: Counterarguments = structured_llm.invoke(prompt)
        counterargs = {
            "plaintiff_arguments": result.plaintiff_arguments,
            "defendant_arguments": result.defendant_arguments,
            "analysis": result.analysis,
        }
        logger.info("Counterarguments for '%s': %d plaintiff, %d defendant",
                    issue_title, len(result.plaintiff_arguments), len(result.defendant_arguments))
        return {
            "counterarguments": counterargs,
            "intermediate_steps": [f"الحجج المقابلة '{issue_title}': اكتمل"],
        }
    except Exception as exc:
        logger.warning("counterargument_node failed for '%s': %s", issue_title, exc)
        return {
            "counterarguments": {
                "plaintiff_arguments": [],
                "defendant_arguments": [],
                "analysis": "لم يتم إنشاء الحجج المقابلة.",
            },
            "error_log": [f"counterargument_node '{issue_title}': {exc}"],
            "intermediate_steps": [f"الحجج المقابلة '{issue_title}': فشل"],
        }
