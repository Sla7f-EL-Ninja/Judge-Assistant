"""
verify_citations.py

Citation-verification node for the Supervisor workflow.

Extracts Arabic article citations (مادة X) from the merged response and
checks them against the raw civil-law agent output.  Appends a warning to
``validation_feedback`` when unknown article numbers are detected so that
the downstream validator can mark hallucination_pass=False.
"""

import logging
import re
from typing import Any, Dict, List, Set

from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

_ARTICLE_RE = re.compile(r"المادة\s+(\d+)", re.UNICODE)


def _extract_article_numbers(text: str) -> Set[str]:
    return set(_ARTICLE_RE.findall(text))


def verify_citations_node(state: SupervisorState) -> Dict[str, Any]:
    """Cross-reference cited article numbers against civil_law_rag raw output.

    If articles are cited in merged_response that do not appear in the
    civil_law_rag raw output, append a citation warning to validation_feedback
    so the validator's hallucination check has a concrete signal (G5.7.2).
    """
    merged = state.get("merged_response", "")
    if not merged:
        return {}

    cited = _extract_article_numbers(merged)
    if not cited:
        return {}

    # Collect article numbers that appear in any agent's raw source material
    known: Set[str] = set()
    agent_results = state.get("agent_results") or {}
    for agent_name, result in agent_results.items():
        raw = result.get("raw_output") or {}
        # civil_law_rag stores structured sources
        for src in result.get("sources", []):
            known.update(_extract_article_numbers(str(src)))
        # Also scan raw_output blob
        known.update(_extract_article_numbers(str(raw)))
        # Scan the agent response text
        known.update(_extract_article_numbers(result.get("response", "")))

    unknown = cited - known
    if not unknown:
        return {}

    warning = (
        f"تحذير: الإجابة تستشهد بمواد ({', '.join(sorted(unknown, key=int))}) "
        "غير موجودة في المصادر المسترجعة — قد تكون هلوسة."
    )
    existing_feedback = state.get("validation_feedback", "")
    combined = f"{existing_feedback}\n{warning}".strip() if existing_feedback else warning

    logger.warning("Unknown cited articles detected: %s", sorted(unknown, key=int))
    return {"validation_feedback": combined}
