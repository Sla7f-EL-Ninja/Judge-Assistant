"""Global Consistency Node — detects cross-issue conflicts and writes reconciliation."""
import logging
from typing import Any, Dict, List

from config import get_llm
from ..prompts import get_prompt

logger = logging.getLogger(__name__)


def _format_analyses_summary(issue_analyses: List[Dict]) -> str:
    parts = []
    for a in issue_analyses:
        parts.append(
            f"[مسألة {a.get('issue_id')}] {a.get('issue_title', '')}\n"
            f"  التحليل: {(a.get('law_application') or '')[:300]}...\n"
            f"  المواد المستشهد بها: {[el.get('cited_articles') for el in a.get('applied_elements') or []]}"
        )
    return "\n\n".join(parts)


def _format_relationships(relationships: List[Dict]) -> str:
    if not relationships:
        return "لا علاقات مكتشفة"
    return "\n".join(
        f"- {r['type']}: مسائل {r.get('issue_ids', [r.get('upstream'), r.get('downstream')])}"
        for r in relationships
    )


def check_global_consistency_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from ..schemas import ConsistencyCheckResult
    _CONSISTENCY_CONFLICT_SYSTEM, _CONSISTENCY_CONFLICT_USER = get_prompt("consistency_conflict")
    _CONSISTENCY_RECONCILIATION_SYSTEM, _CONSISTENCY_RECONCILIATION_USER = get_prompt("consistency_reconciliation")

    issue_analyses: List[Dict] = state.get("issue_analyses") or []
    relationships: List[Dict] = state.get("cross_issue_relationships") or []

    if len(issue_analyses) < 2:
        return {
            "consistency_conflicts": [],
            "reconciliation_paragraphs": [],
            "intermediate_steps": ["الاتساق العام: مسألة واحدة — لا تحقق مطلوب"],
        }

    analyses_summary = _format_analyses_summary(issue_analyses)
    relationships_summary = _format_relationships(relationships)
    conflict_prompt = (
        f"{_CONSISTENCY_CONFLICT_SYSTEM}\n\n"
        f"{_CONSISTENCY_CONFLICT_USER.format(analyses_summary=analyses_summary, relationships_summary=relationships_summary)}"
    )

    try:
        llm = get_llm("high")
        structured_llm = llm.with_structured_output(ConsistencyCheckResult)
        conflict_result: ConsistencyCheckResult = structured_llm.invoke(conflict_prompt)
        conflicts = [
            {"issue_ids": c.issue_ids, "conflict_type": c.conflict_type, "description": c.description}
            for c in conflict_result.conflicts
        ]
    except Exception as exc:
        logger.warning("global consistency conflict detection failed — assuming no conflicts: %s", exc)
        conflicts = []

    reconciliation_paragraphs: List[str] = []
    if conflicts:
        llm_high = get_llm("high")
        issue_map = {a["issue_id"]: a for a in issue_analyses}
        for conflict in conflicts:
            try:
                affected = [
                    f"[{iid}] {issue_map.get(iid, {}).get('issue_title', '')}: "
                    f"{(issue_map.get(iid, {}).get('law_application') or '')[:200]}"
                    for iid in conflict["issue_ids"]
                ]
                rec_prompt = (
                    f"{_CONSISTENCY_RECONCILIATION_SYSTEM}\n\n"
                    f"{_CONSISTENCY_RECONCILIATION_USER.format(conflict_description=conflict['description'], affected_analyses=chr(10).join(affected))}"
                )
                paragraph = llm_high.invoke(rec_prompt).content
                reconciliation_paragraphs.append(paragraph)
            except Exception as exc:
                logger.warning("reconciliation paragraph failed for conflict: %s", exc)
                reconciliation_paragraphs.append(f"[تعذّر كتابة فقرة التوفيق: {exc}]")

    logger.info("Global consistency: %d conflicts, %d reconciliation paragraphs",
                len(conflicts), len(reconciliation_paragraphs))
    return {
        "consistency_conflicts": conflicts,
        "reconciliation_paragraphs": reconciliation_paragraphs,
        "intermediate_steps": [f"الاتساق العام: {len(conflicts)} تناقض، {len(reconciliation_paragraphs)} فقرة توفيق"],
    }