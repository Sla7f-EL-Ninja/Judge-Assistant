"""Aggregation Node — detects cross-issue relationships after all branches merge."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List

from config import get_llm

logger = logging.getLogger(__name__)

# Simple named-entity patterns for shared-fact detection
_ENTITY_PATTERNS = [
    re.compile(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"),           # dates
    re.compile(r"\d[\d,.]* (?:جنيه|دولار|يورو)"),               # monetary amounts
    re.compile(r"(?:السيد|السيدة|الأستاذ|الشركة)\s+\S+"),        # named parties
]

from prompts import get_prompt


def _extract_entities(text: str) -> set:
    entities = set()
    for pattern in _ENTITY_PATTERNS:
        for m in pattern.findall(text or ""):
            entities.add(m.strip())
    return entities


def aggregate_issues_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from schemas import IssueDependencies
    _AGGREGATION_DEPENDENCY_SYSTEM, _AGGREGATION_DEPENDENCY_USER = get_prompt("aggregation_dependency")

    issue_analyses: List[Dict] = state.get("issue_analyses") or []
    identified_issues: List[Dict] = state.get("identified_issues") or []
    relationships: List[Dict] = []

    if not issue_analyses:
        return {
            "cross_issue_relationships": [],
            "intermediate_steps": ["التجميع: لا مسائل للتحليل"],
        }

    # --- Shared articles (rule-based) ---
    article_to_issues: Dict[int, List[int]] = defaultdict(list)
    for analysis in issue_analyses:
        issue_id = analysis.get("issue_id")
        for el in analysis.get("applied_elements") or []:
            for article_num in el.get("cited_articles") or []:
                article_to_issues[int(article_num)].append(issue_id)

    for article_num, issue_ids in article_to_issues.items():
        unique_ids = list(dict.fromkeys(issue_ids))  # preserve order, deduplicate
        if len(unique_ids) > 1:
            relationships.append({
                "type": "shared_article",
                "article_number": article_num,
                "issue_ids": unique_ids,
            })

    # --- Shared facts (entity overlap, rule-based) ---
    issue_entities: Dict[int, set] = {}
    for analysis in issue_analyses:
        issue_id = analysis.get("issue_id")
        issue_entities[issue_id] = _extract_entities(analysis.get("retrieved_facts", ""))

    issue_ids_list = [a.get("issue_id") for a in issue_analyses]
    for i, id_a in enumerate(issue_ids_list):
        for id_b in issue_ids_list[i + 1:]:
            overlap = issue_entities.get(id_a, set()) & issue_entities.get(id_b, set())
            if len(overlap) >= 2:
                relationships.append({
                    "type": "shared_fact",
                    "fact_entities": list(overlap)[:5],
                    "issue_ids": [id_a, id_b],
                })

    # --- Issue dependencies (LLM-assisted) ---
    try:
        issues_summary = "\n".join(
            f"[{iss['issue_id']}] {iss['issue_title']} — {iss.get('legal_domain', '')}"
            for iss in identified_issues
        )
        prompt = f"{_AGGREGATION_DEPENDENCY_SYSTEM}\n\n{_AGGREGATION_DEPENDENCY_USER.format(issues_summary=issues_summary)}"
        llm = get_llm("medium")
        structured_llm = llm.with_structured_output(IssueDependencies)
        dep_result: IssueDependencies = structured_llm.invoke(prompt)
        for dep in dep_result.dependencies:
            relationships.append({
                "type": "dependency",
                "upstream": dep.upstream_issue_id,
                "downstream": dep.downstream_issue_id,
                "dependency_type": dep.dependency_type,
                "explanation": dep.explanation,
            })
    except Exception as exc:
        logger.warning("aggregate_issues dependency detection failed: %s", exc)

    logger.info("Aggregation: %d cross-issue relationships found", len(relationships))
    return {
        "cross_issue_relationships": relationships,
        "intermediate_steps": [f"التجميع: {len(relationships)} علاقة عبر المسائل"],
    }
