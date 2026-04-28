"""Confidence Scoring Node — rule-based signals + LLM justification text."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from typing import Any, Dict, List

from config import get_llm

logger = logging.getLogger(__name__)

_JUSTIFICATION_SYSTEM = """أنت محلل قانوني. اكتب فقرة مختصرة باللغة العربية تبرر مستوى الثقة المحدد
بناءً على الإشارات الكمية المعطاة. لا تحدد مستوى الثقة بنفسك — فقط اشرح لماذا المستوى المعطى مناسب."""

_JUSTIFICATION_USER = """مستوى الثقة: {level}
الإشارات:
{signals_text}

اكتب فقرة توضيحية مختصرة."""


def _compute_issue_signals(analysis: Dict, conflict_issue_ids: set) -> Dict[str, float]:
    classifications: List[Dict] = analysis.get("element_classifications") or []
    applied_elements: List[Dict] = analysis.get("applied_elements") or []
    citation_check: Dict = analysis.get("citation_check") or {}
    consistency_check: Dict = analysis.get("logical_consistency_check") or {}
    completeness_check: Dict = analysis.get("completeness_check") or {}

    total_elements = len(analysis.get("required_elements") or []) or 1
    total_citations = citation_check.get("total_citations") or 1

    unsupported = len(citation_check.get("unsupported_conclusions") or [])
    disputed = sum(1 for c in classifications if c.get("status") == "disputed")
    insufficient = sum(1 for c in classifications if c.get("status") == "insufficient_evidence")
    citation_failures = len(citation_check.get("missing_citations") or [])
    severity_map = {"none": 0.0, "minor": 0.5, "major": 1.0}
    logical = severity_map.get(consistency_check.get("severity", "none"), 0.0)
    completeness_gap = 1.0 - (completeness_check.get("coverage_ratio") or 1.0)
    reconciliation = 1.0 if analysis.get("issue_id") in conflict_issue_ids else 0.0

    return {
        "unsupported_ratio": unsupported / total_elements,
        "disputed_ratio": disputed / total_elements,
        "insufficient_ratio": insufficient / total_elements,
        "citation_failure_ratio": citation_failures / total_citations,
        "logical_issues": logical,
        "completeness_gap": completeness_gap,
        "reconciliation_triggered": reconciliation,
    }


def _level_from_score(score: float, thresholds: Dict) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"


def _arabic_level(level: str) -> str:
    return {"high": "مرتفع", "medium": "متوسط", "low": "منخفض"}.get(level, level)


def compute_confidence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from cr_config import CONFIDENCE_WEIGHTS, CONFIDENCE_THRESHOLDS

    issue_analyses: List[Dict] = state.get("issue_analyses") or []
    consistency_conflicts: List[Dict] = state.get("consistency_conflicts") or []

    conflict_issue_ids = set()
    for c in consistency_conflicts:
        conflict_issue_ids.update(c.get("issue_ids") or [])

    llm = get_llm("medium")
    per_issue_confidence: List[Dict] = []
    raw_scores: List[float] = []

    for analysis in issue_analyses:
        signals = _compute_issue_signals(analysis, conflict_issue_ids)
        penalty = sum(signals[k] * CONFIDENCE_WEIGHTS.get(k, 0.0) for k in signals)
        raw_score = max(0.0, 1.0 - penalty)
        level = _level_from_score(raw_score, CONFIDENCE_THRESHOLDS)
        raw_scores.append(raw_score)

        # LLM justification (text only — level already determined)
        signals_text = "\n".join(f"- {k}: {v:.2f}" for k, v in signals.items())
        try:
            prompt = (
                f"{_JUSTIFICATION_SYSTEM}\n\n"
                f"{_JUSTIFICATION_USER.format(level=_arabic_level(level), signals_text=signals_text)}"
            )
            justification = llm.invoke(prompt).content
        except Exception as exc:
            logger.warning("confidence justification failed for issue %s: %s", analysis.get("issue_id"), exc)
            justification = f"تعذّر إنشاء المبرر: {exc}"

        per_issue_confidence.append({
            "issue_id": analysis.get("issue_id"),
            "issue_title": analysis.get("issue_title"),
            "level": level,
            "raw_score": round(raw_score, 3),
            "signals": {k: round(v, 3) for k, v in signals.items()},
            "justification": justification,
        })

    # Case-level aggregation: 70% min + 30% mean
    if raw_scores:
        case_score = 0.7 * min(raw_scores) + 0.3 * (sum(raw_scores) / len(raw_scores))
    else:
        case_score = 0.0
    case_level = _level_from_score(case_score, CONFIDENCE_THRESHOLDS)

    try:
        signals_text = "\n".join(
            f"- مسألة {pc['issue_id']} ({pc['issue_title']}): {pc['raw_score']:.2f} ({_arabic_level(pc['level'])})"
            for pc in per_issue_confidence
        )
        case_prompt = (
            f"{_JUSTIFICATION_SYSTEM}\n\n"
            f"{_JUSTIFICATION_USER.format(level=_arabic_level(case_level), signals_text=signals_text)}"
        )
        case_justification = llm.invoke(case_prompt).content
    except Exception as exc:
        logger.warning("case-level confidence justification failed: %s", exc)
        case_justification = f"تعذّر إنشاء المبرر: {exc}"

    case_level_confidence = {
        "level": case_level,
        "raw_score": round(case_score, 3),
        "justification": case_justification,
        "per_issue_scores": [{pc["issue_id"]: pc["raw_score"]} for pc in per_issue_confidence],
    }

    logger.info("Confidence: case-level=%s (%.3f), %d issues scored", case_level, case_score, len(per_issue_confidence))
    return {
        "per_issue_confidence": per_issue_confidence,
        "case_level_confidence": case_level_confidence,
        "intermediate_steps": [f"مستوى الثقة: {_arabic_level(case_level)} ({case_score:.2f})"],
    }
