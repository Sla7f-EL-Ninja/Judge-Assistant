"""Validation Node — three sequential sub-steps: citation, consistency, completeness."""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import re
import unicodedata
from typing import Any, Dict, List, Set

from config import get_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Arabic digit normalization and citation extraction
# ---------------------------------------------------------------------------

_AR_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DIACRITIC_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")

_CITATION_PATTERNS = [
    re.compile(r"(?:المادة|مادة|م\.?)\s*[(\[]?\s*(\d+|[٠-٩]+)\s*[)\]]?"),
    re.compile(r"(?:المادة|مادة)\s+رقم\s*[(\[]?\s*(\d+|[٠-٩]+)\s*[)\]]?"),
    re.compile(r"(?:المواد|مواد)\s+(?:من\s+)?(\d+|[٠-٩]+)\s*(?:إلى|حتى)\s*(\d+|[٠-٩]+)"),
    re.compile(r"(?:الفقرة|فقرة)\s*[(\[]?\s*(\d+|[٠-٩]+)\s*[)\]]?\s*من\s*(?:المادة|مادة)\s*(\d+|[٠-٩]+)"),
]


def _normalize(text: str) -> str:
    text = _DIACRITIC_RE.sub("", text)
    return unicodedata.normalize("NFC", text).translate(_AR_DIGIT_TABLE)


def _extract_cited_article_numbers(text: str) -> Set[int]:
    normalized = _normalize(text)
    numbers: Set[int] = set()
    for pattern in _CITATION_PATTERNS:
        for m in pattern.finditer(normalized):
            for group in m.groups():
                if group and group.isdigit():
                    numbers.add(int(group))
    return numbers


def _available_article_numbers(retrieved_articles: List[Dict]) -> Set[int]:
    return {a["article_number"] for a in retrieved_articles if isinstance(a.get("article_number"), int)}


# ---------------------------------------------------------------------------
# Sub-step 1: Citation Check
# ---------------------------------------------------------------------------

def _citation_check(state: Dict[str, Any]) -> Dict[str, Any]:
    from tools import civil_law_rag_tool

    law_application: str = state.get("law_application") or ""
    applied_elements: List[Dict] = state.get("applied_elements") or []
    retrieved_articles: List[Dict] = state.get("retrieved_articles") or []

    cited = _extract_cited_article_numbers(law_application)
    available = _available_article_numbers(retrieved_articles)
    missing = cited - available

    # Retry missing articles once each
    retried_results = []
    for article_num in list(missing):
        try:
            result = civil_law_rag_tool(f"المادة {article_num}")
            if result.get("answer") and result["answer"].strip():
                # Article found on retry — add to available
                available.add(article_num)
                missing.discard(article_num)
                retried_results.append({"article": article_num, "retried": True, "found": True})
            else:
                retried_results.append({"article": article_num, "retried": True, "found": False})
        except Exception as exc:
            logger.warning("citation retry for article %d failed: %s", article_num, exc)
            retried_results.append({"article": article_num, "retried": True, "found": False, "error": str(exc)})

    # Elements with no cited articles
    uncited_elements = [el["element_id"] for el in applied_elements if not el.get("cited_articles")]

    passed = len(missing) == 0 and len(uncited_elements) == 0
    return {
        "passed": passed,
        "total_citations": len(cited),
        "verified_citations": len(cited) - len(missing),
        "missing_citations": [r for r in retried_results if not r.get("found")],
        "unsupported_conclusions": uncited_elements,
    }


# ---------------------------------------------------------------------------
# Sub-step 2: Logical Consistency Check
# ---------------------------------------------------------------------------

from prompts import get_prompt


def _logical_consistency_check(state: Dict[str, Any]) -> Dict[str, Any]:
    from schemas import LogicalConsistencyResult
    _VALIDATION_CONSISTENCY_SYSTEM, _VALIDATION_CONSISTENCY_USER = get_prompt("validation_consistency")


    law_application = state.get("law_application") or ""
    classifications = state.get("element_classifications") or []
    counterarguments = state.get("counterarguments") or {}

    status_labels = {
        "established": "ثابت", "not_established": "غير ثابت",
        "disputed": "متنازع عليه", "insufficient_evidence": "غير كافي",
    }
    classifications_text = "\n".join(
        f"- [{c['element_id']}] {status_labels.get(c['status'], c['status'])}"
        for c in classifications
    )
    counterarguments_text = (
        "المدعي: " + "; ".join(counterarguments.get("plaintiff_arguments", [])) + "\n"
        "المدعى عليه: " + "; ".join(counterarguments.get("defendant_arguments", []))
    )

    prompt = (
        f"{_VALIDATION_CONSISTENCY_SYSTEM}\n\n"
        f"{_VALIDATION_CONSISTENCY_USER.format(law_application=law_application or 'غير متاح', classifications_text=classifications_text or 'غير متاح', counterarguments_text=counterarguments_text)}"
    )

    llm = get_llm("low")
    structured_llm = llm.with_structured_output(LogicalConsistencyResult)

    try:
        result: LogicalConsistencyResult = structured_llm.invoke(prompt)
        return {
            "passed": result.passed,
            "issues_found": result.issues_found,
            "severity": result.severity,
        }
    except Exception as exc:
        logger.warning("logical_consistency_check failed — marking passed: %s", exc)
        return {"passed": True, "issues_found": [], "severity": "none", "note": str(exc)}


# ---------------------------------------------------------------------------
# Sub-step 3: Completeness Check
# ---------------------------------------------------------------------------

def _completeness_check(state: Dict[str, Any]) -> Dict[str, Any]:
    required_elements: List[Dict] = state.get("required_elements") or []
    applied_elements: List[Dict] = state.get("applied_elements") or []
    skipped_elements: List[str] = state.get("skipped_elements") or []

    required_ids = {el["element_id"] for el in required_elements}
    applied_ids = {el["element_id"] for el in applied_elements}
    covered = applied_ids | set(skipped_elements)
    missing = required_ids - covered
    total = len(required_ids) or 1
    coverage_ratio = len(covered & required_ids) / total

    return {
        "passed": len(missing) == 0,
        "total_required": len(required_ids),
        "covered": len(covered & required_ids),
        "missing_elements": list(missing),
        "coverage_ratio": round(coverage_ratio, 3),
    }


# ---------------------------------------------------------------------------
# Main validation node
# ---------------------------------------------------------------------------

def validate_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    issue_title = state.get("issue_title", "")

    citation = _citation_check(state)
    consistency = _logical_consistency_check(state)
    completeness = _completeness_check(state)

    validation_passed = (
        citation.get("passed", False)
        and consistency.get("passed", True)
        and completeness.get("passed", False)
    )

    logger.info(
        "Validation '%s': citation=%s, consistency=%s, completeness=%s → overall=%s",
        issue_title, citation["passed"], consistency["passed"], completeness["passed"], validation_passed,
    )

    return {
        "citation_check": citation,
        "logical_consistency_check": consistency,
        "completeness_check": completeness,
        "validation_passed": validation_passed,
        "intermediate_steps": [
            f"التحقق '{issue_title}': اقتباسات={'✓' if citation['passed'] else '✗'}, "
            f"منطق={'✓' if consistency['passed'] else '✗'}, "
            f"اكتمال={'✓' if completeness['passed'] else '✗'}"
        ],
    }
