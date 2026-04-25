"""Final Report Node — assembles the 8-section Arabic legal analysis report."""
import logging
from typing import Any, Dict, List

from config import get_llm

logger = logging.getLogger(__name__)

_REPORT_SYSTEM = """أنت كاتب قضائي متخصص. أنشئ تقرير تحليل قانوني شامل باللغة العربية الفصحى.
التقرير يجب أن يتضمن الأقسام الثمانية المحددة بالترتيب.
قواعد صارمة واجبة التطبيق:
1. لا تصدر أي حكم ولا توصِ بنتيجة لصالح أي طرف.
2. لا تستخدم لغة اتجاهية كـ "يثبت الحق" أو "تُرفض الدعوى".
3. لا تخترع مواد قانونية أو وقائع — كل ما تكتبه يجب أن يستند إلى البيانات المقدمة.
4. القسم السادس (حالة الملف) يجب أن يكون محايدًا تمامًا بدون أي تركيب أو استنتاج.
5. القسم الثامن (فقرات التوفيق) يُكتب فقط إذا وُجدت تناقضات."""

_REPORT_USER = """البيانات:

=== المسائل المحددة ===
{issues_text}

=== التحليلات ===
{analyses_text}

=== مستوى الثقة ===
{confidence_text}

=== التناقضات والتوفيق ===
{reconciliation_text}

أنشئ التقرير بالهيكل الآتي:
# القسم الأول: المسائل القانونية المحددة
# القسم الثاني: الإطار القانوني
# القسم الثالث: التحليل عنصرًا بعنصر
# القسم الرابع: جدول تصنيف الأدلة
# القسم الخامس: الحجج المقابلة
# القسم السادس: حالة الملف
# القسم السابع: مستوى الثقة
# القسم الثامن: فقرات التوفيق (إن وُجدت)"""


def _format_issues(issues: List[Dict]) -> str:
    return "\n".join(f"{i+1}. [{iss['issue_id']}] {iss['issue_title']} — {iss.get('legal_domain','')}" for i, iss in enumerate(issues))


def _format_analyses(issue_analyses: List[Dict]) -> str:
    parts = []
    for a in issue_analyses:
        applied_text = "\n".join(
            f"  - [{el['element_id']}]: {el['reasoning'][:400]} (مواد: {el.get('cited_articles',[])})"
            for el in (a.get("applied_elements") or [])
        )
        counterargs = a.get("counterarguments") or {}
        parts.append(
            f"مسألة [{a.get('issue_id')}]: {a.get('issue_title','')}\n"
            f"التحليل:\n{applied_text or '(لا تحليل)'}\n"
            f"المدعي: {'; '.join(counterargs.get('plaintiff_arguments',[]))}\n"
            f"المدعى عليه: {'; '.join(counterargs.get('defendant_arguments',[]))}"
        )
    return "\n\n".join(parts)


def _format_confidence(per_issue: List[Dict], case_level: Dict) -> str:
    level_labels = {"high": "مرتفع", "medium": "متوسط", "low": "منخفض"}
    lines = [f"مستوى الثقة الإجمالي: {level_labels.get(case_level.get('level',''), '')} ({case_level.get('raw_score', 0):.2f})"]
    lines.append(case_level.get("justification", ""))
    for pc in per_issue:
        lines.append(f"- مسألة {pc['issue_id']}: {level_labels.get(pc.get('level',''), '')} — {pc.get('justification','')[:200]}")
    return "\n".join(lines)


def _build_fallback_report(state: Dict[str, Any]) -> str:
    """Minimal structured report from raw state when LLM fails."""
    issues = state.get("identified_issues") or []
    analyses = state.get("issue_analyses") or []
    confidence = state.get("case_level_confidence") or {}
    level_labels = {"high": "مرتفع", "medium": "متوسط", "low": "منخفض"}

    lines = ["# تقرير التحليل القانوني\n", "## القسم الأول: المسائل القانونية المحددة"]
    for iss in issues:
        lines.append(f"- [{iss['issue_id']}] {iss['issue_title']}")
    lines.append("\n## القسم الثاني: الإطار القانوني")
    for a in analyses:
        articles = []
        for el in (a.get("applied_elements") or []):
            articles.extend(el.get("cited_articles") or [])
        lines.append(f"- مسألة {a.get('issue_id')}: مواد {list(set(articles))}")
    lines.append(f"\n## القسم السابع: مستوى الثقة\n{level_labels.get(confidence.get('level',''), '')} ({confidence.get('raw_score', 0):.2f})")
    return "\n".join(lines)


def generate_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    issues = state.get("identified_issues") or []
    issue_analyses = state.get("issue_analyses") or []
    per_issue_confidence = state.get("per_issue_confidence") or []
    case_level_confidence = state.get("case_level_confidence") or {}
    consistency_conflicts = state.get("consistency_conflicts") or []
    reconciliation_paragraphs = state.get("reconciliation_paragraphs") or []

    issues_text = _format_issues(issues)
    analyses_text = _format_analyses(issue_analyses)
    confidence_text = _format_confidence(per_issue_confidence, case_level_confidence)
    reconciliation_text = (
        "\n\n".join(reconciliation_paragraphs)
        if reconciliation_paragraphs
        else "لا تناقضات — القسم الثامن لا ينطبق"
    )

    prompt = (
        f"{_REPORT_SYSTEM}\n\n"
        f"{_REPORT_USER.format(issues_text=issues_text, analyses_text=analyses_text, confidence_text=confidence_text, reconciliation_text=reconciliation_text)}"
    )

    try:
        llm = get_llm("high")
        report = llm.invoke(prompt).content
        logger.info("Final report generated: %d chars", len(report))
        return {
            "final_report": report,
            "intermediate_steps": ["التقرير النهائي: اكتمل"],
        }
    except Exception as exc:
        logger.error("generate_report_node failed — using fallback: %s", exc)
        fallback = _build_fallback_report(state)
        return {
            "final_report": fallback,
            "error_log": [f"generate_report_node: {exc} — تم استخدام التقرير الاحتياطي"],
            "intermediate_steps": ["التقرير النهائي: فشل، احتياطي"],
        }


def generate_empty_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_report": "لم يتم استخراج مسائل قانونية من ملخص الدعوى.",
        "intermediate_steps": ["التقرير النهائي: لا مسائل"],
    }
