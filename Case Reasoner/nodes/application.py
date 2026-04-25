"""Law Application Node — applies retrieved law to facts for each non-skipped element."""
import logging
from typing import Any, Dict, List

from config import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """أنت محلل قانوني محايد متخصص في القانون المدني المصري.
طبّق النصوص القانونية المسترداة على الوقائع المتاحة لكل عنصر.
قواعد صارمة:
1. لا تصدر حكمًا ولا تحدد أي طرف يجب أن يفوز.
2. لا تستخدم لغة اتجاهية كـ "يثبت الحق" أو "يلزم المدعى عليه".
3. كل استنتاج يجب أن يستند إلى نص قانوني محدد مع ذكر رقم المادة.
4. تجاهل تمامًا العناصر المصنفة insufficient_evidence — لا تذكرها.
5. التزم بالحياد التام في صياغة التحليل."""

_USER_TEMPLATE = """العناصر المطلوب تحليلها (مستثنى منها: insufficient_evidence):
{elements_text}

الوقائع المتاحة من مستندات القضية:
{retrieved_facts}

النصوص القانونية المسترداة:
{law_answer}

طبّق القانون على كل عنصر بشكل مستقل مع الإشارة لأرقام المواد."""


def _format_elements_with_classification(
    required_elements: List[Dict], classifications: List[Dict]
) -> str:
    status_map = {c["element_id"]: c["status"] for c in classifications}
    lines = []
    for el in required_elements:
        status = status_map.get(el["element_id"], "disputed")
        if status == "insufficient_evidence":
            continue
        lines.append(f"- [{el['element_id']}] {el['description']} (الحالة: {status})")
    return "\n".join(lines) if lines else "لا عناصر قابلة للتحليل"


def apply_law_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from schemas import LawApplicationResult

    required_elements: List[Dict] = state.get("required_elements") or []
    classifications: List[Dict] = state.get("element_classifications") or []
    retrieved_facts: str = state.get("retrieved_facts") or ""
    law_result: Dict = state.get("law_retrieval_result") or {}
    law_answer: str = law_result.get("answer") or ""

    # Identify elements to skip
    skipped = [c["element_id"] for c in classifications if c.get("status") == "insufficient_evidence"]
    active_ids = {el["element_id"] for el in required_elements if el["element_id"] not in skipped}

    if not active_ids:
        logger.info("apply_law_node: all elements skipped (insufficient_evidence)")
        return {
            "law_application": "جميع العناصر مستبعدة بسبب عدم كفاية الأدلة.",
            "applied_elements": [],
            "skipped_elements": skipped,
            "intermediate_steps": ["تطبيق القانون: جميع العناصر مستبعدة"],
        }

    elements_text = _format_elements_with_classification(required_elements, classifications)
    prompt = f"{_SYSTEM_PROMPT}\n\n{_USER_TEMPLATE.format(elements_text=elements_text, retrieved_facts=retrieved_facts or 'لا وقائع متاحة', law_answer=law_answer or 'لا نصوص قانونية متاحة')}"

    llm = get_llm("high")
    structured_llm = llm.with_structured_output(LawApplicationResult)

    try:
        result: LawApplicationResult = structured_llm.invoke(prompt)
        applied = [
            {
                "element_id": el.element_id,
                "reasoning": el.reasoning,
                "cited_articles": el.cited_articles,
            }
            for el in result.elements
        ]
        logger.info("Law application: %d elements analyzed, %d skipped", len(applied), len(skipped))
        return {
            "law_application": result.synthesis,
            "applied_elements": applied,
            "skipped_elements": skipped,
            "intermediate_steps": [f"تطبيق القانون: {len(applied)} عناصر محللة، {len(skipped)} مستبعدة"],
        }
    except Exception as exc:
        logger.error("apply_law_node failed: %s", exc)
        return {
            "law_application": f"فشل تطبيق القانون: {exc}",
            "applied_elements": [],
            "skipped_elements": skipped,
            "error_log": [f"apply_law_node: {exc}"],
            "intermediate_steps": ["تطبيق القانون: فشل"],
        }
