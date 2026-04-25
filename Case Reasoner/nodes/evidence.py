"""Evidence Sufficiency Node — classifies each element against retrieved evidence."""
import logging
from typing import Any, Dict, List

from config import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """أنت مقيّم أدلة قضائية محايد.
صنّف كل عنصر من العناصر المطلوبة بناءً على الأدلة المتاحة فقط.
لا تستنتج أدلة غير موجودة صراحةً في النص.
استخدم هذه التصنيفات فقط (بالإنجليزية):
- established: الدليل واضح وغير متنازع عليه
- not_established: الدليل يدحض هذا العنصر أو ينفيه
- disputed: أطراف مختلفة تقدم روايات متضاربة
- insufficient_evidence: لا توجد أدلة كافية للتقييم
قاعدة: عند الشك، استخدم "disputed" وليس "established"."""

_USER_TEMPLATE = """العناصر المطلوبة:
{elements_text}

الوقائع المستردة من مستندات القضية:
{retrieved_facts}

الإطار القانوني المسترد:
{law_answer}

صنّف كل عنصر بناءً على الأدلة أعلاه."""


def _format_elements(elements: List[Dict]) -> str:
    return "\n".join(f"- [{el['element_id']}] {el['description']} ({el['element_type']})" for el in elements)


def classify_evidence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from schemas import EvidenceSufficiencyResult

    required_elements: List[Dict] = state.get("required_elements") or []
    retrieved_facts: str = state.get("retrieved_facts") or ""
    law_answer: str = (state.get("law_retrieval_result") or {}).get("answer") or ""

    if not required_elements:
        return {
            "element_classifications": [],
            "intermediate_steps": ["تصنيف الأدلة: لا عناصر للتصنيف"],
        }

    elements_text = _format_elements(required_elements)
    prompt = f"{_SYSTEM_PROMPT}\n\n{_USER_TEMPLATE.format(elements_text=elements_text, retrieved_facts=retrieved_facts or 'لا وقائع متاحة', law_answer=law_answer or 'لا إطار قانوني متاح')}"

    llm = get_llm("medium")
    structured_llm = llm.with_structured_output(EvidenceSufficiencyResult)

    try:
        result: EvidenceSufficiencyResult = structured_llm.invoke(prompt)
        classifications = [
            {
                "element_id": c.element_id,
                "status": c.status,
                "evidence_summary": c.evidence_summary,
                "notes": c.notes,
            }
            for c in result.classifications
        ]
        logger.info("Evidence classification: %d elements classified", len(classifications))
        return {
            "element_classifications": classifications,
            "intermediate_steps": [f"تصنيف الأدلة: {len(classifications)} عنصر"],
        }
    except Exception as exc:
        logger.warning("classify_evidence_node failed — defaulting all to 'disputed': %s", exc)
        # Safest default: disputed (analyzed but not asserted)
        fallback = [
            {"element_id": el["element_id"], "status": "disputed",
             "evidence_summary": "تعذّر التصنيف التلقائي", "notes": str(exc)}
            for el in required_elements
        ]
        return {
            "element_classifications": fallback,
            "error_log": [f"classify_evidence_node: {exc} — تم تصنيف الكل كـ 'disputed'"],
            "intermediate_steps": ["تصنيف الأدلة: فشل، تصنيف احتياطي"],
        }
