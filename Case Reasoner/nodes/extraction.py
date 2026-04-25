"""Issue Extraction Node — parses discrete legal issues from the CaseBrief."""
import logging
from typing import Any, Dict

from config import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """أنت مساعد قضائي متخصص في القانون المدني المصري.
مهمتك استخراج المسائل القانونية المستقلة من ملخص الدعوى.
لا تطبق القانون ولا تصدر أحكامًا في هذه المرحلة.
استخرج فقط المسائل التي تحتاج فصلاً قانونيًا مستقلاً.
لكل مسألة، اقتبس النص الأصلي بالضبط من المصدر في حقل source_text.
حدد المسائل غير المتداخلة والمستقلة عن بعضها."""

_USER_TEMPLATE = """نقاط الخلاف الجوهرية:
{key_disputes}

الأسئلة القانونية المطروحة:
{legal_questions}

استخرج المسائل القانونية المستقلة التي يجب الفصل فيها."""


def extract_issues_node(state: Dict[str, Any]) -> Dict[str, Any]:
    from schemas import ExtractedIssues

    brief = state.get("case_brief") or {}
    legal_questions = brief.get("legal_questions", "")
    key_disputes = brief.get("key_disputes", "")

    llm = get_llm("high")
    structured_llm = llm.with_structured_output(ExtractedIssues)
    prompt = f"{_SYSTEM_PROMPT}\n\n{_USER_TEMPLATE.format(key_disputes=key_disputes, legal_questions=legal_questions)}"

    for attempt in range(2):
        try:
            result: ExtractedIssues = structured_llm.invoke(prompt)
            issues = [
                {
                    "issue_id": iss.issue_id,
                    "issue_title": iss.issue_title,
                    "legal_domain": iss.legal_domain,
                    "source_text": iss.source_text,
                }
                for iss in result.issues
            ]
            logger.info("Issue extraction: found %d issues", len(issues))
            return {
                "identified_issues": issues,
                "intermediate_steps": [f"استخراج المسائل: {len(issues)} مسألة"],
            }
        except Exception as exc:
            logger.warning("extract_issues attempt %d failed: %s", attempt + 1, exc)

    logger.error("extract_issues failed after 2 attempts — empty issue list")
    return {
        "identified_issues": [],
        "error_log": ["extract_issues_node: فشل استخراج المسائل بعد محاولتين"],
        "intermediate_steps": ["فشل استخراج المسائل"],
    }
