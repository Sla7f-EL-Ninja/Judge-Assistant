"""
llm_judge.py -- LLM-as-judge for legal answer quality evaluation.

Scores answers on: legal_accuracy (0-3), grounding (0-3),
completeness (0-2), arabic_quality (0-2). Total >= 6 means passed.

Usage:
    python -m tests.eval.llm_judge
"""

import json
import os
from typing import Any, Dict

SYSTEM_PROMPT = """أنت قاضٍ خبير في القانون المدني المصري. مهمتك تقييم جودة الإجابات القانونية.

قيّم الإجابة التالية بناءً على المعايير التالية:

1. الدقة القانونية (legal_accuracy): 0-3
   - 0: خطأ قانوني جسيم
   - 1: يحتوي على أخطاء ملحوظة
   - 2: صحيح في المجمل مع بعض النواقص
   - 3: دقيق وصحيح قانونياً

2. الاستناد (grounding): 0-3
   - 0: لا يستند لأي مصدر
   - 1: إشارات عامة بدون تحديد
   - 2: يستند لمصادر محددة ولكن غير كاملة
   - 3: مستند بالكامل لنصوص قانونية محددة

3. الشمولية (completeness): 0-2
   - 0: يفتقر لجوانب أساسية
   - 1: يغطي الجوانب الرئيسية
   - 2: شامل ومتكامل

4. جودة اللغة العربية (arabic_quality): 0-2
   - 0: لغة ركيكة أو غير مفهومة
   - 1: لغة مقبولة مع بعض الأخطاء
   - 2: لغة قانونية فصيحة وواضحة

أجب بصيغة JSON فقط:
{
    "legal_accuracy": <0-3>,
    "grounding": <0-3>,
    "completeness": <0-2>,
    "arabic_quality": <0-2>,
    "total": <مجموع النقاط>,
    "feedback": "<ملاحظات مختصرة>"
}"""


def judge_legal_answer(
    query: str,
    answer: str,
    context: str = "",
) -> Dict[str, Any]:
    """Evaluate a legal answer using an LLM judge.

    Parameters
    ----------
    query : str
        The original judge question.
    answer : str
        The system's answer to evaluate.
    context : str
        Optional context/sources used to generate the answer.

    Returns
    -------
    dict
        Evaluation scores with keys: legal_accuracy, grounding, completeness,
        arabic_quality, total, passed, feedback.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            temperature=0.0,
        )

        user_prompt = (
            f"السؤال: {query}\n\n"
            f"الإجابة: {answer}\n\n"
        )
        if context:
            user_prompt += f"السياق/المصادر: {context}\n\n"
        user_prompt += "قيّم هذه الإجابة:"

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", user_prompt),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        scores = json.loads(content)

        # Ensure all keys present
        result = {
            "legal_accuracy": scores.get("legal_accuracy", 0),
            "grounding": scores.get("grounding", 0),
            "completeness": scores.get("completeness", 0),
            "arabic_quality": scores.get("arabic_quality", 0),
            "total": scores.get("total", 0),
            "feedback": scores.get("feedback", ""),
        }
        result["passed"] = result["total"] >= 6
        return result

    except ImportError:
        return {
            "legal_accuracy": 0,
            "grounding": 0,
            "completeness": 0,
            "arabic_quality": 0,
            "total": 0,
            "passed": False,
            "feedback": "langchain-google-genai not installed",
        }
    except Exception as exc:
        return {
            "legal_accuracy": 0,
            "grounding": 0,
            "completeness": 0,
            "arabic_quality": 0,
            "total": 0,
            "passed": False,
            "feedback": f"Evaluation error: {exc}",
        }


if __name__ == "__main__":
    # Standalone test
    test_query = "ما هي شروط صحة العقد في القانون المدني المصري؟"
    test_answer = (
        "يشترط لصحة العقد في القانون المدني المصري توافر أركان ثلاثة: "
        "الرضا والمحل والسبب. كما يجب توافر الأهلية لدى المتعاقدين. "
        "ويجب أن يكون الرضا صادراً عن إرادة حرة خالية من عيوب الإرادة "
        "كالغلط والتدليس والإكراه والاستغلال (المواد 89-133 مدني)."
    )

    result = judge_legal_answer(test_query, test_answer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nPassed: {result['passed']}")
