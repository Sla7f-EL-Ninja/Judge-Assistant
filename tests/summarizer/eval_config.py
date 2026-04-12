"""
eval_config.py — Evaluation rubrics, thresholds, and prompts for the summarizer.

Used by test_eval_quality.py to score the pipeline output on 8 dimensions.
"""

from typing import Dict, Any

# ---------------------------------------------------------------------------
# Bias keyword sets
# ---------------------------------------------------------------------------

# From node_5.py BIAS_KEYWORDS
PIPELINE_BIAS_KEYWORDS = ["نوصي", "يجب على المحكمة", "نرى أن", "نقترح", "ينبغي الحكم"]

# Extended set for comprehensive bias detection
EXTENDED_BIAS_KEYWORDS = PIPELINE_BIAS_KEYWORDS + [
    "من الواضح أن",
    "لا شك أن",
    "المحكمة ترى",
    "الطرف المحق",
    "يتضح من الوقائع",
    "الحكم الصحيح",
]

# ---------------------------------------------------------------------------
# Evaluation dimensions and thresholds
# ---------------------------------------------------------------------------

EVAL_DIMENSIONS = {
    "EV-01": {
        "name": "Structural Completeness",
        "description": "All 7 sections non-empty",
        "max_score": 7,
        "pass_threshold": 7,
        "requires_llm": False,
    },
    "EV-02": {
        "name": "Bullet Coverage Preservation",
        "description": "Fraction of bullets surviving through aggregation",
        "max_score": 100,
        "pass_threshold": 95,  # percent
        "requires_llm": False,
    },
    "EV-03": {
        "name": "Source Traceability",
        "description": "All citations map to real fixture documents",
        "max_score": 100,
        "pass_threshold": 100,  # percent
        "requires_llm": False,
    },
    "EV-04": {
        "name": "Neutrality / Bias Detection",
        "description": "No bias keywords, balanced party representation",
        "max_score": 1,  # 1 = passes, 0 = fails
        "pass_threshold": 1,
        "requires_llm": False,
    },
    "EV-05": {
        "name": "Linguistic Quality",
        "description": "Arabic legal register quality (LLM judge)",
        "max_score": 10,
        "pass_threshold": 7,
        "requires_llm": True,
    },
    "EV-06": {
        "name": "Factual Faithfulness",
        "description": "Brief facts traceable to fixtures, no hallucination",
        "max_score": 15,
        "pass_threshold": 11,
        "requires_llm": True,
    },
    "EV-07": {
        "name": "Multi-Party Balance",
        "description": "All parties represented, balanced coverage",
        "max_score": 100,
        "pass_threshold": 100,  # coverage percent
        "requires_llm": False,
    },
    "EV-08": {
        "name": "Pipeline Timing",
        "description": "Total pipeline time < 180s for 7 documents",
        "max_score": 180,  # seconds (lower is better)
        "pass_threshold": 180,
        "requires_llm": False,
    },
}

# ---------------------------------------------------------------------------
# LLM judge prompts (adapted from tests/eval/llm_judge.py)
# ---------------------------------------------------------------------------

LINGUISTIC_QUALITY_PROMPT = """أنت خبير في اللغة العربية القانونية. قيّم جودة المذكرة القضائية التالية:

معايير التقييم:
1. الدقة في استخدام المصطلحات القانونية (legal_terminology): 0-3
   - 3: مصطلحات قانونية مصرية دقيقة ومتسقة
   - 2: مصطلحات صحيحة مع بعض التبسيط
   - 1: بعض الأخطاء في المصطلحات
   - 0: أخطاء جوهرية في المصطلحات

2. اللغة العربية القانونية الرسمية (formal_register): 0-3
   - 3: عربية قانونية فصيحة ومتسقة
   - 2: رسمية بشكل عام مع استثناءات بسيطة
   - 1: مقبولة مع ضعف في الصياغة
   - 0: لغة غير رسمية أو ركيكة

3. التماسك المنطقي (coherence): 0-2
   - 2: تدفق منطقي واضح داخل كل قسم وبين الأقسام
   - 1: تماسك جزئي مع بعض التقطع
   - 0: محتوى متشتت أو متناقض

4. الإيجاز (conciseness): 0-2
   - 2: موجز بدون تكرار
   - 1: بعض التكرار غير الضروري
   - 0: تكرار مفرط أو حشو

أجب بـ JSON فقط:
{
    "legal_terminology": <0-3>,
    "formal_register": <0-3>,
    "coherence": <0-2>,
    "conciseness": <0-2>,
    "total": <مجموع>,
    "feedback": "<ملاحظات>"
}"""

FAITHFULNESS_PROMPT = """أنت قاضٍ خبير في القانون المدني المصري. قيّم أمانة المذكرة القضائية للوثائق الأصلية المرفقة في رسالة المستخدم.

معايير التقييم:
1. استرجاع الحقائق (fact_recall): 0-5
   - هل الحقائق الجوهرية من الوثائق موجودة في المذكرة؟
   - 5: جميع الحقائق الجوهرية مذكورة
   - 0: حقائق جوهرية غائبة

2. دقة الحقائق (fact_precision): 0-5
   - هل المذكرة تحتوي على حقائق غير موجودة في الوثائق؟
   - 5: لا يوجد أي اختلاق
   - 0: اختلاق حقائق جوهرية

3. نسب المواقف (party_attribution): 0-5
   - هل مواقف كل طرف منسوبة بشكل صحيح؟
   - 5: نسب صحيح تام
   - 0: نسب خاطئ لأطراف متعددة

أجب بـ JSON فقط:
{
    "fact_recall": <0-5>,
    "fact_precision": <0-5>,
    "party_attribution": <0-5>,
    "total": <مجموع>,
    "feedback": "<ملاحظات>"
}"""

# ---------------------------------------------------------------------------
# Expected parties in the fixture case
# ---------------------------------------------------------------------------

FIXTURE_PARTIES = [
    "المدعي",
    "المدعى عليه",  # defendant 1 (Mahmoud)
    "خبير",
]

FIXTURE_DOC_IDS = [
    "صحيفة_دعوى",
    "مذكرة_بدفاع_المدعى_عليه_الأول",
    "مذكرة_بدفاع_المدعى_عليها_الثانية",
    "تقرير_الخبير",
    "تقرير_الطب_الشرعي",
    "محضر_جلسة_25_03_2024",
    "حكم_المحكمة",
]
