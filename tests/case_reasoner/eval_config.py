"""
eval_config.py — Metric definitions, thresholds, and Arabic LLM judge prompts
for the Case Reasoner evaluation harness.
"""

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

PIPELINE_TIMING_THRESHOLD_SECONDS = 300

# ---------------------------------------------------------------------------
# Rule-Based and LLM-Judge Metric Definitions
# ---------------------------------------------------------------------------

EVAL_DIMENSIONS = {
    "CR-EV-01": {
        "name": "Branch Coverage Rate",
        "description": "Fraction of extracted issues with fully populated results (validation_passed=True)",
        "max_score": 100,
        "pass_threshold": 100,
        "requires_llm": False,
    },
    "CR-EV-02": {
        "name": "Citation Presence Rate",
        "description": "Fraction of applied elements with >=1 cited_article AND tied to retrieved facts",
        "max_score": 100,
        "pass_threshold": 80,
        "requires_llm": False,
    },
    "CR-EV-03": {
        "name": "Unsupported Element Rate",
        "description": "Fraction of elements NOT in unsupported_conclusions (lower unsupported = higher score)",
        "max_score": 100,
        "pass_threshold": 90,
        "requires_llm": False,
    },
    "CR-EV-04": {
        "name": "Confidence Signal Accuracy",
        "description": "Computed case_level_confidence.level matches expected level from golden set",
        "max_score": 1,
        "pass_threshold": 1,
        "requires_llm": False,
    },
    "CR-EV-05": {
        "name": "Empty Report Rate",
        "description": "Empty report generated if and only if issue_count == 0",
        "max_score": 1,
        "pass_threshold": 1,
        "requires_llm": False,
    },
    "CR-EV-06": {
        "name": "Reconciliation Trigger Accuracy",
        "description": "Reconciliation paragraphs present IFF has_cross_issue_conflicts=True in golden set",
        "max_score": 1,
        "pass_threshold": 1,
        "requires_llm": False,
    },
    "CR-EV-07": {
        "name": "Section Completeness",
        "description": "All expected report sections present and non-empty (7 or 8 depending on reconciliation)",
        "max_score": 8,
        "pass_threshold": 7,
        "requires_llm": False,
    },
    "CR-EV-08": {
        "name": "Arabic Legal Register Quality",
        "description": "Formal legal Arabic throughout: terminology accuracy, register, coherence, conciseness",
        "max_score": 10,
        "pass_threshold": 7,
        "requires_llm": True,
    },
    "CR-EV-09": {
        "name": "Neutrality / Bias Detection",
        "description": "Section VI must contain zero directional language or implied judgments",
        "max_score": 1,
        "pass_threshold": 1,
        "requires_llm": True,
    },
    "CR-EV-10": {
        "name": "Factual Faithfulness",
        "description": "All reasoning traceable to retrieved facts; no hallucinated citations",
        "max_score": 15,
        "pass_threshold": 11,
        "requires_llm": True,
    },
    "CR-EV-11": {
        "name": "Counterargument Balance",
        "description": "Both plaintiff and defendant arguments present and substantive per issue",
        "max_score": 10,
        "pass_threshold": 7,
        "requires_llm": True,
    },
    "CR-EV-12": {
        "name": "Reconciliation Quality",
        "description": "Reconciliation section explains conflict without directional resolution",
        "max_score": 10,
        "pass_threshold": 7,
        "requires_llm": True,
    },
}

# ---------------------------------------------------------------------------
# Arabic bias keyword lists (Section VI neutrality pre-check, rule-based)
# ---------------------------------------------------------------------------

BIAS_KEYWORDS = [
    "يثبت الحق",
    "تُرفض الدعوى",
    "يلزم المدعى عليه",
    "يستحق المدعي",
    "نوصي",
    "يجب على المحكمة",
    "نرى أن",
    "نقترح",
    "ينبغي الحكم",
    "من الواضح أن",
    "لا شك أن",
    "الطرف المحق",
    "الحكم الصحيح",
    "ثبت يقينًا",
    "لا مراء في",
    "يتعين إلزام",
    "يستوجب الرفض",
    "يستوجب القبول",
    "الثابت قانونًا",
    "يتضح جليًا أن",
]

DIRECTIONAL_VERBS = [
    "يستوجب الحكم",
    "يتعين القضاء",
    "يجب التعويض",
    "ثبت حق",
    "تأكد استحقاق",
    "يُقضى بـ",
    "تُقبل الدعوى",
    "يُحكم بـ",
    "يُلزَم بـ",
]

# ---------------------------------------------------------------------------
# Report section detection
# ---------------------------------------------------------------------------

SECTION_ORDINALS = [
    "الأول", "الثاني", "الثالث", "الرابع",
    "الخامس", "السادس", "السابع", "الثامن",
]

SECTION_HEADER_PATTERN = r"القسم\s+(" + "|".join(SECTION_ORDINALS) + r")"

SECTION_VI_HEADER = "القسم السادس"
SECTION_VIII_HEADER = "القسم الثامن"

CONFIDENCE_LEVEL_ARABIC = {
    "high": ["مرتفع", "عالٍ", "عالي"],
    "medium": ["متوسط"],
    "low": ["منخفض", "ضعيف"],
}

# ---------------------------------------------------------------------------
# LLM Judge Prompts (Arabic)
# ---------------------------------------------------------------------------

ARABIC_REGISTER_PROMPT = """أنت خبير في اللغة العربية القانونية المصرية. قيّم جودة التقرير القانوني التالي من حيث اللغة والأسلوب فقط.

معايير التقييم:
1. المصطلحات القانونية (legal_terminology): 0-3
   - 3: مصطلحات قانونية مصرية دقيقة ومتسقة طوال التقرير
   - 2: معظم المصطلحات صحيحة مع بعض الأخطاء البسيطة
   - 1: أخطاء متعددة في المصطلحات القانونية
   - 0: أخطاء جوهرية تجعل المضمون القانوني غير واضح

2. السجل اللغوي الرسمي (formal_register): 0-3
   - 3: عربية قانونية فصيحة رسمية طوال النص
   - 2: رسمي في معظمه مع بعض الانزلاقات
   - 1: خليط من الرسمي وغير الرسمي
   - 0: لغة عامية أو ركيكة

3. التماسك المنطقي (coherence): 0-2
   - 2: تدفق منطقي واضح بين الفقرات والأقسام
   - 1: بعض الانقطاع في التسلسل المنطقي
   - 0: محتوى متشتت وغير متماسك

4. الإيجاز (conciseness): 0-2
   - 2: موجز وواضح بدون حشو أو تكرار
   - 1: بعض الحشو المقبول
   - 0: حشو مفرط أو تكرار يُفقد التقرير قيمته

التقرير:
{report_text}

أجب بـ JSON فقط، بدون أي نص إضافي:
{{"legal_terminology": <0-3>, "formal_register": <0-3>, "coherence": <0-2>, "conciseness": <0-2>, "total": <0-10>, "feedback": "<ملاحظات مختصرة>"}}"""

NEUTRALITY_PROMPT = """أنت مراقب حياد قضائي متخصص. افحص القسم السادس (حالة الملف) من التقرير القانوني التالي.

القسم السادس يجب أن يقتصر على:
- عرض حالة كل عنصر: ثابت / متنازع عليه / غير كافي الأدلة
- تلخيص موقف الأطراف دون ترجيح
- وصف الأدلة المتاحة دون استنتاج

القسم السادس يُحظر فيه:
- أي حكم أو استنتاج قانوني
- لغة اتجاهية تفيد تفضيل طرف على آخر
- استخدام كلمات مثل: "يثبت الحق"، "تُرفض"، "يلزم"، "يستحق"، "يجب الحكم"
- أي صياغة تتجاوز عرض الحقائق

التقرير كاملاً:
{report_text}

افحص القسم السادس تحديدًا وأجب بـ JSON فقط:
{{"neutral": <true/false>, "violations": ["<وصف المخالفة الأولى>", "<وصف المخالفة الثانية>"], "severity": "<none/minor/major>", "section_vi_text": "<النص الذي فحصته>", "feedback": "<ملاحظات مختصرة>"}}"""

FAITHFULNESS_PROMPT = """أنت قاضٍ خبير في تقييم دقة التقارير القانونية. قيّم أمانة التقرير التالي بالنسبة للوقائع والنصوص القانونية المسترداة المذكورة فيه.

معايير التقييم:
1. دقة الاستشهاد بالمواد (citation_accuracy): 0-5
   - 5: جميع أرقام المواد مذكورة بشكل صحيح ومطبقة في سياقها الصحيح
   - 3: معظم المواد صحيحة مع خطأ أو خطأين
   - 1: أخطاء متعددة في أرقام المواد أو تطبيقها
   - 0: مواد مختلقة أو مطبقة بشكل مغلوط بشكل صارخ

2. ربط الاستنتاجات بالوقائع (fact_linkage): 0-5
   - 5: كل استنتاج مرتبط بواقعة مذكورة في الملف أو نص قانوني مسترد
   - 3: معظم الاستنتاجات مدعومة مع بعض الادعاءات غير المربوطة
   - 1: كثير من الاستنتاجات دون دعم واضح
   - 0: الاستنتاجات مفصولة عن الأدلة

3. غياب الاختلاق (no_hallucination): 0-5
   - 5: لا وقائع مختلقة ولا مواد غير موجودة
   - 3: اختلاق طفيف في التفاصيل لا يؤثر على الجوهر
   - 1: اختلاق ملحوظ يؤثر على موثوقية التقرير
   - 0: اختلاق صارخ يجعل التقرير مضللًا

التقرير:
{report_text}

الوقائع المسترداة المرجعية:
{retrieved_facts_summary}

أجب بـ JSON فقط:
{{"citation_accuracy": <0-5>, "fact_linkage": <0-5>, "no_hallucination": <0-5>, "total": <0-15>, "hallucinated_items": ["<عنصر مختلق إن وجد>"], "feedback": "<ملاحظات مختصرة>"}}"""

COUNTERARGUMENT_BALANCE_PROMPT = """أنت محكّم قانوني متخصص. قيّم توازن حجج الأطراف في التقرير القانوني التالي.

لكل مسألة قانونية مطروحة، تحقق من:
1. حضور حجج المدعي (plaintiff_present): 0-2
   - 2: حجج المدعي مذكورة بوضوح وتفصيل كافٍ
   - 1: مذكورة باختصار مفرط
   - 0: غائبة

2. حضور حجج المدعى عليه (defendant_present): 0-2
   - 2: حجج المدعى عليه مذكورة بوضوح وتفصيل كافٍ
   - 1: مذكورة باختصار مفرط
   - 0: غائبة

3. جوهرية الحجج (substantive): 0-3
   - 3: الحجج قانونية وموضوعية، تستند إلى مواد محددة أو وقائع ثابتة
   - 2: الحجج معقولة لكن أحيانًا مبهمة
   - 1: الحجج شكلية أو عامة بدون أساس قانوني واضح
   - 0: الحجج تفتقر إلى المضمون القانوني

4. التوازن بين الطرفين (balance): 0-3
   - 3: الطرفان ممثلان بعمق متوازٍ وعدالة
   - 2: تفاوت بسيط لصالح أحد الطرفين
   - 1: تفاوت ملحوظ يؤثر على موضوعية التقرير
   - 0: ميل واضح لصالح طرف على حساب الآخر

التقرير:
{report_text}

أجب بـ JSON فقط:
{{"plaintiff_present": <0-2>, "defendant_present": <0-2>, "substantive": <0-3>, "balance": <0-3>, "total": <0-10>, "feedback": "<ملاحظات مختصرة>"}}"""

RECONCILIATION_QUALITY_PROMPT = """أنت مراقب جودة قضائية متخصص. قيّم القسم الثامن (التوفيق بين المسائل المتعارضة) في التقرير القانوني التالي.

هذا القسم يُكتب فقط عند وجود تعارض بين مسألتين قانونيتين تستند إلى نفس المادة بنتائج متعارضة.

معايير التقييم:
1. وصف التعارض (conflict_description): 0-3
   - 3: التعارض موصوف بدقة مع تحديد المسألتين والمادة المشتركة
   - 2: موصوف لكن يفتقر إلى بعض التفاصيل
   - 1: وصف مبهم يصعب فهمه
   - 0: التعارض غير موصوف أو مفهوم

2. الحياد في التوفيق (neutrality): 0-4
   - 4: الفقرة تشرح التعارض دون أن تُرجّح طرفًا أو تحسم المسألة
   - 3: محايد في معظمه مع ميل طفيف
   - 2: بعض التوجيه الضمني
   - 1: توجيه واضح يتجاوز دور المساعد القضائي
   - 0: يحسم المسألة بشكل صريح

3. الوضوح والصياغة (clarity): 0-3
   - 3: صياغة واضحة ومفهومة تساعد القاضي في إدراك التعارض
   - 2: مفهوم لكن يحتاج إعادة قراءة
   - 1: غامض أو مطول بشكل مضر
   - 0: غير قابل للفهم

فقرات التوفيق:
{reconciliation_text}

أجب بـ JSON فقط:
{{"conflict_description": <0-3>, "neutrality": <0-4>, "clarity": <0-3>, "total": <0-10>, "feedback": "<ملاحظات مختصرة>"}}"""
