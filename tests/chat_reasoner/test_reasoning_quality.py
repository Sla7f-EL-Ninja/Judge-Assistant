"""
test_reasoning_quality.py — six end-to-end reasoning-quality tests against
the real seeded case 2847/2024 (forged apartment sale contract).

Every test logs session_id to stderr for MongoDB traceability.
All tests require seeded_case (7 Arabic fixture docs under case_id="1234").

Assertable facts from the case:
  - Parties: أحمد محمد عبد الله (plaintiff), محمود سعيد إبراهيم (D1),
             شركة العقارات الحديثة (D2)
  - Contract: 15/04/2022, apartment 140 m², 3rd floor, العقار رقم 14
  - Forensic report 471/2023: similarity 72-78%, threshold 85%,
    signature applied early 2023 not 2022
  - Engineering report (10/04/2024, المهندس سامي رمزي): 200,000 ج.م
  - Relief: فسخ + 2,000,000 ج.م
  - D1 defenses: عدم قبول / إيصالات / ينكر التزوير / طعن سلسلة أدلة
  - D2 defenses: انعدام الصفة / حسن نية / انقطاع علاقة سببية
"""

import json
import re
import sys

_ARABIC_RE = re.compile(r"[؀-ۿ]")
CASE_ID = "1234"


# ---------------------------------------------------------------------------
# Local helpers (mirror test_adapter_e2e.py conventions)
# ---------------------------------------------------------------------------

def _context(**overrides):
    ctx = {"case_id": CASE_ID, "conversation_history": [], "escalation_reason": "اختبار جودة الاستدلال"}
    ctx.update(overrides)
    return ctx


def _log_session(result):
    session_id = (result.raw_output or {}).get("session_id", "unknown")
    print(f"\n[reasoning_quality] session_id={session_id}", file=sys.stderr)
    return session_id


def _collect_step_results(raw):
    out = []
    for r in (raw or {}).get("step_results", []):
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, str):
            try:
                out.append(json.loads(r))
            except Exception:
                pass
    return out


def _plan_tools(raw):
    return {s.get("tool") for s in (raw or {}).get("plan", [])}


def _contains_any(text, candidates):
    return any(c in text for c in candidates)


# ---------------------------------------------------------------------------
# Scenario 1: Document comparison — forensic vs. engineering
# ---------------------------------------------------------------------------

def test_document_comparison_forensic_vs_engineering(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "قارن بين تقرير الطب الشرعي رقم 471 لسنة 2023 وتقرير الخبير الهندسي "
        "من حيث الموضوع والنتائج.",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Adapter error: {result.error}"
    assert _ARABIC_RE.search(result.response), "Response must contain Arabic text"
    assert len(result.response) >= 300, (
        f"Response too short ({len(result.response)} chars) — likely a non-answer"
    )

    # Response should reference at least 2 concrete figures from the case
    signals = ["471", "72", "78", "85", "200,000", "مئتا", "200000", "140"]
    hits = sum(1 for s in signals if s in result.response)
    assert hits >= 2, (
        f"Response should reference ≥2 concrete case figures (471, 72/78, 85, 200000, 140). "
        f"Found {hits}. Response: {result.response[:300]}"
    )

    raw = result.raw_output or {}
    assert raw.get("synth_sufficient") is True, "Synthesizer did not produce a sufficient answer (best-effort path)"
    assert "case_doc_rag" in _plan_tools(raw), "Plan must include case_doc_rag for document retrieval"


# ---------------------------------------------------------------------------
# Scenario 2: Case-law cross-reference
# ---------------------------------------------------------------------------

def test_case_law_cross_reference(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "ما هي المواد في القانون المدني المصري التي تنطبق على دعوى فسخ عقد البيع "
        "بسبب التزوير، وكيف تنطبق على وقائع هذه القضية؟",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Adapter error: {result.error}"
    assert _ARABIC_RE.search(result.response)
    assert len(result.response) >= 400, (
        f"Response too short ({len(result.response)} chars)"
    )

    raw = result.raw_output or {}
    tools_used = _plan_tools(raw)
    assert "civil_law_rag" in tools_used, "Must use civil_law_rag for article lookup"
    assert "case_doc_rag" in tools_used, "Must use case_doc_rag to ground law in case facts"

    assert _contains_any(result.response, ["فسخ", "تزوير", "بطلان"]), (
        "Response must mention فسخ, تزوير, or بطلان"
    )
    assert _contains_any(result.response, ["مادة", "المادة", "القانون المدني"]), (
        "Response must reference a civil law article or القانون المدني"
    )
    assert raw.get("synth_sufficient") is True, "Synthesizer did not produce a sufficient answer (best-effort path)"


# ---------------------------------------------------------------------------
# Scenario 3: Multi-party analysis
# ---------------------------------------------------------------------------

def test_multi_party_analysis(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "حلّل موقف كل من المدعى عليه الأول والمدعى عليها الثانية في هذه القضية "
        "واذكر دفوع كل منهما.",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Adapter error: {result.error}"
    assert _ARABIC_RE.search(result.response)
    assert len(result.response) >= 400, (
        f"Response too short ({len(result.response)} chars)"
    )

    # D1 signals: denial + receipts + premature filing defense
    d1_signals = ["محمود", "المدعى عليه الأول", "إيصالات", "التسوية الودية", "قبل الأوان"]
    assert _contains_any(result.response, d1_signals), (
        f"Response must reference D1 (محمود / إيصالات / قبل الأوان). "
        f"Snippet: {result.response[:400]}"
    )

    # D2 signals: legal standing defense + good faith
    d2_signals = ["شركة العقارات", "المدعى عليها الثانية", "انعدام الصفة", "حسن النية", "انقطاع"]
    assert _contains_any(result.response, d2_signals), (
        f"Response must reference D2 (شركة العقارات / انعدام الصفة / حسن النية). "
        f"Snippet: {result.response[:400]}"
    )

    raw = result.raw_output or {}
    assert raw.get("synth_sufficient") is True, "Synthesizer did not produce a sufficient answer (best-effort path)"
    assert "case_doc_rag" in _plan_tools(raw), "Must use case_doc_rag to retrieve defense memos"


# ---------------------------------------------------------------------------
# Scenario 4: Hard legal reasoning — forensic impact on claim
# ---------------------------------------------------------------------------

def test_hard_legal_reasoning(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "بالاستناد إلى تقرير الطب الشرعي الذي أثبت أن التوقيع ليس أصلياً، "
        "ما الأثر القانوني على عقد البيع وعلى دعوى التعويض بمبلغ 2,000,000 جنيه؟",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Adapter error: {result.error}"
    assert _ARABIC_RE.search(result.response)
    assert len(result.response) >= 400, (
        f"Response too short ({len(result.response)} chars)"
    )

    # Must reference the specific claimed amount
    assert _contains_any(result.response, ["2,000,000", "2000000", "مليوني", "مليونين"]), (
        "Response must reference the 2,000,000 ج.م claim amount"
    )

    # Must address legal consequences
    assert _contains_any(result.response, ["بطلان", "فسخ", "تعويض"]), (
        "Response must address legal effect (بطلان / فسخ / تعويض)"
    )

    raw = result.raw_output or {}
    assert raw.get("synth_sufficient") is True, "Synthesizer did not produce a sufficient answer (best-effort path)"
    assert "civil_law_rag" in _plan_tools(raw), "Must consult civil law for legal consequences"


# ---------------------------------------------------------------------------
# Scenario 5: Summary integration
# ---------------------------------------------------------------------------

def test_summary_integration(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "استخدم الملخص المخزن لهذه القضية مع القانون المدني لتقييم احتمالية نجاح دعوى المدعي.",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Adapter error: {result.error}"
    assert _ARABIC_RE.search(result.response)
    assert len(result.response) >= 300, (
        f"Response too short ({len(result.response)} chars)"
    )

    step_results = _collect_step_results(result.raw_output)
    tools_used = _plan_tools(result.raw_output)

    assert "fetch_summary_report" in tools_used, "Must include fetch_summary_report step"
    assert "civil_law_rag" in tools_used, "Must include civil_law_rag step"

    # fetch_summary_report must succeed (pre-seeded summary for case 1234 exists)
    summary_steps = [r for r in step_results if r.get("tool") == "fetch_summary_report"]
    if summary_steps:
        assert summary_steps[0].get("status") == "success", (
            f"fetch_summary_report returned: {summary_steps[0]}"
        )
    assert (result.raw_output or {}).get("synth_sufficient") is True, "Synthesizer did not produce a sufficient answer (best-effort path)"


# ---------------------------------------------------------------------------
# Scenario 6: Contradiction detection — D1 defense vs. forensic finding
# ---------------------------------------------------------------------------

def test_contradiction_detection(seeded_case):
    from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter

    adapter = ChatReasonerAdapter()
    result = adapter.invoke(
        "هل توجد تناقضات بين دفوع المدعى عليه الأول ونتائج تقرير الطب الشرعي؟ وضّح كيف.",
        _context(),
    )
    _log_session(result)

    assert result.error is None, f"Adapter error: {result.error}"
    assert _ARABIC_RE.search(result.response)
    assert len(result.response) >= 400, (
        f"Response too short ({len(result.response)} chars)"
    )

    # Must reference forensic evidence
    forensic_signals = ["471", "التوقيع", "الطب الشرعي", "72", "78", "85"]
    assert _contains_any(result.response, forensic_signals), (
        f"Response must reference forensic findings (471 / نسبة التشابه / الطب الشرعي). "
        f"Snippet: {result.response[:400]}"
    )

    # Must reference D1 defense
    d1_defense_signals = ["ينكر", "إيصالات", "إجرائي", "سلسلة", "محمود", "التزوير"]
    assert _contains_any(result.response, d1_defense_signals), (
        f"Response must reference D1 defense (ينكر التزوير / إيصالات / إجرائي). "
        f"Snippet: {result.response[:400]}"
    )

    raw = result.raw_output or {}
    assert raw.get("synth_sufficient") is True, "Synthesizer did not produce a sufficient answer (best-effort path)"
    assert "case_doc_rag" in _plan_tools(raw), "Must use case_doc_rag to retrieve defense memos"
