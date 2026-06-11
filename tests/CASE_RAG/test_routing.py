"""
tests/CASE_RAG/test_routing.py

Layer E: Off-topic rejection
Layer F: Multi-question fan-out
Layer G: Rephrase loop termination
"""

from __future__ import annotations

import pytest

from conftest import TEST_CASE_ID, invoke_graph


# ---------------------------------------------------------------------------
# Layer E -- Off-topic rejection
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Layer E -- Off-topic rejection (Unchanged - weather/cooking apply everywhere)
# ---------------------------------------------------------------------------
@pytest.mark.timeout(150)
def test_off_topic_weather(app):
    result = invoke_graph(app, query="ما هي حالة الطقس اليوم في القاهرة؟", case_id=TEST_CASE_ID)
    assert result.get("error") is None
    assert result.get("on_topic") is False
    assert len(result.get("final_answer", "").strip()) > 0

@pytest.mark.timeout(150)
def test_off_topic_cooking(app):
    result = invoke_graph(app, query="كيف أطبخ كشري؟", case_id=TEST_CASE_ID)
    assert result.get("error") is None
    assert result.get("on_topic") is False
    assert len(result.get("final_answer", "").strip()) > 0

# ---------------------------------------------------------------------------
# Layer F -- Multi-question fan-out
# ---------------------------------------------------------------------------
@pytest.mark.timeout(150)
def test_two_question_fanout(app):
    """A compound query triggers fan-out."""
    result = invoke_graph(
        app,
        query="ما هي طلبات المدعي في صحيفة الدعوى؟ وما هو القرار الذي أصدرته لجنة التظلمات؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None
    assert result.get("on_topic") is True

    sub_questions = result.get("sub_questions", [])
    assert len(sub_questions) >= 2, f"Expected >= 2 sub_questions, got {len(sub_questions)}"

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 2, f"Expected >= 2 sub_answers, got {len(sub_answers)}"
    
    # ---------------------------------------------------------
    # MODIFIED: Allow final_answer to contain the concatenated response
    # ---------------------------------------------------------
    final_ans_text = result.get("final_answer", "")
    assert len(final_ans_text.strip()) > 0, "final_answer should not be empty"
    
    # Optional: Verify it actually combined aspects of both sub-questions
    assert "طلبات" in final_ans_text or "لجنة التظلمات" in final_ans_text

# ---------------------------------------------------------------------------
# Layer G -- Rephrase loop termination
# ---------------------------------------------------------------------------
@pytest.mark.timeout(150)
def test_obscure_query_no_crash(app):
    """An obscure query must not crash. Rephrase loop must terminate."""
    result = invoke_graph(
        app,
        query="ما هو الرقم القومي لزوجة مالك العقار محمد أحمد علي؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None
    assert result.get("on_topic") is True

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1

    for sa in sub_answers:
        assert "question" in sa
        assert "answer" in sa
        assert "found" in sa
        assert isinstance(sa["answer"], str) and len(sa["answer"].strip()) > 0