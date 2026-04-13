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

@pytest.mark.timeout(60)
def test_off_topic_weather(app):
    """A weather question must be rejected as off-topic."""
    result = invoke_graph(
        app,
        query="ما هي حالة الطقس اليوم في القاهرة؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("on_topic") is False, (
        "Weather question should be classified as off-topic"
    )
    final_answer = result.get("final_answer", "")
    # offTopicResponse node must write a non-empty Arabic refusal
    assert len(final_answer.strip()) > 0, (
        "offTopicResponse should produce a non-empty final_answer"
    )
    # Sanity check: it should NOT look like a legal answer
    assert "الطقس" not in final_answer or len(final_answer) < 200, (
        "final_answer appears to answer the off-topic question rather than refuse it"
    )


@pytest.mark.timeout(60)
def test_off_topic_cooking(app):
    """A cooking question must be rejected as off-topic."""
    result = invoke_graph(
        app,
        query="كيف أطبخ كشري؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("on_topic") is False, (
        "Cooking question should be classified as off-topic"
    )
    final_answer = result.get("final_answer", "")
    assert len(final_answer.strip()) > 0, (
        "offTopicResponse should produce a non-empty final_answer"
    )


# ---------------------------------------------------------------------------
# Layer F -- Multi-question fan-out
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_two_question_fanout(app):
    """A compound query triggers fan-out: >= 2 sub_questions and >= 2 sub_answers.

    Known behavior (documented): mergeAnswers sets final_answer = '' when
    len(sub_answers) > 1. Tests must read sub_answers directly.
    """
    result = invoke_graph(
        app,
        query="ما هي وقائع الدعوى؟ وما هو منطوق الحكم؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("on_topic") is True, (
        "Compound civil-case query should be on-topic"
    )

    sub_questions = result.get("sub_questions", [])
    assert len(sub_questions) >= 2, (
        f"Expected >= 2 sub_questions from questionRewriter, got {len(sub_questions)}: "
        f"{sub_questions}"
    )

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 2, (
        f"Expected >= 2 sub_answers from fan-out, got {len(sub_answers)}"
    )

    # Documented behavior: final_answer is empty for multi-question (by design)
    # We assert this to catch regressions if mergeAnswers behavior changes.
    assert result.get("final_answer", "") == "", (
        "final_answer should be '' for multi-question queries "
        "(Supervisor reads sub_answers directly)"
    )


# ---------------------------------------------------------------------------
# Layer G -- Rephrase loop termination
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_obscure_query_no_crash(app):
    """An obscure query must not crash. Rephrase loop must terminate.

    The query asks for the company's commercial registry number -- very
    specific detail that may or may not be in the ingested fixtures.
    Acceptable outcomes:
      (a) found=True: rephrase succeeded and a relevant chunk was surfaced
      (b) found=False: rephrase loop exhausted (_MAX_REPHRASE=2) and
          cannotAnswer produced the standard Arabic message
    Either outcome is valid; the test only asserts structural correctness
    and that the graph terminates without exception.
    """
    result = invoke_graph(
        app,
        query="ما هو رقم القيد في السجل التجاري للشركة المدعى عليها الثانية؟",
        case_id=TEST_CASE_ID,
    )

    # Must not crash or set error
    assert result.get("error") is None, f"Graph raised an error: {result.get('error')}"
    assert result.get("on_topic") is True, (
        "Registry number query should be in-scope (references the case)"
    )

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1, "No sub_answers -- graph did not complete a branch"

    # Each sub_answer must be structurally sound
    for sa in sub_answers:
        assert "question" in sa, f"sub_answer missing 'question' key: {sa}"
        assert "answer" in sa, f"sub_answer missing 'answer' key: {sa}"
        assert "found" in sa, f"sub_answer missing 'found' key: {sa}"
        assert isinstance(sa["answer"], str) and len(sa["answer"].strip()) > 0, (
            "sub_answer['answer'] is empty"
        )

    # Log actual outcome for diagnostic purposes
    found_any = any(sa.get("found") for sa in sub_answers)
    import logging
    logging.getLogger(__name__).info(
        "test_obscure_query_no_crash: found=%s, rephraseCount in sub_answers=%s",
        found_any,
        [sa.get("found") for sa in sub_answers],
    )
