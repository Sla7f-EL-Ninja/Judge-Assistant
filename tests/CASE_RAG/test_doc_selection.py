"""
tests/CASE_RAG/test_doc_selection.py

Layer C: retrieve_specific_doc -- DocumentFinalizer path
Layer D: restrict_to_doc -- retrieve branch with doc-scoped filter

Known issue: DocumentFinalizer uses find_one({"title": doc_id}).
Two defense memos share the same title "مذكرة بدفاع"; tests must not
assert WHICH memo is returned, only that A memo is returned.
"""

from __future__ import annotations

import pytest

from conftest import TEST_CASE_ID, invoke_graph


# ---------------------------------------------------------------------------
# Layer C -- retrieve_specific_doc
# ---------------------------------------------------------------------------

@pytest.mark.timeout(60)
def test_retrieve_specific_sahifa(app):
    """'هاتلي صحيفة الدعوى' → DocumentFinalizer returns the full bill of complaint."""
    result = invoke_graph(
        app,
        query="هاتلي صحيفة الدعوى",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("doc_selection_mode") == "retrieve_specific_doc", (
        f"Expected 'retrieve_specific_doc', got '{result.get('doc_selection_mode')}'"
    )

    final_answer = result.get("final_answer", "")
    assert len(final_answer.strip()) > 200, (
        f"final_answer too short ({len(final_answer)} chars); "
        "DocumentFinalizer should return the full document text"
    )

    # At least one sub_answer entry with found=True
    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1, "No sub_answers from DocumentFinalizer"
    assert sub_answers[0].get("found") is True, (
        "sub_answers[0].found is not True"
    )


@pytest.mark.timeout(60)
def test_retrieve_specific_hukm(app):
    """'اعرض لي حكم المحكمة' → DocumentFinalizer returns the court judgment."""
    result = invoke_graph(
        app,
        query="اعرض لي حكم المحكمة",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("doc_selection_mode") == "retrieve_specific_doc", (
        f"Expected 'retrieve_specific_doc', got '{result.get('doc_selection_mode')}'"
    )

    final_answer = result.get("final_answer", "")
    assert len(final_answer.strip()) > 200, (
        f"final_answer too short ({len(final_answer)} chars)"
    )

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1
    assert sub_answers[0].get("found") is True


@pytest.mark.timeout(60)
def test_retrieve_specific_taqrir(app):
    """'عايز أشوف تقرير الخبير' → DocumentFinalizer returns the expert report."""
    result = invoke_graph(
        app,
        query="عايز أشوف تقرير الخبير",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("doc_selection_mode") == "retrieve_specific_doc", (
        f"Expected 'retrieve_specific_doc', got '{result.get('doc_selection_mode')}'"
    )

    final_answer = result.get("final_answer", "")
    assert len(final_answer.strip()) > 200, (
        f"final_answer too short ({len(final_answer)} chars)"
    )

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1
    assert sub_answers[0].get("found") is True


# ---------------------------------------------------------------------------
# Layer D -- restrict_to_doc
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_restrict_to_mahdar(app):
    """'ما هي القرارات في محضر الجلسة؟' → restrict_to_doc, answer non-empty."""
    result = invoke_graph(
        app,
        query="ما هي القرارات التي اتخذت في محضر الجلسة؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("doc_selection_mode") == "restrict_to_doc", (
        f"Expected 'restrict_to_doc', got '{result.get('doc_selection_mode')}'"
    )

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1, "No sub_answers returned"

    # At least one branch found relevant content
    assert any(sa.get("found") for sa in sub_answers), (
        "No sub_answer with found=True for restrict_to_doc query"
    )

    # Some non-empty answer text exists
    all_answer_text = " ".join(sa.get("answer", "") for sa in sub_answers)
    assert len(all_answer_text.strip()) > 0, "All sub_answers have empty answer text"


@pytest.mark.timeout(90)
def test_restrict_to_hukm(app):
    """'ما هي حيثيات الحكم وأسبابه؟' → restrict_to_doc, answer non-empty."""
    result = invoke_graph(
        app,
        query="ما هي حيثيات الحكم وأسبابه؟",
        case_id=TEST_CASE_ID,
    )
    assert result.get("error") is None, f"error: {result.get('error')}"
    assert result.get("doc_selection_mode") == "restrict_to_doc", (
        f"Expected 'restrict_to_doc', got '{result.get('doc_selection_mode')}'"
    )

    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1, "No sub_answers returned"
    assert any(sa.get("found") for sa in sub_answers), (
        "No sub_answer with found=True"
    )

    all_answer_text = " ".join(sa.get("answer", "") for sa in sub_answers)
    assert len(all_answer_text.strip()) > 0, "All sub_answers have empty answer text"
