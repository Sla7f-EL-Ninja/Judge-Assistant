"""
tests/CASE_RAG/test_happy_path.py

Layer B: On-topic retrieval tests.

Features:
- case_doc_rag.* loggers piped to stdout at DEBUG level (visible with pytest -s)
- Mix of simple and moderately harder retrieval questions
- After each test: prints the answer, the source chunks retrieved from Qdrant,
  and the expected keywords so results can be evaluated manually
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest

from conftest import TEST_CASE_ID, invoke_graph

# ---------------------------------------------------------------------------
# Logger setup -- pipe case_doc_rag.* to stdout at DEBUG level
# ---------------------------------------------------------------------------

def _setup_rag_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(levelname)s] %(name)s :: %(message)s",
        )
    )
    for logger_name in (
        "case_doc_rag",
        "case_doc_rag.infrastructure",
        "case_doc_rag.generation_nodes",
        "case_doc_rag.retrieval_nodes",
        "case_doc_rag.selection_nodes",
        "RAG.case_doc_rag",
        "RAG.case_doc_rag.infrastructure",
        "RAG.case_doc_rag.generation_nodes",
        "RAG.case_doc_rag.retrieval_nodes",
        "RAG.case_doc_rag.nodes.generation_nodes",
        "RAG.case_doc_rag.nodes.retrieval_nodes",
        "RAG.case_doc_rag.nodes.selection_nodes",
    ):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.DEBUG)
        if not any(isinstance(h, logging.StreamHandler) for h in lg.handlers):
            lg.addHandler(handler)
        lg.propagate = False


_setup_rag_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_answer_text(result: dict) -> str:
    """Return the best available answer string from a graph result."""
    if result.get("final_answer", "").strip():
        return result["final_answer"]
    sub_answers = result.get("sub_answers", [])
    if sub_answers:
        return sub_answers[0].get("answer", "")
    return ""


def _fetch_source_chunks(file_ingestor, sub_answers: list[dict]) -> list[dict]:
    """Re-query Qdrant to retrieve the actual chunk text for each source reference.

    sub_answers entries carry a `sources` list like:
        ["تقرير_الخبير.txt:chunk_2", "حكم_المحكمة.txt:chunk_0"]

    Strategy: scroll Qdrant filtered by case_id + source_file + chunk_index
    to reconstruct the exact chunks that were surfaced during retrieval.

    Returns a list of dicts:
        {"source": "file.txt:chunk_N", "text": "<chunk text>"}
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant_client = file_ingestor.vectorstore.client
    collection_name = file_ingestor._qdrant_collection_name

    all_sources: list[str] = []
    for sa in sub_answers:
        all_sources.extend(sa.get("sources", []))

    chunks: list[dict] = []
    seen: set[str] = set()

    for source_ref in all_sources:
        if source_ref in seen:
            continue
        seen.add(source_ref)

        # Parse "filename.txt:chunk_N"
        if ":chunk_" not in source_ref:
            chunks.append({"source": source_ref, "text": "(unparseable source ref)"})
            continue

        filename, chunk_part = source_ref.rsplit(":chunk_", 1)
        try:
            chunk_index = int(chunk_part)
        except ValueError:
            chunks.append({"source": source_ref, "text": "(invalid chunk index)"})
            continue

        try:
            scroll_result, _ = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.case_id",
                            match=MatchValue(value=TEST_CASE_ID),
                        ),
                        FieldCondition(
                            key="metadata.source_file",
                            match=MatchValue(value=filename),
                        ),
                        FieldCondition(
                            key="metadata.chunk_index",
                            match=MatchValue(value=chunk_index),
                        ),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if scroll_result:
                payload = scroll_result[0].payload or {}
                # LangChain Qdrant stores chunk text under "page_content"
                text = (
                    payload.get("page_content")
                    or payload.get("content")
                    or "(no text in payload)"
                )
                chunks.append({"source": source_ref, "text": text})
            else:
                chunks.append({"source": source_ref, "text": "(chunk not found in Qdrant)"})
        except Exception as exc:
            chunks.append({"source": source_ref, "text": f"(Qdrant error: {exc})"})

    return chunks


def _print_test_report(
    test_name: str,
    query: str,
    result: dict,
    expected_keywords: list[str],
    file_ingestor: Any,
):
    """Print a structured evaluation report to stdout after each test."""
    divider  = "=" * 72
    thin     = "-" * 72

    sub_answers   = result.get("sub_answers", [])
    answer_text   = _get_answer_text(result)
    source_chunks = _fetch_source_chunks(file_ingestor, sub_answers)
    found_kw      = [kw for kw in expected_keywords if kw in answer_text]
    missing_kw    = [kw for kw in expected_keywords if kw not in answer_text]

    print(f"\n{divider}")
    print(f"TEST  : {test_name}")
    print(f"QUERY : {query}")
    print(thin)

    print("▶ ANSWER:")
    print(answer_text if answer_text.strip() else "(empty)")
    print(thin)

    print(f"▶ EXPECTED KEYWORDS : {expected_keywords}")
    print(f"  FOUND             : {found_kw}")
    print(f"  MISSING           : {missing_kw}")
    verdict = "✓ PASS" if found_kw else "✗ FAIL (no keyword matched)"
    print(f"  VERDICT           : {verdict}")
    print(thin)

    print(f"▶ SOURCE CHUNKS USED BY LLM ({len(source_chunks)}):")
    if source_chunks:
        for i, chunk in enumerate(source_chunks, 1):
            print(f"\n  [{i}] {chunk['source']}")
            text = chunk["text"]
            print(f"  {text[:600]}{'...' if len(text) > 600 else ''}")
    else:
        print("  (none — sources list was empty)")

    print(thin)
    print(f"  on_topic      : {result.get('on_topic')}")
    print(f"  doc_mode      : {result.get('doc_selection_mode')}")
    print(f"  sub_questions : {result.get('sub_questions')}")
    print(f"  error         : {result.get('error')}")
    print(divider)


def _assert_common(result: dict):
    assert result.get("error") is None, f"Graph error: {result.get('error')}"
    assert result.get("on_topic") is True, "Classified as off-topic"
    sub_answers = result.get("sub_answers", [])
    assert len(sub_answers) >= 1, "No sub_answers returned"
    assert any(sa.get("found") for sa in sub_answers), "No sub_answer with found=True"
    assert len(_get_answer_text(result).strip()) > 0, "Answer is empty"


# ---------------------------------------------------------------------------
# B1 -- Simple: plaintiff identity
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_query_plaintiff_address(app, file_ingestor):
    """Simple: Where does the plaintiff live?"""
    query = "أين يسكن المدعي؟"
    expected = ["المعادي", "القاهرة"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B1 - plaintiff address", query, result, expected, file_ingestor)


# ---------------------------------------------------------------------------
# B2 -- Simple: court compensation amount
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_query_compensation_amount(app, file_ingestor):
    """Simple: What compensation amount did the court order?"""
    query = "كم مبلغ التعويض الذي حكمت به المحكمة؟"
    expected = ["مليون وخمسمائة", "1,500,000", "1500000"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B2 - compensation amount", query, result, expected, file_ingestor)


# ---------------------------------------------------------------------------
# B3 -- Simple: forensic conclusion on signature
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_query_forensic_conclusion(app, file_ingestor):
    """Simple: What did the forensic service conclude about the contract signature?"""
    query = "ما هو رأي مصلحة الطب الشرعي في التوقيع الموجود على العقد؟"
    expected = ["تزوير", "زائف", "توقيع"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B3 - forensic conclusion", query, result, expected, file_ingestor)


# ---------------------------------------------------------------------------
# B4 -- Simple: what happened in the 25 March session
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_query_session_decision(app, file_ingestor):
    """Simple: What did the court decide in the 25 March 2024 session?"""
    query = "ما الذي قررته المحكمة في جلسة 25 مارس 2024؟"
    expected = ["خبير", "تأجيل", "هندسي"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B4 - session decision", query, result, expected, file_ingestor)


# ---------------------------------------------------------------------------
# B5 -- Harder: expert damage estimate vs plaintiff's original claim
# ---------------------------------------------------------------------------

@pytest.mark.timeout(190)
def test_query_damage_vs_claim(app, file_ingestor):
    """Harder: Plaintiff claimed 2M EGP. What did the expert actually assess,
    and what is the gap versus the claim in the bill of complaint?
    Requires retrieving صحيفة دعوى (claim amount) AND تقرير خبير (estimate).
    """
    query = (
        "المدعي طلب تعويضاً معيناً في صحيفة الدعوى — كم قدّر الخبير الهندسي "
        "الأضرار الفعلية بالعقار وما هو الفارق بين التقديرين؟"
    )
    expected = ["200", "مليوني", "أضرار", "ترميم"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B5 - damage estimate vs claim", query, result, expected, file_ingestor)


# ---------------------------------------------------------------------------
# B6 -- Harder: what evidence did the court rely on in its judgment
# ---------------------------------------------------------------------------

@pytest.mark.timeout(90)
def test_query_judgment_basis(app, file_ingestor):
    """Harder: What reports/evidence did the court cite when ruling for rescission
    and compensation? Needs حكم + internal references to forensic/expert reports.
    """
    query = (
        "على أي تقارير أو أدلة استندت المحكمة تحديداً لإصدار حكمها "
        "بفسخ العقد والتعويض؟"
    )
    expected = ["طب الشرعي", "471", "خبير", "تقرير"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B6 - judgment evidential basis", query, result, expected, file_ingestor)


# ---------------------------------------------------------------------------
# B7 -- Harder: defense line raised vs what the court actually ruled
# ---------------------------------------------------------------------------

def test_query_defense_vs_ruling(app, file_ingestor):
    """Harder: Did the company defendant raise a jurisdiction defense, and did
    the court address it in the judgment?
    Requires مذكرة بدفاع (defendant 2) + حكم to answer fully.
    """
    query = (
        "هل أثارت شركة العقارات الحديثة دفعاً شكلياً في مذكرة دفاعها، "
        "وكيف تعاملت المحكمة مع هذه الدفوع في حكمها؟"
    )
    expected = ["اختصاص", "شركة", "المحكمة", "دفع"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    _print_test_report("B7 - defense vs ruling", query, result, expected, file_ingestor)