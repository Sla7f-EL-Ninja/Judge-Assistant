"""
tests/CASE_RAG/test_happy_path.py

Layer B: On-topic retrieval tests.

Features:
- case_doc_rag.* loggers piped to stdout at DEBUG level (visible with pytest -s)
- Mix of simple and moderately harder retrieval questions
- After each test: prints the answer, the source chunks retrieved from Qdrant,
  and the expected keywords so results can be evaluated manually
- Registers results via the register_result fixture to populate the JSON telemetry report.
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
    ):
        l = logging.getLogger(logger_name)
        l.addHandler(handler)
        l.setLevel(logging.DEBUG)

def _assert_common(result):
    assert result.get("error") is None
    assert result.get("on_topic") is True
    assert len(result.get("final_answer", "").strip()) > 0

def _get_answer_text(result):
    return result.get("final_answer", "")

def _print_test_report(name, query, result, expected, file_ingestor):
    print(f"\n=== TEST {name} ===")
    print(f"Query: {query}")
    print(f"Answer: {result.get('final_answer')}")
    print(f"Expected keywords: {expected}")


# ---------------------------------------------------------------------------
# B1 -- Basic: Property address
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_property_address(app, file_ingestor, register_result):
    query = "ما هو عنوان العقار محل النزاع بالتفصيل؟"
    expected = ["73", "السكة الحديد", "الإسماعيلية", "عرايشية"]
    
    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected)
    
    register_result({
        "layer": "B",
        "test_id": "B1",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B1 - address", query, result, expected, file_ingestor)

# ---------------------------------------------------------------------------
# B2 -- Basic: Owner name
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_owner_name(app, file_ingestor, register_result):
    query = "من هو مالك العقار المذكور في الأوراق؟"
    expected = ["محمد", "أحمد", "علي"]
    
    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected)
    
    register_result({
        "layer": "B",
        "test_id": "B2",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B2 - owner", query, result, expected, file_ingestor)

# ---------------------------------------------------------------------------
# B3 -- Basic: Tenant name
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_tenant_name(app, file_ingestor, register_result):
    query = "من هو المستأجر الرئيسي أو الطاعن في هذه القضية؟"
    expected = ["محمد", "إبراهيم", "سعيد"]
    
    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected)
    
    register_result({
        "layer": "B",
        "test_id": "B3",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B3 - tenant", query, result, expected, file_ingestor)

# ---------------------------------------------------------------------------
# B4 -- Moderate: Eviction Decision Number (Robust for OCR realities)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_eviction_decision(app, file_ingestor, register_result):
    query = "ما هو رقم قرار الإخلاء الإداري الصادر من حي ثان الإسماعيلية للعقار؟"
    expected = ["17", "١٧", "193", "١٩٣", "۱۹۳", "2020", "٢٠٢٠", "۲۰۲۰"]
    
    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)}"
    )
    
    register_result({
        "layer": "B",
        "test_id": "B4",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B4 - eviction decision", query, result, expected, file_ingestor)

# ---------------------------------------------------------------------------
# B5 -- Moderate: Consulting Engineer Report (Aligned with document phrasing)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_engineer_conclusion(app, file_ingestor, register_result):
    query = "ما هو الرأي الفني في التقرير الهندسي الاستشاري للمهندس برسوم شنودة بشأن العقار؟"
    expected = ["ترميم", "شروخ", "سطحية", "بياض"]
    
    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    
    register_result({
        "layer": "B",
        "test_id": "B5",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B5 - engineer conclusion", query, result, expected, file_ingestor)

# ---------------------------------------------------------------------------
# B6 -- Moderate: Defense Legal Basis
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_defense_legal_basis(app, file_ingestor, register_result):
    query = (
        "ما هي الأسانيد القانونية التي استندت إليها مذكرة دفاع المالك "
        "لتبرير صحة قرار لجنة التظلمات بالإزالة؟"
    )
    expected = ["119", "2008", "البناء الموحد", "90", "92"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected), (
        f"Expected one of {expected}. Got: {_get_answer_text(result)[:300]}"
    )
    
    register_result({
        "layer": "B",
        "test_id": "B6",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B6 - defense legal basis", query, result, expected, file_ingestor)

# ---------------------------------------------------------------------------
# B7 -- Harder: Tenant's specific plea regarding applicant status
# ---------------------------------------------------------------------------

@pytest.mark.timeout(150)
def test_query_tenant_plea_status(app, file_ingestor, register_result):
    query = (
        "ما هو الدفع الرئيسي الذي أثاره المدعي (المستأجر) في صحيفة الدعوى "
        "لإثبات بطلان القرار فيما يخص صفة مقدم التظلم؟"
    )
    expected = ["انعدام", "صفة", "توكيل", "ملكية"]

    result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    _assert_common(result)
    assert any(kw in _get_answer_text(result) for kw in expected)
    
    register_result({
        "layer": "B",
        "test_id": "B7",
        "query": query,
        "expected_keywords": expected,
        "answer_text": _get_answer_text(result),
        "sub_answers": result.get("sub_answers", []),
        "doc_selection_mode": result.get("doc_selection_mode"),
        "on_topic": result.get("on_topic"),
    })
    
    _print_test_report("B7 - tenant plea status", query, result, expected, file_ingestor)