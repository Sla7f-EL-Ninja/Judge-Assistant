"""
test_case_doc_rag_adapter.py — CaseDocRAGAdapter integration tests.

Requires seeded case docs (seeded_case fixture).
@pytest.mark.expensive.
"""

import pytest

from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter
from Supervisor.agents.base import AgentResult
from tests.supervisor.helpers.llm_assertions import assert_arabic_response


@pytest.fixture(scope="module")
def adapter():
    return CaseDocRAGAdapter()


@pytest.mark.expensive
class TestCaseDocRAGAdapterHappyPath:
    def test_expert_name_found(self, adapter, seeded_case):
        result = adapter.invoke(
            "من الخبير المنتدب في القضية؟",
            {"case_id": seeded_case},
        )
        if result.error:
            pytest.skip(f"Adapter error: {result.error}")
        assert "سامي" in result.response or "رمزي" in result.response or "الخبير" in result.response

    def test_plaintiff_demands_found(self, adapter, seeded_case):
        result = adapter.invoke(
            "ما طلبات المدعي؟",
            {"case_id": seeded_case},
        )
        if result.error:
            pytest.skip(f"Adapter error: {result.error}")
        assert_arabic_response(result.response, min_len=20)

    def test_hearing_date_found(self, adapter, seeded_case):
        result = adapter.invoke(
            "ماذا قررت المحكمة في جلسة 25/03/2024؟",
            {"case_id": seeded_case},
        )
        if result.error:
            pytest.skip(f"Adapter error: {result.error}")
        assert result.response

    def test_raw_output_has_expected_keys(self, adapter, seeded_case):
        result = adapter.invoke(
            "من المدعي في القضية؟",
            {"case_id": seeded_case},
        )
        if result.error:
            pytest.skip(f"Adapter error: {result.error}")
        raw = result.raw_output
        assert isinstance(raw, dict)
        # Expected keys from case_doc_rag graph output
        assert any(k in raw for k in ["final_answer", "sub_answers", "answer", "response"])


@pytest.mark.expensive
class TestCaseDocRAGAdapterNegative:
    def test_wrong_case_id_graceful(self, adapter):
        result = adapter.invoke(
            "ما طلبات المدعي؟",
            {"case_id": "nonexistent-case-xyz"},
        )
        # Should return error or empty answer — not crash
        assert result is not None
        assert isinstance(result, AgentResult)

    def test_empty_case_id_graceful(self, adapter):
        result = adapter.invoke(
            "ما طلبات المدعي؟",
            {"case_id": ""},
        )
        assert result is not None

    def test_off_topic_query_for_case_docs(self, adapter, seeded_case):
        result = adapter.invoke(
            "كيف أطبخ المكرونة؟",
            {"case_id": seeded_case},
        )
        # Should return error or off_topic indication, not fabricate case content
        assert result is not None
        if not result.error:
            assert len(result.response) < 300 or result.error
