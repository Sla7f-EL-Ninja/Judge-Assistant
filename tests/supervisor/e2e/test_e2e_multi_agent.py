"""
test_e2e_multi_agent.py — multi-agent E2E tests (≥3 combos).

Uses real Gemini + real DB. @pytest.mark.expensive.
"""

import time
import uuid

import pytest

from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.llm_assertions import (
    assert_arabic_response,
    assert_agents_ordered,
    assert_no_injection_leak,
)
from tests.supervisor.helpers.reporting import emit_evidence


@pytest.mark.expensive
class TestE2EMultiAgent:
    def test_civil_law_plus_case_doc_rag(self, supervisor_app, seeded_case, judicial_docs_available):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="اشرح المادة 163 وطبقها على القضية الحالية",
            case_id=seeded_case,
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_civil_plus_case_doc", final, latency)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
        assert_arabic_response(final["final_response"], min_len=50)
        assert_no_injection_leak(final["final_response"])
        assert latency < 120

        if final["target_agents"]:
            assert_agents_ordered(final["target_agents"])

    def test_civil_law_plus_reason(self, supervisor_app, seeded_case, judicial_docs_available):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="بناءً على المادة 163 ما الحكم في القضية؟",
            case_id=seeded_case,
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_civil_plus_reason", final, latency)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
        assert_arabic_response(final["final_response"], min_len=30)
        assert latency < 120

    def test_case_doc_plus_reason(self, supervisor_app, seeded_case):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="بناءً على ملف القضية، حلل الموقف القانوني للمدعى عليهم",
            case_id=seeded_case,
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_case_doc_plus_reason", final, latency)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
        assert_arabic_response(final["final_response"], min_len=30)
        assert latency < 120

    def test_sources_are_deduped_in_merged(self, supervisor_app, seeded_case, judicial_docs_available):
        state = make_state(
            judge_query="اشرح المادة 163 وطبقها على وقائع القضية",
            case_id=seeded_case,
        )
        final = supervisor_app.invoke(state)
        sources = final.get("sources", [])
        assert len(sources) == len(set(s.lower() for s in sources)), (
            f"Duplicate sources found: {sources}"
        )

    def test_multi_agent_agents_ordered(self, supervisor_app, seeded_case, judicial_docs_available):
        state = make_state(
            judge_query="اشرح المادة 163 ومادة 165، طبقها على القضية، وحلل التناقضات",
            case_id=seeded_case,
        )
        final = supervisor_app.invoke(state)
        if len(final.get("target_agents", [])) > 1:
            assert_agents_ordered(final["target_agents"])
