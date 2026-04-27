"""
test_e2e_single_agent.py — one happy-path E2E test per intent.

Invokes the full supervisor graph (get_app().invoke(state)).
All tests use real Gemini + real DB. @pytest.mark.expensive.
"""

import time
import uuid

import pytest

from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.llm_assertions import assert_arabic_response, assert_no_injection_leak
from tests.supervisor.helpers.reporting import emit_evidence


@pytest.mark.expensive
class TestE2ESingleAgentCivilLaw:
    def test_civil_law_rag_happy_path(self, supervisor_app, mongo_db, judicial_docs_available):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="اشرح المادة 163 من القانون المدني",
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_civil_law_rag", final, latency)

        assert final["validation_status"] in ("pass", "partial_pass"), (
            f"Expected pass, got {final['validation_status']!r}. "
            f"feedback={final.get('validation_feedback')}"
        )
        assert final["retry_count"] == 0
        assert_arabic_response(final["final_response"], min_len=50)
        assert "163" in final["final_response"] or "المسؤولية" in final["final_response"]
        assert_no_injection_leak(final["final_response"])
        assert final["correlation_id"] == cid
        assert latency < 60, f"Single-agent latency {latency:.1f}s > 60s"

        # Verify audit log written
        doc = mongo_db["audit_log"].find_one({"correlation_id": cid})
        assert doc is not None, "Audit log not written"
        assert doc["intent"] == "civil_law_rag"


@pytest.mark.expensive
class TestE2ESingleAgentCaseDoc:
    def test_case_doc_rag_happy_path(self, supervisor_app, mongo_db, seeded_case):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="ما قيمة الأضرار المقدرة من الخبير؟",
            case_id=seeded_case,
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_case_doc_rag", final, latency)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
        assert_arabic_response(final["final_response"], min_len=20)
        assert latency < 60


@pytest.mark.expensive
class TestE2ESingleAgentOffTopic:
    def test_off_topic_happy_path(self, supervisor_app, mongo_db):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="ما عاصمة فرنسا؟",
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_off_topic", final, latency)

        from Supervisor.prompts import OFF_TOPIC_RESPONSE
        assert final["final_response"] == OFF_TOPIC_RESPONSE
        assert final["validation_status"] == "pass"
        assert latency < 15, f"Off-topic should be fast, got {latency:.1f}s"

        # Audit log check
        doc = mongo_db["audit_log"].find_one({"correlation_id": cid})
        assert doc is not None


@pytest.mark.expensive
class TestE2ESingleAgentReason:
    def test_reason_happy_path(self, supervisor_app, seeded_case):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="بناءً على ملف القضية، هل تتحقق المسؤولية التضامنية للمدعى عليهم؟",
            case_id=seeded_case,
            correlation_id=cid,
        )
        start = time.time()
        final = supervisor_app.invoke(state)
        latency = time.time() - start

        emit_evidence("e2e_reason", final, latency)

        assert final["validation_status"] in ("pass", "partial_pass", "fallback")
        assert_arabic_response(final["final_response"], min_len=20)
        assert latency < 120
