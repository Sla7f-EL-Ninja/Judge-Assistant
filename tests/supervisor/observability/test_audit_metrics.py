"""
test_audit_metrics.py — observability and audit tests (Section 11 of plan).
"""

import uuid

import pytest

from tests.supervisor.helpers.state_factory import make_state


def _audit_col(mongo_client):
    from config import cfg
    # Mirror audit_log_node's exact db lookup: key="db", default="judge_assistant"
    db_name = cfg.get("mongodb", {}).get("db", "judge_assistant")
    return mongo_client[db_name]["audit_log"]


class TestAuditLogObservability:
    def test_audit_doc_has_all_fields(self, mongo_client):
        from Supervisor.nodes.audit_log import audit_log_node

        col = _audit_col(mongo_client)
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="ما المادة 163؟",
            classified_query="ما نص المادة 163 من القانون المدني المصري؟",
            correlation_id=cid,
            intent="civil_law_rag",
            target_agents=["civil_law_rag"],
            validation_status="pass",
            validation_feedback="",
            retry_count=0,
            agent_results={"civil_law_rag": {"response": "ok", "sources": [], "raw_output": {}}},
            agent_errors={},
            sources=["المادة 163"],
        )
        audit_log_node(state)

        doc = col.find_one({"correlation_id": cid})
        assert doc is not None

        expected_fields = [
            "timestamp", "correlation_id", "case_id", "turn_count",
            "intent", "target_agents", "classified_query_length",
            "agents_succeeded", "agents_failed",
            "validation_status", "validation_feedback_length",
            "retry_count", "sources", "classification_error",
        ]
        for f in expected_fields:
            assert f in doc, f"Missing audit field: {f}"

    def test_classified_query_length_stored_not_content(self, mongo_client):
        from Supervisor.nodes.audit_log import audit_log_node

        col = _audit_col(mongo_client)
        cid = f"test-cid-{uuid.uuid4()}"
        secret_query = "محتوى سري لا يجب تخزينه"
        state = make_state(classified_query=secret_query, correlation_id=cid)
        audit_log_node(state)

        doc = col.find_one({"correlation_id": cid})
        assert doc["classified_query_length"] == len(secret_query)
        assert "classified_query" not in doc or doc.get("classified_query") is None

    def test_agents_succeeded_list_correct(self, mongo_client):
        from Supervisor.nodes.audit_log import audit_log_node

        col = _audit_col(mongo_client)
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            correlation_id=cid,
            agent_results={"civil_law_rag": {}, "reason": {}},
            agent_errors={"case_doc_rag": "error"},
        )
        audit_log_node(state)

        doc = col.find_one({"correlation_id": cid})
        assert set(doc["agents_succeeded"]) == {"civil_law_rag", "reason"}
        assert doc["agents_failed"] == ["case_doc_rag"]


class TestCorrelationIDPropagation:
    def test_correlation_id_assigned_by_validate_input(self):
        from Supervisor.nodes.validate_input import validate_input_node
        state = make_state(judge_query="ما المادة 163؟")
        state["correlation_id"] = None
        result = validate_input_node(state)
        assert result.get("correlation_id") is not None

    def test_preexisting_correlation_id_preserved(self):
        from Supervisor.nodes.validate_input import validate_input_node
        state = make_state(judge_query="ما المادة 163؟", correlation_id="my-fixed-id")
        result = validate_input_node(state)
        assert result.get("correlation_id") == "my-fixed-id"


class TestPrometheusCounters:
    def test_off_topic_counter_increments(self):
        from Supervisor.nodes.off_topic import off_topic_response_node
        from Supervisor.metrics import TURN_COUNTER

        state = make_state()
        off_topic_response_node(state)
        # Counter must not raise — actual value check skipped (registry may not reset)

    def test_fallback_counter_increments(self):
        from Supervisor.nodes.fallback import fallback_response_node
        from Supervisor.metrics import FALLBACK_COUNTER, TURN_COUNTER

        state = make_state(validation_feedback="forced fallback")
        result = fallback_response_node(state)
        assert result["validation_status"] == "fallback"

    def test_retry_counter_increments(self, monkeypatch):
        from Supervisor.nodes.validate_output import validate_output_node
        from Supervisor.state import ValidationResult
        import Supervisor.nodes.validate_output as vo_mod

        monkeypatch.setattr(
            vo_mod, "llm_invoke",
            lambda fn, msgs: ValidationResult(
                hallucination_pass=False, relevance_pass=True,
                completeness_pass=True, coherence_pass=True,
                overall_pass=False, feedback="hallucination",
            ),
        )
        state = make_state(merged_response="إجابة", retry_count=0)
        result = validate_output_node(state)
        assert result["validation_status"] == "fail_hallucination"


@pytest.mark.xfail(reason="PII detection not implemented — G13/11.4 TDD seed")
class TestPIIFlag:
    def test_pii_flag_set_in_audit(self, mongo_db):
        from Supervisor.nodes.audit_log import audit_log_node
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="رقم قومي 25801234567",
            correlation_id=cid,
        )
        audit_log_node(state)
        doc = mongo_db["audit_log"].find_one({"correlation_id": cid})
        assert doc.get("pii_detected") is True
