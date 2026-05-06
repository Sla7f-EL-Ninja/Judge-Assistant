"""
test_audit_log.py — unit tests for audit_log_node.

Uses real MongoDB. Verifies all 14 fields are written; confirms sensitive content
is NOT stored; verifies Mongo failure is swallowed.
"""

import uuid

import pytest

from Supervisor.nodes.audit_log import audit_log_node
from tests.supervisor.helpers.state_factory import make_state


def _audit_col(mongo_client):
    from config import cfg
    # Mirror audit_log_node's exact db lookup: key="db", default="judge_assistant"
    db_name = cfg.get("mongodb", {}).get("db", "judge_assistant")
    return mongo_client[db_name]["audit_log"]


@pytest.fixture
def audit_col(mongo_client):
    return _audit_col(mongo_client)


class TestAuditLogWrite:
    def test_writes_document(self, audit_col):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="ما المادة 163؟",
            classified_query="ما نص المادة 163 من القانون المدني المصري؟",
            case_id="test-case-audit-001",
            correlation_id=cid,
            intent="civil_law_rag",
            target_agents=["civil_law_rag"],
            validation_status="pass",
            retry_count=0,
            agent_results={"civil_law_rag": {"response": "ok", "sources": [], "raw_output": {}}},
            agent_errors={},
            sources=["المادة 163 — المسؤولية التقصيرية"],
        )
        audit_log_node(state)

        doc = audit_col.find_one({"correlation_id": cid})
        assert doc is not None, "Audit document not written to MongoDB"

    def test_all_required_fields_present(self, audit_col):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            classified_query="ما نص المادة 163؟",
            correlation_id=cid,
            intent="civil_law_rag",
            target_agents=["civil_law_rag"],
            validation_status="pass",
            validation_feedback="",
            retry_count=0,
            agent_results={"civil_law_rag": {"response": "ok", "sources": [], "raw_output": {}}},
            agent_errors={},
            sources=[],
        )
        audit_log_node(state)
        doc = audit_col.find_one({"correlation_id": cid})
        assert doc is not None

        required_fields = [
            "timestamp", "correlation_id", "case_id", "turn_count",
            "intent", "target_agents", "classified_query_length",
            "agents_succeeded", "agents_failed",
            "validation_status", "validation_feedback_length",
            "retry_count", "sources", "classification_error",
        ]
        for field in required_fields:
            assert field in doc, f"Missing field in audit doc: {field}"

    def test_classified_query_length_correct(self, audit_col):
        cid = f"test-cid-{uuid.uuid4()}"
        query = "ما نص المادة 163 من القانون المدني المصري؟"
        state = make_state(classified_query=query, correlation_id=cid)
        audit_log_node(state)
        doc = audit_col.find_one({"correlation_id": cid})
        assert doc["classified_query_length"] == len(query)

    def test_sensitive_query_body_not_stored(self, audit_col):
        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="سؤال سري جداً — لا يجب تخزينه",
            classified_query="استفسار قانوني",
            correlation_id=cid,
        )
        audit_log_node(state)
        doc = audit_col.find_one({"correlation_id": cid})
        assert "judge_query" not in doc or doc.get("judge_query") is None
        # Only length stored, not content
        assert "classified_query_length" in doc

    def test_returns_empty_dict(self, audit_col):
        state = make_state()
        result = audit_log_node(state)
        assert result == {}

    def test_mongo_failure_swallowed(self, monkeypatch):
        """Audit log must never crash the turn even if Mongo is down."""
        import pymongo
        monkeypatch.setattr(pymongo, "MongoClient", lambda *a, **kw: (_ for _ in ()).throw(Exception("mongo down")))
        state = make_state(correlation_id=f"test-cid-{uuid.uuid4()}")
        result = audit_log_node(state)  # must not raise
        assert result == {}
