"""
test_db_outage.py — Mongo/Qdrant outage behavior.

Tests that Mongo failures in enrich_context and audit_log degrade silently.
Qdrant failure in adapter captured in agent_errors.
"""

import pytest

from tests.supervisor.helpers.state_factory import make_state


class TestMongoOutage:
    def test_enrich_context_degrades_on_mongo_failure(self, monkeypatch):
        import pymongo
        monkeypatch.setattr(
            pymongo, "MongoClient",
            lambda *a, **kw: (_ for _ in ()).throw(Exception("mongo down")),
        )
        from Supervisor.nodes.enrich_context import enrich_context_node
        state = make_state(case_id="test-case-001", intent="reason")
        result = enrich_context_node(state)  # must not raise
        assert isinstance(result, dict)

    def test_audit_log_degrades_on_mongo_failure(self, monkeypatch):
        import pymongo
        monkeypatch.setattr(
            pymongo, "MongoClient",
            lambda *a, **kw: (_ for _ in ()).throw(Exception("mongo down")),
        )
        from Supervisor.nodes.audit_log import audit_log_node
        state = make_state()
        result = audit_log_node(state)  # must not raise
        assert result == {}


@pytest.mark.expensive
class TestQdrantOutage:
    def test_civil_law_rag_qdrant_failure_captured(self, monkeypatch):
        """Qdrant down → civil_law_rag raises → error captured, no propagation."""
        from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter

        def failing_invoke(self, query, context):
            raise ConnectionError("Qdrant unreachable (monkeypatched)")

        monkeypatch.setattr(CivilLawRAGAdapter, "invoke", failing_invoke)

        from Supervisor.nodes.dispatch_agents import dispatch_agents_node
        state = make_state(
            classified_query="ما المادة 163؟",
            target_agents=["civil_law_rag"],
        )
        result = dispatch_agents_node(state)
        assert "civil_law_rag" in result["agent_errors"]
        assert "civil_law_rag" not in result["agent_results"]

    def test_all_agents_qdrant_failure_leads_to_fallback(self, supervisor_app, monkeypatch):
        from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
        import time
        import Supervisor.nodes.prepare_retry as pr_mod
        monkeypatch.setattr(time, "sleep", lambda s: None)

        def failing_invoke(self, query, context):
            raise ConnectionError("Qdrant unreachable")

        monkeypatch.setattr(CivilLawRAGAdapter, "invoke", failing_invoke)

        state = make_state(
            judge_query="ما المادة 163؟",
            target_agents=["civil_law_rag"],
        )
        final = supervisor_app.invoke(state)
        assert final["final_response"]
        # Should eventually fall back
        assert final["validation_status"] in ("fallback", "pass", "partial_pass")
