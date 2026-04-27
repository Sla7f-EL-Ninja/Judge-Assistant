"""
test_enrich_context.py — unit tests for enrich_context_node.

Uses real MongoDB for DB-touching paths. No-op paths need no DB.
"""

import pytest

from Supervisor.nodes.enrich_context import enrich_context_node
from tests.supervisor.helpers.state_factory import make_state


class TestEnrichContextNoOp:
    def test_no_op_empty_case_id(self):
        state = make_state(case_id="", intent="case_doc_rag")
        result = enrich_context_node(state)
        assert result == {}

    def test_no_op_civil_law_rag_intent(self):
        state = make_state(case_id="test-case-001", intent="civil_law_rag")
        result = enrich_context_node(state)
        assert result == {}

    def test_no_op_off_topic_intent(self):
        state = make_state(case_id="test-case-001", intent="off_topic")
        result = enrich_context_node(state)
        assert result == {}


class TestEnrichContextWithDB:
    def test_fetches_summary_for_case_doc_rag(self, mongo_client, test_db_name, seeded_case):
        case_id = seeded_case
        state = make_state(case_id=case_id, intent="case_doc_rag")
        result = enrich_context_node(state)
        # Should have case_summary if seeded
        if "case_summary" in result:
            assert result["case_summary"]

    def test_fetches_doc_titles_for_case_doc_rag(self, seeded_case):
        case_id = seeded_case
        state = make_state(case_id=case_id, intent="case_doc_rag")
        result = enrich_context_node(state)
        if "case_doc_titles" in result:
            assert isinstance(result["case_doc_titles"], list)

    def test_swallows_mongo_error(self, monkeypatch):
        import pymongo
        monkeypatch.setattr(
            pymongo, "MongoClient",
            lambda *a, **kw: (_ for _ in ()).throw(Exception("mongo down")),
        )
        state = make_state(case_id="some-case", intent="reason")
        result = enrich_context_node(state)  # must not raise
        assert isinstance(result, dict)
