import pytest

from mcp_servers.client import _RESPAWN_LIMIT
from mcp_servers.lifecycle import _MCP_CLIENTS
from Supervisor.agents.base import AgentResult
from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter

pytestmark = pytest.mark.integration


class TestCivilLawRAGAdapter:
    @pytest.fixture(autouse=True)
    def adapter(self):
        self._adapter = CivilLawRAGAdapter()

    def test_happy_path(self, mcp_servers_started, civil_law_query):
        result = self._adapter.invoke(civil_law_query, context={})
        assert isinstance(result, AgentResult)
        assert result.error is None
        assert result.response
        assert isinstance(result.sources, list)
        assert "classification" in result.raw_output
        assert "retrieval_confidence" in result.raw_output
        assert "from_cache" in result.raw_output

    def test_non_arabic_returns_error_result(self, mcp_servers_started):
        result = self._adapter.invoke("what is contract law", context={})
        assert isinstance(result, AgentResult)
        assert result.error is not None
        assert result.response == ""

    def test_server_unavailable_returns_mcp_unavailable(self, mcp_servers_started, civil_law_query):
        client = _MCP_CLIENTS["legal_rag"]
        orig_respawns = client._respawns
        client._respawns = _RESPAWN_LIMIT
        try:
            client._proc.kill()
            client._proc.wait()
            result = self._adapter.invoke(civil_law_query, context={})
            assert result.error is not None
            assert "MCP_UNAVAILABLE" in result.error
        finally:
            client._respawns = 0
            client.start()

    def test_sources_formatted_as_article_strings(self, mcp_servers_started, civil_law_query):
        result = self._adapter.invoke(civil_law_query, context={})
        if result.sources:
            for src in result.sources:
                assert isinstance(src, str)
                assert src.startswith("المادة")


class TestCaseDocRAGAdapter:
    @pytest.fixture(autouse=True)
    def adapter(self):
        self._adapter = CaseDocRAGAdapter()

    def test_happy_path(self, mcp_servers_started, seeded_case_id):
        result = self._adapter.invoke(
            "ما هي وقائع القضية؟",
            context={"case_id": seeded_case_id, "conversation_history": []},
        )
        assert result.error is None
        assert result.response
        assert "doc_selection_mode" in result.raw_output
        assert "sub_answers" in result.raw_output

    def test_off_topic_exact_error_string(self, mcp_servers_started, seeded_case_id, off_topic_query):
        result = self._adapter.invoke(
            off_topic_query,
            context={"case_id": seeded_case_id, "conversation_history": []},
        )
        assert result.error == "Case Doc RAG: query classified as off-topic for case documents"

    def test_empty_case_id_returns_error_result(self, mcp_servers_started):
        result = self._adapter.invoke(
            "ما هي وقائع القضية؟",
            context={"case_id": "", "conversation_history": []},
        )
        assert result.error is not None
        assert result.response == ""

    def test_correlation_id_forwarded(self, mcp_servers_started, seeded_case_id):
        result = self._adapter.invoke(
            "ما هي وقائع القضية؟",
            context={
                "case_id": seeded_case_id,
                "conversation_history": [],
                "correlation_id": "test-cid-abc123",
            },
        )
        assert isinstance(result, AgentResult)

    def test_server_unavailable_returns_mcp_unavailable(self, mcp_servers_started, seeded_case_id):
        client = _MCP_CLIENTS["case_doc_rag"]
        orig_respawns = client._respawns
        client._respawns = _RESPAWN_LIMIT
        try:
            client._proc.kill()
            client._proc.wait()
            result = self._adapter.invoke(
                "ما هي وقائع القضية؟",
                context={"case_id": seeded_case_id, "conversation_history": []},
            )
            assert result.error is not None
            assert "MCP_UNAVAILABLE" in result.error
        finally:
            client._respawns = 0
            client.start()
