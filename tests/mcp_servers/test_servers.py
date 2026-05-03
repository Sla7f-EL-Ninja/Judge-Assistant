import pytest

from mcp_servers.client import _RESPAWN_LIMIT
from mcp_servers.errors import ErrorCode, MCPUnavailable, ToolError

pytestmark = pytest.mark.integration


class TestLegalRAGServer:
    def test_happy_path(self, isolated_legal_client, civil_law_query):
        resp = isolated_legal_client.call(
            "search_legal_corpus",
            query=civil_law_query,
            corpus="civil_law",
        )
        assert resp["answer"]
        assert isinstance(resp["sources"], list)
        assert resp["corpus"] == "civil_law"
        assert "classification" in resp
        assert "retrieval_confidence" in resp
        assert "from_cache" in resp

    def test_invalid_corpus(self, isolated_legal_client, civil_law_query):
        with pytest.raises(ToolError) as exc_info:
            isolated_legal_client.call(
                "search_legal_corpus",
                query=civil_law_query,
                corpus="bogus_corpus",
            )
        assert exc_info.value.code == ErrorCode.INVALID_ARG

    def test_empty_query(self, isolated_legal_client):
        with pytest.raises(ToolError) as exc_info:
            isolated_legal_client.call(
                "search_legal_corpus",
                query="",
                corpus="civil_law",
            )
        assert exc_info.value.code == ErrorCode.QUERY_VALIDATION

    def test_non_arabic_query(self, isolated_legal_client):
        with pytest.raises(ToolError) as exc_info:
            isolated_legal_client.call(
                "search_legal_corpus",
                query="what is contract law",
                corpus="civil_law",
            )
        assert exc_info.value.code == ErrorCode.QUERY_VALIDATION

    def test_single_crash_recovery(self, isolated_legal_client, civil_law_query):
        assert isolated_legal_client._proc.poll() is None
        isolated_legal_client._proc.kill()
        isolated_legal_client._proc.wait()
        resp = isolated_legal_client.call(
            "search_legal_corpus",
            query=civil_law_query,
            corpus="civil_law",
        )
        assert resp["answer"]
        assert isolated_legal_client._respawns == 0

    def test_double_crash_raises_unavailable(self, isolated_legal_client, civil_law_query):
        isolated_legal_client._proc.kill()
        isolated_legal_client._proc.wait()
        isolated_legal_client._respawns = _RESPAWN_LIMIT
        with pytest.raises(MCPUnavailable):
            isolated_legal_client.call(
                "search_legal_corpus",
                query=civil_law_query,
                corpus="civil_law",
            )


class TestCaseDocServer:
    def test_happy_path(self, isolated_case_doc_client, seeded_case_id):
        resp = isolated_case_doc_client.call(
            "search_case_docs",
            query="ما هي وقائع القضية؟",
            case_id=seeded_case_id,
        )
        assert resp["answer"]
        assert isinstance(resp["sources"], list)
        assert "sub_answers" in resp
        assert "doc_selection_mode" in resp

    def test_empty_case_id(self, isolated_case_doc_client):
        with pytest.raises(ToolError) as exc_info:
            isolated_case_doc_client.call(
                "search_case_docs",
                query="ما هي وقائع القضية؟",
                case_id="",
            )
        assert exc_info.value.code == ErrorCode.INVALID_ARG

    def test_off_topic_query(self, isolated_case_doc_client, seeded_case_id, off_topic_query):
        with pytest.raises(ToolError) as exc_info:
            isolated_case_doc_client.call(
                "search_case_docs",
                query=off_topic_query,
                case_id=seeded_case_id,
            )
        assert exc_info.value.code == ErrorCode.OFF_TOPIC

    def test_conversation_history_accepted(self, isolated_case_doc_client, seeded_case_id):
        history = [
            {"role": "user", "content": "ما هي وقائع القضية؟"},
            {"role": "assistant", "content": "القضية تتعلق بنزاع عقاري."},
        ]
        resp = isolated_case_doc_client.call(
            "search_case_docs",
            query="هل هناك مستندات تثبت ملكية المدعي؟",
            case_id=seeded_case_id,
            conversation_history=history,
        )
        assert isinstance(resp, dict)

    def test_single_crash_recovery(self, isolated_case_doc_client, seeded_case_id):
        assert isolated_case_doc_client._proc.poll() is None
        isolated_case_doc_client._proc.kill()
        isolated_case_doc_client._proc.wait()
        resp = isolated_case_doc_client.call(
            "search_case_docs",
            query="ما هي وقائع القضية؟",
            case_id=seeded_case_id,
        )
        assert resp["answer"]
        assert isolated_case_doc_client._respawns == 0
