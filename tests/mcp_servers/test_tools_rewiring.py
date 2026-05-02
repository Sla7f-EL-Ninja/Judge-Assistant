import importlib
import sys

import pytest

from tests.mcp_servers.conftest import CIVIL_LAW_QUERY, OFF_TOPIC_QUERY, SEEDED_CASE_ID

pytestmark = pytest.mark.integration


class TestNoDirectRAGImports:
    def test_chat_reasoner_tools_no_civil_law_rag_service(self):
        import chat_reasoner.tools  # noqa: F401
        assert "RAG.civil_law_rag.service" not in sys.modules

    def test_cr_tools_no_ask_question_attr(self):
        import CR.tools as cr_module
        assert not hasattr(cr_module, "ask_question")

    def test_chat_reasoner_tools_does_not_modify_sys_path(self):
        before = list(sys.path)
        importlib.reload(sys.modules["chat_reasoner.tools"])
        assert sys.path == before


class TestChatReasonerTools:
    @pytest.fixture(autouse=True)
    def _require_servers(self, mcp_servers_started):
        pass

    def _make_step(self, tool: str, query: str) -> dict:
        return {
            "step_id": "S1",
            "tool": tool,
            "query": query,
            "depends_on": [],
            "request_id": None,
        }

    def test_run_civil_law_rag_success(self):
        from chat_reasoner.tools import _run_civil_law_rag
        step = self._make_step("civil_law_rag", CIVIL_LAW_QUERY)
        result = _run_civil_law_rag(step, "", [])
        assert result.status == "success"
        assert result.response
        assert result.step_id == "S1"
        assert result.tool == "civil_law_rag"
        assert all(isinstance(s, str) for s in result.sources)

    def test_run_civil_law_rag_with_prior_results(self):
        from chat_reasoner.tools import _run_civil_law_rag
        prior = [{"response": "المدعي يملك عقاراً في القاهرة منذ عام 2010.", "tool": "case_doc_rag"}]
        step = self._make_step("civil_law_rag", "ما حكم القانون في حيازة العقار؟")
        result = _run_civil_law_rag(step, "", [], prior_results=prior)
        assert result.status == "success"

    def test_run_case_doc_rag_success(self):
        from chat_reasoner.tools import _run_case_doc_rag
        step = self._make_step("case_doc_rag", "ما هي وقائع القضية؟")
        result = _run_case_doc_rag(step, SEEDED_CASE_ID, [])
        assert result.status == "success"
        assert result.response
        assert result.tool == "case_doc_rag"

    def test_run_case_doc_rag_off_topic(self):
        from chat_reasoner.tools import _run_case_doc_rag
        step = self._make_step("case_doc_rag", OFF_TOPIC_QUERY)
        result = _run_case_doc_rag(step, SEEDED_CASE_ID, [])
        assert result.status == "failure"
        assert result.error == "off_topic"

    def test_run_civil_law_rag_raw_output_has_metadata(self):
        from chat_reasoner.tools import _run_civil_law_rag
        step = self._make_step("civil_law_rag", CIVIL_LAW_QUERY)
        result = _run_civil_law_rag(step, "", [])
        assert "classification" in result.raw_output
        assert "retrieval_confidence" in result.raw_output


class TestCRTools:
    @pytest.fixture(autouse=True)
    def _require_servers(self, mcp_servers_started):
        pass

    def test_civil_law_rag_tool_success(self):
        from CR.tools import civil_law_rag_tool
        result = civil_law_rag_tool(CIVIL_LAW_QUERY)
        assert result["answer"]
        assert result["error"] is None
        assert isinstance(result["sources"], list)
        assert "classification" in result
        assert "retrieval_confidence" in result
        assert "from_cache" in result

    def test_civil_law_rag_tool_invalid_query_returns_dict(self):
        from CR.tools import civil_law_rag_tool
        result = civil_law_rag_tool("what is contract law")
        assert result["answer"] == ""
        assert result["error"] is not None

    def test_case_documents_rag_tool_success(self):
        from CR.tools import case_documents_rag_tool
        result = case_documents_rag_tool("ما هي وقائع القضية؟", SEEDED_CASE_ID)
        assert result["final_answer"]
        assert result["error"] is None
        assert isinstance(result["sources"], list)
        assert isinstance(result["sub_answers"], list)

    def test_case_documents_rag_tool_off_topic(self):
        from CR.tools import case_documents_rag_tool
        result = case_documents_rag_tool(OFF_TOPIC_QUERY, SEEDED_CASE_ID)
        assert result["final_answer"] == ""
        assert result["error"] == "off_topic"
