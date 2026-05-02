import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from mcp_servers.errors import ToolError
from mcp_servers.lifecycle import get_client
from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter
from tests.mcp_servers.conftest import CIVIL_LAW_QUERY, SEEDED_CASE_ID

pytestmark = pytest.mark.integration


class TestConcurrentDispatch:
    @pytest.fixture(autouse=True)
    def _require_servers(self, mcp_servers_started):
        pass

    def test_tier0_parallel_both_succeed(self):
        civil_adapter = CivilLawRAGAdapter()
        case_adapter = CaseDocRAGAdapter()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_civil = pool.submit(
                civil_adapter.invoke,
                CIVIL_LAW_QUERY,
                {},
            )
            f_case = pool.submit(
                case_adapter.invoke,
                "ما هي وقائع القضية؟",
                {"case_id": SEEDED_CASE_ID, "conversation_history": []},
            )
            results = [f.result() for f in as_completed([f_civil, f_case])]

        for result in results:
            assert result.error is None
            assert result.response

    def test_repeated_parallel_calls_stable(self):
        adapter = CivilLawRAGAdapter()
        context = {}
        all_results = []

        for _ in range(3):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(adapter.invoke, CIVIL_LAW_QUERY, context)
                    for _ in range(3)
                ]
                all_results.extend(f.result() for f in as_completed(futures))

        assert len(all_results) == 9
        for result in all_results:
            assert result.error is None

    def test_lock_released_after_tool_error(self):
        client = get_client("legal_rag")

        with pytest.raises(ToolError):
            client.call("search_legal_corpus", query=CIVIL_LAW_QUERY, corpus="bogus")

        success_holder = []

        def valid_call():
            resp = client.call(
                "search_legal_corpus",
                query=CIVIL_LAW_QUERY,
                corpus="civil_law",
            )
            success_holder.append(resp["answer"])

        t = threading.Thread(target=valid_call)
        t.start()
        t.join(timeout=15)

        assert not t.is_alive(), "Lock was not released after ToolError — deadlock detected"
        assert success_holder
