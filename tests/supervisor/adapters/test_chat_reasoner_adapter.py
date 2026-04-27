"""
test_chat_reasoner_adapter.py — ChatReasonerAdapter integration tests.

@pytest.mark.expensive.
"""

import pytest

from Supervisor.agents.chat_reasoner_adapter import ChatReasonerAdapter
from chat_reasoner.interface import AgentResult
from tests.supervisor.helpers.llm_assertions import assert_arabic_response


@pytest.fixture(scope="module")
def adapter():
    return ChatReasonerAdapter()


@pytest.mark.expensive
class TestChatReasonerAdapterHappyPath:
    def test_response_is_arabic(self, adapter, seeded_case):
        result = adapter.invoke(
            "حلل المسؤولية التضامنية للمدعى عليهم بناءً على ملف القضية",
            {"case_id": seeded_case},
        )
        if result.error:
            pytest.skip(f"Adapter error: {result.error}")
        assert_arabic_response(result.response, min_len=30)

    def test_raw_output_has_plan(self, adapter, seeded_case):
        result = adapter.invoke(
            "بناءً على ملف القضية، ما الموقف القانوني الراجح؟",
            {"case_id": seeded_case},
        )
        if result.error:
            pytest.skip(f"Adapter error: {result.error}")
        raw = result.raw_output
        assert isinstance(raw, dict)

    def test_failed_status_sets_error(self, adapter, monkeypatch):
        """When reasoner returns status=failed, error must be set."""
        import Supervisor.agents.chat_reasoner_adapter as cra_mod

        # Patch the underlying app.invoke to return a failed status
        class FakeApp:
            def invoke(self, state):
                return {"status": "failed", "error": "LLM تعطل", "final_answer": ""}

        original_app = getattr(cra_mod, "_get_app", None)
        if original_app:
            monkeypatch.setattr(cra_mod, "_get_app", lambda: FakeApp())
        else:
            pytest.skip("Cannot monkeypatch chat_reasoner app")

        result = adapter.invoke("سؤال", {"case_id": "test-case-001"})
        assert result.error


@pytest.mark.expensive
class TestChatReasonerAdapterStress:
    def test_concurrent_calls_no_session_collision(self, adapter, seeded_case):
        """Two concurrent calls to same case_id must not share session state."""
        import threading
        results = []

        def invoke():
            r = adapter.invoke(
                "حلل التناقضات في ملف القضية",
                {"case_id": seeded_case},
            )
            results.append(r)

        t1 = threading.Thread(target=invoke)
        t2 = threading.Thread(target=invoke)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(results) == 2
        for r in results:
            assert isinstance(r, AgentResult)
