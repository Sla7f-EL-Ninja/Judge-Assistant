"""
test_all_agents_fail.py — all agents fail → merge → fallback path.
"""

import uuid

import pytest

from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.reporting import emit_evidence


@pytest.mark.expensive
class TestAllAgentsFail:
    def test_all_agents_fail_leads_to_fallback(self, supervisor_app, monkeypatch):
        """Patch all adapters to raise → verify fallback path."""
        import Supervisor.nodes.dispatch_agents as da_mod
        import time
        import Supervisor.nodes.prepare_retry as pr_mod

        monkeypatch.setattr(time, "sleep", lambda s: None)

        def always_fail(agent_name, adapter_cls, query, state, snapshot):
            return (agent_name, None, f"Simulated failure for {agent_name}")

        monkeypatch.setattr(da_mod, "_run_single_agent", always_fail)

        cid = f"test-cid-{uuid.uuid4()}"
        state = make_state(
            judge_query="ما المادة 163؟",
            target_agents=["civil_law_rag"],
            correlation_id=cid,
        )
        final = supervisor_app.invoke(state)
        emit_evidence("all_agents_fail", final)

        assert final["validation_status"] == "fallback"
        assert final["final_response"]
        assert "civil_law_rag" in final.get("agent_errors", {})

    def test_merge_emits_all_failed_message(self, monkeypatch):
        from Supervisor.nodes.merge_responses import merge_responses_node
        state = make_state(
            agent_results={},
            agent_errors={"civil_law_rag": "timeout", "reason": "LLM error"},
        )
        result = merge_responses_node(state)
        assert "All agents failed" in result["validation_feedback"]
        assert result["merged_response"] == ""
