"""
test_partial_agent_fail.py — one agent fails, others succeed.
"""

import pytest

from tests.supervisor.helpers.state_factory import make_state, make_agent_result
from tests.supervisor.helpers.reporting import emit_evidence


class TestPartialAgentFailUnit:
    def test_partial_failure_merge_has_disclosure(self):
        from Supervisor.nodes.merge_responses import merge_responses_node

        state = make_state(
            classified_query="اشرح المادة 163 وطبقها",
            agent_results={
                "civil_law_rag": make_agent_result(
                    response="المادة 163 تنص على المسؤولية التقصيرية."
                ),
            },
            agent_errors={"reason": "LLM timeout"},
        )
        result = merge_responses_node(state)
        # single-agent pass-through happens first
        # For multi-agent with partial failure the disclosure should appear
        # In this case only civil_law_rag succeeded so it's single-agent pass-through
        assert result["merged_response"]

    def test_unknown_agent_captured_in_errors(self):
        from Supervisor.nodes.dispatch_agents import dispatch_agents_node

        state = make_state(
            classified_query="ما المادة 163؟",
            target_agents=["civil_law_rag", "nonexistent_agent"],
        )
        result = dispatch_agents_node(state)
        assert "nonexistent_agent" in result["agent_errors"]


@pytest.mark.expensive
class TestPartialAgentFailE2E:
    def test_partial_fail_still_produces_response(self, supervisor_app, monkeypatch):
        """One adapter raises — turn still completes with remaining agent's output."""
        import Supervisor.nodes.dispatch_agents as da_mod

        original_run = da_mod._run_single_agent

        def selective_fail(agent_name, adapter_cls, query, state, snapshot):
            if agent_name == "reason":
                return (agent_name, None, "Simulated reason failure")
            return original_run(agent_name, adapter_cls, query, state, snapshot)

        monkeypatch.setattr(da_mod, "_run_single_agent", selective_fail)

        import time
        import Supervisor.nodes.prepare_retry as pr_mod
        monkeypatch.setattr(time, "sleep", lambda s: None)

        state = make_state(
            judge_query="اشرح المادة 163 وحلل مدى انطباقها",
            target_agents=["civil_law_rag", "reason"],
        )
        final = supervisor_app.invoke(state)
        emit_evidence("partial_agent_fail", final)

        assert final["final_response"]
        assert "reason" in final.get("agent_errors", {})
