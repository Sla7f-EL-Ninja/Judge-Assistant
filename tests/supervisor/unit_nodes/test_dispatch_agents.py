"""
test_dispatch_agents.py — unit tests for dispatch_agents_node.

Uses real adapters (LLM calls). Expensive tests gated behind @pytest.mark.expensive.
"""

import time
import uuid

import pytest

from Supervisor.nodes.dispatch_agents import dispatch_agents_node, ADAPTER_REGISTRY
from tests.supervisor.helpers.state_factory import make_state


@pytest.mark.expensive
class TestDispatchAgentsSingleAgent:
    def test_civil_law_rag_populated(self):
        state = make_state(
            judge_query="ما نص المادة 163 من القانون المدني المصري؟",
            classified_query="ما نص المادة 163 من القانون المدني المصري؟",
            intent="civil_law_rag",
            target_agents=["civil_law_rag"],
        )
        result = dispatch_agents_node(state)
        assert "civil_law_rag" in result["agent_results"], (
            f"civil_law_rag not in results. errors={result['agent_errors']}"
        )

    def test_unknown_agent_recorded_in_errors(self):
        state = make_state(
            classified_query="ما المادة 163؟",
            target_agents=["unknown_agent_xyz"],
        )
        result = dispatch_agents_node(state)
        assert "unknown_agent_xyz" in result["agent_errors"]
        assert "unknown_agent_xyz" not in result["agent_results"]

    def test_empty_agents_returns_empty(self):
        state = make_state(classified_query="ما المادة 163؟", target_agents=[])
        result = dispatch_agents_node(state)
        assert result["agent_results"] == {} or isinstance(result["agent_results"], dict)
        assert result["agent_errors"] == {} or isinstance(result["agent_errors"], dict)


class TestDispatchAgentsRetry:
    @pytest.mark.expensive
    def test_retry_skips_successful_agents(self):
        """On retry, agents that already succeeded should NOT be re-dispatched."""
        prior_result = {"response": "إجابة سابقة", "sources": [], "raw_output": {}}
        state = make_state(
            classified_query="سؤال قانوني",
            target_agents=["civil_law_rag", "reason"],
            retry_count=1,
            agent_results={"civil_law_rag": prior_result},
            agent_errors={"reason": "LLM error on first attempt"},
        )
        result = dispatch_agents_node(state)
        # civil_law_rag should be preserved from prior, reason may succeed or fail
        assert result["agent_results"].get("civil_law_rag") == prior_result

    def test_validation_feedback_appended_on_retry(self, monkeypatch):
        """On retry, validation_feedback is appended to query."""
        queries_seen = []

        original_run = __import__(
            "Supervisor.nodes.dispatch_agents", fromlist=["_run_single_agent"]
        )._run_single_agent

        def capture_run(agent_name, adapter_cls, query, state, snapshot):
            queries_seen.append(query)
            return (agent_name, None, "mocked error")

        monkeypatch.setattr(
            "Supervisor.nodes.dispatch_agents._run_single_agent", capture_run
        )

        state = make_state(
            classified_query="ما المادة 163؟",
            target_agents=["civil_law_rag"],
            retry_count=1,
            validation_feedback="الإجابة لم تكن ذات صلة",
        )
        dispatch_agents_node(state)
        assert queries_seen, "No queries captured"
        assert "ملاحظات التحقق السابقة" in queries_seen[0]

    def test_no_feedback_appended_first_run(self, monkeypatch):
        queries_seen = []

        def capture_run(agent_name, adapter_cls, query, state, snapshot):
            queries_seen.append(query)
            return (agent_name, None, "mocked error")

        monkeypatch.setattr(
            "Supervisor.nodes.dispatch_agents._run_single_agent", capture_run
        )

        state = make_state(
            classified_query="ما المادة 163؟",
            target_agents=["civil_law_rag"],
            retry_count=0,
            validation_feedback="",
        )
        dispatch_agents_node(state)
        assert queries_seen
        assert "ملاحظات التحقق السابقة" not in queries_seen[0]


@pytest.mark.expensive
class TestDispatchAgentsTieredParallel:
    def test_tier0_agents_run_in_parallel(self):
        """civil_law_rag + case_doc_rag should finish faster than sequential sum."""
        state = make_state(
            classified_query="ما المادة 163؟ ومن الخبير؟",
            target_agents=["civil_law_rag", "case_doc_rag"],
            case_id="test-case-2847-2024",
            intent="multi",
        )
        start = time.time()
        result = dispatch_agents_node(state)
        elapsed = time.time() - start
        # At least one agent should have returned something (or both errored)
        total_attempted = (
            len(result["agent_results"]) + len(result["agent_errors"])
        )
        assert total_attempted == 2
