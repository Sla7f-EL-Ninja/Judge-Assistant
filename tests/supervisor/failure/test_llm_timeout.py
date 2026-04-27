"""
test_llm_timeout.py — LLM timeout behavior.

Monkeypatches _LLM_TIMEOUT_S to trigger timeout quickly.
"""

import pytest


@pytest.mark.expensive
class TestLLMTimeout:
    def test_classify_intent_llm_timeout_falls_back_to_off_topic(self, monkeypatch):
        """LLM timeout during classification → off_topic, not crash."""
        import Supervisor.nodes.classify_intent as ci_mod

        def raise_timeout(fn, msgs):
            raise TimeoutError("LLM timed out (monkeypatched)")

        monkeypatch.setattr(ci_mod, "llm_invoke", raise_timeout)

        from Supervisor.nodes.classify_intent import classify_intent_node
        from tests.supervisor.helpers.state_factory import make_state

        state = make_state(judge_query="ما المادة 163؟")
        result = classify_intent_node(state)

        assert result["intent"] == "off_topic"
        assert result.get("classification_error")

    def test_validate_output_llm_timeout_sets_validator_error(self, monkeypatch):
        """LLM timeout during validation → validator_error, not crash."""
        import Supervisor.nodes.validate_output as vo_mod

        def raise_timeout(fn, msgs):
            raise TimeoutError("validator LLM timeout")

        monkeypatch.setattr(vo_mod, "llm_invoke", raise_timeout)

        from Supervisor.nodes.validate_output import validate_output_node
        from tests.supervisor.helpers.state_factory import make_state

        state = make_state(merged_response="المادة 163 تتعلق بالمسؤولية.")
        result = validate_output_node(state)

        assert result["validation_status"] == "validator_error"
        assert result["retry_count"] == 1

    def test_merge_responses_llm_timeout_falls_back_to_concat(self, monkeypatch):
        """LLM timeout during merge → fallback concatenation, not crash."""
        import Supervisor.nodes.merge_responses as mr_mod

        def raise_timeout(fn, msgs):
            raise TimeoutError("merge LLM timeout")

        monkeypatch.setattr(mr_mod, "llm_invoke", raise_timeout)

        from Supervisor.nodes.merge_responses import merge_responses_node
        from tests.supervisor.helpers.state_factory import make_state, make_agent_result

        state = make_state(
            classified_query="اشرح المادة 163 وطبقها",
            agent_results={
                "civil_law_rag": make_agent_result(response="المادة 163..."),
                "reason": make_agent_result(response="التحليل..."),
            },
        )
        result = merge_responses_node(state)
        assert result["merged_response"]
        assert "civil_law_rag" in result["merged_response"] or "المادة" in result["merged_response"]
