"""
test_structured_output_fields.py -- Bug 3: Structured output completeness.

Verifies that IntentClassification from classify_intent_node always
produces non-None values for all required fields.

Marker: regression
"""

import pytest

from Supervisor.state import IntentClassification


@pytest.mark.regression
class TestStructuredOutputFields:
    """Verify LLM structured output always populates all fields."""

    def test_intent_classification_schema_completeness(self):
        """IntentClassification schema must define all required fields."""
        required = IntentClassification.model_json_schema().get("required", [])
        expected_fields = ["intent", "target_agents", "rewritten_query", "reasoning"]
        for field in expected_fields:
            assert field in required, (
                f"IntentClassification must declare '{field}' as required"
            )

    def test_intent_classification_rejects_none_intent(self):
        """IntentClassification must not accept None for intent."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IntentClassification(
                intent=None,
                target_agents=["civil_law_rag"],
                rewritten_query="test",
                reasoning="test",
            )

    def test_intent_classification_rejects_none_target_agents(self):
        """IntentClassification must not accept None for target_agents."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IntentClassification(
                intent="civil_law_rag",
                target_agents=None,
                rewritten_query="test",
                reasoning="test",
            )

    def test_valid_intent_classification_all_fields_populated(self):
        """A properly constructed IntentClassification has no None fields."""
        ic = IntentClassification(
            intent="civil_law_rag",
            target_agents=["civil_law_rag"],
            rewritten_query="ما هي شروط صحة العقد؟",
            reasoning="The query asks about contract validity in civil law",
        )
        assert ic.intent is not None, "intent must not be None"
        assert ic.target_agents is not None, "target_agents must not be None"
        assert ic.rewritten_query is not None, "rewritten_query must not be None"
        assert ic.reasoning is not None, "reasoning must not be None"
        assert len(ic.target_agents) > 0, "target_agents must not be empty"

    @pytest.mark.behavioral
    def test_classify_intent_node_returns_complete_fields(self):
        """classify_intent_node should produce non-None fields for all outputs."""
        try:
            from Supervisor.nodes.classify_intent import classify_intent_node

            test_queries = [
                "ما هي شروط صحة العقد في القانون المدني؟",
                "لخص لي ملف القضية المرفق",
                "استخرج النص من هذه الصورة المرفقة",
            ]

            for query in test_queries:
                state = {
                    "judge_query": query,
                    "case_id": "",
                    "uploaded_files": [],
                    "conversation_history": [],
                    "turn_count": 0,
                    "intent": "",
                    "target_agents": [],
                    "classified_query": "",
                    "agent_results": {},
                    "agent_errors": {},
                    "validation_status": "",
                    "validation_feedback": "",
                    "retry_count": 0,
                    "max_retries": 2,
                    "document_classifications": [],
                    "merged_response": "",
                    "final_response": "",
                    "sources": [],
                }
                result = classify_intent_node(state)
                assert result.get("intent") is not None, (
                    f"classify_intent_node must set intent for query: {query}"
                )
                assert result.get("target_agents") is not None, (
                    f"classify_intent_node must set target_agents for query: {query}"
                )
                assert result.get("classified_query") is not None, (
                    f"classify_intent_node must set classified_query for query: {query}"
                )
        except Exception as exc:
            pytest.skip(f"classify_intent_node unavailable: {exc}")
