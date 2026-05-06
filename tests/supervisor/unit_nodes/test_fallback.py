"""test_fallback.py — unit tests for fallback_response_node."""

from Supervisor.nodes.fallback import fallback_response_node
from tests.supervisor.helpers.state_factory import make_state


class TestFallbackResponseNode:
    def test_returns_fallback_response(self):
        state = make_state(validation_feedback="الإجابة لم تجتز التحقق.")
        result = fallback_response_node(state)
        assert result["final_response"]
        assert "validation_status" in result
        assert result["validation_status"] == "fallback"

    def test_feedback_included_in_response(self):
        feedback = "فشل التحقق بسبب الهلوسة"
        state = make_state(validation_feedback=feedback)
        result = fallback_response_node(state)
        assert feedback in result["final_response"]

    def test_empty_feedback_handled(self):
        state = make_state(validation_feedback="")
        result = fallback_response_node(state)
        assert result["final_response"]  # non-empty even with no feedback
        assert result["validation_status"] == "fallback"
