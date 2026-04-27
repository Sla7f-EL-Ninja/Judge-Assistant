"""test_off_topic.py — unit tests for off_topic_response_node."""

from Supervisor.nodes.off_topic import off_topic_response_node
from Supervisor.prompts import OFF_TOPIC_RESPONSE
from tests.supervisor.helpers.state_factory import make_state


class TestOffTopicResponseNode:
    def test_returns_off_topic_response(self):
        state = make_state()
        result = off_topic_response_node(state)
        assert result["final_response"] == OFF_TOPIC_RESPONSE
        assert result["merged_response"] == OFF_TOPIC_RESPONSE

    def test_sets_validation_status_pass(self):
        state = make_state()
        result = off_topic_response_node(state)
        assert result["validation_status"] == "pass"

    def test_response_is_nonempty(self):
        state = make_state(judge_query="ما عاصمة فرنسا؟")
        result = off_topic_response_node(state)
        assert len(result["final_response"]) > 10
