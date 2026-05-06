"""
test_update_memory.py — unit tests for update_memory_node.

No LLM, no DB. Pure state mutation logic.
"""

import copy

import pytest

from Supervisor.nodes.update_memory import update_memory_node
from config.supervisor import MAX_CONVERSATION_TURNS
from tests.supervisor.helpers.state_factory import make_state


class TestUpdateMemoryBasic:
    def test_appends_user_and_assistant(self):
        state = make_state(
            judge_query="ما المادة 163؟",
            final_response="المادة 163 تتعلق بالمسؤولية التقصيرية.",
            conversation_history=[],
        )
        result = update_memory_node(state)
        history = result["conversation_history"]
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "ما المادة 163؟"}
        assert history[1]["role"] == "assistant"

    def test_turn_count_increments(self):
        state = make_state(turn_count=3)
        result = update_memory_node(state)
        assert result["turn_count"] == 4

    def test_no_query_no_assistant_appended(self):
        state = make_state(judge_query="", final_response="some answer", conversation_history=[])
        result = update_memory_node(state)
        assert result["conversation_history"] == []

    def test_query_without_response_only_user_appended(self):
        state = make_state(judge_query="ما المادة 163؟", final_response="", conversation_history=[])
        result = update_memory_node(state)
        history = result["conversation_history"]
        assert len(history) == 1
        assert history[0]["role"] == "user"


class TestUpdateMemoryTrim:
    def test_trims_to_max_messages(self):
        max_msg = MAX_CONVERSATION_TURNS * 2
        existing = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(max_msg)
        ]
        state = make_state(
            judge_query="سؤال جديد",
            final_response="إجابة جديدة",
            conversation_history=existing,
        )
        result = update_memory_node(state)
        assert len(result["conversation_history"]) <= max_msg

    def test_21st_turn_trimmed(self):
        """22 turns → history trimmed to MAX_CONVERSATION_TURNS*2 messages."""
        max_msg = MAX_CONVERSATION_TURNS * 2
        existing = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(max_msg)
        ]
        state = make_state(
            judge_query="السؤال الجديد",
            final_response="الإجابة الجديدة",
            conversation_history=existing,
        )
        result = update_memory_node(state)
        assert len(result["conversation_history"]) == max_msg


class TestUpdateMemoryDeepCopy:
    def test_mutation_after_call_does_not_affect_state(self):
        state = make_state(
            judge_query="ما المادة 163؟",
            final_response="نص المادة...",
            conversation_history=[],
        )
        result = update_memory_node(state)
        history = result["conversation_history"]
        # Mutate the returned list
        history[0]["content"] = "MUTATED"
        # Re-run with same state — original should be intact
        result2 = update_memory_node(state)
        assert result2["conversation_history"][0]["content"] == "ما المادة 163؟"


class TestUpdateMemoryRoleAlternation:
    def test_role_alternation_preserved(self):
        state = make_state(
            judge_query="سؤال 1",
            final_response="إجابة 1",
            conversation_history=[],
        )
        r1 = update_memory_node(state)
        state2 = make_state(
            judge_query="سؤال 2",
            final_response="إجابة 2",
            conversation_history=r1["conversation_history"],
        )
        r2 = update_memory_node(state2)
        history = r2["conversation_history"]
        roles = [h["role"] for h in history]
        assert roles == ["user", "assistant", "user", "assistant"]
