"""
test_conversation_memory.py — conversation memory management tests.

Section 9 of the test plan.
"""

import copy

import pytest

from Supervisor.nodes.update_memory import update_memory_node
from config.supervisor import MAX_CONVERSATION_TURNS
from tests.supervisor.helpers.state_factory import make_state


class TestConversationMemory:
    def test_22_turns_trimmed(self):
        max_msg = MAX_CONVERSATION_TURNS * 2
        history = []
        turn_count = 0

        for i in range(MAX_CONVERSATION_TURNS + 2):
            state = make_state(
                judge_query=f"سؤال {i}",
                final_response=f"إجابة {i}",
                conversation_history=history,
                turn_count=turn_count,
            )
            result = update_memory_node(state)
            history = result["conversation_history"]
            turn_count = result["turn_count"]

        assert len(history) <= max_msg, (
            f"History {len(history)} exceeds max {max_msg}"
        )

    def test_role_alternation_preserved_over_turns(self):
        history = []
        for i in range(5):
            state = make_state(
                judge_query=f"سؤال {i}",
                final_response=f"إجابة {i}",
                conversation_history=history,
            )
            result = update_memory_node(state)
            history = result["conversation_history"]

        roles = [h["role"] for h in history]
        for i, role in enumerate(roles):
            expected = "user" if i % 2 == 0 else "assistant"
            assert role == expected, f"Role mismatch at index {i}: {role!r} != {expected!r}"

    def test_deepcopy_no_aliasing(self):
        state = make_state(
            judge_query="سؤال أول",
            final_response="إجابة أولى",
            conversation_history=[],
        )
        result = update_memory_node(state)
        history = result["conversation_history"]
        # Mutate returned history
        history[0]["content"] = "MUTATED"
        # Re-run same state — should be unaffected
        result2 = update_memory_node(state)
        assert result2["conversation_history"][0]["content"] == "سؤال أول"

    def test_turn_count_increments_each_call(self):
        history = []
        turn_count = 0
        for i in range(3):
            state = make_state(
                judge_query=f"سؤال {i}",
                final_response=f"إجابة {i}",
                conversation_history=history,
                turn_count=turn_count,
            )
            result = update_memory_node(state)
            history = result["conversation_history"]
            turn_count = result["turn_count"]
            assert turn_count == i + 1

    def test_orphan_assistant_without_query_skipped(self):
        state = make_state(
            judge_query="",
            final_response="إجابة بدون سؤال",
            conversation_history=[],
        )
        result = update_memory_node(state)
        # No user turn → no assistant turn appended
        assert result["conversation_history"] == []

    def test_user_query_without_response_appended(self):
        state = make_state(
            judge_query="سؤال بدون إجابة",
            final_response="",
            conversation_history=[],
        )
        result = update_memory_node(state)
        assert len(result["conversation_history"]) == 1
        assert result["conversation_history"][0]["role"] == "user"


@pytest.mark.expensive
class TestConversationMemoryE2E:
    def test_second_turn_uses_first_turn_history(self, supervisor_app):
        """Turn 2 relative query should be resolved using Turn 1 history."""
        state1 = make_state(
            judge_query="اشرح المادة 163",
            turn_count=0,
            conversation_history=[],
        )
        r1 = supervisor_app.invoke(state1)
        assert r1["turn_count"] == 1
        assert len(r1["conversation_history"]) >= 2

        state2 = make_state(
            judge_query="وهل تنطبق على القضية الحالية؟",
            turn_count=r1["turn_count"],
            conversation_history=r1["conversation_history"],
        )
        r2 = supervisor_app.invoke(state2)
        assert r2["turn_count"] == 2
        assert r2["final_response"]
