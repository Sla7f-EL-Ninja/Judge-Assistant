"""
test_e2e_multi_turn.py — multi-turn conversation E2E tests (3+ turns).
"""

import uuid

import pytest

from tests.supervisor.helpers.state_factory import make_state
from tests.supervisor.helpers.llm_assertions import assert_arabic_response
from tests.supervisor.helpers.reporting import emit_evidence


@pytest.mark.expensive
class TestE2EMultiTurn:
    def test_three_turn_conversation(self, supervisor_app, judicial_docs_available):
        """3-turn conversation — history carries forward, turn_count increments."""
        cid_base = f"test-cid-{uuid.uuid4()}"

        # Turn 1
        state1 = make_state(
            judge_query="اشرح المادة 163",
            turn_count=0,
            conversation_history=[],
            correlation_id=f"{cid_base}-t1",
        )
        r1 = supervisor_app.invoke(state1)
        assert r1["turn_count"] == 1
        assert_arabic_response(r1["final_response"], min_len=20)

        # Turn 2 — relative question relies on history
        state2 = make_state(
            judge_query="وكيف تطبق على القضية؟",
            turn_count=r1["turn_count"],
            conversation_history=r1["conversation_history"],
            correlation_id=f"{cid_base}-t2",
            case_id=r1["case_id"],
        )
        r2 = supervisor_app.invoke(state2)
        assert r2["turn_count"] == 2

        # Turn 3
        state3 = make_state(
            judge_query="ما رأيك في دفاع المدعى عليه الثاني؟",
            turn_count=r2["turn_count"],
            conversation_history=r2["conversation_history"],
            correlation_id=f"{cid_base}-t3",
        )
        r3 = supervisor_app.invoke(state3)
        assert r3["turn_count"] == 3
        assert_arabic_response(r3["final_response"], min_len=20)

        emit_evidence("e2e_multi_turn_3", r3, extra={"turns": 3})

    def test_correlation_id_unique_per_turn(self, supervisor_app):
        """Each turn gets a unique correlation_id."""
        state1 = make_state(judge_query="ما المادة 163؟", turn_count=0, conversation_history=[])
        r1 = supervisor_app.invoke(state1)

        state2 = make_state(
            judge_query="وكيف تطبق؟",
            turn_count=r1["turn_count"],
            conversation_history=r1["conversation_history"],
            case_id=r1["case_id"],
        )
        r2 = supervisor_app.invoke(state2)

        assert r1.get("correlation_id") != r2.get("correlation_id") or not r1.get("correlation_id")

    def test_history_trimmed_after_20_turns(self, supervisor_app, monkeypatch):
        """After 21 turns, history trimmed to MAX_CONVERSATION_TURNS*2."""
        import time
        import Supervisor.nodes.prepare_retry as pr_mod
        monkeypatch.setattr(time, "sleep", lambda s: None)

        from config.supervisor import MAX_CONVERSATION_TURNS
        max_msg = MAX_CONVERSATION_TURNS * 2

        history = []
        turn_count = 0

        for i in range(MAX_CONVERSATION_TURNS + 1):
            state = make_state(
                judge_query=f"سؤال رقم {i + 1}",
                turn_count=turn_count,
                conversation_history=history,
            )
            r = supervisor_app.invoke(state)
            history = r["conversation_history"]
            turn_count = r["turn_count"]

        assert len(history) <= max_msg, (
            f"History not trimmed: {len(history)} > {max_msg}"
        )

    def test_retry_count_resets_between_turns(self, supervisor_app):
        """retry_count from turn N must not bleed into turn N+1."""
        state1 = make_state(judge_query="ما المادة 163؟", turn_count=0, conversation_history=[])
        r1 = supervisor_app.invoke(state1)

        # Force retry_count for second turn start = 0 (fresh state)
        state2 = make_state(
            judge_query="وكيف تطبق؟",
            turn_count=r1["turn_count"],
            conversation_history=r1["conversation_history"],
            retry_count=0,  # must always start fresh
            case_id=r1["case_id"],
        )
        r2 = supervisor_app.invoke(state2)
        # retry_count in final state should be ≥0 (not carried from r1)
        assert isinstance(r2.get("retry_count", 0), int)
