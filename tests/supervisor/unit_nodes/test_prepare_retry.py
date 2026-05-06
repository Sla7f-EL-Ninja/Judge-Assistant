"""
test_prepare_retry.py — unit tests for prepare_retry_node.

Verifies exponential backoff timing and that no state is mutated.
Uses monkeypatching to avoid real sleeps in the unit test suite.
"""

import time

import pytest

from Supervisor.nodes.prepare_retry import prepare_retry_node, _BASE_BACKOFF_S, _MAX_BACKOFF_S
from tests.supervisor.helpers.state_factory import make_state


class TestPrepareRetryBackoff:
    def test_retry1_sleeps_2s(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        state = make_state(retry_count=1, max_retries=3, validation_status="fail_completeness")
        prepare_retry_node(state)
        assert len(slept) == 1
        assert abs(slept[0] - 2.0) < 0.1

    def test_retry2_sleeps_4s(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        state = make_state(retry_count=2, max_retries=3)
        prepare_retry_node(state)
        assert abs(slept[0] - 4.0) < 0.1

    def test_retry3_sleeps_8s(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        state = make_state(retry_count=3, max_retries=3)
        prepare_retry_node(state)
        assert abs(slept[0] - 8.0) < 0.1

    def test_retry4_capped_at_10s(self, monkeypatch):
        slept = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
        state = make_state(retry_count=4, max_retries=5)
        prepare_retry_node(state)
        assert slept[0] <= _MAX_BACKOFF_S

    def test_returns_empty_dict(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        state = make_state(retry_count=1)
        result = prepare_retry_node(state)
        assert result == {}

    def test_no_state_mutation(self, monkeypatch):
        monkeypatch.setattr(time, "sleep", lambda s: None)
        state = make_state(retry_count=1, validation_status="fail_relevance")
        original_count = state["retry_count"]
        prepare_retry_node(state)
        assert state["retry_count"] == original_count
