"""
test_fallback.py
----------------
Unit tests for the two terminal fallback nodes.
No mocks needed — pure state mutation, no I/O.
"""

from __future__ import annotations

import pytest
from RAG.legal_rag.nodes.fallback import cannot_answer_node, off_topic_node


class TestOffTopicNode:

    def test_sets_final_answer(self):
        state = {}
        result = off_topic_node(state)
        assert result["final_answer"]
        assert isinstance(result["final_answer"], str)
        assert len(result["final_answer"]) > 0

    def test_answer_contains_arabic(self):
        import re
        state = {}
        result = off_topic_node(state)
        arabic = re.findall(r"[\u0600-\u06FF]", result["final_answer"])
        assert arabic, "Expected Arabic text in off_topic response"

    def test_does_not_erase_existing_state_keys(self):
        state = {"last_query": "test", "retry_count": 2}
        result = off_topic_node(state)
        assert result["last_query"] == "test"
        assert result["retry_count"] == 2

    def test_returns_same_state_dict(self):
        state = {}
        result = off_topic_node(state)
        assert result is state  # mutates and returns same dict


class TestCannotAnswerNode:

    def test_sets_final_answer(self):
        state = {}
        result = cannot_answer_node(state)
        assert result["final_answer"]
        assert isinstance(result["final_answer"], str)

    def test_default_reason_used_when_none(self):
        state = {}
        result = cannot_answer_node(state)
        # Should still produce a valid answer using the hardcoded default reason
        assert "تعذر" in result["final_answer"]

    def test_custom_failure_reason_included(self):
        reason = "لم يتم العثور على مواد ذات صلة بالموضوع المطلوب."
        state = {"failure_reason": reason}
        result = cannot_answer_node(state)
        assert reason in result["final_answer"]

    def test_does_not_erase_existing_state_keys(self):
        state = {"last_query": "test", "grade": "fail"}
        result = cannot_answer_node(state)
        assert result["last_query"] == "test"
        assert result["grade"] == "fail"

    def test_returns_same_state_dict(self):
        state = {"failure_reason": "some reason"}
        result = cannot_answer_node(state)
        assert result is state

    def test_answer_is_arabic(self):
        import re
        state = {}
        result = cannot_answer_node(state)
        arabic = re.findall(r"[\u0600-\u06FF]", result["final_answer"])
        assert arabic
