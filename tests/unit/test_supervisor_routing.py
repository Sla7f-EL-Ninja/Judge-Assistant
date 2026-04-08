"""
test_supervisor_routing.py -- Test supervisor graph routing functions.

Unit tests for pure-logic router functions (intent_router, post_dispatch_router,
validation_router). Behavioral tests for LLM-based classify_intent_node are in
tests/behavioral/test_routing_accuracy.py.

Marker: unit
"""

import pytest

from Supervisor.graph import intent_router, post_dispatch_router, validation_router


@pytest.mark.unit
class TestIntentRouter:
    """Test intent_router conditional edge function."""

    def test_off_topic_routes_to_off_topic(self):
        state = {"intent": "off_topic"}
        assert intent_router(state) == "off_topic", (
            "off_topic intent should route to 'off_topic'"
        )

    def test_empty_intent_defaults_to_off_topic(self):
        state = {}
        assert intent_router(state) == "off_topic", (
            "Missing intent should default to 'off_topic'"
        )

    @pytest.mark.parametrize(
        "intent",
        ["ocr", "summarize", "civil_law_rag", "case_doc_rag", "reason", "multi"],
    )
    def test_actionable_intents_route_to_dispatch(self, intent):
        state = {"intent": intent}
        assert intent_router(state) == "dispatch", (
            f"Intent '{intent}' should route to 'dispatch'"
        )


@pytest.mark.unit
class TestPostDispatchRouter:
    """Test post_dispatch_router conditional edge function."""

    def test_ocr_agent_routes_to_classify_document(self):
        state = {"target_agents": ["ocr"], "uploaded_files": []}
        assert post_dispatch_router(state) == "classify_document", (
            "OCR agent should trigger document classification"
        )

    def test_uploaded_files_without_ocr_routes_to_classify_document(self):
        state = {"target_agents": ["civil_law_rag"], "uploaded_files": ["/tmp/test.txt"]}
        assert post_dispatch_router(state) == "classify_document", (
            "Uploaded files should trigger document classification even without OCR"
        )

    def test_no_ocr_no_files_routes_to_merge(self):
        state = {"target_agents": ["civil_law_rag"], "uploaded_files": []}
        assert post_dispatch_router(state) == "merge", (
            "No OCR and no files should route to merge"
        )

    def test_empty_state_routes_to_merge(self):
        state = {}
        assert post_dispatch_router(state) == "merge", (
            "Empty state should default to merge"
        )


@pytest.mark.unit
class TestValidationRouter:
    """Test validation_router conditional edge function."""

    def test_pass_status_routes_to_pass(self):
        state = {"validation_status": "pass"}
        assert validation_router(state) == "pass", (
            "Passed validation should route to 'pass'"
        )

    def test_empty_status_defaults_to_pass(self):
        state = {}
        assert validation_router(state) == "pass", (
            "Missing validation_status should default to 'pass'"
        )

    def test_failed_with_retries_remaining_routes_to_retry(self):
        state = {
            "validation_status": "fail_hallucination",
            "retry_count": 0,
            "max_retries": 2,
        }
        assert validation_router(state) == "retry", (
            "Failed validation with retries remaining should route to 'retry'"
        )

    def test_failed_with_no_retries_routes_to_fallback(self):
        state = {
            "validation_status": "fail_relevance",
            "retry_count": 2,
            "max_retries": 2,
        }
        assert validation_router(state) == "fallback", (
            "Failed validation with no retries should route to 'fallback'"
        )

    def test_failed_at_max_retries_routes_to_fallback(self):
        state = {
            "validation_status": "fail_completeness",
            "retry_count": 3,
            "max_retries": 2,
        }
        assert validation_router(state) == "fallback", (
            "Failed validation past max retries should route to 'fallback'"
        )
