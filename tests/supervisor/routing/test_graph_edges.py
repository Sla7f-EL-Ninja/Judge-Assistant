"""
test_graph_edges.py — pure-function tests for all conditional router branches.

No LLM calls, no DB. Calls router functions directly with hand-built state dicts.
"""

import pytest

from Supervisor.graph import (
    input_validation_router,
    intent_router,
    post_classify_store_router,
    post_dispatch_router,
    validation_router,
)
from tests.supervisor.helpers.state_factory import make_state


# ---------------------------------------------------------------------------
# input_validation_router
# ---------------------------------------------------------------------------

class TestInputValidationRouter:
    def test_classify_when_no_off_topic(self):
        state = make_state(intent="civil_law_rag")
        assert input_validation_router(state) == "classify"

    def test_classify_when_intent_empty(self):
        state = make_state(intent="")
        assert input_validation_router(state) == "classify"

    def test_off_topic_when_intent_is_off_topic(self):
        state = make_state(intent="off_topic")
        assert input_validation_router(state) == "off_topic"

    def test_classify_for_multi_intent(self):
        state = make_state(intent="multi")
        assert input_validation_router(state) == "classify"


# ---------------------------------------------------------------------------
# intent_router
# ---------------------------------------------------------------------------

class TestIntentRouter:
    def test_dispatch_civil_law_rag(self):
        state = make_state(intent="civil_law_rag", target_agents=["civil_law_rag"])
        assert intent_router(state) == "dispatch"

    def test_dispatch_case_doc_rag(self):
        state = make_state(intent="case_doc_rag", target_agents=["case_doc_rag"])
        assert intent_router(state) == "dispatch"

    def test_dispatch_reason(self):
        state = make_state(intent="reason", target_agents=["reason"])
        assert intent_router(state) == "dispatch"

    def test_dispatch_multi(self):
        state = make_state(intent="multi", target_agents=["civil_law_rag", "reason"])
        assert intent_router(state) == "dispatch"

    def test_off_topic_direct(self):
        state = make_state(intent="off_topic", target_agents=[])
        assert intent_router(state) == "off_topic"

    def test_off_topic_g551_empty_agents_on_non_off_topic(self):
        """G5.5.1: non-off_topic intent with empty target_agents → off_topic."""
        state = make_state(intent="reason", target_agents=[])
        assert intent_router(state) == "off_topic"

    def test_off_topic_g551_civil_law_empty_agents(self):
        state = make_state(intent="civil_law_rag", target_agents=[])
        assert intent_router(state) == "off_topic"

    def test_off_topic_default_intent(self):
        state = make_state()
        state["intent"] = "off_topic"
        assert intent_router(state) == "off_topic"


# ---------------------------------------------------------------------------
# post_dispatch_router
# ---------------------------------------------------------------------------

class TestPostDispatchRouter:
    def test_classify_document_when_ocr_and_files(self):
        state = make_state(
            target_agents=["ocr"],
            uploaded_files=["doc.pdf"],
        )
        assert post_dispatch_router(state) == "classify_document"

    def test_classify_document_when_case_doc_rag_and_files(self):
        state = make_state(
            target_agents=["case_doc_rag"],
            uploaded_files=["contract.txt"],
        )
        assert post_dispatch_router(state) == "classify_document"

    def test_merge_when_no_files(self):
        state = make_state(target_agents=["civil_law_rag"], uploaded_files=[])
        assert post_dispatch_router(state) == "merge"

    def test_merge_when_files_but_wrong_agent(self):
        state = make_state(target_agents=["civil_law_rag"], uploaded_files=["doc.pdf"])
        assert post_dispatch_router(state) == "merge"

    def test_merge_when_no_agents_no_files(self):
        state = make_state(target_agents=[], uploaded_files=[])
        assert post_dispatch_router(state) == "merge"


# ---------------------------------------------------------------------------
# post_classify_store_router
# ---------------------------------------------------------------------------

class TestPostClassifyStoreRouter:
    def test_merge_default(self):
        state = make_state(
            target_agents=["case_doc_rag"],
            document_classifications=[{"status": "success"}],
        )
        assert post_classify_store_router(state) == "merge"

    def test_merge_partial_success(self):
        state = make_state(
            target_agents=["ocr"],
            document_classifications=[
                {"status": "success"},
                {"status": "failed"},
            ],
        )
        assert post_classify_store_router(state) == "merge"

    def test_fallback_ocr_only_all_failed(self):
        """A6.6.3: OCR-only turn with all classifications failed → fallback."""
        state = make_state(
            target_agents=["ocr"],
            agent_results={},
            document_classifications=[
                {"status": "failed"},
                {"status": "failed"},
            ],
        )
        assert post_classify_store_router(state) == "fallback"

    def test_merge_when_no_classifications(self):
        state = make_state(target_agents=["civil_law_rag"], document_classifications=[])
        assert post_classify_store_router(state) == "merge"

    def test_merge_when_ocr_but_has_other_results(self):
        """A6.6.3 only fires for ocr-only turns with no other successful agents."""
        state = make_state(
            target_agents=["ocr", "civil_law_rag"],
            agent_results={"civil_law_rag": {"response": "ok", "sources": [], "raw_output": {}}},
            document_classifications=[{"status": "failed"}],
        )
        assert post_classify_store_router(state) == "merge"


# ---------------------------------------------------------------------------
# validation_router
# ---------------------------------------------------------------------------

class TestValidationRouter:
    def test_pass(self):
        state = make_state(validation_status="pass", retry_count=0, max_retries=3)
        assert validation_router(state) == "pass"

    def test_partial_pass(self):
        state = make_state(validation_status="partial_pass", retry_count=0, max_retries=3)
        assert validation_router(state) == "pass"

    def test_retry_fail_hallucination(self):
        state = make_state(validation_status="fail_hallucination", retry_count=0, max_retries=3)
        assert validation_router(state) == "retry"

    def test_retry_fail_relevance(self):
        state = make_state(validation_status="fail_relevance", retry_count=1, max_retries=3)
        assert validation_router(state) == "retry"

    def test_retry_fail_completeness(self):
        state = make_state(validation_status="fail_completeness", retry_count=2, max_retries=3)
        assert validation_router(state) == "retry"

    def test_fallback_exhausted(self):
        state = make_state(validation_status="fail_hallucination", retry_count=3, max_retries=3)
        assert validation_router(state) == "fallback"

    def test_fallback_missing_status(self):
        state = make_state(validation_status="", retry_count=0, max_retries=3)
        assert validation_router(state) == "fallback"

    def test_fallback_none_status(self):
        state = make_state(retry_count=0, max_retries=3)
        state["validation_status"] = None
        assert validation_router(state) == "fallback"

    def test_retry_validator_error_with_attempts(self):
        state = make_state(validation_status="validator_error", retry_count=1, max_retries=3)
        assert validation_router(state) == "retry"

    def test_fallback_validator_error_exhausted(self):
        state = make_state(validation_status="validator_error", retry_count=3, max_retries=3)
        assert validation_router(state) == "fallback"

    def test_fallback_validator_error_at_max(self):
        state = make_state(validation_status="validator_error", retry_count=3, max_retries=3)
        assert validation_router(state) == "fallback"

    def test_retry_zero_count(self):
        state = make_state(validation_status="fail_completeness", retry_count=0, max_retries=3)
        assert validation_router(state) == "retry"

    def test_fallback_exactly_at_max(self):
        state = make_state(validation_status="fail_relevance", retry_count=3, max_retries=3)
        assert validation_router(state) == "fallback"
