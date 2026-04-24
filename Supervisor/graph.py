"""
graph.py

Defines and constructs the LangGraph Supervisor workflow.

The supervisor graph orchestrates: intent classification -> agent dispatch
-> response merging -> output validation -> conversation memory update.

Conditional edges handle off-topic routing, validation retries, and
fallback after max retries.
"""

import logging
import threading

from langgraph.graph import END, START, StateGraph

from config.supervisor import MAX_RETRIES

logger = logging.getLogger(__name__)

from Supervisor.nodes.audit_log import audit_log_node
from Supervisor.nodes.classify_intent import classify_intent_node
from Supervisor.nodes.classify_and_store_document import classify_and_store_document_node
from Supervisor.nodes.dispatch_agents import dispatch_agents_node
from Supervisor.nodes.enrich_context import enrich_context_node
from Supervisor.nodes.fallback import fallback_response_node
from Supervisor.nodes.merge_responses import merge_responses_node
from Supervisor.nodes.off_topic import off_topic_response_node
from Supervisor.nodes.prepare_retry import prepare_retry_node
from Supervisor.nodes.update_memory import update_memory_node
from Supervisor.nodes.validate_input import validate_input_node
from Supervisor.nodes.validate_output import validate_output_node
from Supervisor.nodes.verify_citations import verify_citations_node
from Supervisor.state import SupervisorState


# ---------------------------------------------------------------------------
# Router functions (used by conditional edges)
# ---------------------------------------------------------------------------

def input_validation_router(state: SupervisorState) -> str:
    """Skip classifier when validate_input already set intent=off_topic."""
    if state.get("intent") == "off_topic":
        return "off_topic"
    return "classify"


def intent_router(state: SupervisorState) -> str:
    """Route after intent classification.

    Returns ``"dispatch"`` for actionable intents or ``"off_topic"``
    for queries outside the system scope.  Defence-in-depth: empty
    target_agents on a non-off_topic intent is treated as off_topic (G5.5.1).
    """
    intent = state.get("intent", "off_topic")
    if intent == "off_topic":
        return "off_topic"
    # Guard: non-off_topic with no agents is a classifier anomaly
    if not state.get("target_agents"):
        logger.warning("intent=%s but target_agents is empty — routing off_topic", intent)
        return "off_topic"
    return "dispatch"


def post_dispatch_router(state: SupervisorState) -> str:
    """Route after agent dispatch.

    Returns ``"classify_document"`` when documents should be classified and
    indexed — either because case_doc_rag ran (needs case document embedding)
    or OCR ran (needs raw text classified and stored).  A6.2.1/A6.6.1: gate
    on intent rather than blanket uploaded_files presence.
    """
    target_agents = state.get("target_agents", [])
    uploaded_files = state.get("uploaded_files", [])

    if uploaded_files and (
        "case_doc_rag" in target_agents or "ocr" in target_agents
    ):
        return "classify_document"

    return "merge"


def post_classify_store_router(state: SupervisorState) -> str:
    """Route after classify_and_store_document (A6.6.3).

    For OCR-only turns where document storage is the primary deliverable,
    route to fallback when ALL files failed classification.  In all other
    cases — including partial success or non-OCR intents — continue to merge.
    """
    target_agents = state.get("target_agents", [])
    agent_results = state.get("agent_results", {})
    classifications = state.get("document_classifications", [])

    # Only short-circuit for OCR-only turns (no other successful agents)
    ocr_only = target_agents == ["ocr"] or (
        "ocr" in target_agents and not agent_results
    )
    if ocr_only and classifications and all(
        c.get("status") == "failed" for c in classifications
    ):
        logger.warning(
            "OCR-only turn: all %d document(s) failed classification — routing to fallback",
            len(classifications),
        )
        return "fallback"

    return "merge"


def validation_router(state: SupervisorState) -> str:
    """Route after output validation.

    Returns ``"pass"`` if the response passed validation, ``"retry"``
    if retries remain, or ``"fallback"`` otherwise.

    Defaults to ``"fallback"`` (not "pass") when validation_status is
    absent, so a validator crash cannot silently deliver unreviewed content.
    """
    status = state.get("validation_status") or ""
    if status in ("pass", "partial_pass"):
        # partial_pass: hallucination+relevance+coherence OK, completeness weak
        # — response already has disclosure caveat appended (G5.7.6)
        return "pass"

    if not status:
        logger.warning("validation_status not written by validator; routing to fallback")
        return "fallback"

    # Distinct validator-error path — surface as fallback with specific status (A6.6.2)
    if status == "validator_error":
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", MAX_RETRIES)
        if retry_count < max_retries:
            return "retry"
        logger.error("Validator errored %d times — escalating to fallback", retry_count)
        return "fallback"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if retry_count < max_retries:
        return "retry"
    return "fallback"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_supervisor_graph() -> StateGraph:
    """Construct and compile the Supervisor LangGraph workflow.

    Returns the compiled graph ready for ``app.invoke(state)``.
    """
    workflow = StateGraph(SupervisorState)

    # -- Nodes --
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("enrich_context", enrich_context_node)
    workflow.add_node("dispatch_agents", dispatch_agents_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("classify_and_store_document", classify_and_store_document_node)
    workflow.add_node("merge_responses", merge_responses_node)
    workflow.add_node("verify_citations", verify_citations_node)
    workflow.add_node("validate_output", validate_output_node)
    workflow.add_node("update_memory", update_memory_node)
    workflow.add_node("audit_log", audit_log_node)
    workflow.add_node("off_topic_response", off_topic_response_node)
    workflow.add_node("fallback_response", fallback_response_node)

    # -- Edges --
    workflow.add_edge(START, "validate_input")

    workflow.add_conditional_edges(
        "validate_input",
        input_validation_router,
        {
            "classify": "classify_intent",
            "off_topic": "off_topic_response",
        },
    )

    workflow.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "dispatch": "enrich_context",
            "off_topic": "off_topic_response",
        },
    )

    workflow.add_edge("enrich_context", "dispatch_agents")

    # After dispatch, classify documents if OCR ran or files were uploaded,
    # otherwise go straight to merge.
    workflow.add_conditional_edges(
        "dispatch_agents",
        post_dispatch_router,
        {
            "classify_document": "classify_and_store_document",
            "merge": "merge_responses",
        },
    )

    workflow.add_conditional_edges(
        "classify_and_store_document",
        post_classify_store_router,
        {
            "merge": "merge_responses",
            "fallback": "fallback_response",
        },
    )
    workflow.add_edge("merge_responses", "verify_citations")
    workflow.add_edge("verify_citations", "validate_output")

    workflow.add_conditional_edges(
        "validate_output",
        validation_router,
        {
            "pass": "update_memory",
            "retry": "prepare_retry",
            "fallback": "fallback_response",
        },
    )

    # Retry cooldown: prepare_retry sleeps, then re-dispatches (A6.6.4)
    workflow.add_edge("prepare_retry", "dispatch_agents")

    workflow.add_edge("off_topic_response", "update_memory")
    workflow.add_edge("fallback_response", "update_memory")
    workflow.add_edge("update_memory", "audit_log")
    workflow.add_edge("audit_log", END)

    return workflow.compile()


_app = None
_app_lock = threading.Lock()


def get_app():
    """Return the compiled supervisor graph, building it once on first call."""
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                _app = build_supervisor_graph()
    return _app


# Backwards-compatible alias — use get_app() in new code
app = None  # populated lazily; import get_app() instead of app directly
