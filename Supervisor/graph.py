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
from Supervisor.nodes.classify_intent import classify_intent_node
from Supervisor.nodes.classify_and_store_document import classify_and_store_document_node
from Supervisor.nodes.dispatch_agents import dispatch_agents_node
from Supervisor.nodes.fallback import fallback_response_node
from Supervisor.nodes.merge_responses import merge_responses_node
from Supervisor.nodes.off_topic import off_topic_response_node
from Supervisor.nodes.update_memory import update_memory_node
from Supervisor.nodes.validate_output import validate_output_node
from Supervisor.state import SupervisorState


# ---------------------------------------------------------------------------
# Router functions (used by conditional edges)
# ---------------------------------------------------------------------------

def intent_router(state: SupervisorState) -> str:
    """Route after intent classification.

    Returns ``"dispatch"`` for actionable intents or ``"off_topic"``
    for queries outside the system scope.
    """
    intent = state.get("intent", "off_topic")
    if intent == "off_topic":
        return "off_topic"
    return "dispatch"


def post_dispatch_router(state: SupervisorState) -> str:
    """Route after agent dispatch.

    Returns ``"classify_document"`` only when case documents should be
    indexed (case_doc_rag intent with uploaded files), or ``"merge"``
    otherwise.  Previously this routed on any uploaded file regardless
    of intent, causing unintended indexing on civil_law_rag queries.
    """
    target_agents = state.get("target_agents", [])
    uploaded_files = state.get("uploaded_files", [])

    if uploaded_files and "case_doc_rag" in target_agents:
        return "classify_document"

    return "merge"


def validation_router(state: SupervisorState) -> str:
    """Route after output validation.

    Returns ``"pass"`` if the response passed validation, ``"retry"``
    if retries remain, or ``"fallback"`` otherwise.

    Defaults to ``"fallback"`` (not "pass") when validation_status is
    absent, so a validator crash cannot silently deliver unreviewed content.
    """
    status = state.get("validation_status") or ""
    if status == "pass":
        return "pass"

    if not status:
        logger.warning("validation_status not written by validator; routing to fallback")
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
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("dispatch_agents", dispatch_agents_node)
    workflow.add_node("classify_and_store_document", classify_and_store_document_node)
    workflow.add_node("merge_responses", merge_responses_node)
    workflow.add_node("validate_output", validate_output_node)
    workflow.add_node("update_memory", update_memory_node)
    workflow.add_node("off_topic_response", off_topic_response_node)
    workflow.add_node("fallback_response", fallback_response_node)

    # -- Edges --
    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        intent_router,
        {
            "dispatch": "dispatch_agents",
            "off_topic": "off_topic_response",
        },
    )

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

    workflow.add_edge("classify_and_store_document", "merge_responses")
    workflow.add_edge("merge_responses", "validate_output")

    workflow.add_conditional_edges(
        "validate_output",
        validation_router,
        {
            "pass": "update_memory",
            "retry": "dispatch_agents",
            "fallback": "fallback_response",
        },
    )

    workflow.add_edge("off_topic_response", "update_memory")
    workflow.add_edge("fallback_response", "update_memory")
    workflow.add_edge("update_memory", END)

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
