"""
dispatch_agents.py

Agent dispatcher node for the Supervisor workflow.

Iterates over ``target_agents``, invokes the matching adapter for each,
and stores results (or errors) back into state.
"""

import logging
from typing import Any, Dict

from Supervisor.agents.base import AgentAdapter, AgentResult
from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter
from Supervisor.agents.case_reasoner_adapter import CaseReasonerAdapter
from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
from Supervisor.agents.ocr_adapter import OCRAdapter
from Supervisor.agents.summarize_adapter import SummarizeAdapter
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

# Registry mapping canonical agent names to their adapter classes.
ADAPTER_REGISTRY: Dict[str, type] = {
    "ocr": OCRAdapter,
    "summarize": SummarizeAdapter,
    "civil_law_rag": CivilLawRAGAdapter,
    "case_doc_rag": CaseDocRAGAdapter,
    "reason": CaseReasonerAdapter,
}


def _build_context(state: SupervisorState, agent_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build the context dict passed to each adapter."""
    return {
        "uploaded_files": state.get("uploaded_files", []),
        "case_id": state.get("case_id", ""),
        "conversation_history": state.get("conversation_history", []),
        "agent_results": agent_results,
        "validation_feedback": state.get("validation_feedback", ""),
    }


def dispatch_agents_node(state: SupervisorState) -> Dict[str, Any]:
    """Invoke each target agent sequentially and collect results.

    Updates state keys: ``agent_results``, ``agent_errors``.
    """
    target_agents = state.get("target_agents", [])
    query = state.get("classified_query", state.get("judge_query", ""))

    # Append validation feedback to the query on retries
    validation_feedback = state.get("validation_feedback", "")
    retry_count = state.get("retry_count", 0)
    if retry_count > 0 and validation_feedback:
        query = f"{query}\n\n[ملاحظات التحقق السابقة: {validation_feedback}]"

    agent_results: Dict[str, Any] = {}
    agent_errors: Dict[str, str] = {}

    for agent_name in target_agents:
        adapter_cls = ADAPTER_REGISTRY.get(agent_name)
        if adapter_cls is None:
            error_msg = f"Unknown agent: {agent_name}"
            logger.warning(error_msg)
            agent_errors[agent_name] = error_msg
            continue

        logger.info("Dispatching to agent: %s", agent_name)
        try:
            adapter: AgentAdapter = adapter_cls()
            context = _build_context(state, agent_results)
            result: AgentResult = adapter.invoke(query, context)

            if result.error:
                agent_errors[agent_name] = result.error
                logger.warning(
                    "Agent %s returned error: %s", agent_name, result.error
                )
            else:
                agent_results[agent_name] = {
                    "response": result.response,
                    "sources": result.sources,
                    "raw_output": result.raw_output,
                }
                logger.info("Agent %s completed successfully", agent_name)

        except Exception as exc:
            error_msg = f"Agent {agent_name} raised exception: {exc}"
            logger.exception(error_msg)
            agent_errors[agent_name] = error_msg

    return {
        "agent_results": agent_results,
        "agent_errors": agent_errors,
    }
