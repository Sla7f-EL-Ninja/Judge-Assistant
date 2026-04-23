"""
dispatch_agents.py

Agent dispatcher node for the Supervisor workflow.

Independent agents (civil_law_rag, case_doc_rag) run in parallel via
ThreadPoolExecutor.  Dependent agents (reason) run after their tier-0
dependencies have completed so they can read prior results from context.

This replaces the original sequential for-loop (A6.5.3 / P1.2.1).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from Supervisor.agents.base import AgentAdapter, AgentResult
from Supervisor.agents.case_doc_rag_adapter import CaseDocRAGAdapter
from Supervisor.agents.case_reasoner_adapter import CaseReasonerAdapter
from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

# Registry mapping canonical agent names to their adapter classes.
ADAPTER_REGISTRY: Dict[str, type] = {
    "civil_law_rag": CivilLawRAGAdapter,
    "case_doc_rag": CaseDocRAGAdapter,
    "reason": CaseReasonerAdapter,
}

# Dependency tiers.  Agents in the same tier have no inter-dependencies and
# run in parallel.  Agents in a later tier receive the results of all earlier
# tiers via context (G5.2.3 topological ordering).
#
# Tier 0: retrieval agents — independent of each other.
# Tier 1: reasoning agent — needs civil_law_rag output in context.
_TIER_0 = frozenset({"civil_law_rag", "case_doc_rag"})
_TIER_1 = frozenset({"reason"})


def _agent_tier(agent: str) -> int:
    if agent in _TIER_0:
        return 0
    if agent in _TIER_1:
        return 1
    return 2  # unknown agents run last, sequentially


def _build_context(state: SupervisorState, agent_results: Dict[str, Any]) -> Dict[str, Any]:
    """Build the context dict passed to each adapter."""
    intent = state.get("intent", "")
    return {
        "uploaded_files": state.get("uploaded_files", []),
        "case_id": state.get("case_id", ""),
        "conversation_history": state.get("conversation_history", []),
        "agent_results": agent_results,
        "validation_feedback": state.get("validation_feedback", ""),
        # chat_reasoner needs to know why it was escalated (Part 3 gap)
        "escalation_reason": f"Supervisor routed to chat_reasoner with intent={intent}",
        # Pre-fetched by enrich_context_node (A6.5.4)
        "case_summary": state.get("case_summary", ""),
        "case_doc_titles": state.get("case_doc_titles", []),
    }


def _run_single_agent(
    agent_name: str,
    adapter_cls: type,
    query: str,
    state: SupervisorState,
    agent_results_snapshot: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Run one adapter.  Returns (agent_name, result_dict | None, error | None)."""
    try:
        adapter: AgentAdapter = adapter_cls()
        context = _build_context(state, agent_results_snapshot)
        result: AgentResult = adapter.invoke(query, context)

        if result.error:
            logger.warning("Agent %s returned error: %s", agent_name, result.error)
            return (agent_name, None, result.error)

        logger.info("Agent %s completed successfully", agent_name)
        return (
            agent_name,
            {
                "response": result.response,
                "sources": result.sources,
                "raw_output": result.raw_output,
            },
            None,
        )
    except Exception as exc:
        error_msg = f"Agent {agent_name} raised exception: {exc}"
        logger.exception(error_msg)
        return (agent_name, None, error_msg)


def _run_tier_parallel(
    agents: List[str],
    query: str,
    state: SupervisorState,
    agent_results: Dict[str, Any],
    agent_errors: Dict[str, str],
) -> None:
    """Run a list of agents in parallel via ThreadPoolExecutor.

    Results are written directly into *agent_results* and *agent_errors*
    (mutated in-place).
    """
    if not agents:
        return

    if len(agents) == 1:
        # No point spawning a thread for a single agent
        name, result_dict, error = _run_single_agent(
            agents[0], ADAPTER_REGISTRY[agents[0]], query, state, dict(agent_results)
        )
        if error:
            agent_errors[name] = error
        else:
            agent_results[name] = result_dict
        return

    # Snapshot agent_results before fan-out so all workers in this tier see
    # the same prior-tier outputs (not each other's in-flight results).
    snapshot = dict(agent_results)

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(
                _run_single_agent, name, ADAPTER_REGISTRY[name], query, state, snapshot
            ): name
            for name in agents
        }
        for future in as_completed(futures):
            name, result_dict, error = future.result()
            if error:
                agent_errors[name] = error
            else:
                agent_results[name] = result_dict


def dispatch_agents_node(state: SupervisorState) -> Dict[str, Any]:
    """Invoke target agents in dependency-tier order, parallelising within each tier.

    Updates state keys: ``agent_results``, ``agent_errors``.
    """
    raw_agents = state.get("target_agents", [])
    # Deduplicate while preserving LLM-returned order (G5.2.1)
    seen: set = set()
    target_agents: List[str] = []
    for a in raw_agents:
        if a not in seen:
            seen.add(a)
            target_agents.append(a)

    query = state.get("classified_query", state.get("judge_query", ""))

    # Append validation feedback to the query on retries
    validation_feedback = state.get("validation_feedback", "")
    retry_count = state.get("retry_count", 0)
    if retry_count > 0 and validation_feedback:
        query = f"{query}\n\n[ملاحظات التحقق السابقة: {validation_feedback}]"

    # Seed from prior state so retry preserves partial successes (P1.3.3)
    agent_results: Dict[str, Any] = dict(state.get("agent_results") or {})
    agent_errors: Dict[str, str] = dict(state.get("agent_errors") or {})

    # On retry, only re-run agents that previously failed (P1.2.2 / G5.6.2)
    if retry_count > 0:
        agents_to_run = [a for a in target_agents if a not in agent_results]
        skipped = [a for a in target_agents if a in agent_results]
        if skipped:
            logger.info("Retry %d: skipping already-succeeded agents %s", retry_count, skipped)
        if agents_to_run:
            logger.warning(
                "Retry %d: re-dispatching %d agent(s) %s (cost ~%d extra LLM calls)",
                retry_count, len(agents_to_run), agents_to_run, len(agents_to_run),
            )
    else:
        agents_to_run = target_agents

    # Filter out unknown agents up-front and log them
    unknown = [a for a in agents_to_run if ADAPTER_REGISTRY.get(a) is None]
    for a in unknown:
        error_msg = f"Unknown agent: {a}"
        logger.warning(error_msg)
        agent_errors[a] = error_msg
    agents_to_run = [a for a in agents_to_run if ADAPTER_REGISTRY.get(a) is not None]

    if not agents_to_run:
        return {"agent_results": agent_results, "agent_errors": agent_errors}

    # Build tiers and execute
    max_tier = max(_agent_tier(a) for a in agents_to_run)
    for tier_idx in range(max_tier + 1):
        tier_agents = [a for a in agents_to_run if _agent_tier(a) == tier_idx]
        if not tier_agents:
            continue
        logger.info("Dispatching tier %d agents: %s", tier_idx, tier_agents)
        _run_tier_parallel(tier_agents, query, state, agent_results, agent_errors)

    return {
        "agent_results": agent_results,
        "agent_errors": agent_errors,
    }
